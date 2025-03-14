import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from functools import lru_cache
import copy
from pathlib import Path
from dataclasses import dataclass
import time

# Check device availability
mps_available = torch.backends.mps.is_available()
cuda_available = torch.cuda.is_available()

# Define device globally to ensure consistency
device = None

if cuda_available:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda")
elif mps_available:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Prevent a bug on some systems (only needed for initialization)
dummy_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.empty(1, device=dummy_device, requires_grad=True).backward()

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    FLEX_ATTENTION_SUPPORTED = True
except ImportError:
    print("Warning: FlexAttention not supported, using standard attention.")
    FLEX_ATTENTION_SUPPORTED = False


# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

# Custom matrix multiplication operator using FP8 (Float 8-bit precision)

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

# Fallback to a standard PyTorch matrix multiplication when the actual FP8 operation is not available
@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    """
    Unified norm function that handles both CUDA and MPS
    """
    # For MPS, ensure we're using float32 for better numerical stability
    if device.type == "mps":
        x_float32 = x.to(torch.float32)
        return F.layer_norm(x_float32, (x.shape[-1],), weight=None, bias=None)
    else:
        # For CUDA, use rms_norm with original dtype
        return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8 and device.type == "cuda"  # FP8 only on CUDA, not MPS
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            # Ensure MPS compatibility by keeping float32 precision
            if device.type == "mps":
                x = x.to(torch.float32)
            return F.linear(x, self.weight)

# Rotary Positional Embeddings (RoPE), a technique used in transformers to introduce 
# position information into attention mechanisms without requiring explicit positional embeddings
class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # Half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, torch.zeros(dim // 4, dtype=torch.float32)])

        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)  # More efficient than einsum

        # Store cos and sin as buffers (avoid being treated as parameters)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3), "Sequence length exceeds RoPE limit"
        
        # Ensure float32 for MPS
        original_dtype = x_BTHD.dtype
        x_BTHD = x_BTHD.to(torch.float32) if device.type == "mps" else x_BTHD

        # Extract cos and sin for the required sequence length
        cos, sin = self.cos[:x_BTHD.size(-3)].unsqueeze(0).unsqueeze(2), \
                   self.sin[:x_BTHD.size(-3)].unsqueeze(0).unsqueeze(2)

        # Split tensor into two parts for RoPE transformation
        x1, x2 = x_BTHD.chunk(2, dim=-1)

        # Apply rotation
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos

        # Return with original dtype if not MPS
        result = torch.cat((y1, y2), dim=-1)
        return result if device.type == "mps" else result.to(original_dtype)
    

# CausalSelfAttention class implements a multi-head self-attention mechanism using a causal mask
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  
        
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        
        # Only assert batch size on CUDA as it's required for FlexAttention
        if device.type == "cuda":
            assert B == 1, "Batch size must be 1 for FlexAttention (CUDA)."

        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)) \
            .view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)

        # Fix for MPS: Check if ve is not None and has the right shape before using it
        if ve is not None:
            try:
                # Try to reshape ve to match v
                ve_shaped = ve.view_as(v)
                v = self.lambdas[0] * v + self.lambdas[1] * ve_shaped
            except (RuntimeError, ValueError):
                # If reshaping fails, just use v with lambda[0] scaling
                print(f"Warning: Value embedding shape mismatch. Using only primary values.")
                v = self.lambdas[0] * v
        else:
            v = self.lambdas[0] * v

        # Use flex_attention on CUDA, standard attention on MPS
        if device.type == "cuda" and FLEX_ATTENTION_SUPPORTED:
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                            block_mask=block_mask, scale=0.12).transpose(1, 2)
        else:
            # Correctly transpose dimensions for standard attention
            q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = self.standard_attention(q, k, v)
            y = y.transpose(1, 2)  # [B, T, num_heads, head_dim]

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

    def standard_attention(self, q, k, v):
        """
        Standard Scaled Dot-Product Attention (for MPS)
        """
        scale_factor = (self.head_dim ** -0.5)
        
        # Compute attention scores and ensure correct precision
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        
        # Ensure float32 precision on MPS
        if device.type == "mps":
            attn_scores = attn_scores.to(torch.float32)
            
        # Apply causal mask (upper triangular)
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn_probs, v)
        return y
    

# Feedforward Network in Transformer
class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim  # Expansion factor of 4
        
        # Use standard Linear for both MPS and CPU for better compatibility
        if device.type in ["mps", "cpu"]:
            self.c_fc = nn.Linear(dim, hdim)
            self.c_proj = nn.Linear(hdim, dim)
        else:
            # Use CastedLinear for CUDA
            self.c_fc = CastedLinear(dim, hdim)
            self.c_proj = CastedLinear(hdim, dim)
            self.c_proj.weight.detach().zero_()  # Zero init (CUDA only)

    def forward(self, x: Tensor):
        if device.type in ["mps", "cpu"]:
            # Ensure float32 for MPS and handle with activation
            x = self.c_fc(x.to(torch.float32) if device.type == "mps" else x)
            x = F.gelu(x)  # More stable on MPS/CPU
        else:
            # CUDA path
            x = self.c_fc(x)
            x = F.relu(x).square()  # Faster on CUDA
            
        x = self.c_proj(x)
        return x

# Transformer Block
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = (
            CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        )
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)]) # Transformer Blocks (Layers)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        # Skip creating BlockMasks on MPS since they're only used for flex_attention on CUDA
        if device.type == "mps":
            # Return dummy block masks for MPS that will be ignored
            return None, None
            
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, sliding_window_num_blocks: Tensor = None):
        # Handle single token or sequence input
        is_single_token = input_seq.ndim == 0
        if is_single_token:
            input_seq = input_seq.unsqueeze(0)
        
        # Check if input is batched or not
        is_batched = input_seq.ndim == 2
        if not is_batched:
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension
        
        # Set default sliding window blocks if not provided
        if sliding_window_num_blocks is None:
            sliding_window_num_blocks = torch.tensor(max(1, input_seq.size(-1) // 128), 
                                                device=input_seq.device, dtype=torch.int32)
        
        # Handle value embeddings differently for MPS vs CUDA
        if device.type == "mps":
            # For MPS, it's safer to use None for value embeddings to avoid reshape errors
            ve = [None] * len(self.blocks)
        else:
            # Original code for CUDA
            ve = [value_embed(input_seq.squeeze(0)) for value_embed in self.value_embeds]
            # 012 ... 012 structure on token value embeddings
            ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
            assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq.view(-1), sliding_window_num_blocks)
        
        # Handle both CUDA with block masks and MPS without them
        if device.type == "cuda":
            block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, 
                        short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            # Make sure we have enough block masks for all blocks
            if len(block_masks) < len(self.blocks):
                block_masks.extend([long_bm] * (len(self.blocks) - len(block_masks)))
        else:
            # On MPS, block masks aren't used, so we'll just provide None
            block_masks = [None] * len(self.blocks)
            
        assert len(block_masks) == len(self.blocks)

        # Get initial embeddings
        x = x0 = norm(self.embed(input_seq)) 

        # U-net design
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # Sigmoid-based softcapping
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        
        # Return logits directly instead of computing loss
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text from a prompt
        
        Args:
            prompt_tokens: List or tensor of token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from the top k most probable tokens
            top_p: If set, sample from top tokens with cumulative probability >= top_p
            
        Returns:
            Generated token IDs (including prompt)
        """
        # Convert to tensor if needed
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int32, device=device)
        
        # Initialize with prompt
        tokens = prompt_tokens.clone()
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Get only the last chunk that fits in the context window
            input_tokens = tokens[-self.max_seq_len:]
            
            # Get sliding window blocks
            sliding_window_blocks = torch.tensor(max(1, len(input_tokens) // 128), 
                                            device=device, dtype=torch.int32)
            
            # Forward pass to get logits
            logits = self.forward(input_tokens, sliding_window_blocks)
            
            # Get logits for the last token
            logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
                
            # Apply top-p filtering (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[0]
            
            # Add the token to the sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=0)
            
        return tokens


# -----------------------------------------------------------------------------
# Utility functions for inference

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    tensor = torch.tensor(window_size // 128, dtype=torch.int32)
    
    if device.type == "cuda":
        return tensor.pin_memory().cuda(non_blocking=True)
    return tensor.to(device)

def get_window_size_blocks(step: int, max_steps=1000):
    """
    For inference, we typically want the maximum window size
    """
    x = 1.0  # Use maximum window size for inference
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

@dataclass
class Hyperparameters:
    # Model architecture defaults
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 4
    model_dim: int = 256
    max_seq_len: int = 512
    # Implementation details
    use_flex_attention: bool = True

# Function to initialize model and convert embeddings to bfloat16 if on CUDA
def init_model(config: Hyperparameters = None):
    if config is None:
        config = Hyperparameters()
    
    model = GPT(vocab_size=config.vocab_size, 
                num_layers=config.num_layers,
                num_heads=config.num_heads, 
                model_dim=config.model_dim,
                max_seq_len=config.max_seq_len).to(device)
    
    # Convert embeddings to bfloat16 on CUDA
    if device.type == "cuda":
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()
    
    return model