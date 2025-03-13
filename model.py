# model.py
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# Define device globally to ensure consistency
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Helper functions
def norm(x: Tensor):
    """Unified norm function that handles both CUDA and MPS"""
    target_dtype = torch.float32 if device.type == "mps" else x.dtype
    return F.rms_norm(x.to(target_dtype), (x.size(-1),))

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8 and device.type == "cuda"  # FP8 only on CUDA, not MPS
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        # Ensure MPS compatibility by keeping float32 precision
        if device.type == "mps":
            x = x.to(torch.float32)
        return F.linear(x, self.weight)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # Use 128 as the max_seq_len to match the checkpoint size
        actual_seq_len = 128  # Hardcoded to match the checkpoint
        
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, torch.zeros(dim // 4, dtype=torch.float32)])

        t = torch.arange(actual_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)  # More efficient than einsum

        # Store cos and sin as buffers (avoid being treated as parameters)
        self.register_buffer("cos", theta.cos(), persistent=True)
        self.register_buffer("sin", theta.sin(), persistent=True)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3), "Sequence length exceeds RoPE limit"
        
        # Ensure float32 for MPS
        original_dtype = x_BTHD.dtype
        x_BTHD = x_BTHD.to(torch.float32) if device.type == "mps" else x_BTHD

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

    def forward(self, x: Tensor, ve: Tensor | None, block_mask=None):
        B, T = x.size(0), x.size(1)
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)) \
            .view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = self.standard_attention(q, k, v)
        y = y.transpose(1, 2)  # [B, T, num_heads, head_dim]
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

    def standard_attention(self, q, k, v):
        """Standard Scaled Dot-Product Attention (for MPS)"""
        scale_factor = (self.head_dim ** -0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if device.type == "mps":
            attn_scores = attn_scores.to(torch.float32)
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn_probs, v)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim  # Expansion factor of 4
        # No bias for both Linear layers
        self.c_fc = nn.Linear(dim, hdim, bias=False) if device.type == "mps" else CastedLinear(dim, hdim)
        self.c_proj = nn.Linear(hdim, dim, bias=False) if device.type == "mps" else CastedLinear(hdim, dim)
        if device.type != "mps":
            self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        if device.type == "mps":
            x = F.gelu(x)  # More stable on Mac
        else:
            x = F.relu(x).square()  # Faster on CUDA
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                   use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def forward(self, input_seq: Tensor, eval_mode=False, sliding_window_num_blocks=None, target_seq=None):
        """Simplified forward pass for evaluation"""
        if eval_mode:
            # Simplified forward for evaluation (no token values or block masks)
            x = x0 = norm(self.embed(input_seq)[None])
            # Generate dummy value embeddings
            ve = [None] * len(self.blocks)
            
            # Handle block masks for MPS
            block_masks = [None] * len(self.blocks)
            
            # U-net with skip connections
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
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
            return logits
        else:
            # This is for compatibility with your original model forward function
            assert input_seq.ndim == 1
            assert target_seq is not None
            
            # Generate value embeddings
            ve = [value_embed(input_seq) for value_embed in self.value_embeds]
            ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
            
            # On MPS, block masks aren't used
            block_masks = [None] * len(self.blocks)
            
            x = x0 = norm(self.embed(input_seq)[None])
            
            # U-net with skip connections
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
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='mean')
            return loss

class Hyperparameters:
    # Match your training hyperparameters
    vocab_size = 50257
    num_layers = 4
    num_heads = 4
    model_dim = 256
    train_seq_len = 512
    val_seq_len = 512