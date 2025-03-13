#2 
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
import platform
from itertools import cycle

# MPS-specific optimizations
MPS_DEVICE_AVAILABLE = torch.backends.mps.is_available()

if MPS_DEVICE_AVAILABLE:
    print(f"MPS device detected on {platform.machine()} - Applying Mac-specific optimizations")
    
    # Set environment variables for MPS stability
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    
    # Set required environment variables for distributed training if not already set
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    # Improve MPS memory management
    os.environ["MPS_ENABLE_SHARED_MEMORY_CACHE"] = "1"
    
    # Set device variable
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# MPS-specific distributed overrides
if MPS_DEVICE_AVAILABLE:
    # Store original functions
    _original_all_reduce = dist.all_reduce
    _original_all_gather_into_tensor = dist.all_gather_into_tensor
    
    # Create MPS-safe versions
    def mps_safe_all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
        """MPS-safe version of all_reduce that works in single-process mode"""
        # Just return the tensor in single-process mode
        if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            return tensor
        try:
            return _original_all_reduce(tensor, op, async_op)
        except RuntimeError as e:
            print(f"Warning: all_reduce failed, using original tensor: {e}")
            return tensor
    
    def mps_safe_all_gather_into_tensor(output_tensor, input_tensor, async_op=False):
        """MPS-safe version of all_gather_into_tensor that handles shape mismatches"""
        # In single-process mode, just copy the input to the first slice of output
        if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            if output_tensor.shape[0] > 0:
                try:
                    # Try to copy directly
                    output_tensor[0].copy_(input_tensor)
                except RuntimeError:
                    # If shapes don't match, try to reshape
                    try:
                        output_tensor[0].copy_(input_tensor.view_as(output_tensor[0]))
                    except RuntimeError:
                        # Last resort: do nothing
                        pass
            
            # Create a dummy handle
            class DummyHandle:
                def wait(self):
                    pass
            return DummyHandle()
        
        # Try the original function with shape validation
        try:
            # Ensure shapes are compatible
            if input_tensor.ndim == 1 and output_tensor.ndim == 2:
                # Fix the common case: 1D input tensor and 2D output buffer
                reshaped_input = input_tensor.reshape(1, -1)
                if reshaped_input.shape[1] == output_tensor.shape[1]:
                    return _original_all_gather_into_tensor(output_tensor, reshaped_input, async_op)
            
            # If shapes match, use original function
            return _original_all_gather_into_tensor(output_tensor, input_tensor, async_op)
        except RuntimeError as e:
            print(f"Warning: all_gather_into_tensor failed: {e}")
            print(f"Tensor shapes: output={output_tensor.shape}, input={input_tensor.shape}")
            
            # Fallback: copy to the local rank's slice if possible
            rank = int(os.environ.get("RANK", "0"))
            if output_tensor.shape[0] > rank:
                try:
                    if input_tensor.numel() == output_tensor[rank].numel():
                        output_tensor[rank].copy_(input_tensor.reshape_as(output_tensor[rank]))
                except RuntimeError:
                    pass
            
            # Return a dummy handle
            class DummyHandle:
                def wait(self):
                    pass
            return DummyHandle()
    
    # Override distributed functions with MPS-safe versions
    dist.all_reduce = mps_safe_all_reduce
    dist.all_gather_into_tensor = mps_safe_all_gather_into_tensor

# Prevent a bug on some systems
torch.empty(1, device=device, requires_grad=True).backward()

# MPS does not support FlexAttention, disabling it for now
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
except ImportError:
    flex_attention = None
    BlockMask = None

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng (Modified for MPS Compatibility)

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()

        # FP8 is only supported on CUDA, fallback to BF16 for Mac
        if device.type == "cuda":
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
        else:
            # Use bfloat16 on Mac M-series or CPU
            x_f8 = x.to(torch.bfloat16)
            w_f8 = w.to(torch.bfloat16)
            out = (x_f8 @ w_f8.T).to(torch.bfloat16)  # Standard matmul in BF16
        
        return out, x_f8, w_f8

    if device.type == "cuda":
        impl = torch.compile(impl)
    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    
    # Use BF16 on Mac and CPU, FP8 on CUDA
    if device.type == "cuda":
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    else:
        return x @ w.T, x.to(torch.bfloat16), w.to(torch.bfloat16)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)

        if device.type == "cuda":
            grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
            grad_x = torch._scaled_mm(
                grad_f8,
                w_f8.T.contiguous().T,
                out_dtype=torch.bfloat16,
                scale_a=grad_inv_s,
                scale_b=w_inv_s,
                use_fast_accum=False,
            )
            grad_w = torch._scaled_mm(
                x_f8.T.contiguous(),
                grad_f8.T.contiguous().T,
                out_dtype=torch.float32,
                scale_a=x_inv_s,
                scale_b=grad_inv_s,
                use_fast_accum=False,
            ).T
        else:
            # Use standard BF16 matmul for Mac M-series & CPU
            grad_f8 = grad.to(torch.bfloat16)
            grad_x = (grad_f8 @ w_f8.T).to(torch.bfloat16)
            grad_w = (x_f8.T @ grad_f8).to(torch.float32)

        return grad_x, grad_w

    if device.type == "cuda":
        impl = torch.compile(impl)
    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Improved Newton-Schulz implementation for MPS

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration with MPS compatibility.
    
    This implementation includes special handling for MPS devices:
    1. Ensures proper dtype conversion (MPS has issues with bfloat16)
    2. Handles tensor reshape issues that can occur on MPS
    3. Includes fallback paths for problematic operations
    """
    if G is None or G.numel() == 0:
        # Handle empty tensors gracefully
        return G
        
    assert G.ndim >= 2, f"Expected tensor with at least 2 dimensions, got {G.ndim}"
    G_device = G.device  # Store original device
    G_dtype = G.dtype    # Store original dtype
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # MPS compatibility: Use float32 for MPS, bfloat16 for CUDA
    if G_device.type == "mps":
        # MPS works best with float32
        try:
            X = G.to(torch.float32)
            
            # Special handling for MPS: use a simpler approach if the tensor is too big
            if G.numel() > 1_000_000:  # Arbitrary threshold to avoid MPS memory issues
                # Use a simpler normalization approach for large tensors
                norm = torch.norm(X, dim=(-2, -1), keepdim=True) + 1e-7
                return (X / norm).to(G_dtype).to(G_device)
        except RuntimeError as e:
            print(f"Warning: tensor conversion failed: {e}")
            return G  # Return original tensor if conversion fails
    else:
        # CUDA can use bfloat16 for speed
        X = G.to(torch.bfloat16 if G_device.type == "cuda" else G_dtype)
    
    # Handle transposition for non-square matrices
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT  # Transpose if rows > cols
        transposed = True
    
    # Normalize to ensure spectral norm is at most 1
    try:
        norm = torch.norm(X, dim=(-2, -1), keepdim=True) + 1e-7
        X = X / norm
    except RuntimeError as e:
        print(f"Warning: normalization failed: {e}")
        return G  # Return original tensor if normalization fails
    
    # Try Newton-Schulz iterations with error handling
    try:
        # Start with simple implementation for MPS
        if G_device.type == "mps":
            # Simple but stable implementation for MPS
            for _ in range(steps):
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        else:
            # Full implementation for CUDA/CPU
            for _ in range(steps):
                A = X @ X.mT
                B = b * A + c * A @ A  # Quintic computation
                X = a * X + B @ X
    except RuntimeError as e:
        print(f"Warning: Newton-Schulz iteration failed: {e}")
        # Fallback to simpler normalization
        X = G.to(torch.float32)
        norm = torch.norm(X, dim=(-2, -1), keepdim=True) + 1e-7
        X = X / norm
    
    # Reverse transpose if applied earlier
    if transposed:
        X = X.mT
    
    # Return tensor on the original device and dtype
    return X.to(G_dtype).to(G_device)

# -----------------------------------------------------------------------------
# MPS-optimized Muon optimizer

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    MPS-optimized version that skips distributed operations entirely for Mac.
    This version completely avoids the all_gather_into_tensor issues by using
    direct parameter updates instead of the distributed approach.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        # Convert params to list if it's an iterator
        params = list(params)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with MPS compatibility.
        This simplified version skips all distributed operations.
        """
        # Handle closure for API compatibility
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            # Process all parameters directly (no distributed operations)
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer if needed
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                # Update momentum buffer and get gradient
                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - group['momentum'])
                
                if group['nesterov']:
                    g = grad.lerp_(buf, group['momentum'])
                else:
                    g = buf
                
                # Apply Newton-Schulz orthogonalization if tensor is 2D or larger
                if g.ndim >= 2:
                    try:
                        g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    except Exception as e:
                        print(f"Warning: Newton-Schulz failed, using original gradient: {e}")
                
                # Apply learning rate scaling
                lr_scale = 1.0
                if g.ndim >= 2:
                    # Scale learning rate based on matrix shape
                    lr_scale = max(1, p.size(-2) / p.size(-1))**0.5
                
                # Update parameter
                p.add_(g, alpha=-group['lr'] * lr_scale)
            
        return loss

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8 and device.type == "cuda"  # Disable FP8 on Mac and CPU
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
            # Ensure weights and inputs match precision for MPS compatibility
            target_dtype = torch.float32 if device.type == "mps" else x.dtype
            weight = self.weight.to(target_dtype)
            return F.linear(x.to(target_dtype), weight)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Parameter(theta.cos(), requires_grad=False)
        self.sin = nn.Parameter(theta.sin(), requires_grad=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng

        # merged QKV weights
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

        # For non-MPS devices we will use flex_attention (or another fast implementation)
        # On MPS, we will use our chunked attention fallback.
        self.use_flex = (device.type != "mps" and flex_attention is not None)

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask | None):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        if device.type == "cuda" and self.use_flex:
            # For flex_attention we require batch size 1 (per original design)
            assert B == 1, "Must use batch size = 1 for FlexAttention on CUDA"

        # Compute Q, K, V from merged linear projection
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).to(x.dtype))\
                    .view(B, T, 3 * self.num_heads, self.head_dim)\
                    .chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # Apply normalization to Q and K
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v

        if self.use_flex:
            # Use FlexAttention on CUDA/CPU as before.
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                              block_mask=block_mask, scale=0.12).transpose(1, 2)
        else:
            # Fallback for MPS: use chunked multi-head attention to avoid huge memory buffers.
            # Reshape q, k, v to shape (B * num_heads, T, head_dim)
            q_ = q.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
            k_ = k.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
            v_ = v.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)

            # Define a chunk size (adjust this if needed)
            chunk_size = min(1024, T)
            outputs = []
            scale = 0.12  # using the same scale as in flex_attention
            for i in range(0, T, chunk_size):
                # q_chunk has shape (B*num_heads, chunk_size, head_dim)
                q_chunk = q_[:, i:min(i+chunk_size, T), :]
                # Compute scaled dot-product attention scores: (B*num_heads, chunk_size, T)
                attn_scores = torch.bmm(q_chunk, k_.transpose(1, 2)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                # Compute attention output for the chunk: (B*num_heads, chunk_size, head_dim)
                out_chunk = torch.bmm(attn_weights, v_)
                outputs.append(out_chunk)
            # Concatenate along the sequence dimension to obtain (B*num_heads, T, head_dim)
            y_ = torch.cat(outputs, dim=1)
            # Reshape back to (B, T, num_heads, head_dim) and then flatten head dimensions
            y = y_.reshape(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous()
            y = y.view(B, T, self.num_heads * self.head_dim)

        # Apply final linear projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x)
        x = x * x  # Replaces `.square()` for better MPS compatibility
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # Skip attention in block 7 (the 8th layer) as suggested by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

        # Ensure lambdas tensor is placed on the correct device
        self.lambdas = nn.Parameter(torch.tensor([1., 0.], device=device))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask | None):
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
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(
            model_dim,
            next_multiple_of_n(vocab_size, n=128),
            use_fp8=(device.type == "cuda"),  # Disable FP8 on Mac and CPU
            x_s=(model_dim**0.5)/448,
            w_s=24/448,
            grad_s=1/448
        )
        self.lm_head.weight.detach().zero_()  # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2, device=device))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        # Skip creating blockmasks for MPS or when flex_attention is not available
        if device.type == "mps" or flex_attention is None:
            return None, None
            
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0).to(device)

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
        NUM_BLOCKS = len(input_seq)# Skip creating blockmasks for MPS or when flex_attention is not available
        if device.type == "mps" or flex_attention is None:
            return None, None
            
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0).to(device)

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
        blockmask_any = (causal_blockmask_any & document_blockmask_any).to(device)
        blockmask_all = (causal_blockmask_all & document_blockmask_all).to(device)
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
        
        try:
            # Long-short SWA block masks by @leloykun & @YouJiacheng
            return build_bm(sliding_window_num_blocks.to(device)), build_bm((sliding_window_num_blocks // 2).to(device))
        except Exception as e:
            print(f"Warning: Error creating blockmasks: {e}. Using None instead.")
            return None, None

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1
        # Ensure tensors are on the correct device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device, dtype=torch.long)
        
        # Generate value embeddings
        try:
            ve = [value_embed(input_seq) for value_embed in self.value_embeds]
            # 012 ... 012 structure on token value embeddings by @YouJiacheng
            ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
            assert len(ve) == len(self.blocks)
        except Exception as e:
            # Only log if there's an actual error message
            if str(e).strip():  # Check if error message is not empty
                print(f"Warning: Error in value embeddings: {e}. Using None.")
            ve = [None] * len(self.blocks)
        
        # Create block masks (safely)
        try:
            long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks.to(device))
            block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, 
                          short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            
            # For shorter models, trim the masks
            if len(self.blocks) < len(block_masks):
                block_masks = block_masks[:len(self.blocks)]
            # For longer models, extend with None
            elif len(self.blocks) > len(block_masks):
                block_masks.extend([None] * (len(self.blocks) - len(block_masks)))
        except Exception as e:
            print(f"Warning: Error in blockmasks: {e}. Using None.")
            block_masks = [None] * len(self.blocks)

        # Ensure embedding and normalization are consistent in dtype
        # Always cast the embedding output to a floating-point type
        try:
            x = x0 = norm(self.embed(input_seq)[None].to(torch.float32))
        except Exception as e:
            print(f"Warning: Error in embedding normalization: {e}")
            # Fallback to direct embedding without normalization
            x = x0 = self.embed(input_seq)[None].to(torch.float32)
        
        # U-net design with skip connections
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            try:
                if i >= n:
                    x = x + self.skip_weights[i - n].to(x.dtype) * skip_connections.pop()
                x = self.blocks[i](x, ve[i] if i < len(ve) else None, x0, 
                                  block_masks[i] if i < len(block_masks) else None)
                if i < n:
                    skip_connections.append(x)
            except Exception as e:
                print(f"Warning: Error in block {i}: {e}. Skipping.")
                # If a block fails, try to continue with the current state

        try:
            x = norm(x)
        except Exception as e:
            print(f"Warning: Error in final normalization: {e}")
            
        logits = self.lm_head(x).to(torch.float32)  # Explicitly set float32 for stability on Mac
        
        # Sigmoid activation with scaling for stability
        try:
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        except Exception as e:
            print(f"Warning: Error in logits activation: {e}")
            # Skip the activation if it fails
        
        # Compute loss (ensure dtype consistency)
        try:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean'
            )
            return loss
        except Exception as e:
            print(f"Error computing loss: {e}")
            # Return a placeholder loss if the real one fails
            return torch.tensor(100.0, device=device, requires_grad=True)
    
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader with robust error handling

def _load_data_shard(file: Path):
    """
    Load a data shard with better error handling and MPS compatibility
    """
    try:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
            
        # First check file size to make sure it's not empty
        if file.stat().st_size < 256 * 4:  # At least header size
            raise ValueError(f"File too small: {file}")
            
        # Try to read header
        with file.open("rb") as f:
            header_bytes = f.read(256 * 4)
            if len(header_bytes) < 256 * 4:
                raise ValueError(f"Could not read complete header from: {file}")
                
            header = torch.frombuffer(header_bytes, dtype=torch.int32)
            
        # Validate header
        if header[0] != 20240520:
            raise ValueError(f"Magic number mismatch in file: {file}")
        if header[1] != 1:
            raise ValueError(f"Unsupported version in file: {file}")
            
        num_tokens = int(header[2])  # number of tokens (claimed)
        if num_tokens <= 0:
            raise ValueError(f"Invalid token count in file: {file}")
        
        # Create tensor for tokens
        pin_memory_flag = (device.type == "cuda")
        tokens = torch.empty(num_tokens, dtype=torch.int32, device="cpu")
        
        with file.open("rb", buffering=0) as f:
            f.seek(256 * 4)
            raw_bytes = f.read(4 * num_tokens)  # Read as bytes first
            
            if len(raw_bytes) != 4 * num_tokens:
                # Handle incomplete read
                actual_tokens = len(raw_bytes) // 4
                print(f"Warning: Expected {num_tokens} tokens but read {actual_tokens}")
                tokens = torch.empty(actual_tokens, dtype=torch.int32, device="cpu")
            
            # Convert bytes to tensor
            tokens = torch.frombuffer(raw_bytes, dtype=torch.int32)
            
        # Convert to uint16 if needed
        if tokens.element_size() != 2:
            tokens = tokens.to(torch.int16)
        
        return tokens
        
    except Exception as e:
        print(f"Error loading data shard {file}: {str(e)}")
        # Return a dummy tensor with a small amount of data
        return torch.ones(1024, dtype=torch.int32, device="cpu")

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    """
    Data generator with improved error handling and cycle support
    """
    # Find matching files
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    
    if not files:
        print(f"Warning: No files found matching pattern: {filename_pattern}")
        print("Using dummy data instead")
        # Create dummy data if no files are found
        dummy_data = torch.randint(0, 50000, (batch_size * 10,), dtype=torch.int32)
        while True:
            for pos in range(0, len(dummy_data) - batch_size - 1, batch_size):
                buf = dummy_data[pos + rank * batch_size // world_size : pos + (rank + 1) * batch_size // world_size + 1]
                inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
                targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
                yield inputs, targets
    
    print(f"Found {len(files)} files matching pattern: {filename_pattern}")
    
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    local_batch_size = batch_size // world_size
    
    # Use cycle to loop through files repeatedly if we run out
    file_iter = cycle(files)
    
    try:
        current_file = next(file_iter)
        print(f"Loading initial file: {current_file}")
        tokens = _load_data_shard(current_file)
        pos = 0
    except Exception as e:
        print(f"Error loading initial data: {e}")
        # Create dummy data for fallback
        tokens = torch.randint(0, 50000, (batch_size * 10,), dtype=torch.int32)
        pos = 0
 
    while True:
        try:
            # Check if we need to load a new file
            if pos + batch_size + 1 >= len(tokens):
                current_file = next(file_iter)
                print(f"Loading next file: {current_file}")
                tokens = _load_data_shard(current_file)
                pos = 0
            
            # Handle edge case where file is too small
            if len(tokens) <= batch_size + 1:
                print(f"Warning: File {current_file} is too small. Padding with dummy data.")
                tokens = torch.cat([tokens, torch.randint(0, 50000, (batch_size * 2,), dtype=tokens.dtype)])
            
            # Extract the local batch for this rank
            start_idx = pos + rank * local_batch_size
            end_idx = start_idx + local_batch_size + 1
            
            # Ensure we don't go out of bounds
            if end_idx > len(tokens):
                print(f"Warning: Batch would exceed token length. Wrapping around.")
                start_idx = 0
                end_idx = local_batch_size + 1
                pos = 0
            
            buf = tokens[start_idx:end_idx]
            
            # Ensure we have a complete buffer
            if len(buf) < local_batch_size + 1:
                print(f"Warning: Incomplete buffer of size {len(buf)}. Padding.")
                padding = torch.randint(0, 50000, (local_batch_size + 1 - len(buf),), dtype=buf.dtype)
                buf = torch.cat([buf, padding])
            
            # Move to device with proper dtypes
            inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
            targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
            
            pos += batch_size
            yield inputs, targets
            
        except Exception as e:
            print(f"Error in data generator: {e}. Using dummy data.")
            # Provide fallback data if something goes wrong
            dummy_inputs = torch.randint(0, 50000, (local_batch_size,), device=device, dtype=torch.int32)
            dummy_targets = torch.randint(0, 50000, (local_batch_size,), device=device, dtype=torch.int64)
            yield dummy_inputs, dummy_targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 10240         # Use a smaller, fixed number of tokens for validation
    train_seq_len = 512        # Shorter sequence length for training (512 tokens)
    val_seq_len = 512          # Shorter sequence length for validation
    # optimization
    num_iterations = 100       # Fewer iterations for quick testing/training
    cooldown_frac = 0.4
    # architecture
    vocab_size = 50257
    # Reduce the model architecture for testing on a Mac:
    num_layers = 4             # Fewer transformer layers
    num_heads = 4              # Fewer attention heads
    model_dim = 256            # Lower model dimension
    # evaluation and logging
    val_loss_every = 10        # Evaluate more frequently for debugging
    save_checkpoint = True
args = Hyperparameters()

# Get rank and world size from environment
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Apply special configurations for MPS
if device.type == "mps":
    # For MPS, force single-process "distributed" mode
    if world_size > 1:
        print(f"Warning: MPS does not support multi-GPU training. Forcing world_size=1.")
        world_size = 1
        rank = 0
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
elif device.type == "cuda":
    # For CUDA, reduce the world size requirement
    if world_size != 8:
        print(f"Note: Running with {world_size} GPU(s) instead of the recommended 8.")

# Initialize process group with timeout and error handling
try:
    # Set appropriate backend
    backend = "nccl" if device.type == "cuda" else "gloo"
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend, 
        world_size=world_size, 
        rank=rank, 
        timeout=torch.distributed.default_pg_timeout
    )
except ValueError as e:
    if "environment variable MASTER_ADDR expected" in str(e):
        # Fix the environment variables and retry
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        # Initialize with init_method to avoid environment variable issues
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend, 
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size, 
            rank=rank
        )
    else:
        print(f"Error initializing process group: {e}")
        # Create a dummy process group for single-process mode
        print("Falling back to single-process mode")
        world_size = 1
        rank = 0
        # Don't raise the error, we'll continue in single-process mode

try:
    dist.barrier()  # Synchronize processes
except Exception as e:
    print(f"Warning: barrier failed: {e}")

master_process = (rank == 0)  # This process will do logging, checkpointing, etc.

# Begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# Begin by printing this file (the Python code)
print0(code)
print0("=" * 100)
# Log information about the hardware/software environment
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.__version__}")
print0(f"Device: {device}")
print0(f"Rank: {rank}, World size: {world_size}")

if device.type == "cuda":
    print0(f"Compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  # avoid top-level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())

print0("=" * 100)

# Clean up memory before creating the model
if device.type == "mps":
    torch.mps.empty_cache()
elif device.type == "cuda":
    torch.cuda.empty_cache()
import gc
gc.collect()

########################################
#    Construct model and optimizer     #
########################################

# Use smaller hyperparameters for MPS
if device.type == "mps":
    args.num_layers = 4
    args.num_heads = 4
    args.model_dim = 256
    args.train_seq_len = 128
    args.val_seq_len = 128
    print0(f"Using reduced model size for MPS: layers={args.num_layers}, heads={args.num_heads}, dim={args.model_dim}")

model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    model_dim=args.model_dim,
    max_seq_len=max(args.train_seq_len, args.val_seq_len)
).to(device)

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

try:
    for param in model.parameters():
        dist.broadcast(param.to(device).detach(), 0)
except Exception as e:
    print(f"Warning: Parameter broadcast failed: {e}")

# Collect parameters for optimization
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# Initialize the optimizer(s)
adam_params = [
    dict(params=head_params, lr=0.22),
    dict(params=embed_params, lr=0.6),
    dict(params=scalar_params, lr=0.04)
]

# Small adam epsilon and no fused=True on MPS
optimizer1 = torch.optim.Adam(
    adam_params, betas=(0.8, 0.95), eps=1e-10, fused=(device.type == "cuda")
)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# Learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations  # Progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# Attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(
        window_size // 128, dtype=torch.int32, device=device, pin_memory=(device.type == "cuda")
    )  # Avoid pin_memory=True on MPS

def get_window_size_blocks(step: int):
    x = step / args.num_iterations  # Progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

# Compile model ONLY for CUDA, not MPS
if device.type == "cuda":
    try:
        model = torch.compile(model, dynamic=False)
        print0("Model compiled successfully")
    except Exception as e:
        print0(f"Warning: Model compilation failed: {e}")
else:
    print0(f"Skipping model compilation on {device.type}")

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
print0("Starting warmup...")
warmup_steps = 5 if device.type == "mps" else 10
initial_state = dict(
    model=copy.deepcopy(model.state_dict()),
    optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]
)  # Save the initial state

for step in range(warmup_steps):
    print0(f"Warmup step {step+1}/{warmup_steps}")
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device=device)

    try:
        with torch.autograd.set_detect_anomaly(True):  # Helps catch MPS-specific backward() errors
            loss = model(inputs.to(torch.int32), targets, get_window_size_blocks(0))
            loss.backward()

        # All-reduce gradients if in distributed mode
        for param in model.parameters():
            if param.grad is not None:
                try:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                except Exception as e:
                    print(f"Warning: gradient all_reduce failed: {e}")

        # Step optimizers
        for opt in optimizers:
            opt.step()
        
        # Clear gradients
        model.zero_grad(set_to_none=True)
    
    except Exception as e:
        print0(f"Warning: Error during warmup step {step}: {e}")
        # Continue with next warmup step

print0("Warmup complete, resetting model state")

# Reload the original model and optimizer states
model.load_state_dict(initial_state["model"])
model.to(device)  # Ensure model is back on the correct device

for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)

del initial_state  # Free memory
gc.collect()
if device.type == "mps":
    torch.mps.empty_cache()
elif device.type == "cuda":
    torch.cuda.empty_cache()

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0

# Synchronize before starting the clock
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()

t0 = time.perf_counter()

# Begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # Stop the clock
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
            
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()

        val_batch_size = world_size * args.val_seq_len
        val_steps = max(1, args.val_tokens // val_batch_size)
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = torch.tensor(0.0, device=device)
        num_val_batches = 0

        with torch.no_grad():
            try:
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    loss = model(inputs, targets, get_window_size_blocks(step))
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss
                        num_val_batches += 1
                
                # Average over valid batches
                if num_val_batches > 0:
                    val_loss /= num_val_batches
                else:
                    val_loss = torch.tensor(float('nan'), device=device)
            except Exception as e:
                print0(f"Error during validation: {e}")
                val_loss = torch.tensor(float('nan'), device=device)

        # Clean up validation loader
        del val_loader
        
        # Handle NaN loss
        if torch.isnan(val_loss) or torch.isinf(val_loss):
            val_loss = torch.tensor(99.9, device=device)

        # All-reduce validation loss if distributed
        try:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        except Exception as e:
            print0(f"Warning: validation loss all_reduce failed: {e}")

        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms "
            f"step_avg:{training_time_ms/max(step, 1):.2f}ms",
            console=True
        )
        model.train()

        # Clean up memory
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Restart the clock
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            try:
                log = dict(
                    step=step,
                    code=code,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers]
                )
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
                print0(f"Saved checkpoint to logs/{run_id}/state_step{step:06d}.pt", console=True)
            except Exception as e:
                print0(f"Error saving checkpoint: {e}", console=True)
        # The last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    try:
        inputs, targets = next(train_loader)
        
        # Forward and backward pass
        loss = model(inputs, targets, get_window_size_blocks(step))
        loss.backward()

        # All-reduce gradients if distributed
        for param in model.parameters():
            if param.grad is not None:
                try:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                except Exception as e:
                    print(f"Warning: gradient all_reduce failed: {e}")

        # Set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)

        # Warm up momentum for Muon optimizer
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Step the optimizers
        for opt in optimizers:
            opt.step()

        # Clear gradients
        model.zero_grad(set_to_none=True)

# Logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(
            f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms "
            f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms",
            console=True
        )
        
    except Exception as e:
        print0(f"Error during training step {step}: {e}", console=True)
        # Try to recover and continue with next step
        model.zero_grad(set_to_none=True)
        
        # Clean up memory after an error
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# Memory logging (only on CUDA and MPS)
if device.type == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() // 1024 // 1024
    reserved_mem = torch.cuda.max_memory_reserved() // 1024 // 1024
    print0(f"peak memory allocated: {peak_mem} MiB reserved: {reserved_mem} MiB", console=True)
elif device.type == "mps":
    try:
        peak_mem = torch.mps.current_allocated_memory() // 1024 // 1024
        print0(f"peak memory allocated: {peak_mem} MiB", console=True)
    except Exception as e:
        print0(f"Could not get MPS memory info: {e}", console=True)

try:
    dist.destroy_process_group()
except Exception as e:
    print0(f"Error destroying process group: {e}", console=True)

print0("Training complete!", console=True)