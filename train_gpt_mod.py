from dataclasses import dataclass
import os
import sys
import torch
import torch.distributed as dist
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
from torch import Tensor, nn
import torch.nn.functional as F
from functools import lru_cache
import copy
from pathlib import Path
import glob
import time
mps_available = torch.backends.mps.is_available()
cuda_available = torch.cuda.is_available()

# Define device globally to ensure consistency
device = None

if cuda_available:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#MPS does not need this workaround because MPS initializes differently.
# prevents a bug on some systems
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.empty(1, device=device, requires_grad=True).backward()

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    FLEX_ATTENTION_SUPPORTED = True
except ImportError:
    print("Warning: FlexAttention not supported, using standard attention.")
    FLEX_ATTENTION_SUPPORTED = False

#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


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


# Backward pass of a custom FP8 matrix multiplication operation

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(
    g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float
) -> tuple[Tensor, Tensor]:
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)

        global device  # Use the global device variable
        if device.type == "cuda":
            # Optimized CUDA FP8 matmul
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
        elif device.type == "mps":
            # Alternative computation for MPS using bfloat16 matmul
            grad_f8 = grad.to(torch.bfloat16)
            grad_x = (grad_f8 @ w_f8.T).to(torch.bfloat16)
            grad_w = (x_f8.T @ grad_f8).to(torch.float32)
        else:
            # For CPU or unsupported devices, return fake fallback values
            return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

        return grad_x, grad_w

    if device is not None and device.type == "cuda":
        impl = torch.compile(impl)

    return impl(g, x_f8, w_f8)

# **Fallback for unsupported backends**
@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


# Automatic Differentiation (autograd) for the custom FP8 matrix multiplication operator(mm_op)
# They define how gradients are computed during backpropagation
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
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_newtonschulz5_nocompile(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1, device="cuda"):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.float32, device=device)
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # For MPS, we'll use a simpler non-distributed implementation
            if self.device == "mps":
                params: list[Tensor] = group["params"]
                for p in params:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)

                    # Update momentum buffer
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - group["momentum"])

                    # Get gradient with momentum
                    g = p.grad.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                    # Use the non-compiled version for MPS
                    g_processed = zeropower_via_newtonschulz5_nocompile(g, steps=group["ns_steps"])

                    # Apply update with the scaling factor
                    lr_scaled = -group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5
                    p.add_(g_processed, alpha=lr_scaled)
            else:
                # Original distributed implementation for CUDA
                update_buffer: Tensor = group["update_buffer"]
                update_buffer_views: list[Tensor] = group["update_buffer_views"]
                params: list[Tensor] = group["params"]
                handle = None
                params_world = None

                def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.add_(g_world.view_as(p_world),
                                    alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

                for base_i in range(len(params))[::self.world_size]:
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad
                        assert g is not None
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        if self.device == "cuda":
                            g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                        else:
                            g = zeropower_via_newtonschulz5_nocompile(g, steps=group["ns_steps"]).flatten()
                    else:
                        g = update_buffer_views[self.rank]
                    if base_i > 0:
                        update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i : base_i + self.world_size]
                update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8 and device.type == "cuda"  # FP8 only on CUDA, not MPS
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
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

    def forward(self, x_BTHD: Tensor, position_ids=None):
        # Support custom position IDs for APBPB calculation
        B, T, H, D = x_BTHD.shape
        device = x_BTHD.device  # Get device from input tensor

        # Make sure cos and sin are on the same device as the input
        cos_device = self.cos.to(device)
        sin_device = self.sin.to(device)

        if position_ids is not None:
            # Move position_ids to the same device if needed
            if position_ids.device != device:
                position_ids = position_ids.to(device)
                
            # Custom positions - validate and use them
            max_pos = cos_device.size(0) - 1

            # Check if positions contain float values
            if position_ids.dtype in [torch.float32, torch.float64]:
                # Handle float positions with interpolation
                positions_floor = position_ids.floor().to(torch.int64)
                positions_ceil = position_ids.ceil().to(torch.int64)
                fractions = position_ids - positions_floor
                
                # Ensure positions are within valid range
                positions_floor = torch.clamp(positions_floor, 0, max_pos)
                positions_ceil = torch.clamp(positions_ceil, 0, max_pos)
                
                # Handle the shape based on how position_ids is provided
                if positions_floor.dim() == 1:
                    positions_floor = positions_floor.unsqueeze(0)
                    positions_ceil = positions_ceil.unsqueeze(0)
                    fractions = fractions.unsqueeze(0)
                
                # Look up values for floor and ceiling positions
                cos_floor = torch.index_select(cos_device, 0, positions_floor.view(-1)).view(B, T, 1, -1)
                sin_floor = torch.index_select(sin_device, 0, positions_floor.view(-1)).view(B, T, 1, -1)
                cos_ceil = torch.index_select(cos_device, 0, positions_ceil.view(-1)).view(B, T, 1, -1)
                sin_ceil = torch.index_select(sin_device, 0, positions_ceil.view(-1)).view(B, T, 1, -1)
                
                # Linearly interpolate based on fractional part
                fractions = fractions.view(B, T, 1, 1)
                cos = cos_floor * (1 - fractions) + cos_ceil * fractions
                sin = sin_floor * (1 - fractions) + sin_ceil * fractions
            else:
                # Integer positions - ensure they're within bounds and on correct device
                positions = torch.clamp(position_ids, 0, max_pos)
                
                # Handle the shape based on how position_ids is provided
                if positions.dim() == 1:
                    positions = positions.unsqueeze(0)  # [1, T]
                
                # Get cos/sin for these specific positions
                cos = torch.index_select(cos_device, 0, positions.view(-1)).view(B, T, 1, -1)
                sin = torch.index_select(sin_device, 0, positions.view(-1)).view(B, T, 1, -1)
        else:
            # Standard sequential positions
            cos = cos_device[:T].unsqueeze(0).unsqueeze(2)
            sin = sin_device[:T].unsqueeze(0).unsqueeze(2)

        # Apply rotary embeddings
        x1, x2 = x_BTHD.chunk(2, dim=-1)

        # Apply rotation
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos

        return torch.cat((y1, y2), dim=-1)


# CausalSelfAttention class implements a multi-head self-attention mechanism using a causal mask
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask | Tensor, position_ids=None):
        B, T = x.size(0), x.size(1) # batch size, sequence length

        # Only assert batch size on CUDA as it's required for FlexAttention
        if device.type == "cuda":
            assert B == 1, "Batch size must be 1 for FlexAttention (CUDA)."

        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)) \
            .view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)

        # Pass position_ids to Rotary embeddings
        q = self.rotary(q, position_ids)
        k = self.rotary(k, position_ids)

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

        # Use flex_attention on CUDA with BlockMask, standard attention otherwise
        if device.type == "cuda" and FLEX_ATTENTION_SUPPORTED and isinstance(block_mask, BlockMask):
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            block_mask=block_mask, scale=0.12).transpose(1, 2)
        else:
            # Standard attention with custom mask support
            q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = self.standard_attention(q, k, v, attention_mask=block_mask)
            y = y.transpose(1, 2)  # [B, T, num_heads, head_dim]

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

    def standard_attention(self, q, k, v, attention_mask=None):
        """
        Standard Scaled Dot-Product Attention with support for custom masks
        """
        scale_factor = (self.head_dim ** -0.5)

        # Compute attention scores and ensure correct precision
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

        # Apply custom attention mask if provided
        if attention_mask is not None:
            # Prepare the mask for broadcasting to [B, H, T, T]
            if attention_mask.dim() == 2:  # [T, T]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

            # Convert to boolean if it's not already (True = keep, False = mask)
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

            # Apply the mask (set masked positions to -1e9)
            attn_scores = attn_scores.masked_fill(~attention_mask, -1e9)
        else:
            # Fall back to causal mask
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
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # Zero init

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

# Transformer Block
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask | Tensor, position_ids=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            # Pass position_ids to attention mechanism
            x = x + self.attn(norm(x), ve, block_mask, position_ids=position_ids)
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

    def convert_to_block_mask(self, attention_mask):
        """
        Convert a boolean attention mask to the BlockMask format used by FlexAttention.
        """
        if (not FLEX_ATTENTION_SUPPORTED) or (device.type != "cuda"):
            return attention_mask

        BLOCK_SIZE = 128

        # Ensure mask is on the correct device and in the right format
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        if attention_mask.dim() > 2:
            # Extract just the mask for the first batch and head if multi-dimensional
            attention_mask = attention_mask[0, 0]

        T = attention_mask.size(0)
        # Make sure T is a multiple of BLOCK_SIZE for BlockMask
        padded_T = ((T + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if T < padded_T:
            # Pad the mask to a multiple of BLOCK_SIZE
            padded_mask = torch.zeros((padded_T, padded_T), dtype=attention_mask.dtype, device=device)
            padded_mask[:T, :T] = attention_mask
            attention_mask = padded_mask

        num_blocks = padded_T // BLOCK_SIZE

        # Create a mask_mod function that applies the original attention pattern
        def custom_mask_mod(b, h, q_idx, kv_idx):
            q_start = q_idx * BLOCK_SIZE
            q_end = min((q_idx + 1) * BLOCK_SIZE, padded_T)
            kv_start = kv_idx * BLOCK_SIZE
            kv_end = min((kv_idx + 1) * BLOCK_SIZE, padded_T)

            return attention_mask[q_start:q_end, kv_start:kv_end]

        # Create block-level view of the mask
        block_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
        for i in range(num_blocks):
            for j in range(num_blocks):
                i_start, i_end = i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE
                j_start, j_end = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
                # A block is attended to if any element in it can be attended to
                block_mask[i, j] = attention_mask[i_start:i_end, j_start:j_end].any()

        # Create block indices
        full_blocks = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
        partial_blocks = block_mask & ~full_blocks

        # Convert to ordered format
        full_num_blocks, full_indices = self.dense_to_ordered_helper(full_blocks)
        partial_num_blocks, partial_indices = self.dense_to_ordered_helper(partial_blocks)

        # Create and return BlockMask
        return BlockMask.from_kv_blocks(
            partial_num_blocks,
            partial_indices,
            full_num_blocks,
            full_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=custom_mask_mod,
        )

    def dense_to_ordered_helper(self, dense_blockmask: Tensor):
        """Helper for converting dense block masks to ordered format for BlockMask"""
        num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, sliding_window_num_blocks=None,
            input_seq=None, target_seq=None, return_early_embeddings=False, early_layer_idx=0):
        # Support both training and inference modes
        is_training = target_seq is not None

        # Handle input from either input_ids (for APBPB) or input_seq (for regular use)
        if input_ids is not None:
            input_seq = input_ids
        elif input_seq is None and not is_training:
            raise ValueError("Either input_ids or input_seq must be provided")

        # Handle single token or sequence input
        is_single_token = input_seq.ndim == 0
        if is_single_token:
            input_seq = input_seq.unsqueeze(0)

        # Check if input is batched or not
        is_batched = input_seq.ndim == 2
        if not is_batched:
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension

        # Set default sliding window blocks if not provided
        if sliding_window_num_blocks is None and attention_mask is None:
            sliding_window_num_blocks = torch.tensor(max(1, input_seq.size(-1) // 128),
                                                device=input_seq.device, dtype=torch.int32)

        # Determine which attention mechanism to use
        if attention_mask is not None:
            # Custom mask provided for APBPB calculation
            if device.type == "cuda" and FLEX_ATTENTION_SUPPORTED:
                # Convert to BlockMask format for each layer
                block_masks = [self.convert_to_block_mask(attention_mask) for _ in range(len(self.blocks))]
            else:
                # For MPS/CPU, use the mask directly
                block_masks = [attention_mask for _ in range(len(self.blocks))]
        else:
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

        # Get initial embeddings
        x = x0 = norm(self.embed(input_seq))
        early_embeddings = None
        
        # U-net design
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i], position_ids=position_ids)
            
            # Capture embeddings from the specified early layer
            if return_early_embeddings and i == early_layer_idx:
                early_embeddings = x.clone()
                
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))

        # For training with early embeddings
        if is_training and return_early_embeddings:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq,
                                reduction='sum' if self.training else 'mean')
            return loss, early_embeddings
        
        # For training, calculate loss and return it
        if is_training:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq,
                                  reduction='sum' if self.training else 'mean')
            return loss
            
        # For inference with early embeddings
        if return_early_embeddings:
            return logits, early_embeddings
            
        # For standard inference
        return logits

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
import numpy as np
def _load_data_shard(file: Path):
    with file.open("rb") as f:
        # Read just the first few integers of the header instead of all 256
        header_bytes = f.read(3 * 4)  # Read only 3 integers (12 bytes)
        header = np.frombuffer(header_bytes, dtype=np.int32)
        # Check the header values
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        claimed_num_tokens = int(header[2])  # number of tokens (claimed)
        # Determine file size to validate token count
        f.seek(0, 2)  # Seek to end of file
        file_size = f.tell()
        max_possible_tokens = (file_size - 256 * 4) // 2  # Each token is 2 bytes
        # Use the smaller of claimed or possible tokens
        actual_num_tokens = min(claimed_num_tokens, max_possible_tokens)
        if actual_num_tokens < claimed_num_tokens:
            print(f"Warning: Header claims {claimed_num_tokens} tokens but file can only contain {actual_num_tokens}")
        # Now read the token data
        f.seek(256 * 4)  # Skip the 256-int header
        token_bytes = f.read(actual_num_tokens * 2)  # Read token data (2 bytes per token)

        # Create tensor from bytes
        tokens = torch.frombuffer(token_bytes, dtype=torch.uint16)

    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training

    # Handle case where no files are found
    if not files:
        print(f"Warning: No files found matching pattern: {filename_pattern}. Using dummy data.")
        dummy_data = torch.randint(0, 50000, (batch_size * 10,), dtype=torch.int32, device="cpu")
        while True:
            for pos in range(0, len(dummy_data) - batch_size - 1, batch_size):
                inputs = dummy_data[pos + rank * local_batch_size:][:local_batch_size].to(device=device, dtype=torch.int32)
                targets = dummy_data[pos + rank * local_batch_size + 1:][:local_batch_size].to(device=device, dtype=torch.int64)
                yield inputs, targets

    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                print("Warning: Reached end of file list, restarting from beginning")
                file_iter = iter(files)
                tokens, pos = _load_data_shard(next(file_iter)), 0

        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

def sequential_data_generator(filename_pattern: str, batch_size: int):
    """
    Simple non-distributed data generator for MPS.
    Loads data sequentially without rank-based indexing.
    """
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]

    if not files:
        print(f"Warning: No files found matching pattern: {filename_pattern}. Using dummy data.")
        dummy_data = torch.randint(0, 50000, (batch_size * 10,), dtype=torch.int32, device="cpu")
        while True:
            for pos in range(0, len(dummy_data) - batch_size - 1, batch_size):
                buf = dummy_data[pos : pos + batch_size + 1]
                inputs = buf[:-1].to(device=device, dtype=torch.int32)
                targets = buf[1:].to(device=device, dtype=torch.int64)
                yield inputs, targets

    file_iter = iter(files)
    try:
        tokens, pos = _load_data_shard(next(file_iter)), 0
    except Exception as e:
        print(f"Warning: Failed to load data shard: {e}. Using dummy data.")
        dummy_data = torch.randint(0, 50000, (batch_size * 10,), dtype=torch.int32, device="cpu")
        while True:
            for pos in range(0, len(dummy_data) - batch_size - 1, batch_size):
                buf = dummy_data[pos : pos + batch_size + 1]
                inputs = buf[:-1].to(device=device, dtype=torch.int32)
                targets = buf[1:].to(device=device, dtype=torch.int64)
                yield inputs, targets

    while True:
        if pos + batch_size + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                print("Warning: Reached end of file list, restarting from beginning")
                file_iter = iter(files)
                tokens, pos = _load_data_shard(next(file_iter)), 0

        buf = tokens[pos : pos + batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32)
        targets = buf[1:].to(device=device, dtype=torch.int64)
        pos += batch_size
        yield inputs, targets

@dataclass
class Hyperparameters:
    # Instance fields that can be passed to constructor
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 4
    model_dim: int = 256
    max_seq_len: int = 512  # Add this to match usage in load_model_from_checkpoint
    
    # Other configuration options
    train_files: str = "data/fineweb10B/fineweb_train_000001.bin"
    val_files: str = "data/fineweb10B/fineweb_val_000000.bin"
    val_tokens: int = 10240
    train_seq_len: int = 512
    val_seq_len: int = 512
    num_iterations: int = 1000
    cooldown_frac: float = 0.4
    val_loss_every: int = 10
    save_checkpoint: bool = True
    use_segmentation_invariant: bool = False
args = Hyperparameters()

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

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    tensor = torch.tensor(window_size // 128, dtype=torch.int32)

    if device.type == "cuda":
        return tensor.pin_memory().cuda(non_blocking=True)
    return tensor.to(device)

def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

def main():
    if cuda_available:
        device_type = "cuda"
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        distributed_training = True
    elif mps_available:
        device_type = "mps"
        rank = 0
        world_size = 1
        device = torch.device("mps")
        distributed_training = False
    else:
        raise RuntimeError("No compatible GPU found, make sure you have CUDA or MPS")

    # Initialize distributed training only for CUDA
    if distributed_training:
        backend = "nccl"
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
        master_process = (rank == 0) # this process will do logging, checkpointing etc.
    else:
        master_process = True

    # begin logging
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

    # begin by printing this file (the Python code)
    print0(code)
    print0("="*100)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__}")
    if cuda_available:
        print0(f"PyTorch compiled for CUDA {torch.version.cuda}")

        def nvidia_smi():
            import subprocess  # avoid top-level import
            return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

        print0(nvidia_smi())

    elif mps_available:
        print0("Running on Apple MPS (Metal Performance Shaders)")
        # Add MPS-specific memory info if available
        try:
            import platform
            print0(f"Running on {platform.processor()} with macOS {platform.mac_ver()[0]}")
        except:
            pass

    else:
        print0("Running on CPU (No GPU detected)")

    print0("="*100)

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

    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, num_heads=args.num_heads,
                        model_dim=args.model_dim, max_seq_len=max(args.train_seq_len, args.val_seq_len)).to(device)

    if cuda_available:
        # Convert embeddings to bfloat16
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()

        # Broadcast model parameters
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # Select the optimizer based on device type
    if device.type == "cuda":
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    else:  # MPS case
        #optimizer2 = torch.optim.AdamW(hidden_matrix_params, lr=0.05, betas=(0.9, 0.95))
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size, device="mps")

    # First optimizer (same for both CUDA and MPS)
    adam_params = [
        dict(params=head_params, lr=0.22),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04)
    ]
    optimizer1 = torch.optim.AdamW(adam_params, betas=(0.8, 0.95), eps=1e-10)

    # Maintain the same structure for optimizers
    optimizers = [optimizer1, optimizer2]

    # Store initial learning rate for scheduling
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations  # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0  # Keep learning rate at max for most of training
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1  # Linearly decay to 0.1x initial LR


    # Compile model ONLY for CUDA, not MPS
    if device.type == "cuda":
        try:
            model = torch.compile(model, dynamic=False)
            print0("Model compiled successfully")
        except Exception as e:
            print0(f"Warning: Model compilation failed: {e}")
    else:
        print0(f"Skipping model compilation on {device.type}")


    # Use correct data generator based on backend
    if device.type == "cuda":
        train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
        val_loader = distributed_data_generator(args.val_files, world_size * args.val_seq_len, rank, world_size)
    else:  # Use sequential loading on MPS
        warmup_loader = sequential_data_generator(args.train_files, args.train_seq_len)
        train_loader = sequential_data_generator(args.train_files, args.train_seq_len)
        val_loader = sequential_data_generator(args.val_files, args.val_seq_len)


    ########################################
    #            Warmup kernels            #
    ########################################

    from neon.zig.bindings import get_token_to_id_mapping
    from tools.tokenizer_utils import seg_inv_enc
    import tiktoken
    from tools.trainable_position_encoding import TrainableSegInvPositionalEncoding
    tokenizer = tiktoken.get_encoding("gpt2")
    token_dict = get_token_to_id_mapping(tokenizer)

    # Initialize the trainable positional encoding
    trainable_pe = TrainableSegInvPositionalEncoding(token_dict)
    trainable_pe = trainable_pe.to(device)  # Move to the same device as your model

    # Add the trainable_pe parameters to your optimizer
    for opt in optimizers:
        # Create a new parameter group for positional encoding
        opt.add_param_group({'params': trainable_pe.parameters(), 
                    'initial_lr': opt.param_groups[0]['initial_lr']})

    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        trainable_pe=copy.deepcopy(trainable_pe.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]
    )
    
    for _ in range(warmup_steps):
        inputs, targets = next(warmup_loader)
        positions, attn_mask = seg_inv_enc(inputs, token_dict, max_seq_len=args.max_seq_len, trainable_pe=trainable_pe)
        attn_mask = attn_mask.to(device)
        model(input_ids=inputs.to(torch.int32),  target_seq= targets, sliding_window_num_blocks=get_window_size_blocks(0), position_ids=positions, attention_mask=attn_mask).backward()
        if device.type == "cuda":
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            # Also reduce gradients for trainable_pe
            for param in trainable_pe.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        for opt in optimizers:
            opt.step()
        
        model.zero_grad(set_to_none=True)
        trainable_pe.zero_grad(set_to_none=True)

    # Restore initial state
    model.load_state_dict(initial_state["model"])
    trainable_pe.load_state_dict(initial_state["trainable_pe"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)

    del initial_state  # Free up memory

    ########################################
    #        Training and validation       #
    ########################################
    training_time_ms = 0

    # start the clock
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            print0(f"Starting validation at step {step}/{train_steps}...", console=True)
            model.eval()
            trainable_pe.eval()  # Set trainable_pe to evaluation mode too
            val_loss = 0
            val_steps = args.val_tokens // args.val_seq_len
            with torch.no_grad():
                for val_step in range(val_steps):
                    if val_step == 0:
                        print0(f"  Running validation step 1/{val_steps}...", console=True)
                    inputs, targets = next(val_loader)
                    # Generate positions using trainable_pe
                    positions, attn_mask = seg_inv_enc(inputs, token_dict, 
                                                    max_seq_len=args.max_seq_len, 
                                                    trainable_pe=trainable_pe)
                    attn_mask = attn_mask.to(device)
                    batch_loss = model(input_ids=inputs, target_seq=targets, 
                                    sliding_window_num_blocks=get_window_size_blocks(step),
                                    position_ids=positions, attention_mask=attn_mask)
                    val_loss += batch_loss
                    if val_step == val_steps - 1:
                        print0(f"  Completed validation step {val_steps}/{val_steps}", console=True)

            val_loss /= val_steps

            if device.type == "cuda":
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            # More detailed logging
            print0(f"RESULT - step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            model.train()
            trainable_pe.train()  # Set trainable_pe back to training mode

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                try:
                    checkpoint_dir = f"logs/{run_id}"
                    checkpoint_file = f"{checkpoint_dir}/state_step{step:06d}.pt"

                    print0(f"Attempting to save checkpoint to {checkpoint_file}", console=True)

                    os.makedirs(checkpoint_dir, exist_ok=True)

                    log = dict(
                        step=step,
                        code=code,
                        model=model.state_dict(),
                        trainable_pe=trainable_pe.state_dict(),  # Save trainable_pe state
                        optimizers=[opt.state_dict() for opt in optimizers]
                    )

                    torch.save(log, checkpoint_file)
                    print0(f"Successfully saved checkpoint to {checkpoint_file}", console=True)
                except Exception as e:
                    print0(f"ERROR saving checkpoint: {str(e)}", console=True)
            else:
                print0(f"Checkpoint saving skipped: master_process={master_process}, save_checkpoint={args.save_checkpoint}", console=True)
            break  # End training

        # --------------- TRAINING SECTION -----------------
        id_to_token = {v: k for k, v in token_dict.items()}
        inputs, targets = next(train_loader)
        
        # Generate positions using trainable_pe
        positions, attn_mask = seg_inv_enc(inputs, token_dict, 
                                        max_seq_len=args.max_seq_len, 
                                        trainable_pe=trainable_pe)
        attn_mask = attn_mask.to(device)

        # Forward pass with positions, attention mask, and return early embeddings
        loss, early_embeddings = model(input_ids=inputs, target_seq=targets, 
                                    sliding_window_num_blocks=get_window_size_blocks(step),
                                    position_ids=positions, attention_mask=attn_mask, 
                                    return_early_embeddings=True, early_layer_idx=0)  # Use first layer

        # Calculate token end positions
        from loss_calculation import calculate_token_end_positions, calculate_learn_to_spell_loss, combine_losses
        token_end_positions = calculate_token_end_positions(inputs, token_dict, id_to_token)

        # Calculate learn to spell loss
        spell_loss = calculate_learn_to_spell_loss(early_embeddings, token_end_positions)

        # Combine losses with a weight of 0.01 (adjust as needed)
        total_loss = combine_losses(loss, spell_loss, spell_loss_weight=0.01)

        # Backward pass
        total_loss.backward()

        if device.type == "cuda":
            # Reduce gradients for model parameters
            for param in model.parameters():
                if param.grad is not None:  # Check if grad exists before all_reduce
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            
            # Reduce gradients for trainable_pe parameters
            for param in trainable_pe.parameters():
                if param.grad is not None:  # Check if grad exists
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # Update learning rates
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)

        if device.type == "cuda":
            for group in optimizer2.param_groups:
                frac = min(step / 300, 1)
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Optimization step
        for opt in optimizers:
            opt.step()

        # Zero gradients
        model.zero_grad(set_to_none=True)
        trainable_pe.zero_grad(set_to_none=True)  # Zero trainable_pe gradients

        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} orig_loss:{loss:.4f} spell_loss:{spell_loss:.4f} total_loss:{total_loss:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    # Print final memory usage information
    if device.type == "cuda":
        print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    elif device.type == "mps":
        print0(f"Training completed on MPS device", console=True)

    # Clean up distributed resources
    if device.type == "cuda":
        dist.destroy_process_group()

if __name__ == "__main__":
    main()