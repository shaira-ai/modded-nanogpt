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

# Set the appropriate device for Mac M-series
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Remove CUDA-specific environment variable
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Not needed for MPS

import torch
torch.empty(1, device=device, requires_grad=True).backward() # Prevents a bug on some systems

from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# MPS does not support FlexAttention, disabling it for now
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
except ImportError:
    flex_attention = None
    BlockMask = None

# Note: torch._inductor.config.coordinate_descent_tuning is CUDA-specific and should be removed

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng (Modified for MPS Compatibility)

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()

        # FP8 is only supported on CUDA, fallback to BF16 for Mac
        if device == "cuda":
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

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    
    # Use BF16 on Mac and CPU, FP8 on CUDA
    if device == "cuda":
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    else:
        return x @ w.T, x.to(torch.bfloat16), w.to(torch.bfloat16)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)

        if device == "cuda":
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
    assert G.ndim >= 2  # Supports batched operations

    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Ensure tensor is on the correct device
    G = G.to(device)

    # MPS does not always support bfloat16, so fallback to float32 if needed
    if device == "cuda":
        X = G.to(torch.bfloat16)
    else:
        X = G.to(torch.float32)  # More stable on MPS

    if G.size(-2) > G.size(-1):
        X = X.mT  # Transpose if rows > cols

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # Quintic computation
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT  # Reverse transpose if applied earlier

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
    - On Mac (`mps`), `bfloat16` is not always stable. This implementation uses `float32` on Mac to
      prevent numerical issues while maintaining speed.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [p.to(device) for p in params]  # Ensure tensors are on the correct device
        param_groups = []

        for size in {p.numel() for p in params}:
            # Mac MPS does not fully support bfloat16, fallback to float32 if needed
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            b = torch.empty(world_size, size, dtype=dtype, device=device)
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.

        - Uses momentum-based updates and applies Newton-Schulz orthogonalization.
        - Implements an async all_gather communication strategy to minimize synchronization overhead.
        - Uses `bfloat16` on CUDA, but falls back to `float32` on MPS to ensure numerical stability.
        """
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]

            # Generate weight updates in a distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # Optimized Muon implementation contributed by @YouJiacheng
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
                        state["momentum_buffer"] = torch.zeros_like(g, device=device)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    # Apply Newton-Schulz orthogonalization
                    g = zeropower_via_newtonschulz5(g.to(device), steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # Async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()  # Final update after last batch

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8 and device == "cuda"  # Disable FP8 on Mac and CPU
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
            target_dtype = torch.float32 if device == "mps" else x.dtype
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
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

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
        self.use_flex = (device.type != "mps")

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask | None):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        if device.type == "cuda":
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
            chunk_size = 1024
            outputs = []
            scale = 0.12  # using the same scale as in flex_attention
            for i in range(0, T, chunk_size):
                # q_chunk has shape (B*num_heads, chunk_size, head_dim)
                q_chunk = q_[:, i:i+chunk_size, :]
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
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(
            model_dim,
            next_multiple_of_n(vocab_size, n=128),
            use_fp8=(device == "cuda"),  # Disable FP8 on Mac and CPU
            x_s=(model_dim**0.5)/448,
            w_s=24/448,
            grad_s=1/448
        )
        self.lm_head.weight.detach().zero_()  # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2, device=device))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
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
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapted from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks.to(device)), build_bm((sliding_window_num_blocks // 2).to(device))

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1
        # Ensure tensors are on the correct device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device, dtype=torch.long)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks.to(device))
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)
        # Ensure embedding and normalization are consistent in dtype
        # Always cast the embedding output to a floating-point type
        x = x0 = norm(self.embed(input_seq)[None].to(torch.float32))  # use of norm here by @Grad62304977
        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n].to(x.dtype) * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).to(torch.float32)  # Explicitly set float32 for stability on Mac
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        # Compute loss (ensure dtype consistency)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean'
        )
        return loss
    
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    
    with file.open("rb", buffering=0) as f:
        # Avoid pin_memory=True on Mac (`mps`)
        pin_memory_flag = device == "cuda"
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory_flag)
        
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy(force=True))  # `force=True` ensures compatibility
        
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        
        # Use `device=device` instead of `"cuda"`
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    val_tokens = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48 * 1024  # FlexAttention sequence length
    val_seq_len = 4 * 64 * 1024  # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

# Detect available device
if torch.cuda.is_available():
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    backend = "nccl"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "gloo"  # Gloo is the best option for CPU/MPS distributed training
else:
    device = torch.device("cpu")
    backend = "gloo"

# Distributed training setup
rank = int(os.environ.get("RANK", 0))  # Default to 0 if not set
world_size = int(os.environ.get("WORLD_SIZE", 1))  # Default to 1 if not set

# Adjust world_size assertion for Mac/CPU (since it's designed for 8xH100)
if device.type == "cuda":
    assert world_size == 8, "This code is designed for 8xH100 GPUs"

dist.init_process_group(backend=backend, world_size=world_size, rank=rank, timeout=torch.distributed.default_pg_timeout)
dist.barrier()
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
# Log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.__version__}")

if device.type == "cuda":
    print0(f"Compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  # avoid top-level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())

print0("=" * 100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(
    vocab_size=args.vocab_size,
    num_layers=12,
    num_heads=6,
    model_dim=768,
    max_seq_len=max(args.train_seq_len, args.val_seq_len)
).to(device)

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)  # Mac MPS does not fully support bfloat16

for param in model.parameters():
    dist.broadcast(param.to(device).detach(), 0)  # Ensure parameters are moved to the correct device before broadcasting

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

# Small adam epsilon by @YouJiacheng. Fixes world_size dependence discovered by @fernbear.bsky.social
optimizer1 = torch.optim.Adam(
    adam_params, betas=(0.8, 0.95), eps=1e-10, fused=(device.type == "cuda")  # Disable fused=True for MPS
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
    # Increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

# Compile model for optimized execution
if device.type == "cuda":
    model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(
    model=copy.deepcopy(model.state_dict()),
    optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]
)  # Save the initial state

for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device=device)

    with torch.autograd.set_detect_anomaly(True):  # Helps catch MPS-specific backward() errors
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()

    for param in model.parameters():
        if param.grad is not None:  # Ensure gradient exists before reducing
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

# Reload the original model and optimizer states
model.load_state_dict(initial_state["model"])
model.to(device)  # Ensure model is back on the correct device

for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)

del initial_state  # Free memory

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
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps
        del val_loader

        # Ensure `val_loss` exists before reducing
        if val_loss is not None:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms "
            f"step_avg:{training_time_ms/max(step, 1):.2f}ms",
            console=True
        )
        model.train()

        # Restart the clock
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(
                step=step,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers]
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # The last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    inputs, targets = inputs.to(device), targets.to(device)

    model(inputs, targets, get_window_size_blocks(step)).backward()

    for param in model.parameters():
        if param.grad is not None:  # Ensure gradients exist before reducing
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    # Set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)  # Momentum warmup for Muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # Step the optimizers
    for opt in optimizers:
        opt.step()

    # Null the gradients
    model.zero_grad(set_to_none=True)

    # Logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms "
        f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms",
        console=True
    )

# Memory logging (only on CUDA and MPS)
if device.type == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() // 1024 // 1024
    reserved_mem = torch.cuda.max_memory_reserved() // 1024 // 1024
    print0(f"peak memory allocated: {peak_mem} MiB reserved: {reserved_mem} MiB", console=True)
elif device.type == "mps":
    peak_mem = torch.mps.current_allocated_memory() // 1024 // 1024
    print0(f"peak memory allocated: {peak_mem} MiB", console=True)

dist.destroy_process_group()
