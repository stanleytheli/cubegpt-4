"""CubeGPT model architectures, ported unchanged from the CubeGPT-3 notebooks.

Parameter/buffer names must stay identical so existing checkpoints in models/
load with strict=True.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cube import N_CLASSES


class _ieee_fp32:
    """Context: fp32 matmuls in full IEEE precision (no TF32), restoring the setting after."""

    def __enter__(self):
        self.prev = torch.get_float32_matmul_precision()
        if self.prev != "highest":
            torch.set_float32_matmul_precision("highest")

    def __exit__(self, *exc):
        if self.prev != "highest":
            torch.set_float32_matmul_precision(self.prev)


class _ExactAttention(torch.autograd.Function):
    """softmax(q k^T * scale) v with every matmul in full fp32, forward and backward,
    regardless of the global TF32 setting. This matches the precision of PyTorch's
    fused fp32 attention kernel; TF32 here measurably perturbs logits of untrained
    models, and full fp32 costs nothing (tiny 20x16 matrices don't use tensor cores)."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        with _ieee_fp32():
            p = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            out = p @ v
        ctx.save_for_backward(q, k, v, p)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, p = ctx.saved_tensors
        with _ieee_fp32():
            dv = p.transpose(-2, -1) @ dout
            dp = dout @ v.transpose(-2, -1)
            ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True)) * ctx.scale
            dq = ds @ k
            dk = ds.transpose(-2, -1) @ q
        return dq, dk, dv, None


class MultiHeadAttention(nn.Module):
    """backend (same math, different kernels):
      "sdpa":     F.scaled_dot_product_attention. In fp32 this dispatches to the
                  memory-efficient kernel, which is built for long sequences and is slow
                  for 20 tokens x 16-dim heads at large batch (38% of step time at 16k).
      "explicit": softmax(QK^T / sqrt(d_head)) V with small batched matmuls, always in
                  full fp32 (see _ExactAttention), ~3x faster than sdpa at batch >= 4k.
      "auto":     explicit for >= EXPLICIT_MIN_BATCH sequences (faster on H100 from ~1k
                  sequences up), sdpa below (fewer kernel launches when launch-bound).
    """
    backend = "auto"
    EXPLICIT_MIN_BATCH = 512

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "Embedding dim is not divisible by nheads"
        self.bias = bias
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        query = self.q_proj(x).unflatten(-1, [self.n_heads, self.d_head]).transpose(1, 2)
        key = self.k_proj(x).unflatten(-1, [self.n_heads, self.d_head]).transpose(1, 2)
        value = self.v_proj(x).unflatten(-1, [self.n_heads, self.d_head]).transpose(1, 2)
        backend = self.backend
        if backend == "auto":
            backend = "explicit" if x.shape[0] >= self.EXPLICIT_MIN_BATCH else "sdpa"
        if backend == "explicit":
            attn_output = _ExactAttention.apply(query, key, value, 1.0 / math.sqrt(self.d_head))
        else:
            attn_output = F.scaled_dot_product_attention(query, key, value)
        return self.out_proj(attn_output.transpose(1, 2).flatten(-2))


class MLPBlock(nn.Module):
    def __init__(self, d_model, d_ffwd, bias=True, activation=nn.functional.gelu):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ffwd, bias)
        self.activation = activation
        self.down_proj = nn.Linear(d_ffwd, d_model, bias)

    def forward(self, x):
        return self.down_proj(self.activation(self.up_proj(x)))


class MLPBlockGated(nn.Module):
    def __init__(self, d_model, d_ffwd, bias=True):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ffwd, bias)
        self.gate_proj = nn.Linear(d_model, d_ffwd, bias)
        self.down_proj = nn.Linear(d_ffwd, d_model, bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _PostNormLayer(nn.Module):
    """x = norm1(x + attn(x)); x = norm2(x + mlp(x)). Subclasses set mlp_block.
    With pre_norm (set by the model): x = x + attn(norm1(x)); x = x + mlp(norm2(x))."""
    pre_norm = False

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.mha_block = MultiHeadAttention(d_model, n_heads, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(self, x):
        if self.pre_norm:
            x = x + self.mha_block(self.norm1(x))
            return x + self.mlp_block(self.norm2(x))
        x = self.norm1(x + self.mha_block(x))
        return self.norm2(x + self.mlp_block(x))


class CubeTransformerLayerAttn(nn.Module):
    """Attention-only layer: x = norm1(x + attn(x)). No FFN and no second norm."""

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.mha_block = MultiHeadAttention(d_model, n_heads, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, bias=bias)

    pre_norm = False

    def forward(self, x):
        if self.pre_norm:
            return x + self.mha_block(self.norm1(x))
        return self.norm1(x + self.mha_block(x))


class CubeTransformerLayerFFN(nn.Module):
    """FFN-only layer: x = norm2(x + mlp(x)). No attention."""

    def __init__(self, d_model, mlp_block, bias=True):
        super().__init__()
        self.mlp_block = mlp_block
        self.norm2 = nn.LayerNorm(d_model, bias=bias)

    pre_norm = False

    def forward(self, x):
        if self.pre_norm:
            return x + self.mlp_block(self.norm2(x))
        return self.norm2(x + self.mlp_block(x))


class CubeTransformerLayer(_PostNormLayer):
    def __init__(self, d_model, n_heads, d_ffwd, activation=nn.functional.gelu, bias=True):
        super().__init__(d_model, n_heads, bias)
        self.mlp_block = MLPBlock(d_model, d_ffwd, bias=bias, activation=activation)


class ScaledMLP(nn.Module):
    """A gated FFN whose output is multiplied by a learnable per-channel scale (LayerScale),
    initialised small so the FFN starts quiet and has to earn its volume."""

    def __init__(self, d_model, d_ffwd, scale_init, bias=True):
        super().__init__()
        self.mlp = MLPBlockGated(d_model, d_ffwd, bias=bias)
        self.scale = nn.Parameter(torch.full((d_model,), float(scale_init)))

    def forward(self, x):
        return self.mlp(x) * self.scale


class CubeTransformerLayerGated(_PostNormLayer):
    def __init__(self, d_model, n_heads, d_ffwd, bias=True, ffn_scale_init=0.0):
        super().__init__(d_model, n_heads, bias)
        self.mlp_block = (ScaledMLP(d_model, d_ffwd, ffn_scale_init, bias=bias) if ffn_scale_init
                          else MLPBlockGated(d_model, d_ffwd, bias=bias))


class MoEBlock(nn.Module):
    """DeepSeek-V3-style MoE FFN with auxiliary-loss-free load balancing.

    Routing per token: sigmoid affinity s_i; top-k over s_i + b_i (bias used
    only for selection); gate = s_i normalized over the selected experts.
    b_i is a buffer nudged by +/- gamma in update_expert_bias() after every
    optimizer step, based on the load seen in the last *training-mode* forward.

    Because sigmoid scores live in (0, 1), |b_i| >> 1 means routing is decided
    by the bias alone (see expert_stats()).

    Expert dispatch (`impl`):
      "grouped" (default): one host sync per layer (the per-expert token counts) and
          assignments sorted by expert. Up to PADDED_MAX_ASSIGNMENTS assignments, all
          experts run as one batched matmul over a padded (n_experts, max_count, d)
          buffer built with a single gather; beyond that (where launch overhead no
          longer matters but padding waste does), or when routing is so unbalanced
          that padding would more than double the work, one matmul per expert over
          contiguous slices. Results are un-permuted with a gather (no atomics).
      "loop": the original per-expert nonzero() loop (one host sync per expert),
          kept as the reference implementation for equivalence tests.
    Both compute the same function with the same parameters (the parameter layout,
    and therefore checkpoint compatibility, is unchanged).
    """

    impl = "grouped"
    # Above this many token->expert assignments per layer (batch x 20 tokens x
    # n_active), the padded path's wasted rows cost more than the kernel launches it
    # saves. Measured on H100 (8A1): padded wins at 41k assignments, sliced at 82k.
    PADDED_MAX_ASSIGNMENTS = 65536

    def __init__(self, d_model, d_ffwd, n_experts, n_active, bias=True, gated=True,
                 activation=nn.functional.gelu, bias_update_speed=1e-3, routed_scaling_factor=1.0,
                 router="sigmoid"):
        super().__init__()
        assert n_active <= n_experts
        assert router in ("sigmoid", "softmax"), router
        self.router = router
        self.n_experts = n_experts
        self.n_active = n_active
        self.bias_update_speed = bias_update_speed
        self.routed_scaling_factor = routed_scaling_factor
        self.gated = gated
        self.has_bias = bias
        self.activation = activation
        if gated:
            self.experts = nn.ModuleList([MLPBlockGated(d_model, d_ffwd, bias=bias) for _ in range(n_experts)])
        else:
            self.experts = nn.ModuleList([MLPBlock(d_model, d_ffwd, bias=bias, activation=activation)
                                          for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("tokens_per_expert", torch.zeros(n_experts))
        if router == "softmax":
            # Marker only: makes a softmax checkpoint fail strict loading into a sigmoid model (and
            # vice versa) instead of silently computing a different function with the same weights.
            self.register_buffer("router_softmax", torch.ones(()))
        # Router health from the last train-mode forward (not saved in checkpoints):
        #   score_mean    mean routing weight of the chosen expert(s)
        #   bias_override fraction of tokens sent somewhere other than their highest-scoring
        #                 expert(s) because of the balancing bias
        self.register_buffer("score_mean", torch.zeros(()), persistent=False)
        self.register_buffer("bias_override", torch.zeros(()), persistent=False)
        # Auxiliary router losses (set by the trainer when moe_aux_loss / moe_z_loss > 0): every
        # train-mode forward stores aux_balance (Switch load-balancing loss; 1.0 when perfectly
        # balanced) and aux_z (router z-loss), still attached to the graph.
        self.aux_enabled = False
        self.aux_balance = None
        self.aux_z = None

    def forward(self, x):
        B, T, d_model = x.shape
        x = x.reshape(-1, d_model)
        logits = self.gate(x)
        scores = logits.softmax(dim=-1) if self.router == "softmax" else logits.sigmoid()
        topk_idx = (scores + self.expert_bias).topk(self.n_active, dim=-1).indices
        topk_scores = scores.gather(-1, topk_idx)
        if self.n_active == 1:
            topk_weights = topk_scores
        else:
            topk_weights = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        if self.impl == "loop":
            out, load = self._dispatch_loop(x, topk_idx, topk_weights)
        elif self.impl == "grouped":
            out, load = self._dispatch_grouped(x, topk_idx, topk_weights)
        else:
            raise ValueError(f"unknown MoE impl {self.impl!r}")
        if self.training and self.aux_enabled:
            probs = scores if self.router == "softmax" else scores / (scores.sum(-1, keepdim=True) + 1e-20)
            frac = load.detach() / load.detach().sum().clamp_min(1.0)
            self.aux_balance = self.n_experts * (frac * probs.mean(0)).sum()
            self.aux_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        if self.training:
            self.tokens_per_expert.copy_(load.detach())
            with torch.no_grad():
                self.score_mean.copy_(topk_scores.mean())
                raw = scores.topk(self.n_active, dim=-1).indices
                if self.n_active == 1:
                    differs = raw[:, 0] != topk_idx[:, 0]
                else:
                    differs = (raw.sort(-1).values != topk_idx.sort(-1).values).any(-1)
                self.bias_override.copy_(differs.float().mean())
        return out.view(B, T, d_model)

    def _dispatch_loop(self, x, topk_idx, topk_weights):
        """Reference implementation (the original notebook code)."""
        out = torch.zeros_like(x)
        load = torch.zeros(self.n_experts, device=x.device)
        for e, expert in enumerate(self.experts):
            tok, slot = (topk_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            w = topk_weights[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, expert(x[tok]) * w)
            load[e] = tok.numel()
        return out, load

    def _dispatch_grouped(self, x, topk_idx, topk_weights):
        N, d = x.shape
        E, k = self.n_experts, self.n_active
        flat_e = topk_idx.reshape(-1)                              # (A,) expert of assignment a = (token a//k, slot a%k)
        A = flat_e.numel()
        counts = torch.bincount(flat_e, minlength=E)
        counts_host = counts.tolist()                              # the only host sync in this layer
        order = torch.argsort(flat_e, stable=True)                 # assignments grouped by expert
        cap = max(counts_host)
        # Only experts that received tokens may enter the autograd graph: like the
        # reference loop, an idle expert must get grad=None (not zeros), otherwise
        # Adam would keep applying momentum to it and advance its step count.
        active = [e for e, c in enumerate(counts_host) if c > 0]
        n_slots = len(active) * cap
        if A <= self.PADDED_MAX_ASSIGNMENTS and n_slots <= 2 * A:
            # Padded path: all experts in one batched matmul over (n_active_experts, cap, d).
            e_sorted = flat_e[order]
            starts = torch.cumsum(counts, 0) - counts
            pos = torch.arange(A, device=x.device) - starts[e_sorted]
            if len(active) < E:
                remap = [0] * E
                for i, e in enumerate(active):
                    remap[e] = i
                e_sorted = torch.tensor(remap, device=x.device)[e_sorted]
            slot_sorted = e_sorted * cap + pos                     # padded slot of each sorted assignment
            # padded slot -> source token. Padding slots read arbitrary (spread-out) real
            # tokens; their outputs are never gathered, so they get exactly zero gradient.
            src = torch.arange(n_slots, device=x.device) % N
            src[slot_sorted] = order // k
            ys = self._experts_batched(x.index_select(0, src).view(len(active), cap, d), active)
            slot = torch.empty_like(slot_sorted).scatter_(0, order, slot_sorted)   # slot of each assignment
            y = ys.view(n_slots, d).index_select(0, slot)
        else:
            # Sliced path (large batches, or unbalanced routing where padding would
            # more than double the work): one matmul per non-empty expert, no padding.
            chunks = x.index_select(0, order // k).split(counts_host)
            y_sorted = torch.cat([expert(c) if c.shape[0] else c for expert, c in zip(self.experts, chunks)])
            inverse = torch.empty_like(order).scatter_(0, order, torch.arange(A, device=x.device))
            y = y_sorted.index_select(0, inverse)
        out = (y * topk_weights.reshape(-1, 1)).view(N, k, d)      # back in (token, slot) order
        out = out.squeeze(1) if k == 1 else out.sum(dim=1)
        return out, counts.to(self.tokens_per_expert.dtype)

    def _experts_batched(self, xs, active):
        """Experts `active` on a padded (len(active), cap, d) buffer -> same shape.
        Weights are stacked on the fly so the per-expert parameter layout (and so
        checkpoint compatibility) is unchanged."""
        ex = [self.experts[e] for e in active]
        E = len(ex)
        if self.gated:
            # [up_e; gate_e] stacked -> one bmm for both input projections
            w_in = torch.stack([w for m in ex for w in (m.up_proj.weight, m.gate_proj.weight)])
            w_in = w_in.view(E, -1, w_in.shape[-1])                # (E, 2f, d)
            b_in = (torch.stack([b for m in ex for b in (m.up_proj.bias, m.gate_proj.bias)]).view(E, 1, -1)
                    if self.has_bias else None)
        else:
            w_in = torch.stack([m.up_proj.weight for m in ex])     # (E, f, d)
            b_in = torch.stack([m.up_proj.bias for m in ex]).unsqueeze(1) if self.has_bias else None
        h = torch.baddbmm(b_in, xs, w_in.transpose(1, 2)) if b_in is not None else torch.bmm(xs, w_in.transpose(1, 2))
        if self.gated:
            up, gate = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = self.activation(h)
        w_out = torch.stack([m.down_proj.weight for m in ex])      # (E, d, f)
        if self.has_bias:
            b_out = torch.stack([m.down_proj.bias for m in ex]).unsqueeze(1)
            return torch.baddbmm(b_out, h, w_out.transpose(1, 2))
        return torch.bmm(h, w_out.transpose(1, 2))

    @torch.no_grad()
    def update_expert_bias(self):
        """Call once per step, AFTER optimizer.step(), with the model in train mode."""
        if not self.training:
            # In eval mode tokens_per_expert is frozen, so every call would push each
            # bias the same way (+/-gamma per step, forever): this is how the published
            # 8A1 checkpoints ended up with |bias| ~ 300 and dead experts.
            raise RuntimeError("update_expert_bias() called with the model in eval mode; call model.train() first")
        load = self.tokens_per_expert
        err = load.mean() - load
        self.expert_bias += self.bias_update_speed * torch.sign(err)


class CubeTransformerLayerMoE(_PostNormLayer):
    def __init__(self, d_model, n_heads, d_ffwd, n_experts, n_active, activation=nn.functional.gelu, bias=True,
                 router="sigmoid"):
        super().__init__(d_model, n_heads, bias)
        self.mlp_block = MoEBlock(d_model, d_ffwd, n_experts, n_active, bias=bias, activation=activation,
                                  router=router)


class _CubeTransformerBase(nn.Module):
    def __init__(self, layers, d_model):
        super().__init__()
        self.piece_embed_table = nn.Embedding(20, d_model)
        self.placement_embed_table = nn.Embedding(48, d_model)
        self.stack = nn.Sequential(*layers)
        n_tokens = 20
        self.final_linear = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(n_tokens * d_model),
            nn.Linear(n_tokens * d_model, N_CLASSES),
        )

    def _use_pre_norm(self):
        """Switch every layer to pre-norm (x + f(norm(x))). The residual stream is then only
        normalized by the head's LayerNorm. Parameter names don't change, so a marker buffer makes
        a pre-norm checkpoint fail strict loading into a post-norm model and vice versa."""
        for layer in self.stack:
            layer.pre_norm = True
        self.register_buffer("pre_norm_marker", torch.ones(()))

    def forward(self, x):
        # x: (B, 20, 2) -> logits (B, 21)
        emb = self.piece_embed_table(x[:, :, 0]) + self.placement_embed_table(x[:, :, 1])
        return self.final_linear(self.stack(emb))


class CubeTransformer(_CubeTransformerBase):
    def __init__(self, d_model=256, n_heads=16, d_ffwd=512, n_layers=8, activation=nn.functional.gelu):
        super().__init__([CubeTransformerLayer(d_model, n_heads, d_ffwd, activation=activation)
                          for _ in range(n_layers)], d_model)


def _kind(ffn_layers, i):
    """ffn_layers: "" (every layer is attention + FFN) or one character per layer:
    '1' attention + FFN, '0' attention only, 'f' FFN only (e.g. "00000000ffff")."""
    return ffn_layers[i] if ffn_layers else "1"


class CubeTransformerGated(_CubeTransformerBase):
    def __init__(self, d_model=256, n_heads=16, d_ffwd=512, n_layers=8, ffn_layers="", ffn_scale_init=0.0,
                 pre_norm=False):
        def layer(kind):
            if kind == "0":
                return CubeTransformerLayerAttn(d_model, n_heads)
            full = CubeTransformerLayerGated(d_model, n_heads, d_ffwd, ffn_scale_init=ffn_scale_init)
            return full if kind == "1" else CubeTransformerLayerFFN(d_model, full.mlp_block)
        super().__init__([layer(_kind(ffn_layers, i)) for i in range(n_layers)], d_model)
        if pre_norm:
            self._use_pre_norm()


class CubeTransformerMoE(_CubeTransformerBase):
    def __init__(self, d_model=256, n_heads=16, d_ffwd=512, n_layers=8, n_experts=4, n_active=1, router="sigmoid",
                 first_dense_layers=0, ffn_layers="", pre_norm=False):
        # first_dense_layers: the first k layers get a plain gated FFN of width d_ffwd (same compute per
        # token as one expert) instead of an MoE block, as in DeepSeek-V3's dense first layers.
        def layer(i, kind):
            if kind == "0":
                return CubeTransformerLayerAttn(d_model, n_heads)
            full = (CubeTransformerLayerGated(d_model, n_heads, d_ffwd) if i < first_dense_layers else
                    CubeTransformerLayerMoE(d_model, n_heads, d_ffwd, n_experts, n_active, router=router))
            return full if kind == "1" else CubeTransformerLayerFFN(d_model, full.mlp_block)
        super().__init__([layer(i, _kind(ffn_layers, i)) for i in range(n_layers)], d_model)
        self.n_layers = n_layers
        if pre_norm:
            self._use_pre_norm()

    @torch.no_grad()
    def update_expert_bias(self):
        for m in self.modules():
            if isinstance(m, MoEBlock):
                m.update_expert_bias()


def build_model(m) -> nn.Module:
    """m: a cubegpt.config.ModelConfig (or anything with the same attributes)."""
    if m.arch == "moe":
        return CubeTransformerMoE(m.d_model, m.n_heads, m.d_ffwd, m.n_layers, m.n_experts, m.n_active,
                                  router=getattr(m, "router", "sigmoid"),
                                  first_dense_layers=getattr(m, "first_dense_layers", 0),
                                  ffn_layers=getattr(m, "ffn_layers", ""), pre_norm=getattr(m, "pre_norm", False))
    if m.arch == "gated":
        return CubeTransformerGated(m.d_model, m.n_heads, m.d_ffwd, m.n_layers,
                                    ffn_layers=getattr(m, "ffn_layers", ""),
                                    ffn_scale_init=getattr(m, "ffn_scale_init", 0.0),
                                    pre_norm=getattr(m, "pre_norm", False))
    if m.arch == "transformer":
        return CubeTransformer(m.d_model, m.n_heads, m.d_ffwd, m.n_layers)
    if m.arch == "pair":
        from .invariant import CubeTransformerPair   # here: invariant.py imports this module
        return CubeTransformerPair(m.d_model, m.n_heads, m.d_ffwd, m.n_layers, m.n_experts, m.n_active,
                                   d_pair=getattr(m, "d_pair", 64))
    raise ValueError(f"unknown arch {m.arch!r}")


def set_moe_impl(model, impl):
    """Switch every MoE block in a model between "grouped" (fast) and "loop" (reference)."""
    for m in model.modules():
        if isinstance(m, MoEBlock):
            m.impl = impl
    return model


CKPT_FORMAT = 1   # training checkpoints: {"format": 1, "model": state_dict, "optimizer": ..., ...}


def load_weights(path, device):
    """A bare state_dict (models/*.pth, final.pth) or a training checkpoint (ckpt/latest.pt)."""
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and obj.get("format") == CKPT_FORMAT and "model" in obj:
        return obj["model"]
    return obj


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_routing(model, X):
    """Actual per-layer token counts per expert for input batch X, measured with a
    fresh forward pass. Don't trust the stored tokens_per_expert buffer for this: it
    only updates in train mode, so in a loaded checkpoint (or after model.eval()) it
    is a stale snapshot. Restores the model's mode and buffers. None for dense models."""
    blocks = [m for m in model.modules() if isinstance(m, MoEBlock)]
    if not blocks:
        return None
    saved = [(b.training, b.tokens_per_expert.clone()) for b in blocks]
    for b in blocks:
        b.train()   # only makes the block record its load; MoE forward math is mode-independent
    model(X)
    counts = [b.tokens_per_expert.clone() for b in blocks]
    for b, (mode, buf) in zip(blocks, saved):
        b.train(mode)
        b.tokens_per_expert.copy_(buf)
    load = torch.stack(counts)
    bias = torch.stack([b.expert_bias for b in blocks])
    return {
        "n_tokens_per_layer": int(load[0].sum()),
        "dead_experts": int((load == 0).sum()),
        "dead_per_layer": [int((row == 0).sum()) for row in load],
        "share_per_layer": [[round(float(x), 4) for x in row / row.sum()] for row in load],
        "bias_spread": float((bias.max(dim=1).values - bias.min(dim=1).values).max()),
        "bias_absmax": float(bias.abs().max()),
    }


@torch.no_grad()
def expert_stats(model):
    """MoE routing health from the stored buffers: only meaningful during training
    (the load buffer reflects the last *train-mode* forward). For a loaded checkpoint
    use measure_routing(). None for dense models.

    bias_spread = max over layers of (max - min) expert bias. Top-k selection only
    sees bias *differences* within a layer, and the sign-based update makes all
    biases drift together over time (harmless), so the spread, not |bias|, is what
    matters: spread >> 1 (router scores live in (0, 1)) or dead_experts > 0 means
    load balancing has broken down.

    router_score_min = min over layers of the chosen expert's mean routing weight. For the
    sigmoid router, ~0.001 means it has saturated: scores barely differ between experts and
    the expert outputs are scaled by ~0.001 (seen at lr 5.6e-4, batch 1024; healthy runs sit
    around 0.03-0.3). The softmax router's weight can't fall below 1/n_experts for top-1.
    bias_override_max = max over layers of the fraction of tokens the balancing bias sent
    away from their highest-scoring expert: near 1 means the bias, not the token, decides."""
    blocks = [m for m in model.modules() if isinstance(m, MoEBlock)]
    if not blocks:
        return None
    bias = torch.stack([b.expert_bias for b in blocks])
    load = torch.stack([b.tokens_per_expert for b in blocks])
    mean = load.mean(dim=1, keepdim=True).clamp_min(1e-9)
    return {
        "bias_spread": float((bias.max(dim=1).values - bias.min(dim=1).values).max()),
        "bias_absmax": float(bias.abs().max()),
        "dead_experts": int((load == 0).sum()),
        "max_load_ratio": float((load / mean).max()),
        "router_score_min": float(min(b.score_mean for b in blocks)),
        "bias_override_max": float(max(b.bias_override for b in blocks)),
        **({"aux_balance": sum(float(b.aux_balance.detach()) for b in blocks) / len(blocks),
            "aux_z": sum(float(b.aux_z.detach()) for b in blocks) / len(blocks)}
           if all(b.aux_enabled and b.aux_balance is not None for b in blocks) else {}),
    }
