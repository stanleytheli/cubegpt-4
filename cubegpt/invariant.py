"""Rotation-invariant input encoding ("cubelet tokens + pair table") and the model that uses it.

Each cubelet's pose is the rotation g (one of 24) carrying it from its solved placement to its
current one. A whole-cube rotation m acts by (c, g) -> (m.c, m g m^-1). The model only sees
orbit ids under that action: a content class per token (24 classes) and a pair class per ordered
token pair (9,628 classes). It has no piece-identity or positional embedding, so it is exactly
invariant to the 24 rotations by construction. Mirror and inversion are NOT built in.

The numpy tables below are the reference implementation; `descriptor_table()` maps this repo's
(piece, placement) tokens onto (cubelet, pose) descriptors. tests/test_invariant.py checks all of it.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import cube
from .model import MoEBlock, MLPBlockGated

# ---------------------------------------------------------------------------- reference tables
ROTS = []
for _perm in itertools.permutations(range(3)):
    for _signs in itertools.product((1, -1), repeat=3):
        _M = np.zeros((3, 3), dtype=int)
        for _r in range(3):
            _M[_r, _perm[_r]] = _signs[_r]
        if round(np.linalg.det(_M)) == 1:
            ROTS.append(_M)
ROTS = np.array(ROTS)                                             # (24,3,3)
_key = lambda M: tuple(int(x) for x in np.asarray(M).flatten())
RID = {_key(M): i for i, M in enumerate(ROTS)}
MUL = np.array([[RID[_key(A @ B)] for B in ROTS] for A in ROTS])  # MUL[a,b] = index of ROTS[a] @ ROTS[b]
IDENT = RID[_key(np.eye(3, dtype=int))]
INV = np.array([int(np.where(MUL[a] == IDENT)[0][0]) for a in range(24)])

# the 20 movable cubelets, named by home position (8 corners, 12 edges)
HOMES = np.array([p for p in itertools.product((-1, 0, 1), repeat=3) if np.count_nonzero(p) >= 2])
CID = {tuple(int(x) for x in p): i for i, p in enumerate(HOMES)}
POS = np.einsum('gij,cj->cgi', ROTS, HOMES)                       # POS[c,g] = slot of cubelet c in pose g
SLOT = np.array([[CID[tuple(POS[c, g])] for g in range(24)] for c in range(20)])
PERM = np.array([[CID[_key(M @ h)] for h in HOMES] for M in ROTS])   # PERM[m,c]: where rotation m sends cubelet c

# descriptor d = c*24 + g.  Rotation m acts by (c, g) -> (PERM[m,c], m g m^-1)
_c, _g = np.repeat(np.arange(20), 24), np.tile(np.arange(24), 20)
ACT = np.array([PERM[m, _c] * 24 + MUL[MUL[m, _g], INV[m]] for m in range(24)])    # (24,480)
_, CONTENT = np.unique(ACT.min(axis=0), return_inverse=True)                       # (480,)   values 0..23
_pk = (ACT[:, :, None].astype(np.int64) * 480 + ACT[:, None, :]).min(axis=0)
_, _flat = np.unique(_pk, return_inverse=True)
PAIR = _flat.reshape(480, 480).astype(np.int16)                                    # (480,480) values 0..9627
N_CONTENT, N_PAIR = int(CONTENT.max()) + 1, int(PAIR.max()) + 1


def features(g):
    """g: (...,20) pose indices -> content ids (...,20), pair ids (...,20,20)."""
    d = np.arange(20) * 24 + g
    return CONTENT[d], PAIR[d[..., :, None], d[..., None, :]]


def _quarter(axis):
    a, b, c = axis, (axis + 1) % 3, (axis + 2) % 3
    M = np.zeros((3, 3), dtype=int); M[a, a] = 1; M[b, c] = -1; M[c, b] = 1
    return M


TURN = np.array([[RID[_key(np.linalg.matrix_power(_quarter(a), k))] for k in range(4)] for a in range(3)])
C20 = np.arange(20)


def turn(g, axis, side, k):
    """Turn the layer {slot[axis] == side} by k quarter turns. Works on (...,20) arrays."""
    in_layer = POS[C20, g][..., axis] == side
    return np.where(in_layer, MUL[TURN[axis, k], g], g)


MOVES18 = [(a, s, k) for a in range(3) for s in (1, -1) for k in (1, 2, 3)]
SOLVED = np.full(20, IDENT)


def rotate(g, m):                       # the 24 exact symmetries of the model; g: (20,)
    out = np.empty(20, dtype=int); out[PERM[m]] = MUL[MUL[m, g], INV[m]]; return out


_R = np.diag([-1, 1, 1])
_RPERM = np.array([CID[_key(_R @ h)] for h in HOMES])
_RCONJ = np.array([RID[_key(_R @ M @ _R)] for M in ROTS])


def mirror(g):                          # symmetry of the distance, not of the model
    out = np.empty(20, dtype=int); out[_RPERM] = _RCONJ[g]; return out


def inverse(g):                         # symmetry of the distance, not of the model
    out = np.empty(20, dtype=int); out[SLOT[C20, g]] = INV[g]; return out


# ---------------------------------------------------------------------------- repo tokens -> descriptors
def descriptor_table():
    """(20, 48) int array: DESC[piece, placement] = cubelet * 24 + pose for this repo's tokens
    (cube.string_to_tokens), or -1 for placements that piece can never have.

    Geometry is read off cube.py's own tables: a piece's home is the sum of its colours' face
    normals, and it is in pose g when each sticker with home facing n now faces ROTS[g] @ n."""
    orien_idx_to_token = {0: 0, 3: 1, 4: 2, 1: 0, 2: 1, 5: 2}
    desc = np.full((20, 48), -1, dtype=np.int64)
    for pos, facelets in cube.pos_face_idx.items():
        now = [np.array(cube.FACE_NORMAL[cube.face_of_facelet(f)]) for f in facelets]
        for cs, piece in cube.color_str_tensor_idx.items():
            if len(cs) != len(facelets):
                continue
            home_n = [np.array(cube.FACE_NORMAL[cube.SOLVED_COLOR.index(c)]) for c in cs]
            match = [g for g in range(24) if all((ROTS[g] @ h == n).all() for h, n in zip(home_n, now))]
            if not match:
                continue                                  # mirror-image arrangement: never occurs
            (g,) = match
            home = tuple(int(v) for v in sum(home_n))
            assert tuple(ROTS[g] @ np.array(home)) == tuple(np.array(pos) - 1)
            orient = cube.color_str_orientation[cs]
            if len(cs) == 2:
                placement = cube.pos_idx[pos] * 2 + orient
            else:
                placement = cube.pos_idx[pos] * 3 + orien_idx_to_token[orient] + 24
            d = CID[home] * 24 + g
            assert desc[piece, placement] in (-1, d), "ambiguous token -> pose mapping"
            desc[piece, placement] = d
    assert ((desc >= 0).sum(axis=1) == 24).all(), "every piece must have exactly 24 placements"
    assert len(set(desc[desc >= 0].tolist())) == 480
    return desc


def tokens_to_poses(tokens, desc=None):
    """Repo tokens (...,20,2) (numpy) -> pose arrays (...,20) in reference cubelet order."""
    desc = descriptor_table() if desc is None else desc
    d = desc[tokens[..., 0], tokens[..., 1]]
    assert (d >= 0).all()
    out = np.empty_like(d)
    np.put_along_axis(out, d // 24, d % 24, axis=-1)
    return out


# ---------------------------------------------------------------------------- model
class PairAttention(nn.Module):
    """Attention whose logits and values also depend on the pair class of (i, j):
        logit_ij = q_i . (k_j + rk_ij) / sqrt(d_head) + bias_ij[head]
        out_i    = sum_j softmax_j(logit_ij) (v_j + rv_ij)
    bias (one scalar per head), rk and rv (d_head each, shared across heads) are linear maps of
    the shared pair embedding. They are computed once on the 9,628-row table and then gathered."""

    def __init__(self, d_model, n_heads, d_pair, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.pair_proj = nn.Linear(d_pair, n_heads + 2 * self.d_head, bias=False)

    def forward(self, x, pair_table, pair_ids):
        H, dh = self.n_heads, self.d_head
        r = self.pair_proj(pair_table)[pair_ids]                              # (B, T, T, H + 2 dh)
        bias, rk, rv = r[..., :H], r[..., H:H + dh], r[..., H + dh:]
        q = self.q_proj(x).unflatten(-1, [H, dh])                             # (B, T, H, dh)
        k = self.k_proj(x).unflatten(-1, [H, dh]).transpose(1, 2)             # (B, H, T, dh)
        v = self.v_proj(x).unflatten(-1, [H, dh]).transpose(1, 2)
        logits = q.transpose(1, 2) @ k.transpose(-2, -1)                      # (B, H, T, T)
        logits = logits + (q @ rk.transpose(-2, -1)).transpose(1, 2)          # q_i . rk_ij
        logits = logits / math.sqrt(dh) + bias.permute(0, 3, 1, 2)
        p = logits.softmax(dim=-1)
        out = p @ v + (p.transpose(1, 2) @ rv).transpose(1, 2)                # (B, H, T, dh)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class PairLayer(nn.Module):
    """Post-norm layer, as in the baseline: x = norm1(x + attn); x = norm2(x + ffn)."""

    def __init__(self, d_model, n_heads, d_pair, mlp_block):
        super().__init__()
        self.mha_block = PairAttention(d_model, n_heads, d_pair)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp_block = mlp_block
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pair_table, pair_ids):
        x = self.norm1(x + self.mha_block(x, pair_table, pair_ids))
        return self.norm2(x + self.mlp_block(x))


class CubeTransformerPair(nn.Module):
    """Takes the same (B, 20, 2) tokens as the other models and returns (B, 21) ordinal logits.
    Readout: a per-token MLP layer, mean over tokens, then an MLP (independent of token order)."""

    def __init__(self, d_model=256, n_heads=16, d_ffwd=341, n_layers=8, n_experts=8, n_active=1, d_pair=64,
                 d_readout=1024, moe=True):
        super().__init__()
        assert n_layers >= 2, "needs at least 2 attention layers to tell all states apart"
        for name, arr in (("DESC", descriptor_table()), ("CONTENT", CONTENT), ("PAIR", PAIR)):
            self.register_buffer(name, torch.as_tensor(np.asarray(arr, dtype=np.int64)), persistent=False)
        self.content_embed = nn.Embedding(N_CONTENT, d_model)
        self.pair_embed = nn.Embedding(N_PAIR, d_pair)
        self.stack = nn.ModuleList([
            PairLayer(d_model, n_heads, d_pair,
                      MoEBlock(d_model, d_ffwd, n_experts, n_active) if moe else MLPBlockGated(d_model, d_ffwd))
            for _ in range(n_layers)])
        self.readout_in = nn.Linear(d_model, d_readout)
        self.readout = nn.Sequential(nn.LayerNorm(d_readout), nn.Linear(d_readout, d_readout), nn.GELU(),
                                     nn.Linear(d_readout, cube.N_CLASSES))

    def encode(self, x):
        d = self.DESC[x[..., 0], x[..., 1]]                                   # (B, 20) descriptors
        return self.CONTENT[d], self.PAIR[d.unsqueeze(-1), d.unsqueeze(-2)]

    def forward(self, x):
        content_ids, pair_ids = self.encode(x)
        h = self.content_embed(content_ids)
        table = self.pair_embed.weight
        for layer in self.stack:
            h = layer(h, table, pair_ids)
        return self.readout(F.gelu(self.readout_in(h)).mean(dim=1))

    @torch.no_grad()
    def update_expert_bias(self):
        for m in self.modules():
            if isinstance(m, MoEBlock):
                m.update_expert_bias()
