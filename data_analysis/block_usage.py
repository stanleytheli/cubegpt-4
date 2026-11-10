"""How much each block of CubeGPT-3.1 and CubeGPT-4 matters, drawn against how many parameters it holds.
    .venv/bin/python data_analysis/block_usage.py            (CPU, a few minutes; --plot-only reuses the saved numbers)
For every attention / FFN block: replace its output with its average over the batch (mean-ablation) and measure the
loss increase. Writes eval_results/block_usage.json and images/readme/block_usage.png."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data_analysis"))
MODELS = [("CubeGPT-3.1", "models/cubegpt-3-1-t300m.pth", "configs/8a1_wsd_bs1024_t300m.toml"),
          ("CubeGPT-4", "models/cubegpt-4-t1b.pth", "configs/a8f4a1f1_branch230m_to_1b.toml")]
SAVED = ROOT / "eval_results" / "block_usage.json"


def measure():
    import torch
    import torch.nn.functional as F
    from depth_breakdown import positions
    from cubegpt.config import load_model_config
    from cubegpt.cube import CubeTables, to_ordinal_targets
    from cubegpt.model import build_model, load_weights
    T = CubeTables("cpu")
    Xs = positions(T, 300, seed=99)
    X = torch.cat(Xs); y = to_ordinal_targets(torch.cat([torch.full((len(x),), d) for d, x in enumerate(Xs)]))
    out = {}
    for name, weights, cfg in MODELS:
        m = build_model(load_model_config(str(ROOT / cfg))); m.load_state_dict(load_weights(str(ROOT / weights), "cpu")); m.eval()
        blocks = [(i, k, getattr(l, k)) for i, l in enumerate(m.stack) for k in ("mha_block", "mlp_block") if hasattr(l, k)]
        means = {}
        hs = [b.register_forward_hook(lambda mod, inp, o, key=(i, k): means.__setitem__(key, o.mean(0, keepdim=True))) for i, k, b in blocks]

        @torch.no_grad()
        def bce():
            return float(F.binary_cross_entropy_with_logits(m(X), y))
        base = bce()
        for h in hs:
            h.remove()
        rows = []
        for i, k, b in blocks:
            h = b.register_forward_hook(lambda mod, inp, o, key=(i, k): means[key].expand_as(o)); r = bce(); h.remove()
            rows.append({"layer": i, "kind": "attention" if k == "mha_block" else "ffn", "params": sum(p.numel() for p in b.parameters()),
                         "mean_ablate_pct": 100 * (r / base - 1)})
            print(name, rows[-1], flush=True)
        out[name] = {"bce": base, "total_params": sum(p.numel() for p in m.parameters()), "blocks": rows}
    SAVED.write_text(json.dumps(out, indent=1))
    return out


data = json.loads(SAVED.read_text()) if "--plot-only" in sys.argv and SAVED.exists() else measure()

# ---- figure: every block drawn as a glyph whose AREA is its parameter count (one scale for both models) -------
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
INK, INK2, SURFACE, RAIL = "#0b0b0b", "#52514e", "#ffffff", "#c9c8c3"
# red = removing the block costs almost nothing (wasted parameters), neutral near +15%, blue-purple = critical
cmap = LinearSegmentedColormap.from_list("use", [(0.0, "#d21f3c"), (0.17, "#ee7a6a"), (0.277, "#ece7e3"), (0.45, "#9c8aea"),
                                                 (0.604, "#6f55d8"), (0.80, "#43259f"), (1.0, "#1f0f5c")])
norm = LogNorm(vmin=3, vmax=1000, clip=True)
ATT = min(b["params"] for d in data.values() for b in d["blocks"] if b["kind"] == "attention")
GAP, R = 0.34, 0.5                       # an attention block is a circle of diameter 1


def shade(v):
    """(fill, edge, glyph line colour) for a loss increase of v percent."""
    r, g, b, _ = cmap(float(norm(max(v, 3))))
    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.52
    return (r, g, b), (0.62 * r, 0.62 * g, 0.62 * b), ("#ffffff" if dark else "#4b4a57")


def attention(ax, x, y, v):
    """Circle with a small all-to-all mesh: every token looks at every other token."""
    fill, edge, line = shade(v)
    ax.add_patch(Circle((x + R, y), R, facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=3))
    xs = [x + R + dx for dx in (-0.26, 0, 0.26)]
    for a in xs:
        for b in xs:
            ax.plot([a, b], [y + 0.24, y - 0.24], color=line, lw=0.55, alpha=0.85, zorder=4, solid_capstyle="round")
    for a in xs:
        for yy in (y + 0.24, y - 0.24):
            ax.add_patch(Circle((a, yy), 0.055, facecolor=line, edgecolor="none", zorder=5))
    return 2 * R


def ffn(ax, x, y, v, params, experts):
    """Rounded box holding one diamond (widen, then narrow) per expert. A plain FFN is a square whose area matches an
    attention circle with the same parameter count; a mixture-of-experts FFN keeps its to-scale height but is drawn at
    about half its to-scale width, with its experts in a 2-wide grid. Returns (width, height)."""
    hgt = (np.pi * R * R * params / ATT) ** 0.5
    wid = hgt if experts == 1 else 0.52 * hgt
    fill, edge, line = shade(v)
    pad = 0.08 * min(wid, 1.2)
    ax.add_patch(FancyBboxPatch((x + pad, y - hgt / 2 + pad), wid - 2 * pad, hgt - 2 * pad, boxstyle=f"round,pad={pad},rounding_size={1.6 * pad}",
                                facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=3))
    cols, rows = (1, 1) if experts == 1 else (2, experts // 2)
    iw, ih = wid - 3 * pad, hgt - 3 * pad
    cw, ch = iw / cols, ih / rows
    h = 0.27 * wid if experts == 1 else 0.36 * min(cw, ch)
    for r in range(rows):
        for q in range(cols):
            c, yy = x + wid / 2 - iw / 2 + cw * (q + 0.5), y + ih / 2 - ch * (r + 0.5)
            ax.plot([c - h, c, c + h, c, c - h], [yy, yy + h, yy, yy - h, yy], color=line, lw=0.9, zorder=4, solid_joinstyle="round")
            ax.plot([c, c], [yy + h, yy - h], color=line, lw=0.6, alpha=0.8, zorder=4)
    return wid, hgt


FIG_W, FIG_H = 10.5, 4.75
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
ax = fig.add_axes([0.012, 0.19, 0.976, 0.72]); ax.set_aspect("equal"); ax.axis("off")
ROW_Y = {"CubeGPT-3.1": 0.0, "CubeGPT-4": -4.3}
xmax = 0
for name, d in data.items():
    y, x, tops = ROW_Y[name], 0.0, []
    gap = GAP if "3.1" in name else 0.52
    for b in d["blocks"]:
        v = b["mean_ablate_pct"]
        moe = b["kind"] == "ffn" and b["params"] > 3 * ATT
        w, h = (2 * R, 2 * R) if b["kind"] == "attention" else ffn(ax, x, y, v, b["params"], 8 if moe else 1)
        if b["kind"] == "attention":
            attention(ax, x, y, v)
        ax.text(x + w / 2, y - h / 2 - 0.2, f"+{v:.0f}%" if v >= 0.5 else "0%", ha="center", va="top", fontsize=10.5 if moe else 8,
                fontweight="bold" if moe else "normal", color=INK, zorder=6)
        tops.append(h / 2); x += w + gap
    end = x - gap
    ax.plot([-0.5, end + 0.75], [y, y], color=RAIL, lw=2.2, zorder=1, solid_capstyle="round")
    ax.annotate("", (end + 0.95, y), (end + 0.55, y), arrowprops=dict(arrowstyle="-|>", color=RAIL, lw=2.2), zorder=1)
    ax.text(-0.5, y + max(tops) + 0.4, name, fontsize=13, fontweight="bold", color=INK, va="bottom")
    ax.text(-0.5 + (3.95 if "3.1" in name else 3.3), y + max(tops) + 0.42, f"{d['total_params'] / 1e6:.2f}M parameters", fontsize=10, color=INK2, va="bottom")
    xmax = max(xmax, end + 1.0)
ax.set_xlim(-0.7, xmax); ax.set_ylim(ROW_Y["CubeGPT-4"] - 1.2, 2.3)
fig.suptitle("Block usage", x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
# legend: the three glyphs, then the shade scale
unit = 0.976 * FIG_W / (xmax + 0.7)           # inches per data unit in the main axes, so legend glyphs match
lax = fig.add_axes([0.012, 0.015, 0.52, 0.15]); lax.set_aspect("equal"); lax.axis("off")
lax.set_xlim(0, 0.52 * FIG_W / unit); lax.set_ylim(-0.15 * FIG_H / unit / 2, 0.15 * FIG_H / unit / 2)
attention(lax, 0.0, 0, 100); lax.text(1.2, 0, "attention", va="center", fontsize=9.5, color=INK2)
ffn(lax, 3.6, 0, 100, ATT, 1); lax.text(4.7, 0, "FFN", va="center", fontsize=9.5, color=INK2)
ffn(lax, 6.3, 0, 100, ATT * 2.9, 8); lax.text(7.4, 0, "FFN with 8 experts (MoE)", va="center", fontsize=9.5, color=INK2)
cax = fig.add_axes([0.70, 0.085, 0.27, 0.03])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[3, 10, 30, 100, 300, 1000])
cb.ax.set_xticklabels(["≤3%", "10%", "30%", "100%", "300%", "≥1000%"]); cb.ax.tick_params(labelsize=8, colors=INK2, length=0); cb.outline.set_visible(False)
cb.ax.minorticks_off()
fig.text(0.835, 0.135, "loss increase when the block is ablated", ha="center", va="bottom", fontsize=9, color=INK2)
out = ROOT / "images" / "readme" / "block_usage.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(out)
