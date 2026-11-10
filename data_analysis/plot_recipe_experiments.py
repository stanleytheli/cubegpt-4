"""Eval-loss curves for the 2026-11-09 experiments that improved on the baseline:
recipe changes on the 8 + 4 layout, bigger variants, and the 300M-sample runs.
    python data_analysis/plot_recipe_experiments.py        (reads runs/*/metrics.jsonl; run sync_runs.py first)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8985"
S = "wsd25m-bs1024-"
# entity -> (run, label, color): a model/recipe keeps its color in every panel
E_ = {
    "base":   (S + "lr5.6e-4", "CubeGPT-3.1 architecture (MoE ×8), old recipe", GRAY),
    "old":    (S + "lr5.6e-4-attn8-dense4", "8 + 4, old recipe (lr 5.6e-4)", "#2a78d6"),
    "pre":    (S + "lr8e-4-attn8-dense4-prenorm", "8 + 4, pre-norm", "#eb6834"),
    "b2":     (S + "lr8e-4-attn8-dense4-b2-0.95", "8 + 4, Adam beta2 0.95", "#1baf7a"),
    "adamw":  (S + "lr8e-4-attn8-dense4-adamw", "8 + 4, AdamW", "#eda100"),
    "preaw":  ("wsd25m-a8d4-prenorm-adamw-lr8e-4", "8 + 4, pre-norm + AdamW", "#e87ba4"),
    "muon":   ("wsd25m-a8d4-prenorm-muon0.005-lr8e-4", "8 + 4, pre-norm + Muon (lr 0.005)", "#008300"),
    "8411":   ("wsd25m-a8f4a1f1-prenorm-adamw-lr8e-4", "8 + 4 + 1 + 1, pre-norm + AdamW", "#6250d6"),
    "d384":   ("wsd25m-a8d4-d384-prenorm-adamw-lr5.3e-4", "8 + 4 at d384, pre-norm + AdamW (lr 5.3e-4)", "#e34948"),
}
LONG = {
    "base":  ("8a1-wsd-bs1024-t300m", "CubeGPT-3.1 (MoE ×8, 19.09M params)", GRAY),
    "old":   ("attn8-dense4-wsd-bs1024-t300m", "8 + 4, old recipe (3.30M)", "#2a78d6"),
    "preaw": ("a8d4-prenorm-adamw-t300m", "8 + 4, pre-norm + AdamW (3.30M)", "#e87ba4"),
    "8411":  ("a8f4a1f1-prenorm-adamw-t300m", "8 + 4 + 1 + 1, pre-norm + AdamW (3.82M)", "#6250d6"),
}


def curve(run):
    """(samples in millions, eval loss); records superseded by a divergence rollback are dropped."""
    pts = []
    f = ROOT / "runs" / run / "metrics.jsonl"
    if not f.exists():
        return [], []
    for line in open(f):
        r = json.loads(line)
        if "eval_loss" not in r:
            continue
        while pts and pts[-1][0] >= r["samples_seen"]:
            pts.pop()
        pts.append((r["samples_seen"], r["eval_loss"]))
    return [p[0] / 1e6 for p in pts], [p[1] for p in pts]


def panel(ax, entries, title, xlim, ylim, logx=False, total=None):
    for run, label, color in entries:
        x, y = curve(run)
        if not x:
            continue
        done = total is None or x[-1] >= total - 1
        tail = f"{y[-1]:.5f}" if done else f"{y[-1]:.4f} at {x[-1]:.0f}M, running"
        ax.plot(x, y, color=color, lw=2, label=f"{label}  —  {tail}", solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel("training samples (millions)", color=INK2, fontsize=10)
    ax.set_ylabel("eval loss", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID, title="final eval loss",
                    title_fontsize=9, labelcolor=INK)
    leg.get_title().set_color(INK2)


fig, axes = plt.subplots(3, 1, figsize=(11, 16), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
panel(axes[0], [E_[k] for k in ("base", "old", "pre", "b2", "adamw", "preaw", "muon")],
      "Recipe changes on the 8 + 4 layout (25.6M samples, lr 8e-4 unless noted)", (4, 25.8), (0.118, 0.145))
panel(axes[1], [E_[k] for k in ("old", "preaw", "8411", "d384")],
      "Bigger variants with pre-norm + AdamW (25.6M samples)", (4, 25.8), (0.118, 0.145))
panel(axes[2], list(LONG.values()), "300M-sample runs (log x-axis)", (10, 320), (0.104, 0.135), logx=True, total=307.2)
axes[2].set_xticks([10, 20, 50, 100, 200, 300]); axes[2].set_xticklabels(["10", "20", "50", "100", "200", "300"])
axes[2].minorticks_off()
fig.tight_layout(h_pad=2.5)
out = ROOT / "images" / "recipe_experiments_2026-11-09.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
