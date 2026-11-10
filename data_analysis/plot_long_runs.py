"""300M-sample runs: eval loss against samples and against training time.
    python data_analysis/plot_long_runs.py        (reads runs/*/metrics.jsonl; run sync_runs.py first)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8985"
RUNS = [
    ("8a1-wsd-bs1024-t300m", "CubeGPT-3.1 (MoE ×8, 19.09M params)", GRAY),
    ("attn8-dense4-wsd-bs1024-t300m", "8 + 4, old recipe (3.30M)", "#2a78d6"),
    ("a8d4-prenorm-adamw-t300m", "8 + 4, new recipe, lr 8e-4 (3.30M)", "#e87ba4"),
    ("a8f4a1f1-prenorm-adamw-t300m", "8 + 4 + 1 + 1, new recipe, lr 8e-4 (3.82M)", "#6250d6"),
    ("a8f4a1f1-prenorm-adamw-lr4e-4-t300m", "8 + 4 + 1 + 1, new recipe, lr 4e-4 (3.82M)", "#008300"),
]
TOTAL = 307.2


def curve(run):
    """[(samples M, train minutes, eval loss)]; records superseded by a divergence rollback are dropped
    (their training time still counts: train_seconds is cumulative)."""
    pts = []
    for line in open(ROOT / "runs" / run / "metrics.jsonl"):
        r = json.loads(line)
        if "eval_loss" not in r:
            continue
        while pts and pts[-1][0] >= r["samples_seen"] / 1e6:
            pts.pop()
        pts.append((r["samples_seen"] / 1e6, r["train_seconds"] / 60, r["eval_loss"]))
    return pts


fig, axes = plt.subplots(2, 1, figsize=(11, 12.5), facecolor=SURFACE)
for ax, col, xlabel, title in ((axes[0], 0, "training samples (millions, log scale)", "Eval loss per sample"),
                               (axes[1], 1, "training time on one H100 (minutes)", "Eval loss per minute of training")):
    ax.set_facecolor(SURFACE)
    for run, label, color in RUNS:
        p = curve(run)
        x, y = [q[col] for q in p], [q[2] for q in p]
        done = p[-1][0] >= TOTAL - 1
        tail = f"{y[-1]:.5f}" if done else f"stopped at {p[-1][0]:.0f}M ({y[-1]:.4f})"
        ax.plot(x, y, color=color, lw=2.4 if "4e-4" in label else 1.8, label=f"{label}  —  {tail}", solid_capstyle="round",
                zorder=3 if "4e-4" in label else 2)
        ax.plot(x[-1], y[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=4)
    ax.set_ylim(0.102, 0.132)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel("eval loss", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                    title="final eval loss at 307M samples", title_fontsize=9, labelcolor=INK)
    leg.get_title().set_color(INK2)
axes[0].set_xscale("log"); axes[0].set_xlim(10, 320)
axes[0].set_xticks([10, 20, 50, 100, 200, 300]); axes[0].set_xticklabels(["10", "20", "50", "100", "200", "300"])
axes[0].minorticks_off()
axes[1].set_xlim(0, 235)
fig.tight_layout(h_pad=2.5)
out = ROOT / "images" / "long_runs_2026-11-10.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
