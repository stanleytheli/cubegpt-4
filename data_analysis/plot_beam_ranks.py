"""Where the solution that beam search ends up finding sits in the beam, step by step.
    .venv/bin/python data_analysis/plot_beam_ranks.py eval_results/<eval made with --save-beam>.json
Add --linear for a linear rank axis (files get a _linear suffix).
Writes images/readme/beam_rank.png (aligned on moves made) and beam_rank_remaining.png (aligned on
moves left). Rank 1 = the lowest predicted distance in the beam."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images" / "readme"
INK, INK2, GRID, SURFACE, GREEN = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff", "#007f5f"

LINEAR = "--linear" in sys.argv
sys.argv = [a for a in sys.argv if a != "--linear"]
result = json.loads(Path(sys.argv[1]).read_text())
solves = [s for s in result["solves"] if s["solved"] and s.get("ranks")]
width = solves[0]["width"]
SUFFIX = ("" if width == 4000 else f"_w{width}") + ("_linear" if LINEAR else "")   # 4,000 keeps the plain names
title_model = sys.argv[2] if len(sys.argv) > 2 else "CubeGPT-4"


def chart(xlabel):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"Rank of the final solution inside the beam", loc="left", fontsize=13, color=INK,
                 fontweight="bold", pad=24)
    ax.text(0, 1.03, f"{title_model}, beam width {width:,}, {len(solves)} scrambles", transform=ax.transAxes,
            fontsize=10, color=INK2)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10.5)
    ax.set_ylabel("rank in the beam (1 = best score)", color=INK2, fontsize=10.5)
    if LINEAR:
        ax.set_ylim(-width * 0.02, width * 1.05)
        ticks = [1] + list(range(width // 4, width + 1, width // 4))
    else:
        ax.set_yscale("log")
        ax.set_ylim(0.8, width * 1.6)
        ticks = [t for t in (1, 10, 100, 1000, 10000) if t < width / 1.5] + [width]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    return fig, ax


def draw(ax, columns):
    """columns: {x: [ranks]} -> median line with 25-75% and 5-95% bands."""
    xs = sorted(x for x, v in columns.items() if len(v) >= max(5, len(solves) // 20))
    q = np.array([np.percentile(columns[x], [5, 25, 50, 75, 95]) for x in xs])
    ax.fill_between(xs, q[:, 0], q[:, 4], color=GREEN, alpha=0.13, lw=0, label="5th to 95th percentile")
    ax.fill_between(xs, q[:, 1], q[:, 3], color=GREEN, alpha=0.30, lw=0, label="25th to 75th percentile")
    ax.plot(xs, q[:, 2], color=GREEN, lw=2.2, marker="o", ms=4.5, label="median")
    return xs, q


def finish(fig, ax, name, **kw):
    ax.legend(fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, framealpha=1, **kw)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote images/readme/{name}")


# aligned on moves made
fig, ax = chart("moves made")
cols, sizes = {}, {}
for s in solves:
    ax.plot(range(1, len(s["ranks"]) + 1), s["ranks"], color=GREEN, lw=0.5, alpha=0.10)
    for k, (r, n) in enumerate(zip(s["ranks"], s["beam_sizes"]), start=1):
        cols.setdefault(k, []).append(r)
        sizes.setdefault(k, []).append(n)
xs, q = draw(ax, cols)
ax.plot(xs, [np.median(sizes[x]) for x in xs], color=INK2, lw=1.2, ls=(0, (4, 3)), label="beam size")
ax.set_xticks(range(1, max(xs) + 1, 1))
finish(fig, ax, f"beam_rank{SUFFIX}.png", loc="center right")
print("moves made  :", {x: int(m) for x, m in zip(xs, q[:, 2])})

# aligned on moves left
fig, ax = chart("moves left until solved")
cols = {}
for s in solves:
    left = [s["n_moves"] - k for k in range(1, len(s["ranks"]) + 1)]
    ax.plot(left, s["ranks"], color=GREEN, lw=0.5, alpha=0.10)
    for l, r in zip(left, s["ranks"]):
        cols.setdefault(l, []).append(r)
xs, q = draw(ax, cols)
ax.set_xticks(range(1, max(xs) + 1, 1))
ax.invert_xaxis()
finish(fig, ax, f"beam_rank_remaining{SUFFIX}.png", loc="upper right")
print("moves left  :", {x: int(m) for x, m in zip(xs, q[:, 2])})
