"""What happens to the wide-beam solutions inside a narrower search.
    .venv/bin/python data_analysis/plot_wide_in_narrow.py eval_results/<name>.tracked.json [--linear]
(tracked file from track_wide_solutions.py, then pruned_ranks.py for where pruned solutions ranked).
Add --linear for a linear rank axis inside the beam, compressed above it (file gets a _linear
suffix), or --full-linear for one linear axis all the way up (_full_linear).
Writes images/readme/wide_solutions_in_narrow_beam.png."""
import json
import sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE, GREEN, PINK = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff", "#007f5f", "#d6338a"

LINEAR = "--linear" in sys.argv
FULL = "--full-linear" in sys.argv          # one linear axis all the way up (file gets _full_linear)
sys.argv = [a for a in sys.argv if a not in ("--linear", "--full-linear")]
data = json.loads(Path(sys.argv[1]).read_text())
cubes, narrow, wide = data["cubes"], data["narrow_width"], data["wide_width"]
model = sys.argv[2] if len(sys.argv) > 2 else "CubeGPT-4"
top = max([c.get("pruned_candidates", 0) for c in cubes] + [narrow * 15])     # most candidates in a step

fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_title(f"Where the width-{wide:,} solutions go in a width-{narrow:,} search", loc="left", fontsize=13,
             color=INK, fontweight="bold", pad=24)
ax.text(0, 1.03, f"{model}, {len(cubes)} scrambles", transform=ax.transAxes, fontsize=10, color=INK2)

if FULL:
    worst = max(c.get("pruned_rank", narrow) for c in cubes)
    ax.set_ylim(-worst * 0.02, worst * 1.12)
    step = 10000 if narrow <= 4000 else narrow
    ticks = [1, narrow] + [t for t in range(step, int(worst * 1.12), step) if t > narrow * 1.5]
elif LINEAR:                     # linear inside the beam, logarithmic above it (ranks go up to ~60,000)
    span = narrow * 0.45         # axis height given to everything above the beam
    k = span / np.log(top * 1.3 / narrow)
    fwd = lambda y: np.where(np.asarray(y) <= narrow, np.asarray(y, float),  # noqa: E731
                             narrow + k * np.log(np.maximum(np.asarray(y, float), narrow) / narrow))
    inv = lambda v: np.where(np.asarray(v) <= narrow, np.asarray(v, float),  # noqa: E731
                             narrow * np.exp((np.asarray(v, float) - narrow) / k))
    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(-narrow * 0.02, top * 1.3)
    above = [10000, 20000, 50000] if narrow <= 4000 else [narrow * 2 ** k for k in range(1, 6)]
    ticks = [1] + list(range(narrow // 4, narrow + 1, narrow // 4)) + [t for t in above if t <= top * 1.3]
else:
    ax.set_yscale("log", base=2)                 # ticks at the beam width times powers of two
    ax.set_ylim(0.8, top * 1.5)
    ticks = [narrow * 2.0 ** k for k in range(-18, 5) if 0.9 <= narrow * 2.0 ** k <= top * 1.5]
ax.axhspan(narrow, ax.get_ylim()[1], color=PINK, alpha=0.05, lw=0, zorder=0)

kept = pruned = 0
pruned_at = Counter()
for c in cubes:
    ranks = c["wide_ranks"]
    gone = next((i for i, r in enumerate(ranks) if r is None), None)
    if gone is None:
        kept += 1
        ax.plot(range(1, len(ranks) + 1), ranks, color=GREEN, lw=0.9, alpha=0.35, zorder=3)
    else:
        pruned += 1
        pruned_at[gone + 1] += 1
        end = max(c["pruned_rank"], narrow + 1)          # scored locally: can land a hair inside the cutoff
        ax.plot(range(1, gone + 2), ranks[:gone] + [end], color=PINK, lw=0.9, alpha=0.35, zorder=2)
        ax.scatter([gone + 1], [end], s=16, color=PINK, zorder=4, edgecolor=SURFACE, lw=0.6)
for step, count in sorted(pruned_at.items()):
    ax.annotate(str(count), (step, 1), xycoords=("data", "axes fraction"), xytext=(0, -4), textcoords="offset points",
                ha="center", va="top", fontsize=9.5, color=INK)
ax.axhline(narrow, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.plot([], [], color=GREEN, lw=2, label=f"kept to the end: the same solution is found ({kept})")
ax.plot([], [], color=PINK, lw=2, marker="o", ms=4.5, label=f"pruned: dot = its rank when dropped ({pruned})")

ax.set_yticks(ticks)
short = lambda t: f"{t / 1000:g}k" if t >= 1000 else f"{round(t):,}"  # noqa: E731
ax.set_yticklabels([f"{t:,}" for t in ticks] if (LINEAR or FULL) else [short(t) for t in ticks])
ax.minorticks_off()
ax.set_xticks(range(1, max(len(c["wide_ranks"]) for c in cubes) + 1))
ax.set_xlabel("moves made", color=INK2, fontsize=10.5)
ax.set_ylabel("rank among all candidates (1 = best score)", color=INK2, fontsize=10.5)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=INK2, length=0, labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, framealpha=1,
          loc="upper right" if FULL else "center right", bbox_to_anchor=(1, 0.9) if FULL else None)
fig.tight_layout()
out = ROOT / "images" / "readme" / (f"wide_solutions_in_narrow_beam{'' if narrow == 4000 else f'_w{narrow}'}"
                                      f"{'_full_linear' if FULL else '_linear' if LINEAR else ''}.png")
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(f"wrote {out.relative_to(ROOT)}: kept {kept}, pruned {pruned}, pruned at {dict(sorted(pruned_at.items()))}")
