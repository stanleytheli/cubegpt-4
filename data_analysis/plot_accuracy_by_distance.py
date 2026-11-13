"""A metric of each model as a function of the moves left until solved, on solution-path states.
    .venv/bin/python data_analysis/plot_accuracy_by_distance.py [rounded_acc|within1..within4|bce|rmse]
(after solution_path_metrics.py). Writes images/readme/accuracy_by_distance.png for accuracy, and
within1_by_distance.png, bce_by_distance.png, ... for the others."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
COLORS = {"CubeGPT-3": "#2a6fe8", "CubeGPT-3.1": "#ff7a1a", "CubeGPT-4": "#007f5f"}
MARKERS = {"CubeGPT-3": "o", "CubeGPT-3.1": "s", "CubeGPT-4": "D"}
MIN_STATES = 50

metric = sys.argv[1] if len(sys.argv) > 1 else "rounded_acc"
LOSS = metric in ("bce", "rmse")
what = {"rounded_acc": "Accuracy", "bce": "BCE loss", "rmse": "RMS error"}.get(metric, f"Within-{metric[-1]} accuracy")
ylabel = {"rounded_acc": "accuracy", "bce": "mean BCE", "rmse": "root mean squared error (moves)"}.get(
    metric, f"share of predictions within {metric[-1]}")
data = json.loads((ROOT / "eval_results" / "solution_path_metrics.json").read_text())

fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_title(f"{what} by distance from solved", loc="left", fontsize=13, color=INK, fontweight="bold", pad=24)
ax.text(0, 1.03, f"states on the best-known solution paths of {data['cubes']} scrambles", transform=ax.transAxes,
        fontsize=10, color=INK2)
for name, m in data["models"].items():
    value = (lambda v: v["mse"] ** 0.5) if metric == "rmse" else (lambda v: v[metric])
    pts = [(int(k), value(v)) for k, v in m["by_label"].items() if v["n"] >= MIN_STATES]
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, color=COLORS[name], lw=2, marker=MARKERS[name], ms=5.5, mec=SURFACE, mew=1, label=name)
    print(name, {x: round(y, 3) if LOSS else round(100 * y) for x, y in zip(xs, ys)})
ax.set_xticks(range(1, max(xs) + 1))
ax.invert_xaxis()
if LOSS:
    ax.set_ylim(bottom=0)
else:
    ax.set_ylim(-0.03, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("moves left until solved", color=INK2, fontsize=10.5)
ax.set_ylabel(ylabel, color=INK2, fontsize=10.5)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=INK2, length=0, labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, framealpha=1, loc="upper right" if LOSS else "lower right")
fig.tight_layout()
name = ("accuracy" if metric == "rounded_acc" else metric) + "_by_distance.png"
fig.savefig(ROOT / "images" / "readme" / name, dpi=200, facecolor=SURFACE)
print(f"wrote images/readme/{name}")
