"""CubeGPT-4 (attn x8 + ffn x4 + attn + ffn, 3.82M params, 1.15B samples) against CubeGPT-3.1.
    python data_analysis/plot_cubegpt4.py [--log]    (--log: log-log loss panels; reads runs/*/metrics.jsonl and eval_results/*.json)"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8985"
GREEN, BLUE = "#008300", "#2a78d6"
PARENT, BRANCH, V31 = "a8f4a1f1-prenorm-adamw-lr4e-4-t300m", "a8f4a1f1-branch230m-to-1b", "8a1-wsd-bs1024-t300m"
BRANCH_AT = 230.4
LOG = "--log" in sys.argv
EVALS = {  # width -> (CubeGPT-3.1 result, CubeGPT-4 result)
    4000: ("8a1-wsd-bs1024-t300m-final_20261109-021035", "a8f4a1f1-branch230m-to-1b-final_20261110-055921"),
    16000: ("cubegpt-3-1-t300m_20261109-021849", "a8f4a1f1-branch230m-to-1b-final_20261110-055923"),
    64000: ("cubegpt-3-1-t300m_20261109-021938", "a8f4a1f1-branch230m-to-1b-final_20261110-055925"),
}


def curve(run):
    """[(samples M, train minutes, eval loss)], dropping records superseded by a divergence rollback."""
    pts = []
    for line in open(ROOT / "runs" / run / "metrics.jsonl"):
        r = json.loads(line)
        if "eval_loss" not in r:
            continue
        while pts and pts[-1][0] >= r["samples_seen"] / 1e6:
            pts.pop()
        pts.append((r["samples_seen"] / 1e6, r["train_seconds"] / 60, r["eval_loss"]))
    return np.array(pts)


v31, parent, branch = curve(V31), curve(PARENT), curve(BRANCH)
head = parent[parent[:, 0] <= BRANCH_AT + 1e-6]
tail = parent[parent[:, 0] >= BRANCH_AT - 1e-6]
branch = branch + np.array([BRANCH_AT, head[-1, 1], 0.0])
v4 = np.vstack([head, branch])

fig = plt.figure(figsize=(13, 12.5), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax_s, ax_t, ax_q = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def legend(ax, **kw):
    ax.legend(fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, **kw)


for ax, col in ((ax_s, 0), (ax_t, 1)):
    ax.plot(v31[:, col], v31[:, 2], color=GRAY, lw=1.8, label=f"CubeGPT-3.1 (19.09M params)  —  {v31[-1, 2]:.5f}")
    ax.plot(tail[:, col], tail[:, 2], color=BLUE, lw=1.6, label=f"same model, decayed at 307M instead  —  {tail[-1, 2]:.5f}")
    ax.plot(v4[:, col], v4[:, 2], color=GREEN, lw=2.4, zorder=3, label=f"CubeGPT-4 (3.82M params)  —  {v4[-1, 2]:.5f}")
    for c, color in ((v31, GRAY), (tail, BLUE), (v4, GREEN)):
        ax.plot(c[-1, col], c[-1, 2], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=4)
    ax.set_ylim(0.096, 0.124)
    ax.axhline(v31[-1, 2], color=GRAY, lw=1, ls=(0, (2, 3)), zorder=1)

style(ax_s, "Eval loss per training sample", "training samples (millions" + (", log scale)" if LOG else ")"), "eval loss" + (" (log scale)" if LOG else ""))
bounds = [0, BRANCH_AT, BRANCH_AT + 307.2, BRANCH_AT + 614.4, BRANCH_AT + 921.6]
for x in bounds[1:-1]:
    ax_s.axvline(x, color=GRID, lw=1.2, zorder=1)
for (a, b), text in zip(zip(bounds, bounds[1:]), ("lr 4e-4", "lr 2e-4", "lr 1e-4", "lr 1e-4 → 0")):
    ax_s.text(((max(a, 10) * b) ** 0.5 if LOG else (a + b) / 2), 0.1305 if LOG else 0.1228, text, ha="center", va="top", fontsize=9.5, color=INK2)
ax_s.set_xlim(0, 1175)
if LOG:
    for ax, ticks, lim in ((ax_s, [10, 20, 50, 100, 200, 500, 1000], (10, 1250)), (ax_t, [5, 10, 20, 50, 100, 200, 400], (4, 430))):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim)
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        yt = [0.10, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13]
        ax.set_yticks(yt); ax.set_yticklabels([f"{t:.3f}" for t in yt]); ax.minorticks_off(); ax.set_ylim(0.096, 0.132)
ax_s.text(1240 if LOG else 1172, v31[-1, 2] + 0.0004, "3.1 final", ha="right", fontsize=8.5, color=INK2)
legend(ax_s, loc="lower left" if LOG else "upper right", bbox_to_anchor=(0.0, 0.0) if LOG else (1.0, 0.9), title="final eval loss", title_fontsize=9)

style(ax_t, "Eval loss per minute of training (one H100)", "training time (minutes" + (", log scale)" if LOG else ")"), "eval loss" + (" (log scale)" if LOG else ""))
if not LOG:
    ax_t.set_xlim(0, 430)
cross = v4[np.argmax(v4[:, 2] < v31[-1, 2]), 1]
ax_t.annotate(f"passes 3.1's final\nloss at {cross:.0f} min\n(3.1 took {v31[-1, 1]:.0f})", (cross, v31[-1, 2]), (cross * 0.12, 0.1005) if LOG else (cross + 25, 0.1125),
              fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

style(ax_q, "Solution quality against search time", "search time per cube (seconds, log scale)", "moves above optimal (mean)")
for i, (name, color) in enumerate((("CubeGPT-3.1", GRAY), ("CubeGPT-4", GREEN))):
    xs, ys, es = [], [], []
    for width, files in EVALS.items():
        d = json.load(open(ROOT / "eval_results" / f"{files[i]}.json"))
        s = d["solve_summary"][str(width)]
        n = np.array([q["n_moves"] for q in d["solves"]])
        xs.append(s["mean_seconds"]); ys.append(s["mao"]); es.append(n.std(ddof=1) / np.sqrt(len(n)))
    ax_q.errorbar(xs, ys, yerr=es, color=color, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=1.5, capsize=3, elinewidth=1.2,
                  label=name, zorder=3 - (1 - i))
    for x, y, width in zip(xs, ys, EVALS):
        ax_q.annotate(f"{y:.2f}", (x, y), (9, 7) if i == 0 else (-9, -14), textcoords="offset points",
                      ha="left" if i == 0 else "right", fontsize=9, color=INK)
        if i == 0:
            ax_q.annotate(f"width {width:,}", (x, y), (9, 19), textcoords="offset points", fontsize=8.5, color=INK2)
ax_q.set_xscale("log"); ax_q.set_xticks([5, 10, 20, 50, 100]); ax_q.set_xticklabels(["5", "10", "20", "50", "100"])
ax_q.minorticks_off(); ax_q.set_xlim(3.2, 160); ax_q.set_ylim(0, 1.75)
legend(ax_q, loc="upper right", title="bars: ±1 standard error", title_fontsize=8.5)

fig.suptitle("CubeGPT-4 vs CubeGPT-3.1", x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5, w_pad=2)
out = ROOT / "images" / ("cubegpt4_log_2026-11-10.png" if LOG else "cubegpt4_2026-11-10.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out, f"cross {cross:.0f} min, v4 total {v4[-1, 1]:.0f} min")
