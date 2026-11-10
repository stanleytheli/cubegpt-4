"""Training runs of CubeGPT-3, CubeGPT-3.1 and CubeGPT-4: loss and learning rate against samples seen.
    .venv/bin/python data_analysis/plot_generations.py
CubeGPT-3 only logged training loss (transcript.txt + the two batch-4096 continuations in plot_training.py);
3.1 and 4 use eval loss from runs/*/metrics.jsonl. Training data is generated fresh, so the two track closely."""
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORANGE, GRAY, GREEN = "#eb6834", "#8a8985", "#008300"
BRANCH_AT = 230.4


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data_analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def curve(run):
    """[(samples M, eval loss, lr)], dropping records superseded by a divergence rollback."""
    pts = []
    for line in open(ROOT / "runs" / run / "metrics.jsonl"):
        r = json.loads(line)
        if "eval_loss" not in r:
            continue
        while pts and pts[-1][0] >= r["samples_seen"] / 1e6:
            pts.pop()
        pts.append((r["samples_seen"] / 1e6, r["eval_loss"], r["lr"]))
    return np.array(pts)


def final_bce(name):
    return json.load(open(ROOT / "eval_results" / f"{name}.json"))["metrics"]["bce"]


p31 = load("plot_cubegpt31")
o_s, o_l, o_lr_s, o_lr = p31.original_curve()
o_smooth = np.array(p31.log_window_mean(o_s, o_l, factor=1.15))
o_s, o_l, o_lr_s = np.array(o_s) / 1e6, np.array(o_l), np.array(o_lr_s) / 1e6
v31 = curve("8a1-wsd-bs1024-t300m")
parent, branch = curve("a8f4a1f1-prenorm-adamw-lr4e-4-t300m"), curve("a8f4a1f1-branch230m-to-1b")
v4 = np.vstack([parent[parent[:, 0] <= BRANCH_AT + 1e-6], branch + np.array([BRANCH_AT, 0, 0])])
BCE = {"3": final_bce("cubegpt-3-8a1-t300m_20261108-203602"), "3.1": final_bce("8a1-wsd-bs1024-t300m-final_20261109-021035"),
       "4": final_bce("a8f4a1f1-branch230m-to-1b-final_20261110-055921")}

fig = plt.figure(figsize=(13, 15.5), facecolor=SURFACE)
gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.6, 1.0])
ax_log, ax_lr, ax_lin = (fig.add_subplot(g) for g in gs)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def logx(ax):
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.set_xscale("log"); ax.set_xlim(0.9, 1500); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])


for ax in (ax_log, ax_lin):
    ax.plot(o_s, o_l, color=ORANGE, lw=0.8, alpha=0.25)
    ax.plot(o_s, o_smooth, color=ORANGE, lw=1.8, label=f"CubeGPT-3 (19.09M params, 307M samples)  —  {BCE['3']:.4f}")
    ax.plot(v31[:, 0], v31[:, 1], color=GRAY, lw=1.8, label=f"CubeGPT-3.1 (19.09M params, 307M samples)  —  {BCE['3.1']:.4f}")
    ax.plot(v4[:, 0], v4[:, 1], color=GREEN, lw=2.4, zorder=3, label=f"CubeGPT-4 (3.82M params, 1,152M samples)  —  {BCE['4']:.4f}")
    for x, y, c in ((o_s[-1], o_smooth[-1], ORANGE), (v31[-1, 0], v31[-1, 1], GRAY), (v4[-1, 0], v4[-1, 1], GREEN)):
        ax.plot(x, y, "o", ms=6, color=c, mec=SURFACE, mew=1.5, zorder=4)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK,
                    title="final loss on the shared 262k-position eval set", title_fontsize=9)
    leg.get_title().set_color(INK2)

style(ax_log, "Loss against training samples (log-log)", "training samples (millions, log scale)", "loss (log scale)")
logx(ax_log); ax_log.set_yscale("log"); ax_log.set_ylim(0.095, 0.21)
yt = [0.10, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.20]
ax_log.set_yticks(yt); ax_log.set_yticklabels([f"{t:.2f}" for t in yt]); ax_log.minorticks_off()
for x, y, text, dy in ((o_s[-1], o_smooth[-1], "CubeGPT-3", 9), (v31[-1, 0], v31[-1, 1], "CubeGPT-3.1", -18), (v4[-1, 0], v4[-1, 1], "CubeGPT-4", -4)):
    ax_log.annotate(text, (x, y), (-9 if dy < -10 else 9, dy), textcoords="offset points", ha="right" if dy < -10 else "left", fontsize=9, color=INK2)

style(ax_lr, "Learning rate", "training samples (millions, log scale)", "learning rate (log scale)")
known = [(s, v) for s, v in zip(o_lr_s, o_lr) if v]
ax_lr.plot([k[0] for k in known], [k[1] for k in known], color=ORANGE, lw=1.8, drawstyle="steps-post")
ax_lr.plot(v31[:, 0], v31[:, 2], color=GRAY, lw=1.8)
ax_lr.plot(v4[:, 0], v4[:, 2], color=GREEN, lw=2.4)
logx(ax_lr); ax_lr.set_yscale("log"); ax_lr.set_ylim(8e-9, 1.2e-2); ax_lr.minorticks_off()
ax_lr.annotate("CubeGPT-3: halves the lr whenever training loss\nstops improving; below 1e-6 by 50M samples\n(lr of its last 205M samples was not logged, ≤ 5e-7)",
               (45, 1.25e-6), (1.2, 3e-8), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax_lr.text(1.0, 2.2e-3, "CubeGPT-3.1 (gray): 5.6e-4, one rollback to 3.9e-4 at 87M, decay over the last 61M", fontsize=8.5, color=INK2)
ax_lr.text(1400, 2.2e-3, "CubeGPT-4 (green): 4e-4 → 2e-4 → 1e-4 → 0", ha="right", fontsize=8.5, color=INK2)

style(ax_lin, "Loss against training samples (linear)", "training samples (millions)", "loss")
ax_lin.set_xlim(0, 1175); ax_lin.set_ylim(0.095, 0.15)

fig.suptitle("CubeGPT-3, CubeGPT-3.1 and CubeGPT-4 training runs", x=0.012, y=0.992, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.012, 0.962, "CubeGPT-3: training loss (faint: as logged; bold: smoothed), the only loss it logged.  3.1 and 4: eval loss.  "
         "Training data is generated fresh, so the two track closely.", fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.5)
out = ROOT / "images" / "generations_2026-11-10.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out, BCE, o_smooth[-1])
