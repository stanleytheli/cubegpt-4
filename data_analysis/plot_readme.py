"""README figures: one idea per chart, written to images/readme/.
    .venv/bin/python data_analysis/plot_readme.py        (reads runs/*/metrics.jsonl and eval_results/*.json)"""
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images" / "readme"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
# validated with the dataviz palette checker (light mode); model colors are fixed across charts
BLUE, ORANGE, GREEN, PINK, PURPLE, GOLD = "#2a6fe8", "#ff7a1a", "#007f5f", "#d6338a", "#7a4ce0", "#d49a00"
C3, C31, C4 = BLUE, ORANGE, GREEN
BRANCH_AT = 230.4
OPTIMAL = 17.7
S = "wsd25m-bs1024-"


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


def chart(title, xlabel, ylabel, size=(8.5, 5.2)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10.5); ax.set_ylabel(ylabel, color=INK2, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    return fig, ax


def finish(fig, ax, name, legend=True, **kw):
    if legend:
        ax.legend(fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, framealpha=1, **kw)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote images/readme/{name}")


# ---- the three generations --------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("plot_cubegpt31", ROOT / "data_analysis" / "plot_cubegpt31.py")
p31 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p31)
plt.rcdefaults()
o_s, o_l, o_lr_s, o_lr = p31.original_curve()
o_smooth = np.array(p31.log_window_mean(o_s, o_l, factor=1.15))
o_s, o_lr_s = np.array(o_s) / 1e6, np.array(o_lr_s) / 1e6
v31 = curve("8a1-wsd-bs1024-t300m")
parent, branch = curve("a8f4a1f1-prenorm-adamw-lr4e-4-t300m"), curve("a8f4a1f1-branch230m-to-1b")
v4 = np.vstack([parent[parent[:, 0] <= BRANCH_AT + 1e-6], branch + np.array([BRANCH_AT, 0, 0])])
GENS = (("CubeGPT-3", C3, o_s, o_smooth), ("CubeGPT-3.1", C31, v31[:, 0], v31[:, 1]), ("CubeGPT-4", C4, v4[:, 0], v4[:, 1]))


def loss_chart(name, title, log):
    fig, ax = chart(title, "training samples (millions" + (", log scale)" if log else ")"), "loss")
    for label, color, x, y in GENS:
        ax.plot(x, y, color=color, lw=2.2, label=label, solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=6.5, color=color, mec=SURFACE, mew=1.5, zorder=4)
    if log:
        ticks = [1, 3, 10, 30, 100, 300, 1000]
        ax.set_xscale("log"); ax.set_xlim(0.9, 1300); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_ylim(0.095, 0.18)
    else:
        ax.set_xlim(0, 1180); ax.set_ylim(0.095, 0.15)
    finish(fig, ax, name, loc="upper right")


loss_chart("loss_vs_samples.png", "Loss during training", log=False)
loss_chart("loss_vs_log_samples.png", "Loss during training (log samples)", log=True)

fig, ax = chart("Learning rate schedules", "training samples (millions)", "learning rate (log scale)")
known = [(s, v) for s, v in zip(o_lr_s, o_lr) if v]
ax.plot([k[0] for k in known], [k[1] for k in known], color=C3, lw=2.2, drawstyle="steps-post", label="CubeGPT-3")
ax.plot(v31[:, 0], v31[:, 2], color=C31, lw=2.2, label="CubeGPT-3.1")
ax.plot(v4[:, 0], v4[:, 2], color=C4, lw=2.2, label="CubeGPT-4")
ax.set_yscale("log"); ax.set_ylim(1e-8, 2e-3); ax.set_xlim(0, 1180); ax.minorticks_off()
finish(fig, ax, "lr_vs_samples.png", loc="lower right")

# ---- width scaling ----------------------------------------------------------------------------
WIDTHS = [4000, 16000, 64000]
SOLVES = {
    "CubeGPT-3.1": (C31, ["8a1-wsd-bs1024-t300m-final_20261109-021035", "cubegpt-3-1-t300m_20261109-021849", "cubegpt-3-1-t300m_20261109-021938"]),
    "CubeGPT-4": (C4, ["a8f4a1f1-branch230m-to-1b-final_20261110-055921", "a8f4a1f1-branch230m-to-1b-final_20261110-055923",
                       "a8f4a1f1-branch230m-to-1b-final_20261110-055925"]),
}
fig, ax = chart("Solution length against beam width", "beam width (log scale)", "mean solution length (moves)")
series = [("CubeGPT-3", C3, [20.2, 19.6, 19.0])]      # CubeGPT-3-8A1-t100mPFT, from the README solving table
for name, (color, files) in SOLVES.items():
    series.append((name, color, [json.load(open(ROOT / "eval_results" / f"{f}.json"))["solve_summary"][str(w)]["mean_moves"]
                                 for f, w in zip(files, WIDTHS)]))
for name, color, ms in series:
    ax.plot(WIDTHS, ms, color=color, lw=2.4, marker="o", ms=9, mec=SURFACE, mew=1.6, label=name)
    for w, m in zip(WIDTHS, ms):
        ax.annotate(f"{m:.1f}" if name == "CubeGPT-3" else f"{m:.2f}", (w, m),
                    (0, -17 if name == "CubeGPT-4" else 10), textcoords="offset points", ha="center", fontsize=9.5, color=INK)
ax.axhline(OPTIMAL, color=INK2, lw=1.1, ls=(0, (2, 3)), label=f"optimal (≈ {OPTIMAL})")
ax.set_xscale("log", base=2); ax.set_xticks(WIDTHS); ax.set_xticklabels([f"{w:,}" for w in WIDTHS]); ax.minorticks_off()
ax.set_xlim(3200, 80000); ax.set_ylim(17.4, 20.6); ax.grid(False, axis="x")
finish(fig, ax, "width_scaling.png", loc="upper right")


# ---- 25.6M-sample experiments -----------------------------------------------------------------
def experiment_chart(name, title, runs):
    fig, ax = chart(title, "training samples (millions)", "eval loss (5-point moving average)")
    rows = sorted(((curve(run), label, color) for run, label, color in runs), key=lambda r: -r[0][-1, 1])
    for c, label, color in rows:
        y = np.array([c[max(0, i - 2): i + 3, 1].mean() for i in range(len(c))])
        y[-1] = c[-1, 1]
        ax.plot(c[:, 0], y, color=color, lw=2.0, label=f"{label}  —  {c[-1, 1]:.4f}", solid_capstyle="round")
        ax.plot(c[-1, 0], c[-1, 1], "o", ms=6, color=color, mec=SURFACE, mew=1.4, zorder=4)
    ax.set_xlim(5, 26.4); ax.set_ylim(0.118, 0.142); ax.set_xticks([5, 10, 15, 20, 25])
    finish(fig, ax, name, loc="lower left", title="final eval loss", title_fontsize=9.5)


experiment_chart("experiments_training.png", "Training pipeline experiments (8 attention + 4 FFN layers)", [
    (S + "lr5.6e-4-attn8-dense4", "post-norm, Adam (3.1 recipe)", ORANGE),
    (S + "lr8e-4-attn8-dense4-prenorm", "pre-norm", BLUE),
    (S + "lr8e-4-attn8-dense4-b2-0.95", "Adam β₂ 0.95", GOLD),
    (S + "lr8e-4-attn8-dense4-adamw", "AdamW", PURPLE),
    ("wsd25m-a8d4-prenorm-muon0.005-lr8e-4", "pre-norm + Muon", PINK),
    ("wsd25m-a8d4-prenorm-adamw-lr8e-4", "pre-norm + AdamW", GREEN),
])
experiment_chart("experiments_architecture.png", "Architecture experiments (3.1 recipe)", [
    (S + "lr5.6e-4-attnonly", "8 attention, no FFN", BLUE),
    (S + "lr5.6e-4-attn7-dense1", "7 attention + 1 FFN", GOLD),
    (S + "lr5.6e-4-attn10-dense4", "10 attention + 4 FFN", PURPLE),
    (S + "lr5.6e-4", "8 × (attention + MoE)  [CubeGPT-3.1]", ORANGE),
    (S + "lr5.6e-4-attn8-dense6", "8 attention + 6 FFN", PINK),
    (S + "lr5.6e-4-attn8-dense4", "8 attention + 4 FFN", GREEN),
])
experiment_chart("experiments_architecture_new_recipe.png", "Architecture experiments (pre-norm + AdamW)", [
    ("wsd25m-a7f3a1f1-prenorm-adamw-lr8e-4", "7 attn + 3 FFN + 1 attn + 1 FFN", BLUE),
    ("wsd25m-a8d4-prenorm-adamw-lr8e-4", "8 attn + 4 FFN", ORANGE),
    ("wsd25m-a8d4-d384-prenorm-adamw-lr5.3e-4", "8 attn + 4 FFN, width 384", PURPLE),
    ("wsd25m-a8f4a1f1-prenorm-adamw-lr8e-4", "8 attn + 4 FFN + 1 attn + 1 FFN  [CubeGPT-4]", GREEN),
])
