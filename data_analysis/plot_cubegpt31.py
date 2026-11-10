#!/usr/bin/env python
"""Charts for CubeGPT-3.1 (the WSD / batch-1024 rerun of the CubeGPT-3 8A1 recipe).

Reads the local mirrors in runs/ and eval_results/ (python sync_runs.py) plus the
original CubeGPT-3 training transcript, and writes PNGs to images/cubegpt-3-1/.

    .venv/bin/python data_analysis/plot_cubegpt31.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images" / "cubegpt-3-1"

# ---- palette (validated: scripts/validate_palette.js, light mode, all-pairs PASS; slot 3 needs
# visible labels, which every chart here has). Color follows the recipe in every chart.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # 3.1 / batch 1024, eval-plateau / batch 256, original
OPTIMAL_MOVES = 17.7    # estimated mean optimal solution length for random scrambles (MAO baseline)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "legend.fontsize": 9, "lines.solid_capstyle": "round",
})


def title(fig, text, sub):
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.925, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT) / name}")


def metrics(run):
    return [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]


def final_eval(run):
    rows = metrics(run)
    return rows[-1]["eval_loss"]


def rolling(values, window, fn=st.mean):
    half = window // 2
    return [fn(values[max(0, i - half): i + half + 1]) for i in range(len(values))]


def log_window_mean(xs, ys, factor=1.1):
    """Average of the points with x in [x/factor, x*factor]: light smoothing early, more later
    (constant width on a log axis)."""
    out, lo, hi, n = [], 0, 0, len(xs)
    for x in xs:
        while xs[lo] < x / factor:
            lo += 1
        while hi < n and xs[hi] <= x * factor:
            hi += 1
        out.append(st.mean(ys[lo:hi]))
    return out


def millions(x, _pos=None):
    return f"{x / 1e6:.0f}M"


# ---------------------------------------------------------------------------------------------
def original_curve():
    """Original CubeGPT-3 8A1 (t100m -> t300m): training loss per logged point, from the
    notebook transcript (256k samples/point) plus the two batch-4096 continuation runs."""
    spec = importlib.util.spec_from_file_location("plot_training", ROOT / "data_analysis/plot_training.py")
    pt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pt)
    samples, losses = [], []
    lrs = []
    n = 0
    for line in open(ROOT / "data_analysis/transcript.txt"):
        m = pt.LOSS_RE.search(line)
        if not m:
            continue
        n += pt.SAMPLES_PER_POINT_FILE1
        samples.append(n)
        losses.append(float(m.group(1)))
        lr = re.search(r"current LR \[([0-9.eE+-]+)\]", line)
        lrs.append(float(lr.group(1)) if lr else None)
    lr_samples = list(samples)
    for run in pt.continuation_runs:
        for v in run["losses"]:
            n += run["samples_per_point"]
            samples.append(n)
            losses.append(v)
    return samples, losses, lr_samples, lrs


def fig_training_curves():
    o_s, o_l, o_lr_s, o_lr = original_curve()
    ep = metrics("8a1-t100m-evalplateau")
    lr_run = metrics("8a1-wsd-bs1024-t300m")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.12})
    fig.subplots_adjust(left=0.085, right=0.83, top=0.85, bottom=0.09)
    title(fig, "CubeGPT-3.1 trains to a much lower loss on the same 307M samples",
          "Loss on fresh training batches (the data is generated, so this tracks held-out loss); the original's noisy "
          "log is smoothed\n(average over ±10% of samples; raw shown faint). Bottom: learning rate.")

    o_smooth = log_window_mean(o_s, o_l)
    ax.plot(o_s, o_l, color=AQUA, lw=0.8, alpha=0.25)
    ax.plot(o_s, o_smooth, color=AQUA, lw=1.6, label="CubeGPT-3 (original t300m)")
    ax.plot([r["samples_seen"] for r in ep], [r["train_loss"] for r in ep], color=ORANGE, lw=1.6,
            label="eval-plateau rerun (batch 256)")
    ax.plot([r["samples_seen"] for r in lr_run], [r["train_loss"] for r in lr_run], color=BLUE, lw=1.6,
            label="CubeGPT-3.1 (batch 1024, WSD)")
    ylo, yhi = 0.100, 0.160
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("loss (BCE)")

    # direct labels at line ends (identity never by color alone)
    def end_label(xs, ys, text, color, dy=0.0):
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(8, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
        ax.plot([xs[-1]], [ys[-1]], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)
    end_label(o_s, o_smooth, f"CubeGPT-3\n{o_smooth[-1]:.3f}", AQUA, dy=4)
    end_label([r["samples_seen"] for r in ep], [r["train_loss"] for r in ep],
              f"eval-plateau\n{ep[-1]['train_loss']:.3f}", ORANGE, dy=-14)
    end_label([r["samples_seen"] for r in lr_run], [r["train_loss"] for r in lr_run],
              f"CubeGPT-3.1\n{lr_run[-1]['train_loss']:.3f}", BLUE)

    ds = lr_run[-1]["samples_seen"] - 61_440_000
    ax.annotate("LR decays\nto zero", xy=(ds, 0.1093), xytext=(ds - 52e6, 0.1035), fontsize=8.5, color=INK2,
                va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=2))

    # learning rate panel (log scale; the original's continuation LRs weren't logged)
    ax2.plot([s for s, v in zip(o_lr_s, o_lr) if v], [v for v in o_lr if v], color=AQUA, lw=1.4,
             drawstyle="steps-post")
    ax2.plot([r["samples_seen"] for r in ep], [r["lr"] for r in ep], color=ORANGE, lw=1.6, drawstyle="steps-post")
    pos = [r for r in lr_run if r["lr"] > 0]
    ax2.plot([r["samples_seen"] for r in pos], [r["lr"] for r in pos], color=BLUE, lw=1.6)
    ax2.text(150e6, 5.6e-4, "CubeGPT-3.1", fontsize=8.5, color=INK, fontweight="bold", va="bottom")
    ax2.text(9e6, 1.12e-4, "eval-plateau", fontsize=8.5, color=INK, fontweight="bold", va="bottom")
    ax2.set_yscale("log")
    ax2.set_ylim(5e-9, 2e-3)
    ax2.yaxis.set_major_locator(FixedLocator([1e-8, 1e-6, 1e-4]))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(__import__('math').log10(v)))}"))
    ax2.set_ylabel("learning rate")
    ax2.annotate("original: decays on training-loss noise,\nthen ~23M samples at 1.6e-8", xy=(84e6, 1.6e-8),
                 xytext=(118e6, 2.5e-8), fontsize=8.5, color=INK2, va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=2))
    ax2.set_xlabel("training samples")
    ax2.xaxis.set_major_formatter(FuncFormatter(millions))
    ax2.set_xlim(0, 312e6)
    for a in (ax, ax2):
        a.spines["bottom"].set_color(AXIS)
    save(fig, "training_curves.png")


def fig_training_curves_logx():
    """Same three runs as fig_training_curves, loss only, on a log sample axis."""
    o_s, o_l, _, o_lr = original_curve()
    ep = metrics("8a1-t100m-evalplateau")
    lr_run = metrics("8a1-wsd-bs1024-t300m")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.085, right=0.83, top=0.83, bottom=0.11)
    title(fig, "CubeGPT-3.1 keeps improving long after the original stalls",
          "Loss on fresh training batches vs samples seen (log scale); the original's noisy log is smoothed\n"
          "(average over ±10% of samples; raw shown faint).")
    o_smooth = log_window_mean(o_s, o_l)
    ep_x, ep_y = [r["samples_seen"] for r in ep], [r["train_loss"] for r in ep]
    lr_x, lr_y = [r["samples_seen"] for r in lr_run], [r["train_loss"] for r in lr_run]
    ax.plot(o_s, o_l, color=AQUA, lw=0.8, alpha=0.25)
    ax.plot(o_s, o_smooth, color=AQUA, lw=1.6)
    ax.plot(ep_x, ep_y, color=ORANGE, lw=1.6)
    ax.plot(lr_x, lr_y, color=BLUE, lw=1.6)

    for xs, ys, text, color, dy in ((o_s, o_smooth, f"CubeGPT-3\n{o_smooth[-1]:.3f}", AQUA, 4),
                                    (lr_x, lr_y, f"CubeGPT-3.1\n{lr_y[-1]:.3f}", BLUE, 0)):
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(8, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
        ax.plot([xs[-1]], [ys[-1]], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)

    ax.annotate(f"eval-plateau {ep_y[-1]:.3f}\n(stopped at {ep_x[-1] / 1e6:.0f}M)", (ep_x[-1], ep_y[-1]),
                xytext=(20e6, 0.1105), fontsize=9, color=INK, fontweight="bold", va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=3))
    ax.plot([ep_x[-1]], [ep_y[-1]], "o", ms=5, color=ORANGE, mec=SURFACE, mew=1.5, zorder=5)
    ylo, yhi = 0.100, 0.200
    ds = lr_x[-1] - 61_440_000
    ax.annotate("LR decays\nto zero", xy=(ds, 0.1105), xytext=(110e6, 0.1035), fontsize=8.5, color=INK2,
                va="center", ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=2))
    i45 = min(range(len(o_s)), key=lambda i: abs(o_s[i] - 45e6))
    ax.annotate("original stalls: its LR has\ndecayed to ~1e-6 and below", xy=(45e6, o_smooth[i45] + 0.0012),
                xytext=(16e6, 0.168), fontsize=8.5, color=INK2, va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=2))
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1e6, 3e6, 10e6, 30e6, 100e6, 300e6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlim(2.4e5, 340e6)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("training samples (log scale)")
    ax.set_ylabel("loss (BCE)")
    save(fig, "training_curves_logx.png")


# ---------------------------------------------------------------------------------------------
SWEEP = {  # batch -> [(lr, run name)]; every run: 25.6M samples, warmup 5%, constant, linear decay over last 20%
    1024: [(1e-4, "wsd25m-bs1024-lr1e-4"), (2e-4, "wsd25m-bs1024-lr2e-4"), (4e-4, "wsd25m-bs1024-lr4e-4"),
           (5.6e-4, "wsd25m-bs1024-lr5.6e-4"), (8e-4, "wsd25m-bs1024-lr8e-4"), (1.6e-3, "wsd25m-bs1024-lr1.6e-3")],
    4096: [(2e-4, "wsd25m-bs4096-lr2e-4"), (4e-4, "wsd25m-bs4096-lr4e-4"), (8e-4, "wsd25m-bs4096-lr8e-4"),
           (1.1e-3, "wsd25m-bs4096-lr1.1e-3"), (1.6e-3, "wsd25m-bs4096-lr1.6e-3")],
    256: [(1e-4, "wsd25m-bs256-lr1e-4")],
}


def run_result(run):
    status = json.load(open(ROOT / "runs" / run / "status.json"))
    rows = metrics(run)
    ok = status.get("state") == "completed"
    return ok, rows[-1]["eval_loss"] if ok else min(r["eval_loss"] for r in rows)


def fig_lr_sweep():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.1, right=0.80, top=0.83, bottom=0.12)
    title(fig, "Higher learning rates win, right up to the edge where training blows up",
          "Final held-out loss after 25.6M samples (warmup, constant LR, linear decay over the last 20%).")
    colors = {1024: BLUE, 4096: AQUA, 256: ORANGE}
    blow_y = 0.1385
    for bs in (1024, 4096, 256):
        good = [(lr, run_result(r)[1]) for lr, r in SWEEP[bs] if run_result(r)[0]]
        bad = [lr for lr, r in SWEEP[bs] if not run_result(r)[0]]
        xs, ys = zip(*good)
        ax.plot(xs, ys, "-o", color=colors[bs], lw=1.8, ms=6.5, mec=SURFACE, mew=1.5, zorder=3)
        if bad:
            jitter = {1024: 0.96, 4096: 1.04, 256: 1.0}[bs]
            ax.plot([b * jitter for b in bad], [blow_y] * len(bad), "x", color=colors[bs], ms=8, mew=2, zorder=3)
        label = {1024: "batch 1024", 4096: "batch 4096", 256: "batch 256\n(old LR)"}[bs]
        lx, ly = (xs[-1], ys[-1])
        ax.annotate(f"{label}", (lx, ly), xytext=(9, -1 if bs != 4096 else 6), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold")
    best = run_result("wsd25m-bs1024-lr5.6e-4")[1]
    ax.annotate(f"best: {best:.4f}\n(lr 5.6e-4, batch 1024)", (5.6e-4, best), xytext=(5.6e-4 * 0.42, 0.1232),
                fontsize=8.5, color=INK2, va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))
    ax.text(1.75e-3 * 1.04, blow_y, "blew up", fontsize=8.5, color=INK2, va="center")
    ax.set_xscale("log")
    ticks = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1e-4", "2e-4", "4e-4", "8e-4", "1.6e-3"])
    ax.set_xlim(8e-5, 2.4e-3)
    ax.set_ylim(0.122, 0.1395)
    ax.set_yticks([0.122, 0.124, 0.126, 0.128, 0.130, 0.132, 0.134, 0.136])
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("held-out loss after 25.6M samples")
    save(fig, "lr_batch_sweep.png")


# ---------------------------------------------------------------------------------------------
VARIANTS = [  # label, run at lr 5.6e-4, run at lr 8e-4
    ("Sigmoid router + balancing bias\n(the 8A1 default)", "wsd25m-bs1024-lr5.6e-4", "wsd25m-bs1024-lr8e-4"),
    ("Router at 0.1× LR", "wsd25m-bs1024-lr5.6e-4-router0.1", "wsd25m-bs1024-lr8e-4-router0.1"),
    ("Softmax router + aux balance loss\n+ z-loss (no bias)", "wsd25m-bs1024-lr5.6e-4-softmax-aux",
     "wsd25m-bs1024-lr8e-4-softmax-aux"),
    ("Softmax router + balancing bias", "wsd25m-bs1024-lr5.6e-4-softmax", "wsd25m-bs1024-lr8e-4-softmax"),
    ("Dense, same compute per token\n(4.4M params vs 19.1M)", "wsd25m-bs1024-lr5.6e-4-dense341",
     "wsd25m-bs1024-lr8e-4-dense341"),
]


def fig_variants():
    rows = [(lab, run_result(a)[1], run_result(b)) for lab, a, b in VARIANTS]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.subplots_adjust(left=0.31, right=0.80, top=0.80, bottom=0.13)
    title(fig, "None of the routing fixes beat the default router, and dense is worse",
          "Held-out loss after 25.6M samples at lr 5.6e-4, batch 1024 (lower is better). Right: the same setup at lr 8e-4.")
    base = rows[0][1]
    ys = list(range(len(rows)))[::-1]
    for y, (lab, v, (ok8, v8)) in zip(ys, rows):
        hero = y == ys[0]
        ax.plot([base, v], [y, y], color=GRID, lw=1.5, zorder=1)
        ax.plot([v], [y], "o", ms=8, color=BLUE if hero else MUTED, mec=SURFACE, mew=1.5, zorder=3)
        txt = f"{v:.4f}" if hero else f"{v:.4f}  (+{100 * (v / base - 1):.1f}%)"
        ax.annotate(txt, (v, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK if hero else INK2, fontweight="bold" if hero else "normal")
        ax.text(1.02, y, f"{v8:.4f}" if ok8 else "blew up", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color=INK2)
    ax.text(1.02, ys[0] + 0.75, "at lr 8e-4", transform=ax.get_yaxis_transform(), va="center", fontsize=9,
            color=INK, fontweight="bold")
    ax.axvline(base, color=AXIS, lw=0.8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(base - 0.0004, base + 0.0042)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("held-out loss after 25.6M samples")
    save(fig, "routing_variants.png")


# ---------------------------------------------------------------------------------------------
def fig_routing_health():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.78, bottom=0.14)
    title(fig, "At high LR the routing collapses: one expert takes most of a layer's tokens",
          "Busiest expert's load ÷ a fair share, worst layer, on the training batch before each eval "
          "(8 = every token in one expert).\nLine: rolling median of 15 evals; raw values faint.")
    for run, color, label in (("8a1-wsd-bs1024-t300m", BLUE, "CubeGPT-3.1\n(lr 5.6e-4 → 3.9e-4)"),
                              ("8a1-t100m-evalplateau", ORANGE, "eval-plateau\n(lr ≤ 1e-4)")):
        rows = metrics(run)
        xs = [r["samples_seen"] for r in rows]
        ys = [r["moe"]["max_load_ratio"] for r in rows]
        med = rolling(ys, 15, st.median)
        ax.plot(xs, ys, color=color, lw=0.7, alpha=0.25)
        ax.plot(xs, med, color=color, lw=1.8)
        ax.annotate(label, (xs[-1], med[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.text(305e6, 0.85, "perfectly balanced = 1", fontsize=8.5, color=INK2, va="top", ha="right")
    ax.set_ylim(0, 8.4)
    ax.set_xlim(0, 312e6)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlabel("training samples")
    ax.set_ylabel("busiest expert ÷ fair share")
    save(fig, "routing_health.png")


# ---------------------------------------------------------------------------------------------
WIDTH_RUNS = {
    ("CubeGPT-3.1", 4000): "8a1-wsd-bs1024-t300m-final_20261109-021035.json",
    ("CubeGPT-3.1", 16000): "cubegpt-3-1-t300m_20261109-021849.json",
    ("CubeGPT-3.1", 64000): "cubegpt-3-1-t300m_20261109-021938.json",
    ("eval-plateau", 4000): "8a1-t100m-evalplateau-latest_20261108-203603.json",
    ("eval-plateau", 16000): "8a1-t100m-evalplateau-latest_w16000_PARTIAL_recovered_from_log.json",
    ("eval-plateau", 64000): "8a1-t100m-evalplateau-latest_w64000_PARTIAL_recovered_from_log.json",
    ("CubeGPT-3", 4000): "cubegpt-3-8a1-t300m_20261108-203602.json",
}


def fig_width_scaling():
    paths = {k: ROOT / "eval_results" / v for k, v in WIDTH_RUNS.items()}
    missing = [str(p.name) for p in paths.values() if not p.exists()]
    if missing:
        print(f"skipping width_scaling.png (missing {missing})")
        return
    solves = {k: {s["idx"]: s for s in json.load(open(p))["solves"]} for k, p in paths.items()}
    common = sorted(set.intersection(*[set(v) for v in solves.values()]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.80, bottom=0.13)
    title(fig, "A wider beam search finds shorter solutions; CubeGPT-3.1 needs a quarter of the width",
          f"Mean solution length on the same {len(common)} random scrambles (every model and width solved all of them).")
    for name, color in (("CubeGPT-3.1", BLUE), ("eval-plateau", ORANGE), ("CubeGPT-3", AQUA)):
        ws = [w for w in (4000, 16000, 64000) if (name, w) in solves]
        ms = [st.mean(solves[(name, w)][i]["n_moves"] for i in common) for w in ws]
        ax.plot(ws, ms, "-o", color=color, lw=1.8, ms=7, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(f"{name}  {ms[-1]:.2f}", (ws[-1], ms[-1]), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(OPTIMAL_MOVES, color=AXIS, lw=0.8)
    ax.text(4000 * 0.93, OPTIMAL_MOVES + 0.06, f"estimated mean optimal ≈ {OPTIMAL_MOVES}", fontsize=8.5, color=INK2,
            va="bottom")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([4000, 16000, 64000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlim(3400, 76000)
    ax.set_ylim(17.3, 21.1)
    ax.set_xlabel("beam width")
    ax.set_ylabel("mean solution length (moves)")
    save(fig, "width_scaling.png")


if __name__ == "__main__":
    fig_training_curves()
    fig_training_curves_logx()
    fig_lr_sweep()
    fig_variants()
    fig_routing_health()
    fig_width_scaling()
