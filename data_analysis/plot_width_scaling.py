"""Beam-width scaling of CubeGPT-3, 3.1 and 4: mean solution length at widths 4,000 / 16,000 / 64,000.
    .venv/bin/python data_analysis/plot_width_scaling.py
All measured points use the same seed-0, 30-move scrambles (200 cubes; 100 at width 64,000). CubeGPT-3 (t300m) was only
evaluated at width 4,000; its wider points are the README's older numbers for the path-finetuned t100m model on other scrambles."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORANGE, GRAY, GREEN = "#eb6834", "#8a8985", "#008300"
OPTIMAL = 17.7   # estimated mean optimal solution length for random scrambles
WIDTHS = [4000, 16000, 64000]
RUNS = {
    "CubeGPT-3": (ORANGE, {4000: "cubegpt-3-8a1-t300m_20261108-203602"}),
    "CubeGPT-3.1": (GRAY, {4000: "8a1-wsd-bs1024-t300m-final_20261109-021035", 16000: "cubegpt-3-1-t300m_20261109-021849",
                           64000: "cubegpt-3-1-t300m_20261109-021938"}),
    "CubeGPT-4": (GREEN, {4000: "a8f4a1f1-branch230m-to-1b-final_20261110-055921", 16000: "a8f4a1f1-branch230m-to-1b-final_20261110-055923",
                          64000: "a8f4a1f1-branch230m-to-1b-final_20261110-055925"}),
}
README_PFT = {4000: 20.2, 16000: 19.6, 64000: 19.0}   # CubeGPT-3-8A1-t100mPFT, README solving table

fig, ax = plt.subplots(figsize=(10.5, 7), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(list(README_PFT), list(README_PFT.values()), color=ORANGE, lw=1.6, ls=(0, (4, 3)), marker="o", ms=8, mfc=SURFACE, mec=ORANGE, mew=1.6, zorder=2)
for w, m in README_PFT.items():
    ax.annotate(f"{m:.1f}", (w, m), (0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK2)
ax.annotate("CubeGPT-3 path-finetuned\n(README, different scrambles)", (64000, 19.0), (12, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)

for name, (color, files) in RUNS.items():
    ws, ms, es = [], [], []
    for w, f in files.items():
        solves = json.load(open(ROOT / "eval_results" / f"{f}.json"))["solves"]
        n = np.array([s["n_moves"] for s in solves if s["solved"]])
        ws.append(w); ms.append(n.mean()); es.append(n.std(ddof=1) / np.sqrt(len(n)))
        tag = f"{n.mean():.2f}" + ("" if len(n) == len(solves) else f"  ({len(n)} of {len(solves)} solved)")
        up = name != "CubeGPT-4"
        ax.annotate(tag, (w, n.mean()), (0, 11 if up else -17), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.errorbar(ws, ms, yerr=es, color=color, lw=2.4 if name == "CubeGPT-4" else 1.9, marker="o", ms=9, mec=SURFACE, mew=1.5,
                capsize=3, elinewidth=1.2, zorder=3)
    ax.annotate(name, (ws[-1], ms[-1]), (12, 5 if name == "CubeGPT-3.1" else -5 if name == "CubeGPT-4" else 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="bold", color=INK)

ax.axhline(OPTIMAL, color=INK2, lw=1, ls=(0, (2, 3)))
ax.text(3500, OPTIMAL + 0.05, f"estimated mean optimal ≈ {OPTIMAL}", fontsize=9, color=INK2)
ax.set_xscale("log", base=2); ax.set_xticks(WIDTHS); ax.set_xticklabels([f"{w:,}" for w in WIDTHS]); ax.minorticks_off()
ax.set_xlim(3300, 110000); ax.set_ylim(17.4, 21.0)
ax.set_xlabel("beam width (log scale)", color=INK2, fontsize=10); ax.set_ylabel("mean solution length (moves)", color=INK2, fontsize=10)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=INK2, length=0, labelsize=9)
fig.suptitle("Width scaling: CubeGPT-3, 3.1 and 4", x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.text(0.012, 0.915, "Solid: same seed-0 30-move scrambles (200 cubes; 100 at width 64,000), bars ±1 standard error.\n"
         "CubeGPT-3 was only measured at width 4,000; dashed hollow points are older README numbers, not directly comparable.",
         fontsize=9.5, color=INK2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = ROOT / "images" / "width_scaling_2026-11-10.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
