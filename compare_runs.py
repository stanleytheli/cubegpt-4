#!/usr/bin/env python
"""Compare training runs side by side, straight from the Modal volume.

    python compare_runs.py 8a1-t100m-bs256 8a1-t100m-evalplateau
    python compare_runs.py RUN_A RUN_B --every 5 --plot runs.png    # rows every 5M samples; plot needs matplotlib

Prints eval loss and LR at matched sample counts, plus the end-of-stage metrics
(MSE / within-3 / within-4 / rounded accuracy). Eval loss is on each run's fixed
held-out set (same seed -> same set), so runs with the same seed are directly comparable.
"""
from __future__ import annotations

import argparse
import json

import modal

VOLUME_NAME = "cubegpt-3"


def read(vol, path):
    try:
        return b"".join(vol.read_file(path)).decode()
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=float, default=2.56, help="table row spacing in millions of samples")
    ap.add_argument("--plot", default="", help="save a PNG (eval loss and LR vs samples) to this path")
    args = ap.parse_args()
    vol = modal.Volume.from_name(VOLUME_NAME)

    data = {}
    for run in args.runs:
        text = read(vol, f"runs/{run}/metrics.jsonl")
        if not text:
            raise SystemExit(f"no metrics for run {run!r}")
        recs = [json.loads(l) for l in text.splitlines() if l.strip()]
        stages = json.loads(read(vol, f"runs/{run}/stage_metrics.json") or "[]")
        status = json.loads(read(vol, f"runs/{run}/status.json") or "{}")
        data[run] = (recs, stages, status)

    step = int(args.every * 1e6)
    max_s = max(r[-1]["samples_seen"] for r, _, _ in data.values())
    print(f"{'samples':>9} " + " ".join(f"{run[:30]:>32}" for run in args.runs))
    print(f"{'':>9} " + " ".join(f"{'eval loss':>14} {'lr':>9} {'stage':>6}" for _ in args.runs))
    target = step
    while target <= max_s + 1:
        cells = []
        for run in args.runs:
            recs = data[run][0]
            r = min(recs, key=lambda x: abs(x["samples_seen"] - target))
            if abs(r["samples_seen"] - target) > step / 2:
                cells.append(f"{'-':>14} {'':>9} {'':>6}")
            else:
                cells.append(f"{r['eval_loss']:>14.6f} {r['lr']:>9.2g} {r['stage']:>6}")
        print(f"{target / 1e6:>8.2f}M " + " ".join(cells))
        target += step

    print("\nend-of-stage metrics:")
    for run, (recs, stages, status) in data.items():
        print(f"  {run} ({status.get('state', '?')}, {100 * status.get('progress', 0):.0f}%):")
        for m in stages:
            print(f"    stage {m['stage']} @ {m['samples_seen'] / 1e6:6.1f}M: MSE {m['mse']:.4f} ± {m['mse_stderr']:.4f} | "
                  f"W3A {100 * m['within3']:.2f}% | W4A {100 * m['within4']:.2f}% | acc {100 * m['rounded_acc']:.2f}%")
        if not stages:
            print("    (no stage finished yet)")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("--plot needs matplotlib (pip install matplotlib)")
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for run, (recs, _, _) in data.items():
            xs = [r["samples_seen"] / 1e6 for r in recs]
            a1.plot(xs, [r["eval_loss"] for r in recs], label=run, lw=1.2)
            a2.plot(xs, [r["lr"] for r in recs], lw=1.2)
        a1.set_ylabel("eval loss (fixed held-out set)")
        a1.legend()
        a1.grid(alpha=0.3)
        a2.set_yscale("log")
        a2.set_ylabel("lr")
        a2.set_xlabel("samples (millions)")
        a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
