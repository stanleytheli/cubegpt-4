#!/usr/bin/env python
"""Per-scramble-depth accuracy of 8A1 checkpoints on a fixed uniform set of positions.

    .venv/bin/python data_analysis/depth_breakdown.py label=path [label=path ...] [--per-depth 1000]

For each depth 0..20 it scrambles `per-depth` cubes with exactly that many moves (same
positions for every model) and reports the model's mean prediction minus the depth (bias),
RMSE, and rounded accuracy. Runs on CPU (or CUDA if available).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubegpt.config import ModelConfig  # noqa: E402
from cubegpt.cube import (SCRAMBLE_LIMIT, CubeTables, canonicalize_full_batch, scramble_batch,  # noqa: E402
                          states_to_tokens)
from cubegpt.evaluate import ordinal_expectation  # noqa: E402
from cubegpt.model import build_model, load_weights  # noqa: E402

MODEL = ModelConfig(arch="moe", d_model=256, n_heads=16, d_ffwd=341, n_layers=8, n_experts=8, n_active=1)


def positions(T, per_depth, seed=1234):
    g = torch.random.fork_rng()
    with g:
        torch.manual_seed(seed)
        out = []
        for d in range(SCRAMBLE_LIMIT):
            lengths = torch.full((per_depth,), d, dtype=torch.int64, device=T.device)
            states, _ = scramble_batch(T, per_depth, lengths=lengths)
            out.append(states_to_tokens(T, canonicalize_full_batch(T, states)))
    return out


@torch.no_grad()
def predict(model, X, batch=2048):
    return torch.cat([ordinal_expectation(model(X[i:i + batch])) for i in range(0, len(X), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label=path to a .pth / checkpoint")
    ap.add_argument("--per-depth", type=int, default=1000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = CubeTables(device)
    Xs = positions(T, args.per_depth)
    results = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        m = build_model(MODEL).to(device)
        m.load_state_dict(load_weights(path, device))
        m.eval()
        results[label] = [predict(m, X) for X in Xs]
        print(f"scored {label}", file=sys.stderr)
    labels = list(results)
    w = max(12, max(len(l) for l in labels) + 1)
    print(f"{'depth':>5} | " + " | ".join(f"{l:^{w + 13}}" for l in labels))
    print(f"{'':>5} | " + " | ".join(f"{'bias':>6} {'RMSE':>6} {'acc':>{w - 1}}" for _ in labels))
    for d in range(SCRAMBLE_LIMIT):
        cells = []
        for l in labels:
            p = results[l][d]
            cells.append(f"{float(p.mean() - d):+6.2f} {float(((p - d) ** 2).mean().sqrt()):6.2f} "
                         f"{100 * float((p.round() == d).float().mean()):{w - 2}.1f}%")
        print(f"{d:>5} | " + " | ".join(cells))
    for lo, hi in ((0, 9), (10, 14), (15, 20)):
        cells = []
        for l in labels:
            p = torch.cat(results[l][lo:hi + 1])
            y = torch.cat([torch.full_like(results[l][d], d) for d in range(lo, hi + 1)])
            cells.append(f"{float((p - y).mean()):+6.2f} {float(((p - y) ** 2).mean().sqrt()):6.2f} "
                         f"{100 * float((p.round() == y).float().mean()):{w - 2}.1f}%")
        print(f"{f'{lo}-{hi}':>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
