"""Model metrics on states taken from beam-search solution paths instead of random scrambles.
    .venv/bin/python data_analysis/solution_path_metrics.py
For each eval scramble, takes the shortest solution found by any model at any width
(eval_results/*.json) and labels every state on that path with the moves left to solved.
Those labels are upper bounds on the true distance, and much tighter ones than the scramble
length used for the random sets. Writes eval_results/solution_path_metrics.json.
    .venv/bin/python data_analysis/solution_path_metrics.py --optimal
uses the optimal solutions in eval_results/optimal_seed0.json instead (from optimal_solve.py):
every label is then the exact distance. Writes eval_results/solution_path_metrics_optimal.json."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cubegpt.config import load_model_config  # noqa: E402
from cubegpt.cube import (CubeTables, canonicalize_full_batch, cube_string_to_state, rotate_string,  # noqa: E402
                          solved_string, states_to_tokens, to_ordinal_targets)
from cubegpt.evaluate import ordinal_expectation, scramble_for  # noqa: E402
from cubegpt.model import build_model, load_weights  # noqa: E402

MODELS = [("CubeGPT-3", "models/cubegpt-3-8a1-t300m.pth", "configs/cubegpt3_8a1_t100m.toml"),
          ("CubeGPT-3.1", "models/cubegpt-3-1-t300m.pth", "configs/8a1_wsd_bs1024_t300m.toml"),
          ("CubeGPT-4", "models/cubegpt-4-t1b.pth", "configs/a8f4a1f1_branch230m_to_1b.toml")]

# shortest known solution per eval scramble (seed 0, 30-move scrambles), across every eval file
best, sources = {}, {}
for f in sorted((ROOT / "eval_results").glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError:
        continue
    if not isinstance(d, dict) or "solves" not in d:
        continue
    for s in d["solves"]:
        if not s.get("solved") or s.get("scramble") != scramble_for(0, s["idx"], 30):
            continue
        if s["idx"] not in best or len(s["moves"]) < len(best[s["idx"]][1]):
            best[s["idx"]] = (s["scramble"], s["moves"])
            sources[s["idx"]] = f"{f.name} w{s['width']}"

OPTIMAL = "--optimal" in sys.argv
if OPTIMAL:
    best = {c["idx"]: (c["scramble"], c["optimal_solution"])
            for c in json.loads((ROOT / "eval_results" / "optimal_seed0.json").read_text())["cubes"]}
    sources = {i: "vcube optimal" for i in best}
strings, labels = [], []
for idx, (scramble, moves) in sorted(best.items()):
    state = scramble
    for k, move in enumerate(moves):
        strings.append(state); labels.append(len(moves) - k)
        state = rotate_string(state, move)
    assert state == solved_string
lengths = [len(m) for _, m in best.values()]
print(f"{len(best)} cubes, mean best-known solution {np.mean(lengths):.3f} moves, {len(strings)} states, "
      f"labels {min(labels)}..{max(labels)}")

T = CubeTables("cpu")
states = torch.stack([cube_string_to_state(T, s) for s in strings])
X = states_to_tokens(T, canonicalize_full_batch(T, states))
y = torch.tensor(labels)
y_ord = to_ordinal_targets(y)

out = {"cubes": len(best), "states": len(strings), "mean_solution": float(np.mean(lengths)), "models": {}}
for name, weights, cfg in MODELS:
    model = build_model(load_model_config(str(ROOT / cfg)))
    model.load_state_dict(load_weights(str(ROOT / weights), "cpu"))
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])
    pred = ordinal_expectation(logits)
    err = (pred - y).double()
    row = {"bce": F.binary_cross_entropy_with_logits(logits, y_ord).item(), "mse": (err ** 2).mean().item(),
           "within1": (err.abs() <= 1).float().mean().item(), "within2": (err.abs() <= 2).float().mean().item(),
           "within3": (err.abs() <= 3).float().mean().item(), "within4": (err.abs() <= 4).float().mean().item(),
           "rounded_acc": (torch.round(pred) == y).float().mean().item(), "mean_error": err.mean().item(),
           "mae": err.abs().mean().item(),
           "by_label": {int(k): {"n": int((y == k).sum()), "mean_pred": pred[y == k].mean().item(),
                                 "mse": (err[y == k] ** 2).mean().item(),
                                 "bce": F.binary_cross_entropy_with_logits(logits[y == k], y_ord[y == k]).item(),
                                 "rounded_acc": (torch.round(pred[y == k]) == k).float().mean().item(),
                                 **{f"within{w}": (err[y == k].abs() <= w).float().mean().item()
                                    for w in (1, 2, 3, 4)}} for k in sorted(set(labels))}}
    out["models"][name] = row
    print(f"{name:12s} BCE {row['bce']:.4f} | MSE {row['mse']:.3f} | acc {100 * row['rounded_acc']:.2f}% | W1A {100 * row['within1']:.2f}% | W2A {100 * row['within2']:.2f}% | W3A {100 * row['within3']:.2f}% | "
          f"W4A {100 * row['within4']:.2f}% | "
          f"mean error {row['mean_error']:+.3f} | MAE {row['mae']:.3f}")
print("\nmean prediction by moves left (label):")
print("left   n   " + "  ".join(f"{n:>11s}" for n, _, _ in MODELS))
for k in sorted(set(labels)):
    print(f"{k:>4} {out['models']['CubeGPT-4']['by_label'][k]['n']:>4}  "
          + "  ".join(f"{out['models'][n]['by_label'][k]['mean_pred']:>11.2f}" for n, _, _ in MODELS))
(ROOT / "eval_results" / f"solution_path_metrics{'_optimal' if OPTIMAL else ''}.json").write_text(json.dumps(out, indent=1))
import collections
print("\nsolutions came from:", dict(collections.Counter(v.split("_2026")[0] + " " + v.split()[-1] for v in sources.values())))
