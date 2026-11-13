"""Each model's metrics on states from each model's own solution paths (who solved x who is scored).
    .venv/bin/python data_analysis/solution_path_matrix.py [--width 4000]
Paths: per scramble, the shortest solution the solving model found (at any width, or only --width).
Labels are the moves left on that path. Writes eval_results/solution_path_matrix.json."""
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

# name, weights, model config, eval-file prefixes holding that model's solves
MODELS = [("CubeGPT-3", "models/cubegpt-3-8a1-t300m.pth", "configs/cubegpt3_8a1_t100m.toml", ["cubegpt-3-8a1-t300m_"]),
          ("CubeGPT-3.1", "models/cubegpt-3-1-t300m.pth", "configs/8a1_wsd_bs1024_t300m.toml",
           ["8a1-wsd-bs1024-t300m-final_", "cubegpt-3-1-t300m_"]),
          ("CubeGPT-4", "models/cubegpt-4-t1b.pth", "configs/a8f4a1f1_branch230m_to_1b.toml",
           ["a8f4a1f1-branch230m-to-1b-final_"])]
WIDTH = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else None
T = CubeTables("cpu")


def path_set(prefixes):
    best = {}
    for f in sorted((ROOT / "eval_results").glob("*.json")):
        if not any(f.name.startswith(p) for p in prefixes) or "tracked" in f.name:
            continue
        for s in json.loads(f.read_text()).get("solves", []):
            if not s.get("solved") or s.get("scramble") != scramble_for(0, s["idx"], 30):
                continue
            if WIDTH and s["width"] != WIDTH:
                continue
            if s["idx"] not in best or len(s["moves"]) < len(best[s["idx"]][1]):
                best[s["idx"]] = (s["scramble"], s["moves"])
    strings, labels = [], []
    for _, (scramble, moves) in sorted(best.items()):
        state = scramble
        for k, move in enumerate(moves):
            strings.append(state); labels.append(len(moves) - k)
            state = rotate_string(state, move)
        assert state == solved_string
    keep = [i for i, l in enumerate(labels) if l <= 20]            # the models' head covers 0..20
    states = torch.stack([cube_string_to_state(T, strings[i]) for i in keep])
    y = torch.tensor([labels[i] for i in keep])
    return {"X": states_to_tokens(T, canonicalize_full_batch(T, states)), "y": y, "cubes": len(best),
            "mean_len": float(np.mean([len(m) for _, m in best.values()])), "dropped": len(labels) - len(keep)}


sets = {name: path_set(prefixes) for name, _, _, prefixes in MODELS}
for name, s in sets.items():
    print(f"paths from {name}: {s['cubes']} cubes, mean solution {s['mean_len']:.2f}, {len(s['y'])} states"
          + (f" ({s['dropped']} with more than 20 moves left dropped)" if s["dropped"] else ""))

out = {"width": WIDTH, "sets": {n: {k: s[k] for k in ("cubes", "mean_len")} | {"states": len(s["y"])}
                                for n, s in sets.items()}, "scores": {}}
for name, weights, cfg, _ in MODELS:
    model = build_model(load_model_config(str(ROOT / cfg)))
    model.load_state_dict(load_weights(str(ROOT / weights), "cpu"))
    model.eval()
    for src, s in sets.items():
        with torch.no_grad():
            logits = torch.cat([model(s["X"][i:i + 1024]) for i in range(0, len(s["X"]), 1024)])
        pred, y = ordinal_expectation(logits), s["y"]
        err = (pred - y).double()
        out["scores"][f"{name} on {src}"] = {
            "bce": F.binary_cross_entropy_with_logits(logits, to_ordinal_targets(y)).item(),
            "mse": (err ** 2).mean().item(), "rounded_acc": (torch.round(pred) == y).float().mean().item(),
            **{f"within{k}": (err.abs() <= k).float().mean().item() for k in (1, 2, 3, 4)}}

for metric, fmt in (("bce", "{:.4f}"), ("mse", "{:.3f}"), ("rounded_acc", "{:.2%}"), ("within1", "{:.2%}"),
                    ("within2", "{:.2%}"), ("within3", "{:.2%}"), ("within4", "{:.2%}")):
    print(f"\n{metric}  (rows: model scored; columns: whose solution paths)")
    print(" " * 13 + "".join(f"{n:>13s}" for n in sets))
    for name, *_ in MODELS:
        print(f"{name:13s}" + "".join(f"{fmt.format(out['scores'][f'{name} on {src}'][metric]):>13s}" for src in sets))
(ROOT / "eval_results" / f"solution_path_matrix{'_w%d' % WIDTH if WIDTH else ''}.json").write_text(json.dumps(out, indent=1))
