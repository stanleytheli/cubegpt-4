"""Evaluation progress persistence (cubegpt/evaluate.py + modal_app._combine_eval).

    .venv/bin/python tests/test_eval.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubegpt.config import ModelConfig  # noqa: E402
from cubegpt.evaluate import eval_parts_dir, load_eval_parts, run_eval_shard  # noqa: E402
from cubegpt.model import build_model  # noqa: E402

MODEL = {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4, "n_active": 1}


def _setup():
    root = Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    (root / "models").mkdir()
    torch.save(build_model(ModelConfig(**MODEL)).state_dict(), root / "models/tiny.pth")
    spec = {"checkpoint": "models/tiny.pth", "model": MODEL, "metrics_samples": 512, "solve_cubes": 6,
            "widths": [8], "scramble_moves": 3, "max_moves": 6, "seed": 0, "eval_batch_size": 4096,
            "tf32": False, "out_name": "t"}
    return root, spec


def test_progress_saved_and_resumed():
    root, spec = _setup()
    commits = []
    out = run_eval_shard(spec, root, shard=1, n_shards=2, device="cpu", commit=lambda: commits.append(1))
    done, metrics = load_eval_parts(eval_parts_dir(root, "t"))
    assert sorted(done) == [(1, 8), (3, 8), (5, 8)] and metrics is None      # shard 1 has no metrics
    assert len(commits) == 3                                                  # committed after every cube
    # a "retry" of the same shard must skip everything and return identical results
    commits.clear()
    again = run_eval_shard(spec, root, shard=1, n_shards=2, device="cpu", commit=lambda: commits.append(1))
    assert again["solves"] == out["solves"] and not commits
    shutil.rmtree(root)


def test_combine_fills_failed_shard_from_parts():
    import importlib.util
    root, spec = _setup()
    run_eval_shard(spec, root, shard=0, n_shards=2, device="cpu")             # shard 0 "finished" ...
    good = run_eval_shard(spec, root, shard=1, n_shards=2, device="cpu")
    # ... but its return value was lost (coordinator saw an exception); shard 1 returned normally
    modal_app_spec = importlib.util.spec_from_file_location("modal_app", ROOT / "modal_app.py")
    m = importlib.util.module_from_spec(modal_app_spec)
    modal_app_spec.loader.exec_module(m)
    res = m._combine_eval(spec, "cpu", [RuntimeError("workspace disabled"), good], parts_root=root)
    assert [r["idx"] for r in res["solves"]] == list(range(6)), [r["idx"] for r in res["solves"]]
    assert res["metrics"] is not None and res["complete"] and len(res["failed_shards"]) == 1
    print(f"  combined {len(res['solves'])} cubes, metrics from saved parts: MSE {res['metrics']['mse']:.2f}, "
          f"failed shards noted: {res['failed_shards']}")
    shutil.rmtree(root)


if __name__ == "__main__":
    for fn in [test_progress_saved_and_resumed, test_combine_fills_failed_shard_from_parts]:
        print(fn.__name__)
        fn()
        print("  ok")
