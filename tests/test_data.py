"""Training-data depth weighting (cubegpt/cube.py generate_samples(depth_power=...)).

    .venv/bin/python tests/test_data.py
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubegpt import config as C  # noqa: E402
from cubegpt.cube import SCRAMBLE_LIMIT, CubeTables, generate_samples  # noqa: E402

T = CubeTables("cpu")


def depths(y_ord):
    return (y_ord.sum(dim=1) - 1).long()


def test_power_zero_is_unchanged():
    torch.manual_seed(5)
    X0, y0 = generate_samples(T, 3000)
    torch.manual_seed(5)
    X1, y1 = generate_samples(T, 3000, depth_power=0.0)
    assert torch.equal(X0, X1) and torch.equal(y0, y1)


def test_linear_weighting_histogram():
    torch.manual_seed(0)
    n = 211_000
    _, y = generate_samples(T, n, depth_power=1.0, max_batch=65536)
    counts = torch.bincount(depths(y), minlength=SCRAMBLE_LIMIT).double()
    w = torch.arange(SCRAMBLE_LIMIT, dtype=torch.float64).clamp_min(1.0)
    expected = n * w / w.sum()
    z = (counts - expected) / expected.sqrt()
    print(f"  depth 0/1/10/20 counts {[int(counts[i]) for i in (0, 1, 10, 20)]} vs expected "
          f"{[int(expected[i]) for i in (0, 1, 10, 20)]}; max |z| {float(z.abs().max()):.2f}")
    assert float(z.abs().max()) < 5
    assert abs(counts[18] / counts[1] - 18) < 2                  # the requested 18x


def test_config_rules():
    base = {"name": "x", "stages": [{"steps": 10, "batch_size": 21, "lr": 1e-3}]}
    C.from_dict({**base, "stages": [{**base["stages"][0], "depth_weight_power": 1.0}]})
    try:
        C.from_dict({**base, "stages": [{**base["stages"][0], "depth_weight_power": 1.0, "data": "path"}]})
        raise AssertionError("accepted depth_weight_power with path data")
    except C.ConfigError:
        pass


def test_weighted_training_run():
    from cubegpt.train import run_training
    root = tempfile.mkdtemp()
    d = {"name": "dw", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096,
         "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4,
                   "n_active": 1},
         "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                     "warmup_samples": 64 * 30, "decay_samples": 64 * 60, "depth_weight_power": 1.0}]}
    r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    log = (Path(root) / "runs/dw/log.txt").read_text()
    recs = [json.loads(l) for l in open(Path(root) / "runs/dw/metrics.jsonl")]
    assert r["status"] == "completed" and "depth weight max(d,1)^1" in log
    assert recs[-1]["eval_loss"] < recs[0]["eval_loss"] and all(math.isfinite(x["eval_loss"]) for x in recs)
    shutil.rmtree(root)


if __name__ == "__main__":
    for fn in [test_power_zero_is_unchanged, test_linear_weighting_histogram, test_config_rules,
               test_weighted_training_run]:
        print(fn.__name__)
        fn()
        print("  ok")
