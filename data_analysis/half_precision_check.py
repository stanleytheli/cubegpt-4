"""Full vs half precision on the Mac GPU (MPS): the usual metrics on one fixed random set.
    .venv/bin/python data_analysis/half_precision_check.py [n_samples]"""
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cubegpt.config import load_model_config  # noqa: E402
from cubegpt.cube import CubeTables, generate_samples  # noqa: E402
from cubegpt.evaluate import ordinal_expectation  # noqa: E402
from cubegpt.model import build_model, load_weights  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
torch.manual_seed(4242)
X, y_ord = generate_samples(CubeTables("cpu"), N)
y = y_ord.sum(dim=1) - 1
model = build_model(load_model_config(str(ROOT / "configs/a8f4a1f1_branch230m_to_1b.toml")))
model.load_state_dict(load_weights(str(ROOT / "models/cubegpt-4-t1b.pth"), "cpu"))
model.eval().to("mps")
Xg = X.to("mps")


def run(m):
    t0 = time.time()
    with torch.no_grad():
        logits = torch.cat([m(Xg[i:i + 512]).float().cpu() for i in range(0, N, 512)])
    return logits, time.time() - t0


results = {}
for label, m in (("full (fp32)", model), ("half (fp16)", None)):
    if m is None:
        m = model.half()
    logits, dt = run(m)
    pred = ordinal_expectation(logits)
    err = (pred - y).double()
    results[label] = (logits, pred)
    print(f"{label}: BCE {F.binary_cross_entropy_with_logits(logits, y_ord).item():.6f} | MSE {(err ** 2).mean():.4f} | "
          f"acc {(torch.round(pred) == y).float().mean():.2%} | W1A {(err.abs() <= 1).float().mean():.2%} | "
          f"W2A {(err.abs() <= 2).float().mean():.2%} | W3A {(err.abs() <= 3).float().mean():.2%} | "
          f"W4A {(err.abs() <= 4).float().mean():.2%} | {N / dt:,.0f} states/s")
(l32, p32), (l16, p16) = results["full (fp32)"], results["half (fp16)"]
d = (p16 - p32).abs()
print(f"predicted distance, half vs full: mean |diff| {d.mean():.5f}, 99.9th pct {d.quantile(0.999):.4f}, max {d.max():.4f}; "
      f"rounded prediction changes on {(torch.round(p16) != torch.round(p32)).float().mean():.3%} of states; "
      f"non-finite outputs: {(~torch.isfinite(l16)).sum().item()}")
