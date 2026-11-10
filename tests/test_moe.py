"""Fast MoE dispatch ("grouped") vs the original loop ("loop"), on CPU.

    .venv/bin/python tests/test_moe.py          # needs torch locally (see requirements.txt)
    .venv/bin/python -m pytest tests/           # if pytest is installed

The GPU version of these checks (plus throughput) is `modal run modal_app.py::bench`.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubegpt.config import ModelConfig  # noqa: E402
from cubegpt.cube import CubeTables, generate_samples  # noqa: E402
from cubegpt.model import build_model, load_weights  # noqa: E402
from cubegpt.verify import (check_ok, compare_attention, compare_model, compare_training,  # noqa: E402
                            fmt_rel, run_block_suite)

ROOT = Path(__file__).resolve().parent.parent
MOE_8A1 = ModelConfig("moe", 256, 16, 341, 8, 8, 1)


def test_blocks_float64_exact():
    """Same function up to float64 rounding, incl. dead experts / collapse / top-2 / tiny N."""
    for label, r in run_block_suite(torch.float64):
        print(f"  {label:32s} path={r['path']:7s} dead={r['dead_experts']} out {fmt_rel(r['out_rel'])} "
              f"dx {fmt_rel(r['grad_x_rel'])} dW {fmt_rel(r['grad_param_rel'])} load_equal={r['load_equal']}")
        assert check_ok(r, 1e-12), label
    paths = {r["path"] for _, r in run_block_suite(torch.float64)}
    assert paths == {"batched", "sliced"}, "both dispatch paths must be exercised"


def test_blocks_float32():
    for label, r in run_block_suite(torch.float32):
        assert check_ok(r, 1e-5), (label, r)


def test_real_checkpoints():
    T = CubeTables("cpu")
    torch.manual_seed(0)
    X, y = generate_samples(T, 512)
    for name, cfg in [("cubegpt-3-8a1-t300m", MOE_8A1), ("cubegpt-gated-moe-8a1-2", MOE_8A1),
                      ("cubegpt-gated-moe-2a1", ModelConfig("moe", 256, 16, 341, 8, 2, 1))]:
        path = ROOT / "models" / f"{name}.pth"
        if not path.exists():
            continue
        m = build_model(cfg)
        m.load_state_dict(load_weights(path, "cpu"))
        r = compare_model(m, X, y)
        print(f"  {name:28s} logits {fmt_rel(r['logits_rel'])} grads {fmt_rel(r['grad_rel'])} "
              f"loss {r['loss_ref']:.7f} vs {r['loss_fast']:.7f} routing_equal={r['routing_equal']} "
              f"dead-expert grads None in both={r['none_pattern_equal']}")
        assert r["routing_equal"] and r["none_pattern_equal"] and r["logits_rel"] < 1e-5 and r["grad_rel"] < 1e-5, \
            (name, r)


def test_attention_backends():
    """Explicit softmax(QK^T)V vs F.scaled_dot_product_attention on a full model."""
    T = CubeTables("cpu")
    torch.manual_seed(0)
    X, y = generate_samples(T, 256)
    m = build_model(MOE_8A1).double()
    r = compare_attention(m, X, y.double())
    print(f"  float64: logits {fmt_rel(r['logits_rel'])} grads {fmt_rel(r['grad_rel'])}")
    assert r["logits_rel"] < 1e-12 and r["grad_rel"] < 1e-12, r


def test_exact_attention_gradcheck():
    """The hand-written backward of the full-fp32 explicit attention is correct."""
    from cubegpt.model import _ExactAttention
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda a, b, c: _ExactAttention.apply(a, b, c, 0.5), (q, k, v))
    torch.set_float32_matmul_precision("high")
    _ExactAttention.apply(q.float(), k.float(), v.float(), 0.5)
    assert torch.get_float32_matmul_precision() == "high", "global TF32 setting not restored"


def test_training_trajectory():
    T = CubeTables("cpu")
    r = compare_training(ModelConfig("moe", 32, 4, 48, 2, 8, 2), T, steps=200, batch_size=256, lr=1e-3)
    print(f"  200 steps: max rel loss diff {r['max_rel_loss_diff']:.1e}, final loss {r['final_loss_ref']:.6f} vs "
          f"{r['final_loss_fast']:.6f}, final params rel diff {r['final_param_rel']:.1e}")
    assert r["max_rel_loss_diff"] < 1e-3, r


if __name__ == "__main__":
    for fn in [test_blocks_float64_exact, test_blocks_float32, test_real_checkpoints, test_attention_backends,
               test_exact_attention_gradcheck,
               test_training_trajectory]:
        print(fn.__name__)
        fn()
        print("  ok")
