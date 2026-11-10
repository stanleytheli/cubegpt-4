"""MoE router options (cubegpt/model.py MoEBlock(router=...), ModelConfig.router).

    .venv/bin/python tests/test_router.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubegpt import config as C  # noqa: E402
from cubegpt.config import ModelConfig  # noqa: E402
from cubegpt.model import MoEBlock, build_model, expert_stats  # noqa: E402
from cubegpt.verify import check_ok, compare_block  # noqa: E402

TINY = dict(arch="moe", d_model=32, n_heads=4, d_ffwd=48, n_layers=2, n_experts=4, n_active=1)


def test_softmax_fast_matches_reference():
    for kw in (dict(n_active=1), dict(n_active=2), dict(n_active=1, padded_max=0)):
        r = compare_block(n_experts=8, n_tokens=640, router="softmax", **kw)
        assert check_ok(r, 1e-9), (kw, r)


def test_softmax_weights_bounded():
    torch.manual_seed(0)
    b = MoEBlock(32, 48, 8, 1, router="softmax").train()
    x = torch.randn(1, 500, 32)
    out = b(x)
    assert float(b.score_mean) >= 1 / 8 - 1e-6           # top-1 softmax weight is >= 1/n_experts
    assert float(b.bias_override) == 0.0                 # zero bias: every token gets its top expert
    # the router setting changes the function computed by the same weights (hence the marker buffer)
    b2 = MoEBlock(32, 48, 8, 1, router="sigmoid").train()
    b2.load_state_dict({k: v for k, v in b.state_dict().items() if k != "router_softmax"})
    assert not torch.allclose(b2(x), out)                # same weights, different function


def test_checkpoint_router_mismatch_fails_loudly():
    soft = build_model(ModelConfig(**TINY, router="softmax"))
    sig = build_model(ModelConfig(**TINY))
    for src, dst in ((soft, sig), (sig, soft)):
        try:
            dst.load_state_dict(src.state_dict())
            raise AssertionError("router mismatch loaded silently")
        except RuntimeError as e:
            assert "router_softmax" in str(e)
    build_model(ModelConfig(**TINY, router="softmax")).load_state_dict(soft.state_dict())   # same router: fine


def test_router_config_rules():
    for bad in ({"router": "relu"}, {"arch": "gated", "router": "softmax"}):
        try:
            C.validate_model(ModelConfig(**{**TINY, **bad}))
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass


def test_softmax_training_run():
    from cubegpt.train import run_training
    root = tempfile.mkdtemp()
    d = {"name": "soft", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "model": {**TINY, "router": "softmax"},
         "stages": [{"steps": 400, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                     "warmup_samples": 64 * 40, "decay_samples": 64 * 100}]}
    r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    recs = [json.loads(l) for l in open(Path(root) / "runs/soft/metrics.jsonl")]
    m = recs[-1]["moe"]
    print(f"  eval {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}, router score min "
          f"{m['router_score_min']:.3f}, bias override {100 * m['bias_override_max']:.0f}%")
    assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"]
    assert m["router_score_min"] >= 0.2 and 0 <= m["bias_override_max"] < 1   # 1/4 floor with 4 experts
    final = torch.load(Path(root) / "runs/soft/final.pth", weights_only=True)
    assert any(k.endswith("router_softmax") for k in final)
    shutil.rmtree(root)


def test_sigmoid_stats_unchanged_shape():
    m = build_model(ModelConfig(**TINY)).train()
    m(torch.zeros(8, 20, 2, dtype=torch.long))
    s = expert_stats(m)
    assert {"router_score_min", "bias_override_max", "dead_experts"} <= set(s)
    assert not any(k.endswith("router_softmax") for k in m.state_dict())     # published checkpoints still load


def test_aux_balance_loss_balances():
    """The Switch balance loss is 1.0 at perfect balance and, trained alone, evens out a skewed router."""
    torch.manual_seed(0)
    b = MoEBlock(32, 48, 8, 1, router="softmax").train()
    b.aux_enabled = True
    x = torch.randn(1, 2000, 32)
    with torch.no_grad():
        b.gate.weight.zero_()
    b(x)
    assert abs(float(b.aux_balance.detach()) - 1.0) < 1e-5            # uniform probs -> exactly 1 whatever the load
    with torch.no_grad():
        b.gate.weight.normal_(0, 0.02)
        b.gate.weight[0] += 0.3 * x[0].mean(0) / x[0].mean(0).norm() ** 2 * 3   # expert 0 hogs tokens
    b(x)
    share0 = float(b.tokens_per_expert.max() / b.tokens_per_expert.sum())
    opt = torch.optim.Adam(b.gate.parameters(), lr=1e-2)
    for _ in range(200):
        b(x)
        opt.zero_grad()
        (b.aux_balance + 1e-3 * b.aux_z).backward()
        opt.step()
    b(x)
    share1 = float(b.tokens_per_expert.max() / b.tokens_per_expert.sum())
    print(f"  max expert share {share0:.2f} -> {share1:.2f}, aux balance {float(b.aux_balance.detach()):.3f}")
    assert share0 > 0.5 and share1 < 0.25 and float(b.aux_balance.detach()) < 1.1


def test_aux_loss_training_run():
    from cubegpt.train import run_training
    root = tempfile.mkdtemp()
    d = {"name": "aux", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "model": {**TINY, "router": "softmax"},
         "moe_aux_loss": 0.01, "moe_z_loss": 1e-3, "moe_bias_update_speed": 0.0,
         "stages": [{"steps": 400, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                     "warmup_samples": 64 * 40, "decay_samples": 64 * 100}]}
    r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    run = Path(root) / "runs/aux"
    recs = [json.loads(l) for l in open(run / "metrics.jsonl")]
    m = recs[-1]["moe"]
    final = torch.load(run / "final.pth", weights_only=True)
    biases = torch.cat([v for k, v in final.items() if k.endswith("expert_bias")])
    print(f"  eval {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}, aux {m['aux_balance']:.3f} "
          f"z {m['aux_z']:.3f}, max load {m['max_load_ratio']:.2f}x, bias override {m['bias_override_max']:.0%}")
    assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"]
    assert float(biases.abs().max()) == 0.0 and m["bias_override_max"] == 0.0   # bias balancing really off
    assert 1.0 <= m["aux_balance"] < 1.5 and m["aux_z"] > 0
    assert "aux balance loss 0.01, z-loss 0.001" in (run / "log.txt").read_text()
    for bad in ({"moe_aux_loss": -1}, {"moe_z_loss": 0.1, "model": {"arch": "gated"}}):
        try:
            C.from_dict({**d, **bad})
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass
    shutil.rmtree(root)


def test_first_dense_layer():
    from cubegpt.model import MLPBlockGated, count_params
    base = build_model(ModelConfig(**TINY))
    hyb = build_model(ModelConfig(**TINY, first_dense_layers=1))
    assert isinstance(hyb.stack[0].mlp_block, MLPBlockGated) and isinstance(hyb.stack[1].mlp_block, MoEBlock)
    expert = count_params(base.stack[0].mlp_block.experts[0])
    gate = count_params(base.stack[0].mlp_block.gate)
    assert count_params(base) - count_params(hyb) == 3 * expert + gate      # 4 experts + router -> 1 dense FFN
    assert any(k.startswith("stack.0.mlp_block.experts.0") for k in base.state_dict())   # default layout unchanged
    hyb.train()
    hyb(torch.zeros(8, 20, 2, dtype=torch.long)).sum().backward()
    hyb.update_expert_bias()                                                # skips the dense layer
    for src, dst in ((hyb, base), (base, hyb)):
        try:
            dst.load_state_dict(src.state_dict())
            raise AssertionError("dense/MoE layer mismatch loaded silently")
        except RuntimeError:
            pass
    for bad in ({"first_dense_layers": 2}, {"first_dense_layers": -1}, {"arch": "gated", "first_dense_layers": 1}):
        try:
            C.validate_model(ModelConfig(**{**TINY, **bad}))
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass
    from cubegpt.train import run_training
    root = tempfile.mkdtemp()
    d = {"name": "hyb", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "model": {**TINY, "first_dense_layers": 1},
         "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                     "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
    r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    recs = [json.loads(l) for l in open(Path(root) / "runs/hyb/metrics.jsonl")]
    assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"] and "moe" in recs[-1]
    shutil.rmtree(root)


def test_attention_only_layers_and_ffn_scale():
    from cubegpt.model import CubeTransformerLayerAttn, ScaledMLP, count_params
    X = torch.randint(0, 20, (8, 20, 2))
    base = build_model(ModelConfig(**TINY))
    n = TINY["n_layers"]
    half = "0" * (n // 2) + "1" * (n - n // 2)
    m = build_model(ModelConfig(**TINY, ffn_layers=half))
    assert isinstance(m.stack[0], CubeTransformerLayerAttn) and isinstance(m.stack[-1].mlp_block, MoEBlock)
    assert not hasattr(m.stack[0], "mlp_block") and not hasattr(m.stack[0], "norm2")
    layer = base.stack[0]
    dropped = count_params(layer.mlp_block) + count_params(layer.norm2)
    assert count_params(base) - count_params(m) == (n // 2) * dropped
    m.train()
    m(X).sum().backward()
    m.update_expert_bias()
    # attention-only everywhere (dense arch), and a mask of all ones is the plain model
    a = build_model(ModelConfig(**{**TINY, "arch": "gated"}, ffn_layers="0" * n))
    assert all(isinstance(l, CubeTransformerLayerAttn) for l in a.stack) and a(X).shape == (8, 21)
    ones = build_model(ModelConfig(**TINY, ffn_layers="1" * n))
    ones.load_state_dict(base.state_dict())
    # FFN-only layers: 2 attention-only layers, then 2 FFN-only layers
    from cubegpt.model import CubeTransformerLayerFFN
    for arch in ("moe", "gated"):
        f = build_model(ModelConfig(**{**TINY, "arch": arch, "n_layers": 4}, ffn_layers="00ff"))
        assert isinstance(f.stack[1], CubeTransformerLayerAttn) and isinstance(f.stack[2], CubeTransformerLayerFFN)
        assert not hasattr(f.stack[3], "mha_block") and not hasattr(f.stack[3], "norm1")
        ref = build_model(ModelConfig(**{**TINY, "arch": arch, "n_layers": 4}))
        per = count_params(ref.stack[0])
        assert count_params(ref) - count_params(f) == 4 * per - 2 * count_params(f.stack[0]) - 2 * count_params(f.stack[2])
        f.train()
        f(X).sum().backward()
        if arch == "moe":
            f.update_expert_bias()
            assert all(b.tokens_per_expert.sum() == X.shape[0] * 20 for b in f.modules() if isinstance(b, MoEBlock))
    # FFN scale: starts at the given value, is trained, and scale 1 computes the plain dense model
    g = build_model(ModelConfig(**{**TINY, "arch": "gated"}))
    sc = build_model(ModelConfig(**{**TINY, "arch": "gated"}, ffn_scale_init=0.1))
    assert isinstance(sc.stack[0].mlp_block, ScaledMLP) and abs(float(sc.stack[0].mlp_block.scale[0].detach()) - 0.1) < 1e-7
    sc(X).sum().backward()
    assert sc.stack[0].mlp_block.scale.grad.abs().sum() > 0
    one = build_model(ModelConfig(**{**TINY, "arch": "gated"}, ffn_scale_init=1.0))
    one.load_state_dict({k.replace("mlp_block.", "mlp_block.mlp."): v for k, v in g.state_dict().items()}, strict=False)
    assert torch.allclose(one(X), g(X), atol=1e-6)
    try:
        sc.load_state_dict(g.state_dict())
        raise AssertionError("scaled/plain mismatch loaded silently")
    except RuntimeError:
        pass
    for bad in ({"ffn_layers": "011"}, {"ffn_layers": "0" * n}, {"ffn_layers": "2" * n}, {"ffn_layers": 11},
                {"ffn_scale_init": 0.1}, {"arch": "gated", "ffn_scale_init": -1.0},
                {"arch": "transformer", "ffn_layers": "1" * n}):
        try:
            C.validate_model(ModelConfig(**{**TINY, **bad}))
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass
    C.validate_model(ModelConfig(**{**TINY, "arch": "gated"}, ffn_layers="0" * n, ffn_scale_init=0.1))
    d0 = {"name": "x", "model": dict(TINY), "stages": [{"steps": 10, "batch_size": 64, "lr": 1e-3}]}
    ov = C.apply_overrides(d0, 'model.ffn_layers="%s"' % half)
    assert ov["model"]["ffn_layers"] == half, ov["model"]
    from cubegpt.train import run_training
    for name, model in (("attn", {**TINY, "arch": "gated", "ffn_layers": "0" * n}),
                        ("alt", {**TINY, "ffn_layers": ("01" * n)[:n]}),
                        ("tail", {**TINY, "n_layers": 4, "ffn_layers": "00ff"}),
                        ("scale", {**TINY, "arch": "gated", "ffn_scale_init": 0.1})):
        root = tempfile.mkdtemp()
        d = {"name": name, "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, "model": model,
             "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                         "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
        r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
        recs = [json.loads(l) for l in open(Path(root) / f"runs/{name}/metrics.jsonl")]
        assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"], name
        shutil.rmtree(root)


def test_grad_clip():
    from cubegpt.train import run_training
    out = {}
    for clip in (0.0, 1.0, 1e-3):
        root = tempfile.mkdtemp()
        d = {"name": "clip", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, "grad_clip": clip, "model": {**TINY, "arch": "gated"},
             "stages": [{"steps": 200, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                         "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
        r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
        recs = [json.loads(l) for l in open(Path(root) / "runs/clip/metrics.jsonl")]
        assert r["status"] == "completed" and recs[-1]["eval_loss"] < 0.3
        assert ("grad_norm_max" in recs[-1]) == (clip > 0)
        if clip:
            assert 0 < recs[-1]["grad_norm_mean"] <= recs[-1]["grad_norm_max"]
        out[clip] = recs[-1]["eval_loss"]
        shutil.rmtree(root)
    assert out[0.0] != out[1e-3]            # a clip far below the usual norm changes the trajectory
    print(f"  final eval: no clip {out[0.0]:.4f}, clip 1.0 {out[1.0]:.4f}, clip 1e-3 {out[1e-3]:.4f}")
    try:
        C.from_dict({"name": "x", "grad_clip": -1.0, "model": dict(TINY), "stages": [{"steps": 10, "batch_size": 64, "lr": 1e-3}]})
        raise AssertionError("accepted negative grad_clip")
    except C.ConfigError:
        pass


def test_pre_norm_and_adam_options():
    from cubegpt.train import param_groups, run_training
    X = torch.randint(0, 20, (8, 20, 2))
    for cfg in ({**TINY}, {**TINY, "arch": "gated"}, {**TINY, "arch": "gated", "n_layers": 4, "ffn_layers": "00ff"}):
        post, pre = build_model(ModelConfig(**cfg)), build_model(ModelConfig(**cfg, pre_norm=True))
        assert all(l.pre_norm for l in pre.stack) and not any(l.pre_norm for l in post.stack)
        sd = {k: v for k, v in pre.state_dict().items() if k != "pre_norm_marker"}
        assert set(sd) == set(post.state_dict())                       # same parameters, different wiring
        post.load_state_dict(sd)
        assert not torch.allclose(post(X), pre(X), atol=1e-4)
        for src, dst in ((pre, post), (post, pre)):
            try:
                dst.load_state_dict(src.state_dict())
                raise AssertionError("pre/post-norm mismatch loaded silently")
            except RuntimeError:
                pass
        # pre-norm: a layer whose branch outputs zero is the identity (the stream is not re-normalized)
        layer = pre.stack[0]
        h = torch.randn(2, 20, cfg["d_model"])
        hooks = [m.register_forward_hook(lambda mod, i, o: torch.zeros_like(o))
                 for m in (getattr(layer, "mha_block", None), getattr(layer, "mlp_block", None)) if m is not None]
        assert torch.equal(layer(h), h)
        for hk in hooks:
            hk.remove()
    try:
        C.validate_model(ModelConfig(**{**TINY, "arch": "transformer"}, pre_norm=True))
        raise AssertionError("accepted pre_norm for arch transformer")
    except C.ConfigError:
        pass
    m = build_model(ModelConfig(**TINY))
    g = param_groups(m, 1.0, 0.1)
    assert [x["weight_decay"] for x in g] == [0.1, 0.0]
    assert all(p.ndim >= 2 for p in g[0]["params"]) and all(p.ndim < 2 for p in g[1]["params"])
    assert sum(len(x["params"]) for x in g) == len(list(m.parameters()))
    g = param_groups(m, 0.1, 0.1)
    assert [(x["lr_mult"], x["weight_decay"]) for x in g] == [(1.0, 0.1), (1.0, 0.0), (0.1, 0.1)]
    assert len(param_groups(m)) == 1 and "weight_decay" not in param_groups(m)[0]
    for name, extra in (("pre", {"model": {**TINY, "arch": "gated", "pre_norm": True}}),
                        ("b2", {"model": {**TINY, "arch": "gated"}, "adam_beta2": 0.95}),
                        ("adamw", {"model": dict(TINY), "adam_beta2": 0.95, "weight_decay": 0.1, "router_lr_mult": 0.1})):
        root = tempfile.mkdtemp()
        d = {"name": name, "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, **extra,
             "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                         "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
        r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
        recs = [json.loads(l) for l in open(Path(root) / f"runs/{name}/metrics.jsonl")]
        assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"], name
        print(f"  {name}: eval {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}")
        shutil.rmtree(root)
    for bad in ({"adam_beta2": 1.0}, {"weight_decay": -0.1}, {"adam_eps": 0.0}):
        try:
            C.from_dict({"name": "x", **bad, "model": dict(TINY), "stages": [{"steps": 10, "batch_size": 64, "lr": 1e-3}]})
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass


def test_muon():
    from cubegpt.muon import MuonAdam, muon_params, orthogonalize
    from cubegpt.train import run_training
    torch.manual_seed(0)
    for shape in ((5, 32, 32), (3, 48, 32), (3, 32, 48)):
        sv = torch.linalg.svdvals(orthogonalize(torch.randn(shape) * 7.0))
        assert sv.min() > 0.3 and sv.max() < 1.3, (shape, sv.min(), sv.max())   # singular values pushed to ~1
    m = build_model(ModelConfig(**{**TINY, "arch": "gated", "n_layers": 4, "ffn_layers": "00ff"}, pre_norm=True))
    mu = muon_params(m)
    names = {n for n, p in m.named_parameters() if any(p is q for q in mu)}
    assert len(mu) == 2 * 4 + 2 * 3 and all(n.startswith("stack.") and n.endswith(".weight") for n in names)
    rest = [p for p in m.parameters() if not any(p is q for q in mu)]
    opt = MuonAdam([{"params": rest}, {"params": mu, "muon": True}], lr=1e-3)
    before = [p.detach().clone() for p in mu]
    m(torch.randint(0, 20, (8, 20, 2))).sum().backward()
    opt.step()
    for b, p in zip(before, mu):   # the step size doesn't depend on the gradient's scale: spectral norm ~ lr
        s = torch.linalg.matrix_norm(p - b, ord=2).item() / max(1.0, p.shape[0] / p.shape[1]) ** 0.5
        assert 2e-4 < s < 1.3e-3, s
    opt2 = MuonAdam([{"params": rest}, {"params": mu, "muon": True}], lr=1e-3)
    opt2.load_state_dict(opt.state_dict())
    assert torch.equal(opt2.state[mu[0]]["buf"], opt.state[mu[0]]["buf"]) and opt2.adam.state_dict()["state"]
    for name, extra in (("muon", {}), ("muon-wd", {"weight_decay": 0.1, "adam_beta2": 0.95, "muon_weight_decay": 0.01})):
        root = tempfile.mkdtemp()
        d = {"name": name, "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, "optimizer": "muon", "muon_lr_mult": 10.0, **extra,
             "model": {**TINY, "arch": "gated", "pre_norm": True},
             "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                         "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
        r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
        recs = [json.loads(l) for l in open(Path(root) / f"runs/{name}/metrics.jsonl")]
        assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"], name
        print(f"  {name}: eval {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}")
        shutil.rmtree(root)
    for bad in ({"optimizer": "sgd"}, {"muon_lr_mult": 0.0}, {"muon_momentum": 1.0}):
        try:
            C.from_dict({"name": "x", **bad, "model": dict(TINY), "stages": [{"steps": 10, "batch_size": 64, "lr": 1e-3}]})
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass


if __name__ == "__main__":
    for fn in [test_softmax_fast_matches_reference, test_softmax_weights_bounded,
               test_checkpoint_router_mismatch_fails_loudly, test_router_config_rules, test_softmax_training_run,
               test_sigmoid_stats_unchanged_shape, test_aux_balance_loss_balances, test_aux_loss_training_run, test_first_dense_layer,
               test_attention_only_layers_and_ffn_scale, test_grad_clip,
               test_pre_norm_and_adam_options, test_muon]:
        print(fn.__name__)
        fn()
        print("  ok")
