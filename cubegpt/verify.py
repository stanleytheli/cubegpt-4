"""Equivalence checks for the fast ("grouped") MoE dispatch against the reference
("loop") implementation, plus training-throughput benchmarks.

Used by tests/test_moe.py (CPU) and `modal run modal_app.py::bench` (GPU).
"""
from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn

from .cube import CubeTables, generate_samples
from .evaluate import beam_search_gpu, compute_metrics, scramble_for
from .model import MoEBlock, build_model, set_moe_impl


def _max_rel(a, b):
    """max |a-b| / max |a| (0 if both are all-zero)."""
    scale = a.abs().max().item()
    diff = (a - b).abs().max().item()
    return diff / scale if scale > 0 else diff


def _pair(module, ref_impl="loop", fast_impl="grouped"):
    ref = copy.deepcopy(module)
    fast = copy.deepcopy(module)
    set_moe_impl(ref, ref_impl)
    set_moe_impl(fast, fast_impl)
    return ref, fast


def compare_block(n_experts, n_active, n_tokens, d=32, f=48, gated=True, bias=True, expert_bias=None,
                  padded_max=None, dtype=torch.float64, device="cpu", seed=0, router="sigmoid"):
    """One MoEBlock: forward output, input/parameter gradients and recorded load,
    reference vs fast. expert_bias (list) forces (im)balanced routing."""
    torch.manual_seed(seed)
    block = MoEBlock(d, f, n_experts, n_active, bias=bias, gated=gated, router=router).to(device, dtype)
    if expert_bias is not None:
        block.expert_bias.copy_(torch.tensor(expert_bias, dtype=dtype))
    if padded_max is not None:
        block.PADDED_MAX_ASSIGNMENTS = padded_max
    block.train()
    ref, fast = _pair(block)
    x = torch.randn(1, n_tokens, d, device=device, dtype=dtype)
    g_out = torch.randn(1, n_tokens, d, device=device, dtype=dtype)
    res = {}
    for name, m in (("ref", ref), ("fast", fast)):
        xi = x.clone().requires_grad_(True)
        out = m(xi)
        out.backward(g_out)
        res[name] = (out.detach(), xi.grad, {k: p.grad for k, p in m.named_parameters()}, m.tokens_per_expert.clone())
    (o1, gx1, gp1, l1), (o2, gx2, gp2, l2) = res["ref"], res["fast"]
    cap = int(l1.max())
    n_used = int((l1 > 0).sum())
    return {
        "path": ("batched" if n_tokens * n_active <= block.PADDED_MAX_ASSIGNMENTS
                 and n_used * cap <= 2 * n_tokens * n_active else "sliced"),
        "dead_experts": int((l1 == 0).sum()),
        "out_rel": _max_rel(o1, o2), "grad_x_rel": _max_rel(gx1, gx2),
        "grad_param_rel": _grads_rel(list(gp1.values()), list(gp2.values())),
        "none_pattern_equal": _none_pattern(gp1.values()) == _none_pattern(gp2.values()),
        "load_equal": bool(torch.equal(l1, l2)),
    }


def _none_pattern(grads):
    return [g is None for g in grads]


def _grads_rel(g1, g2):
    """max |g1 - g2| over all parameters, relative to the largest gradient anywhere.
    (Per-parameter relative error is meaningless for parameters whose true gradient
    is ~0, e.g. attention key biases, where both sides are pure rounding noise.)"""
    scale = max((a.abs().max().item() for a in g1 if a is not None), default=0.0)
    diff = max(((a - b).abs().max().item() for a, b in zip(g1, g2) if a is not None and b is not None),
               default=0.0)
    return diff / scale if scale > 0 else diff


BLOCK_CASES = [
    # (label, kwargs)
    ("8 experts top-1, balanced", dict(n_experts=8, n_active=1, n_tokens=640)),
    ("8 experts top-2", dict(n_experts=8, n_active=2, n_tokens=640)),
    ("16 experts top-2, 37 tokens", dict(n_experts=16, n_active=2, n_tokens=37)),
    ("1 expert", dict(n_experts=1, n_active=1, n_tokens=50)),
    ("single token", dict(n_experts=4, n_active=1, n_tokens=1)),
    ("3 of 8 experts dead (bias)", dict(n_experts=8, n_active=1, n_tokens=640,
                                        expert_bias=[0, 0, 0, -10, -10, -10, 0, 0])),
    ("all tokens to one expert", dict(n_experts=8, n_active=1, n_tokens=640,
                                      expert_bias=[0, 0, 0, 10, 0, 0, 0, 0])),
    ("top-2 with 6 dead experts", dict(n_experts=8, n_active=2, n_tokens=640,
                                       expert_bias=[10, -10, -10, -10, 10, -10, -10, -10])),
    ("one dominant expert, rest alive", dict(n_experts=8, n_active=1, n_tokens=640,
                                             expert_bias=[0.45, 0, 0, 0, 0, 0, 0, 0])),
    ("balanced, large-batch path", dict(n_experts=8, n_active=1, n_tokens=640, padded_max=0)),
    ("top-2, large-batch path", dict(n_experts=8, n_active=2, n_tokens=640, padded_max=0)),
    ("ungated experts", dict(n_experts=8, n_active=1, n_tokens=640, gated=False)),
    ("experts without bias", dict(n_experts=8, n_active=2, n_tokens=640, bias=False)),
    ("softmax router top-1", dict(n_experts=8, n_active=1, n_tokens=640, router="softmax")),
    ("softmax router top-2", dict(n_experts=8, n_active=2, n_tokens=640, router="softmax")),
    ("softmax router, large-batch path", dict(n_experts=8, n_active=1, n_tokens=640, router="softmax",
                                              padded_max=0)),
]


def run_block_suite(dtype=torch.float64, device="cpu"):
    return [(label, compare_block(dtype=dtype, device=device, **kw)) for label, kw in BLOCK_CASES]


def compare_model(model, X, y):
    """Full model fwd+bwd on a batch, reference vs fast (same weights)."""
    model.train()
    ref, fast = _pair(model)
    out = {}
    for name, m in (("ref", ref), ("fast", fast)):
        m.zero_grad(set_to_none=True)
        logits = m(X)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        out[name] = (logits.detach(), loss.item(), [p.grad.clone() if p.grad is not None else None
                                                    for p in m.parameters()],
                     [b.tokens_per_expert.clone() for b in m.modules() if isinstance(b, MoEBlock)])
    (lg1, ls1, g1, t1), (lg2, ls2, g2, t2) = out["ref"], out["fast"]
    return {"logits_rel": _max_rel(lg1, lg2), "loss_ref": ls1, "loss_fast": ls2, "grad_rel": _grads_rel(g1, g2),
            "none_pattern_equal": _none_pattern(g1) == _none_pattern(g2),
            "routing_equal": all(torch.equal(a, b) for a, b in zip(t1, t2)),
            "routing_moved": sum(int((a - b).abs().sum().item()) // 2 for a, b in zip(t1, t2)),
            "routing_decisions": sum(int(a.sum().item()) for a in t1)}


@torch.no_grad()
def tf32_noise(model, X):
    """Calibration: the reference implementation against itself, TF32 on vs off.
    Any real-weights comparison under TF32 should be judged against this."""
    prev = torch.get_float32_matmul_precision()
    outs = []
    for prec in ("high", "highest"):
        torch.set_float32_matmul_precision(prec)
        m = set_moe_impl(copy.deepcopy(model), "loop").train()
        outs.append((m(X), [b.tokens_per_expert.clone() for b in m.modules() if isinstance(b, MoEBlock)]))
    torch.set_float32_matmul_precision(prev)
    (o1, t1), (o2, t2) = outs
    return {"logits_rel": _max_rel(o1, o2), "routing_moved": sum(int((a - b).abs().sum().item()) // 2
                                                                 for a, b in zip(t1, t2))}


def compare_attention(model, X, y):
    """Full model fwd+bwd with attention backend "sdpa" vs "explicit" (same weights)."""
    from .model import MultiHeadAttention
    model.train()
    res = {}
    for backend in ("sdpa", "explicit"):
        m = copy.deepcopy(model)
        for mod in m.modules():
            if isinstance(mod, MultiHeadAttention):
                mod.backend = backend
        logits = m(X)
        nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
        res[backend] = (logits.detach(), [p.grad for p in m.parameters()])
    (l1, g1), (l2, g2) = res["sdpa"], res["explicit"]
    return {"logits_rel": _max_rel(l1, l2), "grad_rel": _grads_rel(g1, g2)}


@torch.no_grad()
def attention_vs_truth(model, X):
    """Max logit error of each attention backend vs a float64 forward pass of the same
    model, with and without TF32. (Comparing two TF32 runs against each other only
    measures TF32 rounding noise: a 1e-7 perturbation anywhere flips TF32 roundings
    downstream, so they disagree by about TF32's own error.)"""
    from .model import MultiHeadAttention

    def set_backend(m, b):
        for mod in m.modules():
            if isinstance(mod, MultiHeadAttention):
                mod.backend = b
        return m

    model = copy.deepcopy(model).eval()
    truth = set_backend(copy.deepcopy(model).double(), "explicit")(X)
    scale = truth.abs().max().item()
    prev = torch.get_float32_matmul_precision()
    out = {}
    for prec in ("highest", "high"):
        torch.set_float32_matmul_precision(prec)
        for b in ("sdpa", "explicit"):
            err = (set_backend(model, b)(X).double() - truth).abs().max().item() / scale
            out[f"{'fp32' if prec == 'highest' else 'tf32'}_{b}"] = err
    torch.set_float32_matmul_precision(prev)
    return out


def compare_metrics(model, T, n=65536):
    ref, fast = _pair(model)
    return compute_metrics(ref, T, n), compute_metrics(fast, T, n)


def compare_training(model_cfg, T, steps=300, batch_size=1024, lr=1e-3, seed=0):
    """Train reference and fast copies from the same init on the same batches.
    Control: the reference trained on a different data stream, to show how big
    ordinary run-to-run variation is compared with the implementation difference."""
    torch.manual_seed(seed)
    base = build_model(model_cfg).to(T.device)

    def make_data(s):
        with torch.random.fork_rng(devices=[T.device] if torch.device(T.device).type == "cuda" else []):
            torch.manual_seed(s)
            return [generate_samples(T, batch_size) for _ in range(steps)]

    data = make_data(seed + 1)
    losses = {}
    ref, fast = _pair(base)
    control = copy.deepcopy(ref)
    models = {"ref": ref, "fast": fast}
    for name, m, d in (("ref", ref, data), ("fast", fast, data), ("control", control, None)):
        d = d if d is not None else make_data(seed + 2)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        ls = []
        for X, y in d:
            loss = nn.functional.binary_cross_entropy_with_logits(m(X), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.update_expert_bias()
            ls.append(loss.item())
        losses[name] = ls
    p_rel = max(_max_rel(a.detach(), b.detach())
                for a, b in zip(models["ref"].parameters(), models["fast"].parameters()))
    rel = [abs(a - b) / abs(a) for a, b in zip(losses["ref"], losses["fast"])]
    k = max(1, steps // 10)
    return {"steps": steps, "batch_size": batch_size, "max_rel_loss_diff": max(rel),
            "final_loss_ref": sum(losses["ref"][-k:]) / k, "final_loss_fast": sum(losses["fast"][-k:]) / k,
            "final_loss_control_other_data": sum(losses["control"][-k:]) / k,
            "final_param_rel": p_rel,
            "loss_curve_ref": losses["ref"][::k], "loss_curve_fast": losses["fast"][::k]}


def compare_solves(model, T, n_cubes=3, width=1000, seed=0):
    ref, fast = _pair(model)
    out = []
    for i in range(n_cubes):
        start = scramble_for(seed, i, 30)
        _, m1 = beam_search_gpu(T, ref, start, width=width)
        _, m2 = beam_search_gpu(T, fast, start, width=width)
        out.append({"cube": i, "ref_moves": len(m1) if m1 else None, "fast_moves": len(m2) if m2 else None,
                    "identical": m1 == m2})
    return out


def throughput(model_cfg, T, batch_size, impl="grouped", seconds=10.0, warmup=5, lr=1e-4):
    """Training-step throughput (fwd+bwd+Adam+scheduler+bias update, data generation
    amortized like the trainer) and peak memory at a given batch size."""
    torch.manual_seed(0)
    model = set_moe_impl(build_model(model_cfg).to(T.device), impl)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10 ** 9)
    loss_fn = nn.BCEWithLogitsLoss()
    is_moe = hasattr(model, "update_expert_bias")
    cuda = torch.device(T.device).type == "cuda"
    chunk_steps = max(1, (1 << 20) // batch_size)
    X_all, y_all = generate_samples(T, chunk_steps * batch_size)

    def step(i):
        j = i % chunk_steps
        X, y = X_all[j * batch_size:(j + 1) * batch_size], y_all[j * batch_size:(j + 1) * batch_size]
        loss = loss_fn(model(X), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step(loss.item())
        if is_moe:
            model.update_expert_bias()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup):
        step(i)
    if cuda:
        torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        step(n)
        n += 1
    if cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0
    # data generation cost, amortized per step
    t1 = time.time()
    generate_samples(T, chunk_steps * batch_size)
    if cuda:
        torch.cuda.synchronize()
    gen_per_step = (time.time() - t1) / chunk_steps
    step_s = dt / n + gen_per_step
    return {"impl": impl, "batch_size": batch_size, "steps": n, "ms_per_step": 1000 * step_s,
            "samples_per_sec": batch_size / step_s,
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None}


def fmt_rel(x):
    return "0" if x == 0 else f"{x:.1e}"


def check_ok(r, tol):
    return (r["out_rel"] <= tol and r["grad_x_rel"] <= tol and r["grad_param_rel"] <= tol and r["load_equal"]
            and r["none_pattern_equal"])


def run_bench(spec, vol_root=".", device=None, log=print):
    """GPU verification + throughput sweep (modal_app.py::bench). spec keys: model,
    batch_sizes, impls, seconds, verify, checkpoint (volume-relative or ""), tf32."""
    from pathlib import Path

    from .config import ModelConfig, validate_model
    from .cube import check_gpu_pipeline
    from .model import load_weights

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    mcfg = ModelConfig(**spec["model"])
    validate_model(mcfg)
    T = CubeTables(device)
    check_gpu_pipeline(T)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else device
    out = {"gpu": gpu, "torch": torch.__version__, "model": spec["model"], "verify": {}, "throughput": []}
    log(f"{gpu} | torch {torch.__version__} | model {mcfg}")

    def set_tf32(on):
        torch.set_float32_matmul_precision("high" if on else "highest")

    if spec.get("verify", True):
        v = out["verify"]
        set_tf32(False)
        for dtype in (torch.float64, torch.float32):
            rows = run_block_suite(dtype, device)
            tol = 1e-12 if dtype == torch.float64 else 1e-5
            v[f"blocks_{str(dtype)[6:]}"] = [dict(r, case=label, ok=check_ok(r, tol)) for label, r in rows]
            log(f"MoE block cases ({str(dtype)[6:]}, tf32 off): "
                f"{sum(check_ok(r, tol) for _, r in rows)}/{len(rows)} within {tol:g}; worst out "
                f"{fmt_rel(max(r['out_rel'] for _, r in rows))}, worst dW {fmt_rel(max(r['grad_param_rel'] for _, r in rows))}")
        torch.manual_seed(0)
        X, y = generate_samples(T, 4096)
        torch.manual_seed(0)
        rand_model = build_model(mcfg).to(device)
        for tf32 in (False, True):
            set_tf32(tf32)
            r = compare_model(rand_model, X, y)
            v[f"model_random_init_tf32_{tf32}"] = r
            log(f"full model, random init, tf32 {'on ' if tf32 else 'off'}: logits {fmt_rel(r['logits_rel'])} "
                f"grads {fmt_rel(r['grad_rel'])} routing_equal={r['routing_equal']} "
                f"none_pattern_equal={r['none_pattern_equal']}")
        r = attention_vs_truth(rand_model, X)
        v["attention_vs_float64"] = r
        log("attention, max logit error vs float64 (random init): " + ", ".join(f"{k} {v_:.1e}" for k, v_ in r.items())
            + "  [explicit and sdpa should match within each precision]")
        set_tf32(spec.get("tf32", True))
        if spec.get("checkpoint"):
            ck = build_model(mcfg).to(device)
            ck.load_state_dict(load_weights(Path(vol_root) / spec["checkpoint"], device))
            log(f"checkpoint {spec['checkpoint']}:")
            for n in (512, 4096):   # 512 -> padded dispatch path, 4096 -> sliced path
                for tf32 in (False, True):
                    set_tf32(tf32)
                    r = compare_model(ck, X[:n], y[:n])
                    v[f"checkpoint_model_batch{n}_tf32_{tf32}"] = r
                    log(f"  batch {n:>4}, tf32 {'on ' if tf32 else 'off'}: logits {fmt_rel(r['logits_rel'])} "
                        f"grads {fmt_rel(r['grad_rel'])} routed differently {r['routing_moved']}/"
                        f"{r['routing_decisions']} none_pattern_equal={r['none_pattern_equal']}"
                        + ("   (should be exact)" if not tf32 else ""))
                cal = tf32_noise(ck, X[:n])
                v[f"checkpoint_tf32_noise_batch{n}"] = cal
                log(f"  batch {n:>4}, calibration, reference TF32 on vs off: logits {fmt_rel(cal['logits_rel'])} "
                    f"routed differently {cal['routing_moved']}   (tf32-on rows above should be within this)")
            set_tf32(spec.get("tf32", True))
            m1, m2 = compare_metrics(ck, T)
            v["checkpoint_metrics"] = {"ref": m1, "fast": m2}
            log(f"  metrics ref : MSE {m1['mse']:.5f} W3A {100 * m1['within3']:.3f}% acc {100 * m1['rounded_acc']:.3f}%")
            log(f"  metrics fast: MSE {m2['mse']:.5f} W3A {100 * m2['within3']:.3f}% acc {100 * m2['rounded_acc']:.3f}%")
            s = compare_solves(ck, T)
            v["checkpoint_solves"] = s
            log(f"  beam search (width 1000), identical move sequences: {[x['identical'] for x in s]} "
                f"lengths {[(x['ref_moves'], x['fast_moves']) for x in s]}")
        r = compare_training(mcfg, T, steps=300, batch_size=1024, lr=1e-4)
        v["training"] = r
        log(f"training 300 steps x 1024 (tf32 {spec.get('tf32', True)}): max rel loss diff "
            f"{r['max_rel_loss_diff']:.1e}, final loss {r['final_loss_ref']:.6f} (ref) vs {r['final_loss_fast']:.6f} (fast); "
            f"control (ref, other data stream) {r['final_loss_control_other_data']:.6f}")

    set_tf32(spec.get("tf32", True))
    for bs in spec["batch_sizes"]:
        for impl in spec["impls"]:
            try:
                r = throughput(mcfg, T, bs, impl, seconds=spec.get("seconds", 10.0))
            except torch.OutOfMemoryError:
                r = {"impl": impl, "batch_size": bs, "oom": True}
            if device == "cuda":
                torch.cuda.empty_cache()
            out["throughput"].append(r)
            log(f"batch {bs:>6} {impl:8s}: " + ("OOM" if r.get("oom") else
                f"{r['ms_per_step']:8.1f} ms/step  {r['samples_per_sec']:>10,.0f} samples/s  "
                f"peak {r['peak_mem_gib'] or 0:.1f} GiB"))
    # extra models timed in the same container (same GPU and host), for like-for-like speed comparisons
    from .model import count_params
    out["compare"] = []
    for extra in spec.get("compare_models", []):
        ecfg = ModelConfig(**extra)
        validate_model(ecfg)
        n_params = count_params(build_model(ecfg))
        for bs in spec["batch_sizes"]:
            try:
                r = throughput(ecfg, T, bs, "grouped", seconds=spec.get("seconds", 10.0))
            except torch.OutOfMemoryError:
                r = {"batch_size": bs, "oom": True}
            if device == "cuda":
                torch.cuda.empty_cache()
            r.update(model=extra, params=n_params)
            out["compare"].append(r)
            log(f"compare {ecfg} ({n_params:,} params) batch {bs}: " + ("OOM" if r.get("oom") else
                f"{r['samples_per_sec']:,.0f} samples/s"))
    return out
