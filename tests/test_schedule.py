"""EvalPlateau LR detector (cubegpt/schedule.py).

    .venv/bin/python tests/test_schedule.py
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubegpt.schedule import EvalPlateau, lr_factor  # noqa: E402

EVAL = 256_000          # samples between evals
W = 20                  # window in evals


def make(lr=1e-4, threshold=0.005, min_lr=0.0):
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    return EvalPlateau(opt, 0.5, W, threshold, min_lr)


def run(sched, losses):
    """Feed eval losses; return the eval indices at which the LR was decayed."""
    return [i for i, loss in enumerate(losses) if sched.observe((i + 1) * EVAL, loss)]


def test_trend_exact():
    pts = [(x, 1.0 - 0.01 * x) for x in range(11)]           # loss 1.0 -> 0.9, mean 0.95
    assert abs(EvalPlateau.trend(pts) - 0.1 / 0.95) < 1e-12
    assert EvalPlateau.trend([(x, 0.5) for x in range(5)]) == 0.0


def test_improving_never_decays():
    # 2% relative improvement per window (threshold 0.5%), plus 0.013% noise as on the fixed eval set
    rng = random.Random(0)
    per_eval = 0.02 / W
    losses = [0.15 * (1 - per_eval) ** i * (1 + 1.3e-4 * rng.gauss(0, 1)) for i in range(200)]
    s = make()
    assert run(s, losses) == [], "decayed while the loss was still improving 2% per window"


def test_flat_decays_once_per_window():
    rng = random.Random(1)
    losses = [0.13 * (1 + 1.3e-4 * rng.gauss(0, 1)) for i in range(100)]
    s = make()
    decays = run(s, losses)
    # first decision only once a full window exists, then a full fresh window after each decay
    assert decays == [W - 1, 2 * W - 1, 3 * W - 1, 4 * W - 1, 5 * W - 1], decays
    assert abs(s.lr - 1e-4 * 0.5 ** 5) < 1e-18


def test_noisy_eval_needs_real_signal():
    # with original-notebook-style 2% eval noise, a genuinely improving run (5%/window) must not decay
    rng = random.Random(2)
    per_eval = 0.05 / W
    losses = [0.15 * (1 - per_eval) ** i * (1 + 0.004 * rng.gauss(0, 1)) for i in range(100)]
    assert run(make(), losses) == []


def test_min_lr_floor():
    s = make(lr=1e-6, min_lr=1e-7)
    run(s, [0.13] * 200)
    assert s.lr == 1e-7 and s.n_reductions == 4, (s.lr, s.n_reductions)   # 1e-6 -> 5e-7 -> ... -> 1.25e-7 -> 1e-7


def test_state_dict_resume_identical():
    rng = random.Random(3)
    losses = [0.14 * (1 - 0.0002 * i) ** 0.3 * (1 + 1e-4 * rng.gauss(0, 1)) for i in range(150)]
    a = make()
    full = run(a, losses)
    b = make()
    first = [i for i, l in enumerate(losses[:67]) if b.observe((i + 1) * EVAL, l)]
    state, lr = b.state_dict(), b.lr
    c = make(lr=lr)
    c.load_state_dict(state)
    rest = [i for i, l in enumerate(losses[67:], start=67) if c.observe((i + 1) * EVAL, l)]
    assert first + rest == full and c.lr == a.lr


def test_trainer_end_to_end(tmp_root=None):
    """Tiny CPU run with eval_plateau: decays happen, are logged, and survive a resume."""
    import json
    import shutil
    import tempfile

    from cubegpt import config as C
    from cubegpt.train import RetryableInterrupt, run_training
    root = tmp_root or tempfile.mkdtemp()
    d = {"name": "sched", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "checkpoint_every_minutes": 0.02, "timeout_hours": 0.004,
         "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4,
                   "n_active": 1},
         "stages": [{"steps": 3000, "batch_size": 64, "lr": 3e-3, "eval_every": 20, "schedule": "eval_plateau",
                     "plateau_window_samples": 64 * 20 * 5, "plateau_threshold": 0.02, "plateau_min_lr": 1e-5}]}
    cfg = C.to_dict(C.from_dict(d))
    attempts = 0
    while True:
        attempts += 1
        try:
            r = run_training(cfg, root, lambda: None, "fc", device="cpu")
            break
        except RetryableInterrupt:
            assert attempts < 10
    log = open(Path(root) / "runs/sched/log.txt").read()
    recs = [json.loads(l) for l in open(Path(root) / "runs/sched/metrics.jsonl")]
    lrs = [r["lr"] for r in recs]
    print(f"  attempts {attempts}, lr decays logged: {log.count('plateau: eval loss improved')}, "
          f"lr {lrs[0]:.2g} -> {lrs[-1]:.2g}, monotone non-increasing: {all(a >= b for a, b in zip(lrs, lrs[1:]))}")
    assert r["status"] == "completed" and attempts >= 2, "test should exercise a resume"
    assert log.count("plateau: eval loss improved") >= 1
    assert all(a >= b for a, b in zip(lrs, lrs[1:])) and min(lrs) >= 1e-5
    if tmp_root is None:
        shutil.rmtree(root)


def test_wsd_shape():
    f = [lr_factor("wsd", i, 100, warmup_steps=10, decay_steps=20) for i in range(100)]
    assert abs(f[0] - 0.1) < 1e-12 and f[9] == 1.0                     # warmup reaches the peak
    assert all(x == 1.0 for x in f[10:80])                             # stable
    assert abs(f[80] - 0.95) < 1e-12 and f[99] == 0.0                  # linear decay to 0 at the end
    assert all(a >= b for a, b in zip(f[9:], f[10:]))
    g = [lr_factor("wsd", i, 100, decay_steps=20, final_fraction=0.1) for i in range(100)]
    assert g[0] == 1.0 and abs(g[99] - 0.1) < 1e-12                    # no warmup; floor respected


def test_cosine_shape():
    f = [lr_factor("cosine", i, 110, warmup_steps=10, final_fraction=0.0) for i in range(110)]
    assert f[9] == 1.0 and abs(f[59] - 0.5) < 1e-12 and abs(f[109]) < 1e-12
    assert all(a >= b for a, b in zip(f[9:], f[10:]))


def test_fixed_schedule_validation():
    from cubegpt import config as C
    base = {"name": "v", "model": {}, "stages": [{"samples": 25_600_000, "batch_size": 1024, "lr": 4e-4,
                                                   "schedule": "wsd", "warmup_samples": 1_280_000,
                                                   "decay_samples": 5_120_000}]}
    cfg = C.from_dict(base)
    assert C.stage_warmup_steps(cfg.stages[0]) == 1250 and C.stage_decay_steps(cfg.stages[0]) == 5000
    bad = [({"decay_samples": 25_000_000}, "shorter than the stage"),
           ({"schedule": "cosine"}, "decay_samples is for wsd"),
           ({"schedule": "eval_plateau"}, "only apply to"),
           ({"final_lr_fraction": 1.0}, "final_lr_fraction")]
    for patch, msg in bad:
        d = json.loads(json.dumps(base))
        d["stages"][0].update(patch)
        try:
            C.from_dict(d)
        except C.ConfigError as e:
            assert msg in str(e), (patch, e)
        else:
            raise AssertionError(f"accepted {patch}")


def test_trainer_wsd_resume_exact(tmp_root=None):
    """Tiny CPU run with wsd, interrupted and resumed: the lr at every eval must be
    exactly the schedule's value for that step (no drift across the resume)."""
    import shutil
    import tempfile

    from cubegpt import config as C
    from cubegpt.train import RetryableInterrupt, run_training
    root = tmp_root or tempfile.mkdtemp()
    d = {"name": "wsd", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "checkpoint_every_minutes": 0.02, "timeout_hours": 0.002,
         "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4,
                   "n_active": 1},
         "stages": [{"steps": 3000, "batch_size": 64, "lr": 2e-3, "eval_every": 20, "schedule": "wsd",
                     "warmup_samples": 64 * 200, "decay_samples": 64 * 800}]}
    cfg = C.to_dict(C.from_dict(d))
    attempts = 0
    while True:
        attempts += 1
        try:
            r = run_training(cfg, root, lambda: None, "fc", device="cpu")
            break
        except RetryableInterrupt:
            assert attempts < 10
    recs = [json.loads(l) for l in open(Path(root) / "runs/wsd/metrics.jsonl")]
    errs = [abs(rec["lr"] - 2e-3 * lr_factor("wsd", rec["stage_step"] - 1, 3000, 200, 800)) for rec in recs]
    print(f"  attempts {attempts}, {len(recs)} evals, max lr error {max(errs):.2e}, "
          f"lr {recs[0]['lr']:.2e} -> {max(r['lr'] for r in recs):.2e} -> {recs[-1]['lr']:.2e}, "
          f"eval loss {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}")
    assert r["status"] == "completed" and attempts >= 2, "test should exercise a resume"
    assert max(errs) < 1e-15 and len(recs) == 150
    assert [rec["stage_step"] for rec in recs] == list(range(20, 3001, 20))
    if tmp_root is None:
        shutil.rmtree(root)


def test_divergence_fails_fast():
    """An eval loss far above the best so far must fail the run (no retries) with a clear error."""
    import shutil
    import tempfile

    from cubegpt import config as C
    from cubegpt import train as TR
    root = tempfile.mkdtemp()
    seq = iter([0.30, 0.25, 0.20, 0.21, 0.19, 0.45] + [0.19] * 1000)   # spike at the 6th eval (2.37x best)
    real = TR.eval_loss
    TR.eval_loss = lambda *a, **k: next(seq)
    try:
        d = {"name": "div", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48,
                                                    "n_layers": 2, "n_experts": 4, "n_active": 1},
             "stages": [{"steps": 400, "batch_size": 64, "lr": 1e-3, "eval_every": 20}]}
        r = TR.run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    finally:
        TR.eval_loss = real
    status = json.load(open(Path(root) / "runs/div/status.json"))
    msg = str(r.get("error"))
    print(f"  status {r['status']}: {msg[:110]}")
    assert r["status"] == "failed" and status["state"] == "failed" and "diverged" in msg and "2.37x" in msg
    assert status["global_step"] == 120 if "global_step" in status else True
    shutil.rmtree(root)


def _scripted_run(name, seq_values, rollbacks, steps=400):
    """Tiny CPU wsd run whose eval losses come from seq_values; loops over Modal-style retries."""
    import tempfile

    from cubegpt import config as C
    from cubegpt import train as TR
    root = tempfile.mkdtemp()
    seq = iter(seq_values)
    real = TR.eval_loss
    TR.eval_loss = lambda *a, **k: next(seq)
    try:
        d = {"name": name, "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
             "gen_chunk_samples": 4096, "checkpoint_every_minutes": 0.0001, "divergence_rollbacks": rollbacks,
             "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4,
                       "n_active": 1},
             "stages": [{"steps": steps, "batch_size": 64, "lr": 1e-3, "eval_every": 20, "schedule": "wsd",
                         "warmup_samples": 64 * 40, "decay_samples": 64 * 100}]}
        cfg = C.to_dict(C.from_dict(d))
        attempts = 0
        while True:
            attempts += 1
            try:
                r = TR.run_training(cfg, root, lambda: None, "fc", device="cpu")
                break
            except TR.RetryableInterrupt:
                assert attempts < 10
    finally:
        TR.eval_loss = real
    return root, r, attempts


def test_divergence_rollback_recovers():
    import shutil
    good = [0.3 - 0.001 * i for i in range(200)]
    # eval 8 (step 160) spikes; the run rolls back to the step-140 checkpoint and finishes at 0.7x lr
    root, r, attempts = _scripted_run("rb", good[:7] + [0.9] + good[7:], rollbacks=2)
    rb = json.load(open(Path(root) / "runs/rb/rollbacks.json"))
    recs = [json.loads(l) for l in open(Path(root) / "runs/rb/metrics.jsonl")]
    ev = rb["events"][0]
    print(f"  attempts {attempts}, rollback at step {ev['global_step']} -> resumed from step "
          f"{ev['rolled_back_to_step']}, lr mult {rb['lr_mult']}, final status {r['status']}")
    assert r["status"] == "completed" and attempts == 2 and len(rb["events"]) == 1
    assert abs(rb["lr_mult"] - 0.7) < 1e-12 and ev["global_step"] == 160 and ev["rolled_back_to_step"] == 140
    assert [x["stage_step"] for x in recs] == list(range(20, 401, 20))      # spike record dropped, no gaps
    assert max(x["eval_loss"] for x in recs) < 0.31
    for x in recs:   # lr = peak * mult * schedule: full peak before the rollback point, 0.7x after
        mult = 1.0 if x["stage_step"] <= 140 else 0.7
        assert abs(x["lr"] - 1e-3 * mult * lr_factor("wsd", x["stage_step"] - 1, 400, 40, 100)) < 1e-15, x
    shutil.rmtree(root)


def test_divergence_rollbacks_exhausted():
    import shutil
    good = [0.3 - 0.001 * i for i in range(200)]
    root, r, attempts = _scripted_run("rbx", good[:7] + [0.9] + good[7:9] + [0.9] + good[9:], rollbacks=1)
    print(f"  attempts {attempts}, status {r['status']}: {str(r.get('error'))[:100]}")
    assert r["status"] == "failed" and "all 1 rollbacks used" in r["error"]
    shutil.rmtree(root)


def test_rollback_config_rules():
    from cubegpt import config as C
    d = {"name": "c", "model": {}, "divergence_rollbacks": 2,
         "stages": [{"steps": 100, "batch_size": 8, "schedule": "eval_plateau", "plateau_window_samples": 8 * 1000 * 3}]}
    try:
        C.from_dict(d)
    except C.ConfigError as e:
        assert "only supported" in str(e)
    else:
        raise AssertionError("rollbacks accepted with a plateau schedule")
    assert "divergence_rollbacks" in C.OPERATIONAL_KEYS     # can be added to an existing run on --resume


def test_router_lr_mult():
    """router_lr_mult puts the MoE gate weights in their own group at mult x the lr, under
    every schedule, through a resume; the log reports the router score."""
    import shutil
    import tempfile

    from cubegpt import config as C
    from cubegpt.config import ModelConfig
    from cubegpt.model import MoEBlock, build_model
    from cubegpt.schedule import StepSchedule
    from cubegpt.train import RetryableInterrupt, param_groups, run_training
    model = build_model(ModelConfig(arch="moe", d_model=32, n_heads=4, d_ffwd=48, n_layers=2, n_experts=4, n_active=1))
    gates = {id(b.gate.weight) for b in model.modules() if isinstance(b, MoEBlock)}
    groups = param_groups(model, 0.1)
    assert {id(p) for p in groups[1]["params"]} == gates and len(gates) == 2
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    assert len(param_groups(model, 1.0)) == 1                     # default: one group (old checkpoints load)
    opt = torch.optim.Adam(groups, lr=1e-3)
    sched = StepSchedule(opt, "wsd", 100, 10, 20, 0.0)
    for step in (0, 50, 90):
        sched.set(step)
        assert opt.param_groups[1]["lr"] == opt.param_groups[0]["lr"] * 0.1
    ep = EvalPlateau(opt, 0.5, 3, 0.01)
    for i in range(6):
        ep.observe(i, 1.0)                                        # flat: decays once
    assert opt.param_groups[0]["lr"] < 1e-3 * lr_factor("wsd", 90, 100, 10, 20) + 1e-12
    assert abs(opt.param_groups[1]["lr"] - opt.param_groups[0]["lr"] * 0.1) < 1e-15

    base = {"seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
            "gen_chunk_samples": 4096, "checkpoint_every_minutes": 0.02, "timeout_hours": 0.002,
            "model": {"arch": "moe", "d_model": 32, "n_heads": 4, "d_ffwd": 48, "n_layers": 2, "n_experts": 4,
                      "n_active": 1},
            "stages": [{"steps": 3000, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                        "warmup_samples": 64 * 200, "decay_samples": 64 * 800}]}
    for bad in ({"router_lr_mult": 0}, {"router_lr_mult": 0.1, "model": {"arch": "gated"}}):
        try:
            C.from_dict({**base, "name": "bad", **bad})
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass
    root = tempfile.mkdtemp()
    cfg = C.to_dict(C.from_dict({**base, "name": "router", "router_lr_mult": 0.1}))
    run = Path(root) / "runs/router"
    attempts, saved_groups = 0, None
    while True:
        attempts += 1
        try:
            r = run_training(cfg, root, lambda: None, "fc", device="cpu")
            break
        except RetryableInterrupt:            # the state the next attempt resumes from
            assert attempts < 10
            saved_groups = torch.load(run / "ckpt/latest.pt", map_location="cpu",
                                      weights_only=False)["optimizer"]["param_groups"]
    g0, g1 = saved_groups
    log = (run / "log.txt").read_text()
    recs = [json.loads(l) for l in open(run / "metrics.jsonl")]
    print(f"  attempts {attempts}, final group lrs {g0['lr']:.2e} / {g1['lr']:.2e}, "
          f"router score min {recs[-1]['moe']['router_score_min']:.4f}")
    assert r["status"] == "completed" and attempts >= 2, "test should exercise a resume"
    assert g1["lr_mult"] == 0.1 and g1["lr"] == g0["lr"] * 0.1 and len(g1["params"]) == 2
    assert "MoE router trains at 0.1x the lr" in log and "router score min" in log
    assert all(0 < rec["moe"]["router_score_min"] < 1 for rec in recs)
    shutil.rmtree(root)


if __name__ == "__main__":
    for fn in [test_trend_exact, test_improving_never_decays, test_flat_decays_once_per_window,
               test_noisy_eval_needs_real_signal, test_min_lr_floor, test_state_dict_resume_identical,
               test_trainer_end_to_end, test_wsd_shape, test_cosine_shape, test_fixed_schedule_validation,
               test_trainer_wsd_resume_exact, test_divergence_fails_fast, test_divergence_rollback_recovers,
               test_divergence_rollbacks_exhausted, test_rollback_config_rules, test_router_lr_mult]:
        print(fn.__name__)
        fn()
        print("  ok")
