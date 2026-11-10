"""Resumable, fail-fast training loop.

Modal-agnostic: the Modal wrapper passes a run root (the mounted Volume), a
commit() callback and the function-call id. Locally you can call
run_training(..., commit=lambda: None) on CPU for debugging.

Run directory layout (<run_root>/runs/<name>/):
    config.json        frozen run config
    status.json        heartbeat: state, progress, losses, ETA, attempts, error
    metrics.jsonl      one JSON record per periodic eval
    stage_metrics.json MSE / within-k accuracy at the end of every stage
    log.txt            human-readable log (same lines as stdout)
    ckpt/latest.pt     full training state (model, optimizer, scheduler, RNG, counters)
    ckpt/model_<samples>.pt  optional snapshots (keep_every_samples)
    final.pth          bare model state_dict (loads like the files in models/)

Failure policy:
    * Deterministic problems (bad config, failed self-test, NaN loss, CUDA OOM,
      programming errors, or *any* error during setup of a fresh launch) are
      fatal: status -> "failed", the call returns without raising, so Modal does
      not retry.
    * Anything else (preemption, hardware/driver errors, volume hiccups, the
      watchdog firing, the per-attempt time budget) re-raises or exits so Modal
      retries in a fresh container; the retry resumes from ckpt/latest.pt.
    * Crash-loop guard: max_stalled_restarts consecutive attempts that start
      from the same checkpoint step => fatal.
"""
from __future__ import annotations

import faulthandler
import json
import math
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

from . import config as C
from .cube import CubeTables, SCRAMBLE_LIMIT, check_gpu_pipeline, generate_samples, run_string_self_tests
from .evaluate import compute_metrics, eval_loss, format_metrics, make_eval_set
from .model import CKPT_FORMAT, MoEBlock, build_model, count_params, expert_stats, load_weights
from .muon import MuonAdam, muon_params
from .schedule import EvalPlateau, StepSchedule

EVAL_SEED_OFFSET = 1_000_003
METRICS_SEED = 4242
BIAS_WARN = 1.0   # router scores are in (0,1): a within-layer bias spread beyond this overrides affinity


class FatalError(Exception):
    """Deterministic failure: retrying will not help."""


class RunConflictError(FatalError):
    """The run directory belongs to something else; don't touch its status."""


class RetryableInterrupt(Exception):
    """Raised deliberately so Modal restarts the attempt from the last checkpoint."""


def param_groups(model, router_lr_mult=1.0, weight_decay=0.0):
    """Optimizer param groups. With router_lr_mult != 1 the MoE router (gate) weights get
    their own group tagged "lr_mult" (group 0 stays the main group, which the schedules
    and logs read); otherwise one group, so older checkpoints' optimizer state loads.
    With weight_decay > 0 each group is split: tensors with >= 2 dims (weight matrices,
    embeddings) decay, biases and LayerNorm parameters don't."""
    if router_lr_mult == 1.0:
        groups = [{"params": list(model.parameters())}]
    else:
        router = {id(p) for m in model.modules() if isinstance(m, MoEBlock) for p in m.gate.parameters()}
        if not router:
            raise FatalError("router_lr_mult is set but the model has no MoE router")
        groups = [{"params": [p for p in model.parameters() if id(p) not in router], "lr_mult": 1.0},
                  {"params": [p for p in model.parameters() if id(p) in router], "lr_mult": router_lr_mult}]
    if weight_decay <= 0:
        return groups
    out = []
    for g in groups:
        rest = {k: v for k, v in g.items() if k != "params"}
        for decays in (True, False):
            ps = [p for p in g["params"] if (p.ndim >= 2) == decays]
            if ps:
                out.append({"params": ps, "weight_decay": weight_decay if decays else 0.0, **rest})
    return out


class DivergenceRollback(RetryableInterrupt):
    """Training diverged; rollbacks.json now holds a lower lr multiplier and the next
    attempt resumes from the last healthy checkpoint with it."""


# Periodic checkpoints are only written right after an eval at most this far above the
# best eval so far, so a divergence rollback never lands on a state that was already blowing up.
HEALTHY_CKPT_FACTOR = 1.03


FATAL_TYPES = (FatalError, C.ConfigError, ValueError, TypeError, KeyError, AttributeError, NameError,
               IndexError, AssertionError, NotImplementedError, ZeroDivisionError, ImportError,
               torch.OutOfMemoryError)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_duration(s):
    if s is None or not math.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_count(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


class RunDir:
    def __init__(self, root: Path, name: str, commit: Callable[[], None]):
        self.path = root / "runs" / name
        self.ckpt_dir = self.path / "ckpt"
        self.rollbacks = self.path / "rollbacks.json"
        self.config = self.path / "config.json"
        self.status = self.path / "status.json"
        self.metrics = self.path / "metrics.jsonl"
        self.stage_metrics = self.path / "stage_metrics.json"
        self.logfile = self.path / "log.txt"
        self.latest = self.ckpt_dir / "latest.pt"
        self.final = self.path / "final.pth"
        self._commit = commit
        self._commit_failures = 0

    @staticmethod
    def read_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def write_json(path, obj):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)

    def log(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.logfile, "a") as f:
                f.write(line + "\n")
        except OSError as e:  # never let logging kill a run
            print(f"(could not append to log file: {e!r})", flush=True)

    def append_metrics(self, rec):
        with open(self.metrics, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def commit(self):
        try:
            self._commit()
            self._commit_failures = 0
        except Exception as e:  # transient; escalate to a retry if it persists
            self._commit_failures += 1
            self.log(f"WARNING: volume commit failed ({self._commit_failures} in a row): {e!r}")
            if self._commit_failures >= 3:
                raise


class Watchdog:
    """Hard-exits the process if beat() isn't called for `timeout_s` seconds, so a
    hung GPU/kernel turns into a Modal retry instead of a silently stuck run."""

    def __init__(self, timeout_s: float, log: Callable[[str], None]):
        self.timeout_s = timeout_s
        self.log = log
        self.last = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="watchdog")

    def start(self):
        self._thread.start()
        return self

    def beat(self):
        self.last = time.monotonic()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(min(30.0, self.timeout_s / 4)):
            stalled = time.monotonic() - self.last
            if stalled > self.timeout_s:
                self.log(f"WATCHDOG: no progress for {stalled / 60:.1f} min; dumping stacks and exiting "
                         "so Modal restarts from the last checkpoint")
                faulthandler.dump_traceback(all_threads=True)
                sys.stdout.flush()
                os._exit(3)


class Trainer:
    def __init__(self, cfg: C.RunConfig, run: RunDir, call_id: str, resume: bool, device: Optional[str],
                 attempt_started: float):
        self.cfg = cfg
        self.run = run
        self.call_id = call_id
        self.resume = resume
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attempt_started = attempt_started
        self.is_first_attempt: Optional[bool] = None
        self.steps_this_attempt = 0
        self.status: dict = {}
        self.watchdog: Optional[Watchdog] = None
        self.total_samples = C.total_samples(cfg)
        self.gs = 0                # global step
        self.samples = 0
        self.stage_idx = 0
        self.stage_step = 0
        self.best_eval = float("inf")
        self.train_seconds = 0.0   # accumulated across attempts
        self.last_ckpt_time = time.time()
        self.last_status_time = 0.0
        self.rate = None           # samples/s over the last interval
        rb = run.read_json(run.rollbacks) or {}
        self.lr_mult = rb.get("lr_mult", 1.0)   # < 1 after divergence rollbacks
        self.rollback_events = rb.get("events", [])

    # ------------------------------------------------------------------ claim
    def claim_run(self):
        run, cfg_dict = self.run, C.to_dict(self.cfg)
        stored = run.read_json(run.config)
        status = run.read_json(run.status) or {}

        if stored is None and status.get("owner_call_id") not in (None, self.call_id):
            raise RunConflictError(f"{run.path} has a status file from another launch but no config")
        if stored is not None:
            diffs = C.diff_training_fields(stored, cfg_dict)
            if diffs:
                raise RunConflictError("run exists with a different training config:\n  " + "\n  ".join(diffs)
                                       + "\nChoose a new run name (use init_from to start from its weights).")
            if status.get("owner_call_id") != self.call_id:
                if not self.resume:
                    raise RunConflictError(f"run {self.cfg.name!r} already exists; pass --resume to continue it "
                                           "or choose a new name")
                status["owner_call_id"] = self.call_id
        elif self.resume:
            raise RunConflictError(f"--resume given but run {self.cfg.name!r} does not exist")

        if status.get("state") == "completed":
            self.status = status
            return "completed"

        start_step = int(status.get("last_checkpoint_step", 0))
        attempts = status.get("attempts", [])
        stalled = 0
        for a in reversed(attempts):
            if a.get("call_id") == self.call_id and a.get("start_step") == start_step:
                stalled += 1
            else:
                break
        if stalled >= self.cfg.max_stalled_restarts:
            last = status.get("last_error") or {}
            raise FatalError(f"{stalled} consecutive attempts started from step {start_step} without making "
                             f"checkpoint progress; giving up. Last error: {last.get('type')}: {last.get('message')}")
        self.is_first_attempt = not any(a.get("call_id") == self.call_id for a in attempts)
        attempts.append({"call_id": self.call_id, "started_at": utcnow(), "start_step": start_step})

        status.update({
            "name": self.cfg.name, "state": "starting", "owner_call_id": self.call_id,
            "attempts": attempts[-100:], "total_samples": self.total_samples,
            "n_stages": len(self.cfg.stages), "error": None, "updated_at": utcnow(),
            "status_every_minutes": self.cfg.status_every_minutes,
        })
        status.setdefault("created_at", utcnow())
        status.setdefault("last_checkpoint_step", 0)
        self.status = status
        run.ckpt_dir.mkdir(parents=True, exist_ok=True)
        run.write_json(run.status, status)   # status (with owner) first, then config
        if stored is None:
            run.write_json(run.config, cfg_dict)
        run.commit()
        n_att = sum(a["call_id"] == self.call_id for a in attempts)
        how = "fresh start" if n_att == 1 and stored is None else f"resuming from step {start_step}"
        run.log(f"claimed run {self.cfg.name!r} (call {self.call_id}, attempt {n_att}, {how})")
        return "ok"

    # ------------------------------------------------------------------ status
    def write_status(self, state="running", commit=True, **extra):
        st = self.status
        stages = self.cfg.stages
        done = self.stage_idx >= len(stages)
        si = min(self.stage_idx, len(stages) - 1)
        remaining = self.total_samples - self.samples
        st.update({
            "state": state, "global_step": self.gs, "samples_seen": self.samples,
            "progress": self.samples / max(1, self.total_samples),
            "stage": si + 1,
            "stage_step": C.stage_steps(stages[si]) if done else self.stage_step,
            "stage_steps": C.stage_steps(stages[si]),
            "samples_per_sec": self.rate, "eta_seconds": (remaining / self.rate) if self.rate else None,
            "best_eval_loss": self.best_eval if math.isfinite(self.best_eval) else None,
            "train_seconds": self.train_seconds, "updated_at": utcnow(),
        })
        st.update(extra)
        self.run.write_json(self.run.status, st)
        self.last_status_time = time.time()
        if commit:
            self.run.commit()

    # ------------------------------------------------------------------ setup
    def setup_device(self):
        cfg, log = self.cfg, self.run.log
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise FatalError("CUDA requested but not available")
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            log(f"GPU: {name} ({total / 2**30:.0f} GiB, {free / 2**30:.0f} GiB free) | torch {torch.__version__} "
                f"| CUDA {torch.version.cuda}")
            self.status["gpu_name"] = name
        else:
            log(f"running on {self.device} (torch {torch.__version__})")
            self.status["gpu_name"] = self.device
        if cfg.tf32:
            torch.set_float32_matmul_precision("high")

    def checkpoint_state(self, model, optim, sched):
        return {
            "format": CKPT_FORMAT, "model": model.state_dict(),
            "optimizer": optim.state_dict() if optim is not None else None,
            "scheduler": sched.state_dict() if sched is not None else None,
            "stage_idx": self.stage_idx, "stage_step": self.stage_step, "global_step": self.gs,
            "samples_seen": self.samples, "best_eval": self.best_eval, "train_seconds": self.train_seconds,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if self.device == "cuda" else None,
        }

    def save_checkpoint(self, model, optim, sched, reason):
        t0 = time.time()
        tmp = self.run.latest.with_name("latest.pt.tmp")
        torch.save(self.checkpoint_state(model, optim, sched), tmp)
        os.replace(tmp, self.run.latest)
        self.status.update({"last_checkpoint_step": self.gs, "last_checkpoint_at": utcnow()})
        self.write_status(commit=False)
        self.run.commit()
        self.last_ckpt_time = time.time()
        self.run.log(f"checkpoint saved at step {self.gs} ({reason}, {time.time() - t0:.1f}s)")

    def preflight(self, model, T):
        """Exercise every code path that would otherwise first run hours in:
        each stage's data kind at its batch size (fwd+bwd, no optimizer step,
        so peak memory is realistic), and the end-of-stage metrics code."""
        loss_fn = nn.BCEWithLogitsLoss()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # peak data-generation memory: one full canonicalization sub-batch
        generate_samples(T, (self.cfg.gen_max_batch // SCRAMBLE_LIMIT) * SCRAMBLE_LIMIT, "random",
                         self.cfg.gen_max_batch)
        model.train()
        for i, st in enumerate(self.cfg.stages):
            X, y = generate_samples(T, st.batch_size, st.data, self.cfg.gen_max_batch, st.depth_weight_power,
                                    canonical=self.cfg.train_canonical)
            loss = loss_fn(model(X), y)
            loss.backward()
            if not torch.isfinite(loss):
                raise FatalError(f"preflight: non-finite loss for stage {i + 1}")
            model.zero_grad(set_to_none=True)
        compute_metrics(model, T, n_samples=1024, batch_size=1024, max_batch=self.cfg.gen_max_batch)
        model.train()
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            self.run.log(f"preflight ok (peak memory {peak:.1f} / {total:.0f} GiB)")
        else:
            self.run.log("preflight ok")

    # ------------------------------------------------------------------ main
    def train(self):
        cfg, run = self.cfg, self.run
        if self.claim_run() == "completed":
            run.log("run already completed; nothing to do")
            return {"status": "completed", "run": cfg.name, "metrics": self.status.get("final_metrics")}

        self.watchdog = Watchdog(cfg.watchdog_minutes * 60, run.log).start()
        try:
            return self._setup_and_train()
        finally:
            self.watchdog.stop()

    def _setup_and_train(self):
        cfg, run = self.cfg, self.run
        self.setup_device()
        t0 = time.time()
        run_string_self_tests()
        T = CubeTables(self.device)
        check_gpu_pipeline(T)
        run.log(f"cube self-tests + GPU pipeline equivalence checks passed ({time.time() - t0:.1f}s)")

        torch.manual_seed(cfg.seed)   # deterministic init (overwritten on resume)
        model = build_model(cfg.model).to(self.device)
        run.log(f"model: {cfg.model} | {count_params(model):,} params")

        ckpt = None
        if run.latest.exists():
            ckpt = torch.load(run.latest, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            for k in ("stage_idx", "stage_step", "best_eval", "train_seconds"):
                setattr(self, k, ckpt[k])
            self.gs, self.samples = ckpt["global_step"], ckpt["samples_seen"]
            self._drop_records_after(self.gs)
            run.log(f"resumed from checkpoint: step {self.gs}, stage {self.stage_idx + 1}, "
                    f"{fmt_count(self.samples)} samples")
        elif cfg.init_from:
            src = run.path.parent.parent / cfg.init_from
            if not src.exists():
                raise FatalError(f"init_from {cfg.init_from!r} not found in volume")
            try:
                model.load_state_dict(load_weights(src, self.device))
            except RuntimeError as e:
                raise FatalError(f"init_from {cfg.init_from!r} does not match the model config: {e}") from e
            run.log(f"initialized weights from {cfg.init_from}")
        stats = expert_stats(model)
        if stats and stats["bias_spread"] > BIAS_WARN:
            run.log(f"WARNING: MoE expert-bias spread {stats['bias_spread']:.1f} > {BIAS_WARN} at start: routing "
                    "is dominated by the balancing bias (some experts may be dead)")

        eval_X, eval_y = make_eval_set(T, cfg.eval_samples, cfg.seed + EVAL_SEED_OFFSET, cfg.gen_max_batch)
        self.preflight(model, T)

        if ckpt is not None:
            torch.set_rng_state(ckpt["rng_cpu"].cpu())
            if self.device == "cuda" and ckpt.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        else:
            torch.manual_seed(cfg.seed if cfg.data_seed < 0 else cfg.data_seed)
            self.save_checkpoint(model, None, None, "initial; verifying checkpoint round-trip")
            back = torch.load(run.latest, map_location=self.device, weights_only=True)
            for k, v in model.state_dict().items():
                if not torch.equal(back["model"][k], v):
                    raise FatalError(f"checkpoint round-trip mismatch for {k}")

        # Stop a little before Modal's hard timeout so we exit with a fresh checkpoint.
        timeout_s = cfg.timeout_hours * 3600
        deadline = self.attempt_started + timeout_s - min(900.0, 0.1 * timeout_s)
        self.watchdog.beat()
        self.write_status("running")
        return self._train_loop(model, T, eval_X, eval_y, ckpt, deadline)

    def _train_loop(self, model, T, eval_X, eval_y, ckpt, deadline):
        cfg, run = self.cfg, self.run
        loss_fn = nn.BCEWithLogitsLoss()
        is_moe = hasattr(model, "update_expert_bias")
        keep_every = cfg.keep_every_samples
        moe_blocks = [m for m in model.modules() if isinstance(m, MoEBlock)]
        use_aux = bool(moe_blocks) and (cfg.moe_aux_loss > 0 or cfg.moe_z_loss > 0)
        for b in moe_blocks:
            b.bias_update_speed = cfg.moe_bias_update_speed
            b.aux_enabled = use_aux
        update_bias = is_moe and cfg.moe_bias_update_speed > 0

        for si in range(self.stage_idx, len(cfg.stages)):
            st = cfg.stages[si]
            steps_total = C.stage_steps(st)
            eval_every = C.stage_eval_every(st)
            base_lr = C.stage_lr(cfg, st) * self.lr_mult
            groups = param_groups(model, cfg.router_lr_mult, cfg.weight_decay)
            fused = bool(cfg.adam_fused and self.device == "cuda")
            if cfg.optimizer == "muon":
                mu = muon_params(model)
                mu_ids = {id(p) for p in mu}
                for g in groups:   # group 0 stays the main (Adam) group that schedules and logs read
                    g["params"] = [p for p in g["params"] if id(p) not in mu_ids]
                    g.setdefault("lr_mult", 1.0)
                groups = [g for g in groups if g["params"]]
                groups.append({"params": mu, "muon": True, "lr_mult": cfg.muon_lr_mult,
                               "weight_decay": cfg.muon_weight_decay})
                opt_cls = MuonAdam
                optim = MuonAdam(groups, lr=base_lr, momentum=cfg.muon_momentum,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps, fused=fused)
            else:
                opt_cls = torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam
                optim = opt_cls(groups, lr=base_lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
                                eps=cfg.adam_eps, weight_decay=0.0, fused=fused)
            for g in optim.param_groups:
                g["lr"] *= g.get("lr_mult", 1.0)
            if st.schedule == "eval_plateau":
                sched = EvalPlateau(optim, st.plateau_factor, C.stage_window_evals(st), st.plateau_threshold,
                                    st.plateau_min_lr)
                sched_desc = (f"eval-plateau (window {C.stage_window_evals(st)} evals = "
                              f"{fmt_count(st.plateau_window_samples)} samples, threshold "
                              f"{100 * st.plateau_threshold:.2f}%, min lr {st.plateau_min_lr:.3g})")
            elif st.schedule in C.FIXED_SCHEDULES:
                sched = StepSchedule(optim, st.schedule, steps_total, C.stage_warmup_steps(st),
                                     C.stage_decay_steps(st), st.final_lr_fraction)
                sched_desc = sched.describe()
            else:
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, factor=st.plateau_factor, patience=C.stage_patience(st), min_lr=st.plateau_min_lr)
                sched_desc = f"torch plateau (patience {C.stage_patience(st)} steps)"
            step_sched = st.schedule == "torch_plateau"
            fixed_sched = st.schedule in C.FIXED_SCHEDULES
            if ckpt is not None and si == ckpt["stage_idx"] and ckpt.get("optimizer") is not None:
                optim.load_state_dict(ckpt["optimizer"])
                sched.load_state_dict(ckpt["scheduler"])
            ckpt = None
            self.stage_idx = si
            label = f" ({st.label})" if st.label else ""
            if self.lr_mult != 1.0:
                run.log(f"lr multiplier {self.lr_mult:.3g} after {len(self.rollback_events)} divergence rollback(s)")
            if cfg.router_lr_mult != 1.0:
                run.log(f"MoE router trains at {cfg.router_lr_mult:g}x the lr")
            if moe_blocks and (use_aux or cfg.moe_bias_update_speed != 1e-3):
                run.log(f"MoE balancing: bias step {cfg.moe_bias_update_speed:g}/step, aux balance loss "
                        f"{cfg.moe_aux_loss:g}, z-loss {cfg.moe_z_loss:g}")
            run.log(f"=== stage {si + 1}/{len(cfg.stages)}{label}: {steps_total} steps x batch {st.batch_size} "
                    f"({st.data} data{f', depth weight max(d,1)^{st.depth_weight_power:g}' if st.depth_weight_power else ''}), "
                    f"lr {optim.param_groups[0]['lr']:.3g}, {sched_desc}, "
                    f"eval every {eval_every} steps, from step {self.stage_step}")

            chunk_steps = max(1, cfg.gen_chunk_samples // st.batch_size)
            chunk_X = chunk_y = None
            chunk_pos = chunk_len = 0
            interval_loss, interval_n = 0.0, 0
            interval_t0, interval_samples0 = time.time(), self.samples
            self.gn_sum, self.gn_max, self.gn_n = 0.0, 0.0, 0
            if cfg.grad_clip > 0:
                run.log(f"gradient clipping: global norm <= {cfg.grad_clip:g}")
            if (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, cfg.weight_decay) != (0.9, 0.999, 1e-8, 0.0):
                run.log(f"optimizer: {opt_cls.__name__} betas ({cfg.adam_beta1:g}, {cfg.adam_beta2:g}) "
                        f"eps {cfg.adam_eps:g} weight decay {cfg.weight_decay:g}")
            if cfg.optimizer == "muon":
                run.log(f"Muon on {len(mu)} stack matrices ({sum(p.numel() for p in mu):,} params): lr "
                        f"{base_lr * cfg.muon_lr_mult:.3g} ({cfg.muon_lr_mult:g}x), momentum {cfg.muon_momentum:g}, "
                        f"weight decay {cfg.muon_weight_decay:g}; Adam on the rest")
            model.train()

            while self.stage_step < steps_total:
                if chunk_pos >= chunk_len:
                    chunk_len = min(chunk_steps, steps_total - self.stage_step)
                    chunk_X, chunk_y = generate_samples(T, chunk_len * st.batch_size, st.data, cfg.gen_max_batch,
                                                        st.depth_weight_power, canonical=cfg.train_canonical)
                    chunk_X = chunk_X.view(chunk_len, st.batch_size, *chunk_X.shape[1:])
                    chunk_y = chunk_y.view(chunk_len, st.batch_size, *chunk_y.shape[1:])
                    chunk_pos = 0
                X, y = chunk_X[chunk_pos], chunk_y[chunk_pos]
                chunk_pos += 1

                t_step = time.time()
                if fixed_sched:
                    sched.set(self.stage_step)
                loss = loss_fn(model(X), y)
                total = loss
                if use_aux:   # logged train loss stays the plain BCE; the aux terms only shape the gradient
                    n = len(moe_blocks)
                    total = (loss + cfg.moe_aux_loss * sum(b.aux_balance for b in moe_blocks) / n
                             + cfg.moe_z_loss * sum(b.aux_z for b in moe_blocks) / n)
                optim.zero_grad(set_to_none=True)
                total.backward()
                if cfg.grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                lv = loss.item()
                if cfg.grad_clip > 0:
                    gn = gn.item()
                    self.gn_sum += gn
                    self.gn_max = max(self.gn_max, gn)
                    self.gn_n += 1
                if not math.isfinite(lv):
                    raise FatalError(f"non-finite training loss ({lv}) at step {self.gs + 1}; last good "
                                     f"checkpoint is step {self.status.get('last_checkpoint_step')}")
                if step_sched:
                    sched.step(lv)
                if update_bias:
                    model.update_expert_bias()
                self.train_seconds += time.time() - t_step

                prev_samples = self.samples
                self.stage_step += 1
                self.gs += 1
                self.samples += st.batch_size
                self.steps_this_attempt += 1
                interval_loss += lv
                interval_n += 1
                self.watchdog.beat()

                healthy_eval = False
                if self.stage_step % eval_every == 0 or self.stage_step == steps_total:
                    now = time.time()
                    self.rate = (self.samples - interval_samples0) / max(now - interval_t0, 1e-9)
                    el = self._eval_and_log(model, eval_X, eval_y, interval_loss / interval_n, optim, steps_total,
                                            sched if st.schedule == "eval_plateau" else None)
                    healthy_eval = el <= HEALTHY_CKPT_FACTOR * self.best_eval
                    model.train()
                    interval_loss, interval_n = 0.0, 0
                    interval_t0, interval_samples0 = time.time(), self.samples

                if keep_every and self.samples // keep_every > prev_samples // keep_every:
                    torch.save(model.state_dict(), run.ckpt_dir / f"model_{self.samples}.pt")
                    run.log(f"kept snapshot model_{self.samples}.pt")
                if cfg.debug_crash_at_step and self.gs == cfg.debug_crash_at_step:
                    self._debug_crash()
                now = time.time()
                if self.stage_step < steps_total:
                    if healthy_eval and now - self.last_ckpt_time >= cfg.checkpoint_every_minutes * 60:
                        self.save_checkpoint(model, optim, sched, "periodic")
                    elif now - self.last_status_time >= cfg.status_every_minutes * 60:
                        self.write_status("running", lr=optim.param_groups[0]["lr"])
                    if now > deadline:
                        self.save_checkpoint(model, optim, sched, "attempt time budget reached")
                        raise RetryableInterrupt(
                            f"attempt time budget ({cfg.timeout_hours}h) nearly used up; checkpoint saved at "
                            f"step {self.gs}. Raising so Modal restarts this run in a fresh attempt.")

            # ---- stage complete
            self.stage_idx, self.stage_step = si + 1, 0
            self.save_checkpoint(model, None, None, f"end of stage {si + 1}")
            if cfg.final_eval_samples:
                m = compute_metrics(model, T, cfg.final_eval_samples, seed=METRICS_SEED,
                                    max_batch=cfg.gen_max_batch)
                m.update({"stage": si + 1, "global_step": self.gs, "samples_seen": self.samples})
                run.log(f"stage {si + 1} metrics: {format_metrics(m)}")
                all_m = run.read_json(run.stage_metrics) or []
                all_m.append(m)
                run.write_json(run.stage_metrics, all_m)
                self.status["final_metrics"] = m
                self.write_status("running")
                model.train()

        # ---- run complete
        tmp = run.final.with_name("final.pth.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, run.final)
        self.write_status("completed", commit=False, completed_at=utcnow())
        run.commit()
        run.log(f"DONE: {self.gs} steps, {fmt_count(self.samples)} samples, train time "
                f"{fmt_duration(self.train_seconds)}. Final weights: runs/{cfg.name}/final.pth")
        return {"status": "completed", "run": cfg.name, "metrics": self.status.get("final_metrics")}

    def _eval_and_log(self, model, eval_X, eval_y, train_loss, optim, steps_total, eval_sched=None):
        cfg = self.cfg
        el = eval_loss(model, eval_X, eval_y)
        prev_best = self.best_eval
        self.best_eval = min(self.best_eval, el)
        lr = optim.param_groups[0]["lr"]           # the lr used during this interval
        decision = eval_sched.observe(self.samples, el) if eval_sched is not None else None
        stats = expert_stats(model)
        rec = {"time": utcnow(), "global_step": self.gs, "stage": self.stage_idx + 1, "stage_step": self.stage_step,
               "samples_seen": self.samples, "eval_loss": el, "train_loss": train_loss, "lr": lr,
               "samples_per_sec": self.rate, "train_seconds": self.train_seconds}
        gnorm = ""
        if getattr(self, "gn_n", 0):
            rec["grad_norm_mean"], rec["grad_norm_max"] = self.gn_sum / self.gn_n, self.gn_max
            gnorm = f" | grad norm mean {rec['grad_norm_mean']:.3f} max {rec['grad_norm_max']:.3f}"
            self.gn_sum, self.gn_max, self.gn_n = 0.0, 0.0, 0
        trend = ""
        if eval_sched is not None:
            rec["plateau_trend"] = eval_sched.last_trend
            if eval_sched.last_trend is not None:
                trend = f" | trend {100 * eval_sched.last_trend:+.2f}%/window"
        moe = ""
        if stats:
            rec["moe"] = stats
            moe = (f" | moe bias spread {stats['bias_spread']:.2f} dead {stats['dead_experts']} "
                   f"max load {stats['max_load_ratio']:.2f}x router score min {stats['router_score_min']:.4f} "
                   f"bias override {100 * stats['bias_override_max']:.0f}%"
                   + (f" aux {stats['aux_balance']:.3f} z {stats['aux_z']:.2f}" if "aux_balance" in stats else ""))
            if stats["bias_spread"] > BIAS_WARN or stats["dead_experts"] or stats["bias_override_max"] > 0.5:
                moe += " (WARNING: balancing bias dominates routing)"
        self.run.append_metrics(rec)
        remaining = self.total_samples - self.samples
        self.run.log(
            f"step {self.gs} | stage {self.stage_idx + 1}/{len(cfg.stages)} {self.stage_step}/{steps_total} | "
            f"samples {fmt_count(self.samples)} | avg loss {el:.6f} | train {train_loss:.6f} | lr {lr:.3g} | "
            f"{fmt_count(self.rate or 0)} samples/s | eta {fmt_duration(remaining / self.rate if self.rate else None)}"
            + trend + gnorm + moe)
        if decision:
            self.run.log(decision)
        self.status.update({"last_eval_loss": el, "last_train_loss": train_loss,
                            "lr": optim.param_groups[0]["lr"], "moe": stats})
        f = cfg.divergence_factor
        if f and math.isfinite(prev_best) and el > f * prev_best:
            what = (f"diverged: eval loss {el:.4f} is {el / prev_best:.2f}x the best so far ({prev_best:.4f}; "
                    f"limit {f:g}x) at step {self.gs}, lr {lr:.3g}")
            if len(self.rollback_events) < cfg.divergence_rollbacks:
                new_mult = self.lr_mult * cfg.divergence_lr_factor
                self.rollback_events.append({"at": utcnow(), "global_step": self.gs, "samples_seen": self.samples,
                                             "eval_loss": el, "best_eval": prev_best, "lr": lr,
                                             "lr_mult_before": self.lr_mult, "lr_mult_after": new_mult,
                                             "rolled_back_to_step": self.status.get("last_checkpoint_step")})
                self.run.write_json(self.run.rollbacks, {"lr_mult": new_mult, "events": self.rollback_events})
                self.status.update({"lr_mult": new_mult, "rollbacks": len(self.rollback_events)})
                self.run.log(f"ROLLBACK {len(self.rollback_events)}/{cfg.divergence_rollbacks}: {what}. Restarting "
                             f"from the checkpoint at step {self.status.get('last_checkpoint_step')} with lr "
                             f"multiplier {self.lr_mult:.3g} -> {new_mult:.3g}")
                raise DivergenceRollback(what)
            raise FatalError(what + (f" (all {cfg.divergence_rollbacks} rollbacks used)" if cfg.divergence_rollbacks
                                     else "") + ". Lower the lr, lengthen warmup, or raise/disable divergence_factor.")
        return el

    def _drop_records_after(self, step):
        """Steps after the checkpoint are re-run on resume; drop their stale records
        so metrics.jsonl / stage_metrics.json stay monotonic."""
        run = self.run
        if run.metrics.exists():
            lines = run.metrics.read_text().splitlines()
            keep = [ln for ln in lines if ln.strip() and json.loads(ln)["global_step"] <= step]
            if len(keep) != len(lines):
                tmp = run.metrics.with_name(run.metrics.name + ".tmp")
                tmp.write_text("".join(ln + "\n" for ln in keep))
                os.replace(tmp, run.metrics)
        sm = run.read_json(run.stage_metrics)
        if sm and any(m["global_step"] > step for m in sm):
            run.write_json(run.stage_metrics, [m for m in sm if m["global_step"] <= step])

    def _debug_crash(self):
        marker = self.run.path / "debug_crashed"
        if marker.exists():
            return
        marker.write_text(utcnow())
        self.write_status("running")
        self.run.log(f"DEBUG: simulating a hard crash at step {self.gs} (debug_crash_at_step)")
        sys.stdout.flush()
        os._exit(13)


def run_training(cfg_dict: dict, run_root: str, commit: Callable[[], None], call_id: str,
                 resume: bool = False, device: Optional[str] = None) -> dict:
    """Entry point. Returns {"status": "completed" | "failed", ...}. Raises only
    for errors that should be retried by Modal."""
    attempt_started = time.time()
    faulthandler.enable()
    cfg = C.from_dict(cfg_dict)
    run = RunDir(Path(run_root), cfg.name, commit)
    trainer = Trainer(cfg, run, call_id, resume, device, attempt_started)
    try:
        return trainer.train()
    except Exception as e:
        tb = traceback.format_exc()
        setup_failure_on_launch = trainer.is_first_attempt and trainer.steps_this_attempt == 0
        fatal = isinstance(e, FATAL_TYPES) or setup_failure_on_launch
        kind = "FATAL" if fatal else "RETRYABLE"
        if isinstance(e, RunConflictError):
            print(f"FATAL: {e}", file=sys.stderr, flush=True)
            return {"status": "failed", "run": cfg.name, "error": str(e)}
        print(tb, file=sys.stderr, flush=True)
        try:
            run.log(f"{kind} {type(e).__name__}: {e}")
            err = {"type": type(e).__name__, "message": str(e), "traceback": tb[-4000:], "at": utcnow(),
                   "global_step": trainer.gs}
            if fatal:
                trainer.write_status("failed", error=err, last_error=err)
            else:
                trainer.write_status("restarting", last_error=err)
        except Exception as e2:  # don't mask the original error
            print(f"(could not record failure in status.json: {e2!r})", file=sys.stderr, flush=True)
        if fatal:
            return {"status": "failed", "run": cfg.name, "error": f"{type(e).__name__}: {e}"}
        raise
