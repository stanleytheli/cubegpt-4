"""CubeGPT-3 on Modal: resumable training and evaluation on H100/H200.

Setup (once):
    /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    source .venv/bin/activate        # modal CLI on Python 3.12 (matches the container)
    # optional: put existing weights in the volume (for init_from / evaluation)
    modal volume put cubegpt-3 models/cubegpt-3-8a1-t300m.pth models/cubegpt-3-8a1-t300m.pth

Train. Use --detach for anything real: the run then survives closing the terminal
or your laptop sleeping. Without it, disconnecting STOPS the run.
    modal run --detach modal_app.py::train --config configs/cubegpt3_8a1_t100m.toml
    modal run --detach modal_app.py::train --config configs/smoke.toml --name smoke-2 --overrides "gpu=H200,seed=3"
    modal run --detach modal_app.py::train --config configs/X.toml --resume    # continue a stopped/failed run

Batch-size experiments: write stage schedules in samples (samples,
plateau_patience_samples, eval_every_samples; see configs/cubegpt3_8a1_t100m.toml),
then sweep with "*" overrides; lr_scale multiplies every stage's lr:
    modal run --detach modal_app.py::train --config configs/cubegpt3_8a1_t100m.toml \
        --name t100m-bs4096 --overrides "stages.*.batch_size=4096,lr_scale=4"
Measure throughput / memory per batch size first (also re-verifies the fast MoE
dispatch against the reference implementation on that GPU):
    modal run modal_app.py::bench --batch-sizes 256,1024,4096,16384 --gpu H100!
    modal run modal_app.py::bench --checkpoint models/cubegpt-3-8a1-t300m.pth   # + checks on real weights

Monitor (reads the volume directly; no GPU):
    python status.py                  # all runs
    python status.py <run>            # one run: progress, losses, ETA, attempts, recent log
    python status.py <run> --watch 60
    modal app logs <app-id>           # live stdout (app id is printed by `modal run`)
    Files: runs/<run>/{status.json, metrics.jsonl, log.txt, stage_metrics.json, final.pth, ckpt/latest.pt}
    modal volume get cubegpt-3 runs/<run>/final.pth .

Evaluate (MSE / within-k accuracy, and optional beam-search solving split over N GPUs):
    modal run modal_app.py::evaluate --run <run> --solve-cubes 100 --widths 4000,16000 --shards 4
    modal run modal_app.py::evaluate --run <run> --which latest          # an in-progress run's checkpoint
    modal run modal_app.py::evaluate --checkpoint models/cubegpt-3-8a1-t300m.pth \
        --model-config configs/cubegpt3_8a1_t100m.toml
    Results are printed and written to eval_results/*.json, locally and in the volume.
    Add --detach (modal run --detach modal_app.py::evaluate ...) for long solve evals:
    they then finish even if your computer goes off; fetch the JSON from the volume after.
    Add --save-beam to also record the whole beam search of every cube (ranks in the JSON,
    full beams in the volume at eval_results/<name>.beams/; about 0.5 MB per cube at width 4000).
    Add --dedupe to keep one copy of each cube state per beam step (U D and D U reach the
    same state; without this about half of a width-4000 beam is such copies).
    Add --width-schedule 6000x11,3000,1500,750,500 for a beam that is 6000 wide through move 11
    and then shrinks (the last width repeats); results are filed under the largest width.
    Add --score-mix 0.1 to rank by 0.9 * model(state) + 0.1 * mean of the model's predictions
    for its parent and all the parent's children (score mixing).
    Add --parent-alpha 0.1 to rank beam entries by an exponential mean of the predictions
    along their path: rating = 0.9 * model(state) + 0.1 * rating(parent).
    Every solved cube is saved to the volume as it finishes; if an eval dies anyway:
    modal run modal_app.py::collect_eval --out-name <name printed at launch>

Stop a run:  modal app stop <app-id>   (or the dashboard). Resume later with --resume.

How the robustness works (details in cubegpt/train.py):
  * Fail fast: the config is validated and the volume checked locally before any GPU
    is requested; on the GPU, self-tests, a GPU-vs-reference data check, a fwd+bwd at
    every stage's batch size, the metrics code and a checkpoint round-trip all run
    in the first minute.
  * Restarts: Modal retries the call (fresh container) after preemption, a crash, the
    watchdog firing, or the per-attempt 24h limit; each attempt resumes from
    runs/<run>/ckpt/latest.pt (saved every checkpoint_every_minutes).
  * Deterministic failures (NaN loss, OOM, bad config, code errors, any setup error on
    a fresh launch) are NOT retried: status.json says "failed" and this command exits 1.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "cubegpt-3"
VOLUME_NAME = "cubegpt-3"
VOL = "/vol"
REMOTE_TABLES = "/root/lookup_tables/cube_utils.json"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "numpy==2.4.6")
    .env({"PYTHONUNBUFFERED": "1", "CUBEGPT_TABLES": REMOTE_TABLES})
    .add_local_file(ROOT / "lookup_tables" / "cube_utils.json", REMOTE_TABLES)
    .add_local_python_source("cubegpt")
)
app = modal.App(APP_NAME, image=image)


def _retries(n):
    # Restart immediately: after preemption / timeout we want to resume ASAP.
    return modal.Retries(max_retries=n, initial_delay=0.0, backoff_coefficient=1.0)


# gpu / timeout / retries below are defaults; `train` overrides them per run config.
@app.function(gpu="H100", volumes={VOL: volume}, timeout=24 * 3600, retries=_retries(10),
              single_use_containers=True)
def train_remote(cfg: dict, resume: bool = False) -> dict:
    from cubegpt.train import run_training
    return run_training(cfg, VOL, commit=volume.commit, call_id=modal.current_function_call_id(), resume=resume,
                        device="cuda")


@app.function(gpu="H100", volumes={VOL: volume}, timeout=24 * 3600, retries=_retries(2),
              single_use_containers=True)
def eval_remote(spec: dict, shard: int, n_shards: int) -> dict:
    from cubegpt.evaluate import run_eval_shard
    return run_eval_shard(spec, VOL, shard, n_shards, device="cuda", commit=volume.commit)


def _combine_eval(spec: dict, gpu: str, outs: list, parts_root=None) -> dict:
    """Merge shard outputs. Shards that failed (exceptions) are filled in from the
    per-cube parts they persisted to the volume before failing."""
    from cubegpt.evaluate import eval_parts_dir, load_eval_parts, summarize_solves
    ok = [o for o in outs if isinstance(o, dict)]
    failed = [i for i, o in enumerate(outs) if not isinstance(o, dict)]
    solves = {(r["idx"], r["width"]): r for o in ok for r in o["solves"]}
    metrics = next((o["metrics"] for o in ok if o["metrics"]), None)
    if failed and parts_root is not None:
        done, saved_metrics = load_eval_parts(eval_parts_dir(parts_root, spec["out_name"]))
        solves = {**done, **solves}
        metrics = metrics or saved_metrics
    solves = sorted(solves.values(), key=lambda r: (r["width"], r["idx"]))
    expected = spec.get("solve_cubes", 0) * len(spec.get("widths", []))
    return {"spec": spec, "gpu": ok[0]["gpu"] if ok else None, "gpu_requested": gpu,
            "expert_stats": ok[0]["expert_stats"] if ok else None, "metrics": metrics,
            "solve_summary": summarize_solves(solves) if solves else {}, "solves": solves,
            "complete": len(solves) == expected and not (spec.get("metrics_samples") and metrics is None),
            "failed_shards": [f"{i}: {type(outs[i]).__name__}: {outs[i]}" for i in failed]}


# CPU-only coordinator: fans the shards out to GPUs and writes the combined result to the
# volume, so an evaluation launched with --detach finishes even if your computer goes off.
@app.function(volumes={VOL: volume}, timeout=24 * 3600, retries=0, cpu=1.0)
def eval_job(spec: dict, n_shards: int, gpu: str, out_name: str) -> dict:
    spec = {**spec, "out_name": out_name}
    parts = Path(VOL) / "eval_results" / f"{out_name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    (parts / "spec.json").write_text(json.dumps(spec))
    volume.commit()
    outs = list(eval_remote.with_options(gpu=gpu).starmap([(spec, s, n_shards) for s in range(n_shards)],
                                                          return_exceptions=True))
    volume.reload()
    result = _combine_eval(spec, gpu, outs, parts_root=VOL)
    out = Path(VOL) / "eval_results" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(gpu="H100!", cpu=8.0, timeout=1800, retries=0)
def perf_remote(spec: dict) -> dict:
    from cubegpt.perf import run_perf
    return run_perf(spec)


@app.function(gpu="H100", volumes={VOL: volume}, timeout=3600, retries=0)
def bench_remote(spec: dict) -> dict:
    from cubegpt.verify import run_bench
    return run_bench(spec, VOL, device="cuda")


# ----------------------------------------------------------------------------
# Local helpers (run on your machine; read the volume without a container)
# ----------------------------------------------------------------------------
def _vol_read(path: str):
    try:
        return b"".join(volume.read_file(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def _vol_json(path: str):
    data = _vol_read(path)
    return None if data is None else json.loads(data)


def _vol_exists(path: str) -> bool:
    parent, _, leaf = path.rstrip("/").rpartition("/")
    try:
        return any(Path(e.path).name == leaf for e in volume.listdir(parent or "/"))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False


def _age_seconds(iso):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def _die(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


@app.local_entrypoint()
def train(config: str, name: str = "", resume: bool = False, overrides: str = ""):
    """Launch (or --resume) a training run described by a TOML config."""
    from cubegpt import config as C

    try:
        cfg = C.load_run_config(config, name=name, overrides=overrides)
    except (C.ConfigError, OSError) as e:
        _die(f"invalid config: {e}")
    cfg_dict = C.to_dict(cfg)

    # ---- fail fast on the volume state, before requesting a GPU
    run_path = f"runs/{cfg.name}"
    stored = _vol_json(f"{run_path}/config.json")
    if stored is not None and not resume:
        _die(f"run {cfg.name!r} already exists in volume {VOLUME_NAME!r}. Pass --resume to continue it, "
             f"or --name to start a new run.")
    if stored is None and resume:
        _die(f"--resume given but run {cfg.name!r} does not exist")
    if stored is not None:
        diffs = C.diff_training_fields(stored, cfg_dict)
        if diffs:
            _die("cannot resume: training config differs from the stored one:\n  " + "\n  ".join(diffs))
        status = _vol_json(f"{run_path}/status.json") or {}
        if status.get("state") == "completed":
            _die(f"run {cfg.name!r} is already completed")
        age = _age_seconds(status.get("updated_at"))
        live_window = 3 * 60 * status.get("status_every_minutes", 2) + 300
        if status.get("state") in ("starting", "running", "restarting") and age is not None and age < live_window:
            _die(f"run {cfg.name!r} looks live (heartbeat {age / 60:.1f} min ago). Stop it first "
                 f"(`modal app stop <app-id>`) or wait {int((live_window - age) / 60) + 1} min for the "
                 "heartbeat to go stale.")
    if cfg.init_from and not _vol_exists(cfg.init_from):
        _die(f"init_from {cfg.init_from!r} not found in volume {VOLUME_NAME!r} "
             f"(upload with: modal volume put {VOLUME_NAME} <local file> {cfg.init_from})")

    total = C.total_samples(cfg)
    print(f"\nrun      {cfg.name}{'  (resuming)' if resume else ''}")
    print(f"gpu      {cfg.gpu} | per-attempt timeout {cfg.timeout_hours}h | max retries {cfg.max_retries}")
    print(f"model    {cfg.model}")
    for i, st in enumerate(cfg.stages):
        sched = C.describe_schedule(st)
        print(f"stage {i + 1}  {C.stage_steps(st):>8} steps x {st.batch_size:<6} lr {C.stage_lr(cfg, st):<8.3g} "
              f"{sched} | eval every {C.stage_eval_every(st)} | data={st.data} {st.label}")
    print(f"total    {total / 1e6:.1f}M samples")

    fn = train_remote.with_options(gpu=cfg.gpu, timeout=int(cfg.timeout_hours * 3600),
                                   retries=_retries(cfg.max_retries))
    call = fn.spawn(cfg_dict, resume)
    print(f"\nspawned function call {call.object_id}")
    print(f"status:  python status.py {cfg.name}")
    print("(if you did not pass --detach, closing this terminal stops the run)\n")
    try:
        result = call.get()
    except Exception as e:
        _die(f"run {cfg.name!r} failed after exhausting Modal retries: {type(e).__name__}: {e}\n"
             f"See runs/{cfg.name}/status.json and log.txt (python status.py {cfg.name}).")
    print(f"\nresult: {json.dumps(result, indent=2, default=str)}")
    if result.get("status") != "completed":
        _die(f"run {cfg.name!r} failed: {result.get('error')}")


@app.local_entrypoint()
def evaluate(run: str = "", which: str = "final", checkpoint: str = "", model_config: str = "",
             gpu: str = "H100", metrics_samples: int = 262144, solve_cubes: int = 0, widths: str = "4000",
             scramble_moves: int = 30, max_moves: int = 30, seed: int = 0, shards: int = 1,
             eval_batch_size: int = 16384, tf32: bool = True, save_beam: bool = False,
             parent_alpha: float = 0.0, dedupe: bool = False, width_schedule: str = "",
             score_mix: float = 0.0):
    """Evaluate a run (--run, --which final|latest|<path in run dir>) or any checkpoint in
    the volume (--checkpoint + --model-config <toml with a [model] table>)."""
    from cubegpt import config as C

    if bool(run) == bool(checkpoint):
        _die("give exactly one of --run or --checkpoint")
    if gpu not in C.GPUS:
        _die(f"--gpu must be one of {C.GPUS}")
    try:
        width_list = [int(w) for w in widths.split(",") if w.strip()]
    except ValueError:
        _die(f"--widths must be comma-separated integers, got {widths!r}")
    if solve_cubes and (not width_list or min(width_list) <= 0):
        _die("--widths must be positive")
    if shards < 1:
        _die("--shards must be >= 1")
    schedule = None
    if width_schedule:                     # "6000x11,3000,1500,750,500": width after each move, last repeats
        try:
            schedule = [int(w) for part in width_schedule.split(",")
                        for w in [part.split("x")[0]] * (int(part.split("x")[1]) if "x" in part else 1)]
        except (ValueError, IndexError):
            _die(f"--width-schedule must look like 6000x11,3000,1500,750,500, got {width_schedule!r}")
        if not schedule or min(schedule) <= 0:
            _die("--width-schedule widths must be positive")
        width_list = [max(schedule)]       # label only; the search follows the schedule

    if run:
        stored = _vol_json(f"runs/{run}/config.json")
        if stored is None:
            _die(f"run {run!r} not found in volume {VOLUME_NAME!r}")
        model = stored["model"]
        rel = {"final": "final.pth", "latest": "ckpt/latest.pt"}.get(which, which)
        checkpoint = f"runs/{run}/{rel}"
        label = f"{run}-{which.replace('/', '_')}"
    else:
        if not model_config:
            _die("--checkpoint needs --model-config (a TOML file with a [model] table)")
        try:
            model = C.to_dict(C.load_model_config(model_config))
        except (C.ConfigError, OSError) as e:
            _die(f"invalid model config: {e}")
        label = Path(checkpoint).stem
    if not _vol_exists(checkpoint):
        _die(f"checkpoint {checkpoint!r} not found in volume {VOLUME_NAME!r}")

    n_shards = max(1, min(shards, solve_cubes)) if solve_cubes else 1
    spec = {"checkpoint": checkpoint, "model": model, "metrics_samples": metrics_samples,
            "solve_cubes": solve_cubes, "widths": width_list, "scramble_moves": scramble_moves,
            "max_moves": max_moves, "seed": seed, "eval_batch_size": eval_batch_size, "tf32": tf32,
            "save_beam": save_beam, "parent_alpha": parent_alpha, "dedupe": dedupe, "width_schedule": schedule, "mix_alpha": score_mix}
    if score_mix:
        label += f"-mix{score_mix:g}"
    if schedule:
        label += "-ws" + width_schedule.replace(",", "-")
    if dedupe:
        label += "-dedupe"
    if parent_alpha:
        label += f"-pa{parent_alpha:g}"
    print(f"evaluating {checkpoint} on {n_shards} x {gpu}: metrics on {metrics_samples} samples, "
          f"{solve_cubes} cubes x widths {width_list}")
    out_name = f"{label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    t0 = time.time()
    call = eval_job.spawn(spec, n_shards, gpu, out_name)
    print(f"results will be written to the volume: eval_results/{out_name}.json")
    print(f"(with --detach the eval finishes even if this terminal closes; fetch it later with\n"
          f"   modal volume get {VOLUME_NAME} eval_results/{out_name}.json eval_results/ )")
    result = call.get()
    _report_eval(result, checkpoint, out_name, f"{time.time() - t0:.0f}s wall")


def _report_eval(result: dict, checkpoint: str, out_name: str, note: str = ""):
    from cubegpt.evaluate import format_metrics
    print(f"\n==== {checkpoint} ({note}) ====")
    if result.get("failed_shards") or not result.get("complete", True):
        print(f"WARNING: incomplete evaluation ({len(result['solves'])} of "
              f"{result['spec'].get('solve_cubes', 0) * len(result['spec'].get('widths', []))} cubes). "
              f"Failed shards: {result.get('failed_shards')}")
    r = result.get("expert_stats")
    if r:
        print(f"MoE routing (measured): {r['dead_experts']} dead experts, per layer {r['dead_per_layer']}, "
              f"bias spread {r['bias_spread']:.2f}")
    if result.get("metrics"):
        print(format_metrics(result["metrics"]))
    for w, s in result["solve_summary"].items():
        print(f"width {w:>6}: solved {100 * s['solve_rate']:.0f}% of {s['n']} | mean moves {s['mean_moves']:.2f} "
              f"| MAO {s['mao']:.2f} | median {s['median_moves']:.0f} | {s['mean_seconds']:.1f}s/cube")
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


@app.local_entrypoint()
def collect_eval(out_name: str):
    """Assemble an evaluation that was interrupted before it could write its result,
    from the per-cube progress its shards saved in eval_results/<out_name>.parts/."""
    from cubegpt.evaluate import summarize_solves
    parts = f"eval_results/{out_name}.parts"
    spec = _vol_json(f"{parts}/spec.json")
    if spec is None:
        _die(f"no saved progress at {parts} in volume {VOLUME_NAME!r}")
    solves = {}
    for e in volume.listdir(parts):
        if Path(e.path).name.startswith("shard"):
            for line in _vol_read(e.path).decode().splitlines():
                if line.strip():
                    r = json.loads(line)
                    solves[(r["idx"], r["width"])] = r
    solves = sorted(solves.values(), key=lambda r: (r["width"], r["idx"]))
    expected = spec.get("solve_cubes", 0) * len(spec.get("widths", []))
    result = {"spec": spec, "gpu": None, "expert_stats": None, "metrics": _vol_json(f"{parts}/metrics.json"),
              "solve_summary": summarize_solves(solves) if solves else {}, "solves": solves,
              "complete": len(solves) == expected, "failed_shards": [] if len(solves) == expected else ["collected"]}
    _report_eval(result, spec["checkpoint"], out_name + "_collected", "collected from saved progress")


@app.local_entrypoint()
def bench(config: str = "configs/cubegpt3_8a1_t100m.toml", batch_sizes: str = "256,1024,4096,16384",
          gpu: str = "H100!", impls: str = "grouped,loop", seconds: float = 10.0, verify: bool = True,
          checkpoint: str = "", tf32: bool = True, compare: str = ""):
    """Training throughput + peak memory per batch size for the [model] in --config on
    one GPU, after re-verifying the fast MoE dispatch against the reference on that GPU.
    --checkpoint (volume path) adds checks on real weights (metrics, beam search).
    --compare a.toml,b.toml also times those [model]s in the same container (like-for-like speed)."""
    from cubegpt import config as C

    if gpu not in C.GPUS:
        _die(f"--gpu must be one of {C.GPUS}")
    try:
        model = C.to_dict(C.load_model_config(config))
        sizes = [int(b) for b in batch_sizes.split(",") if b.strip()]
        compare_models = [C.to_dict(C.load_model_config(p.strip())) for p in compare.split(",") if p.strip()]
    except (C.ConfigError, OSError, ValueError) as e:
        _die(f"bad arguments: {e}")
    impl_list = [i.strip() for i in impls.split(",") if i.strip()]
    if not sizes or min(sizes) <= 0 or not set(impl_list) <= {"grouped", "loop"}:
        _die("--batch-sizes must be positive ints; --impls a subset of grouped,loop")
    if checkpoint and not _vol_exists(checkpoint):
        _die(f"checkpoint {checkpoint!r} not found in volume {VOLUME_NAME!r}")
    spec = {"model": model, "batch_sizes": sizes, "impls": impl_list, "seconds": seconds, "verify": verify,
            "checkpoint": checkpoint, "tf32": tf32, "compare_models": compare_models}
    out = bench_remote.with_options(gpu=gpu).remote(spec)

    print(f"\n==== {out['gpu']} | {model} ====")
    rows = {(r["batch_size"], r["impl"]): r for r in out["throughput"]}
    print(f"{'batch':>7}  " + "  ".join(f"{i + ' samples/s':>20} {'ms/step':>8} {'GiB':>5}" for i in impl_list)
          + ("   speedup" if len(impl_list) == 2 else ""))
    for bs in sizes:
        cells, rates = [], []
        for i in impl_list:
            r = rows[(bs, i)]
            if r.get("oom"):
                cells.append(f"{'OOM':>20} {'':>8} {'':>5}")
                rates.append(None)
            else:
                cells.append(f"{r['samples_per_sec']:>20,.0f} {r['ms_per_step']:>8.1f} {r['peak_mem_gib'] or 0:>5.1f}")
                rates.append(r["samples_per_sec"])
        sp = (f"   {rates[0] / rates[1]:.1f}x" if len(rates) == 2 and None not in rates else "")
        print(f"{bs:>7}  " + "  ".join(cells) + sp)
    if out.get("compare"):
        print(f"\n{'model (same GPU)':<44} {'params':>11} {'batch':>6} {'samples/s':>11}")
        for r in out["compare"]:
            m = r["model"]
            desc = (f"{m['arch']} d{m['d_model']} f{m['d_ffwd']} L{m['n_layers']}"
                    + (f" {m['n_experts']}E{m['n_active']}" if m["arch"] == "moe" else ""))
            print(f"{desc:<44} {r['params']:>11,} {r['batch_size']:>6} "
                  + ("OOM" if r.get("oom") else f"{r['samples_per_sec']:>11,.0f}"))
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"bench_{gpu.replace('!', '')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path.relative_to(ROOT)}")


@app.local_entrypoint()
def perf(seconds: float = 5.0, batch_size: int = 1024, layouts: str = ""):
    """Training-step profile at one batch size, all in one container: layouts, Adam implementation,
    step breakdown, operator profile, torch.compile, and several runs sharing the GPU."""
    spec = {"seconds": seconds, "batch_size": batch_size}
    if layouts:   # --layouts a,b,c: only time those layouts back to back (names in cubegpt/perf.py LAYOUTS)
        spec.update(layouts=[x.strip() for x in layouts.split(",")], fused=[None], breakdown=[], profile=False,
                    compile=[], concurrency=[])
    out = perf_remote.remote(spec)
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"perf_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path.relative_to(ROOT)}")
