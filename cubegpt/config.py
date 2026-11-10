"""Run configuration: TOML files -> validated dataclasses.

Pure Python (no torch) so the local Modal client can validate a config and fail
*before* any GPU is requested. The remote side re-validates the same dict.

A run config looks like (see configs/ for real examples):

    name = "8a1-t100m"
    gpu = "H100"
    [model]
    arch = "moe"
    d_model = 256
    ...
    [[stages]]
    steps = 100000
    batch_size = 256
    lr = 1e-4
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

ARCHS = ("moe", "gated", "transformer", "pair")
ROUTERS = ("sigmoid", "softmax")
DATA_KINDS = ("random", "path")
SCHEDULES = ("torch_plateau", "eval_plateau", "wsd", "cosine")
FIXED_SCHEDULES = ("wsd", "cosine")
SCRAMBLE_LIMIT = 21  # mirrors cube.SCRAMBLE_LIMIT (kept torch-free here)
# Single-GPU names Modal accepts. "H100" may be silently upgraded to an H200 at
# H100 price; "H100!" pins a real H100 (use it for benchmarking).
GPUS = ("H100", "H100!", "H200", "B200", "B200+", "B300", "A100-80GB", "A100-40GB", "A100",
        "L40S", "L4", "A10", "T4", "RTX-PRO-6000")
MAX_MODAL_RETRIES = 10

# Fields that may differ when resuming an existing run (they don't change what is trained).
OPERATIONAL_KEYS = frozenset({
    "gpu", "timeout_hours", "max_retries", "checkpoint_every_minutes", "status_every_minutes",
    "watchdog_minutes", "max_stalled_restarts", "debug_crash_at_step", "divergence_factor",
    "divergence_rollbacks", "divergence_lr_factor", "adam_fused",
})


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    arch: str = "moe"            # "moe" | "gated" | "transformer" | "pair" (rotation-invariant encoding with
                                 # a pair table and MoE FFNs, cubegpt/invariant.py; train it with
                                 # train_canonical = false)
    d_model: int = 256
    n_heads: int = 16
    d_ffwd: int = 341
    n_layers: int = 8
    n_experts: int = 8           # moe only
    n_active: int = 1            # moe only
    first_dense_layers: int = 0  # moe only: the first k layers use a dense gated FFN (width d_ffwd)
    ffn_layers: str = ""         # moe/gated: one character per layer: '1' attention + FFN, '0' attention
                                 # only, 'f' FFN only, e.g. "00001111" or "00000000ffff"; "" = all '1'.
                                 # No FFN at all needs arch = "gated".
    ffn_scale_init: float = 0.0  # gated only: > 0 multiplies each FFN output by a learnable per-channel
                                 # scale starting at this value (LayerScale); 0 = off
    router: str = "sigmoid"      # moe only: "sigmoid" (DeepSeek-V3 style; the published 8A1 models) or
                                 # "softmax" (Switch style: the chosen expert's weight is its softmax
                                 # probability, >= 1/n_experts for top-1, so it can't sink toward 0)
    d_pair: int = 64             # pair only: width of the shared pair-class embedding
    pre_norm: bool = False       # moe/gated: pre-norm layers, x + f(norm(x)), as in modern transformers,
                                 # instead of the original post-norm, norm(x + f(x))


@dataclass
class StageConfig:
    """One optimizer phase. Each stage starts a fresh Adam + ReduceLROnPlateau,
    exactly like the notebook's successive training cells.

    For batch-size experiments, give lengths in samples (samples,
    plateau_patience_samples, eval_every_samples): then changing batch_size keeps
    the schedule fixed in data terms. The *_samples fields win when > 0."""
    batch_size: int = 4096
    lr: float = 1e-4             # multiplied by the run's lr_scale
    steps: int = 0               # give exactly one of steps / samples
    samples: int = 0
    # LR decay within the stage (each new stage restarts at its own lr: warm restarts).
    #   "torch_plateau": torch ReduceLROnPlateau on the per-step training loss (the
    #       notebook's scheduler; noise-driven, see cubegpt/schedule.py). Uses plateau_patience.
    #   "eval_plateau": trend of the held-out eval loss (cubegpt/schedule.py). Uses
    #       plateau_window_samples and plateau_threshold.
    #   "wsd": warmup-stable-decay. Linear warmup over warmup_samples, constant lr, then
    #       linear decay to final_lr_fraction * lr over the last decay_samples.
    #   "cosine": linear warmup, then cosine decay to final_lr_fraction * lr by the stage end.
    schedule: str = "torch_plateau"
    warmup_samples: int = 0      # wsd / cosine
    decay_samples: int = 0       # wsd
    final_lr_fraction: float = 0.0  # wsd / cosine
    plateau_factor: float = 0.5
    plateau_min_lr: float = 0.0  # never decay below this (both schedules)
    plateau_patience: int = 2000  # torch_plateau: optimizer steps without a new best batch loss
    plateau_patience_samples: int = 0
    plateau_window_samples: int = 5_120_000  # eval_plateau: trend window (needs >= 3 evals)
    plateau_threshold: float = 0.005  # eval_plateau: decay if relative improvement across the window is below this
    eval_every: int = 1000       # steps between held-out evals / log lines
    eval_every_samples: int = 0
    data: str = "random"         # "random": depths uniform in 0..20; "path": whole 20-move paths
    depth_weight_power: float = 0.0  # random data only: draw depth d with weight max(d, 1) ** power
                                     # (1 = depth 18 appears 18x as often as depth 1). Held-out eval
                                     # and metrics stay on uniform depths.
    label: str = ""


@dataclass
class RunConfig:
    name: str = ""
    gpu: str = "H100"
    seed: int = 0
    timeout_hours: float = 24.0          # per Modal attempt (max 24); runs continue across attempts
    max_retries: int = 10                # Modal retries (preemption, timeouts, crashes)
    tf32: bool = True
    lr_scale: float = 1.0                # multiplies every stage's lr (handy for batch-size sweeps)
    moe_bias_update_speed: float = 1e-3  # moe only: DeepSeek-V3 balancing-bias step per optimizer step (0 = off)
    moe_aux_loss: float = 0.0            # moe only: Switch-style load-balancing loss coefficient (typically 0.01),
                                         # E * sum_e(token fraction_e * mean router prob_e), averaged over layers
    moe_z_loss: float = 0.0              # moe only: ST-MoE router z-loss coefficient (typically 1e-3),
                                         # mean(logsumexp(router logits)^2): keeps router logits small
    router_lr_mult: float = 1.0          # moe only: the router (gate) weights train at this x the lr. At high lr
                                         # the sigmoid router saturates (scores ~0.001) and the balancing bias,
                                         # not the token, decides routing; a slower router keeps it healthy
    train_canonical: bool = True         # canonicalize training states (min over the 96 symmetries). False
                                         # trains on raw scrambles, for models that don't need canonical
                                         # inputs (arch = "pair"). Held-out eval states stay canonical.
    grad_clip: float = 0.0               # clip the global gradient norm to this before each optimizer step
                                         # (0 = off; 1.0 is the usual value). The mean / max pre-clip norm per
                                         # eval interval is logged and written to metrics.jsonl.
    adam_beta1: float = 0.9              # Adam betas / eps. PyTorch defaults; common LLM recipes use beta2 0.95
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0            # > 0 switches to AdamW (decoupled decay), applied to weight matrices
                                         # and embeddings only, not to biases or LayerNorm parameters
    optimizer: str = "adam"              # "adam" (Adam / AdamW) or "muon": Muon on the transformer stack's weight
                                         # matrices, Adam / AdamW (settings above) on everything else
    muon_lr_mult: float = 25.0           # Muon's lr = stage lr x this (stage lr 8e-4 -> Muon lr 0.02)
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0       # decoupled decay on the Muon matrices (weight_decay covers the rest)
    adam_fused: bool = True              # use PyTorch's fused Adam on GPU: same update, one kernel per tensor
                                         # (about 7% faster per step for the 8-expert MoE at batch 1024)
    data_seed: int = -1                  # seed of the training-data stream; -1 = use `seed`. Set it when
                                         # continuing from another run's weights (init_from), so the new run
                                         # doesn't replay the samples that run saw. The held-out eval set
                                         # depends on `seed` only, so eval losses stay comparable.
    init_from: str = ""                  # volume path of weights to start from, e.g. "models/x.pth"
    gen_chunk_samples: int = 1048576     # samples generated per data-generation round
    gen_max_batch: int = 65536           # max states per canonicalization call (bounds memory)
    eval_samples: int = 65536            # fixed held-out set for periodic eval loss
    final_eval_samples: int = 262144     # MSE / within-k accuracy at end of each stage (0 = skip)
    checkpoint_every_minutes: float = 10.0
    status_every_minutes: float = 2.0    # heartbeat: status.json refresh + volume commit
    keep_every_samples: int = 0          # also keep model snapshots every N samples (0 = off)
    watchdog_minutes: float = 30.0       # kill + restart the attempt if no step completes for this long
    max_stalled_restarts: int = 3        # give up after this many restarts without checkpoint progress
    divergence_factor: float = 1.5       # fail fast if an eval loss exceeds this x the best so far (0 = off);
                                         # catches high-lr blowups that never produce a NaN
    divergence_rollbacks: int = 0        # on divergence, roll back to the last healthy checkpoint and multiply
                                         # the peak lr by divergence_lr_factor, up to this many times (wsd/cosine
                                         # stages only); 0 = fail the run instead
    divergence_lr_factor: float = 0.7
    debug_crash_at_step: int = 0         # testing only: hard-crash once at this global step
    model: ModelConfig = field(default_factory=ModelConfig)
    stages: List[StageConfig] = field(default_factory=list)


_SCALARS = {"int": int, "float": float, "str": str, "bool": bool}


def _check_scalar(value, type_name, where):
    if type_name == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if type_name == "int" and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got bool")
    expected = _SCALARS[type_name]
    if not isinstance(value, expected):
        raise ConfigError(f"{where}: expected {type_name}, got {type(value).__name__} ({value!r})")
    return value


def _build(cls, data, where):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a table, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where or 'top level'}: unknown key(s) {unknown}. Valid keys: {sorted(known)}")
    kwargs = {}
    for key, value in data.items():
        t = known[key].type
        sub = f"{where}.{key}" if where else key
        if t == "ModelConfig":
            kwargs[key] = _build(ModelConfig, value, sub)
        elif t == "List[StageConfig]":
            if not isinstance(value, list):
                raise ConfigError(f"{sub}: expected an array of tables ([[stages]])")
            kwargs[key] = [_build(StageConfig, v, f"{sub}[{i}]") for i, v in enumerate(value)]
        else:
            kwargs[key] = _check_scalar(value, t, sub)
    return cls(**kwargs)


def stage_steps(st: StageConfig) -> int:
    return st.steps if st.steps > 0 else math.ceil(st.samples / st.batch_size)


def stage_patience(st: StageConfig) -> int:
    """ReduceLROnPlateau patience in optimizer steps."""
    if st.plateau_patience_samples > 0:
        return max(1, round(st.plateau_patience_samples / st.batch_size))
    return st.plateau_patience


def stage_window_evals(st: StageConfig) -> int:
    """eval_plateau window length in evals."""
    return round(st.plateau_window_samples / (stage_eval_every(st) * st.batch_size))


def stage_eval_every(st: StageConfig) -> int:
    if st.eval_every_samples > 0:
        return max(1, round(st.eval_every_samples / st.batch_size))
    return st.eval_every


def stage_warmup_steps(st: StageConfig) -> int:
    return round(st.warmup_samples / st.batch_size)


def stage_decay_steps(st: StageConfig) -> int:
    return round(st.decay_samples / st.batch_size)


def describe_schedule(st: StageConfig) -> str:
    if st.schedule == "eval_plateau":
        s = f"eval-plateau window {stage_window_evals(st)} evals, threshold {100 * st.plateau_threshold:g}%"
    elif st.schedule == "torch_plateau":
        s = f"torch-plateau patience {stage_patience(st)}"
    elif st.schedule == "wsd":
        return (f"wsd warmup {stage_warmup_steps(st)} steps, decay over last {stage_decay_steps(st)} steps "
                f"to {st.final_lr_fraction:g}x")
    else:
        return f"cosine warmup {stage_warmup_steps(st)} steps, decay to {st.final_lr_fraction:g}x"
    return s + (f", min lr {st.plateau_min_lr:g}" if st.plateau_min_lr else "")


def stage_lr(cfg: "RunConfig", st: StageConfig) -> float:
    return st.lr * cfg.lr_scale


def total_samples(cfg: RunConfig) -> int:
    return sum(stage_steps(s) * s.batch_size for s in cfg.stages)


def validate_model(m: ModelConfig, where="model"):
    if m.arch not in ARCHS:
        raise ConfigError(f"{where}.arch must be one of {ARCHS}, got {m.arch!r}")
    if m.arch == "pair" and m.n_layers < 2:
        raise ConfigError(f"{where}: arch = \"pair\" needs n_layers >= 2")
    for k in ("d_model", "n_heads", "d_ffwd", "n_layers", "n_experts", "n_active", "d_pair"):
        if getattr(m, k) <= 0:
            raise ConfigError(f"{where}.{k} must be positive")
    if m.d_model % m.n_heads:
        raise ConfigError(f"{where}: d_model ({m.d_model}) must be divisible by n_heads ({m.n_heads})")
    if m.first_dense_layers and m.arch != "moe":
        raise ConfigError(f"{where}.first_dense_layers only applies to arch = \"moe\"")
    if not 0 <= m.first_dense_layers < m.n_layers:
        raise ConfigError(f"{where}.first_dense_layers must be in [0, n_layers) (use arch = \"gated\" for all-dense)")
    if not isinstance(m.ffn_layers, str):
        raise ConfigError(f"{where}.ffn_layers must be a quoted string like \"00001111\", got {m.ffn_layers!r}")
    if m.ffn_layers:
        if m.arch not in ("moe", "gated"):
            raise ConfigError(f"{where}.ffn_layers only applies to arch = \"moe\" or \"gated\"")
        if len(m.ffn_layers) != m.n_layers or set(m.ffn_layers) - {"0", "1", "f"}:
            raise ConfigError(f"{where}.ffn_layers must be {m.n_layers} characters of 0/1/f, got {m.ffn_layers!r}")
        if m.arch == "moe" and not set(m.ffn_layers[m.first_dense_layers:]) & {"1", "f"}:
            raise ConfigError(f"{where}.ffn_layers leaves no MoE layer (use arch = \"gated\")")
    if m.ffn_scale_init < 0 or (m.ffn_scale_init and m.arch != "gated"):
        raise ConfigError(f"{where}.ffn_scale_init must be >= 0 and only applies to arch = \"gated\"")
    if m.pre_norm and m.arch not in ("moe", "gated"):
        raise ConfigError(f"{where}.pre_norm only applies to arch = \"moe\" or \"gated\"")
    if m.router not in ROUTERS:
        raise ConfigError(f"{where}.router must be one of {ROUTERS}, got {m.router!r}")
    if m.router != "sigmoid" and m.arch != "moe":
        raise ConfigError(f"{where}.router only applies to arch = \"moe\"")
    if m.arch in ("moe", "pair") and m.n_active > m.n_experts:
        raise ConfigError(f"{where}: n_active ({m.n_active}) > n_experts ({m.n_experts})")


def validate(cfg: RunConfig) -> RunConfig:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", cfg.name or ""):
        raise ConfigError(f"name {cfg.name!r} must match [A-Za-z0-9][A-Za-z0-9._-]* (it becomes a directory)")
    if cfg.gpu not in GPUS:
        hint = " (multi-GPU is not supported by this trainer)" if ":" in cfg.gpu else ""
        raise ConfigError(f"gpu must be one of {GPUS}, got {cfg.gpu!r}{hint}")
    if not 0 < cfg.timeout_hours <= 24:
        raise ConfigError("timeout_hours must be in (0, 24] (Modal's per-attempt limit)")
    if not 0 <= cfg.max_retries <= MAX_MODAL_RETRIES:
        raise ConfigError(f"max_retries must be in [0, {MAX_MODAL_RETRIES}]")
    positive = ("gen_chunk_samples", "gen_max_batch", "eval_samples", "checkpoint_every_minutes",
                "status_every_minutes", "watchdog_minutes", "max_stalled_restarts", "lr_scale")
    for k in positive:
        if getattr(cfg, k) <= 0:
            raise ConfigError(f"{k} must be positive")
    for k in ("moe_bias_update_speed", "moe_aux_loss", "moe_z_loss"):
        if getattr(cfg, k) < 0:
            raise ConfigError(f"{k} must be >= 0")
    if cfg.model.arch != "moe" and (cfg.moe_aux_loss or cfg.moe_z_loss):
        raise ConfigError("moe_aux_loss / moe_z_loss only apply to arch = \"moe\"")
    if not (0 <= cfg.adam_beta1 < 1 and 0 <= cfg.adam_beta2 < 1 and cfg.adam_eps > 0 and cfg.weight_decay >= 0):
        raise ConfigError("adam_beta1 / adam_beta2 must be in [0, 1), adam_eps > 0, weight_decay >= 0")
    if cfg.optimizer not in ("adam", "muon"):
        raise ConfigError('optimizer must be "adam" or "muon"')
    if cfg.muon_lr_mult <= 0 or not (0 <= cfg.muon_momentum < 1) or cfg.muon_weight_decay < 0:
        raise ConfigError("muon_lr_mult must be > 0, muon_momentum in [0, 1), muon_weight_decay >= 0")
    if cfg.grad_clip < 0:
        raise ConfigError("grad_clip must be >= 0 (0 = off)")
    if cfg.router_lr_mult <= 0:
        raise ConfigError("router_lr_mult must be positive")
    if cfg.router_lr_mult != 1.0 and cfg.model.arch != "moe":
        raise ConfigError("router_lr_mult only applies to arch = \"moe\"")
    if cfg.gen_max_batch < SCRAMBLE_LIMIT:
        raise ConfigError(f"gen_max_batch must be >= {SCRAMBLE_LIMIT}")
    if cfg.divergence_factor and cfg.divergence_factor <= 1:
        raise ConfigError("divergence_factor must be > 1 (or 0 to disable)")
    if cfg.divergence_rollbacks < 0 or not 0 < cfg.divergence_lr_factor < 1:
        raise ConfigError("divergence_rollbacks must be >= 0 and divergence_lr_factor in (0, 1)")
    if cfg.divergence_rollbacks and not cfg.divergence_factor:
        raise ConfigError("divergence_rollbacks needs divergence_factor > 0 (the divergence detector)")
    if cfg.divergence_rollbacks and any(st.schedule not in FIXED_SCHEDULES for st in cfg.stages):
        raise ConfigError(f"divergence_rollbacks is only supported when every stage uses schedule = {FIXED_SCHEDULES}")
    for k in ("final_eval_samples", "keep_every_samples", "debug_crash_at_step", "divergence_factor"):
        if getattr(cfg, k) < 0:
            raise ConfigError(f"{k} must be >= 0")
    if cfg.init_from and (cfg.init_from.startswith("/") or ".." in cfg.init_from.split("/")):
        raise ConfigError("init_from must be a path relative to the volume root, e.g. 'models/x.pth'")
    validate_model(cfg.model)
    if not cfg.stages:
        raise ConfigError("at least one [[stages]] entry is required")
    for i, st in enumerate(cfg.stages):
        w = f"stages[{i}]"
        if (st.steps > 0) == (st.samples > 0):
            raise ConfigError(f"{w}: give exactly one of steps / samples (> 0)")
        if st.steps < 0 or st.samples < 0:
            raise ConfigError(f"{w}: steps / samples must be >= 0")
        if st.batch_size <= 0 or st.lr <= 0 or st.eval_every <= 0 or st.plateau_patience < 0:
            raise ConfigError(f"{w}: batch_size, lr, eval_every must be > 0 and plateau_patience >= 0")
        if st.plateau_patience_samples < 0 or st.eval_every_samples < 0:
            raise ConfigError(f"{w}: plateau_patience_samples / eval_every_samples must be >= 0")
        if st.schedule not in SCHEDULES:
            raise ConfigError(f"{w}: schedule must be one of {SCHEDULES}, got {st.schedule!r}")
        if st.plateau_min_lr < 0 or st.plateau_threshold < 0:
            raise ConfigError(f"{w}: plateau_min_lr and plateau_threshold must be >= 0")
        if st.schedule == "eval_plateau" and stage_window_evals(st) < 3:
            raise ConfigError(f"{w}: plateau_window_samples ({st.plateau_window_samples}) covers only "
                              f"{stage_window_evals(st)} evals at batch {st.batch_size}; need >= 3 "
                              f"(raise it or eval more often)")
        if st.warmup_samples < 0 or st.decay_samples < 0 or not 0 <= st.final_lr_fraction < 1:
            raise ConfigError(f"{w}: warmup_samples / decay_samples must be >= 0 and final_lr_fraction in [0, 1)")
        if st.schedule not in FIXED_SCHEDULES and (st.warmup_samples or st.decay_samples or st.final_lr_fraction):
            raise ConfigError(f"{w}: warmup_samples / decay_samples / final_lr_fraction only apply to "
                              f"schedule = {FIXED_SCHEDULES}")
        if st.schedule == "cosine" and st.decay_samples:
            raise ConfigError(f"{w}: decay_samples is for wsd; cosine decays over the whole stage after warmup")
        if st.schedule in FIXED_SCHEDULES and stage_warmup_steps(st) + stage_decay_steps(st) >= stage_steps(st):
            raise ConfigError(f"{w}: warmup ({stage_warmup_steps(st)} steps) + decay ({stage_decay_steps(st)} "
                              f"steps) must be shorter than the stage ({stage_steps(st)} steps)")
        if not 0 < st.plateau_factor < 1:
            raise ConfigError(f"{w}: plateau_factor must be in (0, 1)")
        if st.data not in DATA_KINDS:
            raise ConfigError(f"{w}: data must be one of {DATA_KINDS}, got {st.data!r}")
        if st.depth_weight_power and st.data != "random":
            raise ConfigError(f"{w}: depth_weight_power only applies to data = \"random\"")
        if not math.isfinite(st.depth_weight_power):
            raise ConfigError(f"{w}: depth_weight_power must be finite")
        if st.data == "path" and st.batch_size % SCRAMBLE_LIMIT:
            raise ConfigError(f"{w}: path data needs batch_size divisible by {SCRAMBLE_LIMIT} "
                              f"(each batch is whole 0..20-move paths), got {st.batch_size}")
    return cfg


def from_dict(d: Dict[str, Any]) -> RunConfig:
    return validate(_build(RunConfig, d, ""))


def to_dict(cfg: RunConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _assign(node, parts, value, full_key):
    p = parts[0]
    if isinstance(node, list):
        if p == "*":
            targets = range(len(node))
        else:
            try:
                targets = [int(p)]
            except ValueError:
                raise ConfigError(f"override {full_key!r}: {p!r} is not a list index or '*'")
            if not -len(node) <= targets[0] < len(node):
                raise ConfigError(f"override {full_key!r}: index {p} out of range (list has {len(node)} items)")
        for i in targets:
            if len(parts) == 1:
                node[i] = value
            else:
                _assign(node[i], parts[1:], value, full_key)
    elif isinstance(node, dict):
        if len(parts) == 1:
            node[p] = value
        else:
            _assign(node.setdefault(p, {}), parts[1:], value, full_key)
    else:
        raise ConfigError(f"override {full_key!r}: cannot index into {type(node).__name__}")


def apply_overrides(d: Dict[str, Any], overrides: str) -> Dict[str, Any]:
    """overrides: comma-separated dotted assignments, e.g.
    "seed=1,model.d_model=512,stages.0.lr=3e-4,gpu=H200". Use * for every list
    element: "stages.*.batch_size=4096,lr_scale=4". Values are JSON-parsed when
    possible (numbers, true/false), otherwise kept as strings."""
    d = copy.deepcopy(d)
    for item in filter(None, (s.strip() for s in overrides.split(","))):
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not key=value")
        key, value = item.split("=", 1)
        _assign(d, key.strip().split("."), _parse_value(value), key.strip())
    return d


def read_toml(path: str) -> Dict[str, Any]:
    if tomllib is None:
        raise ConfigError("reading TOML configs needs Python >= 3.11 (use the repo's .venv)")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_run_config(path: str, name: str = "", overrides: str = "") -> RunConfig:
    d = read_toml(path)
    if overrides:
        d = apply_overrides(d, overrides)
    if name:
        d["name"] = name
    return from_dict(d)


def load_model_config(path: str) -> ModelConfig:
    """The [model] table of any TOML file (e.g. a run config)."""
    d = read_toml(path)
    if "model" not in d:
        raise ConfigError(f"{path} has no [model] table")
    m = _build(ModelConfig, d["model"], "model")
    validate_model(m)
    return m


def diff_training_fields(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Differences between two run-config dicts, ignoring OPERATIONAL_KEYS. Fields added
    after a run was stored are filled with their defaults first, so older runs stay
    resumable when the new field is left at its default."""
    a, b = (asdict(_build(RunConfig, x, "")) for x in (a, b))
    out = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if not path and k in OPERATIONAL_KEYS:
                    continue
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]")
        elif x != y:
            out.append(f"{path}: {x!r} -> {y!r}")

    walk(a, b, "")
    return out
