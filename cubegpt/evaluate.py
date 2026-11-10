"""Evaluation: held-out loss, MSE / within-k accuracy, and beam-search solving.

All evaluation data is generated from fixed seeds so numbers are comparable
across checkpoints and runs (paired comparisons, lower variance).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .cube import (CubeState, canonicalize_batch, canonicalize_full_batch, cube_string_to_state,
                   generate_samples, scrambled, state_to_cube_string, states_to_tokens)

# Mean optimal (half-turn metric) solution length of a uniformly random cube,
# used for the README's "MAO" column (= mean solve length - 17.7).
MEAN_OPTIMAL_HTM = 17.7


def _fork_rng(device):
    devices = [device] if torch.device(device).type == "cuda" else []
    return torch.random.fork_rng(devices=devices)


@torch.no_grad()
def make_eval_set(T, n, seed, max_batch=65536):
    """A fixed held-out set (X, y_ord), independent of the global RNG stream."""
    with _fork_rng(T.device):
        torch.manual_seed(seed)
        return generate_samples(T, n, "random", max_batch)


@torch.no_grad()
def eval_loss(model, X, y, batch_size=8192):
    """Mean BCE-with-logits over a fixed set. Leaves the model in eval mode."""
    model.eval()
    total = 0.0
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i + batch_size])
        total += F.binary_cross_entropy_with_logits(logits, y[i:i + batch_size], reduction="sum").item()
    return total / (X.shape[0] * y.shape[1])


def ordinal_expectation(logits):
    return torch.sigmoid(logits).sum(dim=1) - 1.0


@torch.no_grad()
def compute_metrics(model, T, n_samples=262144, seed=4242, batch_size=4096, max_batch=65536):
    """The notebook's measure_mse / measure_acc plus the README's other columns.
    Returns mse, within-3 / within-4 accuracy, rounded accuracy (with stderrs)
    and mean BCE. Leaves the model in eval mode."""
    model.eval()
    n_samples = max(batch_size, (n_samples // batch_size) * batch_size)
    sums = {k: 0.0 for k in ("se", "se2", "w3", "w4", "acc", "bce")}
    t0 = time.time()
    with _fork_rng(T.device):
        torch.manual_seed(seed)
        done = 0
        while done < n_samples:
            chunk = min(max_batch, n_samples - done)
            X_all, y_all = generate_samples(T, chunk, "random", max_batch)
            for i in range(0, chunk, batch_size):
                X, y_ord = X_all[i:i + batch_size], y_all[i:i + batch_size]
                logits = model(X)
                y = y_ord.sum(dim=1) - 1
                pred = ordinal_expectation(logits)
                err = (pred - y).double()
                se = err ** 2
                sums["se"] += se.sum().item()
                sums["se2"] += (se ** 2).sum().item()
                sums["w3"] += (err.abs() <= 3).sum().item()
                sums["w4"] += (err.abs() <= 4).sum().item()
                sums["acc"] += (torch.round(pred) == y).sum().item()
                sums["bce"] += F.binary_cross_entropy_with_logits(logits, y_ord, reduction="sum").item()
            done += chunk
    n = float(n_samples)
    mse = sums["se"] / n
    mse_std = math.sqrt(max(sums["se2"] / n - mse ** 2, 0.0)) / math.sqrt(n)

    def p_and_err(k):
        p = sums[k] / n
        return p, math.sqrt(p * (1 - p) / n)

    w3, w3_err = p_and_err("w3")
    w4, w4_err = p_and_err("w4")
    acc, acc_err = p_and_err("acc")
    return {
        "n": int(n), "mse": mse, "mse_stderr": mse_std,
        "within3": w3, "within3_stderr": w3_err, "within4": w4, "within4_stderr": w4_err,
        "rounded_acc": acc, "rounded_acc_stderr": acc_err,
        "bce": sums["bce"] / (n * 21), "seconds": time.time() - t0,
    }


def format_metrics(m):
    return (f"MSE {m['mse']:.4f} ± {m['mse_stderr']:.4f} | W3A {100 * m['within3']:.2f}% | "
            f"W4A {100 * m['within4']:.2f}% | acc {100 * m['rounded_acc']:.2f}% | "
            f"BCE {m['bce']:.6f} | n={m['n']} ({m['seconds']:.1f}s)")


# ============================================================================
# Beam search (ported from cubegpt3_solve.py)
# ============================================================================
def make_model_scorer(T, model, canonical=True, full=True, eval_batch_size=16384):
    """scorer(states (M, 54)) -> (M,) predicted distance; lower = closer."""

    @torch.no_grad()
    def scorer(states):
        out = torch.empty(states.shape[0], device=states.device)
        for i in range(0, states.shape[0], eval_batch_size):
            chunk = states[i:i + eval_batch_size]
            if canonical:
                chunk = canonicalize_full_batch(T, chunk) if full else canonicalize_batch(T, chunk)
            out[i:i + eval_batch_size] = ordinal_expectation(model(states_to_tokens(T, chunk)))
        return out
    return scorer


def _reconstruct(history_moves, history_parents, b, m):
    moves = [int(m)]
    idx = int(b)
    for L in range(len(history_moves) - 1, -1, -1):
        moves.insert(0, int(history_moves[L][idx]))
        idx = int(history_parents[L][idx])
    return moves


@torch.no_grad()
def beam_search_gpu(T, model, start, width=100, max_moves=30, scorer=None, canonical=True,
                    full=True, eval_batch_size=16384):
    """Returns (success, list_of_move_names_or_None). `start` is a cube string
    or a (54,) state tensor."""
    dev = T.device
    if scorer is None:
        model.eval()
        scorer = make_model_scorer(T, model, canonical, full, eval_batch_size)
    frontier = (cube_string_to_state(T, start) if isinstance(start, str) else start.to(dev)).view(1, 54)

    SOLVED = T.SOLVED
    MOVE18 = T.MOVE_GATHER[:18]
    MOVE_FACE, MOVE_AXIS = T.MOVE_FACE, T.MOVE_AXIS
    if torch.equal(frontier[0], SOLVED):
        return True, []

    prev_face = torch.full((1,), -1, dtype=torch.int64, device=dev)
    prev_axis = torch.full((1,), -1, dtype=torch.int64, device=dev)
    prev2_axis = torch.full((1,), -2, dtype=torch.int64, device=dev)
    history_parents, history_moves = [], []

    for _ in range(max_moves):
        B = frontier.shape[0]
        children = frontier[:, MOVE18]                                   # (B, 18, 54)
        is_solved = (children == SOLVED).all(dim=2)
        if is_solved.any():
            flat = int(is_solved.reshape(-1).nonzero(as_tuple=False)[0].item())
            moves = _reconstruct(history_moves, history_parents, flat // 18, flat % 18)
            return True, [T.MOVE_NAMES[m] for m in moves]

        banned = MOVE_FACE.unsqueeze(0) == prev_face.unsqueeze(1)
        banned |= ((MOVE_AXIS.unsqueeze(0) == prev_axis.unsqueeze(1)) & (prev_axis == prev2_axis).unsqueeze(1))
        valid = ~banned

        child_flat = children.reshape(B * 18, 54)
        scores = scorer(child_flat).masked_fill(~valid.reshape(-1), float("inf"))
        k = min(width, int(valid.sum().item()))
        _, top_idx = torch.topk(scores, k, largest=False)
        parent_idx = top_idx // 18
        move_id = top_idx % 18

        frontier = child_flat[top_idx]
        prev2_axis = prev_axis[parent_idx]
        prev_axis = MOVE_AXIS[move_id]
        prev_face = MOVE_FACE[move_id]
        history_parents.append(parent_idx)
        history_moves.append(move_id)
    return False, None


def scramble_for(seed, idx, n_moves):
    """Deterministic scramble #idx for a given seed (independent of sharding)."""
    rng = np.random.default_rng([seed, idx])
    return scrambled(n_moves, rng).cube_string


def verify_solution(start_string, move_names):
    state = CubeState(start_string)
    for name in move_names:
        for a, axis in enumerate(CubeState.moves):
            for s, side in enumerate(axis):
                if name in side:
                    state = state.get_child(a, s, side.index(name))
    return state.cube_string == CubeState().cube_string


def solve_cubes(T, model, idxs, widths, scramble_moves=30, max_moves=30, seed=0,
                eval_batch_size=16384, log=print, done=None, on_result=None):
    """done: {(idx, width): result} already solved (skipped, returned as-is);
    on_result(result) is called after each new cube (used to persist progress)."""
    done = done or {}
    results = []
    for width in widths:
        for idx in idxs:
            if (idx, width) in done:
                results.append(done[(idx, width)])
                continue
            start = scramble_for(seed, idx, scramble_moves)
            t0 = time.time()
            ok, moves = beam_search_gpu(T, model, start, width=width, max_moves=max_moves,
                                        eval_batch_size=eval_batch_size)
            dt = time.time() - t0
            if ok and not verify_solution(start, moves):
                raise AssertionError(f"beam search returned an invalid solution for cube {idx}")
            results.append({"idx": idx, "width": width, "scramble": start, "solved": ok,
                            "n_moves": len(moves) if ok else None, "moves": moves, "seconds": dt})
            log(f"  width {width} cube {idx}: " + (f"solved in {len(moves)} moves" if ok else "NOT solved")
                + f" ({dt:.1f}s)")
            if on_result is not None:
                on_result(results[-1])
    return results


def eval_parts_dir(vol_root, out_name):
    from pathlib import Path
    return Path(vol_root) / "eval_results" / f"{out_name}.parts"


def load_eval_parts(parts):
    """Per-cube results persisted by shards: {(idx, width): result}, plus shard-0 metrics."""
    import json
    done = {}
    for f in sorted(parts.glob("shard*.jsonl")) if parts.exists() else []:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["idx"], r["width"])] = r
    m = parts / "metrics.json"
    return done, (json.loads(m.read_text()) if m.exists() else None)


def run_eval_shard(spec, vol_root, shard=0, n_shards=1, device=None, commit=None):
    """One evaluation worker. Shard 0 also computes the sample metrics; cubes are
    split round-robin across shards. spec keys: checkpoint (volume-relative),
    model (ModelConfig dict), metrics_samples, solve_cubes, widths, scramble_moves,
    max_moves, seed, eval_batch_size, tf32, and optionally out_name: then every solved
    cube is appended to eval_results/<out_name>.parts/ in the volume as it finishes, and
    a retried shard skips cubes already done."""
    import json
    from pathlib import Path

    from .config import ModelConfig, validate_model
    from .cube import CubeTables, check_gpu_pipeline
    from .cube import generate_samples
    from .model import build_model, count_params, load_weights, measure_routing

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if spec.get("tf32", True):
        torch.set_float32_matmul_precision("high")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else device
    log = lambda msg: print(f"[shard {shard}/{n_shards}] {msg}", flush=True)  # noqa: E731

    T = CubeTables(device)
    check_gpu_pipeline(T)
    mcfg = ModelConfig(**spec["model"])
    validate_model(mcfg)
    model = build_model(mcfg).to(device)
    model.load_state_dict(load_weights(Path(vol_root) / spec["checkpoint"], device))
    model.eval()
    log(f"{gpu}: loaded {spec['checkpoint']} ({count_params(model):,} params)")

    with _fork_rng(device):
        torch.manual_seed(spec["seed"])
        routing = measure_routing(model, generate_samples(T, 4096)[0])
    if routing:
        log(f"MoE routing (fresh forward, 4096 cubes): {routing['dead_experts']} dead experts "
            f"{routing['dead_per_layer']}, bias spread {routing['bias_spread']:.2f}")
    out = {"shard": shard, "gpu": gpu, "expert_stats": routing, "metrics": None, "solves": []}
    parts = eval_parts_dir(vol_root, spec["out_name"]) if spec.get("out_name") else None
    done, saved_metrics = load_eval_parts(parts) if parts is not None else ({}, None)
    if parts is not None:
        parts.mkdir(parents=True, exist_ok=True)
        if done:
            log(f"resuming: {len(done)} cubes already in {parts.name}")
    persist = commit or (lambda: None)
    if shard == 0 and spec.get("metrics_samples", 0) > 0:
        out["metrics"] = saved_metrics or compute_metrics(model, T, spec["metrics_samples"])
        log(format_metrics(out["metrics"]))
        if parts is not None and saved_metrics is None:
            (parts / "metrics.json").write_text(json.dumps(out["metrics"]))
            persist()

    def save(r):
        if parts is not None:
            with open(parts / f"shard{shard}.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
            persist()

    idxs = list(range(shard, spec.get("solve_cubes", 0), n_shards))
    if idxs:
        out["solves"] = solve_cubes(T, model, idxs, spec["widths"], spec["scramble_moves"], spec["max_moves"],
                                    spec["seed"], spec["eval_batch_size"], log=log, done=done, on_result=save)
    return out


def summarize_solves(results):
    out = {}
    for width in sorted({r["width"] for r in results}):
        rs = [r for r in results if r["width"] == width]
        lens = [r["n_moves"] for r in rs if r["solved"]]
        mean = float(np.mean(lens)) if lens else float("nan")
        out[width] = {
            "n": len(rs), "solve_rate": len(lens) / len(rs), "mean_moves": mean,
            "mao": mean - MEAN_OPTIMAL_HTM, "median_moves": float(np.median(lens)) if lens else float("nan"),
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
        }
    return out
