#!/usr/bin/env python
"""Status of CubeGPT training runs, read straight from the Modal volume
(no container, no GPU, works while runs are in flight).

    python status.py                   # table of all runs
    python status.py <run>             # details: progress, losses, ETA, attempts, errors, recent log
    python status.py <run> --watch 60  # refresh every 60 s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import modal

VOLUME_NAME = "cubegpt-3"
ACTIVE = ("starting", "running", "restarting")


def read(vol, path):
    try:
        return b"".join(vol.read_file(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def read_json(vol, path):
    data = read(vol, path)
    return None if data is None else json.loads(data)


def ago(iso):
    try:
        s = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None, "?"
    if s < 120:
        return s, f"{s:.0f}s ago"
    if s < 7200:
        return s, f"{s / 60:.0f}m ago"
    return s, f"{s / 3600:.1f}h ago"


def dur(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def health(st):
    """State plus a staleness warning when an active run's heartbeat stops."""
    state = st.get("state", "?")
    age, _ = ago(st.get("updated_at"))
    limit = 3 * 60 * st.get("status_every_minutes", 2) + 300
    if state in ACTIVE and age is not None and age > limit:
        return f"{state} (STALE: no heartbeat for {age / 60:.0f} min: restarting after preemption, or killed/hung)"
    return state


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def list_runs(vol):
    try:
        entries = vol.listdir("runs")
    except (FileNotFoundError, modal.exception.NotFoundError):
        print(f"no runs in volume {VOLUME_NAME!r} yet")
        return
    rows = []
    for e in sorted(entries, key=lambda e: e.path):
        name = e.path.rstrip("/").split("/")[-1]
        st = read_json(vol, f"runs/{name}/status.json") or {}
        _, updated = ago(st.get("updated_at"))
        rows.append((name, health(st), f"{100 * st.get('progress', 0):.1f}%", str(st.get("global_step", "-")),
                     fmt(st.get("last_eval_loss"), ".5f"), fmt(st.get("samples_per_sec"), ",.0f"),
                     dur(st.get("eta_seconds")) if st.get("state") in ACTIVE else "", updated))
    header = ("run", "state", "progress", "step", "eval loss", "samples/s", "eta", "updated")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def show_run(vol, name, n_log=15):
    st = read_json(vol, f"runs/{name}/status.json")
    if st is None:
        sys.exit(f"run {name!r} not found in volume {VOLUME_NAME!r}")
    _, updated = ago(st.get("updated_at"))
    print(f"run        {name}")
    print(f"state      {health(st)}   (updated {updated})")
    print(f"gpu        {st.get('gpu_name', '?')}")
    print(f"progress   {100 * st.get('progress', 0):.2f}%  step {st.get('global_step')}  stage {st.get('stage')}/"
          f"{st.get('n_stages')} ({st.get('stage_step')}/{st.get('stage_steps')})  "
          f"samples {st.get('samples_seen', 0) / 1e6:.2f}M / {st.get('total_samples', 0) / 1e6:.2f}M")
    print(f"loss       eval {fmt(st.get('last_eval_loss'), '.6f')}  best {fmt(st.get('best_eval_loss'), '.6f')}  "
          f"train {fmt(st.get('last_train_loss'), '.6f')}  lr {fmt(st.get('lr'), '.3g')}")
    print(f"speed      {fmt(st.get('samples_per_sec'), ',.0f')} samples/s   train time {dur(st.get('train_seconds'))}"
          f"   eta {dur(st.get('eta_seconds')) if st.get('state') in ACTIVE else '-'}")
    if st.get("moe"):
        print(f"moe        {st['moe']}")
    print(f"checkpoint step {st.get('last_checkpoint_step')} at {st.get('last_checkpoint_at')}")
    attempts = st.get("attempts", [])
    print(f"attempts   {len(attempts)} total; last: " +
          ", ".join(f"[{a['started_at']} from step {a['start_step']}]" for a in attempts[-3:]))
    if st.get("final_metrics"):
        m = st["final_metrics"]
        print(f"metrics    (stage {m.get('stage')}) MSE {m['mse']:.4f}  W3A {100 * m['within3']:.2f}%  "
              f"W4A {100 * m['within4']:.2f}%  acc {100 * m['rounded_acc']:.2f}%")
    err = st.get("error") or (st.get("last_error") if st.get("state") != "completed" else None)
    if err:
        print(f"\n{'ERROR' if st.get('state') == 'failed' else 'last error'}: {err.get('type')}: {err.get('message')}")
        if st.get("state") == "failed" and err.get("traceback"):
            print(err["traceback"])
    log = read(vol, f"runs/{name}/log.txt")
    if log:
        print(f"\n--- last {n_log} log lines ---")
        print("\n".join(log.decode(errors="replace").splitlines()[-n_log:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run name (omit to list all runs)")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    ap.add_argument("--lines", type=int, default=15, help="log lines to show")
    args = ap.parse_args()
    vol = modal.Volume.from_name(VOLUME_NAME)
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
            print(datetime.now().strftime("%H:%M:%S"))
        show_run(vol, args.run, args.lines) if args.run else list_runs(vol)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
