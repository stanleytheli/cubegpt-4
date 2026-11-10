#!/usr/bin/env python
"""Mirror training runs and eval results from the Modal volume into this repo
(runs/ and eval_results/; runs/ is gitignored). Safe to re-run: files whose
size and modification time already match are skipped, and a run that is still training
just gets its newer logs on the next sync. Never deletes local files.

By default only the small files come down (logs, metrics, configs, status, eval results);
weights (76 MB each for the 8A1) stay on the volume, which persists until deleted, unless
asked for:

    python sync_runs.py                         # logs/metrics for every run + eval_results/
    python sync_runs.py 8a1-wsd wsd25m          # only runs whose name starts with these
    python sync_runs.py 8a1-wsd --weights       # + final weights (final.pth)
    python sync_runs.py 8a1-wsd --snapshots     # + kept snapshots (ckpt/model_<samples>.pt)
    python sync_runs.py 8a1-wsd --with-latest   # + ckpt/latest.pt (full resume state, ~230 MB)
    python sync_runs.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import modal

VOLUME_NAME = "cubegpt-3"
ROOT = Path(__file__).resolve().parent


def listdir(vol, path):
    """One recursive listing (per-directory listings hit Modal's rate limit)."""
    for attempt in range(6):
        try:
            return vol.listdir(path, recursive=True)
        except modal.exception.ResourceExhaustedError:
            time.sleep(5 * 2 ** attempt)
    return vol.listdir(path, recursive=True)


def wanted(path: str, args) -> bool:
    name = path.rsplit("/", 1)[-1]
    if path.endswith("ckpt/latest.pt"):
        return args.with_latest
    if "/ckpt/model_" in path:
        return args.snapshots
    if name.endswith((".pt", ".pth")):
        return args.weights
    if ".parts/" in path:    # per-cube eval progress; the finished .json next to it is the result
        return False
    return not path.endswith(".tmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*", help="only runs whose name starts with one of these")
    ap.add_argument("--weights", action="store_true", help="also download final weights (*.pth)")
    ap.add_argument("--snapshots", action="store_true", help="also download ckpt/model_<samples>.pt snapshots")
    ap.add_argument("--with-latest", action="store_true", help="also download ckpt/latest.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vol = modal.Volume.from_name(VOLUME_NAME)
    entries = [e for e in listdir(vol, "runs") if e.type == modal.volume.FileEntryType.FILE]
    if args.prefixes:
        entries = [e for e in entries if any(e.path.split("/")[1].startswith(p) for p in args.prefixes)]
    else:
        entries += [e for e in listdir(vol, "eval_results") if e.type == modal.volume.FileEntryType.FILE]

    n_new = n_same = n_skipped = 0
    total = 0
    got = []
    for e in sorted(entries, key=lambda e: e.path):
        if not wanted(e.path, args):
            n_skipped += 1
            continue
        dest = ROOT / e.path
        if dest.exists() and dest.stat().st_size == e.size and int(dest.stat().st_mtime) == int(e.mtime):
            n_same += 1
            continue
        got.append(e)
    for e in got:
        dest = ROOT / e.path
        print(f"  {'would get' if args.dry_run else 'get'} {e.path} ({e.size / 1e6:.1f} MB)")
        total += e.size
        n_new += 1
        if args.dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        os.utime(tmp, (e.mtime, e.mtime))        # lets the next sync skip it
        tmp.replace(dest)
    verb = "would download" if args.dry_run else "downloaded"
    print(f"{verb} {n_new} files ({total / 1e6:.0f} MB); {n_same} already up to date; "
          f"{n_skipped} left on the volume (weights not requested, eval progress parts)")


if __name__ == "__main__":
    sys.exit(main())
