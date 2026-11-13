"""Optimal (half-turn metric) solutions for the eval scrambles, with Voltara's vcube solver on Modal.

    modal run optimal_solve.py --cubes 200       # CPU only; writes eval_results/optimal_seed0_raw.json
    python optimal_solve.py collect               # verify + write eval_results/optimal_seed0.json

The solver is third-party code: it runs in its own Modal container with no volume, secrets or
network needs beyond the clone, pinned to one commit, and every solution it returns is checked
with our own cube code in `collect`.

vcube (github.com/Voltara/vcube, GPLv3) needs an x86 CPU with AVX2/BMI2 and ~22 GiB for its
default pruning table, which it generates in memory at start-up. It is built inside the container
at run time so it is compiled for the CPU it runs on. The eval scrambles are regenerated as move
sequences (evaluate.scramble_for uses the same seeded generator) because vcube reads moves.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

RAW = Path(__file__).resolve().parent / "eval_results" / "optimal_seed0_raw.json"
VCUBE_COMMIT = "9f5bc2cce18f29437879ace825f22917f6705378"
app = modal.App("cubegpt-optimal")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "clang", "cmake", "make", "cpputest", "libcpputest-dev", "pkg-config", "procps")
         .run_commands("git clone https://github.com/Voltara/vcube /opt/vcube",
                       f"cd /opt/vcube && git checkout {VCUBE_COMMIT}"))


@app.function(image=image, cpu=32.0, memory=40 * 1024, timeout=4 * 3600, retries=0)
def solve_remote(scrambles: list, coord: int = 308) -> dict:
    t0 = time.time()
    env = {"CC": "clang", "CXX": "clang++", "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    print(subprocess.run("grep -m1 'model name' /proc/cpuinfo; nproc; free -g | head -2", shell=True,
                         capture_output=True, text=True).stdout, flush=True)
    for cmd in ("cmake .", "make -j vc-optimal"):
        r = subprocess.run(cmd, shell=True, cwd="/opt/vcube", env=env, capture_output=True, text=True)
        print(f"$ {cmd} -> {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"build failed: {cmd}")
    binary = subprocess.run("find /opt/vcube -name vc-optimal -type f", shell=True, capture_output=True,
                            text=True).stdout.split()[0]
    print(f"built in {time.time() - t0:.0f}s; solving {len(scrambles)} cubes with table {coord}", flush=True)
    proc = subprocess.Popen([binary, "-c", str(coord), "--ordered"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd="/opt/vcube")
    proc.stdin.write("\n".join(scrambles) + "\n")
    proc.stdin.close()
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        print(line.rstrip("\n"), flush=True)
    code = proc.wait()
    result = {"scrambles": scrambles, "coord": coord, "returncode": code, "lines": lines,
              "seconds": time.time() - t0}
    print(f"done in {result['seconds']:.0f}s, exit code {code}", flush=True)
    return result


def scramble_moves(idx, seed=0, n_moves=30):
    import numpy as np
    from cubegpt.cube import scrambled
    state = scrambled(n_moves, np.random.default_rng([seed, idx]))
    return state.move_names(), state.cube_string


@app.local_entrypoint()
def main(cubes: int = 200, coord: int = 308):
    scrambles = [" ".join(scramble_moves(i)[0]) for i in range(cubes)]
    result = solve_remote.remote(scrambles, coord)
    RAW.write_text(json.dumps(result))
    print(f"exit code {result['returncode']}, {result['seconds']:.0f}s; wrote {RAW}")


def collect():
    """Download the raw output, check every solution against our own cube code, write the result."""
    from cubegpt.cube import rotate_string, solved_string
    root = Path(__file__).resolve().parent
    raw = json.loads(RAW.read_text())
    solves = {}
    for line in raw["lines"]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2].isdigit():
            solves[int(parts[0])] = (float(parts[1]), int(parts[2]), parts[3:])
    out = []
    for idx in range(len(raw["scrambles"])):
        moves, cube_string = scramble_moves(idx)
        assert " ".join(moves) == raw["scrambles"][idx]
        seconds, n, solution = solves[idx]
        state = cube_string
        for move in solution:
            state = rotate_string(state, move)
        assert state == solved_string and n == len(solution), f"cube {idx}: vcube solution does not solve it"
        out.append({"idx": idx, "scramble": cube_string, "scramble_moves": moves, "optimal": n,
                    "optimal_solution": solution, "seconds": seconds})
    lengths = [o["optimal"] for o in out]
    dest = root / "eval_results" / "optimal_seed0.json"
    dest.write_text(json.dumps({"solver": "vcube (github.com/Voltara/vcube)", "metric": "HTM", "coord": raw["coord"],
                                "mean_optimal": sum(lengths) / len(lengths), "cubes": out}, indent=1))
    print(f"wrote {dest}: {len(out)} cubes verified, mean optimal {sum(lengths) / len(lengths):.4f}, "
          f"distribution { {k: lengths.count(k) for k in sorted(set(lengths))} }")


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    else:
        print(__doc__)
