"""Follow the solutions found at a wide beam through a recorded narrower search.
    .venv/bin/python data_analysis/track_wide_solutions.py <narrow eval made with --save-beam>.json <wide eval>.json
For every cube in both files, replays the narrow search's stored beams and looks up the state
reached after each move of the wide solution: its rank in the narrow beam, or that it is gone.
Writes eval_results/<narrow name>.tracked.json."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubegpt.cube import ALL_MOVES, move_map, rotate_string  # noqa: E402

narrow_path, wide_path = Path(sys.argv[1]), Path(sys.argv[2])
narrow = {s["idx"]: s for s in json.loads(narrow_path.read_text())["solves"]}
wide = {s["idx"]: s for s in json.loads(wide_path.read_text())["solves"]}
beams = narrow_path.with_suffix(".beams")


def as_array(cube_string):
    return np.frombuffer(cube_string.encode(), dtype=np.uint8)


out = []
for idx in sorted(set(narrow) & set(wide)):
    n, w = narrow[idx], wide[idx]
    z = np.load(beams / f"w{n['width']}_cube{idx:04d}.npz")
    names = ALL_MOVES                      # beam move ids index the 18 face turns in this order
    gather = np.array([move_map[m] for m in names])
    offsets = np.concatenate([[0], np.cumsum(z["sizes"])])
    states = as_array(n["scramble"])[None, :]
    target = n["scramble"]
    ranks = []                             # rank of the wide solution's state after each move; None = not in beam
    for step in range(len(z["sizes"])):
        sl = slice(offsets[step], offsets[step + 1])
        states = np.take_along_axis(states[z["parents"][sl]], gather[z["moves"][sl]], axis=1)
        if step == 0 or step == len(z["sizes"]) - 1:      # check the replay against the narrow solution itself
            mine = n["scramble"]
            for m in n["moves"][:step + 1]:
                mine = rotate_string(mine, m)
            assert (states[z["path"][step]] == as_array(mine)).all(), "beam replay does not match"
        if step < len(w["moves"]) - 1:
            target = rotate_string(target, w["moves"][step])
            hits = np.flatnonzero((states == as_array(target)).all(axis=1))
            ranks.append(int(hits[0]) + 1 if len(hits) else None)
    out.append({"idx": idx, "narrow_moves": n["n_moves"], "wide_moves": w["n_moves"],
                "same_solution": n["moves"] == w["moves"], "narrow_ranks": n["ranks"], "wide_ranks": ranks,
                "beam_sizes": n["beam_sizes"]})

dest = narrow_path.with_suffix(".tracked.json")
dest.write_text(json.dumps({"narrow": narrow_path.name, "wide": wide_path.name, "narrow_width": n["width"],
                            "wide_width": w["width"], "cubes": out}))
print(f"wrote {dest} ({len(out)} cubes)")
