"""Where a pruned wide-beam solution actually ranked at the step the narrow search dropped it.
    .venv/bin/python data_analysis/pruned_ranks.py eval_results/<name>.tracked.json     (from track_wide_solutions.py)
The recorded beams hold only the kept entries, so for each pruned solution this rebuilds every
candidate of that step (all rule-abiding children of the previous beam), scores them and the
solution's state with the model locally, and counts how many candidates score better.
Adds 'pruned_step', 'pruned_rank', 'pruned_candidates' to each pruned cube in the tracked file."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cubegpt.config import load_model_config  # noqa: E402
from cubegpt.cube import (ALL_MOVES, CubeTables, canonicalize_full_batch, move_map, rotate_string,  # noqa: E402
                          states_to_tokens)
from cubegpt.evaluate import ordinal_expectation  # noqa: E402
from cubegpt.model import build_model, load_weights  # noqa: E402

tracked_path = Path(sys.argv[1])
tracked = json.loads(tracked_path.read_text())
narrow_path = tracked_path.parent / tracked["narrow"]
_narrow_json = json.loads(narrow_path.read_text())
narrow = {s["idx"]: s for s in _narrow_json["solves"]}
DEDUPE = bool(_narrow_json["spec"].get("dedupe"))       # a de-duped search ranks distinct states only
wide = {s["idx"]: s for s in json.loads((tracked_path.parent / tracked["wide"]).read_text())["solves"]}
beams = narrow_path.with_suffix(".beams")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
T = CubeTables(dev)
model = build_model(load_model_config(str(ROOT / "configs/a8f4a1f1_branch230m_to_1b.toml")))
model.load_state_dict(load_weights(str(ROOT / "models/cubegpt-4-t1b.pth"), "cpu"))
model.eval().to(dev)
GATHER = np.array([move_map[m] for m in ALL_MOVES])
FACE = np.array([ALL_MOVES.index(m) // 3 for m in ALL_MOVES])          # U D L R F B
AXIS = FACE // 2
COLOR = np.zeros(256, dtype=np.int64)
from cubegpt.cube import SOLVED_COLOR  # noqa: E402
for i, c in enumerate(SOLVED_COLOR):
    COLOR[ord(c)] = i


@torch.no_grad()
def score(byte_states):
    out = []
    for i in range(0, len(byte_states), 512):
        st = torch.from_numpy(COLOR[byte_states[i:i + 512]]).to(dev)
        out.append(ordinal_expectation(model(states_to_tokens(T, canonicalize_full_batch(T, st)))).float().cpu())
    return torch.cat(out).numpy()


for c in tracked["cubes"]:
    gone = next((k for k, r in enumerate(c["wide_ranks"]) if r is None), None)
    if gone is None:
        continue
    idx = c["idx"]
    z = np.load(beams / f"w{tracked['narrow_width']}_cube{idx:04d}.npz")
    off = np.concatenate([[0], np.cumsum(z["sizes"])])
    states = np.frombuffer(narrow[idx]["scramble"].encode(), dtype=np.uint8)[None, :]
    last = np.full(1, -1); prev = np.full(1, -1)                         # last move id, the one before it
    for step in range(gone):                                             # rebuild the beam before the pruning step
        sl = slice(off[step], off[step + 1])
        par, mov = z["parents"][sl], z["moves"][sl].astype(np.int64)
        states = np.take_along_axis(states[par], GATHER[mov], axis=1)
        prev, last = last[par], mov
    children = states[:, GATHER]                                         # (B, 18, 54)
    banned = FACE[None, :] == np.where(last >= 0, FACE[last], -1)[:, None]
    same_axis_twice = (last >= 0) & (prev >= 0) & (AXIS[last] == AXIS[prev])
    banned |= (AXIS[None, :] == AXIS[last][:, None]) & same_axis_twice[:, None]
    cand = children[~banned]
    if DEDUPE:
        cand = np.unique(cand, axis=0)
    target = narrow[idx]["scramble"]
    for m in wide[idx]["moves"][:gone + 1]:
        target = rotate_string(target, m)
    target = np.frombuffer(target.encode(), dtype=np.uint8)
    scores = score(np.concatenate([cand, target[None, :]]))
    c["pruned_step"] = gone + 1
    c["pruned_rank"] = int((scores[:-1] < scores[-1]).sum()) + 1
    c["pruned_candidates"] = int(len(cand))
    c["pruned_was_candidate"] = bool((cand == target).all(axis=1).any())
    kept_cut = float(z["scores"][off[gone]:off[gone + 1]][-1])
    print(f"cube {idx}: pruned at move {gone + 1}, rank {c['pruned_rank']:,} of {len(cand):,} "
          f"(score {scores[-1]:.3f}, beam cutoff {kept_cut:.3f}, was a candidate: {c['pruned_was_candidate']})", flush=True)
tracked_path.write_text(json.dumps(tracked))
print("updated", tracked_path)
