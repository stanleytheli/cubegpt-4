"""Rubik's cube representation, symmetries, and batched GPU data generation.

Ported from the CubeGPT-3 Colab notebooks (src/cubegpt3_train.py). Tokenization,
symmetries, scrambling and canonicalization are unchanged: the checkpoints in
models/ depend on them exactly. `check_gpu_pipeline` re-verifies the GPU
implementation against the string reference implementation at every job start.

Lookup tables come from lookup_tables/cube_utils.json (the same file as
github.com/stanleytheli/cubegpt). Override the location with $CUBEGPT_TABLES.
"""
from __future__ import annotations

import ast
import json
import os
import random
from itertools import permutations, product
from pathlib import Path

import numpy as np
import torch

_DEFAULT_TABLES = Path(__file__).resolve().parent.parent / "lookup_tables" / "cube_utils.json"
TABLES_PATH = Path(os.environ.get("CUBEGPT_TABLES", str(_DEFAULT_TABLES)))

with open(TABLES_PATH) as _f:
    _tables = json.load(_f)

solved_string = "WWWWWWWWWOOOOOOOOOGGGGGGGGGRRRRRRRRRBBBBBBBBBYYYYYYYYY"
SOLVED_COLORS = ["W", "O", "G", "R", "B", "Y"]
pos_idx = ast.literal_eval(_tables["pos_idx"])
pos_face_idx = ast.literal_eval(_tables["pos_face_idx"])
color_str_tensor_idx = _tables["color_str_tensor_idx"]
color_str_orientation = _tables["color_str_orientation"]
move_map = _tables["move_map"]

N_CLASSES = 21          # ordinal head size: scramble depths 0..20
SCRAMBLE_LIMIT = 21     # training scrambles have 0..SCRAMBLE_LIMIT-1 moves


# ============================================================================
# String reference implementation
# ============================================================================
def get_colors_str_by_pos(cube_string, position):
    return "".join([cube_string[i] for i in pos_face_idx[position]])


def string_to_tokens(cube_string):
    tokens_tensor = torch.zeros((20, 2), dtype=torch.int32)
    orien_idx_to_token = {0: 0, 3: 1, 4: 2, 1: 0, 2: 1, 5: 2}
    edge_orientations = 2
    corner_orientations = 3

    for pos, position_idx in pos_idx.items():
        color_str = get_colors_str_by_pos(cube_string, pos)
        is_edge = len(color_str) == 2
        piece_idx = color_str_tensor_idx[color_str]
        orientation_idx = color_str_orientation[color_str]

        if is_edge:
            placement_token = position_idx * edge_orientations + orientation_idx
        else:
            orientation_idx = orien_idx_to_token[orientation_idx]
            placement_token = position_idx * corner_orientations + orientation_idx
            placement_token += 24

        tokens_tensor[piece_idx][0] = piece_idx
        tokens_tensor[piece_idx][1] = placement_token
    return tokens_tensor


def rotate_string(cube_str, move=None):
    """Fast rotate function for cube strings."""
    return "".join([cube_str[i] for i in move_map[move]])


class CubeState:
    # moves[move_class][move_type][move_index]
    # when scrambling:
    #   maximum 2 moves with same move_class in a row
    #   maximum 1 move with same move_class, move_type in a row
    moves = [
        [["U", "U'", "U2"], ["D", "D'", "D2"]],
        [["L", "L'", "L2"], ["R", "R'", "R2"]],
        [["F", "F'", "F2"], ["B", "B'", "B2"]],
    ]

    def __init__(self, cube_string: str = solved_string, prev_state=None, prev_move=None,
                 prev_move_class=None, prev_move_type=None):
        self.cube_string = cube_string
        self.prev_state = prev_state
        self.prev_move = prev_move
        self.prev_move_class = prev_move_class
        self.prev_move_type = prev_move_type
        self.prev2_move_class = prev_state.prev_move_class if prev_state else None

    def get_tokens(self):
        return string_to_tokens(self.cube_string)

    def get_allowed_moves(self):
        for move_class in range(len(CubeState.moves)):
            if move_class == self.prev_move_class and move_class == self.prev2_move_class:
                continue
            for move_type in range(len(CubeState.moves[move_class])):
                if move_class == self.prev_move_class and move_type == self.prev_move_type:
                    continue
                for move_index in range(len(CubeState.moves[move_class][move_type])):
                    yield (move_class, move_type, move_index)

    def get_child(self, move_class, move_type, move_index):
        move = CubeState.moves[move_class][move_type][move_index]
        return CubeState(cube_string=rotate_string(self.cube_string, move), prev_state=self,
                         prev_move=move, prev_move_class=move_class, prev_move_type=move_type)

    def get_random_child(self, rng=None):
        allowed_moves = list(self.get_allowed_moves())
        i = rng.integers(len(allowed_moves)) if rng is not None else np.random.randint(len(allowed_moves))
        return self.get_child(*allowed_moves[i])

    def move_names(self):
        names, cur = [], self
        while cur is not None and cur.prev_move is not None:
            names.append(cur.prev_move)
            cur = cur.prev_state
        return names[::-1]


def scrambled(n, rng=None):
    """A CubeState scrambled with n random (constraint-respecting) moves."""
    state = CubeState()
    for _ in range(n):
        state = state.get_random_child(rng)
    return state


# ============================================================================
# Symmetries (48 = 24 rotations x mirror) and inversion
# ============================================================================
INCLUDE_MIRRORS = True
NUM_FACES = 6
NUM_FACELETS = 54


def face_of_facelet(facelet):
    return facelet // 9


CENTER_FACELET = [9 * face + 4 for face in range(NUM_FACES)]
SOLVED_COLOR = [solved_string[CENTER_FACELET[face]] for face in range(NUM_FACES)]


def outward_normals(position):
    normals = set()
    for axis in range(3):
        if position[axis] == 0:
            normal = [0, 0, 0]; normal[axis] = -1
            normals.add(tuple(normal))
        if position[axis] == 2:
            normal = [0, 0, 0]; normal[axis] = +1
            normals.add(tuple(normal))
    return normals


_candidate_normals = [None] * NUM_FACES
for _position, _facelets in pos_face_idx.items():
    for _facelet in _facelets:
        _face = face_of_facelet(_facelet)
        if _candidate_normals[_face] is None:
            _candidate_normals[_face] = outward_normals(_position)
        else:
            _candidate_normals[_face] &= outward_normals(_position)
assert all(len(c) == 1 for c in _candidate_normals)
FACE_NORMAL = [c.pop() for c in _candidate_normals]

FACELET_AT = {}   # (position, normal) -> facelet index
for _position, _facelets in pos_face_idx.items():
    for _facelet in _facelets:
        FACELET_AT[(tuple(_position), FACE_NORMAL[face_of_facelet(_facelet)])] = _facelet
for _face in range(NUM_FACES):
    _normal = FACE_NORMAL[_face]
    FACELET_AT[((1 + _normal[0], 1 + _normal[1], 1 + _normal[2]), _normal)] = CENTER_FACELET[_face]
assert len(FACELET_AT) == NUM_FACELETS

_symmetry_matrices = []
for _perm in permutations(range(3)):
    for _signs in product((+1, -1), repeat=3):
        _m = np.zeros((3, 3), dtype=int)
        for _row in range(3):
            _m[_row, _perm[_row]] = _signs[_row]
        if round(np.linalg.det(_m)) == 1 or INCLUDE_MIRRORS:
            _symmetry_matrices.append(_m)


def transform_position(matrix, position):
    return tuple(matrix @ (np.array(position) - 1) + 1)


def transform_normal(matrix, normal):
    return tuple(matrix @ np.array(normal))


SYMMETRIES = []   # list of (gather, recolor) ; new_string[i] = old_string[gather[i]]
for _m in _symmetry_matrices:
    _gather = [None] * NUM_FACELETS
    for (_position, _normal), _facelet in FACELET_AT.items():
        _gather[FACELET_AT[(transform_position(_m, _position), transform_normal(_m, _normal))]] = _facelet
    _rotated_solved = "".join(solved_string[_gather[i]] for i in range(NUM_FACELETS))
    _recolor = str.maketrans({_rotated_solved[CENTER_FACELET[f]]: SOLVED_COLOR[f] for f in range(NUM_FACES)})
    SYMMETRIES.append((_gather, _recolor))
assert len(SYMMETRIES) == (48 if INCLUDE_MIRRORS else 24)


def apply_symmetry(cube_string, gather, recolor):
    return "".join(cube_string[gather[i]] for i in range(NUM_FACELETS)).translate(recolor)


def canonicalize(cube_string):
    return min(apply_symmetry(cube_string, g, r) for g, r in SYMMETRIES)


def invert_string(cube_string):
    result = [None] * NUM_FACELETS
    for face in range(NUM_FACES):
        result[CENTER_FACELET[face]] = SOLVED_COLOR[face]
    for position, facelets in pos_face_idx.items():
        home_position = np.array([1, 1, 1])
        for facelet in facelets:
            home_position += np.array(FACE_NORMAL[SOLVED_COLOR.index(cube_string[facelet])])
        home_position = tuple(home_position)
        for facelet in facelets:
            color = cube_string[facelet]
            home_facelet = FACELET_AT[(home_position, FACE_NORMAL[SOLVED_COLOR.index(color)])]
            result[home_facelet] = SOLVED_COLOR[face_of_facelet(facelet)]
    return "".join(result)


def canonicalize_full(cube_string):
    return min(canonicalize(cube_string), canonicalize(invert_string(cube_string)))


ALL_MOVES = [move for move in move_map if move != "null"]


def run_string_self_tests(seed=0):
    """The notebook's symmetry self-tests. Raises AssertionError on failure."""
    rng = random.Random(seed)
    for gather, recolor in SYMMETRIES:
        assert apply_symmetry(solved_string, gather, recolor) == solved_string
    state = solved_string
    for _ in range(30):
        state = rotate_string(state, rng.choice(ALL_MOVES))
    for gather, recolor in SYMMETRIES:
        state_sym = apply_symmetry(state, gather, recolor)
        for move in ALL_MOVES:
            target = apply_symmetry(rotate_string(state, move), gather, recolor)
            assert any(rotate_string(state_sym, m2) == target for m2 in ALL_MOVES)
    for gather, recolor in SYMMETRIES:
        assert canonicalize(apply_symmetry(state, gather, recolor)) == canonicalize(state)
    assert invert_string(rotate_string(solved_string, "U")) == rotate_string(solved_string, "U'")
    assert canonicalize_full(invert_string(state)) == canonicalize_full(state)


# ============================================================================
# GPU data generation. A state is a (54,) int64 tensor of color indices 0..5
# (W O G R B Y); a batch is (B, 54). Moves/symmetries are precomputed gathers.
# ============================================================================
class CubeTables:
    """All constant tensors, built once and moved to the device once."""

    def __init__(self, device):
        self.device = device
        COLOR_TO_IDX = {c: i for i, c in enumerate(SOLVED_COLOR)}

        def t(arr, dtype=torch.int64):
            return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)

        self.SOLVED = t([COLOR_TO_IDX[c] for c in solved_string])

        # moves: flat id = (axis*2 + side)*3 + variant, null = 18
        move_names = [CubeState.moves[a][s][v] for a in range(3) for s in range(2) for v in range(3)]
        move_names.append("null")
        self.MOVE_NAMES = move_names
        self.MOVE_GATHER = t([move_map[m] for m in move_names])              # (19, 54)
        self.NULL_MOVE = 18
        ids = torch.arange(18, device=device)
        self.MOVE_AXIS = ids // 6
        self.MOVE_FACE = ids // 3

        edge_pos = sorted([p for p in pos_face_idx if len(pos_face_idx[p]) == 2], key=lambda p: pos_idx[p])
        corner_pos = sorted([p for p in pos_face_idx if len(pos_face_idx[p]) == 3], key=lambda p: pos_idx[p])
        self.EDGE_FACELETS = t([pos_face_idx[p] for p in edge_pos])          # (12, 2)
        self.CORNER_FACELETS = t([pos_face_idx[p] for p in corner_pos])      # (8, 3)
        self.EDGE_POS_VALUE = t([pos_idx[p] for p in edge_pos])
        self.CORNER_POS_VALUE = t([pos_idx[p] for p in corner_pos])

        orien_idx_to_token = {0: 0, 3: 1, 4: 2, 1: 0, 2: 1, 5: 2}
        epiece = np.zeros(36, dtype=np.int64);  eorient = np.zeros(36, dtype=np.int64)
        cpiece = np.zeros(216, dtype=np.int64); corient = np.zeros(216, dtype=np.int64)
        for cs, piece in color_str_tensor_idx.items():
            idxs = [COLOR_TO_IDX[c] for c in cs]
            if len(cs) == 2:
                code = idxs[0] * 6 + idxs[1]
                epiece[code] = piece
                eorient[code] = color_str_orientation[cs]
            else:
                code = idxs[0] * 36 + idxs[1] * 6 + idxs[2]
                cpiece[code] = piece
                corient[code] = orien_idx_to_token[color_str_orientation[cs]]
        self.EDGE_PIECE = t(epiece);   self.EDGE_ORIENT = t(eorient)
        self.CORNER_PIECE = t(cpiece); self.CORNER_ORIENT = t(corient)

        self.SYM_GATHER = t([g for g, _ in SYMMETRIES])                      # (48, 54)
        self.SYM_RECOLOR = t([[COLOR_TO_IDX[c.translate(trans)] for c in SOLVED_COLOR]
                              for _, trans in SYMMETRIES])                   # (48, 6)

        # lexicographic keys: 54 base-6 digits packed into 3 int64 chunks of 18,
        # compared in alphabetical color order to match the string implementation.
        self.POW6 = t(6 ** np.arange(17, -1, -1, dtype=np.int64))
        self.CMP_RANK = t([sorted(SOLVED_COLOR).index(c) for c in SOLVED_COLOR])

        einv = np.zeros((12, 36, 2), dtype=np.int64)
        cinv = np.zeros((8, 216, 3), dtype=np.int64)
        for cs in color_str_tensor_idx:
            idxs = [COLOR_TO_IDX[c] for c in cs]
            home = np.array([1, 1, 1])
            for ci in idxs:
                home += np.array(FACE_NORMAL[ci])
            home = tuple(home)
            targets = [FACELET_AT[(home, FACE_NORMAL[ci])] for ci in idxs]
            if len(cs) == 2:
                einv[:, idxs[0] * 6 + idxs[1], :] = targets
            else:
                cinv[:, idxs[0] * 36 + idxs[1] * 6 + idxs[2], :] = targets
        self.EDGE_INV_TARGET = t(einv)
        self.CORNER_INV_TARGET = t(cinv)
        self.EDGE_SRC_COLOR = t([[face_of_facelet(i) for i in pos_face_idx[p]] for p in edge_pos])
        self.CORNER_SRC_COLOR = t([[face_of_facelet(i) for i in pos_face_idx[p]] for p in corner_pos])


def scramble_batch(T, batch_size, scramble_limit=SCRAMBLE_LIMIT, lengths=None, return_trajectory=False):
    """Random scrambles; constraints match CubeState.get_allowed_moves exactly:
    never the same face twice in a row, never the same axis 3 times in a row,
    uniform over allowed moves. Finished samples are fed the null move."""
    dev = T.device
    if lengths is None:
        lengths = torch.randint(0, scramble_limit, (batch_size,), device=dev)
    max_steps = scramble_limit - 1

    states = T.SOLVED.unsqueeze(0).expand(batch_size, 54).clone()
    prev_axis = torch.full((batch_size,), -1, dtype=torch.int64, device=dev)
    prev2_axis = torch.full((batch_size,), -2, dtype=torch.int64, device=dev)
    prev_face = torch.full((batch_size,), -1, dtype=torch.int64, device=dev)
    trajectory = [states] if return_trajectory else None

    for step in range(max_steps):
        active = step < lengths
        banned = T.MOVE_FACE.unsqueeze(0) == prev_face.unsqueeze(1)
        banned |= ((T.MOVE_AXIS.unsqueeze(0) == prev_axis.unsqueeze(1))
                   & (prev_axis == prev2_axis).unsqueeze(1))
        scores = torch.rand(batch_size, 18, device=dev).masked_fill(banned, -1.0)
        choice = scores.argmax(dim=1)

        applied = torch.where(active, choice, torch.full_like(choice, T.NULL_MOVE))
        states = torch.gather(states, 1, T.MOVE_GATHER[applied])

        prev2_axis = torch.where(active, prev_axis, prev2_axis)
        prev_face = torch.where(active, T.MOVE_FACE[choice], prev_face)
        prev_axis = torch.where(active, T.MOVE_AXIS[choice], prev_axis)
        if return_trajectory:
            trajectory.append(states)

    if return_trajectory:
        return torch.stack(trajectory, dim=1), lengths                       # (B, scramble_limit, 54)
    return states, lengths


def states_to_tokens(T, states):
    """(B, 54) -> (B, 20, 2), identical to string_to_tokens."""
    B = states.shape[0]
    ec = states[:, T.EDGE_FACELETS]
    ecode = ec[..., 0] * 6 + ec[..., 1]
    epiece = T.EDGE_PIECE[ecode]
    eplace = T.EDGE_POS_VALUE.unsqueeze(0) * 2 + T.EDGE_ORIENT[ecode]

    cc = states[:, T.CORNER_FACELETS]
    ccode = cc[..., 0] * 36 + cc[..., 1] * 6 + cc[..., 2]
    cpiece = T.CORNER_PIECE[ccode]
    cplace = T.CORNER_POS_VALUE.unsqueeze(0) * 3 + T.CORNER_ORIENT[ccode] + 24

    placements = torch.empty(B, 20, dtype=torch.int64, device=T.device)
    placements.scatter_(1, epiece, eplace)
    placements.scatter_(1, cpiece, cplace)
    piece_ids = torch.arange(20, device=T.device).unsqueeze(0).expand(B, 20)
    return torch.stack([piece_ids, placements], dim=2)


def _lex_argmin(T, variants):
    """variants: (B, K, 54) -> index (B,) of the lexicographically smallest row."""
    digits = T.CMP_RANK[variants].view(variants.shape[0], variants.shape[1], 3, 18)
    keys = (digits * T.POW6).sum(dim=3)
    BIG = torch.tensor(2 ** 62, dtype=torch.int64, device=T.device)
    alive = torch.ones(variants.shape[:2], dtype=torch.bool, device=T.device)
    for j in range(3):
        kj = torch.where(alive, keys[..., j], BIG)
        alive &= kj.eq(kj.min(dim=1, keepdim=True).values)
    return alive.to(torch.int64).argmax(dim=1)


def apply_all_symmetries(T, states):
    variants = states[:, T.SYM_GATHER]
    recolor = T.SYM_RECOLOR.unsqueeze(0).expand(states.shape[0], -1, -1)
    return torch.gather(recolor, 2, variants)


def canonicalize_batch(T, states):
    variants = apply_all_symmetries(T, states)
    best = _lex_argmin(T, variants)
    return variants[torch.arange(states.shape[0], device=T.device), best]


def invert_batch(T, states):
    B = states.shape[0]
    result = T.SOLVED.unsqueeze(0).expand(B, 54).clone()
    ec = states[:, T.EDGE_FACELETS]
    ecode = ec[..., 0] * 6 + ec[..., 1]
    etgt = T.EDGE_INV_TARGET[torch.arange(12, device=T.device), ecode]
    result.scatter_(1, etgt.reshape(B, 24), T.EDGE_SRC_COLOR.reshape(1, 24).expand(B, 24))
    cc = states[:, T.CORNER_FACELETS]
    ccode = cc[..., 0] * 36 + cc[..., 1] * 6 + cc[..., 2]
    ctgt = T.CORNER_INV_TARGET[torch.arange(8, device=T.device), ccode]
    result.scatter_(1, ctgt.reshape(B, 24), T.CORNER_SRC_COLOR.reshape(1, 24).expand(B, 24))
    return result


def canonicalize_full_batch(T, states):
    """min over 48 symmetries x {state, inverse} = 96 variants."""
    v = torch.cat([apply_all_symmetries(T, states),
                   apply_all_symmetries(T, invert_batch(T, states))], dim=1)
    best = _lex_argmin(T, v)
    return v[torch.arange(states.shape[0], device=T.device), best]


def to_ordinal_targets(y, num_classes=N_CLASSES):
    """(B,) depths -> (B, num_classes) with 1s where level <= depth."""
    levels = torch.arange(num_classes, device=y.device)
    return (y.view(-1, 1) >= levels).float()


def depth_weights(T, power, scramble_limit=SCRAMBLE_LIMIT):
    """Sampling weights over scramble depths 0..scramble_limit-1: proportional to
    max(depth, 1) ** power (power 1: depth 18 is drawn 18x as often as depth 1; the solved
    state keeps depth 1's share). None for power 0, which means uniform."""
    if power == 0:
        return None
    d = torch.arange(scramble_limit, device=T.device, dtype=torch.float64).clamp_min(1.0)
    return (d ** power / (d ** power).sum()).float()


def generate_batch_ord(T, n, scramble_limit=SCRAMBLE_LIMIT, canonical=True, full=True, depth_power=0.0):
    weights = depth_weights(T, depth_power, scramble_limit)
    lengths = None if weights is None else torch.multinomial(weights, n, replacement=True)
    states, steps = scramble_batch(T, n, scramble_limit, lengths=lengths)
    if canonical:
        states = canonicalize_full_batch(T, states) if full else canonicalize_batch(T, states)
    return states_to_tokens(T, states), to_ordinal_targets(steps)


def generate_path_batch(T, k, k_max=SCRAMBLE_LIMIT, canonical=True, full=True):
    """k random walks of k_max-1 moves; every intermediate state labeled with its depth."""
    lengths = torch.full((k,), k_max - 1, dtype=torch.int64, device=T.device)
    traj, _ = scramble_batch(T, k, k_max, lengths=lengths, return_trajectory=True)
    states = traj.reshape(k * k_max, 54)
    steps = torch.arange(k_max, device=T.device).repeat(k)
    if canonical:
        states = canonicalize_full_batch(T, states) if full else canonicalize_batch(T, states)
    return states_to_tokens(T, states), to_ordinal_targets(steps)


def generate_samples(T, n, kind="random", max_batch=65536, depth_power=0.0, canonical=True):
    """n samples generated in sub-batches of at most max_batch states, so peak
    memory stays bounded (canonicalization materializes 96 variants per state).
    kind="path": n must be a multiple of SCRAMBLE_LIMIT; samples come in whole paths.
    depth_power (random only): depth d drawn with weight max(d, 1) ** depth_power (0 = uniform)."""
    if kind == "random":
        sizes = [min(max_batch, n - i) for i in range(0, n, max_batch)]
        parts = [generate_batch_ord(T, s, depth_power=depth_power, canonical=canonical) for s in sizes]
    elif kind == "path":
        if n % SCRAMBLE_LIMIT:
            raise ValueError(f"path data needs a multiple of {SCRAMBLE_LIMIT} samples, got {n}")
        k_total = n // SCRAMBLE_LIMIT
        k_max_batch = max(1, max_batch // SCRAMBLE_LIMIT)
        parts = [generate_path_batch(T, min(k_max_batch, k_total - i), canonical=canonical) for i in range(0, k_total, k_max_batch)]
    else:
        raise ValueError(f"unknown data kind {kind!r}")
    if len(parts) == 1:
        return parts[0]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


# ============================================================================
# Conversions + GPU-vs-reference equivalence check (run at every job start)
# ============================================================================
_COLOR_TO_IDX = {c: i for i, c in enumerate(SOLVED_COLOR)}


def cube_string_to_state(T, s):
    return torch.tensor([_COLOR_TO_IDX[c] for c in s], dtype=torch.int64, device=T.device)


def state_to_cube_string(state_vec):
    return "".join(SOLVED_COLOR[i] for i in state_vec.tolist())


def check_gpu_pipeline(T, n=128, seed=0):
    """Verify the batched implementation against the string reference on random
    states: moves, tokenization, inversion, canonicalization, scramble validity.
    Raises AssertionError on any mismatch."""
    rng = random.Random(seed)
    strings = []
    for _ in range(n):
        s = solved_string
        for _ in range(rng.randrange(0, 25)):
            s = rotate_string(s, rng.choice(ALL_MOVES))
        strings.append(s)
    states = torch.stack([cube_string_to_state(T, s) for s in strings])

    tok = states_to_tokens(T, states).cpu()
    ref = torch.stack([string_to_tokens(s).long() for s in strings])
    assert torch.equal(tok, ref), "states_to_tokens != string_to_tokens"

    for mid, name in enumerate(T.MOVE_NAMES[:18]):
        moved = torch.gather(states, 1, T.MOVE_GATHER[mid].expand(n, 54))
        assert [state_to_cube_string(m) for m in moved] == [rotate_string(s, name) for s in strings], \
            f"move gather != rotate_string for {name}"

    inv = invert_batch(T, states)
    assert [state_to_cube_string(m) for m in inv] == [invert_string(s) for s in strings], "invert mismatch"
    can = canonicalize_batch(T, states)
    assert [state_to_cube_string(m) for m in can] == [canonicalize(s) for s in strings], "canonicalize mismatch"
    canf = canonicalize_full_batch(T, states)
    assert [state_to_cube_string(m) for m in canf] == [canonicalize_full(s) for s in strings], \
        "canonicalize_full mismatch"

    st, lengths = scramble_batch(T, 4096)
    assert int(lengths.min()) >= 0 and int(lengths.max()) <= SCRAMBLE_LIMIT - 1
    assert bool((st[lengths == 0] == T.SOLVED).all()), "zero-length scramble is not solved"
    pl = states_to_tokens(T, st)[..., 1]
    edge_slots = (pl[:, :12] // 2).sort(dim=1).values
    corner_slots = ((pl[:, 12:] - 24) // 3).sort(dim=1).values
    assert bool((edge_slots == torch.arange(12, device=T.device)).all()), "invalid edge placement"
    assert bool((corner_slots == torch.arange(8, device=T.device)).all()), "invalid corner placement"
