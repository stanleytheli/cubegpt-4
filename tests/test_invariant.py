"""Acceptance tests T1-T8 for the rotation-invariant encoding (cubegpt/invariant.py)."""
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubegpt import config as C                                    # noqa: E402
from cubegpt import invariant as I                                 # noqa: E402
from cubegpt.config import ModelConfig                             # noqa: E402
from cubegpt.cube import CubeTables, generate_samples, scramble_batch, states_to_tokens   # noqa: E402
from cubegpt.model import build_model, count_params                # noqa: E402

T = CubeTables("cpu")
DESC = I.descriptor_table()
TINY = dict(arch="pair", d_model=32, n_heads=4, d_ffwd=48, n_layers=2, n_experts=4, n_active=1, d_pair=16)


def poses_of(states):
    return I.tokens_to_poses(states_to_tokens(T, states).numpy(), DESC)


def pose_tokens(g):
    """Pose arrays (N, 20) in reference cubelet order -> repo tokens (N, 20, 2)."""
    inv = {int(d): (p, pl) for p in range(20) for pl, d in enumerate(DESC[p]) if d >= 0}
    out = np.zeros((len(g), 20, 2), dtype=np.int64)
    for n, row in enumerate(g):
        for c, gg in enumerate(row):
            p, pl = inv[c * 24 + int(gg)]
            out[n, p] = (p, pl)
    return torch.as_tensor(out)


def bfs(depth):
    shells, seen = [np.array([I.SOLVED])], {tuple(I.SOLVED)}
    for _ in range(depth):
        nxt = []
        for a, s, k in I.MOVES18:
            for row in I.turn(shells[-1], a, s, k):
                t = tuple(int(v) for v in row)
                if t not in seen:
                    seen.add(t); nxt.append(t)
        shells.append(np.array(nxt))
    return shells


def random_pose(rng, n=40):
    g = I.SOLVED
    for _ in range(n):
        g = I.turn(g, *I.MOVES18[rng.integers(18)])
    return g


def test_t1_tables():
    assert (I.N_CONTENT, I.N_PAIR) == (24, 9628)
    assert sorted(Counter(Counter(I.CONTENT.tolist()).values()).items()) == [(8, 3), (12, 4), (24, 17)]


def test_t2_solved():
    c, p = I.features(I.SOLVED)
    assert len(set(c.tolist())) == 2 and len(set(p.flatten().tolist())) == 19


def test_t3_simulator_agreement():
    assert [len(s) for s in bfs(4)] == [1, 18, 243, 3240, 43239]
    # each of the repo's 18 moves is exactly one reference move
    solved = T.SOLVED.unsqueeze(0)
    assert (poses_of(solved)[0] == I.SOLVED).all()
    ref_of = []
    for mid in range(18):
        got = poses_of(torch.gather(solved, 1, T.MOVE_GATHER[mid].unsqueeze(0)))[0]
        hits = [mv for mv in I.MOVES18 if (I.turn(I.SOLVED, *mv) == got).all()]
        assert len(hits) == 1, (T.MOVE_NAMES[mid], hits)
        ref_of.append(hits[0])
    assert len(set(ref_of)) == 18
    # the same random scrambles in both simulators give identical pose arrays at every step
    gen = torch.Generator().manual_seed(0)
    states = solved.expand(500, 54).clone()
    g = np.tile(I.SOLVED, (500, 1))
    for _ in range(40):
        mv = torch.randint(0, 18, (500,), generator=gen)
        states = torch.gather(states, 1, T.MOVE_GATHER[mv])
        g = np.stack([I.turn(g[i], *ref_of[int(m)]) for i, m in enumerate(mv)])
        assert (poses_of(states) == g).all()
    # and the batched generator's output is always mappable
    poses_of(scramble_batch(T, 2000)[0])


def test_t4_feature_invariance_and_t6_negative_control():
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = random_pose(rng)
        c0, p0 = I.features(g)
        for m in range(24):
            c1, p1 = I.features(I.rotate(g, m))
            assert (c1[I.PERM[m]] == c0).all() and (p1[np.ix_(I.PERM[m], I.PERM[m])] == p0).all()
        changed = 0
        for m in range(24):
            if m == I.IDENT:
                continue
            c1, p1 = I.features(I.MUL[m, g])                 # left multiplication is NOT a symmetry
            changed += sorted(c1.tolist()) != sorted(c0.tolist()) or sorted(p1.flatten().tolist()) != sorted(p0.flatten().tolist())
        assert changed == 23, changed


def test_t5_model_invariance():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    for cfg in (TINY, {**TINY, "d_model": 64, "n_layers": 3}):
        model = build_model(ModelConfig(**cfg)).double().eval()
        gs = [random_pose(rng) for _ in range(4)]
        with torch.no_grad():
            base = model(pose_tokens(np.array(gs)))
            for m in range(24):
                out = model(pose_tokens(np.array([I.rotate(g, m) for g in gs])))
                assert torch.allclose(out, base, atol=1e-9), (m, float((out - base).abs().max()))
            # negative controls: left multiplication and the mirror change the output
            left = model(pose_tokens(np.array([I.MUL[5, g] for g in gs])))
            mir = model(pose_tokens(np.array([I.mirror(g) for g in gs])))
            assert (left - base).abs().max() > 1e-4 and (mir - base).abs().max() > 1e-4
    # float32, as trained
    model = build_model(ModelConfig(**TINY)).eval()
    with torch.no_grad():
        base = model(pose_tokens(np.array(gs)))
        for m in range(24):
            assert torch.allclose(model(pose_tokens(np.array([I.rotate(g, m) for g in gs]))), base, atol=1e-5)


def test_t7_information():
    def signatures(states):
        sigs = set()
        for g in states:
            c, p = I.features(g)
            col = [hash((int(c[i]), tuple(sorted((int(p[i, j]), int(c[j])) for j in range(20))))) for i in range(20)]
            sigs.add(tuple(sorted(col)))
        return len(sigs)

    def classes(states):
        return len({min(tuple(I.rotate(g, m)) for m in range(24)) for g in states})

    shells = bfs(4)
    s3, s4 = np.concatenate(shells[:4]), np.concatenate(shells)
    assert (len(s3), len(s4)) == (3502, 46741)
    assert classes(s3) == 163 and signatures(s3) == 163
    assert len({tuple(sorted(I.features(g)[0].tolist())) for g in s3}) == 156
    assert classes(s4) == 2008 and signatures(s4) == 2008
    assert len({tuple(sorted(I.features(g)[0].tolist())) for g in s4}) == 1759


def test_t8_symmetries_preserve_depth():
    shells = bfs(3)
    depth = {tuple(int(v) for v in g): d for d, sh in enumerate(shells) for g in sh}
    for d, sh in enumerate(shells):
        for g in sh[:: max(1, len(sh) // 60)]:
            for h in [I.mirror(g), I.inverse(g)] + [I.rotate(g, m) for m in range(24)]:
                assert depth[tuple(int(v) for v in h)] == d


def test_training_run_and_checkpoint():
    from cubegpt.train import run_training
    m = build_model(ModelConfig(**TINY))
    assert not any(k in m.state_dict() for k in ("DESC", "CONTENT", "PAIR"))     # tables aren't saved
    full = build_model(ModelConfig(arch="pair"))
    print(f"  full-size pair model: {count_params(full) / 1e6:.2f}M params")
    X, _ = generate_samples(T, 64, canonical=False)
    assert full(X).shape == (64, 21)
    for bad in ({"n_layers": 1}, {"d_pair": 0}):
        try:
            C.validate_model(ModelConfig(**{**TINY, **bad}))
            raise AssertionError(f"accepted {bad}")
        except C.ConfigError:
            pass
    root = tempfile.mkdtemp()
    d = {"name": "pair", "seed": 0, "eval_samples": 1024, "final_eval_samples": 0, "gen_max_batch": 4096,
         "gen_chunk_samples": 4096, "train_canonical": False, "model": dict(TINY),
         "stages": [{"steps": 300, "batch_size": 64, "lr": 2e-3, "eval_every": 100, "schedule": "wsd",
                     "warmup_samples": 64 * 30, "decay_samples": 64 * 60}]}
    r = run_training(C.to_dict(C.from_dict(d)), root, lambda: None, "fc", device="cpu")
    recs = [json.loads(l) for l in open(Path(root) / "runs/pair/metrics.jsonl")]
    assert r["status"] == "completed" and recs[-1]["eval_loss"] < recs[0]["eval_loss"] and "moe" in recs[-1]
    print(f"  eval {recs[0]['eval_loss']:.4f} -> {recs[-1]['eval_loss']:.4f}")
    shutil.rmtree(root)


if __name__ == "__main__":
    for fn in [test_t1_tables, test_t2_solved, test_t3_simulator_agreement,
               test_t4_feature_invariance_and_t6_negative_control, test_t5_model_invariance, test_t7_information,
               test_t8_symmetries_preserve_depth, test_training_run_and_checkpoint]:
        print(fn.__name__, flush=True)
        fn()
        print("  ok", flush=True)
