"""Render a saved solve as a 3D Rubik's cube animation.

    .venv/bin/python render_solve.py eval_results/<file>.json --idx 0 --out solve.mp4
    .venv/bin/python render_solve.py --scramble <54 chars> --moves "U R' F2" --out solve.gif
    .venv/bin/python render_solve.py eval_results/<file>.json --idx 0 --still 0.5 --out frame.png
    .venv/bin/python render_solve.py eval_results/<file>.json --grid 5x3 --out grid.mp4

Geometry and turn directions are derived from cubegpt.cube (FACELET_AT and move_map), so the
picture always matches the cube representation the models were trained on. Needs Pillow, and
ffmpeg on PATH for mp4/gif output.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cubegpt.cube import FACELET_AT, move_map, rotate_string, solved_string

STICKER_RGB = {
    "W": (245, 245, 245), "Y": (255, 214, 0), "R": (222, 30, 44),
    "O": (255, 122, 26), "G": (0, 166, 81), "B": (30, 96, 224),
}
BODY_RGB = (22, 22, 26)
BG_INNER, BG_OUTER = (38, 42, 52), (12, 13, 17)
INK, INK_DIM, ACCENT = (240, 242, 246), (110, 116, 130), (0, 166, 81)

SUPERSAMPLE = 3
YAW, PITCH = math.radians(-34), math.radians(27)
CAMERA_DISTANCE = 9.0
LIGHT = np.array([-0.25, 0.6, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# ---------------------------------------------------------------------------------------------
# Cube geometry, from the training representation
# ---------------------------------------------------------------------------------------------
def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


STICKER_AT = {(tuple(np.array(pos) - 1), tuple(normal)): f for (pos, normal), f in FACELET_AT.items()}
CUBIES = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
DIRECTIONS = [tuple(int(v) for v in s * np.eye(3, dtype=int)[a]) for a in range(3) for s in (1, -1)]


def _derive_turns():
    """move name -> (axis, layer, signed quarter turns), found by matching move_map."""
    found = {}
    for name, table in move_map.items():
        if name == "null":
            continue
        for axis in range(3):
            for layer in (-1, 1):
                for quarters in (1, 2, 3):
                    r = np.rint(rotation(axis, quarters * math.pi / 2)).astype(int)
                    expected = list(range(54))
                    for (pos, normal), f in STICKER_AT.items():
                        if pos[axis] == layer:
                            expected[STICKER_AT[(tuple(r @ pos), tuple(r @ normal))]] = f
                    if expected == list(table):
                        found[name] = (axis, layer, quarters)
        assert name in found, f"move {name} is not a face turn"
    turns = {}
    for name, (axis, layer, quarters) in found.items():
        base = {1: 1, 3: -1}.get(found[name[0]][2])       # direction of the plain quarter turn
        turns[name] = (axis, layer, {1: 1, 3: -1, 2: 2 * base}[quarters])
    return turns


TURNS = _derive_turns()


def _rounded_square(half=0.43, radius=0.11, steps=5):
    pts = []
    for cx, cy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for k in range(steps + 1):
            a = math.radians(start + 90 * k / steps)
            pts.append((cx * (half - radius) + radius * math.cos(a), cy * (half - radius) + radius * math.sin(a)))
    return np.array(pts)


ROUNDED = _rounded_square()
SQUARE = np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])


def _face_points(center, direction, outline):
    d = np.array(direction, dtype=float)
    axis = int(np.argmax(np.abs(d)))
    u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
    return np.array(center) + 0.5 * d + outline[:, :1] * u + outline[:, 1:] * v


# Static per-face geometry: (cubie, direction, body quad, sticker outline or None, facelet or None)
FACES = []
for _cubie in CUBIES:
    for _d in DIRECTIONS:
        _f = STICKER_AT.get((_cubie, _d))
        FACES.append((_cubie, _d, _face_points(_cubie, _d, SQUARE),
                      _face_points(_cubie, _d, ROUNDED) if _f is not None else None, _f))


# ---------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------
def _shade(rgb, brightness):
    return tuple(int(min(255, c * brightness)) for c in rgb)


def draw_cube(state, size, turn=None, spin=0.0):
    """RGBA image of the cube. turn = (move name, progress 0..1) or None; spin = extra yaw (rad)."""
    px = size * SUPERSAMPLE
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    view = rotation(0, PITCH) @ rotation(1, YAW + spin)
    camera_view = np.array([0.0, 0.0, CAMERA_DISTANCE])
    camera_world = view.T @ camera_view
    focal = px * 1.72

    if turn:
        axis, layer, quarters = TURNS[turn[0]]
        layer_rot = rotation(axis, turn[1] * quarters * math.pi / 2)
        in_layer = lambda cubie: cubie[axis] == layer
        layer_in_front = camera_world[axis] * layer > 0.5
    else:
        layer_rot, in_layer, layer_in_front = np.eye(3), (lambda cubie: False), True

    def cubie_order(cubie):
        center = layer_rot @ np.array(cubie) if in_layer(cubie) else np.array(cubie)
        group = in_layer(cubie) == layer_in_front        # the group nearer the camera is drawn last
        return (group, -np.linalg.norm(camera_world - center))

    order = {cubie: k for k, cubie in enumerate(sorted(CUBIES, key=cubie_order))}

    def project(points, model):
        p = points @ (view @ model).T
        scale = focal / (CAMERA_DISTANCE - p[:, 2])
        return list(zip((px / 2 + p[:, 0] * scale).tolist(), (px / 2 - p[:, 1] * scale).tolist()))

    for cubie, direction, quad, sticker, facelet in sorted(FACES, key=lambda face: order[face[0]]):
        model = layer_rot if in_layer(cubie) else np.eye(3)
        normal = view @ model @ np.array(direction, dtype=float)
        face_center = view @ model @ (np.array(cubie) + 0.5 * np.array(direction))
        if normal @ (camera_view - face_center) <= 0:
            continue
        brightness = min(1.0, 0.74 + 0.30 * max(0.0, float(normal @ LIGHT)))
        draw.polygon(project(quad, model), fill=_shade(BODY_RGB, 0.7 + 0.6 * brightness) + (255,))
        if sticker is not None:
            draw.polygon(project(sticker, model), fill=_shade(STICKER_RGB[state[facelet]], brightness) + (255,))
    return img.resize((size, size), Image.LANCZOS)


def _font(paths, size):
    for path, index in paths:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


SANS_BOLD = [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("/System/Library/Fonts/Helvetica.ttc", 1),
             ("DejaVuSans-Bold.ttf", 0)]
SANS = [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("DejaVuSans.ttf", 0)]
MONO_BOLD = [("/System/Library/Fonts/Menlo.ttc", 1), ("DejaVuSansMono-Bold.ttf", 0)]


def make_background(width, height):
    y, x = np.mgrid[0:height, 0:width]
    r = np.hypot((x - width / 2) / width, (y - height * 0.47) / height) / 0.75
    t = np.clip(r, 0, 1)[..., None] ** 1.5
    rgb = np.array(BG_INNER) * (1 - t) + np.array(BG_OUTER) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


class Scene:
    def __init__(self, scramble, moves, width, height, title, subtitle):
        self.moves, self.width, self.height = moves, width, height
        self.title, self.subtitle = title, subtitle
        self.states = [scramble]
        for move in moves:
            self.states.append(rotate_string(self.states[-1], move))
        self.solved = self.states[-1] == solved_string
        self.background = make_background(width, height)
        self.unit = unit = height / 1080
        self.cube_px = int(min(width, height * 0.82))
        self.f_title = _font(SANS_BOLD, int(46 * unit))
        self.f_sub = _font(SANS, int(27 * unit))
        self.f_count = _font(SANS_BOLD, int(46 * unit))
        cell = min(62 * unit, (width - 80 * unit) / max(1, len(moves)))
        self.cell = cell
        self.f_move = _font(MONO_BOLD, int(cell * 0.44))

    def frame(self, done, turn_progress=None, spin=0.0):
        """done = moves completed; turn_progress = progress of move number `done` (or None)."""
        img = self.background.copy()
        turn = (self.moves[done], turn_progress) if turn_progress is not None else None
        cube = draw_cube(self.states[done], self.cube_px, turn, spin)
        img.paste(cube, ((self.width - self.cube_px) // 2, int(self.height * 0.47 - self.cube_px / 2)), cube)
        draw = ImageDraw.Draw(img)
        unit, margin = self.unit, 54 * self.unit
        draw.text((margin, margin), self.title, font=self.f_title, fill=INK)
        if self.subtitle:
            draw.text((margin, margin + 58 * unit), self.subtitle, font=self.f_sub, fill=INK_DIM)
        count = f"{done} / {len(self.moves)}"
        draw.text((self.width - margin, margin), count, font=self.f_count, fill=INK, anchor="ra")
        draw.text((self.width - margin, margin + 58 * unit), "moves", font=self.f_sub, fill=INK_DIM, anchor="ra")

        cell, y = self.cell, self.height - 92 * unit
        x0 = (self.width - cell * len(self.moves)) / 2
        for k, move in enumerate(self.moves):
            cx = x0 + cell * (k + 0.5)
            current = turn is not None and k == done
            if current:
                draw.rounded_rectangle((cx - cell * 0.46, y - cell * 0.42, cx + cell * 0.46, y + cell * 0.42),
                                       radius=cell * 0.2, fill=ACCENT)
            draw.text((cx, y), move, font=self.f_move, anchor="mm",
                      fill=INK if (current or k < done) else INK_DIM)
        return img


EASE_A = 2.0


def ease(t):
    """Scaled tanh: f(x) = tanh(a (x - 1/2)), rescaled so that 0 -> 0 and 1 -> 1."""
    f0, f1 = math.tanh(-EASE_A / 2), math.tanh(EASE_A / 2)
    return (math.tanh(EASE_A * (t - 0.5)) - f0) / (f1 - f0)


def timeline(scene, fps, turn_seconds, intro=1.0, outro_spin=2.6, outro_hold=1.2):
    """Yields (done, turn_progress, spin) for every frame."""
    n = len(scene.moves)
    for _ in range(round(intro * fps)):
        yield 0, None, 0.0
    for k, move in enumerate(scene.moves):
        seconds = turn_seconds * (1.4 if abs(TURNS[move][2]) == 2 else 1.0)
        frames = max(2, round(seconds * fps))
        for i in range(frames):
            yield k, ease((i + 1) / frames), 0.0
        for _ in range(round(0.07 * fps)):
            yield k + 1, None, 0.0
    for _ in range(round(0.4 * fps)):
        yield n, None, 0.0
    frames = round(outro_spin * fps)
    for i in range(frames):
        yield n, None, -2 * math.pi * ease((i + 1) / frames)
    for _ in range(round(outro_hold * fps)):
        yield n, None, 0.0


class GridScene:
    """Several solves side by side, all starting together and each turning at the same pace."""

    def __init__(self, solves, cols, rows, title, subtitle, turn_seconds, unit=1.0):
        self.solves, self.cols, self.rows = solves, cols, rows
        self.title, self.subtitle = title, subtitle
        self.cell_w, self.cell_h, self.header = int(384 * unit), int(380 * unit), int(110 * unit)
        self.width, self.height = self.cell_w * cols, self.header + self.cell_h * rows
        self.width += self.width % 2
        self.height += self.height % 2
        self.unit = unit
        self.cube_px = int(326 * unit)
        self.background = make_background(self.width, self.height)
        self.f_title = _font(SANS_BOLD, int(46 * unit))
        self.f_sub = _font(SANS, int(27 * unit))
        self.f_cap = _font(MONO_BOLD, int(22 * unit))
        self.states, self.schedule = [], []
        for solve in solves:
            states = [solve["scramble"]]
            for move in solve["moves"]:
                states.append(rotate_string(states[-1], move))
            self.states.append(states)
            t, spans = 0.0, []
            for move in solve["moves"]:
                seconds = turn_seconds * (1.4 if abs(TURNS[move][2]) == 2 else 1.0)
                spans.append((t, seconds))
                t += seconds + 0.07
            self.schedule.append(spans)
        self.solve_seconds = max(spans[-1][0] + spans[-1][1] for spans in self.schedule if spans)

    def pose(self, k, t):
        """(moves done, progress of the current turn or None) for cube k at time t."""
        for done, (start, seconds) in enumerate(self.schedule[k]):
            if t < start:
                return done, None
            if t < start + seconds:
                return done, ease((t - start) / seconds)
        return len(self.schedule[k]), None

    def frame(self, t, spin=0.0):
        img = self.background.copy()
        draw = ImageDraw.Draw(img)
        unit, margin = self.unit, 54 * self.unit
        draw.text((margin, 32 * unit), self.title, font=self.f_title, fill=INK)
        if self.subtitle:
            draw.text((self.width - margin, 46 * unit), self.subtitle, font=self.f_sub, fill=INK_DIM, anchor="ra")
        for k, solve in enumerate(self.solves):
            x0 = (k % self.cols) * self.cell_w
            y0 = self.header + (k // self.cols) * self.cell_h
            done, progress = self.pose(k, t)
            moves = solve["moves"]
            turn = (moves[done], progress) if progress is not None else None
            cube = draw_cube(self.states[k][done], self.cube_px, turn, spin)
            img.paste(cube, (x0 + (self.cell_w - self.cube_px) // 2, y0 + int(4 * self.unit)), cube)
            finished = done == len(moves)
            caption = f"{len(moves)} moves" if finished else f"{done:>2} / {len(moves)}"
            draw.text((x0 + self.cell_w / 2, y0 + self.cell_h - 22 * unit), caption, font=self.f_cap,
                      anchor="mm", fill=ACCENT if finished else INK_DIM)
        return img

    def frames(self, fps, intro=1.0, outro_spin=2.6, outro_hold=1.5):
        for _ in range(round(intro * fps)):
            yield self.frame(0.0)
        for i in range(round((self.solve_seconds + 0.4) * fps)):
            yield self.frame((i + 1) / fps)
        n = round(outro_spin * fps)
        for i in range(n):
            yield self.frame(self.solve_seconds + 1, -2 * math.pi * ease((i + 1) / n))
        for _ in range(round(outro_hold * fps)):
            yield self.frame(self.solve_seconds + 1)


def encode(images, width, height, out, fps, crf=18):
    """Write an iterator of RGB frames to an mp4 or gif; returns the frame count."""
    out = Path(out)
    mp4 = out if out.suffix != ".gif" else Path(tempfile.mkdtemp()) / "solve.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
         "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    frames = 0
    for image in images:
        proc.stdin.write(image.tobytes())
        frames += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    if out.suffix == ".gif":
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
             "fps=25,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
             str(out)], check=True)
    return frames


def main():
    global EASE_A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json", nargs="?", help="eval result file with a 'solves' list")
    ap.add_argument("--idx", type=int, nargs="+", default=None,
                    help="which solve(s) in the eval file, by 'idx' (default: 0, or the first COLSxROWS)")
    ap.add_argument("--grid", help="COLSxROWS: show several solves at once, e.g. 5x3")
    ap.add_argument("--scale", type=float, default=1.0, help="grid size multiplier (1.0 = 384x380 px cells)")
    ap.add_argument("--scramble", help="54-character cube string (instead of an eval file)")
    ap.add_argument("--moves", help="space-separated solution, with --scramble")
    ap.add_argument("--out", default="solve.mp4", help=".mp4, .gif, or .png (with --still)")
    ap.add_argument("--title", default="CubeGPT-4")
    ap.add_argument("--subtitle", help="default: the beam width from the eval file")
    ap.add_argument("--size", default="1080x1080", help="WIDTHxHEIGHT (even numbers)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ease-a", type=float, default=EASE_A,
                    help="sharpness of the tanh easing: near 0 = linear, larger = slower start and stop")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality: lower = better and bigger")
    ap.add_argument("--turn-seconds", type=float, default=0.28, help="duration of a quarter turn")
    ap.add_argument("--still", type=float,
                    help="write one PNG frame: N.5 = halfway through move N (with --grid: seconds into the solve)")
    args = ap.parse_args()
    EASE_A = args.ease_a

    subtitle = args.subtitle
    if args.grid:
        if not args.eval_json:
            ap.error("--grid needs an eval file")
        cols, rows = (int(v) for v in args.grid.lower().split("x"))
        by_idx = {s["idx"]: s for s in json.loads(Path(args.eval_json).read_text())["solves"]}
        wanted = args.idx if args.idx is not None else sorted(by_idx)[:cols * rows]
        if len(wanted) != cols * rows:
            ap.error(f"a {cols}x{rows} grid needs {cols * rows} solves, got {len(wanted)}")
        solves = [by_idx[i] for i in wanted]
        unsolved = [s["idx"] for s in solves if not s["solved"]]
        if unsolved:
            print(f"warning: unsolved cubes in the grid: {unsolved}")
        if subtitle is None:
            subtitle = f"{len(solves)} random scrambles · beam width {solves[0]['width']:,}"
        grid = GridScene(solves, cols, rows, args.title, subtitle, args.turn_seconds, args.scale)
        if args.still is not None:
            grid.frame(args.still).save(args.out)
            print(f"wrote {args.out}")
        else:
            frames = encode(grid.frames(args.fps), grid.width, grid.height, args.out, args.fps, args.crf)
            print(f"wrote {args.out}: {grid.width}x{grid.height}, {frames} frames, {frames / args.fps:.1f} s")
        return
    if args.scramble:
        scramble, moves = args.scramble, (args.moves or "").split()
    elif args.eval_json:
        solves = json.loads(Path(args.eval_json).read_text())["solves"]
        solve = next(s for s in solves if s["idx"] == (args.idx[0] if args.idx else 0))
        scramble, moves = solve["scramble"], solve["moves"]
        if subtitle is None:
            subtitle = f"beam width {solve['width']:,}"
    else:
        ap.error("give an eval file or --scramble")
    width, height = (int(v) for v in args.size.lower().split("x"))
    scene = Scene(scramble, moves, width, height, args.title, subtitle)
    if not scene.solved:
        print("warning: these moves do not solve this scramble")

    if args.still is not None:
        done = min(int(args.still), len(moves))
        progress = args.still - done
        scene.frame(done, progress if progress > 0 and done < len(moves) else None).save(args.out)
        print(f"wrote {args.out}")
    else:
        images = (scene.frame(*f) for f in timeline(scene, args.fps, args.turn_seconds))
        frames = encode(images, width, height, args.out, args.fps, args.crf)
        print(f"wrote {args.out}: {len(moves)} moves, {frames} frames, {frames / args.fps:.1f} s")


if __name__ == "__main__":
    main()
