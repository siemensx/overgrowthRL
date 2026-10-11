#!/usr/bin/env python3
"""Draw a generated level from its XML, without the game: a top-down height map (exact surface height of every
block, from its translation / scale / rotation rows) with the 1v3 spawn positions, and an oblique 3D view.
OGRL-20261010-009: lets an agent SEE a map it generated instead of screen-capturing a game window (screen capture
on a shared desktop picks up other applications -- DEAD_ENDS 2026-10-07).

    python3 Tools/rl/render_level_preview.py arenas/t_nat_501.xml --out /tmp/t_nat_501.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

MODEL_HALF = {"soft_cube": (1.0, 1.0, 1.0), "soft_square_pillar": (0.3, 1.345, 0.3),
              "soft_platform": (1.0, 0.16, 1.0), "soft_cylinder": (1.0, 1.0, 1.0), "soft_disk": (1.0, 0.115, 1.0)}


def parse(level: str, data=None):
    x = ((Path(data) if data else paths.data_dir()) / "Levels" / level).read_text(errors="replace")
    boxes = []
    for m in re.finditer(r"<EnvObject ([^>]*)>", x):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        t = np.array([float(a["t0"]), float(a["t1"]), float(a["t2"])])
        s = np.array([float(a["s0"]), float(a["s1"]), float(a["s2"])])
        r = np.array([float(a[f"r{i}"]) for i in range(16)]).reshape(4, 4)[:3, :3]
        kind = Path(a.get("type_file", "soft_cube")).stem
        half = np.array(MODEL_HALF.get(kind, (1.0, 1.0, 1.0))) * s
        boxes.append((t, r, half))
    spawns = []
    for blk in re.findall(r"<PlaceholderObject[^>]*>.*?</PlaceholderObject>", x, re.S):
        g = re.search(r'name="game_type" type="string" val="(\d+)"', blk)
        tm = re.search(r'name="team" type="string" val="(\d+)"', blk)
        p = re.search(r't0="([-\d.]+)" t1="([-\d.]+)" t2="([-\d.]+)"', blk)
        if g and p:
            spawns.append((int(g.group(1)), int(tm.group(1)) if tm else -1, float(p.group(1)), float(p.group(3))))
    return boxes, spawns


def corners(t, r, half):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    return t + c @ r          # rows of r are the box's local axes in world space


def height_map(boxes, xs, zs, floor_y):
    X, Z = np.meshgrid(xs, zs)
    H = np.full(X.shape, np.nan)
    for t, r, half in boxes:
        # vertical line (X, y, Z): local_i(y) = (p - t) . row_i / half_i, affine in y
        lo, hi = np.full(X.shape, -1e9), np.full(X.shape, 1e9)
        ok = np.ones(X.shape, bool)
        for i in range(3):
            row = r[i]
            a = ((X - t[0]) * row[0] + (-t[1]) * row[1] + (Z - t[2]) * row[2]) / half[i]
            b = row[1] / half[i]
            if abs(b) < 1e-9:
                ok &= np.abs(a) <= 1.0
            else:
                y1, y2 = (-1.0 - a) / b, (1.0 - a) / b
                lo, hi = np.maximum(lo, np.minimum(y1, y2)), np.minimum(hi, np.maximum(y1, y2))
        ok &= lo <= hi
        top = np.where(ok, hi, np.nan)
        H = np.fmax(H, top)
    return H - floor_y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("level")
    ap.add_argument("--out", required=True)
    ap.add_argument("--game-type", type=int, default=4, help="spawn group to mark (4 = 1v3)")
    ap.add_argument("--data", default=None, help="Data/ directory to read from (default: the installed game)")
    a = ap.parse_args()
    boxes, spawns = parse(a.level, a.data)
    floor = max(boxes, key=lambda b: b[2][0] * b[2][2])           # the widest box is the floor slab
    floor_y = floor[0][1] + floor[2][1]
    ext = floor[2][0] + 1.0
    xs = np.linspace(-ext, ext, 360)
    zs = np.linspace(-ext, ext, 360)
    H = height_map(boxes, xs, zs, floor_y)
    fig = plt.figure(figsize=(13, 6), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(np.clip(H, 0, 4), origin="lower", extent=(-ext, ext, -ext, ext), cmap="terrain", vmin=-0.5, vmax=4)
    ax1.contour(xs, zs, np.nan_to_num(H, nan=0), levels=[0.25, 0.75, 1.25, 1.75, 2.25], colors="k", linewidths=0.4)
    for gt, team, x, z in spawns:
        if gt == a.game_type:
            ax1.plot(x, z, "o", ms=8, color="white" if team == 0 else "red", mec="black")
    ax1.set_title(f"{a.level}: height above floor (m); white = agent, red = 1v3 enemies", fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    faces_idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    polys, cols = [], []
    light = np.array([0.4, 0.8, 0.3]) / np.linalg.norm([0.4, 0.8, 0.3])
    for t, r, half in boxes:
        if half[0] * half[2] > 0.9 * floor[2][0] * floor[2][2]:
            continue                                                 # skip the floor slab itself (drawn as a plane)
        c = corners(t, r, half) - np.array([0, floor_y, 0])
        for f in faces_idx:
            q = c[list(f)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            n = n / (np.linalg.norm(n) or 1)
            shade = 0.35 + 0.65 * abs(float(n @ light))
            polys.append([(p[0], p[2], p[1]) for p in q])
            cols.append((0.55 * shade, 0.6 * shade, 0.5 * shade))
    ax2.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.15), linewidths=0.2))
    gx, gz = np.meshgrid([-ext, ext], [-ext, ext])
    ax2.plot_surface(gx, gz, np.zeros_like(gx), color=(0.85, 0.85, 0.8), alpha=0.6)
    for gt, team, x, z in spawns:
        if gt == a.game_type:
            ax2.scatter([x], [z], [0.9], color="white" if team == 0 else "red", edgecolors="black", s=40, depthshade=False)
    ax2.set_xlim(-ext, ext); ax2.set_ylim(-ext, ext); ax2.set_zlim(0, 2 * ext)
    ax2.view_init(elev=32, azim=-60)
    ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
