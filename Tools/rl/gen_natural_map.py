#!/usr/bin/env python3
"""Terrain-like arenas without terrain (OGRL-20261010-009). User review 2026-10-10: the t_train maps are large
(1v3 spawns ~34 m apart, so the agent -- which never attacks first -- waits for enemies to walk up), enemies arrive
one behind another from the same side, and the scattered boxes look like no real map. Real terrain costs ~1.9x
step rate, so slopes are built from a handful of tilted slabs instead:

  * a smaller court: half-size 13-16 m (26-32 m across, vs 40 m)
  * sloped embankments along the boundary walls (the edge reads as a bowl, not a box), split into segments
  * 1-3 hills: a raised plateau (0.6-1.8 m) with 2-4 gentle ramps (12-20 degrees) up its sides
  * no cover boxes, no pillars, no divider
  * enemies start ~6-9 m from the agent, in one of three arrangements per map: front arc, surround, two flanks
  * all RL spawn groups: game_type 0 (1v1), 3 (1v2), 4 (1v3), 5-8 (1v4..1v7)

Uses gen_arena_map.py's Level writer and validator (spawn-in-geometry, spawn height gap, overlap, renderer crash
band). Never reuses a name: refuses to overwrite an existing level.

    python3 Tools/rl/gen_natural_map.py --name t_nat_501 --seed 501
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from gen_arena_map import CRASH_BAND, Level, _hits, _rect, validate_level  # noqa: E402

LAYOUTS = ("front", "surround", "flank")


def oriented_box(lvl: Level, center, half, rows, kind="deck"):
    """Emit a box whose local axes (rows of the rotation, world space) are given explicitly. Verified in the engine
    (OGRL-20261010-009 orientation probe): the level format's r0..r11 are the local axes as ROWS, so a slab pitched
    with gen_arena_map.Level.box(yaw != 0) tilts along world x whatever its yaw. This avoids that."""
    r = [*rows[0], 0.0, *rows[1], 0.0, *rows[2], 0.0, 0.0, 0.0, 0.0, 1.0]
    rs = " ".join(f'r{i}="{v:.6g}"' for i, v in enumerate(r))
    cx, cy, cz = center
    lvl.objects.append(
        f'        <EnvObject t0="{cx:.4f}" t1="{cy:.4f}" t2="{cz:.4f}" '
        f's0="{half[0]:.4f}" s1="{half[1]:.4f}" s2="{half[2]:.4f}" {rs} '
        f'id="{lvl.next_id()}" color_r="1" color_g="1" color_b="1" '
        f'type_file="Data/Objects/Buildings/basics/soft_cube.xml">\n            <parameters />\n        </EnvObject>')
    ext = [abs(rows[0][k]) * half[0] + abs(rows[1][k]) * half[1] + abs(rows[2][k]) * half[2] for k in (0, 2)]
    lvl.rects.append((cx - ext[0], cz - ext[1], cx + ext[0], cz + ext[1], kind))


def slab(lvl: Level, x0, z0, x1, z1, h_low, h_high, floor_top, width, kind="deck", thick=0.25):
    """A flat slab whose top surface rises from (x0, z0) at height h_low to (x1, z1) at h_high, `width` wide."""
    dx, dz, rise = x1 - x0, z1 - z0, h_high - h_low
    run = math.hypot(dx, dz)
    length = math.hypot(run, rise)
    u = (dx / length, rise / length, dz / length)                     # up the slope
    w = (-dz / run, 0.0, dx / run)                                   # across the slope, horizontal
    y = (w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0])   # w x u, points up
    # centre of the TOP surface's midpoint is at the mid height; push the box centre down by its half thickness
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    cy = floor_top + (h_low + h_high) / 2 - thick * y[1]
    oriented_box(lvl, (cx - thick * y[0], cy, cz - thick * y[2]), (length / 2, thick, width / 2), (u, y, w), kind)
    if kind != "floor":
        # exact horizontal outline for spawn placement (the AABB recorded in lvl.rects is far larger for diagonal
        # slabs); spawns are tested against this, the validator against the walls and plateau boxes
        lvl.rects.pop()
        SLABS.append((cx, cz, dx / run, dz / run, run / 2, width / 2))


def ridge(lvl: Level, cx, cz, length, height, slope_deg, angle, floor_top):
    """A walkable mound: two slabs back to back along `angle`, each rising to `height` above the ground at the crest."""
    run = height / math.tan(math.radians(slope_deg))
    ax, az = math.cos(angle), math.sin(angle)               # across the ridge
    crest = g(cx, cz) + height
    for sgn in (-1, 1):
        x0, z0 = cx + sgn * ax * run, cz + sgn * az * run
        slab(lvl, x0, z0, cx, cz, g(x0, z0), crest, floor_top, length, kind="wall")
    return run


GROUND = {"tilt": 0.0, "dir": 0.0}
SLABS: list = []          # exact outlines of tilted slabs: (cx, cz, ux, uz, half_run, half_width)


def on_slab(x, z, margin):
    for cx, cz, ux, uz, hr, hw in SLABS:
        a = (x - cx) * ux + (z - cz) * uz
        b = -(x - cx) * uz + (z - cz) * ux
        if abs(a) <= hr + margin and abs(b) <= hw + margin:
            return True
    return False


def g(x, z):
    """Ground height above floor_top of the tilted court at (x, z)."""
    return math.tan(GROUND["tilt"]) * (x * math.cos(GROUND["dir"]) + z * math.sin(GROUND["dir"]))


def build(lvl: Level, rng: random.Random, floor_top: float, layout: str, half: float) -> tuple:
    """v2 (2026-10-10, after the user saw v1 as "a flat map with some slopes on the sides"): the ground itself has
    shape -- an uneven rim, a broad terrace with wide ramps, hills, and low rolling ridges crossing the open ground.
    Terrain pieces are tagged "wall" for the validator (they are meant to touch each other); spawns are still
    checked against every footprint."""
    feats = {"rim": 0, "terrace": 0, "hills": 0, "ridges": 0}
    SLABS.clear()
    GROUND["tilt"] = math.radians(rng.uniform(2.0, 3.5))
    GROUND["dir"] = rng.uniform(0, 2 * math.pi)
    feats["tilt_deg"] = round(math.degrees(GROUND["tilt"]), 1)
    # the court floor: one big slab tilted along GROUND["dir"] (a hillside), 2 m thick
    dx, dz = math.cos(GROUND["dir"]), math.sin(GROUND["dir"])
    t = math.tan(GROUND["tilt"])
    u = (dx / math.sqrt(1 + t * t), t / math.sqrt(1 + t * t), dz / math.sqrt(1 + t * t))
    w = (-dz, 0.0, dx)
    yv = (w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0])
    span = half * 1.45                                  # covers the square court at any heading
    oriented_box(lvl, (-yv[0] * 1.0, floor_top - yv[1] * 1.0, -yv[2] * 1.0), (span, 1.0, span), (u, yv, w), kind="floor")
    lvl.rects.pop()                                      # the floor is not an obstacle
    # OGRL-20261010-017: the rim banks (up to 2.2 m) and terraces (up to 2.0 m) run right up to the boundary wall, so
    # the old 3 m wall cleared them by only 0.7-1.2 m and agents hopped out of the map. A held full jump peaks at
    # 3.8 m above take-off (measured, t_flat): the wall top is now >= 7 m above the highest terrain beside it, on the
    # uphill side too. Draws no random numbers, so the same seed gives the same terrain and start positions.
    wall_h = 1.5 * math.tan(GROUND["tilt"]) * half + math.tan(GROUND["tilt"]) * half + 2.2 + 7.0
    for dx, dz, sx, sz in ((0, half, half, 0.5), (0, -half, half, 0.5), (half, 0, 0.5, half), (-half, 0, 0.5, half)):
        lvl.box(dx, floor_top - math.tan(GROUND["tilt"]) * half * 1.5 + wall_h / 2, dz, sx, wall_h / 2, sz, kind="wall")
    # uneven rim: 3 segments per side rising toward the wall, plus corner wedges
    rim_runs = []
    for side in range(4):
        ang = side * math.pi / 2
        nx, nz = math.cos(ang), math.sin(ang)
        tx, tz = -nz, nx
        for seg in (-2 / 3, 0.0, 2 / 3):
            run = rng.uniform(2.5, 4.5)
            rise = rng.uniform(0.8, 2.2)
            rim_runs.append(run)
            ox, oz = tx * seg * half, tz * seg * half
            x0, z0 = nx * (half - 0.5 - run) + ox, nz * (half - 0.5 - run) + oz
            x1, z1 = nx * (half - 0.5) + ox, nz * (half - 0.5) + oz
            slab(lvl, x0, z0, x1, z1, g(x0, z0), g(x1, z1) + rise, floor_top, half * 2 / 3, kind="wall")
            feats["rim"] += 1
    for sx_, sz_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        x0, z0, x1, z1 = sx_ * (half - 4.0), sz_ * (half - 4.0), sx_ * (half - 1.0), sz_ * (half - 1.0)
        slab(lvl, x0, z0, x1, z1, g(x0, z0), g(x1, z1) + 1.6, floor_top, 3.0, kind="wall")
    inner = half - max(rim_runs) - 1.0
    # spawns are placed AFTER the terrain (see place_spawns); keep a central clearing so open ground remains
    spawn_ko = [_rect(0.0, 0.0, 3.0, 3.0)]
    terrain = []                                          # footprints of terrace / hills / ridges

    def free(rect):
        return not _hits(rect, spawn_ko) and not _hits(rect, terrain)

    # a broad terrace against one side, with one or two wide ramps down its inner face
    if rng.random() < 0.75:
        for _ in range(40):
            side = rng.randrange(4)
            ang = side * math.pi / 2
            nx, nz = math.cos(ang), math.sin(ang)
            tx, tz = -nz, nx
            wlen = rng.uniform(8.0, 16.0)
            depth = rng.uniform(5.0, 8.0)
            h = rng.uniform(1.0, 2.0)
            off = rng.uniform(-(half - wlen / 2 - 1), half - wlen / 2 - 1)
            ramp_run = h / math.tan(math.radians(rng.uniform(10, 16)))
            ccx = nx * (half - 0.5 - depth / 2) + tx * off
            ccz = nz * (half - 0.5 - depth / 2) + tz * off
            hx = abs(nx) * depth / 2 + abs(tx) * wlen / 2
            hz = abs(nz) * depth / 2 + abs(tz) * wlen / 2
            foot = _rect(ccx - nx * ramp_run / 2, ccz - nz * ramp_run / 2, hx + abs(nx) * ramp_run / 2,
                         hz + abs(nz) * ramp_run / 2)
            if not free(foot):
                continue
            gt_ = g(ccx, ccz)
            lvl.box(ccx, floor_top + gt_ + (h - 1.5) / 2 + 0.0, ccz, hx, (h + 1.5) / 2, hz, kind="wall")
            n_ramps = rng.choice([1, 2])
            for k in range(n_ramps):
                rw = wlen * (0.8 if n_ramps == 1 else 0.38)
                o = 0.0 if n_ramps == 1 else (-1 if k == 0 else 1) * wlen * 0.27
                ex, ez = ccx - nx * depth / 2 + tx * o, ccz - nz * depth / 2 + tz * o
                bx_, bz_ = ex - nx * ramp_run, ez - nz * ramp_run
                slab(lvl, bx_, bz_, ex, ez, g(bx_, bz_), gt_ + h, floor_top, rw, kind="wall")
            terrain.append(foot)
            feats["terrace"] += 1
            break
    # hills: plateau with ramps on 2-4 sides
    for _ in range(300):
        if feats["hills"] >= rng.choice([1, 1, 2]):
            break
        h = rng.uniform(0.6, 1.6)
        top = rng.uniform(1.5, 3.0)
        run = h / math.tan(math.radians(rng.uniform(14, 22)))
        px, pz = rng.uniform(-inner, inner), rng.uniform(-inner, inner)
        foot = _rect(px, pz, top + run, top + run)
        if not free(foot) or abs(px) + top + run > half - 0.6 or abs(pz) + top + run > half - 0.6:
            continue
        gp = g(px, pz)
        lvl.box(px, floor_top + gp + (h - 1.5) / 2, pz, top, (h + 1.5) / 2, top, kind="wall")
        for sd in rng.sample(range(4), rng.randint(2, 4)):
            a = sd * math.pi / 2
            nx, nz = math.cos(a), math.sin(a)
            bx_, bz_ = px + nx * (top + run), pz + nz * (top + run)
            slab(lvl, bx_, bz_, px + nx * top, pz + nz * top, g(bx_, bz_), gp + h, floor_top, top * 1.8, kind="wall")
        terrain.append(foot)
        feats["hills"] += 1
    # rolling ridges across the open ground
    want = rng.randint(3, 5)
    for _ in range(400):
        if feats["ridges"] >= want:
            break
        length = rng.uniform(6.0, 14.0)
        height = rng.uniform(0.5, 1.2)
        slope = rng.uniform(10, 16)
        angle = rng.uniform(0, math.pi)
        cx, cz = rng.uniform(-inner, inner), rng.uniform(-inner, inner)
        run = height / math.tan(math.radians(slope))
        ca, sa = abs(math.cos(angle)), abs(math.sin(angle))
        hx, hz = ca * run + sa * length / 2, sa * run + ca * length / 2
        foot = _rect(cx, cz, hx, hz)
        # ridges may run into hills, terraces and the rim (terrain merges); only the start positions stay clear
        if _hits(foot, spawn_ko) or abs(cx) + hx > half - 0.6 or abs(cz) + hz > half - 0.6:
            continue
        ridge(lvl, cx, cz, length, height, slope, angle, floor_top)
        terrain.append(foot)
        feats["ridges"] += 1
    hostile = place_spawns(lvl, rng, half, inner, floor_top)
    feats["arrangements"] = {n: len(v) for n, v in hostile.items()}
    return hostile, feats

ARRANGEMENTS = ("front", "surround", "staggered", "behind", "pincer", "scattered")
BASE_GT = {1: 0, 2: 3, 3: 4}            # game_type of the base group per opponent count; variants at +100*v


def place_spawns(lvl: Level, rng: random.Random, half: float, inner: float, floor_top: float) -> dict:
    """OGRL-20261010-010: six start arrangements per opponent count (1-3), one drawn per round by the level script:
      front      enemies in an arc ahead of the agent, 6-9 m
      surround   evenly around the agent, 4-6 m
      staggered  one behind another along a line, ~4 / 8.5 / 13 m -- they arrive one at a time
      behind     an arc behind the agent's facing, 5-8 m
      pincer     agent near the rim, enemies from both sides along it, 6-9 m
      scattered  anywhere 5-14 m away, at least 3 m apart
    The agent's facing is random. Every point is on open ground (validator margin 1.2 m off every footprint) and a
    group's start heights differ by < 0.95 m (the validator's 1.0 m rule against storey-separated starts)."""
    lim = half - 1.6

    def ok(x, z):
        return (abs(x) < lim and abs(z) < lim and not _hits(_rect(x, z, 1.25, 1.25), lvl.rects)
                and not on_slab(x, z, 0.6))

    def agent_spot(lay):
        for _ in range(400):
            if lay == "pincer":
                side = rng.randrange(4)
                a = side * math.pi / 2
                d = inner - rng.uniform(0.5, 2.5)
                t = rng.uniform(-inner * 0.6, inner * 0.6)
                x, z = math.cos(a) * d - math.sin(a) * t, math.sin(a) * d + math.cos(a) * t
                face = a + math.pi                                    # back to the rim
            else:
                x, z = rng.uniform(-inner * 0.6, inner * 0.6), rng.uniform(-inner * 0.6, inner * 0.6)
                face = rng.uniform(0, 2 * math.pi)
            if ok(x, z):
                return x, z, face
        return None

    def hostiles(lay, n, ax, az, face):
        if lay == "front":
            angs = [face + (-0.8 + 1.6 * k / max(1, n - 1) if n > 1 else rng.uniform(-0.3, 0.3)) for k in range(n)]
            dists = [rng.uniform(6, 9) for _ in range(n)]
        elif lay == "surround":
            off = rng.uniform(0, 2 * math.pi)
            angs = [off + 2 * math.pi * k / max(n, 2) for k in range(n)]
            dists = [rng.uniform(4, 6) for _ in range(n)]
        elif lay == "staggered":
            d0 = face + rng.uniform(-1.0, 1.0)
            angs = [d0 + rng.uniform(-0.25, 0.25) for _ in range(n)]
            dists = [4.0 + 4.5 * k + rng.uniform(-0.5, 0.5) for k in range(n)]
        elif lay == "behind":
            angs = [face + math.pi + (-0.6 + 1.2 * k / max(1, n - 1) if n > 1 else 0.0) for k in range(n)]
            dists = [rng.uniform(5, 8) for _ in range(n)]
        elif lay == "pincer":
            angs = [face + (math.pi / 2 if k % 2 == 0 else -math.pi / 2) + rng.uniform(-0.3, 0.3) for k in range(n)]
            dists = [rng.uniform(6, 9) for _ in range(n)]
        else:
            angs = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
            dists = [rng.uniform(5, 14) for _ in range(n)]
        return [(ax + math.cos(a) * d, az + math.sin(a) * d) for a, d in zip(angs, dists)]

    out = {n: [] for n in BASE_GT}
    for n in BASE_GT:
        for lay in ARRANGEMENTS:
            for _ in range(300):
                spot = agent_spot(lay)
                if spot is None:
                    break
                ax, az, face = spot
                pts = hostiles(lay, n, ax, az, face)
                if not all(ok(x, z) for x, z in pts):
                    continue
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 2.5 for i_, p in enumerate(pts) for q in pts[i_ + 1:]):
                    continue
                hs = [g(ax, az)] + [g(x, z) for x, z in pts]
                if max(hs) - min(hs) > 0.95:
                    continue
                out[n].append((lay, (ax, az, face), pts))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layout", choices=LAYOUTS + ("random",), default="random")
    ap.add_argument("--script", default="Data/Scripts/arena_level_1v1_unarmed.as")
    ap.add_argument("--overgrowth-data", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data = Path(a.overgrowth_data) if a.overgrowth_data else paths.data_dir()
    out = data / "Levels" / "arenas" / f"{a.name}.xml"
    if out.exists() and not a.dry_run:
        print(f"REFUSING: {out} exists -- never reuse a live map name")
        return 1
    rng = random.Random(a.seed)
    layout = rng.choice(LAYOUTS) if a.layout == "random" else a.layout
    half = rng.uniform(13.0, 16.0)
    floor_top = 500.0
    lvl = Level(a.name, a.script, terrain=False, cam=(0.0, floor_top + 25.0, -half - 20.0))
    hostile, feats = build(lvl, rng, floor_top, layout, half)
    for n, variants in hostile.items():
        if not variants:
            print(f"  REFUSING: no valid start arrangement for 1v{n}")
            return 2
        for v, (lay, (ax, az, face), pts) in enumerate(variants):
            gt = BASE_GT[n] + 100 * v
            lvl.spawn(ax, floor_top + g(ax, az) + 0.7, az, face, gt, 0)
            for x, z in pts:
                lvl.spawn(x, floor_top + g(x, z) + 0.7, z, math.atan2(ax - x, az - z), gt, 1)
    report = validate_level(lvl, half, 1)
    for line in report["lines"]:
        print(f"  {line}")
    print(f"  half-size {half:.1f} m, terrain {feats}, 1v3 arrangements "
          f"{[v[0] for v in hostile[3]]}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"  REFUSING: {e}")
        return 2
    if a.dry_run:
        return 0
    out.write_text(lvl.render(), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
