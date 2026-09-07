#!/usr/bin/env python3
"""Procedurally generate a playable Overgrowth arena level as XML.

Reads the level format from the stock arenas and emits a new, self-contained
combat arena built from scaled primitives. Nothing is copied from a donor
level except the terrain/sky blocks (which reference existing textures) --
all geometry and every spawn is generated here.

Format notes, measured rather than assumed:

* `<EnvObject>` carries translation t0..t2, scale s0..s2 and a row-major 4x4
  matrix r0..r15. Identity is r0=r5=r10=r15=1. A rotation of theta about Y is
  r0=cos, r2=sin, r8=-sin, r10=cos (verified against stock objects).
* Primitive model extents (from the .obj vertex bounds), which is what the
  scale multiplies:
      soft_cube          2.00 x 2.00 x 2.00, centred
      soft_platform      2.00 x 0.32 x 2.00, centred
      soft_square_pillar 0.60 x 2.69 x 0.60, centred
  So a soft_cube with s=(20,1,20) is 40 x 2 x 40 world units.
* `<PlaceholderObject>` spawns carry `character_spawn`, a `game_type` and a
  `team`. arena_level.as picks `game_type_int = rand()%3` and keeps only the
  spawns whose `game_type` matches -- so game_type 0 is the 1v1 pair (teams
  0/1), 1 is 2v2 (0,0,1,1) and 2 is the free-for-all (1,0,2,3). The RL fork
  pins game_type_int to 0, so the two game_type=0 spawns are the ones
  training uses. All three sets are emitted so the map is also playable
  normally.
* `<OutOfDate ... NavMesh="true" />` makes the engine bake a navmesh on first
  load into the per-worker write-dir. Generated maps ship no .nav, unlike the
  stock arenas -- so a corpus must be pre-baked offline, never generated
  inside a training loop.

Geometry is built only from axis-aligned scaled `soft_cube` plus pillars, so
the collision and the baked navmesh stay predictable.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import paths

CUBE = "Data/Objects/Buildings/basics/soft_cube.xml"
PILLAR = "Data/Objects/Buildings/basics/soft_square_pillar.xml"
CYL = "Data/Objects/Buildings/basics/soft_cylinder.xml"     # 2 x 2 x 2, centred
DISK = "Data/Objects/Buildings/basics/soft_disk.xml"        # 2 x 0.23 x 2, centred
ARCH = "Data/Objects/Buildings/basics/soft_arch.xml"        # 1.96 x 1.03 x 0.53
POST = "Data/Objects/Buildings/Post1.xml"                   # 0.23 x 2.73 x 0.19, base ~y=0

# Model half-extents (see docstring).
CUBE_HALF = 1.0
PILLAR_HALF_Y = 1.345


class Level:
    def __init__(self, name: str, script: str, terrain: bool = False,
                 cam: tuple = (0.0, 60.0, 0.0)):
        self.name = name
        self.script = script
        self.terrain = terrain
        self.cam = cam
        self.objects: list[str] = []
        self.spawns: list[str] = []
        # XZ footprint of every prop, so later placement can avoid overlap.
        # The floor slab passes record=False -- it covers the whole court.
        self.rects: list[tuple] = []
        self._id = 100

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def box(self, cx, cy, cz, sx, sy, sz, yaw=0.0, type_file=CUBE, pitch=0.0,
            record=True, kind="prop"):
        """Axis-aligned (or rotated) box. sx/sy/sz are HALF extents in world
        units, so a 40x2x40 slab is sx=20, sy=1, sz=20.

        `pitch` tilts the box about the Z axis, which is what makes a RAMP a
        ramp. Without it every "ramp" this generator emitted was an
        axis-aligned block running from the floor to the deck height -- i.e. a
        solid wall. t_train_105 had three of them, up to 10.5 units tall,
        standing in the middle of the court. The renderer showed them as walls
        because they were walls.
        """
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # R = Ryaw . Rpitch(about Z), row-major, matching the stock levels'
        # convention (rows are the transformed basis vectors).
        r = [cy_ * cp,      -cy_ * sp,  sy_,        0,
             sp,             cp,        0,          0,
             -sy_ * cp,      sy_ * sp,  cy_,        0,
             0,              0,         0,          1]
        rs = " ".join(f'r{i}="{v:.6g}"' for i, v in enumerate(r))
        if record:
            # Yaw-rotated props get their bounding square, which is
            # conservative -- fine, we are only ever avoiding overlap.
            e = max(abs(sx), abs(sz)) if yaw else None
            self.rects.append((cx - (e or sx), cz - (e or sz),
                               cx + (e or sx), cz + (e or sz), kind))
        self.objects.append(
            f'        <EnvObject t0="{cx:.4f}" t1="{cy:.4f}" t2="{cz:.4f}" '
            f's0="{sx:.4f}" s1="{sy:.4f}" s2="{sz:.4f}" {rs} '
            f'id="{self.next_id()}" color_r="1" color_g="1" color_b="1" '
            f'type_file="{type_file}">\n            <parameters />\n        </EnvObject>'
        )

    def pillar(self, cx, cy, cz, height):
        sy = height / (2 * PILLAR_HALF_Y)
        self.box(cx, cy, cz, 1.0, sy, 1.0, 0.0, PILLAR)

    def spawn(self, x, y, z, yaw, game_type, team):
        c, s = math.cos(yaw), math.sin(yaw)
        r = [c, 0, s, 0,  0, 1, 0, 0,  -s, 0, c, 0,  0, 0, 0, 1]
        rs = " ".join(f'r{i}="{v:.6g}"' for i, v in enumerate(r))
        self.spawns.append(
            f'    <PlaceholderObject t0="{x:.4f}" t1="{y:.4f}" t2="{z:.4f}" '
            f's0="1" s1="1" s2="1" {rs} id="{self.next_id()}" '
            f'type_file="Data/Objects/IGF_Characters/IGF_Guard.xml" special_type="0">\n'
            f'        <parameters>\n'
            f'            <parameter name="Name" type="string" val="character_spawn" />\n'
            f'            <parameter name="game_type" type="string" val="{game_type}" />\n'
            f'            <parameter name="team" type="string" val="{team}" />\n'
            f'        </parameters>\n'
            f'        <Connections>\n            <Connection id="-1" />\n        </Connections>\n'
            f'    </PlaceholderObject>'
        )

    def env_object_count(self) -> int:
        return len(self.objects)

    def render(self) -> str:
        # Terrain is OPTIONAL: nothing.xml ships with no <Terrain> block at
        # all. Omitting it puts the arena in open sky, which (a) avoids the
        # generated court being buried in or perched on whatever the donor
        # heightmap happens to do at those coordinates -- the first version of
        # this generator reused oval_arena's dry_canyon terrain and landed the
        # court on a cliff face -- and (b) removes terrain collision and
        # rendering cost entirely.
        terrain = ""
        if self.terrain:
            terrain = """<Terrain>
    <Heightmap>Data/Textures/Terrain/dry_canyon/dry_canyon_hm.png</Heightmap>
    <DetailMap></DetailMap>
    <ColorMap>Data/Textures/Terrain/dry_canyon/dry_canyon_c.tga</ColorMap>
    <WeightMap>Data/Textures/Terrain/dry_canyon/dry_canyon_hm.png_6_dry_canyon_weights.png</WeightMap>
    <DetailMaps>
        <DetailMap colorpath="Data/Textures/Terrain/DetailTextures/rubble.tga" normalpath="Data/Textures/Terrain/DetailTextures/rubble_normal.tga" materialpath="Data/Materials/default.xml" />
        <DetailMap colorpath="Data/Textures/Terrain/DetailTextures/black_rock.tga" normalpath="Data/Textures/Terrain/DetailTextures/black_rock_normal.tga" materialpath="Data/Materials/default.xml" />
        <DetailMap colorpath="Data/Textures/Terrain/DetailTextures/dead_grass.tga" normalpath="Data/Textures/Terrain/DetailTextures/dead_grass_normal.tga" materialpath="Data/Materials/default.xml" />
        <DetailMap colorpath="Data/Textures/Terrain/DetailTextures/pebbles.tga" normalpath="Data/Textures/Terrain/DetailTextures/pebbles_normal.tga" materialpath="Data/Materials/default.xml" />
    </DetailMaps>
    <DetailObjects />
    <DetailObjects />
</Terrain>
"""
        return f"""<?xml version="2.0" ?>
<Type>saved</Type>
<Name>{self.name}</Name>
<Description>Procedurally generated arena (Tools/rl/gen_arena_map.py)</Description>
<Shader>post</Shader>
{terrain}<OutOfDate Shadow="true" AO="true" NavMesh="true" />
<SpawnPoints>
    <SpawnPoint t0="{self.cam[0]}" t1="{self.cam[1]}" t2="{self.cam[2]}" s0="1" s1="1" s2="1" r0="1" r1="0" r2="0" r3="0" r4="0" r5="1" r6="0" r7="0" r8="0" r9="0" r10="1" r11="0" r12="0" r13="0" r14="0" r15="1" />
</SpawnPoints>
<AmbientSounds />
<Script>{self.script}</Script>
<LevelScriptParameters>
    <parameter name="Achievements" type="string" val="flawless, no_injuries, no_kills" />
    <parameter name="Extra AO" type="string" val="0.4" />
    <parameter name="Level Boundaries" type="string" val="1" />
    <parameter name="Objectives" type="string" val="destroy_all" />
    <parameter name="Sky Rotation" type="string" val="112" />
</LevelScriptParameters>
<Sky>
    <DomeTexture>Data/Textures/skies/cloudy2.tga</DomeTexture>
    <SunAngularRad>0.133112</SunAngularRad>
    <SunColorAngle>2.15912</SunColorAngle>
    <RayToSun r0="0.782061" r1="0.572157" r2="-0.247014" />
    <ExtraAO>0.4</ExtraAO>
    <SkyRotation>112</SkyRotation>
</Sky>
<ActorObjects>
    <Group t0="0" t1="0" t2="0" s0="1" s1="1" s2="1" r0="1" r1="0" r2="0" r3="0" r4="0" r5="1" r6="0" r7="0" r8="0" r9="0" r10="1" r11="0" r12="0" r13="0" r14="0" r15="1" id="99">
        <parameters />
{chr(10).join(self.objects)}
    </Group>
{chr(10).join(self.spawns)}
</ActorObjects>
"""


def build_court(lvl: Level, rng: random.Random, half: float, floor_top: float,
                cx: float, cz: float, randomize: bool, minimal: bool = False,
                keep_out: "list | None" = None) -> None:
    """One enclosed court: floor slab, perimeter walls, a divider with
    chokepoints, cover pillars and raised ledges.

    With --randomize, the tactical parameters vary per arena. These are the
    dimensions that plausibly change what the agent must decide -- how many
    ways there are through the middle, how wide they are, how much sight is
    broken, and how much high ground exists -- rather than cosmetic variety.
    """
    if minimal:
        n_gaps, gap_w, wall_h, div_h = 0, 0.0, 4.0, 0.0
        n_pillars, ledge, ledge_h = 0, False, 0.0
    elif randomize:
        n_gaps = rng.choice([1, 2, 2, 3])          # chokepoint count
        gap_w = rng.uniform(3.5, 8.0)              # chokepoint width
        wall_h = rng.uniform(3.0, 6.0)
        div_h = rng.uniform(2.0, 4.0)
        n_pillars = rng.randint(2, 6)              # per half
        ledge = rng.random() < 0.7
        ledge_h = rng.uniform(1.5, 3.0)
    else:
        n_gaps, gap_w, wall_h, div_h = 2, 6.0, 4.0, 3.0
        n_pillars, ledge, ledge_h = 3, True, 2.0

    # Floor slab, top surface at floor_top, 2 units thick.
    lvl.box(cx, floor_top - 1.0, cz, half, 1.0, half, record=False)

    # Perimeter walls. In sky mode these are the only thing between a
    # character and a very long fall, so they are not optional.
    for dx, dz, sx, sz in ((0, half, half, 0.5), (0, -half, half, 0.5),
                           (half, 0, 0.5, half), (-half, 0, 0.5, half)):
        lvl.box(cx + dx, floor_top + wall_h / 2, cz + dz, sx, wall_h / 2, sz, kind="wall")

    # Divider across the middle, leaving n_gaps openings.
    n_seg = n_gaps + 1
    seg = (2 * half - n_gaps * gap_w) / n_seg
    if n_gaps > 0 and seg > 0.5:
        left = cx - half
        for i in range(n_seg):
            c = left + seg / 2
            lvl.box(c, floor_top + div_h / 2, cz, seg / 2, div_h / 2, 0.6, kind="wall")
            left += seg + gap_w

    # Cover pillars, mirrored so neither side is advantaged. Each candidate is
    # tested against the spawn keep-outs and against what is already placed:
    # unconditional placement is how pillars ended up standing on spawn points
    # and inside each other.
    ko = list(keep_out or [])
    for sign in (-1, 1):
        for i in range(n_pillars):
            for _ in range(24):
                a = (i + 0.5) / n_pillars * math.pi
                px = math.cos(a) * half * rng.uniform(0.35, 0.75)
                pz = sign * half * rng.uniform(0.35, 0.78)
                r = _rect(cx + px, cz + pz, 2.0, 2.0)
                if not _hits(r, ko) and not _hits(r, lvl.rects):
                    lvl.pillar(cx + px, floor_top + 1.5, cz + pz, 3.0)
                    break

    if ledge:
        for sign in (-1, 1):
            lx, lz = cx + sign * half * 0.62, cz + sign * half * 0.62
            r = _rect(lx, lz, 6.0, 6.0)
            if _hits(r, ko) or _hits(r, lvl.rects):
                continue
            lvl.box(lx, floor_top + ledge_h / 2, lz, 4.0, ledge_h / 2, 4.0, kind="deck")
            lvl.box(lx - sign * 4.8, floor_top + ledge_h / 4, lz,
                    1.0, ledge_h / 4, 3.0, kind="deck")



# --- placement bookkeeping -------------------------------------------------
# Every prop the generator drops is recorded as an XZ footprint so nothing is
# ever placed inside anything else. The old generator drew clutter positions
# from a uniform and never tested them, which is why t_train_105/106 rendered
# as interpenetrating cubes (z-fighting on every shared face) and why 259
# "small" boxes could still add up to a wall.

def _rect(cx, cz, sx, sz, margin=0.0):
    return (cx - sx - margin, cz - sz - margin, cx + sx + margin, cz + sz + margin)


def _hits(r, taken):
    return any(not (r[2] <= t[0] or t[2] <= r[0] or r[3] <= t[1] or t[3] <= r[1])
               for t in taken)


def _corridor(ax, az, bx, bz, width):
    """Footprints covering the straight line between two spawns, so cover is
    never dropped into the lane the fighters use to find each other."""
    n = max(2, int(math.hypot(bx - ax, bz - az) / (width * 0.5)))
    return [_rect(ax + (bx - ax) * i / n, az + (bz - az) * i / n, width, width)
            for i in range(n + 1)]


def build_platform(lvl, cx, cz, px, pz, deck, height, floor_top, yaw=0.0):
    """A raised deck with a REAL inclined ramp up to it.

    The ramp is a thin slab pitched so its low edge meets the floor and its
    high edge meets the deck. The previous implementation emitted an
    unrotated box of half-height `height/2`, which is a solid wall from the
    floor to the deck -- traversable by nothing.
    """
    lvl.box(px, floor_top + height, pz, deck, 0.4, deck, kind="deck")   # deck
    run = height * 2.6                                            # ~21 degrees
    pitch = math.atan2(height, run)
    mx = px - (deck + run / 2) * math.cos(yaw)
    mz = pz - (deck + run / 2) * math.sin(yaw)
    lvl.box(mx, floor_top + height / 2, mz,
            math.hypot(run, height) / 2, 0.25, deck * 0.75, yaw, pitch=pitch,
            kind="deck")
    return [_rect(px, pz, deck, deck), _rect(mx, mz, run / 2 + 1, deck)]


# Engine renderer bug, isolated 2026-09-05 by bisecting a crashing generated map
# down to object count alone. A level whose EnvObject count lands in [13, 18]
# segfaults in Engine::DrawScene -- EXC_BAD_ACCESS, KERN_INVALID_ADDRESS at
# 0xd8, i.e. a null dereference -- as soon as it is DRAWN. Counts of 12 and
# below, and 19 and above, are fine. Verified on every generated map: 13, 13 and
# 16 objects crash; 19, 20 and 24 do not; the 5-object minimal map does not.
#
# Headless training never touches Draw(), which is why six maps trained happily
# for hours and then killed the window the moment a human opened them.
#
# The engine fix belongs in the renderer's batching path; until then the
# generator simply refuses to emit a level in the dead band, padding with
# mirrored cover pillars (tactically neutral, since both halves get the same).
CRASH_BAND = (13, 18)


def pad_out_of_crash_band(lvl: "Level", half: float, floor_top: float,
                          cx: float, cz: float) -> int:
    """Add mirrored pillars until the EnvObject count clears the crash band."""
    added = 0
    while CRASH_BAND[0] <= lvl.env_object_count() <= CRASH_BAND[1]:
        i = added // 2
        sign = -1 if added % 2 else 1
        a = (i + 0.5) / 4 * math.pi
        lvl.box(cx + math.cos(a) * half * 0.8, floor_top + 1.35,
                cz + sign * math.sin(a) * half * 0.75, 1.0, 1.35, 1.0, 0.0, PILLAR)
        added += 1
        if added > 40:
            raise RuntimeError("could not pad out of the renderer crash band")
    return added




def validate_level(lvl, half, arenas):
    """Refuse to emit a level a fight cannot happen in.

    Every check here corresponds to a defect that shipped and cost training
    time, not to a hypothetical:

      * spawn height gap -- t_train_105 spawned the opponent 11.3u up on a
        deck; 78-91% of its episodes timed out.
      * spawn overlapping geometry -- the 1v3 flank spawns on 103/106 landed
        inside the old ramp blocks; 1v3 timed out on 81% and 46% of episodes.
      * prop overlap -- 105/106 interpenetrated on hundreds of faces, which is
        the z-fighting visible in a render.
      * cover fraction -- unbounded before; 105 reached 217% of floor area.
    """
    errs, lines = [], []
    sp = []
    for blk in lvl.spawns:
        import re as _re
        t = _re.search(r't0="([-\d.]+)" t1="([-\d.]+)" t2="([-\d.]+)"', blk)
        g = _re.search(r'name="game_type" type="string" val="(\d+)"', blk)
        if t and g:
            sp.append((int(g.group(1)), float(t.group(1)), float(t.group(2)), float(t.group(3))))

    for gt in sorted({x[0] for x in sp}):
        ys = [x[2] for x in sp if x[0] == gt]
        gap = max(ys) - min(ys)
        lines.append(f"game_type {gt}: {len(ys)} spawns, height gap {gap:.2f}u")
        if gap > 1.0:
            errs.append(f"game_type {gt} spawns differ in height by {gap:.2f}u "
                        f"(max 1.0) -- fighters start on different storeys")

    for gt, x, y, z in sp:
        r = _rect(x, z, 1.2, 1.2)
        if _hits(r, lvl.rects):
            errs.append(f"game_type {gt} spawn at ({x:.1f},{z:.1f}) is inside geometry")

    # Walls meet at the court corners and a ramp meets its own deck: those are
    # the design, not interpenetration. Free-standing props must never overlap.
    overlaps = 0
    rs = [r for r in lvl.rects if r[4] == "prop"]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a, b = rs[i], rs[j]
            if not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
                overlaps += 1
    lines.append(f"prop pairs overlapping in XZ: {overlaps}")
    if overlaps:
        errs.append(f"{overlaps} overlapping prop pairs -- geometry interpenetrates")

    floor = (2 * half) ** 2 * arenas
    covered = sum((r[2] - r[0]) * (r[3] - r[1]) for r in lvl.rects if r[4] != "wall")
    lines.append(f"floor covered by props: {100 * covered / floor:.1f}%")
    if covered / floor > 0.35:
        errs.append(f"props cover {100 * covered / floor:.0f}% of the floor (max 35%)")

    # The level script decides whether this is an RL episode or a stock
    # multi-round match. Getting it wrong is invisible in the geometry and
    # obvious the moment a human watches: the fallen are revived and the new
    # round's combatants fight each other instead of the agent.
    RL_SCRIPTS = ("Data/Scripts/arena_level_1v1_unarmed.as",
                  "Data/Scripts/arena_level_human_duel.as")
    lines.append(f"script: {lvl.script}")
    if lvl.script not in RL_SCRIPTS:
        errs.append(f"level script is {lvl.script}, not one of {RL_SCRIPTS} -- "
                    f"the stock arena script revives the fallen and runs rounds")

    n = lvl.env_object_count()
    if CRASH_BAND[0] <= n <= CRASH_BAND[1]:
        errs.append(f"{n} EnvObjects is inside the renderer crash band {CRASH_BAND}")
    lines.append(f"EnvObjects: {n}")
    return {"errors": errs, "lines": lines}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="gen_split_court")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--half-size", type=float, default=20.0,
                    help="half a court's side length (default 20 = 40x40)")
    ap.add_argument("--arenas", type=int, default=1,
                    help="number of isolated courts, laid out on a grid in open sky")
    ap.add_argument("--arena-spacing", type=float, default=200.0,
                    help="centre-to-centre distance between courts. Must exceed the "
                         "engine's AI awareness and observation ranges or fights leak.")
    ap.add_argument("--randomize", action="store_true",
                    help="vary chokepoint count/width, wall and divider height, cover "
                         "density and high ground per arena")
    ap.add_argument("--terrain", action="store_true",
                    help="include the dry_canyon heightmap. Off by default: the arena "
                         "then floats in open sky, which is both cheaper and avoids the "
                         "court intersecting terrain.")
    ap.add_argument("--tiers", type=int, default=0,
                    help="build N raised platforms connected by ramps, and spawn the two "
                         "fighters on DIFFERENT tiers. Measured 2026-09-05: every generated "
                         "map had a 0.0 spawn height gap and 3.3 total vertical spread, while "
                         "arena3_jam spawns its fighters 21.1 apart and run17_mac scores 0.287 "
                         "there against 0.875 on the flat corpus. The agent has never fought "
                         "an opponent on another storey.")
    ap.add_argument("--clutter", type=int, default=0,
                    help="scatter N extra small boxes around the arena. Tests whether "
                         "geometry DENSITY, not layout, is what makes oval (1,432 "
                         "EnvObjects) play differently from a generated map (~19): the "
                         "observation casts 16 geometry rays, which hit something on "
                         "every step in a cluttered level and mostly nothing in a bare "
                         "one -- a distribution the sparse-trained policy never saw.")
    ap.add_argument("--max-cover", type=float, default=0.12,
                    help="hard cap on the fraction of floor area covered by cover props "
                         "(default 0.12). Measured on the run21 corpus: maps at 13-17%% "
                         "coverage time out on 0.2-2.5%% of episodes; the generator used "
                         "to have no cap at all and produced maps at 108%% and 217%%.")
    ap.add_argument("--minimal", action="store_true",
                    help="bare floor plus perimeter walls only -- no divider, cover or "
                         "ledges. The floor of achievable geometry cost, for throughput "
                         "baselines where map content is not the variable under test.")
    ap.add_argument("--human-duel", action="store_true",
                    help="emit the map driven by arena_level_human_duel.as, so "
                         "play_match.py can fight a checkpoint on it. The duel "
                         "script spawns both sides as player actors and owns the "
                         "camera; its paths sidecar is keyed to the script name, "
                         "so no per-level path file is needed.")
    # DEFAULT IS THE RL SCRIPT, not the stock one.
    #
    # arena_level.as is Overgrowth's own multi-round arena: it revives the
    # fallen and starts a new round with fresh combatants, who then fight each
    # other rather than the agent. arena_level_1v1_unarmed.as is the RL fork --
    # it pins game_type, owns the episode protocol and never revives.
    #
    # Regenerating t_train_103/105/106 on 2026-09-07 without passing --script
    # silently moved three training maps onto the stock script. Every map this
    # generator has ever produced for training uses the RL one, so that is what
    # it defaults to; pass --script explicitly to make a normally playable map.
    ap.add_argument("--script", default="Data/Scripts/arena_level_1v1_unarmed.as")
    ap.add_argument("--overgrowth-data", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = Path(args.overgrowth_data) if args.overgrowth_data else paths.data_dir()
    if not (data / "Levels" / "arenas").is_dir():
        print(f"ERROR: no Levels/arenas under {data}")
        return 1

    rng = random.Random(args.seed)
    floor_top = 500.0 if not args.terrain else 36.0
    cols = math.ceil(math.sqrt(args.arenas))
    span = (cols - 1) * args.arena_spacing
    cam = (0.0, floor_top + 40.0, -span / 2 - 60.0)
    script = "Data/Scripts/arena_level_human_duel.as" if args.human_duel else args.script
    lvl = Level(args.name, script, terrain=args.terrain, cam=cam)

    sy = floor_top + 0.7
    for i in range(args.arenas):
        cx = (i % cols) * args.arena_spacing - span / 2
        cz = (i // cols) * args.arena_spacing - span / 2
        half = args.half_size
        d = half * 0.6
        # Lanes the fighters must have, computed BEFORE any geometry is placed:
        # every spawn this level will emit, plus the corridor between the 1v1
        # pair. Nothing -- court furniture included -- may be placed in these.
        keep_out = [_rect(cx, cz - d, 3.5, 3.5), _rect(cx, cz + d, 3.5, 3.5)]
        for ox, oz in ((-4, -d), (4, -d), (-4, d), (4, d),
                       (-d, -d), (d, -d), (d, d), (-d, d)):
            keep_out.append(_rect(cx + ox, cz + oz, 3.0, 3.0))
        for ang in (-0.7, -0.45, 0.0, 0.45, 0.7):
            keep_out.append(_rect(cx + math.sin(ang) * d, cz + math.cos(ang) * d, 3.5, 3.5))
        keep_out += _corridor(cx, cz - d, cx, cz + d, 2.5)
        build_court(lvl, rng, args.half_size, floor_top, cx, cz,
                    args.randomize, args.minimal, keep_out=keep_out)

        # Raised ground. One deck per tier, sized as a FEATURE (<=12u square,
        # <=3.5u tall) rather than a roof: t_train_105's three decks were 28.8u
        # square in a 90u court, covering the arena at three heights. Each gets
        # a real inclined ramp. Both fighters still spawn on the floor -- see
        # the spawn block below.
        for t in range(args.tiers):
            th = 2.0 + t * 1.5
            if th > 3.5:
                break
            deck = min(6.0, half * 0.18)
            ang = (t + 0.5) / max(1, args.tiers) * math.pi
            px, pz = cx + math.cos(ang) * half * 0.62, cz + math.sin(ang) * half * 0.62
            r = _rect(px, pz, deck + 2, deck + 2)
            if _hits(r, keep_out) or _hits(r, lvl.rects):
                continue
            keep_out += build_platform(lvl, cx, cz, px, pz, deck, th, floor_top,
                                       yaw=ang + math.pi)

        # Cover. Rejection-sampled with a real gap between props, capped by
        # coverage rather than by count, and never inside a keep-out. The old
        # loop drew uniform positions and tested nothing, so props grew into
        # each other (the z-fighting you can see in 105/106) and a "clutter"
        # count of 264 produced a maze rather than cover.
        placed, tries, cover_area = 0, 0, 0.0
        budget = args.max_cover * (2 * half) ** 2
        while placed < args.clutter and tries < args.clutter * 60:
            tries += 1
            bx = cx + rng.uniform(-half * 0.88, half * 0.88)
            bz = cz + rng.uniform(-half * 0.88, half * 0.88)
            w, dp = rng.uniform(0.6, 1.6), rng.uniform(0.6, 1.6)
            h = rng.uniform(0.25, 0.75)                 # vaultable, always
            r = _rect(bx, bz, w, dp, margin=1.6)        # 1.6u of walking room
            if _hits(r, keep_out) or _hits(r, lvl.rects):
                continue
            if cover_area + 4 * w * dp > budget:
                break
            lvl.box(bx, floor_top + h, bz, w, h, dp, rng.uniform(0, math.pi))
            cover_area += 4 * w * dp
            placed += 1
        if args.clutter:
            print(f"  cover: {placed}/{args.clutter} props placed "
                  f"({100 * cover_area / (2 * half) ** 2:.1f}% of floor, cap "
                  f"{100 * args.max_cover:.0f}%)")
        padded = pad_out_of_crash_band(lvl, args.half_size, floor_top, cx, cz)
        if padded:
            print(f"  padded +{padded} pillars to clear the renderer crash band "
                  f"{CRASH_BAND[0]}-{CRASH_BAND[1]} EnvObjects")
        # BOTH fighters spawn on the floor, always.
        #
        # The previous rule put the opponent on the top deck whenever --tiers
        # was used, to give the agent an opponent on another storey. Measured
        # over 60k episodes it gave the agent no opponent at all: the scripted
        # AI does not come down, so the episode runs to the cap. Timeout rate
        # by spawn height gap -- 0.0u: 0.2-2.5%; 7.8u: 6-81%; 11.3u: 78-91%.
        # Verticality has to be ground the fighters can choose to use, not a
        # wall between them.
        lvl.spawn(cx, sy, cz - d, 0.0, 0, 0)
        lvl.spawn(cx, sy, cz + d, math.pi, 0, 1)
        if i == 0:
            for ox, oz, team in [(-4, -d, 0), (4, -d, 0), (-4, d, 1), (4, d, 1)]:
                lvl.spawn(cx + ox, sy, cz + oz, 0.0 if oz < 0 else math.pi, 1, team)
            for ox, oz, team in [(-d, -d, 1), (d, -d, 0), (d, d, 2), (-d, d, 3)]:
                lvl.spawn(cx + ox, sy, cz + oz, 0.0, 2, team)

            # --- Multi-opponent groups (game_type 3 and 4) ---
            # The stock game types do not express "one agent versus N COOPERATING
            # hostiles": game_type 1 is a 2v2 and game_type 2 puts every character
            # on a distinct team (a free-for-all). arena_level_1v1_unarmed.as
            # refused to wire the opponent-count curriculum for exactly that
            # reason. Since the generator owns the spawns, define the shape
            # explicitly instead: the agent alone on team 0, every hostile on
            # team 1, so they are allies of each other and enemies of the agent.
            #   game_type 3 -> 1v2   teams [0, 1, 1]
            #   game_type 4 -> 1v3   teams [0, 1, 1, 1]
            lvl.spawn(cx, sy, cz - d, 0.0, 3, 0)
            for ang in (-0.45, 0.45):
                lvl.spawn(cx + math.sin(ang) * d, sy, cz + math.cos(ang) * d, math.pi, 3, 1)
            lvl.spawn(cx, sy, cz - d, 0.0, 4, 0)
            for ang in (-0.7, 0.0, 0.7):
                lvl.spawn(cx + math.sin(ang) * d, sy, cz + math.cos(ang) * d, math.pi, 4, 1)

    report = validate_level(lvl, args.half_size, args.arenas)
    for line in report["lines"]:
        print(f"  {line}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"  REFUSING: {e}")
        return 2

    xml = lvl.render()
    out = data / "Levels" / "arenas" / f"{args.name}.xml"
    if args.dry_run:
        print(f"[dry-run] {out}: {len(lvl.objects)} EnvObjects, {len(lvl.spawns)} spawns, {len(xml)} bytes")
        return 0

    out.write_text(xml, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  {args.arenas} arena(s), {len(lvl.objects)} EnvObjects, {len(lvl.spawns)} spawns")
    print(f"  terrain: {'dry_canyon' if args.terrain else 'NONE (open sky)'}, floor y={floor_top}")
    print(f"  randomized layout: {args.randomize}  seed={args.seed}")
    print()
    binary = paths.engine_binary(Path(__file__).resolve().parents[2])
    print("Launch it:")
    print(f'  "{binary}" --write-dir .rl_view --no-dialogues --level arenas/{args.name}.xml')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
