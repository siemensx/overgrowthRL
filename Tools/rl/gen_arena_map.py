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

# Model half-extents (see docstring).
CUBE_HALF = 1.0
PILLAR_HALF_Y = 1.345


class Level:
    def __init__(self, name: str, script: str):
        self.name = name
        self.script = script
        self.objects: list[str] = []
        self.spawns: list[str] = []
        self._id = 100

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def box(self, cx, cy, cz, sx, sy, sz, yaw=0.0, type_file=CUBE):
        """Axis-aligned (or Y-rotated) box. sx/sy/sz are HALF extents in world
        units, so a 40x2x40 slab is sx=20, sy=1, sz=20."""
        c, s = math.cos(yaw), math.sin(yaw)
        r = [c, 0, s, 0,  0, 1, 0, 0,  -s, 0, c, 0,  0, 0, 0, 1]
        rs = " ".join(f'r{i}="{v:.6g}"' for i, v in enumerate(r))
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

    def render(self) -> str:
        return f"""<?xml version="2.0" ?>
<Type>saved</Type>
<Name>{self.name}</Name>
<Description>Procedurally generated arena (Tools/rl/gen_arena_map.py)</Description>
<Shader>post</Shader>
<Terrain>
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
</Terrain>
<OutOfDate Shadow="true" AO="true" NavMesh="true" />
<SpawnPoints>
    <SpawnPoint t0="70" t1="60" t2="280" s0="1" s1="1" s2="1" r0="1" r1="0" r2="0" r3="0" r4="0" r5="1" r6="0" r7="0" r8="0" r9="0" r10="1" r11="0" r12="0" r13="0" r14="0" r15="1" />
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


def build_split_court(lvl: Level, rng: random.Random, half: float, floor_top: float,
                      cx: float, cz: float) -> None:
    """A square court bisected by a wall with two chokepoints, plus corner
    cover and two raised ledges.

    Tactically the opposite of oval_arena's open bowl: line of sight is broken,
    engagement has to happen through one of two gaps, and there is high ground.
    Those gaps are the point -- `funnel_eligible` (>=2 hostiles visible) has
    had zero samples for the whole project partly because no map ever forced a
    choice of approach.
    """
    wall_h = 4.0
    # Floor slab: top surface at floor_top, 2 units thick.
    lvl.box(cx, floor_top - 1.0, cz, half, 1.0, half)

    # Perimeter walls, inset so they sit on the slab edge.
    for dx, dz, sx, sz in (
        (0, half, half, 0.5), (0, -half, half, 0.5),
        (half, 0, 0.5, half), (-half, 0, 0.5, half),
    ):
        lvl.box(cx + dx, floor_top + wall_h / 2, cz + dz, sx, wall_h / 2, sz)

    # Central divider: three segments leaving two gaps.
    gap = 6.0
    seg = (2 * half - 2 * gap) / 3.0
    for i in (-1, 0, 1):
        offset = i * (seg + gap)
        lvl.box(cx + offset, floor_top + 1.5, cz, seg / 2, 1.5, 0.6)

    # Cover pillars, mirrored so neither side has an advantage.
    for sign in (-1, 1):
        for px, pz in ((-half * 0.55, half * 0.45), (half * 0.55, half * 0.45),
                       (0.0, half * 0.72)):
            lvl.pillar(cx + px, floor_top + 1.5, cz + sign * pz, 3.0)

    # Two raised ledges in opposite corners, with a step up.
    for sign in (-1, 1):
        lx = cx + sign * (half * 0.62)
        lz = cz + sign * (half * 0.62)
        lvl.box(lx, floor_top + 1.0, lz, 4.0, 1.0, 4.0)          # ledge, top +2
        lvl.box(lx - sign * 4.8, floor_top + 0.5, lz, 1.0, 0.5, 3.0)  # step, top +1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="gen_split_court")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--half-size", type=float, default=20.0,
                    help="half the court's side length in world units (default 20 = 40x40)")
    ap.add_argument("--script", default="Data/Scripts/arena_level.as",
                    help="level script; use Data/Scripts/arena_level_1v1_unarmed.as for the RL fork")
    ap.add_argument("--overgrowth-data", default=None,
                    help="installed Overgrowth Data/ directory (default: platform Steam path)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = Path(args.overgrowth_data) if args.overgrowth_data else paths.data_dir()
    if not (data / "Levels" / "arenas").is_dir():
        print(f"ERROR: no Levels/arenas under {data}")
        return 1

    rng = random.Random(args.seed)
    lvl = Level(args.name, args.script)

    # Built in the same neighbourhood as oval_arena's floor (y ~= 36), which is
    # known-good ground on this heightmap. The slab is solid and 2 units thick,
    # so terrain irregularity underneath does not matter.
    cx, cz, floor_top = 70.0, 280.0, 36.0
    build_split_court(lvl, rng, args.half_size, floor_top, cx, cz)

    sy = floor_top + 0.7  # spawns sit slightly above the slab, as in oval_arena
    d = args.half_size * 0.6
    # game_type 0 -> the 1v1 pair the RL fork uses: opposite sides of the divider.
    lvl.spawn(cx, sy, cz - d, 0.0, 0, 0)
    lvl.spawn(cx, sy, cz + d, math.pi, 0, 1)
    # game_type 1 -> 2v2, teams 0,0,1,1
    for i, (ox, oz, team) in enumerate([(-4, -d, 0), (4, -d, 0), (-4, d, 1), (4, d, 1)]):
        lvl.spawn(cx + ox, sy, cz + oz, 0.0 if oz < 0 else math.pi, 1, team)
    # game_type 2 -> free-for-all, teams 1,0,2,3
    for (ox, oz, team) in [(-d, -d, 1), (d, -d, 0), (d, d, 2), (-d, d, 3)]:
        lvl.spawn(cx + ox, sy, cz + oz, 0.0, 2, team)

    xml = lvl.render()
    out = data / "Levels" / "arenas" / f"{args.name}.xml"
    n_env = len(lvl.objects)
    if args.dry_run:
        print(f"[dry-run] would write {out}")
        print(f"  {n_env} EnvObjects, {len(lvl.spawns)} spawns, {len(xml)} bytes")
        return 0

    out.write_text(xml, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  {n_env} EnvObjects, {len(lvl.spawns)} spawns")
    print(f"  court {2*args.half_size:.0f}x{2*args.half_size:.0f} centred ({cx},{floor_top},{cz})")
    print(f"  navmesh will be baked by the engine on first load (OutOfDate NavMesh=true)")
    print()
    print("Launch it:")
    binary = paths.engine_binary(Path(__file__).resolve().parents[2])
    print(f'  "{binary}" --level arenas/{args.name}.xml')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
