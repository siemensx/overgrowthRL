"""Regression tests for gen_arena_map.py.

Every case here is a defect that shipped into the run21 training corpus and
was only found by reading the level files after 177M steps. The generator's
own validate_level() enforces these at write time; these tests make sure
validate_level itself keeps working, and that the geometry primitives it
relies on behave.

Cost of the bugs these cover: 29.2% of all training steps went into episodes
that ended in a timeout, and the reported win rates were a blend of ~0.87 on
maps where a fight happens and ~0.09 on maps where one cannot.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_arena_map as G


def _court(half=20.0, seed=3, randomize=True, minimal=False, keep_out=None):
    import random
    lvl = G.Level("t_test", "Data/Scripts/arena_level.as")
    G.build_court(lvl, random.Random(seed), half, 500.0, 0.0, 0.0,
                  randomize, minimal, keep_out=keep_out)
    return lvl


class TestRampGeometry(unittest.TestCase):
    def test_box_without_pitch_is_axis_aligned(self):
        lvl = G.Level("t", "s")
        lvl.box(0, 500, 0, 1, 1, 1)
        self.assertIn('r0="1"', lvl.objects[0])
        self.assertIn('r5="1"', lvl.objects[0])

    def test_pitch_actually_tilts_the_box(self):
        """A ramp with no pitch is a wall. box() only ever applied yaw, so
        every 'ramp' the generator emitted ran vertically from the floor to
        the deck -- t_train_105 had three, up to 10.5u tall."""
        lvl = G.Level("t", "s")
        lvl.box(0, 500, 0, 4, 0.25, 2, pitch=math.radians(20))
        obj = lvl.objects[0]
        # r1 is the XY term of the rotation: zero for any yaw-only matrix.
        r1 = float(obj.split('r1="')[1].split('"')[0])
        self.assertGreater(abs(r1), 0.3, "pitch did not reach the rotation matrix")

    def test_build_platform_ramp_reaches_the_floor(self):
        lvl = G.Level("t", "s")
        G.build_platform(lvl, 0, 0, 12, 0, 6.0, 3.0, 500.0)
        self.assertEqual(len(lvl.objects), 2)          # deck + ramp
        ramp = lvl.objects[1]
        r1 = float(ramp.split('r1="')[1].split('"')[0])
        self.assertGreater(abs(r1), 0.0, "ramp is not inclined")
        # A ramp climbing 3u should be considerably longer than it is tall.
        sx = float(ramp.split('s0="')[1].split('"')[0])
        self.assertGreater(sx * 2, 3.0)


class TestPlacement(unittest.TestCase):
    def test_floor_slab_is_not_an_obstacle(self):
        lvl = _court(minimal=True)
        self.assertTrue(all(r[2] - r[0] < 39 or r[4] == "wall" for r in lvl.rects),
                        "the floor slab was recorded as a prop")

    def test_court_furniture_respects_keep_outs(self):
        """Pillars and ledges used to be placed unconditionally, so they stood
        on spawn points."""
        ko = [G._rect(0.0, -12.0, 3.5, 3.5), G._rect(0.0, 12.0, 3.5, 3.5)]
        lvl = _court(half=20.0, seed=11, keep_out=ko)
        for r in lvl.rects:
            if r[4] == "wall":
                continue
            self.assertFalse(G._hits(r, ko), f"prop {r} placed inside a spawn keep-out")

    def test_props_do_not_interpenetrate(self):
        lvl = _court(half=22.0, seed=5)
        props = [r for r in lvl.rects if r[4] == "prop"]
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                a, b = props[i], props[j]
                overlap = not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
                self.assertFalse(overlap, f"props overlap: {a} {b}")


class TestValidateLevel(unittest.TestCase):
    def _minimal_level_with_spawns(self, y_second=500.7):
        lvl = _court(half=20.0, minimal=True)
        lvl.spawn(0.0, 500.7, -12.0, 0.0, 0, 0)
        lvl.spawn(0.0, y_second, 12.0, math.pi, 0, 1)
        return lvl

    def test_level_spawns_on_one_plane_passes(self):
        rep = G.validate_level(self._minimal_level_with_spawns(), 20.0, 1)
        self.assertEqual(rep["errors"], [], rep["errors"])

    def test_spawn_height_gap_is_rejected(self):
        """The defect itself: --tiers put the opponent on the top deck, and
        the scripted AI does not come down. Timeout rate tracked the gap --
        0.0u gave 0.2-2.5%, 7.8u gave 6-81%, 11.3u gave 78-91%."""
        rep = G.validate_level(self._minimal_level_with_spawns(y_second=511.2), 20.0, 1)
        self.assertTrue(any("differ in height" in e for e in rep["errors"]),
                        f"a 10.5u spawn gap was accepted: {rep}")

    def test_spawn_inside_geometry_is_rejected(self):
        lvl = _court(half=20.0, minimal=True)
        lvl.box(0.0, 501.0, -12.0, 2.0, 1.0, 2.0)      # a block on the spawn
        lvl.spawn(0.0, 500.7, -12.0, 0.0, 0, 0)
        lvl.spawn(0.0, 500.7, 12.0, math.pi, 0, 1)
        rep = G.validate_level(lvl, 20.0, 1)
        self.assertTrue(any("inside geometry" in e for e in rep["errors"]), rep)

    def test_runaway_coverage_is_rejected(self):
        """t_train_105 reached 217% of floor area because nothing capped it."""
        lvl = _court(half=20.0, minimal=True)
        for i in range(40):
            lvl.box(-18 + (i % 8) * 5, 501.0, -18 + (i // 8) * 5, 4.0, 1.0, 4.0)
        lvl.spawn(0.0, 500.7, -12.0, 0.0, 0, 0)
        lvl.spawn(0.0, 500.7, 12.0, math.pi, 0, 1)
        rep = G.validate_level(lvl, 20.0, 1)
        self.assertTrue(any("cover" in e or "overlapping" in e for e in rep["errors"]), rep)

    def test_renderer_crash_band_is_rejected(self):
        lvl = G.Level("t", "s")
        for i in range(15):                             # inside CRASH_BAND 13-18
            lvl.box(i * 4.0, 501.0, 0.0, 1.0, 1.0, 1.0)
        lvl.spawn(0.0, 500.7, -12.0, 0.0, 0, 0)
        lvl.spawn(0.0, 500.7, 12.0, math.pi, 0, 1)
        rep = G.validate_level(lvl, 20.0, 1)
        self.assertTrue(any("crash band" in e for e in rep["errors"]), rep)


if __name__ == "__main__":
    unittest.main()
