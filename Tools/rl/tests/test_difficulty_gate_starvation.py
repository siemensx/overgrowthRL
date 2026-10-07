#!/usr/bin/env python3
"""OGRL-20261007-012: the difficulty gate must be able to reach d_max_cap when d ~ U(0, d_max)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curriculum import ScenarioSampler  # noqa: E402


def test_gate_reaches_cap_from_095():
    s = ScenarioSampler(rng_seed=3, d_max_start=0.15, d_max_cap=1.0, d_step=0.1, d_min=0.0)
    s.load_curriculum_state({"d_max": 0.95})
    assert abs(s.d_max - 0.95) < 1e-9
    rng = random.Random(0)
    for _ in range(3000):                       # run38's situation: solo wins 0.89 near the top
        d = rng.uniform(0.0, s.d_max)
        s.record_episode_outcome(d, rng.random() < 0.89, opponents=1)
    assert s.d_max == 1.0, s.d_max


def test_gate_still_holds_when_losing():
    s = ScenarioSampler(rng_seed=3, d_max_start=0.15, d_max_cap=1.0, d_step=0.1, d_min=0.0)
    s.load_curriculum_state({"d_max": 0.95})
    rng = random.Random(1)
    for _ in range(3000):
        d = rng.uniform(0.0, s.d_max)
        s.record_episode_outcome(d, rng.random() < (0.9 if d < 0.85 else 0.6), opponents=1)
    assert abs(s.d_max - 0.95) < 1e-9, s.d_max


def test_outnumbered_fights_do_not_count():
    s = ScenarioSampler(rng_seed=3, d_max_start=0.15, d_max_cap=1.0, d_step=0.1, d_min=0.0)
    s.load_curriculum_state({"d_max": 0.95})
    for _ in range(3000):
        s.record_episode_outcome(0.95, True, opponents=3)
    assert abs(s.d_max - 0.95) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
