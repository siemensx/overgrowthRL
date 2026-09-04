#!/usr/bin/env python3
"""Unit tests for curriculum.py -- the closing-distance taper (existing) and
the stall-tax ramp-in (OGRL-20260816-019, added after run6's regression made
clear that an abruptly-introduced reward term needs the same gradual on-ramp
an abruptly-removed one does)."""
from __future__ import annotations

from curriculum import Curriculum
from reward import RewardConfig, run8_reward_config


def _check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_closing_distance_full_during_bootstrap():
    c = Curriculum(bootstrap_steps=100, taper_steps=50, bootstrap_closing_weight=0.05)
    cfg = c.reward_config_for_step(0)
    _check("closing weight at step 0 is full bootstrap value", cfg.closing_distance_weight == 0.05)
    cfg = c.reward_config_for_step(99)
    _check("closing weight still full just before taper", cfg.closing_distance_weight == 0.05)


def test_closing_distance_linear_taper():
    c = Curriculum(bootstrap_steps=100, taper_steps=100, bootstrap_closing_weight=0.10)
    cfg = c.reward_config_for_step(150)  # halfway through taper
    _check("closing weight halfway through taper is ~half", abs(cfg.closing_distance_weight - 0.05) < 1e-9)
    cfg = c.reward_config_for_step(200)
    _check("closing weight fully tapered to zero", cfg.closing_distance_weight == 0.0)
    cfg = c.reward_config_for_step(10_000)
    _check("closing weight stays zero well past taper", cfg.closing_distance_weight == 0.0)


def test_stall_ramp_cold_start():
    c = Curriculum(stall_target_weight=0.02, stall_ramp_steps=1000, stall_intro_step=0)
    cfg = c.reward_config_for_step(0)
    _check("stall weight is 0 at the very start of a cold-start ramp", cfg.stall_penalty_weight == 0.0)
    cfg = c.reward_config_for_step(500)
    _check("stall weight is ~half target halfway through the ramp", abs(cfg.stall_penalty_weight - 0.01) < 1e-9)
    cfg = c.reward_config_for_step(1000)
    _check("stall weight reaches full target at the end of the ramp", cfg.stall_penalty_weight == 0.02)
    cfg = c.reward_config_for_step(50_000)
    _check("stall weight stays at target well past the ramp", cfg.stall_penalty_weight == 0.02)


def test_stall_ramp_relative_to_resume_point():
    """The bug this whole fix exists for: run6 resumed at step 2,949,120 with
    stall_penalty_weight already at full strength (the RewardConfig default),
    not ramped -- because nothing tied the ramp to the RESUME point. This
    test is the direct regression guard for OGRL-20260816-019."""
    resume_step = 2_949_120
    c = Curriculum(stall_target_weight=0.02, stall_ramp_steps=300_000, stall_intro_step=resume_step)
    cfg = c.reward_config_for_step(resume_step)
    _check("stall weight is 0 exactly at the resume point, not full strength", cfg.stall_penalty_weight == 0.0)
    cfg = c.reward_config_for_step(resume_step + 150_000)
    _check("stall weight is ~half target halfway through a resumed ramp", abs(cfg.stall_penalty_weight - 0.01) < 1e-9)
    cfg = c.reward_config_for_step(resume_step + 300_000)
    _check("stall weight reaches full target at the end of a resumed ramp", cfg.stall_penalty_weight == 0.02)
    # And a step number that would be "fully ramped" for a COLD start (step
    # number alone is large) must NOT be treated as ramped for a run whose
    # intro point is much later -- this is the actual bug from run6.
    cfg = c.reward_config_for_step(resume_step - 1)
    _check("a step before the intro point never has positive stall weight", cfg.stall_penalty_weight == 0.0)


def test_phase_name_unaffected_by_stall_ramp():
    """phase_name() describes the closing-distance schedule only -- the
    stall ramp is a separate, independent schedule and must not perturb it."""
    c = Curriculum(bootstrap_steps=100, taper_steps=50, stall_intro_step=0, stall_ramp_steps=1000)
    _check("phase_name still bootstrap", c.phase_name(50) == "bootstrap")
    _check("phase_name still taper", c.phase_name(120) == "taper")
    _check("phase_name still main", c.phase_name(500) == "main")


def test_default_base_config_unchanged():
    """No base_config passed must reproduce the exact old behavior (runs 1-7
    comparability) -- a plain RewardConfig()."""
    c = Curriculum()
    cfg = c.reward_config_for_step(10_000_000)  # deep in "main", no shaping active
    default = RewardConfig()
    _check("default base_config matches plain RewardConfig()", cfg.damage_taken_weight == default.damage_taken_weight)
    _check("default base_config matches plain RewardConfig() (2)", cfg.time_cost == default.time_cost)


def test_run8_base_config_applied():
    c = Curriculum(base_config=run8_reward_config(), bootstrap_closing_weight=0.0, stall_target_weight=0.0)
    cfg = c.reward_config_for_step(0)
    run8 = run8_reward_config()
    _check("run8 base config's damage_taken_weight carried through", cfg.damage_taken_weight == run8.damage_taken_weight == 1.0)
    _check("run8 base config's time_cost carried through", cfg.time_cost == run8.time_cost == 0.002)
    _check("run8 base config's symmetric knockout terms carried through",
           cfg.self_knockout_penalty == cfg.opponent_knockout_bonus == 10.0)
    _check("shaping fully zeroed for run8 (closing_distance)", cfg.closing_distance_weight == 0.0)
    _check("shaping fully zeroed for run8 (stall)", cfg.stall_penalty_weight == 0.0)
    _check("run8's stall/ragdoll penalties are off at the base-config level too",
           run8.stall_penalty_weight == 0.0 and run8.ragdoll_penalty == 0.0)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError:
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    if failures:
        raise SystemExit(1)
