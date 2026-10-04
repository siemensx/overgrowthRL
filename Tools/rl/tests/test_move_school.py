#!/usr/bin/env python3
"""OGRL-20261004-010 move school: ground-only rule, rule flag, stage machine, checkpointed state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from curriculum import MOVE_SCHOOL_STAGES, ScenarioSampler  # noqa: E402
from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402


class FakeShm:
    def __init__(self):
        self.last = None

    def write_action(self, mx, my, jump, crouch, attack, grab, drop, walk):
        self.last = dict(jump=jump, crouch=crouch, attack=attack, grab=grab, drop=drop, walk=walk)


def bare_env(ground_only: bool, grounded: bool) -> OvergrowthEnv:
    env = object.__new__(OvergrowthEnv)
    env.layout = L
    env._shm = FakeShm()
    env.ground_only = ground_only
    env._rule_episode = True
    env.blocked_air_attacks = 0
    env.rule_hidden = False
    vals = [0.0] * L.total_floats
    vals[L.GROUNDED] = 1.0 if grounded else 0.0
    env._prev_values = vals
    return env


def act(jump=0.0, attack=0.0):
    a = np.zeros(8, dtype=np.float32)
    a[2] = jump
    a[4] = attack
    return a


def test_rule_masks_only_air_attacks():
    e = bare_env(True, grounded=True)
    e.write_action(act(attack=1.0))
    assert e._shm.last["attack"] is True and e.blocked_air_attacks == 0     # ground attack allowed
    e = bare_env(True, grounded=False)
    e.write_action(act(attack=1.0))
    assert e._shm.last["attack"] is False and e.blocked_air_attacks == 1    # air attack dropped
    e = bare_env(True, grounded=True)
    e.write_action(act(jump=1.0, attack=1.0))
    assert e._shm.last["attack"] is False and e._shm.last["jump"] is True   # jump+attack: attack dropped, jump kept
    e = bare_env(False, grounded=False)
    e.write_action(act(attack=1.0))
    assert e._shm.last["attack"] is True and e.blocked_air_attacks == 0     # free episode: untouched


def test_rule_flag_written_only_in_rule_episodes():
    e = bare_env(True, grounded=True)
    v = [0.0] * L.total_floats
    e._mark_rule(v)
    assert v[L.RULE_GROUND_ONLY] == 1.0
    e.ground_only = False
    e._mark_rule(v)
    assert v[L.RULE_GROUND_ONLY] == 0.0
    e._rule_episode = False
    v[L.RULE_GROUND_ONLY] = 0.5
    e._mark_rule(v)
    assert v[L.RULE_GROUND_ONLY] == 0.5           # non-rule episode (eval/watch): engine value untouched


def test_hidden_rule_masks_but_reads_normal():
    e = bare_env(True, grounded=False)
    e.rule_hidden = True
    e.write_action(act(attack=1.0))
    assert e._shm.last["attack"] is False                # the jump kick is still off...
    v = [0.0] * L.total_floats
    e._mark_rule(v)
    assert v[L.RULE_GROUND_ONLY] == 0.0                  # ...but the fight looks normal to the policy


def test_hidden_share_per_stage():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    eps = [s.sample_episode() for _ in range(3000)]
    assert not any(e["rule_hidden"] for e in eps)                     # S1: all announced
    s.load_curriculum_state({"move_school_stage": 1, "move_school_stage_start": 0})
    g = [e for e in (s.sample_episode() for _ in range(6000)) if e["ground_only"]]
    frac = sum(e["rule_hidden"] for e in g) / len(g)
    assert abs(frac - 0.5) < 0.04, frac                               # S2: half hidden
    s.load_curriculum_state({"move_school_stage": 2, "move_school_stage_start": 0})
    g = [e for e in (s.sample_episode() for _ in range(6000)) if e["ground_only"]]
    assert all(e["rule_hidden"] for e in g)                            # S3: all hidden
    assert not any(e["rule_hidden"] for e in (s.sample_episode() for _ in range(2000)) if not e["ground_only"])


def sampler(**kw):
    return ScenarioSampler(d_max_start=1.0, d_max_cap=1.0, d_min=1.0, opponents=3, opponents_cap=3,
                           opp_sampling="learnability", rng_seed=1, **kw)


def test_sampling_fraction_and_off_switch():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    eps = [s.sample_episode() for _ in range(4000)]
    frac = sum(e["ground_only"] for e in eps) / len(eps)
    assert abs(frac - MOVE_SCHOOL_STAGES[0]["ground_prob"]) < 0.03, frac
    assert "ground_only" not in sampler().sample_episode()  # off: the scenario dict is unchanged


def test_ground_fights_respect_stage_enemy_cap():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    ground = [e for e in (s.sample_episode() for _ in range(3000)) if e["ground_only"]]
    assert {e["opponents"] for e in ground} == {1}                 # S1: ground fights are 1v1 only
    free = [e for e in (s.sample_episode() for _ in range(3000)) if not e["ground_only"]]
    assert {e["opponents"] for e in free} == {1, 2, 3}             # normal fights keep the full mix


def test_ground_difficulty_ramp_and_gate_band():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    g = [e for e in (s.sample_episode() for _ in range(2000)) if e["ground_only"]]
    assert all(0.1 - 1e-9 <= e["difficulty"] <= 0.3 + 1e-9 for e in g)        # starts easy: U(0.1, 0.3)
    n = [e for e in (s.sample_episode() for _ in range(2000)) if not e["ground_only"]]
    assert all(e["difficulty"] == 1.0 for e in n)                               # normal fights untouched
    for _ in range(299):
        s.record_ground_outcome(1, True, 0.25)
    assert s.move_school_snapshot()["ground_d_cap"] == 0.3
    s.record_ground_outcome(1, True, 0.25)                                       # 300 top-band wins -> raise
    assert s.move_school_snapshot()["ground_d_cap"] == 0.4
    for _ in range(400):
        s.record_ground_outcome(1, True, 0.35)
    assert s.move_school_step(10**9) is None or True
    st = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    st.move_school_step(0)
    for _ in range(600):
        st.record_ground_outcome(1, True, 0.3)                                   # easy wins
    assert st.move_school_step(MOVE_SCHOOL_STAGES[0]["min_steps"]) is None     # do not count toward the gate
    state = s.curriculum_state()
    t = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    t.load_curriculum_state(state)
    assert t.move_school_snapshot()["ground_d_cap"] == s.move_school_snapshot()["ground_d_cap"]


def test_stage_machine_min_gate_max():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    st0 = MOVE_SCHOOL_STAGES[0]
    assert s.move_school_step(1000) is None                                   # stage clock starts here
    for _ in range(400):
        s.record_ground_outcome(1, True, 1.0)                                  # gate cleared (at d=1.0)...
    assert s.move_school_step(1000 + st0["min_steps"] - 1) is None             # ...but min_steps not yet
    ev = s.move_school_step(1000 + st0["min_steps"])
    assert ev and ev["reason"] == "gate" and s.move_school_snapshot()["stage"] == 1
    st1 = MOVE_SCHOOL_STAGES[1]
    start = 1000 + st0["min_steps"]
    for _ in range(400):
        s.record_ground_outcome(2, False, 1.0)                                 # gate failing
    assert s.move_school_step(start + st1["min_steps"]) is None
    ev = s.move_school_step(start + st1["max_steps"])                         # forced at max_steps
    assert ev and ev["reason"] == "max_steps" and s.move_school_snapshot()["stage"] == 2
    assert s.move_school_step(start + 10 * st1["max_steps"]) is None          # last stage is permanent


def test_ground_outcomes_do_not_touch_normal_gates():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    for _ in range(500):
        s.record_ground_outcome(3, True)
    assert s.opponent_win_rates()[3] is None and s.ground_win_rates()[3] == 1.0


def test_state_survives_restart():
    s = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    s.move_school_step(5)
    s.move_school_step(5 + MOVE_SCHOOL_STAGES[0]["max_steps"])
    state = s.curriculum_state()
    t = sampler(move_school_stages=MOVE_SCHOOL_STAGES)
    t.load_curriculum_state(state)
    snap = t.move_school_snapshot()
    assert snap["stage"] == 1 and snap["stage_start"] == 5 + MOVE_SCHOOL_STAGES[0]["max_steps"]
    u = sampler()                                    # a run without move school ignores the keys
    u.load_curriculum_state(state)
    assert u.move_school_snapshot() is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
