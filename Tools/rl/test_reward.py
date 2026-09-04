#!/usr/bin/env python3
"""Unit tests for reward.py's RewardComputer. Previously run ad-hoc (per the
research log's OGRL-20260816-014/-015 entries) without being saved as a
reusable file -- written down properly now, extended to cover the stall tax
(OGRL-20260816-018). Run with: python3 -m pytest Tools/rl/test_reward.py -v
or plain python3 Tools/rl/test_reward.py.
"""
from __future__ import annotations

from obs_schema import DEFAULT_LAYOUT, ENTITY_FLOATS
from reward import RewardComputer, RewardConfig

LAYOUT = DEFAULT_LAYOUT
N = LAYOUT.total_floats


def _base_values(self_id=1, temp_health=1.0, blood_health=1.0, awake=True, ragdoll=False):
    v = [0.0] * N
    v[LAYOUT.SELF_ID] = self_id
    v[LAYOUT.TEMP_HEALTH] = temp_health
    v[LAYOUT.BLOOD_HEALTH] = blood_health
    v[LAYOUT.KNOCKED_OUT.start] = 1.0 if awake else 0.0       # awake
    v[LAYOUT.KNOCKED_OUT.start + 1] = 0.0 if awake else 1.0   # unconscious
    if ragdoll:
        v[LAYOUT.STATE.start + 4] = 1.0  # ragdoll one-hot slot
    return v


def _set_entity(values, slot, *, valid=True, entity_id=100, distance=5.0, temp_health=1.0,
                 blood_health=1.0, attacked_by_id=-1, is_ally=False, awake=True, knocked_out_this_step=False):
    e = LAYOUT.entity_slice(slot)
    base = e.start
    values[base + 0] = 1.0 if valid else 0.0
    values[base + 1] = entity_id
    values[base + 8] = distance
    values[base + 10] = 1.0 if awake else 0.0       # knocked_out[0] = awake
    values[base + 11] = 0.0 if awake else 1.0
    values[base + 20] = temp_health
    values[base + 21] = blood_health
    values[base + 22] = attacked_by_id
    values[base + 23] = 1.0 if is_ally else 0.0
    return values


def _computer(**config_kwargs):
    return RewardComputer(LAYOUT, RewardConfig(**config_kwargs))


def _check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_self_vs_ambient_causation():
    """An entity taking damage NOT caused by self must not be credited."""
    rc = _computer()
    prev = _base_values(self_id=1)
    prev = _set_entity(prev, 0, entity_id=100, temp_health=1.0, attacked_by_id=-1, is_ally=False)
    curr = _base_values(self_id=1)
    curr = _set_entity(curr, 0, entity_id=100, temp_health=0.5, attacked_by_id=999, is_ally=False)  # some OTHER character hit it
    _, info = rc.compute(prev, curr)
    _check("ambient damage not credited", info["damage_dealt"] == 0.0 and info["opponent_knockout"] == 0.0)


def test_self_caused_damage_credited():
    rc = _computer()
    prev = _base_values(self_id=1)
    prev = _set_entity(prev, 0, entity_id=100, temp_health=1.0, attacked_by_id=-1, is_ally=False)
    curr = _base_values(self_id=1)
    curr = _set_entity(curr, 0, entity_id=100, temp_health=0.5, attacked_by_id=1, is_ally=False)  # self (id=1) hit it
    _, info = rc.compute(prev, curr)
    _check("self-caused damage credited", info["damage_dealt"] > 0.0)


def test_ally_hit_is_friendly_fire_not_credit():
    rc = _computer()
    prev = _base_values(self_id=1)
    prev = _set_entity(prev, 0, entity_id=100, temp_health=1.0, attacked_by_id=-1, is_ally=True)
    curr = _base_values(self_id=1)
    curr = _set_entity(curr, 0, entity_id=100, temp_health=0.5, attacked_by_id=1, is_ally=True)  # self hit an ALLY
    _, info = rc.compute(prev, curr)
    _check("ally hit not credited as damage_dealt", info["damage_dealt"] == 0.0)
    _check("ally hit penalized as friendly_fire", info["friendly_fire"] < 0.0)


def test_ragdoll_awake_penalized_unconscious_not():
    rc = _computer()
    prev = _base_values(awake=True, ragdoll=True)
    curr_ragdoll_awake = _base_values(awake=True, ragdoll=True)
    _, info = rc.compute(prev, curr_ragdoll_awake)
    _check("ragdoll+awake penalized", info["ragdoll_time"] < 0.0)

    rc2 = _computer()
    curr_ragdoll_unconscious = _base_values(awake=False, ragdoll=True)
    _, info2 = rc2.compute(prev, curr_ragdoll_unconscious)
    _check("ragdoll+unconscious not double-penalized by ragdoll_time", info2["ragdoll_time"] == 0.0)


def test_closing_distance_ignores_ally():
    rc = _computer(closing_distance_weight=0.1)
    prev = _base_values()
    prev = _set_entity(prev, 0, entity_id=100, distance=10.0, is_ally=True)
    curr = _base_values()
    curr = _set_entity(curr, 0, entity_id=100, distance=1.0, is_ally=True)  # got much closer to an ALLY
    _, info = rc.compute(prev, curr)
    _check("closing distance on an ally is not rewarded", info["closing_distance"] == 0.0)


def test_stall_tax_activates_after_grace_period():
    rc = _computer(stall_grace_steps=5, stall_penalty_weight=0.02)
    no_contact_step = _base_values()  # no entities set up -- no damage dealt/taken ever
    for i in range(5):
        _, info = rc.compute(no_contact_step, no_contact_step)
        _check(f"no stall tax within grace period (step {i})", info["stall_time"] == 0.0)
    _, info = rc.compute(no_contact_step, no_contact_step)  # step 6, past grace
    _check("stall tax activates once grace period is exceeded", info["stall_time"] < 0.0)


def test_stall_tax_resets_on_contact():
    rc = _computer(stall_grace_steps=2, stall_penalty_weight=0.02)
    idle = _base_values()
    for _ in range(4):  # push well past the grace period so it's actively taxing
        rc.compute(idle, idle)
    _, info = rc.compute(idle, idle)
    _check("precondition: stalled before contact", info["stall_time"] < 0.0)

    prev_hit = _base_values(self_id=1, temp_health=1.0)
    prev_hit = _set_entity(prev_hit, 0, entity_id=100, temp_health=1.0, attacked_by_id=-1)
    curr_hit = _base_values(self_id=1, temp_health=1.0)
    curr_hit = _set_entity(curr_hit, 0, entity_id=100, temp_health=0.5, attacked_by_id=1)  # self lands a hit
    _, info_after_hit = rc.compute(prev_hit, curr_hit)
    _check("landing a hit clears the stall tax immediately", info_after_hit["stall_time"] == 0.0)

    # and it takes a fresh grace period before it can activate again
    for _ in range(2):
        _, info2 = rc.compute(idle, idle)
        _check("stall tax stays cleared through a fresh grace period", info2["stall_time"] == 0.0)


def test_stall_tax_resets_on_episode_reset():
    rc = _computer(stall_grace_steps=1, stall_penalty_weight=0.02)
    idle = _base_values()
    rc.compute(idle, idle)
    rc.compute(idle, idle)
    _, info = rc.compute(idle, idle)
    _check("precondition: stalled before reset", info["stall_time"] < 0.0)
    rc.reset_episode()
    _, info_after_reset = rc.compute(idle, idle)
    _check("reset_episode() clears the stall streak", info_after_reset["stall_time"] == 0.0)


def test_stall_tax_friendly_fire_still_counts_as_contact():
    """Hitting an ally is still net-negative (friendly_fire penalty) but
    should still reset the stall clock -- it proves the agent isn't idle,
    even though it's a bad outcome on its own."""
    rc = _computer(stall_grace_steps=1, stall_penalty_weight=0.02)
    idle = _base_values()
    rc.compute(idle, idle)
    rc.compute(idle, idle)
    _, info = rc.compute(idle, idle)
    _check("precondition: stalled", info["stall_time"] < 0.0)

    prev_ff = _base_values(self_id=1)
    prev_ff = _set_entity(prev_ff, 0, entity_id=100, temp_health=1.0, attacked_by_id=-1, is_ally=True)
    curr_ff = _base_values(self_id=1)
    curr_ff = _set_entity(curr_ff, 0, entity_id=100, temp_health=0.5, attacked_by_id=1, is_ally=True)
    _, info_ff = rc.compute(prev_ff, curr_ff)
    _check("friendly-fire contact still resets the stall clock", info_ff["stall_time"] == 0.0)
    _check("friendly-fire contact is still net negative", info_ff["friendly_fire"] < 0.0)


def test_damage_taken_is_potential_based_over_drop_and_recover():
    """OGRL-20260817-028 Sec7/-027 Sec4.1: a synthetic trajectory where self
    health drops then fully recovers must sum to ~0 on damage_taken -- the
    whole point of removing the max(0, ...) clamp. Also checks the OLD
    (clamped) behavior would have failed this, so the test is actually
    exercising the fix, not just a tautology."""
    rc = _computer(damage_taken_weight=4.0)
    full = _base_values(self_id=1, temp_health=1.0, blood_health=1.0)
    hurt = _base_values(self_id=1, temp_health=0.3, blood_health=0.3)
    total_damage_taken_component = 0.0
    # full -> hurt (a hit lands) -> full (fully regenerates)
    _, info1 = rc.compute(full, hurt)
    total_damage_taken_component += info1["damage_taken"]
    _, info2 = rc.compute(hurt, full)
    total_damage_taken_component += info2["damage_taken"]
    _check("drop is penalized", info1["damage_taken"] < 0.0)
    _check("regain is credited (this is what the max(0,...) clamp used to prevent)", info2["damage_taken"] > 0.0)
    _check("drop-then-full-recovery sums to ~0 on damage_taken (potential-based)", abs(total_damage_taken_component) < 1e-9)


def test_damage_dealt_signed_delta_credits_opponent_regen_negatively():
    """Same fix, opponent side: an opponent regenerating health while still
    causation-gated to this agent (attacked_by_id == self_id) should reduce
    damage_dealt, not silently floor at 0 like the old max(0, ...) version did."""
    rc = _computer(damage_dealt_weight=3.0)
    prev = _base_values(self_id=1)
    prev = _set_entity(prev, 0, entity_id=100, temp_health=0.3, attacked_by_id=1, is_ally=False)
    curr = _base_values(self_id=1)
    curr = _set_entity(curr, 0, entity_id=100, temp_health=0.6, attacked_by_id=1, is_ally=False)  # regenerated, still last-hit-by-self
    _, info = rc.compute(prev, curr)
    _check("opponent regen under a stale causation flag is a negative (not zero) damage_dealt contribution", info["damage_dealt"] < 0.0)


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
