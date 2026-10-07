#!/usr/bin/env python3
"""OGRL-20261007-002..008: opponent personas, intent hiding, horde groups, training profiles."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ppo"))

from curriculum import ScenarioSampler, parse_persona_mix  # noqa: E402
from env import OvergrowthEnv, PERSONAS  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
import gen_1v1_scenario as gen  # noqa: E402
import gen_arena_map as gam  # noqa: E402
import run_profile as rp  # noqa: E402


def test_parse_persona_mix():
    mix = dict(parse_persona_mix("patient=0.2,expert=0.1"))
    assert mix[PERSONAS.index("patient")] == 0.2 and abs(mix[0] - 0.7) < 1e-9
    assert parse_persona_mix("") == ()
    for bad in ("nobody=0.1", "patient=0.7,expert=0.6", "patient=-0.1"):
        try:
            parse_persona_mix(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_sampler_unchanged_when_off():
    """No persona/intent keys and no extra RNG draws when the features are off."""
    a, b = ScenarioSampler(rng_seed=5, opponents_cap=3), ScenarioSampler(rng_seed=5, opponents_cap=3)
    ea = [a.sample_episode() for _ in range(200)]
    assert all("persona" not in e and "hide_intent" not in e for e in ea)
    assert ea == [b.sample_episode() for _ in range(200)]
    assert a._rng.getstate() == b._rng.getstate()


def test_sampler_mix_and_armed_floor():
    s = ScenarioSampler(rng_seed=1, armed_stage=1, opponents_cap=7,
                        persona_mix=parse_persona_mix("passive=0.5"), hide_intent_prob=0.25)
    s.load_curriculum_state({"d_max": 1.0, "opponents_max": 3, "armed_stage": 0})
    assert s.armed_stage_index == 1 and s.opponents_max == 3
    eps = [s.sample_episode() for _ in range(4000)]
    share = sum(e["persona"] == PERSONAS.index("passive") for e in eps) / len(eps)
    hidden = sum(e["hide_intent"] for e in eps) / len(eps)
    assert 0.45 < share < 0.55 and 0.21 < hidden < 0.29
    for e in eps[:50]:
        s.record_persona_outcome(e["persona"], e["hide_intent"], e["opponents"], True)
    assert "passive" in s.persona_win_rates()


def test_intent_mask():
    env = OvergrowthEnv.__new__(OvergrowthEnv)
    env.layout, env.hide_intent = L, True
    vals = [1.0] * L.total_floats
    env._mask_intent(vals)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    for slot in range(L.max_visible_entities):
        o = L.entities_start + slot * ef
        assert all(vals[o + f] == 0.0 for f in range(33, 46))
        assert vals[o + 32] == 1.0 and vals[o + 46] == 1.0      # neighbours untouched
    assert sum(v == 0.0 for v in vals) == 13 * L.max_visible_entities
    env.hide_intent = False
    vals = [1.0] * L.total_floats
    env._mask_intent(vals)
    assert all(v == 1.0 for v in vals)


def test_generated_script():
    stock = (gen.paths.data_dir() / "Scripts" / "arena_level.as")
    if not stock.exists():
        print("  (skipped: no installed stock arena_level.as)")
        return
    out = gen.transform_script(stock.read_text(encoding="utf-8"))
    assert "rl_persona = species_word / 16;" in out and "rl_species = species_word % 16;" in out
    assert "if(rl_persona > 0){ ApplyRLPersona(params, difficulty); }" in out
    assert "game_type_int = (rl_opponents > 7 ? 7 : rl_opponents) + 1;" in out
    assert 'GetConfigValueInt("rl_human_persona")' in out


def test_horde_positions_spaced():
    for n in range(4, 8):
        pts = gam.horde_positions(0.0, 0.0, 12.0, n)
        assert len(pts) == n
        for i in range(n):
            for j in range(i + 1, n):
                d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
                assert d > 2.5, (n, i, j, d)


def test_profiles_load_and_match_run38():
    for f in sorted(rp.PROFILES.glob("run*.json")):
        prof = rp.load_profile(f.stem)
        assert prof["run_id"] and prof["controls"] in rp.CONTROLS and prof["levels"]
        n = int(rp.flag_value(prof["args"], "--n-envs")) + int(rp.flag_value(prof["args"], "--k-standby"))
        assert n % len(prof["levels"]) == 0, f.stem
    p = rp.load_profile("run38_attn_last")
    assert len(p["levels"]) == 24 and p["levels"][0] == "arenas/t_train_101.xml"
    cmd, _ = rp.build_command(p, "/ogrl_x")
    assert rp.flag_value(cmd, "--button-floor") == "attack=0.1,grab=0.3@threat"
    assert rp.flag_value(rp.load_profile("run40_horde_armed")["args"], "--n-envs") == "12"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
