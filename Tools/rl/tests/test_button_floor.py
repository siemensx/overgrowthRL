#!/usr/bin/env python3
"""OGRL-20261004-015: per-button press floor -- consistent distribution, unchanged mode and checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
sys.path.insert(0, str(HERE.parent))

from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic, set_button_floor  # noqa: E402


def make():
    torch.manual_seed(0)
    pol = ActorCritic(L, frame_stack=4)
    with torch.no_grad():
        pol.discrete_logits.bias[:] = torch.tensor([0.0, 0.0, -9.0, 3.0, 0.0, 0.0])  # attack ~1e-4, like run31
    return pol


def test_floor_raises_min_prob_and_keeps_mode():
    pol = make()
    x = torch.randn(64, L.total_floats * 4)
    feats = pol.actor_trunk(pol._features(x))
    z0 = pol.discrete_logits(feats).detach()
    set_button_floor(pol, "attack=0.1", grounded_only=False)
    pol.discrete_logits.gate = None
    z1 = pol.discrete_logits(feats).detach()
    p1 = torch.sigmoid(z1)
    assert float(p1[:, 2].min()) >= 0.05 - 1e-6           # attack never below 5%
    assert torch.equal(z0[:, [0, 1, 3, 4, 5]] > 0, z1[:, [0, 1, 3, 4, 5]] > 0)
    assert torch.allclose(z0[:, [0, 1, 3, 4, 5]], z1[:, [0, 1, 3, 4, 5]])  # other heads untouched
    assert torch.equal(z0 > 0, z1 > 0)                    # mode unchanged


def test_logprob_matches_sampling_distribution():
    pol = make()
    set_button_floor(pol, "attack=0.1", grounded_only=False)
    x = torch.randn(4096, L.total_floats * 4)
    with torch.no_grad():
        act, logp, ent, val, raw = pol.get_action_and_value(x, return_raw=True)
    press = act[:, 2 + 2].mean().item()
    assert 0.04 < press < 0.07, press                     # sampled at ~5%
    with torch.no_grad():
        _a, logp2, _e, _v = pol.get_action_and_value(x, act, raw_continuous=raw)
    assert torch.allclose(logp, logp2, atol=1e-5)         # re-evaluation agrees with the sampler


def test_grounded_gate():
    pol = make()
    set_button_floor(pol, "attack=0.1")              # grounded_only by default
    x = torch.randn(4096, L.total_floats * 4)
    newest = 3 * L.total_floats + L.GROUNDED
    x[:2048, newest] = 1.0                          # normalised "on the ground"
    x[2048:, newest] = -1.0                         # in the air
    with torch.no_grad():
        act, logp, ent, val, raw = pol.get_action_and_value(x, return_raw=True)
        _a, logp2, _e, _v = pol.get_action_and_value(x, act, raw_continuous=raw)
    ground_press = act[:2048, 4].mean().item()
    air_press = act[2048:, 4].mean().item()
    assert 0.03 < ground_press < 0.08, ground_press   # floored on the ground
    assert air_press < 0.01, air_press                # untouched in the air (bias -9)
    assert torch.allclose(logp, logp2, atol=1e-5)     # gate recomputed identically on re-evaluation


def test_threat_gate():
    sys.path.insert(0, str(HERE.parent / "ppo"))
    from normalize import ObservationNormalizer
    pol = make()
    nrm = ObservationNormalizer(L, frame_stack=4)          # mean 0, var 1: normalised == raw
    print_spec = set_button_floor(pol, "attack=0.1,grab=0.4@threat", normalizer=nrm)
    assert print_spec == {"attack": "0.1@grounded", "grab": "0.4@threat"}
    with torch.no_grad():
        pol.discrete_logits.bias[3] = -9.0                  # grab ~never pressed on its own
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    x = torch.zeros(3000, 4, L.total_floats)
    ent = x[:, -1, L.entities_start:L.entities_start + L.max_visible_entities * ef].view(3000, L.max_visible_entities, ef)
    ent[:, 0, 0] = 1.0                                      # one valid enemy
    ent[:1000, 0, 8] = 1.2; ent[:1000, 0, 15] = 1.0         # close and attacking  -> threat
    ent[1000:2000, 0, 8] = 5.0; ent[1000:2000, 0, 15] = 1.0 # attacking but far    -> no threat
    ent[2000:, 0, 8] = 1.2; ent[2000:, 0, 15] = 0.0         # close, not attacking -> no threat
    x = x.reshape(3000, -1)
    with torch.no_grad():
        act, logp, _e, _v, raw = pol.get_action_and_value(x, return_raw=True)
        _a, logp2, _e2, _v2 = pol.get_action_and_value(x, act, raw_continuous=raw)
    g = act[:, 2 + 3]
    assert 0.14 < g[:1000].mean().item() < 0.26, g[:1000].mean()   # floored at 0.2 under threat
    assert g[1000:].mean().item() < 0.01                             # untouched otherwise
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_checkpoint_keys_unchanged_and_clear():
    pol = make()
    keys = set(pol.state_dict())
    set_button_floor(pol, "attack=0.1")
    assert set(pol.state_dict()) == keys
    set_button_floor(pol, None)
    assert pol.discrete_logits.floor is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
