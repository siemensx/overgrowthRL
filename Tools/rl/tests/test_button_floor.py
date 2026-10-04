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
    set_button_floor(pol, "attack=0.1")
    z1 = pol.discrete_logits(feats).detach()
    p1 = torch.sigmoid(z1)
    assert float(p1[:, 2].min()) >= 0.05 - 1e-6           # attack never below 5%
    assert torch.equal(z0[:, [0, 1, 3, 4, 5]] > 0, z1[:, [0, 1, 3, 4, 5]] > 0)
    assert torch.allclose(z0[:, [0, 1, 3, 4, 5]], z1[:, [0, 1, 3, 4, 5]])  # other heads untouched
    assert torch.equal(z0 > 0, z1 > 0)                    # mode unchanged


def test_logprob_matches_sampling_distribution():
    pol = make()
    set_button_floor(pol, "attack=0.1")
    x = torch.randn(4096, L.total_floats * 4)
    with torch.no_grad():
        act, logp, ent, val, raw = pol.get_action_and_value(x, return_raw=True)
    press = act[:, 2 + 2].mean().item()
    assert 0.04 < press < 0.07, press                     # sampled at ~5%
    with torch.no_grad():
        _a, logp2, _e, _v = pol.get_action_and_value(x, act, raw_continuous=raw)
    assert torch.allclose(logp, logp2, atol=1e-5)         # re-evaluation agrees with the sampler


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
