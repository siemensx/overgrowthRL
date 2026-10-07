#!/usr/bin/env python3
"""OGRL-20261007-014: decoupled weight decay hook and net_health telemetry."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
sys.path.insert(0, str(HERE.parent))

from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from net_health import SUMMARY_WEIGHT, health_stats, install_weight_decay  # noqa: E402

FS = 4


def make():
    torch.manual_seed(0)
    return ActorCritic(L, frame_stack=FS, layer_norm=True, entity_attention=True, attention_last_frame_only=True)


def obs(n=64):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, FS, L.total_floats, generator=g)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    ent = x[:, :, L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(n, FS, L.max_visible_entities, ef)
    ent[..., 0] = 0.0
    ent[:, :, :3, 0] = 1.0
    return x.reshape(n, -1)


def test_decay_targets_and_math():
    p = make()
    opt = torch.optim.Adam(p.parameters(), lr=1e-3, eps=1e-5)
    plain_keys = set(opt.state_dict().keys())
    names = dict(install_weight_decay(p, opt, 0.01, 0.05))
    assert names[SUMMARY_WEIGHT] == 0.05
    assert all(dict(p.named_parameters())[n].dim() == 2 for n in names)
    assert "continuous_log_std" not in names and not any(n.endswith("bias") for n in names)
    before = {n: t.detach().clone() for n, t in p.named_parameters()}
    for t in p.parameters():
        t.grad = torch.zeros_like(t)            # zero gradient: Adam moves nothing, only the decay acts
    opt.step()
    after = dict(p.named_parameters())
    assert torch.allclose(after[SUMMARY_WEIGHT], before[SUMMARY_WEIGHT] * (1 - 1e-3 * 0.05))
    assert torch.allclose(after["actor_trunk.0.weight"], before["actor_trunk.0.weight"] * (1 - 1e-3 * 0.01))
    assert torch.equal(after["continuous_log_std"], before["continuous_log_std"])
    assert torch.equal(after["actor_trunk.0.bias"], before["actor_trunk.0.bias"])
    assert set(opt.state_dict().keys()) == plain_keys       # checkpoint format unchanged


def test_off_is_noop():
    p = make()
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    assert install_weight_decay(p, opt, 0.0) == []


def test_health_stats_shape_and_no_side_effects():
    p = make().eval()
    x = obs()
    with torch.no_grad():
        a = p.discrete_logits(p.actor_trunk(p._features(x)))
    h = health_stats(p, x)
    with torch.no_grad():
        b = p.discrete_logits(p.actor_trunk(p._features(x)))
    assert torch.equal(a, b)
    assert "summary" in h and any(k.startswith("trunk") for k in h)
    for k, v in h.items():
        if isinstance(v, dict):
            assert 0 <= v["saturated"] <= 1 and 0 <= v["grad_pass"] <= 1 and v["eff_dims"] > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
