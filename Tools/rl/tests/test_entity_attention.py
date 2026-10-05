#!/usr/bin/env python3
"""OGRL-20261004-025: attention entity encoder -- invariances, padding, loading, cost."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
sys.path.insert(0, str(HERE.parent))

from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402

FS = 4


def obs_with(n_valid: int, batch: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, FS, L.total_floats, generator=g)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    ent = x[:, :, L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(
        batch, FS, L.max_visible_entities, ef)
    ent[..., 0] = 0.0
    ent[:, :, :n_valid, 0] = 1.0
    return x.reshape(batch, -1)


def net(seed: int = 1) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(L, frame_stack=FS, layer_norm=True, entity_attention=True).eval()


def heads(p, x):
    with torch.no_grad():
        f = p._features(x)
        a = p.actor_trunk(f)
        return torch.cat([p.continuous_mean(a), p.discrete_logits(a), p.critic_out(p.critic_trunk(f))], dim=-1)


def ent_view(x):
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    return x.view(x.shape[0], FS, L.total_floats)[:, :, L.entities_start:L.entities_start + L.max_visible_entities * ef] \
        .reshape(x.shape[0], FS, L.max_visible_entities, ef)


def test_permutation_invariance():
    p = net()
    x = obs_with(3)
    y = x.clone()
    ev = ent_view(y)
    ev[:, :, [0, 1, 2]] = ent_view(x)[:, :, [2, 0, 1]].clone()
    assert torch.allclose(heads(p, x), heads(p, y), atol=1e-5)


def test_padding_ignored():
    p = net()
    x = obs_with(2)
    y = x.clone()
    ent_view(y)[:, :, 2:, 1:] = 1000.0 * torch.randn_like(ent_view(y)[:, :, 2:, 1:])  # garbage in invalid slots
    assert torch.allclose(heads(p, x), heads(p, y), atol=1e-5)


def test_no_entities_finite_and_entities_matter():
    p = net()
    out0 = heads(p, obs_with(0))
    assert torch.isfinite(out0).all()
    x = obs_with(3)
    y = x.clone()
    ent_view(y)[:, :, 1, 2:5] += 3.0                                   # move one enemy
    assert not torch.allclose(heads(p, x), heads(p, y), atol=1e-4)    # the network notices


def test_checkpoint_autodetect_and_grad():
    p = net()
    sd = p.state_dict()
    q = ActorCritic(L, frame_stack=FS)            # plain constructor, as every loader does
    q.load_state_dict(sd)
    assert q.entity_attention and q.layer_norm
    x = obs_with(3)
    assert torch.allclose(heads(p, x), heads(q.eval(), x), atol=1e-6)
    q.train()
    act, logp, ent, val = q.get_action_and_value(x)
    (logp.mean() + val.mean()).backward()
    assert q.entity_encoder.attn.self_attn.in_proj_weight.grad.abs().sum() > 0


def test_cost_report():
    torch.set_num_threads(2)
    old = ActorCritic(L, frame_stack=FS, layer_norm=True)
    new = net()
    for name, bs, train in (("collect (20 envs)", 20, False), ("update minibatch", 1024, True)):
        res = {}
        for tag, p in (("maxpool", old), ("attention", new)):
            x = obs_with(3, batch=bs)
            p.train(train)
            t0 = time.perf_counter()
            for _ in range(20 if train else 200):
                if train:
                    p.zero_grad()
                    _a, lp, _e, v = p.get_action_and_value(x)
                    (lp.mean() + v.mean()).backward()
                else:
                    with torch.no_grad():
                        p.get_action_and_value(x)
            res[tag] = (time.perf_counter() - t0) / (20 if train else 200) * 1000
        print(f"  {name}: maxpool {res['maxpool']:.2f} ms, attention {res['attention']:.2f} ms "
              f"(x{res['attention'] / res['maxpool']:.2f})")
    print(f"  params: maxpool {sum(t.numel() for t in old.parameters()):,}, "
          f"attention {sum(t.numel() for t in new.parameters()):,}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
