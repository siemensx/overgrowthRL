"""OGRL-20260924-007: a value-only (critic warm-up) update must not move any
parameter the actor reads -- entity encoder, proprioception branch, actor
trunk and heads -- while it must move the critic."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ppo"))
sys.path.insert(0, os.path.join(HERE, ".."))

from obs_schema import DEFAULT_LAYOUT  # noqa: E402
from policy import ActorCritic  # noqa: E402
from train import ppo_update  # noqa: E402


class TestValueWarmup(unittest.TestCase):
    def test_value_only_update_freezes_actor_side(self):
        torch.manual_seed(240924)
        policy = ActorCritic(DEFAULT_LAYOUT)
        obs = torch.randn(256, DEFAULT_LAYOUT.total_floats)
        with torch.no_grad():
            actions, log_probs, _e, values, raw = policy.get_action_and_value(obs, return_raw=True)
        batch = {"obs": obs, "actions": actions, "log_probs": log_probs, "values": values,
                 "advantages": torch.randn(256), "returns": values + torch.randn(256),
                 "raw_cont": raw, "valid": torch.ones(256)}
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, eps=1e-5)
        args = SimpleNamespace(n_epochs=2, minibatch_size=64, target_kl=0.02, clip_coef=0.2,
                               value_clip_coef=10.0, value_coef=0.5, entropy_coef=0.01,
                               max_grad_norm=0.5, _value_only=True)
        policy.detach_critic_features = True
        actor_side = {n: p.detach().clone() for n, p in policy.named_parameters() if not n.startswith("critic_")}
        critic_side = {n: p.detach().clone() for n, p in policy.named_parameters() if n.startswith("critic_")}
        ppo_update(policy, optimizer, batch, args)
        now = dict(policy.named_parameters())
        for n, before in actor_side.items():
            self.assertTrue(torch.equal(before, now[n].detach()), f"actor-side param moved: {n}")
        self.assertTrue(any(not torch.equal(b, now[n].detach()) for n, b in critic_side.items()),
                        "critic did not train")


if __name__ == "__main__":
    unittest.main()
