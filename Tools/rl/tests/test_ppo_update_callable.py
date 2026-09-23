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


class TestPPOUpdateForwardOverride(unittest.TestCase):
    def test_override_is_used_without_changing_default_policy_diagnostics(self):
        torch.manual_seed(230923)
        policy = ActorCritic(DEFAULT_LAYOUT)
        obs = torch.zeros(128, DEFAULT_LAYOUT.total_floats)
        with torch.no_grad():
            actions, log_probs, _entropy, values, raw = policy.get_action_and_value(obs, return_raw=True)
        batch = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "advantages": torch.linspace(-1.0, 1.0, 128),
            "returns": values + torch.linspace(-0.5, 0.5, 128),
            "raw_cont": raw,
            "valid": torch.ones(128),
        }
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
        args = SimpleNamespace(
            n_epochs=1,
            minibatch_size=128,
            target_kl=0.02,
            clip_coef=0.2,
            value_clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.003,
            max_grad_norm=0.5,
        )
        calls = []

        def wrapped_forward(*forward_args, **forward_kwargs):
            calls.append(1)
            return policy.get_action_and_value(*forward_args, **forward_kwargs)

        stats = ppo_update(policy, optimizer, batch, args, update_forward=wrapped_forward)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats["nan_skips"], 0)
        self.assertEqual(stats["early_stop_minibatch"], -1)


if __name__ == "__main__":
    unittest.main()
