#!/usr/bin/env python3
"""Regression tests for reusing rollout storage for action telemetry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
from vec_buffer import VecRolloutBuffer  # noqa: E402


class VecBufferActionTelemetry(unittest.TestCase):
    def test_buffer_actions_equal_the_collected_action_stack(self):
        n_steps, n_envs, obs_dim, action_dim = 5, 3, 7, 8
        buffer = VecRolloutBuffer(
            n_steps, n_envs, obs_dim, action_dim, torch.device("cpu")
        )
        collected = []
        for step in range(n_steps):
            actions = np.arange(n_envs * action_dim, dtype=np.float32).reshape(
                n_envs, action_dim
            ) / np.float32(13.0) - np.float32(step / 7.0)
            collected.append(actions.copy())
            buffer.add(
                np.zeros((n_envs, obs_dim), dtype=np.float32),
                actions,
                np.zeros(n_envs, dtype=np.float32),
                np.zeros(n_envs, dtype=np.float32),
                np.zeros(n_envs, dtype=np.float32),
                np.zeros(n_envs, dtype=np.float32),
                raw_cont=np.zeros((n_envs, 2), dtype=np.float32),
            )

        np.testing.assert_array_equal(buffer.actions, np.stack(collected))


if __name__ == "__main__":
    unittest.main()
