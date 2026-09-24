#!/usr/bin/env python3
"""Keep both synchronous and asynchronous frozen-sweep paths callable."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import time
from itertools import count
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ppo"))

import frozen_collector_sweep  # noqa: E402
from env import ACTION_DIM  # noqa: E402
from obs_schema import DEFAULT_LAYOUT  # noqa: E402


class _FakePolicy:
    def load_state_dict(self, _state):
        pass

    def eval(self):
        pass

    def get_action_and_value(self, obs):
        batch_size = len(obs)
        actions = torch.zeros((batch_size, ACTION_DIM), dtype=torch.float32)
        scalar = torch.zeros(batch_size, dtype=torch.float32)
        return actions, scalar, scalar, scalar


class _FakeNormalizer:
    def __init__(self, _layout, frame_stack):
        self.frame_stack = frame_stack

    def load_state_dict(self, _state):
        pass

    def normalize(self, obs, update=False):
        return obs


class _FakeSyncEnv:
    def __init__(self, **kwargs):
        self.workers = kwargs["n_envs"]

    def reset(self, seeds=None):
        return np.zeros((self.workers, DEFAULT_LAYOUT.total_floats), dtype=np.float32)

    def step(self, actions):
        next_obs = np.ones((self.workers, DEFAULT_LAYOUT.total_floats), dtype=np.float32)
        rewards = np.zeros(self.workers, dtype=np.float32)
        terminals = np.zeros(self.workers, dtype=np.float32)
        truncated = np.zeros(self.workers, dtype=np.float32)
        time.sleep(0.002)
        return next_obs, rewards, terminals, truncated, [{} for _ in range(self.workers)]

    def drain_perf(self):
        return {"reset_seconds": 0.0, "pool_hits": 0, "pool_misses": 0}

    def close(self):
        pass


class FrozenCollectorSweep(unittest.TestCase):
    def test_sync_measurement_uses_step_path(self):
        checkpoint = {
            "layout_total_floats": DEFAULT_LAYOUT.total_floats,
            "frame_stack": 1,
            "global_step": 12,
            "policy": {},
            "obs_normalizer": {},
        }
        args = Namespace(
            torch_threads=1,
            torch_interop_threads=1,
            checkpoint="unused.pt",
            frame_stack=1,
            repo_root="unused",
            levels=["arenas/oval_arena_1v1_unarmed.xml"],
            workers=2,
            seed=7,
            shm_tag="unit-test",
            max_episode_steps=10,
            act_period=4,
            collector="sync",
            k_standby=0,
            min_ready_batch=1,
            max_ready_wait_ms=0.5,
            warmup_seconds=0.006,
            measure_seconds=0.01,
            rollout_steps=2,
        )

        perf_tick = count()
        with patch.object(frozen_collector_sweep.torch, "load", return_value=checkpoint), \
                patch.object(frozen_collector_sweep, "ActorCritic", return_value=_FakePolicy()), \
                patch.object(frozen_collector_sweep, "ObservationNormalizer", _FakeNormalizer), \
                patch.object(frozen_collector_sweep, "VecOvergrowthEnv", _FakeSyncEnv), \
                patch.object(frozen_collector_sweep.time, "perf_counter",
                             side_effect=lambda: next(perf_tick) * 0.01):
            result = frozen_collector_sweep.run(args)

        self.assertGreater(result["transitions"], 0)
        self.assertGreater(result["policy_batches"], 0)
        self.assertEqual(result["collector"], "sync")
        self.assertEqual(result["mean_ready_batch"], 0.0)
        self.assertAlmostEqual(result["policy_inference_seconds"], result["policy_batches"] * 0.01)
        self.assertFalse(result["optimizer_created"])


if __name__ == "__main__":
    unittest.main()
