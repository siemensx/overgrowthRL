#!/usr/bin/env python3
"""Focused correctness checks for asynchronous rollout batching."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from async_vec_env import AsyncVecOvergrowthEnv, _StepResult  # noqa: E402
from env import ACTION_DIM  # noqa: E402


class AsyncVecEnvBatching(unittest.TestCase):
    def test_ready_cohort_keeps_worker_trajectories_aligned(self):
        vec = AsyncVecOvergrowthEnv.__new__(AsyncVecOvergrowthEnv)
        vec.n_envs = 4
        vec.observation_dim = 3
        vec.min_ready_batch = 3
        vec.max_batch_wait_seconds = 0.1
        vec._current_obs = np.zeros((4, 3), dtype=np.float32)
        vec._current_obs[:, 0] = np.arange(4, dtype=np.float32) * 10.0
        vec._pool = ThreadPoolExecutor(max_workers=4)
        step_counts = [0] * vec.n_envs

        def step_one(index: int, _action: np.ndarray) -> _StepResult:
            step_counts[index] += 1
            next_obs = np.full(
                (vec.observation_dim,), index * 100 + step_counts[index], dtype=np.float32
            )
            info = {"worker": index, "step": step_counts[index]}
            return _StepResult(next_obs, next_obs.copy(), float(index), False, False, info)

        vec._step_one = step_one
        policy_batch_sizes: list[int] = []

        def act_fn(raw_batch: np.ndarray):
            policy_batch_sizes.append(len(raw_batch))
            actions = np.zeros((len(raw_batch), ACTION_DIM), dtype=np.float32)
            actions[:, 0] = np.tanh(raw_batch[:, 0])
            log_probs = raw_batch[:, 0].copy()
            values = raw_batch[:, 0].copy() + 0.5
            return raw_batch.copy(), actions, log_probs, values

        wait_call = 0

        def phased_wait(futures, timeout=None, return_when=None):
            nonlocal wait_call
            wait_call += 1
            candidates = list(futures)
            if wait_call == 1:
                selected = candidates[:1]
            elif wait_call == 2:
                selected = candidates[:2]
            else:
                selected = candidates
            done = set(selected)
            return done, set(futures) - done

        try:
            with patch("async_vec_env.wait", side_effect=phased_wait):
                rollout = vec.collect_rollout(2, act_fn)
        finally:
            vec._pool.shutdown(wait=True)

        self.assertEqual(policy_batch_sizes[:2], [4, 3])
        self.assertEqual(rollout.ready_batch_sizes[:2], [4, 3])
        self.assertEqual(len(rollout.ready_wait_seconds), 3)
        self.assertEqual(wait_call, 4)  # includes the extra two-future cohort wait
        self.assertEqual(rollout.batches, len(rollout.ready_batch_sizes))
        self.assertEqual(rollout.obs.shape, (2, 4, 3))
        np.testing.assert_array_equal(rollout.raw_obs[0, :, 0], [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(rollout.raw_obs[1, :, 0], [1.0, 101.0, 201.0, 301.0])
        np.testing.assert_array_equal(rollout.last_raw_obs[:, 0], [2.0, 102.0, 202.0, 302.0])
        np.testing.assert_array_equal(rollout.rewards, [[0.0, 1.0, 2.0, 3.0]] * 2)
        np.testing.assert_array_equal(rollout.values[1], [1.5, 101.5, 201.5, 301.5])
        self.assertEqual([rollout.infos[1][i]["worker"] for i in range(4)], [0, 1, 2, 3])

    def test_invalid_cohort_settings_fail_before_engine_startup(self):
        with self.assertRaises(ValueError):
            AsyncVecOvergrowthEnv(n_envs=2, repo_root="", min_ready_batch=3)
        with self.assertRaises(ValueError):
            AsyncVecOvergrowthEnv(n_envs=2, repo_root="", min_ready_batch=1.5)
        with self.assertRaises(ValueError):
            AsyncVecOvergrowthEnv(n_envs=2, repo_root="", max_batch_wait_seconds=float("inf"))


if __name__ == "__main__":
    unittest.main()
