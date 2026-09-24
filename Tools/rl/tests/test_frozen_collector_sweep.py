#!/usr/bin/env python3
"""Keep both synchronous and asynchronous frozen-sweep paths callable."""
from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
import sys
import time
from itertools import count
import unittest
from types import SimpleNamespace
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


class _CapturingEnv(_FakeSyncEnv):
    created = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.scenarios = []
        self.retention_enabled_at_startup = os.environ.get("OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS") == "1"
        self.created.append(self)

    def reset(self, seeds=None):
        scenario_fn = self.kwargs["scenario_fn"]
        self.scenarios.append(scenario_fn() if scenario_fn is not None else None)
        return super().reset(seeds)

    def collect_rollout(self, n_steps, act_fn):
        raw = np.zeros((self.workers, DEFAULT_LAYOUT.total_floats), dtype=np.float32)
        act_fn(raw)
        time.sleep(0.002)
        return SimpleNamespace(
            obs=np.zeros((n_steps, self.workers, DEFAULT_LAYOUT.total_floats), dtype=np.float32),
            batches=1,
            ready_batch_sizes=[self.workers],
            ready_wait_seconds=[0.0],
        )


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
            opponents=None,
            difficulty=None,
            soft_reset=False,
            hard_reset_every=50,
            capture_engine_logs=False,
            warmup_seconds=0.006,
            measure_seconds=0.01,
            rollout_steps=2,
            out="unused.json",
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

    def test_fixed_scenario_and_reset_profile_reach_both_collectors(self):
        checkpoint = {
            "layout_total_floats": DEFAULT_LAYOUT.total_floats,
            "frame_stack": 1,
            "global_step": 12,
            "policy": {},
            "obs_normalizer": {},
        }
        for collector in ("sync", "async"):
            with self.subTest(collector=collector):
                _CapturingEnv.created.clear()
                args = Namespace(
                    torch_threads=1,
                    torch_interop_threads=1,
                    checkpoint="unused.pt",
                    frame_stack=1,
                    repo_root="unused",
                    levels=[f"arenas/t_train_{i}.xml" for i in range(101, 107)],
                    workers=2,
                    seed=7,
                    shm_tag=f"scenario-{collector}",
                    max_episode_steps=1200,
                    act_period=4,
                    collector=collector,
                    k_standby=0,
                    min_ready_batch=4,
                    max_ready_wait_ms=5.0,
                    opponents=3,
                    difficulty=1.0,
                    soft_reset=True,
                    hard_reset_every=50,
                    capture_engine_logs=False,
                    warmup_seconds=0.0,
                    measure_seconds=0.004,
                    rollout_steps=2,
                    out="unused.json",
                )
                target = "VecOvergrowthEnv" if collector == "sync" else "AsyncVecOvergrowthEnv"
                with patch.object(frozen_collector_sweep.torch, "load", return_value=checkpoint), \
                        patch.object(frozen_collector_sweep, "ActorCritic", return_value=_FakePolicy()), \
                        patch.object(frozen_collector_sweep, "ObservationNormalizer", _FakeNormalizer), \
                        patch.object(frozen_collector_sweep, target, _CapturingEnv):
                    frozen_collector_sweep.run(args)

                env = _CapturingEnv.created[-1]
                self.assertEqual(env.kwargs["soft_reset"], True)
                self.assertEqual(env.kwargs["hard_reset_every"], 50)
                self.assertEqual(env.scenarios[0], {"opponents": 3, "difficulty": 1.0})
                if collector == "async":
                    self.assertEqual(env.kwargs["min_ready_batch"], 4)
                    self.assertAlmostEqual(env.kwargs["max_batch_wait_seconds"], 0.005)

    def test_log_capture_requires_exact_actor_and_map_evidence(self):
        checkpoint = {
            "layout_total_floats": DEFAULT_LAYOUT.total_floats,
            "frame_stack": 1,
            "global_step": 12,
            "policy": {},
            "obs_normalizer": {},
        }
        levels = ["arenas/t_train_101.xml", "arenas/t_train_102.xml"]
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp:
            args = Namespace(
                torch_threads=1,
                torch_interop_threads=1,
                checkpoint="unused.pt",
                frame_stack=1,
                repo_root="unused",
                levels=levels,
                workers=2,
                seed=7,
                shm_tag="capture-test",
                max_episode_steps=1200,
                act_period=4,
                collector="async",
                k_standby=0,
                min_ready_batch=4,
                max_ready_wait_ms=5.0,
                opponents=3,
                difficulty=1.0,
                soft_reset=True,
                hard_reset_every=50,
                capture_engine_logs=True,
                warmup_seconds=0.0,
                measure_seconds=0.004,
                rollout_steps=2,
                out=str(Path(temp) / "arm.json"),
            )
            evidence = {
                "valid": True,
                "engines": [
                    {
                        "observed_character_ids_from_notice_logs": [0, 1, 2, 3],
                        "assigned_level": level,
                    }
                    for level in levels
                ],
            }
            before = os.environ.get("OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS")
            try:
                with patch.object(frozen_collector_sweep.torch, "load", return_value=checkpoint), \
                        patch.object(frozen_collector_sweep, "ActorCritic", return_value=_FakePolicy()), \
                        patch.object(frozen_collector_sweep, "ObservationNormalizer", _FakeNormalizer), \
                        patch.object(frozen_collector_sweep, "AsyncVecOvergrowthEnv", _CapturingEnv), \
                        patch.object(frozen_collector_sweep, "collect_engine_character_logs", return_value=evidence) as collect:
                    result = frozen_collector_sweep.run(args)
            finally:
                self.assertEqual(os.environ.get("OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS"), before)

        env = _CapturingEnv.created[-1]
        self.assertTrue(env.retention_enabled_at_startup)
        self.assertEqual(collect.call_args.args[2:6], (2, 0, levels, Path(args.out).with_suffix("")))
        self.assertTrue(result["characters_valid"])
        self.assertTrue(result["engine_character_log_evidence"]["exact_actor_count_valid"])
        self.assertTrue(result["engine_character_log_evidence"]["all_maps_observed"])


if __name__ == "__main__":
    unittest.main()
