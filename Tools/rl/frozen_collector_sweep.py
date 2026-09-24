#!/usr/bin/env python3
"""No-training collector benchmark with a frozen PPO policy.

The environment workers, observation normalization, policy forward pass, and
legal action conversion match rollout collection, but this harness never
creates an optimizer, computes a loss, updates parameters, or writes a
checkpoint. It is deliberately separate from the training entrypoint so an
optimization screen cannot accidentally become a PPO run.

The async path is a frozen-policy throughput screen only. AsyncRollout currently
folds truncations into its terminal mask and does not expose the timeout value
bootstrap used by PPO training, so its output is not a training-equivalence
result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ppo"))

from env import ACTION_DIM  # noqa: E402
from obs_schema import DEFAULT_LAYOUT  # noqa: E402
from ppo.normalize import ObservationNormalizer  # noqa: E402
from ppo.policy import ActorCritic  # noqa: E402
from async_vec_env import AsyncVecOvergrowthEnv  # noqa: E402
from throughput_sweep import collect_engine_character_logs  # noqa: E402
from vec_env import VecOvergrowthEnv  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        try:
            torch.set_num_interop_threads(args.torch_interop_threads)
        except RuntimeError:
            # PyTorch only permits this before inter-op work begins. The result
            # records the requested value even when an embedding invokes PyTorch
            # before this script gets control.
            pass

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frame_stack = int(args.frame_stack)
    layout = DEFAULT_LAYOUT
    expected_total = checkpoint.get("layout_total_floats")
    expected_stack = checkpoint.get("frame_stack")
    if expected_total != layout.total_floats or expected_stack != frame_stack:
        raise ValueError(
            f"checkpoint layout mismatch: checkpoint total={expected_total}, stack={expected_stack}; "
            f"requested total={layout.total_floats}, stack={frame_stack}"
        )
    policy = ActorCritic(layout, frame_stack=frame_stack)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    normalizer = ObservationNormalizer(layout, frame_stack=frame_stack)
    normalizer.load_state_dict(checkpoint["obs_normalizer"])

    tag = args.shm_tag or str(time.time_ns())
    scenario = {}
    if args.opponents is not None:
        scenario["opponents"] = args.opponents
    if args.difficulty is not None:
        scenario["difficulty"] = args.difficulty
    scenario_fn = (lambda values=scenario: dict(values)) if scenario else None
    env_type = AsyncVecOvergrowthEnv if args.collector == "async" else VecOvergrowthEnv
    env_kwargs = {
        "n_envs": args.workers,
        "repo_root": args.repo_root,
        "level": args.levels,
        "shm_prefix": f"/ogrl_fp_{tag}_",
        "base_seed": args.seed,
        "layout": layout,
        "frame_stack": frame_stack,
        "max_episode_steps": args.max_episode_steps,
        "act_period": args.act_period,
        "soft_reset": args.soft_reset,
        "hard_reset_every": args.hard_reset_every,
        "scenario_fn": scenario_fn,
    }
    if args.collector == "sync":
        env_kwargs["k_standby"] = args.k_standby
    else:
        env_kwargs["min_ready_batch"] = args.min_ready_batch
        env_kwargs["max_batch_wait_seconds"] = args.max_ready_wait_ms / 1000.0
    capture_engine_logs = args.capture_engine_logs
    evidence_dir = Path(args.out).with_suffix("") if capture_engine_logs else None
    if capture_engine_logs:
        if evidence_dir.exists():
            raise FileExistsError(f"scenario evidence directory already exists: {evidence_dir}")
        if args.opponents is None:
            raise ValueError("--capture-engine-logs requires an explicit --opponents value")
        evidence_dir.mkdir(parents=True)
    prior_retain_value = os.environ.get("OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS")
    if capture_engine_logs:
        os.environ["OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS"] = "1"
    try:
        vec = env_type(**env_kwargs)
    finally:
        if prior_retain_value is None:
            os.environ.pop("OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS", None)
        else:
            os.environ["OGRL_RETAIN_INITIAL_ENGINE_ARTIFACTS"] = prior_retain_value
    try:
        raw_obs = vec.reset(seeds=[args.seed + i for i in range(args.workers)])

        inference_seconds = 0.0

        def policy_action(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            nonlocal inference_seconds
            normalized = normalizer.normalize(obs, update=False)
            tensor = torch.as_tensor(normalized, dtype=torch.float32)
            started = time.perf_counter()
            with torch.inference_mode():
                actions, log_prob, _entropy, value = policy.get_action_and_value(tensor)
            inference_seconds += time.perf_counter() - started
            action_array = actions.detach().numpy().astype(np.float32, copy=False)
            expected_shape = (len(obs), ACTION_DIM)
            if action_array.shape != expected_shape:
                raise RuntimeError(f"policy returned {action_array.shape}, expected {expected_shape}")
            return normalized, action_array, log_prob.detach().numpy(), value.detach().numpy()

        ready_batches: list[int] = []
        ready_waits: list[float] = []
        transitions = 0
        policy_batches = 0

        if args.collector == "sync":
            def step_once(obs: np.ndarray) -> np.ndarray:
                _normalized, action_array, _log_prob, _value = policy_action(obs)
                next_obs, _rewards, _terminals, _truncateds, _infos = vec.step(action_array)
                return next_obs

            warmup_deadline = time.monotonic() + args.warmup_seconds
            while time.monotonic() < warmup_deadline:
                raw_obs = step_once(raw_obs)

            measured_start = time.monotonic()
            inference_seconds = 0.0
            while time.monotonic() < measured_start + args.measure_seconds:
                raw_obs = step_once(raw_obs)
                transitions += args.workers
                policy_batches += 1
        else:
            def act_fn(raw_batch: np.ndarray):
                return policy_action(raw_batch)

            warmup_deadline = time.monotonic() + args.warmup_seconds
            while time.monotonic() < warmup_deadline:
                vec.collect_rollout(args.rollout_steps, act_fn)

            measured_start = time.monotonic()
            inference_seconds = 0.0
            while time.monotonic() < measured_start + args.measure_seconds:
                rollout = vec.collect_rollout(args.rollout_steps, act_fn)
                transitions += rollout.obs.shape[0] * rollout.obs.shape[1]
                policy_batches += rollout.batches
                ready_batches.extend(rollout.ready_batch_sizes)
                ready_waits.extend(rollout.ready_wait_seconds)
        measured_seconds = time.monotonic() - measured_start
        perf = vec.drain_perf()
        result = {
            "mode": "frozen_policy_collector",
            "collector": args.collector,
            "checkpoint": str(args.checkpoint),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "workers": args.workers,
            "k_standby": args.k_standby,
            "rollout_steps": args.rollout_steps if args.collector == "async" else 1,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "frame_stack": frame_stack,
            "act_period": args.act_period,
            "levels": args.levels,
            "scenario": scenario or None,
            "soft_reset": args.soft_reset,
            "hard_reset_every": args.hard_reset_every if args.soft_reset else None,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "transitions": transitions,
            "policy_batches": policy_batches,
            "measured_seconds": measured_seconds,
            "decisions_per_second": transitions / measured_seconds if measured_seconds else 0.0,
            "policy_batches_per_second": policy_batches / measured_seconds if measured_seconds else 0.0,
            "min_ready_batch": args.min_ready_batch if args.collector == "async" else None,
            "max_batch_wait_ms": args.max_ready_wait_ms if args.collector == "async" else None,
            "mean_ready_batch": float(np.mean(ready_batches)) if ready_batches else 0.0,
            "p10_ready_batch": float(np.percentile(ready_batches, 10)) if ready_batches else 0.0,
            "p90_ready_batch": float(np.percentile(ready_batches, 90)) if ready_batches else 0.0,
            "mean_ready_wait_ms": float(np.mean(ready_waits) * 1000.0) if ready_waits else 0.0,
            "p90_ready_wait_ms": float(np.percentile(ready_waits, 90) * 1000.0) if ready_waits else 0.0,
            "policy_inference_seconds": inference_seconds,
            "policy_inference_share": inference_seconds / measured_seconds if measured_seconds else 0.0,
            "environment_seconds": max(0.0, measured_seconds - inference_seconds),
            "perf": perf,
            "optimizer_created": False,
            "checkpoint_written": False,
            "error": None,
        }
    finally:
        vec.close()

    if capture_engine_logs:
        assert evidence_dir is not None
        standby_count = args.k_standby if args.collector == "sync" else 0
        evidence = collect_engine_character_logs(
            Path(args.repo_root), f"fp_{tag}_", args.workers, standby_count,
            list(args.levels), evidence_dir, args.opponents,
        )
        expected_ids = list(range(args.opponents + 1))
        exact_actor_count = all(
            engine.get("observed_character_ids_from_notice_logs") == expected_ids
            for engine in evidence["engines"]
        )
        expected_maps = {Path(level).name.lower() for level in args.levels}
        observed_maps = {
            Path(engine["assigned_level"]).name.lower()
            for engine in evidence["engines"]
            if engine.get("assigned_level")
        }
        evidence["fixed_scenario_opponents"] = args.opponents
        evidence["exact_actor_count_valid"] = exact_actor_count
        evidence["expected_maps"] = sorted(expected_maps)
        evidence["observed_maps"] = sorted(observed_maps)
        evidence["all_maps_observed"] = expected_maps == observed_maps
        evidence["valid"] = bool(
            evidence["valid"] and exact_actor_count and expected_maps == observed_maps
        )
        result["engine_character_log_evidence"] = evidence
        result["characters_valid"] = evidence["valid"]
    else:
        result["engine_character_log_evidence"] = None
        result["characters_valid"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--k-standby", type=int, default=2)
    parser.add_argument("--collector", choices=("sync", "async"), default="sync")
    parser.add_argument("--rollout-steps", type=int, default=8,
                        help="time-major transitions per worker per async rollout")
    parser.add_argument("--min-ready-batch", type=int, default=1,
                        help="experimental async minimum ready workers per policy batch (1 keeps legacy behavior)")
    parser.add_argument("--max-ready-wait-ms", type=float, default=0.5,
                        help="maximum async cohort wait in milliseconds before scheduling ready workers")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--opponents", type=int, default=None,
                        help="fixed opponents per reset; omitted preserves the level's existing default")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="fixed scenario difficulty in [0, 1]; omitted preserves the existing default")
    parser.add_argument("--soft-reset", action="store_true",
                        help="use the engine soft reset for episode transitions")
    parser.add_argument("--hard-reset-every", type=int, default=50,
                        help="force a hard reset every Nth physical-engine episode when soft reset is enabled")
    parser.add_argument("--capture-engine-logs", action="store_true",
                        help="archive startup logs and require exact actor-count and map evidence")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measure-seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--shm-tag", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.opponents is not None and args.opponents < 1:
        parser.error("--opponents must be at least 1")
    if args.difficulty is not None and not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0, 1]")
    if args.soft_reset and args.hard_reset_every < 1:
        parser.error("--hard-reset-every must be positive when --soft-reset is enabled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:  # preserve the failure as an auditable result
        result = {"mode": "frozen_policy_collector", "error": repr(exc), "optimizer_created": False,
                  "checkpoint_written": False}
        print(f"ERROR: {exc}", flush=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 1
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("characters_valid") is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
