#!/usr/bin/env python3
"""No-training collector benchmark with a frozen PPO policy.

The environment workers, observation normalization, policy forward pass, and
legal action conversion match rollout collection, but this harness never
creates an optimizer, computes a loss, updates parameters, or writes a
checkpoint. It is deliberately separate from the training entrypoint so an
optimization screen cannot accidentally become a PPO run.
"""

from __future__ import annotations

import argparse
import json
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

    tag = args.shm_tag or str(int(time.time()))
    vec = VecOvergrowthEnv(
        n_envs=args.workers,
        repo_root=args.repo_root,
        level=args.levels,
        shm_prefix=f"/ogrl_fp_{tag}_",
        base_seed=args.seed,
        layout=layout,
        frame_stack=frame_stack,
        max_episode_steps=args.max_episode_steps,
        k_standby=args.k_standby,
        act_period=args.act_period,
    )
    try:
        raw_obs = vec.reset(seeds=[args.seed + i for i in range(args.workers)])

        def step_once(obs: np.ndarray) -> tuple[np.ndarray, float]:
            normalized = normalizer.normalize(obs, update=False)
            tensor = torch.as_tensor(normalized, dtype=torch.float32)
            started = time.perf_counter()
            with torch.inference_mode():
                actions, _log_prob, _entropy, _value = policy.get_action_and_value(tensor)
            inference_seconds = time.perf_counter() - started
            action_array = actions.detach().numpy().astype(np.float32, copy=False)
            if action_array.shape != (args.workers, ACTION_DIM):
                raise RuntimeError(f"policy returned {action_array.shape}, expected {(args.workers, ACTION_DIM)}")
            next_obs, _rewards, _terminals, _truncateds, _infos = vec.step(action_array)
            return next_obs, inference_seconds

        warmup_deadline = time.monotonic() + args.warmup_seconds
        while time.monotonic() < warmup_deadline:
            raw_obs, _ = step_once(raw_obs)

        transitions = 0
        policy_batches = 0
        inference_seconds = 0.0
        measured_start = time.monotonic()
        while time.monotonic() < measured_start + args.measure_seconds:
            raw_obs, inference_time = step_once(raw_obs)
            transitions += args.workers
            policy_batches += 1
            inference_seconds += inference_time
        measured_seconds = time.monotonic() - measured_start
        perf = vec.drain_perf()
        return {
            "mode": "frozen_policy_collector",
            "checkpoint": str(args.checkpoint),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "workers": args.workers,
            "k_standby": args.k_standby,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "frame_stack": frame_stack,
            "act_period": args.act_period,
            "levels": args.levels,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "transitions": transitions,
            "policy_batches": policy_batches,
            "measured_seconds": measured_seconds,
            "decisions_per_second": transitions / measured_seconds if measured_seconds else 0.0,
            "policy_batches_per_second": policy_batches / measured_seconds if measured_seconds else 0.0,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--k-standby", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measure-seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--shm-tag", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
