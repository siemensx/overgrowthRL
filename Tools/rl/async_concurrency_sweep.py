#!/usr/bin/env python3
"""Benchmark the independent-worker frozen-policy rollout collector.

This measures the same environment/transport workload as concurrency_sweep.py,
but the policy decision barrier is removed: each worker contributes exactly
``rollout_steps`` transitions while ready workers are rescheduled immediately.
It is intentionally a separate harness so the synchronous reference remains
available for paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from async_vec_env import AsyncVecOvergrowthEnv
from obs_schema import DEFAULT_LAYOUT
from env import ACTION_DIM


def run_point(
    repo_root: str,
    level: str,
    n_envs: int,
    act_period: int,
    max_episode_steps: int,
    warmup_seconds: float,
    measure_seconds: float,
    rollout_steps: int,
    seed: int,
    shm_tag: str,
) -> dict:
    launch_start = time.monotonic()
    vec = AsyncVecOvergrowthEnv(
        n_envs=n_envs,
        repo_root=repo_root,
        level=level,
        shm_prefix=f"/oga{shm_tag}",
        base_seed=seed,
        layout=DEFAULT_LAYOUT,
        frame_stack=1,
        max_episode_steps=max_episode_steps,
        act_period=act_period,
    )
    launch_seconds = time.monotonic() - launch_start
    rng = np.random.default_rng(seed + 991)

    def act_fn(raw_batch: np.ndarray):
        count = raw_batch.shape[0]
        actions = rng.uniform(-1.0, 1.0, size=(count, ACTION_DIM)).astype(np.float32)
        actions[:, 2:] = (actions[:, 2:] > 0.0).astype(np.float32)
        return raw_batch, actions, np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32)

    try:
        vec.reset(seeds=[seed + i for i in range(n_envs)])
        warmup_deadline = time.monotonic() + warmup_seconds
        while time.monotonic() < warmup_deadline:
            vec.collect_rollout(rollout_steps, act_fn)

        measured_start = time.monotonic()
        transitions = 0
        rollouts = 0
        episode_ends = 0
        ready_batches = []
        while time.monotonic() < measured_start + measure_seconds:
            rollout = vec.collect_rollout(rollout_steps, act_fn)
            transitions += rollout.obs.shape[0] * rollout.obs.shape[1]
            rollouts += 1
            episode_ends += int(rollout.terminals.sum())
            ready_batches.extend(rollout.ready_batch_sizes)
        measured_seconds = time.monotonic() - measured_start
        perf = vec.drain_perf()
        return {
            "mode": "async",
            "n_envs": n_envs,
            "act_period": act_period,
            "rollout_steps": rollout_steps,
            "launch_seconds": launch_seconds,
            "transitions": transitions,
            "rollouts": rollouts,
            "measured_seconds": measured_seconds,
            "decisions_per_second": transitions / measured_seconds if measured_seconds else 0.0,
            "decisions_per_second_per_worker": transitions / measured_seconds / n_envs if measured_seconds else 0.0,
            "episode_ends": episode_ends,
            "mean_ready_batch": float(np.mean(ready_batches)) if ready_batches else 0.0,
            "p95_ready_batch": float(np.percentile(ready_batches, 95)) if ready_batches else 0.0,
            "reset_cpu_seconds": perf["reset_seconds"],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- preserve failed sweep points
        return {
            "mode": "async", "n_envs": n_envs, "act_period": act_period,
            "rollout_steps": rollout_steps, "launch_seconds": launch_seconds,
            "transitions": 0, "rollouts": 0, "measured_seconds": 0.0,
            "decisions_per_second": 0.0, "decisions_per_second_per_worker": 0.0,
            "episode_ends": 0, "mean_ready_batch": 0.0, "p95_ready_batch": 0.0,
            "reset_cpu_seconds": 0.0, "error": repr(exc),
        }
    finally:
        vec.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--level", default="arenas/oval_arena_1v1_unarmed.xml")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--measure-seconds", type=float, default=15.0)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for n_envs in args.workers:
        for repetition in range(1, args.repeats + 1):
            print(f"=== async workers={n_envs} repeat={repetition}/{args.repeats} ===", flush=True)
            result = run_point(
                args.repo_root, args.level, n_envs, args.act_period,
                args.max_episode_steps, args.warmup_seconds, args.measure_seconds,
                args.rollout_steps, 20260820 + n_envs * 100 + repetition, f"{n_envs}r{repetition}",
            )
            result["repetition"] = repetition
            results.append(result)
            if result["error"]:
                print(f"  ERROR {result['error']}", flush=True)
            else:
                print(f"  {result['decisions_per_second']:.1f} decisions/s, "
                      f"{result['episode_ends']} episode ends, "
                      f"ready batch mean={result['mean_ready_batch']:.2f}", flush=True)

    fields = list(results[0]) if results else []
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    valid = [r for r in results if not r["error"]]
    by_worker = {}
    for n_envs in args.workers:
        points = [r for r in valid if r["n_envs"] == n_envs]
        by_worker[str(n_envs)] = {
            "repetitions": len(points),
            "median_decisions_per_second": float(np.median([r["decisions_per_second"] for r in points])) if points else 0.0,
            "min_decisions_per_second": float(np.min([r["decisions_per_second"] for r in points])) if points else 0.0,
            "max_decisions_per_second": float(np.max([r["decisions_per_second"] for r in points])) if points else 0.0,
        }
    best = max(valid, key=lambda r: r["decisions_per_second"], default=None)
    summary = {"mode": "async", "results": results, "by_worker": by_worker, "best_point": best}
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"full results: {out_path}")
    print(f"summary: {summary_path}")
    if best:
        print(f"best: workers={best['n_envs']} -> {best['decisions_per_second']:.1f} decisions/s")
    return 0 if len(valid) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

