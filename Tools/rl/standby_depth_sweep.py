#!/usr/bin/env python3
"""OGRL-20260816-024: focused follow-up to concurrency_sweep.py -- that sweep
only measured aggregate decisions/s, which left the ACTUAL mechanism (pool
underrun forcing a fallback to the full ~975ms synchronous reset) invisible.
This one instruments VecOvergrowthEnv._take_standby() directly to report the
pool hit/miss rate alongside throughput, at a fixed n_envs=4 (already
established as the throughput-optimal worker count) across a range of
k_standby depths, with realistic random actions (not synchronized all-zero
timeouts, which understated how bursty real episode-ends are).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vec_env import VecOvergrowthEnv, ACTION_DIM
from obs_schema import DEFAULT_LAYOUT


def run_point(n_envs: int, k_standby: int, measure_seconds: float) -> dict:
    layout = DEFAULT_LAYOUT
    vec = VecOvergrowthEnv(
        n_envs=n_envs, repo_root=".", level="arenas/oval_arena_1v1_unarmed.xml",
        shm_prefix=f"/ogrl_sd{k_standby}_", base_seed=700 + k_standby * 10, layout=layout,
        max_episode_steps=900, k_standby=k_standby, act_period=4,
    )
    vec.reset(seeds=[700 + k_standby * 10 + i for i in range(n_envs)])

    orig_take = vec._take_standby
    hits, misses = [0], [0]

    def timed_take():
        result = orig_take()
        if result is not None:
            hits[0] += 1
        else:
            misses[0] += 1
        return result
    vec._take_standby = timed_take

    # Warmup, not measured -- lets the pool reach steady state.
    for _ in range(30):
        actions = np.random.uniform(-1, 1, size=(n_envs, ACTION_DIM)).astype(np.float32)
        actions[:, 2:] = (actions[:, 2:] > 0).astype(np.float32)
        vec.step(actions)
    hits[0] = misses[0] = 0

    steps = 0
    deadline = time.monotonic() + measure_seconds
    start = time.monotonic()
    while time.monotonic() < deadline:
        actions = np.random.uniform(-1, 1, size=(n_envs, ACTION_DIM)).astype(np.float32)
        actions[:, 2:] = (actions[:, 2:] > 0).astype(np.float32)
        vec.step(actions)
        steps += n_envs
    elapsed = time.monotonic() - start
    vec.close()

    total_events = hits[0] + misses[0]
    return {
        "n_envs": n_envs, "k_standby": k_standby,
        "decisions_per_second": steps / elapsed,
        "pool_hits": hits[0], "pool_misses": misses[0],
        "miss_rate_pct": (misses[0] / total_events * 100) if total_events else 0.0,
    }


def main():
    n_envs = 4
    for k_standby in [2, 3, 4, 5, 6, 8]:
        r = run_point(n_envs, k_standby, measure_seconds=12.0)
        print(f"n_envs={r['n_envs']} k_standby={r['k_standby']}: "
              f"{r['decisions_per_second']:.1f} decisions/s | "
              f"pool {r['pool_hits']} hits / {r['pool_misses']} misses ({r['miss_rate_pct']:.0f}% underrun)",
              flush=True)


if __name__ == "__main__":
    main()
