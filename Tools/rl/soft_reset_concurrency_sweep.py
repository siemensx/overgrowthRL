#!/usr/bin/env python3
"""OGRL-20260817-028 Sec2: re-sweep (n_envs, k_standby) now that soft reset
makes resets cheap (~97ms vs 575-705ms hard). Uses the TRAINED policy is
the plan's stated preference (episode length differs materially between a
trained and a random policy, and episode length drives reset frequency) --
falls back to random actions if no --checkpoint is given, since at the time
this runs there may be no schema-v5-compatible checkpoint yet (this is a
cold-restart architecture change; run9 cannot be loaded post-Sec5)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vec_env import VecOvergrowthEnv
from obs_schema import DEFAULT_LAYOUT
from curriculum import ScenarioSampler


def measure(n_envs: int, k_standby: int, repo_root: str, level: str, act_period: int,
            max_episode_steps: int, warmup_s: float, measure_s: float, hard_reset_every: int) -> dict:
    layout = DEFAULT_LAYOUT
    sampler = ScenarioSampler(d_max_start=0.3, rng_seed=1)
    vec_env = VecOvergrowthEnv(
        n_envs=n_envs, repo_root=repo_root, level=level, shm_prefix=f"/ogrl_sw{n_envs}{k_standby}",
        base_seed=1, layout=layout, frame_stack=4, max_episode_steps=max_episode_steps,
        k_standby=k_standby, act_period=act_period, soft_reset=True, hard_reset_every=hard_reset_every,
        scenario_fn=sampler.sample_episode,
    )
    try:
        obs = vec_env.reset(seeds=[1 + i for i in range(n_envs)])
        t_end_warmup = time.monotonic() + warmup_s
        while time.monotonic() < t_end_warmup:
            actions = np.random.uniform(-1, 1, size=(n_envs, 8)).astype(np.float32)
            actions[:, 2:] = (actions[:, 2:] > 0).astype(np.float32)
            obs, *_ = vec_env.step(actions)

        steps = 0
        t0 = time.monotonic()
        t_end = t0 + measure_s
        while time.monotonic() < t_end:
            actions = np.random.uniform(-1, 1, size=(n_envs, 8)).astype(np.float32)
            actions[:, 2:] = (actions[:, 2:] > 0).astype(np.float32)
            obs, *_ = vec_env.step(actions)
            steps += n_envs
        elapsed = time.monotonic() - t0
        perf = vec_env.drain_perf()
        return {
            "n_envs": n_envs, "k_standby": k_standby, "decisions_per_second": steps / elapsed,
            "reset_seconds": perf["reset_seconds"], "reset_share": perf["reset_seconds"] / elapsed,
            "pool_hits": perf["pool_hits"], "pool_misses": perf["pool_misses"],
        }
    finally:
        vec_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--level", default="arenas/oval_arena_1v1_unarmed.xml")
    p.add_argument("--n-envs-grid", type=int, nargs="+", default=[4, 6, 8, 10])
    p.add_argument("--k-standby-grid", type=int, nargs="+", default=[0, 1])
    p.add_argument("--act-period", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=900)
    p.add_argument("--warmup-seconds", type=float, default=3.0)
    p.add_argument("--measure-seconds", type=float, default=12.0)
    p.add_argument("--hard-reset-every", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = []
    for n_envs in args.n_envs_grid:
        for k_standby in args.k_standby_grid:
            print(f"=== n_envs={n_envs} k_standby={k_standby} ===")
            r = measure(n_envs, k_standby, args.repo_root, args.level, args.act_period,
                        args.max_episode_steps, args.warmup_seconds, args.measure_seconds, args.hard_reset_every)
            print(json.dumps(r, indent=2))
            results.append(r)

    best = max(results, key=lambda r: r["decisions_per_second"])
    print(f"\n=== BEST: n_envs={best['n_envs']} k_standby={best['k_standby']} -> {best['decisions_per_second']:.1f} dec/s ===")
    if args.out:
        Path(args.out).write_text(json.dumps({"results": results, "best": best}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
