#!/usr/bin/env python3
"""Sync vs async collector, on the SAME heterogeneous worker set.

The synchronous vector environment advances every worker in lockstep, so each
policy step costs as much as the SLOWEST worker's step. That was cheap when all
workers ran the same map with the same opponent count. It is not any more: the
corpus now spans 19-417 EnvObjects and the opponent curriculum mixes 1v1 with
1v3, so a sparse 1v1 worker finishes and then waits on a dense 1v3 one, every
step. run18 telemetry showed barrier_idle_seconds at 1.32s of a 2.29s cycle --
58% of wall time spent idle at the barrier.

async_vec_env keeps one step in flight per engine and schedules the next action
for whichever engine is ready. It is still on-policy PPO: one frozen policy for
the whole rollout, exactly n_steps per worker. Only wall-clock scheduling
changes. It has never been measured under HETEROGENEOUS workers, which is the
condition that should favour it -- the 2026-08-20 sweep used one level for every
worker and left "does the advantage survive a properly powered sweep" open.

Reports decisions/s for both, on identical levels, opponent mix and step count.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import ACTION_DIM  # noqa: E402
from vec_env import VecOvergrowthEnv  # noqa: E402
from async_vec_env import AsyncVecOvergrowthEnv  # noqa: E402


def _act(batch):
    n = len(batch)
    a = np.zeros((n, ACTION_DIM), dtype=np.float32)
    return batch, a, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)


def run_sync(levels, n_envs, k_standby, n_steps, seed, prefix, scenario_fn):
    env = VecOvergrowthEnv(n_envs=n_envs, repo_root=str(HERE.parent.parent), level=levels,
                           shm_prefix=prefix, base_seed=seed, frame_stack=1, act_period=4,
                           k_standby=k_standby, soft_reset=True, scenario_fn=scenario_fn)
    try:
        env.reset()
        t0 = time.monotonic()
        for _ in range(n_steps):
            env.step(np.zeros((n_envs, ACTION_DIM), dtype=np.float32))
        dt = time.monotonic() - t0
        perf = env.drain_perf()
    finally:
        env.close()
    return (n_steps * n_envs) / dt, dt, perf


def run_async(levels, n_envs, k_standby, n_steps, seed, prefix, scenario_fn):
    env = AsyncVecOvergrowthEnv(n_envs=n_envs, repo_root=str(HERE.parent.parent), level=levels,
                                shm_prefix=prefix, base_seed=seed, frame_stack=1, act_period=4,
                                soft_reset=True, scenario_fn=scenario_fn)
    try:
        env.reset()
        t0 = time.monotonic()
        env.collect_rollout(n_steps, _act)
        dt = time.monotonic() - t0
        perf = env.drain_perf()
    finally:
        env.close()
    return (n_steps * n_envs) / dt, dt, perf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--k-standby", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--opponents", nargs="+", type=int, default=[1],
                    help="opponent counts to sample per episode; pass several to make workers "
                         "heterogeneous in COST as well as in map")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rng = random.Random(0)
    def scenario_fn():
        return {"difficulty": 0.5, "opponents": rng.choice(a.opponents),
                "weapons": 0.0, "species": 0}

    print(f"levels={len(a.levels)}  n_envs={a.n_envs}  opponents={a.opponents}  "
          f"n_steps={a.n_steps}  repeats={a.repeats}")
    rows = []
    for r in range(a.repeats):
        for name, fn in (("sync", run_sync), ("async", run_async)):
            try:
                rate, dt, perf = fn(a.levels, a.n_envs, a.k_standby, a.n_steps,
                                    1000 + r * 17, f"/ogrl_ab{r}{name[0]}", scenario_fn)
            except Exception as exc:
                print(f"  r{r} {name:<6} FAILED: {type(exc).__name__}: {exc}")
                continue
            idle = perf.get("barrier_idle_seconds")
            rows.append({"repeat": r, "collector": name, "decisions_per_s": rate})
            print(f"  r{r} {name:<6} {rate:8.1f} decisions/s   wall {dt:6.2f}s"
                  + (f"   barrier_idle {idle:.2f}s" if isinstance(idle, (int, float)) else ""))
    for name in ("sync", "async"):
        v = sorted(x["decisions_per_s"] for x in rows if x["collector"] == name)
        if v:
            print(f"\n  {name:<6} median {v[len(v)//2]:8.1f} decisions/s   ({len(v)} samples)")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
