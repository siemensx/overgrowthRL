#!/usr/bin/env python3
"""OGRL-20260816-023: re-sweep (n_envs, k_standby) jointly, post action-repeat
(Stage 6, --rl-act-period) and off-critical-path reset (vec_env.py's standby
pool). OGRL-20260816-021 measured the OLD sweep (n_envs alone, synchronous
674ms reset, 120Hz decisions) as flat-to-inverted -- 2 workers beat 8 -- which
is the textbook signature of a barrier-synchronized bottleneck, not a real
saturation point. Both of that sweep's root causes are fixed now, so this
sweep exists to find the actual optimum, not assume the old default (8) still
applies.

Explicitly two-dimensional per the user's own instruction: k_standby is not a
minor knob, it changes what "more workers" even costs (an idle worker slot vs
an extra pre-warmed process competing for the same cores), so it has to be
swept together with n_envs, not fixed and forgotten.

Random actions, no policy forward pass -- this measures the environment/
transport/reset throughput ceiling, matching OGRL-20260816-021 Sec 1.1's own
methodology, not end-to-end training speed (the learner was independently
measured there at ~3% of per-update cost, i.e. not the bottleneck).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vec_env import VecOvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, ObsLayout
from env import ACTION_DIM


def run_one_point(repo_root: str, level: str, n_envs: int, k_standby: int, act_period: int,
                   max_episode_steps: int, warmup_seconds: float, measure_seconds: float,
                   shm_tag: str, layout: ObsLayout) -> dict:
    launch_start = time.monotonic()
    vec = VecOvergrowthEnv(
        n_envs=n_envs, repo_root=repo_root, level=level, shm_prefix=f"/ogrl_sw{shm_tag}_",
        base_seed=20260817 + n_envs * 1000 + k_standby, layout=layout,
        frame_stack=1, max_episode_steps=max_episode_steps, k_standby=k_standby, act_period=act_period,
    )
    launch_seconds = time.monotonic() - launch_start
    try:
        vec.reset(seeds=[20260817 + i for i in range(n_envs)])

        def _burst(seconds: float) -> tuple[int, int]:
            steps_done = 0
            episode_ends = 0
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                actions = np.random.uniform(-1.0, 1.0, size=(n_envs, ACTION_DIM)).astype(np.float32)
                actions[:, 2:] = (actions[:, 2:] > 0.0).astype(np.float32)
                _, _, terminals, truncateds, _ = vec.step(actions)
                steps_done += n_envs
                episode_ends += int(np.sum(terminals | truncateds))
            return steps_done, episode_ends

        _burst(warmup_seconds)  # let engines settle into steady state before measuring
        measure_start = time.monotonic()
        steps_done, episode_ends = _burst(measure_seconds)
        measured_seconds = time.monotonic() - measure_start

        return {
            "n_envs": n_envs, "k_standby": k_standby, "act_period": act_period,
            "launch_seconds": launch_seconds,
            "steps": steps_done, "measured_seconds": measured_seconds,
            "decisions_per_second": steps_done / measured_seconds if measured_seconds > 0 else 0.0,
            "decisions_per_second_per_worker": (steps_done / measured_seconds / n_envs) if measured_seconds > 0 else 0.0,
            "episode_ends_during_measurement": episode_ends,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- a sweep point failing must not kill the whole sweep
        return {
            "n_envs": n_envs, "k_standby": k_standby, "act_period": act_period,
            "launch_seconds": launch_seconds, "steps": 0, "measured_seconds": 0.0,
            "decisions_per_second": 0.0, "decisions_per_second_per_worker": 0.0,
            "episode_ends_during_measurement": 0, "error": str(exc),
        }
    finally:
        vec.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--level", default="arenas/oval_arena_1v1_unarmed.xml")
    p.add_argument("--n-envs-grid", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--k-standby-grid", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--act-period", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=900)
    p.add_argument("--warmup-seconds", type=float, default=3.0)
    p.add_argument("--measure-seconds", type=float, default=15.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    layout = DEFAULT_LAYOUT
    out_path = Path(args.out) if args.out else Path(args.repo_root) / "Tools/rl/runs" / f"concurrency_sweep_{int(time.time())}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_envs", "k_standby", "act_period", "launch_seconds", "steps", "measured_seconds",
                          "decisions_per_second", "decisions_per_second_per_worker",
                          "episode_ends_during_measurement", "error"])
        for n_envs in args.n_envs_grid:
            for k_standby in args.k_standby_grid:
                shm_tag = f"{n_envs}k{k_standby}"
                print(f"=== n_envs={n_envs} k_standby={k_standby} act_period={args.act_period} ===", flush=True)
                result = run_one_point(
                    args.repo_root, args.level, n_envs, k_standby, args.act_period,
                    args.max_episode_steps, args.warmup_seconds, args.measure_seconds, shm_tag, layout,
                )
                results.append(result)
                writer.writerow([result[k] for k in [
                    "n_envs", "k_standby", "act_period", "launch_seconds", "steps", "measured_seconds",
                    "decisions_per_second", "decisions_per_second_per_worker",
                    "episode_ends_during_measurement", "error",
                ]])
                f.flush()
                if result["error"]:
                    print(f"  ERROR: {result['error']}", flush=True)
                else:
                    print(f"  {result['decisions_per_second']:.1f} decisions/s aggregate "
                          f"({result['decisions_per_second_per_worker']:.1f}/worker), "
                          f"launch={result['launch_seconds']:.1f}s, "
                          f"{result['episode_ends_during_measurement']} episode ends", flush=True)

    best = max((r for r in results if not r["error"]), key=lambda r: r["decisions_per_second"], default=None)
    print(f"\nfull results: {out_path}")
    if best:
        print(f"best: n_envs={best['n_envs']} k_standby={best['k_standby']} -> "
              f"{best['decisions_per_second']:.1f} decisions/s aggregate")
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"results": results, "best": best}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
