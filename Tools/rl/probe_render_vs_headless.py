#!/usr/bin/env python3
"""OGRL-20261007-015: does a checkpoint fight the same when the game is DRAWN (watch.py) as when it is headless
(training, benchmark)? The user watched run38_d1_wd@90M win 1 of 10 rendered 1v3 fights on t_train_101 while
the headless periodic bench on that map read 109/200.

Plays the same seeds greedily in both modes, with the training cap (1200 decisions) and no wall-clock cap, and
reports per-seed outcomes. Rendered mode is watch.py's configuration exactly (render=True, time_scale_mult=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ppo"))
sys.path.insert(0, str(HERE))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from probe_plasticity import load  # noqa: E402
from watch import deterministic_action  # noqa: E402

V6_OMNI = ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
           "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"]


def play(ck_path, level, opponents, seeds, render: bool) -> list:
    _ck, pol, nrm = load(ck_path)
    kw = dict(repo_root=str(HERE.parents[1]), level=level, shm_name=f"/ogrv{os.getpid() % 10000}{int(render)}",
              seed=seeds[0], act_period=4, frame_stack=4, extra_config_lines=V6_OMNI, render=render)
    if render:
        kw["time_scale_mult"] = 1
    env = OvergrowthEnv(**kw)
    out = []
    try:
        first = True
        for seed in seeds:
            obs = env.reset(seed=seed, opponents=opponents, difficulty=1.0)
            if first:
                obs = env.reset(seed=seed, opponents=opponents, difficulty=1.0)
                first = False
            kos, won, done, t0 = 0, False, False, time.time()
            for step in range(1200):
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                obs, _r, done, info = env.step(deterministic_action(pol, x))
                kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                won = kos >= opponents
                if done or won:
                    break
            res = {"seed": seed, "outcome": "won" if won else ("lost" if done else "timeout"), "steps": step + 1,
                   "wall_s": round(time.time() - t0, 1)}
            print(("RENDER " if render else "HEADLESS ") + json.dumps(res), flush=True)
            out.append(res)
    finally:
        env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=951000)
    ap.add_argument("--modes", default="headless,render")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = list(range(a.seed_base, a.seed_base + a.episodes))
    res = {}
    for mode in a.modes.split(","):
        res[mode] = play(a.checkpoint, a.level, a.opponents, seeds, render=(mode == "render"))
    summary = {m: {k: sum(r["outcome"] == k for r in v) for k in ("won", "lost", "timeout")} for m, v in res.items()}
    if "headless" in res and "render" in res:
        summary["same_outcome"] = sum(x["outcome"] == y["outcome"] for x, y in zip(res["headless"], res["render"]))
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "level": a.level, "opponents": a.opponents,
                                       "seeds": seeds, "summary": summary, "fights": res}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
