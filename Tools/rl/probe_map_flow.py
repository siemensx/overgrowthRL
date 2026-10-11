#!/usr/bin/env python3
"""OGRL-20261010-008: how do fights unfold on each map? User: on big maps the agent waits 10-15 s for enemies to
walk up (it never attacks first), and they arrive one by one in a line.

Per map, greedy 1v3 fights at d = 1.0: seconds until the first hostile is within 2.5 m, the gaps between the first,
second and third hostile first coming within 2.5 m, the largest number of hostiles within 3 m at once, fight length,
outcome; plus from the level XML: arena size, EnvObject count, 1v3 spawn distance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ppo"))
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from probe_plasticity import load  # noqa: E402
from watch import deterministic_action  # noqa: E402

V6_OMNI = ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
           "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"]


def level_facts(level: str) -> dict:
    x = (paths.data_dir() / "Levels" / level).read_text(errors="replace")
    sp = []
    for blk in re.findall(r"<PlaceholderObject[^>]*>.*?</PlaceholderObject>", x, re.S):
        if re.search(r'name="game_type" type="string" val="4"', blk):
            m = re.search(r't0="([-\d.]+)" t1="([-\d.]+)" t2="([-\d.]+)"', blk)
            team = re.search(r'name="team" type="string" val="(\d+)"', blk)
            if m:
                sp.append((float(m.group(1)), float(m.group(3)), int(team.group(1)) if team else -1))
    agent = [p for p in sp if p[2] == 0]
    host = [p for p in sp if p[2] == 1]
    d = [((h[0] - agent[0][0]) ** 2 + (h[1] - agent[0][1]) ** 2) ** 0.5 for h in host] if agent else []
    return {"env_objects": len(re.findall(r"<EnvObject ", x)),
            "spawn_dist_1v3": round(float(np.mean(d)), 1) if d else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--levels", required=True, help="comma list, e.g. arenas/t_train_101.xml,...")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=985000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _ck, pol, nrm = load(a.checkpoint)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    rows = []
    for level in a.levels.split(","):
        env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=level, shm_name=f"/ogmf{os.getpid() % 1000}_{a.levels.split(',').index(level)}",
                            seed=a.seed, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
        fights = []
        try:
            for ep in range(a.episodes):
                obs = env.reset(seed=a.seed + ep, opponents=3, difficulty=1.0)
                if ep == 0:
                    obs = env.reset(seed=a.seed + ep, opponents=3, difficulty=1.0)
                first_close, max_close, kos = {}, 0, 0
                for t in range(1200):
                    raw = np.asarray(env._prev_values, dtype=np.float32)
                    ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(-1, ef)
                    close = 0
                    for e in ents:
                        if e[0] > 0.5 and e[23] < 0.5:
                            if e[8] < 2.5 and int(e[1]) not in first_close:
                                first_close[int(e[1])] = t
                            close += int(e[8] < 3.0 and e[10] > 0.5)
                    max_close = max(max_close, close)
                    x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                    obs, _r, done, info = env.step(deterministic_action(pol, x))
                    kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                    if done or kos >= 3:
                        break
                arr = sorted(first_close.values())
                fights.append({"first_contact_s": round(arr[0] / 30, 1) if arr else None,
                               "arrival_gaps_s": [round((b - c) / 30, 1) for c, b in zip(arr, arr[1:])],
                               "max_close": max_close, "length_s": round((t + 1) / 30, 1), "won": kos >= 3})
        finally:
            env.close()
        fc = [f["first_contact_s"] for f in fights if f["first_contact_s"] is not None]
        gaps = [g for f in fights for g in f["arrival_gaps_s"]]
        row = {"level": level, **level_facts(level), "fights": len(fights), "won": sum(f["won"] for f in fights),
               "first_contact_s": round(float(np.mean(fc)), 1) if fc else None,
               "arrival_gap_s": round(float(np.mean(gaps)), 1) if gaps else None,
               "max_close_mean": round(float(np.mean([f["max_close"] for f in fights])), 2),
               "length_s": round(float(np.mean([f["length_s"] for f in fights])), 1)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
