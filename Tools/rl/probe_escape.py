#!/usr/bin/env python3
"""OGRL-20261010-017: how often does the agent leave the map (over the boundary wall), and what does the fight then
count as? Per fight: did the agent get outside the walls or more than 4 m below its start height, when, and the
outcome. Greedy policy; hop-aware enemies optional (the user saw escapes mostly against them)."""
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


def wall_half(level: str) -> float:
    x = (paths.data_dir() / "Levels" / level).read_text(errors="replace")
    best = 0.0
    for m in re.finditer(r'<EnvObject t0="([-\d.]+)" t1="[-\d.]+" t2="([-\d.]+)" s0="([-\d.]+)" s1="[-\d.]+" s2="([-\d.]+)"', x):
        t0, t2, s0, s2 = map(float, m.groups())
        if min(s0, s2) <= 0.51 and max(s0, s2) > 8:
            best = max(best, abs(t0) if s0 <= 0.51 else abs(t2))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--hop-aware", action="store_true")
    ap.add_argument("--seed", type=int, default=987000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _ck, pol, nrm = load(a.checkpoint)
    rows = []
    for li, level in enumerate(a.levels.split(",")):
        half = wall_half(level)
        env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=level, shm_name=f"/ogesc{os.getpid() % 1000}_{li}",
                            seed=a.seed, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
        try:
            for ep in range(a.episodes):
                kw = dict(seed=a.seed + ep, opponents=a.opponents, difficulty=1.0, hop_aware=a.hop_aware)
                obs = env.reset(**kw)
                if ep == 0:
                    obs = env.reset(**kw)
                y0 = float(np.asarray(env._prev_values)[L.POS][1])
                esc_t, kos, outcome = None, 0, "timeout"
                for t in range(1200):
                    p = np.asarray(env._prev_values)[L.POS]
                    if esc_t is None and (max(abs(p[0]), abs(p[2])) > half + 0.6 or p[1] < y0 - 4.0):
                        esc_t = t
                    x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                    obs, _r, done, info = env.step(deterministic_action(pol, x))
                    kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                    if kos >= a.opponents:
                        outcome = "won"
                        break
                    if done:
                        outcome = "won" if info.get("won") else ("timeout" if info.get("timed_out") else "lost")
                        break
                rows.append({"level": level, "ep": ep, "escaped_at_s": None if esc_t is None else round(esc_t / 30, 1),
                             "end_s": round((t + 1) / 30, 1), "outcome": outcome})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
    esc = [r for r in rows if r["escaped_at_s"] is not None]
    summary = {"checkpoint": Path(a.checkpoint).name, "hop_aware": a.hop_aware, "fights": len(rows),
               "won": sum(r["outcome"] == "won" for r in rows), "escaped": len(esc),
               "escaped_outcomes": {k: sum(r["outcome"] == k for r in esc) for k in ("won", "lost", "timeout")},
               "seconds_from_escape_to_end": [round(r["end_s"] - r["escaped_at_s"], 1) for r in esc]}
    Path(a.out).write_text(json.dumps({"summary": summary, "fights": rows}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
