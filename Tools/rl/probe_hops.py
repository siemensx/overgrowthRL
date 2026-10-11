#!/usr/bin/env python3
"""OGRL-20261010-003: the "light jump" habit. Does the agent hop (take off without attacking) to make the
scripted AI react, and what does the AI do while it is in the air?

enemycontrol.as switches an enemy to sub-goal _avoid_jump_kick whenever its target is airborne (any hop, not only
a jump kick) and, with probability ~ Block Skill, rolls/strafes away (`will_avoid_jump_kick`). That costs the enemy
its attack plan. A human does not react to hops at all.

Per checkpoint, greedy fights on one map: every take-off is classified as a HOP (no attack pressed while airborne)
or a JUMP ATTACK; reported: take-offs per minute of game time, hop share, airborne time per hop, and for enemies
within 6 m: share in _avoid_jump_kick and share with ai_attacking, while the agent is airborne vs grounded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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
AVOID_JUMP_KICK = 38 + 1          # entity sub-goal one-hot starts at 38; _avoid_jump_kick is index 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_held_203.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=978000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck, pol, nrm = load(a.checkpoint)
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/oghp{os.getpid() % 10000}",
                        seed=a.seed, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    decisions, takeoffs, hops, air_attacks, hop_air = 0, 0, 0, 0, []
    near = {"air": [0, 0, 0], "ground": [0, 0, 0]}          # [enemy-decisions, in avoid_jump_kick, ai_attacking]
    wins = 0
    try:
        for ep in range(a.episodes):
            obs = env.reset(seed=a.seed + ep, opponents=a.opponents, difficulty=1.0)
            if ep == 0:
                obs = env.reset(seed=a.seed + ep, opponents=a.opponents, difficulty=1.0)
            airborne_prev, attacked_in_air, air_len, kos = False, False, 0, 0
            for _ in range(1200):
                raw = np.asarray(env._prev_values, dtype=np.float32)
                airborne = raw[L.GROUNDED] <= 0.5
                ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(-1, ef)
                for e in ents:
                    if e[0] > 0.5 and e[23] < 0.5 and e[8] < 6.0:
                        k = "air" if airborne else "ground"
                        near[k][0] += 1
                        near[k][1] += int(e[AVOID_JUMP_KICK] > 0.5)
                        near[k][2] += int(e[34] > 0.5)
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                act = deterministic_action(pol, x)
                if airborne and act[4] > 0.5:
                    attacked_in_air = True
                if airborne and not airborne_prev:
                    takeoffs += 1
                    attacked_in_air, air_len = act[4] > 0.5, 0
                if airborne:
                    air_len += 1
                if airborne_prev and not airborne:              # landed: classify the take-off
                    if attacked_in_air:
                        air_attacks += 1
                    else:
                        hops += 1
                        hop_air.append(air_len)
                airborne_prev = airborne
                obs, _r, done, info = env.step(act)
                decisions += 1
                kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                if done or kos >= a.opponents:
                    break
            wins += int(kos >= a.opponents)
    finally:
        env.close()
    minutes = decisions / 30 / 60
    out = {"checkpoint": Path(a.checkpoint).name, "global_step": int(ck.get("global_step", -1)), "level": a.level,
           "opponents": a.opponents, "episodes": a.episodes, "wins": wins, "decisions": decisions,
           "takeoffs_per_min": round(takeoffs / minutes, 1), "hops_per_min": round(hops / minutes, 1),
           "jump_attacks_per_min": round(air_attacks / minutes, 1),
           "hop_share": round(hops / max(1, hops + air_attacks), 3),
           "hop_airtime_s_median": round(float(np.median(hop_air)) / 30, 2) if hop_air else None,
           "near_enemy_avoid_jk_share": {k: round(v[1] / max(1, v[0]), 3) for k, v in near.items()},
           "near_enemy_attacking_share": {k: round(v[2] / max(1, v[0]), 3) for k, v in near.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
