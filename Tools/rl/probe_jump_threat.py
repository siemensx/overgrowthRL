#!/usr/bin/env python3
"""OGRL-20261010-005: what separates a real jump kick from a harmless hop, at the moment an enemy must decide?

Every take-off of the agent is recorded with what an enemy could observe at take-off and over the next ticks:
distance to the nearest hostile, closing speed toward it, the agent's upward speed, plus what happened: did the
agent press attack in the air (and how many decisions after take-off), did a hostile get hit within the airborne
window (+0.5 s), and when did the nearest hostile enter _avoid_jump_kick. Used to design a "hop-aware" enemy
trigger that still leaves time to dodge real jump kicks.

--hop-aware runs the same fights against hop-aware enemies (species word bit 256) for the validation pass.
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
AVOID = 39            # entity field: sub-goal one-hot index 1 = _avoid_jump_kick
HIT_STATES = (16, 17)  # entity state one-hot: hit_reaction, ragdoll


def hostiles(raw, ef):
    ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(-1, ef)
    return [e for e in ents if e[0] > 0.5 and e[23] < 0.5 and e[10] > 0.5]     # valid, not ally, awake


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_held_203.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=981000)
    ap.add_argument("--hop-aware", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck, pol, nrm = load(a.checkpoint)
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogjt{os.getpid() % 10000}",
                        seed=a.seed, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    jumps, wins = [], 0
    try:
        for ep in range(a.episodes):
            kw = dict(seed=a.seed + ep, opponents=a.opponents, difficulty=1.0, hop_aware=a.hop_aware)
            obs = env.reset(**kw)
            if ep == 0:
                obs = env.reset(**kw)
            cur, kos, landed_at = None, 0, None
            for t in range(1200):
                raw = np.asarray(env._prev_values, dtype=np.float32)
                airborne = raw[L.GROUNDED] <= 0.5
                hs = hostiles(raw, ef)
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                act = deterministic_action(pol, x)
                if airborne and cur is None:                          # take-off
                    near = min(hs, key=lambda e: e[8]) if hs else None
                    if near is not None:
                        rel = near[2:5]
                        rv = near[5:8]
                        horiz = np.array([rel[0], rel[2]])
                        dist_h = float(np.linalg.norm(horiz)) or 1e-6
                        closing = float(np.dot(np.array([rv[0], rv[2]]), horiz) / dist_h)  # +: hostile approaching me
                        # the jumper's own velocity: L.VEL is WORLD frame, entity rel_pos is egocentric (x = right,
                        # z = forward, right = (-fz, 0, fx) as in rl_observation.cpp MakeSelfFrame)
                        sv, fw = raw[L.VEL], raw[L.FORWARD]
                        ego_v = np.array([sv[0] * -fw[2] + sv[2] * fw[0], sv[0] * fw[0] + sv[2] * fw[2]])
                        own = float(np.dot(ego_v, horiz) / dist_h)        # +: jumper moving at the hostile
                        cur = {"t0": t, "dist": round(float(near[8]), 2), "closing": round(closing, 2),
                               "jumper_approach": round(own, 2),
                               "vy": round(float(raw[L.VEL][1]), 2), "attack_t": None, "avoid_t": None,
                               "hit": False, "near_id": int(near[1]),
                               "hp0": {int(e[1]): float(e[20]) for e in hs}}
                    else:
                        cur = {"t0": t, "dist": None, "closing": None, "vy": round(float(raw[L.VEL][1]), 2),
                               "attack_t": None, "avoid_t": None, "hit": False, "near_id": -1, "hp0": {}}
                if cur is not None:
                    if airborne and act[4] > 0.5 and cur["attack_t"] is None:
                        cur["attack_t"] = t - cur["t0"]
                    for e in hs:
                        if int(e[1]) == cur["near_id"] and e[AVOID] > 0.5 and cur["avoid_t"] is None:
                            cur["avoid_t"] = t - cur["t0"]
                        if int(e[1]) in cur["hp0"] and (e[20] < cur["hp0"][int(e[1])] - 0.02 or e[16] > 0.5
                                                         or e[17] > 0.5):
                            cur["hit"] = True
                    if not airborne and landed_at is None:
                        landed_at = t
                    if landed_at is not None and t - landed_at >= 15:      # 0.5 s after landing: close the record
                        cur["air_decisions"] = landed_at - cur["t0"]
                        cur.pop("hp0")
                        jumps.append(cur)
                        cur, landed_at = None, None
                obs, _r, done, info = env.step(act)
                kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                if done or kos >= a.opponents:
                    break
            wins += int(kos >= a.opponents)
    finally:
        env.close()
    attacks = [j for j in jumps if j["attack_t"] is not None]
    hops = [j for j in jumps if j["attack_t"] is None]

    def frac(xs, f):
        return round(sum(1 for x in xs if f(x)) / len(xs), 3) if xs else None

    summary = {"checkpoint": Path(a.checkpoint).name, "hop_aware": a.hop_aware, "opponents": a.opponents,
               "episodes": a.episodes, "wins": wins, "jumps": len(jumps), "hops": len(hops),
               "jump_attacks": len(attacks),
               "jump_attack_hit_rate": frac(attacks, lambda j: j["hit"]),
               "hop_hit_rate": frac(hops, lambda j: j["hit"]),
               "nearest_avoided_hop": frac(hops, lambda j: j["avoid_t"] is not None),
               "nearest_avoided_jump_attack": frac(attacks, lambda j: j["avoid_t"] is not None)}
    Path(a.out).write_text(json.dumps({"summary": summary, "jumps": jumps}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
