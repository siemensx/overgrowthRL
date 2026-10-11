#!/usr/bin/env python3
"""OGRL-20261010-006: validate hop-aware enemies with a SCRIPTED attacker (no learned policy in the loop).

Three behaviours, each repeated as attempts against one enemy at difficulty 1.0, stock vs hop-aware enemies:
  running_kick   sprint at the enemy; at <= 2.6 m take a FULL jump (jump held 8 decisions) and press attack in the air
  standing_kick  walk to ~1.5 m, stop, full jump straight up, attack in the air (the fastest, least telegraphed kick)
  hop            stop at ~2.5 m and tap jump for one decision (a small hop), never attack -- the exploit
Per attempt: did the enemy enter _avoid_jump_kick, did the kick hit (enemy health drop / hit-reaction / ragdoll within
the airborne window + 0.5 s). For hop: enemy attacks started per minute, and how often the hopper got hit.
The hop-aware design goal: real kicks dodged about as often as against stock enemies (hit rate NOT higher),
hops ignored (avoid share far lower, enemy keeps attacking).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402

V6_OMNI = ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
           "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"]
AVOID = 39


def enemy(raw, ef):
    ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(-1, ef)
    hs = [e for e in ents if e[0] > 0.5 and e[23] < 0.5]
    return min(hs, key=lambda e: e[8]) if hs else None


def act(mx=0.0, my=0.0, jump=False, attack=False):
    return np.array([mx, my, float(jump), 0, float(attack), 0, 0, 0], dtype=np.float32)


def toward(e, speed=1.0):
    x, z = float(e[2]), float(e[4])
    n = (x * x + z * z) ** 0.5 or 1.0
    return speed * x / n, speed * z / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["running_kick", "standing_kick", "hop"], required=True)
    ap.add_argument("--hop-aware", action="store_true")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--level", default="arenas/t_held_203.xml")
    ap.add_argument("--seed", type=int, default=983000)
    ap.add_argument("--kick-dist", type=float, default=3.0, help="running_kick take-off distance")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogsk{os.getpid() % 10000}",
                        seed=a.seed, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    attempts, decisions, enemy_attack_starts, hopper_hit, approach_ok = [], 0, 0, 0, []
    try:
        for ep in range(a.episodes):
            kw = dict(seed=a.seed + ep, opponents=1, difficulty=1.0, hop_aware=a.hop_aware)
            env.reset(**kw)
            if ep == 0:
                env.reset(**kw)
            phase, timer, cur, prev_state_attack, prev_self_hp = "approach", 0, None, False, None
            for t in range(900):
                raw = np.asarray(env._prev_values, dtype=np.float32)
                e = enemy(raw, ef)
                if e is None or e[10] < 0.5:                       # enemy down or gone: episode over for our purpose
                    break
                airborne = raw[L.GROUNDED] <= 0.5
                dist = float(e[8])
                e_attack = e[15] > 0.5
                if e_attack and not prev_state_attack and dist < 3.0:
                    enemy_attack_starts += 1
                prev_state_attack = e_attack
                hp = float(raw[L.TEMP_HEALTH])
                if prev_self_hp is not None and hp < prev_self_hp - 0.02 and a.mode == "hop":
                    hopper_hit += 1
                prev_self_hp = hp
                stop_at = {"running_kick": a.kick_dist, "standing_kick": 1.6, "hop": 2.5}[a.mode]
                if phase == "approach":
                    if dist > stop_at:
                        mx, my = toward(e)
                        action = act(mx, my)
                        approach_ok.append(dist)
                    else:
                        phase, timer = ("settle" if a.mode != "running_kick" else "jump"), 0
                        action = act(*toward(e)) if a.mode == "running_kick" else act()
                    if phase == "jump":
                        cur = {"t0": t, "dist": round(dist, 2), "avoid": False, "hit": False,
                               "hp0": float(e[20]), "landed": None}
                elif phase == "settle":
                    action = act()
                    timer += 1
                    if timer >= 4:
                        phase, timer = "jump", 0
                        cur = {"t0": t, "dist": round(dist, 2), "avoid": False, "hit": False,
                               "hp0": float(e[20]), "landed": None}
                elif phase == "jump":
                    hold = 1 if a.mode == "hop" else 8
                    mx, my = toward(e) if a.mode == "running_kick" else (0.0, 0.0)
                    # one attack press as soon as airborne (how the agent's kicks that HIT look: attack 0-2
                    # decisions after take-off), released on the next decision so it is a single press
                    if airborne and cur.get("air_t") is None:
                        cur["air_t"] = timer
                    press = a.mode != "hop" and cur.get("air_t") is not None and timer - cur["air_t"] in (1, 2)
                    action = act(mx, my, jump=timer < hold, attack=press)
                    timer += 1
                    if e[AVOID] > 0.5:
                        cur["avoid"] = True
                    if e[20] < cur["hp0"] - 0.02 or e[16] > 0.5 or e[17] > 0.5:
                        cur["hit"] = True
                    if timer > 3 and not airborne and cur["landed"] is None:
                        cur["landed"] = t
                    if cur["landed"] is not None and t - cur["landed"] >= 15 or timer > 120:
                        cur.pop("hp0")
                        attempts.append(cur)
                        cur, phase, timer = None, "recover", 0
                else:                                                  # recover: back off a little, then go again
                    timer += 1
                    action = act(*[-v for v in toward(e, 0.6)]) if timer < 10 else act()
                    if timer >= 20:
                        phase, timer = "approach", 0
                _o, _r, done, _i = env.step(action)
                decisions += 1
                if done:
                    break
    finally:
        env.close()
    n = len(attempts)
    minutes = decisions / 30 / 60
    out = {"mode": a.mode, "hop_aware": a.hop_aware, "episodes": a.episodes, "attempts": n,
           "enemy_avoided_share": round(sum(x["avoid"] for x in attempts) / n, 3) if n else None,
           "hit_share": round(sum(x["hit"] for x in attempts) / n, 3) if n else None,
           "enemy_attack_starts_per_min_within_3m": round(enemy_attack_starts / minutes, 1) if minutes else None,
           "hopper_hits_taken_per_min": round(hopper_hit / minutes, 1) if a.mode == "hop" and minutes else None,
           "approach_samples": len(approach_ok)}
    Path(a.out).write_text(json.dumps({"summary": out, "attempts": attempts}, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
