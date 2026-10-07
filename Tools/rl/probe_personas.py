#!/usr/bin/env python3
"""OGRL-20261007-002/003: do the opponent personas, the intent-hiding switch and the horde spawn
groups actually do what they claim, in a real engine?

For each (persona, opponents) cell it plays a checkpoint greedily and records, per decision, how many
hostiles the observation lists and -- from the RAW observation, before any masking -- what the AI has
decided (ai_attacking, goal_attack), plus the outcome. With --hide-intent it also checks that the
POLICY's observation has the intent fields zeroed while the raw one does not.

The personas are expected to separate on ai_attacking: passive ~0, patient low, berserker high.
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

from env import OvergrowthEnv, PERSONAS  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402
from watch import deterministic_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_horde_301.xml")
    ap.add_argument("--personas", default="stock,patient,passive,berserker,expert,mixed")
    ap.add_argument("--opponents", default="1,3")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--hide-intent", action="store_true")
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    pol = ActorCritic(L, frame_stack=4)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=4)
    nrm.load_state_dict(ck["obs_normalizer"])
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogpp{os.getpid() % 100000}",
                        seed=960000, act_period=4, frame_stack=4, extra_config_lines=a.config_line)
    cells = []
    try:
        seed = 960000
        first = True
        for name in a.personas.split(","):
            for k in (int(x) for x in a.opponents.split(",")):
                n_host, attacking, goal_att, wins, timeouts, leak = [], [], [], 0, 0, 0
                for _ep in range(a.episodes):
                    kw = dict(seed=seed, opponents=k, difficulty=a.difficulty, persona=PERSONAS.index(name),
                              hide_intent=a.hide_intent)
                    obs = env.reset(**kw)
                    if first:                      # the first reset after load can belong to the load
                        obs = env.reset(**kw)
                        first = False
                    seed += 1
                    kos, won = 0, False
                    for t in range(1200):
                        raw = np.asarray(env._prev_values, dtype=np.float32)   # policy view (masked if hidden)
                        ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(
                            L.max_visible_entities, ef)
                        hostile = [i for i in range(L.max_visible_entities) if ents[i, 0] > 0.5 and ents[i, 23] < 0.5]
                        n_host.append(len(hostile))
                        if a.hide_intent and hostile:
                            leak += int(np.abs(ents[hostile][:, L.E_INTENT]).sum() > 0)
                        if hostile and not env.hide_intent:
                            attacking.append(float(ents[hostile, L.E_AI_ATTACKING].mean()))
                            goal_att.append(float(ents[hostile, L.E_GOAL_ATTACK].mean()))
                        x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                        obs, _r, done, info = env.step(deterministic_action(pol, x))
                        kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0) or 0))
                        won = kos >= k                    # vec_env's rule: every hostile down
                        if done or won:
                            break
                    wins += int(won)
                    timeouts += int(not (done or won))
                cell = {"persona": name, "opponents": k, "episodes": a.episodes,
                        "hostiles_listed_max": int(max(n_host) if n_host else 0),
                        "hostiles_listed_mean": round(float(np.mean(n_host)), 2) if n_host else 0,
                        "ai_attacking_share": round(float(np.mean(attacking)), 3) if attacking else None,
                        "goal_attack_share": round(float(np.mean(goal_att)), 3) if goal_att else None,
                        "timeouts": timeouts, "wins": wins,
                        "intent_leak_decisions": leak if a.hide_intent else None}
                print(json.dumps(cell), flush=True)
                cells.append(cell)
    finally:
        env.close()
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "level": a.level, "difficulty": a.difficulty,
                                       "hide_intent": a.hide_intent, "cells": cells}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
