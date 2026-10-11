#!/usr/bin/env python3
"""OGRL-20261010-002: does the policy keep its decisions between milestones, or keep rewriting them?

Weight-space distance is misleading for LayerNorm networks under weight decay (scale-invariant weights keep
rotating at equilibrium without changing the function). This measures FUNCTION drift instead: one driver
checkpoint plays fights and the observations are stored; every checkpoint then decides on exactly those
observations. Reported per pair of checkpoints:
  * button_agree  share of decisions with the identical greedy button set (jump/crouch/attack/grab/drop/walk)
  * attack_agree / jump_agree  agreement on those two buttons alone
  * stick_diff    mean absolute difference of the stick means
and per checkpoint the mean probability of jump and attack (how much of each it wants to do).
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
from probe_plasticity import load  # noqa: E402
from watch import deterministic_action  # noqa: E402

V6_OMNI = ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
           "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"]


def collect(driver, level, opponents, episodes, seed0):
    _ck, pol, nrm = load(driver)
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=level, shm_name=f"/ogpd{os.getpid() % 10000}{opponents}",
                        seed=seed0, act_period=4, frame_stack=4, extra_config_lines=V6_OMNI)
    raw = []
    try:
        for ep in range(episodes):
            obs = env.reset(seed=seed0 + ep, opponents=opponents, difficulty=1.0)
            if ep == 0:
                obs = env.reset(seed=seed0 + ep, opponents=opponents, difficulty=1.0)
            for _ in range(1200):
                raw.append(np.asarray(obs, dtype=np.float32))
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                obs, _r, done, _i = env.step(deterministic_action(pol, x))
                if done:
                    break
    finally:
        env.close()
    return np.stack(raw)


def decide(path, raw):
    ck, pol, nrm = load(path)
    x = torch.as_tensor(nrm.normalize(raw, update=False), dtype=torch.float32)
    with torch.no_grad():
        f = pol.actor_trunk(pol._features(x))
        logits = pol.discrete_logits(f)
        stick = torch.tanh(pol.continuous_mean(f)) if True else pol.continuous_mean(f)
    return int(ck.get("global_step", -1)), (logits > 0).numpy(), torch.sigmoid(logits).numpy(), stick.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--level", default="arenas/t_held_203.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=977000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = collect(a.driver, a.level, a.opponents, a.episodes, a.seed)
    dec = {Path(p).stem: decide(p, raw) for p in a.checkpoints}
    names = list(dec)
    per = {n: {"step": dec[n][0], "p_jump": round(float(dec[n][2][:, 0].mean()), 3),
               "p_attack": round(float(dec[n][2][:, 2].mean()), 3), "p_grab": round(float(dec[n][2][:, 3].mean()), 3),
               "greedy_jump_share": round(float(dec[n][1][:, 0].mean()), 3),
               "greedy_attack_share": round(float(dec[n][1][:, 2].mean()), 3)} for n in names}
    pairs = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            bx, by = dec[x][1], dec[y][1]
            pairs[f"{x} vs {y}"] = {"button_agree": round(float((bx == by).all(1).mean()), 3),
                                    "attack_agree": round(float((bx[:, 2] == by[:, 2]).mean()), 3),
                                    "jump_agree": round(float((bx[:, 0] == by[:, 0]).mean()), 3),
                                    "stick_diff": round(float(np.abs(dec[x][3] - dec[y][3]).mean()), 3)}
    out = {"n_obs": int(raw.shape[0]), "level": a.level, "opponents": a.opponents, "driver": a.driver,
           "per_checkpoint": per, "pairs": pairs}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(per, indent=1))
    for k, v in pairs.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
