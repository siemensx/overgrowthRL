#!/usr/bin/env python3
"""How much win rate is the exploration noise costing?

action_profile.py (OGRL-20260906-074) showed the policy attacks on ~40% of
decisions and moves at near-full stick ~99% of the time. Some of that is learned
behaviour and some is just the stochastic policy's sampling noise -- six
Bernoulli heads sampled independently every decision, plus a tanh-Gaussian on the
movement stick.

This runs the SAME checkpoint two ways over matched seeds:
  * stochastic -- exactly what training does (and what every in-training win rate
    reports), and
  * greedy -- the distribution mode: tanh(mean) for the stick, sigmoid(logits)>0.5
    for each button.

If greedy is much better, the entropy bonus is being paid in real performance and
lowering `--entropy-coef-final` is a cheap, targeted win. If they match, the
behaviour is learned rather than noise, and the entropy setting is not the lever.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402


def _greedy_action(pol, x):
    feats = pol._features(x)
    mean, _log_std, logits = pol._actor_params(feats)
    cont = torch.tanh(mean)
    disc = (torch.sigmoid(logits) > 0.5).float()
    return torch.cat([cont, disc], dim=-1)


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(pol, nrm, fs, level, opponents, episodes, difficulty, seed, shm, greedy):
    env = OvergrowthEnv(repo_root=os.getcwd(), level=level, shm_name=shm, seed=seed,
                        act_period=4, frame_stack=fs)
    env.reset(seed=seed, difficulty=difficulty, opponents=opponents)
    obs = env.reset(seed=seed + 1, difficulty=difficulty, opponents=opponents)
    wins = ep = kos = steps = 0
    lens = []
    while ep < episodes:
        with torch.no_grad():
            x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32).unsqueeze(0)
            act = _greedy_action(pol, x) if greedy else pol.get_action_and_value(x)[0]
        obs, _r, done, info = env.step(act.squeeze(0).numpy())
        steps += 1
        rc = (info or {}).get("reward_components", {}) or {}
        k = rc.get("hostile_kos_this_step")
        kos += int(round(k)) if k is not None else (1 if rc.get("opponent_knockout", 0) > 0 else 0)
        if done:
            if kos >= opponents:
                wins += 1
            lens.append(steps)
            ep += 1
            kos = steps = 0
            # SAME seed sequence for both arms -- paired comparison, not two
            # independent samples of a noisy environment.
            obs = env.reset(seed=seed + 100 + ep, difficulty=difficulty, opponents=opponents)
    env.close()
    return wins, ep, (sum(lens) / max(1, len(lens)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--shm-prefix", default="/ogrl_ga")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs); pol.load_state_dict(ck["policy"]); pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs); nrm.load_state_dict(ck["obs_normalizer"])

    print(f"checkpoint step={ck['global_step']:,}  difficulty={a.difficulty}  "
          f"episodes={a.episodes} per arm, paired seeds\n")
    print(f"{'opp':>3} {'arm':>11} {'win rate':>9} {'95% CI':>16} {'mean len':>9}")
    for opp in a.opponents:
        for greedy in (False, True):
            w, n, ml = run(pol, nrm, fs, a.level, opp, a.episodes, a.difficulty,
                           a.seed, f"{a.shm_prefix}{opp}{int(greedy)}_{os.getpid()}", greedy)
            lo, hi = _wilson(w, n)
            print(f"{opp:>3} {'greedy' if greedy else 'stochastic':>11} "
                  f"{w / n:>9.3f} {f'[{lo:.2f},{hi:.2f}]':>16} {ml:>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
