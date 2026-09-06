#!/usr/bin/env python3
"""What does the policy actually DO? Action-usage profile for a checkpoint.

The standing observation about this agent is that it "only jump-kicks". That has
never been measured -- and after the partial-observability hypothesis was tested
and refuted (OGRL-20260906-073), move diversity is the most-cited untested claim
left about why multi-opponent performance is capped.

The action space is 2 continuous (move_x, move_y) + 6 binary (jump, crouch,
attack, grab, drop, walk). What distinguishes Overgrowth's attacks is not a
separate button but the CONTEXT the attack button is pressed in -- airborne vs
grounded, crouched vs standing, moving vs still. So classify each attack press
by its context, which is exactly how the game itself decides which move comes
out (aschar.as's UpdateAttackState).

Reports the marginal rate of every button, the attack-context breakdown, and
the movement-stick distribution.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

BUTTONS = ["jump", "crouch", "attack", "grab", "drop", "walk"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=4000)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--shm-name", default="/ogrl_ap")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs); pol.load_state_dict(ck["policy"]); pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs); nrm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=os.getcwd(), level=a.level, shm_name=a.shm_name,
                        seed=a.seed, act_period=4, frame_stack=fs)
    env.reset(seed=a.seed, difficulty=a.difficulty, opponents=a.opponents)
    obs = env.reset(seed=a.seed + 1, difficulty=a.difficulty, opponents=a.opponents)

    press = Counter()
    ctx = Counter()
    move_mag = []
    n = 0
    ep = 0
    for _ in range(a.decisions):
        with torch.no_grad():
            x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32).unsqueeze(0)
            act, _, _, _ = pol.get_action_and_value(x)
        av = act.squeeze(0).numpy()
        raw = env._prev_values
        grounded = bool(np.asarray(raw, dtype=np.float32)[L.GROUNDED] > 0.5) if raw is not None else True

        bits = {b: bool(av[2 + i] > 0.5) for i, b in enumerate(BUTTONS)}
        for b, v in bits.items():
            if v:
                press[b] += 1
        mv = float(np.hypot(av[0], av[1]))
        move_mag.append(mv)
        if bits["attack"]:
            if not grounded:
                ctx["AIR  (jump kick)"] += 1
            elif bits["crouch"]:
                ctx["CROUCH (leg sweep / cannon)"] += 1
            elif mv > 0.5:
                ctx["GROUND moving (running strike)"] += 1
            else:
                ctx["GROUND still (standing strike)"] += 1
        n += 1
        obs, _r, done, _i = env.step(av)
        if done:
            ep += 1
            obs = env.reset(seed=a.seed + 100 + ep, difficulty=a.difficulty, opponents=a.opponents)
    env.close()

    print(f"checkpoint step={ck['global_step']:,}  opponents={a.opponents}  "
          f"difficulty={a.difficulty}  decisions={n}  episodes={ep}\n")
    print("button press rate (fraction of decisions):")
    for b in BUTTONS:
        print(f"  {b:8} {press[b] / max(1, n):6.3f}")
    total_atk = sum(ctx.values())
    print(f"\nattack context ({total_atk} attack presses, "
          f"{total_atk / max(1, n):.3f} of decisions):")
    if total_atk == 0:
        print("  never attacked")
    for k, v in ctx.most_common():
        print(f"  {k:32} {v:6d}  {100.0 * v / total_atk:5.1f}%")
    mm = np.asarray(move_mag)
    print(f"\nmovement stick magnitude: mean={mm.mean():.3f} "
          f"p10={np.percentile(mm,10):.3f} p50={np.percentile(mm,50):.3f} p90={np.percentile(mm,90):.3f}")
    print(f"  fraction near-still (<0.2): {(mm < 0.2).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
