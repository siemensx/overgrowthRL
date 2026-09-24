"""OGRL-20260924-009: does the observed "self" move when the policy says move?

Alternates 20 decisions of full forward/right stick with 20 of no input and
reports the mean observed self speed in each phase, plus self_id and whether
any visible entity is player-controlled. If "self" is the player, speed is
high in move phases and ~0 in idle phases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L, ENTITY_FLOATS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--shm-name", default=None)
    args = ap.parse_args()
    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level,
                        shm_name=args.shm_name or f"/ogrl_motion{np.random.randint(1, 99999)}",
                        seed=args.seed, act_period=4, frame_stack=1)
    phases = []
    try:
        obs = env.reset(seed=args.seed, difficulty=args.difficulty, opponents=args.opponents)
        for phase in range(4):
            moving = phase % 2 == 0
            speeds, ctrl_seen = [], 0
            for _ in range(20):
                a = np.zeros(8, np.float32)
                if moving:
                    a[0], a[1] = 0.7, 0.7
                obs, _r, done, _i = env.step(a)
                speeds.append(float(np.linalg.norm(obs[L.VEL][[0, 2]])))
                for slot in range(L.max_visible_entities):
                    o = L.entities_start + slot * ENTITY_FLOATS
                    if obs[o] > 0.5 and obs[o + 18] > 0.5:
                        ctrl_seen += 1
                        break
                if done:
                    break
            phases.append({"moving": moving, "mean_self_speed": round(float(np.mean(speeds)), 3),
                           "self_id": int(obs[0]), "decisions_player_visible_as_entity": ctrl_seen,
                           "n_entities_visible": int(sum(obs[L.entities_start + s * ENTITY_FLOATS] > 0.5
                                                         for s in range(L.max_visible_entities)))})
    finally:
        env.close()
    print(json.dumps({"seed": args.seed, "opponents": args.opponents, "phases": phases}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
