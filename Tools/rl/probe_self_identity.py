"""OGRL-20260924-009 engine probe: is the observed "self" the character the
policy actually controls?

rl_shm_transport.cpp observes the FIRST movement object whose controller_id
equals the RL controller id. AI characters also carry controller_id 0, so
whenever an enemy precedes the player in scenegraph order the policy is fed
that enemy's proprioception as its own -- and the real player then appears in
the entity list with is_controlled = 1 (entity field 18). This counts, per
reset mode and opponent count, the fraction of episodes and decisions in
which that happens.
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

IS_CONTROLLED = 18


def controlled_entity_ids(obs: np.ndarray) -> list[int]:
    out = []
    for slot in range(L.max_visible_entities):
        o = L.entities_start + slot * ENTITY_FLOATS
        if obs[o] > 0.5 and obs[o + IS_CONTROLLED] > 0.5:
            out.append(int(obs[o + 1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--soft", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--shm-name", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level,
                        shm_name=args.shm_name or f"/ogrl_selfid{np.random.randint(1, 99999)}",
                        seed=777, act_period=4, frame_stack=1)
    rows = []
    try:
        for ep in range(args.episodes):
            obs = env.reset(seed=1000 + ep, soft=bool(args.soft) and ep > 0,
                            difficulty=args.difficulty, opponents=args.opponents)
            wrong, seen, self_ids = 0, 0, set()
            for t in range(args.decisions):
                a = rng.uniform(-1, 1, 8).astype(np.float32)
                a[2:] = (a[2:] > 0.6).astype(np.float32)
                obs, _r, done, _i = env.step(a)
                seen += 1
                self_ids.add(int(obs[0]))
                if controlled_entity_ids(obs):
                    wrong += 1
                if done:
                    break
            rows.append({"episode": ep, "self_ids": sorted(self_ids), "decisions": seen,
                         "decisions_with_controlled_entity": wrong})
    finally:
        env.close()
    bad_eps = sum(1 for r in rows if r["decisions_with_controlled_entity"] > 0)
    print(json.dumps({"opponents": args.opponents, "soft": args.soft, "episodes": len(rows),
                      "episodes_self_is_not_player": bad_eps,
                      "decision_fraction": sum(r["decisions_with_controlled_entity"] for r in rows)
                      / max(1, sum(r["decisions"] for r in rows)),
                      "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
