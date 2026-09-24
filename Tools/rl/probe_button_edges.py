"""OGRL-20260924-008 engine probe: does a held grab re-arm the active block?

Holds grab for (period-1) decisions, releases it for one, repeats, standing
still, and counts rising edges of the agent's own ACTIVE_BLOCKING
observation field. Under the legacy RL input path every held physics tick is
a fresh press, which re-arms the 0.2 s block recharge on every tick, so the
count should be ~1 no matter how long the probe runs. Under
`rl_button_edges: 1` a hold is one press, so each release/press cycle long
enough for the recharge should start a new block.

    python Tools/rl/probe_button_edges.py --repo-root . --edges 0
    python Tools/rl/probe_button_edges.py --repo-root . --edges 1
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

GRAB = 2 + 3  # [move_x, move_y, jump, crouch, attack, grab, drop, walk]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--edges", type=int, choices=[0, 1], required=True)
    ap.add_argument("--decisions", type=int, default=390)
    ap.add_argument("--period", type=int, default=13, help="decisions per hold/release cycle (release = 1)")
    ap.add_argument("--shm-name", default=None)
    args = ap.parse_args()

    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level,
                        shm_name=args.shm_name or f"/ogrl_edges{args.edges}_{np.random.randint(1, 99999)}",
                        seed=424242, act_period=4, frame_stack=1,
                        extra_config_lines=[f"rl_button_edges: {args.edges}"])
    try:
        obs = env.reset(seed=424242, difficulty=0.0, opponents=1)
        blocks, prev_blocking, blocking_ticks = 0, False, 0
        max_recharge, wrong_self, self_ids = 0.0, 0, set()
        for t in range(args.decisions):
            action = np.zeros(8, dtype=np.float32)
            action[GRAB] = 0.0 if (t % args.period) == args.period - 1 else 1.0
            obs, _r, done, _info = env.step(action)
            blocking = obs[L.ACTIVE_BLOCKING] > 0.5
            max_recharge = max(max_recharge, float(obs[L.ACTIVE_BLOCK_RECHARGE]))
            self_ids.add(int(obs[0]))
            for slot in range(L.max_visible_entities):
                o = L.entities_start + slot * ENTITY_FLOATS
                if obs[o] > 0.5 and obs[o + 18] > 0.5:
                    wrong_self += 1
                    break
            blocking_ticks += int(blocking)
            if blocking and not prev_blocking:
                blocks += 1
            prev_blocking = blocking
            if done:
                break
        out = {"edges": args.edges, "decisions": t + 1, "cycles": (t + 1) // args.period,
               "block_starts": blocks, "decisions_blocking": blocking_ticks,
               "max_recharge_seen": max_recharge, "self_ids": sorted(self_ids),
               "decisions_self_not_player": wrong_self}
        print(json.dumps(out))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
