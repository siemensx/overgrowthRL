"""OGRL-20260924-011: in what situation does the agent take its damage?

Runs a checkpoint (sampled policy, the training behaviour) on 1v3 and, at every
decision where the agent's own health drops, records the situation at the
PREVIOUS decision (what it was doing when the hit was coming): its own state
(neutral / on-ground / attacking / already reeling / ragdolled), airborne,
actively blocking, how many awake hostiles were within 3 m and how many of
those were in their attack state, how many hostiles it could not see, and how
many knockouts it had already scored. Damage is summed per category, so the
output reads "x% of all damage taken arrived while ...".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ppo"))
from env import OvergrowthEnv  # noqa: E402
from obs_schema import ObsLayout, ENTITY_FLOATS  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

SELF_STATES = ["neutral", "on_ground", "attacking", "reeling", "ragdoll"]


def health(f, L):
    return float(0.5 * (f[L.TEMP_HEALTH] + f[L.BLOOD_HEALTH]))


def situation(f, L, kos_so_far, opponents):
    st = int(np.argmax(f[L.STATE]))
    near, near_attacking, visible_awake = 0, 0, 0
    for slot in range(L.max_visible_entities):
        o = L.entities_start + slot * ENTITY_FLOATS
        if f[o] <= 0.5 or f[o + 23] > 0.5 or f[o + 10] <= 0.5:   # invalid, ally, or not awake
            continue
        visible_awake += 1
        if f[o + 8] <= 3.0:
            near += 1
            if f[o + 13 + 2] > 0.5:
                near_attacking += 1
    unseen = max(0, opponents - kos_so_far - visible_awake)
    return {
        "self_state": SELF_STATES[st] if st < len(SELF_STATES) else str(st),
        "airborne": bool(f[L.GROUNDED] < 0.5),
        "blocking": bool(f[L.ACTIVE_BLOCKING] > 0.5),
        "hostiles_within_3m": min(near, 3),
        "attacking_within_3m": min(near_attacking, 3),
        "unseen_hostiles": unseen,
        "kos_so_far": kos_so_far,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--level", default="arenas/t_train_101.xml")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--opponents", type=int, default=3)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=950_000)
    p.add_argument("--config-line", action="append", default=[])
    p.add_argument("--shm-name", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    L = ObsLayout()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ckpt.get("frame_stack", 4))
    policy = ActorCritic(layout=L, frame_stack=fs)
    policy.load_state_dict(ckpt["policy"]); policy.eval()
    norm = ObservationNormalizer(L, frame_stack=fs); norm.load_state_dict(ckpt["obs_normalizer"])
    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level,
                        shm_name=args.shm_name or f"/ogrl_dmg{np.random.randint(1, 99999)}",
                        seed=args.seed_base, layout=L, frame_stack=fs, act_period=4,
                        extra_config_lines=list(args.config_line))
    dmg = {k: defaultdict(float) for k in ["self_state", "airborne", "blocking", "hostiles_within_3m",
                                           "attacking_within_3m", "unseen_hostiles", "kos_so_far"]}
    hits = defaultdict(int)
    episodes = []
    total = 0.0
    try:
        for ep in range(args.episodes):
            raw = env.reset(seed=args.seed_base + ep, soft=ep > 0, difficulty=args.difficulty,
                            opponents=args.opponents)
            frame = raw[-L.total_floats:]
            kos, taken, first_hit_t = 0, 0.0, None
            outcome = "timeout"
            for t in range(1200):
                with torch.no_grad():
                    a, *_ = policy.get_action_and_value(
                        torch.as_tensor(norm.normalize(raw, update=False), dtype=torch.float32).unsqueeze(0))
                prev = frame
                sit = situation(prev, L, kos, args.opponents)
                raw, _r, done, info = env.step(a.squeeze(0).numpy())
                frame = raw[-L.total_floats:]
                kos += int(round(info["reward_components"].get("hostile_kos_this_step", 0.0)))
                drop = health(prev, L) - health(frame, L)
                if drop > 0.002:
                    total += drop; taken += drop
                    if first_hit_t is None:
                        first_hit_t = t
                    for k, v in sit.items():
                        dmg[k][str(v)] += drop
                    hits[sit["self_state"]] += 1
                if kos >= args.opponents:
                    outcome = "won"; break
                if done:
                    outcome = "lost"; break
            episodes.append({"ep": ep, "outcome": outcome, "kos": kos, "length": t + 1,
                             "damage_taken": round(taken, 3), "first_hit_decision": first_hit_t})
    finally:
        env.close()
    share = {k: {kk: round(vv / max(total, 1e-9), 3) for kk, vv in sorted(v.items())} for k, v in dmg.items()}
    out = {"checkpoint": Path(args.checkpoint).name, "global_step": ckpt.get("global_step"),
           "episodes": len(episodes), "wins": sum(e["outcome"] == "won" for e in episodes),
           "total_damage": round(total, 3), "damage_share_by_situation_before_hit": share,
           "hits_by_self_state": dict(hits), "episode_rows": episodes}
    print(json.dumps(out, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
