"""OGRL-20260924-012: are knockouts the agent causes actually counted?

reward.py credits a knockout only when the SAME hostile is visible on two
consecutive decisions, awake on the first and down on the second, with
attacked_by_id == self. A hostile that goes down while out of sight (knocked
flying, ragdoll settles behind the agent) is never credited, so the win (all
hostiles down) can never register. This plays a checkpoint and tracks, per
hostile id, every observed state: credited KOs vs hostiles ever SEEN down, and
who (attacked_by_id) they were last hit by when seen down.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ppo"))
from env import OvergrowthEnv  # noqa: E402
from obs_schema import ObsLayout, ENTITY_FLOATS  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402


def hostiles(f, L):
    out = {}
    for slot in range(L.max_visible_entities):
        o = L.entities_start + slot * ENTITY_FLOATS
        if f[o] <= 0.5 or f[o + 23] > 0.5:
            continue
        out[int(f[o + 1])] = {"awake": bool(f[o + 10] > 0.5), "by": int(f[o + 22])}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--level", default="arenas/t_train_101.xml")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed-base", type=int, default=960_000)
    p.add_argument("--config-line", action="append", default=[])
    p.add_argument("--shm-name", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    L = ObsLayout()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ckpt.get("frame_stack", 4))
    policy = ActorCritic(layout=L, frame_stack=fs); policy.load_state_dict(ckpt["policy"]); policy.eval()
    norm = ObservationNormalizer(L, frame_stack=fs); norm.load_state_dict(ckpt["obs_normalizer"])
    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level,
                        shm_name=args.shm_name or f"/ogrl_ko{np.random.randint(1, 99999)}",
                        seed=args.seed_base, layout=L, frame_stack=fs, act_period=4,
                        extra_config_lines=list(args.config_line))
    rows = []
    try:
        for ep in range(args.episodes):
            raw = env.reset(seed=args.seed_base + ep, soft=ep > 0, difficulty=1.0, opponents=3)
            credited, outcome = 0, "timeout"
            seen_down, seen_down_by_self, last_awake = set(), set(), {}
            self_id = int(raw[-L.total_floats:][0])
            for t in range(1200):
                with torch.no_grad():
                    a, *_ = policy.get_action_and_value(
                        torch.as_tensor(norm.normalize(raw, update=False), dtype=torch.float32).unsqueeze(0))
                raw, _r, done, info = env.step(a.squeeze(0).numpy())
                f = raw[-L.total_floats:]
                credited += int(round(info["reward_components"].get("hostile_kos_this_step", 0.0)))
                for hid, h in hostiles(f, L).items():
                    last_awake[hid] = h["awake"]
                    if not h["awake"]:
                        seen_down.add(hid)
                        if h["by"] == self_id:
                            seen_down_by_self.add(hid)
                if credited >= 3:
                    outcome = "won"; break
                if done:
                    outcome = "lost"; break
            rows.append({"ep": ep, "outcome": outcome, "length": t + 1, "credited_kos": credited,
                         "hostiles_seen_down": len(seen_down), "seen_down_last_hit_by_self": len(seen_down_by_self),
                         "hostiles_ever_seen": len(last_awake)})
    finally:
        env.close()
    missed = [r for r in rows if r["seen_down_last_hit_by_self"] > r["credited_kos"]]
    out = {"checkpoint": Path(args.checkpoint).name, "episodes": len(rows),
           "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in ("won", "lost", "timeout")},
           "episodes_with_uncredited_self_ko": len(missed),
           "uncredited_kos_total": sum(r["seen_down_last_hit_by_self"] - r["credited_kos"] for r in missed),
           "episodes_seen_all_3_down_but_not_won": sum(1 for r in rows if r["hostiles_seen_down"] >= 3 and r["outcome"] != "won"),
           "rows": rows}
    print(json.dumps(out, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
