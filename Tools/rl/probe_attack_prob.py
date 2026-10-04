#!/usr/bin/env python3
"""OGRL-20261004-014: how likely is the policy to press ATTACK while grounded vs airborne?

Runs a checkpoint greedily (optionally under the move-school ground-only rule) and records, every
decision, the policy's Bernoulli probability for each button head together with GROUNDED and the
distance to the nearest valid enemy. Answers whether ground attacks are ever explored at all.
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
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402
from watch import deterministic_action  # noqa: E402

BUTTONS = ["jump", "crouch", "attack", "grab", "drop", "walk"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--ground-only", action="store_true")
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--shm-name", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs)
    nrm.load_state_dict(ck["obs_normalizer"])
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level,
                        shm_name=a.shm_name or f"/ogap{os.getpid() % 100000}", seed=a.seed_base,
                        act_period=4, frame_stack=fs, extra_config_lines=a.config_line)
    rows = []
    try:
        for ep in range(a.episodes):
            kw = {"opponents": a.opponents, "difficulty": a.difficulty}
            if a.ground_only:
                kw["ground_only"] = True
            obs = env.reset(seed=a.seed_base + ep, **kw)
            if ep == 0:
                obs = env.reset(seed=a.seed_base + ep, **kw)
            for _ in range(1200):
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                with torch.no_grad():
                    feats = pol._features(x)
                    logits = pol.discrete_logits(pol.actor_trunk(feats))[0]
                p = torch.sigmoid(logits).numpy()
                frame = np.asarray(env._prev_values, dtype=np.float32)
                dists = [float(e["distance"]) for e in L.all_entities(list(frame))
                         if e["valid"] and not e["is_controlled"] and not e["is_ally"]]
                rows.append({"grounded": bool(frame[L.GROUNDED] > 0.5),
                             "dist": min(dists) if dists else None,
                             **{b: float(p[i]) for i, b in enumerate(BUTTONS)}})
                action = deterministic_action(pol, x)
                obs, _r, done, _i = env.step(action)
                if done:
                    break
    finally:
        env.close()

    def summary(sel):
        if not sel:
            return None
        return {"n": len(sel), **{b: round(float(np.mean([r[b] for r in sel])), 4) for b in BUTTONS},
                "attack_p_gt_0.5": round(float(np.mean([r["attack"] > 0.5 for r in sel])), 4)}

    g = [r for r in rows if r["grounded"]]
    air = [r for r in rows if not r["grounded"]]
    near = [r for r in g if r["dist"] is not None and r["dist"] < 2.0]
    out = {"checkpoint": a.checkpoint, "global_step": int(ck.get("global_step", -1)),
           "ground_only": a.ground_only, "opponents": a.opponents, "difficulty": a.difficulty,
           "grounded": summary(g), "airborne": summary(air), "grounded_enemy_within_2m": summary(near)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
