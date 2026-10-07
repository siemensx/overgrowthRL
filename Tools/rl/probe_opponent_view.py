#!/usr/bin/env python3
"""OGRL-20261007-001: what does the policy see about its opponent -- a human (play_match setup) vs the
scripted AI? Runs a checkpoint greedily for N decisions and records, for the nearest valid enemy slot,
every entity field, plus the policy's button probabilities and own GROUNDED/RULE fields.

  --mode human : arena_level_human_duel.as level, agent on controller 1, the human slot gets no input
  --mode ai    : a training map, 1v1 against the scripted AI (controller 0, as in training)
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
    ap.add_argument("--mode", choices=["human", "ai"], required=True)
    ap.add_argument("--level", default=None)
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    pol = ActorCritic(L, frame_stack=4)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=4)
    nrm.load_state_dict(ck["obs_normalizer"])
    level = a.level or ("arenas/oval_arena_human_duel.xml" if a.mode == "human" else "arenas/t_train_101.xml")
    kw = dict(repo_root=str(HERE.parents[1]), level=level, shm_name=f"/ogov{os.getpid() % 100000}", seed=950000,
              act_period=4, frame_stack=4, extra_config_lines=a.config_line, render=a.render)
    if a.mode == "human":
        kw["controller_id"] = 1
        kw["time_scale_mult"] = 1
    env = OvergrowthEnv(**kw)
    ef = L.entity_slice(0).stop - L.entity_slice(0).start
    rows = []
    try:
        obs = env.reset(seed=950000) if a.mode == "human" else env.reset(seed=950000, opponents=1, difficulty=1.0)
        if a.mode == "ai":
            obs = env.reset(seed=950000, opponents=1, difficulty=1.0)
        for _ in range(a.decisions):
            raw = np.asarray(env._prev_values, dtype=np.float32)
            ents = raw[L.entities_start:L.entities_start + L.max_visible_entities * ef].reshape(L.max_visible_entities, ef)
            valid = [i for i in range(L.max_visible_entities) if ents[i, 0] > 0.5]
            x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
            with torch.no_grad():
                z = pol.discrete_logits(pol.actor_trunk(pol._features(x)))[0]
            p = torch.sigmoid(z).numpy()
            if valid:
                e = ents[min(valid, key=lambda i: ents[i, 8])]
                rows.append({"ent": e.tolist(), "grounded": float(raw[L.GROUNDED]), "p": p.tolist(),
                             "n_valid": len(valid)})
            obs, _r, done, _i = env.step(deterministic_action(pol, x))
            if done:
                obs = env.reset(seed=950001) if a.mode == "human" else env.reset(seed=950001, opponents=1, difficulty=1.0)
    finally:
        env.close()
    E = np.asarray([r["ent"] for r in rows]) if rows else np.zeros((0, ef))
    P = np.asarray([r["p"] for r in rows]) if rows else np.zeros((0, 6))
    out = {"mode": a.mode, "level": level, "global_step": int(ck["global_step"]), "decisions_with_enemy": len(rows),
           "entity_field_mean": E.mean(0).round(3).tolist() if len(rows) else None,
           "entity_field_std": E.std(0).round(3).tolist() if len(rows) else None,
           "distance_mean": round(float(E[:, 8].mean()), 2) if len(rows) else None,
           "button_p_mean": {b: round(float(P[:, i].mean()), 3) for i, b in enumerate(BUTTONS)} if len(rows) else None,
           "grounded_share": round(float(np.mean([r["grounded"] > 0.5 for r in rows])), 3) if rows else None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("entity_field_mean", "entity_field_std")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
