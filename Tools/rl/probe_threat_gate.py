#!/usr/bin/env python3
"""OGRL-20261005-004: how often does the @threat floor gate fire in real fights?

Runs a checkpoint greedily and reports, per decision, whether an enemy within THREAT_RANGE_M is in
its attack state (the counter-throw window that `--button-floor grab=...@threat` targets), plus the
policy's own grab probability inside and outside that window.
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
from policy import ActorCritic, set_button_floor  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402
from watch import deterministic_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    pol = ActorCritic(L, frame_stack=4)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=4)
    nrm.load_state_dict(ck["obs_normalizer"])
    set_button_floor(pol, "grab=0.3@threat", normalizer=nrm)
    raw_pol = ActorCritic(L, frame_stack=4)
    raw_pol.load_state_dict(ck["policy"])
    raw_pol.eval()
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogtg{os.getpid() % 100000}",
                        seed=940000, act_period=4, frame_stack=4, extra_config_lines=a.config_line)
    gates, grab_p = [], []
    try:
        for ep in range(a.episodes):
            obs = env.reset(seed=940000 + ep, opponents=a.opponents, difficulty=1.0)
            if ep == 0:
                obs = env.reset(seed=940000 + ep, opponents=a.opponents, difficulty=1.0)
            for _ in range(1200):
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                with torch.no_grad():
                    pol._features(x)
                    gates.append(float(pol.discrete_logits.gate[0, 3]))
                    z = raw_pol.discrete_logits(raw_pol.actor_trunk(raw_pol._features(x)))[0]
                    grab_p.append(float(torch.sigmoid(z[3])))
                obs, _r, done, _i = env.step(deterministic_action(raw_pol, x))
                if done:
                    break
    finally:
        env.close()
    g = np.asarray(gates) > 0.5
    p = np.asarray(grab_p)
    out = {"checkpoint": a.checkpoint, "global_step": int(ck["global_step"]), "opponents": a.opponents,
           "decisions": int(g.size), "threat_share": round(float(g.mean()), 4),
           "grab_p_in_threat": round(float(p[g].mean()), 4) if g.any() else None,
           "grab_p_outside": round(float(p[~g].mean()), 4) if (~g).any() else None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
