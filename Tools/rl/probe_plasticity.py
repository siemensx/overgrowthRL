#!/usr/bin/env python3
"""OGRL-20261004-018: is the network losing plasticity?

Collects observations from greedy fights (one checkpoint drives, so every checkpoint is measured on
the SAME observations), then for each checkpoint reports, per tanh layer of the actor and critic:
  * saturated  -- share of (unit, sample) activations with |tanh| > 0.99 (near-zero gradient)
  * dead       -- share of units saturated on >= 99% of samples (effectively frozen)
  * dormant    -- share of units whose mean |activation| is < 10% of the layer's mean (ReDo, Sokar 2023)
plus first-layer weight norms. Long-trained policies that stop learning typically show rising
saturation/dormancy and growing weight norms (Lyle et al. 2023; Nikishin et al. 2022).
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


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs)
    nrm.load_state_dict(ck["obs_normalizer"])
    return ck, pol, nrm


def layer_stats(acts: torch.Tensor) -> dict:
    a = acts.abs()
    sat = (a > 0.99).float()
    per_unit_sat = sat.mean(0)
    mean_act = a.mean(0)
    return {"saturated": round(float(sat.mean()), 4),
            "dead_units": round(float((per_unit_sat >= 0.99).float().mean()), 4),
            "dormant_units": round(float((mean_act < 0.1 * mean_act.mean()).float().mean()), 4),
            "mean_abs": round(float(a.mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="checkpoint that plays the fights")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _ck, dpol, dnrm = load(a.driver)
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogpl{os.getpid() % 100000}",
                        seed=920000, act_period=4, frame_stack=4, extra_config_lines=a.config_line)
    raw = []
    try:
        for ep in range(a.episodes):
            obs = env.reset(seed=920000 + ep, opponents=a.opponents, difficulty=1.0)
            if ep == 0:
                obs = env.reset(seed=920000 + ep, opponents=a.opponents, difficulty=1.0)
            for _ in range(1200):
                raw.append(np.asarray(obs, dtype=np.float32))
                x = torch.as_tensor(dnrm.normalize(obs, update=False), dtype=torch.float32)
                obs, _r, done, _i = env.step(deterministic_action(dpol, x))
                if done:
                    break
    finally:
        env.close()
    raw = np.stack(raw)

    report = {"n_obs": int(raw.shape[0]), "driver": a.driver, "checkpoints": {}}
    for path in a.checkpoints:
        ck, pol, nrm = load(path)
        x = torch.as_tensor(nrm.normalize(raw, update=False), dtype=torch.float32)
        out = {"global_step": int(ck.get("global_step", -1))}
        acts = {}
        hooks = []
        for mname, mod in pol.named_modules():
            if isinstance(mod, (torch.nn.Tanh, torch.nn.ReLU)):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, o, mname=mname: acts.setdefault(mname, []).append(o.detach())))
        with torch.no_grad():
            pol.get_action_and_value(x)
        for hk in hooks:
            hk.remove()
        for mname, lst in acts.items():
            t = torch.cat([v.reshape(-1, v.shape[-1]) for v in lst])
            st = layer_stats(t) if isinstance(dict(pol.named_modules())[mname], torch.nn.Tanh) else {
                "dead_units": round(float(((t > 0).float().mean(0) < 0.01).float().mean()), 4),
                "dormant_units": round(float((t.abs().mean(0) < 0.1 * t.abs().mean()).float().mean()), 4)}
            out[mname] = st
        out["weight_norms"] = {k: round(float(v.norm()), 2) for k, v in ck["policy"].items()
                               if k.endswith("weight") and v.dim() == 2}
        report["checkpoints"][os.path.basename(path)] = out
    Path(a.out).write_text(json.dumps(report, indent=1))
    for name, out in report["checkpoints"].items():
        print(name, out["global_step"])
        for k, v in out.items():
            if k not in ("global_step", "weight_norms"):
                print(f"   {k:14} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
