#!/usr/bin/env python3
"""OGRL-20261007-013: how concerning is a saturated tanh layer?

probe_plasticity.py reports the SHARE of |tanh| > 0.99. That alone does not say whether a layer is
broken. For every tanh layer this probe adds, on the same observations for every checkpoint:
  * grad_pass   mean of 1 - tanh^2: the factor by which the layer scales the learning signal flowing back
                through it (1 = untouched, 0 = blocked). Saturation matters for learning only through this.
  * eff_rank    participation ratio of the activation covariance eigenvalues (sum l)^2 / sum l^2: how many
                independent directions the layer actually uses, out of its width.
  * switch_units share of units that saturate on BOTH signs (behave like on/off switches -- still carry
                information) vs one_sign_units (saturated on >= 50% of samples, always the same sign --
                nearly constant, i.e. wasted width).
  * pre_std     std of the pre-activation (input to the tanh), averaged over units; growth here is what
                drives saturation (weights or inputs getting larger).
Also, for the attention summary only, how much the policy's OUTPUT depends on it: the change in
button probabilities and stick means when the summary is replaced by its batch mean (ablation).
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
from probe_plasticity import load  # noqa: E402
from watch import deterministic_action  # noqa: E402


def tanh_stats(post: torch.Tensor, pre: torch.Tensor) -> dict:
    a = post.abs()
    sat = a > 0.99
    frac = sat.float().mean(0)
    pos = ((post > 0.99).float().mean(0))
    neg = ((post < -0.99).float().mean(0))
    switch = ((pos > 0.05) & (neg > 0.05)).float().mean()
    one_sign = ((frac >= 0.5) & ((pos < 0.01) | (neg < 0.01))).float().mean()
    c = torch.cov(post.T.double())
    ev = torch.linalg.eigvalsh(c).clamp(min=0)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else 0.0
    return {"width": int(post.shape[1]), "saturated": round(float(sat.float().mean()), 4),
            "grad_pass": round(float((1 - post ** 2).mean()), 4), "eff_rank": round(pr, 2),
            "switch_units": round(float(switch), 3), "one_sign_units": round(float(one_sign), 3),
            "pre_std": round(float(pre.std(0).mean()), 3), "pre_abs_mean": round(float(pre.abs().mean()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _ck, dpol, dnrm = load(a.driver)
    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=f"/ogsd{os.getpid() % 100000}",
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
    report = {"n_obs": int(raw.shape[0]), "driver": a.driver, "opponents": a.opponents, "checkpoints": {}}
    for path in a.checkpoints:
        ck, pol, nrm = load(path)
        x = torch.as_tensor(nrm.normalize(raw, update=False), dtype=torch.float32)
        pre, post, hooks = {}, {}, []
        def record(name):
            def hook(_m, i, o):            # must return None: a returned value REPLACES the output
                pre.setdefault(name, []).append(i[0].detach())
                post.setdefault(name, []).append(o.detach())
            return hook
        for name, mod in pol.named_modules():
            if isinstance(mod, torch.nn.Tanh):
                hooks.append(mod.register_forward_hook(record(name)))
        with torch.no_grad():
            feats = pol._features(x)
            base_logits = pol.discrete_logits(pol.actor_trunk(feats))
            base_mean = pol.continuous_mean(pol.actor_trunk(feats))
        for h in hooks:
            h.remove()
        out = {"global_step": int(ck.get("global_step", -1)), "layers": {}}
        for name in post:
            po = torch.cat([v.reshape(-1, v.shape[-1]) for v in post[name]])
            pr = torch.cat([v.reshape(-1, v.shape[-1]) for v in pre[name]])
            out["layers"][name] = tanh_stats(po, pr)
        # ablation: replace the attention summary with its batch mean
        enc = getattr(pol, "entity_encoder", None)
        if enc is not None:
            def mean_out(_m, _i, o):
                return o.mean(0, keepdim=True).expand_as(o)
            h = enc.register_forward_hook(mean_out)
            with torch.no_grad():
                f2 = pol._features(x)
                l2 = pol.discrete_logits(pol.actor_trunk(f2))
                m2 = pol.continuous_mean(pol.actor_trunk(f2))
            h.remove()
            dp = (torch.sigmoid(l2) - torch.sigmoid(base_logits)).abs().mean(0)
            out["summary_ablation"] = {"button_prob_change": [round(float(v), 3) for v in dp],
                                       "stick_mean_change": round(float((m2 - base_mean).abs().mean()), 3),
                                       "greedy_button_flips": round(float(((l2 > 0) != (base_logits > 0)).float().mean()), 3)}
        report["checkpoints"][os.path.basename(path)] = out
        s = out["layers"]
        print(f"{os.path.basename(path):28} {out['global_step']:>10}  " + "  ".join(
            f"{k.split('.')[0][:6]}.{k.split('.')[-1]} sat{v['saturated']:.2f} gp{v['grad_pass']:.2f} "
            f"rk{v['eff_rank']:.1f}/{v['width']} 1s{v['one_sign_units']:.2f}" for k, v in s.items()), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
