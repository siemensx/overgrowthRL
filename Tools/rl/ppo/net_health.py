"""OGRL-20261007-014: keep the network learnable, and watch it while it trains.

run38's attention summary (Linear -> tanh) went from 30% to 60% saturated between 5M and 75M steps as its
pre-activations grew, cutting the learning signal through it to ~15%; the actor trunk's effective
dimensionality halved (8 -> 4) over the same span (OGRL-20261007-013). The standard counter to both is
decoupled weight decay (Loshchilov & Hutter 2019; with LayerNorm it is the combination Lyle et al. 2024
found most robust against plasticity loss). It is applied here as an optimizer step hook rather than by
switching to AdamW, so checkpoints, optimizer state and param groups are unchanged and a resume needs no
surgery: after every optimizer step, each decayed weight is multiplied by (1 - lr * wd).

Only matrices decay (2-D weights: Linear and attention projections). Biases, LayerNorm gains and the
state-independent log-std are left alone -- decaying log-std would quietly change exploration.
"""
from __future__ import annotations

import torch

SUMMARY_WEIGHT = "entity_encoder.out.1.weight"   # the Linear feeding the attention summary's tanh


def install_weight_decay(policy, optimizer, wd: float, wd_summary: float | None = None) -> list:
    """Register the decay hook; returns [(name, rate)] for logging. wd <= 0 and no summary rate -> no-op."""
    wd_summary = wd if wd_summary is None else wd_summary
    decay = []
    for name, p in policy.named_parameters():
        if p.dim() != 2 or not p.requires_grad:
            continue
        rate = wd_summary if name == SUMMARY_WEIGHT else wd
        if rate > 0:
            decay.append((name, p, float(rate)))
    if not decay:
        return []

    def hook(opt, _args, _kwargs):
        lr = float(opt.param_groups[0]["lr"])
        with torch.no_grad():
            for _n, p, rate in decay:
                p.mul_(1.0 - lr * rate)

    optimizer.register_step_post_hook(hook)
    return [(n, r) for n, _p, r in decay]


def _eff_dims(a: torch.Tensor) -> float:
    c = torch.cov(a.T.double())
    ev = torch.linalg.eigvalsh(c).clamp(min=0)
    s = ev.sum()
    return float(s * s / (ev * ev).sum()) if s > 0 else 0.0


def health_stats(policy, obs: torch.Tensor, max_n: int = 512) -> dict:
    """Saturation, gradient pass-through (mean 1 - tanh^2) and effective dimensions of the attention summary
    and the actor trunk's tanh layers, on up to max_n of the given (already normalised) observations."""
    x = obs[torch.randperm(obs.shape[0])[:max_n]] if obs.shape[0] > max_n else obs
    want = {}
    for name, mod in policy.named_modules():
        if isinstance(mod, torch.nn.Tanh) and (name.startswith("actor_trunk") or name.startswith("entity_encoder.out")):
            want[name] = mod
    acts, hooks = {}, []

    def rec(name):
        def h(_m, _i, o):          # returns None: must not replace the output
            acts.setdefault(name, []).append(o.detach())
        return h

    for name, mod in want.items():
        hooks.append(mod.register_forward_hook(rec(name)))
    try:
        with torch.no_grad():
            policy.actor_trunk(policy._features(x))
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for name, lst in acts.items():
        a = torch.cat([v.reshape(-1, v.shape[-1]) for v in lst]).float()
        key = "summary" if name.startswith("entity_encoder") else name.replace("actor_trunk.", "trunk")
        out[key] = {"saturated": round(float((a.abs() > 0.99).float().mean()), 4),
                    "grad_pass": round(float((1 - a * a).mean()), 4),
                    "eff_dims": round(_eff_dims(a), 2)}
    w = dict(policy.named_parameters()).get(SUMMARY_WEIGHT)
    if w is not None:
        out["summary_weight_norm"] = round(float(w.detach().norm()), 3)
    return out
