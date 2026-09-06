#!/usr/bin/env python3
"""Is the agent being hit by opponents it cannot see?

`Source/Main/rl_observation.cpp`'s `InFovCone` ends in `local_dir.z() > 0.0f`,
so a hostile BEHIND the agent is not dimmed in the observation -- it is absent,
`valid=0`, all 33 floats zero, indistinguishable from "no such character". The
policy's only history is the frame stack: frame_stack * act_period / 120 Hz,
i.e. 0.133 s at the run21 settings. There is no recurrent core.

That predicts the multi-opponent curve (1v1 ~0.7, 1v2 ~0.6, 1v3 ~0.3), but a
prediction is not evidence. This measures it directly, with the REAL policy
(random actions would spin the agent and bias visibility badly):

  * how often each opponent is visible, per opponent count;
  * how long contiguous "no hostile visible" intervals last, in seconds,
    against the 0.133 s the frame stack can actually remember;
  * of every decision on which the agent LOST HEALTH, what fraction had no
    hostile visible anywhere in the memory window -- i.e. it was hit by
    something its state representation did not contain.

The last number is the decision rule. If it is small, memory is not the
multi-opponent bottleneck and recurrence/entity-memory would be wasted work.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from collections import Counter, deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv, ACTION_DIM  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

DT = 4 / 120.0  # one decision in seconds at act_period=4


def _hostiles_visible(raw: np.ndarray) -> set[int]:
    """IDs of currently-visible, awake, non-ally entities."""
    out = set()
    for slot in range(L.max_visible_entities):
        e = L.entity_field(raw, slot)
        if e["valid"] and not e["is_ally"] and e["knocked_out"][0] > 0.5:
            out.add(e["id"])
    return out


def _self_health(raw: np.ndarray) -> float:
    return float(raw[L.TEMP_HEALTH]) + float(raw[L.BLOOD_HEALTH])


def run(policy, obs_norm, frame_stack, level, opponents, decisions, shm, seed):
    env = OvergrowthEnv(repo_root=os.getcwd(), level=level, shm_name=shm, seed=seed,
                        act_period=4, frame_stack=frame_stack)
    env.reset(seed=seed, difficulty=0.6, opponents=opponents)
    obs = env.reset(seed=seed + 1, difficulty=0.6, opponents=opponents)

    window = max(1, frame_stack)          # what the policy can actually remember
    recent = deque(maxlen=window)
    vis_hist: Counter = Counter()
    blind_run, blind_runs = 0, []
    dmg_events = dmg_blind = 0
    prev_health = None
    total = 0

    for _ in range(decisions):
        with torch.no_grad():
            x = torch.as_tensor(obs_norm.normalize(obs, update=False), dtype=torch.float32).unsqueeze(0)
            act, _, _, _ = policy.get_action_and_value(x)
        obs, _r, done, info = env.step(act.squeeze(0).numpy())
        raw = env._prev_values                     # single raw frame, pre-stack
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=np.float32)

        vis = _hostiles_visible(raw)
        recent.append(vis)
        vis_hist[len(vis)] += 1
        total += 1

        health = _self_health(raw)
        if prev_health is not None and health < prev_health - 1e-4:
            dmg_events += 1
            if not any(recent):                    # nothing visible anywhere in the window
                dmg_blind += 1
        prev_health = health

        if not vis:
            blind_run += 1
        else:
            if blind_run:
                blind_runs.append(blind_run)
            blind_run = 0

        if done:
            if blind_run:
                blind_runs.append(blind_run)
            blind_run = 0
            recent.clear()
            prev_health = None
            obs = env.reset(seed=seed + 100 + total, difficulty=0.6, opponents=opponents)
    if blind_run:
        blind_runs.append(blind_run)
    env.close()

    srt = sorted(blind_runs)
    return {
        "opponents": opponents, "decisions": total,
        "mean_visible": sum(k * v for k, v in vis_hist.items()) / max(1, total),
        "sees_none_pct": 100.0 * vis_hist[0] / max(1, total),
        "sees_all_pct": 100.0 * vis_hist[opponents] / max(1, total),
        "blind_median_s": (st.median(srt) * DT) if srt else 0.0,
        "blind_p90_s": (srt[int(0.9 * (len(srt) - 1))] * DT) if srt else 0.0,
        "blind_max_s": (max(srt) * DT) if srt else 0.0,
        "dmg_events": dmg_events,
        "dmg_blind_pct": 100.0 * dmg_blind / max(1, dmg_events),
        "hist": dict(sorted(vis_hist.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--decisions", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shm-prefix", default="/ogrl_vis")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    policy = ActorCritic(L, frame_stack=fs)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    obs_norm = ObservationNormalizer(L, frame_stack=fs)
    obs_norm.load_state_dict(ck["obs_normalizer"])

    print(f"checkpoint global_step={ck['global_step']:,}  frame_stack={fs}")
    print(f"policy memory window = {fs} frames x 4 ticks / 120 Hz = {fs * DT:.3f} s\n")
    print(f"{'opp':>3} {'mean vis':>9} {'sees none':>10} {'sees all':>9} "
          f"{'blind med':>10} {'blind p90':>10} {'blind max':>10} {'hits while blind':>17}")
    for opp in a.opponents:
        r = run(policy, obs_norm, fs, a.level, opp, a.decisions,
                f"{a.shm_prefix}{opp}_{os.getpid()}", a.seed)
        print(f"{r['opponents']:>3} {r['mean_visible']:>9.2f} {r['sees_none_pct']:>9.1f}% "
              f"{r['sees_all_pct']:>8.1f}% {r['blind_median_s']:>9.2f}s {r['blind_p90_s']:>9.2f}s "
              f"{r['blind_max_s']:>9.2f}s {r['dmg_blind_pct']:>15.1f}%  (n={r['dmg_events']})")
        print(f"    visible-count histogram: {r['hist']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
