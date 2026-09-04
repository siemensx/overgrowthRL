#!/usr/bin/env python3
"""One-off diagnostic (not part of the training pipeline): runs a checkpoint
headless for N episodes and reports MEAN REWARD COMPONENTS, not just total
reward -- reward.py's own docstring says to always inspect components when
tuning weights, and train_vec.py's CSV logger currently only logs the total,
which is dominated by the flat per-step time_cost over long (~800-step)
episodes and hides whatever the combat-specific terms (damage_dealt,
opponent_knockout, friendly_fire, ragdoll_time) are actually doing. Written
to answer one question directly: is run5 actually learning combat, or is the
flat aggregate mean_episode_reward masking real (if small) progress there.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ppo"))  # Tools/rl/ppo -- for policy.py, watch.py
sys.path.insert(0, str(Path(__file__).resolve().parent))  # Tools/rl -- for env.py, obs_schema.py, normalize.py
from env import OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT
from policy import ActorCritic
from normalize import ObservationNormalizer
from run_config import load_run_env_config
from watch import deterministic_action  # reuse the exact same deterministic-mode action fn


def main():
    p = argparse.ArgumentParser(description=__doc__ + "\n\nSuperseded for anything beyond a quick sanity check by "
                                 "Tools/rl/evaluate.py (OGRL-20260817-028 Sec6.1), which adds a random-policy "
                                 "control, difficulty-band sweeps, and Wilson CIs -- kept working here mainly so "
                                 "it doesn't bit-rot into a third broken tool.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--from-run", default=None, help="reads --level/--frame-stack/--act-period from this run id's own run.json -- see run_config.py")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--level", default=None)
    p.add_argument("--frame-stack", type=int, default=None)
    p.add_argument("--act-period", type=int, default=None)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-episode-steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shm-name", default=None,
                    help="defaults to a fresh, timestamp-unique name -- repeatedly reusing a fixed shm name "
                         "across separate invocations in one session was observed to hang on the 2nd+ use today "
                         "(worked the first time, then stalled every time after); not fully root-caused, but a "
                         "unique name per run avoids it, so that's the default now rather than a fixed one.")
    args = p.parse_args()
    if args.from_run:
        cfg = load_run_env_config(args.repo_root, args.from_run, runs_root=args.runs_root)
        args.level = args.level if args.level is not None else cfg["level"]
        args.frame_stack = args.frame_stack if args.frame_stack is not None else cfg["frame_stack"]
        args.act_period = args.act_period if args.act_period is not None else cfg["act_period"]
    args.level = args.level or "arenas/oval_arena.xml"
    args.frame_stack = args.frame_stack if args.frame_stack is not None else 1
    args.act_period = args.act_period if args.act_period is not None else 1
    shm_name = args.shm_name or f"/ogrl_diag{int(time.time())}"

    device = torch.device("cpu")
    layout = DEFAULT_LAYOUT
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = ActorCritic(layout, frame_stack=args.frame_stack).to(device)
    ckpt_total_floats = checkpoint.get("layout_total_floats")
    if ckpt_total_floats != layout.total_floats or checkpoint.get("frame_stack") != args.frame_stack:
        raise ValueError(
            f"checkpoint has layout_total_floats={ckpt_total_floats}, frame_stack={checkpoint.get('frame_stack')} "
            f"(missing/None means it predates the Sec5 entity-encoder architecture); current build + "
            f"--frame-stack {args.frame_stack} expects layout_total_floats={layout.total_floats}. Pass --from-run "
            f"or matching --frame-stack, or use a build/checkpoint pair that actually matches."
        )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    obs_normalizer = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
    print(f"checkpoint global_step={checkpoint['global_step']}, running {args.episodes} headless episodes...")

    env = OvergrowthEnv(repo_root=args.repo_root, level=args.level, shm_name=shm_name, seed=args.seed,
                         layout=layout, frame_stack=args.frame_stack, act_period=args.act_period, render=False)
    totals = defaultdict(list)
    outcomes = []
    lengths = []
    try:
        for ep in range(args.episodes):
            raw_obs = env.reset(seed=args.seed + ep)
            obs = obs_normalizer.normalize(raw_obs, update=False)
            ep_components = defaultdict(float)
            won = False
            done = False
            step = 0
            for step in range(args.max_episode_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = deterministic_action(policy, obs_tensor)
                raw_obs, reward, done, info = env.step(action)
                for k, v in info["reward_components"].items():
                    ep_components[k] += v
                obs = obs_normalizer.normalize(raw_obs, update=False)
                won = info["reward_components"]["opponent_knockout"] > 0
                if done or won:
                    break
            outcomes.append("WON" if won else ("LOST" if done else "TIMEOUT"))
            lengths.append(step + 1)
            for k, v in ep_components.items():
                totals[k].append(v)
            print(f"  episode {ep}: steps={step + 1} outcome={outcomes[-1]} " +
                  " ".join(f"{k}={v:.2f}" for k, v in ep_components.items()))
    finally:
        env.close()

    print("\n=== summary over", args.episodes, "episodes ===")
    print(f"outcomes: WON={outcomes.count('WON')} LOST={outcomes.count('LOST')} TIMEOUT={outcomes.count('TIMEOUT')}")
    print(f"mean episode length: {np.mean(lengths):.1f}")
    for k in sorted(totals):
        vals = np.array(totals[k])
        print(f"  {k}: mean={vals.mean():.3f} std={vals.std():.3f} n_nonzero={np.count_nonzero(vals)}/{len(vals)}")


if __name__ == "__main__":
    main()
