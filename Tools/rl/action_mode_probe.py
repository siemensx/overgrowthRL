#!/usr/bin/env python3
"""Why is the deterministic policy worse than the stochastic one?

run21 at 277M reads 0.79 stochastic and 0.33 deterministic on 1v3 unarmed.
That is backwards -- greedy normally beats sampling by a few points, because
it drops the exploration noise and keeps the best guess.

The action space explains how it can invert. It is 2 continuous dims plus
SIX INDEPENDENT BERNOULLI buttons (jump, crouch, attack, grab, drop, walk).
watch.deterministic_action thresholds each button separately at p > 0.5. So
the "deterministic action" is the per-coordinate mode, and the per-coordinate
mode of a product of independent Bernoullis need not be an action the policy
ever actually plays. Six buttons at p = 0.4 each give a greedy action of
all-off, which sampling produces only 0.6^6 = 4.7% of the time.

If every attack-ish button sits below 0.5 while jump sits above it, the
greedy policy runs around and jumps forever and never once attacks -- which
is exactly what a human sees on screen -- while the sampled policy attacks
on a decent fraction of steps and wins.

This probe measures that directly. For each step it records the per-button
probability, the greedy button vector, and P(sampled action == greedy action)
under the policy's own distribution. Run it under BOTH drive modes: the two
policies visit different states, so the button statistics have to be read on
the states each one actually reaches.

The number to look at is p_greedy_joint. If it is tiny, the deterministic
policy is off-distribution -- it is playing an action the training run
essentially never took, so nothing it learned applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ppo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, ObsLayout
from policy import ActorCritic
from normalize import ObservationNormalizer

BUTTONS = ["jump", "crouch", "attack", "grab", "drop", "walk"]


def actor_params(policy: ActorCritic, obs_tensor: torch.Tensor):
    """Exactly watch.deterministic_action's path, but returning the raw
    parameters instead of only the thresholded action."""
    with torch.no_grad():
        features = policy.actor_trunk(policy._features(obs_tensor))
        mean = policy.continuous_mean(features)
        log_std = torch.clamp(policy.continuous_log_std, -5.0, 2.0)
        logits = policy.discrete_logits(features)
    return mean.squeeze(0), log_std, logits.squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--level", default="arenas/t_train_101.xml")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--act-period", type=int, default=4)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--opponents", type=int, default=3)
    p.add_argument("--armed-count", type=int, default=0)
    p.add_argument("--weapon-type", type=int, default=0)
    p.add_argument("--throw-aggression", type=float, default=1.0)
    p.add_argument("--species", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--drive", default="deterministic",
                   choices=["deterministic", "stochastic", "cont-only", "disc-only"],
                   help="which policy steps the env. The two mixed modes are the ablation that "
                        "localizes the greedy/sampled gap: 'cont-only' samples the movement head "
                        "and thresholds the buttons, 'disc-only' does the reverse. Whichever mixed "
                        "mode recovers the stochastic win rate is the head that the randomness "
                        "matters in.")
    p.add_argument("--shm-name", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    layout = ObsLayout()
    policy = ActorCritic(layout=layout, frame_stack=args.frame_stack).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    norm = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    norm.load_state_dict(ckpt["obs_normalizer"])
    global_step = ckpt.get("global_step", 0)

    env = OvergrowthEnv(
        repo_root=args.repo_root, level=args.level,
        shm_name=args.shm_name or f"/ogrl_probe{np.random.randint(1, 99999)}",
        seed=args.seed_base, layout=layout, frame_stack=args.frame_stack,
        act_period=args.act_period, render=False,
        throw_aggression_launch=args.throw_aggression,
    )

    probs_sum = np.zeros(6)
    greedy_on = np.zeros(6)
    p_joint = []
    n_greedy_pressed = defaultdict(int)
    cont_abs_sum = np.zeros(2)
    steps = 0
    outcomes = {"won": 0, "lost": 0, "timeout": 0}

    try:
        for ep in range(args.episodes):
            raw = env.reset(seed=args.seed_base + ep, soft=False, difficulty=args.difficulty,
                            opponents=args.opponents, weapons=0.0, species=args.species,
                            armed_count=args.armed_count, weapon_type=args.weapon_type,
                            throw_aggression=args.throw_aggression)
            obs = norm.normalize(raw, update=False)
            kos, need = 0, max(1, args.opponents)
            won = done = False
            for _ in range(args.max_episode_steps):
                t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mean, log_std, logits = actor_params(policy, t)
                pr = torch.sigmoid(logits).numpy()
                g = (pr > 0.5)
                # P(a sampled button vector equals the greedy one)
                pj = float(np.prod(np.where(g, pr, 1.0 - pr)))

                probs_sum += pr
                greedy_on += g.astype(float)
                p_joint.append(pj)
                n_greedy_pressed[int(g.sum())] += 1
                cont_abs_sum += np.abs(torch.tanh(mean).numpy())
                steps += 1

                det_cont = torch.tanh(mean).numpy()
                det_disc = g.astype(np.float32)
                if args.drive == "deterministic":
                    action = np.concatenate([det_cont, det_disc])
                else:
                    with torch.no_grad():
                        a, *_ = policy.get_action_and_value(t)
                    smp = a.squeeze(0).numpy()
                    if args.drive == "stochastic":
                        action = smp
                    elif args.drive == "cont-only":     # sampled movement, greedy buttons
                        action = np.concatenate([smp[:2], det_disc])
                    else:                                # disc-only: greedy movement, sampled buttons
                        action = np.concatenate([det_cont, smp[2:]])

                raw, _r, done, info = env.step(action)
                obs = norm.normalize(raw, update=False)
                rc = info["reward_components"]
                sk = rc.get("hostile_kos_this_step")
                if sk is None:
                    won = rc.get("opponent_knockout", 0.0) > 0.0
                else:
                    kos += int(round(sk))
                    won = kos >= need
                if done or won:
                    break
            outcomes["won" if won else ("lost" if done else "timeout")] += 1
    finally:
        env.close()

    mean_p = probs_sum / max(1, steps)
    greedy_rate = greedy_on / max(1, steps)
    pj = np.array(p_joint)
    result = {
        "global_step": int(global_step), "drive": args.drive, "steps": steps,
        "episodes": args.episodes, "outcomes": outcomes,
        "opponents": args.opponents, "difficulty": args.difficulty,
        "buttons": {b: {"mean_p_sampled": float(mean_p[i]),
                        "greedy_press_rate": float(greedy_rate[i])}
                    for i, b in enumerate(BUTTONS)},
        "p_greedy_joint": {"mean": float(pj.mean()), "median": float(np.median(pj)),
                           "p10": float(np.percentile(pj, 10)),
                           "p90": float(np.percentile(pj, 90))},
        "greedy_buttons_pressed_hist": {str(k): v for k, v in sorted(n_greedy_pressed.items())},
        "continuous_abs_mean": [float(x) for x in cont_abs_sum / max(1, steps)],
        "continuous_log_std": [float(x) for x in log_std.numpy()],
    }

    print(f"\ncheckpoint global_step={global_step:,}   drive={args.drive}   "
          f"{steps} steps over {args.episodes} eps   {outcomes}")
    print(f"{'button':>8}  {'P(press) sampled':>17}  {'greedy presses':>15}")
    for i, b in enumerate(BUTTONS):
        flag = "  <-- greedy never presses this" if greedy_rate[i] < 0.01 and mean_p[i] > 0.15 else ""
        print(f"{b:>8}  {mean_p[i]:>17.3f}  {greedy_rate[i]:>15.3f}{flag}")
    print(f"\nP(sampled action == greedy action): mean {pj.mean():.4f}  median {np.median(pj):.4f}")
    print(f"greedy buttons pressed per step: {dict(sorted(n_greedy_pressed.items()))}")
    print(f"continuous |tanh(mean)|: {result['continuous_abs_mean']}  log_std {result['continuous_log_std']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
