#!/usr/bin/env python3
"""PPO training loop for the Overgrowth RL environment.

Single-environment (non-vectorized) for this first working version --
correctness and a real end-to-end proof came first; running N parallel
engine workers (the infra already exists: worker_pool.py's N-active+K-standby
pool, Stage 4) and batching their rollouts together is the natural next
performance step, not built here yet, flagged rather than silently assumed.

Implementation choices follow the widely-cited "37 Implementation Details of
PPO" reference (Engstrom, Ilyas et al.) and Schulman et al. 2017/2016:
orthogonal init with PPO's standard gains (policy.py), observation and
reward-scale running normalization (normalize.py), GAE with correct
truncation-vs-termination handling (buffer.py), advantage normalization per
minibatch, clipped surrogate + clipped value loss, entropy bonus, gradient
norm clipping, linear learning-rate annealing, and KL-based early stopping
within an epoch (a safety net against a single destructively large update
wasting an expensive, slow-to-collect rollout -- collection here is bounded
by the real engine's physics rate and the shm round-trip, not by GPU
throughput, so a wasted rollout is comparatively much more costly than in a
typical vectorized-simulator PPO setup).
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Tools/rl, for env/obs_schema/reward/curriculum
from env import OvergrowthEnv, ACTION_DIM
from obs_schema import DEFAULT_LAYOUT
from curriculum import Curriculum

from policy import ActorCritic
from buffer import RolloutBuffer
from normalize import ObservationNormalizer, RewardNormalizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--level", default="arenas/oval_arena.xml")
    p.add_argument("--shm-name", default="/ogrl_ppo0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=20_000)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per PPO update")
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--value-clip-coef", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=0.02, help="stop an update's remaining epochs early past this approx_kl")
    p.add_argument("--max-episode-steps", type=int, default=1200, help="~10s at 120Hz; forces a truncation reset")
    p.add_argument("--frame-stack", type=int, default=1, help="concatenate the last N raw observations so the (non-recurrent) policy can perceive trends, not just instantaneous state; 1 = off")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--log-path", default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--checkpoint-every-updates", type=int, default=10)
    return p.parse_args()


def _raise_keyboard_interrupt(signum, frame):
    # SIGTERM's default action is immediate termination, bypassing the
    # existing `finally: env.close()` below entirely -- unlike Ctrl+C
    # (SIGINT), which Python already turns into KeyboardInterrupt. Same fix
    # as train_vec.py's, applied here for consistency even though this
    # script's own engine is a single process (still worth not leaving
    # orphaned on a plain `kill <pid>`). See train_vec.py's version of this
    # function for the incident that motivated it.
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layout = DEFAULT_LAYOUT

    log_path = Path(args.log_path) if args.log_path else Path(args.repo_root) / "Tools/rl/ppo/runs" / f"{int(time.time())}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "global_step", "update", "mean_episode_reward", "mean_episode_length", "episodes_completed",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
        "curriculum_phase", "steps_per_second",
    ])
    print(f"logging to {log_path}")

    curriculum = Curriculum()
    env = OvergrowthEnv(
        repo_root=args.repo_root, level=args.level, shm_name=args.shm_name, seed=args.seed,
        layout=layout, reward_config=curriculum.reward_config_for_step(0), frame_stack=args.frame_stack,
    )
    obs_dim = env.observation_dim

    # OGRL-20260817-028 Sec5: ActorCritic/ObservationNormalizer now take the
    # layout + frame_stack directly (they need to know where the entity
    # region lives within each stacked frame), not just a flat obs_dim.
    policy = ActorCritic(layout, frame_stack=args.frame_stack).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_normalizer = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    reward_normalizer = RewardNormalizer(args.gamma)
    buffer = RolloutBuffer(args.n_steps, obs_dim, ACTION_DIM, device)

    global_step = 0
    update = 0
    episode_reward = 0.0
    episode_length = 0

    try:
        raw_obs = env.reset(seed=args.seed)
        obs = obs_normalizer.normalize(raw_obs)

        while global_step < args.total_timesteps:
            reward_config = curriculum.reward_config_for_step(global_step)
            env.set_reward_config(reward_config)

            episode_rewards_this_update = []
            episode_lengths_this_update = []
            collection_start = time.monotonic()

            for _ in range(args.n_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, _entropy, value = policy.get_action_and_value(obs_tensor)
                action_np = action.squeeze(0).cpu().numpy()

                raw_next_obs, reward, done, info = env.step(action_np)
                episode_reward += reward
                episode_length += 1
                global_step += 1

                # Python-side win condition: the engine's own `done` only
                # reflects the CONTROLLED character's knockout (rl_shm_transport.cpp
                # is deliberately reward/outcome-agnostic); ending the episode
                # on a landed knockout too keeps episodes bounded in both
                # directions and gives more, shorter, more-diverse episodes to
                # learn from rather than one open-ended wandering-around-a-
                # downed-opponent tail per life.
                won = info["reward_components"]["opponent_knockout"] > 0
                terminal = bool(done or won)
                truncated = (not terminal) and episode_length >= args.max_episode_steps

                if truncated:
                    # Time-limit bootstrap fix (Pardo et al. 2018, "Time
                    # Limits in Reinforcement Learning"): fold the truncated
                    # state's own value estimate into this step's reward
                    # before marking the transition terminal for GAE, so
                    # running out of the step cap isn't implicitly taught as
                    # equally bad as actually losing. Must read raw_next_obs
                    # BEFORE env.reset() overwrites it below. Not update=True
                    # on the normalizer -- this is a bootstrap probe, not a
                    # real collected transition, and shouldn't skew the
                    # running statistics.
                    truncated_obs = obs_normalizer.normalize(raw_next_obs, update=False)
                    truncated_obs_tensor = torch.as_tensor(truncated_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        bootstrap_value = policy.get_value(truncated_obs_tensor).item()
                    reward = reward + args.gamma * bootstrap_value

                normalized_reward = reward_normalizer.normalize(reward, terminal or truncated)
                # terminal_or_truncated as the buffer's stop-recursion flag is
                # correct for both cases now: true termination bootstraps
                # zero (nothing to add), and truncation already had its
                # future value folded into `reward` just above -- either way,
                # GAE should not recurse past this step.
                buffer.add(obs, action_np, log_prob.item(), value.item(), normalized_reward, terminal or truncated)

                if terminal or truncated:
                    episode_rewards_this_update.append(episode_reward)
                    episode_lengths_this_update.append(episode_length)
                    episode_reward = 0.0
                    episode_length = 0
                    raw_next_obs = env.reset(seed=args.seed + global_step)  # vary the seed across episodes

                obs = obs_normalizer.normalize(raw_next_obs)

            collection_seconds = max(1e-6, time.monotonic() - collection_start)

            with torch.no_grad():
                last_value = policy.get_value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
            batch = buffer.to_tensors(last_value, args.gamma, args.gae_lambda)
            buffer.reset()

            stats = ppo_update(policy, optimizer, batch, args)

            update += 1
            explained_var = _explained_variance(batch["values"].cpu().numpy(), batch["returns"].cpu().numpy())
            mean_ep_reward = float(np.mean(episode_rewards_this_update)) if episode_rewards_this_update else float("nan")
            mean_ep_length = float(np.mean(episode_lengths_this_update)) if episode_lengths_this_update else float("nan")
            steps_per_second = args.n_steps / collection_seconds

            print(
                f"update={update} step={global_step}/{args.total_timesteps} "
                f"phase={curriculum.phase_name(global_step)} "
                f"episodes={len(episode_rewards_this_update)} mean_reward={mean_ep_reward:.3f} mean_len={mean_ep_length:.1f} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} approx_kl={stats['approx_kl']:.4f} "
                f"explained_var={explained_var:.3f} sps={steps_per_second:.1f}"
            )
            log_writer.writerow([
                global_step, update, mean_ep_reward, mean_ep_length, len(episode_rewards_this_update),
                stats["policy_loss"], stats["value_loss"], stats["entropy"], stats["approx_kl"], stats["clip_fraction"],
                explained_var, curriculum.phase_name(global_step), steps_per_second,
            ])
            log_file.flush()

            if args.checkpoint_path and update % args.checkpoint_every_updates == 0:
                _save_checkpoint(args.checkpoint_path, policy, optimizer, obs_normalizer, reward_normalizer, global_step)
    finally:
        # OGRL-20260816-018: checkpointing only happened on the
        # update % checkpoint_every_updates == 0 boundary, so ANY exit
        # (natural completion, Ctrl+C, SIGTERM) between two boundaries lost
        # whatever training happened since the last one -- confirmed for
        # real on run5, which finished 7 updates (~57k steps) past its last
        # save with nothing capturing that final policy state. One more
        # unconditional save here (cheap, and a no-op duplicate write if the
        # loop happened to already be on a boundary) closes that gap for
        # every exit path, not just the happy one.
        if args.checkpoint_path:
            _save_checkpoint(args.checkpoint_path, policy, optimizer, obs_normalizer, reward_normalizer, global_step)
        env.close()
        log_file.close()


def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer, batch: dict, args) -> dict:
    n = batch["obs"].shape[0]
    indices = np.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "nan_skips": 0}
    n_minibatch_updates = 0
    stop_early = False

    for epoch in range(args.n_epochs):
        if stop_early:
            break
        np.random.shuffle(indices)
        for start in range(0, n, args.minibatch_size):
            mb_idx = indices[start:start + args.minibatch_size]
            mb_obs = batch["obs"][mb_idx]
            mb_actions = batch["actions"][mb_idx]
            mb_old_log_probs = batch["log_probs"][mb_idx]
            mb_old_values = batch["values"][mb_idx]
            mb_advantages = batch["advantages"][mb_idx]
            mb_returns = batch["returns"][mb_idx]

            # Per-minibatch advantage normalization (not per-buffer) --
            # standard practice, keeps the effective learning signal scale
            # consistent across minibatches of possibly different composition.
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _action, new_log_probs, entropy, new_values = policy.get_action_and_value(mb_obs, mb_actions)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

            surrogate1 = mb_advantages * ratio
            surrogate2 = mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Clipped value loss (PPO2-style): bounds how far the value
            # estimate can move in one update, the same spirit as the policy
            # clip, applied to the critic.
            value_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -args.value_clip_coef, args.value_clip_coef)
            value_loss_unclipped = (new_values - mb_returns).pow(2)
            value_loss_clipped = (value_clipped - mb_returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            # NaN/Inf guard (2026-08-17, found live): clip_grad_norm_ clips the
            # gradient NORM, it does NOT sanitize a gradient that is already
            # NaN/Inf -- clipping a NaN-valued norm is a no-op (still NaN), so
            # optimizer.step() would apply the NaN update to every parameter,
            # permanently corrupting the network from that step forward (every
            # later forward pass then raises "Expected parameter loc... invalid
            # values"). Confirmed on the smoke test's very first cold-start run
            # (Sec5's entity encoder, not present in run8/run9's flat-MLP
            # architecture): approx_kl spiked to 0.67 at update 102 (33x
            # target_kl, though nowhere near run9's own largest recorded spike
            # of 12.87, which did NOT corrupt that run's flat-MLP weights --
            # the new architecture is evidently more fragile to an extreme
            # update, root cause not fully isolated under time pressure), and
            # the very next forward pass produced all-NaN continuous-head
            # output. Skipping (not applying) a non-finite update is the
            # standard, well-established mitigation for exactly this failure
            # mode -- it costs one wasted minibatch's compute, not a
            # corrupted multi-hour run.
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                stats["nan_skips"] += 1
                optimizer.zero_grad()
                continue
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_loss.item()
            stats["approx_kl"] += approx_kl
            stats["clip_fraction"] += clip_fraction
            n_minibatch_updates += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                stop_early = True
                break

    for key in stats:
        if key != "nan_skips":  # a count, not something to average
            stats[key] /= max(1, n_minibatch_updates)
    return stats


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)


def _save_checkpoint(path: str, policy, optimizer, obs_normalizer, reward_normalizer, global_step: int) -> None:
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict(),
        "reward_normalizer": reward_normalizer.state_dict(),
        "global_step": global_step,
        # OGRL-20260817-028 Sec5: explicit architecture metadata, not a
        # weight-shape sniff -- the old check (actor_trunk.0.weight.shape[1]
        # == obs_dim) stopped being meaningful once actor_trunk's first layer
        # takes the entity-encoder+proprioception feature vector, not the raw
        # obs vector. A loader should compare these fields against its own
        # layout/frame_stack instead of inspecting a specific tensor's shape.
        "layout_total_floats": policy.layout.total_floats,
        "layout_max_visible_entities": policy.layout.max_visible_entities,
        "layout_local_geometry_rays": policy.layout.local_geometry_rays,
        "layout_action_history_steps": policy.layout.action_history_steps,
        "frame_stack": policy.frame_stack,
    }, path)


if __name__ == "__main__":
    main()
