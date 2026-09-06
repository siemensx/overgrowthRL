#!/usr/bin/env python3
"""Watch a trained checkpoint play, live, rendered, at real (1x) speed.

All of training uses --disable-rendering plus a real 100x game-speed
multiplier (Source/Main/engine.cpp::SetGameSpeed literally scales
game_timer.time_scale -- confirmed by reading it, not assumed) so headless
throughput isn't gated on real-world wall-clock time. This script does the
opposite on both counts: no --disable-rendering (a real window opens) and
time_scale_mult=1 (normal game speed) -- exactly what "watch a checkpoint
play in normal non-sped-up rendered time" (as asked) requires.

Also writes a "ghost" action-trace CSV compatible with RLAction::LoadScript
(--rl-action-script) as it plays -- so a specific watched episode can be
replayed later, exactly, without needing this checkpoint or PyTorch at all,
just the engine and the CSV. See replay_ghost.py.

Episode length is capped by WALL-CLOCK time (--max-episode-real-seconds), not
a physics-tick count. This matters: Engine::Update() caps physics catch-up at
_max_steps_per_frame=4 ticks per rendered frame (Source/Main/engine.cpp), and
--benchmark's ManualStepCount() (which bypasses that cap entirely, driving
physics as fast as the CPU allows) is specifically NOT used here -- this mode
exists to render, which --benchmark forbids. So once real rendering plus the
per-tick Python round-trip can't sustain 120Hz -- routine on a fanless
machine with a full 3D scene -- game-time falls behind real-time, and a
tick-count cap meant to mean "~15 seconds" can silently mean several minutes
instead. Found by a human actually watching it happen, not predicted in
advance.

Actions are the policy's deterministic mode (continuous: tanh(mean), no
sampling noise; discrete: argmax/>0.5, no Bernoulli sampling) rather than a
stochastic rollout sample -- watching what the checkpoint actually believes
is best, not one noisy draw from its exploration distribution.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Tools/rl
from env import OvergrowthEnv, ACTION_DIM
from shm_env import ShmWaitTimeout
from obs_schema import DEFAULT_LAYOUT
from run_config import load_run_env_config

from policy import ActorCritic
from normalize import ObservationNormalizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path saved by train.py/train_vec.py's _save_checkpoint")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--from-run", default=None,
                    help="OGRL-20260817-028 Sec8.1/Sec8.6: run id under --runs-root (default Tools/rl/runs) -- "
                         "reads --level/--frame-stack/--act-period from that run's own run.json via "
                         "run_config.load_run_env_config() instead of trusting CLI defaults that may not match "
                         "what the checkpoint was actually trained with. Explicit --level/--frame-stack/--act-period "
                         "flags below still override individually if given. This is the shared helper the eval "
                         "harness and the dashboard's replay launcher also use -- two tools (this one and "
                         "diagnose_checkpoint.py) had already broken on the same 'defaults to a stale config "
                         "instead of the run's own manifest' bug before it existed.")
    p.add_argument("--runs-root", default=None, help="only used with --from-run; defaults to Tools/rl/runs under --repo-root")
    p.add_argument("--level", default=None, help="default: from --from-run's manifest, else arenas/oval_arena.xml")
    p.add_argument("--shm-name", default=None,
                   help="default: a fresh name per invocation. A FIXED name means a second watch "
                        "run attaches to the shm segment the first one left behind, which hangs on "
                        "the skybox or times out after 120s (DEAD_ENDS.md's orphaned-semaphore trap).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-episode-real-seconds", type=float, default=20.0,
                    help="wall-clock cap per episode, not a physics-tick count -- see module docstring for why "
                         "a tick-count cap is the wrong tool here: rendering + the Python round-trip can make "
                         "game-time fall well behind real-time, so a fixed tick count has no reliable real-world duration")
    p.add_argument("--max-episode-steps", type=int, default=6000, help="backup cap in case something stalls without the wall-clock cap tripping (e.g. a hung engine); should rarely bind")
    p.add_argument("--frame-stack", type=int, default=None, help="must match what the checkpoint was trained with; default: from --from-run's manifest, else 1")
    p.add_argument("--act-period", type=int, default=None,
                    help="OGRL-20260817-028 Sec8.1: this flag did not exist before -- watch.py always ran at "
                         "act_period=1 (120Hz) regardless of what the checkpoint was trained at, so any run8/run9+ "
                         "checkpoint (trained at act_period=4, 30Hz) fed a decision every physics tick instead of "
                         "every 4th, a real behavioral mismatch, not just a cosmetic speed difference. Default: "
                         "from --from-run's manifest, else 1 (the old implicit behavior, unchanged when --from-run "
                         "is not given).")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--ghost-dir", default=None, help="directory to write replayable action-trace CSVs (default: alongside the checkpoint)")
    p.add_argument("--no-ghost", action="store_true", help="skip writing ghost CSVs")
    # Watching the agent fight ONE opponent tells you almost nothing about how it
    # handles being outnumbered, which is the whole multi-opponent curriculum.
    # These reach the level script through the same set_rl_* path training uses.
    p.add_argument("--opponents", type=int, default=1, help="1, 2 or 3 -- needs a level with the generator's game_type 3/4 spawn groups (the t_train_* arenas have them; oval falls back to 1v1)")
    p.add_argument("--difficulty", type=float, default=None, help="0..1 opponent skill; default: whatever the level script picks")
    args = p.parse_args()
    if args.from_run:
        cfg = load_run_env_config(args.repo_root, args.from_run, runs_root=args.runs_root)
        if args.level is None:
            args.level = cfg["level"]
        if args.frame_stack is None:
            args.frame_stack = cfg["frame_stack"]
        if args.act_period is None:
            args.act_period = cfg["act_period"]
    args.level = args.level or "arenas/oval_arena.xml"
    args.frame_stack = args.frame_stack if args.frame_stack is not None else 1
    args.act_period = args.act_period if args.act_period is not None else 1
    return args


def deterministic_action(policy: ActorCritic, obs_tensor: torch.Tensor) -> np.ndarray:
    """Mirrors ActorCritic._distributions() but takes the mode instead of
    sampling -- tanh(mean) for the continuous head (the mode of a
    tanh-squashed Gaussian is the tanh of the underlying mode, since tanh is
    monotonic), logits > 0 (equivalently p > 0.5) for the discrete heads."""
    with torch.no_grad():
        features = policy.actor_trunk(policy._features(obs_tensor))
        continuous_action = torch.tanh(policy.continuous_mean(features))
        discrete_action = (policy.discrete_logits(features) > 0.0).float()
        return torch.cat([continuous_action, discrete_action], dim=-1).squeeze(0).cpu().numpy()


def _raise_keyboard_interrupt(signum, frame):
    # Same fix as train.py/train_vec.py's -- SIGTERM bypasses the existing
    # `finally: env.close()` below, which would otherwise leave the engine
    # window running orphaned. See train_vec.py's version for the incident.
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = parse_args()
    device = torch.device(args.device)
    layout = DEFAULT_LAYOUT

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_dim = layout.total_floats * max(1, args.frame_stack)
    policy = ActorCritic(layout, frame_stack=args.frame_stack).to(device)
    # Explicit-metadata guard (Sec5) -- see train.py's _save_checkpoint comment.
    ckpt_total_floats = checkpoint.get("layout_total_floats")
    ckpt_frame_stack = checkpoint.get("frame_stack")
    if ckpt_total_floats != layout.total_floats or ckpt_frame_stack != args.frame_stack:
        raise ValueError(
            f"this checkpoint has layout_total_floats={ckpt_total_floats}, frame_stack={ckpt_frame_stack} "
            f"(missing/None means it predates the Sec5 entity-encoder architecture and cannot be driven at all "
            f"by this build), but the current engine build + obs_schema.py + --frame-stack {args.frame_stack} "
            f"produce layout_total_floats={layout.total_floats}. This checkpoint cannot be driven live against "
            f"the current engine/policy architecture; it would need either a matching engine+policy build or a "
            f"conversion step, neither of which exists. Not a bug in this script -- pass --frame-stack (or "
            f"--from-run) matching what the checkpoint was actually trained with."
        )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    obs_normalizer = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
    print(f"loaded checkpoint from global_step={checkpoint['global_step']}")

    ghost_dir = Path(args.ghost_dir) if args.ghost_dir else Path(args.checkpoint).parent / "ghosts"
    ghost_dir.mkdir(parents=True, exist_ok=True)

    if args.shm_name is None:
        args.shm_name = f"/ogrl_w{os.getpid() % 100000:05d}"

    env = OvergrowthEnv(
        repo_root=args.repo_root, level=args.level, shm_name=args.shm_name, seed=args.seed,
        layout=layout, frame_stack=args.frame_stack, render=True, time_scale_mult=1, act_period=args.act_period,
    )
    try:
        for episode in range(args.episodes):
            kw = {"opponents": args.opponents}
            if args.difficulty is not None:
                kw["difficulty"] = args.difficulty
            raw_obs = env.reset(seed=args.seed + episode, **kw)
            if episode == 0:
                # env.py's first reset() only consumes the engine's own initial
                # observation and sends no scenario, so a single reset shows 1v1
                # whatever was asked for. The second one is the real request --
                # but under RENDER mode the level load is far slower than
                # headless, and Engine::ResetRLTrainingScenario refuses until its
                # baseline is captured. Retry rather than assume it is ready.
                for attempt in range(12):
                    try:
                        raw_obs = env.reset(seed=args.seed + episode, **kw)
                        break
                    except (RuntimeError, ShmWaitTimeout) as exc:
                        if attempt == 11:
                            raise
                        print(f"  scenario reset not ready yet ({exc.__class__.__name__}), retrying {attempt+1}/12")
                        time.sleep(2.0)
            obs = obs_normalizer.normalize(raw_obs, update=False)  # frozen stats at watch time, not still-learning
            ghost_rows = []
            episode_reward = 0.0
            episode_start = time.monotonic()
            won = False
            kos = 0
            done = False
            step = 0
            for step in range(args.max_episode_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = deterministic_action(policy, obs_tensor)
                if not args.no_ghost:
                    ghost_rows.append([
                        step, float(action[0]), float(action[1]),
                        int(action[2] > 0.5), int(action[3] > 0.5), int(action[4] > 0.5),
                        int(action[5] > 0.5), int(action[6] > 0.5), int(action[7] > 0.5),
                    ])
                raw_obs, reward, done, info = env.step(action)
                episode_reward += reward
                obs = obs_normalizer.normalize(raw_obs, update=False)
                rc = info["reward_components"]
                _k = rc.get("hostile_kos_this_step")
                kos += int(round(_k)) if _k is not None else (1 if rc.get("opponent_knockout", 0) > 0 else 0)
                won = kos >= max(1, args.opponents)   # one KO of three is not a win
                if done or won:
                    break
                if time.monotonic() - episode_start > args.max_episode_real_seconds:
                    break  # wall-clock cap, not a tick-count one -- see module docstring
            outcome = "WON" if won else ("LOST" if done else "timed out")
            print(f"episode {episode}: steps={step + 1} real_seconds={time.monotonic() - episode_start:.1f} reward={episode_reward:.2f} {outcome}")

            if not args.no_ghost:
                ghost_path = ghost_dir / f"ghost_step{checkpoint['global_step']}_ep{episode}_{int(time.time())}.csv"
                with open(ghost_path, "w", newline="") as f:
                    # RLAction::LoadScript's parser (Source/Main/rl_action.cpp) only
                    # skips a line as a comment if it starts with '#' -- a plain CSV
                    # header row gets parsed as data and crashes on stoull("step").
                    # Confirmed the hard way: this exact bug reached a live run
                    # before being caught (research-log OGRL-20260816-014).
                    f.write("# step,move_x,move_y,jump,crouch,attack,grab,drop,walk\n")
                    writer = csv.writer(f)
                    writer.writerows(ghost_rows)
                print(f"  ghost saved: {ghost_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
