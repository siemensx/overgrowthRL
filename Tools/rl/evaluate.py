#!/usr/bin/env python3
"""Evaluation harness (OGRL-20260817-028 Sec6.1) -- the tool that didn't
exist before this: run8/run9's eval/ directories were empty, and
diagnose_checkpoint.py couldn't even load a run8/run9 checkpoint (it
defaulted to the 10-character brawl at 120Hz/frame_stack=1, failing its own
obs_dim guard). This is also the tool the soft-reset policy-equivalence
check (Sec1.3) and the cold-start smoke test both run through.

Reads env configuration from the run's own run.json via run_config.py
(Sec8.1's shared helper) -- explicit --level/--frame-stack/--act-period
flags still override individually. Deterministic by default (matches
watch.py's deterministic_action); --stochastic samples instead. Always runs
a matched random-policy control on the SAME seeds unless --no-control --
this task's floor is ~41% at low difficulty, not 0%, so a win rate alone is
not interpretable (see -027 Sec1.2). Sweeps --difficulty-bands, reporting
each band separately (Sec3.2: never report a pooled win rate once a
difficulty curriculum is running) plus an overall pooled figure for
convenience. Reports Wilson 95% CIs (exact-ish, well-behaved at small n and
p near 0/1, unlike a normal approximation), normalized skill
(p_policy - p_random)/(1 - p_random), episode length, mean reward
components, and the Sec3.4/Sec8.3 conditional action statistics (via
emergence.EmergenceAccumulator, the exact same code the dashboard's
emergence panel uses during training -- one implementation, not two).
Writes runs/<run_id>/eval/<global_step>.json (telemetry.RunLogger.log_eval's
format) when --run-id is given; always prints a summary either way.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ppo"))  # Tools/rl/ppo -- policy.py, normalize.py, watch.py
sys.path.insert(0, str(Path(__file__).resolve().parent))  # Tools/rl -- env.py, obs_schema.py, run_config.py, emergence.py

from env import OvergrowthEnv, ACTION_DIM
from obs_schema import DEFAULT_LAYOUT
from run_config import load_run_env_config
from emergence import EmergenceAccumulator

from policy import ActorCritic
from normalize import ObservationNormalizer
from watch import deterministic_action  # reuse the exact deterministic-mode action fn, one implementation

# Held-out seed range: never used by ScenarioSampler/vec_env's training-time
# seed sequence, which is base_seed (typically 1) + a small monotonic
# counter -- any run reaching 900,000 episodes on one base_seed is not a
# scenario this project trains for. Matches the precedent already set by the
# manual held-out evaluation in research-artifacts/OGRL-20260817-027.
DEFAULT_SEED_BASE = 900_000


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return (None, None)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def random_action(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random policy control (-027's matched-baseline methodology):
    continuous axes uniform in [-1,1], discrete buttons independent p=0.5."""
    move = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    buttons = (rng.random(6) > 0.5).astype(np.float32)
    return np.concatenate([move, buttons])


def run_episodes(
    env: OvergrowthEnv, act_fn, episodes: int, seed_base: int, difficulty: float,
    opponents: int, weapons: float, species: int, max_episode_steps: int, obs_normalizer, layout,
) -> dict:
    outcomes = {"won": 0, "lost": 0, "timeout": 0}
    lengths = []
    component_totals = defaultdict(list)
    emergence = EmergenceAccumulator()
    for ep in range(episodes):
        seed = seed_base + ep
        raw_obs = env.reset(seed=seed, soft=False, difficulty=difficulty, opponents=opponents, weapons=weapons, species=species)
        obs = obs_normalizer.normalize(raw_obs, update=False) if obs_normalizer is not None else None
        ep_components = defaultdict(float)
        won = False
        # Hostile knockouts accumulated this episode. A win is EVERY opponent
        # down, not merely one: at N opponents "any knockout" makes being
        # outnumbered measure as EASIER, because N opponents give N times the
        # chances to land a KO. Observed live in run18 as win rates of
        # 0.71/0.74/0.81 for 1/2/3 before the trainer was fixed; this evaluator
        # carried exactly the same bug and would have reported the same
        # inflated numbers for any multi-opponent checkpoint.
        kos_this_episode = 0
        need_kos = max(1, int(opponents or 1))
        done = False
        step = 0
        for step in range(max_episode_steps):
            frame = raw_obs[-layout.total_floats:]
            entities = layout.all_entities(frame)
            action = act_fn(obs, frame)
            emergence.update(frame, entities, action, layout)
            raw_obs, reward, done, info = env.step(action)
            for k, v in info["reward_components"].items():
                ep_components[k] += v
            obs = obs_normalizer.normalize(raw_obs, update=False) if obs_normalizer is not None else None
            rc = info["reward_components"]
            step_kos = rc.get("hostile_kos_this_step")
            if step_kos is None:
                won = rc.get("opponent_knockout", 0.0) > 0.0     # pre-instrumentation engines
            else:
                kos_this_episode += int(round(step_kos))
                won = kos_this_episode >= need_kos
            if done or won:
                break
        outcomes["won" if won else ("lost" if done else "timeout")] += 1
        lengths.append(step + 1)
        for k, v in ep_components.items():
            component_totals[k].append(v)
    n = episodes
    win_rate = outcomes["won"] / n
    ci = wilson_ci(outcomes["won"], n)
    return {
        "episodes": n, "outcomes": outcomes, "win_rate": win_rate, "win_rate_ci95": list(ci),
        "episode_length_mean": float(np.mean(lengths)), "episode_length_median": float(np.median(lengths)),
        "reward_components_mean": {k: float(np.mean(v)) for k, v in component_totals.items()},
        "emergence": emergence.snapshot(),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--from-run", default=None, help="reads --level/--frame-stack/--act-period from this run id's own run.json")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--run-id", default=None, help="if given, writes runs/<run-id>/eval/<global_step>.json via telemetry.RunLogger.log_eval")
    p.add_argument("--level", default=None)
    p.add_argument("--frame-stack", type=int, default=None)
    p.add_argument("--act-period", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=900)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                    help=f"held-out seed range, never used for training (default {DEFAULT_SEED_BASE})")
    p.add_argument("--difficulty-bands", default="0.1,0.3,0.5,0.7,0.9,1.0")
    p.add_argument("--opponents", type=int, default=1)
    p.add_argument("--weapons", type=float, default=0.0)
    p.add_argument("--species", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="sample from the policy's distribution instead of its deterministic mode")
    p.add_argument("--no-control", action="store_true", help="skip the matched random-policy control (not recommended -- see module docstring)")
    p.add_argument("--shm-name", default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--out", default=None, help="write the full result JSON here regardless of --run-id")
    args = p.parse_args()
    if args.from_run:
        cfg = load_run_env_config(args.repo_root, args.from_run, runs_root=args.runs_root)
        args.level = args.level if args.level is not None else cfg["level"]
        args.frame_stack = args.frame_stack if args.frame_stack is not None else cfg["frame_stack"]
        args.act_period = args.act_period if args.act_period is not None else cfg["act_period"]
        if args.run_id is None:
            args.run_id = args.from_run
    args.level = args.level or "arenas/oval_arena_1v1_unarmed.xml"
    args.frame_stack = args.frame_stack if args.frame_stack is not None else 1
    args.act_period = args.act_period if args.act_period is not None else 1
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    layout = DEFAULT_LAYOUT
    shm_name = args.shm_name or f"/ogrl_eval{int(time.time()) % 100000}"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    global_step = checkpoint.get("global_step", -1)
    policy = ActorCritic(layout, frame_stack=args.frame_stack).to(device)
    ckpt_total_floats = checkpoint.get("layout_total_floats")
    if ckpt_total_floats != layout.total_floats or checkpoint.get("frame_stack") != args.frame_stack:
        raise ValueError(
            f"checkpoint has layout_total_floats={ckpt_total_floats}, frame_stack={checkpoint.get('frame_stack')} "
            f"(missing/None means it predates the Sec5 entity-encoder architecture); current build + "
            f"--frame-stack {args.frame_stack} expects layout_total_floats={layout.total_floats}."
        )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    obs_normalizer = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
    print(f"loaded checkpoint global_step={global_step}, evaluating {args.episodes} episodes/band, "
          f"level={args.level} frame_stack={args.frame_stack} act_period={args.act_period}")

    def policy_act(obs, _frame):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if args.stochastic:
            with torch.no_grad():
                action, *_ = policy.get_action_and_value(obs_tensor)
            return action.squeeze(0).cpu().numpy()
        return deterministic_action(policy, obs_tensor)

    rng = np.random.default_rng(1234)

    def control_act(_obs, _frame):
        return random_action(rng)

    env = OvergrowthEnv(
        repo_root=args.repo_root, level=args.level, shm_name=shm_name, seed=args.seed_base,
        layout=layout, frame_stack=args.frame_stack, act_period=args.act_period, render=False,
    )
    bands = [float(x) for x in args.difficulty_bands.split(",") if x.strip()]
    result = {"global_step": global_step, "checkpoint": args.checkpoint, "episodes": args.episodes,
              "seed_base": args.seed_base, "stochastic": args.stochastic, "level": args.level,
              "frame_stack": args.frame_stack, "act_period": args.act_period, "bands": [], "overall": None}
    try:
        all_policy_won, all_policy_n = 0, 0
        for d in bands:
            print(f"\n=== difficulty {d} ===")
            policy_result = run_episodes(env, policy_act, args.episodes, args.seed_base, d, args.opponents,
                                          args.weapons, args.species, args.max_episode_steps, obs_normalizer, layout)
            all_policy_won += policy_result["outcomes"]["won"]
            all_policy_n += policy_result["episodes"]
            band_entry = {"band": d, "policy": policy_result}
            if not args.no_control:
                control_result = run_episodes(env, control_act, args.episodes, args.seed_base, d, args.opponents,
                                               args.weapons, args.species, args.max_episode_steps, None, layout)
                p_random = control_result["win_rate"]
                normalized_skill = ((policy_result["win_rate"] - p_random) / (1 - p_random)) if p_random < 1.0 else None
                band_entry["random_control"] = control_result
                band_entry["normalized_skill"] = normalized_skill
                print(f"  policy win_rate={policy_result['win_rate']:.3f} {policy_result['win_rate_ci95']} "
                      f"random={p_random:.3f} {control_result['win_rate_ci95']} normalized_skill={normalized_skill}")
            else:
                print(f"  policy win_rate={policy_result['win_rate']:.3f} {policy_result['win_rate_ci95']}")
            result["bands"].append(band_entry)
        result["overall"] = {"win_rate": all_policy_won / all_policy_n if all_policy_n else None,
                              "win_rate_ci95": list(wilson_ci(all_policy_won, all_policy_n))}
    finally:
        env.close()

    print(f"\noverall pooled win_rate={result['overall']['win_rate']:.3f} {result['overall']['win_rate_ci95']} "
          f"(Sec3.2: pooled is for sanity-checking only -- report per-band)")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    if args.run_id:
        # Deliberately NOT telemetry.RunLogger here -- its __init__
        # unconditionally (re)writes run.json from whatever manifest it's
        # given, which would clobber an existing completed run's real
        # manifest just to log an eval result. eval/<step>.json is the only
        # file this needs to touch.
        runs_root = Path(args.runs_root) if args.runs_root else Path(args.repo_root) / "Tools/rl/runs"
        eval_dir = runs_root / args.run_id / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{global_step}.json"
        tmp = eval_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(eval_path)
        print(f"wrote {eval_path}")


if __name__ == "__main__":
    main()
