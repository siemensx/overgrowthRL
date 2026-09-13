#!/usr/bin/env python3
"""Run the game's own enemycontrol.as in the player slot as a diagnostic.

This is a deliberately privileged comparison, not a fair RL baseline. The
player slot is switched to Overgrowth's built-in ``enemycontrol.as`` through a
temporary copy of the level XML; the RL transport still advances the engine
and publishes observations, but zero actions are sent because the AngelScript
controller owns movement and combat. No health, position, damage, or physics
state is modified.

The test answers a narrow question: can the existing deterministic-ish game
AI survive the exact 1v3 scenario? Its controller uses engine state unavailable
to the public RL observation (target history, navmesh, and script internals),
so a strong result is an upper-bound/diagnostic, not evidence that PPO should
already match it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from env import ACTION_DIM, OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT
from paths import aux_data
from shm_env import ShmWaitTimeout


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def prepare_level(level: str, directory: Path) -> Path:
    source = Path(level).expanduser()
    if not source.is_absolute():
        source = aux_data() / "Data" / "Levels" / level
    if not source.exists():
        raise FileNotFoundError(f"level XML not found: {source}")
    xml = source.read_text(encoding="utf-8")
    if re.search(r"<PCScript>.*?</PCScript>", xml, flags=re.DOTALL):
        xml = re.sub(r"<PCScript>.*?</PCScript>", "<PCScript>enemycontrol.as</PCScript>", xml, count=1, flags=re.DOTALL)
    else:
        marker = "<Script>"
        if marker not in xml:
            raise ValueError(f"level has no <Script> element: {source}")
        xml = xml.replace(marker, "<PCScript>enemycontrol.as</PCScript>\n" + marker, 1)
    destination = directory / source.name
    destination.write_text(xml, encoding="utf-8")
    return destination


def reset_scenario(env: OvergrowthEnv, scenario: dict, seed: int) -> np.ndarray:
    observation = env.reset(seed=seed, **scenario)
    if env.episode_count == 0:
        for attempt in range(12):
            try:
                observation = env.reset(seed=seed, **scenario)
                break
            except (RuntimeError, ShmWaitTimeout):
                if attempt == 11:
                    raise
                time.sleep(2.0)
    return observation


def write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing baseline result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--level", default="arenas/t_train_101.xml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--armed-count", type=int, default=0)
    parser.add_argument("--weapon-type", type=int, default=0)
    parser.add_argument("--species", type=int, default=0)
    parser.add_argument("--throw-aggression", type=float, default=1.0)
    parser.add_argument("--no-jumpkick", action="store_true", help="privileged diagnostic ablation: use ground combat only")
    parser.add_argument("--navmesh-movement", action="store_true", help="privileged diagnostic ablation: use enemycontrol navigation")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--out", default=None, help="optional JSON result path; existing files are refused")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.opponents <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes, opponents, and max-steps must be positive")
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    run_dir = Path(args.repo_root) / "Tools" / "rl" / "runs" / "oracle-baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="engine-ai-level-", dir=run_dir) as level_dir_name:
        level_path = prepare_level(args.level, Path(level_dir_name))
        scenario = {
            "opponents": args.opponents,
            "difficulty": args.difficulty,
            "weapons": 0.0,
            "species": args.species,
            "armed_count": args.armed_count,
            "weapon_type": args.weapon_type,
            "throw_aggression": args.throw_aggression,
        }
        env = OvergrowthEnv(
            repo_root=args.repo_root,
            level=str(level_path),
            seed=args.seed_base,
            layout=DEFAULT_LAYOUT,
            frame_stack=args.frame_stack,
            render=False,
            time_scale_mult=100,
            act_period=args.act_period,
            shm_name=f"/ogrl_engine_ai_{os.getpid()}",
            extra_config_lines=[
                "rl_oracle_ai: true",
                f"rl_oracle_jumpkick: {'false' if args.no_jumpkick else 'true'}",
                f"rl_oracle_direct_move: {'false' if args.navmesh_movement else 'true'}",
            ],
        )
        episodes = []
        try:
            for episode in range(args.episodes):
                seed = args.seed_base + episode
                reset_scenario(env, scenario, seed)
                steps = 0
                reward_total = 0.0
                hostile_kos = 0
                done = False
                started = time.monotonic()
                zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
                for _ in range(args.max_steps):
                    _, reward, done, info = env.step(zero_action)
                    reward_total += float(reward)
                    components = info.get("reward_components", {})
                    hostile_kos += int(round(float(components.get("hostile_kos_this_step", 0.0))))
                    steps += 1
                    if hostile_kos >= args.opponents or done:
                        break
                won = hostile_kos >= args.opponents
                outcome = "WON" if won else ("LOST" if done else "timed out")
                result = {
                    "episode": episode,
                    "seed": seed,
                    "outcome": outcome,
                    "steps": steps,
                    "hostile_kos": hostile_kos,
                    "reward": reward_total,
                    "real_seconds": time.monotonic() - started,
                }
                episodes.append(result)
                print(
                    f"episode {episode:03d}: seed={seed} steps={steps} kos={hostile_kos} "
                    f"{outcome} real_seconds={result['real_seconds']:.1f}",
                    flush=True,
                )
        finally:
            env.close()

    wins = sum(episode["outcome"] == "WON" for episode in episodes)
    losses = sum(episode["outcome"] == "LOST" for episode in episodes)
    timeouts = sum(episode["outcome"] == "timed out" for episode in episodes)
    summary = {
        "source": "Tools/rl/engine_ai_baseline.py",
        "profile": "privileged-engine-enemycontrol-oracle",
        "warning": "not a fair RL baseline; player slot uses hidden engine AI state, omniscient target acquisition, and ignores RL actions",
        "level": args.level,
        "scenario": scenario,
        "seed_base": args.seed_base,
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "frame_stack": args.frame_stack,
        "act_period": args.act_period,
        "no_jumpkick": args.no_jumpkick,
        "navmesh_movement": args.navmesh_movement,
        "max_steps": args.max_steps,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / len(episodes) if episodes else 0.0,
        "mean_hostile_kos": sum(episode["hostile_kos"] for episode in episodes) / len(episodes) if episodes else 0.0,
        "episodes": episodes,
    }
    print(
        f"summary: wins={wins}/{len(episodes)} win_rate={summary['win_rate']:.3f} "
        f"losses={losses} timeouts={timeouts} mean_hostile_kos={summary['mean_hostile_kos']:.3f}",
        flush=True,
    )
    if args.out:
        write_json_new(Path(args.out).expanduser().resolve(), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
