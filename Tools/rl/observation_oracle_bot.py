#!/usr/bin/env python3
"""Deterministic expert diagnostic driven by the public RL observation.

This is deliberately not a trained policy and not an engine cheat. It reads
the same structured observation published to the RL controller and emits the
same legal eight-float action used by PPO. It has deterministic target
selection, spacing, active-block timing, ragdoll recovery, and ground-finish
rules so we can separate a learning/credit-assignment ceiling from a basic
control-interface ceiling.

It is called an observation oracle because the rule system can exploit the
complete current visible entity table, including health, animation state,
forward direction, block resource, weapon type, and local geometry. It still
has no hidden entities, no target-id action, and no direct state mutation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import ACTION_DIM, OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, ObsLayout
from shm_env import ShmWaitTimeout


STATE_MOVEMENT = 0
STATE_GROUND = 1
STATE_ATTACK = 2
STATE_HIT = 3
STATE_RAGDOLL = 4
KNOCKOUT_AWAKE = 0


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def norm2(x: float, z: float) -> float:
    return math.hypot(x, z)


def decode_frame(values: np.ndarray, layout: ObsLayout) -> dict:
    """Decode one current 339-float frame without hiding any fields."""
    values = np.asarray(values, dtype=np.float32)
    self_values = values[:layout.entities_start]
    entities = [layout.entity_field(values.tolist(), slot)
                for slot in range(layout.max_visible_entities)]
    return {
        "raw": values,
        "self": self_values,
        "self_id": layout.self_id(values.tolist()),
        "self_knocked_out": layout.self_knocked_out_index(values.tolist()),
        "self_state": int(np.argmax(values[layout.STATE])),
        "history": values[layout.action_history_start:layout.entities_start].reshape(
            layout.action_history_steps, -1),
        "entities": entities,
        "rays": values[layout.rays_start:layout.rays_start + layout.local_geometry_rays],
    }


def unpack_observation(observation: np.ndarray, layout: ObsLayout) -> tuple[dict, list[dict]]:
    """Return the newest frame plus prior frames for short-term trend checks."""
    values = np.asarray(observation, dtype=np.float32)
    if values.size % layout.total_floats != 0:
        raise ValueError(f"observation has {values.size} floats, expected a multiple of {layout.total_floats}")
    frames = values.reshape(-1, layout.total_floats)
    decoded = [decode_frame(frame, layout) for frame in frames]
    return decoded[-1], decoded


def is_awake(entity: dict) -> bool:
    return int(np.argmax(entity["knocked_out"])) == KNOCKOUT_AWAKE


def health_value(entity: dict) -> float:
    # Blood health is the terminal combat resource; temp health is included as
    # a tie-breaker because it captures immediate damage pressure.
    return max(0.0, float(entity["blood_health"])) + 0.25 * max(0.0, float(entity["temp_health"]))


class ObservationOracle:
    """A fixed, inspectable policy over the public observation contract."""

    def __init__(self, layout: ObsLayout, opponents: int, style: str = "engine-inspired"):
        self.layout = layout
        self.opponents = opponents
        self.style = style
        self.decision = 0
        self.last_target_id: int | None = None
        self.last_block_decision = -10_000
        self.last_attack_decision = -10_000
        self.last_recover_decision = -10_000
        self.grab_until_decision = -1
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self) -> None:
        self.__init__(self.layout, self.opponents, self.style)

    def _visible_hostiles(self, current: dict) -> list[dict]:
        # The observation table is nearest-first and may contain allies,
        # invalid padding, or the same controlled actor in a diagnostic mode.
        # Entity IDs are used only for stable tie-breaking and target memory;
        # positions are always read from the current frame.
        hostiles = []
        for entity in current["entities"]:
            if not entity["valid"] or entity["is_ally"] or entity["is_controlled"]:
                continue
            if entity["id"] < 0:
                continue
            hostiles.append(entity)
        return hostiles

    def _choose_target(self, hostiles: list[dict]) -> dict | None:
        if not hostiles:
            return None

        def key(entity: dict) -> tuple:
            down = 0 if not is_awake(entity) else 1
            remembered = 0 if entity["id"] == self.last_target_id else 1
            # Finish a downed opponent first, otherwise pressure the weakest
            # visible opponent. Distance remains a deterministic final tie.
            return down, remembered, health_value(entity), float(entity["distance"]), int(entity["id"])

        target = min(hostiles, key=key)
        self.last_target_id = int(target["id"])
        return target

    def _spacing_vector(self, target: dict, hostiles: list[dict]) -> tuple[float, float]:
        tx, _, tz = (float(value) for value in target["rel_pos"])
        x, z = tx, tz
        # Keep multiple awake enemies from pinning the agent against one body.
        # The vector is built from every currently visible hostile, not only
        # the selected target.
        for entity in hostiles:
            if entity["id"] == target["id"] or not is_awake(entity):
                continue
            ex, _, ez = (float(value) for value in entity["rel_pos"])
            distance = max(0.4, norm2(ex, ez))
            if distance < 3.6:
                weight = (3.6 - distance) / 3.6
                x -= ex * weight / distance
                z -= ez * weight / distance
        return x, z

    def _geometry_steer(self, current: dict) -> tuple[float, float]:
        # Ray 0 is body-forward; angles increase toward body-right. Use all
        # sixteen rays as a soft wall repulsion, with extra weight near front.
        rays = np.asarray(current["rays"], dtype=np.float32)
        steer_x = 0.0
        steer_z = 0.0
        ray_count = len(rays)
        for index, clearance in enumerate(rays):
            angle = 2.0 * math.pi * index / max(1, ray_count)
            danger = max(0.0, 0.42 - float(clearance)) / 0.42
            # A nearby wall in this ray pushes in the opposite direction.
            steer_x -= math.sin(angle) * danger
            steer_z -= math.cos(angle) * danger
        return steer_x, steer_z

    def _enemy_attack_threat(self, target: dict, hostiles: list[dict]) -> dict | None:
        threats = [entity for entity in hostiles
                   if is_awake(entity)
                   and int(np.argmax(entity["state"])) == STATE_ATTACK
                   and float(entity["distance"]) < 3.8]
        if not threats:
            return None
        return min(threats, key=lambda entity: (float(entity["distance"]), int(entity["id"])))

    def _jumpkick_action(self, current: dict, frames: list[dict], hostiles: list[dict], target: dict | None) -> tuple[np.ndarray, dict]:
        """Aggressive rabbit-specific combat style from the source mechanics.

        Aerial attacks select the closest conscious hostile in the engine's
        air-control path, rather than using the camera-facing ground selector.
        The controller therefore enters with a jump kick from farther out,
        keeps moving so crouch/axis input can trigger a legal dodge, and uses a
        grounded low strike only to finish a downed target.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        nearest = min((entity for entity in hostiles if is_awake(entity)),
                      key=lambda entity: (float(entity["distance"]), int(entity["id"])),
                      default=None)
        if target is not None:
            tx, _, tz = (float(value) for value in target["rel_pos"])
            distance = max(1.0, norm2(tx, tz))
            action[0] = clamp(tx / distance)
            action[1] = clamp(tz / distance)
            target_down = not is_awake(target)
            target_distance = float(target["distance"])
            self_state = current["self_state"]
            if (self_state in (STATE_MOVEMENT, STATE_GROUND)
                    and target_down and target_distance < 1.9
                    and self.decision - self.last_attack_decision >= 10):
                action[0] = 0.0
                action[1] = 0.0
                action[3] = 1.0
                action[4] = 1.0
                self.last_attack_decision = self.decision
            elif (self_state in (STATE_MOVEMENT, STATE_GROUND)
                  and not target_down and target_distance < 7.0
                  and self.decision - self.last_attack_decision >= 18):
                action[2] = 1.0
                action[4] = 1.0
                self.last_attack_decision = self.decision
        if current["self_knocked_out"] != KNOCKOUT_AWAKE or current["self_state"] == STATE_RAGDOLL:
            if self.decision - self.last_recover_decision >= 8:
                action[3] = 1.0
                self.last_recover_decision = self.decision
            action[2] = 0.0
            action[4] = 0.0
            if nearest is not None:
                nx, _, nz = (float(value) for value in nearest["rel_pos"])
                distance = max(1.0, norm2(nx, nz))
                action[0] = clamp(-nx / distance)
                action[1] = clamp(-nz / distance)
        self.previous_action = action.copy()
        self.decision += 1
        return action, {
            "visible_hostiles": len(hostiles),
            "visible_awake_hostiles": sum(is_awake(entity) for entity in hostiles),
            "target_id": int(target["id"]) if target is not None else None,
            "target_distance": float(target["distance"]) if target is not None else None,
            "target_down": bool(target is not None and not is_awake(target)),
            "self_state": current["self_state"],
            "self_knocked_out": current["self_knocked_out"],
            "frames_available": len(frames),
            "mode": "jumpkick",
            "action": action.tolist(),
        }

    def act(self, observation: np.ndarray) -> tuple[np.ndarray, dict]:
        current, frames = unpack_observation(observation, self.layout)
        self_values = current["self"]
        self_state = current["self_state"]
        self_knocked_out = current["self_knocked_out"]
        hostiles = self._visible_hostiles(current)
        target = self._choose_target(hostiles)
        if self.style == "jumpkick":
            return self._jumpkick_action(current, frames, hostiles, target)
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Start with short-term action inertia. This consumes the action
        # history exposed to PPO and avoids a pure one-frame pulse policy.
        if current["history"].size:
            recent = current["history"][-1]
            action[:2] = recent[:2] * 0.12

        nearest_awake = [entity for entity in hostiles if is_awake(entity)]
        nearest = min(nearest_awake, key=lambda entity: (float(entity["distance"]), int(entity["id"])), default=None)

        if target is not None:
            x, z = self._spacing_vector(target, hostiles)
            wall_x, wall_z = self._geometry_steer(current)
            # Target attraction dominates; wall/anti-pile steering is a small
            # correction so the bot still closes on a downed opponent.
            x += 0.35 * wall_x
            z += 0.35 * wall_z
            magnitude = max(1.0, norm2(x, z))
            action[0] = clamp(x / magnitude)
            action[1] = clamp(z / magnitude)

            target_distance = float(target["distance"])
            target_down = not is_awake(target)
            target_state = int(np.argmax(target["state"]))
            threat = self._enemy_attack_threat(target, hostiles)
            self_active_block = float(self_values[self.layout.ACTIVE_BLOCKING]) > 0.5
            block_recharge = float(self_values[self.layout.ACTIVE_BLOCK_RECHARGE])

            # The game has a short-range throw path in the normal player
            # controls: grab held for >0.2 seconds becomes a throw attempt.
            # Start it only at point-blank range and never on a downed target.
            # This is stronger than merely tapping grab (which only active-
            # blocks) while remaining an ordinary legal player action.
            if (not target_down and target_distance < 1.25
                    and self_state in (STATE_MOVEMENT, STATE_GROUND)
                    and self.grab_until_decision < self.decision):
                self.grab_until_decision = self.decision + 9

            if self.decision <= self.grab_until_decision:
                action[0] = 0.0
                action[1] = 0.0
                action[5] = 1.0
                action[4] = 0.0
                action[2] = 0.0
                action[3] = 0.0
                self.previous_action = action.copy()
                self.decision += 1
                return action, {
                    "visible_hostiles": len(hostiles),
                    "visible_awake_hostiles": len(nearest_awake),
                    "target_id": int(target["id"]),
                    "target_distance": target_distance,
                    "target_down": target_down,
                    "self_state": self_state,
                    "self_knocked_out": self_knocked_out,
                    "frames_available": len(frames),
                    "mode": "throw_hold",
                    "action": action.tolist(),
                }

            # A single decision pulse starts active block. Do not press grab
            # repeatedly while the block is already active or on cooldown.
            if (threat is not None and float(threat["distance"]) < 3.5 and not self_active_block
                    and block_recharge <= 0.01
                    and self.decision - self.last_block_decision >= 12):
                action[5] = 1.0
                self.last_block_decision = self.decision

            # Ground finishing is intentionally explicit. A full-clear match
            # is not won by merely knocking out one of three opponents.
            can_attack = self_state in (STATE_MOVEMENT, STATE_GROUND)
            attack_window = target_distance < (1.85 if target_down else 1.75)
            jump_window = (not target_down and 1.7 <= target_distance < 2.7
                           and target_state != STATE_ATTACK)
            if (can_attack and (attack_window or jump_window)
                    and self.decision - self.last_attack_decision >= 10):
                action[4] = 1.0
                if jump_window:
                    action[2] = 1.0
                elif target_down:
                    # Crouch selects the player's low ground strike while
                    # movement is stopped, which is the useful finish path.
                    action[3] = 1.0
                self.last_attack_decision = self.decision

            # Once inside ordinary strike range, stop translating for a
            # stationary attack. The engine's player script chooses a moving
            # attack whenever velocity input is nonzero; vanilla AI similarly
            # settles into a chosen attack range before striking. This also
            # leaves the body facing the target so the legacy camera-facing
            # target selector chooses the intended visible opponent.
            if target_distance < 1.7 and not target_down:
                tx, _, tz = (float(value) for value in target["rel_pos"])
                if abs(tx) < 0.45 and tz > 0.0:
                    action[0] = 0.0
                    action[1] = 0.0

            # A grounded strike should be stationary; otherwise the player
            # script deliberately selects a moving attack and the residual
            # steering can carry the body past the target before impact.
            if action[4] > 0.5 and action[2] <= 0.5:
                action[0] = 0.0
                action[1] = 0.0

        # Ragdoll recovery has priority over attack and is pulsed rather than
        # held, matching the legal crouch/recover input path.
        if self_knocked_out != KNOCKOUT_AWAKE or self_state == STATE_RAGDOLL:
            if self.decision - self.last_recover_decision >= 8:
                action[3] = 1.0
                self.last_recover_decision = self.decision
            action[4] = 0.0
            action[2] = 0.0
            if nearest is not None:
                nx, _, nz = (float(value) for value in nearest["rel_pos"])
                distance = max(1.0, norm2(nx, nz))
                action[0] = clamp(-nx / distance)
                action[1] = clamp(-nz / distance)

        # Do not combine block and attack in one action. The action bridge
        # accepts both, but the resulting semantics are engine-dependent and
        # this baseline is meant to be easy to audit.
        if action[5] > 0.5:
            action[4] = 0.0
            action[2] = 0.0

        self.previous_action = action.copy()
        self.decision += 1
        diagnostics = {
            "visible_hostiles": len(hostiles),
            "visible_awake_hostiles": len(nearest_awake),
            "target_id": int(target["id"]) if target is not None else None,
            "target_distance": float(target["distance"]) if target is not None else None,
            "target_down": bool(target is not None and not is_awake(target)),
            "self_state": self_state,
            "self_knocked_out": self_knocked_out,
            "frames_available": len(frames),
            "action": action.tolist(),
        }
        return action, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--level", default="arenas/t_train_101.xml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--armed-count", type=int, default=0)
    parser.add_argument("--weapon-type", type=int, default=0)
    parser.add_argument("--species", type=int, default=0)
    parser.add_argument("--throw-aggression", type=float, default=1.0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--style", choices=["engine-inspired", "jumpkick"], default="engine-inspired")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--out", default=None, help="optional JSON result path; existing files are refused")
    return parser.parse_args()


def write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing oracle result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def reset_scenario(env: OvergrowthEnv, kwargs: dict, seed: int) -> np.ndarray:
    observation = env.reset(seed=seed, **kwargs)
    # The first reset after launch consumes the engine's natural initial
    # observation. The second call is the requested scenario reset.
    if env.episode_count == 0:
        for attempt in range(12):
            try:
                observation = env.reset(seed=seed, **kwargs)
                break
            except (RuntimeError, ShmWaitTimeout):
                if attempt == 11:
                    raise
                time.sleep(2.0)
    return observation


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.opponents <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes, opponents, and max-steps must be positive")
    layout = DEFAULT_LAYOUT
    # Never reuse a named semaphore after an interrupted run. This keeps a
    # hard engine exit from poisoning the next diagnostic invocation.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    env = OvergrowthEnv(
        repo_root=args.repo_root,
        level=args.level,
        seed=args.seed_base,
        layout=layout,
        frame_stack=args.frame_stack,
        render=False,
        time_scale_mult=100,
        act_period=args.act_period,
        shm_name=f"/ogrl_oracle_{os.getpid()}",
    )
    bot = ObservationOracle(layout, args.opponents, args.style)
    scenario = {
        "opponents": args.opponents,
        "difficulty": args.difficulty,
        "weapons": 0.0,
        "species": args.species,
        "armed_count": args.armed_count,
        "weapon_type": args.weapon_type,
        "throw_aggression": args.throw_aggression,
    }
    episodes = []
    try:
        for episode in range(args.episodes):
            seed = args.seed_base + episode
            observation = reset_scenario(env, scenario, seed)
            bot.reset()
            steps = 0
            reward_total = 0.0
            hostile_kos = 0
            done = False
            action_counts = {"attack": 0, "jump": 0, "block": 0, "recover": 0}
            visible_peak = 0
            started = time.monotonic()
            final_diag = {}
            for _ in range(args.max_steps):
                action, final_diag = bot.act(observation)
                action_counts["attack"] += int(action[4] > 0.5)
                action_counts["jump"] += int(action[2] > 0.5)
                action_counts["block"] += int(action[5] > 0.5)
                action_counts["recover"] += int(action[3] > 0.5)
                visible_peak = max(visible_peak, int(final_diag["visible_hostiles"]))
                observation, reward, done, info = env.step(action)
                reward_total += float(reward)
                components = info.get("reward_components", {})
                hostile_kos += int(round(float(components.get("hostile_kos_this_step", 0.0))))
                steps += 1
                if hostile_kos >= args.opponents:
                    break
                if done:
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
                "action_counts": action_counts,
                "visible_hostiles_peak": visible_peak,
                "final_diagnostics": final_diag,
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
        "source": "Tools/rl/observation_oracle_bot.py",
        "profile": "observation-oracle-legal-actions",
        "style": args.style,
        "schema_version": 5,
        "observation_floats": layout.total_floats,
        "scenario": scenario,
        "level": args.level,
        "seed_base": args.seed_base,
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "frame_stack": args.frame_stack,
        "act_period": args.act_period,
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
