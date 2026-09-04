#!/usr/bin/env python3
"""Run a rendered human-versus-checkpoint match.

Controller 0 is native keyboard/mouse/gamepad input. Controller 1 is the
frozen PPO policy. The engine emits a terminal observation only when either
controlled participant is dead; recoverable unconscious/ragdoll states stay
inside the same round. This process holds a finished scene for exactly five
seconds, then performs the canonical reset and continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import ACTION_DIM, OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, SCHEMA_VERSION
from ppo.normalize import ObservationNormalizer
from ppo.policy import ActorCritic
from gen_human_duel_scenario import generate


RESTART_SECONDS = 5.0
def _latest_values(raw_values: list[float] | np.ndarray) -> list[float] | np.ndarray:
    """Read the newest frame when the runtime is using frame stacking."""
    frame_size = DEFAULT_LAYOUT.total_floats
    if len(raw_values) > frame_size:
        return raw_values[-frame_size:]
    return raw_values


def deterministic_action(policy: ActorCritic, obs_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = policy.actor_trunk(policy._features(obs_tensor))
        continuous = torch.tanh(policy.continuous_mean(features))
        discrete = (policy.discrete_logits(features) > 0.0).float()
        return torch.cat([continuous, discrete], dim=-1).squeeze(0).cpu().numpy()


def sampled_action(policy: ActorCritic, obs_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        action, _, _, _ = policy.get_action_and_value(obs_tensor)
        return action.squeeze(0).cpu().numpy()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _human_result(raw_values: list[float]) -> str:
    """Best-effort label for the status page; engine done remains authoritative."""
    raw_values = _latest_values(raw_values)
    self_ko = DEFAULT_LAYOUT.self_knocked_out_index(raw_values)
    if bool(self_ko != 0):
        return "human_win"
    for entity in DEFAULT_LAYOUT.all_entities(raw_values):
        if bool(entity["valid"]) and not bool(entity["is_ally"]) and bool(max(entity["knocked_out"]) == entity["knocked_out"][1]):
            return "checkpoint_win"
    return "round_over"


def _diagnostics(raw_values: list[float], action: np.ndarray, inference_ms: float) -> dict:
    raw_values = _latest_values(raw_values)
    visible_enemy = False
    enemy_out = False
    enemy_hp = None
    enemy_distance = None
    enemy_block = None
    enemy_state = None
    for entity in DEFAULT_LAYOUT.all_entities(raw_values):
        if not entity["valid"] or entity["is_ally"]:
            continue
        visible_enemy = True
        enemy_hp = float(entity["temp_health"])
        enemy_distance = float(entity["distance"])
        enemy_block = float(entity["block_health"])
        enemy_state = int(np.argmax(entity["state"]))
        enemy_out = bool(max(entity["knocked_out"]) == entity["knocked_out"][1] or max(entity["knocked_out"]) == entity["knocked_out"][2])
        break
    return {
        "action": [round(float(value), 4) for value in action.tolist()],
        "inference_ms": round(inference_ms, 3),
        "agent_health": round(float(raw_values[DEFAULT_LAYOUT.TEMP_HEALTH]), 3),
        "agent_out": bool(DEFAULT_LAYOUT.self_knocked_out_index(raw_values) != 0),
        "human_visible": visible_enemy,
        "human_health": round(enemy_hp, 3) if enemy_hp is not None else None,
        "human_distance": round(enemy_distance, 3) if enemy_distance is not None else None,
        "human_block": round(enemy_block, 3) if enemy_block is not None else None,
        "human_state": enemy_state,
        "human_out": enemy_out if visible_enemy else None,
    }


class MatchStatus:
    def __init__(self, path: Path, match_id: str, checkpoint_id: str, mode: str):
        self.path = path
        self.payload = {
            "match_id": match_id,
            "checkpoint_id": checkpoint_id,
            "policy_mode": mode,
            "phase": "loading",
            "round": 0,
            "score": {"you": 0, "checkpoint": 0},
            "restart_seconds": RESTART_SECONDS,
            "updated_at": time.time(),
        }
        self.write()

    def update(self, **values) -> None:
        self.payload.update(values)
        self.payload["updated_at"] = time.time()
        self.write()

    def write(self) -> None:
        _write_json(self.path, self.payload)


def _load_policy(checkpoint_path: Path, device: torch.device, frame_stack: int) -> tuple[ActorCritic, ObservationNormalizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("layout_total_floats") != DEFAULT_LAYOUT.total_floats:
        raise ValueError(
            f"checkpoint layout is {checkpoint.get('layout_total_floats')}; current match engine emits {DEFAULT_LAYOUT.total_floats}"
        )
    checkpoint_stack = checkpoint.get("frame_stack")
    if checkpoint_stack != frame_stack:
        raise ValueError(f"checkpoint frame_stack={checkpoint_stack}; match runtime requested {frame_stack}")
    policy = ActorCritic(DEFAULT_LAYOUT, frame_stack=frame_stack).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    normalizer = ObservationNormalizer(DEFAULT_LAYOUT, frame_stack=frame_stack)
    normalizer.load_state_dict(checkpoint["obs_normalizer"])
    return policy, normalizer, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--binary-path", default=None)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--policy-mode", choices=("deterministic", "sampled"), default="deterministic")
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--round-timeout", type=float, default=60.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    session_dir = Path(args.session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    status = MatchStatus(Path(args.status_path), args.match_id, args.checkpoint_id or Path(args.checkpoint).name, args.policy_mode)
    log_path = session_dir / "rounds.jsonl"
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else Path(
        "/Users/pavlov/Library/Application Support/Steam/steamapps/common/"
        "Overgrowth/Overgrowth.app/Contents/MacOS/Data"
    )
    frame_stack = args.frame_stack
    checkpoint_probe = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if frame_stack is None:
        frame_stack = int(checkpoint_probe.get("frame_stack") or 1)
    device = torch.device(args.device)
    policy, normalizer, checkpoint = _load_policy(Path(args.checkpoint), device, frame_stack)
    status.update(phase="scenario", global_step=checkpoint.get("global_step"), frame_stack=frame_stack, act_period=args.act_period)
    scenario = generate(data_root)
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    _write_json(session_dir / "session.json", {
        "schema": 1,
        "match_id": args.match_id,
        "match_schema": 1,
        "scenario_id": "oval_human_duel_v1",
        "seed": args.seed,
        "rounds_to_win": 0,
        "round_timeout_seconds": args.round_timeout,
        "camera_mode": "shared_arena",
        "overlay": "standard",
        "record_exact": False,
        "checkpoint_id": args.checkpoint_id or Path(args.checkpoint).name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "global_step": checkpoint.get("global_step"),
        "observation_schema": SCHEMA_VERSION,
        "observation_rules": "schema-v5-driver-mask-match-v1",
        "action_schema": "player-v1",
        "policy_adapter": "ppo-actor-critic-v1",
        "policy_mode": args.policy_mode,
        "frame_stack": frame_stack,
        "act_period": args.act_period,
        "restart_seconds": RESTART_SECONDS,
        "participants": [
            {"slot": 0, "role": "checkpoint", "controller_id": 1, "team": 0, "actor_id": "char_player",
             "checkpoint_id": args.checkpoint_id or Path(args.checkpoint).name},
            {"slot": 1, "role": "human", "controller_id": 0, "team": 1,
             "actor_id": "char_player"},
        ],
        "scenario": scenario,
        "started_at": time.time(),
    })

    def stop_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    shm_name = "/ogrl_m" + args.match_id.replace("-", "")[-10:]
    level = "arenas/oval_arena_human_duel.xml"
    env = None
    try:
        env = OvergrowthEnv(
            repo_root=repo_root,
            level=level,
            shm_name=shm_name,
            controller_id=1,
            seed=args.seed,
            frame_stack=frame_stack,
            render=True,
            time_scale_mult=1,
            act_period=args.act_period,
            binary_path=args.binary_path,
            write_dir_parent=repo_root / ".rl_match_write_dirs",
        )
        status.update(phase="fighting", round=1, engine_pid=env._process.pid if env._process else None)
        raw_obs = env.reset(seed=args.seed)
        score = {"you": 0, "checkpoint": 0}
        round_index = 0
        while True:
            round_index += 1
            status.update(phase="fighting", round=round_index, score=score, result=None, restart_at=None)
            obs = normalizer.normalize(raw_obs, update=False)
            round_started = time.monotonic()
            terminal = False
            last_result = "timeout"
            last_diag = {}
            for _step in range(200_000):
                if time.monotonic() - round_started >= args.round_timeout:
                    terminal = True
                    last_result = "timeout"
                    break
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                started = time.perf_counter()
                if args.policy_mode == "sampled":
                    action = sampled_action(policy, obs_tensor)
                else:
                    action = deterministic_action(policy, obs_tensor)
                inference_ms = (time.perf_counter() - started) * 1000.0
                raw_obs, _reward, done, info = env.step(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM))
                obs = normalizer.normalize(raw_obs, update=False)
                last_diag = _diagnostics(raw_obs, action, inference_ms)
                last_diag["action_path"] = "checkpoint-direct"
                status.update(phase="fighting", round=round_index, score=score, **last_diag)
                if done:
                    terminal = True
                    last_result = _human_result(raw_obs)
                    break
            if not terminal:
                break
            if last_result == "human_win":
                score["you"] += 1
            elif last_result == "checkpoint_win":
                score["checkpoint"] += 1
            restart_at = time.time() + RESTART_SECONDS
            status.update(phase="restart_wait", round=round_index, score=score, result=last_result, restart_at=restart_at)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"round": round_index, "result": last_result, "score": score,
                                      "duration_seconds": round(time.monotonic() - round_started, 3),
                                      "t": time.time()}) + "\n")
            # Keep the rendered scene responsive while the engine waits for
            # the fixed result window to elapse. The duel scenario has no
            # stock auto-rematch path, so the dead bodies remain visible.
            while time.time() < restart_at:
                zero = np.zeros(ACTION_DIM, dtype=np.float32)
                raw_obs, _reward, _done, _info = env.step(zero)
                remaining = max(0.0, restart_at - time.time())
                status.update(phase="restart_wait", round=round_index, score=score,
                              result=last_result, restart_at=restart_at,
                              restart_in=round(remaining, 1))
            status.update(phase="resetting", round=round_index, score=score)
            raw_obs = env.reset(seed=args.seed + round_index)
    except KeyboardInterrupt:
        status.update(phase="stopped")
        return 0
    except Exception as exc:  # noqa: BLE001 - durable status is the operator surface
        status.update(phase="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
        if status.payload.get("phase") not in ("error", "stopped"):
            status.update(phase="exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
