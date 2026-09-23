#!/usr/bin/env python3
"""Strict, no-checkpoint replay gate for two Windows engine binaries.

Runs the same legal, deterministic action trace against both binaries using
the production RL shared-memory interface. It preserves each engine digest,
native control trace, engine stdout log, and attack-selection log; checks
within-build repeatability; and compares physics digests plus Python
observations/rewards.
The output directory must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import OvergrowthEnv
from replay_compare import compare as compare_digests, load_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actions(steps: int) -> list[np.ndarray]:
    # Repeatable legal controls that exercise movement and several combat
    # buttons without depending on model sampling or host-side randomness.
    moves = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0),
             (0.70710677, 0.70710677), (-0.70710677, 0.70710677))
    result = []
    for i in range(steps):
        move_x, move_y = moves[(i // 16) % len(moves)]
        result.append(np.asarray((
            move_x, move_y,
            int(i % 31 == 0),                 # jump
            int(i % 43 in (4, 5)),            # crouch/roll
            int(i % 9 in (0, 1)),             # attack
            int(i % 23 == 2),                 # grab
            int(i % 29 == 3),                 # drop
            int(i % 7 < 3),                   # walk
        ), dtype=np.float32))
    return result


def _tick_rows(path: Path) -> list[dict]:
    # The digest has one reset record before its per-physics-tick records.
    return [row for row in load_digest(path) if row.get("kind") == "tick"]


def _attack_events_observed_and_equal(left: dict, right: dict) -> bool:
    """Empty attack logs cannot establish combat-event equivalence."""
    return (
        left.get("attack_event_count", 0) > 0
        and right.get("attack_event_count", 0) > 0
        and left.get("attack_log_sha256") == right.get("attack_log_sha256")
    )


def _preserve_engine_log(source: Path, destination: Path) -> str | None:
    """Copy a run's sibling engine log into its unique replay artifact dir."""
    if not source.is_file():
        return None
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite preserved engine log: {destination}")
    shutil.copy2(source, destination)
    return str(destination)


def _run_one(
    *, repo_root: Path, binary: Path, runtime_dir: Path, output_dir: Path,
    label: str, map_index: int, repetition: int, level: str, seed: int,
    steps: int, difficulty: float, opponents: int, act_period: int,
) -> dict:
    safe_level = re.sub(r"[^A-Za-z0-9_-]+", "_", level)
    stem = f"{label}_{safe_level}_seed{seed}_rep{repetition:02d}"
    digest_path = output_dir / f"{stem}.digest.jsonl"
    control_trace_path = output_dir / f"{stem}.control.jsonl"
    attack_log_path = output_dir / f"{stem}.attacks.txt"
    action_hash = hashlib.sha256()
    observation_reward_hash = hashlib.sha256()
    actions = _actions(steps)

    previous_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(runtime_dir) + os.pathsep + previous_path
    env = None
    attack_lines: list[str] = []
    preserved_engine_log: str | None = None
    preserved_internal_log: str | None = None
    engine_exit_code: int | None = None
    shutdown_timed_out = False
    try:
        shm_name = f"/ogrl_cmp_{label}_{map_index}_{repetition}_{uuid.uuid4().hex[:8]}"
        env = OvergrowthEnv(
            repo_root=repo_root,
            level=level,
            seed=seed,
            binary_path=binary,
            shm_name=shm_name,
            write_dir_parent=output_dir / "write_dirs",
            frame_stack=1,
            act_period=act_period,
            time_scale_mult=100,
            equivalence_digest_path=digest_path,
            equivalence_trace_path=control_trace_path,
            log_attacks=True,
        )
        log_path = env._write_dir.parent / f"{env._write_dir.name}.log"
        try:
            observation = env.reset(
                seed=seed, difficulty=difficulty, opponents=opponents,
                weapons=0.0, species=0,
            )
            observation_reward_hash.update(np.asarray(observation, dtype=np.float32).tobytes())
            done_at = None
            for step_index, action in enumerate(actions):
                action_hash.update(action.tobytes())
                observation, reward, done, _info = env.step(action)
                observation_reward_hash.update(np.asarray(observation, dtype=np.float32).tobytes())
                observation_reward_hash.update(struct.pack("<d?", float(reward), bool(done)))
                if done:
                    done_at = step_index + 1
                    break
        finally:
            # Let the engine flush its digest and attack log before env.close()
            # removes this run's isolated write-dir and sibling stdout log.
            if env._shm is not None:
                try:
                    env._shm.request_shutdown()
                except OSError:
                    pass
            if env._process is not None:
                try:
                    engine_exit_code = env._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    shutdown_timed_out = True
                    env._process.kill()
                    engine_exit_code = env._process.wait(timeout=5)
            if log_path.exists():
                attack_lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                if "RLATK " in line or "RLTHROW " in line]
                preserved_engine_log = _preserve_engine_log(
                    log_path, output_dir / f"{stem}.engine.log")
            # On Windows the engine's internal log is rooted directly in the
            # per-run write directory (not its Data subfolder).
            internal_log = env._write_dir / "logfile.txt"
            preserved_internal_log = _preserve_engine_log(
                internal_log, output_dir / f"{stem}.game.log")
            attack_log_path.write_text("\n".join(attack_lines) + ("\n" if attack_lines else ""), encoding="utf-8")
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()
        os.environ["PATH"] = previous_path

    if shutdown_timed_out:
        raise RuntimeError(f"engine did not shut down gracefully for {stem}")
    if engine_exit_code != 0:
        raise RuntimeError(f"engine exited with code {engine_exit_code} for {stem}")

    if not digest_path.exists() or not control_trace_path.exists():
        raise RuntimeError(f"engine did not finalize digest/control trace for {stem}")
    ticks = _tick_rows(digest_path)
    if not ticks:
        raise RuntimeError(f"engine produced no tick digest rows for {stem}")
    return {
        "label": label,
        "level": level,
        "seed": seed,
        "repetition": repetition,
        "steps_requested": steps,
        "steps_completed": len(ticks),
        "done_at_action": done_at,
        "action_sha256": action_hash.hexdigest(),
        "observation_reward_sha256": observation_reward_hash.hexdigest(),
        "native_control_trace_sha256": _sha256(control_trace_path),
        "attack_log_sha256": _sha256(attack_log_path),
        "attack_event_count": len(attack_lines),
        "engine_log_path": preserved_engine_log,
        "internal_engine_log_path": preserved_internal_log,
        "digest_sha256": _sha256(digest_path),
        "digest_path": str(digest_path),
        "native_control_trace_path": str(control_trace_path),
        "attack_log_path": str(attack_log_path),
        "_tick_rows": ticks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--reference-runtime-dir", type=Path, default=None)
    parser.add_argument("--candidate-runtime-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["arenas/t_train_101.xml"])
    parser.add_argument("--seed", type=int, default=260922)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    if os.name != "nt":
        parser.error("this harness is intended for the Windows numeric regime")
    if args.steps < 1 or args.repetitions < 1:
        parser.error("steps and repetitions must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists; refusing to overwrite: {args.output_dir}")
    if not args.reference_binary.is_file() or not args.candidate_binary.is_file():
        parser.error("both engine binaries must exist")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference_runtime = args.reference_runtime_dir or args.reference_binary.parent
    candidate_runtime = args.candidate_runtime_dir or args.candidate_binary.parent
    results: list[dict] = []
    comparisons: list[dict] = []

    for map_index, level in enumerate(args.levels):
        seed = args.seed + map_index
        pair_runs: dict[str, list[dict]] = {"reference": [], "candidate": []}
        for repetition in range(args.repetitions):
            for label, binary, runtime_dir in (
                ("reference", args.reference_binary, reference_runtime),
                ("candidate", args.candidate_binary, candidate_runtime),
            ):
                run = _run_one(
                    repo_root=args.repo_root.resolve(), binary=binary.resolve(),
                    runtime_dir=runtime_dir.resolve(), output_dir=args.output_dir.resolve(),
                    label=label, map_index=map_index, repetition=repetition,
                    level=level, seed=seed, steps=args.steps,
                    difficulty=args.difficulty, opponents=args.opponents,
                    act_period=args.act_period,
                )
                pair_runs[label].append(run)
                print(json.dumps({k: v for k, v in run.items() if k != "_tick_rows"}), flush=True)

        ref_first = pair_runs["reference"][0]
        cand_first = pair_runs["candidate"][0]
        for label, group in pair_runs.items():
            first = group[0]
            for repeat in group[1:]:
                result = compare_digests(first["_tick_rows"], repeat["_tick_rows"], strict=True,
                                         pos_tol=0.0, vel_tol=0.0, scalar_tol=0.0)
                attack_events_observed = (
                    first["attack_event_count"] > 0 and repeat["attack_event_count"] > 0
                )
                passed = (result["passed"]
                          and first["action_sha256"] == repeat["action_sha256"]
                          and first["observation_reward_sha256"] == repeat["observation_reward_sha256"]
                          and first["native_control_trace_sha256"] == repeat["native_control_trace_sha256"]
                          and _attack_events_observed_and_equal(first, repeat))
                comparisons.append({"level": level, "comparison": f"within_{label}",
                                    "repetition": repeat["repetition"], "passed": passed,
                                    "attack_events_observed": attack_events_observed,
                                    "digest": result})
        for repetition, (reference, candidate) in enumerate(zip(pair_runs["reference"], pair_runs["candidate"])):
            result = compare_digests(reference["_tick_rows"], candidate["_tick_rows"], strict=True,
                                     pos_tol=0.0, vel_tol=0.0, scalar_tol=0.0)
            attack_events_observed = (
                reference["attack_event_count"] > 0 and candidate["attack_event_count"] > 0
            )
            passed = (result["passed"]
                      and reference["action_sha256"] == candidate["action_sha256"]
                      and reference["observation_reward_sha256"] == candidate["observation_reward_sha256"]
                      and reference["native_control_trace_sha256"] == candidate["native_control_trace_sha256"]
                      and _attack_events_observed_and_equal(reference, candidate))
            comparisons.append({"level": level, "comparison": "reference_vs_candidate",
                                "repetition": repetition, "passed": passed,
                                "attack_events_observed": attack_events_observed,
                                "digest": result})
        results.extend({k: v for k, v in run.items() if k != "_tick_rows"}
                       for group in pair_runs.values() for run in group)

    summary = {
        "reference_binary": str(args.reference_binary.resolve()),
        "reference_binary_sha256": _sha256(args.reference_binary),
        "candidate_binary": str(args.candidate_binary.resolve()),
        "candidate_binary_sha256": _sha256(args.candidate_binary),
        "levels": args.levels,
        "seed_base": args.seed,
        "steps": args.steps,
        "difficulty": args.difficulty,
        "opponents": args.opponents,
        "act_period": args.act_period,
        "repetitions": args.repetitions,
        "checkpoint_written": False,
        "attack_event_runs": sum(run["attack_event_count"] > 0 for run in results),
        "attack_event_runs_total": len(results),
        "attack_event_coverage_complete": bool(results) and all(
            run["attack_event_count"] > 0 for run in results
        ),
        "comparisons_passed": sum(bool(row["passed"]) for row in comparisons),
        "comparisons_total": len(comparisons),
        "passed": bool(comparisons) and all(row["passed"] for row in comparisons),
        "comparisons": comparisons,
        "runs": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("comparisons", "runs")}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
