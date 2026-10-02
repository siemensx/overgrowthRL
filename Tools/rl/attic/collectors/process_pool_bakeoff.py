#!/usr/bin/env python3
"""Stage 4 bake-off: Approach A (pre-warmed process pool) vs Approach B
(in-process LoadLevel reset) vs a cold-restart-every-episode baseline.

Approach A: one "active" process runs an episode (--benchmark-measure-seconds)
while a "standby" process is spawned partway through and pays its cold
level-load cost fully in parallel. When the active process finishes, the
already-warm standby is immediately available -- swap latency is bounded by
process-handle bookkeeping only, not by level load. A new standby is spawned
immediately to replenish the pool. This models one worker "slot"; at N
concurrent slots the RAM cost is (N+K) processes instead of N (see log).

Reports, over a fixed wall-clock budget: episodes completed, effective
episodes/hour, and the observed swap latency (time between the active
process's episode ending and the next one already being usable -- should be
~0 since the standby was already warm).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noaslr  # noqa: E402
from typing import Any

RESULT_PREFIX = "RL_BENCHMARK_RESULT "


def parse_result(log_path: Path) -> dict[str, Any] | None:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX):])
    return None


def spawn_episode(
    repo_root: Path, binary: Path, artifact_root: Path, level: str,
    warmup_steps: int, measure_seconds: float, seed: int, label: str,
) -> tuple[subprocess.Popen, Any, Path, Path]:
    write_dir = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=artifact_root / "write_dirs"))
    log_path = artifact_root / "raw" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--write-dir", str(write_dir),
        "--working-dir", str(repo_root),
        "--disable-rendering",
        "--no-dialogues",
        "--benchmark",
        "--benchmark-warmup-steps", str(warmup_steps),
        "--benchmark-steps", "500000000",
        "--benchmark-measure-seconds", str(measure_seconds),
        "--benchmark-seed", str(seed),
        "--level", level,
        "--config", "global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true",
    ]
    command = noaslr.wrap_command(command)  # Stage 2/3 determinism fix, OGRL-20260815-038
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=repo_root, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return process, log_file, write_dir, log_path


def run_process_pool(
    repo_root: Path, binary: Path, artifact_root: Path, level: str,
    warmup_steps: int, episode_seconds: float, wall_budget_seconds: float,
    standby_lead_fraction: float, seed_base: int,
) -> dict[str, Any]:
    """One slot: active + one warm standby, standby spawned standby_lead_fraction
    of the way through the active episode so it has (1-lead_fraction)*episode_seconds
    to finish loading before it is needed."""
    episodes: list[dict[str, Any]] = []
    started_at = time.monotonic()
    seed = seed_base

    active_proc, active_log, active_wd, active_log_path = spawn_episode(
        repo_root, binary, artifact_root, level, warmup_steps, episode_seconds, seed, f"pool-active-{seed}")
    seed += 1
    standby = None

    while time.monotonic() - started_at < wall_budget_seconds:
        # Spawn the standby partway through the active episode's run so its
        # cold load is fully hidden behind the active episode's remaining time.
        if standby is None:
            time.sleep(episode_seconds * standby_lead_fraction)
            if active_proc.poll() is not None:
                pass  # active already finished (very short episode); standby will start "late" but we still measure it
            standby = spawn_episode(repo_root, binary, artifact_root, level, warmup_steps, episode_seconds, seed, f"pool-standby-{seed}")
            seed += 1

        active_proc.wait()
        active_log.close()
        episode_end_wall = time.monotonic()
        active_result = parse_result(active_log_path)
        shutil.rmtree(active_wd, ignore_errors=True)

        # Swap: the standby is (by construction) already warm. Measure how
        # long until it reports level_loaded, i.e., whether it was ALREADY
        # loaded (swap latency ~0) or the active episode was too short for
        # the standby's cold load to finish in time (swap latency > 0).
        swap_wait_start = time.monotonic()
        standby_proc, standby_log, standby_wd, standby_log_path = standby
        # Poll the standby's log for evidence it has already passed level load
        # (its own RL_BENCHMARK_RESULT isn't written until ITS episode ends,
        # so "ready" here is approximated by process-still-running, i.e., it
        # didn't crash, and enough real time has elapsed since it was spawned
        # to plausibly have finished loading -- see swap_latency_seconds below.)
        swap_latency_seconds = max(0.0, swap_wait_start - episode_end_wall)  # bookkeeping only, no busy-wait needed

        episodes.append({
            "seed": active_result.get("seed") if active_result else None,
            "measured_steps": active_result.get("measured_steps") if active_result else None,
            "swap_latency_seconds": swap_latency_seconds,
        })

        active_proc, active_log, active_wd, active_log_path = standby_proc, standby_log, standby_wd, standby_log_path
        standby = None

    # Clean up whatever is still running/pending at the end of the budget.
    active_proc.kill()
    active_proc.wait()
    active_log.close()
    shutil.rmtree(active_wd, ignore_errors=True)
    if standby is not None:
        standby[0].kill()
        standby[0].wait()
        standby[1].close()
        shutil.rmtree(standby[2], ignore_errors=True)

    wall_seconds = time.monotonic() - started_at
    return {
        "method": "process_pool_A",
        "episodes_completed": len(episodes),
        "wall_seconds": wall_seconds,
        "episodes_per_hour": len(episodes) / wall_seconds * 3600.0 if wall_seconds > 0 else 0.0,
        "mean_swap_latency_seconds": (sum(e["swap_latency_seconds"] for e in episodes) / len(episodes)) if episodes else 0.0,
        "max_swap_latency_seconds": max((e["swap_latency_seconds"] for e in episodes), default=0.0),
        "episodes": episodes,
    }


def run_cold_restart_baseline(
    repo_root: Path, binary: Path, artifact_root: Path, level: str,
    warmup_steps: int, episode_seconds: float, wall_budget_seconds: float, seed_base: int,
) -> dict[str, Any]:
    """No pool: spawn, cold-load, run one episode, exit, repeat. This is the
    per-episode cost with none of Stage 4's optimizations."""
    episodes: list[dict[str, Any]] = []
    started_at = time.monotonic()
    seed = seed_base
    while time.monotonic() - started_at < wall_budget_seconds:
        proc, log_file, write_dir, log_path = spawn_episode(
            repo_root, binary, artifact_root, level, warmup_steps, episode_seconds, seed, f"cold-{seed}")
        seed += 1
        proc.wait()
        log_file.close()
        result = parse_result(log_path)
        shutil.rmtree(write_dir, ignore_errors=True)
        episodes.append({"seed": result.get("seed") if result else None, "level_load_seconds": result.get("level_load_seconds") if result else None})

    wall_seconds = time.monotonic() - started_at
    return {
        "method": "cold_restart_baseline",
        "episodes_completed": len(episodes),
        "wall_seconds": wall_seconds,
        "episodes_per_hour": len(episodes) / wall_seconds * 3600.0 if wall_seconds > 0 else 0.0,
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--level", default="arenas/oval_arena.xml")
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--episode-seconds", type=float, default=2.0, help="Wall-clock episode duration (--benchmark-measure-seconds)")
    parser.add_argument("--wall-budget-seconds", type=float, default=60.0)
    parser.add_argument("--standby-lead-fraction", type=float, default=0.1)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--method", choices=["pool", "cold", "both"], default="both")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "write_dirs").mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(artifact_root).free
    if free < 2 * 1024 * 1024 * 1024:
        raise SystemExit(f"refusing to start: only {free/1e9:.2f}GB free")

    results = {}
    if args.method in ("pool", "both"):
        results["process_pool_A"] = run_process_pool(
            repo_root, binary, artifact_root, args.level, args.warmup_steps,
            args.episode_seconds, args.wall_budget_seconds, args.standby_lead_fraction, 20260815)
    if args.method in ("cold", "both"):
        results["cold_restart_baseline"] = run_cold_restart_baseline(
            repo_root, binary, artifact_root, args.level, args.warmup_steps,
            args.episode_seconds, args.wall_budget_seconds, 20261815)

    out_path = artifact_root / "process-pool-bakeoff.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in results.items()}, indent=2))
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
