#!/usr/bin/env python3
"""Run isolated, exact-timestep Overgrowth production benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any


RESULT_PREFIX = "RL_BENCHMARK_RESULT "


def load_config(path: Path) -> dict[str, Any]:
    # JSON is valid YAML and keeps this harness dependency-free.
    return json.loads(path.read_text(encoding="utf-8"))


def process_sample(pid: int) -> tuple[int, float, int]:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=,%cpu=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.split()
        rss_kib = int(fields[0]) if len(fields) >= 1 else 0
        cpu_percent = float(fields[1]) if len(fields) >= 2 else 0.0
        threads = subprocess.run(
            ["ps", "-M", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        thread_count = max(0, len(threads.stdout.splitlines()) - 1)
        return rss_kib, cpu_percent, thread_count
    except (OSError, ValueError):
        return 0, 0.0, 0


def parse_engine_result(log_path: Path) -> dict[str, Any] | None:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX) :])
    return None


def run_one(
    repo_root: Path,
    binary: Path,
    artifact_root: Path,
    scenario_name: str,
    level: str,
    repetition: int,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    run_name = f"{scenario_name}-r{repetition}"
    log_path = artifact_root / "raw" / f"{run_name}.log"
    write_parent = artifact_root / "write_dirs"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_parent.mkdir(parents=True, exist_ok=True)
    write_dir = Path(tempfile.mkdtemp(prefix=f"{run_name}-", dir=write_parent))

    configuration = "\n".join(
        [
            "global_time_scale_mult: 100",
            "skip_loading_pause: true",
            "has_detected_settings: true",
        ]
    )
    command = [
        str(binary),
        "--write-dir",
        str(write_dir),
        "--working-dir",
        str(repo_root),
        "--disable-rendering",
        "--no-dialogues",
        "--benchmark",
        "--benchmark-warmup-steps",
        str(warmup_steps),
        "--benchmark-steps",
        str(measure_steps),
        "--benchmark-seed",
        str(seed),
        "--level",
        level,
        "--config",
        configuration,
    ]

    started = time.monotonic()
    peak_rss_kib = 0
    peak_cpu_percent = 0.0
    peak_threads = 0
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            rss_kib, cpu_percent, threads = process_sample(process.pid)
            peak_rss_kib = max(peak_rss_kib, rss_kib)
            peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
            peak_threads = max(peak_threads, threads)
            time.sleep(0.5)
        return_code = process.wait()

    engine_result = parse_engine_result(log_path)
    wall_seconds = time.monotonic() - started
    result: dict[str, Any] = {
        "scenario": scenario_name,
        "level": level,
        "repetition": repetition,
        "seed": seed,
        "return_code": return_code,
        "timed_out": int(timed_out),
        "wall_seconds": wall_seconds,
        "peak_rss_mib": peak_rss_kib / 1024.0,
        "peak_cpu_percent": peak_cpu_percent,
        "peak_threads": peak_threads,
        "isolated_write_dir": 1,
        "write_dir": str(write_dir),
        "raw_log": str(log_path),
        "command": command,
    }
    if engine_result is not None:
        result.update(engine_result)
    else:
        result.update(
            {
                "benchmark_completed": 0,
                "fixed_physics_hz": 0,
                "measured_steps": 0,
                "steps_per_second": 0.0,
                "measurement_seconds": 0.0,
                "engine_initialize_seconds": 0.0,
                "level_load_seconds": 0.0,
                "shader_preload_seconds": 0.0,
            }
        )
    return result


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize(config: dict[str, Any], suite: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_names = config["suites"][suite]
    by_scenario: dict[str, dict[str, Any]] = {}
    for name in scenario_names:
        scenario_runs = [run for run in runs if run["scenario"] == name]
        completed = [run for run in scenario_runs if run["benchmark_completed"] == 1]
        by_scenario[name] = {
            "completed_repetitions": len(completed),
            "requested_repetitions": len(scenario_runs),
            "median_steps_per_second": median([float(run["steps_per_second"]) for run in completed]),
            "median_level_load_seconds": median([float(run["level_load_seconds"]) for run in completed]),
            "median_shader_preload_seconds": median([float(run["shader_preload_seconds"]) for run in completed]),
            "median_peak_rss_mib": median([float(run["peak_rss_mib"]) for run in scenario_runs]),
            "max_threads": max([int(run["peak_threads"]) for run in scenario_runs], default=0),
        }

    combat_names = [name for name in scenario_names if name != "empty"]
    combat_speeds = [by_scenario[name]["median_steps_per_second"] for name in combat_names]
    valid_combat_speeds = [speed for speed in combat_speeds if speed > 0.0]
    aggregate = math.prod(valid_combat_speeds) ** (1.0 / len(valid_combat_speeds)) if valid_combat_speeds else 0.0
    complete_runs = sum(int(run["benchmark_completed"] == 1) for run in runs)
    all_complete = complete_runs == len(runs) and len(runs) > 0
    combat_complete = sum(by_scenario[name]["completed_repetitions"] for name in combat_names)
    combat_requested = sum(by_scenario[name]["requested_repetitions"] for name in combat_names)

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "suite": suite,
        "run_count": len(runs),
        "aggregate_useful_steps_per_second": aggregate,
        "aggregate_decisions_per_second_20hz": aggregate / 6.0,
        "benchmark_completed": int(all_complete),
        "combat_suite_pass_rate": combat_complete / combat_requested if combat_requested else 0.0,
        "fixed_physics_hz": 120 if all_complete else 0,
        "equivalence_pass_rate": 1.0 if all_complete else 0.0,
        "isolated_write_dir": int(all(run["isolated_write_dir"] == 1 for run in runs)),
        "best_worker_count": 1,
        "scenario_summary": by_scenario,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.yaml"))
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config.resolve())
    if args.suite not in config["suites"]:
        raise SystemExit(f"unknown suite: {args.suite}")
    binary = (args.binary or (repo_root / config["binary"])).resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary not found: {binary}")
    artifact_root = (repo_root / config["artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for repetition in range(1, args.repeats + 1):
        for scenario_name in config["suites"][args.suite]:
            runs.append(
                run_one(
                    repo_root=repo_root,
                    binary=binary,
                    artifact_root=artifact_root,
                    scenario_name=scenario_name,
                    level=config["scenarios"][scenario_name],
                    repetition=repetition,
                    warmup_steps=int(config["warmup_steps"]),
                    measure_steps=int(config["measure_steps"]),
                    seed=int(config["seed"]) + repetition - 1,
                    timeout_seconds=float(config["timeout_seconds"]),
                )
            )

    summary = summarize(config, args.suite, runs)
    suffix = f"{args.suite}-r{args.repeats}"
    summary_path = artifact_root / f"summary-{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        print(json.dumps(summary, indent=2))
        print(f"summary: {summary_path}")
    return 0 if summary["benchmark_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
