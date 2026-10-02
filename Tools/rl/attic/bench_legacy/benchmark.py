#!/usr/bin/env python3
"""Run isolated, exact-timestep Overgrowth production benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noaslr  # noqa: E402


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
    keep_write_dir: bool = False,
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
    command = noaslr.wrap_command(command)  # Stage 2/3 determinism fix, OGRL-20260815-038

    started = time.monotonic()
    peak_rss_kib = 0
    peak_cpu_percent = 0.0
    peak_threads = 0
    cpu_samples: list[float] = []
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
            if cpu_percent > 0.0:
                cpu_samples.append(cpu_percent)
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
        "mean_cpu_percent": statistics.mean(cpu_samples) if cpu_samples else 0.0,
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
    if not keep_write_dir:
        # Each run's --write-dir accumulates asset/script/sound caches; left
        # behind across many runs this leaks disk fast (research-log
        # OGRL-20260815-034 filled the host disk from exactly this in
        # concurrency_sweep.py). Delete by default; --keep-write-dirs opts out
        # for debugging a specific run.
        shutil.rmtree(write_dir, ignore_errors=True)
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

    completed_runs = [run for run in runs if run["benchmark_completed"] == 1]
    diagnostic_values = {
        "empty_world_steps_per_second": by_scenario.get("empty", {}).get("median_steps_per_second", 0.0),
        "duel_steps_per_second": by_scenario.get("duel", {}).get("median_steps_per_second", 0.0),
        "four_actor_steps_per_second": by_scenario.get("four_character", {}).get("median_steps_per_second", 0.0),
        "six_actor_steps_per_second": by_scenario.get("six_character", {}).get("median_steps_per_second", 0.0),
        "best_worker_count": 1,
        "aggregate_decisions_per_second_20hz": aggregate / 6.0,
        "startup_seconds": median(
            [float(run["engine_initialize_seconds"]) + float(run["level_load_seconds"]) for run in completed_runs]
        ),
        "shader_preload_seconds": median([float(run["shader_preload_seconds"]) for run in completed_runs]),
        "reset_latency_ms": -1.0,
        "peak_rss_mb": max([float(run["peak_rss_mib"]) for run in runs], default=0.0),
        "mean_cpu_percent": statistics.mean([float(run["mean_cpu_percent"]) for run in runs]) if runs else 0.0,
        "observation_extraction_percent": -1.0,
        "equivalence_max_position_error": 0.0,
        "equivalence_max_velocity_error": 0.0,
    }

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "suite": suite,
        "run_count": len(runs),
        "aggregate_useful_steps_per_second": aggregate,
        "aggregate_decisions_per_second_20hz": aggregate / 6.0,
        "benchmark_completed": int(all_complete),
        "combat_suite_pass_rate": combat_complete / combat_requested if combat_requested else 0.0,
        "fixed_physics_hz": 120 if all_complete else 0,
        # This is a completion flag, not a correctness measurement -- it only
        # says every requested run finished. `equivalence_pass_rate` is reserved
        # for Stage 1's replay_compare.py comparator (Tools/rl/replay_compare.py),
        # which is the only thing entitled to claim gameplay equivalence.
        "all_runs_completed": 1.0 if all_complete else 0.0,
        "isolated_write_dir": int(all(run["isolated_write_dir"] == 1 for run in runs)),
        "scenario_summary": by_scenario,
        "runs": runs,
    }
    summary.update(diagnostic_values)
    return summary


def stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def run_paired(
    repo_root: Path,
    reference_binary: Path,
    candidate_binary: Path,
    config: dict[str, Any],
    suite: str,
    repeats: int,
    artifact_root: Path,
    margin_percent: float,
    keep_write_dirs: bool = False,
) -> dict[str, Any]:
    """Implements the section 2.1 measurement protocol: reference -> candidate ->
    reference, interleaved, per repetition, so the candidate is always compared
    against the median of a reference measured immediately before and after it in
    the same session and thermal state -- never against a stored historical best.
    """
    scenario_names = config["suites"][suite]
    per_scenario: dict[str, Any] = {}
    all_pass = True

    for scenario_name in scenario_names:
        level = config["scenarios"][scenario_name]
        triples: list[dict[str, Any]] = []
        for repetition in range(1, repeats + 1):
            base_seed = int(config["seed"]) + repetition - 1
            common = dict(
                repo_root=repo_root,
                artifact_root=artifact_root,
                scenario_name=scenario_name,
                level=level,
                repetition=repetition,
                warmup_steps=int(config["warmup_steps"]),
                measure_steps=int(config["measure_steps"]),
                seed=base_seed,
                timeout_seconds=float(config["timeout_seconds"]),
                keep_write_dir=keep_write_dirs,
            )
            # Same seed for all three legs of a triple: the candidate must be
            # compared against a reference run on identical inputs, and the two
            # reference legs bracket it in wall-clock/thermal time.
            ref_pre = run_one(binary=reference_binary, **common)
            candidate = run_one(binary=candidate_binary, **common)
            ref_post = run_one(binary=reference_binary, **common)

            ok = all(
                run["benchmark_completed"] == 1 for run in (ref_pre, candidate, ref_post)
            )
            all_pass = all_pass and ok

            ref_pre_rate = float(ref_pre["steps_per_second"])
            ref_post_rate = float(ref_post["steps_per_second"])
            candidate_rate = float(candidate["steps_per_second"])
            paired_reference_rate = median([ref_pre_rate, ref_post_rate])
            delta_percent = (
                (candidate_rate - paired_reference_rate) / paired_reference_rate * 100.0
                if paired_reference_rate > 0.0
                else 0.0
            )

            triples.append(
                {
                    "repetition": repetition,
                    "seed": base_seed,
                    "completed": int(ok),
                    "reference_pre_steps_per_second": ref_pre_rate,
                    "candidate_steps_per_second": candidate_rate,
                    "reference_post_steps_per_second": ref_post_rate,
                    "paired_reference_steps_per_second": paired_reference_rate,
                    "delta_percent": delta_percent,
                    "reference_pre_run": ref_pre,
                    "candidate_run": candidate,
                    "reference_post_run": ref_post,
                }
            )

        completed_triples = [t for t in triples if t["completed"] == 1]
        candidate_rates = [t["candidate_steps_per_second"] for t in completed_triples]
        paired_reference_rates = [t["paired_reference_steps_per_second"] for t in completed_triples]
        delta_percents = [t["delta_percent"] for t in completed_triples]

        candidate_median = median(candidate_rates)
        candidate_stdev = stdev_or_zero(candidate_rates)
        paired_reference_median = median(paired_reference_rates)
        paired_reference_stdev = stdev_or_zero(paired_reference_rates)
        delta_percent_median = median(delta_percents)
        delta_percent_stdev = stdev_or_zero(delta_percents)

        # Section 2.1.2: if the sample standard deviation of the measured effect
        # exceeds the effect itself, there is no measurable signal -- say so
        # rather than reporting a delta that noise could equally explain.
        no_measurable_signal = abs(delta_percent_median) <= delta_percent_stdev
        beats_margin = delta_percent_median > margin_percent

        per_scenario[scenario_name] = {
            "repetitions_requested": repeats,
            "repetitions_completed": len(completed_triples),
            "candidate_median_steps_per_second": candidate_median,
            "candidate_stdev_steps_per_second": candidate_stdev,
            "paired_reference_median_steps_per_second": paired_reference_median,
            "paired_reference_stdev_steps_per_second": paired_reference_stdev,
            "delta_percent_median": delta_percent_median,
            "delta_percent_stdev": delta_percent_stdev,
            "no_measurable_signal": bool(no_measurable_signal),
            "beats_margin": bool(beats_margin and not no_measurable_signal),
            "margin_percent": margin_percent,
            "triples": triples,
        }

    return {
        "schema_version": 1,
        "mode": "paired",
        "suite": suite,
        "repeats": repeats,
        "margin_percent": margin_percent,
        "reference_binary": str(reference_binary),
        "candidate_binary": str(candidate_binary),
        "all_runs_completed": bool(all_pass),
        "scenario_summary": per_scenario,
        "verdict": {
            name: ("keep" if scenario["beats_margin"] else "reject_or_no_signal")
            for name, scenario in per_scenario.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.yaml"))
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--keep-write-dirs",
        action="store_true",
        help="Do not delete per-run --write-dir trees after each run (default: delete immediately; see OGRL-20260815-034).",
    )
    parser.add_argument(
        "--paired",
        nargs=2,
        metavar=("REFERENCE_BINARY", "CANDIDATE_BINARY"),
        type=Path,
        help=(
            "Section 2.1 paired A/B mode: run reference -> candidate -> reference, "
            "interleaved, per repetition, and report the candidate against the "
            "median of the paired reference legs with standard deviations."
        ),
    )
    parser.add_argument(
        "--margin-percent",
        type=float,
        default=2.0,
        help="Paired mode only: minimum delta_percent_median over the paired reference to accept a change (section 2.1.3).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config.resolve())
    if args.suite not in config["suites"]:
        raise SystemExit(f"unknown suite: {args.suite}")
    artifact_root = (repo_root / config["artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.paired is not None:
        reference_binary, candidate_binary = (p.resolve() for p in args.paired)
        if not reference_binary.is_file():
            raise SystemExit(f"reference binary not found: {reference_binary}")
        if not candidate_binary.is_file():
            raise SystemExit(f"candidate binary not found: {candidate_binary}")
        paired_summary = run_paired(
            repo_root=repo_root,
            reference_binary=reference_binary,
            candidate_binary=candidate_binary,
            config=config,
            suite=args.suite,
            repeats=max(3, args.repeats),  # section 2.1.2: three repetitions minimum
            artifact_root=artifact_root,
            margin_percent=args.margin_percent,
            keep_write_dirs=args.keep_write_dirs,
        )
        suffix = f"paired-{args.suite}-r{max(3, args.repeats)}"
        summary_path = artifact_root / f"summary-{suffix}.json"
        summary_path.write_text(json.dumps(paired_summary, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(paired_summary, separators=(",", ":")))
        else:
            print(json.dumps(paired_summary, indent=2))
            print(f"summary: {summary_path}")
        return 0 if paired_summary["all_runs_completed"] else 1

    binary = (args.binary or (repo_root / config["binary"])).resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary not found: {binary}")

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
                    keep_write_dir=args.keep_write_dirs,
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
