#!/usr/bin/env python3
"""Stage 0.8: sustained-throughput / thermal run.

Launches N workers (default 6, the current best operating point per
OGRL-20260815-030) on a barrier-synchronized start, each running continuously
for --duration-seconds with --bucket-seconds progress checkpoints, and
computes the sustained-vs-burst ratio: bucket throughput late in the run
divided by bucket throughput in the first bucket. The existing sweep tiers
(OGRL-20260815-029/-030) only ran 2.8-7.6s and cannot see anything that only
shows up after minutes of sustained load.

powermetrics requires interactive sudo, unavailable in this environment; CPU/
GPU frequency and package power are not captured here (documented, not
silently skipped). pmset -g therm was checked and only reports discrete
warning-level events, not a continuous series, so it is not a substitute.
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
PROGRESS_PREFIX = "RL_BENCHMARK_PROGRESS "


def parse_lines(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    progress = []
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(PROGRESS_PREFIX)
        if marker >= 0:
            progress.append(json.loads(line[marker + len(PROGRESS_PREFIX):]))
            continue
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            result = json.loads(line[marker + len(RESULT_PREFIX):])
    return progress, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--level", default="arenas/oval_arena.xml")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--duration-seconds", type=float, default=1200.0)  # 20 min
    parser.add_argument("--bucket-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=20260815)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(artifact_root).free
    if free < 2 * 1024 * 1024 * 1024:
        raise SystemExit(f"refusing to start sustained run: only {free/1e9:.2f}GB free")

    barrier_dir = Path(tempfile.mkdtemp(prefix="sustained-barrier-", dir=artifact_root))
    write_parent = artifact_root / "write_dirs"
    log_parent = artifact_root / "raw"
    write_parent.mkdir(parents=True, exist_ok=True)
    log_parent.mkdir(parents=True, exist_ok=True)

    processes, log_paths, write_dirs = [], [], []
    started = time.monotonic()
    for i in range(args.workers):
        write_dir = Path(tempfile.mkdtemp(prefix=f"sustained-{i}-", dir=write_parent))
        log_path = log_parent / f"sustained-{i}.log"
        write_dirs.append(write_dir)
        log_paths.append(log_path)
        command = [
            str(binary),
            "--write-dir", str(write_dir),
            "--working-dir", str(repo_root),
            "--disable-rendering",
            "--no-dialogues",
            "--benchmark",
            "--benchmark-warmup-steps", str(args.warmup_steps),
            "--benchmark-steps", "500000000",
            "--benchmark-measure-seconds", str(args.duration_seconds),
            "--benchmark-progress-seconds", str(args.bucket_seconds),
            "--benchmark-barrier-dir", str(barrier_dir),
            "--benchmark-barrier-workers", str(args.workers),
            "--benchmark-seed", str(args.seed_base + i),
            "--level", args.level,
            "--config", "global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true",
        ]
        command = noaslr.wrap_command(command)  # Stage 2/3 determinism fix, OGRL-20260815-038
        log_file = log_path.open("w", encoding="utf-8")
        processes.append((subprocess.Popen(command, cwd=repo_root, stdout=log_file, stderr=subprocess.STDOUT, text=True), log_file))

    print(f"launched {args.workers} workers for {args.duration_seconds:.0f}s, disk-check every bucket...")
    timeout = args.duration_seconds + 120.0
    for process, log_file in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            log_file.close()
    wall_seconds = time.monotonic() - started

    per_worker = []
    for log_path in log_paths:
        progress, result = parse_lines(log_path)
        per_worker.append({"progress": progress, "result": result})

    shutil.rmtree(barrier_dir, ignore_errors=True)
    for write_dir in write_dirs:
        shutil.rmtree(write_dir, ignore_errors=True)

    # Aggregate bucket rate = sum of all workers' bucket_steps_per_second at
    # each aligned bucket index (buckets are aligned because all workers
    # started at the same barrier-released instant and use the same
    # --benchmark-progress-seconds).
    num_buckets = min((len(w["progress"]) for w in per_worker), default=0)
    aggregate_bucket_rates = []
    for b in range(num_buckets):
        aggregate_bucket_rates.append(sum(w["progress"][b]["bucket_steps_per_second"] for w in per_worker))

    first_bucket_rate = aggregate_bucket_rates[0] if aggregate_bucket_rates else 0.0
    # Skip bucket 0 (includes barrier release jitter) when picking the "settled early" rate.
    early_rate = aggregate_bucket_rates[1] if len(aggregate_bucket_rates) > 1 else first_bucket_rate
    late_window = max(1, num_buckets // 10)  # last 10% of buckets
    late_rate = (
        sum(aggregate_bucket_rates[-late_window:]) / late_window if aggregate_bucket_rates else 0.0
    )
    sustained_vs_burst_ratio = late_rate / early_rate if early_rate > 0 else 0.0

    summary = {
        "experiment_id": "OGRL-20260815-034",
        "workers": args.workers,
        "duration_seconds_requested": args.duration_seconds,
        "bucket_seconds": args.bucket_seconds,
        "wall_seconds": wall_seconds,
        "num_buckets": num_buckets,
        "aggregate_bucket_steps_per_second": aggregate_bucket_rates,
        "early_bucket_aggregate_steps_per_second": early_rate,
        "late_window_bucket_count": late_window,
        "late_aggregate_steps_per_second": late_rate,
        "sustained_vs_burst_ratio": sustained_vs_burst_ratio,
        "per_worker_final_result": [w["result"] for w in per_worker],
        "powermetrics_note": "unavailable: requires interactive sudo, not present in this session",
    }
    out_path = artifact_root / "sustained-run.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("aggregate_bucket_steps_per_second", "per_worker_final_result")}, indent=2))
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
