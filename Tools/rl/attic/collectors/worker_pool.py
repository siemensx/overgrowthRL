#!/usr/bin/env python3
"""Stage 4 Approach A, production-shaped: N active worker slots + K pre-warmed
standby processes, sharing one replenishment pool. When any active slot's
episode ends, it is immediately replaced by a ready standby (swap latency
bounded by process-handle bookkeeping, not level load); a new standby is
spawned right away to keep the pool at depth K. This is the general form of
the single-slot demo in process_pool_bakeoff.py (OGRL-20260815-04x), sized
for the real target worker count instead of one slot.

This still runs --benchmark-measure-seconds episodes (no live Gym action
injection exists yet -- that's Stage 5), so "episode" here means "one
benchmark measurement window," which is the correct proxy for restart
overhead: the pool mechanic is agnostic to what happens inside the episode.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noaslr  # noqa: E402

RESULT_PREFIX = "RL_BENCHMARK_RESULT "


def parse_result(log_path: Path) -> dict[str, Any] | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    for line in reversed(text.splitlines()):
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            return json.loads(line[marker + len(RESULT_PREFIX):])
    return None


def peak_rss_mib(pid: int) -> float:
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
        fields = result.stdout.split()
        return int(fields[0]) / 1024.0 if fields else 0.0
    except (OSError, ValueError, IndexError):
        return 0.0


@dataclass
class Handle:
    process: subprocess.Popen
    log_file: Any
    write_dir: Path
    log_path: Path
    spawned_at: float
    seed: int


@dataclass
class Pool:
    repo_root: Path
    binary: Path
    artifact_root: Path
    level: str
    warmup_steps: int
    episode_seconds: float
    seed_counter: int = 20260815
    active: list[Handle] = field(default_factory=list)
    standby: list[Handle] = field(default_factory=list)
    k_standby_target: int = 0

    def _spawn(self, label: str) -> Handle:
        seed = self.seed_counter
        self.seed_counter += 1
        write_dir = Path(tempfile.mkdtemp(prefix=f"{label}-{seed}-", dir=self.artifact_root / "write_dirs"))
        log_path = self.artifact_root / "raw" / f"{label}-{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.binary),
            "--write-dir", str(write_dir),
            "--working-dir", str(self.repo_root),
            "--disable-rendering",
            "--no-dialogues",
            "--benchmark",
            "--benchmark-warmup-steps", str(self.warmup_steps),
            "--benchmark-steps", "500000000",
            "--benchmark-measure-seconds", str(self.episode_seconds),
            "--benchmark-seed", str(seed),
            "--level", self.level,
            "--config", "global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true",
        ]
        command = noaslr.wrap_command(command)  # Stage 2/3 determinism fix, OGRL-20260815-038
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=self.repo_root, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        return Handle(process, log_file, write_dir, log_path, time.monotonic(), seed)

    def fill(self, n_active: int, k_standby: int) -> None:
        self.k_standby_target = k_standby
        while len(self.active) < n_active:
            self.active.append(self._spawn("active"))
        while len(self.standby) < k_standby:
            self.standby.append(self._spawn("standby"))

    def _finish(self, handle: Handle) -> dict[str, Any]:
        handle.process.wait()
        handle.log_file.close()
        result = parse_result(handle.log_path)
        shutil.rmtree(handle.write_dir, ignore_errors=True)
        return result or {}

    def run(self, wall_budget_seconds: float, poll_seconds: float) -> dict[str, Any]:
        started_at = time.monotonic()
        episodes: list[dict[str, Any]] = []
        peak_rss_samples: list[float] = []

        while time.monotonic() - started_at < wall_budget_seconds:
            # Sample peak RSS across the whole pool periodically for the memory-budget check.
            total_rss = sum(peak_rss_mib(h.process.pid) for h in self.active + self.standby)
            if total_rss > 0:
                peak_rss_samples.append(total_rss)

            finished_index = None
            for i, handle in enumerate(self.active):
                if handle.process.poll() is not None:
                    finished_index = i
                    break
            if finished_index is None:
                time.sleep(poll_seconds)
                continue

            finished_handle = self.active[finished_index]
            episode_end_wall = time.monotonic()
            result = self._finish(finished_handle)

            # Swap in a ready standby if one exists; otherwise (pool underfilled
            # relative to churn rate) spawn fresh and pay the full cold-load cost.
            if self.standby:
                replacement = self.standby.pop(0)
                swap_latency = max(0.0, time.monotonic() - episode_end_wall)
                pool_underrun = False
            else:
                replacement = self._spawn("active-underrun")
                swap_latency = None  # not measurable the same way; this is a cold spawn, not a swap
                pool_underrun = True
            self.active[finished_index] = replacement
            if len(self.standby) < self.k_standby_target:
                self.standby.append(self._spawn("standby"))  # replenish up to target only -- a k_standby_target=0
                # pool must stay a true cold-restart baseline, not accidentally
                # regrow a pool from the replacement churn.

            episodes.append({
                "seed": result.get("seed"),
                "measured_steps": result.get("measured_steps"),
                "swap_latency_seconds": swap_latency,
                "pool_underrun": pool_underrun,
                "wall_time": episode_end_wall - started_at,
            })

        wall_seconds = time.monotonic() - started_at
        for handle in self.active + self.standby:
            handle.process.kill()
            handle.process.wait()
            handle.log_file.close()
            shutil.rmtree(handle.write_dir, ignore_errors=True)

        swap_latencies = [e["swap_latency_seconds"] for e in episodes if e["swap_latency_seconds"] is not None]
        underruns = sum(1 for e in episodes if e["pool_underrun"])
        return {
            "n_active": len(self.active),
            "k_standby_target": None,  # filled in by caller
            "episodes_completed": len(episodes),
            "wall_seconds": wall_seconds,
            "episodes_per_hour_aggregate": len(episodes) / wall_seconds * 3600.0 if wall_seconds > 0 else 0.0,
            "mean_swap_latency_seconds": statistics.mean(swap_latencies) if swap_latencies else 0.0,
            "max_swap_latency_seconds": max(swap_latencies, default=0.0),
            "pool_underrun_count": underruns,
            "peak_total_rss_mib": max(peak_rss_samples, default=0.0),
            "mean_total_rss_mib": statistics.mean(peak_rss_samples) if peak_rss_samples else 0.0,
            "episodes": episodes,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--level", default="arenas/oval_arena.xml")
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--episode-seconds", type=float, default=2.0)
    parser.add_argument("--n-active", type=int, default=6, help="Concurrent active slots (6 = current best operating point, OGRL-20260815-030/034).")
    parser.add_argument("--k-standby", type=int, default=2, help="Standby pool depth (plan suggests K=2-3 on 16GB with 6 active workers).")
    parser.add_argument("--wall-budget-seconds", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary not found: {binary}")
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "write_dirs").mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(artifact_root).free
    if free < 2 * 1024 * 1024 * 1024:
        raise SystemExit(f"refusing to start: only {free/1e9:.2f}GB free")

    pool = Pool(repo_root, binary, artifact_root, args.level, args.warmup_steps, args.episode_seconds)
    pool.fill(args.n_active, args.k_standby)
    result = pool.run(args.wall_budget_seconds, args.poll_seconds)
    result["k_standby_target"] = args.k_standby

    out_path = artifact_root / f"worker-pool-n{args.n_active}-k{args.k_standby}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
