#!/usr/bin/env python3
"""Isolate the Stage 0.3 overlapping-window correction from thermal/host drift.

Runs naive (pre-Stage-0 method: Popen loop, fixed step count, no barrier) and
corrected (barrier + fixed wall-clock window) back-to-back, same session, same
seeds, for each requested worker count -- so any delta reflects the correction
itself, not a comparison against a historical artifact from a colder machine.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrency_sweep import require_free_disk_space, run_sweep_point  # noqa: E402


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--level", default="arenas/oval_arena.xml")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 6, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for workers in args.workers:
        naive_rates, corrected_rates = [], []
        pair_log = []
        for rep in range(args.repeats):
            require_free_disk_space(artifact_root)
            seed_base = 20260815 + rep * 1000
            naive_point = run_sweep_point(
                repo_root=repo_root, binary=binary, artifact_root=artifact_root,
                level=args.level, workers=workers, warmup_steps=600,
                measure_seconds=args.measure_seconds, step_safety_cap=2400,
                seed_base=seed_base, timeout_seconds=120.0, method="naive",
            )
            corrected_point = run_sweep_point(
                repo_root=repo_root, binary=binary, artifact_root=artifact_root,
                level=args.level, workers=workers, warmup_steps=600,
                measure_seconds=args.measure_seconds, step_safety_cap=2_000_000,
                seed_base=seed_base, timeout_seconds=120.0, method="corrected",
            )
            naive_rates.append(naive_point["naive_sum_steps_per_second"])
            corrected_rates.append(corrected_point["overlapping_aggregate_steps_per_second"])
            pair_log.append({"rep": rep, "naive": naive_point["naive_sum_steps_per_second"], "corrected": corrected_point["overlapping_aggregate_steps_per_second"]})
            print(f"workers={workers} rep={rep} naive={naive_point['naive_sum_steps_per_second']:.1f} corrected={corrected_point['overlapping_aggregate_steps_per_second']:.1f}")

        naive_median = statistics.median(naive_rates)
        corrected_median = statistics.median(corrected_rates)
        all_results[workers] = {
            "naive_median": naive_median,
            "naive_stdev": statistics.stdev(naive_rates) if len(naive_rates) >= 2 else 0.0,
            "corrected_median": corrected_median,
            "corrected_stdev": statistics.stdev(corrected_rates) if len(corrected_rates) >= 2 else 0.0,
            "delta_percent": (corrected_median - naive_median) / naive_median * 100.0 if naive_median > 0 else 0.0,
            "pairs": pair_log,
        }

    out_path = artifact_root / "concurrency-method-ab-same-session.json"
    out_path.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(all_results, indent=2))
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
