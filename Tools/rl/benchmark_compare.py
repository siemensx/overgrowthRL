#!/usr/bin/env python3
"""Paired, same-engine checkpoint benchmark for the hard 1v3 cell.

This is deliberately separate from training telemetry. It evaluates two
frozen checkpoints in the same checkout with identical held-out seeds and
reports both ordinary Wilson intervals and a paired bootstrap interval for the
candidate-minus-baseline win-rate delta. A throughput comparison must use
``throughput_sweep.py`` and its own fixed manifest; this tool never treats
policy quality as a simulator-speed measurement.

Example::

    python Tools/rl/benchmark_compare.py \
      --baseline Tools/rl/ppo/checkpoints/run21_baseline_260m.pt \
      --candidate Tools/rl/ppo/checkpoints/run24_opt_smoke_20260921.pt \
      --repo-root . --episodes 400 --seed-base 900000 \
      --level arenas/t_train_101.xml --opponents 3 --difficulty-bands 1.0 \
      --out research-artifacts/OGRL-20260921-005-benchmark/policy_compare.json

The evaluator emits per-episode records only when requested here, so the two
rows can be joined by seed. This makes a two-percentage-point-looking change
auditable instead of relying on two independent aggregate rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one(args: argparse.Namespace, label: str, checkpoint: Path, stochastic: bool, raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{label}_{'stochastic' if stochastic else 'deterministic'}.json"
    log = raw_dir / f"{label}_{'stochastic' if stochastic else 'deterministic'}.log"
    evaluate = Path(args.repo_root) / "Tools" / "rl" / "evaluate.py"
    shm = f"/ogrl_bench_{label}_{'stoch' if stochastic else 'greedy'}"
    command = [
        sys.executable, "-u", str(evaluate),
        "--checkpoint", str(checkpoint), "--repo-root", str(args.repo_root),
        "--level", args.level, "--frame-stack", str(args.frame_stack),
        "--act-period", str(args.act_period), "--episodes", str(args.episodes),
        "--seed-base", str(args.seed_base), "--difficulty-bands", args.difficulty_bands,
        "--opponents", str(args.opponents), "--weapons", str(args.weapons),
        "--species", str(args.species), "--max-episode-steps", str(args.max_episode_steps),
        "--no-control", "--emit-episodes", "--device", args.device,
        "--shm-name", shm, "--out", str(output),
    ]
    if stochastic:
        command.append("--stochastic")
    for line in args.config_line:
        command.extend(["--config-line", line])
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=args.repo_root, stdout=handle, stderr=subprocess.STDOUT,
                                   env=os.environ.copy(), check=False)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(f"{label} evaluation failed with exit={completed.returncode}; see {log}")
    result = json.loads(output.read_text(encoding="utf-8"))
    result["benchmark_label"] = label
    result["stochastic"] = stochastic
    return result


def bootstrap_delta(baseline: list[dict], candidate: list[dict], samples: int, seed: int) -> dict:
    base_by_seed = {int(row["seed"]): row for row in baseline}
    cand_by_seed = {int(row["seed"]): row for row in candidate}
    common = sorted(set(base_by_seed) & set(cand_by_seed))
    if len(common) != len(base_by_seed) or len(common) != len(cand_by_seed):
        raise ValueError("baseline and candidate episode seed sets differ")
    deltas = np.asarray([
        int(cand_by_seed[s]["won"]) - int(base_by_seed[s]["won"])
        for s in common
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    estimates = deltas[indices].mean(axis=1)
    return {
        "paired_episodes": len(common),
        "baseline_only_wins": int(sum(base_by_seed[s]["won"] and not cand_by_seed[s]["won"] for s in common)),
        "candidate_only_wins": int(sum(cand_by_seed[s]["won"] and not base_by_seed[s]["won"] for s in common)),
        "delta_win_rate": float(deltas.mean()),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "delta_ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def compare_band(base: dict, cand: dict, samples: int, seed: int) -> dict:
    base_policy = base["policy"]
    cand_policy = cand["policy"]
    return {
        "band": base["band"],
        "baseline": {
            "global_step": base.get("global_step"),
            "outcomes": base_policy["outcomes"],
            "win_rate": base_policy["win_rate"],
            "win_rate_ci95": base_policy["win_rate_ci95"],
        },
        "candidate": {
            "global_step": cand.get("global_step"),
            "outcomes": cand_policy["outcomes"],
            "win_rate": cand_policy["win_rate"],
            "win_rate_ci95": cand_policy["win_rate_ci95"],
        },
        "paired": bootstrap_delta(base_policy["episode_results"], cand_policy["episode_results"], samples, seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--level", default="arenas/t_train_101.xml")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--act-period", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--difficulty-bands", default="1.0")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--weapons", type=float, default=0.0)
    parser.add_argument("--species", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--config-line", action="append", default=[])
    parser.add_argument("--mode", choices=["deterministic", "stochastic", "both"], default="deterministic")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    return args


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline).resolve()
    candidate = Path(args.candidate).resolve()
    for path in (baseline, candidate):
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist: {path}")
    out = Path(args.out).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else Path(tempfile.mkdtemp(prefix="ogrl-benchmark-"))
    modes = [False, True] if args.mode == "both" else [args.mode == "stochastic"]
    results = []
    for stochastic in modes:
        base = run_one(args, "baseline", baseline, stochastic, raw_dir)
        cand = run_one(args, "candidate", candidate, stochastic, raw_dir)
        if len(base["bands"]) != len(cand["bands"]):
            raise ValueError("baseline and candidate difficulty-band counts differ")
        comparisons = [compare_band(b, c, args.bootstrap_samples, args.bootstrap_seed)
                       for b, c in zip(base["bands"], cand["bands"])]
        results.append({"stochastic": stochastic, "comparisons": comparisons})
    manifest = {
        "protocol_version": "OGRL-benchmark-20260921-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": args.repo_root,
        "git_sha": git_sha(Path(args.repo_root)),
        "baseline_checkpoint": {"path": str(baseline), "sha256": sha256(baseline)},
        "candidate_checkpoint": {"path": str(candidate), "sha256": sha256(candidate)},
        "settings": {k: getattr(args, k) for k in (
            "level", "frame_stack", "act_period", "episodes", "seed_base", "difficulty_bands",
            "opponents", "weapons", "species", "max_episode_steps", "device", "config_line")},
        "raw_dir": str(raw_dir),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
