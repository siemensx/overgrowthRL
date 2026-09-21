#!/usr/bin/env python3
"""Parallel bench: K evaluate.py processes on disjoint seed slices, merged.

One evaluate.py drives ONE engine and takes ~20 minutes for 200 episodes. On
the trainer that pause was 17% of a 10M-step arm. K processes on disjoint seed
ranges finish in ~1/K the time and produce the same statistics; the merged
JSON keeps evaluate.py's shape (bands[0].policy.{outcomes,win_rate,...}) so
every reader keeps working.

    python Tools/rl/bench.py --checkpoint X.pt --episodes 400 --parallel 4 --out out.json [any evaluate.py flags]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=900_000)
    ap.add_argument("--shm-name", default="/ogrl_bench")
    ap.add_argument("--out", required=True)
    args, passthrough = ap.parse_known_args()

    per = [args.episodes // args.parallel + (1 if i < args.episodes % args.parallel else 0) for i in range(args.parallel)]
    tmpdir = Path(tempfile.mkdtemp(prefix="bench-"))
    procs, outs = [], []
    offset = 0
    evaluate = str(Path(__file__).resolve().parent / "evaluate.py")
    for i, n in enumerate(per):
        if n <= 0:
            continue
        out = tmpdir / f"slice{i}.json"
        cmd = [sys.executable, evaluate, "--checkpoint", args.checkpoint, "--episodes", str(n),
               "--seed-base", str(args.seed_base + offset), "--shm-name", f"{args.shm_name}{i}",
               "--no-control", "--out", str(out)] + passthrough
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        outs.append(out)
        offset += n
    for p in procs:
        p.wait()

    slices = [json.loads(o.read_text()) for o in outs if o.exists()]
    if not slices:
        print("bench: no slice produced output", file=sys.stderr)
        return 1
    merged = json.loads(json.dumps(slices[0]))
    band = merged["bands"][0]["policy"]
    outcomes = {"won": 0, "lost": 0, "timeout": 0}
    comp_sum, len_sum, n_total = {}, 0.0, 0
    for s in slices:
        b = s["bands"][0]["policy"]
        n = b["episodes"]
        for k in outcomes:
            outcomes[k] += b["outcomes"].get(k, 0)
        for k, v in b.get("reward_components_mean", {}).items():
            comp_sum[k] = comp_sum.get(k, 0.0) + v * n
        len_sum += b.get("episode_length_mean", 0.0) * n
        n_total += n
    band["episodes"] = n_total
    band["outcomes"] = outcomes
    band["win_rate"] = outcomes["won"] / max(1, n_total)
    band["win_rate_ci95"] = wilson(outcomes["won"], n_total)
    band["reward_components_mean"] = {k: v / n_total for k, v in comp_sum.items()}
    band["episode_length_mean"] = len_sum / max(1, n_total)
    band["slices"] = [{"episodes": s["bands"][0]["policy"]["episodes"], "won": s["bands"][0]["policy"]["outcomes"]["won"],
                       "seed_base": s.get("seed_base")} for s in slices]
    merged["episodes"] = n_total
    merged["parallel"] = len(slices)
    merged["overall"] = {"win_rate": band["win_rate"], "win_rate_ci95": band["win_rate_ci95"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(merged, indent=2) + "\n")
    print(f"bench: {outcomes['won']}/{n_total} = {band['win_rate']:.3f} {band['win_rate_ci95']}  ({len(slices)} slices)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
