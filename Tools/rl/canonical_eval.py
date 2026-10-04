#!/usr/bin/env python3
"""THE benchmark (suite v2, frozen 2026-10-02, OGRL-20261002-007). Use this and nothing else
when a number is meant to say "the policy got better".

Why a new one: by 2026-10-02 eight evaluation definitions coexisted (evaluate.py defaults,
the periodic bench, gate evals, comprehensive_eval.sh, multi_opponent_eval.py,
transfer_eval.py, benchmark_compare.py, greedy_ab.py) with different maps, caps, seeds and
-- worst -- different engine control flags. The same 260M checkpoint read 46/100 or 9/100
depending only on whether `rl_button_edges` was passed (OGRL-20261002-001).

Fixed definition (do not edit; make a v3 instead):
  * maps    training t_train_101, t_train_102, t_train_104; held-out t_held_203
  * cells   1v1, 1v2, 1v3, all at difficulty 1.0, unarmed
  * n       100 episodes per cell (1,200 per suite), greedy policy, 1200-decision cap
  * seeds   7,000,000 + 1000*cell_index + episode  -- never used by training or older benches
  * resets  hard reset every episode, fresh engine process every 20 episodes, fixed episode order
  * controls must be named explicitly (--controls); the profile's engine flags are recorded

Output: one JSON with per-cell wins + Wilson CI + per-episode outcomes, plus a printed
table with TRAIN and HELD-OUT aggregates per opponent count.

Determinism (measured 2026-10-02, OGRL-20261002-007b/c): NOT bit-exact.
  * v5 engine: two passes over the same 100 seeds in one process each -> 34 vs 39 wins, identical
    episode length on 72/100, divergence from episode 0 (v5's garbage `grounded` read).
  * v6 engine: identical for episodes 0-20 of a process, then drifts (process-history effect).
  * v6, one fresh process per episode: identical on 24/30 seeds -- a residual source remains
    (suspected wall-clock/load dependence; the 2026-08-15 test under light load was 100%).
So each cell runs as CHUNK=20-episode processes (removes the history effect), and a difference of
<= 5/100 between two numbers is noise. Compare checkpoints by paired per-seed outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_VERSION = "v2-2026-10-02"
TRAIN_MAPS = ["t_train_101", "t_train_102", "t_train_104"]
HELD_MAPS = ["t_held_203"]
OPPONENTS = [1, 2, 3]
EPISODES = 100
SEED0 = 7_000_000
CONTROLS = {
    "corrected": ["rl_target_select: 2", "rl_button_edges: 1"],
    "corrected-nofeint": ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1"],
    # v6 branch (obs schema v6): the v2-agent profile -- sees every character + AI state, no feint,
    # stick dead zone (dodge/frontkick), stance walk (backpedal). Requires a v6 engine + v6 checkpoint.
    "v6-omni": ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
                "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"],
    "old": [],  # auto-repeating buttons + frozen-camera targeting: everything trained before 2026-09-24
    # "old" (turbo buttons, frozen-camera targeting) plus the v6 see-everyone listing. Legal under the
    # 2026-10-02 fairness decision; for turbo-era checkpoints transplanted by surgery_v5_to_v6.py.
    "v6-omni-old": ["rl_obs_omniscient: 1"],
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def cells():
    out = []
    for i, (m, o) in enumerate([(m, o) for m in TRAIN_MAPS + HELD_MAPS for o in OPPONENTS]):
        out.append({"idx": i, "map": m, "opp": o, "split": "held" if m in HELD_MAPS else "train",
                    "seed_base": SEED0 + 1000 * i})
    return out


CHUNK = 20  # episodes per engine process; see the determinism note in the module docstring


def _run_chunk(ckpt, cell, flags, out, seed_base, episodes):
    if out.exists():
        return
    cmd = [sys.executable, str(HERE / "evaluate.py"), "--checkpoint", ckpt,
           "--level", f"arenas/{cell['map']}.xml", "--opponents", str(cell["opp"]),
           "--difficulty-bands", "1.0", "--episodes", str(episodes), "--max-episode-steps", "1200",
           "--frame-stack", "4", "--act-period", "4", "--no-control", "--emit-episodes",
           # unique per launch (a reused shm name hangs, DEAD_ENDS) and short (macOS caps POSIX names ~31 chars)
           "--seed-base", str(seed_base), "--shm-name", f"/ogc{cell['idx']}_{os.getpid() % 1000}_{int(time.time() * 1000) % 100000}",
           "--out", str(out)]
    for f in flags:
        cmd += ["--config-line", f]
    with open(out.with_suffix(".log"), "w") as lf:
        subprocess.call(cmd, cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT)


def run_cell(ckpt: str, cell: dict, flags: list[str], out_dir: Path, episodes: int, tag: str) -> dict:
    """One cell = ceil(episodes / CHUNK) fresh engine processes on consecutive seed slices.
    OGRL-20261002-007c: within one process, runs stay identical only for the first ~20 episodes."""
    won, n, outcomes, eps = 0, 0, {"won": 0, "lost": 0, "timeout": 0}, []
    for k in range(0, episodes, CHUNK):
        m = min(CHUNK, episodes - k)
        out = out_dir / f"{tag}_{cell['map']}_{cell['opp']}v_s{k:03d}.json"
        _run_chunk(ckpt, cell, flags, out, cell["seed_base"] + k, m)
        pol = json.loads(out.read_text())["bands"][0]["policy"]
        for key in outcomes:
            outcomes[key] += pol["outcomes"].get(key, 0)
        n += pol["episodes"]
        eps += pol.get("episode_results", [])
    won = outcomes["won"]
    return {**cell, "won": won, "n": n, "outcomes": outcomes, "episodes": eps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--controls", required=True, choices=sorted(CONTROLS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=EPISODES, help="ONLY lower this for smoke tests; results are then not suite-v2")
    ap.add_argument("--only-opp", type=int, default=0,
                    help="diagnostic: run only the cells with this opponent count (result is NOT suite v2)")
    ap.add_argument("--repeat-check", action="store_true", help="run cell 0 (t_train_101 1v1) twice and report per-seed agreement")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = CONTROLS[a.controls]
    tag = Path(a.checkpoint).stem
    t0 = time.time()
    todo = cells()
    if a.only_opp:
        todo = [c for c in todo if c["opp"] == a.only_opp]
    if a.repeat_check:
        c = todo[2]  # t_train_101 1v3
        r1 = run_cell(a.checkpoint, c, flags, out_dir, a.episodes, tag + "_rep1")
        r2 = run_cell(a.checkpoint, c, flags, out_dir, a.episodes, tag + "_rep2")
        same = sum(1 for x, y in zip(r1["episodes"], r2["episodes"]) if x["outcome"] == y["outcome"])
        steps = sum(1 for x, y in zip(r1["episodes"], r2["episodes"]) if x["steps"] == y["steps"])
        print(f"repeat check {c['map']} {c['opp']}v: wins {r1['won']} vs {r2['won']}; same outcome on "
              f"{same}/{len(r1['episodes'])} seeds; identical episode length on {steps}/{len(r1['episodes'])}")
        (out_dir / f"{tag}_repeat_check.json").write_text(json.dumps(
            {"cell": c, "wins": [r1["won"], r2["won"]], "same_outcome": same, "same_length": steps,
             "n": len(r1["episodes"])}, indent=1))
        return 0
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(lambda c: run_cell(a.checkpoint, c, flags, out_dir, a.episodes, tag), todo))
    summary = {"suite": SUITE_VERSION + (f"-only{a.only_opp}v-DIAGNOSTIC" if a.only_opp else ""), "checkpoint": a.checkpoint, "controls": a.controls, "config_lines": flags,
               "episodes_per_cell": a.episodes, "seconds": round(time.time() - t0), "cells": []}
    print(f"\n{tag}  suite {SUITE_VERSION}  controls={a.controls}  ({a.episodes}/cell, d=1.0, greedy)")
    print(f"{'map':14} " + " ".join(f"{o}v{'':>9}" for o in OPPONENTS))
    for m in TRAIN_MAPS + HELD_MAPS:
        row = [r for r in res if r["map"] == m]
        print(f"{m:14} " + " ".join(f"{r['won']:>3}/{r['n']:<3}     " for r in sorted(row, key=lambda r: r["opp"])))
    if a.only_opp:
        OPPS = [a.only_opp]
    else:
        OPPS = OPPONENTS
    for split in ("train", "held"):
        for o in OPPS:
            rr = [r for r in res if r["split"] == split and r["opp"] == o]
            k, n = sum(r["won"] for r in rr), sum(r["n"] for r in rr)
            lo, hi = wilson(k, n)
            summary["cells"].append({"split": split, "opp": o, "won": k, "n": n, "ci95": [lo, hi]})
            print(f"  {split:5} {o}v: {k}/{n} = {k / n:.3f}  [{lo:.3f}, {hi:.3f}]")
    summary["per_map"] = [{k: r[k] for k in ("map", "opp", "split", "won", "n", "outcomes", "seed_base")} for r in res]
    (out_dir / f"{tag}_suite_{SUITE_VERSION}.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
