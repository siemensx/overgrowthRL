#!/usr/bin/env python3
"""Empirical map gate: does a fight actually HAPPEN on this level?

The static checks in gen_arena_map.py catch geometry defects -- spawns inside
props, interpenetration, unreachable coverage. They cannot catch "the AI never
finds the agent", which is the failure that actually cost run21 its compute:
t_train_105 passed every eyeball test and timed out on 78-91% of episodes
because the opponent spawned on a deck. This runs real episodes and gates on
the timeout rate, which is the one number that would have caught all of it.

Reference, measured over 60k run21 episodes:
    healthy maps (101/102/104)   0.2 - 2.5% timeouts across 1/2/3 opponents
    tiered maps  (103/105/106)   6 - 91%

usage: validate_maps.py --checkpoint C [--levels a.xml b.xml] [--episodes 12]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = 0.03


def cell(py, ckpt, repo, level, opponents, episodes, seed, shm, max_steps):
    out = Path(tempfile.gettempdir()) / f"mapval_{Path(level).stem}_{opponents}.json"
    out.unlink(missing_ok=True)
    cmd = [py, str(HERE / "evaluate.py"), "--checkpoint", ckpt, "--repo-root", repo,
           "--level", level, "--frame-stack", "4", "--act-period", "4",
           "--episodes", str(episodes), "--seed-base", str(seed),
           "--difficulty-bands", "0.6", "--opponents", str(opponents),
           "--max-episode-steps", str(max_steps), "--shm-name", shm,
           "--device", "cpu", "--no-control", "--out", str(out)]
    subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600)
    if not out.exists():
        return None
    d = json.loads(out.read_text())
    b = d["bands"][0]["policy"]
    return b["outcomes"], b["win_rate"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=str(HERE.parent.parent))
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--opponents", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-episode-steps", type=int, default=1200)
    ap.add_argument("--seed-base", type=int, default=7700000)
    ap.add_argument("--gate", type=float, default=GATE)
    a = ap.parse_args()

    print(f"{'level':<22}{'opp':>4}{'n':>5}{'won':>6}{'lost':>6}{'timeout':>9}{'win%':>7}")
    failed, i = [], 0
    for lv in a.levels:
        worst = 0.0
        for n in a.opponents:
            i += 1
            r = cell(sys.executable, a.checkpoint, a.repo_root, lv, n, a.episodes,
                     a.seed_base, f"/ogrl_v{os.getpid() % 1000:03d}{i:02d}", a.max_episode_steps)
            if r is None:
                print(f"{Path(lv).stem:<22}{n:>4}   -- evaluation produced nothing --")
                failed.append((lv, n, "no result")); continue
            oc, wr = r
            tot = sum(oc.values()); tmo = oc["timeout"] / tot if tot else 1.0
            worst = max(worst, tmo)
            print(f"{Path(lv).stem:<22}{n:>4}{tot:>5}{oc['won']:>6}{oc['lost']:>6}"
                  f"{tmo:>8.1%}{100*wr:>7.1f}")
        if worst > a.gate:
            failed.append((lv, "any", f"timeout {worst:.1%} > gate {a.gate:.0%}"))

    print()
    if failed:
        for lv, n, why in failed:
            print(f"  FAIL {Path(str(lv)).stem} (opp={n}): {why}")
        print("\nThese maps must not enter the training rotation.")
        return 1
    print(f"  all {len(a.levels)} maps pass (timeout <= {a.gate:.0%} at every opponent count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
