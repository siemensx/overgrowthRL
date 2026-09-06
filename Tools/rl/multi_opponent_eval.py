#!/usr/bin/env python3
"""Evaluate a checkpoint across opponent counts AND maps.

Answers the two questions a multi-opponent run has to answer together:

  1. can it beat N opponents, and
  2. has it KEPT its 1v1 competence while learning to?

Reporting only the first is how an "improvement" quietly turns out to be a
distribution shift. Both are printed side by side, with the 1v1 column first.

Uses evaluate.py, which requires a win to be EVERY hostile down -- see its
comment. Any tool that counts a single knockout as a win will report being
outnumbered as easier than fighting alone.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def one(py, ckpt, repo, level, opponents, episodes, bands, seed, shm, out):
    cmd = [py, str(HERE / "evaluate.py"), "--checkpoint", ckpt, "--repo-root", repo,
           "--level", level, "--frame-stack", "4", "--act-period", "4",
           "--episodes", str(episodes), "--seed-base", str(seed),
           "--difficulty-bands", bands, "--opponents", str(opponents),
           "--shm-name", shm, "--device", "cpu", "--no-control", "--out", str(out)]
    subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=7200)
    if not Path(out).exists():
        return None
    d = json.loads(Path(out).read_text())
    rates = [b["policy"]["win_rate"] for b in d.get("bands", []) if "policy" in b]
    return statistics.mean(rates) if rates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=str(HERE.parent.parent))
    ap.add_argument("--levels", nargs="+", default=["arenas/t_train_103.xml", "arenas/t_held_202.xml"])
    ap.add_argument("--opponents", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--difficulty-bands", default="0.4,0.8")
    ap.add_argument("--seed-base", type=int, default=8900000)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tmp = Path(a.repo_root) / ".rl_mo_eval"
    tmp.mkdir(parents=True, exist_ok=True)
    grid, i = {}, 0
    print(f"checkpoint: {a.checkpoint}")
    header = f"{'level':<26}" + "".join(f"{('%dv' % n):>9}" for n in a.opponents)
    print(header)
    for lv in a.levels:
        row = []
        for n in a.opponents:
            i += 1
            r = one(a.python, a.checkpoint, a.repo_root, lv, n, a.episodes,
                    a.difficulty_bands, a.seed_base, f"/ogrl_mo{i}",
                    tmp / f"{Path(lv).stem}_{n}.json")
            grid[(lv, n)] = r
            row.append("  --  " if r is None else f"{r:.3f}")
        print(f"  {Path(lv).stem:<24}" + "".join(f"{v:>9}" for v in row))

    print()
    for n in a.opponents:
        vals = [v for (lv, k), v in grid.items() if k == n and v is not None]
        if vals:
            print(f"  mean across maps, {n} opponent(s): {statistics.mean(vals):.3f}")
    solo = [v for (lv, k), v in grid.items() if k == 1 and v is not None]
    if solo:
        print(f"\n  1v1 RETENTION is the first thing to check: {statistics.mean(solo):.3f}")
        print("  A multi-opponent gain that costs 1v1 competence is a distribution shift, not progress.")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"checkpoint": a.checkpoint, "episodes": a.episodes,
             "grid": {f"{lv}|{n}": v for (lv, n), v in grid.items()}}, indent=2))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
