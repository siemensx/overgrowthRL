#!/usr/bin/env python3
"""Measure map generalisation: trained maps vs held-out maps, same checkpoint.

The project trained 46.9M decisions on a single arena, so "does this policy
generalise across maps" has never been answerable -- there was no held-out set.
gen_arena_map.py now emits a corpus, --levels trains on part of it, and this
compares the two halves under identical evaluation settings.

The number that matters is the GAP: held-out win rate minus trained win rate.
A policy that has learned to fight scores about the same on both. A policy that
has memorised six layouts scores markedly worse on the three it has never seen.
Per-map rates are printed too, because one pathological map can carry the mean.
"""
from __future__ import annotations
import argparse, json, subprocess, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = [f"arenas/t_train_{s}.xml" for s in (101, 102, 103, 104, 105, 106)]
HELD  = [f"arenas/t_held_{s}.xml"  for s in (201, 202, 203)]


def eval_one(py: str, ckpt: str, repo: str, level: str, episodes: int, bands: str,
             seed_base: int, shm: str) -> dict | None:
    out = Path(repo) / ".rl_transfer" / (level.replace("/", "_") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(HERE / "evaluate.py"), "--checkpoint", ckpt, "--repo-root", repo,
           "--level", level, "--frame-stack", "4", "--act-period", "4",
           "--episodes", str(episodes), "--seed-base", str(seed_base),
           "--difficulty-bands", bands, "--shm-name", shm, "--device", "cpu",
           "--no-control", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=7200)
    if not out.exists():
        print(f"  {level}: FAILED\n    {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout[-200:]}")
        return None
    return json.loads(out.read_text())


def win_rate(res: dict) -> float:
    if "overall" in res and isinstance(res["overall"], dict) and "win_rate" in res["overall"]:
        return float(res["overall"]["win_rate"])
    rates = [b["policy"]["win_rate"] for b in res.get("bands", []) if "policy" in b]
    return statistics.mean(rates) if rates else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=str(HERE.parent.parent))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--difficulty-bands", default="0.3,0.7,1.0")
    ap.add_argument("--seed-base", type=int, default=9100000)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--shm-name", default="/ogrl_xfer")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    t0 = time.time()
    results: dict[str, float] = {}
    for group, levels in (("trained", TRAIN), ("held-out", HELD)):
        print(f"\n== {group} ==")
        for lv in levels:
            res = eval_one(a.python, a.checkpoint, a.repo_root, lv, a.episodes,
                           a.difficulty_bands, a.seed_base, a.shm_name)
            if res is None:
                continue
            wr = win_rate(res)
            results[lv] = wr
            print(f"  {lv.split('/')[-1]:<22} win_rate {wr:.3f}")

    tr = [v for k, v in results.items() if k in TRAIN]
    he = [v for k, v in results.items() if k in HELD]
    print("\n== summary ==")
    if tr and he:
        mt, mh = statistics.mean(tr), statistics.mean(he)
        print(f"  trained  maps (n={len(tr)}): {mt:.3f}")
        print(f"  held-out maps (n={len(he)}): {mh:.3f}")
        print(f"  GAP (held-out - trained)   : {mh - mt:+.3f}")
        print("  A gap near zero means it learned to fight; a large negative gap")
        print("  means it memorised the training layouts.")
    else:
        print("  insufficient results to compare")
    print(f"  elapsed {time.time()-t0:.0f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"checkpoint": a.checkpoint, "episodes": a.episodes,
             "bands": a.difficulty_bands, "per_map": results,
             "trained_mean": statistics.mean(tr) if tr else None,
             "heldout_mean": statistics.mean(he) if he else None,
             "gap": (statistics.mean(he) - statistics.mean(tr)) if (tr and he) else None},
            indent=2))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
