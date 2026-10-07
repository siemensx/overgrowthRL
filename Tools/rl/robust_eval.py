#!/usr/bin/env python3
"""Robustness suite r1 (OGRL-20261007-004): how does a checkpoint do against the fights the
canonical suite does NOT contain? Complements canonical_eval.py (suite v2); never replaces it.

Why: run38 is the best suite-v2 agent, yet against a standing human it never attacks
(OGRL-20261007-001). Suite v2 only contains the stock opponent, which always walks in and attacks,
and the policy can read that opponent's decisions from privileged intent fields. This suite asks:

  personas   the same fight against patient / passive / berserker / expert / mixed opponents
             (gen_1v1_scenario.py step 3b), with the stock opponent on the same seeds as reference
  blind      stock opponent, but the AI-intent fields zeroed (what the agent would know of a human)
  armed      the armed ladder's rungs B1, B5 and C (curriculum.ARMED_STAGES)
  horde      1v4, 1v5, 1v7 unarmed and 1v5 with three armed, on a held-out horde map

Fixed definition (do not edit; make an r2 instead):
  * maps     t_held_203 (personas, blind, armed) and t_horde_held_401 (horde): neither is trained on
  * n        40 episodes per cell by default, greedy, difficulty 1.0, 1200-decision cap
  * seeds    8,000,000 + 1000*cell_index + episode -- disjoint from training and suite v2 (7,000,000+)
  * chunks   a fresh engine process every 20 episodes, exactly as suite v2
A timeout is scored as a loss and reported separately. With 40 episodes a cell's 95% interval is
about +-15 points: use it to find failure modes, and compare checkpoints by paired seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from canonical_eval import CONTROLS, CHUNK, wilson, _run_chunk  # noqa: E402

SUITE_VERSION = "r1-2026-10-07"
SEED0 = 8_000_000
EPISODES = 40
HELD = "t_held_203"
HORDE = "t_horde_held_401"


def cells() -> list[dict]:
    out = []

    def add(group, name, m, opp, extra):
        out.append({"group": group, "name": name, "map": m, "opp": opp, "extra": extra})

    for persona in ("stock", "patient", "passive", "berserker", "expert", "mixed"):
        for opp in (1, 3):
            add("personas", f"{persona} 1v{opp}", HELD, opp, ["--persona", persona])
    for opp in (1, 3):
        add("blind", f"intent hidden 1v{opp}", HELD, opp, ["--hide-intent"])
    add("armed", "B1 knife, 1 of 2 armed", HELD, 2,
        ["--armed-count", "1", "--weapon-type", "1", "--throw-aggression", "4.0"])
    add("armed", "B5 all 3 armed", HELD, 3, ["--armed-count", "3", "--throw-aggression", "4.0"])
    add("armed", "C cats + mixed, 3 armed", HELD, 3,
        ["--armed-count", "3", "--throw-aggression", "6.0", "--species", "6"])
    for opp in (4, 5, 7):
        add("horde", f"horde 1v{opp}", HORDE, opp, [])
    add("horde", "horde 1v5, 3 armed", HORDE, 5, ["--armed-count", "3", "--throw-aggression", "4.0"])
    for i, c in enumerate(out):
        c["idx"] = i
        c["seed_base"] = SEED0 + 1000 * i
    return out


def run_cell(ckpt: str, cell: dict, flags: list[str], out_dir: Path, episodes: int, tag: str) -> dict:
    outcomes, eps = {"won": 0, "lost": 0, "timeout": 0}, []
    for k in range(0, episodes, CHUNK):
        m = min(CHUNK, episodes - k)
        out = out_dir / f"{tag}_r{cell['idx']:02d}_{cell['map']}_{cell['opp']}v_s{k:03d}.json"
        _run_chunk(ckpt, cell, flags, out, cell["seed_base"] + k, m, extra=cell["extra"])
        pol = json.loads(out.read_text())["bands"][0]["policy"]
        for key in outcomes:
            outcomes[key] += pol["outcomes"].get(key, 0)
        eps += pol.get("episode_results", [])
    n = sum(outcomes.values())
    return {**cell, "won": outcomes["won"], "n": n, "outcomes": outcomes, "episodes": eps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--controls", required=True, choices=sorted(CONTROLS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--groups", default="personas,blind,armed,horde",
                    help="comma list of cell groups to run (a subset is labelled DIAGNOSTIC)")
    ap.add_argument("--list", action="store_true", help="print the cells and exit")
    a = ap.parse_args()
    groups = a.groups.split(",")
    todo = [c for c in cells() if c["group"] in groups]
    if a.list:
        for c in todo:
            print(f"{c['idx']:2d} {c['group']:9} {c['name']:28} {c['map']:18} 1v{c['opp']}  {' '.join(c['extra'])}")
        return 0
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = CONTROLS[a.controls]
    tag = Path(a.checkpoint).stem
    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(lambda c: run_cell(a.checkpoint, c, flags, out_dir, a.episodes, tag), todo))
    full = set(groups) >= {"personas", "blind", "armed", "horde"} and a.episodes == EPISODES
    summary = {"suite": SUITE_VERSION + ("" if full else "-DIAGNOSTIC"), "checkpoint": a.checkpoint,
               "controls": a.controls, "config_lines": flags, "episodes_per_cell": a.episodes,
               "seconds": round(time.time() - t0), "cells": []}
    print(f"\n{tag}  robustness suite {SUITE_VERSION}  controls={a.controls}  ({a.episodes}/cell, d=1.0, greedy)")
    print(f"  {'cell':30} {'won':>7}  {'lost':>4} {'timeout':>7}   95% interval")
    group = None
    for r in res:
        if r["group"] != group:
            group = r["group"]
            print(f" {group}")
        lo, hi = wilson(r["won"], r["n"])
        o = r["outcomes"]
        print(f"  {r['name']:30} {r['won']:>3}/{r['n']:<3}  {o['lost']:>4} {o['timeout']:>7}   [{lo:.2f}, {hi:.2f}]")
        summary["cells"].append({k: r[k] for k in ("idx", "group", "name", "map", "opp", "extra", "won", "n",
                                                     "outcomes", "seed_base", "episodes")} | {"ci95": [lo, hi]})
    (out_dir / f"{tag}_robust_{SUITE_VERSION}.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
