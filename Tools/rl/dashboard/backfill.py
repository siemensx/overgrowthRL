#!/usr/bin/env python3
"""OGRL-20260816-022 Sec 7: backfill runs 1-7's existing CSVs into the
dashboard's Tools/rl/runs/<run_id>/ schema, so the dashboard is useful on
day one instead of showing only whatever launches after it exists.

Deliberately lossy in a documented way: fields the old CSV never recorded
(reward.components, action stats, entropy split, perf.reset_seconds) are
left ABSENT from each backfilled metrics.jsonl line, not zeroed -- the
front end already renders "no data yet" for an absent series rather than a
misleading flat zero line.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = REPO_ROOT / "Tools/rl/ppo/runs"
RUNS_ROOT = REPO_ROOT / "Tools/rl/runs"

# entropy_random_reference is constant for this action space across every
# run backfilled here (2 continuous + 6 discrete throughout runs 1-7) --
# see train_vec.py's _entropy_random_reference() for the derivation.
ENTROPY_RANDOM_REFERENCE = 6.996760149769017

# Hand-written from research-log/2026-08-16.md -- the real story of runs 1-7,
# not reconstructable from the CSVs alone.
EVENTS_BY_RUN = {
    "run2_vec": [
        {"kind": "anomaly", "title": "ambient-combat reward causation bug (OGRL-20260816-014)",
         "body": "reward credited the agent for ANY visible entity's knockout/damage, no causation check -- "
                 "oval_arena.xml spawns ~10 characters across multiple teams fighting each other independently."},
        {"kind": "checkpoint_retired", "title": "run2 checkpoint retired", "body": "schema-incompatible after the causation fix (v2 -> v3)"},
    ],
    "run3_vec": [{"kind": "note", "title": "run3 paused by user request", "body": "reviewed live via watch.py before deciding whether to continue"}],
    "run4_vec": [
        {"kind": "anomaly", "title": "seed-diversity bug (OGRL-20260816-016)", "body": "vec_env.py's auto-reset silently reused each worker's original launch seed forever -- 8 distinct scenarios for the whole run, not a diverse curriculum."},
        {"kind": "checkpoint_retired", "title": "run4 checkpoint retired", "body": "superseded by run5 with the seed-diversity fix"},
    ],
    "run5_vec": [{"kind": "eval", "title": "run5 diagnostic: 7 WON / 4 LOST / 19 TIMEOUT (30 episodes)", "body": "verified real combat competence, zero friendly fire -- the trusted baseline"}],
    "run6_vec": [
        {"kind": "decision", "title": "stall tax introduced (OGRL-20260816-018)", "body": "resumed from run5, stall_penalty_weight applied at FULL strength immediately"},
        {"kind": "anomaly", "title": "run6 regressed (OGRL-20260816-020)", "body": "2 WON / 25 TIMEOUT (30 ep) -- worse than run5's baseline; abrupt reward-term introduction disrupted the resumed policy"},
    ],
    "run7_vec": [
        {"kind": "decision", "title": "stall-tax ramp-in fix (OGRL-20260816-019)", "body": "resumed from run5 (not run6) with stall_penalty_weight ramping linearly from 0"},
        {"kind": "eval", "title": "run7 diagnostic: 4 WON / 6 LOST / 20 TIMEOUT (30 episodes)", "body": "partial recovery from run6's regression, but not back to run5's own baseline -- self-knockouts increased"},
        {"kind": "anomaly", "title": "runs 5-7 never actually learned (OGRL-20260816-021)", "body": "entropy finished AT OR ABOVE the untrained-policy reference in all three runs -- 120Hz decisions on an unlearnable 10-character random-weapon brawl destroyed credit assignment."},
    ],
}


def backfill_run(csv_path: Path) -> None:
    run_id = csv_path.stem
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"{run_id}: empty CSV, skipping")
        return

    with open(run_dir / "metrics.jsonl", "w") as f:
        for row in rows:
            record = {
                "t": None, "global_step": int(row["global_step"]), "update": int(row["update"]),
                "phase": row.get("curriculum_phase"),
                "reward": {
                    "episode_total": float(row["mean_episode_reward"]) if row["mean_episode_reward"] not in ("", "nan") else None,
                    "episode_length": float(row["mean_episode_length"]) if row["mean_episode_length"] not in ("", "nan") else None,
                    "episodes_completed": int(row["episodes_completed"]),
                    # components: intentionally absent -- not recorded by the old CSV format
                },
                "ppo": {
                    "policy_loss": float(row["policy_loss"]), "value_loss": float(row["value_loss"]),
                    "entropy": float(row["entropy"]), "approx_kl": float(row["approx_kl"]),
                    "clip_fraction": float(row["clip_fraction"]), "explained_variance": float(row["explained_variance"]),
                    "entropy_random_reference": ENTROPY_RANDOM_REFERENCE,
                },
                "perf": {"steps_per_second_collection": float(row["steps_per_second"])},
                # curriculum_live, outcomes, action stats: absent -- not recorded
            }
            f.write(json.dumps(record) + "\n")

    with open(run_dir / "episodes.jsonl", "w"):
        pass  # not recoverable from the old CSV -- an empty file, not a fabricated one

    events = EVENTS_BY_RUN.get(run_id, [])
    with open(run_dir / "events.jsonl", "w") as f:
        for i, ev in enumerate(events):
            f.write(json.dumps({"t": None, "kind": ev["kind"], "title": ev["title"], "body": ev["body"]}) + "\n")

    last_row = rows[-1]
    manifest = {
        "run_id": run_id, "schema": 1, "backfilled": True,
        "started_at": None, "ended_at": None, "status": "completed",
        "final_global_step": int(last_row["global_step"]),
        "purpose": f"backfilled from {csv_path.name}",
        "entropy_random_reference": ENTROPY_RANDOM_REFERENCE,
        "algo": {"total_timesteps": int(last_row["global_step"])},  # true target unknown from CSV alone; best available
        "env": {}, "code": {}, "parent": {"run_id": None, "checkpoint": None, "global_step": None},
        "reward_profile": "default",
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "control.json").write_text(json.dumps({"command": None, "t": None}) + "\n")
    print(f"{run_id}: backfilled {len(rows)} updates, {len(events)} events")


def main() -> int:
    if not CSV_DIR.exists():
        print(f"no CSV dir at {CSV_DIR}", file=sys.stderr)
        return 1
    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        try:
            backfill_run(csv_path)
        except Exception as exc:  # noqa: BLE001 -- one bad CSV must not stop the rest
            print(f"{csv_path.name}: FAILED: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
