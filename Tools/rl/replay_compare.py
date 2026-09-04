#!/usr/bin/env python3
"""Stage 1.4: production-equivalence comparator.

Compares two --equivalence-digest JSONL recordings (reference vs candidate,
same seed/level/scenario) and reports the first divergence step per quantity,
max absolute/relative deviation, and outcome (final knocked_out state) match.

Two tolerance regimes (Stage 1.4):
  --strict   : same architecture, semantically-neutral change expected.
               ANY divergence (including the hash chain) is a failure.
  --numeric  : cross-architecture (x86/SSE vs arm64/NEON) or other change
               where some numeric drift is expected. Reports whether
               divergence stays bounded or grows chaotically over the episode,
               using --pos-tol / --vel-tol / --scalar-tol as the per-step
               absolute tolerances quantities must stay within.

Scope note (research-log OGRL-20260815-035): the digest currently covers
position/velocity/angular-velocity/facing, animation name+phase,
temp/permanent/blood/block health, knocked_out, state, and primary weapon
item id, per character per step -- not yet full per-bone physics transforms,
the ordered Bullet contact set, or per-attack-event records. This still
exercises the comparator's actual job (detect an injected semantic change)
end to end; the deferred quantities are a coverage gap, not a mechanism gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_digest(path: Path) -> list[dict[str, Any]]:
    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


NUMERIC_VECTOR_FIELDS = ["pos", "vel", "ang_vel", "facing"]
NUMERIC_SCALAR_FIELDS = ["temp_health", "permanent_health", "blood_health", "block_health", "anim_phase"]
EXACT_FIELDS = ["knocked_out", "state", "primary_weapon_item_id", "controlled", "anim"]


def index_characters(step: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {c["id"]: c for c in step["characters"]}


def compare(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    strict: bool,
    pos_tol: float,
    vel_tol: float,
    scalar_tol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_steps": len(reference),
        "candidate_steps": len(candidate),
        "step_count_match": len(reference) == len(candidate),
        "hash_chain_match": None,
        "first_divergence_step": None,
        "first_divergence_field": None,
        "first_divergence_detail": None,
        "max_deviation_by_field": {},
        "divergence_bounded": None,
        "outcome_match": None,
        "strict": strict,
    }

    if reference and candidate:
        result["hash_chain_match"] = reference[-1].get("chain") == candidate[-1].get("chain")

    n = min(len(reference), len(candidate))
    max_dev: dict[str, float] = {}
    first_step_seen: dict[str, int] = {}
    deviation_series: dict[str, list[tuple[int, float]]] = {}

    for i in range(n):
        ref_chars = index_characters(reference[i])
        cand_chars = index_characters(candidate[i])
        ids = sorted(set(ref_chars) | set(cand_chars))
        for cid in ids:
            if cid not in ref_chars or cid not in cand_chars:
                if result["first_divergence_step"] is None:
                    result["first_divergence_step"] = i
                    result["first_divergence_field"] = "character_set"
                    result["first_divergence_detail"] = f"character {cid} present in only one run"
                continue
            rc, cc = ref_chars[cid], cand_chars[cid]

            for field in EXACT_FIELDS:
                if rc.get(field) != cc.get(field):
                    key = f"{field}[{cid}]"
                    if strict and result["first_divergence_step"] is None:
                        result["first_divergence_step"] = i
                        result["first_divergence_field"] = key
                        result["first_divergence_detail"] = f"{rc.get(field)!r} != {cc.get(field)!r}"
                    max_dev[key] = max(max_dev.get(key, 0.0), 1.0)  # exact-field mismatch: nominal deviation 1.0
                    deviation_series.setdefault(key, []).append((i, 1.0))

            for field in NUMERIC_VECTOR_FIELDS:
                rv, cv = rc.get(field), cc.get(field)
                if rv is None or cv is None:
                    continue
                dev = max(abs(a - b) for a, b in zip(rv, cv))
                key = f"{field}[{cid}]"
                tol = pos_tol if field in ("pos", "facing") else vel_tol
                if dev > 0.0:
                    first_step_seen.setdefault(key, i)
                if dev > tol and result["first_divergence_step"] is None:
                    result["first_divergence_step"] = i
                    result["first_divergence_field"] = key
                    result["first_divergence_detail"] = f"max component deviation {dev:.8f} > tol {tol}"
                max_dev[key] = max(max_dev.get(key, 0.0), dev)
                deviation_series.setdefault(key, []).append((i, dev))

            for field in NUMERIC_SCALAR_FIELDS:
                rv, cv = rc.get(field), cc.get(field)
                if rv is None or cv is None:
                    continue
                dev = abs(rv - cv)
                key = f"{field}[{cid}]"
                if dev > scalar_tol and result["first_divergence_step"] is None:
                    result["first_divergence_step"] = i
                    result["first_divergence_field"] = key
                    result["first_divergence_detail"] = f"deviation {dev:.8f} > tol {scalar_tol}"
                max_dev[key] = max(max_dev.get(key, 0.0), dev)
                deviation_series.setdefault(key, []).append((i, dev))

    result["max_deviation_by_field"] = max_dev

    # Outcome: final knocked_out state per character.
    if reference and candidate:
        ref_final = {c["id"]: c["knocked_out"] for c in reference[-1]["characters"]}
        cand_final = {c["id"]: c["knocked_out"] for c in candidate[-1]["characters"]}
        result["outcome_match"] = ref_final == cand_final
        result["reference_final_outcome"] = ref_final
        result["candidate_final_outcome"] = cand_final

    # Divergence growth: for the field with the largest max deviation, is it
    # growing over the back half of the episode (chaotic) or flat (bounded)?
    if deviation_series and not strict:
        worst_field = max(max_dev, key=max_dev.get)
        series = deviation_series[worst_field]
        half = len(series) // 2
        first_half_max = max((d for _, d in series[:half]), default=0.0)
        second_half_max = max((d for _, d in series[half:]), default=0.0)
        result["divergence_bounded"] = bool(second_half_max <= first_half_max * 3.0 + 1e-9)
        result["worst_field"] = worst_field
        result["worst_field_first_half_max"] = first_half_max
        result["worst_field_second_half_max"] = second_half_max

    if strict:
        result["passed"] = result["step_count_match"] and result["hash_chain_match"] is True and result["first_divergence_step"] is None
    else:
        result["passed"] = result["step_count_match"] and bool(result.get("divergence_bounded", True)) and bool(result["outcome_match"])

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_digest", type=Path)
    parser.add_argument("candidate_digest", type=Path)
    parser.add_argument("--strict", action="store_true", help="Bitwise/exact regime (same arch, semantically-neutral change expected).")
    parser.add_argument("--pos-tol", type=float, default=1e-4)
    parser.add_argument("--vel-tol", type=float, default=1e-4)
    parser.add_argument("--scalar-tol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reference = load_digest(args.reference_digest)
    candidate = load_digest(args.candidate_digest)

    result = compare(
        reference, candidate,
        strict=args.strict,
        pos_tol=args.pos_tol, vel_tol=args.vel_tol, scalar_tol=args.scalar_tol,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
