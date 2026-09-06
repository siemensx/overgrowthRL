#!/usr/bin/env python3
"""Merge independently-trained checkpoints into one, by weight averaging.

Two runs on two machines produce two policies. This answers "can their
knowledge be united into a single checkpoint" WITHOUT any distributed training
machinery, and it is testable in minutes rather than days.

Why it can work here: both runs descend from the same parent (run15) and were
fine-tuned from it, so they have not drifted into different loss basins --
which is the condition under which averaging fine-tuned weights ("model soup",
Wortsman et al. 2022) tends to match or beat the best individual member.
Averaging two INDEPENDENTLY INITIALISED networks does not work, because
hidden units are permutations of each other; that is not the case here.

Why it might not: the runs saw different seeds and are at different global
steps (70.2M vs 102.4M), so they are unequal in training, and RL policies can
be far less forgiving than supervised fine-tunes.

Either way the answer is measured, not assumed: merge, then evaluate the merge
against both parents on the same maps with the same seeds.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="checkpoints to merge")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="blend weights, defaulting to equal. Passing global_step-"
                         "proportional weights is a defensible alternative when the "
                         "runs are at very different amounts of training.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.inputs]
    steps = [c.get("global_step", 0) for c in cks]
    for key in ("layout_total_floats", "frame_stack", "layout_max_visible_entities"):
        vals = {c.get(key) for c in cks}
        if len(vals) != 1:
            print(f"refusing to merge: {key} differs across inputs: {vals}", file=sys.stderr)
            return 1

    w = a.weights or [1.0 / len(cks)] * len(cks)
    if len(w) != len(cks):
        print("weights must match inputs", file=sys.stderr); return 1
    tot = sum(w); w = [x / tot for x in w]

    merged = dict(cks[0])
    out_policy = {}
    for k in cks[0]["policy"]:
        ref = cks[0]["policy"][k]
        if not torch.is_floating_point(ref):
            out_policy[k] = ref.clone()          # counters/ints: take the first, never average
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for c, wi in zip(cks, w):
            acc += c["policy"][k].to(torch.float64) * wi
        out_policy[k] = acc.to(ref.dtype)
    merged["policy"] = out_policy

    # Observation normaliser statistics are averages over experience, so blending
    # them is meaningful in a way the optimiser moments are not -- Adam state is
    # tied to a specific trajectory and is dropped rather than blended.
    for norm_key in ("obs_normalizer", "reward_normalizer"):
        sds = [c.get(norm_key) for c in cks]
        if all(isinstance(sd, dict) for sd in sds):
            out = {}
            for k in sds[0]:
                v0 = sds[0][k]
                if torch.is_tensor(v0) and torch.is_floating_point(v0):
                    acc = torch.zeros_like(v0, dtype=torch.float64)
                    for sd, wi in zip(sds, w):
                        acc += sd[k].to(torch.float64) * wi
                    out[k] = acc.to(v0.dtype)
                else:
                    out[k] = v0
            merged[norm_key] = out
    merged.pop("optimizer", None)
    merged["global_step"] = max(steps)
    merged["merged_from"] = [{"path": str(p), "global_step": s, "weight": wi}
                             for p, s, wi in zip(a.inputs, steps, w)]

    torch.save(merged, a.out)
    print(f"merged {len(cks)} checkpoints -> {a.out}")
    for p, s, wi in zip(a.inputs, steps, w):
        print(f"  {Path(p).name:<22} step {s:>12,}  weight {wi:.3f}")
    print("  optimizer state dropped (Adam moments are trajectory-specific);")
    print("  resuming from this merge restarts the optimiser, which is expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
