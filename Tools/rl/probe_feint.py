#!/usr/bin/env python3
"""OGRL-20261002-006: does the policy cancel its own attacks by holding grab?

`playercontrol.as::WantsToFeint` is simply "grab held" whenever game_difficulty > 0.5
(training runs at 1.0), and `aschar.as::UpdateAttacking` turns a ground attack into its
blocked animation while `can_feint` is set. The policy holds grab on most decisions.

This runs a checkpoint greedily on fixed seeds with attack telemetry on and counts, for
the agent only, RLATK (an attack resolved to a move) against RLFEINT (that attack then
cancelled). Pass `--config-line "rl_no_feint: 1"` to run the same seeds with feinting
disabled for the RL controller: the win-rate difference is the cost of the trap to the
CURRENT policy, with no retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ppo"))
sys.path.insert(0, str(HERE))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402
from watch import deterministic_action  # noqa: E402
from evaluate import run_episodes, wilson_ci  # noqa: E402

RLATK = re.compile(r"RLATK id=(\d+) kind=(\S+) path=(\S+)")
RLFEINT = re.compile(r"RLFEINT id=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--max-episode-steps", type=int, default=1200)
    ap.add_argument("--config-line", action="append", default=[])
    ap.add_argument("--shm-name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampled", action="store_true",
                    help="sample actions like training (incl. the grounded attack floor) instead of greedy")
    ap.add_argument("--ground-only", action="store_true",
                    help="OGRL-20261004-010: run every episode under the move-school ground-only rule")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"refusing to overwrite {a.out}")
        return 2
    shm = a.shm_name or f"/ogrl_pf{os.getpid() % 100000:05d}"

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    if a.sampled:
        from policy import set_button_floor
        set_button_floor(pol, "attack=0.1")
    nrm = ObservationNormalizer(L, frame_stack=fs)
    nrm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=str(HERE.parents[1]), level=a.level, shm_name=shm, seed=a.seed_base,
                        act_period=4, frame_stack=fs, log_attacks=True, extra_config_lines=a.config_line)
    log_path = env._write_dir.parent / (env._write_dir.name + ".log")
    if os.name == "nt":
        # OGRL-20261005-003: the Windows engine writes its log to <write_dir>/logfile.txt, not stdout,
        # so the stdout capture holds ~300 bytes and every RLATK line is missed.
        log_path = env._write_dir / "logfile.txt"
    if a.ground_only:
        _reset = env.reset
        env.reset = lambda *args, **kw: _reset(*args, ground_only=True, **kw)
    self_ids: set[int] = set()
    # OGRL-20261004-014: the agent's character id is NOT fixed across episodes (1v1 picks the player
    # spawn at random, so it alternates 0/1). Attributing by "any id the agent ever had" counted the
    # enemy's attacks as the agent's. Record (log byte offset, self id) whenever the id changes and
    # attribute each RLATK line by the segment it falls in.
    segments: list[tuple[int, int]] = []

    def act_fn(obs, frame):
        sid = int(round(float(frame[L.SELF_ID])))
        self_ids.add(sid)
        if not segments or segments[-1][1] != sid:
            try:
                env._log_file.flush()
                off = os.path.getsize(log_path)
            except OSError:
                off = 0
            segments.append((off, sid))
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if a.sampled:
            with torch.no_grad():
                return pol.get_action_and_value(x)[0].squeeze(0).numpy()
        return deterministic_action(pol, x)

    t0 = time.time()
    try:
        res = run_episodes(env, act_fn, a.episodes, a.seed_base, a.difficulty, a.opponents, 0.0, 0,
                           a.max_episode_steps, nrm, L)
        keep = Path(a.out).with_suffix(".engine.log")
        shutil.copy(log_path, keep)
    finally:
        env.close()

    attacks, feints, opp_attacks = Counter(), 0, 0
    opp_moves = Counter()
    import bisect
    seg_offsets = [o for o, _ in segments]

    def self_at(offset: int) -> int:
        i = bisect.bisect_right(seg_offsets, offset) - 1
        return segments[max(0, i)][1] if segments else -1

    offset = 0
    for raw in open(keep, "rb"):
        line = raw.decode("utf-8", errors="replace")
        here, offset = offset, offset + len(raw)
        m = RLATK.search(line)
        if m:
            mv = os.path.basename(m.group(3)).replace(".xml", "")
            if int(m.group(1)) == self_at(here):
                attacks[mv] += 1
            else:
                opp_attacks += 1
                opp_moves[mv] += 1
            continue
        m = RLFEINT.search(line)
        if m and int(m.group(1)) == self_at(here):
            feints += 1
    n_att = sum(attacks.values())
    won = res["outcomes"]["won"]
    payload = {
        "experiment": "OGRL-20261002-006",
        "checkpoint": a.checkpoint, "global_step": int(ck.get("global_step", -1)),
        "level": a.level, "opponents": a.opponents, "difficulty": a.difficulty,
        "episodes": a.episodes, "seed_base": a.seed_base, "config_lines": a.config_line,
        "outcomes": res["outcomes"], "win_rate": res["win_rate"], "win_rate_ci95": res["win_rate_ci95"],
        "agent_attacks": n_att, "agent_feints": feints,
        "feint_share_of_attacks": (feints / n_att) if n_att else None,
        "agent_moves": dict(attacks), "opponent_attacks": opp_attacks, "opponent_moves": dict(opp_moves),
        "attribution": "per-segment self id (OGRL-20261004-014)",
        "ground_only": bool(a.ground_only), "sampled": bool(a.sampled), "blocked_air_attacks": int(getattr(env, "blocked_air_attacks", 0)),
        "self_ids": sorted(self_ids), "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in ("outcomes", "win_rate", "agent_attacks", "agent_feints",
                                              "feint_share_of_attacks")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
