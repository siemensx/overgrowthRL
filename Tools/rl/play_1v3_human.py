#!/usr/bin/env python3
"""Play the agent's benchmark fight yourself: keyboard/mouse in the player
slot, the same scripted opponents, same map, same difficulty.

This is the exact cell the gate certifies and the bench measures -- 3 unarmed
opponents at difficulty 1.0 on t_train_101 -- with no RL transport attached,
so the engine leaves the player character under native input. The level
script honors two launch-config keys added for this (gen_1v1_scenario.py step
7); without them a keyboard session gets the stock near-player-skill
difficulty and one opponent.

    python3 Tools/rl/play_1v3_human.py                 # 1v3, difficulty 1.0
    python3 Tools/rl/play_1v3_human.py --opponents 1   # the 1v1 the scripted oracle won 90% of
    python3 Tools/rl/play_1v3_human.py --difficulty 0.5

Controls are stock Overgrowth. Esc quits. The round restarts on its own after
a knockout. Rendering is on, so this is real-time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noaslr
import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--binary-path", default=None)
    ap.add_argument("--write-dir", default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-quit after N seconds (0 = play until Esc)")
    args = ap.parse_args()

    binary = paths.engine_binary(Path(args.repo_root), args.binary_path)
    write_dir = Path(args.write_dir) if args.write_dir else Path(args.repo_root) / "Tools" / "rl" / "runs" / "human_play" / f"wd-{int(time.time())}"
    write_dir.mkdir(parents=True, exist_ok=True)
    config = "\n".join([
        f"rl_human_difficulty: {min(max(args.difficulty, 0.0), 1.0):g}",
        f"rl_human_opponents: {args.opponents}",
        "blood: 0",                  # same NaN-decal mitigation env.py uses for rendered sessions
        "skip_loading_pause: true",
        "has_detected_settings: true",
    ])
    cmd = noaslr.wrap_command([
        str(binary), "--write-dir", str(write_dir), "--working-dir", args.repo_root,
        "--no-dialogues", "--level", args.level, "--config", config,
    ])
    log = write_dir.parent / f"{write_dir.name}.log"
    print(f"level={args.level}  opponents={args.opponents}  difficulty={args.difficulty:g}\nlog: {log}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=args.repo_root, stdout=lf, stderr=subprocess.STDOUT)
        try:
            if args.seconds > 0:
                try:
                    proc.wait(timeout=args.seconds)
                except subprocess.TimeoutExpired:
                    proc.terminate()
            else:
                proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
