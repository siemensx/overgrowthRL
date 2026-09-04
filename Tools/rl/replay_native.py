#!/usr/bin/env python3
"""Render one native-captured episode and verify it inside the engine.

The replay is not a browser animation and it is not a best-effort ghost.  The
recorded applied action is sent back to the same engine build at every physics
tick while the engine compares its post-tick digest chain to the archived
episode.  The window remains visible through the final-state hold interval;
the report is written even when the first divergence is detected.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import noaslr
from ogreplay import ReplayReader, runtime_fingerprint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--binary-path", default=None)
    p.add_argument("--level", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--difficulty", type=float, default=None)
    p.add_argument("--opponents", type=int, default=None)
    p.add_argument("--weapons", type=float, default=None)
    p.add_argument("--species", type=int, default=None)
    p.add_argument("--reset-mode", type=int, default=None, help="Recorded engine reset mode: 0 hard, 1 soft")
    p.add_argument("--controlled-character-id", type=int, default=None, help="Recorded character object ID receiving controller 0")
    p.add_argument("--report", required=True)
    p.add_argument("--hold-seconds", type=float, default=12.0)
    p.add_argument("--time-scale", type=float, default=1.0, help="Rendered engine time scale; 100 matches the training fast-forward profile")
    p.add_argument("--headless", action="store_true", help="Diagnostic only: run the native comparison on the training headless path")
    p.add_argument("--script-actions", action="store_true", help="Diagnostic only: replay decision-cadence script rows instead of native per-tick scheduling")
    p.add_argument("--act-period", type=int, default=4, help="Recorded decision period for --script-actions")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    replay_path = Path(args.replay).resolve()
    report_path = Path(args.report).resolve()
    binary = Path(args.binary_path).resolve() if args.binary_path else repo_root / "BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth"
    reader = ReplayReader(replay_path)
    summary = reader.summary()
    manifest = reader.manifest
    ticks = reader.records("TICK")
    if not reader.complete:
        raise RuntimeError("native replay is incomplete; refusing to launch it")
    if not ticks or any(not isinstance(t.get("action"), list) for t in ticks):
        raise RuntimeError("replay has no complete native applied-action timeline")
    if manifest.get("verification") not in ("native_state_trace", "exact_simulation_verified"):
        raise RuntimeError(f"replay is not a native trace: {manifest.get('verification')!r}")
    recorded_runtime = manifest.get("runtime_fingerprint") or {}
    recorded_binary = ((recorded_runtime.get("binary") or {}).get("sha256"))
    current_runtime = runtime_fingerprint(repo_root, binary=binary)
    current_binary = ((current_runtime.get("binary") or {}).get("sha256"))
    if recorded_binary and recorded_binary != current_binary:
        raise RuntimeError(f"engine binary mismatch: replay={recorded_binary} current={current_binary}")

    work_parent = repo_root / ".rl_write_dirs"
    work_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="native-replay-", dir=work_parent))
    script_path = work_dir / "actions.csv"
    expected_path = work_dir / "expected.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with script_path.open("w", newline="") as stream:
            stream.write("# step,move_x,move_y,jump,crouch,attack,grab,drop,walk\n")
            writer = csv.writer(stream)
            tick_rows = enumerate(ticks) if not args.script_actions else ((index, ticks[index]) for index in range(0, len(ticks), max(1, args.act_period)))
            for index, tick in tick_rows:
                action = list(tick["action"])
                if len(action) != 8:
                    raise RuntimeError(f"native action at tick {index} has {len(action)} fields, expected 8")
                writer.writerow([
                    index if not args.script_actions else index // max(1, args.act_period),
                    float(action[0]), float(action[1]),
                    *[int(float(value) > 0.5) for value in action[2:8]],
                ])
        with expected_path.open("w") as stream:
            for tick in ticks:
                stream.write(json.dumps(tick, sort_keys=True, separators=(",", ":")) + "\n")

        config_lines = [
            f"global_time_scale_mult: {args.time_scale:g}",
            "skip_loading_pause: true",
            "has_detected_settings: true",
        ]
        if not args.headless:
            # Reflection capture is a render-only path. Disabling it keeps the
            # visible verifier on the same no-reflection resource profile as
            # the training engine and avoids a known Metal/OpenGL capture
            # failure on this Mac; it does not mutate gameplay state.
            config_lines.append("no_reflection_capture: true")
        config = "\n".join(config_lines)
        command = [
            str(binary), "--write-dir", str(work_dir), "--working-dir", str(repo_root),
            "--no-dialogues", "--level", args.level, "--config", config,
            "--rl-action-controller-id", "0", "--rl-action-script", str(script_path),
            "--rl-act-period", str(max(1, args.act_period) if args.script_actions else 1),
            "--rl-action-script-hold-seconds", str(args.hold_seconds),
            "--rl-replay-seed", str(args.seed), "--equivalence-expected", str(expected_path),
            "--equivalence-report", str(report_path),
        ]
        if args.script_actions:
            command.append("--rl-replay-script-actions")
        if args.headless:
            command += ["--disable-rendering", "--benchmark", "--benchmark-warmup-steps", "0",
                        # ReplayExhausted pauses the engine after the final
                        # archived tick; matching the benchmark bound keeps
                        # this diagnostic from appending unrecorded settle
                        # ticks. Visible replays do not use benchmark mode.
                        "--benchmark-steps", str(len(ticks)), "--benchmark-seed", str(args.seed)]
        if args.reset_mode is not None:
            command += ["--rl-replay-reset-mode", str(args.reset_mode)]
        if args.controlled_character_id is not None:
            command += ["--rl-replay-controlled-character-id", str(args.controlled_character_id)]
        if args.difficulty is not None:
            command += ["--rl-replay-difficulty", str(args.difficulty)]
        if args.opponents is not None:
            command += ["--rl-replay-opponents", str(args.opponents)]
        if args.weapons is not None:
            command += ["--rl-replay-weapons", str(args.weapons)]
        if args.species is not None:
            command += ["--rl-replay-species", str(args.species)]
        command = noaslr.wrap_command(command)
        print(f"native replay: {replay_path.name} ticks={len(ticks)}")
        completed = subprocess.run(command, cwd=repo_root)
        result = {}
        if report_path.exists():
            raw_report = report_path.read_text().strip()
            if raw_report:
                result = json.loads(raw_report)
            else:
                result = {
                    "verification": "engine_report_missing",
                    "report_path": str(report_path),
                }
        else:
            result = {
                "verification": "engine_report_missing",
                "report_path": str(report_path),
            }
        print(json.dumps({"returncode": completed.returncode, **result}, sort_keys=True))
        if completed.returncode != 0:
            return completed.returncode
        return 0 if result.get("verification") in ("exact_simulation_verified", "diverged") else 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
