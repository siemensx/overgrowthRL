#!/usr/bin/env python3
"""Replays a ghost action-trace CSV (written by Tools/rl/ppo/watch.py) at
real speed, rendered, so a specific watched episode can be re-watched exactly
without needing the checkpoint or PyTorch -- just the engine and the CSV.

Deliberately does NOT go through the shm transport at all: RLAction's own
scripted-action harness (--rl-action-script, built in Stage 5.3 to validate
timing-sensitive combos before the transport existed -- research-log
OGRL-20260816-006) already replays a step-indexed CSV of exactly this shape
natively in the engine. Reusing it here means a ghost replay has one fewer
moving part than the live watch session that recorded it -- no Python-side
policy, no shm handshake, just the engine executing a fixed input sequence.

Determinism -- FIXED 2026-08-17 (OGRL-20260817-030), was a known gap before
tonight: reproducing the *exact* recorded trajectory (not just the same
scripted button presses) also needs the same opponent seed/difficulty the
episode was originally sampled with. The engine's own reset mechanism
(Engine::ResetRLTrainingScenario/SoftResetRLTrainingScenario) is gated on the
shm transport being enabled, which this replay path deliberately isn't -- so
a new, independent mechanism (RLReplaySeed, Source/Main/rl_replay_seed.h/.cpp)
was added: pass --seed (now actually wired, via --rl-replay-seed) and
optionally --difficulty/--opponents/--weapons/--species, and the engine
reseeds its RNG streams + re-applies those curriculum axes once, immediately
after its own natural initial level load, using the exact same
set_rl_*-message-then-post_reset recipe every mid-training reset already
uses (see rl_replay_seed.h's module comment). Omit --seed (leave it at the
default None) to fall back to the old unseeded behavior explicitly, e.g. for
a hand-authored ghost CSV that was never tied to a specific recorded episode.

Playback speed/length -- FIXED 2026-08-17 (OGRL-20260817-031), was a real bug
before tonight: a script's "step" column is a DECISION index (one row per
--act-period physics ticks -- watch.py/tape.py record one row per decision,
not per tick), but RLAction::Apply() used to advance its internal counter
once per TICK unconditionally. With the training-standard act_period=4
(30Hz), every replay played 4x too fast, and once the recorded decisions ran
out, RLAction just held the last one forever while the arena level's own
stock "next round" logic silently kept the window running with stale input.
Pass --act-period matching what the tape/checkpoint was actually trained at
(see run.json's env.act_period) and the engine now paces the script at the
right speed AND requests a clean quit the moment the recording ends -- one
episode in, one episode out, window closes on its own.

Hold time -- ADDED 2026-08-17 (-033), REVISED 2026-08-19 (-039): a recording's
final action is usually the decisive one, and its consequence lands a couple of
ticks AFTER the last scripted input (measured: script ends tick 232, knockout
registers tick 234). --hold-seconds now stages that: the first 0.75s keeps
SIMULATING so the outcome resolves, then the engine FREEZES (paused=true) on
that frame and pulls the camera back to frame both characters, then quits.

The original -033 version simulated for the WHOLE hold, which was actively
misleading: a ~2s recording was followed by ~5s of the arena's own round logic
running on stale held input -- the agent standing still, being killed, and a new
round starting. Viewers read that as the replay, and it contradicted the tape's
recorded outcome. Freezing is the fix; see research-log 2026-08-19.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noaslr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ghost_csv", help="path to a ghost CSV written by watch.py")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))  # Tools/rl/replay_ghost.py -> Tools/rl -> Tools -> repo root
    p.add_argument("--level", default="arenas/oval_arena.xml")
    p.add_argument("--seed", type=int, default=None,
                    help="OGRL-20260817-030: the seed the ghost/tape was originally recorded with -- now actually "
                         "applied (via --rl-replay-seed) to reproduce the same opponent, not just the same inputs. "
                         "Omit for the old unseeded behavior.")
    p.add_argument("--difficulty", type=float, default=None, help="curriculum difficulty (0..1) to apply alongside --seed")
    p.add_argument("--opponents", type=int, default=None, help="opponent count to apply alongside --seed")
    p.add_argument("--weapons", type=float, default=None, help="armed-round probability to apply alongside --seed")
    p.add_argument("--species", type=int, default=None, help="opponent species mode to apply alongside --seed")
    p.add_argument("--act-period", type=int, default=1,
                    help="OGRL-20260817-031: ticks per recorded decision -- must match what the CSV was actually "
                         "recorded at (run.json's env.act_period, typically 4) or playback speed/length is wrong. "
                         "Default 1 preserves the old every-tick behavior for a hand-authored per-tick script.")
    p.add_argument("--hold-seconds", type=float, default=6.0,
                    help="OGRL-20260817-033, revised -039: total time to linger after the recording ends. The first "
                         "0.75s keeps SIMULATING so the final action's consequence lands (a knockout registers a "
                         "couple of ticks after the last scripted input); the REMAINDER is FROZEN on that frame, "
                         "with the camera pulled back to frame both characters. 0 quits instantly.")
    p.add_argument("--controller-id", type=int, default=0)
    p.add_argument("--binary-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    binary_path = Path(args.binary_path) if args.binary_path else repo_root / "BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth"

    write_dir_parent = repo_root / ".rl_write_dirs"
    write_dir_parent.mkdir(parents=True, exist_ok=True)
    write_dir = Path(tempfile.mkdtemp(prefix="replay-ghost-", dir=write_dir_parent))

    config_str = "\n".join(["global_time_scale_mult: 1", "skip_loading_pause: true", "has_detected_settings: true"])
    command = [
        str(binary_path),
        "--write-dir", str(write_dir),
        "--working-dir", str(repo_root),
        "--no-dialogues",
        "--level", args.level,
        "--config", config_str,
        "--rl-action-controller-id", str(args.controller_id),
        "--rl-action-script", str(Path(args.ghost_csv).resolve()),
        "--rl-act-period", str(args.act_period),
        "--rl-action-script-hold-seconds", str(args.hold_seconds),
    ]
    seeded = args.seed is not None
    if seeded:
        command += ["--rl-replay-seed", str(args.seed)]
        if args.difficulty is not None:
            command += ["--rl-replay-difficulty", str(args.difficulty)]
        if args.opponents is not None:
            command += ["--rl-replay-opponents", str(args.opponents)]
        if args.weapons is not None:
            command += ["--rl-replay-weapons", str(args.weapons)]
        if args.species is not None:
            command += ["--rl-replay-species", str(args.species)]
    command = noaslr.wrap_command(command)

    print(f"replaying {args.ghost_csv} at act_period={args.act_period} -- window closes automatically when the "
          f"recording ends, or close it / Ctrl+C to stop early")
    if seeded:
        print(f"seeded replay: seed={args.seed} difficulty={args.difficulty} opponents={args.opponents} "
              f"weapons={args.weapons} species={args.species} -- opponent should match the original recording.")
    else:
        print("NOTE: no --seed given -- unseeded replay, opponent behavior will NOT match any specific original recording.")
    try:
        subprocess.run(command, cwd=repo_root)
    finally:
        import shutil
        shutil.rmtree(write_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
