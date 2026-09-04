#!/usr/bin/env python3
"""Stage 5.4 smoke test / usage example for shm_env.ShmEnv.

Attaches to a running engine's shm transport and drives it with a constant
forward action purely from Python (not --rl-action-test-forward), confirming
the full loop end to end -- this is the reference client the C++ side
(Source/Main/rl_shm_transport.{h,cpp}) was validated against.

Usage: start the engine first, e.g.
    BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth \\
        --write-dir <isolated dir> --working-dir <repo_root> \\
        --disable-rendering --no-dialogues \\
        --benchmark --benchmark-warmup-steps 0 --benchmark-steps 500 --benchmark-seed 1 \\
        --level arenas/oval_arena.xml \\
        --rl-shm-name /ogrl_smoke --rl-action-controller-id 0
then, once the level has loaded:
    python3 Tools/rl/shm_smoketest.py /ogrl_smoke 100
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shm_env import ShmEnv

DEFAULT_OBS_FLOATS = 252  # schema v3: K=8 entities (23 floats each incl. attacked_by_id), 16 rays, 4 history steps


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "/ogrl_smoke0"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    env = ShmEnv(name, obs_floats=DEFAULT_OBS_FLOATS)
    print(f"attached: schema_version={env.schema_version} los_rule_version={env.los_rule_version} obs_floats={env.obs_floats}")

    first_pos = None
    last_pos = None
    done_seen = False
    for _ in range(steps):
        obs = env.wait_for_observation()
        pos = obs.values[0:3]
        if first_pos is None:
            first_pos = pos
        last_pos = pos
        if obs.done:
            done_seen = True
        env.write_action(move_x=0.0, move_y=1.0, jump=False, crouch=False, attack=False, grab=False)

    print(f"steps={steps} first_pos={first_pos} last_pos={last_pos} done_seen={done_seen}")
    env.request_shutdown()
    env.close()


if __name__ == "__main__":
    main()
