"""Stage 2/3 determinism fix (research-log OGRL-20260815-038).

Wraps every engine subprocess launch through noaslr_launcher (see
noaslr_launcher.c), which disables ASLR before exec'ing the real binary.
Confirmed root cause of the Stage 1/2 same-seed determinism failure: address
space layout, not RNG or camera state, both of which were verified identical
across diverging runs. 10/10 same-seed runs launched through this wrapper
were bit-identical, vs. a ~50% match rate launched normally.

Every RL tool that spawns the engine for anything claiming determinism
(equivalence digests, paired A/B, concurrency sweeps, the process pool) should
route through wrap_command() rather than invoking the binary directly.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

_SOURCE_NAME = "noaslr_launcher.c"
_BINARY_NAME = "noaslr_launcher"

# vec_env.py launches N OvergrowthEnv workers concurrently from thread-pool
# threads (research-log OGRL-20260816-012), and each one's first call routes
# through here -- without this lock, N threads could all see a missing/stale
# binary at once and race to `clang -o` the same output path (non-atomic:
# truncate-then-write), risking a corrupted binary or one thread exec'ing it
# mid-write. Only ever exercised in practice when the binary is actually
# missing/stale (the common case -- an up-to-date binary -- takes the fast
# path without contending on the lock), so this wasn't caught by testing;
# found by re-reading this function specifically because vec_env.py was about
# to call it from multiple threads for the first time.
_build_lock = threading.Lock()


def _tools_rl_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_launcher_built() -> Path:
    """Compiles Tools/rl/noaslr_launcher.c if the binary is missing or stale."""
    tools_dir = _tools_rl_dir()
    source = tools_dir / _SOURCE_NAME
    binary = tools_dir / _BINARY_NAME
    with _build_lock:
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(["clang", "-O2", "-o", str(binary), str(source)], check=True)
    return binary


def wrap_command(command: list[str]) -> list[str]:
    """Prepends the ASLR-disabling launcher to an engine invocation.
    `command[0]` must be the engine binary path; the rest are its arguments.
    """
    launcher = ensure_launcher_built()
    return [str(launcher)] + command
