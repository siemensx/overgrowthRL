"""Platform-dependent locations for the engine binary and the purchased assets.

Every RL tool used to hardcode macOS paths (the arm64 .app bundle and the
Steam install under ~/Library/Application Support), which made them unusable
on the Windows trainer host. Resolution order is:

    1. an explicit argument passed by the caller (unchanged behaviour);
    2. the OGRL_BINARY / OGRL_AUX_DATA environment variables;
    3. the platform default below.

AGENTS.md forbids copying purchased assets into the repository, so the asset
root stays a configurable path into the Steam installation on both hosts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


def engine_binary(repo_root: Path, explicit: str | os.PathLike | None = None) -> Path:
    """Path to the RL engine build for this platform."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("OGRL_BINARY")
    if env:
        return Path(env)
    if IS_WINDOWS:
        return Path(repo_root) / "BuildWin64" / "Release" / "Overgrowth.exe"
    if IS_MACOS:
        return Path(repo_root) / "BuildArm64" / "Overgrowth.app" / "Contents" / "MacOS" / "Overgrowth"
    return Path(repo_root) / "BuildRelease64" / "Overgrowth"


def aux_data(explicit: str | os.PathLike | None = None) -> Path:
    """Directory containing the purchased `Data/` tree (its PARENT, matching
    the engine's AUX_DATA convention -- not `Data` itself)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("OGRL_AUX_DATA")
    if env:
        return Path(env)
    if IS_WINDOWS:
        return Path(r"C:\Program Files (x86)\Steam\steamapps\common\Overgrowth")
    return Path(
        "/Users/pavlov/Library/Application Support/Steam/steamapps/common/"
        "Overgrowth/Overgrowth.app/Contents/MacOS"
    )


def data_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The purchased `Data/` directory itself."""
    if explicit:
        return Path(explicit)
    return aux_data() / "Data"


def shm_prefix(default: str = "/ogrl_vec") -> str:
    """POSIX-style shm base name. Kept identical across platforms; the engine
    (RLIpc::ToObjectName) and shm_env.py both translate it for Windows."""
    return os.environ.get("OGRL_SHM_PREFIX", default)
