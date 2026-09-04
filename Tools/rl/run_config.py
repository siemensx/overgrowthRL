"""Shared helper: read a run's OWN environment configuration from its
run.json manifest, rather than trusting a tool's CLI defaults.

OGRL-20260817-028 Sec6.1/Sec8.1: two tools had already broken on the exact
same bug before this existed -- diagnose_checkpoint.py defaulted to the
10-character brawl at 120Hz/frame_stack=1 and couldn't load a run8/run9
checkpoint at all (obs_dim 260 vs. the required 1040, failing its own shape
guard), and watch.py had no --act-period flag at all, so it always ran
checkpoints at 120Hz regardless of what they were trained at (30Hz for
anything using act_period=4). Both are "a tool silently used a stale default
instead of asking the run what it actually was" -- the same root cause,
fixed once here instead of a third time somewhere else (the dashboard's
replay launcher, the evaluator).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_run_env_config(repo_root: str | Path, run_id: str, runs_root: str | Path | None = None) -> dict:
    """Reads runs/<run_id>/run.json and returns the fields every replay/eval
    tool needs to reproduce this run's environment exactly: level,
    frame_stack, act_period, k_standby, n_envs, max_episode_steps,
    reward_profile, and the checkpoint path this run itself wrote (if any --
    useful as a --checkpoint default, not authoritative if the caller wants
    a different step). Raises FileNotFoundError with a clear message (not a
    bare one from json.load) if the run doesn't exist or has no manifest."""
    root = Path(runs_root) if runs_root else Path(repo_root) / "Tools/rl/runs"
    manifest_path = root / run_id / "run.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no run.json found for run '{run_id}' at {manifest_path} -- check --runs-root and the run id")
    manifest = json.loads(manifest_path.read_text())
    env = manifest.get("env", {})
    algo = manifest.get("algo", {})
    return {
        "level": env.get("level") or algo.get("level") or "arenas/oval_arena.xml",
        "frame_stack": env.get("frame_stack", algo.get("frame_stack", 1)),
        "act_period": env.get("act_period", algo.get("act_period", 1)),
        "k_standby": env.get("k_standby", algo.get("k_standby", 0)),
        "n_envs": env.get("n_envs", algo.get("n_envs", 1)),
        "max_episode_steps": env.get("max_episode_steps", algo.get("max_episode_steps", 900)),
        "reward_profile": manifest.get("reward_profile", "default"),
        "checkpoint_path": algo.get("checkpoint_path"),
        "obs_schema_version": (manifest.get("code") or {}).get("obs_schema_version"),
    }
