#!/usr/bin/env python3
"""One supervisor for every training run (OGRL-20261007-008). Replaces the per-run win_runNN.bat
files, which were copies of each other that differed in a handful of flags (archived in attic/launchers).

    python Tools/rl/run_profile.py <profile>            # run it: resume / fork / fresh, restart on exit
    python Tools/rl/run_profile.py <profile> --check    # preflight only: maps, level script, seed checkpoint
    python Tools/rl/run_profile.py <profile> --print    # print the train_vec command line and exit

A profile is Tools/rl/profiles/<name>.json (see profiles/README.md). Start order on every launch:
  1. the run's own checkpoint exists          -> --resume-from it (the normal 4-hour recycle)
  2. else the profile names seed_from          -> --resume-from checkpoints/<run_id>_seed.pt, a frozen copy
                                                  of seed_from made at the first launch
  3. else                                       -> fresh network with the base's fresh_args
Exit 0 from train_vec (a graceful stop via control.json, or the step target) ends the supervisor;
anything else (75 = the --max-wall-hours recycle, a crash) relaunches after 30 s, at most 300 times.
After every exit the supervisor kills engines still carrying this launch's shared-memory prefix, and
every launch uses a fresh prefix (AGENTS.md invariant 3).

Host rules baked in (AGENTS.md): BelowNormal priority on CPUs 0-9 for trainer and engines, one engine
launched at a time, never elevated. The Mac does not train (user decision 2026-10-04): without --force
this refuses to start training on macOS; --check and --print work everywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILES = HERE / "profiles"
CKPT_DIR = HERE / "ppo" / "checkpoints"
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from canonical_eval import CONTROLS  # noqa: E402

IS_WINDOWS = os.name == "nt"
HOST_ROOT = Path(os.environ.get("OGRL_HOST_ROOT", r"C:\ogrl" if IS_WINDOWS else str(REPO / ".ogrl_host")))
WIN_BINARY = r"C:\ogrl\optimization\BuildWinV6_20261002\Release\Overgrowth.exe"
HOST_ENV = {"OGRL_ENGINE_PRIORITY": "below", "OGRL_ENGINE_AFFINITY": "0x3FF",
            "OGRL_TRAINER_PRIORITY": "below", "OGRL_TRAINER_AFFINITY": "0x3FF",
            "OGRL_LAUNCH_WAVE_SIZE": "1", "OGRL_ALLOW_NENVS_CHANGE": "1"}


def expand_levels(spec) -> list[str]:
    """'arenas/t_train_{101..124}.xml' -> 24 names; a list is taken as-is (each entry expanded)."""
    items = spec if isinstance(spec, list) else [spec]
    out = []
    for it in items:
        m = re.search(r"\{(\d+)\.\.(\d+)\}", it)
        if m:
            out += [it[:m.start()] + str(i) + it[m.end():] for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        else:
            out.append(it)
    return out


def load_profile(name: str) -> dict:
    path = Path(name) if name.endswith(".json") else PROFILES / f"{name}.json"
    prof = json.loads(path.read_text())
    base = json.loads((PROFILES / prof["base"]).read_text()) if prof.get("base") else {}
    merged = {**base, **{k: v for k, v in prof.items() if k != "args"}}
    merged["args"] = list(base.get("args", [])) + list(prof.get("args", []))
    merged["levels"] = expand_levels(prof.get("levels", base.get("levels", [])))
    merged["_path"] = str(path)
    return merged


def flag_value(args: list[str], flag: str, default=None):
    """Last occurrence wins, as argparse does."""
    val = default
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
    return val


def checkpoint_path(prof: dict) -> Path:
    return CKPT_DIR / f"{prof['run_id']}.pt"


def seed_path(prof: dict) -> Path:
    return CKPT_DIR / f"{prof['run_id']}_seed.pt"


def preflight(prof: dict, make_seed: bool) -> list[str]:
    """Problems that would make the run train on something other than what the profile says."""
    problems = []
    data = paths.data_dir()
    n_maps = len(prof["levels"])
    n_envs = int(flag_value(prof["args"], "--n-envs", 4))
    k_standby = int(flag_value(prof["args"], "--k-standby", 0))
    if n_maps and (n_envs + k_standby) % n_maps:
        problems.append(f"engines {n_envs}+{k_standby} is not a multiple of the {n_maps}-map corpus "
                        "(AGENTS.md invariant 6: some maps would get no episodes)")
    opp_cap = int(flag_value(prof["args"], "--opponents-cap", 1))
    for lvl in prof["levels"]:
        f = data / "Levels" / lvl
        if not f.exists():
            problems.append(f"missing level {lvl} under {data / 'Levels'}")
            continue
        if opp_cap > 3:
            groups = set(re.findall(r'name="game_type" type="string" val="(\d+)"', f.read_text(errors="replace")))
            if str(opp_cap + 1) not in groups:
                problems.append(f"{lvl} has no 1v{opp_cap} spawn group (game_type {opp_cap + 1}); it would fall "
                                "back to 1v3 and every 1vN fight there would be unwinnable")
    if prof.get("requires_scenario"):
        script = data / "Scripts" / "arena_level_1v1_unarmed.as"
        if not script.exists() or prof["requires_scenario"] not in script.read_text(errors="replace"):
            problems.append(f"level script {script} predates '{prof['requires_scenario']}': run "
                            "Tools/rl/gen_1v1_scenario.py on this host")
    if os.environ.get("OGRL_BINARY") or IS_WINDOWS:
        binary = Path(os.environ.get("OGRL_BINARY", WIN_BINARY))
        if not binary.exists():
            problems.append(f"engine binary missing: {binary}")
    if not checkpoint_path(prof).exists() and prof.get("seed_from"):
        src, dst = CKPT_DIR / prof["seed_from"], seed_path(prof)
        if not dst.exists():
            if not src.exists():
                problems.append(f"seed checkpoint {src} missing")
            elif make_seed:
                tmp = dst.with_suffix(".tmp")
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                print(f"froze seed: {src.name} -> {dst.name}")
            # else: --check only reports; the supervisor freezes the seed at its first launch, i.e. after
            # switch_profile.sh has stopped the run that writes seed_from, so the copy is a final checkpoint.
    return problems


def build_command(prof: dict, shm_prefix: str) -> tuple[list[str], str]:
    ck = checkpoint_path(prof)
    if ck.exists():
        start, how = ["--resume-from", str(ck)], f"resume {ck.name}"
    elif prof.get("seed_from"):
        start, how = ["--resume-from", str(seed_path(prof))], f"fork from {seed_path(prof).name}"
    else:
        start, how = list(prof.get("fresh_args", [])), "fresh network"
    cmd = [sys.executable, "-u", str(HERE / "ppo" / "train_vec.py"), "--repo-root", str(REPO),
           "--levels", ",".join(prof["levels"]), "--shm-prefix", shm_prefix, "--seed", str(prof.get("seed", 0)),
           "--checkpoint-path", str(ck), *start, "--run-id", prof["run_id"], *prof["args"]]
    for line in CONTROLS[prof["controls"]]:
        cmd += ["--engine-config-line", line]
    cmd += ["--purpose", prof.get("purpose", prof["run_id"])]
    return cmd, how


def kill_engines(tag: str) -> None:
    if IS_WINDOWS:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine "
              f"-like '*{tag}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}")
        subprocess.call(["powershell", "-NoProfile", "-Command", ps], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    else:
        subprocess.call(["pkill", "-f", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def other_trainer_running() -> bool:
    if IS_WINDOWS:
        ps = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine "
              "-like '*train_vec.py*' }).Count")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
        return out.strip() not in ("", "0")
    return subprocess.call(["pgrep", "-f", "ppo/train_vec.py"], stdout=subprocess.DEVNULL) == 0


def log(prof: dict, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prof['run_id']}: {msg}"
    print(line, flush=True)
    with open(HOST_ROOT / f"{prof['run_id']}.log", "a") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--check", action="store_true", help="preflight only, changes nothing; no training")
    ap.add_argument("--print", action="store_true", help="print the train_vec command and exit")
    ap.add_argument("--force", action="store_true", help="allow training on macOS (the Mac does not train)")
    ap.add_argument("--max-tries", type=int, default=300)
    a = ap.parse_args()
    prof = load_profile(a.profile)
    HOST_ROOT.mkdir(parents=True, exist_ok=True)
    if a.print:
        cmd, how = build_command(prof, f"/ogrl_{prof['shm_tag']}_PRINT")
        print(f"# {prof['run_id']}: {how}; {len(prof['levels'])} maps")
        print(" ".join(f'"{c}"' if " " in c or c == "" else c for c in cmd))
        return 0
    problems = preflight(prof, make_seed=not a.check)
    if a.check:
        print(json.dumps({"profile": prof["_path"], "run_id": prof["run_id"], "maps": len(prof["levels"]),
                          "data_dir": str(paths.data_dir()), "checkpoint_exists": checkpoint_path(prof).exists(),
                          "seed_exists": seed_path(prof).exists(), "seed_from": prof.get("seed_from"),
                          "problems": problems}, indent=1))
        return 1 if problems else 0
    if problems:
        for p in problems:
            log(prof, f"PREFLIGHT FAILED: {p}")
        return 2
    if not IS_WINDOWS and not a.force:
        print("refusing to train on this host: the Mac does not train (AGENTS.md, 2026-10-04); --force overrides",
              file=sys.stderr)
        return 2
    if other_trainer_running():
        log(prof, "another train_vec.py is running on this host -- refusing to start a second trainer")
        return 3
    env = dict(os.environ)
    env.update(HOST_ENV)
    if IS_WINDOWS:
        env.setdefault("OGRL_BINARY", WIN_BINARY)
    (HOST_ROOT / "active_profile.json").write_text(json.dumps(
        {"profile": Path(prof["_path"]).stem, "run_id": prof["run_id"], "since": time.strftime("%Y-%m-%d %H:%M:%S"),
         "repo": str(REPO)}, indent=1))
    for tries in range(1, a.max_tries + 1):
        shm = f"/ogrl_{prof['shm_tag']}_{random.randint(0, 99999)}{tries}"
        cmd, how = build_command(prof, shm)
        log(prof, f"launch {tries}: {how}, shm {shm}")
        with open(HOST_ROOT / f"{prof['run_id']}.out", "a") as out:
            rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=out, stderr=subprocess.STDOUT)
        log(prof, f"exited {rc}")
        kill_engines(shm)
        if rc == 0:
            return 0
        time.sleep(30)
    return 1


if __name__ == "__main__":
    sys.exit(main())
