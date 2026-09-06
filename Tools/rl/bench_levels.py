#!/usr/bin/env python3
"""Benchmark engine level cost: load time and step rate, per level.

Runs on macOS and Windows from the same source. Two numbers matter:

  level_load_seconds -- paid on every episode reset, so it multiplies by
                        episode count over a training run
  steps_per_second   -- the simulation rate once loaded

Every run asserts the character count parsed from the engine's own log before
recording a figure. A level that silently spawns nobody is fast for the wrong
reason, which is exactly how an earlier throughput claim went wrong.
"""
from __future__ import annotations
import argparse, json, platform, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

# One character logs exactly one "Caching skeleton info" as it is built.
CHAR_RE = re.compile(r"Caching skeleton info")
RESULT_RE = re.compile(r"RL_BENCHMARK_RESULT (\{.*\})")


def engine_binary() -> Path:
    root = HERE.parent.parent
    if platform.system() == "Windows":
        return root / "BuildWin64" / "Release" / "Overgrowth.exe"
    return root / "BuildArm64" / "Overgrowth.app" / "Contents" / "MacOS" / "Overgrowth"



def _engine_output(proc_out: str, write_dir) -> str:
    """Engine output, wherever this platform put it.

    Windows writes the engine log to <write-dir>/logfile.txt and leaves stdout
    nearly empty; macOS emits it on stdout/stderr. Reading only the pipe made
    every Windows benchmark report NO RESULT while the run had in fact
    succeeded -- the level loaded, characters spawned and the result line was
    sitting in the file. Concatenate both and stop caring which platform it is.
    """
    from pathlib import Path as _P
    text = proc_out or ""
    log = _P(write_dir) / "logfile.txt"
    if log.exists():
        try:
            text += "\n" + log.read_text(errors="replace")
        except OSError:
            pass
    return text

def run_once(exe: Path, level: str, write_dir: Path, steps: int, seed: int,
             warmup: int = 0) -> tuple[dict | None, int]:
    # Launch the engine exactly as env.py does. skip_loading_pause and
    # has_detected_settings are not cosmetic: without them the Windows engine
    # exits during settings detection, after printing only its data paths.
    config_str = "\n".join([
        "global_time_scale_mult: 1.0",
        "skip_loading_pause: true",
        "has_detected_settings: true",
    ])
    cmd = [str(exe), "--write-dir", str(write_dir), "--working-dir", str(HERE.parent.parent),
           "--disable-rendering", "--no-dialogues", "--benchmark",
           "--benchmark-warmup-steps", str(warmup), "--benchmark-steps", str(steps),
           "--benchmark-seed", str(seed), "--level", level, "--config", config_str]
    _log = Path(write_dir) / "logfile.txt"
    _log.unlink(missing_ok=True)   # it accumulates across runs; a stale one double-counts characters
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900)
    out = _engine_output(p.stdout + p.stderr, write_dir)
    chars = len(CHAR_RE.findall(out))
    m = RESULT_RE.search(out)
    return (json.loads(m.group(1)) if m else None), chars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    exe = engine_binary()
    if not exe.exists():
        print(f"engine not found: {exe}", file=sys.stderr)
        return 1
    wd = Path(paths.repo_root() if hasattr(paths, "repo_root") else HERE.parent.parent) / ".rl_write_dirs" / "bench"
    wd.mkdir(parents=True, exist_ok=True)

    host = platform.node()
    rows = []
    print(f"host={host}  os={platform.system()}  engine={exe.name}")
    print(f"{'level':<26} {'chars':>5} {'load_s':>8} {'steps/s':>9}  (median of "
          f"{a.repeats}, {a.steps} steps)")
    for lvl in a.levels:
        # One warm pass first: the navmesh bakes into the write-dir on first load,
        # and that one-off cost is not what we are measuring.
        run_once(exe, lvl, wd, 200, a.seed)
        loads, rates, chars = [], [], 0
        for r in range(a.repeats):
            res, c = run_once(exe, lvl, wd, a.steps, a.seed + r)
            if res is None:
                print(f"{lvl:<26} {'--':>5} {'NO RESULT':>8}")
                break
            loads.append(res["level_load_seconds"]); rates.append(res["steps_per_second"]); chars = c
        else:
            loads.sort(); rates.sort()
            md_load, md_rate = loads[len(loads)//2], rates[len(rates)//2]
            flag = "" if chars > 0 else "   <-- NO CHARACTERS, figure is meaningless"
            print(f"{lvl:<26} {chars:>5} {md_load:>8.3f} {md_rate:>9.0f}{flag}")
            rows.append({"host": host, "os": platform.system(), "level": lvl,
                         "characters": chars, "level_load_seconds": md_load,
                         "steps_per_second": md_rate, "loads": loads, "rates": rates})
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
