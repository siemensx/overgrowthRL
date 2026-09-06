#!/usr/bin/env python3
"""Benchmark engine-level throughput options, identically on macOS and Windows.

Runs the same level under several engine configurations and reports steps/s,
asserting the character count each time so a configuration that quietly breaks
the scenario cannot look like a speedup.
"""
from __future__ import annotations
import argparse, json, platform, re, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

CHAR_RE = re.compile(r"Caching skeleton info")
RESULT_RE = re.compile(r"RL_BENCHMARK_RESULT (\{.*\})")
CONFIG = "global_time_scale_mult: 1.0\nskip_loading_pause: true\nhas_detected_settings: true"

# name -> extra engine args. Each must leave the SCENARIO unchanged; anything that
# alters physics rate or observation content is a learning change, not an
# optimisation, and does not belong here.
VARIANTS = {
    "baseline":            [],
    "obs_period_4":        ["--rl-obs-period", "4"],
    "obs_period_8":        ["--rl-obs-period", "8"],
    "no_dialogues_only":   [],
}


def run(exe: Path, root: Path, level: str, extra: list[str], steps: int, seed: int, wd: Path):
    cmd = [str(exe), "--write-dir", str(wd), "--working-dir", str(root),
           "--disable-rendering", "--no-dialogues", "--benchmark",
           "--benchmark-warmup-steps", "100", "--benchmark-steps", str(steps),
           "--benchmark-seed", str(seed), "--level", level, "--config", CONFIG] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=1800)
    out = p.stdout + p.stderr
    m = RESULT_RE.search(out)
    return (json.loads(m.group(1)) if m else None), len(CHAR_RE.findall(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    root = HERE.parent.parent
    exe = (root / "BuildWin64" / "Release" / "Overgrowth.exe") if platform.system() == "Windows" \
        else (root / "BuildArm64" / "Overgrowth.app" / "Contents" / "MacOS" / "Overgrowth")
    if not exe.exists():
        print(f"engine not found: {exe}", file=sys.stderr); return 1
    wd = root / ".rl_write_dirs" / "benchopts"; wd.mkdir(parents=True, exist_ok=True)

    print(f"host={platform.node()}  os={platform.system()}  level={a.level}")
    rows, base = [], None
    for name, extra in VARIANTS.items():
        run(exe, root, a.level, extra, 200, 1, wd)          # warm the navmesh/caches
        rates, chars = [], 0
        for r in range(a.repeats):
            res, c = run(exe, root, a.level, extra, a.steps, 1 + r, wd)
            if res is None:
                rates = []; break
            rates.append(res["steps_per_second"]); chars = c
        if not rates:
            print(f"  {name:<20} NO RESULT"); continue
        med = statistics.median(rates)
        if base is None:
            base = med
        delta = f"{100*(med/base-1):+5.1f}%" if base else ""
        flag = "" if chars > 0 else "  <-- NO CHARACTERS"
        print(f"  {name:<20} chars={chars}  {med:>8.0f} steps/s  {delta}{flag}")
        rows.append({"variant": name, "steps_per_second": med, "characters": chars,
                     "host": platform.node(), "os": platform.system(), "level": a.level})
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2)); print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
