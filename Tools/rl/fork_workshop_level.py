#!/usr/bin/env python3
"""Fork a hand-made (workshop) arena into the deterministic 1v1 RL scenario.

The generated corpus is all machine-built from two primitives. Nothing in the
project has ever evaluated the agent on a level a person designed, so "does it
generalise to real maps" has been unanswerable. Any arena driven by
arena_level.as with the standard 10 character_spawn placeholders forks exactly
the way oval does: repoint <Script> at the 1v1 fork and rename.

Only levels whose every type_file exists in the stock install can be forked this
way -- a level referencing mod-only assets needs its mod loaded, and activating
workshop mods segfaults this build during mod activation (observed 2026-09-05).
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

FORK_SCRIPT = "arena_level_1v1_unarmed.as"


def missing_assets(text: str, data: Path) -> list[str]:
    out = []
    for m in sorted(set(re.findall(r'type_file="([^"]+)"', text))):
        rel = m[5:] if m.startswith("Data/") else m
        if not (data / rel).exists():
            out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to the workshop level .xml")
    ap.add_argument("--name", required=True, help="output level name (no extension)")
    ap.add_argument("--force", action="store_true", help="fork even if assets are missing")
    a = ap.parse_args()

    data = paths.data_dir()
    src = Path(a.source)
    text = src.read_text(errors="ignore")

    miss = missing_assets(text, data)
    if miss and not a.force:
        print(f"refusing: {len(miss)} asset(s) not in the stock install, e.g. {miss[:3]}", file=sys.stderr)
        return 1

    n_spawn = text.count("character_spawn")
    if n_spawn < 2:
        print(f"refusing: only {n_spawn} character_spawn placeholders", file=sys.stderr)
        return 1

    out = text
    replaced = False
    for old in ("data/scripts/arena_level.as", "Data/Scripts/arena_level.as",
                "data/Scripts/arena_level.as", "Data/scripts/arena_level.as"):
        if f"<Script>{old}</Script>" in out:
            out = out.replace(f"<Script>{old}</Script>",
                              f"<Script>{old.rsplit('/', 1)[0]}/{FORK_SCRIPT}</Script>", 1)
            replaced = True
            break
    if not replaced:
        print("refusing: no <Script>…arena_level.as</Script> tag found", file=sys.stderr)
        return 1

    out = re.sub(r"<Name>[^<]*</Name>", f"<Name>{a.name}</Name>", out, count=1)
    dst = data / "Levels" / "arenas" / f"{a.name}.xml"
    dst.write_text(out, encoding="utf-8")
    print(f"wrote {dst}")
    n_assets = len(set(re.findall(r'type_file="([^"]+)"', out)))
    print(f"  {out.count('<EnvObject')} EnvObjects, {n_spawn} spawns, {n_assets} unique assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
