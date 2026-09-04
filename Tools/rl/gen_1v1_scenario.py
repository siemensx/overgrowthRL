#!/usr/bin/env python3
"""Regenerates the project-owned 1v1-unarmed training scenario fork in place,
inside the installed Overgrowth Data/ tree (AUX_DATA), from the stock assets.

Why this exists: `Data/Scripts/arena_level_1v1_unarmed.as`,
`Data/Scripts/arena_level_1v1_unarmed_paths.xml`, and
`Data/Levels/arenas/oval_arena_1v1_unarmed.xml` are the scenario run8/run9/the
overnight run train against. Per AGENTS.md they are project-owned additive
files living in the *installed* Steam asset tree, never copied into this git
repo and never redistributed here -- so a Steam "verify integrity of game
files", a fresh install, or an accidental deletion silently loses them with
no git history to recover from (this happened once already, see
research-log/2026-08-17.md). This script is the recovery path: it contains
only the *transformation* (a handful of line-level edits, all authored by
this project, applied via unique-anchor string replacement) and applies it at
run time to whatever stock `arena_level.as` / `arena_level_paths.xml` /
`oval_arena.xml` the local install actually has -- it never stores or prints
the purchased original content itself.

Usage:
    python3 Tools/rl/gen_1v1_scenario.py [--overgrowth-data PATH] [--dry-run]

Idempotent: running it again just re-derives the same three files from the
stock originals (which this script never modifies) -- safe to re-run any
time the fork is missing or you want to confirm it still matches the current
transformation.
"""
import argparse
import difflib
import sys
from pathlib import Path

import paths

DEFAULT_DATA = str(paths.data_dir())


def transform_script(stock: str) -> str:
    """arena_level.as -> arena_level_1v1_unarmed.as.

    Every replacement below anchors on a short, exact substring of the stock
    script purely to locate the edit point (unavoidable for a line-level
    patch); the inserted text is entirely authored by this project.
    """
    out = stock

    # 1. RL curriculum-hook globals, right after curr_difficulty's declaration.
    anchor = "// Difficulty of current collection of enemies\nfloat curr_difficulty;\n"
    if anchor not in out:
        raise RuntimeError("anchor 1 (curr_difficulty declaration) not found -- stock script changed shape")
    out = out.replace(anchor, anchor + '''
// OGRL-20260817-028 Sec3.1: RL curriculum hook -- set by the training
// harness via set_rl_* ReceiveMessage tokens, sent immediately before every
// post_reset (see rl_shm_transport.cpp's SendScenarioMessages). rl_difficulty
// < 0 means "no RL harness has set this yet", so a level load before the
// first set_rl_difficulty (or a non-RL play session) falls back to this
// script's original initial_difficulty-0.5f behavior, unchanged.
float rl_difficulty = -1.0f;
int rl_opponents = 1;      // 1..3 requested; SetUpLevel currently only honors 1 -- see its comment
float rl_weapons = 0.0f;   // probability the round is armed, 0..1; 0 matches the original unarmed-1v1 default exactly
int rl_species = 0;        // 0 = random guard/raider (legacy default), 1 = guard, 2 = raider, 3 = civ, 4 = random all three
''', 1)

    # 2. ReceiveMessage: new set_rl_* tokens, right before set_all_hostile.
    anchor2 = '    } else if(token == "new_match"){\n        SetUpLevel(curr_difficulty);\n    } else if(token == "set_all_hostile"){'
    if anchor2 not in out:
        raise RuntimeError("anchor 2 (new_match/set_all_hostile) not found -- stock script changed shape")
    out = out.replace(anchor2, '''    } else if(token == "new_match"){
        SetUpLevel(curr_difficulty);
    } else if(token == "set_rl_difficulty"){
        token_iter.FindNextToken(msg);
        rl_difficulty = min(max(atof(token_iter.GetToken(msg)), 0.0f), 1.0f);
    } else if(token == "set_rl_opponents"){
        token_iter.FindNextToken(msg);
        rl_opponents = atoi(token_iter.GetToken(msg));
    } else if(token == "set_rl_weapons"){
        token_iter.FindNextToken(msg);
        rl_weapons = min(max(atof(token_iter.GetToken(msg)), 0.0f), 1.0f);
    } else if(token == "set_rl_species"){
        token_iter.FindNextToken(msg);
        rl_species = atoi(token_iter.GetToken(msg));
    } else if(token == "set_all_hostile"){''', 1)

    # 3. CreateEnemy: species widened from a fixed rand()%2+1 to an rl_species axis.
    anchor3 = "    string actor_path; // Path to actor xml\n    int fur_channel = -1; // Which tint mask channel corresponds to fur\n    int rnd = rand()%2+1;\n    switch(rnd){"
    if anchor3 not in out:
        raise RuntimeError("anchor 3 (CreateEnemy species roll) not found -- stock script changed shape")
    out = out.replace(anchor3, '''    string actor_path; // Path to actor xml
    int fur_channel = -1; // Which tint mask channel corresponds to fur
    // OGRL-20260817-028 Sec3.0: species is an RL curriculum axis now.
    // rl_species 0 (legacy default) reproduces the original rand()%2+1
    // exactly (guard/raider only -- civ was dead code, case 0 never drawn
    // by that expression, since it can only yield 1 or 2).
    int rnd;
    if(rl_species == 1){ rnd = 1; }
    else if(rl_species == 2){ rnd = 2; }
    else if(rl_species == 3){ rnd = 0; }
    else if(rl_species == 4){ rnd = rand()%3; }
    else { rnd = rand()%2+1; }
    switch(rnd){''', 1)

    # 4. SetUpLevel: force game_type_int=0 (the OGRL-20260816-023 1v1 fork
    #    itself) + a comment on why rl_opponents isn't wired to it yet.
    # This generator always reads the untouched installed stock script. The
    # stock arena chooses among three spawn groups; force that choice to the
    # two-spawn 1v1 group here. (The old anchor incorrectly expected the
    # already-generated fork, making both the recovery script and the human
    # duel generator fail on a clean install.)
    anchor4 = ("    bool knife_test = false;\n"
               "    int game_type_int = rand()%3;\n"
               "    if(knife_test){\n"
               "        game_type_int = 0;\n"
               "    }")
    if anchor4 not in out:
        raise RuntimeError("anchor 4 (game_type_int forcing) not found -- is arena_level.as already the 1v1 fork, not stock?")
    out = out.replace(anchor4, '''    // OGRL-20260817-028 Sec3.0/3.3: rl_opponents (opponent-count curriculum
    // axis) is accepted via set_rl_opponents but deliberately NOT wired to
    // game_type selection yet. Checked directly against oval_arena_1v1_unarmed.xml:
    // game_type=1 is 4 spawns with teams [0,0,1,1] (a 2v2 pairing) and
    // game_type=2 is 4 spawns with teams [1,0,2,3] (every character on a
    // DISTINCT team) -- neither maps cleanly onto "1 player vs N cooperating
    // hostiles" without first verifying whether same-team characters
    // actually cooperate and whether different-team hostiles fight each
    // other as much as the player (research-log 2026-08-17 finding; this was
    // an open question in OGRL-20260817-028 Sec3.0 and is still open).
    // Until that's verified in-game, always force the 2-spawn 1v1 group
    // regardless of rl_opponents, rather than silently training on a
    // scenario shape (free-for-all vs. coordinated outnumbering) nobody has
    // confirmed.
    bool knife_test = false;
    int game_type_int = 0;
    if(knife_test){
        game_type_int = 0;
    }''', 1)

    # 5. SetUpLevel: enemy difficulty now reads rl_difficulty when set.
    anchor5 = "        if(i != player_id){\n            CreateEnemy(obj, initial_difficulty-0.5f, character_spawns[i].team);\n        } else {"
    if anchor5 not in out:
        raise RuntimeError("anchor 5 (CreateEnemy call site) not found -- stock script changed shape")
    out = out.replace(anchor5, '''        if(i != player_id){
            // OGRL-20260817-028 Sec3.1: rl_difficulty (>=0 once the harness
            // has sent set_rl_difficulty) takes over from the original
            // initial_difficulty-0.5f expression entirely -- not blended
            // with it -- so the curriculum's sampled 0..1 difficulty maps
            // directly onto CreateEnemy's own 0..1 interpolation range.
            float enemy_difficulty = (rl_difficulty >= 0.0f) ? rl_difficulty : (initial_difficulty-0.5f);
            CreateEnemy(obj, enemy_difficulty, character_spawns[i].team);
        } else {''', 1)

    # 6. SetUpLevel: weapons deterministically OFF (1v1 fork) unless the
    #    harness rolls one in via rl_weapons.
    anchor6 = '    // Determine which weapons we are using\n    bool use_weapons = knife_test || rand()%3==0;\n    if(use_weapons){'
    if anchor6 not in out:
        raise RuntimeError("anchor 6 (use_weapons) not found -- is arena_level.as already forked with a different weapons comment?")
    out = out.replace(anchor6, '''    // Determine which weapons we are using
    // OGRL-20260817-028 Sec3.1: rl_weapons (0.0 by default, matching the
    // original OGRL-20260816-023 unarmed-1v1 behavior exactly when the
    // harness hasn't set it) is the probability this round is armed, rolled
    // fresh on every SetUpLevel call.
    bool use_weapons = (rl_weapons > 0.0f) && (RangedRandomFloat(0.0f, 1.0f) < rl_weapons);
    if(use_weapons){''', 1)

    return out


def transform_level_xml(stock: str) -> str:
    """oval_arena.xml -> oval_arena_1v1_unarmed.xml: rename + point at the fork script."""
    out = stock
    script_anchors = (
        "<Script>Data/Scripts/arena_level.as</Script>",
        "<Script>data/scripts/arena_level.as</Script>",
    )
    for anchor in script_anchors:
        if anchor in out:
            out = out.replace(anchor, anchor.replace(
                "arena_level.as", "arena_level_1v1_unarmed.as"), 1)
            break
    else:
        raise RuntimeError("expected <Script> tag not found in oval_arena.xml -- format changed")
    if "<Name>Oval Arena</Name>" in out:
        out = out.replace("<Name>Oval Arena</Name>", "<Name>Oval Arena 1v1 Unarmed</Name>", 1)
    elif "<Name>oval_arena</Name>" in out:
        out = out.replace("<Name>oval_arena</Name>", "<Name>oval_arena_1v1_unarmed</Name>", 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overgrowth-data", default=DEFAULT_DATA, help="Path to the installed Overgrowth Data/ directory")
    ap.add_argument("--dry-run", action="store_true", help="Print a summary diff instead of writing files")
    args = ap.parse_args()

    data = Path(args.overgrowth_data)
    stock_script = data / "Scripts" / "arena_level.as"
    stock_level = data / "Levels" / "arenas" / "oval_arena.xml"
    fork_script = data / "Scripts" / "arena_level_1v1_unarmed.as"
    fork_paths = data / "Scripts" / "arena_level_1v1_unarmed_paths.xml"
    stock_paths = data / "Scripts" / "arena_level_paths.xml"
    fork_level = data / "Levels" / "arenas" / "oval_arena_1v1_unarmed.xml"

    for p in (stock_script, stock_level, stock_paths):
        if not p.exists():
            print(f"ERROR: stock asset not found: {p}", file=sys.stderr)
            return 1

    new_script = transform_script(stock_script.read_text(encoding="utf-8"))
    new_level = transform_level_xml(stock_level.read_text(encoding="utf-8"))
    new_paths = stock_paths.read_text(encoding="utf-8")  # byte-identical copy; only exists because
                                                          # Level::GetPath() keys off the script's own filename

    if args.dry_run:
        for name, existing_path, new_text in (
            ("arena_level_1v1_unarmed.as", fork_script, new_script),
            ("arena_level_1v1_unarmed_paths.xml", fork_paths, new_paths),
            ("oval_arena_1v1_unarmed.xml", fork_level, new_level),
        ):
            if existing_path.exists():
                old_text = existing_path.read_text(encoding="utf-8")
                diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                                  fromfile=f"existing/{name}", tofile=f"generated/{name}", lineterm=""))
                print(f"--- {name}: {'IDENTICAL' if not diff else str(len(diff)) + ' diff lines'} ---")
            else:
                print(f"--- {name}: MISSING, would be created ({len(new_text)} bytes) ---")
        return 0

    fork_script.write_text(new_script, encoding="utf-8")
    fork_paths.write_text(new_paths, encoding="utf-8")
    fork_level.write_text(new_level, encoding="utf-8")
    print(f"Wrote {fork_script}")
    print(f"Wrote {fork_paths}")
    print(f"Wrote {fork_level}")
    print("Note: LevelInfo/navmesh caches are generated on first load into the per-worker write-dir cache; nothing else is needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
