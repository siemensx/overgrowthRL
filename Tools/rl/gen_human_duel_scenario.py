#!/usr/bin/env python3
"""Generate the additive, two-player human-versus-checkpoint arena.

Purchased game data stays in the installed AUX_DATA tree. This file stores
only the authored transformation from the installed stock arena script and
level, so a fresh install can recreate the runtime scenario without copying
game assets into the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_DATA = (
    "/Users/pavlov/Library/Application Support/Steam/steamapps/common/"
    "Overgrowth/Overgrowth.app/Contents/MacOS/Data"
)

# Keep the human match's gameplay script forked from the exact scenario used
# by run15.  The training fork owns the deterministic 1v1 spawn selection,
# unarmed default, RL curriculum hooks, and reset behavior; re-forking the
# stock arena here silently changes the policy's input distribution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_1v1_scenario import transform_script as transform_training_script


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def transform_script(stock: str) -> str:
    """Turn the run15 training fork into a two-player no-auto-rematch fork."""
    out = transform_training_script(stock)

    spawn_start = out.find("    // Spawn characters for each spawn point, with randomly selected player")
    spawn_end = out.find("    // Determine which weapons we are using", spawn_start)
    if spawn_start < 0 or spawn_end < 0 or spawn_end <= spawn_start:
        raise RuntimeError("training fork: player spawn block anchor not found")
    new_spawn = """    // Spawn both sides as ordinary player actors, in the same two-spawn
    // order as run15's controlled-player branch. Slot 0 is the checkpoint
    // so its native observation self.id remains the first actor id; slot 1
    // is the human. AvatarControlManager maps these tags to controllers.
    int num_char_spawns = character_spawns.size();
    player_id = 0;
    for(int i=0; i<num_char_spawns; ++i){
        Object @obj = ReadObjectFromID(character_spawns[i].obj_id);
        Object@ char_obj = SpawnObjectAtSpawnPoint(obj, level.GetPath("char_player"));
        char_obj.SetPlayer(true);
        ScriptParams@ char_params = char_obj.GetScriptParams();
        int team = character_spawns[i].team;
        char_params.SetString("Teams", ""+team);
        char_params.SetString("OGRL Participant Slot", ""+i);
        if(i == 1) {
            char_params.SetString("OGRL Human Duel Role", "human");
        } else {
            char_params.SetString("OGRL Human Duel Role", "checkpoint");
        }

        vec3 color = FloatTintFromByte(RandReasonableColor());
        float tint_amount = 0.5f;
        color = mix(color, ColorFromTeam(team), tint_amount);
        color = mix(color, vec3(1.0-(player_skill-0.5f)), 0.5f);
        player_colors[2] = color;
        for(int j=0; j<4; ++j){
            char_obj.SetPaletteColor(j, player_colors[j]);
        }
    }

"""
    out = out[:spawn_start] + new_spawn + out[spawn_end:]
    out = _replace_once(
        out,
        "    VictoryCheck();",
        "    // Python owns the result window and reset; the stock arena's\n"
        "    // VictoryCheck would silently start another round.\n"
        "    HumanDuelOverlay();",
        "disable stock round loop",
    )
    # Keep the level-script portion deliberately small and compatible with
    # the stock arena script API. The richer vision visualization is emitted
    # by the native RL transport, which has the exact observation buffer and
    # camera/controller ownership available; arena-level scripts do not.
    overlay = r'''
// OGRL-20260820-046: the native transport owns the match overlay so it can
// render it at the actual frame cadence. This hook intentionally stays empty
// to prevent a second AngelScript text overlay from competing with it.
void HumanDuelOverlay() {
}

'''
    out = _replace_once(out, "\nvoid Update() {\n", "\n" + overlay + "void Update() {\n", "overlay insertion")
    return out


def transform_bot_probe_script(stock: str) -> str:
    """Create a diagnostic 1v2: checkpoint versus human + stock expert NPC.

    This deliberately keeps the checkpoint on external controller 1 and the
    human on native controller 0.  The third actor is produced by the exact
    CreateEnemy(..., difficulty=1.0) path used by Run 15's benchmark, on the
    human's team.  It is an A/B probe for actor/control-state mismatch, not a
    selector-visible gameplay mode.
    """
    out = transform_training_script(stock)
    out = _replace_once(
        out,
        "    int game_type_int = 0;",
        "    int game_type_int = 1;",
        "bot probe four-spawn group",
    )

    spawn_start = out.find("    // Spawn characters for each spawn point, with randomly selected player")
    spawn_end = out.find("    // Determine which weapons we are using", spawn_start)
    if spawn_start < 0 or spawn_end < 0 or spawn_end <= spawn_start:
        raise RuntimeError("training fork: player spawn block anchor not found")
    new_spawn = """    // Diagnostic composition: external checkpoint (team 0), native human
    // (team 1), and a difficulty-1.0 stock expert NPC allied with the human.
    // Ignore the fourth game-type-1 spawn so the probe remains a 1v2.
    int num_char_spawns = min(character_spawns.size(), 3);
    player_id = 0;
    for(int i=0; i<num_char_spawns; ++i){
        Object @obj = ReadObjectFromID(character_spawns[i].obj_id);
        if(i == 2) {
            CreateEnemy(obj, 1.0f, 1);
            continue;
        }

        Object@ char_obj = SpawnObjectAtSpawnPoint(obj, level.GetPath("char_player"));
        char_obj.SetPlayer(true);
        ScriptParams@ char_params = char_obj.GetScriptParams();
        int team = 0;
        if(i == 1) {
            team = 1;
        }
        char_params.SetString("Teams", ""+team);
        char_params.SetString("OGRL Participant Slot", ""+i);
        if(i == 1) {
            char_params.SetString("OGRL Human Duel Role", "human");
        } else {
            char_params.SetString("OGRL Human Duel Role", "checkpoint");
        }

        vec3 color = FloatTintFromByte(RandReasonableColor());
        float tint_amount = 0.5f;
        color = mix(color, ColorFromTeam(team), tint_amount);
        color = mix(color, vec3(1.0-(player_skill-0.5f)), 0.5f);
        player_colors[2] = color;
        for(int j=0; j<4; ++j){
            char_obj.SetPaletteColor(j, player_colors[j]);
        }
        char_obj.UpdateScriptParams();
    }

"""
    out = out[:spawn_start] + new_spawn + out[spawn_end:]
    out = _replace_once(
        out,
        "    VictoryCheck();",
        "    // Diagnostic harness owns reset and termination.",
        "disable stock round loop for bot probe",
    )
    return out


def transform_bot_only_probe_script(stock: str) -> str:
    """Run 15's exact 1v1 composition with its player on controller 1."""
    out = transform_training_script(stock)
    out = _replace_once(
        out,
        "    player_id = rand()%num_char_spawns;",
        "    player_id = 0;",
        "bot-only probe checkpoint spawn",
    )
    out = _replace_once(
        out,
        "            char_params.SetString(\"Teams\", \"\"+team);",
        "            char_params.SetString(\"Teams\", \"\"+team);\n"
        "            char_params.SetString(\"OGRL Participant Slot\", \"0\");\n"
        "            char_params.SetString(\"OGRL Human Duel Role\", \"checkpoint\");\n"
        "            char_obj.UpdateScriptParams();",
        "bot-only probe participant tag",
    )
    out = _replace_once(
        out,
        "    VictoryCheck();",
        "    // Diagnostic harness owns reset and termination.",
        "disable stock round loop for bot-only probe",
    )
    return out


def transform_level_xml(stock: str) -> str:
    if "<Script>data/scripts/arena_level.as</Script>" in stock:
        out = stock.replace("<Script>data/scripts/arena_level.as</Script>",
                            "<Script>data/scripts/arena_level_human_duel.as</Script>", 1)
    elif "<Script>Data/Scripts/arena_level.as</Script>" in stock:
        out = stock.replace("<Script>Data/Scripts/arena_level.as</Script>",
                            "<Script>Data/Scripts/arena_level_human_duel.as</Script>", 1)
    else:
        raise RuntimeError("level script: stock arena script anchor not found")
    if "<Name>Oval Arena</Name>" in out:
        out = out.replace("<Name>Oval Arena</Name>", "<Name>Oval Arena — Human Duel</Name>", 1)
    elif "<Name>oval_arena</Name>" in out:
        out = out.replace("<Name>oval_arena</Name>", "<Name>oval_arena_human_duel</Name>", 1)
    return out


def transform_bot_probe_level_xml(stock: str) -> str:
    if "<Script>data/scripts/arena_level.as</Script>" in stock:
        out = stock.replace(
            "<Script>data/scripts/arena_level.as</Script>",
            "<Script>data/scripts/arena_level_human_duel_bot_probe.as</Script>",
            1,
        )
    elif "<Script>Data/Scripts/arena_level.as</Script>" in stock:
        out = stock.replace(
            "<Script>Data/Scripts/arena_level.as</Script>",
            "<Script>Data/Scripts/arena_level_human_duel_bot_probe.as</Script>",
            1,
        )
    else:
        raise RuntimeError("bot probe level script: stock arena script anchor not found")
    if "<Name>Oval Arena</Name>" in out:
        out = out.replace("<Name>Oval Arena</Name>", "<Name>Oval Arena — Human Duel Bot Probe</Name>", 1)
    elif "<Name>oval_arena</Name>" in out:
        out = out.replace("<Name>oval_arena</Name>", "<Name>oval_arena_human_duel_bot_probe</Name>", 1)
    return out


def transform_bot_only_probe_level_xml(stock: str) -> str:
    if "<Script>data/scripts/arena_level.as</Script>" in stock:
        out = stock.replace(
            "<Script>data/scripts/arena_level.as</Script>",
            "<Script>data/scripts/arena_level_human_duel_bot_only_probe.as</Script>",
            1,
        )
    elif "<Script>Data/Scripts/arena_level.as</Script>" in stock:
        out = stock.replace(
            "<Script>Data/Scripts/arena_level.as</Script>",
            "<Script>Data/Scripts/arena_level_human_duel_bot_only_probe.as</Script>",
            1,
        )
    else:
        raise RuntimeError("bot-only probe level script: stock arena script anchor not found")
    if "<Name>Oval Arena</Name>" in out:
        out = out.replace("<Name>Oval Arena</Name>", "<Name>Oval Arena — Controller 1 Bot Probe</Name>", 1)
    elif "<Name>oval_arena</Name>" in out:
        out = out.replace("<Name>oval_arena</Name>", "<Name>oval_arena_human_duel_bot_only_probe</Name>", 1)
    return out


def generate(data: Path, dry_run: bool = False) -> dict:
    stock_script = data / "Scripts" / "arena_level.as"
    stock_paths = data / "Scripts" / "arena_level_paths.xml"
    stock_level = data / "Levels" / "arenas" / "oval_arena.xml"
    outputs = {
        "script": data / "Scripts" / "arena_level_human_duel.as",
        "paths": data / "Scripts" / "arena_level_human_duel_paths.xml",
        "level": data / "Levels" / "arenas" / "oval_arena_human_duel.xml",
    }
    for path in (stock_script, stock_paths, stock_level):
        if not path.is_file():
            raise FileNotFoundError(f"stock asset missing: {path}")
    script = transform_script(stock_script.read_text(encoding="utf-8"))
    level = transform_level_xml(stock_level.read_text(encoding="utf-8"))
    paths = stock_paths.read_text(encoding="utf-8")
    rendered = {"script": script, "paths": paths, "level": level}
    if not dry_run:
        for key, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered[key], encoding="utf-8")
    return {
        "files": {key: str(path) for key, path in outputs.items()},
        "stock_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in {
            "script": stock_script, "paths": stock_paths, "level": stock_level}.items()},
        "generated_bytes": {key: len(value.encode("utf-8")) for key, value in rendered.items()},
        "dry_run": dry_run,
    }


def generate_bot_probe(data: Path, dry_run: bool = False) -> dict:
    stock_script = data / "Scripts" / "arena_level.as"
    stock_paths = data / "Scripts" / "arena_level_paths.xml"
    stock_level = data / "Levels" / "arenas" / "oval_arena.xml"
    outputs = {
        "script": data / "Scripts" / "arena_level_human_duel_bot_probe.as",
        "paths": data / "Scripts" / "arena_level_human_duel_bot_probe_paths.xml",
        "level": data / "Levels" / "arenas" / "oval_arena_human_duel_bot_probe.xml",
    }
    for path in (stock_script, stock_paths, stock_level):
        if not path.is_file():
            raise FileNotFoundError(f"stock asset missing: {path}")
    rendered = {
        "script": transform_bot_probe_script(stock_script.read_text(encoding="utf-8")),
        "paths": stock_paths.read_text(encoding="utf-8"),
        "level": transform_bot_probe_level_xml(stock_level.read_text(encoding="utf-8")),
    }
    if not dry_run:
        for key, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered[key], encoding="utf-8")
    return {
        "files": {key: str(path) for key, path in outputs.items()},
        "generated_bytes": {key: len(value.encode("utf-8")) for key, value in rendered.items()},
        "dry_run": dry_run,
    }


def generate_bot_only_probe(data: Path, dry_run: bool = False) -> dict:
    stock_script = data / "Scripts" / "arena_level.as"
    stock_paths = data / "Scripts" / "arena_level_paths.xml"
    stock_level = data / "Levels" / "arenas" / "oval_arena.xml"
    outputs = {
        "script": data / "Scripts" / "arena_level_human_duel_bot_only_probe.as",
        "paths": data / "Scripts" / "arena_level_human_duel_bot_only_probe_paths.xml",
        "level": data / "Levels" / "arenas" / "oval_arena_human_duel_bot_only_probe.xml",
    }
    for path in (stock_script, stock_paths, stock_level):
        if not path.is_file():
            raise FileNotFoundError(f"stock asset missing: {path}")
    rendered = {
        "script": transform_bot_only_probe_script(stock_script.read_text(encoding="utf-8")),
        "paths": stock_paths.read_text(encoding="utf-8"),
        "level": transform_bot_only_probe_level_xml(stock_level.read_text(encoding="utf-8")),
    }
    if not dry_run:
        for key, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered[key], encoding="utf-8")
    return {
        "files": {key: str(path) for key, path in outputs.items()},
        "generated_bytes": {key: len(value.encode("utf-8")) for key, value in rendered.items()},
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overgrowth-data", default=DEFAULT_DATA)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bot-probe", action="store_true",
                        help="generate the isolated checkpoint-vs-human-plus-expert diagnostic level")
    parser.add_argument("--bot-only-probe", action="store_true",
                        help="generate the exact 1v1 composition with the checkpoint on controller 1")
    args = parser.parse_args()
    data = Path(args.overgrowth_data).expanduser().resolve()
    if args.bot_probe and args.bot_only_probe:
        parser.error("choose only one probe mode")
    if args.bot_only_probe:
        result = generate_bot_only_probe(data, args.dry_run)
    elif args.bot_probe:
        result = generate_bot_probe(data, args.dry_run)
    else:
        result = generate(data, args.dry_run)
    for key, path in result["files"].items():
        print(f"{key}: {path}")
    if "stock_sha256" in result:
        print(f"stock fingerprints: {result['stock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
