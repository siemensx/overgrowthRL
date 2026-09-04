"""Tier-1 live ghost replay tapes (OGRL-20260817-028 Sec8.1).

Everything needed to reconstruct a fight is already in the observation the
trainer holds in memory: absolute self position, self forward, self health/
state, and per-entity relative position, forward, health, state. Recording
it costs a file append -- no engine window, no rendering, nothing that would
steal cycles from training. This module is the recorder; the dashboard is
the (separately built) viewer.

Selection policy (Sec8.1): worker 0's episode every `tape_every` updates
("watch it learn" mode), PLUS the single best- and worst-reward episode seen
across ALL workers in each `window_updates`-update window ("show me the best
fight and the worst fight"). Kept simple over a fully general N-way
best/worst-of-window tracker: this project's own workers rarely exceed
single digits, so buffering every worker's IN-PROGRESS episode in memory
(a few hundred decisions * ~70 floats each) is cheap, and only the file
writes for kept tapes cost anything.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from ogreplay import write_policy_trace

ACTION_NAMES = ("move_x", "move_y", "jump", "crouch", "attack", "grab", "drop", "walk")


def jsonl_to_ghost_csv(jsonl_path, out_csv_path) -> int:
    """Converts a recorded tape's .jsonl (this module's own decision_record()
    output) into a ghost action-trace CSV of exactly the shape watch.py
    already writes and replay_ghost.py already knows how to play back
    rendered in a real engine window (RLAction::LoadScript,
    Source/Main/rl_action.cpp) -- see replay_ghost.py's module docstring.
    Column order is ACTION_NAMES, matching decision_record()'s "act" field
    and RLAction::LoadScript's expected header exactly, deliberately no
    conversion step in between. Returns the number of rows written.

    This function only reproduces the recorded BUTTON PRESSES -- reproducing
    the original opponent's seed/difficulty too is handled separately, by
    dashboard server.py's _handle_replay("tape") passing --seed/--difficulty/
    etc. through to replay_ghost.py, which now actually wires them to the
    engine via RLReplaySeed (OGRL-20260817-030, Source/Main/rl_replay_seed.h)
    -- see that module's comment for how. Calling THIS function alone (e.g.
    from a script that doesn't also pass those flags) still only reproduces
    inputs, not the opponent -- the fix lives on the replay-launch path, not
    in the CSV conversion itself."""
    import csv
    import json as _json
    from pathlib import Path as _Path

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            act = d.get("act") or [0.0] * len(ACTION_NAMES)
            rows.append([
                len(rows), float(act[0]), float(act[1]),
                int(act[2] > 0.5), int(act[3] > 0.5), int(act[4] > 0.5),
                int(act[5] > 0.5), int(act[6] > 0.5), int(act[7] > 0.5),
            ])
    out_csv_path = _Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        # RLAction::LoadScript's parser only skips a line as a comment if it
        # starts with '#' -- see watch.py's own comment at the matching spot
        # (research-log OGRL-20260816-014, the exact bug that catches a plain
        # header row here).
        f.write("# " + ",".join(("step",) + ACTION_NAMES) + "\n")
        csv.writer(f).writerows(rows)
    return len(rows)


def decision_record(t: float, self_values: list, entity_dicts: list, action: list, reward_components: dict,
                     difficulty: float, layout, nearest_entities: int = 4) -> dict:
    """Builds one tape line from a SINGLE (non-frame-stacked, unnormalized)
    raw observation frame -- callers must pass layout.all_entities()'s output
    (or equivalent) already sorted nearest-first, which RLObservation already
    guarantees at the source."""
    # OGRL-20260817-028 Sec8.1 bugfix: self_values/entity fields are numpy
    # scalars (np.float32/np.int64) whenever the caller's raw observation
    # came from a numpy array (it always does, via vec_env.py's np.stack) --
    # json.dumps cannot serialize those, so every numeric leaving this
    # function is explicitly cast to a native Python float/int/bool.
    self_state_idx = int(max(range(len(self_values[layout.STATE])), key=lambda i: self_values[layout.STATE][i]))
    kept = [e for e in entity_dicts if e["valid"]][:nearest_entities]
    return {
        "t": float(t),
        "self": {
            "pos": [float(x) for x in self_values[layout.POS]],
            "fwd": [float(self_values[layout.FORWARD][0]), float(self_values[layout.FORWARD][2])],
            "hp": float(self_values[layout.TEMP_HEALTH] + self_values[layout.BLOOD_HEALTH]),
            "block_hp": float(self_values[layout.BLOCK_HEALTH]),
            "state": self_state_idx,
            "grounded": bool(self_values[layout.GROUNDED]),
        },
        "ents": [
            {
                "id": int(e["id"]),
                "rel": [float(x) for x in e["rel_pos"]],
                "fwd": [float(x) for x in e["fwd"]],
                "hp": float(e["temp_health"] + e["blood_health"]),
                "state": int(max(range(len(e["state"])), key=lambda i: e["state"][i])),
                "ally": bool(e["is_ally"]),
                "weapon": int(max(range(len(e["weapon_type"])), key=lambda i: e["weapon_type"][i])),
            }
            for e in kept
        ],
        "act": [float(x) for x in action],
        "rew": dict(reward_components),
        "d": difficulty,
    }


def _native_episode_segment(trace_path: str | Path | None, seed: int | None) -> tuple[dict | None, list[dict]]:
    """Extract one flushed native digest episode from a worker process trace.

    The engine resets the per-episode state chain and emits a reset marker
    before the next post-reset tick. Seeds are monotonically diversified by
    VecOvergrowthEnv, so selecting the latest matching marker is stable even
    when a fast worker has already started the following episode.
    """
    if not trace_path:
        return (None, [])
    try:
        lines = Path(trace_path).read_text(errors="replace").splitlines()
    except OSError:
        return (None, [])
    matches: list[tuple[dict, list[dict]]] = []
    current: list[dict] = []
    current_reset: dict | None = None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "reset":
            if current_reset is not None and current_reset.get("seed") == seed and current:
                matches.append((current_reset, current))
            current = []
            current_reset = record
        elif record.get("kind") == "tick":
            current.append(record)
    if current_reset is not None and current_reset.get("seed") == seed and current:
        matches.append((current_reset, current))
    return matches[-1] if matches else (None, [])


def _native_episode_ticks(trace_path: str | Path | None, seed: int | None) -> list[dict]:
    return _native_episode_segment(trace_path, seed)[1]


class TapeRecorder:
    def __init__(self, run_dir: Path, tape_every: int = 10, window_updates: int = 50, keep_recent: int = 200):
        self.tapes_dir = Path(run_dir) / "tapes"
        self.tapes_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.tapes_dir.parent
        self.replay_control_path = self.run_dir / "replay-control.json"
        try:
            self.runtime = json.loads((self.run_dir / "run.json").read_text()).get("runtime")
        except (OSError, json.JSONDecodeError):
            self.runtime = None
        self.tape_every = max(1, tape_every)
        self.window_updates = max(1, window_updates)
        self.keep_recent = keep_recent
        # worker -> list[dict] decisions buffered for the episode CURRENTLY
        # in progress in that worker's slot. Cleared on every episode end
        # regardless of whether it gets saved.
        self._buffers: dict[int, list] = {}
        self._window_best = None   # (reward, worker, decisions, meta)
        self._window_worst = None
        self._window_start_update = 0
        self._saved_this_update = set()  # worker indices already saved as "sampled" this update, avoid double-tagging

    def _capture_request(self) -> dict | None:
        try:
            return json.loads(self.replay_control_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _consume_capture_request(self, outcome: str) -> bool:
        """Return whether this episode was manually requested for capture.

        The request is consumed only when its condition matches. A pending
        ``capture_next_loss`` therefore survives wins, which is what a user
        expects when arming the control before a run enters a failure mode.
        """
        request = self._capture_request() or {}
        command = request.get("command")
        if command == "capture_next":
            request = {"command": None, "t": time.time()}
        elif command == "capture_next_loss":
            if outcome != "lost":
                return False
            request = {"command": None, "t": time.time()}
        elif command == "capture_next_n":
            count = int(request.get("count", 1))
            if count > 1:
                request["count"] = count - 1
            else:
                request = {"command": None, "t": time.time()}
        else:
            return False
        try:
            tmp = self.replay_control_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(request))
            tmp.replace(self.replay_control_path)
        except OSError:
            pass
        return True

    def record_decision(self, worker: int, decision: dict) -> None:
        self._buffers.setdefault(worker, []).append(decision)

    def start_episode_if_sampled(self, worker: int, update: int) -> None:
        """Call at the moment a worker's episode STARTS (i.e. right after a
        reset). Worker 0's episode is tagged 'sampled' every tape_every
        updates -- tagging doesn't affect recording (every worker's current
        episode is always buffered, see record_decision), only whether the
        buffer gets kept as a tape once the episode ends."""
        pass  # sampling decision is made at episode END (see maybe_save), based on `update` at start time is not
              # needed since every episode's buffer already exists by the time it ends -- kept as a documented no-op
              # rather than removed, so the intended call site in train_vec.py's reset-handling stays obvious.

    def episode_ended(self, worker: int, update: int, seed: int | None, outcome: str, reward_total: float,
                       difficulty: float, sampled_worker0: bool, native_trace_path: str | None = None) -> None:
        """Call once a worker's episode ends, BEFORE its buffer is needed
        again for the next episode. Decides what to keep, writes tape files
        for anything kept, then clears the buffer for `worker` either way."""
        decisions = self._buffers.pop(worker, [])
        if not decisions:
            return
        if update - self._window_start_update >= self.window_updates:
            self._window_best = None
            self._window_worst = None
            self._window_start_update = update

        reasons = []
        if worker == 0 and sampled_worker0:
            reasons.append("sampled")
        if self._consume_capture_request(outcome):
            reasons.append("manual")
        if self._window_best is None or reward_total > self._window_best[0]:
            self._window_best = (reward_total, worker, decisions, {"reason": "best"})
            reasons.append("best")
        if self._window_worst is None or reward_total < self._window_worst[0]:
            self._window_worst = (reward_total, worker, decisions, {"reason": "worst"})
            reasons.append("worst")

        if reasons:
            self._write_tape(update, worker, seed, outcome, reward_total, difficulty, decisions, reasons,
                             native_trace_path=native_trace_path)
        self._prune()

    def _write_tape(self, update: int, worker: int, seed, outcome: str, reward_total: float, difficulty: float,
                     decisions: list, reasons: list, native_trace_path: str | None = None) -> None:
        name = f"{update}_w{worker}_e{len(decisions)}"
        jsonl_path = self.tapes_dir / f"{name}.jsonl"
        with open(jsonl_path, "w") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")
        meta = {
            "update": update, "worker": worker, "seed": seed, "outcome": outcome,
            "reward_total": reward_total, "difficulty": difficulty, "length_decisions": len(decisions),
            "reasons": reasons,
        }
        native_ticks = _native_episode_ticks(native_trace_path, seed)
        native_reset, _ = _native_episode_segment(native_trace_path, seed)
        has_native = bool(native_ticks)
        native_controlled_character_id = None
        for tick in native_ticks:
            controlled = [c.get("id") for c in tick.get("characters", []) if c.get("controlled")]
            if controlled:
                native_controlled_character_id = controlled[0]
                break
        # OGRL-20260820-048: the JSONL remains a compatibility artifact, but
        # the binary container now joins the policy trace to the native
        # per-physics-tick state digest emitted by the same engine process.
        # This is the source of truth the rendered launcher verifies online.
        episode_id = str(uuid.uuid4())
        visual_states = []
        for index, decision in enumerate(decisions):
            visual_states.append({
                "decision": index,
                "self": decision.get("self", {}),
                "entities": decision.get("ents", []),
            })
        replay_path = self.tapes_dir / f"{name}.ogreplay"
        replay_meta = {
            "run_id": self.tapes_dir.parent.name,
            "episode_id": episode_id,
            "update": update,
            "worker": worker,
            "seed": seed,
            "outcome": outcome,
            "reward_total": reward_total,
            "difficulty": difficulty,
            "reasons": reasons,
            "trace_kind": "policy_plus_native_state" if has_native else "policy_plus_observation",
            "state_schema": "native-digest-v2" if has_native else "observation-v1-partial",
            "authoritative_complete": bool(has_native),
            "verification": "native_state_trace" if has_native else "recorded_state",
            "native_tick_count": len(native_ticks),
            "native_reset": native_reset,
            "native_controlled_character_id": native_controlled_character_id,
            "native_trace_file": Path(native_trace_path).name if native_trace_path else None,
            "legacy_jsonl": jsonl_path.name,
            "terminal": {"decision": max(0, len(decisions) - 1), "outcome": outcome},
            "runtime_fingerprint": self.runtime,
        }
        try:
            write_policy_trace(replay_path, meta=replay_meta, decisions=decisions, visual_states=visual_states,
                               events=[{"kind": "terminal", "decision": max(0, len(decisions) - 1), "outcome": outcome}],
                               native_ticks=native_ticks)
            meta.update({
                "episode_id": episode_id,
                "replay_file": replay_path.name,
                "replay_status": "NATIVE STATE TRACE / UNVERIFIED" if has_native else "RECORDED STATE PLAYBACK",
                "authoritative_complete": bool(has_native),
            })
        except Exception as exc:  # noqa: BLE001 -- a capture failure must not kill training
            print(f"replay capture disabled for {name}: {exc}", file=sys.stderr)
            meta.update({"replay_status": "LEGACY / UNVERIFIABLE", "replay_error": str(exc)})
        (self.tapes_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))

    def _prune(self) -> None:
        metas = sorted(self.tapes_dir.glob("*.meta.json"), key=lambda p: p.stat().st_mtime)
        if len(metas) <= self.keep_recent:
            return
        # Never prune a tape tagged best/worst -- everything else is fair
        # game, oldest first, once over keep_recent.
        prunable = []
        for m in metas:
            try:
                meta = json.loads(m.read_text())
            except Exception:  # noqa: BLE001 -- a corrupt meta file is prunable, not fatal
                prunable.append(m)
                continue
            if "best" not in meta.get("reasons", []) and "worst" not in meta.get("reasons", []):
                prunable.append(m)
        excess = len(metas) - self.keep_recent
        for m in prunable[:excess]:
            m.unlink(missing_ok=True)
            m.with_suffix("").with_suffix(".jsonl").unlink(missing_ok=True)
            m.with_suffix("").with_suffix(".ogreplay").unlink(missing_ok=True)
