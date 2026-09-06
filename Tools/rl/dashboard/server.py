#!/usr/bin/env python3
"""OGRL-20260816-022/-023: stdlib-only local dashboard server. Reads
Tools/rl/runs/<run_id>/{run.json,metrics.jsonl,episodes.jsonl,events.jsonl}
-- never writes any of those. The ONE exception is control.json, written
via POST /api/runs/{id}/control for the pause/stop button (see
telemetry.py's module docstring for why that one channel exists despite
everything else being read-only).

No third-party dependencies, no build step. Binds 127.0.0.1 only.
"""
from __future__ import annotations

import json
import re
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIME_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
              ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8"}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Tools/rl -- for reward.py
from ogreplay import Float32Bits, Float64Bits, ReplayReader, load_replay_summary

# OGRL-20260816-024: "profile: run8" told the user nothing about what the
# agent is actually rewarded/penalized for -- this maps each RewardConfig
# field to a plain-language, sign-explicit description, so /api/runs/{id}/reward
# can return the ACTUAL numeric values for that run's profile paired with
# what triggers them. Field names are stable across profiles (only the
# VALUES differ), so this dict doesn't need to change per profile.
REWARD_FIELD_INFO = {
    "damage_taken_weight": {"sign": "negative", "trigger": "Per unit of own (temp+blood)/2 health lost this decision, whoever caused it. Always active."},
    "damage_dealt_weight": {"sign": "positive", "trigger": "Per unit of health lost by a NON-ALLY entity this agent itself most recently hit (causation-verified via attacked_by_id)."},
    "friendly_fire_weight": {"sign": "negative", "trigger": "Per unit of health lost by an ALLY this agent itself hit, PLUS a flat +1.0 (before this weight) if that ally is knocked out."},
    "self_knockout_penalty": {"sign": "negative", "trigger": "Once, the decision this agent transitions from awake to unconscious/dead."},
    "opponent_knockout_bonus": {"sign": "positive", "trigger": "Once per non-ally opponent this agent caused to transition from awake to unconscious/dead."},
    "time_cost": {"sign": "negative", "trigger": "Flat, every single decision -- discourages stalling. Always active."},
    "ragdoll_penalty": {"sign": "negative", "trigger": "Every decision this agent is ragdolled (limp) AND still awake -- encourages rolling out of ragdoll instead of waiting for the slow auto-recovery."},
    "closing_distance_weight": {"sign": "positive", "trigger": "Per meter closed toward the nearest non-ally entity within closing_distance_cap. 0.0 = OFF. Curriculum-gated in the default profile (tapers to 0 over training); permanently 0 in run8's profile -- no engagement bootstrap needed on a well-defined 1v1."},
    "closing_distance_cap": {"sign": "n/a", "trigger": "Meters -- beyond this range, closing_distance_weight does not apply even when > 0."},
    "stall_penalty_weight": {"sign": "negative", "trigger": "Every decision once stall_grace_steps consecutive decisions have passed with ZERO combat contact (either side landing a hit resets the clock to 0). 0.0 = OFF -- explicitly disabled in run8's profile."},
    "stall_grace_steps": {"sign": "n/a", "trigger": "Decisions of zero combat contact tolerated before stall_penalty_weight starts applying."},
}


def _sanitize_json(obj):
    """Recursively replace non-finite floats (NaN/Infinity/-Infinity) with
    None. Python's json.dumps emits these as bare NaN/Infinity/-Infinity
    tokens by default -- valid in Python's own JSON dialect (and accepted by
    json.loads on the way in, which is how a bad float from train_vec.py's
    telemetry survives long enough to reach here), but NOT valid JSON. A
    browser's native JSON.parse() (what fetch().json() uses) throws a
    SyntaxError on the bare token, which silently killed the dashboard's
    entire metrics/episodes/events fetch for any run that ever logged one --
    see research-log 2026-08-17 for the mean_ep_reward/mean_ep_length root
    cause this was chasing. Fixing the writer stops NEW bad values; this is
    the belt-and-suspenders read-side fix so already-written lines (smoke1's
    file is frozen; run10's live process needs a restart to pick up the
    writer fix) don't keep breaking the UI in the meantime."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None  # obj == obj is False only for NaN
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _resolve_reward_config(reward_profile: str) -> dict:
    """Returns the ACTUAL numeric RewardConfig for the given profile name,
    computed the same way train_vec.py does -- not a hardcoded guess."""
    try:
        from reward import RewardConfig, run8_reward_config
        import dataclasses
        cfg = run8_reward_config() if reward_profile == "run8" else RewardConfig()
        return dataclasses.asdict(cfg)
    except Exception as exc:  # noqa: BLE001 -- the dashboard must render even if this fails
        return {"_error": str(exc)}

_replay_jobs: dict[str, dict] = {}
_replay_lock = threading.Lock()
_match_jobs: dict[str, dict] = {}
_match_lock = threading.Lock()
_checkpoint_catalog_cache: list[dict] | None = None
_checkpoint_catalog_signature: tuple[tuple[str, int, int], ...] | None = None


def _safe_run_dir(runs_root: Path, run_id: str) -> Path | None:
    """Resolve run_id against runs_root and reject anything that escapes it
    (path traversal via a crafted run_id in the URL)."""
    if not run_id or not all(c.isalnum() or c in "_.-" for c in run_id):
        return None
    candidate = (runs_root / run_id).resolve()
    try:
        candidate.relative_to(runs_root.resolve())
    except ValueError:
        return None
    return candidate


def _safe_filename_component(name: str) -> bool:
    """Same character allowlist as _safe_run_dir, reused for tape/eval names
    that get interpolated into a filesystem path -- no '/', no '..', no
    absolute paths hiding in a URL segment."""
    return bool(name) and all(c.isalnum() or c in "_.-" for c in name)


def _checkpoint_dir() -> Path:
    return Handler.repo_root / "Tools/rl/ppo/checkpoints"


def _checkpoint_catalog() -> list[dict]:
    """Return server-owned checkpoint IDs with a conservative compatibility badge."""
    global _checkpoint_catalog_cache, _checkpoint_catalog_signature
    root = _checkpoint_dir()
    if not root.exists():
        return []
    paths = sorted(root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    signature = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    if _checkpoint_catalog_cache is not None and signature == _checkpoint_catalog_signature:
        return _checkpoint_catalog_cache
    try:
        from obs_schema import DEFAULT_LAYOUT, SCHEMA_VERSION
        current_floats = DEFAULT_LAYOUT.total_floats
    except Exception as exc:  # noqa: BLE001
        current_floats = None
        SCHEMA_VERSION = None
        import_error = str(exc)
    else:
        import_error = None
    out = []
    for path in paths:
        item = {"id": path.name, "label": path.stem, "size_bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime, "status": "blocked", "reason": "unreadable checkpoint"}
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item["sha256"] = digest
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            layout_floats = checkpoint.get("layout_total_floats")
            frame_stack = checkpoint.get("frame_stack")
            has_policy = isinstance(checkpoint.get("policy"), dict)
            has_normalizer = isinstance(checkpoint.get("obs_normalizer"), dict)
            item.update({"global_step": checkpoint.get("global_step"), "frame_stack": frame_stack,
                         "layout_total_floats": layout_floats, "schema_version": SCHEMA_VERSION})
            if import_error:
                item["reason"] = import_error
            elif not has_policy or not has_normalizer:
                item["reason"] = "missing policy or normalizer"
            elif layout_floats != current_floats:
                item["reason"] = f"observation layout {layout_floats} != current {current_floats}"
            elif frame_stack is None:
                item["reason"] = "missing frame-stack metadata"
            else:
                item["status"] = "ready"
                item["reason"] = "PPO / schema-compatible"
        except Exception as exc:  # noqa: BLE001
            item["reason"] = f"cannot inspect: {type(exc).__name__}"
        out.append(item)
    # Put the strongest compatible artifact first. File modification time is
    # useful for discovery, but a freshly copied 1k-step smoke checkpoint
    # must not silently become the default over a 46M-step production run.
    out.sort(key=lambda item: (item.get("global_step") if item.get("global_step") is not None else -1,
                               item.get("mtime", 0.0)), reverse=True)
    _checkpoint_catalog_signature = signature
    _checkpoint_catalog_cache = out
    return out


def _duel_levels() -> list[dict]:
    """Levels playable in Fight-a-checkpoint, i.e. driven by the human-duel script.

    Only these work: play_match spawns the human as a second player actor, which
    arena_level.as does not do. gen_arena_map.py --human-duel emits corpus maps in
    this form. oval is listed last and flagged, because a corpus-trained checkpoint
    has partly forgotten it (OGRL-20260905-064: 82.5% -> 60.0% at band 0.9) and
    fighting it there measures the map it lost rather than the policy it has.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import paths as _paths
        arenas = _paths.data_dir() / "Levels" / "arenas"
    except Exception:
        return []
    out = []
    for f in sorted(arenas.glob("*.xml")):
        try:
            head = f.read_text(errors="ignore")[:4096]
        except OSError:
            continue
        if "arena_level_human_duel.as" not in head.lower():
            continue
        name = f.stem
        out.append({
            "level": f"arenas/{f.name}",
            "label": name.replace("_duel", "").replace("oval_arena_human", "oval (stock)"),
            "trained_on": name.startswith("t_train_"),
            "warn": "forgotten by corpus-trained checkpoints" if name.startswith("oval") else "",
        })
    out.sort(key=lambda r: (r["level"].startswith("arenas/oval"), r["level"]))
    return out


def _match_dir(job_id: str) -> Path | None:
    if not _safe_filename_component(job_id):
        return None
    candidate = (Handler.runs_root / "_matches" / job_id).resolve()
    try:
        candidate.relative_to((Handler.runs_root / "_matches").resolve())
    except ValueError:
        return None
    return candidate


def _match_snapshot(job_id: str, job: dict | None = None) -> dict:
    match_dir = _match_dir(job_id)
    status = {}
    if match_dir is not None:
        status_path = match_dir / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                status = {"phase": "status_unreadable"}
    if job:
        status.setdefault("match_id", job_id)
        status.setdefault("checkpoint_id", job.get("checkpoint_id"))
        status.setdefault("log_tail", _read_match_log(job))
        proc = job.get("process")
        if proc is not None and proc.poll() is not None:
            status.setdefault("process_exit", proc.returncode)
            if status.get("phase") in (None, "loading", "scenario", "fighting", "resetting"):
                status["phase"] = "exited"
    return status


def _read_match_log(job: dict) -> str:
    path = Path(job.get("log_path", ""))
    try:
        return path.read_text(errors="ignore")[-4000:] if path.exists() else ""
    except OSError:
        return ""


def _list_eval_summaries(run_dir: Path) -> list[dict]:
    """OGRL-20260817-028 Sec8.1: runs/<id>/eval/<global_step>.json, one file
    per evaluate.py invocation. Lightweight per-entry summary (not the full
    per-band breakdown -- see the {global_step} sub-resource for that) so a
    run with many eval snapshots stays cheap to list."""
    eval_dir = run_dir / "eval"
    out = []
    if not eval_dir.exists():
        return out
    for p in sorted(eval_dir.glob("*.json")):
        try:
            global_step = int(p.stem)
        except ValueError:
            continue  # a .tmp or otherwise-named file -- not a real eval snapshot
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "global_step": global_step, "checkpoint": data.get("checkpoint"), "episodes": data.get("episodes"),
            "stochastic": data.get("stochastic"), "seed_base": data.get("seed_base"), "level": data.get("level"),
            "frame_stack": data.get("frame_stack"), "act_period": data.get("act_period"),
            "overall": data.get("overall"), "num_bands": len(data.get("bands") or []),
        })
    out.sort(key=lambda e: e["global_step"])
    return out


def _list_tapes(run_dir: Path) -> list[dict]:
    """OGRL-20260817-028 Sec8.1: pairs of <name>.jsonl/<name>.meta.json under
    runs/<id>/tapes/. Returns the parsed meta objects (newest-first by mtime)
    with the filename stem attached as "name" so the front end can fetch the
    matching .jsonl via GET .../tapes/{name}."""
    tapes_dir = run_dir / "tapes"
    out = []
    if not tapes_dir.exists():
        return out
    for p in sorted(tapes_dir.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = p.name
        if name.endswith(".meta.json"):
            name = name[: -len(".meta.json")]
        meta["name"] = name
        out.append(meta)
    return out


def _replay_paths(run_dir: Path) -> list[Path]:
    """Return new binary traces without treating legacy JSONL as exact."""
    paths = []
    for directory in (run_dir / "tapes", run_dir / "replays"):
        if directory.exists():
            paths.extend(directory.glob("*.ogreplay"))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)


def _list_replays(run_dir: Path) -> list[dict]:
    out = []
    for path in _replay_paths(run_dir):
        summary = load_replay_summary(path)
        sidecar = path.with_suffix(".meta.json")
        if sidecar.exists():
            try:
                summary.update(json.loads(sidecar.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        # The container's mechanically-derived label always wins over a
        # stale sidecar label. This prevents a copied/edited meta file from
        # upgrading recorded state into an exact replay claim.
        parsed = load_replay_summary(path)
        summary["status"] = parsed.get("status", summary.get("status"))
        summary["name"] = path.stem
        summary["path"] = str(path)
        out.append(summary)
    return out


def _replay_jsonable(value):
    if isinstance(value, Float32Bits):
        return {"__float32_bits__": value.bits, "value": value.value()}
    if isinstance(value, Float64Bits):
        return {"__float64_bits__": value.bits, "value": value.value()}
    if isinstance(value, dict):
        return {k: _replay_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replay_jsonable(v) for v in value]
    return value


def _resolve_replay_path(run_dir: Path, name: str) -> Path | None:
    if not _safe_filename_component(name):
        return None
    for path in _replay_paths(run_dir):
        if path.stem == name:
            return path
    return None


def _read_jsonl_tail(path: Path, offset: int) -> tuple[int, list[dict]]:
    """Seeks to byte `offset`, reads to EOF, returns only WHOLE lines and the
    new offset -- a line still being written is left for the next poll."""
    if not path.exists():
        return offset, []
    lines = []
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    new_offset = offset
    for raw_line in data.split(b"\n"):
        if raw_line == b"" and data.endswith(b"\n"):
            continue
        line_bytes = raw_line + b"\n"
        if not data.endswith(b"\n") and raw_line == data.rsplit(b"\n", 1)[-1] and not data.endswith(b"\n"):
            break  # partial trailing line, not yet flushed with a newline -- wait for next poll
        try:
            lines.append(json.loads(raw_line))
            new_offset += len(line_bytes)
        except json.JSONDecodeError:
            break
    return new_offset, lines


def _list_runs(runs_root: Path) -> list[dict]:
    out = []
    if not runs_root.exists():
        return out
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "run.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        metrics_path = run_dir / "metrics.jsonl"
        last_metric = None
        mtime = 0.0
        if metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
            try:
                with open(metrics_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    chunk = min(size, 8192)
                    f.seek(size - chunk)
                    tail = f.read().decode("utf-8", errors="ignore")
                    lines = [l for l in tail.splitlines() if l.strip()]
                    if lines:
                        last_metric = json.loads(lines[-1])
            except (OSError, json.JSONDecodeError):
                pass
        live = manifest.get("status") == "running" and (time.time() - mtime) < 120
        stale = manifest.get("status") == "running" and (time.time() - mtime) >= 120
        # OGRL-20260816-024: the pause DID work the first time it was tried,
        # but nothing in the UI showed it -- "live" only reflects "process
        # running and recently touched metrics.jsonl," which stays true while
        # paused too (the trainer is deliberately idle in a sleep loop, not
        # crashed). Surface the actual current command so the front end can
        # show a real PAUSED state instead of silently doing nothing visible.
        control_command = None
        control_path = run_dir / "control.json"
        if control_path.exists():
            try:
                control_command = json.loads(control_path.read_text()).get("command")
            except (OSError, json.JSONDecodeError):
                pass
        out.append({
            "run_id": run_dir.name, "manifest": manifest, "last_metric": last_metric,
            "live": live, "stale": stale, "control_command": control_command,
        })
    out.sort(key=lambda r: r["manifest"].get("started_at") or "", reverse=True)
    return out


class Handler(BaseHTTPRequestHandler):
    runs_root: Path = None  # set by main()
    repo_root: Path = None

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet -- this runs indefinitely alongside training

    def _send_json(self, obj, status=200):
        body = json.dumps(_sanitize_json(obj)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json({"error": message}, status=status)

    def _send_raw(self, body: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "":
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])

        if path == "/api/runs":
            return self._send_json({"runs": _list_runs(self.runs_root)})

        if path == "/api/status":
            runs = _list_runs(self.runs_root)
            live_runs = [r["run_id"] for r in runs if r["live"]]
            return self._send_json({"live_runs": live_runs, "cpu_busy": len(live_runs) > 0})

        if path == "/api/checkpoint-catalog":
            return self._send_json({"checkpoints": _checkpoint_catalog()})

        if path == "/api/duel-levels":
            return self._send_json({"levels": _duel_levels()})

        if path == "/api/matches":
            with _match_lock:
                jobs = list(_match_jobs.items())
            return self._send_json({"matches": [_match_snapshot(job_id, job) for job_id, job in jobs]})

        if path.startswith("/api/matches/"):
            job_id = path[len("/api/matches/"):]
            with _match_lock:
                job = _match_jobs.get(job_id)
            if job is None:
                match_dir = _match_dir(job_id)
                if match_dir is None or not match_dir.exists():
                    return self._send_error_json(404, "unknown match")
                return self._send_json(_match_snapshot(job_id))
            return self._send_json(_match_snapshot(job_id, job))

        parts = [p for p in path.split("/") if p]
        # /api/runs/{id}, /api/runs/{id}/metrics, /api/runs/{id}/episodes, /api/runs/{id}/events
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "runs":
            run_id = parts[2]
            run_dir = _safe_run_dir(self.runs_root, run_id)
            if run_dir is None or not run_dir.exists():
                return self._send_error_json(404, f"unknown run_id: {run_id}")
            sub = parts[3] if len(parts) > 3 else None
            if sub is None:
                manifest_path = run_dir / "run.json"
                if not manifest_path.exists():
                    return self._send_error_json(404, "run.json missing")
                return self._send_json(json.loads(manifest_path.read_text()))
            if sub == "metrics":
                offset = int(query.get("offset", ["0"])[0])
                new_offset, lines = _read_jsonl_tail(run_dir / "metrics.jsonl", offset)
                return self._send_json({"offset": new_offset, "lines": lines})
            if sub == "episodes":
                offset = int(query.get("offset", ["0"])[0])
                new_offset, lines = _read_jsonl_tail(run_dir / "episodes.jsonl", offset)
                return self._send_json({"offset": new_offset, "lines": lines})
            if sub == "events":
                events_path = run_dir / "events.jsonl"
                events = []
                if events_path.exists():
                    for line in events_path.read_text().splitlines():
                        if line.strip():
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                return self._send_json({"events": events})
            if sub == "reward":
                manifest_path = run_dir / "run.json"
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                profile = manifest.get("reward_profile", "default")
                values = _resolve_reward_config(profile)
                fields = []
                for name, info in REWARD_FIELD_INFO.items():
                    fields.append({"name": name, "value": values.get(name), **info})
                return self._send_json({"profile": profile, "fields": fields})
            if sub == "eval":
                # /api/runs/{id}/eval -> list summaries; /api/runs/{id}/eval/{global_step} -> full file.
                rest = parts[4] if len(parts) > 4 else None
                if rest is None:
                    return self._send_json({"evals": _list_eval_summaries(run_dir)})
                try:
                    global_step = int(rest)
                except ValueError:
                    return self._send_error_json(400, "global_step must be an integer")
                eval_path = run_dir / "eval" / f"{global_step}.json"
                if not eval_path.exists():
                    return self._send_error_json(404, f"no eval result at global_step={global_step}")
                try:
                    return self._send_json(json.loads(eval_path.read_text()))
                except (OSError, json.JSONDecodeError) as exc:
                    return self._send_error_json(500, f"failed to read eval result: {exc}")
            if sub == "tapes":
                # /api/runs/{id}/tapes -> list meta; /api/runs/{id}/tapes/{name} -> raw jsonl stream.
                rest = parts[4] if len(parts) > 4 else None
                if rest is None:
                    return self._send_json({"tapes": _list_tapes(run_dir)})
                if not _safe_filename_component(rest):
                    return self._send_error_json(400, "invalid tape name")
                tape_path = run_dir / "tapes" / f"{rest}.jsonl"
                if not tape_path.exists():
                    return self._send_error_json(404, f"unknown tape: {rest}")
                try:
                    body = tape_path.read_bytes()
                except OSError as exc:
                    return self._send_error_json(500, f"failed to read tape: {exc}")
                return self._send_raw(body, "application/x-ndjson; charset=utf-8")
            if sub == "replays":
                # /api/runs/{id}/replays -> summaries; /.../{name} -> a
                # bounded decoded view for the state-playback UI.
                rest = parts[4] if len(parts) > 4 else None
                if rest is None:
                    return self._send_json({"replays": _list_replays(run_dir)})
                replay_path = _resolve_replay_path(run_dir, rest)
                if replay_path is None:
                    return self._send_error_json(404, f"unknown replay: {rest}")
                try:
                    reader = ReplayReader(replay_path)
                    summary = reader.summary()
                    summary["decisions"] = _replay_jsonable(reader.records("DECISION")[:20000])
                    summary["visual_states"] = _replay_jsonable(reader.records("VISUAL")[:20000])
                    summary["events"] = _replay_jsonable(reader.records("EVENT")[:20000])
                    summary["reset"] = _replay_jsonable(reader.records("RESET")[:4])
                    return self._send_json(summary)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    return self._send_error_json(422, f"replay is unreadable: {exc}")
            return self._send_error_json(404, f"unknown sub-resource: {sub}")

        if path == "/api/checkpoints":
            return self._send_json({"checkpoints": _checkpoint_catalog()})

        if path.startswith("/api/replay/"):
            job_id = path[len("/api/replay/"):]
            with _replay_lock:
                job = _replay_jobs.get(job_id)
            if job is None:
                return self._send_error_json(404, "unknown replay job")
            log_tail = ""
            log_path = Path(job["log_path"])
            if log_path.exists():
                log_tail = log_path.read_text(errors="ignore")[-4000:]
            result = None
            report_path = Path(job.get("report_path", "")) if job.get("report_path") else None
            if report_path and report_path.exists():
                try:
                    raw_report = report_path.read_text().strip()
                    result = json.loads(raw_report) if raw_report else {"verification": "engine_report_missing"}
                except (OSError, json.JSONDecodeError):
                    result = {"verification": "report_unreadable"}
            return self._send_json({"job_id": job_id, "status": job["status"], "log_tail": log_tail, "result": result})

        return self._send_error_json(404, "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body_raw) if body_raw else {}
        except json.JSONDecodeError:
            return self._send_error_json(400, "invalid JSON body")

        parts = [p for p in path.split("/") if p]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "control":
            run_id = parts[2]
            run_dir = _safe_run_dir(self.runs_root, run_id)
            if run_dir is None or not run_dir.exists():
                return self._send_error_json(404, f"unknown run_id: {run_id}")
            command = payload.get("command")
            if command not in (None, "pause", "stop"):
                return self._send_error_json(400, f"invalid command: {command!r} (must be pause, stop, or null)")
            control_path = run_dir / "control.json"
            tmp = control_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"command": command, "t": time.time()}))
            import os
            os.replace(tmp, control_path)
            return self._send_json({"ok": True, "command": command})

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "replay-controls":
            run_id = parts[2]
            run_dir = _safe_run_dir(self.runs_root, run_id)
            if run_dir is None or not run_dir.exists():
                return self._send_error_json(404, f"unknown run_id: {run_id}")
            command = payload.get("command")
            allowed = {"capture_next", "capture_next_loss", "capture_next_n", None}
            if command not in allowed:
                return self._send_error_json(400, "invalid replay capture command")
            count = int(payload.get("count", 1)) if command == "capture_next_n" else 1
            if count < 1 or count > 20:
                return self._send_error_json(400, "capture count must be between 1 and 20")
            control_path = run_dir / "replay-control.json"
            tmp = control_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"command": command, "count": count, "t": time.time()}))
            import os
            os.replace(tmp, control_path)
            return self._send_json({"ok": True, "command": command, "count": count})

        if len(parts) == 6 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "replays" and parts[5] == "pin":
            run_id, name = parts[2], parts[4]
            run_dir = _safe_run_dir(self.runs_root, run_id)
            replay_path = _resolve_replay_path(run_dir, name) if run_dir and run_dir.exists() else None
            if replay_path is None:
                return self._send_error_json(404, "unknown replay")
            sidecar = replay_path.with_suffix(".meta.json")
            meta = {}
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            meta["pinned"] = bool(payload.get("pinned", True))
            tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
            tmp.write_text(json.dumps(meta, indent=2) + "\n")
            import os
            os.replace(tmp, sidecar)
            return self._send_json({"ok": True, "pinned": meta["pinned"]})

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "events":
            run_id = parts[2]
            run_dir = _safe_run_dir(self.runs_root, run_id)
            if run_dir is None or not run_dir.exists():
                return self._send_error_json(404, f"unknown run_id: {run_id}")
            record = {"t": time.time(), "kind": payload.get("kind", "note"),
                      "title": payload.get("title", ""), "body": payload.get("body", "")}
            with open(run_dir / "events.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            return self._send_json({"ok": True})

        if path == "/api/matches":
            return self._handle_match(payload)

        if path == "/api/replay":
            return self._handle_replay(payload)

        return self._send_error_json(404, "not found")

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "replay":
            job_id = parts[2]
            with _replay_lock:
                job = _replay_jobs.get(job_id)
            if job is None:
                return self._send_error_json(404, "unknown replay job")
            proc = job.get("process")
            if proc is not None and proc.poll() is None:
                proc.terminate()
            return self._send_json({"ok": True})
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "matches":
            job_id = parts[2]
            with _match_lock:
                job = _match_jobs.get(job_id)
            if job is None:
                return self._send_error_json(404, "unknown match")
            proc = job.get("process")
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    proc.terminate()
            with _match_lock:
                job["status"] = "stopping"
            return self._send_json({"ok": True, "match_id": job_id})
        return self._send_error_json(404, "not found")

    def _handle_match(self, payload: dict):
        """Validate a catalog ID, then launch the dedicated match supervisor."""
        checkpoint_id = payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not _safe_filename_component(checkpoint_id) or not checkpoint_id.endswith(".pt"):
            return self._send_error_json(400, "choose a checkpoint from the catalog")
        catalog = {item["id"]: item for item in _checkpoint_catalog()}
        checkpoint = catalog.get(checkpoint_id)
        if checkpoint is None:
            return self._send_error_json(404, "unknown checkpoint")
        if checkpoint.get("status") != "ready":
            return self._send_error_json(409, checkpoint.get("reason", "checkpoint is not compatible"))

        mode = payload.get("policy_mode", "deterministic")
        if mode not in ("deterministic", "sampled"):
            return self._send_error_json(400, "invalid policy mode")
        # Duel level. Defaults to oval for backwards compatibility, but a
        # corpus-trained checkpoint should be fought on a map it actually
        # trained on -- see play_match.py's --level help.
        level = payload.get("level") or "arenas/oval_arena_human_duel.xml"
        if not re.fullmatch(r"arenas/[A-Za-z0-9_.-]+\.xml", level):
            return self._send_error_json(400, "invalid level")

        with _match_lock:
            active_matches = [
                match_id for match_id, job in _match_jobs.items()
                if job.get("process") is not None and job["process"].poll() is None
            ]
        if active_matches:
            return self._send_error_json(409, "a checkpoint match is already running")

        live = [r["run_id"] for r in _list_runs(self.runs_root) if r["live"]]
        if live and not payload.get("force"):
            return self._send_error_json(409, f"training is live ({', '.join(live)}) -- stop it or pass force:true")

        match_id = f"match-{int(time.time() * 1000)}"
        match_dir = self.runs_root / "_matches" / match_id
        match_dir.mkdir(parents=True, exist_ok=False)
        log_path = match_dir / "match.log"
        status_path = match_dir / "status.json"
        session_dir = match_dir / "session"
        argv = [sys.executable, str(self.repo_root / "Tools/rl/play_match.py"),
                "--checkpoint", str(self.repo_root / "Tools/rl/ppo/checkpoints" / checkpoint_id),
                "--checkpoint-id", checkpoint_id,
                "--repo-root", str(self.repo_root),
                "--status-path", str(status_path),
                "--session-dir", str(session_dir),
                "--match-id", match_id,
                "--policy-mode", mode,
                "--level", level]
        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(argv, cwd=self.repo_root, stdout=log_file, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError as exc:
            try:
                log_file.close()
            except UnboundLocalError:
                pass
            return self._send_error_json(500, f"failed to launch match: {exc}")
        job = {"process": proc, "log_path": str(log_path), "status": "running", "checkpoint_id": checkpoint_id}
        with _match_lock:
            _match_jobs[match_id] = job

        def _watch_match_exit():
            code = proc.wait()
            try:
                log_file.close()
            except OSError:
                pass
            with _match_lock:
                if match_id in _match_jobs:
                    _match_jobs[match_id]["status"] = "exited"
                    _match_jobs[match_id]["exit_code"] = code

        threading.Thread(target=_watch_match_exit, daemon=True).start()
        return self._send_json({"match_id": match_id, "status": "running", "checkpoint": checkpoint})

    def _handle_replay(self, payload: dict):
        # Refuse while a run is live unless the caller explicitly forces it --
        # a rendered engine window at 1x speed visibly slows a live training
        # run on a fanless machine, so this must be an opt-in, not a default.
        live = [r["run_id"] for r in _list_runs(self.runs_root) if r["live"]]
        if live and not payload.get("force"):
            return self._send_error_json(409, f"training is live ({', '.join(live)}) -- pass force:true to replay anyway")

        kind = payload.get("kind")
        job_id = f"{kind}-{int(time.time()*1000)}"
        log_dir = self.runs_root / "_replays"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        report_path = ""

        if kind == "watch":
            checkpoint = payload.get("checkpoint")
            if not checkpoint:
                return self._send_error_json(400, "watch replay requires 'checkpoint'")
            argv = [sys.executable, str(self.repo_root / "Tools/rl/ppo/watch.py"),
                    "--checkpoint", checkpoint, "--episodes", str(payload.get("episodes", 2))]
            # OGRL-20260817-028 Sec8.1/Sec8.6: watch.py's --from-run reads
            # --level/--frame-stack/--act-period straight from that run's own
            # run.json via run_config.py, instead of this server guessing at
            # (or worse, hardcoding) the checkpoint's actual training config.
            run_id_for_watch = payload.get("run_id")
            if run_id_for_watch:
                run_dir_for_watch = _safe_run_dir(self.runs_root, run_id_for_watch)
                if run_dir_for_watch is not None and run_dir_for_watch.exists():
                    argv += ["--from-run", run_id_for_watch, "--runs-root", str(self.runs_root)]
        elif kind == "ghost":
            csv_path = payload.get("csv")
            if not csv_path:
                return self._send_error_json(400, "ghost replay requires 'csv'")
            argv = [sys.executable, str(self.repo_root / "Tools/rl/replay_ghost.py"), csv_path]
        elif kind == "tape":
            # Replay a recorded tape (tape.py) rendered live in the engine,
            # via the same RLAction::LoadScript scripted-input mechanism
            # replay_ghost.py already uses for watch.py's ghosts -- see this
            # dashboard's static/app.js Tapes tab ("Replay in engine" button).
            run_id_for_tape = payload.get("run_id")
            tape_name = payload.get("tape_name")
            if not run_id_for_tape or not tape_name:
                return self._send_error_json(400, "tape replay requires 'run_id' and 'tape_name'")
            run_dir_for_tape = _safe_run_dir(self.runs_root, run_id_for_tape)
            if run_dir_for_tape is None or not run_dir_for_tape.exists():
                return self._send_error_json(404, f"unknown run_id: {run_id_for_tape}")
            if not _safe_filename_component(tape_name):
                return self._send_error_json(400, "invalid tape name")
            tape_jsonl = run_dir_for_tape / "tapes" / f"{tape_name}.jsonl"
            tape_meta_path = run_dir_for_tape / "tapes" / f"{tape_name}.meta.json"
            if not tape_jsonl.exists():
                return self._send_error_json(404, f"unknown tape: {tape_name}")
            try:
                tape_meta = json.loads(tape_meta_path.read_text()) if tape_meta_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                tape_meta = {}
            manifest_path = run_dir_for_tape / "run.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            level = (manifest.get("env") or {}).get("level") or "arenas/oval_arena_1v1_unarmed.xml"
            sys.path.insert(0, str(self.repo_root / "Tools/rl"))
            from tape import jsonl_to_ghost_csv
            ghost_csv = log_dir / f"{job_id}.csv"
            try:
                n_rows = jsonl_to_ghost_csv(tape_jsonl, ghost_csv)
            except Exception as exc:  # noqa: BLE001 -- report, don't crash the server over one bad tape
                return self._send_error_json(500, f"failed to convert tape to ghost CSV: {exc}")
            if n_rows == 0:
                return self._send_error_json(400, "tape has zero decisions")
            # OGRL-20260817-030: seed + difficulty now actually reproduce the
            # tape's original opponent (see replay_ghost.py's module
            # docstring and RLReplaySeed). opponents/species/weapons aren't
            # stored per-tape (constant for Stage A at the time this was
            # written -- tape.py doesn't record them), so they're pulled from
            # the run's own manifest instead; if a future stage varies them
            # per-episode, tape.py's meta.json needs to start storing them too.
            algo = manifest.get("algo") or {}
            # OGRL-20260817-031: --act-period must match what the tape was
            # actually recorded at or playback speed/length is wrong (see
            # replay_ghost.py's module docstring) -- pulled from the run's
            # own manifest, same source env.py itself trains against.
            act_period = (manifest.get("env") or {}).get("act_period", 1)
            argv = [sys.executable, str(self.repo_root / "Tools/rl/replay_ghost.py"), str(ghost_csv),
                    "--level", level, "--seed", str(tape_meta.get("seed", 1)), "--act-period", str(act_period)]
            if tape_meta.get("difficulty") is not None:
                argv += ["--difficulty", str(tape_meta["difficulty"])]
            if algo.get("opponents") is not None:
                argv += ["--opponents", str(algo["opponents"])]
            if algo.get("weapons_prob") is not None:
                argv += ["--weapons", str(algo["weapons_prob"])]
            if algo.get("species_mode") is not None:
                argv += ["--species", str(algo["species_mode"])]
        elif kind == "native":
            run_id_for_replay = payload.get("run_id")
            replay_name = payload.get("replay_name")
            run_dir = _safe_run_dir(self.runs_root, run_id_for_replay) if run_id_for_replay else None
            replay_path = _resolve_replay_path(run_dir, replay_name) if run_dir and run_dir.exists() else None
            if replay_path is None:
                return self._send_error_json(404, "unknown native replay")
            try:
                replay_reader = ReplayReader(replay_path)
                replay_summary = replay_reader.summary()
                replay_manifest = replay_reader.manifest
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_error_json(422, f"native replay is unreadable: {exc}")
            if not replay_reader.complete or not replay_reader.records("TICK"):
                return self._send_error_json(409, "this episode has no complete native 120 Hz trace")
            if replay_manifest.get("verification") not in ("native_state_trace", "exact_simulation_verified"):
                return self._send_error_json(409, "this episode is recorded-state only and cannot launch an exact engine replay")
            run_manifest = json.loads((run_dir / "run.json").read_text())
            env_manifest = run_manifest.get("env") or {}
            algo = run_manifest.get("algo") or {}
            seed = replay_manifest.get("seed")
            if seed is None:
                return self._send_error_json(409, "native replay has no episode seed")
            report_path = log_dir / f"{job_id}.report.json"
            argv = [sys.executable, str(self.repo_root / "Tools/rl/replay_native.py"),
                    "--repo-root", str(self.repo_root), "--replay", str(replay_path),
                    "--level", str(env_manifest.get("level") or algo.get("level") or "arenas/oval_arena_1v1_unarmed.xml"),
                    "--seed", str(seed), "--report", str(report_path),
                    "--binary-path", str(self.repo_root / "BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth")]
            if replay_manifest.get("difficulty") is not None:
                argv += ["--difficulty", str(replay_manifest["difficulty"])]
            native_reset = replay_manifest.get("native_reset") or {}
            if native_reset.get("reset_mode") is not None:
                argv += ["--reset-mode", str(native_reset["reset_mode"])]
            if replay_manifest.get("native_controlled_character_id") is not None:
                argv += ["--controlled-character-id", str(replay_manifest["native_controlled_character_id"])]
            for key, flag in (("opponents", "--opponents"), ("weapons_prob", "--weapons"), ("species_mode", "--species")):
                if algo.get(key) is not None:
                    argv += [flag, str(algo[key])]
        else:
            return self._send_error_json(400, f"unknown replay kind: {kind!r}")

        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(argv, cwd=self.repo_root, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            return self._send_error_json(500, f"failed to launch replay: {exc}")

        with _replay_lock:
            _replay_jobs[job_id] = {"process": proc, "log_path": str(log_path), "status": "running",
                                    "report_path": str(report_path)}

        def _watch_exit():
            proc.wait()
            with _replay_lock:
                if job_id in _replay_jobs:
                    _replay_jobs[job_id]["status"] = "exited"
        threading.Thread(target=_watch_exit, daemon=True).start()

        return self._send_json({"job_id": job_id, "status": "running"})

    def _serve_static(self, rel_path: str):
        if rel_path == "":
            rel_path = "index.html"
        candidate = (STATIC_DIR / rel_path).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._send_error_json(403, "forbidden")
        if not candidate.is_file():
            return self._send_error_json(404, "not found")
        mime = MIME_TYPES.get(candidate.suffix, "application/octet-stream")
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default=None)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root).resolve() if args.runs_root else repo_root / "Tools/rl/runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    Handler.runs_root = runs_root
    Handler.repo_root = repo_root

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"dashboard serving {runs_root} at {url}")
    print("read-only w.r.t. training except control.json (pause/stop) -- see telemetry.py's module docstring")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
