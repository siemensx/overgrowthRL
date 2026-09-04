"""OGRL-20260816-022/-023: append-only JSONL telemetry the trainer writes and
the dashboard (Tools/rl/dashboard/) only ever reads -- plus one narrow,
deliberate exception (control.json) for pause/stop, since the user explicitly
asked for a working pause/stop button in the dashboard, not a purely passive
observability tool. That's still not a live IPC channel: control.json is a
small file the trainer polls once per PPO update (already the natural, cheap
check-in point), the same "files, not sockets/shm" philosophy as everything
else here -- a blocking write or a full pipe can stall a socket-based
control channel and hang training; a stale or missing file just means
"no command," which is always safe to no-op on.

Every public method on RunLogger is wrapped so it can never raise into the
training loop -- a dashboard/telemetry bug must not be able to kill a
six-hour run. On first failure it prints once to stderr and goes quiet
(self._disabled), rather than either crashing training or spamming stderr
every subsequent update.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


class RunLogger:
    def __init__(self, runs_root: Path | str, run_id: str, manifest: dict):
        self.run_id = run_id
        self.run_dir = Path(runs_root) / run_id
        self._disabled = False
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "eval").mkdir(parents=True, exist_ok=True)
            self._manifest = dict(manifest)
            self._manifest.setdefault("run_id", run_id)
            self._manifest.setdefault("schema", 1)
            self._manifest.setdefault("started_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            self._manifest.setdefault("ended_at", None)
            self._manifest.setdefault("status", "running")
            _atomic_write_json(self.run_dir / "run.json", self._manifest)
            self._metrics_f = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
            self._episodes_f = open(self.run_dir / "episodes.jsonl", "a", buffering=1)
            self._events_f = open(self.run_dir / "events.jsonl", "a", buffering=1)
            self._control_path = self.run_dir / "control.json"
            if not self._control_path.exists():
                _atomic_write_json(self._control_path, {"command": None, "t": time.time()})
        except Exception as exc:  # noqa: BLE001 -- telemetry must never block training from starting
            print(f"telemetry: disabled at startup: {exc}", file=sys.stderr)
            self._disabled = True

    def _safe(self, fn) -> None:
        if self._disabled:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- see module docstring
            print(f"telemetry: disabling after error: {exc}", file=sys.stderr)
            self._disabled = True

    def log_update(self, record: dict) -> None:
        self._safe(lambda: self._metrics_f.write(json.dumps(record) + "\n"))

    def log_episode(self, record: dict) -> None:
        self._safe(lambda: self._episodes_f.write(json.dumps(record) + "\n"))

    def log_event(self, kind: str, title: str, **kw) -> None:
        record = {"t": time.time(), "kind": kind, "title": title, **kw}
        self._safe(lambda: self._events_f.write(json.dumps(record) + "\n"))

    def log_eval(self, global_step: int, result: dict) -> None:
        def _write():
            path = self.run_dir / "eval" / f"{global_step}.json"
            _atomic_write_json(path, result)
        self._safe(_write)

    def poll_control(self) -> str | None:
        """Returns the current command ('pause' | 'stop') or None. Never
        raises -- a missing/corrupt control file just means no command."""
        if self._disabled:
            return None
        try:
            data = json.loads(self._control_path.read_text())
            return data.get("command")
        except Exception:  # noqa: BLE001 -- a bad read means "no command," not a crash
            return None

    def clear_control(self) -> None:
        """Called after a pause is acknowledged/resumed, so the SAME
        'resume' doesn't get reprocessed on the next poll."""
        self._safe(lambda: _atomic_write_json(self._control_path, {"command": None, "t": time.time()}))

    def finish(self, status: str, final_global_step: int) -> None:
        def _write():
            self._manifest["status"] = status
            self._manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._manifest["final_global_step"] = final_global_step
            _atomic_write_json(self.run_dir / "run.json", self._manifest)
            self._metrics_f.close()
            self._episodes_f.close()
            self._events_f.close()
        self._safe(_write)
