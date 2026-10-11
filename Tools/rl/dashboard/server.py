#!/usr/bin/env python3
"""Training dashboard (rewritten 2026-10-07). Runs on the Mac; costs the Windows trainer almost nothing.

    python3 Tools/rl/dashboard/server.py            # then open http://127.0.0.1:8770

What it does:
  * every 60 s, ONE ssh/PowerShell call to the trainer reads only the bytes appended to the active run's
    metrics.jsonl / episodes.jsonl since the last call, plus a few file names and process counts. No process
    runs on the trainer for the dashboard and nothing there is parsed beyond tiny bench JSONs.
  * aggregates on the Mac: win rate per opponent count at full difficulty, per 1M steps; steps/s; restarts.
  * "Watch": copies a checkpoint to the Mac (a periodic snapshot, or a fresh copy of the live checkpoint made
    on the trainer first so the live file is held only for a local copy) and runs ppo/watch.py rendered.
    The Mac's game data lives on the external 1TB drive; without it the button says so instead of crashing.

State is cached in Tools/rl/dashboard/cache/ (gitignored) so a restart does not re-download history.
Binds 127.0.0.1 only. Python standard library only (the Mac's system python3 is 3.9).
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
REPO = RL.parents[1]
CACHE = HERE / "cache"
CKPT_CACHE = CACHE / "checkpoints"
sys.path.insert(0, str(RL))
import paths  # noqa: E402

TRAINER_REPO = r"C:\ogrl\overgrowthRL_v6"
HOSTS = [("trainer-lan", "192.168.99.33"), ("trainer-ts", "100.118.2.91")]
POLL_SECONDS = 60
CHUNK = 3_000_000            # bytes per file per poll; history backfills over a few quick polls
BIN = 1_000_000              # steps per chart point
FULL_D = 0.95                # "full difficulty" for win rates
MAPS_TRAIN = [f"t_train_{i}" for i in range(101, 125)]
MAPS_HELD = ["t_held_203"]
MAPS_NAT = [f"t_nat3_{i}" for i in range(701, 725)]          # run39_terrain onwards
MAPS_NAT_HELD = [f"t_nat3_held_{i}" for i in range(801, 805)]
PERSONA_NAMES = ["normal", "patient", "passive", "berserker", "expert", "mixed"]
COND_ORDER = ["normal fight", "hop-aware enemies", "AI intent hidden"] + [f"{n} opponents" for n in PERSONA_NAMES[1:]]


def condition_of(persona: int, hide: bool, hop: bool) -> str | None:
    """The grid row a training fight belongs to: each row changes ONE thing from the normal fight
    (OGRL-20261010-011). Fights that change two things at once belong to no row."""
    changed = int(persona != 0) + int(hide) + int(hop)
    if changed == 0:
        return "normal fight"
    if changed > 1:
        return None
    if hop:
        return "hop-aware enemies"
    if hide:
        return "AI intent hidden"
    return f"{PERSONA_NAMES[persona]} opponents"
CONTROLS = ["rl_target_select: 2", "rl_button_edges: 1", "rl_no_feint: 1", "rl_obs_omniscient: 1",
            "rl_stick_deadzone: 0.3", "rl_stance_walk: 1"]   # v6-omni, kept for reference; watch uses --controls


# ---------------------------------------------------------------- trainer access

def pick_host() -> str | None:
    forced = os.environ.get("OGRL_TRAINER_HOST")
    if forced:
        return forced
    for name, ip in HOSTS:
        if subprocess.call(["nc", "-z", "-G", "3", ip, "22"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0:
            return name
    return None


def ps(host: str, script: str, timeout: int = 90) -> str:
    r = subprocess.run([str(RL / "winps.sh"), host], input=script, capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout.replace("\r", "")


POLL_PS = r"""
$root = '__REPO__\Tools\rl'
function Tail($p, [long]$off, [long]$cap) {
  if (-not (Test-Path $p)) { return "0|0|" }
  $fs = [IO.File]::Open($p, 'Open', 'Read', 'ReadWrite')
  try {
    $len = $fs.Length; if ($off -gt $len) { $off = 0 }
    $n = [Math]::Min($cap, $len - $off); $buf = New-Object byte[] $n
    [void]$fs.Seek($off, 'Begin'); $got = 0
    while ($got -lt $n) { $k = $fs.Read($buf, $got, $n - $got); if ($k -le 0) { break }; $got += $k }
    return "$len|$($off + $got)|" + [Convert]::ToBase64String($buf, 0, $got)
  } finally { $fs.Close() }
}
$a = Get-Content C:\ogrl\active_profile.json -Raw -ErrorAction SilentlyContinue
"@@ACTIVE " + ($a -replace "`r?`n", ' ')
"@@RUNS " + ((Get-ChildItem "$root\runs" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 12 | % Name) -join ',')
$run = '__RUN__'
if ($run -eq '') { if ($a) { $run = ($a | ConvertFrom-Json).run_id } }
"@@RUN " + $run
if ($run -ne '') {
  "@@METRICS " + (Tail "$root\runs\$run\metrics.jsonl" __OFF_M__ __CAP__)
  "@@EPISODES " + (Tail "$root\runs\$run\episodes.jsonl" __OFF_E__ __CAP__)
  $known = '__BENCHES__'.Split(',')
  Get-ChildItem "$root\runs\$run\eval\periodic_*_greedy.json" -ErrorAction SilentlyContinue | % {
    if ($known -notcontains $_.Name) {
      $j = Get-Content $_.FullName -Raw | ConvertFrom-Json; $o = $j.bands[0].policy.outcomes
      "@@BENCH " + $_.Name + " " + $j.global_step + " " + $o.won + " " + $o.lost + " " + $o.timeout + " " + $j.level
    } }
  "@@SNAPS " + ((Get-ChildItem "$root\ppo\checkpoints\snapshots\$($run)_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | % { $_.Name + ':' + $_.Length }) -join ',')
  "@@LOG " + ((Get-Content "C:\ogrl\$run.log" -Tail 6 -ErrorAction SilentlyContinue) -join ' || ')
}
"@@ENGINES " + @(Get-Process Overgrowth -ErrorAction SilentlyContinue).Count
"@@TRAINER " + @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train_vec.py*' }).Count
"@@END"
"""


# ---------------------------------------------------------------- per-run aggregate state

STATE_VERSION = 2                # 2: cond_bins (per-condition history for the expandable graphs)


def new_run_state(run: str) -> dict:
    return {"run": run, "off_m": 0, "off_e": 0, "rest_m": "", "rest_e": "", "size_m": 0, "size_e": 0,
            "bins": {},            # str(bin) -> {"1": [won, n, timeouts], ...} full difficulty only
            "bins_low": 0,         # episodes below full difficulty (reported, not charted)
            "recent": [],          # last 3000 full-difficulty episodes: [step, opp, outcome]
            "metrics": [],         # thinned: [step, t, sps, entropy, summary_sat, summary_gp]
            "last_metric": None, "benches": {}, "first_step": None, "last_episode_t": None,
            "cond_bins": {},       # condition -> str(bin) -> {"1": [won, n, timeouts], ...} full difficulty only
            "version": STATE_VERSION}


def state_path(run: str) -> Path:
    return CACHE / f"run_{re.sub(r'[^A-Za-z0-9_.-]', '_', run)}.json"


def load_run_state(run: str) -> dict:
    p = state_path(run)
    if p.exists():
        try:
            st = json.loads(p.read_text())
            if st.get("version") == STATE_VERSION:
                return st
            if st.get("benches"):                 # keep finished quick checks; the episode history re-reads
                fresh = new_run_state(run)
                fresh["benches"] = st["benches"]
                return fresh
        except Exception:
            pass
    return new_run_state(run)


def save_run_state(st: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = state_path(st["run"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, state_path(st["run"]))


def ingest_episodes(st: dict, text: str) -> None:
    data = st["rest_e"] + text
    lines = data.split("\n")
    st["rest_e"] = lines.pop()                       # partial last line waits for the next poll
    for line in lines:
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        step = int(e.get("global_step", 0))
        if st["first_step"] is None:
            st["first_step"] = step
        st["last_episode_t"] = e.get("t", st["last_episode_t"])
        if float(e.get("d", 0) or 0) < FULL_D:
            st["bins_low"] += 1
            continue
        opp = str(int(e.get("opponents", 1) or 1))
        b = st["bins"].setdefault(str(step // BIN), {})
        cell = b.setdefault(opp, [0, 0, 0])
        cell[1] += 1
        if e.get("outcome") == "won":
            cell[0] += 1
        elif e.get("outcome") == "timeout":
            cell[2] += 1
        tagged = "persona" in e or "hide_intent" in e or "hop_aware" in e
        persona, hide, hop = int(e.get("persona", 0) or 0), bool(e.get("hide_intent")), bool(e.get("hop_aware"))
        st["recent"].append([step, int(opp), e.get("outcome"), persona, hide, hop, tagged])
        cond = condition_of(persona, hide, hop) if tagged else None
        if cond:
            cc = st["cond_bins"].setdefault(cond, {}).setdefault(str(step // BIN), {}).setdefault(opp, [0, 0, 0])
            cc[1] += 1
            cc[0] += e.get("outcome") == "won"
            cc[2] += e.get("outcome") == "timeout"
    st["recent"] = st["recent"][-6000:]


def ingest_metrics(st: dict, text: str) -> None:
    data = st["rest_m"] + text
    lines = data.split("\n")
    st["rest_m"] = lines.pop()
    for line in lines:
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        if "global_step" not in m:
            continue
        perf = m.get("perf") or {}
        nh = (m.get("net_health") or {}).get("summary") or {}
        row = [m["global_step"], m.get("t"), perf.get("steps_per_second_cycle"),
               (m.get("ppo") or {}).get("entropy"), nh.get("saturated"), nh.get("grad_pass")]
        st["last_metric"] = row
        if m.get("personas"):
            st["mix"] = m["personas"]
        if not st["metrics"] or row[0] - st["metrics"][-1][0] >= 200_000:
            st["metrics"].append(row)


# ---------------------------------------------------------------- lineage (earlier runs of the same line)

CONDITION_FLAGS = ("--persona-mix", "--hop-aware-prob", "--hide-intent-prob")


def lineage(run: str, depth: int = 6) -> list[str]:
    """Earlier runs this one was forked from (profile seed_from chain), newest first, stopping at the first run
    that trained without the condition mix (its fights carry no condition flags). A profile switch starts a new
    run, so without this every switch would start the history graphs from nothing."""
    out = []
    cur = run
    for _ in range(depth):
        f = RL / "profiles" / f"{cur}.json"
        try:
            prof = json.loads(f.read_text())
        except Exception:
            break
        parent = (prof.get("seed_from") or "").removesuffix(".pt") if hasattr(str, "removesuffix") else \
            re.sub(r"\.pt$", "", prof.get("seed_from") or "")
        if not parent:
            break
        try:
            pp = json.loads((RL / "profiles" / f"{parent}.json").read_text())
        except Exception:
            break
        if not any(a in pp.get("args", []) for a in CONDITION_FLAGS):
            break
        out.append(parent)
        cur = parent
    return out


# ---------------------------------------------------------------- poller

def parse_sections(out: str) -> tuple[dict, list]:
    sections, benches = {}, []
    for line in out.split("\n"):
        if line.startswith("@@BENCH "):
            benches.append(line[8:].split(" "))
        elif line.startswith("@@"):
            k, _, v = line[2:].partition(" ")
            sections[k] = v.strip()
    return sections, benches


class Poller(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.view_run = ""                 # "" = follow the active run
        self.runs: dict[str, dict] = {}
        self.info = {"host": None, "connected": False, "active": None, "runs": [], "engines": None,
                     "trainer_procs": None, "snaps": [], "log": [], "last_poll": None, "error": None,
                     "run": None}
        self.wake = threading.Event()

    def run(self):
        while True:
            backfilling = False
            try:
                backfilling = self.poll_once()
            except Exception as exc:
                with self.lock:
                    self.info["error"] = repr(exc)[:300]
                    self.info["connected"] = False
            self.wake.wait(3 if backfilling else POLL_SECONDS)
            self.wake.clear()

    def poll_once(self) -> bool:
        host = self.info["host"] or pick_host()
        if host is None:
            with self.lock:
                self.info.update(connected=False, error="trainer unreachable on LAN and Tailscale")
            return False
        follow = not self.view_run
        assumed = self.view_run or self.info.get("run")
        st = None
        if assumed:
            st = self.runs.get(assumed) or load_run_state(assumed)
            self.runs[assumed] = st
        script = (POLL_PS.replace("__REPO__", TRAINER_REPO).replace("__RUN__", "" if follow else self.view_run)
                  .replace("__OFF_M__", str(st["off_m"] if st else 0))
                  .replace("__OFF_E__", str(st["off_e"] if st else 0))
                  .replace("__CAP__", str(CHUNK if st else 0))
                  .replace("__BENCHES__", ",".join(st["benches"]) if st else ""))
        out = ps(host, script)
        if "@@END" not in out:
            self.info["host"] = None          # re-pick next time (LAN vs Tailscale may have changed)
            raise RuntimeError("no answer from trainer: " + out.strip()[-200:])
        sections, benches = parse_sections(out)
        actual = sections.get("RUN", "")
        if not actual:
            with self.lock:
                self.info.update(host=host, connected=True, error="no active run on the trainer")
            return False
        if st is None or actual != st["run"]:
            # the offsets sent were for another run (first contact, or the trainer switched profile):
            # adopt the actual run and poll again straight away with its own cached offsets
            with self.lock:
                self.info.update(host=host, connected=True, run=actual)
            self.runs[actual] = self.runs.get(actual) or load_run_state(actual)
            return True
        self.runs[actual] = st
        more = self._ingest(st, actual, sections, benches)
        if not more:
            more = self.backfill_lineage(host, actual)
        snaps = []
        for item in filter(None, sections.get("SNAPS", "").split(",")):
            name, _, size = item.rpartition(":")
            m = re.search(r"_(\d+)\.pt$", name)
            snaps.append({"name": name, "size": int(size), "step": int(m.group(1)) if m else None})
        try:
            active = json.loads(sections.get("ACTIVE") or "null")
        except Exception:
            active = None
        with self.lock:
            self.info.update(host=host, connected=True, error=None, active=active, run=actual,
                             runs=[r for r in sections.get("RUNS", "").split(",") if r],
                             engines=int(sections.get("ENGINES") or 0),
                             trainer_procs=int(sections.get("TRAINER") or 0),
                             snaps=snaps, log=[x for x in sections.get("LOG", "").split(" || ") if x],
                             last_poll=time.time())
        return more

    def backfill_lineage(self, host: str, run: str) -> bool:
        """Read one more chunk of the oldest unfinished ancestor run; True while any remains."""
        for anc in lineage(run):
            st = self.runs.get(anc) or load_run_state(anc)
            self.runs[anc] = st
            if st.get("done_e") and st["off_e"] >= st["size_e"] and st["off_m"] >= st["size_m"]:
                continue
            script = (POLL_PS.replace("__REPO__", TRAINER_REPO).replace("__RUN__", anc)
                      .replace("__OFF_M__", str(st["off_m"])).replace("__OFF_E__", str(st["off_e"]))
                      .replace("__CAP__", str(CHUNK)).replace("__BENCHES__", ",".join(st["benches"])))
            out = ps(host, script)
            if "@@END" not in out:
                return False
            sections, benches = parse_sections(out)
            if sections.get("RUN", "") != anc:
                return False
            more = self._ingest(st, anc, sections, benches)
            st["done_e"] = not more
            save_run_state(st)
            return True
        return False

    def _ingest(self, st: dict, actual: str, sections: dict, benches: list) -> bool:
        more = False
        for key, off, ingest in (("METRICS", "off_m", ingest_metrics), ("EPISODES", "off_e", ingest_episodes)):
            v = sections.get(key, "0|0|")
            size, new_off, b64 = v.split("|", 2)
            size, new_off = int(size), int(new_off)
            if new_off < st[off]:             # file was replaced/truncated: start over
                fresh = new_run_state(actual)
                fresh["benches"] = st["benches"]
                st.update(fresh)
            if b64:
                ingest(st, base64.b64decode(b64).decode("utf-8", "replace"))
            st[off] = new_off
            st["size_m" if key == "METRICS" else "size_e"] = size
            more = more or new_off < size
        for name, step, won, lost, to, level in benches:
            st["benches"][name] = [int(step), int(won), int(lost), int(to), level]
        save_run_state(st)
        return more

    def snapshot(self) -> dict:
        with self.lock:
            info = dict(self.info)
        run = self.view_run or info.get("run")
        st = self.runs.get(run) if run else None
        anc = [self.runs[a] for a in (lineage(run) if run else []) if a in self.runs]
        return summarize(info, st, anc)


def merge_bins(states: list[dict], key: str) -> dict:
    """Sum per-1M-step bins over runs of one line (the step counter continues across a fork)."""
    out: dict = {}
    for st in states:
        for b, cells in st.get(key, {}).items():
            dst = out.setdefault(b, {})
            for o, c in cells.items():
                d = dst.setdefault(o, [0, 0, 0])
                for i in range(3):
                    d[i] += c[i]
    return out


def summarize(info: dict, st: dict | None, ancestors: list | None = None) -> dict:
    out = {"trainer": {k: info.get(k) for k in ("host", "connected", "engines", "trainer_procs", "error",
                                                 "last_poll", "runs", "log")},
           "active": info.get("active"), "snaps": info.get("snaps", []), "run": None}
    if not st:
        return out
    lm = st.get("last_metric")
    recent = st["recent"]
    last_step = lm[0] if lm else None
    window = [r for r in recent if last_step and r[0] >= last_step - BIN] or recent[-600:]
    rates = {}
    for opp in (1, 2, 3):
        rows = [r for r in window if r[1] == opp]
        if rows:
            rates[str(opp)] = {"won": sum(r[2] == "won" for r in rows), "n": len(rows),
                               "timeouts": sum(r[2] == "timeout" for r in rows)}
    line_states = list(reversed(ancestors or [])) + [st]          # oldest first
    bins_all = merge_bins(line_states, "bins")
    boundaries = [[x["first_step"], x["run"]] for x in line_states if x.get("first_step") is not None]
    chart = []
    for b in sorted(bins_all, key=int):
        cells = bins_all[b]
        chart.append({"step": (int(b) + 1) * BIN,
                      **{o: (round(c[0] / c[1], 3) if c[1] >= 30 else None) for o, c in cells.items()}})
    if chart and last_step and chart[-1]["step"] > last_step:
        chart[-1]["partial"] = True                       # the current bin is still filling
    restarts = [l for l in info.get("log", []) if "exited" in l]
    # OGRL-20261010-011: win rate per opponent count for each condition, changing ONE thing from the normal fight
    # (normal persona, intent shown, normal hop reaction) -- the pooled persona table mixed them all together.
    tagged = [r for r in recent if len(r) >= 7 and r[6]]
    grid = []
    if tagged:
        for name in COND_ORDER:
            row = {"condition": name}
            for opp in (1, 2, 3):
                sub = [r for r in tagged if r[1] == opp and condition_of(r[3], r[4], r[5]) == name]
                row[str(opp)] = {"won": sum(r[2] == "won" for r in sub), "n": len(sub),
                                 "timeouts": sum(r[2] == "timeout" for r in sub)}
            # history for the expandable graph: [bin_end_step, {opp: [won, n, timeouts]}] per 1M steps
            hist = merge_bins([{"b": x.get("cond_bins", {}).get(name, {})} for x in line_states], "b")
            row["history"] = [[(int(b) + 1) * BIN, hist[b]] for b in sorted(hist, key=int)]
            grid.append(row)
    out["run"] = {"run": st["run"], "step": last_step, "metric_t": lm[1] if lm else None,
                  "sps": lm[2] if lm else None, "entropy": lm[3] if lm else None,
                  "summary_sat": lm[4] if lm else None, "summary_gp": lm[5] if lm else None,
                  "rates": rates, "window_steps": BIN, "chart": chart,
                  "low_difficulty_episodes": st["bins_low"],
                  "benches": sorted(st["benches"].values()),
                  "backfill": {"metrics": [st["off_m"], st["size_m"]], "episodes": [st["off_e"], st["size_e"]]},
                  "restarts": restarts, "mix": st.get("mix"), "mix_grid": grid, "boundaries": boundaries}
    return out


# ---------------------------------------------------------------- watching a checkpoint on the Mac

class Watcher:
    def __init__(self, poller: Poller):
        self.poller = poller
        self.proc = None
        self.status = {"state": "idle", "message": "", "checkpoint": None, "results": [], "log_tail": []}
        self.lock = threading.Lock()
        self.log_path = CACHE / "watch.log"

    @staticmethod
    def drive_ok() -> tuple[bool, str]:
        data = paths.data_dir()
        if not (data / "Sounds" / "voice" / "phonemes.txt").exists():
            return False, ("Game data not found -- the external 1TB drive is not connected "
                           f"({data} points to it). Plug it in to watch fights on this Mac.")
        return True, ""

    def set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def fetch(self, host: str, run: str, which: str) -> Path:
        CKPT_CACHE.mkdir(parents=True, exist_ok=True)
        if which == "live":
            remote_tmp = r"C:\ogrl\dashboard_live_copy.pt"
            ps(host, f"Copy-Item -Force '{TRAINER_REPO}\\Tools\\rl\\ppo\\checkpoints\\{run}.pt' '{remote_tmp}'; 'ok'")
            local = CKPT_CACHE / f"{run}_live_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            src = "C:/ogrl/dashboard_live_copy.pt"
        else:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.pt", which):
                raise ValueError("bad checkpoint name")
            local = CKPT_CACHE / which
            if local.exists():
                return local
            src = f"{TRAINER_REPO}/Tools/rl/ppo/checkpoints/snapshots/{which}".replace("\\", "/")
        tmp = local.with_suffix(".part")
        r = subprocess.run(["scp", "-q", f"{host}:{src}", str(tmp)], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError("copy failed: " + r.stderr.strip()[-200:])
        os.replace(tmp, local)
        return local

    def start(self, req: dict) -> dict:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "a watch is already running"}
        if "BENCH" in globals() and BENCH.busy():
            return {"ok": False, "error": "a benchmark is running on this Mac; wait for it to finish"}
        ok, why = self.drive_ok()
        if not ok:
            return {"ok": False, "error": why}
        host = self.poller.info.get("host") or pick_host()
        run = self.poller.view_run or self.poller.info.get("run")
        level = req.get("map", "t_nat3_held_801")
        if level not in MAPS_TRAIN + MAPS_HELD + MAPS_NAT + MAPS_NAT_HELD:
            return {"ok": False, "error": "unknown map"}
        opp = int(req.get("opponents", 3))
        eps = max(1, min(20, int(req.get("episodes", 5))))
        which = req.get("checkpoint", "live")
        persona = str(req.get("persona", "stock"))
        if persona not in ("stock", "patient", "passive", "berserker", "expert", "mixed"):
            return {"ok": False, "error": "unknown enemy type"}
        setup = {"persona": persona, "hop_aware": bool(req.get("hop_aware")), "hide_intent": bool(req.get("hide_intent"))}
        threading.Thread(target=self._run, args=(host, run, which, level, opp, eps, bool(req.get("sampled")), setup),
                         daemon=True).start()
        return {"ok": True}

    def _run(self, host, run, which, level, opp, eps, sampled, setup=None):
        setup = setup or {}
        try:
            self.set(state="copying", message=f"copying {'the live checkpoint' if which == 'live' else which}",
                     results=[], log_tail=[], checkpoint=which)
            ck = self.fetch(host, run, which)
            cmd = [sys.executable, "-u", str(RL / "ppo" / "watch.py"), "--checkpoint", str(ck), "--controls",
                   "v6-omni", "--level", f"arenas/{level}.xml", "--opponents", str(opp), "--difficulty", "1.0",
                   "--episodes", str(eps), "--auto-camera", "--no-ghost"] + (["--sampled"] if sampled else [])
            cmd += ["--persona", setup.get("persona", "stock")]
            cmd += ["--hop-aware"] if setup.get("hop_aware") else []
            cmd += ["--hide-intent"] if setup.get("hide_intent") else []
            what = [{"stock": "normal"}.get(setup.get("persona", "stock"), setup.get("persona")) + " enemies"]
            what += ["hop-aware"] if setup.get("hop_aware") else []
            what += ["AI intent hidden"] if setup.get("hide_intent") else []
            self.set(state="running", message=f"{ck.name} on {level}, 1v{opp}, {', '.join(what)}, {eps} fights",
                     checkpoint=ck.name)
            with open(self.log_path, "w") as log:
                self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
                while self.proc.poll() is None:
                    time.sleep(1)
                    self._read_log()
            self._read_log()
            rc = self.proc.returncode
            self.set(state="done" if rc == 0 else "failed",
                     message="finished" if rc == 0 else f"watch.py exited {rc} (see last lines)")
        except Exception as exc:
            self.set(state="failed", message=repr(exc)[:300])

    def _read_log(self):
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except Exception:
            return
        res = []
        for l in lines:
            m = re.match(r"episode (\d+): steps=(\d+) real_seconds=([\d.]+).*(WON|LOST|timed out)$", l.strip())
            if m:
                res.append({"episode": int(m.group(1)), "steps": int(m.group(2)), "outcome": m.group(4)})
        self.set(results=res, log_tail=lines[-6:])

    def stop(self) -> dict:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            return {"ok": True}
        return {"ok": False, "error": "nothing running"}

    def snapshot(self) -> dict:
        ok, why = self.drive_ok()
        with self.lock:
            return {**self.status, "drive_ok": ok, "drive_message": why,
                    "maps": {"nat_held": MAPS_NAT_HELD, "nat": MAPS_NAT, "held": MAPS_HELD, "train": MAPS_TRAIN}}


# ---------------------------------------------------------------- benchmark on the Mac

BENCH_FILE = CACHE / "benchmarks.json"
UNSEEN = ["t_horde_301", "t_horde_302", "t_horde_303", "t_horde_held_401"]


class Benchmarker:
    """One click: suite-v2 1v3 cells (3 training maps + t_held_203) plus four maps no run trains on, 100 greedy
    fights each = 800 fights, on the Mac, fixed seeds (comparable across checkpoints and with OGRL-20261010-001).
    ~20-25 min at 8 engines. Results are kept in cache/benchmarks.json."""

    def __init__(self, watcher: "Watcher"):
        self.watcher = watcher
        self.lock = threading.Lock()
        self.status = {"state": "idle", "message": "", "done": 0, "total": 0}

    def results(self) -> list:
        try:
            return json.loads(BENCH_FILE.read_text())
        except Exception:
            return []

    def busy(self) -> bool:
        return self.status["state"] in ("copying", "running")

    def start(self, req: dict) -> dict:
        if self.busy() or self.watcher.status["state"] in ("copying", "running"):
            return {"ok": False, "error": "a benchmark or a watch is already running on this Mac"}
        ok, why = self.watcher.drive_ok()
        if not ok:
            return {"ok": False, "error": why}
        threading.Thread(target=self._run, args=(req.get("checkpoint", "live"),), daemon=True).start()
        return {"ok": True}

    def _run(self, which: str):
        try:
            import canonical_eval as ce
            host = self.watcher.poller.info.get("host") or pick_host()
            run = self.watcher.poller.view_run or self.watcher.poller.info.get("run")
            with self.lock:
                self.status.update(state="copying", message=f"copying {which}", done=0, total=0)
            ck = self.watcher.fetch(host, run, which)
            cells = [c for c in ce.cells() if c["opp"] == 3] + [
                {"idx": 100 + i, "map": m, "opp": 3, "split": "unseen", "seed_base": 7_500_000 + 1000 * i}
                for i, m in enumerate(UNSEEN)]
            out = CACHE / "bench" / ck.stem
            out.mkdir(parents=True, exist_ok=True)
            total = len(cells) * (100 // ce.CHUNK)
            with self.lock:
                self.status.update(state="running", message=f"{ck.name}: 800 fights", total=total)
            done_box = {"stop": False}

            def progress():
                while not done_box["stop"]:
                    n = len(list(out.glob("*_s0*.json")))
                    with self.lock:
                        self.status["done"] = n
                    time.sleep(5)
            threading.Thread(target=progress, daemon=True).start()
            ce.run_all_chunks(str(ck), cells, ce.CONTROLS["v6-omni"], out, 100, ck.stem, 8)
            res = [ce.run_cell(str(ck), c, ce.CONTROLS["v6-omni"], out, 100, ck.stem) for c in cells]
            done_box["stop"] = True
            import torch
            step = int(torch.load(str(ck), map_location="cpu", weights_only=False).get("global_step", -1))
            entry = {"checkpoint": ck.name, "step": step, "time": time.strftime("%Y-%m-%d %H:%M"),
                     "train": sum(r["won"] for r in res if r["split"] == "train"),
                     "held": sum(r["won"] for r in res if r["split"] == "held"),
                     "unseen": sum(r["won"] for r in res if r["split"] == "unseen"),
                     "timeouts": sum(r["outcomes"]["timeout"] for r in res)}
            allres = [r for r in self.results() if r.get("checkpoint") != ck.name] + [entry]
            BENCH_FILE.write_text(json.dumps(sorted(allres, key=lambda r: r["step"]), indent=1))
            with self.lock:
                self.status.update(state="done", message=f"{ck.name}: done", done=total)
        except Exception as exc:
            with self.lock:
                self.status.update(state="failed", message=repr(exc)[:300])

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.status, "results": self.results()}


# ---------------------------------------------------------------- switching the trainer's profile

class Switcher:
    """Runs remote/switch_profile.sh <profile> (graceful stop, code update, maps/script, preflight, start)."""

    def __init__(self, poller: Poller):
        self.poller = poller
        self.proc = None
        self.log_path = CACHE / "switch.log"
        self.target = None

    @staticmethod
    def profiles() -> list:
        out = []
        for f in sorted((RL / "profiles").glob("run*.json")):
            try:
                d = json.loads(f.read_text())
                out.append({"name": f.stem, "purpose": d.get("purpose", "")})
            except Exception:
                pass
        return out

    def start(self, req: dict) -> dict:
        if self.proc and self.proc.poll() is None:
            return {"ok": False, "error": "a switch is already running"}
        name = str(req.get("profile", ""))
        if name not in [p["name"] for p in self.profiles()]:
            return {"ok": False, "error": "unknown profile"}
        env = dict(os.environ)
        env["HOST"] = self.poller.info.get("host") or pick_host() or "trainer-lan"
        self.target = name
        self.proc = subprocess.Popen([str(RL / "remote" / "switch_profile.sh"), name], cwd=str(REPO),
                                     stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env)
        return {"ok": True}

    def snapshot(self) -> dict:
        state = "idle"
        if self.proc is not None:
            rc = self.proc.poll()
            state = "running" if rc is None else ("done" if rc == 0 else f"failed (exit {rc})")
        try:
            tail = self.log_path.read_text(errors="replace").splitlines()[-8:] if self.proc else []
        except Exception:
            tail = []
        return {"state": state, "target": self.target, "log_tail": tail, "profiles": self.profiles()}


# ---------------------------------------------------------------- http

POLLER = Poller()
WATCHER = Watcher(POLLER)
BENCH = Benchmarker(WATCHER)
SWITCH = Switcher(POLLER)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json({**POLLER.snapshot(), "watch": WATCHER.snapshot(), "now": time.time(),
                        "view_run": POLLER.view_run, "bench": BENCH.snapshot(), "switch": SWITCH.snapshot()})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/watch":
            self._json(WATCHER.start(body))
        elif self.path == "/api/bench":
            self._json(BENCH.start(body))
        elif self.path == "/api/switch":
            self._json(SWITCH.start(body))
        elif self.path == "/api/watch/stop":
            self._json(WATCHER.stop())
        elif self.path == "/api/run":
            r = str(body.get("run", ""))
            if r and not re.fullmatch(r"[A-Za-z0-9_.-]+", r):
                return self._json({"ok": False, "error": "bad run name"})
            POLLER.view_run = r
            POLLER.wake.set()
            self._json({"ok": True})
        elif self.path == "/api/refresh":
            POLLER.wake.set()
            self._json({"ok": True})
        else:
            self._send(404, b"not found", "text/plain")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    POLLER.start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"dashboard on http://127.0.0.1:{a.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
