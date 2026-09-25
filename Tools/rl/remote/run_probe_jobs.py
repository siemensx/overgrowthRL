"""Run engine probes on the Windows trainer detached from SSH.

Reads C:\\ogrl\\probe_jobs.txt, one job per line: `<engine exe>|<script>|<arg>|<arg>...`,
runs each with OGRL_BINARY set and engine priority AboveNormal, and appends each
job's JSON output lines (or a FAIL line) to C:\\ogrl\\probe_out.txt, then DONE.

Register it as a Scheduled Task with Normal priority (Priority 4). The Task
Scheduler default is BelowNormal, which gets ZERO CPU while a 24-engine training
run is going (DEAD_ENDS, 2026-09-24).
"""
import os
import subprocess
import sys

OUT = r"C:\ogrl\probe_out.txt"
JOBS = r"C:\ogrl\probe_jobs.txt"
REPO = r"C:\ogrl\overgrowthRL_clean"

jobs = [line.split("|") for line in open(JOBS).read().strip().splitlines() if line.strip()]
with open(OUT, "w") as f:
    f.write("start\n"); f.flush()
    for binary, *args in jobs:
        env = dict(os.environ, OGRL_BINARY=binary, OGRL_ENGINE_PRIORITY="above")
        r = subprocess.run([sys.executable, "-u"] + args, cwd=REPO, env=env, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        f.write(("\n".join(lines) if lines else "FAIL rc=%d %s" % (
            r.returncode, (r.stderr or r.stdout)[-600:].replace("\n", " | "))) + "\n")
        f.flush()
    f.write("DONE\n")
