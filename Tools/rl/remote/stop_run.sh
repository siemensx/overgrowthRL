#!/usr/bin/env bash
# Gracefully stop a training run on the Windows trainer: writes {"command":"stop"} to its
# control.json; the trainer finishes the current update, saves a final checkpoint and exits 0.
#   stop_run.sh <run_id> [trainer_repo_root]     (default root C:\ogrl\overgrowthRL_clean)
set -euo pipefail
RUN="${1:?usage: stop_run.sh <run_id> [root]}"
ROOT="${2:-C:\\ogrl\\overgrowthRL_clean}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat <<PS | "$HERE/winps.sh"
\$p = '$ROOT\\Tools\\rl\\runs\\$RUN\\control.json'
if (-not (Test-Path (Split-Path \$p))) { "no such run dir: " + (Split-Path \$p); exit 1 }
Set-Content -Path \$p -Value '{"command": "stop"}'
"wrote " + \$p + ": " + (Get-Content \$p)
PS
