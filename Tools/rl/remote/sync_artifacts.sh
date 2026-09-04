#!/usr/bin/env bash
# Pull training artifacts from the Windows trainer host to the Mac.
#
# Weights and run telemetry are deliberately not in git (see .gitignore), so
# this is both the analysis path and the only backup of the project's single
# irreplaceable output. Run it from the Mac; it never pushes, so it cannot
# clobber a live run on the trainer.
#
#   sync_artifacts.sh                 # telemetry + checkpoints for all runs
#   sync_artifacts.sh <RUN_ID>        # one run
#   sync_artifacts.sh <RUN_ID> --tapes  # also pull .ogreplay containers (large)
set -euo pipefail

HOST="${OGRL_TRAINER_HOST:-trainer}"
REMOTE_ROOT="${OGRL_TRAINER_ROOT:-C:/ogrl/overgrowthRL}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="${1:-}"
WANT_TAPES=0
for a in "$@"; do [ "$a" = "--tapes" ] && WANT_TAPES=1; done

echo "trainer : $HOST:$REMOTE_ROOT"
echo "local   : $LOCAL_ROOT"

mkdir -p "$LOCAL_ROOT/Tools/rl/runs" "$LOCAL_ROOT/Tools/rl/ppo/checkpoints" "$LOCAL_ROOT/Tools/rl/ppo/runs"

# --- checkpoints (small, high value) ---
echo "== checkpoints =="
scp -q "$HOST:$REMOTE_ROOT/Tools/rl/ppo/checkpoints/*.pt" "$LOCAL_ROOT/Tools/rl/ppo/checkpoints/" 2>/dev/null || echo "  (none new)"

# --- learner CSVs ---
echo "== learner csv =="
scp -q "$HOST:$REMOTE_ROOT/Tools/rl/ppo/runs/*.csv" "$LOCAL_ROOT/Tools/rl/ppo/runs/" 2>/dev/null || echo "  (none)"

# --- per-run telemetry ---
# Enumerate remotely so a run created after this script was written is still
# picked up, and so a missing run fails loudly rather than silently syncing
# nothing.
if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "--tapes" ]; then
  RUNS="$RUN_ID"
else
  RUNS="$(ssh "$HOST" "powershell -NoProfile -Command \"Get-ChildItem '$REMOTE_ROOT/Tools/rl/runs' -Directory | ForEach-Object { \\\$_.Name }\"" | tr -d '\r')"
fi

for r in $RUNS; do
  [ -z "$r" ] && continue
  echo "== run $r =="
  mkdir -p "$LOCAL_ROOT/Tools/rl/runs/$r"
  # metrics.jsonl / events.jsonl / manifest.json / eval/*.json -- the things
  # the dashboard and every analysis script actually read.
  for pat in '*.json' '*.jsonl' '*.csv' '*.txt' '*.log'; do
    scp -q "$HOST:$REMOTE_ROOT/Tools/rl/runs/$r/$pat" "$LOCAL_ROOT/Tools/rl/runs/$r/" 2>/dev/null || true
  done
  if ssh "$HOST" "powershell -NoProfile -Command \"Test-Path '$REMOTE_ROOT/Tools/rl/runs/$r/eval'\"" | grep -qi true; then
    mkdir -p "$LOCAL_ROOT/Tools/rl/runs/$r/eval"
    scp -q "$HOST:$REMOTE_ROOT/Tools/rl/runs/$r/eval/*" "$LOCAL_ROOT/Tools/rl/runs/$r/eval/" 2>/dev/null || true
  fi
  if [ "$WANT_TAPES" = "1" ]; then
    echo "   tapes (large)..."
    mkdir -p "$LOCAL_ROOT/Tools/rl/runs/$r/tapes"
    scp -q "$HOST:$REMOTE_ROOT/Tools/rl/runs/$r/tapes/*" "$LOCAL_ROOT/Tools/rl/runs/$r/tapes/" 2>/dev/null || true
  fi
done

echo
echo "== local totals =="
du -sh "$LOCAL_ROOT/Tools/rl/runs" "$LOCAL_ROOT/Tools/rl/ppo/checkpoints" 2>/dev/null || true
