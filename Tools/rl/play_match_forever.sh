#!/usr/bin/env bash
# Keep a human-versus-checkpoint match window coming back, fast.
#
# play_match.py itself is single-shot, and closing the Overgrowth window does
# NOT make it exit promptly: env.reset()/step() blocks on shm_env.py's
# wait_for_observation(), whose default timeout is OGRL_SHM_WAIT_TIMEOUT=120s
# (shm_env.py:223) -- so "close the window" alone can leave the python
# process sitting there for up to two minutes before it notices and exits.
# That is fine for an unattended trainer (supervise_run.sh already relies on
# exactly this for its own ShmWaitTimeout recovery) but wrong for someone
# sitting at the keyboard who just closed the window wanting another round.
#
# So this watches the ENGINE process directly (its pid is in play_match.py's
# own status.json, written before the round loop starts) and kills the
# python process the moment the engine is gone, instead of waiting on
# play_match.py to notice on its own. Then a fixed 2s pause, then relaunch.
#
# Ctrl+C stops the LOOP (no more relaunches). Closing the game window only
# ends the current MATCH; a new one opens ~2s later.
#
# Usage:
#   Tools/rl/play_match_forever.sh [--checkpoint PATH] [--level ARENAS/X.xml] [extra play_match.py args...]
set -uo pipefail
cd "$(dirname "$0")/../.."

CHECKPOINT="Tools/rl/ppo/checkpoints/run21_win.pt"
LEVEL="arenas/oval_arena_human_duel.xml"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --level) LEVEL="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

STOP=0
trap 'STOP=1; echo; echo "[play_match_forever] stopping (no more relaunches)"' INT TERM

engine_pid_from_status() {
  python3 -c "
import json
try:
    print(json.load(open('$1')).get('engine_pid') or '')
except Exception:
    print('')
" 2>/dev/null
}

pid_alive() {
  # kill -0 on a zombie PID still succeeds -- the process table entry
  # survives until its parent reaps it, and play_match.py's parent process
  # never does (it is blocked in shm_env.py's semaphore wait, not in a
  # waitpid loop). Measured live: the engine defuncts within ~1s of being
  # killed and stays a zombie for the rest of the match, so kill -0 alone
  # made this whole script never fire. Must also check process state.
  [ -n "$1" ] || return 1
  local st
  st=$(ps -o stat= -p "$1" 2>/dev/null)
  [ -n "$st" ] && [ "${st#Z}" = "$st" ]
}

echo "[play_match_forever] checkpoint=$CHECKPOINT level=$LEVEL"
round=0
while [ "$STOP" -eq 0 ]; do
  round=$((round + 1))
  MATCH_ID="match-$(python3 -c 'import time; print(int(time.time()*1000))')"
  MATCH_DIR="Tools/rl/runs/_matches/$MATCH_ID"
  mkdir -p "$MATCH_DIR/session"
  STATUS_PATH="$MATCH_DIR/status.json"
  echo "[play_match_forever] round $round: $MATCH_ID"

  python3 Tools/rl/play_match.py \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-id "$(basename "$CHECKPOINT")" \
    --status-path "$STATUS_PATH" \
    --session-dir "$MATCH_DIR/session" \
    --match-id "$MATCH_ID" \
    --level "$LEVEL" \
    --device cpu \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
  PMPID=$!

  ENGINE_PID=""
  while kill -0 "$PMPID" 2>/dev/null; do
    if [ -z "$ENGINE_PID" ] && [ -f "$STATUS_PATH" ]; then
      ENGINE_PID=$(engine_pid_from_status "$STATUS_PATH")
    fi
    if [ -n "$ENGINE_PID" ] && ! pid_alive "$ENGINE_PID"; then
      echo "[play_match_forever] engine (pid $ENGINE_PID) is gone -- ending this round now, not waiting on ShmWaitTimeout"
      kill -9 "$PMPID" 2>/dev/null
      break
    fi
    [ "$STOP" -eq 1 ] && { kill -TERM "$PMPID" 2>/dev/null; break; }
    sleep 0.3
  done
  wait "$PMPID" 2>/dev/null
  code=$?
  [ "$STOP" -eq 1 ] && break
  echo "[play_match_forever] match ended (code $code) -- relaunching in 2s"
  sleep 2
done
echo "[play_match_forever] stopped"
