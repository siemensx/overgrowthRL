#!/usr/bin/env bash
# Render smoke test: does this level survive being DRAWN?
#
# Headless training never calls Engine::DrawScene, so a level can train for
# hours and kill the window the moment a human opens it -- that is how the
# renderer crash band (13-18 EnvObjects) was found, and how a regenerated map
# shipped on the stock multi-round arena script without anyone noticing until
# it was watched. Neither is visible to validate_maps.py.
#
# usage: render_smoke.sh <checkpoint> <level> [level...]
set -uo pipefail
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL
CKPT=$1; shift
fail=0
for lv in "$@"; do
  log=/tmp/rsmoke_$(basename "$lv" .xml).log
  timeout_s=${RSMOKE_TIMEOUT:-150}
  # -u: without it Python block-buffers stdout when redirected, so killing a
  # hung run discards every line it had produced and the log is empty.
  ( python3 -u Tools/rl/ppo/watch.py --checkpoint "$CKPT" --level "$lv" \
      --opponents 3 --difficulty 0.6 --frame-stack 4 --act-period 4 \
      --episodes 1 --max-episode-real-seconds 25 > "$log" 2>&1 ) &
  pid=$!
  waited=0
  while kill -0 $pid 2>/dev/null && [ $waited -lt $timeout_s ]; do sleep 2; waited=$((waited+2)); done
  kill -9 $pid 2>/dev/null
  # Only this probe's engine -- see crash_band_sweep.sh; a bare
  # "MacOS/Overgrowth" pattern takes the training workers with it.
  pkill -9 -f "write-dir.*env-ogrl_w" 2>/dev/null; sleep 2
  if grep -qiE "segmentation|EXC_BAD_ACCESS|Assertion|GenerateStacktrace|Traceback" "$log"; then
    echo "  CRASH  $(basename "$lv")   -- see $log"; fail=1
  elif grep -q "episode 0:" "$log"; then
    echo "  ok     $(basename "$lv")   $(grep -m1 'episode 0:' "$log")"
  else
    echo "  NO RUN $(basename "$lv")   -- see $log"; fail=1
  fi
done
exit $fail
