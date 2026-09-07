#!/usr/bin/env bash
# Map the renderer crash band: vary EnvObject count, hold the layout fixed.
#
# The band recorded in DEAD_ENDS.md is 13-18 (OGRL-20260905-067), found by a
# bisect that only ranged up to ~24 objects. On 2026-09-07 regenerated maps at
# 33/39/43 objects segfaulted with the byte-identical signature -- so there is
# more than one band and the recorded one is only the first.
#
# The record's own method note applies: "any bisect harness must prove it
# preserves the failure before its negatives mean anything." This harness runs
# a KNOWN-GOOD control first and refuses to report anything if the control does
# not pass, and detects crashes by watching for a new macOS crash report rather
# than by grepping a log -- a segfaulting engine makes watch.py hang silently.
set -uo pipefail
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL
CKPT=${CKPT:-Tools/rl/ppo/checkpoints/run21_mac.pt}
DR=~/Library/Logs/DiagnosticReports
ARENAS="$HOME/Library/Application Support/Steam/steamapps/common/Overgrowth/Overgrowth.app/Contents/MacOS/Data/Levels/arenas"
BUDGET=${BUDGET:-90}

probe() {
  local lv=$1
  local xml="$ARENAS/$(basename "$lv")"
  [ -f "$xml" ] || { echo "MISSING FILE"; return; }
  local n; n=$(grep -c "<EnvObject " "$xml")
  local before; before=$(ls "$DR"/Overgrowth* 2>/dev/null | wc -l | tr -d ' ')
  local log=/tmp/cb_$(basename "$lv" .xml).log
  ( python3 -u Tools/rl/ppo/watch.py --checkpoint "$CKPT" --level "$lv" \
      --opponents 1 --difficulty 0.6 --frame-stack 4 --act-period 4 \
      --episodes 1 --max-episode-real-seconds 20 > "$log" 2>&1 ) &
  local p=$! w=0
  while kill -0 $p 2>/dev/null && [ $w -lt $BUDGET ]; do sleep 3; w=$((w+3)); done
  kill -9 $p 2>/dev/null; pkill -9 -f "MacOS/Overgrowth" 2>/dev/null; sleep 2
  local after; after=$(ls "$DR"/Overgrowth* 2>/dev/null | wc -l | tr -d ' ')
  LAST_N=$n
  if [ "$after" -gt "$before" ]; then echo "SEGFAULT"
  elif grep -q "episode 0:" "$log"; then echo "ok"
  else echo "hung"; fi
}

CONTROL=${CONTROL:-arenas/t_train_101.xml}
printf 'control %s ... ' "$(basename "$CONTROL")"
res=$(probe "$CONTROL"); echo "$res (objects=$LAST_N)"
if [ "$res" != "ok" ]; then
  echo "ABORT: the control did not pass, so no negative result here means anything."
  exit 1
fi

printf '\n%-16s %-9s %s\n' level objects result
for lv in "$@"; do
  r=$(probe "$lv")
  printf '%-16s %-9s %s\n' "$(basename "$lv" .xml)" "$LAST_N" "$r"
done
