#!/usr/bin/env bash
# Held-out evaluation on a fixed step cadence, appended to one CSV.
#
# Exists because of OGRL-20260906-079/-080/-081: a solo-for-outnumbered trade was
# concluded from TWO checkpoints, escalated with a per-band sweep whose large n
# tightened the error bars on a single POINT without saying anything about
# direction, and then withdrawn when a third point showed the "regression" was a
# transient dip. Two points are not a trend no matter how many episodes sit
# behind each one -- so make three-plus points the default rather than something
# someone has to remember to go and get.
#
# Runs on the idle Windows trainer so it never competes with training, always at
# matched settings (same maps, bands, episode count and the TRAINING episode cap
# -- evaluate.py defaults to 900 against training's 1200).
#
# usage: eval_cadence.sh [every_steps] [host]
set -uo pipefail
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL

EVERY=${1:-10000000}
HOST=${2:-trainer-lan}
CKPT=Tools/rl/ppo/checkpoints/run21_mac.pt
OUT=Tools/rl/runs/run21_mac/eval/cadence.csv
LOG=/tmp/eval_cadence.log
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || echo "step,restarts_last_hour,1v1,1v2,1v3" > "$OUT"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG"; }
step_of() { python3 -c "
import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['global_step'])" 2>/dev/null || echo 0; }

last=$(tail -n +2 "$OUT" | tail -1 | cut -d, -f1); last=${last:-0}
say "cadence started: every ${EVERY} steps, last recorded ${last}"

while true; do
  now=$(step_of)
  if [ "$now" -ge $((last + EVERY)) ]; then
    # Never evaluate a checkpoint sampled minutes after restart churn -- the
    # 130M reading that caused the false regression was taken mid-perturbation,
    # with d_max having just been reset several times.
    restarts=$(python3 - <<'PY'
import json, time
n=0; cut=time.time()-3600
try:
    for l in open('Tools/rl/runs/run21_mac/events.jsonl',errors='replace'):
        try: d=json.loads(l)
        except: continue
        if d.get('kind')=='run_start' and d.get('t',0)>cut: n+=1
except FileNotFoundError: pass
print(n)
PY
)
    if [ "${restarts:-0}" -gt 1 ]; then
      say "step ${now}: ${restarts} restarts in the last hour -- deferring, policy is mid-perturbation"
      sleep 900; continue
    fi
    say "step ${now}: evaluating (restarts_last_hour=${restarts})"
    cp "$CKPT" /tmp/cadence_ckpt.pt
    scp -q /tmp/cadence_ckpt.pt "$HOST:C:/ogrl/overgrowthRL/Tools/rl/ppo/checkpoints/cadence.pt" || { say "scp failed"; sleep 600; continue; }
    cat > /tmp/cadence.ps1 <<'PS'
Set-Location C:\ogrl\overgrowthRL
$py = (Get-Command python -ErrorAction SilentlyContinue); if (-not $py) { $py = (Get-Command py) }
$env:PYTHONUNBUFFERED="1"
& $py.Source Tools\rl\multi_opponent_eval.py --checkpoint Tools\rl\ppo\checkpoints\cadence.pt `
  --levels arenas/t_train_103.xml arenas/t_held_202.xml --opponents 1 2 3 `
  --episodes 30 --difficulty-bands 0.4,0.8 --max-episode-steps 1200 `
  --out C:\ogrl\cadence.json 2>&1
PS
    res=$(bash Tools/rl/winps.sh "$HOST" < /tmp/cadence.ps1 2>&1)
    v1=$(echo "$res" | grep "1 opponent" | grep -oE "[0-9.]+$")
    v2=$(echo "$res" | grep "2 opponent" | grep -oE "[0-9.]+$")
    v3=$(echo "$res" | grep "3 opponent" | grep -oE "[0-9.]+$")
    if [ -n "$v1" ]; then
      echo "${now},${restarts},${v1},${v2},${v3}" >> "$OUT"
      say "step ${now}: 1v1=${v1} 1v2=${v2} 1v3=${v3}"
      last=$now
    else
      say "step ${now}: eval produced no result, retrying next cycle"
    fi
  fi
  sleep 600
done
