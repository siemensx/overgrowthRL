#!/usr/bin/env bash
# Comprehensive capability snapshot, run on the idle Windows trainer.
#
# Why this exists separately from eval_cadence.sh: cadence answers "is the
# trend moving" with 2 coarse bands and n=30. This answers "what can it
# actually do right now" with a difficulty gradient and a held-out map, at a
# size where a 5-episode impression cannot masquerade as a result. The two
# observation runs on 2026-09-06 (0/5 then 4/5, 1.1M steps apart, with no
# anomaly in entropy/KL/EV between them) are exactly the failure this is for:
# at a ~0.45 per-episode win rate, P(0 wins in 5) is ~5%.
#
# Fixed --seed-base so a rerun is PAIRED with this one: the same scenario
# seeds, the same maps, the same bands. The morning diff is then a policy
# difference, not a scenario difference.
#
# usage: comprehensive_eval.sh [tag] [host]
set -uo pipefail
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL

TAG=${1:-snapshot}
HOST=${2:-trainer-lan}
CKPT=Tools/rl/ppo/checkpoints/run21_mac.pt
BANDS=0.2,0.5,0.8,1.0
EPISODES=40
# One SEEN map and one HELD-OUT map, both with a clean bill of health from
# validate_maps.py. Deliberately not t_held_201/202: regenerated tonight and
# showing 2/12 timeouts, which n=12 cannot separate from noise -- they get a
# bigger validation pass before they are allowed to carry an eval number.
# The old pairing (t_train_103 + t_held_202) is exactly what made the cadence
# numbers uninterpretable: both maps spawned the opponent on a raised deck.
LEVELS="arenas/t_train_101.xml arenas/t_held_203.xml"
OPPONENTS="1 2 3"

STEP=$(python3 -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['global_step'])")
DEST=Tools/rl/runs/run21_mac/eval/comprehensive/${TAG}_step${STEP}
mkdir -p "$DEST"

SNAP=Tools/rl/ppo/checkpoints/eval_snapshots/run21_mac_step${STEP}.pt
mkdir -p "$(dirname "$SNAP")"; cp "$CKPT" "$SNAP"

echo "[$(date +%H:%M:%S)] step=${STEP} tag=${TAG} -> ${DEST}"
scp -q "$SNAP" "$HOST:C:/ogrl/overgrowthRL/Tools/rl/ppo/checkpoints/comprehensive.pt" || { echo "scp failed"; exit 1; }

cat > /tmp/comprehensive.ps1 <<PS
Set-Location C:\ogrl\overgrowthRL
\$py = (Get-Command python -ErrorAction SilentlyContinue); if (-not \$py) { \$py = (Get-Command py) }
\$env:PYTHONUNBUFFERED="1"
Remove-Item C:\ogrl\overgrowthRL\.rl_mo_eval\*.json -ErrorAction SilentlyContinue
& \$py.Source Tools\rl\multi_opponent_eval.py --checkpoint Tools\rl\ppo\checkpoints\comprehensive.pt \`
  --levels ${LEVELS} --opponents ${OPPONENTS} \`
  --episodes ${EPISODES} --difficulty-bands ${BANDS} --max-episode-steps 1200 \`
  --seed-base 8900000 --out C:\ogrl\comprehensive_${TAG}.json 2>&1
PS

bash Tools/rl/winps.sh "$HOST" < /tmp/comprehensive.ps1 | tee "$DEST/console.txt"

# Per-cell JSONs carry the per-band detail the grid summary averages away.
scp -q "$HOST:C:/ogrl/overgrowthRL/.rl_mo_eval/*.json" "$DEST/" 2>/dev/null
scp -q "$HOST:C:/ogrl/comprehensive_${TAG}.json" "$DEST/grid.json" 2>/dev/null
echo "[$(date +%H:%M:%S)] done -> $DEST"
ls -la "$DEST"
