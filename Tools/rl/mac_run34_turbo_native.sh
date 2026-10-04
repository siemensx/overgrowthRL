#!/usr/bin/env bash
# OGRL-20261004-008 run34_turbo_native: the 260M policy in its NATIVE profile (v5 engine, turbo buttons,
# frozen-camera targeting, line-of-sight enemy listing), trained under the v2 learner/reward/sampling.
#
# Seed: run21_baseline_260m.pt (261,284,572), unmodified. Canonical suite v2 `--controls old`, no
# training: train .90/.65/.42, held .77/.51/.22 -- the best suite-v2 agent on record (2026-10-04, -007).
# Same learner, reward, opponent sampling, entropy floor and 6-map corpus as run33_turbo_mac (which
# runs the same weights transplanted to v6 + see-everyone); n_envs 4 + k 2 = 6 engines = 6 maps.
#
#   nohup bash Tools/rl/mac_run34_turbo_native.sh > Tools/rl/runs/run34_supervisor.log 2>&1 &
#   stop gracefully:  echo '{"command":"stop"}' > Tools/rl/runs/run34_turbo_native/control.json
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
RUN=run34_turbo_native
CKPT="$REPO/Tools/rl/ppo/checkpoints/$RUN.pt"
SEED="$REPO/Tools/rl/ppo/checkpoints/run34_seed.pt"
export OGRL_BINARY="${OGRL_BINARY:-$REPO/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth}"
LOG="$REPO/Tools/rl/runs/$RUN.launches.log"
mkdir -p "$REPO/Tools/rl/runs"
for TRIES in $(seq 1 100); do
  if [ -f "$CKPT" ]; then START=(--resume-from "$CKPT")
  else START=(--resume-from "$SEED" --reset-critic --value-warmup-updates 20); fi
  echo "[$(date)] $RUN launch $TRIES ${START[*]}" >> "$LOG"
  nice -n 10 python3 -u Tools/rl/ppo/train_vec.py \
    --repo-root "$REPO" \
    --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml,arenas/t_train_107.xml,arenas/t_train_109.xml,arenas/t_train_112.xml \
    --shm-prefix "/r34_$TRIES$RANDOM" --n-envs 4 --k-standby 2 --seed 34 --allow-n-envs-change \
    --checkpoint-path "$CKPT" "${START[@]}" --run-id "$RUN" \
    --total-timesteps 2000000000 --n-steps 2560 --n-epochs 2 --minibatch-size 1024 \
    --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 \
    --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 \
    --entropy-coef 0.003 --entropy-target 0.5 --entropy-coef-max 0.01 \
    --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 \
    --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 \
    --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 \
    --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 \
    --collection-torch-threads 1 --update-torch-threads 4 --torch-interop-threads 1 \
    --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 \
    --max-wall-hours 12 \
    --no-tapes --no-native-capture --device cpu \
    --purpose "OGRL-20261004-008 260M policy, native turbo + LOS profile, v2 learner, 6 maps" \
    >> "$REPO/Tools/rl/runs/$RUN.out" 2>&1
  RC=$?
  echo "[$(date)] $RUN exited $RC" >> "$LOG"
  pkill -f "r34_$TRIES" 2>/dev/null
  [ "$RC" = "0" ] && exit 0
  sleep 30
done
