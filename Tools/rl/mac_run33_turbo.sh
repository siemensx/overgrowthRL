#!/usr/bin/env bash
# OGRL-20261004-006 run33_turbo_mac: the turbo arm, on the Mac (the trainer holds run32).
#
# Seed: run21_baseline_260m transplanted into v6 by surgery_v5_to_v6.py (exact, --check 1.4e-5).
# Canonical suite v2 with v6-omni-old (turbo buttons + frozen-camera targeting + see-everyone), no
# training: train .89/.67/.41, held .69/.25/.13 -- the best training-map 1v3 the suite has recorded
# (run31 @378M under corrected controls: .25). Turbo is legal under the 2026-10-02 fairness decision.
# Trained under run32's v2 learner/reward/sampling so the two arms differ in controls, seed policy and
# map count (run32: 12 maps on the trainer; here 6, because n_envs 4 + k 2 = 6 engines must equal a
# multiple of the corpus -- AGENTS invariant 6).
#
#   nohup bash Tools/rl/mac_run33_turbo.sh > /tmp/run33.log 2>&1 &
#   stop gracefully:  echo '{"command":"stop"}' > Tools/rl/runs/run33_turbo_mac/control.json
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
RUN=run33_turbo_mac
CKPT="$REPO/Tools/rl/ppo/checkpoints/$RUN.pt"
SEED="$REPO/Tools/rl/ppo/checkpoints/run33_seed.pt"
export OGRL_BINARY="${OGRL_BINARY:-$REPO/../badbunny-gru/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth}"
LOG="$REPO/Tools/rl/runs/$RUN.launches.log"
mkdir -p "$REPO/Tools/rl/runs"
for TRIES in $(seq 1 100); do
  if [ -f "$CKPT" ]; then START=(--resume-from "$CKPT")
  else START=(--resume-from "$SEED" --reset-critic --value-warmup-updates 20); fi
  echo "[$(date)] $RUN launch $TRIES ${START[*]}" >> "$LOG"
  nice -n 10 python3 -u Tools/rl/ppo/train_vec.py \
    --repo-root "$REPO" \
    --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml,arenas/t_train_107.xml,arenas/t_train_109.xml,arenas/t_train_112.xml \
    --shm-prefix "/r33_$TRIES$RANDOM" --n-envs 4 --k-standby 2 --seed 33 --allow-n-envs-change \
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
    --engine-config-line "rl_obs_omniscient: 1" \
    --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 \
    --max-wall-hours 12 \
    --no-tapes --no-native-capture --device cpu \
    --purpose "OGRL-20261004-006 260M turbo-era policy -> v6, turbo + see-everyone, v2 learner, 6 maps" \
    >> "$REPO/Tools/rl/runs/$RUN.out" 2>&1
  RC=$?
  echo "[$(date)] $RUN exited $RC" >> "$LOG"
  pkill -f "r33_$TRIES" 2>/dev/null
  [ "$RC" = "0" ] && exit 0
  sleep 30
done
