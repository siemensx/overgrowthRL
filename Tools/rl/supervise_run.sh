#!/usr/bin/env bash
# Keep a training run alive unattended.
#
# Three failure modes have actually ended runs on this machine, all recoverable
# without a human:
#   1. --stop-below-free-gb fires. The trainer exits CLEANLY and checkpoints
#      first, so the fix is simply to restart once space is back.
#   2. A worker goes silent and the trainer dies on ShmWaitTimeout.
#   3. Reusing an shm prefix after the engines were hard-killed attaches to an
#      orphaned semaphore, so every restart MUST use a fresh name.
#
# It also watches the real disk consumer: macOS swap. 16GB of RAM shared with
# the user's browser and other apps means engines push the machine into swap,
# swapfiles live on /System/Volumes/VM, and THAT is what took the disk from
# 4.7GB to 0.44GB in 35 minutes -- not the run's own files. When free space gets
# tight the supervisor steps n_envs down rather than letting the run die.
set -uo pipefail
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL

RUN_ID=run21_mac
LOG=/tmp/supervisor.log
N_ENVS=${1:-8}
K_STANDBY=${2:-2}
# Last-resort floor, not a target. Measured 2026-09-06: the memory pressure on
# this machine is dominated by the browser (7.99GB) not the run (1.73GB), so
# shedding workers buys little -- but two envs still makes progress where a
# paused run makes none.
MIN_ENVS=2
LAUNCH_N=0
CKPT=Tools/rl/ppo/checkpoints/${RUN_ID}.pt
# Only maps whose fighters can actually reach each other. 103/105/106 were
# generated with --tiers, which spawns the OPPONENT on a raised deck 7.8-11.3
# units above the agent and drops the 1v3 hostiles inside the ramp geometry.
# Measured timeout rates, last 60k episodes:
#   101/102/104 (no tiers): 0.2-2.5% across 1/2/3 opponents
#   103: 7/6/81%   105: 78/84/91%   106: 19/3/46%
# They are out until gen_arena_map.py places spawns on reachable ground and
# the regenerated maps pass the timeout gate in Tools/rl/validate_maps.py.
# All six regenerated/verified. Empirical gate (validate_maps.py, 12 episodes
# per cell at 1 and 3 opponents, 2026-09-07): 103/105/106 now time out on
# 0.0%% of episodes, down from 7-91%%. 101/102/104 were already healthy and
# were NOT regenerated -- they are 60%% of the policy's training history.
# 101/102/104 ONLY. These three were never regenerated -- they are the
# original geometry on the RL level script, and 60% of the policy's
# training history. 103/105/106 were regenerated twice tonight and are
# out until they pass validate_maps.py AND render_smoke.sh on the RL
# script: the first validation pass scored them while they were still on
# the stock multi-round arena script, which revives the fallen, so its
# "0.0% timeouts" result said nothing about whether a fight resolves.
LEVELS=arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG"; }

# A fresh supervisor OWNS the run. Without this it only calls launch() when it
# sees the run DOWN, so starting a new supervisor over a live trainer silently
# leaves the old one in charge -- with the old level list, the old write-dirs
# and the old baked navmeshes. That happened on 2026-09-07: two "restarts" to
# change the map rotation both appeared to succeed while the 20:14 process kept
# running, and it then read regenerated map files against navmeshes baked from
# the geometry they replaced. 76% of episodes timed out before it was spotted.
for pid in $(pgrep -f "ppo/train_vec.py"); do
  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
done
pkill -f "MacOS/Overgrowth" 2>/dev/null; sleep 2

free_gb() { df -k /System/Volumes/Data | tail -1 | awk '{printf "%.2f", $4/1048576}'; }

launch() {
  # Do not relaunch into a disk condition that will stop the run within seconds.
  # Observed 2026-09-06: with the machine at ~0.39GB free the supervisor
  # restarted, --stop-below-free-gb fired immediately, and it restarted again --
  # a churn loop that spawns and kills ten engine processes every two minutes
  # and makes the disk situation marginally worse. Wait for real headroom
  # instead; the checkpoint is already safe and nothing is lost by waiting.
  local waited=0
  while awk "BEGIN{exit !($(free_gb) < 0.75)}"; do
    if [ "$waited" -eq 0 ]; then say "free=$(free_gb)GB -- holding launch until >=0.75GB"; fi
    waited=$((waited + 60)); sleep 60
  done
  [ "$waited" -gt 0 ] && say "space returned after ${waited}s (free=$(free_gb)GB)"
  local shm="/ogrl_s$(date +%H%M%S)"     # fresh every time -- see (3) above
  # Rotate which map each worker is dealt. Workers are dealt levels round-robin
  # from the head of LEVELS, so any launch with fewer workers than maps -- every
  # launch after memory-pressure shedding -- trains only on the head of the list.
  # run21_mac spent its low-memory stretches on 101/102 alone. Advancing the
  # offset each launch makes the dropped maps differ instead.
  LAUNCH_N=$((LAUNCH_N + 1))
  pkill -f "MacOS/Overgrowth" 2>/dev/null; sleep 3
  nohup env OGRL_ALLOW_NENVS_CHANGE=1 OGRL_LEVEL_OFFSET="$LAUNCH_N" caffeinate -i python3 -u Tools/rl/ppo/train_vec.py \
    --repo-root "$PWD" --levels "$LEVELS" --shm-prefix "$shm" \
    --n-envs "$N_ENVS" --k-standby "$K_STANDBY" --seed 21 \
    --total-timesteps 400000000 --n-steps 256 --n-epochs 1 --minibatch-size 128 \
    --entropy-coef 0.008 --entropy-coef-final 0.003 --entropy-anneal-steps 10000000 \
    --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 \
    --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 \
    --d-max-start 0.15 --d-max-cap 1.0 --d-step 0.1 \
    --gate-window 300 --gate-min-samples 50 --gate-win-rate 0.75 \
    --opponents-cap 3 --opp-gate-win-rate 0.6 --opp-gate-window 400 \
    --opp-gate-min-samples 150 --opp-keep-solo 0.35 \
    --collection-torch-threads 1 --update-torch-threads 4 --torch-interop-threads 1 \
    --checkpoint-path "$CKPT" --checkpoint-every-updates 100 \
    --run-id "$RUN_ID" --purpose "supervised: n_envs=$N_ENVS k_standby=$K_STANDBY" \
    --no-tapes --no-native-capture --pause-below-free-gb 0.55 --stop-below-free-gb 0.35 \
    --resume-from "$CKPT" > /tmp/${RUN_ID}.log 2>&1 &
  say "launched n_envs=$N_ENVS k_standby=$K_STANDBY shm=$shm free=$(free_gb)GB"
}

# Bound the run's own footprint: per-worker logfile.txt is diagnostic only.
janitor() {
  while true; do
    for f in .rl_write_dirs/*/logfile.txt; do
      [ -f "$f" ] || continue
      [ "$(stat -f%z "$f" 2>/dev/null || echo 0)" -gt 20000000 ] && : > "$f"
    done
    sleep 60
  done
}
pkill -f "writedir_janitor" 2>/dev/null
janitor & JANITOR=$!
trap 'kill $JANITOR 2>/dev/null' EXIT

say "supervisor starting (n_envs=$N_ENVS k_standby=$K_STANDBY, free=$(free_gb)GB)"
pgrep -f "run-id $RUN_ID" >/dev/null || launch

while true; do
  sleep 120
  free=$(free_gb)

  # Proactive shed. macOS NEVER shrinks swapfiles once allocated, so disk lost
  # to swap is gone until reboot -- waiting for the run to die and then
  # stepping down means the space is already spent. Shed a worker pair while
  # the run is still healthy instead.
  if pgrep -f "run-id $RUN_ID" >/dev/null && awk "BEGIN{exit !($free < 0.9)}" && [ "$N_ENVS" -gt "$MIN_ENVS" ]; then
    N_ENVS=$((N_ENVS - 2))
    say "free=${free}GB while running -- shedding to n_envs=$N_ENVS before it stops us"
    p=$(pgrep -f "run-id $RUN_ID" | head -1)
    [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
    for _ in $(seq 1 60); do pgrep -f "run-id $RUN_ID" >/dev/null || break; sleep 1; done
    launch
    continue
  fi

  if ! pgrep -f "run-id $RUN_ID" >/dev/null; then
    step=$(python3 -c "
import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['global_step'])" 2>/dev/null || echo '?')
    # Down with little space: shed a worker pair before retrying, so we do not
    # relaunch straight back into the condition that killed it.
    if awk "BEGIN{exit !($free < 0.8)}" && [ "$N_ENVS" -gt "$MIN_ENVS" ]; then
      N_ENVS=$((N_ENVS - 2))
      say "DOWN at step $step, free=${free}GB -- stepping down to n_envs=$N_ENVS"
    else
      say "DOWN at step $step, free=${free}GB -- restarting"
    fi
    launch
  fi
done
