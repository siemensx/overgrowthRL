# Scripted combat diagnostics

These tools answer two different questions and must not be reported as the
canonical RL score.

## What the game bots actually do

The native controller is `Data/Scripts/enemycontrol.as`. It is a state machine,
not a single attack rule. It maintains a visible-target history, predicts target
motion, uses navmesh approach points and attack ranges, selects attack classes,
reacts to attack animation events with active block/dodge, recovers from
ragdoll, and coordinates groups through leader/follower and wait states.

The installed Dynamic AI Aggression 1.5.1 script is a separate workshop version
and adds stochastic group timing, jump-kick decisions, weapon-defense branches,
and extra attack-range modes. It is an opponent behavior axis, not evidence of
a 90% deterministic player policy. Do not silently mix it into a canonical
evaluation; record `--throw-aggression` and the active mod state.

## Fair observation-only oracle

This controller reads the public structured observation and emits only the
normal eight-value RL action. It has no target-ID action and cannot see through
walls:

```bash
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL
python3 Tools/rl/observation_oracle_bot.py \
  --level arenas/t_train_101.xml --opponents 3 --difficulty 1.0 \
  --armed-count 0 --frame-stack 4 --act-period 4 --episodes 200 \
  --seed-base 900000 --style engine-inspired \
  --out Tools/rl/runs/oracle-baseline/t_train_101_1v3u_seed900000_obsoracle_$(date +%Y%m%d-%H%M%S).json
```

Use a new output stem every run. Existing JSON files are refused.

## Privileged engine oracle

This diagnostic runs the game's combat/physics pipeline in the player slot,
but uses hidden engine state for omniscient target acquisition and deterministic
target/facing/attack choices. It is useful as an upper-bound investigation, not
as a fair policy score:

```bash
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL
python3 Tools/rl/engine_ai_baseline.py \
  --level arenas/t_train_101.xml --opponents 3 --difficulty 1.0 \
  --armed-count 0 --frame-stack 4 --act-period 4 --episodes 50 \
  --seed-base 970000 --no-jumpkick \
  --out Tools/rl/runs/oracle-baseline/t_train_101_1v3u_seed970000_engineai_oracle_$(date +%Y%m%d-%H%M%S).json
```

`--no-jumpkick` is the currently strongest tested 1v3 ablation. Omit it to
test the jump-kick profile. The oracle may choose legal throws, but it never
calls a damage function or edits health, position, velocity, or physics state.

Both tools use unique shared-memory names, stop cleanly on SIGTERM, write
results atomically, and refuse to overwrite an existing result. Keep each JSON
beside its run evidence; do not copy it over an earlier result.

## Interpreting results

Report episode count, seeds, wins, losses, timeouts, mean hostile knockouts,
scenario parameters, and profile. A deterministic policy does not imply a
deterministic win rate: the game physics and opponent scripts still contain
seeded/random behavior. A state-mutating “cheater” that directly damages or
teleports entities is not a meaningful ceiling test and is not part of this
project.
