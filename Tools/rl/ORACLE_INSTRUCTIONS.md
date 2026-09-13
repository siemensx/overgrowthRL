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

For the direct-movement diagnostic, `--flank` is now an actual exposed-side
approach: it circles toward a deterministic rear quarter and suppresses a
strike while the target is still facing the rabbit. Without `--flank`, the
oracle is allowed to attack a standing target head-on and will often hit the
normal passive guard. The direct movement vector is world-space, while the
engine still supplies the normal combat-facing/locomotion animation.

### Counter-throw diagnostic

To test the specific “active-block, then hold grab/throw” hypothesis, use
`--block-throw`. It withholds free-form attacks until the first legal counter,
arms the ordinary active-block mechanic from the opponent's real
`blockprepare`/`attackimpact` events, selects only a target with
`block_stunned_by_id == player_id`, and then allows a follow-up strike only
while that victim is ragdolled. This is still privileged because target IDs
and animation timers are hidden from the public RL observation.

For a visible, real-time run with a wider spectator view:

```bash
cd /Users/pavlov/Documents/GitHub/badbunny/overgrowthRL
python3 Tools/rl/engine_ai_baseline.py \
  --controller oracle --level arenas/t_train_101.xml \
  --opponents 3 --difficulty 1.0 --armed-count 0 \
  --frame-stack 4 --act-period 4 --episodes 1 --max-steps 1200 \
  --no-jumpkick --block-throw --render --auto-camera \
  --spectator-fov 110 --trace \
  --out Tools/rl/runs/oracle-baseline/watch_blockthrow_$(date +%Y%m%d-%H%M%S).json
```

The timestamped result path is intentional. The runner refuses to overwrite
an existing JSON result, and `--trace` keeps the engine log under
`Tools/rl/runs/oracle-baseline/engine-traces/`. A throw is not guaranteed by
holding grab: the real game requires a successful active block first, and an
opponent may escape or a second attacker may land during the same frame.

The first revised 20-episode spacing batch (seeds 906000–906019, stock
unarmed guards, difficulty 1.0) produced 1/20 wins, 11 losses, 8 timeouts,
and 0.35 hostile KOs per episode. Later source-audit samples produced 0/10
with the follow-up transition fix, 0/10 with direct exposed-side flanking,
and 0/10 after selecting the rabbit's moving-low finisher for ragdolls; these
small samples are not evidence of a monotonic improvement. A 20-episode 1v1
flank sample produced 8 wins, 4 losses, and 8 timeouts. The changes fixed
observable controller inconsistencies, but did not establish a 90% 1v3
controller. A repeated single-seed launch is not expected to be bit-identical
here because opponent AngelScript contains stochastic behavior and reset does
not reset every global random stream.

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
