# Optimization contract — every lever, tested, in order

Written 2026-09-19 after the final run21_win bench: 829M steps scored 70/200 on
the 1v3 unarmed d=1.0 cell, statistically identical to the 261M start (77/200).
This document is the plan for the next runs. Nothing in it is a suggestion:
each lever has a test, a budget, a confirm threshold, a kill threshold, and a
place in the order. A lever that is not on this list does not get pulled
without being added here first.

Throughput reference: 715 steps/s active on the Windows trainer. 10M steps ≈ 4 h.
Bench = `evaluate.py`, seeds 900000+, t_train_101, 1v3, unarmed, d=1.0,
1200-step cap. **Corrected 2026-09-20:** the engine is not deterministic
(identical checkpoint, identical seeds: 44 then 59), so the honest noise at
n=200, p≈0.35 is **±13 wins** (Wilson), not the ±4 a lucky 5-sample spread of
the baseline suggested. Every bench is now **400 greedy + 200 sampled**
(±9 / ±13), and the periodic eval snapshots the checkpoint it benched
(`checkpoints/snapshots/<run>_<step>.pt`) so any number can be re-benched.
A policy whose button probabilities sit near 0.5 is a knife-edge under
greedy decoding (`walk` crossed 0.5 in sel0's last 1.3M steps and the
greedy bench moved 38 wins); report sampled alongside greedy until Phase 4.2
replaces the independent Bernoullis.

## Phase 0 — correctness, before a single training step (zero budget)

These are verified bugs. All are fixed identically on every arm below so that
arms differ in exactly one thing.

| # | defect | fix | verification |
|---|---|---|---|
| 0.1 | `vec_env._reset_env()` drops `armed_count`, `weapon_type`, `throw_aggression` from the sampled scenario. The armed curriculum was never wired into training; the gate would have certified a fight the collector never ran. | forward all three to `env.reset()` | unit test: sampler at stage B1 → `env.reset` receives `armed_count=1`; 200 training episodes at stage B1 log `armed_count>0` |
| 0.2 | Time-limit bootstrap adds `gamma·V(s')` (normalized-reward units) to the RAW reward, then normalizes the sum. Double-scales the bootstrap and pollutes reward RMS with value estimates. 11–15% of episodes time out. | normalize env reward first; add `gamma·V(s')` to the normalized reward on truncated transitions only (SB3 VecNormalize order) | offline replay of 20 stored rollout batches through both formulas: report median/p95 abs Δadvantage on (a) ordinary transitions (b) last 30 decisions before a timeout (c) whole timeout episodes, and the sign-flip count. Then it ships regardless; the replay only tells us how much it mattered. |
| 0.3 | `train_vec.py:706` rebuilds `won` from `opponent_knockout > 0`; `vec_env.py` already computes all-hostiles-down. Two definitions of the one number that gates everything. | `vec_env` sets `info["won"]`; every collector consumes only that field; delete the reconstruction | grep shows one definition; a same-step "last KO + timeout" synthetic case labels correctly |
| 0.4 | `GetAttackTarget()` picks the RL character's attack target by **camera** facing. Under RL no look axes exist and headless never sets auto-camera, so the yaw is frozen at spawn for the whole episode. Every attack in 829M steps went to whichever enemy aligned with a value the policy neither sees nor controls. Invisible in 1v1; decisive in 1v3. Confirmed against stock WolfireGames/overgrowth. | `rl_target_select` launch config: 0 stock, 1 self-facing dot, 2 nearest (native AI rule). Gated on `IsExternalRLController`. Human play unchanged. | zero-training 3×200 A/B on 829M (running); then Phase 1 |
| 0.5 | `target_kl` early stop checks approx-KL AFTER `optimizer.step()`, so the pathological minibatch is applied before the guard fires. 46 updates in 40k had KL>1 with clip_fraction ≈ 1/128. | compute KL before the step and `break` before applying (SB3 order). Add exact per-head KL: Gaussian ×2, Bernoulli ×6, joint, plus the sampled estimator, logged every update. | first-minibatch invariant (37-details audit): on minibatch 0 of epoch 0, `new_log_prob − old_log_prob` must be ~0 (|max| < 1e-4) for 500k steps. Any violation = rollout/update reconstruction bug, stop everything. Then: fraction of KL-spike mass attributable to a single Bernoulli head. |
| 0.7 | The continuous log-prob was rebuilt at update time from `atanh(clamp(tanh(raw)))`; any \|raw\| > 3.8 shifts that sample by nats. ~1 per 3584-step update — enough to trip the sampled-KL guard (0.09 vs 0.02) on **100% of updates** at median minibatch 9/28. `mb0_max_abs_logratio` read 5.9. | buffer stores the pre-tanh sample; update evaluates there; guard gates on exact per-minibatch KL | `mb0_max_abs_logratio` = 0.0 in smoke; `early_stop_minibatch` distribution on Phase 1b |
| 0.6 | No deterministic eval ever ran after 277M because the gate pre-filter never fired. 550M steps with zero automated measurements of the target metric. | periodic 200-episode deterministic bench every 25M steps regardless of gate state, written to `events.jsonl` and `eval/` | first event appears at +25M on the next run |

## Phase 1 — actuator: target selection (the lever with the highest prior)

**Result 2026-09-20 (research-artifacts/OGRL-20260920-001-phase1):** both arms
ended BELOW the 261M start on every measure after 10M (greedy 83 → 44/59 and
76 → 62; sampled 67 → 53 and 60; in-run sampled 0.34 → 0.28 and 0.36 → 0.26).
No selector effect distinguishable. The run was invalid as a learner test:
the target_kl guard fired on 100% of updates at median minibatch 9/28 because
of the tanh-clamp log-prob reconstruction (0.5 below, now 0.7). **Phase 1b**
re-runs the same two arms on the fixed learner at the OLD hyperparameters so
the delta is attributable, with the corrected bench.

The 829M policy is an equilibrium under the frozen-camera selector; a flat
zero-training A/B does not kill this. Retrain is required.

| arm | change | budget | confirm | kill |
|---|---|---|---|---|
| 1.0 control | `rl_target_select=0` | 10M | — | — |
| 1.1 | `rl_target_select=1` (self-facing) | 10M, bench at 2M/5M/10M | pre-attack **target acquisition** emerges (below) and selected→first-contact consistency ≥ +10 pp vs control; hard-cell bench moving by 10M | no acquisition signal by 5M and no bench difference at 10M |
| 1.2 | `rl_target_select=2` (nearest) | 10M | same; preferred if self-facing turns out to be animation-driven rather than policy-driven | same |

All three arms resume from **261M** (higher entropy 1.9 vs 1.15, best-characterized baseline, no plateau-tuned optimizer state) with Phase 0 applied identically.

**Behavioural metric (not win rate):** on every attack launch with ≥2 legal
candidates, look back 8 decisions (~267 ms). Margin
`M = |θ_runner-up| − |θ_selected|` (bearing from self-facing; distance margin for
mode 2). The selector guarantees `M>0` at launch; the question is whether the
policy **grows** `ΔM` during the pre-attack window relative to matched
non-attack windows. CI excluding zero and growing from the 261M init = the
agent is manufacturing the geometry that makes its intended target win. Log
per attack: button edge, attack kind, candidate IDs, selector output, first
contacted victim, self pos/facing, candidate pos/attack-state, preceding 8
actions. The engine already emits `rl_log_attacks`; extend it with candidates
and selector output.

Prerequisite to choose 1.1 vs 1.2: read the RL player's facing-update path
(does the policy have fast causal authority over `this_mo.GetFacing()`, or is
it animation/target driven?). If facing is not directly controllable, mode 2
is the clean interface and 1.1 is dropped.

## Phase 2 — representation: max-pool vs attention (offline first)

The entity encoder max-pools 8 independently-embedded entities per frame. That
cannot bind "attacker on my left at phase X while the one on my right is
recovering" — it produces an aggregate that corresponds to no actual opponent.
Necto and Lucy-SKG (Rocket League) both use a Perceiver-style cross-attention
with the player as query over entity keys/values; EARL ships max-pool only as
an "experimental" alternative.

| step | what | budget | confirm | kill |
|---|---|---|---|---|
| 2.0 probe | collect 1M transitions from the Phase-1 winner; train frozen probes on (a) current max-pooled representation (b) raw entity tokens + small cross-attention (c) attention + ~1 s history, predicting: which enemy makes next hostile contact; whether two distinct enemies go attack-active within 10 decisions; which enemy the next resolved attack hits; self-KO within 30 decisions | 0 training steps, ~1 GPU-hour equivalent on CPU | attention beats max-pool by ≥0.05 AUC on first-attacker / next-contact labels | max-pool matches attention → drop the architecture story, do not spend Phase 2.1 |
| 2.1 RL | replace EntityEncoder pooling with 4-head cross-attention (256-d, proprioception as query), keep everything else; start from Phase-1 winner with trunk weights carried, attention cold | 15M | ≥ +8 wins/200 over the Phase-1 arm AND 2-KO→3-KO conversion up, not only first-KO rate | ≤ +5 wins at 15M |
| 2.2 (only if 2.0c > 2.0b) | add ~1 s recurrence (GRU) after attention | 15M | same | same |

## Phase 3 — learner recipe

Only after Phase 1 has shown the metric can move. Each is a single-variable A/B against the current best.

| lever | change | budget | confirm | kill |
|---|---|---|---|---|
| 3.1 discount | γ 0.99 → 0.9975 (matches Lucy-SKG/Necto's 0.995 at 15 Hz: ~9 s half-life instead of 2.3 s). Copy actor; **reset critic, optimizer, reward normalizer** (value scale changes). λ stays 0.95 for a clean A/B. | 20M | survival and 2→3 conversion up on the hard cell after the critic restabilizes | only critic statistics change |
| 3.2 reuse | n_epochs 1 → 3, LR 3e-4 → 1e-4 constant, minibatch 256, rollout-wide (not per-minibatch) advantage normalization. Requires 0.5 (pre-step KL) first. | 10M | more policy movement per transition without true-KL blowups; ≥ +8 wins | optimization metrics change, behaviour does not |
| 3.3 λ | only after 3.1; sweep {0.95, 0.97} | 10M each | — | — |

## Phase 4 — objective

| lever | change | budget | metric | kill |
|---|---|---|---|---|
| 4.1 backloaded KO | keep max successful-episode KO reward constant: 16+16+16=48 → 8+8+8+24(clear). Not "+bonus" — same nominal total, dying after two kills is worth much less. | 10M | primary: `P(clear | reached 2 KOs)` ≥ +10 pp with `P(reached 2 KOs)` not collapsing | conversion flat |
| 4.2 action grammar (later) | six independent Bernoullis → small legal compositional action set or autoregressive button heads (grab/drop pressed 85–89% of the time unarmed is the symptom) | 15M | attack press rate and useful-combo rate up; ≥ +8 wins | — |

## Phase 5 — curriculum (last, and only shapes that keep 3 opponents)

Not the old 1v1/1v2 mixture — that spent 94% of compute on solved cells. Test
only scaffolds that keep 1v3 and the same state/action semantics: reduced
attack staggering/aggression, or start states immediately after the first KO.
Budget 15M. Confirm: improvement transfers to untouched d=1.0 1v3 by ≥ +8
wins. Kill: only the assisted scenario improves.

## Phase 6 — throughput (parallel track, never blocks Phases 0–4)

| lever | test | status |
|---|---|---|
| 6.1 (n_envs, k_standby) | `throughput_sweep.py`, real trainer, 8.5 min/point, median `steps_per_second_cycle` | running; winner goes into `run_forever.bat` |
| 6.2 async collector | only after 0.3 (won-label) lands in `async_vec_env.py` and `rollout_worker.py`; then same sweep | blocked on 0.3 |
| 6.3 update threads | 4 vs 8 at the winning (n,k) | after 6.1 |
| 6.4 engine rewrite (Gym-style multi-sim in-process / JAX combat core) | **not now.** 568M steps at 715 sps bought nothing; throughput is not the binding constraint until a 10–20M intervention moves the bench. Revisit only after Phase 3. | deferred |

## Phase 7 — infrastructure (before any multi-day run)

- Laptop AC loss killed 20.9 h of 269 h (three battery deaths). Keep it plugged in; additionally test `powercfg` critical-battery action → hibernate with 12 engines + shm live, and confirm the run resumes.
- `ShmWaitTimeout` ~daily: engine goes silent. Root-cause with the engine's own log around the silence; the watchdog restart is a bandage.
- Scheduled-task hygiene: every ONCE-trigger task deleted after it runs (two of them re-fired and started duplicate supervisors).
- Periodic deterministic bench (0.6) plus a pushed summary line per bench so nobody waits a week to find out.

## Order of execution

```
Phase 0 (all six) ──► zero-training selector A/B (running)
                  ──► Phase 1 (3 arms × 10M from 261M)  ≈ 12 h
                  ──► Phase 2.0 probe (offline)          ≈ 2 h
                  ──► Phase 2.1 if probe passes          ≈ 6 h
                  ──► Phase 3.1, 3.2 as single-variable A/Bs
                  ──► Phase 4.1
Phase 6 runs alongside on the trainer's idle windows; Phase 7 before the first >24 h run.
```

Stop rule for every arm: bench at 5M and 10M (two independent 200-episode
greedy replicates + 200 sampled, checkpoint snapshotted); an arm still under
0.50 at 10M and not moving is killed. No arm gets a 50–100M budget until a
10–15M arm has moved the hard cell by more than the noise floor — ±13 wins at
n=200, ±9 at n=400 — on BOTH greedy and sampled.
