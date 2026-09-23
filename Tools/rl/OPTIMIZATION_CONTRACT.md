# Optimization contract — every lever, tested, in order

Written 2026-09-19 after the final run21_win bench: 829M steps scored 70/200 on
the 1v3 unarmed d=1.0 cell, statistically identical to the 261M start (77/200).
This document is the plan for the next runs. Nothing in it is a suggestion:
each lever has a test, a budget, a confirm threshold, a kill threshold, and a
place in the order. A lever that is not on this list does not get pulled
without being added here first.

**Adoption rule updated 2026-09-22 per owner direction:** there is no fixed
minimum percentage gain for a safe, semantics-preserving optimization. A
repeatable positive gain is worth retaining even when it is small, because
independent gains can compound. Use paired/reversed-order repeats and confidence
intervals to separate signal from thermal/order noise; do not reject a lever
merely for missing a 5% bar. Correctness, map coverage, useful transitions,
pool/recovery health, and checkpoint safety remain hard gates. A one-off positive
measurement is a lead, not an adopted gain.

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

## Takeover addendum — Claude Code session and Windows optimization ledger

Added 2026-09-21 under `OGRL-20260921-001`. This is the continuation contract
after the Claude Code session titled **Project setup on Windows training
machine** exhausted its context. The complete transcript is preserved at
`/Users/pavlov/.claude/projects/-Users-pavlov-Documents-GitHub-badbunny/31de9611-da6d-41ea-84ec-5db8ea852c4f.jsonl`.

### Exact takeover point

Claude stopped after profiling and partial throughput testing, not after a
successful optimization handoff. The last real training process was
`run23_sel0`, resumed from `run21_baseline_260m.pt`, with 14 active engines and
4 standbys. Its last checkpoint is at global step **262,972,636** and its last
recorded cycle rate is **737.45 steps/s**. The run manifest still says
`running`, but the Windows process is gone; this is a stale manifest, not an
active run. No training process is currently running on the trainer.

Claude's immediate research context was:

1. The old 500M-step result did not improve the 1v3 cell because attacks were
   selected using a frozen camera direction. The target-selection correction
   was made and is a correctness change, not proof of a better policy.
2. A `>`/`>=` controller-identification error was fixed. The 829M policy was
   worse under the corrected controls, while the 261M checkpoint was retained
   as the clean restart point.
3. The learner was discarding most updates because the continuous log
   probability was reconstructed after tanh-clamping. That made the KL safety
   guard stop almost every update. The fix stores the pre-tanh sample and was
   the first valid learner rerun; the first million steps moved from roughly
   0.29 to 0.37–0.40 in the in-run signal, but this is not the final held-out
   1v3 result.
4. Claude then switched to throughput work: asynchronous collection, Python
   bookkeeping, inference mode, thread counts, process priority/affinity, and
   phase-level timing. The session ended before a clean winner was adopted.

### Ledger of levers Claude actually attempted

The result column records measured output, not a claim that the change is
safe for a production run. `n_envs` means active game processes; `k` means
standby processes.

| lever | what was changed or measured | output | disposition |
|---|---|---|---|
| Target selection | Compared the old camera-facing selector with self-facing/nearest modes; fixed the RL targeting path first | The earlier learner A/B was invalid because the tanh/log-prob bug made the KL guard fire on 100% of updates. Both arms fell below the 261M start after 10M; no selector effect can be inferred | Keep as a correctness prerequisite; re-run Phase 1b only after the learner smoke test passes |
| Controller identity | Fixed the one-character `>`/`>=` test so the RL character is identified correctly | The old control behavior was not the intended experiment; the 829M policy degraded under the corrected controls | Adopted for correctness; do not compare old and corrected-policy scores as an optimization A/B |
| KL/log-prob update | Moved the KL check before the optimizer step and stored the pre-tanh continuous sample | The previous reconstructed log ratio reached about 5.9 and tripped the guard on every update; this was fixed in the current code path | Adopted as a correctness fix; require the first-minibatch invariant before further learner runs |
| Critic detach and related learner review | Reviewed the value path so the critic cannot rewrite actor inputs; corrected three adjacent issues found during review | No isolated throughput or held-out-win result was produced | Keep in the fixed baseline; no separate adoption claim |
| Asynchronous collector, 14 engines | Same general workload, two repeats: sync 2698.795 and 1911.467; async 1821.138 and 1977.925 decisions/s | Two-repeat medians: sync 2305.13, async 1899.53; async was slower and highly variable | Rejected for now; only revisit after a clean current-code test proves the implementation is materially different |
| Asynchronous collector, 20 engines | Two repeats: sync 1961.686 and 2022.916; async 1916.168 and 1883.045 decisions/s | Medians: sync 1992.30, async 1899.61 | Rejected; it did not remove the Windows straggler cost |
| Nonblocking periodic evaluation/bench | Changed the trainer to launch periodic evaluation without blocking collection, then drain the result at completion | Infrastructure change was written; no isolated end-to-end rate A/B was completed | Keep, but validate that evaluation never changes on-policy ordering or checkpoint selection |
| Observation unpacking | Vectorized observation unpacking and finite checks | cProfile still showed `obs_schema.entity_field` and `math.isfinite` as hotspots; Claude reported the lite reward/entity path was equivalent on its smoke data, but no clean training A/B was completed | Keep as a candidate; measure against a same-seed control before adoption |
| Reward/entity bookkeeping | Added a reduced entity-diff path instead of rebuilding full dictionaries where possible | No isolated wall-rate result; the cProfile profile still put reward/observation bookkeeping in the top group | Pending clean A/B |
| PyTorch inference mode | Used `torch.inference_mode()` during collection | On Windows policy forward timing improved from about 0.96 ms to 0.61 ms at one thread; the isolated microbenchmark reached about 0.42 ms versus 0.46 ms for no-grad | Keep; the end-to-end gain is not yet independently measured |
| PyTorch update threads | Tested the policy microbenchmark at 1/2/4 threads; 2 threads was fastest in that microbenchmark (~0.46 ms), while 4 threads was slower (~0.52–0.58 ms) | The real trainer tests used 4 update threads, so this is not a completed sweep | Pending 1/2/4/8 end-to-end A/B at the same engine count |
| ctypes signatures | Added explicit ctypes argument types for the shared-memory calls | No isolated rate result; correctness smoke passed during the profiling work | Keep, but count as unproven until a controlled A/B |
| Timer and phase instrumentation | Switched timing to `perf_counter`; added policy, environment-step, bookkeeping, worker-wait, and barrier metrics | Instrumentation showed, at n14/k4, roughly 2.38 ms policy, 19.04 ms vector step, 1.11 ms bookkeeping, 22.57 ms collection; mean engine wait 13.11 ms per worker and worker-latency p99 about 20.99 ms | Adopted as measurement infrastructure; timers themselves are not a speed win |
| Engine priority/affinity | Scheduler sweep at n14/k4: normal trainer + normal engine **748.38 median sps**; above + above **720.19**; normal trainer + above engine with affinity `0xFFF` **754.53** | Above priority for both was worse. The mixed setting was only about 0.8% above normal in this run | Keep normal trainer priority, test engine priority/affinity only in a clean interleaved repeat; do not call this adopted yet |
| Python-side fixes plus mixed scheduler | n14/k4: **728.31 median sps**; n18/k4: **716.70 median sps** under the same reported launch settings | n18 added engines but reduced throughput; this repeat did not reproduce the older 828 sps point | n14 remains provisional; n18 is rejected for this workload unless thermally controlled retesting reverses it |
| Earlier n_env/k sweep | Existing Windows points: n8/k2 668.84; n10/k2 772.95; n10/k4 682.67; n12/k2 712.27; n14/k4 828.10 median sps | These points were taken under different source/runtime/launch states and cannot be ranked against the later Python-fix sweep as one experiment | Preserve as evidence; repeat the winner and control under one clean configuration |
| cProfile | Profiled 100 updates: 321,467,733 calls in 493.565 s | Largest costs included semaphore wait 79.38 s, queue get 65.15 s, semaphore post 54.00 s, entity-field extraction 22.78 s, finite checks 13.93 s, and policy/vec-step call overhead | Diagnosis only; use it to target a new A/B, not as a throughput result |
| py-spy | Tried live attach, then spawn profiling | Attach/quoting/PID attempts failed; the final spawn capture mostly measured process startup and DLL loading, not steady-state collection | Rejected as evidence for the hot loop; do not repeat without a verified live PID and a bounded capture plan |
| Phase sweep at n12 | Attempted to extend the phase breakdown | Failed because the temporary Windows disk ran out of space while creating the phase script; this is a harness failure, not a performance result | Fix temporary-space hygiene before repeating; never label this point slow or fast |

### What is left from earlier proposals

These are the remaining levers, ordered by likely return and by how safely they
can be tested without touching the Mac:

1. **Clean concurrency baseline:** repeat n4, n6, n8, n10, n12, n14, and
   n18 with the same binary, same three-map corpus, same checkpoint, same
   `n_steps=256`, same `update_threads`, fresh shared-memory prefix per point,
   and an explicit character-count assertion from each engine log. Use a
   warmup followed by at least two measured repeats. This resolves the
   contradictory 715/728/748/754/828 results.
2. **Standby sweep:** at the winning active count, compare k=2 and k=4, then
   k=6 only if the process and memory budget remain healthy. The map corpus
   size and active+standby allocation must remain compatible with the project
   map-axis rule.
3. **Thread sweep:** at the clean winner, test update threads 1, 2, 4, and 8.
   Record learner time, collection time, wall steps/s, CPU placement, RAM, and
   thermal throttling. Do not infer this from a single policy microbenchmark.
4. **Scheduler/affinity repeat:** interleave normal and above engine priority,
   with and without the tested affinity mask. Do not discard a repeatable gain
   merely because it is under 5%: small gains compound across the stack. A CPU mask must
   be described in terms of the actual 165U logical processors; `0xFFF` is not
   automatically a P-core-only mask.
5. **Nonblocking evaluation validation:** run a short trainer job with
   periodic evaluation enabled and compare update ordering, checkpoint step,
   and collection rate against evaluation-disabled control. Keep it only if
   the reported training state is identical apart from the intended eval work.
6. **Decision frequency:** test 30, 20, and 15 Hz only as a controlled
   throughput/fidelity A/B. This changes the action-hold contract, so it is not
   a free optimization and cannot be adopted from speed alone. Evaluate both
   1v1 retention and 1v3 performance.
7. **Engine-side sound early-outs:** the null-backend `SetPosition` and
`TranslatePosition` early-outs are already present in source commit `517beb56`
and its descendant Windows binary anchor `d6f67601`; they are therefore part of
the current optimized baseline, not an untested edit to apply mid-run. The
remaining work is only an isolated before/after build A/B if a clean historical
measurement is needed.
8. **Async collector:** currently rejected. Revisit only if the clean current
   collector shows a reproducible straggler regime that an updated async
   implementation addresses, and then compare on-policy ordering and final
   metrics, not rate alone.
9. **Large engine rewrite/JIT:** deferred. The AngelScript/JIT ceiling and
   prior source probes are already recorded as negative evidence. Do not spend
   the next training window here.

The policy-quality work remains separate from throughput: Phase 1b target
selection, Phase 2 representation probes, Phase 3 learner recipe, Phase 4
backloaded knockout reward/action grammar, and Phase 5 curriculum. No quality
arm gets a long budget until the fixed learner passes the first-minibatch
log-ratio invariant and a 10–15M arm moves both greedy and sampled 1v3 results
past the documented noise floor.

### Test and adoption contract for this takeover

Every optimization trial must record the exact Mac source commit, Windows
binary hash or build identifier, Python/Torch versions, map list, character
count, n/k, thread count, priority, affinity, shared-memory prefix, seed,
warmup, measurement window, median/p10/wall steps/s, CPU/RAM/thermal state,
crashes/recoveries, and raw artifact paths. A trial is invalid if it reuses a
shared-memory name after a hard kill, silently changes the map corpus, loses
characters, or measures an empty level.

There is no blanket 5% rejection threshold. The project should accumulate
small improvements because ten independent 2% gains can materially change the
training time. The rule is:

- Low-risk, behavior-preserving changes (timers, allocation, IPC, inference
  bookkeeping): keep any repeatable positive gain that is larger than the
  measurement uncertainty, even around 0.5–1%, if the maintenance cost is
  negligible.
- Worker, thread, priority, and affinity settings: compare complete stacks as
  well as individual deltas. A 1–2% setting gain remains eligible for the
  winner if it repeats in both orders and does not increase failures. The
  winning stack is judged against the original status quo, not against the
  immediately previous setting.
- High-risk changes that can affect physics, action timing, observations,
  resets, learner semantics, or episode outcomes require correctness and
  production-equivalence gates regardless of speed.

Any result inside noise stays in the candidate pool until it is combined with
other candidates or measured longer; it is not silently discarded. A setting
is written into the scheduled launcher only after the full stack beats the
status quo, preserves character count and outcome/recovery behavior, and
survives a longer stability run.

The Windows checkout is currently massively dirty and is not a clean checkout
of the Mac branch. It may be used for bounded diagnostics already present on
the trainer, but no new production training run is authorized from that state.
The safe deployment path is: commit the source on the Mac topic branch, push
it, pull/build on Windows in a separately verified checkout or reconciled
working tree, run the correctness gates, then start a Scheduled Task. This
contract therefore separates “measured on the dirty trainer runtime” from
“adopted for production training.”

### Takeover diagnostic result — scheduler/affinity

The first Tailscale trial exposed a flaw in the old sweep harness: its forced
Windows cleanup reused `/ogrl_sw14411` after killing the control engines. That
control exited before metrics were written, so its result is invalid. The
harness now makes the shared-memory prefix unique for every point, including
repeated n/k points. This is a safety/correctness fix, not a speed claim.

With the corrected harness, the same dirty Windows runtime, checkpoint
`run23_sel0.pt`, map corpus `t_train_101,t_train_102,t_train_104`, n14/k4,
update threads 4, and 45 s warmup + 90 s measurement produced:

| order | normal engine | above engine + `0xFFF` affinity | relative wall gain |
|---|---:|---:|---:|
| normal first | 730.05 wall steps/s | 941.53 wall steps/s | +29.0% |
| mixed first | 688.13 wall steps/s | 895.54 wall steps/s | +30.1% |

The raw remote summaries are
`Tools/rl/runs/throughput_sweep_takeover_unique_20260921.json` and
`Tools/rl/runs/throughput_sweep_takeover_reverse_20260921.json` on the trainer.
Both trials completed without an early exit. The engine's own per-worker logs
showed players 0 through 3 and `Num_threats: 3`, confirming the intended 1v3
scenario rather than an empty or 1v1 level. Pool misses were 0–1%; the trainer
had no remaining engine or trainer process after cleanup.

**Decision:** provisionally adopt `OGRL_ENGINE_PRIORITY=above` and
`OGRL_ENGINE_AFFINITY=0xFFF` as the next trainer benchmark configuration. This
passes the two-repeat repeatability screen, but it is not yet written into a
production Scheduled Task. The result was measured on the dirty Windows
checkout, without a thermal/CPU-placement capture, and still needs the full
n/k and update-thread sweep plus a longer stability run after clean source
deployment. It is already part of the cumulative optimization stack; do not
claim a final percentage until the complete stack is measured against status
quo.

### Continued takeover evidence — clean deployment, adoption threshold, and cooling

The dirty Windows checkout was not repaired or reset. A separate deployment was
created at `C:\ogrl\overgrowthRL_clean` from source commit `228a3bf3`, with the
existing Release binary copied in place and `run23_sel0.pt` copied only as the
resume input. The original dirty tree remains preserved. All measurements below
use the clean deployment, the same three-map diagnostic corpus
(`t_train_101`, `t_train_102`, `t_train_104`), a real resumed checkpoint, and
the trainer's own character-bearing logs. `n_envs + k_standby` is a multiple of
three for this corpus; the production six-map corpus is likewise compatible
with `n14/k4` because 18 is a multiple of six.

The earlier adoption question is resolved explicitly: there is no veto at 5%.
For a low-risk stack, a repeatable 0.5–1% gain above measurement uncertainty is
worth keeping; a 1–2% worker/thread/scheduler gain remains eligible when it
survives reversed order or a longer run; high-risk reset, physics, observation,
action, and learner changes still require correctness gates. The adoption test is
therefore cumulative: compare the complete candidate stack with the status quo,
then keep each component that is repeatable and behavior-neutral. Ten genuine
2% gains are treated as a real compounded improvement (about 21.9% if
independent), not rounded away as ten separate “too small” results.

#### Clean sustained throughput

| configuration | measured rows | median cycle steps/s | p10 | wall steps/s | pool misses | result |
|---|---:|---:|---:|---:|---:|---|
| status quo `n14/k4`, normal priority, no affinity | 59 | 697.5 | 671.1 | 700.1 | 0.284% | clean control |
| `n14/k4`, engine above-normal + `0xFFF`, threads 2/4/1 | 70 | 856.2 | 669.8 | 833.6 | 0.225% | **adopt candidate** |
| `n18/k3`, engine above-normal + `0xFFF`, threads 2/4/1 | 56 | 857.5 | 822.5 | 855.7 | 2.309% | not adopted yet |

Each long point used 60 seconds of warmup and 300 seconds of measurement and
completed without early exit. The safe resumable `n14/k4` stack is +22.7% on
median cycle rate and +19.1% on wall rate against the matched clean control.
`n18/k3` is not a production resume target: the checkpoint's reward normalizer
has one running accumulator per active worker, so changing from the saved
`n_envs=14` to 18 needs an explicit migration or a fresh n18 checkpoint. It also
has a higher pool-miss rate and no sustained advantage over n14. It stays as a
future fresh-run candidate rather than being forced onto `run23_sel0.pt`.

The adopted production candidate is therefore:

```text
n_envs=14, k_standby=4
OGRL_ENGINE_PRIORITY=above
OGRL_ENGINE_AFFINITY=0xFFF
collection torch threads=2, update torch threads=4, inter-op threads=1
hard-reset-every=20 for future unattended launches
```

The remote launcher now accepts these settings explicitly in
`Tools/rl/remote/launch_training.ps1`; the source change is to be committed and
deployed before starting an unattended Scheduled Task. The source change does
not change physics or action timing.

#### Thread and worker follow-ups

The clean short sweeps support `collection_threads=2`; collection-thread results
were t1=876.2, t2=894.8, t4=827.5, and t8=676.1 wall steps/s under the same
above/affinity stack. Update-thread=8 was not adopted: two repeat points varied
from 793.6 to 931.0 median steps/s, with the lower point's p10 at 649.8, while
the longer stable t4 point was 856.2. This is too variable to call a gain and
does not justify changing the known-stable t4 default. The raw sweep is kept at
`C:\ogrl\overgrowthRL_clean\Tools\rl\runs\throughput_sweep_update8_repeat_20260921.json`.

The valid worker/standby screen showed throughput rising through the available
workers, but short points were noisy and some diagnostic totals were invalid
for the map-corpus rule. The long n14/n18 comparison above is the adoption
evidence. Invalid totals, startup failures, and inter-op-thread failures remain
negative evidence, not silent omissions.

#### Cooling and power ceiling

The trainer is AC-powered and already uses the `High performance` plan with
minimum and maximum AC processor state at 100%, active cooling policy, aggressive
boost, energy-performance preference 0, and 100% core-parking minimum. No Dell
thermal-control service was present to select a stronger profile. During the
long n18 run Windows reported 90% maximum frequency, 90% performance limit, and
performance-limit flag `2`; Microsoft documents flag `0x2` as a power-safety
limit ([official definition](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/pepfx/ns-pepfx-_pep_ppm_query_perf_constraints)).
This is a laptop firmware/hardware ceiling, not an unmet grid-power setting.
The software cooling lever is exhausted; more load would risk throttling rather
than add useful training throughput.

#### Reset correctness boundary

The first soft-reset validation was rejected because `t_train_101.xml` produced
zero entities. The rerun used the valid `oval_arena_1v1_unarmed.xml` fixture and
confirmed: self id stable, 0/10 scenario-distribution mismatches, hard reset
median 1313 ms, soft reset median 187.5 ms, but replay/physics equivalence still
failed at step 0 (maximum position deviation 0.808, velocity deviation 4.954).
The result is consistent with the previously documented deep-sequence reset
anomaly and keeps soft reset behind the existing periodic hard-reset hedge. The
new scheduler/affinity adoption does not add this risk; it preserves the reset
mode already used by the status quo. A Windows RSS probe was also added to the
validator because its old Unix-only `ps` probe silently returned NaN on the
trainer; rerun the leak audit after that source change is deployed.

**Current decision.** Adopt the clean `n14/k4` plus above-normal engine priority,
`0xFFF` affinity, collection=2/update=4/inter-op=1 stack for the next bounded
resume smoke and, if it remains healthy, the unattended run. Do not adopt t8,
n18 resume, inter-op changes, or any new soft-reset semantics. Any long run must
still be started by Scheduled Task, with checkpoint monotonicity and run-specific
shared-memory names intact.

#### Validator correction after deployment

The Windows RSS probe was deployed at `2c93f36f` and rerun on the valid oval
fixture for 100 soft resets. Working set rose from 486.9 MB to 510.6 MB, a
23.7 MB increase below the 50 MB ceiling; self id stayed 2 with zero measured
object-id growth. Hard reset median was 1297.0 ms and soft reset median 203.0
ms; scenario distribution mismatches were 0/10. Replay/physics equivalence
still failed at step 0 with position deviation 0.80832 and velocity deviation
4.95399. The memory gate passes; the replay gate remains unresolved and soft
reset is not a newly adopted lever.

#### Further lever coverage — reset interval and CPU mask

The reset interval was parameterized in the sweep harness and compared under
the same n14/k4, above/affinity, threads 2/4/1 stack. Hard-reset-every=20
completed at 878.0 wall steps/s (median cycle 874.3, p10 841.7, pool misses
0.78%). A first 50 point failed startup and was discarded; a fresh retry
completed at 886.0 wall steps/s (median cycle 896.9, p10 846.2, zero pool
misses). The short-run speed edge is about 0.9% wall and is not enough to
override the correctness reason for the safer 20-reset hedge.

The physical logical-CPU mask screen, using hard-reset-every=20, produced
`0x3FF`=803.2, `0x7FF`=839.8, `0xFFF`=810.5, and `0x3FFF`=790.6 wall steps/s.
A longer reversed repeat gave `0xFFF`=786.8 and `0x7FF`=783.3 wall steps/s.
The apparent short `0x7FF` edge did not repeat; keep `0xFFF` and leave the
mask decomposition closed unless a thermal-state-controlled run changes the
result.

Defender inspection found real-time scanning enabled but path exclusions already
cover `C:\ogrl` and the purchased Overgrowth install, with the relevant build,
engine, and Python processes excluded. No blanket Defender disable is justified.

This closes the first low-risk throughput inventory: clean worker/standby,
collector and update threads, engine priority/affinity, reset interval,
shared-memory cleanup, clean deployment, Defender exclusions, and software
power-plan settings have all been measured or inspected. Remaining meaningful
levers are a fresh-checkpoint n18 run/migration, component-level scheduler
repeat under captured thermals, evaluation-overhead validation, decision-rate
experiments with gameplay gates, and rebuilt engine-side early-outs. They are
not declared exhausted or adopted from a speed number alone.

#### Scheduler component decomposition and unattended handoff

At n14/k4 with hard-reset-every=20, the short component screen measured
normal/no-affinity 708.4, above/no-affinity 825.0, normal+`0xFFF` 733.2, and
above+`0xFFF` 816.3 wall steps/s. This indicates that priority is the dominant
component; affinity was not a reliably additive gain in this short order. The
longer above+`0xFFF` stability result remains the deployment basis because it
has the stronger measurement window and lower demonstrated risk than changing
the production mask from one short screen.

After the bounded smoke completed at step 264,001,244, a durable Scheduled
Task `OGRL_Train_run25_optimized_20260921` was launched from the clean checkout
using its new checkpoint, six maps, n14/k4, above priority, `0xFFF`, threads
2/4/1, hard-reset-every=20, and a 320M target. The first observed update was
global step 264,022,748 at about 796 cycle steps/s, with zero NaN skips, no KL
spike, no pool misses, and `mb0_max_abs_logratio=2.48e-5`. The task is designed
to survive SSH disconnect and logoff; its progress is monitored from telemetry,
not from the agent session.

#### Windows-only benchmark freeze

The apples-to-apples benchmark protocol is frozen in the outer artifact
`research-artifacts/OGRL-20260921-005-benchmark/`. Policy quality uses the
paired `Tools/rl/benchmark_compare.py` harness, which joins per-episode results
by held-out seed and reports a bootstrap interval. It must run on Windows for
the Windows claim; a Mac trial was aborted and is not evidence for trainer
throughput.

The accepted speed evidence is narrower than a historical total-system claim:
on the same Windows Release binary (SHA-256
`6985E73DD06C572D8F474E53ADF1D20D03ED039352D70EA904CCCD3A89D769D0`) and the
same three-map diagnostic cell, clean normal/no-affinity measured 700.120 wall
steps/s and above+`0xFFF` measured 833.627 wall steps/s. The optimized stack is
adopted provisionally. The old `715 active SPS` number is not ratioed against
833.627 because its denominator is not proven identical.

The production confirmation is still required after an idle window: a
six-map, n14/k4 ABBA sequence and reversed-order repeat, with wall
`delta_global_step / monotonic_elapsed_seconds` as the primary metric and
cycle median/p10, phase timings, reset latency, pool misses, CPU placement,
and thermal performance-limit samples as diagnostics. Historical `run23_sel0`
telemetry is preserved as an anchor but is not a causal A/B because it used
three maps, collection threads=1, and hard-reset-every=50; current run25 uses
six maps, threads=2/4/1, and hard-reset-every=20.

#### OGRL-20260921-006 — optimization-only correction and collector screen

The scheduled run above was launched before this takeover turn's optimization-
only boundary was made explicit. It was stopped cleanly over Tailscale, not
force-killed. The first Windows control write used PowerShell's UTF-8 BOM;
Python's `json.loads(read_text())` rejected that file and continued polling
normally. Rewriting the control file as BOM-free UTF-8 produced the expected
`stop_requested` and `run_stop` events. The completed manifest records
`final_global_step=268402396`. The task is `Ready`, and the trainer has no
remaining `python.exe` or `Overgrowth.exe` process. No regular training is
authorized or running after this point; all following work in this phase is
measurement, source audit, or harness work.

The repaired `$chatgpt-advisor` skill was tested through
`skill/bin/chatgpt-advisor`, not the private driver. `doctor` passed with the
dedicated advisor tab, login, composer, and High reasoning available. A real
`ask` uploaded four files using four repeated `--file` options; the returned
JSON had `ok=true`, `state=COMPLETE`, and explicitly distinguished all four
preserved paths. Session `20260921-165327-986529` completed in 351.5 seconds.
The advisor's stale-state warning was correct for the evidence bundle's time
of capture and is superseded by the stop proof above. The old direct-driver
path was not used.

The existing `throughput_sweep.py` is barred in optimization-only mode: it
launches `train_vec.py` with `--n-epochs 1` and performs PPO updates. A
separate collector-only screen was therefore run after the stop gate. The
six-map worker/standby screen tested n=1/2/4/6/8/10 with k=0/2, act-period 4,
8-second warmup and 25-second measurement. The complete normal/no-affinity
arm peaked at n8/k2 = 488.3 decisions/s; n10/k2 fell to 402.4. AboveNormal
without affinity reached 515.5 at n6/k2 but was order-dependent and later
failed to launch n8/k2. AboveNormal plus `0xFFF` reached 434.5 at n2/k2 and
failed at n6/k0. These candidate arms are preserved as incomplete negative
evidence, not adoption evidence. The robust result is that k=2 helped the
collector screen and too many active workers hurt; this cannot be transplanted
to a PPO resume because reward-normalizer state is per active worker.

A frozen Windows policy forward microbenchmark used the run24 checkpoint and
fresh processes: 1/2/4 intra-op threads produced 3924.3/3991.6/3958.4
forward calls/s. The +1.7% two-thread burst is below a causal adoption claim;
five repeated in-process changes converged near 1.81–1.82k for all settings.
The existing collection=2 setting remains reasonable and no thread setting is
changed from this test.

The Windows trainer is already at the software cooling ceiling: High
performance, AC min/max processor state 100%, active cooling, aggressive
boost, EPP 0, and 100% core parking. Captures reported 90% maximum frequency
and 90% performance limit. This is consistent with the documented firmware
power-safety ceiling; no stronger software cooling lever was found.

The collector harness now accepts an explicit map corpus and records a launch
failure as a point-level error rather than aborting the entire screen. This
source-only change is nested commit `e2f9cbb9`, deployed to the clean Windows
checkout; it does not alter engine or learner semantics. Raw evidence and the
next no-training levers are in
`research-artifacts/OGRL-20260921-006-optimization/README.md`.

Decision after this experiment: keep the prior n14/k4 + above/`0xFFF`
configuration only as a provisional benchmark candidate from the earlier
longer matched diagnostic; do not claim it as final, do not launch training,
and do not broaden it based on the incomplete collector screens. The next
safe test is an interleaved, thermal-captured, frozen-policy collector ABBA
that separates priority from affinity and maps actual 165U EfficiencyClass
before selecting a CPU mask.

#### Topology follow-up

`GetSystemCpuSetInformation` resolved the 165U mask instead of leaving
`0xFFF` as an assumption: logical 0/1 and 10/11 are EfficiencyClass 1 on
physical cores 0 and 10 (the two hyper-threaded P-cores); logical 2–9 are
EfficiencyClass 0 E-cores; logical 12–13 are EfficiencyClass 0 LP-E-cores.
Microsoft defines the higher EfficiencyClass as faster but less power-efficient
([SYSTEM_CPU_SET_INFORMATION](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)).
Therefore `0xFFF` includes both P-cores and all eight regular E-cores while
excluding the two LP-E-cores; it is not a P-core-only mask. The untested P-core
mask is `0xC03` (bits 0,1,10,11). It remains a diagnostic candidate only:
restricting 14 engines to two physical P-cores may worsen contention, and it
must be tested in an interleaved thermal ABBA before any adoption decision.

#### Corrected SHM namespace and fresh affinity comparison

The first mask comparison reused its shared-memory prefix across separate
invocations. The failed `0xFFF` launch and the very low P-only point from that
sequence are preserved, but neither is a CPU result. The collector harness was
corrected in `407ff261` to namespace every invocation and sweep point.

With the corrected harness, a fresh six-map n6/k2 collector sequence used
act-period 4, a 1200-step cap, 5 seconds of warmup, and 40 seconds of
measurement. Normal/all (`0x3FFF`) produced 553.8 decisions/s; AboveNormal/all
produced 476.9; AboveNormal+`0xFFF` produced 479.4; and AboveNormal+P-only
`0xC03` produced 583.2. A separate fresh C/D pair reproduced 490.8 versus
572.5. This is a repeated collector-only signal for P-only affinity, but the
absolute values remain sensitive to order and thermal state and the test has no
learner. The next safe action is to run a frozen-policy collector benchmark
with `0xC03`; no training launcher, PPO resume, checkpoint, or reward state is
changed by this result.

#### Frozen-policy confirmation and worker knee

Nested commit `8ce0df70` adds a benchmark-only collector that loads a frozen
checkpoint and its observation normalizer, runs the real policy forward pass,
and steps the real environments. It creates no optimizer, performs no PPO
update, and writes no checkpoint. On the six-map corpus at n6/k2, frame-stack
4, act-period 4, AboveNormal, the mask results were normal/all `0x3FFF` 407.0
decisions/s, AboveNormal/all 421.5, AboveNormal+`0xFFF` 439.0, and
AboveNormal+P-only `0xC03` 463.9. Policy inference was only 8.1% of the P-only
wall time, so the mask direction survives learner-shaped action generation.

Under P-only with 4 Torch threads, the active-worker screen measured n1=292.5,
n2=413.4, n4=438.5, n6=487.5, n8=505.2, and n10=489.0 decisions/s. A
separate thread screen at n6/k2 measured threads 1=509.2, threads 2=494.8 on
retry after one startup failure, and threads 4=536.2. The current safe
optimization candidate is n8/k2, P-only `0xC03`, AboveNormal, Torch threads 4,
inter-op 1. This candidate is for the next learner-shaped benchmark only; it
does not alter a PPO default, resume checkpoint, worker-normalizer state, or
training task.

#### No-checkpoint PPO throughput correction

The user clarified that optimization probes may run the real PPO loop and
update policy parameters in memory, provided test checkpoints are never
written. Nested commit `78d42e67` adds `throughput_sweep.py --no-checkpoint`:
the resume checkpoint is read-only, no output checkpoint path is passed, and
the probe records `checkpoint_written=false`.

This resolves the apparent 505-versus-890 discrepancy. The frozen-policy n8
collector number was not a replacement for training throughput. A matched
real-PPO n14/k4 six-map probe produced 719.4 wall SPS for normal control and
923.2 for AboveNormal+`0xFFF`, the same regime as the historical 890+ points.

The real PPO worker screen showed n10/k2=777.9, n14/k4=830.4, n16/k2=836.4,
and n18/k0=672.9 wall SPS; n18/k0 had 100% pool misses. Map-aligned n18/k6
then produced 896.6 and a fresh retry produced 964.1 wall SPS, with p10=910.3
and zero pool misses. The forward n14/k4 versus n18/k6 block measured 820.9
versus 867.6 wall SPS. The reverse block failed in startup and is retained as
harness evidence, not a speed result.

Decision: the current real-PPO optimization candidate is n18/k6,
AboveNormal+`0xFFF`, Torch threads 2/4/1. This supersedes the frozen n8/505
collector candidate. It is still a bounded benchmark setting; a clean reverse
repeat, thermal capture, correctness gate, and policy-quality gate remain
before changing any unattended training launcher.

An isolated n14/k4 rerun after n18/k6 produced 903.5 wall SPS, while the
fresh n18/k6 retry produced 964.1. A live n18/k6 Torch-thread screen then
measured threads 1=956.0 wall SPS, threads 2=822.4 in a warmer later slot,
and threads 4 failed during startup. The earlier clean threads-2 point was
964.1, so the thread result is order/thermal-sensitive; retain the established
2/4/1 stack rather than switching to threads 4.

The thermal-captured n18/k6 attempt had one of 24 engines remain silent during
standby initialization and produced no speed rows. The sampler recorded High
Performance, 90% maximum frequency, 90% performance limit, and 23 live engines
before cleanup. Preserve this as a startup-reliability failure, not as a
throughput result or a reason to discard the 964.1 SPS n18/k6 candidate.

#### Async collector result

The asynchronous collector hypothesis was tested with the same frozen policy
and n8/P-only/AboveNormal/Torch-4 stack. Synchronous n8/k2 produced 464.3
decisions/s over a 30-second measurement; asynchronous n8/k0 with rollout size
8 produced 291.1. The async path is 37.3% slower on this host/configuration,
so it is rejected for the current optimization candidate. It remains in the
source and is available for a later heterogeneous-cost experiment; no async
implementation was deleted or silently replaced.

#### PPO batching sweep

Nested commit `c8f9f277` exposes rollout horizon, PPO epochs, and minibatch
size to the no-checkpoint harness while preserving the existing defaults.
On n18/k6, 512 steps/128 minibatch/1 epoch measured 1,000.4 wall SPS;
1024/128/1 measured 790.6; and 512/256/1 measured 856.5. All had zero pool
misses. Use 512/128/1 for subsequent optimization probes, but do not change a
long-run training default until policy-quality and sample-efficiency checks
are complete.

#### Thread and affinity follow-up

At the 512/128/1 learner point, collection Torch thread-1 measured 996.0 wall
SPS, effectively tied with thread-2 at 1,000.4. Thread-4 failed during reset
because one live engine published no observation for 120 seconds. Expanding
the engine affinity from `0xFFF` to all logical CPUs (`0x3FFF`) measured 971.5
wall SPS, 2.9% slower. Retain collection threads 2 and `0xFFF`; record
thread-4 as a startup reliability failure, not a throughput result.

#### PPO update-thread sweep

At n18/k6 with 512/128/1, update threads 2 measured 973.2 wall SPS, update
threads 4 measured 1,000.4, and update threads 8 measured 886.8. Valid points
had zero pool misses; the first update-2 launch failed at startup. Retain
update threads 4 and reject 2 and 8 for this host.

#### Sustained thermal benchmark

With 60 seconds warmup and 300 seconds measurement, the successful n18/k6
retry measured 903.8 wall SPS (median 917.7, p10 825.6, zero pool misses).
The same-duration n14/k4 retry measured 670.8 wall SPS (median 667.8, p10
639.2, 0.56% pool misses), a sustained +34.7% gain. Each first launch failed
during startup and is recorded separately. Treat 903.8 versus 670.8 as the
robust comparison; 987–1,006 SPS are short-window peaks.

#### Reset-policy sweep

On n18/k6 with the established AboveNormal/`0xFFF` and Torch 2/4/1 stack,
hard-reset-every 50 measured 957.9 wall SPS, while disabling periodic hard
resets measured 821.0 wall SPS. Both had zero pool misses; reset-50 had one
startup retry. The established hard-reset-every 20 setting remains the fastest
observed and retains the cleanup safety valve. Reject reset-0 and reset-50 as
adoption changes; preserve hard-reset-every 20.

#### Clean repeat and aligned worker sweep

After correcting the resume path to the clean checkout, n18/k6 completed a
45-second warmup plus 120-second real-PPO/no-checkpoint measurement at
1,006.4 wall SPS (median 1,011.6, p10 928.0), with zero pool misses. An
ordered n14/k4 then n18/k6 repeat then measured n14/k4=929.3 wall SPS and
n18/k6=854.5, both with zero misses. Thus n18/k6 has the higher observed
ceiling but is not a guaranteed winner in every thermal/order slot.

At the same 24-engine aligned total, n20/k4 measured 869.4 wall SPS with no
pool misses and n22/k2 measured 832.9 with a 16.3% pool-miss rate. More active
workers do not improve this host. Keep n18/k6 as the maximum-throughput
candidate and n14/k4 as the stability fallback; do not promote n20/k4 or
n22/k2. All probes used in-memory PPO updates and recorded
`checkpoint_written=false`.

#### Status-quo versus optimized benchmark

With the same six maps, 45-second warmup, 90-second measurement, 512/128/1
PPO batching, hard-reset-every 20, update threads 4, inter-op 1, and no
checkpoint output, the valid status-quo retry (n14/k4, default priority and
affinity) measured 720.8 wall SPS. The optimized n18/k6 point with AboveNormal
and `0xFFF` measured 987.5 wall SPS, a 37.0% gain, with zero pool misses in
both. The first baseline launch failed at startup and is excluded from the
comparison. Treat n18/k6 + AboveNormal + `0xFFF` + 512/128/1 as the peak
optimization candidate; n14/k4 remains the stability fallback pending
multi-hour thermal and policy-quality gates.

#### Trainer-process priority check

Setting `OGRL_TRAINER_PRIORITY=above` had one startup failure; a retry measured
1,001.5 wall SPS, effectively tied with normal priority at 1,000.4. An
interleaved follow-up had the normal arm at 702.3 wall SPS and the AboveNormal
arm failed during reset. This is not a stable causal gain. Keep the trainer at
normal priority and retain only engine AboveNormal/`0xFFF`.
## Advisor audit: remaining optimization and correctness work (2026-09-21)

The repaired `chatgpt-advisor` skill was given the implementation contract, operating
manual, recent source history, benchmark artifacts, and the sustained n14/n18 results.
Its conclusion is that the current n18/k6 candidate is a real improvement, but the
optimization is not exhausted and the harness is not yet ready for an unattended
production-training default.

Observed evidence accepted by the audit:

- Sustained n18/k6: 903.8 wall decisions/s after 60 s warmup and 300 s measurement,
  zero pool misses.
- Sustained n14/k4: 670.8 wall decisions/s under the same protocol, with 0.56% pool
  misses. The robust observed gain is therefore +34.7%.
- Short-window peaks near 987-1,006 decisions/s are not the sustained headline.
- n18/k6 requires six standby engines to avoid the n18/k0 starvation regime. The
  n18/k6 + engine AboveNormal + `0xFFF` + Torch 2/4/1 stack remains the leading
  systems candidate, while n14/k4 remains the stability fallback.
- 512 rollout steps / 128 minibatch / 1 PPO epoch is a useful systems probe, but is
  not behavior-neutral relative to the default 256/256/4 PPO schedule. It must not
  become the production training default without policy-quality and sample-efficiency
  evidence.

P0 work required before adoption or further claims:

1. Repair benchmark accounting. `throughput_sweep.py` advances and scores from
   `global_step` before recovery validity is established, so reported wall SPS can
   include invalid/recovered transitions. Add `valid_transition_count`, useful SPS,
   recovery counts, per-map episode accounting, and post-ready timing. Separate
   recovery telemetry from legitimate timeout/outcome telemetry.
2. Repair startup reliability. The 24-engine configurations intermittently leave an
   engine alive but silent for 120 seconds. Replace simultaneous construction with
   staged, map-balanced launch waves (six at a time), per-engine timeouts, explicit
   ready counts, fresh IPC prefixes, retry counts, and exception-safe cleanup. A
   speed point is not adoptable unless its startup/recovery ledger is clean.
3. Fix the readiness/reset boundary. The first reset currently consumes a natural
   startup observation instead of proving that the requested seed, difficulty,
   opponents, and scenario were applied. Initialize to ready, then issue and verify
   the requested reset before marking a worker ready.
4. Audit worker-count migration. `throughput_sweep.py` currently sets
   `OGRL_ALLOW_NENVS_CHANGE=1`; that is not by itself a documented or verified
   reward-normalizer migration. Prove the normalizer state mapping or forbid worker
   changes on resume.
5. Complete the learner audit with `Tools/rl/ppo/train.py` and
   `Tools/rl/ppo/vec_buffer.py`, then run the exact in-memory PPO path separately
   from the environment-only speed instrument.

Remaining safe levers, in priority order:

- Windows multi-object wait for observation semaphores, feature-gated and preserving
  bounded recovery semantics and worker ordering.
- A NumPy shared-memory fast path that avoids `np.frombuffer(...).tolist()` followed
  by list-to-array conversion, with owned copies, finite checks, and a preallocated
  four-frame ring buffer.
- Standby placement experiments: active engines AboveNormal/`0xFFF`; standby engines
  Normal on LP-E or regular E cores, restoring the active mask/priority before ready.
- Trainer hard process-affinity/priority was tested and rejected by the ABBA
  follow-up; no CPU-set preference is adopted. Do not move engine affinity away
  from the validated `0xFFF` without measurement.
- Verify the active Torch runtime with `torch.__config__.parallel_info()` and test
  fixed versus dynamic thread settings rather than relying on environment folklore.
- Verify HighQoS/execution-speed power state. The host already reports High
  Performance, EPP 0, active cooling, and 100% core parking, so expected gain is low.
- Use ETW/WPR or an equivalent profile to identify engine/AngelScript hotspots before
  editing gameplay-adjacent code.

Do not adopt: act-period reduction, physics/Bullet/contact changes, weakened
non-finite checks, disabled hard resets, blanket Defender disablement, High or
Realtime priority, thermal/BIOS-limit defeat, or a revived async collector. The
current async probe was 37.3% slower than synchronous collection on the tested host.

Advisor consultation status: completed successfully after one recoverable `NO_TAB`
response and stale-lock cleanup; the repaired multi-file upload path worked. No source
implementation was changed by the consultation itself. The next falsifiable gate is
the repaired cold-start and useful-SPS harness, followed by paired post-ready n14/n18
and lever-factorial runs.

## Measurement-harness implementation pass (nested commit deffca9c)

The first implementation pass after the audit is source-only infrastructure; it does
not change gameplay, physics, action timing, reward semantics, or checkpoint output.

- `OvergrowthEnv.reset()` now drains the natural post-load observation and then issues
  the caller's requested reset even on the first call. The initial unrequested
  episode is no longer counted as a training episode.
- `VecOvergrowthEnv` launches engines in bounded waves of six by default, with
  `OGRL_LAUNCH_WAVE_SIZE=0` retained as an explicit all-at-once comparison. A partial
  constructor failure closes already-built engines.
- Failed engine connection/schema/affinity launches now clean up their process and
  write directory before raising. Windows affinity application checks the API return,
  reads the applied mask back, and logs the PID/mask.
- The trainer emits `[RL_READY]` only after active workers have completed the requested
  initial reset. `throughput_sweep.py` waits for that barrier and starts warmup after
  it, rather than charging engine startup against the timing window.
- Recovery transitions are excluded from win/loss/timeout and curriculum episode
  accounting. Per-update telemetry now includes recoveries, valid transitions, and
  recovered transitions; the sweep reports useful wall SPS separately from raw PPO
  global-step SPS.
- Checkpoint worker-count migration is now explicit through
  `--allow-n-envs-change`; the sweep passes that flag instead of silently injecting
  `OGRL_ALLOW_NENVS_CHANGE`.

Static checks and the 11-test checkpoint-safety suite pass. A Windows/Tailscale
no-checkpoint smoke run at n2/k1 reached `[RL_READY]` in 5.0 s, produced 14 measured
rows, 7,168 valid transitions before boundary-row correction, zero recoveries, and
no checkpoint. The corrected cumulative accounting excludes the first boundary row;
the next n14/n18 comparison is the first benchmark of this harness revision.

## Standby LP-E placement (paused partial)

Nested commit `35829ca3` adds per-engine Windows priority/affinity overrides for
standby processes only. Active n18 workers retain AboveNormal and `0xFFF`;
`OGRL_STANDBY_PRIORITY` and `OGRL_STANDBY_AFFINITY` can place reset standbys on
the two LP-E logical CPUs (`0x3000`) without changing engine code or gameplay.

The forward baseline arm completed at 1,018.562 useful transitions/s, with zero
recoveries and zero pool misses, under the same six-map, 512/128/1, Torch 2/4/1,
60 s post-ready warmup, 120 s measurement, no-checkpoint protocol. The LP-E
standby arm reached `[RL_READY]` and its logs verified six standby processes
with applied mask `0x3000`, but was intentionally interrupted by the owner
after four update rows while relocating the computer. It has no valid measured
throughput and was not adopted. Resume with a fresh paired ABBA comparison.

## 2026-09-22 takeover: corrected placement, worker depth, threads, and ETW

The repaired source is nested commit `da4c07ce`, deployed to the clean Windows
checkout before these probes. It fixes two correctness issues identified by the
advisor audit: Windows `_sem_wait` now distinguishes `WAIT_OBJECT_0`,
`WAIT_TIMEOUT`, and `WAIT_FAILED` instead of treating a failed wait as success;
and a standby promoted into an active vector slot is restored to the active
priority/affinity while the retired engine is moved to the standby role before
its background reset. A two-active/one-standby smoke run logged the full
launch -> promote -> retire transitions and completed with zero recoveries.

### Standby placement ABBA

The previously paused LP-E experiment was not evidence because the old source
left promoted engines on the LP-E mask. The corrected n18/k6 real-PPO ABBA
tested active AboveNormal/`0xFFF`, standby Normal/`0x3000`, six maps,
512/128/1, Torch 2/4/1, hard-reset-every 20, and no checkpoint output:

| order | normal standby useful SPS | LP-E standby result | recovery/pool result |
|---|---:|---:|---|
| normal -> LP-E | 779.1 | 137.0 | 0 / 0 in both |
| LP-E -> normal | 741.9 | 159.8 row median* | 0 / 0 in both |

*The LP-E reverse arm produced only one post-boundary row because its cycles
were 57--61 seconds; its cumulative useful-SPS field is zero after the harness
correctly drops the boundary row. The per-row value is still sufficient to
show the severe slowdown. The placement is rejected: six standby resets
contend on the two LP-E logical processors and create about 22--23 seconds of
barrier idle per worker. The role-transition implementation is retained for
future targeted experiments but both placement variables remain default-off.

Regular-E-only standby placement (`0x3FC`) was also screened. Forward order
was 796.5 normal versus 678.8 regular-E useful SPS; reverse order had a
regular-E startup failure and a 753.4 normal result. It is rejected and the
placement family is closed for this host. The raw ABBA summaries and logs are
under `research-artifacts/OGRL-20260921-006-optimization/telemetry/` in the
Mac outer repository.

### Worker/standby depth

An n18 depth screen (`k=2..8`, six maps, no checkpoint) was completed. The
map-axis rule still governs adoption: with six maps, active plus standby
engines must be a multiple of six. Thus most n18/k points are diagnostics,
not canonical training candidates. Useful wall SPS in the measured order was

| n18/k | useful SPS | pool-miss rate | status |
|---:|---:|---:|---|
| 2 | startup failure | -- | reject |
| 3 | 771.2 | 3.31% | diagnostic/map-skewed |
| 4 | 799.9 / 755.0 repeat | 1.47% / 0% | diagnostic/map-skewed |
| 5 | 732.5 | 0% | diagnostic/map-skewed |
| 6 | 695.0 | 0% | valid map-aligned candidate |
| 7 | 692.0 | 0% | diagnostic/map-skewed |
| 8 | 648.0 | 0% | diagnostic/map-skewed |

The sequence is strongly order/thermal-sensitive, so it does not overturn the
two-order post-repair n14/n18 result or the sustained n18/k6 result. n18/k6
remains the only valid six-map member of this local family and the adopted
throughput candidate; no worker-count default was changed.

### Fine learner-thread screen

At n18/k6 with the same no-checkpoint protocol, update threads 3 and 4
produced 787.8 and 801.3 useful SPS respectively, both with zero recoveries
and pool misses. The 1.7% difference is within the observed thermal/order
spread and does not beat the adoption threshold. Keep update threads 4;
collection 2 and inter-op 1 remain unchanged.

### Power and ETW evidence

The trainer is already at the software cooling ceiling: High Performance,
AC minimum and maximum processor state 100%, Active cooling, Aggressive boost,
and Defender path exclusions for `C:\\ogrl` and the Overgrowth install. No
power-plan mutation was necessary. The raw WPR CPU trace is retained on the
trainer at `C:\\ogrl\\wpr\\wpr_candidate_20260922.etl` (1,733,296,128 bytes,
SHA-256 `FFDB168861FA57505E29E82A702182617A2FD0CCE1DD0F7DCE3E212053B77627`).
`tracerpt` processed its 217-second trace with 17,763,758 events, zero lost,
1,546,605 CPU samples, and 9,935,057 stack-walk events. The local summary is
`telemetry/wpr/wpr_candidate_20260922_summary.txt`; WPA/xperf is not installed,
so no function-level hotspot claim is made. The next high-value engineering
lever is an engine-side profile/hot-path change backed by this trace or an
equivalent analyzer, not another blind affinity permutation.

### Current decision

The optimization candidate remains n18/k6, engine AboveNormal with affinity
`0xFFF`, trainer normal priority, Torch 2/4/1, hard-reset-every 20, and the
512/128/1 learner-shaped benchmark settings. All 2026-09-22 probes used
in-memory PPO updates and `checkpoint_written=false`. No unattended training
task was started, no gameplay/physics semantics were changed, and no test
checkpoint was written.

## Per-map transition telemetry

The global standby pool deliberately lets an engine keep its own map while it
swaps vector slots. That is fast, but startup engine-count alignment alone does
not prove that transitions remain balanced across the six-map corpus over a long
run. Nested source now records valid and recovered transition counts keyed by
the actual `info["level"]` in every PPO update. This is telemetry only; it does
not change pool scheduling, map assignment, resets, or gameplay. A future
training gate must inspect these counters before accepting a long run. The
implementation passed `py_compile` and the 11-test checkpoint-safety suite.

## OGRL-20260922-006 — collector/engine follow-through and OEM thermal profile

### Current n18/k6 throughput and line-cue build

The current stable reference remains the Release x64 status-quo engine with
line cues enabled, n18/k6, engine AboveNormal/`0xFFF`, trainer Normal, Torch
2/4/1, hard-reset-every 20, six maps, and the in-memory 512/128/1 PPO probe.
It is not the production PPO recipe and never writes checkpoints. The previous
five-minute sustained record remains **903.840 useful wall transitions/s**
(60 s warmup + 300 s measurement, zero pool misses). The fresh matched control
arms below averaged 896.860, consistent with that record within normal thermal
variation; short peaks near 1,000/s are not sustained evidence.

An ABBA compared the new benchmark-only AngelScript no-line-cue binary with the
status-quo binary. All four completed arms had zero recoveries and zero pool
misses; each full measured arm contained 267,264 useful transitions except the
slower B2 arm at 239,616.

| arm | binary SHA-256 | useful wall SPS |
|---|---|---:|
| A1 control | `19EAB598E098A2C66CDA190675C8278636598D1DC1A929829CA709B43B854C13` | 897.052 |
| B1 no-line-cue | `F8EFFE0C8D84AF91A82BE8D4C2BF396C7F902D523769804959471997832BBA59` | 920.692 |
| B2 no-line-cue | same B binary | 842.092 |
| A2 control retry | same A binary | 896.669 |

Pooled B averaged 881.392 versus A 896.860 (−1.7%). The effect changes size
with run order and is not repeatable as a gain; keep the feature gated and do
not adopt it for speed. The initial A2 launch failed before readiness and has
no metric; the uniquely tagged A2 retry is the value above. The engine-side
replay test was exact for 140/140 comparisons across maps 101–105. Map 106 has
baseline same-seed divergence, and the attack-event side log was empty; therefore
this is not yet a complete attack-event equivalence certificate across all six
maps. Raw build/run files are still on the trainer under `C:\ogrl\optimization`
and its clean checkout run directory; fetch and archive them after remote access
returns.

### Batch wait and frame-stack preallocation

The existing untracked evidence in
`research-artifacts/OGRL-20260921-006-optimization/telemetry/` contains two
real-PPO no-checkpoint ABBA screens that were not yet transcribed into this
contract. Both tested n18/k6, the same six-map/timing stack, and zero
recoveries/pool misses. Their per-arm summary JSONs do not record their source
commit or full window settings, so retain those provenance gaps rather than
backfilling assumptions.

| lever | forward A control | forward B | reverse B | reverse A | pooled result |
|---|---:|---:|---:|---:|---|
| `OGRL_BATCH_WAIT=1` | 922.656 | 819.280 | 797.394 | 844.392 | 808.337 vs 883.524 SPS; −8.5% |
| `OGRL_PREALLOC_FRAME_STACK=1` | 958.481 | 827.912 | 778.554 | 804.235 | 803.233 vs 881.358 SPS; −8.9% |

Both alternatives lost in each paired ordering and are rejected for this host.
Keep both default-off; preserve the implementations for future profiling, not
as active speed settings. Separately, the SHM ndarray fast path ABBA remains
near tied (839.5 vs 842.0 SPS with the sign reversing by order); it is not an
adopted gain either.

### Dell thermal lever: staged, not yet verified active

This corrects the earlier statement that firmware cooling was exhausted. The
trainer had only been shown to use Windows High Performance plus its stock Dell
`Optimized` thermal profile; DCC had not been installed, so OEM UltraPerformance
was still an untested lever. Dell's own DCC guide lists `Optimized`, `Cool`,
`Quiet`, and `UltraPerformance`; the Latitude 7450 manual confirms
`Optimized` is its default. This is an OEM profile, not a firmware-limit
override.

Downloaded Dell Command Configure 5.2.2.292, SHA-256
`bac829a34b53eaa98afa10c033646932db9f9775d285799180d5bec30fe90225`; the
package and extracted MSI both passed Authenticode validation. Installed the
Dell utility with restart suppressed, read `ThermalManagement=Optimized`, set
only `ThermalManagement=UltraPerformance`, and read back that value with exit
code 0. Then verified no Overgrowth/Python process, confirmed the run24 resume
checkpoint SHA-256 was still
`1F98963795CB5D1123EE5AD8A51DF870DF917B387205F3D51E0C36635CE4D88D`, and
disabled six expired one-shot tasks (run24/run25 training and FB1–FB4 eval) to
prevent an unintended launch during restart. Those task changes are reversible.

The idle trainer was rebooted to apply the OEM setting. At the last check,
Tailscale marked `100.118.2.91` offline and SSH port 22 was also unavailable
over the configured LAN address. Therefore the post-reboot mode, device health,
and any throughput effect are **unverified**. Do not claim UltraPerformance is
active or count it as an adopted gain until SSH returns, CCTK reads the profile
back, and the six-map n18/k6 benchmark is repeated. Do not modify voltage,
firmware power limits, fan overrides, or thermal safety controls.

### Benchmark harness hardening

Uncommitted source changes in `Tools/rl/throughput_sweep.py` now request
`control.json` stop at an update boundary (BOM-free UTF-8, atomic replace),
wait for a clean `completed` manifest, use bounded terminate/kill only as a
fallback, clip measurement rows to the requested window, mark invalid starts
as failures, and refuse to overwrite prior run or checkpoint artifacts. This
preserves no-checkpoint probe semantics and avoids stale `running` manifests.
Three unit tests for clean stop, fallback, and control-file encoding pass, as
do `py_compile`, `git diff --check`, and all 11 checkpoint-safety tests. It has
not yet been deployed to the trainer because SSH went offline during the OEM
reboot.

### Finite lever-coverage accounting

For an honest progress percentage, this contract tracks 39 concrete
optimization families—not every possible future optimization. “Screened” means
the family has an evidence-backed setting decision or explicit inspection;
“partial” means evidence exists but the causal/end-to-end gate is incomplete;
“open” means no adequate test has run.

| coverage | count | families |
|---|---:|---|
| Screened or inspected | 16/39 (41%) | engine priority; active affinity; trainer scheduling; standby placement; collection threads; reset cadence; soft-reset fidelity; sync vs async; SHM ndarray path; Win32 batch wait; preallocated frame stack; null-sound early-out; line-cue suppression; Defender exclusions; Windows AC power plan; CPU topology/masks |
| Partial | 18/39 (46%) | active-worker count; standby depth; six-map/per-map balance; update threads; inter-op threads; rollout horizon; minibatch/epochs; inference mode; observation/finite-check path; reward/entity bookkeeping; ctypes signatures; headless-path delta; global/targeted MSVC LTCG (global BUILD_SERVER-coupled attempt diverged; isolated Release-only candidate prepared); profile-guided optimization; WPR/WPA hotspot analysis; Dell UltraPerformance (set before reboot, post-reboot readback unavailable); startup/reset path hardening; I/O/telemetry overhead |
| Open | 5/39 (13%) | bounded launch-wave A/B; nonblocking-evaluation overhead gate; accelerator/backend comparison; 15/20/30 Hz decision-rate quality-and-throughput gate; AngelScript runtime upgrade and production-equivalent engine hot-path optimization |

Thus 34/39 (87%) have at least been touched by evidence or inspection, but
only 16/39 (41%) are screened; **23/39 (59%) remain partial or open**. This is
coverage of the declared inventory, not a claim that optimization is exhausted.
Next highest-value safe work after connectivity returns: read back the staged
thermal mode without rebooting; verify and benchmark the isolated MSVC LTCG
candidate; deploy the benchmark fix; use a WPA-capable ETW analysis or bounded
native profiler to identify another hot path; and test engine edits only after
production-reference replay. The SHM array path's pooled result is near zero and
needs a substantially longer paired run before a default change.

### Remote reboot safety correction (2026-09-23)

The owner confirmed that this trainer's Tailscale connection must be started
manually at the machine after reboot. This was also consistent with the manual
Tailscale setup step in `Tools/rl/remote/WINDOWS_HOST_SETUP.md`, which I failed
to check before rebooting for the OEM thermal change. That reboot caused an
avoidable loss of remote access while the owner was away. **Do not reboot or
power-cycle the trainer remotely** unless the owner explicitly authorizes that
specific action and local recovery or verified independent management is
available. Do not assume SSH/Tailscale being online before reboot means it will
return after reboot. No further restart is authorized for the current work.

## OGRL-20260923-001 — isolated MSVC LTCG experiment prepared

The earlier `BUILD_SERVER=ON` result is not a clean LTCG comparison because that
switch changes deployment, packaging, console, Steamworks, and other build
settings in addition to `/GL` and `/LTCG`. Added the default-off CMake option
`RL_MSVC_LTCG`, which applies CMake IPO only to the Release `Overgrowth` and
`angelscript` targets, checks that the MSVC toolchain supports IPO, and does not
change `BUILD_SERVER` or any physics/runtime setting. Official compiler docs
confirm `/GL` enables cross-module optimization and requires the cooperating
link-time stage; CMake's IPO target property is the supported target-scoped path.

The option is an experiment switch, not an adopted build. On this Mac, a full
Release CMake configure with the option off completed under AppleClang 17.0.0;
turning it on correctly stopped at the non-MSVC guard. No engine was compiled,
no Windows flag line was inspected, and no throughput or replay conclusion is
claimed. The Windows gate is: configure with `BUILD_SERVER=OFF` and
`RL_MSVC_LTCG=ON`; prove `/GL` appears on both AngelScript and Overgrowth Release
compiles and `/LTCG` on the Overgrowth link; run the strict deterministic replay
gate against the same-source non-LTCG build; then run an interleaved real-PPO
no-checkpoint benchmark. Keep the option off unless every gate passes.

## OGRL-20260923-003 — scoped benchmark cleanup and AngelScript runtime lead

### Throughput probe safety

Source audit found the checked-out `throughput_sweep.py` still called a
machine-wide `taskkill /IM Overgrowth.exe` before a sweep and after every point.
That can kill an unrelated trainer or interactive game, contrary to the Windows
handoff rule. This turn did not invoke the sweep on the trainer. Replaced the
global kill with a read-only preflight that refuses to run while any Overgrowth
process exists, and a bounded fallback scoped to the benchmark Python process
and its child tree (`taskkill /PID <owned-pid> /T /F` on Windows; a private
POSIX process group elsewhere). Normal shutdown first uses the trainer's
update-boundary `control.json` stop. The probe now always omits
`--checkpoint-path`, never deletes checkpoints, reserves unique run/log paths,
refuses pre-existing summary output, and atomically writes its result summary.
An unexpected checkpoint is preserved and invalidates the point rather than
being removed.

Seven focused harness tests now cover a read-only busy-host check, BOM-free
atomic stop requests, clean stop, owned-process-tree fallback, refusal to
overwrite old telemetry, and atomic summary output. The full local safety set
(`test_throughput_sweep`, strict engine comparison tests, and checkpoint safety)
passes 20 tests; `py_compile` and `git diff --check` pass. Process-tree
termination and Windows process-list parsing were tested with mocks on the
Mac, not exercised on Windows. No trainer process, engine, checkpoint, or
scheduled task was touched by this turn.

### AngelScript 2.38 candidate

Official AngelScript change notes identify a runtime-performance improvement in
2.37 that reduces script-function-call overhead; the vendored 2.38 SDK includes
that release and its subsequent fixes. Overgrowth currently selects 2.32 on
Windows and 2.38 only for `RL_NATIVE_ARM64_TRAINING`. Added default-off
`RL_ANGELSCRIPT_238` to make the newer runtime a separately testable build
candidate without changing default builds. Official release note:
[AngelScript change history](https://angelcode.com/angelscript/changes.php).

Observed locally: CMake Release configuration with the option ON selected and
reported AngelScript 2.38.0, and the vendored `angelscript` library compiled
successfully with AppleClang 17.0.0. This is a dependency compile check only:
the Overgrowth executable was not built, the Windows/MSVC path was not tested,
no replay equivalence test ran, and no speed gain is claimed. The Windows gate
is to build status quo 2.32 and candidate 2.38 from the same source, verify
scripts/API initialization and strict transition/attack-event replay across the
frozen maps, then run matched real-PPO no-checkpoint ABBA under the established
trainer configuration. Keep the new selector off unless correctness passes;
retain even a small repeatable positive speed result per the owner's adoption
rule.

## OGRL-20260923-004 — Windows worker/thread screens, LTCG ABBA, and PGO startup failure

**Timestamp:** 2026-09-23 11:19 PDT. **Source:** nested repository branch
`optimize/overgrowth-training-throughput`, commit
`5c4e21d0be39e3872d3fd6e3f1acf444fb6231e2`; the only nested-repository dirty
paths remain two user-owned video deletions, left untouched. **Host:** Windows
11 Dell trainer, Intel Core Ultra 7 165U (2 P-cores); remote access through the
already-running Tailscale/SSH connection. No reboot, Tailscale/service change,
scheduled task, production training run, or checkpoint write occurred.

### No-checkpoint worker and update-thread screen

Runs used the real trainer with an in-memory PPO update path, six-map corpus,
checkpoint `run24_opt_smoke_20260921.pt` loaded read-only, and no
`--checkpoint-path`. The short screen measured 120 seconds after readiness;
the established workload settings were `n_steps=512`, one epoch, minibatch
128, collection/inter-op threads 2/1, hard reset every 20, engine AboveNormal
with mask `0xFFF`, trainer Normal. Every accepted point had completed cleanly
and actor-count/map-load evidence; raw JSON and engine evidence are under
`research-artifacts/OGRL-20260923-004-throughput/telemetry/thread_worker_sweeps/`.

The worker screen tested n14/k4, n16/k8, n18/k6, n20/k4, n22/k2, and n24/k6.
n18/k6 measured **1,013.277 useful transitions/s**, zero recoveries and zero
pool misses. n20/k4 measured **962.719 useful transitions/s**, zero recoveries,
and a 0.769% pool-miss rate. n14/k4 and n22/k2 failed before readiness;
n16/k8 and n24/k6 completed but failed the exact character-count proof, so
their speed numbers are invalid and excluded. The valid n18 point is one short
screen, not a new sustained record.

At n20/k4, the update-thread points were:

| update threads | valid useful wall SPS points | evidence/disposition |
|---:|---:|---|
| 1 | 1,011.661; 928.479 | second point had 0.781% pool misses; no recoveries |
| 2 | 1,045.068; 1,035.452 | both zero pool misses and recoveries; paired mean 1,040.260 SPS |
| 4 | 971.166; startup failure; 655.278 retry | retry had 1.351% pool misses and fewer measured transitions; too noisy to rank |
| 8 | 966.458; 1,054.030 | both valid; second was the highest 120-second point, not a five-minute result |

The two T=2 points average **1,040.260 SPS versus 970.070 for T=1**, a
matched-screen increase of **70.190 SPS / 7.236%**. Both T=2 runs were clean,
which makes this the clearest new short-window systems lead. It remains a
partial lever: run a longer, interleaved confirmation at the chosen worker
count and compare learner/update time and policy-quality evidence before
changing production defaults. T=8 averages 1,010.244 SPS over its two valid
points; the T=4 samples are disrupted by startup failure and a low outlier.

### LTCG controlled comparison

The fixed-base status-quo Release engine and scoped Release LTCG candidate each
passed the strict replay gate: 28/28 exact comparisons across ten repetitions
per binary; 20 replays included observed attacks; 732 ticks; action/control,
observation/reward and canonical attack-payload hashes matched exactly. Four
300-second useful-throughput windows per arm produced:

| binary arm | useful wall SPS, in order | mean |
|---|---|---:|
| status quo | 974.779, 914.221, 912.363, 1,021.810 | 955.793 |
| LTCG | 958.256, 1,001.884, 962.186, 998.548 | 980.218 |

The pooled observed change is **+24.425 SPS / +2.555%**. The four matched
order-pair differences have mixed signs (two favor each arm); keep LTCG as a
candidate and do not attribute a reliable sustained gain until the variance is
resolved. The highest valid 300-second point in this block was status-quo A4
at 1,021.810 SPS. The separate T=8 point at 1,054.030 was a 120-second run and
must not replace that sustained figure.

### Traditional MSVC PGO instrumented build and bounded failure

Built a separate Release instrumented engine from this source commit with
`RL_MSVC_PGO_MODE=INSTRUMENT`, `RL_WINDOWS_FIXED_BASE=ON`, CMake/MSVC 19.44
(VS 2022), `/GENPROFILE:EXACT,PGD=...`, and no `BUILD_SERVER` mode. The build
completed and linked a 14,204,416-byte executable; the PGD was initialized at
26,849,280 bytes. CMake's embedded git describe remained
`HEAD-HASH-NOTFOUND`; the reproducible source identity is the commit above.

The first no-checkpoint n18/k6 profiling attempt exited during process startup
with Windows status `0xC0000135`. Inspection of the instrumented binary's
imports and installed MSVC 14.44 files identified missing `pgort140.dll` in
the child PATH. Retried with the matching MSVC bin directory prepended only to
that probe's child environment. The instrumented processes started, selected
the assigned maps, and began configuration, but the all-worker ready barrier
did not arrive within 180 seconds. At timeout the sweep requested its normal
stop; initialization had not entered the trainer loop, so the 120-second
grace expired and the harness used its process-tree-scoped fallback. The run
is invalid: 0 metric rows, 24 engine startup logs, no `.pgc` profiles. SSH
accepted TCP but timed out during banner exchange under that instrumented load
for approximately five minutes; access then recovered. A subsequent read-only
check found zero Overgrowth/Python processes and the checkpoint SHA-256 still
`1F98963795CB5D1123EE5AD8A51DF870DF917B387205F3D51E0C36635CE4D88D`. No
machine restart occurred. Raw run manifests/logs and build outputs remain on
the trainer under `C:\ogrl\optimization\pgo_20260923\` and
`C:\ogrl\overgrowthRL_clean\Tools\rl\runs\`.

**Decision:** do not repeat instrumented PGO at n18/k6. The profile-use stage
is blocked because no profile data was collected. An independent audit is
pending through the requested ChatGPT-advisor skill. If that audit does not
identify a lower-risk solution, profile acquisition must first use a reduced,
bounded worker count and an explicit pre-ready cleanup budget; only use the
profile-use binary after strict production-reference replay, then compare it
against the identical status-quo build on matched 300-second no-checkpoint
windows. The current lever ledger remains 16/39 screened (41%), 18/39 partial
(46%), and 5/39 open (13%): the worker/thread families are still partial, and
PGO remains unproven rather than exhausted.
