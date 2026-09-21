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
7. **Engine-side sound early-outs:** the patch is documented in `DEAD_ENDS.md`
   but was not rebuilt and measured in the later run. A fresh trainer-boundary
   A/B is allowed; never swap the binary during a live run.
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
