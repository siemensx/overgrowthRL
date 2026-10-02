# Tools/rl — the Overgrowth RL pipeline (current as of 2026-10-02)

This is the single entry point. Older docs live in `attic/docs/` with SUPERSEDED banners; the
research chronology lives in the outer repo's `research-log/`, traps in `DEAD_ENDS.md`.

## The loop in one paragraph

The engine (C++ + AngelScript, `Source/Main/rl_*.cpp`) runs headless and talks to Python over shared
memory (`shm_env.py`). Every 4 physics ticks (120 Hz → 30 Hz decisions) it publishes a 339-float
observation (schema v5, `obs_schema.py`) and waits for an 8-float action (2 stick axes + jump, crouch,
attack, grab, drop, walk). `env.py` wraps one engine; `vec_env.py` runs N active + K standby engines
synchronously; `ppo/train_vec.py` is the trainer (`ppo/policy.py` actor-critic, `ppo/train.py`
`ppo_update`), `reward.py` + `curriculum.py` define the objective and the scenario sampler.

## Controls: name them, every time

Engine control flags change what the same weights score by 5× (OGRL-20261002-001):

| profile | flags | meaning |
|---|---|---|
| `corrected` | `rl_target_select: 2`, `rl_button_edges: 1` | a held button is one press, attack targets the nearest enemy. Every run since run27. |
| `corrected-nofeint` | + `rl_no_feint: 1` | also stops a held grab from cancelling the agent's own ground attacks (run29+) |
| `old` | none | auto-repeating buttons + frozen-camera targeting: everything trained before 2026-09-24 |

## Measure: `canonical_eval.py` (suite v2) and nothing else

```bash
python3 Tools/rl/canonical_eval.py --checkpoint <ckpt> --controls corrected-nofeint --out-dir <dir>
python3 Tools/rl/canonical_eval.py --checkpoint <ckpt> --controls corrected --out-dir <dir> --repeat-check
```

Training maps t_train_101/102/104 + held-out t_held_203; 1v1/1v2/1v3 at d=1.0; 100 greedy episodes per
cell; seeds 7,000,000+ (never used elsewhere); cap 1200; hard reset. Report TRAIN and HELD-OUT per
opponent count. The in-training periodic bench (`bench.py`, t_train_101 only, seeds 900000+) is a
health signal, not a result. See the script docstring for the determinism caveat.

## Train

Windows trainer (the grinder): one tracked launcher per run, e.g. `win_run29.bat`, registered as a
non-elevated scheduled task. Binding co-tenant rules (the box also runs the user's fbm / Aura /
vastwatch workers): engines + trainer `OGRL_*_PRIORITY=below`, `OGRL_*_AFFINITY=0x3FF` (CPUs 0–9;
10–13 stay free), ≤12 engines, `--max-wall-hours 6` (engines leak ~190 MB/h committed memory), kill
engines only by your shm prefix. Stop: `{"command":"stop"}` → `runs/<run>/control.json`.

Learner settings (OGRL-20261002-018): minibatch 1024, 2 epochs, `--kl-mode adaptive --critic-full-batch`,
lr cap 3e-4, entropy 0.003. Same-seed from-scratch A/B judged GREEDILY at equal steps (4.1M, 1v1, d=1.0):
mb128/1-epoch 26/150 train + 10/50 held; mb1024/2-epoch 39/150 + 14/50. The mb1024 arm's *training*
(sampled) win rate looked worse because it kept more exploration noise — never pick a learner from
training curves. The exploration bonus uses the unsquashed Gaussian entropy, so a high coefficient
inflates the stick's sigma (OGRL-20261002-017); keep it at 0.003 or fix the estimate.

## Tool map

| purpose | files |
|---|---|
| core library | `env.py shm_env.py vec_env.py obs_schema.py reward.py curriculum.py paths.py telemetry.py run_config.py tape.py ogreplay.py emergence.py noaslr.py` |
| trainer | `ppo/train_vec.py ppo/train.py ppo/policy.py ppo/normalize.py ppo/vec_buffer.py ppo/watch.py`, `remote_rollout.py` (only with `--remote-workers`) |
| evaluation | `canonical_eval.py` (THE benchmark), `evaluate.py` (one cell), `bench.py` (periodic), `benchmark_compare.py` (paired per-seed stats) |
| maps | `gen_arena_map.py` (no overwrite guard — never reuse a live map name), `gen_1v1_scenario.py`, `gen_human_duel_scenario.py`, `validate_maps.py`, `fork_workshop_level.py` |
| behaviour probes | `probe_feint.py probe_button_edges.py probe_self_motion.py probe_self_identity.py probe_damage_context.py probe_ko_accounting.py probe_throws.py action_profile.py move_stats.py kill_attribution.py measure_visibility.py visibility_outcome.py greedy_ab.py action_mode_probe.py diagnose_checkpoint.py` |
| scripted baselines | `engine_ai_baseline.py` (privileged), `observation_oracle_bot.py` (**left/right mirrored, results invalid until fixed**) |
| watch / play | `play_match.py play_1v3_human.py play_match_forever.sh record_watch.py render_smoke.sh replay_*.py dashboard/` |
| throughput | `throughput_sweep.py concurrency_sweep.py bench_levels.py bench_opts.py compare_engine_builds.py validate_soft_reset.py shm_smoketest.py` |
| ops | `build_engine.sh winps.sh remote/` (`sync_artifacts.sh <run>` defaults to `trainer-lan:C:/ogrl/overgrowthRL_clean`) |
| tests | `tests/` — run each file directly (`cd tests && python3 test_x.py`); no pytest on the Mac |

## Known open defects (not yet fixed; see research-log 2026-10-02)

- `on_ground` read as int from a 1-byte bool (`rl_observation.cpp:353`) — partially informative.
- Observation "right" is mirrored relative to action "right".
- Policy's entity mask is taken from the normalised valid flag (`policy.py:247`) — works by rounding luck.
- Self position / velocity / facing are world-frame; ids are raw floats; obs normaliser is frozen at
  1.5e9 samples.
- Active dodge cannot re-arm (needs ≥0.2 s of neutral stick); the agent cannot walk backwards while
  facing an enemy (`WantsToWalkBackwards` always FORWARDS).
- Seeds do not reset every AngelScript RNG stream: same seed ≠ same episode.
