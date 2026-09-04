# Overgrowth RL training pipeline

A from-scratch RL environment and PPO trainer built on top of the real
Overgrowth engine (`Source/Main/rl_*`), following the 8-stage plan in
`research-artifacts/implementation_plan_m4_gym.md` through Stage 5, then
extending past it (reward/curriculum/learner — a real gap in that plan, not
covered by any of its stages; see `research-log/`'s `ogrl-plan-scope`
discussion). Everything here is documented in detail, with the reasoning and
the bugs found along the way, in `research-log/2026-08-15.md` and
`research-log/2026-08-16.md` (entries `OGRL-20260816-005` through `-012`
cover this pipeline specifically).

## Architecture

```
 Engine process (Source/Main/*)                    Python process
 ─────────────────────────────                     ──────────────
 RLObservation::Extract()   ──┐
   (schema v2, egocentric,    │  shm segment + two named
    K-capped entities, LOS)   │  POSIX semaphores (obs/action),
                               │  lock-step request/response
 RLAction::Apply()          ◄─┘  (Source/Main/rl_shm_transport.{h,cpp})
   (writes into Input::
    PlayerInput -- the        shm_env.ShmEnv         (raw transport client)
    ONLY injection point,          │
    zero script changes)      env.OvergrowthEnv       (Gym-shaped: reset/step,
                                    │                   owns the engine subprocess,
                               vec_env.VecOvergrowthEnv  computes reward)
                                    │                  (N parallel workers,
                               ppo/train.py               thread-based)
                               ppo/train_vec.py        (PPO: policy.py, buffer.py /
                                                         vec_buffer.py, normalize.py)
```

The engine never computes reward or owns episode-outcome judgment — only
`episode_done` (self-knockout) is native. Reward, curriculum, and the win
condition (opponent knockout) all live in Python, specifically so they can be
iterated without an engine rebuild.

## Files

**C++ (engine side)** — `Source/Main/`:
- `rl_observation.{h,cpp}` — observation extraction. `ObservationConfig` is
  runtime-configurable (entity cap, ray count, FOV profile). Schema version
  and LOS rule version are both explicit constants, bumped whenever the
  buffer layout or visibility rule changes.
- `rl_action.{h,cpp}` — action injection via `Input::PlayerInput::key_down`,
  plus a scripted-action test harness (`--rl-action-script`) used to validate
  timing-sensitive combos before the transport existed.
- `rl_shm_transport.{h,cpp}` — the shm transport itself: header layout,
  lock-step protocol, episode reset (`Engine::ResetRLTrainingScenario`, Stage
  4, extended to work under a live shm-driven run, not just the benchmark
  harness).
- `rl_obs_test.{h,cpp}` — diagnostic-only dump/cost-measurement harness, not
  part of the training path.

**Python (training side)** — `Tools/rl/`:
- `obs_schema.py` — named field layout mirroring `RLObservation`'s buffer
  exactly. Never hardcode a float offset; import from here.
- `reward.py` — reward function (dense damage-based + sparse
  knockout/terminal), entities matched by id across frames (slot order is
  nearest-first and reorders every step).
- `curriculum.py` — reward-shaping curriculum (phased weight changes over
  training). **Not** an environment-composition curriculum (progressively
  harder scenarios) — that would need new training levels or an engine hook,
  neither of which exists yet.
- `shm_env.py` — raw transport client (ctypes bindings to the same libc
  calls the C++ side uses; zero third-party dependencies).
- `env.py` — `OvergrowthEnv`: Gym-shaped single-environment wrapper. Owns the
  engine subprocess's lifecycle. Supports frame-stacking (`frame_stack=N`) so
  the (non-recurrent) policy can perceive trends, not just instantaneous
  state.
- `vec_env.py` — `VecOvergrowthEnv`: N parallel `OvergrowthEnv` workers,
  thread-based (verified: `ctypes` releases the GIL during blocking calls, so
  N threads blocked in `sem_wait()` genuinely overlap).
- `ppo/policy.py` — hybrid actor-critic: tanh-squashed Gaussian for the 2
  continuous move dims (SAC-style bounded-action treatment) + 6 independent
  Bernoulli heads for the discrete buttons, one joint PPO objective.
- `ppo/buffer.py` / `ppo/vec_buffer.py` — rollout storage + GAE, with correct
  truncation-vs-termination handling (Pardo et al. 2018).
- `ppo/normalize.py` — running observation/return normalization (Welford's
  algorithm, matching OpenAI's VecNormalize).
- `ppo/train.py` — single-environment PPO training loop + CLI.
- `ppo/train_vec.py` — N-worker vectorized PPO training loop + CLI (reuses
  `train.py`'s `ppo_update`/`_explained_variance`/`_save_checkpoint`, not a
  duplicate implementation).
- `shm_smoketest.py` — minimal reference client for the raw transport,
  useful for debugging the transport in isolation from the RL stack.

## Running a training job

Single environment (simpler, useful for debugging):
```bash
python3 Tools/rl/ppo/train.py \
  --repo-root /path/to/overgrowthRL \
  --shm-name /ogrl_train0 \
  --total-timesteps 300000 \
  --log-path Tools/rl/ppo/runs/my_run.csv \
  --checkpoint-path Tools/rl/ppo/checkpoints/my_run.pt
```

Vectorized (N parallel engine workers, near-linear throughput scaling —
measured ~2785 steps/sec at N=4 vs. ~750-830 steps/sec single-environment):
```bash
python3 Tools/rl/ppo/train_vec.py \
  --repo-root /path/to/overgrowthRL \
  --shm-prefix /ogrl_v \
  --n-envs 4 \
  --total-timesteps 2000000 \
  --log-path Tools/rl/ppo/runs/my_vec_run.csv \
  --checkpoint-path Tools/rl/ppo/checkpoints/my_vec_run.pt
```

`--shm-prefix`/`--shm-name` must stay short — Darwin's POSIX shm/semaphore
name limit is ~31 bytes, and both the transport and (for the vectorized
trainer) a per-worker index get appended.

Each run's engine write-dirs live under `.rl_write_dirs/` (repo-relative,
gitignored) and are cleaned up automatically on `env.close()` — never leave
one behind (see `research-log/2026-08-15.md`'s disk-full incident for why
this matters).

## Known limitations (honestly, not silently)

- **No environment-composition curriculum.** Only reward-shaping. Progressive
  opponent/scenario difficulty would need new training levels or an engine
  hook.
- **No temporal signal in the raw observation.** The policy is a plain MLP;
  `frame_stack > 1` is the mitigation, not yet the default in either training
  script.
- **No CPU-vs-MPS benchmark for the training loop specifically.** PyTorch
  with MPS is available and confirmed working on this machine; `--device`
  exists in both training scripts but real numbers haven't been collected.
- **No watchdog on the shm transport.** Darwin has no `sem_timedwait`; a
  stalled Python side blocks the engine indefinitely. Fine for an attended
  run, a real gap for unattended long-running training.
- **Reward weights are not tuned.** They're a defensible literature-typical
  starting point (see `reward.py`'s docstring), not a result of any
  experiment.
