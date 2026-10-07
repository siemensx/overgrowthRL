# attic — archived, not maintained, not expected to run from here

Moved here on 2026-10-02 (OGRL-20261002-008) so that `Tools/rl/` contains only what the current
training / evaluation path uses. Nothing was deleted: history is intact (`git log --follow`), and the
research log cites these files by their old paths.

| folder | what | why archived |
|---|---|---|
| `collectors/` | async / frozen / process-pool collectors, remote rollout worker, their sweeps and tests | rejected three times against the synchronous `vec_env.py` (DEAD_ENDS: async collector, cohort batching) |
| `bench_legacy/` | Stage-0 `benchmark.py` + x86 configs, one-off throughput / concurrency sweeps, PGO script, checkpoint merging, crash-band bisect | single experiments from 08-15..09-07, all concluded; superseded by `throughput_sweep.py` / `bench_levels.py` |
| `eval_legacy/` | `multi_opponent_eval.py`, `transfer_eval.py`, `comprehensive_eval.sh`, `eval_cadence.sh`, `supervise_run.sh`, `eval_suites/v1.json` | competing benchmark definitions (wrong caps, missing engine control flags, hard-coded run21_mac); replaced by `canonical_eval.py` |
| `launchers/` | `win_run26..28.bat`, phase-1 launchers, `win_run_forever.bat`, `launch_training.ps1` | finished runs; several kill engines by image name, which DEAD_ENDS bans |
| `launchers/` (2026-10-07) | `win_run29.bat`, `win_run30_v2` .. `win_run37_two_teacher.bat` | finished runs; replaced by `run_profile.py` + `profiles/*.json` + `win_train.bat` (OGRL-20261007-008). `win_run38_scratch_attn.bat` stays in `Tools/rl/` while its task runs: cmd reads a .bat as it executes, so moving it under a live run breaks the restart loop. Archive it after the first `switch_profile.sh`. |
| `docs/` | `EVALUATION_CONTRACT.md`, `CURRICULUM.md`, `WINDOWS_HANDOFF.md` | contradicted by later findings; each carries a SUPERSEDED banner |

Python files here lost their `sys.path` neighbours when moved. To run one for archaeology, copy it back
to `Tools/rl/` temporarily.
