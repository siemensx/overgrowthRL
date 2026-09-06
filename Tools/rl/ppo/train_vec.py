#!/usr/bin/env python3
"""Vectorized PPO training: N parallel engine workers (vec_env.py) feeding
one policy update, extending train.py's single-environment loop -- see that
file's docstring for the PPO implementation details shared by both (GAE with
truncation handling, normalization, clipped objectives, etc.); this file
only adds the N-worker batching on top, using vec_buffer.VecRolloutBuffer in
place of buffer.RolloutBuffer and batched policy forward passes.

Deliberately a separate script from train.py rather than a --n-envs flag
bolted onto it: train.py is proven (validated against a live 300k-step run,
research-log OGRL-20260816-011) and this keeps that path completely
unmodified rather than risking it while adding vectorization.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Tools/rl
from vec_env import VecOvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, SCHEMA_VERSION
from curriculum import Curriculum, ScenarioSampler
from reward import run8_reward_config
from telemetry import RunLogger
from tape import TapeRecorder, decision_record
from ogreplay import runtime_fingerprint
from emergence import EmergenceAccumulator

from policy import ActorCritic, CONTINUOUS_DIM, DISCRETE_DIM
from vec_buffer import VecRolloutBuffer
from normalize import ObservationNormalizer, RewardNormalizer
from train import ppo_update, _explained_variance, _save_checkpoint  # reuse, not reimplement


def _entropy_random_reference() -> float:
    """OGRL-20260816-021 Sec 2.1/-022 Sec 5.2: the entropy of the untrained
    (maximally random) policy, computed from the action space's own shape
    rather than hardcoded -- stays correct if CONTINUOUS_DIM/DISCRETE_DIM
    ever change. 2 Gaussian axes at sigma=1 contribute 2 * 1/2*ln(2*pi*e)
    nats each; DISCRETE_DIM Bernoulli heads at p=0.5 contribute ln(2) nats
    each. This is "the single most important line on the whole dashboard"
    per the review -- runs 5/6/7 all finished AT OR ABOVE this line, meaning
    the actor never moved from its random initialization, and nothing was
    watching this number in real time to catch it."""
    return CONTINUOUS_DIM * 0.5 * math.log(2 * math.pi * math.e) + DISCRETE_DIM * math.log(2)


def _perf_with_reset_share(perf: dict, cycle_seconds: float) -> dict:
    """OGRL-20260817-028 Sec8.2: 'the single most important throughput
    number in this system' per -027 -- how much of this update's wall time
    went to resets (soft or hard, blocking or backgrounded -- see
    vec_env.py's drain_perf() comment on why background time counts too:
    it's still real CPU cores spent on resets, which is what actually
    explains the n_envs=4-vs-6 throughput coupling -027 measured, not just
    the blocking-time subset)."""
    reset_seconds = perf["reset_cpu_seconds"]
    reset_blocking_seconds = perf["reset_blocking_seconds"]
    return {
        "reset_seconds": reset_seconds,  # compatibility alias for older dashboards
        "reset_cpu_seconds": reset_seconds,
        "reset_share": (reset_seconds / cycle_seconds) if cycle_seconds > 0 else 0.0,
        "reset_blocking_seconds": reset_blocking_seconds,
        "reset_blocking_share": (reset_blocking_seconds / cycle_seconds) if cycle_seconds > 0 else 0.0,
        "pool_hits": perf["pool_hits"], "pool_misses": perf["pool_misses"],
        "step_wall_seconds": perf["step_wall_seconds"],
        "step_count": perf["step_count"],
        "worker_step_latency_p50_seconds": perf["worker_step_latency_p50_seconds"],
        "worker_step_latency_p90_seconds": perf["worker_step_latency_p90_seconds"],
        "worker_step_latency_p99_seconds": perf["worker_step_latency_p99_seconds"],
        "barrier_idle_seconds": perf["barrier_idle_seconds"],
        "active_workers": perf["active_workers"],
        "ready_standby_workers": perf["ready_standby_workers"],
    }


def _git_sha(repo_root: str) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- telemetry metadata, never worth failing a run over
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--levels", default="",
                   help="comma-separated level list for the map axis (C6). Workers are "
                        "assigned round-robin and keep their level for the run, so every "
                        "PPO batch mixes maps without any extra reset cost. Overrides "
                        "--level when set. Hold maps back from this list to keep a "
                        "genuine transfer test.")
    p.add_argument("--level", default="arenas/oval_arena.xml")
    p.add_argument("--shm-prefix", default="/ogrl_vec")
    p.add_argument("--n-envs", type=int, default=4,
                    help="OGRL-20260816-023: benchmarked, not a carried-over guess -- concurrency_sweep.py measured "
                         "(n_envs, k_standby) jointly at act_period=4 and n_envs=4 won clearly (705.8 decisions/s vs "
                         "533.5 at n_envs=8, the old default from OGRL-20260816-014's pre-fix sweep). Re-sweep if "
                         "act_period or the level/scenario changes.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=100_000, help="total env steps across ALL workers combined")
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per worker, per PPO update (n_steps * n_envs transitions/update)")
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--value-clip-coef", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--entropy-coef-final", type=float, default=None,
                    help="OGRL-20260816-021 Sec 2.4/OGRL-20260816-023: if set, entropy_coef anneals LINEARLY from "
                         "--entropy-coef down to this value over --entropy-anneal-steps, instead of staying constant. "
                         "Runs 5-7 never had a converging policy partly because a constant entropy bonus was, per the "
                         "review, 'the only consistent gradient in the system and it is winning' -- annealing it down "
                         "stops the exploration bonus from being able to outcompete the actual objective indefinitely. "
                         "None (default) preserves the old constant-entropy_coef behavior exactly.")
    p.add_argument("--entropy-anneal-steps", type=int, default=1_000_000,
                    help="global_step at which the entropy_coef anneal reaches --entropy-coef-final; linear before that, "
                         "held at --entropy-coef-final after. Ignored if --entropy-coef-final is not set.")
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--act-period", type=int, default=1,
                    help="OGRL-20260816-021 Sec 1.3(a)/Stage 6: engine decisions every N physics ticks (1=120Hz, the "
                         "original; 4=30Hz, matching vanilla AI's own control period). See rl_shm_transport.cpp.")
    p.add_argument("--k-standby", type=int, default=5,
                    help="OGRL-20260816-021 Sec 1.3(b)/OGRL-20260816-025: pre-warmed spare envs for off-critical-path "
                         "reset (see vec_env.py). Default 5 is the BENCHMARKED optimum at n_envs=4, not a guess -- "
                         "standby_depth_sweep.py measured miss-rate-vs-throughput across k_standby=2..8 and found a "
                         "real crossover: k=2 (run8's original value) sits at 31%% pool-underrun and 571.5 decisions/s; "
                         "k=5 hits 8%% underrun at 593.5 decisions/s (the actual peak); k=6-8 reach 0%% underrun but are "
                         "SLOWER (500-521 decisions/s) because each extra standby is a full engine process competing "
                         "for the same cores. More standbys is not free and is not monotonically better -- this "
                         "default is the measured crossover point, not the deepest pool tested. Re-sweep if n_envs "
                         "changes; the optimum is n_envs-relative, not an absolute constant. 0 reproduces the "
                         "original fully-synchronous reset behavior.")
    p.add_argument("--reward-profile", choices=["default", "run8"], default="default",
                    help="'default' reproduces runs 1-7's RewardConfig exactly, for comparability. 'run8' "
                         "(OGRL-20260816-023) uses reward.run8_reward_config() -- symmetric +/-10 terminal outcome, "
                         "dense damage at a matched +/-1 scale, a much smaller time_cost, stall tax and ragdoll "
                         "penalty off, no closing-distance shaping -- see that function's docstring for why.")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--collection-torch-threads", type=int, default=1,
                   help="PyTorch intra-op threads during tiny rollout inference; 1 avoids competing with engine workers")
    p.add_argument("--update-torch-threads", type=int, default=4,
                   help="PyTorch intra-op threads during PPO minibatch updates")
    p.add_argument("--torch-interop-threads", type=int, default=1,
                   help="PyTorch inter-op threads, set once at startup")
    p.add_argument("--log-path", default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--checkpoint-every-updates", type=int, default=10)
    p.add_argument("--run-id", default=None, help="OGRL-20260816-022: dashboard run identifier; defaults to the same "
                                                    "vec-<unix_ts> stem used for --log-path when not given.")
    p.add_argument("--runs-root", default=None, help="dashboard telemetry root; defaults to Tools/rl/runs under --repo-root.")
    p.add_argument("--purpose", default="", help="one-line free text shown on the dashboard's run list -- what this run is for.")
    # --- OGRL-20260817-028 Sec1: soft reset ---
    p.add_argument("--soft-reset", action="store_true",
                    help="Sec1: use Engine::SoftResetRLTrainingScenario (reseed + Level::Message(\"post_reset\"), "
                         "no ClearLoadedLevel/LoadLevel) for per-episode resets instead of the original full "
                         "LoadLevel path. Default off so nothing changes unless explicitly opted in -- pass this "
                         "only after the Sec1.3 validation suite has passed on this build.")
    p.add_argument("--hard-reset-every", type=int, default=20,
                    help="Sec1.2 safety valve: force a hard reset every Nth reset of a given engine PROCESS's "
                         "lifetime (bounds decal/dropped-item/object-id accumulation a soft reset doesn't clear). "
                         "Only consulted when --soft-reset is set. 0 disables the valve (always soft). Tightened "
                         "from the plan's suggested 50 to 20 (2026-08-17): validate_soft_reset.py's leak audit and "
                         "interleaved distribution-equivalence checks both passed clean, but a deep-in-sequence "
                         "physics-replay comparison (many hundred resets into one engine process) showed a position "
                         "deviation (~0.9 units) that isolated few-reset A/B pairs could not reproduce -- root cause "
                         "not fully identified, so this is a deliberate conservative hedge until it is.")
    # --- OGRL-20260817-028 Sec3: environment-composition curriculum ---
    p.add_argument("--d-max-start", type=float, default=0.15,
                    help="Sec10: start LOW, not 0.30 -- see curriculum.ScenarioSampler's docstring for why "
                         "(shock-avoidance on a resumed policy, signal-density on a cold start).")
    p.add_argument("--d-max-cap", type=float, default=1.0)
    p.add_argument("--d-step", type=float, default=0.10)
    p.add_argument("--d-min", type=float, default=0.0,
                    help="OGRL-20260817-034: floor of the per-episode d~Uniform(d_min, d_max) sample range -- 0.0 "
                         "(default) is the original full-range behavior. Raise this once d_max has been at its cap "
                         "for a while to stop spending new episodes on already-mastered easy difficulty; see "
                         "curriculum.ScenarioSampler's d_min comment for the measured justification.")
    p.add_argument("--gate-window", type=int, default=300, help="episodes considered for the d_max advance gate")
    p.add_argument("--gate-min-samples", type=int, default=50, help="minimum top-band episodes before the gate can fire")
    p.add_argument("--gate-win-rate", type=float, default=0.75)
    p.add_argument("--remote-port", type=int, default=0,
                   help="listen on this port for remote rollout workers (0 = disabled). Workers "
                        "collect with the weights broadcast at the start of each update and the "
                        "learner waits for all of them, so this stays exactly on-policy -- no "
                        "staleness and no importance correction. The learner therefore runs at the "
                        "SLOWEST participant's pace, so give each machine env slots proportional to "
                        "its measured speed; an equal split across a fast and a slow machine is "
                        "worse than running the fast one alone.")
    p.add_argument("--remote-workers", type=int, default=0,
                   help="number of remote workers to wait for before training starts")
    p.add_argument("--opponents-cap", type=int, default=1,
                   help="maximum opponents the curriculum may unlock (1 disables it). Needs maps "
                        "carrying game_type 3 (1v2) and 4 (1v3); any level without them falls back "
                        "to the 1v1 pair, so this is safe on oval and every stock arena.")
    p.add_argument("--opp-gate-win-rate", type=float, default=0.60,
                   help="win rate AT THE CURRENT MAX opponent count needed to unlock the next. "
                        "Lower than the difficulty gate on purpose -- being outnumbered should stay hard.")
    p.add_argument("--opp-gate-window", type=int, default=400)
    p.add_argument("--opp-gate-min-samples", type=int, default=150)
    p.add_argument("--opp-keep-solo", type=float, default=0.35,
                   help="fraction of episodes held at 1v1 once the curriculum advances. This is the "
                        "anti-forgetting term: without it the opponent axis is a distribution shift "
                        "rather than an addition, and 1v1 competence can quietly decay.")
    p.add_argument("--opponents", type=int, default=1, help="Stage D/E axis; NOT wired to game_type in the level "
                                                               "script yet, see arena_level_1v1_unarmed.as -- keep at 1")
    p.add_argument("--stall-target-weight", type=float, default=None,
                    help="per-step stall tax once stall_grace_steps of zero combat contact have passed. "
                         "Omit to keep the profile default (run8 pins it to 0.0). ~0.02 is the value the "
                         "term was designed around. Outcome-based: it constrains WHETHER the agent engages, "
                         "never HOW -- see reward.py on why distance shaping was rejected.")
    p.add_argument("--stall-ramp-steps", type=int, default=None,
                    help="decisions over which the stall tax ramps from 0 to --stall-target-weight, counted "
                         "from this run's own resume point (stall_intro_step). Ramped, never stepped: run6 "
                         "regressed on an abrupt introduction of this exact term.")
    p.add_argument("--species-mode", type=int, default=0, help="0=legacy random guard/raider (Stage A), 4=random of all 3 (Stage B)")
    p.add_argument("--weapons-prob", type=float, default=0.0, help="probability a round is armed (Stage C axis)")
    # --- OGRL-20260817-028 Sec8.1: Tier-1 tape recording ---
    p.add_argument("--tape-every", type=int, default=10, help="record worker 0's full episode every N updates, "
                                                                 "in addition to always keeping the best/worst-reward "
                                                                 "episode per --tape-window-updates. 0 disables sampled recording (best/worst still recorded).")
    p.add_argument("--tape-window-updates", type=int, default=50)
    p.add_argument("--tape-keep-recent", type=int, default=200)
    p.add_argument("--no-tapes", action="store_true", help="disable Tier-1 tape recording entirely")
    p.add_argument("--no-native-capture", action="store_true",
                   help="disable the native per-physics-tick digest capture used by exact rendered replay")
    p.add_argument("--pause-below-free-gb", type=float, default=3.0,
                    help="2026-08-17 disk-space safety net: pause (checkpoint saved) when free disk on "
                         "--repo-root's filesystem drops below this many GB, resume automatically once it's "
                         "back above it. This machine has hit a real disk-full incident before during an "
                         "unattended run (AGENTS.md).")
    p.add_argument("--stop-below-free-gb", type=float, default=1.0,
                    help="stop cleanly (checkpoint saved) below this many GB free -- the harder floor under "
                         "--pause-below-free-gb, for when pausing alone isn't enough (nobody freed space).")
    p.add_argument("--resume-from", default=None,
                    help="OGRL-20260816-018: path to a checkpoint (policy/optimizer/both normalizers/global_step) "
                         "to continue training from, instead of a cold start. Use this for a reward/curriculum "
                         "tweak that doesn't invalidate what's already been learned (e.g. adding a new reward "
                         "component) -- as opposed to a correctness bug in the reward/observation/action itself "
                         "(e.g. the causation or seed-diversity fixes), which DOES invalidate prior training and "
                         "should stay a cold start. --total-timesteps is still an ABSOLUTE global_step target, not "
                         "an additional budget -- e.g. resuming from step 2,949,120 needs --total-timesteps "
                         "6000000 for 3M more steps, not 3000000 (which would be a no-op).")
    return p.parse_args()


def _raise_keyboard_interrupt(signum, frame):
    # SIGTERM's default action is immediate termination -- it does NOT raise
    # KeyboardInterrupt the way Ctrl+C (SIGINT) does, so it bypasses the
    # existing `finally: vec_env.close()` below entirely. That's a real gap
    # found the hard way: stopping a run via `kill <pid>` (plain SIGTERM, not
    # -9) left all N engine subprocesses orphaned, each stuck blocking on its
    # own sem_wait() forever, requiring a manual per-process kill -9 sweep
    # (research-log OGRL-20260816-014's addendum). Translating SIGTERM into
    # the same KeyboardInterrupt path SIGINT already takes means both stop
    # this run the same, already-correct way.
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = parse_args()
    # Preserved separately from args.entropy_coef, which the anneal below
    # mutates in place every update once training starts -- the anneal
    # formula needs the ORIGINAL starting value throughout, not whatever
    # args.entropy_coef was most recently overwritten to.
    args.entropy_coef_start = args.entropy_coef
    if min(args.collection_torch_threads, args.update_torch_threads, args.torch_interop_threads) < 1:
        raise ValueError("PyTorch thread counts must be positive")
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.set_num_threads(args.collection_torch_threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layout = DEFAULT_LAYOUT

    log_path = Path(args.log_path) if args.log_path else Path(args.repo_root) / "Tools/rl/ppo/runs" / f"vec-{int(time.time())}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "global_step", "update", "mean_episode_reward", "mean_episode_length", "episodes_completed",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
        "curriculum_phase", "steps_per_second",
    ])
    print(f"logging to {log_path} ({args.n_envs} parallel workers)")

    # Resumed runs start their curriculum lookup from the checkpoint's own
    # global_step, not 0 -- otherwise the initial VecOvergrowthEnv construction
    # below would (harmlessly, since the loop's first set_reward_config()
    # call overwrites it before any real step happens, but confusingly)
    # briefly configure bootstrap-phase reward weights for a run that's
    # actually resuming deep into main phase.
    resumed_checkpoint = None
    if args.resume_from:
        resumed_checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
    initial_global_step = resumed_checkpoint["global_step"] if resumed_checkpoint else 0

    run_id = args.run_id or log_path.stem
    runs_root = Path(args.runs_root) if args.runs_root else Path(args.repo_root) / "Tools/rl/runs"
    entropy_random_reference = _entropy_random_reference()
    manifest = {
        "purpose": args.purpose,
        # OGRL-20260817-028 Sec8.6: obs_schema_version is the SCHEMA VERSION
        # (obs_schema.SCHEMA_VERSION), not the observation dimension --
        # those were conflated here before (recorded 260, the float count,
        # under a key named "version"), which is what the dashboard needs
        # fixed to display a real version number rather than a byte count.
        # obs_dim is now its own field so neither piece of information is lost.
        "code": {"git_sha": _git_sha(args.repo_root), "obs_schema_version": SCHEMA_VERSION, "obs_dim": layout.total_floats},
        # OGRL-20260820-044: the Git SHA alone is not a replay identity. Keep
        # an immutable runtime fingerprint in the run manifest and carry it
        # into every retained .ogreplay artifact.
        "runtime": runtime_fingerprint(
            args.repo_root,
            binary=Path(args.repo_root) / "BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth",
            observation_schema=SCHEMA_VERSION,
            action_schema="action-v1-8-floats",
            reset_schema="canonical-reset-v1-zero-action-settle",
            physics_hz=120,
            act_period=args.act_period,
        ),
        "replay": {"container": "ogreplay", "version": 1,
                   "authoritative_state_capture": "native-digest-v2" if not args.no_native_capture else "disabled",
                   "engine_launch": "native-action-and-digest" if not args.no_native_capture else "disabled"},
                "env": {"level": args.levels or args.level, "n_envs": args.n_envs, "max_episode_steps": args.max_episode_steps,
                "frame_stack": args.frame_stack, "act_period": args.act_period, "k_standby": args.k_standby},
        "algo": {k: v for k, v in vars(args).items() if k not in ("entropy_coef_start",)},
        "reward_profile": args.reward_profile,
        "entropy_random_reference": entropy_random_reference,
        "parent": {"run_id": None, "checkpoint": args.resume_from, "global_step": initial_global_step if resumed_checkpoint else None},
    }
    logger = RunLogger(runs_root, run_id, manifest)
    logger.log_event("run_start", f"{run_id} started", body=args.purpose)
    print(f"dashboard telemetry: {runs_root / run_id}")

    # stall_intro_step=initial_global_step (OGRL-20260816-019): the stall-tax
    # ramp needs to start counting from wherever THIS run's policy actually
    # begins seeing it, not from absolute step 0 -- a cold start (0) ramps in
    # early during bootstrap; a resumed run ramps in starting at its resume
    # point, so a policy that's never seen the term before gets the same
    # gradual on-ramp a fresh policy would, instead of the abrupt full-weight
    # introduction that regressed run6.
    # OGRL-20260816-023: run8's reward profile is a genuinely different base
    # (reward.run8_reward_config()), with its own shaping fully zeroed --
    # runs 1-7's default profile and curriculum are completely unaffected by
    # this flag (base_config=None reproduces the exact old behavior).
    reward_base_config = run8_reward_config() if args.reward_profile == "run8" else None
    curriculum_kwargs = {"stall_intro_step": initial_global_step, "base_config": reward_base_config}
    if args.reward_profile == "run8":
        curriculum_kwargs["bootstrap_closing_weight"] = 0.0
        curriculum_kwargs["stall_target_weight"] = 0.0
    # OGRL-20260904-058: the run8 profile pinned stall_target_weight to 0.0 on
    # the reasoning that "a well-defined 1v1 needs no engagement bootstrap".
    # That holds against an opponent that closes on its own and fails against
    # one that does not -- run15 wins 3/3 against the stock expert and 0/3
    # against an idle actor, never closing inside 2.8 m. The stall tax is the
    # right lever for that (outcome-based: it fires after stall_grace_steps of
    # zero combat contact and never dictates HOW to engage, unlike the
    # distance shaping that was deliberately rejected). These flags make it
    # settable without editing code; omitting them reproduces the pinned
    # behaviour exactly.
    if args.stall_target_weight is not None:
        curriculum_kwargs["stall_target_weight"] = args.stall_target_weight
    if args.stall_ramp_steps is not None:
        curriculum_kwargs["stall_ramp_steps"] = args.stall_ramp_steps
    curriculum = Curriculum(**curriculum_kwargs)

    # OGRL-20260817-028 Sec3: environment-composition curriculum (separate
    # concern from Curriculum above, which only shapes reward WEIGHTS on a
    # fixed scenario -- this samples the SCENARIO itself, per episode).
    levels_list = [x.strip() for x in args.levels.split(",") if x.strip()] or args.level
    if isinstance(levels_list, str):
        levels_list = [levels_list]
    sampler_kwargs = dict(
        d_max_start=args.d_max_start, d_max_cap=args.d_max_cap, d_step=args.d_step, d_min=args.d_min,
        gate_window=args.gate_window, gate_min_samples=args.gate_min_samples,
        gate_win_rate=args.gate_win_rate, opponents=args.opponents,
        species_mode=args.species_mode, weapons_prob=args.weapons_prob,
        opponents_cap=args.opponents_cap, opp_gate_win_rate=args.opp_gate_win_rate,
        opp_gate_window=args.opp_gate_window, opp_gate_min_samples=args.opp_gate_min_samples,
        opp_keep_solo=args.opp_keep_solo, rng_seed=args.seed,
    )
    sampler = ScenarioSampler(
        d_max_start=args.d_max_start, d_max_cap=args.d_max_cap, d_step=args.d_step, d_min=args.d_min,
        gate_window=args.gate_window, gate_min_samples=args.gate_min_samples, gate_win_rate=args.gate_win_rate,
        opponents=args.opponents, species_mode=args.species_mode, weapons_prob=args.weapons_prob,
        opponents_cap=args.opponents_cap, opp_gate_win_rate=args.opp_gate_win_rate,
        opp_gate_window=args.opp_gate_window, opp_gate_min_samples=args.opp_gate_min_samples,
        opp_keep_solo=args.opp_keep_solo,
        rng_seed=args.seed,
    )
    vec_env = VecOvergrowthEnv(
        n_envs=args.n_envs, repo_root=args.repo_root,
        level=levels_list,
        shm_prefix=args.shm_prefix,
        base_seed=args.seed, layout=layout, reward_config=curriculum.reward_config_for_step(initial_global_step),
        frame_stack=args.frame_stack, max_episode_steps=args.max_episode_steps,
        k_standby=args.k_standby, act_period=args.act_period,
        soft_reset=args.soft_reset, hard_reset_every=args.hard_reset_every, scenario_fn=sampler.sample_episode,
        native_trace_dir=None if args.no_native_capture else (logger.run_dir / "native-traces"),
    )
    obs_dim = vec_env.observation_dim

    tape_recorder = None if args.no_tapes else TapeRecorder(
        logger.run_dir, tape_every=args.tape_every, window_updates=args.tape_window_updates, keep_recent=args.tape_keep_recent,
    )

    # OGRL-20260817-028 Sec5: ActorCritic/ObservationNormalizer now take the
    # layout + frame_stack directly (they need to know where the entity
    # region lives within each stacked frame), not just a flat obs_dim.
    policy = ActorCritic(layout, frame_stack=args.frame_stack).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_normalizer = ObservationNormalizer(layout, frame_stack=args.frame_stack)
    reward_normalizer = RewardNormalizer(args.gamma, n_envs=args.n_envs)
    # Remote rollout workers (optional). Accepted BEFORE the first update so the
    # buffer width and the reward normaliser are sized to the real total.
    remote_conns: list = []
    remote_env_counts: list = []
    if args.remote_port and args.remote_workers > 0:
        import socket as _socket
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from remote_rollout import send_msg as _send, recv_msg as _recv, configure_socket as _cfgsock
        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.remote_port))
        srv.listen(args.remote_workers)
        print(f"waiting for {args.remote_workers} remote worker(s) on port {args.remote_port} ...", flush=True)
        for _ in range(args.remote_workers):
            conn, addr = srv.accept()
            _cfgsock(conn)
            hello = _recv(conn)
            n_remote = int(hello["n_envs"])
            _send(conn, {"type": "config", "n_steps": args.n_steps, "levels": levels_list,
                         "act_period": args.act_period, "frame_stack": args.frame_stack,
                         "max_episode_steps": args.max_episode_steps,
                         "hard_reset_every": args.hard_reset_every, "gamma": args.gamma,
                         "sampler_kwargs": sampler_kwargs})
            remote_conns.append(conn)
            remote_env_counts.append(n_remote)
            print(f"  worker {addr[0]} joined with {n_remote} envs", flush=True)
        srv.close()
    total_envs = args.n_envs + sum(remote_env_counts)
    if remote_conns:
        print(f"distributed: {args.n_envs} local + {sum(remote_env_counts)} remote = {total_envs} envs "
              f"({args.n_steps * total_envs} transitions per update)", flush=True)

    buffer = VecRolloutBuffer(args.n_steps, args.n_envs, obs_dim, 8, device)

    if resumed_checkpoint:
        # Explicit-metadata schema-mismatch guard (Sec5) -- see
        # _save_checkpoint's comment in train.py for why this replaced a
        # weight-shape sniff. A checkpoint saved before this change has none
        # of these keys at all, which is itself proof it predates schema
        # v5/the entity encoder and cannot be resumed across -- treated the
        # same as an explicit mismatch, not a KeyError.
        ckpt_total_floats = resumed_checkpoint.get("layout_total_floats")
        ckpt_frame_stack = resumed_checkpoint.get("frame_stack")
        if ckpt_total_floats != layout.total_floats or ckpt_frame_stack != args.frame_stack:
            raise ValueError(
                f"--resume-from checkpoint has layout_total_floats={ckpt_total_floats}, frame_stack={ckpt_frame_stack} "
                f"(missing/None means it predates the Sec5 entity-encoder architecture), but the current engine "
                f"build + obs_schema.py + --frame-stack {args.frame_stack} produce layout_total_floats={layout.total_floats} "
                f"-- cannot resume across an observation schema or architecture change, this needs a cold start instead."
            )
        # Restore policy + optimizer (Adam's moment estimates -- a cold reset
        # there would itself cause a transient instability right at resume,
        # the same failure mode this is trying to avoid) + BOTH normalizers'
        # running statistics (obs and reward) -- resuming the policy alone
        # while the normalizers restart from scratch would make every
        # observation/reward look like a sudden distribution shift on step
        # one, which is exactly the kind of thing that produced the approx_kl
        # spikes seen during ordinary training, not something to invite at a
        # resume boundary on purpose.
        policy.load_state_dict(resumed_checkpoint["policy"])
        optimizer.load_state_dict(resumed_checkpoint["optimizer"])
        obs_normalizer.load_state_dict(resumed_checkpoint["obs_normalizer"])
        reward_normalizer.load_state_dict(resumed_checkpoint["reward_normalizer"])
        print(f"resumed from {args.resume_from} at global_step={initial_global_step}")

    global_step = initial_global_step
    update = 0  # this run's own update counter for ITS log/checkpoint cadence -- global_step is what actually
                # carries continuity (curriculum phase, total-timesteps target), not this index
    episode_reward = np.zeros(args.n_envs, dtype=np.float64)
    episode_length = np.zeros(args.n_envs, dtype=np.int64)
    episode_components = [defaultdict(float) for _ in range(args.n_envs)]  # per-worker running reward-component sums,
                                                                            # reset to 0 each time that worker's episode ends
    episode_start_time = [time.time()] * args.n_envs
    episode_seed_used = list(range(args.n_envs))  # worker index, NOT the actual per-episode seed -- vec_env.py's
                                                   # auto-reset seed diversity (OGRL-20260816-016) is internal to
                                                   # VecOvergrowthEnv and isn't currently surfaced back to the
                                                   # caller; this is a placeholder until that's plumbed through,
                                                   # not a claim of per-episode reproducibility
    previous_cycle_end = time.monotonic()  # for perf.cycle_seconds -- the full update-to-update wall time,
                                            # not just collection_seconds (OGRL-20260816-020's sps blind spot)
    run_status = "interrupted"  # pessimistic default -- only overwritten to "completed" right after a clean loop
                                 # exit (natural completion or an explicit dashboard stop), so a real exception
                                 # (crash, Ctrl+C, SIGTERM) leaves this as-is and the dashboard shows it honestly

    try:
        raw_obs = vec_env.reset(seeds=[args.seed + i for i in range(args.n_envs)])
        obs = obs_normalizer.normalize(raw_obs)
        raw_obs_current = raw_obs  # OGRL-20260817-028 Sec8.1: the raw (unnormalized) observation each
                                    # step's action was actually chosen from -- tape.decision_record needs
                                    # real health/position values, not normalized ones. Updated at the end
                                    # of every step below, mirroring how `obs` tracks the normalized copy.

        while global_step < args.total_timesteps:
            # Pause/stop control (OGRL-20260816-023): polled once per update,
            # not per step -- cheap, and this is the only point in the loop
            # where pausing mid-flight can't leave the rollout buffer half
            # full. See telemetry.py's module docstring for why this one
            # file-based write channel exists despite the rest of the system
            # being strictly read-only.
            command = logger.poll_control()
            stopped_while_paused = False
            if command == "pause":
                logger.log_event("pause", f"{run_id} paused via dashboard control", body=f"at global_step={global_step}")
                while command == "pause":
                    time.sleep(0.5)
                    command = logger.poll_control()
                if command == "stop":
                    stopped_while_paused = True
                else:
                    logger.log_event("note", f"{run_id} resumed via dashboard control", body=f"at global_step={global_step}")
                    logger.clear_control()
            if command == "stop" or stopped_while_paused:
                # "stop_requested", not "run_stop" -- OGRL-20260817-028's
                # "a run_stop event on every exit path" is the canonical one
                # in the `finally` block below, which fires for every exit
                # (this one included); this event is the WHY, not the
                # terminal marker itself.
                logger.log_event("stop_requested", f"{run_id} stopped via dashboard control", body=f"at global_step={global_step}")
                logger.clear_control()
                break

            # Disk-space safety net (2026-08-17): this machine has hit a
            # disk-full incident before during an unattended run (see
            # AGENTS.md), and an uncapped overnight run writes checkpoints/
            # logs/tapes for hours with nobody watching. Checked at the same
            # cadence as pause/stop control -- cheap, and the natural place
            # to intervene before a write fails mid-flight and corrupts a
            # checkpoint. Below --stop-below-free-gb: save a checkpoint and
            # stop cleanly, same as a dashboard stop. Below --pause-below-free-gb
            # (checked first, higher threshold): pause and keep polling, same
            # as a dashboard pause -- gives the user a chance to free space
            # without losing the run outright.
            free_gb = shutil.disk_usage(args.repo_root).free / (1024 ** 3)
            if free_gb < args.stop_below_free_gb:
                logger.log_event("disk_low_stop", f"{run_id} stopping: {free_gb:.2f}GB free < --stop-below-free-gb {args.stop_below_free_gb}",
                                  body=f"at global_step={global_step}")
                break
            if free_gb < args.pause_below_free_gb:
                logger.log_event("disk_low_pause", f"{run_id} pausing: {free_gb:.2f}GB free < --pause-below-free-gb {args.pause_below_free_gb}",
                                  body=f"at global_step={global_step}")
                if args.checkpoint_path:
                    _save_checkpoint(args.checkpoint_path, policy, optimizer, obs_normalizer, reward_normalizer, global_step)
                while shutil.disk_usage(args.repo_root).free / (1024 ** 3) < args.pause_below_free_gb:
                    time.sleep(30.0)
                    if logger.poll_control() == "stop":
                        logger.log_event("stop_requested", f"{run_id} stopped via dashboard control while disk-paused", body=f"at global_step={global_step}")
                        logger.clear_control()
                        run_status = "completed"
                        raise KeyboardInterrupt  # reuse the existing clean-exit path below rather than a second copy of it
                logger.log_event("note", f"{run_id} resuming: disk freed", body=f"at global_step={global_step}")

            reward_config = curriculum.reward_config_for_step(global_step)
            vec_env.set_reward_config(reward_config)

            episode_rewards_this_update = []
            episode_lengths_this_update = []
            episode_components_this_update = []  # list of per-episode component dicts, this update only
            outcomes_this_update = {"won": 0, "lost": 0, "timeout": 0}
            actions_this_update = []  # OGRL-20260817-028 Sec8.2: raw sampled actions, for action_stats below
            emergence = EmergenceAccumulator()  # Sec8.3: fresh each update, same sample size as action_stats
            collection_start = time.monotonic()
            # Tiny policy batches lose more to thread-pool coordination than
            # they gain from intra-op parallelism. Keep collection on one
            # thread so it does not compete with the engine workers; the
            # larger PPO minibatches switch to the measured update setting
            # immediately before ppo_update() below.
            torch.set_num_threads(args.collection_torch_threads)

            # Broadcast the weights this rollout will be collected with. Sent
            # BEFORE local collection so the workers collect in parallel with
            # the learner rather than after it.
            if remote_conns:
                _payload = {"type": "weights",
                            "policy": {k: v.cpu() for k, v in policy.state_dict().items()},
                            "obs_normalizer": obs_normalizer.state_dict(),
                            "reward_normalizer": reward_normalizer.state_dict()}
                for _c in remote_conns:
                    _send(_c, _payload)

            for _ in range(args.n_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, log_probs, _entropy, values = policy.get_action_and_value(obs_tensor)
                actions_np = actions.cpu().numpy()
                actions_this_update.append(actions_np)

                raw_next_obs, rewards, terminals, truncateds, infos = vec_env.step(actions_np)
                episode_reward += rewards
                episode_length += 1
                global_step += args.n_envs
                for i, info in enumerate(infos):
                    for name, value in info["reward_components"].items():
                        episode_components[i][name] += value

                # Sec8.3: the always-on emergence panel uses the fixed schema
                # directly, avoiding Python dict/list allocation for every
                # entity on every step. Tier-1 tapes remain deliberately
                # detailed, but their unpacking now runs only when recording
                # is enabled.
                frames = raw_obs_current[:, -layout.total_floats:]
                emergence.update_batch(frames, actions_np, layout)
                if tape_recorder is not None:
                    for i in range(args.n_envs):
                        frame = frames[i]
                        self_entities = layout.all_entities(frame)
                        decision = decision_record(
                            t=time.time(), self_values=frame, entity_dicts=self_entities,
                            action=actions_np[i], reward_components=infos[i]["reward_components"],
                            difficulty=(infos[i].get("scenario") or {}).get("difficulty"), layout=layout,
                        )
                        tape_recorder.record_decision(i, decision)

                # Time-limit bootstrap fix (Pardo et al. 2018), vectorized --
                # see train.py's single-env version for the full rationale.
                # Only the truncated (not terminal) workers need a bootstrap
                # value folded into their reward this step.
                trunc_idx = np.where(truncateds)[0]
                if len(trunc_idx) > 0:
                    trunc_raw = np.stack([infos[i]["terminal_observation"] for i in trunc_idx])
                    trunc_normed = obs_normalizer.normalize(trunc_raw, update=False)
                    with torch.no_grad():
                        bootstrap_values = policy.get_value(torch.as_tensor(trunc_normed, dtype=torch.float32, device=device)).cpu().numpy()
                    rewards[trunc_idx] = rewards[trunc_idx] + args.gamma * bootstrap_values

                stop_flags = terminals | truncateds
                normalized_rewards = reward_normalizer.normalize(rewards, stop_flags)
                buffer.add(obs, actions_np, log_probs.cpu().numpy(), values.cpu().numpy(), normalized_rewards, stop_flags.astype(np.float32))

                for i in np.where(stop_flags)[0]:
                    episode_rewards_this_update.append(episode_reward[i])
                    episode_lengths_this_update.append(episode_length[i])
                    won = bool(infos[i]["reward_components"].get("opponent_knockout", 0.0) > 0.0)
                    outcome = "won" if won else ("lost" if terminals[i] else "timeout")
                    outcomes_this_update[outcome] += 1
                    episode_components_this_update.append(dict(episode_components[i]))
                    # OGRL-20260817-028 Sec3.2/Sec8.6: the REAL reset seed
                    # (not the worker-index placeholder train_vec.py used to
                    # log -- ghost replay needs the actual seed) and the
                    # scenario THIS episode was actually sampled at (not
                    # d_max/the sampler's current state).
                    ended_scenario = infos[i].get("scenario") or {}
                    ended_seed = infos[i].get("seed")
                    ended_difficulty = ended_scenario.get("difficulty")
                    if ended_difficulty is not None:
                        sampler.record_episode_outcome(ended_difficulty, won, ended_scenario.get("opponents", 1) or 1)
                    # Opponent-count curriculum advances on its own gate, kept
                    # separate from difficulty so neither can advance the other.
                    sampler.record_opponent_outcome(ended_scenario.get("opponents", 1) or 1, won)
                    logger.log_episode({
                        "t": time.time(), "global_step": global_step, "worker": int(i),
                        "seed": ended_seed if ended_seed is not None else episode_seed_used[i],
                        "length": int(episode_length[i]), "outcome": outcome,
                        "total_reward": float(episode_reward[i]), "components": dict(episode_components[i]),
                        "duration_seconds": time.time() - episode_start_time[i],
                        "d": ended_difficulty, "opponents": ended_scenario.get("opponents"),
                        "species": ended_scenario.get("species"), "armed": (ended_scenario.get("weapons") or 0) > 0,
                        "soft_reset": ended_scenario.get("soft_reset"),
                        "level": infos[i].get("level"),
                    })
                    if tape_recorder is not None:
                        sampled_worker0 = (i == 0 and args.tape_every > 0 and update % args.tape_every == 0)
                        tape_recorder.episode_ended(
                            int(i), update, ended_seed, outcome, float(episode_reward[i]),
                            ended_difficulty, sampled_worker0,
                            native_trace_path=infos[i].get("native_trace_path"),
                        )
                    episode_reward[i] = 0.0
                    episode_length[i] = 0
                    episode_components[i] = defaultdict(float)
                    episode_start_time[i] = time.time()

                obs = obs_normalizer.normalize(raw_next_obs)
                raw_obs_current = raw_next_obs

            collection_seconds = max(1e-6, time.monotonic() - collection_start)

            # OGRL-20260817-028 Sec8.2: action statistics -- press_prob's
            # SPREAD across the batch (std across n_envs of each env's own
            # mean press rate), not just the marginal mean, is what actually
            # separated a coin-flip policy from a controller in run5-9
            # (-027 Sec1.3: 0.09 in the broken runs, 0.35 in run9). One numpy
            # reduction over this update's collected actions.
            actions_arr = np.stack(actions_this_update)  # (n_steps, n_envs, 8)
            button_names = ["jump", "crouch", "attack", "grab", "drop", "walk"]
            button_pressed = actions_arr[:, :, 2:8] > 0.5  # (n_steps, n_envs, 6)
            per_env_press_rate = button_pressed.mean(axis=0)  # (n_envs, 6)
            press_prob = {name: float(per_env_press_rate[:, j].mean()) for j, name in enumerate(button_names)}
            press_prob_spread = {name: float(per_env_press_rate[:, j].std()) for j, name in enumerate(button_names)}
            flip_events = np.abs(np.diff(button_pressed.astype(np.float32), axis=0))  # (n_steps-1, n_envs, 6)
            flip_rate = {name: float(flip_events[:, :, j].mean()) for j, name in enumerate(button_names)}
            move = actions_arr[:, :, 0:2]
            with torch.no_grad():
                continuous_sigma = float(torch.clamp(policy.continuous_log_std, -5.0, 2.0).exp().mean().item())
            action_stats = {
                "press_prob": press_prob, "press_prob_spread": press_prob_spread, "flip_rate": flip_rate,
                "move_abs_mean": float(np.mean(np.abs(move))), "move_std": float(np.std(move)),
                "continuous_sigma": continuous_sigma,
            }
            emergence_snapshot = emergence.snapshot()

            with torch.no_grad():
                last_values = policy.get_value(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()

            # Collect the remote rollouts. The learner blocks here until every
            # worker has returned a full n_steps rollout collected with THIS
            # update's weights -- that is what keeps the algorithm on-policy.
            remote_rollouts = []
            if remote_conns:
                _rt0 = time.monotonic()
                for _c in list(remote_conns):
                    try:
                        _msg = _recv(_c)
                    except Exception as _exc:
                        print(f"remote worker lost ({type(_exc).__name__}: {_exc}); "
                              f"continuing with the remaining participants", flush=True)
                        remote_conns.remove(_c)
                        continue
                    remote_rollouts.append(_msg)
                    for _ep in _msg.get("episodes", []):
                        if _ep.get("difficulty") is not None:
                            sampler.record_episode_outcome(_ep["difficulty"], _ep["won"], _ep.get("opponents", 1) or 1)
                        sampler.record_opponent_outcome(_ep.get("opponents", 1) or 1, _ep["won"])
                    global_step += args.n_steps * _msg["obs"].shape[1]
                remote_wait_seconds = time.monotonic() - _rt0
            else:
                remote_wait_seconds = 0.0

            if remote_rollouts:
                merged = buffer.merged_with(remote_rollouts, device)
                all_last = np.concatenate([last_values] + [r["last_values"] for r in remote_rollouts])
                batch = merged.to_tensors(all_last, args.gamma, args.gae_lambda)
            else:
                batch = buffer.to_tensors(last_values, args.gamma, args.gae_lambda)
            buffer.reset()

            # Entropy anneal (OGRL-20260816-021 Sec 2.4/-023): mutate args.entropy_coef
            # in place before each ppo_update() call rather than changing that
            # function's signature -- it already reads args.entropy_coef directly.
            # None (--entropy-coef-final not set) leaves args.entropy_coef untouched,
            # exactly reproducing the old constant-coefficient behavior.
            if args.entropy_coef_final is not None:
                anneal_progress = min(1.0, global_step / max(1, args.entropy_anneal_steps))
                args.entropy_coef = args.entropy_coef_start + (args.entropy_coef_final - args.entropy_coef_start) * anneal_progress

            torch.set_num_threads(args.update_torch_threads)
            stats = ppo_update(policy, optimizer, batch, args)

            update += 1
            explained_var = _explained_variance(batch["values"].cpu().numpy(), batch["returns"].cpu().numpy())
            # NaN (not None) here used to be a real dashboard-breaking bug: Python's
            # json.dumps serializes float("nan") as the bare token `NaN`, which is
            # valid in Python's JSON dialect but NOT valid JSON -- the browser's
            # native JSON.parse() (used by fetch().json()) throws a SyntaxError on
            # it, which silently killed the whole metrics/episodes/events fetch in
            # the dashboard's selectRun() for any run that ever had a zero-episode
            # update window (i.e. almost every run, since a PPO update fires every
            # n_steps regardless of whether a fight has finished yet). Use None,
            # like episode_total_median right below already correctly does, so the
            # printf below stays the only place a literal NaN float is allowed to
            # exist -- it never crosses the JSON boundary.
            mean_ep_reward = float(np.mean(episode_rewards_this_update)) if episode_rewards_this_update else None
            mean_ep_length = float(np.mean(episode_lengths_this_update)) if episode_lengths_this_update else None
            steps_per_second = (args.n_steps * args.n_envs) / collection_seconds

            mean_reward_str = f"{mean_ep_reward:.3f}" if mean_ep_reward is not None else "n/a"
            mean_len_str = f"{mean_ep_length:.1f}" if mean_ep_length is not None else "n/a"
            print(
                f"update={update} step={global_step}/{args.total_timesteps} "
                f"phase={curriculum.phase_name(global_step)} "
                f"episodes={len(episode_rewards_this_update)} mean_reward={mean_reward_str} mean_len={mean_len_str} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} approx_kl={stats['approx_kl']:.4f} "
                f"explained_var={explained_var:.3f} sps={steps_per_second:.1f}"
            )
            log_writer.writerow([
                global_step, update, mean_ep_reward, mean_ep_length, len(episode_rewards_this_update),
                stats["policy_loss"], stats["value_loss"], stats["entropy"], stats["approx_kl"], stats["clip_fraction"],
                explained_var, curriculum.phase_name(global_step), steps_per_second,
            ])
            log_file.flush()

            cycle_end = time.monotonic()
            cycle_seconds = max(1e-6, cycle_end - previous_cycle_end)
            previous_cycle_end = cycle_end
            component_means = {}
            if episode_components_this_update:
                names = set().union(*(d.keys() for d in episode_components_this_update))
                component_means = {name: float(np.mean([d.get(name, 0.0) for d in episode_components_this_update])) for name in names}
            logger.log_update({
                "t": time.time(), "global_step": global_step, "update": update,
                "phase": curriculum.phase_name(global_step),
                "reward": {
                    "episode_total": mean_ep_reward,
                    "episode_total_median": float(np.median(episode_rewards_this_update)) if episode_rewards_this_update else None,
                    "episode_length": mean_ep_length, "episodes_completed": len(episode_rewards_this_update),
                    "components": component_means,
                },
                "outcomes": outcomes_this_update,
                "ppo": {
                    "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"], "approx_kl": stats["approx_kl"],
                    "clip_fraction": stats["clip_fraction"], "explained_variance": explained_var,
                    "entropy": stats["entropy"], "entropy_random_reference": entropy_random_reference,
                    "learning_rate": args.learning_rate, "entropy_coef": args.entropy_coef,
                    "nan_skips": stats["nan_skips"],  # 2026-08-17: non-finite-loss/grad minibatches skipped
                                                        # this update, see ppo_update's NaN guard in train.py
                },
                # kl_spike (OGRL-20260817-028 Sec8.2): run9 had a single
                # approx_kl of 12.87 against a 0.02 target, buried in a
                # percentile until someone went looking for it -- an
                # explicit boolean flag per update is the fix, not a
                # threshold anyone has to remember to check for.
                "kl_spike": bool(stats["approx_kl"] > args.target_kl * 10),
                "curriculum_live": {
                        "opponents_max": sampler.opponents_max,
                        "opponent_win_rates": {str(k): v for k, v in sampler.opponent_win_rates().items()},
                    # Reward-SHAPING curriculum (Curriculum, unchanged concern).
                    "closing_distance_weight": reward_config.closing_distance_weight,
                    "stall_penalty_weight": reward_config.stall_penalty_weight,
                    # Environment-COMPOSITION curriculum (ScenarioSampler, Sec3.2) --
                    # nested under "scenario" so dashboard code can tell the two apart
                    # without guessing from key names.
                    "scenario": sampler.snapshot(),
                },
                "action_stats": action_stats,
                "emergence": emergence_snapshot,
                "perf": {
                    "collection_seconds": collection_seconds, "cycle_seconds": cycle_seconds,
                    "steps_per_second_collection": steps_per_second,
                    "steps_per_second_cycle": (args.n_steps * args.n_envs) / cycle_seconds,
                    **_perf_with_reset_share(vec_env.drain_perf(), cycle_seconds),
                },
            })

            if args.checkpoint_path and update % args.checkpoint_every_updates == 0:
                _save_checkpoint(args.checkpoint_path, policy, optimizer, obs_normalizer, reward_normalizer, global_step)
        run_status = "completed"  # reached either by the while condition going false, or the dashboard-stop break above
    finally:
        # OGRL-20260816-018 -- see train.py's identical fix for the full
        # rationale: confirmed on run5 itself, which exited 7 updates
        # (~57k steps) past its last periodic checkpoint with nothing
        # capturing that final state. One more unconditional save here
        # covers every exit path (natural completion, Ctrl+C, SIGTERM).
        if args.checkpoint_path:
            _save_checkpoint(args.checkpoint_path, policy, optimizer, obs_normalizer, reward_normalizer, global_step)
        # OGRL-20260817-028 Sec10: "a run_stop event on every exit path" --
        # the canonical terminal marker, distinct from "stop_requested"
        # above (the WHY, only fired on the dashboard-stop path). Fires here
        # for every exit: natural completion, dashboard stop, Ctrl+C/SIGTERM
        # (translated to KeyboardInterrupt), or an uncaught exception --
        # sys.exc_info() is non-None only in the last case.
        exc_type = sys.exc_info()[0]
        logger.log_event(
            "run_stop", f"{run_id} exiting (status={run_status})",
            body=f"at global_step={global_step}" + (f", exception={exc_type.__name__}" if exc_type else ""),
        )
        vec_env.close()
        log_file.close()
        logger.finish(run_status, global_step)


if __name__ == "__main__":
    main()
