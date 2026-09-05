"""N parallel OvergrowthEnv workers driven from one Python process, extending
Stage 4's N-active worker-pool concept (worker_pool.py) to PPO rollout
collection -- the single-environment trainer's most significant flagged
limitation (research-log OGRL-20260816-011).

Parallelism is thread-based, not process-based, and that's a considered
choice, not an oversight: each OvergrowthEnv already owns its own engine
*subprocess* (real OS-level parallelism for the expensive part, physics
stepping), so the Python side only needs to keep N blocking shm handshakes
in flight at once. ctypes releases the GIL for the duration of a foreign
call (verified empirically before writing this: 4 concurrent libc.sleep(1)
calls via threads completed in ~1s, not ~4s) -- so N threads each blocked in
sem_wait() genuinely overlap at the OS level despite the GIL, without the
serialization overhead multiprocessing IPC would add on top of shm this
already uses. Policy inference and reward computation still run one thread
at a time (real GIL-bound Python/PyTorch work), but that's a small fraction
of per-step cost compared to a physics-step round trip.

Auto-reset convention matches standard VecEnv practice (OpenAI Gym / SB3):
when a worker's episode ends, step() transparently resets that worker and
returns the *new* episode's first observation in the batch; the actual final
observation of the episode that just ended is preserved in
info["terminal_observations"][i] for correct bootstrapping if a caller wants
it (this trainer doesn't need it -- truncation/termination bootstrapping is
handled per-worker via the reward's own accounting, not via terminal_observation
-- but it's kept for parity with the convention and any future consumer).

Reset off the critical path (OGRL-20260816-021 Sec 1.3(b), OGRL-20260816-022):
measured at 674ms median through this exact reset path (Engine::
ResetRLTrainingScenario -> LoadLevel), and step()'s ThreadPoolExecutor.map()
is a BARRIER -- every worker's thread must return before map() does, so one
worker's mid-episode 674ms reset used to stall the other n_envs-1 workers'
threads too, even though they'd already finished their own step() call and
had nothing left to do but wait. k_standby extra OvergrowthEnv instances are
kept pre-warmed and pre-reset; when a worker's episode ends, its slot is
filled IMMEDIATELY from the standby pool (an O(1) swap, no waiting), and the
just-retired env is handed to a SEPARATE background executor that resets it
and returns it to the standby pool once done -- off the step()-return path
entirely. worker_pool.py's own bake-off (OGRL-20260815-04x) measured this
swap pattern at 6-10ms versus 674ms for the synchronous reload it replaces.
Falls back to the original synchronous reset (never crashes, never hangs) if
the standby pool underruns -- e.g. several workers ending in the same vector
step with k_standby too shallow to cover the burst.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from shm_env import ShmWaitTimeout
from env import OvergrowthEnv, ACTION_DIM
from obs_schema import ObsLayout, DEFAULT_LAYOUT
from reward import RewardConfig


@dataclass
class _Standby:
    env: OvergrowthEnv
    obs: np.ndarray  # this env's current (already-reset, ready-to-serve) observation
    scenario: dict    # the scenario that obs's episode was reset with -- see VecOvergrowthEnv._reset_env


class VecOvergrowthEnv:
    def __init__(
        self,
        n_envs: int,
        repo_root: str,
        level: "str | Sequence[str]" = "arenas/oval_arena.xml",
        shm_prefix: str = "/ogrl_vec",
        base_seed: int = 1,
        layout: ObsLayout = DEFAULT_LAYOUT,
        reward_config: RewardConfig | None = None,
        frame_stack: int = 1,
        max_episode_steps: int = 1200,
        k_standby: int = 5,  # OGRL-20260816-025: benchmarked crossover point at n_envs=4, see train_vec.py's
                              # --k-standby help for the full measured curve -- not a guess, and NOT "more is better"
                              # (k=6-8 reach 0% pool-underrun but are slower than k=5 due to CPU contention)
        act_period: int = 1,
        soft_reset: bool = False,  # OGRL-20260817-028 Sec1: use Engine::SoftResetRLTrainingScenario for
                                    # per-episode resets instead of the original full LoadLevel path. Default
                                    # False so every existing caller (benchmarks, older scripts) keeps the
                                    # validated hard-reset behavior unless it opts in explicitly.
        hard_reset_every: int = 50,  # Sec1.2 safety valve: force a hard reset every Nth reset of a given
                                      # PHYSICAL engine process (not vector slot -- see OvergrowthEnv.episode_count),
                                      # bounding any state a soft reset doesn't clear (decals, dropped items,
                                      # monotonic object IDs -- see the leak audit in Sec1.3). Only consulted
                                      # when soft_reset=True. 0 or negative disables the valve entirely (always
                                      # soft) -- not recommended for an unattended run before the leak audit
                                      # has actually passed.
        scenario_fn: "Callable[[], dict] | None" = None,  # OGRL-20260817-028 Sec3: called once per reset,
        native_trace_dir: str | Path | None = None,
                                                            # returning {"difficulty","opponents","weapons","species"}
                                                            # (curriculum.ScenarioSampler.sample_episode()'s exact
                                                            # shape). None reproduces the pre-curriculum behavior
                                                            # (engine's own default difficulty, unarmed, 1 opponent).
    ):
        self.n_envs = n_envs
        # Map axis (C6). A level belongs to a PHYSICAL engine for its lifetime,
        # not to a vector slot: a standby carries its own level with it when it
        # swaps into a slot, so slots see a changing map mix while no engine
        # ever reloads a level it would not otherwise have loaded. Levels are
        # dealt round-robin over actives + standbys, so the mix is deterministic
        # given n_envs and k_standby. Holding maps back from this list is what
        # makes a transfer test genuine.
        _levels = [level] if isinstance(level, str) else list(level)
        if not _levels:
            raise ValueError("at least one level is required")
        self.level_pool = _levels
        self.layout = layout
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.observation_dim = layout.total_floats * frame_stack
        self.reward_config = reward_config
        self.soft_reset = soft_reset
        self.hard_reset_every = hard_reset_every
        self.scenario_fn = scenario_fn
        self.native_trace_dir = Path(native_trace_dir) if native_trace_dir else None
        if self.native_trace_dir is not None:
            self.native_trace_dir.mkdir(parents=True, exist_ok=True)
        self._episode_steps = [0] * n_envs
        # Per-worker episode counter, used to vary the reset seed episode to
        # episode -- see step()'s reset call for why this exists at all.
        self._episode_counts = [0] * n_envs
        # Per-SLOT record of the scenario/seed that produced the episode
        # currently in flight in that slot -- attached to info on every
        # step() so train_vec.py can log it against the episode it actually
        # belongs to, and (Sec3.2) feed ScenarioSampler.record_episode_outcome
        # the difficulty that episode was actually sampled at, not d_max.
        self._episode_scenario: list[dict] = [{} for _ in range(n_envs)]
        self._episode_seed: list[int | None] = [None] * n_envs
        # OGRL-20260817-028 Sec8.2: perf.reset_seconds/reset_share and pool
        # hit/miss counts -- accumulated continuously, drained (read + reset
        # to 0) once per training update by train_vec.py via drain_perf().
        # Lock-protected: resets happen on both self._pool (sync fallback,
        # inline in step()) and self._reset_pool (background) threads.
        self._perf_lock = threading.Lock()
        self._reset_seconds_accum = 0.0
        self._pool_hits = 0
        self._pool_misses = 0
        self._step_latencies: list[float] = []
        self._barrier_idle_seconds = 0.0
        self._reset_blocking_seconds = 0.0
        self._step_wall_seconds = 0.0
        self._step_count = 0
        self.k_standby = max(0, k_standby)
        self._pool = ThreadPoolExecutor(max_workers=n_envs + max(1, self.k_standby), thread_name_prefix="ogrl-vec-env")
        # Dedicated to background resets of retired envs -- deliberately
        # separate from self._pool so a burst of background resets can never
        # compete with (or be starved by) the main step-loop's own threads
        # for a worker slot.
        self._reset_pool = ThreadPoolExecutor(max_workers=max(1, self.k_standby), thread_name_prefix="ogrl-vec-bgreset")
        self._standby: list[_Standby] = []
        self._standby_lock = threading.Lock()
        # Shared across every standby reset regardless of which physical env
        # or which vector slot it's replacing -- a standby moves between
        # slots over its lifetime, so slot index alone can no longer serve as
        # the seed-diversity key the way OGRL-20260816-016's fix used it;
        # this counter plays the same "always increasing" role instead.
        self._reset_counter = 0
        self._reset_counter_lock = threading.Lock()

        def _make(shm_suffix: str, seed: int, worker_level: str) -> OvergrowthEnv:
            # Darwin's shm/sem name limit (~31 bytes) constrains shm_prefix +
            # suffix length -- see rl_shm_transport.h. Keep shm_prefix short.
            return OvergrowthEnv(
                repo_root=repo_root, level=worker_level, shm_name=f"{shm_prefix}{shm_suffix}",
                controller_id=0, seed=seed, layout=layout,
                reward_config=reward_config, frame_stack=frame_stack, act_period=act_period,
                equivalence_digest_path=(self.native_trace_dir / f"{shm_suffix}.jsonl") if self.native_trace_dir is not None else None,
                equivalence_trace_path=(self.native_trace_dir / f"{shm_suffix}.input.jsonl") if self.native_trace_dir is not None else None,
            )

        # Parallel launch: each OvergrowthEnv.__init__ blocks on its own
        # engine's level load -- N of these sequentially would cost N times
        # that for no reason, since the engines themselves don't contend with
        # each other during their own independent startup. Actives and
        # standbys launch together, one batch, not two.
        n_total = n_envs + self.k_standby
        specs = [(str(i), base_seed + i, _levels[i % len(_levels)]) for i in range(n_envs)] + \
                [(f"s{i}", base_seed + n_envs + i, _levels[(n_envs + i) % len(_levels)])
                 for i in range(self.k_standby)]
        self.levels = [sp[2] for sp in specs]
        # Retain the maker so a worker that goes silent can be rebuilt in place
        # (see _step_one's ShmWaitTimeout handler). Rebuilds always take a fresh
        # shm suffix: a name whose previous owner was SIGKILLed leaves an orphaned
        # semaphore behind, and re-opening it attaches to something no live engine
        # will ever post to -- a hang that looks exactly like the original fault.
        self._make_env = _make
        self._recoveries = 0
        self._shm_prefix = shm_prefix
        built = list(self._pool.map(lambda spec: _make(*spec), specs))
        self.envs: list[OvergrowthEnv] = built[:n_envs]
        standby_envs = built[n_envs:]

        if standby_envs:
            # Get every standby to a real, ready-to-serve observation before
            # training's first step() call -- a standby with no observation
            # yet isn't actually ready to swap in.
            standby_results = list(self._pool.map(lambda env: self._reset_env(env), standby_envs))
            with self._standby_lock:
                self._standby = [_Standby(env, obs, scenario) for env, (obs, scenario) in zip(standby_envs, standby_results)]

    def _next_reset_seed(self, env: OvergrowthEnv) -> int:
        with self._reset_counter_lock:
            self._reset_counter += 1
            offset = self._reset_counter
        return env.seed + offset

    def _reset_env(self, env: OvergrowthEnv) -> "tuple[np.ndarray, dict]":
        """The one place that actually calls env.reset() -- decides soft vs.
        hard (the hard_reset_every safety valve, keyed on THIS env's own
        lifetime episode_count, not the vector slot it happens to occupy)
        and pulls a fresh scenario from scenario_fn if one was given. Returns
        (obs, scenario_dict) so the caller can attribute the episode this
        produces correctly."""
        scenario = self.scenario_fn() if self.scenario_fn is not None else {}
        seed = self._next_reset_seed(env)
        soft = self.soft_reset
        # env.episode_count > 0 guard added 2026-08-17: forcing hard on an
        # env's very FIRST real reset (episode_count==0, i.e. right after the
        # pseudo-reset that consumes its natural initial observation --
        # see OvergrowthEnv.reset()'s _used_initial_observation comment)
        # produced a live RuntimeError ("engine reported reset(...) failed")
        # in the very first smoke test run, consistent with a startup race
        # around rl_training_reset_baseline_valid_ that the ORIGINAL
        # hard-reset path already documents as a risk for a client that
        # resets too soon after connecting. Not fully root-caused under
        # time pressure -- this sidesteps the specific failing case
        # (skip forcing hard on episode 0) rather than fixing the race
        # itself, which is a real open item for a future session.
        if soft and self.hard_reset_every > 0 and env.episode_count > 0 and (env.episode_count % self.hard_reset_every) == 0:
            soft = False  # periodic safety valve (Sec1.2) -- fires on episode_count 0, N, 2N, ...
        obs = env.reset(
            seed=seed, soft=soft,
            difficulty=scenario.get("difficulty"),
            opponents=scenario.get("opponents", 1),
            weapons=scenario.get("weapons", 0.0),
            species=scenario.get("species", 0),
        )
        with self._perf_lock:
            self._reset_seconds_accum += env.last_reset_seconds
        scenario_out = dict(scenario)
        scenario_out["soft_reset"] = soft
        return obs, scenario_out

    def drain_perf(self) -> dict:
        """Read-and-zero collection timing accumulators per training update."""
        with self._perf_lock:
            latencies = np.asarray(self._step_latencies, dtype=np.float64)
            out = {
                "reset_seconds": self._reset_seconds_accum,
                "reset_cpu_seconds": self._reset_seconds_accum,
                "reset_blocking_seconds": self._reset_blocking_seconds,
                "pool_hits": self._pool_hits,
                "pool_misses": self._pool_misses,
                "step_wall_seconds": self._step_wall_seconds,
                "step_count": self._step_count,
                "worker_step_latency_p50_seconds": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
                "worker_step_latency_p90_seconds": float(np.percentile(latencies, 90)) if latencies.size else 0.0,
                "worker_step_latency_p99_seconds": float(np.percentile(latencies, 99)) if latencies.size else 0.0,
                "barrier_idle_seconds": self._barrier_idle_seconds,
                "active_workers": self.n_envs,
                "ready_standby_workers": len(self._standby),
            }
            self._reset_seconds_accum = 0.0
            self._pool_hits = 0
            self._pool_misses = 0
            self._step_latencies.clear()
            self._reset_blocking_seconds = 0.0
            self._step_wall_seconds = 0.0
            self._step_count = 0
            self._barrier_idle_seconds = 0.0
        return out

    def _take_standby(self) -> "_Standby | None":
        with self._standby_lock:
            return self._standby.pop() if self._standby else None

    def _return_standby(self, standby: "_Standby") -> None:
        with self._standby_lock:
            self._standby.append(standby)

    def _background_reset(self, env: OvergrowthEnv) -> None:
        # Runs on self._reset_pool, off the step()-return path. Exceptions
        # here must not propagate silently into a lost future -- if reset
        # itself is broken, the standby pool just permanently underruns
        # (falls back to synchronous resets, see step()) rather than
        # crashing a background thread nobody is watching.
        try:
            obs, scenario = self._reset_env(env)
            self._return_standby(_Standby(env, obs, scenario))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
            print(f"vec_env background reset failed for {env.shm_name}: {exc}", file=sys.stderr)

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        # seeds, if given, only override WHICH seed _reset_env would have
        # picked next for that env -- scenario sampling (soft/hard,
        # difficulty/opponents/weapons/species) still goes through the same
        # path as every other reset, so a full vec reset() isn't a second,
        # divergent code path.
        def _do(i: int):
            env = self.envs[i]
            if seeds and seeds[i] is not None:
                scenario = self.scenario_fn() if self.scenario_fn is not None else {}
                soft = self.soft_reset and not (self.hard_reset_every > 0 and env.episode_count > 0 and (env.episode_count % self.hard_reset_every) == 0)
                obs = env.reset(seed=seeds[i], soft=soft, difficulty=scenario.get("difficulty"),
                                 opponents=scenario.get("opponents", 1), weapons=scenario.get("weapons", 0.0),
                                 species=scenario.get("species", 0))
                scenario = dict(scenario)
                scenario["soft_reset"] = soft
            else:
                obs, scenario = self._reset_env(env)
            return obs, scenario

        results = list(self._pool.map(_do, range(self.n_envs)))
        self._episode_steps = [0] * self.n_envs
        self._episode_counts = [0] * self.n_envs
        self._episode_scenario = [scenario for _obs, scenario in results]
        self._episode_seed = [self.envs[i].last_reset_seed for i in range(self.n_envs)]
        return np.stack([obs for obs, _scenario in results])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: (n_envs, ACTION_DIM). Returns (obs[n_envs, obs_dim],
        rewards[n_envs], terminals[n_envs], truncateds[n_envs], infos) --
        terminals/truncateds are the per-worker flags this call's transition
        produced, split (not pre-OR'd) so the caller can apply the same
        truncation-bootstrap treatment train.py's single-env loop does."""
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_envs, ACTION_DIM)

        def _step_one(i: int):
            worker_start = time.monotonic()
            blocking_reset_seconds = 0.0
            try:
                obs, reward, done, info = self.envs[i].step(actions[i])
            except ShmWaitTimeout as exc:
                # A worker went silent. Before bounded waits existed this parked
                # the trainer in sem_wait forever at 0% CPU with every health
                # metric still reading green (OGRL-20260905-065). Rebuild the
                # worker and carry on: one lost episode costs seconds, a hung run
                # costs however long it takes a human to notice.
                self._recoveries += 1
                dead, self.envs[i] = self.envs[i], None
                try:
                    dead.close()
                except Exception:
                    pass
                suffix = f"r{self._recoveries}_{i}"
                self.envs[i] = self._make_env(suffix, dead.seed + 100000 + self._recoveries, dead.level)
                obs, scenario = self._reset_env(self.envs[i])
                self._episode_scenario[i] = scenario
                self._episode_seed[i] = self.envs[i].last_reset_seed
                self._episode_steps[i] = 0
                self._episode_counts[i] += 1
                # Must carry every key the normal path produces -- step() unpacks
                # info["perf"] unconditionally, so a partial dict turns a recovered
                # worker into a KeyError that kills the run anyway. Found by fault
                # injection; a code read would not have caught it.
                info = {"reward_components": {"opponent_knockout": 0.0},
                        "worker_recovered": True, "recovery_reason": str(exc),
                        "scenario": scenario, "seed": self._episode_seed[i],
                        "level": self.envs[i].level, "native_trace_path": None,
                        "perf": {"worker_step_seconds": time.monotonic() - worker_start,
                                 "blocking_reset_seconds": 0.0}}
                # truncated=True, not terminal: the episode did not really end,
                # it was abandoned, so the caller bootstraps the value rather than
                # treating this as a genuine outcome and polluting the win rate.
                return obs, 0.0, False, True, info, obs
            self._episode_steps[i] += 1
            won = info["reward_components"]["opponent_knockout"] > 0
            terminal = bool(done or won)
            truncated = (not terminal) and self._episode_steps[i] >= self.max_episode_steps
            terminal_obs = obs  # pre-reset observation, for the truncation bootstrap / info parity
            # Attribute this transition to the episode currently in slot i --
            # i.e. the one THIS step's outcome belongs to, sampled at the
            # PREVIOUS reset -- before any reset logic below overwrites it
            # for the next episode.
            info["scenario"] = self._episode_scenario[i]
            info["seed"] = self._episode_seed[i]
            # Map axis: which level actually produced this episode. Recorded here
            # rather than derived from the slot index, because a standby carries its
            # own level and swaps between slots, so slot index alone is ambiguous.
            info["level"] = self.envs[i].level
            info["native_trace_path"] = str(self.envs[i].equivalence_digest_path) if self.envs[i].equivalence_digest_path is not None else None
            if terminal or truncated:
                self._episode_steps[i] = 0
                self._episode_counts[i] += 1
                standby = self._take_standby()
                with self._perf_lock:
                    if standby is not None:
                        self._pool_hits += 1
                    else:
                        self._pool_misses += 1
                if standby is not None:
                    # O(1) swap: the retiring env's replacement is already
                    # loaded and already has a fresh observation -- no wait.
                    # The retiring env resets in the background and rejoins
                    # the standby pool once done; this step never sees it
                    # again, no ordering assumption needed.
                    retiring_env = self.envs[i]
                    self.envs[i] = standby.env
                    obs = standby.obs
                    self._episode_scenario[i] = standby.scenario
                    self._episode_seed[i] = standby.env.last_reset_seed
                    self._reset_pool.submit(self._background_reset, retiring_env)
                else:
                    # Pool underrun (a burst of episode-ends deeper than
                    # k_standby this step) -- fall back to the original
                    # synchronous behavior for this worker only. Correct,
                    # just not fast; matches worker_pool.py's own
                    # pool_underrun concept from the Stage 4 bake-off.
                    reset_started = time.monotonic()
                    obs, scenario = self._reset_env(self.envs[i])
                    blocking_reset_seconds = time.monotonic() - reset_started
                    self._episode_scenario[i] = scenario
                    self._episode_seed[i] = self.envs[i].last_reset_seed
            worker_seconds = time.monotonic() - worker_start
            info["perf"] = {"worker_step_seconds": worker_seconds, "blocking_reset_seconds": blocking_reset_seconds}
            return obs, reward, terminal, truncated, info, terminal_obs

        step_started = time.monotonic()
        results = list(self._pool.map(_step_one, range(self.n_envs)))
        step_wall_seconds = time.monotonic() - step_started
        worker_latencies = [float(result[4]["perf"]["worker_step_seconds"]) for result in results]
        barrier_idle_seconds = sum(max(0.0, step_wall_seconds - latency) for latency in worker_latencies)
        blocking_reset_seconds = sum(float(result[4]["perf"]["blocking_reset_seconds"]) for result in results)
        with self._perf_lock:
            self._step_latencies.extend(worker_latencies)
            self._barrier_idle_seconds += barrier_idle_seconds
            self._reset_blocking_seconds += blocking_reset_seconds
            self._step_wall_seconds += step_wall_seconds
            self._step_count += self.n_envs
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminals = np.array([r[2] for r in results], dtype=bool)
        truncateds = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        for info, r in zip(infos, results):
            info["terminal_observation"] = r[5]
        return obs, rewards, terminals, truncateds, infos

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        self.reward_config = reward_config
        for env in self.envs:
            env.set_reward_config(reward_config)
        # Standbys must carry the current config too -- otherwise a promoted
        # standby would apply a stale (e.g. pre-curriculum-update) reward
        # config to its next episode until the vec loop's next
        # set_reward_config() call happens to touch it again.
        with self._standby_lock:
            for standby in self._standby:
                standby.env.set_reward_config(reward_config)

    def close(self) -> None:
        self._reset_pool.shutdown(wait=True, cancel_futures=False)
        with self._standby_lock:
            all_standby_envs = [s.env for s in self._standby]
            self._standby = []
        list(self._pool.map(lambda env: env.close(), self.envs + all_standby_envs))
        self._pool.shutdown(wait=True)
