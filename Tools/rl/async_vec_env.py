"""Independent frozen-policy rollout collection for Overgrowth.

The synchronous vector environment waits for every engine at every policy
decision.  This collector keeps one step future in flight per engine and
immediately schedules the next action for whichever engine is ready.  A
worker performs its own reset inside that worker future, so a slow reset or a
long collision sequence does not stop unrelated workers.

This is still on-policy PPO: the caller supplies one action callback for the
whole rollout and does not update the policy until every worker has contributed
exactly ``n_steps`` transitions.  The only changed boundary is wall-clock
scheduling.  ``AsyncRollout`` is time-major by worker index, so the existing
vector GAE and PPO code can consume it without changing trajectory semantics.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from env import ACTION_DIM, OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT, ObsLayout
from reward import RewardConfig


@dataclass
class _StepResult:
    next_obs: np.ndarray
    terminal_observation: np.ndarray
    reward: float
    terminal: bool
    truncated: bool
    info: dict


@dataclass
class AsyncRollout:
    """Collected transitions in the shape expected by ``VecRolloutBuffer``."""

    obs: np.ndarray
    raw_obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    infos: list[list[dict]]
    last_raw_obs: np.ndarray
    wall_seconds: float
    batches: int
    ready_batch_sizes: list[int]


class AsyncVecOvergrowthEnv:
    """N independent Overgrowth workers with asynchronous rollout stepping."""

    def __init__(
        self,
        n_envs: int,
        repo_root: str,
        level: "str | Sequence[str]" = "arenas/oval_arena.xml",
        shm_prefix: str = "/ogrl_async",
        base_seed: int = 1,
        layout: ObsLayout = DEFAULT_LAYOUT,
        reward_config: RewardConfig | None = None,
        frame_stack: int = 1,
        max_episode_steps: int = 1200,
        act_period: int = 1,
        soft_reset: bool = False,
        hard_reset_every: int = 50,
        scenario_fn: Callable[[], dict] | None = None,
        native_trace_dir: str | Path | None = None,
    ):
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        self.n_envs = n_envs
        # Map axis. A worker holds one level for its whole life: assigning per
        # worker rather than per episode means every PPO batch mixes maps, while
        # no episode ever pays a level reload it would not otherwise pay. Levels
        # are dealt round-robin so the mix is deterministic given n_envs.
        levels = [level] if isinstance(level, str) else list(level)
        if not levels:
            raise ValueError("at least one level is required")
        self.levels = [levels[i % len(levels)] for i in range(n_envs)]
        self.level = self.levels[0]
        self.layout = layout
        self.frame_stack = max(1, frame_stack)
        self.observation_dim = layout.total_floats * self.frame_stack
        self.max_episode_steps = max_episode_steps
        self.soft_reset = soft_reset
        self.hard_reset_every = hard_reset_every
        self.scenario_fn = scenario_fn
        self.native_trace_dir = Path(native_trace_dir) if native_trace_dir else None
        if self.native_trace_dir is not None:
            self.native_trace_dir.mkdir(parents=True, exist_ok=True)

        self._perf_lock = threading.Lock()
        self._reset_seconds_accum = 0.0
        self._reset_counter = 0
        self._reset_counter_lock = threading.Lock()
        self._episode_steps = [0] * n_envs
        self._episode_counts = [0] * n_envs
        self._episode_scenario: list[dict] = [{} for _ in range(n_envs)]
        self._episode_seed: list[int | None] = [None] * n_envs

        self._pool = ThreadPoolExecutor(max_workers=n_envs, thread_name_prefix="ogrl-async-env")

        def make(suffix: str, seed: int, worker_level: str) -> OvergrowthEnv:
            return OvergrowthEnv(
                repo_root=repo_root,
                level=worker_level,
                shm_name=f"{shm_prefix}{suffix}",
                controller_id=0,
                seed=seed,
                layout=layout,
                reward_config=reward_config,
                frame_stack=frame_stack,
                act_period=act_period,
                equivalence_digest_path=(self.native_trace_dir / f"{suffix}.jsonl") if self.native_trace_dir else None,
                equivalence_trace_path=(self.native_trace_dir / f"{suffix}.input.jsonl") if self.native_trace_dir else None,
            )

        specs = [(str(i), base_seed + i, self.levels[i]) for i in range(n_envs)]
        self.envs = list(self._pool.map(lambda spec: make(*spec), specs))
        self._current_obs: np.ndarray | None = None
        self._closed = False

    def _next_reset_seed(self, env: OvergrowthEnv) -> int:
        with self._reset_counter_lock:
            self._reset_counter += 1
            return env.seed + self._reset_counter

    def _reset_env(self, index: int, env: OvergrowthEnv) -> tuple[np.ndarray, dict]:
        scenario = self.scenario_fn() if self.scenario_fn is not None else {}
        seed = self._next_reset_seed(env)
        soft = self.soft_reset
        if soft and self.hard_reset_every > 0 and env.episode_count > 0:
            soft = (env.episode_count % self.hard_reset_every) != 0
        obs = env.reset(
            seed=seed,
            soft=soft,
            difficulty=scenario.get("difficulty"),
            opponents=scenario.get("opponents", 1),
            weapons=scenario.get("weapons", 0.0),
            species=scenario.get("species", 0),
        )
        with self._perf_lock:
            self._reset_seconds_accum += env.last_reset_seconds
        scenario_out = dict(scenario)
        scenario_out["soft_reset"] = soft
        self._episode_scenario[index] = scenario_out
        self._episode_seed[index] = env.last_reset_seed
        return obs, scenario_out

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        def reset_one(index: int) -> tuple[np.ndarray, dict]:
            env = self.envs[index]
            if seeds is not None and seeds[index] is not None:
                scenario = self.scenario_fn() if self.scenario_fn is not None else {}
                soft = self.soft_reset
                if soft and self.hard_reset_every > 0 and env.episode_count > 0:
                    soft = (env.episode_count % self.hard_reset_every) != 0
                obs = env.reset(
                    seed=seeds[index],
                    soft=soft,
                    difficulty=scenario.get("difficulty"),
                    opponents=scenario.get("opponents", 1),
                    weapons=scenario.get("weapons", 0.0),
                    species=scenario.get("species", 0),
                )
                scenario = dict(scenario)
                scenario["soft_reset"] = soft
                self._episode_scenario[index] = scenario
                self._episode_seed[index] = env.last_reset_seed
                return obs, scenario
            return self._reset_env(index, env)

        results = list(self._pool.map(reset_one, range(self.n_envs)))
        self._episode_steps = [0] * self.n_envs
        self._current_obs = np.stack([result[0] for result in results]).astype(np.float32, copy=False)
        return self._current_obs.copy()

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        for env in self.envs:
            env.set_reward_config(reward_config)

    def _step_one(self, index: int, action: np.ndarray) -> _StepResult:
        env = self.envs[index]
        obs, reward, done, info = env.step(action)
        self._episode_steps[index] += 1
        won = info["reward_components"]["opponent_knockout"] > 0
        terminal = bool(done or won)
        truncated = (not terminal) and self._episode_steps[index] >= self.max_episode_steps
        terminal_observation = obs

        info = dict(info)
        info["scenario"] = self._episode_scenario[index]
        info["seed"] = self._episode_seed[index]
        info["native_trace_path"] = str(env.equivalence_digest_path) if env.equivalence_digest_path is not None else None

        if terminal or truncated:
            self._episode_steps[index] = 0
            self._episode_counts[index] += 1
            next_obs, _scenario = self._reset_env(index, env)
        else:
            next_obs = obs

        return _StepResult(next_obs, terminal_observation, reward, terminal, truncated, info)

    def collect_rollout(
        self,
        n_steps: int,
        act_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> AsyncRollout:
        """Collect exactly ``n_steps`` transitions per worker.

        ``act_fn(raw_batch)`` returns ``(normalized_obs, actions, log_probs,
        values)``. It is called only between engine futures, so the caller can
        keep policy inference in the main thread and freeze one policy version
        throughout this rollout.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        started = time.monotonic()
        shape = (n_steps, self.n_envs)
        obs = np.empty((n_steps, self.n_envs, self.observation_dim), dtype=np.float32)
        raw_obs = np.empty_like(obs)
        actions = np.empty((n_steps, self.n_envs, ACTION_DIM), dtype=np.float32)
        log_probs = np.empty(shape, dtype=np.float32)
        values = np.empty(shape, dtype=np.float32)
        rewards = np.empty(shape, dtype=np.float32)
        terminals = np.empty(shape, dtype=np.float32)
        infos: list[list[dict]] = [[{} for _ in range(self.n_envs)] for _ in range(n_steps)]
        next_raw = [None] * self.n_envs
        next_index = [0] * self.n_envs
        pending: dict[Future, tuple[int, int]] = {}
        ready_batch_sizes: list[int] = []

        if self._current_obs is None:
            self.reset()
        current_raw = [self._current_obs[i].copy() for i in range(self.n_envs)]

        def schedule(indices: list[int]) -> None:
            if not indices:
                return
            raw_batch = np.stack([current_raw[i] for i in indices]).astype(np.float32, copy=False)
            norm_batch, batch_actions, batch_log_probs, batch_values = act_fn(raw_batch)
            batch_actions = np.asarray(batch_actions, dtype=np.float32).reshape(len(indices), ACTION_DIM)
            batch_log_probs = np.asarray(batch_log_probs, dtype=np.float32).reshape(len(indices))
            batch_values = np.asarray(batch_values, dtype=np.float32).reshape(len(indices))
            ready_batch_sizes.append(len(indices))
            for row, index in enumerate(indices):
                t = next_index[index]
                obs[t, index] = norm_batch[row]
                raw_obs[t, index] = current_raw[index]
                actions[t, index] = batch_actions[row]
                log_probs[t, index] = batch_log_probs[row]
                values[t, index] = batch_values[row]
                future = self._pool.submit(self._step_one, index, batch_actions[row].copy())
                pending[future] = (index, t)

        schedule(list(range(self.n_envs)))
        try:
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                # Let already-finished engine calls join the same policy batch
                # without imposing a meaningful wait on a straggler.
                if len(done) < len(pending):
                    extra, _ = wait(set(pending) - done, timeout=0.0005)
                    done |= extra
                ready: list[int] = []
                for future in done:
                    index, t = pending.pop(future)
                    result = future.result()
                    rewards[t, index] = result.reward
                    terminals[t, index] = float(result.terminal or result.truncated)
                    result.info["terminal_observation"] = result.terminal_observation
                    infos[t][index] = result.info
                    current_raw[index] = result.next_obs
                    next_raw[index] = result.next_obs
                    next_index[index] += 1
                    if next_index[index] < n_steps:
                        ready.append(index)
                schedule(ready)
        finally:
            for future in pending:
                future.cancel()
            for future in pending:
                try:
                    future.result()
                except Exception:
                    pass

        self._current_obs = np.stack(next_raw).astype(np.float32, copy=False)

        return AsyncRollout(
            obs=obs,
            raw_obs=raw_obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            terminals=terminals,
            infos=infos,
            last_raw_obs=np.stack(next_raw).astype(np.float32, copy=False),
            wall_seconds=max(1e-9, time.monotonic() - started),
            batches=len(ready_batch_sizes),
            ready_batch_sizes=ready_batch_sizes,
        )

    def drain_perf(self) -> dict:
        with self._perf_lock:
            result = {"reset_seconds": self._reset_seconds_accum, "pool_hits": 0, "pool_misses": 0}
            self._reset_seconds_accum = 0.0
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        for env in self.envs:
            env.close()
