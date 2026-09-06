"""Vectorized rollout storage + GAE for N parallel workers (vec_env.py).

Same correctness requirements as buffer.py's RolloutBuffer (true-termination
vs. truncation must already be folded into the `terminal` flag by the caller
before add() -- see buffer.py's docstring, unchanged here), generalized from
(n_steps,) to (n_steps, n_envs) storage. GAE's recursion is still sequential
over the time dimension (advantages at step t depend on step t+1), but fully
vectorized across the n_envs dimension within each time step -- N independent
trajectories, not N independent buffers, so one array op per time step
instead of N.
"""

from __future__ import annotations

import numpy as np
import torch


class VecRolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.terminals = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, value, reward, terminal) -> None:
        """All arguments except obs/action are (n_envs,)-shaped; obs is
        (n_envs, obs_dim), action is (n_envs, action_dim)."""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.terminals[i] = terminal
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_value = last_values if t == self.n_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.terminals[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def merged_with(self, remotes: list, device) -> "VecRolloutBuffer":
        """A new buffer whose env columns are this buffer's followed by each
        remote rollout's, concatenated along the ENV axis.

        Exact, not an approximation: compute_gae runs its recursion over time
        INDEPENDENTLY per env column, so appending columns adds trajectories
        without any term crossing between them. No trajectory is split across
        machines. Requires identical n_steps and obs/action dims, which the
        learner enforces when a worker registers.
        """
        if not remotes:
            return self
        for r in remotes:
            if r["obs"].shape[0] != self.n_steps:
                raise ValueError(f"remote rollout has {r['obs'].shape[0]} steps, expected {self.n_steps}")
        extra = sum(r["obs"].shape[1] for r in remotes)
        out = VecRolloutBuffer(self.n_steps, self.n_envs + extra,
                               self.obs.shape[2], self.actions.shape[2], device)
        def cat(name, mine):
            return np.concatenate([mine] + [r[name] for r in remotes], axis=1)
        out.obs = cat("obs", self.obs)
        out.actions = cat("actions", self.actions)
        out.log_probs = cat("log_probs", self.log_probs)
        out.values = cat("values", self.values)
        out.rewards = cat("rewards", self.rewards)
        out.terminals = cat("dones", self.terminals)
        out.ptr = self.n_steps
        return out

    def to_tensors(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> dict:
        """Flattens the (n_steps, n_envs, ...) storage into (n_steps*n_envs, ...)
        for minibatch sampling -- PPO doesn't care which worker a transition
        came from, only that transitions aren't reused across the boundary
        GAE was computed over."""
        advantages, returns = self.compute_gae(last_values, gamma, gae_lambda)

        def flatten(x):
            return x.reshape(self.n_steps * self.n_envs, *x.shape[2:])

        return {
            "obs": torch.as_tensor(flatten(self.obs), device=self.device),
            "actions": torch.as_tensor(flatten(self.actions), device=self.device),
            "log_probs": torch.as_tensor(flatten(self.log_probs), device=self.device),
            "values": torch.as_tensor(flatten(self.values), device=self.device),
            "advantages": torch.as_tensor(flatten(advantages), device=self.device),
            "returns": torch.as_tensor(flatten(returns), device=self.device),
        }
