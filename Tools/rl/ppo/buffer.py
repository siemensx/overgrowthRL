"""Rollout storage + Generalized Advantage Estimation (Schulman et al. 2016).

Deliberately distinguishes true termination from time-limit truncation
(Pardo et al., "Time Limits in Reinforcement Learning", 2018): a training
loop that forces episode resets at a step cap (this one does, via
max_episode_steps in train.py) must bootstrap the value function at a
truncated episode's last step using the *value estimate*, not zero, or the
agent is implicitly taught that running out of time is exactly as bad as
actually losing -- a well-documented, easy-to-miss correctness bug in PPO
implementations that don't handle it. `terminal` (passed into add()) must
therefore mean "this episode genuinely ended" (self knocked out, or the
Python-side win condition on opponent knockout -- see train.py), never
"the loop decided to stop collecting here."
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.terminals = np.zeros(n_steps, dtype=np.float32)  # true termination only, see module docstring
        self.ptr = 0

    def add(self, obs, action, log_prob, value, reward, terminal: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.terminals[i] = float(terminal)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.terminals[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def to_tensors(self, last_value: float, gamma: float, gae_lambda: float) -> dict:
        advantages, returns = self.compute_gae(last_value, gamma, gae_lambda)
        return {
            "obs": torch.as_tensor(self.obs, device=self.device),
            "actions": torch.as_tensor(self.actions, device=self.device),
            "log_probs": torch.as_tensor(self.log_probs, device=self.device),
            "values": torch.as_tensor(self.values, device=self.device),
            "advantages": torch.as_tensor(advantages, device=self.device),
            "returns": torch.as_tensor(returns, device=self.device),
        }
