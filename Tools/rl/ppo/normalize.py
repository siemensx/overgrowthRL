"""Running mean/std normalization for observations and returns.

This is not an optional nicety -- our observation vector mixes world-scale
positions (tens to hundreds of units), one-hot categoricals (0/1), and
pre-normalized ray distances (0/1) in the same vector, with no common scale.
Feeding that directly into a freshly-initialized network is one of the most
common reasons a PPO run fails to learn at all before any algorithmic issue
does; per-dimension running normalization (Welford's online algorithm, the
same approach OpenAI's VecNormalize and most serious PPO baselines use) is
standard practice, not an experimental addition.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class ObservationNormalizer:
    """Normalizes observations with a running mean/std, clipped to
    +/-clip_value after normalization (guards against a rare outlier reading
    -- e.g. a spawn-frame transient -- producing a huge gradient).

    OGRL-20260817-028 Sec5: layout/frame_stack-aware, not a flat obs_dim-length
    vector. Non-entity floats (proprioception+action-history+rays, per stacked
    frame) keep the original per-flat-position running stats. Entity floats
    use ONE shared per-field RunningMeanStd across every entity slot and every
    stacked frame, updated ONLY from valid (real, not zero-padded) entity
    rows -- the fix for the dead-dimension pathology at its root (measured
    directly on run9: 178/260 floats with variance <1e-6 across the whole
    run, entity slots 1-7 exactly zero for a 1-opponent scenario, so a flat
    per-position normalizer for slot 3 never sees real data whenever fewer
    than 4 entities are visible, then saturates its clip the instant a
    different entity count shows up). A shared per-field normalizer instead
    gets fed from whichever slot IS valid every step, regardless of which
    physical slot an opponent happens to land in."""

    def __init__(self, layout, frame_stack: int = 1, clip_value: float = 10.0):
        self.layout = layout
        self.frame_stack = max(1, frame_stack)
        self.clip_value = clip_value
        self.entity_floats = layout.entity_slice(0).stop - layout.entity_slice(0).start
        self.n_entities = layout.max_visible_entities
        self.entities_start = layout.entities_start
        self.entities_region = self.n_entities * self.entity_floats
        self.frame_floats = layout.total_floats
        self.non_entity_per_frame = self.frame_floats - self.entities_region
        self.obs_dim = self.frame_floats * self.frame_stack  # kept for logging/back-compat only

        self.non_entity_rms = RunningMeanStd((self.non_entity_per_frame * self.frame_stack,))
        self.entity_rms = RunningMeanStd((self.entity_floats,))

    def _split(self, obs: np.ndarray):
        batch = obs.shape[0]
        frames = obs.reshape(batch, self.frame_stack, self.frame_floats)
        entities = frames[:, :, self.entities_start:self.entities_start + self.entities_region]
        entities = entities.reshape(batch, self.frame_stack, self.n_entities, self.entity_floats)
        non_entity = np.concatenate(
            [frames[:, :, :self.entities_start], frames[:, :, self.entities_start + self.entities_region:]],
            axis=-1,
        ).reshape(batch, -1)
        return non_entity, entities

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        obs = obs.reshape(-1, obs.shape[-1]) if obs.ndim > 1 else obs.reshape(1, -1)
        batch = obs.shape[0]
        non_entity, entities = self._split(obs)
        valid = entities[..., 0] > 0.5  # (batch, frame_stack, n_entities) -- RLObservation's `valid` field is field 0

        if update:
            self.non_entity_rms.update(non_entity)
            valid_rows = entities[valid]  # (n_valid_rows, entity_floats)
            if valid_rows.shape[0] > 0:
                self.entity_rms.update(valid_rows)

        non_entity_norm = np.clip((non_entity - self.non_entity_rms.mean) / self.non_entity_rms.std, -self.clip_value, self.clip_value)
        non_entity_norm = non_entity_norm.reshape(batch, self.frame_stack, self.non_entity_per_frame)
        entity_norm = np.clip((entities - self.entity_rms.mean) / self.entity_rms.std, -self.clip_value, self.clip_value)
        # Invalid (padding) slots: zero out AFTER normalization rather than
        # leaving them at whatever (0-mean)/std happens to be -- a padding
        # slot should read as "nothing here" to the entity encoder, which
        # also masks by the raw `valid` field, so this is redundant-but-safe
        # belt-and-suspenders, not load-bearing on its own.
        entity_norm = entity_norm * valid[..., None]

        frames_norm = np.empty((batch, self.frame_stack, self.frame_floats), dtype=np.float32)
        frames_norm[:, :, :self.entities_start] = non_entity_norm[:, :, :self.entities_start]
        frames_norm[:, :, self.entities_start:self.entities_start + self.entities_region] = entity_norm.reshape(batch, self.frame_stack, self.entities_region)
        frames_norm[:, :, self.entities_start + self.entities_region:] = non_entity_norm[:, :, self.entities_start:]
        return frames_norm.reshape(batch, -1)

    def state_dict(self) -> dict:
        return {
            "non_entity_mean": self.non_entity_rms.mean, "non_entity_var": self.non_entity_rms.var, "non_entity_count": self.non_entity_rms.count,
            "entity_mean": self.entity_rms.mean, "entity_var": self.entity_rms.var, "entity_count": self.entity_rms.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.non_entity_rms.mean = state["non_entity_mean"]
        self.non_entity_rms.var = state["non_entity_var"]
        self.non_entity_rms.count = state["non_entity_count"]
        self.entity_rms.mean = state["entity_mean"]
        self.entity_rms.var = state["entity_var"]
        self.entity_rms.count = state["entity_count"]


class RewardNormalizer:
    """Scales rewards by a running estimate of the discounted-return std
    (not raw reward std) -- the standard approach (Engstrom et al. 2020's
    PPO-details study, and OpenAI's VecNormalize) since it's the return scale,
    not the per-step reward scale, that actually determines value-function
    learning difficulty. Does NOT subtract the mean -- centering rewards
    changes the sign of sparse terminal rewards and is deliberately avoided."""

    def __init__(self, gamma: float, clip_value: float = 10.0, n_envs: int = 1):
        # n_envs independent running returns -- each parallel worker
        # (vec_env.py) has its own episode trajectory, and folding them into
        # one shared accumulator would mix unrelated episodes' discounted
        # histories together. n_envs=1 (the original, still-used-by-train.py
        # shape) is just the n_envs=n case with n=1 and a squeezed I/O shape,
        # not a separate code path.
        self.rms = RunningMeanStd((1,))
        self.gamma = gamma
        self.clip_value = clip_value
        self.n_envs = n_envs
        self._running_return = np.zeros(n_envs, dtype=np.float64)

    def normalize(self, reward, done):
        """reward/done: scalars if n_envs==1 (matches the original scalar
        API train.py uses), else array-likes of length n_envs. Returns the
        same shape it was given."""
        scalar_input = self.n_envs == 1 and np.isscalar(reward)
        reward_arr = np.atleast_1d(np.asarray(reward, dtype=np.float64))
        done_arr = np.atleast_1d(np.asarray(done, dtype=bool))

        # Standard order (matches OpenAI's VecNormalize): accumulate this
        # step's contribution into the running discounted return FIRST --
        # for a terminal step, that return legitimately includes this step's
        # reward on top of the episode's prior history -- update statistics
        # from that, THEN reset the accumulator so the *next* episode starts
        # from zero. Zeroing before accumulating (the earlier, incorrect
        # version of this) would have discarded the terminal reward's
        # contribution to the return estimate on the very step it matters most.
        self._running_return = self._running_return * self.gamma + reward_arr
        self.rms.update(self._running_return.reshape(-1, 1))
        scaled = reward_arr / self.rms.std[0]
        self._running_return = np.where(done_arr, 0.0, self._running_return)

        clipped = np.clip(scaled, -self.clip_value, self.clip_value)
        return float(clipped[0]) if scalar_input else clipped.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.rms.mean, "var": self.rms.var, "count": self.rms.count, "running_return": self._running_return}

    def load_state_dict(self, state: dict) -> None:
        # OGRL-20260816-018: found live, not by inspection -- a --resume-from
        # smoke test at n_envs=4 against a checkpoint saved at n_envs=8 blew
        # up 3 steps into training with a raw numpy broadcast error
        # ((8,) vs (4,)) instead of a clear message at the actual point of
        # the mistake. running_return is one independent accumulator PER
        # WORKER (see __init__) -- it's meaningless to graft an 8-worker
        # checkpoint's per-worker state onto a differently-sized run.
        saved_n_envs = len(state["running_return"])
        if saved_n_envs != self.n_envs:
            raise ValueError(
                f"cannot resume: this checkpoint's reward_normalizer was saved with n_envs={saved_n_envs}, "
                f"but this run was started with --n-envs {self.n_envs} -- running_return is one running "
                f"accumulator per worker, so a resume must use the same worker count the checkpoint was saved with."
            )
        self.rms.mean = state["mean"]
        self.rms.var = state["var"]
        self.rms.count = state["count"]
        self._running_return = state["running_return"]
