"""Hybrid actor-critic for the Overgrowth action space: 2 continuous dims
(move_x, move_y, both genuinely bounded to [-1,1]) + 6 independent binary
dims (jump, crouch, attack, grab, drop, walk). This is a parameterized/hybrid
action space in the RL literature (Hausknecht & Stone 2016), and the
established way to handle it under PPO -- used at scale in e.g. OpenAI Five's
and AlphaStar's factorized policies -- is exactly this: one shared-or-separate
trunk producing independent distributions per action group, combined into a
single joint log-probability (sum of each group's log-prob) for one PPO
objective, rather than a separate learner per action type.

Continuous head is a **tanh-squashed Gaussian**, not a plain clipped Gaussian
-- borrowed from SAC's treatment of bounded continuous actions rather than
the plainer "unbounded Gaussian + clip" some PPO baselines use, because the
squash keeps a real (corrected) gradient signal at the action boundary
instead of the flat, uninformative gradient a hard clip produces there,
which matters here since "hold the stick fully forward" is a completely
ordinary action, not an edge case. Entropy for this head uses the
untransformed Gaussian's entropy as an approximation (the tanh transform
has no simple closed form) -- standard practice in tanh-Gaussian SAC/PPO
implementations, not a shortcut specific to this project.

Orthogonal initialization with the standard PPO gains (Engstrom et al. 2020 /
the widely-cited "37 implementation details of PPO" study): hidden layers
gain=sqrt(2), policy output layers gain=0.01 (start near-uniform, not
overconfident), value output gain=1.0, all biases zero.

--- Entity-set encoder (OGRL-20260817-028 Sec5) ---

Before this, the actor/critic trunks took a flat obs_dim-length vector --
the 8 fixed entity slots (nearest-first) were just concatenated in with
everything else. Measured directly on run9 (research-log 2026-08-17): 178 of
260 floats had running variance < 1e-6 across the ENTIRE run; entity slots
1-7 sat at 3.3e-11 (identically zero the whole time, since the training
scenario only ever has one opponent). ObservationNormalizer divides by that
near-zero std and clips at +/-10 -- so the instant a second opponent becomes
visible, 168 inputs snap from exactly 0.0 to a saturated +/-10, an input
regime the network has never seen. Flat concatenation is also permutation-
sensitive (the same world state produces different input vectors as slot
order changes) and shares no parameters across entities (whatever the
network learns about "an opponent in slot 0" doesn't transfer to slot 1).

The fix, used at this scale by both OpenAI Five and AlphaStar: a shared
per-entity MLP (EntityEncoder below) applied independently to each entity
slot, masked so invalid (padding) slots don't contribute, then pooled
(masked max-pool here -- Sec5 names this the baseline, attention keyed on
the proprioception embedding as "the better version," left as a documented
future upgrade rather than built tonight) into one fixed-size summary. That
summary is permutation-invariant and count-agnostic: the policy can train at
1 opponent and evaluate at 3 without any input-layout change, which is the
actual precondition for OGRL-20260817-028's later multi-opponent stages, not
just a variance-normalization fix.

Frame-stacking interacts with this by applying the SAME entity encoder
(shared weights) independently to each stacked frame, then concatenating the
per-frame pooled entity embeddings -- preserving frame_stack's existing
temporal-signal role (a plain MLP core, no recurrence yet) without changing
what any single frame's entity encoding means.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli

# --- distribution math, written out ---------------------------------------
# torch.distributions objects cost real time at this scale: the policy is tiny
# (256 hidden) and the rollout batch is n_envs, so per-call object construction
# is not amortised by any matmul. The functions below are the exact formulas
# torch uses, applied directly to the raw parameters.
#
# Measured 2026-09-06 (batch=4, 1 thread, 3 interleaved rounds of 4000 calls):
#   rollout act():  376 us -> 326 us   (1.15x)
#   update path, batch=128 with autograd: 4351 us -> 4264 us  (1.02x)
#
# So this is worth ~13% of the rollout action call and almost nothing on the
# update -- real, free, and much smaller than a first, WRONG reading of a
# _features-vs-full-call comparison suggested (that gap is mostly the actor and
# critic trunks, which are genuine math, not plumbing).
#
# `test_policy_fast_path.py` asserts these agree with torch.distributions to
# float precision, including the gradient-carrying update path. That is why
# Normal and Bernoulli are still imported.
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _normal_log_prob(value: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((value - mean) / log_std.exp()) ** 2) - log_std - _HALF_LOG_2PI


def _normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return 0.5 + _HALF_LOG_2PI + log_std


def _bernoulli_log_prob(value: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, torch.sigmoid(logits), reduction="none")

CONTINUOUS_DIM = 2
DISCRETE_DIM = 6
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
# Tightened from 1e-6 to 1e-3 after a 2M-step run (research-log OGRL-20260816-013)
# showed recurring, increasingly frequent approx_kl spikes (up to 1.89, ~95x
# the 0.02 target) with no counterpart cause in the reward/entropy trends --
# consistent with a numerical, not a real policy-shift, origin: atanh(x)
# diverges as x -> +/-1, so recovering raw_continuous for an action sampled
# very close to the tanh boundary produces a large, highly sensitive value --
# a small difference between the rollout-time and update-time policy's
# implied position at that boundary then produces a wildly out-of-proportion
# importance ratio for just that one sample, which can dominate a whole
# minibatch's mean approx_kl despite gradient clipping keeping the actual
# parameter step small. At eps=1e-6, atanh(1-eps) ~= 7.25; at eps=1e-3,
# atanh(1-eps) ~= 3.8 -- a ~2x tighter bound on how large any single sample's
# recovered raw value (and hence its log-prob swing) can get, while still
# leaving the clamp far enough from +/-1 to not meaningfully bias sampled
# actions that aren't already saturated.
_TANH_EPS = 1e-3


def _layer_init(layer: nn.Linear, gain: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, output_gain: float) -> nn.Sequential:
    return nn.Sequential(
        _layer_init(nn.Linear(input_dim, hidden_dim)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden_dim, output_dim), gain=output_gain),
    )


class EntityEncoder(nn.Module):
    """Shared per-entity MLP + masked max-pool. See module docstring."""

    def __init__(self, entity_floats: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            _layer_init(nn.Linear(entity_floats, embed_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.Tanh(),
        )

    def forward(self, entities: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """entities: (..., n_entities, entity_floats). valid_mask: (..., n_entities),
        1.0/True for a real entity, 0.0/False for a zero-filled padding slot
        (matches RLObservation's own `valid` field exactly -- see
        obs_schema.py's ENTITY_FLOATS layout, valid is field 0). Returns
        (..., embed_dim)."""
        embedded = self.mlp(entities)  # (..., n_entities, embed_dim)
        mask = valid_mask.bool().unsqueeze(-1)
        neg_fill = torch.finfo(embedded.dtype).min
        masked = embedded.masked_fill(~mask, neg_fill)
        pooled, _ = masked.max(dim=-2)
        # All-invalid entity set (shouldn't happen in practice -- self is
        # never an entity, but a step with genuinely zero visible entities is
        # not impossible) would leave pooled at neg_fill; fall back to zero
        # rather than feeding a huge negative constant into the trunk.
        any_valid = mask.any(dim=-2)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return pooled


class ActorCritic(nn.Module):
    def __init__(self, layout, frame_stack: int = 1, hidden_dim: int = 256, entity_embed_dim: int = 64):
        """layout: obs_schema.ObsLayout (or any object exposing the same
        entities_start/max_visible_entities/total_floats contract). Replaces
        the old flat obs_dim constructor arg (Sec5's checkpoint-invalidating
        change) -- the network needs to know where the entity region lives
        within each stacked frame, not just how many floats there are in
        total."""
        super().__init__()
        self.layout = layout
        self.frame_stack = max(1, frame_stack)
        self.entity_floats = layout.entity_slice(0).stop - layout.entity_slice(0).start
        self.n_entities = layout.max_visible_entities
        self.entities_start = layout.entities_start
        self.entities_region = self.n_entities * self.entity_floats
        self.frame_floats = layout.total_floats
        self.non_entity_per_frame = self.frame_floats - self.entities_region
        self.obs_dim = self.frame_floats * self.frame_stack  # kept for logging/back-compat only

        self.entity_encoder = EntityEncoder(self.entity_floats, entity_embed_dim)
        self.proprioception_branch = nn.Sequential(
            _layer_init(nn.Linear(self.non_entity_per_frame * self.frame_stack, hidden_dim)),
            nn.Tanh(),
        )
        trunk_input_dim = hidden_dim + entity_embed_dim * self.frame_stack

        # Separate actor/critic trunks (not shared) -- the more common choice
        # in continuous-control PPO baselines (the original PPO paper's
        # MuJoCo experiments, most spinning-up-style implementations), which
        # avoids the actor and critic losses fighting over shared features --
        # a real risk here given how different "what should I do" and "how
        # good is this state" are for a combat task. The entity encoder
        # itself IS shared between actor and critic (one set of weights) --
        # both need the same "what's out there" summary, and splitting it
        # would double the entity-encoder parameter count for no known benefit.
        self.actor_trunk = nn.Sequential(
            _layer_init(nn.Linear(trunk_input_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.continuous_mean = _layer_init(nn.Linear(hidden_dim, CONTINUOUS_DIM), gain=0.01)
        self.continuous_log_std = nn.Parameter(torch.zeros(CONTINUOUS_DIM))  # state-independent, standard PPO practice
        self.discrete_logits = _layer_init(nn.Linear(hidden_dim, DISCRETE_DIM), gain=0.01)

        self.critic_trunk = nn.Sequential(
            _layer_init(nn.Linear(trunk_input_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.critic_out = _layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, frame_stack * frame_floats), the exact flat layout
        env.py's frame stacking already produces (oldest frame first). Splits
        each frame into its entity region and everything else, runs the
        shared entity encoder per frame, and concatenates
        [proprioception_branch_out, entity_embed_frame_0, ..., entity_embed_frame_{K-1}]."""
        batch = obs.shape[0]
        frames = obs.view(batch, self.frame_stack, self.frame_floats)

        entities = frames[:, :, self.entities_start:self.entities_start + self.entities_region]
        entities = entities.reshape(batch, self.frame_stack, self.n_entities, self.entity_floats)
        valid_mask = entities[..., 0]

        non_entity = torch.cat(
            [frames[:, :, :self.entities_start], frames[:, :, self.entities_start + self.entities_region:]],
            dim=-1,
        )  # (batch, frame_stack, non_entity_per_frame)

        entity_embed = self.entity_encoder(entities, valid_mask)  # (batch, frame_stack, entity_embed_dim)
        entity_embed_flat = entity_embed.reshape(batch, -1)
        prop_flat = non_entity.reshape(batch, -1)
        prop_features = self.proprioception_branch(prop_flat)
        return torch.cat([prop_features, entity_embed_flat], dim=-1)

    def _actor_params(self, features: torch.Tensor):
        """Raw actor parameters: (continuous mean, clamped log_std, discrete logits).

        The fast path below works from these directly instead of wrapping them
        in Normal/Bernoulli -- see the module-level note on why."""
        features = self.actor_trunk(features)
        mean = self.continuous_mean(features)
        log_std = torch.clamp(self.continuous_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, self.discrete_logits(features)

    def _distributions_from_features(self, features: torch.Tensor):
        """Build the actor distributions from the shared trunk input.

        ``get_action_and_value`` evaluates the actor and critic for the same
        observation during both rollout collection and PPO updates. Keep the
        expensive entity/proprioception encoder outside this helper so that
        both heads consume one feature pass.
        """
        features = self.actor_trunk(features)
        mean = self.continuous_mean(features)
        log_std = torch.clamp(self.continuous_log_std, LOG_STD_MIN, LOG_STD_MAX)
        continuous_dist = Normal(mean, log_std.exp())
        discrete_dist = Bernoulli(logits=self.discrete_logits(features))
        return continuous_dist, discrete_dist

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.critic_trunk(self._features(obs))
        return self.critic_out(features).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """action, if given, is the FULL 8-dim env action already taken
        (tanh-squashed continuous dims + 0/1 discrete dims) -- used during
        the PPO update pass to re-evaluate log-prob/entropy under the
        current (updated) policy for a batch of previously-collected
        transitions. If None, samples a fresh action (rollout collection).
        Returns (action[8], log_prob[scalar], entropy[scalar], value[scalar])."""
        # The entity encoder and proprioception branch are shared by the
        # actor and critic. The previous implementation called _features()
        # once through the actor and again through get_value(), duplicating
        # the largest part of this model's forward pass. Compute it once and
        # let the two independent trunks specialize from the same features.
        shared_features = self._features(obs)
        mean, log_std, discrete_logits = self._actor_params(shared_features)

        if action is None:
            # Normal.rsample() is loc + eps*scale with eps ~ N(0,1); Bernoulli
            # .sample() is torch.bernoulli(probs). Same draws, no objects.
            raw_continuous = mean + torch.randn_like(mean) * log_std.exp()
            continuous_action = torch.tanh(raw_continuous)
            discrete_action = torch.bernoulli(torch.sigmoid(discrete_logits))
        else:
            continuous_action = action[..., :CONTINUOUS_DIM].clamp(-1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
            raw_continuous = torch.atanh(continuous_action)
            discrete_action = action[..., CONTINUOUS_DIM:]

        # Tanh-squash log-prob correction (SAC, Haarnoja et al. 2018 appendix C):
        # log pi(a) = log N(u) - sum(log(1 - tanh(u)^2 + eps)), u = atanh(a).
        continuous_log_prob = _normal_log_prob(raw_continuous, mean, log_std) - torch.log(1.0 - continuous_action.pow(2) + _TANH_EPS)
        continuous_log_prob = continuous_log_prob.sum(dim=-1)
        discrete_log_prob = _bernoulli_log_prob(discrete_action, discrete_logits).sum(dim=-1)
        log_prob = continuous_log_prob + discrete_log_prob

        # Entropy: exact for the discrete heads; the untransformed Gaussian's
        # entropy is used as an approximation for the continuous heads (see
        # module docstring -- the tanh transform has no simple closed form).
        entropy = _normal_entropy(log_std).sum(dim=-1).expand(discrete_logits.shape[:-1]) + _bernoulli_entropy(discrete_logits).sum(dim=-1)

        joint_action = torch.cat([continuous_action, discrete_action], dim=-1)
        value = self.critic_out(self.critic_trunk(shared_features)).squeeze(-1)
        return joint_action, log_prob, entropy, value
