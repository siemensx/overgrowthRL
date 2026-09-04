"""Emergence panel (OGRL-20260817-028 Sec8.3/Sec3.4): turns "have any
strategies emerged" from a judgement call into a chart. Each signature is
computed as (conditional probability - unconditional probability) so zero
means "no relationship" and a rising line means a behavior is being learned.

Computed fresh each PPO update from that update's own collected steps (same
sample size/denominator as action_stats) rather than accumulated forever --
consistent with how every other per-update stat in this system is already a
noisy time series the dashboard charts, not a single running number.

Five signatures, each needing only the observation + the action actually
taken that step -- no engine change, no new logging path:
  - P(grab | opponent attacking) - P(grab)                    -> parry
  - P(attack | opponent in hit_reaction) - P(attack)           -> punish
  - P(crouch | self ragdolled) - P(crouch)                     -> roll_recovery
  - P(attack | opp block broken) - P(attack | opp block healthy) -> guard_pressure
  - mean hostiles within 3m, steps where >=2 hostiles visible  -> funnelling
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATTACK_STATE_INDEX = 2       # STATE one-hot: movement, ground, attack, hit_reaction, ragdoll
HIT_REACTION_STATE_INDEX = 3
RAGDOLL_STATE_INDEX = 4
BLOCK_BROKEN_THRESHOLD = 0.1  # block_health below this = "broken"
BLOCK_HEALTHY_THRESHOLD = 0.5  # block_health at/above this = "healthy" -- the band between is neither, excluded
FUNNEL_RADIUS = 3.0  # meters


@dataclass
class EmergenceAccumulator:
    n_total: int = 0
    n_grab: int = 0
    n_opp_attacking: int = 0
    n_grab_and_opp_attacking: int = 0
    n_attack: int = 0
    n_opp_hitreaction: int = 0
    n_attack_and_opp_hitreaction: int = 0
    n_crouch: int = 0
    n_self_ragdoll: int = 0
    n_crouch_and_self_ragdoll: int = 0
    n_opp_block_broken: int = 0
    n_attack_and_opp_block_broken: int = 0
    n_opp_block_healthy: int = 0
    n_attack_and_opp_block_healthy: int = 0
    n_funnel_eligible: int = 0  # steps with >=2 hostiles visible
    sum_hostiles_within_radius: int = 0

    def update(self, frame: list, entities: list, action, layout) -> None:
        """frame: one raw (unnormalized) observation, this step's PRE-action
        state. entities: layout.all_entities(frame). action: the 8-float
        action actually taken this step (index order: move_x, move_y, jump,
        crouch, attack, grab, drop, walk)."""
        self.n_total += 1
        grab = bool(action[5] > 0.5)
        attack_action = bool(action[4] > 0.5)
        crouch = bool(action[3] > 0.5)

        valid = [e for e in entities if e["valid"]]
        hostiles = [e for e in valid if not e["is_ally"]]
        nearest_hostile = min(hostiles, key=lambda e: e["distance"]) if hostiles else None

        if grab:
            self.n_grab += 1
        if nearest_hostile is not None:
            opp_state = nearest_hostile["state"]
            opp_attacking = opp_state[ATTACK_STATE_INDEX] > 0.5
            opp_hitreaction = opp_state[HIT_REACTION_STATE_INDEX] > 0.5
            if opp_attacking:
                self.n_opp_attacking += 1
                if grab:
                    self.n_grab_and_opp_attacking += 1
            if opp_hitreaction:
                self.n_opp_hitreaction += 1
                if attack_action:
                    self.n_attack_and_opp_hitreaction += 1
            block_hp = nearest_hostile["block_health"]
            if block_hp < BLOCK_BROKEN_THRESHOLD:
                self.n_opp_block_broken += 1
                if attack_action:
                    self.n_attack_and_opp_block_broken += 1
            elif block_hp >= BLOCK_HEALTHY_THRESHOLD:
                self.n_opp_block_healthy += 1
                if attack_action:
                    self.n_attack_and_opp_block_healthy += 1

        if attack_action:
            self.n_attack += 1
        if crouch:
            self.n_crouch += 1
        self_ragdoll = frame[layout.STATE][RAGDOLL_STATE_INDEX] > 0.5
        if self_ragdoll:
            self.n_self_ragdoll += 1
            if crouch:
                self.n_crouch_and_self_ragdoll += 1

        if len(hostiles) >= 2:
            self.n_funnel_eligible += 1
            self.sum_hostiles_within_radius += sum(1 for e in hostiles if e["distance"] <= FUNNEL_RADIUS)

    def update_batch(self, frames: np.ndarray, actions: np.ndarray, layout) -> None:
        """Vectorized equivalent of :meth:`update` for one rollout batch.

        The old hot loop unpacked every entity slot into Python dictionaries
        even when tape recording was disabled. This method keeps the same
        counters and thresholds while doing the fixed-schema reductions in
        NumPy. It deliberately accepts the layout so a schema change cannot
        silently move the entity region.
        """
        frames = np.asarray(frames, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if frames.ndim != 2 or actions.ndim != 2 or frames.shape[0] != actions.shape[0]:
            raise ValueError("frames must be (batch, floats) and actions must share its batch dimension")
        count = frames.shape[0]
        if count == 0:
            return

        entity_start = layout.entities_start
        entity_region = layout.rays_start - entity_start
        entity_floats = entity_region // layout.max_visible_entities
        entities = frames[:, entity_start:layout.rays_start].reshape(count, layout.max_visible_entities, entity_floats)
        valid = entities[:, :, 0] > 0.5
        hostiles = valid & (entities[:, :, 23] <= 0.5)
        distances = np.where(hostiles, entities[:, :, 8], np.inf)
        nearest = np.argmin(distances, axis=1)
        has_nearest = np.any(hostiles, axis=1)
        rows = np.arange(count)
        nearest_state = entities[rows, nearest, 13:18]
        nearest_block = entities[rows, nearest, 27]

        grab = actions[:, 5] > 0.5
        attack = actions[:, 4] > 0.5
        crouch = actions[:, 3] > 0.5
        opp_attacking = has_nearest & (nearest_state[:, ATTACK_STATE_INDEX] > 0.5)
        opp_hitreaction = has_nearest & (nearest_state[:, HIT_REACTION_STATE_INDEX] > 0.5)
        opp_block_broken = has_nearest & (nearest_block < BLOCK_BROKEN_THRESHOLD)
        opp_block_healthy = has_nearest & (nearest_block >= BLOCK_HEALTHY_THRESHOLD)
        self_ragdoll = frames[:, layout.STATE.start + RAGDOLL_STATE_INDEX] > 0.5
        funnel_eligible = hostiles.sum(axis=1) >= 2
        hostiles_within = (hostiles & (entities[:, :, 8] <= FUNNEL_RADIUS)).sum(axis=1)

        self.n_total += count
        self.n_grab += int(grab.sum())
        self.n_opp_attacking += int(opp_attacking.sum())
        self.n_grab_and_opp_attacking += int((grab & opp_attacking).sum())
        self.n_attack += int(attack.sum())
        self.n_opp_hitreaction += int(opp_hitreaction.sum())
        self.n_attack_and_opp_hitreaction += int((attack & opp_hitreaction).sum())
        self.n_crouch += int(crouch.sum())
        self.n_self_ragdoll += int(self_ragdoll.sum())
        self.n_crouch_and_self_ragdoll += int((crouch & self_ragdoll).sum())
        self.n_opp_block_broken += int(opp_block_broken.sum())
        self.n_attack_and_opp_block_broken += int((attack & opp_block_broken).sum())
        self.n_opp_block_healthy += int(opp_block_healthy.sum())
        self.n_attack_and_opp_block_healthy += int((attack & opp_block_healthy).sum())
        self.n_funnel_eligible += int(funnel_eligible.sum())
        self.sum_hostiles_within_radius += int(hostiles_within[funnel_eligible].sum())

    def snapshot(self) -> dict:
        def _rate(num, den):
            return (num / den) if den > 0 else None

        p_grab = _rate(self.n_grab, self.n_total)
        p_grab_given_atk = _rate(self.n_grab_and_opp_attacking, self.n_opp_attacking)
        p_attack = _rate(self.n_attack, self.n_total)
        p_attack_given_hitreaction = _rate(self.n_attack_and_opp_hitreaction, self.n_opp_hitreaction)
        p_crouch = _rate(self.n_crouch, self.n_total)
        p_crouch_given_ragdoll = _rate(self.n_crouch_and_self_ragdoll, self.n_self_ragdoll)
        p_attack_given_broken = _rate(self.n_attack_and_opp_block_broken, self.n_opp_block_broken)
        p_attack_given_healthy = _rate(self.n_attack_and_opp_block_healthy, self.n_opp_block_healthy)

        parry = (p_grab_given_atk - p_grab) if (p_grab_given_atk is not None and p_grab is not None) else None
        punish = (p_attack_given_hitreaction - p_attack) if (p_attack_given_hitreaction is not None and p_attack is not None) else None
        roll_recovery = (p_crouch_given_ragdoll - p_crouch) if (p_crouch_given_ragdoll is not None and p_crouch is not None) else None
        guard_pressure = (p_attack_given_broken - p_attack_given_healthy) if (p_attack_given_broken is not None and p_attack_given_healthy is not None) else None
        funnelling = _rate(self.sum_hostiles_within_radius, self.n_funnel_eligible)

        return {
            "parry": parry, "punish": punish, "roll_recovery": roll_recovery,
            "guard_pressure": guard_pressure, "funnelling": funnelling,
            "samples": {
                "opp_attacking": self.n_opp_attacking, "opp_hitreaction": self.n_opp_hitreaction,
                "self_ragdoll": self.n_self_ragdoll, "opp_block_broken": self.n_opp_block_broken,
                "opp_block_healthy": self.n_opp_block_healthy, "funnel_eligible": self.n_funnel_eligible,
            },
        }
