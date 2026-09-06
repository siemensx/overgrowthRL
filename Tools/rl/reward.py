"""Reward function for the Overgrowth RL environment.

Not part of the plan's 8 stages (see research-log's ogrl-plan-scope memory --
the plan covers environment/transport mechanics through Stage 8's
characterization report, but never scopes reward/curriculum) and not part of
RLObservation's transport contract either -- Source/Main/rl_shm_transport.cpp
deliberately does not carry a reward field; only episode_done is native. This
keeps reward shaping entirely in Python, where it can be iterated without an
engine rebuild, and keeps the environment contract (what the character can
observe/do) separate from the training-time judgment of what's "good" --
those are genuinely different kinds of decisions with different rates of
change, and coupling them would slow down both.

Composed of components a fighting-game agent needs to learn combat, built
from what the observation (schema v2, obs_schema.py) can actually measure:
  - dense self-damage-taken penalty (temp_health + blood_health deltas)
  - dense opponent-damage-dealt bonus (same two pools, on visible entities,
    matched by entity id across steps since slot order is nearest-first and
    reorders freely as the fight moves)
  - sparse opponent-knockout bonus and self-knockout terminal penalty
  - a small per-step time cost, to discourage a degenerate "do nothing and
    survive" policy once damage/knockout rewards are in the mix
  - an optional, curriculum-gated closing-distance shaping term (bootstraps
    engagement against a policy that starts out doing nothing) -- meant to be
    weighted down/removed in later curriculum phases, since left permanently
    on it reward-hacks into "walk toward the enemy and stop," see curriculum.py
  - a permanent (non-curriculum-gated) stall tax: a small per-step penalty
    that only activates once no combat contact has happened for a while,
    added after OGRL-20260816-017's diagnostic found 63% of episodes never
    resolving and 30% having literally zero combat contact once
    closing_distance had fully tapered off. Deliberately outcome-based (did
    a hit land recently) rather than movement-based (are you close), so it
    doesn't dictate a single "always advance" strategy -- see compute()

None of these weights are tuned by any training run yet -- they're a
defensible, literature-typical starting point (dense shaping small relative
to sparse terminal-outcome rewards, so the agent optimizes for the real
objective and not the shaping term), not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

from obs_schema import ObsLayout, DEFAULT_LAYOUT


@dataclass
class RewardConfig:
    damage_taken_weight: float = 4.0     # per unit of own (temp+blood)/2 health lost, both pools in [0,1]
    damage_dealt_weight: float = 3.0     # per unit of a tracked opponent's (temp+blood)/2 health lost
    friendly_fire_weight: float = 6.0    # per unit of health lost by a teammate the agent itself hit (weighted
                                          # above damage_dealt -- hurting an ally should cost noticeably more
                                          # than a whiffed attack on an enemy would have earned) -- see compute()
    self_knockout_penalty: float = 10.0  # terminal, applied once on the step self becomes non-awake
    opponent_knockout_bonus: float = 8.0  # per opponent transitioning awake -> unconscious/dead this step
    time_cost: float = 0.01              # flat per-step cost, discourages stalling
    ragdoll_penalty: float = 0.05        # per-step cost while limp (ragdolled) and still awake -- see compute()
    closing_distance_weight: float = 0.0  # curriculum-gated; 0.0 = off (set by Curriculum, not by hand normally)
    closing_distance_cap: float = 30.0    # meters; beyond this, no closing-distance shaping (avoid rewarding
                                           # "walk toward a target 200m away" as if it were meaningful progress)
    stall_penalty_weight: float = 0.02   # per-step cost once genuinely stalled (see compute()) -- deliberately
                                          # NOT curriculum-gated/tapered like closing_distance: this is a permanent
                                          # backstop against total non-engagement, not an on-ramp bootstrap
    stall_grace_steps: int = 250         # steps of zero combat contact tolerated before the stall tax starts --
                                          # long enough to not punish early positioning/reads, short enough to bite
                                          # well before a 900-step episode cap (see compute())


def run8_reward_config() -> RewardConfig:
    """OGRL-20260816-021 Sec 2.4 / OGRL-20260816-022: runs 5-7's reward was
    never the limiting factor -- the actor never received a usable gradient
    at 120Hz decisions on a 10-character random-weapon brawl, so no amount of
    reward tuning could have shown up in behavior (Sec 2.3: "do not tune
    stall_penalty_weight next... that is optimising a term that has never
    influenced the policy"). This profile is deliberately the SIMPLEST
    defensible reward for the actual run8 question -- can the loop learn
    ANYTHING at all on a well-posed 1v1 -- not a tuned result:
      - symmetric terminal outcome (+10 win / -10 loss, was +8/-10 -- the
        asymmetry made aggression risk-averse in a task about aggression,
        Sec 2.2(c))
      - dense damage at a SYMMETRIC +/-1 scale per unit health (was 4.0
        taken / 3.0 dealt), matched to the terminal reward's own scale
        instead of dominating it
      - a much smaller time_cost (0.002/decision, not 0.01/step) -- at 30Hz
        over a 900-decision episode that's a maximum of -1.8, not -9, so it
        can no longer be larger than the win/loss terms it should be
        subordinate to
      - stall tax and ragdoll penalty OFF entirely (both are shaping for
        problems that don't have evidence of applying to a policy that has
        never learned anything yet; re-add them only once run8 clears its
        entropy gate and can demonstrably fight)
      - closing-distance shaping OFF too (via Curriculum's zeroed weights,
        not a field here) -- a well-defined 1v1 needs no engagement
        bootstrap the way a 10-character arena did
    """
    return RewardConfig(
        damage_taken_weight=1.0,
        damage_dealt_weight=1.0,
        friendly_fire_weight=6.0,  # unreachable on a clean 1v1 map (no ally can exist), left at its old
                                   # value purely so the field means the same thing if this profile is
                                   # ever pointed at a multi-character map later
        self_knockout_penalty=10.0,
        opponent_knockout_bonus=10.0,
        time_cost=0.002,
        ragdoll_penalty=0.0,
        stall_penalty_weight=0.0,
        stall_grace_steps=250,
    )


def _health_scalar(entity_or_self: dict | list) -> float:
    """(temp_health + blood_health) / 2, the two regenerating/bleed pools
    that respond to ordinary combat damage -- see rl_observation.cpp's own
    comment on why permanent_health is excluded from this kind of shaping
    (too sparse, only drops on serious/lethal hits)."""
    if isinstance(entity_or_self, dict):
        return 0.5 * (entity_or_self["temp_health"] + entity_or_self["blood_health"])
    layout = DEFAULT_LAYOUT
    return 0.5 * (entity_or_self[layout.TEMP_HEALTH] + entity_or_self[layout.BLOOD_HEALTH])


class RewardComputer:
    def __init__(self, layout: ObsLayout = DEFAULT_LAYOUT, config: RewardConfig | None = None):
        self.layout = layout
        self.config = config or RewardConfig()
        # Stall-tax state (OGRL-20260816-018) -- the one piece of per-episode
        # state this class carries; everything else is a pure function of
        # (prev_values, values). Must be reset on env.reset(), see
        # reset_episode() and OvergrowthEnv.reset()'s call to it.
        self._steps_since_contact = 0

    def reset_episode(self) -> None:
        """Call on every env.reset() -- otherwise a stall streak from the end
        of one episode would carry into the next and immediately tax a fresh
        start."""
        self._steps_since_contact = 0

    def compute(self, prev_values: list, values: list) -> tuple[float, dict]:
        """Returns (reward, info) for the transition prev_values -> values.
        `info` breaks the reward into named components for logging -- always
        inspect this when tuning weights, a single scalar hides which term is
        actually driving behavior."""
        cfg = self.config
        layout = self.layout
        components: dict[str, float] = {}

        # --- Self damage taken ---
        # OGRL-20260817-028 Sec7/-027 Sec4.1: SIGNED delta, not max(0, ...).
        # Overgrowth's temp_health regenerates -- clamping to only ever
        # charge for a drop and never credit a regain breaks the telescoping
        # that makes this term potential-based (Ng, Harada & Russell 1999):
        # summed over an episode, Sigma(self_health_prev - self_health_now)
        # collapses to (health_at_episode_start - health_at_episode_end) only
        # if regains are credited symmetrically with losses. The clamped
        # version was a real, non-policy-invariant penalty on taking damage
        # AT ALL rather than on ending the episode damaged -- invisible at
        # low difficulty (opponents rarely land enough hits for regen to
        # matter) and a genuine risk-aversion bias at high difficulty, where
        # trading two hits for one is often the winning play.
        self_health_prev = _health_scalar(prev_values)
        self_health_now = _health_scalar(values)
        damage_taken = self_health_prev - self_health_now  # positive = took damage, negative = regenerated
        components["damage_taken"] = -cfg.damage_taken_weight * damage_taken

        # --- Self knockout (terminal) ---
        was_awake = prev_values[layout.KNOCKED_OUT][0] > 0.5
        now_awake = values[layout.KNOCKED_OUT][0] > 0.5
        self_knocked_out_this_step = was_awake and not now_awake
        components["self_knockout"] = -cfg.self_knockout_penalty if self_knocked_out_this_step else 0.0

        # --- Opponent damage dealt / knockouts, matched by entity id ---
        # CAUSATION-GATED (OGRL-20260816-014): the training arena
        # (arenas/oval_arena.xml, Data/Scripts/arena_level.as) spawns ~10
        # characters across multiple teams that fight EACH OTHER, not just
        # this agent -- confirmed directly (two other characters' entity
        # blocks showing attacked_by_id pointing at each other, mid-fight,
        # while this agent was still elsewhere). Crediting the agent for any
        # visible entity's health loss/knockout, regardless of who caused
        # it, was a real bug: it rewarded proximity to ambient combat, not
        # the agent's own actions. schema v3's attacked_by_id (the target's
        # own MovementObject::attacked_by_id, set by the engine to the
        # attacker's id on every hit) fixes this -- only entities this agent
        # itself most recently hit are eligible for credit. Not a perfect
        # frame-exact trace (attacked_by_id has no timestamp, so a knockout
        # landing immediately after a *different* attacker's hit on the same
        # target could misattribute in rare cases) but a large, direct
        # improvement over the unconditional version, verified against real
        # engine data before being trusted (see the research-log entry).
        # is_ally added in schema v4 (OGRL-20260816-015), found while
        # re-checking for the SAME class of bug, not a separate incident:
        # causation alone (attacked_by_id == self.id) doesn't rule out
        # crediting the agent for hitting its own teammate. A hit on an ally
        # is penalized, not just excluded from credit -- it's actively
        # counterproductive behavior, not merely uninformative.
        self_id = layout.self_id(values)
        prev_entities = {e["id"]: e for e in layout.all_entities(prev_values) if e["valid"]}
        curr_entities = {e["id"]: e for e in layout.all_entities(values) if e["valid"]}
        damage_dealt = 0.0
        knockout_bonus = 0.0
        friendly_fire = 0.0
        hostile_kos = 0
        for entity_id, curr in curr_entities.items():
            prev = prev_entities.get(entity_id)
            if prev is None:
                continue  # only entered view this step; no prior health reading to diff against
            if curr["attacked_by_id"] != self_id:
                continue  # this entity's most recent hit wasn't from this agent -- not its credit
            prev_health = _health_scalar(prev)
            curr_health = _health_scalar(curr)
            # SIGNED delta (Sec7, see damage_taken's comment above for the
            # full potential-based-shaping rationale) -- positive = this
            # entity lost health, negative = it regenerated. Note the
            # causation gate just above (attacked_by_id == self_id) already
            # breaks the clean Sigma-telescopes-to-(start-end) property this
            # would otherwise have over a full episode, since credit stops
            # the instant someone else lands the next hit -- so this term is
            # only APPROXIMATELY potential-based, unlike damage_taken's exact
            # version. Still strictly better than the clamped version: it no
            # longer manufactures a one-directional bias on top of the
            # approximation the gate already introduces.
            health_lost = prev_health - curr_health
            prev_target_awake = prev["knocked_out"][0] > 0.5
            curr_target_awake = curr["knocked_out"][0] > 0.5
            target_knocked_out_this_step = prev_target_awake and not curr_target_awake
            if curr["is_ally"]:
                friendly_fire += health_lost
                if target_knocked_out_this_step:
                    friendly_fire += 1.0  # a full ally knockout is worse than chip damage, weighted like one below
            else:
                damage_dealt += health_lost
                if target_knocked_out_this_step:
                    knockout_bonus += cfg.opponent_knockout_bonus
                    hostile_kos += 1
        components["damage_dealt"] = cfg.damage_dealt_weight * damage_dealt
        components["opponent_knockout"] = knockout_bonus

        # Knockout EVENT count this step, for the win condition (OGRL-20260905).
        # Counting events and accumulating them over the episode is robust to
        # visibility; inferring "live hostiles" from the current observation is
        # NOT, because entities appear only while visible, so a hostile walking
        # out of line of sight would look like a knockout and end the episode as
        # a false win.
        # `won` used to mean "knocked out any opponent this step", which is
        # correct only at 1 opponent. At N it means "KO one of N and stop", and
        # since N opponents give N times the chances to land a KO, the measured
        # win rate RISES with opponent count -- observed live in run18 at
        # 0.71 / 0.74 / 0.81 for 1 / 2 / 3, which advanced the curriculum to its
        # cap on a meaningless signal. A win must mean every hostile is down.
        components["hostile_kos_this_step"] = float(hostile_kos)
        components["friendly_fire"] = -cfg.friendly_fire_weight * friendly_fire

        # --- Time cost ---
        components["time_cost"] = -cfg.time_cost

        # --- Stall tax (OGRL-20260816-018) ---
        # run5's checkpoint diagnostic (30 headless episodes at step
        # 2,539,520/3,000,000) found 63% of episodes hit the 900-step cap
        # with the fight never resolving -- and 9/30 (30%) had literally
        # ZERO combat contact for the entire episode, not just an
        # inconclusive fight. That's exactly the "standing by the walls"
        # behavior flagged from watching run3 live.
        #
        # Deliberately NOT a distance-based fix: closing_distance_weight
        # already tried "reward getting closer" and it's a movement-shaping
        # term that dictates HOW to engage (walk toward the nearest
        # hostile), which -- per explicit user direction -- risks collapsing
        # legitimate tactical variety (baiting, spacing, waiting for an
        # opening, retreating to reposition) into "always push forward."
        # This term is outcome-based instead: it only cares whether combat
        # contact (either side landing a hit -- including a friendly-fire
        # hit, since that still proves the agent isn't simply idle; the
        # friendly_fire weight above already makes that outcome net-negative
        # on its own) has happened recently, not how the agent got there or
        # where anyone is standing. A policy that stalks, retreats, and
        # circles for 200 steps before landing a clean hit pays nothing
        # extra; a policy that does nothing for the whole episode does.
        contact_this_step = (damage_taken > 0.0) or (damage_dealt > 0.0) or (friendly_fire > 0.0)
        if contact_this_step:
            self._steps_since_contact = 0
        else:
            self._steps_since_contact += 1
        stalled = self._steps_since_contact > cfg.stall_grace_steps
        components["stall_time"] = -cfg.stall_penalty_weight if stalled else 0.0

        # --- Ragdoll (limp) time penalty ---
        # Data/Scripts/aschar.as exposes a manual fast-recovery input while
        # ragdolled -- WantsToRollFromRagdoll(), the SAME edge-triggered
        # "crouch" press already proven to work for the standing roll
        # (OGRL-20260816-008) -- as an alternative to the slow automatic
        # get-up animation. Nothing in the reward function incentivized using
        # it. Found by a human actually watching the agent play, not by a
        # metric or a code review: the agent stayed limp waiting for
        # auto-recovery instead of rolling out. Small per-step cost while in
        # ragdoll state AND still awake (self_knockout above already covers
        # the "actually lost" case; this is specifically about vulnerable-
        # but-recoverable time, which is a distinct thing worth discouraging
        # on its own even before/unless a knockout follows from it).
        now_ragdoll = values[layout.STATE][4] > 0.5
        still_awake = values[layout.KNOCKED_OUT][0] > 0.5
        components["ragdoll_time"] = -cfg.ragdoll_penalty if (now_ragdoll and still_awake) else 0.0

        # --- Curriculum-gated closing-distance shaping ---
        # is_ally-filtered as of schema v4/OGRL-20260816-015 -- bootstrapping
        # "get closer to an enemy" is meaningful engagement shaping; "get
        # closer to whoever's nearest, ally or not" isn't, and silently
        # rewarding the latter is the same category of mistake as the
        # causation bug, just smaller in consequence.
        closing = 0.0
        if cfg.closing_distance_weight != 0.0:
            prev_hostiles = {eid: e for eid, e in prev_entities.items() if not e["is_ally"]}
            curr_hostiles = {eid: e for eid, e in curr_entities.items() if not e["is_ally"]}
            prev_nearest = _nearest_distance(prev_hostiles, cfg.closing_distance_cap)
            curr_nearest = _nearest_distance(curr_hostiles, cfg.closing_distance_cap)
            if prev_nearest is not None and curr_nearest is not None:
                closing = max(0.0, prev_nearest - curr_nearest) * cfg.closing_distance_weight
        components["closing_distance"] = closing

        total = sum(components.values())
        return total, components


def _nearest_distance(entities: dict, cap: float) -> float | None:
    distances = [e["distance"] for e in entities.values() if e["distance"] <= cap]
    return min(distances) if distances else None
