"""Curriculum for the Overgrowth RL environment -- like reward.py, this is the
other half of the gap flagged in the plan-scope memory: the 8-stage plan
covers environment mechanics, never reward or curriculum.

Scoped narrowly and honestly: this is a **reward-shaping curriculum**
(phased changes to RewardConfig over the course of training), not an
**environment-composition curriculum** (progressively harder scenarios/
opponent counts/AI difficulty). The latter would need either new training
scenario levels or an engine-side hook to vary opponent composition at reset
time, neither of which exists yet -- flagged as a real, concrete next step
for whoever picks this up, not silently substituted for.

The one phase transition implemented: early training gets a small
closing-distance shaping bonus (reward.py's closing_distance_weight) to
bootstrap engagement, since a near-random initial policy in an open arena may
rarely make contact with an opponent at all, and damage/knockout rewards are
uninformative if contact never happens. That bonus is **linearly tapered to
zero** over `taper_steps`, not cut off abruptly -- the first version of this
curriculum used a hard cutoff, and `run1` (research-log OGRL-20260816-012)
showed a visible reward dip exactly at that boundary, the textbook signature
of a shaping term disappearing faster than the policy can adapt. A taper
gives the policy a gradual on-ramp to the sparse regime instead of a cliff.

`bootstrap_steps`/`taper_steps` widened after OGRL-20260816-014's causation
fix: the reward signal is now genuinely sparser than it was in run1/run2
(ambient-combat credit, which was a real bug, is gone), so bootstrapping
real engagement behavior plausibly needs more time before the shaping term
should start fading, not less.

Step-count-gated (not performance-gated) because there is no real training
run yet *with the corrected reward* to calibrate a performance threshold
against -- still a defensible starting point per the phase-based curricula
common in the PPO literature, explicitly not a tuned result.

Second phase transition (OGRL-20260816-019): the stall tax (reward.py's
stall_penalty_weight) ramps LINEARLY IN from 0 over stall_ramp_steps, rather
than applying at full strength from the moment it's introduced. This is the
same lesson as the closing_distance taper above, just facing the other
direction: run6 introduced the stall tax at full weight on top of a policy
resumed from run5 (already settled under a reward function that didn't have
it) and outcomes got WORSE, not better (30-episode diagnostic: run5's 7
WON/19 TIMEOUT baseline fell to 2 WON/25 TIMEOUT after run6, with the stall
tax firing in 29/30 episodes -- the textbook signature of a shaping term
appearing faster than the policy can adapt, mirroring run1's cutoff dip from
the other direction). A ramp gives the policy the same kind of gradual
on-ramp for a term being ADDED that closing_distance already gets for a term
being REMOVED. `stall_intro_step` is the global_step the ramp starts counting
from -- 0 for a cold start (where the ramp finishes early in bootstrap phase,
before there's any established policy behavior to disrupt), or the resumed
checkpoint's global_step when continuing a run, so the ramp is relative to
when the term is actually new to that policy, not to absolute training time.

base_config (OGRL-20260816-023): reward_config_for_step() used to always
construct a hardcoded RewardConfig() -- correct while every run shared one
reward profile, but run8 (OGRL-20260816-021 Sec 2.4) needs a genuinely
different base (reward.py's run8_reward_config()), not just different
shaping weights on top of the old one. Curriculum now takes that base as a
parameter (still defaulting to plain RewardConfig() for runs 1-7's
comparability) and applies the SAME shaping logic on top of whichever base
it's given -- the shaping mechanism (taper/ramp) and the reward profile it
shapes are genuinely separate concerns and were only coupled by an
implementation shortcut, not by necessity.
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass, field, replace

from reward import RewardConfig


class Curriculum:
    def __init__(
        self, bootstrap_steps: int = 500_000, taper_steps: int = 300_000, bootstrap_closing_weight: float = 0.05,
        stall_target_weight: float = 0.02, stall_ramp_steps: int = 300_000, stall_intro_step: int = 0,
        base_config: RewardConfig | None = None,
    ):
        self.bootstrap_steps = bootstrap_steps
        self.taper_steps = taper_steps
        self.bootstrap_closing_weight = bootstrap_closing_weight
        # See module docstring, OGRL-20260816-019 -- stall_intro_step lets a
        # resumed run ramp relative to ITS resume point, not absolute step 0.
        self.stall_target_weight = stall_target_weight
        self.stall_ramp_steps = stall_ramp_steps
        self.stall_intro_step = stall_intro_step
        self.base_config = base_config if base_config is not None else RewardConfig()

    def _closing_weight_for_step(self, global_step: int) -> float:
        if global_step < self.bootstrap_steps:
            return self.bootstrap_closing_weight
        taper_progress = (global_step - self.bootstrap_steps) / max(1, self.taper_steps)
        if taper_progress >= 1.0:
            return 0.0
        return self.bootstrap_closing_weight * (1.0 - taper_progress)

    def _stall_weight_for_step(self, global_step: int) -> float:
        ramp_progress = (global_step - self.stall_intro_step) / max(1, self.stall_ramp_steps)
        ramp_progress = max(0.0, min(1.0, ramp_progress))
        return self.stall_target_weight * ramp_progress

    def reward_config_for_step(self, global_step: int) -> RewardConfig:
        return replace(
            self.base_config,
            closing_distance_weight=self._closing_weight_for_step(global_step),
            stall_penalty_weight=self._stall_weight_for_step(global_step),
        )

    def phase_name(self, global_step: int) -> str:
        if global_step < self.bootstrap_steps:
            return "bootstrap"
        if global_step < self.bootstrap_steps + self.taper_steps:
            return "taper"
        return "main"


# --- Environment-composition curriculum (OGRL-20260817-028 Sec3) ---
#
# The gap the module docstring above names explicitly: everything before this
# point is reward-SHAPING (same scenario, different weights over training).
# ScenarioSampler is the environment-COMPOSITION curriculum -- what opponent,
# how hard, how many, armed or not -- made possible by the set_rl_* hook
# added to arena_level_1v1_unarmed.as and the ShmHeader fields that carry it
# (rl_shm_transport.cpp, shm_env.py). Per Sec3.2: sample d ~ U(0, d_max) fresh
# EVERY episode, not staged at d_max directly -- this keeps easy fights in
# the mix permanently (prevents catastrophic forgetting, the run6 stall-tax
# lesson generalized) and gives a continuous, always-available read of the
# skill-vs-difficulty curve. d_max only ever increases (never decreases: a
# regression should show up as a stalled/falling top-band win rate, which is
# a dashboard finding, not something this class should silently paper over
# by walking d_max back down on its own).

# Difficulty bands used for the reporting/conditioning side (§3.2's "every
# win rate reported after this lands must be conditioned on difficulty
# band") -- distinct from the advance-gate window, which always looks at
# [d_max-0.10, d_max] regardless of these fixed bands.
DIFFICULTY_BANDS = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)]


def band_for(d: float) -> str:
    for lo, hi in DIFFICULTY_BANDS:
        if lo <= d <= hi or (lo <= d and d < hi):
            return f"[{lo:.1f},{hi:.1f}]"
    return f"[{DIFFICULTY_BANDS[-1][0]:.1f},{DIFFICULTY_BANDS[-1][1]:.1f}]"


@dataclass
class ScenarioSampler:
    # d_max schedule (Sec3.2, Sec10 "tonight's run"): start LOW. Run9 trained
    # at an effective opponent difficulty of ~0.0-0.25; jumping straight to
    # 0.30 against either a settled resumed policy (shock) or a cold-started
    # one (signal-density -- at d=0.3 the opponent's block skill/damage are
    # already well above what a fresh near-random policy can get useful
    # learning signal against, since random play's ~41% baseline win rate at
    # d~0.1 is what let bootstrapping work at all) is the same class of
    # mistake run6's abrupt stall-tax introduction was. Let the gate raise it.
    d_max_start: float = 0.15
    d_max_cap: float = 1.0
    d_step: float = 0.10
    # OGRL-20260817-034: once d_max has climbed for a while, `d ~
    # Uniform(0, d_max)` keeps spending roughly d_min/d_max of all NEW
    # episodes on difficulty already comfortably mastered (band win rates
    # >90% below ~0.5, per run11/run12/run13's own telemetry) -- compute
    # that isn't going toward the actual measured gap (a deterministic eval
    # against run12.pt found normalized_skill=0.45 at d=1.0: the trained
    # policy has captured under half the available improvement over a
    # random-action baseline there). d_min raises the FLOOR of the sampled
    # range so a resumed/already-capped run can concentrate on the
    # difficulty band that still has real headroom, without fully pinning to
    # one exact configuration (which would lose the robustness the original
    # full-range design was for -- see this class's own module-level
    # reasoning above). 0.0 (default) preserves the original Uniform(0,
    # d_max) behavior exactly for any caller that doesn't set it.
    d_min: float = 0.0
    gate_window: int = 300          # episodes considered for the advance gate
    gate_min_samples: int = 50      # minimum qualifying (top-band) episodes before the gate can fire at all --
                                     # without this, a handful of lucky early wins right after start could advance
                                     # d_max on pure noise
    gate_win_rate: float = 0.75
    rng_seed: int = 0

    # Stage axes (Sec3.3's ladder) -- ALL represented here so the sampler is
    # forward-ready for B-G without a rewrite, but only Stage A's config is
    # exercised by anything that constructs this with the defaults below.
    # "stage" is advisory/telemetry only tonight; nothing auto-advances it --
    # per Sec10's own tonight's-recipe, Stage A is deliberately the only
    # stage actually launched, opponents/species/weapons axes are unlocked
    # by hand (new kwargs), not by an internal auto-progression this class
    # would otherwise need its own separate, unvalidated gate logic for.
    stage: str = "A"
    opponents: int = 1              # starting/minimum opponent count; see opponents_cap for the curriculum
    # --- Opponent-count curriculum (OGRL-20260905) ---
    # Now wired: gen_arena_map.py emits game_type 3 (1v2, teams [0,1,1]) and 4
    # (1v3, teams [0,1,1,1]), and arena_level_1v1_unarmed.as maps rl_opponents
    # onto them, falling back to the 1v1 pair on any level that lacks them.
    #
    # Growth mirrors the difficulty gate: unlock the next opponent count once the
    # win rate AT THE CURRENT MAXIMUM clears opp_gate_win_rate over a window.
    #
    # opp_keep_solo is the anti-forgetting term and is the whole point of the
    # mix: a fixed share of episodes stays 1v1 forever, so learning to survive a
    # crowd cannot quietly cost the 1v1 competence that run15-run17 spent 100M
    # decisions acquiring. Without it this axis is a distribution shift, not an
    # addition.
    opponents_cap: int = 1          # 1 disables the curriculum entirely (default = old behaviour)
    opp_gate_win_rate: float = 0.60 # lower than the difficulty gate: outnumbered fights are meant to be hard
    opp_gate_window: int = 400
    opp_gate_min_samples: int = 150
    opp_keep_solo: float = 0.35     # fraction of episodes held at 1v1 once the curriculum has advanced
    species_mode: int = 0           # rl_species value: 0 = legacy random guard/raider (Stage A default,
                                     # matches run8/run9's own opponent mix exactly), 4 = random of all 3 (Stage B)
    weapons_prob: float = 0.0       # probability a round is armed (Stage C axis)

    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # sample_episode/record_episode_outcome
                                                                                 # are called concurrently from
                                                                                 # VecOvergrowthEnv's worker threads
    _d_max: float = field(default=0.0, repr=False)
    _recent: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)  # (d, won, opponents) triples, most-recent-last;
                                                                                         # capped generously, gate only ever
                                                                                         # looks at the last gate_window
    _opp_max: int = field(default=1, repr=False)
    _opp_recent: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)  # (opponents, won)
    _opp_advance_log: list = field(default_factory=list, repr=False)
    _advance_log: list = field(default_factory=list, repr=False)  # (episode_index, old_d_max, new_d_max) for the research log / events.jsonl

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)
        self._d_max = self.d_max_start
        self._opp_max = max(1, min(self.opponents, self.opponents_cap))

    @property
    def d_max(self) -> float:
        return self._d_max

    def sample_episode(self) -> dict:
        """One call per episode reset. Returns the full set_rl_* payload."""
        with self._lock:
            # Clamp, don't assume d_min <= d_max: a fresh/resumed run that
            # hasn't climbed d_max past d_min yet (every resume restarts
            # d_max at d_max_start -- ScenarioSampler state isn't
            # checkpointed, see the known-open-items list) should fall back
            # to the original full-range behavior until it has, not sample
            # an empty or inverted range.
            lo = min(self.d_min, self._d_max)
            d = self._rng.uniform(lo, self._d_max)
            if self._opp_max <= 1:
                opponents = 1
            elif self._rng.random() < self.opp_keep_solo:
                opponents = 1                       # anti-forgetting: keep fighting 1v1
            else:
                opponents = self._rng.randint(2, self._opp_max)
        return {
            "difficulty": d,
            "opponents": opponents,
            "weapons": self.weapons_prob,
            "species": self.species_mode,
        }

    @property
    def opponents_max(self) -> int:
        return self._opp_max

    def record_opponent_outcome(self, opponents: int, won: bool) -> None:
        """Advance the opponent-count curriculum. Separate from the difficulty
        gate on purpose: they measure different things and must not be able to
        advance each other."""
        if self.opponents_cap <= 1:
            return
        with self._lock:
            self._opp_recent.append((int(opponents), bool(won)))
            if self._opp_max >= self.opponents_cap:
                return
            window = list(self._opp_recent)[-self.opp_gate_window:]
            at_max = [w for o, w in window if o >= self._opp_max]
            if len(at_max) < self.opp_gate_min_samples:
                return
            if sum(at_max) / len(at_max) >= self.opp_gate_win_rate:
                old = self._opp_max
                self._opp_max = min(self.opponents_cap, self._opp_max + 1)
                self._opp_advance_log.append((len(self._opp_recent), old, self._opp_max))

    def curriculum_state(self) -> dict:
        """The position both curricula have climbed to, for checkpointing.

        Deliberately NOT the outcome histories: those are windows used to decide
        the next advance, and replaying a resumed run's stale window would let a
        checkpoint advance the curriculum on episodes the new process never saw.
        Only the position is restored; the gates re-earn their next step."""
        with self._lock:
            return {"d_max": self._d_max, "opponents_max": self._opp_max}

    def load_curriculum_state(self, state: dict | None) -> None:
        """Restore a checkpointed position. Clamped to this run's own caps, so
        lowering --d-max-cap or --opponents-cap on a resume is still honoured
        rather than being silently overridden by the checkpoint."""
        if not state:
            return
        with self._lock:
            d = state.get("d_max")
            if isinstance(d, (int, float)):
                self._d_max = max(self.d_max_start, min(float(d), self.d_max_cap))
            o = state.get("opponents_max")
            if isinstance(o, int):
                self._opp_max = max(1, min(o, self.opponents_cap))

    def opponent_win_rates(self, window: int | None = None) -> dict:
        """Win rate per opponent count over the last `window` episodes -- the
        number that says whether 1v1 competence is being retained."""
        w = window if window is not None else self.opp_gate_window
        with self._lock:
            recent = list(self._opp_recent)[-w:]
        out = {}
        for n in range(1, self.opponents_cap + 1):
            sub = [won for o, won in recent if o == n]
            out[n] = (sum(sub) / len(sub)) if sub else None
        return out

    def record_episode_outcome(self, difficulty: float, won: bool, opponents: int = 1) -> None:
        """Call once per completed episode with the difficulty it was
        actually sampled at (not the current d_max), whether the RL agent won,
        and how many opponents it faced. Advances d_max in place when the gate
        is satisfied.

        The gate counts SOLO episodes only. record_opponent_outcome's docstring
        already states the intent -- "Separate from the difficulty gate on
        purpose: they measure different things and must not be able to advance
        each other" -- and it honours that by filtering to the current opponent
        maximum. This gate did not filter at all, so outnumbered fights fed
        straight into the difficulty signal. With opp_keep_solo=0.35 that means
        ~65% of the samples came from 1v2/1v3 fights whose win rate is far below
        gate_win_rate, dragging the pooled rate under the threshold forever.

        Measured on run21_mac (2026-09-06): 4575 episodes recorded, `last_advance`
        still None, d_max pinned at its 0.15 start and mean sampled difficulty
        0.073 against a cap of 1.0 -- i.e. the agent had been training against
        near-trivial opponents for the whole multi-opponent phase, and its
        1v2/1v3 win rates were measured against them.

        Every episode is still stored, so band_win_rates and the telemetry keep
        reporting the full mix; only the gate is filtered."""
        with self._lock:
            self._recent.append((difficulty, won, int(opponents)))
            if self._d_max >= self.d_max_cap:
                return
            window = list(self._recent)[-self.gate_window:]
            top_band_lo = self._d_max - self.d_step
            qualifying = [w for d, w, o in window if d >= top_band_lo and o == 1]
            if len(qualifying) < self.gate_min_samples:
                return
            win_rate = sum(qualifying) / len(qualifying)
            if win_rate >= self.gate_win_rate:
                old = self._d_max
                self._d_max = min(self.d_max_cap, self._d_max + self.d_step)
                self._advance_log.append((len(self._recent), old, self._d_max))

    def band_win_rates(self, window: int | None = None) -> dict:
        """Per-fixed-band (DIFFICULTY_BANDS) win rate over the last `window`
        episodes (default: gate_window) -- for curriculum_live telemetry.
        None (not a float) for a band with zero samples in the window,
        rather than silently reporting 0.0 as if it had been measured."""
        w = window if window is not None else self.gate_window
        with self._lock:
            recent = list(self._recent)[-w:]
        out = {}
        for lo, hi in DIFFICULTY_BANDS:
            label = f"[{lo:.1f},{hi:.1f}]"
            outcomes = [won for d, won, _o in recent if (lo <= d <= hi if hi == 1.0 else lo <= d < hi)]
            out[label] = (sum(outcomes) / len(outcomes)) if outcomes else None
        return out

    def snapshot(self) -> dict:
        """Everything metrics.jsonl's curriculum_live block needs (Sec3.2)."""
        with self._lock:
            recent_d = [d for d, _w, _o in list(self._recent)[-self.gate_window:]]
            d_max = self._d_max
            episodes_recorded = len(self._recent)
            last_advance = self._advance_log[-1] if self._advance_log else None
        return {
            "d_max": d_max,
            "d_max_cap": self.d_max_cap,
            "stage": self.stage,
            "opponents": self.opponents,
            "species_mode": self.species_mode,
            "weapons_prob": self.weapons_prob,
            "band_win_rate": self.band_win_rates(),
            "d_mean_sampled": (sum(recent_d) / len(recent_d)) if recent_d else None,
            "episodes_recorded": episodes_recorded,
            "last_advance": last_advance,
        }
