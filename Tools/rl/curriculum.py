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


# stage: (label, armed_count, weapon_type, throw_aggression, species_mode, min_opponents)
# weapon_type 0=random 1=knife 2=big_sword 3=sword 4=spear
# species_mode 0=guard/raider 6=guard/raider/cat
ARMED_STAGES = [
    ("A unarmed",        0, 0, 1.0, 0, 1),
    ("B1 1-armed-of-2",  1, 1, 4.0, 0, 2),
    ("B2 2-armed-of-2",  2, 1, 4.0, 0, 2),
    ("B3 1-armed-of-3",  1, 1, 4.0, 0, 3),
    ("B4 2-armed-of-3",  2, 0, 4.0, 0, 3),
    ("B5 3-armed-of-3",  3, 0, 4.0, 0, 3),
    ("C  cats + mixed",  3, 0, 6.0, 6, 3),
]


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
    # reasoning above).
    #
    # 2026-09-09: pinned to 1.0. The knob existed since -034 and was never
    # turned. The difficulty curriculum only ever raises the CEILING (d_max);
    # nothing lowers the amount of easy work once the ceiling is up, so a run
    # that reached d_max=1.0 at ~120M steps went on sampling U(0, 1) for the
    # next 160M. Measured over 20,004 episodes at 279M, that is what the
    # compute was buying:
    #
    #     opp1 d0.0-0.2  0.896     opp3 d0.0-0.2  0.872
    #     opp1 d0.8-1.0  0.789     opp3 d0.8-1.0  0.374
    #
    # ~94% of episodes were in cells already won 65-96% of the time, and 6.3%
    # in the only cell that is stuck. The robustness argument above does not
    # survive contact with the target: a human opponent always plays at 1.0,
    # so difficulty below 1.0 is not a band worth being robust across.
    d_min: float = 1.0
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
    # opp_keep_solo is the anti-forgetting term: a fixed share of episodes stays
    # 1v1 forever, so learning to survive a crowd cannot quietly cost the 1v1
    # competence that run15-run17 spent 100M decisions acquiring. Without it
    # this axis is a distribution shift, not an addition.
    #
    # 2026-09-09: set to 0.0. That is a deliberate acceptance of the shift.
    # Measured at 279M with difficulty pinned to 1.0, the mixture was spending
    # 36% of episodes on 1v1 (won 0.778) and 31% on 1v2 (0.550) while the only
    # cell that matters, 1v3, sat at 0.408 and had not moved in 20M steps.
    # There is no compute to spare defending competence at counts the agent is
    # not going to be judged on. With this at zero the sampler pins every
    # episode to _opp_max, so the training distribution IS the evaluation
    # distribution: 1v3, unarmed, difficulty 1.0. The armed ladder does not
    # open until that clears armed_gate_win_rate (0.70) deterministically.
    #
    # Retention at 1v1/1v2 is expected to decay. That is the trade. Set this
    # back above zero to restore the mixture.
    opponents_cap: int = 1          # 1 disables the curriculum entirely (default = old behaviour)
    opp_gate_win_rate: float = 0.60 # lower than the difficulty gate: outnumbered fights are meant to be hard
    opp_gate_window: int = 400
    opp_gate_min_samples: int = 150
    opp_keep_solo: float = 0.0      # fraction of episodes held at 1v1 once the curriculum has advanced;
                                    # 0.0 pins every episode to _opp_max (see the note above)
    # --- Stage B/C: armed opponents (2026-09-07) ---
    #
    # The jump-kick monoculture survives every knob on the UNARMED scripted AI:
    # its only counter, the roll-away, is saturated at d_max=1.0 and costs the
    # policy nothing there. An armed opponent is a different matter, because
    # Dynamic AI Aggression's throw rule fires on sub_goal == _avoid_jump_kick,
    # which is set precisely when the agent is AIRBORNE. A knife thrown at a
    # committed jump kick is the first thing in this game that punishes it.
    #
    # Ramp: how many of the hostiles are armed, then which weapons, then cats
    # (whose controller throws on its own account too). Each row is gated on
    # win rate exactly like difficulty and opponent count.
    armed_stage: int = 0            # index into ARMED_STAGES; advances automatically
    # Sized to be EVIDENCE, not a formality. The first version (150 samples at
    # 0.60) let the ladder climb four rungs in ten minutes: ~40% of ~690
    # episodes per 5 min are armed, so 150 armed rounds accrue in about two
    # minutes, and the window clears on every advance so it simply refilled.
    # A rung must now cost a real sample at a real standard.
    armed_gate_win_rate: float = 0.70
    armed_gate_window: int = 1200
    armed_gate_min_difficulty: float = 0.8   # the gate evaluates at 1.0; only near-1.0
                                             # training episodes are evidence for it
    armed_gate_min_samples: int = 600
    species_mode: int = 0           # rl_species value: 0 = legacy random guard/raider (Stage A default,
                                     # matches run8/run9's own opponent mix exactly), 4 = random of all 3 (Stage B)
    weapons_prob: float = 0.0       # probability a round is armed (Stage C axis)

    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # sample_episode/record_episode_outcome
                                                                                 # are called concurrently from
                                                                                 # VecOvergrowthEnv's worker threads
    _d_max: float = field(default=0.0, repr=False)
    # The difficulty gate windows over SOLO episodes only. Windowing over
    # episodes of any kind and then filtering to solo makes the effective sample
    # count a fraction of gate_window -- with opp_keep_solo=0.35 and d ~ U(0,
    # d_max), only 300 x 0.35 x 0.286 ~= 30 of a 300-episode window qualify,
    # below gate_min_samples=50, so the gate can never fire however well the
    # agent performs. Measured on run21_mac: solo win rate 0.822 in the top band
    # against a 0.75 threshold, and d_max still pinned with last_advance None.
    _recent_solo: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)  # (d, won) for opponents == 1
    _recent: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)  # (d, won, opponents) triples, most-recent-last;
                                                                                         # capped generously, gate only ever
                                                                                         # looks at the last gate_window
    _opp_max: int = field(default=1, repr=False)
    _armed_stage: int = field(default=0, repr=False)
    _armed_recent: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)
    _armed_advance_log: list = field(default_factory=list, repr=False)
    _gate_pending: object = field(default=None, repr=False)
    _opp_recent: deque = field(default_factory=lambda: deque(maxlen=100_000), repr=False)  # (opponents, won)
    _opp_advance_log: list = field(default_factory=list, repr=False)
    _advance_log: list = field(default_factory=list, repr=False)  # (episode_index, old_d_max, new_d_max) for the research log / events.jsonl

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)
        self._d_max = self.d_max_start
        self._opp_max = max(1, min(self.opponents, self.opponents_cap))
        self._armed_stage = max(0, min(int(self.armed_stage), len(ARMED_STAGES) - 1))

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
            elif self.opp_keep_solo <= 0.0:
                # No anti-forgetting term means no mixture at all: train the
                # exact configuration the gate certifies, nothing adjacent.
                opponents = self._opp_max
            elif self._rng.random() < self.opp_keep_solo:
                opponents = 1                       # anti-forgetting: keep fighting 1v1
            else:
                opponents = self._rng.randint(2, self._opp_max)
            label, armed, weap, throw_aggr, species, min_opp = ARMED_STAGES[
                min(self._armed_stage, len(ARMED_STAGES) - 1)]
            # A stage that needs 2+ hostiles must not be sampled as a 1v1; the
            # anti-forgetting solo episodes stay UNARMED so the 1v1 skill the
            # eval tracks is still measured against the same opponent it always was.
            if opponents < min_opp:
                armed = 0
        return {
            "difficulty": d,
            "opponents": opponents,
            "weapons": self.weapons_prob,
            "species": species if armed > 0 else self.species_mode,
            "armed_count": armed,
            "weapon_type": weap,
            "throw_aggression": throw_aggr if armed > 0 else 1.0,
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

    def record_armed_outcome(self, won: bool, armed: int,
                             opponents: int = None, difficulty: float = None) -> None:
        """Advance the armed-opponent ladder. Only ARMED episodes count: the
        unarmed anti-forgetting solo rounds must not be able to promote a stage
        whose difficulty they never sampled -- the same independence the
        difficulty and opponent gates already keep from each other.

        The window must also be restricted to the SAME CELL the deterministic
        gate tests. Training samples difficulty ~ U(d_min, d_max) and opponents
        over {1..opp_max}, so the pooled win rate is dominated by easy cells.
        Measured on run21_win at 279M over 20,004 episodes:

            opp1  d0.0-0.2  0.896      opp3  d0.0-0.2  0.872
            opp1  d0.8-1.0  0.789      opp3  d0.8-1.0  0.374
            POOLED                                     0.782

        The pre-filter was reading that 0.782, clearing its 0.70 bar every
        time, and nominating -- while the deterministic gate ran at 3
        opponents and difficulty 1.0 and read 0.33. The two numbers were
        measuring different fights, which is why the gate fired every ~220k
        steps and held every single time for 5.4M steps. That is the third
        instance of this bug class in this run: the stochastic gate measured
        the wrong POLICY, the opponent-count bug measured the wrong COUNT, and
        this one measured the wrong DIFFICULTY MIX.

        opponents/difficulty are optional so the existing tests, which only
        exercise the armed filter, keep passing unchanged."""
        if self._armed_stage >= len(ARMED_STAGES) - 1:
            return
        with self._lock:
            if self._armed_stage > 0 and int(armed) <= 0:
                return          # unarmed round at an armed stage: not evidence
            if opponents is not None and int(opponents) < self._opp_max:
                return          # easier count than the gate tests
            if difficulty is not None and float(difficulty) < self.armed_gate_min_difficulty:
                return          # easier difficulty than the gate tests
            self._armed_recent.append(bool(won))
            window = list(self._armed_recent)[-self.armed_gate_window:]
            if len(window) < self.armed_gate_min_samples:
                return
            if sum(window) / len(window) >= self.armed_gate_win_rate:
                # DO NOT advance on this number. It is the STOCHASTIC policy --
                # training samples from the action distribution. On 2026-09-07
                # that read 58.5% at stage 6 while the DETERMINISTIC policy, the
                # one that gets deployed and the one a human watches, scored
                # ZERO wins and zero knockdowns there. The ladder promoted six
                # times on a number that corresponded to nothing, and 74M steps
                # were spent in fights the agent could not score in.
                #
                # This only flags a CANDIDATE. train_vec then runs a real
                # deterministic evaluation at this rung and calls
                # confirm_advance() with the result.
                self._gate_pending = (sum(window) / len(window), len(window))

    def gate_pending(self):
        """(stochastic_win_rate, n) if the cheap pre-filter has fired, else None."""
        with self._lock:
            return self._gate_pending

    def confirm_advance(self, det_win_rate: float, det_n: int) -> bool:
        """Advance only if the DETERMINISTIC policy clears the bar at this rung."""
        with self._lock:
            pend, self._gate_pending = self._gate_pending, None
            if pend is None:
                return False
            if det_win_rate < self.armed_gate_win_rate:
                self._armed_recent.clear()   # re-earn the pre-filter before retrying
                return False
            old = self._armed_stage
            self._armed_stage = min(len(ARMED_STAGES) - 1, self._armed_stage + 1)
            self._armed_recent.clear()
            self._armed_advance_log.append(
                (old, self._armed_stage, ARMED_STAGES[self._armed_stage][0],
                 det_win_rate, det_n))
            return True

    def stage_params(self) -> dict:
        """The current rung's scenario, for the deterministic gate eval."""
        with self._lock:
            label, armed, weap, aggr, species, min_opp = ARMED_STAGES[self._armed_stage]
            # The gate must fight the HARDEST configuration this rung trains
            # on, not the easiest one it permits. min_opp only decides which
            # episodes get armed; using it as the gate's opponent count made
            # stage 0 promote on a 1v1 -- a fight the policy wins ~90% of the
            # time -- while its actual 1v3 deterministic rate was 0.450.
            return {"label": label, "armed_count": armed, "weapon_type": weap,
                    "throw_aggression": aggr, "species": species,
                    "opponents": max(min_opp, self._opp_max)}

    @property
    def armed_stage_index(self) -> int:
        return self._armed_stage

    @property
    def armed_stage_label(self) -> str:
        return ARMED_STAGES[self._armed_stage][0]

    def take_armed_advances(self) -> list:
        """Pop stage transitions so the trainer can log them WITH the global
        step they happened at -- the sampler has no idea what step it is."""
        with self._lock:
            out, self._armed_advance_log = self._armed_advance_log, []
            return out

    def curriculum_state(self) -> dict:
        """The position both curricula have climbed to, for checkpointing.

        Deliberately NOT the outcome histories: those are windows used to decide
        the next advance, and replaying a resumed run's stale window would let a
        checkpoint advance the curriculum on episodes the new process never saw.
        Only the position is restored; the gates re-earn their next step."""
        with self._lock:
            return {"d_max": self._d_max, "opponents_max": self._opp_max,
                    "armed_stage": self._armed_stage}

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
            st = state.get("armed_stage")
            if st is not None:
                self._armed_stage = max(0, min(int(st), len(ARMED_STAGES) - 1))


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
            if int(opponents) == 1:
                self._recent_solo.append((difficulty, won))
            if self._d_max >= self.d_max_cap:
                return
            # Window over SOLO episodes, then filter to the top band -- not the
            # other way round. See _recent_solo's comment for why.
            window = list(self._recent_solo)[-self.gate_window:]
            top_band_lo = self._d_max - self.d_step
            qualifying = [w for d, w in window if d >= top_band_lo]
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
