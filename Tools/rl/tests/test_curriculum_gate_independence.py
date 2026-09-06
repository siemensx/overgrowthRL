"""The difficulty gate and the opponent-count gate must not be able to advance
or block each other.

`record_opponent_outcome` says exactly that in its own docstring and honours it
by filtering to the current opponent maximum. `record_episode_outcome` did not
filter at all, so outnumbered fights fed straight into the difficulty signal.
With `opp_keep_solo=0.35`, ~65% of episodes are 1v2/1v3, whose win rate is far
below `gate_win_rate=0.75`, so the pooled rate never cleared the threshold.

Measured on run21_mac before the fix: 4575 episodes recorded, `last_advance`
still None, `d_max` pinned at its 0.15 start, mean sampled difficulty 0.073
against a cap of 1.0.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from curriculum import ScenarioSampler  # noqa: E402


def _sampler(**kw):
    return ScenarioSampler(d_max_start=0.15, d_step=0.1, d_max_cap=1.0,
                           gate_window=300, gate_min_samples=50, gate_win_rate=0.75,
                           opponents_cap=3, **kw)


class TestGateIndependence(unittest.TestCase):
    def test_outnumbered_losses_do_not_block_the_difficulty_gate(self):
        """The regression. Solo competence is high (90%); outnumbered fights are
        losing (20%) and are the majority of episodes, exactly as opp_keep_solo
        produces. d_max must still advance on the solo evidence."""
        s = _sampler()
        d = s.d_max
        for i in range(300):
            solo = (i % 3 == 0)                       # ~33% solo, like opp_keep_solo=0.35
            won = (i % 10 != 0) if solo else (i % 5 == 0)   # solo ~90%, multi ~20%
            s.record_episode_outcome(d, won, opponents=1 if solo else 3)
        self.assertGreater(s.d_max, 0.15,
                           "difficulty gate stayed pinned despite strong solo performance -- "
                           "outnumbered fights are contaminating the difficulty signal again")

    def test_gate_does_not_advance_when_solo_performance_is_poor(self):
        """The other direction: easy multi-opponent wins must not push difficulty
        up when the solo evidence does not support it."""
        s = _sampler()
        d = s.d_max
        for i in range(300):
            solo = (i % 3 == 0)
            won = (i % 5 == 0) if solo else True      # solo 20%, multi 100%
            s.record_episode_outcome(d, won, opponents=1 if solo else 3)
        self.assertEqual(s.d_max, 0.15,
                         "difficulty advanced on outnumbered wins alone")

    def test_solo_only_stream_still_advances(self):
        s = _sampler()
        d = s.d_max
        for _ in range(120):
            s.record_episode_outcome(d, True, opponents=1)
        self.assertGreater(s.d_max, 0.15)

    def test_band_win_rates_still_report_the_full_mix(self):
        """Only the GATE is filtered. Telemetry must keep showing every episode,
        or the dashboard silently starts reporting solo-only performance."""
        s = _sampler()
        for i in range(200):
            s.record_episode_outcome(0.05, i % 2 == 0, opponents=3)
        bands = s.band_win_rates()
        measured = [v for v in bands.values() if v is not None]
        self.assertTrue(measured, "band_win_rates reported nothing for a 200-episode stream")
        self.assertAlmostEqual(measured[0], 0.5, places=1)

    def test_opponent_gate_is_unaffected_by_difficulty(self):
        s = _sampler(opp_gate_window=400, opp_gate_min_samples=50, opp_gate_win_rate=0.60)
        for _ in range(200):
            s.record_opponent_outcome(1, True)
        self.assertGreaterEqual(s.opponents_max, 2)


if __name__ == "__main__":
    unittest.main()


class TestCurriculumStatePersistence(unittest.TestCase):
    """Without checkpointed curriculum state, every resume restarts d_max at
    --d-max-start. An unattended run that restarts a few times re-climbs
    difficulty from scratch each time and never approaches the cap -- observed
    on run21_mac, where several restarts in one night each dropped d_max back
    to 0.15 after it had reached 0.45."""

    def test_round_trip(self):
        s = _sampler()
        for _ in range(400):
            s.record_episode_outcome(s.d_max, True, opponents=1)
        self.assertGreater(s.d_max, 0.15)
        state = s.curriculum_state()

        fresh = _sampler()
        self.assertEqual(fresh.d_max, 0.15)
        fresh.load_curriculum_state(state)
        self.assertAlmostEqual(fresh.d_max, s.d_max, places=6)

    def test_missing_state_is_a_no_op(self):
        """Older checkpoints have no curriculum key at all."""
        s = _sampler()
        s.load_curriculum_state(None)
        s.load_curriculum_state({})
        self.assertEqual(s.d_max, 0.15)

    def test_restore_is_clamped_to_this_runs_caps(self):
        """Lowering --d-max-cap or --opponents-cap on a resume must be honoured,
        not silently overridden by whatever the checkpoint recorded."""
        s = ScenarioSampler(d_max_start=0.15, d_step=0.1, d_max_cap=0.5, opponents_cap=2)
        s.load_curriculum_state({"d_max": 0.95, "opponents_max": 3})
        self.assertEqual(s.d_max, 0.5)
        self.assertEqual(s.opponents_max, 2)

    def test_restore_never_goes_below_the_start(self):
        s = _sampler()
        s.load_curriculum_state({"d_max": 0.01})
        self.assertEqual(s.d_max, 0.15)

    def test_outcome_history_is_not_restored(self):
        """Only the POSITION is carried. Replaying a stale outcome window would
        let a checkpoint advance the curriculum on episodes the new process
        never ran."""
        s = _sampler()
        state = s.curriculum_state()
        self.assertEqual(set(state.keys()), {"d_max", "opponents_max"})


class TestGateWindowSizing(unittest.TestCase):
    """The gate must window over SOLO episodes, then filter to the top band --
    not window over all episodes and filter twice.

    With opp_keep_solo=0.35 and d ~ U(0, d_max), only about
    gate_window * 0.35 * 0.286 ~= 30 of a 300-episode mixed window are solo AND
    in the top band, which is below gate_min_samples=50. The gate then never
    fires however well the agent performs -- observed on run21_mac at a measured
    solo win rate of 0.822 against a 0.75 threshold.
    """

    def test_gate_fires_under_a_realistic_episode_mix(self):
        import random
        rng = random.Random(0)
        s = _sampler()
        start = s.d_max
        for _ in range(1200):
            solo = rng.random() < 0.35                 # opp_keep_solo
            d = rng.uniform(0.0, s.d_max)              # exactly how sample_episode draws
            won = rng.random() < 0.82                  # the measured solo rate
            s.record_episode_outcome(d, won, opponents=1 if solo else 3)
        self.assertGreater(
            s.d_max, start,
            "gate never fired under a realistic mix despite an 0.82 solo win rate "
            "against a 0.75 threshold -- the qualifying sample count is starved again")

    def test_still_requires_genuine_solo_evidence(self):
        import random
        rng = random.Random(1)
        s = _sampler()
        for _ in range(1200):
            solo = rng.random() < 0.35
            d = rng.uniform(0.0, s.d_max)
            won = (rng.random() < 0.40) if solo else True   # solo poor, multi perfect
            s.record_episode_outcome(d, won, opponents=1 if solo else 3)
        self.assertEqual(s.d_max, 0.15, "advanced without solo evidence")
