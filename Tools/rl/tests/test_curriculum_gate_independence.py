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
