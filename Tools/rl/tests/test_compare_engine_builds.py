from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_engine_builds import _actions, _attack_events_observed_and_equal


class CompareEngineBuildsTests(unittest.TestCase):
    def test_action_trace_is_deterministic_and_legal(self) -> None:
        first = _actions(240)
        second = _actions(240)

        self.assertEqual(len(first), 240)
        self.assertEqual(len(second), 240)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left.shape, (8,))
            self.assertTrue(np.isfinite(left).all())
            self.assertLessEqual(float(np.linalg.norm(left[:2])), 1.000001)
            self.assertTrue(np.isin(left[2:], (0.0, 1.0)).all())

    def test_empty_action_trace_is_supported(self) -> None:
        self.assertEqual(_actions(0), [])

    def test_empty_attack_logs_do_not_establish_event_equivalence(self) -> None:
        empty = {"attack_event_count": 0, "attack_log_sha256": "same"}
        self.assertFalse(_attack_events_observed_and_equal(empty, empty))

    def test_nonempty_identical_attack_logs_establish_event_equivalence(self) -> None:
        left = {"attack_event_count": 2, "attack_log_sha256": "same"}
        right = {"attack_event_count": 1, "attack_log_sha256": "same"}
        self.assertTrue(_attack_events_observed_and_equal(left, right))

    def test_mismatched_attack_logs_fail_event_equivalence(self) -> None:
        left = {"attack_event_count": 1, "attack_log_sha256": "left"}
        right = {"attack_event_count": 1, "attack_log_sha256": "right"}
        self.assertFalse(_attack_events_observed_and_equal(left, right))


if __name__ == "__main__":
    unittest.main()
