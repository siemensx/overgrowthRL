from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_engine_builds import _actions


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


if __name__ == "__main__":
    unittest.main()
