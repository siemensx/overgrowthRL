from __future__ import annotations

import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vec_env import (  # noqa: E402
    _build_initial_wave,
    _cleanup_failed_initialization,
    _collect_initial_resets,
)


class _FakeEnv:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class InitialEngineWaveCleanupTests(unittest.TestCase):
    def test_wave_failure_waits_for_and_closes_every_successful_peer(self):
        created: list[_FakeEnv] = []
        completed: set[str] = set()

        def make(name: str, delay: float):
            time.sleep(delay)
            completed.add(name)
            if name == "fails":
                raise RuntimeError("engine launch failed")
            env = _FakeEnv(name)
            created.append(env)
            return env

        with ThreadPoolExecutor(max_workers=3) as pool:
            with self.assertRaisesRegex(RuntimeError, "engine launch failed"):
                _build_initial_wave(
                    pool,
                    make,
                    [("slow", 0.03), ("fails", 0.01), ("fast", 0.0)],
                )

        self.assertEqual(completed, {"slow", "fails", "fast"})
        self.assertEqual(len(created), 2)
        self.assertTrue(all(env.closed for env in created))

    def test_successful_wave_returns_environment_order_not_completion_order(self):
        def make(name: str, delay: float):
            time.sleep(delay)
            return _FakeEnv(name)

        with ThreadPoolExecutor(max_workers=2) as pool:
            built = _build_initial_wave(pool, make, [("slow", 0.03), ("fast", 0.0)])

        self.assertEqual([env.name for env in built], ["slow", "fast"])
        self.assertFalse(any(env.closed for env in built))


class InitialStandbyResetCleanupTests(unittest.TestCase):
    def test_reset_wave_waits_for_every_sibling_before_raising(self):
        completed: set[str] = set()

        def reset(name: str, delay: float):
            time.sleep(delay)
            completed.add(name)
            if name == "fails":
                raise RuntimeError("standby reset failed")
            return name

        with ThreadPoolExecutor(max_workers=3) as pool:
            with self.assertRaisesRegex(RuntimeError, "standby reset failed"):
                _collect_initial_resets(
                    pool,
                    lambda item: reset(*item),
                    [("slow", 0.03), ("fails", 0.01), ("fast", 0.0)],
                )

        self.assertEqual(completed, {"slow", "fails", "fast"})

    def test_constructor_failure_closes_all_engines_and_both_executors(self):
        class FakeExecutor:
            def __init__(self):
                self.shutdown_args = None

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_args = (wait, cancel_futures)

        envs = [_FakeEnv("active"), _FakeEnv("standby")]
        executors = [FakeExecutor(), FakeExecutor()]
        _cleanup_failed_initialization(envs, executors)

        self.assertTrue(all(env.closed for env in envs))
        self.assertEqual(
            [executor.shutdown_args for executor in executors],
            [(True, True), (True, True)],
        )


if __name__ == "__main__":
    unittest.main()
