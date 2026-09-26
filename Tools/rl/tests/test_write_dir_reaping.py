#!/usr/bin/env python3
"""Regression tests for the stale-write-dir leak that parked run27 for 35 hours.

On 2026-09-25 run27 stopped training at 300,017,372 steps and sat motionless
for ~35 h (OGRL-20260926-002). It was NOT a deadlock: `train_vec.py` was in
the disk-low pause loop's `time.sleep(30.0)`, waiting for a human to free
space on a 476 GB disk with 1.3 GB left. 55.5 GB of that was 3,268 orphaned
engine `--write-dir` sandboxes belonging to prior runs.

`env._cleanup_stale_write_dirs` was written years-of-project-time earlier to
prevent exactly this, and had already been paid for twice
(OGRL-20260815-034, 2026-08-17/run11). It did nothing on the trainer because
it shelled out to `ps -eo command`, which does not exist on Windows: the
resulting FileNotFoundError is an OSError, the handler caught it, and the
function returned having deleted nothing. A safety net that fails silently
is worse than no safety net, because the failure mode looks like a deadlock.

What these tests pin:
  1. Liveness must be resolvable on THIS platform. This is the test that
     would have failed on the trainer on day one.
  2. Unknown liveness must never be read as "nothing is live" -- the sweep
     must delete nothing rather than risk deleting a live sandbox.
  3. A live write-dir is never reaped; a stale one always is, with its
     sibling .log.
  4. Only `env-*` directories are touched.
  5. force=True re-sweeps a parent this process already swept, which is what
     the trainer's disk-low recovery depends on.

Run:  python3 Tools/rl/tests/test_write_dir_reaping.py
"""
from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import env as env_mod  # noqa: E402


def _make(parent: Path, name: str, *, with_log: bool = True, size: int = 1024) -> Path:
    d = parent / name
    d.mkdir(parents=True)
    (d / "payload.bin").write_bytes(b"\0" * size)
    if with_log:
        (parent / f"{name}.log").write_text("engine log\n")
    return d


class ResetSweepMemory(unittest.TestCase):
    """_STALE_CLEANED_PARENTS is module-global and one-shot per parent."""

    def setUp(self) -> None:
        env_mod._STALE_CLEANED_PARENTS.clear()

    def tearDown(self) -> None:
        env_mod._STALE_CLEANED_PARENTS.clear()


class TestLivenessIsResolvableHere(ResetSweepMemory):
    def test_platform_can_list_process_command_lines(self):
        """THE test that was missing. On Windows this returned None forever
        (no `ps`), which disabled the whole sweep silently."""
        out = env_mod._live_process_command_lines()
        self.assertIsNotNone(
            out, "process liveness unresolvable on this platform -- the stale "
                 "write-dir sweep is disabled here, which is the run27 bug")
        self.assertIsInstance(out, str)

    def test_listing_contains_this_python_process(self):
        """A listing that cannot see our own process cannot be trusted to see
        a live engine either."""
        out = env_mod._live_process_command_lines()
        self.assertIsNotNone(out)
        self.assertIn("python", out.lower())


class TestUnknownLivenessDeletesNothing(ResetSweepMemory):
    def test_none_from_liveness_probe_reaps_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / ".rl_write_dirs"
            parent.mkdir()
            d = _make(parent, "env-ogrl_x-abc123")
            orig = env_mod._live_process_command_lines
            env_mod._live_process_command_lines = lambda: None
            try:
                removed = env_mod._cleanup_stale_write_dirs(parent)
            finally:
                env_mod._live_process_command_lines = orig
            self.assertEqual(removed, 0)
            self.assertTrue(d.exists(), "deleted a sandbox while liveness was UNKNOWN")
            self.assertTrue((parent / "env-ogrl_x-abc123.log").exists())

    def test_unknown_liveness_does_not_mark_parent_swept(self):
        """Otherwise one failed probe permanently disables the sweep for the
        process's whole lifetime -- a multi-week run."""
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / ".rl_write_dirs"
            parent.mkdir()
            _make(parent, "env-ogrl_x-abc123")
            orig = env_mod._live_process_command_lines
            env_mod._live_process_command_lines = lambda: None
            try:
                env_mod._cleanup_stale_write_dirs(parent)
            finally:
                env_mod._live_process_command_lines = orig
            self.assertNotIn(parent.resolve(), env_mod._STALE_CLEANED_PARENTS)


class TestReapingSelectivity(ResetSweepMemory):
    def test_live_kept_stale_reaped_nonenv_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / ".rl_write_dirs"
            parent.mkdir()
            live = _make(parent, "env-ogrl_r99_1-liveaaaa")
            stale = _make(parent, "env-ogrl_r99_2-stalebbb")
            bench = _make(parent, "bench", with_log=False)  # fixed-name scratch, reused not leaked

            orig = env_mod._live_process_command_lines
            env_mod._live_process_command_lines = (
                lambda: f'Overgrowth.exe --write-dir "{live}" --working-dir C:\\ogrl\n')
            try:
                removed = env_mod._cleanup_stale_write_dirs(parent)
            finally:
                env_mod._live_process_command_lines = orig

            self.assertEqual(removed, 1)
            self.assertTrue(live.exists(), "reaped a LIVE engine sandbox")
            self.assertTrue((parent / "env-ogrl_r99_1-liveaaaa.log").exists())
            self.assertFalse(stale.exists(), "failed to reap an orphaned sandbox")
            self.assertFalse((parent / "env-ogrl_r99_2-stalebbb.log").exists(),
                             "orphaned sandbox reaped but its sibling .log leaked")
            self.assertTrue(bench.exists(), "reaped a non-env-* directory")

    def test_live_match_survives_windows_path_quoting(self):
        """Windows reports the path as it was PASSED, which need not equal
        Path.resolve()'s spelling -- the old exact-full-path substring match
        was fragile here. Basename matching is what makes this safe."""
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / ".rl_write_dirs"
            parent.mkdir()
            live = _make(parent, "env-ogrl_r99_3-quotecc")
            weird = str(live).replace("/", "\\").upper()
            orig = env_mod._live_process_command_lines
            env_mod._live_process_command_lines = lambda: f'Overgrowth.exe --write-dir {weird}\n'
            try:
                env_mod._cleanup_stale_write_dirs(parent)
            finally:
                env_mod._live_process_command_lines = orig
            self.assertTrue(live.exists(),
                            "a live sandbox was reaped because the OS spelled its path differently")


class TestForceResweep(ResetSweepMemory):
    def test_second_sweep_is_a_noop_without_force(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / ".rl_write_dirs"
            parent.mkdir()
            orig = env_mod._live_process_command_lines
            env_mod._live_process_command_lines = lambda: "no engines here\n"
            try:
                env_mod._cleanup_stale_write_dirs(parent)          # first sweep marks parent
                later = _make(parent, "env-ogrl_later-ddddeeee")   # accumulated afterwards
                self.assertEqual(env_mod._cleanup_stale_write_dirs(parent), 0)
                self.assertTrue(later.exists())
                # ...and force=True is what the disk-low recovery path uses.
                self.assertEqual(env_mod._cleanup_stale_write_dirs(parent, force=True), 1)
                self.assertFalse(later.exists())
            finally:
                env_mod._live_process_command_lines = orig


class TestTrainerRecoveryWiring(unittest.TestCase):
    def test_disk_low_pause_reaps_before_waiting(self):
        """The pause loop must attempt recovery itself. run27 waited 35 h for a
        human to delete files the run itself had orphaned."""
        src = (HERE.parent / "ppo" / "train_vec.py").read_text()
        self.assertIn("from env import _cleanup_stale_write_dirs", src)
        i_pause = src.index("disk_low_pause")
        i_reap = src.index("_cleanup_stale_write_dirs(")
        i_sleep = src.index("time.sleep(30.0)")
        self.assertLess(i_pause, i_reap, "reap must happen after entering the disk-low branch")
        self.assertLess(i_reap, i_sleep, "reap must happen BEFORE parking on the 30s wait")
        self.assertIn("force=True", src[i_reap:i_reap + 200])

    def test_disk_low_pause_heartbeats(self):
        """A pause with no ongoing record is externally indistinguishable from
        a deadlock -- which is exactly how run27 was first diagnosed."""
        src = (HERE.parent / "ppo" / "train_vec.py").read_text()
        self.assertIn("disk_low_waiting", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
