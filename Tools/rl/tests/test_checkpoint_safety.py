#!/usr/bin/env python3
"""Regression tests for the checkpoint failure that cost 25.4M steps.

On 2026-09-05 a sync job restored a 4-hour-old checkpoint over live training
state every 15 minutes (OGRL-20260905-066). Nothing detected it: throughput,
win rate, difficulty and engine count were all sampled continuously and all
looked healthy, because none of them reads the checkpoint. The run only
appeared broken at the moment it had to be resumed.

These tests pin the three properties that make that class of failure
impossible at the point of writing, rather than relying on any external tool
behaving well.

Run:  python3 Tools/rl/tests/test_checkpoint_safety.py
"""
from __future__ import annotations
import os, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from train import _save_checkpoint, _checkpoint_step  # noqa: E402


class _Layout:
    total_floats = 339
    max_visible_entities = 8
    local_geometry_rays = 16
    action_history_steps = 4


class _Fake:
    """Minimal stand-in exposing the surface _save_checkpoint touches."""
    layout = _Layout()
    frame_stack = 4

    def state_dict(self):
        return {"w": torch.zeros(2)}


def _save(path, step, **kw):
    f = _Fake()
    _save_checkpoint(str(path), f, f, f, f, step, **kw)


class CheckpointSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ckpt = self.dir / "run.pt"
        os.environ.pop("OGRL_ALLOW_CHECKPOINT_REGRESSION", None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_and_reads_back(self):
        _save(self.ckpt, 1_000)
        self.assertEqual(_checkpoint_step(self.ckpt), 1_000)

    def test_leaves_no_temp_file(self):
        """A partial .tmp visible to a reader is how a torn copy happens."""
        _save(self.ckpt, 1_000)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_advancing_step_is_allowed(self):
        _save(self.ckpt, 1_000)
        _save(self.ckpt, 2_000)
        self.assertEqual(_checkpoint_step(self.ckpt), 2_000)

    def test_regression_is_refused(self):
        """THE incident: stale state written over newer state."""
        _save(self.ckpt, 5_000_000)
        with self.assertRaises(RuntimeError) as cm:
            _save(self.ckpt, 1_000_000)
        self.assertIn("refusing to overwrite", str(cm.exception))
        self.assertEqual(_checkpoint_step(self.ckpt), 5_000_000,
                         "a refused write must leave the good checkpoint intact")

    def test_equal_step_is_allowed(self):
        """Resuming and re-saving the same step is normal, not a regression."""
        _save(self.ckpt, 3_000)
        _save(self.ckpt, 3_000)
        self.assertEqual(_checkpoint_step(self.ckpt), 3_000)

    def test_regression_override_works(self):
        _save(self.ckpt, 5_000)
        os.environ["OGRL_ALLOW_CHECKPOINT_REGRESSION"] = "1"
        try:
            _save(self.ckpt, 10)
            self.assertEqual(_checkpoint_step(self.ckpt), 10)
        finally:
            os.environ.pop("OGRL_ALLOW_CHECKPOINT_REGRESSION", None)

    def test_archive_snapshots_are_kept(self):
        for i in range(1, 7):
            _save(self.ckpt, i * 1_000, archive_every=2)
        snaps = sorted((self.dir / "archive").glob("run_*.pt"))
        self.assertEqual(len(snaps), 3, f"expected 3 snapshots, got {[s.name for s in snaps]}")
        self.assertTrue(snaps[-1].name.endswith("000000006000.pt"))

    def test_archive_is_immutable_history(self):
        """Snapshots must not be rewritten by later saves."""
        _save(self.ckpt, 2_000, archive_every=1)
        snap = next((self.dir / "archive").glob("run_*.pt"))
        before = snap.read_bytes()
        _save(self.ckpt, 9_000, archive_every=1)
        self.assertEqual(snap.read_bytes(), before)

    def test_unreadable_existing_file_does_not_block(self):
        """A corrupt checkpoint must not wedge training forever."""
        self.ckpt.write_bytes(b"not a torch file")
        _save(self.ckpt, 1_000)
        self.assertEqual(_checkpoint_step(self.ckpt), 1_000)


class SyncSafety(unittest.TestCase):
    """sync_artifacts.sh must never mirror every checkpoint back from the trainer."""

    def test_sync_does_not_glob_all_checkpoints(self):
        sh = (HERE.parent / "remote" / "sync_artifacts.sh").read_text()
        self.assertNotIn('checkpoints/*.pt"', sh,
                         "sync must not pull every *.pt -- that is the 2026-09-05 incident")
        self.assertIn("$RUN_ID.pt", sh, "sync must pull only the named run's checkpoint")

    def test_sync_refuses_without_run_id(self):
        sh = (HERE.parent / "remote" / "sync_artifacts.sh").read_text()
        self.assertIn("refusing to mirror every", sh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
