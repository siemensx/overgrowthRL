#!/usr/bin/env python3
"""Tests that failed engine startup retains diagnostics without write dirs."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from env import OvergrowthEnv  # noqa: E402


class FailedEngineShell:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.shm_name = "/ogrl_test_startup_failure0"
        self._write_dir = repo_root / ".rl_write_dirs" / "env-test"
        self._write_dir.mkdir(parents=True)
        self._log_path = self._write_dir.parent / f"{self._write_dir.name}.log"
        self._log_file = self._log_path.open("wb")
        self._log_file.write(b"engine startup diagnostic\n")
        self._log_file.flush()

    def close(self):
        self._log_file.close()
        shutil.rmtree(self._write_dir)
        self._log_path.unlink()


class EngineStartupFailureDiagnostics(unittest.TestCase):
    def test_preserves_only_log_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            env = FailedEngineShell(repo_root)

            with self.assertRaisesRegex(RuntimeError, "engine startup log preserved at") as error:
                OvergrowthEnv._fail_launch(env, "engine exited early")

            preserved = Path(str(error.exception).split("engine startup log preserved at ", 1)[1])
            self.assertEqual(preserved.read_bytes(), b"engine startup diagnostic\n")
            self.assertFalse(env._write_dir.exists())
            self.assertFalse(env._log_path.exists())


if __name__ == "__main__":
    unittest.main()
