#!/usr/bin/env python3
"""OGRL-20261006-002: a checkpoint held open by a reader must not crash training."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
sys.path.insert(0, str(HERE.parent))

import train  # noqa: E402


def test_retries_then_succeeds():
    d = Path(tempfile.mkdtemp())
    tmp, dst = d / "a.pt.tmp", d / "a.pt"
    tmp.write_text("new"); dst.write_text("old")
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        real(a, b)
    train.os.replace = flaky
    try:
        assert train._replace_with_retry(tmp, dst, attempts=5, delay=0.0)
    finally:
        train.os.replace = real
    assert dst.read_text() == "new" and calls["n"] == 3


def test_gives_up_without_raising():
    d = Path(tempfile.mkdtemp())
    tmp, dst = d / "a.pt.tmp", d / "a.pt"
    tmp.write_text("new"); dst.write_text("old")
    real = os.replace

    def locked(a, b):
        raise PermissionError("[WinError 5] Access is denied")
    train.os.replace = locked
    try:
        assert train._replace_with_retry(tmp, dst, attempts=3, delay=0.0) is False
    finally:
        train.os.replace = real
    assert dst.read_text() == "old" and tmp.exists()      # old checkpoint intact, new state kept


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} tests passed")
