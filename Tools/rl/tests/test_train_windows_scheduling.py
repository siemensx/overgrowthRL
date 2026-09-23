#!/usr/bin/env python3
"""Regression tests for pointer-sized Windows process scheduling handles."""
from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppo"))
from train_vec import _apply_windows_process_scheduling  # noqa: E402


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeKernel32:
    def __init__(self):
        self.priority_calls = []
        self.affinity = 0
        self.handle = ctypes.c_void_p(-1).value
        self.GetCurrentProcess = FakeFunction(lambda: self.handle)
        self.SetPriorityClass = FakeFunction(self._set_priority)
        self.SetProcessAffinityMask = FakeFunction(self._set_affinity)
        self.GetProcessAffinityMask = FakeFunction(self._get_affinity)

    def _set_priority(self, handle, priority_class):
        self.priority_calls.append((handle, priority_class))
        return 1

    def _set_affinity(self, handle, requested):
        self.affinity = requested.value
        return 1

    def _get_affinity(self, handle, applied_ptr, system_ptr):
        ctypes.cast(applied_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = self.affinity
        ctypes.cast(system_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = 0x3FFF
        return 1


class WindowsSchedulingSignatures(unittest.TestCase):
    def test_priority_and_affinity_use_pointer_sized_handle(self):
        api = FakeKernel32()

        applied = _apply_windows_process_scheduling("above", "0xFFF", api)

        self.assertEqual(applied, 0xFFF)
        self.assertEqual(api.priority_calls, [(api.handle, 0x8000)])
        self.assertIs(api.GetCurrentProcess.restype, ctypes.c_void_p)
        self.assertIs(api.SetPriorityClass.argtypes[0], ctypes.c_void_p)
        self.assertIs(api.SetProcessAffinityMask.argtypes[0], ctypes.c_void_p)
        self.assertEqual(api.SetProcessAffinityMask.argtypes[1], ctypes.c_size_t)

    def test_inherit_skips_priority_class_change(self):
        api = FakeKernel32()

        applied = _apply_windows_process_scheduling("inherit", None, api)

        self.assertIsNone(applied)
        self.assertEqual(api.priority_calls, [])


if __name__ == "__main__":
    unittest.main()
