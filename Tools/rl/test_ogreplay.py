#!/usr/bin/env python3
"""stdlib regression tests for the OGRL-20260820-044 replay container."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ogreplay import (
    Float32Bits,
    Float64Bits,
    ReplayReader,
    ReplayWriter,
    canonical_pack,
    canonical_unpack,
    write_policy_trace,
)
from tape import TapeRecorder


class ReplayCodecTests(unittest.TestCase):
    def test_scalars_preserve_float_bits(self):
        values = [
            None, True, False, -7, 0, 42, (1 << 63) + 7, (1 << 64) - 1,
            Float32Bits(0x7FC01234), Float32Bits.from_float(-3.25),
            Float64Bits(0x7FF8000000001234), Float64Bits.from_float(-9.5),
            float("inf"), float("-inf"),
        ]
        for value in values:
            decoded, used = canonical_unpack(canonical_pack(value))
            self.assertEqual(used, len(canonical_pack(value)))
            if isinstance(value, (Float32Bits, Float64Bits)):
                self.assertEqual(decoded, value)
            elif isinstance(value, float):
                self.assertEqual(decoded, value)
            else:
                self.assertEqual(decoded, value)

    def test_mapping_order_is_stable(self):
        left = {"b": [1, 2], "a": {"z": 4.5, "x": True}}
        right = {"a": {"x": True, "z": 4.5}, "b": [1, 2]}
        self.assertEqual(canonical_pack(left), canonical_pack(right))

    def test_round_trip_chunks_and_footer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.ogreplay"
            with ReplayWriter(path, {"run_id": "test", "authoritative_complete": False}) as writer:
                writer.add_chunk("RESET", 0, [{"seed": 19, "zero_action": True}])
                writer.add_chunk("TICK", 0, [{"tick": i, "value": Float32Bits.from_float(i / 3)} for i in range(8)])
                writer.add_chunk("DECISION", 0, [{"decision": 0, "action": [0.0, 1.0]}])
                writer.finalize(outcome="won", terminal={"tick": 7})
            reader = ReplayReader(path)
            self.assertTrue(reader.complete)
            self.assertEqual(reader.manifest["run_id"], "test")
            self.assertEqual(len(reader.records("TICK")), 8)
            self.assertEqual(reader.summary()["status"], "RECORDED STATE PLAYBACK")
            self.assertEqual(reader.summary()["outcome"], "won")

    def test_corruption_and_partial_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.ogreplay"
            with ReplayWriter(path, {"run_id": "test"}) as writer:
                writer.add_chunk("TICK", 0, [{"tick": 0, "x": 1.0}])
                writer.add_chunk("TICK", 1, [{"tick": 1, "x": 2.0}])
                writer.finalize()

            corrupted = Path(tmp) / "corrupt.ogreplay"
            data = bytearray(path.read_bytes())
            data[-1] ^= 0x01
            corrupted.write_bytes(data)
            corrupt_reader = ReplayReader(corrupted)
            self.assertFalse(corrupt_reader.complete)

            partial = Path(tmp) / "episode.ogreplay.partial"
            full = path.read_bytes()
            # Keep the header and first full chunk, then cut in the second
            # chunk. The reader must expose the first complete chunk and mark
            # the artifact incomplete rather than pretending it is whole.
            partial.write_bytes(full[: len(full) // 2])
            recovered = ReplayReader(partial)
            self.assertTrue(len(recovered.chunks) >= 1)
            self.assertTrue(recovered.incomplete)

    def test_policy_trace_is_explicitly_non_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.ogreplay"
            write_policy_trace(path, meta={"run_id": "run1", "seed": 5, "outcome": "lost"}, decisions=[{"act": [0] }])
            summary = ReplayReader(path).summary()
            self.assertFalse(summary["authoritative_complete"])
            self.assertEqual(summary["status"], "RECORDED STATE PLAYBACK")

    def test_legacy_recorder_emits_binary_companion(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TapeRecorder(Path(tmp), window_updates=1, keep_recent=10)
            recorder.record_decision(0, {
                "t": 0.0, "self": {"pos": [0.0, 0.0, 0.0]}, "ents": [],
                "act": [0.0] * 8, "rew": {"time": -0.01}, "d": 0.5,
            })
            recorder.episode_ended(0, 0, 11, "won", 1.0, 0.5, True)
            binary = list((Path(tmp) / "tapes").glob("*.ogreplay"))
            self.assertEqual(len(binary), 1)
            self.assertEqual(ReplayReader(binary[0]).summary()["status"], "RECORDED STATE PLAYBACK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
