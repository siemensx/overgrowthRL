from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import throughput_sweep  # noqa: E402


class _CleanStopProcess:
    returncode = None

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        control = json.loads((self.run_dir / "control.json").read_text(encoding="utf-8"))
        if control["command"] != "stop":
            raise AssertionError("trainer was not asked to stop cleanly")
        self.returncode = 0
        return 0

    def terminate(self):
        raise AssertionError("clean control stop should not terminate the trainer")


class _EscalatedStopProcess:
    pid = 4321
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("trainer", timeout)
        return self.returncode

    def terminate(self):
        raise AssertionError("fallback must target the process tree, not just its parent")


class ThroughputSweepStopTests(unittest.TestCase):
    def test_pre_ready_failure_finalizes_only_the_owned_running_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            manifest_path = run_dir / "run.json"
            manifest_path.write_text(json.dumps({"status": "running", "run_id": "failed-start"}))

            throughput_sweep.mark_pre_ready_failure(run_dir, ready_at=None, returncode=1)

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["run_id"], "failed-start")
            self.assertEqual(manifest["failure"]["stage"], "before_rl_ready")
            self.assertEqual(manifest["failure"]["trainer_returncode"], 1)
            self.assertIsNotNone(manifest["ended_at"])
            self.assertFalse(manifest_path.with_suffix(".json.tmp").exists())

    def test_pre_ready_finalizer_preserves_terminal_or_ready_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            manifest_path = run_dir / "run.json"
            for manifest, ready_at in (
                ({"status": "completed", "run_id": "finished"}, None),
                ({"status": "running", "run_id": "ready"}, 123.0),
            ):
                manifest_path.write_text(json.dumps(manifest))
                before = manifest_path.read_bytes()
                throughput_sweep.mark_pre_ready_failure(run_dir, ready_at, returncode=1)
                self.assertEqual(manifest_path.read_bytes(), before)

    def test_per_update_metrics_exclude_warmup_straddling_row(self):
        rows = [
            {"perf": {"steps_per_second_cycle": 999.0, "pool_misses": 9}},
            {"perf": {"steps_per_second_cycle": 501.0, "pool_misses": 0}},
            {"perf": {"steps_per_second_cycle": 503.0, "pool_misses": 1}},
        ]
        measured = throughput_sweep._complete_measurement_rows(rows)
        self.assertEqual(throughput_sweep._metric_column(measured, "steps_per_second_cycle"), [501.0, 503.0])
        self.assertEqual(throughput_sweep._metric_column(measured, "pool_misses"), [0, 1])

    def test_no_complete_interval_yields_no_per_update_metrics(self):
        self.assertEqual(throughput_sweep._complete_measurement_rows([]), [])
        self.assertEqual(throughput_sweep._complete_measurement_rows([{"perf": {"x": 1}}]), [])

    def test_windows_process_preflight_is_read_only(self):
        result = SimpleNamespace(
            stdout='"Overgrowth.exe","1234","Console","1","10,000 K"\n'
        )
        with mock.patch.object(throughput_sweep.os, "name", "nt"):
            with mock.patch.object(throughput_sweep.subprocess, "run", return_value=result) as run:
                existing = throughput_sweep.find_existing_engines()
        self.assertEqual(existing, ["PID 1234"])
        self.assertEqual(
            run.call_args.args[0],
            ["tasklist.exe", "/FI", "IMAGENAME eq Overgrowth.exe", "/FO", "CSV", "/NH"],
        )

    def test_busy_host_is_refused_without_killing_anything(self):
        with mock.patch.object(throughput_sweep, "find_existing_engines", return_value=["PID 1234"]):
            with self.assertRaisesRegex(RuntimeError, "refusing throughput sweep"):
                throughput_sweep.refuse_busy_engine_host()

    def test_stop_request_is_atomic_parseable_utf8_without_bom(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            throughput_sweep.request_stop(run_dir)
            data = (run_dir / "control.json").read_bytes()
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(json.loads(data.decode("utf-8"))["command"], "stop")
            self.assertFalse((run_dir / "control.json.tmp").exists())

    def test_clean_update_boundary_stop_is_not_escalated(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, escalated = throughput_sweep.stop_trainer(
                _CleanStopProcess(Path(temp)), Path(temp), 1.0
            )
        self.assertTrue(clean)
        self.assertFalse(escalated)

    def test_timeout_uses_bounded_termination_fallback(self):
        proc = _EscalatedStopProcess()

        def kill_owned_tree(*args, **kwargs):
            proc.returncode = 1
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(throughput_sweep, "request_stop", side_effect=OSError("read-only")):
                with contextlib.redirect_stderr(io.StringIO()):
                    if throughput_sweep.os.name == "nt":
                        with mock.patch.object(throughput_sweep.subprocess, "run", side_effect=kill_owned_tree) as run:
                            clean, escalated = throughput_sweep.stop_trainer(proc, Path(temp), 0.01)
                        self.assertEqual(run.call_args.args[0], ["taskkill.exe", "/PID", "4321", "/T", "/F"])
                    else:
                        with mock.patch.object(throughput_sweep.os, "killpg", side_effect=kill_owned_tree) as killpg:
                            clean, escalated = throughput_sweep.stop_trainer(proc, Path(temp), 0.01)
                        self.assertEqual(killpg.call_args.args, (4321, throughput_sweep.signal.SIGTERM))
        self.assertFalse(clean)
        self.assertTrue(escalated)

    def test_existing_run_artifact_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "Tools" / "rl" / "runs" / "sweep_tag_n2k1"
            run_dir.mkdir(parents=True)
            marker = run_dir / "keep.txt"
            marker.write_text("prior evidence")
            args = SimpleNamespace(repo_root=temp)
            with self.assertRaises(FileExistsError):
                throughput_sweep.run_point(args, 2, 1, "tag")
            self.assertEqual(marker.read_text(), "prior evidence")

    def test_summary_writer_atomically_publishes_results(self):
        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp) / "summary.json"
            payload = [{"valid_point": True}]
            throughput_sweep.write_results(result, payload)
            self.assertEqual(json.loads(result.read_text()), payload)
            self.assertFalse(result.with_suffix(".json.tmp").exists())

    def test_engine_character_logs_are_counted_archived_and_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            run_id = "sweep_test_n2k2"
            root = repo / ".rl_write_dirs"
            root.mkdir()
            for suffix, count in (("0", 4), ("1", 3), ("s0", 4), ("s1", 2)):
                write_dir = root / f"env-ogrl_{run_id}{suffix}-test"
                write_dir.mkdir()
                map_name = "map-a.xml" if suffix in {"0", "s0"} else "map-b.xml"
                (write_dir / "logfile.txt").write_text(
                    "Caching skeleton info\n" * count
                    + f'Chose "C:/Levels/arenas/{map_name}" (valid)\n'
                    + "Telling characters 0 and 1 to notice each other.\n"
                )
                write_dir.with_name(write_dir.name + ".log").write_text("engine stdout\n")
            unrelated = root / "env-ogrl_another_run0-test"
            unrelated.mkdir()

            run_dir = repo / "Tools" / "rl" / "runs" / run_id
            run_dir.mkdir(parents=True)
            evidence = throughput_sweep.collect_engine_character_logs(
                repo, run_id, 2, 2, ["arenas/map-a.xml", "arenas/map-b.xml"], run_dir,
                expected_opponents=1,
            )

            self.assertTrue(evidence["valid"])
            self.assertEqual(evidence["expected_engine_count"], 4)
            self.assertEqual(evidence["expected_opponents_from_restored_curriculum"], 1)
            self.assertEqual(
                evidence["skeleton_cache_marker_histogram_diagnostic_only"],
                {"4": 2, "3": 1, "2": 1},
            )
            self.assertTrue(all(engine["valid"] for engine in evidence["engines"]))
            self.assertTrue(all(
                engine["observed_character_ids_from_notice_logs"] == [0, 1]
                for engine in evidence["engines"]
            ))
            self.assertEqual(len(list((run_dir / "engine_logs").glob("*.logfile.txt"))), 4)
            self.assertFalse(any(root.glob(f"env-ogrl_{run_id}*")))
            self.assertTrue(unrelated.exists())

    def test_character_gate_rejects_missing_or_empty_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            run_id = "sweep_invalid_n1k1"
            root = repo / ".rl_write_dirs"
            root.mkdir()
            write_dir = root / f"env-ogrl_{run_id}0-test"
            write_dir.mkdir()
            (write_dir / "logfile.txt").write_text("Caching skeleton info\n")
            run_dir = repo / "Tools" / "rl" / "runs" / run_id
            run_dir.mkdir(parents=True)

            evidence = throughput_sweep.collect_engine_character_logs(
                repo, run_id, 1, 1, ["arenas/map-a.xml", "arenas/map-b.xml"], run_dir,
                expected_opponents=1,
            )

            self.assertFalse(evidence["valid"])
            self.assertEqual(evidence["observed_engine_count"], 1)
            self.assertEqual(evidence["engines"][0]["skeleton_cache_markers_diagnostic_only"], 1)
            self.assertEqual(evidence["engines"][0]["observed_character_ids_from_notice_logs"], [])

    def test_character_gate_rejects_wrong_actor_count_and_map(self):
        cases = (
            ('Chose "C:/levels/map-a.xml" (valid)\n',
             "Telling characters 0 and 1 to notice each other.\n", True),
            ('Chose "C:/levels/map-a.xml" (valid)\n',
             "Telling characters 0 and 1 to notice each other.\n"
             "Telling characters 0 and 2 to notice each other.\n", False),
            ('Chose "C:/levels/map-b.xml" (valid)\n',
             "Telling characters 0 and 1 to notice each other.\n", False),
        )
        for level_line, notices, expected_valid in cases:
            with self.subTest(expected_valid=expected_valid, notices=notices):
                evidence = throughput_sweep._character_scenario_evidence(
                    level_line + notices, expected_opponents=1,
                    expected_level="arenas/map-a.xml",
                )
                self.assertEqual(evidence["valid"], expected_valid)

    def test_character_gate_accepts_sampled_width_below_curriculum_max(self):
        evidence = throughput_sweep._character_scenario_evidence(
            'Chose "C:/levels/map-a.xml" (valid)\n'
            "Telling characters 0 and 1 to notice each other.\n",
            expected_opponents=3,
            expected_level="arenas/map-a.xml",
        )
        self.assertTrue(evidence["valid"])
        self.assertEqual(evidence["expected_character_ids"], [0, 1, 2, 3])
        self.assertEqual(evidence["observed_opponents_from_notice_logs"], 1)
        self.assertTrue(evidence["scenario_width_within_curriculum"])
        self.assertTrue(evidence["complete_observed_pair_set"])

    def test_character_gate_rejects_incomplete_or_over_limit_sampled_width(self):
        incomplete = throughput_sweep._character_scenario_evidence(
            'Chose "C:/levels/map-a.xml" (valid)\n'
            "Telling characters 0 and 1 to notice each other.\n"
            "Telling characters 0 and 2 to notice each other.\n",
            expected_opponents=3,
            expected_level="arenas/map-a.xml",
        )
        over_limit = throughput_sweep._character_scenario_evidence(
            'Chose "C:/levels/map-a.xml" (valid)\n'
            "Telling characters 0 and 1 to notice each other.\n"
            "Telling characters 0 and 2 to notice each other.\n"
            "Telling characters 0 and 3 to notice each other.\n"
            "Telling characters 0 and 4 to notice each other.\n",
            expected_opponents=3,
            expected_level="arenas/map-a.xml",
        )
        self.assertFalse(incomplete["valid"])
        self.assertFalse(over_limit["valid"])

    def test_restored_opponent_count_comes_from_trainer_output(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "trainer.log"
            log.write_text(
                "restored curriculum: d_max=1.00 opponents_max=2\n"
                "restored curriculum: d_max=1.00 opponents_max=3\n",
                encoding="utf-8",
            )
            self.assertEqual(throughput_sweep.restored_opponents_from_log(log), 3)
            log.write_text("training started\n", encoding="utf-8")
            self.assertIsNone(throughput_sweep.restored_opponents_from_log(log))


class ThroughputSweepProvenanceTests(unittest.TestCase):
    def test_repeated_grid_conditions_get_distinct_artifact_tags(self):
        overrides = {
            "OGRL_ENGINE_PRIORITY": "above",
            "OGRL_ENGINE_AFFINITY": "0xFFF",
        }
        first = throughput_sweep.grid_point_tag("priority_abba", 1, overrides)
        second = throughput_sweep.grid_point_tag("priority_abba", 2, overrides)

        self.assertNotEqual(first, second)
        self.assertIn("p01", first)
        self.assertIn("p02", second)
        self.assertTrue(first.endswith("above_0xFFF"))

    def test_file_fingerprint_records_resolved_path_size_and_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "engine.bin"
            payload = b"engine-build-fingerprint"
            target.write_bytes(payload)

            fingerprint = throughput_sweep.file_fingerprint(target)

            self.assertEqual(fingerprint["path"], str(target.resolve()))
            self.assertEqual(fingerprint["bytes"], len(payload))
            self.assertEqual(fingerprint["sha256"], hashlib.sha256(payload).hexdigest())

    def test_relative_resume_checkpoint_resolves_from_repo_root(self):
        repo = Path("/repo/checkout")
        self.assertEqual(
            throughput_sweep.resolve_input_checkpoint(repo, "weights/run.pt"),
            repo / "weights" / "run.pt",
        )

    def test_experiment_identity_captures_engine_checkpoint_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            engine = repo / "engine.exe"
            checkpoint = repo / "run.pt"
            engine.write_bytes(b"engine")
            checkpoint.write_bytes(b"read-only-checkpoint")
            args = SimpleNamespace(
                resume_from="run.pt",
                levels="arenas/a.xml,arenas/b.xml",
                warmup=60.0,
                measure=300.0,
                collection_threads=2,
                update_threads=1,
                interop_threads=1,
                hard_reset_every=20,
                n_steps=512,
                n_epochs=1,
                minibatch_size=128,
            )
            with mock.patch.object(throughput_sweep, "engine_binary", return_value=engine):
                with mock.patch.object(throughput_sweep, "aux_data", return_value=repo):
                    identity = throughput_sweep.experiment_identity(repo, args)

            self.assertEqual(identity["engine"]["sha256"], hashlib.sha256(b"engine").hexdigest())
            self.assertEqual(identity["resume_checkpoint"]["bytes"], len(b"read-only-checkpoint"))
            self.assertEqual(identity["protocol"]["levels"], ["arenas/a.xml", "arenas/b.xml"])
            self.assertEqual(identity["protocol"]["update_threads"], 1)
            self.assertEqual(identity["assets_root"], str(repo))


if __name__ == "__main__":
    unittest.main()
