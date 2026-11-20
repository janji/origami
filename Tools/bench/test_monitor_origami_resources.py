"""Model-free gate, identity, deadline and evidence tests for the host watcher."""
import copy
import datetime
import errno
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import monitor_origami_resources as watch


NORMAL_THERMAL = "No thermal warning level has been recorded\nNo performance warning level has been recorded\n"


def arguments(**changes):
    values = dict(comparator_pid=12345, comparator_executable="/test/python", comparator_start_abstime=123,
                  comparator_script="/test/compare.py", comparator_output="/test/result",
                  model_executable=["/test/origami"], model_config=["/test/config.json"], disk_path="/test",
                  memory_cap_bytes=14_000_000_000, disk_floor_bytes=20_000_000_000, paging_policy="abort",
                  interval_seconds=15, model_discovery_grace_seconds=30, cleanup_grace_seconds=3,
                  end_utc=datetime.datetime.now(watch.UTC) + datetime.timedelta(seconds=300))
    values.update(changes)
    return types.SimpleNamespace(**values)


def baseline():
    return {"utc": "2026-11-07T15:00:00+00:00", "counters": {
        "pageouts": 100, "swapouts": 0, "swap_used_bytes": 0}}


def sample():
    return {"host": {**baseline(), "raw": {key: {"returncode": 0, "stdout": text} for key, text in (
        ("vm_stat", "Pageouts: 100."), ("swap_usage", "used = 0M"), ("thermal", NORMAL_THERMAL))}},
        "pressure_level": 1, "disk_free_bytes": 30_000_000_000,
        "models": [{"pid": 23456, "measured": True, "peak_footprint_bytes": 10_000_000_000}]}


class FakeClock:
    def __init__(self):
        self.value = 0
        self.base = datetime.datetime.now(watch.UTC)

    def clock(self):
        return self.value

    def now(self):
        return self.base + datetime.timedelta(seconds=self.value)

    def sleep(self, duration):
        self.value += duration


class FakeProbe:
    def __init__(self, clock, samples=2, observation=None, complete=True):
        self.clock, self.remaining = clock, samples
        self.observation = observation or sample()
        self.complete, self.signals, self.starts = complete, 0, []

    def comparator(self, check_arguments=True):
        return {"state": "gone" if self.signals or self.remaining == 0 else "verified"}

    def sample(self):
        self.starts.append(self.clock.value)
        self.clock.sleep(2)
        self.remaining -= 1
        return copy.deepcopy(self.observation)

    def interrupt(self):
        self.signals += 1
        return {"sent": True, "identity": {"state": "verified"}}

    def terminal_outcome(self):
        return {"complete": self.complete}


class ResourceWatcherTests(unittest.TestCase):
    def run_monitor(self, probe, clock, args=None, emitter=None):
        rows = []
        result = watch.monitor(args or arguments(end_utc=clock.base + datetime.timedelta(seconds=300)),
                               baseline(), probe, emitter or (lambda row, **_: rows.append(row)),
                               clock.clock, clock.now, clock.sleep)
        return result, rows

    def test_pageouts_policy_does_not_allow_swap_growth(self):
        row = sample()
        row["host"]["counters"]["pageouts"] += 1
        self.assertEqual(watch.assess(row, baseline(), arguments())["stop_reasons"], ["increased_pageouts"])
        result = watch.assess(row, baseline(), arguments(paging_policy="mark-ineligible"))
        self.assertEqual(result["stop_reasons"], [])
        self.assertEqual(result["paging_classification"], "increase")
        self.assertEqual(result["performance_ineligible_reasons"], ["increased_pageouts"])
        for counter in ("swapouts", "swap_used_bytes"):
            with self.subTest(counter=counter):
                row["host"]["counters"][counter] = 1
                self.assertIn("increased_" + counter, watch.assess(
                    row, baseline(), arguments(paging_policy="mark-ineligible"))["stop_reasons"])

    def test_missing_reset_and_noninteger_counters_fail_closed(self):
        for value in (None, True, -1, 99):
            with self.subTest(value=value):
                row = sample()
                row["host"]["counters"]["pageouts"] = value
                result = watch.assess(row, baseline(), arguments())
                self.assertTrue(result["stop_reasons"])
                self.assertEqual(result["paging_classification"], "unknown")

    def test_independent_resource_gates(self):
        mutations = (
            (lambda r: r.update(pressure_level=2), "pressure_not_normal"),
            (lambda r: r["host"]["raw"]["thermal"].update(stdout="Thermal warning level: 1"), "thermal_not_verified_normal"),
            (lambda r: r.update(disk_free_bytes=19_999_999_999), "disk_floor_or_missing_measurement"),
            (lambda r: r["models"][0].update(peak_footprint_bytes=14_000_000_001), "model_memory_cap"),
            (lambda r: r["models"][0].update(measured=False), "missing_model_memory"),
            (lambda r: r["host"]["raw"]["vm_stat"].update(returncode=1), "missing_vm_stat"),
        )
        for mutate, reason in mutations:
            with self.subTest(reason=reason):
                row = sample()
                mutate(row)
                self.assertIn(reason, watch.assess(row, baseline(), arguments())["stop_reasons"])

    def test_cadence_and_retirement_are_not_performance_qualification(self):
        clock = FakeClock()
        probe = FakeProbe(clock)
        status, rows = self.run_monitor(probe, clock)
        self.assertEqual(status, 0)
        self.assertEqual(probe.starts, [0, 15])
        self.assertEqual(probe.signals, 0)
        self.assertTrue(rows[-1]["observed_stop_gates_clear"])
        self.assertIsNone(rows[-1]["performance_eligible"])

    def test_retirement_without_model_or_terminal_receipt_is_incomplete(self):
        for count, observation, complete in ((0, sample(), True), (1, {**sample(), "models": []}, True),
                                               (1, sample(), False)):
            with self.subTest(count=count, complete=complete):
                clock = FakeClock()
                status, rows = self.run_monitor(FakeProbe(clock, count, observation, complete), clock)
                self.assertEqual(status, 2)
                self.assertFalse(rows[-1]["observed_stop_gates_clear"])

    def test_stop_gate_signals_comparator_once_then_retires(self):
        clock = FakeClock()
        probe = FakeProbe(clock, observation={**sample(), "pressure_level": 2})
        status, rows = self.run_monitor(probe, clock)
        self.assertEqual(status, 1)
        self.assertEqual(probe.signals, 1)
        self.assertEqual(rows[-1]["final_comparator_identity"]["state"], "gone")

    def test_missing_model_stops_after_bounded_grace(self):
        clock = FakeClock()
        probe = FakeProbe(clock, samples=100, observation={**sample(), "models": []})
        _, rows = self.run_monitor(probe, clock)
        self.assertEqual(probe.starts, [0, 15, 30])
        self.assertIn("model_not_observed_within_grace", rows[-1]["stop_reasons"])

    def test_deadline_reserves_stop_time_and_uses_local_monotonic(self):
        clock = FakeClock()
        probe = FakeProbe(clock, samples=100)
        args = arguments(end_utc=clock.base + datetime.timedelta(seconds=40))
        _, rows = self.run_monitor(probe, clock, args)
        self.assertEqual(probe.starts, [0, 15])
        self.assertEqual(clock.value, 25)
        self.assertEqual(rows[-1]["stop_reasons"], ["absolute_deadline"])

    def test_whole_observation_alarm_triggers_owned_stop(self):
        clock = FakeClock()
        probe = FakeProbe(clock)
        original = watch.WallDeadline
        with patch.object(probe, "sample", side_effect=lambda: time.sleep(.2)), \
                patch.object(watch, "WallDeadline", side_effect=lambda _: original(.02)):
            _, rows = self.run_monitor(probe, clock)
        self.assertEqual(probe.signals, 1)
        self.assertEqual(rows[-1]["stop_reasons"], ["missing_measurement"])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_evidence_failure_still_requests_stop_and_reports_terminal(self):
        for fail_kind in ("header", "sample"):
            with self.subTest(fail_kind=fail_kind):
                clock, rows = FakeClock(), []
                probe = FakeProbe(clock)
                def emit(row, **_):
                    if row["kind"] == fail_kind:
                        raise OSError("synthetic disk error")
                    rows.append(row)
                status, _ = self.run_monitor(probe, clock, emitter=emit)
                self.assertEqual(status, 1)
                self.assertEqual(probe.signals, 1)
                self.assertEqual(rows[-1]["kind"], "final")
        clock = FakeClock()
        probe = FakeProbe(clock, observation={**sample(), "pressure_level": 2})
        def unavailable(*_, **__):
            raise OSError("synthetic full disk")
        with patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
            self.run_monitor(probe, clock, emitter=unavailable)
        self.assertEqual(probe.signals, 1)
        self.assertEqual(json.loads(stderr.getvalue())["kind"], "terminal_write_failed")

    def test_jsonl_is_exclusive_private_bounded_and_reserves_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            output = watch.JSONL(path, maximum=17000)
            try:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                output.emit({"kind": "header"})
                with self.assertRaises(ValueError):
                    output.emit({"payload": "x" * 1000})
                output.emit({"kind": "final"}, terminal=True)
                with self.assertRaises(FileExistsError):
                    watch.JSONL(path)
            finally:
                output.file.close()
            self.assertEqual([json.loads(x)["kind"] for x in path.read_text().splitlines()], ["header", "final"])

    def test_output_open_failure_stops_verified_comparator_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            stamp = datetime.datetime.now(watch.UTC)
            before = Path(tmp) / "baseline.json"
            before.write_text(json.dumps({**baseline(), "utc": (stamp - datetime.timedelta(seconds=1)).isoformat()}))
            target = Path(tmp) / "existing.jsonl"
            target.write_text("keep original evidence\n")
            probe = FakeProbe(FakeClock())
            argv = ["--comparator-pid", "12345", "--comparator-executable", "/test/python",
                    "--comparator-start-abstime", "123", "--comparator-started-utc", stamp.isoformat(),
                    "--comparator-script", "/test/compare.py", "--comparator-output", "/test/result",
                    "--model-executable", "/test/origami", "--model-config", "/test/config.json",
                    "--baseline", str(before), "--output", str(target), "--disk-path", tmp,
                    "--end-utc", (stamp + datetime.timedelta(seconds=300)).isoformat()]
            with patch.object(watch, "SystemProbe", return_value=probe), \
                    patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(watch.main(argv), 1)
            self.assertEqual(probe.signals, 1)
            self.assertEqual(target.read_text(), "keep original evidence\n")
            self.assertEqual(json.loads(stderr.getvalue())["stop_reasons"], ["header_write_failure"])

    def test_identity_rechecked_after_argument_probe_before_signal(self):
        probe = watch.SystemProbe(arguments())
        good = {"process_start_abstime": 123, "verified_executable": "/test/python"}
        bad = {**good, "process_start_abstime": 124}
        with patch.object(watch, "kernel_process_memory", side_effect=[good, bad]), \
                patch.object(watch, "process_arguments", return_value=["/test/compare.py", "--output", "/test/result"]), \
                patch.object(watch.os, "kill") as kill:
            result = probe.interrupt()
        self.assertFalse(result["sent"])
        self.assertEqual(result["identity"]["state"], "identity_changed")
        kill.assert_not_called()

    def test_wrong_script_gone_and_unavailable_identity_never_signaled(self):
        good = {"process_start_abstime": 123, "verified_executable": "/test/python"}
        cases = (good, OSError(errno.ESRCH, "gone"), OSError(errno.EPERM, "unavailable"))
        for identity in cases:
            with self.subTest(identity=identity), \
                    patch.object(watch, "kernel_process_memory", side_effect=identity if isinstance(identity, Exception) else None,
                                 return_value=identity), \
                    patch.object(watch, "process_arguments", return_value=["/wrong/script", "--output", "/test/result"]), \
                    patch.object(watch.os, "kill") as kill:
                self.assertFalse(watch.SystemProbe(arguments()).interrupt()["sent"])
                kill.assert_not_called()

    def test_matching_identity_signals_only_explicit_comparator(self):
        good = {"process_start_abstime": 123, "verified_executable": "/test/python"}
        with patch.object(watch, "kernel_process_memory", return_value=good), \
                patch.object(watch, "process_arguments", return_value=["/test/compare.py", "--output", "/test/result"]), \
                patch.object(watch.os, "kill") as kill:
            self.assertTrue(watch.SystemProbe(arguments()).interrupt()["sent"])
        kill.assert_called_once_with(12345, signal.SIGINT)

    def test_model_session_parent_config_and_respawn_are_checked(self):
        info = {"process_start_abstime": 345, "verified_executable": "/test/origami", "measured": True,
                "peak_footprint_bytes": 100}
        def command(argv, **_):
            if "sysctl" in argv[0]:
                return "1"
            return "23456 12345\n" if "-axo" in argv else "12345"
        with patch.object(watch, "host_boundary_snapshot", return_value=sample()["host"]), \
                patch.object(watch, "bounded_command", side_effect=command), \
                patch.object(watch, "executable_path", return_value="/test/origami"), \
                patch.object(watch, "kernel_process_memory", return_value=info), \
                patch.object(watch, "process_arguments", return_value=["/test/origami", "--single-model", "--config", "/test/config.json"]), \
                patch.object(watch.os, "getpgid", return_value=23456), \
                patch.object(watch.os, "getsid", return_value=23456), \
                patch.object(watch.shutil, "disk_usage", return_value=types.SimpleNamespace(free=30_000_000_000)):
            probe = watch.SystemProbe(arguments())
            self.assertEqual(probe.sample()["models"][0]["pid"], 23456)
            with patch.object(watch.os, "getsid", return_value=9), self.assertRaisesRegex(ValueError, "session-leader"):
                probe.sample()
            with patch.object(watch, "process_arguments", return_value=["--single-model", "--config", "/wrong"]), \
                    self.assertRaisesRegex(ValueError, "config differs"):
                probe.sample()
            probe.model_owners["/test/origami"] = (999, 345)
            with self.assertRaisesRegex(ValueError, "respawn"):
                probe.sample()
            probe.model_owners.clear()
            probe.children_seen[23456] = 344
            with self.assertRaisesRegex(ValueError, "PID reused"):
                probe.sample()

    def test_second_identity_query_memory_is_retained_and_gated(self):
        first = {"process_start_abstime": 345, "verified_executable": "/test/origami", "measured": True,
                 "peak_footprint_bytes": 100, "memory_observations": [{"current_footprint_bytes": 100}]}
        second = {**first, "peak_footprint_bytes": 14_000_000_001,
                  "memory_observations": [{"current_footprint_bytes": 14_000_000_001}]}
        def command(argv, **_):
            if "sysctl" in argv[0]: return "1"
            return "23456 12345\n" if "-axo" in argv else "12345"
        with patch.object(watch, "host_boundary_snapshot", return_value=sample()["host"]), \
                patch.object(watch, "bounded_command", side_effect=command), \
                patch.object(watch, "executable_path", return_value="/test/origami"), \
                patch.object(watch, "kernel_process_memory", side_effect=[first, second]), \
                patch.object(watch, "process_arguments", return_value=["/test/origami", "--single-model", "--config", "/test/config.json"]), \
                patch.object(watch.os, "getpgid", return_value=23456), \
                patch.object(watch.os, "getsid", return_value=23456), \
                patch.object(watch.shutil, "disk_usage", return_value=types.SimpleNamespace(free=30_000_000_000)):
            observed = watch.SystemProbe(arguments()).sample()
        self.assertEqual(observed["models"][0]["memory_observation_queries"], [first, second])
        self.assertEqual(observed["models"][0]["peak_footprint_bytes"], 14_000_000_001)
        self.assertIn("model_memory_cap", watch.assess(observed, baseline(), arguments())["stop_reasons"])

    def test_comparator_first_query_retained_when_second_query_fails(self):
        raw = {"process_start_abstime": 123, "current_footprint_bytes": 100}
        good = {"process_start_abstime": 123, "verified_executable": "/test/python", "memory_observations": [raw]}
        error = ValueError("memory unavailable after argument probe")
        partial = {"process_start_abstime": 123, "current_footprint_bytes": 0}
        error.memory_observations = [partial]
        with patch.object(watch, "kernel_process_memory", side_effect=[good, error]), \
                patch.object(watch, "process_arguments", return_value=["/test/compare.py", "--output", "/test/result"]), \
                patch.object(watch.os, "kill") as kill:
            result = watch.SystemProbe(arguments()).interrupt()
        self.assertFalse(result["sent"]); kill.assert_not_called()
        self.assertEqual(result["identity"]["memory_observations"], [raw, partial])

    def test_comparator_identity_changed_returns_retain_available_observations(self):
        raw = {"process_start_abstime": 123, "current_footprint_bytes": 100}
        good = {"process_start_abstime": 123, "verified_executable": "/test/python", "memory_observations": [raw]}
        changed_raw = {**raw, "process_start_abstime": 124}
        changed = {**good, "process_start_abstime": 124, "memory_observations": [changed_raw]}
        cases = (([changed], ["/test/compare.py", "--output", "/test/result"], [changed_raw]),
                 ([good], ["/wrong/script", "--output", "/test/result"], [raw]),
                 ([good, changed], ["/test/compare.py", "--output", "/test/result"], [raw, changed_raw]))
        for queries, argv, expected in cases:
            with self.subTest(expected=expected), patch.object(watch, "kernel_process_memory", side_effect=queries), \
                    patch.object(watch, "process_arguments", return_value=argv), patch.object(watch.os, "kill") as kill:
                result = watch.SystemProbe(arguments()).interrupt()
            self.assertFalse(result["sent"]); kill.assert_not_called()
            self.assertEqual(result["identity"]["state"], "identity_changed")
            self.assertEqual(result["identity"]["memory_observations"], expected)

    def test_first_query_partial_memory_then_esrch_is_not_silently_skipped(self):
        error = OSError(errno.ESRCH, "process retired during retry")
        error.memory_observations = [{"process_start_abstime": 345,
            "current_footprint_bytes": 14_000_000_001, "kernel_reported_lifetime_peak_bytes": 14_000_000_000},
            {"error": "process retired during retry"}]
        def command(argv, **_):
            if "sysctl" in argv[0]: return "1"
            return "23456 12345\n"
        with patch.object(watch, "host_boundary_snapshot", return_value=sample()["host"]), \
                patch.object(watch, "bounded_command", side_effect=command), \
                patch.object(watch, "executable_path", return_value="/test/origami"), \
                patch.object(watch, "kernel_process_memory", side_effect=error):
            with self.assertRaises(OSError) as caught: watch.SystemProbe(arguments()).sample()
        self.assertEqual(caught.exception.memory_observations[0]["current_footprint_bytes"], 14_000_000_001)
        self.assertEqual(len(caught.exception.memory_observations), 2)

    def test_memory_error_raw_attempts_survive_stop_receipt(self):
        clock = FakeClock(); probe = FakeProbe(clock)
        error = ValueError("kernel physical-footprint fields are absent")
        error.memory_observations = [{"process_start_abstime": 345, "current_footprint_bytes": 0,
                                      "kernel_reported_lifetime_peak_bytes": 100}]
        with patch.object(probe, "sample", side_effect=error):
            status, rows = self.run_monitor(probe, clock)
        observed = next(row for row in rows if row["kind"] == "sample")
        self.assertEqual(observed["memory_observations"], error.memory_observations)
        self.assertEqual(observed["stop_reasons"], ["missing_measurement"])
        self.assertEqual(status, 1); self.assertEqual(probe.signals, 1)

    def test_comparator_memory_error_keeps_evidence_without_signal(self):
        error = ValueError("kernel process birth changed during memory resampling")
        error.memory_observations = [{"process_start_abstime": 123}, {"process_start_abstime": 124}]
        with patch.object(watch, "kernel_process_memory", side_effect=error), patch.object(watch.os, "kill") as kill:
            result = watch.SystemProbe(arguments()).interrupt()
        self.assertFalse(result["sent"]); kill.assert_not_called()
        self.assertEqual(result["identity"]["memory_observations"], error.memory_observations)

    def test_small_owned_helpers_are_bounded_and_reaped(self):
        self.assertEqual(watch.bounded_command([sys.executable, "-c", "print('ok')"]), "ok\n")
        with self.assertRaises(TimeoutError):
            watch.bounded_command([sys.executable, "-c", "import time; time.sleep(60)"], timeout=.05)
        with self.assertRaisesRegex(ValueError, "output bound"):
            watch.bounded_command([sys.executable, "-c", "print('x' * 10000)"], maximum=100)

    def test_terminal_summary_requires_exact_expected_completed_arm_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = watch.SystemProbe(arguments(comparator_output=tmp))
            path = Path(tmp) / "summary.json"
            self.assertFalse(probe.terminal_outcome()["complete"])
            for rows, expected in (([{"status": "completed"}], True), ([{"status": "cancelled"}], False),
                                   ([{"status": "completed"}] * 2, False), ([], False)):
                path.write_text(json.dumps(rows))
                self.assertEqual(probe.terminal_outcome()["complete"], expected)


if __name__ == "__main__":
    unittest.main()
