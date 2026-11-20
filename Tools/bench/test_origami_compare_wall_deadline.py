#!/usr/bin/env python3
"""Model-free deadline/ownership regression checks; only tiny Python sleep children."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import origami_compare as compare
from native_compare import OwnedProcess


class WallDeadlineChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="origami-wall-deadline-check-")
        self.root = Path(self.temporary.name)
        self.binary = self.root / "origami-fixture"
        self.binary.write_text("Never execute this synthetic placeholder.\n")
        self.model = self.root / "metadata-only-model-placeholder"
        self.model.touch()
        for bundle in compare.BUNDLES:
            (self.root / bundle).mkdir()
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({"messages": [{"role": "user", "content": "fixture"}]}))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"engines": {"fixture": {"family": "deepseek", "model": str(self.model)}}}))
        self.children = []

    def tearDown(self):
        for child in self.children:
            child.stop()
            for stream in (child.p.stdout, child.p.stderr):
                self.assertTrue(stream.closed, "owned reader must close its pipe after successful stop")
        self.temporary.cleanup()

    def arguments(self, *extra):
        return ["--before", str(self.binary), "--candidate", str(self.binary),
                "--families", "deepseek", "--arms", "before,candidate,before",
                "--request", str(self.request), "--manifest", str(self.manifest),
                "--encoded-dir", str(self.root / "absent-encoding"),
                "--output", str(self.root / "output"), *extra]

    def plan(self):
        return {"binary": str(self.binary), "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                "model": str(self.model), "explicit_environment": {}, "speculation": "off",
                "memory_limit_bytes": compare.LIMIT, "mtp": {"enabled": False}}

    def owner_factory(self, *, construction_delay=0, cleanup_delay=0, cleanup_timers=None, sleep_only=False):
        children = self.children
        class FixtureOwner(OwnedProcess):
            def __init__(self, *args, **kwargs):
                if sleep_only:
                    args = ([sys.executable, "-c", "import time; time.sleep(60)"], *args[1:])
                super().__init__(*args, **kwargs)
                children.append(self)
                if construction_delay:
                    time.sleep(construction_delay)
            def stop(self):
                if self.p.poll() is None:
                    if cleanup_timers is not None:
                        cleanup_timers.append(signal.getitimer(signal.ITIMER_REAL))
                    if cleanup_delay:
                        time.sleep(cleanup_delay)
                super().stop()
        return FixtureOwner

    def execute_fixture(self, deadline, http, owner):
        directory = self.root / "arm"
        directory.mkdir()
        args = SimpleNamespace(port=0, load_deadline=10, request_deadline=10)
        with patch.object(compare, "OwnedProcess", owner), \
             patch.object(compare, "host_boundary_snapshot", return_value={"counters": {}, "raw": {}}), \
             patch.object(compare, "host_boundary_delta", return_value={}), \
             patch.object(compare, "http_json", side_effect=http), \
             patch.object(compare, "validate_status", return_value=None):
            return compare.execute(self.plan(), directory, self.request, None, args, deadline)

    def test_blocked_work_interrupts_and_stops_owned_child(self):
        def blocked(*_, **__):
            time.sleep(5)
            return {"status": "loading"}
        started = time.monotonic()
        with compare.WallDeadline(0.15) as deadline:
            receipt = self.execute_fixture(deadline, blocked, self.owner_factory())
            self.assertTrue(deadline.expired)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("Stage wall deadline", receipt["error"])
        self.assertFalse(receipt["eligible_for_strict_gold_comparison"])
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].p.poll())

    def test_blocked_collector_wait_stops_both_owned_children(self):
        with compare.WallDeadline(0.2) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: {"status": "ready"},
                                           self.owner_factory(sleep_only=True))
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("Stage wall deadline", receipt["error"])
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.p.poll() is not None for child in self.children))

    def test_expiry_during_cleanup_does_not_interrupt_child_retirement(self):
        timers = []
        def fail(*_, **__):
            raise ValueError("synthetic work failure")
        with compare.WallDeadline(0.15) as deadline:
            receipt = self.execute_fixture(deadline, fail, self.owner_factory(cleanup_delay=0.25, cleanup_timers=timers))
            with self.assertRaises(compare.WallDeadlineExceeded):
                deadline.check()
        self.assertEqual(receipt["error"], "synthetic work failure")
        self.assertEqual(timers, [(0.0, 0.0)])
        self.assertIsNotNone(self.children[0].p.poll())

    def test_process_created_across_cutoff_is_owned_before_exception(self):
        with compare.WallDeadline(0.15) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: self.fail("HTTP must not start"),
                                           self.owner_factory(construction_delay=0.25))
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("Stage wall deadline", receipt["error"])
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].p.poll())

    def test_collector_cleanup_error_preserves_primary_failure_and_stops_server(self):
        factory = self.owner_factory(sleep_only=True)
        children = self.children
        class CollectorCleanupFailure(factory):
            def stop(self):
                super().stop()
                if len(children) == 2 and self is children[1] and not getattr(self, "reported_failure", False):
                    self.reported_failure = True
                    raise RuntimeError("synthetic collector cleanup failure")
        with compare.WallDeadline(0.2) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: {"status": "ready"},
                                           CollectorCleanupFailure)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("Stage wall deadline", receipt["error"])
        self.assertEqual(receipt["cleanup_errors"], [{"step": "collector.stop", "type": "RuntimeError",
                                                     "error": "synthetic collector cleanup failure"}])
        self.assertFalse(receipt["eligible_for_strict_gold_comparison"])
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.p.poll() is not None for child in self.children))
        self.assertIn("host_boundary_after", receipt)
        persisted = json.loads((self.root / "arm/receipt.json").read_text())
        self.assertEqual(persisted, receipt)

    def test_boundary_failure_keeps_primary_error_and_persists_receipt(self):
        directory = self.root / "arm"
        directory.mkdir()
        args = SimpleNamespace(port=0, load_deadline=10, request_deadline=10)
        with compare.WallDeadline(1) as deadline, \
             patch.object(compare, "OwnedProcess", self.owner_factory()), \
             patch.object(compare, "host_boundary_snapshot", side_effect=[{}, RuntimeError("synthetic boundary failure")]), \
             patch.object(compare, "http_json", side_effect=ValueError("synthetic work failure")):
            receipt = compare.execute(self.plan(), directory, self.request, None, args, deadline)
        self.assertEqual(receipt["error"], "synthetic work failure")
        self.assertEqual(receipt["cleanup_errors"][0]["step"], "host_boundary_snapshot")
        self.assertIsNotNone(self.children[0].p.poll())
        self.assertEqual(json.loads((directory / "receipt.json").read_text()), receipt)

    def test_sigint_records_cancelled_receipt_and_retires_owned_child(self):
        def interrupt(*_, **__):
            signal.raise_signal(signal.SIGINT)
            self.fail("SIGINT must interrupt work")
        with compare.WallDeadline(1) as deadline:
            receipt = self.execute_fixture(deadline, interrupt, self.owner_factory())
        self.assertEqual(receipt["status"], "cancelled")
        self.assertEqual(receipt["interruption"], "KeyboardInterrupt")
        self.assertIn("SIGINT", receipt["error"])
        self.assertFalse(receipt["eligible_for_strict_gold_comparison"])
        self.assertIsNotNone(self.children[0].p.poll())
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(json.loads((self.root / "arm/receipt.json").read_text()), receipt)

    def test_cancelled_arm_persists_stage_summary_and_never_starts_later_arm(self):
        original_execute = compare.execute
        calls = []
        def execute_with_synthetic_interrupt(plan, directory, request, encoded, args, deadline):
            calls.append(plan["arm"])
            def interrupt(*_, **__):
                raise KeyboardInterrupt()
            args.port = 0  # Kernel-selected port; no model or real HTTP server.
            with patch.object(compare, "OwnedProcess", self.owner_factory(sleep_only=True)), \
                 patch.object(compare, "host_boundary_snapshot", return_value={"counters": {}, "raw": {}}), \
                 patch.object(compare, "host_boundary_delta", return_value={}), \
                 patch.object(compare, "http_json", side_effect=interrupt):
                return original_execute(plan, directory, request, encoded, args, deadline)
        with patch.object(compare, "execute", side_effect=execute_with_synthetic_interrupt), \
             patch.object(compare, "host_memory_controls", return_value={}), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = compare.main(self.arguments("--wall-deadline-seconds", "1"))
        self.assertEqual(result, 130)
        self.assertEqual(calls, ["before"])
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual([row["status"] for row in summary], ["cancelled"])
        self.assertFalse(summary[0]["eligible_for_strict_gold_comparison"])
        self.assertIsNotNone(self.children[0].p.poll())
        self.assertFalse((self.root / "output/deepseek-1-candidate").exists())

    def test_collector_wait_interrupt_retires_both_children(self):
        factory = self.owner_factory(sleep_only=True)
        children = self.children
        class InterruptingCollector(factory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if len(children) == 2:
                    original_wait = self.p.wait
                    def interrupt_first_wait(*args, **kwargs):
                        self.p.wait = original_wait
                        raise KeyboardInterrupt()
                    self.p.wait = interrupt_first_wait
        with compare.WallDeadline(1) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: {"status": "ready"}, InterruptingCollector)
        self.assertEqual(receipt["status"], "cancelled")
        self.assertFalse(receipt["eligible_for_strict_gold_comparison"])
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.p.poll() is not None for child in self.children))
        self.assertEqual(json.loads((self.root / "arm/receipt.json").read_text()), receipt)

    def test_sigint_during_collector_construction_keeps_both_owners(self):
        factory = self.owner_factory(sleep_only=True)
        children = self.children
        previous = signal.getsignal(signal.SIGINT)
        class InterruptedConstruction(factory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if len(children) == 2:
                    signal.raise_signal(signal.SIGINT)
        with compare.WallDeadline(1) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: {"status": "ready"}, InterruptedConstruction)
        self.assertEqual(receipt["status"], "cancelled")
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.p.poll() is not None for child in self.children))
        self.assertIs(signal.getsignal(signal.SIGINT), previous)
        self.assertEqual(json.loads((self.root / "arm/receipt.json").read_text()), receipt)

    def test_first_sigint_during_cleanup_preserves_failure_and_retires_both(self):
        factory = self.owner_factory(sleep_only=True)
        children = self.children
        previous = signal.getsignal(signal.SIGINT)
        class InterruptedCleanup(factory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if len(children) == 2:
                    original_wait = self.p.wait
                    def fail_first_wait(*args, **kwargs):
                        self.p.wait = original_wait
                        raise ValueError("synthetic collector work failure")
                    self.p.wait = fail_first_wait
            def stop(self):
                if len(children) == 2 and self is children[1] and not getattr(self, "interrupted", False):
                    self.interrupted = True
                    signal.raise_signal(signal.SIGINT)
                super().stop()
        with compare.WallDeadline(1) as deadline:
            receipt = self.execute_fixture(deadline, lambda *_args, **_kw: {"status": "ready"}, InterruptedCleanup)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["error"], "synthetic collector work failure")
        self.assertEqual(receipt["interruption"], "KeyboardInterrupt")
        self.assertFalse(receipt["eligible_for_strict_gold_comparison"])
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.p.poll() is not None for child in self.children))
        self.assertIs(signal.getsignal(signal.SIGINT), previous)
        self.assertEqual(json.loads((self.root / "arm/receipt.json").read_text()), receipt)

    def test_interrupt_outside_execute_returns_cancelled_exit_code(self):
        with patch.object(compare, "run", side_effect=KeyboardInterrupt()), redirect_stderr(io.StringIO()) as stderr:
            result = compare.main(self.arguments("--wall-deadline-seconds", "1"))
        self.assertEqual(result, 130)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "cancelled")
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_absolute_cutoff_in_cleanup_prevents_later_arm(self):
        calls = []
        def fake_execute(plan, directory, request, encoded, args, deadline):
            with deadline.work():
                calls.append(plan["arm"])
            time.sleep(0.25)  # Simulate reserved, deliberately uninterruptible cleanup.
            return {"status": "completed", "arm": plan["arm"]}
        with patch.object(compare, "execute", side_effect=fake_execute), \
             patch.object(compare, "host_memory_controls", return_value={}), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = compare.main(self.arguments("--wall-deadline-seconds", "0.15"))
        self.assertEqual(result, 1)
        self.assertEqual(calls, ["before"])
        self.assertFalse((self.root / "output/deepseek-1-candidate").exists())

    def test_prepare_only_never_launches_or_installs_alarm(self):
        with patch.object(compare, "execute", side_effect=AssertionError("must not execute")), \
             patch.object(compare, "OwnedProcess", side_effect=AssertionError("must not spawn")), \
             patch.object(compare.signal, "setitimer", side_effect=AssertionError("must not install alarm")), \
             patch.object(compare, "host_memory_controls", return_value={}), redirect_stdout(io.StringIO()):
            result = compare.main(self.arguments("--prepare-only", "--wall-deadline-seconds", "0.000001"))
        self.assertEqual(result, 0)
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual([row["status"] for row in summary], ["prepared"] * 3)
        self.assertTrue(all(row["wall_deadline_seconds"] == 0.000001 for row in summary))

    def test_default_does_not_install_alarm(self):
        with patch.object(compare.signal, "setitimer", side_effect=AssertionError("default must not install alarm")):
            with compare.WallDeadline() as deadline:
                with deadline.work():
                    deadline.check()

    def test_handler_and_disabled_timer_are_restored(self):
        previous = signal.getsignal(signal.SIGALRM)
        handler = lambda *_: None
        signal.signal(signal.SIGALRM, handler)
        try:
            with compare.WallDeadline(0.15) as deadline:
                with self.assertRaises(compare.WallDeadlineExceeded):
                    with deadline.work():
                        time.sleep(5)
            self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        finally:
            signal.signal(signal.SIGALRM, previous)

    def test_existing_alarm_is_not_stolen(self):
        previous = signal.getsignal(signal.SIGALRM)
        handler = lambda *_: None
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, 10)
        try:
            with self.assertRaisesRegex(RuntimeError, "existing real-time alarm"):
                with compare.WallDeadline(1):
                    pass
            self.assertIs(signal.getsignal(signal.SIGALRM), handler)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 9)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def test_all_deadlines_reject_nonfinite_and_nonpositive_cli_values(self):
        for option in ("--load-deadline", "--request-deadline", "--wall-deadline-seconds"):
            for value in ("nan", "inf", "-inf", "0", "-1", "1e309"):
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        compare.main(self.arguments(option + "=" + value))
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("finite and positive", stderr.getvalue())
        self.assertEqual(compare.positive_seconds("0.001"), 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
