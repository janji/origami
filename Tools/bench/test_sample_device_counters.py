"""Synthetic plist, clock, file, and tiny subprocess fixtures; no model/device run."""
import json
import os
from pathlib import Path
import plistlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import sample_device_counters as sampler


def media(name="disk0", identity=900, children=None):
    result = {"IOObjectClass": "IOMedia", "Whole": True, "BSD Name": name,
              "IORegistryEntryID": identity, "IORegistryEntryName": "Synthetic media"}
    if children is not None:
        result["IORegistryEntryChildren"] = children
    return result


def driver(name="disk0", identity=10, media_id=900, counter=100):
    return {"IOObjectClass": "IOBlockStorageDriver", "IORegistryEntryID": identity,
            "Statistics": {key: counter for key in sampler.COUNTERS},
            "IORegistryEntryChildren": [media(name, media_id)]}


def snapshot(roots=None, start=1_000_000_000, end=1_020_000_000, devices=("disk0",)):
    return {"valid": True, "monotonic_start_ns": start, "monotonic_end_ns": end,
            "devices": sampler.parse_devices(plistlib.dumps(roots or [driver()]), devices)}


class CounterChecks(unittest.TestCase):
    def test_descendant_identity_is_not_registry_arithmetic(self):
        root = driver()
        root["IORegistryEntryChildren"] = [{"IOObjectClass": "SyntheticBridge",
                                            "IORegistryEntryChildren": [media(identity=777)]}]
        row = snapshot([root])["devices"]["disk0"]
        self.assertTrue(row["valid"])
        self.assertEqual(row["driver_registry_id"], 10)
        self.assertEqual(row["media_registry_id"], 777)

    def test_physical_media_stops_apfs_descendant_double_count(self):
        root = driver()
        root["IORegistryEntryChildren"][0]["IORegistryEntryChildren"] = [media("disk3", 903)]
        rows = snapshot([root], devices=("disk0", "disk3"))["devices"]
        self.assertTrue(rows["disk0"]["valid"])
        self.assertEqual(rows["disk3"]["reasons"], ["missing_device"])

    def test_duplicate_bsd_and_driver_id_fail_closed(self):
        rows = snapshot([driver(), driver(identity=12, media_id=902)])["devices"]
        self.assertEqual(rows["disk0"]["reasons"], ["ambiguous_bsd_mapping"])
        rows = snapshot([driver(), driver("disk6", media_id=906)], devices=("disk0", "disk6"))["devices"]
        self.assertTrue(all("duplicate_driver_identity" in row["reasons"] for row in rows.values()))

    def test_multiple_whole_media_are_ambiguous_even_if_only_one_requested(self):
        root = driver()
        root["IORegistryEntryChildren"].append(media("disk6", 906))
        self.assertIn("ambiguous_driver_media", snapshot([root])["devices"]["disk0"]["reasons"])

    def test_missing_properties_and_bad_counters_are_not_zeroes(self):
        for bad in (-1, True, float("nan"), float("inf"), "123"):
            root = driver()
            root["Statistics"]["Bytes (Read)"] = bad
            row = snapshot([root])["devices"]["disk0"]
            self.assertFalse(row["valid"])
            self.assertIsNone(row["raw_counters"]["Bytes (Read)"])
            json.dumps(row, allow_nan=False)
        root = driver()
        del root["Statistics"]["Bytes (Read)"]
        del root["IORegistryEntryChildren"][0]["IORegistryEntryID"]
        row = snapshot([root])["devices"]["disk0"]
        self.assertIn("missing_registry_identity", row["reasons"])
        self.assertIn("invalid_or_missing_counter", row["reasons"])

    def test_counter_delta_and_time_bracket_not_since_boot_rate(self):
        old = snapshot([driver(counter=1_000_000)])
        new = snapshot([driver(counter=1_001_000)], start=3_000_000_000, end=3_040_000_000)
        delta = sampler.interval_deltas(old, new, ("disk0",))["disk0"]
        self.assertTrue(delta["valid"])
        self.assertEqual(delta["counter_deltas"]["Bytes (Read)"], 1000)
        self.assertEqual(delta["elapsed_seconds_bounds"], [1.98, 2.04])
        self.assertAlmostEqual(delta["estimated_bytes_per_second"]["Read"], 1000 / 2.01)
        initial = sampler.interval_deltas(None, old, ("disk0",))["disk0"]
        self.assertEqual(initial["reasons"], ["initial_baseline_only"])
        self.assertNotIn("estimated_bytes_per_second", initial)

    def test_reset_and_hotplug_identity_change_invalidated(self):
        old = snapshot()
        for root, reason in [(driver(counter=99), "counter_reset_or_wrap"),
                             (driver(identity=11), "device_identity_changed"),
                             (driver(media_id=901), "device_identity_changed")]:
            new = snapshot([root], start=3_000_000_000, end=3_010_000_000)
            row = sampler.interval_deltas(old, new, ("disk0",))["disk0"]
            self.assertEqual(row["reasons"], [reason])
            self.assertNotIn("counter_deltas", row)

    def test_missing_or_failed_capture_breaks_continuity(self):
        old = snapshot()
        invalid = {"valid": False, "devices": {}, "monotonic_start_ns": 2_000_000_000,
                   "monotonic_end_ns": 2_010_000_000}
        new = snapshot(start=3_000_000_000, end=3_010_000_000)
        self.assertFalse(sampler.interval_deltas(old, invalid, ("disk0",))["disk0"]["valid"])
        self.assertFalse(sampler.interval_deltas(invalid, new, ("disk0",))["disk0"]["valid"])
        missing = snapshot([driver("disk6")], start=3_000_000_000, end=3_010_000_000)
        self.assertIn("invalid_device_snapshot", sampler.interval_deltas(old, missing, ("disk0",))["disk0"]["reasons"])

    def test_overlapping_or_reversed_time_brackets_invalidated(self):
        old = snapshot()
        for start, end in [(1_010_000_000, 1_030_000_000), (3_000_000_000, 2_000_000_000)]:
            row = sampler.interval_deltas(old, snapshot(start=start, end=end), ("disk0",))["disk0"]
            self.assertIn("invalid_or_overlapping_time_brackets", row["reasons"])

    def test_malformed_payload_is_recorded_as_failure(self):
        for payload in [b"garbage", b"<?xml version='1.0'?><plist><", plistlib.dumps({"unexpected": []})]:
            row = sampler.capture(("disk0",), 1, command=lambda timeout: payload, clock=iter([100, 200]).__next__)
            self.assertFalse(row["valid"])
            self.assertIn("error", row)
            self.assertEqual(row["monotonic_end_ns"], 200)

    def test_payload_bounds_and_unexpected_tree(self):
        with patch.object(sampler, "MAX_COMMAND_BYTES", 20):
            with self.assertRaisesRegex(ValueError, "size limit"):
                sampler.parse_devices(b"x" * 21, ("disk0",))
        root = driver()
        root["IORegistryEntryChildren"] = "wrong type"
        with self.assertRaisesRegex(ValueError, "descendant list"):
            sampler.parse_devices(plistlib.dumps([root]), ("disk0",))

    def test_plan_rejects_unbounded_and_virtual_partition_arguments(self):
        sampler.validate_plan(["disk0", "disk6"], 2, 300, 3, False)
        for args in [(["disk0s2"], 2, 300, 3, False), (["disk0", "disk0"], 2, 300, 3, False),
                     (["disk0"], .1, 7200, 3, False), (["disk0"], float("nan"), 300, 3, False),
                     (["disk0"], 2, 1, 3, False), (["disk0"], 2, 300, 0, False)]:
            with self.assertRaises(ValueError):
                sampler.validate_plan(*args)

    def test_tiny_subprocess_output_and_timeout_bounds(self):
        with patch.object(sampler, "COMMAND", (sys.executable, "-c", "print('fixture')")):
            self.assertEqual(sampler.bounded_command(2), b"fixture\n")
        with patch.object(sampler, "COMMAND", (sys.executable, "-c", "import sys; sys.stderr.write('x'*129)")), \
                patch.object(sampler, "MAX_COMMAND_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "size limit"):
                sampler.bounded_command(2)
        with patch.object(sampler, "COMMAND", (sys.executable, "-c", "import time; time.sleep(10)")):
            with self.assertRaisesRegex(TimeoutError, "timeout"):
                sampler.bounded_command(.05)

    def test_cli_one_shot_preserves_output_and_has_no_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "once.jsonl"
            row = {"kind": "sample", **snapshot()}
            with patch.object(sampler, "capture", return_value=row) as capture:
                self.assertEqual(sampler.main(["--device", "disk0", "--output", str(path), "--once"]), 0)
                self.assertEqual(capture.call_count, 1)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["header", "sample", "footer"])
            self.assertEqual(rows[1]["deltas"]["disk0"]["reasons"], ["initial_baseline_only"])
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            with patch.object(sampler, "capture") as capture:
                with self.assertRaises(FileExistsError):
                    sampler.main(["--device", "disk0", "--output", str(path), "--once"])
                capture.assert_not_called()

    def test_cli_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            row = {"kind": "sample", "valid": False, "devices": {},
                   "monotonic_start_ns": 100, "monotonic_end_ns": 200}
            with patch.object(sampler, "capture", return_value=row):
                self.assertEqual(sampler.main(["--device", "disk0", "--output", str(path), "--once"]), 1)
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])["invalid_capture_count"], 1)

    def test_cli_deadline_clips_timeout_and_skips_slow_capture_slots(self):
        for interval, capture_duration, expected_starts in [(.45, .01, [0, .45, .9]), (.2, .3, [0, .5])]:
            now, calls = [0.0], []
            def fake_capture(devices, timeout):
                calls.append((now[0], timeout))
                start = int(now[0] * 1e9)
                now[0] += min(capture_duration, timeout)
                return {"kind": "sample", **snapshot(start=start, end=int(now[0] * 1e9))}
            def fake_sleep(duration):
                now[0] += duration
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "intervals.jsonl"
                with patch.object(sampler.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(sampler.time, "sleep", side_effect=fake_sleep), \
                        patch.object(sampler, "capture", side_effect=fake_capture):
                    self.assertEqual(sampler.main(["--device", "disk0", "--output", str(path),
                                                  "--duration-seconds", "1", "--interval-seconds", str(interval),
                                                  "--command-timeout-seconds", ".3"]), 0)
                self.assertEqual(len(calls), len(expected_starts))
                for (start, timeout), expected in zip(calls, expected_starts):
                    self.assertAlmostEqual(start, expected)
                    self.assertLessEqual(timeout, 1 - start)
                    self.assertLess(start, 1)
                self.assertAlmostEqual(now[0], 1)

    def test_cli_interrupt_preserves_completed_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interrupted.jsonl"
            with patch.object(sampler, "capture", side_effect=KeyboardInterrupt):
                self.assertEqual(sampler.main(["--device", "disk0", "--output", str(path)]), 1)
            footer = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(footer["reason"], "interrupted")
            self.assertEqual(footer["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
