"""Synthetic libproc/ordering checks only: no live process or memory-load probe."""
import ctypes
import errno
import json
import os
import tempfile
import types
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import native_compare as native


def snapshot(current=100, peak=100, birth=11, exited=0):
    return {'started_utc': '2026-11-07T00:00:00+00:00', 'finished_utc': '2026-11-07T00:00:00.000001+00:00',
        'monotonic_start_ns': 1, 'monotonic_end_ns': 1001, 'rusage_returncode': 0,
        'verified_executable_before': '/test/origami', 'verified_executable_after': '/test/origami',
        'current_rss_bytes': current + 7, 'current_footprint_bytes': current,
        'kernel_reported_lifetime_peak_bytes': peak, 'process_start_abstime': birth, 'process_exit_abstime': exited}


class FakeFunction:
    def __init__(self, fn): self.fn = fn; self.calls = 0
    def __call__(self, *args): self.calls += 1; return self.fn(*args)


class FakeLib:
    def __init__(self, paths=('/test/origami', '/test/origami'), result=0):
        self.paths = iter(paths)
        def path(pid, out, length):
            out.value = next(self.paths).encode(); return len(out.value)
        def usage(pid, flavor, out):
            if result: ctypes.set_errno(errno.EIO); return result
            row = ctypes.cast(out, ctypes.POINTER(native._RUsageInfoV4)).contents
            row.phys_footprint = 100; row.lifetime_max_phys_footprint = 120
            row.resident_size = 107; row.proc_start_abstime = 11
            row.user_time = 13; row.system_time = 17; row.diskio_bytesread = 101; row.diskio_byteswritten = 19
            return 0
        self.proc_pidpath = FakeFunction(path); self.proc_pid_rusage = FakeFunction(usage)


class MemorySnapshotTests(unittest.TestCase):
    def test_abi_matches_installed_sdk_field_order_and_offsets(self):
        header = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/resource.h')
        if header.is_file():
            block = header.read_text().split('struct rusage_info_v4 {', 1)[1].split('};', 1)[0]
            self.assertEqual(re.findall(r'uint64_t\s+ri_(\w+)\s*;', block), native._RUSAGE_V4_NAMES)
        self.assertEqual(ctypes.sizeof(native._RUsageInfoV4), 296)
        self.assertEqual(native._RUsageInfoV4.phys_footprint.offset, 72)
        self.assertEqual(native._RUsageInfoV4.proc_start_abstime.offset, 80)
        self.assertEqual(native._RUsageInfoV4.lifetime_max_phys_footprint.offset, 240)

    def test_raw_query_brackets_executable_and_records_raw_values_times(self):
        lib = FakeLib()
        with patch.object(native.ctypes, 'CDLL', return_value=lib):
            result = native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(lib.proc_pidpath.calls, 2); self.assertEqual(lib.proc_pid_rusage.calls, 1)
        self.assertEqual(result['peak_footprint_bytes'], 120)
        raw = result['memory_observations'][0]
        self.assertEqual(raw['current_footprint_bytes'], 100)
        self.assertLessEqual(raw['monotonic_start_ns'], raw['monotonic_end_ns'])
        self.assertTrue(result['last_snapshot_peak_at_least_current'])
        self.assertEqual(result['resource_counters'], {'user_time_mach': 13, 'system_time_mach': 17,
            'diskio_bytesread': 101, 'diskio_byteswritten': 19, 'pageins': 0})

    def test_positive_race_resamples_same_birth_then_returns_known_max(self):
        with patch.object(native, '_kernel_process_memory_snapshot', side_effect=[snapshot(120, 100), snapshot(115, 130)]) as read:
            result = native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(read.call_count, 2); self.assertEqual(result['peak_footprint_bytes'], 130)
        self.assertEqual(len(result['memory_observations']), 2)

    def test_earlier_over_cap_never_hidden_by_lower_retry(self):
        cap = 12_500_000_000
        with patch.object(native, '_kernel_process_memory_snapshot', side_effect=[snapshot(cap + 1, cap - 1), snapshot(100, 200)]):
            result = native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(result['peak_footprint_bytes'], cap + 1)
        self.assertEqual(result['kernel_reported_lifetime_peak_bytes'], cap - 1)

    def test_three_positive_races_are_bounded_and_explicitly_non_atomic(self):
        with patch.object(native, '_kernel_process_memory_snapshot', side_effect=[snapshot(110, 100), snapshot(120, 110), snapshot(130, 120)]) as read:
            result = native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(read.call_count, 3); self.assertEqual(result['peak_footprint_bytes'], 130)
        self.assertEqual(result['kernel_reported_lifetime_peak_bytes'], 120)
        self.assertFalse(result['last_snapshot_peak_at_least_current'])
        self.assertIn('not an atomic snapshot', result['peak_measurement_scope'])

    def test_zero_or_dead_fields_fail_without_retry_and_retain_raw(self):
        for raw in [snapshot(0, 100), snapshot(100, 0), snapshot(100, 100, birth=0), snapshot(100, 100, exited=1)]:
            with self.subTest(raw=raw), patch.object(native, '_kernel_process_memory_snapshot', return_value=raw) as read:
                with self.assertRaises(ValueError) as caught: native.kernel_process_memory(10, '/test/origami')
                self.assertEqual(read.call_count, 1); self.assertEqual(caught.exception.memory_observations, [raw])

    def test_birth_change_fails_and_retains_both_attempts(self):
        attempts = [snapshot(120, 100), snapshot(120, 130, birth=12)]
        with patch.object(native, '_kernel_process_memory_snapshot', side_effect=attempts) as read:
            with self.assertRaisesRegex(ValueError, 'birth changed') as caught: native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(read.call_count, 2); self.assertEqual(caught.exception.memory_observations, attempts)

    def test_wrong_executable_not_retried_before_or_after_rusage(self):
        for paths, calls in [(('/other',), 0), (('/test/origami', '/other'), 1)]:
            lib = FakeLib(paths=paths)
            with self.subTest(paths=paths), patch.object(native.ctypes, 'CDLL', return_value=lib):
                with self.assertRaisesRegex(ValueError, 'executable') as caught: native.kernel_process_memory(10, '/test/origami')
                self.assertEqual(lib.proc_pid_rusage.calls, calls)
                self.assertEqual(len(caught.exception.memory_observations), 1)
                self.assertIn('finished_utc', caught.exception.memory_observations[0])

    def test_kernel_error_keeps_original_errno_and_attempt_evidence(self):
        lib = FakeLib(result=-1)
        with patch.object(native.ctypes, 'CDLL', return_value=lib):
            with self.assertRaises(OSError) as caught: native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(caught.exception.memory_observations[0]['rusage_returncode'], -1)
        self.assertEqual(lib.proc_pid_rusage.calls, 1)

    def test_inferssd_postmeasurement_ownership_failure_keeps_raw(self):
        with tempfile.TemporaryDirectory(prefix="T13-owner-fixture-") as temp:
            directory = Path(temp)
            owner = {"controller_pid": 100, "token": "fixture-owner"}
            state = {"pid": 101, "returncode": None}
            for name, value in [(".inferssd-owner", owner), ("server-state.json", state),
                                ("runtime.json", {"argv": ["/test/origami"], "directory": str(directory)})]:
                (directory / name).write_text(json.dumps(value))
            raw = snapshot()
            def measured_then_owner_changes(pid, expected):
                (directory / "server-state.json").write_text(json.dumps({"pid": 101, "returncode": 0}))
                return {"measured": True, "peak_footprint_bytes": 100, "memory_observations": [raw]}
            owned = types.SimpleNamespace(p=types.SimpleNamespace(pid=100, poll=lambda: None))
            tree = "101 100 %d\n100 %d %d\n" % (os.geteuid(), os.getpid(), os.geteuid())
            with patch.object(native.os, "getpgid", return_value=100), \
                    patch.object(native.subprocess, "run", return_value=types.SimpleNamespace(stdout=tree)), \
                    patch.object(native, "kernel_process_memory", side_effect=measured_then_owner_changes):
                result = native.inferssd_process_memory(owned, "/test/origami", directory)
        self.assertFalse(result["measured"])
        self.assertIn("ownership changed", result["error"])
        self.assertEqual(result["memory_observations"], [raw])

    def test_failed_retry_keeps_earlier_successful_raw_attempt(self):
        error = OSError(errno.ESRCH, 'gone')
        error.memory_observation = {'error': 'gone', 'rusage_returncode': -1}
        raw = snapshot(120, 100)
        with patch.object(native, '_kernel_process_memory_snapshot', side_effect=[raw, error]):
            with self.assertRaises(OSError) as caught: native.kernel_process_memory(10, '/test/origami')
        self.assertEqual(caught.exception.memory_observations, [raw, error.memory_observation])


if __name__ == '__main__': unittest.main()
