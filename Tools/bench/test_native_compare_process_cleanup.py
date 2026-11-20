"""Deterministic owned-child exit races; no process launch, model, or I/O workload."""
import signal
import subprocess
import unittest
from unittest.mock import Mock, call, patch

from native_compare import OwnedProcess


class OwnedProcessCleanupChecks(unittest.TestCase):
    def owner(self, *, poll=None, waits=None):
        owned = OwnedProcess.__new__(OwnedProcess)
        owned.p = Mock(pid=12345)
        owned.p.poll.return_value = poll
        owned.p.wait.side_effect = waits
        owned.threads = [Mock(), Mock()]
        return owned

    def assert_readers_joined(self, owned):
        for thread in owned.threads:
            thread.join.assert_called_once_with(timeout=2)

    def test_exit_between_poll_and_term_still_reaps_and_joins(self):
        owned = self.owner(waits=[0])
        with patch("native_compare.os.killpg", side_effect=ProcessLookupError) as kill:
            owned.stop()
        kill.assert_called_once_with(12345, signal.SIGTERM)
        owned.p.wait.assert_called_once_with(timeout=35)
        self.assert_readers_joined(owned)

    def test_exit_between_timeout_and_kill_still_reaps_and_joins(self):
        owned = self.owner(waits=[subprocess.TimeoutExpired("fixture", 35), 0])
        with patch("native_compare.os.killpg", side_effect=[None, ProcessLookupError]) as kill:
            owned.stop()
        self.assertEqual(kill.call_args_list, [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])
        self.assertEqual(owned.p.wait.call_args_list, [call(timeout=35), call(timeout=5)])
        self.assert_readers_joined(owned)

    def test_other_signal_errors_remain_visible_and_readers_join(self):
        owned = self.owner()
        with patch("native_compare.os.killpg", side_effect=PermissionError("synthetic permission failure")):
            with self.assertRaises(PermissionError):
                owned.stop()
        owned.p.wait.assert_not_called()
        self.assert_readers_joined(owned)

    def test_reap_failure_remains_visible_and_readers_join(self):
        owned = self.owner(waits=[subprocess.TimeoutExpired("fixture", 35), subprocess.TimeoutExpired("fixture", 5)])
        with patch("native_compare.os.killpg"):
            with self.assertRaises(subprocess.TimeoutExpired):
                owned.stop()
        self.assert_readers_joined(owned)

    def test_already_exited_owner_preserves_no_signal_behavior(self):
        owned = self.owner(poll=0)
        with patch("native_compare.os.killpg") as kill:
            owned.stop()
        kill.assert_not_called()
        owned.p.wait.assert_not_called()
        self.assert_readers_joined(owned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
