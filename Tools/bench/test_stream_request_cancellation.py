import importlib.util
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).with_name("stream_request.py")
SPEC = importlib.util.spec_from_file_location("collector_under_test", SCRIPT)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


class CollectorCancellationTests(unittest.TestCase):
    def interrupted_stream(self, sig):
        sent = threading.Event()
        finish = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n')
                self.wfile.flush()
                sent.set()
                finish.wait(5)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = root / "request.json"
                request.write_text(json.dumps({"model": "synthetic", "messages": []}))
                child = subprocess.Popen([sys.executable, "-B", str(SCRIPT), "--url",
                    "http://127.0.0.1:%d/test" % server.server_port, "--request", str(request),
                    "--output", str(root / "output"), "--repeat", "2", "--timeout", "5"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertTrue(sent.wait(3), "synthetic frame not sent")
                    time.sleep(.2)
                    child.send_signal(sig)
                    stdout, stderr = child.communicate(timeout=3)
                    receipt_file = root / "output/request-00/receipt.json"
                    self.assertTrue(receipt_file.exists(), (child.returncode, stdout, stderr))
                    receipt = json.loads(receipt_file.read_text())
                    self.assertFalse(receipt["complete"])
                    self.assertEqual(receipt["interrupted_signal"], int(sig))
                    self.assertEqual(child.returncode, 128 + int(sig))
                    self.assertEqual((root / "output/request-00/content.txt").read_text(), "partial")
                    frames = (root / "output/request-00/frames.jsonl").read_text()
                    self.assertIn("partial", frames)
                    self.assertNotIn("[DONE]", frames)
                    self.assertNotIn("completion_tokens_per_wall_second", receipt)
                    self.assertEqual(json.loads((root / "output/summary.json").read_text()), [receipt])
                    self.assertFalse((root / "output/request-01").exists())
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate(timeout=3)
        finally:
            finish.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_sigterm_retains_partial_stream_and_terminal_receipt(self):
        self.interrupted_stream(signal.SIGTERM)

    def test_sigint_retains_partial_stream_and_terminal_receipt(self):
        self.interrupted_stream(signal.SIGINT)

    def test_first_and_repeated_sigterm_during_flush_preserve_artifacts_and_handlers(self):
        self.interrupted_flush("receipt.json")

    def test_first_sigterm_during_summary_flush_updates_terminal_receipt(self):
        self.interrupted_flush("summary.json")

    def test_signal_at_measure_return_recovers_owned_receipt_before_summary(self):
        self.interrupted_flush("handoff")

    def interrupted_flush(self, trigger):
        class Response:
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"complete"}}]}\n'
                yield b'data: [DONE]\n'

        before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text(json.dumps({"model": "synthetic", "messages": []}))
            args = argparse.Namespace(url="http://localhost/fake", request=request,
                                      output=root / "out", repeat=1 if trigger == "summary.json" else 2, timeout=1)
            original_write = COLLECTOR.write_json
            original_measure = COLLECTOR.measure
            injected = []

            def write(path, value):
                if path.name == trigger and not injected:
                    injected.append(True)
                    os.kill(os.getpid(), signal.SIGTERM)
                    os.kill(os.getpid(), signal.SIGTERM)
                original_write(path, value)

            def measure(*values):
                result = original_measure(*values)
                if trigger == "handoff":
                    os.kill(os.getpid(), signal.SIGTERM)
                return result

            with COLLECTOR.CollectionSignals() as signals:
                with mock.patch.object(COLLECTOR.urllib.request, "urlopen", return_value=Response()), \
                        mock.patch.object(COLLECTOR, "write_json", side_effect=write), \
                        mock.patch.object(COLLECTOR, "measure", side_effect=measure), \
                        mock.patch("builtins.print"):
                    if trigger == "handoff":
                        with self.assertRaises(COLLECTOR.CollectionInterrupted):
                            COLLECTOR.run(args, argparse.ArgumentParser(), signals)
                    else:
                        self.assertEqual(COLLECTOR.run(args, argparse.ArgumentParser(), signals), 1)
                self.assertEqual(signals.interrupted, signal.SIGTERM)
            receipt = json.loads((root / "out/request-00/receipt.json").read_text())
            self.assertEqual(receipt["interrupted_signal"], signal.SIGTERM)
            self.assertFalse(receipt["complete"])
            self.assertEqual((root / "out/request-00/content.txt").read_text(), "complete")
            self.assertEqual(json.loads((root / "out/summary.json").read_text()), [receipt])
            self.assertFalse((root / "out/request-01").exists())
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)


if __name__ == "__main__":
    unittest.main()
