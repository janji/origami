#!/usr/bin/env python3
"""Run the real `origami tune --memory` against an isolated HTTP fixture, without a model or GPU.

Usage: python3 -B Tests/tune-memory-checks.py /path/to/origami
The fixture answers status, reloads and replies as a loaded model would. Checked: a whole run reloads at each trial limit,
then the saved settings, and stores its result (setting an unreadable tuning.json aside, never overwriting it); Ctrl-C
(SIGTERM, the job's Cancel) and an app's request each stop it with the saved settings reloaded and nothing stored.
A tune process killed outright cannot restore anything itself: the front's job hook does (Tests/tuning-job-checks.py).
Below the planned limits: when the lowest is cut, the lowest level a cut set is tried next, even if the budget grew back (lower, grown_back, lower_none), at
most three more and never under the model's least budget (lower_minimum), with a line naming what else holds memory; when
that least budget is out too, the result says the model needs it and what macOS left, and names the two remedies.
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import http.server

MARKER = "ORIGAMI_TUNING_OUTCOME "
KEY = "fixture-model@8192"
GB = 1_000_000_000
SETTLE = 1_234_567_890  # how far a cut lowers the budget in the fixture
MINIMUM_GAP = 750_000_000  # lower_minimum: the model's least budget, this far under the lowest planned limit
PLANNED = {}  # the lowest planned limit, from an earlier "lower" scenario (the same on every run on one Mac)


class Fixture:
    def __init__(self, root, scenario):
        self.scenario = scenario
        self.data = root / "data"
        self.data.mkdir(exist_ok=True)
        self.config = root / "config.json"
        self.completed = 10
        self.reloads = []
        self.chats = 0
        self.unexpected = []
        self.lock = threading.Lock()
        self.process = None
        self.limit = None  # the trial limit loaded, or None for the saved settings
        self.first = PLANNED.get("first")  # the first (lowest planned) trial limit

    def cut(self):
        """Whether the trial loaded now is cut once by the Mac."""
        if self.limit is None:
            return False
        if self.scenario in ("lower_none", "lower_minimum"):
            return True
        return self.scenario in ("lower", "grown_back") and self.limit == self.first

    def state(self):
        cut = self.cut()
        # A cut lowers the budget by SETTLE; in grown_back it has climbed back to the limit by the trial's end.
        budget = self.limit - SETTLE if cut and self.scenario != "grown_back" else self.limit or 4 << 30
        learned = {"minimum_bytes": self.first - MINIMUM_GAP if self.scenario == "lower_minimum" and self.first else 2 * GB}
        return {"status": "ready", "queue": {"busy": False, "queued": 0},
                "tuning": {"key": KEY, "identity": {"model": "fixture-model"}, "knobs": {}, "state": "none"},
                "requests": {"completed": self.completed}, "memory": {"budget_bytes": budget},
                "memory_learned": learned, "memory_pressure_cuts": 1 if cut else 0,
                "memory_pressure_cut_to": self.limit - SETTLE if cut else None, "expert_cache": {"bytes": 3 << 30}}


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, content_type="application/json", status=200):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        fixture = self.server.fixture
        if self.path == "/health":
            self.reply({"status": "ready"})
        elif self.path == "/v1/dashboard":
            self.reply({"dashboard": {"tune_block": "Reload to apply saved settings first" if fixture.scenario == "reload_first" else ""}})
        elif self.path == "/v1/status":
            with fixture.lock:
                self.reply(fixture.state())
        else:
            fixture.unexpected.append(("GET", self.path))
            self.reply({"error": {"message": "Unexpected fixture route"}}, status=404)

    def do_POST(self):
        fixture = self.server.fixture
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/v1/reload":
            with fixture.lock:
                fixture.reloads.append(body)
                fixture.limit = body.get("memory_limit_trial")
                if fixture.first is None:
                    fixture.first = fixture.limit
                trials = sum(1 for r in fixture.reloads if "memory_limit_trial" in r)
            if fixture.scenario == "cancel" and trials == 1 and "memory_limit_trial" in body:
                threading.Timer(0.5, lambda: fixture.process.send_signal(signal.SIGTERM)).start()
            if fixture.scenario == "time_limit" and trials == 1 and "memory_limit_trial" in body:
                time.sleep(5)  # the reload outlasts the limit: its request is cut off there
            self.reply({"reloading": True, "model": "fixture-model", "pending": None}, status=202)
            return
        if self.path != "/v1/chat/completions":
            fixture.unexpected.append(("POST", self.path))
            self.reply({"error": {"message": "Unexpected fixture route"}}, status=405)
            return
        with fixture.lock:
            fixture.chats += 1
            fixture.completed += 1
            if fixture.scenario == "app_request" and fixture.chats == 2 or fixture.scenario == "resume_stop" and fixture.chats == 5:
                fixture.completed += 1  # an app's request finished meanwhile
        tokens = min(16, body.get("max_tokens", 16))
        events = [{"choices": [{"delta": {"content": "token "}}]} for _ in range(tokens)]
        # `check_timing`: read only with `reported_timing` (the resume scenarios), so their speeds repeat exactly.
        events.append({"choices": [], "usage": {"prompt_tokens": 25, "prompt_tokens_details": {"cached_tokens": 0},
                                                "completion_tokens": tokens,
                                                "check_timing": {"ttft": 0.05, "decode": tokens * 0.05}}})
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        self.reply((payload + "data: [DONE]\n\n").encode(), "text/event-stream")


def run(binary, scenario):
    with tempfile.TemporaryDirectory(prefix="origami-tune-memory-") as temporary:
        fixture = Fixture(pathlib.Path(temporary), scenario)
        server = Server(("127.0.0.1", 0), Handler)
        server.fixture = fixture
        fixture.config.write_text(json.dumps({"data_dir": str(fixture.data), "context": 8192, "models_dirs": "[]",
                                              "port": server.server_address[1]}))
        fixture.config.chmod(0o600)
        tuning = fixture.data / "tuning.json"
        before = b"{not json" if scenario == "normal" else b'{"entries": {"unrelated": {"values": {}}}}\n'
        tuning.write_bytes(before)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        env = {k: v for k, v in os.environ.items() if not k.startswith("ORIGAMI_")}
        if scenario == "time_limit":
            env["ORIGAMI_TUNING_HOST"] = json.dumps({"minute_seconds": 1})  # the 3-minute limit lasts 3 s
        try:
            fixture.process = subprocess.Popen([binary, "tune", "--memory", "--minutes=3", "--config", str(fixture.config)],
                                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output, _ = fixture.process.communicate(timeout=170)
        finally:
            server.shutdown()
            server.server_close()
        prefix = scenario + ": "
        if scenario == "reload_first":
            # Refused from the command line with the dashboard's words, before anything reloads.
            assert fixture.process.returncode == 1 and "Reload to apply saved settings first" in output, prefix + output
            assert not fixture.reloads and not fixture.chats and not fixture.unexpected, prefix + output
            print("ok   " + scenario, flush=True)
            return
        outcomes = [json.loads(line[len(MARKER):]) for line in output.splitlines() if line.startswith(MARKER)]
        assert len(outcomes) == 1, prefix + "expected one outcome\n" + output
        assert not fixture.unexpected, prefix + repr(fixture.unexpected)
        trials = [r for r in fixture.reloads if "memory_limit_trial" in r]
        assert trials, prefix + "no trial limit was loaded\n" + output
        assert fixture.reloads[-1] == {'restore_memory': True}, prefix + "the saved settings were not reloaded last: " + repr(fixture.reloads) + "\n" + output
        limits = [r["memory_limit_trial"] for r in trials]
        other = "Other apps hold memory: " in output
        assert other == (scenario.startswith("lower") or scenario == "grown_back"), prefix + "the other-apps line, only when the lowest limit was cut\n" + output
        if scenario.startswith("lower") or scenario == "grown_back":
            assert fixture.process.returncode == 0 and outcomes[0]["outcome"] == "comparison", prefix + output
            first = PLANNED.setdefault("first", limits[0])
            assert all(limit <= first for limit in limits), prefix + "a limit above a cut one was loaded: " + repr(limits)
            settle = lambda limit: (limit - SETTLE) // (GB // 10) * (GB // 10)
            def size(limit):
                return f"{limit // GB}.{limit // (GB // 10) % 10}GB"
            if scenario in ("lower", "grown_back"):
                assert limits == [first, settle(first)], prefix + repr(limits) + "\n" + output
                assert ("Chosen: " + size(settle(first))) in outcomes[0]["reason"], prefix + output
            elif scenario == "lower_none":
                expected = [first]
                for _ in range(3):
                    expected.append(settle(expected[-1]))
                assert limits == expected, prefix + repr(limits) + "\n" + output
                assert "down to " + size(expected[-1]) + "; nothing to suggest" in outcomes[0]["reason"], prefix + output
                assert "The model needs at least" not in outcomes[0]["reason"], prefix + output
            else:
                least = -(-(first - MINIMUM_GAP) // (GB // 10)) * (GB // 10)
                assert limits == [first, least], prefix + repr(limits) + "\n" + output
                assert "down to " + size(least) in outcomes[0]["reason"], prefix + output
                short = (f"The model needs at least {size(least)} to start with this context and macOS left it about "
                         f"{size(least - SETTLE)}: use a smaller context (`origami config set context 4096`) or close other apps")
                assert short in outcomes[0]["reason"], prefix + output
        elif scenario == "normal":
            assert fixture.process.returncode == 0 and outcomes[0]["outcome"] == "comparison", prefix + output
            # The fixture's expert cache is the same size at every limit, as on a Mac that keeps every expert in memory:
            # the lowest limit tried is chosen and the result says why, instead of the highest by one trial's speed.
            assert "Chosen: " + f"{min(limits) // GB}.{min(limits) // (GB // 10) % 10}GB" in outcomes[0]["reason"], prefix + output
            assert "hold the same experts in memory, so they add nothing" in outcomes[0]["reason"], prefix + output
            entries = json.loads(tuning.read_text())["entries"]
            assert "memory@" + KEY in entries and entries["memory@" + KEY]["trials"], prefix + "result not stored"
            aside = [p for p in fixture.data.iterdir() if p.name.startswith("tuning.json.unreadable-")]
            assert len(aside) == 1 and aside[0].read_bytes() == before, prefix + "the unreadable tuning.json was not kept aside"
        elif scenario == "time_limit":
            # A request cut off by the limit ends as the limit, not as a server that stopped answering.
            assert outcomes[0]["outcome"] == "incomplete" and "Time limit reached" in outcomes[0]["reason"], prefix + output
        else:
            assert outcomes[0]["outcome"] == "aborted", prefix + output
            assert tuning.read_bytes() == before, prefix + "a stopped run changed tuning.json"
            if scenario == "cancel":
                assert fixture.process.returncode != 0 and len(trials) == 1, prefix + output
        print("ok   " + scenario, flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: tune-memory-checks.py /path/to/origami [scenario...]")
    binary = str(pathlib.Path(sys.argv[1]).resolve(strict=True))
    scenarios = sys.argv[2:] or ["normal", "cancel", "app_request", "reload_first", "lower", "lower_none", "lower_minimum", "grown_back", "time_limit"]
    if "lower_minimum" in scenarios and "first" not in PLANNED and not {"lower", "lower_none"} & set(scenarios):
        scenarios.insert(0, "lower")  # lower_minimum needs the lowest planned limit before the run starts
    for scenario in scenarios:
        run(binary, scenario)
    if not sys.argv[2:]:
        resume(binary)
        scenarios.append("resume")
    print(f"{len(scenarios)} model-free memory-search scenarios passed")


class Paced(Handler):
    """Replies paced at 20 tokens a second (an app's request can arrive during a trial); the speeds the resume's
    agreement test compares are the reported ones (`reported_timing`), so a busy Mac does not change them."""
    def reply(self, body, content_type="application/json", status=200):
        if content_type != "text/event-stream":
            return super().reply(body, content_type, status)
        pieces = [piece + b"\n\n" for piece in body.split(b"\n\n") if piece]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(sum(map(len, pieces))))
        self.send_header("Connection", "close")
        self.end_headers()
        for piece in pieces:
            self.wfile.write(piece)
            self.wfile.flush()
            time.sleep(0.05)


def resume(binary):
    """An app's request during the second trial stops a run with the first trial kept; the next run checks that limit
    again, agrees, and continues from it; a finished run drops what was kept. Every ending says measured, kept, next."""
    with tempfile.TemporaryDirectory(prefix="origami-tune-memory-") as temporary:
        root = pathlib.Path(temporary)
        outputs = []
        for scenario in ("resume_stop", "resume_go"):
            fixture = Fixture(root, scenario)
            server = Server(("127.0.0.1", 0), Paced)
            server.fixture = fixture
            fixture.config.write_text(json.dumps({"data_dir": str(fixture.data), "context": 8192, "models_dirs": "[]",
                                                  "port": server.server_address[1]}))
            fixture.config.chmod(0o600)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            env = {k: v for k, v in os.environ.items() if not k.startswith("ORIGAMI_")}
            env["ORIGAMI_TUNING_HOST"] = json.dumps({"reported_timing": True})
            try:
                fixture.process = subprocess.Popen([binary, "tune", "--memory", "--minutes=6", "--config", str(fixture.config)],
                                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                output, _ = fixture.process.communicate(timeout=300)
            finally:
                server.shutdown()
                server.server_close()
            outcome = [json.loads(line[len(MARKER):]) for line in output.splitlines() if line.startswith(MARKER)][0]
            assert all(word in outcome["reason"] for word in ("Measured:", "Kept:", "Next:")), scenario + ": a bare ending\n" + output
            entries = json.loads((fixture.data / "tuning.json").read_text())["entries"]
            limits = [r["memory_limit_trial"] for r in fixture.reloads if "memory_limit_trial" in r]
            outputs.append((output, outcome, entries, limits))
        (stopped, first, kept, a), (going, last, after, b) = outputs
        assert first["outcome"] == "aborted" and "Measured: 1 trial" in first["reason"] and len(kept["memory-partial@" + KEY]["trials"]) == 1, stopped
        assert "continuing: 1 limit kept" in going and "Kept measurements dropped" not in going and b[0] == a[0] and b.count(a[0]) == 1, going
        assert last["outcome"] == "comparison" and "memory-partial@" + KEY not in after and "memory@" + KEY in after, going
        print("ok   resume", flush=True)


if __name__ == "__main__":
    main()
