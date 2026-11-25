#!/usr/bin/env python3
"""Run the real tuning CLI against an isolated HTTP fixture, without loading a model.

Usage: python3 -B Tests/tuning-flow-checks.py /path/to/origami
Only the ephemeral loopback server is contacted. Saved production settings, models,
and the GPU are not used. The CLI's host safety checks stay enabled but read a fixed,
scaled-down Mac (8 GB, no swapping, normal pressure; ORIGAMI_TUNING_HOST), with the
fixture's memory budget scaled to it: the flow is the same on a small CI runner that
swaps as on a large Mac. Their policy is covered separately by the Swift policy checks.
Then `origami tune` itself (as the dashboard runs it, --review=true) against a fixture that streams its replies, each
ending once (a minute of the limit lasts `minute_seconds`): no candidate faster (a result, "the defaults are fastest"),
a winner, the default limit grown to fit, an owner's limit kept with fewer candidates, the limit reached mid-run after
a slowdown, an app's request and a server restart; each stopped run keeps its rounds and the next run continues them.
"""
import time
import datetime
import http.server
import json
import math
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading


MARKER = "ORIGAMI_TUNING_OUTCOME "
GIB = 1024 ** 3
# The Mac the CLI's host checks read: 8 GB, calm, so no run is refused for the runner's own swapping.
HOST = {"memory_bytes": 8 * GIB, "swapout_bytes": 0, "memory_pressure": 1, "thermal_state": 0, "low_power": False}
# The levels the fixture model compares down to the floor "dynamic".
LEVELS = ("exact", "dynamic")


class Fixture:
    def __init__(self, root, scenario):
        self.scenario = scenario
        self.data = root / "data"
        self.data.mkdir(exist_ok=True)
        self.config = root / "config.json"
        self.key = "fixture-model@8192"
        self.budget = 5 * GIB
        self.completed = 40
        self.requests = []
        self.recent = []
        self.injected_between_samples = False
        self.unexpected = []
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.config_values = {"data_dir": str(self.data), "context": 8192,
                              "quality": "exact", "reasoning_effort": "none",
                              "allow_fast": True, "models_dirs": "[]"}
        self.protected = {}

    def configure(self, port):
        self.config_values["port"] = port
        self.config.write_text(json.dumps(self.config_values))
        self.config.chmod(0o600)
        (self.data / "models.json").write_text('{"models": []}\n')
        (self.data / "tuning.json").write_text('{"entries": {"unrelated": {"values": {"family_spin": false}}}}\n')
        for path in (self.config, self.data / "models.json", self.data / "tuning.json"):
            self.protected[path] = path.read_bytes()
        # A failed run must not leave a previously generated proposal available.
        (self.data / "tuning-proposal.json").write_text('{"kind": "stale-fixture"}')

    def state(self):
        requests = {"recent": self.recent.copy()}
        if self.scenario not in ("legacy_recent", "legacy_between_samples"):
            requests["completed"] = self.completed
        return {"status": "ready", "queue": {"busy": False, "queued": 0}, "quality_levels": list(LEVELS),
                "tuning": {"key": self.key, "identity": {"model": "fixture-model"},
                           "knobs": {}, "state": "none"},
                "requests": requests,
                "memory": {"budget_bytes": self.budget, "cap_bytes": 5 * GIB,
                           "target_bytes": 5 * GIB},
                "context": {"served": 8192}, "experts": {"bits": 4}}

    def after_status_snapshot(self):
        # The third request's post-sample snapshot has already been captured. Insert
        # an app completion before the next pre-sample/settling snapshot: checking
        # only each request's own before/after delta would incorrectly miss this.
        if self.scenario in ("between_samples", "legacy_between_samples") and len(self.requests) == 3 and not self.injected_between_samples:
            self.injected_between_samples = True
            self.append_recent("fast")

    def append_recent(self, mode, completion_tokens=16):
        self.completed += 1
        # Preserve real wire fields; distinct duration also distinguishes rows
        # when every request finishes in the same timestamp second.
        self.recent.append({"time": self.timestamp, "seconds": self.completed / 100000,
                            "mode": mode, "reasoning_effort": "none", "finish": "length",
                            "prompt_tokens": 25, "completion_tokens": completion_tokens})
        self.recent = self.recent[-50:]

    def finish_request(self, body):
        self.requests.append(body)
        count = len(self.requests)
        completion_tokens = 1 if self.scenario == "last_short_reply" and count == 6 else 16
        self.append_recent(body.get("origami_mode"), completion_tokens)
        if self.scenario == "external_request" and count == 3:
            self.append_recent("fast")
        elif self.scenario == "counter_reset" and count == 3:
            self.completed = 0
        elif self.scenario == "changed_model" and count == 6:
            self.key = "another-model@8192"
        elif self.scenario == "changed_settings" and count == 6:
            self.config_values["reasoning_effort"] = "low"
            self.config.write_text(json.dumps(self.config_values))
            self.protected[self.config] = self.config.read_bytes()
        elif self.scenario == "last_budget_change" and count == 6:
            self.budget = 5 * GIB // 2
        return completion_tokens


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
        if getattr(fixture, "down_until", 0) > time.monotonic():
            self.close_connection = True  # restarting: no answer at all
            return
        if self.path == "/health":
            self.reply({"status": "ready"})
        elif self.path == "/v1/dashboard":
            self.reply({"dashboard": {"tune_block": ""}})
        elif self.path == "/v1/status":
            state = fixture.state()
            fixture.after_status_snapshot()
            self.reply(state)
        else:
            fixture.unexpected.append(("GET", self.path))
            self.reply({"error": {"message": "Unexpected fixture route"}}, status=404)

    def do_POST(self):
        fixture = self.server.fixture
        if self.path != "/v1/chat/completions":
            fixture.unexpected.append(("POST", self.path))
            self.reply({"error": {"message": "Mutations are forbidden in the fixture"}}, status=405)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if hasattr(fixture, "chat"):
            fixture.chat(self, body)
            return
        completion_tokens = fixture.finish_request(body)
        events = [{"choices": [{"delta": {"content": "token "}}]} for _ in range(completion_tokens)]
        events.append({"choices": [], "usage": {"prompt_tokens": 25,
                       "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens": completion_tokens}})
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        self.reply((payload + "data: [DONE]\n\n").encode(), "text/event-stream")


def check_tradeoff_refusals(binary):
    """`--tradeoffs` says which cause refused it: an unknown floor, or a model with no fast level."""
    for floor, levels, wanted, code in (("bogus", LEVELS, "is not a quality level", 2),
                                        ("dynamic", ("exact",), "This model has no fast level", 1)):
        with tempfile.TemporaryDirectory(prefix="origami-tuning-flow-") as temporary:
            fixture = Fixture(pathlib.Path(temporary), "normal")
            state = fixture.state
            fixture.state = lambda: dict(state(), quality_levels=list(levels))
            server = Server(("127.0.0.1", 0), Handler)
            server.fixture = fixture
            fixture.configure(server.server_address[1])
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            env = {key: value for key, value in os.environ.items() if not key.startswith("ORIGAMI_")}
            env["ORIGAMI_TUNING_HOST"] = json.dumps(HOST)
            try:
                result = subprocess.run([binary, "tune", "--config", str(fixture.config), "--tradeoffs=true",
                                         "--quality-floor=" + floor, "--minutes=3"],
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            assert result.returncode == code and wanted in result.stdout, floor + ": " + result.stdout
            assert not fixture.requests, floor + ": measured anyway"


def run_scenario(binary, scenario):
    with tempfile.TemporaryDirectory(prefix="origami-tuning-flow-") as temporary:
        fixture = Fixture(pathlib.Path(temporary), scenario)
        server = Server(("127.0.0.1", 0), Handler)
        server.fixture = fixture
        fixture.configure(server.server_address[1])
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        env = {key: value for key, value in os.environ.items() if not key.startswith("ORIGAMI_")}
        env["ORIGAMI_TUNING_HOST"] = json.dumps(HOST)
        try:
            result = subprocess.run([binary, "tune", "--config", str(fixture.config),
                                     "--tradeoffs=true", "--quality-floor=dynamic", "--minutes=3"],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        output = result.stdout
        prefix = scenario + ": "
        assert result.returncode == 0, prefix + output
        outcomes = [json.loads(line[len(MARKER):]) for line in output.splitlines() if line.startswith(MARKER)]
        assert len(outcomes) == 1, prefix + "expected one explicit outcome marker\n" + output
        outcome = outcomes[0]
        assert not fixture.unexpected, prefix + repr(fixture.unexpected)
        for path, expected in fixture.protected.items():
            assert path.read_bytes() == expected, prefix + "CLI mutated " + path.name
        for body in fixture.requests:
            assert body.get("max_tokens") == 16, prefix + "comparison token count exceeded its bounded sample"
            assert body.get("origami_tune") == {}, prefix + "request was not isolated tuning"
            assert body.get("prompt_cache_policy") == "no_store", prefix + "request could pollute prompt cache"
            assert body.get("temperature") == 0, prefix + "comparison sampling changed"
            assert body.get("origami_mode") in LEVELS, prefix + "unexpected quality mode"
        history_path = fixture.data / "tuning-history.jsonl"
        assert history_path.exists(), prefix + "missing run history\n" + output
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        assert history[-1]["outcome"] == outcome["outcome"], prefix + "history and explicit outcome disagree"
        proposal_path = fixture.data / "tuning-proposal.json"

        if scenario in ("normal", "legacy_recent", "last_budget_change", "last_short_reply"):
            assert outcome["outcome"] == "comparison", prefix + output
            proposal = json.loads(proposal_path.read_text())
            assert proposal["kind"] == "tradeoffs" and proposal["key"] == fixture.key, prefix + repr(proposal)
            results = proposal["results"]
            assert {row["quality"] for row in results} == set(LEVELS), prefix + repr(results)
            assert all(row["samples"] == 2 for row in results), prefix + "warm-up or discarded samples leaked into result"
            assert all(math.isfinite(row["decode_tps"]) and row["decode_tps"] > 0 for row in results), prefix + repr(results)
            retried = scenario in ("last_budget_change", "last_short_reply")
            expected_count = 8 if retried else 6
            assert len(fixture.requests) == expected_count, prefix + repr(len(fixture.requests)) + "\n" + output
            orders = [[body["origami_mode"] for body in fixture.requests[start:start + 2]] for start in (0, 2, 4)]
            assert all(set(order) == set(LEVELS) for order in orders) and orders[1] == list(reversed(orders[2])), prefix + "comparison order is not balanced and rotated"
            for start in (2, 4):
                assert fixture.requests[start]["messages"] == fixture.requests[start + 1]["messages"], prefix + "modes in a balanced round saw different prompts"
            if retried:
                assert {body["origami_mode"] for body in fixture.requests[6:]} == set(LEVELS), prefix + "retry was not a complete balanced round"
                assert "retry" in output.lower() or "discard" in output.lower(), prefix + "disrupted final measurement was accepted silently"
        else:
            assert outcome["outcome"] in ("aborted", "incomplete"), prefix + output
            assert not proposal_path.exists(), prefix + "unsafe or stale comparison remains applicable"
            reason = str(outcome.get("reason", "")).lower()
            assert reason and reason in output.lower(), prefix + "specific abort reason is not visible\n" + output
            expected = {"external_request": ("request", "app"),
                        "between_samples": ("request", "app"),
                        "legacy_between_samples": ("request", "app"),
                        "counter_reset": ("restart", "changed", "request"),
                        "changed_model": ("model",), "changed_settings": ("settings",)}[scenario]
            assert any(word in reason for word in expected), prefix + repr(outcome)
            maximum = 3 if scenario in ("external_request", "between_samples", "legacy_between_samples", "counter_reset") else 6
            assert len(fixture.requests) <= maximum, prefix + "continued after the comparison became invalid"
            if scenario in ("between_samples", "legacy_between_samples"):
                assert fixture.injected_between_samples, prefix + "external completion injection was not exercised"
            assert "Comparison ready" not in output, prefix + "aborted run was presented as a success"
        print("ok   " + scenario, flush=True)


KNOBS = {name: {"value": True, "source": "default", "default": True} for name in ("gpu_keepalive", "family_spin")}
CANDIDATES = ({"gpu_keepalive": False}, {"family_spin": False})


class TuneFixture(Fixture):
    """A model with two timing switches. Each reply reports its own first-token and writing seconds (`check_timing`,
    read with `reported_timing`), so every decision is fixed by the check: `rate` tokens a second, twice that for
    `winner`. Only the time limit runs on the clock: each reply takes `pause` seconds, more after `slow_after` candidate
    requests; at `inject_at` candidate requests an app's request completes, or the server restarts."""

    def __init__(self, root, scenario, winner=None, slow_after=None, inject_at=None, rate=100.0, short=None):
        super().__init__(root, scenario)
        self.winner, self.slow_after, self.inject_at, self.rate = winner, slow_after, inject_at, rate
        self.short = short  # the candidate whose replies are one token long (each of its rounds is disrupted)
        self.pause, self.candidate_requests, self.down_until = 0.05, 0, 0

    def configure(self, port):
        self.config_values["port"] = port
        self.config.write_text(json.dumps(self.config_values))
        self.config.chmod(0o600)
        (self.data / "models.json").write_text('{"models": []}\n')

    def state(self):
        state = super().state()
        state["tuning"]["knobs"] = KNOBS
        return state

    def chat(self, handler, body):
        if self.down_until > time.monotonic():
            handler.close_connection = True
            return
        arm = body.get("origami_tune") or {}
        if arm:
            self.candidate_requests += 1
            if self.slow_after and self.candidate_requests > self.slow_after:
                self.pause = 1.5  # the rest can never fit the longest limit
            if self.inject_at == self.candidate_requests:
                if self.scenario == "restart":
                    self.down_until = time.monotonic() + 1.5
                    self.completed = 0  # a new process counts from zero
                    handler.close_connection = True
                    return
                self.append_recent("fast")  # an app's request completed meanwhile
        self.requests.append(body)
        tokens = 1 if self.short is not None and arm == self.short else body.get("max_tokens", 1)
        self.append_recent(body.get("origami_mode"), tokens)
        rate = self.rate * (2 if self.winner is not None and arm == self.winner else 1)
        prompt = len(body["messages"][0]["content"]) // 4
        timing = {"ttft": 0.1, "decode": (tokens - 1) / rate}
        usage = {"choices": [], "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": 0},
                                          "completion_tokens": tokens, "check_timing": timing}}
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in
                          ({"choices": [{"delta": {"content": "token " * tokens}}]}, usage)) + "data: [DONE]\n\n"
        time.sleep(self.pause)
        handler.reply(payload.encode(), "text/event-stream")


def run_tune(binary, root, scenario, minute, arguments=(), **options):
    fixture = TuneFixture(root, scenario, **options)
    server = Server(("127.0.0.1", 0), Handler)
    server.fixture = fixture
    fixture.configure(server.server_address[1])
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items() if not key.startswith("ORIGAMI_")}
    env["ORIGAMI_TUNING_HOST"] = json.dumps({**HOST, "minute_seconds": minute, "reported_timing": True})
    env["ORIGAMI_JOB"] = "1"  # as the dashboard runs it
    try:
        result = subprocess.run([binary, "tune", "--config", str(fixture.config), "--review=true", *arguments],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    prefix = scenario + ": "
    output = result.stdout
    assert result.returncode == 0, prefix + output
    assert not fixture.unexpected, prefix + repr(fixture.unexpected)
    outcomes = [json.loads(line[len(MARKER):]) for line in output.splitlines() if line.startswith(MARKER)]
    assert len(outcomes) == 1, prefix + "expected one explicit outcome marker\n" + output
    outcome = outcomes[0]
    # Every ending names what was measured, what was kept and what to do next.
    assert "Kept:" in outcome.get("reason", "") and "Next:" in outcome["reason"], prefix + "a bare ending\n" + output
    entries = json.loads((fixture.data / "tuning.json").read_text())["entries"] if (fixture.data / "tuning.json").exists() else {}
    return output, outcome, fixture, entries


def check_tune(binary):
    count = 0

    def ok(name):
        nonlocal count
        count += 1
        print("ok   tune " + name, flush=True)

    key, partial = "fixture-model@8192", "partial@fixture-model@8192"
    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        # Nothing faster on a default limit too short for this model: the limit grows, and "the defaults are fastest" is
        # the result, kept for the page.
        output, outcome, _, entries = run_tune(binary, pathlib.Path(temporary), "defaults_fastest", 1.2)
        assert outcome["outcome"] == "already_fastest" and "the defaults are fastest" in outcome["reason"], output
        assert "the default limit grows to" in output, output
        assert entries[key]["outcome"] == "already_fastest" and entries[key]["values"] == {} and partial not in entries, entries
        ok("no winner: success, the defaults are fastest, recorded")
        ok("the default limit grows to what the measurements need")
    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        output, outcome, _, entries = run_tune(binary, pathlib.Path(temporary), "winner", 6, winner=CANDIDATES[0])
        proposal = json.loads((pathlib.Path(temporary) / "data" / "tuning-proposal.json").read_text())
        assert outcome["outcome"] == "faster" and "GPU keep-alive off" in outcome["reason"], output
        assert proposal["entry"]["values"] == CANDIDATES[0] and key not in entries, (proposal, entries)
        ok("a winner waits for review")

    def resumed(root, first, kept):
        """The next run continues from the kept rounds, measuring none of them again, and decides."""
        output, outcome, fixture, entries = run_tune(binary, root, "resume", 6)
        assert "Kept measurements dropped" not in output, first + ": " + output
        assert "continuing: %d of 18 measurements kept from" % kept in output, first + ": " + output
        assert outcome["outcome"] == "already_fastest" and partial not in entries, first + ": " + output
        measured = [body.get("origami_tune") for body in fixture.requests if body.get("origami_tune")]
        rounds = len(measured) // 3
        assert rounds == (18 - kept) // 3, first + ": measured %d rounds again\n%s" % (rounds, output)

    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        root = pathlib.Path(temporary)
        output, outcome, fixture, entries = run_tune(binary, root, "owner_limit", 2, ["--minutes=3"])
        assert "your 3-minute limit fits 1 of 2 candidates; left out: spinning off" in output, output
        assert outcome["outcome"] == "incomplete" and "run tuning again" in outcome["reason"], output
        assert not any(body.get("origami_tune") == CANDIDATES[1] for body in fixture.requests), "a left-out candidate ran"
        kept = sum(len(arm["rounds"]) * 3 for arm in entries[partial]["arms"].values())
        assert kept >= 3 and key not in entries, entries
        resumed(root, "owner_limit", kept)
        ok("an owner's limit is kept, with fewer candidates, and the next run continues")
    for scenario, options, expected in (("slowdown", {"slow_after": 3}, "incomplete"),
                                        ("app_request", {"inject_at": 4}, "aborted"),
                                        ("restart", {"inject_at": 4}, "aborted")):
        with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
            root = pathlib.Path(temporary)
            output, outcome, _, entries = run_tune(binary, root, scenario, 1.2 if scenario == "slowdown" else 5, **options)
            assert outcome["outcome"] == expected, scenario + ": " + output
            kept = sum(len(arm["rounds"]) * 3 for arm in entries[partial]["arms"].values())
            reason = outcome["reason"]
            assert kept >= 3 and ("Kept: %d measurements" % kept in reason or "Kept: those measurements" in reason), scenario + ": " + output
            if scenario == "slowdown":
                assert "Time limit: tested" in outcome["reason"] and "Time limit reached" not in output, output
            resumed(root, scenario, kept)
            ok({"slowdown": "the limit reached mid-run decides from what it has, keeps it, and the next run continues",
                "app_request": "an app's request stops it, keeping its rounds; the next run continues",
                "restart": "a server restart stops it, keeping its rounds; the next run continues"}[scenario])
    # Every candidate decided by kept rounds, with a winner: the default limit grows for its confirmation alone.
    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        root = pathlib.Path(temporary)
        # Stopped by an app's request at the confirmation's first request (after 2 candidates x 3 rounds x 3 requests).
        output, outcome, _, entries = run_tune(binary, root, "confirm_stop", 6, winner=CANDIDATES[0], inject_at=19)
        kept = sum(len(arm["rounds"]) * 3 for arm in entries[partial]["arms"].values())
        assert outcome["outcome"] == "aborted" and kept == 18, output
        output, outcome, _, _ = run_tune(binary, root, "confirm_resume", 0.3, winner=CANDIDATES[0])
        assert "continuing: 18 of 18" in output and outcome["outcome"] == "faster" and "the default limit grows to" in output, output
        ok("a run continued with every candidate decided grows the default limit to confirm the winner")
    # A candidate left after two disrupted rounds: the ending says the disruption, not the time limit.
    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        output, outcome, _, _ = run_tune(binary, pathlib.Path(temporary), "disrupted", 6, ["--minutes=6"], short=CANDIDATES[0])
        reason = outcome["reason"]
        assert "Moving on without applying it" in output and outcome["outcome"] == "incomplete", output
        assert "Discarded comparisons: The reply was too short" in reason and "Time limit" not in reason and "longer limit" not in reason, output
        ok("a candidate left after disrupted rounds is said as such, not as the time limit")
    # Kept rounds from a Mac that has since drifted (its baseline more than 3% off) are dropped, not mixed in.
    with tempfile.TemporaryDirectory(prefix="origami-tune-flow-") as temporary:
        root = pathlib.Path(temporary)
        output, outcome, _, entries = run_tune(binary, root, "app_request", 5, inject_at=4)
        assert outcome["outcome"] == "aborted" and partial in entries, output
        output, outcome, fixture, entries = run_tune(binary, root, "drifted", 4, rate=110.0)
        assert "Kept measurements dropped: the baseline now writes 10% faster" in output, output
        measured = [body for body in fixture.requests if body.get("origami_tune")]
        assert outcome["outcome"] == "already_fastest" and len(measured) == 18 and partial not in entries, output
        ok("kept rounds from a Mac that drifted are dropped and measured again")
    return count


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: tuning-flow-checks.py /path/to/origami")
    binary = str(pathlib.Path(sys.argv[1]).resolve(strict=True))
    scenarios = ("normal", "legacy_recent", "external_request", "between_samples", "legacy_between_samples", "counter_reset",
                 "changed_model", "changed_settings", "last_budget_change", "last_short_reply")
    for scenario in scenarios:
        run_scenario(binary, scenario)
    check_tradeoff_refusals(binary)
    tune = check_tune(binary)
    print(f"{len(scenarios)} model-free tuning flow scenarios and {tune} tune endings passed")


if __name__ == "__main__":
    main()
