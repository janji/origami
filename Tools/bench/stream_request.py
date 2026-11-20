#!/usr/bin/env python3
"""Bounded, sequential OpenAI chat measurements; no server or model management.

Writes each exact request, received SSE frame and receipt into a new directory.
Token counts come from server usage, never from SSE chunk counts. The reported
wall throughput includes prefill; use server timings for decode-only throughput.
"""
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import hashlib
import json
import pathlib
import signal
import time
import urllib.error
import urllib.request


class CollectionInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__("Collection interrupted by " + signal.Signals(signum).name)


class CollectionSignals:
    """Turn CLI termination into a receipt, then defer repeats during flushing."""
    def __enter__(self):
        self.previous = {}
        self.interrupted = None
        self.deferring = 0
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            if previous != signal.SIG_IGN:
                self.previous[signum] = previous
                signal.signal(signum, self.interrupt)
        return self

    def interrupt(self, signum, _frame):
        if self.interrupted is None:
            self.interrupted = signum
            # Further signals return normally, allowing finally/summary writes.
            # The owner's existing kill deadline still bounds a stuck flush.
            if not self.deferring:
                raise CollectionInterrupted(signum)

    @contextmanager
    def flushing(self):
        self.deferring += 1
        try:
            yield
        finally:
            self.deferring -= 1

    def __exit__(self, *_):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def measure(url, body, directory, timeout, signals=None):
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    write_json(directory / "request.json", body)
    receipt = {
        "url": url,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "timeout_seconds": timeout,
    }
    request = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Accept": "text/event-stream",
    })
    frames, text, reasoning = [], [], []
    first, last, done = None, None, False
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            receipt["status"] = response.status
            receipt["headers"] = dict(response.headers.items())
            for raw in response:
                now = time.perf_counter() - started
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                frames.append({"seconds": now, "data": data})
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    receipt["usage"] = event["usage"]
                if event.get("timings"):
                    receipt["engine_timings"] = event["timings"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content or thought or delta.get("tool_calls"):
                        if first is None:
                            first = now
                        last = now
                    text.append(content)
                    reasoning.append(thought)
                    if choice.get("finish_reason") is not None:
                        receipt["finish_reason"] = choice["finish_reason"]
        if not done:
            raise RuntimeError("Stream ended without [DONE]")
    except Exception as error:
        receipt["error"] = str(error)
        if isinstance(error, CollectionInterrupted):
            receipt["interrupted_signal"] = int(error.signum)
        if isinstance(error, urllib.error.HTTPError):
            receipt["error_body"] = error.read(65536).decode("utf-8", errors="replace")
    finally:
        with signals.flushing() if signals is not None else nullcontext():
            receipt["wall_seconds"] = time.perf_counter() - started
            receipt["first_content_seconds"] = first
            receipt["last_content_seconds"] = last
            receipt["complete"] = done and "error" not in receipt
            count = receipt.get("usage", {}).get("completion_tokens")
            if isinstance(count, int) and receipt["complete"]:
                receipt["completion_tokens_per_wall_second"] = count / receipt["wall_seconds"]
            (directory / "frames.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in frames))
            (directory / "content.txt").write_text("".join(text))
            (directory / "reasoning.txt").write_text("".join(reasoning))
            write_json(directory / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full /v1/chat/completions URL")
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--repeat", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    with CollectionSignals() as signals:
        try:
            result = run(args, parser, signals)
            return 128 + int(signals.interrupted) if signals.interrupted is not None else result
        except CollectionInterrupted as error:
            # Also classify interruption before/after the active request. A
            # request receipt exists only once measure() owns that attempt.
            print(json.dumps({"complete": False, "interrupted_signal": int(error.signum),
                              "error": str(error)}), flush=True)
            return 128 + int(error.signum)


def run(args, parser, signals):
    body = json.loads(args.request.read_text())
    if not isinstance(body, dict) or body.get("n", 1) != 1:
        parser.error("A single-completion JSON request is required")
    body["stream"] = True
    body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []
    directory = None
    current_added = True

    def reconcile_interruption():
        if receipts and current_added and signals.interrupted is not None:
            receipt = receipts[-1]
            receipt.setdefault("error", str(CollectionInterrupted(signals.interrupted)))
            receipt.update(interrupted_signal=int(signals.interrupted), complete=False)
            receipt.pop("completion_tokens_per_wall_second", None)
            write_json(directory / "receipt.json", receipt)

    try:
        for index in range(args.repeat):
            if signals.interrupted is not None:
                break
            with signals.flushing():
                directory = args.output / ("request-%02d" % index)
                current_added = False
                directory.mkdir()
            if signals.interrupted is not None:
                break
            receipt = measure(args.url, body, directory, args.timeout, signals)
            with signals.flushing():
                receipts.append(receipt)
                current_added = True
                reconcile_interruption()
                print(json.dumps({"index": index, **receipt}), flush=True)
            if not receipt["complete"] or signals.interrupted is not None:
                break
    finally:
        with signals.flushing():
            # An interrupt after measure() flushed its result but before the
            # local result was appended must not discard that owned receipt.
            if directory is not None and not current_added and (directory / "receipt.json").is_file():
                receipts.append(json.loads((directory / "receipt.json").read_text()))
                current_added = True
            reconcile_interruption()
            write_json(args.output / "summary.json", receipts)
            # The first signal may arrive during the summary write itself.
            # Repeated signals are already deferred, so this second write is
            # stable once interruption has been observed.
            if signals.interrupted is not None:
                reconcile_interruption()
                write_json(args.output / "summary.json", receipts)
    return 0 if all(x["complete"] for x in receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
