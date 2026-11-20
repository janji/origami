#!/usr/bin/env python3
"""Sequential, isolated Origami before/candidate benchmark. Drain production first.

Every arm gets fresh config/data/context/store directories and one request. The
real HOME is never changed. --prepare-only writes launch plans without executing
Origami. The caller supplies binaries with their matching Swift resource bundles.
Optional per-arm speculation paths select a separate Qwen-only MTP experiment;
these receipts are excluded from the strict target-only comparator results.
An optional stage wall deadline interrupts active work; owned cleanup retains its
reserved grace and later arms cannot start after the same absolute cutoff.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error

from native_compare import (OwnedProcess, host_boundary_delta, host_boundary_snapshot,
                            host_memory_controls, http_json, memory_budget, owned_process_memory, save)

LIMIT = 25770000000
BUNDLES = ("swift-transformers_Hub.bundle", "swift-crypto_Crypto.bundle")


class WallDeadlineExceeded(Exception):
    """Ordinary exception, deliberately not OSError (HTTP polling catches that)."""


class DeferredSIGINT:
    """Deliver Ctrl-C after a short ownership/cleanup critical region finishes."""
    def __init__(self):
        self.pending = False
        self.installed = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self.previous = signal.getsignal(signal.SIGINT)
            if self.previous != signal.SIG_IGN:
                signal.signal(signal.SIGINT, self._remember)
                self.installed = True
        return self

    def _remember(self, *_):
        self.pending = True

    def __exit__(self, error_type, *_):
        if self.installed:
            signal.signal(signal.SIGINT, self.previous)
        if self.pending and error_type is None:
            raise KeyboardInterrupt()


def record_interruption(receipt):
    # Retain an earlier work/cleanup failure if interruption arrived afterward.
    if receipt.get("status") != "failed":
        receipt["status"] = "cancelled"
    receipt.setdefault("error", "Interrupted (KeyboardInterrupt/SIGINT)")
    receipt.update(interruption="KeyboardInterrupt", eligible_for_strict_gold_comparison=False)


def positive_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("deadline must be finite and positive")
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("deadline must be finite and positive")
    return seconds


class WallDeadline:
    """One absolute stage cutoff; active work is interruptible, owned cleanup is not.

    The one-shot alarm is armed only in work() and always disarmed before the
    caller's finally cleanup. Cleanup counts against the same monotonic cutoff,
    so a later arm cannot start after it. The caller reserves cleanup grace
    outside this active-work deadline; this class never signals another process.
    """
    def __init__(self, seconds=None):
        self.seconds = None if seconds is None else positive_seconds(seconds)
        self.ends_at = None if seconds is None else time.monotonic() + self.seconds
        self.expired = False
        self.active = False
        self.previous_handler = None
        self.installed = False

    def __enter__(self):
        if self.ends_at is not None:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("wall deadline requires the main thread")
            # Do not silently steal an unrelated process-wide alarm.
            if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
                raise RuntimeError("wall deadline cannot replace an existing real-time alarm")
            self.previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._alarm)
            self.installed = True
        return self

    def __exit__(self, *_):
        if self.installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.active = False
            signal.signal(signal.SIGALRM, self.previous_handler)
            self.installed = False

    def _alarm(self, *_):
        self.expired = True
        raise WallDeadlineExceeded("Stage wall deadline reached; no later arms may start")

    def check(self):
        if self.ends_at is not None and (self.expired or time.monotonic() >= self.ends_at):
            self.expired = True
            raise WallDeadlineExceeded("Stage wall deadline reached; no later arms may start")

    def _arm(self):
        self.check()
        signal.setitimer(signal.ITIMER_REAL, max(0.000001, self.ends_at - time.monotonic()))

    @contextmanager
    def work(self):
        if self.ends_at is None:
            yield
            return
        if not self.installed or self.active:
            raise RuntimeError("wall deadline needs one non-nested active work scope")
        self.check()
        self.active = True
        try:
            self._arm()
            yield
        finally:
            # An expiry even during this context exit propagates to execute's
            # outer finally. The timer is one-shot, so it cannot interrupt cleanup.
            self.active = False
            signal.setitimer(signal.ITIMER_REAL, 0)

    @contextmanager
    def assigning_owner(self):
        """Defer interruption until a freshly spawned process owner is assigned."""
        if not self.active:
            yield
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            yield
        finally:
            if time.monotonic() >= self.ends_at or self.expired:
                self.expired = True
            else:
                self._arm()
        # This check happens after the caller has assigned its local owner, so
        # execute's finally can stop it even if construction crossed the cutoff.
        self.check()


def validate_status(status, model, config, speculation="off", memory_limit=LIMIT, *, context=8192, allow_fast=False):
    if status.get("status") != "ready":
        raise ValueError("Origami is not ready")
    if status.get("config_file") != str(config):
        raise ValueError("server did not load the isolated config")
    loaded = status.get("model", {})
    if not isinstance(loaded, dict) or Path(loaded.get("path", "")).resolve() != Path(model).resolve():
        raise ValueError("loaded model does not match the selected checkpoint")
    for store in status.get("stores", []):
        if any(store.get(key) for key in ("pack_in_use", "copy_in_use", "mirror_in_use")):
            raise ValueError("an expert pack/copy/mirror entered the strict source-only comparison")
    settings = status.get("settings", {})
    for key, value in (("hot_copy", "off"), ("quality", "exact"), ("allow_fast", "true" if allow_fast else "false"),
                       ("speculation", speculation), ("context", str(context))):
        actual = settings.get(key, {}).get("value")
        if key == "context":
            if str(actual).lower() not in (str(context), str(context // 1024) + "k"):
                raise ValueError("effective context differs: " + str(actual))
        elif actual != value:
            raise ValueError("effective setting differs: " + key + "=" + str(actual))
    memory = status.get("memory", {})
    cap = memory.get("cap_bytes")
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= memory_limit:
        raise ValueError("reported memory cap absent or above the requested memory envelope")


def execute(plan, directory, request_path, encoded, args, wall_deadline=None):
    wall_deadline = wall_deadline or WallDeadline()
    receipt = {**plan, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    server = None
    collector = None
    try:
        with wall_deadline.work():
            missing = [str(Path(plan["binary"]).parent / name) for name in BUNDLES
                       if not (Path(plan["binary"]).parent / name).is_dir()]
            if missing:
                raise ValueError("Copy matching resource bundles beside this binary before launch: " + ", ".join(missing))
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", args.port))
            env = {k: v for k, v in os.environ.items() if not k.startswith("ORIGAMI_")}
            env.update(plan["explicit_environment"])
            receipt["host_boundary_before"] = host_boundary_snapshot()
            save(directory / "host-before.json", receipt["host_boundary_before"])
            with DeferredSIGINT(), wall_deadline.assigning_owner():
                server = OwnedProcess(plan["argv"], str(Path(plan["binary"]).parent), env, directory)
            receipt["server_pid"] = server.p.pid
            base = "http://127.0.0.1:" + str(args.port)
            while True:
                if server.p.poll() is not None:
                    raise RuntimeError("Origami exited during loading: " + str(server.p.returncode))
                if time.monotonic() - server.started > args.load_deadline:
                    raise TimeoutError("Origami load deadline")
                try:
                    health = http_json(base + "/health", timeout=2)
                    if health.get("status") == "ready":
                        break
                    if health.get("status") in ("failed", "error"):
                        raise RuntimeError("Origami load failed: " + json.dumps(health))
                except urllib.error.HTTPError as error:
                    failure = error.read(65536).decode("utf-8", "replace")
                    if error.code == 503 and ('"failed"' in failure or '"error"' in failure):
                        raise RuntimeError("Origami failed while loading: " + failure)
                except OSError:
                    pass
                time.sleep(0.2)
            receipt["ready_seconds"] = time.monotonic() - server.started
            before = http_json(base + "/v1/status")
            save(directory / "status-before.json", before)
            validate_status(before, plan["model"], directory / "config.json", plan["speculation"], plan["memory_limit_bytes"])
            if plan["mtp"]["enabled"]:
                startup = (directory / "stderr.log").read_text()
                if "speculation: MTP head from " + Path(plan["speculation"]).name not in startup:
                    raise ValueError("MTP was requested but no successful head load was logged; inspect stderr")
            collection = directory / "collector"
            collection.mkdir()
            collector_argv = [sys.executable, "-B", str(Path(__file__).resolve().with_name("stream_request.py")),
                              "--url", base + "/v1/chat/completions", "--request", str(request_path),
                              "--output", str(directory / "request"), "--repeat", "1", "--timeout", str(args.request_deadline)]
            with DeferredSIGINT(), wall_deadline.assigning_owner():
                collector = OwnedProcess(collector_argv, str(directory), os.environ.copy(), collection)
            try:
                collector.p.wait(timeout=args.request_deadline)
            except subprocess.TimeoutExpired:
                raise TimeoutError("Origami request hard deadline")
            collector.stop()
            summary = json.loads((directory / "request/summary.json").read_text())
            if len(summary) != 1 or not summary[0].get("complete") or collector.p.returncode:
                raise ValueError("collector did not receive one complete request")
            result = summary[0]
            usage = result.get("usage", {})
            if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
                raise ValueError("prefix reuse entered the cold-prompt comparison")
            if encoded and usage.get("prompt_tokens") != encoded["prompt_tokens"]:
                raise ValueError("Origami prompt count differs from the exact tokenizer export")
            tokens = usage.get("origami_tokens")
            if not isinstance(tokens, list) or len(tokens) != usage.get("completion_tokens"):
                raise ValueError("Origami output token IDs/count disagree or are absent")
            receipt.update({"request_receipt": result, "output_ids": tokens, "output_tokens": len(tokens),
                            "count_reached": len(tokens) == args.tokens,
                            "decode_token_intervals": max(0, len(tokens) - 1)})
            if not plan["mtp"]["enabled"]:
                receipt["decode_forward_steps"] = max(0, len(tokens) - 1)
            after = http_json(base + "/v1/status")
            save(directory / "status-after.json", after)
            validate_status(after, plan["model"], directory / "config.json", plan["speculation"], plan["memory_limit_bytes"])
            if "--single-model" not in plan["argv"]:
                raise ValueError("Cannot attribute memory to a model process without --single-model")
            resources = owned_process_memory(server, plan["binary"])
            receipt["process_resources"] = resources
            if resources["measured"]:
                receipt["memory_observed_within_budget"] = resources["peak_footprint_bytes"] <= plan["memory_limit_bytes"]
                if not receipt["memory_observed_within_budget"]:
                    raise ValueError("Origami model-process lifetime peak exceeds the requested memory envelope")
            else:
                receipt["memory_observed_within_budget"] = None
                receipt["eligible_for_strict_gold_comparison"] = False
                receipt["memory_qualification"] = "Model-process peak unavailable; do not claim measured equal-memory fairness."
            stderr = (directory / "stderr.log").read_text()
            if plan["mtp"]["enabled"]:
                receipt["mtp_diagnostics"] = [line for line in stderr.splitlines() if "speculation:" in line]
                stats = re.findall(r"speculation: (\d+) drafts, acceptance ([\d.]+), pair step ([\d.]+) ms, plain step ([\d.]+) ms( \(paused\))?", stderr)
                if stats:
                    attempts, acceptance, pair_ms, plain_ms, paused = stats[-1]
                    receipt["mtp_measurement"] = {"draft_attempts": int(attempts), "acceptance_rounded": float(acceptance),
                                                   "pair_step_ms_rounded": float(pair_ms), "plain_step_ms_rounded": float(plain_ms),
                                                   "paused": bool(paused)}
            native = re.findall(r"reply: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", stderr)
            if native:
                count, seconds, rate = native[-1]
                if int(count) != len(tokens):
                    raise ValueError("native decode log disagrees with returned token count")
                receipt["native_decode"] = {"output_tokens": int(count), "seconds_rounded": float(seconds),
                                            "tokens_per_second": float(rate), "forward_steps": max(0, len(tokens) - 1),
                                            "note": "Before/candidate family binaries both use n-1 decode forwards; retain rounded seconds and native rate."}
            receipt["status"] = "completed"
    except KeyboardInterrupt:
        # SIGINT is an expected operator/resource-gate cancellation. Returning a
        # terminal receipt after cleanup lets run() persist its stage summary
        # before exiting, without mistaking a partial attempt for eligible work.
        record_interruption(receipt)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        receipt["eligible_for_strict_gold_comparison"] = False
    finally:
        try:
            with DeferredSIGINT():
                cleanup_errors = []

                def cleanup_step(name, action):
                    try:
                        return action()
                    except KeyboardInterrupt:
                        # Also tolerate an explicit KeyboardInterrupt raised by
                        # a helper: every independent retirement still runs.
                        record_interruption(receipt)
                        return None
                    except Exception as error:
                        cleanup_errors.append({"step": name, "type": type(error).__name__, "error": str(error)})
                        return None

                # A failed collector retirement must not strand the model process
                # or discard the primary work failure. Defer SIGINT until all
                # owners are retired and the first receipt has been persisted.
                if collector:
                    cleanup_step("collector.stop", collector.stop)
                if server:
                    cleanup_step("server.stop", server.stop)
                    receipt["server_exit_code"] = server.p.returncode
                    after_boundary = cleanup_step("host_boundary_snapshot", host_boundary_snapshot)
                    if after_boundary is not None:
                        receipt["host_boundary_after"] = after_boundary
                        cleanup_step("host-after.json", lambda: save(directory / "host-after.json", after_boundary))
                        delta = cleanup_step("host_boundary_delta", lambda: host_boundary_delta(
                            receipt["host_boundary_before"], after_boundary))
                        if delta is not None:
                            receipt["host_boundary_delta"] = delta
                if cleanup_errors:
                    receipt["cleanup_errors"] = cleanup_errors
                    receipt["status"] = "failed"
                    receipt.setdefault("error", "Cleanup or boundary evidence failed; see cleanup_errors")
                    receipt["eligible_for_strict_gold_comparison"] = False
                receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                save(directory / "receipt.json", receipt)
        except KeyboardInterrupt:
            record_interruption(receipt)
            receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(directory / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--before-speculation", type=Path, help="Qwen-only NUMERIC experiment: MTP safetensors for the before arm")
    parser.add_argument("--candidate-speculation", type=Path, help="Qwen-only NUMERIC experiment: MTP safetensors for the candidate arm")
    parser.add_argument("--families", default="qwen,glm,deepseek")
    parser.add_argument("--arms", default="before,candidate", help="ordered arms; repeat names for ABBA if wanted")
    parser.add_argument("--candidate-env", action="append", default=[], help="explicit ORIGAMI_FAMILY_* or ORIGAMI_EXECUTION=JSON override")
    parser.add_argument("--output", type=Path, required=True, help="new isolated output root")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--encoded-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=11449)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--memory-gb", default="25.77", help="total memory envelope in exact decimal GB (default: 25.77)")
    parser.add_argument("--load-deadline", type=positive_seconds, default=360)
    parser.add_argument("--request-deadline", type=positive_seconds, default=300)
    parser.add_argument("--wall-deadline-seconds", type=positive_seconds,
                        help="absolute stage active-work cutoff; owned cleanup has reserved grace (default: none)")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        # Preparation writes plans only and must never start a runtime timer or model.
        with WallDeadline(None if args.prepare_only else args.wall_deadline_seconds) as wall_deadline:
            return run(args, parser, wall_deadline)
    except WallDeadlineExceeded as error:
        print(json.dumps({"status": "wall_deadline", "error": str(error)}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Covers interruption outside execute(), such as between prepared arms.
        print(json.dumps({"status": "cancelled", "error": "Interrupted (KeyboardInterrupt/SIGINT)"}), file=sys.stderr)
        return 130


def run(args, parser, wall_deadline):
    with wall_deadline.work():
        try:
            memory_gb, memory_limit = memory_budget(args.memory_gb)
        except ValueError as error:
            parser.error(str(error))
        families, arms = args.families.split(","), args.arms.split(",")
        if any(f not in ("qwen", "glm", "deepseek") for f in families) or any(a not in ("before", "candidate") for a in arms):
            parser.error("unknown family or arm")
        speculation_paths = {"before": args.before_speculation, "candidate": args.candidate_speculation}
        if any(speculation_paths.values()) and any(f != "qwen" for f in families):
            parser.error("speculation experiments require --families qwen exclusively")
        mtp_metadata = {}
        for arm, path in speculation_paths.items():
            if path is None:
                mtp_metadata[arm] = {"enabled": False}
                continue
            path = path.resolve()
            if not path.is_file():
                parser.error("MTP safetensors missing: " + str(path))
            speculation_paths[arm] = path
            mtp_metadata[arm] = {"enabled": True, "path": str(path), "file_bytes": path.stat().st_size}
            provenance_path = path.with_suffix(".provenance.json")
            if provenance_path.is_file():
                if provenance_path.stat().st_size > 1_000_000:
                    parser.error("MTP provenance exceeds the metadata-only read limit")
                provenance = json.loads(provenance_path.read_text())
                if provenance.get("file_bytes") != path.stat().st_size:
                    parser.error("MTP file size differs from its provenance receipt")
                mtp_metadata[arm].update({"provenance_file": str(provenance_path),
                                          "source_repo": provenance.get("source_repo"),
                                          "source_revision": provenance.get("source_revision"),
                                          "recorded_sha256": provenance.get("file_sha256"),
                                          "integrity_note": "Recorded digest from extraction verification; large head is not rehashed during benchmark setup."})
        mtp_experiment = any(speculation_paths[arm] is not None for arm in arms)
        if not 1 <= args.tokens <= 256 or not 1024 <= args.port <= 65535 or args.port == 11439:
            parser.error("tokens must be 1..256 and port a private non-production port")
        overrides = {}
        for item in args.candidate_env:
            key, sep, value = item.partition("=")
            if not sep or not (key.startswith("ORIGAMI_FAMILY_") or key == "ORIGAMI_EXECUTION"):
                parser.error("candidate-env accepts only explicit engine experiment knobs")
            overrides[key] = value
        body = json.loads(args.request.read_text())
        body.update({"model": "origami", "temperature": 0, "top_p": 1, "seed": 37, "max_tokens": args.tokens,
                     "top_k": 0, "min_p": 0, "presence_penalty": 0,
                     "reasoning_effort": "none", "origami_mode": "exact", "origami_return_tokens": True,
                     "prompt_cache_policy": "no_store", "origami_tune": {}})
        if body.get("n", 1) != 1 or body.get("tools") or body.get("origami_teacher_tokens"):
            parser.error("use a single plain exact completion for this strict baseline comparison")
        manifest = json.loads(args.manifest.read_text())
        models = {v["family"]: v["model"] for v in manifest["engines"].values()}
        for family in families:
            if not Path(models[family]).exists():
                parser.error("selected checkpoint is unavailable; check its mounted volume: " + models[family])
        binaries = {"before": args.before.resolve(), "candidate": args.candidate.resolve()}
        for arm in set(arms):
            if not binaries[arm].is_file():
                parser.error("binary missing: " + str(binaries[arm]))
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        request_path = output / "request.json"
        save(request_path, body)
        receipts = []
    for family in families:
        wall_deadline.check()
        encoded_file = args.encoded_dir / (family + ".json")
        encoded = json.loads(encoded_file.read_text()) if encoded_file.is_file() else None
        for index, arm in enumerate(arms):
            wall_deadline.check()
            with wall_deadline.work():
                directory = output / (family + "-" + str(index) + "-" + arm)
                directory.mkdir()
                for name in ("data", "context", "empty-store", "empty-hot-copy"):
                    (directory / name).mkdir()
                save(directory / "config.json", {"stores": []})
                binary = binaries[arm]
                speculation = str(speculation_paths[arm]) if speculation_paths[arm] is not None else "off"
                argv = [str(binary), "serve", "--single-model", "--config", str(directory / "config.json"),
                        "--model", models[family], "--port", str(args.port), "--listen", "local",
                        "--data-dir", str(directory / "data"), "--context-dir", str(directory / "context"),
                        "--store", str(directory / "empty-store"), "--hot-copy-dir", str(directory / "empty-hot-copy"),
                        "--context", "8192", "--memory-limit", memory_gb + "GB", "--hot-copy", "off", "--speculation", speculation,
                        "--allow-fast", "false", "--quality", "exact", "--reasoning-effort", "none",
                        "--context-on-disk", "false", "--commands", "false", "--list-models", "false",
                        "--status-messages", "false", "--request-log", "off"]
                plan = {"family": family, "arm": arm, "binary": str(binary), "model": models[family], "argv": argv,
                        "explicit_environment": overrides if arm == "candidate" else {},
                        "memory_limit_bytes": memory_limit, "memory_budget_decimal_gb": memory_gb,
                        "context": 8192, "requested_tokens": args.tokens,
                        "wall_deadline_seconds": args.wall_deadline_seconds,
                        "host_memory_controls": host_memory_controls(),
                        "speculation": speculation, "mtp": mtp_metadata[arm],
                        "comparison_class": "mtp_experiment" if mtp_experiment else "strict_target_only",
                        "eligible_for_strict_gold_comparison": not mtp_experiment,
                        "arithmetic_class": "NUMERIC" if mtp_metadata[arm]["enabled"] else "ordinary_target_decode",
                        "arithmetic_note": ("MTP two-row verification changes floating-point grouping relative to ordinary decode. "
                                            "Measure output parity; no bitwise equivalence is assumed. These arms are excluded from the strict gold comparison."
                                            if mtp_experiment else "Speculation disabled for the strict source-only comparison."),
                        "cache_condition": "new isolated process and empty data/context/stores; OS page cache uncontrolled",
                        "resource_bundles": {name: (binary.parent / name).is_dir() for name in BUNDLES},
                        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
                save(directory / "launch.json", plan)
            result = {**plan, "status": "prepared"} if args.prepare_only else execute(plan, directory, request_path, encoded, args, wall_deadline)
            receipts.append(result)
            save(output / "summary.json", receipts)
            print(json.dumps({"family": family, "arm": arm, "status": result["status"],
                              "directory": str(directory), "error": result.get("error")}), flush=True)
            if result.get("interruption") == "KeyboardInterrupt" or result["status"] == "cancelled":
                return 130
            if result["status"] == "failed":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
