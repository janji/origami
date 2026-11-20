#!/usr/bin/env python3
"""Bounded host watcher for an explicitly identified, already-owned comparator.

Never launches models. On a stop gate, verifies the comparator again and sends
one SIGINT to that PID; the comparator owns model/collector cleanup. A prelaunch
host baseline is mandatory. Pageouts may be marked ineligible by explicit policy;
swap growth, pressure, thermal, memory, disk and missing observations still stop.
"""
import argparse
import ctypes
import datetime
import errno
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time

from family_features import thermal_clear
from native_compare import host_boundary_snapshot, kernel_process_memory
from origami_compare import WallDeadline

MAX_JSONL = 16 << 20
MAX_SAMPLES = 2048
STOP_RESERVE_SECONDS = 15  # Covers probe expiry, bounded helper cleanup and identity recheck.
UTC = datetime.timezone.utc


def utc(value):
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include its timezone")
    return parsed.astimezone(UTC)


def bounded_command(argv, timeout=2, maximum=1 << 20):
    """Small owned read-only probes; cap both capture bytes and wall duration."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for name in streams:
                selector.register(getattr(process, name), selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("read-only probe exceeded deadline")
                for key, _ in selector.select(remaining):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    else:
                        streams[key.data].extend(data)
                        if sum(map(len, streams.values())) > maximum:
                            raise ValueError("read-only probe exceeded output bound")
        status = process.wait(timeout=max(.001, deadline - time.monotonic()))
        if status:
            raise RuntimeError("read-only probe failed with exit " + str(status))
        return streams["stdout"].decode("utf-8", "strict")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        process.stdout.close()
        process.stderr.close()


def executable_path(pid):
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    lib.proc_pidpath.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    if lib.proc_pidpath(pid, buffer, len(buffer)) <= 0:
        raise OSError(ctypes.get_errno(), "proc_pidpath failed")
    return str(Path(os.fsdecode(buffer.value)).resolve())


def option(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv):
        raise ValueError("required process option absent or ambiguous: " + flag)
    return argv[argv.index(flag) + 1]


def process_arguments(pid):
    # Only called for an executable whose identity was already verified. Raw
    # process arguments are never emitted (they could contain unrelated values).
    return shlex.split(bounded_command(["/bin/ps", "-ww", "-p", str(pid), "-o", "args="], maximum=65536))


class SystemProbe:
    def __init__(self, args):
        self.args = args
        self.children_seen = {}
        self.model_owners = {}

    def comparator(self, check_arguments=True):
        a = self.args
        info = again = None
        def retained(error=None):
            result = []
            for query in (info, again):
                if query is not None:
                    result.extend(query.get("memory_observations", []))
            if error is not None:
                result.extend(getattr(error, "memory_observations", []))
            return result
        try:
            info = kernel_process_memory(a.comparator_pid, a.comparator_executable)
            if info["process_start_abstime"] != a.comparator_start_abstime:
                return {"state": "identity_changed", "reason": "comparator birth time differs", "memory_observations": retained()}
            if check_arguments:
                argv = process_arguments(a.comparator_pid)
                if (str(Path(a.comparator_script).resolve()) not in argv or
                        Path(option(argv, "--output")).resolve() != Path(a.comparator_output).resolve()):
                    return {"state": "identity_changed", "reason": "comparator script/output differs", "memory_observations": retained()}
                # Arg inspection is a separate probe: reject exit/reuse/exec that
                # happened while it was running, before declaring signal safety.
                again = kernel_process_memory(a.comparator_pid, a.comparator_executable)
                if again["process_start_abstime"] != a.comparator_start_abstime:
                    return {"state": "identity_changed", "reason": "comparator changed during argument probe", "memory_observations": retained()}
            return {"state": "verified", "pid": a.comparator_pid,
                    "start_abstime": info["process_start_abstime"], "executable": info["verified_executable"]}
        except OSError as error:
            if error.errno == errno.ESRCH:
                return {"state": "gone", "memory_observations": retained(error)}
            # An unavailable identity is not permission to signal a PID.
            return {"state": "unverified", "reason": str(error),
                    "memory_observations": retained(error)}
        except Exception as error:
            return {"state": "unverified", "reason": str(error),
                    "memory_observations": retained(error)}

    def sample(self):
        a = self.args
        host = host_boundary_snapshot()  # Three fixed probes, each bounded at3s.
        pressure = bounded_command(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], maximum=4096)
        models = []
        listing = bounded_command(["/bin/ps", "-axo", "pid=,ppid="])
        candidates = []
        for line in listing.splitlines():
            pid, parent = map(int, line.split())
            if parent == a.comparator_pid:
                candidates.append(pid)
        if len(candidates) > 64:
            raise ValueError("unexpected number of comparator children")
        allowed = dict(zip(a.model_executable, a.model_config))
        for pid in candidates:
            info = again = None
            try:
                path = executable_path(pid)
                if path not in allowed:
                    continue
                info = kernel_process_memory(pid, path)
                parent = int(bounded_command(["/bin/ps", "-p", str(pid), "-o", "ppid="], maximum=4096).strip())
                if parent != a.comparator_pid or os.getpgid(pid) != pid or os.getsid(pid) != pid:
                    raise ValueError("model is not the comparator's direct session-leader child")
                argv = process_arguments(pid)
                if "--single-model" not in argv or Path(option(argv, "--config")).resolve() != Path(allowed[path]).resolve():
                    raise ValueError("model's isolated config differs")
                birth = info["process_start_abstime"]
                again = kernel_process_memory(pid, path)
                if again["process_start_abstime"] != birth:
                    raise ValueError("model changed during argument probe")
                if pid in self.children_seen and self.children_seen[pid] != birth:
                    raise ValueError("model PID reused; refusing to adopt it")
                if path in self.model_owners and self.model_owners[path] != (pid, birth):
                    raise ValueError("unexpected model respawn; refusing to adopt it")
                self.children_seen[pid] = birth
                self.model_owners[path] = (pid, birth)
                # The argument/identity recheck is another memory observation. Its
                # larger values must participate in the cap; retain both queries.
                models.append({**again, "pid": pid, "verified_parent_pid": parent, "verified_session_id": pid,
                               "peak_footprint_bytes": max(info["peak_footprint_bytes"], again["peak_footprint_bytes"]),
                               "kernel_reported_lifetime_peak_bytes": max(
                                   info.get("kernel_reported_lifetime_peak_bytes", info["peak_footprint_bytes"]),
                                   again.get("kernel_reported_lifetime_peak_bytes", again["peak_footprint_bytes"])),
                               "memory_observations": info.get("memory_observations", []) + again.get("memory_observations", []),
                               "memory_snapshot_attempts": info.get("memory_snapshot_attempts", 1) + again.get("memory_snapshot_attempts", 1),
                               "memory_observation_queries": [info, again],
                               "peak_measurement_scope": "maximum of both verified memory queries; each query retains its non-atomic current/lifetime-peak observations"})
            except Exception as error:
                retained = []
                for query in (info, again):
                    if query is not None:
                        retained.extend(query.get("memory_observations", []))
                retained.extend(getattr(error, "memory_observations", []))
                if retained:
                    error.memory_observations = retained
                if isinstance(error, OSError) and error.errno == errno.ESRCH and info is None and not retained:
                    continue  # No memory observation to discard; process already retired.
                raise
        if len(models) > 1:
            raise ValueError("multiple simultaneous model children")
        return {"host": host, "pressure_level": int(pressure.strip()),
                "disk_free_bytes": shutil.disk_usage(a.disk_path).free, "models": models}

    def interrupt(self):
        identity = self.comparator()  # Recheck executable, birth and argv immediately before signaling.
        if identity["state"] != "verified":
            return {"sent": False, "identity": identity}
        try:
            os.kill(self.args.comparator_pid, signal.SIGINT)
            return {"sent": True, "signal": "SIGINT", "identity": identity}
        except ProcessLookupError:
            return {"sent": False, "identity": {"state": "gone"}}
        except OSError as error:
            return {"sent": False, "identity": identity, "error": str(error)}

    def terminal_outcome(self):
        path = Path(self.args.comparator_output) / "summary.json"
        try:
            if path.stat().st_size > 4 << 20:
                raise ValueError("terminal summary exceeds metadata bound")
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or len(rows) != len(self.args.model_executable):
                raise ValueError("terminal summary has an unexpected arm count")
            complete = (isinstance(rows, list) and bool(rows) and
                        all(isinstance(row, dict) and row.get("status") == "completed" for row in rows))
            return {"complete": complete, "summary": str(path),
                    "statuses": [row.get("status") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []}
        except Exception as error:
            return {"complete": False, "summary": str(path), "error": str(error)}


def assess(sample, baseline, args):
    stops, ineligible, delta = [], [], {}
    host = sample.get("host", {})
    for name in ("vm_stat", "swap_usage", "thermal"):
        raw = host.get("raw", {}).get(name, {})
        if raw.get("returncode") != 0 or not raw.get("stdout", "").strip() or raw.get("error"):
            stops.append("missing_" + name)
    for name in ("pageouts", "swapouts", "swap_used_bytes"):
        current, previous = host.get("counters", {}).get(name), baseline.get("counters", {}).get(name)
        if type(current) is not int or type(previous) is not int or min(current, previous) < 0:
            stops.append("missing_" + name)
            continue
        delta[name] = current - previous
        if delta[name] < 0 and name != "swap_used_bytes":
            stops.append("counter_reset_" + name)
        elif delta[name] > 0:
            ineligible.append("increased_" + name)
            if name != "pageouts" or args.paging_policy == "abort":
                stops.append("increased_" + name)
    if sample.get("pressure_level") != 1:
        stops.append("pressure_not_normal")
    if not thermal_clear(host.get("raw", {}).get("thermal", {}).get("stdout")):
        stops.append("thermal_not_verified_normal")
    free = sample.get("disk_free_bytes")
    if type(free) is not int or free < args.disk_floor_bytes:
        stops.append("disk_floor_or_missing_measurement")
    for model in sample.get("models", []):
        peak = model.get("peak_footprint_bytes")
        if model.get("measured") is not True or type(peak) is not int or peak <= 0:
            stops.append("missing_model_memory")
        elif peak > args.memory_cap_bytes:
            stops.append("model_memory_cap")
    return {"stop_reasons": sorted(set(stops)), "performance_ineligible_reasons": sorted(set(ineligible)),
            "baseline_deltas": delta, "paging_classification": "increase" if ineligible else (
                "unknown" if any(x.startswith(("missing_", "counter_reset_")) for x in stops) else "no_observed_increase")}


class JSONL:
    def __init__(self, path, maximum=MAX_JSONL):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, "wb")
        self.written = 0
        self.maximum = maximum

    def emit(self, row, terminal=False):
        data = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
        limit = self.maximum if terminal else self.maximum - 16384
        if self.written + len(data) > limit:
            raise ValueError("watcher output bound reached")
        self.file.write(data)
        self.file.flush()
        self.written += len(data)


def monitor(args, baseline, probe, emit, clock=time.monotonic,
            now=lambda: datetime.datetime.now(UTC), sleep=time.sleep):
    start = clock()
    monotonic_end = start + max(0, (args.end_utc - now()).total_seconds())
    next_due, count, model_absent_since, model_seen = start, 0, None, False
    ineligible, reasons = set(), []
    def terminal(row):
        try:
            emit(row, terminal=True)
            return True
        except Exception as error:
            # Do not lose the stop outcome when the evidence disk itself fails.
            # One small stderr fallback; never print the discarded raw sample.
            print(json.dumps({"kind": "terminal_write_failed", "state": row.get("state"),
                              "stop_reasons": row.get("stop_reasons"), "signal": row.get("signal"),
                              "error": str(error)[:512]}), file=sys.stderr)
            return False
    try:
        emit({"kind": "header", "schema": "origami-resource-watcher-v1", "utc": now().isoformat(),
              "end_utc": args.end_utc.isoformat(), "interval_seconds": args.interval_seconds,
              "stop_reserve_seconds": STOP_RESERVE_SECONDS,
              "performance_qualification": "Root must reconcile final comparator receipts and host boundaries; samples alone cannot qualify performance.",
              "paging_policy": args.paging_policy, "baseline": {"utc": baseline["utc"], "counters": {
                  k: baseline["counters"][k] for k in ("pageouts", "swapouts", "swap_used_bytes")}},
              "comparator_pid": args.comparator_pid, "comparator_start_abstime": args.comparator_start_abstime,
              "memory_cap_bytes": args.memory_cap_bytes, "disk_floor_bytes": args.disk_floor_bytes})
    except Exception:
        reasons = ["header_write_failure"]
    while not reasons:
        if ((args.end_utc - now()).total_seconds() <= STOP_RESERVE_SECONDS or
                monotonic_end - clock() <= STOP_RESERVE_SECONDS):
            reasons = ["absolute_deadline"]
            break
        if count >= MAX_SAMPLES:
            reasons = ["sample_limit"]
            break
        if clock() < next_due:
            sleep(min(next_due - clock(), max(0, monotonic_end - clock() - STOP_RESERVE_SECONDS)))
            continue
        begun, stamp = clock(), now().isoformat()
        try:
            # Bound the complete probe round, not merely each subprocess. The
            # fixed host probes each also carry their own3second timeout.
            with WallDeadline(max(.001, min(10, monotonic_end - begun))) as deadline:
                with deadline.work():
                    identity = probe.comparator()
                    sample = probe.sample() if identity["state"] == "verified" else None
                    outcome = probe.terminal_outcome() if identity["state"] == "gone" else {"complete": False}
            if identity["state"] != "verified":
                complete = outcome["complete"] and count > 0 and model_seen
                written = terminal({"kind": "final", "state": "comparator_" + identity["state"], "identity": identity,
                      "terminal_outcome": outcome, "samples": count,
                      "observed_stop_gates_clear": complete,
                      "performance_eligible": None, "performance_qualification": "requires_root_review_of_final_boundaries_and_receipts",
                      "performance_ineligible_reasons": sorted(ineligible)})
                return 0 if complete and written else 2
            result = assess(sample, baseline, args)
            if now() >= args.end_utc or clock() >= monotonic_end:
                result["stop_reasons"].append("absolute_deadline")
            if not sample.get("models"):
                model_absent_since = begun if model_absent_since is None else model_absent_since
                if clock() - model_absent_since >= args.model_discovery_grace_seconds:
                    result["stop_reasons"].append("model_not_observed_within_grace")
            else:
                model_absent_since = None
                model_seen = True
        except Exception as error:
            sample = {"error": str(error)}
            if hasattr(error, "memory_observations"):
                sample["memory_observations"] = error.memory_observations
            result = {"stop_reasons": ["missing_measurement"], "performance_ineligible_reasons": ["missing_measurement"]}
        ineligible.update(result["performance_ineligible_reasons"])
        count += 1
        row = {"kind": "sample", "index": count - 1, "utc": stamp, "elapsed_seconds": begun - start,
               "observation_seconds": clock() - begun, **sample, **result}
        try:
            emit(row)
        except Exception:
            reasons = ["output_limit_or_write_failure"]
            break
        if result["stop_reasons"]:
            reasons = result["stop_reasons"]
            break
        next_due += args.interval_seconds  # Start-to-start cadence, not interval after observation ends.
        if next_due < clock():
            next_due = clock()  # No burst: at most one immediate next capture after a slow sample.
    signal_result = probe.interrupt()
    retire_by = clock() + args.cleanup_grace_seconds
    final_identity = signal_result["identity"]
    while signal_result["sent"] and clock() < retire_by:
        final_identity = probe.comparator(check_arguments=False)
        if final_identity["state"] != "verified":
            break
        sleep(min(1, max(0, retire_by - clock())))
    terminal({"kind": "final", "state": "stop_requested" if signal_result["sent"] else "stop_not_sent",
          "utc": now().isoformat(), "stop_reasons": reasons,
          "signal": signal_result, "final_comparator_identity": final_identity, "samples": count,
          "performance_eligible": False, "performance_ineligible_reasons": sorted(ineligible)})
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-pid", required=True, type=int)
    parser.add_argument("--comparator-executable", required=True)
    parser.add_argument("--comparator-start-abstime", required=True, type=int)
    parser.add_argument("--comparator-started-utc", required=True)
    parser.add_argument("--comparator-script", required=True)
    parser.add_argument("--comparator-output", required=True)
    parser.add_argument("--model-executable", action="append", required=True)
    parser.add_argument("--model-config", action="append", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--disk-path", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--memory-cap-bytes", type=int, default=14_000_000_000)
    parser.add_argument("--disk-floor-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--interval-seconds", type=float, default=15)
    parser.add_argument("--model-discovery-grace-seconds", type=float, default=30)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=90)
    parser.add_argument("--paging-policy", choices=("abort", "mark-ineligible"), default="abort")
    args = parser.parse_args(argv)
    try:
        args.end_utc = utc(args.end_utc)
        started = utc(args.comparator_started_utc)
        remaining = (args.end_utc - datetime.datetime.now(UTC)).total_seconds()
        if remaining > 7200 or args.comparator_pid <= 1 or args.comparator_start_abstime <= 0:
            raise ValueError("invalid process identity or deadline beyond two hours")
        for value, low, high in ((args.interval_seconds, 1, 60), (args.model_discovery_grace_seconds, 1, 300),
                                 (args.cleanup_grace_seconds, 1, 120)):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("invalid interval or grace bound")
        if math.ceil(max(0, remaining) / args.interval_seconds) + 1 > MAX_SAMPLES:
            raise ValueError("plan exceeds sample bound")
        if not 0 < args.memory_cap_bytes <= 14_000_000_000 or args.disk_floor_bytes < 20_000_000_000:
            raise ValueError("memory cap must be at most14GB and disk floor at least20GB")
        if len(args.model_executable) != len(args.model_config) or not 1 <= len(args.model_executable) <= 2:
            raise ValueError("provide one or two aligned executable/config pairs")
        args.comparator_executable = str(Path(args.comparator_executable).resolve())
        args.model_executable = [str(Path(x).resolve()) for x in args.model_executable]
        if len(set(args.model_executable)) != len(args.model_executable):
            raise ValueError("model executable pairs must be unique")
        if any(any(c.isspace() or c in "'\"" for c in value) for value in
               [args.comparator_script, args.comparator_output, *args.model_config]):
            raise ValueError("ps identity matching requires script/output/config paths without whitespace or quotes")
        if args.baseline.stat().st_size > 1 << 20:
            raise ValueError("baseline exceeds metadata bound")
        baseline = json.loads(args.baseline.read_text())
        if utc(baseline["utc"]) > started or started > datetime.datetime.now(UTC):
            raise ValueError("baseline must precede comparator launch")
        if (started - utc(baseline["utc"])).total_seconds() > 300:
            raise ValueError("prelaunch baseline must be no more than five minutes old at launch")
        for key in ("pageouts", "swapouts", "swap_used_bytes"):
            if type(baseline.get("counters", {}).get(key)) is not int or baseline["counters"][key] < 0:
                raise ValueError("baseline counter missing: " + key)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    probe = SystemProbe(args)
    try:
        output = JSONL(args.output)
    except OSError as error:
        # The comparator is already owned and may be active. Failure to open
        # evidence must take the same verified stop path as a later disk error.
        def unavailable(*_, **__):
            raise OSError("watcher evidence could not be opened: " + str(error))
        return monitor(args, baseline, probe, unavailable)
    try:
        return monitor(args, baseline, probe, output.emit)
    finally:
        output.file.close()


if __name__ == "__main__":
    raise SystemExit(main())
