#!/usr/bin/env python3
"""Bounded, read-only IOBlockStorageDriver observations for an approved test window.

This does not start Origami or read model payloads. Device counters include other
processes. They are not NAND traffic, exclusive I/O wait, or per-request bytes.
"""
import argparse
import datetime
import json
import math
import os
import plistlib
import re
import selectors
import subprocess
import time
from xml.parsers.expat import ExpatError


COMMAND = ("/usr/sbin/ioreg", "-a", "-r", "-l", "-c", "IOBlockStorageDriver", "-d", "3")
COUNTERS = ("Bytes (Read)", "Bytes (Write)", "Operations (Read)", "Operations (Write)")
MAX_COMMAND_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 4096
DISK = re.compile(r"disk[0-9]+\Z")


def uint(value):
    return type(value) is int and 0 <= value <= (1 << 64) - 1


def parse_devices(payload, requested):
    """Map actual descendant whole-media properties, never registry-ID arithmetic.

    Stop at the first whole IOMedia below a driver: its APFS descendants must not
    become additional devices. Duplicate mappings fail closed for affected disks.
    """
    if len(payload) > MAX_COMMAND_BYTES:
        raise ValueError("ioreg output exceeds size limit")
    roots = plistlib.loads(payload)
    if not isinstance(roots, list) or len(roots) > 256:
        raise ValueError("invalid or excessive driver root list")
    candidates, id_counts = {name: [] for name in requested}, {}
    for root in roots:
        if not isinstance(root, dict) or root.get("IOObjectClass") != "IOBlockStorageDriver":
            raise ValueError("unexpected driver root")
        driver_id = root.get("IORegistryEntryID")
        if uint(driver_id):
            id_counts[driver_id] = id_counts.get(driver_id, 0) + 1
        children = root.get("IORegistryEntryChildren", [])
        if not isinstance(children, list):
            raise ValueError("invalid driver descendant list")
        pending = [(child, 0) for child in children]
        media, visited = [], 0
        while pending:
            child, depth = pending.pop()
            visited += 1
            if visited > 8192 or depth > 16 or not isinstance(child, dict):
                raise ValueError("invalid or excessive descendant tree")
            if child.get("IOObjectClass") == "IOMedia" and child.get("Whole") is True:
                media.append(child)
                continue
            children = child.get("IORegistryEntryChildren", [])
            if not isinstance(children, list):
                raise ValueError("invalid descendant list")
            pending.extend((node, depth + 1) for node in children)
        stats = root.get("Statistics", {})
        for child in media:
            name = child.get("BSD Name")
            if not isinstance(name, str) or name not in candidates:
                continue
            media_id = child.get("IORegistryEntryID")
            reasons = []
            if len(media) != 1:
                reasons.append("ambiguous_driver_media")
            if not uint(driver_id) or not uint(media_id):
                reasons.append("missing_registry_identity")
            counters = {key: stats.get(key) for key in COUNTERS} if isinstance(stats, dict) else {}
            if any(not uint(counters.get(key)) for key in COUNTERS):
                reasons.append("invalid_or_missing_counter")
            counters = {key: value if uint(value) else None for key, value in counters.items()}
            candidates[name].append({
                "bsd_name": name, "driver_registry_id": driver_id if uint(driver_id) else None,
                "media_registry_id": media_id if uint(media_id) else None,
                "media_name": str(child.get("IORegistryEntryName", ""))[:256],
                "raw_counters": counters, "reasons": reasons,
            })
    result = {}
    for name, rows in candidates.items():
        if len(rows) != 1:
            result[name] = {"valid": False, "reasons": ["missing_device" if not rows else "ambiguous_bsd_mapping"],
                            "candidate_count": len(rows)}
            continue
        row = rows[0]
        if id_counts.get(row["driver_registry_id"], 0) > 1:
            row["reasons"].append("duplicate_driver_identity")
        row["valid"] = not row["reasons"]
        result[name] = row
    return result


def interval_deltas(previous, current, requested):
    """Only adjacent, successful, nonoverlapping captures have eligible deltas.

    Sampling instants lie somewhere inside each command's start/end bracket.
    Midpoint rates are estimates, not exact rates or exclusive device bandwidth.
    """
    result = {}
    for name in requested:
        reasons = []
        if previous is None:
            result[name] = {"valid": False, "reasons": ["initial_baseline_only"]}
            continue
        old, new = previous.get("devices", {}).get(name, {}), current.get("devices", {}).get(name, {})
        if not previous.get("valid") or not current.get("valid"):
            reasons.append("invalid_capture")
        if not old.get("valid") or not new.get("valid"):
            reasons.append("invalid_device_snapshot")
        identity = ("bsd_name", "driver_registry_id", "media_registry_id", "media_name")
        if not reasons and any(old.get(key) != new.get(key) for key in identity):
            reasons.append("device_identity_changed")
        times = [previous.get("monotonic_start_ns"), previous.get("monotonic_end_ns"),
                 current.get("monotonic_start_ns"), current.get("monotonic_end_ns")]
        if any(not uint(value) for value in times) or not (times[0] <= times[1] < times[2] <= times[3]):
            reasons.append("invalid_or_overlapping_time_brackets")
        delta = {}
        if not reasons:
            delta = {key: new["raw_counters"][key] - old["raw_counters"][key] for key in COUNTERS}
            if any(value < 0 for value in delta.values()):
                reasons.append("counter_reset_or_wrap")
        row = {"valid": not reasons, "reasons": reasons}
        if not reasons:
            lower = (times[2] - times[1]) / 1e9
            upper = (times[3] - times[0]) / 1e9
            midpoint = (lower + upper) / 2
            row.update({"counter_deltas": delta, "elapsed_seconds_bounds": [lower, upper],
                        "midpoint_elapsed_seconds": midpoint,
                        "estimated_bytes_per_second": {kind: delta[f"Bytes ({kind})"] / midpoint
                                                       for kind in ("Read", "Write")}})
        result[name] = row
    return result


def bounded_command(timeout):
    """Fixed read-only command, wall timeout and hard combined stdout/stderr cap."""
    deadline = time.monotonic() + timeout
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    process = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ioreg command timeout")
                for key, _ in selector.select(remaining):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(data)
                    if total > MAX_COMMAND_BYTES:
                        raise ValueError("ioreg command output exceeds size limit")
                    chunks[key.data].extend(data)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ioreg command timeout")
            status = process.wait(timeout=remaining)
        if status:
            raise RuntimeError(f"ioreg exit {status}: {chunks['stderr'][:1024].decode(errors='replace')}")
        return bytes(chunks["stdout"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        process.stdout.close()
        process.stderr.close()


def capture(requested, timeout, command=bounded_command, clock=time.monotonic_ns):
    row = {"kind": "sample", "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "monotonic_start_ns": clock(), "valid": False, "devices": {}}
    try:
        payload = command(timeout)
        row["devices"] = parse_devices(payload, requested)
        row["valid"] = True
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError,
            plistlib.InvalidFileException, TypeError, OverflowError, ExpatError) as error:
        row["error"] = f"{type(error).__name__}: {error}"[:2048]
    row["monotonic_end_ns"] = clock()
    row["capture_seconds"] = (row["monotonic_end_ns"] - row["monotonic_start_ns"]) / 1e9
    return row


def validate_plan(devices, interval, duration, timeout, once):
    if not devices or len(devices) > 8 or len(set(devices)) != len(devices) or any(not DISK.fullmatch(d) for d in devices):
        raise ValueError("select 1–8 distinct whole disks, for example --device disk0 --device disk6")
    for name, value, low, high in (("interval", interval, .1, 60), ("duration", duration, .1, 7200),
                                   ("command timeout", timeout, .05, 30)):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and between {low} and {high} seconds")
    if not once and (duration < interval or math.ceil(duration / interval) + 1 > MAX_SAMPLES):
        raise ValueError("duration must cover one interval and at most 4096 scheduled samples")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="append", required=True, dest="devices")
    parser.add_argument("--output", required=True, help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--interval-seconds", type=float, default=2)
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--command-timeout-seconds", type=float, default=3)
    parser.add_argument("--once", action="store_true", help="One capability snapshot, without intervals or rates")
    args = parser.parse_args(argv)
    try:
        validate_plan(args.devices, args.interval_seconds, args.duration_seconds, args.command_timeout_seconds, args.once)
    except ValueError as error:
        parser.error(str(error))
    # The output is deliberately exclusive, owner-only, and streamed; memory does
    # not grow with duration. Partial output remains useful after interruption.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        written = 0
        def emit(row):
            nonlocal written
            line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            written += len(line.encode("utf-8"))
            if written > MAX_OUTPUT_BYTES:
                raise ValueError("64 MiB output limit reached")
            output.write(line)
            output.flush()
        emit({"kind": "header", "schema": "origami-device-counters-v1", "command": COMMAND,
              "devices": args.devices, "interval_seconds": args.interval_seconds,
              "duration_seconds": args.duration_seconds, "once": args.once,
              "scope": "Device-wide kernel counters, including OS/background traffic; not request-attributed or NAND bytes.",
              "time_semantics": "Monotonic command brackets; midpoint rates estimated. Total Time and Latency Time are not interpreted.",
              "topology_requirement": "Resolve physical backing stores before use; APFS volumes are not additional physical devices."})
        start = time.monotonic()
        deadline = start + args.duration_seconds
        next_due, count, previous, invalid_count = start, 0, None, 0
        reason = "duration_reached"
        try:
            while count < MAX_SAMPLES:
                now = time.monotonic()
                if count and (args.once or now >= deadline):
                    reason = "one_shot_complete" if args.once else "duration_reached"
                    break
                if next_due > now:
                    time.sleep(min(next_due - now, max(0, deadline - now)))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                row = capture(args.devices, min(args.command_timeout_seconds, remaining))
                if not row["valid"] or not all(device["valid"] for device in row["devices"].values()):
                    invalid_count += 1
                row["index"] = count
                row["deltas"] = interval_deltas(previous, row, args.devices)
                emit(row)
                previous = row  # An invalid capture breaks continuity; never bridge it silently.
                count += 1
                next_due = start + count * args.interval_seconds
                # Slow captures skip missed slots; never issue catch-up bursts.
                now = time.monotonic()
                if next_due <= now:
                    next_due = now + args.interval_seconds
            else:
                reason = "sample_limit_reached"
        except KeyboardInterrupt:
            reason = "interrupted"
        emit({"kind": "footer", "sample_count": count, "invalid_capture_count": invalid_count, "reason": reason,
              "elapsed_seconds": time.monotonic() - start})
    return 1 if invalid_count or not count else 0


if __name__ == "__main__":
    raise SystemExit(main())
