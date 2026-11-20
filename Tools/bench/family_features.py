#!/usr/bin/env python3
"""Serial feature screening in one isolated, loaded Origami candidate per family.

The arms come from --variants-file (control, each variant, control again), for any of
the three models. Defaults to 24 tokens; this is a warm-process feature screen, not a
cold-process gold comparison. Optional reverse confirmation runs only promising
arms. No configuration, host memory limit, or production process is modified.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error

from native_compare import (OwnedProcess, host_boundary_delta, host_boundary_snapshot,
                            host_memory_controls, http_json, memory_budget, owned_process_memory, save)
from origami_compare import BUNDLES, validate_status


def normalize_variants(family, variants):
    """Qwen no_store alone permits live prefix reuse; an explicit scope isolates every arm."""
    normalized = {name: dict(arm) for name, arm in variants.items()}
    if family == "qwen":
        for arm in normalized.values():
            # This harness loads the default C256/R8192 plan in a fresh data directory with ORIGAMI_* removed.
            # Keep explicit prefill experiments intact; the decode experiment receives the same scope on all arms.
            arm.setdefault("qwen_prefill_scope", 8192)
    return normalized


def arithmetic_class(family, variants):
    different_scopes = family == "qwen" and len({arm["qwen_prefill_scope"] for arm in variants.values()}) > 1
    return ("NUMERIC for Qwen prefill grouping; exact output IDs are a conservative promotion diagnostic."
            if different_scopes else "BITWISE intended; exact generated token ID parity is measured, not assumed.")


def validate_prompt_isolation(usage, encoded):
    if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
        raise ValueError("prefix reuse entered the feature batch")
    if usage.get("prompt_tokens") != encoded["prompt_tokens"]:
        raise ValueError("prompt token count differs from exact tokenizer export")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def thermal_clear(text):
    """Require affirmative normal thermal/performance observations; unknown is not clean."""
    if not isinstance(text, str) or not text.strip():
        return False
    for category in ("thermal", "performance"):
        normal = re.search(r"No " + category + r" warning level has been recorded", text, re.I)
        levels = re.findall(category + r"\s+warning\s+level\s*[:=]\s*(\d+)", text, re.I)
        if any(int(level) != 0 for level in levels) or not (normal or levels):
            return False
    for field in ("CPU_Scheduler_Limit", "CPU_Speed_Limit"):
        limits = re.findall(field + r"\s*=\s*(\d+)", text, re.I)
        if any(int(limit) < 100 for limit in limits):
            return False
    return True


def boundary_issues(row):
    """Reject missing observations, fresh paging, or non-normal thermal state."""
    issues = []
    snapshots = []
    for edge in ("before", "after"):
        snapshot = row.get("host_boundary_" + edge)
        if not isinstance(snapshot, dict):
            issues.append(edge + ": host snapshot missing")
            continue
        snapshots.append(snapshot)
        counters = snapshot.get("counters", {})
        for counter in ("pageouts", "swapouts", "swap_used_bytes"):
            if type(counters.get(counter)) is not int or counters[counter] < 0:
                issues.append(edge + ": counter missing/invalid: " + counter)
        raw = snapshot.get("raw", {})
        for command in ("vm_stat", "swap_usage", "thermal"):
            observation = raw.get(command, {})
            if observation.get("returncode") != 0 or not observation.get("stdout", "").strip() or observation.get("error"):
                issues.append(edge + ": observation missing/failed: " + command)
        if not thermal_clear(raw.get("thermal", {}).get("stdout")):
            issues.append(edge + ": thermal/performance warning or normal state unverified")
    if len(snapshots) == 2:
        before, after = [snapshot.get("counters", {}) for snapshot in snapshots]
        for counter in ("pageouts", "swapouts", "swap_used_bytes"):
            a, b = before.get(counter), after.get(counter)
            if type(a) is int and type(b) is int:
                if b > a:
                    issues.append("fresh " + counter + ": +" + str(b - a))
                elif b < a and counter != "swap_used_bytes":
                    issues.append(counter + " counter reset; interval unverified")
    return issues


def assess(rows, metric="decode"):
    """Conservative screening only: compare each arm with the faster control."""
    controls = [row for row in rows if row["variant"] == "control"]
    result = {"promising": [], "promotion_authorized": False, "metric": metric,
              "note": "Short warm-process screening; reverse confirmation and correctness review are required before promotion."}
    if len(controls) != 2 or any(row.get("status") != "completed" for row in rows):
        return {**result, "reason": "incomplete controls or requests"}
    if any(not row.get("count_reached") or not row.get("memory_observed_within_budget") for row in rows):
        return {**result, "reason": "incomplete token counts or measured memory budget unavailable/exceeded"}
    contaminated = [{"name": row.get("name", row["variant"]), "issues": boundary_issues(row)} for row in rows]
    contaminated = [item for item in contaminated if item["issues"]]
    if contaminated:
        return {**result, "reason": "paging, thermal state, or missing host observations disqualify this screen",
                "host_observation_issues": contaminated}
    if any(row.get("content_gap_summary", {}).get("severe_gap_observed") for row in controls):
        return {**result, "reason": "control content timing has severe intermittent gaps; no stalls are subtracted"}
    rate_key = "server_" + metric + "_tps"
    rates = [row.get(rate_key) for row in controls]
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 for value in rates):
        return {**result, "reason": "server " + metric + " rates unavailable"}
    reference = controls[0]["output_ids"]
    if controls[1]["output_ids"] != reference:
        return {**result, "reason": "control output IDs differ"}
    drift = abs(rates[1] / rates[0] - 1)
    threshold = max(0.05, 2 * drift)
    result.update({"control_drift_fraction": drift, "required_gain_fraction": threshold,
                   "reference_" + metric + "_tps": max(rates), "arms": {}})
    if drift > 0.15:
        return {**result, "reason": "controls differ by more than 15%; cache/order/noise prevents a decision"}
    decode_reference = None
    if metric == "prefill":
        decode_rates = [row.get("server_decode_tps") for row in controls]
        if any(not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 for rate in decode_rates):
            return {**result, "reason": "decode regression guard lacks control timings"}
        if abs(decode_rates[1] / decode_rates[0] - 1) > 0.15:
            return {**result, "reason": "decode control drift prevents prefill promotion"}
        decode_reference = max(decode_rates)
        result["minimum_decode_tps"] = 0.95 * decode_reference
    for row in rows:
        if row["variant"] == "control":
            continue
        rate = row.get(rate_key)
        gain = rate / max(rates) - 1 if isinstance(rate, (float, int)) and math.isfinite(rate) and rate > 0 else None
        parity = row["output_ids"] == reference
        decode = row.get("server_decode_tps")
        decode_ok = decode_reference is None or (isinstance(decode, (int, float)) and math.isfinite(decode) and decode >= 0.95 * decode_reference)
        result["arms"][row["variant"]] = {"gain_over_faster_control": gain, "output_ids_match": parity, "decode_regression_guard_passed": decode_ok}
        if parity and gain is not None and gain >= threshold and decode_ok:
            result["promising"].append(row["variant"])
    return result


def wait_ready(server, base, deadline):
    while True:
        if server.p.poll() is not None:
            raise RuntimeError("Origami exited during loading: " + str(server.p.returncode))
        if time.monotonic() - server.started > deadline:
            raise TimeoutError("Origami load deadline")
        try:
            health = http_json(base + "/health", timeout=2)
            if health.get("status") == "ready":
                return
            if health.get("status") in ("failed", "error"):
                raise RuntimeError("Origami load failed: " + json.dumps(health))
        except urllib.error.HTTPError as error:
            body = error.read(65536).decode("utf-8", "replace")
            if error.code == 503 and ('"failed"' in body or '"error"' in body):
                raise RuntimeError("Origami load failed: " + body)
        except OSError:
            pass
        time.sleep(0.2)


def request_arm(server, plan, output, name, variant, body, encoded, args):
    directory = output / name
    directory.mkdir()
    (directory / "collector").mkdir()
    request = {**body, "origami_tune": plan["variants"][variant]}
    save(directory / "request.json", request)
    receipt = {"name": name, "variant": variant, "origami_tune": plan["variants"][variant],
               "started_utc": utc(), "requested_tokens": args.tokens,
               "memory_scope": "Process lifetime peak includes prior requests in this feature batch."}
    collector = None
    base = "http://127.0.0.1:" + str(args.port)
    try:
        before = http_json(base + "/v1/status")
        save(directory / "status-before.json", before)
        validate_status(before, plan["model"], output / "config.json", memory_limit=plan["memory_limit_bytes"])
        previous = before.get("requests", {}).get("recent", [])
        receipt["host_boundary_before"] = host_boundary_snapshot()
        save(directory / "host-before.json", receipt["host_boundary_before"])
        argv = [sys.executable, "-B", str(Path(__file__).with_name("stream_request.py").resolve()),
                "--url", base + "/v1/chat/completions", "--request", str(directory / "request.json"),
                "--output", str(directory / "request"), "--repeat", "1", "--timeout", str(args.request_deadline)]
        collector = OwnedProcess(argv, str(directory), os.environ.copy(), directory / "collector")
        try:
            collector.p.wait(timeout=args.request_deadline)
        except subprocess.TimeoutExpired:
            raise TimeoutError("feature request hard deadline")
        collector.stop()
        summary = json.loads((directory / "request/summary.json").read_text())
        if len(summary) != 1 or not summary[0].get("complete") or collector.p.returncode:
            raise ValueError("collector did not receive one complete request")
        result = summary[0]
        receipt["request_receipt"] = result
        frame_file = directory / "request/request-00/frames.jsonl"
        content_times = []
        for line in frame_file.read_text().splitlines():
            frame = json.loads(line)
            if frame["data"] == "[DONE]":
                continue
            event = json.loads(frame["data"])
            if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                content_times.append(frame["seconds"])
        gaps = [b - a for a, b in zip(content_times, content_times[1:])]
        if gaps:
            median = statistics.median(gaps)
            receipt["content_gap_summary"] = {"median_seconds": median, "maximum_seconds": max(gaps),
                "severe_gap_observed": max(gaps) > max(1.0, 4 * median),
                "note": "Post-request SSE diagnostic; transport chunks can differ from tokens. No timing correction or causal attribution."}
        usage = result.get("usage", {})
        validate_prompt_isolation(usage, encoded)
        tokens = usage.get("origami_tokens")
        if (not isinstance(tokens, list) or len(tokens) != usage.get("completion_tokens")
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("output token IDs are absent or disagree with usage")
        receipt.update({"output_ids": tokens, "output_tokens": len(tokens), "count_reached": len(tokens) == args.tokens,
                        "decode_token_intervals": max(0, len(tokens) - 1)})
        first, last = result.get("first_content_seconds"), result.get("last_content_seconds")
        if isinstance(first, (int, float)) and isinstance(last, (int, float)) and last > first:
            receipt.update({"client_first_to_last_seconds": last - first,
                            "client_first_to_last_tps": (len(tokens) - 1) / (last - first),
                            "client_timing_note": "Visible SSE content endpoints; token decoding and network buffering can group pieces."})
        after = http_json(base + "/v1/status")
        save(directory / "status-after.json", after)
        validate_status(after, plan["model"], output / "config.json", memory_limit=plan["memory_limit_bytes"])
        recent = after.get("requests", {}).get("recent", [])
        if len(recent) != len(previous) + 1:
            raise ValueError("server recent requests did not advance by exactly one")
        native = recent[-1]
        if (native.get("completion_tokens") != len(tokens) or native.get("prompt_tokens") != encoded["prompt_tokens"]
                or native.get("cached_tokens") != 0 or native.get("mode") != "exact"
                or native.get("reasoning_effort") != "none" or native.get("hot_copy") is not False):
            raise ValueError("server request metadata differs from isolated exact request")
        receipt["server_request"] = native
        receipt["server_prefill_tps"] = native.get("prefill_tps")
        rate = native.get("decode_tps")
        if isinstance(rate, (float, int)) and math.isfinite(rate) and rate > 0 and len(tokens) > 1:
            receipt.update({"server_decode_tps": rate, "server_decode_interval_seconds": (len(tokens) - 1) / rate,
                            "server_timing_note": "n-1 divided by first emitted content to server request completion; includes finalization."})
        resources = owned_process_memory(server, plan["binary"])
        receipt["process_resources"] = resources
        receipt["memory_observed_within_budget"] = resources["peak_footprint_bytes"] <= plan["memory_limit_bytes"] if resources["measured"] else None
        if receipt["memory_observed_within_budget"] is False:
            raise ValueError("candidate lifetime physical footprint exceeds the requested budget")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update({"status": "failed", "error": str(error)})
    finally:
        if collector:
            collector.stop()
        if "host_boundary_before" in receipt:
            receipt["host_boundary_after"] = host_boundary_snapshot()
            save(directory / "host-after.json", receipt["host_boundary_after"])
            receipt["host_boundary_delta"] = host_boundary_delta(receipt["host_boundary_before"], receipt["host_boundary_after"])
        receipt["finished_utc"] = utc()
        save(directory / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen", "glm", "deepseek"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--port", type=int, default=11469)
    parser.add_argument("--load-deadline", type=float, default=360)
    parser.add_argument("--request-deadline", type=float, default=300)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--memory-gb", default="25.77", help="total memory envelope in exact decimal GB")
    parser.add_argument("--variants-file", type=Path, help="ordered JSON mapping of arm names to origami_tune objects, including control")
    parser.add_argument("--metric", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--encoded-dir", type=Path, required=True)
    parser.add_argument("--confirm-promising", action="store_true", help="append reverse-order confirmation only for meaningful matching-output gains")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        memory_gb, memory_limit = memory_budget(args.memory_gb)
    except ValueError as error:
        parser.error(str(error))
    if args.variants_file:
        variants = json.loads(args.variants_file.read_text())
        if (not isinstance(variants, dict) or "control" not in variants or not 2 <= len(variants) <= 8
                or any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name) or not isinstance(value, dict)
                       for name, value in variants.items())):
            parser.error("variants must contain control and 1..7 named origami_tune objects")
    else:
        parser.error("--variants-file is required")
    requested_variants = variants
    variants = normalize_variants(args.family, variants)
    if not 2 <= args.tokens <= 256 or not 1024 <= args.port <= 65535 or args.port == 11439 or min(args.load_deadline, args.request_deadline) <= 0:
        parser.error("tokens must be 2..256, port private, and deadlines positive")
    binary = args.candidate.resolve()
    if not binary.is_file() or any(not (binary.parent / bundle).is_dir() for bundle in BUNDLES):
        parser.error("candidate and matching adjacent Swift resource bundles are required")
    manifest = json.loads(args.manifest.read_text())
    model = next(engine["model"] for engine in manifest["engines"].values() if engine["family"] == args.family)
    if not Path(model).exists():
        parser.error("checkpoint is unavailable: " + model)
    encoded = json.loads((args.encoded_dir / (args.family + ".json")).read_text())
    ids = encoded.get("tokens")
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids) or encoded.get("prompt_tokens") != len(ids):
        parser.error("exact encoded prompt IDs/count are required")
    body = json.loads(args.request.read_text())
    if not isinstance(body, dict) or body.get("n", 1) != 1 or body.get("tools") or body.get("origami_teacher_tokens"):
        parser.error("a single plain request is required")
    body.update({"model": "origami", "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
                 "presence_penalty": 0, "seed": 37, "max_tokens": args.tokens, "reasoning_effort": "none",
                 "origami_mode": "exact", "origami_return_tokens": True, "prompt_cache_policy": "no_store"})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("data", "context", "empty-store", "empty-hot-copy"):
        (output / name).mkdir()
    save(output / "config.json", {"stores": []})
    save(output / "encoded.json", encoded)
    save(output / "request-base.json", body)
    argv = [str(binary), "serve", "--single-model", "--config", str(output / "config.json"),
            "--model", model, "--port", str(args.port), "--listen", "local", "--data-dir", str(output / "data"),
            "--context-dir", str(output / "context"), "--store", str(output / "empty-store"),
            "--hot-copy-dir", str(output / "empty-hot-copy"), "--context", "8192", "--memory-limit", memory_gb + "GB",
            "--hot-copy", "off", "--speculation", "off", "--allow-fast", "false", "--quality", "exact",
            "--reasoning-effort", "none", "--context-on-disk", "false", "--commands", "false",
            "--list-models", "false", "--status-messages", "false", "--request-log", "off"]
    sequence = ["control"] + [name for name in variants if name != "control"] + ["control"]
    plan = {"family": args.family, "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "model": model, "argv": argv, "sequence": sequence, "variants": variants,
            "requested_variants": requested_variants, "score_metric": args.metric,
            "requested_tokens": args.tokens, "context": 8192, "memory_limit_bytes": memory_limit,
            "host_memory_controls": host_memory_controls(), "speculation": "off", "hot_copy": "off",
            "comparison_class": "single_loaded_process_feature_screen", "eligible_for_strict_gold_comparison": False,
            "cache_condition": "No prompt reuse; expert and OS caches may warm across serial requests. Initial/final controls bracket order effects.",
            "prompt_isolation": ("Every Qwen arm has an explicit prefill scope, disabling live/disk prefix reuse and capture. "
                                 "Unspecified scopes use this harness plan's default 8192; expert and OS caches are not reset. "
                                 "Every response must report cached_tokens = 0."
                                 if args.family == "qwen" else "Every response must report cached_tokens = 0; expert and OS caches are not reset."),
            "arithmetic_class": arithmetic_class(args.family, variants),
            "confirm_promising": args.confirm_promising}
    save(output / "launch.json", plan)
    receipt = {**plan, "started_utc": utc(), "requests": []}
    server = None
    try:
        if args.prepare_only:
            receipt["status"] = "prepared"
            return 0
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", args.port))
        env = {key: value for key, value in os.environ.items() if not key.startswith("ORIGAMI_")}
        receipt["host_boundary_before"] = host_boundary_snapshot()
        save(output / "host-before.json", receipt["host_boundary_before"])
        server = OwnedProcess(argv, str(binary.parent), env, output)
        receipt["server_pid"] = server.p.pid
        wait_ready(server, "http://127.0.0.1:" + str(args.port), args.load_deadline)
        receipt["ready_seconds"] = time.monotonic() - server.started
        def run_sequence(choices, phase):
            rows = []
            for variant in choices:
                name = "%02d-%s-%s" % (len(receipt["requests"]), phase, variant)
                row = request_arm(server, plan, output, name, variant, body, encoded, args)
                receipt["requests"].append(row)
                save(output / "receipt.json", receipt)
                print(json.dumps({"family": args.family, "name": name, "status": row["status"],
                                  "server_decode_tps": row.get("server_decode_tps"), "error": row.get("error")}), flush=True)
                if row["status"] != "completed":
                    raise RuntimeError(name + ": " + row.get("error", "request failed"))
                rows.append(row)
            return rows
        receipt["screen"] = assess(run_sequence(sequence, "screen"), args.metric)
        if args.confirm_promising and receipt["screen"]["promising"]:
            confirmed = run_sequence(["control"] + list(reversed(receipt["screen"]["promising"])) + ["control"], "reverse")
            receipt["reverse_confirmation"] = assess(confirmed, args.metric)
            stable = all(row["output_ids"] == receipt["requests"][0]["output_ids"] for row in confirmed)
            receipt["cross_phase_output_ids_match"] = stable
            receipt["repeatable_candidates"] = (sorted(set(receipt["screen"]["promising"]) & set(receipt["reverse_confirmation"]["promising"]))
                                                  if stable else [])
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update({"status": "failed", "error": str(error)})
    finally:
        if server:
            server.stop()
            receipt["server_exit_code"] = server.p.returncode
            receipt["host_boundary_after"] = host_boundary_snapshot()
            save(output / "host-after.json", receipt["host_boundary_after"])
            receipt["host_boundary_delta"] = host_boundary_delta(receipt["host_boundary_before"], receipt["host_boundary_after"])
        receipt["finished_utc"] = utc()
        save(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
