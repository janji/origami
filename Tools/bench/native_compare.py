#!/usr/bin/env python3
"""Bounded, one-request comparator driver; invoke only after production is drained.

Inputs are engine-checks --encode-request JSON. No model, configuration or cache
files are modified. Each invocation creates a new output directory and owns only
the processes it launches. Machine profiling is a separate, untimed invocation.
"""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import datetime
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MEMORY_LIMIT = 25770000000


def memory_budget(value):
    """Canonical decimal GB and exact integer bytes, without float rounding."""
    try:
        gb = Decimal(str(value))
        amount = gb * 1_000_000_000
        if not gb.is_finite() or gb <= 0 or amount != amount.to_integral_value() or amount > 2**63 - 1:
            raise ValueError("memory-gb must be positive finite decimal GB with whole-byte precision")
        return format(gb, "f"), int(amount)
    except InvalidOperation:
        raise ValueError("memory-gb must be positive finite decimal GB with whole-byte precision")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def profile_resources(source, output, formats):
    """Select measured CPU formats through the native probe's supported CLI.

    The upstream profile command has no format option. An isolated resource
    descriptor points only its CPU probe to this exec wrapper; engine and probe
    binaries remain unchanged, and all thread/token sweeps remain upstream's.
    """
    if not formats or len(set(formats)) != len(formats) or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", item) for item in formats):
        raise ValueError("CPU formats must be unique uppercase ggml type names")
    source = Path(source).resolve()
    descriptor = json.loads(source.read_text())
    descriptor["binary"] = str((source.parent / descriptor["binary"]).resolve())
    for key in ("templates", "probes"):
        descriptor[key] = {name: str((source.parent / path).resolve()) for name, path in descriptor.get(key, {}).items()}
    probe = descriptor["probes"].get("cpu")
    if not probe or not Path(probe).is_file():
        raise ValueError("Native CPU probe is missing")
    wrapper = output / "cpu-probe-formats.sh"
    wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(probe) + ' "$@" --formats ' + shlex.quote(",".join(formats)) + "\n")
    wrapper.chmod(0o700)
    descriptor["probes"]["cpu"] = str(wrapper)
    target = output / "profile-resources.json"
    save(target, descriptor)
    return target, {"formats": formats, "native_probe": probe, "wrapper": str(wrapper),
                    "resources_source": str(source), "resources_effective": str(target),
                    "note": "One native CPU sweep restricted to this checkpoint's routed formats; unchanged native probe, default thread/token/DRAM measurements; no fabricated profile values."}


def host_memory_controls():
    """Read the system-wide Metal ceiling once, outside request timing."""
    command = ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        value = int(result.stdout.strip())
        return {"iogpu.wired_limit_mb": value, "wired_limit_bytes": value * 1024 * 1024,
                "scope": "system-wide wired-memory ceiling; separate from this engine's total memory budget",
                "changed_by_harness": False}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"error": str(error), "changed_by_harness": False}


def host_boundary_snapshot():
    """Three bounded read-only host observations, outside timed inference."""
    commands = {"vm_stat": ["/usr/bin/vm_stat"],
                "swap_usage": ["/usr/sbin/sysctl", "vm.swapusage"],
                "thermal": ["/usr/bin/pmset", "-g", "therm"]}
    def observe(item):
        name, argv = item
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=3, check=False)
            return name, {"argv": argv, "returncode": result.returncode,
                          "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.SubprocessError) as error:
            return name, {"argv": argv, "error": str(error)}
    started = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=3) as pool:
        raw = dict(pool.map(observe, commands.items()))
    counters = {}
    for name in ("Pageouts", "Swapouts"):
        match = re.search(r"^" + name + r":\s+(\d+)\.", raw["vm_stat"].get("stdout", ""), re.M)
        counters[name.lower()] = int(match[1]) if match else None
    page = re.search(r"page size of (\d+) bytes", raw["vm_stat"].get("stdout", ""))
    counters["page_size_bytes"] = int(page[1]) if page else None
    used = re.search(r"used\s*=\s*([\d.]+)([KMGT])", raw["swap_usage"].get("stdout", ""))
    counters["swap_used_bytes"] = int(float(used[1]) * 1024 ** ("KMGT".index(used[2]) + 1)) if used else None
    return {"utc": stamp, "monotonic_seconds": started, "observation_seconds": time.monotonic() - started,
            "counters": counters, "raw": raw,
            "scope": "system-wide boundary observations; no per-token polling or host changes"}


def host_boundary_delta(before, after):
    result = {}
    for key in ("pageouts", "swapouts", "swap_used_bytes"):
        a, b = before["counters"].get(key), after["counters"].get(key)
        result[key] = b - a if isinstance(a, int) and isinstance(b, int) else None
    result["paging_activity_observed"] = any(isinstance(result[k], int) and result[k] > 0 for k in ("pageouts", "swapouts"))
    result["qualification"] = "Positive deltas can contaminate a comparison; global activity is not automatically attributable to this engine. Inspect raw pmset output for thermal warnings."
    return result


# SDK sys/resource.h rusage_info_v4: UUID followed by 35 uint64_t fields.
_RUSAGE_V4_NAMES = """user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size
phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time
child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread
diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background
cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive
billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions
cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time""".split()


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_V4_NAMES]


_PROCESS_MEMORY_SCOPE = ("maximum of known kernel-reported lifetime physical-footprint peaks and observed current "
                         "physical footprints; fields are not an atomic snapshot or an upper bound on unobserved allocations")


def _kernel_process_memory_snapshot(pid, expected_binary):
    """One raw kernel query bracketed by executable checks; no resampling."""
    observation = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "monotonic_start_ns": time.monotonic_ns()}
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        lib.proc_pidpath.restype = ctypes.c_int
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        expected = Path(expected_binary).resolve()
        def check_path():
            executable = ctypes.create_string_buffer(4096)
            if lib.proc_pidpath(pid, executable, len(executable)) <= 0:
                raise OSError(ctypes.get_errno(), "proc_pidpath failed")
            actual = Path(os.fsdecode(executable.value)).resolve()
            if actual != expected:
                raise ValueError("owned PID executable is not the expected single-model binary")
            return str(actual)
        observation["verified_executable_before"] = check_path()
        usage = _RUsageInfoV4()
        result = lib.proc_pid_rusage(pid, 4, ctypes.byref(usage))
        observation["rusage_returncode"] = result
        if result:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        observation.update({"current_rss_bytes": usage.resident_size,
                            "current_footprint_bytes": usage.phys_footprint,
                            "kernel_reported_lifetime_peak_bytes": usage.lifetime_max_phys_footprint,
                            "process_start_abstime": usage.proc_start_abstime,
                            "process_exit_abstime": usage.proc_exit_abstime})
        # These fields already arrive in the same kernel query. CPU times are Mach ticks, not nanoseconds;
        # convert with mach_timebase_info. Disk counters are OS-attributed process I/O, not NAND traffic.
        observation["resource_counters"] = {"user_time_mach": usage.user_time, "system_time_mach": usage.system_time,
            "diskio_bytesread": usage.diskio_bytesread, "diskio_byteswritten": usage.diskio_byteswritten,
            "pageins": usage.pageins}
        observation["verified_executable_after"] = check_path()
        return observation
    except Exception as error:
        observation["error"] = str(error)
        error.memory_observation = observation
        raise
    finally:
        observation["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        observation["monotonic_end_ns"] = time.monotonic_ns()


def kernel_process_memory(pid, expected_binary):
    """Read one verified PID, retrying only positive non-atomic fields at most 3 times.

    XNU gather_rusage_info reads lifetime_max in its V4 case, then the V0 path
    reads current footprint later. Allocation between them can yield a current footprint above
    the earlier peak. Keep all observations; never replace an earlier high value
    with a lower retry. No process is paused and no sleep/polling loop is added.
    """
    observations = []
    birth = None
    try:
        for _ in range(3):
            try:
                raw = _kernel_process_memory_snapshot(pid, expected_binary)
            except Exception as error:
                partial = getattr(error, "memory_observation", None)
                if partial is not None:
                    observations.append(partial)
                raise
            observations.append(raw)
            if raw["process_exit_abstime"] or raw["process_start_abstime"] <= 0:
                raise ValueError("kernel identity is not a live process")
            if birth is not None and raw["process_start_abstime"] != birth:
                raise ValueError("kernel process birth changed during memory resampling")
            birth = raw["process_start_abstime"]
            current, peak = raw["current_footprint_bytes"], raw["kernel_reported_lifetime_peak_bytes"]
            if current <= 0 or peak <= 0:
                raise ValueError("kernel physical-footprint fields are absent")
            if peak >= current:
                break
        last = observations[-1]
        return {"measured": True, "verified_executable": last["verified_executable_after"],
                "current_rss_bytes": last["current_rss_bytes"],
                "current_footprint_bytes": last["current_footprint_bytes"],
                "peak_footprint_bytes": max(max(row["current_footprint_bytes"], row["kernel_reported_lifetime_peak_bytes"])
                                            for row in observations),
                "kernel_reported_lifetime_peak_bytes": max(row["kernel_reported_lifetime_peak_bytes"] for row in observations),
                "peak_measurement_scope": _PROCESS_MEMORY_SCOPE,
                "last_snapshot_peak_at_least_current": last["kernel_reported_lifetime_peak_bytes"] >= last["current_footprint_bytes"],
                "memory_snapshot_attempts": len(observations), "memory_observations": observations,
                "process_start_abstime": birth, "resource_counters": last.get("resource_counters")}
    except Exception as error:
        # Bounded numeric/path evidence survives both identity and measurement
        # failures. Callers must not signal a PID whose identity failed.
        error.memory_observations = observations
        raise


def owned_process_memory(owned, expected_binary):
    """One bounded post-request observation of a verified direct child, never a front PID."""
    pid = owned.p.pid
    result = {"pid": pid, "measured": False,
              "measurement": "bounded post-request proc_pid_rusage(RUSAGE_INFO_V4) lifetime-peak/current observations; at most 3 same-birth snapshots, no inference polling"}
    try:
        if owned.p.poll() is not None or os.getpgid(pid) != pid:
            raise ValueError("owned process is no longer its live process-group leader")
        identity = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "ppid="],
                                  capture_output=True, text=True, timeout=3, check=True)
        if int(identity.stdout.strip()) != os.getpid():
            raise ValueError("PID is not this harness's direct child")
        result.update(kernel_process_memory(pid, expected_binary))
        result["verified_parent_pid"] = os.getpid()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        if hasattr(error, "memory_observations"):
            result["memory_observations"] = error.memory_observations
    return result


def inferssd_process_memory(owned, expected_binary, directory):
    """Use existing ownership metadata to measure the native descendant once."""
    directory = Path(directory).resolve()
    launcher = owned.p.pid
    result = {"measured": False, "launcher_pid": launcher,
              "measurement": "bounded post-request proc_pid_rusage(RUSAGE_INFO_V4) lifetime-peak/current observations of the verified native server; includes loading; no added inference polling",
              "scope": "native server only; Python launcher/guardian/worker excluded, matching the single model-process envelope"}
    def read_metadata(name):
        with (directory / name).open() as source:
            info = os.fstat(source.fileno())
            if info.st_uid != os.geteuid() or not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
                raise ValueError("InferSSD ownership metadata has unexpected UID, type or size")
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("InferSSD ownership metadata is not an object")
        return value
    measured = None
    try:
        if owned.p.poll() is not None or os.getpgid(launcher) != launcher:
            raise ValueError("InferSSD owned launcher is no longer live")
        owner = read_metadata(".inferssd-owner")
        if type(owner.get("controller_pid")) is not int or owner["controller_pid"] != launcher or not isinstance(owner.get("token"), str) or not owner["token"]:
            raise ValueError("InferSSD run-directory owner differs from this launcher")
        state = read_metadata("server-state.json")
        pid = state.get("pid")
        if type(pid) is not int or pid <= 1 or "returncode" not in state or state["returncode"] is not None:
            raise ValueError("InferSSD server metadata does not identify a live native process")
        result["pid"] = pid
        runtime = read_metadata("runtime.json")
        argv = runtime.get("argv")
        if not isinstance(argv, list) or not argv or not isinstance(argv[0], str) or Path(argv[0]).resolve() != Path(expected_binary).resolve():
            raise ValueError("InferSSD recorded executable differs from the expected native binary")
        if not isinstance(runtime.get("directory"), str) or Path(runtime["directory"]).resolve() != directory:
            raise ValueError("InferSSD recorded run directory differs")
        # InferSSD starts separate sessions for guardian, worker and server. Verify
        # parentage, not process-group membership; inspect no arguments/environment.
        snapshot = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,uid="],
                                  capture_output=True, text=True, timeout=3, check=True)
        processes = {}
        for line in snapshot.stdout.splitlines():
            row = tuple(map(int, line.split()))
            if len(row) != 3 or row[0] in processes:
                raise ValueError("Process ownership snapshot is malformed")
            processes[row[0]] = row[1:]
        chain = []
        current = pid
        for _ in range(16):
            if current in chain or current not in processes:
                raise ValueError("InferSSD native process has no verified live ancestry")
            parent, uid = processes[current]
            if uid != os.geteuid():
                raise ValueError("InferSSD process ancestry UID differs from this harness")
            chain.append(current)
            if current == launcher:
                if parent != os.getpid() or len(chain) < 2:
                    raise ValueError("InferSSD launcher is not this harness's direct parent-owned child")
                break
            if parent <= 1:
                raise ValueError("InferSSD native process is outside the owned launcher ancestry")
            current = parent
        else:
            raise ValueError("InferSSD process ancestry exceeds the bounded depth")
        measured = kernel_process_memory(pid, expected_binary)
        if owned.p.poll() is not None or read_metadata(".inferssd-owner") != owner or read_metadata("server-state.json") != state:
            raise ValueError("InferSSD ownership changed during the memory observation")
        result.update(measured)
        result.update({"verified_uid": os.geteuid(), "verified_ancestry_server_to_launcher": chain,
                       "verified_launcher_parent_pid": os.getpid()})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        retained = measured.get("memory_observations", []) if measured is not None else []
        retained = retained + getattr(error, "memory_observations", [])
        if retained:
            result["memory_observations"] = retained
    return result


def record_inferssd_memory(receipt, resources, memory_limit):
    receipt["process_resources"] = resources
    receipt["timing_eligible"] = False
    receipt["memory_observed_within_budget"] = None
    if not resources.get("measured"):
        receipt["memory_qualification"] = "Native physical-footprint peak unavailable; response timings are retained but excluded from a measured equal-memory comparison. The native cache admission ledger and existing RSS watchdog do not prove a lifetime physical peak."
        return
    receipt["memory_observed_within_budget"] = resources["peak_footprint_bytes"] <= memory_limit
    if not receipt["memory_observed_within_budget"]:
        receipt["memory_qualification"] = "Known physical-footprint observation exceeds the exact decimal-GB envelope; comparison excluded."
        raise ValueError("InferSSD native process peak exceeds the requested memory envelope")
    receipt["memory_qualification"] = "Known kernel-reported lifetime peaks and observed current footprints were within the exact decimal-GB envelope through completion; non-atomic readings are retained and do not bound unobserved allocations; host paging/thermal and output correctness remain separate qualifications."
    receipt["timing_eligible"] = True


def time_resources(stderr):
    """macOS time -l reports child lifetime peaks without periodic sampling."""
    fields = {"peak_rss_bytes": "maximum resident set size",
              "peak_footprint_bytes": "peak memory footprint"}
    result = {}
    for key, label in fields.items():
        match = re.search(r"^\s*(\d+)\s+" + label + r"\s*$", stderr, re.M)
        if not match or int(match[1]) <= 0:
            raise ValueError("Maya resource receipt lacks " + label)
        result[key] = int(match[1])
    start = re.search(r"^\s*[\d.]+ real\s+[\d.]+ user\s+[\d.]+ sys\s*$", stderr, re.M)
    if not start:
        raise ValueError("Maya resource receipt lacks time -l summary")
    result["raw"] = stderr[start.start():]
    result["measurement"] = "macOS /usr/bin/time -l child lifetime peaks; includes loading and graph/scratch allocations; no inference polling"
    return result


def maya_result(stdout, stderr, expected_prompt, requested):
    pp = re.search(r"prompt (\d+) tokens in ([\d.]+) s", stdout)
    tg = re.search(r"decode (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", stdout)
    ids = re.search(r"^tokens:([ \d]*)$", stdout, re.M)
    mem = re.search(r"dense ([\d.]+) GB, state ([\d.]+) GB .*expert pool ([\d.]+) GB", stderr)
    if not (pp and tg and ids and mem):
        raise ValueError("Maya did not emit complete native timings/token IDs/allocation metadata")
    tokens = [int(x) for x in ids[1].split()]
    if int(pp[1]) != expected_prompt or len(tokens) != int(tg[1]):
        raise ValueError("Maya prompt or output count mismatch")
    if len(tokens) != requested:
        raise ValueError("Maya one-shot did not emit the requested count")
    return {"prompt_tokens": int(pp[1]), "output_tokens": len(tokens), "output_ids": tokens,
            "count_reached": requested == int(tg[1]) == len(tokens),
            "prefill_seconds": float(pp[2]), "decode_seconds": float(tg[2]),
            "decode_forward_steps": max(0, len(tokens) - 1),
            "decode_tokens_per_second": float(tg[3]),
            "timing_note": "Maya first output is from prefill; decode rate uses n-1 forwards. Timings printed to 0.01 s.",
            "allocation_gb": dict(zip(("dense", "state", "experts"), map(float, mem.groups()))),
            "eos_note": "Maya one-shot continues to requested count even through EOS; inspect IDs before comparing."}


def maya_startup_guard(line, receipt, output):
    """Reject oversized native allocations as soon as startup discloses them."""
    if "maya-metal: cannot pin " in line:
        receipt["startup_pin_failure"] = line.strip()
        receipt["timing_eligible"] = False
        save(output / "receipt.json", receipt)
        raise ValueError("Maya could not pin its resident allocation; aborted before accepting inference timings")
    if "maya-metal: loaded in " not in line:
        return
    match = re.search(r"dense ([\d.]+) GB, state ([\d.]+) GB .*expert pool ([\d.]+) GB in (\d+) slots", line)
    if not match:
        raise ValueError("Maya startup lacks parseable native allocation metadata")
    allocation = dict(zip(("dense", "state", "experts"), (float(value) for value in match.groups()[:3])))
    # describe() prints each actual byte count / 1e9 to two decimals, not GiB.
    upper = int((sum(allocation.values()) + 0.015) * 1e9)
    receipt["startup_allocation"] = {"gb": allocation, "slots": int(match[4]),
                                      "rounded_total_bytes_upper_bound": upper,
                                      "unit": "decimal GB (native describe divides bytes by 1e9)",
                                      "raw": line.strip(),
                                      "note": "Three values rounded to 0.01 GB; add 0.015 GB for rounding. Scratch/compiler allocations require separate lifetime peak validation."}
    save(output / "receipt.json", receipt)
    if upper > receipt.get("memory_budget_bytes", MEMORY_LIMIT):
        receipt["timing_eligible"] = False
        raise ValueError("Maya startup dense/state/expert allocation exceeds the requested memory envelope")


class OwnedProcess:
    def __init__(self, argv, cwd, env, output):
        self.output = output
        self.events = queue.Queue()
        self.started = time.monotonic()
        self.p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, start_new_session=True)
        self.threads = []
        for name in ("stdout", "stderr"):
            t = threading.Thread(target=self._copy, args=(name,), daemon=True)
            t.start()
            self.threads.append(t)

    def _copy(self, name):
        # The reader owns its pipe. Closing it here also covers natural child exit,
        # and avoids a cross-thread close while stop() is waiting on a blocked read.
        stream = getattr(self.p, name)
        with stream, (self.output / (name + ".log")).open("wb") as log:
            for line in iter(stream.readline, b""):
                log.write(line)
                log.flush()
                self.events.put((time.monotonic() - self.started, name, line.decode("utf-8", "replace")))

    def stop(self):
        def signal_group(sig):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                # The owned child/group can exit after poll() or the preceding
                # wait timed out. Still reap the child and retire its readers.
                pass
        try:
            if self.p.poll() is None:
                signal_group(signal.SIGTERM)
                try:
                    self.p.wait(timeout=35)  # InferSSD's owned server cleanup policy is 30 s.
                except subprocess.TimeoutExpired:
                    signal_group(signal.SIGKILL)
                    self.p.wait(timeout=5)
        finally:
            for thread in self.threads:
                thread.join(timeout=2)

    def wait_cli(self, ready_pattern, load_deadline, request_deadline, on_line=None):
        ready = None
        while self.p.poll() is None:
            now = time.monotonic() - self.started
            if now > (load_deadline if ready is None else ready + request_deadline):
                raise TimeoutError("load deadline" if ready is None else "request deadline")
            try:
                elapsed, _, line = self.events.get(timeout=0.05)
                if on_line:
                    on_line(line)
                if ready is None and re.search(ready_pattern, line):
                    ready = elapsed
            except queue.Empty:
                pass
        self.stop()
        while on_line and not self.events.empty():
            _, _, line = self.events.get_nowait()
            on_line(line)
        if self.p.returncode:
            raise RuntimeError("comparator exited " + str(self.p.returncode))
        return {"ready_seconds": ready, "launch_wall_seconds": time.monotonic() - self.started}


def http_json(url, body=None, timeout=3):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("slotstream", "maya", "inferssd", "inferssd-profile"), required=True)
    parser.add_argument("--encoded", type=Path, help="engine-checks --encode-request output")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--machine", type=Path, help="measured InferSSD machine profile; required for inference")
    parser.add_argument("--profile-cpu-formats", default="IQ1_S,IQ2_XXS", help="InferSSD profile: one native CPU sweep over these routed quant types")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--memory-gb", default="25.77", help="total memory envelope in exact decimal GB (default: 25.77)")
    parser.add_argument("--load-deadline", type=float, default=360)
    parser.add_argument("--request-deadline", type=float, default=300)
    parser.add_argument("--port", type=int, default=11459)
    parser.add_argument("--maya-expert-gb", type=float, default=4,
                        help="Maya pool request; mixed-format group floors make 4 GB allocate 14.63 GB on this checkpoint")
    args = parser.parse_args()
    try:
        memory_gb, memory_limit = memory_budget(args.memory_gb)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.tokens <= 256 or min(args.load_deadline, args.request_deadline) <= 0:
        parser.error("tokens must be 1..256 and deadlines positive")
    manifest = json.loads(args.manifest.read_text())
    engine = manifest["engines"][args.engine.replace("-profile", "")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLOTSTREAM_", "MAYA_", "INFERSSD_", "LLAMA_", "GGML_"))}
    receipt = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "engine": args.engine, "engine_commit": engine["commit"], "model": engine["model"],
               "source_ids_file": str(args.encoded.resolve()) if args.encoded else None,
               "requested_tokens": args.tokens, "context": 8192, "memory_budget_bytes": memory_limit,
               "memory_budget_decimal_gb": memory_gb,
               "host_memory_controls": host_memory_controls(),
               "load_deadline_seconds": args.load_deadline, "request_deadline_seconds": args.request_deadline,
               "cache_condition": "new process; OS page cache uncontrolled; no claim of cold SSD or warm expert pool"}
    if args.engine == "maya":
        receipt.update({"timing_eligible": False, "expert_pool_request_gb": args.maya_expert_gb,
                        "pool_budget_qualification": "Native minimum 296 slots per mixed-format group can exceed --expert-gb. The 4 GB request yields 14.634451 GB pool on this exact checkpoint; startup and physical peak are independently checked."})
    elif args.engine == "inferssd":
        receipt["timing_eligible"] = False
    owned = None
    try:
        if not Path(engine["model"]).exists():
            raise FileNotFoundError("Selected checkpoint is unavailable; check its mounted volume: " + engine["model"])
        if not Path(engine["binary"]).is_file():
            raise FileNotFoundError("Comparator binary is unavailable: " + engine["binary"])
        encoded = json.loads(args.encoded.read_text()) if args.encoded else None
        ids = encoded.get("tokens") if encoded else None
        if args.engine != "inferssd-profile" and (not isinstance(ids, list) or not ids or
                any(type(x) is not int or x < 0 for x in ids)):
            raise ValueError("--encoded must contain nonempty integer tokens")
        if encoded:
            if encoded.get("prompt_tokens") != len(ids):
                raise ValueError("Encoded prompt count differs from its token array")
            save(output / "encoded.json", encoded)
        cwd = engine["cwd"]
        if args.engine == "slotstream":
            prompt = encoded.get("text")
            if not isinstance(prompt, str):
                raise ValueError("SlotStream requires decoded exact prompt text as well as token IDs")
            (output / "prompt.txt").write_text(prompt)
            argv = [engine["binary"], "run", "--model", engine["model"], "--memory-gb", memory_gb,
                    "--max-context", "8192", "--mtp", "off", "--vision", "off", "--raw", "--prompt-file",
                    str(output / "prompt.txt"), "--greedy", "--seed", "37", "--max-tokens", str(args.tokens),
                    "--stats-json", str(output / "stats.json")]
            ready = r"prompt tokens: \d+"
        elif args.engine == "maya":
            if not 0 < args.maya_expert_gb <= 13:
                raise ValueError("Maya pool must be positive and <=13 GB for this dense/state envelope")
            argv = [engine["binary"], "--gguf", engine["model"], "--ids", ",".join(map(str, ids)),
                    "--n", str(args.tokens), "--max-context", "8192", "--no-mtp", "--kv-f32",
                    "--expert-gb", str(args.maya_expert_gb), "--ubatch", "64"]
            argv = ["/usr/bin/time", "-l"] + argv
            env["MAYA_GLM_NO_MTP"] = "1"
            ready = r"maya-metal: loaded in"
        else:
            env["PYTHONPATH"] = str(Path(cwd) / "framework/src")
            env["INFERSSD_HOME"] = str(output / "inferssd-home")
            if args.engine == "inferssd-profile":
                resource_file, receipt["cpu_profile_selection"] = profile_resources(engine["resources"], output, args.profile_cpu_formats.split(","))
                argv = [sys.executable, "-m", "inferssd", "profile", "--resources", str(resource_file),
                        "--model", engine["model"], "--output", str(output / "machine.json")]
                ready = r"(?!)"  # Entire separate profiling invocation has the load deadline.
            else:
                if not args.machine or not args.machine.is_file():
                    raise ValueError("InferSSD requires --machine from one prior untimed inferssd-profile invocation")
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", args.port))  # Never attach to an existing service.
                save(output / "runtime-policy.json", {"platform": "darwin", "lock_path": str(output / "runtime.lock"),
                     "stop_path": str(output / "runtime.stop"), "memory_budget_bytes": memory_limit,
                     "minimum_memlock_bytes": 0, "cleanup_seconds": 30})
                argv = [sys.executable, "-c", "from inferssd.launch.typed_command import main; raise SystemExit(main())",
                        "--execute", "--resources", engine["resources"], "--model", engine["model"],
                        "--context", "8192", "--port", str(args.port), "--prompt-cache", "off",
                        "--runtime-policy", str(output / "runtime-policy.json"), "--output-dir", str(output / "native-run"),
                        "--machine", str(args.machine.resolve()), "--set", "host.budget_mib=" + str(memory_limit // (1 << 20)),
                        "--", "--spec-type", "none"]
                receipt["native_host_budget_bytes"] = (memory_limit // (1 << 20)) * (1 << 20)
                receipt["native_budget_note"] = "InferSSD host budget accepts integer MiB; rounded down within the exact decimal-GB envelope. Runtime policy uses the exact byte cap."
        receipt["argv"] = argv
        receipt["cwd"] = cwd
        receipt["host_boundary_before"] = host_boundary_snapshot()
        save(output / "host-before.json", receipt["host_boundary_before"])
        save(output / "receipt.json", receipt)
        owned = OwnedProcess(argv, cwd, env, output)
        receipt["launcher_pid"] = owned.p.pid
        if args.engine != "inferssd":
            callback = (lambda line: maya_startup_guard(line, receipt, output)) if args.engine == "maya" else None
            receipt.update(owned.wait_cli(ready, args.load_deadline, args.request_deadline, callback))
        else:
            base = "http://127.0.0.1:" + str(args.port)
            while True:
                if owned.p.poll() is not None:
                    raise RuntimeError("InferSSD exited before readiness")
                if time.monotonic() - owned.started > args.load_deadline:
                    raise TimeoutError("InferSSD load deadline")
                try:
                    if http_json(base + "/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.HTTPError, ValueError):
                    pass
                time.sleep(0.2)
            receipt["ready_seconds"] = time.monotonic() - owned.started
            body = {"prompt": ids, "n_predict": args.tokens, "temperature": 0, "top_p": 1,
                    "top_k": 0, "min_p": 0, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1,
                    "seed": 37, "cache_prompt": False, "return_tokens": True, "stream": False}
            save(output / "request.json", body)
            result_queue = queue.Queue()
            def request():
                try:
                    result_queue.put((True, http_json(base + "/completion", body, args.request_deadline)))
                except Exception as error:
                    result_queue.put((False, error))
            start = time.monotonic()
            threading.Thread(target=request, daemon=True).start()
            try:
                ok, result = result_queue.get(timeout=args.request_deadline)
            except queue.Empty:
                raise TimeoutError("InferSSD request deadline")
            if not ok:
                raise result
            receipt["request_wall_seconds"] = time.monotonic() - start
            save(output / "response.json", result)
            (output / "text.txt").write_text(result.get("content", ""))
            output_ids = result.get("tokens", [])
            native = result.get("timings", {})
            if result.get("tokens_evaluated") != len(ids):
                raise ValueError("InferSSD did not evaluate the supplied prompt token count")
            if not isinstance(output_ids, list) or any(type(x) is not int or x < 0 for x in output_ids) or len(output_ids) != result.get("tokens_predicted"):
                raise ValueError("InferSSD returned token IDs/count disagree; inspect response before using timings")
            if native.get("cache_n") != 0 or native.get("prompt_n") != len(ids) or native.get("predicted_n") != len(output_ids):
                raise ValueError("InferSSD native timing counts or zero-prefix-reuse condition differ")
            if result.get("truncated") is not False:
                raise ValueError("InferSSD response does not prove an untruncated prompt")
            effective = result.get("generation_settings", {})
            for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty"):
                if effective.get(key) != body[key]:
                    raise ValueError("InferSSD effective sampling differs: " + key)
            receipt.update({"output_ids": output_ids, "output_tokens": len(output_ids), "prompt_tokens": len(ids),
                            "native_timings": native, "stop_type": result.get("stop_type"),
                            "decode_forward_steps": max(0, len(output_ids) - 1),
                            "decode_tokens_per_second": (max(0, len(output_ids) - 1) * 1000 / native["predicted_ms"])
                                if native.get("predicted_ms", 0) > 0 else None,
                            "timing_note": "InferSSD starts generation timer after sampling first output (server-context.cpp:3436). Normalize n-1 over predicted_ms, retaining its native n/t rate separately.",
                            "count_reached": len(output_ids) == args.tokens})
            # Timing is already closed. The kernel maintained this lifetime peak;
            # observe it once while the owned native descendant is still alive.
            record_inferssd_memory(receipt,
                inferssd_process_memory(owned, engine["binary"], output / "native-run"), memory_limit)
        if args.engine == "maya":
            if "startup_allocation" not in receipt:
                raise ValueError("Maya completed without validated startup allocation metadata")
            stderr = (output / "stderr.log").read_text()
            receipt.update(maya_result((output / "stdout.log").read_text(), stderr, len(ids), args.tokens))
            resources = time_resources(stderr)
            (output / "time-resource.txt").write_text(resources["raw"])
            receipt["process_resources"] = resources
            receipt["memory_observed_within_budget"] = max(resources["peak_rss_bytes"], resources["peak_footprint_bytes"]) <= memory_limit
            if not receipt["memory_observed_within_budget"]:
                raise ValueError("Maya process peak exceeds the requested memory envelope")
            allocation = receipt.get("allocation_gb")
            if sum(allocation.values()) > memory_limit / 1e9:
                raise ValueError("Maya native dense/state/expert allocations exceed the envelope")
            receipt["timing_eligible"] = True
        elif args.engine == "slotstream":
            stats = json.loads((output / "stats.json").read_text())
            if stats.get("prompt_ids") != ids:
                raise ValueError("SlotStream raw retokenization differs from Origami; not a fair prompt")
            if stats.get("effective_mtp"):
                raise ValueError("SlotStream unexpectedly enabled MTP")
            output_ids = stats["output_ids"]
            native = stats["stats"]
            receipt["native_stats"] = native
            if not isinstance(output_ids, list) or any(type(x) is not int or x < 0 for x in output_ids):
                raise ValueError("SlotStream output IDs are invalid")
            if native.get("decodeTokens") != len(output_ids) or native.get("prefillTokens") != len(ids):
                raise ValueError("SlotStream native prompt/output counts disagree with IDs")
            if native.get("reusedPrefixTokens") != 0:
                raise ValueError("SlotStream reused a prefix in the fresh-prompt comparison")
            for key, value in (("expected", stats.get("effective_expected_peak_gb")), ("observed", native.get("peakMemoryGB"))):
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= memory_limit / 1e9:
                    raise ValueError("SlotStream " + key + " memory peak is absent or exceeds the requested memory envelope")
            if stats.get("sampling", {}).get("greedy") is not True or stats["sampling"].get("requested_max_tokens") != str(args.tokens):
                raise ValueError("SlotStream effective greedy/output settings differ")
            intervals = native.get("interTokenSeconds", [])
            if len(intervals) != max(0, len(output_ids) - 1) or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in intervals):
                raise ValueError("SlotStream callback intervals disagree with output IDs")
            interval_seconds = sum(intervals)
            receipt.update({"output_ids": output_ids, "output_tokens": len(output_ids), "prompt_tokens": len(ids),
                            "count_reached": len(output_ids) == args.tokens, "native_stats": native,
                            "decode_forward_steps": max(0, len(output_ids) - 1),
                            "first_to_last_seconds": interval_seconds if len(intervals) == len(output_ids) - 1 else None,
                            "decode_tokens_per_second": max(0, len(output_ids) - 1) / native["decodeSeconds"]
                                if native.get("decodeSeconds", 0) > 0 else None,
                            "first_to_last_tokens_per_second": len(intervals) / interval_seconds if interval_seconds > 0 else None,
                            "timing_note": "SlotStream native decodeTokens counts first prefill-derived output. Normalize n-1/decodeSeconds; preserve interTokenSeconds for first-to-last timing.",
                            "effective_expected_peak_gb": stats["effective_expected_peak_gb"],
                            "observed_peak_memory_gb": native["peakMemoryGB"]})
        if args.engine == "inferssd-profile":
            machine = json.loads((output / "machine.json").read_text())
            measured = sorted({row["format"].upper() for row in machine.get("compute", {}).get("cpu", [])
                               if any(point.get("tokens") == 1 and point.get("weight_Bps", 0) > 0 for point in row.get("points", []))})
            receipt["cpu_profile_measured_formats"] = measured
            missing = sorted(set(receipt["cpu_profile_selection"]["formats"]) - set(measured))
            if missing:
                raise ValueError("CPU profile lacks measured checkpoint formats: " + ", ".join(missing))
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        if args.engine in ("maya", "inferssd"):
            receipt["timing_eligible"] = False
    finally:
        if owned:
            owned.stop()
            receipt["exit_code"] = owned.p.returncode
            receipt["host_boundary_after"] = host_boundary_snapshot()
            save(output / "host-after.json", receipt["host_boundary_after"])
            receipt["host_boundary_delta"] = host_boundary_delta(receipt["host_boundary_before"], receipt["host_boundary_after"])
        receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "error": receipt.get("error")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
