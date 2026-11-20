#!/usr/bin/env python3
"""Exercise the real dashboard Jobs subprocess reader without loading OrigamiEngine.

Only its unused engine import is removed; two error types receive interface stubs.
The fixture reexecutes itself through Jobs.start, so process exit, pipes, EOF,
cancellation, protocol parsing and completion callbacks use the production code.
Usage: python3 Tests/tuning-job-checks.py --output /path/to/artifacts
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


STUBS = r'''
import Foundation
struct CLIError: Error, CustomStringConvertible { let description: String }
struct ServingFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String, status: Int = 400) { self.description = description }
    static func from(_ error: Error) -> ServingFailure { ServingFailure(String(describing: error)) }
}
'''

HARNESS = r'''
import Foundation
import Darwin

private struct CheckFailure: Error, CustomStringConvertible { let description: String }
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() { lock.withLock { count += 1 } }
    var value: Int { lock.withLock { count } }
    private var heard: String?
    var state: String? {
        get { lock.withLock { heard } }
        set { lock.withLock { heard = newValue } }
    }
}

@main struct JobFixture {
    static let marker = "ORIGAMI_TUNING_OUTCOME "
    static func write(_ text: String) { FileHandle.standardOutput.write(Data(text.utf8)) }
    static func outcome(_ name: String, reason: String? = nil, newline: Bool = true) throws {
        var object = ["outcome": name]
        if let reason { object["reason"] = reason }
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        write(marker + String(decoding: data, as: UTF8.self) + (newline ? "\n" : ""))
    }
    static func emit(_ scenario: String) throws {
        write("Measuring  100%\n")
        switch scenario {
        case "missing": write("Comparison stopped before its outcome.\n")
        case "malformed": write(marker + "{broken}\n")
        case "trailing": try outcome("aborted", reason: "No trailing newline", newline: false)
        case "nonzero":
            try outcome("comparison")
            write("origami: fixture child failed\n")
            exit(7)
        case "nonzero-aborted":
            try outcome("aborted", reason: "Interrupted result")
            exit(7)
        case "cancel-zero":
            signal(SIGTERM, SIG_IGN)
            try outcome("comparison", reason: "Child will exit zero")
            Thread.sleep(forTimeInterval: 0.4)
        case "cancel-aborted":
            try outcome("aborted", reason: "Child still running")
            Thread.sleep(forTimeInterval: 5)
        case "killed":
            write("Loading at 20.0GB\n")
            kill(getpid(), SIGKILL)
        case "ordinary": write("Added fixture-model: ready\n")
        case "aborted": try outcome(scenario, reason: "The loaded model changed. Settings unchanged.")
        case "incomplete": try outcome(scenario, reason: "Two samples per mode were not completed.")
        case "too_noisy": try outcome(scenario, reason: "Variation prevented a reliable winner.")
        default: try outcome(scenario)
        }
    }
    static func require(_ ok: Bool, _ description: String) throws {
        if !ok { throw CheckFailure(description: description) }
    }
    static func waitFor(_ description: String, _ predicate: () -> Bool) throws {
        let end = Date().addingTimeInterval(8)
        while !predicate() {
            try require(Date() < end, "Timed out: " + description)
            Thread.sleep(forTimeInterval: 0.005)
        }
    }
    static func check(_ scenario: String, kind: String = "tune", expected: String,
                      reason: String? = nil, cancel: Bool = false) throws -> [String: Any] {
        let jobs = Jobs(options: []), calls = Counter()
        try jobs.start(kind: kind, title: "Fixture " + scenario,
                       arguments: ["--emit-outcome", scenario], then: { state in calls.increment(); calls.state = state; return nil })
        if cancel {
            try waitFor("child terminal marker before cancellation") { jobs.report?["outcome"] != nil }
            try require(jobs.report?["state"] as? String == "running", scenario + ": child finished before cancellation")
            jobs.cancel()
        }
        try waitFor("terminal job " + scenario) { jobs.report?["state"] as? String != "running" }
        guard let report = jobs.report else { throw CheckFailure(description: scenario + ": report disappeared") }
        try require(report["state"] as? String == expected, scenario + ": expected " + expected + ", got " + String(describing: report["state"]))
        if expected == "done" {
            try waitFor("completion callback " + scenario) { calls.value > 0 }
            try require(calls.value == 1 && report["percent"] as? Double == 1, scenario + ": successful completion changed")
        } else if expected == "cancelled" {
            // Since 3.1 a cancelled job is heard too (the front restores a memory trial whatever ended its search).
            try waitFor("completion callback " + scenario) { calls.value > 0 }
            Thread.sleep(forTimeInterval: 0.025)
            try require(calls.value == 1 && calls.state == "cancelled", scenario + ": cancellation was not heard once, as cancelled")
        } else {
            // Since 3.1 the callback also hears a failed or incomplete end (with the state), to keep the job's outcome.
            try waitFor("completion callback " + scenario) { calls.value > 0 }
            try require(calls.value == 1, scenario + ": terminal failure invoked the completion callback twice")
        }
        if expected == "interrupted" || expected == "incomplete" {
            try require(report["percent"] == nil, scenario + ": incomplete job retained100% progress")
            try require(!(report["message"] as? String ?? "").isEmpty, scenario + ": missing terminal reason")
            try require(report["stage"] as? String == report["message"] as? String, scenario + ": reason was not the terminal stage")
        }
        if let reason { try require(report["message"] as? String == reason, scenario + ": terminal reason changed") }
        let lines = report["lines"] as? [String] ?? []
        if kind == "tune" {
            try require(!lines.contains(where: { $0.hasPrefix(marker) }), scenario + ": protocol leaked into human log")
            try require(!(report["stage"] as? String ?? "").hasPrefix(marker), scenario + ": protocol leaked into stage")
        } else {
            try require(report["outcome"] == nil, scenario + ": non-tuning job consumed tuning protocol")
            if scenario == "aborted" { try require(lines.contains(where: { $0.hasPrefix(marker) }), "non-tuning ordinary output was hidden") }
            if scenario == "ordinary" { try require(report["added"] as? String == "fixture-model", "model-add parsing changed") }
        }
        if scenario == "nonzero" { try require(report["error"] as? String == "fixture child failed", "nonzero exit lost its error") }
        return ["scenario": scenario, "kind": kind, "passed": true, "state": expected,
                "completion_callbacks": calls.value, "report": report]
    }
    static func main() {
        do {
            if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--emit-outcome" {
                try emit(CommandLine.arguments[2]); return
            }
            var results: [[String: Any]] = []
            results.append(try check("aborted", expected: "interrupted", reason: "The loaded model changed. Settings unchanged."))
            results.append(try check("incomplete", expected: "incomplete", reason: "Two samples per mode were not completed."))
            results.append(try check("too_noisy", expected: "incomplete", reason: "Variation prevented a reliable winner."))
            for name in ["comparison", "faster", "already_fastest", "unchanged", "undone"] {
                results.append(try check(name, expected: "done"))
            }
            for name in ["missing", "malformed", "unknown_future_outcome"] {
                results.append(try check(name, expected: "incomplete"))
            }
            results.append(try check("trailing", expected: "interrupted", reason: "No trailing newline"))
            results.append(try check("nonzero", expected: "failed"))
            results.append(try check("nonzero-aborted", expected: "failed"))
            results.append(try check("cancel-zero", expected: "cancelled", cancel: true))
            results.append(try check("cancel-aborted", expected: "cancelled", cancel: true))
            // A child killed (no outcome, no restore of its own) is heard as failed.
            results.append(try check("killed", expected: "failed"))
            results.append(try check("ordinary", kind: "add", expected: "done"))
            results.append(try check("aborted", kind: "prepare", expected: "done"))
            results.append(try check("nonzero", kind: "prepare", expected: "failed"))
            let summary: [String: Any] = ["testCount": results.count, "failureCount": 0, "tests": results]
            write(String(decoding: try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self) + "\n")
        } catch {
            FileHandle.standardError.write(Data(("FAIL: " + String(describing: error) + "\n").utf8)); exit(1)
        }
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='origami-jobs-checks-'))
    output.mkdir(parents=True, exist_ok=True)
    source = (root / 'Sources/origami/Jobs.swift').read_bytes()
    text = source.decode()
    if text.count('import OrigamiEngine\n') != 1:
        raise RuntimeError('Expected exactly one engine import; review source isolation before running')
    (output / 'Jobs.swift').write_text(text.replace('import OrigamiEngine\n', '', 1))
    (output / 'Stubs.swift').write_text(STUBS)
    (output / 'Harness.swift').write_text(HARNESS)
    binary = output / 'tuning-job-fixture'
    compile_command = ['xcrun', 'swiftc', '-swift-version', '6', '-enable-bare-slash-regex', '-parse-as-library',
                       str(output / 'Jobs.swift'), str(output / 'Stubs.swift'), str(output / 'Harness.swift'), '-o', str(binary)]
    start = time.monotonic()
    compiled = subprocess.run(compile_command, text=True, capture_output=True, timeout=120)
    (output / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    receipt = {'source': str(root / 'Sources/origami/Jobs.swift'), 'source_sha256': hashlib.sha256(source).hexdigest(),
               'scope': 'Actual Jobs subprocess implementation; only engine import removed and error-type interfaces stubbed. No GPU, model or production access.',
               'compile_returncode': compiled.returncode, 'compile_seconds': time.monotonic() - start}
    if compiled.returncode == 0:
        start = time.monotonic()
        tested = subprocess.run([str(binary)], text=True, capture_output=True, timeout=40)
        (output / 'fixture.stdout.json').write_text(tested.stdout)
        (output / 'fixture.stderr.log').write_text(tested.stderr)
        receipt.update(test_returncode=tested.returncode, test_seconds=time.monotonic() - start)
        if tested.returncode == 0:
            result = json.loads(tested.stdout)
            receipt.update(testCount=result['testCount'], failureCount=result['failureCount'])
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({**receipt, 'output': str(output)}, indent=2))
    return compiled.returncode or receipt.get('test_returncode', 1)


if __name__ == '__main__':
    raise SystemExit(main())
