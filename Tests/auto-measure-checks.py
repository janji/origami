#!/usr/bin/env python3
"""Check `memory_auto_measure` (Sources/origami/AutoMeasure.swift) without a model, GPU or server.

Builds the real AutoMeasure.swift and Jobs.swift (its engine import removed, two error types stubbed) with a harness
that reexecutes itself as the job's child, standing in for `origami tune --memory`: it measures for a moment, or stops
at once with the tuner's own reason when a request arrives. Checked: the rule over every condition (table), the back-off,
that the setting off or an explicit limit always blocks it, the stored lines, and through the real Jobs: an idle stretch
starts the job once, a request during it stops it, and the result is stored and said. That the stopped job's request is
then answered is the tuner's own behaviour, shown on a real model.
Usage: python3 Tests/auto-measure-checks.py [--output folder]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

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

struct Failed: Error, CustomStringConvertible { let description: String }
final class Box: @unchecked Sendable {
    let lock = NSLock()
    var record: [String: Any]?
    var lines: [String] = []
    var starts = 0
}

@main struct AutoMeasureChecks {
    nonisolated(unsafe) static var checks = 0
    static func require(_ ok: Bool, _ what: String) throws {
        if !ok { throw Failed(description: what) }
        checks += 1
    }
    static func waitFor(_ what: String, seconds: Double = 8, _ done: () -> Bool) throws {
        let end = Date().addingTimeInterval(seconds)
        while !done() {
            if Date() > end { throw Failed(description: "timed out: " + what) }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    /// The job's child: `ok` measures and keeps a limit, `none` keeps none, `watch <file>` stops when the file appears.
    static func child(_ arguments: [String]) {
        func outcome(_ name: String, _ reason: String) {
            let data = try! JSONSerialization.data(withJSONObject: ["outcome": name, "reason": reason])
            print("ORIGAMI_TUNING_OUTCOME " + String(decoding: data, as: UTF8.self))
        }
        print("Measuring memory limits 20.0 GB, 24.0 GB: each reloads the model; a request from an app stops it")
        if arguments[0] == "watch" {
            let end = Date().addingTimeInterval(20)
            while Date() < end {
                if FileManager.default.fileExists(atPath: arguments[1]) {
                    print("Reloading your saved memory limit")
                    print("Stopped: A request from an app arrived. Nothing was changed.")
                    outcome("aborted", "A request from an app arrived")
                    return
                }
                Thread.sleep(forTimeInterval: 0.01)
            }
        }
        Thread.sleep(forTimeInterval: 0.3)
        outcome("comparison", "24.0 GB was fastest")
    }

    nonisolated(unsafe) static let measured: [String: Any] = ["suggested_bytes": NSNumber(value: 24_000_000_000), "stale": NSNull()]

    static func rules() throws {
        let idle = AutoMeasure.idle, now = Date()
        let go = AutoMeasure.Conditions(enabled: true, limitSet: false, jobRunning: false, measurement: nil, power: nil, lastTry: nil)
        func with(_ change: (inout AutoMeasure.Conditions) -> Void) -> AutoMeasure.Conditions {
            var c = go
            change(&c)
            return c
        }
        let table: [(String, AutoMeasure.Conditions, TimeInterval, Bool)] = [
            ("all conditions hold", go, idle, true),
            ("setting off", with { $0.enabled = false }, idle, false),
            ("explicit memory_limit", with { $0.limitSet = true }, idle, false),
            ("current measurement", with { $0.measurement = measured }, idle, false),
            ("stale measurement", with { $0.measurement = ["suggested_bytes": NSNumber(value: 24_000_000_000), "stale": "another context size"] }, idle, true),
            ("measurement that kept no limit", with { $0.measurement = ["suggested_bytes": NSNull(), "stale": NSNull()] }, idle, true),
            ("another job runs", with { $0.jobRunning = true }, idle, false),
            ("on battery", with { $0.power = "the Mac is on battery" }, idle, false),
            ("a request came recently", go, idle - 1, false),
            ("tried an hour ago", with { $0.lastTry = now.addingTimeInterval(-3600) }, idle, false),
            ("tried a day and a minute ago", with { $0.lastTry = now.addingTimeInterval(-AutoMeasure.backOff - 60) }, idle, true),
        ]
        for (what, conditions, idleFor, starts) in table {
            try require((AutoMeasure.blocked(conditions, idleFor: idleFor, now: now) == nil) == starts, "rule: " + what)
        }
        // Whatever else holds, the setting off or a limit of the user's blocks it.
        for idleFor in [0, idle, 1e9] {
            for c in [with { $0.enabled = false }, with { $0.limitSet = true }, with { $0.enabled = false; $0.limitSet = true }] {
                try require(AutoMeasure.blocked(c, idleFor: idleFor, now: now) != nil, "setting off or explicit limit always blocks")
            }
        }
        // Why it waits is told while a measurement is due (power, a request, the back-off), never when it is off or current.
        for (what, conditions, wanted) in [
            ("on battery", with { $0.power = "the Mac is on battery" }, "the Mac is on battery" as String?),
            ("setting off", with { $0.enabled = false }, nil), ("current measurement", with { $0.measurement = measured }, nil),
        ] {
            var told: String?? = .none
            AutoMeasure().tick(conditions, start: { _ in }, store: { _ in }, log: { _ in }, waiting: { told = $0 })
            try require(told == .some(wanted), "waiting: " + what)
        }
        let backOff = AutoMeasure.blocked(with { $0.lastTry = now.addingTimeInterval(-3600) }, idleFor: idle, now: now) ?? ""
        try require(backOff.hasPrefix("tried at ") && backOff.contains("tries again after"), "waiting: the back-off says when it tries again")
        try require(AutoMeasure.waitingLine("the Mac is on battery", since: now).hasPrefix("memory: not measured yet: the Mac is on battery since "), "waiting line")
        try require(AutoMeasure.idle >= 1 && AutoMeasure.backOff == 86_400, "constants")
        try require(AutoMeasure.line(state: "done", reason: "24.0 GB was fastest", measured: 24_000_000_000) == "memory: measured automatically while idle: 24.0 GB", "line: measured")
        try require(AutoMeasure.line(state: "interrupted", reason: "A request from an app arrived", measured: nil) == "memory: automatic measurement stopped: a request from an app arrived; will try again tomorrow", "line: stopped")
        try require(AutoMeasure.line(state: "done", reason: "No limit kept", measured: nil) == "memory: automatic measurement found no limit this Mac kept; will try again tomorrow", "line: none kept")
        try require(AutoMeasure.line(state: "failed", reason: "The model did not load.", measured: nil) == "memory: automatic measurement did not finish (the model did not load); will try again tomorrow", "line: failed")
    }

    /// One server's worth: a front that asks every 50 ms, with the given conditions besides the job and the last try.
    static func flow(_ scenario: [String], base: AutoMeasure.Conditions, seconds: Double, request: String? = nil, cancel: Bool = false) throws -> (Box, Double?) {
        let jobs = Jobs(options: []), auto = AutoMeasure(), box = Box()
        let end = Date().addingTimeInterval(seconds)
        var stopSeconds: Double?
        var requested: Date?
        while Date() < end {
            var c = base
            c.jobRunning = jobs.report?["state"] as? String == "running"
            c.lastTry = box.lock.withLock { (box.record?["date"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) } }
            auto.tick(c, start: { ended in
                try jobs.start(kind: "tune", title: "Finding the fastest memory limit while idle", arguments: ["--child"] + scenario) { state in
                    ended(state, jobs.report?["message"] as? String, scenario[0] == "ok" ? 24_000_000_000 : nil)
                }
                box.lock.withLock { box.starts += 1 }
            }, store: { record in box.lock.withLock { box.record = record } }, log: { line in box.lock.withLock { box.lines.append(line) } })
            // The job's Cancel while it runs.
            if cancel, requested == nil, jobs.report?["state"] as? String == "running" {
                Thread.sleep(forTimeInterval: 0.1)
                jobs.cancel()
                requested = Date()
            }
            // A request while it runs: noted by the front, seen by the job.
            if let request, requested == nil, jobs.report?["state"] as? String == "running" {
                Thread.sleep(forTimeInterval: 0.1)
                auto.noteRequest()
                requested = Date()
                FileManager.default.createFile(atPath: request, contents: Data())
            }
            if let requested, stopSeconds == nil, jobs.report?["state"] as? String != "running" { stopSeconds = Date().timeIntervalSince(requested) }
            Thread.sleep(forTimeInterval: 0.05)
        }
        try waitFor("job end") { jobs.report?["state"] as? String != "running" }
        return (box, stopSeconds)
    }

    static func main() {
        if CommandLine.arguments.count > 2, CommandLine.arguments[1] == "--child" { return child(Array(CommandLine.arguments.dropFirst(2))) }
        do {
            try rules()
            let go = AutoMeasure.Conditions(enabled: true, limitSet: false, jobRunning: false, measurement: nil, power: nil, lastTry: nil)
            // Idle (the harness runs with a 1 s stretch): one start, its result stored and said, and no second start.
            var (box, _) = try flow(["ok"], base: go, seconds: AutoMeasure.idle + 2)
            try require(box.starts == 1, "idle starts the job once (started \(box.starts))")
            try require(box.record?["line"] as? String == "memory: measured automatically while idle: 24.0 GB", "the result is stored")
            try require(box.lines == [AutoMeasure.startedLine, "memory: measured automatically while idle: 24.0 GB"], "the log says it started and what it measured")
            // A request during the job stops it: its reason stored, and no new try within a day.
            let file = NSTemporaryDirectory() + "auto-measure-request-" + UUID().uuidString
            let stopped: Double?
            (box, stopped) = try flow(["watch", file], base: go, seconds: AutoMeasure.idle * 3 + 2, request: file)
            try? FileManager.default.removeItem(atPath: file)
            try require(box.starts == 1, "a stopped run is not retried at the next idle stretch (started \(box.starts))")
            try require(box.record?["line"] as? String == "memory: automatic measurement stopped: a request from an app arrived; will try again tomorrow", "the stop is stored with its reason")
            try require((stopped ?? 99) < 2, "the job ends as soon as it sees the request")
            // Cancelled: its line says so, not that it still runs.
            (box, _) = try flow(["watch", NSTemporaryDirectory() + "never-" + UUID().uuidString], base: go, seconds: AutoMeasure.idle + 1.5, cancel: true)
            try require(box.starts == 1, "a cancelled run is not retried (started \(box.starts))")
            try waitFor("the cancelled run's line") { box.lock.withLock { box.record?["line"] as? String } == "memory: automatic measurement stopped: cancelled; will try again tomorrow" }
            // A server that stopped (or crashed) during a run: its stored line is settled, its date kept.
            let running: [String: Any] = ["date": "2026-11-16T00:00:00Z", "line": AutoMeasure.startedLine]
            try require(AutoMeasure.settled(running)?["line"] as? String == "memory: automatic measurement stopped: the server stopped; will try again tomorrow"
                && AutoMeasure.settled(running)?["date"] as? String == "2026-11-16T00:00:00Z", "a run the server stopped is settled")
            try require(AutoMeasure.settled(["line": "memory: measured automatically while idle: 24.0 GB"]) == nil && AutoMeasure.settled(nil) == nil, "a finished run is left as it is")
            // Off, or a limit of the user's: never started, however idle.
            for (what, c) in [("setting off", AutoMeasure.Conditions(enabled: false)), ("explicit limit", AutoMeasure.Conditions(enabled: true, limitSet: true))] {
                (box, _) = try flow(["ok"], base: c, seconds: AutoMeasure.idle + 0.5)
                try require(box.starts == 0 && box.record == nil, what + " never starts it")
            }
            print("{\"checks\": \(checks), \"failures\": 0}")
        } catch {
            FileHandle.standardError.write(Data(("FAIL: \(error)\n").utf8))
            exit(1)
        }
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or Path(tempfile.mkdtemp(prefix='origami-auto-measure-checks-'))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = (root / 'Sources/origami/Jobs.swift').read_text()
    if jobs.count('import OrigamiEngine\n') != 1:
        raise RuntimeError('Expected exactly one engine import in Jobs.swift')
    (output / 'Jobs.swift').write_text(jobs.replace('import OrigamiEngine\n', '', 1))
    (output / 'Stubs.swift').write_text(STUBS)
    (output / 'Harness.swift').write_text(HARNESS)
    binary = output / 'auto-measure-checks'
    compiled = subprocess.run(['xcrun', 'swiftc', '-swift-version', '6', '-enable-bare-slash-regex', '-parse-as-library',
                               str(root / 'Sources/origami/AutoMeasure.swift'), str(output / 'Jobs.swift'),
                               str(output / 'Stubs.swift'), str(output / 'Harness.swift'), '-o', str(binary)],
                              text=True, capture_output=True, timeout=300)
    if compiled.returncode:
        print(compiled.stdout + compiled.stderr)
        return 1
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_AUTO_MEASURE_IDLE'] = '1'
    tested = subprocess.run([str(binary)], text=True, capture_output=True, timeout=120, env=env)
    print(tested.stdout + tested.stderr, end='')
    if tested.returncode == 0:
        print(json.dumps({'passed': True, 'output': str(output)}))
    return tested.returncode


if __name__ == '__main__':
    raise SystemExit(main())
