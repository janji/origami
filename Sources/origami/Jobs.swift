import Foundation
import OrigamiEngine

/// The dashboard's long tasks: preparing 3- or 2-bit experts (`origami setup`) and adding a model (`origami model add`,
/// which downloads and converts when it must). Each runs this same program as a child process with the server's settings
/// options, one at a time, so it does exactly what the command does in Terminal; its output is kept for the page, and a
/// `  label  42%` or `  12.3 / 104.0 GB` line is its progress. Moving and removing files (`Storage`) run in this process
/// instead (`run`), reporting the same way.
final class Jobs: @unchecked Sendable {
    private struct TuningOutcome: Decodable {
        let outcome: String
        let reason: String?
    }
    private struct Job {
        let kind: String, title: String, started: Date, process: Process?
        var state = "running", stage = "Starting", percent: Double?, lines: [String] = [], finished: Date?
        var error: String?, added: String?, message: String?, cancelled = false
        var tuningOutcome: TuningOutcome?
    }
    private let lock = NSLock()
    private var job: Job?
    private let options: [String]
    private let log: @Sendable (String) -> Void

    /// `options`: the server's settings options (`--config=…`, `--store=…`), passed to every job; `log`: the server's
    /// log, which gets one line as each job ends.
    init(options: [String], log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.options = options
        self.log = log
    }

    /// `then` runs once the job has ended, however it ended, with its state (`done`, `failed`, `cancelled`, a process
    /// killed is `failed`); the line it returns, if any, becomes the job's message (the front reloads the model when a
    /// hot-copy job changed what it reads, and the saved memory limit after a memory search).
    func start(kind: String, title: String, arguments: [String], then: (@Sendable (String) -> String?)? = nil) throws {
        try lock.withLock {
            if let job, job.state == "running" { throw ServingFailure("\(job.title) is still running", status: 409) }
            let process = Process()
            let output = Pipe()
            process.executableURL = Bundle.main.executableURL
            process.arguments = arguments + options
            process.standardOutput = output
            process.standardError = output
            process.standardInput = FileHandle.nullDevice
            process.environment = ProcessInfo.processInfo.environment.merging(["ORIGAMI_JOB": "1"]) { $1 }
            try process.run()
            job = Job(kind: kind, title: title, started: Date(), process: process)
            Thread.detachNewThread { [self] in
                // Always told, even if the page dismissed the job or another started meanwhile (the memory search's
                // restore depends on it).
                let state = follow(output.fileHandleForReading, process)
                if let line = then?(state) {
                    lock.withLock { if job?.process === process { job?.message = line } }
                }
                ended { $0.process === process }
            }
        }
    }

    /// Runs `work` on its own thread as the job: it reports a stage and a share done with `step`, stops when `cancelled`
    /// says so, and returns the line the page shows once it is done.
    func run(
        kind: String, title: String,
        _ work: @escaping @Sendable (_ step: @escaping (String, Double?) -> Void, _ cancelled: @escaping () -> Bool) throws -> String
    ) throws {
        try lock.withLock {
            if let job, job.state == "running" { throw ServingFailure("\(job.title) is still running", status: 409) }
            job = Job(kind: kind, title: title, started: Date(), process: nil)
        }
        Thread.detachNewThread { [self] in
            let result = Result {
                try work(
                    { stage, percent in
                        self.lock.withLock {
                            self.job?.stage = stage
                            self.job?.percent = percent
                        }
                    },
                    { self.lock.withLock { self.job?.cancelled ?? true } })
            }
            lock.withLock {
                guard var job else { return }
                job.finished = Date()
                switch result {
                case .success(let line):
                    job.state = "done"
                    job.percent = 1
                    job.message = line
                case .failure(let error):
                    job.state = job.cancelled ? "cancelled" : "failed"
                    job.error = (error as? CLIError)?.description ?? ServingFailure.from(error).description
                }
                self.job = job
            }
            ended { $0.process == nil }
        }
    }

    /// One line in the server's log for the job that just ended: its title, state and outcome.
    private func ended(_ mine: (Job) -> Bool) {
        guard
            let line = lock.withLock({ job.flatMap { mine($0) ? $0 : nil } }).map({ job in
                "\(job.title.lowercased()): \(job.state)"
                    + ((job.state == "done" ? job.message : job.error ?? job.message).map { ": " + $0 } ?? "")
            })
        else { return }
        log(line)
    }

    /// Stops a running job (a pack or conversion left half done is removed by the next run), or forgets a finished one.
    func cancel() {
        lock.withLock {
            guard var job else { return }
            guard job.state == "running" else {
                self.job = nil
                return
            }
            job.cancelled = true
            self.job = job
            job.process?.terminate()
        }
    }

    /// The current or last job for `GET v1/jobs`, or nil: a finished one is kept, with how it ended (`ended`, seconds since
    /// 1970), until the next starts or the page dismisses it. `started` tells one job from the next, so a page acts on a
    /// job's end only when it saw that job run, not on the last job still reported when it loads.
    var report: [String: Any]? {
        lock.withLock {
            guard let job else { return nil }
            var object: [String: Any] = [
                "kind": job.kind, "title": job.title, "state": job.state, "stage": job.stage, "lines": job.lines,
                "started": job.started.timeIntervalSince1970,
                "seconds": Int((job.finished ?? Date()).timeIntervalSince(job.started)),
            ]
            if let finished = job.finished { object["ended"] = finished.timeIntervalSince1970 }
            if let percent = job.percent { object["percent"] = percent }
            if let error = job.error { object["error"] = error }
            if let added = job.added { object["added"] = added }
            if let message = job.message { object["message"] = message }
            if let outcome = job.tuningOutcome { object["outcome"] = outcome.outcome }
            return object
        }
    }

    /// Reads the job's output until it exits; its final state.
    private func follow(_ handle: FileHandle, _ process: Process) -> String {
        var pending = Data()
        while case let chunk = handle.availableData, !chunk.isEmpty {
            pending.append(chunk)
            while let end = pending.firstIndex(of: 10) {
                let line = String(decoding: pending[pending.startIndex..<end], as: UTF8.self)
                pending.removeSubrange(pending.startIndex...end)
                note(line)
            }
        }
        if !pending.isEmpty { note(String(decoding: pending, as: UTF8.self)) }
        process.waitUntilExit()
        return lock.withLock { () -> String in
            guard var job, job.process === process else { return process.terminationStatus == 0 ? "done" : "failed" }
            job.finished = Date()
            job.state = job.cancelled ? "cancelled" : process.terminationStatus == 0 ? "done" : "failed"
            if job.state == "done", job.kind == "tune" {
                let outcome = job.tuningOutcome
                let reason = outcome?.reason?.trimmingCharacters(in: .whitespacesAndNewlines)
                if let reason, !reason.isEmpty { job.message = reason }
                switch outcome?.outcome {
                case "aborted":
                    job.state = "interrupted"
                    job.message = job.message ?? "Tuning was interrupted. Settings were not changed."
                case "too_noisy":
                    job.state = "incomplete"
                    job.message = job.message ?? "Measurements varied too much to confirm an improvement. Settings were not changed."
                case "incomplete":
                    job.state = "incomplete"
                    job.message = job.message ?? "Tuning did not finish all measurements. Settings were not changed."
                case "comparison", "faster", "already_fastest", "applied", "undone", "rechecked", "unchanged": break
                default:
                    job.state = "incomplete"
                    job.message = job.message ?? "Tuning ended without a reported outcome."
                }
                if job.state != "done" {
                    job.percent = nil
                    job.stage = job.message ?? "Tuning incomplete"
                }
            }
            if job.state == "done" { job.percent = 1 }
            if job.state == "failed", job.error == nil {
                job.error = job.lines.last ?? "it stopped (exit status \(process.terminationStatus))"
            }
            self.job = job
            return job.state
        }
    }

    private func note(_ raw: String) {
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return }
        let step = Self.parse(line)
        lock.withLock {
            guard var job else { return }
            // Only the tuner emits this terminal protocol. Keep it out of the progress band and human log;
            // the process exit status and cancellation still determine whether it can count as completion.
            let marker = "ORIGAMI_TUNING_OUTCOME"
            if job.kind == "tune", line == marker || line.hasPrefix(marker + " ") {
                job.tuningOutcome =
                    (try? JSONDecoder().decode(TuningOutcome.self, from: Data(line.dropFirst(marker.count).utf8)))
                    ?? TuningOutcome(outcome: "incomplete", reason: "Tuning returned an unreadable outcome.")
                self.job = job
                return
            }
            if let error = step.error {
                job.error = error
            } else if let percent = step.percent {
                job.percent = percent
                if let stage = step.stage { job.stage = stage }
            } else {
                if let added = step.added { job.added = added }
                // A finished step keeps its 100%; a new one starts without a figure until its first progress line.
                if !line.hasSuffix(" done") && line != "done" {
                    job.percent = nil
                    job.stage = line
                }
                job.lines = Array((job.lines + [line]).suffix(8))
            }
            self.job = job
        }
    }

    /// One output line: progress (`3-bit pack  42%`, `12.3 / 104.0 GB  (85 MB/s)`), an error (`origami: …`, as `main`
    /// prints one), the name `model add` gave (`Added <name>: …`), or nothing (a step to show as it is).
    static func parse(_ line: String) -> (percent: Double?, stage: String?, error: String?, added: String?) {
        if let match = line.firstMatch(of: /^(.*?)\s+(\d{1,3})%$/), let value = Double(match.2) {
            return (min(1, value / 100), String(match.1), nil, nil)
        }
        if let match = line.firstMatch(of: /^([\d.]+) \/ ([\d.]+) GB/), let done = Double(match.1), let total = Double(match.2), total > 0 {
            return (min(1, done / total), "Downloading", nil, nil)
        }
        if line.hasPrefix("origami: ") { return (nil, nil, String(line.dropFirst("origami: ".count)), nil) }
        if let match = line.firstMatch(of: /^(?:Added|Updated) ([A-Za-z0-9._-]+):/) { return (nil, nil, nil, String(match.1)) }
        return (nil, nil, nil, nil)
    }
}
