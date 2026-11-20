import Foundation

/// `memory_auto_measure`: a running server in automatic memory mode starts the dashboard's "Fastest memory limit" job
/// (`origami tune --memory`, unchanged) by itself, once it has had no request for `idle`, when no limit is measured for
/// this model and context or the one measured went stale. The job stops at once when a request arrives and reloads the
/// saved state; its result is stored where a manual run stores it, and the reload it ends with serves with it. The front
/// (`ModelFront`) asks `tick` once a minute; whatever happened is one stored line (`record`), shown by the log,
/// `origami status`, `/v1/status` (`memory_learned.auto_measure`) and the dashboard's Memory line. Foundation only, so
/// `Tests/auto-measure-checks.py` builds it with `Jobs` and no engine.
final class AutoMeasure: @unchecked Sendable {
    /// No request for this long before measuring: tens of minutes, so nobody waits on a reload. `ORIGAMI_AUTO_MEASURE_IDLE`
    /// (seconds) shortens it, for tests only.
    static let idle: TimeInterval =
        ProcessInfo.processInfo.environment["ORIGAMI_AUTO_MEASURE_IDLE"].flatMap(Double.init).map { max(1, $0) } ?? 30 * 60
    /// After a try, whatever its end, none for a day.
    static let backOff: TimeInterval = 86_400
    /// How often the front looks.
    static var every: TimeInterval { min(60, idle / 4) }

    /// What decides whether to measure now; `measurement` is what `Tuning.memoryAdvice` returns (nil: none).
    struct Conditions {
        var enabled = false, limitSet = false, jobRunning = false
        var measurement: [String: Any]?
        /// Why the Mac is not on power (`DriveSpeed.refusal`: on battery, low power mode), or nil.
        var power: String?
        var lastTry: Date?
    }

    /// No current measurement: none, a stale one, or one that kept no limit.
    static func needed(_ measurement: [String: Any]?) -> Bool {
        guard let measurement else { return true }
        return measurement["stale"] is String || (measurement["suggested_bytes"] as? NSNumber) == nil
    }

    /// Why not to measure now, or nil to start. An explicit limit or the setting off always blocks it.
    static func blocked(_ c: Conditions, idleFor: TimeInterval, now: Date = Date()) -> String? {
        if !c.enabled { return "memory_auto_measure is off" }
        if c.limitSet { return "memory_limit is set" }
        if !needed(c.measurement) { return "the measured limit is current" }
        if c.jobRunning { return "another job runs" }
        if let power = c.power { return power }
        if idleFor < idle { return "a request came within \(Int(idle / 60)) minutes" }
        if let last = c.lastTry, now.timeIntervalSince(last) < backOff {
            return "tried at \(shown(last)); tries again after \(shown(last.addingTimeInterval(backOff)))"
        }
        return nil
    }

    /// Whether `blocked` waits for something to pass (power, a request, a job, the back-off), not off or not needed.
    static func waits(_ c: Conditions) -> Bool { c.enabled && !c.limitSet && needed(c.measurement) }

    /// A time as the lines show it: the time today, with the day otherwise.
    static func shown(_ date: Date, now: Date = Date()) -> String {
        date.formatted(date: Calendar.current.isDate(date, inSameDayAs: now) ? .omitted : .abbreviated, time: .shortened)
    }

    /// The line while it waits (`origami status`, the dashboard's measure panel): "memory: not measured yet: on battery
    /// since 14:02".
    static func waitingLine(_ reason: String, since: Date) -> String {
        "memory: not measured yet: \(reason)" + (reason.hasPrefix("tried at") ? "" : " since \(shown(since))")
    }

    /// The stored line for a run that ended `state` (`Jobs`: done, interrupted, incomplete, failed) with the job's
    /// `reason`, and the limit then measured for this model and context (nil: none current).
    static func line(state: String, reason: String?, measured: UInt64?) -> String {
        if state == "done", let measured {
            return "memory: measured automatically while idle: " + String(format: "%.1f GB", Double(measured) / 1e9)
        }
        let why = (reason ?? "").trimmingCharacters(in: CharacterSet(charactersIn: " .\n"))
        let said = why.isEmpty ? "" : why.prefix(1).lowercased() + why.dropFirst()
        let what =
            state == "done"
            ? "found no limit this Mac kept"
            : state == "interrupted" || state == "cancelled"
                ? "stopped" + (said.isEmpty ? (state == "cancelled" ? ": cancelled" : "") : ": " + said)
                : "did not finish" + (said.isEmpty ? "" : " (" + said + ")")
        return "memory: automatic measurement \(what); will try again tomorrow"
    }
    static let startedLine = "memory: measuring automatically while idle; a request stops it"
    /// While the job's closing reload brings the saved limit back, after it ended `state`.
    static func restoringLine(_ state: String) -> String {
        "memory: automatic measurement \(state == "cancelled" ? "cancelled" : "ended"); restoring your saved limit"
    }

    /// A stored try that still says it runs once its server stops (or stopped without saying: a crash), with the line
    /// saying so and its date kept; nil when there is nothing to settle.
    static func settled(_ record: [String: Any]?) -> [String: Any]? {
        guard let record, let said = record["line"] as? String, said == startedLine || said.hasSuffix("restoring your saved limit")
        else { return nil }
        return record.merging(["line": line(state: "interrupted", reason: "the server stopped", measured: nil)]) { $1 }
    }

    private final class Ended: @unchecked Sendable {
        let lock = NSLock()
        var done = false
    }
    private let lock = NSLock()
    private var lastRequest = Date()
    /// A request came (to the model, not a status read).
    func noteRequest() { lock.withLock { lastRequest = Date() } }
    var idleFor: TimeInterval { lock.withLock { Date().timeIntervalSince(lastRequest) } }

    /// Starts the job when nothing blocks it: the try is stored first (so the back-off holds even if the job is cancelled
    /// or the server stops), then its line when it ends. `start` runs the job and calls its argument once it has ended,
    /// with its state, reason and the limit then measured. True when it started. `waiting` is told at each look why it
    /// did not start while a measurement is due (nil: none is due, or it started).
    @discardableResult
    func tick(
        _ conditions: Conditions,
        start: (_ ended: @escaping @Sendable (_ state: String, _ reason: String?, _ measured: UInt64?) -> String) throws -> Void,
        store: @escaping @Sendable (_ record: [String: Any]) -> Void, log: @escaping @Sendable (String) -> Void,
        waiting: (String?) -> Void = { _ in }
    ) -> Bool {
        if let reason = Self.blocked(conditions, idleFor: idleFor) {
            waiting(Self.waits(conditions) ? reason : nil)
            return false
        }
        let date = ISO8601DateFormatter().string(from: Date())
        let ended = Ended()
        do {
            try start { state, reason, measured in
                let line = Self.line(state: state, reason: reason, measured: measured)
                ended.lock.withLock {
                    ended.done = true
                    store(["date": date, "line": line])
                }
                log(line)
                return line
            }
        } catch {
            // Not now (the model is busy or loading): asked again at the next look.
            let said = "\(error)".trimmingCharacters(in: CharacterSet(charactersIn: " ."))
            waiting(said.prefix(1).lowercased() + said.dropFirst())
            return false
        }
        waiting(nil)
        // A job that ended at once keeps its own line.
        ended.lock.withLock { if !ended.done { store(["date": date, "line": Self.startedLine]) } }
        log(Self.startedLine)
        return true
    }
}
