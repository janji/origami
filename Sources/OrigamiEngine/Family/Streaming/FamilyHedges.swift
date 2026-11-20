import Dispatch
import Foundation

/// Each drive's read rate beside `FamilyExperts`' moving average: the current rate (bytes over busy seconds, both
/// decaying with a one-second time constant, so it follows the last second or so, a stalled read's wait included) and
/// the healthy rate (the moving average's running maximum, decaying toward it with a one-minute time constant, so a
/// slow period of many seconds barely lowers it, while a drive that really is slower is learned within minutes). Both
/// constants are times, not drive speeds: nothing here depends on the Mac.
final class FamilyDriveRates: @unchecked Sendable {
    private let lock = NSLock()
    private var current: [IOSource: (bytes: Double, seconds: Double, at: UInt64)] = [:]
    private var healthy: [IOSource: (rate: Double, at: UInt64)] = [:]

    /// `bytes` read from `source` over `seconds` (a batch, a stalled read's wait so far, or a lost read's remainder).
    func add(_ source: IOSource, bytes: Double, seconds: Double) {
        lock.withLock {
            let now = DispatchTime.now().uptimeNanoseconds
            let old = current[source] ?? (0, 0, now)
            let keep = exp(-Double(now - min(now, old.at)) / 1e9)
            current[source] = (old.bytes * keep + bytes, old.seconds * keep + seconds, now)
        }
    }
    /// The moving average of `source` after a measurement.
    func observe(_ source: IOSource, average: Double) {
        lock.withLock {
            let now = DispatchTime.now().uptimeNanoseconds
            let old = healthy[source] ?? (average, now)
            let keep = exp(-Double(now - min(now, old.at)) / 60e9)
            healthy[source] = (max(average, average + (old.rate - average) * keep), now)
        }
    }
    func reset(_ source: IOSource) {
        lock.withLock {
            current[source] = nil
            healthy[source] = nil
        }
    }
    func current(_ source: IOSource) -> Double? {
        lock.withLock { current[source].flatMap { $0.seconds > 0 ? $0.bytes / $0.seconds : nil } }
    }
    func healthy(_ source: IOSource) -> Double? { lock.withLock { healthy[source]?.rate } }
    /// The rate to plan with: the current rate while the drive is stalled (below a third of its healthy rate: outside
    /// the factor of two a healthy drive's short-window rate moves by with queue depth and record size), else `average`;
    /// once the drive has had no reads for a while, settled toward `stored`, its measured rate (`LiveDriveEstimate`).
    func rate(_ source: IOSource, average: Double, stored: Double) -> Double {
        let now = DispatchTime.now().uptimeNanoseconds
        let idle = lock.withLock { current[source].map { Double(now - min(now, $0.at)) / 1e9 } } ?? 0
        guard idle <= LiveDriveEstimate.graceSeconds else { return LiveDriveEstimate.settled(average, stored: stored, idleSeconds: idle) }
        guard let rate = current(source), rate < (healthy(source) ?? average) / 3 else { return average }
        return max(1, rate)
    }
}

/// The last 256 hedge decisions (`FamilyExperts.hedgeTrace`): per demand record both drives hold whose read outlasted
/// its minimum delay, the latest estimates, whether its replica fired and why, the winner, and when each read ended.
final class FamilyHedgeTrace: @unchecked Sendable {
    /// One record's decision (changed under `lock`); times in uptime nanoseconds.
    final class Mark: @unchecked Sendable {
        let key: Int, primary: IOSource, start: UInt64
        var elapsed = 0.0, primaryLeft = 0.0, replica = 0.0
        var decision: String?
        var fired: UInt64?, primaryEnd: UInt64?, replicaEnd: UInt64?
        var winner: String?
        init(key: Int, primary: IOSource, start: UInt64) {
            self.key = key
            self.primary = primary
            self.start = start
        }
    }
    let lock = NSLock()
    private var ring: [Mark] = []
    private var next = 0
    /// Adds `mark` (under `lock`).
    func appendLocked(_ mark: Mark) {
        if ring.count < 256 { ring.append(mark) } else { ring[next] = mark }
        next = (next + 1) % 256
    }
    var count: Int { lock.withLock { ring.count } }
    /// Oldest first; seconds from the batch's start.
    var snapshot: [[String: Any]] {
        lock.withLock {
            let ordered = ring.count < 256 ? ring : Array(ring[next...] + ring[..<next])
            return ordered.map { m in
                func seconds(_ at: UInt64?) -> Any { at.map { Double($0 - min($0, m.start)) / 1e9 } ?? NSNull() }
                return [
                    "key": m.key, "primary": m.primary == .internalSSD ? "copy" : "original", "elapsed_seconds": m.elapsed,
                    "expected_primary_left_seconds": m.primaryLeft, "expected_replica_seconds": m.replica,
                    "decision": m.decision ?? "", "fired_seconds": seconds(m.fired), "winner": m.winner ?? NSNull(),
                    "primary_end_seconds": seconds(m.primaryEnd), "replica_end_seconds": seconds(m.replicaEnd),
                ]
            }
        }
    }
}

/// One `fetch` batch's hedged reads: per missing record both drives hold, a race (`ExpertReadReplica`) between its
/// primary (its job, or the GGUF's three part jobs, which `FamilyExperts.perform` submits) and a replica of the same
/// record from the other drive into an arena lease. Per job: its record; per hedged record: its race and a group its
/// primary's jobs leave as they end (a primary that lost while still reading becomes a guess that waits on it).
/// A replica fires once its primary failed, or once the race's minimum delay passed and `decide` says it would end first.
final class FamilyHedges: @unchecked Sendable {
    /// At `elapsed` seconds into the batch: whether hedged record `n`'s replica would end before its primary, both
    /// estimates (seconds from now), and when not, the elapsed seconds at which to ask again.
    typealias Decide = (_ n: Int, _ elapsed: Double) -> (fire: Bool, primaryLeft: Double, replica: Double, again: Double)

    let wake: DispatchSemaphore
    let record: [Int]
    let races: [ExpertReadReplica?]
    let done: [DispatchGroup?]
    private let marks: [FamilyHedgeTrace.Mark?]
    private let trace: FamilyHedgeTrace
    private let rates: FamilyDriveRates
    private let recordBytes: Int
    private let decide: Decide
    private let start: UInt64
    /// Per record, when `decide` is asked again (polls run on the waiting thread only).
    private var again: [UInt64]
    private let lock = NSLock()
    /// Jobs of records not hedged that have not ended, and why each hedged record's read failed (set by its job before
    /// the job's completion runs).
    private var open: Int
    private var failures: [String?]

    /// `record`: each job's record; `races`: each record's race (nil: not hedged), already given its primary (`expect`),
    /// and `marks` its trace mark; `start`: the batch's start. A stalled or lost primary's time goes to `rates`.
    init(
        wake: DispatchSemaphore, record: [Int], races: [ExpertReadReplica?], marks: [FamilyHedgeTrace.Mark?],
        trace: FamilyHedgeTrace, rates: FamilyDriveRates, recordBytes: Int, start: UInt64, decide: @escaping Decide
    ) {
        self.wake = wake
        self.record = record
        self.races = races
        self.marks = marks
        self.trace = trace
        self.rates = rates
        self.recordBytes = recordBytes
        self.start = start
        self.decide = decide
        again = Array(repeating: 0, count: races.count)
        failures = Array(repeating: nil, count: races.count)
        done = races.map { $0 == nil ? nil : DispatchGroup() }
        open = record.filter { races[$0] == nil }.count
        for miss in record { done[miss]?.enter() }
    }

    /// A job of hedged record `miss` failed to read it.
    func fail(_ miss: Int, _ error: Error) {
        lock.lock()
        if failures[miss] == nil { failures[miss] = String(describing: error) }
        lock.unlock()
    }

    /// Fires record `n`'s replica when due: its minimum delay passed, and its primary failed or `decide` says so.
    private func consider(_ n: Int, _ race: ExpertReadReplica, now: UInt64) {
        guard let due = race.hedgeDeadline, due <= now, again[n] <= now, let mark = marks[n] else { return }
        let elapsed = Double(now - min(now, start)) / 1e9
        let failed = race.primaryFinished
        let estimate = failed ? (fire: true, primaryLeft: 0, replica: 0, again: 0) : decide(n, elapsed)
        if estimate.fire {
            race.tryReplica()
        } else {
            again[n] = start + UInt64(max(elapsed + 0.001, estimate.again) * 1e9)
        }
        let fired = race.isHedged
        trace.lock.withLock {
            if mark.decision == nil { trace.appendLocked(mark) }
            guard mark.fired == nil else { return }
            (mark.elapsed, mark.primaryLeft, mark.replica) = (elapsed, estimate.primaryLeft, estimate.replica)
            mark.decision = failed ? "fired: primary failed" : estimate.fire ? "fired: replica sooner" : "held: primary sooner"
            if fired { mark.fired = now }
        }
        // A primary still reading when its hedge fires: its wait so far counts toward its drive's current rate.
        if fired && !failed { rates.add(mark.primary, bytes: Double(recordBytes), seconds: elapsed) }
    }

    var hedging: FamilyExperts.Hedging {
        .init(
            wake: wake, submitted: { [self] job, ticket in races[record[job]]?.addPrimary(ticket) },
            ended: { [self] job, outcome in
                let miss = record[job]
                lock.lock()
                let failure = failures[miss]
                if races[miss] == nil { open -= 1 }
                lock.unlock()
                guard let race = races[miss] else { return }
                if let failure, case .succeeded = outcome { race.finishPart(.failed(failure)) } else { race.finishPart(outcome) }
                if race.primaryFinished, let mark = marks[miss] {
                    let lost = race.winningReplica != nil && !race.primaryDropped
                    let now = DispatchTime.now().uptimeNanoseconds
                    let fired: UInt64? = trace.lock.withLock {
                        guard mark.primaryEnd == nil else { return nil }
                        mark.primaryEnd = now
                        return mark.fired
                    }
                    // A primary that lost while reading: the rest of its wait counts too, once it ends.
                    if lost, let fired { rates.add(mark.primary, bytes: 0, seconds: Double(now - min(now, fired)) / 1e9) }
                }
                done[miss]!.leave()
            },
            poll: { [self] in
                lock.lock()
                var settled = open == 0
                lock.unlock()
                var next: UInt64?
                let now = DispatchTime.now().uptimeNanoseconds
                for (n, race) in races.enumerated() {
                    guard let race else { continue }
                    consider(n, race, now: now)
                    guard race.wait(timeout: 0) == nil else { continue }
                    settled = false
                    // A due hedge that found no free staging record waits for the next wake (a replica ending).
                    if let due = race.hedgeDeadline.map({ max($0, again[n]) }), due > now { next = min(next ?? due, due) }
                }
                return (settled, next)
            }, left: { [self] job in races[record[job]]?.winningReplica != nil })
    }

    /// Records each traced race's winner (after the batch's wait).
    func traceWinners() {
        for (race, mark) in zip(races, marks) {
            guard let race, let mark else { continue }
            let winner = race.winningReplica != nil ? "replica" : race.primarySucceeded ? "primary" : "none"
            trace.lock.withLock { if mark.decision != nil { mark.winner = winner } }
        }
    }

    /// Why the batch fails: a hedged record whose primary and replica both failed (or whose failed primary found no
    /// staging record), or one cancelled.
    var failure: Error? {
        for race in races.compactMap({ $0 }) {
            switch race.wait(timeout: 0) {
            case .failed(let reason)?: return EngineError(race.isHedged ? "Both expert read sources failed: \(reason)" : reason)
            case .cancelled?: return EngineError("An expert read was cancelled before it ran")
            default: continue
            }
        }
        return nil
    }

    /// Waits for every read of every race (the error path, before the slots are freed).
    func joinAll() { for race in races { race?.joinAll() } }
}
