import Foundation
import OrigamiEngine

/// One simulated drive: up to four reads run at once and share its true rate equally; the rest wait in order.
private struct SimulatedDrive {
    var rate: Double
    var running: [(bytes: Double, left: Double, started: Double, depth: Int)] = []
    var queued: [Double] = []
    var outstanding: Int { running.count + queued.count }
    var outstandingBytes: UInt64 { UInt64((running.map(\.left) + queued).reduce(0, +)) }
    mutating func admit(now: Double) {
        while running.count < 4, !queued.isEmpty {
            let bytes = queued.removeFirst()
            running.append((bytes, bytes, now, running.count + 1))
        }
    }
    var nextEnd: Double { running.isEmpty ? .infinity : running.map(\.left).min()! * Double(running.count) / rate }
    /// Advances `seconds` (at most `nextEnd`): the finished reads as (bytes, service seconds, depth at start).
    mutating func advance(_ seconds: Double, now: Double) -> [(Double, Double, Int)] {
        guard !running.isEmpty else { return [] }
        let each = seconds * rate / Double(running.count)
        for i in running.indices { running[i].left -= each }
        let done = running.filter { $0.left <= 1e-3 }.map { ($0.bytes, now - $0.started, $0.depth) }
        running.removeAll { $0.left <= 1e-3 }
        admit(now: now)
        return done
    }
}

private final class SimulatedClock: @unchecked Sendable { var now = 0.0 }

/// Qwen's source rule (`ExpertSourceOnlineEstimator` + `IOScheduler.earliestSource`, the release bounds) against two
/// simulated drives holding the same records, `inFlight` reads kept outstanding: the model's drive runs at its measured
/// rate, then 20x slower, then at its measured rate again. Returns the model drive's share of reads per phase window.
private func slowdownShares(inFlight: Int, perPhase: Int, window: Int) throws -> [[Double]] {
    let clock = SimulatedClock()
    let record = 2_764_800.0
    let calibration = ExpertSourceCalibration(
        model: DriveSpeed(
            queueDepths: [1, 4, 8], bytesPerSecond: [2.38e9, 3.06e9, 3.07e9], recordBytes: 2_764_800, measuredAt: 0, name: "model"),
        pack: DriveSpeed(
            queueDepths: [1, 4, 8], bytesPerSecond: [2.72e9, 3.44e9, 3.44e9], recordBytes: 2_764_800, measuredAt: 0, name: "copy"),
        recordBytes: 2_764_800)
    let estimator = try ExpertSourceOnlineEstimator(
        calibration: calibration,
        settings: ExpertSourceOnlineCalibration(
            floorMultiple: 0.25, ceilingMultiple: 4, smoothing: 0.02, warmupObservations: 32, delayCeilingMultiple: 1),
        maximumConcurrent: 4, clock: { [clock] in clock.now })
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let limits = IOScheduler.Limits(maximumOutstanding: 64, reservedDemandOutstanding: 16, reservedDemandBytes: 64 * 1_048_576)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    // `.internalSSD` is the model's drive in Qwen's naming, `.externalSSD` the pack (the hot copy).
    var drives: [IOSource: SimulatedDrive] = [.internalSSD: .init(rate: 3.06e9), .externalSSD: .init(rate: 3.44e9)]
    var shares: [[Double]] = [[], [], []]
    var placed: [IOSource] = []
    for phase in 0..<3 {
        drives[.internalSSD]!.rate = phase == 1 ? 3.06e9 / 20 : 3.06e9
        for _ in 0..<perPhase {
            while drives.values.map(\.outstanding).reduce(0, +) >= inFlight {
                let step = drives.values.map(\.nextEnd).min()!
                clock.now += step
                for source in IOSource.allCases {
                    for (bytes, seconds, depth) in drives[source]!.advance(step, now: clock.now) {
                        estimator.observe(source: source, bytes: Int(bytes), seconds: seconds, concurrency: depth)
                    }
                }
            }
            let placement = estimator.placementCalibration(hotInternal: false)
            let source =
                try scheduler.earliestSource(
                    availableSources: [.externalSSD, .internalSSD], estimate: placement.estimatedSeconds,
                    idleProbeSource: estimator.idleProbeSource(hotInternal: false), priority: .demand, byteCount: Int(record),
                    pendingBytes: drives.mapValues(\.outstandingBytes), pendingJobs: drives.mapValues(\.outstanding)) ?? .externalSSD
            drives[source]!.queued.append(record)
            drives[source]!.admit(now: clock.now)
            placed.append(source)
            if placed.count == window {
                shares[phase].append(Double(placed.filter { $0 == .internalSSD }.count) / Double(window))
                placed = []
            }
        }
    }
    return shares
}

/// A drive that falls to a twentieth of its speed mid-run gets fewer reads within a bounded number of reads, and its
/// share comes back once it is fast again (every drive holds the same bytes: placement cannot change output). Lone reads
/// go to the faster copy throughout; with 4 or 8 in flight the model drive's share falls from about half to at most
/// twice its fair 4% within 100 reads, and is back above 40% within 700 reads of the drive's recovery (about 2,000
/// before an avoided drive's rare reads counted for the time they stand for, `LiveDriveEstimate.weight`).
func driveSlowdownShiftsReads() throws {
    for inFlight in [1, 4, 8] {
        let shares = try slowdownShares(inFlight: inFlight, perPhase: 3_000, window: 100)
        let shown = shares.map { $0.map { String(format: "%.2f", $0) }.joined(separator: " ") }.joined(separator: " | ")
        func require(_ condition: Bool, _ what: String) throws {
            if !condition { throw EngineError("\(what) with \(inFlight) reads in flight: \(shown)") }
        }
        if inFlight == 1 {
            try require(shares.joined().allSatisfy { $0 <= 0.05 }, "Lone reads left the faster drive")
            continue
        }
        try require(shares[0].allSatisfy { $0 >= 0.4 }, "Two healthy drives did not share the reads")
        try require(shares[1].allSatisfy { $0 <= 0.09 }, "The slowed drive kept its share")
        try require(shares[2].dropFirst(7).allSatisfy { $0 >= 0.4 }, "The recovered drive did not get its share back")
    }
    // An idle drive's estimate settles back to its measurement; a rare read counts for the time since the previous one.
    let settled = [0, 0.25, 1.25, 20].map { LiveDriveEstimate.settled(20, stored: 1, idleSeconds: $0) }
    guard settled[0] == 20, settled[1] == 20, abs(settled[2] - (1 + 19 / M_E)) < 1e-9, settled[3] < 1.001,
        LiveDriveEstimate.weight(0.02, gapSeconds: 0) == 0.02,
        abs(LiveDriveEstimate.weight(0.02, gapSeconds: 0.5) - (1 - 1 / M_E)) < 1e-9
    else { throw EngineError("Live drive estimate settling: \(settled)") }
}
