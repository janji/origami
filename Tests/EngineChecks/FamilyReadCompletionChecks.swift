import Foundation
import OrigamiEngine
import Synchronization

/// A family read scheduler for checks, as `FamilyExperts.reads` is made: room for `jobs` demand jobs; `lanes` the
/// GGUF's and the hot copy's reads at once (`FamilyExperts.checkReadLimits`).
func familyCheckScheduler(jobs: Int, lanes: (main: Int, hot: Int)? = nil) throws -> IOScheduler {
    let saved = FamilyExperts.checkReadLimits
    defer { FamilyExperts.checkReadLimits = saved }
    FamilyExperts.checkReadLimits = lanes
    return try FamilyExperts.makeReadScheduler(ledger: MemoryLedger(), jobs: jobs)
}

extension FamilyExperts {
    /// Reads admitted to the pool's scheduler and not yet ended (queued, running, or cancelled and not yet completed).
    var pendingReads: Int { reads.snapshot().drives.reduce(0) { $0 + $1.outstanding } }
    /// Waits, up to 30 s, until no read is pending.
    func waitForReads() {
        let end = Date(timeIntervalSinceNow: 30)
        while pendingReads > 0 && Date() < end { usleep(200) }
    }
    /// No queued guess starts until the hold is released; demand, a promoted guess included, still runs.
    func holdGuesses() -> IOPrefetchHold { reads.holdPrefetch()! }
}

/// The public join contract includes all worker-written bytes. This is a runtime contract check;
/// the release/acquire ordering itself is established by source review, not a probabilistic race test.
func familyDemandJoinPublication() throws {
    final class Cells: @unchecked Sendable {
        let count = 24 * 64, values: UnsafeMutablePointer<Int>
        init() {
            values = .allocate(capacity: count)
            values.initialize(repeating: -1, count: count)
        }
        deinit {
            values.deinitialize(count: count)
            values.deallocate()
        }
    }
    let plain = try familyCheckScheduler(jobs: 24)
    let split = try familyCheckScheduler(jobs: 24, lanes: (3, 2))
    for spin: Double? in [nil, 0, 0.000001, 0.001] {
        for lanes in [false, true] {
            let cells = Cells()
            let read: @Sendable (Int) -> Void = { job in
                for offset in 0..<64 { cells.values[job * 64 + offset] = job * 1_000 + offset }
            }
            if lanes {
                try FamilyExperts.perform(24, on: split, source: { $0 % 3 == 0 ? .internalSSD : .externalSSD }, spin: spin, read)
            } else {
                try FamilyExperts.perform(24, on: plain, spin: spin, read)
            }
            for job in 0..<24 {
                for offset in 0..<64 where cells.values[job * 64 + offset] != job * 1_000 + offset {
                    throw EngineError("Demand join did not publish every worker's bytes")
                }
            }
        }
    }
}

func familyReadCompletionsOverlap() throws {
    let scheduler = try familyCheckScheduler(jobs: 2)
    // Every demand worker must be submitted before the new callback, even in the blocking path. Hold the
    // workers until that callback observes both starts, then release them. No timing-based speed assertion.
    for spin: Double? in [nil, 0.1] {
        for streaming in [false, true] {
            final class Submitted: Sendable {
                let began = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
                let timedOut = Atomic<Bool>(false), finished = Atomic<Int>(0)
            }
            let state = Submitted()
            var calls = 0
            let callback = {
                calls += 1
                for _ in 0..<2 {
                    if state.began.wait(timeout: .now() + 5) == .timedOut { state.timedOut.store(true, ordering: .releasing) }
                }
                state.release.signal()
                state.release.signal()
            }
            let read: @Sendable (Int) -> Void = { _ in
                state.began.signal()
                if state.release.wait(timeout: .now() + 5) == .timedOut { state.timedOut.store(true, ordering: .releasing) }
                state.finished.add(1, ordering: .releasing)
            }
            if streaming {
                try FamilyExperts.perform(2, on: scheduler, spin: spin, submitted: callback, meanwhile: {}, read)
            } else {
                try FamilyExperts.perform(2, on: scheduler, spin: spin, submitted: callback, read)
            }
            guard calls == 1, !state.timedOut.load(ordering: .acquiring), state.finished.load(ordering: .acquiring) == 2 else {
                throw EngineError("Family demand submission callback ran before submission or failed to join")
            }
        }
    }
    var calls = 0
    try FamilyExperts.perform(0, on: scheduler, spin: nil, meanwhile: { calls += 1 }) { _ in }
    guard calls == 1 else { throw EngineError("Family completion empty read did not call meanwhile once") }
}
