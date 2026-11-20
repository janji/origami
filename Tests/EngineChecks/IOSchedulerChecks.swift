import Dispatch
import Foundation
import OrigamiEngine

private func ioRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError(message) }
}

private func ioMustFail(_ operation: () throws -> Void) throws {
    do { try operation() } catch is EngineError { return }
    throw EngineError("Expected bounded/stale I/O request to fail")
}

private func ioWait(_ semaphore: DispatchSemaphore, _ message: String) throws {
    if semaphore.wait(timeout: .now() + 5) != .success { throw EngineError(message) }
}

private final class IOEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []
    func add(_ event: String) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }
    func values() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

func schedulerDemandHasReservedExecutionAndAdmission() throws {
    let limits = IOScheduler.Limits(
        maximumConcurrent: 2, reservedDemandConcurrent: 1,
        maximumOutstanding: 5, reservedDemandOutstanding: 2,
        maximumOutstandingBytes: 50, reservedDemandBytes: 20)
    let ledger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let generation = scheduler.currentGeneration
    let prefetchStarted = DispatchSemaphore(value: 0)
    let releasePrefetch = DispatchSemaphore(value: 0)
    let demandStarted = DispatchSemaphore(value: 0)
    let releaseDemand = DispatchSemaphore(value: 0)
    let secondDemandStarted = DispatchSemaphore(value: 0)
    let events = IOEvents()
    defer {
        releasePrefetch.signal()
        releaseDemand.signal()
        scheduler.shutdown()
    }

    let first = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 10) { _ in
        prefetchStarted.signal()
        try ioWait(releasePrefetch, "Prefetch test gate timed out")
        events.add("prefetch1")
    }
    try ioWait(prefetchStarted, "First prefetch never started")
    let second = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 10) { _ in
        events.add("prefetch2")
    }
    let third = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 10) { _ in
        events.add("prefetch3")
    }
    try ioMustFail {
        _ = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 1) { _ in }
    }
    let demand = try scheduler.submit(source: .externalSSD, priority: .demand, generation: generation, byteCount: 10) { _ in
        demandStarted.signal()
        try ioWait(releaseDemand, "Demand test gate timed out")
        events.add("demand1")
    }
    try ioWait(demandStarted, "Prefetch occupied the reserved demand worker")
    let nextDemand = try scheduler.submit(source: .externalSSD, priority: .demand, generation: generation, byteCount: 10) { _ in
        events.add("demand2")
        secondDemandStarted.signal()
    }
    try ioMustFail {
        _ = try scheduler.submit(source: .externalSSD, priority: .demand, generation: generation, byteCount: 1) { _ in }
    }
    let before = scheduler.snapshot().drives.first { $0.source == .externalSSD }!
    try ioRequire(
        before.runningPrefetch == 1 && before.runningDemand == 1 && before.outstanding == 5,
        "Reserved execution/admission accounting is wrong")
    releaseDemand.signal()
    try ioWait(secondDemandStarted, "Queued demand did not run ahead of prefetch")
    try ioRequire(
        !events.values().contains("prefetch2") && !events.values().contains("prefetch3"),
        "Background work consumed reserved demand capacity")
    releasePrefetch.signal()
    for ticket in [first, second, third, demand, nextDemand] {
        try ioRequire(ticket.join() == .succeeded(bytes: 10), "Admitted operation did not complete")
    }
    let after = scheduler.snapshot().drives.first { $0.source == .externalSSD }!
    try ioRequire(
        after.outstanding == 0 && after.outstandingBytes == 0 && after.successfulBytes == 50,
        "Joined operations retained queue capacity")
}

func schedulerCancellationJoinsWritersAndInvalidatesGenerations() throws {
    let limits = IOScheduler.Limits(
        maximumConcurrent: 1, reservedDemandConcurrent: 1,
        maximumOutstanding: 4, reservedDemandOutstanding: 2,
        maximumOutstandingBytes: 100, reservedDemandBytes: 50)
    let ledger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let old = scheduler.currentGeneration
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let events = IOEvents()
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let writer = try scheduler.submit(
        source: .internalSSD, priority: .demand, generation: old, byteCount: 1,
        operation: { cancellation in
            started.signal()
            try ioWait(release, "Cancelled writer test gate timed out")
            events.add("writer-stopped")
            try ioRequire(cancellation.isCancelled, "Running writer did not observe cancellation")
        }, completion: { _, result in events.add(result == .cancelled ? "cancelled-callback" : "wrong-callback") })
    try ioWait(started, "Writer never started")
    let queued = try scheduler.submit(source: .internalSSD, priority: .demand, generation: old, byteCount: 1) { _ in
        events.add("forbidden-queued-write")
    }
    let current = try scheduler.beginGeneration()
    try ioRequire(current.value > old.value, "Generation ID did not advance")
    try ioRequire(queued.join() == .cancelled, "Queued cancellation did not retire")
    try ioRequire(writer.join(timeout: 0.02) == nil, "Cancellation returned before the writer stopped")
    try ioMustFail {
        _ = try scheduler.submit(source: .internalSSD, priority: .demand, generation: old, byteCount: 1) { _ in }
    }
    let replacement = try scheduler.submit(source: .internalSSD, priority: .demand, generation: current, byteCount: 1) { _ in
        events.add("replacement")
    }
    try ioRequire(replacement.join(timeout: 0.02) == nil, "A cancelled active pread released its worker early")
    release.signal()
    try ioRequire(writer.join() == .cancelled, "Late successful read overrode cancellation")
    try ioRequire(replacement.join() == .succeeded(bytes: 1), "New generation was damaged by old completion")
    let values = events.values()
    try ioRequire(!values.contains("forbidden-queued-write"), "Cancelled queued operation ran")
    guard let stopped = values.firstIndex(of: "writer-stopped"), let callback = values.firstIndex(of: "cancelled-callback") else {
        throw EngineError("Writer or cancellation callback did not retire")
    }
    try ioRequire(
        stopped < callback,
        "Completion freed a live writer's destination")
    try ioRequire(try scheduler.cancelAndJoin(current).isEmpty, "Completed tickets leaked in scheduler history")
    try ioMustFail {
        _ = try scheduler.submit(source: .internalSSD, priority: .demand, generation: current, byteCount: 1) { _ in }
    }
}

func schedulerDriveIsolationFailureAndMetadataAccounting() throws {
    let limits = IOScheduler.Limits(
        maximumConcurrent: 1, reservedDemandConcurrent: 1,
        maximumOutstanding: 2, reservedDemandOutstanding: 1,
        maximumOutstandingBytes: 100, reservedDemandBytes: 50)
    let ledger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
    var empty: IOScheduler? = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    try ioRequire(
        ledger.snapshot().categories["io-scheduler-metadata"] == 8_192,
        "Scheduler metadata has no byte reservation")
    empty?.shutdown()
    empty = nil
    try ioRequire(ledger.snapshot().reserved == 0, "Idle scheduler leaked metadata reservation")
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let other = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let generation = scheduler.currentGeneration
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let events = IOEvents()
    defer {
        release.signal()
        scheduler.shutdown()
        other.shutdown()
    }
    let internalRead = try scheduler.submit(source: .internalSSD, priority: .demand, generation: generation, byteCount: 1) { _ in
        started.signal()
        try ioWait(release, "Drive isolation gate timed out")
    }
    try ioWait(started, "Internal queue did not start")
    let failed = try scheduler.submit(
        source: .externalSSD, priority: .demand, generation: generation, byteCount: 1,
        operation: { _ in throw EngineError("simulated drive loss") },
        completion: { _, _ in events.add("failed-callback-finished") })
    try ioRequire(failed.join() == .failed("simulated drive loss"), "I/O error was hidden")
    try ioRequire(events.values() == ["failed-callback-finished"], "Join preceded its completion callback")
    try ioRequire(internalRead.join(timeout: 0) == nil, "Other drive's failure altered the blocked queue")
    try ioRequire(!other.cancel(internalRead), "Foreign scheduler cancelled a colliding ticket ID")
    try ioRequire(!scheduler.cancel(failed), "Completed ticket accepted cancellation")
    try ioMustFail {
        _ = try scheduler.submit(
            source: .externalSSD, priority: .demand,
            generation: other.currentGeneration, byteCount: 1
        ) { _ in }
    }
    try ioMustFail {
        _ = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 1) { _ in }
    }
    release.signal()
    try ioRequire(internalRead.join() == .succeeded(bytes: 1), "Independent drive operation failed")
}

func schedulerPromotionAndByteLimits() throws {
    let limits = IOScheduler.Limits(
        maximumConcurrent: 2, reservedDemandConcurrent: 1,
        maximumOutstanding: 8, reservedDemandOutstanding: 2,
        maximumOutstandingBytes: 100, reservedDemandBytes: 40)
    let ledger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let generation = scheduler.currentGeneration
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let promotedStarted = DispatchSemaphore(value: 0)
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let first = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 30) { _ in
        started.signal()
        try ioWait(release, "Promotion gate timed out")
    }
    try ioWait(started, "Prefetch never started")
    let second = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 30) { _ in
        promotedStarted.signal()
    }
    try ioMustFail {
        _ = try scheduler.submit(source: .externalSSD, priority: .prefetch, generation: generation, byteCount: 1) { _ in }
    }
    try ioRequire(scheduler.promote(second), "Queued prefetch could not become demand")
    try ioWait(promotedStarted, "Promoted prefetch did not use demand reserve")
    try ioRequire(second.join() == .succeeded(bytes: 30), "Promoted operation failed")
    try ioRequire(!scheduler.promote(first), "Executing prefetch was incorrectly moved between queues")
    try ioMustFail {
        _ = try scheduler.submit(source: .externalSSD, priority: .demand, generation: generation, byteCount: 71) { _ in }
    }
    release.signal()
    try ioRequire(first.join() == .succeeded(bytes: 30), "Original prefetch was lost during promotion")
}
