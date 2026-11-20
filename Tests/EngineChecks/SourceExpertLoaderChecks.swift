import Dispatch
import Foundation
import Metal
import OrigamiEngine

private final class LoaderReads: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [(Int, IOSource)] = []
    @discardableResult func record(_ key: ExpertKey, _ source: IOSource) -> Int {
        lock.lock()
        defer { lock.unlock() }
        values.append((key.expert, source))
        return values.count
    }
    func sources() -> [IOSource] {
        lock.lock()
        defer { lock.unlock() }
        return values.map(\.1)
    }
    func count() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return values.count
    }
}

private func loaderRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError(message) }
}
private func loaderWait(_ gate: DispatchSemaphore) throws {
    guard gate.wait(timeout: .now() + 5) == .success else { throw EngineError("Expert loader gate timed out") }
}
private func loaderFixture(slots: Int = 2) throws -> (ExpertCache, IOScheduler) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(
        device: device, ledger: ledger,
        byteLimit: UInt64(PackStore.expertBytes * slots), slotsPerSlab: slots)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 2, reservedDemandConcurrent: 1,
        maximumOutstanding: 8, reservedDemandOutstanding: 2,
        reservedDemandBytes: 10 * UInt64(PackStore.expertBytes))
    return (cache, try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits))
}

func expertLoaderCoalescesAndRetainsConsumerPins() throws {
    let (cache, scheduler) = try loaderFixture()
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let reads = LoaderReads()
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { key, source, destination in
        reads.record(key, source)
        started.signal()
        try loaderWait(release)
        destination.initializeMemory(as: UInt8.self, repeating: 0x2b)
    }
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let generation = scheduler.currentGeneration
    let key = try ExpertKey(layer: 0, expert: 0)
    let first = try loader.request(key, generation: generation, priority: .prefetch)
    try loaderWait(started)
    let duplicate = try loader.request(key, generation: generation, priority: .demand)
    try loaderRequire(reads.count() == 1, "Coalesced expert used two producers")
    release.signal()
    let a = try loader.joinAndPin(first)
    let b = try loader.joinAndPin(duplicate)
    try loaderRequire(
        a.ticket == b.ticket && a.buffer.bytes[0] == 0x2b && b.buffer.bytes[PackStore.expertBytes - 1] == 0x2b,
        "Coalesced reads published wrong bytes/generation")
    let hit = try loader.loadAndPin(key, generation: generation)
    try loaderRequire(hit.ticket == a.ticket && reads.count() == 1, "Resident cache hit performed I/O")
    try a.retireAfterGPUCompletion()
    try b.retireAfterGPUCompletion()
    try hit.retireAfterGPUCompletion()
    do {
        try a.retireAfterGPUCompletion()
        throw EngineError("Double consumer retirement was accepted")
    } catch let error as EngineError {
        try loaderRequire(error.description.contains("already retired"), "Wrong double-retirement error")
    }
    // The still-retained request owns a host handoff pin independently of all
    // consumer pins; it cannot be evicted between completion and a later join.
    try withExtendedLifetime((first, duplicate)) {
        try loaderRequire(cache.snapshot().pinnedSlots == 1, "Completed request lost its handoff pin")
    }
}

func expertLoaderCancellationWaitsForProducerAndRejectsLateData() throws {
    let (cache, scheduler) = try loaderFixture(slots: 1)
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let reads = LoaderReads()
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { key, source, destination in
        let call = reads.record(key, source)
        if call == 1 {
            started.signal()
            try loaderWait(release)
        }
        destination.initializeMemory(as: UInt8.self, repeating: call == 1 ? 0xba : 0x37)
    }
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let key = try ExpertKey(layer: 1, expert: 7)
    let old = scheduler.currentGeneration
    let request = try loader.request(key, generation: old)
    try loaderWait(started)
    try loaderRequire(loader.cancel(request), "Active expert read refused cancellation")
    try loaderRequire(cache.snapshot().inFlightReads == 1, "Cancelled producer's slot was released before join")
    let next = try scheduler.beginGeneration()
    do {
        _ = try loader.request(key, generation: next)
        throw EngineError("Unjoined writer's destination was reused")
    } catch let error as EngineError {
        try loaderRequire(error.description.contains("must join"), "Wrong unjoined-writer rejection")
    }
    release.signal()
    do {
        _ = try loader.joinAndPin(request)
        throw EngineError("Cancelled expert bytes were consumed")
    } catch is IOOperationCancelled {}
    try loaderRequire(cache.snapshot().inFlightReads == 0, "Joined cancelled read remained in flight")
    let replacement = try loader.loadAndPin(key, generation: next)
    try loaderRequire(
        replacement.buffer.bytes[0] == 0x37 && replacement.buffer.bytes[PackStore.expertBytes - 1] == 0x37,
        "Old completion overwrote the new expert generation")
    try replacement.retireAfterGPUCompletion()
    try loader.cancelAndJoin(next)
    do {
        _ = try loader.loadAndPin(key, generation: next)
        throw EngineError("Cancelled request generation accepted a cache hit")
    } catch let error as EngineError {
        try loaderRequire(error.description.contains("Stale"), "Wrong stale-generation error")
    }
}

func expertLoaderFallbackRunsOnOriginalQueueAndNeverSubstitutes() throws {
    let (cache, scheduler) = try loaderFixture()
    let reads = LoaderReads()
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { key, source, destination in
        reads.record(key, source)
        if source == .externalSSD {
            destination[0] = 0xff
            throw EngineError("simulated external loss after partial read")
        }
        if key.expert == 1 { throw EngineError("simulated original loss") }
        destination.initializeMemory(as: UInt8.self, repeating: 0x5c)
    }
    defer { scheduler.shutdown() }
    let generation = scheduler.currentGeneration
    let loaded = try loader.loadAndPin(ExpertKey(layer: 0, expert: 0), generation: generation)
    try loaderRequire(reads.sources() == [.externalSSD, .internalSSD], "Fallback bypassed the original-drive queue")
    try loaderRequire(
        loaded.buffer.bytes[0] == 0x5c && loaded.buffer.bytes[PackStore.expertBytes - 1] == 0x5c,
        "Partial packed data leaked into original fallback")
    let stats = scheduler.snapshot().drives
    try loaderRequire(
        stats.first { $0.source == .externalSSD }!.failed == 1
            && stats.first { $0.source == .internalSSD }!.succeeded == 1,
        "Fallback traffic was attributed to the wrong drive")
    try loaded.retireAfterGPUCompletion()
    do {
        _ = try loader.loadAndPin(ExpertKey(layer: 0, expert: 1), generation: generation)
        throw EngineError("Missing original expert was substituted")
    } catch let error as EngineError {
        try loaderRequire(error.description.contains("Original expert read failed"), "Original read failure was hidden")
    }
}

/// ROADMAP R2: a demand read whose drive stopped answering. The first read of `key` hangs until `release`, then writes
/// bytes no request may see; every other read writes the key's own pattern (what an unstalled run reads).
private func stalledLoaderFixture(_ key: ExpertKey, release: DispatchSemaphore) throws -> (ExpertCache, IOScheduler, SourceExpertLoader) {
    let (cache, scheduler) = try loaderFixture(slots: 1)
    let reads = LoaderReads()
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { read, source, destination in
        if read == key && reads.record(read, source) == 1 {
            _ = release.wait(timeout: .now() + 10)
            destination.initializeMemory(as: UInt8.self, repeating: 0xee)
        } else {
            for i in destination.indices { destination[i] = stalledLoaderPattern(read, i) }
        }
    }
    return (cache, scheduler, loader)
}
private func stalledLoaderPattern(_ key: ExpertKey, _ i: Int) -> UInt8 {
    UInt8(truncatingIfNeeded: i &* 31 &+ key.expert &+ 7 &* key.layer)
}
private func stalledLoaderCode(_ body: () throws -> Void) -> String? {
    do { try body() } catch { return (error as? EngineError)?.code ?? "\(error)" }
    return nil
}
/// The next request, the read no longer stalled, loads the record bit for bit as an unstalled run does.
private func stalledLoaderRetry(_ cache: ExpertCache, _ scheduler: IOScheduler, _ loader: SourceExpertLoader, _ key: ExpertKey) throws {
    let until = Date().addingTimeInterval(5)
    while cache.snapshot().inFlightReads > 0 && Date() < until { usleep(1_000) }
    try loaderRequire(cache.snapshot().inFlightReads == 0, "The hung read's slot was not freed once it returned")
    let lease = try loader.loadAndPin(key, generation: try scheduler.beginGeneration())
    try loaderRequire(
        (0..<PackStore.expertBytes).allSatisfy { lease.buffer.bytes[$0] == stalledLoaderPattern(key, $0) },
        "The retried record differs from an unstalled read (the hung read's late bytes were published)")
    try lease.retireAfterGPUCompletion()
}

/// A Qwen demand read that does not return fails the request with `drive_stalled` within its bound; the next request
/// reads the record again, exactly.
func qwenDemandReadStallBounded() throws {
    let saved = ExpertReadStall.floor
    ExpertReadStall.floor = 0.3
    defer { ExpertReadStall.floor = saved }
    let key = try ExpertKey(layer: 2, expert: 5)
    let release = DispatchSemaphore(value: 0)
    let (cache, scheduler, loader) = try stalledLoaderFixture(key, release: release)
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let generation = scheduler.currentGeneration
    let request = try loader.request(key, generation: generation)
    let started = Date()
    try loaderRequire(stalledLoaderCode { _ = try loader.joinAndPin(request) } == "drive_stalled", "A hung demand read did not fail")
    let seconds = Date().timeIntervalSince(started)
    try loaderRequire(seconds >= 0.25 && seconds < 3, "The hung demand read's bound took \(seconds) s")
    try loaderRequire(cache.snapshot().inFlightReads == 1, "The hung read's slot was released while it ran")
    release.signal()
    try loader.cancelAndJoin(generation)
    try stalledLoaderRetry(cache, scheduler, loader, key)

    // A read slow before anyone waits (queued behind a throttled drive) is not a stall: the bound counts from the wait.
    let (slowCache, slowScheduler) = try loaderFixture(slots: 1)
    let slow = DispatchSemaphore(value: 0)
    let slowLoader = SourceExpertLoader(cache: slowCache, scheduler: slowScheduler) { read, _, destination in
        _ = slow.wait(timeout: .now() + 10)
        for i in destination.indices { destination[i] = stalledLoaderPattern(read, i) }
    }
    defer {
        slow.signal()
        slowScheduler.shutdown()
    }
    let held = try slowLoader.request(key, generation: slowScheduler.currentGeneration)
    usleep(600_000)  // twice the bound before the waiter arrives
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { slow.signal() }
    let lease = try slowLoader.joinAndPin(held)
    try loaderRequire(
        (0..<PackStore.expertBytes).allSatisfy { lease.buffer.bytes[$0] == stalledLoaderPattern(key, $0) },
        "A read slow before its waiter arrived published other bytes")
    try lease.retireAfterGPUCompletion()
}

/// The request's teardown (discarding its reads, ending its generation) gives up on a hung read at its bound and leaves
/// its slot reading: no request gets that slot or key until the read returns, whose late bytes are then not published.
func qwenReadTeardownStallBounded() throws {
    let saved = ExpertReadStall.floor
    ExpertReadStall.floor = 0.3
    defer { ExpertReadStall.floor = saved }
    let key = try ExpertKey(layer: 2, expert: 5)
    let release = DispatchSemaphore(value: 0)
    let (cache, scheduler, loader) = try stalledLoaderFixture(key, release: release)
    defer {
        release.signal()
        scheduler.shutdown()
    }
    let generation = scheduler.currentGeneration
    let request = try loader.request(key, generation: generation)
    try loaderRequire(stalledLoaderCode { _ = try loader.joinAndPin(request) } == "drive_stalled", "A hung demand read did not fail")
    let started = Date()
    loader.joinDiscarding(request)
    try loader.cancelAndJoin(generation)
    let seconds = Date().timeIntervalSince(started)
    // Each join counts its own bound from its start: discarding, then ending the generation.
    try loaderRequire(seconds >= 0.5 && seconds < 2, "The teardown took \(seconds) s for a hung read (two bounds)")
    try loaderRequire(cache.snapshot().inFlightReads == 1, "The hung read's slot was released while it ran")
    let next = try scheduler.beginGeneration()
    try loaderRequire(
        stalledLoaderCode { _ = try loader.request(key, generation: next) } != nil, "A new generation adopted the hung read's slot")
    try loaderRequire(
        stalledLoaderCode { _ = try loader.request(ExpertKey(layer: 0, expert: 1), generation: next) } != nil,
        "Another key took the hung read's slot")
    release.signal()
    try stalledLoaderRetry(cache, scheduler, loader, key)
}
