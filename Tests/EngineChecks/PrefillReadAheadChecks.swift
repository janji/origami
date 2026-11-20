import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func aheadRequire(_ value: @autoclosure () -> Bool, _ message: String) throws {
    guard value() else { throw EngineError(message) }
}
private func aheadDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}
private func aheadConfig(slots: Int = 528) -> ExpertExecutionConfiguration {
    ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: slots, slotsPerSlab: 8,
        demandWindow: 16, readConcurrency: 4, prefillCacheSlots: slots, prefillReadAhead: true)
}
private func aheadSentinel(_ key: ExpertKey) -> UInt8 { UInt8((key.layer * 31 + key.expert) % 251) }
private func aheadRetired(_ ledger: MemoryLedger) throws {
    for _ in 0..<1000 {
        if ledger.snapshot().reserved == 0 { return }
        Thread.sleep(forTimeInterval: 0.001)
    }
    throw EngineError("Read-ahead owner remained charged: \(ledger.snapshot().categories)")
}

/// Counts concurrent fixture readers and holds the first read of expert 0 until released.
private final class AheadRefillReaders: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 0, maximum = 0, oldestCalls = 0
    func begin(_ key: ExpertKey) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        active += 1
        maximum = max(maximum, active)
        if key.expert == 0 {
            oldestCalls += 1
            return oldestCalls == 1
        }
        return false
    }
    func end() {
        lock.lock()
        active -= 1
        lock.unlock()
    }
    func snapshot() -> (active: Int, maximum: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (active, maximum)
    }
}

func prefillReadAheadCompletionRefillAndLateRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
    var metadata: MemoryLedger.Reservation? = try ledger.reserve(
        bytes: PrefillExpertReadAhead.metadataBytes, category: "fixture-read-ahead")
    defer { if let metadata { try? ledger.release(metadata) } }
    try autoreleasepool {
        let config = aheadConfig()
        let cache = try ExpertCache(
            device: aheadDevice(), ledger: ledger,
            byteLimit: config.prefillCacheBytes, slotsPerSlab: 8)
        let limits = IOScheduler.Limits(
            maximumConcurrent: 4, reservedDemandConcurrent: 1,
            maximumOutstanding: 16, reservedDemandOutstanding: 1,
            reservedDemandBytes: 10 * UInt64(PackStore.expertBytes))
        let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
        let oldestStarted = DispatchSemaphore(value: 0)
        let oldestRelease = DispatchSemaphore(value: 0)
        let refilled = DispatchSemaphore(value: 0)
        let lateRelease = DispatchSemaphore(value: 0)
        let closeStarted = DispatchSemaphore(value: 0)
        let closed = DispatchSemaphore(value: 0)
        let readers = AheadRefillReaders()
        let closer = DispatchGroup()
        let loader = SourceExpertLoader(cache: cache, scheduler: scheduler, packedAvailable: { false }) { key, _, destination in
            let delayedOldest = readers.begin(key)
            defer { readers.end() }
            if delayedOldest {
                oldestStarted.signal()
                guard oldestRelease.wait(timeout: .now() + 10) == .success else {
                    throw EngineError("Delayed oldest fixture read timed out")
                }
            } else if key.expert >= 15 {
                // These late readers keep cancellation's destination owners
                // live without allowing an unbounded fixture read loop.
                if key.expert == 15 { refilled.signal() }
                guard lateRelease.wait(timeout: .now() + 10) == .success else {
                    throw EngineError("Late refill fixture read timed out")
                }
            }
            let value = key.expert == 0 && !delayedOldest ? UInt8(0xa7) : aheadSentinel(key)
            destination.initializeMemory(as: UInt8.self, repeating: value)
        }
        let generation = scheduler.currentGeneration
        let ahead = try PrefillExpertReadAhead(
            loader: loader, generation: generation,
            configuration: config, layers: 1)
        defer {
            oldestRelease.signal()
            for _ in 0..<32 { lateRelease.signal() }
            closer.wait()
            ahead.close()
            scheduler.shutdown()
        }
        try aheadRequire(oldestStarted.wait(timeout: .now() + 5) == .success, "Oldest read did not start")
        // A FIFO join cannot submit expert 15 until the held expert 0 completes.
        try aheadRequire(
            refilled.wait(timeout: .now() + 5) == .success,
            "Completed later reads did not refill behind the delayed oldest")
        let state = ahead.snapshot()
        try aheadRequire(
            state.submitted >= 16 && state.joined > 0 && state.maximumPending <= 15
                && state.maximumRetained <= 512, "Completion refill changed pending/resident bounds")
        try aheadRequire(
            state.outOfOrderRetirements > 0 && state.outOfOrderRetirements <= state.joined,
            "Delayed oldest did not record a scope-local out-of-order retirement")
        let laterKey = try ExpertKey(layer: 0, expert: 1)
        var later: ExpertLoadRequest? = try ahead.request(laterKey, shouldContinue: { true })
        let consumer = try loader.joinAndPin(later!)
        try aheadRequire(
            consumer.buffer.bytes[0] == aheadSentinel(laterKey)
                && consumer.buffer.bytes[PackStore.expertBytes - 1] == aheadSentinel(laterKey),
            "Out-of-order completion changed exact consumer bytes")
        try consumer.retireAfterGPUCompletion()
        ahead.release([laterKey][...])
        later = nil
        let oldest = try ahead.request(ExpertKey(layer: 0, expert: 0), shouldContinue: { true })
        try aheadRequire(loader.cancel(oldest), "Held oldest read was not cancellable")
        closer.enter()
        DispatchQueue.global().async {
            defer { closer.leave() }
            closeStarted.signal()
            ahead.close()
            closed.signal()
        }
        try aheadRequire(closeStarted.wait(timeout: .now() + 5) == .success, "Close did not start")
        try aheadRequire(
            closed.wait(timeout: .now() + 0.05) == .timedOut
                && cache.snapshot().inFlightReads > 0 && readers.snapshot().active > 0,
            "Cancellation released a destination before its late writer joined")
        oldestRelease.signal()
        for _ in 0..<32 { lateRelease.signal() }
        try aheadRequire(closed.wait(timeout: .now() + 5) == .success, "Close did not join late completions")
        closer.wait()
        do {
            _ = try loader.joinAndPin(oldest)
            throw EngineError("Cancelled late bytes escaped to a consumer")
        } catch is IOOperationCancelled {}
        let retired = ahead.snapshot()
        let io = scheduler.snapshot()
        try aheadRequire(
            retired.closed && retired.retained == 0 && readers.snapshot().active == 0
                && readers.snapshot().maximum <= 4 && cache.snapshot().inFlightReads == 0
                && cache.snapshot().pinnedSlots == 0 && io.drives.allSatisfy { $0.outstanding == 0 },
            "Completion refill retained a writer/pin or exceeded drive QD4")
        let fresh = try scheduler.beginGeneration()
        let replacement = try loader.loadAndPin(oldest.key, generation: fresh, source: .internalSSD)
        try aheadRequire(
            replacement.buffer.bytes[0] == 0xa7
                && replacement.buffer.bytes[PackStore.expertBytes - 1] == 0xa7,
            "A retired completion corrupted the next generation")
        try replacement.retireAfterGPUCompletion()
    }
    try ledger.release(metadata!)
    metadata = nil
    try aheadRetired(ledger)
}

func prefillReadAheadMaximumRollingOwnership() throws {
    let ledger = try MemoryLedger(ceiling: 7_000_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let engine = try ExpertExecution(
            device: aheadDevice(), ledger: ledger, configuration: aheadConfig(slots: 1984), packedAvailable: { false }
        ) {
            key, _, destination in destination.initializeMemory(as: UInt8.self, repeating: aheadSentinel(key))
        }
        try engine.beginSequence()
        try engine.beginPrefillReadAhead(layers: 5)
        defer { try? engine.finishSequence() }
        let deadline = Date(timeIntervalSinceNow: 15)
        while engine.snapshot().prefillReadAhead!.maximumRetained < 1968 && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        try aheadRequire(engine.snapshot().prefillReadAhead!.maximumRetained == 1968, "Maximum read-ahead retention was not exercised")
        for layer in 0..<5 {
            // A sparse union that ends at 511 checks that unused early records
            // cannot deadlock the bounded producer at a layer transition.
            let experts = [0, 1, 7, 64, 127, 255, 400, 511]
            let window = try engine.makeWindow(layer: layer, experts: experts)
            try window.consumeBatch(experts: experts) { buffers in
                for (expert, buffer) in zip(experts, buffers) {
                    let expected = aheadSentinel(try ExpertKey(layer: layer, expert: expert))
                    try aheadRequire(
                        buffer.bytes[0] == expected && buffer.bytes[PackStore.expertBytes - 1] == expected,
                        "Read-ahead reused a live record or supplied the wrong layer")
                }
                try aheadRequire(engine.snapshot().cache.pinnedSlots <= 1984, "Read-ahead exceeded actual slot capacity")
            }
            try engine.finishPrefillReadAheadLayer(layer)
        }
        engine.finishPrefillReadAhead()
        let state = engine.snapshot()
        let ahead = state.prefillReadAhead!
        try aheadRequire(
            ahead.closed && ahead.retained == 0 && ahead.consumed == 40,
            "Read-ahead did not retire all handoff owners")
        try aheadRequire(
            ahead.maximumPending <= 15 && state.cache.inFlightReads == 0 && state.cache.pinnedSlots == 0,
            "Read-ahead queue/pin bound failed")
        try aheadRequire(
            state.cache.allocatedBytes <= UInt64(1984 * PackStore.expertBytes)
                && state.io.drives.allSatisfy { $0.outstanding == 0 }, "Actual maximum cache or writer retirement failed")
        try engine.beginDecodePhase()
        try engine.finishSequence()
    }
    try aheadRetired(ledger)
}

/// A one-shot gate that holds fixture readers until released.
private final class AheadGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var open = false
    func wait() {
        condition.lock()
        while !open { condition.wait() }
        condition.unlock()
    }
    func release() {
        condition.lock()
        open = true
        condition.broadcast()
        condition.unlock()
    }
}

/// The memory governor shrinks the cache while a prompt is read: the read-ahead holds back to the smaller cache
/// instead of asking for slots that no longer exist (a 500 under memory pressure before).
func prefillReadAheadFollowsShrunkCache() throws {
    let ledger = try MemoryLedger(ceiling: 7_000_000_000, transientMargin: 1_000_000)
    let gate = AheadGate()
    try autoreleasepool {
        let engine = try ExpertExecution(
            device: aheadDevice(), ledger: ledger, configuration: aheadConfig(slots: 1984), packedAvailable: { false }
        ) {
            key, _, destination in
            if key.layer >= 3 { gate.wait() }  // hold the producer with layers 0 to 2 read
            destination.initializeMemory(as: UInt8.self, repeating: aheadSentinel(key))
        }
        defer { gate.release() }
        try engine.beginSequence()
        try engine.beginPrefillReadAhead(layers: 5)
        defer { try? engine.finishSequence() }
        let deadline = Date(timeIntervalSinceNow: 15)
        while engine.snapshot().prefillReadAhead!.retained <= 1536 && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
        Thread.sleep(forTimeInterval: 0.1)
        func consume(_ layer: Int) throws {
            let experts = [0, 1, 7, 64, 127, 255, 400, 511]
            let window = try engine.makeWindow(layer: layer, experts: experts)
            try window.consumeBatch(experts: experts) { buffers in
                for (expert, buffer) in zip(experts, buffers) {
                    let expected = aheadSentinel(try ExpertKey(layer: layer, expert: expert))
                    try aheadRequire(buffer.bytes[0] == expected, "Wrong record after the cache shrank")
                }
            }
            try engine.finishPrefillReadAheadLayer(layer)
        }
        try consume(0)
        // As the governor does: a lower budget, then the trim releases the idle slabs (layer 0's).
        let slab = UInt64(8 * PackStore.expertBytes)
        ledger.setUsable(ledger.reservedBytes - 60 * slab)
        let released = try engine.trimToBudget()
        try aheadRequire(released > 0, "The trim released nothing")
        let shrunk = Int(engine.snapshot().cache.byteLimit) / PackStore.expertBytes
        gate.release()
        Thread.sleep(forTimeInterval: 0.2)
        try aheadRequire(
            engine.snapshot().prefillReadAhead!.retained <= shrunk - 16,
            "Read-ahead held \(engine.snapshot().prefillReadAhead!.retained) records in a \(shrunk)-slot cache")
        for layer in 1..<5 { try consume(layer) }
        engine.finishPrefillReadAhead()
        let ahead = engine.snapshot().prefillReadAhead!
        try aheadRequire(ahead.closed && ahead.retained == 0 && ahead.consumed == 40, "Read-ahead did not finish after the cache shrank")
        try engine.beginDecodePhase()
        try engine.finishSequence()
    }
    try aheadRetired(ledger)
}

func prefillReadAheadExactFallback() throws {
    let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let engine = try ExpertExecution(
            device: aheadDevice(), ledger: ledger,
            configuration: aheadConfig()
        ) { key, source, destination in
            if source == .externalSSD && key.expert == 17 { throw EngineError("Injected planned-source failure") }
            destination.initializeMemory(as: UInt8.self, repeating: aheadSentinel(key))
        }
        try engine.beginSequence()
        try engine.beginPrefillReadAhead(layers: 1)
        defer { try? engine.finishSequence() }
        let window = try engine.makeWindow(layer: 0, experts: [17, 511])
        for expert in [17, 511] {
            try window.consume(expert: expert) { buffer in
                let expected = aheadSentinel(try ExpertKey(layer: 0, expert: expert))
                try aheadRequire(buffer.bytes[0] == expected, "Fallback bytes differ")
            }
        }
        try engine.finishPrefillReadAheadLayer(0)
        engine.finishPrefillReadAhead()
        let snapshot = engine.snapshot()
        try aheadRequire(
            snapshot.io.drives.first { $0.source == .internalSSD }!.succeeded == 1,
            "Read-ahead bypassed or duplicated the exact-source fallback")
        try aheadRequire(
            snapshot.cache.inFlightReads == 0 && snapshot.cache.pinnedSlots == 0,
            "Fallback retained a writer or pin")
    }
    try aheadRetired(ledger)
}

func prefillReadAheadCancellationAndPhaseGuard() throws {
    let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    try autoreleasepool {
        let engine = try ExpertExecution(device: aheadDevice(), ledger: ledger, configuration: aheadConfig(), packedAvailable: { false }) {
            key, _, destination in
            if key.expert == 0 {
                started.signal()
                guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Fixture writer timed out") }
            }
            destination.initializeMemory(as: UInt8.self, repeating: aheadSentinel(key))
        }
        try engine.beginSequence()
        try engine.beginPrefillReadAhead(layers: 2)
        defer { try? engine.finishSequence() }
        guard started.wait(timeout: .now() + 5) == .success else { throw EngineError("No actual read-ahead writer") }
        var refused = false
        do { try engine.beginDecodePhase() } catch is EngineError { refused = true }
        try aheadRequire(refused, "Phase change admitted a live read-ahead scope")
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { release.signal() }
        engine.finishPrefillReadAhead()
        engine.finishPrefillReadAhead()
        let snapshot = engine.snapshot()
        try aheadRequire(
            snapshot.cache.inFlightReads == 0 && snapshot.cache.pinnedSlots == 0
                && snapshot.io.drives.allSatisfy { $0.outstanding == 0 }, "Cancellation released a live writer or handoff pin")
        try engine.beginDecodePhase()
        try engine.finishSequence()
    }
    try aheadRetired(ledger)
}

func prefillReadAheadMaximumAdmission() throws {
    let url = CheckFixtures.file("config/prefill-read-ahead-8192.json")
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: path) }
    var object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var execution = object["execution"] as! [String: Any]
    // The largest prefill cache in whole 8-record slabs that the fixture's 15 GB usable budget
    // (16 GB less a 1 GB margin) holds beside the fixed owners, the replicas and the read-ahead.
    let fixture = try RuntimeConfiguration.load(url)
    let fixed = try ConversationGeometry(context: 8192).maximumFixedBytes(
        input: 8192,
        residentBytes: fixture.residentBytes + fixture.denseResidentBytes)
    let owners = fixed + fixture.experts.replicaOwnerBytes + fixture.experts.readAheadOwnerBytes
    let slab = UInt64(8 * PackStore.expertBytes)
    let maximal = Int((15_000_000_000 - owners) / slab) * 8
    execution["prefill_expert_cache_slots"] = maximal
    object["execution"] = execution
    try JSONSerialization.data(withJSONObject: object).write(to: path)
    let config = try RuntimeConfiguration.load(path)
    try aheadRequire(
        config.experts.prefillReadAhead && config.experts.effectivePrefillCacheSlots == maximal,
        "Actual maximal configuration (\(maximal) prefill slots) was not loaded")
    execution["prefill_expert_cache_slots"] = maximal + 8
    object["execution"] = execution
    try JSONSerialization.data(withJSONObject: object).write(to: path)
    var refused = false
    do { _ = try RuntimeConfiguration.load(path) } catch is EngineError { refused = true }
    try aheadRequire(refused, "Next slab incorrectly passed native admission")
}

/// After a cut with the pressure gone (growth still latched, the budget below its target) a broad prompt scope keeps its
/// read-ahead inside the slots the cache already holds; live pressure sends it back to selective reads (and demotes a
/// running one), counted as live; peak reserved bytes never pass the budget; reopened growth widens it within the budget.
func prefillReadAheadAfterCut() throws {
    let ledger = try MemoryLedger(ceiling: 1_900_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let engine = try ExpertExecution(
            device: aheadDevice(), ledger: ledger,
            configuration: ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 600, slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4,
                prefillReadAhead: true, prefillLayerCommand: true),
            packedAvailable: { false }
        ) {
            key, _, destination in
            destination.initializeMemory(as: UInt8.self, repeating: aheadSentinel(key))
        }
        try engine.beginSequence()
        defer { try? engine.finishSequence() }
        let rows = PrefillReadPolicy.minimumBroadRows
        // One completed dense scope teaches the coverage gate (400 of 512 experts).
        try engine.beginPrefillReadAhead(layers: 1, rows: rows)
        try aheadRequire(engine.usesSelectivePrefill, "An untaught scope read broad")
        try engine.makeWindow(layer: 0, experts: Array(0..<400)).consumePrefillReady { _, _ in }
        try engine.finishPrefillReadAheadLayer(0)
        engine.finishPrefillReadAhead()
        // As after a cut: the budget at the governor's minimum (the cache's 528-slot streaming floor), growth latched,
        // reads open.
        let slots = { Int(ledger.snapshot().categories["expert-slabs"] ?? 0) / PackStore.expertBytes }
        try aheadRequire(slots() == 400, "Fixture cache holds \(slots()) slots")
        // (16 MB more for the layer commands' transient metadata, a mandatory owner.)
        ledger.setUsable(ledger.reservedBytes + UInt64(128 * PackStore.expertBytes) + 16_000_000)
        ledger.setOptionalGrowthAllowed(false)
        // Scopes of `layer + 1` layers, consuming only the last (earlier ones are read ahead and skipped).
        func broad(_ layer: Int = 0) throws -> Bool {
            try engine.beginPrefillReadAhead(layers: layer + 1, rows: rows)
            defer { engine.finishPrefillReadAhead() }
            let started = !engine.usesSelectivePrefill
            if started {
                let deadline = Date(timeIntervalSinceNow: 10)
                while engine.snapshot().prefillReadAhead?.submitted ?? 0 < 400 * (layer + 1) && Date() < deadline {
                    try aheadRequire(ledger.reservedBytes <= ledger.usableBytes, "Read-ahead passed the budget")
                    Thread.sleep(forTimeInterval: 0.002)
                }
                for l in 0...layer {
                    _ = try engine.makeWindow(layer: l, experts: Array(0..<400)).consumeLayer { _ in }
                    try engine.finishPrefillReadAheadLayer(l)
                }
            }
            return started
        }
        let recovering = try broad(1)
        try aheadRequire(recovering, "A broad scope fell back to selective reads during recovery")
        try aheadRequire(
            slots() <= ConversationPhaseBudget.minimumSlots && ledger.snapshot().reservationHighWater <= ledger.usableBytes,
            "Read-ahead during recovery grew the cache past its floor or the reduced budget (\(slots()) slots, high water \(ledger.snapshot().reservationHighWater), budget \(ledger.usableBytes))"
        )
        try aheadRequire(
            ledger.optionalRefusals.live == 0 && ledger.optionalRefusals.belowTarget == 0, "Recovery counted refusals")
        ledger.setOptionalReadsAllowed(false)
        let pressed = try broad()
        try aheadRequire(!pressed && ledger.optionalRefusals.live == 1, "Live pressure admitted a broad scope")
        // Pressure arriving during a broad scope demotes it at the next layer.
        ledger.setOptionalReadsAllowed(true)
        try engine.beginPrefillReadAhead(layers: 1, rows: rows)
        try aheadRequire(!engine.usesSelectivePrefill, "Broad scope did not start")
        ledger.setOptionalReadsAllowed(false)
        try engine.makeWindow(layer: 0, experts: Array(0..<400)).consumePrefillReady { _, _ in }
        try engine.finishPrefillReadAheadLayer(0)
        engine.finishPrefillReadAhead()
        try aheadRequire(
            engine.prefillSchedulingSnapshot.broadDemotions == 1 && ledger.optionalRefusals.live == 2,
            "Live pressure did not demote a running scope")
        // Recovered: growth reopens with a higher budget; the cache may grow, never past it.
        ledger.setOptionalReadsAllowed(true)
        ledger.setOptionalGrowthAllowed(true)
        ledger.setUsable(ledger.maximumUsableBytes)
        let recovered = try broad(1)
        try aheadRequire(
            recovered && slots() > ConversationPhaseBudget.minimumSlots && ledger.reservedBytes <= ledger.usableBytes,
            "Recovered read-ahead did not widen into the regrown budget, or passed it")
        try engine.beginDecodePhase()
    }
    try aheadRetired(ledger)
}
