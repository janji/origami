import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func phaseRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func phaseFailure(_ code: String? = nil, line: Int = #line, _ body: () throws -> Void) throws {
    do { try body() } catch let error as EngineError {
        if let code { try phaseRequire(error.code == code, "Expected \(code), received \(String(describing: error.code)): \(error)") }
        return
    }
    throw EngineError("Expected expert-phase operation to fail (ExpertPhaseChecks.swift:\(line))")
}
private func phaseDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}
private func phaseCacheSignature(_ cache: ExpertCache) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return try encoder.encode(cache.snapshot())
}

func expertPhaseConfigurationDefaultsAndBounds() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-phase-config-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("runtime.json")
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let defaults = try load([:]).experts
    try phaseRequire(
        defaults.prefillCacheSlots == nil && defaults.effectivePrefillCacheSlots == 64 && defaults.cacheSlots == 64,
        "Missing phase configuration changed the existing limit")
    let good: [String: Any] = [
        "expert_mode": "slabPackedPreferred", "expert_cache_slots": 16,
        "expert_slots_per_slab": 2, "expert_window": 2, "io_queue_depth": 1, "prefill_expert_cache_slots": 4,
    ]
    let parsed = try load(good).experts
    try phaseRequire(
        parsed.effectivePrefillCacheSlots == 4 && parsed.cacheSlots == 16 && parsed.prefillCacheBytes == UInt64(4 * PackStore.expertBytes),
        "Phase configuration lost its independent prefill/decode limits")
    for invalid in [0, 1, 3, 18, true, "4"] as [Any] {
        var changed = good
        changed["prefill_expert_cache_slots"] = invalid
        try phaseFailure { _ = try load(changed) }
    }
    var exceedsWindow = good
    exceedsWindow["expert_window"] = 5
    try phaseFailure("invalid_configuration") { _ = try load(exceedsWindow) }
    var maximum = good
    maximum["expert_cache_slots"] = 48 * 512 + 2
    try phaseFailure("invalid_configuration") { _ = try load(maximum) }
    var legacy = good
    legacy["expert_mode"] = "legacyOriginal"
    try phaseFailure("invalid_configuration") { _ = try load(legacy) }
    let rounded = try ExpertExecutionConfiguration(
        mode: .slabPackedPreferred,
        cacheBytes: UInt64(16 * PackStore.expertBytes + 99), slotsPerSlab: 2,
        demandWindow: 2, readConcurrency: 4, prefillCacheSlots: 4)
    try phaseRequire(
        rounded.cacheSlots == 16 && rounded.effectivePrefillCacheSlots == 4 && rounded.readConcurrency == 2,
        "Byte-budget rounding altered phase bounds or queue clamping")
    for (context, slots) in [(8192, 2192), (32768, 2128)] {
        let complete = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: slots,
            slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4, prefillCacheSlots: 1024)
        for count in [0, 1, 255] {
            try phaseRequire(
                try ConversationPhaseBudget.expertBytes(
                    context: context, input: count,
                    residentBytes: 3_450_000_000, experts: complete, usable: 15_000_000_000) == complete.cacheBytes,
                "Complete conversation bound discarded feasible hot experts")
        }
        // A budget too small for the fixed owners yields the minimum cache, never an error.
        try phaseRequire(
            try ConversationPhaseBudget.expertBytes(
                context: context, input: 255,
                residentBytes: 3_450_000_000, experts: complete, usable: 9_000_000_000) == UInt64(528 * PackStore.expertBytes),
            "A short budget did not fall back to the minimum cache")
    }
    // Caller-owned diagnostics must reduce short-append residency without
    // changing the selected decode capacity or rounding above the ceiling.
    for (context, slots, httpShort, screenShort) in [(8192, 3008, 2600, 2584), (32768, 2576, 2168, 2152)] {
        let complete = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: slots,
            slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4, prefillCacheSlots: 1024)
        for (additional, shortSlots) in [(UInt64(16 * 1024 * 1024), httpShort), (UInt64(64_000_000), screenShort)] {
            for count in [0, 1, 255] {
                let bytes = try ConversationPhaseBudget.expertBytes(
                    context: context, input: count,
                    residentBytes: 3_450_000_000, experts: complete, usable: 15_000_000_000,
                    additionalOwners: additional)
                let fixed = try ConversationGeometry(context: context).maximumFixedBytes(input: count, residentBytes: 3_450_000_000)
                // Small suffix GPU work now owns additional memory. Check the admission contract,
                // including maximal slab-rounded residency, rather than the previous fixed slot counts.
                let owners =
                    additional + complete.replicaOwnerBytes + complete.readAheadOwnerBytes
                    + complete.layerCommandOwnerBytes + complete.reuseOwnerBytes + complete.decodeSlabOwnerBytes
                let slab = UInt64(complete.slotsPerSlab * PackStore.expertBytes)
                try phaseRequire(
                    bytes <= UInt64((count == 0 ? slots : shortSlots) * PackStore.expertBytes)
                        && bytes % slab == 0 && fixed + owners + bytes <= 15_000_000_000
                        && (bytes == complete.cacheBytes || fixed + owners + bytes + slab > 15_000_000_000),
                    "Phase admission omitted an owner or discarded a feasible slab: context \(context), input \(count), bytes \(bytes), fixed \(fixed), owners \(owners)"
                )
            }
        }
        try phaseFailure("insufficient_memory") {
            _ = try ConversationPhaseBudget.expertBytes(
                context: context, input: 255,
                residentBytes: 3_450_000_000, experts: complete, usable: 15_000_000_000,
                additionalOwners: UInt64.max)
        }
    }
    // Earlier serialized configurations omit the new optional field.
    let encoder = JSONEncoder()
    var oldObject = try JSONSerialization.jsonObject(with: encoder.encode(parsed)) as! [String: Any]
    oldObject.removeValue(forKey: "prefillCacheSlots")
    let restored = try JSONDecoder().decode(ExpertExecutionConfiguration.self, from: JSONSerialization.data(withJSONObject: oldObject))
    try phaseRequire(
        restored.prefillCacheSlots == nil && restored.effectivePrefillCacheSlots == restored.cacheSlots,
        "Earlier execution configurations no longer decode")
}

func expertPhaseLazyGrowthColdShrinkAndResourceCharge() throws {
    let bytes = UInt64(2 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1)
    let cache = try ExpertCache(device: phaseDevice(), ledger: ledger, byteLimit: bytes, slotsPerSlab: 2, maximumByteLimit: 2 * bytes)
    func populate(_ expert: Int) throws -> ExpertBuffer {
        let acquisition = try cache.acquire(ExpertKey(layer: 0, expert: expert))
        try phaseRequire(acquisition.newlyReserved, "Fixture expected a new record")
        let buffer = try cache.beginRead(acquisition.ticket)
        buffer.bytes[0] = UInt8(expert)
        try phaseRequire(cache.completeRead(acquisition.ticket, verified: true), "Fixture record did not publish")
        return buffer
    }
    try autoreleasepool {
        _ = try populate(0)
        _ = try populate(1)
        let before = ledger.snapshot().reserved
        try phaseRequire(try cache.resizeQuiescent(to: 2 * bytes) == bytes, "Raising a phase limit allocated storage")
        try phaseRequire(
            cache.snapshot().byteLimit == 2 * bytes && ledger.snapshot().reserved == before,
            "Lazy growth changed the ledger or failed to publish its target")
        var escaped: MTLBuffer? = try populate(2).buffer
        _ = try populate(3)
        // A fifth demand sweeps CLOCK's reference bits. Slab 0 has the new hot
        // record; slab 1 is entirely cold and should be the whole-slab victim.
        _ = try populate(4)
        try phaseRequire(cache.snapshot().evictions == 1, "Fixture did not clear CLOCK reference bits")
        try phaseRequire(try cache.resizeQuiescent(to: bytes) == bytes, "Cold shrink did not release one whole slab")
        let hot = try cache.acquire(ExpertKey(layer: 0, expert: 4))
        try phaseRequire(!hot.newlyReserved, "Shrinking discarded the hot slab")
        let pinned = try cache.pin(hot.ticket)
        try phaseRequire(pinned.bytes[0] == 4, "Retained hot record changed")
        try cache.unpinAfterGPUCompletion(hot.ticket)
        try withExtendedLifetime(escaped) {
            try phaseRequire(
                escaped != nil && ledger.snapshot().reserved == 2 * bytes,
                "Escaped raw Metal storage lost its ledger charge after cold shrink")
        }
        escaped = nil
    }
    try phaseRequire(ledger.snapshot().reserved == bytes, "Released cold resource retained its charge after autorelease drain")
    for iteration in 0..<32 {
        try autoreleasepool {
            try cache.resizeQuiescent(to: 2 * bytes)
            _ = try populate(100 + 2 * iteration)
            _ = try populate(101 + 2 * iteration)
            try cache.resizeQuiescent(to: bytes)
        }
        try phaseRequire(
            cache.snapshot().metadataSlots <= 4 && ledger.snapshot().reserved == bytes,
            "Repeated phase changes grew slot metadata or orphaned slab reservations")
    }
}

func expertPhaseBusyChangesAreAtomic() throws {
    let device = try phaseDevice()
    let bytes = UInt64(2 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: 50_000_000, transientMargin: 1)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: bytes, slotsPerSlab: 2, maximumByteLimit: 2 * bytes)
    let ticket = try cache.acquire(ExpertKey(layer: 1, expert: 0)).ticket
    func rejectBusy() throws {
        let before = try phaseCacheSignature(cache)
        for target in [UInt64(0), bytes, 2 * bytes] {
            try phaseFailure("expert_phase_busy") { try cache.resizeQuiescent(to: target) }
            try phaseRequire(try phaseCacheSignature(cache) == before, "Busy phase change mutated cache state")
        }
    }
    try rejectBusy()  // A reservation is busy even before its producer starts.
    var writer: ExpertBuffer? = try cache.beginRead(ticket)
    try rejectBusy()
    try phaseRequire(!cache.completeRead(ticket, verified: false), "Unverified producer was published")
    writer = nil
    let next = try cache.acquire(ExpertKey(layer: 1, expert: 1)).ticket
    _ = try cache.beginRead(next)
    try phaseRequire(cache.completeRead(next, verified: true), "Second producer did not publish")
    var escapedPin: ExpertBuffer? = try cache.pin(next)
    try withExtendedLifetime(escapedPin) { try rejectBusy() }
    try cache.unpinAfterGPUCompletion(next)
    escapedPin = nil
    try cache.resizeQuiescent(to: 2 * bytes)
    let before = try phaseCacheSignature(cache)
    for target in [UInt64(PackStore.expertBytes), 3 * bytes] {
        try phaseFailure("invalid_configuration") { try cache.resizeQuiescent(to: target) }
        try phaseRequire(try phaseCacheSignature(cache) == before, "Invalid resize mutated cache state")
    }
    withExtendedLifetime(writer) {}
}

func expertPhaseAdmissionFailureIsNotEvictionFallback() throws {
    let bytes = UInt64(2 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: bytes + 4096, transientMargin: 1)
    let cache = try ExpertCache(device: phaseDevice(), ledger: ledger, byteLimit: bytes, slotsPerSlab: 2, maximumByteLimit: 2 * bytes)
    for expert in 0..<2 {
        let ticket = try cache.acquire(ExpertKey(layer: 2, expert: expert)).ticket
        _ = try cache.beginRead(ticket)
        try phaseRequire(cache.completeRead(ticket, verified: true), "Fixture record did not publish")
    }
    try cache.resizeQuiescent(to: 2 * bytes)
    let before = try phaseCacheSignature(cache)
    try phaseFailure("insufficient_memory") { _ = try cache.acquire(ExpertKey(layer: 2, expert: 2)) }
    try phaseRequire(
        try phaseCacheSignature(cache) == before && ledger.snapshot().reserved == bytes,
        "Denied lazy growth evicted a live record or mutated accounting")
    for expert in 0..<2 {
        let hit = try cache.acquire(ExpertKey(layer: 2, expert: expert))
        try phaseRequire(!hit.newlyReserved, "Admission failure silently replaced an existing expert")
    }
}

func expertPhaseExecutionLifecycleAndBusyBoundary() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1)
    let configuration = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 8,
        slotsPerSlab: 2, demandWindow: 2, readConcurrency: 1, prefillCacheSlots: 4)
    let engine = try ExpertExecution(device: phaseDevice(), ledger: ledger, configuration: configuration, packedAvailable: { false }) {
        key, _, destination in
        destination[0] = UInt8(key.expert)
    }
    let fresh = engine.snapshot()
    try phaseRequire(
        fresh.phase == .prefill && fresh.targetCacheSlots == 4 && fresh.cache.allocatedBytes == 0
            && fresh.cache.maximumByteLimit == configuration.cacheBytes,
        "Expert helper did not start at the bounded prefill target")
    try phaseFailure("expert_phase_state") { try engine.beginDecodePhase() }
    try phaseFailure("expert_phase_state") { try engine.joinReadProducers() }
    try engine.beginSequence()
    let first = try engine.makeWindow(layer: 3, experts: [0, 1])
    try phaseFailure("expert_phase_busy") { try engine.joinReadProducers() }
    try phaseFailure("expert_phase_busy") { try engine.beginDecodePhase() }
    try phaseFailure("expert_phase_busy") { try engine.preparePrefillPhase() }
    try phaseRequire(
        engine.snapshot().phase == .prefill && engine.snapshot().phaseTransitions == 0,
        "Busy execution phase change mutated its phase")
    try first.consume(expert: 0) { buffer in
        try phaseRequire(buffer.bytes[0] == 0, "Wrong first expert")
        try phaseFailure("expert_phase_busy") { try engine.beginDecodePhase() }
    }
    try first.consume(expert: 1) { buffer in try phaseRequire(buffer.bytes[0] == 1, "Wrong second expert") }
    let before = engine.snapshot().cache.allocatedBytes
    try engine.beginDecodePhase()
    try phaseRequire(
        engine.snapshot().phase == .decode && engine.snapshot().targetCacheSlots == 8 && engine.snapshot().cache.allocatedBytes == before,
        "Decode transition lost existing slabs or allocated eagerly")
    let second = try engine.makeWindow(layer: 3, experts: [2, 3, 4, 5])
    for expert in 2...5 { try second.consume(expert: expert) { try phaseRequire($0.bytes[0] == UInt8(expert), "Wrong decode expert") } }
    try phaseRequire(
        engine.snapshot().cache.allocatedBytes > configuration.prefillCacheBytes, "Fixture did not grow into the decode allowance")
    let joinedBefore = engine.snapshot()
    try engine.joinReadProducers()
    let joinedAfter = engine.snapshot()
    try phaseRequire(
        joinedAfter.io.generation == joinedBefore.io.generation
            && joinedAfter.phase == joinedBefore.phase && joinedAfter.phaseTransitions == joinedBefore.phaseTransitions
            && joinedAfter.cache.allocatedBytes == joinedBefore.cache.allocatedBytes
            && joinedAfter.cache.hits == joinedBefore.cache.hits && joinedAfter.cache.misses == joinedBefore.cache.misses
            && joinedAfter.cache.reuseRows == joinedBefore.cache.reuseRows
            && joinedAfter.cache.inFlightReads == 0 && joinedAfter.io.drives.allSatisfy({ $0.outstanding == 0 }),
        "Joined read boundary changed the active sequence/cache or retained producers")
    try engine.finishSequence()
    try phaseRequire(engine.snapshot().phase == .decode, "Sequence cleanup eagerly discarded decode cache state")
    let retained = engine.snapshot().cache
    try engine.preparePrefillPhase(cacheBytes: configuration.cacheBytes)
    try phaseRequire(
        engine.snapshot().phase == .prefill && engine.snapshot().cache.allocatedBytes == retained.allocatedBytes,
        "Reused prompt lost hot storage or retained decode arithmetic")
    try engine.beginSequence(prefillCacheBytes: configuration.cacheBytes)
    try phaseRequire(
        engine.snapshot().targetCacheBytes == configuration.cacheBytes,
        "Second preparation discarded the prospectively admitted target")
    let hitsBefore = engine.snapshot().cache.hits
    let repeated = try engine.makeWindow(layer: 3, experts: [0, 1, 2, 3, 4, 5])
    for expert in 0...5 { try repeated.consume(expert: expert) { try phaseRequire($0.bytes[0] == UInt8(expert), "Reused expert changed") } }
    try phaseRequire(engine.snapshot().cache.hits >= hitsBefore + 6, "Prospective reuse reread retained records")
    try engine.finishSequence()
    for invalid in [configuration.prefillCacheBytes - 1, configuration.cacheBytes + 1] {
        try phaseFailure("invalid_configuration") { try engine.preparePrefillPhase(cacheBytes: invalid) }
        try phaseRequire(
            engine.snapshot().targetCacheBytes == configuration.cacheBytes,
            "Invalid prospective capacity mutated the stable target")
    }
    try autoreleasepool { try engine.preparePrefillPhase() }
    let prepared = engine.snapshot()
    try phaseRequire(
        prepared.phase == .prefill && prepared.targetCacheBytes == configuration.prefillCacheBytes
            && prepared.cache.allocatedBytes <= configuration.prefillCacheBytes,
        "Next request did not shrink before admission")
    try phaseRequire(
        prepared.phaseTransitions == 2 && prepared.cache.pinnedSlots == 0 && prepared.cache.inFlightReads == 0,
        "Phase lifecycle left live work or miscounted transitions")
    // Preparation is idempotent if a later state admission fails. No active
    // generation has been created, so the next request can prepare/begin again.
    try phaseFailure("insufficient_memory") {
        _ = try ledger.reserve(bytes: ledger.ceiling, category: "synthetic_sequence_admission")
    }
    try phaseRequire(
        engine.snapshot().phase == .prefill && engine.snapshot().phaseTransitions == 2,
        "Denied sequence admission changed the stable prefill phase")
    try engine.preparePrefillPhase()
    try engine.beginSequence()
    try phaseRequire(engine.snapshot().phaseTransitions == 2, "Repeated preparation changed phase accounting")
    try engine.finishSequence()
}

func expertPhaseCancellationJoinsAndKeepsStableTarget() throws {
    let ledger = try MemoryLedger(ceiling: 50_000_000, transientMargin: 1)
    let control = GenerationControl()
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let configuration = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 4,
        slotsPerSlab: 2, demandWindow: 1, readConcurrency: 1, prefillCacheSlots: 2)
    let engine = try ExpertExecution(device: phaseDevice(), ledger: ledger, configuration: configuration, packedAvailable: { false }) {
        _, _, destination in
        started.signal()
        guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Phase cancellation producer timeout") }
        destination[0] = 0xca
    }
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 4, experts: [1], shouldContinue: { !control.isCancelled })
    try phaseFailure("expert_phase_busy") { try engine.beginDecodePhase() }
    let group = DispatchGroup()
    group.enter()
    DispatchQueue.global().async {
        _ = started.wait(timeout: .now() + 5)
        control.cancel()
        release.signal()
        group.leave()
    }
    do {
        try window.consume(expert: 1) { _ in throw EngineError("Cancelled phase producer became visible") }
        throw EngineError("Cancelled phase window returned")
    } catch is IOOperationCancelled {}
    group.wait()
    try engine.finishSequence()
    let result = engine.snapshot()
    try phaseRequire(
        result.phase == .prefill && result.phaseTransitions == 0 && result.targetCacheSlots == 2,
        "Cancellation mutated phase configuration")
    try phaseRequire(
        result.cache.inFlightReads == 0 && result.cache.pinnedSlots == 0 && result.cache.reservedSlots == 0
            && result.io.drives.allSatisfy { $0.outstanding == 0 },
        "Cancelled request returned before producer and pins retired")
    try engine.preparePrefillPhase()
    try engine.beginSequence()
    try engine.beginDecodePhase()
    try engine.finishSequence()
    try engine.preparePrefillPhase()
    try phaseRequire(
        engine.snapshot().phase == .prefill && engine.snapshot().phaseTransitions == 2,
        "Cancelled request prevented the next complete phase lifecycle")
}
