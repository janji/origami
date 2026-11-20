import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func warmRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
/// Counts fixture expert reads and how many overlap, so a check can prove slab preparation read nothing.
private final class WarmReaderState: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 0, calls = 0, maximum = 0
    func begin() {
        lock.lock()
        active += 1
        calls += 1
        maximum = max(maximum, active)
        lock.unlock()
    }
    func end() {
        lock.lock()
        active -= 1
        lock.unlock()
    }
    func value() -> (active: Int, calls: Int, maximum: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (active, calls, maximum)
    }
}
func decodeSlabPreparationConfigurationAndAdmission() throws {
    let source = CheckFixtures.file("config/prefill-layer-command-8192.json")
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-last-prompt-warm-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ value: Any?) throws -> RuntimeConfiguration {
        var object = original
        var execution = object["execution"] as! [String: Any]
        execution["preallocate_decode_slabs"] = value
        object["execution"] = execution
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let off = try load(nil)
    let on = try load(true)
    let before = try off.memoryAdmission()
    let after = try on.memoryAdmission()
    try warmRequire(
        !off.experts.preallocateDecodeSlabs && on.experts.preallocateDecodeSlabs
            && off.experts.decodeSlabOwnerBytes == 0 && on.experts.decodeSlabOwnerBytes == 16_384
            && after.nonGeometryOwnerBytes == before.nonGeometryOwnerBytes + 16_384
            && after.coldFixedBytes == before.coldFixedBytes + 16_384
            && after.coldTotalBytes == before.coldTotalBytes + 16_384
            && after.noFrontierFixedBytes == before.noFrontierFixedBytes + 16_384
            && after.fullFrontierFixedBytes == before.fullFrontierFixedBytes + 16_384
            && after.shortFixedBytes == before.shortFixedBytes + 16_384,
        "Slab receipt owner was omitted or double charged across live phases")
    let remaining = min(
        before.usableBytes - before.coldTotalBytes,
        before.usableBytes - before.lentArenaFixedBytes - UInt64(before.configuredSlots * PackStore.expertBytes))
    _ = try off.memoryAdmission(additionalOwnerBytes: remaining)
    var overfullRefused = false
    do { _ = try on.memoryAdmission(additionalOwnerBytes: remaining) } catch is EngineError { overfullRefused = true }
    try warmRequire(overfullRefused, "Slab receipt metadata spent the complete admission margin")
    for value in [1, "true"] as [Any] {
        var refused = false
        do { _ = try load(value) } catch { refused = true }
        try warmRequire(refused, "Non-Boolean slab configuration was admitted")
    }
}

func decodeSlabPreparationPreservesDemandAndCancellation() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
    let readers = WarmReaderState()
    try autoreleasepool {
        let config = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred,
            cacheSlots: 16, slotsPerSlab: 8, demandWindow: 8, readConcurrency: 4,
            prefillCacheSlots: 8, preallocateDecodeSlabs: true)
        let engine = try ExpertExecution(device: device, ledger: ledger, configuration: config, packedAvailable: { false }) { _, _, bytes in
            readers.begin()
            defer { readers.end() }
            bytes.initializeMemory(as: UInt8.self, repeating: 0xA7)
        }
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0])
        try window.consume(expert: 0) { bytes in
            try warmRequire(bytes.bytes.allSatisfy { $0 == 0xA7 }, "Fixture read changed")
        }
        try engine.beginDecodePhase()
        let before = engine.snapshot()
        let calls = readers.value().calls
        var cancelled = false
        do { try engine.prepareDecodeSlabs(shouldContinue: { false }) } catch is IOOperationCancelled { cancelled = true }
        try warmRequire(
            cancelled && engine.snapshot().decodeSlabPreparation == nil
                && engine.snapshot().cache.allocatedBytes == before.cache.allocatedBytes,
            "Cancelled slab preparation grew cache or claimed completion")
        try engine.prepareDecodeSlabs(shouldContinue: { true })
        let after = engine.snapshot()
        let receipt = after.decodeSlabPreparation!
        try warmRequire(
            receipt.transitions == 1 && receipt.lastTargetBytes == UInt64(16 * PackStore.expertBytes)
                && receipt.lastCoveredBytes == receipt.lastTargetBytes && receipt.seconds >= 0
                && receipt.physicalHighWater > 0 && receipt.minimumHeadroom >= 3 * 1024 * 1024 * 1024
                && after.cache.allocatedBytes == receipt.lastTargetBytes && readers.value().calls == calls
                && after.cache.hits == before.cache.hits && after.cache.misses == before.cache.misses
                && after.cache.evictions == before.cache.evictions && after.windows == before.windows,
            "Slab preparation omitted capacity, readers or admission/path evidence")
        let reused = try engine.makeWindow(layer: 0, experts: [0])
        try reused.consume(expert: 0) { bytes in
            try warmRequire(bytes.bytes.allSatisfy { $0 == 0xA7 }, "Preserving slab touch changed expert bytes")
        }
        try warmRequire(readers.value().calls == calls, "Slab preparation displaced a resident expert")
        try engine.finishSequence()
    }
    try warmRequire(
        ledger.snapshot().reserved == 0 && readers.value().active == 0,
        "Slab preparation escaped a charged owner or writer")
}

/// `wire_decode_slabs`: boolean, requires slab preallocation, adds no admitted bytes, and its
/// whole-execution path proof is absent when off and present (fallbacks zero) when on.
func decodeSlabWiringConfigurationAndReporting() throws {
    let source = CheckFixtures.file("config/prefill-layer-command-8192.json")
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-slab-wire-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(preallocate: Any?, wire: Any?) throws -> RuntimeConfiguration {
        var object = original
        var execution = object["execution"] as! [String: Any]
        execution["preallocate_decode_slabs"] = preallocate
        execution["wire_decode_slabs"] = wire
        object["execution"] = execution
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let off = try load(preallocate: true, wire: nil)
    let explicitOff = try load(preallocate: true, wire: false)
    let on = try load(preallocate: true, wire: true)
    try warmRequire(
        !off.experts.wireDecodeSlabs && !explicitOff.experts.wireDecodeSlabs && on.experts.wireDecodeSlabs,
        "wire_decode_slabs was not honoured")
    let before = try off.memoryAdmission()
    let after = try on.memoryAdmission()
    try warmRequire(
        after.coldTotalBytes == before.coldTotalBytes && after.nonGeometryOwnerBytes == before.nonGeometryOwnerBytes
            && after.shortFixedBytes == before.shortFixedBytes && after.fullFrontierFixedBytes == before.fullFrontierFixedBytes,
        "Slab wiring changed admitted bytes")
    let sorted = JSONEncoder()
    sorted.outputFormatting = .sortedKeys
    try warmRequire(
        try sorted.encode(off.experts) == sorted.encode(on.experts),
        "Slab wiring changed the coded expert configuration")
    for (preallocate, wire) in [(false, true), (true, 1), (true, "true")] as [(Any, Any)] {
        var refused = false
        do { _ = try load(preallocate: preallocate, wire: wire) } catch is EngineError { refused = true }
        try warmRequire(refused, "Invalid wire_decode_slabs configuration was admitted")
    }
}

/// Wiring replaces the preserving touch without changing expert bytes, reads or cache policy;
/// every slab is wired exactly once (a repeated preparation re-wires nothing), fallbacks stay zero,
/// and releasing the cache returns every charged byte.
func decodeSlabWiringPreservesBytesAndCounts() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
    let readers = WarmReaderState()
    try autoreleasepool {
        var config = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred,
            cacheSlots: 16, slotsPerSlab: 8, demandWindow: 8, readConcurrency: 4,
            prefillCacheSlots: 8, preallocateDecodeSlabs: true)
        config.wireDecodeSlabs = true
        let engine = try ExpertExecution(device: device, ledger: ledger, configuration: config, packedAvailable: { false }) { _, _, bytes in
            readers.begin()
            defer { readers.end() }
            bytes.initializeMemory(as: UInt8.self, repeating: 0x5C)
        }
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0])
        try window.consume(expert: 0) { bytes in
            try warmRequire(bytes.bytes.allSatisfy { $0 == 0x5C }, "Fixture read changed")
        }
        try engine.beginDecodePhase()
        let before = engine.snapshot()
        let calls = readers.value().calls
        try engine.prepareDecodeSlabs(shouldContinue: { true })
        let page = UInt64(getpagesize())
        let target = UInt64(16 * PackStore.expertBytes)
        let receipt = engine.snapshot().decodeSlabPreparation!
        try warmRequire(
            receipt.transitions == 1 && receipt.lastCoveredBytes == target
                && receipt.slabsWired == 2 && receipt.pagesWired == target / page
                && receipt.lastWiredBytes == target && receipt.wireFallbacks == 0,
            "Slab wiring omitted a slab or fell back: \(receipt)")
        try engine.prepareDecodeSlabs(shouldContinue: { true })
        let again = engine.snapshot()
        let repeated = again.decodeSlabPreparation!
        try warmRequire(
            repeated.transitions == 2 && repeated.slabsWired == 2 && repeated.pagesWired == target / page
                && repeated.lastWiredBytes == target && repeated.wireFallbacks == 0,
            "A repeated preparation re-wired or dropped a wired slab: \(repeated)")
        try warmRequire(
            readers.value().calls == calls && again.cache.hits == before.cache.hits
                && again.cache.misses == before.cache.misses && again.cache.evictions == before.cache.evictions
                && again.cache.allocatedBytes == target, "Slab wiring changed reads or cache policy")
        let reused = try engine.makeWindow(layer: 0, experts: [0])
        try reused.consume(expert: 0) { bytes in
            try warmRequire(bytes.bytes.allSatisfy { $0 == 0x5C }, "Slab wiring changed expert bytes")
        }
        try warmRequire(readers.value().calls == calls, "Slab wiring displaced a resident expert")
        try engine.finishSequence()
    }
    try warmRequire(
        ledger.snapshot().reserved == 0 && readers.value().active == 0,
        "Slab wiring escaped a charged owner or writer")
}
