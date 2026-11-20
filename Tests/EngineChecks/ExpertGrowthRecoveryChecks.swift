import Foundation
import Metal
import OrigamiEngine

private func demand(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError(message) }
}

/// A decode phase may begin while pressure has already reduced the budget. Its recovery ceiling belongs to the
/// full phase plan, not that initial allowance. The same records survive two recovery steps without a new request.
func expertPreparedGrowthRecoversReducedStart() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 64_000_000_000, transientMargin: 1_000_000)
    let slab = UInt64(8 * PackStore.expertBytes)
    let full = ledger.maximumUsableBytes
    let otherOwners = full - 8 * slab
    let fixed = try ledger.reserve(bytes: otherOwners, category: "fixture-phase-owners")
    defer { try? ledger.release(fixed) }
    ledger.setUsable(otherOwners + 2 * slab)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: 2 * slab, slotsPerSlab: 8, maximumByteLimit: 8 * slab)
    var kept: [ExpertTicket] = []
    for expert in 0..<9 {
        let acquired = try cache.acquire(ExpertKey(layer: 2, expert: expert))
        let buffer = try cache.beginRead(acquired.ticket)
        buffer.bytes[0] = UInt8(expert)
        buffer.bytes[buffer.length - 1] = UInt8(expert + 1)
        try demand(cache.completeRead(acquired.ticket, verified: true), "Initial record did not verify")
        kept.append(acquired.ticket)
    }
    cache.setGrowthCeiling(8 * slab, budget: full)
    try demand(
        try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 0,
        "Reduced-start growth consumed the phase's other owners")
    ledger.setUsable(otherOwners + 5 * slab)
    try demand(
        try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 3 && cache.attachGrowth() == 3,
        "Reduced-start phase did not admit its first recovery step")
    try demand(
        cache.snapshot().byteLimit == 5 * slab && ledger.reservedBytes == otherOwners + 5 * slab,
        "First recovery exceeded the current owner allowance")
    ledger.setUsable(full)
    try demand(
        try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 3 && cache.attachGrowth() == 3,
        "Reduced-start phase remained capped at its initial cache size")
    try demand(
        cache.snapshot().byteLimit == 8 * slab && cache.snapshot().allocatedBytes == 8 * slab && ledger.reservedBytes == full,
        "Final recovery did not reach the admitted target")
    for ticket in kept {
        let buffer = try cache.pin(ticket)
        try demand(
            buffer.bytes[0] == UInt8(ticket.key.expert) && buffer.bytes[buffer.length - 1] == UInt8(ticket.key.expert + 1),
            "Recovery changed an existing expert record")
        try cache.unpinAfterGPUCompletion(ticket)
    }
}

/// File-backed decode uses lazy slabs. The governor restores its target synchronously without allocating, reading,
/// or changing existing GPU addresses; the next demand still goes through normal slab admission.
func expertLazyGrowthRecoversReducedStart() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 64_000_000_000, transientMargin: 1_000_000)
    let slab = UInt64(8 * PackStore.expertBytes)
    let full = ledger.maximumUsableBytes
    let configuration = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 64, slotsPerSlab: 8,
        demandWindow: 2, readConcurrency: 1, prefillCacheSlots: 8, preallocateDecodeSlabs: false)
    let engine = try ExpertExecution(device: device, ledger: ledger, configuration: configuration, packedAvailable: { false }) { _, _, _ in
        throw EngineError("Restoring a lazy cache limit must not read an expert")
    }
    ledger.setUsable(full - 6 * slab)
    try engine.beginSequence()
    try engine.beginDecodePhase(cacheBytes: 2 * slab, recoveryBytes: 8 * slab)
    let reserved = ledger.reservedBytes
    engine.prepareGrowth()
    try demand(engine.cacheSnapshot.byteLimit == 2 * slab, "Lazy growth ignored the reduced allowance")
    var rejected = false
    do { try engine.beginDecodePhase(cacheBytes: 2 * slab, recoveryBytes: slab) } catch let error as EngineError {
        rejected = error.code == "invalid_configuration"
    }
    try demand(rejected && engine.cacheSnapshot.byteLimit == 2 * slab, "Invalid recovery ceiling changed the phase")
    ledger.setUsable(full - 3 * slab)
    engine.prepareGrowth()
    try demand(engine.cacheSnapshot.byteLimit == 5 * slab, "Lazy decode did not recover during the same phase")
    ledger.setUsable(full)
    engine.prepareGrowth()
    try demand(
        engine.cacheSnapshot.byteLimit == 8 * slab && engine.cacheSnapshot.allocatedBytes == 0
            && ledger.reservedBytes == reserved, "Lazy recovery allocated memory or failed to reach the target")
    // No expert slab exists yet, so actual reservations fit both budgets. Pressure must nevertheless remove the
    // prospective capacity, or later demand could fill the stale eight-slab limit while the allowance is only two.
    ledger.setUsable(full - 6 * slab)
    try demand(ledger.reservedBytes < ledger.usableBytes, "Fixture must exercise the unallocated pressure case")
    _ = try engine.trimToBudget()
    engine.prepareGrowth()
    try demand(
        engine.cacheSnapshot.byteLimit == 2 * slab && engine.cacheSnapshot.allocatedBytes == 0
            && ledger.reservedBytes == reserved, "Pressure left an oversized lazy allowance with no live slabs")
    ledger.setUsable(full)
    engine.prepareGrowth()
    try demand(engine.cacheSnapshot.byteLimit == 8 * slab, "Lazy capacity did not recover after a second pressure cycle")
    try engine.finishSequence()
    try engine.preparePrefillPhase(cacheBytes: slab)
    engine.prepareGrowth()
    try demand(
        engine.cacheSnapshot.byteLimit == slab && engine.cacheSnapshot.allocatedBytes == 0,
        "A queued recovery crossed into the next prefill phase")
}
