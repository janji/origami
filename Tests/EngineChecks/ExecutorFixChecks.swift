import Foundation
import Metal
import ObjectiveC
import OrigamiEngine

private func demand(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError(message) }
}
private final class Charge {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(_ ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "check_slab")
    }
    deinit { try? ledger.release(reservation) }
}
private enum ChargeKey { nonisolated(unsafe) static var key: UInt8 = 0 }

/// A Metal residency set keeps every buffer it holds alive, so a slab's ledger charge (an associated object, as in
/// ExpertCache) stays until the set is emptied: why WholeExecutionSchedule.trim() empties it after each request.
func residencySetKeepsSlabsUntilEmptied() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let bytes = 1 << 20
    let descriptor = MTLResidencySetDescriptor()
    descriptor.initialCapacity = 4
    let residency = try device.makeResidencySet(descriptor: descriptor)
    try autoreleasepool {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Check buffer unavailable")
        }
        objc_setAssociatedObject(buffer, &ChargeKey.key, try Charge(ledger, bytes: UInt64(bytes)), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        residency.addAllocation(buffer)
        residency.commit()
        residency.requestResidency()
    }
    try demand(ledger.reservedBytes == UInt64(bytes), "The residency set did not keep its buffer")
    autoreleasepool {
        residency.removeAllAllocations()
        residency.commit()
    }
    for _ in 0..<1_000 where ledger.reservedBytes > 0 { Thread.sleep(forTimeInterval: 0.001) }
    try demand(ledger.reservedBytes == 0, "Emptying the residency set did not free its buffer")
}

/// Slab preparation on a host already short of memory keeps the governor's minimum cache as its limit instead of
/// dropping it to the slabs it managed to allocate (none here), which would leave every demand waiting for a slot.
func slabPreparationKeepsMinimumCache() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 64_000_000_000, transientMargin: 1_000_000)
    let slab = UInt64(8 * PackStore.expertBytes)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: 80 * slab, slotsPerSlab: 8)
    // No host has UInt64.max bytes available: preparation stops before its first slab.
    let touch = try cache.prepareSlabs(shouldContinue: { true }, minimumAvailableBytes: .max)
    try demand(
        touch.allocatedBytes == 0 && touch.targetBytes == 66 * slab && cache.slotLimit == 528,
        "Stopped slab preparation lowered the cache below its minimum: \(touch.targetBytes) bytes, \(cache.slotLimit) slots")
}

/// Slab preparation touches (or wires) its pages outside the cache lock (ROADMAP R5): a status snapshot taken from
/// another thread while it runs answers at once instead of waiting for the whole cache to be faulted in.
func slabPreparationAnswersStatus() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 64_000_000_000, transientMargin: 1_000_000)
    let slab = UInt64(8 * PackStore.expertBytes)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: 4 * slab, slotsPerSlab: 8)
    var asked = 0
    var answered = 0
    // Called between slabs, while the pass is under way: each time, a snapshot from another thread must answer in 1 s.
    let touch = try cache.prepareSlabs(
        shouldContinue: {
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.global().async {
                _ = cache.snapshot()
                done.signal()
            }
            asked += 1
            if done.wait(timeout: .now() + 1) == .success { answered += 1 }
            return true
        }, minimumAvailableBytes: 0)
    try demand(
        asked > 1 && answered == asked && touch.slabsTouched == 4 && touch.coveredBytes == 4 * slab,
        "Status waited for slab preparation: \(answered) of \(asked) snapshots answered, \(touch.slabsTouched) slabs touched")
}

/// The prompt read-ahead needs one layer plus the demand window within the governor's minimum cache (528 slots),
/// so a larger `expert_window` is refused when the configuration loads, not at the first prompt read under a squeeze.
func readAheadWindowFitsMinimumCache() throws {
    func configuration(window: Int) throws -> ExpertExecutionConfiguration {
        let object: [String: Any] = [
            "mode": "slabPackedPreferred", "cacheSlots": 1024, "slotsPerSlab": 8,
            "demandWindow": window, "readConcurrency": 4, "prefillReadAhead": true,
        ]
        return try JSONDecoder().decode(ExpertExecutionConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
    }
    _ = try configuration(window: 16)
    var refused = false
    do { _ = try configuration(window: 32) } catch let error as EngineError { refused = "\(error)".contains("expert_window 2...16") }
    try demand(refused, "A read-ahead window beyond the minimum cache was accepted")
}

/// A refused fast request names what the user sets: the `origami_mode` field and the `allow_fast` setting.
func fastRefusalNamesUserSettings() throws {
    var refusal: EngineError?
    do { _ = try FastLateMissSkip.requestTau(.lateMissSkip, allowedTau: nil) } catch let error as EngineError { refusal = error }
    let text = refusal.map { "\($0)" } ?? ""
    try demand(
        refusal?.code == "fast_not_allowed" && text.contains("origami_mode") && text.contains("allow_fast")
            && !text.contains("qfs_mode") && !text.contains("fast.allow_opt_in"), "Fast refusal text: \(text)")
}

/// Mid-reply cache growth: slabs a trim took come back at a token boundary. Prepared slabs stay outside the cache with
/// their ledger charge, a trim releases them, and while slab growth is held (pre-encoded commands queued) they do not
/// attach. Attaching keeps every slot's record bytes, pins and GPU table entries, and the new slots start free.
func expertCacheGrowsAtTokenBoundary() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let ledger = try MemoryLedger(ceiling: 64_000_000_000, transientMargin: 1_000_000)
    let slab = UInt64(8 * PackStore.expertBytes)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: 4 * slab, slotsPerSlab: 8, maximumByteLimit: 8 * slab)
    _ = try cache.gpuTable()
    func fill(_ key: ExpertKey) throws -> ExpertTicket {
        let a = try cache.acquire(key)
        let record = try cache.beginRead(a.ticket)
        record.bytes.initializeMemory(as: UInt8.self, repeating: UInt8(truncatingIfNeeded: key.layer * 31 + key.expert))
        try demand(cache.completeRead(a.ticket, verified: true), "Fixture record did not verify")
        return a.ticket
    }
    let tickets = try (0..<32).map { try fill(try ExpertKey(layer: 3, expert: $0)) }
    // A GPU pin in each of the first two slabs, so the trim keeps them and drops the other two.
    for i in [0, 8] { _ = try cache.pin(tickets[i]) }
    try cache.resize(to: 2 * slab)
    cache.setGrowthCeiling(8 * slab)
    try demand(cache.snapshot().allocatedBytes == 2 * slab, "The trim did not drop the two cold slabs")
    let kept = Array(tickets[0..<16])
    let reserved = ledger.reservedBytes
    let generation = cache.slabBuffers().generation
    // A trim before the boundary releases the prepared slabs and their charge.
    try demand(
        try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 6 && ledger.reservedBytes == reserved + 6 * slab,
        "Growth did not prepare the six slabs up to the ceiling")
    cache.dropGrowth()
    try demand(
        ledger.reservedBytes == reserved && cache.attachGrowth() == 0,
        "A trim did not release the prepared slabs: \(ledger.reservedBytes) vs \(reserved)")
    // A budget that grew back 3 slabs short of the phase's: the rest of the model keeps its share, so 3 come back.
    let usable = ledger.usableBytes
    ledger.setUsable(usable - 3 * slab)
    try demand(try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 3, "Growth took the rest of the model's share")
    cache.dropGrowth()
    ledger.setUsable(usable)
    try demand(try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 6, "Growth did not prepare again after the trim")
    try demand(
        cache.snapshot().allocatedBytes == 2 * slab && cache.slabBuffers().generation == generation,
        "Prepared slabs reached the cache before the boundary")
    cache.holdSlabGrowth(true)
    try demand(cache.attachGrowth() == 0, "Prepared slabs attached while pre-encoded commands held the layout")
    cache.holdSlabGrowth(false)
    try demand(cache.attachGrowth() == 6, "Prepared slabs did not attach at the boundary")
    let after = cache.snapshot()
    try demand(
        after.allocatedBytes == 8 * slab && after.byteLimit == 8 * slab && cache.slabBuffers().generation != generation
            && cache.slabBuffers().buffers.count == 8 && ledger.reservedBytes == reserved + 6 * slab,
        "Attached growth: \(after.allocatedBytes) bytes allocated, limit \(after.byteLimit)")
    // Every kept record is still published, readable with its own bytes, and the old pins retire normally.
    let epoch = cache.holdForGPU(layer: 3)
    cache.releaseForGPU(layer: nil)
    try demand(cache.gpuComputes(kept.map(\.key), epoch: epoch).allSatisfy { $0 }, "Attaching unpublished a kept record")
    for ticket in kept {
        let record = try cache.pin(ticket)
        let expected = UInt8(truncatingIfNeeded: ticket.key.layer * 31 + ticket.key.expert)
        try demand(record.bytes.allSatisfy { $0 == expected }, "Attaching changed record \(ticket.key.expert)")
        try cache.unpinAfterGPUCompletion(ticket)
    }
    for i in [0, 8] { try cache.unpinAfterGPUCompletion(tickets[i]) }
    // The 48 new slots start free: new records fill them without evicting a kept one.
    for expert in 0..<48 { _ = try fill(try ExpertKey(layer: 4, expert: expert)) }
    try demand(
        cache.snapshot().evictions == 0 && (try cache.draftResidentExperts(layer: 3)) == Array(0..<16),
        "New slots were not free, or a kept record was evicted")
    try demand(try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 0, "Growth went past its ceiling")
}
