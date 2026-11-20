import Foundation
import Metal
import OrigamiEngine

/// The plainest policy: a free slot, then a new slab, then the lowest-index slot the core calls evictable.
private final class LowestSlotPolicy: ExpertSlotPolicy {
    typealias Observer = HoldingObserver
    func victim(for key: Int, sizeClass: Int, in core: ExpertSlotCore<LowestSlotPolicy>) throws -> Int? {
        let slots = core.classes[sizeClass]
        if let index = slots.firstFree(from: 0) { return index }
        if let index = try core.allocateSlabLocked(sizeClass: sizeClass) { return index }
        guard let index = slots.slots.indices.first(where: { core.isEvictableLocked($0, sizeClass: sizeClass) }) else { return nil }
        core.evictLocked(index, sizeClass: sizeClass)
        return index
    }
    func didHit(_ slot: Int, sizeClass: Int) {}
    func didReserve(_ slot: Int, sizeClass: Int) {}
    func didChange(_ slot: Int, key: Int, sizeClass: Int, to phase: ExpertSlotPhase) {}
    func releasesOnLastUnpin(_ slot: Int, sizeClass: Int) -> Bool { false }
    func didForget(_ key: Int) {}
    func didInsertSlab(_ slots: [Int], sizeClass: Int) {}
    func orderRetirement(_ slabs: inout [Int], sizeClass: Int, in core: ExpertSlotCore<LowestSlotPolicy>) { slabs.sort() }
}
private final class HoldingObserver: ExpertSlotObserver {
    var held: Set<[Int]> = []
    var forgotten: [Int] = []
    var verified = 0, retiredSlabs = 0
    func isHeld(_ slot: Int, key: Int?, sizeClass: Int) -> Bool { held.contains([sizeClass, slot]) }
    func didVerify(_ slot: Int, sizeClass: Int, in slots: ExpertSlotClass) { verified += 1 }
    func willForget(_ key: Int, slot: Int?, sizeClass: Int) { forgotten.append(key) }
    func didInsertSlab(_ id: Int, sizeClass: Int, in slots: ExpertSlotClass) {}
    func didRetireSlab(_ id: Int, sizeClass: Int) { retiredSlabs += 1 }
}

/// `slot_core_size_classes`: three layers of three record sizes in one core. Bytes written into a slot of each class
/// read back exactly; a miss only ever evicts from its own class and each class stays within its limit; a recycled
/// slot's old ticket is rejected; only a reserved slot cancels; pins and the observer's hold block eviction; a retired
/// idle slab returns its ledger bytes and its holes are reused lowest first.
func slotCoreSizeClasses() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError("Slot core: " + message) } }
    func mustFail(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Slot core: an invalid transition succeeded")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 64 << 20, transientMargin: 1)
    let classes: [ExpertSlotShape.SizeClass] = [
        .init(recordBytes: 4_096, slotsPerSlab: 2, byteLimit: 16_384, maximumByteLimit: 16_384, category: "check-slots"),
        .init(recordBytes: 12_288, slotsPerSlab: 1, byteLimit: 24_576, maximumByteLimit: 24_576, category: "check-slots"),
        .init(recordBytes: 1_000, slotsPerSlab: 4, byteLimit: 4_000, maximumByteLimit: 4_000, category: "check-slots"),
    ]
    let observer = HoldingObserver()
    let core = try ExpertSlotCore(
        device: device, ledger: ledger,
        shape: ExpertSlotShape(layers: 3, expertsPerLayer: 8, classes: classes, layerClasses: [0, 1, 2]),
        policy: LowestSlotPolicy(), observer: observer)
    func pattern(_ key: Int, _ i: Int) -> UInt8 { UInt8(truncatingIfNeeded: key &* 131 &+ i &* 7 &+ i >> 8) }
    func record(_ place: (storage: ExpertSlabStorage, offset: Int), _ ticket: ExpertSlotTicket) -> UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(
            start: place.storage.buffer.contents() + place.offset, count: classes[ticket.sizeClass].recordBytes)
    }
    // Metal hands back autoreleased references: drain them so a retired slab's charge goes at once.
    @discardableResult func fill(_ layer: Int, _ expert: Int) throws -> ExpertSlotTicket {
        try autoreleasepool { try fillBody(layer, expert) }
    }
    func fillBody(_ layer: Int, _ expert: Int) throws -> ExpertSlotTicket {
        let acquisition = try core.acquire(layer * 8 + expert)
        try require(acquisition.newlyReserved && acquisition.ticket.sizeClass == layer, "miss not reserved in its class")
        let bytes = record(try core.beginRead(acquisition.ticket), acquisition.ticket)
        for i in bytes.indices { bytes[i] = pattern(acquisition.ticket.key, i) }
        try require(core.completeRead(acquisition.ticket, verified: true), "read did not verify")
        return acquisition.ticket
    }
    func locked<T>(_ body: () throws -> T) rethrows -> T {
        core.lock.lock()
        defer { core.lock.unlock() }
        return try body()
    }
    func allocated(_ c: Int) -> UInt64 { locked { core.classes[c].allocated } }

    // Exact bytes in a slot of each class.
    var firsts: [ExpertSlotTicket] = []
    for layer in 0..<3 {
        let ticket = try fill(layer, 0)
        firsts.append(ticket)
        try autoreleasepool {
            let bytes = record(try core.pin(ticket), ticket)
            try require(bytes.indices.allSatisfy { bytes[$0] == pattern(ticket.key, $0) }, "class \(layer) bytes differ")
        }
        try core.unpin(ticket)
    }
    try require(observer.verified == 3, "observer missed a verification")

    // Class 0 (4 slots) overflows: every victim is a layer-0 record and no class outgrows its limit.
    var tickets = [Int: ExpertSlotTicket]()
    for expert in 1..<8 {
        tickets[expert] = try fill(0, expert)
        try require((0..<3).allSatisfy { allocated($0) <= classes[$0].byteLimit }, "a class exceeded its limit")
    }
    try require(observer.forgotten.count == 4 && observer.forgotten.allSatisfy { $0 / 8 == 0 }, "victims left class 0")
    try require(locked { core.lookup[8] != nil && core.lookup[16] != nil }, "other classes lost a record")
    try require(allocated(0) == 16_384 && allocated(1) == 12_288 && allocated(2) == 4_000, "class allocations wrong")
    try require(locked { core.evictions } == 4, "eviction count")

    // A stale ticket after its slot was recycled.
    try fill(1, 1)
    let first = firsts[1].slot
    let reused = try fill(1, 2)  // evicts the lowest evictable class-1 slot: (1, 0)'s
    try require(reused.slot == first, "class-1 victim was not the lowest slot")
    let rejectedBefore = locked { core.rejected }
    try mustFail { _ = try core.pin(firsts[1]) }
    try require(!core.completeRead(firsts[1], verified: true), "stale completion")
    try require(locked { core.rejected } == rejectedBefore + 1, "stale completion not counted")

    // Cancel only from reserved.
    let reserved = try core.acquire(1 * 8 + 3).ticket
    try core.cancel(reserved)
    try mustFail { try core.cancel(reserved) }
    let reading = try core.acquire(1 * 8 + 3).ticket
    _ = try core.beginRead(reading)
    try mustFail { try core.cancel(reading) }
    try require(!core.completeRead(reading, verified: false), "unverified read published")
    let done = try fill(1, 3)
    try mustFail { try core.cancel(done) }

    // Pins and the observer's hold block eviction.
    let resident = locked { core.lookup.indices.filter { $0 / 8 == 1 && core.lookup[$0] != nil } }
    let pinned = try resident.map { try core.acquire($0).ticket }
    try autoreleasepool { for ticket in pinned { _ = try core.pin(ticket) } }
    try mustFail { _ = try core.acquire(1 * 8 + 5) }
    try core.unpin(pinned[0])
    try require(try fill(1, 5).slot == pinned[0].slot, "unpinned slot not taken")
    for ticket in pinned.dropFirst() { try core.unpin(ticket) }
    _ = try (1..<4).map { try fill(2, $0) }
    observer.held = Set((0..<4).map { [2, $0] })
    try mustFail { _ = try core.acquire(2 * 8 + 4) }
    observer.held.remove([2, 2])
    try require(try fill(2, 4).slot == 2, "the only unheld class-2 slot was not taken")
    observer.held = []

    // Retiring: a slab with a pin stays; an idle one goes and returns its ledger bytes; holes are reused lowest first.
    let keep = try core.acquire(0 * 8 + 7).ticket
    try autoreleasepool { _ = try core.pin(keep) }
    let charged = ledger.reservedBytes
    var retired: [ExpertSlabStorage] = []
    locked {
        core.classes[0].byteLimit = 0
        core.retireSlabsLocked(sizeClass: 0, maximumSlabs: .max, retired: &retired)
    }
    try require(retired.count == 1 && allocated(0) == 8_192 && locked { core.classes[0].pendingTrim }, "pinned slab retired")
    retired = []
    try require(ledger.reservedBytes == charged - 8_192 && observer.retiredSlabs == 1, "retired slab kept its charge")
    try core.unpin(keep)
    locked { core.retireSlabsLocked(sizeClass: 0, maximumSlabs: .max, retired: &retired) }
    retired = []
    try require(allocated(0) == 0 && ledger.reservedBytes == charged - 16_384, "idle slab not retired")
    try mustFail { _ = try core.pin(keep) }
    locked { core.classes[0].byteLimit = 16_384 }
    try require(try fill(0, 1).slot == 0, "holes not reused lowest first")
    try require(allocated(1) == 24_576 && allocated(2) == 4_000, "retiring class 0 touched another class")
}
