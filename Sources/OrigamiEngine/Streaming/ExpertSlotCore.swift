import CExactIO
import Foundation
import Metal
import ObjectiveC

package final class ExpertSlabReservation {
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    package init(ledger: MemoryLedger, reservation: MemoryLedger.Reservation) {
        self.ledger = ledger
        self.reservation = reservation
    }
    deinit { try? ledger.release(reservation) }
}
package enum AllocationAssociation { nonisolated(unsafe) package static var key: UInt8 = 0 }
package final class ExpertSlabStorage {
    package let buffer: MTLBuffer
    /// Pages user-wired by `wire_decode_slabs` (mutated under the cache lock). Unwired when the
    /// last cache-side owner drops, before the storage's own retain of the Metal resource ends.
    package var wiredBytes = 0
    /// The owner's own slab object (`init(buffer:retaining:)`), released with this storage.
    private let owner: AnyObject?
    deinit {
        if wiredBytes > 0 { _ = origami_unwire_pages(buffer.contents(), wiredBytes) }
    }
    package init(buffer: MTLBuffer, ledger: MemoryLedger, reservation: MemoryLedger.Reservation) {
        self.buffer = buffer
        owner = nil
        // The charge follows the Objective-C Metal resource itself, including a
        // raw MTLBuffer retained by a command encoder or another caller.
        objc_setAssociatedObject(
            buffer, &AllocationAssociation.key,
            ExpertSlabReservation(ledger: ledger, reservation: reservation), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
    /// A slab its owner charged and wired (`owner` carries the reservation and unwires when it goes).
    package init(buffer: MTLBuffer, retaining owner: AnyObject) {
        self.buffer = buffer
        self.owner = owner
    }
}

/// Slot life cycle: free -> reserved -> reading (-> free when not verified) -> verified -> pinned <-> evictable.
public enum ExpertSlotPhase: String, Codable, Sendable {
    case free, reserved, reading, verified, pinned, evictable
}

/// A generation ticket cannot publish into a recycled slot. `key` is the flat key, `layer * expertsPerLayer + expert`;
/// `slot` indexes the key's size class.
package struct ExpertSlotTicket: Hashable, Sendable {
    package let sizeClass, slot: Int
    package let generation: UInt64
    package let key: Int
}

/// The result of a lookup: the record's ticket, its phase, and whether this call reserved it (a miss to read).
package struct ExpertSlotAcquisition {
    package let ticket: ExpertSlotTicket
    package let phase: ExpertSlotPhase
    package let newlyReserved: Bool
}

/// One record size: its slabs, slots, free counts, hole heap and limits. Read-only outside the core, except the limit and
/// the pending-trim mark, which the owner's pacing sets under the core lock. This is the view a policy reads.
package final class ExpertSlotClass {
    package struct Slot {
        package let slab: Int
        package let offset: Int
        package fileprivate(set) var key: Int?
        package fileprivate(set) var generation: UInt64 = 0
        package fileprivate(set) var phase: ExpertSlotPhase = .free
        package fileprivate(set) var pins = 0
    }
    package final class Slab {
        package let storage: ExpertSlabStorage
        package var buffer: MTLBuffer { storage.buffer }
        package let slots: [Int]
        init(storage: ExpertSlabStorage, slots: [Int]) {
            self.storage = storage
            self.slots = slots
        }
    }
    package let recordBytes, slotsPerSlab: Int
    package let maximumByteLimit: UInt64
    /// Records allocated below this count ignore the ledger's optional-growth gate (a compulsory streaming cache).
    package let mandatorySlots: Int
    /// The free counter `freeRegular` counts slots at or above this index (a policy's reserved low slots below it).
    package let regularFrom: Int
    let category: String
    let ownerSized: Bool
    package var byteLimit: UInt64
    package var pendingTrim = false
    package fileprivate(set) var slabs: [Int: Slab] = [:]
    // Unchecked (like `lookup`): only touched under the core lock, in single statements; the runtime exclusivity check on
    // every transition cost more than the cache this core replaced spent per call.
    @exclusivity(unchecked) package fileprivate(set) var slots: [Slot?] = []
    /// Minimum heap of holes from retired slabs. Reuse preserves the old lowest-index
    /// order without rescanning all historical slot metadata for every recovered slab.
    private var recycledSlots: [Int] = []
    package fileprivate(set) var allocated: UInt64 = 0
    /// Free slots at any index and at or above `regularFrom`, so a full cache skips the free-slot scans.
    private var freeAll = 0, freeRegular = 0
    package var slabBytes: UInt64 { UInt64(recordBytes * slotsPerSlab) }

    fileprivate init(_ shape: ExpertSlotShape.SizeClass) {
        recordBytes = shape.recordBytes
        slotsPerSlab = shape.slotsPerSlab
        byteLimit = shape.byteLimit
        maximumByteLimit = shape.maximumByteLimit
        mandatorySlots = shape.mandatorySlots
        regularFrom = shape.regularFrom
        category = shape.category
        ownerSized = shape.ownerSized
    }

    fileprivate func recycleSlot(_ index: Int) {
        recycledSlots.append(index)
        var child = recycledSlots.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard recycledSlots[child] < recycledSlots[parent] else { break }
            recycledSlots.swapAt(child, parent)
            child = parent
        }
    }
    fileprivate func takeRecycledSlot() -> Int? {
        guard let first = recycledSlots.first else { return nil }
        let last = recycledSlots.removeLast()
        guard !recycledSlots.isEmpty else { return first }
        recycledSlots[0] = last
        var parent = 0
        while parent * 2 + 1 < recycledSlots.count {
            let left = parent * 2 + 1
            let right = left + 1
            let child = right < recycledSlots.count && recycledSlots[right] < recycledSlots[left] ? right : left
            guard recycledSlots[child] < recycledSlots[parent] else { break }
            recycledSlots.swapAt(child, parent)
            parent = child
        }
        return first
    }

    /// The lowest free slot at or above `start` (0, or `regularFrom`); none without a scan when the
    /// matching counter is zero, which is every miss once the cache is full.
    package func firstFree(from start: Int) -> Int? {
        assert(start == 0 || start == regularFrom)
        assert(
            (start == 0 ? freeAll : freeRegular) == slots.indices.filter { $0 >= start && slots[$0]?.phase == .free }.count,
            "Free-slot counter drifted")
        guard (start == 0 ? freeAll : freeRegular) > 0 else { return nil }
        return slots.indices.first(where: { $0 >= start && slots[$0]?.phase == .free })
    }
    fileprivate func countFree(_ index: Int, _ change: Int) {
        freeAll += change
        if index >= regularFrom { freeRegular += change }
    }
    /// Every phase change into or out of `.free` goes through here, so the free counters stay exact.
    fileprivate func setPhase(_ index: Int, _ phase: ExpertSlotPhase) {
        countFree(index, (phase == .free ? 1 : 0) - (slots[index]!.phase == .free ? 1 : 0))
        slots[index]!.phase = phase
    }
    // The transitions' slot reads and writes, as methods on `self`: the same code through another object's property
    // retains the slot array on every call.
    fileprivate func checked(_ ticket: ExpertSlotTicket) throws -> Slot {
        guard slots.indices.contains(ticket.slot), let slot = slots[ticket.slot],
            slot.generation == ticket.generation, slot.key == ticket.key
        else { throw EngineError("Stale expert generation") }
        return slot
    }
    fileprivate func reserve(_ index: Int, key: Int, generation: UInt64) {
        slots[index]!.key = key
        slots[index]!.generation = generation
        setPhase(index, .reserved)
        slots[index]!.pins = 0
    }
    fileprivate func clear(_ index: Int) {
        slots[index]!.key = nil
        setPhase(index, .free)
    }
    /// A change between phases other than `.free`.
    fileprivate func move(_ index: Int, to phase: ExpertSlotPhase) { slots[index]!.phase = phase }
    fileprivate func addPin(_ index: Int) {
        slots[index]!.pins += 1
        slots[index]!.phase = .pinned
    }
    fileprivate func removePin(_ index: Int) -> Int {
        slots[index]!.pins -= 1
        return slots[index]!.pins
    }

    /// Every slot is free, verified or evictable and unpinned: no reader, writer or consumer holds one.
    package var quiescent: Bool {
        slots.allSatisfy { entry in
            guard let slot = entry else { return true }
            return [.free, .verified, .evictable].contains(slot.phase) && slot.pins == 0
        }
    }
}

/// The keys and record sizes a core serves: `layers x expertsPerLayer` keys, each layer's records in one size class.
package struct ExpertSlotShape {
    package struct SizeClass {
        package var recordBytes, slotsPerSlab: Int
        package var byteLimit, maximumByteLimit: UInt64
        package var mandatorySlots = 0, regularFrom = 0
        package var category: String
        /// The owner inserts its own slabs of up to `slotsPerSlab` slots (`insertSlabLocked(slots:)`) and keeps its own
        /// limits: only `retireSlabsLocked` reads `byteLimit`, and the limits need not be whole slabs.
        package var ownerSized = false
        package init(
            recordBytes: Int, slotsPerSlab: Int, byteLimit: UInt64, maximumByteLimit: UInt64, mandatorySlots: Int = 0,
            regularFrom: Int = 0, category: String, ownerSized: Bool = false
        ) {
            self.recordBytes = recordBytes
            self.slotsPerSlab = slotsPerSlab
            self.byteLimit = byteLimit
            self.maximumByteLimit = maximumByteLimit
            self.mandatorySlots = mandatorySlots
            self.regularFrom = regularFrom
            self.category = category
            self.ownerSized = ownerSized
        }
    }
    package var layers, expertsPerLayer: Int
    package var classes: [SizeClass]
    /// The size class of each layer.
    package var layerClasses: [Int]
    package init(layers: Int, expertsPerLayer: Int, classes: [SizeClass], layerClasses: [Int]) {
        self.layers = layers
        self.expertsPerLayer = expertsPerLayer
        self.classes = classes
        self.layerClasses = layerClasses
    }
}

/// Chooses victims and keeps its own per-slot and per-key bookkeeping (reference bits, scan flags, scores, bypass
/// slots). The core calls every requirement under its lock, so a policy needs no lock of its own and must not call
/// back into a locking core method. A conforming type is a final class; the core is generic over it (no boxing).
package protocol ExpertSlotPolicy: AnyObject {
    associatedtype Observer: ExpertSlotObserver
    /// A slot of `sizeClass` for `key`'s miss, or nil when none may be taken: a free slot (`firstFree`), a slot of a
    /// slab it allocates (`allocateSlabLocked`), or a slot it evicted with `evictLocked` (one `isEvictableLocked` allows).
    func victim(for key: Int, sizeClass: Int, in core: ExpertSlotCore<Self>) throws -> Int?
    /// A lookup found `slot` (any phase).
    func didHit(_ slot: Int, sizeClass: Int)
    /// A demand miss reserved `slot` for its key.
    func didReserve(_ slot: Int, sizeClass: Int)
    /// `slot` became verified, pinned (each pin) or evictable (its last pin retired).
    func didChange(_ slot: Int, key: Int, sizeClass: Int, to phase: ExpertSlotPhase)
    /// Whether `slot` goes straight back to free when its last pin retires (instead of becoming evictable).
    func releasesOnLastUnpin(_ slot: Int, sizeClass: Int) -> Bool
    /// `key` left the lookup (evicted, freed, retired or moved over).
    func didForget(_ key: Int)
    /// New free slots (new indices or recycled holes) of a slab just added.
    func didInsertSlab(_ slots: [Int], sizeClass: Int)
    /// Orders idle slab ids, first retired first.
    func orderRetirement(_ slabs: inout [Int], sizeClass: Int, in core: ExpertSlotCore<Self>)
}

/// Publication hooks (for example a GPU residency table), called under the core lock.
package protocol ExpertSlotObserver: AnyObject {
    /// A veto: a held slot is never evicted, and its slab is never retired.
    func isHeld(_ slot: Int, key: Int?, sizeClass: Int) -> Bool
    /// `slot` holds a complete verified record.
    func didVerify(_ slot: Int, sizeClass: Int, in slots: ExpertSlotClass)
    /// `key` is about to leave the lookup; `slot` is where the lookup still points.
    func willForget(_ key: Int, slot: Int?, sizeClass: Int)
    func didInsertSlab(_ id: Int, sizeClass: Int, in slots: ExpertSlotClass)
    func didRetireSlab(_ id: Int, sizeClass: Int)
}
extension ExpertSlotObserver {
    package func didRetireSlab(_ id: Int, sizeClass: Int) {}
}

/// The model-neutral expert slot cache: a dense `layers x expertsPerLayer` lookup over slots carved from shared Metal
/// slabs, one set of slabs per record size class. Slabs are allocated lazily up to each class's `byteLimit` and
/// released only whole and cold. Every mutation runs under `lock`; tickets carry a generation, so a producer can never
/// publish into a recycled slot. Victims come from `Policy`; publication goes to `Policy.Observer`. Methods named
/// `...Locked` (and every `package` var) need the caller to hold `lock`.
package final class ExpertSlotCore<Policy: ExpertSlotPolicy> {
    package let lock = NSLock()
    package let policy: Policy
    package let observer: Policy.Observer
    package let layers, expertsPerLayer: Int
    package let classes: [ExpertSlotClass]
    /// `classes[0]` as a stored constant: reading it costs no retain, unlike an array element (every transition reads one).
    private let firstClass: ExpertSlotClass
    private let layerClasses: [Int]
    private let device: MTLDevice
    private let ledger: MemoryLedger
    /// Key -> slot in the key's size class. Unchecked exclusivity, as `ExpertSlotClass.slots`.
    @exclusivity(unchecked) package private(set) var lookup: [Int?]
    private var nextSlab = 0
    package private(set) var nextGeneration: UInt64 = 0
    /// No slab is added while held (queued commands hold the slab layout).
    package var slabGrowthHeld = false
    /// Moves whenever a slab comes or goes.
    package private(set) var slabGeneration: UInt64 = 0
    package private(set) var hits: UInt64 = 0, misses: UInt64 = 0, evictions: UInt64 = 0, rejected: UInt64 = 0

    package init(device: MTLDevice, ledger: MemoryLedger, shape: ExpertSlotShape, policy: Policy, observer: Policy.Observer) throws {
        guard shape.layers > 0, shape.expertsPerLayer > 0, !shape.classes.isEmpty, shape.layerClasses.count == shape.layers,
            shape.layerClasses.allSatisfy({ shape.classes.indices.contains($0) })
        else { throw EngineError("Invalid expert slot shape", code: "invalid_configuration") }
        for (c, size) in shape.classes.enumerated() {
            if size.ownerSized {
                guard size.recordBytes > 0, size.slotsPerSlab > 0 else {
                    throw EngineError("Invalid expert slab geometry", code: "invalid_configuration")
                }
                continue
            }
            guard size.recordBytes > 0, (1...shape.expertsPerLayer).contains(size.slotsPerSlab) else {
                throw EngineError("Invalid expert slab geometry", code: "invalid_configuration")
            }
            let slabBytes = UInt64(size.recordBytes * size.slotsPerSlab)
            let keys = shape.layerClasses.filter { $0 == c }.count * shape.expertsPerLayer
            guard size.byteLimit >= slabBytes, size.byteLimit % slabBytes == 0,
                size.maximumByteLimit >= size.byteLimit, size.maximumByteLimit % slabBytes == 0,
                size.maximumByteLimit <= UInt64(keys * size.recordBytes)
            else {
                throw EngineError("Expert cache limits must fit bounded whole slabs", code: "invalid_configuration")
            }
        }
        self.device = device
        self.ledger = ledger
        self.policy = policy
        self.observer = observer
        layers = shape.layers
        expertsPerLayer = shape.expertsPerLayer
        classes = shape.classes.map(ExpertSlotClass.init)
        firstClass = classes[0]
        layerClasses = shape.layerClasses
        lookup = [Int?](repeating: nil, count: shape.layers * shape.expertsPerLayer)
    }

    @inline(__always) private func sizeClassState(_ c: Int) -> ExpertSlotClass { c == 0 ? firstClass : classes[c] }
    @inline(__always) package func sizeClass(ofKey key: Int) -> Int {
        classes.count == 1 ? 0 : layerClasses[key / expertsPerLayer]
    }

    package func acquire(_ key: Int) throws -> ExpertSlotAcquisition {
        lock.lock()
        defer { lock.unlock() }
        let c = sizeClass(ofKey: key)
        let cls = sizeClassState(c)
        if let index = lookup[key], let slot = cls.slots[index] {
            policy.didHit(index, sizeClass: c)
            if [.verified, .pinned, .evictable].contains(slot.phase) { hits += 1 }
            return ExpertSlotAcquisition(
                ticket: ExpertSlotTicket(sizeClass: c, slot: index, generation: slot.generation, key: key), phase: slot.phase,
                newlyReserved: false)
        }
        guard nextGeneration < UInt64.max else { throw EngineError("Cache generation exhausted") }
        guard let index = try policy.victim(for: key, sizeClass: c, in: self) else {
            throw EngineError("All expert slots are reading, reserved or GPU-pinned", code: "expert_slots_busy")
        }
        let ticket = reserveLocked(index, key: key)
        policy.didReserve(index, sizeClass: c)
        return ExpertSlotAcquisition(ticket: ticket, phase: .reserved, newlyReserved: true)
    }

    /// Makes `index` (a slot a policy chose: free, or evicted with `evictLocked`) the reserved home of `key`, under a new
    /// generation. The caller has checked `nextGeneration < UInt64.max`.
    package func reserveLocked(_ index: Int, key: Int) -> ExpertSlotTicket {
        let c = sizeClass(ofKey: key)
        let cls = sizeClassState(c)
        assert(
            [.free, .verified, .evictable].contains(cls.slots[index]!.phase) && cls.slots[index]!.pins == 0
                && (cls.slots[index]!.key.map { lookup[$0] != index } ?? true), "Policy returned a live slot")
        nextGeneration += 1
        cls.reserve(index, key: key, generation: nextGeneration)
        lookup[key] = index
        misses += 1
        return ExpertSlotTicket(sizeClass: c, slot: index, generation: nextGeneration, key: key)
    }

    private func checked(_ ticket: ExpertSlotTicket) throws -> ExpertSlotClass.Slot {
        try sizeClassState(ticket.sizeClass).checked(ticket)
    }

    /// Exactly one read producer can transition a reservation into reading.
    package func beginRead(_ ticket: ExpertSlotTicket) throws -> (storage: ExpertSlabStorage, offset: Int) {
        lock.lock()
        defer { lock.unlock() }
        return try beginReadLocked(ticket)
    }
    /// `beginRead` for a caller that holds `lock`.
    @inline(__always) package func beginReadLocked(_ ticket: ExpertSlotTicket) throws -> (storage: ExpertSlabStorage, offset: Int) {
        let slot = try checked(ticket)
        let cls = sizeClassState(ticket.sizeClass)
        guard slot.phase == .reserved, let slab = cls.slabs[slot.slab] else { throw EngineError("Expert slot is not reserved") }
        cls.move(ticket.slot, to: .reading)
        return (slab.storage, slot.offset)
    }

    /// Must be called only after every writer for the ticket has stopped touching
    /// its destination. Hedge reads need separate staging; the winner copies here.
    @discardableResult package func completeRead(_ ticket: ExpertSlotTicket, verified: Bool) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return completeReadLocked(ticket, verified: verified)
    }
    /// `completeRead` for a caller that holds `lock`.
    @discardableResult @inline(__always) package func completeReadLocked(_ ticket: ExpertSlotTicket, verified: Bool) -> Bool {
        guard let slot = try? checked(ticket), slot.phase == .reading else {
            rejected += 1
            return false
        }
        if !verified {
            freeLocked(ticket.slot, sizeClass: ticket.sizeClass)
            return false
        }
        let cls = sizeClassState(ticket.sizeClass)
        cls.move(ticket.slot, to: .verified)
        policy.didChange(ticket.slot, key: ticket.key, sizeClass: ticket.sizeClass, to: .verified)
        observer.didVerify(ticket.slot, sizeClass: ticket.sizeClass, in: cls)
        return true
    }

    package func cancel(_ ticket: ExpertSlotTicket) throws {
        lock.lock()
        defer { lock.unlock() }
        let slot = try checked(ticket)
        switch slot.phase {
        case .reserved: freeLocked(ticket.slot, sizeClass: ticket.sizeClass)
        default: throw EngineError("Only a reserved cache entry can be cancelled; a read ends with completeRead")
        }
    }

    package func pin(_ ticket: ExpertSlotTicket) throws -> (storage: ExpertSlabStorage, offset: Int) {
        lock.lock()
        defer { lock.unlock() }
        return try pinLocked(ticket)
    }
    /// `pin` for a caller that holds `lock` (one lock for several transitions).
    @inline(__always) package func pinLocked(_ ticket: ExpertSlotTicket) throws -> (storage: ExpertSlabStorage, offset: Int) {
        let slot = try checked(ticket)
        let cls = sizeClassState(ticket.sizeClass)
        guard [.verified, .evictable, .pinned].contains(slot.phase), let slab = cls.slabs[slot.slab] else {
            throw EngineError("Incomplete expert cannot be read by GPU")
        }
        cls.addPin(ticket.slot)
        policy.didChange(ticket.slot, key: ticket.key, sizeClass: ticket.sizeClass, to: .pinned)
        return (slab.storage, slot.offset)
    }

    /// Call from command-buffer completion, never immediately after encoding.
    package func unpin(_ ticket: ExpertSlotTicket) throws {
        lock.lock()
        defer { lock.unlock() }
        try unpinLocked(ticket)
    }
    /// `unpin` for a caller that holds `lock`.
    @inline(__always) package func unpinLocked(_ ticket: ExpertSlotTicket) throws {
        let slot = try checked(ticket)
        guard slot.phase == .pinned, slot.pins > 0 else { throw EngineError("Expert has no GPU pin to retire") }
        let cls = sizeClassState(ticket.sizeClass)
        if cls.removePin(ticket.slot) == 0 {
            if policy.releasesOnLastUnpin(ticket.slot, sizeClass: ticket.sizeClass) {
                freeLocked(ticket.slot, sizeClass: ticket.sizeClass)
            } else {
                cls.move(ticket.slot, to: .evictable)
                policy.didChange(ticket.slot, key: ticket.key, sizeClass: ticket.sizeClass, to: .evictable)
            }
        }
    }

    /// Drops a key from the lookup and, first, from the observer.
    package func forgetLocked(_ key: Int) {
        observer.willForget(key, slot: lookup[key], sizeClass: sizeClass(ofKey: key))
        lookup[key] = nil
        policy.didForget(key)
    }
    /// Evicts the record in `index` (a policy's victim): its key leaves the lookup and counts as an eviction.
    package func evictLocked(_ index: Int, sizeClass c: Int) {
        assert(isEvictableLocked(index, sizeClass: c), "Policy evicted a busy or held slot")
        if let key = classes[c].slots[index]!.key { forgetLocked(key) }
        evictions += 1
    }
    /// Complete, unpinned and not held by the observer: a slot a policy may evict.
    package func isEvictableLocked(_ index: Int, sizeClass c: Int) -> Bool {
        guard let slot = classes[c].slots[index] else { return false }
        return [.verified, .evictable].contains(slot.phase) && slot.pins == 0
            && !observer.isHeld(index, key: slot.key, sizeClass: c)
    }
    /// Returns `index` to free, its key (if any) forgotten.
    package func freeLocked(_ index: Int, sizeClass c: Int) {
        let cls = sizeClassState(c)
        // A key that moved to another slot (a stalled read's slot is anonymous) stays where it is.
        if let key = cls.slots[index]!.key, lookup[key] ?? index == index { forgetLocked(key) }
        cls.clear(index)
    }

    /// Allocates and charges one slab of class `c` while it fits the class limit (and, above the class's mandatory
    /// records, the ledger's optional growth); returns its first slot, or nil.
    package func allocateSlabLocked(sizeClass c: Int) throws -> Int? {
        let cls = sizeClassState(c)
        let bytes = cls.slabBytes
        guard !slabGrowthHeld, cls.allocated <= cls.byteLimit, bytes <= cls.byteLimit - cls.allocated else { return nil }
        let mandatory = min(cls.byteLimit, (UInt64(cls.mandatorySlots * cls.recordBytes) + bytes - 1) / bytes * bytes)
        if cls.allocated >= mandatory {
            guard ledger.optionalGrowthAllowed else { return nil }
            let reserved = ledger.reservedBytes
            let usable = ledger.usableBytes
            guard reserved <= usable, bytes <= usable - reserved else { return nil }
        }
        guard let reservation = try ledger.reservePhysicalIfAvailable(bytes: bytes, category: cls.category) else { return nil }
        // Any thread can grow the cache (the shared-event listener too): never leave a slab in a pool that a trim cannot drain.
        guard let buffer = autoreleasepool(invoking: { device.makeBuffer(length: Int(bytes), options: .storageModeShared) }) else {
            try ledger.release(reservation)
            throw EngineError("Metal expert slab allocation failed")
        }
        return insertSlabLocked(ExpertSlabStorage(buffer: buffer, ledger: ledger, reservation: reservation), sizeClass: c)
    }
    /// Adds a charged slab (allocated here or prepared by the owner) of `count` slots (`slotsPerSlab`; fewer only in an
    /// `ownerSized` class) to class `c`; its slots start free. Returns its first slot.
    @discardableResult package func insertSlabLocked(_ storage: ExpertSlabStorage, sizeClass c: Int, slots count: Int? = nil) -> Int {
        let cls = sizeClassState(c)
        let count = count ?? cls.slotsPerSlab
        assert(count == cls.slotsPerSlab || (cls.ownerSized && (1...cls.slotsPerSlab).contains(count)), "Invalid slab slot count")
        let id = nextSlab
        nextSlab += 1
        // Reuse holes from released slabs to bound host metadata under repeated resizing.
        var indices: [Int] = []
        while indices.count < count, let index = cls.takeRecycledSlot() { indices.append(index) }
        while indices.count < count {
            indices.append(cls.slots.count)
            cls.slots.append(nil)
        }
        for (offset, index) in indices.enumerated() {
            cls.slots[index] = ExpertSlotClass.Slot(slab: id, offset: offset * cls.recordBytes)
            cls.countFree(index, 1)
        }
        cls.slabs[id] = ExpertSlotClass.Slab(storage: storage, slots: indices)
        cls.allocated += UInt64(storage.buffer.length)
        slabGeneration &+= 1
        policy.didInsertSlab(indices, sizeClass: c)
        observer.didInsertSlab(id, sizeClass: c, in: cls)
        return indices[0]
    }

    /// Releases whole idle slabs of class `c` (every slot free, verified or evictable, unpinned and not held), in the
    /// policy's order, until the class fits its limit or `maximumSlabs` went. The storages go to `retired`, so the
    /// caller drops them (Metal destruction, unwiring and the ledger release) after unlocking. Sets `pendingTrim`
    /// while the class is still over its limit.
    package func retireSlabsLocked(sizeClass c: Int, maximumSlabs: Int, retired: inout [ExpertSlabStorage]) {
        let cls = sizeClassState(c)
        // Lazy growth changes only the limit and avoids sorting every slab.
        guard cls.allocated > cls.byteLimit else {
            cls.pendingTrim = false
            return
        }
        var candidates: [Int] = []
        for (id, slab) in cls.slabs {
            guard
                slab.slots.allSatisfy({ index in
                    guard let slot = cls.slots[index] else { return true }
                    return [.free, .verified, .evictable].contains(slot.phase) && slot.pins == 0
                        && !observer.isHeld(index, key: slot.key, sizeClass: c)
                })
            else { continue }
            candidates.append(id)
        }
        policy.orderRetirement(&candidates, sizeClass: c, in: self)
        var chosen: [Int] = []
        var remaining = cls.allocated
        for id in candidates where remaining > cls.byteLimit && chosen.count < maximumSlabs {
            chosen.append(id)
            remaining -= UInt64(cls.slabs[id]!.buffer.length)
        }
        for id in chosen {
            let slab = cls.slabs[id]!
            for index in slab.slots {
                if let key = cls.slots[index]?.key { forgetLocked(key) }
                if cls.slots[index]?.phase == .free { cls.countFree(index, -1) }
                cls.slots[index] = nil
                cls.recycleSlot(index)
            }
            cls.allocated -= UInt64(slab.buffer.length)
            // Hold the storage through unlock: munlock/Metal destruction and the final
            // reservation release must not stall readers waiting for the cache mutex.
            retired.append(slab.storage)
            cls.slabs.removeValue(forKey: id)
            slabGeneration &+= 1
            observer.didRetireSlab(id, sizeClass: c)
        }
        cls.pendingTrim = cls.allocated > cls.byteLimit
    }
}
