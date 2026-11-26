import CExactIO
import Darwin
import Foundation
import Metal

/// One routed expert of one layer (48 layers x 512 experts), flattened to `layer * 512 + expert` for the cache
/// lookup and the GPU residency table.
public struct ExpertKey: Hashable, Sendable {
    public let layer, expert: Int
    public init(layer: Int, expert: Int) throws {
        guard (0..<48).contains(layer), (0..<512).contains(expert) else { throw EngineError("Invalid expert key") }
        self.layer = layer
        self.expert = expert
    }
    fileprivate var index: Int { layer * 512 + expert }
}

/// A generation ticket cannot publish into a recycled slot. A cancelled read
/// retains its slab until its producer explicitly reports completion.
public struct ExpertTicket: Hashable, Sendable {
    fileprivate let sizeClass, slot: Int
    public let generation: UInt64
    public let key: ExpertKey
    /// The slot core's ticket.
    fileprivate var core: ExpertSlotTicket { ExpertSlotTicket(sizeClass: sizeClass, slot: slot, generation: generation, key: key.index) }
}

/// A view of one complete expert record inside a cache slab or the replica arena: `PackStore.expertBytes`, or a cache
/// slot of a smaller size class (`ExpertCache.SizeClasses`), which holds every record of its layers.
/// Holding it keeps the Metal buffer and its ledger charge alive, but it does not pin the slot: only a cache pin
/// (or a LoadedExpert lease) keeps the record from being recycled.
public final class ExpertBuffer: @unchecked Sendable {
    private let storage: ExpertSlabStorage?
    public let buffer: MTLBuffer
    public let offset, length: Int
    fileprivate init(storage: ExpertSlabStorage, offset: Int, length: Int) {
        self.storage = storage
        self.buffer = storage.buffer
        self.offset = offset
        self.length = length
    }
    /// The replica arena's Metal resource independently carries its reservation.
    /// The LoadedExpert consumer pin protects this particular record from reuse.
    package init(replicaBuffer: MTLBuffer, offset: Int) {
        precondition(offset >= 0 && offset <= replicaBuffer.length - PackStore.expertBytes)
        storage = nil
        buffer = replicaBuffer
        self.offset = offset
        length = PackStore.expertBytes
    }
    public var bytes: UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: buffer.contents().advanced(by: offset), count: length)
    }
}

/// The expert record cache: a fixed 48 x 512 lookup over slots carved from shared Metal slabs of `slotsPerSlab`
/// records. Slabs are allocated lazily up to `byteLimit` and released only whole and cold. Eviction is CLOCK (in prefill
/// after the nearest passed scan record), or the reuse-score policy (`ExpertReusePolicy`) in decode with reuse admission.
/// Every mutation runs under `lock`; tickets carry a generation, so a producer can never publish into a recycled slot.
/// Only complete verified records may be pinned for the GPU or published in the GPU residency table.
/// The slots, tickets and slab mechanics are the model-neutral `ExpertSlotCore`; the victims are
/// `ExpertCacheEviction`'s, the GPU table `ExpertGPUResidency`'s; this type keeps the API and the pacing of resizes,
/// trims and growth. One size class (`PackStore.expertBytes`), except for a GGUF whose layers' records differ in size:
/// up to three classes (`SizeClasses`), the byte limit split between them by their layers' bytes, so each layer keeps the
/// same share of records as with one class while the padding of smaller layers' slots holds more records.
public final class ExpertCache: @unchecked Sendable {
    /// Record size classes: each class's slot bytes, and each layer's class.
    public struct SizeClasses: Equatable, Sendable {
        public let recordBytes, layerClasses: [Int]
        public init(recordBytes: [Int], layerClasses: [Int]) {
            self.recordBytes = recordBytes
            self.layerClasses = layerClasses
        }
        /// One class of `PackStore.expertBytes` (the layout's, read when asked).
        public static var one: SizeClasses {
            SizeClasses(recordBytes: [PackStore.expertBytes], layerClasses: Array(repeating: 0, count: 48))
        }
        /// Each class's share of a byte limit: its layers' count times its slot bytes.
        var weights: [UInt64] {
            recordBytes.indices.map { c in UInt64(layerClasses.filter { $0 == c }.count * recordBytes[c]) }
        }
        /// For records of `layerBytes` (per layer, each at most the largest): at most three classes, the grouping of the
        /// sorted distinct sizes that leaves the least padding (each class is its largest member's size). One class when
        /// the records are one size.
        public static func grouping(_ layerBytes: [Int]) -> SizeClasses {
            let sizes = Set(layerBytes).sorted()
            guard layerBytes.count == 48, sizes.count > 1, let largest = sizes.last else { return .one }
            let count = sizes.map { size in layerBytes.filter { $0 == size }.count }
            var best: (padded: Int, ends: [Int]) = (.max, [sizes.count])
            // Class ends (exclusive indices into `sizes`), the last always the largest size.
            for first in 1...sizes.count {
                for second in first...sizes.count {
                    let ends = Array(Set([first, second, sizes.count])).sorted()
                    var start = 0
                    var padded = 0
                    for end in ends {
                        padded += sizes[end - 1] * count[start..<end].reduce(0, +)
                        start = end
                    }
                    if padded < best.padded || padded == best.padded && ends.count < best.ends.count { best = (padded, ends) }
                }
            }
            let bytes = best.ends.map { sizes[$0 - 1] }
            assert(bytes.last == largest)
            return SizeClasses(recordBytes: bytes, layerClasses: layerBytes.map { size in bytes.firstIndex { $0 >= size }! })
        }
    }
    /// Slot life cycle: free -> reserved -> reading (-> free when not verified) -> verified -> pinned <-> evictable.
    public typealias Phase = ExpertSlotPhase
    /// The result of a lookup: the record's ticket, its phase, and whether this call reserved it (a miss to read).
    public struct Acquisition: Sendable {
        public let ticket: ExpertTicket
        public let phase: Phase
        public let newlyReserved: Bool
    }
    /// Cache counters and gauges (one locked scan of the slots); safe from any thread.
    public struct Snapshot: Codable, Sendable {
        /// Allocated bytes belong to cache-owned slabs. Escaped raw resources
        /// remain charged in MemoryLedger even after a slab leaves this cache.
        public let allocatedBytes, byteLimit, maximumByteLimit: UInt64
        public let slabs, activeSlots, reservedSlots, pinnedSlots, inFlightReads, metadataSlots: Int
        public let hits, misses, evictions, rejectedCompletions: UInt64
        public let reuseRows, reuseBypasses, reuseAdmissions, reuseDecisionNanoseconds, reuseReplayResets, reuseBorrowedBypasses: UInt64?
        public let reuseTransientSlots: Int?
        /// Decode demand misses on layer 0, which has no queued guess (each one stalls a token).
        public var layerZeroMisses: UInt64 = 0
        /// With several record size classes: each class's slot bytes and limit (nil with one).
        public struct SizeClass: Codable, Sendable {
            public let recordBytes: Int
            public let byteLimit: UInt64
        }
        public var sizeClasses: [SizeClass]? = nil
    }
    /// The receipt of `prepareSlabs`: what was allocated, touched or wired, and the host memory seen meanwhile.
    public struct PhysicalTouch: Codable, Sendable {
        public let targetBytes, allocatedBytes, coveredBytes, pagesTouched: UInt64
        public let slabsTouched, hostPageBytes: Int
        public let physicalFootprintHighWater, minimumAvailableMemoryEstimate: UInt64
        public let memory: MemoryLedger.Snapshot
        /// `wire_decode_slabs` only (nil otherwise): slabs and pages newly user-wired by this pass,
        /// slabs already wired by an earlier pass, and slabs whose wire failed and fell back to the
        /// preserving touch (must be zero).
        public var slabsWired: UInt64? = nil
        public var pagesWired: UInt64? = nil
        public var slabsAlreadyWired: UInt64? = nil
        public var wireFallbacks: UInt64? = nil
    }
    /// Allocates the admitted decode cache and faults (or wires) its pages before decode, so
    /// the first tokens do not stall on page faults. Best effort: when the host's available
    /// memory reaches `minimumAvailableBytes` or the process reaches its budget, growth stops
    /// there and the cache limit drops to what is resident. Never fails a request for memory.
    /// `wire`: user-wire (mlock) each slab so the VM cannot compress or reclaim it; a failed
    /// wire falls back to the preserving touch and is counted.
    package func prepareSlabs(shouldContinue: () -> Bool, minimumAvailableBytes: UInt64, wire: Bool = false) throws -> PhysicalTouch {
        // The lock is taken per slab, never across a page fault or an mlock, so reads and status are not stalled
        // (ROADMAP R5). The caller joined every read producer and is the only thread that starts reads, so no record
        // is written into a slab while it is touched; a trim meanwhile only drops slabs this pass still holds alive.
        try lock.withLock {
            guard quiescentLocked else { throw EngineError("Slab preparation requires joined readers and consumers") }
        }
        let page = Int(getpagesize())
        var peak: UInt64 = 0
        var available = UInt64.max
        let swapStart = origami_swapout_bytes()
        // Stops at the floor, or as soon as anything is swapped out: growing further would push other apps to disk.
        func roomy() throws -> Bool {
            guard shouldContinue() else { throw IOOperationCancelled() }
            guard ledger.optionalGrowthAllowed else { return false }
            let footprint = origami_physical_footprint()
            let headroom = origami_available_memory()
            peak = max(peak, footprint)
            available = min(available, headroom)
            let hostAllowsGrowth =
                ledger.usesAvailableMemoryFloor
                ? headroom >= minimumAvailableBytes
                : MemoryPressureLevel.current()?.isElevated != true
            let footprintLimit = ledger.usesAvailableMemoryFloor ? ledger.usableBytes : ledger.ceiling
            return footprint <= footprintLimit && hostAllowsGrowth && origami_swapout_bytes() == swapStart
        }
        func allocated() throws -> Bool {
            try lock.withLock {
                // The class furthest below its limit, so a stop leaves the classes in proportion.
                let open = classes.indices.filter { classes[$0].allocated < classes[$0].byteLimit }
                guard let c = open.min(by: { fill($0) < fill($1) }) else { return false }
                return try core.allocateSlabLocked(sizeClass: c) != nil
            }
        }
        var stopped = !(try roomy())
        while !stopped, try allocated() { stopped = !(try roomy()) }
        var covered: UInt64 = 0
        var pages: UInt64 = 0
        var count = 0
        var slabsWired: UInt64 = 0
        var pagesWired: UInt64 = 0
        var alreadyWired: UInt64 = 0
        var wireFallbacks: UInt64 = 0
        var storages = lock.withLock { classes.flatMap { $0.slabs.values.map(\.storage) } }
        for storage in storages where !stopped {
            var touched: UInt64 = 0
            let buffer = storage.buffer
            if wire && lock.withLock({ storage.wiredBytes == buffer.length }) {
                alreadyWired += 1
            } else if wire, origami_wire_pages(buffer.contents(), buffer.length, page, &touched) == 0 {
                lock.withLock { storage.wiredBytes = buffer.length }
                slabsWired += 1
                pagesWired += touched
            } else {
                if wire { wireFallbacks += 1 }
                _ = origami_touch_preserving_pages(buffer.contents(), buffer.length, page, &touched)
            }
            covered += UInt64(buffer.length)
            pages += touched
            count += 1
            stopped = !(try roomy())
        }
        storages = []  // a slab a trim retired meanwhile goes now, outside the lock
        lock.lock()
        defer { lock.unlock() }
        // A short host keeps the slabs it has; the limit follows so no read allocates more, but never below the
        // governor's minimum cache (an empty cache would leave every demand waiting for a slot).
        for slots in classes {
            let slab = slots.slabBytes
            let minimum = min((UInt64(slots.mandatorySlots * slots.recordBytes) + slab - 1) / slab * slab, slots.byteLimit)
            if slots.allocated < slots.byteLimit { slots.byteLimit = max(slots.allocated, minimum) }
        }
        // This preparation boundary joined readers/consumers. Only successfully wired slabs
        // prove residency; untouched or unwired allocations must not erase measured overhead.
        let resident = classes.reduce(UInt64(0)) { $0 + $1.slabs.values.reduce(UInt64(0)) { $0 + UInt64($1.storage.wiredBytes) } }
        ledger.reconcilePhysicalOverhead(residentOwnedBytes: resident)
        let memory = ledger.snapshot()
        var receipt = PhysicalTouch(
            targetBytes: limitLocked, allocatedBytes: allocatedLocked, coveredBytes: covered,
            pagesTouched: pages, slabsTouched: count, hostPageBytes: page,
            physicalFootprintHighWater: max(peak, memory.physicalFootprint),
            minimumAvailableMemoryEstimate: min(available, memory.availableMemoryEstimate), memory: memory)
        if wire {
            receipt.slabsWired = slabsWired
            receipt.pagesWired = pagesWired
            receipt.slabsAlreadyWired = alreadyWired
            receipt.wireFallbacks = wireFallbacks
        }
        return receipt
    }
    private let core: ExpertSlotCore<ExpertCacheEviction>
    /// The size classes in use, and the core's classes: their slabs, slots and limits.
    package let sizes: SizeClasses
    private let classes: [ExpertSlotClass]
    private let eviction: ExpertCacheEviction
    private let residency: ExpertGPUResidency
    package let device: MTLDevice
    package let ledger: MemoryLedger
    private let lock: NSLock
    private let slotsPerSlab: Int
    private let maximumByteLimit: UInt64
    /// Mid-reply growth (`prepareGrowth`, `attachGrowth`): slabs allocated, charged and faulted (or wired) off the token loop,
    /// kept out of the cache until a token boundary adds them. `growthCeiling` is the decode phase's limit (0: no growth);
    /// a trim drops them and moves `growthEpoch`, so a preparation it raced drops its own too. `growthOthers`: the budget the
    /// phase left the rest of the model (its usable less the ceiling), which growth never takes even before they reserve it.
    private var preparedGrowth: [(sizeClass: Int, storage: ExpertSlabStorage)] = [], growthCeiling: UInt64 = 0, growthOthers: UInt64 = 0,
        growthEpoch: UInt64 = 0,
        growing = false
    private static let growthBatchBytes: UInt64 = 128 * 1_024 * 1_024
    private var layerZeroMisses: UInt64 = 0
    private var reuse: ExpertReusePolicy? { eviction.reuse }

    /// `sizes`: the record size classes asked for (`SizeClasses.grouping` of a mixed-size GGUF's layers). They are used
    /// only when each class's share of `byteLimit` holds `minimumClassRecords` records (and the 32 reuse admission needs);
    /// otherwise every slot is `PackStore.expertBytes`, as with one class. Limits are in bytes either way.
    public init(
        device: MTLDevice, ledger: MemoryLedger, byteLimit: UInt64, slotsPerSlab: Int = 8,
        maximumByteLimit: UInt64? = nil, reuseAdmission: Bool = false, sizes asked: SizeClasses = .one, minimumClassRecords: Int = 0
    ) throws {
        let maximum = maximumByteLimit ?? byteLimit
        let enough = max(minimumClassRecords, reuseAdmission ? 32 : 0, slotsPerSlab)
        let sizes =
            asked.recordBytes.count > 1
                && zip(Self.split(byteLimit, asked, slotsPerSlab: slotsPerSlab), asked.recordBytes).allSatisfy {
                    $0 >= UInt64(enough * $1)
                }
            ? asked : .one
        let eviction = ExpertCacheEviction()
        let residency = ExpertGPUResidency(eviction: eviction)
        let limits = Self.split(byteLimit, sizes, slotsPerSlab: slotsPerSlab)
        let maxima = Self.split(maximum, sizes, slotsPerSlab: slotsPerSlab)
        let shape = ExpertSlotShape(
            layers: 48, expertsPerLayer: 512,
            classes: sizes.recordBytes.indices.map { c in
                let layers = sizes.layerClasses.filter { $0 == c }.count
                return .init(
                    recordBytes: sizes.recordBytes[c], slotsPerSlab: slotsPerSlab, byteLimit: limits[c], maximumByteLimit: maxima[c],
                    mandatorySlots: (ConversationPhaseBudget.minimumSlots * layers + 47) / 48,
                    regularFrom: ExpertReusePolicy.transientSlots,
                    category: "expert-slabs")
            }, layerClasses: sizes.layerClasses)
        core = try ExpertSlotCore(device: device, ledger: ledger, shape: shape, policy: eviction, observer: residency)
        self.sizes = sizes
        classes = core.classes
        self.eviction = eviction
        self.residency = residency
        lock = core.lock
        self.device = device
        self.ledger = ledger
        self.slotsPerSlab = slotsPerSlab
        self.maximumByteLimit = maximum
        guard !reuseAdmission || (slotsPerSlab == 8 && byteLimit >= UInt64(32 * PackStore.expertBytes)) else {
            throw EngineError("Reuse admission requires whole 8-slot slabs and 24 transient slots")
        }
        eviction.configure(classes: classes.count, layerClasses: sizes.layerClasses)
        residency.configure(classes: classes, slotsPerSlab: slotsPerSlab)
        eviction.reuse = reuseAdmission ? try ExpertReusePolicy(ledger: ledger) : nil
    }

    /// `total` split between the classes by their weights, in whole slabs of each, rounded down and within the class's
    /// keys; one class: `total` itself.
    private static func split(_ total: UInt64, _ sizes: SizeClasses, slotsPerSlab: Int) -> [UInt64] {
        guard sizes.recordBytes.count > 1 else { return [total] }
        let weights = sizes.weights
        let sum = Double(weights.reduce(0, +))
        return sizes.recordBytes.indices.map { c in
            let slab = UInt64(slotsPerSlab * sizes.recordBytes[c])
            let keys = UInt64(sizes.layerClasses.filter { $0 == c }.count * 512 * sizes.recordBytes[c]) / slab * slab
            return min(UInt64(Double(total) * Double(weights[c]) / sum) / slab * slab, keys)
        }
    }
    /// Sets each class's limit to its share of `total`.
    private func splitLocked(_ total: UInt64) {
        for (slots, bytes) in zip(classes, Self.split(total, sizes, slotsPerSlab: slotsPerSlab)) { slots.byteLimit = bytes }
    }
    private var allocatedLocked: UInt64 { classes.reduce(0) { $0 + $1.allocated } }
    /// The classes' limits together (one class: its limit).
    private var limitLocked: UInt64 { classes.reduce(0) { $0 + $1.byteLimit } }
    private var pendingTrimLocked: Bool { classes.contains { $0.pendingTrim } }
    private var quiescentLocked: Bool { classes.allSatisfy(\.quiescent) }
    /// How full class `c` is against its limit.
    private func fill(_ c: Int) -> Double { Double(classes[c].allocated) / Double(max(1, classes[c].byteLimit)) }
    /// Every class's slabs back to its limit (or `maximumSlabs` each).
    private func retireLocked(maximumSlabs: Int, retired: inout [ExpertSlabStorage]) {
        for c in classes.indices { core.retireSlabsLocked(sizeClass: c, maximumSlabs: maximumSlabs, retired: &retired) }
    }
    /// The slot bytes of `key`'s class: what a record of it is read into.
    package func recordBytes(_ key: ExpertKey) -> Int { sizes.recordBytes[sizes.layerClasses[key.layer]] }

    /// Phase switches occur after the same quiescent ownership check as resize.
    package func setReuseDecode(_ enabled: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        try eviction.setReuseDecode(enabled, core)
    }

    package func beginDecodeRow(layer: Int, experts: [Int]) throws -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        return try eviction.beginDecodeRow(layer: layer, experts: experts)
    }

    /// Immutable source-cost snapshot supplied outside the cache lock. No source or
    /// scheduler callback runs while cache metadata is locked.
    package func setStorageCosts(_ costs: [Double]) {
        lock.lock()
        defer { lock.unlock() }
        eviction.setStorageCosts(costs)
    }

    package func setPrefillScan(_ enabled: Bool) {
        lock.lock()
        eviction.setPrefillScan(enabled)
        lock.unlock()
    }

    /// Pre-routed (speculative) admission. A resident or in-flight record
    /// issues nothing. Otherwise only a free non-transient slot, a new slab, a
    /// CLOCK victim, or (reuse admission) the lowest-score unprotected, unpinned
    /// verified record that the predicted key's own reuse score beats may be
    /// taken: never the 24 transient bypass slots nor a borrowed bypass, which
    /// exist so that exact demand cannot fail. Protected (current-row), reserved,
    /// reading and GPU-pinned slots are never victims. No slot => noSlot.
    /// `overrideScore` (decode_prerouting_slot_override, reuse decode only): when
    /// the score rule refuses the key, take instead the lowest-score eligible
    /// record outside the target layer and the layer before it (the current row
    /// and every record the next layer may demand) => overridden. Same minima.
    package enum SpeculativeAdmission {
        case resident, inFlight
        case admitted(Acquisition)
        case overridden(Acquisition)
        case noSlot
    }
    package func acquireSpeculative(_ key: ExpertKey, overrideScore: Bool = false) throws -> SpeculativeAdmission {
        lock.lock()
        defer { lock.unlock() }
        let c = sizes.layerClasses[key.layer]
        if let index = core.lookup[key.index], let slot = classes[c].slots[index] {
            return [.verified, .pinned, .evictable].contains(slot.phase) ? .resident : .inFlight
        }
        guard core.nextGeneration < UInt64.max else { throw EngineError("Cache generation exhausted") }
        guard let victim = try eviction.speculativeVictim(key.index, overrideScore: overrideScore, core) else { return .noSlot }
        let index = victim.index
        let ticket = core.reserveLocked(index, key: key.index)
        eviction.didReserveSpeculative(index, sizeClass: c)
        let acquisition = Acquisition(
            ticket: ExpertTicket(sizeClass: c, slot: index, generation: ticket.generation, key: key), phase: .reserved,
            newlyReserved: true)
        return victim.overridden ? .overridden(acquisition) : .admitted(acquisition)
    }

    /// `cache_victim_index`: the counter and per-layer answers beside a copy of the full scans they replaced (the
    /// speculative one as if `overrideScore`). Changes nothing but the minima's refresh; nil when all agree.
    package func victimIndexMismatch(_ key: ExpertKey) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return eviction.victimIndexMismatch(key.index, core)
    }

    /// A decode demand miss on layer 0 (`Snapshot.layerZeroMisses`), counted by the execution, which knows the phase.
    package func countLayerZeroMiss() { lock.withLock { layerZeroMisses += 1 } }

    public func acquire(_ key: ExpertKey) throws -> Acquisition {
        let acquisition = try core.acquire(key.index)
        return Acquisition(
            ticket: ExpertTicket(
                sizeClass: acquisition.ticket.sizeClass, slot: acquisition.ticket.slot, generation: acquisition.ticket.generation, key: key),
            phase: acquisition.phase, newlyReserved: acquisition.newlyReserved)
    }

    /// Read-only resident membership (pre-routing slot-override checks). Never admits
    /// a miss or changes the target reuse history or CLOCK reference bits.
    package func draftResidentExperts(layer: Int) throws -> [Int] {
        guard (0..<48).contains(layer) else { throw EngineError("Invalid draft layer") }
        lock.lock()
        defer { lock.unlock() }
        let slots = classes[sizes.layerClasses[layer]]
        return (0..<512).filter { expert in
            guard let index = core.lookup[layer * 512 + expert], let slot = slots.slots[index],
                slots.slabs[slot.slab] != nil
            else { return false }
            return [.verified, .evictable, .pinned].contains(slot.phase)
        }
    }

    /// Exactly one read producer can transition a reservation into reading.
    public func beginRead(_ ticket: ExpertTicket) throws -> ExpertBuffer {
        let record = try core.beginRead(ticket.core)
        return ExpertBuffer(storage: record.storage, offset: record.offset, length: sizes.recordBytes[ticket.sizeClass])
    }

    /// Must be called only after every writer for the ticket has stopped touching
    /// its destination. Hedge reads need separate staging; the winner copies here.
    @discardableResult public func completeRead(_ ticket: ExpertTicket, verified: Bool) -> Bool {
        core.completeRead(ticket.core, verified: verified)
    }

    public func cancel(_ ticket: ExpertTicket) throws { try core.cancel(ticket.core) }

    public func pin(_ ticket: ExpertTicket) throws -> ExpertBuffer {
        let record = try core.pin(ticket.core)
        return ExpertBuffer(storage: record.storage, offset: record.offset, length: sizes.recordBytes[ticket.sizeClass])
    }

    /// Call from command-buffer completion, never immediately after encoding.
    public func unpinAfterGPUCompletion(_ ticket: ExpertTicket) throws { try core.unpin(ticket.core) }

    /// Release cold whole slabs; a busy slab retains its bytes until readers join.
    /// Return actual bytes rather than claiming the requested limit was reached.
    @discardableResult public func resize(to bytes: UInt64) throws -> UInt64 {
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        try validateLimit(bytes)
        return resizeLocked(to: bytes, retired: &retired)
    }

    /// Pressure reduces prospective lazy capacity as well as live slabs. A large unallocated allowance must not
    /// survive merely because current reservations still fit. Keep both actual non-cache owners and the phase's
    /// planned owners, and never shrink below the execution engine's compulsory streaming cache.
    package func trimToBudget(minimumBytes: UInt64) throws -> UInt64 {
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        let before = allocatedLocked
        dropGrowthLocked(retired: &retired)
        let prepared = retired.reduce(UInt64(0)) { $0 + UInt64($1.buffer.length) }
        // These detached prepared buffers cannot be observed by a reader and retire
        // immediately after unlock. They must not count as another owner's footprint
        // while deriving the cache target, just because destruction was moved outside.
        let charged = ledger.reservedBytes
        let reserved = charged > prepared ? charged - prepared : 0
        let usable = ledger.usableBytes
        let allocated = allocatedLocked
        let actualOthers = reserved > allocated ? reserved - allocated : 0
        let others = max(actualOthers, growthCeiling > 0 ? growthOthers : 0)
        let slab = UInt64(slotsPerSlab * PackStore.expertBytes)
        let target = max(minimumBytes, (usable > others ? usable - others : 0) / slab * slab)
        if target < limitLocked {
            try validateLimit(target)
            _ = resizeLocked(to: target, retired: &retired)
        } else if pendingTrimLocked {
            retireLocked(maximumSlabs: .max, retired: &retired)
        }
        return before - allocatedLocked + prepared
    }

    /// A busy reader or queued GPU command can outlive the original pressure callback.
    /// Retry its already-lowered limit at existing token/phase boundaries, without a new
    /// pressure signal or another target reduction. Bound each boundary's retirement work.
    package func completePendingTrim() {
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        guard pendingTrimLocked, !core.slabGrowthHeld else { return }
        retireLocked(maximumSlabs: 8, retired: &retired)
    }

    /// Phase changes must be all-or-nothing with respect to the target limit.
    /// Unlike general trimming, reject any reserved writer or live GPU pin,
    /// even when growing or requesting the current limit. No mutation on error.
    @discardableResult public func resizeQuiescent(
        to bytes: UInt64, growthCeiling: UInt64? = nil,
        growthBudget: UInt64? = nil
    ) throws -> UInt64 {
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        try validateLimit(bytes)
        guard quiescentLocked else {
            throw EngineError("Expert cache phase change has reserved, reading or pinned slots", code: "expert_phase_busy")
        }
        let result = resizeLocked(to: bytes, retired: &retired)
        if let growthCeiling {
            configureGrowthLocked(growthCeiling, budget: growthBudget ?? ledger.usableBytes)
        }
        return result
    }
    private func validateLimit(_ bytes: UInt64) throws {
        guard bytes <= maximumByteLimit, bytes % UInt64(slotsPerSlab * PackStore.expertBytes) == 0 else {
            throw EngineError("Expert cache resize exceeds bounded whole slabs", code: "invalid_configuration")
        }
    }
    private func resizeLocked(to bytes: UInt64, retired: inout [ExpertSlabStorage]) -> UInt64 {
        splitLocked(bytes)
        dropGrowthLocked(retired: &retired)
        retireLocked(maximumSlabs: .max, retired: &retired)
        return allocatedLocked
    }

    package func holdSlabGrowth(_ held: Bool) {
        lock.lock()
        core.slabGrowthHeld = held
        lock.unlock()
    }

    /// The limit mid-reply growth may restore (the decode phase's own limit; 0 outside decode). `budget` is the
    /// owner budget used to calculate that limit, which may exceed a temporarily reduced current allowance.
    /// The difference keeps the phase's other owners reserved throughout recovery. Drops prepared slabs.
    package func setGrowthCeiling(_ bytes: UInt64, budget: UInt64? = nil) {
        let usable = budget ?? ledger.usableBytes
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        configureGrowthLocked(bytes, budget: usable)
        dropGrowthLocked(retired: &retired)
    }
    private func configureGrowthLocked(_ bytes: UInt64, budget: UInt64) {
        growthCeiling = min(bytes, maximumByteLimit)
        growthOthers = budget > growthCeiling ? budget - growthCeiling : 0
    }
    /// Lazy decode (including file-backed contexts) recovers its allowance without allocating or faulting pages on
    /// the governor thread. Demand still admits every new slab against the ledger and physical process ceiling.
    /// The cache lock makes this atomic with trims and phase changes; existing records and GPU addresses do not move.
    package func restoreGrowthLimit() {
        lock.lock()
        defer { lock.unlock() }
        guard ledger.optionalGrowthAllowed else { return }
        let usable = ledger.usableBytes
        let reserved = ledger.reservedBytes
        let ceiling = min(growthCeiling, usable > growthOthers ? usable - growthOthers : 0)
        let available = usable > reserved ? usable - reserved : 0
        let (room, overflow) = allocatedLocked.addingReportingOverflow(available)
        let slab = UInt64(slotsPerSlab * PackStore.expertBytes)
        let target = min(ceiling, overflow ? .max : room) / slab * slab
        if target > limitLocked {
            // A class that grew past its share keeps its limit.
            for (slots, bytes) in zip(classes, Self.split(target, sizes, slotsPerSlab: slotsPerSlab)) {
                slots.byteLimit = max(slots.byteLimit, bytes)
                slots.pendingTrim = slots.allocated > slots.byteLimit
            }
        }
    }
    /// A trim: prepared slabs are released (their charge and wiring with them) and a running preparation drops its own.
    package func dropGrowth() {
        var retired: [ExpertSlabStorage] = []
        lock.lock()
        defer {
            lock.unlock()
            withExtendedLifetime(retired) {}
        }
        dropGrowthLocked(retired: &retired)
    }
    private func dropGrowthLocked(retired: inout [ExpertSlabStorage]) {
        retired += preparedGrowth.map(\.storage)
        preparedGrowth.removeAll()
        growthEpoch &+= 1
    }
    /// Mid-reply growth, off the token loop (any thread): allocates, charges and faults (or wires) the whole slabs that fit
    /// both the room left in the ledger's budget and the growth ceiling, without touching the cache. Stops at the host
    /// floor or on fresh swap, like `prepareSlabs`. Publishes at most 128 MiB per tranche (one slab minimum), pausing
    /// the utility worker between tranches so recovery does not prefault a whole GB at once. No token thread sleeps.
    /// `attachGrowth` adds a bounded tranche at a token boundary. Returns total slabs prepared by this pass.
    @discardableResult package func prepareGrowth(minimumAvailableBytes: UInt64, wire: Bool) throws -> Int {
        lock.lock()
        let epoch = growthEpoch
        let allocated = allocatedLocked
        let limit = limitLocked
        let top = max(allocated, limit) + preparedGrowth.reduce(UInt64(0)) { $0 + UInt64($1.storage.buffer.length) }
        // Lazy growth may still reserve up to the current limit: that room is not ours.
        let reserved = ledger.reservedBytes + (limit > allocated ? limit - allocated : 0)
        let usable = ledger.usableBytes
        // The phase's own split of a smaller budget: the rest of the model keeps what the phase left it (KV and owners grow later).
        let ceiling = min(growthCeiling, usable > growthOthers ? usable - growthOthers : 0)
        // The class of each slab to prepare, in order: the one furthest below its share, while its slab fits the room.
        var plan: [Int] = []
        if !(growing || pendingTrimLocked || !ledger.optionalGrowthAllowed) {
            var room = min(usable > reserved ? usable - reserved : 0, ceiling > top ? ceiling - top : 0)
            var have = classes.map { max($0.allocated, $0.byteLimit) }
            for (c, storage) in preparedGrowth { have[c] += UInt64(storage.buffer.length) }
            let weights = sizes.weights
            func after(_ c: Int) -> Double { Double(have[c] + classes[c].slabBytes) / Double(weights[c]) }
            while let c = classes.indices.min(by: { after($0) < after($1) }), classes[c].slabBytes <= room {
                plan.append(c)
                have[c] += classes[c].slabBytes
                room -= classes[c].slabBytes
            }
        }
        let count = plan.count
        if count > 0 { growing = true }
        lock.unlock()
        guard count > 0 else { return 0 }
        var made: [(sizeClass: Int, storage: ExpertSlabStorage)] = []
        defer {
            lock.lock()
            growing = false
            if epoch == growthEpoch && ledger.optionalGrowthAllowed {
                preparedGrowth += made
                made = []
            }
            lock.unlock()
        }
        let batchSize = max(1, Int(Self.growthBatchBytes / UInt64(slotsPerSlab * PackStore.expertBytes)))
        var total = 0
        let page = Int(getpagesize())
        let swapStart = origami_swapout_bytes()
        func current() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return epoch == growthEpoch && !pendingTrimLocked
        }
        func hostAllowsGrowth(_ bytes: UInt64) -> Bool {
            guard ledger.optionalGrowthAllowed else { return false }
            let reserved = ledger.reservedBytes
            let usable = ledger.usableBytes
            guard reserved <= usable, bytes <= usable - reserved else { return false }
            if ledger.usesAvailableMemoryFloor { return origami_available_memory() >= minimumAvailableBytes + bytes }
            return MemoryPressureLevel.current()?.isElevated != true && ledger.physicalAllocationShortfall(bytes: bytes) == 0
        }
        while total < count, current(), hostAllowsGrowth(classes[plan[total]].slabBytes), origami_swapout_bytes() == swapStart {
            let c = plan[total]
            let bytes = classes[c].slabBytes
            // The pool keeps no autoreleased reference: a dropped slab must free its charge at once.
            let admitted = try autoreleasepool { () throws -> Bool in
                guard let reservation = try ledger.reservePhysicalIfAvailable(bytes: bytes, category: "expert-slabs") else { return false }
                guard let buffer = device.makeBuffer(length: Int(bytes), options: .storageModeShared) else {
                    try ledger.release(reservation)
                    throw EngineError("Metal expert slab allocation failed")
                }
                let storage = ExpertSlabStorage(buffer: buffer, ledger: ledger, reservation: reservation)
                var touched: UInt64 = 0
                if wire, origami_wire_pages(buffer.contents(), buffer.length, page, &touched) == 0 {
                    storage.wiredBytes = buffer.length
                } else {
                    _ = origami_touch_preserving_pages(buffer.contents(), buffer.length, page, &touched)
                }
                made.append((c, storage))
                return true
            }
            guard admitted else { break }
            total += 1
            if made.count >= batchSize {
                lock.lock()
                let publish = epoch == growthEpoch && !pendingTrimLocked && ledger.optionalGrowthAllowed
                if publish {
                    preparedGrowth += made
                    made.removeAll(keepingCapacity: true)
                }
                lock.unlock()
                guard publish else { break }
                if total < count { Thread.sleep(forTimeInterval: 0.2) }
            }
        }
        return total
    }
    /// Adds the prepared slabs and raises the limit by their bytes, only where no encoded command holds the slab layout
    /// (slab growth not held: between decode tokens). New slots start free; every existing slot, record, pin and GPU
    /// publication is untouched, and the generation moves so the next command's residency set includes them. Returns slabs added.
    @discardableResult package func attachGrowth() -> Int {
        lock.lock()
        defer { lock.unlock() }
        guard ledger.optionalGrowthAllowed, !core.slabGrowthHeld, !pendingTrimLocked, !preparedGrowth.isEmpty else { return 0 }
        let added = min(preparedGrowth.count, max(1, Int(Self.growthBatchBytes / UInt64(slotsPerSlab * PackStore.expertBytes))))
        let before = classes.map { max($0.byteLimit, $0.allocated) }
        var grown = [UInt64](repeating: 0, count: classes.count)
        for (c, storage) in preparedGrowth.prefix(added) {
            core.insertSlabLocked(storage, sizeClass: c)
            grown[c] += classes[c].slabBytes
        }
        preparedGrowth.removeFirst(added)
        for c in classes.indices where grown[c] > 0 {
            classes[c].byteLimit = min(before[c] + grown[c], classes[c].maximumByteLimit)
        }
        return added
    }
    /// Slots the current limit allows in the smallest class (one class: all of them); the memory governor may lower it
    /// while a phase runs. A prompt read needs a whole layer of records in its class (`slotLimit(sizeClass:)`).
    package var slotLimit: Int {
        lock.lock()
        defer { lock.unlock() }
        return classes.map { Int($0.byteLimit) / $0.recordBytes }.min()!
    }
    package func slotLimit(sizeClass c: Int) -> Int { lock.withLock { Int(classes[c].byteLimit) / classes[c].recordBytes } }

    /// The residency table (allocated on first use): per key, slot + 1 and epoch; per slot, its GPU address.
    package func gpuTable() throws -> (keys: MTLBuffer, slots: MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }
        if let keys = residency.gpuKeys, let addresses = residency.gpuSlots { return (keys, addresses) }
        let table = try residency.table(device: device, ledger: ledger)
        eviction.markAllDirty()
        return table
    }
    /// Holds `layer`'s published records for a command about to be committed and returns its epoch: the command trusts
    /// only records published before this call.
    package func holdForGPU(layer: Int) -> UInt32 {
        lock.lock()
        defer { lock.unlock() }
        eviction.markDirty(layer)
        return residency.hold(layer: layer)
    }
    package func releaseForGPU(layer: Int?) {
        lock.lock()
        defer { lock.unlock() }
        residency.release(layer: layer)
        if let layer { eviction.markDirty(layer) } else { eviction.markAllDirty() }
    }
    /// Which keys a command holding `epoch` computes on the GPU (its own rule, so host and GPU agree).
    package func gpuComputes(_ keys: [ExpertKey], epoch: UInt32) -> [Bool] {
        lock.lock()
        defer { lock.unlock() }
        guard let gpuKeys = residency.gpuKeys else { return keys.map { _ in false } }
        return keys.map { key in
            let entry = gpuKeys.contents().load(fromByteOffset: key.index * 8, as: UInt64.self)
            return entry & 0xFFFF_FFFF != 0 && UInt32(entry >> 32) < epoch
        }
    }
    package func slabBuffers() -> (generation: UInt64, buffers: [MTLBuffer]) {
        lock.lock()
        defer { lock.unlock() }
        return (core.slabGeneration, classes.flatMap { $0.slabs.values.map(\.buffer) })
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let all = classes.flatMap(\.slots)
        var snapshot = Snapshot(
            allocatedBytes: allocatedLocked, byteLimit: limitLocked, maximumByteLimit: maximumByteLimit,
            slabs: classes.reduce(0) { $0 + $1.slabs.count },
            activeSlots: all.compactMap { $0 }.filter { $0.phase != .free }.count,
            reservedSlots: all.compactMap { $0 }.filter { $0.phase == .reserved }.count,
            pinnedSlots: all.compactMap { $0 }.filter { $0.pins > 0 }.count,
            inFlightReads: all.compactMap { $0 }.filter { $0.phase == .reading }.count,
            metadataSlots: all.count, hits: core.hits, misses: core.misses,
            evictions: core.evictions, rejectedCompletions: core.rejected,
            reuseRows: reuse?.rows, reuseBypasses: reuse == nil ? nil : eviction.reuseBypasses,
            reuseAdmissions: reuse == nil ? nil : eviction.reuseAdmissions, reuseDecisionNanoseconds: reuse?.decisionNanoseconds,
            reuseReplayResets: reuse?.replayResets,
            reuseBorrowedBypasses: reuse == nil ? nil : eviction.reuseBorrowedBypasses,
            reuseTransientSlots: reuse == nil ? nil : ExpertReusePolicy.transientSlots, layerZeroMisses: layerZeroMisses)
        if classes.count > 1 { snapshot.sizeClasses = classes.map { .init(recordBytes: $0.recordBytes, byteLimit: $0.byteLimit) } }
        return snapshot
    }
}
