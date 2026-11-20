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
    fileprivate let slot: Int
    public let generation: UInt64
    public let key: ExpertKey
    /// The slot core's ticket (Qwen's cache has one size class).
    fileprivate var core: ExpertSlotTicket { ExpertSlotTicket(sizeClass: 0, slot: slot, generation: generation, key: key.index) }
}

/// A view of one complete expert record (`PackStore.expertBytes`) inside a cache slab or the replica arena.
/// Holding it keeps the Metal buffer and its ledger charge alive, but it does not pin the slot: only a cache pin
/// (or a LoadedExpert lease) keeps the record from being recycled.
public final class ExpertBuffer: @unchecked Sendable {
    private let storage: ExpertSlabStorage?
    public let buffer: MTLBuffer
    public let offset: Int
    public let length = PackStore.expertBytes
    fileprivate init(storage: ExpertSlabStorage, offset: Int) {
        self.storage = storage
        self.buffer = storage.buffer
        self.offset = offset
    }
    /// The replica arena's Metal resource independently carries its reservation.
    /// The LoadedExpert consumer pin protects this particular record from reuse.
    package init(replicaBuffer: MTLBuffer, offset: Int) {
        precondition(offset >= 0 && offset <= replicaBuffer.length - PackStore.expertBytes)
        storage = nil
        buffer = replicaBuffer
        self.offset = offset
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
/// The slots, tickets and slab mechanics are the model-neutral `ExpertSlotCore` (one size class); the victims are
/// `ExpertCacheEviction`'s, the GPU table `ExpertGPUResidency`'s; this type keeps the API and the pacing of resizes,
/// trims and growth.
public final class ExpertCache: @unchecked Sendable {
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
            guard slots.quiescent else { throw EngineError("Slab preparation requires joined readers and consumers") }
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
                guard slots.allocated < slots.byteLimit else { return false }
                return try core.allocateSlabLocked(sizeClass: 0) != nil
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
        var storages = lock.withLock { slots.slabs.values.map(\.storage) }
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
        let slab = UInt64(slotsPerSlab * PackStore.expertBytes)
        let minimum = min((UInt64(ConversationPhaseBudget.minimumSlots * PackStore.expertBytes) + slab - 1) / slab * slab, slots.byteLimit)
        if slots.allocated < slots.byteLimit { slots.byteLimit = max(slots.allocated, minimum) }
        // This preparation boundary joined readers/consumers. Only successfully wired slabs
        // prove residency; untouched or unwired allocations must not erase measured overhead.
        let resident = slots.slabs.values.reduce(UInt64(0)) { $0 + UInt64($1.storage.wiredBytes) }
        ledger.reconcilePhysicalOverhead(residentOwnedBytes: resident)
        let memory = ledger.snapshot()
        var receipt = PhysicalTouch(
            targetBytes: slots.byteLimit, allocatedBytes: slots.allocated, coveredBytes: covered,
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
    /// The core's one size class: its slabs, slots and limit.
    private let slots: ExpertSlotClass
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
    private var preparedGrowth: [ExpertSlabStorage] = [], growthCeiling: UInt64 = 0, growthOthers: UInt64 = 0, growthEpoch: UInt64 = 0,
        growing = false
    private static let growthBatchBytes: UInt64 = 128 * 1_024 * 1_024
    private var layerZeroMisses: UInt64 = 0
    private var reuse: ExpertReusePolicy? { eviction.reuse }

    public init(
        device: MTLDevice, ledger: MemoryLedger, byteLimit: UInt64, slotsPerSlab: Int = 8,
        maximumByteLimit: UInt64? = nil, reuseAdmission: Bool = false
    ) throws {
        let maximum = maximumByteLimit ?? byteLimit
        let eviction = ExpertCacheEviction()
        let residency = ExpertGPUResidency(eviction: eviction)
        let shape = ExpertSlotShape(
            layers: 48, expertsPerLayer: 512,
            classes: [
                .init(
                    recordBytes: PackStore.expertBytes, slotsPerSlab: slotsPerSlab, byteLimit: byteLimit, maximumByteLimit: maximum,
                    mandatorySlots: ConversationPhaseBudget.minimumSlots, regularFrom: ExpertReusePolicy.transientSlots,
                    category: "expert-slabs")
            ], layerClasses: Array(repeating: 0, count: 48))
        core = try ExpertSlotCore(device: device, ledger: ledger, shape: shape, policy: eviction, observer: residency)
        slots = core.classes[0]
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
        eviction.reuse = reuseAdmission ? try ExpertReusePolicy(ledger: ledger) : nil
    }

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
        if let index = core.lookup[key.index], let slot = slots.slots[index] {
            return [.verified, .pinned, .evictable].contains(slot.phase) ? .resident : .inFlight
        }
        guard core.nextGeneration < UInt64.max else { throw EngineError("Cache generation exhausted") }
        guard let victim = try eviction.speculativeVictim(key.index, overrideScore: overrideScore, core) else { return .noSlot }
        let index = victim.index
        let ticket = core.reserveLocked(index, key: key.index)
        eviction.didReserveSpeculative(index)
        let acquisition = Acquisition(
            ticket: ExpertTicket(slot: index, generation: ticket.generation, key: key), phase: .reserved, newlyReserved: true)
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
            ticket: ExpertTicket(slot: acquisition.ticket.slot, generation: acquisition.ticket.generation, key: key),
            phase: acquisition.phase, newlyReserved: acquisition.newlyReserved)
    }

    /// Read-only resident membership (pre-routing slot-override checks). Never admits
    /// a miss or changes the target reuse history or CLOCK reference bits.
    package func draftResidentExperts(layer: Int) throws -> [Int] {
        guard (0..<48).contains(layer) else { throw EngineError("Invalid draft layer") }
        lock.lock()
        defer { lock.unlock() }
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
        return ExpertBuffer(storage: record.storage, offset: record.offset)
    }

    /// Must be called only after every writer for the ticket has stopped touching
    /// its destination. Hedge reads need separate staging; the winner copies here.
    @discardableResult public func completeRead(_ ticket: ExpertTicket, verified: Bool) -> Bool {
        core.completeRead(ticket.core, verified: verified)
    }

    public func cancel(_ ticket: ExpertTicket) throws { try core.cancel(ticket.core) }

    public func pin(_ ticket: ExpertTicket) throws -> ExpertBuffer {
        let record = try core.pin(ticket.core)
        return ExpertBuffer(storage: record.storage, offset: record.offset)
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
        let before = slots.allocated
        dropGrowthLocked(retired: &retired)
        let prepared = retired.reduce(UInt64(0)) { $0 + UInt64($1.buffer.length) }
        // These detached prepared buffers cannot be observed by a reader and retire
        // immediately after unlock. They must not count as another owner's footprint
        // while deriving the cache target, just because destruction was moved outside.
        let charged = ledger.reservedBytes
        let reserved = charged > prepared ? charged - prepared : 0
        let usable = ledger.usableBytes
        let actualOthers = reserved > slots.allocated ? reserved - slots.allocated : 0
        let others = max(actualOthers, growthCeiling > 0 ? growthOthers : 0)
        let slab = UInt64(slotsPerSlab * PackStore.expertBytes)
        let target = max(minimumBytes, (usable > others ? usable - others : 0) / slab * slab)
        if target < slots.byteLimit {
            try validateLimit(target)
            _ = resizeLocked(to: target, retired: &retired)
        } else if slots.pendingTrim {
            core.retireSlabsLocked(sizeClass: 0, maximumSlabs: .max, retired: &retired)
        }
        return before - slots.allocated + prepared
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
        guard slots.pendingTrim, !core.slabGrowthHeld else { return }
        core.retireSlabsLocked(sizeClass: 0, maximumSlabs: 8, retired: &retired)
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
        guard slots.quiescent else {
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
        slots.byteLimit = bytes
        dropGrowthLocked(retired: &retired)
        core.retireSlabsLocked(sizeClass: 0, maximumSlabs: .max, retired: &retired)
        return slots.allocated
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
        let (room, overflow) = slots.allocated.addingReportingOverflow(available)
        let slab = UInt64(slotsPerSlab * PackStore.expertBytes)
        let target = min(ceiling, overflow ? .max : room) / slab * slab
        if target > slots.byteLimit {
            slots.byteLimit = target
            slots.pendingTrim = slots.allocated > slots.byteLimit
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
        retired += preparedGrowth
        preparedGrowth.removeAll()
        growthEpoch &+= 1
    }
    /// Mid-reply growth, off the token loop (any thread): allocates, charges and faults (or wires) the whole slabs that fit
    /// both the room left in the ledger's budget and the growth ceiling, without touching the cache. Stops at the host
    /// floor or on fresh swap, like `prepareSlabs`. Publishes at most 128 MiB per tranche (one slab minimum), pausing
    /// the utility worker between tranches so recovery does not prefault a whole GB at once. No token thread sleeps.
    /// `attachGrowth` adds a bounded tranche at a token boundary. Returns total slabs prepared by this pass.
    @discardableResult package func prepareGrowth(minimumAvailableBytes: UInt64, wire: Bool) throws -> Int {
        let bytes = UInt64(slotsPerSlab * PackStore.expertBytes)
        lock.lock()
        let epoch = growthEpoch
        let top = max(slots.allocated, slots.byteLimit) + UInt64(preparedGrowth.count) * bytes
        // Lazy growth may still reserve up to the current limit: that room is not ours.
        let reserved = ledger.reservedBytes + (slots.byteLimit > slots.allocated ? slots.byteLimit - slots.allocated : 0)
        let usable = ledger.usableBytes
        // The phase's own split of a smaller budget: the rest of the model keeps what the phase left it (KV and owners grow later).
        let ceiling = min(growthCeiling, usable > growthOthers ? usable - growthOthers : 0)
        let count =
            growing || slots.pendingTrim || !ledger.optionalGrowthAllowed
            ? 0 : min(usable > reserved ? (usable - reserved) / bytes : 0, ceiling > top ? (ceiling - top) / bytes : 0)
        if count > 0 { growing = true }
        lock.unlock()
        guard count > 0 else { return 0 }
        var made: [ExpertSlabStorage] = []
        defer {
            lock.lock()
            growing = false
            if epoch == growthEpoch && ledger.optionalGrowthAllowed {
                preparedGrowth += made
                made = []
            }
            lock.unlock()
        }
        let batchSize = max(1, Int(Self.growthBatchBytes / bytes))
        var total = 0
        let page = Int(getpagesize())
        let swapStart = origami_swapout_bytes()
        func current() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return epoch == growthEpoch && !slots.pendingTrim
        }
        func hostAllowsGrowth() -> Bool {
            guard ledger.optionalGrowthAllowed else { return false }
            let reserved = ledger.reservedBytes
            let usable = ledger.usableBytes
            guard reserved <= usable, bytes <= usable - reserved else { return false }
            if ledger.usesAvailableMemoryFloor { return origami_available_memory() >= minimumAvailableBytes + bytes }
            return MemoryPressureLevel.current()?.isElevated != true && ledger.physicalAllocationShortfall(bytes: bytes) == 0
        }
        while total < count, current(), hostAllowsGrowth(), origami_swapout_bytes() == swapStart {
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
                made.append(storage)
                return true
            }
            guard admitted else { break }
            total += 1
            if made.count >= batchSize {
                lock.lock()
                let publish = epoch == growthEpoch && !slots.pendingTrim && ledger.optionalGrowthAllowed
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
        guard ledger.optionalGrowthAllowed, !core.slabGrowthHeld, !slots.pendingTrim, !preparedGrowth.isEmpty else { return 0 }
        let bytes = UInt64(slotsPerSlab * PackStore.expertBytes)
        let added = min(preparedGrowth.count, max(1, Int(Self.growthBatchBytes / bytes)))
        let limit = max(slots.byteLimit, slots.allocated)
        for storage in preparedGrowth.prefix(added) { core.insertSlabLocked(storage, sizeClass: 0) }
        preparedGrowth.removeFirst(added)
        slots.byteLimit = min(limit + UInt64(added * slotsPerSlab * PackStore.expertBytes), maximumByteLimit)
        return added
    }
    /// Slots the current limit allows; the memory governor may lower it while a phase runs.
    package var slotLimit: Int {
        lock.lock()
        defer { lock.unlock() }
        return Int(slots.byteLimit) / PackStore.expertBytes
    }

    /// The residency table (allocated on first use): per key, slot + 1 and epoch; per slot, its GPU address.
    package func gpuTable() throws -> (keys: MTLBuffer, slots: MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }
        if let keys = residency.gpuKeys, let addresses = residency.gpuSlots { return (keys, addresses) }
        let table = try residency.table(device: device, ledger: ledger, slots: slots)
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
        return (core.slabGeneration, slots.slabs.values.map(\.buffer))
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let all = slots.slots
        return Snapshot(
            allocatedBytes: slots.allocated, byteLimit: slots.byteLimit, maximumByteLimit: maximumByteLimit, slabs: slots.slabs.count,
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
    }
}
