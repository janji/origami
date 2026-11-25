import Foundation
import Metal

/// Qwen's replica arena: 32 records of one expert each.
extension ExpertReplicaArena {
    package static let recordCount = 32
    package static var admittedBytes: UInt64 { chargedBytes(records: recordCount, recordBytes: PackStore.expertBytes) }
    package convenience init(ledger: MemoryLedger, device: MTLDevice) throws {
        try self.init(ledger: ledger, device: device, records: Self.recordCount, recordBytes: PackStore.expertBytes)
    }
    package func acquire() -> ExpertReplicaLease? { acquire(bytes: PackStore.expertBytes) }
}
extension ExpertReplicaLease {
    package var expertBuffer: ExpertBuffer { ExpertBuffer(replicaBuffer: metalBuffer, offset: offset) }
}

/// Owns one consumer pin. Retire only after the last CPU read / GPU command
/// using this lease completes. Dropping a live lease deliberately does not
/// unpin a GPU-visible slot early; request cancellation must retire its leases.
public final class LoadedExpert: @unchecked Sendable {
    public let ticket: ExpertTicket
    public let buffer: ExpertBuffer
    private let cache: ExpertCache?
    private var replicaLease: ExpertReplicaLease?
    private let lock = NSLock()
    private var retired = false

    fileprivate init(ticket: ExpertTicket, buffer: ExpertBuffer, cache: ExpertCache) {
        self.ticket = ticket
        self.buffer = buffer
        self.cache = cache
    }
    fileprivate init(ticket: ExpertTicket, replica: ExpertReplicaLease) {
        self.ticket = ticket
        buffer = replica.expertBuffer
        cache = nil
        replica.pinConsumer()
        replicaLease = replica
    }

    public func retireAfterGPUCompletion() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !retired else { throw EngineError("Expert lease already retired") }
        if let replicaLease {
            try replicaLease.retireConsumer()
            self.replicaLease = nil
        } else if let cache {
            try cache.unpinAfterGPUCompletion(ticket)
        }
        retired = true
    }
}

/// The single producer of one expert's cache slot: its scheduler ticket or replica race, and the host
/// handoff pin that keeps the completed record resident while any `ExpertLoadRequest` handle lives.
private final class ExpertLoadEntry: @unchecked Sendable {
    let cache: ExpertCache
    let cacheTicket: ExpertTicket
    let residentSource: IOSource
    var source: IOSource { io?.source ?? residentSource }
    let generation: IOScheduler.Generation
    let residentAtRequest: Bool
    /// A demand read issued while no other demand read was in flight on a drive, of a record both drives hold: its
    /// wait counts in `split_reads`' `lone_demand_wait_seconds`, split or not.
    var lone = false
    /// A demand read (or a prediction demand adopted): what `lone` is judged against.
    var demand = false
    // These fields are loader-lock-owned and become stable before IOTicket.join.
    var io: IOTicket?
    var replica: ExpertReadReplica?
    private let handoffLock = NSLock()
    private var handles = 0
    private var handoffPin: ExpertBuffer?
    var failure: String?

    init(cache: ExpertCache, ticket: ExpertTicket, source: IOSource, generation: IOScheduler.Generation, residentAtRequest: Bool) {
        self.cache = cache
        self.cacheTicket = ticket
        self.residentSource = source
        self.generation = generation
        self.residentAtRequest = residentAtRequest
    }

    func retainHandle() {
        handoffLock.lock()
        handles += 1
        handoffLock.unlock()
    }
    func releaseHandle() {
        handoffLock.lock()
        defer { handoffLock.unlock() }
        precondition(handles > 0)
        handles -= 1
        if handles == 0 && handoffPin != nil {
            try? cache.unpinAfterGPUCompletion(cacheTicket)
            handoffPin = nil
        }
    }
    func installHandoffPin() throws {
        handoffLock.lock()
        defer { handoffLock.unlock() }
        // A dropped speculative request needs no host pin after completion.
        // Scheduler callback metadata can outlive join briefly; it must not
        // retain a slot after the final actual request handle disappears.
        if handles > 0 && handoffPin == nil { handoffPin = try cache.pin(cacheTicket) }
    }
    var hasHandoffPin: Bool {
        handoffLock.lock()
        defer { handoffLock.unlock() }
        return handoffPin != nil
    }
    /// Live ExpertLoadRequest handles sharing this producer (read hygiene's sharing guard).
    var handleCount: Int {
        handoffLock.lock()
        defer { handoffLock.unlock() }
        return handles
    }
}

/// A caller's handle on one in-flight or resident expert; requests for the same key share one entry.
public final class ExpertLoadRequest: @unchecked Sendable {
    fileprivate let entry: ExpertLoadEntry
    fileprivate init(_ entry: ExpertLoadEntry) {
        self.entry = entry
        entry.retainHandle()
    }
    deinit { entry.releaseHandle() }
    public var key: ExpertKey { entry.cacheTicket.key }
    public var generation: IOScheduler.Generation { entry.generation }
    public var source: IOSource { entry.source }
    /// Immutable classification at acquisition, not an unsynchronized I/O poll.
    /// A resident request already owns its verified producer handoff pin.
    package var residentAtRequest: Bool { entry.residentAtRequest }
}

/// One producer per expert, exact per-drive scheduling and joined cancellation.
/// This loader must own all cache read production; consumers may use returned
/// leases and resize the cache but must not overwrite/cancel its cache tickets.
public final class SourceExpertLoader: @unchecked Sendable {
    public typealias Reader = @Sendable (ExpertKey, IOSource, UnsafeMutableRawBufferPointer) throws -> Void
    private let cache: ExpertCache
    private let scheduler: IOScheduler
    private let reader: Reader
    private let packedAvailable: @Sendable () -> Bool
    private let hotInternal: @Sendable (ExpertKey) -> Bool
    private let lock = NSLock()
    private var inFlight: [ExpertKey: ExpertLoadEntry] = [:]
    private let replicaArena: ExpertReplicaArena?
    /// Online source calibration; nil keeps the static calibration path unchanged.
    private let estimator: ExpertSourceOnlineEstimator?
    private var replicasEnabled = false
    /// Whether the executor is in its decode phase: only decode's lone waits are counted (`setDecoding`).
    private var decoding = false
    // Published requests may disappear from inFlight while their losing pread
    // remains live. Retain these races until the scheduler's actual joins pass.
    private var readRaces: [ExpertTicket: ExpertReadReplica] = [:]
    /// Part reads for split reads (`SplitReads`); nil: every read is whole.
    private var splitReader: SplitPartReader?
    package let splitReads = SplitReadCounters()
    /// Whether demand reads may be split or sent whole to another source (`SplitReads.enabled` at start; checks turn it
    /// on and off). Off, a lone read is still counted (`lone_demand_waits`) and nothing else is planned or allocated.
    package var splitsReads = SplitReads.enabled
    /// The bytes planned on each source of a split read and not yet read (`planSplit`).
    package let plannedBytes = SplitPlannedBytes()
    /// Staging for split reads when no replica arena exists (replicas off, as on one drive with a hot copy): the experts
    /// one token activates in a layer, since every miss of a layer may split; charged to the ledger at the first split
    /// (ten records), never before.
    private lazy var splitArena: ExpertReplicaArena? =
        replicaArena
        ?? (try? ExpertReplicaArena(
            ledger: cache.ledger, device: cache.device, records: ModelContract.geometry["num_experts_per_tok"]!,
            recordBytes: PackStore.expertBytes, lazily: true))

    /// How a split read reads parts of a record from either drive: whether both drives hold `key` in a form parts can be
    /// read from, whether a drive's read tally says it is wearing, what a source's own reads measured, a part read (its
    /// checks left in `PartChecks`), and the assembled record's check.
    package struct SplitPartReader: Sendable {
        /// A record's sources now (none, or one: read whole).
        let sources: @Sendable (ExpertKey) -> [PackStore.SplitSource]
        let wearing: @Sendable (ExpertKey, PackStore.SplitSource) -> Bool
        /// A source's live rate from its own completed reads, whole and part (0: not measured yet), the seconds since a
        /// read last fed it, and its healthy rate (0: unknown).
        let measured: @Sendable (ExpertKey, PackStore.SplitSource) -> (bytesPerSecond: Double, secondsAgo: Double, healthy: Double)
        let read: @Sendable (ExpertKey, PackStore.SplitSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws -> Void
        let verify: @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        package init(
            sources: @escaping @Sendable (ExpertKey) -> [PackStore.SplitSource],
            wearing: @escaping @Sendable (ExpertKey, PackStore.SplitSource) -> Bool,
            measured:
                @escaping @Sendable (ExpertKey, PackStore.SplitSource) -> (bytesPerSecond: Double, secondsAgo: Double, healthy: Double) =
                { _, _ in (0, .infinity, 0) },
            read:
                @escaping @Sendable (ExpertKey, PackStore.SplitSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws ->
                Void,
            verify: @escaping @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        ) {
            self.sources = sources
            self.wearing = wearing
            self.measured = measured
            self.read = read
            self.verify = verify
        }
        /// Two sources, the model's drive and the pack's (checks): `available` says whether both hold the record.
        package init(
            available: @escaping @Sendable (ExpertKey) -> Bool, wearing: @escaping @Sendable (ExpertKey, IOSource) -> Bool,
            measured: @escaping @Sendable (IOSource) -> (bytesPerSecond: Double, secondsAgo: Double, healthy: Double) = { _ in
                (0, .infinity, 0)
            },
            read: @escaping @Sendable (ExpertKey, IOSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws -> Void,
            verify: @escaping @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        ) {
            @Sendable func queue(_ source: PackStore.SplitSource) -> IOSource {
                if case .pack = source { .externalSSD } else { .internalSSD }
            }
            self.init(
                sources: { available($0) ? [.model, .pack] : [] }, wearing: { wearing($0, queue($1)) }, measured: { measured(queue($1)) },
                read: { key, source, offset, into, checks in try read(key, queue(source), offset, into, checks) }, verify: verify)
        }
        /// `store`'s part reads: the checkpoint's spans, each hot copy place holding the record and the pack queue's
        /// copies (`PackStore.splitSources`, the live lookups). Each is measured by its own reads alone: the model's
        /// drive by its slow-read tally (every read of the model's files, whole or part), a place and a pack copy by
        /// their lanes (`ReadLane.measured`).
        package init(store: PackStore) {
            self.init(
                sources: { store.splitSources($0) },
                wearing: { key, source in
                    switch source {
                    case .model: store.modelDriveReads.wearing
                    case .place(let set, let place): set.stores[place].reads.wearing
                    case .pack: false
                    }
                },
                measured: { key, source in
                    switch source {
                    case .model:
                        let rates = store.modelDriveReads.rates
                        return (rates.current, store.modelDriveReads.age, rates.healthy)
                    case .place(let set, let place):
                        let lane = set.lane(place).measured
                        return (lane?.bytesPerSecond ?? 0, lane?.secondsAgo ?? .infinity, set.stores[place].reads.rates.healthy)
                    case .pack:
                        let lane = store.packLane(key)?.measured
                        return (lane?.bytesPerSecond ?? 0, lane?.secondsAgo ?? .infinity, 0)
                    }
                },
                read: { key, source, offset, into, checks in
                    try store.readExpertPart(key, source: source, at: offset, into: into, checks: checks)
                }, verify: { key, record, checks in try store.verifySplit(key, record: record, checks: checks) })
        }
    }
    /// Sets `reader`'s part reads (before serving): they find a record's sources, so lone reads are counted, and split
    /// the read (or send it whole to another source) when `splitsReads`.
    package func enableSplitReads(_ reader: SplitPartReader) {
        lock.lock()
        splitReader = reader
        lock.unlock()
    }

    public init(
        cache: ExpertCache, scheduler: IOScheduler,
        packedAvailable: @escaping @Sendable () -> Bool = { true },
        hotInternal: @escaping @Sendable (ExpertKey) -> Bool = { _ in false }, reader: @escaping Reader
    ) {
        self.cache = cache
        self.scheduler = scheduler
        self.packedAvailable = packedAvailable
        self.hotInternal = hotInternal
        self.reader = reader
        replicaArena = nil
        estimator = nil
    }

    /// Original, mirror and packed reads train separate service curves. Queue admission
    /// consumes a frozen curve, with no estimator lock held by the scheduler.
    package init(
        cache: ExpertCache, scheduler: IOScheduler, replicaArena: ExpertReplicaArena?,
        estimator: ExpertSourceOnlineEstimator? = nil,
        packedAvailable: @escaping @Sendable () -> Bool,
        hotInternal: @escaping @Sendable (ExpertKey) -> Bool = { _ in false }, reader: @escaping Reader
    ) {
        self.cache = cache
        self.scheduler = scheduler
        self.replicaArena = replicaArena
        self.estimator = estimator
        self.packedAvailable = packedAvailable
        self.hotInternal = hotInternal
        let readTimes = readTimes
        self.reader = { [scheduler] key, source, destination in
            let mirrored = source == .internalSSD && hotInternal(key)
            let concurrency = scheduler.runningCount(source)
            let start = DispatchTime.now().uptimeNanoseconds
            try reader(key, source, destination)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            readTimes.record(source == .externalSSD ? 0 : mirrored ? 2 : 1, seconds: seconds)
            estimator?.observe(
                source: source, bytes: PackStore.expertBytes, seconds: seconds, concurrency: concurrency,
                hotInternal: mirrored && hotInternal(key))
        }
    }

    /// Record reads by drive (external, internal original, hot copy): count, seconds and service-time buckets
    /// (under 1 ms, 1-2, 2-4, 4-8, 8-16, 16-32, 32-64 and from 64 ms). Any thread.
    package final class ReadTimes: @unchecked Sendable {
        private let lock = NSLock()
        private var values = [(count: UInt64, seconds: Double, buckets: [UInt64])](
            repeating: (0, 0, [UInt64](repeating: 0, count: 8)), count: 3)
        func record(_ drive: Int, seconds: Double) {
            let bucket = min(7, max(0, Int(floor(log2(max(seconds * 1000, 0.5)))) + 1))
            lock.lock()
            values[drive].count &+= 1
            values[drive].seconds += seconds
            values[drive].buckets[bucket] &+= 1
            lock.unlock()
        }
        package var snapshot: [(count: UInt64, seconds: Double, buckets: [UInt64])] {
            lock.lock()
            defer { lock.unlock() }
            return values
        }
    }
    package let readTimes = ReadTimes()

    package func onlineCalibrationSnapshot() -> ExpertSourceOnlineSnapshot? { estimator?.snapshot() }

    /// Joined phase boundary only: the cache retains this immutable, bounded array, never a callback.
    package func storageCostMultipliers(calibration: ExpertSourceCalibration) throws -> [Double] {
        let original = estimator?.placementCalibration(hotInternal: false, recordDecision: false) ?? calibration
        let mirror = estimator?.placementCalibration(hotInternal: true, recordDecision: false) ?? calibration
        let packed = packedAvailable()
        func seconds(_ curve: ExpertSourceCalibration, _ source: IOSource) -> Double {
            curve.estimatedSeconds(
                source: source, bytesIncludingNew: UInt64(PackStore.expertBytes), jobsIncludingNew: 1, maximumConcurrent: 1)
        }
        let external = packed ? seconds(original, .externalSSD) : Double.infinity
        let baseline = min(seconds(original, .internalSSD), external)
        let copied = min(seconds(mirror, .internalSSD), external)
        let multiplier = baseline > 0 && copied.isFinite ? min(1, max(0.125, copied / baseline)) : 1
        return try (0..<(ModelContract.layers * ModelContract.experts)).map { index in
            let key = try ExpertKey(layer: index / ModelContract.experts, expert: index % ModelContract.experts)
            return hotInternal(key) ? multiplier : 1
        }
    }

    /// The executor's phase at its boundary: `lone_demand_wait_seconds` counts decode's waits only.
    package func setDecoding(_ decode: Bool) {
        lock.lock()
        decoding = decode
        lock.unlock()
    }

    package func setReplicasEnabled(_ enabled: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        readRaces = readRaces.filter { !$0.value.isRetired }
        guard inFlight.isEmpty, readRaces.isEmpty, (replicaArena?.snapshot().consumerPins ?? 0) == 0 else {
            throw EngineError("Replica phase requires joined read producers and retired consumer pins")
        }
        replicasEnabled = enabled && replicaArena != nil
    }

    /// A losing read still running at `deadline` (by default the demand read bound from now) stays retained.
    package func joinRetainedReads(until deadline: Date? = nil) {
        let deadline = deadline ?? Date(timeIntervalSinceNow: Self.stallBound)
        lock.lock()
        let current = Array(readRaces.values)
        lock.unlock()
        for race in current { race.joinAll(until: deadline) }
        lock.lock()
        readRaces = readRaces.filter { !$0.value.isRetired }
        lock.unlock()
    }

    package func replicaSnapshot() -> ExpertReplicaSnapshot? { replicaArena?.snapshot() }

    private func publish(
        _ entry: ExpertLoadEntry, key: ExpertKey,
        generation: IOScheduler.Generation, outcome: IOOutcome, retireEntry: Bool = true
    ) -> IOOutcome {
        lock.lock()
        defer { lock.unlock() }
        let valid =
            outcome == .succeeded(bytes: PackStore.expertBytes)
            && scheduler.currentGeneration == generation && entry.replica?.isCancelled != true
        var result = outcome
        if cache.completeRead(entry.cacheTicket, verified: valid) {
            do { try entry.installHandoffPin() } catch {
                entry.failure = "Cannot retain completed expert: \(error)"
                result = .failed(entry.failure!)
            }
        } else if case .failed(let reason) = outcome {
            entry.failure = reason
        } else {
            result = .cancelled
        }
        if retireEntry, inFlight[key] === entry { inFlight.removeValue(forKey: key) }
        return result
    }

    private func releaseEntry(_ entry: ExpertLoadEntry, key: ExpertKey) {
        lock.lock()
        defer { lock.unlock() }
        if inFlight[key] === entry { inFlight.removeValue(forKey: key) }
    }

    private func offerReplicas() {
        lock.lock()
        let pending = inFlight.values.compactMap { $0.replica }
        lock.unlock()
        for race in pending { race.tryReplica() }
    }

    public convenience init(store: PackStore, cache: ExpertCache, scheduler: IOScheduler) {
        self.init(
            cache: cache, scheduler: scheduler, packedAvailable: { store.packedAvailable },
            hotInternal: { store.hasHotExpert($0) }
        ) { key, source, destination in
            if source == .internalSSD {
                try store.readInternalExpert(key, into: destination)
            } else {
                try store.readExpert(layer: key.layer, expert: key.expert, into: destination, source: .packed)
            }
        }
    }

    /// A returned request keeps its completed record pinned until the request
    /// is released, even before joinAndPin. This avoids an eviction race between
    /// I/O completion and demand consumption. Retain prefetch requests only for
    /// their intended scope; dropping them leaves the cache entry evictable.
    public func request(
        _ key: ExpertKey, generation: IOScheduler.Generation,
        priority: IOPriority = .demand, source preferredSource: IOSource? = nil,
        policy: ExpertSourcePolicy = .packedPreferred, calibration: ExpertSourceCalibration = .neutral
    ) throws -> ExpertLoadRequest {
        let explicitSource = preferredSource
        let preferredSource =
            preferredSource ?? ((policy == .packedPreferred || estimator == nil) && hotInternal(key) ? IOSource.internalSSD : nil)
        lock.lock()
        defer { lock.unlock() }
        readRaces = readRaces.filter { !$0.value.isRetired }
        guard scheduler.currentGeneration == generation else { throw EngineError("Stale expert-load generation") }
        // A race that published its failure is only waiting to leave inFlight (its cache slot is already
        // free); adopting it would fail this demand, so start a fresh hedged read instead.
        if let pending = inFlight[key], case .failed? = pending.replica?.wait(timeout: 0) { inFlight.removeValue(forKey: key) }
        if let pending = inFlight[key] {
            guard pending.generation == generation, let io = pending.io else {
                throw EngineError("Older expert read must join before this generation can reuse its slot")
            }
            // Authoritative demand adopts a pre-routed read in flight: one
            // producer, one slot; the queued job is promoted and may hedge.
            if priority == .demand {
                _ = scheduler.promote(io)
                pending.demand = true
                pending.replica?.markDemanded()
            }
            return ExpertLoadRequest(pending)
        }
        let acquisition = try cache.acquire(key)
        let source = preferredSource ?? (packedAvailable() ? .externalSSD : .internalSSD)
        let entry = ExpertLoadEntry(
            cache: cache, ticket: acquisition.ticket, source: source,
            generation: generation, residentAtRequest: !acquisition.newlyReserved)
        entry.demand = priority == .demand
        let handle = ExpertLoadRequest(entry)
        if !acquisition.newlyReserved {
            guard [.verified, .evictable, .pinned].contains(acquisition.phase) else {
                throw EngineError("Cache read producer exists outside the expert loader")
            }
            try entry.installHandoffPin()
            return handle
        }
        return try startRead(
            entry, handle: handle, key: key, acquisition: acquisition, generation: generation,
            priority: priority, preferredSource: preferredSource, source: source, policy: policy, calibration: calibration,
            speculative: false, explicitSource: explicitSource)
    }

    /// Pre-routed (speculative) admission: nothing is issued for a resident or
    /// in-flight record, a missing slot or a refused queue; the outcome is
    /// returned for the caller's counters. An issued read is the ordinary
    /// producer (same cache slot, reader, source policy and replica race) at
    /// prefetch priority, withheld from the hedge until demand shares it.
    /// `overrideScore` passes to ExpertCache.acquireSpeculative; such a read is `overridden`.
    package enum SpeculativeRequest {
        case issued(ExpertLoadRequest)
        case overridden(ExpertLoadRequest)
        case resident, inFlight, noSlot
        case refused(String)
    }
    package func requestSpeculative(
        _ key: ExpertKey, generation: IOScheduler.Generation, source preferredSource: IOSource? = nil,
        policy: ExpertSourcePolicy = .packedPreferred, calibration: ExpertSourceCalibration = .neutral,
        overrideScore: Bool = false
    ) throws -> SpeculativeRequest {
        let preferredSource =
            preferredSource ?? ((policy == .packedPreferred || estimator == nil) && hotInternal(key) ? IOSource.internalSSD : nil)
        lock.lock()
        defer { lock.unlock() }
        readRaces = readRaces.filter { !$0.value.isRetired }
        guard scheduler.currentGeneration == generation else { throw EngineError("Stale expert-load generation") }
        if inFlight[key] != nil { return .inFlight }
        let acquisition: ExpertCache.Acquisition
        let overridden: Bool
        switch try cache.acquireSpeculative(key, overrideScore: overrideScore) {
        case .resident: return .resident
        case .inFlight: return .inFlight
        case .noSlot: return .noSlot
        case .admitted(let admitted):
            acquisition = admitted
            overridden = false
        case .overridden(let admitted):
            acquisition = admitted
            overridden = true
        }
        let source = preferredSource ?? (packedAvailable() ? .externalSSD : .internalSSD)
        let entry = ExpertLoadEntry(
            cache: cache, ticket: acquisition.ticket, source: source, generation: generation, residentAtRequest: false)
        let handle = ExpertLoadRequest(entry)
        do {
            let request = try startRead(
                entry, handle: handle, key: key, acquisition: acquisition, generation: generation,
                priority: .prefetch, preferredSource: preferredSource, source: source, policy: policy, calibration: calibration,
                speculative: true)
            return overridden ? .overridden(request) : .issued(request)
        } catch let error as EngineError { return .refused(String(describing: error)) }
    }

    /// Loader lock held. The reservation is released on any failure before it throws.
    private func startRead(
        _ entry: ExpertLoadEntry, handle: ExpertLoadRequest, key: ExpertKey, acquisition: ExpertCache.Acquisition,
        generation: IOScheduler.Generation, priority: IOPriority, preferredSource: IOSource?, source: IOSource,
        policy: ExpertSourcePolicy, calibration: ExpertSourceCalibration, speculative: Bool, explicitSource: IOSource? = .internalSSD
    ) throws -> ExpertLoadRequest {
        let destination: ExpertBuffer
        do { destination = try cache.beginRead(acquisition.ticket) } catch {
            try? cache.cancel(acquisition.ticket)
            throw error
        }
        var planned: SplitPlannedRead?
        do {
            let mirrored = hotInternal(key)
            let selecting = preferredSource == nil && policy == .earliestCompletion
            let placement = estimator?.placementCalibration(hotInternal: mirrored, recordDecision: selecting) ?? calibration
            let probe = selecting && priority == .demand ? estimator?.idleProbeSource(hotInternal: mirrored) : nil
            // Any demand read whose caller named no drive (the hot-copy rule's default included): every source of the
            // record is a candidate (`PackStore.splitSources`), with any source policy. Every read is counted on the
            // source that reads it when that is known.
            let split = planSplit(
                key, entry: entry, calibration: calibration, normal: selecting ? nil : source,
                routed: explicitSource == nil && priority == .demand && !speculative)
            planned = split?.planned
            entry.lone = split?.lone ?? false
            let parts = split?.parts
            let splitSource: IOSource? = parts.map { split!.queues[$0[0].source] }
            let splitRead: (@Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void)? = parts.map { parts in
                splitOperation(
                    key, entry: entry, parts: parts, order: split!.order, sources: split!.sources, queues: split!.queues,
                    calibration: calibration, planned: split!.planned)
            }
            let reader: Reader = { [reader] key, source, destination in
                if let splitRead, source == splitSource { try splitRead(source, destination) } else { try reader(key, source, destination) }
            }
            if replicasEnabled, let arena = replicaArena {
                let race = ExpertReadReplica(
                    scheduler: scheduler, arena: arena, generation: generation, recordBytes: PackStore.expertBytes,
                    read: { [reader] source, bytes in try reader(key, source, bytes) },
                    completePrimary: { [weak self, weak entry] outcome in
                        guard let self, let entry else { return .cancelled }
                        return self.publish(entry, key: key, generation: generation, outcome: outcome, retireEntry: false)
                    },
                    releaseEntry: { [weak self, weak entry] in
                        guard let self, let entry else { return }
                        self.releaseEntry(entry, key: key)
                    })
                entry.replica = race
                if speculative { race.markSpeculative() }
                let available: [IOSource] = packedAvailable() ? [.externalSSD, .internalSSD] : [.internalSSD]
                let base =
                    calibration.replicaDelaySeconds ?? 4
                    * calibration.estimatedSeconds(
                        source: .externalSSD,
                        bytesIncludingNew: UInt64(PackStore.expertBytes), jobsIncludingNew: 1, maximumConcurrent: 1)
                entry.io = try race.start(
                    destination: destination.bytes, available: available,
                    delay: { [estimator] primary in
                        estimator?.hedgeDelaySeconds(base: base, primary: primary, hotInternal: mirrored) ?? base
                    }
                ) { [planned] operation, primaryCompletion in
                    // The read's planned bytes end with its job, whether it ran or not.
                    let completion: @Sendable (IOTicket, IOOutcome) -> Void = { ticket, outcome in
                        planned?.endAll()
                        primaryCompletion(ticket, outcome)
                    }
                    if splitSource == nil && preferredSource == nil && policy == .earliestCompletion {
                        return try scheduler.submitEarliest(
                            availableSources: available, estimate: placement.estimatedSeconds, idleProbeSource: probe,
                            priority: priority, generation: generation, byteCount: PackStore.expertBytes,
                            operation: operation, completion: completion)
                    }
                    let selected = splitSource ?? preferredSource ?? available[0]
                    return try scheduler.submit(
                        source: selected, priority: priority, generation: generation,
                        byteCount: parts?[0].length ?? PackStore.expertBytes, operation: { try operation(selected, $0) },
                        completion: { ticket, outcome in
                            // A split's job is charged its own part's bytes; its success is the whole record's.
                            let whole = parts != nil && outcome == .succeeded(bytes: parts![0].length)
                            completion(ticket, whole ? .succeeded(bytes: PackStore.expertBytes) : outcome)
                        })
                }
                inFlight[key] = entry
                readRaces[entry.cacheTicket] = race
                return handle
            }
            let operation: @Sendable (IOSource, IOCancellation) throws -> Void = { [reader] source, cancellation in
                try cancellation.throwIfCancelled()
                try reader(key, source, destination.bytes)
                try cancellation.throwIfCancelled()
            }
            let completion: @Sendable (IOTicket, IOOutcome) -> Void = { [self, entry, planned] _, outcome in
                planned?.endAll()
                _ = publish(entry, key: key, generation: generation, outcome: outcome)
            }
            if let splitSource, let parts {
                entry.io = try scheduler.submit(
                    source: splitSource, priority: priority, generation: generation, byteCount: parts[0].length,
                    operation: { cancellation in try operation(splitSource, cancellation) },
                    completion: { ticket, outcome in
                        // A split's job is charged its own part's bytes; its success is the whole record's.
                        completion(
                            ticket, outcome == .succeeded(bytes: parts[0].length) ? .succeeded(bytes: PackStore.expertBytes) : outcome)
                    })
            } else if preferredSource == nil && policy == .earliestCompletion {
                entry.io = try scheduler.submitEarliest(
                    availableSources: packedAvailable() ? [.externalSSD, .internalSSD] : [.internalSSD],
                    estimate: placement.estimatedSeconds, idleProbeSource: probe, priority: priority, generation: generation,
                    byteCount: PackStore.expertBytes,
                    operation: operation, completion: completion)
            } else {
                entry.io = try scheduler.submit(
                    source: source, priority: priority, generation: generation, byteCount: PackStore.expertBytes,
                    operation: { cancellation in try operation(source, cancellation) }, completion: completion)
            }
            inFlight[key] = entry
            return handle
        } catch {
            planned?.endAll()
            _ = cache.completeRead(acquisition.ticket, verified: false)
            throw error
        }
    }

    /// A read's plan (`planSplit`): whether it is lone, its parts (a split, or one part: the whole record sent to another
    /// source than the normal path's; nil: read on the normal path), the sources fastest first, the sources and each
    /// one's queue, and the bytes it planned on them.
    private struct SplitDecision {
        let lone: Bool
        let parts: [SplitReadPart]?
        let order: [Int]
        let sources: [PackStore.SplitSource]
        let queues: [IOSource]
        let planned: SplitPlannedRead
    }

    /// The plan for read `key` (loader lock held), nil without part reads. `normal`: the queue the normal path reads it
    /// on (nil: the scheduler chooses); `routed`: a demand read whose caller named no drive (only those are planned).
    /// With more than one source (`SplitPartReader.sources`): whether it is lone (decode, no other demand read of this
    /// loader in flight) and, with `splitsReads`, the plan: per source its own measured rate (`SplitPartReader.measured`,
    /// no latency) and that rate's age, else the static calibration's one-read rate and latency, never measured (the
    /// model's files and the pack on the original curve, a place on the mirror curve); healthy from its own tally, else
    /// the static calibration's; busy for the bytes planned on it and not yet read (`plannedBytes`) at that rate. The
    /// plan's whole read goes on the normal path, unless the plan found the model's files or the pack best while the
    /// normal path reads another known source: then it is a one-part read there. The bytes are counted on each part's
    /// source, else on the normal path's source when it is known (the model's files, the pack, or the only place
    /// holding a hot record; several places: not counted, the copies choose one at the read).
    private func planSplit(
        _ key: ExpertKey, entry: ExpertLoadEntry, calibration: ExpertSourceCalibration, normal: IOSource?, routed: Bool
    ) -> SplitDecision? {
        guard let splitReader else { return nil }
        let record = PackStore.expertBytes
        let sources = splitReader.sources(key)
        let queues = sources.map { source -> IOSource in if case .pack = source { .externalSSD } else { .internalSSD } }
        let places = sources.indices.filter { if case .place = sources[$0] { true } else { false } }
        let normalSource: Int? =
            switch normal {
            case .externalSSD?: sources.firstIndex { if case .pack = $0 { true } else { false } }
            case .internalSSD? where hotInternal(key): places.count == 1 ? places[0] : nil
            case .internalSSD?: sources.firstIndex { if case .model = $0 { true } else { false } }
            case nil: nil
            }
        func decision(lone: Bool, parts: [SplitReadPart]?, order: [Int] = []) -> SplitDecision {
            let items =
                parts?.map { (Self.plannedKey(sources[$0.source]), $0.length) }
                ?? normalSource.map { [(Self.plannedKey(sources[$0]), record)] }
            return SplitDecision(
                lone: lone, parts: parts, order: order, sources: sources, queues: queues,
                planned: SplitPlannedRead(items ?? [], ledger: plannedBytes))
        }
        guard routed, sources.count >= 2 else { return decision(lone: false, parts: nil) }
        // A published race waits in `inFlight` only for its loser to end: not a demand read still in flight.
        let lone =
            decoding
            && !inFlight.values.contains { other in
                other.demand && other !== entry
                    && (other.replica.map { $0.wait(timeout: 0) == nil } ?? (other.io.map { $0.join(timeout: 0) == nil } ?? false))
            }
        guard splitsReads else { return decision(lone: lone, parts: nil) }
        let healthy = estimator?.calibration ?? calibration
        let original = estimator?.placementCalibration(hotInternal: false, recordDecision: false) ?? calibration
        let mirror = estimator?.placementCalibration(hotInternal: true, recordDecision: false) ?? calibration
        let described = sources.indices.map { n -> SplitReadSource in
            let measured = splitReader.measured(key, sources[n])
            let internalSource = queues[n] == .internalSSD
            let curve = places.contains(n) ? mirror : original
            var source = SplitReadSource(
                bytesPerSecond: measured.bytesPerSecond > 0
                    ? measured.bytesPerSecond : (internalSource ? curve.internalBytesPerSecond : curve.externalBytesPerSecond)[0],
                latencySeconds: measured.bytesPerSecond > 0
                    ? 0 : internalSource ? curve.internalLatencySeconds : curve.externalLatencySeconds,
                healthyBytesPerSecond: measured.healthy > 0
                    ? measured.healthy : (internalSource ? healthy.internalBytesPerSecond : healthy.externalBytesPerSecond)[0],
                measuredSecondsAgo: measured.bytesPerSecond > 0 ? measured.secondsAgo : .infinity,
                wearing: splitReader.wearing(key, sources[n]))
            source.busySeconds = Double(plannedBytes[Self.plannedKey(sources[n])]) / source.plannedBytesPerSecond
            return source
        }
        let plan = SplitReads.plan(recordBytes: record, sources: described, waited: true)
        splitReads.record(plan, sources: described, names: Self.splitNames(sources))
        let order = described.indices.sorted { described[$0].seconds(record) < described[$1].seconds(record) }
        switch plan {
        case .split(let parts): return decision(lone: lone, parts: parts, order: order)
        case .whole(let best?, _) where normalSource != nil && best != normalSource && !places.contains(best):
            splitReads.routedWhole()
            return decision(lone: lone, parts: [SplitReadPart(source: best, offset: 0, length: record)], order: order)
        case .whole: return decision(lone: lone, parts: nil)
        }
    }
    /// The ledger key of a split source: one per physical source (a place by its store).
    private static func plannedKey(_ source: PackStore.SplitSource) -> AnyHashable {
        switch source {
        case .model: "model"
        case .place(let set, let place): ObjectIdentifier(set.stores[place])
        case .pack: "pack"
        }
    }
    /// Each source's name in `split_reads`.
    private static func splitNames(_ sources: [PackStore.SplitSource]) -> [String] {
        sources.map { source -> String in
            switch source {
            case .model: "model_drive"
            case .place(_, let place): "copy\(place)"
            case .pack: "pack"
            }
        }
    }

    /// The read of a split record into its slot (`SplitReads.execute`): each other part on a thread of
    /// its own outside the queues (a part waiting for a queue slot that split reads' own jobs hold would never start), staged in a replica-arena record (`splitArena` without replicas; none free, or a one-part read: into the
    /// slot, unguarded).
    /// The fastest part is read in place on the job's thread (covered only by the replica race, if any); a late part is re-read
    /// after the replica hedge's static delay for its bytes; a staged part is copied in only while the race (if any)
    /// still owns the record; every part's time trains its queue's online estimate as a whole read's does (a place's
    /// lane too, in `HotCopySet.read`). Then the record is checked as a whole as its parts' sources check one. A split
    /// that fails reads the record again whole from `source`, unless a replica published it meanwhile. Each part's
    /// planned bytes end with its own read, the rest when the split returns (a staged loser still reading then is no
    /// longer counted). A one-part read is counted only as `whole_from_other_source`.
    private func splitOperation(
        _ key: ExpertKey, entry: ExpertLoadEntry, parts: [SplitReadPart], order: [Int], sources: [PackStore.SplitSource],
        queues: [IOSource], calibration: ExpertSourceCalibration, planned: SplitPlannedRead
    ) -> @Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void {
        let (reader, wholeRead, arena, counters, scheduler, estimator) = (
            splitReader!, self.reader, parts.count > 1 ? splitArena : nil, splitReads, scheduler, estimator
        )
        let partSources = sources
        let names = Self.splitNames(sources)
        let split = parts.count > 1
        return { [weak entry] source, destination in
            defer { planned.endAll() }
            let checks = PackStore.PartChecks()
            let staging = arena?.acquire()
            nonisolated(unsafe) let stagingBytes = staging?.bytes
            // Weak: the race holds this closure, and the entry the race.
            func owned(_ copy: () -> Void) -> Bool {
                entry?.replica?.whileOwned(copy)
                    ?? {
                        copy()
                        return true
                    }()
            }
            do {
                let outcome = try SplitReads.execute(
                    parts, into: destination, order: order, staging: stagingBytes, keep: staging,
                    lateAfter: { part in
                        calibration.replicaDelaySeconds.map { $0 * Double(part.length) / Double(PackStore.expertBytes) }
                            ?? 4
                            * calibration.estimatedSeconds(
                                source: .externalSSD, bytesIncludingNew: UInt64(part.length), jobsIncludingNew: 1, maximumConcurrent: 1)
                    }, owns: owned
                ) { part, offset, into in
                    // A part's own read (a guard or fallback of its range comes from another source).
                    let own = parts.firstIndex { $0.source == part && $0.offset == offset }
                    defer { if let own { planned.end(own) } }
                    let drive = queues[part]
                    let concurrency = scheduler.runningCount(drive)
                    let begun = DispatchTime.now().uptimeNanoseconds
                    try reader.read(key, partSources[part], offset, into, checks)
                    var hot = false
                    if case .place = partSources[part] { hot = true }
                    // Inside the read: a staged part that ends after its split trains its drive too.
                    estimator?.observe(
                        source: drive, bytes: into.count, seconds: Double(DispatchTime.now().uptimeNanoseconds - begun) / 1e9,
                        concurrency: concurrency, hotInternal: hot)
                }
                try reader.verify(key, UnsafeRawBufferPointer(destination), checks)
                if split { counters.record(outcome, names: names) }
            } catch let orphaned as SplitReadOrphaned {
                if split { counters.record(nil, names: []) }
                throw orphaned
            } catch let cancelled as IOOperationCancelled {
                // Its generation or scheduler ended: no whole read after it.
                throw cancelled
            } catch {
                if split { counters.record(nil, names: []) }
                guard owned({}) else { throw SplitReadOrphaned() }
                try wholeRead(key, source, destination)
            }
        }
    }

    /// How long a demand read may take (`ExpertReadStall`, as at 1 GB/s, as the keeper's wait).
    private static var stallBound: Double { ExpertReadStall.bound(bytes: PackStore.expertBytes, bytesPerSecond: 1e9) }

    /// Joins before acquiring the consumer pin. A failed external read is retried
    /// once through the internal queue, never inside an external-drive worker.
    /// Original-source errors remain explicit. No missing expert is substituted.
    /// A read not ended within the demand read bound from this wait's start (a queue behind a slow drive is not a stall)
    /// is cancelled and fails the request with `drive_stalled`: a hung pread cannot be stopped, so its slot stays
    /// `reading` until it returns, and its completion then frees it (ROADMAP R2).
    public func joinAndPin(_ request: ExpertLoadRequest, shouldContinue: () -> Bool = { true }) throws -> LoadedExpert {
        let entry = request.entry
        let deadline = Date(timeIntervalSinceNow: Self.stallBound)
        let waitStarted = DispatchTime.now().uptimeNanoseconds
        defer { if entry.lone { splitReads.waited(Double(DispatchTime.now().uptimeNanoseconds - waitStarted) / 1e9) } }
        if let race = entry.replica {
            var outcome: IOOutcome?
            while outcome == nil {
                if !shouldContinue() {
                    race.cancel()
                    race.joinAll(until: deadline)
                    throw IOOperationCancelled()
                }
                offerReplicas()
                outcome = race.wait(timeout: 0.002)
                if outcome == nil && Date() >= deadline {
                    race.cancel()
                    throw ExpertReadStall.error(bytes: PackStore.expertBytes, seconds: Self.stallBound)
                }
            }
            switch outcome! {
            case .succeeded: break
            case .cancelled:
                race.joinAll(until: deadline)
                throw IOOperationCancelled()
            case .failed(let reason):
                race.joinAll(until: deadline)
                guard entry.source == .externalSSD else {
                    throw ModelDrive.away?.awayError ?? EngineError("Original expert read failed: \(reason)")
                }
                if race.isHedged { throw EngineError("Both expert replica sources failed: \(reason)") }
                // No hedge ran (no staging record was free): the ordinary internal retry.
                guard scheduler.currentGeneration == entry.generation else { throw IOOperationCancelled() }
                let retry = try self.request(request.key, generation: entry.generation, source: .internalSSD)
                return try joinAndPin(retry, shouldContinue: shouldContinue)
            }
        } else if let io = entry.io {
            var outcome: IOOutcome?
            while outcome == nil {
                if !shouldContinue() {
                    _ = scheduler.cancel(io)
                    // Cancellation never releases a live writer: a hung one keeps its slot `reading` until it returns.
                    _ = io.join(timeout: deadline.timeIntervalSinceNow)
                    throw IOOperationCancelled()
                }
                outcome = io.join(timeout: 0.025)
                if outcome == nil && Date() >= deadline {
                    _ = scheduler.cancel(io)
                    throw ExpertReadStall.error(bytes: PackStore.expertBytes, seconds: Self.stallBound)
                }
            }
            switch outcome! {
            case .succeeded: break
            case .cancelled: throw IOOperationCancelled()
            case .failed(let reason):
                guard scheduler.currentGeneration == entry.generation else { throw IOOperationCancelled() }
                if entry.source == .externalSSD {
                    let retry = try self.request(request.key, generation: entry.generation, source: .internalSSD)
                    return try joinAndPin(retry, shouldContinue: shouldContinue)
                }
                // A drive that is gone fails the request with its own error (an outcome carries only words).
                throw ModelDrive.away?.awayError ?? EngineError("Original expert read failed: \(reason)")
            }
        }
        guard shouldContinue() else {
            if let race = entry.replica {
                race.cancel()
                race.joinAll(until: deadline)
            }
            throw IOOperationCancelled()
        }
        lock.lock()
        defer { lock.unlock() }
        guard scheduler.currentGeneration == entry.generation, entry.replica?.isCancelled != true else { throw IOOperationCancelled() }
        if let replica = entry.replica?.winningReplica {
            return LoadedExpert(ticket: entry.cacheTicket, replica: replica)
        }
        // A race publishes before it leaves inFlight: a request that adopted it after its last handle went got no handoff
        // pin. Pin the published record now; the ticket check refuses one evicted in between.
        if !entry.hasHandoffPin && entry.failure == nil { try? entry.installHandoffPin() }
        guard entry.hasHandoffPin else {
            throw EngineError(entry.failure ?? "Expert read was discarded before publication")
        }
        return LoadedExpert(ticket: entry.cacheTicket, buffer: try cache.pin(entry.cacheTicket), cache: cache)
    }

    public func loadAndPin(
        _ key: ExpertKey, generation: IOScheduler.Generation,
        source: IOSource? = nil
    ) throws -> LoadedExpert {
        let request = try self.request(key, generation: generation, source: source)
        return try joinAndPin(request)
    }

    /// Requests sharing an in-flight expert share cancellation. Use a request
    /// generation for sequence-wide cancellation, then retire returned GPU leases
    /// on command-buffer completion before recycling any consumer-owned buffers.
    @discardableResult public func cancel(_ request: ExpertLoadRequest) -> Bool {
        lock.lock()
        let io = request.entry.io
        let race = request.entry.replica
        lock.unlock()
        if let race {
            race.cancel()
            return true
        }
        return io.map { scheduler.cancel($0) } ?? false
    }

    /// `decode_prerouting_read_hygiene`: outcome of cancelling one predicted read.
    package enum SpeculativeCancellation { case cancelled, notQueued, shared }

    /// Read hygiene, serial owner, for a predicted request whose key the authoritative route
    /// did not select. Cancels the read only if it is still queued (never started), still this
    /// key's in-flight producer, held by no other request (demand never shared it) and not
    /// demanded; otherwise nothing changes. A started read always finishes (`notQueued`).
    /// Under the loader lock no demand can adopt the entry between the check and the
    /// cancellation. `shared` is a consistency failure the caller counts (expected never).
    /// The producer cleanup (slot freed, admission released, entry retired) runs on the
    /// scheduler's cancellation completion; join with `joinDiscarding` before reusing the key.
    package func cancelQueuedSpeculative(_ request: ExpertLoadRequest) -> SpeculativeCancellation {
        lock.lock()
        defer { lock.unlock() }
        let entry = request.entry
        guard inFlight[request.key] === entry, let io = entry.io else { return .notQueued }
        guard entry.handleCount == 1 else { return .shared }
        if let race = entry.replica {
            guard race.isSpeculative else { return .shared }
            return race.cancelIfQueued() ? .cancelled : .notQueued
        }
        return scheduler.cancelQueued(io) ? .cancelled : .notQueued
    }

    /// True while `request`'s own producer is still this key's in-flight entry.
    package func isInFlight(_ request: ExpertLoadRequest) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight[request.key] === request.entry
    }

    /// Fast late-miss skip: true once `request`'s read has an outcome (a resident request has none to wait for).
    /// Never blocks, cancels or publishes anything.
    package func readFinished(_ request: ExpertLoadRequest) -> Bool {
        lock.lock()
        let io = request.entry.io
        let race = request.entry.replica
        lock.unlock()
        if let race { return race.wait(timeout: 0) != nil }
        return io.map { $0.join(timeout: 0) != nil } ?? true
    }

    /// Wait without acquiring a consumer pin or retrying another drive. Used
    /// after cancellation when a bounded demand window discards its requests.
    /// A read not ended within the demand read bound from now is left running (ROADMAP R2): its slot stays `reading`
    /// until it returns.
    public func joinDiscarding(_ request: ExpertLoadRequest) {
        let deadline = Date(timeIntervalSinceNow: Self.stallBound)
        if let race = request.entry.replica {
            race.joinAll(until: deadline)
        } else if let io = request.entry.io {
            _ = io.join(timeout: deadline.timeIntervalSinceNow)
        }
    }

    /// Prefill has one read per request, with no replica producer. Keep the
    /// actual join separate: this wake-up only makes that request eligible for
    /// the rolling producer's next joined retirement and bounded refill.
    package func notifyPrefillCompletion(
        _ request: ExpertLoadRequest,
        _ wake: @escaping @Sendable () -> Void
    ) throws {
        lock.lock()
        let io = request.entry.io
        let replica = request.entry.replica
        lock.unlock()
        guard replica == nil, io != nil || request.residentAtRequest else {
            throw EngineError("Prefill completion requires one read or a pinned resident record")
        }
        if let io { try io.notifyCompletion(wake) } else { wake() }
    }

    /// Reads still running at the demand read bound from now are left running (ROADMAP R2): their slots stay `reading`,
    /// and their completions, refused by the new generation, free them.
    public func cancelAndJoin(_ generation: IOScheduler.Generation) throws {
        let deadline = Date(timeIntervalSinceNow: Self.stallBound)
        lock.lock()
        let races = Array(readRaces.values)
        lock.unlock()
        for race in races { race.cancel() }
        _ = try scheduler.cancelAndJoin(generation, until: deadline)
        joinRetainedReads(until: deadline)
    }
}
