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
    /// A split read's part jobs (`SplitReads`), joined with the read's own (`joinDiscarding`).
    let splitJobs = SplitReadJobs()
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
    /// Whether lone decode misses may be split (`SplitReads.enabled` at start; checks of split reads turn it on). Off,
    /// a lone read is still counted (`lone_demand_waits`) and nothing else is planned or allocated.
    package var splitsReads = SplitReads.enabled
    /// Staging for split reads when no replica arena exists (replicas off, as on one drive with a hot copy): two records,
    /// charged to the ledger at the first split (about 5.6 MB), never before.
    private lazy var splitArena: ExpertReplicaArena? =
        replicaArena
        ?? (try? ExpertReplicaArena(
            ledger: cache.ledger, device: cache.device, records: 2, recordBytes: PackStore.expertBytes, lazily: true))

    /// How a split read reads parts of a record from either drive: whether both drives hold `key` in a form parts can be
    /// read from, whether a drive's read tally says it is wearing, a part read (its checks left in `PartChecks`), and the
    /// assembled record's check.
    package struct SplitPartReader: Sendable {
        /// A record's sources now (none, or one: read whole).
        let sources: @Sendable (ExpertKey) -> [PackStore.SplitSource]
        let wearing: @Sendable (ExpertKey, PackStore.SplitSource) -> Bool
        let read: @Sendable (ExpertKey, PackStore.SplitSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws -> Void
        let verify: @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        package init(
            sources: @escaping @Sendable (ExpertKey) -> [PackStore.SplitSource],
            wearing: @escaping @Sendable (ExpertKey, PackStore.SplitSource) -> Bool,
            read:
                @escaping @Sendable (ExpertKey, PackStore.SplitSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws ->
                Void,
            verify: @escaping @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        ) {
            self.sources = sources
            self.wearing = wearing
            self.read = read
            self.verify = verify
        }
        /// Two sources, the model's drive and the pack's (checks): `available` says whether both hold the record.
        package init(
            available: @escaping @Sendable (ExpertKey) -> Bool, wearing: @escaping @Sendable (ExpertKey, IOSource) -> Bool,
            read: @escaping @Sendable (ExpertKey, IOSource, Int, UnsafeMutableRawBufferPointer, PackStore.PartChecks) throws -> Void,
            verify: @escaping @Sendable (ExpertKey, UnsafeRawBufferPointer, PackStore.PartChecks) throws -> Void
        ) {
            @Sendable func queue(_ source: PackStore.SplitSource) -> IOSource {
                if case .pack = source { .externalSSD } else { .internalSSD }
            }
            self.init(
                sources: { available($0) ? [.model, .pack] : [] }, wearing: { wearing($0, queue($1)) },
                read: { key, source, offset, into, checks in try read(key, queue(source), offset, into, checks) }, verify: verify)
        }
        /// `store`'s part reads: the checkpoint's spans, each hot copy place holding the record and the pack queue's
        /// copies (`PackStore.splitSources`, the live lookups).
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
                read: { key, source, offset, into, checks in
                    try store.readExpertPart(key, source: source, at: offset, into: into, checks: checks)
                }, verify: { key, record, checks in try store.verifySplit(key, record: record, checks: checks) })
        }
    }
    /// Sets `reader`'s part reads (before serving): they find a record's sources, so lone reads are counted, and split
    /// the read when `splitsReads`.
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
        do {
            let mirrored = hotInternal(key)
            let selecting = preferredSource == nil && policy == .earliestCompletion
            let placement = estimator?.placementCalibration(hotInternal: mirrored, recordDecision: selecting) ?? calibration
            let probe = selecting && priority == .demand ? estimator?.idleProbeSource(hotInternal: mirrored) : nil
            // Any demand read whose caller named no drive (the hot-copy rule's default included): every source of the
            // record is a candidate (`PackStore.splitSources`), with any source policy.
            let split =
                explicitSource == nil && priority == .demand && !speculative
                ? planSplit(key, entry: entry, calibration: calibration) : nil
            entry.lone = split?.lone ?? false
            let parts = split?.parts
            let splitSource: IOSource? = parts.map { split!.queues[$0[0].source] }
            let splitRead: (@Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void)? = parts.map { parts in
                splitOperation(
                    key, entry: entry, parts: parts, order: split!.order, sources: split!.sources, queues: split!.queues,
                    calibration: calibration, generation: generation)
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
                ) { operation, completion in
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
            let completion: @Sendable (IOTicket, IOOutcome) -> Void = { [self, entry] _, outcome in
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
            _ = cache.completeRead(acquisition.ticket, verified: false)
            throw error
        }
    }

    /// The split plan for demand read `key` (loader lock held), nil unless more than one source holds it now
    /// (`SplitPartReader.sources`): whether it is lone (decode, no other demand read of this loader in flight), the
    /// plan's parts (nil: read whole), the sources fastest first, the sources and each one's queue. Per source: the
    /// model's spans and the pack at the placement curve's one-read rate and latency (live), healthy at the static
    /// calibration's; a hot copy place at its lane's time per read once it has 8 reads (else the mirror curve),
    /// healthy from its own slow-read tally. Outstanding: its queue's reads (a split's part jobs included, so a read
    /// planned during a split finds it busy) plus a place's own reads in flight.
    private func planSplit(
        _ key: ExpertKey, entry: ExpertLoadEntry, calibration: ExpertSourceCalibration
    ) -> (lone: Bool, parts: [SplitReadPart]?, order: [Int], sources: [PackStore.SplitSource], queues: [IOSource])? {
        guard let splitReader else { return nil }
        let sources = splitReader.sources(key)
        guard sources.count >= 2 else { return nil }
        // A published race waits in `inFlight` only for its loser to end: not a demand read still in flight.
        let lone =
            decoding
            && !inFlight.values.contains { other in
                other.demand && other !== entry
                    && (other.replica.map { $0.wait(timeout: 0) == nil } ?? (other.io.map { $0.join(timeout: 0) == nil } ?? false))
            }
        let queues = sources.map { source -> IOSource in if case .pack = source { .externalSSD } else { .internalSSD } }
        guard lone else { return nil }
        guard splitsReads else { return (true, nil, [], sources, queues) }
        let drives = scheduler.snapshot().drives
        let healthy = estimator?.calibration ?? calibration
        let original = estimator?.placementCalibration(hotInternal: false, recordDecision: false) ?? calibration
        let mirror = estimator?.placementCalibration(hotInternal: true, recordDecision: false) ?? calibration
        let described = sources.indices.map { n -> SplitReadSource in
            let queued = drives.first { $0.source == queues[n] }?.outstanding ?? 0
            let wearing = splitReader.wearing(key, sources[n])
            switch sources[n] {
            case .place(let set, let place):
                let lane = set.lane(place)
                let rates = set.stores[place].reads.rates
                let measured = lane.reads >= 8 && lane.serviceSeconds > 0
                return SplitReadSource(
                    bytesPerSecond: measured ? Double(PackStore.expertBytes) / lane.serviceSeconds : mirror.internalBytesPerSecond[0],
                    latencySeconds: measured ? 0 : mirror.internalLatencySeconds,
                    healthyBytesPerSecond: rates.healthy > 0 ? rates.healthy : healthy.internalBytesPerSecond[0],
                    outstanding: queued + lane.inFlight, wearing: wearing)
            case .model, .pack:
                let internalSource = queues[n] == .internalSSD
                return SplitReadSource(
                    bytesPerSecond: (internalSource ? original.internalBytesPerSecond : original.externalBytesPerSecond)[0],
                    latencySeconds: internalSource ? original.internalLatencySeconds : original.externalLatencySeconds,
                    healthyBytesPerSecond: (internalSource ? healthy.internalBytesPerSecond : healthy.externalBytesPerSecond)[0],
                    outstanding: queued, wearing: wearing)
            }
        }
        let plan = SplitReads.plan(recordBytes: PackStore.expertBytes, sources: described, waited: true)
        splitReads.record(plan)
        let order = described.indices.sorted { described[$0].seconds(PackStore.expertBytes) < described[$1].seconds(PackStore.expertBytes) }
        if case .split(let parts) = plan { return (true, parts, order, sources, queues) }
        return (true, nil, order, sources, queues)
    }

    /// The read of a split record into its slot (`SplitReads.execute`): each other part a demand job on its source's
    /// queue, staged in a replica-arena record (`splitArena` without replicas; none free: into the slot, unguarded).
    /// The fastest part is read in place on the job's thread (covered only by the replica race, if any); a late part is re-read
    /// after the replica hedge's static delay for its bytes; a staged part is copied in only while the race (if any)
    /// still owns the record; every part's time trains its queue's online estimate as a whole read's does (a place's
    /// lane too, in `HotCopySet.read`). Then the record is checked as a whole as its parts' sources check one. A split
    /// that fails reads the record again whole from `source`, unless a replica published it meanwhile.
    private func splitOperation(
        _ key: ExpertKey, entry: ExpertLoadEntry, parts: [SplitReadPart], order: [Int], sources: [PackStore.SplitSource],
        queues: [IOSource], calibration: ExpertSourceCalibration, generation: IOScheduler.Generation
    ) -> @Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void {
        let (reader, wholeRead, arena, counters, scheduler, estimator) = (
            splitReader!, self.reader, splitArena, splitReads, scheduler, estimator
        )
        let jobs = entry.splitJobs
        let partSources = sources
        let names = sources.map { source -> String in
            switch source {
            case .model: "model_drive"
            case .place(_, let place): "copy\(place)"
            case .pack: "pack"
            }
        }
        return { [weak entry] source, destination in
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
                    }, owns: owned,
                    submit: { part, bytes, job in
                        jobs.add(
                            try scheduler.submit(
                                source: queues[part], priority: .demand, generation: generation, byteCount: bytes,
                                operation: { _ in job(nil) },
                                // Ends the part when its job ended without running it (cancelled while queued).
                                completion: { _, _ in job(IOOperationCancelled()) }))
                    }
                ) { part, offset, into in
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
                counters.record(outcome, names: names)
            } catch let orphaned as SplitReadOrphaned {
                counters.record(nil, names: [])
                throw orphaned
            } catch let cancelled as IOOperationCancelled {
                // Its generation or scheduler ended: no whole read after it.
                throw cancelled
            } catch {
                counters.record(nil, names: [])
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
    /// until it returns. The read's own job first: it adds its split parts to `splitJobs` while it runs, so once it
    /// has ended every part is there to join.
    public func joinDiscarding(_ request: ExpertLoadRequest) {
        let deadline = Date(timeIntervalSinceNow: Self.stallBound)
        if let race = request.entry.replica {
            race.joinAll(until: deadline)
        } else if let io = request.entry.io {
            _ = io.join(timeout: deadline.timeIntervalSinceNow)
        }
        request.entry.splitJobs.join(until: deadline)
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
