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
    // Published requests may disappear from inFlight while their losing pread
    // remains live. Retain these races until the scheduler's actual joins pass.
    private var readRaces: [ExpertTicket: ExpertReadReplica] = [:]
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
                pending.replica?.markDemanded()
            }
            return ExpertLoadRequest(pending)
        }
        let acquisition = try cache.acquire(key)
        let source = preferredSource ?? (packedAvailable() ? .externalSSD : .internalSSD)
        let entry = ExpertLoadEntry(
            cache: cache, ticket: acquisition.ticket, source: source,
            generation: generation, residentAtRequest: !acquisition.newlyReserved)
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
            speculative: false)
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
        policy: ExpertSourcePolicy, calibration: ExpertSourceCalibration, speculative: Bool
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
                    if preferredSource == nil && policy == .earliestCompletion {
                        return try scheduler.submitEarliest(
                            availableSources: available, estimate: placement.estimatedSeconds, idleProbeSource: probe,
                            priority: priority, generation: generation, byteCount: PackStore.expertBytes,
                            operation: operation, completion: completion)
                    }
                    let selected = preferredSource ?? available[0]
                    return try scheduler.submit(
                        source: selected, priority: priority, generation: generation,
                        byteCount: PackStore.expertBytes, operation: { try operation(selected, $0) }, completion: completion)
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
            if preferredSource == nil && policy == .earliestCompletion {
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
