import Foundation
import Metal

/// Where expert records come from: `legacyOriginal` (no slab cache; refused by `ExpertExecution`) or
/// `slabPackedPreferred` (slab cache, the pack first when there is one, by the source policy).
/// Every source yields the same record bytes, so
/// the mode changes only where reads come from, never arithmetic.
public enum ExpertExecutionMode: String, Codable, Sendable {
    case legacyOriginal, slabPackedPreferred
}

/// The expert cache, read window and source settings (`execution.expert_*`). The default mode is legacyOriginal;
/// `OrigamiPlan` writes slabPackedPreferred. Slot counts and limits change speed and memory, never arithmetic.
public struct ExpertExecutionConfiguration: Codable, Sendable {
    public let mode: ExpertExecutionMode
    /// cacheSlots is the maximum used by decode; nil keeps the same prefill limit.
    public let cacheSlots, slotsPerSlab, demandWindow, readConcurrency: Int
    public let prefillCacheSlots: Int?
    public let prefillSourcePolicy, decodeSourcePolicy: ExpertSourcePolicy
    public let sourceCalibration: ExpertSourceCalibration
    /// Online source calibration bounds; nil (the default and every older configuration) keeps the static path.
    public let onlineCalibration: ExpertSourceOnlineCalibration?
    public let tailReplicas: Bool
    public let prefillReadAhead: Bool
    public let prefillLayerCommand: Bool
    public let reuseAdmission: Bool
    public let preallocateDecodeSlabs: Bool
    public var decodeSlabOwnerBytes: UInt64 { preallocateDecodeSlabs ? 16_384 : 0 }
    /// `wire_decode_slabs`: slab preparation user-wires (mlock) every admitted decode slab
    /// instead of touching it, so no slab page can be compressed or reclaimed after TTFT. Requires
    /// `preallocateDecodeSlabs`. No new bytes: wiring keeps already-charged slab pages resident.
    /// Like `layerWaves`, set after init by RuntimeConfiguration and not part of the coded keys, so
    /// every existing encoded configuration is byte-identical when the flag is off.
    public var wireDecodeSlabs = false
    /// Drives holding expert copies read beside each other (`expert_pack_drives`): the packed queue runs this many
    /// times the reads at once, so every drive gets the depth one drive had. Set after init like `wireDecodeSlabs`.
    public var packDrives = 1
    public var reuseOwnerBytes: UInt64 { reuseAdmission ? ExpertReusePolicy.admittedBytes : 0 }
    /// `expert_layer_waves`: experts per stage wave in the prefill layer command (1 = the serial chain). Each extra
    /// wave slot owns one more C256 scratch set.
    public var layerWaves = 1
    public var layerCommandOwnerBytes: UInt64 {
        prefillLayerCommand
            ? FusedExpertExecution.layerCommandOwnerBytes + UInt64(layerWaves - 1) * FusedExpertExecution.layerWaveScratchBytes : 0
    }
    public var readAheadOwnerBytes: UInt64 { prefillReadAhead ? PrefillExpertReadAhead.metadataBytes : 0 }
    public var replicaOwnerBytes: UInt64 { tailReplicas ? ExpertReplicaArena.admittedBytes : 0 }
    public var effectivePrefillCacheSlots: Int { prefillCacheSlots ?? cacheSlots }
    public init(
        mode: ExpertExecutionMode = .legacyOriginal, cacheSlots: Int = 64,
        slotsPerSlab: Int = 8, demandWindow: Int = 8, readConcurrency: Int = 4,
        prefillCacheSlots: Int? = nil, prefillSourcePolicy: ExpertSourcePolicy = .packedPreferred,
        decodeSourcePolicy: ExpertSourcePolicy = .packedPreferred, sourceCalibration: ExpertSourceCalibration = .neutral,
        tailReplicas: Bool = false, prefillReadAhead: Bool = false, prefillLayerCommand: Bool = false, reuseAdmission: Bool = false,
        preallocateDecodeSlabs: Bool = false,
        onlineCalibration: ExpertSourceOnlineCalibration? = nil
    ) {
        self.mode = mode
        self.cacheSlots = cacheSlots
        self.slotsPerSlab = slotsPerSlab
        self.demandWindow = demandWindow
        self.readConcurrency = readConcurrency
        self.prefillCacheSlots = prefillCacheSlots
        self.prefillSourcePolicy = prefillSourcePolicy
        self.decodeSourcePolicy = decodeSourcePolicy
        self.sourceCalibration = sourceCalibration
        self.onlineCalibration = onlineCalibration
        self.preallocateDecodeSlabs = preallocateDecodeSlabs
        self.tailReplicas = tailReplicas
        self.prefillReadAhead = prefillReadAhead
        self.prefillLayerCommand = prefillLayerCommand
        self.reuseAdmission = reuseAdmission
    }
    /// A byte ceiling is rounded down to whole physical slabs. Requested queue
    /// depth above the demand window has no extra work to execute, so its actual
    /// concurrency is clamped to the window and recorded in this configuration.
    public init(
        mode: ExpertExecutionMode, cacheBytes: UInt64, slotsPerSlab: Int = 8,
        demandWindow: Int = 8, readConcurrency: Int = 4, prefillCacheSlots: Int? = nil,
        prefillSourcePolicy: ExpertSourcePolicy = .packedPreferred, decodeSourcePolicy: ExpertSourcePolicy = .packedPreferred,
        sourceCalibration: ExpertSourceCalibration = .neutral, tailReplicas: Bool = false, prefillReadAhead: Bool = false,
        prefillLayerCommand: Bool = false, reuseAdmission: Bool = false, preallocateDecodeSlabs: Bool = false,
        onlineCalibration: ExpertSourceOnlineCalibration? = nil
    ) throws {
        guard (1...512).contains(slotsPerSlab), (1...64).contains(readConcurrency), demandWindow > 0 else {
            throw EngineError("Invalid expert byte-budget configuration", code: "invalid_configuration")
        }
        let slabBytes = UInt64(slotsPerSlab * PackStore.expertBytes)
        let slabCount = cacheBytes / slabBytes
        guard slabCount > 0, slabCount <= UInt64(48 * 512 / slotsPerSlab) else {
            throw EngineError("Expert cache byte ceiling is outside bounded slab capacity", code: "invalid_configuration")
        }
        self.init(
            mode: mode, cacheSlots: Int(slabCount) * slotsPerSlab, slotsPerSlab: slotsPerSlab,
            demandWindow: demandWindow, readConcurrency: min(readConcurrency, demandWindow),
            prefillCacheSlots: prefillCacheSlots, prefillSourcePolicy: prefillSourcePolicy, decodeSourcePolicy: decodeSourcePolicy,
            sourceCalibration: sourceCalibration,
            tailReplicas: tailReplicas, prefillReadAhead: prefillReadAhead, prefillLayerCommand: prefillLayerCommand,
            reuseAdmission: reuseAdmission, preallocateDecodeSlabs: preallocateDecodeSlabs,
            onlineCalibration: onlineCalibration)
        try validate()
    }
    enum CodingKeys: String, CodingKey {
        case mode, cacheSlots, slotsPerSlab, demandWindow, readConcurrency, prefillCacheSlots
        case prefillSourcePolicy, decodeSourcePolicy, sourceCalibration, tailReplicas, prefillReadAhead, prefillLayerCommand,
            reuseAdmission, preallocateDecodeSlabs
        case onlineCalibration
    }
    /// Keep older recorded configurations decodable with placement disabled.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            mode: try values.decode(ExpertExecutionMode.self, forKey: .mode),
            cacheSlots: try values.decode(Int.self, forKey: .cacheSlots),
            slotsPerSlab: try values.decode(Int.self, forKey: .slotsPerSlab),
            demandWindow: try values.decode(Int.self, forKey: .demandWindow),
            readConcurrency: try values.decode(Int.self, forKey: .readConcurrency),
            prefillCacheSlots: try values.decodeIfPresent(Int.self, forKey: .prefillCacheSlots),
            prefillSourcePolicy: try values.decodeIfPresent(ExpertSourcePolicy.self, forKey: .prefillSourcePolicy) ?? .packedPreferred,
            decodeSourcePolicy: try values.decodeIfPresent(ExpertSourcePolicy.self, forKey: .decodeSourcePolicy) ?? .packedPreferred,
            sourceCalibration: try values.decodeIfPresent(ExpertSourceCalibration.self, forKey: .sourceCalibration) ?? .neutral,
            tailReplicas: try values.decodeIfPresent(Bool.self, forKey: .tailReplicas) ?? false,
            prefillReadAhead: try values.decodeIfPresent(Bool.self, forKey: .prefillReadAhead) ?? false,
            prefillLayerCommand: try values.decodeIfPresent(Bool.self, forKey: .prefillLayerCommand) ?? false,
            reuseAdmission: try values.decodeIfPresent(Bool.self, forKey: .reuseAdmission) ?? false,
            preallocateDecodeSlabs: try values.decodeIfPresent(Bool.self, forKey: .preallocateDecodeSlabs) ?? false,
            onlineCalibration: try values.decodeIfPresent(ExpertSourceOnlineCalibration.self, forKey: .onlineCalibration))
        try validate()
    }
    public var cacheBytes: UInt64 { UInt64(cacheSlots * PackStore.expertBytes) }
    public var prefillCacheBytes: UInt64 { UInt64(effectivePrefillCacheSlots * PackStore.expertBytes) }
    func validate() throws {
        try sourceCalibration.validate()
        try onlineCalibration?.validate()
        guard onlineCalibration == nil || mode == .slabPackedPreferred else {
            throw EngineError("Online source calibration requires slabPackedPreferred mode", code: "invalid_configuration")
        }
        guard !preallocateDecodeSlabs || (mode != .legacyOriginal && slotsPerSlab == 8) else {
            throw EngineError("Decode slab preallocation requires eight-slot slabs", code: "invalid_configuration")
        }
        guard !wireDecodeSlabs || preallocateDecodeSlabs else {
            throw EngineError("Decode slab wiring requires decode slab preallocation", code: "invalid_configuration")
        }
        guard !reuseAdmission || (mode != .legacyOriginal && slotsPerSlab == 8 && effectivePrefillCacheSlots >= 32 && demandWindow <= 16)
        else {
            throw EngineError("Reuse admission requires 8-slot slabs, 24 transient slots and bounded demand", code: "invalid_configuration")
        }
        guard !prefillLayerCommand || prefillReadAhead else {
            throw EngineError("Layer command requires the complete retained prefill producer", code: "invalid_configuration")
        }
        // The governor may shrink the prefill cache to its minimum: one layer plus the demand window must still fit.
        guard
            !prefillReadAhead
                || (mode != .legacyOriginal && effectivePrefillCacheSlots >= 512 + demandWindow && demandWindow >= 2
                    && 512 + demandWindow <= ConversationPhaseBudget.minimumSlots)
        else {
            let minimum = ConversationPhaseBudget.minimumSlots
            throw EngineError(
                "Read-ahead needs expert_window 2...\(minimum - 512) (one layer plus the window within the \(minimum)-slot "
                    + "minimum cache) and at least 512 + window prefill cache slots; got window \(demandWindow), \(effectivePrefillCacheSlots) slots",
                code: "invalid_configuration")
        }
        guard !tailReplicas || (mode == .slabPackedPreferred && readConcurrency == 4 && demandWindow == 16) else {
            throw EngineError(
                "Expert tail replicas require packed mode, queue depth 4 and 16 outstanding records", code: "invalid_configuration")
        }
        guard mode == .slabPackedPreferred || (prefillSourcePolicy == .packedPreferred && decodeSourcePolicy == .packedPreferred) else {
            throw EngineError("Adaptive exact-source policy requires slabPackedPreferred mode", code: "invalid_configuration")
        }
        guard (1...48 * 512).contains(cacheSlots), (1...512).contains(slotsPerSlab),
            slotsPerSlab <= cacheSlots, cacheSlots % slotsPerSlab == 0,
            (slotsPerSlab...cacheSlots).contains(effectivePrefillCacheSlots),
            effectivePrefillCacheSlots % slotsPerSlab == 0,
            (1...min(effectivePrefillCacheSlots, 512)).contains(demandWindow),
            (1...min(demandWindow, 64)).contains(readConcurrency)
        else {
            throw EngineError("Invalid expert slab/window/phase configuration", code: "invalid_configuration")
        }
    }
}

/// The three projections of one expert record, in the layout PackStore verified. Offsets are relative to the
/// start of the record (2,764,800 bytes in the checkpoint's 4-bit group-64 format), not to the enclosing slab.
public enum ExpertProjection: Int, CaseIterable, Sendable {
    case gate, up, down
    public var width: Int { self == .down ? 640 : 2560 }
    public var outputs: Int { self == .down ? 2560 : 640 }
    /// Offsets in a record of the expert format this process runs (`ModelLayout.experts`); the three
    /// projections have the same size: codes, then scales, then biases.
    public var weightOffset: Int { rawValue * ModelLayout.experts.bytes(width: 2560, outputs: 640) }
    public var scaleOffset: Int { weightOffset + ModelLayout.experts.codeBytes(width: width, outputs: outputs) }
    public var biasOffset: Int { scaleOffset + ModelLayout.experts.scaleBytes(width: width, outputs: outputs) }
}

private final class ExpertExecutionMetadata: Sendable {
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64, category: String = "expert-execution-metadata") throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: category)
    }
    deinit { try? ledger.release(reservation) }
}

/// A serial sequence owner over thread-safe cache/I/O primitives. Cache records
/// persist across windows and sequences; pins and pending jobs do not. The
/// controlling thread must serialize begin/window/finish calls. Cancellation
/// arrives only through the supplied shouldContinue callback, which may read a
/// separately synchronized flag. Optional whole-prefill read-ahead has one
/// separately joined producer; decode remains demand-driven.
public final class ExpertExecution {
    /// The cache policy phase: prefill streams layer unions through the prefill limit; decode keeps the hot set.
    public enum Phase: String, Codable, Sendable { case prefill, decode }
    /// Execution, cache, scheduler and store counters. Reads unsynchronized owner state: call on the serving thread.
    public struct Snapshot: Codable, Sendable {
        public let mode: ExpertExecutionMode
        public let configuration: ExpertExecutionConfiguration
        public let cache: ExpertCache.Snapshot
        public let io: IOScheduler.Snapshot
        public let phase: Phase
        public let targetCacheSlots: Int
        public let targetCacheBytes: UInt64
        public let phaseTransitions: UInt64
        public let sequences, windows, demandedExperts, consumedExperts: UInt64
        /// Exposed consumer join/pin wall time; absent in older observations.
        public let consumerJoinSeconds: Double?
        public let residentFirstWindows, residentFirstExperts: UInt64?
        public let replicas: ExpertReplicaSnapshot?
        public let store: PackStore.Status?
        package let prefillReadAhead: PrefillExpertReadAhead.Snapshot?
        package let decodeSlabPreparation: DecodeSlabPreparation?
        /// Online source calibration counters; present only when the configuration enables it.
        public var expertSourceOnline: ExpertSourceOnlineSnapshot? = nil
        public var prefillScheduling: PrefillSchedulingSnapshot? = nil
    }
    public struct PrefillSchedulingSnapshot: Codable, Sendable {
        public var selectiveScopes: UInt64 = 0, broadScopes: UInt64 = 0
        public var broadDemotions: UInt64 = 0
        public var routedExperts: UInt64 = 0, routedLayers: UInt64 = 0
        public var readyGroups: UInt64 = 0, readyExperts: UInt64 = 0, residentExperts: UInt64 = 0
        public var readAheadSubmitted: UInt64 = 0, readAheadConsumed: UInt64 = 0, readAheadSkipped: UInt64 = 0
        public var readAheadDemandBypasses: UInt64 = 0
        public var readAheadWaitSeconds: Double = 0
        /// Summed over prompts, counted as decode starts: the cache's limit while the prompt was read, and how far that was
        /// below the decode cache that follows (the records the prompt took from decode), in records.
        public var promptCacheSlots: UInt64 = 0, promptCacheSlotsBelowDecode: UInt64 = 0
    }
    public let configuration: ExpertExecutionConfiguration
    /// Cumulative receipts of `prepareDecodeSlabs` over the decode transitions of this owner.
    package struct DecodeSlabPreparation: Codable, Sendable {
        package var transitions: UInt64 = 0
        package var seconds: Double = 0
        package var lastTargetBytes: UInt64 = 0
        package var lastCoveredBytes: UInt64 = 0
        package var physicalHighWater: UInt64 = 0
        package var minimumHeadroom: UInt64 = UInt64.max
        /// `wire_decode_slabs` only (nil otherwise; cumulative over transitions): slabs and pages newly
        /// user-wired, bytes wired at the last transition, and slabs that fell back to the touch (must be 0).
        package var slabsWired: UInt64? = nil
        package var pagesWired: UInt64? = nil
        package var lastWiredBytes: UInt64? = nil
        package var wireFallbacks: UInt64? = nil
    }
    private var decodeSlabPreparation: DecodeSlabPreparation?
    /// The slab preparation receipt without a full cache snapshot (handoff/whole-execution reporting).
    package var decodeSlabPreparationReceipt: DecodeSlabPreparation? { decodeSlabPreparation }
    private let storeSnapshot: @Sendable () -> PackStore.Status?
    private let ledger: MemoryLedger
    private let metadata: ExpertExecutionMetadata
    private let cache: ExpertCache
    private let replicaArena: ExpertReplicaArena?
    private let scheduler: IOScheduler
    fileprivate let loader: SourceExpertLoader
    private var generation: IOScheduler.Generation?
    fileprivate var sourcePolicy: ExpertSourcePolicy {
        phase == .prefill ? configuration.prefillSourcePolicy : configuration.decodeSourcePolicy
    }
    private var phase: Phase = .prefill
    /// Counts every window's experts for the model's own hot-expert ranking; nil counts nothing.
    public var routing: RoutingProfile?
    public var isDecodePhase: Bool { phase == .decode }
    private var phaseTransitions: UInt64 = 0
    private var readAhead: PrefillExpertReadAhead?
    private var lastReadAhead: PrefillExpertReadAhead.Snapshot?
    private var prefillReadPolicy = PrefillReadPolicy()
    private var adaptivePrefillScope = false
    private var adaptivePrefillLayers = 0
    private var prefillScopeCompleted = false
    private var prefillScheduling = PrefillSchedulingSnapshot()
    public var prefillSchedulingSnapshot: PrefillSchedulingSnapshot { prefillScheduling }
    package var usesSelectivePrefill: Bool { adaptivePrefillScope && readAhead == nil }
    private weak var activeWindow: ExpertDemandWindow?
    private var sequences: UInt64 = 0, windows: UInt64 = 0, demanded: UInt64 = 0, consumed: UInt64 = 0
    private var consumerJoinSeconds = 0.0
    private var residentFirstWindows: UInt64 = 0, residentFirstExperts: UInt64 = 0
    private let preroutingEnabled: Bool
    /// `decode_prerouting_read_hygiene`: queued-only cancellation of unrouted predictions,
    /// reader-pool priority tokens and decode PLE prefetch holds (scheduler-owned counters).
    private let readHygiene: Bool
    /// Fast late-miss skip: handles of skipped reads without an outcome yet, so read boundaries can join them.
    /// Those with an outcome are dropped at the next skip point (their handoff pins retire with the handle).
    private var detachedReads: [ExpertLoadRequest] = []
    /// Pre-routing: adopted predictions the route did not select, kept until their reads have an outcome or a read
    /// boundary joins them (a started read is never cancelled, and a phase change needs every drive idle).
    private var adoptedPredictions: [ExpertLoadRequest] = []
    /// Fast late-miss skip: demand records added to every drive's scheduler limits (0 = the exact limits). Skips are
    /// admitted only while the detached reads without an outcome stay within it, so they never take demand capacity.
    private let fastSkipReserve: Int

    public convenience init(
        store: PackStore, device: MTLDevice, ledger: MemoryLedger,
        configuration: ExpertExecutionConfiguration, earlyDemand: Bool = false, prerouting: Bool = false,
        readHygiene: Bool = false, fastSkipReserve: Int = 0
    ) throws {
        try self.init(
            device: device, ledger: ledger, configuration: configuration, earlyDemand: earlyDemand, prerouting: prerouting,
            readHygiene: readHygiene, fastSkipReserve: fastSkipReserve, packedAvailable: { store.packedAvailable },
            hotInternal: { store.hasHotExpert($0) }, storeSnapshot: { store.status() }
        ) { key, source, destination in
            if source == .internalSSD {
                try store.readInternalExpert(key, into: destination)
            } else {
                try store.readExpert(layer: key.layer, expert: key.expert, into: destination, source: .packed)
            }
        }
        loader.enableSplitReads(SourceExpertLoader.SplitPartReader(store: store))
    }
    /// Injected exact reader supports deterministic fault/cancellation tests.
    /// `prerouting` adds DecodePrerouting.readLanes lanes and outstandingRecords
    /// records per drive that only prefetch-priority predicted reads may use;
    /// demand keeps its configured lanes and its complete outstanding reserve.
    /// `readHygiene` (requires `prerouting`) enables the scheduler's read hygiene.
    /// `fastSkipReserve` (fast late-miss skip; 0 = off) adds that many demand records per drive for skipped reads.
    public init(
        device: MTLDevice, ledger: MemoryLedger, configuration: ExpertExecutionConfiguration, earlyDemand: Bool = false,
        prerouting: Bool = false, readHygiene: Bool = false, fastSkipReserve: Int = 0,
        packedAvailable: @escaping @Sendable () -> Bool = { true },
        hotInternal: @escaping @Sendable (ExpertKey) -> Bool = { _ in false },
        storeSnapshot: @escaping @Sendable () -> PackStore.Status? = { nil },
        reader: @escaping SourceExpertLoader.Reader
    ) throws {
        guard configuration.mode != .legacyOriginal else { throw EngineError("Legacy expert mode bypasses the slab helper") }
        try configuration.validate()
        self.configuration = configuration
        self.ledger = ledger
        self.storeSnapshot = storeSnapshot
        guard !readHygiene || prerouting else {
            throw EngineError("Pre-routing read hygiene requires decode pre-routing", code: "invalid_configuration")
        }
        guard (0...64).contains(fastSkipReserve) else {
            throw EngineError("Fast skip read reserve must be in 0...64", code: "invalid_configuration")
        }
        preroutingEnabled = prerouting
        self.readHygiene = readHygiene
        self.fastSkipReserve = fastSkipReserve
        // Fixed lookup, slot metadata, the complete 512-ID layer union and
        // bounded pending request references. Scheduler jobs reserve separately.
        let metadata = try ExpertExecutionMetadata(
            ledger: ledger,
            bytes: UInt64(48 * 512 * MemoryLayout<Int?>.stride + configuration.cacheSlots * 256 + 256 * 1024)
                + configuration.readAheadOwnerBytes + configuration.decodeSlabOwnerBytes)
        self.metadata = metadata
        do {
            // A GGUF whose layers' records differ in size keeps them in up to three slot sizes (ROADMAP C5), each class
            // able to hold one layer's prompt read beside the demand window.
            cache = try ExpertCache(
                device: device, ledger: ledger,
                byteLimit: configuration.prefillCacheBytes, slotsPerSlab: configuration.slotsPerSlab,
                maximumByteLimit: configuration.cacheBytes, reuseAdmission: configuration.reuseAdmission,
                sizes: ModelLayout.ggufExperts.map { .grouping($0.layers.map(\.bytes)) } ?? .one,
                minimumClassRecords: 512 + configuration.demandWindow + ExpertReusePolicy.transientSlots)
            let demandOutstanding = max(configuration.demandWindow, earlyDemand ? 20 : 0) + fastSkipReserve
            let outstanding = demandOutstanding + (prerouting ? DecodePrerouting.outstandingRecords : 0)
            let lanes = configuration.readConcurrency + (prerouting ? DecodePrerouting.readLanes : 0)
            let demandBytes = UInt64(demandOutstanding * PackStore.expertBytes)
            let pendingBytes = UInt64(outstanding * PackStore.expertBytes)
            let limits = IOScheduler.Limits(
                maximumConcurrent: lanes, reservedDemandConcurrent: configuration.readConcurrency,
                maximumOutstanding: outstanding, reservedDemandOutstanding: demandOutstanding,
                maximumOutstandingBytes: pendingBytes, reservedDemandBytes: demandBytes)
            let drives = configuration.packDrives
            let running = min(lanes * drives, outstanding)
            let packed =
                drives == 1
                ? limits
                : IOScheduler.Limits(
                    maximumConcurrent: running,
                    reservedDemandConcurrent: min(configuration.readConcurrency * drives, demandOutstanding, running),
                    maximumOutstanding: outstanding, reservedDemandOutstanding: demandOutstanding,
                    maximumOutstandingBytes: pendingBytes, reservedDemandBytes: demandBytes)
            scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: packed, readHygiene: readHygiene)
            let arena = configuration.tailReplicas ? try ExpertReplicaArena(ledger: ledger, device: device) : nil
            let estimator = try configuration.onlineCalibration.map {
                try ExpertSourceOnlineEstimator(
                    calibration: configuration.sourceCalibration, settings: $0,
                    maximumConcurrent: configuration.readConcurrency)
            }
            replicaArena = arena
            loader = SourceExpertLoader(
                cache: cache, scheduler: scheduler, replicaArena: arena, estimator: estimator,
                packedAvailable: packedAvailable,
                hotInternal: hotInternal
            ) { [metadata, reader] key, source, destination in
                // A completed scheduler closure may outlive join by its final
                // ARC cleanup. Keep lookup/request metadata charged with it.
                try withExtendedLifetime(metadata) { try reader(key, source, destination) }
            }
        } catch { throw error }
    }
    deinit {
        try? finishSequence()
        scheduler.shutdown()
    }

    /// Queued decode experts: every Metal buffer a decode record can live in (cache slabs and the replica arena), with a generation that
    /// moves when slabs come or go, and a hold on slab growth while pre-encoded commands are queued.
    package func recordBuffers() -> (generation: UInt64, buffers: [MTLBuffer]) {
        let slabs = cache.slabBuffers()
        return (slabs.generation, slabs.buffers + ((replicaArena?.metalBuffer).map { [$0] } ?? []))
    }
    package func holdSlabGrowth(_ held: Bool) { cache.holdSlabGrowth(held) }
    package func completePendingTrim() { cache.completePendingTrim() }
    /// The GPU residency table and its holds (see `ExpertCache.gpuTable`).
    package func gpuTable() throws -> (keys: MTLBuffer, slots: MTLBuffer) { try cache.gpuTable() }
    package func holdForGPU(layer: Int) -> UInt32 { cache.holdForGPU(layer: layer) }
    package func releaseForGPU(layer: Int?) { cache.releaseForGPU(layer: layer) }
    package func gpuComputes(_ keys: [ExpertKey], epoch: UInt32) -> [Bool] { cache.gpuComputes(keys, epoch: epoch) }

    /// Called before sequence-state admission, so an old decode cache gives its
    /// cold slabs back before the new prompt reserves projected state/workspace.
    /// A failed admission leaves this stable prefill limit and no generation.
    public func preparePrefillPhase(cacheBytes: UInt64? = nil) throws {
        guard generation == nil else {
            throw EngineError("Prefill cache preparation requires a finished sequence", code: "expert_phase_busy")
        }
        try setPhase(.prefill, prefillBytes: cacheBytes)
    }
    public func beginSequence(prefillCacheBytes: UInt64? = nil) throws {
        guard generation == nil, activeWindow == nil else { throw EngineError("Previous expert sequence must finish first") }
        try preparePrefillPhase(cacheBytes: prefillCacheBytes)
        generation = try scheduler.beginGeneration()
        sequences += 1
    }
    /// Caller owns a committed prompt frontier. Raising this ceiling allocates
    /// nothing: demand fills new slabs while already-hot records stay in place.
    public func beginDecodePhase(cacheBytes: UInt64? = nil, recoveryBytes: UInt64? = nil) throws {
        guard generation != nil else {
            throw EngineError("Decode cache preparation requires an active sequence", code: "expert_phase_state")
        }
        try setPhase(.decode, decodeBytes: cacheBytes, recoveryBytes: recoveryBytes)
    }
    /// Join retained losing reads without ending the sequence or changing its
    /// cache policy. A published replica can precede its original read's exit.
    package func joinReadProducers() throws {
        guard generation != nil else { throw EngineError("Read boundary requires an active sequence", code: "expert_phase_state") }
        guard activeWindow == nil, readAhead == nil, !adaptivePrefillScope else {
            throw EngineError("Read boundary requires closed consumer windows", code: "expert_phase_busy")
        }
        joinDetachedReads()
        loader.joinRetainedReads()
        let current = cache.snapshot()
        guard current.pinnedSlots == 0, current.reservedSlots == 0, current.inFlightReads == 0,
            scheduler.snapshot().drives.allSatisfy({ $0.outstanding == 0 }),
            (loader.replicaSnapshot()?.consumerPins ?? 0) == 0
        else {
            throw EngineError("Read boundary retains producers or consumers", code: "expert_phase_busy")
        }
    }
    /// Fast late-miss skip and pre-routing: wait for every skipped or unrouted predicted read still running, then
    /// release its handle (and handoff pin).
    private func joinDetachedReads() {
        for request in detachedReads + adoptedPredictions { loader.joinDiscarding(request) }
        detachedReads.removeAll()
        adoptedPredictions.removeAll()
    }
    /// Pre-routing, serial owner: keeps `requests` (from DecodePrerouting.adopt) and drops the handles whose reads ended.
    package func retainPredictions(_ requests: [ExpertLoadRequest]) {
        adoptedPredictions = (adoptedPredictions + requests).filter { !loader.readFinished($0) }
    }
    /// Fast late-miss skip: how many more reads may be skipped now. Drops the handles of detached reads that have an
    /// outcome; each remaining one holds at most one job per drive (its primary, or an internal replica) beyond the
    /// exact path's own load, and only skips add to them, so they never exceed the reserve between skip points.
    fileprivate func detachedReadCapacity() -> Int {
        detachedReads.removeAll { loader.readFinished($0) }
        return fastSkipReserve - detachedReads.count
    }
    /// Fast late-miss skip: keep the skipped requests (uncancelled) until they have an outcome or a boundary joins them.
    fileprivate func detachLateReads(_ requests: [ExpertLoadRequest]) { detachedReads += requests }
    /// The layer's experts in memory now (read only: no admission, reuse history or reference bits change).
    package func residentExperts(layer: Int) throws -> [Int] { try cache.draftResidentExperts(layer: layer) }
    /// Releases cold cache slabs until the ledger is back within its budget (never below the
    /// minimum cache). Safe from any thread: the cache lock orders it with readers, and busy
    /// slabs simply stay until their readers finish. Returns cache-owned bytes retired now;
    /// busy owners retire at a later safe boundary.
    @discardableResult
    package func trimToBudget() throws -> UInt64 {
        // shortcut: a preparation still running when the trim arrives counts its slabs as cache here, so the trim can take
        // that much extra; they come back at the next growth. Prepared slabs are released first and never count.
        let slab = UInt64(configuration.slotsPerSlab * PackStore.expertBytes)
        let minimum = min(
            (UInt64(ConversationPhaseBudget.minimumSlots * PackStore.expertBytes) + slab - 1) / slab * slab, configuration.prefillCacheBytes
        )
        return try cache.trimToBudget(minimumBytes: minimum)
    }
    /// Memory governor hook (any thread), after the budget grew back. Lazy decode restores only its allowance;
    /// prepared decode restores slabs off the token loop, for the next token boundary to attach (`attachGrowth`).
    package func prepareGrowth() {
        guard configuration.preallocateDecodeSlabs else {
            cache.restoreGrowthLimit()
            return
        }
        let cache = cache
        let floor = ledger.availableFloorBytes
        let wire = configuration.wireDecodeSlabs
        DispatchQueue.global(qos: .utility).async { _ = try? cache.prepareGrowth(minimumAvailableBytes: floor, wire: wire) }
    }
    /// Token boundary (serving thread, nothing pre-encoded queued): adds the prepared slabs. Returns slabs added.
    @discardableResult package func attachGrowth() -> Int { cache.attachGrowth() }
    /// Allocate and fault the complete admitted decode cache before TTFT ends.
    /// No expert reads, route predictions, evictions or cache-policy training.
    package func prepareDecodeSlabs(shouldContinue: () -> Bool) throws {
        guard configuration.preallocateDecodeSlabs else { return }
        guard isDecodePhase else { throw EngineError("Slab preparation requires decode phase") }
        let started = ProcessInfo.processInfo.systemUptime
        try joinReadProducers()
        let touch = try cache.prepareSlabs(
            shouldContinue: shouldContinue, minimumAvailableBytes: ledger.availableFloorBytes, wire: configuration.wireDecodeSlabs)
        var receipt = decodeSlabPreparation ?? DecodeSlabPreparation()
        if configuration.wireDecodeSlabs {
            receipt.slabsWired = (receipt.slabsWired ?? 0) + (touch.slabsWired ?? 0)
            receipt.pagesWired = (receipt.pagesWired ?? 0) + (touch.pagesWired ?? 0)
            receipt.wireFallbacks = (receipt.wireFallbacks ?? 0) + (touch.wireFallbacks ?? 0)
            receipt.lastWiredBytes =
                touch.coveredBytes - (touch.wireFallbackBytes ?? 0)
        }
        receipt.transitions += 1
        receipt.seconds += ProcessInfo.processInfo.systemUptime - started
        receipt.lastTargetBytes = touch.targetBytes
        receipt.lastCoveredBytes = touch.coveredBytes
        receipt.physicalHighWater = max(receipt.physicalHighWater, touch.physicalFootprintHighWater)
        receipt.minimumHeadroom = min(receipt.minimumHeadroom, touch.minimumAvailableMemoryEstimate)
        decodeSlabPreparation = receipt
    }
    private func setPhase(_ next: Phase, prefillBytes: UInt64? = nil, decodeBytes: UInt64? = nil, recoveryBytes: UInt64? = nil) throws {
        guard activeWindow == nil, readAhead == nil, !adaptivePrefillScope else {
            throw EngineError("Expert cache phase change requires joined windows", code: "expert_phase_busy")
        }
        joinDetachedReads()
        loader.joinRetainedReads()
        // Reads whose handles were dropped (a running unrouted prediction, a skipped late read pruned once it had an
        // outcome) can hold their drive job a moment longer; they end on their own, so wait for them (bounded).
        let waitStart = DispatchTime.now().uptimeNanoseconds
        var waited: Double { Double(DispatchTime.now().uptimeNanoseconds - waitStart) / 1e9 }
        while !scheduler.snapshot().drives.allSatisfy({ $0.outstanding == 0 }) {
            guard waited < 5 else {
                throw EngineError("Expert cache phase change requires joined windows and reads", code: "expert_phase_busy")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
        if waited > 0.1 {
            FileHandle.standardError.write(Data(String(format: "origami: phase change waited %.2f s for reads to end\n", waited).utf8))
        }
        guard phase == next || phaseTransitions < UInt64.max else {
            throw EngineError("Expert cache phase counter exhausted", code: "expert_phase_state")
        }
        let bytes = next == .prefill ? (prefillBytes ?? configuration.prefillCacheBytes) : (decodeBytes ?? configuration.cacheBytes)
        let recovery = recoveryBytes ?? bytes
        guard bytes >= min(UInt64(ConversationPhaseBudget.minimumSlots * PackStore.expertBytes), configuration.prefillCacheBytes),
            bytes <= configuration.cacheBytes,
            bytes % UInt64(configuration.slotsPerSlab * PackStore.expertBytes) == 0,
            recovery >= bytes, recovery <= configuration.cacheBytes,
            recovery % UInt64(configuration.slotsPerSlab * PackStore.expertBytes) == 0
        else {
            throw EngineError("Prospective phase cache is outside configured slab bounds", code: "invalid_configuration")
        }
        // A phase entered under pressure must still be able to recover its full admitted target. Prepared decode
        // restores slabs off-loop; lazy decode restores only the allowance and demand supplies the records.
        // Set both limits under the cache lock so a concurrent lazy recovery cannot restore the preceding phase.
        if next == .decode, phase == .prefill {
            let prompt = cache.snapshot().byteLimit
            prefillScheduling.promptCacheSlots += prompt / UInt64(PackStore.expertBytes)
            prefillScheduling.promptCacheSlotsBelowDecode += (bytes > prompt ? bytes - prompt : 0) / UInt64(PackStore.expertBytes)
        }
        try cache.resizeQuiescent(
            to: bytes, growthCeiling: next == .decode ? recovery : 0,
            growthBudget: recoveryBytes == nil ? nil : ledger.maximumUsableBytes)
        if next == .decode, configuration.reuseAdmission {
            cache.setStorageCosts(try loader.storageCostMultipliers(calibration: configuration.sourceCalibration))
        }
        try cache.setReuseDecode(next == .decode)
        cache.setPrefillScan(next == .prefill)
        try loader.setReplicasEnabled(next == .decode && configuration.tailReplicas)
        loader.setDecoding(next == .decode)
        if phase != next {
            phaseTransitions += 1
            phase = next
        }
    }
    package func beginPrefillReadAhead(layers: Int, rows: Int? = nil) throws {
        guard configuration.prefillReadAhead else { return }
        guard let generation, phase == .prefill, activeWindow == nil, readAhead == nil, !adaptivePrefillScope else {
            throw EngineError("Read-ahead requires an idle prefill scope")
        }
        if let rows {
            guard !adaptivePrefillScope, (1...8192).contains(rows), (1...48).contains(layers) else {
                throw EngineError("Invalid adaptive prefill scope")
            }
            adaptivePrefillScope = true
            adaptivePrefillLayers = layers
            prefillScopeCompleted = false
            // After a memory cut a broad scope still runs, at the width of the smaller cache (`held`): it fills only
            // slots the current budget holds. It falls back to selective reads, per scope, while pressure is live or
            // when the cache cannot hold one layer plus the demand window. Reads only: the output is unchanged.
            let observedBroad = prefillReadPolicy.begin(rows: rows)
            let live = observedBroad && !ledger.optionalReadsAllowed
            let small =
                observedBroad && !live
                && min(configuration.effectivePrefillCacheSlots, cache.slotLimit) < 512 + configuration.demandWindow
            if !observedBroad || live || small {
                if live || small && ledger.usableBytes < ledger.maximumUsableBytes { ledger.noteOptionalRefusal(live: live) }
                prefillScheduling.selectiveScopes += 1
                return
            }
            prefillScheduling.broadScopes += 1
        }
        readAhead = try PrefillExpertReadAhead(
            loader: loader, generation: generation,
            configuration: configuration, layers: layers, cacheSlots: { [cache] in cache.slotLimit(sizeClass: $0) },
            layerClasses: cache.sizes.layerClasses)
    }
    package func finishPrefillReadAheadLayer(_ layer: Int) throws {
        if let readAhead, layer < readAhead.layers { try readAhead.finishLayer(layer) }
        if adaptivePrefillScope && layer + 1 == adaptivePrefillLayers { prefillScopeCompleted = true }
    }
    private func retirePrefillReadAhead() {
        if let readAhead {
            readAhead.close()
            let snapshot = readAhead.snapshot()
            lastReadAhead = snapshot
            self.readAhead = nil
            prefillScheduling.readAheadSubmitted += UInt64(snapshot.submitted)
            prefillScheduling.readAheadConsumed += UInt64(snapshot.consumed)
            prefillScheduling.readAheadSkipped += UInt64(snapshot.skipped)
            prefillScheduling.readAheadDemandBypasses += UInt64(snapshot.demandBypasses ?? 0)
            prefillScheduling.readAheadWaitSeconds += snapshot.requestWaitSeconds
        }
    }
    package func finishPrefillReadAhead() {
        retirePrefillReadAhead()
        if adaptivePrefillScope {
            prefillReadPolicy.end(completed: prefillScopeCompleted, domain: ModelLayout.routedExperts)
            adaptivePrefillScope = false
            adaptivePrefillLayers = 0
            prefillScopeCompleted = false
        }
    }
    /// `wait` false (the MTP pair step, which decodes plainly instead): a busy cache fails at once.
    fileprivate func request(
        _ key: ExpertKey, generation: IOScheduler.Generation,
        shouldContinue: () -> Bool, wait: Bool = true
    ) throws -> ExpertLoadRequest {
        // Layers past the read-ahead's end (a final layer that skips its unread rows) read on demand.
        if let readAhead, key.layer < readAhead.layers { return try readAhead.request(key, shouldContinue: shouldContinue) }
        if !wait {
            return try loader.request(
                key, generation: generation, priority: .demand, policy: sourcePolicy,
                calibration: configuration.sourceCalibration)
        }
        return try Self.waitingForSlot(shouldContinue, landed: { scheduler.finishedReads }) {
            try loader.request(
                key, generation: generation, priority: .demand,
                policy: sourcePolicy, calibration: configuration.sourceCalibration)
        }
    }
    /// The memory governor can shrink the cache between a slot request and the reads holding it, so a demand may find
    /// every remaining slot reading, reserved or held by the GPU (`expert_slots_busy`). Those finish within moments:
    /// wait for one, as the prompt read-ahead does, instead of failing the request. While reads keep landing (`landed`,
    /// the reads finished so far, moves; a landed read frees a slot) a slow drive only makes the request slower; `limit`
    /// seconds of wall time in which no read lands mean a stuck cache or a hung read, and fail it.
    package static func waitingForSlot<T>(
        _ shouldContinue: () -> Bool, landed: () -> UInt64, limit: Double = 30,
        _ body: () throws -> T
    ) throws -> T {
        var idleSince = DispatchTime.now().uptimeNanoseconds
        var last = landed()
        while true {
            do { return try body() } catch let error as EngineError where error.code == "expert_slots_busy" {
                let now = DispatchTime.now().uptimeNanoseconds
                let count = landed()
                if count != last {
                    last = count
                    idleSince = now
                } else if Double(now - idleSince) / 1e9 >= limit {
                    throw error
                }
                guard shouldContinue() else { throw IOOperationCancelled() }
                Thread.sleep(forTimeInterval: 0.005)
            }
        }
    }
    /// Capture immutable generation/policy on the serial owner. The listener
    /// only calls the thread-safe loader; it never mutates execution counters.
    package func earlyDemandFactory() throws -> @Sendable ([Int], Int, @Sendable () -> Bool) throws -> [ExpertLoadRequest] {
        guard let generation, phase == .decode else { throw EngineError("Early demand requires decode generation") }
        let loader = loader
        let cache = cache
        let scheduler = scheduler
        let policy = sourcePolicy
        let calibration = configuration.sourceCalibration
        return { ids, layer, keepGoing in
            var requests: [ExpertLoadRequest] = []
            do {
                guard ids.count == 10, Set(ids).count == 10 else { throw EngineError("Early demand is exactly ten keys") }
                for id in try cache.beginDecodeRow(layer: layer, experts: ids) {
                    guard keepGoing() else { throw IOOperationCancelled() }
                    requests.append(
                        try ExpertExecution.waitingForSlot(keepGoing, landed: { scheduler.finishedReads }) {
                            try loader.request(
                                ExpertKey(layer: layer, expert: id), generation: generation,
                                priority: .demand, policy: policy, calibration: calibration)
                        })
                }
                return requests.sorted { $0.key.expert < $1.key.expert }
            } catch {
                for request in requests { _ = loader.cancel(request) }
                for request in requests { loader.joinDiscarding(request) }
                throw error
            }
        }
    }
    /// Decode pre-routing: the callback issues at most ten speculative reads for
    /// one target layer through the loader's speculative path. No reuse-history
    /// row, no cache-policy training; every outcome is returned for counting.
    /// `overrideWidth` (decode_prerouting_slot_override; 0 = off) lets at most
    /// that many keys per target layer, in rank order, past the reuse-score rule
    /// (ExpertCache.acquireSpeculative overrideScore); the rest stay noSlot.
    package func preroutingFactory(overrideWidth: Int = 0) throws -> DecodePrerouting.Factory {
        guard preroutingEnabled, let generation, phase == .decode else {
            throw EngineError("Pre-routing requires an enabled decode generation")
        }
        guard (0...10).contains(overrideWidth), overrideWidth == 0 || configuration.reuseAdmission else {
            throw EngineError("Pre-routing slot override requires reuse admission and a width in 1...10")
        }
        let loader = loader
        let policy = sourcePolicy
        let calibration = configuration.sourceCalibration
        return { ids, layer, keepGoing in
            var issue = DecodePrerouting.Issue()
            do {
                guard (1...10).contains(ids.count), Set(ids).count == ids.count else {
                    throw EngineError("Pre-routing issues at most ten distinct keys")
                }
                for id in ids {
                    guard keepGoing() else { throw IOOperationCancelled() }
                    let overriding = issue.overrides.count < overrideWidth
                    switch try loader.requestSpeculative(
                        ExpertKey(layer: layer, expert: id), generation: generation,
                        policy: policy, calibration: calibration, overrideScore: overriding)
                    {
                    case .issued(let request):
                        issue.requests.append(request)
                        issue.issued += 1
                    case .overridden(let request):
                        issue.requests.append(request)
                        issue.issued += 1
                        issue.overrides.append(id)
                    case .resident, .inFlight: issue.resident += 1
                    case .noSlot:
                        issue.noSlot += 1
                        if overriding { issue.overrideNoVictim += 1 } else if overrideWidth > 0 { issue.overrideWidthExhausted += 1 }
                    case .refused: issue.refused += 1
                    }
                }
                return issue
            } catch {
                for request in issue.requests { _ = loader.cancel(request) }
                for request in issue.requests { loader.joinDiscarding(request) }
                throw error
            }
        }
    }
    /// `decode_prerouting_read_hygiene` (nil when off), serial owner at adoption: of the
    /// predicted requests whose keys the authoritative route did not select, cancel those
    /// still queued and join their cancellation completions, so every cancelled key is out of
    /// the loader (slot free, admission released) before any later request can name it.
    /// Started reads finish as before. `races` counts requests found shared with another
    /// handle or still in flight after their join (expected 0).
    package func preroutingCancelUnrouted() -> (([ExpertLoadRequest]) -> (cancelled: Int, races: Int))? {
        guard readHygiene else { return nil }
        let loader = loader
        return { requests in
            var cancelled: [ExpertLoadRequest] = []
            var races = 0
            for request in requests {
                switch loader.cancelQueuedSpeculative(request) {
                case .cancelled: cancelled.append(request)
                case .notQueued: break
                case .shared: races += 1
                }
            }
            for request in cancelled {
                loader.joinDiscarding(request)
                if loader.isInFlight(request) { races += 1 }
            }
            return (cancelled.count, races)
        }
    }
    /// Read hygiene: hold every predicted-read start while decode PLE rows are read (nil when off).
    package func holdPredictedReads() -> IOPrefetchHold? { readHygiene ? scheduler.holdPrefetch() : nil }
    /// Read hygiene counters (nil when off).
    package var readHygieneCounts: IOScheduler.HygieneCounts? { scheduler.hygieneCounts() }
    package func earlyDemandDiscard() -> @Sendable ([ExpertLoadRequest]) -> Void {
        let loader = loader
        return { requests in
            for request in requests { _ = loader.cancel(request) }
            for request in requests { loader.joinDiscarding(request) }
        }
    }
    fileprivate var permitsLayerCommand: Bool {
        configuration.prefillLayerCommand && phase == .prefill && (readAhead != nil || adaptivePrefillScope)
    }
    fileprivate func didComputePrefillGroup(count: Int, resident: Int) {
        prefillScheduling.readyGroups += 1
        prefillScheduling.readyExperts += UInt64(count)
        prefillScheduling.residentExperts += UInt64(resident)
    }
    fileprivate func releaseReadAhead(_ keys: ArraySlice<ExpertKey>) { readAhead?.release(keys) }
    /// `historyRow` (the MTP verify step's union): the ten experts that drive decode reuse admission; the rest of the
    /// union is requested directly.
    public func makeWindow(
        layer: Int, experts: [Int], staged: [ExpertLoadRequest] = [], historyRow: [Int]? = nil,
        shouldContinue: @escaping () -> Bool = { true }
    ) throws -> ExpertDemandWindow {
        guard let generation, activeWindow == nil else {
            throw EngineError("Expert window requires one active sequence and no earlier window")
        }
        guard (0..<48).contains(layer), experts.count <= 512, !experts.isEmpty else { throw EngineError("Invalid layer demand union") }
        routing?.record(layer: layer, experts: experts)
        let keys = try Set(experts).sorted().map { try ExpertKey(layer: layer, expert: $0) }
        if adaptivePrefillScope && layer < adaptivePrefillLayers {
            prefillReadPolicy.observe(experts: keys.count, domain: ModelLayout.routedExperts)
            prefillScheduling.routedExperts += UInt64(keys.count)
            prefillScheduling.routedLayers += 1
            // Routing is authoritative now. Stop a formerly useful scan as
            // soon as this layer is sparse, or pressure is live (its held
            // records go back to a trim). Completed bytes remain exact reusable
            // cache hits; all remaining work goes through bounded mandatory demand.
            let live = !ledger.optionalReadsAllowed
            if readAhead != nil
                && (live || Double(keys.count) / Double(ModelLayout.routedExperts) < PrefillReadPolicy.minimumCoverage)
            {
                if live { ledger.noteOptionalRefusal(live: true) }
                retirePrefillReadAhead()
                prefillScheduling.broadDemotions += 1
            }
        }
        var prepared = staged
        if phase == .decode && staged.isEmpty && configuration.reuseAdmission {
            do {
                for id in try cache.beginDecodeRow(layer: layer, experts: historyRow ?? keys.map(\.expert)) {
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    prepared.append(
                        try request(
                            ExpertKey(layer: layer, expert: id), generation: generation, shouldContinue: shouldContinue,
                            wait: historyRow == nil))
                }
                for key in keys where historyRow != nil && !prepared.contains(where: { $0.key == key }) {
                    prepared.append(try request(key, generation: generation, shouldContinue: shouldContinue, wait: false))
                }
                prepared.sort { $0.key.expert < $1.key.expert }
            } catch {
                for value in prepared { _ = loader.cancel(value) }
                for value in prepared { loader.joinDiscarding(value) }
                throw error
            }
        }
        let window: ExpertDemandWindow
        do {
            window = try ExpertDemandWindow(
                owner: self, keys: keys, generation: generation, staged: prepared, shouldContinue: shouldContinue)
        } catch {
            // The window refuses mismatched requests before owning them: discard the ones made here (the caller owns `staged`).
            if staged.isEmpty {
                for value in prepared { _ = loader.cancel(value) }
                for value in prepared { loader.joinDiscarding(value) }
            }
            throw error
        }
        activeWindow = window
        windows += 1
        demanded += UInt64(keys.count)
        return window
    }
    /// Cancels and joins actual reads, then invalidates the generation. Call
    /// only after synchronous expert computations have left their consume body.
    public func finishSequence() throws {
        // Skipped (detached) and predicted reads end with the generation; dropping a handle never frees a running read's slot.
        defer {
            detachedReads.removeAll()
            adoptedPredictions.removeAll()
        }
        activeWindow?.close()
        finishPrefillReadAhead()
        if let generation { try loader.cancelAndJoin(generation) }
        generation = nil
        // Between requests the next phase change sizes the cache: no idle growth.
        cache.setGrowthCeiling(0)
    }
    fileprivate func closed(_ window: ExpertDemandWindow) {
        if activeWindow === window { activeWindow = nil }
    }
    fileprivate func didConsume() { consumed += 1 }
    package func makeDelayedRead() -> ExpertDelayedRead { ExpertDelayedRead(owner: self) }
    /// A demand the window uses that found its record outside memory, including a prediction still reading that the
    /// demand adopted (the miss it saved); counted once per window, since each key enters it once.
    fileprivate func countMiss(_ request: ExpertLoadRequest) {
        if !request.residentAtRequest {
            routing?.recordMiss(layer: request.key.layer, expert: request.key.expert, weight: phase == .decode ? 4 : 1)
            if phase == .decode && request.key.layer == 0 { cache.countLayerZeroMiss() }
        }
    }
    fileprivate func didComputeResident(_ count: Int) {
        if count > 0 {
            residentFirstWindows += 1
            residentFirstExperts += UInt64(count)
        }
    }
    fileprivate func joinAndPin(_ request: ExpertLoadRequest, shouldContinue: () -> Bool) throws -> LoadedExpert {
        let start = DispatchTime.now().uptimeNanoseconds
        defer { consumerJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }
        return try loader.joinAndPin(request, shouldContinue: shouldContinue)
    }
    fileprivate func withBatchMetadata<T>(count: Int, _ body: () throws -> T) throws -> T {
        // References, leases, buffer views, and error cleanup metadata. Host
        // activation/output tiles remain inside the executor's phase bound.
        let metadata = try ExpertExecutionMetadata(
            ledger: ledger, bytes: UInt64(4096 + count * 1024),
            category: "expert-consumer-batch-metadata")
        return try withExtendedLifetime(metadata) { try body() }
    }
    /// Three buffer-native projections over one pinned record. Each grid keeps
    /// the row-one QMV schedule independently for every token, even above the
    /// ordinary QMM dispatch threshold. An explicitly selected enclosing sweep
    /// instead retains full-K grouped QMM, including one-row buckets/tails.
    /// No weight data or typed arrays are copied in either mode.
    public static func evaluate(
        _ x: [Float], buffer: ExpertBuffer, kernels: ReferenceKernels,
        groupedExpert: Bool = false, simdMatrix: Bool = false
    ) throws -> [Float] {
        guard !x.isEmpty, x.count % 2560 == 0, x.count / 2560 <= 256 else { throw EngineError("Invalid expert activation geometry") }
        // The reference library is compiled for the dense layout; NVFP4 records need the fused kernels.
        guard ModelLayout.experts == AffineFormat(bits: 4, group: ModelLayout.group), ModelLayout.ggufExperts == nil else {
            throw EngineError("The reference expert path reads only the checkpoint's affine 4-bit records")
        }
        return try autoreleasepool {
            func projection(_ values: [Float], _ kind: ExpertProjection) throws -> [Float] {
                if groupedExpert {
                    return try kernels.qlinearPinnedBF16(
                        x: values, width: kind.width, outputs: kind.outputs,
                        words: buffer.buffer, scales: buffer.buffer, biases: buffer.buffer,
                        wordOffset: buffer.offset + kind.weightOffset, scaleOffset: buffer.offset + kind.scaleOffset,
                        biasOffset: buffer.offset + kind.biasOffset, simdMatrix: simdMatrix, groupedExpert: true
                    ).output
                }
                return try kernels.qmvPinnedBF16(
                    x: values, width: kind.width, outputs: kind.outputs, weights: buffer.buffer,
                    weightOffset: buffer.offset + kind.weightOffset, scaleOffset: buffer.offset + kind.scaleOffset,
                    biasOffset: buffer.offset + kind.biasOffset)
            }
            let gate = try projection(x, .gate)
            let up = try projection(x, .up)
            let product = zip(gate, up).map { LayerArithmetic.round(kernels.arithmetic.silu($0) * $1) }
            return try projection(product, .down)
        }
    }
    /// The cache counters alone, under the cache's lock: safe from any thread, unlike `snapshot()`, which reads the
    /// serving thread's window and read-ahead state.
    package var cacheSnapshot: ExpertCache.Snapshot { cache.snapshot() }
    public func snapshot() -> Snapshot {
        let current = cache.snapshot()
        var value = Snapshot(
            mode: configuration.mode, configuration: configuration, cache: current, io: scheduler.snapshot(),
            phase: phase, targetCacheSlots: Int(current.byteLimit / UInt64(PackStore.expertBytes)),
            targetCacheBytes: current.byteLimit,
            phaseTransitions: phaseTransitions, sequences: sequences, windows: windows,
            demandedExperts: demanded, consumedExperts: consumed, consumerJoinSeconds: consumerJoinSeconds,
            residentFirstWindows: residentFirstWindows, residentFirstExperts: residentFirstExperts,
            replicas: loader.replicaSnapshot(), store: storeSnapshot(), prefillReadAhead: readAhead?.snapshot() ?? lastReadAhead,
            decodeSlabPreparation: decodeSlabPreparation)
        value.expertSourceOnline = loader.onlineCalibrationSnapshot()
        value.prefillScheduling = prefillScheduling
        return value
    }
    /// Split-read counters (`SplitReads`); any thread.
    package var splitReads: SplitReadCounters { loader.splitReads }
    /// Record read times by drive; any thread.
    package var readTimes: SourceExpertLoader.ReadTimes { loader.readTimes }
    /// Counters of the online source calibration; nil when the configuration leaves it off.
    public var onlineCalibrationSnapshot: ExpertSourceOnlineSnapshot? { loader.onlineCalibrationSnapshot() }
}

/// Sliding window over a complete, sorted per-layer expert demand union. Only
/// demandWindow records may hold pending handoff pins at once. Ordinary
/// consumption follows expert order; selective prefill explicitly permits ready
/// groups whose independent canonical destinations are reduced only at the end.
/// A single reusable request-owned handoff. Retains the source read/slot until its original
/// expert has consumed it at the next attention boundary. Unlike skipped reads it contributes once.
package final class ExpertDelayedRead {
    private let owner: ExpertExecution
    private var request: ExpertLoadRequest?
    package var isPending: Bool { request != nil }
    fileprivate init(owner: ExpertExecution) { self.owner = owner }
    fileprivate func adopt(_ request: ExpertLoadRequest) throws {
        guard self.request == nil else { throw EngineError("Delayed expert slot already occupied") }
        self.request = request
    }
    package func consume(shouldContinue: () -> Bool, _ body: (ExpertBuffer) throws -> Void) throws {
        guard let pending = request else { throw EngineError("Delayed expert request is absent") }
        let lease = try owner.joinAndPin(pending, shouldContinue: shouldContinue)
        request = nil
        do { try body(lease.buffer) } catch {
            try? lease.retireAfterGPUCompletion()
            throw error
        }
        try lease.retireAfterGPUCompletion()
    }
    package func discard() {
        guard let request else { return }
        _ = owner.loader.cancel(request)
        owner.loader.joinDiscarding(request)
        self.request = nil
    }
    deinit { discard() }
}

public final class ExpertDemandWindow {
    public let keys: [ExpertKey]
    private let owner: ExpertExecution
    private let generation: IOScheduler.Generation
    private let shouldContinue: () -> Bool
    private var pending: [ExpertKey: ExpertLoadRequest] = [:]
    private var enqueued = 0, consumed = 0
    private var consumingBatch = false
    public var maximumBatchCount: Int { min(8, owner.configuration.demandWindow) }
    public private(set) var isClosed = false
    public var pendingCount: Int { pending.count }
    public var remainingCount: Int { keys.count - consumed }

    fileprivate init(
        owner: ExpertExecution, keys: [ExpertKey], generation: IOScheduler.Generation,
        staged: [ExpertLoadRequest] = [], shouldContinue: @escaping () -> Bool
    ) throws {
        self.owner = owner
        self.keys = keys
        self.generation = generation
        self.shouldContinue = shouldContinue
        guard
            staged.isEmpty
                || (staged.count == keys.count && staged.map(\.key) == keys
                    && staged.allSatisfy({ $0.generation == generation }))
        else { throw EngineError("Staged demand differs from joined routing") }
        for request in staged {
            pending[request.key] = request
            owner.countMiss(request)
        }
        enqueued = staged.count
        do { try fill() } catch {
            close()
            throw error
        }
    }
    deinit { close() }
    private func fill() throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        while enqueued < keys.count && pending.count < owner.configuration.demandWindow {
            let key = keys[enqueued]
            pending[key] = try owner.request(key, generation: generation, shouldContinue: shouldContinue)
            owner.countMiss(pending[key]!)
            enqueued += 1
        }
    }
    /// Body must finish every CPU/GPU access before returning or throwing. The
    /// pinned buffer-native QMV API is synchronous and satisfies this contract.
    /// Do not return/retain a view of the supplied storage or manually unpin it.
    public func consume<T>(expert: Int, _ body: (ExpertBuffer) throws -> T) throws -> T {
        guard !isClosed, !consumingBatch, consumed < keys.count, keys[consumed].expert == expert else {
            throw EngineError("Expert demand must be consumed once in sorted union order")
        }
        do {
            guard shouldContinue() else { throw IOOperationCancelled() }
            let key = keys[consumed]
            var request = pending.removeValue(forKey: key)
            guard request != nil else { throw EngineError("Expert window lost a demand request") }
            let lease = try owner.joinAndPin(request!, shouldContinue: shouldContinue)
            // Drop the host-only handoff pin before refill. The consumer pin
            // now owns the slot through the final synchronous GPU completion.
            request = nil
            let value: T
            do { value = try body(lease.buffer) } catch {
                try? lease.retireAfterGPUCompletion()
                throw error
            }
            try lease.retireAfterGPUCompletion()
            owner.releaseReadAhead(keys[consumed..<(consumed + 1)])
            consumed += 1
            owner.didConsume()
            if consumed == keys.count { close() } else { try fill() }
            return value
        } catch {
            close()
            throw error
        }
    }
    /// Join and pin the next sorted contiguous slice of the expert union. Every
    /// handoff handle is dropped before computation; no new request is submitted
    /// until all consumer pins retire. The synchronous body must join all GPU
    /// commands before returning OR throwing, exactly like consume's contract.
    /// Invalid order/size leaves the window intact for a corrected request.
    public func consumeBatch<T>(experts: [Int], _ body: ([ExpertBuffer]) throws -> T) throws -> T {
        try consumePinned(experts: experts, maximum: maximumBatchCount, body)
    }
    /// Exact prefill groups may run in completion order: they write disjoint
    /// canonical (token, rank) destinations, with one ordered reduction later.
    /// The next bounded read window is submitted before the current GPU group.
    /// Every callback must join its readers before returning or throwing.
    package func consumePrefillReady(_ body: ([Int], [ExpertBuffer]) throws -> Void) throws {
        guard owner.usesSelectivePrefill, owner.permitsLayerCommand, !isClosed, !consumingBatch,
            consumed == 0, (1...512).contains(keys.count)
        else {
            throw EngineError("Ready prefill requires a fresh selective layer window")
        }
        consumingBatch = true
        defer { consumingBatch = false }
        let wake = DispatchSemaphore(value: 0)
        var registered: Set<ExpertKey> = []
        func register() throws {
            for (key, request) in pending where registered.insert(key).inserted {
                try owner.loader.notifyPrefillCompletion(request) { wake.signal() }
            }
        }
        do {
            try owner.withBatchMetadata(count: keys.count) {
                try register()
                while consumed < keys.count {
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    let ready = keys.filter { key in pending[key].map { owner.loader.readFinished($0) } ?? false }
                    if ready.isEmpty {
                        _ = wake.wait(timeout: .now() + .milliseconds(25))
                        continue
                    }
                    var leases: [LoadedExpert] = []
                    leases.reserveCapacity(ready.count)
                    let resident = ready.reduce(0) { $0 + (pending[$1]!.residentAtRequest ? 1 : 0) }
                    func retire() throws {
                        var failure: Error?
                        for lease in leases {
                            do { try lease.retireAfterGPUCompletion() } catch { if failure == nil { failure = error } }
                        }
                        leases.removeAll()
                        if let failure { throw failure }
                    }
                    do {
                        for key in ready {
                            guard let request = pending.removeValue(forKey: key) else { throw EngineError("Ready prefill lost a demand") }
                            leases.append(try owner.joinAndPin(request, shouldContinue: shouldContinue))
                        }
                        // Consumer pins replace handoff pins. At most one group
                        // plus one demand window is live, below the admitted
                        // layer command's 512-pin bound.
                        try fill()
                        try register()
                        try body(ready.map(\.expert), leases.map(\.buffer))
                        try retire()
                        consumed += ready.count
                        for _ in ready { owner.didConsume() }
                        owner.didComputePrefillGroup(count: ready.count, resident: resident)
                    } catch {
                        let original = error
                        do { try retire() } catch {
                            throw EngineError("\(error); original ready prefill error: \(original)", code: "expert_batch_cleanup")
                        }
                        throw original
                    }
                }
                close()
            }
        } catch {
            close()
            throw error
        }
    }
    /// Pin the complete union under the retained producer or the explicitly
    /// admitted selective scope owner. The latter fills only routed keys.
    /// The body must join all GPU readers before returning or throwing.
    package func consumeLayer<T>(_ body: ([ExpertBuffer]) throws -> T) throws -> T {
        guard owner.permitsLayerCommand, !isClosed, !consumingBatch, consumed == 0,
            (1...512).contains(keys.count)
        else {
            throw EngineError("Layer command requires a fresh retained prefill window")
        }
        consumingBatch = true
        defer { consumingBatch = false }
        do {
            return try owner.withBatchMetadata(count: keys.count) {
                var leases: [LoadedExpert] = []
                leases.reserveCapacity(keys.count)
                func retire() throws {
                    var failure: Error?
                    for lease in leases {
                        do { try lease.retireAfterGPUCompletion() } catch { if failure == nil { failure = error } }
                    }
                    leases.removeAll()
                    if let failure { throw EngineError("Layer pin retirement failed: \(failure)", code: "expert_batch_cleanup") }
                }
                do {
                    for key in keys {
                        guard shouldContinue() else { throw IOOperationCancelled() }
                        var request = pending.removeValue(forKey: key)
                        guard request != nil else { throw EngineError("Layer demand disappeared") }
                        let lease = try owner.joinAndPin(request!, shouldContinue: shouldContinue)
                        leases.append(lease)
                        request = nil
                        try fill()
                    }
                    guard shouldContinue(), leases.capacity <= 1024 else { throw IOOperationCancelled() }
                    let buffers = leases.map(\.buffer)
                    guard buffers.capacity <= 1024, MemoryLayout<ExpertBuffer>.stride <= 256 else {
                        throw EngineError("Layer consumer metadata exceeds its admitted capacity")
                    }
                    let result = try body(buffers)
                    try retire()
                    owner.releaseReadAhead(keys[...])
                    consumed = keys.count
                    for _ in keys { owner.didConsume() }
                    close()
                    return result
                } catch {
                    let original = error
                    do { try retire() } catch {
                        throw EngineError("\(error); original layer error: \(original)", code: "expert_batch_cleanup")
                    }
                    throw original
                }
            }
        } catch {
            close()
            throw error
        }
    }

    /// Decode's complete sorted top-ten union. Ordinary batch limits remain
    /// unchanged. Validate window capacity before removing any pending handle.
    /// The whole union at once (the MTP verify step: two rows, up to twenty experts), past the usual demand window.
    package func consumeUnion<T>(_ body: ([ExpertBuffer]) throws -> T) throws -> T {
        guard consumed == 0, !isClosed else { throw EngineError("Union consumption requires a fresh window") }
        while enqueued < keys.count {
            guard shouldContinue() else { throw IOOperationCancelled() }
            let key = keys[enqueued]
            pending[key] = try owner.request(key, generation: generation, shouldContinue: shouldContinue, wait: false)
            owner.countMiss(pending[key]!)
            enqueued += 1
        }
        return try consumePinned(experts: keys.map(\.expert), maximum: keys.count, body)
    }
    public func consumeTen<T>(experts: [Int], _ body: ([ExpertBuffer]) throws -> T) throws -> T {
        guard owner.configuration.demandWindow >= 10, consumed == 0, keys.count == 10, experts.count == 10 else {
            throw EngineError("Gathered decode requires a complete ten-expert demand window")
        }
        return try consumePinned(experts: experts, maximum: 10, body)
    }

    /// A single fixed split of the same sorted demand: known resident records,
    /// then the remaining records. No new requests or cache policy (polls only for skipLate), except a substitute's
    /// pin, which `substitute` asks the owner for.
    /// Both callbacks must join all readers of their supplied buffers before
    /// returning OR throwing. All consumer pins survive the complete scope.
    /// Empty/all-resident windows use the original complete ten-expert group.
    /// Count before the once-only consumption removes requests; used only for
    /// truthful miss-window command attribution, never routing or admission.
    package var residentAtRequestCount: Int { pending.values.filter(\.residentAtRequest).count }

    /// `skipLate` (fast late-miss skip; nil = exact, unchanged): called once after `ready` with the remaining
    /// experts whose reads have no outcome yet ("late") and how many may be skipped within the scheduler reserve
    /// for skipped reads; it returns the late experts that contribute nothing (at most that many).
    /// Those requests are released uncancelled (their reads complete into the cache, joined at the next read
    /// boundary) and `finish` receives a pinned kept record in their place, which the caller's zero gate removes.
    /// `substitute` (quality dial `lateMiss: substitute`): called with the skipped experts; it returns an expert in
    /// memory for any of them, which is pinned and given to `finish` in that one's place.
    /// `dropped` (quality dial): routed experts outside the window, which the caller's zero gate removes; `finish`
    /// receives them after the window's, each with a pinned kept record, so the ranks still cover the routed ten.
    package func consumeTenReadyFirst<T>(
        experts: [Int],
        joinReaders: () -> Void = {},
        skipLate: (([Int], Int) throws -> [Int])? = nil,
        substitute: (([Int]) throws -> [Int: Int])? = nil,
        dropped: [Int] = [],
        residentEarly: Bool = false,
        early earlyExperts: Set<Int>? = nil,
        delayed: ExpertDelayedRead? = nil,
        ready: ([Int], [ExpertBuffer]) throws -> Void,
        finish: ([Int], [ExpertBuffer]) throws -> T
    ) throws -> T {
        guard !isClosed, !consumingBatch, owner.configuration.demandWindow >= 10, consumed == 0, !keys.isEmpty,
            keys.count + dropped.count <= 10, experts == keys.map(\.expert), pending.count == keys.count,
            Set(dropped).count == dropped.count, Set(dropped).isDisjoint(with: experts)
        else {
            throw EngineError("Resident-first decode requires the sorted window of the routed ten (less any dropped)")
        }
        consumingBatch = true
        defer { consumingBatch = false }
        do {
            return try owner.withBatchMetadata(count: 10) {
                let hits = keys.filter { pending[$0]!.residentAtRequest }
                // An all-resident window runs as one group in `finish`, unless the caller wants every hit early.
                // Queued decode experts name the early group themselves: the resident experts the GPU computes from its table.
                let early =
                    earlyExperts.map { set in keys.filter { set.contains($0.expert) } }
                    ?? (hits.count < keys.count || residentEarly ? hits : [])
                let later = keys.filter { !early.contains($0) }
                var leases = [LoadedExpert]()
                leases.reserveCapacity(10)
                func pin(_ group: [ExpertKey]) throws -> [ExpertBuffer] {
                    var buffers = [ExpertBuffer]()
                    buffers.reserveCapacity(group.count)
                    for key in group {
                        guard shouldContinue() else { throw IOOperationCancelled() }
                        var request = pending.removeValue(forKey: key)
                        guard let current = request else { throw EngineError("Resident-first demand disappeared") }
                        let lease = try owner.joinAndPin(current, shouldContinue: shouldContinue)
                        leases.append(lease)
                        buffers.append(lease.buffer)
                        request = nil
                    }
                    return buffers
                }
                func retire() throws {
                    var failure: Error?
                    for lease in leases {
                        do { try lease.retireAfterGPUCompletion() } catch { if failure == nil { failure = error } }
                    }
                    leases.removeAll()
                    if let failure { throw failure }
                }
                do {
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    try ready(early.map(\.expert), pin(early))
                    owner.didComputeResident(early.count)
                    var skipped: [ExpertKey] = []
                    if let skipLate {
                        let capacity = owner.detachedReadCapacity()
                        let late = later.filter { !owner.loader.readFinished(pending[$0]!) }
                        let named = try skipLate(late.map(\.expert), capacity)
                        skipped = late.filter { named.contains($0.expert) }
                        guard skipped.count == named.count, skipped.count < keys.count, skipped.count <= capacity else {
                            throw EngineError("Fast skip must name distinct late records within the read reserve and keep one")
                        }
                        if let delayed, !skipped.isEmpty {
                            guard skipped.count == 1, substitute == nil else {
                                throw EngineError("Delayed decode permits one original expert")
                            }
                            try delayed.adopt(pending.removeValue(forKey: skipped[0])!)
                        } else {
                            owner.detachLateReads(skipped.map { pending.removeValue(forKey: $0)! })
                        }
                    }
                    var buffers = try pin(later.filter { !skipped.contains($0) })
                    var replacements: [ExpertKey: ExpertBuffer] = [:]
                    if let substitute, !skipped.isEmpty {
                        for (expert, replacement) in try substitute(skipped.map(\.expert)) {
                            guard let late = skipped.first(where: { $0.expert == expert }), replacements[late] == nil,
                                !keys.contains(where: { $0.expert == replacement }), !dropped.contains(replacement)
                            else {
                                throw EngineError("A late-miss substitute must replace a skipped record with an unrouted expert")
                            }
                            // shortcut: a substitute evicted since the caller saw it in memory is read and waited for here
                            // (rare); the reply is the same, only that step is slower.
                            let request = try owner.request(
                                ExpertKey(layer: late.layer, expert: replacement), generation: generation,
                                shouldContinue: shouldContinue)
                            let lease = try owner.joinAndPin(request, shouldContinue: shouldContinue)
                            leases.append(lease)
                            replacements[late] = lease.buffer
                        }
                    }
                    if !skipped.isEmpty {
                        var kept = buffers.makeIterator()
                        buffers = later.map { skipped.contains($0) ? replacements[$0] ?? leases[0].buffer : kept.next()! }
                    }
                    buffers += dropped.map { _ in leases[0].buffer }
                    let result = try finish(later.map(\.expert) + dropped, buffers)
                    joinReaders()
                    try retire()
                    consumed = 10
                    for _ in keys { owner.didConsume() }
                    close()
                    return result
                } catch {
                    let original = error
                    // A later pin/read can fail before finish is entered.
                    // Ready-group readers must join before any lease retires.
                    joinReaders()
                    do { try retire() } catch {
                        throw EngineError("\(error); original resident-first error: \(original)", code: "expert_batch_cleanup")
                    }
                    throw original
                }
            }
        } catch {
            close()
            throw error
        }
    }
    private func consumePinned<T>(experts: [Int], maximum: Int, _ body: ([ExpertBuffer]) throws -> T) throws -> T {
        guard !isClosed, !consumingBatch, (1...maximum).contains(experts.count),
            experts.count <= keys.count - consumed,
            experts == keys[consumed..<(consumed + experts.count)].map(\.expert)
        else {
            throw EngineError("Expert batch must be the next bounded sorted contiguous union slice")
        }
        consumingBatch = true
        defer { consumingBatch = false }
        do {
            return try owner.withBatchMetadata(count: experts.count) {
                var leases: [LoadedExpert] = []
                leases.reserveCapacity(experts.count)
                func retire() throws {
                    var failure: Error?
                    for lease in leases {
                        do { try lease.retireAfterGPUCompletion() } catch { if failure == nil { failure = error } }
                    }
                    leases.removeAll()
                    if let failure { throw EngineError("Expert batch pin retirement failed: \(failure)", code: "expert_batch_cleanup") }
                }
                do {
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    for offset in experts.indices {
                        let key = keys[consumed + offset]
                        var request = pending.removeValue(forKey: key)
                        guard request != nil else { throw EngineError("Expert batch lost a pending demand request") }
                        let lease = try owner.joinAndPin(request!, shouldContinue: shouldContinue)
                        leases.append(lease)
                        request = nil  // retire this record's producer handoff pin
                    }
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    let value = try body(leases.map(\.buffer))
                    try retire()
                    owner.releaseReadAhead(keys[consumed..<(consumed + experts.count)])
                    consumed += experts.count
                    for _ in experts { owner.didConsume() }
                    if consumed == keys.count { close() } else { try fill() }
                    return value
                } catch {
                    let original = error
                    do { try retire() } catch {
                        throw EngineError("\(error); original batch error: \(original)", code: "expert_batch_cleanup")
                    }
                    throw original
                }
            }
        } catch {
            close()
            throw error
        }
    }

    /// Idempotent cleanup, including partial submission and exceptional compute.
    /// No current consumer is retired here: its synchronous consume scope owns
    /// that pin. Only pending producer requests are cancelled and joined.
    public func close() {
        guard !isClosed else { return }
        isClosed = true
        let requests = Array(pending.values)
        for request in requests { _ = owner.loader.cancel(request) }
        for request in requests { owner.loader.joinDiscarding(request) }
        pending.removeAll()
        owner.closed(self)
    }
}
