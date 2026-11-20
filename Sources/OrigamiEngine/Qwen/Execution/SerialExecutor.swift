import Foundation
import Metal

/// The model executor: runs the prefill scopes and decode tokens of one sequence at a time
/// through the 48 layers, with explicit chronological state, canonical (fixed-order) reductions
/// and no inference framework. The whole-schedule paths (forwardWholePrefill, forwardWholeDecode,
/// the MTP pair step) encode GPU commands per layer; a tile or layer whose exception flags are set
/// is replayed on the original host-joined path, so a flagged value never reaches the state.
public final class SerialExecutor {
    public let weights: ReferenceWeights
    public let ledger: MemoryLedger
    public let contextLimit: Int
    /// The 262,144 tier with its key/value pages in a file (false when it was planned in memory).
    public let fileBackedContext: Bool
    let registerQSADecode: Bool
    let raggedPrefillScope: Bool
    private let math: LayerMath
    private let qsaDecodeReuse: Bool
    private let prefillMoETransfer: PrefillMoETransfer?
    private let gatheredExperts: GatheredExpertDecode?
    private let gdnDeviceBlock: GDNDeviceBlock?
    private let wholeSchedule: WholeExecutionSchedule?
    private let ple: ReferencePLE
    /// MTP drafting (`speculation`): how many final multi-stream residual rows each forward keeps (`lastHidden`, the
    /// newest last). 0 keeps none.
    package var captureHiddenRows = 0
    private var pairGrids: MTLBuffer?
    private var pairResidencyBackoff = PairResidencyBackoff()
    /// GPU seconds of the last resident pair step's commands (also when it missed), for the speculation log.
    package private(set) var pairResidentGPUSeconds = 0.0
    package var pairResidentMisses = 0
    package private(set) var lastHidden: [[Float]] = []
    private let devicePLE: DevicePLE?
    private var devicePLEDecodeWorkspace: DevicePLE.Workspace?
    /// Device PLE lookahead: the driver names the next prefill scope's tokens; once the current scope's
    /// layer 1 has consumed its rows, that scope's rows are read in the background. A scope adopts the
    /// prefetch only when the two preceding tokens (all a PLE row hash reads, and all that
    /// SequenceState.history keeps) and its tokens match exactly; anything else discards it.
    package var nextPrefillScope: [Int]?
    private var pendingPLEPrefetch: (prefix: [Int64], ids: [Int], prefetch: PLERowPrefetch)?
    /// Host-stage prefetch: embedding rows read concurrently before the expert read-ahead (and the
    /// next scope's rows behind layers 2-47, adopted only for identical tokens), and dense layer
    /// weights warmed into the resident caches ahead of the prefill layer loop.
    private let hostStagePrefetch: Bool
    /// Decode next-layer pre-routing (see DecodePrerouting); its counters live in the whole schedule.
    private let decodeNextLayerPrerouting: Bool
    /// Pre-routing slot override width per target layer (0 = off; see ExpertCache.acquireSpeculative).
    private let decodePreroutingOverrideWidth: Int
    /// `decode_prerouting_read_hygiene`: cancel queued unrouted predictions at adoption, give demand
    /// spans priority in the reader pool and hold predicted starts while decode PLE rows are read.
    private let decodePreroutingReadHygiene: Bool
    /// Per-request fast decode allowance (`fast.allow_opt_in`): the τ an opted-in request runs with; nil = refused.
    private let fastOptInTau: Double?
    /// The current request's quality dial (exact: the unchanged path) and its late-miss τ (nil unless the dial skips
    /// or substitutes late misses); set only by `selectRequestMode`. See QualityDial and FastLateMissSkip.
    private var requestDial = QualityDial.exact
    private var delayedStorage: DelayedExpertStorage?
    private var delayedRead: ExpertDelayedRead?
    private var delayedStorageAttempted = false
    public private(set) var delayedExpertCounters = DelayedExpertCounters()
    public func releaseDelayedExpertStorage() {
        // Every token's fixed boundary is joined before the forward returns, including cancellation.
        delayedRead?.discard()
        delayedRead = nil
        delayedStorage = nil
        delayedStorageAttempted = false
    }
    private var fastLateMissSkipTau: Double?
    /// The configuration allows per-request non-exact quality levels.
    public var fastAllowed: Bool { fastOptInTau != nil }
    /// Selects the quality dial of the request about to run (before its first forward). A non-exact dial needs the
    /// configuration's allowance; otherwise it is refused, never run exact instead.
    public func selectRequestMode(_ quality: QualityDial) throws {
        let tau = try FastLateMissSkip.requestTau(quality.validated(), allowedTau: fastOptInTau)
        guard !executingScope else { throw EngineError("Request mode cannot change inside a forward scope", code: "expert_phase_busy") }
        if quality.delayedExperts != nil, !delayedStorageAttempted {
            delayedStorageAttempted = true
            delayedExpertCounters = .init()
            guard wholeSchedule != nil, expertExecution != nil else {
                throw EngineError("Delayed Qwen experts require whole decode", code: "invalid_configuration")
            }
            if ledger.optionalGrowthAllowed, let device = MTLCreateSystemDefaultDevice() {
                do {
                    delayedStorage = try DelayedExpertStorage(device: device, ledger: ledger, hidden: 2560)
                    delayedRead = expertExecution!.makeDelayedRead()
                } catch let error as EngineError where error.code == "insufficient_memory" { delayedExpertCounters.capacityRefusals += 1 }
            } else {
                delayedExpertCounters.capacityRefusals += 1
            }
        }
        requestDial = quality
        fastLateMissSkipTau = quality.lateMiss == .wait || quality.delayedExperts != nil ? nil : tau
    }
    /// Double-buffered prefill tiles: two C256 workspaces; a GDN attention tile (layers other than 1)
    /// or a canonical mix tile is joined and flag-checked only after the next tile was submitted.
    private let prefillDoubleBuffer: Bool
    private let quantizedMMABF16: Bool, gdnExactKernels: Bool
    private let gdnPreparationHelper: FusedGDNPreparation?
    /// Exact QSA prefill kernels (`qsa_exact_kernels`), `hc_exact_fusion` and `hc_decode_fusion`.
    private let qsaExactKernels: Bool, hcExactFusion: Bool, hcDecodeFusion: Bool
    private var pendingEmbeddingPrefetch: (ids: [Int], prefetch: EmbeddingRowPrefetch)?
    private var embeddingLayout: EmbeddingRowPrefetch.Layout?
    /// `io_reader_pool`: decode PLE rows are read by persistent dedicated threads, not GCD's pool.
    private let decodeRowPool: SpanReadPool?
    /// `decode_ple_internal_race` (nil when off): the original-span row pool.
    private let decodePLERacePool: SpanReadPool?
    /// `prefill_ple_overlap` (nil when off): the original-span pool that races every prefill scope's PLE rows
    /// (alternate source descending). A scope's rows are no longer joined before layer 0:
    /// layer 0's load and attention tiles run while they are read, the expert read-ahead starts as soon as
    /// they are complete (polled between layer-0 tiles, joined at the latest before layer 0's MoE), and a
    /// host tail tile takes its rows from the joined scope instead of reading them again.
    private let prefillPLERacePool: SpanReadPool?
    /// The scope whose rows are being read behind layer 0 (set only inside forwardWholePrefill).
    private var deferredPLEJoin: (prefetch: PLERowPrefetch, layers: Int, readAhead: Bool, tokens: Int, joinedWait: Double?)?
    private let groupedExpertPrefill: Bool
    private let expertCommandBatch: Int
    private let moeSharedReadOverlap: Bool
    private let residualInjection: FusedResidualInjection?
    private let reductions: PinnedReductions?
    private let selector: BatchedTopIndices?
    private var active: UUID?
    private var reportedResidentWireFallbacks: UInt64 = 0
    private var executingScope = false
    private let expertExecution: ExpertExecution?
    private var cleanupFailure: Error?
    public var isHealthy: Bool { cleanupFailure == nil }
    /// False once packs are disabled or were never opened (external drive unavailable).
    public var packedSourceAvailable: Bool { weights.store.packedAvailable }
    /// The expert cache's counters, from any thread (status pages).
    public var expertCacheSnapshot: ExpertCache.Snapshot? { expertExecution?.cacheSnapshot }
    /// Split-read counters (`SplitReads`); any thread.
    package var splitReads: SplitReadCounters? { expertExecution?.splitReads }
    /// Coarse serving-thread counters; callers may take two request-boundary snapshots.
    public var prefillSchedulingSnapshot: ExpertExecution.PrefillSchedulingSnapshot? { expertExecution?.prefillSchedulingSnapshot }
    /// Full existing counters, including source bytes and slab-preparation time.
    /// Serving-thread/request boundaries only; never dashboard polling during inference.
    public var expertExecutionSnapshot: ExpertExecution.Snapshot? { expertExecution?.snapshot() }
    package var decodePreparationReceipt: ExpertExecution.DecodeSlabPreparation? { expertExecution?.decodeSlabPreparationReceipt }
    /// Decode time-split counters (whole schedule, pre-routing, per-drive read times); request boundaries only.
    package var decodeTiming:
        (
            counts: WholeExecutionSchedule.Counts, prerouting: DecodePreroutingCounts?,
            reads: [(count: UInt64, seconds: Double, buckets: [UInt64])]?
        )?
    {
        wholeSchedule.map { ($0.snapshot().decode, $0.prerouting?.counts, expertExecution?.readTimes.snapshot) }
    }
    /// The model's expert routing counts (the saved routing profile), set by the runtime.
    public var routingProfile: RoutingProfile? {
        get { expertExecution?.routing }
        set { expertExecution?.routing = newValue }
    }
    private let workspace: ReferencePhaseWorkspace
    private let residentBudget: UInt64
    private let diagnosticBudget: UInt64
    private let elasticDecodeResidency: Bool
    /// The conversation arena that lends its unreached pages to the expert cache (elastic tiers), set per request.
    private weak var lendingArena: QSAPageArena?
    private let decodeTokenPlan: Bool
    public var shouldContinue: () -> Bool = { true } {
        didSet { math.shouldContinue = shouldContinue }
    }
    public var progressObserver: ((Int, Int, Int) -> Void)?
    public init(
        store: PackStore, ledger: MemoryLedger, contextLimit: Int = 32_768, fileBackedContext: Bool? = nil,
        experts: ExpertExecutionConfiguration = .init(), additionalPhaseOwnerBytes: UInt64 = 0,
        simdMatrix: Bool = false, pinnedUnary: Bool = false,
        pinnedRMS: Bool = false, residentBytes: UInt64 = 0, residentSmallTensors: Bool = false,
        groupedExpertPrefill: Bool = false,
        pinnedReductions: Bool = false, batchedSelector: Bool = false, denseResidentBytes: UInt64 = 0, fusedExperts: Bool = false,
        pinnedAttention: Bool = false, registerQSADecode: Bool = false, batchedPinnedAttention: Bool = false,
        fusedSelectedPrefillAttention: Bool = false, compactPVGrouped: Bool = false, pinnedDense: Bool = false,
        expertCommandBatch: Int = 1, groupedExpertTiled: Bool = false, batchedRoPE: Bool = false, pinnedRoPE: Bool = false,
        pinnedIndexer: Bool = false, quantizedTiled: Bool = false, fusedHCEpilogue: Bool = false, fusedHCMinRows: Int = 8,
        fusedResidualInjection: Bool = false, fusedResidualMinRows: Int = 8, fusedGDNPreparation: Bool = false, fusedGDNMinRows: Int = 8,
        quantizedTiledBM32: Bool = false, quantizedBM32MinRows: Int = 32,
        groupedExpertBM32: Bool = false, groupedExpertBM32MinRows: Int = 128, expertScratchReuse: Bool = false,
        gdnResidentState: Bool = false, qsaDecodeReuse: Bool = false, fusedHCDecode: Bool = false, fusedHCPrefill: Bool = false,
        moeDevicePrefill: Bool = false, gdnInputComposition: Bool = false, gdnBufferRegion: Bool = false,
        gatheredExpertDecode: Bool = false, moeSharedReadOverlap: Bool = false, gdnDeviceBlock: Bool = false,
        wholeExecutionSchedule: Bool = false, raggedPrefillScope: Bool = false, decodeEarlyDemand: Bool = false,
        decodePreencodedLayers: Bool = false, decodeTokenPlan: Bool = false, elasticDecodeResidency: Bool = false,
        devicePLE devicePLEEnabled: Bool = false, hostStagePrefetch: Bool = false, quantizedTiledPrefetch: Bool = false,
        decodeRowPool: SpanReadPool? = nil, prefillDoubleBuffer: Bool = false, quantizedMMABF16: Bool = false,
        groupedExpertMMAMinRows: Int = 0, gdnExactKernels: Bool = false, decodeNextLayerPrerouting: Bool = false,
        decodePreroutingMargin: Double = 0.25, decodePreroutingOverrideWidth: Int = 0, qsaExactKernels: Bool = false,
        hcExactFusion: Bool = false, decodePreroutingReadHygiene: Bool = false, decodePLERacePool: SpanReadPool? = nil,
        hcDecodeFusion: Bool = false, prefillPLERacePool: SpanReadPool? = nil, fastOptInTau: Double? = nil
    ) throws {
        guard !quantizedMMABF16 || quantizedTiledPrefetch else {
            throw EngineError("quantized_mma_bf16 requires quantized_tiled_prefetch", code: "invalid_configuration")
        }
        guard groupedExpertMMAMinRows == 0 || ((64...256).contains(groupedExpertMMAMinRows) && quantizedMMABF16 && groupedExpertBM32) else {
            throw EngineError(
                "grouped_expert_mma_min_rows requires quantized_mma_bf16, grouped_expert_bm32 and a floor in 64...256",
                code: "invalid_configuration")
        }
        guard !gdnExactKernels || fusedGDNPreparation else {
            throw EngineError("gdn_exact_kernels requires fused_gdn_preparation", code: "invalid_configuration")
        }
        self.quantizedMMABF16 = quantizedMMABF16
        self.gdnExactKernels = gdnExactKernels
        guard !hcExactFusion || (fusedHCPrefill && simdMatrix) else {
            throw EngineError("hc_exact_fusion requires fused_hc_prefill and simd_matrix", code: "invalid_configuration")
        }
        self.hcExactFusion = hcExactFusion
        guard !hcDecodeFusion || fusedHCDecode else {
            throw EngineError("hc_decode_fusion requires fused_hc_decode", code: "invalid_configuration")
        }
        self.hcDecodeFusion = hcDecodeFusion
        let fileBacked = fileBackedContext ?? (contextLimit == ConversationGeometry.maximumFileContext)
        guard
            !registerQSADecode
                || (wholeExecutionSchedule && ConversationGeometry.inMemory(contextLimit, fileBacked: fileBacked) && pinnedAttention
                    && batchedPinnedAttention)
        else {
            throw EngineError("Register QSA requires explicit in-memory whole decode", code: "invalid_configuration")
        }
        self.registerQSADecode = registerQSADecode
        guard
            !raggedPrefillScope
                || (wholeExecutionSchedule && ConversationGeometry.inMemory(contextLimit, fileBacked: fileBacked)
                    && experts.prefillLayerCommand)
        else {
            throw EngineError("Mixed-tail scope requires whole layer-command execution", code: "invalid_configuration")
        }
        self.raggedPrefillScope = raggedPrefillScope
        guard !decodeTokenPlan || (decodeEarlyDemand && gdnResidentState && wholeExecutionSchedule) else {
            throw EngineError("Token plan requires whole decode, exact early demand and resident GDN", code: "invalid_configuration")
        }
        self.decodeTokenPlan = decodeTokenPlan
        guard !decodeNextLayerPrerouting || (decodeTokenPlan && decodeEarlyDemand && experts.mode != .legacyOriginal) else {
            throw EngineError("Decode pre-routing requires the whole token plan with exact early demand", code: "invalid_configuration")
        }
        self.decodeNextLayerPrerouting = decodeNextLayerPrerouting
        guard
            decodePreroutingOverrideWidth == 0
                || (decodeNextLayerPrerouting && experts.reuseAdmission
                    && (1...10).contains(decodePreroutingOverrideWidth))
        else {
            throw EngineError(
                "Pre-routing slot override requires decode pre-routing, reuse admission and a width in 1...10",
                code: "invalid_configuration")
        }
        self.decodePreroutingOverrideWidth = decodePreroutingOverrideWidth
        guard !decodePreroutingReadHygiene || decodeNextLayerPrerouting else {
            throw EngineError("Pre-routing read hygiene requires decode pre-routing", code: "invalid_configuration")
        }
        self.decodePreroutingReadHygiene = decodePreroutingReadHygiene
        guard fastOptInTau.map({ $0 > 0 && $0 <= 1 && wholeExecutionSchedule }) ?? true else {
            throw EngineError("Fast late-miss skip requires tau in (0, 1] and whole decode", code: "invalid_configuration")
        }
        self.fastOptInTau = fastOptInTau
        guard
            !decodeEarlyDemand
                || (wholeExecutionSchedule && ConversationGeometry.inMemory(contextLimit, fileBacked: fileBacked)
                    && experts.demandWindow >= 10)
        else {
            throw EngineError("Early demand requires whole decode with ten-record capacity", code: "invalid_configuration")
        }
        guard
            !experts.preallocateDecodeSlabs
                || (wholeExecutionSchedule && elasticDecodeResidency
                    && ConversationGeometry.inMemory(contextLimit, fileBacked: fileBacked))
        else {
            throw EngineError("Decode slab preallocation requires elastic whole in-memory execution", code: "invalid_configuration")
        }
        guard !elasticDecodeResidency || (wholeExecutionSchedule && ConversationGeometry.inMemory(contextLimit, fileBacked: fileBacked))
        else {
            throw EngineError("Elastic decode requires in-memory whole execution", code: "invalid_configuration")
        }
        self.elasticDecodeResidency = elasticDecodeResidency
        guard
            !wholeExecutionSchedule
                || (ConversationGeometry.contexts.contains(contextLimit) && gdnDeviceBlock
                    && gatheredExpertDecode && moeSharedReadOverlap && moeDevicePrefill && fusedHCDecode && fusedHCPrefill
                    && fusedResidualInjection && batchedSelector && residentSmallTensors && qsaDecodeReuse
                    && batchedPinnedAttention && pinnedRoPE && pinnedIndexer)
        else {
            throw EngineError(
                "Whole schedule requires declared pinned device dependencies and replaces the old frontier", code: "invalid_configuration")
        }
        guard !gdnDeviceBlock || (gdnInputComposition && gdnBufferRegion && fusedGDNPreparation && pinnedUnary && pinnedReductions) else {
            throw EngineError(
                "gdn_device_block requires composed GDN, fused preparation and pinned unary/reductions", code: "invalid_configuration")
        }
        guard !moeSharedReadOverlap || gatheredExpertDecode else {
            throw EngineError("moe_shared_read_overlap requires gathered_expert_decode", code: "invalid_configuration")
        }
        self.moeSharedReadOverlap = moeSharedReadOverlap
        guard !gatheredExpertDecode || (fusedExperts && pinnedReductions && experts.mode != .legacyOriginal && experts.demandWindow >= 10)
        else {
            throw EngineError(
                "gathered_expert_decode requires fused slab experts, pinned reductions and a window of at least ten",
                code: "invalid_configuration")
        }
        guard !gdnBufferRegion || (pinnedRMS && gdnResidentState) else {
            throw EngineError("gdn_buffer_region requires pinned_rms and gdn_resident_state", code: "invalid_configuration")
        }
        guard !qsaDecodeReuse || pinnedAttention else {
            throw EngineError("QSA decode reuse requires pinned attention", code: "invalid_configuration")
        }
        self.qsaDecodeReuse = qsaDecodeReuse
        guard
            !moeDevicePrefill
                || (fusedExperts && groupedExpertPrefill && pinnedReductions && expertCommandBatch > 1 && experts.mode != .legacyOriginal)
        else {
            throw EngineError(
                "moe_device_prefill requires fused grouped slab experts, pinned reductions and command batching",
                code: "invalid_configuration")
        }
        prefillMoETransfer = moeDevicePrefill ? try PrefillMoETransfer() : nil
        guard
            !(fusedHCDecode || fusedHCPrefill)
                || (pinnedReductions && pinnedDense && pinnedUnary && residentBytes > 0 && denseResidentBytes > 0)
        else {
            throw EngineError(
                "fused_hc_decode requires pinned_reductions, pinned_dense, pinned_unary and positive resident caches",
                code: "invalid_configuration")
        }
        guard !expertScratchReuse || fusedExperts else {
            throw EngineError("Expert scratch reuse requires fused experts", code: "invalid_configuration")
        }
        guard (1...256).contains(groupedExpertBM32MinRows), !groupedExpertBM32 || groupedExpertTiled else {
            throw EngineError("Grouped BM32 execution requires grouped tiling and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard (1...256).contains(quantizedBM32MinRows), !quantizedTiledBM32 || quantizedTiled else {
            throw EngineError("BM32 requires quantized_tiled and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard (1...256).contains(fusedGDNMinRows), !fusedGDNPreparation || pinnedReductions else {
            throw EngineError(
                "fused_gdn_preparation requires pinned_reductions and fused_gdn_min_rows in 1...256", code: "invalid_configuration")
        }
        guard (1...256).contains(fusedResidualMinRows) else {
            throw EngineError("fused_residual_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard (1...256).contains(fusedHCMinRows) else {
            throw EngineError("fused_hc_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard !quantizedTiled || simdMatrix else {
            throw EngineError("quantized_tiled requires simd_matrix", code: "invalid_configuration")
        }
        guard !fusedSelectedPrefillAttention || (batchedPinnedAttention && pinnedAttention) else {
            throw EngineError("Fused prefill requires batched pinned attention")
        }
        guard !batchedPinnedAttention || pinnedAttention else { throw EngineError("batched_pinned_attention requires pinned_attention") }
        guard (2...ModelContract.maximumContext).contains(contextLimit) else { throw EngineError("Invalid served context") }
        guard !fusedExperts || experts.mode != .legacyOriginal else { throw EngineError("Fused experts require pinned slab records") }
        guard (1...8).contains(expertCommandBatch), expertCommandBatch == 1 || (fusedExperts && experts.mode != .legacyOriginal) else {
            throw EngineError("Expert command batching requires one to eight jobs and fused slab execution", code: "invalid_configuration")
        }
        guard !groupedExpertTiled || (fusedExperts && simdMatrix && groupedExpertPrefill) else {
            throw EngineError("grouped_expert_tiled requires fused grouped SIMD experts", code: "invalid_configuration")
        }
        self.expertCommandBatch = expertCommandBatch
        self.residentBudget = residentBytes + denseResidentBytes
        // The caller has separately reserved these live owners. Include them when short appends grow the
        // expert cache; do not reserve them twice.
        let (phaseOwners, ownerOverflow) = additionalPhaseOwnerBytes.addingReportingOverflow(
            (decodeEarlyDemand ? DecodeEarlyDemand.metadataBytes : 0) + (decodeNextLayerPrerouting ? DecodePrerouting.ownerBytes : 0)
                + (decodeTokenPlan ? ReferenceWeights.tokenPlanOwnerBytes : 0)
                + (fusedSelectedPrefillAttention ? RuntimeConfiguration.fusedSelectedPrefillCounterOwnerBytes : 0))
        guard !ownerOverflow, phaseOwners <= ledger.usableBytes else {
            throw EngineError("Additional phase owners exceed admission", code: "insufficient_memory")
        }
        diagnosticBudget = phaseOwners
        self.ledger = ledger
        self.contextLimit = contextLimit
        self.fileBackedContext = fileBacked
        self.groupedExpertPrefill = groupedExpertPrefill
        ledger.checkAdmission()
        residualInjection = fusedResidualInjection ? try FusedResidualInjection(ledger: ledger, minimumRows: fusedResidualMinRows) : nil
        reductions = pinnedReductions ? try PinnedReductions(ledger: ledger) : nil
        selector = batchedSelector ? try BatchedTopIndices(ledger: ledger) : nil
        let kernels = try ReferenceKernels(
            pinnedUnary: pinnedUnary, ledger: ledger, pinnedRMS: pinnedRMS, quantizedTiled: quantizedTiled,
            quantizedTiledBM32: quantizedTiledBM32, quantizedBM32MinRows: quantizedBM32MinRows,
            quantizedTiledPrefetch: quantizedTiledPrefetch, quantizedMMABF16: quantizedMMABF16, gdnExactKernels: gdnExactKernels,
            hcExactFusion: hcExactFusion)
        weights = try ReferenceWeights(
            store: store, kernels: kernels, ledger: ledger, simdMatrix: simdMatrix, residentBytes: residentBytes,
            residentSmallTensors: residentSmallTensors, denseResidentBytes: denseResidentBytes, fusedExperts: fusedExperts,
            pinnedDense: pinnedDense, groupedExpertTiled: groupedExpertTiled, groupedExpertBM32: groupedExpertBM32,
            groupedExpertBM32MinRows: groupedExpertBM32MinRows, expertScratchReuse: expertScratchReuse, fusedHCDecode: fusedHCDecode,
            fusedHCPrefill: fusedHCPrefill, gdnInputComposition: gdnInputComposition, groupedExpertMMAMinRows: groupedExpertMMAMinRows,
            hcDecodeFusion: hcDecodeFusion, wireResident: experts.wireDecodeSlabs)
        gatheredExperts = gatheredExpertDecode ? try weights.makeGatheredExpertDecode(reductions: reductions!) : nil
        if experts.mode == .legacyOriginal {
            expertExecution = nil
        } else {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal expert device unavailable") }
            let execution = try ExpertExecution(
                store: store, device: device, ledger: ledger, configuration: experts, earlyDemand: decodeEarlyDemand,
                prerouting: decodeNextLayerPrerouting,
                readHygiene: decodePreroutingReadHygiene,
                fastSkipReserve: fastOptInTau == nil ? 0 : FastLateMissSkip.readReserve)
            expertExecution = execution
        }
        let source = weights
        let gdnHelper: FusedGDNPreparation?
        if fusedGDNPreparation, let reductions {
            gdnHelper = try FusedGDNPreparation(
                arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger, minimumRows: fusedGDNMinRows,
                exactKernels: gdnExactKernels)
        } else {
            gdnHelper = nil
        }
        gdnPreparationHelper = gdnHelper
        self.gdnDeviceBlock = gdnDeviceBlock ? try source.makeGDNDeviceBlock(preparation: gdnHelper!) : nil
        math = LayerMath(
            kernels: kernels, linear: { x, name, bf16 in try source.linear(x, name: name, outputBF16: bf16) },
            tensor: { try source.tensor($0) },
            reductions: reductions,
            selector: selector,
            pinnedAttention: pinnedAttention
                ? try {
                    let attention = try PinnedAttention(
                        ledger: ledger, architecture: .medium, batched: batchedPinnedAttention,
                        registerDecode: registerQSADecode, fusedPrefill: fusedSelectedPrefillAttention)
                    attention.compactPVGrouped = compactPVGrouped
                    attention.qsaExactKernels = qsaExactKernels
                    if qsaExactKernels { try attention.prepareExactKernels() }
                    return attention
                }() : nil, batchedRoPE: batchedRoPE, pinnedRoPE: pinnedRoPE ? try PinnedRoPE(ledger: ledger) : nil,
            pinnedIndexer: pinnedIndexer ? try PinnedIndexer(ledger: ledger) : nil,
            hyperConnectionEpilogue: fusedHCEpilogue
                ? try FusedHyperConnection(arithmetic: kernels.arithmetic, ledger: ledger, minimumRows: fusedHCMinRows) : nil,
            gdnPreparation: gdnHelper,
            gdnBufferRegion: gdnBufferRegion ? try GDNBufferRegion(kernels: kernels, ledger: ledger, gate: gdnHelper) : nil)
        if fusedHCDecode || fusedHCPrefill {
            math.hyperConnectionDecode = { residual, base, combine, capture, keepGoing in
                try source.hyperConnectionDecode(residual, base: base, useCombine: combine, capture: capture, shouldContinue: keepGoing)
            }
        }
        if gdnInputComposition {
            math.gdnInputProjections = { input, base, keepGoing in
                try source.gdnInputProjections(input, base: base, shouldContinue: keepGoing)
            }
        }
        if let block = self.gdnDeviceBlock {
            math.gdnDeviceBlock = { input, base, state, capture, keepGoing in
                try source.gdnDeviceBlock(input, base: base, state: state, block: block, capture: capture, shouldContinue: keepGoing)
            }
        }
        wholeSchedule =
            wholeExecutionSchedule
            ? try WholeExecutionSchedule(
                weights: weights, math: math, ledger: ledger,
                context: contextLimit, gdn: self.gdnDeviceBlock!, gathered: gatheredExperts!, selector: selector!, reductions: reductions!,
                injection: residualInjection!, decodeEarlyDemand: decodeEarlyDemand, decodeTokenPlan: decodeTokenPlan,
                decodePrerouting: decodeNextLayerPrerouting, preroutingMargin: decodePreroutingMargin,
                preroutingOverrideWidth: decodePreroutingOverrideWidth, preroutingReadHygiene: decodePreroutingReadHygiene,
                preencodedLayers: decodePreencodedLayers) : nil
        ple = try ReferencePLE(weights: weights, ledger: ledger, reductions: reductions)
        if devicePLEEnabled {
            guard wholeExecutionSchedule, let reductions, residentBytes > 0 else {
                throw EngineError(
                    "Device PLE requires whole execution, pinned reductions and resident tensors", code: "invalid_configuration")
            }
            let base = "model.layers.1.ple"
            devicePLE = try DevicePLE(
                kernels: weights.kernels, reductions: reductions, simdMatrix: weights.simdMatrix,
                arithmetic: weights.kernels.arithmetic, normKey: weights.tensor(base + ".norm_key.weight"),
                normQuery: weights.tensor(base + ".norm_query.weight"), normConv: weights.tensor(base + ".norm_conv.weight"),
                convolution: weights.tensor(base + ".conv1d.weight"), ledger: ledger)
        } else {
            devicePLE = nil
        }
        guard !hostStagePrefetch || (wholeExecutionSchedule && residentBytes > 0 && denseResidentBytes > 0) else {
            throw EngineError("Host-stage prefetch requires whole execution and both resident caches", code: "invalid_configuration")
        }
        self.hostStagePrefetch = hostStagePrefetch
        guard !prefillDoubleBuffer || wholeExecutionSchedule else {
            throw EngineError("Double-buffered prefill tiles require whole execution", code: "invalid_configuration")
        }
        self.prefillDoubleBuffer = prefillDoubleBuffer
        self.qsaExactKernels = qsaExactKernels
        self.decodeRowPool = decodeRowPool
        guard decodePLERacePool == nil || (decodeRowPool != nil && devicePLEEnabled) else {
            throw EngineError("decode_ple_internal_race requires device_ple and io_reader_pool", code: "invalid_configuration")
        }
        self.decodePLERacePool = decodePLERacePool
        guard prefillPLERacePool == nil || devicePLEEnabled else {
            throw EngineError("prefill_ple_overlap requires device_ple", code: "invalid_configuration")
        }
        self.prefillPLERacePool = prefillPLERacePool
        workspace = try ReferencePhaseWorkspace(ledger: ledger)
    }

    public func begin(inputTokens: Int, outputTokens: Int) throws -> SequenceState {
        try begin(inputTokens: inputTokens, outputTokens: outputTokens, pageArena: nil)
    }
    package func begin(inputTokens: Int, outputTokens: Int, pageArena: QSAPageArena?, upcomingInput: Int? = nil) throws -> SequenceState {
        guard active == nil else { throw EngineError("Only one active sequence is supported") }
        guard !fileBackedContext || wholeSchedule == nil || pageArena?.fileStore != nil else {
            throw EngineError("262K whole execution requires file-backed QSA state", code: "invalid_configuration")
        }
        if let cleanupFailure { throw EngineError("Previous expert cleanup failed: \(cleanupFailure); restart the server") }
        try weights.trimExpertScratch()
        try gatheredExperts?.trim()
        try wholeSchedule?.trim()
        let cacheBytes = try prepareUpcomingInput(upcomingInput, arena: pageArena, growth: inputTokens + outputTokens)
        ledger.checkAdmission()
        let state = try SequenceState(
            inputTokens: inputTokens, outputTokens: outputTokens, ledger: ledger, contextLimit: contextLimit, pageArena: pageArena)
        try expertExecution?.beginSequence(prefillCacheBytes: cacheBytes)
        active = state.id
        return state
    }

    /// A complete persistent frontier can survive transport retirement. All
    /// scratch/mirrors are joined and retired; only immutable model state lives.
    /// `growth`: the tokens this request can still add (the rest of the prompt and its output budget).
    package func resume(_ state: SequenceState, upcomingInput: Int? = nil, growth: Int? = nil) throws {
        guard active == nil else { throw EngineError("Cannot resume while another sequence is active") }
        if let cleanupFailure { throw EngineError("Previous expert cleanup failed: \(cleanupFailure); restart the server") }
        try state.validate()
        try weights.trimExpertScratch()
        try gatheredExperts?.trim()
        try wholeSchedule?.trim()
        let cacheBytes = try prepareUpcomingInput(upcomingInput, arena: state.pageArena, growth: growth)
        ledger.checkAdmission()
        try expertExecution?.beginSequence(prefillCacheBytes: cacheBytes)
        active = state.id
    }
    /// Persistent conversation owners supply this prospective suffix on both KV storage routes.
    /// Arithmetic remains prefill even when we retain the hot decode capacity.
    /// Larger scopes stream layer-local expert unions with the cache their real rows leave (`promptCacheBytes`).
    /// KV loan (elastic tiers): the arena holds only what this request's pages can reach, and the expert cache may
    /// use the rest. The cache is sized (and shrunk, quiescently) first, then the arena takes its charge back, so the
    /// conversation always gets its pages and nothing is resident twice.
    private func prepareUpcomingInput(_ count: Int?, arena: QSAPageArena? = nil, growth: Int? = nil) throws -> UInt64? {
        var bytes: UInt64?
        lendingArena = elasticDecodeResidency && growth != nil ? arena : nil
        let held = try lendingArena.map {
            try ConversationGeometry(context: contextLimit, fileBacked: fileBackedContext)
                .heldArenaBytes(live: $0.snapshot().liveBytes, growth: growth!)
        }
        if count != nil || (wholeSchedule != nil && expertExecution != nil && ConversationGeometry.contexts.contains(contextLimit)) {
            let count = count ?? 8192
            guard count >= 0, count <= contextLimit, wholeSchedule != nil,
                let expertExecution
            else {
                throw EngineError("Prospective conversation phase requires the complete whole schedule")
            }
            bytes = try Self.promptCacheBytes(
                context: contextLimit, fileBacked: fileBackedContext, input: count, residentBytes: residentBudget,
                experts: expertExecution.configuration, usable: ledger.usableBytes, additionalOwners: diagnosticBudget,
                heldArenaBytes: held)
        }
        try autoreleasepool { try expertExecution?.preparePrefillPhase(cacheBytes: bytes) }
        // A slab still retained by a GPU owner keeps its charge a moment longer: pages past the hold then take the room
        // as it frees (QSAPageArena.acquire), so a short ledger here does not refuse the request. Decode then counts the
        // whole arena: sized by the short hold, its cache would take back the room this request's pages still need.
        if let held {
            do { try lendingArena!.hold(held) } catch {
                lendingArena = nil
                FileHandle.standardError.write(
                    Data("origami: warning: the expert cache has not returned the key/value room yet: \(error)\n".utf8))
            }
        }
        if count == 0 { try workspace.retirePrefill() } else { try workspace.requirePrefill() }
        return bytes
    }

    /// The expert cache while a prompt of `input` new rows (after any restored prefix) is read. Fewer than 256 keep the
    /// hot decode capacity. Longer ones are charged their real rows in whole 256-row tiles, up to one 8,192-row scope, and
    /// keep the plan's prefill cache plus what their scope leaves of the 8,192-row one; all under the current budget.
    package static func promptCacheBytes(
        context: Int, fileBacked: Bool?, input: Int, residentBytes: UInt64, experts: ExpertExecutionConfiguration,
        usable: UInt64, additionalOwners: UInt64 = 0, heldArenaBytes: UInt64? = nil
    ) throws -> UInt64 {
        let rows = input < 256 ? input : min(8192, (input + 255) / 256 * 256)
        var ceiling: UInt64?
        if input >= 256 {
            let geometry = try ConversationGeometry(context: context, fileBacked: fileBacked)
            let longest = try geometry.maximumFixedBytes(input: 8192, residentBytes: 0)
            let this = try geometry.maximumFixedBytes(input: rows, residentBytes: 0)
            let slab = UInt64(experts.slotsPerSlab * PackStore.expertBytes)
            ceiling = min(experts.cacheBytes, experts.prefillCacheBytes + (longest > this ? longest - this : 0) / slab * slab)
        }
        return try ConversationPhaseBudget.expertBytes(
            context: context, fileBacked: fileBacked, input: rows, residentBytes: residentBytes, experts: experts,
            usable: usable, additionalOwners: additionalOwners, heldArenaBytes: heldArenaBytes, ceiling: ceiling)
    }

    package func suspend(_ state: SequenceState) throws {
        guard active == state.id, !executingScope else { throw EngineError("Persistent frontier still executing") }
        try state.validate()
        retire(state, invalidate: false)
        if let cleanupFailure {
            state.valid = false
            throw cleanupFailure
        }
    }

    /// Only the committed complete prompt (including a restored prefix) may
    /// transfer borrowed expert capacity back to decode. All preceding public
    /// scopes have returned through their joined autorelease drain.
    public func prepareDecode(
        _ state: SequenceState, consumedPromptTokens: Int,
        retainedFrontierBytes: UInt64? = nil
    ) throws {
        guard active == state.id, !executingScope, state.valid, consumedPromptTokens > 0,
            state.tokenCount == consumedPromptTokens, state.emittedButUnconsumed == nil
        else {
            throw EngineError("Decode phase requires the committed prompt frontier", code: "expert_phase_state")
        }
        try boundary()
        weights.recoverCommandBudget()
        _ = try autoreleasepool { () throws -> Bool in
            try wholeSchedule?.prepareDecode()
            if decodeTokenPlan { try weights.prepareWholeTokenPlan(boundary: { try self.boundary() }) }
            return true
        }
        if qsaDecodeReuse { try math.beginDecodeStorage(requestID: state.id, capacity: state.capacity) }
        if wholeSchedule != nil {
            try workspace.retirePrefill()
        }
        if elasticDecodeResidency, let expertExecution {
            func cacheBytes(_ budget: UInt64) throws -> UInt64 {
                try ConversationPhaseBudget.expertBytes(
                    context: contextLimit, fileBacked: fileBackedContext, input: 0,
                    residentBytes: residentBudget, experts: expertExecution.configuration,
                    usable: budget, additionalOwners: diagnosticBudget,
                    retainedFrontierBytes: retainedFrontierBytes, heldArenaBytes: lendingArena?.heldBytes)
            }
            try expertExecution.beginDecodePhase(
                cacheBytes: cacheBytes(ledger.usableBytes),
                recoveryBytes: cacheBytes(ledger.maximumUsableBytes))
        } else {
            try expertExecution?.beginDecodePhase()
        }
        let wireFallbacks = expertExecution?.decodeSlabPreparationReceipt?.wireFallbacks ?? 0
        try autoreleasepool { try expertExecution?.prepareDecodeSlabs(shouldContinue: shouldContinue) }
        if let total = expertExecution?.decodeSlabPreparationReceipt?.wireFallbacks, total > wireFallbacks {
            FileHandle.standardError.write(
                Data(
                    "origami: warning: \(total - wireFallbacks) decode slabs fell back from wiring (mlock) to page touch; expect page-fault stalls\n"
                        .utf8))
        }
        // Resident weights wire at admission (prefill); their failures are reported here, next to the slabs'.
        let residentFallbacks = (weights.residentSnapshot?.wireFallbacks ?? 0) + (weights.denseResidentSnapshot?.wireFallbacks ?? 0)
        if residentFallbacks > reportedResidentWireFallbacks {
            FileHandle.standardError.write(
                Data(
                    "origami: warning: \(residentFallbacks - reportedResidentWireFallbacks) resident weight buffers could not be wired (mlock); memory pressure can compress them\n"
                        .utf8))
            reportedResidentWireFallbacks = residentFallbacks
        }
    }

    /// Return borrowed frontier bytes before a new image or draft can grow. A bounded
    /// in-memory anchor supplies its prospective total; other captures reclaim the whole allowance.
    /// resizeQuiescent refuses outstanding I/O or escaped expert pins;
    /// callers must propagate that failure before allocating a checkpoint.
    package func reclaimFrontierLoan(retainedFrontierBytes: UInt64? = nil) throws {
        guard elasticDecodeResidency else { return }
        guard active != nil, !executingScope, let expertExecution, expertExecution.isDecodePhase else {
            throw EngineError("Frontier loan reclamation requires joined decode", code: "expert_phase_state")
        }
        func cacheBytes(_ budget: UInt64) throws -> UInt64 {
            try ConversationPhaseBudget.expertBytes(
                context: contextLimit, fileBacked: fileBackedContext, input: 0,
                residentBytes: residentBudget, experts: expertExecution.configuration,
                usable: budget, additionalOwners: diagnosticBudget,
                retainedFrontierBytes: retainedFrontierBytes, heldArenaBytes: lendingArena?.heldBytes)
        }
        try expertExecution.beginDecodePhase(
            cacheBytes: cacheBytes(ledger.usableBytes),
            recoveryBytes: cacheBytes(ledger.maximumUsableBytes))
    }

    public func finish(_ state: SequenceState) { retire(state, invalidate: true) }

    private func retire(_ state: SequenceState, invalidate: Bool) {
        if active == state.id {
            math.finishDecodeStorage(requestID: state.id)
            do { try expertExecution?.finishSequence() } catch { cleanupFailure = error }
            do { try weights.trimExpertScratch() } catch { cleanupFailure = error }
            do { try wholeSchedule?.trim() } catch { cleanupFailure = error }
            weights.retireWholeTokenPlan()
            // Nothing of this sequence reaches the next one: MTP's hidden rows, and the next-scope lookahead of a
            // failed prompt read (dropping a prefetch cancels its readers, joins them and releases its rows).
            lastHidden = []
            nextPrefillScope = nil
            pendingPLEPrefetch = nil
            pendingEmbeddingPrefetch = nil
            pairResidencyBackoff = .init()
            active = nil
            if invalidate { state.valid = false }
        }
    }

    /// Memory governor hook: shrink the expert cache to the ledger's current budget. Any thread.
    @discardableResult
    public func trimToMemoryBudget() -> UInt64 { (try? expertExecution?.trimToBudget()) ?? 0 }
    /// Memory governor hook: the budget grew back; prepare the trimmed decode slabs off the token loop. Any thread.
    public func growToMemoryBudget() { expertExecution?.prepareGrowth() }
    /// Token boundary (single steps and MTP pairs): the previous token's commands are joined and growth is not held, so
    /// prepared slabs attach here.
    private func attachGrowth() {
        expertExecution?.completePendingTrim()
        if expertExecution?.isDecodePhase == true, let added = expertExecution?.attachGrowth(), added > 0 {
            pairResidencyBackoff = .init()
            FileHandle.standardError.write(Data("origami: expert cache grew back by \(added) slabs mid-reply\n".utf8))
        }
    }
    /// M-RoPE positions for a prompt with images; nil for text.
    public func setPositionMap(_ map: RoPEPositionMap?) throws { try math.setPositionMap(map) }

    private func boundary() throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        ledger.checkAdmission()
    }

    /// A single arithmetic tile. Larger read scopes must use forwardScope so
    /// projection shapes and the enclosing MoE dispatch remain explicit.
    public func forward(_ ids: [Int], state: SequenceState, maximumLayers: Int = 48, returnLogits: Bool = true) throws -> [Float] {
        guard !ids.isEmpty, ids.count <= 256 else { throw EngineError("Invalid reference tile") }
        return try forwardScope(ids, computeTile: ids.count, state: state, maximumLayers: maximumLayers, returnLogits: returnLogits)
    }

    /// Layer-major expert reads, chronological attention tiles. Each scope has
    /// uniform complete arithmetic tiles; a short terminal tile is its own scope.
    /// State becomes reusable only after all layers commit this scope.
    public func forwardScope(
        _ ids: [Int], computeTile: Int, state: SequenceState,
        maximumLayers: Int = 48, returnLogits: Bool = true
    ) throws -> [Float] {
        // Includes short prefill tails and final readout bridges. Escaped
        // persistent storage retains its own charge; all GPU/I/O work joins
        // inside the implementation before temporary references drain here.
        try autoreleasepool {
            try forwardScopeImpl(
                ids, computeTile: computeTile, state: state,
                maximumLayers: maximumLayers, returnLogits: returnLogits, conversationCapture: nil)
        }
    }

    package func forwardConversationScope(
        _ ids: [Int], state: SequenceState, returnLogits: Bool,
        capture: ConversationScopeCapture
    ) throws -> [Float] {
        try autoreleasepool {
            try forwardScopeImpl(
                ids, computeTile: 256, state: state, maximumLayers: 48,
                returnLogits: returnLogits, conversationCapture: capture)
        }
    }

    private func forwardScopeImpl(
        _ ids: [Int], computeTile: Int, state: SequenceState,
        maximumLayers: Int, returnLogits: Bool, conversationCapture: ConversationScopeCapture?
    ) throws -> [Float] {
        let mixedLayout =
            raggedPrefillScope && computeTile == 256 && ids.count % 256 != 0
            ? try WholePrefillScope.layout(tokens: ids.count, allowTail: true) : nil
        guard active == state.id, !executingScope, !ids.isEmpty, ids.count <= 8192,
            (1...256).contains(computeTile), ids.count % computeTile == 0 || mixedLayout != nil,
            ids.count <= state.capacity - state.tokenCount, (1...48).contains(maximumLayers)
        else {
            throw EngineError("Invalid reference read scope")
        }
        try state.validate()
        if let pending = state.emittedButUnconsumed {
            guard ids == [pending] else { throw EngineError("The pending generated token must be consumed exactly once") }
        }
        let wholeDecode =
            wholeSchedule != nil && expertExecution?.isDecodePhase == true
            && ids.count == 1 && computeTile == 1
        guard
            conversationCapture == nil
                || (wholeSchedule != nil && !wholeDecode && maximumLayers == 48 && computeTile == 256 && state.pageArena != nil)
        else {
            throw EngineError("Intra-scope checkpoints require the actual complete whole-prefill schedule")
        }
        if !wholeDecode { try workspace.requirePrefill() }
        attachGrowth()
        executingScope = true
        defer { executingScope = false }
        if let wholeSchedule {
            if let expertExecution, expertExecution.isDecodePhase, ids.count == 1, computeTile == 1 {
                return try forwardWholeDecode(
                    ids[0], state: state, maximumLayers: maximumLayers,
                    returnLogits: returnLogits, schedule: wholeSchedule, execution: expertExecution)
            }
            // Preserve the caller's segmentation: a standalone short suffix uses its actual row count,
            // while C256 scopes retain their existing full tiles and optional mixed tail. No padding or merging.
            if computeTile == 256 || ids.count == computeTile,
                let layout = try WholePrefillScope.layout(
                    tokens: ids.count,
                    allowTail: raggedPrefillScope || ids.count == computeTile, admitSmallScope: true),
                layout.supplementalBytes == 0 || (math.capture == nil && !ple.hasCapture)
            {
                return try forwardWholePrefill(
                    ids, state: state, maximumLayers: maximumLayers,
                    returnLogits: returnLogits, schedule: wholeSchedule, layout: layout, capture: conversationCapture,
                    computeTile: computeTile)
            }
        }
        guard mixedLayout == nil else {
            throw EngineError("Mixed-tail arithmetic requires its whole-scope path")
        }
        // Existing 2 GB tile bound plus the retained scope frontiers. The
        // additional charge includes old/new residuals, mixed inputs, and ten
        // canonical expert contributions. Reserve before any state mutation.
        let originalExtraBytes = UInt64(max(0, ids.count - 256)) * 240_000
        let useDeviceMoE = prefillMoETransfer != nil && computeTile == 256 && ids.count >= 512
        let deviceBytes = useDeviceMoE ? try PrefillMoEBudget.bytesForScope(tokens: ids.count) : 0
        guard deviceBytes <= originalExtraBytes else { throw EngineError("Device MoE cannot fit the unchanged scope frontier bound") }
        // The fixed2GB reservation is untouched. Atomically monotone admission
        // of these two independent owners totals exactly the old frontier bound.
        // No active memory is uncharged/recharged to make a device buffer fit.
        let extra = try ledger.reserve(bytes: originalExtraBytes - deviceBytes, category: "read_scope_frontier_bound")
        defer { try? ledger.release(extra) }
        let deviceBudget = useDeviceMoE ? try PrefillMoEBudget(ledger: ledger, bytes: deviceBytes) : nil
        do {
            try boundary()
            try weights.enterLayer(nil)
            var embeddings = try weights.embedding(ids)
            try state.images?.inject(into: &embeddings, start: state.tokenCount)
            var residual: [Float] = []
            residual.reserveCapacity(ids.count * 10_240)
            for token in ids.indices {
                let row = embeddings[(token * 2560)..<((token + 1) * 2560)]
                for _ in 0..<4 { residual.append(contentsOf: row) }
            }
            let history = state.history + ids.map(Int64.init)
            let historyBase = state.history.count
            for layer in 0..<maximumLayers {
                try boundary()
                try weights.enterLayer(layer)
                residual = try autoreleasepool {
                    var bases: [Float] = []
                    var inputs: [Float] = []
                    var injections: [Float] = []
                    bases.reserveCapacity(residual.count)
                    inputs.reserveCapacity(ids.count * 2560)
                    injections.reserveCapacity(ids.count * 4)
                    for lo in stride(from: 0, to: ids.count, by: computeTile) {
                        try boundary()
                        let hi = lo + computeTile
                        try autoreleasepool {
                            var hidden = Array(residual[(lo * 10_240)..<(hi * 10_240)])
                            if layer == 1 {
                                let begin = max(0, historyBase + lo - 2)
                                let extra = try ple.call(
                                    hidden, history: Array(history[begin..<(historyBase + hi)]),
                                    newCount: computeTile, state: &state.ple)
                                hidden = zip(hidden, extra).map { LayerArithmetic.round($0 + $1) }
                            }
                            let base = "model.layers.\(layer)"
                            let first = try math.hyperConnection(residual: hidden, base: base + ".attn_hyper_connection")
                            let attention: [Float]
                            if layer % 4 == 3 {
                                guard var qsa = state.qsa.removeValue(forKey: layer) else { throw EngineError("Missing QSA state") }
                                attention = try math.qsa(x: first.mixed, layer: layer, state: &qsa, requestID: state.id)
                                state.qsa[layer] = qsa
                            } else {
                                guard var gdn = state.gdn.removeValue(forKey: layer) else { throw EngineError("Missing GDN state") }
                                attention = try math.gdn(x: first.mixed, layer: layer, state: &gdn)
                                state.gdn[layer] = gdn
                            }
                            hidden = try inject(attention, into: hidden, gates: first.injection)
                            let second = try math.hyperConnection(residual: hidden, base: base + ".mlp_hyper_connection")
                            bases.append(contentsOf: hidden)
                            inputs.append(contentsOf: second.mixed)
                            injections.append(contentsOf: second.injection)
                            // A scope too small for the whole-prefill layout lands here; its staged frontiers
                            // take each tile's attention state as the whole path does.
                            if let conversationCapture {
                                try conversationCapture.retain(layer: layer, offset: state.tokenCount + hi, state: state)
                            }
                        }
                    }
                    let output = try moe(inputs, layer: layer, computeTile: computeTile, deviceBudget: deviceBudget)
                    return try inject(output, into: bases, gates: injections)
                }
                progressObserver?(state.tokenCount, ids.count, layer + 1)
            }
            guard maximumLayers == 48 else {
                state.valid = false
                return residual
            }
            if captureHiddenRows > 0 { lastHidden = Self.lastRows(residual, count: ids.count, keep: captureHiddenRows) }
            var logits: [Float] = []
            if returnLogits {
                do {
                    try weights.enterLayer(nil)
                    for lo in stride(from: 0, to: ids.count, by: computeTile) {
                        let mixed = try math.hyperConnection(
                            residual: Array(residual[(lo * 10_240)..<((lo + computeTile) * 10_240)]),
                            base: "model.hyper_connection_mixer", useCombine: false
                        ).mixed
                        if lo + computeTile == ids.count { logits = try weights.linear(Array(mixed.suffix(2560)), name: "lm_head") }
                    }
                }
            }
            state.history = Array(history.suffix(2))
            state.tokenCount += ids.count
            state.emittedButUnconsumed = nil
            try state.validate()
            try boundary()
            return logits
        } catch {
            state.valid = false
            throw error
        }
    }

    private func originalDecodeMoE(_ values: (base: [Float], input: [Float], gates: [Float]), layer: Int) throws -> [Float] {
        let output = try moe(values.input, layer: layer, computeTile: 1)
        return try inject(output, into: values.base, gates: values.gates)
    }

    private func inject(_ output: [Float], into residual: [Float], gates: [Float]) throws -> [Float] {
        if let residualInjection, residualInjection.usesGPU(rows: residual.count / 10240) {
            return try residualInjection.evaluate(output: output, residual: residual, gates: gates, shouldContinue: shouldContinue)
        }
        return try FusedResidualInjection.reference(output: output, residual: residual, gates: gates)
    }

    /// Batch only command submission. Per-expert row order, 256-row tiles, and
    /// canonical (token, router-rank) scatter destinations remain unchanged.
    private func batchedExperts(
        _ x: [Float], selectedRows: [Int: [(Int, Int)]],
        window: ExpertDemandWindow, grouped: Bool,
        contributions: inout [Float], deviceWorkspace: PrefillMoEWorkspace? = nil
    ) throws {
        let sorted = selectedRows.keys.sorted()
        let recordsPerBatch = min(expertCommandBatch, window.maximumBatchCount)
        for start in stride(from: 0, to: sorted.count, by: recordsPerBatch) {
            try boundary()
            let experts = Array(sorted[start..<min(start + recordsPerBatch, sorted.count)])
            try window.consumeBatch(experts: experts) { buffers in
                var jobs: [FusedExpertExecution.Job] = []
                var destinations: [(expert: Int, lower: Int, upper: Int)] = []
                var deviceJobs: [FusedExpertExecution.DeviceJob] = []
                jobs.reserveCapacity(expertCommandBatch)
                destinations.reserveCapacity(expertCommandBatch)
                deviceJobs.reserveCapacity(expertCommandBatch)
                func flush() throws {
                    if let deviceWorkspace {
                        guard !deviceJobs.isEmpty else { return }
                        try weights.expertDeviceBatch(deviceJobs, workspace: deviceWorkspace, shouldContinue: self.shouldContinue)
                        deviceJobs.removeAll(keepingCapacity: true)
                        return
                    }
                    guard !jobs.isEmpty else { return }
                    try weights.expertBatch(jobs, shouldContinue: self.shouldContinue) { index, output in
                        let destination = destinations[index]
                        let positions = selectedRows[destination.expert]!
                        guard output.count == (destination.upper - destination.lower) * 2560 else {
                            throw EngineError("Expert command batch returned an invalid tile")
                        }
                        for row in destination.lower..<destination.upper {
                            let position = positions[row]
                            let source = (row - destination.lower) * 2560
                            let target = (position.0 * 10 + position.1) * 2560
                            contributions.replaceSubrange(target..<(target + 2560), with: output[source..<(source + 2560)])
                        }
                    }
                    jobs.removeAll(keepingCapacity: true)
                    destinations.removeAll(keepingCapacity: true)
                }
                for (index, expert) in experts.enumerated() {
                    let positions = selectedRows[expert]!
                    for lower in stride(from: 0, to: positions.count, by: 256) {
                        try boundary()
                        let upper = min(lower + 256, positions.count)
                        // At most eight 256-row host input tiles are live; these
                        // and the one delivered output tile fit the existing
                        // phase workspace bound. Metal allocations are charged
                        // separately by the joined command implementation.
                        if deviceWorkspace != nil {
                            let slots = positions[lower..<upper].map { UInt32($0.0 * 10 + $0.1) }
                            deviceJobs.append(
                                FusedExpertExecution.DeviceJob(
                                    expert: expert, slots: slots, buffer: buffers[index],
                                    groupedExpert: grouped, simdMatrix: weights.simdMatrix))
                            if deviceJobs.count == expertCommandBatch { try flush() }
                        } else {
                            let input = positions[lower..<upper].flatMap { Array(x[($0.0 * 2560)..<(($0.0 + 1) * 2560)]) }
                            jobs.append(
                                FusedExpertExecution.Job(
                                    input: input, buffer: buffers[index],
                                    groupedExpert: grouped, simdMatrix: weights.simdMatrix))
                            destinations.append((expert, lower, upper))
                            if jobs.count == expertCommandBatch { try flush() }
                        }
                    }
                }
                try flush()
            }
        }
    }

    private func moe(_ x: [Float], layer: Int, computeTile: Int, deviceBudget: PrefillMoEBudget? = nil) throws -> [Float] {
        let tokens = x.count / 2560
        let base = "model.layers.\(layer).mlp"
        // Routing and shared projections preserve the original compute shapes.
        var logits: [Float] = []
        for lo in stride(from: 0, to: tokens, by: computeTile) {
            logits.append(
                contentsOf: try weights.linear(
                    Array(x[(lo * 2560)..<((lo + computeTile) * 2560)]),
                    name: base + ".gate", outputBF16: false))
        }
        BatchedTopIndices.maskColumns(&logits, columns: 512, eligible: ModelLayout.routedExperts)
        var selectedBatch: [[UInt32]] = []
        if let selector {
            for lo in stride(from: 0, to: tokens, by: computeTile) {
                selectedBatch.append(
                    contentsOf: try selector.topIndices(
                        scores: Array(logits[(lo * 512)..<((lo + computeTile) * 512)]), columns: 512, count: 10))
            }
        }
        var ids: [[Int]] = []
        var scores: [[Float]] = []
        var selectedLogits: [Float] = []
        selectedLogits.reserveCapacity(tokens * 10)
        var selectedRows: [Int: [(Int, Int)]] = [:]
        for row in 0..<tokens {
            let values = Array(logits[(row * 512)..<((row + 1) * 512)])
            let selected = try (selector == nil ? weights.kernels.topIndices(scores: values, count: 10) : selectedBatch[row]).map(Int.init)
            selectedLogits.append(contentsOf: selected.map { values[$0] })
            let maximum = selected.map { values[$0] }.max()!
            let exponentials = selected.map { expf(values[$0] - maximum) }
            let denominator = exponentials.reduce(Float(0), +)
            guard denominator.isFinite, denominator > 0 else { throw EngineError("Nonfinite router probabilities") }
            ids.append(selected)
            scores.append(exponentials.map { $0 / denominator })
            for (rank, expert) in selected.enumerated() { selectedRows[expert, default: []].append((row, rank)) }
        }
        if let reductions {
            let pinned = try reductions.softmax(rows: selectedLogits, width: 10)
            guard pinned.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw EngineError("Nonfinite pinned router probabilities") }
            scores = (0..<tokens).map { Array(pinned[($0 * 10)..<(($0 + 1) * 10)]) }
        }
        let deviceWorkspace: PrefillMoEWorkspace?
        if let deviceBudget, let prefillMoETransfer {
            deviceWorkspace = try PrefillMoEWorkspace(
                input: x, selectedRows: selectedRows, budget: deviceBudget, transfer: prefillMoETransfer)
        } else {
            deviceWorkspace = nil
        }
        let gathered = tokens == 1 && expertExecution?.isDecodePhase == true ? gatheredExperts : nil
        var gatheredRouted: [Float]?
        var contributions = deviceWorkspace == nil && gathered == nil ? [Float](repeating: 0, count: tokens * 10 * 2560) : []
        func sharedExpert() throws -> (value: [Float], gate: [Float]) {
            var shared: [Float] = []
            var sharedGate: [Float] = []
            shared.reserveCapacity(x.count)
            sharedGate.reserveCapacity(tokens)
            for lo in stride(from: 0, to: tokens, by: computeTile) {
                let tile = Array(x[(lo * 2560)..<((lo + computeTile) * 2560)])
                let gate = try weights.linear(tile, name: base + ".shared_expert.gate_proj")
                let up = try weights.linear(tile, name: base + ".shared_expert.up_proj")
                let activation = zip(gate, up).map { LayerArithmetic.round(weights.kernels.arithmetic.silu($0) * $1) }
                shared.append(contentsOf: try weights.linear(activation, name: base + ".shared_expert.down_proj"))
                sharedGate.append(
                    contentsOf: try weights.linear(tile, name: base + ".shared_expert_gate").map(weights.kernels.arithmetic.sigmoid))
            }
            return (shared, sharedGate)
        }
        let demandStart = gathered == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let window = try expertExecution?.makeWindow(
            layer: layer, experts: Array(selectedRows.keys),
            shouldContinue: { [unowned self] in self.shouldContinue() })
        defer { window?.close() }
        var earlyShared: (value: [Float], gate: [Float])?
        var sharedDuringDemandNanoseconds: UInt64 = 0
        if moeSharedReadOverlap, let gathered, window != nil {
            // All demand reads are already owned by the window. These unchanged
            // projections depend only on x; no routed result is needed. Any
            // throw still closes and joins pending producers via the defer.
            try boundary()
            let start = DispatchTime.now().uptimeNanoseconds
            earlyShared = try sharedExpert()
            sharedDuringDemandNanoseconds = DispatchTime.now().uptimeNanoseconds - start
            gathered.recordSharedDuringDemand(seconds: Double(sharedDuringDemandNanoseconds) / 1e9)
            try boundary()
        }
        if let gathered, let window {
            let sorted = selectedRows.keys.sorted()
            gatheredRouted = try window.consumeTen(experts: sorted) { buffers in
                // Keep this counter comparable: useful overlapped work is
                // disclosed separately, not counted as demand/pin waiting.
                gathered.recordDemandPinJoin(
                    seconds: Double(DispatchTime.now().uptimeNanoseconds - demandStart - sharedDuringDemandNanoseconds) / 1e9)
                return try gathered.evaluate(
                    x, records: buffers.map { .init(buffer: $0.buffer, offset: $0.offset) },
                    canonicalRanks: sorted.map { selectedRows[$0]![0].1 }, weights: scores[0], shouldContinue: shouldContinue)
            }
        } else if let window, expertCommandBatch > 1 {
            try batchedExperts(
                x, selectedRows: selectedRows, window: window,
                grouped: groupedExpertPrefill && computeTile >= 256, contributions: &contributions, deviceWorkspace: deviceWorkspace)
        } else {
            for expert in selectedRows.keys.sorted() {
                try boundary()
                let positions = selectedRows[expert]!
                let grouped = groupedExpertPrefill && computeTile >= 256
                func evaluate(_ buffer: ExpertBuffer?) throws -> [Float] {
                    var output: [Float] = []
                    output.reserveCapacity(positions.count * 2560)
                    for lo in stride(from: 0, to: positions.count, by: 256) {
                        try boundary()
                        let hi = min(positions.count, lo + 256)
                        let inputs = positions[lo..<hi].flatMap { Array(x[($0.0 * 2560)..<(($0.0 + 1) * 2560)]) }
                        if let buffer {
                            output.append(
                                contentsOf: try weights.expert(
                                    inputs, buffer: buffer, groupedExpert: grouped, shouldContinue: self.shouldContinue))
                        } else {
                            output.append(contentsOf: try weights.expert(inputs, layer: layer, expert: expert, groupedExpert: grouped))
                        }
                    }
                    return output
                }
                let output: [Float]
                if let window { output = try window.consume(expert: expert) { try evaluate($0) } } else { output = try evaluate(nil) }
                for (index, position) in positions.enumerated() {
                    let start = (position.0 * 10 + position.1) * 2560
                    contributions.replaceSubrange(start..<(start + 2560), with: output[(index * 2560)..<((index + 1) * 2560)])
                }
            }
        }
        let shared: [Float]
        let sharedGate: [Float]
        if let earlyShared {
            shared = earlyShared.value
            sharedGate = earlyShared.gate
        } else {
            let result = try sharedExpert()
            shared = result.value
            sharedGate = result.gate
        }
        let routed: [Float]?
        if let gatheredRouted {
            routed = gatheredRouted
        } else if let deviceWorkspace {
            guard let reductions else { throw EngineError("Device MoE lost its pinned reducer") }
            routed = try deviceWorkspace.reduce(weights: scores.flatMap { $0 }, using: reductions)
        } else {
            routed = try reductions?.weightedExpertReduce10(experts: contributions, weights: scores.flatMap { $0 })
        }
        var result = [Float](repeating: 0, count: x.count)
        for row in 0..<tokens {
            for column in 0..<2560 {
                var sum: Float = 0
                if let routed {
                    sum = routed[row * 2560 + column]
                } else {
                    for rank in 0..<10 { sum += contributions[(row * 10 + rank) * 2560 + column] * scores[row][rank] }
                }
                result[row * 2560 + column] = LayerArithmetic.round(
                    LayerArithmetic.round(sum) + LayerArithmetic.round(shared[row * 2560 + column] * sharedGate[row]))
            }
        }
        return result
    }
}

extension SerialExecutor {
    /// Final-layer union of a scope's read rows (`from...`): every expert routing one of them, with its complete row
    /// set, so its tiles and those rows' ten contributions are exactly the full union's.
    package static func keptExperts(_ selectedRows: [Int: [(Int, Int)]], from: Int) -> [Int: [(Int, Int)]] {
        selectedRows.filter { $0.value.contains { $0.0 >= from } }
    }
    static func lastRows(_ residual: [Float], count: Int, keep: Int) -> [[Float]] {
        (max(0, count - keep)..<count).map { Array(residual[($0 * 10240)..<(($0 + 1) * 10240)]) }
    }
    /// The MTP head's ten routed experts on the GPU (records in the head's own buffer), weighted and summed.
    package func mtpExperts(_ x: [Float], head: MTPHead, ids: [Int], weights gates: [Float]) throws -> [Float] {
        guard let schedule = wholeSchedule else { throw EngineError("MTP experts need the whole schedule", code: "pair_unsupported") }
        let bytes = AffineFormat(bits: 4, group: 64).expertRecordBytes
        return try schedule.gathered.evaluate(
            x, records: ids.map { .init(buffer: head.experts, offset: $0 * bytes) },
            canonicalRanks: Array(0..<10), weights: gates)
    }
    package func embeddingRow(_ id: Int) throws -> [Float] {
        try weights.enterLayer(nil)
        return try weights.embedding([id])
    }
    package func lmHead(_ hidden: [Float]) throws -> [Float] {
        try weights.enterLayer(nil)
        return try weights.linear(hidden, name: "lm_head")
    }

    /// A sequence's state before an MTP verify step, restored when its draft is rejected. GDN/QSA/PLE values share
    /// their immutable storage (a later append builds new pages/storage), so the copies are cheap.
    package struct PairSnapshot {
        let tokenCount: Int, history: [Int64], gdn: [Int: GDNState], qsa: [Int: QSAState], ple: PLEState
    }
    package func pairSnapshot(_ state: SequenceState) -> PairSnapshot {
        PairSnapshot(tokenCount: state.tokenCount, history: state.history, gdn: state.gdn, qsa: state.qsa, ple: state.ple)
    }
    package func restore(_ snapshot: PairSnapshot, into state: SequenceState) throws {
        state.tokenCount = snapshot.tokenCount
        state.history = snapshot.history
        state.gdn = snapshot.gdn
        state.qsa = snapshot.qsa
        state.ple = snapshot.ple
        state.emittedButUnconsumed = nil
        state.valid = true
        try math.rollbackWholeStorage(requestID: state.id, tokens: snapshot.tokenCount)
        try state.validate()
    }
    /// MTP verify step: the pending token and its draft as one two-row step (each layer's attention, router and shared
    /// expert for both rows in one command, the union of their experts in one window). Returns both rows' logits and
    /// final multi-stream residuals; consumes both tokens. Throws `pair_unsupported` (state restored) where a step
    /// needs the original path, a file-backed context or a missing layer owner; the caller decodes one token instead.
    package func forwardDecodePair(_ ids: [Int], state: SequenceState) throws -> (logits: [[Float]], hidden: [[Float]]) {
        guard ids.count == 2, let schedule = wholeSchedule, let execution = expertExecution, execution.isDecodePhase,
            active == state.id, !executingScope,
            state.pageArena?.fileStore == nil, state.tokenCount + 2 <= state.capacity
        else {
            throw EngineError("Pair step unavailable here", code: "pair_unsupported")
        }
        try QwenPairSchedule.requireUnionCapacity(execution.configuration, earlyDemand: schedule.earlyDemand != nil)
        if let pending = state.emittedButUnconsumed {
            guard pending == ids[0] else { throw EngineError("The pending generated token must be consumed exactly once") }
        }
        try state.validate()
        let before = pairSnapshot(state)
        attachGrowth()
        executingScope = true
        defer { executingScope = false }
        pairResidentGPUSeconds = 0
        do {
            return try autoreleasepool {
                if schedule.preencoded && pairResidencyBackoff.shouldProbe() {
                    do {
                        let result = try pairLayersResident(ids, state: state, schedule: schedule, execution: execution)
                        pairResidencyBackoff.recordHit()
                        return result
                    } catch let error as EngineError where error.code == "pair_miss" {
                        pairResidentMisses += 1
                        pairResidencyBackoff.recordMiss()
                        try restore(before, into: state)
                    }
                }
                return try pairLayersTile(ids, state: state, schedule: schedule, execution: execution)
            }
        } catch {
            try? schedule.joinAll()
            try restore(before, into: state)
            if (error as? EngineError)?.code == "pair_unsupported" || !(error is IOOperationCancelled) {
                throw EngineError("Pair step fell back: \(error)", code: "pair_unsupported")
            }
            throw error
        }
    }
    /// Pair step without host round trips (Macs that keep the experts in memory): the two-row workspace, each layer's
    /// experts found on the GPU in the residency table, one command per segment ending at a QSA layer (whose published
    /// keys are read back) and after layer 0 (the host PLE). Throws `pair_miss` when any routed expert was not
    /// resident (the caller restores and takes the host path).
    private func pairLayersResident(_ ids: [Int], state: SequenceState, schedule: WholeExecutionSchedule, execution: ExpertExecution) throws
        -> (logits: [[Float]], hidden: [[Float]])
    {
        func unsupported(_ why: String) -> EngineError { EngineError("Pair step: \(why)", code: "pair_unsupported") }
        try schedule.beginScope(decode: true)
        defer {
            try? schedule.joinAll()
            schedule.endScope()
        }
        try boundary()
        let p = try schedule.pairTile()
        let tiles = [try schedule.decodeTile(0), try schedule.decodeTile(1)]
        let groupBytes = GatheredExpertDecode.deferredGridBytes
        if pairGrids == nil { pairGrids = weights.kernels.device.makeBuffer(length: 48 * 2 * groupBytes, options: .storageModeShared) }
        guard let grids = pairGrids else { throw unsupported("no grid buffer") }
        // Hold growth first, so no slab is added after the residency set's buffers are taken.
        execution.holdSlabGrowth(true)
        defer {
            execution.releaseForGPU(layer: nil)
            execution.holdSlabGrowth(false)
        }
        let records = execution.recordBuffers()
        let table = try execution.gpuTable()
        try weights.enterLayer(nil)
        let embedding = try weights.embedding(ids)
        for row in 0..<2 {
            for branch in 0..<4 {
                try schedule.copy(
                    Array(embedding[(row * 2560)..<((row + 1) * 2560)]), to: p[.residual], offset: (row * 10240 + branch * 2560) * 4)
            }
        }
        let history = state.history + ids.map(Int64.init)
        var gpuSeconds = 0.0

        var segmentStart = 0
        while segmentStart < 48 {
            let segmentEnd = segmentStart == 0 ? 0 : min(47, (segmentStart / 4) * 4 + 3)
            try boundary()
            try autoreleasepool {
                if segmentStart == 1 {
                    let old = try schedule.read(p[.result], count: 2 * 10240)
                    let extra = try ple.call(old, history: history, newCount: 2, state: &state.ple)
                    try schedule.copy(zip(old, extra).map { LayerArithmetic.round($0 + $1) }, to: p[.residual])
                }
                p.clearFlags()
                for t in tiles { t.clearFlags() }
                var command = try schedule.command("pair-resident")
                var pendings: [WholeLayerExecution.Pending] = []
                var owners: [WholeLayerWeights] = []
                var published = 0
                defer { for p in pendings.dropFirst(published) { try? p.discard() } }
                for layer in segmentStart...segmentEnd {
                    try weights.enterLayer(layer)
                    guard let owner = try weights.acquireWholeLayerWeights(layer: layer) else {
                        throw unsupported("layer \(layer) has no whole owner")
                    }
                    owners.append(owner)
                    let mirror =
                        layer % 4 == 3 && state.pageArena == nil ? try math.wholeDecodeMirror(layer: layer, requestID: state.id) : nil
                    let paged =
                        layer % 4 == 3 && state.pageArena != nil ? try math.wholePagedStorage(layer: layer, requestID: state.id) : nil
                    pendings.append(
                        try schedule.layer.encodeAttentionRouter(
                            &command, residual: layer <= 1 ? p[.residual] : p[.result],
                            weights: owner, state: state, mirror: mirror, paged: paged, workspace: p))
                    try schedule.layer.encodeShared(command, weights: owner, workspace: p)
                    let epoch = execution.holdForGPU(layer: layer)
                    guard let toRows = command.makeBlitCommandEncoder() else { throw unsupported("blit unavailable") }
                    for (row, t) in tiles.enumerated() {
                        toRows.copy(from: p[.routerPairs], sourceOffset: row * 80, to: t[.routerPairs], destinationOffset: 0, size: 80)
                        toRows.copy(from: p[.routerWeights], sourceOffset: row * 40, to: t[.routerWeights], destinationOffset: 0, size: 40)
                        toRows.copy(
                            from: p[.mixedMoEBF16], sourceOffset: row * 5120, to: t[.mixedMoEBF16], destinationOffset: 0, size: 5120)
                    }
                    toRows.endEncoding()
                    for (row, t) in tiles.enumerated() {
                        guard
                            try schedule.encodeResidentExperts(
                                command, tile: t, layer: layer, records: records, table: table, epoch: epoch,
                                grids: grids, offset: (layer * 2 + row) * groupBytes)
                        else { throw unsupported("pre-encoded decode is off") }
                    }
                    guard let back = command.makeBlitCommandEncoder() else { throw unsupported("blit unavailable") }
                    for (row, t) in tiles.enumerated() {
                        back.copy(from: t[.routed], sourceOffset: 0, to: p[.routed], destinationOffset: row * 10240, size: 10240)
                    }
                    back.endEncoding()
                    try schedule.layer.encodeMix(command, workspace: p)
                }
                try schedule.joined(command, shouldContinue: shouldContinue)
                gpuSeconds += max(0, command.gpuEndTime - command.gpuStartTime)
                for layer in segmentStart...segmentEnd {
                    for row in 0..<2 {
                        let found = grids.contents().load(fromByteOffset: (layer * 2 + row) * groupBytes + 4, as: UInt32.self)
                        guard found == 10 else {
                            pairResidentGPUSeconds = gpuSeconds
                            throw EngineError("Pair step: layer \(layer) has experts outside memory", code: "pair_miss")
                        }
                    }
                }
                guard !schedule.flagged(p, [.sharedFlags, .combinationFlags, .finalInjectionFlags]) else {
                    throw unsupported("a layer flagged a replay")
                }
                for pending in pendings {
                    guard try !pending.requiresReplay() else { throw unsupported("a layer needs the original path") }
                    try pending.publish(state: state)
                    published += 1
                }
                withExtendedLifetime(owners) {}
            }
            segmentStart = segmentEnd + 1
        }
        pairResidentGPUSeconds = gpuSeconds
        try weights.enterLayer(nil)
        var logits: [[Float]] = []
        var hidden: [[Float]] = []
        for row in 0..<2 {
            let residual = try schedule.read(p[.result], count: 10240, offset: row * 10240)
            let mixed = try math.hyperConnection(residual: residual, base: "model.hyper_connection_mixer", useCombine: false).mixed
            logits.append(try weights.linear(mixed, name: "lm_head"))
            hidden.append(residual)
        }
        state.history = Array(history.suffix(2))
        state.tokenCount += 2
        state.emittedButUnconsumed = nil
        try state.validate()
        try boundary()
        return (logits, hidden)
    }

    /// Pair step on the two-row workspace: every projection of both rows reads its weights once (two-row tile8).
    /// The router is joined before shared work, so the union's SSD reads overlap the independent shared expert.
    private func pairLayersTile(_ ids: [Int], state: SequenceState, schedule: WholeExecutionSchedule, execution: ExpertExecution) throws
        -> (logits: [[Float]], hidden: [[Float]])
    {
        func unsupported(_ why: String) -> EngineError { EngineError("Pair step: \(why)", code: "pair_unsupported") }
        try schedule.beginScope(decode: true)
        defer {
            try? schedule.joinAll()
            schedule.endScope()
        }
        try boundary()
        let p = try schedule.pairTile()
        let rowTiles = [try schedule.decodeTile(0), try schedule.decodeTile(1)]
        try weights.enterLayer(nil)
        let embedding = try weights.embedding(ids)
        for row in 0..<2 {
            for branch in 0..<4 {
                try schedule.copy(
                    Array(embedding[(row * 2560)..<((row + 1) * 2560)]), to: p[.residual], offset: (row * 10240 + branch * 2560) * 4)
            }
        }
        let history = state.history + ids.map(Int64.init)
        for layer in 0..<48 {
            try boundary()
            try autoreleasepool {
                try weights.enterLayer(layer)
                guard let owner = try weights.acquireWholeLayerWeights(layer: layer) else {
                    throw unsupported("layer \(layer) has no whole owner")
                }
                if layer == 1 {
                    let old = try schedule.read(p[.result], count: 2 * 10240)
                    let extra = try ple.call(old, history: history, newCount: 2, state: &state.ple)
                    try schedule.copy(zip(old, extra).map { LayerArithmetic.round($0 + $1) }, to: p[.residual])
                }
                p.clearFlags()
                var command = try schedule.command("pair-attention-router")
                let mirror = layer % 4 == 3 && state.pageArena == nil ? try math.wholeDecodeMirror(layer: layer, requestID: state.id) : nil
                let paged = layer % 4 == 3 && state.pageArena != nil ? try math.wholePagedStorage(layer: layer, requestID: state.id) : nil
                let pending = try schedule.layer.encodeAttentionRouter(
                    &command, residual: layer <= 1 ? p[.residual] : p[.result], weights: owner,
                    state: state, mirror: mirror, paged: paged, workspace: p)
                var published = false
                defer { if !published { try? pending.discard() } }
                try schedule.joined(command, shouldContinue: shouldContinue)
                guard try !pending.requiresReplay() else { throw unsupported("layer \(layer) needs the original path") }
                let routing = try pending.routing()
                for row in 0..<2 {
                    let t = rowTiles[row]
                    t[.mixedMoEBF16].contents().copyMemory(
                        from: p[.mixedMoEBF16].contents().advanced(by: row * 2560 * 2), byteCount: 2560 * 2)
                    t[.routerWeights].contents().copyMemory(
                        from: p[.routerWeights].contents().advanced(by: row * 10 * 4), byteCount: 10 * 4)
                }
                let union = Array(Set(routing.ids[0] + routing.ids[1])).sorted()
                let window = try execution.makeWindow(
                    layer: layer, experts: union, historyRow: routing.ids[0], shouldContinue: shouldContinue)
                defer { window.close() }
                let shared = try schedule.pairSharedDuringReads(
                    missing: window.residentAtRequestCount < union.count,
                    shouldContinue: shouldContinue
                ) { command in
                    try schedule.layer.encodeShared(command, weights: owner, workspace: p)
                }
                try window.consumeUnion { buffers in
                    let record = Dictionary(
                        uniqueKeysWithValues: zip(union, buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }))
                    let command = try schedule.command("pair-experts")
                    // All-resident unions keep two commands per layer; only a missing union needs the overlap command.
                    if shared == nil { try schedule.layer.encodeShared(command, weights: owner, workspace: p) }
                    for row in 0..<2 {
                        let t = rowTiles[row]
                        let sorted = routing.ids[row].sorted()
                        try schedule.gathered.encodeRegion(
                            command, inputBF16: t[.mixedMoEBF16], weights: t[.routerWeights],
                            records: sorted.map { record[$0]! }, canonicalRanks: sorted.map { routing.ids[row].firstIndex(of: $0)! },
                            output: t[.routed], buffers: t.experts)
                    }
                    guard let blit = command.makeBlitCommandEncoder() else { throw EngineError("Pair routed copy unavailable") }
                    for row in 0..<2 {
                        blit.copy(
                            from: rowTiles[row][.routed], sourceOffset: 0, to: p[.routed], destinationOffset: row * 2560 * 4, size: 2560 * 4
                        )
                    }
                    blit.endEncoding()
                    try schedule.layer.encodeMix(command, workspace: p)
                    try schedule.joined(command, shouldContinue: shouldContinue)
                }
                if let shared { try schedule.join(shared) }
                guard !schedule.flagged(p, [.sharedFlags, .combinationFlags, .finalInjectionFlags]) else {
                    throw unsupported("layer \(layer) flagged a replay")
                }
                try pending.publish(state: state)
                published = true
            }
        }
        try weights.enterLayer(nil)
        var logits: [[Float]] = []
        var hidden: [[Float]] = []
        for row in 0..<2 {
            let residual = try schedule.read(p[.result], count: 10240, offset: row * 10240)
            let mixed = try math.hyperConnection(residual: residual, base: "model.hyper_connection_mixer", useCombine: false).mixed
            logits.append(try weights.linear(mixed, name: "lm_head"))
            hidden.append(residual)
        }
        state.history = Array(history.suffix(2))
        state.tokenCount += 2
        state.emittedButUnconsumed = nil
        try state.validate()
        try boundary()
        return (logits, hidden)
    }

    private func forwardWholeDecode(
        _ id: Int, state: SequenceState, maximumLayers: Int, returnLogits: Bool,
        schedule: WholeExecutionSchedule, execution: ExpertExecution
    ) throws -> [Float] {
        if decodeTokenPlan { try weights.validateWholeTokenPlan() }
        let delayPolicy = requestDial.delayedExperts
        guard delayPolicy == nil || captureHiddenRows == 0 else {
            throw EngineError("Delayed experts cannot run with speculation", code: "invalid_configuration")
        }
        let earlyDemand = delayPolicy == nil ? schedule.earlyDemand : nil
        var delayedSource: (layer: Int, rank: Int, gate: Float, workspace: WholeExecutionWorkspace)?
        var injectedAt: Int?
        try schedule.beginScope(decode: true)
        var pending: WholeLayerExecution.Pending?
        var lookahead: WholeLayerExecution.Pending?
        var stagedDemand: DecodeEarlyDemand.Batch?
        // Pre-routing batches: target T is staged with the command that computes layer T-1's post-attention
        // row (so batches T and T+1 are alive together) and adopted at iteration T; any replay closes all.
        var prerouted: [DecodePrerouting.Batch] = []
        let prerouting = earlyDemand == nil ? nil : schedule.prerouting
        // Read hygiene (nil when off): cancels the still-queued unrouted predictions at adoption.
        let cancelUnrouted = prerouting == nil ? nil : execution.preroutingCancelUnrouted()
        var sharedPreencoded = false
        var nextSharedPreencoded = false
        func closePrerouted() {
            for batch in prerouted { prerouting?.close(batch) }
            prerouted.removeAll()
        }
        defer {
            // Weight/state owners remain strong through this joined cleanup,
            // including a shared command that was submitted before demand failed.
            try? schedule.joinAll()
            stagedDemand?.close()
            try? lookahead?.discard()
            try? pending?.discard()
            closePrerouted()
            delayedRead?.discard()
            delayedSource = nil
            lookahead = nil
            pending = nil
            schedule.endScope()
        }
        func injectDelayed(_ command: inout MTLCommandBuffer, target: Int, into targetStorage: WholeExecutionWorkspace) throws {
            guard let source = delayedSource, let read = delayedRead, read.isPending, let storage = delayedStorage else { return }
            guard source.layer + 1 == target else { throw EngineError("Delayed Qwen contribution crossed its attention deadline") }
            try schedule.submit(command, shouldContinue: shouldContinue)
            try schedule.join(command)
            delayedExpertCounters.deadlineJoins += 1
            try read.consume(shouldContinue: shouldContinue) { record in
                let work = try schedule.command("delayed-source-expert")
                do {
                    try schedule.gathered.encodeReadyGroup(
                        work, inputBF16: storage.input,
                        records: [.init(buffer: record.buffer, offset: record.offset)], canonicalRanks: [source.rank],
                        buffers: source.workspace.experts)
                    try schedule.submit(work, shouldContinue: shouldContinue)
                    try schedule.join(work)
                } catch {
                    try? schedule.joinAll()
                    throw error
                }
                let expert = source.workspace.experts.down.contents().assumingMemoryBound(to: Float.self) + source.rank * 2560
                let post = storage.post.contents().assumingMemoryBound(to: Float.self)
                let streams = targetStorage[.afterAttention].contents().assumingMemoryBound(to: Float.self)
                for d in 0..<4 {
                    for j in 0..<2560 {
                        let delta = LayerArithmetic.round(expert[j] * source.gate)
                        streams[d * 2560 + j] = LayerArithmetic.round(streams[d * 2560 + j] + LayerArithmetic.round(delta * post[d]))
                    }
                }
            }
            delayedSource = nil
            injectedAt = target
            delayedExpertCounters.appliedContributions += 1
            command = try schedule.command("whole-decode-after-delayed")
        }
        /// Layer `target`'s MoE HC + router over `afterAttentionBF16` (the row of layer target-1), encoded after
        /// every exact-demand signal of `command`. Missing plan weights are counted, never replaced.
        func stagePrerouting(_ command: MTLCommandBuffer, afterAttentionBF16: MTLBuffer, target: Int) throws {
            guard let prerouting, target < maximumLayers else { return }
            guard prerouted.count < 2, !prerouted.contains(where: { $0.target == target }) else {
                throw EngineError("Pre-routing exceeded its two-batch boundary")
            }
            guard let owner = weights.wholeTokenPlanWeights(layer: target) else {
                prerouting.recordWeightsUnavailable()
                return
            }
            let factory = try execution.preroutingFactory(overrideWidth: decodePreroutingOverrideWidth)
            let batch = try prerouting.stage(
                command: command, afterAttentionBF16: afterAttentionBF16,
                moeHC: owner.moeHC, router: owner.router, target: target, dial: requestDial,
                operation: { ids, layer, keepGoing in try factory(ids, layer, keepGoing) }, discard: execution.earlyDemandDiscard())
            prerouted.append(batch)
        }
        // Pre-encoded layers (`decode_preencoded_layers`): from layer 2 on, each layer's experts, reduce and mix are queued behind its
        // attention before its routing is known. The host waits for the router signal, fills the records and signals,
        // and queues the next layer while the GPU runs this one. Slabs stop growing while such commands are queued.
        // A file-backed context (256K) joins its attention's selection on the host, so it keeps the joined path.
        let preencode =
            schedule.preencoded && state.pageArena?.fileStore == nil && earlyDemand != nil && delayPolicy == nil
        var growthHeld = false
        var armedEpoch: [UInt32] = [0, 0]
        var routerSignal: (event: MTLSharedEvent, value: UInt64)?
        defer {
            if growthHeld {
                try? schedule.joinAll()
                execution.releaseForGPU(layer: nil)
                execution.holdSlabGrowth(false)
            }
        }
        /// Encodes and submits layer `next`'s attention, router, early demand, pre-routing and shared expert behind the
        /// work that writes `s`'s result, then (pre-encoded layers) its experts. Nil when the next layer takes the joined path.
        func queueNextLayer(from s: WholeExecutionWorkspace, next: Int, nextGeneration: Int) throws -> MTLCommandBuffer? {
            guard next < maximumLayers, next != 1 else { return nil }
            let began = DispatchTime.now().uptimeNanoseconds
            defer { schedule.recordDecodeWait(encode: Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9) }
            try weights.enterLayer(next)
            guard let owner = try weights.acquireWholeLayerWeights(layer: next) else { return nil }
            var lookaheadCommand = try schedule.command("whole-decode-next-router")
            let nextStorage = try schedule.decodeTile(nextGeneration)
            nextStorage.clearFlags()
            let mirror = next % 4 == 3 && state.pageArena == nil ? try math.wholeDecodeMirror(layer: next, requestID: state.id) : nil
            let paged = next % 4 == 3 && state.pageArena != nil ? try math.wholePagedStorage(layer: next, requestID: state.id) : nil
            lookahead = try schedule.layer.encodeAttentionRouter(
                &lookaheadCommand, residual: s[.result], weights: owner,
                state: state, mirror: mirror, paged: paged, workspace: nextStorage,
                joinSelection: { try schedule.joinedFileSelection($0, shouldContinue: self.shouldContinue) },
                afterAttention: delayPolicy == nil ? nil : { command in try injectDelayed(&command, target: next, into: nextStorage) })
            if let early = earlyDemand {
                // A pre-encoded layer's host issues its own reads when it sees the router signal (spinning, so sooner
                // than the demand listener would); the joined path stages them on the listener.
                if preencode {
                    routerSignal = try schedule.encodeRouterSignal(lookaheadCommand)
                } else if !requestDial.dropsExperts {
                    // shortcut: a dial that drops experts reads them only when the host sees the routing (early demand
                    // reads the routed ten); in-memory plans pre-encode these layers, so this costs only other plans.
                    let view = DecodeEarlyDemand.RouterView(next: nextStorage, preceding: s, layer: next)
                    let factory = try execution.earlyDemandFactory()
                    stagedDemand = try early.stage(
                        command: lookaheadCommand,
                        operation: { keepGoing in try factory(view.ids(), view.layer, keepGoing) },
                        discard: execution.earlyDemandDiscard())
                    schedule.stagedDemand()
                }
                try stagePrerouting(lookaheadCommand, afterAttentionBF16: nextStorage[.afterAttentionBF16], target: next + 1)
                try schedule.layer.encodeShared(lookaheadCommand, weights: owner, workspace: nextStorage)
                nextSharedPreencoded = true
            }
            try schedule.submit(lookaheadCommand, shouldContinue: shouldContinue)
            if preencode {
                if !growthHeld {
                    execution.holdSlabGrowth(true)
                    growthHeld = true
                }
                // The layer's cached records stay put until its command finishes; it trusts those published before now.
                armedEpoch[nextGeneration] = execution.holdForGPU(layer: next)
                _ = try schedule.submitPreencodedExperts(
                    tile: nextGeneration, layer: next, workspace: nextStorage,
                    records: execution.recordBuffers(), table: execution.gpuTable(), epoch: armedEpoch[nextGeneration],
                    shouldContinue: shouldContinue)
            }
            return lookaheadCommand
        }
        do {
            try boundary()
            let first = try schedule.decodeTile(0)
            _ = try schedule.decodeTile(1)  // Both generations admitted before state mutation.
            try weights.enterLayer(nil)
            let embedding = try weights.embedding([id])
            for branch in 0..<4 { try schedule.copy(embedding, to: first[.residual], offset: branch * 2560 * 4) }
            let history = state.history + [Int64(id)]
            // Device PLE reads this token's 16 rows while layers 0 and 1 are prepared.
            let deviceDecode = devicePLE != nil && !ple.hasCapture && maximumLayers > 1
            var tokenRows: PLERowPrefetch?
            if deviceDecode {
                tokenRows = try decodePLERows(history: history, prerouting: prerouting, execution: execution)
            }
            var generation = 0
            /// A layer whose attention the previous layer did not queue: its weights, device PLE (layer 1), attention and
            /// router, joined. True when it has no whole-layer weights (the original arithmetic runs it).
            func encodeFirstRouter(_ layerIndex: Int, _ s: WholeExecutionWorkspace) throws -> Bool {
                var original = false
                try weights.enterLayer(layerIndex)
                let owner = try weights.acquireWholeLayerWeights(layer: layerIndex)
                var deviceWork: (ple: DevicePLE, workspace: DevicePLE.Workspace, projections: DevicePLE.Projections, rows: PLERowPrefetch)?
                if layerIndex == 1, owner != nil, let devicePLE, let rows = tokenRows,
                    let projections = try weights.devicePLEProjections()
                {
                    if devicePLEDecodeWorkspace == nil { devicePLEDecodeWorkspace = try devicePLE.makeWorkspace(rows: 1) }
                    deviceWork = (devicePLE, devicePLEDecodeWorkspace!, projections, rows)
                } else if layerIndex == 1 {
                    if deviceDecode { devicePLE?.recordFallback() }
                    let old = try schedule.read(s[.residual], count: 10240)
                    let extra = try ple.call(old, history: history, newCount: 1, state: &state.ple)
                    try schedule.copy(zip(old, extra).map { LayerArithmetic.round($0 + $1) }, to: s[.residual])
                }
                if let owner {
                    s.clearFlags()
                    var command = try schedule.command("whole-decode-first-router")
                    if let work = deviceWork {
                        let waited = try work.rows.wait(shouldContinue: shouldContinue)
                        work.ple.recordRows(requested: 16, read: work.rows.uniqueRows, waitSeconds: waited, decode: true)
                        let packed = try work.rows.packed(tokens: 0..<1)
                        try packed.withUnsafeBytes {
                            try work.ple.encode(
                                command, residual: s[.residual], rows: 1, packedRows: $0,
                                state: state.ple.convolution, projections: work.projections, workspace: work.workspace, decode: true
                            )
                        }
                    }
                    let mirror =
                        layerIndex % 4 == 3 && state.pageArena == nil
                        ? try math.wholeDecodeMirror(layer: layerIndex, requestID: state.id) : nil
                    let paged =
                        layerIndex % 4 == 3 && state.pageArena != nil
                        ? try math.wholePagedStorage(layer: layerIndex, requestID: state.id) : nil
                    pending = try schedule.layer.encodeAttentionRouter(
                        &command, residual: s[.residual], weights: owner,
                        state: state, mirror: mirror, paged: paged, workspace: s,
                        joinSelection: { try schedule.joinedFileSelection($0, shouldContinue: self.shouldContinue) },
                        afterAttention: delayPolicy == nil
                            ? nil : { command in try injectDelayed(&command, target: layerIndex, into: s) })
                    try stagePrerouting(command, afterAttentionBF16: s[.afterAttentionBF16], target: layerIndex + 1)
                    try schedule.joined(command, shouldContinue: shouldContinue)
                    // Both the encoded and the replay path use this PLE result.
                    if let work = deviceWork { state.ple.convolution = work.ple.nextState(work.workspace) }
                } else {
                    original = true
                }
                return original
            }
            /// A routed layer's experts (resident first, the rest as they are read), its reduce and mix, and the next layer
            /// queued behind them. True when its flags require the original arithmetic's replay.
            func runRoutedLayer(
                _ layerIndex: Int, _ generation: Int, _ s: WholeExecutionWorkspace, _ preencoded: MTLCommandBuffer?
            ) throws -> Bool {
                var original = false
                guard let current = pending, current.weights.layer == layerIndex, current.workspace === s else {
                    throw EngineError("Whole decode transaction/generation differs")
                }
                let routing = try current.routing()
                let routed = routing.ids[0]
                let sorted = routed.sorted()
                // Quality dial (exact: unchanged): the routed ranks this token keeps. Their renormalised gates are
                // written before anything that reads them is encoded or signalled; dropped experts are not read
                // (the window holds only the kept ones) and get a pinned kept record with a zero gate.
                var gates = Array(routing.probabilities.prefix(10))
                var kept = sorted
                try applyQualityDial(
                    layerIndex: layerIndex, routed: routed, gates: &gates, kept: &kept, s: s, schedule: schedule,
                    execution: execution)
                // Pre-encoded layers: the GPU computes the experts its residency table held from before the command was queued, the
                // host reads and fills the rest (both apply the same rule, so they split the ten exactly). When the
                // GPU takes all ten, its late wait is released before the window's bookkeeping.
                var onGPU = Set<Int>()
                var signalled = false
                if preencoded != nil {
                    let keys = try sorted.map { try ExpertKey(layer: layerIndex, expert: $0) }
                    onGPU = Set(zip(sorted, execution.gpuComputes(keys, epoch: armedEpoch[generation])).filter(\.1).map(\.0))
                    if onGPU.count == 10 {
                        try schedule.fillPreencoded(tile: generation, workspace: s, records: [], ranks: [])
                        signalled = true
                    }
                }
                var stagedRequests: [ExpertLoadRequest] = []
                if let batch = stagedDemand {
                    stagedRequests = try batch.adopt(expected: kept.map { try ExpertKey(layer: layerIndex, expert: $0) })
                    stagedDemand = nil
                    schedule.adoptedDemand(batch.observation)
                }
                let window: ExpertDemandWindow
                do {
                    let began = DispatchTime.now().uptimeNanoseconds
                    window = try execution.makeWindow(
                        layer: layerIndex, experts: kept, staged: stagedRequests, shouldContinue: shouldContinue)
                    schedule.recordDecodeWait(window: Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9)
                } catch {
                    execution.earlyDemandDiscard()(stagedRequests)
                    throw error
                }
                stagedRequests.removeAll()
                // The window now holds its own handles for every demanded record; the
                // prediction is scored against this authoritative routing and released.
                if let index = prerouted.firstIndex(where: { $0.target == layerIndex }) {
                    // Its unrouted reads may still run: kept until a read boundary joins them.
                    execution.retainPredictions(
                        prerouting?.adopt(prerouted.remove(at: index), actual: kept, cancelUnrouted: cancelUnrouted) ?? [])
                }
                for batch in prerouted where batch.target < layerIndex { prerouting?.close(batch) }
                prerouted.removeAll { $0.target < layerIndex }
                defer { window.close() }
                // The preceding span already encoded shared work. For
                // zero-hit misses and all-hit layers the former shared
                // command had no encoders; do not submit that boundary.
                let missingLayer = window.residentAtRequestCount < kept.count
                var shared: MTLCommandBuffer?
                var earlyComputed = false
                var sharedJoined = false
                var currentPublished = false
                var sharedEncoded = sharedPreencoded
                // Fast late-miss skip (nil = exact): the skipped late records get gate 0 and the kept gates are
                // rescaled to the routed mass, written before the reduce that reads them is encoded. With
                // `substitute`, a skipped record's rank computes the best expert in memory instead, weighted by its
                // own router score.
                var skipLate: (([Int], Int) throws -> [Int])?
                var substitute: (([Int]) throws -> [Int: Int])?
                if let policy = delayPolicy, let storage = delayedStorage, delayedSource == nil, layerIndex > 0,
                    layerIndex + 1 < maximumLayers, weights.wholeTokenPlanWeights(layer: layerIndex + 1) != nil,
                    ledger.optionalGrowthAllowed
                {
                    let originalGates = gates
                    skipLate = { late, capacity in
                        let ranks = Set(late.compactMap { routed.firstIndex(of: $0) })
                        guard let rank = policy.rank(weights: originalGates, late: ranks, capacity: capacity) else { return [] }
                        storage.input.contents().copyMemory(from: s[.mixedMoEBF16].contents(), byteCount: 2560 * 2)
                        storage.post.contents().copyMemory(from: s[.hcMInjection].contents(), byteCount: 16)
                        var immediate = originalGates
                        immediate[rank] = 0
                        try schedule.copy(immediate, to: s[.routerWeights])
                        delayedSource = (layerIndex, rank, originalGates[rank], s)
                        self.delayedExpertCounters.deferredExperts += 1
                        self.delayedExpertCounters.deferredMass += Double(originalGates[rank] / originalGates.reduce(0, +))
                        return [routed[rank]]
                    }
                }
                if let tau = fastLateMissSkipTau {
                    let made = fastLateMissSkip(
                        tau: tau, gates: gates, layerIndex: layerIndex, routed: routed, s: s, schedule: schedule,
                        execution: execution)
                    skipLate = made.skipLate
                    substitute = made.substitute
                }
                let dropped = sorted.filter { !kept.contains($0) && !onGPU.contains($0) }
                var readyAt = DispatchTime.now().uptimeNanoseconds
                func lateWait(_ ids: [Int]) {
                    schedule.recordDecodeWait(
                        late: Double(DispatchTime.now().uptimeNanoseconds - readyAt) / 1e9, lateExperts: ids.count)
                }
                if let experts = preencoded {
                    var queued = false
                    func queueNext() throws {
                        _ = try queueNextLayer(from: s, next: layerIndex + 1, nextGeneration: 1 - generation)
                        queued = true
                    }
                    try window.consumeTenReadyFirst(
                        experts: kept, joinReaders: { try? schedule.joinIfInFlight(experts) },
                        skipLate: skipLate, substitute: substitute, dropped: dropped, early: onGPU,
                        ready: { _, _ in
                            // Queue the next layer while late reads run; an all-resident layer signals first.
                            if !signalled { try queueNext() }
                            readyAt = DispatchTime.now().uptimeNanoseconds
                        },
                        finish: { ids, records in
                            lateWait(ids)
                            if !signalled {
                                try schedule.fillPreencoded(
                                    tile: generation, workspace: s,
                                    records: records.map { .init(buffer: $0.buffer, offset: $0.offset) },
                                    ranks: ids.map { routing.ids[0].firstIndex(of: $0)! })
                            }
                            if !queued { try queueNext() }
                            try schedule.join(experts)
                            execution.releaseForGPU(layer: layerIndex)
                            try schedule.joinIfInFlight(current.command)
                            try boundary()
                        })
                } else {
                    try window.consumeTenReadyFirst(
                        experts: kept, joinReaders: { try? schedule.joinAll(keepingArmed: 1 - generation) }, skipLate: skipLate,
                        substitute: substitute, dropped: dropped,
                        delayed: delayPolicy == nil ? nil : delayedRead,
                        ready: { ids, records in
                            do {
                                shared = try schedule.independentCommand(
                                    sharedPreencoded: sharedEncoded, readyCount: records.count,
                                    coalesce: decodeTokenPlan, missing: missingLayer)
                                if !sharedEncoded {
                                    try schedule.layer.encodeShared(shared!, weights: current.weights, workspace: s)
                                    sharedEncoded = true
                                }
                                if !records.isEmpty {
                                    let ranks = ids.map { routing.ids[0].firstIndex(of: $0)! }
                                    try schedule.gathered.encodeReadyGroup(
                                        shared!, inputBF16: s[.mixedMoEBF16],
                                        records: records.map { .init(buffer: $0.buffer, offset: $0.offset) },
                                        canonicalRanks: ranks, buffers: s.experts)
                                    earlyComputed = true
                                }
                                if let shared { try schedule.submit(shared, shouldContinue: shouldContinue) }
                                if earlyComputed && earlyDemand != nil { schedule.skippedResidentJoin() }
                                if let shared, earlyComputed && earlyDemand == nil {
                                    // I/O continues during this GPU join. The fixed
                                    // descriptor/rank backing is now safe to reuse.
                                    try schedule.join(shared)
                                    sharedJoined = true
                                }
                            } catch {
                                try? schedule.joinAll()
                                throw error
                            }
                            readyAt = DispatchTime.now().uptimeNanoseconds
                        },
                        finish: { ids, records in
                            lateWait(ids)
                            // Pin cleanup in the demand owner must never precede
                            // submitted expert readers, including a failed submit
                            // boundary or dependent next-layer encoding failure.
                            var publication: DecodeEarlyDemand.Frontier?
                            // The next layer's armed (pre-encoded) experts read none of this layer's records and stay queued.
                            var joined = false
                            defer {
                                if !joined { try? schedule.joinAll() }
                                publication?.close()
                            }
                            try boundary()
                            let command = try schedule.command("whole-decode-experts")
                            let ranks = ids.map { routing.ids[0].firstIndex(of: $0)! }
                            let views = records.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
                            if earlyComputed {
                                try schedule.gathered.encodeReadyGroup(
                                    command, inputBF16: s[.mixedMoEBF16],
                                    records: views, canonicalRanks: ranks,
                                    buffers: earlyDemand == nil ? s.experts : s.missingExperts)
                                try schedule.reductions.encodeWeightedExpertReduce10(
                                    command, input: s.experts.down,
                                    weights: s[.routerWeights], output: s[.routed], tokens: 1)
                            } else {
                                try schedule.gathered.encodeRegion(
                                    command, inputBF16: s[.mixedMoEBF16], weights: s[.routerWeights],
                                    records: views, canonicalRanks: ranks, output: s[.routed], buffers: s.experts)
                            }
                            try schedule.layer.encodeMix(command, workspace: s)
                            let next = layerIndex + 1
                            if next < maximumLayers, next != 1, let early = earlyDemand {
                                publication = try early.checkpoint(command: command)
                            }
                            // The experts' command starts on the GPU before the next layer's attention is
                            // encoded, so that encoding overlaps it; queue order keeps the dependency.
                            try schedule.submit(command, shouldContinue: shouldContinue)
                            // PLE consumes host history/state and is an explicit
                            // boundary. No speculative IDs ever drive next I/O.
                            // Queue order protects shared outputs without a
                            // separate CPU join. All ten slots stay pinned until
                            // both their readers and dependent attention finish.
                            let nextCommand = try queueNextLayer(from: s, next: next, nextGeneration: 1 - generation)
                            if let publication {
                                let beforeCompletion = try publication.waitUntilReady(shouldContinue: shouldContinue)
                                if !schedule.flagged(s, [.sharedFlags, .combinationFlags, .finalInjectionFlags]) {
                                    // Current attention already joined. Its state
                                    // storage is disjoint from next attention;
                                    // the checkpoint finishes current MoE flags.
                                    try current.publish(state: state)
                                    currentPublished = true
                                }
                                schedule.publicationCheckpoint(
                                    early: beforeCompletion && currentPublished,
                                    waitSeconds: publication.waitSeconds)
                            }
                            try schedule.join(command)
                            if let nextCommand { try schedule.join(nextCommand) }
                            if let shared, !sharedJoined { try schedule.join(shared) }
                            joined = true
                            try boundary()
                        })
                }
                window.close()
                if schedule.flagged(s, [.sharedFlags, .combinationFlags, .finalInjectionFlags]) {
                    try schedule.joinAll()
                    execution.releaseForGPU(layer: nil)  // the queued next layer (G4: its late group runs empty)
                    stagedDemand?.close()
                    stagedDemand = nil
                    nextSharedPreencoded = false
                    closePrerouted()
                    try lookahead?.discard()
                    if let discarded = lookahead, discarded.weights.layer % 4 == 3 {
                        try math.retireInvalidWholeMirror(layer: discarded.weights.layer, requestID: state.id)
                    }
                    lookahead = nil
                    guard !currentPublished else { throw EngineError("Decode layer published before its flags required a replay") }
                    try current.discard()
                    original = true
                } else {
                    // Only now may the previous state become committed.
                    // The next attention remains a separate transaction.
                    if !currentPublished { try current.publish(state: state) }
                }
                return original
            }
            for layerIndex in 0..<maximumLayers {
                try boundary()
                try autoreleasepool {
                    let s = try schedule.decodeTile(generation)
                    var original = false
                    if pending == nil { original = try encodeFirstRouter(layerIndex, s) }
                    // Pre-encoded layers: this layer's experts are queued behind its attention; the router signal makes its flags final.
                    let preencoded = schedule.armedExperts(tile: generation)
                    if preencoded != nil, let pending {
                        guard let batch = routerSignal, sharedPreencoded else {
                            throw EngineError("Pre-encoded decode layer lacks its router signal or shared expert")
                        }
                        routerSignal = nil
                        try waitForRouter(batch, command: pending.command)
                        pending.markRouterSignaled()
                    }
                    if let pending, try pending.requiresReplay() {
                        if preencoded != nil {
                            try schedule.joinAll()
                            execution.releaseForGPU(layer: nil)
                        }
                        stagedDemand?.close()
                        stagedDemand = nil
                        closePrerouted()
                        try pending.discard()
                        original = true
                    }
                    if !original { original = try runRoutedLayer(layerIndex, generation, s, preencoded) }
                    if original {
                        guard injectedAt != layerIndex else {
                            throw EngineError("Delayed attention requires an unsupported numerical replay", code: "delayed_replay")
                        }
                        try weights.enterLayer(layerIndex)
                        if layerIndex % 4 == 3 { try math.retireInvalidWholeMirror(layer: layerIndex, requestID: state.id) }
                        let values = try wholeOriginalAttention(schedule.read(s[.residual], count: 10240), layer: layerIndex, state: state)
                        let result = try originalDecodeMoE(values, layer: layerIndex)
                        try schedule.copy(result, to: s[.result])
                    }
                    pending = nil
                    sharedPreencoded = nextSharedPreencoded
                    nextSharedPreencoded = false
                    schedule.layerCompleted(replayed: original)
                    if decodeTokenPlan { schedule.tokenPlanLayer(replayed: original) }
                    progressObserver?(state.tokenCount, 1, layerIndex + 1)
                    if layerIndex + 1 < maximumLayers {
                        if let ready = lookahead {
                            pending = ready
                            lookahead = nil
                        } else {
                            let next = try schedule.decodeTile(1 - generation)
                            // This path is only first/PLE/repair/cache-bypass;
                            // both source and destination have no live GPU user.
                            try schedule.copy(schedule.read(s[.result], count: 10240), to: next[.residual])
                        }
                        generation = 1 - generation
                    }
                }
            }
            guard delayedSource == nil, delayedRead?.isPending != true else {
                throw EngineError("Delayed Qwen contribution escaped its token", code: "delayed_deadline")
            }
            let tailStart = DispatchTime.now().uptimeNanoseconds
            defer { schedule.recordDecodeWait(tail: Double(DispatchTime.now().uptimeNanoseconds - tailStart) / 1e9) }
            return try finishWholeDecodeToken(
                generation: generation, maximumLayers: maximumLayers, returnLogits: returnLogits, history: history, state: state,
                schedule: schedule)
        } catch {
            state.valid = false
            throw error
        }
    }

    /// Pre-encoded layers: the router signal is on the GPU's critical path, so spin (a blocked wait wakes about 30 µs later), and
    /// block in short slices only after 20 ms (a stalled GPU).
    private func waitForRouter(_ signal: (event: MTLSharedEvent, value: UInt64), command: MTLCommandBuffer) throws {
        let start = DispatchTime.now().uptimeNanoseconds
        defer { wholeSchedule?.recordDecodeWait(router: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9) }
        while signal.event.signaledValue < signal.value {
            if command.status == .error { throw EngineError.gpu("Decode attention failed before its router signal", command.error) }
            if DispatchTime.now().uptimeNanoseconds - start > 20_000_000 {
                _ = signal.event.wait(untilSignaledValue: signal.value, timeoutMS: 5)
                guard shouldContinue() else { throw IOOperationCancelled() }
                // The signal itself is waited for, never the whole buffer: later work in it may wait for this host.
                // One that never comes within a wait's bound and `settle` more takes the GPU for lost (`GPUWait`).
                let limit = GPUWait.bound(work: 0, budget: nil) * (1 + GPUWait.settle)
                if GPUWait.lost || Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 > limit {
                    throw GPUWait.declareLost(String(format: "The decode router signal did not come in %.0f s", limit))
                }
            }
        }
    }

    /// The decode token's 16 device-PLE rows, read while layers 0 and 1 are prepared.
    private func decodePLERows(history: [Int64], prerouting: DecodePrerouting?, execution: ExpertExecution) throws
        -> PLERowPrefetch
    {
        let store = weights.store
        // Read hygiene: no predicted read starts until this token's rows are read (nil when off).
        let hold = prerouting == nil ? nil : execution.holdPredictedReads()
        // Internal race: only while the packed PLE source is open (otherwise both would read originals).
        var race: (pool: SpanReadPool, read: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void)?
        if let racePool = decodePLERacePool {
            if store.packedPLEAvailable { race = (racePool, { try store.readPLERowOriginalRace($0, into: $1) }) }
        }
        return try PLERowPrefetch(
            tokenRows: ple.rowIndices(history: history, newCount: 1), ledger: ledger,
            workers: 16, pool: decodeRowPool, hold: hold, race: race
        ) { try store.readPLERow($0, into: $1) }
    }

    /// Quality dial (exact: unchanged): the routed ranks this token keeps, and their renormalised gates.
    private func applyQualityDial(
        layerIndex: Int, routed: [Int], gates: inout [Float], kept: inout [Int], s: WholeExecutionWorkspace,
        schedule: WholeExecutionSchedule, execution: ExpertExecution
    ) throws {
        if !requestDial.isExact {
            if requestDial.dropsExperts {
                let resident: Set<Int> =
                    try requestDial.tailSkipMissesOnly ? Set(execution.residentExperts(layer: layerIndex)) : []
                let ranks = requestDial.keptRanks(gates) { resident.contains(routed[$0]) }
                if ranks.count < 10 {
                    gates = QualityDial.gates(gates, kept: ranks)
                    try schedule.copy(gates, to: s[.routerWeights])
                    kept = ranks.map { routed[$0] }.sorted()
                }
            }
        }
    }

    /// Fast late-miss skip: the late records it skips (gate 0, kept gates rescaled), and with `substitute` the
    /// resident experts computed instead.
    private func fastLateMissSkip(
        tau: Double, gates: [Float], layerIndex: Int, routed: [Int], s: WholeExecutionWorkspace, schedule: WholeExecutionSchedule,
        execution: ExpertExecution
    ) -> (skipLate: ([Int], Int) throws -> [Int], substitute: (([Int]) throws -> [Int: Int])?) {
        let gates = gates
        let layer = layerIndex
        let gateBuffer = s[.routerWeights]
        let logitBuffer = s[.routerLogits]
        let substitutes = requestDial.lateMiss == .substitute
        let skipLate: ([Int], Int) throws -> [Int] = { late, capacity in
            let decision = try FastLateMissSkip.decide(
                ids: routed, weights: gates, late: late, tau: tau, capacity: capacity)
            if !decision.skipped.isEmpty && !substitutes { try schedule.copy(decision.weights, to: gateBuffer) }
            return decision.skipped
        }
        var substitute: (([Int]) throws -> [Int: Int])?
        if substitutes {
            substitute = { skipped in
                let ranks = skipped.map { routed.firstIndex(of: $0)! }
                let logits = try schedule.read(logitBuffer, count: 512)
                let chosen = QualityDial.substitutes(
                    late: ranks, weights: gates, routed: routed, logits: logits,
                    resident: try execution.residentExperts(layer: layer))
                let active = gates.indices.map { rank -> Float in
                    guard ranks.contains(rank) else { return gates[rank] }
                    return chosen[rank].map {
                        QualityDial.substituteWeight(
                            gates[rank], replacedLogit: logits[routed[rank]],
                            substituteLogit: logits[$0])
                    } ?? 0
                }
                try schedule.copy(QualityDial.gates(gates, active: active), to: gateBuffer)
                return Dictionary(uniqueKeysWithValues: chosen.map { (routed[$0.key], $0.value) })
            }
        }
        return (skipLate, substitute)
    }

    /// The decode token's end: its hidden row, logits and committed state.
    private func finishWholeDecodeToken(
        generation: Int, maximumLayers: Int, returnLogits: Bool, history: [Int64], state: SequenceState,
        schedule: WholeExecutionSchedule
    ) throws -> [Float] {
        let last = try schedule.decodeTile(generation)
        guard maximumLayers == 48 else {
            state.valid = false
            return try schedule.read(last[.result], count: 10240)
        }
        if captureHiddenRows > 0 { lastHidden = [try schedule.read(last[.result], count: 10240)] }
        var logits: [Float] = []
        if returnLogits {
            do {
                try weights.enterLayer(nil)
                let mixed = try math.hyperConnection(
                    residual: schedule.read(last[.result], count: 10240),
                    base: "model.hyper_connection_mixer", useCombine: false
                ).mixed
                logits = try weights.linear(mixed, name: "lm_head")
            }
        }
        if decodeTokenPlan { try weights.validateWholeTokenPlan() }
        state.history = Array(history.suffix(2))
        state.tokenCount += 1
        state.emittedButUnconsumed = nil
        try state.validate()
        try boundary()
        return logits
    }

    /// Joined original arithmetic for one exceptional tile. Expert I/O remains
    /// scope-major: this computes only attention/HC and the independent router
    /// and shared branch, then preserves the normal scope's expert grouping.
    private func wholeOriginalAttention(_ residual: [Float], layer: Int, state: SequenceState)
        throws -> (base: [Float], input: [Float], gates: [Float])
    {
        let base = "model.layers.\(layer)"
        let first = try math.hyperConnection(residual: residual, base: base + ".attn_hyper_connection")
        let attended: [Float]
        if layer % 4 == 3 {
            guard var value = state.qsa.removeValue(forKey: layer) else { throw EngineError("Missing original QSA replay state") }
            do {
                attended = try math.qsa(x: first.mixed, layer: layer, state: &value, requestID: state.id)
                state.qsa[layer] = value
            } catch {
                state.qsa[layer] = value
                throw error
            }
        } else {
            guard var value = state.gdn.removeValue(forKey: layer) else { throw EngineError("Missing original GDN replay state") }
            do {
                attended = try math.gdn(x: first.mixed, layer: layer, state: &value)
                state.gdn[layer] = value
            } catch {
                state.gdn[layer] = value
                throw error
            }
        }
        let hidden = try inject(attended, into: residual, gates: first.injection)
        let second = try math.hyperConnection(residual: hidden, base: base + ".mlp_hyper_connection")
        return (hidden, second.mixed, second.injection)
    }

    private func wholeOriginalRoutingShared(_ input: [Float], layer: Int)
        throws -> (routing: WholeLayerExecution.Routing, shared: [Float], gate: [Float])
    {
        guard let selector, let reductions, input.count % 2560 == 0 else { throw EngineError("Missing original whole replay primitives") }
        let rows = input.count / 2560
        let base = "model.layers.\(layer).mlp"
        var logits = try weights.linear(input, name: base + ".gate", outputBF16: false)
        BatchedTopIndices.maskColumns(&logits, columns: 512, eligible: ModelLayout.routedExperts)
        let ids = try selector.topIndices(scores: logits, columns: 512, count: 10).map { $0.map(Int.init) }
        var selected: [Float] = []
        selected.reserveCapacity(rows * 10)
        for row in 0..<rows {
            let values = ids[row].map { logits[row * 512 + $0] }
            let maximum = values.max()!
            let denominator = values.map { expf($0 - maximum) }.reduce(Float(0), +)
            guard denominator.isFinite, denominator > 0 else { throw EngineError("Nonfinite original whole router probabilities") }
            selected.append(contentsOf: values)
        }
        let probabilities = try reductions.softmax(rows: selected, width: 10)
        guard probabilities.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw EngineError("Nonfinite pinned whole replay probabilities") }
        let gate = try weights.linear(input, name: base + ".shared_expert.gate_proj")
        let up = try weights.linear(input, name: base + ".shared_expert.up_proj")
        let activation = zip(gate, up).map { LayerArithmetic.round(weights.kernels.arithmetic.silu($0) * $1) }
        let shared = try weights.linear(activation, name: base + ".shared_expert.down_proj")
        let outputGate = try weights.linear(input, name: base + ".shared_expert_gate").map(weights.kernels.arithmetic.sigmoid)
        return (.init(ids: ids, probabilities: probabilities), shared, outputGate)
    }

    /// A prefill scope's PLE rows (current scope or lookahead). With `prefill_ple_overlap` the packed rows race
    /// the original spans (alternate descending); without an open pack there is nothing to race (fallback).
    private func scopePLEPrefetch(_ tokenRows: [[Int]], store: PackStore) throws -> PLERowPrefetch {
        var race: (pool: SpanReadPool, read: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void)?
        if let racePool = prefillPLERacePool {
            if store.packedPLEAvailable { race = (racePool, { try store.readPLERowOriginalRace($0, into: $1) }) }
        }
        return try PLERowPrefetch(tokenRows: tokenRows, ledger: ledger, race: race, alternateDescending: race != nil) {
            try store.readPLERow($0, into: $1)
        }
    }

    /// `prefill_ple_overlap`: once the deferred rows are complete, join them (no wait) and start the expert
    /// read-ahead, so the experts are read as soon as the PLE rows no longer need the drives.
    private func pollDeferredPLEJoin() throws {
        guard let pending = deferredPLEJoin, pending.joinedWait == nil, pending.prefetch.readsComplete else { return }
        let waited = try joinDeferredPLE(pending.prefetch, layers: pending.layers, readAhead: pending.readAhead, tokens: pending.tokens)
        deferredPLEJoin?.joinedWait = waited
    }

    /// Join the deferred rows (waiting if still incomplete; the read-ahead starts then) and clear the state.
    /// Returns the exposed row wait of the scope, or nil when no join was deferred.
    private func finishDeferredPLEJoin() throws -> Double? {
        guard let pending = deferredPLEJoin else { return nil }
        deferredPLEJoin = nil
        if let waited = pending.joinedWait { return waited }
        return try joinDeferredPLE(pending.prefetch, layers: pending.layers, readAhead: pending.readAhead, tokens: pending.tokens)
    }

    private func joinDeferredPLE(_ prefetch: PLERowPrefetch, layers: Int, readAhead: Bool, tokens: Int) throws -> Double {
        let waited = try prefetch.wait(shouldContinue: shouldContinue)
        if readAhead {
            guard let execution = expertExecution else { throw EngineError("Whole prefill lost admitted expert dependencies") }
            try execution.beginPrefillReadAhead(layers: layers, rows: tokens)
        }
        return waited
    }

    /// Embedding rows of `ids`, read by concurrent workers from the same checkpoint spans as
    /// `ReferenceWeights.embedding`.
    private func makeEmbeddingPrefetch(_ ids: [Int]) throws -> EmbeddingRowPrefetch {
        let layout = try embeddingLayout ?? weights.embeddingRowLayout()
        embeddingLayout = layout
        let store = weights.store
        let words = layout.wordBytes
        let scales = layout.wordBytes + layout.scaleBytes
        return try EmbeddingRowPrefetch(ids: ids, layout: layout, ledger: ledger) { id, row in
            try store.readTensor("model.embed_tokens.weight", row: id, into: UnsafeMutableRawBufferPointer(rebasing: row[0..<words]))
            try store.readTensor("model.embed_tokens.scales", row: id, into: UnsafeMutableRawBufferPointer(rebasing: row[words..<scales]))
            try store.readTensor(
                "model.embed_tokens.biases", row: id, into: UnsafeMutableRawBufferPointer(rebasing: row[scales..<row.count]))
        }
    }

    private func forwardWholePrefill(
        _ ids: [Int], state: SequenceState, maximumLayers: Int, returnLogits: Bool,
        schedule: WholeExecutionSchedule, layout: WholePrefillScope.Layout,
        capture: ConversationScopeCapture?, computeTile: Int
    ) throws -> [Float] {
        guard let transfer = prefillMoETransfer, let execution = expertExecution else {
            throw EngineError("Whole prefill lost admitted expert dependencies")
        }
        try schedule.beginScope(decode: false)
        var pending: WholeLayerExecution.Pending?
        defer {
            try? schedule.joinAll()
            try? pending?.discard()
            pending = nil
            deferredPLEJoin = nil
            execution.finishPrefillReadAhead()
            schedule.endScope()
        }
        do {
            try boundary()
            // Both admissions precede state mutation; the scope itself keeps
            // device and host portions, including an explicitly admitted small-scope supplement, charged.
            let tileRows = computeTile
            let s = try schedule.prefillTile(rows: tileRows)
            // Both double-buffer workspaces are admitted before any tile of the scope is submitted.
            let doubleBuffered = prefillDoubleBuffer && tileRows == 256 && ids.count >= 512
            let tilePair = doubleBuffered ? try schedule.prefillTilePair() : [s]
            let scope = try WholePrefillScope(device: weights.kernels.device, ledger: ledger, layout: layout)
            // WholePrefillScope has already admitted this actual-row layout. Prompt
            // tokenization can leave a shorter whole scope (7,424 tokens for an
            // 8K prompt, say); it must not silently bypass configured layer work.
            let history = state.history + ids.map(Int64.init)
            let historyBase = state.history.count
            // Device PLE reads this scope's rows first and joins them before the expert read-ahead starts:
            // under concurrent multi-megabyte expert reads the cold 100-byte row reads ran at 8-16k rows/s,
            // against about 47k on an otherwise idle drive. Rows are hashed from two tokens
            // before the scope, which yields the same rows as each tile's window.
            let devicePrefill = devicePLE != nil && !ple.hasCapture && maximumLayers > 1
            // Layer 47's routed experts reach only the rows the reply reads: the logits row of the prompt's last scope
            // and MTP's hidden rows (a prompt cache saves state and those logits only). Their union runs with complete
            // row sets; the other rows' slots keep stale values nothing reads, and the read-ahead stops at layer 47.
            let keptRows = min(ids.count, max(returnLogits ? 1 : 0, captureHiddenRows))
            let skipFrom =
                maximumLayers == 48 && keptRows < ids.count && execution.configuration.prefillLayerCommand
                    && math.capture == nil ? ids.count - keptRows : nil
            let readAheadLayers = skipFrom == nil ? maximumLayers : 47
            var plePrefetch: PLERowPrefetch?
            var pleWorkspace: DevicePLE.Workspace?
            var pleProjections: DevicePLE.Projections?
            var pleFrontWait = 0.0
            let lookahead = pendingPLEPrefetch
            pendingPLEPrefetch = nil
            let nextScope = nextPrefillScope
            nextPrefillScope = nil
            // Embedding rows start before the PLE rows are joined, so both small-read sets
            // finish before the expert read-ahead begins.
            let hostPrefill = hostStagePrefetch && maximumLayers > 1
            var embeddingLookahead = pendingEmbeddingPrefetch
            pendingEmbeddingPrefetch = nil
            var embeddingPrefetch: EmbeddingRowPrefetch?
            if hostPrefill, let lookahead = embeddingLookahead, lookahead.ids == ids {
                embeddingPrefetch = lookahead.prefetch
            } else {
                // A lookahead for another scope (a wrong next-scope prediction, or a request without
                // host prefill) must neither keep reading the wrong rows beside this scope's small reads
                // nor hold its reservation while this scope reserves its own: dropping the last
                // reference cancels its readers, joins them and releases the reservation now.
                embeddingLookahead = nil
                if hostPrefill { embeddingPrefetch = try makeEmbeddingPrefetch(ids) }
            }
            if devicePrefill {
                let store = weights.store
                let begin = max(0, historyBase - 2)
                let prefetch: PLERowPrefetch
                if let lookahead, lookahead.prefix == state.history, lookahead.ids == ids {
                    prefetch = lookahead.prefetch
                    devicePLE?.recordLookahead()
                } else {
                    prefetch = try scopePLEPrefetch(
                        ple.rowIndices(
                            history: Array(history[begin..<(historyBase + ids.count)]),
                            newCount: ids.count), store: store)
                }
                if prefillPLERacePool != nil {
                    // Read behind layer 0; the expert read-ahead waits for these rows (below), not the GPU.
                    deferredPLEJoin = (prefetch, readAheadLayers, true, ids.count, nil)
                } else {
                    pleFrontWait = try prefetch.wait(shouldContinue: shouldContinue)
                }
                plePrefetch = prefetch
            }
            if let embeddingPrefetch { _ = try embeddingPrefetch.wait(shouldContinue: shouldContinue) }
            if deferredPLEJoin == nil { try execution.beginPrefillReadAhead(layers: readAheadLayers, rows: ids.count) }
            // Dense layer weights: the first request reads them into the resident caches; warm
            // them at most two layers ahead so the reads overlap earlier layers' GPU work.
            let densePrefetch: DenseLayerPrefetch?
            if hostPrefill && !weights.hasWholeTokenPlan {
                let owner = HostStageWeights(weights)
                densePrefetch = try DenseLayerPrefetch(layers: maximumLayers, distance: 2) { try owner.weights.prefetchWholeLayer($0) }
            } else {
                densePrefetch = nil
            }
            defer { _ = densePrefetch?.finish() }
            try weights.enterLayer(nil)
            try autoreleasepool {
                var embedding = try embeddingPrefetch?.embedding() ?? weights.embedding(ids)
                try state.images?.inject(into: &embedding, start: state.tokenCount)
                try scope.stageEmbeddings(embedding)
                schedule.uploaded(ids.count * 10240 * 4)
            }
            embeddingPrefetch = nil
            for layerIndex in 0..<maximumLayers {
                try boundary()
                let skipping = layerIndex == 47 ? skipFrom : nil
                let replaysBefore = schedule.prefillReplays
                densePrefetch?.enter(layerIndex)
                try weights.enterLayer(layerIndex)
                try autoreleasepool {
                    let owner = try weights.acquireWholeLayerWeights(layer: layerIndex)
                    var mirror: QSADecodeStorage?
                    var paged: QSAPagedAttentionStorage?
                    var selectedRows: [Int: [(Int, Int)]] = [:]
                    try pollDeferredPLEJoin()
                    if layerIndex == 1, let devicePLE, let prefetch = plePrefetch {
                        pleProjections = try weights.devicePLEProjections()
                        if pleProjections != nil {
                            pleWorkspace = try devicePLE.makeWorkspace(rows: tileRows)
                            let waited = try prefetch.wait(shouldContinue: shouldContinue)
                            devicePLE.recordRows(
                                requested: ids.count * 16, read: prefetch.uniqueRows, waitSeconds: pleFrontWait + waited, decode: false)
                        }
                    }
                    // Double-buffered GDN layers (never layer 1 or a QSA layer): the full tiles run
                    // through the two-workspace pipeline; a short tail keeps the sequential path below.
                    let chainedTiles = doubleBuffered && owner != nil && layerIndex != 1 && layerIndex % 4 != 3 ? ids.count / tileRows : 0
                    if chainedTiles > 0 {
                        try doubleBufferedAttentionTiles(
                            ids: ids, layerIndex: layerIndex, tileRows: tileRows, tiles: chainedTiles, owner: owner!,
                            state: state, schedule: schedule, scope: scope, workspaces: tilePair, execution: execution,
                            capture: capture, selectedRows: &selectedRows)
                    }
                    for start in stride(from: chainedTiles * tileRows, to: ids.count, by: tileRows) {
                        let rows = min(tileRows, ids.count - start)
                        try boundary()
                        try pollDeferredPLEJoin()
                        try autoreleasepool {
                            s.clearFlags()
                            var pleResidual: [Float]?
                            let deviceTile = layerIndex == 1 && owner != nil && rows == tileRows && pleWorkspace != nil
                            if layerIndex == 1 && !deviceTile {
                                if devicePrefill { devicePLE?.recordFallback() }
                                let old = try schedule.read(scope[.residual], count: rows * 10240, offset: start * 10240)
                                let begin = max(0, historyBase + start - 2)
                                // prefill_ple_overlap: the scope already read this tail's rows; take those bytes.
                                var scopeRows: PLEScopeRowSource?
                                if prefillPLERacePool != nil, devicePrefill, let prefetch = plePrefetch, prefetch.isJoined {
                                    let store = weights.store
                                    scopeRows = try PLEScopeRowSource(prefetch: prefetch) { try store.readPLERow($0, into: $1) }
                                }
                                let extra = try ple.call(
                                    old, history: Array(history[begin..<(historyBase + start + rows)]), newCount: rows, state: &state.ple,
                                    knownRows: true, shouldContinue: shouldContinue,
                                    rowSource: scopeRows?.reader)
                                pleResidual = zip(old, extra).map { LayerArithmetic.round($0 + $1) }
                            }
                            do {
                                var routing: WholeLayerExecution.Routing?
                                if let owner, rows == tileRows {
                                    var command = try schedule.command("whole-prefill-attention-router-shared")
                                    if let pleResidual {
                                        try schedule.copy(pleResidual, to: s[.residual])
                                    } else {
                                        try scope.encodeTileInput(command, start: start, workspace: s)
                                    }
                                    if deviceTile, let devicePLE, let workspace = pleWorkspace, let projections = pleProjections,
                                        let prefetch = plePrefetch
                                    {
                                        let packed = try prefetch.packed(tokens: start..<(start + rows))
                                        try packed.withUnsafeBytes {
                                            try devicePLE.encode(
                                                command, residual: s[.residual], rows: rows, packedRows: $0,
                                                state: state.ple.convolution, projections: projections, workspace: workspace, decode: false)
                                        }
                                    }
                                    if layerIndex % 4 == 3, mirror == nil, paged == nil {
                                        if state.pageArena != nil {
                                            paged = try schedule.attention.makePagedStorage(capacityTokens: state.capacity)
                                        } else {
                                            mirror = try schedule.attention.makeDecodeStorage(capacityTokens: state.capacity)
                                        }
                                    }
                                    pending = try schedule.layer.encodeAttentionRouter(
                                        &command, residual: s[.residual], weights: owner,
                                        state: state, mirror: mirror, paged: paged, workspace: s,
                                        joinSelection: { try schedule.joinedFileSelection($0, shouldContinue: self.shouldContinue) })
                                    try schedule.layer.encodeShared(command, weights: owner, workspace: s)
                                    try scope.encodeTileOutputs(command, start: start, workspace: s)
                                    try schedule.joined(command, shouldContinue: shouldContinue)
                                    // The device PLE result is used by both the encoded and the replay path.
                                    if deviceTile, let devicePLE, let workspace = pleWorkspace {
                                        state.ple.convolution = devicePLE.nextState(workspace)
                                    }
                                    if try !pending!.requiresReplay(), !schedule.flagged(s, [.sharedFlags]) {
                                        routing = try pending!.routing()
                                        try pending!.publish(state: state)
                                    } else {
                                        try pending!.discard()
                                        mirror = nil
                                        paged = nil
                                    }
                                    pending = nil
                                }
                                if routing == nil {
                                    // Preserve scope-wide expert routing even when
                                    // one tile or a resident-weight group bypasses.
                                    let original =
                                        try pleResidual
                                        ?? (owner == nil || rows != tileRows
                                            ? schedule.read(scope[.residual], count: rows * 10240, offset: start * 10240)
                                            : schedule.read(s[.residual], count: rows * 10240))
                                    let values = try wholeOriginalAttention(original, layer: layerIndex, state: state)
                                    let replay = try wholeOriginalRoutingShared(values.input, layer: layerIndex)
                                    for (role, values, width) in [
                                        (WholePrefillScope.Slot.residual, values.base, 10240),
                                        (.input, values.input, 2560), (.injection, values.gates, 4),
                                        (.routingWeights, replay.routing.probabilities, 10),
                                        (.shared, replay.shared, 2560), (.sharedGate, replay.gate, 1),
                                    ] {
                                        try schedule.copy(values, to: scope[role], offset: start * width * 4)
                                    }
                                    routing = replay.routing
                                    if rows == tileRows { schedule.replay() } else { schedule.recordMixedTail(rows: rows) }
                                }
                                for (row, selected) in routing!.ids.enumerated() {
                                    for (rank, expert) in selected.enumerated() {
                                        selectedRows[expert, default: []].append((start + row, rank))
                                    }
                                }
                            }
                            if let capture { try capture.retain(layer: layerIndex, offset: state.tokenCount + start + rows, state: state) }
                        }
                    }
                    mirror = nil
                    paged = nil
                    // prefill_ple_overlap: layer 0's MoE needs the expert read-ahead, so the rows join here at the latest.
                    if let pleWait = try finishDeferredPLEJoin() { pleFrontWait = pleWait }
                    if layerIndex == 1 {
                        pleWorkspace = nil
                        pleProjections = nil
                        plePrefetch = nil
                        // This scope's rows are consumed: read the next scope's rows behind layers 2-47.
                        if devicePrefill, let next = nextScope, !next.isEmpty {
                            let store = weights.store
                            let nextBase = historyBase + ids.count
                            let extended = Array(history[max(0, nextBase - 2)..<nextBase]) + next.map(Int64.init)
                            pendingPLEPrefetch = (
                                prefix: Array(history.suffix(2)), ids: next,
                                prefetch: try scopePLEPrefetch(ple.rowIndices(history: extended, newCount: next.count), store: store)
                            )
                        }
                        // The next scope's embedding rows depend only on its tokens.
                        if hostPrefill, let next = nextScope, !next.isEmpty {
                            pendingEmbeddingPrefetch = (ids: next, prefetch: try makeEmbeddingPrefetch(next))
                        }
                    }
                    let layerRows = skipping.map { Self.keptExperts(selectedRows, from: $0) } ?? selectedRows
                    let deviceWorkspace = try PrefillMoEWorkspace(
                        tokens: ids.count, input: scope[.input], selectedRows: layerRows,
                        budget: scope.budget, transfer: transfer, stale: skipping != nil)
                    if !layerRows.isEmpty {
                        let window = try execution.makeWindow(
                            layer: layerIndex, experts: Array(layerRows.keys), shouldContinue: shouldContinue)
                        defer { window.close() }
                        var unused: [Float] = []
                        if execution.configuration.prefillLayerCommand && execution.usesSelectivePrefill {
                            try window.consumePrefillReady { experts, buffers in
                                let rows = Dictionary(uniqueKeysWithValues: experts.map { ($0, layerRows[$0]!) })
                                try weights.expertDeviceLayer(
                                    experts: experts, buffers: buffers,
                                    selectedRows: rows, workspace: deviceWorkspace,
                                    ungroupedTailStart: tileRows == 256 && ids.count % 256 != 0 ? ids.count / 256 * 256 : nil,
                                    waves: execution.configuration.layerWaves, groupedExpert: tileRows == 256,
                                    partial: true, shouldContinue: shouldContinue)
                            }
                        } else if execution.configuration.prefillLayerCommand && tileRows == 256 {
                            try window.consumeLayer { buffers in
                                try weights.expertDeviceLayer(
                                    experts: window.keys.map(\.expert), buffers: buffers,
                                    selectedRows: layerRows, workspace: deviceWorkspace,
                                    ungroupedTailStart: ids.count % 256 == 0 ? nil : ids.count / 256 * 256,
                                    waves: execution.configuration.layerWaves,
                                    shouldContinue: shouldContinue)
                            }
                        } else {
                            // C<256 historically uses one QMV per token/expert, even when its input/output
                            // now stay on the GPU. Preserve that schedule and canonical rank scatter.
                            try batchedExperts(
                                [], selectedRows: layerRows, window: window, grouped: tileRows == 256,
                                contributions: &unused, deviceWorkspace: deviceWorkspace)
                        }
                    }
                    // The original complete canonical reduction uses the same
                    // input layout. Its command also mixes the first tile.
                    do {
                        let deferredTiles = doubleBuffered ? ids.count / tileRows : 0
                        if deferredTiles > 0 {
                            try doubleBufferedMixTiles(
                                ids: ids, tileRows: tileRows, tiles: deferredTiles, schedule: schedule,
                                scope: scope, workspaces: tilePair, deviceWorkspace: deviceWorkspace)
                        }
                        for start in stride(from: deferredTiles * tileRows, to: ids.count, by: tileRows) {
                            let rows = min(tileRows, ids.count - start)
                            try boundary()
                            try autoreleasepool {
                                if rows < tileRows {
                                    // The first full tile already joined the canonical reduction
                                    // for every valid row. Preserve the original short-tile mix.
                                    let routed = try schedule.read(scope[.routed], count: rows * 2560, offset: start * 2560)
                                    let shared = try schedule.read(scope[.shared], count: rows * 2560, offset: start * 2560)
                                    let gate = try schedule.read(scope[.sharedGate], count: rows, offset: start)
                                    let mixed = routed.indices.map { i in
                                        LayerArithmetic.round(
                                            LayerArithmetic.round(routed[i]) + LayerArithmetic.round(shared[i] * gate[i / 2560]))
                                    }
                                    let result = try inject(
                                        mixed,
                                        into: schedule.read(scope[.residual], count: rows * 10240, offset: start * 10240),
                                        gates: schedule.read(scope[.injection], count: rows * 4, offset: start * 4))
                                    try schedule.copy(result, to: scope[.residual], offset: start * 10240 * 4)
                                    return
                                }
                                s[.combinationFlags].contents().storeBytes(of: UInt32(0), as: UInt32.self)
                                s[.finalInjectionFlags].contents().storeBytes(of: UInt32(0), as: UInt32.self)
                                let command = try schedule.command("whole-prefill-canonical-mix")
                                if start == 0 {
                                    try deviceWorkspace.encodeReduction(
                                        command, weights: scope[.routingWeights], output: scope[.routed], using: schedule.reductions)
                                }
                                try scope.encodeMixInput(command, start: start, workspace: s)
                                try schedule.layer.encodeMix(command, workspace: s)
                                try schedule.joined(command, shouldContinue: shouldContinue)
                                if start == 0 { try deviceWorkspace.completeEncodedReduction(command) }
                                if schedule.flagged(s, [.combinationFlags, .finalInjectionFlags]) {
                                    let routed = try schedule.read(s[.routed], count: rows * 2560)
                                    let shared = try schedule.read(s[.sharedOutput], count: rows * 2560)
                                    let gate = try schedule.read(s[.sharedSigmoid], count: rows)
                                    let mixed = routed.indices.map { i in
                                        LayerArithmetic.round(
                                            LayerArithmetic.round(routed[i]) + LayerArithmetic.round(shared[i] * gate[i / 2560]))
                                    }
                                    let result = try inject(
                                        mixed, into: schedule.read(s[.afterAttention], count: rows * 10240),
                                        gates: schedule.read(s[.hcMInjection], count: rows * 4))
                                    try schedule.copy(result, to: s[.result])
                                    schedule.replay()
                                }
                                guard scope.publishMix(start: start, workspace: s, command: command) else {
                                    throw EngineError("Whole scope mix publication is unjoined or outside bounds")
                                }
                            }
                        }
                    }
                }
                try execution.finishPrefillReadAheadLayer(layerIndex)
                schedule.layerCompleted(replayed: false)
                if skipping != nil { schedule.skippedLastLayerRows(replays: schedule.prefillReplays - replaysBefore) }
                progressObserver?(state.tokenCount, ids.count, layerIndex + 1)
            }
            execution.finishPrefillReadAhead()
            guard maximumLayers == 48 else {
                state.valid = false
                return try schedule.read(scope[.residual], count: ids.count * 10240)
            }
            if captureHiddenRows > 0 {
                let keep = min(captureHiddenRows, ids.count)
                lastHidden = Self.lastRows(
                    try schedule.read(scope[.residual], count: keep * 10240, offset: (ids.count - keep) * 10240), count: keep, keep: keep)
            }
            var logits: [Float] = []
            if returnLogits {
                do {
                    try weights.enterLayer(nil)
                    // Only the last tile's last row reaches the LM head; under the skip the earlier rows are stale.
                    for start in stride(from: skipFrom == nil ? 0 : (ids.count - 1) / tileRows * tileRows, to: ids.count, by: tileRows) {
                        let rows = min(tileRows, ids.count - start)
                        let tile = try schedule.read(scope[.residual], count: rows * 10240, offset: start * 10240)
                        let mixed = try math.hyperConnection(residual: tile, base: "model.hyper_connection_mixer", useCombine: false).mixed
                        if start + rows == ids.count { logits = try weights.linear(Array(mixed.suffix(2560)), name: "lm_head") }
                    }
                }
            }
            state.history = Array(history.suffix(2))
            state.tokenCount += ids.count
            state.emittedButUnconsumed = nil
            try state.validate()
            try boundary()
            return logits
        } catch {
            state.valid = false
            throw error
        }
    }

    /// `prefill_double_buffer`, attention phase of one GDN layer other than layer 1: the full tiles
    /// `0..<tiles` alternate between the two workspaces. Tile t+1 is encoded from tile t's
    /// provisional GPU state (`WholeLayerExecution.GDNChain`) and submitted before tile t joins;
    /// tile t's join, flag check, routing readback, state publication and frontier capture then
    /// overlap the GPU. A workspace is reused only after the schedule joined the command that read
    /// it. A flagged tile joins and discards its successor, replays on the host exactly as the
    /// sequential path does, and the successor is re-issued from the published state. Layer 1
    /// (PLE state chains on the host), QSA layers (KV publication chains on the host) and a short
    /// tail keep the sequential path; the pipeline adopts routing in tile order.
    private func doubleBufferedAttentionTiles(
        ids: [Int], layerIndex: Int, tileRows: Int, tiles: Int, owner: WholeLayerWeights,
        state: SequenceState, schedule: WholeExecutionSchedule, scope: WholePrefillScope,
        workspaces: [WholeExecutionWorkspace], execution: ExpertExecution,
        capture: ConversationScopeCapture?,
        selectedRows: inout [Int: [(Int, Int)]]
    ) throws {
        struct Ticket {
            let start: Int, workspace: WholeExecutionWorkspace, command: MTLCommandBuffer, pending: WholeLayerExecution.Pending
        }
        guard workspaces.count == 2, workspaces[0] !== workspaces[1], tileRows == 256, tiles >= 1, tiles * tileRows <= ids.count,
            layerIndex != 1, layerIndex % 4 != 3, owner.layer == layerIndex, owner.gdn != nil, owner.qsa == nil
        else {
            throw EngineError("Double-buffered attention tiles require two workspaces on a chained GDN layer")
        }
        var live: [WholeLayerExecution.Pending] = []
        defer {
            // Error exit only (the normal path drains): join before any unjoined writer is discarded.
            if !live.isEmpty {
                try? schedule.joinAll()
                for pending in live { try? pending.discard() }
            }
        }
        func adopt(_ routing: WholeLayerExecution.Routing, start: Int) throws {
            for (row, selected) in routing.ids.enumerated() {
                for (rank, expert) in selected.enumerated() { selectedRows[expert, default: []].append((start + row, rank)) }
            }
        }
        func retain(start: Int) throws {
            if let capture { try capture.retain(layer: layerIndex, offset: state.tokenCount + start + tileRows, state: state) }
        }
        _ = try PrefillTilePipeline.run(
            tiles: tiles,
            issue: { tile, slot, predecessor -> Ticket in
                let start = tile * tileRows
                let w = workspaces[slot]
                try boundary()
                try pollDeferredPLEJoin()
                return try autoreleasepool { () throws -> Ticket in
                    w.clearFlags()
                    var command = try schedule.command("whole-prefill-attention-router-shared")
                    try scope.encodeTileInput(command, start: start, workspace: w)
                    var chain: WholeLayerExecution.GDNChain?
                    if let predecessor {
                        guard var chained = predecessor.pending.oldGDN, predecessor.workspace !== w else {
                            throw EngineError("Whole GDN chain lost its predecessor state")
                        }
                        chained.offset += tileRows
                        chain = .init(previous: predecessor.workspace, state: chained)
                    }
                    let pending = try schedule.layer.encodeAttentionRouter(
                        &command, residual: w[.residual], weights: owner,
                        state: state, mirror: nil, paged: nil, workspace: w, gdnChain: chain)
                    live.append(pending)
                    try schedule.layer.encodeShared(command, weights: owner, workspace: w)
                    try scope.encodeTileOutputs(command, start: start, workspace: w)
                    try schedule.submit(command, shouldContinue: shouldContinue)
                    return Ticket(start: start, workspace: w, command: command, pending: pending)
                }
            },
            finish: { _, ticket -> Bool in
                let routing: WholeLayerExecution.Routing? = try autoreleasepool { () throws -> WholeLayerExecution.Routing? in
                    defer { live.removeAll { $0 === ticket.pending } }
                    try schedule.join(ticket.command)
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    if try !ticket.pending.requiresReplay(), !schedule.flagged(ticket.workspace, [.sharedFlags]) {
                        let routing = try ticket.pending.routing()
                        try ticket.pending.publish(state: state)
                        return routing
                    }
                    try ticket.pending.discard()
                    return nil
                }
                guard let routing else { return false }
                try adopt(routing, start: ticket.start)
                try retain(start: ticket.start)
                return true
            },
            discard: { _, ticket in
                // Joined before the host replay writes the scope or state; a failed command still reports here.
                defer { live.removeAll { $0 === ticket.pending } }
                try schedule.join(ticket.command)
                try ticket.pending.discard()
                // Its command overwrote its own scope input rows; the re-issue reads them again.
                guard scope.restoreTileInput(start: ticket.start, workspace: ticket.workspace, command: ticket.command) else {
                    throw EngineError("Discarded whole tile input restore is unjoined or outside bounds")
                }
            },
            replay: { tile in
                let start = tile * tileRows
                let w = workspaces[tile % 2]
                let routing: WholeLayerExecution.Routing = try autoreleasepool { () throws -> WholeLayerExecution.Routing in
                    let original = try schedule.read(w[.residual], count: tileRows * 10240)
                    let values = try wholeOriginalAttention(original, layer: layerIndex, state: state)
                    let replay = try wholeOriginalRoutingShared(values.input, layer: layerIndex)
                    for (role, values, width) in [
                        (WholePrefillScope.Slot.residual, values.base, 10240),
                        (.input, values.input, 2560), (.injection, values.gates, 4), (.routingWeights, replay.routing.probabilities, 10),
                        (.shared, replay.shared, 2560), (.sharedGate, replay.gate, 1),
                    ] {
                        try schedule.copy(values, to: scope[role], offset: start * width * 4)
                    }
                    schedule.replay()
                    return replay.routing
                }
                try adopt(routing, start: start)
                try retain(start: start)
            })
    }

    /// `prefill_double_buffer`, canonical mix of one layer: the full tiles alternate between the two
    /// workspaces and tile t+1 is submitted before tile t joins, so tile t's reduction completion,
    /// flag check and host publication overlap the GPU. A mix tile reads only its own scope rows,
    /// so an exceptional tile is repaired in place exactly as on the sequential path and never
    /// invalidates its successor.
    private func doubleBufferedMixTiles(
        ids: [Int], tileRows: Int, tiles: Int, schedule: WholeExecutionSchedule,
        scope: WholePrefillScope, workspaces: [WholeExecutionWorkspace],
        deviceWorkspace: PrefillMoEWorkspace
    ) throws {
        struct Ticket { let start: Int, workspace: WholeExecutionWorkspace, command: MTLCommandBuffer }
        guard workspaces.count == 2, workspaces[0] !== workspaces[1], tileRows == 256, tiles >= 1, tiles * tileRows <= ids.count else {
            throw EngineError("Double-buffered mix tiles require two C256 workspaces")
        }
        _ = try PrefillTilePipeline.run(
            tiles: tiles,
            issue: { tile, slot, _ -> Ticket in
                let start = tile * tileRows
                let w = workspaces[slot]
                try boundary()
                return try autoreleasepool { () throws -> Ticket in
                    w[.combinationFlags].contents().storeBytes(of: UInt32(0), as: UInt32.self)
                    w[.finalInjectionFlags].contents().storeBytes(of: UInt32(0), as: UInt32.self)
                    let command = try schedule.command("whole-prefill-canonical-mix")
                    if start == 0 {
                        try deviceWorkspace.encodeReduction(
                            command, weights: scope[.routingWeights], output: scope[.routed], using: schedule.reductions)
                    }
                    try scope.encodeMixInput(command, start: start, workspace: w)
                    try schedule.layer.encodeMix(command, workspace: w)
                    try schedule.submit(command, shouldContinue: shouldContinue)
                    return Ticket(start: start, workspace: w, command: command)
                }
            },
            finish: { _, ticket -> Bool in
                try autoreleasepool { () throws -> Bool in
                    let w = ticket.workspace
                    let start = ticket.start
                    let rows = tileRows
                    try schedule.join(ticket.command)
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    if start == 0 { try deviceWorkspace.completeEncodedReduction(ticket.command) }
                    if schedule.flagged(w, [.combinationFlags, .finalInjectionFlags]) {
                        let routed = try schedule.read(w[.routed], count: rows * 2560)
                        let shared = try schedule.read(w[.sharedOutput], count: rows * 2560)
                        let gate = try schedule.read(w[.sharedSigmoid], count: rows)
                        let mixed = routed.indices.map { i in
                            LayerArithmetic.round(LayerArithmetic.round(routed[i]) + LayerArithmetic.round(shared[i] * gate[i / 2560]))
                        }
                        let result = try inject(
                            mixed, into: schedule.read(w[.afterAttention], count: rows * 10240),
                            gates: schedule.read(w[.hcMInjection], count: rows * 4))
                        try schedule.copy(result, to: w[.result])
                        schedule.replay()
                    }
                    guard scope.publishMix(start: start, workspace: w, command: ticket.command) else {
                        throw EngineError("Whole scope mix publication is unjoined or outside bounds")
                    }
                    return true
                }
            }, discard: { _, _ in throw EngineError("Mix tiles never discard a successor") },
            replay: { _ in throw EngineError("Mix tiles never replay through the pipeline") })
    }
}
