import CoreFoundation
import Foundation

/// Every execution switch of one engine instance. Origami builds this from its defaults
/// (`OrigamiPlan`); a JSON file can override switches for debugging.
public struct RuntimeConfiguration: Sendable {
    /// Largest process budget plus its transient margin; the governor works below it.
    public var processCeiling: UInt64 = 22_000_000_000
    public var transientMargin: UInt64 = 1_000_000_000
    /// Budget at start when lower than the maximum (a busy Mac); nil starts at the maximum.
    public var startUsableBytes: UInt64?
    /// Explicit budgets target the chosen cap; automatic budgets fit available-memory estimates.
    public var memoryBudgetMode: MemoryBudgetMode = .automatic
    /// Host memory the engine leaves available to the rest of the Mac.
    public var availableFloorBytes: UInt64 = 3 * 1024 * 1024 * 1024
    public var contextLimit = 32_768
    /// The 262,144 tier with every key/value page in memory instead of a file (planned where the Mac has room).
    public var contextInMemory = false
    public var fileBackedContext: Bool { contextLimit == ConversationGeometry.maximumFileContext && !contextInMemory }
    public var computeTile = 256
    public var readScope = 256
    public var pinnedRMS = false
    public var batchedSelector = false
    public var pinnedReductions = false
    public var pinnedAttention = false
    public var qsaDecodeReuse = false
    public var batchedPinnedAttention = false
    public var fusedSelectedPrefillAttention = false
    /// Grouped compact PV (one threadgroup per query pair and KV head, sharing V tile loads; bit-identical to the
    /// per-head-quad PV); requires fused_selected_prefill_attention.
    public var compactPVGrouped = false
    /// Exact QSA prefill kernels (double-buffered compact QK, direct register-fragment PV on both
    /// prefill paths); requires batched pinned attention. Path proof: qsaExactPVTiles > 0.
    public var qsaExactKernels = false
    /// Exact hc_ple fusion. Tile HCs run one fused narrow + grouped RMS + narrow entry instead of
    /// three dispatches, and encoded pinned projections with 2-31 split-K partitions reduce with one thread per
    /// output (the current lane order replayed); 32+ partitions keep the current reducer (counted). Requires
    /// fused_hc_prefill and simd_matrix. Path proofs: hcFusedEntryTiles > 0, serialSplitReduces > 0.
    public var hcExactFusion = false
    /// Exact fused row-one (decode) hyper-connection. Each row HC runs four dispatches instead of
    /// ten (fused narrow + grouped RMS + narrow entry; QMV down + activation lookup; QMV up + sigmoid/sum4 epilogue
    /// + mixed narrowing; injection GEMV + lookup), every stored word unchanged. Requires fused_hc_decode.
    /// Path proofs: hcDecodeFusedRows > 0, hcDecodeFusedEntries > 0; hcDecodeFusionFallbacks == 0.
    public var hcDecodeFusion = false
    public var batchedRoPE = false
    public var pinnedRoPE = false
    public var pinnedIndexer = false
    public var gpuKeepAlive = false
    public var registerQSADecode = false
    public var pinnedDense = false
    public var denseResidentBytes: UInt64 = 0
    public var persistentConversation = false
    public var conversationFrontierDiskBytes: UInt64 = 0
    /// Append and hash frontier recurrent images on a background writer (prefix_cache.frontier_async_writes).
    /// Same bytes in the same per-file order as the synchronous path.
    public var frontierAsyncWrites = false
    /// Width of the asynchronous writer's staging window in recurrent layer slots (prefix_cache.frontier_write_staging_slots;
    /// default 4, range 4...96, wider widths need frontier_async_writes; the plan uses 64).
    public var frontierWriteStagingSlots = 4
    public var prefixMemoryBytes: UInt64 = 0
    public var prefixDiskBytes: UInt64 = 0
    public var prefixDirectory: URL?
    public var residentBytes: UInt64 = 0
    public var residentSmallTensors = false
    public var groupedExpertPrefill = false
    public var groupedExpertTiled = false
    public var groupedExpertBM32 = false
    public var groupedExpertBM32MinRows = 128
    public var fusedExperts = false
    public var expertCommandBatch = 1
    public var expertScratchReuse = false
    public var moeDevicePrefill = false
    public var simdMatrix = false
    public var quantizedTiled = false
    public var quantizedTiledBM32 = false
    public var quantizedBM32MinRows = 32
    public var gdnInputComposition = false
    public var gatheredExpertDecode = false
    public var moeSharedReadOverlap = false
    public var gdnBufferRegion = false
    public var gdnDeviceBlock = false
    public var wholeExecutionSchedule = false
    public var raggedPrefillScope = false
    public var decodeEarlyDemand = false
    /// Queued experts: each decode layer's experts are queued before its routing is known; the host only fills and signals them.
    public var decodePreencodedLayers = false
    public var decodeTokenPlan = false
    public var devicePLE = false
    /// Persistent dedicated reader threads: the nine spans of original expert records are read
    /// concurrently, and decode PLE rows use their own pool instead of GCD's.
    public var ioReaderPool = false
    public var quantizedTiledPrefetch = false
    public var hostStagePrefetch = false
    /// Two C256 prefill tile workspaces: GDN attention tiles (layers other than 1) and canonical
    /// mix tiles are joined and flag-checked only after the next tile was submitted.
    public var prefillDoubleBuffer = false
    /// Pre-routing: layer L+1's MoE HC + router applied to layer L's post-attention row at decode, gated by
    /// logit margin, issuing prefetch-priority reads of the predicted experts into spare bandwidth.
    public var decodeNextLayerPrerouting = false
    public var decodePreroutingMargin = 0.25
    /// Pre-routing slot override: a predicted key the reuse-score rule refuses may instead evict the
    /// lowest-score unprotected, unpinned verified record outside its target layer and the layer
    /// before it, at most `decodePreroutingOverrideWidth` (1...10) keys per target layer. Reuses
    /// admitted expert-cache slots (no new memory owner). Requires pre-routing and reuse admission.
    public var decodePreroutingSlotOverride = false
    public var decodePreroutingOverrideWidth = 10
    /// Pre-routing read hygiene: at adoption, predicted reads outside the authoritative route that
    /// are still queued are cancelled (started reads finish); the original-span reader pool serves
    /// demand spans before predicted ones; no predicted read starts while a decode token's PLE rows
    /// are read. Order and residency only; bytes and arithmetic unchanged. Requires pre-routing.
    public var decodePreroutingReadHygiene = false
    /// Decode PLE rows race two sources: the packed PLE row (pack volume) through the `io_reader_pool`
    /// row pool, and the same 100 bytes from the original checkpoint spans (model path) through a
    /// second dedicated 16-thread pool. A row is ready when either source has read it, so the pack
    /// drive's queue state cannot delay a token alone. Bytes and arithmetic unchanged. Requires
    /// device_ple and io_reader_pool.
    public var decodePLEInternalRace = false
    /// Prefill PLE rows stop blocking the GPU: each scope's rows race the packed `ple.bin` against the
    /// original checkpoint spans (a dedicated 32-thread pool, reading from the last row down) while layer
    /// 0's load and attention tiles run; the expert read-ahead starts as soon as the rows are complete
    /// (at the latest before layer 0's MoE), so the rows still never compete with expert reads. A host
    /// tail tile takes its rows from the joined scope. Bytes and arithmetic unchanged. Requires device_ple.
    public var prefillPLEOverlap = false
    /// Dense BM32-prefetch projections with rows >= 64 and N % 64 == 0 use the BM64 x BN64 bfloat-MMA
    /// kernel (exact; the standalone check proves it on this GPU). Requires quantized_tiled_prefetch.
    public var quantizedMMABF16 = false
    /// 0 = off; otherwise grouped BM32 expert jobs with at least this many rows (64...256) use the
    /// bfloat-MMA kernel. Requires quantized_mma_bf16 and grouped_expert_bm32.
    public var groupedExpertMMAMinRows = 0
    /// Eight-row register-tiled GDN recurrence and vector-load preparation convolution (both exact).
    /// Requires fused_gdn_preparation.
    public var gdnExactKernels = false
    /// Per-request fast decode allowance (`fast.allow_opt_in` with `fast.late_miss_skip_tau` in (0, 1]): the τ a
    /// request that sends `"origami_mode": "fast"` (or the `-fast` model id) runs with; nil = such requests are refused.
    /// Requests without it stay exact; fast requests are NUMERIC. See FastLateMissSkip.
    public var fastOptInTau: Double?
    public var elasticDecodeResidency = false
    public var fusedHCDecode = false
    public var fusedHCPrefill = false
    public var fusedHCEpilogue = false
    public var fusedHCMinRows = 8
    public var gdnResidentState = false
    public var fusedGDNPreparation = false
    public var fusedGDNMinRows = 8
    public var fusedResidualInjection = false
    public var fusedResidualMinRows = 8
    public var pinnedUnary = false
    public var experts = ExpertExecutionConfiguration()
    /// The hot copy's folder (`HotCopyKeeper` builds and attaches it while serving), when the copy is on.
    public var hotCopyFolder: URL?
    /// storage.integrity: packed read verification (sampled by default, or sha256 for every read); the hot copy checks
    /// each record's CRC32 on its first read.
    public var readIntegrity: ReadIntegrity = .sampled
    public init() {}

    /// Every key the loader reads, per object. Anything else is reported by `unknownKeys` and ignored.
    static let allowedKeys: [String: Set<String>] = [
        "": [
            "context_limit", "context_in_memory", "process_ceiling_bytes", "transient_margin_bytes",
            "execution", "prefill", "prefix_cache", "quotas", "storage", "fast",
        ],
        "prefill": ["compute_tile", "read_scope"],
        "prefix_cache": [
            "persistent_conversation", "durable_format", "memory_bytes", "disk_bytes", "disk_directory",
            "frontier_disk_bytes", "frontier_layout", "frontier_async_writes", "frontier_write_staging_slots",
        ],
        "quotas": ["snapshot_bytes", "spill_bytes"],
        "storage": ["integrity"],
        "fast": ["enabled", "allow_opt_in", "late_miss_skip_tau"],
        "execution": [
            "expert_mode", "expert_cache_slots", "expert_slots_per_slab", "expert_window", "io_queue_depth",
            "prefill_expert_cache_slots", "expert_prefill_source_policy", "expert_decode_source_policy", "expert_source_calibration",
            "expert_source_online_calibration", "expert_source_online_bounds", "expert_tail_replicas", "expert_prefill_read_ahead",
            "expert_prefill_layer_command", "expert_reuse_admission", "preallocate_decode_slabs", "wire_decode_slabs", "expert_pack_drives",
            "expert_layer_waves", "simd_matrix",
            "pinned_unary", "gathered_expert_decode", "moe_shared_read_overlap", "gdn_buffer_region", "gdn_device_block",
            "decode_early_demand", "decode_preencoded_layers", "decode_token_plan", "device_ple", "io_reader_pool",
            "host_stage_prefetch",
            "prefill_double_buffer", "decode_next_layer_prerouting", "decode_prerouting_slot_override", "decode_prerouting_read_hygiene",
            "decode_ple_internal_race", "prefill_ple_overlap", "quantized_mma_bf16", "gdn_exact_kernels",
            "whole_execution_schedule", "ragged_prefill_scope", "elastic_decode_residency", "gdn_input_composition",
            "resident_small_tensors", "gdn_resident_state", "fused_hc_prefill", "fused_hc_decode", "fused_gdn_preparation",
            "fused_hc_epilogue", "fused_residual_injection", "quantized_tiled_bm32", "quantized_tiled_prefetch", "quantized_tiled",
            "pinned_rms", "batched_selector", "pinned_reductions", "pinned_attention", "qsa_decode_reuse",
            "batched_pinned_attention", "fused_selected_prefill_attention", "compact_pv_grouped", "qsa_exact_kernels",
            "hc_exact_fusion", "hc_decode_fusion", "batched_rope", "pinned_rope", "pinned_indexer",
            "register_qsa_decode", "gpu_keepalive", "pinned_dense", "grouped_expert_bm32",
            "grouped_expert_tiled", "moe_device_prefill", "expert_scratch_reuse", "fused_experts", "grouped_expert_prefill",
            "fused_gdn_min_rows", "fused_hc_min_rows", "fused_residual_min_rows", "quantized_bm32_min_rows",
            "grouped_expert_bm32_min_rows", "grouped_expert_mma_min_rows", "expert_command_batch", "resident_tensor_bytes",
            "dense_resident_bytes", "decode_prerouting_margin", "decode_prerouting_override_width",
        ],
        "execution.expert_source_calibration": [
            "queue_depths", "internal_bytes_per_second", "external_bytes_per_second",
            "internal_latency_seconds", "external_latency_seconds", "label", "replica_delay_seconds",
        ],
        "execution.expert_source_online_bounds": [
            "floor_multiple", "ceiling_multiple", "smoothing", "warmup_observations",
            "delay_ceiling_multiple",
        ],
    ]
    /// Switches this engine no longer implements; ignored with a warning when set.
    static let retiredExecutionSwitches = [
        "hc_four_stage", "expert_striped_reads", "expert_all_hot_striping", "profile_expert_reads",
        "exact_greedy_128", "decode_device_frontier", "last_prompt_expert_warm", "dense_row_invariant", "pre_routing_capture_directory",
        "batched_qsa", "decode_arriving_first",
    ]

    /// Keys this engine ignores in `object`, as dotted paths, for a warning.
    public static func unknownKeys(_ object: [String: Any]) -> [String] {
        var unknown: [String] = []
        for (path, allowed) in allowedKeys {
            var value: Any? = object
            for part in path.split(separator: ".") { value = (value as? [String: Any])?[String(part)] }
            for key in (value as? [String: Any])?.keys.sorted() ?? [] where !allowed.contains(key) {
                unknown.append(path.isEmpty ? key : path + "." + key)
            }
        }
        return unknown.sorted()
    }

    /// The largest owner budget: what the configured caches are sized for.
    public var effectiveUsableBytes: UInt64 {
        get throws {
            guard processCeiling > transientMargin else { throw EngineError("Invalid memory ceiling/margin") }
            return processCeiling - transientMargin
        }
    }
    public func makeLedger() throws -> MemoryLedger {
        try MemoryLedger(
            ceiling: processCeiling, transientMargin: transientMargin,
            usableLimit: startUsableBytes.map { min($0, try effectiveUsableBytes) }, availableFloor: availableFloorBytes,
            budgetMode: memoryBudgetMode)
    }

    /// Made even when off, so the tuner's requests can turn it on for themselves (`TuneOverrides`); off, one that
    /// cannot be made is simply missing.
    public func makeKeepAlive(ledger: MemoryLedger) throws -> RequestGPUKeepAlive? {
        let policy: RequestGPUKeepAlive.Policy = registerQSADecode ? .shortRegisterAttention : .original
        return gpuKeepAlive
            ? try RequestGPUKeepAlive(ledger: ledger, policy: policy) : try? RequestGPUKeepAlive(ledger: ledger, policy: policy)
    }

    /// Shared configuration wiring for generation, serving, and state diagnostics.
    public func makeExecutor(
        store: PackStore, ledger: MemoryLedger, contextLimitOverride: Int? = nil,
        additionalPhaseOwnerBytes: UInt64 = 0
    ) throws -> SerialExecutor {
        guard !wholeExecutionSchedule || contextLimitOverride == nil || contextLimitOverride == contextLimit else {
            throw EngineError("Whole schedule context override differs from its configured owners", code: "invalid_configuration")
        }
        if hotCopyFolder != nil {
            do { try store.reserveHotCopyMetadata(ledger: ledger) } catch {
                FileHandle.standardError.write(Data("origami: the hot copy's index is not charged: \(error)\n".utf8))
            }
        }
        let decodeRowPool = ioReaderPool ? try SpanReadPool(threads: 16) : nil
        if ioReaderPool { try store.enableConcurrentOriginalSpans(pool: try SpanReadPool(threads: 16)) }
        let decodePLERacePool = decodePLEInternalRace && ioReaderPool && devicePLE ? try SpanReadPool(threads: 16) : nil
        let prefillPLERacePool = prefillPLEOverlap && devicePLE ? try SpanReadPool(threads: 32) : nil
        // Both owners are reserved by their own objects; the executor only budgets phase owners, never reserves them.
        let mirrorOwners =
            (hotCopyFolder == nil ? 0 : PackStore.hotCopyMetadataBytes)
            + (frontierAsyncWrites ? ConversationFrontierStore.stagingReservationBytes(slots: frontierWriteStagingSlots) : 0)
        let (phaseOwners, overflow) = additionalPhaseOwnerBytes.addingReportingOverflow(mirrorOwners)
        guard !overflow else { throw EngineError("Hot copy phase admission overflow") }
        return try SerialExecutor(
            store: store, ledger: ledger, contextLimit: (contextLimitOverride ?? contextLimit),
            fileBackedContext: contextLimitOverride == nil ? fileBackedContext : nil,
            experts: experts, additionalPhaseOwnerBytes: phaseOwners, simdMatrix: simdMatrix, pinnedUnary: pinnedUnary,
            pinnedRMS: pinnedRMS, residentBytes: residentBytes, residentSmallTensors: residentSmallTensors,
            groupedExpertPrefill: groupedExpertPrefill,
            pinnedReductions: pinnedReductions, batchedSelector: batchedSelector, denseResidentBytes: denseResidentBytes,
            fusedExperts: fusedExperts, pinnedAttention: pinnedAttention, registerQSADecode: registerQSADecode,
            batchedPinnedAttention: batchedPinnedAttention, fusedSelectedPrefillAttention: fusedSelectedPrefillAttention,
            compactPVGrouped: compactPVGrouped, pinnedDense: pinnedDense, expertCommandBatch: expertCommandBatch,
            groupedExpertTiled: groupedExpertTiled, batchedRoPE: batchedRoPE, pinnedRoPE: pinnedRoPE, pinnedIndexer: pinnedIndexer,
            quantizedTiled: quantizedTiled, fusedHCEpilogue: fusedHCEpilogue, fusedHCMinRows: fusedHCMinRows,
            fusedResidualInjection: fusedResidualInjection, fusedResidualMinRows: fusedResidualMinRows,
            fusedGDNPreparation: fusedGDNPreparation, fusedGDNMinRows: fusedGDNMinRows, quantizedTiledBM32: quantizedTiledBM32,
            quantizedBM32MinRows: quantizedBM32MinRows, groupedExpertBM32: groupedExpertBM32,
            groupedExpertBM32MinRows: groupedExpertBM32MinRows, expertScratchReuse: expertScratchReuse, gdnResidentState: gdnResidentState,
            qsaDecodeReuse: qsaDecodeReuse, fusedHCDecode: fusedHCDecode, fusedHCPrefill: fusedHCPrefill,
            moeDevicePrefill: moeDevicePrefill, gdnInputComposition: gdnInputComposition, gdnBufferRegion: gdnBufferRegion,
            gatheredExpertDecode: gatheredExpertDecode, moeSharedReadOverlap: moeSharedReadOverlap, gdnDeviceBlock: gdnDeviceBlock,
            wholeExecutionSchedule: wholeExecutionSchedule, raggedPrefillScope: raggedPrefillScope, decodeEarlyDemand: decodeEarlyDemand,
            decodePreencodedLayers: decodePreencodedLayers, decodeTokenPlan: decodeTokenPlan,
            elasticDecodeResidency: elasticDecodeResidency, devicePLE: devicePLE, hostStagePrefetch: hostStagePrefetch,
            quantizedTiledPrefetch: quantizedTiledPrefetch, decodeRowPool: decodeRowPool, prefillDoubleBuffer: prefillDoubleBuffer,
            quantizedMMABF16: quantizedMMABF16, groupedExpertMMAMinRows: groupedExpertMMAMinRows, gdnExactKernels: gdnExactKernels,
            decodeNextLayerPrerouting: decodeNextLayerPrerouting, decodePreroutingMargin: decodePreroutingMargin,
            decodePreroutingOverrideWidth: decodePreroutingSlotOverride ? decodePreroutingOverrideWidth : 0,
            qsaExactKernels: qsaExactKernels, hcExactFusion: hcExactFusion, decodePreroutingReadHygiene: decodePreroutingReadHygiene,
            decodePLERacePool: decodePLERacePool, hcDecodeFusion: hcDecodeFusion, prefillPLERacePool: prefillPLERacePool,
            fastOptInTau: fastOptInTau)
    }

    public static func load(_ url: URL) throws -> RuntimeConfiguration {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw EngineError("\(url.path) is not a JSON object", code: "invalid_configuration")
        }
        return try load(object: object)
    }

    /// Builds a configuration from a JSON object. Unknown keys and retired switches are
    /// ignored (`warn` reports them); switches that contradict each other are refused,
    /// because the dependent kernel paths would not exist.
    public static func load(object: [String: Any], warn: (String) -> Void = { _ in }) throws -> RuntimeConfiguration {
        var object = object
        for key in unknownKeys(object) { warn("ignoring unknown configuration key \(key)") }
        if var execution = object["execution"] as? [String: Any] {
            for key in retiredExecutionSwitches where execution[key] != nil {
                if (execution[key] as? Bool) == true || !(execution[key] is Bool) { warn("ignoring retired switch execution.\(key)") }
                execution.removeValue(forKey: key)
            }
            object["execution"] = execution
        }
        func integer(_ object: [String: Any], _ key: String, _ fallback: Int) throws -> Int {
            guard let value = object[key] else { return fallback }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
                number.doubleValue >= 0, number.doubleValue < Double(Int.max)
            else {
                throw EngineError("Invalid nonnegative configuration integer: \(key)")
            }
            return number.intValue
        }
        var result = RuntimeConfiguration()
        result.processCeiling = UInt64(try integer(object, "process_ceiling_bytes", Int(result.processCeiling)))
        result.transientMargin = UInt64(try integer(object, "transient_margin_bytes", Int(result.transientMargin)))
        result.contextLimit = try integer(object, "context_limit", result.contextLimit)
        if let value = object["context_in_memory"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID(),
                !number.boolValue || result.contextLimit == ConversationGeometry.maximumFileContext
            else {
                throw EngineError("context_in_memory is a boolean for the 262,144 context", code: "invalid_configuration")
            }
            result.contextInMemory = number.boolValue
        }
        if let prefill = object["prefill"] as? [String: Any] {
            result.computeTile = try integer(prefill, "compute_tile", result.computeTile)
            result.readScope = try integer(prefill, "read_scope", result.computeTile)
        }
        if let prefix = object["prefix_cache"] as? [String: Any] {
            if let value = prefix["persistent_conversation"] {
                guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                    throw EngineError("persistent_conversation must be boolean")
                }
                result.persistentConversation = number.boolValue
            }
            // The one durable format; the key stays accepted for the configurations that name it.
            if let value = prefix["durable_format"], value as? String != "paged-v2" { throw EngineError("Unknown prefix durable format") }
            result.prefixMemoryBytes = UInt64(try integer(prefix, "memory_bytes", 0))
            result.prefixDiskBytes = UInt64(try integer(prefix, "disk_bytes", 0))
            result.conversationFrontierDiskBytes = UInt64(try integer(prefix, "frontier_disk_bytes", 0))
            if let value = prefix["frontier_async_writes"] {
                guard let flag = value as? NSNumber, CFGetTypeID(flag) == CFBooleanGetTypeID() else {
                    throw EngineError("frontier_async_writes must be a boolean", code: "invalid_configuration")
                }
                result.frontierAsyncWrites = flag.boolValue
            }
            result.frontierWriteStagingSlots = try integer(prefix, "frontier_write_staging_slots", 4)
            if let path = prefix["disk_directory"] {
                guard let directory = path as? String, directory.hasPrefix("/"),
                    !directory.split(separator: "/").contains("..")
                else {
                    throw EngineError("prefix_cache.disk_directory must be an absolute path without '..'", code: "invalid_configuration")
                }
                result.prefixDirectory = URL(fileURLWithPath: directory, isDirectory: true)
            }
            // 9 GB covers the 256K in-memory conversation (its pages and the 1.1 GB the other tiers also get).
            guard result.prefixMemoryBytes <= 9_000_000_000, result.prefixDiskBytes <= 16_000_000_000,
                result.prefixDirectory == nil || (result.prefixMemoryBytes > 0 && result.prefixDiskBytes > 0)
            else {
                throw EngineError("Prefix cache exceeds its bounded configuration")
            }
        }
        let execution = object["execution"] as? [String: Any] ?? [:]
        let modeName = execution["expert_mode"] as? String ?? ExpertExecutionMode.legacyOriginal.rawValue
        guard let mode = ExpertExecutionMode(rawValue: modeName) else { throw EngineError("Unknown expert execution mode") }
        func sourcePolicy(_ key: String) throws -> ExpertSourcePolicy {
            guard let value = execution[key] else { return .packedPreferred }
            guard let text = value as? String, let policy = ExpertSourcePolicy(rawValue: text) else {
                throw EngineError("Invalid expert-source policy: " + key, code: "invalid_configuration")
            }
            return policy
        }
        let calibration: ExpertSourceCalibration
        if let object = execution["expert_source_calibration"] {
            guard let dictionary = object as? [String: Any] else { throw EngineError("Expert source calibration must be an object") }
            calibration = try JSONDecoder().decode(ExpertSourceCalibration.self, from: JSONSerialization.data(withJSONObject: dictionary))
        } else {
            calibration = .neutral
        }
        var tailReplicas = false
        if let value = execution["expert_tail_replicas"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("expert_tail_replicas must be boolean", code: "invalid_configuration")
            }
            tailReplicas = number.boolValue
        }
        var prefillReadAhead = false
        if let value = execution["expert_prefill_read_ahead"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("expert_prefill_read_ahead must be boolean", code: "invalid_configuration")
            }
            prefillReadAhead = number.boolValue
        }
        var prefillLayerCommand = false
        if let value = execution["expert_prefill_layer_command"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("expert_prefill_layer_command must be boolean", code: "invalid_configuration")
            }
            prefillLayerCommand = number.boolValue
        }
        var reuseAdmission = false
        if let value = execution["expert_reuse_admission"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("expert_reuse_admission must be boolean", code: "invalid_configuration")
            }
            reuseAdmission = number.boolValue
        }
        var preallocateDecodeSlabs = false
        if let value = execution["preallocate_decode_slabs"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("preallocate_decode_slabs must be boolean", code: "invalid_configuration")
            }
            preallocateDecodeSlabs = number.boolValue
        }
        // Online source calibration: a boolean flag (default false keeps the static path byte-for-byte)
        // and optional explicit bounds. Partial, non-numeric or out-of-range bounds are refused here.
        var onlineCalibration: ExpertSourceOnlineCalibration?
        if let value = execution["expert_source_online_calibration"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("expert_source_online_calibration must be boolean", code: "invalid_configuration")
            }
            if number.boolValue { onlineCalibration = ExpertSourceOnlineCalibration() }
        }
        if let value = execution["expert_source_online_bounds"] {
            guard onlineCalibration != nil else {
                throw EngineError("expert_source_online_bounds requires expert_source_online_calibration", code: "invalid_configuration")
            }
            guard let dictionary = value as? [String: Any] else {
                throw EngineError(
                    "expert_source_online_bounds must be an object with floor_multiple, ceiling_multiple, smoothing, warmup_observations and an optional delay_ceiling_multiple",
                    code: "invalid_configuration")
            }
            let bounds: ExpertSourceOnlineCalibration
            do {
                bounds = try JSONDecoder().decode(
                    ExpertSourceOnlineCalibration.self, from: JSONSerialization.data(withJSONObject: dictionary))
            } catch {
                throw EngineError(
                    "expert_source_online_bounds must carry numeric floor_multiple, ceiling_multiple, smoothing, integer warmup_observations and an optional numeric delay_ceiling_multiple: \(error)",
                    code: "invalid_configuration")
            }
            try bounds.validate()
            onlineCalibration = bounds
        }
        result.experts = ExpertExecutionConfiguration(
            mode: mode,
            cacheSlots: try integer(execution, "expert_cache_slots", 64),
            slotsPerSlab: try integer(execution, "expert_slots_per_slab", 8),
            demandWindow: try integer(execution, "expert_window", 8),
            readConcurrency: try integer(execution, "io_queue_depth", 4),
            prefillCacheSlots: execution["prefill_expert_cache_slots"] == nil
                ? nil : try integer(execution, "prefill_expert_cache_slots", 64),
            prefillSourcePolicy: try sourcePolicy("expert_prefill_source_policy"),
            decodeSourcePolicy: try sourcePolicy("expert_decode_source_policy"),
            sourceCalibration: calibration, tailReplicas: tailReplicas, prefillReadAhead: prefillReadAhead,
            prefillLayerCommand: prefillLayerCommand, reuseAdmission: reuseAdmission,
            preallocateDecodeSlabs: preallocateDecodeSlabs, onlineCalibration: onlineCalibration)
        result.experts.layerWaves = try integer(execution, "expert_layer_waves", 1)
        result.experts.packDrives = try integer(execution, "expert_pack_drives", 1)
        guard (1...16).contains(result.experts.packDrives) else {
            throw EngineError("expert_pack_drives must be in 1...16", code: "invalid_configuration")
        }
        if let value = execution["wire_decode_slabs"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("wire_decode_slabs must be boolean", code: "invalid_configuration")
            }
            guard !number.boolValue || preallocateDecodeSlabs else {
                throw EngineError("wire_decode_slabs requires preallocate_decode_slabs", code: "invalid_configuration")
            }
            result.experts.wireDecodeSlabs = number.boolValue
        }
        guard (1...16).contains(result.experts.layerWaves), result.experts.layerWaves == 1 || prefillLayerCommand else {
            throw EngineError(
                "expert_layer_waves must be in 1...16 and requires expert_prefill_layer_command", code: "invalid_configuration")
        }
        try result.experts.validate()
        guard result.experts.prefillCacheSlots == nil || mode != .legacyOriginal else {
            throw EngineError("prefill_expert_cache_slots requires a slab expert mode", code: "invalid_configuration")
        }
        if let value = execution["simd_matrix"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("simd_matrix must be boolean")
            }
            result.simdMatrix = number.boolValue
        }
        if let value = execution["pinned_unary"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw EngineError("pinned_unary must be boolean")
            }
            result.pinnedUnary = number.boolValue
        }
        for (key, assign) in [
            ("gathered_expert_decode", { (value: Bool) in result.gatheredExpertDecode = value }),
            ("moe_shared_read_overlap", { (value: Bool) in result.moeSharedReadOverlap = value }),
            ("gdn_buffer_region", { (value: Bool) in result.gdnBufferRegion = value }),
            ("gdn_device_block", { (value: Bool) in result.gdnDeviceBlock = value }),
            ("decode_early_demand", { (value: Bool) in result.decodeEarlyDemand = value }),
            ("decode_preencoded_layers", { (value: Bool) in result.decodePreencodedLayers = value }),
            ("decode_token_plan", { (value: Bool) in result.decodeTokenPlan = value }),
            ("device_ple", { (value: Bool) in result.devicePLE = value }),
            ("io_reader_pool", { (value: Bool) in result.ioReaderPool = value }),
            ("host_stage_prefetch", { (value: Bool) in result.hostStagePrefetch = value }),
            ("prefill_double_buffer", { (value: Bool) in result.prefillDoubleBuffer = value }),
            ("decode_next_layer_prerouting", { (value: Bool) in result.decodeNextLayerPrerouting = value }),
            ("decode_prerouting_slot_override", { (value: Bool) in result.decodePreroutingSlotOverride = value }),
            ("decode_prerouting_read_hygiene", { (value: Bool) in result.decodePreroutingReadHygiene = value }),
            ("decode_ple_internal_race", { (value: Bool) in result.decodePLEInternalRace = value }),
            ("prefill_ple_overlap", { (value: Bool) in result.prefillPLEOverlap = value }),
            ("quantized_mma_bf16", { (value: Bool) in result.quantizedMMABF16 = value }),
            ("gdn_exact_kernels", { (value: Bool) in result.gdnExactKernels = value }),
            ("whole_execution_schedule", { (value: Bool) in result.wholeExecutionSchedule = value }),
            ("ragged_prefill_scope", { (value: Bool) in result.raggedPrefillScope = value }),
            ("elastic_decode_residency", { (value: Bool) in result.elasticDecodeResidency = value }),
            ("gdn_input_composition", { (value: Bool) in result.gdnInputComposition = value }),
            ("resident_small_tensors", { (value: Bool) in result.residentSmallTensors = value }),
            ("gdn_resident_state", { (value: Bool) in result.gdnResidentState = value }),
            ("fused_hc_prefill", { (value: Bool) in result.fusedHCPrefill = value }),
            ("fused_hc_decode", { (value: Bool) in result.fusedHCDecode = value }),
            ("fused_gdn_preparation", { (value: Bool) in result.fusedGDNPreparation = value }),
            ("fused_hc_epilogue", { (value: Bool) in result.fusedHCEpilogue = value }),
            ("fused_residual_injection", { (value: Bool) in result.fusedResidualInjection = value }),
            ("quantized_tiled_bm32", { (value: Bool) in result.quantizedTiledBM32 = value }),
            ("quantized_tiled_prefetch", { (value: Bool) in result.quantizedTiledPrefetch = value }),
            ("quantized_tiled", { (value: Bool) in result.quantizedTiled = value }),
            ("pinned_rms", { (value: Bool) in result.pinnedRMS = value }),
            ("batched_selector", { (value: Bool) in result.batchedSelector = value }),
            ("pinned_reductions", { (value: Bool) in result.pinnedReductions = value }),
            ("pinned_attention", { (value: Bool) in result.pinnedAttention = value }),
            ("qsa_decode_reuse", { (value: Bool) in result.qsaDecodeReuse = value }),
            ("batched_pinned_attention", { (value: Bool) in result.batchedPinnedAttention = value }),
            ("fused_selected_prefill_attention", { (value: Bool) in result.fusedSelectedPrefillAttention = value }),
            ("compact_pv_grouped", { (value: Bool) in result.compactPVGrouped = value }),
            ("qsa_exact_kernels", { (value: Bool) in result.qsaExactKernels = value }),
            ("hc_exact_fusion", { (value: Bool) in result.hcExactFusion = value }),
            ("hc_decode_fusion", { (value: Bool) in result.hcDecodeFusion = value }),
            ("batched_rope", { (value: Bool) in result.batchedRoPE = value }),
            ("pinned_rope", { (value: Bool) in result.pinnedRoPE = value }),
            ("pinned_indexer", { (value: Bool) in result.pinnedIndexer = value }),
            ("register_qsa_decode", { (value: Bool) in result.registerQSADecode = value }),
            ("gpu_keepalive", { (value: Bool) in result.gpuKeepAlive = value }),
            ("pinned_dense", { (value: Bool) in result.pinnedDense = value }),
            ("grouped_expert_bm32", { (value: Bool) in result.groupedExpertBM32 = value }),
            ("grouped_expert_tiled", { (value: Bool) in result.groupedExpertTiled = value }),
            ("moe_device_prefill", { (value: Bool) in result.moeDevicePrefill = value }),
            ("expert_scratch_reuse", { (value: Bool) in result.expertScratchReuse = value }),
            ("fused_experts", { (value: Bool) in result.fusedExperts = value }),
            ("grouped_expert_prefill", { (value: Bool) in result.groupedExpertPrefill = value }),
        ] {
            if let value = execution[key] {
                guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                    throw EngineError("\(key) must be boolean")
                }
                assign(number.boolValue)
            }
        }
        result.fusedGDNMinRows = try integer(execution, "fused_gdn_min_rows", 8)
        guard (1...256).contains(result.fusedGDNMinRows), !result.fusedGDNPreparation || result.pinnedReductions else {
            throw EngineError(
                "fused_gdn_preparation requires pinned_reductions and fused_gdn_min_rows in 1...256", code: "invalid_configuration")
        }
        result.fusedHCMinRows = try integer(execution, "fused_hc_min_rows", 8)
        result.fusedResidualMinRows = try integer(execution, "fused_residual_min_rows", 8)
        guard (1...256).contains(result.fusedResidualMinRows) else {
            throw EngineError("fused_residual_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard (1...256).contains(result.fusedHCMinRows) else {
            throw EngineError("fused_hc_min_rows must be in 1...256", code: "invalid_configuration")
        }
        result.quantizedBM32MinRows = try integer(execution, "quantized_bm32_min_rows", 32)
        guard !result.quantizedTiledPrefetch || result.quantizedTiledBM32 else {
            throw EngineError("quantized_tiled_prefetch requires quantized_tiled_bm32", code: "invalid_configuration")
        }
        guard (1...256).contains(result.quantizedBM32MinRows), !result.quantizedTiledBM32 || result.quantizedTiled else {
            throw EngineError("quantized_tiled_bm32 requires quantized_tiled and minimum rows in 1...256", code: "invalid_configuration")
        }
        result.groupedExpertBM32MinRows = try integer(execution, "grouped_expert_bm32_min_rows", 128)
        guard (1...256).contains(result.groupedExpertBM32MinRows), !result.groupedExpertBM32 || result.groupedExpertTiled else {
            throw EngineError(
                "grouped_expert_bm32 requires grouped_expert_tiled and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard !result.quantizedMMABF16 || result.quantizedTiledPrefetch else {
            throw EngineError("quantized_mma_bf16 requires quantized_tiled_prefetch", code: "invalid_configuration")
        }
        result.groupedExpertMMAMinRows = try integer(execution, "grouped_expert_mma_min_rows", 0)
        guard
            result.groupedExpertMMAMinRows == 0
                || ((64...256).contains(result.groupedExpertMMAMinRows) && result.quantizedMMABF16 && result.groupedExpertBM32)
        else {
            throw EngineError(
                "grouped_expert_mma_min_rows requires quantized_mma_bf16, grouped_expert_bm32 and a floor in 64...256",
                code: "invalid_configuration")
        }
        guard !result.gdnExactKernels || result.fusedGDNPreparation else {
            throw EngineError("gdn_exact_kernels requires fused_gdn_preparation", code: "invalid_configuration")
        }
        result.expertCommandBatch = try integer(execution, "expert_command_batch", 1)
        result.residentBytes = UInt64(try integer(execution, "resident_tensor_bytes", 0))
        result.denseResidentBytes = UInt64(try integer(execution, "dense_resident_bytes", 0))
        let storage = object["storage"] as? [String: Any] ?? [:]
        if let value = storage["integrity"] {
            guard let text = value as? String, let integrity = ReadIntegrity(rawValue: text) else {
                throw EngineError("storage.integrity must be sampled or sha256", code: "invalid_configuration")
            }
            result.readIntegrity = integrity
        }
        guard (2...ModelContract.maximumContext).contains(result.contextLimit),
            (1...256).contains(result.computeTile),
            result.readScope >= result.computeTile, result.readScope <= 8192, result.readScope % result.computeTile == 0,
            result.residentBytes <= 4_000_000_000, result.denseResidentBytes <= 600_000_000, result.processCeiling > result.transientMargin
        else {
            throw EngineError("Runtime configuration exceeds the current memory/context contract")
        }
        guard result.prefixMemoryBytes == 0 || (result.contextLimit + result.computeTile - 1) / result.computeTile <= 8192 else {
            throw EngineError("Prefix reuse requires at most 8192 compute segments within the configured context")
        }
        guard
            !result.moeDevicePrefill
                || (result.fusedExperts && result.groupedExpertPrefill && result.pinnedReductions && result.expertCommandBatch > 1
                    && result.experts.mode != .legacyOriginal)
        else {
            throw EngineError(
                "moe_device_prefill requires fused grouped slab experts, pinned reductions and command batching",
                code: "invalid_configuration")
        }
        guard !result.expertScratchReuse || result.fusedExperts else {
            throw EngineError("expert_scratch_reuse requires fused_experts", code: "invalid_configuration")
        }
        guard !result.fusedExperts || mode != .legacyOriginal else { throw EngineError("fused_experts requires a slab expert mode") }
        guard (1...8).contains(result.expertCommandBatch),
            result.expertCommandBatch == 1 || (result.fusedExperts && mode != .legacyOriginal)
        else {
            throw EngineError("expert_command_batch requires one to eight jobs and fused slab execution", code: "invalid_configuration")
        }
        guard !result.groupedExpertTiled || (result.fusedExperts && result.simdMatrix && result.groupedExpertPrefill) else {
            throw EngineError(
                "grouped_expert_tiled requires fused_experts, simd_matrix, and grouped_expert_prefill", code: "invalid_configuration")
        }
        guard !result.qsaDecodeReuse || result.pinnedAttention else { throw EngineError("qsa_decode_reuse requires pinned_attention") }
        guard !result.fusedSelectedPrefillAttention || (result.batchedPinnedAttention && result.pinnedAttention) else {
            throw EngineError("fused_selected_prefill_attention requires batched_pinned_attention")
        }
        guard !result.compactPVGrouped || result.fusedSelectedPrefillAttention else {
            throw EngineError("compact_pv_grouped requires fused_selected_prefill_attention", code: "invalid_configuration")
        }
        guard !result.qsaExactKernels || (result.batchedPinnedAttention && result.pinnedAttention) else {
            throw EngineError("qsa_exact_kernels requires batched_pinned_attention", code: "invalid_configuration")
        }
        guard !result.hcExactFusion || (result.fusedHCPrefill && result.simdMatrix) else {
            throw EngineError("hc_exact_fusion requires fused_hc_prefill and simd_matrix", code: "invalid_configuration")
        }
        guard !result.hcDecodeFusion || result.fusedHCDecode else {
            throw EngineError("hc_decode_fusion requires fused_hc_decode", code: "invalid_configuration")
        }
        guard !result.batchedPinnedAttention || result.pinnedAttention else {
            throw EngineError("batched_pinned_attention requires pinned_attention")
        }
        guard !result.quantizedTiled || result.simdMatrix else {
            throw EngineError("quantized_tiled requires simd_matrix", code: "invalid_configuration")
        }
        guard
            !(result.fusedHCDecode || result.fusedHCPrefill)
                || (result.pinnedReductions && result.pinnedDense && result.pinnedUnary && result.residentBytes > 0
                    && result.denseResidentBytes > 0)
        else {
            throw EngineError(
                "fused_hc_decode requires pinned_reductions, pinned_dense, pinned_unary and positive resident caches",
                code: "invalid_configuration")
        }
        guard !result.gdnInputComposition || (result.pinnedDense && result.residentBytes > 0 && result.denseResidentBytes > 0) else {
            throw EngineError("gdn_input_composition requires pinned_dense and positive resident caches", code: "invalid_configuration")
        }
        guard !result.residentSmallTensors || result.residentBytes > 0 else {
            throw EngineError("resident_small_tensors requires a positive existing resident tensor cache", code: "invalid_configuration")
        }
        guard !result.moeSharedReadOverlap || result.gatheredExpertDecode else {
            throw EngineError("moe_shared_read_overlap requires gathered_expert_decode", code: "invalid_configuration")
        }
        guard
            !result.gatheredExpertDecode
                || (result.fusedExperts && result.pinnedReductions && result.experts.mode != .legacyOriginal
                    && result.experts.demandWindow >= 10)
        else {
            throw EngineError(
                "gathered_expert_decode requires fused slab experts, pinned reductions and a window of at least ten",
                code: "invalid_configuration")
        }
        guard !result.gdnBufferRegion || (result.pinnedRMS && result.gdnResidentState) else {
            throw EngineError("gdn_buffer_region requires pinned_rms and gdn_resident_state", code: "invalid_configuration")
        }
        guard
            !result.gdnDeviceBlock
                || (result.gdnInputComposition && result.gdnBufferRegion && result.fusedGDNPreparation && result.pinnedUnary
                    && result.pinnedReductions)
        else {
            throw EngineError(
                "gdn_device_block requires composed GDN, fused preparation and pinned unary/reductions", code: "invalid_configuration")
        }
        guard
            !result.registerQSADecode
                || (result.wholeExecutionSchedule
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext)
                    && result.pinnedAttention && result.batchedPinnedAttention)
        else {
            throw EngineError("Register QSA requires explicit in-memory whole decode", code: "invalid_configuration")
        }
        guard !result.decodeTokenPlan || (result.decodeEarlyDemand && result.gdnResidentState && result.wholeExecutionSchedule) else {
            throw EngineError("Token plan requires whole decode, exact early demand and resident GDN", code: "invalid_configuration")
        }
        guard !result.devicePLE || (result.wholeExecutionSchedule && result.pinnedReductions && result.residentBytes > 0) else {
            throw EngineError("Device PLE requires whole execution, pinned reductions and resident tensors", code: "invalid_configuration")
        }
        guard !result.hostStagePrefetch || (result.wholeExecutionSchedule && result.residentBytes > 0 && result.denseResidentBytes > 0)
        else {
            throw EngineError("Host-stage prefetch requires whole execution and both resident caches", code: "invalid_configuration")
        }
        guard !result.prefillDoubleBuffer || result.wholeExecutionSchedule else {
            throw EngineError("prefill_double_buffer requires whole execution", code: "invalid_configuration")
        }
        if let value = execution["decode_prerouting_margin"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
                (0...64).contains(number.doubleValue)
            else {
                throw EngineError("decode_prerouting_margin must be a finite number in 0...64", code: "invalid_configuration")
            }
            result.decodePreroutingMargin = number.doubleValue
        }
        guard !result.decodeNextLayerPrerouting || (result.decodeTokenPlan && result.decodeEarlyDemand && result.wholeExecutionSchedule)
        else {
            throw EngineError(
                "decode_next_layer_prerouting requires decode_token_plan with exact early demand", code: "invalid_configuration")
        }
        result.decodePreroutingOverrideWidth = try integer(execution, "decode_prerouting_override_width", 10)
        guard (1...10).contains(result.decodePreroutingOverrideWidth),
            !result.decodePreroutingSlotOverride
                || (result.decodeNextLayerPrerouting && result.experts.reuseAdmission)
        else {
            throw EngineError(
                "decode_prerouting_slot_override requires decode_next_layer_prerouting and expert_reuse_admission; decode_prerouting_override_width must be in 1...10",
                code: "invalid_configuration")
        }
        guard !result.decodePreroutingReadHygiene || result.decodeNextLayerPrerouting else {
            throw EngineError("decode_prerouting_read_hygiene requires decode_next_layer_prerouting", code: "invalid_configuration")
        }
        guard !result.decodePLEInternalRace || (result.devicePLE && result.ioReaderPool) else {
            throw EngineError("decode_ple_internal_race requires device_ple and io_reader_pool", code: "invalid_configuration")
        }
        guard !result.prefillPLEOverlap || result.devicePLE else {
            throw EngineError("prefill_ple_overlap requires device_ple", code: "invalid_configuration")
        }
        guard !result.decodePreencodedLayers || result.decodeEarlyDemand else {
            throw EngineError("decode_preencoded_layers requires decode_early_demand", code: "invalid_configuration")
        }
        guard
            !result.decodeEarlyDemand
                || (result.wholeExecutionSchedule
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext)
                    && result.experts.demandWindow >= 10)
        else {
            throw EngineError("Early exact demand requires whole decode and a complete ten-record window", code: "invalid_configuration")
        }
        guard
            !result.wholeExecutionSchedule
                || (ConversationGeometry.contexts.contains(result.contextLimit)
                    && result.computeTile == 256 && result.readScope == 8192 && result.gdnDeviceBlock && result.gatheredExpertDecode
                    && result.moeSharedReadOverlap && result.moeDevicePrefill && result.fusedHCDecode && result.fusedHCPrefill
                    && result.fusedResidualInjection && result.batchedSelector && result.residentSmallTensors
                    && result.qsaDecodeReuse && result.batchedPinnedAttention && result.pinnedRoPE && result.pinnedIndexer)
        else {
            throw EngineError(
                "whole_execution_schedule requires declared conversation geometry and R8192/C256 pinned device paths",
                code: "invalid_configuration")
        }
        let conversationGeometry =
            result.persistentConversation
            ? try ConversationGeometry(context: result.contextLimit, fileBacked: result.fileBackedContext) : nil
        guard !result.frontierAsyncWrites || result.conversationFrontierDiskBytes > 0 else {
            throw EngineError("frontier_async_writes requires file-backed conversation frontiers", code: "invalid_configuration")
        }
        guard (4...96).contains(result.frontierWriteStagingSlots), result.frontierAsyncWrites || result.frontierWriteStagingSlots == 4
        else {
            throw EngineError("frontier_write_staging_slots needs frontier_async_writes and 4...96 slots", code: "invalid_configuration")
        }
        let frontierLayout = (object["prefix_cache"] as? [String: Any])?["frontier_layout"] as? String
        guard result.conversationFrontierDiskBytes == 0 ? frontierLayout == nil : frontierLayout == "dense-c256-v1" else {
            throw EngineError("File-backed conversation requires the explicit dense C256 contract", code: "invalid_configuration")
        }
        guard
            result.conversationFrontierDiskBytes == 0
                || (result.persistentConversation && result.prefixDirectory != nil
                    && result.conversationFrontierDiskBytes == conversationGeometry?.frontierDiskBytes)
        else {
            throw EngineError("File-backed frontiers require the complete external conversation quota", code: "invalid_configuration")
        }
        guard
            !result.persistentConversation
                || (result.wholeExecutionSchedule && result.prefixMemoryBytes == conversationGeometry?.conversationBytes)
        else {
            throw EngineError(
                "Persistent conversation requires the whole schedule and declared state budget", code: "invalid_configuration")
        }
        if result.prefixDirectory != nil {
            let layout = try PagedCheckpointFormat.layout(capacity: result.contextLimit, tokens: result.contextLimit, hasLogits: true)
            guard result.persistentConversation, result.prefixDiskBytes >= layout.bytes * 2 else {
                throw EngineError(
                    "Paged durable snapshots require persistent conversation and old+staged quota", code: "invalid_configuration")
            }
        }
        guard
            !result.raggedPrefillScope
                || (result.wholeExecutionSchedule && result.computeTile == 256
                    && result.readScope > 256 && result.experts.prefillLayerCommand && result.persistentConversation
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext))
        else {
            throw EngineError("Mixed-tail scope requires complete whole-prefill conversation admission", code: "invalid_configuration")
        }
        guard !result.experts.prefillLayerCommand || (result.moeDevicePrefill && result.groupedExpertPrefill && result.fusedExperts) else {
            throw EngineError("Layer command requires grouped device expert execution", code: "invalid_configuration")
        }
        guard
            !result.experts.prefillReadAhead
                || (result.wholeExecutionSchedule && result.persistentConversation
                    && result.conversationFrontierDiskBytes > 0
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext))
        else {
            throw EngineError("Prefill read-ahead requires complete conversation admission", code: "invalid_configuration")
        }
        guard
            !result.experts.preallocateDecodeSlabs
                || (result.wholeExecutionSchedule
                    && result.elasticDecodeResidency
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext))
        else {
            throw EngineError("Decode slab preallocation requires elastic whole in-memory execution", code: "invalid_configuration")
        }
        guard
            !result.experts.tailReplicas
                || (result.elasticDecodeResidency && result.wholeExecutionSchedule
                    && result.conversationFrontierDiskBytes > 0)
        else {
            throw EngineError("Tail replicas require complete elastic conversation admission", code: "invalid_configuration")
        }
        if result.conversationFrontierDiskBytes > 0, let geometry = conversationGeometry {
            let fixed = try geometry.maximumFixedBytes(
                input: result.readScope, residentBytes: result.residentBytes + result.denseResidentBytes)
            // Elastic decode may borrow the arena an empty conversation leaves (SerialExecutor.prepareUpcomingInput).
            let decode = try geometry.maximumFixedBytes(
                input: 0, residentBytes: result.residentBytes + result.denseResidentBytes,
                retainedFrontierBytes: result.elasticDecodeResidency ? 0 : nil,
                heldArenaBytes: result.elasticDecodeResidency ? geometry.heldArenaBytes(live: 0, growth: 0) : nil)
            let diagnostic =
                result.experts.readAheadOwnerBytes + result.experts.layerCommandOwnerBytes
                + result.experts.replicaOwnerBytes + result.experts.reuseOwnerBytes + result.experts.decodeSlabOwnerBytes
                + (result.decodeNextLayerPrerouting ? DecodePrerouting.ownerBytes : 0)
            guard diagnostic < (try result.effectiveUsableBytes) else {
                throw EngineError("Decode diagnostic owners exceed admission", code: "insufficient_memory")
            }
            let usable = try result.effectiveUsableBytes - diagnostic
            // The configured maximum assumes no retained frontiers; admission
            // at each joined decode boundary uses the actual reader quota.
            // A full frontier quota must still leave the complete prefill pool.
            let full = try geometry.maximumFixedBytes(input: 0, residentBytes: result.residentBytes + result.denseResidentBytes)
            guard fixed <= usable, result.experts.prefillCacheBytes <= usable - fixed,
                decode <= usable, result.experts.cacheBytes <= usable - decode,
                full <= usable, result.experts.prefillCacheBytes <= usable - full,
                let quotas = object["quotas"] as? [String: Any]
            else {
                throw EngineError("Conversation simultaneous owners exceed their complete envelope", code: "insufficient_memory")
            }
            let aggregate =
                (geometry.fileBacked ? ConversationGeometry.fileDiskBytes : 0) + result.prefixDiskBytes
                + result.conversationFrontierDiskBytes
            let spill = UInt64(try integer(quotas, "spill_bytes", 0))
            guard aggregate <= spill, spill <= ConversationGeometry.aggregateDiskLimit,
                UInt64(try integer(quotas, "snapshot_bytes", 0)) >= result.prefixDiskBytes
            else {
                throw EngineError("Conversation storage omits simultaneous owners", code: "invalid_configuration")
            }
        }
        guard
            !result.elasticDecodeResidency
                || (result.wholeExecutionSchedule && result.persistentConversation
                    && result.conversationFrontierDiskBytes > 0
                    && ConversationGeometry.inMemory(result.contextLimit, fileBacked: result.fileBackedContext))
        else {
            throw EngineError("Elastic decode requires complete in-memory file-frontier owners", code: "invalid_configuration")
        }
        if result.wholeExecutionSchedule && result.fileBackedContext {
            guard result.persistentConversation, result.prefixDirectory != nil,
                result.prefixDiskBytes == ConversationGeometry.durableDiskBytes,
                result.conversationFrontierDiskBytes == conversationGeometry?.frontierDiskBytes,
                let quotas = object["quotas"] as? [String: Any]
            else {
                throw EngineError(
                    "262K whole execution requires persistent file-KV and complete disk quotas", code: "invalid_configuration")
            }
            let snapshot = UInt64(try integer(quotas, "snapshot_bytes", 0))
            let spill = UInt64(try integer(quotas, "spill_bytes", 0))
            let aggregate = ConversationGeometry.fileDiskBytes + result.prefixDiskBytes + result.conversationFrontierDiskBytes
            guard snapshot >= result.prefixDiskBytes, spill >= aggregate, spill <= ConversationGeometry.aggregateDiskLimit else {
                throw EngineError("262K snapshot/spill quotas omit simultaneous runtime owners", code: "invalid_configuration")
            }
        }
        if let value = object["fast"] {
            guard let fast = value as? [String: Any] else { throw EngineError("fast must be an object", code: "invalid_configuration") }
            // Fast never activates implicitly; a configuration may only allow per-request opt-in.
            if (fast["enabled"] as? Bool) == true { warn("ignoring fast.enabled: fast is a per-request opt-in (allow_opt_in)") }
            if let raw = fast["allow_opt_in"] {
                guard let allow = raw as? NSNumber, CFGetTypeID(allow) == CFBooleanGetTypeID() else {
                    throw EngineError("fast.allow_opt_in must be a boolean", code: "invalid_configuration")
                }
                if allow.boolValue {
                    guard let tau = fast["late_miss_skip_tau"] as? NSNumber, CFGetTypeID(tau) != CFBooleanGetTypeID(),
                        tau.doubleValue > 0, tau.doubleValue <= 1, result.wholeExecutionSchedule
                    else {
                        throw EngineError(
                            "fast.allow_opt_in requires late_miss_skip_tau in (0, 1] and whole decode", code: "invalid_configuration")
                    }
                    result.fastOptInTau = tau.doubleValue
                }
            }
        }
        // Every complete file-frontier conversation must pass the full simultaneous-owner memory admission.
        if result.wholeExecutionSchedule && result.persistentConversation && result.conversationFrontierDiskBytes > 0 {
            _ = try result.memoryAdmission()
        }
        return result
    }
}
