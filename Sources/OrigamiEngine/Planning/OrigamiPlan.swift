import CExactIO
import Darwin
import Foundation
import Metal

/// The Mac Origami runs on, as far as planning needs it.
public struct OrigamiHost: Sendable {
    public let physicalMemory: UInt64
    /// Effective planning ceiling: a positive explicit macOS wired limit (within OS headroom), otherwise Metal's recommendation.
    public let workingSet: UInt64
    public let gpuName: String
    public let bfloatMatrices: Bool
    public let notes: [String]

    public static func current() throws -> OrigamiHost {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EngineError("No Metal GPU found; Origami needs an Apple-silicon Mac", code: "device_unsupported")
        }
        var notes = try ModelContract.validate(device: device)
        var bfloat = ModelContract.supportsBFloatMatrices(device)
        // Runs the M1 kernels on a newer Mac, to compare or to rule the bfloat16 kernels out.
        if bfloat, ProcessInfo.processInfo.environment["ORIGAMI_NO_BFLOAT_MATRICES"] != nil {
            bfloat = false
            notes.append("bfloat16 matrix kernels disabled by ORIGAMI_NO_BFLOAT_MATRICES")
        }
        let physical = ProcessInfo.processInfo.physicalMemory
        // Metal may cache its recommendation in a long-lived front process. An explicit sysctl ceiling is read
        // fresh so an authorized increase becomes usable after model reload; zero keeps the advertised default.
        var raw: Int64 = 0
        var size = MemoryLayout<Int64>.size
        let status = sysctlbyname("iogpu.wired_limit_mb", &raw, &size, nil, 0)
        let signed = size == MemoryLayout<Int32>.size ? Int64(Int32(truncatingIfNeeded: raw)) : raw
        let explicit =
            status == 0 && [MemoryLayout<Int32>.size, MemoryLayout<Int64>.size].contains(size)
                && signed > 0 && signed <= Int64(Int32.max) ? UInt64(signed) : nil
        let working =
            explicit.map { min($0 * WiredMemoryPolicy.mebibyte, WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: physical)) }
            ?? device.recommendedMaxWorkingSetSize
        return OrigamiHost(
            physicalMemory: physical, workingSet: working,
            gpuName: device.name, bfloatMatrices: bfloat, notes: notes)
    }
    public init(physicalMemory: UInt64, workingSet: UInt64, gpuName: String, bfloatMatrices: Bool, notes: [String] = []) {
        self.physicalMemory = physicalMemory
        self.workingSet = workingSet
        self.gpuName = gpuName
        self.bfloatMatrices = bfloatMatrices
        self.notes = notes
    }
}

/// Where expert records can be read from, as decided by `origami setup` and the stores found at start.
public struct OrigamiStorage: Sendable {
    /// Folder of a complete pack, if one is usable.
    public var packFolder: URL?
    /// True when the pack sits on the same physical drive as the model: no hedging between them.
    public var packSharesModelDrive = false
    /// Folder of the hot copy of the most-used experts (`HotCopyKeeper` builds it while serving), when it is on.
    public var hotCopyFolder: URL?
    /// More expert copies read beside the pack, off the model's drive: packs (fastest first), then copies of the
    /// most-used experts. Each read goes to the least busy copy holding its expert.
    public var copyFolders: [URL] = []
    /// Physical drives the pack and its copies sit on.
    public var packDrives = 1
    /// Measured read rates for the model drive and the pack drive (`DriveSpeed`); nil uses the neutral curve.
    public var calibration: ExpertSourceCalibration?
    public init(
        packFolder: URL? = nil, packSharesModelDrive: Bool = false, hotCopyFolder: URL? = nil, calibration: ExpertSourceCalibration? = nil
    ) {
        self.packFolder = packFolder
        self.packSharesModelDrive = packSharesModelDrive
        self.hotCopyFolder = hotCopyFolder
        self.calibration = calibration
    }
}

/// Memory, in bytes. `cap` is the most the engine will use; `start` is what it uses now
/// (automatic mode fits a busy Mac; explicit mode starts lower only under real pressure).
/// `floor` is the available-memory reserve used by automatic mode.
public struct OrigamiMemory: Sendable, Codable {
    public let cap, start, floor, minimum: UInt64
    public init(cap: UInt64, start: UInt64, floor: UInt64, minimum: UInt64) {
        self.cap = cap
        self.start = start
        self.floor = floor
        self.minimum = minimum
    }
}

/// Turns a model, a Mac and a few user settings into the engine configuration. The
/// defaults are the v1 release settings: every optimization on, the expert cache fitted
/// to this Mac's memory, and read policies chosen from the drives that are present.
public struct OrigamiPlan: Sendable {
    public static let contexts = ConversationGeometry.contexts
    /// Any context up to this many tokens; the plan serves it on the smallest tier that holds it.
    public static let maximumContext = 262_144
    public let configuration: RuntimeConfiguration
    public let memory: OrigamiMemory
    /// The transient allowance a memory_limit holds besides the owners' budget (`memory`).
    public static let limitMargin: UInt64 = 1_000_000_000
    /// The least memory_limit this plan starts with: the owners' least budget and the transient allowance.
    public var minimumLimit: UInt64 { memory.minimum + Self.limitMargin }
    public let notes: [String]

    /// - Parameters:
    ///   - memoryLimit: user cap on the engine's memory, if any; above the automatic share up to Metal's working set,
    ///     leaving the floor to macOS and other apps.
    ///   - expertMemory: user cap on the cached experts, if any; the rest of the memory is left to other apps.
    ///   - promptCache: folder for the prompt cache and conversation frontiers (any drive).
    ///   - overrides: optional `execution` switches (debugging); unknown keys are ignored with a warning.
    ///   - availableDisk: free bytes on the prompt-cache drive (256K only); nil probes the drive.
    public static func make(
        model: ModelSource, host: OrigamiHost, context requested: Int, memoryLimit: UInt64? = nil, expertMemory: UInt64? = nil,
        allowFast: Bool = true, storage: OrigamiStorage = .init(), promptCache: URL,
        overrides: [String: Any] = [:], availableNow: UInt64 = availableMemoryBytes(),
        footprintNow: UInt64 = processFootprintBytes(), availableDisk: UInt64? = nil,
        pressureNow: MemoryPressureLevel? = MemoryPressureLevel.current(),
        tier forced: Int? = nil, inMemory forcedInMemory: Bool = false,
        warn: (String) -> Void = { _ in }
    ) throws -> OrigamiPlan {
        ModelLayout.use(model)
        var notes = host.notes
        guard (1...maximumContext).contains(requested) else {
            throw EngineError("context \(requested) is not a token count up to \(maximumContext)", code: "invalid_configuration")
        }
        // A context above 32K stays in memory when this Mac has room for it (256K too), and goes to the file when it has not.
        if forced == nil, requested > 32768 {
            let candidates =
                ConversationGeometry.inMemoryContexts.filter { $0 >= requested }.map { ($0, true) }
                + [(ConversationGeometry.maximumFileContext, true), (ConversationGeometry.maximumFileContext, false)]
            for (tier, inMemory) in candidates {
                do {
                    return try make(
                        model: model, host: host, context: requested, memoryLimit: memoryLimit, expertMemory: expertMemory,
                        allowFast: allowFast, storage: storage, promptCache: promptCache, overrides: overrides,
                        availableNow: availableNow, footprintNow: footprintNow, availableDisk: availableDisk,
                        pressureNow: pressureNow,
                        tier: tier, inMemory: inMemory, warn: warn)
                } catch let error as EngineError where error.code == "insufficient_memory" && inMemory { continue }
            }
        }
        let context = forced ?? contexts.first { $0 >= requested }!
        // Only the 262,144 tier can keep its pages in a file, and it does unless planned in memory.
        let fileBacked = context == ConversationGeometry.maximumFileContext && !forcedInMemory
        if context != requested { notes.append("context \(requested) tokens, served on the \(context / 1024)K tier") }
        if fileBacked {
            // The key/value cache lives in a file under the prompt cache; the fixed owners (file metadata, 1 GiB
            // staging, the 256K scope) only leave a useful expert cache from 32 GB (plan_256k_file_backed check).
            guard host.physicalMemory >= 32 << 30 else {
                throw EngineError(
                    "A context longer than this Mac keeps in memory goes to a file, which needs a Mac with 32 GB of RAM or more (this one has \(gigabytes(host.physicalMemory))); use a shorter context",
                    code: "insufficient_memory")
            }
            if let drive = PromptCache.missingDrive(promptCache) {
                throw EngineError(
                    "context_dir \(promptCache.path) is on the drive \(drive), which is not connected; connect it or set context_dir to another folder",
                    code: "insufficient_disk")
            }
            let needed = ConversationGeometry.fileDiskBytes + ConversationGeometry.durableDiskBytes + 10 * 1_000_000_000
            let free = availableDisk ?? PromptCache.freeDiskBytes(at: promptCache)
            guard free >= needed else {
                throw EngineError(
                    "A context longer than this Mac keeps in memory goes to a file, which needs \(gigabytes(needed)) free on the drive of \(promptCache.path) for it and the saved states (\(gigabytes(free)) free); use a shorter context or free disk",
                    code: "insufficient_disk")
            }
            notes.append(
                "256K context: key/value cache in a file under \(promptCache.path); decode runs without register attention, early demand, token plan, pre-routing, read-ahead, elastic residency and tail replicas (in-memory contexts only)"
            )
        }

        // Memory: leave 30% of RAM to macOS and other apps (at least 4 GB, at most 24 GB, so a 96 GB
        // Mac decodes with every expert in memory), and stay inside the machine's working-set ceiling.
        // Automatic mode fits current free memory; explicit mode targets the full chosen allowance.
        let gb: UInt64 = 1_000_000_000
        let floor = host.physicalMemory - WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: host.physicalMemory)
        let automatic = automaticMemory(host: host)
        var cap = automatic
        if let memoryLimit {
            // The user may give Origami more than the automatic share, up to Metal's working set and the floor.
            cap = min(memoryLimit, max(automatic, memoryCeiling(host: host)))
            if cap > automatic {
                notes.append(
                    "memory_limit \(gigabytes(cap)) is above the automatic \(gigabytes(automatic)): macOS and other apps keep \(gigabytes(host.physicalMemory - cap)); if the Mac starts swapping, lower it"
                )
            }
        }
        // Explicit memory_limit includes the transient allowance: the whole process stays inside the
        // selected envelope. Automatic plans retain their existing owner-budget convention.
        let margin = Self.limitMargin
        let explicitMargin = memoryLimit == nil ? 0 : margin
        var ownerCap = cap > explicitMargin ? cap - explicitMargin : 0
        guard ownerCap > 0 else { throw EngineError("memory_limit is too small for Origami", code: "insufficient_memory") }
        let free = availableNow + footprintNow
        let start =
            memoryLimit == nil
            ? min(ownerCap, free > floor ? free - floor : 0)
            : (pressureNow?.isElevated == true ? 0 : ownerCap)

        var object = try baseObject(
            context: context, fileBacked: fileBacked, storage: storage, promptCache: promptCache, allowFast: allowFast,
            bfloatMatrices: host.bfloatMatrices, overrides: overrides)
        if ModelLayout.ggufExperts != nil {
            notes.append(
                "experts: read from the GGUF in its own encodings (\(ModelLayout.ggufExperts!.typeSummary)); "
                    + "decode runs without packs and MTP")
        }
        if ModelLayout.group == 32 {
            // Group 32 stores twice the scales and biases: 0.625 instead of 0.5625 bytes per 4-bit weight.
            var execution = object["execution"] as! [String: Any]
            execution["resident_tensor_bytes"] = (execution["resident_tensor_bytes"] as! Int) / 9 * 10
            object["execution"] = execution
        }
        // The margin covers transient Metal and Foundation allocations outside the ledger.
        object["process_ceiling_bytes"] = ownerCap + margin
        object["transient_margin_bytes"] = margin
        // Size the caches: fixed owners first (from a minimal cache), then the rest goes to experts.
        var execution = object["execution"] as! [String: Any]
        execution["expert_cache_slots"] = minimumSlots
        execution["prefill_expert_cache_slots"] = minimumSlots
        object["execution"] = execution
        // A refusal that even the largest limit this Mac allows (with nothing else running) cannot lift says so, and what
        // would fit: a smaller context, or at the smallest one a Mac with more memory (the other models need more still).
        let ceiling = memoryCeiling(host: host)
        func beyondThisMac(_ needs: String) -> EngineError? {
            guard memoryLimit != ceiling else {
                return cap >= ceiling ? beyondThisMacError(needs, ceiling: ceiling, smallest: context <= 8192) : nil
            }
            let fits =
                (try? make(
                    model: model, host: host, context: requested, memoryLimit: ceiling, expertMemory: expertMemory,
                    allowFast: allowFast, storage: storage, promptCache: promptCache, overrides: overrides,
                    availableNow: host.physicalMemory, footprintNow: 0, availableDisk: availableDisk, pressureNow: nil,
                    tier: forced, inMemory: forcedInMemory)) != nil
            return fits ? nil : beyondThisMacError(needs, ceiling: ceiling, smallest: context <= 8192)
        }
        var configuration: RuntimeConfiguration
        do { configuration = try RuntimeConfiguration.load(object: object, warn: warn) } catch let error as EngineError
            where error.code == "insufficient_memory"
        {
            if let refusal = beyondThisMac("more memory") { throw refusal }
            throw EngineError(
                "This Mac has too little memory for Origami at \(requested / 1024)K context (budget \(gigabytes(cap))); "
                    + "raise memory_limit or close other apps" + (context > 8192 ? ", or use a smaller context" : ""),
                code: "insufficient_memory")
        }
        let fixed = try configuration.memoryAdmission()
        let slab = UInt64(slabSlots * PackStore.expertBytes)
        func slots(_ budget: UInt64, less fixed: UInt64) -> Int {
            budget > fixed ? Int((budget - fixed) / slab) * slabSlots : 0
        }
        let minimum = fixed.coldFixedBytes + UInt64(minimumSlots * PackStore.expertBytes) + slab
        guard ownerCap >= minimum else {
            if let refusal = beyondThisMac("a budget of \(gigabytes(minimum + margin))") { throw refusal }
            throw EngineError(
                "This Mac has too little memory for Origami at \(requested / 1024)K context: it needs a budget of "
                    + "\(gigabytes(minimum + explicitMargin)) and can use \(gigabytes(cap))"
                    + (memoryLimit == nil ? "" : " (memory_limit)"), code: "insufficient_memory")
        }
        var startBudget = max(start, minimum)
        if start < minimum {
            notes.append(
                memoryLimit != nil && pressureNow?.isElevated == true
                    ? "macOS reports memory pressure; starting with the smallest cache (\(gigabytes(minimum))) and recovering toward memory_limit when pressure clears"
                    : "only \(gigabytes(availableNow)) of memory is free; starting with the smallest cache (\(gigabytes(minimum))). Expect slow prefill until apps free memory"
            )
        }
        // Both caches are sized for the cap; each phase then takes what the current budget allows
        // (ConversationPhaseBudget), so they grow as memory frees up and shrink under pressure.
        // Decode may borrow the key/value arena a conversation has not reached yet; prefill and expert_memory's cap do not.
        var decode = min(maximumSlots, slots(ownerCap, less: fixed.lentArenaFixedBytes))
        if let expertMemory {
            // The experts' share: at least what prefill streams through; the rest of the memory goes back to other apps.
            decode = min(decode, max(minimumSlots + slabSlots, Int(expertMemory / slab) * slabSlots))
            ownerCap = min(ownerCap, max(minimum, fixed.noFrontierFixedBytes + UInt64(decode / slabSlots) * slab))
            cap = ownerCap + explicitMargin
            object["process_ceiling_bytes"] = ownerCap + margin
            startBudget = min(startBudget, ownerCap)
        }
        // The cache while the longest prompt scope (8,192 rows) is read: what the cap leaves after that scope's state and
        // working buffers. A shorter prompt keeps more (SerialExecutor.prepareUpcomingInput sizes it from its real rows).
        let prefill = max(minimumSlots, min(slots(ownerCap, less: fixed.coldFixedBytes) - slabSlots, decode))
        execution["expert_cache_slots"] = max(decode, prefill)
        execution["prefill_expert_cache_slots"] = prefill
        // Wiring keeps the decode cache out of the compressor; only when the rest of the Mac keeps plenty, and only with
        // the preallocated slabs of 8K/32K. A setting given in `execution` wins.
        if overrides["wire_decode_slabs"] == nil {
            execution["wire_decode_slabs"] = host.physicalMemory >= cap + 8 * gb && !fileBacked
        }
        object["execution"] = execution
        configuration = try RuntimeConfiguration.load(object: object)
        configuration.startUsableBytes = startBudget
        configuration.memoryBudgetMode = memoryLimit == nil ? .automatic : .explicit
        configuration.availableFloorBytes = floor
        configuration.hotCopyFolder = storage.hotCopyFolder
        return OrigamiPlan(
            configuration: configuration, memory: OrigamiMemory(cap: cap, start: startBudget, floor: floor, minimum: minimum), notes: notes)
    }

    /// RAM a context of `context` tokens keeps resident on its tier: every key/value page in memory, or for the
    /// file-backed tier its page metadata and 1 GiB of staging.
    public static func contextMemoryBytes(context: Int, fileBacked: Bool? = nil) -> UInt64 {
        guard let tier = contexts.first(where: { $0 >= context }),
            let geometry = try? ConversationGeometry(context: tier, fileBacked: fileBacked)
        else { return 0 }
        return geometry.arenaBytes + (geometry.fileBacked ? 1 << 30 : 0)
    }
    /// The automatic budget: 30% of RAM left to macOS and other apps (at least 4 GB, at most 24 GB), within the working set.
    public static func automaticMemory(host: OrigamiHost) -> UInt64 {
        let gb: UInt64 = 1_000_000_000
        let reserve = max(4 * gb, min(host.physicalMemory * 3 / 10, 24 * gb))
        return min(host.physicalMemory > reserve ? host.physicalMemory - reserve : 0, host.workingSet)
    }
    /// The most memory_limit may give Origami: Metal's working set, leaving the floor to macOS and other apps.
    public static func memoryCeiling(host: OrigamiHost) -> UInt64 {
        min(host.workingSet, WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: host.physicalMemory))
    }

    /// Every expert stays in memory once read, in prefill too: expert copies on disk only speed up the first reads.
    public var allResident: Bool { configuration.experts.effectivePrefillCacheSlots >= Self.maximumSlots }

    static let slabSlots = 8
    /// One layer's experts plus the read window: the least prefill can stream through.
    static let minimumSlots = ConversationPhaseBudget.minimumSlots
    /// Every expert of the model (of the layout `make` last set: a pruned GGUF has fewer).
    public static var maximumSlots: Int { ModelContract.layers * ModelLayout.routedExperts }

    /// Qwen's refusal when its least budget (`needs`) is above the most this Mac allows (`ceiling`): the same opening as
    /// the family engines' (`FamilyRunner.beyondThisMac`), then what would fit.
    static func beyondThisMacError(_ needs: String, ceiling: UInt64, smallest: Bool) -> EngineError {
        EngineError(
            "this model does not fit this Mac at this context: it needs \(needs) and this Mac allows at most \(gigabytes(ceiling)); "
                + (smallest
                    ? "8K is its smallest context, so it needs a Mac with more memory (GLM and DeepSeek need more still)"
                    : "use a smaller context"),
            code: "insufficient_memory")
    }
    public static func gigabytes(_ bytes: UInt64) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }

    /// Where this Mac starts keeping the context's key/value cache in a file on disk, and what that costs.
    public struct ContextPlan: Sendable {
        /// Requests may use up to `served` tokens; the geometry behind them is `tier`.
        public let served, tier: Int
        /// The largest context this Mac keeps in memory with these settings (0 when no in-memory tier fits).
        public let inMemory: Int
        public var paged: Bool { served > inMemory }
        /// What a paged context costs: the M2 Pro 32 GB measurements, scaled to this size where it says "about".
        public var note: String? {
            guard paged else { return nil }
            let c = Double(served)
            let k = served / 1024
            // The 259K run read its prompt at about 80 tokens/s falling linearly to 55, and decoded at 2.6 tok/s after a
            // 30K prompt and 1.8 at 259K; a saved conversation is about 26.2 KB a token.
            let minutes = 259_200.0 / 25 * log(80 / (80 - 25 * c / 259_200)) / 60
            let decode = 2.6 - 0.8 * max(0, c - 30_000) / 229_200
            // This size's own figures only when it is not the measured 256K.
            let scaled =
                k >= 256
                ? ("", "")
                : (String(format: ", about %.1f at %dK", decode, k), String(format: ", a %dK one about %.0f", k, max(1, minutes)))
            return String(
                format: "Above %dK on this Mac the context's key/value cache is kept in a file on disk, not in memory. "
                    + "Measured on an M2 Pro 32 GB: replies at 1.8 tokens/s at 256K and 1.5–2.6 after 30K (5.6–6.3 in memory)%@; "
                    + "a full 256K prompt took 67 minutes to read%@; it needs about 37 GB free on the data folder's drive "
                    + "(52 GB at the peak of a full 256K conversation), and each reply's save is up to %.1f GB (6.8 GB at 259K tokens, 40–115 s).",
                inMemory / 1024, scaled.0, scaled.1, c * 26_200 / 1e9)
        }
        public var report: [String: Any] {
            ["served": served, "tier": tier, "in_memory_up_to": inMemory, "paged": paged, "note": note ?? NSNull()]
        }
    }

    /// The context plan for these settings: the in-memory tiers are planned against this Mac's memory to find where paging starts.
    public static func contextPlan(
        model: ModelSource, host: OrigamiHost, context: Int, memoryLimit: UInt64? = nil, expertMemory: UInt64? = nil,
        storage: OrigamiStorage = .init(), promptCache: URL,
        availableNow: UInt64 = availableMemoryBytes(), footprintNow: UInt64 = processFootprintBytes()
    ) -> ContextPlan {
        let served = min(max(1, context), maximumContext)
        // The largest tier this Mac plans in memory: 8K up to 256K.
        let inMemory =
            (ConversationGeometry.inMemoryContexts + [ConversationGeometry.maximumFileContext]).reversed().first { size in
                (try? make(
                    model: model, host: host, context: size, memoryLimit: memoryLimit, expertMemory: expertMemory, storage: storage,
                    promptCache: promptCache,
                    availableNow: availableNow, footprintNow: footprintNow, tier: size, inMemory: true)) != nil
            } ?? 0
        // As `make` serves it: in memory up to what fits here, the file tier above.
        let tier = served <= inMemory ? contexts.first { $0 >= served }! : ConversationGeometry.maximumFileContext
        return ContextPlan(served: served, tier: tier, inMemory: inMemory)
    }

    /// The v1 release configuration for `context`, with the read policies that fit the drives present.
    package static func baseObject(
        context: Int, fileBacked: Bool? = nil, storage: OrigamiStorage, promptCache: URL, allowFast: Bool,
        bfloatMatrices: Bool, overrides: [String: Any]
    ) throws -> [String: Any] {
        let fileBacked = fileBacked ?? (context == ConversationGeometry.maximumFileContext)
        var execution = try JSONSerialization.jsonObject(with: Data(releaseExecution.utf8)) as! [String: Any]
        // The fused compact prefill attention (bit-identical to the split path at every key count, checked to 262,144
        // keys) on every tier from 32K up whose key/value cache is in memory.
        if context >= 32768 && !fileBacked {
            execution["fused_selected_prefill_attention"] = true
            execution["compact_pv_grouped"] = true
        }
        if fileBacked {
            // RuntimeConfiguration and SerialExecutor admit these on in-memory tiers only (and what depends on them; the plan's
            // first load reads this object before the wiring is decided below).
            for key in [
                "register_qsa_decode", "ragged_prefill_scope", "decode_early_demand", "decode_preencoded_layers", "decode_token_plan",
                "decode_next_layer_prerouting", "decode_prerouting_read_hygiene", "decode_prerouting_slot_override",
                "expert_prefill_read_ahead", "expert_prefill_layer_command", "preallocate_decode_slabs", "wire_decode_slabs",
                "elastic_decode_residency", "expert_tail_replicas",
            ] {
                execution[key] = false
            }
            execution.removeValue(forKey: "expert_layer_waves")  // waves belong to the layer command
        }
        // Two drives: route each read to the drive that finishes first and hedge slow tail reads.
        // One drive: a pack saves eight of nine reads per expert, so it always wins; hedging a
        // drive against itself only adds load.
        let twoDrives = storage.packFolder != nil && !storage.packSharesModelDrive
        if !twoDrives {
            execution["expert_prefill_source_policy"] = "packedPreferred"
            execution["expert_decode_source_policy"] = "packedPreferred"
            execution["expert_tail_replicas"] = false
            execution["decode_ple_internal_race"] = false
            execution["expert_source_online_calibration"] = false
            execution.removeValue(forKey: "expert_source_online_bounds")
        }
        // Two drives: this Mac's measured curves, else the neutral one (never another Mac's numbers).
        if twoDrives {
            let base = storage.calibration ?? .neutral
            execution["expert_source_calibration"] = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(storage.packDrives > 1 ? base.spread(over: storage.packDrives) : base))
        } else if let calibration = storage.calibration {
            execution["expert_source_calibration"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(calibration))
        }
        // Copies on several drives: the packed queue reads them all at once.
        if storage.packDrives > 1 { execution["expert_pack_drives"] = storage.packDrives }
        if !bfloatMatrices {
            execution["quantized_mma_bf16"] = false
            execution["grouped_expert_mma_min_rows"] = 0
        }
        for (key, value) in overrides { execution[key] = value }

        let geometry = try ConversationGeometry(context: context, fileBacked: fileBacked)
        // 256K: the durable quota the file-KV contract pins (RuntimeConfiguration requires it exactly).
        let snapshot =
            try geometry.fileBacked
            ? ConversationGeometry.durableDiskBytes
            : PagedCheckpointFormat.layout(capacity: context, tokens: context, hasLogits: true).bytes * 2
        let prefix: [String: Any] = [
            "persistent_conversation": true, "memory_bytes": geometry.conversationBytes,
            "durable_format": "paged-v2", "disk_bytes": snapshot, "disk_directory": promptCache.standardizedFileURL.path,
            "frontier_disk_bytes": geometry.frontierDiskBytes, "frontier_layout": "dense-c256-v1",
            "frontier_async_writes": true, "frontier_write_staging_slots": 64,
        ]
        var object: [String: Any] = [
            "context_limit": context, "execution": execution,
            "prefill": ["compute_tile": 256, "read_scope": 8192],
            "prefix_cache": prefix,
            "quotas": [
                "snapshot_bytes": snapshot,
                "spill_bytes": snapshot + geometry.frontierDiskBytes + (geometry.fileBacked ? ConversationGeometry.fileDiskBytes : 0),
            ],
            "storage": ["integrity": "sampled"],
            "fast": ["allow_opt_in": allowFast, "late_miss_skip_tau": 1.0],
        ]
        if !fileBacked && context == ConversationGeometry.maximumFileContext { object["context_in_memory"] = true }
        return object
    }
}

/// Host memory counters for planning and the governor.
public func availableMemoryBytes() -> UInt64 { origami_available_memory() }
public func processFootprintBytes() -> UInt64 { origami_physical_footprint() }
public func swapoutBytes() -> UInt64 { origami_swapout_bytes() }
