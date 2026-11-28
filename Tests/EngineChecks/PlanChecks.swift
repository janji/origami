import Foundation
import OrigamiEngine

/// The memory plan on simulated Macs from 16 GB to 192 GB, both contexts, with and without the
/// bfloat16 matrix kernels (M1): 32 GB keeps v1's caches, bigger Macs grow the expert cache past
/// v1's 8,192 slots and keep it through prefill, 96 GB decodes with every expert in memory and
/// 128 GB prefills with all of them too. No model needed.
func planFitsMacSizes() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-plan-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let gib: UInt64 = 1 << 30
    let gb: UInt64 = 1_000_000_000
    let all = OrigamiPlan.maximumSlots
    for context in [8192, 32768] {
        for bfloat in [true, false] {
            var previous = 0
            for size in [16, 24, 32, 64, 96, 128, 192] as [UInt64] {
                let ram = size * gib
                let host = OrigamiHost(
                    physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated \(size) GB", bfloatMatrices: bfloat)
                let plan: OrigamiPlan
                do {
                    plan = try OrigamiPlan.make(
                        model: model, host: host, context: context, promptCache: root, availableNow: ram, footprintNow: 0)
                } catch let error as EngineError where error.code == "insufficient_memory" {
                    try require(size <= 16 && context == 32768, "\(size) GB at \(context): \(error)")
                    continue
                }
                let experts = plan.configuration.experts
                let decode = experts.cacheSlots
                let prefill = experts.effectivePrefillCacheSlots
                let label = "\(size) GB at \(context) (bfloat \(bfloat)): decode \(decode), prefill \(prefill), cap \(plan.memory.cap)"
                try require(decode <= all && decode % 8 == 0 && (528...decode).contains(prefill) && prefill % 8 == 0, label)
                try require(plan.memory.cap <= host.workingSet && plan.memory.cap + plan.memory.floor <= ram, label + ": leaves too little")
                try require(experts.wireDecodeSlabs == (ram >= plan.memory.cap + 8 * gb), label + ": wiring rule")
                try require(
                    plan.configuration.quantizedMMABF16 == bfloat && (bfloat || plan.configuration.groupedExpertMMAMinRows == 0),
                    label + ": kernels")
                try require(decode >= previous, label + ": shrank with more memory")
                previous = decode
                switch size {
                case 32: try require(decode <= 8192 && prefill >= (context == 8192 ? 4840 : 4416), label + ": below v1's 32 GB caches")
                case 64: try require(decode > 8192 && decode < all && prefill > 8192 && !plan.allResident, label + ": 64 GB")
                case 96: try require(decode == all && prefill >= all * 97 / 100, label + ": 96 GB")
                case 128, 192: try require(decode == all && prefill == all && plan.allResident, label + ": not all resident")
                default: break
                }
            }
        }
    }
}

/// ROADMAP R6, small Macs: on simulated 8, 16 and 24 GB Macs at 8K and 32K (all memory free, and half of it), Qwen's
/// plan either starts within this Mac's limits or refuses, and a refusal tells the truth: "raise memory_limit or close
/// other apps" only when a plan at the largest limit this Mac allows (nothing else running) would start, else "this
/// model does not fit this Mac at this context" with what would (a smaller context; at 8K, a Mac with more memory).
/// Today: 8 GB refuses, 16 GB starts at 8K only, 24 GB starts at both. GLM's and DeepSeek's start refusal says the same
/// at each Mac's largest limit. No model needed.
func planSmallMacs() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-small-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let beyond = "this model does not fit this Mac at this context"
    var started: [String] = []
    for size in [8, 16, 24] as [UInt64] {
        let ram = size << 30
        let host = OrigamiHost(physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated \(size) GB", bfloatMatrices: true)
        let ceiling = OrigamiPlan.memoryCeiling(host: host)
        for context in [8192, 32768] {
            for available in [ram, ram / 2] {
                let label = "\(size) GB at \(context / 1024)K, \(available >> 30) GB free"
                do {
                    let plan = try OrigamiPlan.make(
                        model: model, host: host, context: context, promptCache: root, availableNow: available, footprintNow: 0)
                    try require(
                        plan.memory.cap <= host.workingSet && plan.memory.cap + plan.memory.floor <= ram
                            && plan.memory.start >= plan.memory.minimum && plan.configuration.experts.cacheSlots >= 528,
                        "\(label): starts outside this Mac's limits")
                    if available == ram { started.append("\(size)/\(context / 1024)K") }
                } catch let error as EngineError where error.code == "insufficient_memory" {
                    let text = error.description
                    let atCeiling = try? OrigamiPlan.make(
                        model: model, host: host, context: context, memoryLimit: ceiling, promptCache: root, availableNow: ram,
                        footprintNow: 0, pressureNow: nil)
                    if text.hasPrefix(beyond) {
                        try require(atCeiling == nil, "\(label): says it does not fit, but this Mac's largest limit starts it")
                        try require(
                            text.contains(context > 8192 ? "use a smaller context" : "a Mac with more memory")
                                && text.contains(OrigamiPlan.gigabytes(ceiling)),
                            "\(label): the refusal does not say what would fit: \(text)")
                    } else {
                        try require(
                            atCeiling != nil && text.contains("raise memory_limit"), "\(label): an unhelpful refusal: \(text)")
                    }
                }
            }
        }
    }
    try require(started == ["16/8K", "24/8K", "24/32K"], "the Mac sizes that start moved: \(started)")
    for size in [8, 16, 24] as [UInt64] {
        let most = WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: size << 30)
        try require(FamilyRunner.beyondThisMac(least: most, maximum: most) == nil, "\(size) GB: a family model that fits was refused")
        let text = FamilyRunner.beyondThisMac(least: most + 1_000_000_000, maximum: most)?.description ?? ""
        try require(
            text.hasPrefix(beyond) && text.contains(OrigamiPlan.gigabytes(most)) && text.contains("smaller context")
                && text.contains("needs less memory"), "\(size) GB: the family refusal: \(text)")
    }
}

/// A prompt's expert cache follows its real new rows (`SerialExecutor.promptCacheBytes`): under 256 rows it keeps the
/// decode cache, an 8,192-row scope gets the plan's prefill cache, and the sizes between give back only what their own
/// scope needs, in whole 256-row tiles. Every size still leaves its scope's fixed owners within the budget, and a
/// lowered budget shrinks the cache instead of failing. A simulated 32 GB Mac, no model.
func planPromptCacheFollowsRows() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prompt-cache-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let ram: UInt64 = 32 << 30
    let host = OrigamiHost(physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    for context in [8192, 32768] {
        let plan = try OrigamiPlan.make(
            model: try ModelSource(fixture: root, spans: [:]), host: host, context: context, promptCache: root,
            availableNow: ram, footprintNow: 0)
        let configuration = plan.configuration
        let experts = configuration.experts
        let usable = try configuration.memoryAdmission().usableBytes
        let resident = configuration.residentBytes + configuration.denseResidentBytes
        let geometry = try ConversationGeometry(context: context)
        func cache(_ input: Int, usable: UInt64 = usable) throws -> UInt64 {
            try SerialExecutor.promptCacheBytes(
                context: context, fileBacked: nil, input: input, residentBytes: resident, experts: experts, usable: usable)
        }
        let label = "\(context): decode \(experts.cacheSlots), prefill \(experts.effectivePrefillCacheSlots)"
        try require(
            try cache(255)
                == ConversationPhaseBudget.expertBytes(
                    context: context, input: 255, residentBytes: resident, experts: experts, usable: usable),
            label + ": a short prompt lost the decode cache")
        try require(
            try cache(8192) == experts.prefillCacheBytes && cache(20_000) == cache(8192),
            label + ": the longest scope did not get the plan's prefill cache")
        var previous = UInt64.max
        for input in [256, 600, 2_500, 7_700, 8_192] {
            let bytes = try cache(input)
            let rows = min(8192, (input + 255) / 256 * 256)
            let owners =
                experts.replicaOwnerBytes + experts.readAheadOwnerBytes + experts.layerCommandOwnerBytes + experts.reuseOwnerBytes
                + experts.decodeSlabOwnerBytes
            try require(bytes <= previous && bytes <= experts.cacheBytes, label + ": \(input) rows kept more than a shorter prompt")
            try require(
                try geometry.maximumFixedBytes(input: rows, residentBytes: resident) + owners + bytes <= usable,
                label + ": \(input) rows overcommit the budget")
            // Every scope the prompt can run reserves no more than its rounded rows were charged (the scope term: fixed
            // owners above a 1-row scope's, plus that scope's own buffers).
            let one = try WholePrefillScope.layout(tokens: 1, allowTail: true, admitSmallScope: true)?.supplementalBytes ?? 0
            let charged =
                try geometry.maximumFixedBytes(input: rows, residentBytes: 0) - geometry.maximumFixedBytes(input: 1, residentBytes: 0)
                + one
            for scope in [1, 255, 256, 300, input, rows - 1, rows] where scope <= rows {
                let layout = try WholePrefillScope.layout(tokens: scope, allowTail: true, admitSmallScope: true)
                let reserved = UInt64(max(0, scope - 256)) * 240_000 + (layout?.supplementalBytes ?? 0)
                try require(
                    reserved <= charged, label + ": a \(scope)-row scope of a \(input)-row prompt reserves \(reserved) > \(charged)")
            }
            previous = bytes
        }
        try require(try cache(600) > cache(8192), label + ": a 600-token prompt kept no more than an 8,192-row one")
        try require(try cache(600, usable: usable - 4_000_000_000) < cache(600), label + ": a lowered budget did not shrink it")
    }
}

/// ROADMAP C1': a prompt read's expert cache counts only the key/value arena the conversation holds plus what this
/// request can add (`heldArenaBytes`), not the whole context's. At 128K, a 28K conversation reading 1.5K new tokens with
/// 2,048 reply tokens keeps about 2.5 GB more cache; a conversation that fills the context keeps the old limit; and across
/// every conversation size and reply length the owners stay within the budget, the request's pages within the hold.
func planPromptCacheLendsUnusedArena() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prompt-lend-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let ram: UInt64 = 32 << 30
    let host = OrigamiHost(physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    let context = 131072
    let plan = try OrigamiPlan.make(
        model: try ModelSource(fixture: root, spans: [:]), host: host, context: context, promptCache: root,
        availableNow: ram, footprintNow: 0)
    let configuration = plan.configuration
    let experts = configuration.experts
    let usable = try configuration.memoryAdmission().usableBytes
    let resident = configuration.residentBytes + configuration.denseResidentBytes
    let geometry = try ConversationGeometry(context: context)
    let slab = UInt64(experts.slotsPerSlab * PackStore.expertBytes)
    let owners =
        experts.replicaOwnerBytes + experts.readAheadOwnerBytes + experts.layerCommandOwnerBytes + experts.reuseOwnerBytes
        + experts.decodeSlabOwnerBytes
    let perToken = geometry.pooledBytesPerToken
    func cache(_ input: Int, held: UInt64) throws -> UInt64 {
        try SerialExecutor.promptCacheBytes(
            context: context, fileBacked: nil, input: input, residentBytes: resident, experts: experts, usable: usable,
            heldArenaBytes: held)
    }
    // The limit before C1': the same room, under a ceiling that counted the whole arena.
    func before(_ input: Int, held: UInt64) throws -> UInt64 {
        let rows = min(8192, (input + 255) / 256 * 256)
        let longest = try geometry.maximumFixedBytes(input: 8192, residentBytes: 0)
        let this = try geometry.maximumFixedBytes(input: rows, residentBytes: 0)
        return try ConversationPhaseBudget.expertBytes(
            context: context, input: rows, residentBytes: resident, experts: experts, usable: usable, heldArenaBytes: held,
            ceiling: min(experts.cacheBytes, experts.prefillCacheBytes + (longest - this) / slab * slab))
    }
    let (conversation, input, reply) = (28 * 1024, 1536, 2048)
    let held = geometry.heldArenaBytes(live: UInt64(conversation) * perToken, growth: input + reply)
    let (old, new) = (try before(input, held: held), try cache(input, held: held))
    // The gain is the arena past the hold (3.59 GB less 1.10 GB), in whole slabs: about 2.5 GB.
    let gain = Double(new) - Double(old)
    try require(
        new - old == (geometry.arenaBytes - held) / slab * slab && gain >= 1.7e9 && gain <= 2.6e9,
        String(format: "28K + 1.5K + 2,048 at 128K: %.2f GB before, %.2f GB after", Double(old) / 1e9, Double(new) / 1e9))
    let full = geometry.heldArenaBytes(live: UInt64(context - input) * perToken, growth: input + reply)
    try require(
        full == geometry.arenaBytes && (try cache(input, held: full)) == before(input, held: full)
            && (try cache(input, held: geometry.arenaBytes)) == before(input, held: geometry.arenaBytes),
        "a conversation filling the context changed the limit")
    for conversation in stride(from: 0, through: context, by: 4096) {
        for input in [256, 1536, 8192] where conversation + input <= context {
            // Replies of a few tokens, 2,048, and one that runs to the context limit (past any estimate).
            for reply in [1, 2048, context - conversation - input] {
                let live = UInt64(conversation) * perToken
                let held = geometry.heldArenaBytes(live: live, growth: input + reply)
                let bytes = try cache(input, held: held)
                let rows = min(8192, (input + 255) / 256 * 256)
                let label = "\(conversation) + \(input) + \(reply)"
                try require(bytes >= (try before(input, held: held)) && bytes <= experts.cacheBytes, label + ": the limit fell")
                try require(
                    try geometry.maximumFixedBytes(input: rows, residentBytes: resident, heldArenaBytes: held) + owners + bytes
                        <= usable, label + ": owners overcommit the budget")
                try require(
                    held >= min(geometry.arenaBytes, live + UInt64(input + reply) * perToken), label + ": the pages outgrow the hold")
            }
        }
    }
}

/// The 256K tier (file-backed KV): refused below 32 GB and without about 37 GB of free disk; from 32 GB it
/// plans its own, smaller expert cache with the 8K/32K-only decode switches off and the file-KV disk quotas
/// pinned. `planFitsMacSizes` above pins the 8K/32K plans, which this tier must not move. No model needed.
func planFits256K() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-plan256-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let gib: UInt64 = 1 << 30
    let gb: UInt64 = 1_000_000_000
    let context = ConversationGeometry.maximumFileContext
    func make(_ size: UInt64, disk: UInt64 = 100_000_000_000, context: Int = ConversationGeometry.maximumFileContext, file: Bool = true)
        throws -> OrigamiPlan
    {
        let ram = size * gib
        let host = OrigamiHost(physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated \(size) GB", bfloatMatrices: true)
        return try OrigamiPlan.make(
            model: model, host: host, context: context, promptCache: root, availableNow: ram, footprintNow: 0, availableDisk: disk,
            tier: file ? context : nil)
    }
    // Without a choice, 256K keeps its pages in memory where the Mac has room (32 GB does), with every decode switch on.
    let resident = try make(32, disk: 20 * gb, file: false)
    try require(
        resident.configuration.contextLimit == context && resident.configuration.contextInMemory
            && !resident.configuration.fileBackedContext
            && resident.configuration.decodeEarlyDemand && resident.configuration.decodePreencodedLayers, "256K in memory on 32 GB")
    // A next-layer guess closer than 0.1 to the eleventh logit is not read early (R15: 12% fewer bytes, same speed).
    try require(resident.configuration.decodePreroutingMargin == 0.1, "release pre-routing margin")
    for size in [16, 24] as [UInt64] {
        do {
            _ = try make(size)
            throw EngineError("\(size) GB planned 256K")
        } catch let error as EngineError where error.code == "insufficient_memory" {
            try require(error.description.contains("32 GB"), "\(size) GB: \(error)")
        }
    }
    do {
        _ = try make(32, disk: 20 * gb)
        throw EngineError("256K planned with 20 GB of free disk")
    } catch let error as EngineError where error.code == "insufficient_disk" {}
    var previous = 0
    for size in [32, 64, 96, 128] as [UInt64] {
        let plan = try make(size)
        let configuration = plan.configuration
        let experts = configuration.experts
        let decode = experts.cacheSlots
        let prefill = experts.effectivePrefillCacheSlots
        let label = "\(size) GB at 256K: decode \(decode), prefill \(prefill), cap \(plan.memory.cap)"
        try require(
            configuration.contextLimit == context && configuration.prefixDiskBytes == ConversationGeometry.durableDiskBytes,
            label + ": geometry")
        try require(decode >= 528 && decode % 8 == 0 && (528...decode).contains(prefill) && prefill % 8 == 0 && decode >= previous, label)
        try require(plan.memory.cap + plan.memory.floor <= size * gib, label + ": leaves too little")
        try require(
            !configuration.registerQSADecode && !configuration.raggedPrefillScope && !configuration.decodeEarlyDemand
                && !configuration.elasticDecodeResidency && !experts.prefillReadAhead && !experts.preallocateDecodeSlabs
                && !experts.tailReplicas,
            label + ": 8K/32K-only switches")
        // File metadata, staging and the 256K scope leave fewer experts than 32K on the same Mac (equal once every expert fits).
        let at32K = try make(size, context: 32768).configuration.experts.cacheSlots
        try require(decode <= at32K && (size > 32 || decode < at32K), label + ": not below the 32K cache (\(at32K))")
        previous = decode
    }
}

/// Any context up to 262,144 tokens is served on the smallest tier that holds it; past what this Mac keeps in
/// memory (its largest in-memory tier that plans) the context plan pages to disk and carries the cost note. No model needed.
func planContextAnyLength() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-plan-any-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let gib: UInt64 = 1 << 30
    func host(_ size: UInt64) -> OrigamiHost {
        OrigamiHost(physicalMemory: size * gib, workingSet: size * gib * 3 / 4, gpuName: "simulated \(size) GB", bfloatMatrices: true)
    }
    func make(_ context: Int, limit: UInt64? = nil, tier: Int? = nil) throws -> OrigamiPlan {
        try OrigamiPlan.make(
            model: model, host: host(32), context: context, memoryLimit: limit, promptCache: root, availableNow: 32 * gib, footprintNow: 0,
            availableDisk: 100_000_000_000, pressureNow: .normal, tier: tier)
    }
    // A context above 32K stays in memory where it fits: 100,000 tokens on 32 GB take the 128K tier.
    let plan = try make(100_000)
    try require(
        plan.configuration.contextLimit == 131_072 && plan.notes.contains { $0.contains("100000 tokens, served on the 128K tier") },
        "100,000 tokens: \(plan.notes)")
    let held = try make(200_000)
    try require(held.configuration.contextLimit == 262_144 && held.configuration.contextInMemory, "200,000 tokens in memory on 32 GB")
    let filed = try make(200_000, limit: 15_000_000_000)
    try require(
        filed.configuration.fileBackedContext && filed.notes.contains { $0.contains("200000 tokens, served on the 256K tier") },
        "200,000 tokens with 15 GB: \(filed.notes)")
    // An explicit 15 GB is the whole process envelope: its 1 GB transient allowance leaves 14 GB for owners.
    // Pin both sides of the boundary, rather than accepting a smaller reported tier without proving admission.
    let fitting = try make(65_536, limit: 15_000_000_000)
    let explicitLedger = try fitting.configuration.makeLedger()
    // contextInMemory selects the memory variant of 256K only; every smaller tier is inherently in memory.
    try require(
        fitting.configuration.contextLimit == 65_536 && !fitting.configuration.fileBackedContext
            && explicitLedger.ceiling == 15_000_000_000 && explicitLedger.maximumUsableBytes == 14_000_000_000
            && explicitLedger.maximumUsableBytes + explicitLedger.transientMargin == explicitLedger.ceiling,
        "64K explicit envelope: tier \(fitting.configuration.contextLimit), file-backed \(fitting.configuration.fileBackedContext), "
            + "ceiling \(explicitLedger.ceiling), owners \(explicitLedger.maximumUsableBytes), transient \(explicitLedger.transientMargin)")
    do {
        _ = try make(131_072, limit: 15_000_000_000, tier: 131_072)
        throw EngineError("128K in memory exceeded the explicit 15 GB process envelope without refusal")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    for bad in [0, 262_145] {
        do {
            _ = try make(bad)
            throw EngineError("context \(bad) was planned")
        } catch let error as EngineError where error.code == "invalid_configuration" {}
    }
    func context(_ size: UInt64, _ tokens: Int, limit: UInt64? = nil) -> OrigamiPlan.ContextPlan {
        OrigamiPlan.contextPlan(
            model: model, host: host(size), context: tokens, memoryLimit: limit, promptCache: root, availableNow: size * gib,
            footprintNow: 0)
    }
    let paged = context(32, 200_000, limit: 15_000_000_000)
    let note = paged.note ?? ""
    try require(
        paged.served == 200_000 && paged.tier == 262_144 && paged.inMemory == 65_536 && paged.paged
            && note.contains("Above 64K") && note.contains("67 minutes") && note.contains("37 GB"),
        "200K on 32 GB with 15 GB: \(paged.report)")
    let whole = context(32, 200_000)
    try require(whole.inMemory == 262_144 && !whole.paged && whole.note == nil, "200K on 32 GB in memory: \(whole.report)")
    let resident = context(32, 65_536)
    try require(
        resident.served == 65_536 && resident.tier == 65_536 && !resident.paged && resident.note == nil, "64K on 32 GB: \(resident.report)")
    try require(context(16, 8192).inMemory >= 8192 && context(192, 8192).inMemory == 262_144, "Where paging starts on 16 and 192 GB")
}

/// The data folder as the user gave it: a path with `..` plans (it is standardized), and the file tier names a
/// drive that is not connected instead of planning against the boot drive. No model needed.
func planDataFolderPaths() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-plan-paths-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let gib: UInt64 = 1 << 30
    let host = OrigamiHost(physicalMemory: 32 * gib, workingSet: 32 * gib * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    _ = try OrigamiPlan.make(
        model: model, host: host, context: 8192, promptCache: URL(fileURLWithPath: root.path + "/a/../b", isDirectory: true),
        availableNow: 32 * gib, footprintNow: 0)
    let missing = URL(fileURLWithPath: "/Volumes/origami-missing-\(UUID().uuidString)/data", isDirectory: true)
    do {
        _ = try OrigamiPlan.make(
            model: model, host: host, context: 262_144, promptCache: missing, availableNow: 32 * gib, footprintNow: 0,
            availableDisk: 100_000_000_000, tier: 262_144)
        throw EngineError("The file tier planned on a drive that is not connected")
    } catch let error as EngineError where error.code == "insufficient_disk" {
        try require(error.description.contains("not connected"), "Missing drive: \(error)")
    }
}

/// Memory for experts is the user's choice (`expert_memory`), and memory_limit may give Origami more than
/// the automatic share, up to Metal's working set and the floor left to macOS and other apps.
func planExpertMemoryAndCeiling() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-plan-memory-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:])
    let gib: UInt64 = 1 << 30
    let gb: UInt64 = 1_000_000_000
    let host = OrigamiHost(physicalMemory: 32 * gib, workingSet: 32 * gib * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    func make(limit: UInt64? = nil, experts: UInt64? = nil) throws -> OrigamiPlan {
        try OrigamiPlan.make(
            model: model, host: host, context: 8192, memoryLimit: limit, expertMemory: experts, promptCache: root,
            availableNow: 32 * gib, footprintNow: 0)
    }
    let automatic = try make()
    let ceiling = OrigamiPlan.memoryCeiling(host: host)
    try require(
        ceiling == 32 * gib * 3 / 4 && ceiling > automatic.memory.cap, "Ceiling \(ceiling) against automatic \(automatic.memory.cap)")
    let raised = try make(limit: 30 * gb)
    try require(
        raised.memory.cap == ceiling && raised.configuration.experts.cacheSlots > automatic.configuration.experts.cacheSlots
            && raised.notes.contains { $0.contains("above the automatic") },
        "memory_limit above automatic: \(raised.memory.cap), \(raised.notes)")
    let lowered = try make(limit: 20 * gb)
    try require(
        lowered.memory.cap == 20 * gb && !lowered.notes.contains { $0.contains("above the automatic") }, "memory_limit below automatic")
    let share = try make(experts: 10 * gb)
    let slots = share.configuration.experts.cacheSlots
    let bytes = UInt64(slots * ModelLayout.expertRecordBytes)
    try require(
        bytes <= 10 * gb && bytes > 9 * gb && share.memory.cap < automatic.memory.cap && share.memory.start <= share.memory.cap,
        "expert_memory 10 GB: \(slots) slots, cap \(share.memory.cap)")
    let tiny = try make(experts: 1)
    try require(
        tiny.configuration.experts.cacheSlots >= 528 && tiny.memory.cap >= tiny.memory.minimum,
        "expert_memory below what prefill streams through")
    try require(
        OrigamiPlan.contextMemoryBytes(context: 8192) < OrigamiPlan.contextMemoryBytes(context: 32_768)
            && OrigamiPlan.contextMemoryBytes(context: 100_000) > 1 << 30, "Context memory by tier")
}
