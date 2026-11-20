import Foundation
import Metal
import OrigamiEngine

private func residencyRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError(message) }
}
private func residencyRefused(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid decode residency owner was admitted")
}

func decodeResidencyCompleteEnvelope() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    for (context, maximum, fixedWithoutFrontier, quota) in [
        // The two decode workspaces carry no 32 MiB fp32 pooled-index copy: 67,108,864 bytes less in all.
        // HTTP's two reserved control slots (headers x 8 + one 16 KiB receive each) add 557,056 bytes.
        (8192, 3552, UInt64(5_092_755_968), UInt64(427_819_008)),
        (32768, 3320, UInt64(5_740_775_936), UInt64(981_092_352)),
    ] {
        let path = CheckFixtures.file("config/elastic-residency-\(context).json")
        let runtime = try RuntimeConfiguration.load(path)
        let geometry = try ConversationGeometry(context: context)
        let resident = runtime.residentBytes + runtime.denseResidentBytes
        try residencyRequire(
            runtime.elasticDecodeResidency && runtime.experts.cacheSlots == maximum
                && resident == 2_962_292_736 && runtime.processCeiling == 16_000_000_000,
            "Frozen complete resident/configuration bound differs")
        try residencyRequire(
            try geometry.frontierMemoryBytes == quota,
            "Elastic capacity reduced the retained-frontier guarantee")
        let fixed = try geometry.maximumFixedBytes(input: 0, residentBytes: resident, retainedFrontierBytes: 0)
        try residencyRequire(
            fixed == fixedWithoutFrontier, "Independent complete owner categories differ: \(fixed), expected \(fixedWithoutFrontier)")
        let slab: UInt64 = 8 * 2_764_800
        let additionalOwners: [UInt64] = [0, 16 * 1024 * 1024, 64_000_000]
        for live in [UInt64(0), 1, quota / 3, quota - 1, quota] {
            for extra in additionalOwners {
                let actual = try ConversationPhaseBudget.expertBytes(
                    context: context, input: 0,
                    residentBytes: resident, experts: runtime.experts, usable: 15_000_000_000,
                    additionalOwners: extra, retainedFrontierBytes: live)
                let expected = min(
                    UInt64(maximum) * 2_764_800,
                    (15_000_000_000 - fixedWithoutFrontier - live - extra) / slab * slab)
                try residencyRequire(
                    actual == expected && actual >= runtime.experts.prefillCacheBytes
                        && fixedWithoutFrontier + live + extra + actual <= 15_000_000_000,
                    "Loan omitted a reader/diagnostic or rounded upward")
                if actual < runtime.experts.cacheBytes {
                    try residencyRequire(
                        fixedWithoutFrontier + live + extra + actual + slab > 15_000_000_000,
                        "Loan failed to select the maximum safe capacity")
                }
            }
        }
        for input in [1, 255] {
            try residencyRefused { _ = try geometry.maximumFixedBytes(input: input, residentBytes: resident, retainedFrontierBytes: 0) }
        }
        try residencyRefused { _ = try geometry.maximumFixedBytes(input: 0, residentBytes: resident, retainedFrontierBytes: quota + 1) }
        let complete = try ConversationPhaseBudget.expertBytes(
            context: context, input: 0,
            residentBytes: resident, experts: runtime.experts, usable: 15_000_000_000)
        let withQuota = try ConversationPhaseBudget.expertBytes(
            context: context, input: 0,
            residentBytes: resident, experts: runtime.experts, usable: 15_000_000_000, retainedFrontierBytes: quota)
        try residencyRequire(complete == withQuota, "Missing reader proof did not preserve full quota")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
        var execution = object["execution"] as! [String: Any]
        execution["elastic_decode_residency"] = false
        object["execution"] = execution
        let refused = directory.appendingPathComponent("disabled-\(context).json")
        try JSONSerialization.data(withJSONObject: object).write(to: refused)
        try residencyRefused { _ = try RuntimeConfiguration.load(refused) }
        // Five slabs past the frozen count: without the pooled-index copies (67,108,864 bytes, 24 slots) the first refused count is higher.
        // The configured cache may also borrow the arena an empty conversation lends (decodeResidencyArenaLoan).
        let lent = (geometry.arenaBytes - geometry.heldArenaBytes(live: 0, growth: 0) + slab - 1) / slab * 8
        execution["elastic_decode_residency"] = true
        execution["expert_cache_slots"] = maximum + 40 + Int(lent)
        object["execution"] = execution
        try JSONSerialization.data(withJSONObject: object).write(to: refused)
        try residencyRefused { _ = try RuntimeConfiguration.load(refused) }
    }
}

func decodeResidencyJoinedLoanRetirement() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No residency fixture device") }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1)
    try autoreleasepool {
        let configuration = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 8,
            slotsPerSlab: 2, demandWindow: 2, readConcurrency: 1, prefillCacheSlots: 2)
        let owner = try ExpertExecution(device: device, ledger: ledger, configuration: configuration, packedAvailable: { false }) {
            key, _, bytes in
            bytes[0] = UInt8(key.expert)
        }
        try owner.beginSequence()
        try owner.beginDecodePhase(cacheBytes: UInt64(6 * PackStore.expertBytes))
        try residencyRequire(owner.snapshot().cache.allocatedBytes == 0, "Loan eagerly allocated expert payload")
        let window = try owner.makeWindow(layer: 0, experts: Array(0..<6))
        for expert in 0..<6 {
            try window.consume(expert: expert) { buffer in
                try residencyRequire(buffer.bytes[0] == UInt8(expert), "Loan changed expert bytes")
                let before = owner.snapshot()
                try residencyRefused { try owner.beginDecodePhase(cacheBytes: configuration.prefillCacheBytes) }
                try residencyRequire(
                    owner.snapshot().targetCacheBytes == before.targetCacheBytes
                        && owner.snapshot().cache.allocatedBytes == before.cache.allocatedBytes,
                    "Busy reclamation partially mutated its target or readers")
            }
        }
        try residencyRequire(owner.snapshot().cache.allocatedBytes == UInt64(6 * PackStore.expertBytes), "Loan was not populated")
        try owner.beginDecodePhase(cacheBytes: configuration.prefillCacheBytes)
        try residencyRequire(
            owner.snapshot().cache.allocatedBytes == configuration.prefillCacheBytes
                && owner.snapshot().phase == .decode && owner.snapshot().phaseTransitions == 1,
            "Reclamation failed to return all slabs or changed arithmetic phase")
        try owner.beginDecodePhase(cacheBytes: configuration.cacheBytes)
        try residencyRequire(
            owner.snapshot().cache.allocatedBytes == configuration.prefillCacheBytes, "Regrowth copied live payload eagerly")
        for invalid in [configuration.prefillCacheBytes - 1, configuration.cacheBytes + 1] {
            try residencyRefused { try owner.beginDecodePhase(cacheBytes: invalid) }
            try residencyRequire(owner.snapshot().targetCacheBytes == configuration.cacheBytes, "Invalid loan mutated capacity")
        }
        try owner.finishSequence()
    }
    try residencyRequire(ledger.snapshot().reserved == 0, "Loan owner failed final retirement")
}

/// KV loan: the conversation arena keeps the ledger charge a request can reach and lends the rest to the expert
/// cache. The charge follows `hold` (never below the live pages, never above the maximum); a page past it takes free
/// ledger room, or is refused while the cache holds that room; the fixed owners count only the held arena; and the
/// plan's decode cache grows by the arena an empty conversation lends, which the phase budget takes back as pages grow.
func decodeResidencyArenaLoan() throws {
    let mib: UInt64 = 1 << 20
    let ledger = try MemoryLedger(ceiling: 400 * mib, transientMargin: 1)
    try autoreleasepool {
        let arena = try QSAPageArena(ledger: ledger, maximumBytes: 200 * mib)
        try residencyRequire(ledger.reservedBytes == 200 * mib && arena.heldBytes == 200 * mib, "Arena did not start whole")
        try arena.hold(10 * mib)
        try residencyRequire(ledger.reservedBytes == 10 * mib, "Lending kept the charge")
        try arena.hold(150 * mib)
        try arena.hold(90 * mib)  // grows by a second piece, then shrinks across it
        try residencyRequire(ledger.reservedBytes == 90 * mib && arena.heldBytes == 90 * mib, "Pieces miscounted")
        try arena.hold(500 * mib)
        try residencyRequire(ledger.reservedBytes == 200 * mib, "Hold passed the arena maximum")
        var vector = try QSAPagedVector(arena: arena, pageElements: 8192).appending([1, 2, 3])
        try arena.hold(0)
        let live = arena.snapshot().liveBytes
        try residencyRequire(live == 16384 && arena.heldBytes == live && ledger.reservedBytes == live, "Hold dropped below live pages")
        // Past the hold the page takes free ledger room, 64 MiB at a time.
        vector = try vector.appending([UInt16](repeating: 7, count: 8192))
        try residencyRequire(arena.heldBytes == live + 64 * mib && ledger.reservedBytes == arena.heldBytes, "Page past the hold")
        // While the expert cache holds the room, the page is refused and nothing changes.
        try arena.hold(0)
        let held = arena.heldBytes
        let cache = try ledger.reserve(bytes: ledger.maximumUsableBytes - held, category: "expert-slabs")
        try residencyRefused { _ = try vector.appending([UInt16](repeating: 8, count: 8192)) }
        try residencyRequire(
            arena.snapshot().refusals == 1 && arena.heldBytes == held && vector.count == 8195, "Refused page changed the arena")
        try ledger.release(cache)
    }
    try residencyRequire(ledger.reservedBytes == 0, "Arena pieces were not released")

    let context = 131072
    let geometry = try ConversationGeometry(context: context)
    let idle = geometry.heldArenaBytes(live: 0, growth: 0)
    let lendable = geometry.arenaBytes - idle
    try residencyRequire(
        idle == 2 * 256 * 25_344 + 268_435_456 && geometry.heldArenaBytes(live: 0, growth: context) == geometry.arenaBytes
            && geometry.heldArenaBytes(live: geometry.arenaBytes, growth: 0) == geometry.arenaBytes
            && geometry.heldArenaBytes(live: 1000, growth: 255) == idle + 1000
            && geometry.heldArenaBytes(live: 0, growth: 256) == idle + 256 * 25_344,
        "Held arena estimate differs")
    let whole = try geometry.maximumFixedBytes(input: 0, residentBytes: 0)
    try residencyRequire(
        try geometry.maximumFixedBytes(input: 0, residentBytes: 0, heldArenaBytes: geometry.arenaBytes) == whole
            && geometry.maximumFixedBytes(input: 0, residentBytes: 0, heldArenaBytes: idle) == whole - lendable,
        "Fixed owners ignore the held arena")

    // 128K in memory on a 32 GB Mac: the configured decode cache is what an empty conversation leaves.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-loan-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let ram: UInt64 = 32 << 30
    let host = OrigamiHost(physicalMemory: ram, workingSet: ram * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    let plan = try OrigamiPlan.make(
        model: try ModelSource(fixture: root, spans: [:]), host: host, context: context,
        promptCache: root, availableNow: ram, footprintNow: 0)
    let configuration = plan.configuration
    let admission = try configuration.memoryAdmission()
    let experts = configuration.experts
    let slab = UInt64(8 * PackStore.expertBytes)
    let usable = admission.usableBytes
    try residencyRequire(
        configuration.elasticDecodeResidency && configuration.contextLimit == context && !configuration.fileBackedContext
            && admission.lentArenaFixedBytes == admission.noFrontierFixedBytes - lendable
            && experts.cacheBytes == (usable - admission.lentArenaFixedBytes) / slab * slab
            && experts.cacheBytes >= (usable - admission.noFrontierFixedBytes) / slab * slab + lendable / slab * slab,
        "Plan did not lend the idle arena: \(experts.cacheSlots) slots")
    // The phase budget gives the loan back as the request's pages grow, and never sums past the budget.
    let resident = configuration.residentBytes + configuration.denseResidentBytes
    func budget(_ held: UInt64?) throws -> UInt64 {
        try ConversationPhaseBudget.expertBytes(
            context: context, input: 0, residentBytes: resident, experts: experts,
            usable: usable, retainedFrontierBytes: 0, heldArenaBytes: held)
    }
    var previous = UInt64.max
    for live in [0, geometry.arenaBytes / 4, geometry.arenaBytes / 2, geometry.arenaBytes] {
        let held = geometry.heldArenaBytes(live: live, growth: 8192)
        let bytes = try budget(held)
        let fixed = try geometry.maximumFixedBytes(input: 0, residentBytes: resident, retainedFrontierBytes: 0, heldArenaBytes: held)
        try residencyRequire(bytes <= previous && fixed + bytes <= usable, "Loan at \(live) live bytes: \(bytes)")
        previous = bytes
    }
    try residencyRequire(
        try budget(nil) == budget(geometry.arenaBytes) && budget(idle) == experts.cacheBytes
            && budget(nil) < experts.cacheBytes, "Whole-arena budget changed")
}
