import Foundation
import OrigamiEngine
import OrigamiEngineChecks

func wholeScheduleCommandsJoinBeforeRetirement() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-whole-owner-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    try WholeExecutionSelfChecks.scheduleOwnership(store: PackStore(model: ModelSource(fixture: root, spans: [:])))
}

func wholeScheduleConfigurationIsBoundedAndDefaultOff() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("whole-schedule-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ object: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let minimal: [String: Any] = ["schema": "origami-config-v1", "profile": "exact"]
    guard try !load(minimal).wholeExecutionSchedule else { throw EngineError("Whole schedule is not default-off") }
    // Retired: ignored, even without the whole schedule it once required.
    var retired = minimal
    retired["execution"] = ["decode_arriving_first": true]
    _ = try load(retired)
    for context in [8192, 32768] {
        let source = CheckFixtures.file("config/exact-per-layer-\(context).json")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
        var execution = object["execution"] as! [String: Any]
        execution["whole_execution_schedule"] = true
        object["execution"] = execution
        let valid = try load(object)
        guard valid.wholeExecutionSchedule else { throw EngineError("Whole schedule flag was lost") }
        for (key, value) in [
            ("whole_execution_schedule", 1 as Any), ("whole_execution_schedule", "true" as Any),
            ("resident_small_tensors", false), ("fused_hc_prefill", false),
            ("qsa_decode_reuse", false), ("moe_device_prefill", false), ("pinned_indexer", false),
        ] {
            var candidate = object
            var changed = execution
            changed[key] = value
            candidate["execution"] = changed
            var refused = false
            do { _ = try load(candidate) } catch is EngineError { refused = true }
            guard refused else { throw EngineError("Invalid whole schedule dependency was accepted: " + key) }
        }
        for (key, value) in [("context_limit", 4096)] {
            var candidate = object
            candidate[key] = value
            var refused = false
            do { _ = try load(candidate) } catch is EngineError { refused = true }
            guard refused else { throw EngineError("Whole schedule changed its admission/profile contract: " + key) }
        }
        var candidate = object
        candidate["prefill"] = ["compute_tile": 128, "read_scope": 8192]
        var refused = false
        do { _ = try load(candidate) } catch is EngineError { refused = true }
        guard refused else { throw EngineError("Whole schedule changed its arithmetic tile") }
    }
}

func fileConversationConfigurationAndOwners() throws {
    let source = CheckFixtures.file("config/exact-dense-frontiers-262144.json")
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("file-kv-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ object: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let runtime = try load(object)
    let geometry = try ConversationGeometry(context: runtime.contextLimit)
    guard geometry.fileBacked, geometry.arenaBytes == 469_762_048,
        try geometry.fileMetadataBytes == 65_329_152,
        geometry.stagingBytes == 1_073_741_824,
        try geometry.frontierMemoryBytes == 1_171_166_976,
        geometry.http.maximumBodyBytes == 25_165_824, geometry.http.structureBytes == 1_048_576, geometry.http.bufferedRequests == 2,
        geometry.http.admittedBytes == 875_298_816 + 557_056, geometry.http.largeBodyBytes == 3 * 3 * 64 * 1_048_576,
        ConversationGeometry.fileDiskBytes + runtime.prefixDiskBytes + runtime.conversationFrontierDiskBytes
            <= ConversationGeometry.aggregateDiskLimit
    else {
        throw EngineError("Complete file-KV geometry or HTTP ownership differs")
    }
    for (group, key, value) in [
        ("prefix_cache", "frontier_layout", "unknown" as Any),
        ("prefix_cache", "persistent_conversation", false as Any),
        ("prefix_cache", "disk_bytes", 2_200_000_000),
        ("prefix_cache", "frontier_disk_bytes", 0),
        ("prefix_cache", "memory_bytes", 1_100_000_000),
        ("quotas", "snapshot_bytes", 8_589_934_592),
        ("quotas", "spill_bytes", 64_000_000_000),
        ("execution", "expert_cache_slots", 3448),
    ] {
        var changed = object
        var nested = object[group] as! [String: Any]
        nested[key] = value
        changed[group] = nested
        var refused = false
        do { _ = try load(changed) } catch is EngineError { refused = true }
        guard refused else { throw EngineError("Incomplete file-KV owner accepted: \(group).\(key)") }
    }
    let resident = runtime.residentBytes + runtime.denseResidentBytes
    let fixed = try geometry.maximumFixedBytes(input: 8192, residentBytes: resident)
    guard fixed + runtime.experts.prefillCacheBytes <= 15_000_000_000 else { throw EngineError("Maximum cold scope was not admitted") }
    for input in [0, 1, 255] {
        let bytes = try ConversationPhaseBudget.expertBytes(
            context: runtime.contextLimit, input: input,
            residentBytes: resident, experts: runtime.experts, usable: 15_000_000_000)
        let fixed = try geometry.maximumFixedBytes(input: input, residentBytes: resident)
        guard bytes >= runtime.experts.prefillCacheBytes, fixed + bytes <= 15_000_000_000,
            input != 0 || bytes == runtime.experts.cacheBytes,
            input == 0 || bytes < runtime.experts.cacheBytes
        else {
            throw EngineError("Short file-KV phase did not preserve the simultaneous memory ceiling")
        }
    }
    let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
    var refused = false
    do { _ = try geometry.makeArena(ledger: ledger, parent: nil) } catch is EngineError { refused = true }
    guard refused, ledger.snapshot().reserved == 0 else { throw EngineError("Missing external owner allocated resident KV") }
    for context in [8192, 32768] {
        let runtime = try RuntimeConfiguration.load(CheckFixtures.file("config/exact-dense-frontiers-\(context).json"))
        let geometry = try ConversationGeometry(context: context)
        let resident = runtime.residentBytes + runtime.denseResidentBytes
        guard geometry.maximumFrontiers == context / 256 + 2,
            geometry.frontierDiskBytes == UInt64(context / 256 + 3) * 115_458_048,
            try geometry.maximumFixedBytes(input: 8192, residentBytes: resident) + runtime.experts.prefillCacheBytes <= 15_000_000_000
        else {
            throw EngineError("Required tier omitted its complete dense frontier owners")
        }
        for input in [0, 1, 255] {
            let bytes = try ConversationPhaseBudget.expertBytes(
                context: context, input: input, residentBytes: resident,
                experts: runtime.experts, usable: 15_000_000_000)
            guard try geometry.maximumFixedBytes(input: input, residentBytes: resident) + bytes <= 15_000_000_000,
                bytes == runtime.experts.cacheBytes
            else { throw EngineError("Required dense tier unexpectedly displaced expert capacity") }
        }
    }
}

func fileConversationMaximumRotationOwnership() throws {
    let ledger = try MemoryLedger(ceiling: 50_000_000, transientMargin: 1_000_000)
    let host = try ledger.reserve(bytes: 32_000_000, category: "maximum_rotation_fixture")
    try autoreleasepool {
        let rope = try PinnedRoPE(ledger: ledger)
        let input = (0..<(256 * 24 * 256)).map { BF16.decode(BF16.encode(Float($0 % 97 - 48) / 64)) }
        for start in [262144 - 256, 0] {
            let result = try rope.rotation(input, width: 256, heads: 24, positions: Array(start..<(start + 256)))
            guard result.count == input.count, result.allSatisfy(\.isFinite),
                result.indices.filter({ $0 % 256 >= 64 }).allSatisfy({ result[$0] == input[$0] }),
                ledger.snapshot().reservationHighWater - 32_000_000 <= ConversationGeometry.maximumRoPEBytes
            else {
                throw EngineError("Maximum original rotation exceeded its complete owner bound")
            }
        }
    }
    try ledger.release(host)
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Maximum rotation owners did not retire") }
}
