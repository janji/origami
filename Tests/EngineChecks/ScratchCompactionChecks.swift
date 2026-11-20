import Foundation
import Metal
import OrigamiEngine
import OrigamiEngineChecks

private func compactRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError("Scratch compaction: " + message) }
}

/// Aliases retain one physical owner, including when a logical view escapes the runner. An oversized sibling
/// cannot authorize an absent role, arithmetic overflow or growth beyond the role's own admitted extent.
func familyScratchCompactionOwnership() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-compact-owner-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try expertFixtureGGUF(in: folder)
    let page = try FamilyRunner.scratchLength(1)
    let ledger = try MemoryLedger(ceiling: UInt64(32 * page), transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    var escaped: MTLBuffer?
    try autoreleasepool {
        let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(path), device: device), context: 64)
        runner.ledger = ledger
        try runner.configureScratchCompaction(true)
        let counts = [
            "embedded": page / 2, "ffn-in": page / 2, "attn-in": page / 4, "head-in": 1,
            "ffn-out": page / 4, "attn-out": page / 4, "exp-x": page / 4, "exp-out": page / 2,
            "hc-post": page / 4,
        ]
        var probes = 0
        try runner.prepareScratch(
            counts,
            footprint: {
                probes += 1
                return 0
            })
        let input = try runner.work("embedded", page / 2)
        let output = try runner.work("ffn-out", page / 4)
        let expert = try runner.work("exp-x", page / 4)
        let post = try runner.work("hc-post", page / 4)
        try compactRequire(
            try input === runner.work("ffn-in", page / 2)
                && input === runner.work("attn-in", page / 4) && input === runner.work("head-in", 1), "input lifetimes did not share")
        try compactRequire(
            try output === runner.work("attn-out", page / 4)
                && expert === runner.work("exp-out", page / 2), "output lifetimes did not share")
        try compactRequire(Set([input, output, expert, post].map { ObjectIdentifier($0) }).count == 4, "overlapping roles alias")
        // Counts are floats: the nine logical owners occupy [2,2,1,1,1,1,1,2,1] pages (12 total).
        // The four physical owners occupy [2,1,2,1] pages (6 total), saving exactly 6 pages.
        let physicalBytes = UInt64(6 * page)
        let savedBytes = UInt64((12 - 6) * page)
        try compactRequire(
            runner.workingAllocatedBytes == physicalBytes,
            "physical allocations expected \(physicalBytes), got \(runner.workingAllocatedBytes)")
        try compactRequire(
            runner.workingRetainedBytes == physicalBytes,
            "retained ownership expected \(physicalBytes), got \(runner.workingRetainedBytes)")
        try compactRequire(
            ledger.reservedBytes == physicalBytes,
            "ledger ownership expected \(physicalBytes), got \(ledger.reservedBytes)")
        try compactRequire(
            runner.scratchCompactionSavedBytes == savedBytes,
            "alias saving expected \(savedBytes), got \(runner.scratchCompactionSavedBytes)")
        memset(input.contents(), 0xA5, input.length)
        memset(post.contents(), 0x3C, post.length)
        try compactRequire(
            try runner.work("ffn-in", 1).contents().load(as: UInt8.self) == 0xA5
                && post.contents().load(as: UInt8.self) == 0x3C, "poison crossed a live metadata owner")
        for (name, count) in [("mix-in", 1), ("head-in", page / 4 + 1)] {
            do {
                _ = try runner.work(name, count)
                throw EngineError("Unplanned alias was accepted", code: "fixture_unexpected")
            } catch let error as EngineError where error.code == "scratch_plan_mismatch" {}
        }
        do {
            try runner.prepareScratch(["ffn-in": page / 2, "embedded": -1])
            throw EngineError("Invalid alias hidden by max", code: "fixture_unexpected")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        do {
            try runner.configureScratchCompaction(false)
            throw EngineError("Live alias plan changed", code: "fixture_unexpected")
        } catch let error as EngineError where error.code == "invalid_configuration" {}
        let before = probes
        try runner.prepareScratch(
            counts,
            footprint: {
                probes += 1
                return 0
            })
        try compactRequire(probes == before && runner.timing.commits == 0, "reuse probed or joined GPU work")
        escaped = try runner.work("head-in", 1)
    }
    try compactRequire(escaped != nil && ledger.reservedBytes == UInt64(2 * page), "escaped alias lost or duplicated its allocation charge")
    autoreleasepool { escaped = nil }
    try compactRequire(ledger.reservedBytes == 0, "last alias did not retire its owner")
}

/// Extend the tiny pooled-attention fixture to two independent layers, so attention overwrites the preceding
/// layer's FFN scratch. Only the tiny generated fixture is read; no model checkpoint is consulted.
private func compactTwoLayerFixture(_ family: ModelFamily, in folder: URL) throws -> URL {
    let original = try poolModelFixture(family, in: folder)
    let source = try GGUFSource(original)
    guard case .string(let architecture) = source.metadata["general.architecture"],
        case .string(let json) = source.metadata[architecture + ".config"]
    else { throw EngineError("Missing compact fixture config") }
    var config = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    config["num_hidden_layers"] = 2
    if family == .deepseek {
        config["compress_ratios"] = [2, 2]
        config["kv_source_layer_ids"] = [0, 1]
        config["index_source_layer_ids"] = [0, 1]
    } else {
        config["layer_types"] = ["linear_attention", "full_attention"]
        config["linear_attn_config"] = ["num_heads": 1, "head_dim": 128, "short_conv_kernel_size": 4]
    }
    let file = try Data(contentsOf: original)
    var tensors: [(String, [Int], Int)] = []
    var payload: [String: Data] = [:]
    for tensor in source.tensors.values.sorted(by: { $0.name < $1.name }) {
        let bytes = file.subdata(in: Int(tensor.offset)..<Int(tensor.offset + tensor.length))
        let names =
            tensor.name.hasPrefix("blk.0.") ? [tensor.name, tensor.name.replacingOccurrences(of: "blk.0.", with: "blk.1.")] : [tensor.name]
        for name in names {
            tensors.append((name, tensor.shape, tensor.type))
            payload[name] = bytes
        }
    }
    if family == .glm {
        // Exercise KDA -> DSA, including recurrent state and the convolution carry, with the real 128-wide KDA kernel.
        func add(_ suffix: String, _ shape: [Int], ones: Bool = false) {
            let name = "blk.0.kda_" + suffix + ".weight"
            let values = (0..<shape.reduce(1, *)).map { ones ? Float(1) : Float(($0 * 11 + name.utf8.count) % 19 - 9) / 128 }
            tensors.append((name, shape, 0))
            payload[name] = values.withUnsafeBytes { Data($0) }
        }
        for part in ["q", "k", "v"] {
            add(part, [128, 32])
            add(part + "_conv", [4, 128])
        }
        for part in ["f", "g"] {
            add(part + "_a", [128, 32])
            add(part + "_b", [128, 128])
        }
        add("beta", [1, 32])
        add("output", [32, 128])
        add("dt_bias", [128])
        add("a_log", [1])
        add("o_norm", [128], ones: true)
    }
    let path = folder.appendingPathComponent("compact-two-\(architecture).gguf")
    let updated = String(decoding: try JSONSerialization.data(withJSONObject: config), as: UTF8.self)
    try familyGGUF(
        [("general.architecture", architecture), (architecture + ".config", updated)], tensors: tensors, data: payload, to: path)
    return path
}

/// Compare every logit bit and all live conversation-state bytes across multi-chunk prefill, two decodes,
/// reset, and a short new prompt. Poison every unique allocation before each pass; no stale scratch is valid input.
func familyScratchCompactionParity() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-compact-parity-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for family in [ModelFamily.glm, .deepseek] {
        let path = try compactTwoLayerFixture(family, in: folder)
        func run(_ compact: Bool) throws -> (logits: [[UInt32]], states: [[Data]], bytes: UInt64) {
            try autoreleasepool {
                let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
                let model = try makeFamilyModel(weights, expertBudget: 1 << 20, context: 512) { _ in }
                let runner = model as! FamilyRunner
                try runner.configureScratchCompaction(compact)
                var logits: [[UInt32]] = []
                var states: [[Data]] = []
                var maximum: UInt64 = 0
                for (step, tokens) in [(0, (0..<269).map { 1 + $0 % 7 }), (1, [9]), (2, [3]), (3, [2, 5, 1, 7, 4])] {
                    if step == 1 { runner.beginDecode() }
                    if step == 3 {
                        try model.reset()
                        try runner.beginPrefill()
                    }
                    let plan = runner.scratchPlan(rows: tokens.count)
                    try runner.prepareScratch(plan)
                    var poisoned = Set<ObjectIdentifier>()
                    for (name, count) in plan {
                        let buffer = try runner.work(name, count)
                        if poisoned.insert(ObjectIdentifier(buffer)).inserted {
                            memset(buffer.contents(), step % 2 == 0 ? 0xA5 : 0x5A, buffer.length)
                        }
                    }
                    let result = try model.forward(tokens)
                    try compactRequire(
                        result.allSatisfy(\.isFinite) && result.contains { $0 != 0 }, "\(family) produced empty/nonfinite logits")
                    logits.append(result.map(\.bitPattern))
                    states.append(model.stateParts(at: model.position).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) })
                    maximum = max(maximum, runner.workingAllocatedBytes)
                }
                return (logits, states, maximum)
            }
        }
        let baseline = try run(false)
        let compact = try run(true)
        try compactRequire(baseline.logits == compact.logits && baseline.states == compact.states, "\(family) changed logits/state bits")
        try compactRequire(compact.bytes < baseline.bytes, "\(family) did not reduce physical scratch")
    }
}

func qwenScratchCompaction() throws { try ScratchCompactionSelfChecks.qwen() }
