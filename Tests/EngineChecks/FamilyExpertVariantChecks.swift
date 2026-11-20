import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// Low-bit experts on an exact GLM-5.3-Flash or DeepSeek-V4.1-Flash GGUF (`model add <experts.gguf> --base`,
/// `FamilyExpertVariant`), on the synthetic models of `syntheticFamilyGGUF`.
func familyExpertVariantChecks() -> [(String, () throws -> Void)] {
    [
        ("family_expert_variant_pairing", familyExpertVariantPairing),
        ("family_expert_variant_glm", { try familyExpertVariantRuns(.glm) }),
        ("family_expert_variant_deepseek", { try familyExpertVariantRuns(.deepseek) }),
    ]
}

/// A GGUF of `source`'s tensors (bytes copied) that `keep` keeps, with `metadata`; `edit` may change a tensor's type
/// and bytes on the way.
private func rewriteGGUF(
    _ source: GGUFSource, to url: URL, metadata: [(String, Any)], keep: (String) -> Bool = { _ in true },
    edit: (String, inout Int, inout Data) -> Void = { _, _, _ in }
) throws {
    var tensors: [(String, [Int], Int)] = []
    var data: [String: Data] = [:]
    for t in source.tensors.values.sorted(by: { $0.offset < $1.offset }) where keep(t.name) {
        let handle = try FileHandle(forReadingFrom: source.url(t.file))
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(t.offset))
        var bytes = try handle.read(upToCount: Int(t.length)) ?? Data()
        var type = t.type
        edit(t.name, &type, &bytes)
        tensors.append((t.name, t.shape, type))
        data[t.name] = bytes
    }
    try familyGGUF(metadata, tensors: tensors, data: data, to: url)
}

private func expertsOnly(_ layers: Set<Int>) -> (String) -> Bool {
    { name in name.contains("_exps.") && layers.contains(Int(name.split(separator: ".")[1]) ?? -1) }
}

/// The pairing: a whole low-bit GLM GGUF of the same checkpoint (Q2_K gate and up, Q3_K down; the same tokens and
/// routers) pairs with every layer covered and nothing left unverified; an experts-only file of two layers pairs, those
/// layers covered and its missing architecture, tokenizer and routers named; six one-field changes are refused naming
/// what differs: the architecture, the geometry, a token, a router value, a bank type no kernel runs, a bank missing.
func familyExpertVariantPairing() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("family-variant-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func folder(_ name: String) throws -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    let tokens = (0..<64).map { "t\($0)" }
    let base = try GGUFSource(try syntheticFamilyGGUF(.glm, in: try folder("base"), tokens: tokens), tokenizer: true)
    let low = Dictionary(uniqueKeysWithValues: (1...6).map { ($0, [10, 10, 11]) })
    let whole = try GGUFSource(try syntheticFamilyGGUF(.glm, in: try folder("whole"), expertTypes: low, tokens: tokens), tokenizer: true)
    guard case .string(let config)? = whole.metadata["glm5-next.config"] else { throw EngineError("the synthetic GLM has no config") }
    let metadata: [(String, Any)] = [
        ("general.architecture", "glm5-next"), ("glm5-next.config", config), ("tokenizer.ggml.tokens", tokens),
    ]

    let good = try FamilyExpertVariant.pair(whole, base: base)
    guard good.layers == [1, 2, 3, 4, 5, 6], good.unverified.isEmpty,
        good.source.expertFiles == whole.files.map({ whole.url($0).standardizedFileURL.path }),
        good.source.tensors["blk.3.ffn_down_exps.weight"]?.type == 11,
        good.source.tensors["blk.3.ffn_gate_inp.weight"]?.file == base.files[0],
        FamilySupport.key(good.source) != FamilySupport.key(base)
    else { throw EngineError("the whole low-bit GGUF did not pair as the same checkpoint: \(good.layers) \(good.unverified)") }

    let sidecarURL = try folder("sidecar").appendingPathComponent("experts.gguf")
    try rewriteGGUF(whole, to: sidecarURL, metadata: [], keep: expertsOnly([3, 6]))
    let sidecar = try FamilyExpertVariant.pair(try GGUFSource(sidecarURL, tokenizer: true, partial: true), base: base)
    guard sidecar.layers == [3, 6],
        sidecar.unverified == ["names no architecture", "carries no tokenizer to compare", "carries no routers to compare"],
        sidecar.source.tensors["blk.3.ffn_up_exps.weight"]?.file == sidecarURL.standardizedFileURL.path,
        sidecar.source.tensors["blk.4.ffn_up_exps.weight"]?.file == base.files[0]
    else { throw EngineError("the experts-only file did not pair for its two layers: \(sidecar.layers) \(sidecar.unverified)") }

    var cases: [(String, [(String, Any)], (String) -> Bool, (String, inout Int, inout Data) -> Void)] = []
    var other = metadata
    other[0] = ("general.architecture", "deepseek41")
    cases.append(("it is deepseek41, not GLM-5.3-Flash", other, { _ in true }, { _, _, _ in }))
    other = metadata
    other[1] = ("glm5-next.config", config.replacingOccurrences(of: "\"num_experts_per_tok\":4", with: "\"num_experts_per_tok\":6"))
    cases.append(("its experts per token is 6, the base's 4", other, { _ in true }, { _, _, _ in }))
    other = metadata
    other[2] = ("tokenizer.ggml.tokens", tokens.enumerated().map { $0.offset == 9 ? "x" : $0.element })
    cases.append(("token 9 differs from the base's", other, { _ in true }, { _, _, _ in }))
    cases.append(
        (
            "layer 4's router differs from the base's", metadata, { _ in true },
            { name, _, bytes in if name == "blk.4.ffn_gate_inp.weight" { bytes[100] ^= 1 } }
        ))
    cases.append(
        (
            "blk.2.ffn_down_exps.weight is Q1_0, which Origami's expert kernels do not run", metadata, { _ in true },
            { name, type, bytes in
                if name == "blk.2.ffn_down_exps.weight" {
                    type = 41
                    bytes = Data(count: 16 * 256 * 256 / 128 * 18)
                }
            }
        ))
    cases.append(
        (
            "layer 5 has blk.5.ffn_gate_exps.weight, blk.5.ffn_down_exps.weight but not all three", metadata,
            { $0 != "blk.5.ffn_up_exps.weight" }, { _, _, _ in }
        ))
    for (i, (reason, metadata, keep, edit)) in cases.enumerated() {
        let url = try folder("case-\(i)").appendingPathComponent("experts.gguf")
        try rewriteGGUF(whole, to: url, metadata: metadata, keep: keep, edit: edit)
        do {
            _ = try FamilyExpertVariant.pair(try GGUFSource(url, tokenizer: true, partial: true), base: base)
            throw EngineError("a pairing that should fail for \"\(reason)\" was accepted")
        } catch let error as EngineError where error.code == "model_incompatible" && error.description.contains(reason) {}
    }
}

/// The variant runs as "the base with its expert banks swapped": a GGUF whose experts are all low-bit (GLM: Q2_K gate
/// and up, Q3_K down; DeepSeek: IQ2_XXS gate and up, Q2_K down) and an experts-only file of two layers, each paired with
/// the base, give bit for bit the replies, logits and saved state of one GGUF with those banks in place, also with a hot
/// copy of the variant's records, and differ from the base's. The base's run and file are the same before and after.
/// Its banks are the variant's records, and the same budget holds more of them.
func familyExpertVariantRuns(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-variant-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    func folder(_ name: String) throws -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    let moe = family == .glm ? Array(1...6) : Array(0...5)
    let lowTypes = family == .glm ? [10, 10, 11] : [16, 16, 10]
    let covered: Set<Int> = family == .glm ? [3, 6] : [2, 5]
    let basePath = try syntheticFamilyGGUF(family, in: try folder("base"))
    let wholePath = try syntheticFamilyGGUF(
        family, in: try folder("whole"), expertTypes: Dictionary(uniqueKeysWithValues: moe.map { ($0, lowTypes) }))
    let referencePath = try syntheticFamilyGGUF(
        family, in: try folder("reference"), expertTypes: Dictionary(uniqueKeysWithValues: covered.map { ($0, lowTypes) }))
    let base = try GGUFSource(basePath)
    let sidecarPath = try folder("sidecar").appendingPathComponent("experts.gguf")
    try rewriteGGUF(try GGUFSource(wholePath), to: sidecarPath, metadata: [], keep: expertsOnly(covered))
    let whole = try FamilyExpertVariant.pair(try GGUFSource(wholePath, partial: true), base: base).source
    let sidecar = try FamilyExpertVariant.pair(try GGUFSource(sidecarPath, partial: true), base: base).source
    let baseBytes = SHA256.hash(data: try Data(contentsOf: basePath))

    func run(_ path: URL, _ source: GGUFSource? = nil, hot: Bool = false, name: String) throws -> SyntheticRun {
        try syntheticConversation(path, source: source, hotCopy: hot, flipPrefetch: false, in: try folder(name), device: device)
    }
    func same(_ a: SyntheticRun, _ b: SyntheticRun) -> Bool { a.reply == b.reply && a.continued == b.continued && a.digests == b.digests }
    let before = try run(basePath, name: "before")
    for (variant, reference, what) in [(whole, wholePath, "the whole low-bit GGUF"), (sidecar, referencePath, "the experts-only file")] {
        let ran = try run(basePath, variant, name: what + " variant")
        let expected = try run(reference, name: what + " reference")
        guard same(ran, expected) else {
            throw EngineError(
                "\(family): the variant with \(what) differs from the GGUF with those banks in place: \(ran.digests) vs \(expected.digests)"
            )
        }
        guard !same(ran, before) else { throw EngineError("\(family): the variant with \(what) gave the base's output") }
        let hot = try run(basePath, variant, hot: true, name: what + " hot")
        guard same(hot, ran), (hot.counters["hot_reads"] ?? 0) > 0 else {
            throw EngineError("\(family): a hot copy of the variant's records changed its output or was not read")
        }
    }
    let after = try run(basePath, name: "after")
    guard same(before, after), SHA256.hash(data: try Data(contentsOf: basePath)) == baseBytes else {
        throw EngineError("\(family): the base's output or file changed with a variant beside it")
    }

    // The slots hold the variant's records: its banks' types and bytes, and more of them in the same budget.
    let routed = base.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
    let plain = try FamilyExperts(weights: try FamilyWeights(gguf: base, device: device), budget: UInt64(routed * 2 / 5))
    let low = try FamilyExperts(weights: try FamilyWeights(gguf: whole, device: device), budget: UInt64(routed * 2 / 5))
    for layer in moe {
        let banks = ["gate", "up", "down"].map { whole.tensors["blk.\(layer).ffn_\($0)_exps.weight"]! }
        guard let bank = low.bank(layer), bank.types == lowTypes, bank.bytes == banks.map({ Int($0.length) / 16 }) else {
            throw EngineError("\(family): layer \(layer)'s slots do not hold the variant's records: \(String(describing: low.bank(layer)))")
        }
    }
    guard low.slotCount > plain.slotCount else {
        throw EngineError("\(family): the variant's smaller records give \(low.slotCount) slots, the base's \(plain.slotCount)")
    }
}
