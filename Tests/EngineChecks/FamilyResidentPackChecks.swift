import CryptoKit
import Foundation
import Metal
import OrigamiEngine

func familyResidentPackIsExplicitBoundedAndVerified() throws {
    func require(_ condition: Bool, _ message: String) throws { if !condition { throw EngineError("Resident pack: " + message) } }
    func rejects(_ operation: () throws -> Void) throws {
        do { try operation() } catch { return }
        throw EngineError("Resident pack: invalid operation accepted")
    }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-resident-pack-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let width = 256
    let outputs = 96
    let name = "blk.0.attn_q_b.weight"
    var values = [Float](repeating: 0, count: width * outputs)
    for i in values.indices {
        let code = (i * 137 + 71) % 1021 - 510
        values[i] = Float(code) / 1024
    }
    var encoded = Data(count: GGUFQuantize.bytes(8, count: values.count))
    values.withUnsafeBufferPointer { v in
        encoded.withUnsafeMutableBytes { GGUFQuantize.encode(v.baseAddress!, count: values.count, type: 8, into: $0.baseAddress!) }
    }
    let names = [
        name, "token_embd.weight", "output.weight", "blk.0.ffn_gate_inp.weight", "blk.0.engram_embd.weight", "blk.1.attn_q_b.weight",
    ]
    let url = folder.appendingPathComponent("model.gguf")
    try familyGGUF(
        [
            ("general.architecture", "deepseek41"), ("deepseek41.num_hidden_layers", 1), ("deepseek41.hidden_size", width),
            ("deepseek41.vocab_size", outputs), ("deepseek41.n_routed_experts", 4), ("deepseek41.num_experts_per_tok", 2),
            ("deepseek41.moe_intermediate_size", width),
        ], tensors: names.map { ($0, [outputs, width], 8) },
        data: Dictionary(uniqueKeysWithValues: names.map { ($0, encoded) }), to: url)
    let originalSource = try Data(contentsOf: url)
    let gguf = try GGUFSource(url)
    let geometry = try ModelGeometry(gguf: gguf)
    let plan = try FamilyResidentPack.plan(gguf, geometry: geometry)
    try require(plan.map(\.name) == [name], "embeddings, head, router, lookup or MTP weights entered the plan")
    let destination = folder.appendingPathComponent("q4")
    try FamilyResidentPack.build(gguf, geometry: geometry, into: destination)
    try require(try Data(contentsOf: url) == originalSource, "conversion modified source weights")
    let pack = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry)
    try require(pack.savedBytes > 0 && pack.payloadBytes == Int64(GGUFQuantize.bytes(12, count: values.count)), "wrong capacity accounting")
    let reopened = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry)
    try require(
        reopened.representationIdentity == pack.representationIdentity && !pack.representationIdentity.isEmpty,
        "stable representation identity was lost")
    try rejects { try FamilyResidentPack.build(gguf, geometry: geometry, into: destination) }
    var raw = Data(count: Int(pack.payloadBytes))
    try raw.withUnsafeMutableBytes { try pack.read(name, into: $0) }
    var decoded = [Float](repeating: 0, count: values.count)
    raw.withUnsafeBytes { GGUFQuant.dequantize($0.baseAddress!, type: 12, count: values.count, into: &decoded) }
    try require(decoded.allSatisfy { $0.isFinite } && decoded.contains(where: { $0 != 0 }), "invalid converted values")
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let original = try FamilyWeights(gguf: gguf, device: device)
    let variant = try FamilyWeights(gguf: gguf, device: device, residentPack: pack)
    let sourceMatrix = try original.matrix(name)
    let variantMatrix = try variant.matrix(name)
    try require(
        sourceMatrix.type == 8 && variantMatrix.type == 12 && variant.residentBytes < original.residentBytes,
        "opt-in representation did not reduce actual resident buffer bytes")
    try require(Data(bytes: variantMatrix.buffer.contents(), count: raw.count) == raw, "loaded converted payload differs")
    for preserved in names.dropFirst() {
        let matrix = try variant.matrix(preserved)
        try require(
            matrix.type == 8 && Data(bytes: matrix.buffer.contents(), count: encoded.count) == encoded,
            "unselected matrix changed: " + preserved)
    }
    let indexURL = destination.appendingPathComponent(FamilyResidentPack.indexName)
    let originalIndex = try Data(contentsOf: indexURL)
    var index = try JSONSerialization.jsonObject(with: originalIndex) as! [String: Any]
    let records = index["records"] as! [[String: Any]]
    index["records"] = records + records
    try JSONSerialization.data(withJSONObject: index).write(to: indexURL)
    try rejects { _ = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry) }
    try originalIndex.write(to: indexURL)
    let payloadURL = destination.appendingPathComponent(FamilyResidentPack.dataName)
    let handle = try FileHandle(forWritingTo: payloadURL)
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: Data([raw[0] ^ 1]))
    try handle.close()
    try rejects { try raw.withUnsafeMutableBytes { try pack.read(name, into: $0) } }
    let corrupted = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry)
    try rejects { try raw.withUnsafeMutableBytes { try corrupted.read(name, into: $0) } }
    // A changed source incarnation cannot silently reuse even an otherwise well-formed pack.
    try originalSource.write(to: url, options: .atomic)
    try rejects { _ = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry) }
}

/// GLM target conversion plans trunk Q8_0 matrices only: not the router, the MTP (nextn) block, embeddings or head.
func familyResidentPackPlansGLMTrunk() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-resident-glm-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let width = 256
    let outputs = 96
    let encoded = Data(count: GGUFQuantize.bytes(8, count: width * outputs))
    let names = [
        "blk.0.attn_q_b.weight", "blk.1.attn_q_b.weight", "blk.2.attn_q_b.weight", "blk.0.ffn_gate_inp.weight",
        "token_embd.weight", "output.weight",
    ]
    let url = folder.appendingPathComponent("glm.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", width), ("glm5-next.vocab_size", outputs),
            ("glm5-next.expert_count", 4), ("glm5-next.expert_used_count", 2), ("glm5-next.expert_feed_forward_length", width),
            ("glm5-next.leading_dense_block_count", 1), ("glm5-next.nextn_predict_layers", 1), ("glm5-next.hyper_connection.count", 4),
            ("glm5-next.block_count", 3), ("glm5-next.trunk_block_count", 2), ("glm5-next.layer_types", [0, 1, 1]),
            ("glm5-next.linear_attention.head_count", 2), ("glm5-next.linear_attention.gate_lower_bound", Float(-5)),
        ],
        tensors: names.map { ($0, [outputs, width], 8) }, data: Dictionary(uniqueKeysWithValues: names.map { ($0, encoded) }), to: url)
    let gguf = try GGUFSource(url)
    let geometry = try ModelGeometry(gguf: gguf)
    let plan = try FamilyResidentPack.plan(gguf, geometry: geometry).map(\.name)
    guard geometry.family == .glm, plan == ["blk.0.attn_q_b.weight", "blk.1.attn_q_b.weight"] else {
        throw EngineError("Resident pack: GLM plan selected \(plan) for \(geometry.family) with \(geometry.layers) layers")
    }
}
