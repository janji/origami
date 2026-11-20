import Foundation
import OrigamiEngine

/// `GGUFExperts.variant` (`origami model add <experts.gguf> --base <model>`) pairs a GGUF's routed experts with an exact
/// model only when they are the same checkpoint. A synthetic base (ten experts, a three-token tokenizer, bf16 routers) and
/// a sparse GGUF split part (Q2_0 banks, the same routers and tokens) pair, and its manifest reads back as the engine reads
/// it; each one-field change is refused naming what differs: a router value, a token, folded rotations, the expert count,
/// a bank type the kernels do not run, a bank missing from the parts given.
func ggufExpertsVariantPairing() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("variant-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let config: [String: Any] = [
        "num_hidden_layers": 48, "hidden_size": 2560, "num_experts": 10, "num_experts_per_tok": 8, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640, "num_attention_heads": 24, "num_key_value_heads": 2,
    ]
    try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
    try Data(#"{"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": [{"id": 2, "content": "<x>"}]}"#.utf8)
        .write(to: root.appendingPathComponent("tokenizer.json"))
    // Per compared layer: a [5, 2] router then a [1, 4] shared-expert gate (only values are compared), bf16, in the base
    // and at 32-byte steps in the GGUF.
    let layers = [0, 11, 23, 35, 47]
    let values = (0..<(layers.count * 14)).map { BF16.encode(Float($0 % 29) / 7 - 2) }
    let bytes = values.withUnsafeBytes { Data($0) }
    try bytes.write(to: root.appendingPathComponent("base.bin"))
    var spans: [String: Any] = [:]
    for (i, layer) in layers.enumerated() {
        for (name, at, shape) in [("gate", i * 28, [5, 2]), ("shared_expert_gate", i * 28 + 20, [1, 4])] {
            spans["language_model.model.layers.\(layer).mlp.\(name).weight"] = [
                "file": "base.bin", "dtype": "BF16", "offset": at, "length": shape.reduce(2, *), "shape": shape,
            ]
        }
    }
    let base = try ModelSource(fixture: root, spans: spans)

    struct Change {
        var tokens = ["a", "b", "<x>"], rotations = false, experts: UInt32 = 10, downType: UInt32 = 42, dropLast = false, flip = false
    }
    func write(_ change: Change) throws -> URL {
        var head = Data("GGUF".utf8)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func string(_ s: String) {
            u64(UInt64(s.utf8.count))
            head.append(contentsOf: s.utf8)
        }
        var keys: [(String, Any)] = [
            ("general.architecture", "qwen4exp"), ("split.count", UInt32(2)), ("qwen4exp.block_count", UInt32(48)),
            ("qwen4exp.embedding_length", UInt32(2560)), ("qwen4exp.expert_count", change.experts),
            ("qwen4exp.expert_used_count", UInt32(8)), ("qwen4exp.expert_feed_forward_length", UInt32(640)),
            ("qwen4exp.expert_shared_feed_forward_length", UInt32(640)), ("qwen4exp.attention.head_count", UInt32(24)),
            ("qwen4exp.attention.head_count_kv", UInt32(2)), ("tokenizer.ggml.tokens", change.tokens),
        ]
        if change.rotations { keys.append(("prism.hadamard.size", UInt32(128))) }
        var tensors: [(String, [UInt64], UInt32, UInt64)] = []
        var offset: UInt64 = 0
        for (i, layer) in layers.enumerated() {
            tensors.append(("blk.\(layer).ffn_gate_inp.weight", [5, 2], 30, UInt64(i * 64)))
            tensors.append(("blk.\(layer).ffn_gate_inp_shexp.weight", [4], 30, UInt64(i * 64 + 32)))
        }
        offset = UInt64(layers.count * 64)
        for layer in 0..<48 {
            for (part, shape) in [("gate", [10, 640, 2560]), ("up", [10, 640, 2560]), ("down", [10, 2560, 640])] {
                if change.dropLast && layer == 47 && part == "down" { continue }
                let type: UInt32 = part == "down" && layer == 5 ? change.downType : 42
                let (block, size) = GGUFQuant.sizes[Int(type)]!
                tensors.append(("blk.\(layer).ffn_\(part)_exps.weight", shape.map(UInt64.init), type, offset))
                offset += UInt64(shape[0] * shape[1] * shape[2] / block * size + 31) / 32 * 32
            }
        }
        u32(3)
        u64(UInt64(tensors.count))
        u64(UInt64(keys.count))
        for (key, value) in keys {
            string(key)
            if let v = value as? UInt32 {
                u32(4)
                u32(v)
            } else if let v = value as? String {
                u32(8)
                string(v)
            } else if let v = value as? [String] {
                u32(9)
                u32(8)
                u64(UInt64(v.count))
                v.forEach(string)
            }
        }
        for (name, shape, type, at) in tensors {
            string(name)
            u32(UInt32(shape.count))
            shape.reversed().forEach(u64)
            u32(type)
            u64(at)
        }
        head.append(Data(count: (32 - head.count % 32) % 32))
        let start = head.count
        for i in layers.indices {
            var router = bytes.subdata(in: (i * 28)..<(i * 28 + 20))
            if change.flip && i == 2 { router[3] ^= 0x01 }
            head.append(router + Data(count: 12) + bytes.subdata(in: (i * 28 + 20)..<(i * 28 + 28)) + Data(count: 24))
        }
        let url = root.appendingPathComponent("experts-00001-of-00002.gguf")
        try head.write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(start) + offset)  // sparse: the banks are never read here
        try handle.close()
        return url
    }
    func pair(_ change: Change) throws -> Data {
        try GGUFExperts.variant(gguf: try GGUFSource(try write(change), tokenizer: true, partial: true), base: base)
    }

    let layout = try GGUFExperts(manifest: try pair(Change()))
    guard layout.experts == 10, layout.layers.count == 48, layout.layers.allSatisfy({ $0.parts.map(\.type) == [42, 42, 42] }),
        layout.files.count == 1, layout.files[0].path.hasSuffix("experts-00001-of-00002.gguf")
    else { throw EngineError("the matching pair's manifest is not the GGUF's Q2_0 banks") }
    do {
        _ = try GGUFSource(try write(Change()))
        throw EngineError("one part of two was read as a whole GGUF")
    } catch let error as EngineError where error.description.contains("has 1 of its 2 parts") {}

    var cases: [(Change, String)] = []
    var change = Change()
    change.flip = true
    cases.append((change, "layer 23's router differs from the base's"))
    change = Change()
    change.tokens = ["a", "c", "<x>"]
    cases.append((change, "token 1 is \"b\" in the base and not in the GGUF"))
    change = Change()
    change.rotations = true
    cases.append((change, "Hadamard rotations are folded"))
    change = Change()
    change.experts = 12
    cases.append((change, "its expert_count is 12, the base's num_experts 10"))
    change = Change()
    change.downType = 41
    cases.append((change, "blk.5.ffn_down_exps.weight is Q1_0, which Origami's expert kernels do not run"))
    change = Change()
    change.dropLast = true
    cases.append((change, "blk.47.ffn_down_exps.weight is not in the GGUF parts given"))
    for (change, reason) in cases {
        do {
            _ = try pair(change)
            throw EngineError("a pairing that should fail for \"\(reason)\" was accepted")
        } catch let error as EngineError where error.code == "model_incompatible" && error.description.contains(reason) {}
    }
}
