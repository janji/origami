import Foundation
import Metal
import OrigamiEngine

/// A minimal GGUF v3 header and tensor table (tensor data from `data`, else zero-filled): metadata values are UInt32,
/// Float, String or arrays of UInt32 / Int32; tensors are (name, PyTorch-order shape, ggml type).
/// `fill` gives a tensor's bytes when `data` has none (the prefill benchmark's weights, made one tensor at a time);
/// a tensor with neither is zeros, left as a hole in the file.
func familyGGUF(
    _ metadata: [(String, Any)], tensors: [(String, [Int], Int)], data: [String: Data] = [:], fill: ((String, Int) -> Data?)? = nil,
    to url: URL
) throws {
    var file = Data("GGUF".utf8)
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { file.append(contentsOf: $0) } }
    func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { file.append(contentsOf: $0) } }
    func string(_ s: String) {
        u64(UInt64(s.utf8.count))
        file.append(contentsOf: s.utf8)
    }
    u32(3)
    u64(UInt64(tensors.count))
    u64(UInt64(metadata.count))
    for (key, value) in metadata {
        string(key)
        switch value {
        case let v as String:
            u32(8)
            string(v)
        case let v as Float:
            u32(6)
            u32(v.bitPattern)
        case let v as Bool:
            u32(7)
            file.append(v ? 1 : 0)
        case let v as [Int]:
            u32(9)
            u32(5)
            u64(UInt64(v.count))
            v.forEach { u32(UInt32(bitPattern: Int32($0))) }
        case let v as [String]:
            u32(9)
            u32(8)
            u64(UInt64(v.count))
            v.forEach(string)
        case let v as Int:
            u32(4)
            u32(UInt32(v))
        default: throw EngineError("familyGGUF: unsupported value for \(key)")
        }
    }
    var at: UInt64 = 0
    var sizes: [Int] = []
    for (name, shape, type) in tensors {
        string(name)
        u32(UInt32(shape.count))
        shape.reversed().forEach { u64(UInt64($0)) }
        u32(UInt32(type))
        u64(at)
        let (block, bytes) = GGUFQuant.sizes[type]!
        let size = shape.reduce(1, *) / block * bytes
        sizes.append(size)
        at += UInt64((size + 31) / 32 * 32)
    }
    file.append(Data(count: (32 - file.count % 32) % 32))
    try file.write(to: url)
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    var end = UInt64(file.count)
    for ((name, _, _), size) in zip(tensors, sizes) {
        if let bytes = data[name] ?? fill?(name, size) {
            guard bytes.count == size else { throw EngineError("familyGGUF: \(name) needs \(size) bytes, not \(bytes.count)") }
            try handle.seek(toOffset: end)
            try handle.write(contentsOf: bytes)
        }
        end += UInt64((size + 31) / 32 * 32)
    }
    try handle.truncate(atOffset: end)
}

/// Families and geometry come from what a checkpoint says about itself: GLM-5.3-Flash and DeepSeek-V4.1-Flash from
/// their config.json and from GGUF metadata in llama.cpp's and DwarfStar's spellings; tensor roles across namings;
/// header-only verdicts that run, refuse with the reason, or name a missing engine. Qwen's geometry is ModelContract's.
func modelFamilyGeometry() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Model family: " + what) } }
    func config(_ name: String) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/\(name)/config.json"))) as! [String: Any]
    }
    let qwen = ModelGeometry.qwen
    try require(
        qwen.family == .qwen && qwen.layers == ModelContract.layers && qwen.experts == ModelContract.experts && qwen.expertsPerToken == 10
            && qwen.mixers.filter { $0 == .attention }.count == 12 && qwen.mixers[3] == .attention && qwen.hyperConnections == 4,
        "Qwen geometry \(qwen)")
    let glm = try ModelGeometry(config: config("glm-5.3-flash"))
    try require(
        glm.family == .glm && glm.layers == 45 && glm.hidden == 4096 && glm.experts == 288 && glm.expertsPerToken == 8
            && glm.expertWidth == 2048
            && glm.mixers.filter { $0 == .recurrent }.count == 34 && glm.moeLayers.count == 42 && glm.moeLayers.first == 3
            && glm.mtpLayers == 1
            && glm.int("linear_attn_config.num_heads") == 64 && glm.int("index_topk") == 2048
            && glm.values["linear_attn_config.gate_lower_bound"] == -5,
        "GLM geometry from config.json: \(glm)")
    let deepseek = try ModelGeometry(config: config("deepseek-v4.1-flash"))
    try require(
        deepseek.family == .deepseek && deepseek.layers == 40 && deepseek.hidden == 5120 && deepseek.experts == 384
            && deepseek.expertsPerToken == 6
            && deepseek.lists["engram_layer_ids"] == [1, 14] && deepseek.int("o_groups") == 8
            && deepseek.lists["compress_ratios"]?.count == 43,
        "DeepSeek geometry from config.json: \(deepseek)")
    do {
        _ = try ModelGeometry(config: ["model_type": "llama"])
        throw EngineError("Model family: llama accepted")
    } catch let error as EngineError where error.description.contains("model_type llama") {}

    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    // A two-layer GLM (one KDA, one DSA with experts), as DwarfStar writes it, then with llama.cpp's keys.
    func glmTensors(expertType: Int) -> [(String, [Int], Int)] {
        [
            ("token_embd.weight", [512, 256], 8), ("output.weight", [512, 256], 8), ("output_norm.weight", [256], 0),
            ("blk.0.kda_q.weight", [256, 256], 8), ("blk.1.attn_q_a.weight", [256, 256], 8),
            ("blk.1.ffn_gate_exps.weight", [4, 256, 256], expertType), ("blk.1.ffn_up_exps.weight", [4, 256, 256], expertType),
            ("blk.1.ffn_down_exps.weight", [4, 256, 256], expertType), ("blk.1.ffn_gate_inp.weight", [4, 256], 0),
            ("blk.2.nextn.eh_proj.weight", [256, 512], 30),
        ]
    }
    let common: [(String, Any)] = [
        ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512),
        ("glm5-next.expert_count", 4), ("glm5-next.expert_used_count", 2), ("glm5-next.expert_feed_forward_length", 256),
        ("glm5-next.leading_dense_block_count", 1), ("glm5-next.nextn_predict_layers", 1), ("glm5-next.hyper_connection.count", 4),
    ]
    let dwarf = root.appendingPathComponent("glm-dwarfstar.gguf")
    try familyGGUF(
        common + [
            ("glm5-next.block_count", 3), ("glm5-next.trunk_block_count", 2), ("glm5-next.layer_types", [0, 1, 1]),
            ("glm5-next.linear_attention.head_count", 2), ("glm5-next.linear_attention.gate_lower_bound", Float(-5)),
        ],
        tensors: glmTensors(expertType: 16), to: dwarf)
    var report = try ModelCompatibility.check(local: dwarf)
    try require(
        report.family == .glm && report.geometry?.layers == 2 && report.geometry?.mixers == [.recurrent, .attention]
            && report.geometry?.moe == [false, true]
            && report.geometry?.int("linear_attn_config.num_heads") == 2
            && report.geometry?.values["linear_attn_config.gate_lower_bound"] == -5
            && report.types["routed experts"] == ["IQ2_XXS": 3] && report.types["multi-token prediction"] == ["BF16": 1],
        "DwarfStar GLM header: \(report)")
    try require(
        report.verdict == (ModelCompatibility.servedFamilies.contains(.glm) ? .runs : .refused)
            && (ModelCompatibility.servedFamilies.contains(.glm) || report.reason.contains("no GLM-5.3-Flash engine")),
        "GLM verdict \(report.verdict): \(report.reason)")
    let llama = root.appendingPathComponent("glm-llama.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("glm5-next.kda.head_dim", 128)],
        tensors: glmTensors(expertType: 12), to: llama)
    report = try ModelCompatibility.check(local: llama)
    try require(
        report.geometry?.layers == 2 && report.geometry?.mixers == [.recurrent, .attention]
            && report.geometry?.int("linear_attn_config.head_dim") == 128,
        "llama.cpp GLM header: \(report)")
    // True incompatibilities are named: a type no kernel reads, experts missing from a MoE layer, an unknown architecture.
    let ternary = root.appendingPathComponent("glm-ternary.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.layer_types", [0, 1, 1])], tensors: glmTensors(expertType: 34), to: ternary)
    report = try ModelCompatibility.check(local: ternary)
    try require(
        report.verdict == .refused && report.reason.contains("TQ1_0") && report.reason.contains("no Origami kernel"),
        "ternary experts: \(report.reason)")
    let missing = root.appendingPathComponent("glm-missing.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.layer_types", [0, 1, 1])],
        tensors: glmTensors(expertType: 12).filter { !$0.0.contains("down_exps") }, to: missing)
    report = try ModelCompatibility.check(local: missing)
    try require(report.verdict == .refused && report.reason.contains("blk.1.ffn_down_exps"), "missing experts: \(report.reason)")
    let partial = root.appendingPathComponent("glm-layer-000.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.layer_types", [0, 1, 1])],
        tensors: glmTensors(expertType: 12).filter { !$0.0.hasPrefix("blk.1.") }, to: partial)
    report = try ModelCompatibility.check(local: partial)
    try require(report.verdict == .refused && report.reason.contains("1 of the model's 2 layers"), "a per-layer file: \(report.reason)")
    let other = root.appendingPathComponent("other.gguf")
    try familyGGUF(
        [("general.architecture", "llama"), ("llama.block_count", 2)], tensors: [("token_embd.weight", [512, 256], 8)], to: other)
    report = try ModelCompatibility.check(local: other)
    try require(
        report.verdict == .refused && report.family == nil && report.reason.contains("general.architecture llama"),
        "unknown architecture: \(report.reason)")
    // Hostile header numbers are refused, never trapped on: a layer count past any model, NaN, infinity, fewer blocks
    // than prediction layers; a tensor named `blk.` alone.
    for (i, layers) in ([Float(1e19), .nan, .infinity, Float(1 << 30), 0] as [Float]).enumerated() {
        let hostile = root.appendingPathComponent("glm-hostile-\(i).gguf")
        try familyGGUF(
            common + [("glm5-next.block_count", layers), ("glm5-next.layer_types", [0, 1, 1])], tensors: glmTensors(expertType: 12),
            to: hostile)
        do { report = try ModelCompatibility.check(local: hostile) } catch is EngineError { continue }
        try require(report.verdict == .refused, "block_count \(layers) is \(report.verdict): \(report.reason)")
    }
    let stray = root.appendingPathComponent("glm-stray.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.layer_types", [0, 1, 1])],
        tensors: glmTensors(expertType: 12) + [("blk.", [256], 0)], to: stray)
    _ = try ModelCompatibility.check(local: stray)
    // A folder with a config.json but no safetensors shards is judged by its GGUF.
    let folder = root.appendingPathComponent("gguf-with-config", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try FileManager.default.copyItem(
        at: CheckFixtures.file("models/glm-5.3-flash/config.json"), to: folder.appendingPathComponent("config.json"))
    try FileManager.default.copyItem(at: dwarf, to: folder.appendingPathComponent("glm.gguf"))
    report = try ModelCompatibility.check(local: folder)
    try require(
        report.format == "gguf" && report.geometry?.layers == 2,
        "a GGUF folder with a config.json was judged as \(report.format): \(report.reason)")

    // Forgiving reads of other converters: mainline llama.cpp's `glm5next`, config-named keys (DwarfStar's DeepSeek
    // writes `deepseek41.num_hidden_layers`), an embedded config.json, a second split naming, integer data tables,
    // MXFP8 tables; a LoRA adapter and FP8 bytes stored as raw integers are refused by name.
    let mainline = root.appendingPathComponent("glm-mainline.gguf")
    let renamed: [(String, Any)] = common.map { key, value in
        key == "general.architecture" ? (key, "glm5next") : (key.replacingOccurrences(of: "glm5-next.", with: "glm5next."), value)
    }
    try familyGGUF(
        renamed + [("glm5next.block_count", 3), ("glm5next.attention.head_count_kv", [0, 64, 64])], tensors: glmTensors(expertType: 12),
        to: mainline)
    report = try ModelCompatibility.check(local: mainline)
    try require(
        report.family == .glm && report.geometry?.layers == 2 && report.geometry?.mixers == [.recurrent, .attention],
        "glm5next alias: \(report.reason)")
    // Another converter's keys: a scalar head_count_kv, ssm.state_size for the KDA head size, indexer.block_size for
    // the k-pool, so each layer's mixer comes from its tensors.
    let fork = root.appendingPathComponent("glm-fork.gguf")
    try familyGGUF(
        renamed + [
            ("glm5next.block_count", 3), ("glm5next.attention.head_count_kv", 1), ("glm5next.ssm.state_size", 128),
            ("glm5next.attention.indexer.block_size", 4),
        ], tensors: glmTensors(expertType: 29), to: fork)
    report = try ModelCompatibility.check(local: fork)
    try require(
        report.verdict == .runs && report.geometry?.mixers == [.recurrent, .attention]
            && report.geometry?.int("linear_attn_config.head_dim") == 128
            && report.geometry?.int("index_kpool") == 4, "layer types from tensors: \(report.verdict) \(report.reason)")
    // A split set whose metadata lives in a first part without tensors, and whose second part was edited on its own
    // (another architecture spelling, as in a public GLM GGUF): the first part's metadata is the model's.
    let split = root.appendingPathComponent("split", isDirectory: true)
    try FileManager.default.createDirectory(at: split, withIntermediateDirectories: true)
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("split.count", 2)],
        tensors: [], to: split.appendingPathComponent("glm-00001-of-00002.gguf"))
    try familyGGUF(
        [("general.architecture", "glm5next"), ("split.count", 2), ("split.no", 1)], tensors: glmTensors(expertType: 12),
        to: split.appendingPathComponent("glm-00002-of-00002.gguf"))
    report = try ModelCompatibility.check(local: split.appendingPathComponent("glm-00001-of-00002.gguf"))
    try require(
        report.family == .glm && report.architecture == "glm5-next" && report.geometry?.layers == 2 && report.verdict == .runs,
        "split set with an edited second part: \(report.verdict) \(report.reason)")
    // A set whose last part is named on its own in the folder above (ISTA-DASLab's shared DeepSeek Engram file) is found
    // by its split keys, and a file there whose keys name another set is not; `split.count` 0 is a single file.
    let build = split.appendingPathComponent("build", isDirectory: true)
    try FileManager.default.createDirectory(at: build, withIntermediateDirectories: true)
    let parts = glmTensors(expertType: 12)
    let keys: [(String, Any)] = [("general.architecture", "glm5next"), ("split.count", 3), ("split.tensors.count", parts.count)]
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64])] + keys.dropFirst(), tensors: [],
        to: build.appendingPathComponent("glm-00001-of-00003.gguf"))
    try familyGGUF(keys + [("split.no", 1)], tensors: Array(parts.dropLast()), to: build.appendingPathComponent("glm-00002-of-00003.gguf"))
    try familyGGUF(
        [("general.architecture", "glm5next"), ("split.count", 3), ("split.tensors.count", parts.count + 1), ("split.no", 2)],
        tensors: [parts[parts.count - 1]], to: split.appendingPathComponent("another.gguf"))
    try familyGGUF(keys + [("split.no", 2)], tensors: [parts[parts.count - 1]], to: split.appendingPathComponent("shared.gguf"))
    let first = build.appendingPathComponent("glm-00001-of-00003.gguf")
    try require(
        try GGUFSource.parts(first).files == ["glm-00001-of-00003.gguf", "glm-00002-of-00003.gguf", "../shared.gguf"],
        "the last part named on its own: \(try GGUFSource.parts(first).files)")
    report = try ModelCompatibility.check(local: first)
    try require(report.verdict == .runs, "a set completed from the folder above: \(report.verdict) \(report.reason)")
    let single = root.appendingPathComponent("single.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("split.count", 0)],
        tensors: parts, to: single)
    report = try ModelCompatibility.check(local: single)
    try require(report.verdict == .runs, "split.count 0 in a single file: \(report.verdict) \(report.reason)")
    let dsTensors: [(String, [Int], Int)] = [
        ("token_embd.weight", [512, 256], 1), ("output.weight", [512, 256], 8),
        ("blk.0.ffn_gate_exps.weight", [4, 256, 256], 16), ("blk.0.ffn_up_exps.weight", [4, 256, 256], 16),
        ("blk.0.ffn_down_exps.weight", [4, 256, 256], 10),
        ("blk.0.engram_offsets.weight", [8], 26), ("per_layer_token_embd.weight", [64, 32], 43),
    ]
    let keyed = root.appendingPathComponent("ds-keys.gguf")
    try familyGGUF(
        [
            ("general.architecture", "deepseek41"), ("deepseek41.num_hidden_layers", 1), ("deepseek41.hidden_size", 256),
            ("deepseek41.vocab_size", 512), ("deepseek41.n_routed_experts", 4), ("deepseek41.num_experts_per_tok", 2),
            ("deepseek41.moe_intermediate_size", 256), ("deepseek41.engram_layer_ids", [0]),
        ], tensors: dsTensors, to: keyed)
    report = try ModelCompatibility.check(local: keyed)
    try require(
        report.geometry?.layers == 1 && report.geometry?.lists["engram_layer_ids"] == [0] && !report.reason.contains("I32")
            && !report.reason.contains("MXFP8"),
        "DeepSeek config-named keys, an I32 table and an MXFP8 table: \(report.reason)")
    var embedded = try config("deepseek-v4.1-flash")
    var text = embedded["text_config"] as! [String: Any]
    text["num_hidden_layers"] = 1
    embedded["text_config"] = text
    let carried = root.appendingPathComponent("ds-config.gguf")
    try familyGGUF(
        [
            ("general.architecture", "deepseek41"),
            ("deepseek41.config", String(decoding: try JSONSerialization.data(withJSONObject: embedded), as: UTF8.self)),
        ],
        tensors: dsTensors, to: carried)
    report = try ModelCompatibility.check(local: carried)
    try require(
        report.geometry?.layers == 1 && report.geometry?.experts == 384 && report.geometry?.int("o_groups") == 8,
        "embedded config.json: \(report)")
    // DwarfStar's DeepSeek format keeps an I8 Engram table per layer (256 E4M3 codes, then 8 E8M0 scales): its rows
    // decode to what an independent decoder gives (E4M3 by its fields, the scale, bf16 rounding in Double). A layer
    // with no table at all is refused. A format no kernel reads is refused for its types even when its expert tensors
    // carry other names.
    let dwarfDS = root.appendingPathComponent("ds-dwarf-engram.gguf")
    var engramEmbedded = embedded
    var engramText = engramEmbedded["text_config"] as! [String: Any]
    engramText["engram_layer_ids"] = [0]
    engramEmbedded["text_config"] = engramText
    let scales: [[UInt8]] = [
        Array(repeating: 127, count: 8), [0, 1, 2, 3, 4, 5, 6, 7], [100, 110, 120, 126, 128, 140, 200, 254],
        [255, 127, 9, 10, 11, 12, 13, 250],
    ]
    var i8 = Data()
    for r in 0..<8 { i8 += (0..<256).map { UInt8(($0 * (2 * r + 1) + 13 * r) & 255) } + scales[r % scales.count] }
    let dwarfConfig = String(decoding: try JSONSerialization.data(withJSONObject: engramEmbedded), as: UTF8.self)
    try familyGGUF(
        [("general.architecture", "deepseek41"), ("deepseek41.config", dwarfConfig)],
        tensors: dsTensors.filter { $0.0 != "per_layer_token_embd.weight" } + [("blk.0.engram_embd.weight", [8, 264], 24)],
        data: ["blk.0.engram_embd.weight": i8], to: dwarfDS)
    report = try ModelCompatibility.check(local: dwarfDS)
    try require(report.verdict == .runs, "DwarfStar Engram: \(report.verdict) \(report.reason)")
    let dwarfTable = try GGUFSource(dwarfDS).tensors["blk.0.engram_embd.weight"]!
    let dwarfBytes = try Data(contentsOf: dwarfDS)
    try require(
        DeepSeekModel.engramRowBytes(dwarfTable, dim: 256) == 264 && DeepSeekModel.engramRowBytes(dwarfTable, dim: 128) == nil,
        "I8 Engram row bytes")
    func referenceRow(_ row: Data) -> [Float] {
        (0..<256).map { i in
            let code = Int(row[row.startIndex + i])
            let scale = Int(row[row.startIndex + 256 + i / 32])
            let e = (code >> 3) & 15
            let m = code & 7
            if code & 127 == 127 || scale == 255 { return .nan }
            let value =
                (code & 128 != 0 ? -1.0 : 1.0) * (e == 0 ? Double(m) / 512 : Double(8 + m) * pow(2, Double(e - 10)))
                * pow(2, Double(scale - 127))
            guard value != 0 else { return Float(value) }
            let ulp = abs(value) < 0x1p-126 ? 0x1p-133 : pow(2, Double(value.exponent - 7))  // bf16: 8 significant bits, subnormal below 2^-126
            return Float((value / ulp).rounded(.toNearestOrEven) * ulp)
        }
    }
    for r in 0..<8 {
        let start = Int(dwarfTable.offset) + r * 264
        let row = dwarfBytes.subdata(in: start..<(start + 264))
        var got = [Float](repeating: 0, count: 256)
        row.withUnsafeBytes { DeepSeekModel.decodeEngramRow($0.baseAddress!, type: 24, dim: 256, into: &got) }
        let want = referenceRow(row)
        if let i = (0..<256).first(where: { i in got[i].bitPattern != want[i].bitPattern && !(got[i].isNaN && want[i].isNaN) }) {
            throw EngineError("Model family: I8 Engram row \(r) value \(i) decodes to \(got[i]), the reference to \(want[i])")
        }
    }
    let noTable = root.appendingPathComponent("ds-no-engram.gguf")
    try familyGGUF(
        [("general.architecture", "deepseek41"), ("deepseek41.config", dwarfConfig)],
        tensors: dsTensors.filter { $0.0 != "per_layer_token_embd.weight" }, to: noTable)
    report = try ModelCompatibility.check(local: noTable)
    try require(report.verdict == .refused && report.reason.contains("blk.0.engram_embd"), "no Engram table: \(report.reason)")
    // vcruz305's converter (mrfakename's GGUFs): engram.* keys, a ggml-typed table per layer, no source-layer keys (the
    // layers carrying the compressor and the indexer's query are the sources).
    let vcruz = root.appendingPathComponent("ds-vcruz.gguf")
    try familyGGUF(
        [
            ("general.architecture", "deepseek41"), ("deepseek41.num_hidden_layers", 1), ("deepseek41.hidden_size", 256),
            ("deepseek41.vocab_size", 512), ("deepseek41.n_routed_experts", 4), ("deepseek41.num_experts_per_tok", 2),
            ("deepseek41.moe_intermediate_size", 256), ("deepseek41.engram.layer_ids", [0]), ("deepseek41.engram.key_length", 256),
            ("deepseek41.engram.head_count", 8), ("deepseek41.engram.max_ngram_size", 4), ("deepseek41.engram.pad_id", 2),
        ],
        tensors: dsTensors.filter { $0.0 != "per_layer_token_embd.weight" } + [
            ("blk.0.engram_embd.weight", [16, 256], 10),
            ("blk.0.attn_compressor_kv.weight", [64, 256], 8), ("blk.0.indexer.attn_q_b.weight", [64, 256], 8),
        ], to: vcruz)
    report = try ModelCompatibility.check(local: vcruz)
    try require(
        report.verdict == .runs && report.geometry?.lists["kv_source_layer_ids"] == [0]
            && report.geometry?.lists["index_source_layer_ids"] == [0]
            && report.geometry?.int("engram_head_dim") == 256 && report.geometry?.int("engram_n_heads") == 8,
        "vcruz305 DeepSeek: \(report.verdict) \(report.reason) \(String(describing: report.geometry?.lists))")
    try require(
        DeepSeekModel.engramRowBytes(try GGUFSource(vcruz).tensors["blk.0.engram_embd.weight"]!, dim: 256) == 84, "Q2_K Engram row bytes")
    let custom = root.appendingPathComponent("ds-custom.gguf")
    try familyGGUF(
        [
            ("general.architecture", "deepseek41"),
            ("deepseek41.config", String(decoding: try JSONSerialization.data(withJSONObject: embedded), as: UTF8.self)),
        ],
        tensors: dsTensors.filter { !$0.0.contains("_exps") } + [("blk.0.ffn_gate_exps.qweight", [4, 256, 256], 34)], to: custom)
    report = try ModelCompatibility.check(local: custom)
    try require(
        report.verdict == .refused && report.reason.contains("TQ1_0") && report.reason.contains("no Origami kernel"),
        "custom format: \(report.reason)")
    let adapter = root.appendingPathComponent("lora.gguf")
    try familyGGUF(
        [("general.architecture", "deepseek41"), ("general.type", "adapter")], tensors: [("blk.0.attn_q_a.weight.lora_a", [16, 256], 0)],
        to: adapter)
    try require(try ModelCompatibility.check(local: adapter).reason.contains("LoRA adapter"), "adapter not named")
    let raw = root.appendingPathComponent("glm-i8.gguf")
    try familyGGUF(
        common + [("glm5-next.block_count", 3), ("glm5-next.layer_types", [0, 1, 1])],
        tensors: glmTensors(expertType: 12).map { $0.0 == "blk.1.attn_q_a.weight" ? ($0.0, $0.1, 24) : $0 }, to: raw)
    try require(try ModelCompatibility.check(local: raw).reason.contains("raw I8"), "raw I8 matrix not named")
    try require(
        GGUFSource.modelPrefix("GLM-5.3-Flash-MTP-01-of-04.gguf") == "GLM-5.3-Flash-MTP"
            && GGUFSource.modelPrefix("x-00002-of-00007.gguf") == "x",
        "split names")
    // MXFP8: scale 2^1 (E8M0 128), E4M3 codes 1.0 (0x38), -0.5 (0xB0), the smallest subnormal (0x01), NaN (0x7F).
    var block = [UInt8](repeating: 0, count: 33)
    block[0] = 128
    block[1] = 0x38
    block[2] = 0xB0
    block[3] = 0x01
    block[4] = 0x7F
    var decoded = [Float](repeating: 0, count: 32)
    block.withUnsafeBytes { raw in
        decoded.withUnsafeMutableBufferPointer { GGUFQuant.dequantize(raw.baseAddress!, type: 43, count: 32, into: $0.baseAddress!) }
    }
    try require(
        decoded[0] == 2 && decoded[1] == -1 && decoded[2] == 0x1p-8 && decoded[3].isNaN && decoded[4] == 0, "MXFP8 \(decoded.prefix(5))")

    // GLM's chat template uses Jinja's numeric attributes; they become item access, float literals stay.
    try require(
        FamilySupport.jinjaCompatible("{% if m.content.0.type == 1.0 %}{{ a[0].b.12 }}{% endif %}")
            == "{% if m.content[0].type == 1.0 %}{{ a[0].b[12] }}{% endif %}", "numeric attributes")

    // Safetensors verdicts: GLM in FP8 with block scales or BF16 converts; another encoding, or DeepSeek's FP8, is refused.
    func spans(_ dtypes: [String]) throws -> [String: TensorSpan] {
        var json: [String: Any] = [:]
        for (i, dtype) in dtypes.enumerated() {
            json["t\(i)"] = ["file": "a.safetensors", "dtype": dtype, "offset": 0, "length": 2, "shape": [1]]
        }
        return try JSONDecoder().decode([String: TensorSpan].self, from: JSONSerialization.data(withJSONObject: json))
    }
    var bf16 = try config("glm-5.3-flash")
    bf16["quantization_config"] = nil
    var mxfp4 = bf16
    mxfp4["quantization_config"] = ["quant_method": "mxfp4"]
    let verdicts: [(String, [String: Any], [String], CompatibilityReport.Verdict)] = [
        ("GLM FP8", try config("glm-5.3-flash"), ["F8_E4M3", "F32", "BF16"], .converts), ("GLM BF16", bf16, ["BF16", "F32"], .converts),
        ("GLM MXFP4", mxfp4, ["U8", "BF16"], .refused), ("GLM I32 tensors", bf16, ["BF16", "I32"], .refused),
        ("DeepSeek FP8", try config("deepseek-v4.1-flash"), ["F8_E4M3", "F32"], .refused),
    ]
    for (what, checkpoint, dtypes, verdict) in verdicts {
        let report = ModelCompatibility.check(config: checkpoint, headers: try spans(dtypes), source: what)
        try require(report.verdict == verdict, "\(what) is \(report.verdict): \(report.reason)")
    }

    // Tensor roles in the four namings.
    let roles: [(String, TensorRole)] = [
        ("blk.7.ffn_gate_exps.weight", .experts), ("model.language_model.layers.7.mlp.experts.12.up_proj.weight", .experts),
        ("language_model.model.layers.3.mlp.switch_mlp.down_proj.weight", .experts), ("blk.7.ffn_up_shexp.weight", .sharedExpert),
        ("model.layers.4.mlp.shared_experts.gate_proj.weight", .sharedExpert), ("blk.4.ffn_gate_inp.weight", .router),
        ("blk.4.exp_probs_b.bias", .router), ("blk.0.kda_q_conv.weight", .recurrent), ("blk.0.ssm_conv1d_q.weight", .recurrent),
        ("model.layers.0.self_attn.f_a_proj.weight", .recurrent), ("model.layers.0.self_attn.b_proj.weight", .recurrent),
        ("model.layers.3.self_attn.kv_b_proj.weight", .attention), ("blk.3.indexer.pool_gate.weight", .attention),
        ("blk.3.attn_kv_a_mqa.weight", .attention), ("blk.2.attn_compressor_kv.weight", .attention),
        ("per_layer_token_embd.weight", .table),
        ("blk.1.engram_kv.weight", .table), ("blk.0.hc_attn_fn.weight", .hyperConnections), ("token_embd.weight", .embeddings),
        ("output.weight", .output), ("lm_head.weight", .output), ("blk.45.nextn.eh_proj.weight", .mtp),
        ("model.visual.blocks.0.attn.qkv.weight", .vision),
        ("blk.0.attn_norm.weight", .norms), ("blk.0.ffn_gate.weight", .denseFFN),
    ]
    for (name, role) in roles { try require(TensorRole.of(name) == role, "\(name) is \(TensorRole.of(name)), expected \(role)") }
}

/// A GGUF's own tokenizer and template serve chats; a template that always opens `<think>` (GLM-5.3's has no
/// thinking switch) gets it closed when thinking is off, and is left open when thinking is on.
func familyTemplateThinking() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family template: " + what) } }
    let stack = try tinyFamilyTokenizer(
        "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|><think>{% endif %}")
    let user: [[String: TemplateValue]] = [["role": .string("user"), "content": .string("Hi")]]
    let off = try stack.encodeNormalized(user, reasoningEffort: "none")
    let on = try stack.encodeNormalized(user, reasoningEffort: "medium")
    try require(on == [259, 256, 260, 257], "thinking on renders \(on)")
    try require(off == [259, 256, 260, 257, 258], "thinking off renders \(off)")
    // A template with DeepSeek's switch gets it: `<think>` when thinking, `</think>` when not, and nothing appended.
    let deepseek = try tinyFamilyTokenizer(
        "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% if thinking %}<think>{% else %}</think>{% endif %}{% endif %}"
    )
    let dsOn = try deepseek.encodeNormalized(user, reasoningEffort: "high")
    let dsOff = try deepseek.encodeNormalized(user, reasoningEffort: "none")
    try require(dsOn == [259, 256, 260, 257] && dsOff == [259, 256, 260, 258], "thinking switch renders \(dsOn) and \(dsOff)")
    // Python's dict(): a mapping's entries, then keywords (DeepSeek-V4.1's template merges message fields this way), or
    // a list of pairs; and the top effort as the template names it.
    let merging = try tinyFamilyTokenizer(
        "{% for m in messages %}{% set n = dict(m, content=m.content ~ 'i') %}<|user|>{{ n.role }}{{ n.content }}{% endfor %}{{ dict([['k', 'v']]).k }}"
    )
    try require(
        try merging.encodeNormalized(user, reasoningEffort: "none") == merging.encode("<|user|>userHiiv"), "dict(mapping, key=value)")
    for (template, expected) in [("{{ reasoning_effort }}", "high"), ("{# xhigh #}{{ reasoning_effort }}", "xhigh")] {
        let effort = try tinyFamilyTokenizer(template)
        try require(try effort.encodeNormalized(user, reasoningEffort: "high") == effort.encode(expected), "effort high in \(template)")
    }
    // A template that refuses a level's name gets its number (ISTA-DASLab's DeepSeek map has no "medium"), or else the
    // nearest name it takes; a template that refuses every level keeps its error, and one that takes the name gets it.
    let refusing = "{% else %}{{ raise_exception('bad effort') }}{% endif %}"
    let numeric = try tinyFamilyTokenizer(
        "{% set m = {'low': 50, 'high': 75} %}{% if reasoning_effort is string and reasoning_effort in m %}{{ m[reasoning_effort] }}"
            + "{% elif reasoning_effort is integer %}{{ reasoning_effort }}" + refusing)
    try require(try numeric.encodeNormalized(user, reasoningEffort: "medium") == numeric.encode("62"), "medium as a number")
    try require(try numeric.encodeNormalized(user, reasoningEffort: "low") == numeric.encode("50"), "low by its name")
    let named = try tinyFamilyTokenizer("{% if reasoning_effort in ['low', 'high'] %}{{ reasoning_effort }}" + refusing)
    try require(try named.encodeNormalized(user, reasoningEffort: "medium") == named.encode("high"), "medium as the nearest name")
    let none = try tinyFamilyTokenizer("{% if false %}" + refusing)
    try require((try? none.encodeNormalized(user, reasoningEffort: "medium")) == nil, "a template refusing every level fails")
}

/// A TokenizerStack from a tiny GGUF through FamilySupport: the byte-level alphabet (GPT-2's bytes_to_unicode, ids
/// 0...255), "Hi" (256, the one merge), then `<think>` 257, `</think>` 258, `<|user|>` 259 (eos), `<|assistant|>` 260.
/// `fillers` tokens ("<w0>", "<w1>", ...) follow, so a model's larger vocabulary decodes: plain ones (text never
/// encodes to one) or, with `fillerType` 4, user-defined ones that text encodes back to.
func tinyFamilyTokenizer(_ template: String, fillers: Int = 0, fillerType: Int = 1, fillerTexts: [String]? = nil) throws -> TokenizerStack {
    let fillerTexts = fillerTexts ?? (0..<fillers).map { "<w\($0)>" }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-tokenizer-\(getpid())-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    var shifted = 0
    let alphabet = (0..<256).map { b -> String in
        if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) { return String(UnicodeScalar(UInt8(b))) }
        shifted += 1
        return String(UnicodeScalar(UInt32(255 + shifted))!)
    }
    let gguf = folder.appendingPathComponent("tiny.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "glm4"),
            (
                "tokenizer.ggml.tokens",
                alphabet + ["Hi", "<think>", "</think>", "<|user|>", "<|assistant|>"] + fillerTexts
            ),
            ("tokenizer.ggml.merges", ["H i"]),
            (
                "tokenizer.ggml.token_type",
                [Int](repeating: 1, count: 257) + [3, 3, 3, 3] + [Int](repeating: fillerType, count: fillerTexts.count)
            ),
            ("tokenizer.ggml.eos_token_id", 259), ("tokenizer.chat_template", template),
        ], tensors: [], to: gguf)
    let support = try FamilySupport.prepare(gguf, into: folder.appendingPathComponent("support"))
    final class Box: @unchecked Sendable {
        var stack: TokenizerStack?
        var error: Error?
    }
    let box = Box()
    let done = DispatchSemaphore(value: 0)
    Task {
        do { box.stack = try await TokenizerStack(modelDirectory: support) } catch { box.error = error }
        done.signal()
    }
    done.wait()
    if let error = box.error { throw error }
    return box.stack!
}

/// A two-MoE-layer GLM GGUF of four experts each, layer 1 in Q8_0 and layer 2 in Q4_K, every byte of expert e's part p
/// in layer l equal to l * 50 + e * 10 + p (`familyExpertPools`, `familyCommandSplit`).
/// Expert tensors whose shape is not the geometry's (gate/up experts × expertWidth × hidden, down experts × hidden ×
/// expertWidth) are refused at load, naming the tensor and both shapes; one of the right size but transposed too.
func familyExpertShapes() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family expert shapes: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-expert-shapes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    // hidden 256, expertWidth 128
    for (n, shapes) in [
        [[6, 128, 256], [6, 128, 256], [6, 256, 128]], [[6, 256, 256], [6, 256, 256], [6, 256, 256]],
        [[6, 128, 256], [6, 128, 256], [6, 128, 256]],
    ].enumerated() {
        let file = folder.appendingPathComponent("experts-\(n).gguf")
        try familyGGUF(
            [
                ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512),
                ("glm5-next.expert_count", 6), ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 128),
                ("glm5-next.leading_dense_block_count", 1), ("glm5-next.block_count", 2),
                ("glm5-next.attention.head_count_kv", [0, 64]), ("glm5-next.hyper_connection.count", 4),
            ],
            tensors: zip(["gate", "up", "down"], shapes).map { ("blk.1.ffn_\($0)_exps.weight", $1, 8) }, to: file)
        let weights = try FamilyWeights(gguf: GGUFSource(file), device: device)
        do {
            _ = try FamilyExperts(weights: weights, budget: 64 << 20)
            try require(n == 0, "shapes \(shapes) were accepted")
        } catch let error as EngineError where n > 0 {
            let name = n == 1 ? "blk.1.ffn_gate_exps.weight" : "blk.1.ffn_down_exps.weight"
            let expected = n == 1 ? "[6, 128, 256]" : "[6, 256, 128]"
            try require(
                error.code == "model_incompatible" && error.description.contains(name)
                    && error.description.contains("\(shapes[n == 1 ? 0 : 2])") && error.description.contains(expected),
                "refusal does not name the tensor and both shapes: \(error)")
        }
    }
}

func expertFixtureGGUF(in folder: URL, used: Int = 1, types: [Int: Int] = [1: 8, 2: 12]) throws -> URL {
    try? FileManager.default.removeItem(at: folder)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let parts = ["gate", "up", "down"]
    var tensors: [(String, [Int], Int)] = []
    var data: [String: Data] = [:]
    for layer in types.keys.sorted() {
        let perExpert = GGUFQuantize.bytes(types[layer]!, count: 256 * 256)
        for (part, name) in parts.enumerated() {
            let tensor = "blk.\(layer).ffn_\(name)_exps.weight"
            tensors.append((tensor, [4, 256, 256], types[layer]!))
            data[tensor] = Data((0..<4).flatMap { e in [UInt8](repeating: UInt8(layer * 50 + e * 10 + part), count: perExpert) })
        }
    }
    let gguf = folder.appendingPathComponent("experts.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512),
            ("glm5-next.expert_count", 4), ("glm5-next.expert_used_count", used), ("glm5-next.expert_feed_forward_length", 256),
            ("glm5-next.leading_dense_block_count", 1), ("glm5-next.block_count", types.count + 1),
            ("glm5-next.attention.head_count_kv", [0] + Array(repeating: 64, count: types.count)),
            ("glm5-next.hyper_connection.count", 4),
        ], tensors: tensors, data: data, to: gguf)
    return gguf
}

/// Long prompt work is cut into several command buffers: a runner commits once a buffer holds `commitLimit`
/// multiply-adds (one buffer per layer slice tripped macOS's GPU watchdog), and the results are those of one buffer.
func familyCommandSplit() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family command split: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-split-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 64)
    // A Q8_0 matrix of ones (scale 1, codes 1) and ones in: every output is the width.
    let width = 4096
    let outputs = 4096
    let rows = 64
    let blocks = width / 32 * outputs
    var block = [UInt8](repeating: 1, count: 34)
    block[0] = 0x00
    block[1] = 0x3C
    guard
        let weights = device.makeBuffer(
            bytes: [UInt8]((0..<blocks).flatMap { _ in block }), length: blocks * 34, options: .storageModeShared),
        let input = device.makeBuffer(
            bytes: [Float](repeating: 1, count: rows * width), length: rows * width * 4, options: .storageModeShared),
        let output = device.makeBuffer(length: rows * outputs * 4, options: .storageModeShared)
    else { throw EngineError("no buffers") }
    let matrix = FamilyWeights.Matrix(buffer: weights, offset: 0, type: 8, width: width, outputs: outputs, name: "ones")
    let calls = 200
    let per = Double(rows * width * outputs)
    // Three products a buffer (about a ninth of the starting limit): far shorter than `commitSeconds` on any Mac, so the
    // limit stays put.
    let limit = 3 * per
    runner.commitLimit = limit
    runner.begin()
    for _ in 0..<calls {
        try runner.mm(runner.encoder, matrix, FamilyKernels.Ref(input), FamilyKernels.Ref(output), rows: rows)
        try runner.commitIfLong()
    }
    runner.end()
    let expected = Int(Double(calls) * per / limit) + 1
    try require(
        runner.timing.commits == expected, "\(runner.timing.commits) buffers for \(Double(calls) * per) multiply-adds, expected \(expected)"
    )
    let values = output.contents().bindMemory(to: Float.self, capacity: rows * outputs)
    try require((0..<rows * outputs).allSatisfy { values[$0] == Float(width) }, "products after the split")
    // The indexer's slices (work that grows with the context): at most `commitLimit` each, contiguous, the open buffer
    // committed before a slice that would take it past it; a slice of one row when a row is wider than that.
    for (perRow, size) in [(limit / 10.5, 10), (2 * limit, 1), (limit / 1000, 256)] {
        runner.begin()
        let before = runner.timing.commits
        runner.encoded(0.5 * limit)
        var slices: [Range<Int>] = []
        try runner.sliced(256, perRow: perRow) { first, n in slices.append(first..<first + n) }
        runner.end()
        let expected = (256 + size - 1) / size
        try require(
            slices.count == expected
                && slices.enumerated().allSatisfy { $0.element.lowerBound == $0.offset * size && $0.element.count <= size }
                && slices.last?.upperBound == 256,
            "slices \(slices.prefix(4)) for \(perRow) a row")
        let wanted = size == 256 ? 1 : expected + 1
        try require(
            runner.timing.commits - before == wanted,
            "\(runner.timing.commits - before) buffers for \(expected) slices of \(perRow * Double(size)), expected \(wanted)")
    }
    // DeepSeek's candidate layer at 21K entries (32 heads x 128): 256 rows were one 0.9 s buffer; slices of 10 rows now.
    let deepSeek = FamilyRunner.indexWork(entries: 21000, heads: 32, dim: 128, blocks: 2625)
    try require(
        Int(FamilyRunner.commitWork / deepSeek) == 10
            && FamilyRunner.indexWork(entries: 1_048_576, heads: 32, dim: 128) > FamilyRunner.commitWork / 2,
        "indexer work \(deepSeek) a row at 21K entries")
    // A waited buffer macOS stops (faked by `interruptWait`) fails the pass as gpu_interrupted; the next pass runs clean.
    for interrupt in [true, false] {
        runner.begin()
        try runner.mm(runner.encoder, matrix, FamilyKernels.Ref(input), FamilyKernels.Ref(output), rows: rows)
        runner.interruptWait = interrupt ? 0 : nil
        do {
            try runner.flush(wait: true)
            try require(!interrupt, "the faked abort did not fail the wait")
        } catch let error as EngineError where interrupt {
            try require(
                error.code == "gpu_interrupted"
                    && String(describing: error).contains("kIOGPUCommandBufferCallbackErrorImpactingInteractivity"), "faked abort: \(error)"
            )
        }
        runner.end()
        // The retry that follows runs buffers half as long.
        try require(runner.commitLimit == limit / 2, "limit \(runner.commitLimit) after the abort")
    }
    try require(runner.interruptWait == nil, "the faked abort stayed armed")
}

/// The expert pools: a pool per record size with the same slots per layer, every record read byte for byte from the
/// file into its slot, hits and misses counted, the least recently used slot of a layer's own pool reused (decode's
/// rule; a prompt's is `familyPromptReadAhead`'s), and a fetch that cannot be served leaving no unread record behind.
func familyExpertPools() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family experts: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-experts-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let types = [1: 8, 2: 12]
    func perExpert(_ layer: Int) -> Int { GGUFQuantize.bytes(types[layer]!, count: 256 * 256) }
    func pattern(_ layer: Int, _ expert: Int, _ part: Int) -> UInt8 { UInt8(layer * 50 + expert * 10 + part) }
    let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
    let size = [1: 212_992, 2: 114_688]  // records rounded up to 16 KiB
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (size[1]! + size[2]!)))
    try require(
        experts.slots(layer: 1) == 3 && experts.slots(layer: 2) == 3 && experts.slotCount == 6
            && experts.poolBytes == 3 * (size[1]! + size[2]!),
        "pools \(experts.slots(layer: 1)), \(experts.slots(layer: 2)), \(experts.poolBytes)")
    func verify(_ layer: Int, _ asked: [Int]) throws {
        for (expert, slot) in zip(asked, try experts.fetch(layer: layer, experts: asked, decode: true)) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: perExpert(layer))
                try require(
                    slot.types[part] == types[layer]! && bytes.allSatisfy { $0 == pattern(layer, expert, part) },
                    "layer \(layer) expert \(expert) part \(part)")
            }
        }
    }
    try verify(1, [0, 1])
    try verify(2, [3])
    try verify(1, [2])
    try require(experts.counters.misses == 4 && experts.counters.hits == 0, "first reads \(experts.counters)")
    try verify(1, [1, 3])  // 1 is resident; 3 takes the least recently used slot of layer 1's pool (expert 0)
    try require(
        experts.counters.hits == 1 && experts.resident(layer: 1) == [1, 2, 3] && experts.resident(layer: 2) == [3],
        "eviction \(experts.resident(layer: 1)) \(experts.resident(layer: 2))")
    try verify(2, [3, 0, 1])
    try require(experts.resident(layer: 1) == [1, 2, 3] && experts.resident(layer: 2) == [0, 1, 3], "pools are separate")
    try require(experts.residentBytes == 3 * (size[1]! + size[2]!), "growth metadata enlarged the initial slabs")
    // `keepingLast` (a prompt's next wave read while the last one runs): expert 1, the coolest, came in last and stays.
    for keep in [false, true] {
        let two = try FamilyExperts(weights: weights, budget: UInt64(2 * (size[1]! + size[2]!)))
        _ = try two.fetch(layer: 1, experts: [0], decode: true)
        _ = try two.fetch(layer: 1, experts: [0], decode: true)
        _ = try two.fetch(layer: 1, experts: [1], decode: true)
        _ = try two.fetch(layer: 1, experts: [2], decode: true, keepingLast: keep)
        try require(two.resident(layer: 1) == (keep ? [1, 2] : [0, 2]), "keepingLast \(keep): resident \(two.resident(layer: 1))")
    }
    // More experts at once than the slots: the call fails and leaves none of the records it never read as a hit.
    let few = try FamilyExperts(weights: weights, budget: UInt64(3 * (size[1]! + size[2]!)))
    try require(
        (try? few.fetch(layer: 1, experts: [0, 1, 2, 3])) == nil && few.resident(layer: 1).isEmpty,
        "a failed fetch left \(few.resident(layer: 1))")
}

/// Pool allocation uses whole-slot bytes and marginal saved read time, protects both prompt waves, and respects
/// physical capacities. A cold start retains the old equal-layer split without letting per-pool floors overspend.
func familyPoolAllocation() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family pool allocation: " + what) } }
    func split(_ bytes: UInt64, _ values: [[Double]]? = nil, capacities: [Int] = [4, 4]) -> [Int] {
        FamilyExperts.poolLimits(bytes: bytes, sizes: [2, 1], layers: [1, 1], capacities: capacities, minimum: 1, values: values)
    }
    try require(split(6) == [2, 2], "cold split changed")
    try require(split(6, [[10, 1, 1, 1], [10, 8, 8, 8]]) == [1, 4], "reuse did not favor the smaller records")
    try require(split(7, [[10, 8, 8, 8], [10, 1, 1, 1]]) == [3, 1], "reuse did not favor the larger records")
    // Equal demand, but the large record costs four times as long to read: twice the saved latency per byte.
    try require(split(7, [[2, 2, 2, 2], [1, 1, 1, 1]]) == [3, 1], "read latency was not included")
    try require(split(0, [[2, 2, 2, 2], [1, 1, 1, 1]]) == [1, 1], "two-wave floor was lost")
    try require(split(100, [[2, 2], [1, 1, 1]], capacities: [2, 3]) == [2, 3], "capacity exceeded")
    try require(
        FamilyExperts.poolLimits(bytes: 22, sizes: [1, 10], layers: [100, 1], capacities: [100, 100], minimum: 2) == [2, 2],
        "cold floors overspent the budget")
    let rounded = FamilyExperts.poolLimits(
        bytes: 14, sizes: [2, 1], layers: [1, 1], capacities: [8, 8], minimum: 1,
        values: [Array(repeating: 4, count: 8), Array(repeating: 1, count: 8)],
        allocatedBytes: { pool, slots in UInt64((slots + 2) / 3 * 3 * (pool == 0 ? 2 : 1)) })
    let allocated = (rounded[0] + 2) / 3 * 6 + (rounded[1] + 2) / 3 * 3
    try require(allocated <= 14, "slab rounding escaped the budget: \(rounded) allocate \(allocated)")
    // A slab too large to release must not prevent the logical limit from shrinking before the next prompt wave.
    let trimProfiles: [[[Double]]?] = [nil, [Array(repeating: 2, count: 8), Array(repeating: 1, count: 8)]]
    for values in trimProfiles {
        let trimmed = FamilyExperts.poolLimits(
            bytes: 9, sizes: [2, 1], layers: [1, 1], capacities: [8, 8], minimum: 2,
            values: values, allocatedBytes: { pool, slots in UInt64(slots == 0 ? 0 : (pool == 0 ? 16 : 8)) })
        try require(trimmed == [3, 3], "an unreleasable slab prevented the logical trim: \(trimmed)")
    }
    let profiles: [[[Double]]?] = [nil, [[8, 7, 6, 5], [4, 3, 2, 1]]]
    for bytes in UInt64(0)..<128 {
        for values in profiles {
            let counts = split(bytes, values)
            let held = UInt64(counts[0] * 2 + counts[1])
            try require(counts.allSatisfy { (1...4).contains($0) } && held <= max(bytes, 3), "invalid limits \(counts) at \(bytes)")
        }
    }
}

/// A learned split stays within the byte budget and returns the exact file bytes after shrinking, growing and a
/// memory trim. Measured read costs may select either legal split; the pure policy check covers allocation direction.
func familyAdaptiveExpertPools() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family adaptive pools: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-adaptive-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let saved = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 114_688  // one slot per slab makes one slot's memory transferable between these tiny pools
    defer { FamilyExperts.slabBytes = saved }
    let full = UInt64(3 * (212_992 + 114_688))
    let pool = try FamilyExperts(weights: weights, budget: full)
    for i in 0..<256 {
        _ = try pool.fetch(layer: 2, experts: [i % 4], decode: true)
        _ = try pool.fetch(layer: 1, experts: [0], decode: true)
    }
    let hold = pool.holdGuesses()
    defer { hold.release() }
    pool.prefetch(layer: 1, experts: [3])
    let queued = pool.counters.prefetches
    pool.rebalance()
    try require(
        (2...3).contains(pool.slots(layer: 1)) && (2...4).contains(pool.slots(layer: 2))
            && UInt64(pool.poolBytes) <= full && UInt64(pool.residentBytes) <= full, "learned split exceeded its limits")
    // A rebalance need not change the split. A trim certainly does, and must cancel the suspended guess safely.
    pool.setLimit(bytes: 0)
    try require(pool.slots(layer: 1) == 2 && pool.slots(layer: 2) == 2, "trim did not preserve both waves")
    try require(queued == 1 && pool.counters.prefetchCancelled == 1, "queued guess was not cancelled")
    hold.release()
    pool.setLimit(bytes: full)
    let selected = [pool.slots(layer: 1), pool.slots(layer: 2)]
    for layer in 1...2 {
        let record = pool.bank(layer)!.bytes.reduce(0, +)
        let file = UnsafeMutableRawPointer.allocate(byteCount: record, alignment: 16)
        defer { file.deallocate() }
        let asked = Array(0..<selected[layer - 1])
        for (expert, slot) in zip(asked, try pool.fetch(layer: layer, experts: asked, decode: true)) {
            try pool.readRecord(layer: layer, expert: expert, into: file)
            try require(
                memcmp(slot.buffer.contents() + slot.offsets[0], file, record) == 0, "grown slot changed layer \(layer) expert \(expert)")
        }
    }
    try require(UInt64(pool.poolBytes) <= full && UInt64(pool.residentBytes) <= full, "filling the learned split exceeded its budget")
    pool.rebalance()
    try require([pool.slots(layer: 1), pool.slots(layer: 2)] == selected, "too few new samples changed the split")
    pool.setLimit(bytes: 0)
    try require(pool.slots(layer: 1) == 2 && pool.slots(layer: 2) == 2, "trim did not preserve both waves")
    pool.setLimit(bytes: full)
    try require([pool.slots(layer: 1), pool.slots(layer: 2)] == selected && UInt64(pool.poolBytes) <= full, "growth forgot the profile")
}

/// The victim scan by level picks the slot the full scan by decayed heat picks (the one before it, kept here): empty and
/// taken slots, ties, heat decayed past underflow and equal decayed heat from different heat and age included. The scan
/// is the family policy's over a slot core (`FamilyEviction.storageVictim`): a busy slot is reserved, being read or
/// pinned, and some records carry a storage cost (log2 of a hot copy's share of the read time) from the cost array.
func familyVictimChoice() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next(_ n: Int) -> Int {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((seed >> 33) % UInt64(n))
    }
    // Costs and the kind of busy slot come from a sequence of their own, so the trials above them stay as they were.
    var costSeed: UInt64 = 0x2545_F491_4F6C_DD1D
    func nextCost(_ n: Int) -> Int {
        costSeed = costSeed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((costSeed >> 33) % UInt64(n))
    }
    let halfLife = 96.0
    let count = 48
    // One pool of `count` slots, key s for slot s (one expert per layer, so a key's layer is the key).
    let policy = FamilyEviction(first: [0], slotBytes: [16384], slots: count, experts: 1, perToken: 1, halfLife: halfLife)
    let core = try ExpertSlotCore(
        device: device, ledger: MemoryLedger(),
        shape: ExpertSlotShape(
            layers: count, expertsPerLayer: 1,
            classes: [
                .init(
                    recordBytes: 16384, slotsPerSlab: count, byteLimit: .max, maximumByteLimit: .max, category: "check",
                    ownerSized: true)
            ], layerClasses: Array(repeating: 0, count: count)),
        policy: policy, observer: policy)
    guard let buffer = device.makeBuffer(length: count * 16384, options: .storageModeShared) else {
        throw FixtureUnavailable(description: "no Metal buffer")
    }
    core.lock.lock()
    core.insertSlabLocked(ExpertSlabStorage(buffer: buffer, retaining: buffer), sizeClass: 0, slots: count)
    core.lock.unlock()
    var tickets = [ExpertSlotTicket?](repeating: nil, count: count)
    for trial in 0..<20_000 {
        let clock = UInt64(1 + next(3)) * UInt64([200, 20_000, 400_000][trial % 3])
        let start = clock - UInt64(next(4))
        var owner = [Int?](repeating: nil, count: count)
        var heat = [Double](repeating: 0, count: count)
        var stamp = [UInt64](repeating: 0, count: count)
        var busy = [Bool](repeating: false, count: count)
        core.lock.lock()
        // The last trial's records go, pinned ones unpinned first.
        for s in 0..<count {
            guard let ticket = tickets[s] else { continue }
            for _ in 0..<core.classes[0].slots[s]!.pins { try core.unpinLocked(ticket) }
            core.freeLocked(s, sizeClass: 0)
            tickets[s] = nil
        }
        core.lock.unlock()
        for s in 0..<count {
            if next(40) == 0 && trial % 2 == 0 { continue }  // empty
            owner[s] = s
            // Small integer heats and age steps of halfLife give exact ties of decayed heat across different slots.
            heat[s] = next(3) == 0 ? Double(1 + next(4)) : 1 + Double(next(1 << 20)) / 4096
            stamp[s] =
                next(2) == 0 ? clock - UInt64(next(Int(clock))) : clock - UInt64(next(Int(clock / UInt64(halfLife)))) * UInt64(halfLife)
            if next(30) == 0 { stamp[s] = start + 1 }  // taken by this call
            busy[s] = next(25) == 0  // a guessed record being read, a reserved miss, or a record a fetch returned
            core.lock.lock()
            let ticket = core.reserveLocked(s, key: s)
            policy.didReserve(s, sizeClass: 0)
            tickets[s] = ticket
            core.lock.unlock()
            switch busy[s] ? nextCost(3) : 3 {
            case 0: break
            case 1: _ = try core.beginRead(ticket)
            default:
                _ = try core.beginRead(ticket)
                core.completeRead(ticket, verified: true)
                if busy[s] { _ = try core.pin(ticket) }
            }
        }
        var cost = [Double](repeating: 0, count: count)
        var copied: [Int: Int64] = [:]
        var logCost: [Int: Double] = [:]
        for s in 0..<count where trial % 4 != 0 && nextCost(3) == 0 {
            copied[s] = 0
            // Whole steps give exact ties with other slots' heat and age as well.
            logCost[s] = -Double(nextCost(25)) / (nextCost(2) == 0 ? 1 : 8)
            cost[s] = logCost[s]!
        }
        core.lock.lock()
        policy.heat = heat
        policy.stamp = stamp
        policy.level = (0..<count).map { log2(heat[$0]) + Double(stamp[$0]) / halfLife }
        policy.clock = clock
        policy.setStorage(hot: copied, logCost: logCost, core)
        let got = policy.storageVictim(0..<count, start: start)
        core.lock.unlock()
        // The scan before the level: every slot's decayed heat, the first of equals.
        var want = -1
        var coolest = Double.infinity
        for s in 0..<count where stamp[s] <= start && !busy[s] {
            let h = owner[s] == nil ? -1 : heat[s] * exp2(cost[s] - Double(clock - stamp[s]) / halfLife)
            if h < coolest {
                coolest = h
                want = s
                if h < 0 { break }
            }
        }
        if got != want { throw EngineError("Family victim: trial \(trial) chose slot \(got), the full scan \(want)") }
    }
}

/// Decode's guessed reads of the next layer (`FamilyExperts.prefetch`): a guessed record asked for is the file's bytes
/// and no read of its own; one still read is awaited, not read again; a guess not begun and not asked for is cancelled
/// and leaves no record behind; records being read are not resident to the quality dial; guesses never take the slots
/// the last fetch returned or ones being read; a guess never asked for is the first to go, counted as wasted. A GGUF
/// record's three parts are three reads; reads begun and used are counted by rank in the guess; a depth caps the reads.
/// `FamilyExperts.perform` (on a family read scheduler) runs every job once and `meanwhile` once, and returns after the last, spinning or not; `spin` stops at its bound;
/// the prefetch check passes with decode's waits blocking too (ORIGAMI_FAMILY_SPIN=0), as it does spinning.
func familySpinWait() throws {
    final class Ran: @unchecked Sendable {
        var counts = [Int](repeating: 0, count: 24)
        let lock = NSLock()
    }
    let scheduler = try familyCheckScheduler(jobs: 24)
    for spin in [nil, 0, 0.002, 1.0] as [Double?] {
        for count in [0, 1, 7, 24] {
            let ran = Ran()
            var meanwhile = 0
            try FamilyExperts.perform(count, on: scheduler, spin: spin, meanwhile: { meanwhile += 1 }) { i in
                usleep(UInt32(200 * (i % 3)))
                ran.lock.lock()
                ran.counts[i] += 1
                ran.lock.unlock()
            }
            ran.lock.lock()
            let counts = ran.counts
            ran.lock.unlock()
            if counts != (0..<24).map({ $0 < count ? 1 : 0 }) || meanwhile != 1 {
                throw EngineError(
                    "Family spin: \(count) jobs spinning \(String(describing: spin)) ran \(counts), meanwhile \(meanwhile) times")
            }
        }
    }
    let started = DispatchTime.now().uptimeNanoseconds
    FamilyExperts.spin(0.005) { false }
    let spun = DispatchTime.now().uptimeNanoseconds - started
    if spun < 5_000_000 || spun > 500_000_000 { throw EngineError("Family spin: a 5 ms bound spun \(spun) ns") }
    var asked = 0
    FamilyExperts.spin(10) {
        asked += 1
        return asked == 3
    }
    if asked != 3 { throw EngineError("Family spin: stopped after \(asked) polls, not when ready") }
    let spins = FamilyExperts.spins
    defer { FamilyExperts.spins = spins }
    FamilyExperts.spins = !spins
    try familyExpertPrefetch()
}

func familyExpertPrefetch() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family prefetch: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-prefetch-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    func pool() throws -> FamilyExperts { try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688))) }  // 3 slots per layer
    func record(_ experts: FamilyExperts, _ layer: Int) -> Int { experts.bank(layer)!.bytes.reduce(0, +) }
    func fetch(_ experts: FamilyExperts, _ layer: Int, _ asked: [Int]) throws {
        let per = GGUFQuantize.bytes(layer == 1 ? 8 : 12, count: 256 * 256)
        for (expert, slot) in zip(asked, try experts.fetch(layer: layer, experts: asked, decode: true)) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
                try require(
                    bytes.allSatisfy { $0 == UInt8(layer * 50 + expert * 10 + part) }, "layer \(layer) expert \(expert) part \(part)")
            }
        }
    }
    // Guessed, read, asked for: no read of its own, counted as the miss it saved; never asked for: evicted first, wasted.
    let used = try pool()
    used.prefetch(layer: 2, experts: [0])
    try fetch(used, 2, [0])
    try require(
        used.counters.prefetches == 1 && used.counters.prefetchUsed == 1 && used.counters.misses == 0 && used.counters.readBytes == 0
            && used.counters.predicted == 1 && used.counters.predictedRight == 1 && used.missed == [8: 4]
            && used.counters.guessIssued == [1] && used.counters.guessUsed == [1], "used guess \(used.counters) \(used.missed)")
    used.prefetch(layer: 2, experts: [1])
    used.waitForReads()
    try fetch(used, 2, [2])
    try fetch(used, 2, [3])  // 2 takes the empty slot, 3 the unasked guess's
    try require(
        used.resident(layer: 2) == [0, 2, 3] && used.counters.prefetchWastedBytes == record(used, 2), "wasted guess \(used.counters)")
    // Queued guesses: not resident; the one asked for is awaited, the other cancelled and forgotten.
    let queued = try pool()
    // Hold every start, so the guess the fetch asks for (promoted to demand) is still queued when the fetch waits.
    let suspended = queued.reads.holdStarts()
    queued.prefetch(layer: 1, experts: [1, 2])
    try require(queued.resident(layer: 1).isEmpty, "records being read count as resident: \(queued.resident(layer: 1))")
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { suspended.release() }
    try fetch(queued, 1, [1])
    let c = queued.counters
    try require(
        c.prefetchLate == 1 && c.prefetchUsed == 1 && c.prefetchCancelled == 1 && c.misses == 0 && c.readBytes == 0
            && c.prefetchBytes == record(queued, 1) && queued.resident(layer: 1) == [1],
        "awaited and cancelled guesses \(c) \(queued.resident(layer: 1))")
    try fetch(queued, 1, [2])
    try require(
        queued.counters.misses == 1 && queued.counters.readBytes == record(queued, 1),
        "a cancelled guess left a stale record \(queued.counters)")
    // Victims: the last fetch's slot and slots being read are never taken; with neither free the guess is dropped.
    let held = try pool()
    try fetch(held, 1, [0])
    let heldGuesses = held.holdGuesses()
    held.prefetch(layer: 1, experts: [1, 2])
    held.prefetch(layer: 1, experts: [3])
    try require(held.counters.prefetches == 2 && held.resident(layer: 1) == [0], "victims \(held.counters) \(held.resident(layer: 1))")
    heldGuesses.release()
    held.waitForReads()
    try require(held.resident(layer: 1) == [0, 1, 2], "landed guesses \(held.resident(layer: 1))")
    try fetch(held, 1, [1, 2, 0])
    // Ranks: 0 is resident, so ranks 1 and 2 are read, their parts at once; the fetch uses rank 2's.
    let ranked = try pool()
    try fetch(ranked, 1, [0])
    let rankedGuesses = ranked.holdGuesses()
    ranked.prefetch(layer: 1, experts: [0, 1, 2])
    try require(ranked.pendingReads == 6, "\(ranked.pendingReads) reads for two GGUF records")
    rankedGuesses.release()
    ranked.waitForReads()
    try fetch(ranked, 1, [0, 2])
    try require(ranked.counters.guessIssued == [0, 1, 1] && ranked.counters.guessUsed == [0, 0, 1], "ranks \(ranked.counters)")
    // A depth of one reads only the best guess not resident.
    let shallow = try pool()
    shallow.guessDepth = 1
    shallow.prefetch(layer: 1, experts: [2, 1])
    shallow.waitForReads()
    try require(
        shallow.counters.prefetches == 1 && shallow.resident(layer: 1) == [2] && shallow.counters.guessIssued == [1],
        "depth \(shallow.counters)")
    try fetch(shallow, 1, [1])
    try require(shallow.counters.misses == 1 && shallow.counters.guessUsed == [], "depth fetch \(shallow.counters)")
}

/// Several experts of one prompt wave encoded into one command buffer each read their own rows and weights, and waves
/// read while the wave before runs change nothing: a layer whose four experts run in one wave (no overlap) gives the
/// bits of one where each expert has a wave of its own, the next one read meanwhile (pools of two, three and five slots);
/// decode that guesses the next layer's experts gives the bits of decode that does not.
func familyExpertWave() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-wave-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let h = 256
    let rows = 8
    // Q8_0 matrices: the router's expert j reads only block j of a row (scale 1, codes 1); the shared expert is zero.
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { o in
            (0..<(h / 32)).flatMap { b in [0x00, code(o, b) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(o, b), count: 32) }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    let router = FamilyWeights.Matrix(buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: 4, name: "router")
    let zero = FamilyWeights.Matrix(buffer: q8(h) { _, _ in 0 }, offset: 0, type: 8, width: h, outputs: h, name: "shared")
    // Row r is ones in block r % 4, so it routes to expert r % 4: two rows per expert.
    let input = (0..<rows).flatMap { r in (0..<h).map { $0 / 32 == r % 4 ? Float(1) : 0 } }
    var misses = 0
    func run(slots: Int, pending: Int? = nil, keep: Range<Int>? = nil) throws -> [Float] {
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let runner = try FamilyRunner(weights: weights, context: 64)
        runner.experts = try FamilyExperts(weights: weights, budget: UInt64(slots * (212_992 + 114_688)))
        defer { misses = runner.experts.counters.misses }
        if pending != nil {
            let others = UInt64(weights.residentBytes) + runner.workingBytes + runner.heldState
            runner.ledger = try MemoryLedger(
                ceiling: others + UInt64(slots * (212_992 + 114_688)) + 1, transientMargin: 1, availableFloor: 1)
            runner.poolMaximum = UInt64(slots * (212_992 + 114_688))
        }
        let x = device.makeBuffer(bytes: input, length: input.count * 4, options: .storageModeShared)!
        let y = device.makeBuffer(length: input.count * 4, options: .storageModeShared)!
        runner.poolMaximum = runner.experts.budget
        try runner.prepareScratch(runner.moeScratchPlan(rows: rows, sharedWidth: h))
        if let pending {
            // The actual scratch plan replaces the startup allowance. Apply the
            // simulated pressure after that handoff so the requested cache limit
            // still means `pending` slots when the layer starts.
            let others = UInt64(weights.residentBytes) + runner.workingBytes + runner.heldState
            runner.ledger!.setUsable(others + UInt64(pending * (212_992 + 114_688)))
            _ = runner.trimExperts(idle: false)
        }
        runner.begin()
        try runner.moe(
            layer: 1, router: router, bias: [], shared: FamilyRunner.FFN(gate: zero, up: zero, down: zero), input: x, output: y, rows: rows,
            probability: { $0 }, normalize: false, scale: 1, sum: { $0 }, limit: 0, keepRows: keep)
        runner.end()
        if let pending, runner.experts.slots(layer: 1) != pending {
            throw EngineError("Family expert wave: a trim was not applied at the layer's start")
        }
        return Array(UnsafeBufferPointer(start: y.contents().bindMemory(to: Float.self, capacity: input.count), count: input.count))
    }
    let together = try run(slots: 8)
    guard together.contains(where: { $0 != 0 }), together.allSatisfy({ $0.isFinite }) else {
        throw EngineError("Family expert wave: no output")
    }
    // A prompt's last layer keeping its last row (`keepRows`): that row's bits, only the one record it routes to read.
    let kept = try run(slots: 8, keep: rows - 1..<rows)
    guard misses == 1, together[((rows - 1) * h)...].elementsEqual(kept[((rows - 1) * h)...], by: { $0.bitPattern == $1.bitPattern }) else {
        throw EngineError("Family expert wave: keeping the last row read \(misses) records or changed its bits")
    }
    // (8, 3): a trim the governor asked for while the request ran applies at the layer's start, before the wave size.
    for (slots, pending) in [(2, nil), (3, nil), (5, nil), (8, 3)] {
        let apart = try run(slots: slots, pending: pending)
        guard together.withUnsafeBytes({ a in apart.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } }) else {
            throw EngineError("Family expert wave: one wave of four experts differs from overlapped waves in \(slots) slots")
        }
    }
    // Decode guessing the next layer's experts (`next`) gives the bits of decode without. The guess from this layer's
    // input (the same input here) is the expert layer 2 asks for; one from a separate input (row 2: expert 2) reads that
    // record alone, and layer 2 still reads its own.
    func decode(guessing: Bool, from row: Int? = nil) throws -> ([Float], FamilyExperts) {
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let runner = try FamilyRunner(weights: weights, context: 64)
        runner.experts = try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688)))
        let x = device.makeBuffer(bytes: Array(input[h..<(2 * h)]), length: h * 4, options: .storageModeShared)!
        let guessIn = row.map {
            device.makeBuffer(bytes: Array(input[($0 * h)..<(($0 + 1) * h)]), length: h * 4, options: .storageModeShared)!
        }
        let outs = [1, 2].map { _ in device.makeBuffer(length: h * 4, options: .storageModeShared)! }
        let shared = FamilyRunner.FFN(gate: zero, up: zero, down: zero)
        runner.poolMaximum = runner.experts.budget
        try runner.prepareScratch(runner.moeScratchPlan(rows: 1, sharedWidth: h))
        runner.begin()
        for layer in [1, 2] {
            try runner.moe(
                layer: layer, router: router, bias: [], shared: shared, input: x, output: outs[layer - 1], rows: 1, probability: { $0 },
                normalize: false, scale: 1, sum: { $0 }, limit: 0, next: guessing && layer == 1 ? (2, router, [], guessIn) : nil)
        }
        runner.end()
        runner.experts.waitForReads()
        return (
            outs.flatMap { Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: h), count: h)) },
            runner.experts
        )
    }
    let (plain, _) = try decode(guessing: false)
    let (guessed, same) = try decode(guessing: true)
    let (apart, other) = try decode(guessing: true, from: 2)
    for out in [guessed, apart] {
        guard plain.contains(where: { $0 != 0 }),
            plain.withUnsafeBytes({ a in out.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } })
        else {
            throw EngineError("Family expert wave: decode guessing the next layer differs from decode without")
        }
    }
    guard same.counters.prefetches == 1 && same.counters.prefetchUsed == 1 && same.counters.predictedRight == 1 && same.counters.misses == 1
    else {
        throw EngineError("Family expert wave: decode's guess \(same.counters)")
    }
    guard
        other.counters.prefetches == 1 && other.counters.prefetchUsed == 0 && other.counters.misses == 2
            && other.resident(layer: 2) == [1, 2]
    else {
        throw EngineError("Family expert wave: decode's guess from a separate input \(other.counters) \(other.resident(layer: 2))")
    }
    // A prompt call of two slices gives the bits of one call per slice and reads each record once (the two calls read
    // them twice): more than 8 rows of an expert a slice (mul_mm), and 6, which unsplit would be 12 (mul_mv, not mul_mm).
    for slice in [64, 24] {
        let rows = 2 * slice
        var input = [Float](repeating: 0, count: rows * h)
        for r in 0..<rows { for i in (r % 4 * 32)..<(r % 4 * 32 + 32) { input[r * h + i] = Float(1 + r % 5) } }
        func prompt(_ calls: [Range<Int>], slice: Int, keep: Range<Int>? = nil) throws -> ([Float], Int) {
            let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
            let runner = try FamilyRunner(weights: weights, context: 64)
            runner.experts = try FamilyExperts(weights: weights, budget: UInt64(2 * (212_992 + 114_688)))
            let buffers = calls.map { call in
                (
                    device.makeBuffer(
                        bytes: Array(input[(call.lowerBound * h)..<(call.upperBound * h)]), length: call.count * h * 4,
                        options: .storageModeShared)!,
                    device.makeBuffer(length: call.count * h * 4, options: .storageModeShared)!
                )
            }
            runner.poolMaximum = runner.experts.budget
            try runner.prepareScratch(runner.moeScratchPlan(rows: calls.map(\.count).max() ?? 1, sharedWidth: h, slice: slice))
            runner.begin()
            for (call, (x, y)) in zip(calls, buffers) {
                try runner.moe(
                    layer: 1, router: router, bias: [], shared: FamilyRunner.FFN(gate: zero, up: zero, down: zero), input: x, output: y,
                    rows: call.count,
                    probability: { $0 }, normalize: false, scale: 1, sum: { $0 }, limit: 0, slice: slice, keepRows: keep)
            }
            runner.end()
            let out: [Float] = buffers.flatMap { (_, y) -> [Float] in
                Array(UnsafeBufferPointer(start: y.contents().bindMemory(to: Float.self, capacity: y.length / 4), count: y.length / 4))
            }
            return (out, runner.experts.counters.misses)
        }
        let (apart, twice) = try prompt([0..<slice, slice..<rows], slice: slice)
        let (together, once) = try prompt([0..<rows], slice: slice)
        guard apart.contains(where: { $0 != 0 }),
            apart.withUnsafeBytes({ a in together.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } })
        else {
            throw EngineError("Family expert wave: a call of two \(slice)-row slices differs from a call per slice")
        }
        guard once == 4 && twice == 8 else {
            throw EngineError("Family expert wave: \(slice)-row slices read \(once) records together, \(twice) apart (not 4 and 8)")
        }
        // Keeping the last row over both slices: its expert still runs once per slice over all its rows, the same bits.
        let (last, read) = try prompt([0..<rows], slice: slice, keep: rows - 1..<rows)
        guard read == 1, together[((rows - 1) * h)...].elementsEqual(last[((rows - 1) * h)...], by: { $0.bitPattern == $1.bitPattern })
        else {
            throw EngineError("Family expert wave: keeping the last of \(rows) rows read \(read) records or changed its bits")
        }
    }
}

/// `FamilyRunner.overlapsRouting` moves only command-buffer boundaries: two MoE layers decoded (guessing the next
/// layer's experts) and a prompt call of eight rows, with a shared expert that is not zero, give the same bits with it
/// on and off; decode commits one more buffer a layer (the routed experts) with it on.
func familyDecodeOverlap() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family decode overlap: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-overlap-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let h = 256
    // The fixture's 1 of 4 experts per row: read-ahead starts at 25 rows (`PromptReadRows.minimum`).
    let rows = 25
    // As in `familyExpertWave`: row r routes to expert r % 4; the shared expert's codes are 1 on a third of the blocks.
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { o in
            (0..<(h / 32)).flatMap { b in [0x00, code(o, b) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(o, b), count: 32) }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    let router = FamilyWeights.Matrix(buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: 4, name: "router")
    let third = FamilyWeights.Matrix(buffer: q8(h) { ($0 + $1) % 3 == 0 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: h, name: "shared")
    let shared = FamilyRunner.FFN(gate: third, up: third, down: third)
    let input = (0..<rows * h).map { i -> Float in i % h / 32 == i / h % 4 ? Float(1 + i / h % 3) / 8 : 0 }
    let saved = FamilyRunner.overlapsRouting
    defer { FamilyRunner.overlapsRouting = saved }
    func run(overlap: Bool, rows n: Int) throws -> ([Float], commits: Int) {
        FamilyRunner.overlapsRouting = overlap
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let runner = try FamilyRunner(weights: weights, context: 64)
        runner.experts = try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688)))
        let x = device.makeBuffer(
            bytes: Array(input[(n == 1 ? h : 0)..<(n == 1 ? 2 * h : n * h)]), length: n * h * 4, options: .storageModeShared)!
        let outs = [1, 2].map { _ in device.makeBuffer(length: n * h * 4, options: .storageModeShared)! }
        runner.poolMaximum = runner.experts.budget
        try runner.prepareScratch(runner.moeScratchPlan(rows: n, sharedWidth: h))
        runner.begin()
        for layer in [1, 2] {
            try runner.moe(
                layer: layer, router: router, bias: [], shared: shared, input: x, output: outs[layer - 1], rows: n, probability: { $0 },
                normalize: false, scale: 1, sum: { $0 }, limit: 0, next: layer == 1 ? (2, router, [], nil) : nil)
        }
        runner.end()
        runner.experts.waitForReads()
        return (
            outs.flatMap { Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: n * h), count: n * h)) },
            runner.timing.commits
        )
    }
    for n in [1, rows] {
        let (off, serial) = try run(overlap: false, rows: n)
        let (on, overlapped) = try run(overlap: true, rows: n)
        try require(off.contains { $0 != 0 } && off.allSatisfy(\.isFinite), "\(n) rows: no output")
        try require(
            off.withUnsafeBytes { a in on.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } },
            "\(n) rows: the outputs differ")
        try require(overlapped == serial + (n == 1 ? 2 : 0), "\(n) rows: \(overlapped) command buffers with the overlap, \(serial) without")
    }
}

/// Decode's resident experts run while the others are read (`FamilyRunner.runsResidentFirst`): three experts a token of
/// four, two tokens of two layers with the next layer's guess, so the second token's layer 1 has two resident and one
/// read (and layer 2 its guesses, landed or still read); the outputs are the bits of the layout that runs them all once
/// read, with decode's waits spinning and blocking, and the split commits once more.
func familyResidentFirst() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family resident first: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-resident-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder, used: 3)
    let h = 256
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { o in
            (0..<(h / 32)).flatMap { b in [0x00, code(o, b) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(o, b), count: 32) }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    // Expert o's logit is 32 times the input's value on block o: token A picks 3, 2, 1, token B 0, 2, 3.
    let router = FamilyWeights.Matrix(buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: 4, name: "router")
    let third = FamilyWeights.Matrix(buffer: q8(h) { ($0 + $1) % 3 == 0 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: h, name: "shared")
    let shared = FamilyRunner.FFN(gate: third, up: third, down: third)
    let tokens = [[1, 2, 3, 4], [4, 1, 3, 2]].map { v in (0..<h).map { i -> Float in i / 32 < 4 ? Float(v[i / 32]) / 256 : 0 } }
    let saved = (FamilyRunner.runsResidentFirst, FamilyExperts.spins)
    defer { (FamilyRunner.runsResidentFirst, FamilyExperts.spins) = saved }
    func run(first: Bool, spins: Bool) throws -> ([Float], commits: Int) {
        FamilyRunner.runsResidentFirst = first
        FamilyExperts.spins = spins
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let runner = try FamilyRunner(weights: weights, context: 64)
        runner.experts = try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688)))
        runner.experts.adaptiveGuesses = false
        _ = try runner.experts.prepareWave(layer: 2, maximumExperts: 3)
        var outs: [MTLBuffer] = []
        runner.poolMaximum = runner.experts.budget
        try runner.prepareScratch(runner.moeScratchPlan(rows: 1, sharedWidth: h))
        runner.begin()
        for token in tokens {
            let x = device.makeBuffer(bytes: token, length: h * 4, options: .storageModeShared)!
            for layer in [1, 2] {
                outs.append(device.makeBuffer(length: h * 4, options: .storageModeShared)!)
                try runner.moe(
                    layer: layer, router: router, bias: [], shared: shared, input: x, output: outs.last!, rows: 1, probability: { $0 },
                    normalize: false, scale: 1, sum: { $0 }, limit: 0, next: layer == 1 ? (2, router, [], nil) : nil)
            }
        }
        runner.end()
        runner.experts.waitForReads()
        try require(runner.experts.counters.hits >= 2, "\(runner.experts.counters.hits) hits")
        try require(runner.experts.counters.earlyBatches == 0, "a fetch without a delayed expert used early submission")
        return (
            outs.flatMap { Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: h), count: h)) },
            runner.timing.commits
        )
    }
    let (off, once) = try run(first: false, spins: true)
    try require(off.contains { $0 != 0 }, "no output")
    for spins in [true, false] {
        let (on, split) = try run(first: true, spins: spins)
        try require(
            off.withUnsafeBytes { a in on.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } },
            "the outputs differ (spins \(spins))")
        try require(split > once, "\(split) command buffers resident first, \(once) without")
    }
}

/// A long prompt's read of the next layer's first wave (`FamilyExperts.readAhead`) and the passed-record victims:
/// (a) layer 1 reading layer 2's first wave ahead gives both layers the bits they get without, the wave no miss of
/// its own and no extra read; (b) a second prompt sweep hits the records the first kept, where the heat rule alone
/// keeps none; (c) a record decode placed or used outlives a prompt sweep that has its own records to reuse.
func familyPromptReadAhead() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family prompt read-ahead: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-read-ahead-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let h = 256
    // The fixture's 1 of 4 experts per row: read-ahead starts at 25 rows (`PromptReadRows.minimum`).
    let rows = 25
    // As in `familyExpertWave`: row r routes to expert r % 4, the shared expert is zero.
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { o in
            (0..<(h / 32)).flatMap { b in [0x00, code(o, b) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(o, b), count: 32) }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    let router = FamilyWeights.Matrix(buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: 4, name: "router")
    let zero = FamilyWeights.Matrix(buffer: q8(h) { _, _ in 0 }, offset: 0, type: 8, width: h, outputs: h, name: "shared")
    func run(slots: Int, ahead: Bool, rows: Int = rows) throws -> ([Float], misses: Int, FamilyExperts.Counters) {
        let input = (0..<rows * h).map { i -> Float in i % h / 32 == i / h % 4 ? Float(1 + i / h % 3) : 0 }
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let runner = try FamilyRunner(weights: weights, context: 64)
        try require(runner.readAheadRows == 25, "read-ahead starts at \(runner.readAheadRows) rows, not 25")
        runner.experts = try FamilyExperts(weights: weights, budget: UInt64(slots * (212_992 + 114_688)))
        let x = device.makeBuffer(bytes: input, length: input.count * 4, options: .storageModeShared)!
        let outs = [1, 2].map { _ in device.makeBuffer(length: input.count * 4, options: .storageModeShared)! }
        let shared = FamilyRunner.FFN(gate: zero, up: zero, down: zero)
        var before = 0
        runner.poolMaximum = runner.experts.budget
        try runner.prepareScratch(runner.moeScratchPlan(rows: rows, sharedWidth: h))
        runner.begin()
        for layer in [1, 2] {
            before = runner.experts.counters.misses
            try runner.moe(
                layer: layer, router: router, bias: [], shared: shared, input: x, output: outs[layer - 1], rows: rows, probability: { $0 },
                normalize: false, scale: 1, sum: { $0 }, limit: 0, next: ahead && layer == 1 ? (2, router, [], nil) : nil)
        }
        runner.end()
        let out: [Float] = outs.flatMap {
            Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: input.count), count: input.count))
        }
        return (out, runner.experts.counters.misses - before, runner.experts.counters)
    }
    // One row short of the threshold reads nothing ahead.
    try require(try run(slots: 8, ahead: true, rows: rows - 1).2.readAheads == 0, "\(rows - 1) rows read ahead")
    // Eight slots: each layer is one wave (read ahead right after layer 1's fetch); two: four waves of one expert.
    for slots in [8, 2] {
        let wave = slots / 2
        let (plain, _, without) = try run(slots: slots, ahead: false)
        let (read, misses, with) = try run(slots: slots, ahead: true)
        try require(
            plain.contains(where: { $0 != 0 })
                && plain.withUnsafeBytes({ a in read.withUnsafeBytes { memcmp(a.baseAddress!, $0.baseAddress!, a.count) == 0 } }),
            "\(slots) slots: reading layer 2 ahead changed the outputs")
        try require(
            misses == 4 - wave && with.readAheads == wave && with.readAheadUsed == wave && with.readBytes == without.readBytes,
            "\(slots) slots: layer 2 missed \(misses) with \(with)")
    }
    // (b) Three slots per layer, waves of one: sweep both layers twice as a prompt's waves do.
    let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
    func pool() throws -> FamilyExperts { try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688))) }
    func sweep(_ experts: FamilyExperts, _ layers: [Int], _ asked: [Int], decode: Bool = false) throws {
        for layer in layers {
            for (i, expert) in asked.enumerated() {
                _ = try experts.fetch(layer: layer, experts: [expert], decode: decode, keepingLast: i > 0)
            }
        }
    }
    for decode in [false, true] {
        let experts = try pool()
        try sweep(experts, [1, 2], Array(0..<4), decode: decode)
        let first = experts.counters.hits
        try sweep(experts, [1, 2], Array(0..<4), decode: decode)
        let hits = experts.counters.hits - first
        try require(
            decode ? hits == 0 : hits >= 2 * (3 - 2 * 1),
            "\(decode ? "the heat rule" : "a prompt") sweep hit \(hits) records the sweep before kept")
    }
    // (c) Decode's record of expert 3 stays through a prompt's 0, 1, 2; a decode hit on the prompt's 2 keeps it over 1.
    let kept = try pool()
    _ = try kept.fetch(layer: 1, experts: [3], decode: true)
    try sweep(kept, [1], [0, 1, 2])
    try require(kept.resident(layer: 1) == [1, 2, 3], "a prompt evicted decode's record: \(kept.resident(layer: 1))")
    _ = try kept.fetch(layer: 1, experts: [2], decode: true)
    try sweep(kept, [1], [0])
    try require(kept.resident(layer: 1) == [0, 2, 3], "a decode hit left a prompt record a prompt victim: \(kept.resident(layer: 1))")
}

/// The drive split follows the scheduler's real queues: with reads outstanding on the hot copy's drive the next
/// copy-held record goes to the GGUF's, with them on the GGUF's it goes to the copy, and a batch of GGUF-only records
/// sends nothing to the copy however idle it is.
func familyDriveSplitOutstanding() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family drive split: " + what) } }
    let reads = try familyCheckScheduler(jobs: 64)
    func share(held: Int, other: Int) -> Int {
        FamilyExperts.copyShare(
            held: held, other: other, record: 1_000, partJobs: 3, on: reads, priority: .demand, bandwidth: (1e6, 1e6),
            latency: (0, 0))
    }
    try require(share(held: 1, other: 0) == 1, "an idle tie did not go to the copy")
    func staged(on source: IOSource, _ body: () throws -> Void) throws {
        let hold = reads.holdStarts()
        let tickets = try (0..<4).map { _ in
            try reads.submit(
                source: source, priority: .demand, generation: reads.currentGeneration, byteCount: 4_000, operation: { _ in })
        }
        defer {
            hold.release()
            for ticket in tickets { _ = ticket.join() }
        }
        try body()
    }
    try staged(on: .internalSSD) {
        try require(share(held: 1, other: 0) == 0, "a copy-held record queued behind the copy's outstanding reads")
        try require(share(held: 3, other: 0) == 0, "copy-held records went to the busier copy")
        try require(share(held: 1, other: 20) == 1, "the batch's GGUF-only records were not counted before the copy-held one")
    }
    try staged(on: .externalSSD) {
        try require(share(held: 2, other: 0) == 2, "copy-held records queued behind the GGUF's outstanding reads")
        try require(share(held: 0, other: 5) == 0, "a GGUF-only record went to the copy")
    }
}

/// The hot copy: the most-missed records (then the most used) within its size are copied to one file, read from there
/// byte for byte while the rest come from the GGUF; a record both hold goes to the drive whose queue finishes first
/// (`copyShare`, on an idle scheduler), and one the copy holds read from the GGUF has the same bytes; misses count 4 per decode miss and 1
/// per prompt miss; a finished copy opens again for its model only; a top-up appends records and keeps the others where
/// they were; use counts survive a save and load.
func familyHotCopy() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family hot copy: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-hot-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    // Keys are layer * 4 + expert; layer 1's records take 212,992 bytes, layer 2's 114,688 (rounded to 16 KiB).
    let uses = [1 * 4 + 2: 5, 2 * 4 + 1: 4, 1 * 4 + 0: 3, 2 * 4 + 3: 1]
    let budget = UInt64(212_992 + 114_688 + 212_992)
    try require(
        HotCopy.choose(uses, source: experts, budget: budget) == [6, 9, 4],
        "choice \(HotCopy.choose(uses, source: experts, budget: budget))")
    let byMisses = HotCopy.choose(uses, misses: [5: 8, 11: 4], source: experts, budget: budget)
    try require(byMisses == [5, 11, 6], "choice by misses \(byMisses)")
    for (hot, other, bandwidth, share) in [
        (4, 0, (1.0, 1.0), 2), (3, 0, (2.0, 1.0), 2), (2, 4, (1.0, 1.0), 2), (3, 1, (1.0, 1.0), 2), (0, 3, (1.0, 1.0), 0),
        (2, 0, (1.0, 9.0), 0),
    ] {
        let got = FamilyExperts.copyShare(
            held: hot, other: other, record: 1, partJobs: 3, on: experts.reads, priority: .demand, bandwidth: bandwidth,
            latency: (0, 0))
        try require(got == share, "copyShare(\(hot), \(other), \(bandwidth)) = \(got), not \(share)")
    }
    let hot = folder.appendingPathComponent("hot")
    var written: Int64 = 0
    let copy = try HotCopy.write(source: experts, keys: [6, 9, 4], folder: hot, identity: "a", wrote: { written += $0 })
    try require(copy.offsets.count == 3 && copy.bytes == Int64(budget) && written == copy.bytes, "built \(copy.offsets) \(written)")
    experts.attach(copy)
    func pattern(_ layer: Int, _ expert: Int, _ part: Int) -> UInt8 { UInt8(layer * 50 + expert * 10 + part) }
    func fetch(_ experts: FamilyExperts, _ layer: Int, _ asked: [Int], decode: Bool = true) throws {
        let per = GGUFQuantize.bytes(layer == 1 ? 8 : 12, count: 256 * 256)
        for (expert, slot) in zip(asked, try experts.fetch(layer: layer, experts: asked, decode: decode)) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
                try require(bytes.allSatisfy { $0 == pattern(layer, expert, part) }, "layer \(layer) expert \(expert) part \(part)")
            }
        }
    }
    try fetch(experts, 1, [2, 1])
    // The split follows each drive's measured speed; re-attaching forgets the copy's, so a loaded Mac still reads from it.
    experts.attach(copy)
    try fetch(experts, 2, [1, 3])
    try fetch(experts, 1, [3], decode: false)
    let record = { (layer: Int) in experts.bank(layer)!.bytes.reduce(0, +) }
    try require(
        experts.counters.hotReads == 2 && experts.counters.hotBytes == record(1) + record(2) && experts.counters.misses == 5,
        "hot reads \(experts.counters)")
    try require(experts.uses == [6: 1, 5: 1, 9: 1, 11: 1], "decode uses \(experts.uses)")
    try require(experts.missed == [6: 4, 5: 4, 9: 4, 11: 4, 7: 1], "misses \(experts.missed)")
    // Both records of layer 1 the copy holds miss at once: one from each drive, the GGUF's in its three parts.
    let split = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    split.attach(copy)
    try fetch(split, 1, [0, 2])
    try require(split.counters.hotReads == 1 && split.counters.hotBytes == record(1), "split \(split.counters)")
    try require(
        HotCopy.open(folder: hot, identity: "a")?.keys == [6, 9, 4] && HotCopy.open(folder: hot, identity: "b") == nil, "open")
    // A top-up: record 11 appended after the three, which keep their offsets; every record reads back as the GGUF's.
    let grown = try HotCopy.write(source: experts, keys: [11], folder: hot, identity: "a", onto: copy)
    try require(
        grown.keys == [6, 9, 4, 11] && grown.bytes == copy.bytes + 114_688
            && [6, 9, 4].allSatisfy { grown.offsets[$0] == copy.offsets[$0] },
        "top-up \(grown.keys) \(grown.bytes)")
    let reopened = HotCopy.open(folder: hot, identity: "a")
    try require(reopened?.keys == grown.keys && reopened?.bytes == grown.bytes, "top-up index")
    let want = UnsafeMutableRawPointer.allocate(byteCount: 212_992, alignment: 16384)
    let got = UnsafeMutableRawPointer.allocate(byteCount: 212_992, alignment: 16384)
    defer {
        want.deallocate()
        got.deallocate()
    }
    for key in grown.keys {
        let bytes = record(key / 4)
        try experts.readRecord(layer: key / 4, expert: key % 4, into: want)
        try grown.file.read(offset: grown.offsets[key]!, into: UnsafeMutableRawBufferPointer(start: got, count: bytes))
        try require(memcmp(want, got, bytes) == 0, "record \(key) bytes")
    }
    // A rebuild writes a new file under another name and leaves no other file's index with it.
    let rebuilt = try HotCopy.write(source: experts, keys: [5], folder: hot, identity: "a")
    try require(
        rebuilt.keys == [5] && HotCopy.open(folder: hot, identity: "a")?.keys == [5]
            && !FileManager.default.fileExists(atPath: hot.appendingPathComponent("experts.next").path), "rebuild")
    try require(
        rebuilt.payload != grown.payload && !FileManager.default.fileExists(atPath: grown.payload.path),
        "successful full rebuild retained the superseded payload path")
    // The unlinked old generation is still immutable for an already-open reader.
    let retainedBytes = record(1)
    try experts.readRecord(layer: 1, expert: 2, into: want)
    try grown.file.read(offset: grown.offsets[6]!, into: UnsafeMutableRawBufferPointer(start: got, count: retainedBytes))
    try require(memcmp(want, got, retainedBytes) == 0, "full rebuild changed the old reader's bytes")
    // A build that stops part way leaves the copy as it was.
    struct Stop: Error {}
    var asked = 0
    _ = try? HotCopy.write(
        source: experts, keys: [6, 9], folder: hot, identity: "a",
        wait: {
            asked += 1
            if asked > 1 { throw Stop() }
        })
    try require(
        HotCopy.open(folder: hot, identity: "a")?.keys == [5]
            && !FileManager.default.fileExists(atPath: hot.appendingPathComponent("experts.next").path),
        "a stopped build changed the copy")
    // A copy that cannot be read (here cut short, as a drive gone fails) gives way to the GGUF: same bytes, no error.
    try FileHandle(forWritingTo: rebuilt.payload).truncate(atOffset: 16384)
    let broken = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    try require(HotCopy.open(folder: hot, identity: "a") == nil, "truncated copies must be rejected at open")
    broken.attach(rebuilt)
    try fetch(broken, 1, [1])
    // A record whose bytes differ from its checksum (a block a power cut left unwritten) fails its first read: the read
    // falls back to the GGUF's bytes and the copy drops that record; an intact one is read from the copy.
    let checked = try HotCopy.write(source: experts, keys: [6, 9], folder: hot, identity: "a")
    try require(Set(checked.checksums.keys) == [6, 9], "records written without checksums")
    let corrupt = try FileHandle(forWritingTo: checked.payload)
    try corrupt.seek(toOffset: UInt64(checked.offsets[9]!) + 100)
    try corrupt.write(contentsOf: Data([0xEE]))
    try corrupt.close()
    for _ in 0..<2 {
        let verifying = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
        verifying.attach(checked)
        try fetch(verifying, 2, [1, 3])
        try require(checked.rejectedRecords == 1 && verifying.counters.hotReads == 0, "corrupt record \(verifying.counters)")
        verifying.attach(checked)
        try fetch(verifying, 1, [2, 1])
        try require(verifying.counters.hotReads == 1, "intact record not read from the copy \(verifying.counters)")
    }
    // An index written before checksums still opens, its records read unchecked; a top-up onto it checks only its own.
    let indexURL = hot.appendingPathComponent("index.json")
    var index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as! [String: Any]
    try require(index["checksums"] != nil, "index without checksums")
    index["checksums"] = nil
    try JSONSerialization.data(withJSONObject: index).write(to: indexURL)
    guard let legacy = HotCopy.open(folder: hot, identity: "a", source: experts), legacy.checksums.isEmpty else {
        throw EngineError("Family hot copy: an index without checksums did not open")
    }
    try legacy.read(6, into: UnsafeMutableRawBufferPointer(start: got, count: record(1)))
    let toppedUp = try HotCopy.write(source: experts, keys: [5], folder: hot, identity: "a", onto: legacy)
    try require(
        Set(toppedUp.checksums.keys) == [5] && HotCopy.open(folder: hot, identity: "a")?.checksums == toppedUp.checksums,
        "top-up onto an unchecked copy: \(toppedUp.checksums)")
    let saved = folder.appendingPathComponent("uses.json")
    try HotCopy.saveUses(uses, to: saved)
    try require(
        HotCopy.loadUses(saved) == uses && HotCopy.loadUses(folder.appendingPathComponent("none.json")).isEmpty,
        "uses round trip")
}

/// `HotCopyKeeper.plan`: no copy builds; a copy grows by the chosen records it lacks (up to the size) once they reach
/// 1 GB or `Build now` asks; a size below the copy's rebuilds (a shifted ranking is followed by swaps, `HotCopyKeeper.swaps`).
/// The keeper, on its own thread: it builds after the counts arrive, pauses while a request holds the engine, attaches
/// the copy only when the engine is free, grows it, and on `off` detaches and removes it.
func familyHotCopyKeeper() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family hot copy keeper: " + what) } }
    let gb: UInt64 = 1_000_000_000
    let size = { (_: Int) in gb / 2 }
    typealias P = HotCopyKeeper.Plan
    let cases: [(String, P, P)] = [
        ("none held", HotCopyKeeper.plan(held: [], heldBytes: 0, chosen: [1, 2], budget: 2 * gb, size: size), .build([1, 2])),
        ("no counts", HotCopyKeeper.plan(held: [1], heldBytes: gb / 2, chosen: [], budget: 2 * gb, size: size), .none),
        (
            "small copy, larger choice",
            HotCopyKeeper.plan(held: [1, 2, 3, 4], heldBytes: 2 * gb, chosen: [1, 2, 3, 4, 5, 6, 7], budget: 4 * gb, size: size),
            .topUp([5, 6, 7])
        ),
        (
            "top-up under 1 GB",
            HotCopyKeeper.plan(held: [1, 2, 3, 4], heldBytes: 2 * gb, chosen: [1, 2, 3, 4, 5], budget: 4 * gb, size: size), .none
        ),
        (
            "top-up asked",
            HotCopyKeeper.plan(held: [1, 2, 3, 4], heldBytes: 2 * gb, chosen: [1, 2, 3, 4, 5], budget: 4 * gb, force: true, size: size),
            .topUp([5])
        ),
        (
            "routing drift is followed by swaps, not a rebuild",
            HotCopyKeeper.plan(held: [1, 2, 3, 4, 5], heldBytes: 5 * gb / 2, chosen: [1, 2, 3, 8, 9], budget: 3 * gb, size: size),
            .none
        ),
        (
            "lowered size", HotCopyKeeper.plan(held: [1, 2, 3, 4], heldBytes: 2 * gb, chosen: [1, 2], budget: gb, size: size),
            .rebuild([1, 2])
        ),
        (
            "top-up within the size",
            HotCopyKeeper.plan(
                held: [1, 2, 3, 4], heldBytes: 2 * gb, chosen: [1, 2, 3, 4, 5, 6, 7, 8, 9], budget: 3 * gb, force: true, size: size),
            .topUp([5, 6])
        ),
    ]
    for (name, got, want) in cases { try require(got == want, "\(name): \(got), not \(want)") }

    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-keeper-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    final class Engine: @unchecked Sendable {
        let lock = NSLock()
        var busy = true, attached: [HotCopy?] = [], events: [String] = [], off = false, gone = false
    }
    let engine = Engine()
    let hot = folder.appendingPathComponent("hot")
    let place: (_ held: (URL) -> UInt64) -> HotCopyKeeper.Place = { _ in
        engine.lock.withLock { engine.off }
            ? .none(state: "off", detail: "Turned off in the settings.")
            : engine.lock.withLock { engine.gone }
                ? .none(state: "not_connected", detail: "Not connected.")
                : .folder(hot, drive: "internal drive", budget: UInt64(212_992 + 114_688), free: 1 << 40)
    }
    let keeper = HotCopyKeeper(
        source: experts, identity: "k", copy: nil, counts: ([:], [:]), place: place,
        attach: { copy in
            engine.lock.withLock {
                guard !engine.busy else { return false }
                engine.attached.append(copy)
                experts.attach(copy)
                return true
            }
        },
        idle: { engine.lock.withLock { !engine.busy } },
        withIdle: { work in
            try engine.lock.withLock {
                guard !engine.busy else { return false }
                try work()
                return true
            }
        },
        event: { line in engine.lock.withLock { engine.events.append(line) } })
    func wait(_ what: String, _ done: () -> Bool) throws {
        let until = Date().addingTimeInterval(10)
        while !done() {
            guard Date() < until else { throw EngineError("Family hot copy keeper: timed out waiting for \(what): \(keeper.report)") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }
    func state() -> String { keeper.report["state"] as? String ?? "" }
    keeper.trigger()
    try wait("learning") { state() == "learning" }
    // A reply's counts while it still holds the engine: the build waits, then runs and attaches once the engine is free.
    keeper.trigger(counts: ([1 * 4 + 2: 5, 2 * 4 + 1: 4], [:]))
    try wait("paused") { state() == "paused" }
    try require(engine.lock.withLock { engine.attached.isEmpty } && (keeper.report["has_counts"] as? Bool) == true, "attached while busy")
    engine.lock.withLock { engine.busy = false }
    try wait("ready") { state() == "ready" && engine.lock.withLock { !engine.events.isEmpty } }
    try require(
        engine.lock.withLock { engine.attached.count == 1 && engine.attached[0]?.keys == [6, 9] } && keeper.records == 2,
        "attached \(engine.lock.withLock { engine.attached.map { $0?.keys } })")
    try require(
        engine.lock.withLock {
            engine.events.last?.hasPrefix("Hot copy ready: 2 experts") == true && engine.events.last?.contains("on This Mac") == true
        },
        "event \(engine.lock.withLock { engine.events })")
    // The attached copy serves fetches byte for byte.
    let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
    try require(
        experts.counters.hotReads == 1
            && UnsafeRawBufferPointer(start: slots[0].buffer.contents() + slots[0].offsets[0], count: 64).allSatisfy {
                $0 == 1 * 50 + 2 * 10
            },
        "fetch through the attached copy \(experts.counters)")
    // A drive gone: no longer read, its files kept; back, the same copy is read again without a rebuild.
    engine.lock.withLock { engine.gone = true }
    keeper.trigger()
    try wait("not connected") { state() == "not_connected" }
    try require(
        engine.lock.withLock { engine.attached.count == 2 && engine.attached[1] == nil } && keeper.records == 0
            && HotCopy.open(folder: hot, identity: "k") != nil, "not connected")
    engine.lock.withLock { engine.gone = false }
    keeper.trigger()
    try wait("ready again") { state() == "ready" && keeper.records == 2 }
    try require(
        engine.lock.withLock {
            engine.attached.count == 3 && engine.attached[2]?.keys == [6, 9]
                && engine.events.filter { $0.hasPrefix("Hot copy ready") }.count == 1
                && engine.events.contains { $0.hasPrefix("Hot copy detached (2 experts") }
                && engine.events.last?.hasPrefix("Hot copy attached again: 2 experts") == true
        },
        "back \(engine.lock.withLock { engine.events })")
    // Off: detached at a free point; the reusable file is retained.
    engine.lock.withLock { engine.off = true }
    keeper.trigger()
    try wait("off") { state() == "off" }
    try require(
        engine.lock.withLock { engine.attached.count == 4 && engine.attached[3] == nil } && keeper.records == 0
            && FileManager.default.fileExists(atPath: hot.path), "off retains its files")
    // Both destinations already contain a compatible copy: changing location must reuse either one.
    let other = folder.appendingPathComponent("other-hot")
    try FileManager.default.copyItem(at: hot, to: other)
    engine.lock.withLock { engine.off = false }
    keeper.trigger(place: { held in .folder(other, drive: "other", budget: held(other), free: 1 << 40) })
    try wait("existing destination") { keeper.report["path"] as? String == other.path && state() == "ready" }
    keeper.trigger(place: place)
    try wait("original destination") { keeper.report["path"] as? String == hot.path && state() == "ready" }
    try require(
        engine.lock.withLock { engine.events.filter { $0.hasPrefix("Hot copy ready") }.count == 1 },
        "changing destinations rebuilt an existing copy")
    engine.lock.withLock { engine.off = true }
    keeper.trigger()
    try wait("off after relocation") { state() == "off" }
    // Force a new location for the interrupted-build check; a retained valid copy needs no rebuild.
    try FileManager.default.removeItem(at: hot)
    // Request acquisition now serializes file creation too: a busy request admits no partial writer.
    // Stopping a paused build must leave no partial file or published copy.
    engine.lock.withLock {
        engine.off = false
        engine.busy = true
    }
    keeper.trigger(counts: ([1 * 4 + 2: 5, 2 * 4 + 1: 4], [:]), force: true)
    try wait("paused again") { state() == "paused" }
    let partial = hot.appendingPathComponent("experts.next").path
    try require(!FileManager.default.fileExists(atPath: partial), "busy request allowed a partial copy writer")
    keeper.stop(waiting: 3)
    try require(!FileManager.default.fileExists(atPath: partial) && state() == "learning", "stopped: \(keeper.report)")
}

/// A hot-copy record failing its checksum: its first read falls back to the GGUF's bytes, later reads are planned on the
/// GGUF without touching the copy, and the keeper writes it again so it is read from the copy, as from an intact one.
func familyHotCopyRepair() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family hot copy repair: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-repair-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let budget = UInt64(3 * (212_992 + 114_688))
    let source = try FamilyExperts(weights: weights, budget: budget)
    let hot = folder.appendingPathComponent("hot")
    let damaged = try HotCopy.write(source: source, keys: [6, 9], folder: hot, identity: "r")
    let clean = try HotCopy.write(source: source, keys: [6, 9], folder: folder.appendingPathComponent("intact"), identity: "r")
    let corrupt = try FileHandle(forWritingTo: damaged.payload)
    try corrupt.seek(toOffset: UInt64(damaged.offsets[9]!) + 100)
    try corrupt.write(contentsOf: Data([0xEE]))
    try corrupt.close()
    // Record 9 (layer 2, expert 1) through a fresh cache on `copy`: its bytes exact; the cache's counters.
    func fetch(_ copy: HotCopy) throws -> FamilyExperts.Counters {
        let experts = try FamilyExperts(weights: weights, budget: budget)
        experts.attach(copy)
        let slot = try experts.fetch(layer: 2, experts: [1], decode: true)[0]
        let per = GGUFQuantize.bytes(12, count: 256 * 256)
        for part in 0..<3 {
            let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
            try require(bytes.allSatisfy { $0 == UInt8(2 * 50 + 1 * 10 + part) }, "part \(part) bytes")
        }
        return experts.counters
    }
    final class Engine: @unchecked Sendable {
        let lock = NSLock()
        var attached: HotCopy?
    }
    let engine = Engine()
    engine.attached = damaged
    let keeper = HotCopyKeeper(
        source: source, identity: "r", copy: damaged, counts: ([:], [:]),
        place: { _ in .folder(hot, drive: "internal drive", budget: 1 << 30, free: 1 << 40) },
        attach: { copy in
            engine.lock.withLock { engine.attached = copy }
            return true
        }, idle: { true }, event: { _ in })
    defer { keeper.stop(waiting: 5) }
    func report(_ name: String) -> Int { keeper.report[name] as? Int ?? -1 }
    // The eviction cost of record 9: the copy's (an eighth of a read from the GGUF here) while the copy holds it, the
    // GGUF's once it is rejected (from the next request boundary), the copy's again once the repaired copy is attached.
    let costs = try FamilyExperts(weights: weights, budget: budget)
    costs.seed(copy: 8e9, model: 1e9)
    costs.attach(damaged)
    try require(costs.storageMultiplier(9) == 0.125, "cost while held: \(costs.storageMultiplier(9))")
    // One log line when the record is rejected and one when it is repaired.
    let lines = HotCopyLog()
    let saved = HotCopy.log
    HotCopy.log = { lines.add($0) }
    defer { HotCopy.log = saved }
    let first = try fetch(damaged)
    try require(
        first.demandHotJobs == 1 && first.hotReads == 0 && damaged.rejectedRecords == 1 && report("rejected_records") == 1,
        "first read \(first.demandTiming) \(first.hotReads)")
    costs.rebalance()
    try require(
        costs.storageMultiplier(9) == 1 && costs.storageMultiplier(6) == 0.125,
        "cost while rejected: \(costs.storageMultiplier(9)), \(costs.storageMultiplier(6))")
    let second = try fetch(damaged)
    try require(
        second.demandHotJobs == 0 && second.demandMainJobs == 3 && second.hotReads == 0, "planned on the copy \(second.demandTiming)")
    keeper.trigger()
    let until = Date().addingTimeInterval(10)
    while report("repaired_records") != 1 {
        guard Date() < until else { throw EngineError("Family hot copy repair: timed out: \(keeper.report)") }
        Thread.sleep(forTimeInterval: 0.02)
    }
    guard let repaired = engine.lock.withLock({ engine.attached }), repaired !== damaged else {
        throw EngineError("Family hot copy repair: no fresh copy attached")
    }
    try require(
        report("rejected_records") == 0 && repaired.keys == damaged.keys && repaired.offsets == damaged.offsets, "repaired state")
    costs.attach(repaired)
    try require(costs.storageMultiplier(9) == 0.125, "cost after repair: \(costs.storageMultiplier(9))")
    try require(
        lines.all.count == 2 && lines.all[0].contains("record 9 in \(damaged.payload.path) failed its checksum")
            && lines.all[1].contains("record 9 in \(damaged.payload.path) repaired"),
        "log: \(lines.all)")
    try require(
        (try Data(contentsOf: repaired.payload)) == (try Data(contentsOf: clean.payload)), "repaired file differs from an intact one")
    let after = try fetch(repaired)
    let reference = try fetch(clean)
    try require(
        after.hotReads == 1 && repaired.rejectedRecords == 0 && after.hotBytes == reference.hotBytes
            && after.readBytes == reference.readBytes && after.demandHotJobs == reference.demandHotJobs
            && after.demandMainJobs == reference.demandMainJobs,
        "after repair \(after.demandTiming), intact \(reference.demandTiming)")
    // A reader that opened the copy earlier still reads the rejected record from the GGUF, with exact bytes.
    let earlier = try fetch(damaged)
    try require(earlier.demandHotJobs == 0 && earlier.hotReads == 0, "earlier reader \(earlier.demandTiming)")
}

/// A start's warm-up (`FamilyExperts.warm`): records go only into empty slots, lowest first in rank order, each pool
/// keeping two prompt waves' slots empty; resident, unknown and dense-layer keys are skipped and nothing is evicted;
/// the bytes are `readRecord`'s (from the GGUF and from the hot copy), a later fetch hits them, and they count as no
/// miss; `stop` ends it after the batch being read; a request stops the serving backend's warm-up for good.
func familyWarmStart() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family warm start: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-warm-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    // Keys are layer * 4 + expert, one expert per token. 10 slots per layer: waves of 4 (a layer's experts), 8 kept empty.
    func pool(_ slots: Int) throws -> FamilyExperts { try FamilyExperts(weights: weights, budget: UInt64(slots * (212_992 + 114_688))) }
    func same(_ experts: FamilyExperts, _ layer: Int, _ asked: [Int]) throws -> [Int] {
        let record = experts.bank(layer)!.bytes.reduce(0, +)
        let want = UnsafeMutableRawPointer.allocate(byteCount: record, alignment: 16384)
        defer { want.deallocate() }
        return try zip(asked, experts.fetch(layer: layer, experts: asked, decode: true)).map { expert, slot in
            try experts.readRecord(layer: layer, expert: expert, into: want)
            try require(memcmp(slot.buffer.contents() + slot.offsets[0], want, record) == 0, "layer \(layer) expert \(expert) bytes")
            return slot.offsets[0]
        }
    }
    let experts = try pool(10)
    _ = try experts.fetch(layer: 1, experts: [3], decode: true)  // resident: layer 1 keeps 9 empty, room for 1
    let placed = experts.warm(keys: [1 * 4 + 3, 1 * 4 + 0, 1 * 4 + 2, 2 * 4 + 1, 99, 0, 2 * 4 + 0, 2 * 4 + 2, 2 * 4 + 1])
    try require(
        placed == 3 && experts.resident(layer: 1) == [3, 0] && experts.resident(layer: 2) == [1, 0],
        "placed \(placed): \(experts.resident(layer: 1)) \(experts.resident(layer: 2))")
    try require(
        experts.counters.misses == 1 && experts.counters.readBytes == experts.bank(1)!.bytes.reduce(0, +), "counted \(experts.counters)")
    let offsets = try same(experts, 2, [1, 0])
    _ = try same(experts, 1, [3, 0])
    try require(
        offsets[0] < offsets[1] && experts.counters.hits == 4 && experts.counters.misses == 1,
        "hits \(experts.counters), offsets \(offsets)")
    try require(
        experts.warm(keys: [1 * 4 + 1, 2 * 4 + 2]) == 0 && experts.resident(layer: 2) == [0, 1], "a second warm-up took the waves' slots")
    // Through the hot copy, one record a batch: `stop` asked before each batch ends it after the one being read.
    let copy = try HotCopy.write(source: experts, keys: [5, 9], folder: folder.appendingPathComponent("hot"), identity: "w")
    let stopped = try pool(12)
    stopped.attach(copy)
    var asked = 0
    try require(
        stopped.warm(keys: [5, 9, 4, 10]) {
            asked += 1
            return asked > 2
        } == 2 && asked == 3, "stop: asked \(asked)")
    try require(
        stopped.resident(layer: 1) == [1] && stopped.resident(layer: 2) == [1],
        "stopped \(stopped.resident(layer: 1)) \(stopped.resident(layer: 2))")
    _ = try same(stopped, 1, [1])
    _ = try same(stopped, 2, [1])
    try require(stopped.counters.hits == 2 && stopped.counters.hotReads == 0, "hot warm-up \(stopped.counters)")
    // The backend: a request stops the warm-up, which does not start again.
    final class Idle: FamilyModel {
        var position = 0, contextLimit = 8192, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 1 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { FamilyRunner.Timing() }
        func statusNotes() -> [String: Any] { [:] }
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [] }
        func restored(_ tokens: [Int]) {}
        func reset() throws {}
        func forward(_ tokens: [Int]) throws -> [Float] { [Float](repeating: 0, count: 261) }
    }
    let backend = FamilyServingBackend(
        model: Idle(), tokenizer: try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}"),
        stops: [259], version: "check", fastAllowed: false)
    final class Count: @unchecked Sendable {
        var placed: [Int] = []
        let lock = NSLock()
    }
    let count = Count()
    let late = try pool(12)
    let done: @Sendable (Int, Double) -> Void = { placed, _ in
        count.lock.lock()
        count.placed.append(placed)
        count.lock.unlock()
    }
    backend.warm(late, keys: [4, 5, 6, 7], done: done)
    let body: [String: Any] = ["model": "family", "max_tokens": 1, "messages": [["role": "user", "content": "hi"]]]
    _ = try backend.prepare(
        ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192), cancellation: GenerationControl())
    let first = late.resident(layer: 1)
    backend.warm(late, keys: [8, 9, 10, 11], done: done)
    _ = try backend.prepare(
        ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192), cancellation: GenerationControl())
    try require(
        count.placed.count == 2 && count.placed[0] == first.count && count.placed[1] == 0 && late.resident(layer: 1) == first
            && late.resident(layer: 2).isEmpty, "backend \(count.placed) \(first)")
    // A tuner's request (`origami_tune`, even empty) holds the hot copy's builder after it ends; an app's does not.
    try require(backend.idle, "idle after an app's reply")
    let tuned = body.merging(["origami_mode": "exact", "origami_tune": [String: Any]()]) { $1 }
    _ = try backend.prepare(
        ServingRequest.parse(JSONSerialization.data(withJSONObject: tuned), contextTier: 8192), cancellation: GenerationControl())
    try require(!backend.idle && !backend.attachHotCopy(nil, to: late), "a tuner's request holds the hot copy")
}

/// A context kept on disk: a family engine's per-position state in a mapped file, zeroed, unlinked at once (nothing
/// left in its folder), and read and written by the GPU as any buffer.
func familyContextFile() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family context file: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw FixtureUnavailable(description: "no Metal device")
    }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-context-\(getpid())")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let count = 100_000
    let mapped = try FamilyRunner.mappedBuffer(device, count: count, folder: folder)
    try require(mapped.length >= count * 4 && mapped.length % Int(getpagesize()) == 0, "length \(mapped.length)")
    try require(try FileManager.default.contentsOfDirectory(atPath: folder.path).isEmpty, "the file was left in its folder")
    let values = mapped.contents().bindMemory(to: Float.self, capacity: count)
    try require((0..<count).allSatisfy { values[$0] == 0 }, "not zeroed")
    for i in 0..<count { values[i] = Float(i) }
    guard let copy = device.makeBuffer(length: count * 4, options: .storageModeShared), let command = queue.makeCommandBuffer(),
        let blit = command.makeBlitCommandEncoder()
    else { throw EngineError("Family context file: no blit") }
    blit.copy(from: mapped, sourceOffset: 0, to: copy, destinationOffset: 0, size: count * 4)
    blit.fill(buffer: mapped, range: 0..<4096, value: 0)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    let copied = copy.contents().bindMemory(to: Float.self, capacity: count)
    try require(
        (0..<count).allSatisfy { copied[$0] == Float($0) } && values[0] == 0 && values[1023] == 0 && values[1024] == 1024,
        "the GPU did not read or write the mapped file")
}

/// The per-position state lent to the expert slots: a lowered limit forgets the records past it, releases the slabs
/// wholly past it and cancels queued guesses (a running one is waited for), never returns a slot past it, and reads
/// what the file holds; a raised one uses more slots again. A regrown state buffer keeps the old one's bytes; the
/// capacity is fitted for an empty conversation and grows geometrically, never past the context.
func familyStateLoan() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family state loan: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-loan-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let size = [1: 212_992, 2: 114_688]
    let saved = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 2 * size[1]!  // slabs of two layer-1 slots, three layer-2 ones
    defer { FamilyExperts.slabBytes = saved }
    func fetch(_ experts: FamilyExperts, _ layer: Int, _ asked: [Int]) throws {
        let record = experts.bank(layer)!.bytes.reduce(0, +)
        let file = UnsafeMutableRawPointer.allocate(byteCount: record, alignment: 16)
        defer { file.deallocate() }
        for (expert, slot) in zip(asked, try experts.fetch(layer: layer, experts: asked, decode: true)) {
            try experts.readRecord(layer: layer, expert: expert, into: file)
            try require(
                memcmp(slot.buffer.contents() + slot.offsets[0], file, record) == 0, "layer \(layer) expert \(expert) differs from the file"
            )
        }
    }
    let full = UInt64(4 * (size[1]! + size[2]!))
    let half = full / 2
    let pool = try FamilyExperts(weights: weights, budget: full)
    try fetch(pool, 1, [0, 1, 2, 3])
    try fetch(pool, 2, [0, 1])
    let hold = pool.holdGuesses()
    pool.prefetch(layer: 2, experts: [2, 3])
    try require(
        pool.slotCount == 8 && pool.residentBytes == 4 * size[1]! + 4 * size[2]!,
        "before: \(pool.slotCount) slots, \(pool.residentBytes) bytes")
    pool.setLimit(bytes: half)
    hold.release()
    try require(
        pool.slots(layer: 1) == 2 && pool.slots(layer: 2) == 2 && pool.slotCount == 4 && pool.poolBytes == 2 * (size[1]! + size[2]!),
        "lowered limits \(pool.slots(layer: 1)), \(pool.slots(layer: 2))")
    try require(
        pool.resident(layer: 1) == [0, 1] && pool.resident(layer: 2) == [0, 1] && pool.counters.prefetchCancelled == 2,
        "lowered: resident \(pool.resident(layer: 1)) \(pool.resident(layer: 2)), \(pool.counters)")
    // Layer 1's second slab and layer 2's (slot 3, taken by a guess) are released; layer 2's first holds a slot past the limit.
    try require(pool.residentBytes == 2 * size[1]! + 3 * size[2]!, "lowered: \(pool.residentBytes) bytes resident")
    try fetch(pool, 1, [2, 3])
    try require(pool.resident(layer: 1) == [2, 3] && pool.residentBytes == 2 * size[1]! + 3 * size[2]!, "a slot past the limit was used")
    try require((try? pool.fetch(layer: 1, experts: [0, 1, 2], decode: true)) == nil, "three experts fit two slots")
    pool.setLimit(bytes: full)
    try fetch(pool, 1, [0, 1])
    // The refused fetch above left expert 2 alone; 0 and 1 take an empty slot below the old limit and one past it.
    try require(
        pool.slotCount == 8 && pool.resident(layer: 1) == [0, 1, 2] && pool.residentBytes == 4 * size[1]! + 3 * size[2]!,
        "raised: \(pool.slotCount) slots, resident \(pool.resident(layer: 1)), \(pool.residentBytes) bytes")
    // Guesses running when the limit drops are waited for or cancelled; none is left in flight or stale.
    let before = pool.counters
    pool.prefetch(layer: 2, experts: [2, 3])
    pool.setLimit(bytes: half)
    let record2 = pool.bank(2)!.bytes.reduce(0, +)
    let after = pool.counters
    try require(
        (after.prefetchCancelled - before.prefetchCancelled) + (after.prefetchWastedBytes - before.prefetchWastedBytes) / record2 == 2
            && pool.resident(layer: 2) == [0, 1], "running guesses: \(after), resident \(pool.resident(layer: 2))")
    pool.setLimit(bytes: full)
    let misses = pool.counters.misses
    try fetch(pool, 2, [2, 3])
    try require(pool.counters.misses == misses + 2, "a guess forgotten by the limit stayed a hit")
    // A regrown buffer holds the old one's bytes up to the shorter length, and zeros past them.
    let runner = try FamilyRunner(weights: weights, context: 64)
    let old = device.makeBuffer(bytes: (0..<1000).map { Float($0) + 0.5 }, length: 4000, options: .storageModeShared)!
    let grown = try runner.regrown(old, 3000)
    let shrunk = try runner.regrown(old, 500)
    try require(
        memcmp(grown.contents(), old.contents(), 4000) == 0
            && UnsafeRawBufferPointer(start: grown.contents() + 4000, count: 8000).allSatisfy { $0 == 0 }
            && memcmp(shrunk.contents(), old.contents(), 2000) == 0, "regrown buffers do not hold the old bytes")
    // (positions, position, capacity, limit) -> capacity.
    for (positions, position, capacity, limit, want) in [
        (100, 0, 4096, 32768, 4096), (5000, 0, 32768, 32768, 8192), (4097, 4096, 4096, 32768, 8192),
        (8193, 8192, 8192, 10000, 10000), (20000, 10, 4096, 131072, 20480), (4096, 5, 4096, 32768, 4096),
        (10, 0, 64, 64, 64), (300, 0, 4096, 131072, 4096),
    ] {
        let got = FamilyRunner.grownCapacity(for: positions, position: position, holding: capacity, limit: limit)
        try require(got == want, "grownCapacity(\(positions), at \(position), holding \(capacity), limit \(limit)) = \(got), not \(want)")
    }
}

/// The memory governor's hold on the expert slots: each slab is reserved in the ledger while allocated; a trim while a
/// request runs waits for `fitIfTrimmed` (a MoE layer's start, a request's end) and fits the budget of then, one while
/// idle applies at once, both releasing slabs past the target; `fitExperts` regrows the limit; never below the least.
func familyMemoryGovernor() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family memory governor: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-governor-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let size = [1: 212_992, 2: 114_688]
    let saved = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 2 * size[1]!  // slabs of two layer-1 slots, three layer-2 ones
    defer { FamilyExperts.slabBytes = saved }
    let full = UInt64(4 * (size[1]! + size[2]!))
    let half = full / 2
    let runner = try FamilyRunner(weights: weights, context: 64)
    let others = UInt64(weights.residentBytes) + runner.workingBytes + runner.heldState
    let ledger = try MemoryLedger(ceiling: others + 2 * full, transientMargin: 1, availableFloor: 1)
    _ = try ledger.reserve(bytes: others, category: "dense and state")  // as makeExperts reserves them
    var slabsReserved: Int { Int(ledger.reservedBytes - others) }
    let pool = try FamilyExperts(weights: weights, budget: full)
    pool.ledger = ledger
    runner.experts = pool
    runner.ledger = ledger
    runner.poolMaximum = full
    func fill() throws {
        // Production forwardPass drains its Metal autoreleases at the completed
        // forward boundary; this direct-cache fixture must model that boundary.
        try autoreleasepool {
            _ = try pool.fetch(layer: 1, experts: [0, 1, 2, 3], decode: true)
            _ = try pool.fetch(layer: 2, experts: [0, 1, 2, 3], decode: true)
        }
    }
    try fill()
    try require(
        pool.slotCount == 8 && slabsReserved == pool.residentBytes && pool.residentBytes == 4 * (size[1]! + size[2]!),
        "filled: \(pool.slotCount) slots, \(slabsReserved) reserved for \(pool.residentBytes) resident")
    // An escaped raw Metal view keeps a retired slab charged even after the cache
    // drops it. Other retired owners release immediately after the forward pool drains.
    var escaped: MTLBuffer? = try autoreleasepool { try pool.fetch(layer: 1, experts: [3], decode: true)[0].buffer }
    let escapedBytes = escaped!.length
    try withExtendedLifetime(escaped) {
        ledger.recordCacheReliefRequest(targetBytes: others + half)
        // Busy: nothing changes until the pending limit is applied.
        ledger.setUsable(others + half)
        try require(runner.trimExperts(idle: false) == 0 && pool.slotCount == 8, "a busy trim reported bytes it had not retired")
        autoreleasepool { runner.fitIfTrimmed() }
        try require(
            pool.slotCount == 4 && pool.residentBytes == 2 * size[1]! + 3 * size[2]!
                && slabsReserved == pool.residentBytes + escapedBytes,
            "trim applied: \(pool.slotCount) slots, \(pool.residentBytes) resident, \(slabsReserved) reserved")
        try require(
            ledger.cacheReliefSnapshot.cacheBytesReleasedSinceLastRequest == full - UInt64(slabsReserved),
            "retirement telemetry claimed an escaped owner's bytes")
    }
    escaped = nil
    try require(
        slabsReserved == pool.residentBytes
            && ledger.cacheReliefSnapshot.cacheBytesReleasedSinceLastRequest == full - UInt64(pool.residentBytes),
        "final Metal owner did not complete the deferred retirement")
    ledger.setUsable(others + full)
    runner.fitIfTrimmed()
    try require(pool.slotCount == 4, "a trim applied twice")
    // Growth at a request's start, then an idle trim at once, returning the bytes released.
    runner.fitExperts()
    try fill()
    try require(pool.slotCount == 8 && slabsReserved == pool.residentBytes, "regrown: \(pool.slotCount) slots, \(slabsReserved) reserved")
    ledger.setUsable(others + half)
    let resident = pool.residentBytes
    let released = autoreleasepool { runner.trimExperts(idle: true) }
    try require(
        pool.slotCount == 4 && released == UInt64(resident - pool.residentBytes) && released > 0 && slabsReserved == pool.residentBytes,
        "idle trim: \(pool.slotCount) slots, \(released) released")
    try require(runner.trimExperts(idle: true) == 0, "a second trim to the same target released more")
    // A trim left while a request ran fits the budget of when it applies, not of when it came.
    ledger.setUsable(others + full)
    runner.fitExperts()
    try fill()
    ledger.setUsable(others + half)
    _ = runner.trimExperts(idle: false)
    ledger.setUsable(others + full)
    runner.fitIfTrimmed()
    try require(pool.slotCount == 8, "a stale trim lowered the slots")
    // Never below two layers' experts per token in every pool.
    ledger.setUsable(1)
    _ = runner.trimExperts(idle: true)
    let least = 2 * weights.geometry.expertsPerToken
    try require(
        pool.slotCount == 2 * least && pool.leastBytes == UInt64(least * (size[1]! + size[2]!)),
        "floor: \(pool.slotCount) slots, least \(pool.leastBytes)")
}

/// The family serving backend and the quality dial, on a stub model whose greedy pick is always "H" (72): prompts run
/// exact and replies at the request's level; a conversation continues from exact rows or rows of its own level, never
/// exact from non-exact ones; teacher forcing scores the greedy picks and continues from the teacher's tokens; a
/// configuration without the fast allowance refuses every non-exact level; `dynamic` decodes at fast's knobs after a
/// confident token and exact after an unsure one; a reply's tool call decodes exact from its opener on.
func familyServingDial() throws {
    // The stub model's state lives in Metal buffers.
    guard MTLCreateSystemDefaultDevice() != nil else { throw FixtureUnavailable(description: "no Metal device") }
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family serving: " + what) } }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 8192, resets = 0, routing = QualityDial.exact, confident = true
        var shouldContinue: () -> Bool = { true }
        var forwards: [(rows: Int, dial: QualityDial)] = []
        /// With a tail, the rows of each slice only fed (not in `forwards`).
        var promptTail: Int?, fed: [Int] = []
        var promptRows = FamilyRunner.promptSlice
        func feed(_ tokens: [Int]) throws {
            _ = try forward(tokens)
            fed.append(forwards.removeLast().rows)
        }
        var routedExperts: Int { 8 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { FamilyRunner.Timing() }
        func statusNotes() -> [String: Any] { [:] }
        /// The state: the position and the sum of every token read, so a restore can be checked against its tokens.
        let state = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16, options: .storageModeShared)!
        var restoredTokens: [Int]?
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [(state, 16)] }
        func restored(_ tokens: [Int]) {
            let saved = state.contents().bindMemory(to: Int64.self, capacity: 2)
            restoredTokens = saved[0] == tokens.count && saved[1] == tokens.reduce(0, +) ? tokens : []
            position = tokens.count
        }
        func reset() throws {
            position = 0
            resets += 1
            memset(state.contents(), 0, 16)
        }
        /// Each `reserveState`: the positions asked, the position then, and whether a restore had run.
        var reserves: [(positions: Int, at: Int, restored: Bool)] = []
        func reserveState(_ positions: Int) throws { reserves.append((positions, position, restoredTokens != nil)) }
        /// Forward calls (counted from 1, `feed`'s too) that fail as macOS's Impacting Interactivity abort would, before
        /// any state is written; `varied`: logits over a..z that follow the state, so a reply samples from it.
        var calls = 0, interruptAt: Set<Int> = [], varied = false
        /// Runs at the end of each forward call (after its state is written).
        var onForward: () -> Void = {}
        func forward(_ tokens: [Int]) throws -> [Float] {
            defer { onForward() }
            calls += 1
            if interruptAt.contains(calls) {
                throw EngineError.gpu("The GPU failed a family-engine step: layer 1 attention", EngineError.fakeInterruption)
            }
            position += tokens.count
            forwards.append((tokens.count, routing))
            let saved = state.contents().bindMemory(to: Int64.self, capacity: 2)
            saved[0] = Int64(position)
            saved[1] += Int64(tokens.reduce(0, +))
            var logits = [Float](repeating: 0, count: 261)
            logits[72] = confident ? 100 : 1  // 100: above any repetition penalty
            if varied {
                for i in 97...122 { logits[i] = Float((Int(saved[1]) &* 31 &+ i &* 17) % 13) / 4 }
                logits[72] = -50
            }
            return logits
        }
    }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}")
    // Each turn's text joins into one word, so a longer conversation's prompt extends the previous one's tokens.
    func run(
        _ backend: FamilyServingBackend, _ turns: [String], _ extra: [String: Any] = [:],
        _ control: GenerationControl = GenerationControl()
    ) throws -> GenerationStatistics {
        var body: [String: Any] = [
            "model": "family", "max_tokens": 3, "temperature": 0, "reasoning_effort": "none",
            "messages": turns.enumerated().map { ["role": $0.offset % 2 == 0 ? "user" : "assistant", "content": $0.element] },
        ]
        for (key, value) in extra { body[key] = value }
        let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
        return try backend.prepare(request, cancellation: control).run { _ in true }
    }
    let locked = FamilyServingBackend(model: Stub(), tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
    do {
        _ = try run(locked, ["Hi"], ["origami_mode": "fastest"])
        throw EngineError("Family serving: fastest ran without allow_fast")
    } catch let failure as ServingFailure { try require(failure.code == "fast_not_allowed", "refusal \(failure)") }
    try require(!locked.status().fastOptIn && (try run(locked, ["Hi"])).completionTokens == 3, "exact without allow_fast")

    let stub = Stub()
    let backend = FamilyServingBackend(model: stub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
    let a = try run(backend, ["Hi"])  // prompt [Hi]; reply H H H, the first two run
    try require(
        a.cachedTokens == 0 && a.quality.isExact && stub.forwards.map(\.rows) == [1, 1, 1], "first turn \(stub.forwards.map(\.rows))")
    let b = try run(backend, ["Hi", "HH", "HX"])  // [Hi H H H X] extends [Hi H H]
    try require(
        b.cachedTokens == 3 && stub.forwards.last!.rows == 1 && stub.forwards.dropLast().last!.rows == 1 && stub.resets == 0,
        "an exact turn continues an exact one (cached \(b.cachedTokens))")
    try require(stub.reserves.map(\.positions) == [4, 8] && stub.reserves.map(\.at) == [0, 3], "reserved \(stub.reserves) for two turns")
    stub.forwards = []
    let c = try run(backend, ["Hi", "HH", "HX", "HH", "HY"], ["origami_mode": "fastest"])
    try require(
        c.cachedTokens == 7 && c.quality == QualityDial.level("fastest"), "a fastest turn continues exact rows (cached \(c.cachedTokens))")
    try require(
        stub.forwards.first!.dial.isExact && stub.forwards.dropFirst().allSatisfy { $0.dial == QualityDial.level("fastest") },
        "the prompt runs exact, the reply at fastest: \(stub.forwards.map(\.dial.name))")
    let d = try run(backend, ["Hi", "HH", "HX", "HH", "HY", "HH", "HZ"])
    try require(
        d.cachedTokens == 0 && stub.resets == 1 && stub.forwards.last!.dial.isExact,
        "an exact turn continued fastest rows (cached \(d.cachedTokens))")
    let e = try run(backend, ["Hi"], ["origami_mode": "fast", "origami_teacher_tokens": [72, 73, 72], "origami_return_tokens": true])
    try require(
        e.teacher
            == {
                var t = TeacherAgreement()
                t.positions = 3
                t.agreed = 2
                t.firstDifference = 1
                t.tokenLogProbabilities = [0, -100, 0]
                return t
            }(),
        "teacher agreement \(String(describing: e.teacher))")
    try require(
        e.outputTokens == [72, 73, 72], "teacher forcing continues from the teacher's tokens: \(String(describing: e.outputTokens))")
    for bad: [String: Any] in [["origami_teacher_tokens": [261]], ["origami_mode": "fastest", "origami_teacher_tokens": [-1]]] {
        do {
            _ = try run(backend, ["Hi"], bad)
            throw EngineError("Family serving: accepted \(bad)")
        } catch is ServingFailure {}
    }
    // dynamic and dynamic-fastest: fast's or fastest's knobs after a confident token, exact after any other.
    for (mode, knobs) in [("dynamic", "fast"), ("dynamic-fastest", "fastest")] {
        for confident in [true, false] {
            stub.forwards = []
            stub.confident = confident
            let f = try run(backend, ["Hi"], ["origami_mode": mode])
            try require(
                f.quality == QualityDial.level(mode) && stub.forwards.first!.dial.isExact
                    && stub.forwards.dropFirst().allSatisfy { $0.dial == (confident ? QualityDial.level(knobs)! : .exact) },
                "\(mode) steps (confident \(confident)): \(stub.forwards.map(\.dial.name))")
        }
    }
    stub.confident = true
    // A tool call decodes exact from its opener on, at any level (the tiny vocabulary finds it on the text): the
    // forward after `>` and every later one are exact, and the statistics name the first exact reply token.
    let call = Array("a<tool_call>bc".utf8).map(Int.init)
    for mode in ["fast", "dynamic"] {
        stub.forwards = []
        let t = try run(backend, ["Hi"], ["origami_mode": mode, "origami_teacher_tokens": call, "max_tokens": call.count])
        let knobs = stub.forwards.dropFirst().map(\.dial)
        try require(
            t.exactFrom == 12 && knobs.count == call.count - 1 && knobs.prefix(11).allSatisfy { $0 == QualityDial.level("fast")! }
                && knobs.dropFirst(11).allSatisfy(\.isExact),
            "\(mode) tool call: exact from \(String(describing: t.exactFrom)), \(knobs.map(\.name))")
    }
    try require(
        (try run(backend, ["Hi"], ["origami_teacher_tokens": call, "max_tokens": call.count])).exactFrom == nil, "an exact reply switched")
    // A client that leaves during a decode step: the step finishes (a one-row step never stops mid-way), the request
    // stops before the next token, and the conversation stays: the next turn continues it without a reset.
    let departing = GenerationControl()
    let decodeStep = stub.calls + 2
    stub.onForward = { if stub.calls == decodeStep { departing.cancel() } }
    var left = false
    do { _ = try run(backend, ["Hi"], [:], departing) } catch { left = true }
    stub.onForward = {}
    let resetsAfterLeaving = stub.resets
    try require(left && stub.position == 2, "a cancelled reply ran on or lost its decode step (position \(stub.position))")
    let resumed = try run(backend, ["Hi", "H", "HX"])
    try require(
        resumed.cachedTokens == 2 && stub.resets == resetsAfterLeaving,
        "a between-token cancel lost the conversation (cached \(resumed.cachedTokens))")
    // A request prepared and then dropped (its client left before it ran) releases the engine.
    let abandoned = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: ["model": "family", "messages": [["role": "user", "content": "Hi"]]]),
        contextTier: 8192)
    do { _ = try backend.prepare(abandoned, cancellation: GenerationControl()) }
    try require((try run(backend, ["Hi"])).completionTokens == 3, "an abandoned request kept the engine busy")
    // What /v1/status shows of a prompt read: the reused part, and every token read once the prompt is in.
    let control = GenerationControl()
    let g = try run(backend, ["Hi", "HH", "HX"], [:], control)
    try require(
        g.cachedTokens == 3 && control.progress.restoredTokens == 3 && control.progress.completedPromptTokens == g.promptTokens,
        "prompt progress \(control.progress) for \(g.promptTokens) tokens, \(g.cachedTokens) reused")
    // Saved prompts, in a prompt cache that outlives the backend (a restart): the system prompt's end and the whole
    // prompt (with its logits) are saved; the same chat restores all of it, a new chat with that system prompt the
    // system part, and a state that fails its digest is removed, never used.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-cache-\(getpid())-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    func cached(hostPhaseTiming: Bool? = nil, location: URL? = nil) throws -> (FamilyServingBackend, Stub) {
        let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
        let cache = try PromptCache(
            root: location ?? folder, epoch: "check", entries: 3, entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2, ledger: ledger,
            capacity: 8192)
        let stub = Stub()
        let backend = FamilyServingBackend(
            model: stub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true,
            hostPhaseTiming: hostPhaseTiming)
        backend.promptCache = cache
        backend.epoch = "check"
        return (backend, stub)
    }
    let system = String(repeating: "a", count: 700)
    func chat(_ backend: FamilyServingBackend, _ user: String) throws -> GenerationStatistics {
        let request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: [
                "model": "family", "max_tokens": 3, "temperature": 0, "reasoning_effort": "none",
                "messages": [["role": "system", "content": system], ["role": "user", "content": user]],
            ]), contextTier: 8192)
        return try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
    }
    let (first, _) = try cached()
    let fresh = try chat(first, "Hi")
    let entries = first.promptCache!.entries
    try require(
        fresh.cachedTokens == 0 && entries.count == 2 && entries.contains { $0.kind == "system" && $0.tokens == 700 }
            && entries.contains { $0.kind == "prompt" && $0.tokens == fresh.promptTokens },
        "saved after the first chat: \(entries.map { "\($0.kind) \($0.tokens)" })")
    let (restarted, again) = try cached()
    let same = try chat(restarted, "Hi")
    try require(
        same.cachedTokens == same.promptTokens && again.restoredTokens?.count == same.promptTokens && again.forwards.count == 2,
        "the same chat after a restart restored it whole (cached \(same.cachedTokens) of \(same.promptTokens), forwards \(again.forwards.map(\.rows)))"
    )
    // The state is reserved for the prompt and max_tokens (at most 1,024) before the restore writes into it.
    try require(
        again.reserves.count == 1 && again.reserves[0].positions == same.promptTokens + 3 && again.reserves[0].at == 0
            && !again.reserves[0].restored,
        "reserved \(again.reserves) before a restore of \(same.promptTokens) tokens")
    let (other, otherStub) = try cached()
    let next = try chat(other, "Hello there")
    try require(
        next.cachedTokens == 700 && otherStub.restoredTokens?.count == 700,
        "a new chat with the system prompt restored it (cached \(next.cachedTokens))")
    for slot in 0..<7 {
        let data = FamilyStateFile.dataURL(folder.appendingPathComponent("entry-\(slot)"))
        if var bytes = try? Data(contentsOf: data) {
            bytes[0] ^= 0xff
            try bytes.write(to: data)
        }
    }
    let (damaged, damagedStub) = try cached()
    let afterDamage = try chat(damaged, "Hi")
    try require(
        afterDamage.cachedTokens == 0 && damagedStub.restoredTokens == nil, "a damaged state was used (cached \(afterDamage.cachedTokens))")
    let (saving, _) = try cached()
    let saveRequest = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family",
            "messages": [["role": "system", "content": system], ["role": "user", "content": "origami save notes"]],
        ]), contextTier: 8192)
    let saved = try saving.saveSystemPrompt(saveRequest, name: "notes", cancellation: GenerationControl())
    try require(
        saved.contains("**notes**") && saving.promptCache!.systemSnapshots.contains { $0.name == "notes" && $0.pinned },
        "origami save: \(saved)")

    // Leaving a conversation saves it (in a free slot while there is one, else in the slot of the prompt end it extends):
    // after a side request the chat restores its reply too. A request branching from that saved end (the same prompt again) or a no_store chat saves nothing.
    try? FileManager.default.removeItem(at: folder)
    let (leaving, leavingStub) = try cached()
    let cache = leaving.promptCache!
    func turn(_ backend: FamilyServingBackend, _ turns: [String], _ extra: [String: Any] = [:]) throws -> GenerationStatistics {
        var messages: [[String: Any]] = [["role": "system", "content": system]]
        for (index, turn) in turns.enumerated() { messages.append(["role": index % 2 == 0 ? "user" : "assistant", "content": turn]) }
        var body: [String: Any] = ["model": "family", "max_tokens": 3, "temperature": 0, "reasoning_effort": "none", "messages": messages]
        for (key, value) in extra { body[key] = value }
        return try backend.prepare(
            ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192), cancellation: GenerationControl()
        ).run { _ in true }
    }
    func kinds() -> [String] { cache.entries.map { "\($0.kind) \($0.tokens)" }.sorted() }
    let opened = try turn(leaving, ["Hi"])  // 701 tokens; the reply's first two of three are read: 703
    try require(kinds() == ["prompt 701", "system 700"], "before leaving: \(kinds())")
    _ = try run(leaving, ["Hi"])  // a side request, no system prompt: the chat's 703 beside its prompt's 701 (a slot is free)
    try require(kinds() == ["prompt 701", "prompt 703", "system 700"], "after leaving: \(kinds())")
    leavingStub.restoredTokens = nil
    let back = try turn(leaving, ["Hi", "HH", "HX"])
    try require(
        opened.promptTokens == 701 && back.cachedTokens == 703 && leavingStub.restoredTokens?.count == 703,
        "back to the chat after a side request: \(back.cachedTokens) restored, \(kinds())")
    let before = kinds()
    let regenerated = try turn(leaving, ["Hi", "HH", "HX"])  // the same prompt again branches from its saved end
    try require(
        kinds() == before && regenerated.cachedTokens == regenerated.promptTokens,
        "a regenerated turn: \(kinds()), \(regenerated.cachedTokens) restored")
    _ = try turn(leaving, ["Hello"], ["prompt_cache_policy": "no_store"])
    let unsaved = kinds()
    _ = try run(leaving, ["Hi"])
    try require(kinds() == unsaved, "a no_store chat was saved on leaving: \(unsaved) became \(kinds())")

    // A long prompt runs in slices of `FamilyRunner.promptSlice` rows.
    stub.forwards = []
    _ = try run(backend, [String(repeating: "H", count: FamilyRunner.promptSlice + 100)])
    try require(
        stub.forwards.first!.rows == FamilyRunner.promptSlice && stub.forwards.dropFirst().first!.rows < FamilyRunner.promptSlice,
        "a long prompt was not sliced: \(stub.forwards.prefix(3).map(\.rows))")
    // With a tail, whole slices more than the tail before each kept point (the system prompt's saved end, the prompt's
    // end) are only fed, at the slices' own places; a restart restores the state saved at that end.
    let longSystem = String(repeating: "b", count: 2 * FamilyRunner.promptSlice + 300)
    func longChat(_ backend: FamilyServingBackend, _ system: String = longSystem) throws -> GenerationStatistics {
        let request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: [
                "model": "family", "max_tokens": 3, "temperature": 0, "reasoning_effort": "none",
                "messages": [["role": "system", "content": system], ["role": "user", "content": "Hi"]],
            ]), contextTier: 8192)
        return try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
    }
    let (tailed, tailStub) = try cached()
    tailStub.promptTail = 100
    _ = try longChat(tailed)
    try require(
        tailStub.fed == [FamilyRunner.promptSlice, FamilyRunner.promptSlice] && tailStub.forwards.map(\.rows) == [300, 1, 1, 1]
            && tailed.promptCache!.entries.contains { $0.kind == "system" && $0.tokens == longSystem.count },
        "fed \(tailStub.fed), forwards \(tailStub.forwards.map(\.rows)), saved \(tailed.promptCache!.entries.map { "\($0.kind) \($0.tokens)" })"
    )
    let (tailRestarted, tailAgain) = try cached()
    tailAgain.promptTail = 300
    let restoredLong = try longChat(tailRestarted)
    try require(
        restoredLong.cachedTokens == restoredLong.promptTokens && tailAgain.restoredTokens?.count == restoredLong.promptTokens
            && tailAgain.fed.isEmpty,
        "a fed prompt's state was not restored whole (cached \(restoredLong.cachedTokens), fed \(tailAgain.fed))")
    let (tailOther, tailOtherStub) = try cached()
    tailOtherStub.promptTail = 2 * FamilyRunner.promptSlice + 300  // no slice lies before this tail: every slice runs whole
    let request = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 1, "temperature": 0, "reasoning_effort": "none",
            "messages": [["role": "user", "content": String(repeating: "c", count: 2 * FamilyRunner.promptSlice + 300)]],
        ]), contextTier: 8192)
    _ = try tailOther.prepare(request, cancellation: GenerationControl()).run { _ in true }
    try require(
        tailOtherStub.fed.isEmpty && tailOtherStub.forwards.map(\.rows) == [FamilyRunner.promptSlice, FamilyRunner.promptSlice, 300],
        "a tail as long as the prompt fed \(tailOtherStub.fed)")
    // A model taking two slices a call (`promptRows`): consecutive slices run together, fed ones before whole ones (the
    // stub feeds and forwards them in turn), cut where a state is saved.
    let S = FamilyRunner.promptSlice
    let merging = Stub()
    merging.promptRows = 2 * S
    let wide = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 1, "temperature": 0, "reasoning_effort": "none",
            "messages": [["role": "user", "content": String(repeating: "c", count: 5 * S + 100)]],
        ]), contextTier: 16384)
    _ = try FamilyServingBackend(model: merging, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
        .prepare(wide, cancellation: GenerationControl()).run { _ in true }
    try require(merging.forwards.map(\.rows) == [2 * S, 2 * S, S + 100], "two slices a call ran \(merging.forwards.map(\.rows))")
    let (marked, markedStub) = try cached()
    markedStub.promptRows = 2 * S
    _ = try longChat(marked, String(repeating: "d", count: S + S / 2))
    try require(
        markedStub.forwards.map(\.rows) == [S + S / 2, 1, 1, 1],
        "a saved system prompt's end did not cut the call: \(markedStub.forwards.map(\.rows))")
    for (tail, system, fed, forwards) in [(100, "e", [2 * S], [300, 1, 1, 1]), (S + 300, "f", [S], [S, 300, 1, 1, 1])] {
        let (backend, stub) = try cached()
        stub.promptTail = tail
        stub.promptRows = 2 * S
        _ = try longChat(backend, String(repeating: system, count: 2 * S + 300))
        try require(
            stub.fed == fed && stub.forwards.map(\.rows) == forwards, "tail \(tail): fed \(stub.fed), forwards \(stub.forwards.map(\.rows))"
        )
    }
    // Text only: an image is refused, never dropped silently.
    let image: [String: Any] = [
        "role": "user",
        "content": [["type": "text", "text": "Hi"], ["type": "image_url", "image_url": ["url": "data:image/png;base64,iVBORw0KGgo="]]],
    ]
    do {
        let request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: ["model": "family", "messages": [image]]), contextTier: 8192)
        _ = try backend.prepare(request, cancellation: GenerationControl())
        throw EngineError("Family serving: an image was accepted")
    } catch let failure as ServingFailure { try require(failure.code == "images_unsupported", "image refusal \(failure)") }

    // macOS interrupting the GPU (Impacting Interactivity): the request reads again from the last saved state, at most
    // twice more, and the reply (a sampled one) is bitwise the uninterrupted one, its text sent once; a third
    // interruption is a 503 the client retries.
    final class Lines: @unchecked Sendable {
        let lock = NSLock()
        var all: [String] = []
        func add(_ s: String) { lock.withLock { all.append(s) } }
    }
    let interrupted = Stub()
    let lines = Lines()
    interrupted.varied = true
    let retrying = FamilyServingBackend(model: interrupted, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
    retrying.log = { lines.add($0) }
    func sampled(_ at: Set<Int>) throws -> (text: String, stats: GenerationStatistics) {
        interrupted.calls = 0
        interrupted.interruptAt = at
        let request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: [
                "model": "family", "max_tokens": 8, "temperature": 1, "seed": 7,
                "reasoning_effort": "none", "origami_return_tokens": true,
                "messages": [["role": "user", "content": String(repeating: "H", count: S + 100)]],
            ]),
            contextTier: 8192)
        final class Text: @unchecked Sendable { var value = "" }
        let text = Text()
        let stats = try retrying.prepare(request, cancellation: GenerationControl()).run {
            text.value += $0
            return true
        }
        return (text.value, stats)
    }
    let plain = try sampled([])
    try require(
        plain.stats.completionTokens == 8 && Set(plain.text).count > 1 && interrupted.calls == 9,
        "uninterrupted: \(plain.text), \(interrupted.calls) forwards")
    // The second prompt slice; the third reply step (three tokens made, and sent); the first slice, then the second.
    for (at, said) in [
        ([2], "reading again from the start, retry 1 of 2"), ([5], "the start, then the 3 reply tokens already made, retry 1 of 2"),
        ([1, 3], "retry 2 of 2"),
    ] as [(Set<Int>, String)] {
        lines.all = []
        let again = try sampled(at)
        try require(
            again.text == plain.text && again.stats.outputTokens == plain.stats.outputTokens
                && again.stats.finishReason == plain.stats.finishReason,
            "interrupted at \(at): \(again.text) (\(String(describing: again.stats.outputTokens))), not \(plain.text)")
        try require(
            again.stats.gpuRetries == at.count && plain.stats.gpuRetries == nil,
            "interrupted at \(at): \(String(describing: again.stats.gpuRetries)) retries recorded")
        try require(
            lines.all.contains {
                $0.hasPrefix("prompt: macOS needed the GPU; reading again from")
                    && $0.hasSuffix(said + " (The GPU failed a family-engine step: layer 1 attention)")
            }
                && !lines.all.contains { $0.hasPrefix("retry: reply token") }, "interrupted at \(at), the log: \(lines.all)")
    }
    lines.all = []
    do {
        _ = try sampled([1, 2, 3])
        throw EngineError("Family serving: three interruptions answered")
    } catch let failure as ServingFailure {
        try require(
            failure.status == 503 && failure.type == "server_busy" && failure.code == "gpu_interrupted"
                && failure.description == GPURetry.message
                && failure.description.hasSuffix("please retry your request")
                && lines.all.last == "request failed: macOS interrupted the GPU 3 times; the client was asked to retry",
            "after three interruptions: \(failure.wire), \(lines.all)")
    }
    // With a prompt cache: again from the system prompt's saved end.
    let (savedRetry, savedStub) = try cached()
    savedRetry.log = { lines.add($0) }
    savedStub.interruptAt = [2]
    lines.all = []
    let retried = try longChat(savedRetry, String(repeating: "q", count: 700))
    try require(
        retried.completionTokens == 3 && savedStub.restoredTokens?.count == 700
            && lines.all.contains { $0.contains("reading again from token 700 (saved state), retry 1 of 2 (") },
        "with saved prompts: restored \(String(describing: savedStub.restoredTokens?.count)), \(lines.all)")

    // Optional host observations preserve output and stage names, retain failed attempts, and count call sites
    // independently of token count. These are stub calls/shared host buffers, never model inference.
    let timedStub = Stub()
    let timedControl = GenerationControl()
    let timed = FamilyServingBackend(
        model: timedStub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true,
        hostPhaseTiming: true)
    let unmeasured = FamilyServingBackend(
        model: Stub(), tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true,
        hostPhaseTiming: false)
    let timedResult = try run(timed, ["Hi"], ["origami_return_tokens": true], timedControl)
    let unmeasuredResult = try run(unmeasured, ["Hi"], ["origami_return_tokens": true])
    try require(
        timedResult.outputTokens == unmeasuredResult.outputTokens && timedResult.finishReason == unmeasuredResult.finishReason
            && Set(timedResult.stageSeconds.keys) == Set(unmeasuredResult.stageSeconds.keys)
            && unmeasuredResult.phaseCounters.isEmpty, "timing changed output/stages or emitted disabled counters")
    let timedCounters = timedResult.phaseCounters
    try require(
        timedCounters["host_wall.prefill_forward.calls"] == 1 && timedCounters["host_wall.decode_forward.calls"] == 2
            && timedCounters["host_wall.sample_force_replay.calls"] == 3 && timedCounters["host_wall.detokenize_tool_detection.calls"] == 3
            && timedCounters["host_wall.stop_filter.calls"] == 4 && timedCounters["host_wall.emit_callback.calls"] == 3,
        "host observations confused token, forward and parser call counts: \(timedCounters)")
    try require(
        timedControl.progress.attempts.count == 1 && timedControl.progress.attempts[0].phaseCounters == timedCounters,
        "successful attempt lost host counters")
    let (timedCached, timedTail) = try cached(hostPhaseTiming: true, location: folder.appendingPathComponent("timed"))
    timedTail.promptTail = 100
    let timedPrefill = try longChat(timedCached)
    try require(
        timedPrefill.phaseCounters["host_wall.prefill_feed.calls"] == 2
            && timedPrefill.phaseCounters["host_wall.prefill_forward.calls"] == 2
            && timedPrefill.phaseCounters["host_wall.prefill_checkpoint_save.calls"] == 2,
        "prefill feed/forward/checkpoint observations overlap or are missing: \(timedPrefill.phaseCounters)")
    let failedStub = Stub()
    let retryControl = GenerationControl()
    failedStub.interruptAt = [2]
    let timedRetry = FamilyServingBackend(
        model: failedStub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true,
        hostPhaseTiming: true)
    let afterRetry = try run(timedRetry, ["Hi"], ["origami_return_tokens": true], retryControl)
    let attempts = retryControl.progress.attempts
    try require(
        afterRetry.outputTokens == unmeasuredResult.outputTokens && attempts.count == 2
            && attempts[0].phaseCounters["host_wall.decode_forward.calls"] == 1
            && attempts[1].phaseCounters["host_wall.decode_forward.calls"] == 2,
        "failed retry work was erased or added to successful-attempt counters")
    let failedEmit = GenerationControl()
    let emitRequest = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 3,
            "temperature": 0, "reasoning_effort": "none", "messages": [["role": "user", "content": "Hi"]],
        ]), contextTier: 8192)
    do {
        _ = try timed.prepare(emitRequest, cancellation: failedEmit).run { _ in
            failedEmit.cancel()
            throw IOOperationCancelled()
        }
        throw EngineError("A failing emit callback was swallowed")
    } catch is IOOperationCancelled {}
    try require(
        failedEmit.progress.attempts.count == 1
            && failedEmit.progress.attempts[0].phaseCounters["host_wall.emit_callback.calls"] == 1,
        "failed callback did not retain its host observation")
}

/// DeepSeek's prompt tail is exact and no longer than it must be: with the rows before T fed (every layer before the
/// last key/value source, its window and compressed rows), the last row's output and every layer's window for the next
/// token are what full reads give exactly when T <= P - tail, on small geometries traced row by row; DeepSeek-V4.1-Flash's
/// (40 layers, last source 20, window 128) is 2,414 rows.
func deepSeekPromptTail() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("DeepSeek prompt tail: " + what) } }
    for (layers, source, window) in [(6, 2, 3), (5, 0, 4), (4, 3, 2), (7, 1, 1), (9, 4, 5)] {
        let tail = DeepSeekModel.promptTail(layers: layers, lastSource: source, window: window)
        let p = tail + 2 * window + 3
        for t in [0, p - tail - 1, p - tail, p - tail + 1, p - 1] {
            // input[r]: this layer's input at row r is what full reads give; a row's key/value row exists where it ran.
            var input = [Bool](repeating: true, count: p)
            var rings = true
            for l in 0..<layers {
                let kv = (0..<p).map { r in (r >= t || l <= source) && input[r] }
                rings = rings && (max(0, p - window + 1)..<p).allSatisfy { kv[$0] }
                input = (0..<p).map { r in (r >= t || l < source) && (max(0, r - window + 1)...r).allSatisfy { kv[$0] } }
            }
            try require(
                (rings && input[p - 1]) == (t <= p - tail),
                "layers \(layers), source \(source), window \(window): tail \(tail), fed rows \(t) of \(p)")
        }
    }
    let config =
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/deepseek-v4.1-flash/config.json")))
        as! [String: Any]
    let geo = try ModelGeometry(config: config)
    let tail = DeepSeekModel.promptTail(
        layers: geo.layers, lastSource: geo.lists["kv_source_layer_ids"]?.max() ?? -1, window: try geo.int("sliding_window", 0, 1...4096))
    try require(tail == 2414, "DeepSeek-V4.1-Flash tail \(tail)")
}

/// The converter's quantizers: each type decodes (GGUFQuant, bit-exact to gguf-py) back to within its expected error
/// of random weights, and degenerate blocks (all zero, all equal, all positive) encode without NaN.
func ggufQuantizeRoundTrip() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("GGUF quantize: " + what) } }
    var seed: UInt64 = 0xC0FFEE
    func gaussian() -> Float {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let u = Float(seed >> 40) / Float(1 << 24) + 1e-7
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let v = Float(seed >> 40) / Float(1 << 24)
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
    let count = 256 * 64
    let x = (0..<count).map { _ in gaussian() * 0.02 }
    // RMS error over RMS value: Q8_0 well under 1%, Q4_K about 10%, Q2_K about 30% on Gaussian weights.
    for (type, limit) in [(0, Float(0)), (30, 0.005), (8, 0.01), (12, 0.12), (10, 0.38)] as [(Int, Float)] {
        var encoded = [UInt8](repeating: 0, count: GGUFQuantize.bytes(type, count: count))
        var decoded = [Float](repeating: .nan, count: count)
        x.withUnsafeBufferPointer { src in
            encoded.withUnsafeMutableBytes { GGUFQuantize.encode(src.baseAddress!, count: count, type: type, into: $0.baseAddress!) }
        }
        encoded.withUnsafeBytes { raw in
            decoded.withUnsafeMutableBufferPointer {
                GGUFQuant.dequantize(raw.baseAddress!, type: type, count: count, into: $0.baseAddress!)
            }
        }
        let err = zip(x, decoded).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
        let norm = x.map { $0 * $0 }.reduce(0, +)
        let relative = (err / norm).squareRoot()
        try require(relative <= limit, "\(GGUFQuant.name(type)) relative RMS error \(relative) above \(limit)")
    }
    for block in [[Float](repeating: 0, count: 256), [Float](repeating: 0.5, count: 256), (0..<256).map { Float($0) / 255 }] {
        for type in [8, 10, 12] {
            var encoded = [UInt8](repeating: 0, count: GGUFQuantize.bytes(type, count: 256))
            var decoded = [Float](repeating: 0, count: 256)
            block.withUnsafeBufferPointer { src in
                encoded.withUnsafeMutableBytes { GGUFQuantize.encode(src.baseAddress!, count: 256, type: type, into: $0.baseAddress!) }
            }
            encoded.withUnsafeBytes { raw in
                decoded.withUnsafeMutableBufferPointer {
                    GGUFQuant.dequantize(raw.baseAddress!, type: type, count: 256, into: $0.baseAddress!)
                }
            }
            try require(
                decoded.allSatisfy(\.isFinite) && zip(block, decoded).allSatisfy { abs($0 - $1) <= 0.1 * max(1, abs($0)) },
                "\(GGUFQuant.name(type)) of a degenerate block decodes to \(decoded.prefix(4))")
        }
    }
}

/// The GLM converter on real data (skipped unless set): layers ORIGAMI_CONVERT_LAYERS (default 0,3) of the checkpoint at
/// ORIGAMI_CONVERT_GLM into a GGUF in the scratch folder, read back by the compatibility check and the weight loader;
/// a dense Q8_0 matrix and an expert's rows decode within quantization error of the source floats.
func familyConverterReal() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["ORIGAMI_CONVERT_GLM"] else {
        throw FixtureUnavailable(description: "set ORIGAMI_CONVERT_GLM to a GLM-5.3-Flash safetensors folder to convert two layers")
    }
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family converter: " + what) } }
    let layers = Set((environment["ORIGAMI_CONVERT_LAYERS"] ?? "0,3").split(separator: ",").compactMap { Int($0) })
    let converter = try FamilyConverter(source: URL(fileURLWithPath: path), expertType: 10, layers: layers)
    let output = CheckFixtures.scratch.appendingPathComponent("glm-convert-\(UUID().uuidString).gguf")
    defer { try? FileManager.default.removeItem(at: output) }
    let started = Date()
    try converter.convert(to: output)
    FileHandle.standardError.write(
        Data(
            String(
                format: "converter: layers %@ in %.0f s, %.2f GB\n", "\(layers.sorted())", Date().timeIntervalSince(started),
                Double(converter.outputBytes) / 1e9
            ).utf8))
    let gguf = try GGUFSource(output)
    let geometry = try ModelGeometry(gguf: gguf)
    try require(geometry.family == .glm && geometry.layers == 45 && geometry.experts == 288, "geometry \(geometry)")
    for l in layers where geometry.moe[l] {
        try require(
            gguf.tensors["blk.\(l).ffn_gate_exps.weight"]?.type == 10
                && gguf.tensors["blk.\(l).ffn_down_exps.weight"]?.shape == [288, 4096, 2048],
            "layer \(l) experts \(String(describing: gguf.tensors["blk.\(l).ffn_down_exps.weight"]))")
    }
    // Spot values: layer 0's KDA q projection (Q8_0) against the source's floats.
    var files: [String: ExactFile] = [:]
    if layers.contains(0), let tensor = gguf.tensors["blk.0.kda_q.weight"] {
        let (source, _) = try converter.floats("model.language_model.layers.0.self_attn.q_proj.weight", files: &files)
        let file = try ExactFile(url: output)
        let raw = try file.read(offset: tensor.offset, length: 4096 / 32 * 34)
        var row = [Float](repeating: 0, count: 4096)
        raw.withUnsafeBytes { r in
            row.withUnsafeMutableBufferPointer { GGUFQuant.dequantize(r.baseAddress!, type: 8, count: 4096, into: $0.baseAddress!) }
        }
        let err = zip(row, source.prefix(4096)).map { abs($0 - $1) }.max()!
        let scale = source.prefix(4096).map(abs).max()!
        try require(err <= scale / 100, "kda_q row 0 differs from the source by \(err) at scale \(scale)")
    }
}
