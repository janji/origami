import Foundation
import OrigamiEngine
import OrigamiEngineChecks

/// The converter's quantizer reproduces the MLX conversion of the official weights exactly: 16 rows of layer 0's
/// `linear_attn.in_proj_z` from Qwen/Qwen3.8-Flash-Next against the same rows of carloslfu/Qwen3.8-Flash-Next-MLX-4bit
/// (codes, scales and biases, bit for bit).
func converterQuantizerMatchesMLX() throws {
    func load(_ part: String) throws -> Data { try Data(contentsOf: CheckFixtures.file("convert/in_proj_z.\(part).bin")) }
    let source = try load("bf16")
    let rows = 16
    let columns = 2560
    let group = 64
    var codes = [UInt32](repeating: 0, count: rows * columns / 8)
    var scales = [UInt16](repeating: 0, count: rows * columns / group)
    var biases = scales
    source.withUnsafeBytes { raw in
        codes.withUnsafeMutableBufferPointer { c in
            scales.withUnsafeMutableBufferPointer { s in
                biases.withUnsafeMutableBufferPointer { b in
                    CheckpointConverter.quantize(
                        raw.baseAddress!, dtype: "BF16", rows: rows, columns: columns, group: group,
                        codes: c.baseAddress!, scales: s.baseAddress!, biases: b.baseAddress!)
                }
            }
        }
    }
    guard codes.withUnsafeBytes({ Data($0) }) == (try load("codes")), scales.withUnsafeBytes({ Data($0) }) == (try load("scales")),
        biases.withUnsafeBytes({ Data($0) }) == (try load("biases"))
    else {
        throw EngineError("The converter's 4-bit codes, scales or biases differ from the MLX conversion")
    }
}

/// A small two-shard checkpoint in the official (transformers) layout converts to the MLX layout: names, dtypes
/// and shapes; experts split into gate, up and down; zero-centred norms stored as 1 + w; conv kernels relabelled;
/// matrices kept as floats copied; the PLE table at group 32 with its config override; vision renamed; MTP dropped;
/// the tokenizer copied; the config passes the contract. A progress file from another source is ignored.
func converterLayout() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("converter-\(UUID().uuidString)", isDirectory: true)
    let source = root.appendingPathComponent("source", isDirectory: true)
    let destination = root.appendingPathComponent("converted", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try FileManager.default.copyItem(
        at: CheckFixtures.file("models/official-bf16/config.json"), to: source.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    var seed: UInt64 = 0x5EED
    func random(_ count: Int) -> [Float] {
        (0..<count).map { _ in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: seed >> 11) % 2_000_001) / 1_000_000 - 1
        }
    }
    func bf16(_ values: [Float]) -> Data { values.map(BF16.encode).withUnsafeBytes { Data($0) } }
    var originals: [String: [Float]] = [:]
    func tensor(_ name: String, _ shape: [Int]) -> (String, [Int], Data) {
        let values = random(shape.reduce(1, *)).map { BF16.decode(BF16.encode($0)) }
        originals[name] = values
        return (name, shape, bf16(values))
    }
    func write(_ file: String, _ tensors: [(String, [Int], Data)]) throws {
        var table: [String: Any] = [:]
        var body = Data()
        for (name, shape, data) in tensors {
            table[name] = ["dtype": "BF16", "shape": shape, "data_offsets": [body.count, body.count + data.count]]
            body.append(data)
        }
        let json = try JSONSerialization.data(withJSONObject: table, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        try (Data(bytes: &length, count: 8) + json + body).write(to: source.appendingPathComponent(file))
    }
    let p = "model.language_model.layers."
    let first = [
        tensor(p + "0.linear_attn.in_proj_z.weight", [4, 64]), tensor(p + "0.mlp.experts.gate_up_proj", [2, 4, 64]),
        tensor(p + "0.mlp.experts.down_proj", [2, 8, 64]), tensor(p + "3.self_attn.q_norm.weight", [8]),
        tensor(p + "0.linear_attn.norm.weight", [8]), tensor(p + "0.linear_attn.conv1d.weight", [8, 1, 4]),
        tensor(p + "0.linear_attn.A_log", [8]), tensor(p + "0.mlp.gate.weight", [4, 64]),
    ]
    // The contract wants the whole PLE table (128 shards) at group 32.
    let second =
        (0..<128).map { tensor(p + "1.ple.ple_embedding.ngram_embedding.shard_\($0).weight", [4, 64]) }
        + [
            tensor("model.visual.blocks.0.attn.qkv.weight", [4, 64]), tensor("mtp.fc_hidden.weight", [4, 64]),
            tensor("lm_head.weight", [4, 64]),
        ]
    try write("model-00001-of-00002.safetensors", first)
    try write("model-00002-of-00002.safetensors", second)
    var map: [String: String] = [:]
    for t in first { map[t.0] = "model-00001-of-00002.safetensors" }
    for t in second { map[t.0] = "model-00002-of-00002.safetensors" }
    try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(to: source.appendingPathComponent("model.safetensors.index.json"))

    let converter = try CheckpointConverter(source: source)
    // A stale progress file (another source) and a damaged shard: both are replaced.
    let partial = root.appendingPathComponent("converted.partial", isDirectory: true)
    try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
    try Data("garbage".utf8).write(to: partial.appendingPathComponent("model-00001-of-00002.safetensors"))
    try JSONSerialization.data(withJSONObject: ["source": "another", "done": ["model-00001-of-00002.safetensors"]])
        .write(to: partial.appendingPathComponent("conversion-progress.json"))
    try converter.convert(to: destination)
    guard !FileManager.default.fileExists(atPath: partial.path),
        FileManager.default.fileExists(atPath: destination.appendingPathComponent("tokenizer.json").path),
        FileManager.default.fileExists(atPath: destination.appendingPathComponent("origami-model.json").path)
    else {
        throw EngineError("Conversion left its partial folder or missed the tokenizer or manifest")
    }
    var spans: [String: TensorSpan] = [:]
    for file in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
        try ModelSource.header(of: destination.appendingPathComponent(file), file: file).forEach { spans[$0.key] = $0.value }
    }
    func bytes(_ name: String) throws -> Data {
        guard let span = spans[name] else { throw EngineError("Converted checkpoint lacks \(name)") }
        let handle = try FileHandle(forReadingFrom: destination.appendingPathComponent(span.file))
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(span.offset))
        return try handle.read(upToCount: Int(span.length)) ?? Data()
    }
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Converter: " + what) } }
    let l = "language_model.model.layers."
    let expected: [String: (String, [Int])] = [
        l + "0.linear_attn.in_proj_z.weight": ("U32", [4, 8]), l + "0.linear_attn.in_proj_z.scales": ("BF16", [4, 1]),
        l + "0.mlp.switch_mlp.gate_proj.weight": ("U32", [2, 2, 8]), l + "0.mlp.switch_mlp.up_proj.biases": ("BF16", [2, 2, 1]),
        l + "0.mlp.switch_mlp.down_proj.weight": ("U32", [2, 8, 8]), l + "3.self_attn.q_norm.weight": ("BF16", [8]),
        l + "0.linear_attn.norm.weight": ("BF16", [8]), l + "0.linear_attn.conv1d.weight": ("BF16", [8, 4, 1]),
        l + "0.linear_attn.A_log": ("BF16", [8]), l + "0.mlp.gate.weight": ("BF16", [4, 64]),
        l + "1.ple.ple_embedding.ngram_embedding.shard_0.scales": ("BF16", [4, 2]),
        "vision_tower.blocks.0.attn.qkv.weight": ("BF16", [4, 64]),
        "language_model.lm_head.weight": ("U32", [4, 8]),
    ]
    for (name, (dtype, shape)) in expected {
        try require(
            spans[name]?.dtype == dtype && spans[name]?.shape == shape,
            "\(name) is \(spans[name].map { "\($0.dtype) \($0.shape)" } ?? "absent")")
    }
    try require(spans.count == 17 + 128 * 3 + 4 && !spans.keys.contains { $0.contains("mtp") }, "\(spans.count) tensors, or MTP was kept")
    try require(
        try bytes(l + "0.linear_attn.conv1d.weight") == bf16(originals[p + "0.linear_attn.conv1d.weight"]!), "conv kernel bytes changed")
    try require(
        try bytes(l + "0.linear_attn.norm.weight") == bf16(originals[p + "0.linear_attn.norm.weight"]!), "the gated norm was changed")
    try require(
        try bytes(l + "3.self_attn.q_norm.weight") == bf16(originals[p + "3.self_attn.q_norm.weight"]!.map { $0 + 1 }),
        "q_norm is not 1 + w")
    try require(
        try bytes("vision_tower.blocks.0.attn.qkv.weight") == bf16(originals["model.visual.blocks.0.attn.qkv.weight"]!), "vision changed")
    // Gate is the first half of each expert's rows, up the second.
    let gateUp = originals[p + "0.mlp.experts.gate_up_proj"]!
    for (output, rows) in [("gate_proj", [0, 1, 4, 5]), ("up_proj", [2, 3, 6, 7])] {
        let slice = bf16(rows.flatMap { Array(gateUp[($0 * 64)..<($0 * 64 + 64)]) })
        var codes = [UInt32](repeating: 0, count: 32)
        var scales = [UInt16](repeating: 0, count: 4)
        var biases = scales
        slice.withUnsafeBytes { raw in
            codes.withUnsafeMutableBufferPointer { c in
                scales.withUnsafeMutableBufferPointer { s in
                    biases.withUnsafeMutableBufferPointer { b in
                        CheckpointConverter.quantize(
                            raw.baseAddress!, dtype: "BF16", rows: 4, columns: 64, group: 64, codes: c.baseAddress!, scales: s.baseAddress!,
                            biases: b.baseAddress!)
                    }
                }
            }
        }
        try require(
            try bytes(l + "0.mlp.switch_mlp.\(output).weight") == codes.withUnsafeBytes { Data($0) }
                && bytes(l + "0.mlp.switch_mlp.\(output).scales") == scales.withUnsafeBytes { Data($0) }, "\(output) rows differ")
    }
    let config = try Data(contentsOf: destination.appendingPathComponent("config.json"))
    let quantization = (try JSONSerialization.jsonObject(with: config) as? [String: Any])?["quantization"] as? [String: Any]
    try require(
        quantization?["group_size"] as? Int == 64
            && (quantization?["model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"] as? [String: Any])?["group_size"] as? Int == 32,
        "config quantization block \(String(describing: quantization))")
    try require(try ModelContract.validate(config: config) == .affine4Group64, "converted config fails the contract")
}

/// FP8: the official FP8 release keeps one E4M3 tensor per expert with bf16 scales per block. Experts stack into
/// switch_mlp at their rows; each value is the FP8 value times its block's scale before MLX's quantizer; bf16 tensors
/// beside them convert as before; FP8 without its scales is refused. The E4M3 table matches known encodings.
func converterFP8Experts() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Converter FP8: " + what) } }
    let table = CheckpointConverter.e4m3
    try require(
        table[0x38] == 1 && table[0xB8] == -1 && table[0x7E] == 448 && table[0x01] == 0x1p-9 && table[0x08] == 0x1p-6
            && table[0x7F].isNaN && table[0x00] == 0, "E4M3 table")
    let root = CheckFixtures.scratch.appendingPathComponent("converter-fp8-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var seed: UInt64 = 0xF8
    func byte() -> UInt8 {
        while true {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let b = UInt8(truncatingIfNeeded: seed >> 33)
            if b & 0x7F != 0x7F { return b }
        }
    }
    let block = [2, 32]
    let p = "model.language_model.layers.0.mlp.experts."
    var fp8: [String: ([Int], [UInt8], [Float])] = [:]
    func expert(_ name: String, _ shape: [Int]) {
        let scales = (0..<((shape[0] + 1) / 2 * (shape[1] / 32))).map { _ in BF16.decode(BF16.encode(Float(byte() % 50 + 1) / 64)) }
        fp8[name] = (shape, (0..<(shape[0] * shape[1])).map { _ in byte() }, scales)
    }
    for e in 0..<2 {
        expert(p + "\(e).gate_proj.weight", [4, 64])
        expert(p + "\(e).up_proj.weight", [4, 64])
        expert(p + "\(e).down_proj.weight", [8, 64])
    }
    let dense = (0..<(4 * 64)).map { _ in BF16.decode(BF16.encode(Float(byte()) / 255 - 0.5)) }
    func make(_ folder: URL, dropScale: Bool) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var config =
            try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/official-bf16/config.json")))
            as! [String: Any]
        config["quantization_config"] = ["quant_method": "fp8", "weight_block_size": block, "activation_scheme": "dynamic"]
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        var header: [String: Any] = [:]
        var body = Data()
        func put(_ name: String, _ dtype: String, _ shape: [Int], _ data: Data) {
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [body.count, body.count + data.count]]
            body.append(data)
        }
        for (name, (shape, bytes, scales)) in fp8.sorted(by: { $0.key < $1.key }) {
            put(name, "F8_E4M3", shape, Data(bytes))
            if !dropScale || name != p + "1.down_proj.weight" {
                put(name + "_scale_inv", "BF16", [(shape[0] + 1) / 2, shape[1] / 32], scales.map(BF16.encode).withUnsafeBytes { Data($0) })
            }
        }
        put(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight", "BF16", [4, 64],
            dense.map(BF16.encode).withUnsafeBytes { Data($0) })
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        try (Data(bytes: &length, count: 8) + json + body).write(to: folder.appendingPathComponent("model.safetensors"))
    }
    let source = root.appendingPathComponent("source", isDirectory: true)
    let destination = root.appendingPathComponent("converted", isDirectory: true)
    try make(source, dropScale: false)
    let converter = try CheckpointConverter(source: source, checkContract: false)
    try require(
        converter.sourceKind == "FP8" && converter.chain.hasPrefix("fp8 e4m3 (blocks 2x32)"),
        "source kind \(converter.sourceKind) / \(converter.chain)")
    try converter.convert(to: destination)
    let url = destination.appendingPathComponent("model.safetensors")
    let spans = try ModelSource.header(of: url, file: url.lastPathComponent)
    let file = try Data(contentsOf: url)
    func bytes(_ name: String) throws -> Data {
        guard let span = spans[name] else { throw EngineError("Converter FP8: no \(name) among \(spans.keys.sorted())") }
        return file.subdata(in: Int(span.offset)..<Int(span.offset + span.length))
    }
    let l = "language_model.model.layers.0."
    try require(
        spans[l + "mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 4, 8]
            && spans[l + "mlp.switch_mlp.down_proj.scales"]?.shape == [2, 8, 1]
            && spans[l + "linear_attn.in_proj_z.weight"]?.shape == [4, 8]
            && !spans.keys.contains { $0.contains("scale_inv") || $0.contains("experts") },
        "stacked shapes: \(spans.keys.sorted())")
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        var codes = Data()
        var scales = Data()
        for e in 0..<2 {
            let (shape, raw, blockScales) = fp8[p + "\(e).\(projection).weight"]!
            // Each value times the scale of its 2 x 32 block, worked out here independently of the converter.
            let values = (0..<raw.count).map { i in
                table[Int(raw[i])] * blockScales[(i / shape[1]) / 2 * (shape[1] / 32) + (i % shape[1]) / 32]
            }
            var c = [UInt32](repeating: 0, count: raw.count / 8)
            var s = [UInt16](repeating: 0, count: raw.count / 64)
            var b = s
            values.withUnsafeBytes { v in
                c.withUnsafeMutableBufferPointer { cp in
                    s.withUnsafeMutableBufferPointer { sp in
                        b.withUnsafeMutableBufferPointer { bp in
                            CheckpointConverter.quantize(
                                v.baseAddress!, dtype: "F32", rows: shape[0], columns: shape[1], group: 64, codes: cp.baseAddress!,
                                scales: sp.baseAddress!, biases: bp.baseAddress!)
                        }
                    }
                }
            }
            codes += c.withUnsafeBytes { Data($0) }
            scales += s.withUnsafeBytes { Data($0) }
        }
        try require(
            try bytes(l + "mlp.switch_mlp.\(projection).weight") == codes && bytes(l + "mlp.switch_mlp.\(projection).scales") == scales,
            "\(projection) differs from the FP8 values times their block scales")
    }
    let missing = root.appendingPathComponent("missing", isDirectory: true)
    try make(missing, dropScale: true)
    do {
        _ = try CheckpointConverter(source: missing, checkContract: false)
        throw EngineError("FP8 without scales was accepted")
    } catch let error as EngineError where error.description.contains("block scales") {}
}

/// MLX affine sources at another width: an 8-bit checkpoint (one module at 2-bit group 32 by the config's own entry)
/// converts to affine 4-bit group 64, each code decoded as q × scale + bias of its group (decoded here from 32-bit words,
/// independently of the converter's byte reader) then quantized as MLX does; a module Origami keeps in floats (the
/// router) is written as bf16 of the decoded values; a 6-bit checkpoint is refused by name.
func converterMLXAffine() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Converter MLX affine: " + what) } }
    let root = CheckFixtures.scratch.appendingPathComponent("converter-affine-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var seed: UInt64 = 0xA8
    func byte() -> UInt8 {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt8(truncatingIfNeeded: seed >> 33)
    }
    let l = "language_model.model.layers.0."
    // name: (logical shape, bits, group, codes, scales, biases)
    var tensors: [String: (shape: [Int], bits: Int, group: Int, codes: [UInt8], scales: [Float], biases: [Float])] = [:]
    func tensor(_ name: String, _ shape: [Int], bits: Int, group: Int) {
        let rows = shape.dropLast().reduce(1, *)
        let columns = shape.last!
        tensors[name] = (
            shape, bits, group, (0..<(rows * columns * bits / 8)).map { _ in byte() },
            (0..<(rows * columns / group)).map { _ in BF16.decode(BF16.encode(Float(byte() % 40 + 1) / 512)) },
            (0..<(rows * columns / group)).map { _ in BF16.decode(BF16.encode(-Float(byte() % 40) / 64)) }
        )
    }
    tensor(l + "mlp.switch_mlp.gate_proj", [2, 4, 64], bits: 8, group: 64)
    tensor(l + "mlp.gate", [4, 64], bits: 8, group: 64)
    tensor(l + "linear_attn.in_proj_z", [4, 64], bits: 2, group: 32)
    func decoded(_ t: (shape: [Int], bits: Int, group: Int, codes: [UInt8], scales: [Float], biases: [Float])) -> [Float] {
        let columns = t.shape.last!
        let count = t.shape.reduce(1, *)
        let mask = UInt32((1 << t.bits) - 1)
        return (0..<count).map { i in
            let bit = i * t.bits
            let at = bit / 32 * 4
            let word = UInt32(t.codes[at]) | UInt32(t.codes[at + 1]) << 8 | UInt32(t.codes[at + 2]) << 16 | UInt32(t.codes[at + 3]) << 24
            let g = (i / columns) * (columns / t.group) + (i % columns) / t.group
            return t.scales[g] * Float((word >> UInt32(bit % 32)) & mask) + t.biases[g]
        }
    }
    func make(_ folder: URL, bits: Int) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var config =
            try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/official-bf16/config.json")))
            as! [String: Any]
        config["quantization"] = ["group_size": 64, "bits": bits, l + "linear_attn.in_proj_z": ["group_size": 32, "bits": 2]]
        config.removeValue(forKey: "quantization_config")
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        var header: [String: Any] = [:]
        var body = Data()
        func put(_ name: String, _ dtype: String, _ shape: [Int], _ data: Data) {
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [body.count, body.count + data.count]]
            body.append(data)
        }
        for (base, t) in tensors.sorted(by: { $0.key < $1.key }) {
            put(base + ".weight", "U32", t.shape.dropLast() + [t.shape.last! * t.bits / 32], Data(t.codes))
            put(
                base + ".scales", "BF16", t.shape.dropLast() + [t.shape.last! / t.group],
                t.scales.map(BF16.encode).withUnsafeBytes { Data($0) })
            put(
                base + ".biases", "BF16", t.shape.dropLast() + [t.shape.last! / t.group],
                t.biases.map(BF16.encode).withUnsafeBytes { Data($0) })
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        try (Data(bytes: &length, count: 8) + json + body).write(to: folder.appendingPathComponent("model.safetensors"))
    }
    let source = root.appendingPathComponent("source", isDirectory: true)
    let destination = root.appendingPathComponent("converted", isDirectory: true)
    try make(source, bits: 8)
    let converter = try CheckpointConverter(source: source, checkContract: false)
    try require(converter.sourceKind == "MLX 8-bit (group 64)", "source kind \(converter.sourceKind)")
    try converter.convert(to: destination)
    let url = destination.appendingPathComponent("model.safetensors")
    let spans = try ModelSource.header(of: url, file: url.lastPathComponent)
    let file = try Data(contentsOf: url)
    func bytes(_ name: String) throws -> Data {
        guard let span = spans[name] else { throw EngineError("Converter MLX affine: no \(name) among \(spans.keys.sorted())") }
        return file.subdata(in: Int(span.offset)..<Int(span.offset + span.length))
    }
    for base in [l + "mlp.switch_mlp.gate_proj", l + "linear_attn.in_proj_z"] {
        let t = tensors[base]!
        let values = decoded(t)
        let rows = t.shape.dropLast().reduce(1, *)
        let columns = t.shape.last!
        var c = [UInt32](repeating: 0, count: values.count / 8)
        var s = [UInt16](repeating: 0, count: values.count / 64)
        var b = s
        values.withUnsafeBytes { v in
            c.withUnsafeMutableBufferPointer { cp in
                s.withUnsafeMutableBufferPointer { sp in
                    b.withUnsafeMutableBufferPointer { bp in
                        CheckpointConverter.quantize(
                            v.baseAddress!, dtype: "F32", rows: rows, columns: columns, group: 64, codes: cp.baseAddress!,
                            scales: sp.baseAddress!, biases: bp.baseAddress!)
                    }
                }
            }
        }
        try require(
            try bytes(base + ".weight") == c.withUnsafeBytes { Data($0) } && bytes(base + ".scales") == s.withUnsafeBytes { Data($0) }
                && bytes(base + ".biases") == b.withUnsafeBytes { Data($0) },
            "\(base) is not MLX's 4-bit quantization of its decoded values")
    }
    try require(
        spans[l + "mlp.gate.weight"]?.dtype == "BF16" && !spans.keys.contains(l + "mlp.gate.scales")
            && (try bytes(l + "mlp.gate.weight")) == decoded(tensors[l + "mlp.gate"]!).map(BF16.encode).withUnsafeBytes { Data($0) },
        "the router is not its decoded values in bf16")
    let six = root.appendingPathComponent("six", isDirectory: true)
    try make(six, bits: 6)
    do {
        _ = try CheckpointConverter(source: six, checkContract: false)
        throw EngineError("Converter MLX affine: a 6-bit checkpoint was accepted")
    } catch let error as EngineError where error.description.contains("6-bit") {}
}

/// GGUF: a small GGUF written here as llama.cpp's qwen4exp converter writes one (value heads tiled across key heads,
/// A_log as -exp, conv kernels squeezed, merged pairs in layer 0 and separate ones elsewhere, the indexer's q and k apart,
/// the 128 PLE shards in one table, the hash constants as metadata, its own chat template, an MTP block) converts back to
/// the tensors the checkpoint had: bf16 tensors equal, matrices equal to MLX's quantization of the original values. A 2-bit
/// IQ tensor is accepted; an encoding Origami does not read is refused by name. Expert banks under 4.5 bits per weight
/// (Q2_0) stay in the GGUF: nothing of them is converted, `gguf-experts.json` names each bank's file, offset, stride, type
/// and shape, and the config says `expert_format: gguf`; banks at 4.5 bits (Q4_0) are requantized as before; a pruned GGUF
/// (one of two experts) keeps its Q4_0 banks, its router padded with a zero row and its config at one expert.
func converterGGUF() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Converter GGUF: " + what) } }
    let root = CheckFixtures.scratch.appendingPathComponent("converter-gguf-\(UUID().uuidString)", isDirectory: true)
    let source = root.appendingPathComponent("source", isDirectory: true)
    let support = root.appendingPathComponent("support", isDirectory: true)
    let destination = root.appendingPathComponent("converted", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for folder in [source, support] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
    var seed: UInt64 = 0x99F
    func random(_ count: Int) -> [Float] {
        (0..<count).map { _ in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return BF16.decode(BF16.encode(Float(Int64(bitPattern: seed >> 11) % 2_000_001) / 1_000_000 - 1))
        }
    }
    // Two key heads with three value heads each (head dims 32), hidden 64, four hyper-connection streams, two experts.
    let kHeads = 2
    let perK = 3
    let vHeads = 6
    let dim = 32
    let inner = vHeads * dim
    let qk = 2 * kHeads * dim
    let hidden = 64
    let width = 4 * hidden
    var tensors: [(name: String, shape: [Int], type: Int, data: Data)] = []
    var original: [String: ([Int], [Float])] = [:]
    func put(_ name: String, _ shape: [Int], _ values: [Float]) { tensors.append((name, shape, 0, values.withUnsafeBytes { Data($0) })) }
    func make(_ name: String, _ shape: [Int]) -> [Float] {
        let v = random(shape.reduce(1, *))
        original[name] = (shape, v)
        return v
    }
    /// llama.cpp's `_reorder_v_heads`: tiled position p holds grouped position `grouped(p)`.
    func grouped(_ p: Int, _ d: Int) -> Int {
        let head = p / d
        return ((head % kHeads) * perK + head / kHeads) * d + p % d
    }
    func tileRows(_ values: [Float], columns: Int, from: Int, count: Int, d: Int) -> [Float] {
        var out = values
        for p in 0..<count {
            out.replaceSubrange(
                ((from + p) * columns)..<((from + p + 1) * columns),
                with: values[((from + grouped(p, d)) * columns)..<((from + grouped(p, d) + 1) * columns)])
        }
        return out
    }
    let l = { (i: Int) in "language_model.model.layers.\(i)." }
    for i in 0..<4 {
        let b = "blk.\(i)."
        let merged = i == 0
        for (side, hc) in [("attn", "attn_hyper_connection."), ("ffn", "mlp_hyper_connection.")] {
            put(b + "hc_\(side)_norm.weight", [width], make(l(i) + hc + "hc_norm.weight", [width]))
            let down = make(l(i) + hc + "input_mix_weight_down", [64, width])
            let inject = make(l(i) + hc + "block_inject_weight.weight", [4, width])
            if merged {
                put(b + "hc_\(side)_down_inject.weight", [68, width], down + inject)
            } else {
                put(b + "hc_\(side)_down.weight", [64, width], down)
                put(b + "hc_\(side)_inject.weight", [4, width], inject)
            }
            put(b + "hc_\(side)_up.weight", [width, 64], make(l(i) + hc + "input_mix_weight_up", [width, 64]))
        }
        if i < 3 {
            let a = l(i) + "linear_attn."
            let qkv = tileRows(make(a + "in_proj_qkv", [qk + inner, hidden]), columns: hidden, from: qk, count: inner, d: dim)
            let z = tileRows(make(a + "in_proj_z", [inner, hidden]), columns: hidden, from: 0, count: inner, d: dim)
            let beta = tileRows(make(a + "in_proj_b.weight", [vHeads, hidden]), columns: hidden, from: 0, count: vHeads, d: 1)
            let alpha = tileRows(make(a + "in_proj_a.weight", [vHeads, hidden]), columns: hidden, from: 0, count: vHeads, d: 1)
            if merged {
                put(b + "attn_qkv_gate.weight", [qk + 2 * inner, hidden], qkv + z)
                put(b + "ssm_beta_alpha.weight", [2 * vHeads, hidden], beta + alpha)
            } else {
                put(b + "attn_qkv.weight", [qk + inner, hidden], qkv)
                put(b + "attn_gate.weight", [inner, hidden], z)
                put(b + "ssm_beta.weight", [vHeads, hidden], beta)
                put(b + "ssm_alpha.weight", [vHeads, hidden], alpha)
            }
            let decay = make(a + "A_log", [vHeads]).map { BF16.decode(BF16.encode($0 - 1.5)) }
            original[a + "A_log"] = ([vHeads], decay)
            put(b + "ssm_a", [vHeads], (0..<vHeads).map { -exp(decay[grouped($0, 1)]) })
            let bias = make(a + "dt_bias", [vHeads])
            put(b + "ssm_dt.bias", [vHeads], (0..<vHeads).map { bias[grouped($0, 1)] })
            put(
                b + "ssm_conv1d.weight", [qk + inner, 4],
                tileRows(make(a + "conv1d.weight", [qk + inner, 4]), columns: 4, from: qk, count: inner, d: dim))
            put(b + "ssm_norm.weight", [dim], make(a + "norm.weight", [dim]))
            let out = make(a + "out_proj", [hidden, inner])
            put(b + "ssm_out.weight", [hidden, inner], (0..<(hidden * inner)).map { out[($0 / inner) * inner + grouped($0 % inner, dim)] })
        } else {
            let s = l(i) + "self_attn."
            for (from, to) in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")] {
                put(b + from + ".weight", [64, hidden], make(s + to, [64, hidden]))
            }
            put(b + "attn_q_norm.weight", [dim], make(s + "q_norm.weight", [dim]))
            put(b + "attn_k_norm.weight", [dim], make(s + "k_norm.weight", [dim]))
            let index = make(s + "indexer.index_qk_proj.weight", [12, hidden])
            put(b + "indexer.q_proj.weight", [8, hidden], Array(index[0..<(8 * hidden)]))
            put(b + "indexer.k_proj.weight", [4, hidden], Array(index[(8 * hidden)...]))
            put(b + "indexer.q_norm.weight", [16], make(s + "indexer.q_layernorm.weight", [16]))
            put(b + "indexer.k_norm.weight", [16], make(s + "indexer.k_layernorm.weight", [16]))
        }
        let m = l(i) + "mlp."
        put(b + "ffn_gate_inp.weight", [2, hidden], make(m + "gate.weight", [2, hidden]))
        put(b + "ffn_gate_inp_shexp.weight", [hidden], make(m + "shared_expert_gate.weight", [hidden]))
        for p in ["gate", "up", "down"] {
            put(b + "ffn_\(p)_shexp.weight", [64, hidden], make(m + "shared_expert.\(p)_proj", [64, hidden]))
        }
        for p in ["gate", "up", "down"] {
            put(b + "ffn_\(p)_exps.weight", [2, 64, hidden], make(m + "switch_mlp.\(p)_proj", [2, 64, hidden]))
        }
        if i == 1 {
            let p = l(i) + "ple."
            put(b + "ple_conv1d.weight", [width, 4], make(p + "conv1d.weight", [width, 4]))
            put(b + "ple_key.weight", [width, hidden], make(p + "key_proj", [width, hidden]))
            put(b + "ple_value.weight", [hidden, hidden], make(p + "value_proj", [hidden, hidden]))
            for n in ["conv", "key", "query"] { put(b + "ple_norm_\(n).weight", [width], make(p + "norm_\(n).weight", [width])) }
        }
    }
    // token_embd in Q8_0, so a quantized GGUF tensor is read on the way.
    let embed = random(32 * hidden)
    var q8 = Data()
    for block in 0..<(embed.count / 32) {
        let values = embed[(block * 32)..<(block * 32 + 32)]
        let d = Float16(values.map(abs).max()! / 127)
        q8.append(contentsOf: withUnsafeBytes(of: d.bitPattern.littleEndian) { Array($0) })
        q8.append(contentsOf: values.map { UInt8(bitPattern: Int8(($0 / Float(d)).rounded())) })
    }
    tensors.append(("token_embd.weight", [32, hidden], 8, q8))
    var decoded = [Float](repeating: 0, count: embed.count)
    q8.withUnsafeBytes { raw in
        decoded.withUnsafeMutableBufferPointer {
            GGUFQuant.dequantize(raw.baseAddress!, type: 8, count: embed.count, into: $0.baseAddress!)
        }
    }
    original["language_model.model.embed_tokens"] = ([32, hidden], decoded)
    put("output.weight", [32, hidden], make("language_model.lm_head", [32, hidden]))
    put("output_hc_norm.weight", [width], make("language_model.model.hyper_connection_mixer.hc_norm.weight", [width]))
    put("output_hc_down.weight", [64, width], make("language_model.model.hyper_connection_mixer.input_mix_weight_down", [64, width]))
    put("output_hc_up.weight", [width, 64], make("language_model.model.hyper_connection_mixer.input_mix_weight_up", [width, 64]))
    let table = random(128 * 2 * 32)
    put("per_layer_token_embd.weight", [128 * 2, 32], table)
    put("blk.4.nextn_fc_hidden.weight", [4, 32], random(128))  // the MTP block: dropped

    let multipliers: [UInt64] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    func write(_ tensors: [(name: String, shape: [Int], type: Int, data: Data)], to url: URL, experts: UInt32 = 2) throws {
        var file = Data("GGUF".utf8)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { file.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { file.append(contentsOf: $0) } }
        func string(_ s: String) {
            u64(UInt64(s.utf8.count))
            file.append(contentsOf: s.utf8)
        }
        let scalars: [(String, UInt32)] = [
            ("block_count", 5), ("embedding_length", 64), ("expert_count", experts), ("ssm.group_count", 2), ("ssm.time_step_rank", 6),
            ("ssm.state_size", 32), ("ssm.inner_size", UInt32(inner)), ("full_attention_interval", 4), ("hyper_connection.low_rank", 64),
            ("hyper_connection.count", 4),
        ]
        u32(3)
        u64(UInt64(tensors.count))
        u64(UInt64(scalars.count + 6))
        string("general.architecture")
        u32(8)
        string("qwen4exp")
        string("tokenizer.chat_template")
        u32(8)
        string("{{ messages }}")
        for (key, value) in scalars {
            string("qwen4exp." + key)
            u32(4)
            u32(value)
        }
        string("qwen4exp.ple.layers")
        u32(9)
        u32(5)
        u64(1)
        u32(1)
        for (key, values) in [
            ("layer_multipliers", multipliers), ("head_offsets", (0..<16).map { UInt64($0 * 1000) }),
            ("head_vocab_sizes", (0..<16).map { UInt64(999 + $0) }),
        ] {
            string("qwen4exp.ple." + key)
            u32(9)
            u32(10)
            u64(UInt64(values.count))
            values.forEach(u64)
        }
        var at: UInt64 = 0
        for t in tensors {
            string(t.name)
            u32(UInt32(t.shape.count))
            t.shape.reversed().forEach { u64(UInt64($0)) }
            u32(UInt32(t.type))
            u64(at)
            at += UInt64((t.data.count + 31) / 32 * 32)
        }
        file.append(Data(count: (32 - file.count % 32) % 32))
        for t in tensors {
            file.append(t.data)
            file.append(Data(count: (32 - t.data.count % 32) % 32))
        }
        try file.write(to: url)
    }
    try write(tensors, to: source.appendingPathComponent("tiny-Q8_0.gguf"))
    var config =
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/official-bf16/config.json"))) as! [String: Any]
    var text = config["text_config"] as! [String: Any]
    text["num_hidden_layers"] = 4
    text["hidden_size"] = 64
    text["num_experts"] = 2
    config["text_config"] = text
    try JSONSerialization.data(withJSONObject: config).write(to: support.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: support.appendingPathComponent("tokenizer.json"))

    let converter = try CheckpointConverter(gguf: GGUFSource(source), support: support, checkContract: false)
    try require(converter.sourceKind == "GGUF (F32/Q8_0)", "source kind \(converter.sourceKind)")
    // The header-only classification (docs/models.md's matrix) names what the conversion quantizes, from names alone:
    // the tensors `init(gguf:)` quantizes in this fixture, and no other. Here only the Q8_0 embeddings are quantized a
    // second time; an all-float GGUF is quantized once, not lossy.
    let kept: Set<String> = [
        "token_embd.weight", "output.weight", "output_hc_down.weight", "output_hc_up.weight", "per_layer_token_embd.weight",
        "blk.1.ple_key.weight", "blk.1.ple_value.weight",
    ]
    let perLayer = [
        "hc_attn_up", "hc_ffn_up", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
    ]
    let quantizedNames =
        kept.union((0..<4).flatMap { i in perLayer.map { "blk.\(i).\($0).weight" } })
        .union(["blk.0.hc_attn_down_inject.weight", "blk.0.hc_ffn_down_inject.weight", "blk.0.attn_qkv_gate.weight"])
        .union((1..<4).flatMap { i in ["blk.\(i).hc_attn_down.weight", "blk.\(i).hc_ffn_down.weight"] })
        .union(["blk.1.attn_qkv.weight", "blk.1.attn_gate.weight", "blk.2.attn_qkv.weight", "blk.2.attn_gate.weight"])
        .union((0..<3).map { "blk.\($0).ssm_out.weight" })
        .union(["attn_q", "attn_k", "attn_v", "attn_output"].map { "blk.3.\($0).weight" })
    let predicted = Set(try GGUFSource(source).tensors.keys.filter { CheckpointConverter.ggufQuantizes($0, nativeExperts: false) })
    try require(
        predicted == quantizedNames,
        "quantized tensors predicted from names differ: \(predicted.symmetricDifference(quantizedNames).sorted())")
    try require(
        CheckpointConverter.ggufLoss(try GGUFSource(source), experts: 2)
            == "re-quantizes 1 tensors stored as Q8_0 ×1 (token embeddings) onto 4-bit affine group 64",
        "Q8_0 loss: \(CheckpointConverter.ggufLoss(try GGUFSource(source), experts: 2) ?? "none")")
    let floatsOnly = root.appendingPathComponent("floats", isDirectory: true)
    try FileManager.default.createDirectory(at: floatsOnly, withIntermediateDirectories: true)
    try write(
        tensors.map { $0.name == "token_embd.weight" ? ($0.name, $0.shape, 0, embed.withUnsafeBytes { Data($0) }) : $0 },
        to: floatsOnly.appendingPathComponent("tiny-F32.gguf"))
    try require(CheckpointConverter.ggufLoss(try GGUFSource(floatsOnly), experts: 2) == nil, "an all-float GGUF named lossy")
    try converter.convert(to: destination)
    try require(
        try String(contentsOf: destination.appendingPathComponent("chat_template.jinja"), encoding: .utf8) == "{{ messages }}"
            && FileManager.default.fileExists(atPath: destination.appendingPathComponent("tokenizer.json").path),
        "template or tokenizer not carried over")
    var spans: [String: TensorSpan] = [:]
    var files: [String: Data] = [:]
    for name in try FileManager.default.contentsOfDirectory(atPath: destination.path) where name.hasSuffix(".safetensors") {
        let url = destination.appendingPathComponent(name)
        files[name] = try Data(contentsOf: url)
        try ModelSource.header(of: url, file: name).forEach { spans[$0.key] = $0.value }
    }
    func bytes(_ name: String) throws -> Data {
        guard let span = spans[name] else { throw EngineError("Converter GGUF: no \(name)") }
        return files[span.file]!.subdata(in: Int(span.offset)..<Int(span.offset + span.length))
    }
    func quantized(_ values: [Float], columns: Int, group: Int) -> (Data, Data, Data) {
        var c = [UInt32](repeating: 0, count: values.count / 8)
        var s = [UInt16](repeating: 0, count: values.count / group)
        var bias = s
        values.withUnsafeBytes { v in
            c.withUnsafeMutableBufferPointer { cp in
                s.withUnsafeMutableBufferPointer { sp in
                    bias.withUnsafeMutableBufferPointer { bp in
                        CheckpointConverter.quantize(
                            v.baseAddress!, dtype: "F32", rows: values.count / columns, columns: columns, group: group,
                            codes: cp.baseAddress!, scales: sp.baseAddress!, biases: bp.baseAddress!)
                    }
                }
            }
        }
        return (c.withUnsafeBytes { Data($0) }, s.withUnsafeBytes { Data($0) }, bias.withUnsafeBytes { Data($0) })
    }
    for (name, (shape, values)) in original {
        if name.hasSuffix("A_log") {
            // exp then log in float, then bf16: within one bf16 step of the original.
            let got = try bytes(name).withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }.map(BF16.decode)
            try require(zip(got, values).allSatisfy { abs($0 - $1) <= abs($1) / 128 }, "\(name) \(got) for \(values)")
        } else if spans[name] != nil {
            try require(try bytes(name) == values.map(BF16.encode).withUnsafeBytes { Data($0) }, "\(name) differs")
        } else {
            let (codes, scales, biases) = quantized(values, columns: shape.last!, group: 64)
            try require(
                try bytes(name + ".weight") == codes && bytes(name + ".scales") == scales && bytes(name + ".biases") == biases,
                "\(name) differs from MLX's quantization of the original")
        }
    }
    for k in [0, 77, 127] {
        let (codes, scales, _) = quantized(Array(table[(k * 64)..<((k + 1) * 64)]), columns: 32, group: 32)
        try require(
            try bytes(l(1) + "ple.ple_embedding.ngram_embedding.shard_\(k).weight") == codes
                && bytes(l(1) + "ple.ple_embedding.ngram_embedding.shard_\(k).scales") == scales, "PLE shard \(k)")
    }
    let constants = try bytes(l(1) + "ple.ple_embedding.layer_multipliers").withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    try require(
        constants == multipliers.map { Int64($0) } && spans.keys.filter { $0.contains("shard_") && $0.hasSuffix(".weight") }.count == 128
            && !spans.keys.contains { $0.contains("nextn") || $0.hasPrefix("blk.") }, "PLE constants, shard count or a dropped tensor kept")
    // A 2-bit IQ tensor is read; an encoding Origami does not read (ternary TQ1_0) is refused by name, before anything is written.
    for type in [16, 34] {
        var variant = tensors
        variant[0].type = type
        let folder = root.appendingPathComponent("type-\(type)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try write(variant, to: folder.appendingPathComponent("tiny-\(GGUFQuant.name(type)).gguf"))
        do {
            let accepted = try CheckpointConverter(gguf: GGUFSource(folder), support: support, checkContract: false)
            try require(type == 16 && accepted.sourceKind.contains("IQ2_XXS"), "\(GGUFQuant.name(type)) accepted as \(accepted.sourceKind)")
        } catch let error as EngineError
            where type == 34 && error.description.contains("TQ1_0") && error.description.contains("does not read")
        {}
    }
    // Expert banks as raw blocks of `type` (Q2_0 or Q4_0, d = 1/64): native below 4.5 bits, requantized at 4.5, native when pruned.
    for (type, experts) in [(42, 2), (2, 2), (2, 1)] {
        let (block, size) = GGUFQuant.sizes[type]!
        var variant = tensors.filter { !$0.name.contains("_exps.") }
        for i in 0..<4 {
            for p in ["gate", "up", "down"] {
                let blocks = (0..<(experts * 64 * 64 / block)).map { b -> [UInt8] in
                    [0x00, 0x24] + (0..<(size - 2)).map { UInt8(truncatingIfNeeded: (b * 37 + $0 * 11) & 0xFF) }
                }
                variant.append(("blk.\(i).ffn_\(p)_exps.weight", [experts, 64, hidden], type, Data(blocks.joined())))
            }
            if experts == 1, let router = variant.firstIndex(where: { $0.name == "blk.\(i).ffn_gate_inp.weight" }) {
                variant[router] = (variant[router].name, [1, hidden], 0, variant[router].data.prefix(hidden * 4))
            }
        }
        let folder = root.appendingPathComponent("experts-\(type)-\(experts)", isDirectory: true)
        let target = root.appendingPathComponent("converted-\(type)-\(experts)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try write(variant, to: folder.appendingPathComponent("tiny-experts.gguf"), experts: UInt32(experts))
        let gguf = try GGUFSource(folder)
        let native = try CheckpointConverter(gguf: gguf, support: support, checkContract: false)
        try require(
            CheckpointConverter.ggufLoss(gguf, experts: 2)
                == (native.expertManifest != nil
                    ? "re-quantizes 1 tensors stored as Q8_0 ×1 (token embeddings) onto 4-bit affine group 64; the routed experts stay as stored"
                    : "re-quantizes 13 tensors stored as Q4_0 ×12, Q8_0 ×1 (routed experts, token embeddings) onto 4-bit affine group 64"),
            "\(GGUFQuant.name(type)) banks of \(experts) experts: loss \(CheckpointConverter.ggufLoss(gguf, experts: 2) ?? "none")")
        try require(
            (native.expertManifest != nil) == (type == 42 || experts == 1)
                && (type == 2 && experts == 2
                    ? native.outputBytes == converter.outputBytes
                    : native.outputBytes < converter.outputBytes),
            "\(GGUFQuant.name(type)) banks of \(experts) experts: native \(native.expertManifest != nil)")
        guard native.expertManifest != nil else { continue }
        try native.convert(to: target)
        let manifest =
            try JSONSerialization.jsonObject(with: Data(contentsOf: target.appendingPathComponent("gguf-experts.json"))) as! [String: Any]
        let layers = manifest["layers"] as! [[String: [String: Any]]]
        let path = folder.appendingPathComponent("tiny-experts.gguf").standardizedFileURL.path
        try require(
            manifest["format"] as? String == "origami-gguf-experts-v1" && manifest["experts"] as? Int == experts && layers.count == 4
                && (manifest["files"] as? [[String: Any]])?.first?["path"] as? String == path,
            "manifest header \(manifest["experts"] ?? "?")")
        for (i, layer) in layers.enumerated() {
            for p in ["gate", "up", "down"] {
                let bank = gguf.tensors["blk.\(i).ffn_\(p)_exps.weight"]!
                let entry = layer[p]!
                try require(
                    entry["path"] as? String == path && entry["offset"] as? Int == Int(bank.offset) && entry["type"] as? Int == type
                        && entry["stride"] as? Int == 64 * 64 / block * size && entry["expert_bytes"] as? Int == 64 * 64 / block * size
                        && entry["rows"] as? Int == 64 && entry["columns"] as? Int == hidden, "layer \(i) \(p) bank entry \(entry)")
            }
        }
        var names: [String: TensorSpan] = [:]
        var router = Data()
        for name in try FileManager.default.contentsOfDirectory(atPath: target.path) where name.hasSuffix(".safetensors") {
            let url = target.appendingPathComponent(name)
            for (key, span) in try ModelSource.header(of: url, file: name) {
                names[key] = span
                if key == l(2) + "mlp.gate.weight" {
                    router = try Data(contentsOf: url).subdata(in: Int(span.offset)..<Int(span.offset + span.length))
                }
            }
        }
        let config =
            try JSONSerialization.jsonObject(with: Data(contentsOf: target.appendingPathComponent("config.json"))) as! [String: Any]
        try require(
            !names.keys.contains { $0.contains("switch_mlp") }
                && (config["quantization"] as? [String: Any])?["expert_format"] as? String == "gguf"
                && (config["text_config"] as? [String: Any])?["num_experts"] as? Int == experts,
            "converted folder keeps expert banks or lacks the GGUF marker")
        // The router: the model's rows, then a zero row for each expert it does not have.
        let gate = original[l(2) + "mlp.gate.weight"]!.1
        try require(
            names[l(2) + "mlp.gate.weight"]?.shape == [2, hidden]
                && router
                    == (Array(gate.prefix(experts * hidden)).map(BF16.encode) + Array(repeating: 0, count: (2 - experts) * hidden))
                    .withUnsafeBytes { Data($0) },
            "router rows of a GGUF with \(experts) experts")
    }
}

/// GGUF: every ggml type Origami reads decodes bit for bit as llama.cpp's gguf-py does (random raw blocks and
/// gguf-py's floats in the fixture), and the types it does not read are named, not guessed.
func ggufDequantize() throws {
    let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("gguf/dequantize.json"))) as! [String: Any]
    var checked: [String] = []
    for entry in fixture["types"] as! [[String: Any]] {
        let type = entry["type"] as! Int
        let name = entry["name"] as! String
        let hex = entry["hex"] as! String
        let expected = (entry["expected"] as! [NSNumber]).map(\.uint32Value)
        var raw = [UInt8]()
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            raw.append(UInt8(hex[index..<next], radix: 16)!)
            index = next
        }
        var values = [Float](repeating: .nan, count: expected.count)
        raw.withUnsafeBytes { source in
            values.withUnsafeMutableBufferPointer {
                GGUFQuant.dequantize(source.baseAddress!, type: type, count: expected.count, into: $0.baseAddress!)
            }
        }
        if let wrong = values.indices.first(where: { values[$0].bitPattern != expected[$0] }) {
            throw EngineError("GGUF \(name) value \(wrong): \(values[wrong]) where gguf-py gives \(Float(bitPattern: expected[wrong]))")
        }
        guard GGUFQuant.readable.contains(type), GGUFQuant.name(type) == name else {
            throw EngineError("GGUF \(name) is not listed as readable")
        }
        checked.append(name)
    }
    guard checked.count == GGUFQuant.readable.count, !GGUFQuant.readable.contains(34), GGUFQuant.name(21) == "IQ3_S" else {
        throw EngineError("GGUF types checked \(checked) against readable \(GGUFQuant.readable.sorted())")
    }
}

/// Misses rank before uses, ties fall back to uses and then the shipped order; a file without misses loads, a save
/// and load keeps both arrays, and the window counts an adopted prediction once and a resident hit never.
func routingProfileMissRanking() throws {
    let profile = RoutingProfile()
    for _ in 0..<5 { profile.record(layer: 2, experts: [1]) }
    for _ in 0..<2 { profile.record(layer: 2, experts: [8, 9]) }
    profile.record(layer: 3, experts: [4])
    profile.recordMiss(layer: 3, expert: 4, weight: 4)
    profile.recordMiss(layer: 2, expert: 9, weight: 1)
    profile.recordMiss(layer: 2, expert: 8, weight: 1)
    let shipped = StoreBuilder.ranked(nil).first { ![3 * 512 + 4, 2 * 512 + 8, 2 * 512 + 9, 2 * 512 + 1].contains($0) }!
    profile.recordMiss(layer: shipped / 512, expert: shipped % 512, weight: 1)  // missed, never counted as used
    let expected = [3 * 512 + 4, 2 * 512 + 8, 2 * 512 + 9, shipped, 2 * 512 + 1]
    let ranking = profile.ranking(minimumTokens: 0) ?? []
    guard Array(ranking.prefix(5)) == expected else {
        throw EngineError("Miss ranking order differs: \(ranking.prefix(5)) for \(expected)")
    }
    let url = CheckFixtures.scratch.appendingPathComponent("routing-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try profile.save(to: url)
    guard let loaded = RoutingProfile.load(url), loaded.missed == profile.missed, loaded.ranking(minimumTokens: 0) == ranking else {
        throw EngineError("Routing misses did not survive a save and load")
    }
    var object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    object["missed"] = nil
    try JSONSerialization.data(withJSONObject: object).write(to: url)
    guard let old = RoutingProfile.load(url), old.missed.allSatisfy({ $0 == 0 }), old.ranking(minimumTokens: 0)?.first == 2 * 512 + 1 else {
        throw EngineError("A routing file without misses did not load with none")
    }
    object["missed"] = Data(count: 8).base64EncodedString()
    try JSONSerialization.data(withJSONObject: object).write(to: url)
    guard RoutingProfile.load(url) == nil else { throw EngineError("A routing file with a short miss array loaded") }
    try DecodePreroutingSelfChecks.missAccounting()
}

/// Routing counts rank experts most needed first (ties in the shipped order), only once they cover enough tokens,
/// and survive a save and load.
func routingProfileRanking() throws {
    let profile = RoutingProfile()
    for _ in 0..<3 { profile.record(layer: 5, experts: [7, 7, 9]) }  // a window counts an expert once
    profile.record(layer: 0, experts: [1])
    let counted: [Int] = [5 * 512 + 7, 5 * 512 + 9, 1]
    let ranking: [Int] = profile.ranking(minimumTokens: 0) ?? []
    guard profile.ranking() == nil, Array(ranking.prefix(3)) == counted else {
        throw EngineError("Routing ranking order or threshold differs: \(ranking.prefix(3))")
    }
    let uncounted: [Int] = StoreBuilder.ranked(nil).filter { !counted.contains($0) }
    guard Array(ranking.dropFirst(3).prefix(5)) == Array(uncounted.prefix(5)) else {
        throw EngineError("Uncounted experts do not keep the shipped order")
    }
    let url = CheckFixtures.scratch.appendingPathComponent("routing-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try profile.save(to: url)
    guard let loaded = RoutingProfile.load(url), loaded.ranking(minimumTokens: 0) == profile.ranking(minimumTokens: 0) else {
        throw EngineError("Routing counts did not survive a save and load")
    }
}
