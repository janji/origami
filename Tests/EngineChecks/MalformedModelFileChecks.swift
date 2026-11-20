import Foundation
import Metal
import OrigamiEngine

// Model files are untrusted input: each malformed header must be refused with `model_incompatible`, never trap,
// over-read or allocate what the file cannot hold. The cases are written to a scratch folder.

/// The body must throw an `EngineError` with `code` whose message contains `fragment`.
private func refuses(_ what: String, _ fragment: String, code: String = "model_incompatible", _ body: () throws -> Void) throws {
    do { try body() } catch let error as EngineError where error.code == code && error.description.contains(fragment) { return } catch {
        throw EngineError("\(what): expected a \(code) error naming \"\(fragment)\", got \(error)")
    }
    throw EngineError("\(what) was accepted")
}
private func demand(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }

private struct Bytes {
    var data = Data()
    mutating func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func string(_ s: String) {
        u64(UInt64(s.utf8.count))
        data.append(contentsOf: s.utf8)
    }
    mutating func kv(_ key: String, u32 value: UInt32) {
        string(key)
        u32(4)
        u32(value)
    }
    mutating func kv(_ key: String, string value: String) {
        string(key)
        u32(8)
        string(value)
    }
    /// `shape` in PyTorch order; GGUF lists the fastest dimension first.
    mutating func tensor(_ name: String, _ shape: [UInt64], type: UInt32 = 0, offset: UInt64 = 0) {
        string(name)
        u32(UInt32(shape.count))
        shape.reversed().forEach { u64($0) }
        u32(type)
        u64(offset)
    }
}

/// A GGUF v3 file: the counts, what `write` adds (metadata, then tensor infos), padding to 32 bytes and `body`.
private func gguf(tensors: UInt64, kvs: UInt64, version: UInt32 = 3, body: Data = Data(), _ write: (inout Bytes) -> Void = { _ in }) -> Data
{
    var b = Bytes(data: Data("GGUF".utf8))
    b.u32(version)
    b.u64(tensors)
    b.u64(kvs)
    write(&b)
    b.data.append(Data(count: (32 - b.data.count % 32) % 32))
    b.data.append(body)
    return b.data
}

/// A safetensors file: the length prefix (the header's own unless given), the header and `body`.
private func safetensors(_ header: String, length: UInt64? = nil, body: Data = Data()) -> Data {
    var b = Bytes()
    b.u64(length ?? UInt64(header.utf8.count))
    b.data.append(contentsOf: header.utf8)
    b.data.append(body)
    return b.data
}

/// Safetensors headers S1–S18 (`ModelSource.header` and the converter's shard names).
func malformedSafetensorsHeaders() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("malformed-st-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    func parse(_ name: String, _ data: Data) throws -> [String: TensorSpan] {
        let url = root.appendingPathComponent(name + ".safetensors")
        try data.write(to: url)
        return try ModelSource.header(of: url, file: name)
    }
    func entry(_ value: String) -> String { "{\"t\":\(value)}" }
    let invalidHeader = "invalid safetensors header"
    let invalidEntry = "invalid tensor entry t"
    try refuses("S1", "is not a safetensors file") { _ = try parse("s1", Data([0, 0, 0, 0])) }
    try refuses("S2", invalidHeader) { _ = try parse("s2", safetensors("{}", length: 0)) }
    try refuses("S3", invalidHeader) { _ = try parse("s3", safetensors("{}", length: .max)) }
    // S4: a header longer than the 64 MiB cap in a (sparse) file large enough to hold it.
    let big = root.appendingPathComponent("s4.safetensors")
    try safetensors("{}", length: 65 << 20).write(to: big)
    let handle = try FileHandle(forWritingTo: big)
    try handle.truncate(atOffset: 70 << 20)
    try handle.close()
    try refuses("S4", invalidHeader) { _ = try ModelSource.header(of: big, file: "s4") }
    try refuses("S5", invalidHeader) { _ = try parse("s5", safetensors("[1,2]")) }
    try refuses("S6", invalidHeader) { _ = try parse("s6", safetensors("{\"a\":")) }
    let cases: [(String, String, Data)] = [
        ("S7", "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,9223372036854775807]}", Data(count: 4)),
        ("S8", "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[10,5]}", Data(count: 16)),
        ("S9", "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[-8,0]}", Data(count: 16)),
        ("S10", "{\"dtype\":\"BF16\",\"shape\":[2048],\"data_offsets\":[0,4096]}", Data(count: 60)),
        ("S11", "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0.5,4]}", Data(count: 16)),
        ("S12", "{\"dtype\":\"BF16\",\"shape\":[-1,4],\"data_offsets\":[0,8]}", Data(count: 16)),
        ("S13", "{\"dtype\":\"BF16\",\"shape\":[1099511627776,1099511627776],\"data_offsets\":[0,2]}", Data(count: 16)),
        ("S14", "{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,30]}", Data(count: 32)),
        ("S15", "{\"shape\":[2],\"data_offsets\":[0,4]}", Data(count: 16)),
        ("S15b", "{\"dtype\":3,\"shape\":[2],\"data_offsets\":[0,4]}", Data(count: 16)),
    ]
    for (label, value, body) in cases {
        try refuses(label, invalidEntry) { _ = try parse(label, safetensors(entry(value), body: body)) }
    }
    // S20 (found by the fuzzer): two tensors sharing bytes.
    try refuses("S20", "tensors a and b overlap") {
        _ = try parse(
            "s20",
            safetensors(
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[4,12]}}",
                body: Data(count: 12)))
    }
    // S16: metadata of any form is ignored; a well-formed tensor of a dtype without a fixed width is not size-checked.
    let s16 = try parse(
        "s16",
        safetensors(
            "{\"__metadata__\":\"x\",\"t\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]},"
                + "\"u\":{\"dtype\":\"F4\",\"shape\":[3],\"data_offsets\":[8,10]}}", body: Data(count: 10)))
    try demand(s16.count == 2 && s16["t"]?.length == 8 && s16["t"]?.shape == [2, 2], "S16: \(s16)")
    // S18: a shard that is a relative symlink (a Hugging Face cache snapshot) is measured through the link.
    try FileManager.default.createDirectory(at: root.appendingPathComponent("blobs"), withIntermediateDirectories: true)
    try safetensors("{\"t\":{\"dtype\":\"F32\",\"shape\":[64],\"data_offsets\":[0,256]}}", body: Data(count: 256))
        .write(to: root.appendingPathComponent("blobs/abc"))
    let link = root.appendingPathComponent("s18.safetensors")
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: "blobs/abc")
    try demand(try ModelSource.header(of: link, file: "s18")["t"]?.length == 256, "S18: a symlinked shard was not read through the link")

    // S19 (found by the fuzzer): a config number beyond a double's range (-1e400) is read as -infinity, which Foundation's JSON
    // writer raised an uncatchable exception on.
    let infinite = try JSONSerialization.jsonObject(with: Data(#"{"model_type":"qwen4_exp","rope_theta":-1e400}"#.utf8)) as! [String: Any]
    try refuses("S19", "number out of range") { _ = try ModelSource.configJSON(infinite) }
    // S17: an index naming a shard outside the folder is refused before anything is opened.
    let source = root.appendingPathComponent("source", isDirectory: true)
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try FileManager.default.copyItem(
        at: CheckFixtures.file("models/official-bf16/config.json"), to: source.appendingPathComponent("config.json"))
    try safetensors(
        "{\"model.language_model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,16]}}", body: Data(count: 16)
    )
    .write(to: root.appendingPathComponent("x.safetensors"))
    // A ModelOpt (NVFP4) config served as it is says how to convert it.
    var modelopt = try JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as! [String: Any]
    modelopt["quantization_config"] = ["quant_method": "modelopt"]
    try refuses("A ModelOpt config", "NVFP4 (ModelOpt) release; `origami model add") {
        try ModelContract.validate(config: JSONSerialization.data(withJSONObject: modelopt))
    }
    // A matrix with a zero dimension would size the converter's row blocks by zero.
    try safetensors(
        "{\"model.language_model.layers.0.linear_attn.in_proj_z.weight\":{\"dtype\":\"BF16\",\"shape\":[4,0],\"data_offsets\":[0,0]}}"
    )
    .write(to: source.appendingPathComponent("model.safetensors"))
    try refuses("A zero-width matrix", "does not split into rows of groups") {
        _ = try CheckpointConverter(source: source, checkContract: false)
    }
    for name in ["../x.safetensors", ".hidden.safetensors", ""] {
        try JSONSerialization.data(withJSONObject: ["weight_map": ["model.language_model.norm.weight": name]])
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
        try refuses("S17 \(name)", "names a shard outside the folder") { _ = try CheckpointConverter(source: source) }
    }
}

/// GGUF headers G1–G17 (`GGUFSource` and the GGUF converter's geometry).
func malformedGGUFHeaders() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("malformed-gguf-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    func read(_ name: String, _ data: Data) throws -> GGUFSource {
        let url = root.appendingPathComponent(name + ".gguf")
        try data.write(to: url)
        return try GGUFSource(url)
    }
    var magic = gguf(tensors: 0, kvs: 0)
    magic[3] = UInt8(ascii: "X")
    try refuses("G1", "is not a GGUF file") { _ = try read("g1", magic) }
    try refuses("G2", "GGUF version 1") { _ = try read("g2a", gguf(tensors: 0, kvs: 0, version: 1)) }
    try refuses("G2", "GGUF version 4") { _ = try read("g2b", gguf(tensors: 0, kvs: 0, version: 4)) }
    try refuses("G3", "ends early") { _ = try read("g3", Data(gguf(tensors: 1, kvs: 0).prefix(24))) }
    try refuses("G4", "claims 0 tensors and 9223372036854775808 metadata keys") {
        _ = try read("g4", Data(gguf(tensors: 0, kvs: 1 << 63).prefix(24)))
    }
    // G21 (found by the fuzzer): a corrupt count over a large file walked its weights as empty keys for seconds.
    try refuses("G21", "claims 2097152 tensors") { _ = try read("g21", gguf(tensors: 1 << 21, kvs: 0, body: Data(count: 4096))) }
    try refuses("G5", "string of") {
        _ = try read(
            "g5",
            gguf(tensors: 0, kvs: 1) {
                $0.u64(512 << 20)
                $0.data.append(Data(count: 60))
            })
    }
    try refuses("G6", "metadata type 13 is unknown") {
        _ = try read(
            "g6",
            gguf(tensors: 0, kvs: 1) {
                $0.string("k")
                $0.u32(13)
            })
    }
    try refuses("G7", "nested arrays") {
        _ = try read(
            "g7",
            gguf(tensors: 0, kvs: 1) { b in
                b.string("k")
                b.u32(9)
                for _ in 0..<100_000 {
                    b.u32(9)
                    b.u64(1)
                }
            })
    }
    try refuses("G8", "array larger than the file") {
        _ = try read(
            "g8",
            gguf(tensors: 0, kvs: 1) {
                $0.string("k")
                $0.u32(9)
                $0.u32(10)
                $0.u64(1 << 62)
            })
    }
    try refuses("G9", "has 4294967295 dimensions") {
        _ = try read(
            "g9",
            gguf(tensors: 1, kvs: 0) {
                $0.string("t")
                $0.u32(.max)
                $0.u64(1)
            })
    }
    try refuses("G10", "dimension too large") { _ = try read("g10", gguf(tensors: 1, kvs: 0) { $0.tensor("t", [1 << 63]) }) }
    try refuses("G11", "too many elements") { _ = try read("g11", gguf(tensors: 1, kvs: 0) { $0.tensor("t", [1 << 32, 1 << 32]) }) }
    for (label, write) in [
        ("u32 0", { (b: inout Bytes) in b.kv("general.alignment", u32: 0) }),
        (
            "i32 -1",
            { (b: inout Bytes) in
                b.string("general.alignment")
                b.u32(5)
                b.u32(UInt32(bitPattern: -1))
            }
        ),
        ("u32 3", { (b: inout Bytes) in b.kv("general.alignment", u32: 3) }),
    ] {
        try refuses("G12 \(label)", "is not a power of two") { _ = try read("g12", gguf(tensors: 0, kvs: 1, write)) }
    }
    try refuses("G13", "lies past the end of the file") {
        _ = try read("g13", gguf(tensors: 1, kvs: 0, body: Data(count: 64)) { $0.tensor("t", [4], offset: .max - 15) })
    }
    // G18–G20 (found by `engine-checks --fuzz gguf`): tensors that share bytes, a name listed twice (the second silently
    // replaced the first) and an offset off the alignment are refused, as ggml refuses them.
    try refuses("G18", "tensors a and b overlap") {
        _ = try read(
            "g18",
            gguf(tensors: 2, kvs: 0, body: Data(count: 128)) {
                $0.tensor("a", [16])
                $0.tensor("b", [16], offset: 32)
            })
    }
    try refuses("G19", "tensor a is listed twice") {
        _ = try read(
            "g19",
            gguf(tensors: 2, kvs: 0, body: Data(count: 128)) {
                $0.tensor("a", [16])
                $0.tensor("a", [16], offset: 64)
            })
    }
    try refuses("G20", "a starts at 4, not a multiple of 32") {
        _ = try read("g20", gguf(tensors: 1, kvs: 0, body: Data(count: 128)) { $0.tensor("a", [16], offset: 4) })
    }
    // G22 (found by the fuzzer): a 6 MB tensor name over a large file took seconds in every reader of names.
    try refuses("G22", "GGUF string of 257 bytes") {
        _ = try read("g22", gguf(tensors: 1, kvs: 0, body: Data(count: 64)) { $0.tensor(String(repeating: "a", count: 257), [16]) })
    }
    // G14: a K-quant row of 640 values is 2.5 blocks of 256.
    try refuses("G14", "rows of 640 values, not whole Q4_K blocks of 256") {
        _ = try read("g14", gguf(tensors: 1, kvs: 0, body: Data(count: 5 * 144)) { $0.tensor("t", [2, 640], type: 12) })
    }
    // G15: an unknown type is kept as it is and named by the conversion check.
    let g15 = try read(
        "g15",
        gguf(tensors: 1, kvs: 1, body: Data(count: 32)) {
            $0.kv("general.architecture", string: "qwen4exp")
            $0.tensor("t", [4], type: 99)
        })
    try demand(
        CheckpointConverter.ggufProblem(g15)?.contains("type 99") == true, "G15: \(CheckpointConverter.ggufProblem(g15) ?? "no problem")")
    // G16: a split set with a part missing.
    let split = root.appendingPathComponent("split", isDirectory: true)
    try FileManager.default.createDirectory(at: split, withIntermediateDirectories: true)
    for part in 1...2 {
        try gguf(tensors: 0, kvs: 1) { $0.kv("split.count", u32: 3) }.write(to: split.appendingPathComponent("m-0000\(part)-of-00003.gguf"))
    }
    try refuses("G16", "has 2 of its 3 parts") { _ = try GGUFSource(split.appendingPathComponent("m-00001-of-00003.gguf")) }

    // G17: geometry from metadata that the tensors do not have. One key head of one state, one value head of one,
    // hyper-connection rank 8 with 4 streams, while hc_attn_down_inject has only 4 rows.
    let support = root.appendingPathComponent("support", isDirectory: true)
    try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: ["text_config": ["num_hidden_layers": 1, "hidden_size": 4, "num_experts": 2]])
        .write(to: support.appendingPathComponent("config.json"))
    func model(_ name: String, timeStepRank: UInt32) throws -> GGUFSource {
        let scalars: [(String, UInt32)] = [
            ("ssm.group_count", 1), ("ssm.time_step_rank", timeStepRank), ("ssm.state_size", 1), ("ssm.inner_size", 1),
            ("full_attention_interval", 4), ("hyper_connection.low_rank", 8), ("hyper_connection.count", 4), ("block_count", 1),
            ("embedding_length", 4), ("expert_count", 2),
        ]
        let tensors: [(String, [UInt64])] = [
            ("blk.0.ssm_conv1d.weight", [3, 4]), ("blk.0.hc_attn_norm.weight", [16]), ("blk.0.hc_attn_down_inject.weight", [4, 16]),
        ]
        var body = Data()
        var offsets: [UInt64] = []
        for (_, shape) in tensors {
            offsets.append(UInt64(body.count))
            body.append(Data(count: Int(shape.reduce(4, *) + 31) / 32 * 32))
        }
        return try read(
            name,
            gguf(tensors: UInt64(tensors.count), kvs: UInt64(scalars.count + 1), body: body) { b in
                b.kv("general.architecture", string: "qwen4exp")
                for (key, value) in scalars { b.kv("qwen4exp." + key, u32: value) }
                for (index, (name, shape)) in tensors.enumerated() { b.tensor(name, shape, offset: offsets[index]) }
            })
    }
    try refuses("G17 rank 0", "usable qwen4exp.ssm.time_step_rank") {
        _ = try CheckpointConverter(gguf: model("g17a", timeStepRank: 0), support: support, checkContract: false)
    }
    try refuses("G17 rows", "has no rows 0..<8") {
        _ = try CheckpointConverter(gguf: model("g17b", timeStepRank: 1), support: support, checkContract: false)
    }
}

/// The MTP head file (opt-in speculation) is untrusted too: an expert bank that is not 512 records of the 4-bit
/// layout is refused before anything is reserved or copied, a matrix of the wrong rank is refused, and a failed load
/// returns its ledger charge.
func malformedMTPHead() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = CheckFixtures.scratch.appendingPathComponent("malformed-mtp-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 1)
    /// The nine expert banks at their real sizes (a sparse file: nothing reads them before the refusals) and a 1-D
    /// `fc_embedding`; `scalesShape` replaces the shape of the gate projection's scales.
    func head(_ name: String, scalesShape: [Int]) throws -> URL {
        var header: [String: Any] = [:]
        var offset = 0
        func put(_ tensor: String, _ dtype: String, _ shape: [Int], _ bytes: Int) {
            header["language_model.mtp." + tensor] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        put("fc_embedding.weight", "BF16", [2560], 5120)
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let bank = "layers.0.mlp.switch_mlp.\(projection)."
            put(bank + "weight", "U32", [512, 640, 320], 512 * 819_200)
            put(bank + "scales", "BF16", projection == "gate_proj" ? scalesShape : [512, 640, 40], 512 * 51_200)
            put(bank + "biases", "BF16", [512, 640, 40], 512 * 51_200)
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let url = root.appendingPathComponent(name)
        var length = UInt64(json.count).littleEndian
        try (Data(bytes: &length, count: 8) + json).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(8 + json.count + offset))
        try handle.close()
        return url
    }
    try refuses("MTP bank of one row", "has no MTP head in the 4-bit group-64 MLX layout") {
        _ = try MTPHead(file: head("bank.safetensors", scalesShape: [1, 13_107_200]), device: device, ledger: ledger)
    }
    try demand(ledger.snapshot().reserved == 0, "MTP: a refused bank reserved memory")
    try refuses("MTP 1-D matrix", "lacks the matrix fc_embedding") {
        _ = try MTPHead(file: head("vector.safetensors", scalesShape: [512, 640, 40]), device: device, ledger: ledger)
    }
    try demand(ledger.snapshot().reserved == 0, "MTP: a failed load kept its ledger charge")
}

/// Conversion from a Hugging Face cache snapshot of an F32/F16 checkpoint: shards and small files behind relative
/// symlinks convert and are copied as files; F16/F32 floats are written as bf16 (1 + w for the zero-centred norms,
/// not for vision); I32 tensors keep 4 bytes per value; other weight formats and partial downloads stay behind; the
/// manifest names the folder, not its path; the identity follows the shards' modification times; a folder that is not
/// an Origami conversion is never replaced, an earlier conversion is.
func converterSourcesAndDestination() throws {
    let fm = FileManager.default
    let root = CheckFixtures.scratch.appendingPathComponent("converter-sources-\(UUID().uuidString)", isDirectory: true)
    defer { try? fm.removeItem(at: root) }
    let blobs = root.appendingPathComponent("blobs", isDirectory: true)
    let snapshot = root.appendingPathComponent("snapshot", isDirectory: true)
    for folder in [blobs, snapshot] { try fm.createDirectory(at: folder, withIntermediateDirectories: true) }
    let floats = (0..<256).map { Float($0 % 37) / 16 - 1 }
    let half = (0..<32).map { Float16(Float($0) / 8 - 2) }
    var header: [String: Any] = [:]
    var body = Data()
    func put(_ name: String, _ dtype: String, _ shape: [Int], _ data: Data) {
        header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [body.count, body.count + data.count]]
        body.append(data)
    }
    let l = "model.language_model.layers."
    put(l + "3.self_attn.q_norm.weight", "F32", [8], Array(floats[0..<8]).withUnsafeBytes { Data($0) })
    put(l + "0.linear_attn.conv1d.weight", "F16", [8, 1, 4], half.withUnsafeBytes { Data($0) })
    put(l + "0.mlp.gate.weight", "F32", [4, 64], floats.withUnsafeBytes { Data($0) })
    put(l + "0.linear_attn.in_proj_z.weight", "F32", [4, 64], floats.withUnsafeBytes { Data($0) })
    put(l + "1.ple.ple_embedding.layer_multipliers", "I32", [3], [Int32(7), -9, 123_456_789].withUnsafeBytes { Data($0) })
    put("model.visual.blocks.0.norm1.weight", "F32", [8], Array(floats[8..<16]).withUnsafeBytes { Data($0) })
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(json.count).littleEndian
    try (Data(bytes: &length, count: 8) + json + body).write(to: blobs.appendingPathComponent("weights"))
    try fm.copyItem(at: CheckFixtures.file("models/official-bf16/config.json"), to: blobs.appendingPathComponent("config"))
    try Data("{\"t\":1}".utf8).write(to: blobs.appendingPathComponent("tokenizer"))
    for (name, blob) in [("model.safetensors", "weights"), ("config.json", "config"), ("tokenizer.json", "tokenizer")] {
        try fm.createSymbolicLink(atPath: snapshot.appendingPathComponent(name).path, withDestinationPath: "../blobs/" + blob)
    }
    for name in ["README.md", "pytorch_model.bin", "model-00002.safetensors.part", "optimizer.pt"] {
        try Data("x".utf8).write(to: snapshot.appendingPathComponent(name))
    }
    let converter = try CheckpointConverter(source: snapshot, checkContract: false)
    let destination = root.appendingPathComponent("converted", isDirectory: true)
    // A folder that is not an Origami conversion is refused, untouched.
    try fm.createDirectory(at: destination, withIntermediateDirectories: true)
    try Data("mine".utf8).write(to: destination.appendingPathComponent("notes.txt"))
    try refuses("A user's folder as destination", "is not an Origami conversion", code: "destination_exists") {
        try converter.convert(to: destination)
    }
    try demand(fm.fileExists(atPath: destination.appendingPathComponent("notes.txt").path), "The converter touched a folder it refused")
    try fm.removeItem(at: destination)
    try converter.convert(to: destination)
    try converter.convert(to: destination)  // an earlier conversion is replaced

    let copied = Set(try fm.contentsOfDirectory(atPath: destination.path))
    try demand(
        copied == [
            "config.json", "model.safetensors", "model.safetensors.index.json", "origami-model.json", "README.md", "tokenizer.json",
        ],
        "Converted folder holds \(copied.sorted())")
    try demand(
        (try fm.attributesOfItem(atPath: destination.appendingPathComponent("tokenizer.json").path))[.type] as? FileAttributeType
            == .typeRegular
            && (try Data(contentsOf: destination.appendingPathComponent("tokenizer.json"))) == Data("{\"t\":1}".utf8),
        "tokenizer.json was copied as a link")
    let manifest =
        try JSONSerialization.jsonObject(with: Data(contentsOf: destination.appendingPathComponent("origami-model.json"))) as? [String: Any]
    try demand(
        manifest?["converted_from"] as? String == "snapshot", "origami-model.json records \(manifest?["converted_from"] ?? "nothing")")

    let url = destination.appendingPathComponent("model.safetensors")
    let file = try Data(contentsOf: url)
    let spans = try ModelSource.header(of: url, file: "model.safetensors")
    func bytes(_ name: String) throws -> Data {
        guard let span = spans[name] else { throw EngineError("Converter: no \(name) among \(spans.keys.sorted())") }
        return file.subdata(in: Int(span.offset)..<Int(span.offset + span.length))
    }
    func bf16(_ values: [Float]) -> Data { values.map(BF16.encode).withUnsafeBytes { Data($0) } }
    let o = "language_model.model.layers."
    try demand(try bytes(o + "3.self_attn.q_norm.weight") == bf16(floats[0..<8].map { $0 + 1 }), "F32 q_norm is not bf16(1 + w)")
    try demand(
        spans[o + "0.linear_attn.conv1d.weight"]?.dtype == "BF16" && spans[o + "0.linear_attn.conv1d.weight"]?.shape == [8, 4, 1]
            && (try bytes(o + "0.linear_attn.conv1d.weight")) == bf16(half.map { Float($0) }), "F16 conv kernel")
    try demand(try bytes(o + "0.mlp.gate.weight") == bf16(floats), "F32 router is not bf16")
    try demand(try bytes("vision_tower.blocks.0.norm1.weight") == bf16(Array(floats[8..<16])), "F32 vision norm changed beyond bf16")
    try demand(
        try bytes(o + "1.ple.ple_embedding.layer_multipliers") == [Int32(7), -9, 123_456_789].withUnsafeBytes { Data($0) },
        "I32 tensor bytes")
    var codes = [UInt32](repeating: 0, count: 32)
    var scales = [UInt16](repeating: 0, count: 4)
    var biases = scales
    floats.withUnsafeBytes { raw in
        codes.withUnsafeMutableBufferPointer { c in
            scales.withUnsafeMutableBufferPointer { s in
                biases.withUnsafeMutableBufferPointer { b in
                    CheckpointConverter.quantize(
                        raw.baseAddress!, dtype: "F32", rows: 4, columns: 64, group: 64, codes: c.baseAddress!, scales: s.baseAddress!,
                        biases: b.baseAddress!)
                }
            }
        }
    }
    try demand(
        try bytes(o + "0.linear_attn.in_proj_z.weight") == codes.withUnsafeBytes { Data($0) }
            && bytes(o + "0.linear_attn.in_proj_z.scales") == scales.withUnsafeBytes { Data($0) }, "F32 matrix quantization")

    // The same names and sizes saved again: another identity.
    try fm.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -3600)], ofItemAtPath: blobs.appendingPathComponent("weights").path)
    try demand(
        try CheckpointConverter(source: snapshot, checkContract: false).identity != converter.identity,
        "A shard saved again kept the conversion identity")
}

/// A folder whose chat template lives only in tokenizer_config.json loads with that template, under the same identity
/// as a folder holding it in chat_template.jinja; a folder with neither is refused.
func tokenizerTemplateFallback() throws {
    let fm = FileManager.default
    let root = CheckFixtures.scratch.appendingPathComponent("tokenizer-template-\(UUID().uuidString)", isDirectory: true)
    defer { try? fm.removeItem(at: root) }
    let template = "{{ messages[0]['content'] }}"
    let tokenizer =
        [
            "model": ["type": "BPE", "vocab": ["s": 0, "y": 1, "t": 2, "e": 3, "m": 4, "sy": 5], "merges": [["s", "y"]]],
            "added_tokens": [],
        ] as [String: Any]
    func folder(_ name: String, config: [String: Any], jinja: Bool) throws -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys]).write(
            to: url.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(
            to: url.appendingPathComponent("tokenizer_config.json"))
        if jinja { try Data(template.utf8).write(to: url.appendingPathComponent("chat_template.jinja")) }
        return url
    }
    final class Outcome: @unchecked Sendable { var value: Result<TokenizerStack, Error>? }
    func load(_ url: URL) throws -> TokenizerStack {
        let outcome = Outcome()
        let done = DispatchSemaphore(value: 0)
        Task {
            do { outcome.value = .success(try await TokenizerStack(modelDirectory: url)) } catch { outcome.value = .failure(error) }
            done.signal()
        }
        done.wait()
        return try outcome.value!.get()
    }
    let config: [String: Any] = ["tokenizer_class": "Qwen2Tokenizer", "chat_template": template]
    let embedded = try load(folder("embedded", config: config, jinja: false))
    let file = try load(folder("file", config: config, jinja: true))
    let messages: [[String: TemplateValue]] = [["role": .string("user"), "content": .string("system")]]
    try demand(embedded.identity == file.identity, "The embedded template changed the tokenizer identity")
    try demand(
        try embedded.encodeChat(messages: messages) == file.encodeChat(messages: messages)
            && embedded.encodeChat(messages: messages) == embedded.encode("system"), "The embedded template renders differently")
    try refuses("No template anywhere", "lacks chat_template.jinja", code: "model_incomplete") {
        _ = try load(folder("none", config: ["tokenizer_class": "Qwen2Tokenizer"], jinja: false))
    }
}
