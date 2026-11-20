import CExactIO
import Foundation
import Metal
import OrigamiEngine

private func group32Require(_ value: @autoclosure () throws -> Bool, _ message: @autoclosure () -> String) throws {
    guard try value() else { throw EngineError(message()) }
}

/// Runs a check with this process in the group-32 layout (kernels compiled with QGROUP 32, group-32
/// records), then restores group 64.
func withGroup32(_ body: () throws -> Void) throws {
    ModelLayout.use(.affine4Group32)
    defer { ModelLayout.use(.affine4Group64) }
    try body()
}

/// mlx-community/Qwen3.8-Flash-Next-4bit's config is accepted as 4-bit group 32 with an 8-bit router, other
/// layouts are still refused, and the group-32 record layout matches that build's tensor shapes.
func group32SchemeAndRecordLayout() throws {
    let folder = CheckFixtures.file("models/mlx-community-4bit")
    let config = try Data(contentsOf: folder.appendingPathComponent("config.json"))
    try group32Require(try ModelContract.validate(config: config) == .affine4Group32, "The group-32 build was not recognised")
    // Its PLE shards are named `shards.<n>`; Origami reads them as `shard_<n>` (the tensors and their quantization).
    try group32Require(
        ModelSource.canonical("model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight")
            == "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight"
            && ModelSource.canonical("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.scales")
                == "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.scales"
            && ModelContract.formats(config: config).modules["language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127"]
                == AffineFormat(bits: 4, group: 32),
        "The group-32 build's PLE shard names were not read")
    let (global, modules) = ModelContract.formats(config: config)
    try group32Require(
        global == AffineFormat(bits: 4, group: 32)
            && modules["language_model.model.layers.0.mlp.gate"] == AffineFormat(bits: 8, group: 64)
            && modules["language_model.model.layers.47.mlp.shared_expert_gate"] == AffineFormat(bits: 8, group: 64),
        "Router and shared-expert gate formats were not read")
    var root = try JSONSerialization.jsonObject(with: config) as! [String: Any]
    var quantization = root["quantization"] as! [String: Any]
    for (bits, group) in [(8, 64), (4, 128), (3, 32)] {
        quantization["bits"] = bits
        quantization["group_size"] = group
        root["quantization"] = quantization
        var refused = false
        do { _ = try ModelContract.validate(config: JSONSerialization.data(withJSONObject: root)) } catch { refused = true }
        try group32Require(refused, "\(bits)-bit group \(group) was accepted without kernels")
    }

    // The build's layer-0 expert banks: 512 records of the group-32 record size.
    let tensors =
        (try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("layer0-mlp.json"))) as! [String: Any])[
            "tensors"] as! [String: [String: Any]]
    let record =
        ["gate", "up", "down"].flatMap { projection in
            ["weight", "scales", "biases"].map { part in
                tensors["language_model.model.layers.0.mlp.switch_mlp.\(projection)_proj.\(part)"]!["length"] as! Int
            }
        }.reduce(0, +) / 512
    try group32Require(
        record == 3_072_000 && QuantizationScheme.affine4Group32.expertRecordBytes == record,
        "Group-32 expert record is \(record) bytes")
    try withGroup32 {
        try group32Require(PackStore.expertBytes == 3_072_000 && ModelLayout.group == 32, "Layout did not switch")
        let offsets = ExpertProjection.allCases.map { [$0.weightOffset, $0.scaleOffset, $0.biasOffset] }
        try group32Require(
            offsets == [[0, 819_200, 921_600], [1_024_000, 1_843_200, 1_945_600], [2_048_000, 2_867_200, 2_969_600]],
            "Group-32 projection offsets: \(offsets)")
    }
    try group32Require(PackStore.expertBytes == 2_764_800 && ExpertProjection.down.biasOffset == 2_713_600, "Group-64 layout not restored")
}

/// Small matrices the engine keeps as floats are dequantized when they come quantized: an 8-bit group-64
/// router and a 4-bit group-32 block inject, each value scale * code + bias. The host-joined projection (scopes too
/// short for the whole schedule, replays) reads them the same way: the router's FP32 logits and the inject's BF16
/// output equal a float matmul over those values, with and without the resident copy, on the host and pinned paths.
func group32QuantizedFloatMatrices() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-float-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    var data = Data()
    var spans: [String: Any] = [:]
    var expected: [String: [Float]] = [:]
    var formats: [String: AffineFormat] = [:]
    func quantized(_ module: String, rows: Int, width: Int, bits: Int, group: Int) {
        var state = UInt64(rows * 31 + bits)
        func next() -> UInt32 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt32(truncatingIfNeeded: state >> 33)
        }
        let words = (0..<(rows * width * bits / 32)).map { _ in next() }
        let scales = (0..<(rows * width / group)).map { _ in BF16.encode(Float(next() % 19 + 1) / 512) }
        let biases = (0..<(rows * width / group)).map { _ in BF16.encode(Float(Int(next() % 23) - 11) / 256) }
        for (part, bytes, dtype, shape) in [
            ("weight", words.withUnsafeBytes { Data($0) }, "U32", [rows, width * bits / 32]),
            ("scales", scales.withUnsafeBytes { Data($0) }, "BF16", [rows, width / group]),
            ("biases", biases.withUnsafeBytes { Data($0) }, "BF16", [rows, width / group]),
        ] {
            spans["language_model.\(module).\(part)"] = [
                "file": "tensor.bin", "dtype": dtype, "offset": data.count, "length": bytes.count, "shape": shape,
            ]
            data.append(bytes)
        }
        expected[module] = (0..<(rows * width)).map { index in
            let row = index / width
            let column = index % width
            let bit = column * bits
            let code = (words[row * width * bits / 32 + bit / 32] >> UInt32(bit % 32)) & UInt32((1 << bits) - 1)
            let g = row * (width / group) + column / group
            return BF16.decode(scales[g]) * Float(code) + BF16.decode(biases[g])
        }
        formats["language_model." + module] = AffineFormat(bits: bits, group: group)
    }
    quantized("model.layers.0.mlp.gate", rows: 512, width: 2560, bits: 8, group: 64)
    quantized("model.layers.0.attn_hyper_connection.block_inject_weight", rows: 4, width: 10240, bits: 4, group: 32)
    try data.write(to: directory.appendingPathComponent("tensor.bin"))
    let store = try PackStore(model: ModelSource(fixture: directory, spans: spans, formats: formats))
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentFloatTensorCache(store: store, device: device, ledger: ledger, byteLimit: 16_000_000)
        for (module, values) in expected {
            guard let lease = try cache.acquire(module + ".weight") else { throw EngineError("Quantized float matrix \(module) bypassed") }
            let actual = Array(
                UnsafeBufferPointer(start: lease.buffer.contents().assumingMemoryBound(to: Float.self), count: lease.elementCount))
            try group32Require(actual.map(\.bitPattern) == values.map(\.bitPattern), "Dequantized \(module) differs")
        }
        let kernels = try ReferenceKernels()
        let host = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger)
        let hostResident = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger, denseResidentBytes: 16_000_000)
        let pinned = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger, pinnedDense: true)
        let pinnedResident = try ReferenceWeights(
            store: store, kernels: kernels, ledger: ledger, denseResidentBytes: 16_000_000, pinnedDense: true)
        for (module, values) in expected {
            let router = module.hasSuffix(".mlp.gate")
            let rows = router ? 512 : 4
            let width = values.count / rows
            let x = (0..<(2 * width)).map { Float(($0 % 13) - 6) / 16 }
            var reference = [Float](repeating: 0, count: 2 * rows)
            origami_sgemm_nt(2, Int32(rows), Int32(width), x, values, &reference)
            if !router { reference = reference.map { BF16.decode(BF16.encode($0)) } }
            for weights in [host, hostResident] {
                let actual = try weights.linear(x, name: module, outputBF16: !router)
                try group32Require(
                    actual.map(\.bitPattern) == reference.map(\.bitPattern), "Host projection of quantized \(module) differs")
            }
            let gpu = try pinned.linear(x, name: module, outputBF16: !router)
            let gpuResident = try pinnedResident.linear(x, name: module, outputBF16: !router)
            try group32Require(
                gpu.map(\.bitPattern) == gpuResident.map(\.bitPattern)
                    && zip(gpu, reference).allSatisfy { abs($0 - $1) <= 1e-2 * max(1, abs($1)) },
                "Pinned projection of quantized \(module) differs")
        }
    }
    try group32Require(ledger.snapshot().reserved == 0, "Quantized float matrices retained resources")
}

/// Zero-centred norms stored as w (mlx-community's build) are read as 1 + w, rounded as the converter rounds it, and
/// norms already stored as 1 + w (carloslfu's build) are read as stored: the detector tells a set of q/k norms near 0
/// from the same set plus 1, only `CheckpointConverter.plusOneNorms` change, and with `ORIGAMI_MODEL` (the regular
/// group-64 build) the real checkpoint is detected as 1 + w.
func zeroCentredNormsReadAsOnePlusW() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-norms-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    // Trained zero-centred norms: mean about -0.07, spread about 0.3 (as in mlx-community's q/k norms).
    let stored = (0..<256).map { BF16.encode(Float(($0 * 37) % 61 - 30) / 100 - 0.07) }
    let shifted = stored.map { BF16.encode(BF16.decode($0) + 1) }
    let other = (0..<128).map { BF16.encode(Float($0 % 7) / 8 - 0.4) }
    func source(_ norm: [UInt16], file: String) throws -> ModelSource {
        var data = Data()
        var spans: [String: Any] = [:]
        func add(_ name: String, _ values: [UInt16]) {
            spans["language_model.model." + name] = [
                "file": file, "dtype": "BF16", "offset": data.count, "length": values.count * 2, "shape": [values.count],
            ]
            values.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        for layer in [3, 7] {
            add("layers.\(layer).self_attn.q_norm.weight", norm)
            add("layers.\(layer).self_attn.k_norm.weight", norm)
        }
        add("layers.0.attn_hyper_connection.hc_norm.weight", norm)
        add("layers.0.linear_attn.norm.weight", other)
        try data.write(to: directory.appendingPathComponent(file))
        let tensors = try JSONDecoder().decode([String: TensorSpan].self, from: JSONSerialization.data(withJSONObject: spans))
        let detected = try ModelSource.storesZeroCentredNorms(tensors, directory: directory)
        return try ModelSource(fixture: directory, spans: spans, zeroCentredNorms: detected)
    }
    func read(_ model: ModelSource, _ name: String, count: Int) throws -> [UInt16] {
        var values = [UInt16](repeating: 0, count: count)
        try values.withUnsafeMutableBytes { try PackStore(model: model).readTensor("model." + name, into: $0) }
        return values
    }
    let raw = try source(stored, file: "w.bin")
    let plusOne = try source(shifted, file: "one-plus-w.bin")
    try group32Require(raw.zeroCentredNorms && !plusOne.zeroCentredNorms, "Zero-centred norms were not told from 1 + w")
    for model in [raw, plusOne] {
        for name in [
            "layers.3.self_attn.q_norm.weight", "layers.7.self_attn.k_norm.weight", "layers.0.attn_hyper_connection.hc_norm.weight",
        ] {
            try group32Require(try read(model, name, count: 256) == shifted, "\(name) was not read as 1 + w")
        }
        try group32Require(try read(model, "layers.0.linear_attn.norm.weight", count: 128) == other, "A norm outside plusOneNorms changed")
    }
    // A checkpoint without q/k norms keeps what it stores.
    try group32Require(!(try ModelSource.storesZeroCentredNorms([:], directory: directory)), "No norms read as zero-centred")
    if let folder = ProcessInfo.processInfo.environment["ORIGAMI_MODEL"] {
        let model = try ModelSource(directory: URL(fileURLWithPath: folder))
        try group32Require(!model.zeroCentredNorms, "\(folder) stores 1 + w but was detected as zero-centred")
    }
}
