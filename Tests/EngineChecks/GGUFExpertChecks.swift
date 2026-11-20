import Accelerate
import Foundation
import Metal
import OrigamiEngine

private func ggufRequire(_ value: @autoclosure () throws -> Bool, _ message: @autoclosure () -> String) throws {
    guard try value() else { throw EngineError(message()) }
}

/// The GGUF layout the expert checks run: ten experts per layer (a pruned model); layer 0 mixes three encodings (IQ2_XS
/// gate, IQ3_S up, IQ4_NL down), layer 1 is Q2_0 throughout, layer 2 IQ1_M, IQ2_S and Q4_0, the others Q2_0.
private let ggufCheckTypes: [[Int]] = [[17, 21, 20], [42, 42, 42], [29, 22, 2]] + Array(repeating: [42, 42, 42], count: 45)

/// Seeded records of `layer`'s ten experts (header, then random valid blocks at the layout's offsets) and their weights
/// as GGUFQuant dequantizes them, [expert][projection][output * width + input].
private func ggufRecords(_ layout: GGUFExperts, layer: Int) -> (records: [[UInt8]], weights: [[[Double]]]) {
    var random = MatrixRandom(state: 0x6767_0000 + UInt64(layer))
    var records: [[UInt8]] = []
    var weights: [[[Double]]] = []
    for expert in 0..<10 {
        var record = [UInt8](repeating: 0xEE, count: layout.recordBytes)
        var matrices: [[Double]] = []
        record.withUnsafeMutableBytes { layout.prepare($0, layer: layer, expert: expert) }
        for part in layout.layers[layer].parts {
            let raw = randomGGMLBlocks(part.type, values: part.outputs * part.width, &random)
            record.replaceSubrange(part.record..<(part.record + part.bytes), with: raw)
            var values = [Float](repeating: 0, count: part.outputs * part.width)
            raw.withUnsafeBytes { r in
                values.withUnsafeMutableBufferPointer {
                    GGUFQuant.dequantize(r.baseAddress!, type: part.type, count: $0.count, into: $0.baseAddress!)
                }
            }
            matrices.append(values.map(Double.init))
        }
        records.append(record)
        weights.append(matrices)
    }
    return (records, weights)
}

/// One expert in float64: down(SiLU(gate x) × up x).
private func ggufExpert(_ w: [[Double]], _ x: ArraySlice<Float>) -> [Double] {
    func matvec(_ m: [Double], _ v: [Double], outputs: Int, width: Int) -> [Double] {
        m.withUnsafeBufferPointer { mp in
            v.withUnsafeBufferPointer { vp in
                (0..<outputs).map { o in
                    var s = 0.0
                    vDSP_dotprD(mp.baseAddress! + o * width, 1, vp.baseAddress!, 1, &s, vDSP_Length(width))
                    return s
                }
            }
        }
    }
    let input = x.map(Double.init)
    let gate = matvec(w[0], input, outputs: 640, width: 2560)
    let up = matvec(w[1], input, outputs: 640, width: 2560)
    return matvec(w[2], zip(gate, up).map { $0 / (1 + exp(-$0)) * $1 }, outputs: 2560, width: 640)
}

private func ggufClose(_ actual: [Float], _ expected: [Double], _ tolerance: Double, _ what: String) throws {
    try ggufRequire(actual.count == expected.count, "\(what): \(actual.count) outputs, expected \(expected.count)")
    var error = 0.0
    var norm = 0.0
    for (a, e) in zip(actual, expected) {
        error += (Double(a) - e) * (Double(a) - e)
        norm += e * e
    }
    let relative = (error / max(norm, 1e-30)).squareRoot()
    try ggufRequire(
        norm > 0 && relative < tolerance, "\(what): relative error \(relative) against the float64 reference of the dequantized weights")
}

/// GGUF experts (`gguf-experts`) on every path the engine serves them on agree with a float64 reference of GGUFQuant's
/// dequantized weights (SiLU, products, the rank-weighted sum) within relative 5e-3 (worst seen 1.7e-3, the reduction's bf16
/// output): the ten-expert decode (`evaluate`); its resident-first split into groups of scattered ranks
/// (`diagnosticReadySplit`, the encoded decode's groups), equal bit for bit; the prefill layer command (`evaluateDeviceLayer`,
/// waves 1 and 2; mul_mv at 5 rows, mul_mm at 12 and 16) and the device batches of the 256K tier (`evaluateDeviceBatch`),
/// equal to each other bit for bit. Layers: IQ2_XS, IQ3_S and IQ4_NL mixed; Q2_0; IQ1_M, IQ2_S and Q4_0. The layout is
/// pruned to ten experts, and the prefill read-ahead reads no other. A wrong type, part offset or rank is off by ~100%.
func ggufExpertsMatchReference() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let layout = GGUFExperts(experts: 10, types: ggufCheckTypes)
    ModelLayout.use(.ggufExperts, gguf: layout)
    defer { ModelLayout.use(.affine4Group64) }
    try ggufRequire(PackStore.expertBytes == layout.recordBytes && ModelLayout.routedExperts == 10, "GGUF layout did not switch")
    let data = (0..<3).map { ggufRecords(layout, layer: $0) }
    // Small activations: mul_mm stages them and the products as half, as ggml does.
    let input = (0..<(16 * 2560)).map { BF16.decode(BF16.encode(Float(($0 * 103) % 991 - 495) / 4000)) }
    let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 100_000_000)
    let reader: SourceExpertLoader.Reader = { key, _, destination in
        guard key.layer < 3, key.expert < 10 else {
            throw EngineError("Read expert \(key.expert) of layer \(key.layer), which the model does not have")
        }
        data[key.layer].records[key.expert].withUnsafeBytes { destination.copyMemory(from: $0) }
    }
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let gathered = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reductions)
        let ranks = [3, 8, 0, 9, 1, 6, 4, 2, 7, 5]
        let mix: [Float] = (0..<10).map { Float($0 + 1) / 55 }
        let decode = try ExpertExecution(
            device: device, ledger: ledger,
            configuration: ExpertExecutionConfiguration(
                mode: .slabPackedPreferred,
                cacheSlots: 12, slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2), packedAvailable: { false }, reader: reader)
        try decode.beginSequence()
        for layer in 0..<3 {
            let window = try decode.makeWindow(layer: layer, experts: Array(0..<10))
            try window.consumeTen(experts: Array(0..<10)) { buffers in
                let records = buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
                let x = input[0..<2560]
                let actual = try gathered.evaluate(Array(x), records: records, canonicalRanks: ranks, weights: mix)
                var expected = [Double](repeating: 0, count: 2560)
                for e in 0..<10 {
                    for (o, v) in ggufExpert(data[layer].weights[e], x).enumerated() { expected[o] += Double(mix[ranks[e]]) * v }
                }
                try ggufClose(actual, expected, 5e-3, "layer \(layer) decode")
                for first in [[0, 2, 4, 6, 8], [3], [9, 1, 0]] {
                    let split = try gathered.diagnosticReadySplit(
                        Array(x), records: records, canonicalRanks: ranks, weights: mix, firstIndices: first)
                    try ggufRequire(
                        split.map(\.bitPattern) == actual.map(\.bitPattern),
                        "layer \(layer) decode split \(first) differs from the whole group")
                }
                let arrival = try gathered.diagnosticReadySplit(
                    Array(x), records: records, canonicalRanks: ranks,
                    weights: mix, firstIndices: [], groups: [[4], [7], [0, 3, 8], [1, 2, 5, 6, 9]])
                try ggufRequire(
                    arrival.map(\.bitPattern) == actual.map(\.bitPattern),
                    "layer \(layer) incremental arrival differs from the whole group")
            }
        }
        try decode.finishSequence()
        try gathered.trim()

        let prefill = try ExpertExecution(
            device: device, ledger: ledger,
            configuration: ExpertExecutionConfiguration(
                mode: .slabPackedPreferred,
                cacheSlots: 528, slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4, prefillReadAhead: true, prefillLayerCommand: true),
            packedAvailable: { false },
            reader: reader)
        let transfer = try PrefillMoETransfer()
        try prefill.beginSequence()
        for (layer, tokens) in [(0, 12), (1, 16), (2, 5)] {
            var rows: [Int: [(Int, Int)]] = [:]
            for token in 0..<tokens { for rank in 0..<10 { rows[(token * 7 + rank) % 10, default: []].append((token, rank)) } }
            let experts = rows.keys.sorted()
            let x = Array(input[0..<(tokens * 2560)])
            let weights = (0..<(tokens * 10)).map { Float($0 % 17 + 1) / 128 }
            func workspace() throws -> PrefillMoEWorkspace {
                try PrefillMoEWorkspace(
                    input: x, selectedRows: rows,
                    budget: PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens)),
                    transfer: transfer)
            }
            let control = try workspace()
            let candidate = try workspace()
            let batches = try prefill.makeWindow(layer: layer, experts: experts)
            for first in stride(from: 0, to: experts.count, by: 8) {
                let group = Array(experts[first..<min(first + 8, experts.count)])
                try batches.consumeBatch(experts: group) { buffers in
                    try fused.evaluateDeviceBatch(
                        zip(group, buffers).map { expert, buffer in
                            FusedExpertExecution.DeviceJob(
                                expert: expert, slots: rows[expert]!.map { UInt32($0.0 * 10 + $0.1) }, buffer: buffer,
                                groupedExpert: true, simdMatrix: true)
                        }, workspace: control)
                }
            }
            batches.close()
            try prefill.beginPrefillReadAhead(layers: layer + 1)
            for earlier in 0..<layer { try prefill.finishPrefillReadAheadLayer(earlier) }
            let window = try prefill.makeWindow(layer: layer, experts: experts)
            try window.consumeLayer { buffers in
                try fused.evaluateDeviceLayer(
                    experts: experts, buffers: buffers, selectedRows: rows, workspace: candidate,
                    simdMatrix: true, waves: layer == 0 ? 2 : 1)
            }
            try prefill.finishPrefillReadAheadLayer(layer)
            prefill.finishPrefillReadAhead()
            let actual = try candidate.reduce(weights: weights, using: reductions)
            let batched = try control.reduce(weights: weights, using: reductions)
            try ggufRequire(
                actual.map(\.bitPattern) == batched.map(\.bitPattern),
                "layer \(layer) prefill: the layer command differs from the device batches")
            var expected = [Double](repeating: 0, count: tokens * 2560)
            for (expert, positions) in rows {
                for (token, rank) in positions {
                    for (o, v) in ggufExpert(data[layer].weights[expert], x[(token * 2560)..<((token + 1) * 2560)]).enumerated() {
                        expected[token * 2560 + o] += Double(weights[token * 10 + rank]) * v
                    }
                }
            }
            try ggufClose(actual, expected, 5e-3, "layer \(layer) prefill of \(tokens) tokens")
        }
        let read = prefill.snapshot().cache
        try prefill.finishSequence()
        try ggufRequire(read.pinnedSlots == 0, "prefill kept a record pinned")
    }
    try ggufRequire(ledger.snapshot().reserved == 0, "GGUF expert fixture leaked resources")
}

/// `gguf-experts.json` is untrusted input: a manifest of real banks (in a sparse file) gives the layout its records are
/// read with (16-byte aligned parts after the header, the slot the largest layer's record), and a manifest with another
/// format, a layer missing, a type without a kernel, a shape off the architecture, a bank past its file, a relative path or
/// a file of another size is refused with a coded error.
func ggufExpertManifestBounds() throws {
    let root = CheckFixtures.scratch.appendingPathComponent("gguf-manifest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let path = root.appendingPathComponent("experts.gguf").path
    let size: UInt64 = 1 << 30
    FileManager.default.createFile(atPath: path, contents: nil)
    let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
    try handle.truncate(atOffset: size)
    try handle.close()
    func bank(_ type: Int, rows: Int, columns: Int, at offset: Int) -> (
        path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int
    ) {
        let (block, bytes) = GGUFQuant.sizes[type]!
        return (path, offset, rows * (columns / block) * bytes, type, rows, columns)
    }
    // Ten experts per layer, layer 0 mixed (IQ2_XS, IQ3_S, IQ4_NL), the others Q2_0; banks one after another.
    var layers: [[(path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int)]] = []
    var cursor = 0
    for types in ggufCheckTypes {
        var list: [(path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int)] = []
        for (type, shape) in zip(types, [(640, 2560), (640, 2560), (2560, 640)]) {
            list.append(bank(type, rows: shape.0, columns: shape.1, at: cursor))
            cursor += 10 * list.last!.stride
        }
        layers.append(list)
    }
    try ggufRequire(UInt64(cursor) <= size, "the fixture's banks do not fit its file")
    let valid = try GGUFExperts(manifest: GGUFExperts.manifest(experts: 10, files: [(path, size)], layers: layers))
    let parts = valid.layers[0].parts
    try ggufRequire(
        valid.experts == 10 && parts.map(\.type) == [17, 21, 20] && parts[0].record == 16
            && parts.allSatisfy({ $0.record % 16 == 0 }) && parts[1].record >= parts[0].record + parts[0].bytes
            && valid.recordBytes == valid.layers.map(\.bytes).max() && valid.recordBytes % 16 == 0
            && parts[2].offset == layers[0][2].offset
            && valid == GGUFExperts(manifest: GGUFExperts.manifest(experts: 10, files: [(path, size)], layers: layers)),
        "the manifest's layout differs from its banks")
    func refused(_ what: String, _ data: Data, code: String = "model_incompatible") throws {
        do { _ = try GGUFExperts(manifest: data) } catch let error as EngineError where error.code == code { return }
        throw EngineError("A manifest with \(what) was accepted")
    }
    func edited(_ change: (inout [String: Any]) -> Void) throws -> Data {
        var object =
            try JSONSerialization.jsonObject(with: GGUFExperts.manifest(experts: 10, files: [(path, size)], layers: layers))
            as! [String: Any]
        change(&object)
        return try JSONSerialization.data(withJSONObject: object)
    }
    func part(_ object: inout [String: Any], layer: Int, _ name: String, _ key: String, _ value: Any) {
        var list = object["layers"] as! [[String: Any]]
        var entry = list[layer][name] as! [String: Any]
        entry[key] = value
        list[layer][name] = entry
        object["layers"] = list
    }
    try refused("another format", edited { $0["format"] = "origami-gguf-experts-v0" })
    try refused("47 layers", edited { $0["layers"] = Array(($0["layers"] as! [Any]).dropLast()) })
    try refused("9 experts", edited { $0["experts"] = 9 })
    try refused("a type without a kernel", edited { part(&$0, layer: 3, "up", "type", 43) })
    try refused("a shape off the architecture", edited { part(&$0, layer: 0, "down", "rows", 640) })
    try refused("a bank past its file", edited { part(&$0, layer: 47, "down", "offset", Int(size) - 1000) })
    try refused("a negative offset", edited { part(&$0, layer: 5, "gate", "offset", -16) })
    try refused("a relative path", edited { $0["files"] = [["path": "experts.gguf", "size": Int(size)]] })
    try refused("a file of another size", edited { $0["files"] = [["path": path, "size": Int(size) + 1]] }, code: "model_incomplete")
    try refused("a size that is a boolean", edited { part(&$0, layer: 1, "gate", "rows", true) })
    try refused("no JSON", Data("{".utf8))
}

/// Queued decode of GGUF experts (`GGUFExpertKernels.encodeDeferredGroup`): a group queued before its records are known,
/// then filled by the host or by the GPU's residency lookup, computes the joined ready group's bytes bit for bit (one row)
/// on seeded records of the types the published low-bit files use: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S gate and up
/// with Q2_0 or IQ4_NL down, and Q2_0 throughout. An empty fill writes nothing (`deferredGroupParity`).
func ggufQueuedDecodeMatchesJoined() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let types = [[16, 16, 42], [17, 17, 20], [18, 18, 42], [21, 21, 20], [22, 22, 42], [42, 42, 42]]
    let layout = GGUFExperts(experts: 10, types: types + Array(repeating: [42, 42, 42], count: ModelContract.layers - types.count))
    ModelLayout.use(.ggufExperts, gguf: layout)
    defer { ModelLayout.use(.affine4Group64) }
    let data = types.indices.map { ggufRecords(layout, layer: $0).records }
    let ledger = try MemoryLedger(ceiling: 1_000_000_000, transientMargin: 100_000_000)
    let reader: SourceExpertLoader.Reader = { key, _, destination in
        guard key.layer < types.count, key.expert < 10 else { throw EngineError("Read a record the fixture does not have") }
        data[key.layer][key.expert].withUnsafeBytes { destination.copyMemory(from: $0) }
    }
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let gathered = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reductions)
        let decode = try ExpertExecution(
            device: device, ledger: ledger,
            configuration: ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 12, slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2),
            packedAvailable: { false }, reader: reader)
        try decode.beginSequence()
        let input = (0..<2560).map { BF16.decode(BF16.encode(Float(($0 * 103) % 991 - 495) / 4000)) }
        for layer in types.indices {
            let window = try decode.makeWindow(layer: layer, experts: Array(0..<10))
            try window.consumeTen(experts: Array(0..<10)) { buffers in
                let records = buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
                try gathered.deferredGroupParity(
                    records: Array(records.prefix(7)), canonicalRanks: [3, 8, 0, 9, 1, 6, 4], input: input, layer: layer)
            }
        }
        try decode.finishSequence()
        try gathered.trim()
    }
    try ggufRequire(ledger.snapshot().reserved == 0, "GGUF queued decode fixture leaked resources")
}
