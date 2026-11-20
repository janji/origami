import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// The checkpoint's vision tower: a Qwen3-VL-style ViT, 27 pre-norm blocks of hidden 1152 with 16 heads,
/// 2-D rotary positions, interpolated learned position embeddings and a 2x2 patch merger to the language
/// model's 2560. Runs on the GPU through MPSGraph in float32 from the bf16 weights, which stay resident
/// (about 0.9 GB, charged to the ledger). Input: patches in merge-block order as the preprocessor emits them.
/// NUMERIC, not bitwise: MPSGraph chooses its own reduction order; engine-checks hold it to cosine 0.9999 of the
/// transformers reference per merged token. One encoder serves one request on one thread (not thread-safe).
public final class VisionEncoder {
    public static let hidden = 1152, heads = 16, headDim = 72, depth = 27, intermediate = 4304
    public static let patchValues = 3 * 2 * 16 * 16, merge = 2, gridSide = 48, output = 2560

    private let device: MTLDevice, queue: MTLCommandQueue
    private let weights: [String: (buffer: MTLBuffer, shape: [Int])]
    private let reservation: MemoryLedger.Reservation, ledger: MemoryLedger
    private let positionTable: [Float]  // the learned 48 x 48 grid, widened once
    private var graphs: [Int: (graph: MPSGraph, inputs: [String: MPSGraphTensor], output: MPSGraphTensor)] = [:]

    /// Every tensor the graph reads (without the `vision_tower.` prefix) and its shape.
    package static let shapes: [String: [Int]] = {
        var shapes = [
            "patch_embed.proj.weight": [hidden, 3, 2, 16, 16], "patch_embed.proj.bias": [hidden],
            "pos_embed.weight": [gridSide * gridSide, hidden], "merger.norm.weight": [hidden], "merger.norm.bias": [hidden],
            "merger.linear_fc1.weight": [4 * hidden, 4 * hidden], "merger.linear_fc1.bias": [4 * hidden],
            "merger.linear_fc2.weight": [output, 4 * hidden], "merger.linear_fc2.bias": [output],
        ]
        for block in 0..<depth {
            for (name, shape) in [
                ("norm1", [hidden]), ("norm2", [hidden]), ("attn.qkv", [3 * hidden, hidden]), ("attn.proj", [hidden, hidden]),
                ("mlp.linear_fc1", [intermediate, hidden]), ("mlp.linear_fc2", [hidden, intermediate]),
            ] {
                // A LayerNorm's bias has its weight's shape; a linear's bias has its output size.
                shapes["blocks.\(block).\(name).weight"] = shape
                shapes["blocks.\(block).\(name).bias"] = [shape[0]]
            }
        }
        return shapes
    }()

    /// The tower's bf16 weights, held while an encoder lives.
    public static func weightBytes(_ model: ModelSource) -> UInt64 {
        model.tensors.reduce(0) { $1.key.hasPrefix("vision_tower.") ? $0 + UInt64($1.value.length) : $0 }
    }
    /// A bound on the graph's float32 activations for `patches`: about 160 KB a patch, plus two 1024-query
    /// chunks of attention scores and probabilities (measured: 1.1 GB at 4096 patches).
    public static func workingBytes(patches: Int) -> UInt64 { UInt64(patches) * (160 * 1_024 + 2 * 16 * 1_024 * 4) }

    public init(model: ModelSource, ledger: MemoryLedger) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Vision encoder needs a Metal GPU")
        }
        let names = model.tensors.keys.filter { $0.hasPrefix("vision_tower.") }.sorted()
        guard names.count == Self.shapes.count else {
            throw EngineError("This checkpoint has no complete vision tower (\(names.count) tensors)", code: "model_incompatible")
        }
        // The graph force-reads these names and the buffers by these shapes: anything else would trap or read past a buffer.
        for name in names {
            let span = model.tensors[name]!
            let shape = Self.shapes[String(name.dropFirst("vision_tower.".count))]
            guard let shape, span.shape == shape, span.dtype == "BF16", span.length == Int64(shape.reduce(2, *)) else {
                throw EngineError(
                    "This checkpoint's vision tower does not match the encoder: \(name) is \(span.dtype) \(span.shape) "
                        + "(\(span.length) bytes); the encoder expects \(shape.map { "BF16 \($0)" } ?? "no tensor of that name")",
                    code: "model_incompatible")
            }
        }
        let bytes = names.reduce(UInt64(0)) { $0 + UInt64(model.tensors[$1]!.length) }
        reservation = try ledger.reserve(bytes: bytes, category: "vision-tower")
        self.ledger = ledger
        self.device = device
        self.queue = queue
        var weights: [String: (buffer: MTLBuffer, shape: [Int])] = [:]
        do {
            // Metal autoreleases new buffers: without this pool they would live until the serving thread's pool
            // drains at the end of the request, not when the encoder goes. Reads go straight into the buffers.
            var files: [String: ExactFile] = [:]
            try autoreleasepool {
                for name in names {
                    let span = model.tensors[name]!
                    guard let buffer = device.makeBuffer(length: Int(span.length), options: .storageModeShared) else {
                        throw EngineError("Vision weight allocation failed", code: "insufficient_memory")
                    }
                    let file = try files[span.file] ?? ExactFile(url: model.directory.appendingPathComponent(span.file), bypassCache: false)
                    files[span.file] = file
                    try file.read(
                        offset: Int64(span.offset), into: UnsafeMutableRawBufferPointer(start: buffer.contents(), count: Int(span.length)))
                    weights[String(name.dropFirst("vision_tower.".count))] = (buffer, span.shape)
                }
            }
        } catch {
            try? ledger.release(reservation)
            throw error
        }
        self.weights = weights
        let table = weights["pos_embed.weight"]!
        positionTable = (0..<(Self.gridSide * Self.gridSide * Self.hidden)).map {
            BF16.decode(table.buffer.contents().load(fromByteOffset: $0 * 2, as: UInt16.self))
        }
    }
    deinit { try? ledger.release(reservation) }

    /// `patches`: count x 1536 values (channel, frame, row, column), in merge-block order for a grid of
    /// `rows` x `columns` patches (one frame pair). Returns count / 4 merged tokens of 2560.
    public func encode(patches: [Float], rows: Int, columns: Int) throws -> [Float] {
        let count = rows * columns
        guard rows % Self.merge == 0, columns % Self.merge == 0, count > 0, patches.count == count * Self.patchValues else {
            throw EngineError("Vision patches do not match a \(rows) x \(columns) grid")
        }
        // Its own pool for the same reason: the graph's Metal buffers (about 1 GB at the cap) go with the call.
        return autoreleasepool {
            let (positions, cosines, sines) = Self.positions(rows: rows, columns: columns, table: positionTable)
            let compiled = graph(count: count)
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
            func floats(_ values: [Float], _ shape: [Int]) -> MPSGraphTensorData {
                let data = values.withUnsafeBytes { Data($0) }
                return MPSGraphTensorData(
                    device: MPSGraphDevice(mtlDevice: device), data: data, shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            }
            feeds[compiled.inputs["patches"]!] = floats(patches, [count, Self.patchValues])
            feeds[compiled.inputs["positions"]!] = floats(positions, [count, Self.hidden])
            feeds[compiled.inputs["cos"]!] = floats(cosines, [count, Self.headDim])
            feeds[compiled.inputs["sin"]!] = floats(sines, [count, Self.headDim])
            for (name, weight) in weights where name != "pos_embed.weight" {
                feeds[compiled.inputs[name]!] = MPSGraphTensorData(
                    weight.buffer, shape: weight.shape.map { NSNumber(value: $0) }, dataType: .bFloat16)
            }
            let results = compiled.graph.run(with: queue, feeds: feeds, targetTensors: [compiled.output], targetOperations: nil)
            var output = [Float](repeating: 0, count: count / 4 * Self.output)
            output.withUnsafeMutableBytes { results[compiled.output]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
            return output
        }
    }

    /// Interpolated position embeddings (bilinear over the 48 x 48 grid) and the 2-D rotary cos/sin, both in
    /// merge-block patch order, as the reference computes them.
    static func positions(rows: Int, columns: Int, table: [Float]) -> (embeddings: [Float], cos: [Float], sin: [Float]) {
        func linspace(_ n: Int) -> [Float] {
            n == 1 ? [0] : (0..<n).map { Float($0) * Float(gridSide - 1) / Float(n - 1) }
        }
        let hs = linspace(rows)
        let ws = linspace(columns)
        let inverse = (0..<(headDim / 4)).map { 1 / pow(Float(10_000), Float(2 * $0) / Float(headDim / 2)) }
        var embeddings = [Float](repeating: 0, count: rows * columns * hidden)
        var cosines = [Float](repeating: 0, count: rows * columns * headDim)
        var sines = cosines
        var index = 0
        for blockRow in 0..<(rows / merge) {
            for blockColumn in 0..<(columns / merge) {
                for innerRow in 0..<merge {
                    for innerColumn in 0..<merge {
                        let row = blockRow * merge + innerRow
                        let column = blockColumn * merge + innerColumn
                        let h = hs[row]
                        let w = ws[column]
                        let h0 = Int(h)
                        let w0 = Int(w)
                        let h1 = min(h0 + 1, gridSide - 1)
                        let w1 = min(w0 + 1, gridSide - 1)
                        let dh = h - Float(h0)
                        let dw = w - Float(w0)
                        let corners = [
                            (h0 * gridSide + w0, (1 - dh) * (1 - dw)), (h0 * gridSide + w1, (1 - dh) * dw),
                            (h1 * gridSide + w0, dh * (1 - dw)), (h1 * gridSide + w1, dh * dw),
                        ]
                        for d in 0..<hidden {
                            var value: Float = 0
                            for (cell, weight) in corners { value += table[cell * hidden + d] * weight }
                            embeddings[index * hidden + d] = value
                        }
                        // Rotary pairs: 18 row frequencies, then 18 column frequencies, repeated for the second half.
                        for k in 0..<(headDim / 2) {
                            let angle = k < headDim / 4 ? Float(row) * inverse[k] : Float(column) * inverse[k - headDim / 4]
                            cosines[index * headDim + k] = cos(angle)
                            cosines[index * headDim + k + headDim / 2] = cos(angle)
                            sines[index * headDim + k] = sin(angle)
                            sines[index * headDim + k + headDim / 2] = sin(angle)
                        }
                        index += 1
                    }
                }
            }
        }
        return (embeddings, cosines, sines)
    }

    private func graph(count: Int) -> (graph: MPSGraph, inputs: [String: MPSGraphTensor], output: MPSGraphTensor) {
        if let cached = graphs[count] { return cached }
        let graph = MPSGraph()
        var inputs: [String: MPSGraphTensor] = [:]
        func input(_ name: String, _ shape: [Int], _ type: MPSDataType) -> MPSGraphTensor {
            let tensor = graph.placeholder(shape: shape.map { NSNumber(value: $0) }, dataType: type, name: name)
            inputs[name] = tensor
            return tensor
        }
        func weight(_ name: String) -> MPSGraphTensor {
            graph.cast(input(name, weights[name]!.shape, .bFloat16), to: .float32, name: nil)
        }
        func constant(_ value: Double) -> MPSGraphTensor { graph.constant(value, dataType: .float32) }
        /// x [n, in] · W[out, in]^T + b
        func linear(_ x: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let w = graph.transposeTensor(weight(name + ".weight"), dimension: 0, withDimension: 1, name: nil)
            return graph.addition(graph.matrixMultiplication(primary: x, secondary: w, name: nil), weight(name + ".bias"), name: nil)
        }
        func layerNorm(_ x: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let mean = graph.mean(of: x, axes: [-1], name: nil)
            let centered = graph.subtraction(x, mean, name: nil)
            let variance = graph.mean(of: graph.multiplication(centered, centered, name: nil), axes: [-1], name: nil)
            let scale = graph.reciprocalSquareRoot(graph.addition(variance, constant(1e-6), name: nil), name: nil)
            let normalized = graph.multiplication(centered, scale, name: nil)
            return graph.addition(graph.multiplication(normalized, weight(name + ".weight"), name: nil), weight(name + ".bias"), name: nil)
        }
        func geluTanh(_ x: MPSGraphTensor) -> MPSGraphTensor {
            let cube = graph.multiplication(graph.multiplication(x, x, name: nil), x, name: nil)
            let inner = graph.multiplication(
                constant(0.7978845608028654), graph.addition(x, graph.multiplication(constant(0.044715), cube, name: nil), name: nil),
                name: nil)
            return graph.multiplication(
                graph.multiplication(constant(0.5), x, name: nil),
                graph.addition(constant(1), graph.tanh(with: inner, name: nil), name: nil), name: nil)
        }
        func geluErf(_ x: MPSGraphTensor) -> MPSGraphTensor {
            let erf = graph.erf(with: graph.multiplication(x, constant(0.7071067811865476), name: nil), name: nil)
            return graph.multiplication(
                graph.multiplication(constant(0.5), x, name: nil), graph.addition(constant(1), erf, name: nil), name: nil)
        }
        let patches = input("patches", [count, Self.patchValues], .float32)
        let positions = input("positions", [count, Self.hidden], .float32)
        let cosines = graph.reshape(
            input("cos", [count, Self.headDim], .float32), shape: [NSNumber(value: count), 1, NSNumber(value: Self.headDim)], name: nil)
        let sines = graph.reshape(
            input("sin", [count, Self.headDim], .float32), shape: [NSNumber(value: count), 1, NSNumber(value: Self.headDim)], name: nil)
        // The Conv3d with kernel = stride is a linear map of the flattened patch.
        let projection = graph.reshape(
            weight("patch_embed.proj.weight"), shape: [NSNumber(value: Self.hidden), NSNumber(value: Self.patchValues)], name: nil)
        var x = graph.addition(
            graph.matrixMultiplication(
                primary: patches, secondary: graph.transposeTensor(projection, dimension: 0, withDimension: 1, name: nil), name: nil),
            weight("patch_embed.proj.bias"), name: nil)
        x = graph.addition(x, positions, name: nil)
        func rotate(_ t: MPSGraphTensor) -> MPSGraphTensor {
            let half = Self.headDim / 2
            let first = graph.sliceTensor(t, dimension: 2, start: 0, length: half, name: nil)
            let second = graph.sliceTensor(t, dimension: 2, start: half, length: half, name: nil)
            let rotated = graph.concatTensors([graph.negative(with: second, name: nil), first], dimension: 2, name: nil)
            return graph.addition(graph.multiplication(t, cosines, name: nil), graph.multiplication(rotated, sines, name: nil), name: nil)
        }
        let shape3 = [NSNumber(value: count), NSNumber(value: Self.heads), NSNumber(value: Self.headDim)]
        for block in 0..<Self.depth {
            let prefix = "blocks.\(block)"
            let qkv = linear(layerNorm(x, prefix + ".norm1"), prefix + ".attn.qkv")
            func part(_ i: Int) -> MPSGraphTensor {
                graph.reshape(
                    graph.sliceTensor(qkv, dimension: 1, start: i * Self.hidden, length: Self.hidden, name: nil), shape: shape3, name: nil)
            }
            // [count, heads, dim] -> [heads, count, dim]
            let q = graph.transposeTensor(rotate(part(0)), dimension: 0, withDimension: 1, name: nil)
            let k = graph.transposeTensor(rotate(part(1)), dimension: 0, withDimension: 1, name: nil)
            let v = graph.transposeTensor(part(2), dimension: 0, withDimension: 1, name: nil)
            // Attention in chunks of 1024 queries: the whole 16 x count x count score matrix would be 1 GB at 4096
            // patches. The softmax is normalised after the value product: MPSGraph fuses a softmax (or exp over its sum)
            // followed by a matmul into its attention kernel, whose sdpa_tile_fwd_8x8x8_* pipelines fail to compile
            // under 512 patches on macOS 26 with an assertion that aborts the process.
            let keys = graph.transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
            let pieces = stride(from: 0, to: count, by: 1024).map { start -> MPSGraphTensor in
                let queries = graph.sliceTensor(q, dimension: 1, start: start, length: min(1024, count - start), name: nil)
                let scores = graph.multiplication(
                    graph.matrixMultiplication(primary: queries, secondary: keys, name: nil),
                    constant(1 / Double(Self.headDim).squareRoot()), name: nil)
                let exponentials = graph.exponent(
                    with: graph.subtraction(scores, graph.reductionMaximum(with: scores, axis: -1, name: nil), name: nil), name: nil)
                return graph.division(
                    graph.matrixMultiplication(primary: exponentials, secondary: v, name: nil),
                    graph.reductionSum(with: exponentials, axis: -1, name: nil), name: nil)
            }
            let attended = pieces.count == 1 ? pieces[0] : graph.concatTensors(pieces, dimension: 1, name: nil)
            let merged = graph.reshape(
                graph.transposeTensor(attended, dimension: 0, withDimension: 1, name: nil),
                shape: [NSNumber(value: count), NSNumber(value: Self.hidden)], name: nil)
            x = graph.addition(x, linear(merged, prefix + ".attn.proj"), name: nil)
            x = graph.addition(
                x, linear(geluTanh(linear(layerNorm(x, prefix + ".norm2"), prefix + ".mlp.linear_fc1")), prefix + ".mlp.linear_fc2"),
                name: nil)
        }
        let grouped = graph.reshape(
            layerNorm(x, "merger.norm"), shape: [NSNumber(value: count / 4), NSNumber(value: 4 * Self.hidden)], name: nil)
        let output = linear(geluErf(linear(grouped, "merger.linear_fc1")), "merger.linear_fc2")
        let compiled = (graph, inputs, output)
        graphs[count] = compiled
        return compiled
    }
}
