import Accelerate
import Foundation
import Metal
import OrigamiEngine

/// A small seeded generator for the ggml matrix checks.
struct MatrixRandom {
    var state: UInt64
    mutating func next() -> UInt32 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt32(truncatingIfNeeded: state >> 33)
    }
    /// Uniform in [-1, 1].
    mutating func unit() -> Float { Float(next() % 2_000_001) / 1_000_000 - 1 }
}

/// `values` (whole rows of whole blocks) of `type` as random raw blocks: random code bytes, and finite scales of sane
/// magnitude where the type keeps them (f16 super-block scales in ±[0.002, 0.05], MXFP4's exponent within 2^±8), so the
/// weights stay well inside half range for mul_mm's staging.
func randomGGMLBlocks(_ type: Int, values: Int, _ random: inout MatrixRandom) -> [UInt8] {
    let (block, bytes) = GGUFQuant.sizes[type]!
    switch type {
    case 0: return (0..<values).flatMap { _ in withUnsafeBytes(of: random.unit().bitPattern) { Array($0) } }
    case 1: return (0..<values).flatMap { _ in withUnsafeBytes(of: Float16(random.unit()).bitPattern) { Array($0) } }
    case 30: return (0..<values).flatMap { _ in withUnsafeBytes(of: BF16.encode(random.unit())) { Array($0) } }
    default: break
    }
    var raw = (0..<(values / block * bytes)).map { _ in UInt8(truncatingIfNeeded: random.next()) }
    let halves: [Int: [Int]] = [
        2: [0], 3: [0, 2], 42: [0], 6: [0], 7: [0, 2], 8: [0], 10: [80, 82], 11: [108], 12: [0, 2], 13: [0, 2],
        14: [208], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 29: [], 39: [],
    ]
    for at in stride(from: 0, to: raw.count, by: bytes) {
        let scale = Float16((0.002 + Float(random.next() % 48_001) / 1_000_000) * (random.next() & 1 == 0 ? 1 : -1)).bitPattern
        if type == 39 { raw[at] = UInt8(119 + random.next() % 17) }
        if type == 29 {  // IQ1_M keeps d in the top nibbles of its four scale words
            for k in 0..<4 { raw[at + 49 + 2 * k] = raw[at + 49 + 2 * k] & 0x0F | UInt8((scale >> UInt16(4 * k)) & 15) << 4 }
        }
        for offset in halves[type]! {
            let value = offset == halves[type]!.first ? scale : Float16((Float(random.next() % 48_001) / 1_000_000)).bitPattern
            raw[at + offset] = UInt8(value & 255)
            raw[at + offset + 1] = UInt8(value >> 8)
        }
    }
    return raw
}

/// GGMLMatrixKernels against GGUFQuant's CPU dequantization: for every type it runs, random valid blocks at widths 2048,
/// 4096, 5120 and 512 (IQ split mode) with output counts that are no multiple of any kernel's row tile, rows 1, 2, 7
/// (mul_mv) and 9 to 70 (mul_mm, partial tiles), plus 2048 x 128 and 1024 x 64 at 32 to 256 rows (whole mul_mm tiles),
/// on exactly sized buffers at nonzero offsets with canary bytes around each output; the float types also at
/// widths 99 (unaligned rows) and 24 (ggml's short kernel). Each output matches the float64 dot product of the
/// dequantized weights within 1e-5 + `tolerance` x sum |w x|: 1e-5 for mul_mv, which accumulates in f32 (worst seen
/// 2.3e-7); 5e-4 for mul_mm, which stages weights and activations as half (worst seen 9.2e-5), and 1e-3 for BF16,
/// staged as bfloat (worst seen 5.7e-4).
/// Misuse (a width off the block size, a misaligned offset, a short buffer, an output over the input, an unknown
/// type) is refused.
func ggmlMatrixKernelsMatchDequantized() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("Missing Metal device")
    }
    let kernels = try GGMLMatrixKernels(device: device)
    var random = MatrixRandom(state: 0x6767_1234)
    let canary = [UInt8](repeating: 0xA5, count: 64)
    func buffer(_ bytes: [UInt8], offset: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: offset + bytes.count, options: .storageModeShared)!
        bytes.withUnsafeBytes { b.contents().advanced(by: offset).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return b
    }
    // Misuse, refused before anything is encoded.
    do {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Encoder unavailable")
        }
        defer { encoder.endEncoding() }
        let w = device.makeBuffer(length: 4096 * 144, options: .storageModeShared)!
        let x = device.makeBuffer(length: 4 * 4096, options: .storageModeShared)!
        let y = device.makeBuffer(length: 4 * 4096, options: .storageModeShared)!
        let tiny = device.makeBuffer(length: 16, options: .storageModeShared)!
        let misuse: [(String, () throws -> Void)] = [
            (
                "width off the block",
                {
                    try kernels.encode(
                        encoder, type: 12, weights: w, weightOffset: 0, width: 100, outputs: 8, input: x, inputOffset: 0, rows: 1,
                        output: y, outputOffset: 0)
                }
            ),
            (
                "misaligned offset",
                {
                    try kernels.encode(
                        encoder, type: 12, weights: w, weightOffset: 8, width: 256, outputs: 8, input: x, inputOffset: 0, rows: 1,
                        output: y, outputOffset: 0)
                }
            ),
            (
                "short output",
                {
                    try kernels.encode(
                        encoder, type: 12, weights: w, weightOffset: 0, width: 256, outputs: 8, input: x, inputOffset: 0, rows: 1,
                        output: tiny, outputOffset: 0)
                }
            ),
            (
                "short weights",
                {
                    try kernels.encode(
                        encoder, type: 12, weights: w, weightOffset: 0, width: 256, outputs: 8192, input: x, inputOffset: 0, rows: 1,
                        output: y, outputOffset: 0)
                }
            ),
            (
                "output over input",
                {
                    try kernels.encode(
                        encoder, type: 12, weights: w, weightOffset: 0, width: 256, outputs: 8, input: x, inputOffset: 0, rows: 1,
                        output: x, outputOffset: 512)
                }
            ),
            (
                "unknown type",
                {
                    try kernels.encode(
                        encoder, type: 34, weights: w, weightOffset: 0, width: 256, outputs: 8, input: x, inputOffset: 0, rows: 1,
                        output: y, outputOffset: 0)
                }
            ),
        ]
        for (what, attempt) in misuse {
            do {
                try attempt()
                throw EngineError("ggml matrix kernels accepted a \(what)")
            } catch let error as EngineError where !error.description.contains("accepted") {}
        }
    }
    var problems: [String] = []
    for type in GGMLMatrixKernels.types.sorted() {
        let block = GGUFQuant.sizes[type]!.block
        // 2048 x 128 and 1024 x 64 at 32, 64 and 256 rows fill whole mul_mm tiles (its direct simdgroup stores).
        var shapes = [
            (2048, 67, [1, 2, 7, 70]), (4096, 37, [1, 9]), (5120, 21, [1, 2, 64]), (512, 13, [1, 3, 40]), (2048, 128, [32, 64]),
            (1024, 64, [256]),
        ]
        if block == 1 { shapes += [(99, 11, [1, 2, 33]), (24, 5, [1, 12])] }
        var worst = (relative: 0.0, at: "")
        var failed = false
        for (width, outputs, rowCounts) in shapes {
            let raw = randomGGMLBlocks(type, values: width * outputs, &random)
            var dequantized = [Float](repeating: .nan, count: width * outputs)
            raw.withUnsafeBytes { source in
                dequantized.withUnsafeMutableBufferPointer {
                    GGUFQuant.dequantize(source.baseAddress!, type: type, count: width * outputs, into: $0.baseAddress!)
                }
            }
            let w = dequantized.map(Double.init)
            let wAbs = w.map(abs)
            let weights = buffer(raw, offset: 48)
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw EngineError("Encoder unavailable")
            }
            var runs: [(rows: Int, x: [Double], output: MTLBuffer)] = []
            for rows in rowCounts {
                let x = (0..<(rows * width)).map { _ in random.unit() }
                let input = buffer(x.withUnsafeBytes { Array($0) }, offset: 16)
                // Canary bytes before and after the output show a store outside it.
                let output = buffer([Float](repeating: .nan, count: rows * outputs).withUnsafeBytes { Array($0) } + canary, offset: 32)
                memset(output.contents(), 0xA5, 32)
                do {
                    try kernels.encode(
                        encoder, type: type, weights: weights, weightOffset: 48, width: width, outputs: outputs, input: input,
                        inputOffset: 16,
                        rows: rows, output: output, outputOffset: 32)
                } catch {
                    encoder.endEncoding()
                    throw error
                }
                runs.append((rows, x.map(Double.init), output))
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw EngineError("\(GGUFQuant.name(type)) \(width) x \(outputs): \(error)") }
            for (rows, x, output) in runs {
                let bytes = output.contents().assumingMemoryBound(to: UInt8.self)
                let end = 32 + rows * outputs * 4
                if (0..<32).contains(where: { bytes[$0] != 0xA5 }) || (end..<(end + canary.count)).contains(where: { bytes[$0] != 0xA5 }) {
                    problems.append("\(GGUFQuant.name(type)) \(width) x \(outputs), \(rows) rows wrote outside its output")
                }
                let got = output.contents().advanced(by: 32).assumingMemoryBound(to: Float.self)
                let xAbs = x.map(abs)
                let tolerance = rows <= 8 || width < 64 ? 1e-5 : type == 30 ? 1e-3 : 5e-4
                for r in 0..<rows {
                    for o in 0..<outputs {
                        var reference = 0.0
                        var magnitude = 0.0
                        w.withUnsafeBufferPointer { wp in
                            x.withUnsafeBufferPointer { xp in
                                vDSP_dotprD(wp.baseAddress! + o * width, 1, xp.baseAddress! + r * width, 1, &reference, vDSP_Length(width))
                            }
                        }
                        wAbs.withUnsafeBufferPointer { wp in
                            xAbs.withUnsafeBufferPointer { xp in
                                vDSP_dotprD(wp.baseAddress! + o * width, 1, xp.baseAddress! + r * width, 1, &magnitude, vDSP_Length(width))
                            }
                        }
                        let value = Double(got[r * outputs + o])
                        let error = abs(value - reference)
                        let relative = error.isNaN ? .infinity : error / max(magnitude, 1e-30)
                        if relative > worst.relative {
                            worst = (relative, "\(width) x \(outputs), \(rows) rows, row \(r) output \(o): \(value) for \(reference)")
                        }
                        if !(error <= tolerance * magnitude + 1e-5) { failed = true }
                    }
                }
            }
        }
        if failed { problems.append("\(GGUFQuant.name(type)) worst relative error \(worst.relative) at \(worst.at)") }
    }
    guard problems.isEmpty else {
        throw EngineError("ggml matrix kernels differ from the dequantized reference: " + problems.joined(separator: "; "))
    }
}

/// `encodeBatch` against one `encode` per matrix, byte for byte: every quantized type at width 2048 and 512 (the IQ
/// split mode), 1 and 3 rows, five matrices spread over two buffers at different offsets, with separate inputs and
/// with one shared input (decode's gate and up). The weighted add of decode's routed experts (`add_weighted_rows`)
/// against one `scatter_add_rows` per expert, in order, byte for byte, and `add_weighted_rows_at` over the rows permuted
/// against it. A batch of none, of more than the table holds,
/// of a float type, or with outputs overlapping is refused.
func ggmlMatrixBatchMatchesSingle() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("Missing Metal device")
    }
    let kernels = try GGMLMatrixKernels(device: device)
    let family = try FamilyKernels(device: device)
    var random = MatrixRandom(state: 0xBA7C_4001)
    func buffer(_ bytes: [UInt8]) -> MTLBuffer {
        let b = device.makeBuffer(length: max(16, bytes.count), options: .storageModeShared)!
        bytes.withUnsafeBytes { if let base = $0.baseAddress { b.contents().copyMemory(from: base, byteCount: $0.count) } }
        return b
    }
    func floats(_ count: Int) -> [UInt8] { (0..<count).map { _ in random.unit() }.withUnsafeBytes { Array($0) } }
    func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Encoder unavailable")
        }
        do { try body(encoder) } catch {
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw EngineError("batch check: \(error)") }
    }
    var problems: [String] = []
    let count = 5
    let outputs = 67
    for type in GGMLMatrixKernels.types.sorted() where kernels.batches(type: type) {
        for width in [2048, 512] {
            let rowBytes = width / GGUFQuant.sizes[type]!.block * GGUFQuant.sizes[type]!.bytes
            let matrixBytes = outputs * rowBytes
            let stride = (matrixBytes + 15) / 16 * 16
            // Matrices 0, 2, 4 in one buffer, 1 and 3 in another, at offsets that are not a multiple of the stride.
            let a = buffer(
                randomGGMLBlocks(type, values: width * outputs * 3, &random) + [UInt8](repeating: 0, count: 3 * (stride - matrixBytes) + 48)
            )
            let b = buffer(
                [UInt8](repeating: 0, count: 32) + randomGGMLBlocks(type, values: width * outputs * 2, &random)
                    + [UInt8](repeating: 0, count: 2 * stride))
            let places = (0..<count).map { z -> (buffer: MTLBuffer, offset: Int) in
                z % 2 == 0 ? (a, z / 2 * stride) : (b, 32 + z / 2 * stride)
            }
            for rows in [1, 3] {
                for shared in [false, true] where !shared || rows == 1 {
                    let inputStride = shared ? 0 : rows * width * 4
                    let outputStride = (rows * outputs * 4 + 31) / 16 * 16  // a gap after each output
                    let input = buffer(floats(shared ? rows * width : count * rows * width))
                    let single = buffer([UInt8](repeating: 0xA5, count: count * outputStride))
                    let batch = buffer([UInt8](repeating: 0xA5, count: count * outputStride))
                    try run { encoder in
                        for (z, place) in places.enumerated() {
                            try kernels.encode(
                                encoder, type: type, weights: place.buffer, weightOffset: place.offset, width: width, outputs: outputs,
                                input: input, inputOffset: z * inputStride, rows: rows, output: single, outputOffset: z * outputStride)
                        }
                        try kernels.encodeBatch(
                            encoder, type: type, matrices: places, width: width, outputs: outputs, input: input, inputOffset: 0,
                            inputStride: inputStride, rows: rows, output: batch, outputOffset: 0, outputStride: outputStride)
                    }
                    if memcmp(single.contents(), batch.contents(), count * outputStride) != 0 {
                        problems.append("\(GGUFQuant.name(type)) \(width) wide, \(rows) rows\(shared ? ", shared input" : "")")
                    }
                }
            }
        }
    }
    // Decode's weighted add of eight experts against eight one-row scatter adds.
    let n = 300
    let k = 8
    let x = buffer(floats(k * n))
    let start = floats(n)
    let w = (0..<k).map { _ in random.unit() }
    let ids = buffer([Int32](repeating: 0, count: 1).withUnsafeBytes { Array($0) })
    let one = buffer(start)
    let all = buffer(start)
    try run { encoder in
        for z in 0..<k {
            try family.dispatch(
                encoder, "scatter_add_rows",
                [.init(x, z * n * 4), .init(ids), .init(buffer(withUnsafeBytes(of: w[z]) { Array($0) })), .init(one)],
                ints: [Int32(n), 1], grid: MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                group: MTLSize(width: 256, height: 1, depth: 1))
        }
        try family.dispatch(
            encoder, "add_weighted_rows", [.init(x), .init(all)], ints: [Int32(n), Int32(k)],
            floats: w + Array(repeating: 0, count: 64 - k),
            grid: MTLSize(width: (n + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }
    if memcmp(one.contents(), all.contents(), n * 4) != 0 { problems.append("add_weighted_rows differs from scatter_add_rows") }
    // The same add with x's rows in another order (decode's resident experts first): row z of x at row[z].
    let row = (0..<k).map { (3 * $0 + 1) % k }
    let permuted = buffer([UInt8](repeating: 0, count: k * n * 4))
    let at = buffer(start)
    for z in 0..<k {
        permuted.contents().advanced(by: row[z] * n * 4).copyMemory(from: x.contents().advanced(by: z * n * 4), byteCount: n * 4)
    }
    try run { encoder in
        try family.dispatch(
            encoder, "add_weighted_rows_at", [.init(permuted), .init(at)],
            ints: [Int32(n), Int32(k)] + row.map(Int32.init) + Array(repeating: 0, count: 64 - k),
            floats: w + Array(repeating: 0, count: 64 - k), grid: MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
            group: MTLSize(width: 256, height: 1, depth: 1))
    }
    if memcmp(at.contents(), all.contents(), n * 4) != 0 { problems.append("add_weighted_rows_at differs from add_weighted_rows") }
    // Misuse, refused before anything is encoded.
    do {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Encoder unavailable")
        }
        defer { encoder.endEncoding() }
        let w = device.makeBuffer(length: 1 << 20, options: .storageModeShared)!
        let x = device.makeBuffer(length: 1 << 16, options: .storageModeShared)!
        let y = device.makeBuffer(length: 1 << 20, options: .storageModeShared)!
        func batch(_ type: Int, _ matrices: Int, outputStride: Int) throws {
            try kernels.encodeBatch(
                encoder, type: type, matrices: Array(repeating: (w, 0), count: matrices), width: 256, outputs: 8, input: x, inputOffset: 0,
                inputStride: 0, rows: 1, output: y, outputOffset: 0, outputStride: outputStride)
        }
        let misuse: [(String, () throws -> Void)] = [
            ("batch of none", { try batch(12, 0, outputStride: 32) }),
            ("batch over the table", { try batch(12, GGMLMatrixKernels.batchCapacity + 1, outputStride: 32) }),
            ("float batch", { try batch(0, 2, outputStride: 32) }),
            ("overlapping outputs", { try batch(12, 2, outputStride: 16) }),
        ]
        for (what, attempt) in misuse {
            do {
                try attempt()
                problems.append("accepted a \(what)")
            } catch {}
        }
    }
    guard problems.isEmpty else {
        throw EngineError("ggml matrix batches differ from single dispatches: " + problems.joined(separator: "; "))
    }
}
