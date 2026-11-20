import Foundation
import Metal
import OrigamiEngine

/// A prompt's routed experts a group at a time (`FamilyRunner.runWave`): `encodeRowsBatch` of runs of awkward row counts
/// (9 to 2,048) gives the bits of one `encode` per run, for the real models' expert types and others; and
/// `scatter_add_ordered` gives the bits of one `scatter_add_rows` per run in run order, rows shared between runs included
/// (runs of 1 to 8 rows too, which the engine still runs on `mm`'s own path).
func familyExpertRunsMatchSingle() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw FixtureUnavailable(description: "no Metal device")
    }
    let kernels = try GGMLMatrixKernels(device: device)
    let family = try FamilyKernels(device: device)
    var random = MatrixRandom(state: 0xE7E2_0001)
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
        try body(encoder)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw EngineError("expert runs check: \(error)") }
    }
    var problems: [String] = []
    let counts = [9, 63, 64, 65, 2_047, 2_048, 33]
    let total = counts.reduce(0, +)
    let starts = counts.indices.map { counts[..<$0].reduce(0, +) }
    for type in [12, 13, 39, 8, 10, 14] {
        for (width, outputs) in [(512, 200), (256, 64)] {
            let (block, bytes) = GGUFQuant.sizes[type]!
            let matrixBytes = outputs * width / block * bytes
            let stride = (matrixBytes + 15) / 16 * 16
            let blocks = randomGGMLBlocks(type, values: width * outputs * counts.count, &random)
            let weights = buffer([UInt8](repeating: 0, count: counts.count * stride))
            for z in counts.indices {
                blocks.withUnsafeBytes {
                    weights.contents().advanced(by: z * stride).copyMemory(from: $0.baseAddress! + z * matrixBytes, byteCount: matrixBytes)
                }
            }
            let input = buffer(floats(total * width))
            let single = buffer([UInt8](repeating: 0xA5, count: total * outputs * 4))
            let batch = buffer([UInt8](repeating: 0xA5, count: total * outputs * 4))
            try run { encoder in
                for z in counts.indices {
                    try kernels.encode(
                        encoder, type: type, weights: weights, weightOffset: z * stride, width: width, outputs: outputs, input: input,
                        inputOffset: starts[z] * width * 4, rows: counts[z], output: single, outputOffset: starts[z] * outputs * 4)
                }
                try kernels.encodeRowsBatch(
                    encoder, type: type, matrices: counts.indices.map { (weights, $0 * stride) }, width: width, outputs: outputs,
                    input: input, inputOffsets: starts.map { $0 * width * 4 }, rows: counts, output: batch,
                    outputOffsets: starts.map { $0 * outputs * 4 })
            }
            if memcmp(single.contents(), batch.contents(), total * outputs * 4) != 0 {
                problems.append("\(GGUFQuant.name(type)) \(outputs) x \(width)")
            }
        }
    }
    // Runs of 1 to 8 rows, each count's in one `encodeFewRowsBatch` (mul_mv), at awkward offsets.
    for type in [12, 13, 39, 8] {
        let (width, outputs) = (512, 200)
        let (block, bytes) = GGUFQuant.sizes[type]!
        let stride = (outputs * width / block * bytes + 15) / 16 * 16
        let weights = buffer(randomGGMLBlocks(type, values: width * outputs * 5, &random) + [UInt8](repeating: 0, count: 80))
        let blocks = weights.contents()
        let packed = buffer([UInt8](repeating: 0, count: 5 * stride))
        for z in 0..<5 {
            packed.contents().advanced(by: z * stride).copyMemory(
                from: blocks + z * outputs * width / block * bytes, byteCount: outputs * width / block * bytes)
        }
        for rows in 1...8 {
            let starts = [0, 3, 17, 9, 30].map { $0 * 8 }
            let input = buffer(floats(40 * 8 * width))
            let single = buffer([UInt8](repeating: 0xA5, count: 40 * 8 * outputs * 4))
            let batch = buffer([UInt8](repeating: 0xA5, count: 40 * 8 * outputs * 4))
            try run { encoder in
                for z in 0..<5 {
                    try kernels.encode(
                        encoder, type: type, weights: packed, weightOffset: z * stride, width: width, outputs: outputs, input: input,
                        inputOffset: starts[z] * width * 4, rows: rows, output: single, outputOffset: starts[z] * outputs * 4)
                }
                try kernels.encodeFewRowsBatch(
                    encoder, type: type, matrices: (0..<5).map { (packed, $0 * stride) }, width: width, outputs: outputs, input: input,
                    inputOffsets: starts.map { $0 * width * 4 }, rows: rows, output: batch, outputOffsets: starts.map { $0 * outputs * 4 })
            }
            if memcmp(single.contents(), batch.contents(), 40 * 8 * outputs * 4) != 0 {
                problems.append("\(GGUFQuant.name(type)) \(rows) rows (mul_mv)")
            }
        }
    }
    // mul_mm's rows do not depend on the dispatch they are in: rows of a 9-, 64-, 2,048-, 4,096-row dispatch at any place
    // equal the same rows of one 8,192-row dispatch. (Runs of 8 rows or fewer take mul_mv, whose sums differ.)
    for type in [12, 13, 39] {
        let (width, outputs) = (512, 200)
        let weights = buffer(randomGGMLBlocks(type, values: width * outputs, &random))
        let input = buffer(floats(8_192 * width))
        let whole = buffer([UInt8](repeating: 0, count: 8_192 * outputs * 4))
        let parts = buffer([UInt8](repeating: 0xA5, count: 8_192 * outputs * 4))
        let pieces = [(0, 9), (9, 64), (73, 2_048), (2_121, 4_096), (6_217, 1_975)]
        try run { encoder in
            try kernels.encode(
                encoder, type: type, weights: weights, weightOffset: 0, width: width, outputs: outputs, input: input, inputOffset: 0,
                rows: 8_192, output: whole, outputOffset: 0)
            for (first, rows) in pieces {
                try kernels.encode(
                    encoder, type: type, weights: weights, weightOffset: 0, width: width, outputs: outputs, input: input,
                    inputOffset: first * width * 4, rows: rows, output: parts, outputOffset: first * outputs * 4)
            }
        }
        if memcmp(whole.contents(), parts.contents(), 8_192 * outputs * 4) != 0 {
            problems.append("\(GGUFQuant.name(type)) rows depend on their dispatch")
        }
    }
    // The ordered add: runs of every size into 300 output rows, rows repeated across runs, in run order.
    let n = 96
    let runRows = [1, 7, 8, 9, 63, 64, 65, 200, 3]
    var ids: [Int32] = []
    // A run holds a row at most once, as an expert does.
    for count in runRows { ids += Array(Set((0..<count * 4).map { _ in Int32(random.next() % 300) })).sorted().prefix(count) }
    let rows = ids.count
    let x = buffer(floats(rows * n))
    let w = buffer(floats(rows))
    let start = floats(300 * n)
    let one = buffer(start)
    let all = buffer(start)
    let idBuffer = buffer(ids.withUnsafeBytes { Array($0) })
    let sorted = (0..<rows).sorted { (ids[$0], $0) < (ids[$1], $1) }
    var targets: [Int32] = []
    var firsts: [Int32] = []
    for (j, q) in sorted.enumerated() where j == 0 || ids[q] != ids[sorted[j - 1]] {
        targets.append(ids[q])
        firsts.append(Int32(j))
    }
    firsts.append(Int32(rows))
    let order = buffer(sorted.map(Int32.init).withUnsafeBytes { Array($0) })
    let targetBuffer = buffer(targets.withUnsafeBytes { Array($0) })
    let firstBuffer = buffer(firsts.withUnsafeBytes { Array($0) })
    try run { encoder in
        var at = 0
        for count in runRows {
            try family.dispatch(
                encoder, "scatter_add_rows", [.init(x, at * n * 4), .init(idBuffer, at * 4), .init(w, at * 4), .init(one)],
                ints: [Int32(n), Int32(count)], grid: MTLSize(width: 1, height: count, depth: 1),
                group: MTLSize(width: 256, height: 1, depth: 1))
            at += count
        }
        try family.dispatch(
            encoder, "scatter_add_ordered", [.init(x), .init(order), .init(w), .init(all), .init(targetBuffer), .init(firstBuffer)],
            ints: [Int32(n), Int32(targets.count)], grid: MTLSize(width: 1, height: targets.count, depth: 1),
            group: MTLSize(width: 256, height: 1, depth: 1))
    }
    if memcmp(one.contents(), all.contents(), 300 * n * 4) != 0 { problems.append("scatter_add_ordered differs from scatter_add_rows") }
    if !problems.isEmpty { throw EngineError("Expert runs differ from one run at a time: " + problems.joined(separator: "; ")) }
}
