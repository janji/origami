import Foundation
import Metal
import OrigamiEngine

/// O05 compares every score bit and selected ID against the existing full scorer followed by its mask pass.
/// Fixtures have no model weights. FP4-representable inputs also exercise the scorer independently of whether a
/// future arithmetic contract rounds queries/keys before this point; this is not a test of that quantizer.
func familyMaskedIndexPreservesScoresAndSelection() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw EngineError("No Metal device") }
    let kernels = try FamilyKernels(device: device)
    let poison: UInt32 = 0x7FC0_0061
    func buffer(_ values: [Float]) -> MTLBuffer { device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)! }
    func filled(_ count: Int) -> MTLBuffer { buffer([Float](repeating: Float(bitPattern: poison), count: count)) }
    func bits(_ b: MTLBuffer) -> [UInt32] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt32.self), count: b.length / 4))
    }
    func encode(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("No Metal command")
        }
        do { try body(encoder) } catch {
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw EngineError("Masked index fixture Metal failure: \(String(describing: command.error))")
        }
    }
    typealias Ref = FamilyKernels.Ref
    // Real head geometry, a remainder after the eight-head loop, and a small one-head shape. Partial final blocks,
    // ratio-2 causal boundaries and padded/sliced rows exercise every independent address calculation.
    let cases: [(heads: Int, dim: Int, entries: Int, ratio: Int, first: Int, rows: Int)] = [
        (32, 128, 257, 1, 254, 3), (12, 128, 73, 2, 142, 4), (1, 16, 17, 1, 0, 2),
        (32, 128, 16393, 1, 16390, 2),
    ]
    let levels: [Float] = [-6, -4, -3, -2, -1.5, -1, -0.5, -0.0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    var state: UInt64 = 0xD1B5_4A32_D192_ED03
    func sample() -> Float {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return Float(state >> 40) / Float(1 << 23) - 1
    }
    for c in cases {
        let blocks = (c.entries + 7) / 8
        let stride = blocks + 3
        let prefix = 4
        let suffix = 7
        let grid = MTLSize(width: (c.entries + 63) / 64, height: c.rows, depth: 1)
        let group = MTLSize(width: 64, height: 1, depth: 1)
        for pattern in 0..<5 {
            let q = buffer(
                (0..<c.rows * c.heads * c.dim).map { i in pattern == 3 ? levels[i % levels.count] : pattern == 4 ? 0.25 : sample() })
            var keyValues = (0..<c.entries * c.dim).map { i in
                pattern == 3 ? levels[(i * 3) % levels.count] : pattern == 4 ? Float(1) : sample()
            }
            // Exceptional values in a discarded block must still become the same -inf as the baseline mask pass.
            if pattern == 2 && c.entries > 8 {
                keyValues[8 * c.dim] = .nan
                keyValues[8 * c.dim + 1] = .infinity
            }
            let keys = buffer(keyValues)
            let weights = buffer((0..<c.rows * c.heads).map { _ in pattern == 4 ? Float(1) : sample() })
            let keep = filled(prefix + c.rows * stride + suffix)
            let mask = keep.contents().assumingMemoryBound(to: Float.self) + prefix
            for row in 0..<c.rows {
                for block in 0..<blocks {
                    switch pattern {
                    case 0: mask[row * stride + block] = 1
                    case 1: mask[row * stride + block] = 0
                    case 2: mask[row * stride + block] = block % 2 == 0 ? 1 : 0
                    default: mask[row * stride + block] = (block + row) % 3 == 0 ? 1 : 0
                    }
                }
            }
            // Last pattern gets a real candidate mask, including equal-score block ties and causal visibility.
            let expected = filled(prefix + c.rows * c.entries + suffix)
            let scoreArgs = [Int32(c.heads), Int32(c.dim), Int32(c.entries), Int32(c.ratio), Int32(c.first)]
            try encode { e in
                try kernels.dispatch(
                    e, "index_scores", [Ref(q), Ref(keys), Ref(weights), Ref(expected, prefix * 4)], ints: scoreArgs, grid: grid,
                    group: group)
                if pattern == 4 {
                    try kernels.dispatch(
                        e, "candidate_blocks", [Ref(expected, prefix * 4), Ref(keep, prefix * 4)],
                        ints: [
                            Int32(c.entries), 8, Int32(blocks), Int32(min(2048, max(1, blocks / 2))), Int32(c.first), Int32(c.ratio),
                            Int32(stride),
                        ],
                        grid: MTLSize(width: c.rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
                }
                try kernels.dispatch(
                    e, "apply_block_mask", [Ref(expected, prefix * 4), Ref(keep, prefix * 4)],
                    ints: [Int32(c.entries), 8, Int32(blocks), Int32(stride)],
                    grid: MTLSize(width: (c.entries + 255) / 256, height: c.rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1)
                )
            }
            let originalMask = bits(keep)
            for sliced in [false, true] {
                let got = filled(prefix + c.rows * c.entries + suffix)
                try encode { e in
                    var row = 0
                    while row < c.rows {
                        let count = sliced ? 1 : c.rows
                        try kernels.dispatch(
                            e, "index_scores_masked",
                            [
                                Ref(q, row * c.heads * c.dim * 4), Ref(keys), Ref(weights, row * c.heads * 4),
                                Ref(got, (prefix + row * c.entries) * 4), Ref(keep, (prefix + row * stride) * 4),
                            ],
                            ints: [Int32(c.heads), Int32(c.dim), Int32(c.entries), Int32(c.ratio), Int32(c.first + row), 8, Int32(stride)],
                            grid: MTLSize(width: grid.width, height: count, depth: 1), group: group)
                        row += count
                    }
                }
                guard bits(got) == bits(expected), bits(keep) == originalMask else {
                    throw EngineError("Masked index score bits/padding differ: \(c), pattern \(pattern), sliced \(sliced)")
                }
                let selected = min(512, c.entries)
                let expectedIDs = filled(prefix + c.rows * selected + suffix)
                let gotIDs = filled(prefix + c.rows * selected + suffix)
                try encode { e in
                    for (scores, ids) in [(expected, expectedIDs), (got, gotIDs)] {
                        try kernels.dispatch(
                            e, "index_select", [Ref(scores, prefix * 4), Ref(ids, prefix * 4)],
                            ints: [
                                Int32(c.entries), Int32(c.ratio), Int32(selected), Int32(selected), Int32(c.first), 0, 1, 384,
                                Int32(selected),
                            ],
                            grid: MTLSize(width: c.rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
                guard bits(gotIDs) == bits(expectedIDs) else {
                    throw EngineError("Masked index selected IDs differ: \(c), pattern \(pattern), sliced \(sliced)")
                }
            }
        }
    }
}
