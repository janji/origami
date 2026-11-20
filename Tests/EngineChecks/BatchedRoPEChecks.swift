import Foundation
import OrigamiEngine

func batchedRoPEPreservesScalarPositionsAndHeads() throws {
    let kernels = try ReferenceKernels()
    for (width, heads, rows) in [(128, 1, 64), (128, 4, 9), (256, 2, 7), (256, 24, 256)] {
        let positions = (0..<rows).map { [0, 3, 4, 5, 2047, 2048, 32767, 262143][$0 % 8] }
        let x = (0..<(rows * heads * width)).map { BF16.decode(BF16.encode(Float(($0 * 17) % 311 - 155) / 32)) }
        var expected: [Float] = []
        for row in 0..<rows {
            let start = row * heads * width
            expected += try kernels.rope(x: Array(x[start..<(start + heads * width)]), width: width, position: positions[row])
        }
        let actual = try kernels.ropeBatch(x: x, width: width, heads: heads, positions: positions)
        guard zip(actual, expected).allSatisfy({ $0.bitPattern == $1.bitPattern }) else {
            throw EngineError("Batched RoPE changed scalar bits at width \(width), heads \(heads), rows \(rows)")
        }
    }
}

func batchedRoPERejectsGeometryAndDefaultsOff() throws {
    guard !RuntimeConfiguration().batchedRoPE else { throw EngineError("Batched RoPE must default off") }
    let kernels = try ReferenceKernels()
    let cases: [([Float], Int, Int, [Int])] = [
        ([], 128, 1, []), ([1], 128, 1, [0]),
        ([Float](repeating: 0, count: 128), 128, 1, [-1]), ([Float](repeating: 0, count: 128), 128, 1, [262144]),
        ([Float](repeating: 0, count: 257 * 128), 128, 1, Array(0..<257)),
        ([Float](repeating: 0, count: 64), 64, 1, [0]), ([Float](repeating: 0, count: 3 * 128), 128, 3, [0]),
    ]
    for (x, width, heads, positions) in cases {
        var rejected = false
        do { _ = try kernels.ropeBatch(x: x, width: width, heads: heads, positions: positions) } catch { rejected = true }
        guard rejected else { throw EngineError("Invalid batched RoPE accepted") }
    }
}
