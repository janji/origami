import Foundation
import Metal
import OrigamiEngine

/// Checks of a pruned GGUF's routing (`ModelLayout.routedExperts` below 512): the GPU router selection that decode, prefill
/// tiles and pre-routing share (`WholeLayerExecution.encodeRouterSelection`) and both host selections of the joined path
/// never choose an expert past the model's, although the converter's zero router rows give the padding logit 0 and every
/// real logit here is negative (without the mask the padding wins, which the check confirms).
package enum GGUFExpertSelfChecks {
    package static func prunedRouting() throws {
        func require(_ ok: Bool, _ what: @autoclosure () -> String) throws { if !ok { throw EngineError("Pruned routing: " + what()) } }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device")
        }
        let routed = 256
        ModelLayout.use(
            .ggufExperts, gguf: GGUFExperts(experts: routed, types: Array(repeating: [42, 42, 42], count: ModelContract.layers)))
        defer { ModelLayout.use(.affine4Group64) }
        try require(ModelLayout.routedExperts == routed, "the layout did not switch")
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 10_000_000)
        try autoreleasepool {
            let dense = try PinnedDenseLinear(ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            var router = [Float](repeating: 0, count: 512 * 2560)
            for e in 0..<routed { for i in 0..<2560 { router[e * 2560 + i] = -Float((e * 31 + i * 7) % 97 + 1) / 4096 } }
            func buffer(_ values: [Float]) throws -> MTLBuffer {
                guard
                    let b = values.withUnsafeBytes({
                        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                    })
                else {
                    throw EngineError("Pruned routing fixture allocation failed")
                }
                return b
            }
            let weights = try buffer(router)
            for rows in [1, 3] {
                let mixed = try buffer((0..<(rows * 2560)).map { (i: Int) in Float((i * 13) % 89 + 1) / 512 })
                let logits = try buffer([Float](repeating: 0, count: rows * 512))
                let plan = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 512)
                let partial = plan.kind == .splitK ? try buffer([Float](repeating: 0, count: rows * 512 * plan.splitKPartitions)) : nil
                let pairs = try buffer(
                    [Float](repeating: 0, count: try BatchedTopIndices.bufferPlan(rows: rows, columns: 512, count: 10).firstBytes / 4))
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Pruned routing command unavailable") }
                try WholeLayerExecution.encodeRouterSelection(
                    dense, selector, command, mixed: mixed, router: weights, logits: logits,
                    partial: partial, pairs: pairs, rows: rows)
                command.commit()
                command.waitUntilCompleted()
                try require(command.status == .completed, "the router command failed")
                let scores = Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float.self), count: rows * 512))
                let selected = pairs.contents().assumingMemoryBound(to: UInt32.self)
                // The router as the converter pads it, unmasked: real logits from the GPU, the padding's exactly 0.
                var unmasked = scores
                for row in 0..<rows { for column in routed..<512 { unmasked[row * 512 + column] = 0 } }
                var masked = unmasked
                BatchedTopIndices.maskColumns(&masked, columns: 512, eligible: routed)
                let padded = try selector.topIndices(scores: unmasked, columns: 512, count: 10)
                let host = try selector.topIndices(scores: masked, columns: 512, count: 10)
                for row in 0..<rows {
                    let real = Array(scores[(row * 512)..<(row * 512 + routed)])
                    try require(real.allSatisfy { $0.isFinite && $0 < 0 }, "the fixture's real logits are not all negative")
                    let expected = real.indices.sorted { (real[$0], $1) > (real[$1], $0) }.prefix(10).map(UInt32.init)
                    let gpu = (0..<10).map { selected[(row * 10 + $0) * 2 + 1] }
                    try require(gpu == expected, "the GPU router chose \(gpu) for rows \(rows), expected \(expected)")
                    try require(host[row] == expected, "the batched host selection chose \(host[row])")
                    let reference = try kernels.topIndices(scores: Array(masked[(row * 512)..<((row + 1) * 512)]), count: 10)
                    try require(reference == expected, "the reference host selection chose \(reference)")
                    try require(padded[row].allSatisfy { $0 >= UInt32(routed) }, "the fixture does not need the mask: \(padded[row])")
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "the fixture leaked reservations")
    }
}
