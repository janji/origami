import Darwin
import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of the row-one decode scalars (DecodeFrontierScalars), the router gather and
/// softmax and the shared-gate row: GPU results must be bit-identical to the host reference and
/// exceptional values must raise their flag. Also the small helpers the other decode checks share.
package enum DecodeFrontierSelfChecks {
    static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError("Decode frontier: " + message) }
    }
    static func equal(_ a: [Float], _ b: [Float], _ label: String) throws {
        try require(a.count == b.count, label + " count")
        if let i = a.indices.first(where: { a[$0].bitPattern != b[$0].bitPattern }) {
            throw EngineError("Decode frontier \(label)[\(i)] \(a[i].bitPattern) != \(b[i].bitPattern)")
        }
    }
    static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("Decode fixture allocation failed")
        }
        return result
    }
    static func values(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
    static func joined(_ command: MTLCommandBuffer) throws {
        command.commit()
        command.waitUntilCompleted()
        try require(command.status == .completed, "fixture command failed: \(String(describing: command.error))")
    }
    /// Inf, signalling NaN and subnormal shared-expert operands must raise the exception word.
    package static func exceptions() throws {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
            let fused = try FusedExpertExecution(arithmetic: arithmetic, ledger: ledger)
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let scalar = try DecodeFrontierScalars(device: device, ledger: ledger, arithmetic: arithmetic, fused: fused)
            // These values require original replay, never acceptance of a
            // canonicalized GPU NaN, flushed subnormal or underflow product.
            for exception in [Float.infinity, Float(bitPattern: 0x7fa1_2345), Float(bitPattern: 1)] {
                let gate = try buffer([Float](repeating: 1, count: 640), device)
                let up = try buffer([Float](repeating: exception, count: 640), device)
                let out = try buffer([UInt16](repeating: 0, count: 640), device)
                let flag = try buffer([UInt32(0)], device)
                let command = queue.makeCommandBuffer()!
                try scalar.encodeSharedProduct(command, gate: gate, up: up, output: out, exceptions: flag)
                try joined(command)
                try require(flag.contents().load(as: UInt32.self) != 0, "exceptional shared activation accepted")
            }
        }
        try require(ledger.snapshot().reserved == 0, "scalar fixture leaked")
    }

    package static func routing() throws {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        let host = try ledger.reserve(bytes: 16_000_000, category: "decode_router_fixture")
        try autoreleasepool {
            let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
            let fused = try FusedExpertExecution(arithmetic: arithmetic, ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let dense = try PinnedDenseLinear(ledger: ledger)
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let scalar = try DecodeFrontierScalars(device: device, ledger: ledger, arithmetic: arithmetic, fused: fused)
            let ordinary = (0..<512).map { (i: Int) in Float((i * 73) % 31 - 15) / 8 }
            var special = ordinary
            special[1] = .infinity
            special[2] = .nan
            special[3] = -.infinity
            special[4] = -0.0
            for scores in [ordinary, [Float](repeating: 1, count: 512), special] {
                let expectedIDs = try selector.topIndices(scores: scores, columns: 512, count: 10)[0]
                let expectedWeights = try reductions.softmax(rows: expectedIDs.map { scores[Int($0)] }, width: 10)
                let input = try buffer(scores, device)
                let pairs = try buffer([UInt32](repeating: 0, count: 20), device)
                let selected = try buffer([Float](repeating: 0, count: 10), device)
                let weights = try buffer([Float](repeating: 0, count: 10), device)
                let flag = try buffer([UInt32(0)], device)
                let command = queue.makeCommandBuffer()!
                _ = try selector.encodeTopIndices(command, scores: input, first: pairs, second: nil, rows: 1, columns: 512, count: 10)
                try scalar.encodeRouterGather(command, logits: input, pairs: pairs, selected: selected, exceptions: flag)
                try reductions.encodeRouterSoftmax10(command, input: selected, output: weights)
                try joined(command)
                let p = pairs.contents().assumingMemoryBound(to: UInt32.self)
                try require(
                    (0..<10).map { p[$0 * 2 + 1] } == expectedIDs && flag.contents().load(as: UInt32.self) == 0,
                    "router rank/tie identity differs")
                try equal(values(weights, 10), expectedWeights, "router softmax")
                p[1] = 512
                let bad = queue.makeCommandBuffer()!
                try scalar.encodeRouterGather(bad, logits: input, pairs: pairs, selected: selected, exceptions: flag)
                try joined(bad)
                try require(
                    flag.contents().load(as: UInt32.self) != 0 && values(selected, 1)[0] == 0, "invalid router index was not contained")
            }
            let x = (0..<2560).map { LayerArithmetic.round(Float(($0 * 73) % 127 - 63) / 64) }
            let w = try buffer((0..<2560).map { LayerArithmetic.round(Float(($0 * 31) % 97 - 48) / 128) }, device)
            let expected = try dense.linear(x, weights: w, width: 2560, outputs: 1, outputBF16: true)
            let input = try buffer(x, device)
            let output = try buffer([Float(0)], device)
            let command = queue.makeCommandBuffer()!
            try dense.encodeSharedGateRow(command, input: input, weights: w, output: output)
            try joined(command)
            try equal(values(output, 1), expected, "matrix-is-input shared gate")
        }
        try ledger.release(host)
        try require(ledger.snapshot().reserved == 0, "routing fixture leaked")
    }
}
