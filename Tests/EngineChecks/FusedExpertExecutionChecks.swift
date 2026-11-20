import Darwin
import Foundation
import Metal
import OrigamiEngine

private func fusedExpertRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func fusedExpertReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid fused expert unexpectedly succeeded")
}
private func fusedExpertDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}
private func fusedExpertEngine(_ ledger: MemoryLedger, slots: Int = 2) throws -> ExpertExecution {
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: slots, slotsPerSlab: slots,
        demandWindow: min(2, slots), readConcurrency: min(2, slots))
    return try ExpertExecution(device: fusedExpertDevice(), ledger: ledger, configuration: config, packedAvailable: { false }) {
        key, _, destination in
        for kind in ExpertProjection.allCases {
            let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
            let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
            let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
            for index in 0..<(819_200 / 4) {
                words[index] = UInt32(truncatingIfNeeded: UInt64(index + key.expert + 1) &* 0x9e37_79b9_7f4a_7c15)
            }
            for index in 0..<((kind.biasOffset - kind.scaleOffset) / 2) {
                scales[index] = BF16.encode(Float(index % 7 + 1) / 128)
                biases[index] = BF16.encode(Float(index % 9 - 4) / 64)
            }
        }
    }
}
private func fusedExpertInputs(_ rows: Int) -> [Float] {
    (0..<(rows * 2560)).map { index in
        let value: Int = (index * 13 + index / 2560 * 7) % 31 - 15
        return BF16.decode(BF16.encode(Float(value) / 32))
    }
}

func fusedExpertMatchesCachedAndGroupedProjectionSchedules() throws {
    let ledger = try MemoryLedger(ceiling: 150_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        let engine = try fusedExpertEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1])
        _ = try window.consume(expert: 0) { $0.bytes[0] }
        try window.consume(expert: 1) { buffer in
            try fusedExpertRequire(buffer.offset != 0, "Fused fixture did not exercise a nonzero slab offset")
            let cases: [(Bool, Bool, Int)] = [
                (false, false, 1), (false, false, 2), (false, true, 7),
                (false, false, 8), (false, true, 9), (false, true, 11), (false, false, 254), (false, false, 256),
                (true, false, 1), (true, false, 7), (true, true, 1), (true, true, 7), (true, true, 9),
            ]
            for (grouped, simd, rows) in cases {
                let input = fusedExpertInputs(rows)
                let expected = try ExpertExecution.evaluate(
                    input, buffer: buffer, kernels: kernels,
                    groupedExpert: grouped, simdMatrix: simd)
                let before = fused.snapshot().multirowQMVEncodedProjections
                let actual = try fused.evaluate(input, buffer: buffer, groupedExpert: grouped, simdMatrix: simd)
                try fusedExpertRequire(
                    fused.snapshot().multirowQMVEncodedProjections - before == (!grouped && rows > 1 ? 3 : 0),
                    "Multirow QMV route was bypassed or changed a grouped/row-one projection")
                try fusedExpertRequire(
                    actual.map(\.bitPattern) == expected.map(\.bitPattern),
                    "Fused expert changed output: rows=\(rows), grouped=\(grouped), SIMD=\(simd)")
                try fusedExpertRequire(
                    ledger.snapshot().categories["fused-expert-workspace"] == nil,
                    "Joined expert command retained private workspace")
            }
        }
        try engine.finishSequence()
        try fusedExpertRequire(
            engine.snapshot().cache.pinnedSlots == 0 && engine.snapshot().cache.inFlightReads == 0,
            "Successful fused evaluation retained a cache consumer")
    }
    try fusedExpertRequire(
        ledger.snapshot().categories["fused-expert-unary-table"] == nil && ledger.snapshot().categories["fused-expert-workspace"] == nil,
        "Fused helper outlived its resource reservations")
}

func fusedExpertPreservesIntermediateBF16AndUnaryBoundaries() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels()
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        try fusedExpertRequire(fused.unaryPolicy == kernels.arithmetic.policy, "Fused helper silently changed unary policy")
        let engine = try fusedExpertEngine(ledger, slots: 1)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 1, experts: [2])
        try window.consume(expert: 2) { buffer in
            func projection(_ x: [Float], _ kind: ExpertProjection) throws -> [Float] {
                try kernels.qmvPinnedBF16(
                    x: x, width: kind.width, outputs: kind.outputs, weights: buffer.buffer,
                    weightOffset: buffer.offset + kind.weightOffset, scaleOffset: buffer.offset + kind.scaleOffset,
                    biasOffset: buffer.offset + kind.biasOffset)
            }
            let input = fusedExpertInputs(3)
            let gate = try projection(input, .gate)
            let up = try projection(input, .up)
            let product = zip(gate, up).map { LayerArithmetic.round(kernels.arithmetic.silu($0) * $1) }
            let expected = try projection(product, .down)
            var captures: [String: [UInt32]] = [:]
            let actual = try fused.evaluate(
                input, buffer: buffer,
                capture: { name, values in captures[name] = values.map(\.bitPattern) })
            try fusedExpertRequire(
                captures["gate"] == gate.map(\.bitPattern) && captures["up"] == up.map(\.bitPattern)
                    && captures["product"] == product.map(\.bitPattern) && actual.map(\.bitPattern) == expected.map(\.bitPattern),
                "Fused gate, up, unary/product or down changed its BF16 boundary")
            // Exceptional input rows exercise SIMD reduction, table indexing,
            // product rounding and propagation through the down projection.
            var exceptional = fusedExpertInputs(3)
            exceptional[0] = Float(bitPattern: 0x7fc3_0000)
            exceptional[2560 + 11] = .infinity
            exceptional[5120 + 17] = -.infinity
            let reference = try ExpertExecution.evaluate(exceptional, buffer: buffer, kernels: kernels)
            let exceptionalOutput = try fused.evaluate(exceptional, buffer: buffer)
            try fusedExpertRequire(
                reference.map(\.bitPattern) == exceptionalOutput.map(\.bitPattern),
                "Fused expert changed exceptional-value output bits")
        }
        try engine.finishSequence()
    }
}

func fusedExpertCancellationJoinsBeforeConsumerRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        let engine = try fusedExpertEngine(ledger, slots: 1)
        for allowedChecks in [0, 1, 2] {
            try engine.beginSequence()
            let window = try engine.makeWindow(layer: 2, experts: [allowedChecks])
            var checks = 0
            var captures = 0
            do {
                _ = try window.consume(expert: allowedChecks) { buffer in
                    try fused.evaluate(
                        fusedExpertInputs(3), buffer: buffer,
                        shouldContinue: {
                            checks += 1
                            return checks <= allowedChecks
                        }, capture: { _, _ in captures += 1 })
                }
                throw EngineError("Cancelled fused expert returned successfully")
            } catch is IOOperationCancelled {}
            try fusedExpertRequire(
                checks == allowedChecks + 1 && captures == 0 && window.isClosed && engine.snapshot().cache.pinnedSlots == 0
                    && engine.snapshot().cache.inFlightReads == 0,
                "Fused cancellation returned before its command or cache pin retired")
            // A submitted command's scratch goes when Metal drops it, a moment after the join.
            try fusedExpertRequire(
                ledger.waitForRelease { _, categories in categories["fused-expert-workspace"] == nil },
                "Fused cancellation leaked its scratch")
            try engine.finishSequence()
        }
        // Reuse the same single physical slot after the submitted-command
        // cancellation case. The replacement record must produce its own data.
        try engine.beginSequence()
        let reused = try engine.makeWindow(layer: 2, experts: [9])
        try reused.consume(expert: 9) { buffer in
            let input = fusedExpertInputs(1)
            let expected = try ExpertExecution.evaluate(input, buffer: buffer, kernels: kernels)
            let actual = try fused.evaluate(input, buffer: buffer)
            try fusedExpertRequire(
                actual.map(\.bitPattern) == expected.map(\.bitPattern),
                "Cancelled command accessed the subsequently reused cache slot")
        }
        try engine.finishSequence()
    }
}

func fusedExpertGeometryAndAllocationFailureAreBounded() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger)
        let engine = try fusedExpertEngine(ledger, slots: 1)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [3])
        try window.consume(expert: 3) { buffer in
            for input in [[], [Float(0)], [Float](repeating: 0, count: 257 * 2560)] {
                try fusedExpertReject { _ = try fused.evaluate(input, buffer: buffer) }
            }
            for offset in [-4, 1, 4, Int.max] {
                try fusedExpertReject {
                    _ = try fused.evaluate(fusedExpertInputs(1), weights: buffer.buffer, recordOffset: offset)
                }
            }
            try fusedExpertRequire(
                ledger.snapshot().categories["fused-expert-workspace"] == nil,
                "Invalid fused geometry allocated private buffers")
            let page = UInt64(getpagesize())
            let small = try MemoryLedger(ceiling: 131_072 + page + 1, transientMargin: 1)
            try autoreleasepool {
                let denied = try FusedExpertExecution(arithmetic: .hostReference, ledger: small)
                try fusedExpertReject { _ = try denied.evaluate(fusedExpertInputs(1), buffer: buffer) }
                try fusedExpertRequire(
                    small.snapshot().categories["fused-expert-workspace"] == nil
                        && small.snapshot().categories["fused-expert-unary-table"] == 131_072,
                    "Partial fused allocation failure leaked scratch or lost the live table charge")
            }
            try fusedExpertRequire(small.snapshot().reserved == 0, "Fused helper destruction leaked its unary table")
        }
        try engine.finishSequence()
    }
}
