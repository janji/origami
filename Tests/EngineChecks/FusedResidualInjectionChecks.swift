import Foundation
import OrigamiEngine

private func injectionReference(_ x: [Float], _ residual: [Float], _ gates: [Float]) -> [Float] {
    var result = residual
    for token in 0..<(residual.count / 10240) {
        for branch in 0..<4 {
            for column in 0..<2560 {
                let index = token * 10240 + branch * 2560 + column
                result[index] = LayerArithmetic.round(
                    residual[index] + LayerArithmetic.round(x[token * 2560 + column] * gates[token * 4 + branch]))
            }
        }
    }
    return result
}

func fusedResidualInjectionPreservesStripedRowsAndBF16Domains() throws {
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernel = try FusedResidualInjection(ledger: ledger)
        guard !kernel.usesGPU(rows: 1), kernel.usesGPU(rows: 8), kernel.usesGPU(rows: 8192), !kernel.usesGPU(rows: 8193) else {
            throw EngineError("Residual injection threshold changed")
        }
        func compare(_ x: [Float], _ residual: [Float], _ gates: [Float], _ name: String) throws {
            let expected = injectionReference(x, residual, gates)
            let actual = try kernel.evaluate(output: x, residual: residual, gates: gates)
            if let mismatch = zip(expected, actual).enumerated().first(where: { $0.element.0.bitPattern != $0.element.1.bitPattern }) {
                throw EngineError(
                    "Residual injection \(name) mismatch at\(mismatch.offset): expected\(String(mismatch.element.0.bitPattern, radix: 16)) actual\(String(mismatch.element.1.bitPattern, radix: 16))"
                )
            }
            guard expected.count == actual.count, ledger.snapshot().reserved == 0,
                kernel.completedCommands == kernel.submittedCommands
            else { throw EngineError("Residual injection count, join, or ledger mismatch") }
        }
        for rows in [1, 3, 17, 256, 257] {
            let x = (0..<(rows * 2560)).map { LayerArithmetic.round(Float(($0 * 113 + 17) % 1009 - 504) / 127) }
            let r = (0..<(rows * 10240)).map { LayerArithmetic.round(Float(($0 * 71 + 23) % 1009 - 504) / 61) }
            let g = (0..<(rows * 4)).map { LayerArithmetic.round(Float($0 % 67 - 31) / 23) }
            try compare(x, r, g, "rows\(rows)")
        }
        let rows = 26
        let x = (0..<(rows * 2560)).map { BF16.decode(UInt16(truncatingIfNeeded: $0)) }
        let allResidual = (0..<(rows * 10240)).map { BF16.decode(UInt16(truncatingIfNeeded: $0)) }
        try compare(x, .init(repeating: 0.5, count: rows * 10240), .init(repeating: 0.75, count: rows * 4), "all-output-bf16")
        try compare(.init(repeating: 0.5, count: x.count), allResidual, .init(repeating: 0.75, count: rows * 4), "all-residual-bf16")
        // Opposite infinities, signed zeros and both NaN operands are separate
        // from the one-domain sweep, exposing contraction and payload behavior.
        let specials: [Float] = [0, -0.0, .infinity, -.infinity, BF16.decode(0x7fc1), BF16.decode(0xffc7), BF16.decode(0x7fa3), 1, -1]
        for gate in specials {
            let sx = (0..<2560).map { specials[$0 % specials.count] }
            let sr = (0..<10240).map { specials[($0 / specials.count) % specials.count] }
            try compare(sx, sr, .init(repeating: gate, count: 4), "special-gate\(gate.bitPattern)")
        }
        let tinyProducts: [Float] = [
            Float(bitPattern: 0x1c80_0000), Float(bitPattern: 0x2480_0000),
            Float(bitPattern: 0x0080_0000), Float(bitPattern: 0x8080_0000),
        ]
        for value in tinyProducts {
            try compare(
                .init(repeating: value, count: 2560), .init(repeating: BF16.decode(1), count: 10240),
                [Float(bitPattern: 0x2180_0000), -0.25, 0.25, 0.5], "underflow-\(value.bitPattern)")
        }
        guard kernel.repairedTiles > 0 else { throw EngineError("Exceptional residual tiles skipped exact host repair") }
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Residual injection retained resources") }
}

func fusedResidualInjectionDenialsAndCancellationJoin() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let kernel = try FusedResidualInjection(ledger: ledger)
    let x = [Float](repeating: 1, count: 2560)
    let r = [Float](repeating: 1, count: 10240)
    let g = [Float](repeating: 0.5, count: 4)
    for stop in [1, 3, 4] {
        var calls = 0
        do {
            _ = try kernel.evaluate(
                output: x, residual: r, gates: g,
                shouldContinue: {
                    calls += 1
                    return calls < stop
                })
            throw EngineError("Residual injection ignored cancellation")
        } catch is IOOperationCancelled {}
        guard kernel.submittedCommands == kernel.completedCommands, ledger.snapshot().reserved == 0 else {
            throw EngineError("Residual injection cancellation escaped storage or a command")
        }
    }
    do {
        _ = try kernel.evaluate(output: x, residual: r, gates: [])
        throw NSError(domain: "accepted-invalid", code: 1)
    } catch is EngineError {}
    let tiny = try MemoryLedger(ceiling: 1024, transientMargin: 1)
    let denied = try FusedResidualInjection(ledger: tiny)
    do {
        _ = try denied.evaluate(output: x, residual: r, gates: g)
        throw NSError(domain: "accepted-unadmitted", code: 1)
    } catch let error as EngineError { guard error.code == "insufficient_memory" else { throw error } }
    guard denied.submittedCommands == 0, tiny.snapshot().reserved == 0 else {
        throw EngineError("Unadmitted residual injection submitted or retained bytes")
    }
}
