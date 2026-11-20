import Darwin
import Foundation
import OrigamiEngine

private func gdnPreparationRequire(_ condition: Bool, _ message: String) throws {
    if !condition { throw EngineError(message) }
}

func gdnPreparationPreservesBoundariesAndJoinedResults() throws {
    let ledger = try MemoryLedger()
    let reductions = try PinnedReductions(ledger: ledger)
    try autoreleasepool {
        let oneRow = [Float](repeating: 1, count: 128)
        let normalized = try reductions.l2Norm128(x: oneRow)
        let expectedNorm = BF16.decode(BF16.encode(Float(pow(Double(128), -0.5))))
        try gdnPreparationRequire(normalized.allSatisfy { $0 == expectedNorm }, "GDN L2 changed constant row normalization")
        let scaled = try reductions.l2Norm128(x: oneRow, scale: Float(pow(Double(128), -0.5)))
        try gdnPreparationRequire(scaled.allSatisfy { $0 == 1 / 128 }, "GDN query scale lost the BF16 scalar boundary")
        let zeros = (0..<128).map { Float(bitPattern: $0 % 2 == 0 ? 0 : 0x8000_0000) }
        let zeroResult = try reductions.l2Norm128(x: zeros)
        try gdnPreparationRequire(zeroResult.map(\.bitPattern) == zeros.map(\.bitPattern), "GDN L2 changed signed zeros")
        let exactDecay = try reductions.gdnDecay(
            aLog: .init(repeating: 0, count: 48),
            softplus: (0..<96).map { $0 % 2 == 0 ? Float.zero : Float.infinity })
        try gdnPreparationRequire(
            exactDecay == (0..<96).map { $0 % 2 == 0 ? Float(1) : Float.zero },
            "GDN decay changed zero or infinity identities")
        // Both model row extremes exercise the 16 normalized and 48 decay heads.
        for rows in [1, 7, 256] {
            let query = (0..<(rows * 2048)).map { BF16.decode(BF16.encode(Float(($0 * 17) % 211 - 105) / 64)) }
            let keys = (0..<(rows * 2048)).map { BF16.decode(BF16.encode(Float(($0 * 29) % 181 - 90) / 32)) }
            let aLog = (0..<48).map { Float($0 - 24) / 16 }
            let softplus = (0..<(rows * 48)).map { BF16.decode(BF16.encode(Float($0 % 157) / 32)) }
            let joined = try reductions.prepareGDN(query: query, keys: keys, aLog: aLog, softplus: softplus)
            let q = try reductions.l2Norm128(x: query, scale: Float(pow(Double(128), -0.5)))
            let k = try reductions.l2Norm128(x: keys)
            let decay = try reductions.gdnDecay(aLog: aLog, softplus: softplus)
            try gdnPreparationRequire(
                joined.query.map(\.bitPattern) == q.map(\.bitPattern)
                    && joined.keys.map(\.bitPattern) == k.map(\.bitPattern)
                    && joined.decay.map(\.bitPattern) == decay.map(\.bitPattern),
                "Joined GDN preparation differs from the separate graph operations")
            try gdnPreparationRequire(
                joined.decay.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
                "Finite nonnegative softplus produced invalid GDN decay")
        }
    }
    try gdnPreparationRequire(ledger.snapshot().reserved == 0, "GDN preparation escaped ledger retirement")
}

func gdnPreparationRejectsInvalidAndUnadmittedWork() throws {
    let ledger = try MemoryLedger()
    let reductions = try PinnedReductions(ledger: ledger)
    let invalid: [() throws -> Void] = [
        { _ = try reductions.l2Norm128(x: []) },
        { _ = try reductions.l2Norm128(x: .init(repeating: 1, count: 129)) },
        { _ = try reductions.l2Norm128(x: .init(repeating: 1, count: 128), scale: .infinity) },
        { _ = try reductions.l2Norm128(x: .init(repeating: 1, count: 257 * 2048)) },
        { _ = try reductions.gdnDecay(aLog: [], softplus: .init(repeating: 1, count: 48)) },
        { _ = try reductions.gdnDecay(aLog: .init(repeating: 1, count: 48), softplus: []) },
        { _ = try reductions.gdnDecay(aLog: .init(repeating: 1, count: 48), softplus: .init(repeating: 1, count: 49)) },
        { _ = try reductions.gdnDecay(aLog: .init(repeating: 1, count: 48), softplus: .init(repeating: 1, count: 257 * 48)) },
        {
            _ = try reductions.prepareGDN(
                query: .init(repeating: 1, count: 2048), keys: .init(repeating: 1, count: 2047),
                aLog: .init(repeating: 1, count: 48), softplus: .init(repeating: 1, count: 48))
        },
        {
            _ = try reductions.prepareGDN(
                query: .init(repeating: 1, count: 2048), keys: .init(repeating: 1, count: 2048),
                aLog: .init(repeating: 1, count: 48), softplus: .init(repeating: 1, count: 96))
        },
    ]
    for operation in invalid {
        var rejected = false
        do { try operation() } catch is EngineError { rejected = true }
        try gdnPreparationRequire(rejected, "Invalid GDN preparation was accepted")
    }
    try gdnPreparationRequire(ledger.snapshot().reserved == 0, "Rejected GDN preparation retained memory")
    let limited = try MemoryLedger(ceiling: UInt64(getpagesize()) + 1, transientMargin: 0)
    let denied = try PinnedReductions(ledger: limited)
    var rejected = false
    do {
        _ = try denied.prepareGDN(
            query: .init(repeating: 1, count: 2048), keys: .init(repeating: 1, count: 2048),
            aLog: .init(repeating: 1, count: 48), softplus: .init(repeating: 1, count: 48))
    } catch let error as EngineError { rejected = error.code == "insufficient_memory" }
    try gdnPreparationRequire(
        rejected && limited.snapshot().reserved == 0,
        "GDN preparation admission failure retained partial allocations")
}
