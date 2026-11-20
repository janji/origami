import Darwin
import Foundation
import OrigamiEngine

private func pleDotRequire(_ value: Bool, _ message: String) throws {
    if !value { throw EngineError(message) }
}

func pinnedPLEDotPreservesBF16ProductLocalAndSIMDBoundaries() throws {
    let ledger = try MemoryLedger()
    let width = 2560
    let kernel = try PinnedReductions(ledger: ledger)
    try autoreleasepool {
        // Cross source dispatch boundary (row-looped <32, row-simple >=32),
        // then reach the full 256-token/four-branch geometry.
        for rows in [1, 31, 32, 1024] {
            let actual = try kernel.pleDot(
                key: .init(repeating: 1, count: rows * width),
                query: .init(repeating: 1, count: rows * width))
            try pleDotRequire(actual == .init(repeating: 2560, count: rows), "PLE dot changed uniform source reduction")
        }
        var key = [Float](repeating: 1, count: 5 * width)
        var query = [Float](repeating: 0, count: key.count)
        // Thread-local addition narrows after each of four reads: result 1,
        // while the old FP32 serial sum produced 2.
        for (i, value) in [Float(256), 1, -256, 1].enumerated() { query[i] = value }
        // Each SIMD's result also narrows: first SIMD 256+1 ->256, second
        // SIMD -256; final result zero. Omitting this boundary produces 1.
        query[width] = 256
        query[width + 4] = 1
        query[width + 128] = -256
        // The product is narrowed before it reaches a local accumulator.
        key[2 * width] = 1.0078125
        query[2 * width] = 1.5
        key[2 * width + 1] = -1
        query[2 * width + 1] = 1.5
        // Distinct signs at distant groups expose row/branch indexing errors.
        query[3 * width + 2559] = -4
        query[4 * width + 256] = 8
        let actual = try kernel.pleDot(key: key, query: query)
        let expected: [Float] = [1, 0, 0.015625, -4, 8]
        try pleDotRequire(
            actual.map(\.bitPattern) == expected.map(\.bitPattern),
            "PLE dot lost product, lane-local, SIMD partial or row boundaries")
        var exceptional = [Float](repeating: 0, count: 4 * width)
        exceptional[0] = .infinity
        exceptional[width] = -.infinity
        exceptional[2 * width] = .nan
        exceptional[3 * width] = .infinity
        exceptional[3 * width + 1] = -.infinity
        let special = try kernel.pleDot(key: .init(repeating: 1, count: exceptional.count), query: exceptional)
        try pleDotRequire(
            special[0] == .infinity && special[1] == -.infinity && special[2].isNaN && special[3].isNaN,
            "PLE dot changed nonfinite reduction classes")
    }
    try pleDotRequire(ledger.snapshot().reserved == 0, "PLE dot resources escaped joined completion")
}

func pinnedPLEDotRejectsInvalidOrUnadmittedWork() throws {
    let ledger = try MemoryLedger()
    let kernel = try PinnedReductions(ledger: ledger)
    for (key, query) in [
        ([Float](), []), ([Float](repeating: 1, count: 2560), [1]),
        ([Float](repeating: 1, count: 2559), [Float](repeating: 1, count: 2559)),
    ] {
        var rejected = false
        do { _ = try kernel.pleDot(key: key, query: query) } catch is EngineError { rejected = true }
        try pleDotRequire(rejected, "PLE dot accepted invalid geometry")
    }
    let oversized = [Float](repeating: 0, count: 1025 * 2560)
    var rejected = false
    do { _ = try kernel.pleDot(key: oversized, query: oversized) } catch is EngineError { rejected = true }
    try pleDotRequire(rejected && ledger.snapshot().reserved == 0, "PLE dot accepted oversized geometry or allocated before validating")
    // One input fits one physical page, but a second does not. Both failure
    // rollback and result retirement use each actual Metal resource's owner.
    let limited = try MemoryLedger(ceiling: UInt64(getpagesize()) + 1, transientMargin: 0)
    let denied = try PinnedReductions(ledger: limited)
    rejected = false
    do { _ = try denied.pleDot(key: .init(repeating: 1, count: 2560), query: .init(repeating: 1, count: 2560)) } catch let error
        as EngineError
    { rejected = error.code == "insufficient_memory" }
    try pleDotRequire(rejected && limited.snapshot().reserved == 0, "PLE dot admission failure leaked an input reservation")
}
