import Darwin
import Foundation
import OrigamiEngine

private func batchedSelectionRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func batchedSelectionReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid batched selection unexpectedly succeeded")
}

private func batchedSelectionKey(_ value: Float) -> UInt32 {
    var bits = value.bitPattern
    let magnitude = bits & 0x7fff_ffff
    if magnitude > 0x7f80_0000 { return 0 }
    if magnitude < 0x0080_0000 { bits = 0 }
    return bits & 0x8000_0000 != 0 ? ~bits : bits ^ 0x8000_0000
}

/// Independent CPU full sort. The implementation under test sorts bounded
/// chunks and merges their retained prefixes instead of sorting a full row.
private func batchedSelectionOracle(_ scores: [Float], columns: Int, count: Int) -> [[UInt32]] {
    (0..<(scores.count / columns)).map { row in
        let keyed: [(key: UInt32, index: UInt32)] = (0..<columns).map { column in
            (key: batchedSelectionKey(scores[row * columns + column]), index: UInt32(column))
        }
        let sorted = keyed.sorted { $0.key == $1.key ? $0.index < $1.index : $0.key > $1.key }
        return sorted.prefix(count).map(\.index)
    }
}

private func batchedSelectionInputs(rows: Int, columns: Int) -> [Float] {
    // Repeat some values across chunk boundaries and otherwise cover arbitrary
    // IEEE bit patterns, including both signs, exceptional values and subnormals.
    (0..<(rows * columns)).map { index in
        if index % 17 == 0 { return Float(3) }
        let bits = UInt32(truncatingIfNeeded: UInt64(index + 1) &* 0x9e37_79b9_7f4a_7c15)
        return Float(bitPattern: bits)
    }
}

func batchedTopIndicesMatchesScalarAndFullSort() throws {
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    let batched = try BatchedTopIndices(ledger: ledger)
    let scalar = try ReferenceKernels()
    let cases = [
        (1, 1, 1), (3, 3, 3), (33, 10, 5), (512, 10, 4), (1023, 512, 2),
        (1024, 512, 2), (1025, 512, 3), (2049, 511, 2), (4097, 512, 2),
        (8192, 512, 1), (65536, 512, 1),
    ]
    for (columns, count, rows) in cases {
        let scores = batchedSelectionInputs(rows: rows, columns: columns)
        let expected = batchedSelectionOracle(scores, columns: columns, count: count)
        let actual = try batched.topIndices(scores: scores, columns: columns, count: count)
        try batchedSelectionRequire(actual == expected, "Batched selection differs from full CPU sort for \(columns)x\(rows), k=\(count)")
        for row in 0..<rows {
            let one = try scalar.topIndices(scores: Array(scores[(row * columns)..<((row + 1) * columns)]), count: count)
            try batchedSelectionRequire(actual[row] == one, "Batched selection changed scalar indices for \(columns)x\(rows), row \(row)")
        }
        try batchedSelectionRequire(ledger.snapshot().reserved == 0, "Batched selection retained joined GPU buffers")
    }
}

func batchedTopIndicesPreservesExceptionalAndGlobalTieOrder() throws {
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    let batched = try BatchedTopIndices(ledger: ledger)
    let scalar = try ReferenceKernels()
    let bits: [UInt32] = [
        0xff80_0000, 0x7f80_0000, 0x7fc0_0000, 0xffc1_2345, 0x7f80_0001,
        0xff80_0001, 0x8000_0000, 0x0000_0000, 0x0000_0001, 0x007f_ffff, 0x8000_0001, 0x807f_ffff,
        0x0080_0000, 0x8080_0000, 0x3f80_0000, 0xbf80_0000, 0x7f7f_ffff, 0xff7f_ffff,
    ]
    let columns = 2053
    let mixed = (0..<columns).map { Float(bitPattern: bits[$0 % bits.count]) }
    let nan = (0..<columns).map { index in
        let prefix: UInt32 = index % 2 == 0 ? 0x7fc0_0000 : 0xffc0_0000
        return Float(bitPattern: prefix | UInt32(index))
    }
    let zeros = (0..<columns).map { Float(bitPattern: bits[6 + $0 % 6]) }
    let minusInfinity = [Float](repeating: -.infinity, count: columns)
    let scores = mixed + nan + zeros + minusInfinity
    let expected = batchedSelectionOracle(scores, columns: columns, count: 512)
    for _ in 0..<3 {
        let actual = try batched.topIndices(scores: scores, columns: columns, count: 512)
        try batchedSelectionRequire(actual == expected, "Exceptional-value selection changed across chunks, padding or repeat")
        for row in 1..<4 {
            try batchedSelectionRequire(
                actual[row] == (0..<512).map(UInt32.init),
                "Equal selection keys lost ascending global-index order")
        }
    }
    let reference = try scalar.topIndices(scores: mixed, count: 512)
    try batchedSelectionRequire(reference == expected[0], "Exceptional full-sort oracle differs from existing reference")
    try batchedSelectionRequire(ledger.snapshot().reserved == 0, "Exceptional selection leaked Metal resources")
}

func batchedTopIndicesCoversMaximumBatchAndDomain() throws {
    let rows = 256
    let columns = 65536
    let count = 512
    let ledger = try MemoryLedger(ceiling: 175_000_000, transientMargin: 1)
    let batched = try BatchedTopIndices(ledger: ledger)
    var scores = [Float](repeating: 0, count: rows * columns)
    for row in 0..<rows {
        let rotation = row * 257
        for column in 0..<columns { scores[row * columns + column] = Float((column + rotation) & 65535) }
    }
    let output = try batched.topIndices(scores: scores, columns: columns, count: count)
    try batchedSelectionRequire(output.count == rows, "Maximum selection batch lost rows")
    for row in 0..<rows {
        let expected = (0..<count).map { rank in
            let rotated: Int = columns - 1 - rank + columns - row * 257
            return UInt32(rotated & 65535)
        }
        try batchedSelectionRequire(output[row] == expected, "Maximum selection domain changed order at row \(row)")
    }
    try batchedSelectionRequire(
        ledger.snapshot().reserved == 0 && ledger.snapshot().reservationHighWater > 0,
        "Maximum selection buffers were not ledger-bound and retired")
}

func batchedTopIndicesRejectsGeometryAndRetiresDeniedAllocations() throws {
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    let batched = try BatchedTopIndices(ledger: ledger)
    let invalid: [([Float], Int, Int)] = [
        ([], 1, 1), ([1], 0, 1), ([1], 65537, 1),
        ([1], 1, 0), ([1], 1, 2), ([1, 2, 3], 2, 1),
        ([Float](repeating: 1, count: 257), 1, 1), ([Float](repeating: 1, count: 513), 513, 513),
    ]
    for (scores, columns, count) in invalid {
        try batchedSelectionReject { _ = try batched.topIndices(scores: scores, columns: columns, count: count) }
    }
    try batchedSelectionRequire(
        ledger.snapshot().reserved == 0 && ledger.snapshot().reservationHighWater == 0,
        "Invalid selection geometry allocated buffers before validation")
    let page = UInt64(getpagesize())
    let smallLedger = try MemoryLedger(ceiling: page + 1, transientMargin: 1)
    let small = try BatchedTopIndices(ledger: smallLedger)
    try batchedSelectionReject { _ = try small.topIndices(scores: [1, 2, 3, 4, 5], columns: 5, count: 3) }
    try batchedSelectionRequire(
        smallLedger.snapshot().reserved == 0 && smallLedger.snapshot().reservationHighWater == page,
        "Denied selection allocation leaked previously admitted input buffer")
}

func batchedIndexerSelectionPreservesCausalExceptionalDomain() throws {
    let ledger = try MemoryLedger(ceiling: 300_000_000, transientMargin: 1)
    let selection = try BatchedTopIndices(ledger: ledger)
    let reference = try ReferenceKernels()
    for length in [2047, 2048, 2049, 4099] {
        let positions = [0, 3, 4, min(length - 1, 2047), length - 1]
        let columns = length / 4
        var all: [Float] = []
        for row in positions.indices {
            var scores = (0..<columns).map { Float(($0 * 17 + row * 5) % 31) }
            if row == 1 { scores = [Float](repeating: .nan, count: columns) }
            if row == 2 {
                scores[0] = -.infinity
                scores[1] = Float.leastNonzeroMagnitude
            }
            all.append(contentsOf: scores)
        }
        let actual = try selection.selectedPositions(scores: all, positions: positions, kvLength: length)
        for row in positions.indices {
            let expected = try reference.selectedPositions(
                scores: Array(all[(row * columns)..<((row + 1) * columns)]),
                position: positions[row], kvLength: length)
            guard actual[row] == expected else { throw EngineError("Batched causal indexer changed full-domain exceptional selection") }
        }
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Batched causal selector retained transient allocations") }
}
