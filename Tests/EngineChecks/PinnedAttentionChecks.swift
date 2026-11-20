import Darwin
import Foundation
import OrigamiEngine

private func pinnedAttentionRequire(_ test: Bool, _ message: String) throws {
    if !test { throw EngineError(message) }
}

func pinnedAttentionDispatchBoundaries() throws {
    let cases: [(Int, Int)] = [
        (1, 0), (1023, 0), (1024, 64), (1025, 128), (8192, 128), (8193, 256), (32768, 256), (32769, 512), (65536, 512), (65537, 1024),
        (262144, 1024),
    ]
    for (keys, partitions) in cases {
        for rows in 1...min(keys, 2) {
            let actual = try PinnedAttention.vectorPartitions(keys: keys, queries: rows, architecture: .medium)
            try pinnedAttentionRequire(actual == partitions, "Pinned medium-device vector partition boundary changed")
        }
    }
    try pinnedAttentionRequire(
        try PinnedAttention.vectorPartitions(keys: 1024, queries: 1, architecture: .small) == 0, "Small-device vector threshold changed")
    try pinnedAttentionRequire(
        try PinnedAttention.vectorPartitions(keys: 4096, queries: 2, architecture: .small) == 64, "Small-device partition count changed")
    try pinnedAttentionRequire(
        try PinnedAttention.vectorPartitions(keys: 65536, queries: 1, architecture: .large) == 1024, "Large-device partition count changed")
}

func pinnedAttentionGroupingMaskAndRetirement() throws {
    let ledger = try MemoryLedger()
    let kernel = try PinnedAttention(ledger: ledger, architecture: .medium)
    try autoreleasepool {
        for (rows, keys) in [(1, 24), (2, 24), (3, 24), (9, 24), (1, 1024), (2, 1025), (3, 4097)] {
            let q = [Float](repeating: 0, count: rows * 6144)
            let k = [UInt16](repeating: 0, count: keys * 512)
            let v = (0..<(keys * 512)).map { BF16.encode($0 % 512 < 256 ? Float(0.125) : Float(-0.25)) }
            let positions = (0..<rows).map { _ in [UInt32(0)] }
            let oldCount = (keys - rows) * 512
            let actual = try kernel.attention(
                queries: q, oldKeys: Array(k[..<oldCount]), newKeys: Array(k[oldCount...]),
                oldValues: Array(v[..<oldCount]), newValues: Array(v[oldCount...]), selectedPositions: positions)
            let expected = (0..<(rows * 6144)).map { Float(($0 % 6144) / 256 < 12 ? 0.125 : -0.25) }
            try pinnedAttentionRequire(
                actual.map(\.bitPattern) == expected.map(\.bitPattern),
                "Pinned attention lost GQA12, selected key, row order, or old/new boundary at \(rows)x\(keys)")
            let empty = try kernel.attention(
                queries: q, oldKeys: k, newKeys: [], oldValues: v, newValues: [],
                selectedPositions: .init(repeating: [], count: rows))
            if rows <= 2 {
                try pinnedAttentionRequire(empty.allSatisfy { $0 == 0 }, "Fused fully masked attention must be zero")
            } else {
                let p = BF16.decode(BF16.encode(1 / Float(keys)))
                let expectedPositive = BF16.decode(BF16.encode(p * Float(keys) * 0.125))
                let expectedNegative = BF16.decode(BF16.encode(p * Float(keys) * -0.25))
                try pinnedAttentionRequire(
                    empty.enumerated().allSatisfy { index, value in
                        value == (index % 6144 / 256 < 12 ? expectedPositive : expectedNegative)
                    }, "Unfused fully masked attention must retain its full-domain uniform row")
            }
        }
    }
    try pinnedAttentionRequire(ledger.snapshot().reserved == 0, "Pinned attention buffers outlived synchronous work")
}

func pinnedAttentionRejectsUnadmittedAndInvalidWork() throws {
    let ledger = try MemoryLedger()
    let kernel = try PinnedAttention(ledger: ledger, architecture: .medium)
    let query = [Float](repeating: 0, count: 6144)
    let kv = [UInt16](repeating: 0, count: 1024)
    let badSelections: [[UInt32]] = [[0, 0], [1, 0], [2]]
    for ids in badSelections {
        var failed = false
        do {
            _ = try kernel.attention(queries: query, oldKeys: kv, newKeys: [], oldValues: kv, newValues: [], selectedPositions: [ids])
        } catch is EngineError { failed = true }
        try pinnedAttentionRequire(failed, "Invalid pinned attention selection was accepted")
    }
    let limited = try MemoryLedger(ceiling: UInt64(getpagesize()) + 1, transientMargin: 0)
    let denied = try PinnedAttention(ledger: limited, architecture: .medium)
    var failed = false
    do {
        _ = try denied.attention(queries: query, oldKeys: kv, newKeys: [], oldValues: kv, newValues: [], selectedPositions: [[0]])
    } catch let error as EngineError { failed = error.code == "insufficient_memory" }
    try pinnedAttentionRequire(failed && limited.snapshot().reserved == 0, "Attention memory denial retained prior resources")
    try pinnedAttentionRequire(ledger.snapshot().reserved == 0, "Rejected attention retained memory")
}

func pinnedAttentionPartialKeyReductionOrder() throws {
    // BF16 outputs recorded from MLX's steel_gemm_fused on these inputs, not from this engine.
    // Large cancellation makes tail-first vs tail-last accumulation observable.
    let cases: [(Int, Float)] = [
        (15, 1.625), (16, 0), (17, 0.8828125),
        (23, 0.90234375), (24, 0.9375), (25, 0.9140625), (31, 0.921875), (32, 0), (33, 0.9375),
    ]
    let ledger = try MemoryLedger()
    for batched in [false, true] {
        let kernel = try PinnedAttention(ledger: ledger, architecture: .medium, batched: batched)
        for (keys, positive) in cases {
            for rows in [3, 9] {
                try autoreleasepool {
                    let q = [Float](repeating: 0, count: rows * 6144)
                    let k = [UInt16](repeating: 0, count: keys * 512)
                    let v = (0..<(keys * 512)).map { index -> UInt16 in
                        let key = index / 512
                        let sign: Float = index % 512 < 256 ? 1 : -1
                        return BF16.encode(sign * (key == 0 ? 16_777_216 : key == keys - 1 ? -16_777_216 : 1))
                    }
                    let split = (keys - rows) * 512
                    let positions = (0..<rows).map { row in (0..<(keys - rows + row + 1)).map(UInt32.init) }
                    let actual = try kernel.attention(
                        queries: q, oldKeys: Array(k[..<split]), newKeys: Array(k[split...]),
                        oldValues: Array(v[..<split]), newValues: Array(v[split...]), selectedPositions: positions)
                    // The final causal row sees the complete frozen key domain.
                    try pinnedAttentionRequire(
                        actual.suffix(6144).enumerated().allSatisfy { index, value in
                            let expected: Float = positive == 0 ? 0 : index % 6144 / 256 < 12 ? positive : -positive
                            return value.bitPattern == expected.bitPattern
                        }, "Pinned PV changed fused GEMM tail-first arithmetic at \(rows)x\(keys), batched=\(batched)")
                }
                try pinnedAttentionRequire(ledger.waitForRelease { reserved, _ in reserved == 0 }, "PV tail check retained buffers")
            }
        }
    }
}
