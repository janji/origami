import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of `CompactSelectedAttention`: layout bounds for every row count, and bitwise
/// agreement with the original batched path, including a masked query's 0 x NaN/Inf in a shared tile.
public enum CompactAttentionSelfChecks {
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError(message) }
    }
    public static func layoutBounds() throws {
        var previous = 0
        for rows in 3...256 {
            let p = try CompactSelectedAttention.layout(rows: rows)
            try require(
                p.probabilitiesOffset >= rows * 7179 * 4
                    && p.defaultsOffset >= p.probabilitiesOffset + rows * 24 * 2051 * 2
                    && p.bytes >= p.defaultsOffset + rows * 24 * 2
                    && [p.probabilitiesOffset, p.defaultsOffset, p.bytes].allSatisfy { $0 % 256 == 0 }
                    && p.bytes > previous && p.bytes <= PinnedAttention.maximumScoreScratchBytes,
                "Compact scratch subregions overlap, are unaligned or exceed the admitted ceiling")
            previous = p.bytes
        }
        try require(previous == 32_566_272, "Maximum compact layout changed without its simultaneous bound")
        for rows in [-1, 0, 1, 2, 257, Int.max] {
            var refused = false
            do { _ = try CompactSelectedAttention.layout(rows: rows) } catch is EngineError { refused = true }
            try require(refused, "Invalid compact geometry accepted")
        }
    }
    public static func originalPairSemantics() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Metal unavailable for compact attention check")
        }
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            // Charge all actual allocations before creating them; retain until
            // the command and its autoreleased resources have fully drained.
            let lease = try ledger.reserve(bytes: 100_000_000, category: "compact_attention_fixture")
            defer { try? ledger.release(lease) }
            var admitted = 0
            func buffer(_ bytes: Int) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw EngineError("Compact fixture allocation failed")
                }
                admitted += b.allocatedSize
                try require(admitted <= 100_000_000, "Compact fixture exceeded its real-size bound")
                return b
            }
            let rows = 3
            let keys = 32769
            let query = try buffer(rows * 6144 * 2)
            let key = try buffer(keys * 512 * 2)
            let value = try buffer(keys * 512 * 2)
            let ids = try buffer(rows * 4)
            let counts = try buffer(rows * 4)
            let expected = try buffer(rows * 6144 * 4)
            let actual = try buffer(rows * 6144 * 4)
            let plan = try attention.bufferScratchPlan(rows: rows, keys: keys)
            let scratch = try buffer(plan.scoreBytes)
            let oldScratch = try buffer(PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys).scoreScratchBytes)
            let tooSmall = try buffer(plan.scoreBytes - 1)
            for (b, count) in [(query, rows * 6144), (key, keys * 512), (value, keys * 512)] {
                let ptr = b.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count { ptr[i] = BF16.encode(Float((i * 13 + 3) % 127 - 63) / 128) }
            }
            for exceptional: UInt16 in [0x7fc1, 0x7f80] {
                // Query0 uses this tile; query1 masks every key in it. The
                // original pair still performs 0 * NaN/Inf in query1's MMA.
                for col in 0..<512 { value.contents().assumingMemoryBound(to: UInt16.self)[7719 * 512 + col] = exceptional }
                guard let command = queue.makeCommandBuffer() else { throw EngineError("No compact fixture command") }
                let positions = try attention.encodeHostPositions(
                    command, selected: [[7719], [0], [32768]], keys: keys, ids: ids, counts: counts)
                var refused = false
                do {
                    try attention.encodeAttention(
                        command, query: query, key: key, value: value, positions: positions, output: actual,
                        scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: tooSmall))
                } catch is EngineError { refused = true }
                try require(refused, "Compact dispatch accepted an undersized scratch resource")
                try attention.encodeOriginalBatched(
                    command, query: query, key: key, value: value, ids: ids, counts: counts,
                    output: expected, scores: oldScratch, rows: rows, keys: keys, stride: 1)
                try attention.encodeAttention(
                    command, query: query, key: key, value: value, positions: positions, output: actual,
                    scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: scratch))
                command.commit()
                command.waitUntilCompleted()
                try require(command.status == .completed, "Compact native dispatch failed")
                let a = expected.contents().assumingMemoryBound(to: UInt32.self)
                let b = actual.contents().assumingMemoryBound(to: UInt32.self)
                for i in 0..<rows * 6144 { try require(a[i] == b[i], "Compact query-pair exceptional bits differ at \(i)") }
                for i in 6144..<12288 {
                    try require(Float(bitPattern: a[i]).isNaN, "Asymmetric value fixture no longer exercises masked0*nonfinite")
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Compact fixture retained an allocation charge")
    }
}
