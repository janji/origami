import Foundation
import Metal
import OrigamiEngine

/// Model-free bitwise checks of the 8K prefill-reuse path (and its exact stripe PV) against the original
/// batched kernels, nonpaged and paged, including masked-query NaN propagation and the fallback shape.
public enum PrefillAttentionReuseSelfChecks {
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError(message) }
    }
    public static func parity() throws { try parity(exact: false) }
    /// `exact`: the direct stripe PV (`qsa_exact_kernels`) against the original batched control, with the
    /// (255, 7681) shape kept on the current kernels and counted as a fallback.
    package static func parity(exact: Bool) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for prefill reuse check")
        }
        for (rows, keys) in [(256, 7680), (256, 4096), (256, 256), (255, 7681)] {
            let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
            try autoreleasepool {
                let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
                attention.qsaExactKernels = exact
                if exact { try attention.prepareExactKernels() }
                let lease = try ledger.reserve(bytes: 128 * 1024 * 1024, category: "prefill_reuse_fixture")
                defer { try? ledger.release(lease) }
                var allocated = 0
                func buffer(_ bytes: Int) throws -> MTLBuffer {
                    guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                        throw EngineError("Prefill reuse fixture allocation failed")
                    }
                    allocated += b.allocatedSize
                    try require(allocated <= 128 * 1024 * 1024, "Prefill reuse fixture exceeded its charge")
                    return b
                }
                let query = try buffer(rows * 6144 * 2)
                let key = try buffer(keys * 512 * 2)
                let value = try buffer(keys * 512 * 2)
                let ids = try buffer(rows * 2051 * 4)
                let counts = try buffer(rows * 4)
                let expected = try buffer(rows * 6144 * 4)
                let actual = try buffer(rows * 6144 * 4)
                let size = try PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys).scoreScratchBytes
                let controlScratch = try buffer(size)
                let scratch = try buffer(size)
                for (b, count) in [(query, rows * 6144), (key, keys * 512), (value, keys * 512)] {
                    let p = b.contents().assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count { p[i] = BF16.encode(Float((i * 13 + 3) % 127 - 63) / 128) }
                }
                let positions: [[UInt32]] = (0..<rows).map { row in
                    if row == 0 { return [0] }
                    if row == 1 { return [1] }
                    if row == 2 { return [] }
                    let end = keys - rows + row + 1
                    return Array(Swift.stride(from: 0, to: end, by: max(1, (end + 1999) / 2000)).prefix(2051)).map(UInt32.init)
                }
                let stride = max(1, positions.map(\.count).max()!)
                for exceptional in [false, true] {
                    if exceptional { value.contents().assumingMemoryBound(to: UInt16.self)[3] = 0x7fc1 }
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("No prefill reuse command") }
                    let ticket = try attention.encodeHostPositions(command, selected: positions, keys: keys, ids: ids, counts: counts)
                    try attention.encodeOriginalBatched(
                        command, query: query, key: key, value: value, ids: ids, counts: counts,
                        output: expected, scores: controlScratch, rows: rows, keys: keys, stride: stride)
                    try attention.encodeAttention(
                        command, query: query, key: key, value: value, positions: ticket, output: actual,
                        scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: scratch))
                    command.commit()
                    command.waitUntilCompleted()
                    try require(command.status == .completed, "Combined native prefill dispatch failed")
                    let a = expected.contents().assumingMemoryBound(to: UInt32.self)
                    let b = actual.contents().assumingMemoryBound(to: UInt32.self)
                    for i in 0..<rows * 6144 {
                        try require(a[i] == b[i], "Combined prefill differs at rows=\(rows) keys=\(keys) index=\(i)")
                    }
                    if exceptional { try require(Float(bitPattern: a[6144 + 3]).isNaN, "Fixture lost masked0*NaN pair propagation") }
                }
                try require(attention.prefillReuseDispatches == (rows == 256 ? 2 : 0), "Native reuse/fallback coverage is missing")
                try require(
                    attention.qsaExactPVTiles == (exact && rows == 256 ? 2 * 768 : 0)
                        && attention.qsaExactFallbacks == (exact && rows != 256 ? 2 : 0) && attention.qsaExactQKTiles == 0,
                    "Exact stripe PV path/fallback counters differ rows=\(rows) keys=\(keys) exact=\(exact)")
            }
            try require(ledger.snapshot().reserved == 0, "Prefill reuse check retained a charge")
        }
        try QSAPagedAttentionSelfChecks.prefillReuseParity(exact: exact)
    }
}
