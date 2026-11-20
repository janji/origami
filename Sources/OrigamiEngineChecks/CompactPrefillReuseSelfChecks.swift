import Foundation
import Metal
import OrigamiEngine

/// GPU checks of the fused compact prefill attention: layout bounds, the largest packed union, and bit-identity
/// against the original batched path on real shapes and exceptional values. `reports` feeds the checks' JSON.
package enum CompactPrefillReuseSelfChecks {
    nonisolated(unsafe) package static var reports: [[String: Any]] = []
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError(message) }
    }
    package static func layout() throws {
        for rows in 3...256 {
            for keys in [8192, 32768, 262144] {
                let p = try CompactPrefillReuse.layout(rows: rows, keys: keys)
                try require(
                    p.unionCountsOffset == p.compact.bytes && p.unionIDsOffset % 256 == 0
                        && p.unionIDsOffset >= p.unionCountsOffset + ((rows + 7) / 8) * 4
                        && p.bytes >= p.unionIDsOffset + ((rows + 7) / 8) * p.unionWordsPerGroup * 4
                        && p.bytes <= PinnedAttention.maximumScoreScratchBytes, "Compact union owners overlap or exceed 32 MiB")
            }
        }
        try require(
            try CompactPrefillReuse.layout(rows: 256, keys: 262144).bytes == 33_551_104,
            "Maximum 15-bit union packing changed its physical bound")
        try require(!RuntimeConfiguration().fusedSelectedPrefillAttention, "Split prefill must default off")
    }
    package static func maximumUnion() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for maximum compact union")
        }
        let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: true)
            let owner = try ledger.reserve(bytes: 50_000_000, category: "maximum_compact_union_fixture")
            defer { try? ledger.release(owner) }
            let rows = 256
            let keys = 262144
            let selected = 2051
            let layout = try CompactPrefillReuse.layout(rows: rows, keys: keys)
            guard let ids = device.makeBuffer(length: rows * selected * 4, options: .storageModeShared),
                let counts = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                let scratch = device.makeBuffer(length: layout.bytes, options: .storageModeShared),
                let command = queue.makeCommandBuffer()
            else { throw EngineError("Maximum compact union fixture allocation failed") }
            try require(
                ids.allocatedSize + counts.allocatedSize + scratch.allocatedSize <= 50_000_000,
                "Maximum union actual allocations exceeded the independent owner")
            let ip = ids.contents().assumingMemoryBound(to: UInt32.self)
            let cp = counts.contents().assumingMemoryBound(to: UInt32.self)
            for row in 0..<rows {
                cp[row] = UInt32(selected)
                let base = row / 8 == 31 ? 16360 : 0
                for i in 0..<selected { ip[row * selected + i] = UInt32((base + (row % 8) * selected + i) * 8) }
            }
            _ = try attention.encodeCompactUnion(
                command, ids: ids, counts: counts, scores: scratch,
                rows: rows, keys: keys, stride: selected)
            command.commit()
            command.waitUntilCompleted()
            try require(command.status == .completed, "Maximum compact union command failed")
            let packed = scratch.contents().advanced(by: layout.unionIDsOffset).assumingMemoryBound(to: UInt32.self)
            let amounts = scratch.contents().advanced(by: layout.unionCountsOffset).assumingMemoryBound(to: UInt32.self)
            for group in 0..<32 {
                try require(amounts[group] == 16408, "Maximum union lost or duplicated selected tiles")
                for i in 0..<16408 {
                    let bit = i * 15
                    let word = group * layout.unionWordsPerGroup + bit / 32
                    let shift = bit % 32
                    var tile = packed[word] >> shift
                    if shift > 17 { tile |= packed[word + 1] << (32 - shift) }
                    try require(
                        (tile & 32767) == UInt32((group == 31 ? 16360 : 0) + i),
                        "Maximum packed union ID differs group=\(group) index=\(i)")
                }
            }
            let path = attention.prefillPathSnapshot
            try require(
                path.compactUnionGroupsCompleted == 32 && path.compactUnionTiles == 525056
                    && path.compactSelectedTiles == 525056 && path.compactQKGroupsCompleted == 0
                    && path.compactSoftmaxGroupsCompleted == 0 && path.compactPVGroupsCompleted == 0,
                "Maximum union reported an unexecuted stage")
        }
        try require(ledger.snapshot().reserved == 0, "Maximum compact union leaked its physical owner")
    }
    package static func actualShapes() throws { try actualShapes(grouped: false) }
    /// `exact` runs the same shapes with `qsa_exact_kernels` (double-buffered compact QK, direct compact PV)
    /// against the original batched control and requires the exact path counters.
    package static func actualShapes(grouped: Bool, exact: Bool = false) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for fused prefill check")
        }
        reports = []
        // The last domain bound exercises the changed path at the actual max,
        // without claiming a full-model maximum-context qualification.
        let shapes = [(256, 32768), (255, 32767), (3, 32769), (3, 262144), (3, 262143), (256, 7678)]
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: true)
            attention.compactPVGrouped = grouped
            attention.qsaExactKernels = exact
            if exact { try attention.prepareExactKernels() }
            for (rows, keys) in shapes {
                try autoreleasepool {
                    let lease = try ledger.reserve(bytes: 800_000_000, category: "compact_prefill_reuse_actual_shape_fixture")
                    defer { try? ledger.release(lease) }
                    var allocated = 0
                    func buffer(_ bytes: Int) throws -> MTLBuffer {
                        guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                            throw EngineError("Compact prefill reuse fixture allocation failed")
                        }
                        allocated += b.allocatedSize
                        try require(allocated <= 800_000_000, "Compact prefill reuse fixture exceeded charged owner")
                        return b
                    }
                    let query = try buffer(rows * 6144 * 2)
                    let key = try buffer(keys * 512 * 2)
                    let value = try buffer(keys * 512 * 2)
                    let ids = try buffer(rows * 2051 * 4)
                    let counts = try buffer(rows * 4)
                    let expected = try buffer(rows * 6144 * 4)
                    let actual = try buffer(rows * 6144 * 4)
                    let oldScratch = try buffer(PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys).scoreScratchBytes)
                    let scratch = try buffer(attention.bufferScratchPlan(rows: rows, keys: keys).scoreBytes)
                    for (b, count) in [(query, rows * 6144), (key, keys * 512), (value, keys * 512)] {
                        let p = b.contents().assumingMemoryBound(to: UInt16.self)
                        for i in 0..<count { p[i] = BF16.encode(Float((i * 13 + 3) % 127 - 63) / 128) }
                    }
                    let positions: [[UInt32]] = (0..<rows).map { row in
                        let end = keys - rows + row + 1
                        if row == 0 { return [UInt32(end - 1)] }
                        // 2051 unique keys with explicit context and page tails.
                        return (0..<min(2051, end)).map { UInt32($0 * (end - 1) / (min(2051, end) - 1)) }
                    }
                    let stride = positions.map(\.count).max()!
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("No fused prefill fixture command") }
                    _ = try attention.encodeHostPositions(command, selected: positions, keys: keys, ids: ids, counts: counts)
                    try attention.encodeOriginalBatched(
                        command, query: query, key: key, value: value, ids: ids, counts: counts,
                        output: expected, scores: oldScratch, rows: rows, keys: keys, stride: stride)
                    command.commit()
                    command.waitUntilCompleted()
                    try require(command.status == .completed, "Original compact reuse control failed")
                    let originalSeconds = command.gpuEndTime - command.gpuStartTime
                    guard let candidate = queue.makeCommandBuffer() else { throw EngineError("No compact reuse candidate command") }
                    let selected = try attention.encodeHostPositions(candidate, selected: positions, keys: keys, ids: ids, counts: counts)
                    let before = attention.prefillPathSnapshot
                    let exactBefore = attention.qsaExactKernelCounts
                    try attention.encodeAttention(
                        candidate, query: query, key: key, value: value, positions: selected, output: actual,
                        scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: scratch))
                    candidate.commit()
                    candidate.waitUntilCompleted()
                    try require(
                        candidate.status == .completed, "Compact reuse actual-shape command failed: \(String(describing: candidate.error))")
                    let a = expected.contents().assumingMemoryBound(to: UInt32.self)
                    let b = actual.contents().assumingMemoryBound(to: UInt32.self)
                    var firstDifference: Int?
                    var differences = 0
                    for i in 0..<rows * 6144 where a[i] != b[i] {
                        differences += 1
                        if firstDifference == nil { firstDifference = i }
                    }
                    try require(
                        differences == 0,
                        "Compact prefill reuse bits differ rows=\(rows) keys=\(keys) count=\(differences) first=\(firstDifference ?? -1) expected=\(a[firstDifference ?? 0]) actual=\(b[firstDifference ?? 0])"
                    )
                    let after = attention.prefillPathSnapshot
                    let exactAfter = attention.qsaExactKernelCounts
                    reports.append([
                        "rows": rows, "keys": keys, "groupedPV": grouped, "exactKernels": exact, "comparisons": rows * 6144,
                        "bitDifferences": differences,
                        "qsaExactPVTiles": exactAfter.pvTiles - exactBefore.pvTiles,
                        "qsaExactQKTiles": exactAfter.qkTiles - exactBefore.qkTiles,
                        "qsaExactFallbacks": exactAfter.fallbacks - exactBefore.fallbacks,
                        "compactQKGroups": after.compactQKGroupsCompleted - before.compactQKGroupsCompleted,
                        "compactQKSkippedGroups": after.compactQKGroupsSkipped - before.compactQKGroupsSkipped,
                        "compactSoftmaxGroups": after.compactSoftmaxGroupsCompleted - before.compactSoftmaxGroupsCompleted,
                        "compactPVGroups": after.compactPVGroupsCompleted - before.compactPVGroupsCompleted,
                        "compactUnionGroups": after.compactUnionGroupsCompleted - before.compactUnionGroupsCompleted,
                        "compactUnionTiles": after.compactUnionTiles - before.compactUnionTiles,
                        "compactSelectedTiles": after.compactSelectedTiles - before.compactSelectedTiles,
                        "allocatedBytes": allocated, "originalGPUSeconds": originalSeconds,
                        "candidateGPUSeconds": candidate.gpuEndTime - candidate.gpuStartTime,
                        "scope": "One ordered original/candidate primitive pair; not qualified performance",
                    ])
                    let selectedTiles = positions.reduce(0) { $0 + Set($1.map { $0 / 8 }).count }
                    let unionTiles = Swift.stride(from: 0, to: rows, by: 8).reduce(0) { total, first in
                        total + Set(positions[first..<min(first + 8, rows)].flatMap { $0.map { $0 / 8 } }).count
                    }
                    let layout = try CompactPrefillReuse.layout(rows: rows, keys: keys)
                    try require(
                        after.compactUnionGroupsCompleted - before.compactUnionGroupsCompleted == UInt64((rows + 7) / 8)
                            && after.compactUnionTiles - before.compactUnionTiles == UInt64(unionTiles)
                            && after.compactSelectedTiles - before.compactSelectedTiles == UInt64(selectedTiles),
                        "GPU union compaction differs from independent selected-index sets")
                    try require(
                        after.compactReuseDispatches - before.compactReuseDispatches == 1
                            && after.compactQKGroupsCompleted > before.compactQKGroupsCompleted
                            && after.compactQKGroupsCompleted + after.compactQKGroupsSkipped
                                - before.compactQKGroupsCompleted - before.compactQKGroupsSkipped
                                == UInt64(((layout.maximumUnionTiles + 15) / 16) * ((rows + 7) / 8) * 2)
                            && after.compactSoftmaxGroupsCompleted - before.compactSoftmaxGroupsCompleted == UInt64(rows * 24)
                            && after.compactPVGroupsCompleted - before.compactPVGroupsCompleted == UInt64(((rows + 1) / 2) * 6),
                        "Finite fixture bypassed compact QK/softmax/paired PV")
                    let exactQK = exact ? UInt64(((layout.maximumUnionTiles + 15) / 16) * ((rows + 7) / 8) * 2) : 0
                    try require(
                        exactAfter.pvTiles - exactBefore.pvTiles == (exact ? UInt64(((rows + 1) / 2) * 6) : 0)
                            && exactAfter.qkTiles - exactBefore.qkTiles == exactQK
                            && exactAfter.fallbacks == exactBefore.fallbacks,
                        "Exact QSA kernel path counters differ rows=\(rows) keys=\(keys) exact=\(exact)")
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Compact prefill reuse fixture retained an allocation charge")
    }
    package static func exceptionalRepair() throws { try exceptionalRepair(grouped: false) }
    package static func exceptionalRepair(grouped: Bool, exact: Bool = false) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for fused exceptional check")
        }
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: true)
            attention.compactPVGrouped = grouped
            attention.qsaExactKernels = exact
            if exact { try attention.prepareExactKernels() }
            let lease = try ledger.reserve(bytes: 100_000_000, category: "compact_prefill_reuse_exception_fixture")
            defer { try? ledger.release(lease) }
            let rows = 3
            let keys = 32769
            var admitted = 0
            func buffer(_ bytes: Int) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw EngineError("Compact prefill reuse exception allocation failed")
                }
                admitted += b.allocatedSize
                try require(admitted <= 100_000_000, "Compact prefill reuse exception exceeded admission")
                return b
            }
            let query = try buffer(rows * 6144 * 2)
            let key = try buffer(keys * 512 * 2)
            let value = try buffer(keys * 512 * 2)
            let ids = try buffer(rows * 4)
            let counts = try buffer(rows * 4)
            let expected = try buffer(rows * 6144 * 4)
            let actual = try buffer(rows * 6144 * 4)
            let oldScratch = try buffer(PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys).scoreScratchBytes)
            let scratch = try buffer(attention.bufferScratchPlan(rows: rows, keys: keys).scoreBytes)
            for (b, count) in [(query, rows * 6144), (key, keys * 512), (value, keys * 512)] {
                let p = b.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count { p[i] = BF16.encode(Float((i * 13 + 3) % 127 - 63) / 128) }
            }
            for exceptional: UInt16 in [0, 0x7fc1, 0x7f80] {
                for col in 0..<512 { value.contents().assumingMemoryBound(to: UInt16.self)[7719 * 512 + col] = exceptional }
                let positions: [[UInt32]] = exceptional == 0 ? [[], [0], [32768]] : [[7719], [0], [32768]]
                guard let command = queue.makeCommandBuffer() else { throw EngineError("No fused exception command") }
                let ticket = try attention.encodeHostPositions(command, selected: positions, keys: keys, ids: ids, counts: counts)
                try attention.encodeOriginalBatched(
                    command, query: query, key: key, value: value, ids: ids, counts: counts,
                    output: expected, scores: oldScratch, rows: rows, keys: keys, stride: 1)
                let before = attention.prefillPathSnapshot
                let exactBefore = attention.qsaExactKernelCounts
                try attention.encodeAttention(
                    command, query: query, key: key, value: value, positions: ticket, output: actual,
                    scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: scratch))
                command.commit()
                command.waitUntilCompleted()
                try require(command.status == .completed, "Compact prefill reuse exception repair command failed")
                let a = expected.contents().assumingMemoryBound(to: UInt32.self)
                let b = actual.contents().assumingMemoryBound(to: UInt32.self)
                for i in 0..<rows * 6144 {
                    try require(a[i] == b[i], "Compact prefill reuse exception repair differs at \(i) case=\(exceptional)")
                }
                let after = attention.prefillPathSnapshot
                let exactAfter = attention.qsaExactKernelCounts
                try require(
                    after.compactReuseDispatches - before.compactReuseDispatches == 1
                        && after.compactSoftmaxGroupsCompleted - before.compactSoftmaxGroupsCompleted == UInt64(rows * 24)
                        && after.compactPVGroupsCompleted - before.compactPVGroupsCompleted == UInt64(((rows + 1) / 2) * 6),
                    "Exceptional fixture bypassed compact exact semantics")
                try require(
                    exactAfter.pvTiles - exactBefore.pvTiles == (exact ? UInt64(((rows + 1) / 2) * 6) : 0)
                        && (exactAfter.qkTiles > exactBefore.qkTiles) == exact && exactAfter.fallbacks == exactBefore.fallbacks,
                    "Exceptional fixture exact QSA kernel path counters differ exact=\(exact)")
            }
        }
        try require(ledger.snapshot().reserved == 0, "Compact prefill reuse exception retained an owner")
    }
}
