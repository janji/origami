import Foundation
import Metal
import OrigamiEngine

/// Standalone exactness of the `qsa_exact_kernels` variants, model-free: the engine's own pipelines run
/// the current kernels and the exact variants from the same host positions on synthetic selections
/// shaped like real prompts ("calibrated") and denser ones ("heavy"), with and without a NaN/Inf/-0 seed
/// set, comparing the FP32 outputs, the compact score bytes after QK (before the in-place softmax) and
/// the compact probability bytes after the whole pipeline byte for byte, plus the original batched
/// control on clean inputs and the paged library's forms at the production shapes. The GPU times in
/// `reports` are single runs, not performance evidence.
package enum QSAExactKernelSelfChecks {
    nonisolated(unsafe) package static var reports: [[String: Any]] = []
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError(message) }
    }

    // splitmix64: deterministic inputs and selections for every run.
    private struct RNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func below(_ n: Int) -> Int { n <= 1 ? 0 : Int(next() % UInt64(n)) }
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    }
    private static func bf16(_ x: Float) -> UInt16 {
        let b = x.bitPattern
        if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 { return UInt16((b | 0x0040_0000) >> 16) }
        return UInt16((b &+ 0x7fff &+ ((b >> 16) & 1)) >> 16)
    }
    /// Per query 512 four-key blocks (a per-8-query-group core plus private blocks) and the causal tail;
    /// dense when keys <= 2048 or fewer than 512 blocks exist. Blocks 1 and 2 (keys 4-11) are always in
    /// the core so the K/V seeds at keys 5, 7, 9 and 10 are selected by every query.
    static func selections(rows: Int, keys: Int, heavy: Bool, seed: UInt64) -> [[UInt32]] {
        var result: [[UInt32]] = []
        let corePairs = heavy ? 50 : 154
        let coreSize = heavy ? 200 : 455
        for group in stride(from: 0, to: rows, by: 8) {
            var rng = RNG(state: seed &+ UInt64(group) &* 7919 &+ UInt64(keys))
            let groupRows = min(8, rows - group)
            let pMin = keys - rows + group
            let limitMin = (pMin + 1) / 4
            var core = Set<Int>()
            if keys > 2048 && limitMin > 512 {
                core.insert(1)
                core.insert(2)
                var attempts = 0
                while core.count < 2 * corePairs + 2 && attempts < 100_000 {
                    attempts += 1
                    let b = rng.below(limitMin - 1)
                    if !core.contains(b) && !core.contains(b + 1) {
                        core.insert(b)
                        core.insert(b + 1)
                    }
                }
                while core.count < coreSize { core.insert(rng.below(limitMin)) }
            }
            for r in 0..<groupRows {
                let p = keys - rows + group + r
                var ids: [UInt32]
                if keys <= 2048 {
                    ids = (0...p).map(UInt32.init)
                } else {
                    let available = (p + 1) / 4
                    var blocks: Set<Int>
                    if available <= 512 {
                        blocks = Set(0..<available)
                    } else {
                        blocks = core.count > 0 ? core : Set([1, 2])
                        while blocks.count < 512 { blocks.insert(rng.below(available)) }
                    }
                    ids = []
                    for b in blocks.sorted() { for j in 0..<4 { ids.append(UInt32(b * 4 + j)) } }
                    let tail = (p + 1) / 4 * 4
                    if tail <= p { for k in tail...p { ids.append(UInt32(k)) } }
                }
                result.append(ids)
            }
        }
        return result
    }
    private static func fill(_ p: UnsafeMutablePointer<UInt16>, count: Int, seed: UInt64, scale: Float) {
        var rng = RNG(state: seed)
        for i in 0..<count { p[i] = bf16((rng.unit() * 2 - 1) * scale) }
    }
    /// The inputs: Q (256 rows), K and V (32768 keys; smaller shapes use the prefix), with the
    /// seed set when `seeded`: Q[3] NaN, Q[6145] +Inf, Q[2] -0; K key 5 NaN (both KV heads), key 6 -0;
    /// V key 7 +Inf, key 9 NaN, key 10 -0.
    private static func inputs(
        query: UnsafeMutablePointer<UInt16>, key: UnsafeMutablePointer<UInt16>,
        value: UnsafeMutablePointer<UInt16>, maxKeys: Int, seeded: Bool
    ) {
        fill(query, count: 256 * 6144, seed: 11, scale: 2.0)
        fill(key, count: maxKeys * 512, seed: 13, scale: 2.0)
        fill(value, count: maxKeys * 512, seed: 17, scale: 4.0)
        guard seeded else { return }
        query[3] = bf16(.nan)
        query[6144 + 1] = bf16(.infinity)
        query[2] = 0x8000
        key[5 * 512 + 7] = bf16(.nan)
        key[5 * 512 + 256 + 9] = bf16(.nan)
        key[6 * 512 + 1] = 0x8000
        value[7 * 512 + 3] = bf16(.infinity)
        value[9 * 512 + 300] = bf16(.nan)
        value[10 * 512 + 100] = 0x8000
    }
    private static func firstDifference(_ a: UnsafeRawPointer, _ b: UnsafeRawPointer, bytes: Int) -> (index: Int, differing: Int)? {
        let x = a.assumingMemoryBound(to: UInt8.self)
        let y = b.assumingMemoryBound(to: UInt8.self)
        var first = -1
        var differing = 0
        for i in 0..<bytes where x[i] != y[i] {
            differing += 1
            if first < 0 { first = i }
        }
        return first < 0 ? nil : (first, differing)
    }
    /// The compact region with each row's prepare bucket lists canonicalized: `origami_compact_prepare` appends a
    /// bucket's entries (`meta[1025 + index]`) with a threadgroup atomic, so their order inside a bucket differs
    /// from run to run by construction, in the control too. Every entry is sorted within its bucket's bounds
    /// (`meta[b] ..< meta[b + 1]`); every other byte, the bucket bounds included, is compared as written.
    private static func compactDifference(_ a: UnsafeRawPointer, _ b: UnsafeRawPointer, rows: Int, bytes: Int) -> (
        index: Int, differing: Int
    )? {
        func canonical(_ p: UnsafeRawPointer) -> [UInt8] {
            var copy = [UInt8](UnsafeRawBufferPointer(start: p, count: bytes))
            copy.withUnsafeMutableBytes { raw in
                let words = raw.bindMemory(to: UInt32.self)
                for row in 0..<rows {
                    let meta = row * 7179
                    for bucket in 0..<1024 {
                        let lo = min(Int(words[meta + bucket]), 2051)
                        let hi = min(Int(words[meta + bucket + 1]), 2051)
                        if lo < hi { words[(meta + 1025 + lo)..<(meta + 1025 + hi)].sort() }
                    }
                }
            }
            return copy
        }
        let x = canonical(a)
        let y = canonical(b)
        return x.withUnsafeBytes { xr in y.withUnsafeBytes { yr in firstDifference(xr.baseAddress!, yr.baseAddress!, bytes: bytes) } }
    }
    private static func sameOutputs(_ a: MTLBuffer, _ b: MTLBuffer, rows: Int, _ what: String) throws {
        let x = a.contents().assumingMemoryBound(to: UInt32.self)
        let y = b.contents().assumingMemoryBound(to: UInt32.self)
        var first: Int?
        var differing = 0
        for i in 0..<rows * 6144 where x[i] != y[i] {
            differing += 1
            if first == nil { first = i }
        }
        guard let first else { return }
        throw EngineError(
            String(
                format: "%@: element %d (row %d head %d col %d) %08x/%08x, %d differing", what,
                first, first / 6144, (first % 6144) / 256, first % 256, x[first], y[first], differing))
    }
    private struct Fixture {
        let device: MTLDevice, queue: MTLCommandQueue
        var allocated = 0
        mutating func buffer(_ bytes: Int, _ limit: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw EngineError("Exact QSA kernel fixture allocation failed")
            }
            allocated += b.allocatedSize
            guard allocated <= limit else { throw EngineError("Exact QSA kernel fixture exceeded its charged owner") }
            return b
        }
        /// One command: host positions, then the attention path under test.
        func run(
            _ attention: PinnedAttention, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, ids: MTLBuffer, counts: MTLBuffer,
            positions: [[UInt32]], keys: Int, output: MTLBuffer, scratch: MTLBuffer, original: Bool = false
        ) throws -> Double {
            guard let command = queue.makeCommandBuffer() else { throw EngineError("Exact QSA kernel fixture command unavailable") }
            let rows = positions.count
            let stride = max(1, positions.map(\.count).max() ?? 1)
            let ticket = try attention.encodeHostPositions(command, selected: positions, keys: keys, ids: ids, counts: counts)
            if original {
                try attention.encodeOriginalBatched(
                    command, query: query, key: key, value: value, ids: ids, counts: counts,
                    output: output, scores: scratch, rows: rows, keys: keys, stride: stride)
            } else {
                try attention.encodeAttention(
                    command, query: query, key: key, value: value, positions: ticket, output: output,
                    scratch: .init(intermediate: nil, sums: nil, maxima: nil, scores: scratch))
            }
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw EngineError.gpu("Exact QSA kernel fixture command failed", command.error) }
            return command.gpuEndTime - command.gpuStartTime
        }
    }

    /// 32K path (compact reuse): `origami_compact_qk_reuse_dbuf` and `origami_pv_direct_compact_c64` against
    /// `origami_compact_qk_reuse` and the grouped compact PV from the same host positions; rows 256 at keys
    /// 32768 (calibrated and heavy), 16384, 8192, 4096, 2048, plus rows 253 at 32768 and 11 at 8192;
    /// seeded and clean inputs. Bytes compared: compact scores after QK, fp32 outputs and the compact
    /// probability/default/metadata region after the whole pipeline; clean outputs also against the
    /// original batched control. The exact path counters must equal the dispatched geometry.
    package static func compactExactness() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for the exact QSA kernel check")
        }
        reports = []
        let shapes: [(rows: Int, keys: Int, heavy: Bool)] = [
            (256, 32768, false), (256, 32768, true), (256, 16384, false),
            (256, 8192, false), (256, 4096, false), (256, 2048, false), (253, 32768, false), (11, 8192, false),
        ]
        let limit = 1_000_000_000
        let maxKeys = 32768
        let ledger = try MemoryLedger(ceiling: 1_200_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let control = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: true)
            control.compactPVGrouped = true
            let variant = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: true)
            variant.compactPVGrouped = true
            variant.qsaExactKernels = true
            try variant.prepareExactKernels()
            let lease = try ledger.reserve(bytes: UInt64(limit), category: "qsa_exact_kernel_compact_fixture")
            defer { try? ledger.release(lease) }
            var fixture = Fixture(device: device, queue: queue)
            let query = try fixture.buffer(256 * 6144 * 2, limit)
            let key = try fixture.buffer(maxKeys * 512 * 2, limit)
            let value = try fixture.buffer(maxKeys * 512 * 2, limit)
            let ids = try fixture.buffer(256 * 2051 * 4, limit)
            let counts = try fixture.buffer(256 * 4, limit)
            let controlOut = try fixture.buffer(256 * 6144 * 4, limit)
            let variantOut = try fixture.buffer(256 * 6144 * 4, limit)
            let originalOut = try fixture.buffer(256 * 6144 * 4, limit)
            let controlScratch = try fixture.buffer(PinnedAttention.maximumScoreScratchBytes, limit)
            let variantScratch = try fixture.buffer(PinnedAttention.maximumScoreScratchBytes, limit)
            let originalScratch = try fixture.buffer(PinnedAttention.maximumScoreScratchBytes, limit)
            for seeded in [true, false] {
                inputs(
                    query: query.contents().assumingMemoryBound(to: UInt16.self), key: key.contents().assumingMemoryBound(to: UInt16.self),
                    value: value.contents().assumingMemoryBound(to: UInt16.self), maxKeys: maxKeys, seeded: seeded)
                for shape in shapes {
                    try autoreleasepool {
                        let rows = shape.rows
                        let keys = shape.keys
                        let layout = try CompactPrefillReuse.layout(rows: rows, keys: keys)
                        let positions = selections(rows: rows, keys: keys, heavy: shape.heavy, seed: 101)
                        let label = "rows=\(rows) keys=\(keys) \(shape.heavy ? "heavy" : "calibrated") \(seeded ? "seeded" : "clean")"
                        let before = variant.prefillPathSnapshot
                        let controlBefore = control.prefillPathSnapshot
                        let exactBefore = variant.qsaExactKernelCounts
                        let controlExactBefore = control.qsaExactKernelCounts
                        // Stage 1: QK only, from identical 0xA5-filled scratch; the compact region holds the
                        // metadata (prepare) and the selected scores (QK) in the same addresses for both.
                        for (attention, scratch, out) in [(control, controlScratch, controlOut), (variant, variantScratch, variantOut)] {
                            memset(scratch.contents(), 0xA5, layout.bytes)
                            attention.diagnosticCompactReuseQKOnly = true
                            defer { attention.diagnosticCompactReuseQKOnly = false }
                            _ = try fixture.run(
                                attention, query: query, key: key, value: value, ids: ids, counts: counts,
                                positions: positions, keys: keys, output: out, scratch: scratch)
                        }
                        if let d = compactDifference(
                            controlScratch.contents(), variantScratch.contents(), rows: rows, bytes: layout.compact.bytes)
                        {
                            throw EngineError(
                                "Double-buffered compact QK scores differ \(label): byte \(d.index) (\(d.differing) differing)")
                        }
                        // Stage 2: the whole pipeline; outputs and the post-softmax compact region.
                        memset(controlScratch.contents(), 0xA5, layout.bytes)
                        memset(variantScratch.contents(), 0xA5, layout.bytes)
                        memset(controlOut.contents(), 0xA5, rows * 6144 * 4)
                        memset(variantOut.contents(), 0xA5, rows * 6144 * 4)
                        let controlSeconds = try fixture.run(
                            control, query: query, key: key, value: value, ids: ids, counts: counts,
                            positions: positions, keys: keys, output: controlOut, scratch: controlScratch)
                        let variantSeconds = try fixture.run(
                            variant, query: query, key: key, value: value, ids: ids, counts: counts,
                            positions: positions, keys: keys, output: variantOut, scratch: variantScratch)
                        try sameOutputs(
                            controlOut, variantOut, rows: rows, "Exact compact kernels differ from the current kernels \(label)")
                        if let d = compactDifference(
                            controlScratch.contents(), variantScratch.contents(), rows: rows, bytes: layout.compact.bytes)
                        {
                            throw EngineError(
                                "Compact probability bytes differ after the exact pipeline \(label): byte \(d.index) (\(d.differing) differing)"
                            )
                        }
                        var originalSeconds: Double?
                        if !seeded {
                            memset(originalOut.contents(), 0xA5, rows * 6144 * 4)
                            originalSeconds = try fixture.run(
                                control, query: query, key: key, value: value, ids: ids, counts: counts,
                                positions: positions, keys: keys, output: originalOut, scratch: originalScratch, original: true)
                            try sameOutputs(
                                originalOut, variantOut, rows: rows,
                                "Exact compact kernels differ from the original batched control \(label)")
                        }
                        let after = variant.prefillPathSnapshot
                        let controlAfter = control.prefillPathSnapshot
                        let exactAfter = variant.qsaExactKernelCounts
                        let controlExactAfter = control.qsaExactKernelCounts
                        let units = UInt64(((rows + 1) / 2) * 6)
                        let qkGroups = UInt64(((layout.maximumUnionTiles + 15) / 16) * ((rows + 7) / 8) * 2)
                        try require(
                            exactAfter.qkTiles - exactBefore.qkTiles == 2 * qkGroups
                                && exactAfter.pvTiles - exactBefore.pvTiles == units
                                && exactAfter.fallbacks == exactBefore.fallbacks
                                && after.compactPVGroupsCompleted - before.compactPVGroupsCompleted == units
                                && after.compactQKGroupsCompleted + after.compactQKGroupsSkipped - before.compactQKGroupsCompleted
                                    - before.compactQKGroupsSkipped == 2 * qkGroups
                                && after.compactReuseDispatches - before.compactReuseDispatches == 2,
                            "Exact compact path counters differ from the dispatched geometry \(label)")
                        try require(
                            controlExactAfter.qkTiles == 0 && controlExactAfter.pvTiles == 0 && controlExactAfter.fallbacks == 0
                                && controlExactBefore.pvTiles == 0
                                && controlAfter.compactPVGroupsCompleted - controlBefore.compactPVGroupsCompleted == units,
                            "Control attention reported exact-kernel work \(label)")
                        reports.append([
                            "path": "compact", "rows": rows, "keys": keys, "selection": shape.heavy ? "heavy" : "calibrated",
                            "inputs": seeded ? "seeded" : "clean", "comparisons": rows * 6144, "compactBytes": layout.compact.bytes,
                            "qkGroups": qkGroups, "pvUnits": units, "controlGPUSeconds": controlSeconds,
                            "variantGPUSeconds": variantSeconds,
                            "originalGPUSeconds": originalSeconds.map { $0 as Any } ?? NSNull(),
                            "scope": "One current/variant pair per stage; byte identity only, not qualified performance",
                        ])
                    }
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Exact compact kernel fixture retained an allocation charge")
    }

    /// 8K path (prefill reuse): `origami_pv_direct_stripe_c64` against `origami_pv_full_width` and the original
    /// batched control, rows 256 at keys 7680, 4096, 2048, 1024 on seeded and clean inputs; (256, 7681)
    /// and (255, 7680) stay on the current kernels and must be counted as fallbacks.
    package static func stripeExactness() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device for the exact stripe PV check")
        }
        let shapes: [(rows: Int, keys: Int)] = [(256, 7680), (256, 4096), (256, 2048), (256, 1024), (256, 7681), (255, 7680)]
        let limit = 400_000_000
        let maxKeys = 8192
        let ledger = try MemoryLedger(ceiling: 600_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let control = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            let variant = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            variant.qsaExactKernels = true
            try variant.prepareExactKernels()
            let lease = try ledger.reserve(bytes: UInt64(limit), category: "qsa_exact_kernel_stripe_fixture")
            defer { try? ledger.release(lease) }
            var fixture = Fixture(device: device, queue: queue)
            let query = try fixture.buffer(256 * 6144 * 2, limit)
            let key = try fixture.buffer(maxKeys * 512 * 2, limit)
            let value = try fixture.buffer(maxKeys * 512 * 2, limit)
            let ids = try fixture.buffer(256 * 2051 * 4, limit)
            let counts = try fixture.buffer(256 * 4, limit)
            let controlOut = try fixture.buffer(256 * 6144 * 4, limit)
            let variantOut = try fixture.buffer(256 * 6144 * 4, limit)
            let originalOut = try fixture.buffer(256 * 6144 * 4, limit)
            let scratch = try (0..<3).map { _ in try fixture.buffer(PinnedAttention.maximumScoreScratchBytes, limit) }
            for seeded in [true, false] {
                inputs(
                    query: query.contents().assumingMemoryBound(to: UInt16.self), key: key.contents().assumingMemoryBound(to: UInt16.self),
                    value: value.contents().assumingMemoryBound(to: UInt16.self), maxKeys: maxKeys, seeded: seeded)
                for shape in shapes {
                    try autoreleasepool {
                        let rows = shape.rows
                        let keys = shape.keys
                        let served = rows == 256 && keys <= 7680
                        let positions = selections(rows: rows, keys: keys, heavy: false, seed: 101)
                        let label = "rows=\(rows) keys=\(keys) \(seeded ? "seeded" : "clean")"
                        let before = variant.qsaExactKernelCounts
                        for out in [controlOut, variantOut, originalOut] { memset(out.contents(), 0xA5, rows * 6144 * 4) }
                        let controlSeconds = try fixture.run(
                            control, query: query, key: key, value: value, ids: ids, counts: counts,
                            positions: positions, keys: keys, output: controlOut, scratch: scratch[0])
                        let variantSeconds = try fixture.run(
                            variant, query: query, key: key, value: value, ids: ids, counts: counts,
                            positions: positions, keys: keys, output: variantOut, scratch: scratch[1])
                        try sameOutputs(controlOut, variantOut, rows: rows, "Direct stripe PV differs from the full-width PV \(label)")
                        var originalSeconds: Double?
                        if !seeded {
                            originalSeconds = try fixture.run(
                                control, query: query, key: key, value: value, ids: ids, counts: counts,
                                positions: positions, keys: keys, output: originalOut, scratch: scratch[2], original: true)
                            try sameOutputs(
                                originalOut, variantOut, rows: rows, "Direct stripe PV differs from the original batched control \(label)")
                        }
                        let after = variant.qsaExactKernelCounts
                        try require(
                            after.pvTiles - before.pvTiles == (served ? 768 : 0)
                                && after.fallbacks - before.fallbacks == (served ? 0 : 1) && after.qkTiles == 0
                                && control.qsaExactKernelCounts.pvTiles == 0 && control.qsaExactKernelCounts.fallbacks == 0,
                            "Exact stripe path/fallback counters differ \(label)")
                        reports.append([
                            "path": "stripe", "rows": rows, "keys": keys, "selection": "calibrated", "inputs": seeded ? "seeded" : "clean",
                            "comparisons": rows * 6144, "served": served, "pvUnits": served ? 768 : 0, "fallbacks": served ? 0 : 1,
                            "controlGPUSeconds": controlSeconds, "variantGPUSeconds": variantSeconds,
                            "originalGPUSeconds": originalSeconds.map { $0 as Any } ?? NSNull(),
                            "scope": "One current/variant pair; byte identity only, not qualified performance",
                        ])
                    }
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Exact stripe kernel fixture retained an allocation charge")
    }

    /// The paged library (persistent-conversation prefill reads prefix pages through the argument table):
    /// the exact kernels' page-row V loads and K bits must reproduce the nonpaged current kernels at the
    /// production shapes, seeded and clean: compact reuse at (256, 32768) and (11, 8192), the prefill
    /// reuse stripe at (256, 7680). The nonpaged exact form is compared as well (three-way identity).
    package static func pagedExactness() throws {
        let cases: [(fused: Bool, rows: Int, keys: Int)] = [(true, 256, 32768), (true, 11, 8192), (false, 256, 7680)]
        for (fused, rows, keys) in cases {
            for seeded in [true, false] {
                try autoreleasepool {
                    let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 16_000_000)
                    let scratch = try ledger.reserve(
                        bytes: 2 * UInt64(keys) * 512 * 4 + 64_000_000, category: "qsa_exact_kernel_paged_fixture")
                    defer { try? ledger.release(scratch) }
                    let label = "fused=\(fused) rows=\(rows) keys=\(keys) \(seeded ? "seeded" : "clean")"
                    var queryBits = [UInt16](repeating: 0, count: 256 * 6144)
                    var keyBits = [UInt16](repeating: 0, count: keys * 512)
                    var valueBits = [UInt16](repeating: 0, count: keys * 512)
                    queryBits.withUnsafeMutableBufferPointer { q in
                        keyBits.withUnsafeMutableBufferPointer { k in
                            valueBits.withUnsafeMutableBufferPointer { v in
                                inputs(query: q.baseAddress!, key: k.baseAddress!, value: v.baseAddress!, maxKeys: keys, seeded: seeded)
                            }
                        }
                    }
                    let queries = queryBits.prefix(rows * 6144).map(BF16.decode)
                    for (index, bits) in queryBits.prefix(rows * 6144).enumerated() where BF16.encode(queries[index]) != bits {
                        throw EngineError("Host query bits do not round-trip the BF16 encoder at \(index)")
                    }
                    let positions = selections(rows: rows, keys: keys, heavy: false, seed: 101)
                    let old = keys - rows
                    let oldKeys = Array(keyBits[0..<old * 512])
                    let oldValues = Array(valueBits[0..<old * 512])
                    let newKeys = Array(keyBits[(old * 512)...])
                    let newValues = Array(valueBits[(old * 512)...])
                    let arena = try QSAPageArena(ledger: ledger, maximumBytes: UInt64(keys) * 2112 + 16_777_216)
                    var state = QSAState(arena: arena)
                    for start in stride(from: 0, to: old, by: 256) {
                        let count = min(256, old - start)
                        let end = start + count
                        try state.append(
                            keys: Array(keyBits[start * 512..<end * 512]), values: Array(valueBits[start * 512..<end * 512]),
                            pooledKeys: [UInt16](repeating: 0x3f80, count: (end / 4 - start / 4) * 128),
                            rawTail: [UInt16](repeating: 0x3f80, count: end % 4 * 128), tokens: count)
                    }
                    let control = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: fused)
                    control.compactPVGrouped = fused
                    let variant = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, fusedPrefill: fused)
                    variant.compactPVGrouped = fused
                    variant.qsaExactKernels = true
                    try variant.prepareExactKernels()
                    let expected = try control.attention(
                        queries: queries, oldKeys: oldKeys, newKeys: newKeys,
                        oldValues: oldValues, newValues: newValues, selectedPositions: positions)
                    let nonpaged = try variant.attention(
                        queries: queries, oldKeys: oldKeys, newKeys: newKeys,
                        oldValues: oldValues, newValues: newValues, selectedPositions: positions)
                    let storage = try variant.makePagedStorage(capacityTokens: keys)
                    let paged = try variant.attentionPaged(
                        queries: queries, state: state, newKeys: newKeys,
                        newValues: newValues, selectedPositions: positions, storage: storage)
                    try require(
                        expected.count == rows * 6144 && nonpaged.count == expected.count && paged.count == expected.count,
                        "Paged fixture output geometry differs")
                    for i in 0..<expected.count {
                        try require(
                            expected[i].bitPattern == nonpaged[i].bitPattern,
                            "Nonpaged exact kernels differ from the current kernels \(label) at \(i): \(String(nonpaged[i].bitPattern, radix: 16))/\(String(expected[i].bitPattern, radix: 16))"
                        )
                        try require(
                            expected[i].bitPattern == paged[i].bitPattern,
                            "Paged exact kernels differ from the current kernels \(label) at \(i): \(String(paged[i].bitPattern, radix: 16))/\(String(expected[i].bitPattern, radix: 16))"
                        )
                    }
                    let units = UInt64(((rows + 1) / 2) * 6)
                    let snapshot = variant.qsaExactKernelCounts
                    try require(
                        snapshot.pvTiles == 2 * units && snapshot.fallbacks == 0
                            && (snapshot.qkTiles > 0) == fused && control.qsaExactKernelCounts.pvTiles == 0,
                        "Paged exact kernel path counters differ \(label): \(snapshot)")
                    try require(storage.snapshot().committedTokens == keys, "Paged exact fixture did not commit its frontier \(label)")
                    reports.append([
                        "path": fused ? "compact-paged" : "stripe-paged", "rows": rows, "keys": keys, "selection": "calibrated",
                        "inputs": seeded ? "seeded" : "clean", "comparisons": rows * 6144, "pvUnits": 2 * units,
                        "scope": "Nonpaged current against nonpaged and paged exact forms; byte identity only",
                    ])
                }
            }
        }
    }
}
