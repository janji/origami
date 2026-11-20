import Foundation
import Metal
import OrigamiEngine

/// Standalone, model-free exactness of `hc_exact_fusion` through the engine's own routes:
/// the encoded split-K projection with the one-thread reducer against the synchronous route of an
/// instance without the flag (the current `qmm_pinned_split_reduce`), and the fused HC tile entry against
/// the three-dispatch tile entry, every output word compared, with NaN/Inf/-0/subnormal/3e38 seeds and the
/// path counters checked. Diagnostic primitives only; no model or performance claim.
package enum HCExactFusionSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError("hc_exact_fusion: " + message) }
    }
    private static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("hc_exact_fusion fixture allocation failed")
        }
        return result
    }
    private static func zeroed(_ bytes: Int, _ device: MTLDevice) throws -> MTLBuffer {
        try buffer([UInt8](repeating: 0, count: max(16, bytes)), device)
    }
    private static func poisoned(_ bytes: Int, _ device: MTLDevice) throws -> MTLBuffer {
        try buffer([UInt8](repeating: 0xA5, count: max(16, bytes)), device)
    }
    private static func words(_ buffer: MTLBuffer, bytes: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: bytes))
    }
    private static func joined(_ command: MTLCommandBuffer) throws {
        command.commit()
        command.waitUntilCompleted()
        try require(command.status == .completed, "fixture command failed: \(String(describing: command.error))")
    }
    /// Exceptional BF16 activations: quiet NaNs with payloads, +-Inf, +-0, +-subnormal, 3e38, largest finite.
    private static let specialBF16: [UInt16] = [
        0x7fc0, 0xffc1, 0x7fff, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x807f, 0x7f62, 0xff62, 0x7f7f,
    ]
    /// Exceptional FP32 residual words for the entry (NaN payloads that narrowing quiets, ties, subnormals).
    private static let specialFP32: [UInt32] = [
        0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 0x7fbf_ffff, 0x7f80_0000, 0xff80_0000, 0x0000_0000,
        0x8000_0000, 0x0000_0001, 0x807f_ffff, 0x7f61_d1af, 0xff61_d1af, 0x3f80_8000, 0x3f81_8000,
    ]

    /// Encoded split-K projections: 2-31 partitions on the one-thread reducer (counted), 32+ partitions on
    /// the current reducer (counted fallback); every output bit equals the synchronous current route.
    package static func splitReduceExactness() throws {
        let plain = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true)
        let fused = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true, hcExactFusion: true)
        let device = fused.device
        guard let queue = device.makeCommandQueue() else { throw EngineError("hc_exact_fusion queue unavailable") }
        // (rows, width, outputs): HC down M256 split 5 / M128 split 10 / M64 split 20 (eight lanes, several parts
        // each); shared gate/up split 2; QSA k/v split 4; a ragged row count (split 5); M7 and M100xN96 split 40
        // (simd_sum: counted fallback); M255xN98 on the untiled SIMD kernel, split 10, 24990 outputs (scalar reducer).
        let shapes = [
            (256, 10240, 320), (128, 10240, 320), (64, 10240, 320), (33, 10240, 320), (256, 2560, 640),
            (256, 2560, 512), (157, 2560, 640), (7, 10240, 320), (100, 2560, 96), (255, 2560, 98),
        ]
        var expectedSerial: UInt64 = 0
        var expectedFallback: UInt64 = 0
        var splitShapes = 0
        for (index, (rows, width, outputs)) in shapes.enumerated() {
            try autoreleasepool {
                var state = UInt64(0x9E37_79B9 &+ index)
                func next() -> UInt64 {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return state >> 33
                }
                let input: [Float] = (0..<(rows * width)).map { _ in
                    next() % 211 == 0
                        ? BF16.decode(specialBF16[Int(next() % UInt64(specialBF16.count))])
                        : BF16.decode(BF16.encode(Float(Int(next() % 2001) - 1000) / 173))
                }
                let packed = (0..<(width * outputs / 8)).map { _ in UInt32(truncatingIfNeeded: next() &* 0x9e37_79b9_7f4a_7c15) }
                var scales = (0..<(width * outputs / 64)).map { _ in BF16.encode(Float(next() % 19 + 1) / 257) }
                var biases = scales.indices.map { _ in BF16.encode(Float(Int(next() % 23) - 11) / 131) }
                // NaN, +-Inf, 3e38 (overflow without NaN), subnormal and zero scales; -0, +Inf and -Inf biases.
                for (slot, value) in [(7, 0x7fc0), (3, 0x7f80), (11, 0xff80), (13, 0x7f62), (17, 0x0001), (19, 0x0000)] as [(Int, UInt16)] {
                    scales[slot % scales.count] = value
                }
                for (slot, value) in [(5, 0x8000), (23, 0x7f80), (29, 0xff80)] as [(Int, UInt16)] { biases[slot % biases.count] = value }
                let reference = try plain.qlinearPinnedBF16(
                    x: input, width: width, words: packed, scales: scales, biases: biases, simdMatrix: true)
                let plan = try fused.pinnedProjectionPlan(rows: rows, width: width, outputs: outputs, simdMatrix: true)
                try require(plan.schedule == reference.schedule, "plan schedule differs at M\(rows)/K\(width)/N\(outputs)")
                if plan.partialBytes > 0 {
                    splitShapes += 1
                    if plan.schedule.splitK < 32 { expectedSerial += 1 } else { expectedFallback += 1 }
                }
                let x = try buffer(input.map(BF16.encode), device)
                let w = try buffer(packed, device)
                let s = try buffer(scales, device)
                let b = try buffer(biases, device)
                let output = try poisoned(rows * outputs * 4, device)
                let partials = plan.partialBytes > 0 ? try poisoned(plan.partialBytes, device) : nil
                guard let command = queue.makeCommandBuffer() else { throw EngineError("hc_exact_fusion command unavailable") }
                try fused.encodePinnedProjection(
                    command, input: x, words: w, scales: s, biases: b, output: output,
                    partials: partials, plan: plan, simdMatrix: true)
                try joined(command)
                let encoded = Array(
                    UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: rows * outputs))
                if let i = encoded.indices.first(where: { encoded[$0] != reference.output[$0].bitPattern }) {
                    throw EngineError(
                        String(
                            format: "hc_exact_fusion split reduce differs at M%d/K%d/N%d split %d element %d: %08x/%08x",
                            rows, width, outputs, plan.schedule.splitK, i, encoded[i], reference.output[i].bitPattern))
                }
            }
        }
        try require(
            splitShapes == 10 && expectedSerial == 8 && expectedFallback == 2,
            "split-K case coverage differs: \(splitShapes)/\(expectedSerial)/\(expectedFallback)")
        try require(
            fused.serialSplitReduces == expectedSerial && fused.serialSplitReduceFallbacks == expectedFallback
                && plain.serialSplitReduces == 0 && plain.serialSplitReduceFallbacks == 0,
            "split reduce path counters differ: \(fused.serialSplitReduces)/\(fused.serialSplitReduceFallbacks)")
    }

    /// The fused HC tile entry against the current entry (host narrowing, which is hc_decode_narrow's
    /// expression, then the three-dispatch tile): the BF16 residual it writes and every tile output word.
    package static func fusedEntryExactness() throws {
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1)
        try autoreleasepool {
            let plainKernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let fusedKernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, quantizedTiled: true, quantizedTiledBM32: true,
                quantizedBM32MinRows: 256, hcExactFusion: true)
            let plain = try FusedHCDecode(
                arithmetic: plainKernels.arithmetic, ledger: ledger, prefillKernels: plainKernels, prefillSIMDMatrix: true)
            let fused = try FusedHCDecode(
                arithmetic: fusedKernels.arithmetic, ledger: ledger, prefillKernels: fusedKernels, prefillSIMDMatrix: true)
            try require(fused.fusedEntryEnabled && !plain.fusedEntryEnabled, "fused entry enablement differs from the flag")
            let device = fusedKernels.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("hc_exact_fusion queue unavailable") }
            let reserve = try ledger.reserve(bytes: 320 * 1024 * 1024, category: "hc_exact_fusion_fixture")
            defer { try? ledger.release(reserve) }
            func projection(_ width: Int, _ outputs: Int, _ seed: UInt32) throws -> FusedHCDecode.Projection {
                let packed = (0..<(width * outputs / 8)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
                let scales = (0..<(width * outputs / 64)).map { BF16.encode(Float($0 % 3 + 1) / 512) }
                let biases = (0..<(width * outputs / 64)).map { BF16.encode(-Float($0 % 3 + 1) * 7.5 / 512) }
                return .init(words: try buffer(packed, device), scales: try buffer(scales, device), biases: try buffer(biases, device))
            }
            var normWeights = (0..<10240).map { BF16.encode(Float($0 % 17 + 1) / 16) }
            for (slot, value) in [(5, 0x7fc0), (77, 0x7f80), (901, 0x8000), (4099, 0x0001), (7001, 0xff80)] as [(Int, UInt16)] {
                normWeights[slot] = value
            }
            let norm = try buffer(normWeights, device)
            let injection = try buffer((0..<(4 * 10240)).map { BF16.decode(BF16.encode(Float(($0 * 37) % 31 - 15) / 512)) }, device)
            let down = try projection(10240, 320, 17)
            let up = try projection(320, 10240, 71)
            var expectedTiles: UInt64 = 0
            for rows in [3, 157, 256] {
                try autoreleasepool {
                    var state = UInt64(0xC0FFEE &+ rows)
                    func next() -> UInt64 {
                        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                        return state >> 33
                    }
                    var residual: [UInt32] = (0..<(rows * 10240)).map { _ in
                        next() % 97 == 0
                            ? specialFP32[Int(next() % UInt64(specialFP32.count))]
                            : (Float(Int(next() % 20001) - 10000) / 3331).bitPattern
                    }
                    for i in 0..<2560 { residual[i] = 0 }  // all-zero group: rsqrt(eps)
                    for i in 2560..<5120 { residual[i] = 0x0000_0001 &+ UInt32(i & 7) }  // all-subnormal group
                    for i in 5120..<7680 { residual[i] = i & 1 == 0 ? 0x7f61_d1af : 0xff61_d1af }  // squares overflow
                    for i in 10240..<12800 { residual[i] = 0x8000_0000 }  // all -0
                    let narrowed = residual.map { BF16.encode(Float(bitPattern: $0)) }
                    let fp32 = try buffer(residual, device)
                    let bf16 = try buffer(narrowed, device)
                    let fusedBF16 = try poisoned(rows * 10240 * 2, device)
                    let partial = try plain.tilePartialBytes(rows: rows)
                    func rowBuffers() throws -> FusedHCDecode.RowBuffers {
                        try FusedHCDecode.RowBuffers(
                            norm: zeroed(rows * 10240 * 4, device), normBF16: zeroed(rows * 10240 * 2, device),
                            down: zeroed(rows * 320 * 4, device), activation: zeroed(rows * 320 * 2, device),
                            rawUp: zeroed(rows * 10240 * 4, device),
                            mixed: zeroed(rows * 2560 * 4, device), exceptions: zeroed(rows * 80 * 4, device),
                            rawInjection: zeroed(rows * 16, device),
                            injection: zeroed(rows * 16, device), capturedUp: zeroed(rows * 10240 * 4, device))
                    }
                    let a = try rowBuffers()
                    let b = try rowBuffers()
                    func partials(_ bytes: Int) throws -> MTLBuffer? { bytes > 0 ? try zeroed(bytes, device) : nil }
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("hc_exact_fusion command unavailable") }
                    try plain.encodeTile(
                        command, rows: rows, residualBF16: bf16, normWeights: norm, downWeights: down, upWeights: up,
                        injectionWeights: injection, buffers: a, downPartials: partials(partial.down), upPartials: partials(partial.up),
                        injectionPartials: partials(partial.injection))
                    try fused.encodeTile(
                        command, rows: rows, residualBF16: fusedBF16, normWeights: norm, downWeights: down, upWeights: up,
                        injectionWeights: injection, buffers: b, downPartials: partials(partial.down), upPartials: partials(partial.up),
                        injectionPartials: partials(partial.injection), residualFP32: fp32)
                    try joined(command)
                    expectedTiles += 1
                    try require(
                        words(fusedBF16, bytes: rows * 10240 * 2) == words(bf16, bytes: rows * 10240 * 2),
                        "M\(rows) fused BF16 residual differs")
                    let sizes = [
                        rows * 10240 * 4, rows * 10240 * 2, rows * 320 * 4, rows * 320 * 2, rows * 10240 * 4,
                        rows * 2560 * 4, rows * 80 * 4, rows * 16, rows * 16, rows * 10240 * 4,
                    ]
                    let names = [
                        "norm", "normBF16", "down", "activation", "rawUp", "mixed", "exceptions", "rawInjection", "injection", "capturedUp",
                    ]
                    for (index, (x, y)) in zip(a.all, b.all).enumerated() {
                        try require(words(x, bytes: sizes[index]) == words(y, bytes: sizes[index]), "M\(rows) tile \(names[index]) differs")
                    }
                }
            }
            try require(
                fused.fusedEntryTiles == expectedTiles && plain.fusedEntryTiles == 0,
                "fused entry counters differ: \(fused.fusedEntryTiles)")
            // The flag-off owner refuses an FP32 entry instead of silently narrowing.
            let rows = 3
            let partial = try plain.tilePartialBytes(rows: rows)
            let s = try FusedHCDecode.RowBuffers(
                norm: zeroed(rows * 10240 * 4, device), normBF16: zeroed(rows * 10240 * 2, device),
                down: zeroed(rows * 320 * 4, device), activation: zeroed(rows * 320 * 2, device), rawUp: zeroed(rows * 10240 * 4, device),
                mixed: zeroed(rows * 2560 * 4, device), exceptions: zeroed(rows * 80 * 4, device), rawInjection: zeroed(rows * 16, device),
                injection: zeroed(rows * 16, device), capturedUp: nil)
            guard let command = queue.makeCommandBuffer() else { throw EngineError("hc_exact_fusion command unavailable") }
            var refused = false
            do {
                try plain.encodeTile(
                    command, rows: rows, residualBF16: try zeroed(rows * 10240 * 2, device), normWeights: norm, downWeights: down,
                    upWeights: up, injectionWeights: injection, buffers: s,
                    downPartials: partial.down > 0 ? try zeroed(partial.down, device) : nil,
                    upPartials: partial.up > 0 ? try zeroed(partial.up, device) : nil,
                    injectionPartials: partial.injection > 0 ? try zeroed(partial.injection, device) : nil,
                    residualFP32: try zeroed(rows * 10240 * 4, device))
            } catch is EngineError { refused = true }
            try require(refused && plain.fusedEntryTiles == 0, "flag-off owner accepted an FP32 entry")
        }
        try require(ledger.snapshot().reserved == 0, "fixture retained ledger reservations")
    }
}
