import Foundation
import Metal
import OrigamiEngine

/// Standalone, model-free exactness of `hc_decode_fusion` through the engine's own encoders:
/// the engine dispatches the v2 kernels (hcd_down_act_v2 R4, hcd_up_mix_v2 8 columns x 4 SIMD groups,
/// hcd_injection_lookup_v2) after the fused entry;
/// the fused row HC (four dispatches) against the current eight-dispatch row HC of an owner without the flag, in
/// one command each, every stored word compared (norm, norm BF16, down, activation, raw up, mixed, exception
/// bitmap, raw injection, injection, captured up), plus the fused entry's BF16 residual and the fused mixed BF16
/// narrowing against BF16.encode (the caller-side hc_decode_narrow they replace). Inputs carry NaN payloads,
/// +-Inf, +-0, subnormals, overflowing squares and BF16 ties; the norm weights and the injection weights carry
/// exceptional words, so the epilogue's exception bitmap and repair columns are exercised. Path counters and the
/// flag-off refusal are checked. Diagnostic primitives only; no model or performance claim.
package enum HCDecodeFusionSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError("hc_decode_fusion: " + message) }
    }
    private static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("hc_decode_fusion fixture allocation failed")
        }
        return result
    }
    private static func filled(_ bytes: Int, _ byte: UInt8, _ device: MTLDevice) throws -> MTLBuffer {
        try buffer([UInt8](repeating: byte, count: max(16, bytes)), device)
    }
    private static func bytes(_ buffer: MTLBuffer, _ count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: count))
    }
    private static let specialFP32: [UInt32] = [
        0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 0x7fbf_ffff, 0x7f80_0000, 0xff80_0000, 0x0000_0000,
        0x8000_0000, 0x0000_0001, 0x807f_ffff, 0x7f61_d1af, 0xff61_d1af, 0x3f80_8000, 0x3f81_8000,
    ]

    package static func exactness() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let plain = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger)
            let fused = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger, decodeFusion: true)
            try require(
                fused.decodeFusionEnabled && !plain.decodeFusionEnabled && fused.decodeFusion && !plain.decodeFusion,
                "fused row enablement differs from the flag")
            let device = kernels.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("hc_decode_fusion queue unavailable") }
            let reserve = try ledger.reserve(bytes: 64 * 1024 * 1024, category: "hc_decode_fusion_fixture")
            defer { try? ledger.release(reserve) }
            var expectedRows: UInt64 = 0
            var expectedEntries: UInt64 = 0
            var expectedMixed: UInt64 = 0
            // Cases: (seed, FP32 entry + mixed narrowing, capture, weight regime).
            let cases: [(UInt64, Bool, Bool, Int)] = [
                (1, true, false, 0), (2, true, true, 1), (3, false, false, 0),
                (4, false, true, 2), (5, true, false, 2), (6, true, true, 0),
            ]
            for (seed, entry, capture, regime) in cases {
                try autoreleasepool {
                    var state = 0xD1CE_F00D &+ seed &* 0x9E37_79B9
                    func next() -> UInt64 {
                        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                        return state >> 33
                    }
                    func projection(_ width: Int, _ outputs: Int) throws -> FusedHCDecode.Projection {
                        let packed = (0..<(width * outputs / 8)).map { _ in UInt32(truncatingIfNeeded: next() &* 0x9e37_79b9_7f4a_7c15) }
                        // Regime 0: modest scales; 1: wide scales (large raw ups, saturating sigmoids); 2: tiny scales
                        // (sub-BF16-normal products flagged by the epilogue).
                        let divisor: Float = regime == 1 ? 3 : (regime == 2 ? 1e30 : 257)
                        let scales = (0..<(width * outputs / ModelLayout.group)).map { _ in BF16.encode(Float(next() % 19 + 1) / divisor) }
                        let biases = (0..<(width * outputs / ModelLayout.group)).map { _ in
                            BF16.encode(Float(Int(next() % 23) - 11) / (divisor * 0.5))
                        }
                        return .init(
                            words: try buffer(packed, device), scales: try buffer(scales, device), biases: try buffer(biases, device))
                    }
                    let down = try projection(10240, 320)
                    let up = try projection(320, 10240)
                    var normWords = (0..<10240).map { _ in BF16.encode(Float(next() % 33 + 1) / 16) }
                    for (slot, value) in [(5, 0x7fc0), (77, 0x7f80), (901, 0x8000), (4099, 0x0001), (7001, 0xff80)] as [(Int, UInt16)] {
                        normWords[slot] = value
                    }
                    let normWeights = try buffer(normWords, device)
                    var injectionWords = (0..<(4 * 10240)).map { _ in BF16.decode(BF16.encode(Float(Int(next() % 61) - 30) / 512)) }
                    if regime == 1 {
                        injectionWords[17] = .infinity
                        injectionWords[10240 + 3] = -.nan
                    }
                    let injection = try buffer(injectionWords, device)
                    var residual: [UInt32] = (0..<10240).map { _ in
                        next() % 97 == 0
                            ? specialFP32[Int(next() % UInt64(specialFP32.count))]
                            : (Float(Int(next() % 20001) - 10000) / 3331).bitPattern
                    }
                    if seed == 3 { for i in 0..<2560 { residual[i] = 0 } }  // all-zero group: rsqrt(eps)
                    if seed == 4 { for i in 2560..<5120 { residual[i] = 0x0000_0001 &+ UInt32(i & 7) } }  // all-subnormal group
                    if seed == 5 { for i in 5120..<7680 { residual[i] = i & 1 == 0 ? 0x7f61_d1af : 0xff61_d1af } }  // overflowing squares
                    let narrowed = residual.map { BF16.encode(Float(bitPattern: $0)) }
                    let fp32 = try buffer(residual, device)
                    let bf16 = try buffer(narrowed, device)
                    let fusedBF16 = try (entry ? filled(10240 * 2, 0xA5, device) : buffer(narrowed, device))
                    let mixedBF16 = try filled(2560 * 2, 0xA5, device)
                    func rowBuffers() throws -> FusedHCDecode.RowBuffers {
                        // Exceptions start zeroed (the caller's contract); everything else is poisoned.
                        try FusedHCDecode.RowBuffers(
                            norm: filled(10240 * 4, 0xA5, device), normBF16: filled(10240 * 2, 0xA5, device),
                            down: filled(320 * 4, 0xA5, device), activation: filled(320 * 2, 0xA5, device),
                            rawUp: filled(10240 * 4, 0xA5, device),
                            mixed: filled(2560 * 4, 0xA5, device), exceptions: filled(80 * 4, 0, device),
                            rawInjection: filled(16, 0xA5, device),
                            injection: filled(16, 0xA5, device), capturedUp: capture ? filled(10240 * 4, 0xA5, device) : nil)
                    }
                    let a = try rowBuffers()
                    let b = try rowBuffers()
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("hc_decode_fusion command unavailable") }
                    try plain.encodeRow(
                        command, residualBF16: bf16, normWeights: normWeights, downWeights: down, upWeights: up,
                        injectionWeights: injection, buffers: a)
                    try fused.encodeRow(
                        command, residualBF16: fusedBF16, normWeights: normWeights, downWeights: down, upWeights: up,
                        injectionWeights: injection, buffers: b, residualFP32: entry ? fp32 : nil, mixedBF16: entry ? mixedBF16 : nil)
                    command.commit()
                    command.waitUntilCompleted()
                    try require(command.status == .completed, "fixture command failed: \(String(describing: command.error))")
                    expectedRows += 1
                    if entry {
                        expectedEntries += 1
                        expectedMixed += 1
                    }
                    let label = "case \(seed) (entry \(entry), capture \(capture), regime \(regime))"
                    try require(bytes(fusedBF16, 10240 * 2) == bytes(bf16, 10240 * 2), label + " BF16 residual differs")
                    let sizes = [10240 * 4, 10240 * 2, 320 * 4, 320 * 2, 10240 * 4, 2560 * 4, 80 * 4, 16, 16, 10240 * 4]
                    let names = [
                        "norm", "normBF16", "down", "activation", "rawUp", "mixed", "exceptions", "rawInjection", "injection", "capturedUp",
                    ]
                    for (index, (x, y)) in zip(a.all, b.all).enumerated() {
                        let first = zip(bytes(x, sizes[index]), bytes(y, sizes[index])).enumerated().first(where: {
                            $0.element.0 != $0.element.1
                        })?.offset
                        try require(first == nil, label + " \(names[index]) differs at byte \(first ?? -1)")
                    }
                    // Norm weight 5 is a NaN: every case flags epilogue column 5, so the bitmap path is exercised.
                    let flagged = UnsafeBufferPointer(start: b.exceptions.contents().assumingMemoryBound(to: UInt32.self), count: 80)
                        .reduce(0) { $0 + $1.nonzeroBitCount }
                    try require(flagged > 0, label + " flagged no exceptional epilogue column")
                    if entry {
                        let mixed = UnsafeBufferPointer(start: a.mixed.contents().assumingMemoryBound(to: Float.self), count: 2560)
                        let expected = mixed.map { BF16.encode($0) }
                        try require(
                            bytes(mixedBF16, 2560 * 2) == expected.withUnsafeBytes { Array($0) }, label + " mixed BF16 narrowing differs")
                    } else {
                        try require(bytes(mixedBF16, 2560 * 2).allSatisfy { $0 == 0xA5 }, label + " wrote an unrequested mixed narrowing")
                    }
                }
            }
            try require(
                fused.decodeFusedRows == expectedRows && fused.decodeFusedEntries == expectedEntries
                    && fused.decodeFusedMixedNarrows == expectedMixed && fused.decodeFusionFallbacks == 0,
                "fused path counters differ: \(fused.decodeFusedRows)/\(fused.decodeFusedEntries)/\(fused.decodeFusedMixedNarrows)")
            try require(plain.decodeFusedRows == 0 && plain.decodeFusedEntries == 0, "flag-off owner counted fused rows")
            // The flag-off owner refuses an FP32 entry or a mixed narrowing instead of silently ignoring it.
            let s = try FusedHCDecode.RowBuffers(
                norm: filled(10240 * 4, 0, device), normBF16: filled(10240 * 2, 0, device),
                down: filled(320 * 4, 0, device), activation: filled(320 * 2, 0, device), rawUp: filled(10240 * 4, 0, device),
                mixed: filled(2560 * 4, 0, device), exceptions: filled(80 * 4, 0, device), rawInjection: filled(16, 0, device),
                injection: filled(16, 0, device), capturedUp: nil)
            let weights = FusedHCDecode.Projection(
                words: try filled(10240 * 320 / 2, 0, device), scales: try filled(10240 * 320 / 32, 0, device),
                biases: try filled(10240 * 320 / 32, 0, device))
            let norm = try filled(10240 * 2, 0, device)
            let inj = try filled(4 * 10240 * 4, 0, device)
            let entryOnly: (MTLBuffer?, MTLBuffer?) = (try filled(10240 * 4, 0, device), nil)
            let mixedOnly: (MTLBuffer?, MTLBuffer?) = (nil, try filled(2560 * 2, 0, device))
            for (entryBuffer, mixedBuffer) in [entryOnly, mixedOnly] {
                guard let command = queue.makeCommandBuffer() else { throw EngineError("hc_decode_fusion command unavailable") }
                var refused = false
                do {
                    try plain.encodeRow(
                        command, residualBF16: try filled(10240 * 2, 0, device), normWeights: norm, downWeights: weights,
                        upWeights: weights, injectionWeights: inj, buffers: s, residualFP32: entryBuffer, mixedBF16: mixedBuffer)
                } catch is EngineError { refused = true }
                try require(refused, "flag-off owner accepted a fused row entry or mixed narrowing")
            }
        }
        try require(ledger.snapshot().reserved == 0, "fixture retained ledger reservations")
    }
}
