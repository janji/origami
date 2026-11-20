import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of GDNDeviceBlock. `composed` runs the
/// complete block against LayerMath.gdn's host-joined path on synthetic weights, bit for bit, for
/// outputs, captures, state, tail and offset, and the `gdn_exact_kernels` block (eight-row recurrence, vector
/// convolution) against it, then covers the tile interface, cancellation, exception replay, admission denial
/// and a masked token.
package enum GDNDeviceBlockSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError(message) }
    }
    private static func equal(_ a: [Float], _ b: [Float], _ label: String) throws {
        try require(a.count == b.count, "GDN block count: " + label)
        if let i = a.indices.first(where: { a[$0].bitPattern != b[$0].bitPattern }) {
            throw EngineError("GDN block \(label)[\(i)]: \(a[i].bitPattern) != \(b[i].bitPattern)")
        }
    }
    private static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("GDN check buffer allocation failed")
        }
        return result
    }
    private static func allocate(_ bytes: Int, _ device: MTLDevice) throws -> MTLBuffer {
        guard let result = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("GDN check allocation failed")
        }
        return result
    }
    private static func values(_ buffer: MTLBuffer, _ count: Int, _ offset: Int = 0) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self).advanced(by: offset), count: count))
    }
    /// No original weights are needed.
    package static func composed() throws -> [String] {
        let ledger = try MemoryLedger(ceiling: 800_000_000, transientMargin: 1_000_000)
        let host = try ledger.reserve(bytes: 300_000_000, category: "gdn_block_fixed_fixture")
        defer { try? ledger.release(host) }
        return try autoreleasepool {
            let simd = true
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let device = kernels.device
            let dense = try PinnedDenseLinear(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let prep = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let helper = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: prep, ledger: ledger, simdMatrix: simd)
            let exactKernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: kernels.quantizedTiled, quantizedTiledBM32: kernels.quantizedTiledBM32,
                quantizedBM32MinRows: kernels.quantizedBM32MinRows, gdnExactKernels: true)
            let exactPrep = try FusedGDNPreparation(
                arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger, exactKernels: true)
            let exact = try GDNDeviceBlock(kernels: exactKernels, dense: dense, preparation: exactPrep, ledger: ledger, simdMatrix: simd)
            func projection(_ width: Int, _ outputs: Int, _ seed: UInt32) throws -> GDNDeviceBlock.Projection {
                let words = (0..<(outputs * width / 8)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
                let scales = (0..<(outputs * width / 64)).map { BF16.encode(Float($0 % 5 + 1) / 2048) }
                let biases = (0..<(outputs * width / 64)).map { BF16.encode(-Float($0 % 5 + 1) * 7.5 / 2048) }
                return try .init(words: buffer(words, device), scales: buffer(scales, device), biases: buffer(biases, device))
            }
            let qkv = try projection(2560, 10240, 17)
            let z = try projection(2560, 6144, 731)
            let out = try projection(6144, 2560, 83)
            let b = try buffer((0..<(48 * 2560)).map { (i: Int) in LayerArithmetic.round(Float((i * 31) % 127 - 63) / 4096) }, device)
            let a = try buffer((0..<(48 * 2560)).map { (i: Int) in LayerArithmetic.round(Float((i * 73) % 109 - 54) / 4096) }, device)
            let convolution = (0..<40960).map { (i: Int) in LayerArithmetic.round(Float((i * 17 + 7) % 251 - 125) / 128) }
            let bias = (0..<48).map { Float($0 % 5 - 2) / 16 }
            let logs = (0..<48).map { Float($0 % 7 - 3) / 16 }
            let norm = (0..<128).map { Float($0 % 7 + 1) / 8 }
            func linear(_ x: [Float], _ name: String, _ bf16: Bool) throws -> [Float] {
                if name.hasSuffix("in_proj_b") { return try dense.linear(x, weights: b, width: 2560, outputs: 48, outputBF16: true) }
                if name.hasSuffix("in_proj_a") { return try dense.linear(x, weights: a, width: 2560, outputs: 48, outputBF16: true) }
                let p: GDNDeviceBlock.Projection
                let width: Int
                let outputs: Int
                if name.hasSuffix("in_proj_qkv") {
                    p = qkv
                    width = 2560
                    outputs = 10240
                } else if name.hasSuffix("in_proj_z") {
                    p = z
                    width = 2560
                    outputs = 6144
                } else {
                    p = out
                    width = 6144
                    outputs = 2560
                }
                return try kernels.qlinearPinnedBF16(
                    x: x, width: width, outputs: outputs,
                    words: p.words, scales: p.scales, biases: p.biases, simdMatrix: simd
                ).output
            }
            func tensor(_ name: String) throws -> [Float] {
                if name.hasSuffix("conv1d.weight") { return convolution }
                if name.hasSuffix("dt_bias") { return bias }
                if name.hasSuffix("A_log") { return logs }
                return norm
            }
            let old = LayerMath(
                kernels: kernels, linear: linear, tensor: tensor, reductions: reductions,
                gdnPreparation: prep, gdnBufferRegion: try GDNBufferRegion(kernels: kernels, ledger: ledger, gate: prep))
            var initial = GDNState()
            initial.convolutionTail = (0..<30720).map { BF16.encode(Float($0 % 17 - 8) / 64) }
            initial.recurrent = (0..<(48 * 128 * 128)).map { (i: Int) in Float(i % 13 - 6) / 1024 }
            var expectedState = initial
            var state = initial
            var checks: [String] = []
            let geometries = [1, 1, 2, 3, 7, 8, 116, 255, 256, 1]
            for rows in geometries {
                try autoreleasepool {
                    let x: [Float] = (0..<(rows * 2560)).map { (index: Int) -> Float in
                        let centered = (index * 73 + 5) % 1009 - 504
                        return Float(centered) / Float(512)
                    }
                    var captures: [String: [Float]] = [:]
                    old.capture = { name, values in captures[String(name.split(separator: ".").last!)] = values }
                    let previous = state.recurrent
                    let tail = state.convolutionTail
                    let offset = state.offset
                    let expected = try old.gdn(x: x, layer: 0, state: &expectedState)
                    guard
                        let result = try helper.evaluate(
                            x, qkv: qkv, z: z, b: b, a: a, out: out, convolution: convolution,
                            dtBias: bias, aLog: logs, normWeights: norm, state: state, capture: true)
                    else { throw EngineError("Ordinary GDN fixture required replay") }
                    try equal(result.output, expected, "output rows\(rows)")
                    for (name, values) in captures { try equal(result.captures[name] ?? [], values, "\(name) rows\(rows)") }
                    try equal(result.state.recurrent, expectedState.recurrent, "state rows\(rows)")
                    try equal(state.recurrent, previous, "immutable prior rows\(rows)")
                    try require(
                        result.state.offset == expectedState.offset && result.state.convolutionTail == expectedState.convolutionTail
                            && state.offset == offset && state.convolutionTail == tail, "GDN block tail/offset publication differs")
                    if let prior = state.recurrentStorage {
                        try require(result.state.recurrentStorage !== prior, "GDN block overwrote prior storage")
                    }
                    guard
                        let exactResult = try exact.evaluate(
                            x, qkv: qkv, z: z, b: b, a: a, out: out, convolution: convolution,
                            dtBias: bias, aLog: logs, normWeights: norm, state: state, capture: true)
                    else { throw EngineError("Exact-kernel GDN fixture required replay") }
                    try equal(exactResult.output, result.output, "gdn_exact_kernels output rows\(rows)")
                    for (name, values) in result.captures {
                        try equal(exactResult.captures[name] ?? [], values, "gdn_exact_kernels \(name) rows\(rows)")
                    }
                    try equal(exactResult.state.recurrent, result.state.recurrent, "gdn_exact_kernels state rows\(rows)")
                    if [3, 116, 256].contains(rows) {
                        let charge = try ledger.reserve(bytes: 96 * 1024 * 1024, category: "whole_gdn_tile_fixture")
                        defer { try? ledger.release(charge) }
                        try autoreleasepool {
                            func output(_ bytes: Int) throws -> MTLBuffer {
                                let b = try allocate(bytes, device)
                                b.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
                                return b
                            }
                            let p = try helper.tilePartialBytes(rows: rows)
                            func partial(_ bytes: Int) throws -> MTLBuffer? { bytes == 0 ? nil : try output(bytes) }
                            let s = try GDNDeviceBlock.RowBuffers(
                                mixed: output(rows * 10240 * 4), z: output(rows * 6144 * 4),
                                b: output(rows * 48 * 4), a: output(rows * 48 * 4), history: buffer(state.convolutionTail, device),
                                convolution: buffer(convolution, device), bias: buffer(bias, device), logs: buffer(logs, device),
                                prepared: output(rows * 10240 * 4), beta: output(rows * 48 * 4), decay: output(rows * 48 * 4),
                                prepareFlags: output(4), mask: buffer([UInt32](repeating: 1, count: rows), device),
                                normWeights: buffer(norm, device),
                                normed: output(rows * 6144 * 4), gated: output(rows * 6144 * 4), gateFlags: output(4),
                                narrowed: output(rows * 6144 * 2),
                                output: output(rows * 2560 * 4), convolved: output(rows * 10240 * 4))
                            let input = try buffer(x, device)
                            let bits = try buffer(x.map(BF16.encode), device)
                            let initial = try buffer(state.recurrent, device)
                            let recurrence = try output((rows * 6144 + 48 * 128 * 128) * 4)
                            guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer() else {
                                throw EngineError("GDN tile fixture command unavailable")
                            }
                            try helper.encodeTile(
                                command, rows: rows, inputFP32: input, inputBF16: bits, qkv: qkv, z: z, b: b, a: a, out: out,
                                initial: initial, initialOffset: 0, recurrence: recurrence, buffers: s,
                                partials: .init(
                                    qkv: partial(p.qkv), z: partial(p.z), b: partial(p.dense), a: partial(p.dense),
                                    output: partial(p.output)))
                            command.commit()
                            command.waitUntilCompleted()
                            try require(
                                command.status == .completed && s.prepareFlags.contents().load(as: UInt32.self) == 0
                                    && s.gateFlags.contents().load(as: UInt32.self) == 0, "GDN tile interface failed or requested repair")
                            try equal(values(s.output, rows * 2560), result.output, "tile buffer output")
                            try equal(values(recurrence, 48 * 128 * 128, rows * 6144), result.state.recurrent, "tile buffer state")
                            for (name, b, count, offset) in [
                                ("mixed", s.mixed, rows * 10240, 0), ("z", s.z, rows * 6144, 0),
                                ("a", s.a, rows * 48, 0), ("b", s.b, rows * 48, 0), ("convolved", s.convolved!, rows * 10240, 0),
                                ("beta", s.beta, rows * 48, 0), ("decay", s.decay, rows * 48, 0), ("normed", s.normed, rows * 6144, 0),
                                ("gated", s.gated, rows * 6144, 0), ("query", s.prepared, rows * 2048, 0),
                                ("keys", s.prepared, rows * 2048, rows * 2048), ("values", s.prepared, rows * 6144, rows * 4096),
                                ("recurrence", recurrence, rows * 6144, 0),
                            ] {
                                try equal(values(b, count, offset), result.captures[name]!, "tile buffer " + name)
                            }
                            try equal(values(initial, 48 * 128 * 128), state.recurrent, "tile immutable initial")
                        }
                    }
                    state = result.state
                    checks.append("gdn-device-block-rows\(rows)")
                }
            }
            old.capture = nil
            try require(
                exactKernels.gdnExactRecurrenceTiles > 0 && exactKernels.gdnExactRecurrenceFallbacks == 0 && exactPrep.exactConvTiles > 0,
                "gdn_exact_kernels fixture did not take the exact kernels")
            let snapshot = helper.snapshot()
            let unchanged = state.recurrent
            let tail = state.convolutionTail
            try require(
                snapshot.exceptionReplays == 0 && snapshot.submittedCommands == UInt64(geometries.count)
                    && snapshot.completedCommands == snapshot.submittedCommands,
                "GDN block bypassed one joined command")
            let baseline = ledger.snapshot().reserved
            let x = [Float](repeating: 0.25, count: 2560)
            func call(_ conv: [Float] = convolution, _ normWeights: [Float] = norm, keepGoing: () -> Bool = { true }) throws
                -> GDNDeviceBlock.Result?
            {
                try helper.evaluate(
                    x, qkv: qkv, z: z, b: b, a: a, out: out, convolution: conv,
                    dtBias: bias, aLog: logs, normWeights: normWeights, state: state, shouldContinue: keepGoing)
            }
            for stopAt in [1, 2, 3, 4] {
                let before = helper.snapshot()
                var calls = 0
                var cancelled = false
                try autoreleasepool {
                    do {
                        _ = try call(keepGoing: {
                            calls += 1
                            return calls != stopAt
                        })
                    } catch is IOOperationCancelled { cancelled = true }
                }
                let after = helper.snapshot()
                let submitted: UInt64 = stopAt >= 3 ? 1 : 0
                try require(
                    cancelled && after.submittedCommands == before.submittedCommands + submitted
                        && after.completedCommands == before.completedCommands + submitted && ledger.snapshot().reserved == baseline,
                    "GDN block cancellation did not join/retire")
            }
            for exceptional in [Float(bitPattern: 1), Float.infinity, Float(bitPattern: 0x7fa1_2345)] {
                try autoreleasepool {
                    var conv = convolution
                    conv[0] = exceptional
                    try require(try call(conv) == nil, "GDN preparation exception published a result")
                    if !exceptional.isFinite {
                        var weights = norm
                        weights[0] = exceptional
                        try require(try call(convolution, weights) == nil, "GDN gating exception published a result")
                    }
                }
                try require(ledger.snapshot().reserved == baseline, "GDN exception retained private workspace")
            }
            for gateException in [false, true] {
                try autoreleasepool {
                    var exceptionalConv = convolution
                    var exceptionalNorm = norm
                    if gateException { exceptionalNorm[0] = .infinity } else { exceptionalConv[0] = Float(bitPattern: 1) }
                    func exceptionalTensor(_ name: String) throws -> [Float] {
                        if name.hasSuffix("conv1d.weight") { return exceptionalConv }
                        if name.hasSuffix("norm.weight") { return exceptionalNorm }
                        return try tensor(name)
                    }
                    func math() throws -> LayerMath {
                        LayerMath(
                            kernels: kernels, linear: linear, tensor: exceptionalTensor, reductions: reductions,
                            gdnPreparation: prep, gdnBufferRegion: try GDNBufferRegion(kernels: kernels, ledger: ledger, gate: prep))
                    }
                    let control = try math()
                    let candidate = try math()
                    candidate.gdnDeviceBlock = { input, _, oldState, capture, keepGoing in
                        try helper.evaluate(
                            input, qkv: qkv, z: z, b: b, a: a, out: out, convolution: exceptionalConv,
                            dtBias: bias, aLog: logs, normWeights: exceptionalNorm, state: oldState, capture: capture,
                            shouldContinue: keepGoing)
                    }
                    var expected = state
                    var actual = state
                    let before = helper.snapshot().exceptionReplays
                    try equal(
                        candidate.gdn(x: x, layer: 0, state: &actual), control.gdn(x: x, layer: 0, state: &expected),
                        "original exception replay")
                    try equal(actual.recurrent, expected.recurrent, "exception replay state")
                    try require(
                        actual.offset == expected.offset && actual.convolutionTail == expected.convolutionTail
                            && helper.snapshot().exceptionReplays == before + 1, "Exception replay changed transaction")
                }
                try require(ledger.snapshot().reserved == baseline, "Original exception replay retained workspace")
            }
            try equal(state.recurrent, unchanged, "cancel/exception prior state")
            try require(state.convolutionTail == tail, "Cancel/exception changed tail")
            let tiny = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
            let denied = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: prep, ledger: tiny, simdMatrix: simd)
            var rejected = false
            do {
                _ = try denied.evaluate(
                    x, qkv: qkv, z: z, b: b, a: a, out: out, convolution: convolution,
                    dtBias: bias, aLog: logs, normWeights: norm, state: initial)
            } catch let error as EngineError { rejected = error.code == "insufficient_memory" }
            try require(
                rejected && denied.snapshot().submittedCommands == 0 && tiny.snapshot().reserved == 0,
                "GDN denied workspace submitted/leaked")
            // Exercise a mask and an unchanged resident checkpoint after the new
            // current state is retired. Both are immutable owners of distinct buffers.
            try autoreleasepool {
                let before = state.recurrent
                guard
                    let masked = try helper.evaluate(
                        x, qkv: qkv, z: z, b: b, a: a, out: out, convolution: convolution,
                        dtBias: bias, aLog: logs, normWeights: norm, state: state, mask: [0])
                else { throw EngineError("Masked GDN replay") }
                try equal(masked.state.recurrent, before, "masked state")
                try equal(masked.output, .init(repeating: 0, count: 2560), "masked output")
                try require(masked.state.recurrentStorage !== state.recurrentStorage, "Masked GDN reused a published owner")
            }
            try require(ledger.snapshot().reserved == baseline, "GDN masked/checkpoint owner leaked")
            return checks
        }
    }
}
