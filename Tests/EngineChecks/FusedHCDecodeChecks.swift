import Darwin
import Foundation
import Metal
import OrigamiEngine
import OrigamiEngineChecks

private func hcDecodeRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func hcDecodeReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("HC decode accepted an invalid contract")
}
private func hcDecodeEqual(_ actual: [Float]?, _ expected: [Float], _ label: String) throws {
    guard let actual, actual.count == expected.count else { throw EngineError("HC decode missing " + label) }
    if let index = actual.indices.first(where: { actual[$0].bitPattern != expected[$0].bitPattern }) {
        throw EngineError("HC decode \(label)[\(index)] differs: \(actual[index].bitPattern) != \(expected[index].bitPattern)")
    }
}

/// Small synthetic fixed-shape weights. No source model or checkpoint is read.
/// The surrounding check reserves a 32 MiB bound for these host/Metal fixtures.
private final class HCDecodeFixture {
    let device: MTLDevice
    var norm: [Float]
    let injection: [Float]
    let normBuffer, injectionBuffer: MTLBuffer
    let down, up: FusedHCDecode.Projection
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("HC fixture device unavailable") }
        self.device = device
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try values.withUnsafeBytes { bytes in
                guard let result = device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared) else {
                    throw EngineError("HC fixture allocation failed")
                }
                return result
            }
        }
        func projection(_ width: Int, _ outputs: Int, _ seed: UInt32) throws -> FusedHCDecode.Projection {
            let words = (0..<(width * outputs / 8)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
            let scales = (0..<(width * outputs / 64)).map { BF16.encode(Float($0 % 3 + 1) / 512) }
            let biases = (0..<(width * outputs / 64)).map { BF16.encode(-Float($0 % 3 + 1) * 7.5 / 512) }
            return try .init(words: buffer(words), scales: buffer(scales), biases: buffer(biases))
        }
        norm = (0..<10240).map { BF16.decode(BF16.encode(Float($0 % 17 + 1) / 16)) }
        injection = (0..<(4 * 10240)).map { BF16.decode(BF16.encode(Float(($0 * 37) % 31 - 15) / 512)) }
        normBuffer = try buffer(norm.map(BF16.encode))
        injectionBuffer = try buffer(injection)
        down = try projection(10240, 320, 17)
        up = try projection(320, 10240, 71)
    }
    func replaceNorm(_ values: [Float]) {
        precondition(values.count == 10240)
        norm = values.map(LayerArithmetic.round)
        let bits = norm.map(BF16.encode)
        bits.withUnsafeBytes { normBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    }
    func linear(_ input: [Float], name: String, kernels: ReferenceKernels, dense: PinnedDenseLinear) throws -> [Float] {
        if name.hasSuffix("input_mix_weight_down") {
            return try kernels.qlinearPinnedBF16(
                x: input, width: 10240, outputs: 320,
                words: down.words, scales: down.scales, biases: down.biases
            ).output
        }
        if name.hasSuffix("input_mix_weight_up") {
            return try kernels.qlinearPinnedBF16(
                x: input, width: 320, outputs: 10240,
                words: up.words, scales: up.scales, biases: up.biases
            ).output
        }
        guard name.hasSuffix("block_inject_weight") else { throw EngineError("Unknown HC fixture projection") }
        return try dense.linear(input, weights: injectionBuffer, width: 10240, outputs: 4, outputBF16: true)
    }
}

func fusedHCDecodeConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hc-decode-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let valid: [String: Any] = [
        "fused_hc_decode": true, "pinned_reductions": true, "pinned_dense": true,
        "pinned_unary": true, "resident_tensor_bytes": 10_000_000, "dense_resident_bytes": 1_000_000,
    ]
    let defaults = try load([:])
    let enabled = try load(valid)
    try hcDecodeRequire(!defaults.fusedHCDecode && enabled.fusedHCDecode, "HC decode default or explicit flag differs")
    for key in ["pinned_reductions", "pinned_dense", "pinned_unary"] {
        var invalid = valid
        invalid[key] = false
        try hcDecodeReject { _ = try load(invalid) }
    }
    for key in ["resident_tensor_bytes", "dense_resident_bytes"] {
        var invalid = valid
        invalid[key] = 0
        try hcDecodeReject { _ = try load(invalid) }
    }
    var invalid = valid
    for value in [1, "true"] as [Any] {
        invalid = valid
        invalid["fused_hc_decode"] = value
        try hcDecodeReject { _ = try load(invalid) }
    }
}

func fusedHCDecodeExhaustiveHandoffs() throws {
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    try autoreleasepool {
        let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
        let helper = try FusedHCDecode(arithmetic: arithmetic, ledger: ledger)
        let initial = ledger.snapshot().reserved
        // Every BF16 sign/exponent/payload, then explicit FP32 half-way rounding
        // cases proving the GPU boundary uses BF16.encode rather than truncation.
        var input = (0..<65536).map { BF16.decode(UInt16($0)) }
        input += [
            UInt32(0x3f80_7fff), 0x3f80_8000, 0x3f80_8001, 0x3f81_8000, 0xbf80_8000,
            0x0000_7fff, 0x0000_8000, 0x0000_8001, 0x8000_8001, 0x7f7f_8000,
            0x7f80_0001, 0xff80_0001, 0x7fff_ffff,
        ].map { Float(bitPattern: $0) }
        for start in stride(from: 0, to: input.count, by: 10240) {
            let values = Array(input[start..<min(start + 10240, input.count)])
            let actual = try helper.diagnosticHandoffs(values)
            let divided = values.map { LayerArithmetic.round(LayerArithmetic.round($0) / 4) }
            try hcDecodeEqual(actual.activation, divided.map(arithmetic.silu), "exhaustive BF16 /4 SiLU")
            try hcDecodeEqual(
                actual.injection, divided.map { LayerArithmetic.round(2 * arithmetic.sigmoid($0)) }, "exhaustive BF16 /4 sigmoid *2")
            try hcDecodeRequire(ledger.snapshot().reserved == initial, "HC decode handoff tile retained workspace")
        }
    }
    try hcDecodeRequire(ledger.snapshot().reserved == 0, "HC decode table lifetime retained reservations")
}

func fusedHCDecodeAllIntermediatesAndLayerParity() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1)
    let fixtureBound = try ledger.reserve(bytes: 32 * 1024 * 1024, category: "hc_decode_test_fixture")
    try autoreleasepool {
        let fixture = try HCDecodeFixture()
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let dense = try PinnedDenseLinear(ledger: ledger)
        let helper = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger)
        let initial = ledger.snapshot().reserved
        let residual = (0..<10240).map { BF16.decode(BF16.encode(Float(($0 * 73 + 5) % 1009 - 504) / 29)) }
        let normalWeights = fixture.norm
        let edge: [UInt16] = [0, 0x8000, 1, 63, 127, 128, 0x8001, 0x807f, 0x3f80, 0xbf80]
        let nonfinite: [UInt16] = [0, 0x8000, 1, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffd5, 0x7f81]
        let normCases = [
            normalWeights, (0..<10240).map { BF16.decode(edge[$0 % edge.count]) },
            (0..<10240).map { BF16.decode(nonfinite[$0 % nonfinite.count]) },
        ]
        for (caseIndex, weights) in normCases.enumerated() {
            fixture.replaceNorm(weights)
            // All reference stages call the existing unfused primitives with
            // the same immutable weights. Host math is the LayerMath source.
            let norm = try reductions.groupedRMS(x: residual, weights: fixture.norm, groupSize: 2560)
            let down = try fixture.linear(norm, name: "input_mix_weight_down", kernels: kernels, dense: dense)
            let activation = down.map { kernels.arithmetic.silu(LayerArithmetic.round($0 / 4)) }
            let rawUp = try fixture.linear(activation, name: "input_mix_weight_up", kernels: kernels, dense: dense)
            let up = rawUp.map(kernels.arithmetic.sigmoid)
            var mixed = [Float](repeating: 0, count: 2560)
            for column in 0..<2560 {
                var total: Float = 0
                for branch in 0..<4 {
                    let index = branch * 2560 + column
                    total = LayerArithmetic.round(total + LayerArithmetic.round(up[index] * norm[index]))
                }
                mixed[column] = LayerArithmetic.round(total / 4)
            }
            let rawInjection = try fixture.linear(norm, name: "block_inject_weight", kernels: kernels, dense: dense)
            let injection = rawInjection.map { LayerArithmetic.round(2 * kernels.arithmetic.sigmoid(LayerArithmetic.round($0 / 4))) }
            for combine in [false, true] {
                let actual = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer,
                    downWeights: fixture.down, upWeights: fixture.up, injectionWeights: combine ? fixture.injectionBuffer : nil,
                    capture: true)
                try hcDecodeEqual(actual.norm, norm, "norm case\(caseIndex)")
                try hcDecodeEqual(actual.down, down, "down case\(caseIndex)")
                try hcDecodeEqual(actual.activated, activation, "SiLU case\(caseIndex)")
                try hcDecodeEqual(actual.rawUp, rawUp, "up projection case\(caseIndex)")
                try hcDecodeEqual(actual.sigmoidUp, up, "up sigmoid case\(caseIndex)")
                try hcDecodeEqual(actual.mixed, mixed, "mixed case\(caseIndex)")
                try hcDecodeEqual(actual.injection, combine ? injection : [], "injection case\(caseIndex)")
                if combine {
                    try hcDecodeEqual(actual.rawInjection, rawInjection, "raw injection case\(caseIndex)")
                } else {
                    try hcDecodeRequire(actual.rawInjection == nil, "Final HC produced an unused injection capture")
                }
                try hcDecodeRequire(ledger.snapshot().reserved == initial, "HC pipeline retained intermediate buffers")
                if caseIndex == 2 { try hcDecodeRequire(actual.repairedOutputs > 0, "Nonfinite HC bypassed host payload repair") }
            }
        }
        fixture.replaceNorm(normalWeights)
        let uncaptured = try helper.evaluate(
            residual: residual, normWeights: fixture.normBuffer,
            downWeights: fixture.down, upWeights: fixture.up, injectionWeights: fixture.injectionBuffer)
        try hcDecodeRequire(
            uncaptured.norm == nil && uncaptured.down == nil && uncaptured.activated == nil && uncaptured.rawUp == nil
                && uncaptured.sigmoidUp == nil && uncaptured.rawInjection == nil,
            "HC materialized disabled intermediate captures")
        let linear: LayerMath.Linear = { input, name, _ in try fixture.linear(input, name: name, kernels: kernels, dense: dense) }
        let original = LayerMath(kernels: kernels, linear: linear, tensor: { _ in fixture.norm }, reductions: reductions)
        let candidate = LayerMath(kernels: kernels, linear: linear, tensor: { _ in fixture.norm }, reductions: reductions)
        var callbacks = 0
        var oldCaptures: [String: [UInt32]] = [:]
        var newCaptures: [String: [UInt32]] = [:]
        candidate.hyperConnectionDecode = { input, base, combine, capture, keepGoing in
            callbacks += 1
            guard base == "fixture" else { throw EngineError("HC lost tensor prefix") }
            return try helper.evaluate(
                residual: input, normWeights: fixture.normBuffer,
                downWeights: fixture.down, upWeights: fixture.up, injectionWeights: combine ? fixture.injectionBuffer : nil,
                capture: capture, shouldContinue: keepGoing)
        }
        original.capture = { oldCaptures[$0] = $1.map(\.bitPattern) }
        candidate.capture = { newCaptures[$0] = $1.map(\.bitPattern) }
        for rows in [1, 2] {
            let input = Array(repeating: residual, count: rows).flatMap { $0 }
            for combine in [false, true] {
                let expected = try original.hyperConnection(residual: input, base: "fixture", useCombine: combine)
                let actual = try candidate.hyperConnection(residual: input, base: "fixture", useCombine: combine)
                try hcDecodeEqual(actual.mixed, expected.mixed, "LayerMath mixed M\(rows)")
                try hcDecodeEqual(actual.injection, expected.injection, "LayerMath injection M\(rows)")
                try hcDecodeRequire(oldCaptures == newCaptures, "HC LayerMath activation captures differ")
            }
        }
        try hcDecodeRequire(
            callbacks == 2 && helper.snapshot().submittedCommands == 9 && helper.snapshot().completedCommands == 9,
            "HC one-row callback eligibility or joined command count differs")
    }
    try ledger.release(fixtureBound)
    try hcDecodeRequire(ledger.snapshot().reserved == 0, "HC pipeline or fixture retained resources")
}

func fusedHCDecodeAdmissionCancellationAndRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 60_000_000, transientMargin: 1)
    let fixtureBound = try ledger.reserve(bytes: 32 * 1024 * 1024, category: "hc_decode_test_fixture")
    try autoreleasepool {
        let fixture = try HCDecodeFixture()
        let helper = try FusedHCDecode(arithmetic: .hostReference, ledger: ledger)
        let initial = ledger.snapshot().reserved
        let residual = [Float](repeating: 1, count: 10240)
        for stopAt in [1, 2, 3] {
            var calls = 0
            do {
                _ = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.down,
                    upWeights: fixture.up, injectionWeights: fixture.injectionBuffer, capture: true,
                    shouldContinue: {
                        calls += 1
                        return calls < stopAt
                    })
                throw EngineError("HC decode ignored cancellation")
            } catch is IOOperationCancelled {}
            try hcDecodeRequire(
                helper.snapshot().submittedCommands == helper.snapshot().completedCommands, "HC cancellation returned before GPU join")
            // A submitted command's buffers go when Metal drops it, a moment after the join.
            try hcDecodeRequire(ledger.waitForRelease { reserved, _ in reserved == initial }, "HC cancellation leaked buffers")
        }
        try hcDecodeRequire(helper.snapshot().submittedCommands == 1, "HC precommit cancellation submitted a command")
        // Deny successively late allocations, including the optional capture.
        // All admitted earlier scratch must retire, and none may be submitted.
        for remaining in [UInt64(0), 50_000, 150_000, 280_000] {
            let blocker = try ledger.reserve(
                bytes: ledger.ceiling - ledger.transientMargin - initial - remaining, category: "hc_test_admission_blocker")
            do {
                _ = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.down,
                    upWeights: fixture.up, injectionWeights: fixture.injectionBuffer, capture: true)
                throw EngineError("HC decode accepted denied workspace")
            } catch let error as EngineError {
                guard error.code == "insufficient_memory" else { throw error }
            }
            try ledger.release(blocker)
            try hcDecodeRequire(
                ledger.snapshot().reserved == initial && helper.snapshot().submittedCommands == 1,
                "HC admission failure retained partial workspace or submitted")
        }
        try hcDecodeReject {
            _ = try helper.evaluate(
                residual: [1], normWeights: fixture.normBuffer,
                downWeights: fixture.down, upWeights: fixture.up, injectionWeights: nil)
        }
        // Recovery after every cancellation/admission failure proves idle state
        // is restored and successful temporary buffers also drain after join.
        _ = try helper.evaluate(
            residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.down,
            upWeights: fixture.up, injectionWeights: nil)
        try hcDecodeRequire(
            helper.snapshot().submittedCommands == 2 && helper.snapshot().completedCommands == 2
                && ledger.waitForRelease { reserved, _ in reserved == initial },
            "HC failure recovery or successful retirement differs")
    }
    try ledger.release(fixtureBound)
    try hcDecodeRequire(ledger.snapshot().reserved == 0, "HC failure lifecycle retained tables")
}

func fusedHCPrefillActualRowSchedulesAndParity() throws {
    let ledger = try MemoryLedger(ceiling: 350_000_000, transientMargin: 1)
    let hostBound = try ledger.reserve(bytes: 128 * 1024 * 1024, category: "hc_prefill_fixture_and_capture_bound")
    try autoreleasepool {
        let fixture = try HCDecodeFixture()
        let kernels = try ReferenceKernels(
            pinnedUnary: true, ledger: ledger, quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
        let helper = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true)
        let reductions = try PinnedReductions(ledger: ledger)
        let dense = try PinnedDenseLinear(ledger: ledger)
        let baseline = ledger.snapshot().reserved
        let rowCases = [3, 5, 6, 7, 8, 17, 31, 32, 33, 255, 256]
        for rows in rowCases {
            let before = helper.snapshot()
            try autoreleasepool {
                let residual = (0..<(rows * 10240)).map { BF16.decode(BF16.encode(Float(($0 * 73 + 5) % 1009 - 504) / 29)) }
                let norm = try reductions.groupedRMS(x: residual, weights: fixture.norm, groupSize: 2560)
                let downResult = try kernels.qlinearPinnedBF16(
                    x: norm, width: 10240, outputs: 320,
                    words: fixture.down.words, scales: fixture.down.scales, biases: fixture.down.biases, simdMatrix: true)
                let activated = downResult.output.map { kernels.arithmetic.silu(LayerArithmetic.round($0 / 4)) }
                let upResult = try kernels.qlinearPinnedBF16(
                    x: activated, width: 320, outputs: 10240,
                    words: fixture.up.words, scales: fixture.up.scales, biases: fixture.up.biases, simdMatrix: true)
                let sigmoid = upResult.output.map(kernels.arithmetic.sigmoid)
                var mixed = [Float](repeating: 0, count: rows * 2560)
                for row in 0..<rows {
                    for column in 0..<2560 {
                        var sum: Float = 0
                        for branch in 0..<4 {
                            let index = row * 10240 + branch * 2560 + column
                            sum = LayerArithmetic.round(sum + LayerArithmetic.round(sigmoid[index] * norm[index]))
                        }
                        mixed[row * 2560 + column] = LayerArithmetic.round(sum / 4)
                    }
                }
                let rawInjection = try dense.linear(norm, weights: fixture.injectionBuffer, width: 10240, outputs: 4, outputBF16: true)
                let injection = rawInjection.map { LayerArithmetic.round(2 * kernels.arithmetic.sigmoid(LayerArithmetic.round($0 / 4))) }
                let actual = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer,
                    downWeights: fixture.down, upWeights: fixture.up, injectionWeights: fixture.injectionBuffer, capture: true)
                if [3, 255, 256].contains(rows) {
                    try WholeExecutionSelfChecks.hcTile(
                        helper: helper, ledger: ledger, residual: residual, normWeights: fixture.normBuffer,
                        down: fixture.down, up: fixture.up, injection: fixture.injectionBuffer, expected: actual)
                }
                for (actualValues, expected, name) in [
                    (actual.norm, norm, "norm"), (actual.down, downResult.output, "down"),
                    (actual.activated, activated, "SiLU"), (actual.rawUp, upResult.output, "raw up"),
                    (actual.sigmoidUp, sigmoid, "sigmoid up"),
                    (Optional(actual.mixed), mixed, "mixed"), (actual.rawInjection, rawInjection, "raw injection"),
                    (Optional(actual.injection), injection, "injection"),
                ] {
                    try hcDecodeEqual(actualValues, expected, "prefill M\(rows) " + name)
                }
                let snapshot = helper.snapshot()
                try hcDecodeRequire(
                    snapshot.lastPrefillRows == rows && snapshot.lastPrefillDown?.schedule == downResult.schedule
                        && snapshot.lastPrefillDown?.implementation == downResult.implementation
                        && snapshot.lastPrefillUp?.schedule == upResult.schedule
                        && snapshot.lastPrefillUp?.implementation == upResult.implementation,
                    "HC changed actual row/config projection dispatch")
                let densePlan = try PinnedDenseLinear.plan(rows: rows, width: 10240, outputs: 4)
                try hcDecodeRequire(snapshot.lastPrefillInjection == densePlan, "HC changed dense split-K plan")
                if rows == 256 {
                    try hcDecodeRequire(
                        snapshot.lastPrefillDown?.schedule.splitK == 5 && snapshot.lastPrefillDown?.schedule.partitionSize == 2048
                            && snapshot.lastPrefillDown?.implementation == "simdgroup_tiled32x32_k32_bf16_partitions"
                            && snapshot.lastPrefillUp?.schedule.splitK == 1 && densePlan.splitKPartitions == 32
                            && densePlan.splitKPartitionSize == 320,
                        "HC M256 lost down split5/up split1/dense split32 source schedules")
                    let final = try helper.evaluate(
                        residual: residual, normWeights: fixture.normBuffer,
                        downWeights: fixture.down, upWeights: fixture.up, injectionWeights: nil)
                    try hcDecodeEqual(final.mixed, mixed, "prefill final without injection")
                    try hcDecodeRequire(
                        final.injection.isEmpty && final.norm == nil && final.rawUp == nil, "HC final materialized disabled outputs")
                }
            }
            try hcDecodeRequire(ledger.snapshot().reserved == baseline, "HC prefill retained admitted scratch or partials")
            let after = helper.snapshot()
            let expected: UInt64 = rows == 256 ? 2 : 1
            try hcDecodeRequire(
                after.prefillSubmittedCommands == before.prefillSubmittedCommands + expected
                    && after.prefillCompletedCommands == before.prefillCompletedCommands + expected,
                "HC prefill M\(rows) command coverage differs: submitted=\(after.prefillSubmittedCommands) completed=\(after.prefillCompletedCommands) expected delta=\(expected)"
            )
        }
        // Every shape has one captured evaluation; M256 also evaluates the
        // final mixer without injection. Keep this tied to the declared cases.
        let expectedCommands = UInt64(rowCases.count + rowCases.filter { $0 == 256 }.count)
        let complete = helper.snapshot()
        try hcDecodeRequire(
            complete.prefillSubmittedCommands == expectedCommands && complete.prefillCompletedCommands == expectedCommands,
            "HC prefill commands were not all joined: submitted=\(complete.prefillSubmittedCommands) completed=\(complete.prefillCompletedCommands) expected=\(expectedCommands)"
        )
        try hcDecodeReject {
            _ = try helper.evaluate(
                residual: .init(repeating: 1, count: 2 * 10240), normWeights: fixture.normBuffer,
                downWeights: fixture.down, upWeights: fixture.up, injectionWeights: nil)
        }
        let disabled = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger)
        try hcDecodeReject {
            _ = try disabled.evaluate(
                residual: .init(repeating: 1, count: 3 * 10240), normWeights: fixture.normBuffer,
                downWeights: fixture.down, upWeights: fixture.up, injectionWeights: nil)
        }
        try hcDecodeRequire(disabled.snapshot().submittedCommands == 0, "Disabled prefill path submitted work")
    }
    try ledger.release(hostBound)
    try hcDecodeRequire(ledger.snapshot().reserved == 0, "HC prefill fixture/helper lifetime retained resources")
}

func fusedHCPrefillAdmissionCancellationAndRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 250_000_000, transientMargin: 1)
    let hostBound = try ledger.reserve(bytes: 64 * 1024 * 1024, category: "hc_prefill_lifecycle_fixture_bound")
    try autoreleasepool {
        let fixture = try HCDecodeFixture()
        let kernels = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
        let helper = try FusedHCDecode(arithmetic: .hostReference, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true)
        let initial = ledger.snapshot().reserved
        let residual = [Float](repeating: 1, count: 256 * 10240)
        for stopAt in [1, 2, 3] {
            var calls = 0
            do {
                _ = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.down, upWeights: fixture.up,
                    injectionWeights: fixture.injectionBuffer, capture: true,
                    shouldContinue: {
                        calls += 1
                        return calls < stopAt
                    })
                throw EngineError("HC prefill ignored cancellation")
            } catch is IOOperationCancelled {}
            try hcDecodeRequire(
                ledger.snapshot().reserved == initial
                    && helper.snapshot().prefillSubmittedCommands == helper.snapshot().prefillCompletedCommands,
                "HC prefill cancellation leaked scratch or escaped GPU join")
        }
        try hcDecodeRequire(helper.snapshot().submittedCommands == 1, "HC prefill submitted before cancellation admission boundary")
        for remaining in [UInt64(0), 11_000_000, 30_000_000, 36_000_000, 46_000_000] {
            let blocker = try ledger.reserve(
                bytes: ledger.ceiling - ledger.transientMargin - initial - remaining, category: "hc_prefill_test_blocker")
            do {
                _ = try helper.evaluate(
                    residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.down, upWeights: fixture.up,
                    injectionWeights: fixture.injectionBuffer, capture: true)
                throw EngineError("HC prefill accepted denied split/capture workspace")
            } catch let error as EngineError { guard error.code == "insufficient_memory" else { throw error } }
            try ledger.release(blocker)
            try hcDecodeRequire(
                ledger.snapshot().reserved == initial && helper.snapshot().submittedCommands == 1,
                "HC prefill admission denial retained partials or submitted work")
        }
        _ = try helper.evaluate(
            residual: .init(repeating: 1, count: 3 * 10240), normWeights: fixture.normBuffer,
            downWeights: fixture.down, upWeights: fixture.up, injectionWeights: nil)
        try hcDecodeRequire(
            ledger.snapshot().reserved == initial && helper.snapshot().submittedCommands == 2 && helper.snapshot().completedCommands == 2,
            "HC prefill did not recover its idle owner after denied work")
    }
    try ledger.release(hostBound)
    try hcDecodeRequire(ledger.snapshot().reserved == 0, "HC prefill cancellation retained resource owners")
}

func fusedHCPrefillConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hc-prefill-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let valid: [String: Any] = [
        "fused_hc_prefill": true, "pinned_unary": true, "pinned_dense": true, "pinned_reductions": true,
        "resident_tensor_bytes": 10_000_000, "dense_resident_bytes": 1_000_000,
    ]
    let defaults = try load([:])
    let enabled = try load(valid)
    try hcDecodeRequire(
        !defaults.fusedHCPrefill && enabled.fusedHCPrefill && !enabled.fusedHCDecode, "HC prefill is not an independent opt-in")
    for (key, value) in [("pinned_unary", false), ("pinned_dense", false), ("pinned_reductions", false)] {
        var invalid = valid
        invalid[key] = value
        try hcDecodeReject { _ = try load(invalid) }
    }
    for value in [1, "true"] as [Any] {
        var invalid = valid
        invalid["fused_hc_prefill"] = value
        try hcDecodeReject { _ = try load(invalid) }
    }
}
