import Foundation
import OrigamiEngine

private func hcRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func hcReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Fused HC accepted invalid input")
}
private func hcHost(norm: [Float], raw: [Float], arithmetic: LayerUnaryArithmetic) -> (mixed: [Float], up: [Float]) {
    let up = raw.map(arithmetic.sigmoid)
    var output = [Float](repeating: 0, count: norm.count / 4)
    // Independent transcription of the existing LayerMath operation boundaries.
    for token in 0..<(norm.count / 10240) {
        for column in 0..<2560 {
            var total: Float = 0
            for branch in 0..<4 {
                let offset = token * 10240 + branch * 2560 + column
                total = BF16.decode(BF16.encode(total + BF16.decode(BF16.encode(up[offset] * norm[offset]))))
            }
            output[token * 2560 + column] = BF16.decode(BF16.encode(total / 4))
        }
    }
    return (output, up)
}

func fusedHCConfigurationAndGeometry() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hc-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let defaults = try load([:])
    let enabled = try load(["fused_hc_epilogue": true, "fused_hc_min_rows": 17])
    try hcRequire(
        !defaults.fusedHCEpilogue && defaults.fusedHCMinRows == 8 && enabled.fusedHCEpilogue && enabled.fusedHCMinRows == 17,
        "Fused HC default/explicit configuration differs")
    for value in [0, 257, 1.5, true, "8"] as [Any] { try hcReject { _ = try load(["fused_hc_min_rows": value]) } }
    for value in [1, "true"] as [Any] { try hcReject { _ = try load(["fused_hc_epilogue": value]) } }
    let ledger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
    try hcReject { _ = try FusedHyperConnection(arithmetic: .hostReference, ledger: ledger, minimumRows: 0) }
    try hcRequire(ledger.snapshot().reserved == 0, "Invalid HC threshold reserved memory")
    try autoreleasepool {
        let helper = try FusedHyperConnection(arithmetic: .hostReference, ledger: ledger)
        let before = ledger.snapshot().reserved
        try hcRequire(
            !helper.usesGPU(rows: 1) && !helper.usesGPU(rows: 7) && helper.usesGPU(rows: 8) && helper.usesGPU(rows: 256)
                && !helper.usesGPU(rows: 257), "HC row floor changed")
        let malformed: [([Float], [Float])] = [([], []), ([1], [1]), ([Float](repeating: 0, count: 10240), [1])]
        for (norm, raw) in malformed {
            try hcReject { _ = try helper.evaluate(norm: norm, rawUp: raw) }
        }
        try hcRequire(ledger.snapshot().reserved == before && helper.submittedCommands == 0, "Invalid HC geometry allocated or dispatched")
    }
    try hcRequire(ledger.snapshot().reserved == 0, "HC table retained reservation")
}

func fusedHCRoundingCaptureAndRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1)
    try autoreleasepool {
        let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
        let helper = try FusedHyperConnection(arithmetic: arithmetic, ledger: ledger, minimumRows: 1)
        let baseline = ledger.snapshot().reserved
        for rows in [1, 15, 16, 17, 255, 256] {
            let norm = (0..<(rows * 10240)).map { BF16.decode(BF16.encode(Float(($0 * 73 + 5) % 1009 - 504) / 29)) }
            let raw = norm.indices.map { BF16.decode(BF16.encode(Float(($0 * 31 + 9) % 521 - 260) / 17)) }
            let expected = hcHost(norm: norm, raw: raw, arithmetic: arithmetic)
            for capture in [false, true] {
                let result = try helper.evaluate(norm: norm, rawUp: raw, captureUp: capture)
                try hcRequire(result.repairedOutputs == 0, "Normal HC values unexpectedly required CPU repair")
                try hcRequire(result.mixed.map(\.bitPattern) == expected.mixed.map(\.bitPattern), "HC rounding changed at M\(rows)")
                try hcRequire(
                    capture ? result.sigmoidUp?.map(\.bitPattern) == expected.up.map(\.bitPattern) : result.sigmoidUp == nil,
                    "HC capture changed the sigmoid or materialized a disabled capture")
                try hcRequire(ledger.snapshot().reserved == baseline, "HC retained temporary buffers")
            }
        }
        // These dyadic values distinguish the four sequential BF16 additions
        // from pairwise/FP32 reduction: exact source result is 1/8, not zero.
        let branches: [Float] = [512, 1, -512, 1]
        let norm = (0..<10240).map { branches[$0 / 2560] }
        let result = try helper.evaluate(norm: norm, rawUp: .init(repeating: 0, count: norm.count))
        try hcRequire(result.mixed.allSatisfy { $0.bitPattern == Float(0.125).bitPattern }, "HC reassociated sequential branch additions")
        let edgeNorm: [UInt16] = [
            0, 0x8000, 1, 2, 63, 64, 65, 127, 128, 129, 255, 0x8001, 0x807f, 0x8080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1,
            0xffd5, 0x7f81, 0xffaa,
        ]
        let edgeRaw: [UInt16] =
            [-100, -90, -88, -86, -85, -84, 0, 10, 100].map { BF16.encode(Float($0)) } + [0x7fc2, 0xffd6, 0x7f82, 0xffab]
        let count = 7 * 10240
        let tinyNorm = (0..<count).map { BF16.decode(edgeNorm[$0 % edgeNorm.count]) }
        let tinyRaw = (0..<count).map { BF16.decode(edgeRaw[($0 / edgeNorm.count) % edgeRaw.count]) }
        let tinyExpected = hcHost(norm: tinyNorm, raw: tinyRaw, arithmetic: arithmetic)
        let tinyActual = try helper.evaluate(norm: tinyNorm, rawUp: tinyRaw, captureUp: true)
        try hcRequire(
            tinyActual.repairedOutputs > 0 && tinyActual.mixed.map(\.bitPattern) == tinyExpected.mixed.map(\.bitPattern)
                && tinyActual.sigmoidUp?.map(\.bitPattern) == tinyExpected.up.map(\.bitPattern),
            "Tiny arithmetic or signaling/both-NaN operand repair changed HC output")
        // Every BF16 input encoding, including signed zeros, subnormals, largest
        // finite values, infinities and NaN payloads. No mismatch is suppressed.
        for exhaustiveNorm in [false, true] {
            let count = 7 * 10240
            let raw = (0..<count).map { exhaustiveNorm ? Float(0) : BF16.decode(UInt16(truncatingIfNeeded: $0)) }
            let norm = (0..<count).map { exhaustiveNorm ? BF16.decode(UInt16(truncatingIfNeeded: $0)) : branches[($0 / 2560) % 4] }
            let expected = hcHost(norm: norm, raw: raw, arithmetic: arithmetic)
            let actual = try helper.evaluate(norm: norm, rawUp: raw, captureUp: true)
            try hcRequire(
                actual.mixed.map(\.bitPattern) == expected.mixed.map(\.bitPattern)
                    && actual.sigmoidUp?.map(\.bitPattern) == expected.up.map(\.bitPattern),
                "HC exhaustive BF16 encoding parity failed (norm=\(exhaustiveNorm))")
        }
    }
    try hcRequire(ledger.snapshot().reserved == 0, "HC or unary tables leaked after retirement")
}

func fusedHCCancellationAndAllocationDenial() throws {
    let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
    try autoreleasepool {
        let helper = try FusedHyperConnection(arithmetic: .hostReference, ledger: ledger)
        let initial = ledger.snapshot().reserved
        let norm = [Float](repeating: 1, count: 8 * 10240)
        for stopAt in [1, 2, 3] {
            var calls = 0
            do {
                _ = try helper.evaluate(
                    norm: norm, rawUp: norm, captureUp: true,
                    shouldContinue: {
                        calls += 1
                        return calls < stopAt
                    })
                throw EngineError("HC ignored cancellation")
            } catch is IOOperationCancelled {}
            try hcRequire(
                ledger.snapshot().reserved == initial && helper.submittedCommands == helper.completedCommands,
                "HC cancellation escaped with live buffers or an unjoined command")
        }
        try hcRequire(helper.submittedCommands == 1, "HC cancellation dispatched before its submission boundary")
    }
    try hcRequire(ledger.snapshot().reserved == 0, "HC cancellation retained its table")
    let denied = try MemoryLedger(ceiling: 300_000, transientMargin: 1)
    try autoreleasepool {
        let helper = try FusedHyperConnection(arithmetic: .hostReference, ledger: denied)
        let initial = denied.snapshot().reserved
        let values = [Float](repeating: 1, count: 2 * 10240)
        do {
            _ = try helper.evaluate(norm: values, rawUp: values, captureUp: true)
            throw EngineError("HC accepted insufficient allocation budget")
        } catch let error as EngineError {
            guard error.code == "insufficient_memory" else { throw error }
        }
        try hcRequire(
            denied.snapshot().reserved == initial && helper.submittedCommands == 0, "HC denial leaked earlier buffers or dispatched")
    }
    try hcRequire(denied.snapshot().reserved == 0, "Denied HC lifetime retained its table")
}

func fusedHCLayerThresholdAndCapture() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let epilogue = try FusedHyperConnection(arithmetic: kernels.arithmetic, ledger: ledger, minimumRows: 8)
        let projection: LayerMath.Linear = { input, name, _ in
            if name.hasSuffix("input_mix_weight_down") { return .init(repeating: 0.125, count: input.count / 10240 * 320) }
            if name.hasSuffix("input_mix_weight_up") {
                return (0..<(input.count / 320 * 10240)).map { Float($0 % 17 - 8) / 4 }
            }
            return .init(repeating: 0, count: input.count / 10240 * 4)
        }
        let original = LayerMath(kernels: kernels, linear: projection, tensor: { _ in .init(repeating: 1, count: 10240) })
        let fused = LayerMath(
            kernels: kernels, linear: projection, tensor: { _ in .init(repeating: 1, count: 10240) }, hyperConnectionEpilogue: epilogue)
        var oldCaptures: [String: [UInt32]] = [:]
        var newCaptures: [String: [UInt32]] = [:]
        original.capture = { oldCaptures[$0] = $1.map(\.bitPattern) }
        fused.capture = { newCaptures[$0] = $1.map(\.bitPattern) }
        for rows in [1, 7, 8, 17] {
            let input = (0..<(rows * 10240)).map { Float($0 % 13 - 6) / 8 }
            let a = try original.hyperConnection(residual: input, base: "fixture", useCombine: rows != 17)
            let b = try fused.hyperConnection(residual: input, base: "fixture", useCombine: rows != 17)
            try hcRequire(
                a.mixed.map(\.bitPattern) == b.mixed.map(\.bitPattern) && a.injection.map(\.bitPattern) == b.injection.map(\.bitPattern)
                    && oldCaptures == newCaptures,
                "HC row threshold or capture changed LayerMath output")
        }
        try hcRequire(epilogue.submittedCommands == 2 && epilogue.completedCommands == 2, "HC decode fallback or row threshold bypassed")
    }
    try hcRequire(ledger.snapshot().reserved == 0, "LayerMath HC integration retained resources")
}
