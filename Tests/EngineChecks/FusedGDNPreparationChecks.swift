import Foundation
import OrigamiEngine

private func gdnPrepRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func gdnPrepReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Fused GDN accepted invalid input")
}

func fusedGDNConfigurationAndAdmission() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-gdn-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let defaults = try load([:])
    let enabled = try load(["pinned_reductions": true, "fused_gdn_preparation": true, "fused_gdn_min_rows": 17])
    try gdnPrepRequire(
        !defaults.fusedGDNPreparation && defaults.fusedGDNMinRows == 8 && enabled.fusedGDNPreparation && enabled.fusedGDNMinRows == 17,
        "GDN configuration defaults differ")
    try gdnPrepReject { _ = try load(["fused_gdn_preparation": true]) }
    for value in [0, 257, 1.5, true, "8"] as [Any] { try gdnPrepReject { _ = try load(["fused_gdn_min_rows": value]) } }
    for value in [1, "true"] as [Any] { try gdnPrepReject { _ = try load(["fused_gdn_preparation": value]) } }
    let ledger = try MemoryLedger(ceiling: 900_000, transientMargin: 1)
    try autoreleasepool {
        let reductions = try PinnedReductions(ledger: ledger)
        try gdnPrepReject {
            _ = try FusedGDNPreparation(arithmetic: .hostReference, reductions: reductions, ledger: ledger, minimumRows: 0)
        }
        let helper = try FusedGDNPreparation(arithmetic: .hostReference, reductions: reductions, ledger: ledger)
        let initial = ledger.snapshot().reserved
        try gdnPrepRequire(
            !helper.usesGPU(rows: 7) && helper.usesGPU(rows: 8) && helper.usesGPU(rows: 256) && !helper.usesGPU(rows: 257),
            "GDN row floor differs")
        try gdnPrepReject { _ = try helper.gate(normed: [1], z: [1]) }
        do {
            _ = try helper.gate(normed: .init(repeating: 1, count: 8 * 6144), z: .init(repeating: 1, count: 8 * 6144))
            throw EngineError("GDN ignored insufficient allocation budget")
        } catch let error as EngineError { guard error.code == "insufficient_memory" else { throw error } }
        try gdnPrepRequire(
            ledger.snapshot().reserved == initial && helper.submittedCommands == 0, "GDN denial leaked allocations or submitted work")
    }
    try gdnPrepRequire(ledger.snapshot().reserved == 0, "GDN table ownership leaked")
}

func fusedGDNFixedValuesExceptionsAndCancellation() throws {
    let ledger = try MemoryLedger(ceiling: 128 * 1024 * 1024, transientMargin: 1)
    try autoreleasepool {
        let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let helper = try FusedGDNPreparation(arithmetic: arithmetic, reductions: reductions, ledger: ledger)
        let baseline = ledger.snapshot().reserved
        let rows = 8
        var mixed = [Float](repeating: 0, count: rows * 10240)
        var weights = [Float](repeating: 0, count: 40960)
        for row in 0..<rows { for column in [0, 2048, 4096] { mixed[row * 10240 + column] = 2 } }
        for column in 0..<10240 { weights[column * 4 + 3] = 1 }
        let result = try helper.prepare(
            mixed: mixed, tail: .init(repeating: 0, count: 30720), convolution: weights,
            a: .init(repeating: -128, count: rows * 48), b: .init(repeating: 0, count: rows * 48), dtBias: .init(repeating: 0, count: 48),
            aLog: .init(repeating: 0, count: 48), captureConvolved: true)
        for row in 0..<rows {
            try gdnPrepRequire(
                result.query[row * 2048].bitPattern == 0x3db5_0000 && result.keys[row * 2048] == 1
                    && result.values[row * 6144] == 1.7578125, "GDN fixed convolution/QK boundary differs")
        }
        try gdnPrepRequire(
            result.beta.allSatisfy { $0 == 0.5 } && result.decay.allSatisfy { $0 == 1 }
                && result.nextTail == mixed.suffix(30720).map(BF16.encode), "GDN factors or committed tail differs")
        try gdnPrepRequire(
            helper.preparationFallbacks == 0 && result.convolved?.count == mixed.count, "GDN fixed input bypassed normal GPU path")
        let all = (0..<(11 * 6144)).map { BF16.decode(UInt16(truncatingIfNeeded: $0)) }
        for (norm, z) in [(all, [Float](repeating: 0, count: all.count)), ([Float](repeating: 1, count: all.count), all)] {
            let expected = zip(norm, z).map { LayerArithmetic.round($0 * arithmetic.sigmoidFP32($1)) }
            let actual = try helper.gate(normed: norm, z: z)
            try gdnPrepRequire(actual.map(\.bitPattern) == expected.map(\.bitPattern), "GDN full BF16 gate domain differs")
        }
        try gdnPrepRequire(helper.gatingFallbacks == 2, "GDN exception fixtures did not exercise original CPU repair")
        for stopAt in [1, 2, 3] {
            var calls = 0
            do {
                _ = try helper.gate(
                    normed: .init(repeating: 2, count: 6144), z: .init(repeating: 0, count: 6144),
                    shouldContinue: {
                        calls += 1
                        return calls < stopAt
                    })
                throw EngineError("GDN gate ignored cancellation")
            } catch is IOOperationCancelled {}
            try gdnPrepRequire(
                helper.submittedCommands == helper.completedCommands && ledger.snapshot().reserved == baseline,
                "GDN cancellation escaped live buffers/work")
        }
    }
    try gdnPrepRequire(ledger.snapshot().reserved == 0, "GDN arithmetic lifetime leaked")
}

func fusedGDNLayerParityThresholdAndTransactionalState() throws {
    let ledger = try MemoryLedger(ceiling: 300_000_000, transientMargin: 1)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let helper = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
        var rejectProjection = false
        let linear: LayerMath.Linear = { input, name, _ in
            if name.hasSuffix("out_proj") {
                if rejectProjection { throw EngineError("Injected output projection failure") }
                return (0..<(input.count / 6144 * 2560)).map { input[($0 / 2560) * 6144 + $0 % 2560] }
            }
            let rows = input.count / 2560
            let width = name.hasSuffix("in_proj_qkv") ? 10240 : (name.hasSuffix("in_proj_z") ? 6144 : 48)
            return (0..<(rows * width)).map { LayerArithmetic.round(Float(($0 * 7 + 3) % 31 - 15) / 16) }
        }
        let tensor: LayerMath.Tensor = { name in
            if name.hasSuffix("conv1d.weight") { return (0..<40960).map { Float($0 % 4 + 1) / 16 } }
            if name.hasSuffix("norm.weight") { return .init(repeating: 1, count: 128) }
            return .init(repeating: 0, count: 48)
        }
        let old = LayerMath(kernels: kernels, linear: linear, tensor: tensor, reductions: reductions)
        let new = LayerMath(kernels: kernels, linear: linear, tensor: tensor, reductions: reductions, gdnPreparation: helper)
        var oldCaptures: [String: [UInt32]] = [:]
        var newCaptures: [String: [UInt32]] = [:]
        old.capture = { oldCaptures[$0] = $1.map(\.bitPattern) }
        new.capture = { newCaptures[$0] = $1.map(\.bitPattern) }
        var first = GDNState()
        var second = GDNState()
        for rows in [1, 7, 8, 17] {
            oldCaptures.removeAll()
            newCaptures.removeAll()
            let x = [Float](repeating: 1, count: rows * 2560)
            let a = try old.gdn(x: x, layer: 0, state: &first)
            let b = try new.gdn(x: x, layer: 0, state: &second)
            try gdnPrepRequire(
                a.map(\.bitPattern) == b.map(\.bitPattern) && first.recurrent.map(\.bitPattern) == second.recurrent.map(\.bitPattern)
                    && first.convolutionTail == second.convolutionTail && first.offset == second.offset && oldCaptures == newCaptures,
                "GDN layer threshold/captures/state changed")
        }
        try gdnPrepRequire(
            helper.submittedCommands == 4 && helper.completedCommands == 4 && helper.preparationFallbacks == 0
                && helper.gatingFallbacks == 0, "GDN row threshold or normal dispatch bypassed")
        let before = second
        rejectProjection = true
        try gdnPrepReject { _ = try new.gdn(x: .init(repeating: 1, count: 8 * 2560), layer: 0, state: &second) }
        try gdnPrepRequire(
            before.offset == second.offset && before.convolutionTail == second.convolutionTail
                && before.recurrent.map(\.bitPattern) == second.recurrent.map(\.bitPattern),
            "GDN failure published partial convolution/recurrent state")
    }
    try gdnPrepRequire(ledger.snapshot().reserved == 0, "GDN LayerMath integration leaked resources")
}
