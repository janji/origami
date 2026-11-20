import Foundation
import Metal
import OrigamiEngine

private func gdnInputRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func gdnInputReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid GDN input composition contract succeeded")
}
private func gdnInputEqual(_ actual: [Float], _ expected: [Float], _ label: String) throws {
    guard actual.count == expected.count else { throw EngineError("GDN quartet count differs: " + label) }
    if let index = actual.indices.first(where: { actual[$0].bitPattern != expected[$0].bitPattern }) {
        throw EngineError("GDN quartet \(label)[\(index)] differs: \(actual[index].bitPattern) != \(expected[index].bitPattern)")
    }
}

func gdnBufferRegionConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-gdn-region-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ values: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": values]).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let valid: [String: Any] = ["gdn_buffer_region": true, "pinned_rms": true, "gdn_resident_state": true]
    try gdnInputRequire(try !load([:]).gdnBufferRegion && load(valid).gdnBufferRegion, "GDN region is not opt-in")
    for (key, value) in [
        ("pinned_rms", false as Any), ("gdn_resident_state", false as Any),
        ("gdn_buffer_region", 1 as Any), ("gdn_buffer_region", "true" as Any),
    ] {
        var invalid = valid
        invalid[key] = value
        try gdnInputReject { _ = try load(invalid) }
    }
}

func gdnInputCompositionConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-gdn-inputs-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ values: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": values]).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    let valid: [String: Any] = [
        "gdn_input_composition": true, "pinned_dense": true,
        "resident_tensor_bytes": 30_000_000, "dense_resident_bytes": 2_000_000,
    ]
    try gdnInputRequire(try !load([:]).gdnInputComposition && load(valid).gdnInputComposition, "GDN quartet default or flag differs")
    for (key, value) in [
        ("pinned_dense", false as Any),
        ("resident_tensor_bytes", 0 as Any), ("dense_resident_bytes", 0 as Any),
        ("gdn_input_composition", 1 as Any), ("gdn_input_composition", "true" as Any),
    ] {
        var invalid = valid
        invalid[key] = value
        try gdnInputReject { _ = try load(invalid) }
    }
}

func gdnInputCompositionActualRowsAndJoinedOwnership() throws {
    let ledger = try MemoryLedger(ceiling: 220_000_000, transientMargin: 1)
    let fixture = try ledger.reserve(bytes: 128 * 1024 * 1024, category: "gdn_input_test_fixture")
    try autoreleasepool {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("GDN test Metal unavailable") }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            guard
                let result = values.withUnsafeBytes({
                    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                })
            else {
                throw EngineError("GDN test allocation failed")
            }
            return result
        }
        func projection(_ outputs: Int, seed: UInt32) throws -> GDNInputProjections.Projection {
            let words = (0..<(outputs * 320)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
            let scales = (0..<(outputs * 40)).map { BF16.encode(Float($0 % 5 + 1) / 512) }
            let biases = (0..<(outputs * 40)).map { BF16.encode(-Float($0 % 5 + 1) * 7.5 / 512) }
            return try .init(words: buffer(words), scales: buffer(scales), biases: buffer(biases))
        }
        let qkv = try projection(10240, seed: 17)
        let z = try projection(6144, seed: 731)
        let b = try buffer((0..<(48 * 2560)).map { LayerArithmetic.round(Float(($0 * 31) % 127 - 63) / 1024) })
        let a = try buffer((0..<(48 * 2560)).map { LayerArithmetic.round(Float(($0 * 73) % 109 - 54) / 512) })
        let kernels = try ReferenceKernels(ledger: ledger, quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
        let dense = try PinnedDenseLinear(ledger: ledger)
        let helper = try GDNInputProjections(kernels: kernels, dense: dense, ledger: ledger, simdMatrix: true)
        let before = ledger.snapshot().reserved
        let cases = [1, 2, 3, 7, 8, 16, 17, 116, 255, 256]
        for rows in cases {
            try autoreleasepool {
                // Keep non-BF16 host values: the affine and dense views must
                // apply their own existing boundaries, not a shared truncation.
                let x: [Float] = (0..<(rows * 2560)).map { (index: Int) -> Float in
                    let centered = (index * 73 + 5) % 1009 - 504
                    return Float(centered) / Float(29)
                }
                let expectedQKV = try kernels.qlinearPinnedBF16(
                    x: x, width: 2560, outputs: 10240,
                    words: qkv.words, scales: qkv.scales, biases: qkv.biases, simdMatrix: true
                ).output
                let expectedZ = try kernels.qlinearPinnedBF16(
                    x: x, width: 2560, outputs: 6144,
                    words: z.words, scales: z.scales, biases: z.biases, simdMatrix: true
                ).output
                let expectedB = try dense.linear(x, weights: b, width: 2560, outputs: 48, outputBF16: true)
                let expectedA = try dense.linear(x, weights: a, width: 2560, outputs: 48, outputBF16: true)
                let actual = try helper.evaluate(x, qkv: qkv, z: z, b: b, a: a)
                try gdnInputEqual(actual.qkv, expectedQKV, "qkv rows\(rows)")
                try gdnInputEqual(actual.z, expectedZ, "z rows\(rows)")
                try gdnInputEqual(actual.b, expectedB, "b rows\(rows)")
                try gdnInputEqual(actual.a, expectedA, "a rows\(rows)")
            }
            try gdnInputRequire(ledger.snapshot().reserved == before, "GDN quartet retained tile workspace")
        }
        let snapshot = helper.snapshot()
        let rows = UInt64(cases.reduce(0, +))
        try gdnInputRequire(
            snapshot.submittedCommands == 10 && snapshot.completedCommands == 10 && snapshot.submittedRows == rows
                && snapshot.inputUploadCalls == 20 && snapshot.inputUploadBytes == rows * 2560 * 6,
            "GDN quartet did not use one join and two shared inputs per call")
        let x = [Float](repeating: 1, count: 2560)
        for cancelAt in [1, 2, 3] {
            let prior = helper.snapshot()
            var calls = 0
            var cancelled = false
            do {
                _ = try helper.evaluate(
                    x, qkv: qkv, z: z, b: b, a: a,
                    shouldContinue: {
                        calls += 1
                        return calls != cancelAt
                    })
            } catch is IOOperationCancelled { cancelled = true }
            let after = helper.snapshot()
            let joined: UInt64 = cancelAt == 3 ? 1 : 0
            try gdnInputRequire(
                cancelled && after.submittedCommands == prior.submittedCommands + joined
                    && after.completedCommands == prior.completedCommands + joined && ledger.snapshot().reserved == before,
                "GDN cancellation returned before its command/ownership retired")
        }
        let short = try buffer([Float](repeating: 0, count: 2))
        try gdnInputReject { _ = try helper.evaluate([], qkv: qkv, z: z, b: b, a: a) }
        try gdnInputReject { _ = try helper.evaluate(x, qkv: qkv, z: z, b: short, a: a) }
        try gdnInputReject {
            _ = try helper.evaluate(x, qkv: .init(words: short, scales: qkv.scales, biases: qkv.biases), z: z, b: b, a: a)
        }
        let tiny = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
        let denied = try GDNInputProjections(kernels: kernels, dense: dense, ledger: tiny, simdMatrix: true)
        try gdnInputReject { _ = try denied.evaluate(.init(repeating: 1, count: 116 * 2560), qkv: qkv, z: z, b: b, a: a) }
        try gdnInputRequire(
            tiny.snapshot().reserved == 0 && denied.snapshot().submittedCommands == 0,
            "Denied GDN workspace allocated or submitted partial work")
        try gdnInputRequire(ledger.snapshot().reserved == before, "Invalid GDN geometry retained workspace")
    }
    try ledger.release(fixture)
    try gdnInputRequire(ledger.snapshot().reserved == 0, "GDN quartet destruction retained reservations")
}

func gdnDeviceBlockConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-gdn-block-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    let valid: [String: Any] = [
        "gdn_device_block": true, "gdn_input_composition": true, "gdn_buffer_region": true,
        "pinned_dense": true, "pinned_rms": true, "pinned_unary": true, "pinned_reductions": true,
        "fused_gdn_preparation": true, "gdn_resident_state": true, "resident_tensor_bytes": 40_000_000, "dense_resident_bytes": 2_000_000,
    ]
    try gdnInputRequire(try !load([:]).gdnDeviceBlock && load(valid).gdnDeviceBlock, "GDN block must be explicit")
    for key in ["gdn_input_composition", "gdn_buffer_region", "fused_gdn_preparation", "pinned_unary", "pinned_reductions"] {
        var bad = valid
        bad[key] = false
        try gdnInputReject { _ = try load(bad) }
    }
    for value: Any in [1, "true"] {
        var bad = valid
        bad["gdn_device_block"] = value
        try gdnInputReject { _ = try load(bad) }
    }
}
