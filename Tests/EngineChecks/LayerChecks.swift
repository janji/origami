import Foundation
import Metal
import OrigamiEngine

private func layerRequire(_ value: Bool, _ message: String) throws {
    if !value { throw EngineError(message) }
}

func layerStateAppendAndRestoreAreTransactional() throws {
    var state = QSAState()
    try state.reserveCapacity(tokens: 5)
    try state.append(
        keys: .init(repeating: BF16.encode(1), count: 3 * 512),
        values: .init(repeating: BF16.encode(2), count: 3 * 512),
        pooledKeys: [], rawTail: .init(repeating: BF16.encode(3), count: 3 * 128), tokens: 3)
    let snapshot = state
    do {
        try state.append(
            keys: .init(repeating: 0, count: 512), values: .init(repeating: 0, count: 512),
            pooledKeys: [], rawTail: [], tokens: 1)
        throw EngineError("Invalid pooling append unexpectedly succeeded")
    } catch let error as EngineError {
        try layerRequire(error.description != "Invalid pooling append unexpectedly succeeded", error.description)
    }
    try layerRequire(
        state.offset == snapshot.offset && state.keys == snapshot.keys && state.rawTail == snapshot.rawTail,
        "Rejected append changed state")
    try state.append(
        keys: .init(repeating: BF16.encode(4), count: 2 * 512),
        values: .init(repeating: BF16.encode(5), count: 2 * 512),
        pooledKeys: .init(repeating: BF16.encode(6), count: 128),
        rawTail: .init(repeating: BF16.encode(7), count: 128), tokens: 2)
    try layerRequire(state.offset == 5 && state.pooledKeys.count == 128 && state.rawTail.count == 128, "Block crossing lost tail")
    try layerRequire(snapshot.offset == 3 && snapshot.keys.count == 3 * 512, "Snapshot was mutated")
    try layerRequire(
        state.storageBytes == UInt64((5 * 1024 + 256) * 2) && state.capacityBytes >= state.storageBytes,
        "State memory accounting mismatch")
    let restored = try QSAState(
        offset: state.offset, keys: state.keys, values: state.values,
        pooledKeys: state.pooledKeys, rawTail: state.rawTail)
    try layerRequire(restored.values == state.values, "Restored state changed BF16 payloads")
    let gdn = GDNState()
    let gdnExpectedBytes: UInt64 = 3_207_168
    try layerRequire(gdn.storageBytes == gdnExpectedBytes, "GDN state size mismatch")
}

func pinnedUnaryNativePromotionMatchesExplicit() throws {
    let kernels = try ReferenceKernels()
    let inputs = (0...65535).map { BF16.decode(UInt16($0)) }
    for operation in [ReferenceKernels.BF16Unary.sigmoid, .silu, .softplus] {
        let explicit = try kernels.pinnedUnary(x: inputs, operation: operation)
        let native = try kernels.pinnedUnary(x: inputs, operation: operation, native: true)
        for index in inputs.indices {
            try layerRequire(
                explicit[index].isNaN && native[index].isNaN || explicit[index].bitPattern == native[index].bitPattern,
                "Native BF16 promotion differs for operation \(operation), bits \(index): \(explicit[index]) versus \(native[index])")
        }
    }
    try layerRequire(
        LayerArithmetic.sigmoid(0) == 0.5 && LayerArithmetic.sigmoid(-.infinity) == 0 && LayerArithmetic.sigmoid(.infinity) == 1
            && LayerArithmetic.silu(0) == 0,
        "Pinned unary endpoints failed")
    try layerRequire(
        LayerArithmetic.softplus(-.infinity) == 0 && LayerArithmetic.softplus(.infinity) == .infinity
            && LayerArithmetic.softplus(.nan).isNaN, "Softplus exceptional values failed")
}

private func syntheticLayers(_ kernels: ReferenceKernels, failFinal: Bool = false) -> LayerMath {
    LayerMath(
        kernels: kernels,
        linear: { x, name, _ in
            if failFinal && (name.hasSuffix(".out_proj") || name.hasSuffix(".o_proj")) {
                throw EngineError("Injected final projection failure")
            }
            let inputWidth = name.hasSuffix(".out_proj") || name.hasSuffix(".o_proj") ? 6144 : 2560
            let rows = x.count / inputWidth
            let width: Int
            if name.hasSuffix(".in_proj_qkv") {
                width = 10240
            } else if name.hasSuffix(".in_proj_z") {
                width = 6144
            } else if name.hasSuffix(".in_proj_a") || name.hasSuffix(".in_proj_b") {
                width = 48
            } else if name.hasSuffix(".q_proj") {
                width = 24 * 512
            } else if name.hasSuffix(".k_proj") || name.hasSuffix(".v_proj") {
                width = 512
            } else if name.hasSuffix(".index_qk_proj") {
                width = 640
            } else if name.hasSuffix(".out_proj") || name.hasSuffix(".o_proj") {
                width = 2560
            } else {
                throw EngineError("Unexpected synthetic projection: \(name)")
            }
            var output = [Float](repeating: 0, count: rows * width)
            for row in 0..<rows {
                for column in 0..<width {
                    let value: Float
                    if width == 2560 {
                        value = x[row * inputWidth + column]
                    } else if name.hasSuffix(".v_proj") {
                        value = x[row * inputWidth] + Float(column / 256)
                    } else if name.hasSuffix(".in_proj_qkv") {
                        value = x[row * inputWidth] / 16 + Float(column % 7 - 3) / 64
                    } else if name.hasSuffix(".index_qk_proj") {
                        value = x[row * inputWidth] / 16 + Float(column % 5) / 64
                    } else {
                        value = 0
                    }
                    output[row * width + column] = LayerArithmetic.round(value)
                }
            }
            return output
        },
        tensor: { name in
            if name.hasSuffix(".conv1d.weight") {
                return (0..<(10240 * 4)).map { $0 % 4 == 3 ? Float(0.5) : $0 % 4 == 2 ? Float(0.25) : 0 }
            }
            if name.hasSuffix(".dt_bias") || name.hasSuffix(".A_log") { return .init(repeating: 0, count: 48) }
            if name.hasSuffix(".q_norm.weight") || name.hasSuffix(".k_norm.weight") { return .init(repeating: 1, count: 256) }
            if name.hasSuffix(".norm.weight") || name.hasSuffix(".q_layernorm.weight") || name.hasSuffix(".k_layernorm.weight") {
                return .init(repeating: 1, count: 128)
            }
            throw EngineError("Unexpected synthetic tensor: \(name)")
        })
}

func assembledLayersPreserveContinuation() throws {
    let kernels = try ReferenceKernels()
    let math = syntheticLayers(kernels)
    let input = (1...5).flatMap { [Float](repeating: Float($0), count: 2560) }
    var fullGDN = GDNState()
    var splitGDN = GDNState()
    let full = try math.gdn(x: input, layer: 0, state: &fullGDN)
    let first = try math.gdn(x: Array(input[..<(2 * 2560)]), layer: 0, state: &splitGDN)
    let rest = try math.gdn(x: Array(input[(2 * 2560)...]), layer: 0, state: &splitGDN)
    try layerRequire(full.map(\.bitPattern) == (first + rest).map(\.bitPattern), "GDN continuation changed output")
    try layerRequire(
        fullGDN.recurrent.map(\.bitPattern) == splitGDN.recurrent.map(\.bitPattern) && fullGDN.convolutionTail == splitGDN.convolutionTail
            && splitGDN.offset == 5,
        "GDN continuation changed recurrence or convolution state")
    var fullQSA = QSAState()
    var splitQSA = QSAState()
    let fullAttention = try math.qsa(x: input, layer: 3, state: &fullQSA)
    let firstAttention = try math.qsa(x: Array(input[..<(2 * 2560)]), layer: 3, state: &splitQSA)
    let restAttention = try math.qsa(x: Array(input[(2 * 2560)...]), layer: 3, state: &splitQSA)
    try layerRequire(
        fullAttention.map(\.bitPattern) == (firstAttention + restAttention).map(\.bitPattern), "QSA continuation changed output")
    try layerRequire(
        fullQSA.keys == splitQSA.keys && fullQSA.values == splitQSA.values && fullQSA.pooledKeys == splitQSA.pooledKeys
            && fullQSA.rawTail == splitQSA.rawTail && splitQSA.offset == 5,
        "QSA continuation changed K/V, pooled blocks, or raw tail")
    // Inject a final projection failure; every preceding operation must remain private.
    let failing = syntheticLayers(kernels, failFinal: true)
    let before = splitQSA
    do { _ = try failing.qsa(x: Array(input[..<2560]), layer: 3, state: &splitQSA) } catch is EngineError {}
    try layerRequire(splitQSA.offset == before.offset && splitQSA.keys == before.keys, "Failed layer published QSA state")
}

func hyperConnectionMeanUsesPinnedColumnReduction() throws {
    let kernels = try ReferenceKernels()
    let math = LayerMath(
        kernels: kernels,
        linear: { _, name, _ in
            if name.hasSuffix(".input_mix_weight_down") { return .init(repeating: 0, count: 320) }
            if name.hasSuffix(".input_mix_weight_up") { return .init(repeating: 0, count: 10240) }
            throw EngineError("Unexpected projection in mean cancellation fixture")
        },
        tensor: { _ in
            [Float(256), 1, -256, 1].flatMap { [Float](repeating: $0, count: 2560) }
        })
    let result = try math.hyperConnection(residual: .init(repeating: 1, count: 10240), base: "fixture", useCombine: false)
    // Normed branch values [256,1,-256,1], sigmoid(0)=1/2.
    // Native column reduction drops the first half at the BF16 midpoint:
    // BF16(BF16(BF16(128+0.5)-128)+0.5)/4 == 0.125; FP32 mean is 0.25.
    try layerRequire(
        result.mixed == [Float](repeating: 0.125, count: 2560) && result.injection.isEmpty,
        "Hyper-connection mean lost pinned BF16 column reduction order")
}

func ordinaryRMSNormRoundsBeforeWeight() throws {
    let kernels = try ReferenceKernels()
    let result = try kernels.rmsnorm(x: [1, 1, 1, 2], weights: [1.5, 1.5, 1.5, 1.5])
    // The normalized BF16 values are [.7578125,.7578125,.7578125,1.515625].
    // Multiplying these by1.5 lands on BF16 ties that round upward to even.
    // A single final cast instead produces [1.1328125,...,2.265625].
    try layerRequire(
        result == [1.140625, 1.140625, 1.140625, 2.28125],
        "Ordinary RMSNorm must narrow its normalized value before learned weighting")
}

func pinnedUnaryTablesPreserveDomainAndOwnership() throws {
    let ledger = try MemoryLedger()
    var escaped: LayerUnaryArithmetic?
    try autoreleasepool {
        let owner = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        escaped = owner.arithmetic
        try layerRequire(
            owner.arithmetic.policy == "pinned_safe_fast_unary_tables_v1"
                && ledger.snapshot().reserved == LayerUnaryArithmetic.pinnedStorageBytes,
            "Unary table policy/accounting is incomplete")
        let direct = try ReferenceKernels(fastMath: false, fastFunctions: true)
        let inputs = (0..<65_536).map { BF16.decode(UInt16($0)) }
        for operation in [ReferenceKernels.BF16Unary.sigmoid, .silu, .softplus, .sigmoidFP32] {
            let expected = try direct.pinnedUnary(x: inputs, operation: operation, native: true)
            let actual: [Float]
            switch operation {
            case .sigmoid: actual = inputs.map(owner.arithmetic.sigmoid)
            case .silu: actual = inputs.map(owner.arithmetic.silu)
            case .softplus: actual = inputs.map(owner.arithmetic.softplus)
            case .sigmoidFP32: actual = inputs.map(owner.arithmetic.sigmoidFP32)
            }
            try layerRequire(
                zip(actual, expected).allSatisfy { $0.bitPattern == $1.bitPattern },
                "Unary table lookup changed a BF16 input or output bit")
        }
    }
    try layerRequire(
        escaped?.sigmoid(0) == 0.5 && ledger.snapshot().reserved == LayerUnaryArithmetic.pinnedStorageBytes,
        "Escaped arithmetic released its backing reservation too early")
    escaped = nil
    try layerRequire(ledger.snapshot().reserved == 0, "Unary table backing reservation leaked")
    let denied = try MemoryLedger(ceiling: 2_000_000, transientMargin: 0)
    var rejected = false
    do { _ = try LayerUnaryArithmetic.pinned(ledger: denied) } catch { rejected = true }
    try layerRequire(
        rejected && denied.snapshot().reserved == 0,
        "Unary generation must honor its workspace budget and unwind the persistent reservation")
}

func pinnedRMSGeometryAndResidentQuantizedBuffers() throws {
    let kernels = try ReferenceKernels(pinnedRMS: true)
    for width in [128, 256, 2560, 10_240] {
        let x = (0..<(width * 2)).map { index -> Float in index % 4 == 3 ? 2 : 1 }
        let norm = try kernels.rmsnorm(x: x, weights: .init(repeating: 1.5, count: width))
        let expected = x.map { $0 == 1 ? Float(1.140625) : Float(2.28125) }
        try layerRequire(norm == expected, "Pinned RMS lane/group/looped geometry or BF16 boundary changed")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal missing for resident quantized check") }
    func padded<T>(_ values: [T], offset: Int) throws -> MTLBuffer {
        guard
            let result = device.makeBuffer(
                length: offset + values.count * MemoryLayout<T>.stride + 32,
                options: .storageModeShared)
        else { throw EngineError("Fixture allocation failed") }
        values.withUnsafeBytes { bytes in
            result.contents().advanced(by: offset).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return result
    }
    let width = 640
    let outputs = 37
    let words = (0..<(outputs * width / 8)).map { UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) }
    let scales = (0..<(outputs * width / 64)).map { BF16.encode(Float(($0 % 3) + 1) / 32) }
    let biases = (0..<(outputs * width / 64)).map { BF16.encode(Float(($0 % 5) - 2) / 16) }
    let wordBuffer = try padded(words, offset: 16)
    let scaleBuffer = try padded(scales, offset: 8)
    let biasBuffer = try padded(biases, offset: 24)
    for rows in [1, 17] {
        let x = (0..<(width * rows)).map { Float(($0 % 17) - 8) / 8 }
        for simd in [false, true] {
            for grouped in [false, true] {
                let expected = try kernels.qlinearPinnedBF16(
                    x: x, width: width, words: words, scales: scales, biases: biases,
                    simdMatrix: simd, groupedExpert: grouped)
                let actual = try kernels.qlinearPinnedBF16(
                    x: x, width: width, outputs: outputs,
                    words: wordBuffer, scales: scaleBuffer, biases: biasBuffer, wordOffset: 16, scaleOffset: 8, biasOffset: 24,
                    simdMatrix: simd, groupedExpert: grouped)
                try layerRequire(
                    expected.schedule == actual.schedule && expected.implementation == actual.implementation
                        && zip(expected.output, actual.output).allSatisfy { $0.bitPattern == $1.bitPattern },
                    "Resident buffer offsets or actual-row quantized schedule changed projection bits")
                if grouped {
                    try layerRequire(
                        actual.schedule.kind == "qmm_grouped_expert_no_split" && actual.schedule.splitK == 1
                            && actual.schedule.partitionSize == width && actual.implementation.hasPrefix("grouped_expert_"),
                        "A short grouped expert slice fell back to QMV or dense split-K")
                }
            }
        }
    }
    for offset in [-4, 2, wordBuffer.length] {
        var rejected = false
        do {
            _ = try kernels.qlinearPinnedBF16(
                x: .init(repeating: 1, count: width), width: width, outputs: outputs,
                words: wordBuffer, scales: scaleBuffer, biases: biasBuffer, wordOffset: offset, scaleOffset: 8, biasOffset: 24)
        } catch { rejected = true }
        try layerRequire(rejected, "Resident quantized projection accepted an invalid word span")
    }
}
