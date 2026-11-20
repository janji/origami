import Foundation
import Metal
import OrigamiEngine

private func nvfp4Require(_ value: @autoclosure () throws -> Bool, _ message: @autoclosure () -> String) throws {
    guard try value() else { throw EngineError(message()) }
}

private let e2m1: [Double] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
private func e4m3(_ b: UInt8) -> Double {
    let e = Int(b >> 3) & 15
    let m = Double(b & 7)
    let v = e == 0 ? m / 8 * pow(2, -6) : (1 + m / 8) * pow(2, Double(e - 7))
    return b & 0x80 != 0 ? -v : v
}

/// Seeded NVFP4 expert records (codes, E4M3 scales in a finite positive range, a global scale per projection)
/// and their weights as float64, [projection][output][input].
private func nvfp4Records(_ count: Int) -> (records: [[UInt8]], weights: [[[Double]]]) {
    let format = AffineFormat.nvfp4
    var records: [[UInt8]] = []
    var weights: [[[Double]]] = []
    for expert in 0..<count {
        var state = UInt64(0xF4F4_0000 + expert)
        func next() -> UInt32 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt32(truncatingIfNeeded: state >> 33)
        }
        var record = [UInt8](repeating: 0, count: format.expertRecordBytes)
        var w: [[Double]] = []
        for (index, projection) in ExpertProjection.allCases.enumerated() {
            let base = index * format.bytes(width: projection.width, outputs: projection.outputs)
            let values = projection.outputs * projection.width
            let codes = values / 2
            for i in 0..<codes { record[base + i] = UInt8(truncatingIfNeeded: next()) }
            for g in 0..<(values / 16) { record[base + codes + g] = UInt8(40 + next() % 32) }  // exponents 5...8
            let global = Float(next() % 7 + 1) / 1024
            withUnsafeBytes(of: global.bitPattern.littleEndian) { for k in 0..<4 { record[base + codes + values / 16 + k] = $0[k] } }
            var matrix = [Double](repeating: 0, count: values)
            for i in 0..<values {
                let byte = record[base + i / 2]
                let code = Int(i % 2 == 0 ? byte & 15 : byte >> 4)
                matrix[i] = e2m1[code] * e4m3(record[base + codes + i / 16]) * Double(global)
            }
            w.append(matrix)
        }
        records.append(record)
        weights.append(w)
    }
    return (records, weights)
}

/// Every expert kernel reads NVFP4 records natively (`QFP4`): fused QMV, multirow, QMM, SIMD and grouped
/// tiled/BM32 schedules and the gathered ten-expert decode agree with a float64 reference of code × scale ×
/// global within bf16 rounding, and the tile8 decode equals the gathered QMV decode bit for bit. A wrong nibble
/// order, scale index or missing global scale is off by ~100%.
func nvfp4ExpertKernelsMatchReference() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let (records, weights) = nvfp4Records(10)
    let ranks = [3, 8, 0, 9, 1, 6, 4, 2, 7, 5]
    let input = (0..<(128 * 2560)).map { BF16.decode(BF16.encode(Float(($0 * 103) % 991 - 495) / 193)) }
    func matvec(_ w: [Double], _ x: ArraySlice<Float>, outputs: Int, width: Int) -> [Double] {
        (0..<outputs).map { o in
            var s = 0.0
            for i in 0..<width { s += w[o * width + i] * Double(x[x.startIndex + i]) }
            return s
        }
    }
    func silu(_ v: Double) -> Double { v / (1 + exp(-v)) }
    func expert(_ e: Int, _ x: ArraySlice<Float>) -> [Double] {
        let gate = matvec(weights[e][0], x, outputs: 640, width: 2560)
        let up = matvec(weights[e][1], x, outputs: 640, width: 2560)
        let product = zip(gate, up).map { Float(silu($0) * $1) }
        return matvec(weights[e][2], product[...], outputs: 2560, width: 640)
    }
    func close(_ actual: [Float], _ expected: [Double], _ what: String) throws {
        try nvfp4Require(actual.count == expected.count, "\(what): \(actual.count) outputs, expected \(expected.count)")
        var error = 0.0
        var norm = 0.0
        for (a, e) in zip(actual, expected) {
            error += (Double(a) - e) * (Double(a) - e)
            norm += e * e
        }
        let relative = (error / max(norm, 1e-30)).squareRoot()
        try nvfp4Require(relative < 0.02, "\(what): relative error \(relative) against the float64 NVFP4 reference")
    }
    ModelLayout.use(.nvfp4Experts)
    defer { ModelLayout.use(.affine4Group64) }
    try nvfp4Require(PackStore.expertBytes == 3 * (819_200 + 102_400 + 4), "NVFP4 layout did not switch")
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reducer = try PinnedReductions(ledger: ledger)
        let plain = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let tiled = try FusedExpertExecution(
            arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true,
            groupedBM32: true, groupedBM32MinRows: 32, groupedPrefetch: true, groupedMMAMinRows: 64)
        let gathered = try GatheredExpertDecode(ledger: ledger, fused: plain, reductions: reducer)
        let configuration = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 12, slotsPerSlab: 2,
            demandWindow: 10, readConcurrency: 2)
        let engine = try ExpertExecution(device: device, ledger: ledger, configuration: configuration, packedAvailable: { false }) {
            key, _, destination in
            records[key.expert].withUnsafeBytes { destination.copyMemory(from: $0) }
        }
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
        try window.consumeTen(experts: Array(0..<10)) { buffers in
            for (fused, grouped, simd, rows) in [
                (plain, false, false, 1), (plain, false, false, 2), (plain, false, true, 7),
                (plain, false, false, 9), (plain, true, false, 7), (plain, true, true, 9),
                (tiled, true, true, 17), (tiled, true, true, 33), (tiled, true, true, 64), (tiled, true, true, 128),
            ] {
                let e = rows % 10
                let actual = try fused.evaluate(
                    Array(input[0..<(rows * 2560)]), buffer: buffers[e], groupedExpert: grouped, simdMatrix: simd)
                let expected = (0..<rows).flatMap { r in expert(e, input[(r * 2560)..<((r + 1) * 2560)]) }
                try close(actual, expected, "rows \(rows) grouped \(grouped) simd \(simd)")
            }
            let records = buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
            let x = input[0..<2560]
            let product = Array(input[0..<(10 * 640)])
            for (index, kind) in ExpertProjection.allCases.enumerated() {
                let actual = try gathered.diagnosticProjection(
                    kind == .down ? product : Array(x), records: records, canonicalRanks: ranks, kind: kind)
                var expected = [Double](repeating: 0, count: 10 * kind.outputs)
                for e in 0..<10 {
                    let v = kind == .down ? product[(ranks[e] * 640)..<((ranks[e] + 1) * 640)] : x
                    let out = matvec(weights[e][index], v, outputs: kind.outputs, width: kind.width)
                    for o in 0..<kind.outputs { expected[ranks[e] * kind.outputs + o] = out[o] }
                }
                try close(actual, expected, "gathered \(kind)")
            }
            let mix: [Float] = (0..<10).map { Float($0 + 1) / 55 }
            let actual = try gathered.evaluate(Array(x), records: records, canonicalRanks: ranks, weights: mix)
            var expected = [Double](repeating: 0, count: 2560)
            // Weight k scales the expert at canonical rank k.
            for e in 0..<10 { for (o, v) in expert(e, x).enumerated() { expected[o] += Double(mix[ranks[e]]) * v } }
            try close(actual, expected, "gathered ten-expert decode")
            // The tile8 decode kernels production runs give the same bits as these separate QMV dispatches.
            let split = try gathered.diagnosticReadySplit(
                Array(x), records: records, canonicalRanks: ranks, weights: mix,
                firstIndices: [0, 2, 4, 6, 8])
            try nvfp4Require(split.map(\.bitPattern) == actual.map(\.bitPattern), "NVFP4 tile8 decode differs from the gathered QMV decode")
            try gathered.trim()
        }
        try engine.finishSequence()
    }
    try nvfp4Require(ledger.snapshot().reserved == 0, "NVFP4 expert fixture leaked resources")
}

/// A demand that finds every slot busy (the governor shrank the cache under it) waits and succeeds once a slot frees;
/// a cancelled request stops waiting; other errors pass through.
func demandWaitsForBusySlots() throws {
    var calls = 0
    let value = try ExpertExecution.waitingForSlot({ true }, landed: { 0 }) { () -> Int in
        calls += 1
        if calls < 4 { throw EngineError("busy", code: "expert_slots_busy") }
        return 7
    }
    try nvfp4Require(value == 7 && calls == 4, "Busy-slot demand did not retry until a slot freed")
    var cancelled = false
    do {
        _ = try ExpertExecution.waitingForSlot({ false }, landed: { 0 }) { () -> Int in throw EngineError("busy", code: "expert_slots_busy")
        }
    } catch is IOOperationCancelled { cancelled = true }
    try nvfp4Require(cancelled, "A cancelled request kept waiting for a slot")
    var passed = false
    do {
        _ = try ExpertExecution.waitingForSlot({ true }, landed: { 0 }) { () -> Int in throw EngineError("other", code: "other") }
    } catch let error as EngineError { passed = error.code == "other" }
    try nvfp4Require(passed, "A different error was retried")
}

/// A slow drive only slows a demand: while reads keep landing the wait outlasts the limit and succeeds. When none lands
/// (nothing in flight, or a hung read) the limit still fails it, in wall time however slow each attempt is.
func slotWaitCountsOnlyIdleTime() throws {
    // 60 attempts sleep at least 0.3 s, past the 0.1 s limit; one read lands per attempt.
    var calls = 0
    var landed: UInt64 = 0
    let value = try ExpertExecution.waitingForSlot(
        { true },
        landed: {
            landed += 1
            return landed
        }, limit: 0.1
    ) { () -> Int in
        calls += 1
        if calls < 60 { throw EngineError("busy", code: "expert_slots_busy") }
        return 7
    }
    try nvfp4Require(value == 7, "A demand failed while slow reads were still landing")
    var failed = false
    let idle = Date()
    do {
        _ = try ExpertExecution.waitingForSlot({ true }, landed: { 0 }, limit: 0.1) { () -> Int in
            Thread.sleep(forTimeInterval: 0.05)
            throw EngineError("busy", code: "expert_slots_busy")
        }
    } catch let error as EngineError { failed = error.code == "expert_slots_busy" }
    let waited = Date().timeIntervalSince(idle)
    try nvfp4Require(failed && waited >= 0.1 && waited < 1, "Idle slot wait did not end at its wall-time limit: \(waited) s")
}

/// The M5 TensorOps twin of the 64 x 64 prefill MMA kernel, forced on this GPU (TensorOps run here without matrix
/// units), agrees with the simdgroup MMA kernel within bf16 output rounding: a dense projection and grouped experts.
func tensorOpsPrefillMatchesMMA() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    guard #available(macOS 26.0, *), device.supportsFamily(.metal4) else {
        throw FixtureUnavailable(description: "TensorOps need Metal 4 (macOS 26)")
    }
    let rows = 128
    let width = 2560
    let outputs = 640
    var state: UInt64 = 11
    func next() -> UInt32 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt32(truncatingIfNeeded: state >> 33)
    }
    let words = (0..<(outputs * width / 8)).map { _ in next() }
    let scales = (0..<(outputs * width / 64)).map { _ in BF16.encode(Float(next() % 19 + 1) / 512) }
    let biases = (0..<(outputs * width / 64)).map { _ in BF16.encode(Float(Int(next() % 23) - 11) / 256) }
    let x: [Float] = (0..<(rows * width)).map { (i: Int) -> Float in BF16.decode(BF16.encode(Float((i * 103) % 991 - 495) / 193)) }
    let records = (0..<10).map { _ in
        (0..<AffineFormat(bits: 4, group: 64).expertRecordBytes).map { _ in UInt8(truncatingIfNeeded: next()) & 0x3B }
    }
    func run(_ forced: Bool) throws -> (dense: [Float], experts: [Float], mma: UInt64) {
        QuantizedTensorOpsKernel.forced = forced
        defer { QuantizedTensorOpsKernel.forced = false }
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        return try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, quantizedTiled: true, quantizedTiledBM32: true,
                quantizedTiledPrefetch: true, quantizedMMABF16: true)
            func buffer<T>(_ values: [T]) -> MTLBuffer {
                values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
            }
            let dense = try kernels.qlinearPinnedBF16(
                x: x, width: width, outputs: outputs, words: buffer(words), scales: buffer(scales),
                biases: buffer(biases), simdMatrix: true
            ).output
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true,
                groupedBM32MinRows: 32, groupedPrefetch: true, groupedMMAMinRows: 64)
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 12, slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2)
            let engine = try ExpertExecution(device: device, ledger: ledger, configuration: configuration, packedAvailable: { false }) {
                key, _, destination in
                records[key.expert].withUnsafeBytes { destination.copyMemory(from: $0) }
            }
            try engine.beginSequence()
            var experts: [Float] = []
            let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
            try window.consumeTen(experts: Array(0..<10)) { buffers in
                experts = try fused.evaluate(Array(x[0..<(rows * 2560)]), buffer: buffers[3], groupedExpert: true, simdMatrix: true)
            }
            try engine.finishSequence()
            return (dense, experts, kernels.quantizedMMAProjections)
        }
    }
    let mma = try run(false)
    let tensor = try run(true)
    try nvfp4Require(mma.mma > 0 && tensor.mma > 0, "The dense projection did not take the 64 x 64 MMA path")
    for (name, a, b) in [("dense", tensor.dense, mma.dense), ("experts", tensor.experts, mma.experts)] {
        var error = 0.0
        var norm = 0.0
        for (u, v) in zip(a, b) {
            error += Double(u - v) * Double(u - v)
            norm += Double(v) * Double(v)
        }
        let relative = (error / max(norm, 1e-30)).squareRoot()
        try nvfp4Require(a.count == b.count && relative < 5e-3, "TensorOps \(name) differs from the MMA kernel: relative \(relative)")
    }
}

/// MTP speculation: the sampler's check of a one-token draft keeps the target distribution (accepted with p(draft),
/// otherwise drawn from p without the draft), is exact argmax at temperature 0, and the controller speculates only
/// where (1 + a) plain steps cost more than a pair step plus a redo.
func speculativeVerifyAndController() throws {
    let logits: [Float] = [2.0, 1.0, 0.5, -1.0]
    var greedy = ReferenceSampler(seed: 1)
    var zero = SamplingParameters()
    zero.temperature = 0
    try nvfp4Require(
        try greedy.verify(draft: 0, logits: logits, parameters: zero) == (true, 0)
            && greedy.verify(draft: 2, logits: logits, parameters: zero) == (false, 0), "Greedy draft check is not the argmax")
    var parameters = SamplingParameters()
    parameters.temperature = 1
    let masses = try ReferenceSampler.masses(logits: logits, parameters: parameters)  // top-p leaves them unnormalized
    let p = masses.map { $0 / masses.reduce(0, +) }
    var sampler = ReferenceSampler(seed: 7)
    var counts = [Int](repeating: 0, count: 4)
    let draws = 40_000
    for _ in 0..<draws { counts[try sampler.verify(draft: 1, logits: logits, parameters: parameters).token] += 1 }
    for i in 0..<4 {
        let observed = Double(counts[i]) / Double(draws)
        try nvfp4Require(abs(observed - Double(p[i])) < 0.01, "Speculative draws changed the distribution at \(i): \(observed) vs \(p[i])")
    }
    try nvfp4Require(
        GenerationDriver.speculationPays(plain: 0.1, pair: 0.15, acceptance: 0.8, attempts: 8) == true
            && GenerationDriver.speculationPays(plain: 0.1, pair: 0.35, acceptance: 0.9, attempts: 8) == false
            && GenerationDriver.speculationPays(plain: 0.1, pair: 0.15, acceptance: 0.8, attempts: 3) == nil,
        "Speculation controller rule differs")
}

/// The MTP pair step's two-row tile8 projection (see `DecodeProjectionKernel.twoRowSelfCheck`).
func twoRowProjectionMatchesOneRow() throws { try DecodeProjectionKernel.twoRowSelfCheck() }
