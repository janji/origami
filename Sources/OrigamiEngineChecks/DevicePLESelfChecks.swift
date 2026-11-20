import Foundation
import Metal
import OrigamiEngine

/// DevicePLE against an independent host oracle of `ReferencePLE.call` (same synchronous pinned
/// primitives, same scalar code) on synthetic weights. Bitwise comparison of the updated residual
/// and the next convolution state for a full tile, one decode token, chained tiles, a short tile
/// that reads state rows, and exceptional values. No original weights are needed.
package enum DevicePLESelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        if try !value() { throw EngineError(message) }
    }

    private struct Fixture {
        let keyWords, valueWords: [UInt32]
        let keyScales, keyBiases, valueScales, valueBiases: [UInt16]
        let normKey, normQuery, normConv, convolution: [Float]
    }

    private static func fixture() -> Fixture {
        func words(_ count: Int, _ seed: UInt32) -> [UInt32] { (0..<count).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed } }
        func scales(_ count: Int, _ step: Int) -> [UInt16] { (0..<count).map { BF16.encode(Float($0 % 7 + step) / 4096) } }
        func biases(_ count: Int, _ step: Int) -> [UInt16] { (0..<count).map { BF16.encode(-Float($0 % 5 + step) * 7.5 / 4096) } }
        func values(_ count: Int, _ scale: Float, _ seed: Int) -> [Float] {
            (0..<count).map { LayerArithmetic.round(Float(($0 * 37 + seed) % 255 - 127) * scale) }
        }
        return Fixture(
            keyWords: words(10240 * 320, 17), valueWords: words(2560 * 320, 991),
            keyScales: scales(10240 * 40, 1), keyBiases: biases(10240 * 40, 1),
            valueScales: scales(2560 * 40, 2), valueBiases: biases(2560 * 40, 2),
            normKey: values(2560, 1.0 / 128, 3).map { 1 + $0 }, normQuery: values(2560, 1.0 / 256, 11).map { 1 + $0 },
            normConv: values(2560, 1.0 / 192, 29).map { 1 + $0 }, convolution: values(40960, 1.0 / 512, 5))
    }

    private static func packedRows(tokens: Int, seed: UInt32) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: tokens * 1600)
        for row in 0..<(tokens * 16) {
            let base = row * 100
            for i in 0..<80 { bytes[base + i] = UInt8(truncatingIfNeeded: (UInt32(row * 80 + i) &* 2_246_822_519 &+ seed) >> 13) }
            for g in 0..<5 {
                let scale = BF16.encode(Float((row + g) % 9 + 1) / 512)
                let bias = BF16.encode(-Float((row * 3 + g) % 11) / 256)
                bytes[base + 80 + g * 2] = UInt8(scale & 0xff)
                bytes[base + 81 + g * 2] = UInt8(scale >> 8)
                bytes[base + 90 + g * 2] = UInt8(bias & 0xff)
                bytes[base + 91 + g * 2] = UInt8(bias >> 8)
            }
        }
        return bytes
    }

    /// ReferencePLE.call (pinned-dot path) plus its caller's residual addition, on explicit arrays.
    private static func oracle(
        hidden: [Float], packed: [UInt8], state: [UInt16], rows: Int, f: Fixture,
        kernels: ReferenceKernels, reductions: PinnedReductions
    ) throws -> (residual: [Float], state: [UInt16]) {
        let width = 10_240
        let arithmetic = kernels.arithmetic
        var embedding: [Float] = []
        embedding.reserveCapacity(rows * 2560)
        for row in 0..<(rows * 16) {
            let data = Data(packed[(row * 100)..<(row * 100 + 100)])
            embedding += try ReferenceWeights.dequantizedRow(
                words: ReferenceWeights.array(data.subdata(in: 0..<80), UInt32.self),
                scales: ReferenceWeights.array(data.subdata(in: 80..<90), UInt16.self),
                biases: ReferenceWeights.array(data.subdata(in: 90..<100), UInt16.self), groupSize: 32)
        }
        let keyRaw = try kernels.qlinearPinnedBF16(
            x: embedding, width: 2560, words: f.keyWords, scales: f.keyScales,
            biases: f.keyBiases, simdMatrix: true
        ).output
        let key = try reductions.groupedRMS(x: keyRaw, weights: f.normKey, groupSize: 2560)
        let value = try kernels.qlinearPinnedBF16(
            x: embedding, width: 2560, words: f.valueWords, scales: f.valueScales,
            biases: f.valueBiases, simdMatrix: true
        ).output
        let query = try reductions.groupedRMS(x: hidden, weights: f.normQuery, groupSize: 2560)
        let dots = try reductions.pleDot(key: key, query: query)
        var gated = [Float](repeating: 0, count: hidden.count)
        for token in 0..<rows {
            for branch in 0..<4 {
                let start = token * width + branch * 2560
                let reduced = dots[token * 4 + branch]
                let scaled = LayerArithmetic.round(reduced / LayerArithmetic.round(sqrtf(2560)))
                let magnitude = max(abs(scaled), LayerArithmetic.round(1e-6))
                let gate = arithmetic.sigmoid(LayerArithmetic.round(sqrtf(magnitude)) * (scaled < 0 ? -1 : (scaled > 0 ? 1 : 0)))
                for column in 0..<2560 { gated[start + column] = LayerArithmetic.round(gate * value[token * 2560 + column]) }
            }
        }
        let normalized = try reductions.groupedRMS(x: gated, weights: f.normConv, groupSize: 2560)
        let full = state.map(BF16.decode) + normalized
        var residual = hidden
        for token in 0..<rows {
            for column in 0..<width {
                var sum: Float = 0
                for tap in 0..<4 { sum = sum.addingProduct(full[(token + tap * 3) * width + column], f.convolution[column * 4 + tap]) }
                let output = LayerArithmetic.round(gated[token * width + column] + arithmetic.silu(LayerArithmetic.round(sum)))
                residual[token * width + column] = LayerArithmetic.round(hidden[token * width + column] + output)
            }
        }
        return (residual, full.suffix(9 * width).map(BF16.encode))
    }

    package static func exactness() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true, quantizedTiled: true,
                quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let reductions = try PinnedReductions(ledger: ledger)
            let device = kernels.device
            let f = fixture()
            guard let queue = device.makeCommandQueue() else { throw EngineError("Device PLE check queue unavailable") }
            func buffer<T>(_ values: [T]) throws -> MTLBuffer {
                guard
                    let value = values.withUnsafeBytes({
                        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                    })
                else {
                    throw EngineError("Device PLE check allocation failed")
                }
                return value
            }
            let projections = DevicePLE.Projections(
                keyWords: try buffer(f.keyWords), keyScales: try buffer(f.keyScales),
                keyBiases: try buffer(f.keyBiases), valueWords: try buffer(f.valueWords), valueScales: try buffer(f.valueScales),
                valueBiases: try buffer(f.valueBiases), retained: [])
            let ple = try DevicePLE(
                kernels: kernels, reductions: reductions, simdMatrix: true, arithmetic: kernels.arithmetic,
                normKey: f.normKey, normQuery: f.normQuery, normConv: f.normConv, convolution: f.convolution, ledger: ledger)
            func onDevice(hidden: [Float], packed: [UInt8], state: [UInt16], rows: Int) throws -> (residual: [Float], state: [UInt16]) {
                let workspace = try ple.makeWorkspace(rows: rows)
                let residual = try buffer(hidden)
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Device PLE check command unavailable") }
                try packed.withUnsafeBytes {
                    try ple.encode(
                        command, residual: residual, rows: rows, packedRows: $0, state: state,
                        projections: projections, workspace: workspace, decode: rows == 1)
                }
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed else { throw EngineError("Device PLE check command failed") }
                let output = Array(UnsafeBufferPointer(start: residual.contents().assumingMemoryBound(to: Float.self), count: hidden.count))
                return (output, ple.nextState(workspace))
            }
            func hidden(_ rows: Int, _ seed: Int) -> [Float] {
                (0..<(rows * 10240)).map { LayerArithmetic.round(Float(($0 * 131 + seed) % 509 - 254) / 97) }
            }
            func compare(_ label: String, _ a: (residual: [Float], state: [UInt16]), _ b: (residual: [Float], state: [UInt16])) throws {
                let residual = a.residual.indices.filter { a.residual[$0].bitPattern != b.residual[$0].bitPattern }
                let state = a.state.indices.filter { a.state[$0] != b.state[$0] }
                func show(_ i: Int) -> String {
                    String(format: "%d:%08x/%08x", i, a.residual[i].bitPattern, b.residual[i].bitPattern)
                }
                try require(
                    residual.isEmpty,
                    "Device PLE residual differs: \(label): \(residual.count) elements, first "
                        + residual.prefix(4).map(show).joined(separator: " ") + " (oracle/device bits)")
                try require(
                    state.isEmpty,
                    "Device PLE convolution state differs: \(label): \(state.count) elements, first "
                        + state.prefix(4).map { String(format: "%d:%04x/%04x", $0, a.state[$0], b.state[$0]) }.joined(separator: " "))
            }
            let state0 = (0..<92160).map { BF16.encode(Float(($0 * 7) % 61 - 30) / 64) }
            // Full prefill tile, then a chained second tile from the adopted state.
            let h1 = hidden(256, 1)
            let p1 = packedRows(tokens: 256, seed: 5)
            let r1 = try oracle(hidden: h1, packed: p1, state: state0, rows: 256, f: f, kernels: kernels, reductions: reductions)
            let d1 = try onDevice(hidden: h1, packed: p1, state: state0, rows: 256)
            try compare("tile256", r1, d1)
            let h2 = hidden(256, 77)
            let p2 = packedRows(tokens: 256, seed: 91)
            try compare(
                "chained tile",
                try oracle(hidden: h2, packed: p2, state: r1.state, rows: 256, f: f, kernels: kernels, reductions: reductions),
                try onDevice(hidden: h2, packed: p2, state: d1.state, rows: 256))
            // Decode token and a short tile whose taps read state rows.
            for rows in [1, 5] {
                let h = hidden(rows, rows * 13)
                let p = packedRows(tokens: rows, seed: UInt32(rows * 41))
                try compare(
                    "rows\(rows)",
                    try oracle(hidden: h, packed: p, state: state0, rows: rows, f: f, kernels: kernels, reductions: reductions),
                    try onDevice(hidden: h, packed: p, state: state0, rows: rows))
            }
            // Exceptional values: NaN/Inf/-0 activations and an infinite row scale (NaN embedding).
            var h = hidden(5, 3)
            h[7] = .nan
            h[10240 + 3] = .infinity
            h[2 * 10240 + 9] = -.infinity
            h[3 * 10240 + 1] = -0.0
            var p = packedRows(tokens: 5, seed: 7)
            let infinite = BF16.encode(.infinity)
            p[100 * 18 + 82] = UInt8(infinite & 0xff)
            p[100 * 18 + 83] = UInt8(infinite >> 8)
            try compare(
                "exceptional", try oracle(hidden: h, packed: p, state: state0, rows: 5, f: f, kernels: kernels, reductions: reductions),
                try onDevice(hidden: h, packed: p, state: state0, rows: 5))
            try require(ple.counts.prefillTiles == 4 && ple.counts.decodeTokens == 1, "Device PLE path counters differ")
        }
        try require(ledger.snapshot().reserved == 0, "Device PLE check retained ledger reservations")
    }
}
