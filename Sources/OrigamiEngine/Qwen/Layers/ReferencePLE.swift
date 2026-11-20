import Foundation

/// The PLE's causal convolution history: the last nine normalized rows, [9 x 10240] BF16 bits
/// (four taps at stride 3 reach nine rows back). Part of each sequence's state and checkpoints.
public struct PLEState {
    public var convolution = [UInt16](repeating: 0, count: 9 * 10_240)
    public init() {}
}

/// Row addresses of the PLE n-gram embedding table (320,001,536 rows of 160 4-bit values, group 32).
/// Each new token gets 16 rows: heads 0-7 hash the bigram (token, previous), heads 8-15 the trigram
/// (token, previous, the one before); a previous token at or before the most recent EOS (id 248044) is
/// replaced by that id. Only the two previous tokens matter, which is why sequence state keeps a
/// history of two (seeded with two EOS ids).
public enum PLEHash {
    /// Signed wrapping multiply/XOR and Euclidean remainder; an EOS at p only
    /// clears the history for positions after p, exactly as the pinned model.
    public static func rows(history: [Int64], newCount: Int, multipliers: [Int64], sizes: [Int64], offsets: [Int64]) throws -> [[Int]] {
        guard newCount >= 0, newCount <= history.count, multipliers.count == 3, sizes.count == 16,
            offsets.count == 16, sizes.allSatisfy({ $0 > 0 }), offsets.allSatisfy({ $0 >= 0 })
        else {
            throw EngineError("Invalid PLE hash geometry")
        }
        var lastEOS = -1
        var result: [[Int]] = []
        for (position, token) in history.enumerated() {
            var shifted = [token]
            for distance in 1...2 { shifted.append(position - lastEOS - 1 >= distance ? history[position - distance] : 248_044) }
            if position >= history.count - newCount {
                var mixed = shifted[0] &* multipliers[0]
                var row: [Int] = []
                for order in 1...2 {
                    mixed ^= shifted[order] &* multipliers[order]
                    for head in ((order - 1) * 8)..<(order * 8) {
                        var remainder = mixed % sizes[head]
                        if remainder < 0 { remainder += sizes[head] }
                        let (index, overflow) = remainder.addingReportingOverflow(offsets[head])
                        guard !overflow, index >= 0, index < 320_001_536 else { throw EngineError("PLE hash address exceeds table") }
                        row.append(Int(index))
                    }
                }
                result.append(row)
            }
            if token == 248_044 { lastEOS = position }
        }
        return result
    }
}

/// The host PLE (layer 1's per-token n-gram embedding), the reference DevicePLE is BITWISE against:
/// 16 table rows -> embedding [2560]; key = grouped RMS(key_proj(embedding)) [4 x 2560], value =
/// value_proj(embedding) [2560], query = grouped RMS(residual) [4 x 2560]; per stream a gate =
/// sigmoid(sign(d) * sqrt(max(|d|, 1e-6))) of d = dot(key, query) / sqrt(2560); gated = gate * value;
/// output = gated + SiLU(causal depthwise conv (4 taps, stride 3) of grouped RMS(gated)). The caller adds
/// the output to the residual. Rows are read from the store through a FIFO cache of packed 100-byte rows
/// (immutable model data, shared across requests).
public final class ReferencePLE {
    public var capture: ((String, [Float]) -> Void)?
    private let weights: ReferenceWeights
    private let reductions: PinnedReductions?
    private let multipliers, sizes, offsets: [Int64]
    private let capacity: Int
    private var rows: [Int: Data] = [:], fifo: [Int]
    private var next = 0
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation

    public init(weights: ReferenceWeights, ledger: MemoryLedger, capacity: Int = 4096, reductions: PinnedReductions? = nil) throws {
        guard capacity > 0, capacity <= 400_000 else { throw EngineError("Invalid PLE cache capacity") }
        self.weights = weights
        self.ledger = ledger
        self.reductions = reductions
        self.capacity = capacity
        let base = "model.layers.1.ple.ple_embedding."
        multipliers = try weights.integers(base + "layer_multipliers")
        sizes = try weights.integers(base + "ngram_heads_vocab_sizes")
        offsets = try weights.integers(base + "ngram_heads_offsets")
        reservation = try ledger.reserve(bytes: UInt64(capacity) * 512, category: "ple_compact_cache_and_metadata")
        fifo = [Int](repeating: -1, count: capacity)
    }
    deinit { try? ledger.release(reservation) }

    private func row(_ index: Int, batch: KnownPLERows?) throws -> [Float] {
        let data: Data
        if let hit = rows[index] {
            data = hit
        } else {
            let fresh: Data
            if let batch {
                fresh = try batch.consume(index)
            } else {
                var data = Data(count: 100)
                try data.withUnsafeMutableBytes { try weights.store.readPLERow(index, into: $0) }
                fresh = data
            }
            if fifo[next] >= 0 { rows.removeValue(forKey: fifo[next]) }
            rows[index] = fresh
            fifo[next] = index
            next = (next + 1) % capacity
            data = fresh
        }
        return try ReferenceWeights.dequantizedRow(
            words: ReferenceWeights.array(data.subdata(in: 0..<80), UInt32.self),
            scales: ReferenceWeights.array(data.subdata(in: 80..<90), UInt16.self),
            biases: ReferenceWeights.array(data.subdata(in: 90..<100), UInt16.self), groupSize: 32)
    }

    /// The 16 row indices per new token, as `call` computes them for the same history window.
    func rowIndices(history: [Int64], newCount: Int) throws -> [[Int]] {
        try PLEHash.rows(history: history, newCount: newCount, multipliers: multipliers, sizes: sizes, offsets: offsets)
    }

    /// Host trace/diagnostic captures keep the reference path.
    var hasCapture: Bool { capture != nil }

    private func groupNorm(_ x: [Float], name: String) throws -> [Float] {
        let values = try weights.tensor(name)
        return try reductions?.groupedRMS(x: x, weights: values, groupSize: 2560)
            ?? weights.kernels.rmsnorm(x: x, weights: values, groupSize: 2560)
    }

    /// `rowSource` (known rows only; `prefill_ple_overlap`): reads the FIFO misses from already-read scope rows
    /// instead of the store. The bytes are the same rows, so the result is unchanged.
    public func call(
        _ hidden: [Float], history: [Int64], newCount: Int, state: inout PLEState, knownRows: Bool = false,
        shouldContinue: () -> Bool = { true },
        rowSource: (@Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void)? = nil
    ) throws -> [Float] {
        let width = 10_240
        let base = "model.layers.1.ple"
        guard rowSource == nil || knownRows else { throw EngineError("A PLE row source requires known rows") }
        guard newCount > 0, newCount <= 256, hidden.count == newCount * width, state.convolution.count == 9 * width else {
            throw EngineError("Invalid reference PLE state or tile")
        }
        let ids = try PLEHash.rows(history: history, newCount: newCount, multipliers: multipliers, sizes: sizes, offsets: offsets)
        let batch: KnownPLERows?
        if knownRows {
            let prepared = try KnownPLERows(ids: ids, fifo: fifo, next: next, ledger: ledger)
            let store = weights.store
            let read: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = rowSource ?? { try store.readPLERow($0, into: $1) }
            try prepared.load(shouldContinue: shouldContinue, read: read)
            batch = prepared
        } else {
            batch = nil
        }
        var embedding: [Float] = []
        embedding.reserveCapacity(newCount * 2560)
        for position in ids { for index in position { embedding.append(contentsOf: try row(index, batch: batch)) } }
        try batch?.finishConsumption()
        capture?(base + ".embedding", embedding)
        let keyRaw = try weights.linear(embedding, name: base + ".key_proj")
        capture?(base + ".key_raw", keyRaw)
        let key = try groupNorm(keyRaw, name: base + ".norm_key.weight")
        let value = try weights.linear(embedding, name: base + ".value_proj")
        let query = try groupNorm(hidden, name: base + ".norm_query.weight")
        capture?(base + ".key", key)
        capture?(base + ".value", value)
        capture?(base + ".query", query)
        let pinnedDots = try reductions?.pleDot(key: key, query: query)
        var dots = capture == nil ? nil : [Float](repeating: 0, count: newCount * 4)
        var scaledDots = dots
        var gates = dots
        var gated = [Float](repeating: 0, count: hidden.count)
        for token in 0..<newCount {
            for branch in 0..<4 {
                let start = token * width + branch * 2560
                let reduced: Float
                if let pinnedDots {
                    reduced = pinnedDots[token * 4 + branch]
                } else {
                    var dot: Float = 0
                    for column in 0..<2560 { dot += LayerArithmetic.round(key[start + column] * query[start + column]) }
                    reduced = LayerArithmetic.round(dot)
                }
                // MLX scalar operands are converted to the array's BF16 dtype.
                let scaled = LayerArithmetic.round(reduced / LayerArithmetic.round(sqrtf(2560)))
                let magnitude = max(abs(scaled), LayerArithmetic.round(1e-6))
                let gate = weights.kernels.arithmetic.sigmoid(
                    LayerArithmetic.round(sqrtf(magnitude)) * (scaled < 0 ? -1 : (scaled > 0 ? 1 : 0)))
                dots?[token * 4 + branch] = reduced
                scaledDots?[token * 4 + branch] = scaled
                gates?[token * 4 + branch] = gate
                for column in 0..<2560 { gated[start + column] = LayerArithmetic.round(gate * value[token * 2560 + column]) }
            }
        }
        let normalized = try groupNorm(gated, name: base + ".norm_conv.weight")
        if let dots { capture?(base + ".dot", dots) }
        if let scaledDots { capture?(base + ".scaled", scaledDots) }
        if let gates { capture?(base + ".gate", gates) }
        capture?(base + ".gated", gated)
        capture?(base + ".normalized", normalized)
        let convolutionWeights = try weights.tensor(base + ".conv1d.weight")
        guard convolutionWeights.count == width * 4 else { throw EngineError("Invalid PLE convolution weights") }
        let full = state.convolution.map(BF16.decode) + normalized
        var output = gated
        var convRaw = capture == nil ? nil : [Float](repeating: 0, count: hidden.count)
        for token in 0..<newCount {
            for column in 0..<width {
                var sum: Float = 0
                for tap in 0..<4 { sum = sum.addingProduct(full[(token + tap * 3) * width + column], convolutionWeights[column * 4 + tap]) }
                convRaw?[token * width + column] = LayerArithmetic.round(sum)
                output[token * width + column] = LayerArithmetic.round(
                    gated[token * width + column] + weights.kernels.arithmetic.silu(LayerArithmetic.round(sum)))
            }
        }
        state.convolution = full.suffix(9 * width).map(BF16.encode)
        if let convRaw { capture?(base + ".conv_raw", convRaw) }
        capture?(base + ".output", output)
        return output
    }
}
