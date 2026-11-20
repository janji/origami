import Foundation

/// A request's sampling settings. `sanitized()` turns non-finite or out-of-range values into the neutral setting
/// (temperature 0 is greedy, top_p 1 and top_k 0 are off, min_p 0, no presence or frequency penalty) instead of failing.
public struct SamplingParameters: Codable, Sendable, Equatable {
    public var temperature: Float = 1
    public var topP: Float = 0.95
    public var topK: Int = 20
    public var minP: Float = 0
    public var presencePenalty: Float = 0
    /// OpenAI's `frequency_penalty`: subtracted from a token's logit once per time it was generated so far.
    public var frequencyPenalty: Float = 0
    public init() {}
    public func sanitized() -> Self {
        var p = self
        if !p.temperature.isFinite { p.temperature = 0 }
        p.temperature = max(0, p.temperature)
        if !p.topP.isFinite || p.topP <= 0 || p.topP > 1 { p.topP = 1 }
        if !p.minP.isFinite { p.minP = 0 }
        p.minP = min(1, max(0, p.minP))
        if !p.presencePenalty.isFinite { p.presencePenalty = 0 }
        if !p.frequencyPenalty.isFinite { p.frequencyPenalty = 0 }
        p.topK = max(0, p.topK)
        return p
    }
}

/// SplitMix64: the sampler's only randomness, one 53-bit uniform draw per call. The state is the whole generator,
/// so a saved sampler (Codable) replays the same draws. Seed 0 becomes 0xDEADBEEF (the two give one stream).
public struct SplitMixStream: Codable, Sendable {
    public var state: UInt64
    public init(seed: UInt64 = 0x9E37_79B9_7F4A_7C15) { state = seed == 0 ? 0xDEAD_BEEF : seed }
    public static func mix(_ input: UInt64) -> UInt64 {
        var value = input &+ 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
    public mutating func uniform() -> Double {
        state = Self.mix(state &+ 1)
        return Double(state >> 11) / Double(UInt64(1) << 53)
    }
}

/// The sampler the server runs on the CPU for every reply token, with the pinned SlotStream sampling semantics
/// (MIT notice in THIRD_PARTY_NOTICES.md): presence and frequency penalties, temperature, top-k (threshold ties kept), softmax,
/// top-p, min-p, then one inverse-CDF draw. Temperature 0 is argmax (lowest index on ties) and draws nothing,
/// so a temperature-0 reply depends only on the logits and the penalties; otherwise the seed fixes it too.
/// `generated` counts the reply's tokens so far (token ID: times generated).
public struct ReferenceSampler: Codable, Sendable {
    public var random: SplitMixStream
    public init(seed: UInt64 = 0x9E37_79B9_7F4A_7C15) { random = SplitMixStream(seed: seed) }

    private static func argmax(_ values: [Float]) throws -> Int {
        guard !values.isEmpty else { throw EngineError("Empty logits") }
        return values.withUnsafeBufferPointer { v in
            var best = 0
            for i in 1..<v.count where v[i] > v[best] { best = i }
            return best
        }
    }

    /// The logits after the penalties, checked.
    private static func penalized(_ logits: [Float], _ p: SamplingParameters, _ generated: [Int: Int]) throws -> [Float] {
        guard !logits.isEmpty, logits.count <= 248_320,
            generated.allSatisfy({ logits.indices.contains($0.key) && $0.value > 0 })
        else { throw EngineError("Invalid sampler vocabulary/history") }
        // A NaN logit is an engine fault; it would also break the orderings below.
        guard !logits.contains(where: \.isNaN) else { throw EngineError("The model produced a NaN logit") }
        var values = logits
        if p.presencePenalty != 0 { for index in generated.keys { values[index] -= p.presencePenalty } }
        if p.frequencyPenalty != 0 { for (index, count) in generated { values[index] -= p.frequencyPenalty * Float(count) } }
        return values
    }

    /// Includes top-k threshold ties, as in the baseline; never arbitrarily drops
    /// equal-threshold candidates just to force exactly k surviving token IDs.
    public static func masses(logits: [Float], parameters: SamplingParameters, generated: [Int: Int] = [:]) throws -> [Float] {
        let p = parameters.sanitized()
        var values = try penalized(logits, p, generated)
        // The greedy token, and the fallback when a tiny temperature overflows the scaled logits.
        let best = try argmax(values)
        if p.temperature == 0 {
            var result = [Float](repeating: 0, count: values.count)
            result[best] = 1
            return result
        }
        for i in values.indices { values[i] /= p.temperature }
        if p.topK > 0 && p.topK < values.count {
            let threshold = kthLargest(values, p.topK)
            for i in values.indices where values[i] < threshold { values[i] = -.infinity }
        }
        let maximum = values[try argmax(values)]
        var probabilities = values.map { expf($0 - maximum) }
        let total = probabilities.reduce(Float(0), +)
        guard total.isFinite && total > 0 else {
            var result = [Float](repeating: 0, count: logits.count)
            result[best] = 1
            return result
        }
        for i in probabilities.indices { probabilities[i] /= total }
        if p.topP < 1 {
            // Zero masses add nothing to the running sum and sort after every positive one, so only positive ones are ordered.
            let sorted = probabilities.indices.filter { probabilities[$0] > 0 }.sorted { a, b in
                probabilities[a] != probabilities[b] ? probabilities[a] > probabilities[b] : a < b
            }
            var cumulative: Float = 0
            for i in sorted {
                let current = probabilities[i]
                cumulative += current
                if cumulative - current >= p.topP { probabilities[i] = 0 }
            }
        }
        if p.minP > 0 {
            let cutoff = probabilities.max()! * p.minP
            for i in probabilities.indices where probabilities[i] < cutoff { probabilities[i] = 0 }
        }
        return probabilities
    }

    public mutating func next(logits: [Float], parameters: SamplingParameters, generated: [Int: Int] = [:]) throws -> Int {
        // Greedy: the one-hot masses' argmax is the penalized logits' argmax, so skip building them.
        let p = parameters.sanitized()
        if p.temperature == 0 { return try Self.argmax(Self.penalized(logits, p, generated)) }
        let probabilities = try Self.masses(logits: logits, parameters: parameters, generated: generated)
        let draw = Float(random.uniform())
        var cumulative: Float = 0
        var cdf: [Float] = []
        cdf.reserveCapacity(probabilities.count)
        for p in probabilities {
            cumulative += p
            cdf.append(cumulative)
        }
        guard cumulative.isFinite && cumulative > 0 else { return try Self.argmax(probabilities) }
        let threshold = draw * cumulative
        let first = draw == 0 ? cdf.firstIndex(where: { $0 > 0 }) : cdf.firstIndex(where: { $0 >= threshold })
        return first ?? cdf.count - 1
    }

    /// Speculative check of a one-token greedy draft (draft distribution one-hot at `draft`): accepted with probability
    /// p(draft); otherwise a token from p with the draft removed, so the output keeps p's distribution. At temperature 0
    /// the draft is accepted exactly when it is the argmax.
    public mutating func verify(draft: Int, logits: [Float], parameters: SamplingParameters, generated: [Int: Int] = [:]) throws -> (
        accepted: Bool, token: Int
    ) {
        var probabilities = try Self.masses(logits: logits, parameters: parameters, generated: generated)
        guard probabilities.indices.contains(draft) else { throw EngineError("Draft outside the vocabulary") }
        if parameters.sanitized().temperature == 0 {
            let best = try Self.argmax(probabilities)
            return (best == draft, best)
        }
        let total = probabilities.reduce(Float(0), +)
        if Float(random.uniform()) * total < probabilities[draft] { return (true, draft) }
        probabilities[draft] = 0
        let rest = probabilities.reduce(Float(0), +)
        guard rest > 0 else { return (true, draft) }
        let threshold = Float(random.uniform()) * rest
        var cumulative: Float = 0
        for (index, p) in probabilities.enumerated() where p > 0 {
            cumulative += p
            if cumulative >= threshold { return (false, index) }
        }
        return (false, probabilities.lastIndex(where: { $0 > 0 })!)
    }

    /// The k-th largest value, which a full (value desc, index asc) sort places at k - 1, without sorting the
    /// vocabulary: only `values[i] < threshold` reads it, so the top-k cut is the same bit for bit.
    private static func kthLargest(_ values: [Float], _ k: Int) -> Float {
        guard k <= 64 else { return values.sorted(by: >)[k - 1] }
        var top: [Float] = []  // the k largest so far, descending
        top.reserveCapacity(k + 1)
        for v in values where top.count < k || v > top[k - 1] {
            top.insert(v, at: top.firstIndex { $0 < v } ?? top.count)
            if top.count > k { top.removeLast() }
        }
        return top[k - 1]
    }

    package static func syntheticLogits(vocabulary: Int, seed: UInt64) throws -> [Float] {
        guard (1...248_320).contains(vocabulary) else { throw EngineError("Invalid synthetic vocabulary") }
        var state = seed
        return (0..<vocabulary).map { _ in
            state = SplitMixStream.mix(state)
            return Float(state >> 40) / Float(1 << 24) * 8 - 4
        }
    }
}

/// General speculative-sampling algebra (acceptance min(1, p/q) and the residual max(p - q, 0)) for any draft
/// distribution. Only engine-checks use it; serving verifies one-hot greedy drafts with `ReferenceSampler.verify`.
package enum SpeculativeCorrection {
    private static func normalized(_ masses: [Double]) throws -> [Double] {
        guard !masses.isEmpty, masses.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw EngineError("Invalid speculative masses") }
        let total = masses.reduce(0, +)
        guard total.isFinite && total > 0 else { throw EngineError("Empty speculative distribution") }
        return masses.map { $0 / total }
    }
    package static func acceptance(token: Int, target: [Double], draft: [Double]) throws -> Double {
        guard target.count == draft.count, target.indices.contains(token) else { throw EngineError("Mismatched speculative distributions") }
        let p = try normalized(target)
        let q = try normalized(draft)
        guard q[token] > 0 else { throw EngineError("Draft proposed a zero-probability token") }
        return min(1, p[token] / q[token])
    }
    package static func residual(target: [Double], draft: [Double]) throws -> [Double] {
        guard target.count == draft.count else { throw EngineError("Mismatched speculative distributions") }
        let p = try normalized(target)
        let q = try normalized(draft)
        return try normalized(zip(p, q).map { max($0 - $1, 0) })
    }
}
