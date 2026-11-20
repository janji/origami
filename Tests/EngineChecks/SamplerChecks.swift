import Foundation
import OrigamiEngine

func samplerGoldenMatchesPinnedBaseline() throws {
    let path = CheckFixtures.file("sampler/sampler-golden.json")
    let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
    for item in fixture["cases"] as! [[String: Any]] {
        var parameters = SamplingParameters()
        parameters.temperature = (item["temperature"] as! NSNumber).floatValue
        parameters.topP = (item["top_p"] as! NSNumber).floatValue
        parameters.topK = item["top_k"] as! Int
        parameters.minP = (item["min_p"] as! NSNumber).floatValue
        parameters.presencePenalty = (item["presence_penalty"] as! NSNumber).floatValue
        var sampler = ReferenceSampler(seed: (item["seed"] as! NSNumber).uint64Value)
        let logits = try ReferenceSampler.syntheticLogits(
            vocabulary: item["vocab"] as! Int, seed: (item["logit_seed"] as! NSNumber).uint64Value)
        var generated: [Int: Int] = [:]
        for expected in item["tokens"] as! [Int] {
            let actual = try sampler.next(logits: logits, parameters: parameters, generated: generated)
            guard actual == expected else {
                throw EngineError("Sampler baseline mismatch in \(item["name"]!): got \(actual), expected \(expected)")
            }
            if item["accumulate"] as! Bool { generated[actual, default: 0] += 1 }
        }
    }
}

func samplerThresholdTiesAndCheckpointReplay() throws {
    var params = SamplingParameters()
    params.topK = 1
    params.topP = 1
    let masses = try ReferenceSampler.masses(logits: [2, 2, -5], parameters: params)
    guard masses == [0.5, 0.5, 0] else { throw EngineError("Top-k threshold tie was lost") }
    var sampler = ReferenceSampler(seed: 73)
    let saved = try JSONEncoder().encode(sampler)
    let logits = try ReferenceSampler.syntheticLogits(vocabulary: 1024, seed: 99)
    let first = try (0..<64).map { _ in try sampler.next(logits: logits, parameters: params) }
    var restored = try JSONDecoder().decode(ReferenceSampler.self, from: saved)
    let replay = try (0..<64).map { _ in try restored.next(logits: logits, parameters: params) }
    guard first == replay, sampler.random.state == restored.random.state else {
        throw EngineError("Sampler frontier replay changed outputs")
    }
    let before = sampler.random.state
    params.temperature = 0
    _ = try sampler.next(logits: logits, parameters: params)
    guard sampler.random.state == before else { throw EngineError("Greedy sampler consumed random state") }
}

/// Greedy `next` skips the one-hot masses: it must pick what the full path picks (lowest index on ties), with
/// penalties and masked (-inf) entries, and refuse NaN logits the same way.
func samplerGreedyMatchesFullPath() throws {
    var random = SplitMixStream(seed: 11)
    for trial in 0..<200 {
        let vocabulary = trial < 100 ? 1 + Int(random.uniform() * 64) : 248_320
        // Few distinct values, so ties are common; some masked as JSON mode does.
        var logits = (0..<vocabulary).map { _ in Float(Int(random.uniform() * 6)) - 3 }
        for i in logits.indices where random.uniform() < 0.2 { logits[i] = -.infinity }
        var parameters = SamplingParameters()
        parameters.temperature = 0
        if trial % 3 == 1 { parameters.presencePenalty = 1 }
        if trial % 3 == 2 { parameters.frequencyPenalty = 0.5 }
        var generated: [Int: Int] = [:]
        for _ in 0..<(trial % 4) { generated[Int(random.uniform() * Double(vocabulary)), default: 0] += 1 }
        if trial % 10 == 9 { logits[Int(random.uniform() * Double(vocabulary))] = .nan }
        let full = Result { try ReferenceSampler.masses(logits: logits, parameters: parameters, generated: generated) }
        var sampler = ReferenceSampler(seed: 5)
        let quick = Result { try sampler.next(logits: logits, parameters: parameters, generated: generated) }
        switch (full, quick) {
        case (.success(let masses), .success(let token)):
            guard masses[token] == 1, masses.firstIndex(of: 1) == token else {
                throw EngineError("Greedy pick \(token) differs from the full path in trial \(trial)")
            }
        case (.failure(let a), .failure(let b)):
            guard String(describing: a) == String(describing: b) else { throw EngineError("Greedy refusal differs: \(a) / \(b)") }
        default: throw EngineError("Greedy path and full path disagree on refusing trial \(trial)")
        }
    }
}

func speculativeCorrectionConservesTargetDistribution() throws {
    let target = [0.1, 0.4, 0.5]
    let draft = [0.6, 0.3, 0.1]
    let accepted = try target.indices.map { draft[$0] * (try SpeculativeCorrection.acceptance(token: $0, target: target, draft: draft)) }
    let rejectedMass = 1 - accepted.reduce(0, +)
    let corrected = try SpeculativeCorrection.residual(target: target, draft: draft)
    for i in target.indices {
        guard abs(accepted[i] + rejectedMass * corrected[i] - target[i]) < 1e-14 else {
            throw EngineError("Speculative correction changed the target distribution")
        }
    }
    for i in target.indices {
        guard try SpeculativeCorrection.acceptance(token: i, target: target, draft: target) == 1 else {
            throw EngineError("Identical draft rejected")
        }
    }
}

/// The top-k and top-p cuts as 2.0.4 made them, by sorting the whole vocabulary (index order on ties); the fallback keeps
/// the penalties. The served sampler, which finds the k-th value and orders only positive masses, must match it.
private func fullSortMasses(_ logits: [Float], _ parameters: SamplingParameters, _ generated: [Int: Int]) -> [Float] {
    let p = parameters.sanitized()
    var values = logits
    if p.presencePenalty != 0 { for index in generated.keys { values[index] -= p.presencePenalty } }
    if p.frequencyPenalty != 0 { for (index, count) in generated { values[index] -= p.frequencyPenalty * Float(count) } }
    func argmax(_ v: [Float]) -> Int { v.indices.reduce(0) { v[$1] > v[$0] ? $1 : $0 } }
    var oneHot = [Float](repeating: 0, count: values.count)
    oneHot[argmax(values)] = 1
    if p.temperature == 0 { return oneHot }
    for i in values.indices { values[i] /= p.temperature }
    let order = values.indices.sorted { a, b in values[a] != values[b] ? values[a] > values[b] : a < b }
    if p.topK > 0 && p.topK < values.count {
        let threshold = values[order[p.topK - 1]]
        for i in values.indices where values[i] < threshold { values[i] = -.infinity }
    }
    let maximum = values[argmax(values)]
    var probabilities = values.map { expf($0 - maximum) }
    let total = probabilities.reduce(Float(0), +)
    guard total.isFinite && total > 0 else { return oneHot }
    for i in probabilities.indices { probabilities[i] /= total }
    if p.topP < 1 {
        let sorted = probabilities.indices.sorted { a, b in
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

/// Random logits (with ties, signed zeros, underflowing masses, ±inf and overflow) and parameters at their edges: the
/// sampler's masses equal the full-sort ones bit for bit, and its draws and generator state follow them for every seed.
func samplerMatchesFullSortReference() throws {
    var generator = SplitMixStream(seed: 2026)
    func random(_ n: Int) -> Int { Int(generator.uniform() * Double(n)) }
    func pick<T>(_ options: [T]) -> T { options[random(options.count)] }
    func logits(_ count: Int, _ kind: Int) -> [Float] {
        (0..<count).map { _ -> Float in
            switch kind {
            case 0: return Float(generator.uniform() * 16 - 8)
            case 1: return Float(random(7) - 3) / 2
            case 2: return 1.25
            case 3: return pick([-0.0, 0.0, 1, -1])
            case 4: return Float(generator.uniform() * 2e4 - 1e4)
            case 5: return random(50) == 0 ? pick([3e38, -3e38, -.infinity]) : Float(generator.uniform() * 10 - 5)
            case 6: return random(400) == 0 ? .infinity : Float(random(5))
            default: return Float(random(64)) / 8 - 4
            }
        }
    }
    let shapes = (0..<800).map { _ in (pick([1, 2, 3, 5, 17, 64, 65, 100, 256, 1000]), random(8)) } + [(4_096, 0), (4_096, 7)]
    for (vocabulary, kind) in shapes {
        let values = logits(vocabulary, kind)
        var p = SamplingParameters()
        p.temperature = pick([0, 1e-38, 1e-6, 0.05, 0.2, 0.7, 1, 1.5, 100, -1, .nan, .infinity])
        p.topP = pick([1, 0.95, 0.8, 0.5, 0.1, 1e-7, 0, 1.5, .nan, 0.99999])
        p.topK = pick([0, 1, 2, 3, 20, 63, 64, 65, 100, vocabulary - 1, vocabulary, vocabulary + 1, -5])
        p.minP = pick([0, 0, 0.02, 0.5, 1, -1, 2, .nan])
        p.presencePenalty = pick([0, 0, 1.5, -2, 1e30, .infinity, .nan])
        p.frequencyPenalty = pick([0, 0, 0, 0.5, -2, 1e30, .infinity, .nan])
        var generated: [Int: Int] = [:]
        for _ in 0..<random(vocabulary / 4 + 1) { generated[random(vocabulary), default: 0] += 1 }
        let seed: UInt64 = pick([0, 1, 7, 0xDEAD_BEEF, .max, UInt64(random(1 << 30)) << 34 | UInt64(random(1 << 30))])
        var sampler = ReferenceSampler(seed: seed)
        var reference = SplitMixStream(seed: seed)
        for _ in 0..<6 {
            let expected = fullSortMasses(values, p, generated)
            guard
                try ReferenceSampler.masses(logits: values, parameters: p, generated: generated).map(\.bitPattern)
                    == expected.map(\.bitPattern)
            else {
                throw EngineError("Sampler masses differ from the full sort: vocabulary \(vocabulary), kind \(kind), \(p)")
            }
            var token = expected.indices.reduce(0) { expected[$1] > expected[$0] ? $1 : $0 }
            if p.sanitized().temperature != 0 {
                let draw = Float(reference.uniform())
                var cumulative: Float = 0
                let cdf = expected.map {
                    cumulative += $0
                    return cumulative
                }
                token = (draw == 0 ? cdf.firstIndex { $0 > 0 } : cdf.firstIndex { $0 >= draw * cumulative }) ?? cdf.count - 1
            }
            guard try sampler.next(logits: values, parameters: p, generated: generated) == token, sampler.random.state == reference.state
            else {
                throw EngineError("Sampler draw differs from the full sort: vocabulary \(vocabulary), kind \(kind), seed \(seed), \(p)")
            }
            generated[token, default: 0] += 1
        }
    }
}

/// When a tiny temperature overflows the scaled logits the sampler falls back to the greedy token with the presence
/// penalty applied, and a NaN logit is refused at any temperature.
func samplerFallbackKeepsPenaltyAndRefusesNaN() throws {
    var parameters = SamplingParameters()
    parameters.temperature = 1e-38
    parameters.presencePenalty = 1.5  // 3.5 / 1e-38 overflows to infinity
    var sampler = ReferenceSampler(seed: 5)
    guard try ReferenceSampler.masses(logits: [1, 4, 3.5], parameters: parameters, generated: [1: 1]) == [0, 0, 1],
        try sampler.next(logits: [1, 4, 3.5], parameters: parameters, generated: [1: 1]) == 2
    else {
        throw EngineError("The overflow fallback dropped the presence penalty")
    }
    for temperature: Float in [0, 1] {
        parameters.temperature = temperature
        do {
            _ = try ReferenceSampler.masses(logits: [.nan, 1, 2], parameters: parameters)
            throw EngineError("A NaN logit was sampled")
        } catch let error as EngineError where error.description.contains("NaN logit") {}
    }
}

/// OpenAI's frequency penalty, by hand: token 1 generated 3 times and token 2 once, presence 0.5 and frequency 0.25
/// make the logits [2, 3 - 0.5 - 0.75, 2.9 - 0.5 - 0.25] = [2, 1.75, 2.15], so greedy picks token 2 (not 1) and
/// temperature 1 gives softmax masses. A zero or absent penalty leaves masses and draws the same bit for bit.
func samplerFrequencyPenaltyByHand() throws {
    var p = SamplingParameters()
    (p.temperature, p.topK, p.topP, p.presencePenalty, p.frequencyPenalty) = (0, 0, 1, 0.5, 0.25)
    let logits: [Float] = [2, 3, 2.9]
    let counts = [1: 3, 2: 1]
    guard try ReferenceSampler.masses(logits: logits, parameters: p, generated: counts) == [0, 0, 1] else {
        throw EngineError("The frequency penalty did not move the greedy pick to token 2")
    }
    p.temperature = 1
    let penalized: [Float] = [2, 1.75, 2.15]
    let total = penalized.map { expf($0 - 2.15) }.reduce(0, +)
    let expected = penalized.map { expf($0 - 2.15) / total }
    let masses = try ReferenceSampler.masses(logits: logits, parameters: p, generated: counts)
    guard zip(masses, expected).allSatisfy({ abs($0 - $1) < 1e-6 }) else {
        throw EngineError("Frequency-penalty masses \(masses) differ from the hand-computed \(expected)")
    }
    var off = p
    off.frequencyPenalty = 0
    var unset = SamplingParameters()
    (unset.temperature, unset.topK, unset.topP, unset.presencePenalty) = (1, 0, 1, 0.5)
    let vocabulary = try ReferenceSampler.syntheticLogits(vocabulary: 4096, seed: 11)
    let history = [7: 4, 99: 1, 4095: 2]
    var a = ReferenceSampler(seed: 3)
    var b = ReferenceSampler(seed: 3)
    for _ in 0..<32 {
        guard
            try a.next(logits: vocabulary, parameters: off, generated: history)
                == b.next(logits: vocabulary, parameters: unset, generated: history),
            try ReferenceSampler.masses(logits: vocabulary, parameters: off, generated: history).map(\.bitPattern)
                == ReferenceSampler.masses(logits: vocabulary, parameters: unset, generated: history).map(\.bitPattern)
        else { throw EngineError("frequency_penalty 0 changed the sampler's output") }
    }
}
