import Foundation

/// Which routed experts one token runs in a family engine, from the router's probabilities: the trained top-k by
/// probability plus the selection bias (DeepSeek's noaux_tc), weights normalised over the chosen ones and scaled, then
/// the quality dial's knobs (`QualityDial`). Exact (the default dial) is the trained rule. A family engine reads every
/// expert a layer routes before the layer runs, so every expert not in memory is a late miss: `skip` drops them (the
/// heaviest is kept), `substitute` runs the resident expert with the highest selection score in place of each.
package enum FamilyRouting {
    /// Whether `select` needs the layer's experts in memory (the miss knobs).
    package static func needsResidency(_ dial: QualityDial) -> Bool { dial.tailSkipMissesOnly || dial.lateMiss != .wait }
    /// `probs` weigh, `probs + bias` select; `k` is the trained count. `resident` is the layer's experts in memory
    /// (needed by the miss knobs, `needsResidency`). `sum` turns the chosen probabilities' sum into the normaliser (GLM clamps it,
    /// DeepSeek adds 1e-20). Returns (expert, weight) in selection order.
    package static func select(
        probs: [Float], bias: [Float], k: Int, dial: QualityDial, resident: Set<Int>?, normalize: Bool, scale: Float,
        sum normaliser: (Float) -> Float
    ) -> [(expert: Int, weight: Float)] {
        // Deferral replaces skipping; retain the trained original gates for both immediate and delayed work.
        if dial.delayedExperts != nil {
            return select(probs: probs, bias: bias, k: k, dial: .exact, resident: nil, normalize: normalize, scale: scale, sum: normaliser)
        }
        let n = probs.count
        let count = max(1, min(k, dial.expertsPerToken ?? k))
        let scores = bias.isEmpty ? probs : (0..<n).map { probs[$0] + bias[$0] }
        // Score descending, the lower index first on ties: the normaliser sums the chosen in this order.
        func before(_ a: Int, _ b: Int) -> Bool { scores[a] != scores[b] ? scores[a] > scores[b] : a < b }
        // Substitution needs the whole order; otherwise the first `count` of it, kept sorted as each index arrives.
        let order: [Int]
        if dial.lateMiss == .substitute, resident != nil {
            order = (0..<n).sorted(by: before)
        } else {
            var top: [Int] = []
            top.reserveCapacity(count + 1)
            for i in 0..<n where top.count < count || before(i, top[count - 1]) {
                var at = top.count
                while at > 0 && before(i, top[at - 1]) { at -= 1 }
                top.insert(i, at: at)
                if top.count > count { top.removeLast() }
            }
            order = top
        }
        var chosen = Array(order.prefix(count))
        if dial.lateMiss == .substitute, let resident {
            var taken = Set(chosen)
            let spare = order.filter { resident.contains($0) && !taken.contains($0) }
            var next = 0
            for (slot, expert) in chosen.enumerated() where !resident.contains(expert) && next < spare.count {
                taken.remove(expert)
                chosen[slot] = spare[next]
                taken.insert(spare[next])
                next += 1
            }
        }
        if dial.lateMiss == .skip, let resident, chosen.count > 1 {
            let heaviest = chosen.max { probs[$0] < probs[$1] }!
            chosen = chosen.filter { $0 == heaviest || resident.contains($0) }
        }
        if let tau = dial.tailSkip.map(Float.init), chosen.count > 1 {
            let mass = chosen.reduce(Float(0)) { $0 + probs[$1] }
            let heaviest = chosen.max { probs[$0] < probs[$1] }!
            chosen = chosen.filter { e in
                e == heaviest || probs[e] / mass >= tau || (dial.tailSkipMissesOnly && (resident?.contains(e) ?? true))
            }
        }
        let total = normaliser(chosen.reduce(Float(0)) { $0 + probs[$1] })
        return chosen.map { ($0, (normalize ? probs[$0] / total : probs[$0]) * scale) }
    }
}
