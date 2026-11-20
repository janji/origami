import Foundation

/// A prediction is scored even when its read was declined, so a formerly poor rank can recover without probe I/O.
/// Only predictions of experts not in memory at the guess are scored: those are the ones a read is for, and resident
/// ones the router also picks would make a rank look better than its reads are. The serial expert owner updates these
/// tiny per-layer histories at the next real router result.
package struct FamilyPrefetchPolicy {
    package init() {}
    /// The hit rate a mature rank needs. Measured on both family models' decode (200 tokens): 0.6 cut DeepSeek's wasted
    /// reads 40 -> 24 GB and GLM's 42 -> 27 GB at the same speed as 0.5 with resident hits counted.
    static let confidence = 0.6
    package enum Decision: Equatable { case admit, confidence, queue, deadline }
    private struct Layer {
        var observations = 0
        var accuracy: [Double] = []
        var horizon = 0.0
    }
    private var layers: [Int: Layer] = [:]

    /// `absent`: the predicted experts not in memory at the guess (nil: all of them).
    package mutating func observe(layer: Int, predicted: [Int], needed: Set<Int>, absent: Set<Int>? = nil, seconds: Double) {
        var state = layers[layer] ?? Layer()
        let alpha = 1 / Double(min(16, state.observations + 1))
        if state.accuracy.count < predicted.count {
            state.accuracy += repeatElement(0.5, count: predicted.count - state.accuracy.count)
        }
        for (rank, expert) in predicted.enumerated() where absent?.contains(expert) ?? true {
            state.accuracy[rank] += alpha * ((needed.contains(expert) ? 1 : 0) - state.accuracy[rank])
        }
        if seconds.isFinite, seconds > 0 {
            state.horizon += alpha * (min(10, seconds) - state.horizon)
        }
        state.observations += 1
        layers[layer] = state
    }

    package func horizon(layer: Int) -> Double? {
        guard let state = layers[layer], state.observations >= 4, state.horizon > 0 else { return nil }
        return state.horizon
    }

    package func decide(
        layer: Int, rank: Int, queuedRecords: Int, maximumRecords: Int,
        queuedSeconds: Double, serviceSeconds: Double
    ) -> Decision {
        let state = layers[layer]
        // Bootstrap with at most two complete records; the real next router teaches all ranks without extra reads.
        let mature = (state?.observations ?? 0) >= 4
        guard queuedRecords < (mature ? max(1, maximumRecords) : min(2, max(1, maximumRecords))) else { return .queue }
        if mature, let state, rank < state.accuracy.count, state.accuracy[rank] < Self.confidence { return .confidence }
        // A read that begins before the router still saves the part it has done; one that would only begin after it saves nothing.
        if let horizon = horizon(layer: layer), queuedSeconds > horizon { return .deadline }
        return .admit
    }
}
