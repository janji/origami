import Foundation

/// Local, explicitly requested diagnostic. A target-only pair is a lower bound
/// on draft plus verification cost; it cannot establish a drafter's acceptance.
package enum QwenVerifierCostProbe {
    package static func summarize(plainPairs: [Double], verifyPairs: [Double]) throws -> [String: Double] {
        guard plainPairs.count == 2, verifyPairs.count == 2,
            (plainPairs + verifyPairs).allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw EngineError("Verifier probe requires two positive samples per arm")
        }
        let plain = (plainPairs[0] + plainPairs[1]) / 4
        let pair = (verifyPairs[0] + verifyPairs[1]) / 2
        return [
            "plain_pair_1_seconds": plainPairs[0], "plain_pair_2_seconds": plainPairs[1],
            "verify_pair_1_seconds": verifyPairs[0], "verify_pair_2_seconds": verifyPairs[1],
            "plain_step_seconds": plain, "verify_pair_seconds": pair,
            "verify_to_plain_ratio": pair / plain,
            "ideal_drafter_minimum_acceptance": pair / (2 * plain),
        ]
    }
}

extension SerialExecutor {
    /// Restore the same live state after every arm. Cache residency is intentionally
    /// uncontrolled and disclosed by the caller; this is a bounded feasibility screen.
    package func probeVerifierCost(tokens: [Int], state: SequenceState) throws -> [String: Double] {
        guard tokens.count == 2, !tokens.contains(where: TokenizerStack.eos.contains),
            state.capacity - state.tokenCount >= 2
        else {
            throw EngineError("Verifier probe needs two ordinary teacher tokens and admitted state capacity", code: "invalid_configuration")
        }
        let snapshot = pairSnapshot(state)
        let pending = state.emittedButUnconsumed
        func restoreOriginal() throws {
            try restore(snapshot, into: state)
            state.emittedButUnconsumed = pending
        }
        func run(pair: Bool) throws -> Double {
            guard shouldContinue() else { throw IOOperationCancelled() }
            let start = ProcessInfo.processInfo.systemUptime
            do {
                if pair {
                    _ = try forwardDecodePair(tokens, state: state)
                } else {
                    _ = try forward([tokens[0]], state: state)
                    _ = try forward([tokens[1]], state: state)
                }
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                guard state.tokenCount == snapshot.tokenCount + 2 else {
                    throw EngineError("Verifier probe consumed a different token count")
                }
                try restoreOriginal()
                return elapsed
            } catch {
                try restoreOriginal()
                throw error
            }
        }
        // Allocate the pair workspace and warm both paths before the balanced block.
        let warmPair = try run(pair: true)
        let warmPlain = try run(pair: false)
        let a = try run(pair: false)
        let b = try run(pair: true)
        let b2 = try run(pair: true)
        let a2 = try run(pair: false)
        var result = try QwenVerifierCostProbe.summarize(plainPairs: [a, a2], verifyPairs: [b, b2])
        result["warm_verify_pair_seconds"] = warmPair
        result["warm_plain_pair_seconds"] = warmPlain
        return result
    }
}
