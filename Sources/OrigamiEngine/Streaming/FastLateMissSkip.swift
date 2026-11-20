import Foundation

/// Opt-in fast decode (NUMERIC, never exact), one of the quality dial's levels (`QualityDial`): the configuration may
/// only allow it (`fast.allow_opt_in` with `fast.late_miss_skip_tau`; `allow_fast` in the CLI settings), and a request
/// chooses a level with `"origami_mode"` or a `-<level>` model id, or gets it from the server's default level or a
/// level kept for its system prompt. A request whose level is exact runs on the unchanged path; a non-exact request
/// on a configuration without the allowance is refused (`notAllowed`), never run exact or fast silently.
///
/// Carried state: fast decode rows enter a persistent conversation under their own timeline epoch
/// (`timelineEpoch`); an exact request never continues from a timeline that holds them (`isExactTimeline`; saved
/// images keep them after `allow_fast` is turned off), so it re-reads a fast turn's tokens as exact prefill input, as it would any
/// client-supplied assistant turn. Prefill rows are always exact.
///
/// "Late" (the implemented definition): a routed expert of a decode layer whose record read has no outcome yet at
/// the layer's join point, i.e. right after the shared expert and the resident-at-request group were submitted and
/// immediately before the exact path would join the remaining reads (`ExpertDemandWindow.consumeTenReadyFirst`).
/// A late expert with normalized gate weight below τ contributes nothing; its read is not cancelled and still
/// completes into the cache. Records resident at request, or whose read finished by the join point, are never
/// skipped. The kept gates are rescaled to the original routed mass (the rule was fixed before any measurement).
///
/// Skipped reads keep running as demand jobs. The exact path's scheduler limits leave no room for them (its demand
/// reserve covers the current and the staged next layer), so fast mode adds `readReserve` demand records per drive
/// and a skip is admitted only while the skipped reads without an outcome stay within that reserve. Late records
/// beyond it are joined as in the exact path, the heaviest first, and counted (`capacityDeclines`).
package enum FastLateMissSkip {
    /// One layer's routed records: the most a single skip point can add.
    package static let readReserve = 10

    package struct Decision: Sendable {
        /// Skipped experts and their normalized gate weights (w / routed mass), in routed rank order.
        package let skipped: [Int], skippedWeights: [Float]
        /// The ten gates to apply in routed rank order: skipped ranks 0, kept ranks scaled by mass / kept mass.
        package let weights: [Float]
        /// Every routed record was late (or only zero-weight ones would remain): the heaviest late one is kept.
        package let keptHeaviestLate: Bool
        /// Late records under tau that were kept (joined) because the skipped-read reserve was full.
        package let declinedForCapacity: Int
    }

    /// `ids` and `weights`: the layer's routed experts and gates in rank order; `late`: experts late at the join point;
    /// `capacity`: how many may still be skipped within the read reserve (the lightest are skipped first).
    package static func decide(ids: [Int], weights: [Float], late: [Int], tau: Double, capacity: Int) throws -> Decision {
        let mass = weights.reduce(Float(0), +)
        guard ids.count == 10, Set(ids).count == 10, weights.count == 10, weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
            mass.isFinite, mass > 0, tau > 0, tau <= 1, Set(late).count == late.count, late.allSatisfy(ids.contains)
        else {
            throw EngineError("Fast late-miss skip requires ten finite routed gates, late routed experts and tau in (0, 1]")
        }
        var skip = ids.indices.filter { late.contains(ids[$0]) && Double(weights[$0] / mass) < tau }
        let declined = max(0, skip.count - max(0, capacity))
        if declined > 0 {
            let lightest = Set(
                skip.sorted { weights[$0] < weights[$1] || (weights[$0] == weights[$1] && $0 > $1) }.prefix(skip.count - declined))
            skip = skip.filter(lightest.contains)
        }
        let keptHeaviestLate = !skip.isEmpty && ids.indices.allSatisfy { skip.contains($0) || weights[$0] == 0 }
        if keptHeaviestLate, let heaviest = skip.max(by: { weights[$0] < weights[$1] }) { skip.removeAll { $0 == heaviest } }
        guard !skip.isEmpty else {
            return Decision(
                skipped: [], skippedWeights: [], weights: weights, keptHeaviestLate: keptHeaviestLate, declinedForCapacity: declined)
        }
        let kept = ids.indices.filter { !skip.contains($0) }.reduce(Float(0)) { $0 + weights[$1] }
        let scale = mass / kept
        return Decision(
            skipped: skip.map { ids[$0] }, skippedWeights: skip.map { weights[$0] / mass },
            weights: ids.indices.map { skip.contains($0) ? 0 : weights[$0] * scale }, keptHeaviestLate: keptHeaviestLate,
            declinedForCapacity: declined)
    }

    /// The refusal of a non-exact quality level on a configuration without `fast.allow_opt_in` (code `fast_not_allowed`).
    package static let notAllowed =
        "origami_mode refused: this server runs only exact decode "
        + "(origami config set allow_fast true turns the other quality levels on); the request was not run"

    /// The τ a request runs with: nil for exact (the unchanged path); for any other quality level the configuration's
    /// allowance (`allowedTau`), without which the request is refused.
    package static func requestTau(_ quality: QualityDial, allowedTau: Double?) throws -> Double? {
        guard !quality.isExact else { return nil }
        guard let allowedTau else { throw EngineError(notAllowed, code: "fast_not_allowed") }
        return allowedTau
    }

    /// The arithmetic epoch of a request's decode rows in a conversation timeline: the conversation's own for exact
    /// (unchanged segments), a distinct one per non-exact dial, so its state stays recognizable in any saved image.
    /// v1's late-miss skip keeps its epoch, so conversations saved in v1's fast mode still continue.
    package static func timelineEpoch(_ epoch: String, quality: QualityDial) -> String {
        quality.isExact ? epoch : epoch + (quality == .lateMissSkip ? ":fast-late-miss-skip-v1" : ":quality-" + quality.signature)
    }

    /// True when every decode row of `segments` carries the exact decode signature of `arithmeticEpoch` (prefill
    /// rows are always exact). An exact request reuses only such timelines.
    package static func isExactTimeline(_ segments: [CheckpointSegment], arithmeticEpoch: String) throws -> Bool {
        // shortcut: rehashes every decode row per check (O(decode rows): each exact disk restore, and in memory on
        // fast-allowing configurations only); memoize per start offset if long conversations show the cost.
        for segment in segments where segment.phase == .decode {
            let exact = try PrefillReadSchedule.decodeSegment(start: segment.start, arithmeticEpoch: arithmeticEpoch)
            guard exact.dispatchSHA256 == segment.dispatchSHA256 else { return false }
        }
        return true
    }
}
