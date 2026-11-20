import Foundation
import OrigamiEngine

func upgradePolicyChecks() -> [(String, () throws -> Void)] {
    [
        (
            "upgrade_request_policy_boundaries",
            {
                func require(_ value: Bool, _ message: String) throws {
                    if !value { throw EngineError(message) }
                }
                func parse(_ extra: [String: Any]) throws -> ServingRequest {
                    var body: [String: Any] = [
                        "model": "qwen3.8-flash-next",
                        "messages": [["role": "user", "content": "hi"]], "max_tokens": 8,
                    ]
                    body.merge(extra) { $1 }
                    return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
                }
                func refuses(_ extra: [String: Any]) -> Bool {
                    do {
                        _ = try parse(extra)
                        return false
                    } catch { return true }
                }
                let enabled = try parse(["origami_mode": "fast", "origami_tune": ["delayed_experts": true]])
                try require(
                    enabled.quality.name == "faster" && enabled.quality.delayedExperts != nil,
                    "Delayed policy lost the mode alias")
                for early in [false, true] {
                    let scheduled = try parse([
                        "origami_mode": "faster",
                        "origami_tune": [
                            "delayed_experts": true, "family_early_delayed_reads": early,
                        ],
                    ])
                    try require(
                        scheduled.tune?.familyEarlyDelayedReads == early && scheduled.quality == enabled.quality,
                        "Earlier delayed reads were rejected or changed the declared delayed policy")
                }
                try require(
                    refuses([
                        "origami_mode": "faster",
                        "origami_tune": [
                            "delayed_experts": true, "family_early_delayed_reads": 1,
                        ],
                    ]), "Numeric early-read flag was accepted")
                let baseline = QualityDial.level("faster")!
                try require(
                    FastLateMissSkip.timelineEpoch("x", quality: enabled.quality)
                        != FastLateMissSkip.timelineEpoch("x", quality: baseline), "Delayed policy reused the baseline timeline")
                try require(refuses(["origami_tune": ["delayed_experts": true]]), "Exact deferral was accepted")
                try require(
                    refuses(["origami_mode": "fast", "origami_tune": ["family_spin": true]]),
                    "Unqualified approximate timing combination was accepted")
                let isolated = try parse(["origami_mode": "fast", "origami_tune": ["qwen_prefill_scope": 256]])
                try require(
                    isolated.tune?.qwenPrefillScope == 256 && isolated.quality == baseline,
                    "Exact prefill isolation changed approximate decode arithmetic")
                let invalidPolicies: [Any] = [0, "yes", ["max_mass": 0.2], ["max_experts": 2], ["version": true], ["unknown": 1]]
                for bad in invalidPolicies {
                    try require(
                        refuses(["origami_mode": "fast", "origami_tune": ["delayed_experts": bad]]),
                        "Malformed delayed policy accepted")
                }
                let disabled = try parse(["origami_mode": "fast", "origami_tune": ["delayed_experts": false]])
                try require(disabled.quality == baseline, "Disabled delayed policy altered the mode")
                // Retired: accepted with any value and ignored, in any mode.
                for (mode, value) in [("fast", "margin" as Any), ("exact", "invalid")] {
                    let request = try parse(["origami_mode": mode, "origami_tune": ["expert_admission": value]])
                    try require(
                        request.tune == TuneOverrides() && (mode == "exact" || request.quality == baseline),
                        "Retired admission field was not ignored")
                }
                try require(refuses(["origami_tune": ["qwen_pair_probe": true]]), "Pair probe without teacher tokens accepted")
                let probe = try parse(["origami_tune": ["qwen_pair_probe": true], "origami_teacher_tokens": [1, 2, 3]])
                try require(probe.tune?.qwenPairProbe == true && probe.noStore, "Diagnostic probe persisted teacher state")
                try require(
                    refuses(["origami_teacher_tokens": [NSNumber(value: UInt64.max)]]),
                    "Out-of-range teacher ID wrapped to an integer")
                try require(
                    refuses(["origami_teacher_tokens": [NSDecimalNumber(string: "184467440737095516160000")]]),
                    "Oversized decimal teacher ID accepted")
            }
        ),
        (
            "qwen_verifier_probe_economics",
            {
                let result = try QwenVerifierCostProbe.summarize(plainPairs: [2, 2], verifyPairs: [2.5, 2.5])
                guard result["plain_step_seconds"] == 1, result["verify_to_plain_ratio"] == 2.5,
                    result["ideal_drafter_minimum_acceptance"] == 1.25
                else {
                    throw EngineError("Verifier lower bound hides an uneconomic draft path")
                }
                for pair in [0.0, -.infinity, .nan] {
                    do {
                        _ = try QwenVerifierCostProbe.summarize(plainPairs: [2, 2], verifyPairs: [pair, 2])
                    } catch { continue }
                    throw EngineError("Invalid verifier timing accepted")
                }
            }
        ),
    ]
}
