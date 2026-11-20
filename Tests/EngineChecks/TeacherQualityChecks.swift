import Foundation
import OrigamiEngine

func teacherQualityChecks() -> [(String, () throws -> Void)] {
    [
        (
            "teacher_logprob_stable_and_wire",
            {
                func require(_ value: Bool, _ message: String) throws {
                    if !value { throw EngineError(message) }
                }
                let p = TeacherAgreement.logProbability(0, logits: [logf(9), 0])!
                try require(abs(p - log(0.9)) < 1e-7, "Teacher score is not raw normalized model probability")
                let shifted = TeacherAgreement.logProbability(1, logits: [1000, 1000])!
                try require(abs(shifted + log(2)) < 1e-12, "Log probability overflowed after a logit offset")
                try require(
                    TeacherAgreement.logProbability(1, logits: [0, -1000]) == -1000,
                    "Low probability was clipped or underflowed")
                try require(
                    TeacherAgreement.logProbability(0, logits: [0, -.infinity]) == 0,
                    "Masked logits changed the valid distribution")
                let invalid: [[Float]] = [[], [.nan, 0], [.infinity, 0], [-.infinity, -.infinity]]
                for logits in invalid {
                    try require(TeacherAgreement.logProbability(0, logits: logits) == nil, "Invalid distribution was scored")
                }
                var score = TeacherAgreement()
                score.record(1, 0, position: 0, logits: [logf(9), 0])
                score.record(0, 0, position: 1, logits: [.nan, 0])
                try require(
                    score.positions == 2 && score.agreed == 1 && score.firstDifference == 0
                        && score.tokenLogProbabilities?.count == 2 && score.tokenLogProbabilities![1] == nil,
                    "Position alignment or invalid-score reporting differs")
                let legacy = try JSONDecoder().decode(
                    TeacherAgreement.self,
                    from: Data(#"{"positions":1,"agreed":1}"#.utf8))
                try require(legacy.tokenLogProbabilities == nil, "Legacy teacher result cannot decode")
                var stats = GenerationStatistics(promptTokens: 1, completionTokens: 2)
                stats.teacher = score
                stats.stageSeconds = ["generation": 1.25]
                stats.phaseCounters = ["generation.reads": 3]
                let wire = try ServingWire.usage(stats)
                _ = try JSONSerialization.data(withJSONObject: wire)
                let teacher = wire["origami_teacher"] as! [String: Any]
                let values = teacher["token_log_probabilities"] as! [Any]
                try require(values.count == 2 && values[1] is NSNull, "Teacher wire lost invalid position")
                try require(
                    (wire["origami_stage_seconds"] as? [String: Double])?["generation"] == 1.25,
                    "Explicit measurement lost boundary timing")
                stats.teacher = nil
                try require(
                    try ServingWire.usage(stats)["origami_stage_seconds"] == nil,
                    "Ordinary traffic emitted diagnostic timing")
            }
        )
    ]
}
