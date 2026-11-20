import Foundation
import OrigamiEngine

/// Model-free replay: unique output counts survive a failed retry, while repeated computation stays visible.
func generationAttemptMetrics() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    let control = GenerationControl()
    enum Failure: Error { case interrupted, cancelled }
    do {
        try control.withAttempt(number: 1, replayedTokens: 0, preparationSeconds: ["prompt_encoding": 0.25]) {
            control.reportGeneratedTokens(100)
            control.report {
                $0.stageSeconds = ["prompt_encoding": 0.25, "generation": 12]
                $0.phaseCounters = ["partial_attempt.read_bytes": 1000]
            }
            throw Failure.interrupted
        }
    } catch Failure.interrupted {}
    let first = control.progress.firstGeneratedAt
    do {
        try control.withAttempt(number: 2, replayedTokens: 100, preparationSeconds: ["prompt_encoding": 0.25]) {
            control.reportGeneratedTokens(0)
            try require(
                control.progress.generatedTokens == 100 && control.progress.firstGeneratedAt == first, "Retry erased prior unique output")
            control.report {
                $0.stageSeconds = ["restore_prefill_and_save": 2]
                $0.phaseCounters = ["partial_attempt.read_bytes": 500]
            }
            throw Failure.cancelled
        }
    } catch Failure.cancelled {}
    try control.withAttempt(number: 3, replayedTokens: 100, preparationSeconds: ["prompt_encoding": 0.25]) {
        for i in 1...100 { control.reportGeneratedTokens(i) }
        try require(control.progress.generatedTokens == 100, "Replayed tokens were double-counted")
        control.reportGeneratedTokens(103)
    }
    let progress = control.progress
    try require(progress.generatedTokens == 103 && progress.firstGeneratedAt == first, "Output high-water timestamp changed")
    try require(progress.attempts.count == 3 && progress.attempts.map(\.generatedTokens) == [100, 0, 103], "Attempt output history lost")
    try require(
        progress.attempts[0].phaseCounters["partial_attempt.read_bytes"] == 1000
            && progress.attempts[1].phaseCounters["partial_attempt.read_bytes"] == 500, "Retry work overwritten")
    try require(
        progress.attempts.allSatisfy { $0.stageSeconds["prompt_encoding"] == nil }
            && progress.preparationStageSeconds["prompt_encoding"] == 0.25, "One-time preparation repeated")
    try require(JSONSerialization.isValidJSONObject(progress.attempts.map(\.wire)), "Attempt history is not serializable")
    for n in 4...12 { try control.withAttempt(number: n, replayedTokens: 103) {} }
    try require(control.progress.attempts.count == 8 && control.progress.omittedAttempts == 4, "Attempt history is not bounded")
}
