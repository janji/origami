import Foundation
import OrigamiEngine

private func lifecycleRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}

func generationProgressWriterIsBoundedAndFailureVisible() throws {
    var now: UInt64 = 100
    var records: [GenerationProgress] = []
    let progress = GenerationProgressRecorder(
        requestOrdinal: 4, promptTokens: 32_256, maximumRecords: 3,
        clock: {
            now += 100
            return now
        }, publish: { records.append($0) })
    progress.record("request_started", phase: "prefill")
    progress.record("prefill_layer_complete", phase: "prefill", scopeStart: 0, scopeTokens: 8192, layersComplete: 1)
    progress.record("prefill_complete", phase: "decode", completedPromptTokens: 32_256)
    progress.record("decode_token_sampled", phase: "decode", outputTokens: 1)
    progress.record("decode_scope_complete", phase: "decode", outputTokens: 1, decodeForwardScopes: 1)
    progress.record(
        "request_abandoned", phase: "decode", outputTokens: 1, decodeForwardScopes: 1,
        failure: GenerationFailureObservation(engineCode: "gpu_execution_failed"))
    try lifecycleRequire(
        records.count == 5 && records.map(\.sequence) == [1, 2, 3, 4, 5] && records[3].event == "progress_record_limit_reached"
            && records.last?.event == "request_abandoned" && records.allSatisfy({ $0.requestOrdinal == 4 && $0.promptTokens == 32_256 })
            && records[1].scopeTokens == 8192 && records[1].layersComplete == 1,
        "Bounded progress lost its failure marker, request identity, layer or terminal record")
    try lifecycleRequire(
        records.last?.event != "request_complete" && records.last?.failure?.engineCode == "gpu_execution_failed"
            && records[0].failure == nil,
        "An abandoned request was published as completed or lost its bounded internal failure code")
    let bounded = GenerationFailureObservation(engineCode: String(repeating: "x\"\n", count: 4096))
    try lifecycleRequire(
        bounded.engineCode.utf8.count == 96 && bounded.engineCode.allSatisfy({ $0 == "x" || $0 == "_" }),
        "Failure code exceeded its fixed ASCII bound")
}
