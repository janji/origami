import Foundation

/// Only a bounded internal code is retained on failure;
/// arbitrary Error descriptions, prompt text, IDs and tensors are excluded.
public struct GenerationFailureObservation: Sendable {
    public let engineCode: String
    public init(engineCode: String?) {
        let code = engineCode ?? "unclassified_generation_error"
        self.engineCode = String(
            decoding: code.utf8.prefix(96).map {
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 95 ? $0 : 95
            }, as: UTF8.self)
    }
}

/// A small incremental observation of a request (the server's progress lines and frames); never contains prompt text,
/// IDs or tensors.
public struct GenerationProgress: Sendable {
    public let requestOrdinal, sequence: UInt64
    public let event, phase: String
    public let elapsedSeconds: Double
    public let promptTokens, completedPromptTokens, outputTokens, decodeForwardScopes: Int
    public let scopeStart, scopeTokens, layersComplete: Int?
    public let failure: GenerationFailureObservation?
}

/// Serial request-local publisher of `GenerationProgress`. Its state is fixed size; past its record limit it says so
/// once and then publishes only the terminal record, never a silent omission or an inferred completed request.
public final class GenerationProgressRecorder {
    private let ordinal: UInt64, promptTokens: Int, maximumRecords: UInt64
    private let clock: () -> UInt64
    private let start: UInt64
    private let publish: (GenerationProgress) -> Void
    private var sequence: UInt64 = 0
    private var truncated = false

    public init(
        requestOrdinal: UInt64, promptTokens: Int, maximumRecords: UInt64 = 65_536,
        clock: @escaping () -> UInt64 = { DispatchTime.now().uptimeNanoseconds },
        publish: @escaping (GenerationProgress) -> Void
    ) {
        self.ordinal = requestOrdinal
        self.promptTokens = promptTokens
        self.maximumRecords = maximumRecords
        self.clock = clock
        start = clock()
        self.publish = publish
    }

    public func record(
        _ event: String, phase: String, completedPromptTokens: Int = 0,
        outputTokens: Int = 0, decodeForwardScopes: Int = 0,
        scopeStart: Int? = nil, scopeTokens: Int? = nil, layersComplete: Int? = nil,
        failure: GenerationFailureObservation? = nil
    ) {
        let terminal = event == "request_complete" || event == "request_abandoned"
        if truncated && !terminal { return }
        let emitted: String
        if sequence >= maximumRecords && !terminal {
            truncated = true
            emitted = "progress_record_limit_reached"
        } else {
            emitted = event
        }
        sequence += 1
        let now = clock()
        publish(
            GenerationProgress(
                requestOrdinal: ordinal, sequence: sequence,
                event: emitted, phase: phase, elapsedSeconds: Double(now >= start ? now - start : 0) / 1e9,
                promptTokens: promptTokens, completedPromptTokens: completedPromptTokens,
                outputTokens: outputTokens, decodeForwardScopes: decodeForwardScopes,
                scopeStart: scopeStart, scopeTokens: scopeTokens, layersComplete: layersComplete,
                failure: event == "request_abandoned" ? failure : nil))
    }
}
