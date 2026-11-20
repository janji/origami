import Foundation

/// The reply's token ledger, following the baseline's EOS-before-append rule: an end-of-sequence draw ends the
/// reply without becoming an output token, a penalized token or an emitted-but-unconsumed sequence token.
/// `generated` counts each output token (token ID: times generated), for the presence and frequency penalties.
/// Sampling is owned by the caller; `maximumOutput` bounds the appended tokens.
public struct GeneratedTokenSequence: Sendable {
    public private(set) var ids: [Int] = []
    public private(set) var generated: [Int: Int] = [:]
    public private(set) var terminationTokenID: Int?
    public let maximumOutput: Int

    public init(maximumOutput: Int) throws {
        guard (1...ModelContract.maximumContext).contains(maximumOutput) else { throw EngineError("Invalid generated output bound") }
        self.maximumOutput = maximumOutput
    }

    @discardableResult public mutating func recordSample(_ token: Int) throws -> Bool {
        guard terminationTokenID == nil, ids.count < maximumOutput, (0..<248_320).contains(token) else {
            throw EngineError("Sample follows completed output or has invalid vocabulary ID")
        }
        if TokenizerStack.eos.contains(token) {
            terminationTokenID = token
            return false
        }
        ids.append(token)
        generated[token, default: 0] += 1
        return true
    }
}
