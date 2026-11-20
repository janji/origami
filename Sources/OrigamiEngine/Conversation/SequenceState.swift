import Foundation
import Metal

/// One owned, admitted sequence: per-layer recurrent state (GDN for the 36 linear-attention layers, QSA key/value
/// state for the 12 attention layers, every fourth), the PLE convolution history and the last two token IDs, with
/// one ledger reservation for its host state. A failed forward invalidates it; partial state is never reused.
public final class SequenceState {
    public let id = UUID()
    public let capacity: Int
    public package(set) var tokenCount = 0
    public internal(set) var valid = true
    public package(set) var emittedButUnconsumed: Int?
    package var gdn: [Int: GDNState] = [:]
    package var qsa: [Int: QSAState] = [:]
    package var ple = PLEState()
    /// The last two consumed token IDs for the PLE n-gram rows; before any token they are end-of-text (248,044).
    package var history: [Int64] = [248_044, 248_044]
    /// The prompt's images: merged vision tokens injected as embeddings, and their positions.
    public var images: PromptImages?
    package let pageArena: QSAPageArena?
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation

    public convenience init(inputTokens: Int, outputTokens: Int, ledger: MemoryLedger, contextLimit: Int) throws {
        try self.init(inputTokens: inputTokens, outputTokens: outputTokens, ledger: ledger, contextLimit: contextLimit, pageArena: nil)
    }
    package init(inputTokens: Int, outputTokens: Int, ledger: MemoryLedger, contextLimit: Int, pageArena: QSAPageArena?) throws {
        guard inputTokens > 0, outputTokens > 0, inputTokens <= contextLimit,
            outputTokens <= contextLimit - inputTokens, contextLimit <= ModelContract.maximumContext
        else {
            throw EngineError("Input plus requested output exceeds context limit")
        }
        self.ledger = ledger
        self.pageArena = pageArena
        capacity = pageArena == nil ? inputTokens + outputTokens : contextLimit
        let logical = try MemoryLedger.sequenceStateBytes(inputTokens: inputTokens, outputTokens: outputTokens)
        // Two state copies cover reference dispatch and transactional growth.
        // Includes bounded GDN and PLE convolution and raw indexer tails.
        reservation = try ledger.reserve(
            bytes: pageArena == nil ? 2 * (logical + 3_000_000) : 240_000_000,
            category: pageArena == nil ? "reference_sequence_capacity_and_copy_bound" : "conversation_active_host_state_bound")
        do {
            for layer in 0..<48 {
                if layer % 4 == 3 {
                    var state = pageArena.map { QSAState(arena: $0) } ?? QSAState()
                    try state.reserveCapacity(tokens: capacity)
                    qsa[layer] = state
                } else {
                    gdn[layer] = GDNState()
                }
            }
        } catch {
            try? ledger.release(reservation)
            throw error
        }
    }
    deinit { try? ledger.release(reservation) }

    public func validate() throws {
        guard valid, tokenCount >= 0, tokenCount <= capacity, history.count == 2,
            history.allSatisfy({ (0..<248_320).contains($0) }), ple.convolution.count == 9 * 10_240,
            Set(gdn.keys) == Set((0..<48).filter { $0 % 4 != 3 }), Set(qsa.keys) == Set((0..<48).filter { $0 % 4 == 3 }),
            emittedButUnconsumed == nil || (0..<248_320).contains(emittedButUnconsumed!)
        else {
            throw EngineError("Sequence is invalid, incomplete or exhausted")
        }
        for state in gdn.values {
            try state.validate()
            guard state.offset == tokenCount else { throw EngineError("GDN frontier mismatch") }
        }
        for state in qsa.values {
            try state.validate()
            guard state.offset == tokenCount else { throw EngineError("QSA frontier mismatch") }
        }
    }
}
