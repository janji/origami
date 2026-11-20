import Foundation

/// The ledger charge for the executor's host temporaries (activations, partials, transfer copies). Every retained
/// tensor, state, Metal workspace and expert slab keeps its own reservation; this never shrinks a live owner.
/// Two parts: a permanent 512 MiB bound for one-row (decode) work, which also covers the original-arithmetic
/// repair path at contexts up to 32K, and the rest of a 2 GB bound for multi-row prefill and capture, which
/// `retirePrefill` hands back for decode and `requirePrefill` takes again before the next prompt read.
package final class ReferencePhaseWorkspace {
    package static let fullBytes: UInt64 = 2_000_000_000
    package static let decodeBytes: UInt64 = 512 * 1024 * 1024
    private let ledger: MemoryLedger
    private let base: MemoryLedger.Reservation
    private var prefill: MemoryLedger.Reservation?

    package init(ledger: MemoryLedger) throws {
        let base = try ledger.reserve(bytes: Self.decodeBytes, category: "reference_decode_workspace_bound")
        let prefill: MemoryLedger.Reservation
        do {
            prefill = try ledger.reserve(
                bytes: Self.fullBytes - Self.decodeBytes,
                category: "reference_prefill_workspace_bound")
        } catch {
            try? ledger.release(base)
            throw error
        }
        self.ledger = ledger
        self.base = base
        self.prefill = prefill
    }
    deinit {
        if let prefill { try? ledger.release(prefill) }
        try? ledger.release(base)
    }

    /// Admission precedes any larger scope allocation or sequence mutation.
    /// On denial the existing row-one reservation remains intact.
    package func requirePrefill() throws {
        if prefill == nil {
            prefill = try ledger.reserve(
                bytes: Self.fullBytes - Self.decodeBytes,
                category: "reference_prefill_workspace_bound")
        }
    }

    /// Caller has joined every prefill reader and drained its enclosing autorelease pool.
    /// Only the unused prefill allowance retires; the base owner never changes.
    package func retirePrefill() throws {
        if let prefill {
            try ledger.release(prefill)
            self.prefill = nil
        }
    }
}
