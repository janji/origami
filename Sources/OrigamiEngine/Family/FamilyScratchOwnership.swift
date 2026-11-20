import Foundation
import Metal
import ObjectiveC

/// Exact lifetime compaction, with an explicit reference-layout override for release qualification.
/// A workspace keeps its selected layout for its entire lifetime.
/// Runtime qualification must compare fresh processes; toggling a live plan is deliberately unsupported.
package enum ScratchLifetimeCompaction {
    package static let enabledByDefault = ProcessInfo.processInfo.environment["ORIGAMI_SCRATCH_COMPACTION"] != "0"

    /// These roles never overlap on a family runner's ordered GPU queue:
    ///
    /// * embedding -> attention collapse -> FFN input -> final head collapse;
    /// * attention output (consumed by HC expand) -> FFN output (consumed by HC expand);
    /// * gathered expert input (consumed by gate/up) -> expert down output (consumed by scatter).
    ///
    /// Every layer completes the consumers before the next producer is encoded. Persistent streams, router
    /// records, HC post/combine metadata and deferred-contribution owners are intentionally outside these groups.
    /// Shared/routed gate buffers also stay separate: compaction changes storage only, never arithmetic order.
    package static func familyStorageName(_ role: String) -> String {
        switch role {
        case "embedded", "mix-in", "attn-in", "ffn-in", "head-in": return "ffn-in"
        case "mix-out", "attn-out", "ffn-out": return "ffn-out"
        case "exp-x", "exp-out": return "exp-out"
        default: return role
        }
    }

    package struct FamilyPlan {
        package let counts: [String: Int]
        package let logicalLengths: [String: Int]
        package let savedBytes: UInt64
    }

    /// Validate every logical request before merging; a negative/overflowing alias must not disappear under max.
    package static func familyPlan(_ counts: [String: Int], compact: Bool) throws -> FamilyPlan {
        var physical: [String: Int] = [:]
        var logical: [String: Int] = [:]
        var plainBytes: UInt64 = 0
        var physicalBytes: UInt64 = 0
        for (role, count) in counts {
            let bytes = try FamilyRunner.scratchLength(count)
            let (sum, overflow) = plainBytes.addingReportingOverflow(UInt64(bytes))
            guard !overflow else { throw EngineError("Family scratch capacity overflow", code: "insufficient_memory") }
            plainBytes = sum
            logical[role] = bytes
            let name = compact ? familyStorageName(role) : role
            physical[name] = max(physical[name] ?? 0, count)
        }
        for count in physical.values { physicalBytes += UInt64(try FamilyRunner.scratchLength(count)) }
        return FamilyPlan(counts: physical, logicalLengths: logical, savedBytes: plainBytes - physicalBytes)
    }
}

/// Live scratch bytes include views retained outside the runner's dictionary (including Metal command buffers).
/// Each physical allocation keeps its own charge until the last owner, even when a smaller plan replaces it.
package final class FamilyScratchOwnership: @unchecked Sendable {
    private let lock = NSLock()
    private var count: UInt64 = 0
    package var bytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
    private func add(_ bytes: UInt64) {
        lock.lock()
        count += bytes
        lock.unlock()
    }
    private func remove(_ bytes: UInt64) {
        lock.lock()
        count -= bytes
        lock.unlock()
    }
    private final class Owner {
        let accounting: FamilyScratchOwnership, bytes: UInt64, ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?
        init(_ accounting: FamilyScratchOwnership, bytes: UInt64, ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?) {
            self.accounting = accounting
            self.bytes = bytes
            self.ledger = ledger
            self.reservation = reservation
            accounting.add(bytes)
        }
        deinit {
            accounting.remove(bytes)
            if let reservation { try? ledger?.release(reservation) }
        }
    }
    private enum Association { nonisolated(unsafe) static var key: UInt8 = 0 }
    package func attach(_ buffer: MTLBuffer, ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?) {
        objc_setAssociatedObject(
            buffer, &Association.key,
            Owner(self, bytes: UInt64(buffer.length), ledger: ledger, reservation: reservation), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
}
