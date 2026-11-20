import Foundation

/// Prospective simultaneous owner bound, separate from physical host eligibility.
/// Sizes the expert cache of each whole-schedule phase (a prefill scope or decode) on every tier, against the
/// ledger's current usable budget, so a governor trim between requests shrinks the next phase's cache.
/// The cache may still exceed a budget lowered below fixed + minimum; the ledger then decides.
package enum ConversationPhaseBudget {
    /// The least expert cache prefill can stream through: one layer's experts plus the read window.
    package static let minimumSlots = 528

    /// Expert cache bytes for a phase under the current budget: what is left after the fixed
    /// owners, capped by the configured cache, never below `minimumSlots`. A budget the
    /// governor lowered shrinks the cache; it never fails the request.
    package static func expertBytes(
        context: Int, fileBacked: Bool? = nil, input: Int, residentBytes: UInt64,
        experts: ExpertExecutionConfiguration, usable: UInt64,
        additionalOwners: UInt64 = 0, retainedFrontierBytes: UInt64? = nil,
        heldArenaBytes: UInt64? = nil, ceiling: UInt64? = nil
    ) throws -> UInt64 {
        guard ConversationGeometry.contexts.contains(context), (0...8192).contains(input) else {
            throw EngineError("Conversation phase budget geometry is unsupported")
        }
        let fixed = try ConversationGeometry(context: context, fileBacked: fileBacked).maximumFixedBytes(
            input: input,
            residentBytes: residentBytes, retainedFrontierBytes: retainedFrontierBytes, heldArenaBytes: heldArenaBytes)
        let (owners, overflow) = additionalOwners.addingReportingOverflow(
            experts.replicaOwnerBytes + experts.readAheadOwnerBytes
                + experts.layerCommandOwnerBytes + experts.reuseOwnerBytes + experts.decodeSlabOwnerBytes)
        guard !overflow, owners <= usable else { throw EngineError("Phase owners exceed the whole budget", code: "insufficient_memory") }
        let slab = UInt64(experts.slotsPerSlab * PackStore.expertBytes)
        let minimum = min((UInt64(minimumSlots * PackStore.expertBytes) + slab - 1) / slab * slab, experts.prefillCacheBytes)
        let room = owners < usable && fixed < usable - owners ? (usable - fixed - owners) / slab * slab : 0
        return max(minimum, min(ceiling ?? experts.cacheBytes, room))
    }
}
