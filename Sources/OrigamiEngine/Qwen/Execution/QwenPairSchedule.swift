import Metal

package enum QwenPairSchedule {
    /// A pair may route to twenty distinct experts. Custom plans without early demand only reserve their ordinary
    /// window in the I/O queue, and a smaller phase cache cannot retain the complete union through GPU consumption.
    /// Refuse before changing sequence state; the caller can continue with the ordinary one-token decoder.
    package static func requireUnionCapacity(
        _ configuration: ExpertExecutionConfiguration,
        earlyDemand: Bool
    ) throws {
        guard configuration.effectivePrefillCacheSlots >= 20,
            earlyDemand || configuration.demandWindow >= 20
        else {
            throw EngineError("Pair step requires cache and I/O capacity for twenty expert records", code: "pair_unsupported")
        }
    }
}

extension WholeExecutionSchedule {
    /// A pair's authoritative union is already enqueued. Missing records leave room for independent shared work;
    /// a resident union coalesces that work into the expert command instead. The caller joins the returned command
    /// after the expert continuation (same queue), or in its exceptional cleanup, before reusing this workspace.
    package func pairSharedDuringReads(
        missing: Bool, shouldContinue: () -> Bool,
        encode: (MTLCommandBuffer) throws -> Void
    ) throws -> MTLCommandBuffer? {
        guard missing else { return nil }
        let command = try self.command("pair-shared-during-reads")
        try encode(command)
        try submit(command, shouldContinue: shouldContinue)
        return command
    }
}
