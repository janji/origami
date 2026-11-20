import Foundation

extension RuntimeConfiguration {
    // Contiguous and paged attention each retain one path-counter allocation.
    // Charge both 16 KiB page-rounded owners before either variant is created.
    static let fusedSelectedPrefillCounterOwnerBytes: UInt64 = 2 * 16_384
    /// Planning arithmetic for one configuration (allocates nothing): the fixed owner bytes of each phase shape
    /// (cold prefill, no or full retained frontier, a short append) and the expert slots the usable budget leaves.
    public struct MemoryAdmission {
        public let context: Int
        public let usableBytes, additionalOwnerBytes, nonGeometryOwnerBytes: UInt64
        public let coldFixedBytes, coldTotalBytes, noFrontierFixedBytes, fullFrontierFixedBytes, shortFixedBytes: UInt64
        /// Decode with no retained frontier and an empty conversation lending its key/value arena (elastic tiers;
        /// otherwise `noFrontierFixedBytes`): the most the expert cache can be configured to.
        public let lentArenaFixedBytes: UInt64
        public let configuredSlots, noFrontierSlots, fullFrontierSlots, shortSlots: Int
    }

    /// Source-only arithmetic; does not allocate a model or claim physical fit.
    public func memoryAdmission(additionalOwnerBytes: UInt64 = 0) throws -> MemoryAdmission {
        guard wholeExecutionSchedule, persistentConversation, conversationFrontierDiskBytes > 0,
            residentBytes <= 4_000_000_000, denseResidentBytes <= 600_000_000,
            readScope == 8192
        else {
            throw EngineError("Memory admission requires complete dense conversation owners")
        }
        try experts.validate()
        let geometry = try ConversationGeometry(context: contextLimit, fileBacked: fileBackedContext)
        let own =
            experts.replicaOwnerBytes + experts.readAheadOwnerBytes + experts.layerCommandOwnerBytes + experts.reuseOwnerBytes
            + experts.decodeSlabOwnerBytes
            + (decodeEarlyDemand ? DecodeEarlyDemand.metadataBytes : 0)
            + (decodeTokenPlan ? ReferenceWeights.tokenPlanOwnerBytes : 0)
            + (decodeNextLayerPrerouting ? DecodePrerouting.ownerBytes : 0)
            + (fusedSelectedPrefillAttention ? Self.fusedSelectedPrefillCounterOwnerBytes : 0)
            + (hotCopyFolder == nil ? 0 : PackStore.hotCopyMetadataBytes)
            + (frontierAsyncWrites ? ConversationFrontierStore.stagingReservationBytes(slots: frontierWriteStagingSlots) : 0)
        let usable = try effectiveUsableBytes
        guard additionalOwnerBytes <= usable, own <= usable - additionalOwnerBytes else {
            throw EngineError("Additional memory owners exceed the memory budget", code: "insufficient_memory")
        }
        let extras = own + additionalOwnerBytes
        func fixed(_ input: Int, _ retained: UInt64? = nil, _ arena: UInt64? = nil) throws -> UInt64 {
            let geometryBytes = try geometry.maximumFixedBytes(
                input: input,
                residentBytes: residentBytes + denseResidentBytes, retainedFrontierBytes: retained, heldArenaBytes: arena)
            guard geometryBytes <= usable - extras else {
                throw EngineError("Fixed memory owners exceed the memory budget", code: "insufficient_memory")
            }
            return geometryBytes + extras
        }
        let cold = try fixed(readScope)
        let full = try fixed(0)
        let short = try fixed(1)
        let noFrontier = try fixed(0, elasticDecodeResidency ? 0 : nil)
        let lent = elasticDecodeResidency ? try fixed(0, 0, geometry.heldArenaBytes(live: 0, growth: 0)) : noFrontier
        let slab = UInt64(experts.slotsPerSlab * PackStore.expertBytes)
        func slots(_ fixed: UInt64) throws -> Int {
            let bytes = min(experts.cacheBytes, (usable - fixed) / slab * slab)
            guard bytes >= experts.prefillCacheBytes else {
                throw EngineError("Phase owners displace required prefill experts", code: "insufficient_memory")
            }
            return Int(bytes / UInt64(PackStore.expertBytes))
        }
        guard experts.prefillCacheBytes <= usable - cold, experts.cacheBytes <= usable - lent else {
            throw EngineError("Configured expert capacity exceeds complete admission", code: "insufficient_memory")
        }
        return try MemoryAdmission(
            context: contextLimit,
            usableBytes: usable, additionalOwnerBytes: additionalOwnerBytes, nonGeometryOwnerBytes: extras,
            coldFixedBytes: cold, coldTotalBytes: cold + experts.prefillCacheBytes,
            noFrontierFixedBytes: noFrontier, fullFrontierFixedBytes: full, shortFixedBytes: short, lentArenaFixedBytes: lent,
            configuredSlots: experts.cacheSlots, noFrontierSlots: slots(noFrontier), fullFrontierSlots: slots(full),
            shortSlots: slots(short))
    }
}
