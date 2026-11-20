import Foundation

/// Sampled only at request/phase boundaries. Bytes are completed logical record
/// reads (including speculative reads), not an estimate of physical disk traffic.
package struct RequestPhaseCounters {
    private let values: [String: Double]

    package init(executor: SerialExecutor) {
        let source = executor.weights.store.status()
        var values: [String: Double] = [
            "original_expert_bytes": Double(source.originalExpertBytes),
            "packed_expert_bytes": Double(source.packedExpertBytes),
            "hot_copy_expert_bytes": Double(source.hotCopyReads * UInt64(PackStore.expertBytes)),
            "original_lookup_bytes": Double(source.originalPLEBytes),
            "packed_lookup_bytes": Double(source.packedPLEBytes),
        ]
        let refused = executor.ledger.optionalRefusals
        values.merge([
            "memory_refused_live": Double(refused.live), "memory_refused_live_seconds": refused.liveSeconds,
            "memory_refused_below_target": Double(refused.belowTarget),
            "memory_refused_below_target_seconds": refused.belowTargetSeconds,
        ]) { $1 }
        if let schedule = executor.prefillSchedulingSnapshot {
            values.merge([
                "selective_scopes": Double(schedule.selectiveScopes),
                "broad_scopes": Double(schedule.broadScopes),
                "broad_demotions": Double(schedule.broadDemotions),
                "routed_experts": Double(schedule.routedExperts),
                "routed_layers": Double(schedule.routedLayers),
                "ready_groups": Double(schedule.readyGroups),
                "ready_experts": Double(schedule.readyExperts),
                "resident_experts": Double(schedule.residentExperts),
                "read_ahead_submitted": Double(schedule.readAheadSubmitted),
                "read_ahead_consumed": Double(schedule.readAheadConsumed),
                "read_ahead_skipped": Double(schedule.readAheadSkipped),
                "read_ahead_demand_bypasses": Double(schedule.readAheadDemandBypasses),
                "read_ahead_wait_seconds": schedule.readAheadWaitSeconds,
                "prompt_cache_slots": Double(schedule.promptCacheSlots),
                "prompt_cache_slots_below_decode": Double(schedule.promptCacheSlotsBelowDecode),
            ]) { $1 }
        }
        if let preparation = executor.decodePreparationReceipt {
            values["slab_preparation_seconds"] = preparation.seconds
            values["slab_preparation_transitions"] = Double(preparation.transitions)
            values["slab_preparation_wire_fallbacks"] = Double(preparation.wireFallbacks ?? 0)
        }
        if let cache = executor.expertCacheSnapshot {
            values["cache_hits"] = Double(cache.hits)
            values["cache_misses"] = Double(cache.misses)
            values["cache_layer0_misses"] = Double(cache.layerZeroMisses)
        }
        if let split = executor.splitReads?.report {
            values["split_reads"] = Double(split["split_reads"] as? Int ?? 0)
            values["split_guard_rereads"] = Double(split["guard_rereads"] as? Int ?? 0)
            values["split_failed_part_rereads"] = Double(split["failed_part_rereads"] as? Int ?? 0)
            values["split_lone_demand_wait_seconds"] = split["lone_demand_wait_seconds"] as? Double ?? 0
            for (source, bytes) in split["part_bytes"] as? [String: Int] ?? [:] { values["split_part_bytes_\(source)"] = Double(bytes) }
            for (reason, count) in split["read_whole"] as? [String: Int] ?? [:] { values["split_read_whole_\(reason)"] = Double(count) }
        }
        if let timing = executor.decodeTiming {
            let c = timing.counts
            values.merge([
                "decode_wall_seconds": c.wallSeconds, "decode_gpu_seconds": c.gpuSeconds, "decode_gpu_union_seconds": c.gpuUnionSeconds,
                "decode_armed_stall_seconds": c.armedStallSeconds, "decode_join_seconds": c.explicitJoinSeconds,
                "decode_router_wait_seconds": c.routerWaitSeconds, "decode_late_wait_seconds": c.lateWaitSeconds,
                "decode_late_wait_layers": Double(c.lateWaitLayers), "decode_late_wait_experts": Double(c.lateWaitExperts),
                "decode_token_tail_seconds": c.tokenTailSeconds, "decode_layers": Double(c.layers),
                "decode_preencoded_layers": Double(c.preencodedLayers ?? 0), "decode_commands": Double(c.joined),
                "decode_replays": Double(c.originalReplays),
                "decode_window_seconds": c.windowSeconds, "decode_encode_next_seconds": c.encodeNextSeconds,
                "decode_residency_rebuilds": Double(c.residencyRebuilds), "decode_residency_seconds": c.residencySeconds,
            ]) { $1 }
            if let p = timing.prerouting {
                values.merge([
                    "prerouting_predicted": Double(p.predictedExperts), "prerouting_resident": Double(p.predictedResident),
                    "prerouting_issued": Double(p.predictionsIssued), "prerouting_hits": Double(p.predictedHits),
                    "prerouting_false": Double(p.predictedFalse), "prerouting_covered": Double(p.coveredDemand),
                    "prerouting_dropped_no_slot": Double(p.predictionsDroppedNoSlot),
                    "prerouting_refused_queue": Double(p.predictionsRefusedQueue),
                    "prerouting_gated": Double(p.predictionsGated),
                ]) { $1 }
            }
            if let reads = timing.reads {
                for (name, drive) in zip(["external", "internal", "hot"], reads) {
                    values["reads_\(name)_count"] = Double(drive.count)
                    values["reads_\(name)_seconds"] = drive.seconds
                    for (i, n) in drive.buckets.enumerated() { values["reads_\(name)_bucket\(i)"] = Double(n) }
                }
            }
        }
        let delayed = executor.delayedExpertCounters
        if delayed.deferredExperts > 0 || delayed.capacityRefusals > 0 {
            values.merge([
                "delayed_experts": Double(delayed.deferredExperts), "delayed_mass": delayed.deferredMass,
                "delayed_deadline_joins": Double(delayed.deadlineJoins), "delayed_capacity_refusals": Double(delayed.capacityRefusals),
                "delayed_applied_contributions": Double(delayed.appliedContributions),
            ]) { $1 }
        }
        self.values = values
    }

    package func delta(from previous: Self, phase: String) -> [String: Double] {
        Dictionary(
            uniqueKeysWithValues: values.map {
                (phase + "." + $0.key, max(0, $0.value - (previous.values[$0.key] ?? 0)))
            })
    }
}
