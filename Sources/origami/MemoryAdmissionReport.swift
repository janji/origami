import OrigamiEngine

/// Coarse shared status for all model families. Retirement means the reservation's
/// last owner released it; it is not a claim about system-wide swap or footprint.
func memoryAdmissionReport(_ memory: MemoryLedger.Snapshot) -> [String: Any] {
    var report: [String: Any] = ["optional_growth_allowed": memory.optionalGrowthAllowed ?? true]
    if let overhead = memory.physicalOverheadBytes { report["physical_overhead_bytes"] = overhead }
    if let allowance = memory.prospectiveAllowanceBytes { report["prospective_allowance_bytes"] = allowance }
    if let relief = memory.cacheRelief {
        report["relief_requests"] = relief.requestCount
        if let target = relief.requestedTargetBytes { report["requested_target_bytes"] = target }
        report["expert_bytes_at_request"] = relief.cacheBytesAtLastRequest
        report["expert_bytes_retired"] = relief.cacheBytesReleasedSinceLastRequest
        report["expert_bytes_owned"] = relief.cacheOwnedBytes
    }
    return report
}
