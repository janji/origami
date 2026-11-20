import Foundation
import OrigamiEngine

// The asynchronous frontier writer's staging window width
// (prefix_cache.frontier_write_staging_slots, default 4; the plan uses 64).

private func frontierWindowFixture() throws -> [String: Any] {
    // The Origami 8K defaults at the v1 budget (23 GB usable), synchronous frontier writes.
    var object = try OrigamiPlan.baseObject(
        context: 8192, storage: .init(), promptCache: FileManager.default.temporaryDirectory,
        allowFast: true, bfloatMatrices: true, overrides: ["expert_cache_slots": 6312, "prefill_expert_cache_slots": 4840])
    object["process_ceiling_bytes"] = UInt64(25_000_000_000)
    object["transient_margin_bytes"] = UInt64(2_000_000_000)
    var prefix = object["prefix_cache"] as! [String: Any]
    prefix.removeValue(forKey: "frontier_async_writes")
    prefix.removeValue(forKey: "frontier_write_staging_slots")
    object["prefix_cache"] = prefix
    return object
}

private func loadFrontierWindow(_ object: [String: Any]) throws -> RuntimeConfiguration {
    try RuntimeConfiguration.load(object: object)
}

private func frontierWindowPrefix(_ object: [String: Any], _ values: [String: Any]) -> [String: Any] {
    var copy = object
    var prefix = copy["prefix_cache"] as! [String: Any]
    for (key, value) in values { prefix[key] = value }
    copy["prefix_cache"] = prefix
    return copy
}

/// The width parses, defaults to 4, charges exactly its page-rounded slots in native admission
/// and refuses out-of-range, untyped or synchronous widened windows.
func frontierWriteWindowConfiguration() throws {
    let object = try frontierWindowFixture()
    let synchronous = try loadFrontierWindow(object)
    let narrow = try loadFrontierWindow(frontierWindowPrefix(object, ["frontier_async_writes": true]))
    let wide = try loadFrontierWindow(frontierWindowPrefix(object, ["frontier_async_writes": true, "frontier_write_staging_slots": 64]))
    guard synchronous.frontierWriteStagingSlots == 4, !synchronous.frontierAsyncWrites,
        narrow.frontierWriteStagingSlots == 4, narrow.frontierAsyncWrites,
        wide.frontierWriteStagingSlots == 64, wide.frontierAsyncWrites
    else {
        throw EngineError("Frontier write window width did not parse")
    }
    let base = try synchronous.memoryAdmission()
    let four = try narrow.memoryAdmission()
    let sixtyFour = try wide.memoryAdmission()
    let slot = ConversationFrontierStore.stagingReservationBytes(slots: 1)
    guard slot % 16_384 == 0, slot >= UInt64(ConversationFrontierStore.stagingSlotBytes),
        four.nonGeometryOwnerBytes - base.nonGeometryOwnerBytes == 4 * slot,
        four.nonGeometryOwnerBytes - base.nonGeometryOwnerBytes == ConversationFrontierStore.stagingReservationBytes,
        sixtyFour.nonGeometryOwnerBytes - base.nonGeometryOwnerBytes == 64 * slot,
        sixtyFour.usableBytes == four.usableBytes, sixtyFour.configuredSlots == four.configuredSlots,
        sixtyFour.fullFrontierSlots <= four.fullFrontierSlots, sixtyFour.noFrontierSlots <= four.noFrontierSlots
    else {
        throw EngineError("Frontier write window admission differs from its slots")
    }
    let refused: [([String: Any], String)] = [
        (["frontier_async_writes": true, "frontier_write_staging_slots": 3], "below 4"),
        (["frontier_async_writes": true, "frontier_write_staging_slots": 97], "above 96"),
        (["frontier_async_writes": true, "frontier_write_staging_slots": 32.5], "fractional"),
        (["frontier_async_writes": true, "frontier_write_staging_slots": true], "boolean"),
        (["frontier_async_writes": true, "frontier_write_staging_slots": "32"], "string"),
        (["frontier_write_staging_slots": 32], "synchronous widened window"),
        (["frontier_async_writes": false, "frontier_write_staging_slots": 8], "explicitly synchronous widened window"),
    ]
    for (values, label) in refused {
        var admitted = true
        do { _ = try loadFrontierWindow(frontierWindowPrefix(object, values)) } catch is EngineError { admitted = false }
        guard !admitted else { throw EngineError("Frontier write window was admitted: " + label) }
    }
}
