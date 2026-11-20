import Darwin
import Foundation
import Metal
import OrigamiEngine

func familyPrefetchObservationChecks() -> [(String, () throws -> Void)] {
    [
        ("prefetch_observations_cold_full_router", prefetchObservationsColdFullRouter),
        ("prefetch_observations_outcomes_and_sources", prefetchObservationsOutcomesAndSources),
        ("prefetch_observations_ticket_and_rank_bounds", prefetchObservationsTicketAndRankBounds),
        ("prefetch_observations_geometry_and_floor", prefetchObservationsGeometryAndFloor),
        ("prefetch_observations_fetch_bins_and_snapshot", prefetchObservationsFetchBinsAndSnapshot),
        ("prefetch_observations_cache_integration", prefetchObservationsCacheIntegration),
        ("prefetch_observations_paired_precision_and_admission", prefetchObservationsPairedPrecisionAndAdmission),
        ("family_prefetch_policy_late_start", familyPrefetchPolicyLateStart),
        ("family_prefetch_policy_scores_absent", familyPrefetchPolicyScoresAbsent),
    ]
}
/// A guess whose read begins before the next router is admitted even when it ends after it; one that would only
/// begin after the router, or of a rank that misses, is declined.
private func familyPrefetchPolicyLateStart() throws {
    var policy = FamilyPrefetchPolicy()
    for _ in 0..<4 { policy.observe(layer: 0, predicted: [1, 2], needed: [1], seconds: 0.004) }
    let admit = policy.decide(layer: 0, rank: 0, queuedRecords: 0, maximumRecords: 6, queuedSeconds: 0.001, serviceSeconds: 0.01)
    try observationRequire(admit == .admit, "a read starting before the router was declined: \(admit)")
    let late = policy.decide(layer: 0, rank: 0, queuedRecords: 0, maximumRecords: 6, queuedSeconds: 0.005, serviceSeconds: 0.01)
    try observationRequire(late == .deadline, "a read starting after the router was admitted: \(late)")
    let miss = policy.decide(layer: 0, rank: 1, queuedRecords: 0, maximumRecords: 6, queuedSeconds: 0, serviceSeconds: 0.01)
    try observationRequire(miss == .confidence, "a missing rank was admitted: \(miss)")
}
/// A rank is judged by the guesses a read would be for: one right only while its expert was already in memory, and
/// wrong whenever it was not, is declined.
private func familyPrefetchPolicyScoresAbsent() throws {
    var policy = FamilyPrefetchPolicy()
    for i in 0..<16 {
        let resident = i % 2 == 0
        policy.observe(
            layer: 0, predicted: [1, resident ? 2 : 3], needed: resident ? [1, 2] : [1], absent: resident ? [] : [1, 3],
            seconds: 0.01)
    }
    let decide = { (rank: Int) in
        policy.decide(layer: 0, rank: rank, queuedRecords: 0, maximumRecords: 6, queuedSeconds: 0, serviceSeconds: 0.01)
    }
    try observationRequire(decide(0) == .admit, "a rank right on every read was declined")
    try observationRequire(decide(1) == .confidence, "resident hits hid a rank whose reads all miss")
}
private func observationRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try condition() == false { throw EngineError("Prefetch observations: " + message) }
}
private func observationLedger() throws -> MemoryLedger {
    try MemoryLedger(ceiling: 16 << 20, transientMargin: 1, availableFloor: 1)
}
private func observations(_ ledger: MemoryLedger, layers: Int = 3, experts: Int = 16, ranks: Int = 4) throws -> FamilyPrefetchObservations {
    guard let result = FamilyPrefetchObservations(layers: layers, experts: experts, ranks: ranks, ledger: ledger, retainingBytes: 0) else {
        throw EngineError("Observation fixture was not admitted")
    }
    return result
}
private func layerCount(_ snapshot: [String: Any], _ layer: Int, _ column: String) -> UInt64 {
    let names = snapshot["layer_columns"] as! [String]
    let counts = snapshot["layer_counts"] as! [UInt64]
    return counts[layer * names.count + names.firstIndex(of: column)!]
}
private func rankCount(_ snapshot: [String: Any], _ layer: Int, _ rank: Int, _ column: String) -> UInt64 {
    let names = snapshot["rank_columns"] as! [String]
    let counts = snapshot["rank_counts"] as! [UInt64]
    let buckets = snapshot["rank_buckets"] as! Int
    return counts[(layer * buckets + min(rank, buckets - 1)) * names.count + names.firstIndex(of: column)!]
}

private func prefetchObservationsColdFullRouter() throws {
    let ledger = try observationLedger()
    let o = try observations(ledger)
    o.predict(layer: 1, ranked: [1, 3, 4, 6], resident: [1, 2], inFlight: [3])
    let ready = o.issued(layer: 1, expert: 4, planned: .gguf)
    let cancelled = o.issued(layer: 1, expert: 6, planned: .hot)
    o.finished(ready, outcome: .completed, delivered: .gguf, bytes: 300)
    o.finished(cancelled, outcome: .cancelled)
    // Full router A includes a cold ID5 in a later fetch wave. R0={1,2}, I0={3}; C={3,4,5}, fresh={4,5}.
    o.routed(layer: 1, needed: [2, 3, 4, 5], ready: [2, 4], inFlight: [3])
    let s = o.snapshot
    try observationRequire(
        layerCount(s, 1, "needed") == 4 && layerCount(s, 1, "cold_needed") == 3
            && layerCount(s, 1, "fresh_needed") == 2 && layerCount(s, 1, "cold_recalled") == 2
            && layerCount(s, 1, "fresh_recalled") == 1, "cold/fresh denominators lost the later wave or included warm IDs")
    try observationRequire(
        rankCount(s, 1, 2, "issued_needed_ready_at_router") == 1
            && rankCount(s, 1, 0, "cold_predicted") == 0 && rankCount(s, 1, 1, "fresh_predicted") == 0,
        "resident or already-reading predictions became fresh I/O opportunities")
    try observationRequire(
        s["live_outcome_tickets"] as? Int == 0 && layerCount(s, 1, "scored") == 1, "terminal tickets survived router scoring")
}

private func prefetchObservationsOutcomesAndSources() throws {
    let ledger = try observationLedger()
    let o = try observations(ledger)
    o.predict(layer: 1, ranked: [1, 2, 3], resident: [], inFlight: [])
    let hot = o.issued(layer: 1, expert: 1, planned: .hot)
    let failed = o.issued(layer: 1, expert: 2, planned: .gguf)
    let cancelled = o.issued(layer: 1, expert: 3, planned: .pack)
    o.routed(layer: 1, needed: [1, 2], ready: [], inFlight: [1, 2, 3])
    o.finished(hot, outcome: .completed, delivered: .gguf, bytes: 300)
    o.finished(failed, outcome: .failed)
    o.finished(cancelled, outcome: .cancelled)
    o.finished(hot, outcome: .completed, delivered: .hot, bytes: 999)  // duplicate terminal notification must be inert
    let s = o.snapshot
    let source = s["source_counts"] as! [UInt64]
    try observationRequire(
        source[10] == 1 && source[14] == 1 && source[17] == 300 && source[19] == 1,
        "planned hot source was confused with actual GGUF fallback or bytes")
    try observationRequire(
        source[13] == 0 && rankCount(s, 1, 0, "completed") == 1 && rankCount(s, 1, 1, "failed") == 1
            && rankCount(s, 1, 2, "cancelled") == 1 && rankCount(s, 1, 0, "issued_needed_inflight_at_router") == 1,
        "ready/late/failure/cancellation terminal semantics were conflated")
}

private func prefetchObservationsTicketAndRankBounds() throws {
    let ledger = try observationLedger()
    let o = try observations(ledger, ranks: 2)
    o.predict(layer: 0, ranked: [0, 1, 2, 3, 3], resident: [], inFlight: [])
    o.routed(layer: 0, needed: [2, 3], ready: [], inFlight: [])
    try observationRequire(rankCount(o.snapshot, 0, 2, "predicted") == 2, "overflow rank was dropped or duplicate ID counted twice")
    var tickets: [FamilyPrefetchObservations.Ticket] = []
    for _ in 0..<150 {
        o.predict(layer: 1, ranked: [4], resident: [], inFlight: [])
        if let ticket = o.issued(layer: 1, expert: 4, planned: .gguf) { tickets.append(ticket) }
    }
    try observationRequire(
        tickets.count == 128 && o.snapshot["live_outcome_tickets"] as? Int == 128
            && o.snapshot["dropped_outcome_tickets"] as? UInt64 == 22, "live ticket overflow was unbounded or hidden")
    o.abandonPrediction()
    for ticket in tickets { o.finished(ticket, outcome: .completed, delivered: .gguf, bytes: 1) }
    try observationRequire(
        o.snapshot["live_outcome_tickets"] as? Int == 0 && layerCount(o.snapshot, 1, "abandoned") == 150
            && rankCount(o.snapshot, 1, 0, "outcome_without_scored_router") == 128,
        "superseded predictions became false negatives or leaked outcome tickets")
}

private func prefetchObservationsGeometryAndFloor() throws {
    let ledger = try observationLedger()
    let floor: UInt64 = 512 << 10
    try observationRequire(
        FamilyPrefetchObservations.estimate(layers: 40, experts: 384, ranks: 6)! <= 256 << 10,
        "production geometry does not fit the stated cap")
    // Large arrays cross allocator size classes: 53,760 rank payload bytes occupy a 65,536 backing allocation.
    // Use a tiny expert dimension so generous scratch allowances cannot hide omitted numeric rounding.
    let numericCounts = [
        40 * 7 * FamilyPrefetchObservations.rankColumns.count,
        40 * FamilyPrefetchObservations.layerColumns.count, 400, 80, 80, 40,
    ]
    let roundedNumeric = numericCounts.reduce(UInt64(0)) { $0 + UInt64(malloc_good_size($1 * 8 + 64)) }
    try observationRequire(
        FamilyPrefetchObservations.estimate(layers: 40, experts: 1, ranks: 6)! >= 2 * roundedNumeric + 16_384,
        "numeric arrays and their COW snapshot were charged by payload instead of allocation size")
    ledger.setUsable(floor)
    try observationRequire(
        FamilyPrefetchObservations(layers: 40, experts: 384, ranks: 6, ledger: ledger, retainingBytes: floor) == nil
            && ledger.reservedBytes == 0, "collector borrowed the lazy expert floor")
    ledger.setUsable(floor + FamilyPrefetchObservations.capacityBytes)
    var admitted = FamilyPrefetchObservations(layers: 40, experts: 384, ranks: 6, ledger: ledger, retainingBytes: floor)
    try observationRequire(admitted != nil && ledger.reservedBytes == 256 << 10, "realistic collector was not charged")
    let mandatory = try ledger.reserve(bytes: floor, category: "fixture-experts")
    try ledger.release(mandatory)
    admitted = nil
    try observationRequire(ledger.reservedBytes == 0, "collector charge survived owner deinit")
    try observationRequire(
        FamilyPrefetchObservations(layers: 512, experts: 4096, ranks: 4096, ledger: ledger, retainingBytes: 0) == nil,
        "oversized geometry allocated unbounded arrays")
    let physical = try MemoryLedger(ceiling: 16 << 20, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    try observationRequire(
        FamilyPrefetchObservations(
            layers: 40, experts: 384, ranks: 6, ledger: physical, retainingBytes: floor,
            footprint: physical.ceiling - FamilyPrefetchObservations.capacityBytes - floor + 1) == nil && physical.reservedBytes == 0,
        "collector consumed physical room reserved for mandatory slabs")
}

private func prefetchObservationsFetchBinsAndSnapshot() throws {
    let ledger = try observationLedger()
    let o = try observations(ledger)
    o.demandFetch(layer: 0, queued: 0, running: 0, nanoseconds: 100, bytes: 20, succeeded: true)
    let previous = o.snapshot
    o.demandFetch(layer: 0, queued: 2, running: 1, nanoseconds: 300, bytes: 60, succeeded: false)
    let current = o.snapshot
    try observationRequire(
        Array((current["fetch_wall_nanoseconds"] as! [UInt64]).prefix(2)) == [100, 300]
            && Array((current["successful_demand_read_bytes"] as! [UInt64]).prefix(2)) == [20, 60], "competitor timing/byte bins merged")
    try observationRequire(
        layerCount(previous, 0, "fetches") == 1 && layerCount(current, 0, "fetches") == 2
            && layerCount(current, 0, "failed_fetches") == 1, "published value snapshot mutated with later counters")
}

/// Fake expert records and shared buffers only. Suspended queue establishes incomplete guesses deterministically;
/// collecting a full router choice must not wait, issue reads, or consume only the following first fetch wave.
private func prefetchObservationsCacheIntegration() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prefetch-observations-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder, used: 2)), device: device)
    let pool = try FamilyExperts(weights: weights, budget: UInt64(4 * (212_992 + 114_688)))
    pool.ledger = try observationLedger()
    pool.adaptiveGuesses = false
    pool.guessDepth = nil
    _ = try pool.fetch(layer: 2, experts: [0], decode: true)
    pool.admitPrefetchObservations(enabled: true)
    try observationRequire(pool.observationReservedBytes == 256 << 10, "integration collector admission failed")
    let hold = pool.holdGuesses()
    defer {
        hold.release()
        pool.waitForReads()
    }
    pool.prefetch(layer: 2, experts: [0, 1, 2])
    let before = pool.counters.readBytes
    pool.observeRoutedDemand(layer: 2, needed: [0, 1, 3])
    try observationRequire(pool.counters.readBytes == before, "observing router demand performed I/O")
    _ = try pool.fetch(layer: 2, experts: [0], decode: true)  // first wave; unused unstarted guesses cancel
    let s = pool.observationNotes
    try observationRequire(
        layerCount(s, 2, "cold_needed") == 2 && layerCount(s, 2, "cold_recalled") == 1
            && rankCount(s, 2, 1, "issued_needed_inflight_at_router") == 1,
        "collector confused resident/incomplete guesses or consumed only the first wave")
    try observationRequire(
        rankCount(s, 2, 1, "cancelled") == 1 && rankCount(s, 2, 2, "cancelled") == 1,
        "cancelled queued operations lost or duplicated their outcomes")
    // Prompt read-ahead is charged at issue and subtracted on cancellation. Its net counter movement must not
    // erase a successful decode demand batch from this observer's local byte count.
    pool.readAhead(layer: 2, experts: [1, 2])
    let charged = pool.counters.readBytes
    _ = try pool.fetch(layer: 2, experts: [3], decode: true)
    let bytes = pool.observationNotes["successful_demand_read_bytes"] as! [UInt64]
    try observationRequire(
        pool.counters.readBytes < charged && bytes[4] > 0,
        "cancelled prompt precharge hid successful demand bytes")
}

/// Abandoned predictions retain proposal counts, but never enter paired precision denominators. Admission
/// reasons are candidate-level decisions, not evidence that refusing a read saved time.
private func prefetchObservationsPairedPrecisionAndAdmission() throws {
    let ledger = try observationLedger()
    let o = try observations(ledger, experts: 16, ranks: 8)
    o.predict(layer: 0, ranked: [1, 2], resident: [], inFlight: [], now: 10)
    o.predict(layer: 0, ranked: [1, 2, 3, 4, 5, 6, 7], resident: [1], inFlight: [2], now: 100)
    for (expert, reason) in [
        (3, FamilyPrefetchObservations.Rejection.pressure), (4, .confidence),
        (5, .queue), (6, .deadline), (7, .noVictimOrCapacity),
    ] {
        o.rejected(layer: 0, expert: expert, reason: reason)
    }
    o.rejected(layer: 0, expert: 1, reason: .pressure)  // completed residents cannot become pressure rejections
    o.depthLimited(layer: 0, count: 2)
    o.routed(layer: 0, needed: [1, 2, 3], ready: [1], inFlight: [2], now: 450)
    let s = o.snapshot
    try observationRequire(
        rankCount(s, 0, 0, "predicted") == 2 && rankCount(s, 0, 0, "scored_predicted") == 1
            && rankCount(s, 0, 0, "cold_predicted") == 1 && rankCount(s, 0, 0, "scored_cold_predicted") == 0
            && rankCount(s, 0, 1, "scored_cold_predicted") == 1 && rankCount(s, 0, 1, "scored_fresh_predicted") == 0
            && rankCount(s, 0, 2, "scored_fresh_predicted") == 1,
        "abandoned/unmatched proposals contaminated precision denominators")
    for (rank, name) in [
        "already_ready", "already_inflight", "rejected_pressure", "rejected_confidence",
        "rejected_queue", "rejected_deadline", "rejected_no_victim_or_capacity",
    ].enumerated() {
        try observationRequire(rankCount(s, 0, rank, name) == 1, "admission reason lost its rank")
    }
    try observationRequire(
        rankCount(s, 0, 0, "rejected_pressure") == 0 && layerCount(s, 0, "excluded_by_depth_limit") == 2
            && (s["scored_prediction_to_router_nanoseconds"] as! [UInt64])[0] == 350,
        "warm rejection or abandoned prediction contaminated admission/elapsed counters")
    let other = try observations(ledger)
    try observationRequire(
        s["instance_id"] as? String == o.snapshot["instance_id"] as? String
            && s["instance_id"] as? String != other.snapshot["instance_id"] as? String, "model instance identity is missing or unstable")
}
