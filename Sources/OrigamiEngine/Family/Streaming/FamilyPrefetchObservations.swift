import Darwin
import Foundation

/// Bounded observations, never a scheduling policy. R0 is completed valid residency when a prediction is made;
/// I0 is the reads already outstanding then. The next complete router choice, not its first fetch wave, is A.
/// Counters describe conditional opportunity/readiness, not causal latency saved or physical device traffic.
package final class FamilyPrefetchObservations: @unchecked Sendable {
    package static let capacityBytes: UInt64 = 256 << 10
    private static let ticketLimit = 128
    package enum Source: Int, Sendable { case hot, gguf, pack }
    package enum Outcome: Sendable { case completed, failed, cancelled, unavailable }
    package enum Rejection: Int, Sendable {
        case pressure = 19
        case confidence, queue, deadline, noVictimOrCapacity
    }
    package struct Ticket: Sendable { let slot: Int, serial: UInt64 }
    private struct TicketState {
        let serial, epoch: UInt64
        let layer, rank, expert: Int
        let planned: Source
        var scored = false, abandoned = false, terminal = false
    }
    private struct Prediction {
        let epoch, began: UInt64
        let layer: Int, ranked: [Int]
        // Bits: completed at prediction, outstanding at prediction, issued, seen during rank deduplication.
        var flags: [UInt8]
    }
    // Columnar numeric snapshots avoid a dictionary allocation per rank/layer. These columns are part of the
    // observation schema; cumulative counters are delta-able only within the same model instance/configuration.
    package static let rankColumns = [
        "predicted", "cold_predicted", "fresh_predicted", "needed", "cold_needed", "fresh_needed",
        "issued", "issued_needed_ready_at_router", "issued_needed_inflight_at_router", "issued_needed_unavailable_at_router",
        "completed", "failed", "cancelled", "outcome_without_scored_router",
        "scored_predicted", "scored_cold_predicted", "scored_fresh_predicted", "already_ready", "already_inflight",
        "rejected_pressure", "rejected_confidence", "rejected_queue", "rejected_deadline", "rejected_no_victim_or_capacity",
    ]
    package static let layerColumns = [
        "predictions", "scored", "abandoned", "needed", "cold_needed", "fresh_needed", "cold_recalled", "fresh_recalled",
        "needed_ready_at_router", "previously_resident_now_cold", "fetches", "fetches_with_competitors", "queued_competitors",
        "running_competitors",
        "successful_fetches", "failed_fetches", "excluded_by_depth_limit",
    ]
    private let layers, experts, rankLimit, rankBuckets: Int
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    private let lock = NSLock()
    private let instanceID = UUID().uuidString
    private var ranks: [UInt64], layerCounts: [UInt64], sources: [UInt64], fetchNanoseconds: [UInt64], fetchBytes: [UInt64],
        predictionNanoseconds: [UInt64]
    private var tickets: [TicketState?]
    private var pending: Prediction?
    private var epoch: UInt64 = 0, serial: UInt64 = 0
    private var droppedTickets: UInt64 = 0, unavailableOutcomes: UInt64 = 0, unmatchedRouters: UInt64 = 0
    private var rejectedInputs: UInt64 = 0
    package let estimatedBytes: UInt64
    package var reservedBytes: UInt64 { Self.capacityBytes }

    /// Round every numeric backing allocation, with a conservative 64-byte Array header, including one copied
    /// status snapshot. Scratch covers four geometry-sized ID arrays and four byte-flag arrays; no observer-owned
    /// Set/Dictionary has an undocumented growth factor. Two ticket buffers plus 16KiB cover control/snapshot work.
    /// Downstream HTTP/history ownership of retained status copies remains serving telemetry's owner.
    package static func estimate(layers: Int, experts: Int, ranks: Int) -> UInt64? {
        guard (1...512).contains(layers), (1...4096).contains(experts), ranks > 0 else { return nil }
        let buckets = min(32, ranks) + 1
        func backing(_ count: Int, stride: Int) -> UInt64 { UInt64(malloc_good_size(count * stride + 64)) }
        let numeric = [
            layers * buckets * rankColumns.count, layers * layerColumns.count, layers * 10,
            layers * 2, layers * 2, layers,
        ].reduce(UInt64(0)) { $0 + backing($1, stride: 8) }
        let scratch = 4 * backing(experts, stride: MemoryLayout<Int>.stride) + 4 * backing(experts, stride: 1)
        let ticketBytes = 2 * backing(ticketLimit, stride: MemoryLayout<TicketState?>.stride)
        return 2 * numeric + scratch + ticketBytes + 16_384
    }
    package init?(
        layers: Int, experts: Int, ranks: Int, ledger: MemoryLedger, retainingBytes: UInt64,
        footprint: UInt64? = nil
    ) {
        guard let estimate = Self.estimate(layers: layers, experts: experts, ranks: ranks), estimate <= Self.capacityBytes,
            let token = try? ledger.reserveOptionalPhysicalIfAvailable(
                bytes: Self.capacityBytes,
                category: "family-prefetch-observations", retainingBytes: retainingBytes, footprint: footprint)
        else { return nil }
        self.layers = layers
        self.experts = experts
        rankLimit = min(32, ranks)
        rankBuckets = rankLimit + 1
        self.ledger = ledger
        reservation = token
        estimatedBytes = estimate
        tickets = Array(repeating: nil, count: Self.ticketLimit)
        self.ranks = Array(repeating: 0, count: layers * rankBuckets * Self.rankColumns.count)
        layerCounts = Array(repeating: 0, count: layers * Self.layerColumns.count)
        // Per layer: planned records by hot/GGUF/pack; completed records and bytes by actual delivery source;
        // successful hot-copy fallbacks. Failed/partial physical bytes are intentionally not invented.
        sources = Array(repeating: 0, count: layers * 10)
        fetchNanoseconds = Array(repeating: 0, count: 2 * layers)
        fetchBytes = Array(repeating: 0, count: 2 * layers)
        predictionNanoseconds = Array(repeating: 0, count: layers)
    }
    deinit { try? ledger.release(reservation) }
    private func rankIndex(_ layer: Int, _ rank: Int, _ column: Int) -> Int {
        (layer * rankBuckets + min(rankLimit, max(0, rank))) * Self.rankColumns.count + column
    }
    private func addRank(_ layer: Int, _ rank: Int, _ column: Int, _ n: UInt64 = 1) { ranks[rankIndex(layer, rank, column)] &+= n }
    private func addLayer(_ layer: Int, _ column: Int, _ n: UInt64 = 1) { layerCounts[layer * Self.layerColumns.count + column] &+= n }

    package func predict(
        layer: Int, ranked: [Int], resident: [Int], inFlight: [Int],
        now: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) {
        lock.lock()
        defer { lock.unlock() }
        abandonLocked()
        guard (0..<layers).contains(layer), ranked.count <= experts, resident.count <= experts, inFlight.count <= experts,
            ranked.allSatisfy({ (0..<experts).contains($0) }), resident.allSatisfy({ (0..<experts).contains($0) }),
            inFlight.allSatisfy({ (0..<experts).contains($0) })
        else {
            rejectedInputs &+= 1
            return
        }
        // A byte per expert bounds all retained membership metadata and deduplicates input without a hash table.
        var flags = [UInt8](repeating: 0, count: experts)
        var unique: [Int] = []
        unique.reserveCapacity(ranked.count)
        for expert in resident { flags[expert] |= 1 }
        for expert in inFlight { flags[expert] |= 2 }
        for expert in ranked where flags[expert] & 8 == 0 {
            unique.append(expert)
            flags[expert] |= 8
        }
        epoch &+= 1
        pending = Prediction(epoch: epoch, began: now, layer: layer, ranked: unique, flags: flags)
        addLayer(layer, 0)
        for (rank, expert) in unique.enumerated() {
            addRank(layer, rank, 0)
            if flags[expert] & 1 == 0 {
                addRank(layer, rank, 1)
                if flags[expert] & 2 == 0 { addRank(layer, rank, 2) } else { addRank(layer, rank, 18) }
            } else {
                addRank(layer, rank, 17)
            }
        }
    }

    /// Admission dispositions concern only fresh predicted records. Already-ready/in-flight exclusions are
    /// counted once by predict(). Depth is a layer aggregate because its ranks can exceed the bounded rank range.
    package func rejected(layer: Int, expert: Int, reason: Rejection) {
        lock.lock()
        defer { lock.unlock() }
        guard let prediction = pending, prediction.layer == layer,
            (0..<experts).contains(expert), prediction.flags[expert] & 3 == 0,
            let rank = prediction.ranked.firstIndex(of: expert)
        else { return }
        addRank(layer, rank, reason.rawValue)
    }
    package func depthLimited(layer: Int, count: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard pending?.layer == layer else { return }
        addLayer(layer, 16, UInt64(max(0, count)))
    }

    /// At most 128 outstanding/awaiting-score outcome tickets. Overflow is explicit and never limits I/O.
    package func issued(layer: Int, expert: Int, planned: Source) -> Ticket? {
        lock.lock()
        defer { lock.unlock() }
        guard var prediction = pending, prediction.layer == layer, let rank = prediction.ranked.firstIndex(of: expert) else {
            unavailableOutcomes &+= 1
            return nil
        }
        guard prediction.flags[expert] & 4 == 0 else { return nil }
        prediction.flags[expert] |= 4
        pending = prediction
        addRank(layer, rank, 6)
        sources[layer * 10 + planned.rawValue] &+= 1
        guard let slot = tickets.firstIndex(where: { $0 == nil }) else {
            droppedTickets &+= 1
            return nil
        }
        serial &+= 1
        tickets[slot] = TicketState(serial: serial, epoch: prediction.epoch, layer: layer, rank: rank, expert: expert, planned: planned)
        return Ticket(slot: slot, serial: serial)
    }

    package func routed(
        layer: Int, needed: [Int], ready: [Int], inFlight: [Int],
        now: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard let prediction = pending, prediction.layer == layer else {
            unmatchedRouters &+= 1
            return
        }
        guard needed.count <= experts, ready.count <= experts, inFlight.count <= experts,
            needed.allSatisfy({ (0..<experts).contains($0) }), ready.allSatisfy({ (0..<experts).contains($0) }),
            inFlight.allSatisfy({ (0..<experts).contains($0) })
        else {
            rejectedInputs &+= 1
            abandonLocked()
            return
        }
        // Bits in this bounded scratch are needed/ready/outstanding at the router snapshot. Count each actual
        // expert once, including later fetch waves, without allocating intersections or difference hash tables.
        var current = [UInt8](repeating: 0, count: experts)
        for expert in needed { current[expert] |= 1 }
        for expert in ready { current[expert] |= 2 }
        for expert in inFlight { current[expert] |= 4 }
        predictionNanoseconds[layer] &+= now >= prediction.began ? now - prediction.began : 0
        addLayer(layer, 1)
        for expert in 0..<experts where current[expert] & 1 != 0 {
            addLayer(layer, 3)
            let flags = prediction.flags[expert]
            if flags & 1 == 0 {
                addLayer(layer, 4)
                if flags & 8 != 0 { addLayer(layer, 6) }
                if flags & 2 == 0 {
                    addLayer(layer, 5)
                    if flags & 8 != 0 { addLayer(layer, 7) }
                }
            } else if current[expert] & 2 == 0 {
                addLayer(layer, 9)
            }
            if current[expert] & 2 != 0 { addLayer(layer, 8) }
        }
        for (rank, expert) in prediction.ranked.enumerated() {
            let flags = prediction.flags[expert]
            let atRouter = current[expert]
            addRank(layer, rank, 14)
            if flags & 1 == 0 {
                addRank(layer, rank, 15)
                if flags & 2 == 0 { addRank(layer, rank, 16) }
            }
            guard atRouter & 1 != 0 else { continue }
            addRank(layer, rank, 3)
            if flags & 1 == 0 {
                addRank(layer, rank, 4)
                if flags & 2 == 0 { addRank(layer, rank, 5) }
            }
            if flags & 4 != 0 {
                addRank(layer, rank, atRouter & 2 != 0 ? 7 : atRouter & 4 != 0 ? 8 : 9)
            }
        }
        for i in tickets.indices where tickets[i]?.epoch == prediction.epoch {
            tickets[i]!.scored = true
            if tickets[i]!.terminal { tickets[i] = nil }
        }
        pending = nil
    }

    package func finished(_ ticket: Ticket?, outcome: Outcome, delivered: Source? = nil, bytes: Int = 0) {
        guard let ticket else { return }
        lock.lock()
        defer { lock.unlock() }
        guard tickets.indices.contains(ticket.slot), var state = tickets[ticket.slot], state.serial == ticket.serial, !state.terminal else {
            return
        }
        switch outcome {
        case .completed:
            addRank(state.layer, state.rank, 10)
            if let delivered {
                sources[state.layer * 10 + 3 + delivered.rawValue] &+= 1
                sources[state.layer * 10 + 6 + delivered.rawValue] &+= UInt64(max(0, bytes))
                if state.planned == .hot && delivered != .hot { sources[state.layer * 10 + 9] &+= 1 }
            } else {
                unavailableOutcomes &+= 1
            }
        case .failed: addRank(state.layer, state.rank, 11)
        case .cancelled: addRank(state.layer, state.rank, 12)
        case .unavailable: unavailableOutcomes &+= 1
        }
        state.terminal = true
        if state.abandoned { addRank(state.layer, state.rank, 13) }
        tickets[ticket.slot] = state.scored || state.abandoned ? nil : state
    }
    package func abandonPrediction() {
        lock.lock()
        defer { lock.unlock() }
        abandonLocked()
    }
    private func abandonLocked() {
        guard let prediction = pending else { return }
        addLayer(prediction.layer, 2)
        for i in tickets.indices where tickets[i]?.epoch == prediction.epoch {
            tickets[i]!.abandoned = true
            if tickets[i]!.terminal {
                addRank(tickets[i]!.layer, tickets[i]!.rank, 13)
                tickets[i] = nil
            }
        }
        pending = nil
    }

    /// Fetch wall includes host scheduling, ready callbacks and waits; read bytes count existing successful demand
    /// accounting. Competitors exclude this fetch's demanded records and already-promoted reads. These are bins for
    /// correlation, not a measured contention penalty or an exclusive storage timer.
    package func demandFetch(layer: Int, queued: Int, running: Int, nanoseconds: UInt64, bytes: Int, succeeded: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard (0..<layers).contains(layer) else {
            rejectedInputs &+= 1
            return
        }
        addLayer(layer, 10)
        if queued + running > 0 { addLayer(layer, 11) }
        addLayer(layer, 12, UInt64(max(0, queued)))
        addLayer(layer, 13, UInt64(max(0, running)))
        addLayer(layer, succeeded ? 14 : 15)
        let bin = layer * 2 + (queued + running > 0 ? 1 : 0)
        fetchNanoseconds[bin] &+= nanoseconds
        fetchBytes[bin] &+= UInt64(max(0, bytes))
    }
    package var snapshot: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        return [
            "experimental": true, "enabled": true, "activity": "eligible_one_row_moe_including_single_row_prefill",
            "scope": "model_instance_cumulative", "instance_id": instanceID, "reserved_bytes": Self.capacityBytes,
            "estimated_internal_bytes": estimatedBytes, "layers": layers, "experts": experts, "rank_buckets": rankBuckets,
            "overflow_rank": rankLimit, "rank_columns": Self.rankColumns, "rank_counts": ranks,
            "layer_columns": Self.layerColumns, "layer_counts": layerCounts,
            "source_columns": [
                "planned_hot", "planned_gguf", "planned_pack", "completed_hot", "completed_gguf", "completed_pack",
                "completed_hot_bytes", "completed_gguf_bytes", "completed_pack_bytes", "hot_fallback_records",
            ],
            "scored_prediction_to_router_nanoseconds": predictionNanoseconds,
            "source_counts": sources, "fetch_wall_nanoseconds": fetchNanoseconds, "successful_demand_read_bytes": fetchBytes,
            "fetch_competitor_bins": ["none", "other_queued_or_running_decode_guess"],
            "pending_prediction": pending != nil, "live_outcome_tickets": tickets.reduce(0) { $0 + ($1 == nil ? 0 : 1) },
            "dropped_outcome_tickets": droppedTickets, "unavailable_outcomes": unavailableOutcomes,
            "unmatched_routers": unmatchedRouters, "rejected_inputs": rejectedInputs,
            "semantics":
                "deltas_require_same_instance_id;completed_sources_cover_all_ticketed_issues;unload_has_no_durable_terminal_snapshot;cold=A-R0;fresh=A-R0-I0;precision_uses_scored_predicted_columns;prediction_elapsed_divisor=layer_scored;ready_is_router_snapshot_not_causal_saved_io;fetch_wall_overlaps_gpu;failed_partial_physical_bytes_unknown",
        ]
    }
}
