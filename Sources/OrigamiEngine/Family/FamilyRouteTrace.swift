import Foundation

/// Opt-in, bounded diagnostic data for expert-union and reuse experiments. The serial model owner records
/// selected IDs after routing; there are no extra model passes, GPU waits, activation copies or weight reads.
/// A snapshot describes observed sequential routes, not routes a speculative draft would necessarily produce.
package final class FamilyRouteTrace {
    package static let capacity = 16384
    /// Chosen-but-not-resident entries kept for the near-tie (O52) diagnostic; five floats each, well inside the reservation.
    package static let missingCapacity = 12288
    package static let reservedBytes: UInt64 = 16 << 20
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    private let layers, experts, ranks, stride: Int
    private var words: [Int32]
    private var count = 0
    private var dropped = 0
    private var rejected = 0
    private var prefillRows = [Int](repeating: 0, count: 10)
    private var eligiblePrefillCalls = 0
    private var missing: [Float] = []
    private var missingDropped = 0
    /// Per decode row, for scoring guess rules offline: the chosen experts' weights in selection order, the next layer's
    /// guess ranked by its score (up to three times the trained count) with those scores, and which of this layer's experts were in memory.
    private var details: [Int32] = []
    private let detailStride: Int
    private let detailCapacity: Int
    package var capturesDecode = false

    package init?(
        layers: Int, experts: Int, ranks: Int, ledger: MemoryLedger, retainingBytes: UInt64 = 0,
        footprint: UInt64? = nil
    ) {
        guard (1...512).contains(layers), (1...4096).contains(experts), (1...32).contains(ranks),
            let token = try? ledger.reserveOptionalPhysicalIfAvailable(
                bytes: Self.reservedBytes,
                category: "family-route-trace", retainingBytes: retainingBytes, footprint: footprint)
        else { return nil }
        self.layers = layers
        self.experts = experts
        self.ranks = ranks
        stride = ranks + 3
        detailStride = 2 + 2 * ranks + 6 * ranks + (experts + 31) / 32
        // The rows take under 2.3 MiB at the largest geometry; the details what is left of half the reservation (the
        // other half covers snapshot copying).
        detailCapacity = min(Self.capacity, (Int(Self.reservedBytes / 2) - 4 * Self.capacity * (ranks + 3)) / (4 * detailStride))
        self.ledger = ledger
        reservation = token
        words = Array(repeating: -1, count: Self.capacity * (ranks + 3))
    }
    deinit { try? ledger.release(reservation) }

    package func reset() {
        count = 0
        dropped = 0
        rejected = 0
        capturesDecode = false
        prefillRows = Array(repeating: 0, count: 10)
        eligiblePrefillCalls = 0
        missing = []
        missingDropped = 0
        details = []
    }
    /// `chosen` this layer's experts and weights in selection order, `guess` the next MoE layer's ranked guess (empty
    /// without one), `resident` this layer's experts in memory when it routed.
    package func recordDetail(
        position: Int, layer: Int, chosen: [(expert: Int, weight: Float)], guess: [(expert: Int, score: Float)], resident: Set<Int>
    ) {
        guard capturesDecode, details.count < detailCapacity * detailStride, let p = Int32(exactly: position),
            chosen.count <= ranks, guess.count <= 3 * ranks
        else { return }
        var row = [Int32](repeating: -1, count: detailStride)
        row[0] = p
        row[1] = Int32(layer)
        for (rank, pick) in chosen.enumerated() {
            row[2 + rank] = Int32(pick.expert)
            row[2 + ranks + rank] = Int32(bitPattern: pick.weight.bitPattern)
        }
        for (rank, pick) in guess.enumerated() {
            row[2 + 2 * ranks + rank] = Int32(pick.expert)
            row[2 + 5 * ranks + rank] = Int32(bitPattern: pick.score.bitPattern)
        }
        let mask = 2 + 8 * ranks
        for word in 0..<(experts + 31) / 32 { row[mask + word] = 0 }
        for expert in resident where (0..<experts).contains(expert) {
            row[mask + expert / 32] |= Int32(bitPattern: 1 << UInt32(expert % 32))
        }
        details += row
    }
    /// One chosen expert that was not in memory: its selection score, the best unchosen resident expert's score (if any),
    /// its share of the chosen gate weight, and its selection rank.
    package func recordMissing(score: Float, alternative: Float?, share: Float, rank: Int) {
        guard capturesDecode, score.isFinite, share.isFinite, alternative?.isFinite ?? true else {
            rejected += 1
            return
        }
        guard missing.count < Self.missingCapacity * 5 else {
            missingDropped += 1
            return
        }
        missing += [score, alternative ?? 0, alternative == nil ? 0 : 1, share, Float(rank)]
    }
    package func record(position: Int, layer: Int, needed: [Int]) {
        guard capturesDecode else { return }
        guard let p = Int32(exactly: position), p >= 0, (0..<layers).contains(layer),
            !needed.isEmpty, needed.count <= ranks, needed.allSatisfy({ (0..<experts).contains($0) })
        else {
            rejected += 1
            return
        }
        guard count < Self.capacity else {
            dropped += 1
            return
        }
        let start = count * stride
        words[start] = p
        words[start + 1] = Int32(layer)
        words[start + 2] = Int32(needed.count)
        for rank in 0..<ranks { words[start + 3 + rank] = rank < needed.count ? Int32(needed[rank]) : -1 }
        count += 1
    }
    package func recordPrefill(rows: Int, eligible: Bool) {
        guard !capturesDecode, rows > 0 else { return }
        prefillRows[min(rows, 9)] += 1
        if eligible { eligiblePrefillCalls += 1 }
    }
    package var snapshot: [String: Any] {
        [
            "enabled": true, "reserved_bytes": Self.reservedBytes, "capacity": Self.capacity,
            "count": count, "dropped": dropped, "rejected": rejected, "row_stride": stride,
            "columns": ["position", "layer", "needed_count"] + (0..<ranks).map { "expert_\($0)" },
            "rows": Array(words.prefix(count * stride)), "prefill_expert_calls_by_rows": prefillRows,
            "prefill_row_bucket_9": "9 or more", "prefill_grouped_eligible_calls": eligiblePrefillCalls,
            "missing_columns": ["score", "best_resident_alternative", "has_alternative", "gate_share", "rank"],
            "missing": missing.map(Double.init), "missing_dropped": missingDropped,
            "detail_stride": detailStride, "detail_capacity": detailCapacity, "details": details,
            "detail_columns":
                "position, layer, \(ranks) chosen experts, their \(ranks) weights (float bits), the next layer's guess ranked (\(3 * ranks)), their selection scores (float bits), resident mask words (bit e of word e/32)",
            "scope":
                "current request attempt; sequential decode selections, excluding prefill; IDs are layer-local; -1 pads unused ranks; no physical-I/O or speculative acceptance claim",
        ]
    }
}
