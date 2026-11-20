import Foundation
import OrigamiEngine

func familyRouteTraceBoundsAndOwnership() throws {
    func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
        if !value() { throw EngineError("Route trace: " + message) }
    }
    let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
    var trace = FamilyRouteTrace(layers: 40, experts: 384, ranks: 6, ledger: ledger)
    try require(trace != nil && ledger.reservedBytes == FamilyRouteTrace.reservedBytes, "uncharged diagnostic storage")
    trace!.record(position: 615, layer: 0, needed: [1, 2])
    trace!.recordPrefill(rows: 4, eligible: true)
    trace!.capturesDecode = true
    trace!.record(position: 615, layer: 0, needed: [1, 2, 3, 4, 5, 6])
    trace!.record(position: 615, layer: 1, needed: [8, 9, 10, 11])
    trace!.record(position: -1, layer: 1, needed: [8])
    trace!.record(position: 616, layer: 40, needed: [8])
    trace!.record(position: 616, layer: 1, needed: [384])
    trace!.recordDetail(
        position: 615, layer: 1, chosen: [(8, 0.75), (40, 0.25)], guess: [(3, 0.5), (1, 0.25), (2, 0.125)], resident: [0, 40, 383])
    trace!.recordDetail(position: 615, layer: 1, chosen: [], guess: (0..<19).map { ($0, 0) }, resident: [])
    let saved = trace!.snapshot
    let detail = saved["details"] as? [Int32] ?? []
    try require(
        saved["detail_stride"] as? Int == 62 && detail.count == 62 && detail[0..<4] == [615, 1, 8, 40]
            && detail[8] == Int32(bitPattern: Float(0.75).bitPattern) && detail[14..<18] == [3, 1, 2, -1]
            && detail[33] == Int32(bitPattern: Float(0.25).bitPattern)
            && detail[50] == 1 && detail[51] == 1 << 8 && detail[61] == Int32(bitPattern: 1 << 31),
        "detail row geometry changed, or an oversized guess was kept")
    try require(saved["count"] as? Int == 2 && saved["rejected"] as? Int == 3, "prefill or invalid rows contaminated decode")
    try require(saved["rows"] as? [Int32] == [615, 0, 6, 1, 2, 3, 4, 5, 6, 615, 1, 4, 8, 9, 10, 11, -1, -1], "row geometry changed")
    try require(saved["prefill_grouped_eligible_calls"] as? Int == 1, "eligible prefill work lost")
    for _ in 0..<FamilyRouteTrace.capacity { trace!.record(position: 616, layer: 2, needed: [7]) }
    try require(
        trace!.snapshot["count"] as? Int == FamilyRouteTrace.capacity && trace!.snapshot["dropped"] as? Int == 2,
        "capacity overflow was hidden")
    trace!.reset()
    trace!.capturesDecode = true
    trace!.record(position: 700, layer: 0, needed: [12])
    try require(trace!.snapshot["rows"] as? [Int32] == [700, 0, 1, 12, -1, -1, -1, -1, -1], "prior request survived reset")
    trace!.recordMissing(score: 0.5, alternative: 0.25, share: 0.125, rank: 2)
    trace!.recordMissing(score: 0.75, alternative: nil, share: 0.5, rank: 0)
    trace!.recordMissing(score: .nan, alternative: 0.1, share: 0.1, rank: 1)
    try require(
        trace!.snapshot["missing"] as? [Double] == [0.5, 0.25, 1, 0.125, 2, 0.75, 0, 0, 0.5, 0] && trace!.snapshot["rejected"] as? Int == 1,
        "near-tie entries lost, or a non-finite score was kept")
    try require(JSONSerialization.isValidJSONObject(trace!.snapshot), "near-tie entries cannot be retained as JSON evidence")
    try require(saved["count"] as? Int == 2 && (saved["rows"] as? [Int32])?.first == 615, "published snapshot mutated")
    try require(JSONSerialization.isValidJSONObject(saved), "trace cannot be retained as JSON evidence")
    trace = nil
    try require(ledger.reservedBytes == 0, "diagnostic owner leaked its charge")
    ledger.setOptionalGrowthAllowed(false)
    try require(FamilyRouteTrace(layers: 40, experts: 384, ranks: 6, ledger: ledger) == nil, "optional pressure latch ignored")
    ledger.setOptionalGrowthAllowed(true)
    try require(
        FamilyRouteTrace(layers: 40, experts: 384, ranks: 6, ledger: ledger, retainingBytes: 32 << 20) == nil,
        "diagnostic owner borrowed mandatory capacity")
    try require(
        FamilyRouteTrace(layers: 40, experts: 384, ranks: 33, ledger: ledger) == nil && ledger.reservedBytes == 0,
        "unsupported geometry allocated")
}
