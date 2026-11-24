import Foundation
import OrigamiEngine

/// Records of fixed sizes for curve checks: no file, no model.
private final class Sized: HotCopySource, @unchecked Sendable {
    let sizes: [Int: Int]
    init(_ sizes: [Int: Int]) { self.sizes = sizes }
    func hotCopyRecordBytes(_ key: Int) -> Int? { sizes[key] }
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws { memset(into, Int32(key & 0xFF), sizes[key] ?? 0) }
}

/// The usage curve (`HotCopyCurve`): a hand-computed case (two record sizes, tied uses, a key with only misses, a record
/// skipped that does not fit), then its figure against a replay of the counters on the copy `HotCopy.choose` (family) or
/// `RoutingProfile.hotCopyChoice` (Qwen) really builds for each budget: the dashboard's figure matches that replay.
func hotCopyCurve() throws {
    func require(_ ok: Bool, _ what: @autoclosure () -> String) throws { if !ok { throw EngineError("hot copy curve: " + what()) } }
    let small = 16384
    let large = 32768
    let source = Sized(Dictionary(uniqueKeysWithValues: (0..<10).map { ($0, $0 < 5 ? small : large - 100) }))
    let uses = [0: 10, 1: 10, 2: 5, 5: 20, 6: 1]
    let curve = HotCopyCurve(uses: uses, misses: [7: 3], source: source)
    // Order: 7 (missed, no uses), 5, 0 and 1 (tied, lower key first), 2, 6. 46 uses counted.
    for (budget, held) in [
        (32768, 0), (50_000, 10), (65536, 20), (81920, 30), (98304, 40), (100_000, 40), (114_688, 45), (131_072, 45), (147_456, 46),
        (1 << 30, 46),
    ] {
        try require(curve.coverage(UInt64(budget)) == Double(held) / 46, "\(budget) bytes cover \(curve.coverage(UInt64(budget)))")
    }
    try require(curve.full == 147_456, "full at \(curve.full)")
    // A 40 GB copy is a few records short of 40 GB: status quotes it once, not beside the round 40 GB.
    try require(
        HotCopyCurve.quoted(copy: 39_990_000_000) == [20_000_000_000, 39_990_000_000, 80_000_000_000]
            && HotCopyCurve.quoted(copy: 0).count == 3 && HotCopyCurve.quoted(copy: 68_000_000_000).count == 4, "quoted sizes")
    // The dashboard's graph (`popularity`): the same order, a bar per record here.
    let graph = curve.popularity()
    let bars = graph["bars"] as? [[Int]] ?? []
    try require(
        bars.map { $0[2] } == [0, 20, 10, 10, 5, 1] && bars.map { $0[1] } == [32768, 32768, 16384, 16384, 16384, 32768]
            && graph["counted_uses"] as? Int == 46 && graph["bytes"] as? UInt64 == 147_456, "popularity bars \(bars)")
    try require(HotCopyCurve(uses: [:], misses: [3: 1], source: source).coverage(1 << 30) == 0, "no uses counted")

    var rng = SystemRandomNumberGenerator()
    for round in 0..<20 {
        let sizes = Dictionary(uniqueKeysWithValues: (0..<400).map { ($0, Int.random(in: 1...70_000, using: &rng)) })
        let source = Sized(sizes)
        var uses: [Int: Int] = [:]
        var misses: [Int: Int] = [:]
        for key in 0..<400 where Bool.random(using: &rng) { uses[key] = Int.random(in: 1...(round % 2 == 0 ? 3 : 1000), using: &rng) }
        for key in 0..<400 where Int.random(in: 0..<5, using: &rng) == 0 { misses[key] = Int.random(in: 1...4, using: &rng) }
        let curve = HotCopyCurve(uses: uses, misses: misses, source: source)
        let total = uses.values.reduce(0, +)
        for _ in 0..<30 {
            let budget = UInt64.random(in: 0...(400 * 70_000), using: &rng)
            let held = Set(HotCopy.choose(uses, misses: misses, source: source, budget: budget))
            let replay = Double(uses.filter { held.contains($0.key) }.values.reduce(0, +)) / Double(total)
            try require(abs(curve.coverage(budget) - replay) < 1e-12, "family replay at \(budget): \(curve.coverage(budget)) vs \(replay)")
        }
        // At each bar's edge the graph's share of uses is the curve's: one ranking feeds both.
        var edge: UInt64 = 0
        var below = 0
        let runs = curve.popularity(bars: 7)["bars"] as? [[Int]] ?? []
        for run in runs {
            edge += UInt64(run[1])
            below += run[2]
            try require(
                abs(curve.coverage(edge) - Double(below) / Double(total)) < 1e-12, "popularity edge at \(edge) differs from coverage")
        }
        try require(
            runs.count == 7 && runs.map { $0[0] }.reduce(0, +) == Set(uses.keys).union(misses.keys).count,
            "popularity runs \(runs.map { $0[0] })")
    }
    // Qwen: records of one size, ranked by the routing profile (ties in the shipped order).
    let record = 16384
    var counts: (uses: [Int: Int], misses: [Int: Int]) = ([:], [:])
    let keys = ModelContract.layers * ModelContract.experts
    for _ in 0..<3000 { counts.uses[Int.random(in: 0..<keys, using: &rng), default: 0] += Int.random(in: 1...50, using: &rng) }
    for _ in 0..<500 { counts.misses[Int.random(in: 0..<keys, using: &rng), default: 0] += 4 }
    let qwen = HotCopyCurve(
        uses: counts.uses, order: RoutingProfile.hotCopyChoice(counts, budget: .max, record: record), size: { _ in UInt64(record) })
    let total = counts.uses.values.reduce(0, +)
    for budget in [0, 1, 100, 2000, 3000, 3499, 5000, keys].map({ UInt64($0 * record) }) {
        let held = Set(RoutingProfile.hotCopyChoice(counts, budget: budget, record: record))
        let replay = Double(counts.uses.filter { held.contains($0.key) }.values.reduce(0, +)) / Double(total)
        try require(abs(qwen.coverage(budget) - replay) < 1e-12, "Qwen replay at \(budget): \(qwen.coverage(budget)) vs \(replay)")
    }
}

/// The automatic size (`HotCopyCurve.automatic`, applied by `HotCopyKeeper` for a `Place` with `auto`): it grows past
/// the starting size while each tenth of the expert bytes adds at least 2% coverage, stops at the drive's room, at a
/// flat curve and at the whole set; with no counts it is the starting size; an explicit size is never changed; a larger
/// size is reached by a top-up, and a room shrinking within the rebuild tolerance rebuilds nothing.
func hotCopyAutomaticSize() throws {
    func require(_ ok: Bool, _ what: @autoclosure () -> String) throws {
        if !ok { throw EngineError("hot copy automatic size: " + what()) }
    }
    // Family-shaped: 1,000 records of 64 KiB, uses falling as 1/rank: 40% of the bytes hold 88% of the uses; the next
    // tenths add 3.0%, 2.4%, 2.1%, then 1.8%: the size grows from 40% to 70% and stops there.
    let record = 65536
    let source = Sized(Dictionary(uniqueKeysWithValues: (0..<1000).map { ($0, record) }))
    let whole = UInt64(1000 * record)
    let tenth = whole / 10
    let zipf = Dictionary(uniqueKeysWithValues: (0..<1000).map { ($0, 1_000_000 / ($0 + 1)) })
    let curve = HotCopyCurve(uses: zipf, misses: [:], source: source)
    var got = curve.automatic(floor: 4 * tenth, room: .max, whole: whole)
    try require(got.bytes == 7 * tenth && got.limit == .flat, "a paying curve grows to 70%: \(got)")
    got = curve.automatic(floor: 4 * tenth, room: 5 * tenth + 123, whole: whole)
    try require(got.bytes == 5 * tenth + 123 && got.limit == .freeSpace, "free space stops it: \(got)")
    got = curve.automatic(floor: 4 * tenth, room: 3 * tenth, whole: whole)
    try require(got.bytes == 3 * tenth && got.limit == .freeSpace, "the starting size within the room: \(got)")
    got = HotCopyCurve(uses: [:], misses: [:], source: source).automatic(floor: 4 * tenth, room: .max, whole: whole)
    try require(got.bytes == 4 * tenth && got.limit == .noCounts, "no counts keep the starting size: \(got)")
    let narrow = HotCopyCurve(uses: Dictionary(uniqueKeysWithValues: (0..<100).map { ($0, 5) }), misses: [:], source: source)
    got = narrow.automatic(floor: 4 * tenth, room: .max, whole: whole)
    try require(got.bytes == 4 * tenth && got.limit == .flat, "a curve flat beyond the start: \(got)")
    got = narrow.automatic(floor: tenth / 2, room: .max, whole: whole)
    try require(got.bytes == tenth && got.limit == .flat, "a curve that stops growing where every counted record is held: \(got)")
    let even = HotCopyCurve(uses: Dictionary(uniqueKeysWithValues: (0..<1000).map { ($0, 7) }), misses: [:], source: source)
    got = even.automatic(floor: 4 * tenth, room: .max, whole: whole)
    try require(got.bytes == whole && got.limit == .wholeSet, "an even curve takes the whole set: \(got)")
    // Family's first size: half the routed bytes, at most 40 GB and the room above `keepFree` (the keeper's one
    // reserve, not 30 GB), at least 20 GB.
    let gb: UInt64 = 1_000_000_000
    for (room, routed, start) in [
        (25 * gb, 200 * gb, 25 * gb), (100 * gb, 200 * gb, 40 * gb), (100 * gb, 60 * gb, 30 * gb), (5 * gb, 200 * gb, 20 * gb),
    ] {
        let first = HotCopyKeeper.Auto(room: room, whole: routed).familyStart
        try require(first == start, "Family's first size with \(room) room of \(routed): \(first), not \(start)")
    }

    // Qwen-shaped: uses spread over 40% of the experts, ranked by the routing profile.
    let qwenRecord = 16384
    let keys = ModelContract.layers * ModelContract.experts
    let counts = (uses: Dictionary(uniqueKeysWithValues: (0..<(keys * 2 / 5)).map { ($0 * 5 / 2, 3 + $0 % 7) }), misses: [Int: Int]())
    let qwen = HotCopyCurve(
        uses: counts.uses, order: RoutingProfile.hotCopyChoice(counts, budget: .max, record: qwenRecord),
        size: { _ in UInt64(qwenRecord) })
    let qwenWhole = UInt64(keys * qwenRecord)
    got = qwen.automatic(floor: qwenWhole / 5, room: .max, whole: qwenWhole)
    try require(
        got.bytes == qwen.full && got.limit == .flat && qwen.full == UInt64(keys * 2 / 5 * qwenRecord),
        "Qwen-shaped stops at its counted set: \(got)")

    // A larger automatic size is a top-up of the copy it has; a room shrinking within the rebuild tolerance (10%, at
    // least 1 GB) rebuilds nothing; beyond it, it does. Records of 100 MB: a 100 GB expert set, ranked as `zipf`.
    let gigabyte: UInt64 = 1_000_000_000
    let plan = HotCopyKeeper.plan(
        held: Array(0..<400), heldBytes: 40 * gigabyte, chosen: Array(0..<700), budget: 70 * gigabyte, size: { _ in gigabyte / 10 })
    try require(plan == .topUp(Array(400..<700)), "growth is a top-up: \(plan)")
    for (room, expected) in [(66, HotCopyKeeper.Plan.none), (60, .rebuild(Array(0..<600)))] {
        let got = HotCopyKeeper.plan(
            held: Array(0..<700), heldBytes: 70 * gigabyte, chosen: Array(0..<(room * 10)), budget: UInt64(room) * gigabyte,
            size: { _ in gigabyte / 10 })
        try require(got == expected, "room \(room) GB: \(got)")
    }

    // The keeper: an automatic place grows the budget from the counts; an explicit one is left as it is.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-auto-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    for auto in [true, false] {
        let folder = root.appendingPathComponent(auto ? "auto" : "explicit")
        let keeper = HotCopyKeeper(
            source: source, identity: "fixture", copy: nil, counts: (zipf, [:]),
            place: { _ in
                .folder(
                    folder, drive: "fixture", budget: 4 * tenth, free: .max,
                    auto: auto ? HotCopyKeeper.Auto(room: .max, whole: whole) : nil)
            },
            attach: { _ in true }, idle: { true }, event: { _ in })
        defer { keeper.stop(waiting: 3) }
        keeper.trigger()
        let end = Date().addingTimeInterval(20)
        while keeper.report["state"] as? String != "ready" || keeper.records == 0 {
            guard Date() < end else { throw EngineError("hot copy automatic size: keeper \(keeper.report)") }
            Thread.sleep(forTimeInterval: 0.01)
        }
        let report = keeper.report
        let budget = report["budget_bytes"] as? UInt64 ?? 0
        try require(budget == (auto ? 7 * tenth : 4 * tenth), "keeper budget \(budget), auto \(auto)")
        try require(keeper.records == Int(budget) / record, "keeper built \(keeper.records) records")
        try require(report["size_limit"] as? String == (auto ? "flat" : nil), "size limit \(String(describing: report["size_limit"]))")
        try require(report["coverage"] as? Double == curve.coverage(budget), "coverage \(String(describing: report["coverage"]))")
    }
}
