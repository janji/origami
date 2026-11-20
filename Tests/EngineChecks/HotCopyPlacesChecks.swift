import Foundation
import OrigamiEngine
import OrigamiEngineChecks

private func placesRequire(_ value: Bool, _ message: @autoclosure () -> String) throws {
    if !value { throw EngineError("Hot copy places: " + message()) }
}
private func placesRecord(_ expert: Int) -> [UInt8] {
    (0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: $0 &* 7 &+ expert &* 31 &+ 3) }
}
/// A checkpoint whose layer 0 holds experts 0 to 15 (`placesRecord`), in the spans `PackStore` reads them from.
private func placesModel(_ directory: URL) throws -> ModelSource {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent("model.bin")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    let handle = try FileHandle(forWritingTo: file)
    let records = (0..<16).map(placesRecord)
    var spans: [String: Any] = [:]
    var offset = 0
    var recordOffset = 0
    for projection in ExpertProjection.allCases {
        let values = projection.outputs * projection.width
        for (part, bytes, dtype, columns) in [
            ("weight", values / 2, "U32", projection.width / 8), ("scales", values / 32, "BF16", projection.width / 64),
            ("biases", values / 32, "BF16", projection.width / 64),
        ] {
            spans["language_model.model.layers.0.mlp.switch_mlp.\(projection)_proj.\(part)"] = [
                "file": "model.bin", "dtype": dtype, "offset": offset, "length": bytes * 512, "shape": [512, projection.outputs, columns],
            ]
            for (expert, record) in records.enumerated() {
                try handle.seek(toOffset: UInt64(offset + expert * bytes))
                try handle.write(contentsOf: Data(record[recordOffset..<(recordOffset + bytes)]))
            }
            offset += bytes * 512
            recordOffset += bytes
        }
    }
    try handle.truncate(atOffset: UInt64(offset))
    try handle.close()
    return try ModelSource(fixture: directory, spans: spans, fingerprint: String(repeating: "e", count: 64))
}

private final class PlacesFixture: @unchecked Sendable {
    let lock = NSLock()
    var budgets: [UInt64]
    var refusals: [String?]
    var engine: [HotCopy] = []
    var writing = 0, overlapped = false
    /// Records held by any attached copy after each swap: coverage must never drop.
    var coverage: [Int] = []
    init(budgets: [UInt64]) {
        self.budgets = budgets
        refusals = budgets.map { _ in nil }
    }
}

/// `HotCopyPlaces` (F10): one ranking split over the places fastest drive first. `order` puts measured drives by rate,
/// then unmeasured ones in the given order; `fit` takes the next ranks after what faster places hold. Three keepers on
/// three "drives" (speeds 1, 3, 2) fill the fastest with ranks 1-2, the next with 3-4, the slowest with 5-6, no record
/// on two places, one keeper writing at a time; the fastest growing pulls ranks up (the slower places re-split once it
/// holds them, and the records held never drop); a place whose drive is not permitted or full is skipped with its reason
/// (its copy detached, the others serving) and attached again when it comes back; every record reads back exact through
/// one `HotCopySet`.
func hotCopyPlacesSplitBySpeed() throws {
    try placesRequire(HotCopyPlaces.order([1, 3, 2]) == [1, 2, 0], "order by speed")
    try placesRequire(HotCopyPlaces.order([nil, 2, nil, 5]) == [3, 1, 0, 2], "unmeasured last, in the given order")
    try placesRequire(HotCopyPlaces.order([2, 2]) == [0, 1], "a tie keeps the given order")
    // A swap in a slower place replaces a record a faster place also holds before its least used one.
    let uses = (uses: [5: 100, 1: 1, 7: 50], misses: [Int: Int]())
    let swapSlots: [(key: Int, bytes: UInt64)] = [(5, 2), (1, 2)]
    try placesRequire(
        HotCopyKeeper.swaps(slots: swapSlots, chosen: [7], counts: uses, size: { _ in 2 }, limit: 10).map(\.slot) == [1], "least used first"
    )
    try placesRequire(
        HotCopyKeeper.swaps(slots: swapSlots, chosen: [7], counts: uses, size: { _ in 2 }, limit: 10, redundant: [5]).map(\.slot) == [0],
        "a duplicate is the first victim")
    let size: (Int) -> UInt64 = { $0 == 3 ? 3 : 2 }
    try placesRequire(HotCopyPlaces.fit([0, 1, 2, 3, 4, 5], excluding: [], budget: 4, size: size) == [0, 1], "first place")
    try placesRequire(
        HotCopyPlaces.fit([0, 1, 2, 3, 4, 5], excluding: [0, 1], budget: 4, size: size) == [2, 4], "a record that does not fit is skipped")
    // Room left once every rank outside the faster places is in: the top ranks again (redundancy), after the new ones.
    try placesRequire(HotCopyPlaces.fit([0, 1, 2, 3], excluding: [0, 1, 2, 3], budget: 4, size: size) == [0, 1], "all held faster")
    try placesRequire(
        HotCopyPlaces.fit([0, 1, 2, 4], excluding: [0, 1], budget: 8, size: size) == [2, 4, 0, 1], "coverage, then duplicates")
    try placesRequire(
        HotCopyPlaces.fit([0, 1, 2, 4], excluding: [0, 1], budget: 6, size: size) == [2, 4, 0], "a smaller cap drops duplicates first")
    try placesRequire(
        HotCopyPlaces.fit([0, 1, 2, 4, 5], excluding: [0, 1], budget: 6, size: size) == [2, 4, 5], "no duplicates before coverage")

    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-places-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // The fixture drives' free space is the check's, never this Mac's (swaps fill empty slots only above the reserve).
    DiskBudget.shared.freeForChecks = { _ in 1_000_000_000_000 }
    defer {
        DiskBudget.shared.freeForChecks = nil
        try? FileManager.default.removeItem(at: root)
    }
    let model = try placesModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model, integrity: .sha256)
    let identity = PackStore.hotCopyIdentity(model)
    let padded = UInt64((PackStore.expertBytes + 16383) / 16384 * 16384)
    let state = PlacesFixture(budgets: [2 * padded, 2 * padded, 2 * padded])
    let counts = (uses: Dictionary(uniqueKeysWithValues: (0..<16).map { ($0, 100 - $0) }), misses: [Int: Int]())
    let places: [(place: (_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place, copy: HotCopy?)] = (0..<3).map { n in
        let folder = root.appendingPathComponent("drive\(n)/m/hot")
        return (
            { _ in
                let (budget, refusal) = state.lock.withLock { (state.budgets[n], state.refusals[n]) }
                if let refusal { return .none(state: refusal, detail: "fixture \(n): \(refusal)") }
                return .folder(folder, drive: "fixture \(n)", budget: budget, free: .max)
            }, nil
        )
    }
    let set = HotCopyPlaces(
        source: source, identity: identity, places: places, speeds: [1, 3, 2], counts: ([:], [:]),
        attach: { copies in
            state.lock.withLock {
                state.engine = copies
                state.coverage.append(Set(copies.flatMap(\.keys)).count)
            }
            return true
        }, idle: { true },
        withIdle: { work in
            let alone = state.lock.withLock {
                state.writing += 1
                if state.writing > 1 { state.overlapped = true }
                return true
            }
            defer { state.lock.withLock { state.writing -= 1 } }
            if alone { try work() }
            return true
        }, event: { _ in })
    defer { set.stop(waiting: 3) }
    func held() -> [[Int]] { set.keepers.map { $0.current?.keys ?? [] } }
    func waitFor(_ what: String, _ condition: () -> Bool) throws {
        let end = Date().addingTimeInterval(60)
        while !condition() {
            guard Date() < end else { throw EngineError("Hot copy places: \(what) (\(held()), \(set.keepers.map(\.report)))") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }
    func settled() -> Bool { set.keepers.allSatisfy { !$0.building && ["ready", "learning"].contains($0.report["state"] as? String) } }
    try placesRequire(set.ranking == [1, 2, 0], "ranking")
    set.trigger(counts: counts)
    try waitFor("ranks split by speed") { settled() && held().map { Set($0) } == [[4, 5], [0, 1], [2, 3]] }
    try placesRequire(!state.lock.withLock { state.overlapped }, "two keepers wrote at once")
    try placesRequire(
        state.lock.withLock { state.engine.map(\.keys).map(Set.init) } == [[0, 1], [2, 3], [4, 5]], "attached fastest first")
    // However many passes the keepers make, the fastest place keeps exactly the top ranks (a slower place's duplicate
    // never makes it a victim there).
    func fastestKeepsTop(_ top: Set<Int>) throws {
        for pass in 0..<3 {
            set.trigger(counts: counts)
            Thread.sleep(forTimeInterval: 0.1)
            try waitFor("pass \(pass) settled") { settled() }
            try placesRequire(Set(held()[1]) == top, "the fastest place after pass \(pass): \(held())")
        }
    }
    try fastestKeepsTop([0, 1])

    // The fastest place grows by two records: it takes ranks 3-4, the next place moves to 5-6, the slowest to 7-8.
    state.lock.withLock { state.budgets[1] = 4 * padded }
    let before = state.lock.withLock { state.coverage.count }
    set.trigger(force: true)
    try waitFor("ranks pulled up") {
        settled() && held().map { Set($0) } == [[6, 7], [0, 1, 2, 3], [4, 5]]
    }
    try fastestKeepsTop([0, 1, 2, 3])
    let all = held().flatMap { $0 }
    try placesRequire(all.count == Set(all).count, "a record on two places: \(held())")
    let after = state.lock.withLock { Array(state.coverage[before...]) }
    try placesRequire(zip(after, after.dropFirst()).allSatisfy { $0 <= $1 } && (after.first ?? 6) >= 6, "coverage dropped: \(after)")
    try placesRequire(!state.lock.withLock { state.overlapped }, "two keepers wrote at once while re-splitting")

    // The middle place's drive is not permitted: skipped with its reason, its copy detached, the others still read.
    state.lock.withLock { state.refusals[2] = "not_allowed" }
    set.trigger()
    try waitFor("a place skipped") { set.keepers[2].report["state"] as? String == "not_allowed" && set.keepers[2].current == nil }
    try placesRequire(
        state.lock.withLock { state.engine.count } == 2 && set.keepers[1].current != nil && set.keepers[0].current != nil,
        "the others keep serving")
    // Full: the same, with its own reason.
    state.lock.withLock { state.refusals[2] = "not_possible" }
    set.trigger()
    try waitFor("a full place skipped") { set.keepers[2].report["state"] as? String == "not_possible" }
    // Back: its finished copy is attached again (opened after the validity check), nothing rewritten.
    state.lock.withLock { state.refusals[2] = nil }
    set.trigger()
    try waitFor("a place back") { settled() && set.keepers[2].current != nil && state.lock.withLock { state.engine.count } == 3 }
    try placesRequire(Set(held()[2]) == [4, 5], "the returned place's records \(held())")

    // The slowest place's cap grows past every rank no faster place holds: the rest go in first, then ranks 1-2 again.
    state.lock.withLock { state.budgets[0] = 12 * padded }
    set.trigger(force: true)
    try waitFor("duplicates of the top ranks") { settled() && Set(held()[0]) == Set([0, 1] + Array(6..<16)) }
    try placesRequire(set.duplicates(0) == 2 && set.duplicates(1) == 0 && set.duplicates(2) == 0, "duplicates counted")
    let reading = HotCopySet(state.lock.withLock { state.engine })
    try placesRequire(reading.sources(0).count == 2 && reading.sources(2).count == 1 && reading.sources(99).isEmpty, "sources per record")
    for key in 0..<8 {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { _ = try reading.read(key, into: $0) }
        try placesRequire(bytes == placesRecord(key), "key \(key) from the places")
    }
}

/// Qwen's reads with copies on two "drives" (`PackStore.attachHotCopies`): each record comes exact from a copy holding
/// it; one drive's file going away mid-run (a read error) gives way to the other copy, then to the model's spans, so no
/// read fails while the model's files hold the record.
func qwenHotCopiesOnTwoDrives() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-qwen-places-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try placesModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model)
    let identity = PackStore.hotCopyIdentity(model)
    let a = try HotCopy.write(source: source, keys: [0, 1, 2, 3], folder: root.appendingPathComponent("driveA/hot"), identity: identity)
    let b = try HotCopy.write(source: source, keys: [2, 3, 4, 5], folder: root.appendingPathComponent("driveB/hot"), identity: identity)
    let store = try PackStore(model: model, log: { _ in })
    store.attachHotCopies([a, b])
    func read(_ expert: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { try store.readInternalExpert(ExpertKey(layer: 0, expert: expert), into: $0) }
        return bytes
    }
    for _ in 0..<2 { for key in 0..<8 { try placesRequire(try read(key) == placesRecord(key), "key \(key)") } }
    // A record a swap takes out of a copy's lookup leaves that record's sources at once; put back, it is offered again.
    let both = HotCopySet([a, b])
    let slot = a.slots.firstIndex { $0.key == 2 }!
    try placesRequire(both.sources(2) == [0, 1], "sources \(both.sources(2))")
    let owner = a.vacate(slot)!
    try placesRequire(both.sources(2) == [1] && !a.holds(2), "a vacated slot is no source")
    a.restore(slot, owner: owner)
    try placesRequire(both.sources(2) == [0, 1], "a restored slot is a source again")
    try placesRequire(store.status().hotCopyReads == 12 && store.attachedHotCopies.count == 2, "copy reads \(store.status().hotCopyReads)")
    // Drive B unplugged mid-run: keys 2-3 from A, 4-5 from the spans, exact.
    try FileHandle(forWritingTo: b.payload).truncate(atOffset: 0)
    for key in 0..<8 { try placesRequire(try read(key) == placesRecord(key), "key \(key) without drive B") }
    try placesRequire(b.disabled && store.hasHotExpert(try ExpertKey(layer: 0, expert: 2)), "the gone drive is skipped")
    try placesRequire(!store.hasHotExpert(try ExpertKey(layer: 0, expert: 4)), "a record only the gone drive held")
}
