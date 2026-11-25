import Foundation
import Metal
import OrigamiEngine
import OrigamiEngineChecks

private func multiRequire(_ value: Bool, _ message: @autoclosure () -> String) throws {
    if !value { throw EngineError("Multi-drive copies: " + message()) }
}
private func multiRecord(_ expert: Int) -> [UInt8] {
    (0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: $0 &* 13 &+ expert &* 101 &+ 5) }
}
/// A checkpoint whose layer 0 holds experts 0 to 7 (`multiRecord`), in the spans `PackStore` reads them from.
private func multiSpanModel(_ directory: URL) throws -> ModelSource {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent("model.bin")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    let handle = try FileHandle(forWritingTo: file)
    let records = (0..<8).map(multiRecord)
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
    return try ModelSource(fixture: directory, spans: spans, fingerprint: String(repeating: "d", count: 64))
}
private func multiRoot(_ name: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-\(name)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
private func corrupt(_ copy: HotCopy, key: Int) throws {
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.seek(toOffset: UInt64(copy.offsets[key]!) + 11)
    try handle.write(contentsOf: Data([0x5A ^ multiRecord(key)[11]]))
    try handle.close()
}

/// The shared choice among N sources and the shared reader of N `HotCopy` stores (one per drive): `IOScheduler.earliest`
/// takes the smallest finite estimate, the first on a tie; `HotCopySet.pick` probes each idle store before measuring
/// and then spreads by reads in flight; N stores on N folders read every record exactly as one store (also from eight
/// threads at once, each store serving), a record failing its CRC32 in one store comes from another, a store whose
/// file goes away mid-run gives way to the others, and a record no store can read throws so the caller reads the
/// model's files.
func hotCopySetReadsSeveralStores() throws {
    try multiRequire(
        IOScheduler.earliest([2, 1, 1]) == 1 && IOScheduler.earliest([nil, 3, 3]) == 1 && IOScheduler.earliest([1, 1]) == 0
            && IOScheduler.earliest([nil, nil]) == nil && IOScheduler.earliest([.infinity]) == nil && IOScheduler.earliest([]) == nil,
        "earliest")
    var idle = [ReadLane(), ReadLane(), ReadLane()]
    try multiRequire(HotCopySet.pick(idle, placements: 1, bytes: 1 << 20) == 0, "first probe")
    idle[0].lastSelected = 1
    try multiRequire(HotCopySet.pick(idle, placements: 2, bytes: 1 << 20) == 1, "probe the one chosen longest ago")
    var measured = [ReadLane(), ReadLane()]
    for n in measured.indices {
        for _ in 0..<8 { measured[n].served(seconds: n == 0 ? 0.004 : 0.001, bytes: 1 << 20) }
        measured[n].lastSelected = 100
    }
    try multiRequire(HotCopySet.pick(measured, placements: 101, bytes: 1 << 20) == 1, "the faster drive")
    measured[1].inFlight = 4
    measured[0].inFlight = 0
    // 5 × 1 ms queued on the fast one against 1 × 4 ms on the slow one.
    try multiRequire(HotCopySet.pick(measured, placements: 102, bytes: 1 << 20) == 0, "a busy fast drive spreads to the slow one")

    let root = try multiRoot("hot-copy-set")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try multiSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model)
    // Three "drives": keys 0-3, 2-5 and 4-7 (each key on one or two of them).
    let stores = try (0..<3).map { n in
        try HotCopy.write(
            source: source, keys: Array((2 * n)..<(2 * n + 4)), folder: root.appendingPathComponent("drive\(n)/copy"), identity: "m")
    }
    let one = try HotCopy.write(source: source, keys: Array(0..<8), folder: root.appendingPathComponent("one/copy"), identity: "m")
    let set = HotCopySet(stores)
    let single = HotCopySet([one])
    try multiRequire(
        Set(set.heldOffsets.keys) == Set(0..<8) && single.heldOffsets == one.offsets && (0..<8).allSatisfy(set.holds) && !set.holds(8),
        "membership")
    @Sendable func read(_ set: HotCopySet, _ key: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { _ = try set.read(key, into: $0) }
        return bytes
    }
    for key in 0..<8 {
        let many = try read(set, key)
        let alone = try read(single, key)
        try multiRequire(many == multiRecord(key) && alone == many, "key \(key)")
    }
    final class Lines: @unchecked Sendable {
        let lock = NSLock()
        var all: [String] = []
        func add(_ line: String) { lock.withLock { all.append(line) } }
    }
    let wrong = Lines()
    DispatchQueue.concurrentPerform(iterations: 8) { thread in
        for n in 0..<24 {
            let key = (thread + n) % 8
            do { if try read(set, key) != multiRecord(key) { wrong.add("bytes \(key)") } } catch { wrong.add("\(error)") }
        }
    }
    try multiRequire(
        wrong.all.isEmpty && set.reads.allSatisfy { $0 > 0 } && set.reads.reduce(0, +) == 8 + 192, "concurrent \(wrong.all) \(set.reads)")

    // A record failing its CRC32 in the store chosen first comes from the other store holding it, exact.
    let fresh = HotCopySet((0..<3).map { HotCopy.open(folder: stores[$0].folder, identity: "m", source: source)! })
    try corrupt(fresh.stores[0], key: 2)
    try multiRequire(try read(fresh, 2) == multiRecord(2) && fresh.reads == [0, 1, 0], "a bad record \(fresh.reads)")
    try multiRequire(!fresh.stores[0].holds(2) && fresh.holds(2) && fresh.heldOffsets[2] == fresh.stores[1].offsets[2], "dropped alone")
    // A store whose file goes away: its first failed read pauses it and the same read comes from another store.
    let saved = try Data(contentsOf: fresh.stores[1].payload)
    fresh.stores[1].retryDelay = 0.2
    try FileHandle(forWritingTo: fresh.stores[1].payload).truncate(atOffset: 0)
    for key in [3, 4, 5] { try multiRequire(try read(fresh, key) == multiRecord(key), "key \(key) after a drive went away") }
    try multiRequire(fresh.stores[1].disabled && !fresh.holds(2) && fresh.holds(3), "the gone store is skipped: \(fresh.reads)")
    // Key 2: bad in store 0, gone with store 1: no store reads it, so the caller reads the model's files.
    var refused = false
    do { _ = try read(fresh, 2) } catch { refused = true }
    try multiRequire(refused, "a record no store can read")
    // The drive comes back (the same file, its bytes again): once the pause is over, the store is offered and read again.
    let handle = try FileHandle(forWritingTo: fresh.stores[1].payload)
    try handle.write(contentsOf: saved)
    try handle.close()
    let end = Date().addingTimeInterval(10)
    while !fresh.holds(2) {
        guard Date() < end else { throw EngineError("Multi-drive copies: the returned store was never offered again") }
        Thread.sleep(forTimeInterval: 0.02)
    }
    let before = fresh.reads[1]
    try multiRequire(try read(fresh, 2) == multiRecord(2) && fresh.reads[1] == before + 1, "the returned store \(fresh.reads)")
}

/// Qwen's further copies: an older-format mirror (`mirror.json`, read in place) beside new-format `HotCopy` stores
/// (`copy/` in the store's model folder) on other "drives", read by `PackStore` with the shared choice. Every record
/// comes back exact from whichever copy holds it, each copy serving; a record failing its CRC32 in a new store, a new
/// store whose file goes away and a mirror whose file goes away all give way to another copy and finally to the model's
/// spans, with the same bytes.
func packStoreReadsMirrorBesideNewStores() throws {
    let root = try multiRoot("mirror-beside-stores")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try multiSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model)
    let identity = PackStore.hotCopyIdentity(model)
    // Drive A: an older mirror of keys 0 and 1, written by the 3.0 builder (digests checked on every read here).
    let mirror = root.appendingPathComponent("driveA/m")
    try LegacyMirrorFixture.build(model: model, folder: mirror, records: 2, ranking: [0, 1])
    // Drives B and C: new-format stores of keys 1-4 and 3-6.
    let b = root.appendingPathComponent("driveB/m")
    let c = root.appendingPathComponent("driveC/m")
    let newB = try HotCopy.write(
        source: source, keys: [1, 2, 3, 4], folder: b.appendingPathComponent(PackStore.furtherCopyFolder), identity: identity)
    let newC = try HotCopy.write(
        source: source, keys: [3, 4, 5, 6], folder: c.appendingPathComponent(PackStore.furtherCopyFolder), identity: identity)
    // Another model's store is not opened.
    let foreign = root.appendingPathComponent("driveD/m")
    _ = try HotCopy.write(
        source: source, keys: [7], folder: foreign.appendingPathComponent(PackStore.furtherCopyFolder), identity: "qwen other")
    let store = try PackStore(model: model, copies: [mirror, b, c, foreign], integrity: .sha256, log: { _ in })
    try multiRequire(
        store.status().copies?.map(\.experts) == [2, 4, 4, 0] && store.status().copies?[3].problem != nil,
        "opened \(String(describing: store.status().copies))")
    func read(_ key: Int, _ from: ExpertReadSource = .packed) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { try store.readExpert(layer: 0, expert: key, into: $0, source: from) }
        return bytes
    }
    for _ in 0..<3 { for key in 0..<7 { try multiRequire(try read(key) == multiRecord(key), "key \(key)") } }
    let reads = store.status().copies!.map(\.reads)
    try multiRequire(reads[0] > 0 && reads[1] > 0 && reads[2] > 0 && reads.reduce(0, +) == 21, "each copy serves: \(reads)")
    var refused = false
    do { _ = try read(7) } catch { refused = true }
    let fromModel = try read(7, .automatic)
    try multiRequire(refused && fromModel == multiRecord(7), "a key no copy holds reads from the model")
    // Key 2 is only in store B: a bad record there reads from the model's spans, exact; B keeps serving its others.
    try corrupt(newB, key: 2)
    let restarted = try PackStore(model: model, copies: [mirror, b, c], integrity: .sha256, log: { _ in })
    func readRestarted(_ key: Int, _ from: ExpertReadSource = .automatic) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { try restarted.readExpert(layer: 0, expert: key, into: $0, source: from) }
        return bytes
    }
    try multiRequire(try readRestarted(2) == multiRecord(2), "a bad record in a new store")
    refused = false
    do { _ = try readRestarted(2, .packed) } catch { refused = true }
    let other = try readRestarted(1, .packed)
    try multiRequire(refused && other == multiRecord(1), "the bad record is dropped alone")
    // Store C's file goes away: keys 3-6 come from B, then the model's spans; the mirror still serves 0 and 1.
    try FileHandle(forWritingTo: newC.payload).truncate(atOffset: 0)
    for _ in 0..<2 { for key in 3..<7 { try multiRequire(try readRestarted(key) == multiRecord(key), "key \(key) without C") } }
    // The mirror's file goes away: keys 0 and 1 come from B or the spans.
    try FileHandle(forWritingTo: mirror.appendingPathComponent("mirror.bin")).truncate(atOffset: 0)
    for _ in 0..<2 { for key in 0..<2 { try multiRequire(try readRestarted(key) == multiRecord(key), "key \(key) without the mirror") } }
    try multiRequire(restarted.status().copies?[0].problem != nil, "the cut mirror is dropped")
}

/// Two keepers keep one model's copies on two "drives": the second ranks only records the first does not hold. Each
/// builds within its own size, the first then grows (a forced top-up) within its new size, never past it, and every
/// record of both reads back exactly through one `HotCopySet`.
func hotCopyKeepersBuildTwoStores() throws {
    let root = try multiRoot("two-keepers")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try multiSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model, integrity: .sha256)
    let identity = PackStore.hotCopyIdentity(model)
    let padded = UInt64((PackStore.expertBytes + 16383) / 16384 * 16384)
    final class Attached: @unchecked Sendable {
        let lock = NSLock()
        var copies: [HotCopy?] = [nil, nil]
        var budgets: [UInt64]
        init(_ budgets: [UInt64]) { self.budgets = budgets }
    }
    let state = Attached([2 * padded, 3 * padded])
    let counts = (uses: [0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2], misses: [Int: Int]())
    func keeper(_ n: Int) -> HotCopyKeeper {
        let folder = root.appendingPathComponent("drive\(n)/m/copy")
        return HotCopyKeeper(
            source: source, identity: identity, copy: nil, counts: ([:], [:]),
            choose: { counts, budget in
                // The second drive's copy holds the next records after the first's.
                let held = n == 0 ? [] : Set(state.lock.withLock { state.copies[0]?.keys } ?? [])
                return HotCopy.choose(counts.uses.filter { !held.contains($0.key) }, source: source, budget: budget)
            },
            place: { _ in .folder(folder, drive: "fixture \(n)", budget: state.lock.withLock { state.budgets[n] }, free: .max) },
            attach: { copy in
                state.lock.withLock { state.copies[n] = copy }
                return true
            }, idle: { true }, event: { _ in })
    }
    let first = keeper(0)
    let second = keeper(1)
    defer {
        first.stop(waiting: 3)
        second.stop(waiting: 3)
    }
    func waitFor(_ what: String, _ condition: () -> Bool) throws {
        let end = Date().addingTimeInterval(30)
        while !condition() {
            guard Date() < end else { throw EngineError("Multi-drive copies: \(what) (\(first.report), \(second.report))") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }
    first.trigger(counts: counts)
    try waitFor("the first copy") { state.lock.withLock { state.copies[0]?.keys.count == 2 } }
    second.trigger(counts: counts)
    try waitFor("the second copy") { state.lock.withLock { state.copies[1]?.keys.count == 3 } }
    var copies = state.lock.withLock { state.copies.map { $0! } }
    try multiRequire(copies[0].keys == [0, 1] && copies[1].keys == [2, 3, 4], "built \(copies.map(\.keys))")
    try multiRequire(UInt64(copies[0].bytes) <= 2 * padded && UInt64(copies[1].bytes) <= 3 * padded, "sizes \(copies.map(\.bytes))")
    // The first drive's size grows by one record: a forced top-up adds the next record not held anywhere, no more.
    state.lock.withLock { state.budgets[0] = 3 * padded + padded / 2 }
    first.trigger(counts: counts, force: true)
    try waitFor("the first copy grown") { state.lock.withLock { state.copies[0]?.keys.count == 3 } }
    copies = state.lock.withLock { state.copies.map { $0! } }
    try multiRequire(
        copies[0].keys == [0, 1, 2] && UInt64(copies[0].bytes) == 3 * padded && UInt64(copies[1].bytes) == 3 * padded,
        "grown \(copies.map(\.keys)) \(copies.map(\.bytes))")
    let set = HotCopySet(copies)
    for key in 0..<5 {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { _ = try set.read(key, into: $0) }
        try multiRequire(bytes == multiRecord(key), "key \(key) from the two copies")
    }
}

/// GLM's and DeepSeek's expert reads from copies on two "drives" (`FamilyExperts.attach(copies:)`): each record comes
/// from whichever store holds it with the GGUF's bytes, counted as a copy read; a record failing its CRC32 in one store
/// comes from the other, and a store whose file goes away gives way to the GGUF, with the same bytes.
/// Two copy places of different speed (F10): a batch of records both hold is spread over them in proportion to their
/// rates (`HotCopySet.pick`, each record to the place expected to finish it first), and a place that slows down gets
/// less within 16 of its reads (each read moves its estimate an eighth of the way, at most fourfold).
func copyPlacesFollowSpeeds() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("copy places follow speeds: " + what) } }
    var lanes = [ReadLane(), ReadLane()]
    for _ in 0..<8 {
        lanes[0].served(seconds: 0.001, bytes: 1 << 20)
        lanes[1].served(seconds: 0.002, bytes: 1 << 20)
    }
    func batch(_ count: Int) -> [Int] {
        var queued = lanes
        var placed = [0, 0]
        for n in 0..<count {
            guard let lane = HotCopySet.pick(queued, placements: UInt64(n), bytes: 1 << 20) else { break }
            queued[lane].inFlight += 1
            placed[lane] += 1
        }
        return placed
    }
    let even = batch(30)
    try require(even == [20, 10], "a 2:1 pair took \(even)")
    // The fast place slows to a quarter of its speed: within 16 of its reads it takes the smaller share.
    var reads = 0
    while batch(30)[0] >= 15 && reads < 16 {
        lanes[0].served(seconds: 0.004, bytes: 1 << 20)
        reads += 1
    }
    try require(batch(30)[0] < 15, "a slowed place still took \(batch(30)) after \(reads) reads")
}

func familyCopiesOnTwoDrives() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family copies on two drives: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = try multiRoot("family-two-drives")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let budget = UInt64(3 * (212_992 + 114_688))
    let source = try FamilyExperts(weights: weights, budget: budget)
    // Keys are layer * 4 + expert: drive A holds 6 and 4, drive B 9 and 6.
    let a = try HotCopy.write(source: source, keys: [6, 4], folder: folder.appendingPathComponent("driveA/hot"), identity: "x")
    let b = try HotCopy.write(source: source, keys: [9, 6], folder: folder.appendingPathComponent("driveB/hot"), identity: "x")
    func pattern(_ layer: Int, _ expert: Int, _ part: Int) -> UInt8 { UInt8(layer * 50 + expert * 10 + part) }
    func fetch(_ experts: FamilyExperts, _ layer: Int, _ asked: [Int]) throws {
        let per = GGUFQuantize.bytes(layer == 1 ? 8 : 12, count: 256 * 256)
        for (expert, slot) in zip(asked, try experts.fetch(layer: layer, experts: asked, decode: true)) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
                try require(bytes.allSatisfy { $0 == pattern(layer, expert, part) }, "layer \(layer) expert \(expert) part \(part)")
            }
        }
    }
    let experts = try FamilyExperts(weights: weights, budget: budget)
    // Two places are planned at the sum of their measured rates, one alone at its own (F10).
    experts.seed(copy: 2e9, model: 1e9, places: [a.folder: 2e9, b.folder: 1e9])
    experts.attach(copies: [a, b])
    try require(experts.driveBandwidth.copy == 3e9, "two places planned at \(experts.driveBandwidth.copy)")
    experts.seed(copy: 2e9, model: 1e9, places: [a.folder: 2e9, b.folder: 1e9])
    try require(experts.driveBandwidth.copy == 3e9, "seeded again after the attach: \(experts.driveBandwidth.copy)")
    do {
        // Idle past `LiveDriveEstimate.graceSeconds` after a copy read, the pair's planned rate settles back to the sum,
        // not to one drive's seed.
        let settling = try FamilyExperts(weights: weights, budget: budget)
        settling.seed(copy: 2e9, model: 1e9, places: [a.folder: 2e9, b.folder: 1e9])
        settling.attach(copies: [a, b])
        try fetch(settling, 1, [2])
        try require(settling.counters.hotReads == 1, "no copy read to settle from: \(settling.counters)")
        Thread.sleep(forTimeInterval: 2)
        let planned = settling.splitPlannerNotes["split.copy_planned_rate_bytes_per_second"] ?? 0
        try require(planned > 2.5e9, "an idle pair settled to \(planned), not the sum of its places")
    }
    experts.attach(copies: [b])
    try require(experts.driveBandwidth.copy == 2e9, "one place planned at \(experts.driveBandwidth.copy)")
    experts.seed(copy: nil, model: nil, places: [:])
    let modelFiles = try weights.gguf.files.map { try weights.file($0) }
    let (modelBefore, copyBefore) = (modelFiles.reduce(0) { $0 + $1.readBytes }, a.file.readBytes + b.file.readBytes)
    // Attached afresh before each fetch: the split follows measured drive speeds, and attaching forgets the copies'.
    for (layer, expert) in [(1, 2), (1, 0), (2, 1), (2, 0)] {
        experts.attach(copies: [a, b])
        try fetch(experts, layer, [expert])
    }
    try require(experts.counters.hotReads == 3, "copy reads \(experts.counters)")
    // Bytes counted per source (R14) are the bytes each store and the GGUF were read: both drives count as the copy.
    let read = (copy: a.file.readBytes + b.file.readBytes - copyBefore, model: modelFiles.reduce(0) { $0 + $1.readBytes } - modelBefore)
    try require(
        UInt64(experts.counters.copyBytes) == read.copy && UInt64(experts.counters.modelBytes) == read.model && read.model > 0,
        "counted \(experts.counters.copyBytes) and \(experts.counters.modelBytes) bytes against \(read) read")
    // Key 6 fails its CRC32 on drive A (chosen first): drive B serves it.
    try corruptFamily(a, key: 6)
    let reopened = [
        HotCopy.open(folder: a.folder, identity: "x", source: source)!, HotCopy.open(folder: b.folder, identity: "x", source: source)!,
    ]
    let checked = try FamilyExperts(weights: weights, budget: budget)
    checked.attach(copies: reopened)
    try fetch(checked, 1, [2])
    try require(checked.counters.hotReads == 1 && reopened[0].rejectedRecords == 1, "a bad record on one drive \(checked.counters)")
    // Drive B's file goes away: key 9 (only there) comes from the GGUF, exact.
    try FileHandle(forWritingTo: reopened[1].payload).truncate(atOffset: 0)
    checked.attach(copies: reopened)
    try fetch(checked, 2, [1])
    try require(checked.counters.hotReads == 1 && reopened[1].disabled, "a drive gone \(checked.counters)")
}
private func corruptFamily(_ copy: HotCopy, key: Int) throws {
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.seek(toOffset: UInt64(copy.offsets[key]!) + 100)
    try handle.write(contentsOf: Data([0xEE]))
    try handle.close()
}

/// A real `PackStore` for split reads: the checkpoint's spans, a hot copy of experts 0 to 7 on the model's drive and a
/// further copy of them on another drive, read by a decode loader with split reads (`SplitPartReader(store:)`).
private func splitStore(_ root: URL) throws -> (PackStore, HotCopy, HotCopy, SourceExpertLoader, IOScheduler) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let model = try multiSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model)
    let identity = PackStore.hotCopyIdentity(model)
    let drive = root.appendingPathComponent("driveB/m")
    let further = try HotCopy.write(
        source: source, keys: Array(0..<8), folder: drive.appendingPathComponent(PackStore.furtherCopyFolder), identity: identity)
    let store = try PackStore(model: model, copies: [drive], log: { _ in })
    // Every folder of a check is on one disk: count each as its own drive.
    store.disk = { _ in nil }
    let hot = try HotCopy.write(source: source, keys: Array(0..<8), folder: root.appendingPathComponent("hot"), identity: identity)
    store.attachHotCopy(hot)
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(8 * PackStore.expertBytes), slotsPerSlab: 8)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 4, reservedDemandConcurrent: 2, maximumOutstanding: 16, reservedDemandOutstanding: 8,
        maximumOutstandingBytes: 64_000_000, reservedDemandBytes: 32_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: try ExpertReplicaArena(ledger: ledger, device: device),
        // No estimator here, so a hot key would be pinned to the model's drive (`request`); production has one.
        packedAvailable: { true }, hotInternal: { _ in false }
    ) { key, source, destination in
        if source == .internalSSD {
            try store.readInternalExpert(key, into: destination)
        } else {
            try store.readExpert(layer: key.layer, expert: key.expert, into: destination, source: .packed)
        }
    }
    loader.enableSplitReads(SourceExpertLoader.SplitPartReader(store: store))
    loader.splitsReads = true
    try loader.setReplicasEnabled(true)
    loader.setDecoding(true)
    return (store, hot, further, loader, scheduler)
}
private let splitStoreCalibration = ExpertSourceCalibration(
    queueDepths: [1], internalBytesPerSecond: [2.5e9], externalBytesPerSecond: [2.4e9], label: "split-store", replicaDelaySeconds: 1)
private func splitStoreRead(_ loader: SourceExpertLoader, _ scheduler: IOScheduler, _ expert: Int) throws -> [UInt8] {
    let request = try loader.request(
        try ExpertKey(layer: 0, expert: expert), generation: scheduler.currentGeneration, policy: .earliestCompletion,
        calibration: splitStoreCalibration)
    let loaded = try loader.joinAndPin(request)
    defer { try? loaded.retireAfterGPUCompletion() }
    return Array(loaded.buffer.bytes)
}
private func corruptAt(_ copy: HotCopy, key: Int, byte: Int) throws {
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.seek(toOffset: UInt64(copy.offsets[key]!) + UInt64(byte))
    try handle.write(contentsOf: Data([0x5A ^ multiRecord(key)[byte]]))
    try handle.close()
}

/// `PackStore.readExpertPart` and `verifySplit` through a split demand read over three sources (the checkpoint's spans,
/// the hot copy, the pack queue's copy): the parts assemble the exact record and are checked against both copies' checksums; a wrong byte in either part fails
/// the split, and the record is read again whole, exact.
func splitReadPackStore() throws {
    let root = try multiRoot("split-store")
    defer { try? FileManager.default.removeItem(at: root) }
    let (_, hot, further, loader, scheduler) = try splitStore(root)
    defer { scheduler.shutdown() }
    try multiRequire(try splitStoreRead(loader, scheduler, 1) == multiRecord(1), "split bytes")
    let report = loader.splitReads.report
    let parts = report["part_bytes"] as? [String: Int] ?? [:]
    try multiRequire(
        report["split_reads"] as? Int == 1 && parts["model_drive", default: 0] > 0 && parts["copy0", default: 0] > 0
            && parts["pack", default: 0] > 0, "not split three ways: \(report)")
    // Wrong bytes all along one copy's record: whichever range it serves holds some.
    for byte in stride(from: 11, to: PackStore.expertBytes, by: 65_536) { try corruptAt(further, key: 2, byte: byte) }
    try multiRequire(try splitStoreRead(loader, scheduler, 2) == multiRecord(2), "a wrong byte in the pack's part")
    for byte in stride(from: 11, to: PackStore.expertBytes, by: 65_536) { try corruptAt(hot, key: 3, byte: byte) }
    try multiRequire(try splitStoreRead(loader, scheduler, 3) == multiRecord(3), "a wrong byte in the hot copy's part")
    try multiRequire(
        // The split's check drops nothing; the whole re-read (from the hot copy) drops only the copy that is wrong.
        loader.splitReads.report["failed_splits"] as? Int == 2 && further.rejectedRecords == 0 && hot.rejectedRecords == 1,
        "wrong parts not caught: \(loader.splitReads.report)")
}

/// A key an in-place swap takes out of both copies between the split's plan and its part reads: the hot copy's part
/// comes from the checkpoint's spans and the pack copy's part fails over to the model's drive; the record is exact.
func splitReadSwappedKey() throws {
    let root = try multiRoot("split-swap")
    defer { try? FileManager.default.removeItem(at: root) }
    let (store, hot, further, loader, scheduler) = try splitStore(root)
    defer { scheduler.shutdown() }
    loader.enableSplitReads(
        .init(
            sources: { key in
                let index = key.layer * ModelContract.experts + key.expert
                let sources = store.splitSources(key)
                for copy in [hot, further] { if let slot = copy.slots.firstIndex(where: { $0.key == index }) { copy.vacate(slot) } }
                return sources
            }, wearing: { _, _ in false },
            read: { key, source, offset, into, checks in
                try store.readExpertPart(key, source: source, at: offset, into: into, checks: checks)
            },
            verify: { key, record, checks in try store.verifySplit(key, record: record, checks: checks) }))
    try multiRequire(try splitStoreRead(loader, scheduler, 4) == multiRecord(4), "a swapped-out key's split bytes")
    try multiRequire(
        !hot.holds(4) && !further.holds(4) && loader.splitReads.report["split_reads"] as? Int == 1, "\(loader.splitReads.report)")
}

/// Qwen's loader as serving builds it on one drive with hot copies (no pack, no replicas, no online estimator, the
/// `packedPreferred` policy, `SplitPartReader(store:)`): a record two copy places and the model's spans hold splits
/// three ways, one a single place holds two ways (`HotCopySet.sources`), bytes exact; a place detached while its part
/// reads fails that part over to another source.
func splitReadQwenPlaces() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = try multiRoot("split-places")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try multiSpanModel(root.appendingPathComponent("model"))
    let store = try PackStore(model: model)
    store.disk = { _ in nil }
    let identity = PackStore.hotCopyIdentity(model)
    let a = try HotCopy.write(source: store, keys: [0, 1, 2, 3], folder: root.appendingPathComponent("placeA"), identity: identity)
    let b = try HotCopy.write(source: store, keys: [0, 1, 4], folder: root.appendingPathComponent("placeB"), identity: identity)
    store.attachHotCopies([a, b])
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(8 * PackStore.expertBytes), slotsPerSlab: 8)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 4, reservedDemandConcurrent: 4, maximumOutstanding: 16, reservedDemandOutstanding: 16,
        maximumOutstandingBytes: 64_000_000, reservedDemandBytes: 64_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    defer { scheduler.shutdown() }
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: nil, packedAvailable: { store.packedAvailable },
        hotInternal: { store.hasHotExpert($0) }
    ) { key, source, destination in
        if source == .internalSSD {
            try store.readInternalExpert(key, into: destination)
        } else {
            try store.readExpert(layer: key.layer, expert: key.expert, into: destination, source: .packed)
        }
    }
    // No measured speeds: the plan follows the calibration, not how fast this Mac read the last record.
    loader.enableSplitReads(
        .init(
            sources: { store.splitSources($0) }, wearing: { _, _ in false },
            read: { key, source, offset, into, checks in
                try store.readExpertPart(key, source: source, at: offset, into: into, checks: checks)
            },
            verify: { key, record, checks in try store.verifySplit(key, record: record, checks: checks) }))
    loader.splitsReads = true
    loader.setDecoding(true)
    let calibration = ExpertSourceCalibration(
        queueDepths: [1], internalBytesPerSecond: [2.5e9], externalBytesPerSecond: [2.5e9], label: "one-drive", replicaDelaySeconds: 1)
    func read(_ expert: Int) throws -> [UInt8] {
        let request = try loader.request(
            try ExpertKey(layer: 0, expert: expert), generation: scheduler.currentGeneration, policy: .packedPreferred,
            calibration: calibration)
        let loaded = try loader.joinAndPin(request)
        defer { try? loaded.retireAfterGPUCompletion() }
        return Array(loaded.buffer.bytes)
    }
    func sources() -> Int { (loader.splitReads.report["part_bytes"] as? [String: Int] ?? [:]).filter { $0.value > 0 }.count }
    try multiRequire(try read(0) == multiRecord(0) && sources() == 3, "two places and the model: \(loader.splitReads.report)")
    let before = loader.splitReads.report["part_bytes"] as? [String: Int] ?? [:]
    try multiRequire(try read(2) == multiRecord(2), "one place and the model")
    let after = loader.splitReads.report["part_bytes"] as? [String: Int] ?? [:]
    try multiRequire(
        loader.splitReads.splitCount == 2 && after["copy1"] == before["copy1"] && after["copy0", default: 0] > before["copy0", default: 0],
        "one place: \(before) then \(after)")
    // Place A gives key 1 away right after the plan (as a swap or a detach would): its part fails over, exact.
    let slot = a.slots.firstIndex { $0.key == 1 }!
    loader.enableSplitReads(
        .init(
            sources: { key in
                let sources = store.splitSources(key)
                a.vacate(slot)
                return sources
            }, wearing: { _, _ in false },
            read: { key, source, offset, into, checks in
                try store.readExpertPart(key, source: source, at: offset, into: into, checks: checks)
            },
            verify: { key, record, checks in try store.verifySplit(key, record: record, checks: checks) }))
    try multiRequire(try read(1) == multiRecord(1), "a place gone mid-read")
    try multiRequire(
        loader.splitReads.splitCount == 3 && (loader.splitReads.report["failed_part_rereads"] as? Int ?? 0) >= 1,
        "the gone place's part did not fail over: \(loader.splitReads.report)")
}

/// Qwen, every demand read planned (`SourceExpertLoader`): eight concurrent demand reads of records the model's spans, a
/// hot copy and a pack copy hold publish the exact records and leave no planned bytes behind.
func splitReadQwenConcurrent() throws {
    let root = try multiRoot("split-concurrent")
    defer { try? FileManager.default.removeItem(at: root) }
    let (_, _, _, loader, scheduler) = try splitStore(root)
    defer { scheduler.shutdown() }
    final class Results: @unchecked Sendable {
        let lock = NSLock()
        var exact = 0
        var errors: [String] = []
    }
    let results = Results()
    DispatchQueue.concurrentPerform(iterations: 8) { expert in
        do {
            let exact = try splitStoreRead(loader, scheduler, expert) == multiRecord(expert)
            results.lock.withLock { results.exact += exact ? 1 : 0 }
        } catch { results.lock.withLock { results.errors.append("\(error)") } }
    }
    try multiRequire(results.errors.isEmpty && results.exact == 8, "concurrent reads: \(results.exact) exact, \(results.errors)")
    try multiRequire(loader.splitReads.splitCount >= 1, "no concurrent read was split: \(loader.splitReads.report)")
    let end = Date(timeIntervalSinceNow: 2)
    while loader.plannedBytes.total != 0 && Date() < end { Thread.sleep(forTimeInterval: 0.001) }
    try multiRequire(loader.plannedBytes.total == 0, "planned bytes left: \(loader.plannedBytes.total)")
}
