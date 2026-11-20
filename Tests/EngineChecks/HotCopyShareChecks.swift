import Darwin
import Foundation
import OrigamiEngine

/// Hot copies of several models sharing one drive (`IdleHotCopy`, `DiskBudget.take`): the loaded model's copy grows at
/// the cost of idle models' copies, each down to its floor, used longest ago first, and grows back when its model is
/// loaded again. Free space is injected (`DiskBudget.freeForChecks`): the drive is a fixed capacity less every copy's
/// record bytes.
func hotCopyShareChecks() -> [(String, () throws -> Void)] {
    [
        ("hot_copy_share_loaded_grows_idle_give_back", hotCopyShareLoadedGrows),
        ("hot_copy_share_keep_when_idle_and_open_copies", hotCopyShareKeepAndOpen),
        ("hot_copy_share_crash_at_each_step", hotCopyShareCrashSteps),
        ("hot_copy_keeper_backlog_counter", hotCopyKeeperBacklog),
    ]
}

private func require(_ ok: Bool, _ what: @autoclosure () -> String) throws {
    if !ok { throw EngineError("hot copy share: " + what()) }
}
/// One model's records: 64 KiB each, every byte depending on the model (`salt`), the key and its position.
private final class Model: HotCopySource, @unchecked Sendable {
    static let bytes = 65536
    let salt: Int, count: Int
    init(salt: Int, count: Int) {
        self.salt = salt
        self.count = count
    }
    func byte(_ key: Int, _ at: Int) -> UInt8 { UInt8(truncatingIfNeeded: key &* 31 &+ at &* 7 &+ salt &* 101 &+ 1) }
    func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<count).contains(key) ? Self.bytes : nil }
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
        for at in 0..<Self.bytes { into.storeBytes(of: byte(key, at), toByteOffset: at, as: UInt8.self) }
    }
    /// Every record the copy in `folder` maps reads back as this model's bytes, in a fresh open from disk too.
    func check(_ folder: URL, identity: String, _ copy: HotCopy? = nil) throws -> Bool {
        guard let fresh = HotCopy.open(folder: folder, identity: identity, source: self) else { return false }
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: Self.bytes, alignment: 16384)
        defer { buffer.deallocate() }
        for one in [fresh] + (copy.map { [$0] } ?? []) {
            for key in one.keys {
                try one.read(key, into: buffer)
                guard buffer.indices.allSatisfy({ buffer[$0] == byte(key, $0) }) else { return false }
            }
        }
        return true
    }
}
private let record = UInt64(Model.bytes)
private func uses(_ keys: Range<Int>, _ value: (Int) -> Int) -> [Int: Int] {
    Dictionary(uniqueKeysWithValues: keys.map { ($0, value($0)) })
}
/// Record keys an index maps now, read from disk.
private func held(_ folder: URL, _ identity: String) -> Set<Int> {
    Set(HotCopy.open(folder: folder, identity: identity, attaching: false)?.keys ?? [])
}
private final class Fixture: @unchecked Sendable {
    let root: URL
    let folders: [String: URL]
    let room: UInt64
    /// The copies' records when the drive had `room` above its reserve (`settle`).
    var settled: UInt64 = 0
    init(root: URL, folders: [String: URL], room: UInt64) {
        self.root = root
        self.folders = folders
        self.room = room
    }
    func identity(_ name: String) -> String { "fixture-" + name }
    var used: UInt64 { folders.keys.reduce(UInt64(0)) { $0 + UInt64(held(folders[$1]!, identity($1)).count) * record } }
    func settle() { settled = used }
    /// The drive: the reserve and `room`, less what the copies hold beyond what they held when settled.
    func free() -> UInt64 {
        let total = HotCopyKeeper.keepFree + room + settled
        let now = used
        return total > now ? total - now : 0
    }
}
private func withFixture(room: UInt64, _ body: (Fixture) throws -> Void) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-share-\(UUID().uuidString)")
    let fixture = Fixture(
        root: root, folders: Dictionary(uniqueKeysWithValues: ["A", "B", "C"].map { ($0, root.appendingPathComponent($0)) }), room: room)
    defer {
        DiskBudget.shared.freeForChecks = nil
        HotCopy.stopForChecks = nil
        try? FileManager.default.removeItem(at: root)
    }
    DiskBudget.shared.freeForChecks = { _ in fixture.free() }
    try body(fixture)
}
private final class Lines: @unchecked Sendable {
    let lock = NSLock()
    var all: [String] = []
    func add(_ line: String) { lock.withLock { all.append(line) } }
}
/// A keeper for the loaded model `name` wanting `wanted` records, sized as both engines' places do, with what idle
/// copies on the drive may give counted as room.
private func keeper(
    _ fixture: Fixture, _ name: String, _ model: Model, copy: HotCopy?, counts: [Int: Int], wanted: UInt64, lines: Lines
) -> HotCopyKeeper {
    let folder = fixture.folders[name]!
    let keeper = HotCopyKeeper(
        source: model, identity: fixture.identity(name), copy: copy, counts: (counts, [:]),
        place: { heldHere in
            let free = fixture.free()
            let spare = DiskBudget.shared.spare(on: folder, for: name)
            let budget =
                HotCopyRefreshPlan.capacity(wanted: wanted, held: heldHere(folder), free: free + spare, reserve: HotCopyKeeper.keepFree)
                ?? 0
            return .folder(folder, drive: "fixture", budget: budget, free: free)
        }, attach: { _ in true }, idle: { true }, event: lines.add)
    keeper.model = name
    return keeper
}
private func wait(_ what: String, _ done: () -> Bool) throws {
    let end = Date().addingTimeInterval(20)
    while !done() {
        guard Date() < end else { throw EngineError("hot copy share: timed out: \(what)") }
        Thread.sleep(forTimeInterval: 0.02)
    }
}

/// A (loaded) grows to its size; B (used two days ago) and C (one day ago), idle on its drive, give their least-used
/// records, B first, each down to its floor (the most-used records covering half its uses) and never below; every
/// record of the three reads back right, also opened again from disk. Then B is loaded: its emptied slots fill again
/// at the cost of A, now idle (C, at its floor, gives nothing), with no hourly swap allowance spent.
private func hotCopyShareLoadedGrows() throws {
    try withFixture(room: 5 * record) { fixture in
        let a = Model(salt: 1, count: 60)
        let b = Model(salt: 2, count: 30)
        let c = Model(salt: 3, count: 30)
        let usesA = uses(0..<57) { 100 - $0 }
        let usesB = uses(0..<30) { $0 + 1 }
        let usesC = usesB
        var built: HotCopy? = try HotCopy.write(
            source: a, keys: Array(0..<10), folder: fixture.folders["A"]!, identity: fixture.identity("A"))
        _ = try HotCopy.write(source: b, keys: Array(0..<30), folder: fixture.folders["B"]!, identity: fixture.identity("B"))
        _ = try HotCopy.write(source: c, keys: Array(0..<30), folder: fixture.folders["C"]!, identity: fixture.identity("C"))
        fixture.settle()
        // B's floor: its top 9 records (30 + … + 22 = 234 of 465 uses); C's the same.
        let floorB = IdleHotCopy.floor(slots: (0..<30).map { ($0, record) }, counts: (usesB, [:]), redundant: [])
        try require(floorB == 9 * record, "B's floor is \(floorB / record) records, not 9")
        let day: TimeInterval = 86400
        var idleB: IdleHotCopy? = IdleHotCopy(
            model: "B", folder: fixture.folders["B"]!, identity: fixture.identity("B"), cap: nil, keep: false,
            lastUsed: Date().addingTimeInterval(-2 * day), counts: (usesB, [:]))
        let idleC = IdleHotCopy(
            model: "C", folder: fixture.folders["C"]!, identity: fixture.identity("C"), cap: nil, keep: false,
            lastUsed: Date().addingTimeInterval(-day), counts: (usesC, [:]))
        idleB!.register()
        idleC.register()
        try require(DiskBudget.shared.spare(on: fixture.root, for: "A") == 42 * record, "the idle copies' spare is not 2 x 21 records")
        // A wants 57 records: 47 more, 5 from free space, 42 from the idle copies.
        let lines = Lines()
        var keeperA: HotCopyKeeper! = keeper(fixture, "A", a, copy: built, counts: usesA, wanted: 57 * record, lines: lines)
        keeperA.trigger(force: true)
        try wait("A grows to 57 records") { held(fixture.folders["A"]!, fixture.identity("A")).count == 57 && !keeperA.building }
        try require(
            held(fixture.folders["B"]!, fixture.identity("B")) == Set(21..<30)
                && held(fixture.folders["C"]!, fixture.identity("C")) == Set(21..<30),
            "B and C did not give their least-used records down to their floors: B \(held(fixture.folders["B"]!, fixture.identity("B")).sorted()), C \(held(fixture.folders["C"]!, fixture.identity("C")).sorted())"
        )
        let gaveLines = lines.lock.withLock { lines.all.filter { $0.contains("gave") && $0.contains("to A's copy") } }
        try require(gaveLines.count == 2 && gaveLines[0].hasPrefix("Hot copy of B"), "the gives were not logged B first: \(gaveLines)")
        try require(fixture.free() >= HotCopyKeeper.keepFree, "the drive went under its reserve")
        try require(
            try a.check(fixture.folders["A"]!, identity: fixture.identity("A"), keeperA.current)
                && b.check(fixture.folders["B"]!, identity: fixture.identity("B"))
                && c.check(fixture.folders["C"]!, identity: fixture.identity("C")),
            "a record kept read other bytes")
        let took = keeperA.report["took"] as? [[String: Any]] ?? []
        try require(
            took.map { $0["from"] as? String ?? "" }.sorted() == ["B", "C"]
                && took.reduce(UInt64(0)) { $0 + ($1["bytes"] as? UInt64 ?? 0) } == 42 * record,
            "A's row does not show what it took: \(took)")
        let volume = DiskBudget.shared.report.first { $0["volume"] as? String == DiskBudget.volume(fixture.root) }
        let rows = volume?["holders"] as? [[String: Any]] ?? []
        try require(
            rows.contains { $0["model"] as? String == "B" && $0["floor_bytes"] as? UInt64 == 9 * record && $0["loaded"] as? Bool == false }
                && rows.contains { $0["model"] as? String == "A" && $0["loaded"] as? Bool == true },
            "the disk budget does not list every model's copy: \(rows)")
        try require(
            DiskBudget.growthLine(DiskBudget.shared.report)?.contains("A's hot copy grew by") == true, "no status line for the growth")
        // B is loaded (A's server ends, closing its copy; B's copy leaves the idle list, A's joins it, used just now).
        keeperA.stop(waiting: 5)
        keeperA = nil
        built = nil
        idleB = nil
        let idleA = IdleHotCopy(
            model: "A", folder: fixture.folders["A"]!, identity: fixture.identity("A"), cap: nil, keep: false, lastUsed: Date(),
            counts: (usesA, [:]))
        idleA.register()
        let floorA = idleA.floor
        HotCopyKeeper.swapBytesPerHour = 0
        defer { HotCopyKeeper.swapBytesPerHour = HotCopyKeeper.topUpBytes }
        let copyB = HotCopy.open(folder: fixture.folders["B"]!, identity: fixture.identity("B"), source: b)
        let keeperB = keeper(fixture, "B", b, copy: copyB, counts: usesB, wanted: 30 * record, lines: lines)
        defer { keeperB.stop(waiting: 5) }
        keeperB.trigger()
        try wait("B grows back to 30 records") { held(fixture.folders["B"]!, fixture.identity("B")).count == 30 }
        let keptA = held(fixture.folders["A"]!, fixture.identity("A"))
        try require(
            keptA == Set(0..<36) && UInt64(keptA.count) * record >= floorA
                && held(fixture.folders["C"]!, fixture.identity("C")) == Set(21..<30),
            "A did not give its 21 least-used records (C at its floor gives none): A \(keptA.count), C \(held(fixture.folders["C"]!, fixture.identity("C")).count)"
        )
        try require(
            try a.check(fixture.folders["A"]!, identity: fixture.identity("A"))
                && b.check(fixture.folders["B"]!, identity: fixture.identity("B"), keeperB.current)
                && c.check(fixture.folders["C"]!, identity: fixture.identity("C")),
            "a record read other bytes after B grew back")
    }
}

/// An idle copy whose owner set a size and ticked "keep this size when idle" gives nothing, nor does one any process
/// has open (this one through its `.attached.lock`, another through its payload, as a build without that lock would):
/// the next idle copy gives instead. Without a size the box does not hold.
private func hotCopyShareKeepAndOpen() throws {
    try withFixture(room: 0) { fixture in
        let b = Model(salt: 2, count: 30)
        let c = Model(salt: 3, count: 30)
        let counts = uses(0..<30) { $0 + 1 }
        _ = try HotCopy.write(source: b, keys: Array(0..<30), folder: fixture.folders["B"]!, identity: fixture.identity("B"))
        _ = try HotCopy.write(source: c, keys: Array(0..<30), folder: fixture.folders["C"]!, identity: fixture.identity("C"))
        fixture.settle()
        func idle(_ name: String, cap: UInt64?, keep: Bool, daysAgo: Double) -> IdleHotCopy {
            IdleHotCopy(
                model: name, folder: fixture.folders[name]!, identity: fixture.identity(name), cap: cap, keep: keep,
                lastUsed: Date().addingTimeInterval(-daysAgo * 86400), counts: (counts, [:]))
        }
        // Kept with a size: nothing from B, all from C.
        do {
            let kept = idle("B", cap: 30 * record, keep: true, daysAgo: 2)
            let other = idle("C", cap: nil, keep: false, daysAgo: 1)
            kept.register()
            other.register()
            try require(kept.spare == 0 && kept.giveBack(record, for: "A") == 0, "a kept copy offered or gave space")
            let given = DiskBudget.shared.take(2 * record, in: fixture.root, for: "A")
            try require(
                given == ["C": 2 * record] && held(fixture.folders["B"]!, fixture.identity("B")).count == 30, "the kept copy gave: \(given)"
            )
        }
        // Ticked without a size: the box does not hold (it keeps the owner's size only).
        try require(idle("B", cap: nil, keep: true, daysAgo: 2).spare == 21 * record, "keep without a size held the copy")
        // Open in this process (attached, `.attached.lock` shared): nothing.
        let source = idle("B", cap: nil, keep: false, daysAgo: 2)
        do {
            let reading = HotCopy.open(folder: fixture.folders["B"]!, identity: fixture.identity("B"), source: b)
            try require(reading != nil && source.spare == 0 && source.giveBack(record, for: "A") == 0, "a copy open here gave space")
            try require(try b.check(fixture.folders["B"]!, identity: fixture.identity("B"), reading), "the open copy read other bytes")
        }
        // Open in another process without the lock (an older build): nothing.
        let payload = try FileManager.default.contentsOfDirectory(atPath: fixture.folders["B"]!.path).first { $0.hasSuffix(".bin") }!
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sleep")
        process.arguments = ["30"]
        process.standardInput = FileHandle(forReadingAtPath: fixture.folders["B"]!.appendingPathComponent(payload).path)
        try process.run()
        defer { process.terminate() }
        try require(source.spare == 0 && source.giveBack(record, for: "A") == 0, "a copy another process has open gave space")
        process.terminate()
        process.waitUntilExit()
        try require(source.giveBack(record, for: "A") == record, "a closed idle copy did not give")
    }
}

/// A give-back stopped at each step (before the index, after the index, while freeing) leaves both copies valid: every
/// record each maps reads its model's bytes (opened again from disk), the slots of a stopped one are either all mapped
/// or all empty, and the next give-back finishes the job (an emptied slot still holding blocks goes first).
private func hotCopyShareCrashSteps() throws {
    try withFixture(room: 0) { fixture in
        let a = Model(salt: 1, count: 30)
        let b = Model(salt: 2, count: 30)
        let counts = uses(0..<30) { $0 + 1 }
        _ = try HotCopy.write(source: a, keys: Array(0..<10), folder: fixture.folders["A"]!, identity: fixture.identity("A"))
        _ = try HotCopy.write(source: b, keys: Array(0..<30), folder: fixture.folders["B"]!, identity: fixture.identity("B"))
        fixture.settle()
        let idle = IdleHotCopy(
            model: "B", folder: fixture.folders["B"]!, identity: fixture.identity("B"), cap: nil, keep: false, lastUsed: nil,
            counts: (counts, [:]))
        // Stopped before the index: nothing changed. While freeing: three records out of the index, the first slot freed,
        // two still holding blocks. Right after the index: those two go first, with one more record, out of the index.
        let steps: [(String, Set<Int>)] = [("vacated", []), ("freeing", [0, 1, 2]), ("published", [3])]
        for (step, gone) in steps {
            HotCopy.stopForChecks = { $0 == step }
            let before = held(fixture.folders["B"]!, fixture.identity("B"))
            try require(idle.giveBack(3 * record, for: "A") == 0, "a give-back stopped at \(step) counted bytes")
            let after = held(fixture.folders["B"]!, fixture.identity("B"))
            try require(after == before.subtracting(gone), "stopped at \(step): \(before.count) -> \(after.count) records")
            try require(
                try b.check(fixture.folders["B"]!, identity: fixture.identity("B"))
                    && a.check(fixture.folders["A"]!, identity: fixture.identity("A")),
                "stopped at \(step), a copy read other bytes")
        }
        HotCopy.stopForChecks = nil
        let expected = held(fixture.folders["B"]!, fixture.identity("B"))
        let stopped = HotCopy.open(folder: fixture.folders["B"]!, identity: fixture.identity("B"), attaching: false)!
        try require(stopped.emptyHeld.count == 3, "the emptied slots still holding blocks are \(stopped.emptyHeld.count), not 3")
        // The next give-back frees those first (they serve nothing and count toward no floor), then a record.
        try require(idle.giveBack(4 * record, for: "A") == record, "the next give-back did not free the stopped slots, then one record")
        let reopened = HotCopy.open(folder: fixture.folders["B"]!, identity: fixture.identity("B"), attaching: false)!
        try require(
            reopened.emptyHeld.isEmpty && held(fixture.folders["B"]!, fixture.identity("B")) == expected.subtracting([4]),
            "after the stops, the copy is not whole: \(reopened.emptyHeld.count) slots hold blocks")
        try require(try b.check(fixture.folders["B"]!, identity: fixture.identity("B")), "after the stops, B read other bytes")
    }
}

/// The keeper's backlog (`backlog_records`, `_bytes`, `_uses`): the swaps the ranking wants now beyond the hourly
/// allowance, in its row and once per hour in the log while not zero. Empty slots refill without spending it.
private func hotCopyKeeperBacklog() throws {
    try withFixture(room: 100 * record) { fixture in
        let a = Model(salt: 1, count: 40)
        let folder = fixture.folders["A"]!
        let built = try HotCopy.write(source: a, keys: Array(0..<10), folder: folder, identity: fixture.identity("A"))
        fixture.settle()
        let slots = built.slots
        // Three slots empty, then keys 10..<20 used far more than 3..<10: 3 fills and 7 swaps wanted.
        _ = try built.giveBack([0, 1, 2])
        let counts = uses(0..<20) { $0 < 10 ? 1 : 100 }
        HotCopyKeeper.swapBytesPerHour = 2 * record
        defer { HotCopyKeeper.swapBytesPerHour = HotCopyKeeper.topUpBytes }
        let lines = Lines()
        let keeper = keeper(fixture, "A", a, copy: built, counts: counts, wanted: UInt64(slots.count) * record, lines: lines)
        defer { keeper.stop(waiting: 5) }
        keeper.trigger()
        try wait("the fills and two swaps") { built.keys.filter { $0 >= 10 }.count == 5 && !keeper.building }
        let report = keeper.report
        try require(
            report["backlog_records"] as? Int == 5 && report["backlog_bytes"] as? UInt64 == 5 * record
                && report["backlog_uses"] as? Int == 500,
            "the backlog is not 5 records, 500 uses: \(report["backlog_records"] ?? "nil"), \(report["backlog_uses"] ?? "nil")")
        keeper.trigger()
        Thread.sleep(forTimeInterval: 0.3)
        let logged = lines.lock.withLock { lines.all.filter { $0.hasPrefix("Hot copy backlog") } }
        try require(logged.count == 1 && logged[0].contains("5 swaps"), "the backlog was not logged once: \(logged)")
        try require(try a.check(folder, identity: fixture.identity("A"), built), "a record read other bytes")
    }
}
