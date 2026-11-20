import Foundation
import OrigamiEngine

/// The keeper never gives up a working copy for a free-space dip: (a) free space under the reserve while the ranking
/// moves wholly: the same copy, attached once, follows usage by swaps in place (no room for a second copy needed); (b) a lower size from the owner
/// is written beside the copy in use, which serves until the smaller one is attached (and stays when there is no room
/// for both); (c) a failed build is planned again after a reply once its back-off has passed; (d) a detached copy is
/// reported as such (state, no records, no bytes), with a log line for each detach, re-attach and failure. The place
/// sizes the copy as both engines' places do (`HotCopyRefreshPlan.capacity`).
func hotCopyKeeperKeepsWorkingCopy() throws {
    func require(_ ok: Bool, _ what: @autoclosure () -> String) throws {
        if !ok { throw EngineError("hot copy keeper keeps its copy: " + what()) }
    }
    final class Source: HotCopySource, @unchecked Sendable {
        let lock = NSLock()
        var fail = false, reads = 0
        func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<20).contains(key) ? 16384 : nil }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
            try lock.withLock {
                if fail { throw EngineError("No space left on device") }
                reads += 1
            }
            memset(into, Int32(key + 1), 16384)
        }
    }
    final class Engine: @unchecked Sendable {
        let lock = NSLock()
        var busy = false, gone = false, wanted: UInt64 = 0, free: UInt64 = 0
        var attached: [HotCopy?] = [], events: [String] = []
    }
    let record: UInt64 = 16384
    let keep = HotCopyKeeper.keepFree
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-keep-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let source = Source()
    let engine = Engine()
    func keeper(_ folder: URL) -> HotCopyKeeper {
        HotCopyKeeper(
            source: source, identity: "fixture", copy: nil, counts: ([:], [:]),
            place: { heldHere in
                let (gone, wanted, free) = engine.lock.withLock { (engine.gone, engine.wanted, engine.free) }
                if gone { return .none(state: "not_connected", detail: "Fixture is not connected.") }
                guard let budget = HotCopyRefreshPlan.capacity(wanted: wanted, held: heldHere(folder), free: free, reserve: keep) else {
                    return .none(state: "not_possible", detail: "No room.")
                }
                return .folder(folder, drive: "fixture", budget: budget, free: free)
            },
            attach: { copy in
                engine.lock.withLock {
                    guard !engine.busy else { return false }
                    engine.attached.append(copy)
                    return true
                }
            },
            idle: { engine.lock.withLock { !engine.busy } },
            withIdle: { work in
                try engine.lock.withLock {
                    guard !engine.busy else { return false }
                    try work()
                    return true
                }
            },
            event: { line in engine.lock.withLock { engine.events.append(line) } })
    }
    func set(wanted: UInt64? = nil, free: UInt64? = nil, busy: Bool? = nil, gone: Bool? = nil) {
        engine.lock.withLock {
            if let wanted { engine.wanted = wanted }
            if let free { engine.free = free }
            if let busy { engine.busy = busy }
            if let gone { engine.gone = gone }
        }
    }
    func waitFor(_ keeper: HotCopyKeeper, _ what: String, _ done: () -> Bool) throws {
        let end = Date().addingTimeInterval(10)
        while !done() {
            guard Date() < end else { throw EngineError("hot copy keeper keeps its copy: \(what) (\(keeper.report))") }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }
    /// Uses ranking `first` highest, then the rest.
    func counts(_ first: [Int]) -> (uses: [Int: Int], misses: [Int: Int]) {
        (Dictionary(uniqueKeysWithValues: (0..<20).map { ($0, first.contains($0) ? 100 - first.firstIndex(of: $0)! : 1) }), [:])
    }
    func state(_ keeper: HotCopyKeeper) -> String { keeper.report["state"] as? String ?? "" }
    func reads(_ copy: HotCopy, _ key: Int) throws -> Bool {
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 16384, alignment: 16384)
        defer { buffer.deallocate() }
        try copy.read(key, into: buffer)
        return buffer.allSatisfy { $0 == UInt8(key + 1) }
    }

    let folder = root.appendingPathComponent("hot")
    let hot = keeper(folder)
    defer { hot.stop(waiting: 3) }
    set(wanted: 10 * record, free: keep + 1_000_000_000)
    hot.trigger(counts: counts(Array(0..<10)))
    try waitFor(hot, "first build") { state(hot) == "ready" && hot.records == 10 }
    guard let working = hot.current else { throw EngineError("hot copy keeper keeps its copy: none attached") }

    // (a) Free space under the reserve, replies keep arriving and the ranking moves wholly: the same copy follows usage.
    set(free: keep - 5 * record)
    for _ in 0..<3 {
        hot.trigger(counts: counts(Array(10..<20)))
        try waitFor(hot, "a reply planned") { (hot.report["detail"] as? String) == "Following usage: 10 experts swapped today." }
    }
    try require(
        hot.current === working && Set(working.keys) == Set(10..<20) && engine.lock.withLock { engine.attached.count } == 1
            && (hot.report["budget_bytes"] as? UInt64 ?? 0) >= UInt64(working.bytes),
        "free space dipped: \(hot.report)")
    try require(
        try reads(working, 13) && FileManager.default.fileExists(atPath: working.payload.path)
            && (try FileManager.default.contentsOfDirectory(atPath: folder.path)).filter { $0.hasSuffix(".bin") }.count == 1,
        "reads after the swaps")

    // (b) The owner lowers the size without room for both: the copy shrinks in place (its least-used records given back)
    // and stays in use.
    set(wanted: 4 * record)
    hot.trigger(counts: counts(Array(0..<10)))
    try waitFor(hot, "a lower size shrinks in place") { (hot.report["detail"] as? String)?.contains("shrunk in place") == true }
    try require(
        hot.current === working && working.keys.count == 4 && hot.report["build_reason"] as? String == "Size limit reduced",
        "lower size, no room: \(working.keys.count) records")
    // With room, the smaller copy is written beside the one in use, which serves until the new one is attached.
    set(free: keep + 1_000_000_000, busy: true)
    hot.trigger()
    try waitFor(hot, "paused while busy") { state(hot) == "paused" }
    try require(hot.current === working && (try reads(working, working.keys[0])), "the copy in use during the rebuild")
    set(busy: false)
    try waitFor(hot, "smaller copy attached") { state(hot) == "ready" && hot.records == 4 }
    let attached = engine.lock.withLock { engine.attached }
    try require(
        attached.count == 2 && attached.allSatisfy { $0 != nil } && hot.current?.keys == Array(0..<4)
            && !FileManager.default.fileExists(atPath: working.payload.path) && (try reads(hot.current!, 2)),
        "lowered: \(attached.map { $0?.keys })")
    try require(
        engine.lock.withLock { engine.events.last?.contains("It replaces the previous copy (0.0 GB): Size limit reduced") == true },
        "rebuild event \(engine.lock.withLock { engine.events })")

    // (d) Detached: the report says so, with no records or bytes, and a log line; back, a log line for the re-attach.
    set(gone: true)
    hot.trigger()
    try waitFor(hot, "detached") { state(hot) == "not_connected" }
    let report = hot.report
    try require(
        hot.current == nil && hot.records == 0 && (report["bytes"] as? NSNumber)?.int64Value == 0 && report["since"] is NSNull
            && engine.lock.withLock {
                engine.attached.last! == nil && engine.events.last?.hasPrefix("Hot copy detached (4 experts") == true
            },
        "after a detach: \(report) \(engine.lock.withLock { (engine.events, engine.attached.count) })")
    set(gone: false)
    hot.trigger()
    try waitFor(hot, "re-attached") { state(hot) == "ready" && hot.records == 4 }
    try require(engine.lock.withLock { engine.events.last?.hasPrefix("Hot copy attached again: 4 experts") == true }, "re-attach log")

    // (c) A build that failed is not retried on every reply, but is once its back-off has passed.
    let second = keeper(root.appendingPathComponent("second"))
    defer { second.stop(waiting: 3) }
    second.retryDelay = 1
    source.lock.withLock { source.fail = true }
    second.trigger(counts: counts(Array(0..<10)))
    try waitFor(second, "failed") { state(second) == "failed" }
    try require(
        engine.lock.withLock {
            engine.events.last?.hasPrefix("Hot copy could not be built (0.0 GB of 0.0 GB written; no copy in use)") == true
        },
        "failure log \(engine.lock.withLock { engine.events })")
    source.lock.withLock { source.fail = false }
    let before = source.lock.withLock { source.reads }
    second.trigger(counts: counts(Array(0..<10)))
    Thread.sleep(forTimeInterval: 0.2)
    try require(state(second) == "failed" && source.lock.withLock { source.reads } == before, "retried at once")
    Thread.sleep(forTimeInterval: 1)
    second.trigger(counts: counts(Array(0..<10)))
    try waitFor(second, "retried") { state(second) == "ready" && second.records == 4 }
}
