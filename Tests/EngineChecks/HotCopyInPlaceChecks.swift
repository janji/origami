import Foundation
import OrigamiEngine

func hotCopyInPlaceChecks() -> [(String, () throws -> Void)] {
    [
        ("hot_copy_inplace_reads_exact_during_swaps", hotCopyInPlaceReadsExact),
        ("hot_copy_inplace_kill_points", hotCopyInPlaceKillPoints),
        ("hot_copy_inplace_index_versions", hotCopyInPlaceIndexVersions),
        ("hot_copy_inplace_swap_rule", hotCopyInPlaceSwapRule),
        ("hot_copy_inplace_margin_holds_on_a_real_boundary", hotCopyInPlaceMarginHolds),
        ("hot_copy_inplace_request_stops_batch", hotCopyInPlaceRequestStopsBatch),
        ("hot_copy_inplace_hung_read_abandons_swap", hotCopyInPlaceHungReadAbandonsSwap),
        ("hot_copy_inplace_verify_during_swaps", hotCopyInPlaceVerifyDuringSwaps),
        ("hot_copy_inplace_give_back_owns_its_slots", hotCopyInPlaceGiveBackOwnsSlots),
        ("hot_copy_inplace_swaps_against_give_backs_hammer", hotCopyInPlaceSwapGiveBackHammer),
        ("hot_copy_inplace_shared_copy_not_changed", hotCopyInPlaceSharedNotChanged),
        ("hot_copy_inplace_rewrite_keeps_the_copy", hotCopyInPlaceRewriteKeeps),
        ("hot_copy_keeper_give_back_counts_what_was_freed", hotCopyKeeperGiveBackCounts),
    ]
}

private func inPlaceRequire(_ ok: @autoclosure () throws -> Bool, _ message: @autoclosure () -> String) throws {
    if try !ok() { throw EngineError("hot copy in place: " + message()) }
}

/// Records of `sizes` bytes (16 KiB unless given) whose every byte depends on the key and its position.
private final class InPlaceRecords: HotCopySource, @unchecked Sendable {
    let sizes: [Int: Int]
    let lock = NSLock()
    var onRead: ((Int) -> Void)?
    init(_ keys: Range<Int>, sizes: [Int: Int] = [:]) {
        self.sizes = Dictionary(uniqueKeysWithValues: keys.map { ($0, sizes[$0] ?? 16384) })
    }
    static func byte(_ key: Int, _ at: Int) -> UInt8 { UInt8(truncatingIfNeeded: key &* 31 &+ at &* 7 &+ 1) }
    func hotCopyRecordBytes(_ key: Int) -> Int? { sizes[key] }
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
        for at in 0..<sizes[key]! { into.storeBytes(of: Self.byte(key, at), toByteOffset: at, as: UInt8.self) }
        lock.withLock { onRead }?(key)
    }
    func matches(_ key: Int, _ bytes: UnsafeRawBufferPointer) -> Bool { bytes.indices.allSatisfy { bytes[$0] == Self.byte(key, $0) } }
    /// Every key `copy` maps reads back as the model's bytes.
    func check(_ copy: HotCopy, _ what: String) throws {
        for key in copy.keys {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: sizes[key]!, alignment: 16384)
            defer { buffer.deallocate() }
            try copy.read(key, into: buffer)
            try inPlaceRequire(matches(key, UnsafeRawBufferPointer(buffer)), "\(what): key \(key) read other bytes")
        }
    }
}

/// `origami refresh --copy` (`rewriteInPlace`): refused while another process has the copy open; alone, both passes
/// run over the whole payload, and the copy is the same afterwards: its bytes, its index, its payload's name, birth and
/// modification time (the `generation` the drive history counts by), read by the copy that was open and by a fresh one.
private func hotCopyInPlaceRewriteKeeps() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<6, sizes: [2: 49152])
    let copy = try HotCopy.write(source: source, keys: [3, 0, 2, 5], folder: folder, identity: "fixture")
    func facts() throws -> [String] {
        let attributes = try FileManager.default.attributesOfItem(atPath: copy.payload.path)
        return [
            copy.generation, "\(attributes[.modificationDate]!)", "\(attributes[.size]!)",
            try Data(contentsOf: copy.payload).base64EncodedString(),
            try Data(contentsOf: folder.appendingPathComponent("index.json")).base64EncodedString(),
        ]
    }
    let before = try facts()
    do {
        let other = HotCopy.open(folder: folder, identity: "fixture", source: source)!
        do {
            try copy.rewriteInPlace { _ in }
            throw EngineError("hot copy in place: rewritten while another process has the copy open")
        } catch is HotCopy.SharedElsewhere {}
        try source.check(other, "the other process")
    }
    var swept: Int64 = 0
    try copy.rewriteInPlace { swept = $0 }
    try inPlaceRequire(swept == 2 * copy.bytes, "swept \(swept) of \(2 * copy.bytes)")
    try inPlaceRequire(try facts() == before, "the copy changed")
    try source.check(copy, "the open copy")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "a fresh open")
}

private func inPlaceFolder() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-inplace-\(UUID().uuidString)")
}

/// (b) Four threads read every key continuously (through `holds` and `read`, as the engines do) while slots are swapped
/// back and forth: every read served by the copy is the model's bytes for that key, and every key is served by the copy
/// again once its swap is published.
private func hotCopyInPlaceReadsExact() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<16)
    let copy = try HotCopy.write(source: source, keys: Array(0..<8), folder: folder, identity: "fixture")
    final class Tally: @unchecked Sendable {
        let lock = NSLock()
        var served = 0, fellBack = 0, wrong: [Int] = []
        var done = false
    }
    let tally = Tally()
    let group = DispatchGroup()
    for thread in 0..<4 {
        group.enter()
        Thread.detachNewThread {
            defer { group.leave() }
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 16384, alignment: 16384)
            defer { buffer.deallocate() }
            var key = thread
            while !tally.lock.withLock({ tally.done }) {
                key = (key + 1) % 16
                let ok = copy.holds(key) && (try? copy.read(key, into: buffer)) != nil
                let right = !ok || source.matches(key, UnsafeRawBufferPointer(buffer))
                tally.lock.withLock {
                    if ok { tally.served += 1 } else { tally.fellBack += 1 }
                    if !right { tally.wrong.append(key) }
                }
            }
        }
    }
    for round in 0..<6 {
        let (from, to) = round % 2 == 0 ? (0..<8, 8..<16) : (8..<16, 0..<8)
        for (slot, key) in zip(copy.slots.indices, to) {
            try inPlaceRequire(from.contains(copy.slots[slot].key), "round \(round) slot \(slot)")
            try copy.replace(slot: slot, with: key, source: source)
        }
        try inPlaceRequire(copy.keys == Array(to), "round \(round) keys \(copy.keys)")
    }
    tally.lock.withLock { tally.done = true }
    group.wait()
    try tally.lock.withLock {
        try inPlaceRequire(tally.wrong.isEmpty && tally.served > 0, "wrong bytes for \(tally.wrong) (\(tally.served) served)")
    }
    try source.check(copy, "after the swaps")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "reopened")
}

/// (c) A stop after each step of `HotCopy.replace` (the write order) leaves a copy that opens with every mapped key on
/// its own bytes: before the empty index is published the old key, then an empty slot (even with torn bytes in it),
/// then the new key. A leftover staging index is ignored.
private func hotCopyInPlaceKillPoints() throws {
    struct Killed: Error {}
    let source = InPlaceRecords(0..<8)
    for kill in 1...5 {
        let folder = inPlaceFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        _ = try HotCopy.write(source: source, keys: [0, 1, 2], folder: folder, identity: "fixture")
        // The killed process's copy is gone before the next one opens (an open copy is changed by no one else).
        let payload = try { () throws -> URL in
            let copy = HotCopy.open(folder: folder, identity: "fixture", source: source)!
            do {
                try copy.replace(slot: 1, with: 5, source: source) { if $0 == kill { throw Killed() } }
            } catch is Killed {}
            return copy.payload
        }()
        if kill == 2 || kill == 3 {
            // Torn or unflushed bytes in the slot the index says is empty.
            let handle = try FileHandle(forWritingTo: payload)
            try handle.seek(toOffset: 16384 + 100)
            try handle.write(contentsOf: Data(repeating: 0xEE, count: 64))
            try handle.close()
        }
        try Data("{\"partial".utf8).write(to: folder.appendingPathComponent(".index.json.staging"))
        guard let reopened = HotCopy.open(folder: folder, identity: "fixture", source: source) else {
            throw EngineError("hot copy in place: no copy opens after a stop at step \(kill)")
        }
        let want = kill == 1 ? [0, 1, 2] : kill == 5 ? [0, 5, 2] : [0, 2]
        try inPlaceRequire(reopened.keys == want, "stop at step \(kill): keys \(reopened.keys)")
        try source.check(reopened, "stop at step \(kill)")
        // A later swap fills the slot left empty.
        if want == [0, 2] {
            try reopened.replace(slot: 1, with: 6, source: source)
            try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "refilled after step \(kill)")
        }
    }
}

/// (d) A copy in the original index (no `version`) opens, is appended to without changing form, and is written in the
/// slot form (`version` 2) only at its first swap; an empty slot is key -1, which an older build's reader refuses (its
/// keys must be nonnegative), as this one refuses -1 without a version and any newer version. The keeper then builds a
/// new copy there.
private func hotCopyInPlaceIndexVersions() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<8)
    let index = folder.appendingPathComponent("index.json")
    func json() throws -> [String: Any] { try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as! [String: Any] }
    // The first object is gone once the top-up returns (an open copy is changed by no one else).
    let grown = try HotCopy.write(
        source: source, keys: [2], folder: folder, identity: "fixture",
        onto: try HotCopy.write(source: source, keys: [0, 1], folder: folder, identity: "fixture"))
    try inPlaceRequire(try json()["version"] == nil && !grown.slotIndex, "original form changed before a swap")
    struct Killed: Error {}
    try? grown.replace(slot: 0, with: 4, source: source) { if $0 == 2 { throw Killed() } }
    try inPlaceRequire(try json()["version"] as? Int == HotCopy.slotVersion && (try json()["keys"] as? [Int]) == [-1, 1, 2], "slot form")
    try grown.replace(slot: 0, with: 4, source: source)
    // An append onto a slot-form copy keeps the form.
    let more = try HotCopy.write(source: source, keys: [3], folder: folder, identity: "fixture", onto: grown)
    try inPlaceRequire(try json()["version"] as? Int == HotCopy.slotVersion && more.keys == [4, 1, 2, 3], "append after a swap")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "slot form reopened")
    let saved = try Data(contentsOf: index)
    var unversioned = try json()
    unversioned["version"] = nil
    unversioned["keys"] = [-1, 1, 2, 3]
    try JSONSerialization.data(withJSONObject: unversioned).write(to: index)
    try inPlaceRequire(HotCopy.open(folder: folder, identity: "fixture", source: source) == nil, "an empty slot without a version opened")
    var newer = try JSONSerialization.jsonObject(with: saved) as! [String: Any]
    newer["version"] = HotCopy.slotVersion + 1
    try JSONSerialization.data(withJSONObject: newer).write(to: index)
    try inPlaceRequire(HotCopy.open(folder: folder, identity: "fixture", source: source) == nil, "a newer index opened")
    // The keeper meets the newer index: no copy, so it builds one.
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: (Dictionary(uniqueKeysWithValues: (0..<8).map { ($0, 10 - $0) }), [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: 4 * 16384, free: .max) }, attach: { _ in true }, idle: { true },
        event: { _ in })
    defer { keeper.stop(waiting: 3) }
    keeper.trigger()
    let end = Date().addingTimeInterval(10)
    while keeper.records != 4 || keeper.report["state"] as? String != "ready" {
        guard Date() < end else { throw EngineError("hot copy in place: a newer index was not rebuilt (\(keeper.report))") }
        Thread.sleep(forTimeInterval: 0.01)
    }
    try inPlaceRequire(try json()["version"] == nil && keeper.current?.keys == [0, 1, 2, 3], "rebuilt over a newer index")
    try source.check(keeper.current!, "rebuilt")
}

/// (e) The swap rule: an outside record takes the least-used slot of its own size only when clearly more used (by more
/// than a fifth, misses first); similar counts, alternating, swap nothing; an empty slot is filled first; the byte
/// bound holds. Through the keeper, alternating similar counts leave the copy and its index untouched.
private func hotCopyInPlaceSwapRule() throws {
    let size = { (key: Int) -> UInt64 in key >= 100 ? 32768 : 16384 }
    let slots: [(key: Int, bytes: UInt64)] = [(1, 16384), (2, 16384), (100, 32768)]
    func swaps(
        _ uses: [Int: Int], _ misses: [Int: Int] = [:], chosen: [Int], slots: [(key: Int, bytes: UInt64)] = slots, limit: UInt64 = .max
    )
        -> [Int]
    {
        HotCopyKeeper.swaps(slots: slots, chosen: chosen, counts: (uses, misses), size: size, limit: limit).map { $0.key * 10 + $0.slot }
    }
    try inPlaceRequire(swaps([1: 100, 2: 300, 3: 110, 100: 5], chosen: [2, 3, 100]).isEmpty, "similar uses swapped")
    try inPlaceRequire(swaps([1: 110, 2: 300, 3: 100, 100: 5], chosen: [2, 1, 100]).isEmpty, "nothing to swap")
    try inPlaceRequire(swaps([1: 100, 2: 300, 3: 126, 100: 5], chosen: [2, 3, 100]) == [30], "clearly more not swapped")
    try inPlaceRequire(swaps([1: 100, 2: 300, 3: 1, 100: 5], [3: 1], chosen: [2, 3, 100]) == [30], "a miss ranks first")
    try inPlaceRequire(swaps([1: 5, 2: 300, 100: 1, 101: 1000], chosen: [101, 2]) == [1012], "size class")
    try inPlaceRequire(
        swaps([2: 300, 3: 1, 100: 5], chosen: [2, 3, 100], slots: [(-1, 16384), (2, 16384), (100, 32768)]) == [30],
        "empty slot not filled")
    try inPlaceRequire(swaps([1: 1, 2: 1, 3: 9, 4: 9], chosen: [3, 4], limit: 16384).count == 1, "byte bound")

    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<4)
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: ([0: 100, 1: 100, 2: 1, 3: 1], [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: 2 * 16384, free: 0) }, attach: { _ in true }, idle: { true },
        event: { _ in })
    defer { keeper.stop(waiting: 3) }
    keeper.trigger()
    func settle() throws {
        let end = Date().addingTimeInterval(10)
        repeat {
            guard Date() < end else { throw EngineError("hot copy in place: keeper never settled (\(keeper.report))") }
            Thread.sleep(forTimeInterval: 0.05)
        } while keeper.report["state"] as? String != "ready"
    }
    try settle()
    guard let copy = keeper.current, copy.keys == [0, 1] else { throw EngineError("hot copy in place: first build \(keeper.report)") }
    for n in 0..<6 {
        let high = n % 2 == 0 ? 110 : 95
        keeper.trigger(counts: ([0: 100, 1: 100, 2: high, 3: 1], [:]))
        try settle()
    }
    Thread.sleep(forTimeInterval: 0.3)
    try inPlaceRequire(
        keeper.current === copy && copy.keys == [0, 1] && !copy.slotIndex && keeper.report["swapped_today"] as? Int == 0,
        "alternating similar counts swapped: \(copy.keys) \(keeper.report)")
}

/// (f) A request arriving while a batch runs stops it after the record in progress; the copy stays attached with every
/// key on its bytes, and the next idle window finishes the batch. One log line per batch, and the state says how many
/// experts were swapped today.
private func hotCopyInPlaceRequestStopsBatch() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<8)
    final class Engine: @unchecked Sendable {
        let lock = NSLock(), work = NSLock()
        var busy = false, attached = 0, events: [String] = []
    }
    let engine = Engine()
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: ([0: 9, 1: 9, 2: 9, 3: 9], [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: 4 * 16384, free: 0) },
        attach: { _ in
            engine.lock.withLock { engine.attached += 1 }
            return true
        },
        idle: { engine.lock.withLock { !engine.busy } },
        withIdle: { work in
            guard engine.lock.withLock({ !engine.busy }) else { return false }
            try engine.work.withLock { try work() }
            return true
        },
        event: { line in engine.lock.withLock { engine.events.append(line) } })
    defer { keeper.stop(waiting: 3) }
    func settle(_ what: String, _ done: () -> Bool) throws {
        let end = Date().addingTimeInterval(10)
        while !done() {
            guard Date() < end else { throw EngineError("hot copy in place: \(what) (\(keeper.report))") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }
    keeper.trigger()
    try settle("first build") { keeper.records == 4 && keeper.report["state"] as? String == "ready" }
    let copy = keeper.current!
    // The first record read for a swap is when a request arrives.
    source.lock.withLock { source.onRead = { _ in engine.lock.withLock { engine.busy = true } } }
    keeper.trigger(counts: ([0: 1, 1: 1, 2: 1, 3: 1, 4: 50, 5: 50, 6: 50, 7: 50], [:]))
    try settle("one swap") { engine.lock.withLock { engine.events.count } == 2 }
    try inPlaceRequire(
        Set(copy.keys).intersection(4..<8).count == 1 && copy.keys.count == 4
            && engine.lock.withLock { engine.events.last!.hasPrefix("Hot copy follows usage: 1 swapped") },
        "a request did not stop the batch: \(copy.keys) \(engine.lock.withLock { engine.events })")
    try source.check(copy, "stopped batch")
    source.lock.withLock { source.onRead = nil }
    engine.lock.withLock { engine.busy = false }
    keeper.trigger()
    try settle("batch finished") { Set(copy.keys) == Set(4..<8) }
    try settle("state text") { keeper.report["detail"] as? String == "Following usage: 4 experts swapped today." }
    let events = engine.lock.withLock { engine.events }
    try inPlaceRequire(
        keeper.current === copy && engine.lock.withLock { engine.attached } == 1 && events.count == 3
            && events[2].contains("3 swapped (") && events[2].contains("copy before: least used inside 0 misses/1 uses")
            && events[2].contains("closest swap: 0 misses/50 uses in for 0 misses/1 uses out")
            && events[2].contains("after: least used inside 0 misses/50 uses, most used outside 0 misses/1 uses"),
        "after the batch: \(events)")
    let payloads = try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasSuffix(".bin") }
    try inPlaceRequire(payloads.count == 1, "a second payload: \(payloads)")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "reopened")
}

/// A read of a slot that never returns: the swap waits at most the demand read bound (`ExpertReadStall`), then puts the
/// old key back (it reads its bytes again, the index unchanged); through the keeper, the wait holds no request out.
private func hotCopyInPlaceHungReadAbandonsSwap() throws {
    let saved = ExpertReadStall.floor
    ExpertReadStall.floor = 0.3
    defer { ExpertReadStall.floor = saved }
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<8)
    let direct = try HotCopy.write(source: source, keys: [0, 1, 2, 3], folder: folder, identity: "fixture")
    let release = direct.pinRead(0)!
    let started = Date()
    try inPlaceRequire((try? direct.replace(slot: 0, with: 5, source: source)) == nil, "a swap over a hung read went ahead")
    try inPlaceRequire(Date().timeIntervalSince(started) < 5 && direct.holds(0) && direct.keys == [0, 1, 2, 3], "old key not back")
    release()
    try source.check(direct, "after an abandoned swap")
    try inPlaceRequire(!HotCopy.open(folder: folder, identity: "fixture", source: source)!.slotIndex, "the index changed")
    try FileManager.default.removeItem(at: folder)

    final class Engine: @unchecked Sendable {
        let lock = NSLock(), gate = NSLock()
        var events: [String] = []
    }
    let engine = Engine()
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: ([0: 9, 1: 9, 2: 9, 3: 9], [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: 4 * 16384, free: 0) }, attach: { _ in true }, idle: { true },
        withIdle: { work in
            try engine.gate.withLock { try work() }
            return true
        },
        event: { line in engine.lock.withLock { engine.events.append(line) } })
    defer { keeper.stop(waiting: 3) }
    func settle(_ what: String, _ done: () -> Bool) throws {
        let end = Date().addingTimeInterval(10)
        while !done() {
            guard Date() < end else { throw EngineError("hot copy in place: \(what) (\(keeper.report))") }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }
    keeper.trigger()
    try settle("first build") { keeper.records == 4 && keeper.report["state"] as? String == "ready" }
    let copy = keeper.current!
    let hung = copy.pinRead(0)!
    defer { hung() }
    keeper.trigger(counts: ([0: 1, 1: 9, 2: 9, 3: 9, 4: 50], [:]))
    try settle("slot vacated") { !copy.holds(0) }
    // A request is admitted while the keeper waits for the hung read.
    try inPlaceRequire(engine.gate.lock(before: Date().addingTimeInterval(0.2)), "the wait held requests out")
    engine.gate.unlock()
    try settle("swap abandoned") { engine.lock.withLock { engine.events.last?.contains("abandoned") == true } }
    try inPlaceRequire(copy.holds(0) && copy.keys == [0, 1, 2, 3], "old key not back: \(copy.keys)")
    try source.check(copy, "after the keeper abandoned a swap")
}

/// `origami verify` (`HotCopy.verify`) while the server swaps: verify has the copy open, so the server's swap waits for
/// it (the copy is not changed under a reader); an empty slot is not a problem; real damage still is.
private func hotCopyInPlaceVerifyDuringSwaps() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<8)
    let copy = try HotCopy.write(source: source, keys: [0, 1, 2], folder: folder, identity: "fixture")
    var refused = false
    let raced = HotCopy.verify(folder: folder, identity: "fixture") { key in
        if key == 1 && !refused {
            do { try copy.replace(slot: 1, with: 5, source: source) } catch is HotCopy.SharedElsewhere { refused = true } catch {}
        }
        return source.hotCopyRecordBytes(key)
    }
    try inPlaceRequire(refused && copy.keys == [0, 1, 2] && raced?.problems == [], "a swap during verify: \(String(describing: raced))")
    try copy.replace(slot: 1, with: 5, source: source)
    struct Killed: Error {}
    try? copy.replace(slot: 1, with: 6, source: source) { if $0 == 2 { throw Killed() } }
    let empty = HotCopy.verify(folder: folder, identity: "fixture", recordBytes: source.hotCopyRecordBytes)
    try inPlaceRequire(empty?.problems == [] && empty?.checked == 2, "an empty slot: \(String(describing: empty))")
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.seek(toOffset: 100)
    try handle.write(contentsOf: Data([0xEE]))
    try handle.close()
    let damaged = HotCopy.verify(folder: folder, identity: "fixture", recordBytes: source.hotCopyRecordBytes)
    try inPlaceRequire(damaged?.problems.count == 1, "damage not reported: \(String(describing: damaged))")
}

/// The first real-model swap log: inside 858 misses/2242 uses, outside 863 misses/831 uses. Within the margin, so no
/// swap, except where the record inside is one a faster drive's copy also holds (then it counts as unused). Through the
/// keeper, a boundary jittering within the margin swaps nothing over 100 passes.
private func hotCopyInPlaceMarginHolds() throws {
    let counts = (uses: [0: 3000, 1: 2242, 2: 831, 3: 10], misses: [0: 900, 1: 858, 2: 863, 3: 10])
    let slots: [(key: Int, bytes: UInt64)] = [(0, 16384), (1, 16384)]
    let size = { (_: Int) -> UInt64 in 16384 }
    try inPlaceRequire(
        !HotCopyKeeper.clearlyMore((863, 831), than: (858, 2242))
            && HotCopyKeeper.swaps(slots: slots, chosen: [0, 2], counts: counts, size: size, limit: .max).isEmpty,
        "863/831 replaced 858/2242")
    try inPlaceRequire(
        HotCopyKeeper.swaps(slots: slots, chosen: [0, 2], counts: counts, size: size, limit: .max, redundant: [1]).map(\.key) == [2],
        "a duplicate was not replaced first")

    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<4)
    final class Events: @unchecked Sendable {
        let lock = NSLock()
        var lines: [String] = []
    }
    let events = Events()
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: counts,
        place: { _ in .folder(folder, drive: "fixture", budget: 2 * 16384, free: 0) }, attach: { _ in true }, idle: { true },
        event: { line in events.lock.withLock { events.lines.append(line) } })
    defer { keeper.stop(waiting: 3) }
    keeper.trigger()
    let end = Date().addingTimeInterval(10)
    while keeper.records != 2 || keeper.report["state"] as? String != "ready" {
        guard Date() < end else { throw EngineError("hot copy in place: first build (\(keeper.report))") }
        Thread.sleep(forTimeInterval: 0.01)
    }
    let copy = keeper.current!
    try inPlaceRequire(Set(copy.keys) == [0, 2], "first build \(copy.keys)")
    // From here 1 (outside) and 2 (inside) trade ranks back and forth within a fifth of each other.
    for pass in 0..<100 {
        let low = 858 + pass % 7
        let high = low + 1 + pass % 100
        let (inside, outside) = pass % 2 == 0 ? (low, high) : (high, low)
        keeper.trigger(
            counts: (uses: [0: 3000, 1: 2242, 2: 2242 + pass % 300, 3: 10], misses: [0: 900, 1: outside, 2: inside, 3: 10]))
        Thread.sleep(forTimeInterval: 0.01)
    }
    Thread.sleep(forTimeInterval: 0.3)
    try inPlaceRequire(
        keeper.current === copy && Set(copy.keys) == [0, 2] && !copy.slotIndex && keeper.report["swapped_today"] as? Int == 0
            && events.lock.withLock { !events.lines.contains { $0.contains("follows usage") } },
        "swaps within the margin: \(copy.keys) \(events.lock.withLock { events.lines })")
}

/// Reads `key` from `copy` into a fresh buffer: nil when the copy does not serve it, else whether the bytes are the
/// model's.
private func inPlaceServes(_ copy: HotCopy, _ key: Int, _ source: InPlaceRecords) -> Bool? {
    let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: source.sizes[key]!, alignment: 16384)
    defer { buffer.deallocate() }
    guard copy.holds(key), (try? copy.read(key, into: buffer)) != nil else { return nil }
    return source.matches(key, UnsafeRawBufferPointer(buffer))
}
private func inPlaceIndexKeys(_ folder: URL) throws -> [Int] {
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("index.json"))) as! [String: Any]
    return object["keys"] as! [Int]
}
private final class InPlaceLines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func add(_ line: String) { lock.withLock { lines.append(line) } }
    var all: [String] { lock.withLock { lines } }
}
private func inPlaceWait(_ what: String, _ keeper: HotCopyKeeper, _ done: () -> Bool) throws {
    let end = Date().addingTimeInterval(10)
    while !done() {
        guard Date() < end else { throw EngineError("hot copy in place: \(what) (\(keeper.report))") }
        Thread.sleep(forTimeInterval: 0.01)
    }
}

/// A give-back (from a save's thread) owns the slots it empties: the keeper's swap picking the same slot meanwhile (after
/// the give-back took the slot out of the lookup, before its index) can neither take it nor put its key back, so the
/// freed slot is never mapped to its punched (zero) bytes; it serves again only once a swap wrote and verified it. The
/// other way round: a slot a swap holds is not freed by a give-back, and its key comes back on its own bytes.
private func hotCopyInPlaceGiveBackOwnsSlots() throws {
    let folder = inPlaceFolder()
    defer {
        HotCopy.stopForChecks = nil
        try? FileManager.default.removeItem(at: folder)
    }
    let source = InPlaceRecords(0..<16)
    let copy = try HotCopy.write(source: source, keys: Array(0..<6), folder: folder, identity: "fixture")
    // Every key read (and so checked) once: a key mapped to other bytes later would be read unchecked.
    try source.check(copy, "before")
    var swap: (owned: Bool, replaced: Bool)?
    HotCopy.stopForChecks = { step in
        if step == "vacated" && swap == nil {
            // The keeper's `follow` picks slot 2 now: its vacate, then its swap (which wants the writer lock).
            let owned = copy.vacate(2) != nil
            swap = (owned, (try? copy.replace(slot: 2, with: 10, source: source)) != nil)
        }
        return false
    }
    let freed = try copy.giveBack([2])
    HotCopy.stopForChecks = nil
    try inPlaceRequire(swap.map { !$0.owned && !$0.replaced } == true, "the swap took a slot being given back: \(String(describing: swap))")
    try inPlaceRequire(freed.count == 1 && freed[0] == (2, 2), "freed \(freed)")
    try inPlaceRequire(
        !copy.holds(2) && inPlaceServes(copy, 2, source) == nil && copy.keys == [0, 1, 3, 4, 5], "key 2 still mapped: \(copy.keys)")
    try inPlaceRequire(try inPlaceIndexKeys(folder) == [0, 1, -1, 3, 4, 5] && copy.emptyHeld.isEmpty, "slot 2 not empty and freed on disk")
    try source.check(copy, "after the give-back")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "reopened after the give-back")
    // A later swap fills the freed slot, written and verified.
    try copy.replace(slot: 2, with: 10, source: source)
    try inPlaceRequire(inPlaceServes(copy, 10, source) == true, "the refilled slot")

    // A swap owns slot 4: the give-back frees slot 5 only; another owner cannot end the swap; the swap ends and key 4
    // reads its own bytes again.
    let owner = copy.vacate(4)!
    let second = try copy.giveBack([4, 5])
    try inPlaceRequire(second.map { $0.slot } == [5] && second[0].key == 5, "a give-back freed a slot a swap holds: \(second)")
    copy.restore(4, owner: owner &+ 1)
    try inPlaceRequire(!copy.holds(4), "another owner put the key back")
    copy.restore(4, owner: owner)
    try inPlaceRequire(
        inPlaceServes(copy, 4, source) == true && copy.keys == [0, 1, 10, 3, 4], "slot 4 after the swap ended: \(copy.keys)")
    try source.check(HotCopy.open(folder: folder, identity: "fixture", source: source)!, "reopened at the end")
}

/// Swaps as the keeper's `follow` makes them (vacate, drain, replace with that ownership) against give-backs from a
/// second thread on the same few slots, while a third thread reads every mapped key: every read the copy serves is the
/// model's bytes for that key (never zeros, never another record's), over 2,000 rounds of each; at the end the copy
/// opened again from disk maps the same keys on their own bytes.
private func hotCopyInPlaceSwapGiveBackHammer() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<12)
    let copy = try HotCopy.write(source: source, keys: Array(0..<4), folder: folder, identity: "fixture")
    final class Tally: @unchecked Sendable {
        let lock = NSLock()
        var served = 0, wrong: [Int] = [], swaps = 0, freed = 0, taken = 0, writers = 2, errors: [String] = []
    }
    let tally = Tally()
    let rounds = 2000
    let group = DispatchGroup()
    func run(writer: Bool, _ body: @escaping @Sendable () -> Void) {
        group.enter()
        Thread.detachNewThread {
            defer {
                if writer { tally.lock.withLock { tally.writers -= 1 } }
                group.leave()
            }
            body()
        }
    }
    run(writer: false) {
        var key = 0
        while tally.lock.withLock({ tally.writers > 0 }) {
            key = (key + 1) % 12
            guard let right = inPlaceServes(copy, key, source) else { continue }
            tally.lock.withLock {
                tally.served += 1
                if !right { tally.wrong.append(key) }
            }
        }
    }
    run(writer: true) {
        for _ in 0..<rounds {
            // A pause between rounds, as between the keeper's records: else each writer takes the lock straight back.
            defer { Thread.sleep(forTimeInterval: .random(in: 0...0.004)) }
            let slot = Int.random(in: 0..<4)
            let held = Set(copy.keys)
            guard let key = (0..<12).filter({ !held.contains($0) }).randomElement() else { continue }
            guard let owner = copy.vacate(slot) else {
                tally.lock.withLock { tally.taken += 1 }
                continue
            }
            defer { copy.restore(slot, owner: owner) }
            guard copy.drain(slot, within: 1) else { continue }
            do {
                try copy.replace(slot: slot, with: key, source: source, within: 0, owner: owner)
                tally.lock.withLock { tally.swaps += 1 }
            } catch {
                // The give-back holding the writer lock, or the key taken meanwhile: the swap waits for the next round.
                let text = "\(error)"
                if !text.contains("Another hot-copy writer") && !text.contains("does not fit") {
                    tally.lock.withLock { tally.errors.append(text) }
                }
            }
        }
    }
    run(writer: true) {
        for _ in 0..<rounds {
            defer { Thread.sleep(forTimeInterval: .random(in: 0...0.004)) }
            let slots = (0..<Int.random(in: 1...2)).map { _ in Int.random(in: 0..<4) }
            do {
                let freed = try copy.giveBack(slots)
                tally.lock.withLock { tally.freed += freed.filter { $0.key >= 0 }.count }
            } catch {
                // A swap holding the writer lock past the wait: the give-back is skipped this round.
                if !"\(error)".contains("Another hot-copy writer") { tally.lock.withLock { tally.errors.append("\(error)") } }
            }
        }
    }
    guard group.wait(timeout: .now() + 1200) == .success else { throw EngineError("hot copy in place: the hammer did not finish") }
    try tally.lock.withLock {
        try inPlaceRequire(
            tally.wrong.isEmpty && tally.errors.isEmpty && tally.served > 0 && tally.swaps > 0 && tally.freed > 0,
            "wrong bytes for \(tally.wrong.prefix(10)) (\(tally.served) served, \(tally.swaps) swaps, \(tally.freed) freed, "
                + "\(tally.taken) slots held by a give-back), errors \(tally.errors.prefix(3))")
    }
    let reopened = HotCopy.open(folder: folder, identity: "fixture", source: source)!
    try inPlaceRequire(reopened.keys == copy.keys, "on disk \(reopened.keys), in memory \(copy.keys)")
    try source.check(reopened, "reopened after the hammer")
    try source.check(copy, "after the hammer")
}

/// Another process with the copy open (a second copy object of the folder stands in for it: `flock` holds per open
/// file, so two in one process exclude each other as two processes do): the copy is neither swapped nor given back in
/// place, and the keeper says so; a save's room comes from the next giver. A new payload (a top-up past the other
/// reader's records, a rebuild) is still written. A reader opening during a swap waits for it and reads the new index.
/// A process without the lock (an older build) holding the payload open counts the same.
private func hotCopyInPlaceSharedNotChanged() throws {
    let folder = inPlaceFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = InPlaceRecords(0..<12)
    var server: HotCopy? = try HotCopy.write(source: source, keys: Array(0..<4), folder: folder, identity: "fixture")
    do {
        let other = HotCopy.open(folder: folder, identity: "fixture", source: source)!
        try source.check(other, "the other process")
        do {
            try server!.replace(slot: 1, with: 6, source: source)
            throw EngineError("hot copy in place: swapped while another process has the copy open")
        } catch is HotCopy.SharedElsewhere {}
        do {
            _ = try server!.giveBack([0])
            throw EngineError("hot copy in place: gave back while another process has the copy open")
        } catch is HotCopy.SharedElsewhere {}
        try inPlaceRequire(
            server!.keys == [0, 1, 2, 3] && !server!.slotIndex && (try inPlaceIndexKeys(folder)) == [0, 1, 2, 3], "changed under a reader")
        try source.check(server!, "the server after refusing")
    }
    // The other process is gone: the server's own hold stayed shared through the refused tries (no one takes it whole).
    let probe = open(folder.appendingPathComponent(".attached.lock").path, O_RDONLY)
    defer { close(probe) }
    try inPlaceRequire(probe >= 0 && flock(probe, LOCK_EX | LOCK_NB) != 0, "the server's copy lost its shared hold")
    do {
        let other = HotCopy.open(folder: folder, identity: "fixture", source: source)!
        // Whole-file changes: a top-up (past every record the other reader maps), then a rebuild in a new payload.
        server = try HotCopy.write(source: source, keys: [4], folder: folder, identity: "fixture", onto: server)
        try source.check(other, "the other process after a top-up")
        server = try HotCopy.write(source: source, keys: [8, 9], folder: folder, identity: "fixture")
        try source.check(other, "the other process after a rebuild")
        try source.check(server!, "the rebuilt copy")
    }
    let opened = DispatchSemaphore(value: 0)
    final class Box: @unchecked Sendable { var keys: [Int]? }
    let box = Box()
    try server!.replace(slot: 0, with: 10, source: source) { step in
        guard step == 2 else { return }
        Thread.detachNewThread {
            box.keys = HotCopy.open(folder: folder, identity: "fixture", source: source)?.keys
            opened.signal()
        }
        try inPlaceRequire(opened.wait(timeout: .now() + 0.2) == .timedOut, "a reader opened in the middle of a swap")
    }
    opened.wait()
    try inPlaceRequire(box.keys == [10, 9], "the reader waiting for the swap read \(String(describing: box.keys))")
    // An older build (no lock) with the payload open.
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sleep")
    process.arguments = ["30"]
    process.standardInput = FileHandle(forReadingAtPath: server!.payload.path)
    try process.run()
    defer { process.terminate() }
    do {
        try server!.replace(slot: 1, with: 11, source: source)
        throw EngineError("hot copy in place: swapped while an older build has the payload open")
    } catch is HotCopy.SharedElsewhere {}
    process.terminate()
    process.waitUntilExit()
    // A scan that does not come back (a process blocked on a busy drive): open after its bound, open at once while that
    // scan is still out, and the real answer again once it has returned.
    let held = DispatchSemaphore(value: 0)
    let began = Date()
    try inPlaceRequire(
        HotCopy.openElsewhere(folder, seconds: 0.2) { _ in
            held.wait()
            return false
        }, "a scan that hangs read as not open")
    try inPlaceRequire(HotCopy.openElsewhere(folder), "a second scan ran beside one that hangs")
    try inPlaceRequire(Date().timeIntervalSince(began) < 1.5, "the scan was not bounded")
    held.signal()
    var recovered = false
    for _ in 0..<100 where !recovered {
        recovered = !HotCopy.openElsewhere(folder)
        if !recovered { usleep(20_000) }
    }
    try inPlaceRequire(recovered, "the scan did not recover once the hanging one returned")

    // Through the keeper: no swap and no give-back while another copy object is open; a save's room comes from the
    // next giver. Once it closes, both go ahead.
    let lines = InPlaceLines()
    let attached = server!
    server = nil
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: attached, counts: ([10: 1, 9: 1, 0: 50, 1: 50], [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: UInt64(attached.bytes), free: .max) }, attach: { _ in true },
        idle: { true }, event: lines.add)
    defer { keeper.stop(waiting: 3) }
    var other: HotCopy? = HotCopy.open(folder: folder, identity: "fixture", source: source)
    keeper.trigger()
    let shared = "Another Origami process has this copy open: it is not changed while that lasts."
    try inPlaceWait("the keeper says why", keeper) { keeper.report["detail"] as? String == shared }
    final class Asked: @unchecked Sendable { var bytes: UInt64 = 0 }
    let asked = Asked()
    let id = DiskBudget.shared.add(
        "next giver", folder: { folder }, bytes: { 0 },
        giveBack: { bytes, _ in
            asked.bytes += bytes
            return bytes
        })
    defer { DiskBudget.shared.remove(id) }
    let refused = keeper.giveBack(16384, for: "check")
    let fits = DiskBudget.shared.makeRoom(16384, in: folder, for: "check", floor: 0, free: { asked.bytes > 0 ? 16384 : 0 })
    try inPlaceRequire(
        refused == 0 && fits && asked.bytes > 0 && attached.keys == [10, 9] && keeper.report["given_back_records"] as? Int == 0
            && lines.all.isEmpty && keeper.report["detail"] as? String == shared,
        "a save while shared: gave \(refused), fits \(fits), asked \(asked.bytes), keys \(attached.keys), \(lines.all)")
    try source.check(other!, "the other process at the end")
    other = nil
    keeper.trigger()
    try inPlaceWait("swaps once alone", keeper) { Set(attached.keys) == [0, 1] }
    try inPlaceRequire(keeper.giveBack(16384, for: "check") == 16384, "no give-back once alone")
    try source.check(attached, "after the keeper's swaps")
}

/// The keeper's give-back report counts what was freed, not what it planned: a slot whose read does not return is kept,
/// and a give-back that stops after its index counts nothing.
private func hotCopyKeeperGiveBackCounts() throws {
    let folder = inPlaceFolder()
    let saved = ExpertReadStall.floor
    ExpertReadStall.floor = 0.2
    defer {
        ExpertReadStall.floor = saved
        HotCopy.stopForChecks = nil
        try? FileManager.default.removeItem(at: folder)
    }
    let source = InPlaceRecords(0..<8)
    let copy = try HotCopy.write(source: source, keys: Array(0..<8), folder: folder, identity: "fixture")
    let lines = InPlaceLines()
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: copy, counts: (Dictionary(uniqueKeysWithValues: (0..<8).map { ($0, $0 + 1) }), [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: UInt64(copy.bytes), free: .max) }, attach: { _ in true },
        idle: { false }, event: lines.add)
    defer { keeper.stop(waiting: 3) }
    // Keys 0, 1, 2 are the least used; a read of key 1 does not return.
    let release = copy.pinRead(1)!
    let given = keeper.giveBack(3 * 16384, for: "check")
    release()
    try inPlaceRequire(
        given == 2 * 16384 && keeper.report["given_back_records"] as? Int == 2
            && lines.all.last?.contains("(2 least-used experts)") == true,
        "counted \(keeper.report["given_back_records"] ?? "nil") for \(given) bytes: \(lines.all)")
    HotCopy.stopForChecks = { $0 == "published" }
    let stopped = keeper.giveBack(16384, for: "check")
    try inPlaceRequire(
        stopped == 0 && keeper.report["given_back_records"] as? Int == 2 && lines.all.last?.contains("(0 least-used experts)") == true,
        "a stopped give-back counted \(keeper.report["given_back_records"] ?? "nil"): \(lines.all)")
    try source.check(copy, "after the give-backs")
}
