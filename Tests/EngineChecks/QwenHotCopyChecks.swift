import Foundation
import OrigamiEngine

private func qwenHotRequire(_ value: Bool, _ message: @autoclosure () -> String) throws {
    if !value { throw EngineError("Qwen hot copy: " + message()) }
}
func qwenRecord(_ expert: Int) -> [UInt8] {
    (0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: $0 &* 13 &+ expert &* 101 &+ 5) }
}
/// A checkpoint whose layer 0 holds experts 0 to 3 (`qwenRecord`), in the spans `PackStore` reads them from.
func qwenSpanModel(_ directory: URL, fingerprint: String = String(repeating: "e", count: 64)) throws -> ModelSource {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent("model.bin")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    let handle = try FileHandle(forWritingTo: file)
    let records = (0..<4).map(qwenRecord)
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
    return try ModelSource(fixture: directory, spans: spans, fingerprint: fingerprint)
}
private func qwenRead(_ store: PackStore, _ expert: Int) throws -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
    try bytes.withUnsafeMutableBytes { try store.readInternalExpert(ExpertKey(layer: 0, expert: expert), into: $0) }
    return bytes
}

/// Qwen's hot copy (a `HotCopy` in place of the offline mirror): it opens only for its model's identity and record
/// size, reads exact records through `PackStore.readInternalExpert`, counts them, and charges its index while the store
/// lives; a key it does not hold, a record failing its CRC32 (that key alone) and a copy whose file fails a read (all
/// keys, for 30 s) read the same bytes from the model's spans.
func hotExpertCopyIdentityBytesAndRetirement() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-qwen-hot-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try qwenSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model)
    let folder = root.appendingPathComponent("hot")
    let copy = try HotCopy.write(source: source, keys: [1, 2], folder: folder, identity: "qwen m")
    try qwenHotRequire(
        HotCopy.open(folder: folder, identity: "qwen other") == nil
            && HotCopy.open(folder: folder, identity: "qwen m", source: source) != nil,
        "identity")
    final class Wider: HotCopySource {
        func hotCopyRecordBytes(_ key: Int) -> Int? { PackStore.expertBytes + 16384 }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {}
    }
    try qwenHotRequire(HotCopy.open(folder: folder, identity: "qwen m", source: Wider()) == nil, "a copy of another record size opened")
    let ledger = try MemoryLedger(ceiling: 64 * 1024 * 1024, transientMargin: 1024 * 1024)
    var store: PackStore? = try PackStore(model: model)
    try store!.reserveHotCopyMetadata(ledger: ledger)
    try qwenHotRequire(ledger.snapshot().reserved == PackStore.hotCopyMetadataBytes, "index not charged")
    store!.attachHotCopy(copy)
    let held = try ExpertKey(layer: 0, expert: 1)
    let absent = try ExpertKey(layer: 0, expert: 3)
    try qwenHotRequire(store!.hasHotExpert(held) && !store!.hasHotExpert(absent), "key lookup")
    try qwenHotRequire(try qwenRead(store!, 1) == qwenRecord(1) && store!.status().hotCopyReads == 1, "a held record")
    try qwenHotRequire(try qwenRead(store!, 3) == qwenRecord(3) && store!.status().hotCopyReads == 1, "an absent record")
    store = nil
    try qwenHotRequire(ledger.snapshot().reserved == 0, "the store's end leaked the index charge")
    // A record failing its CRC32: that key reads from the spans, the other still from the copy.
    let corrupt = try FileHandle(forWritingTo: copy.payload)
    try corrupt.seek(toOffset: UInt64(copy.offsets[2]!) + 100)
    try corrupt.write(contentsOf: Data([0xEE]))
    try corrupt.close()
    let checked = HotCopy.open(folder: folder, identity: "qwen m", source: source)!
    let serving = try PackStore(model: model)
    serving.attachHotCopy(checked)
    try qwenHotRequire(try qwenRead(serving, 2) == qwenRecord(2) && serving.status().hotCopyReads == 0, "a corrupt record")
    try qwenHotRequire(!serving.hasHotExpert(try ExpertKey(layer: 0, expert: 2)) && !checked.disabled, "a corrupt record is dropped alone")
    try qwenHotRequire(try qwenRead(serving, 1) == qwenRecord(1) && serving.status().hotCopyReads == 1, "an intact record after it")
    // A file that fails a read: every key reads from the spans, and the copy stops being offered.
    try FileHandle(forWritingTo: copy.payload).truncate(atOffset: 0)
    let cut = try PackStore(model: model)
    let reopened = HotCopy.open(folder: folder, identity: "qwen m")
    try qwenHotRequire(reopened == nil, "a cut file opened")
    cut.attachHotCopy(checked)
    try qwenHotRequire(try qwenRead(cut, 1) == qwenRecord(1) && checked.disabled && !cut.hasHotExpert(held), "a failed read")
}

/// The keeper builds Qwen's copy while serving: nothing while a request holds the engine (`ServingGate`, as the backend
/// admits one), then the most missed records
/// of the counts (`RoutingProfile.hotCopyChoice`), attached between requests; reads then come from it with exact bytes,
/// a corrupted record reads from the spans with exact bytes, and the offline mirror it replaces is retired once.
func hotCopyQwenKeeperBuildsWhileServing() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-qwen-keeper-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try qwenSpanModel(root.appendingPathComponent("model"))
    let source = try PackStore(model: model, integrity: .sha256)
    let serving = try PackStore(model: model)
    let folder = root.appendingPathComponent("store/hot")
    final class Engine: @unchecked Sendable {
        let lock = NSLock()
        var retired: [URL] = [], events: [String] = []
        let gate = ServingGate()
        var admission = ServingAdmission()
    }
    let engine = Engine()
    // The backend's own gate and admission, with a request admitted (running) before the keeper starts.
    let gate = engine.gate
    let request = try gate.arrive { try gate.lock.withLock { try engine.admission.begin() } }
    let padded = UInt64((PackStore.expertBytes + 16383) / 16384 * 16384)
    // Weighted misses lead: keys 2, 0 and 3 fill a three-record copy; key 1 is used most but missed least.
    let counts = (uses: [1: 900, 2: 5, 0: 4, 3: 3], misses: [2: 40, 0: 30, 3: 20, 1: 1])
    try qwenHotRequire(RoutingProfile.hotCopyChoice(counts, budget: 3 * padded, record: PackStore.expertBytes) == [2, 0, 3], "ranking")
    // The counts a reply hands the keeper: none before the first reply, then the profile's nonzero ones.
    let profile = RoutingProfile()
    try qwenHotRequire(profile.hotCopyCounts.uses.isEmpty && profile.hotCopyCounts.misses.isEmpty, "counts before any reply")
    profile.record(layer: 0, experts: [2, 3])
    profile.recordMiss(layer: 0, expert: 2, weight: 4)
    try qwenHotRequire(profile.hotCopyCounts.uses == [2: 1, 3: 1] && profile.hotCopyCounts.misses == [2: 4], "counts after a reply")
    let keeper = HotCopyKeeper(
        source: source, identity: "qwen m", copy: nil, counts: ([:], [:]),
        choose: { counts, budget in RoutingProfile.hotCopyChoice(counts, budget: budget, record: PackStore.expertBytes) },
        place: { _ in .folder(folder, drive: "fixture", budget: 3 * padded, free: .max) },
        attach: { copy in gate.whileIdle({ engine.admission.isIdle }) { serving.attachHotCopy(copy) } },
        idle: { gate.idle { engine.admission.isIdle } }, withIdle: { work in try gate.whileIdle({ engine.admission.isIdle }, work) },
        retire: { folder in engine.lock.withLock { engine.retired.append(folder) } },
        event: { line in engine.lock.withLock { engine.events.append(line) } })
    defer { keeper.stop(waiting: 3) }
    keeper.trigger()
    func waitFor(_ what: String, _ condition: () -> Bool) throws {
        let end = Date().addingTimeInterval(20)
        while !condition() {
            guard Date() < end else { throw EngineError("Qwen hot copy keeper: \(what) (\(keeper.report))") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }
    try waitFor("learning without counts") { keeper.report["state"] as? String == "learning" }
    keeper.trigger(counts: counts)
    // A request holds the engine: the build waits and nothing is attached.
    try waitFor("paused during a request") { keeper.report["state"] as? String == "paused" }
    try qwenHotRequire(
        serving.attachedHotCopy == nil && !FileManager.default.fileExists(atPath: folder.appendingPathComponent("index.json").path),
        "built or attached during a request")
    gate.lock.withLock { engine.admission.retire(request) {} }
    try waitFor("ready and attached") { keeper.report["state"] as? String == "ready" && serving.attachedHotCopy != nil }
    let copy = serving.attachedHotCopy!
    try qwenHotRequire(copy.keys == [2, 0, 3] && engine.lock.withLock({ engine.retired }) == [folder], "built \(copy.keys)")
    for expert in [2, 0, 3] {
        try qwenHotRequire(serving.hasHotExpert(try ExpertKey(layer: 0, expert: expert)), "expert \(expert) not offered from the copy")
        try qwenHotRequire(try qwenRead(serving, expert) == qwenRecord(expert), "expert \(expert) bytes from the copy")
    }
    try qwenHotRequire(serving.status().hotCopyReads == 3, "reads from the copy \(serving.status().hotCopyReads)")
    // A corrupted record in a copy opened afresh (a restart): its first read fails the CRC32 and the spans serve it.
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.seek(toOffset: UInt64(copy.offsets[0]!) + 7)
    try handle.write(contentsOf: Data([0x5A ^ qwenRecord(0)[7]]))
    try handle.close()
    let restarted = try PackStore(model: model)
    restarted.attachHotCopy(HotCopy.open(folder: folder, identity: "qwen m", source: source))
    try qwenHotRequire(try qwenRead(restarted, 0) == qwenRecord(0) && restarted.status().hotCopyReads == 0, "a corrupted record")
    try qwenHotRequire(try qwenRead(restarted, 2) == qwenRecord(2) && restarted.status().hotCopyReads == 1, "an intact record")
}
