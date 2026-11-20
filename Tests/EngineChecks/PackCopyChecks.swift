import CryptoKit
import Darwin
import Foundation
import OrigamiEngine
import OrigamiEngineChecks

private func copyRequire(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
private func copyDigest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private let copyFingerprint = String(repeating: "c", count: 64)
private var copyStride: Int { (PackStore.expertBytes + 4095) / 4096 * 4096 }
private func copyPayload() -> Data { Data((0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) }) }
private func copyRoot(_ prefix: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("\(prefix)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
/// A sparse file of `length` bytes (only the records written take space).
private func sparseFile(_ url: URL, length: Int) throws -> FileHandle {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: UInt64(length))
    return handle
}
/// A complete pack in `root/name`: every record zero except `held`, which hold `copyPayload()`.
private func copyPack(_ root: URL, _ name: String, held: [Int] = [1, 513]) throws -> URL {
    let folder = root.appendingPathComponent(name)
    let count = ModelContract.layers * ModelContract.experts
    let payload = copyPayload()
    let zero = copyDigest(Data(count: PackStore.expertBytes))
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let experts = try sparseFile(folder.appendingPathComponent("experts.bin"), length: count * copyStride)
    for key in held {
        try experts.seek(toOffset: UInt64(key * copyStride))
        try experts.write(contentsOf: payload)
    }
    try experts.close()
    try sparseFile(folder.appendingPathComponent("ple.bin"), length: ModelContract.pleRows * 100).close()
    try StoreHeader(
        kind: .pack, modelFingerprint: copyFingerprint, quantization: .affine4Group64, expertFile: "experts.bin",
        recordBytes: PackStore.expertBytes, stride: copyStride,
        records: (0..<count).map { .init(key: $0, offset: Int64($0 * copyStride), sha256: held.contains($0) ? copyDigest(payload) : zero) },
        pleFile: "ple.bin", pleRowBytes: 100, pleRowCount: ModelContract.pleRows, created: "fixture"
    ).write(to: folder)
    return folder
}
private func damage(_ pack: URL, key: Int) throws {
    let handle = try FileHandle(forWritingTo: pack.appendingPathComponent("experts.bin"))
    try handle.seek(toOffset: UInt64(key * copyStride))
    try handle.write(contentsOf: Data(repeating: 0xEE, count: 64))
    try handle.close()
}

/// Several expert copies: initial ordinary reads probe each copy, concurrent reads remain exact, a damaged
/// record moves to another copy while its copy keeps serving the rest, a damaged copy is dropped while the others keep
/// serving, and the plan scales the packed queue and its rates with the drives.
func packCopiesSpreadAndFallBack() throws {
    try copyRequire(
        PackStore.leastBusy([0, 0, 0]) == 0 && PackStore.leastBusy([nil, 0, 0]) == 1 && PackStore.leastBusy([2, 1, nil]) == 1
            && PackStore.leastBusy([1, 3, 0]) == 2 && PackStore.leastBusy([nil, nil]) == nil, "Least-busy choice")

    let root = try copyRoot("origami-copies")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:], fingerprint: copyFingerprint)
    let bytes = PackStore.expertBytes
    let count = ModelContract.layers * ModelContract.experts
    let payload = copyPayload()
    let held = [1, 513]
    // Two complete packs and a partial copy of keys 1 and 513, every copy with the same bytes for a key.
    let first = try copyPack(root, "first")
    let second = try copyPack(root, "second")
    let partial = root.appendingPathComponent("partial")
    try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
    try (payload + payload).write(to: partial.appendingPathComponent("mirror.bin"))
    try StoreHeader(
        kind: .mirror, modelFingerprint: copyFingerprint, quantization: .affine4Group64, expertFile: "mirror.bin", recordBytes: bytes,
        stride: bytes, records: held.enumerated().map { .init(key: $1, offset: Int64($0 * bytes), sha256: copyDigest(payload)) },
        created: "fixture"
    ).write(to: partial)
    // Key 1 of the first copy is damaged at rest (a rewrite while serving is caught by the file identity instead).
    try damage(first, key: 1)

    let logged = LockedLines()
    let store = try PackStore(model: model, pack: first, copies: [second, partial], integrity: .sha256, log: { logged.add($0) })
    func reads() -> [UInt64] { store.status().copies?.map(\.reads) ?? [] }
    @Sendable func read(_ key: Int) throws -> Data {
        var output = Data(count: bytes)
        try output.withUnsafeMutableBytes { try store.readExpert(layer: key / 512, expert: key % 512, into: $0, source: .packed) }
        return output
    }
    try copyRequire(
        store.packedAvailable && store.status().copies?.map(\.experts) == [count, count, 2],
        "Copies opened: \(String(describing: store.status().copies))")
    for _ in 0..<3 { try copyRequire(try read(513) == payload, "Record from the first copy") }
    try copyRequire(reads() == [1, 1, 1], "Initial ordinary reads must probe each eligible copy: \(reads())")
    // Many readers at once: every record exact, all of them counted.
    let failures = LockedLines()
    DispatchQueue.concurrentPerform(iterations: 8) { _ in
        for _ in 0..<6 { do { if try read(513) != payload { failures.add("bytes") } } catch { failures.add("\(error)") } }
    }
    try copyRequire(failures.all.isEmpty && reads().reduce(0, +) == 51, "Concurrent reads: \(failures.all), \(reads())")

    // Fresh placement history makes the first corrupt read deterministic without assuming that
    // observed service-time scheduling always prefers a particular drive after warm-up.
    let fallback = try PackStore(model: model, pack: first, copies: [second, partial], integrity: .sha256, log: { logged.add($0) })
    func fallbackReads() -> [UInt64] { fallback.status().copies?.map(\.reads) ?? [] }
    func fallbackRead(_ key: Int) throws -> Data {
        var output = Data(count: bytes)
        try output.withUnsafeMutableBytes { try fallback.readExpert(layer: key / 512, expert: key % 512, into: $0, source: .packed) }
        return output
    }
    try copyRequire(try fallbackRead(1) == payload && fallbackReads() == [0, 1, 0], "Corrupt primary did not fail over")
    for _ in 0..<3 { try copyRequire(try fallbackRead(513) == payload, "Good record lost after per-record quarantine") }
    try copyRequire(fallbackReads()[0] > 0, "The first copy stopped serving its good records")
    let successfulPrimaryReads = fallbackReads()[0]
    for _ in 0..<3 { try copyRequire(try fallbackRead(1) == payload, "Quarantined record did not use an exact alternate") }
    try copyRequire(
        fallbackReads()[0] == successfulPrimaryReads && fallback.packedAvailable
            && fallback.status().copies?[0].problem == nil && fallback.status().degradedReason == nil
            && logged.all.contains { $0.contains("expert 1 failed its checksum") }, "Record quarantine affected the whole copy")
    // Live shrink detection is independent of which healthy copy wins a measured placement.
    try FileHandle(forWritingTo: second.appendingPathComponent("experts.bin")).truncate(atOffset: 4096)
    for _ in 0..<32 {
        try copyRequire(try fallbackRead(1) == payload, "Alternate failed after a live pack shrink")
        if fallback.status().copies?[1].problem != nil { break }
    }
    try copyRequire(
        fallback.status().copies?[1].problem != nil && fallbackReads()[2] > 0
            && logged.all.contains { $0.contains("serving from the other copies") }, "Shrunk copy was not dropped")
    try FileHandle(forWritingTo: first.appendingPathComponent("experts.bin")).truncate(atOffset: 4096)
    for _ in 0..<32 {
        try copyRequire(try fallbackRead(513) == payload, "Partial copy failed after both packs shrank")
        if !fallback.packedAvailable { break }
    }
    try copyRequire(!fallback.packedAvailable && fallback.status().degradedReason != nil, "Only a partial copy is no complete pack")
    var refused = false
    do { _ = try fallbackRead(0) } catch { refused = true }
    try copyRequire(refused, "A key no copy holds must fall back to the model drive")

    // The plan: two drives double the packed queue and its rate from depth 2 on, the model drive's rates unchanged.
    let gib: UInt64 = 1 << 30
    let host = OrigamiHost(physicalMemory: 32 * gib, workingSet: 32 * gib * 3 / 4, gpuName: "simulated 32 GB", bfloatMatrices: true)
    func plan(_ drives: Int) throws -> OrigamiPlan {
        var storage = OrigamiStorage(packFolder: first)
        storage.copyFolders = drives > 1 ? [second] : []
        storage.packDrives = drives
        return try OrigamiPlan.make(
            model: model, host: host, context: 8192, memoryLimit: nil, storage: storage, promptCache: root,
            availableNow: 32 * gib, footprintNow: 0)
    }
    let one = try plan(1).configuration.experts
    let two = try plan(2).configuration.experts
    let rate = one.sourceCalibration.externalBytesPerSecond
    let spread = two.sourceCalibration.externalBytesPerSecond
    try copyRequire(one.packDrives == 1 && two.packDrives == 2, "Packed queue drives: \(one.packDrives), \(two.packDrives)")
    // Depths 1, 4 and 8: one read keeps one drive busy; four keep both at depth 2 (read at depth 1's rate); eight at 4.
    try copyRequire(
        two.sourceCalibration.queueDepths == [1, 4, 8] && spread[0] == rate[0] && spread[1] == 2 * rate[0] && spread[2] == 2 * rate[1]
            && two.sourceCalibration.internalBytesPerSecond == one.sourceCalibration.internalBytesPerSecond,
        "Spread rates: \(rate) -> \(spread)")
}

/// Store headers are untrusted files: a short, missing or escaping PLE table, an
/// overflowing stride and an oversized header are each refused (as "rebuild"), never trusted or trapped on.
func storeHeaderRejectsMalformedTables() throws {
    let root = try copyRoot("origami-header")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:], fingerprint: copyFingerprint)
    let pack = try copyPack(root, "pack", held: [])
    guard let header = try StoreHeader.load(.pack, in: pack) else { throw EngineError("Fixture pack header missing") }
    try copyRequire(header.mismatch(for: model, folder: pack) == nil, "Fixture pack is valid")
    func variant(stride: Int? = nil, pleFile: String?) -> StoreHeader {
        StoreHeader(
            kind: .pack, modelFingerprint: copyFingerprint, quantization: .affine4Group64, expertFile: "experts.bin",
            recordBytes: header.recordBytes, stride: stride ?? header.stride, records: header.records, pleFile: pleFile,
            pleRowBytes: 100, pleRowCount: ModelContract.pleRows, created: "fixture")
    }
    try copyRequire(
        variant(stride: 1 << 62, pleFile: "ple.bin").mismatch(for: model, folder: pack) == "record layout differs",
        "Overflowing stride")
    try sparseFile(root.appendingPathComponent("ple.bin"), length: ModelContract.pleRows * 100).close()
    try copyRequire(
        variant(pleFile: "../ple.bin").mismatch(for: model, folder: pack) == "PLE table is missing or short",
        "PLE table outside the store folder")
    let ple = pack.appendingPathComponent("ple.bin")
    try FileHandle(forWritingTo: ple).truncate(atOffset: 100)
    try copyRequire(header.mismatch(for: model, folder: pack) == "PLE table is missing or short", "Short PLE table")
    try FileManager.default.removeItem(at: ple)
    try copyRequire(header.mismatch(for: model, folder: pack) == "PLE table is missing or short", "Missing PLE table")

    let large = root.appendingPathComponent("large")
    try FileManager.default.createDirectory(at: large, withIntermediateDirectories: true)
    try sparseFile(large.appendingPathComponent(StoreHeader.fileName(.mirror)), length: 65 << 20).close()
    var refused = false
    do { _ = try StoreHeader.load(.mirror, in: large) } catch { refused = true }
    try copyRequire(refused, "Oversized store header was accepted")
}

/// Store builds: stale partials are removed, a planted link at the temporary name is replaced (never written
/// through), and a mirror built from a pack refuses a damaged record (never copied under a fresh digest),
/// leaving no partial or header behind.
func storeBuilderLeavesNoPartialsAndCopiesOnlyVerifiedRecords() throws {
    let root = try copyRoot("origami-builder")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:], fingerprint: copyFingerprint)
    let pack = try copyPack(root, "pack")
    func contents(_ folder: URL) -> [String] { ((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []).sorted() }

    let stale = root.appendingPathComponent("stale")
    try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
    for name in ["experts.bin.partial", "ple.bin.partial", "pack.json"] { try Data([1]).write(to: stale.appendingPathComponent(name)) }
    StoreBuilder.removePartials(in: stale)
    try copyRequire(contents(stale) == ["pack.json"], "Stale partials: \(contents(stale))")

    let mirror = root.appendingPathComponent("mirror")
    let victim = root.appendingPathComponent("victim")
    try FileManager.default.createDirectory(at: mirror, withIntermediateDirectories: true)
    try Data("keep".utf8).write(to: victim)
    try FileManager.default.createSymbolicLink(at: mirror.appendingPathComponent("mirror.bin.partial"), withDestinationURL: victim)
    try LegacyMirrorFixture.build(model: model, folder: mirror, records: 1, from: pack, ranking: [513])
    try copyRequire(try Data(contentsOf: victim) == Data("keep".utf8), "The build wrote through a planted link")
    try copyRequire(
        try Data(contentsOf: mirror.appendingPathComponent("mirror.bin")) == copyPayload()
            && StoreHeader.load(.mirror, in: mirror)?.mismatch(for: model, folder: mirror) == nil
            && contents(mirror) == ["mirror.bin", "mirror.json"], "Mirror built beside a planted link: \(contents(mirror))")

    // The damaged record is the ninth read (a worker's second), which a sampled check would skip.
    try damage(pack, key: 1)
    let bad = root.appendingPathComponent("bad")
    var refused = false
    do { try LegacyMirrorFixture.build(model: model, folder: bad, records: 9, from: pack, ranking: [0, 2, 3, 4, 5, 6, 7, 8, 1]) } catch {
        refused = true
    }
    try copyRequire(refused && contents(bad).isEmpty, "A damaged pack record was copied, or the build left \(contents(bad))")

    // A refused rebuild over a finished mirror keeps its header and payload; the header goes only at the payload swap.
    refused = false
    do { try LegacyMirrorFixture.build(model: model, folder: mirror, records: 9, from: pack, ranking: [0, 2, 3, 4, 5, 6, 7, 8, 1]) } catch {
        refused = true
    }
    try copyRequire(
        refused && StoreHeader.load(.mirror, in: mirror)?.mismatch(for: model, folder: mirror) == nil
            && contents(mirror) == ["mirror.bin", "mirror.json"], "A refused rebuild damaged the finished mirror: \(contents(mirror))")

    // Setup's further copy (a `HotCopy` store) reads the pack the same way: the damaged record is not copied, here with
    // no checkpoint spans to read it from instead, so the build refuses and a finished copy keeps its index and records.
    let source = try PackStore(model: model, pack: pack, integrity: .sha256)
    let further = root.appendingPathComponent("further/copy")
    _ = try HotCopy.write(source: source, keys: [513], folder: further, identity: "q")
    refused = false
    do { _ = try HotCopy.write(source: source, keys: [0, 2, 3, 4, 5, 6, 7, 8, 1], folder: further, identity: "q") } catch {
        refused = true
    }
    try copyRequire(
        refused && HotCopy.open(folder: further, identity: "q", source: source)?.keys == [513],
        "A further copy took a damaged pack record, or a refused build changed it")
}

/// A pack whose header appears after start is picked up at a batch boundary; the reopen runs off the calling thread.
func packCopyRevivesInTheBackground() throws {
    let root = try copyRoot("origami-revive")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try ModelSource(fixture: root, spans: [:], fingerprint: copyFingerprint)
    let pack = try copyPack(root, "pack")
    let header = pack.appendingPathComponent(StoreHeader.fileName(.pack))
    let aside = pack.appendingPathComponent("header.aside")
    try FileManager.default.moveItem(at: header, to: aside)
    let logged = LockedLines()
    let store = try PackStore(model: model, pack: pack, log: { logged.add($0) })
    try copyRequire(!store.packedAvailable, "Pack without a header was opened")
    try FileManager.default.moveItem(at: aside, to: header)
    try store.validateBatchBoundary()
    let deadline = Date(timeIntervalSinceNow: 10)
    while !store.packedAvailable && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
    try copyRequire(
        store.packedAvailable && logged.all.contains { $0.contains("is back in use") }, "Returned pack not revived: \(logged.all)")
}

private final class LockedLines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func add(_ line: String) {
        lock.lock()
        lines.append(line)
        lock.unlock()
    }
    var all: [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}
