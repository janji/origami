import Darwin
import Foundation
import Metal
import OrigamiEngine
import zlib

func hotCopyRefreshChecks() -> [(String, () throws -> Void)] {
    [
        ("hot_copy_refresh_plan_headroom", hotCopyRefreshPlanHeadroom),
        ("hot_copy_refresh_preserves_partial_capacity", hotCopyRefreshPreservesPartialCapacity),
        ("hot_copy_refresh_publication_failure_preserves_binding", hotCopyRefreshPublicationFailurePreservesBinding),
        ("hot_copy_refresh_reuses_bytes_and_pins_readers", hotCopyRefreshReusesBytesAndPinsReaders),
        ("hot_copy_refresh_interruption_preserves_manifest", hotCopyRefreshInterruptionPreservesManifest),
        ("hot_copy_refresh_reserve_fails_closed", hotCopyRefreshReserveFailsClosed),
        ("hot_copy_refresh_recovers_orphans", hotCopyRefreshRecoversOrphans),
        ("hot_copy_refresh_rejects_stale_append", hotCopyRefreshRejectsStaleAppend),
        ("hot_copy_refresh_serializes_writers", hotCopyRefreshSerializesWriters),
        ("hot_copy_rejects_invalid_record_layout", hotCopyRejectsInvalidRecordLayout),
        ("hot_copy_count_metadata", hotCopyCountMetadata),
        ("hot_copy_opens_3_0_1_layout", hotCopyOpens301Layout),
        ("hot_copy_read_error_disables_then_revives", hotCopyReadErrorDisablesThenRevives),
        ("hot_copy_keeper_keeps_working_copy", hotCopyKeeperKeepsWorkingCopy),
    ]
}

private func hotCopyRejectsInvalidRecordLayout() throws {
    try withRefreshFixture { experts, folder in
        let copy = try HotCopy.write(source: experts, keys: [4, 5, 9], folder: folder, identity: "fixture")
        let url = folder.appendingPathComponent("index.json")
        let original = try Data(contentsOf: url)
        let index = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        let offsets = copy.keys.map { copy.offsets[$0]! }
        var variants: [[String: Any]] = []
        var overlap = index
        overlap["offsets"] = [offsets[0], offsets[1] - 16384, offsets[2]]
        variants.append(overlap)
        var unaligned = index
        unaligned["offsets"] = [offsets[0], offsets[1] - 1, offsets[2]]
        variants.append(unaligned)
        var gap = index
        gap["offsets"] = [offsets[0], offsets[1] + 16384, offsets[2]]
        variants.append(gap)
        var extent = index
        extent["byteCount"] = copy.bytes - 1
        variants.append(extent)
        var unknown = index
        unknown["keys"] = [4, 5, Int.max]
        variants.append(unknown)
        for (n, variant) in variants.enumerated() {
            try JSONSerialization.data(withJSONObject: variant).write(to: url, options: .atomic)
            try refreshRequire(
                HotCopy.open(folder: folder, identity: "fixture", source: experts) == nil,
                "invalid record layout accepted: \(n)")
        }
        try original.write(to: url, options: .atomic)
        // A crash after an append may leave extra file bytes; the previous published prefix stays valid.
        let handle = try FileHandle(forWritingTo: copy.payload)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(repeating: 0x5A, count: 16384))
        try handle.close()
        guard let reopened = HotCopy.open(folder: folder, identity: "fixture", source: experts) else {
            throw EngineError("Hot-copy metadata rejected a valid published prefix")
        }
        for key in copy.keys { try compareRefreshRecord(reopened, key: key, experts: experts) }
    }
}

private func hotCopyCountMetadata() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-counts-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("uses.json")
    let object = ["1": 7, "01": 9, "+1": 11, "-1": 13, "invalid": 15, "2": 0, "3": -1, "4": Int.max]
    try JSONSerialization.data(withJSONObject: object).write(to: file)
    let read = HotCopy.loadUses(file)
    try refreshRequire(read == [1: 7, 4: Int.max], "canonical count parsing changed or admitted aliases")
    let merged = HotCopy.mergeUses(read, [1: 3, 4: 1, 5: 2])
    try refreshRequire(merged == [1: 10, 4: Int.max, 5: 2], "cumulative count overflow was not saturated")
    try HotCopy.saveUses(merged, to: file)
    try refreshRequire(HotCopy.loadUses(file) == merged, "valid counts did not round-trip")
}

private func refreshRequire(_ ok: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !ok() { throw EngineError("Hot-copy refresh: " + message) }
}

private func hotCopyRefreshPlanHeadroom() throws {
    try refreshRequire(HotCopyRefreshPlan.room(free: UInt64.max, reserve: UInt64.max) == 0, "reserve arithmetic overflowed")
    for invalid in ["../experts.bin", "/experts.bin", "experts-generation-no.bin", "experts-generation-../x.bin"] {
        try refreshRequire(!HotCopyRefreshPlan.validPayload(invalid), "manifest accepts unsafe payload name")
    }
}

private func hotCopyRefreshPreservesPartialCapacity() throws {
    let gb: UInt64 = 1_000_000_000
    try refreshRequire(
        HotCopyRefreshPlan.capacity(wanted: 20 * gb, held: 8 * gb, free: 21 * gb, reserve: 20 * gb) == 9 * gb,
        "partial copy rejected while growth can proceed")
    try refreshRequire(
        HotCopyRefreshPlan.capacity(wanted: 20 * gb, held: 8 * gb, free: 19 * gb, reserve: 20 * gb) == 8 * gb,
        "partial copy detached when growth must wait")
    try refreshRequire(
        HotCopyRefreshPlan.capacity(wanted: 20 * gb, held: 0, free: 21 * gb, reserve: 20 * gb) == nil,
        "first-build minimum was lost")
    try refreshRequire(
        HotCopyRefreshPlan.capacity(wanted: 20 * gb, held: 0, free: 40 * gb, reserve: 20 * gb) == 20 * gb,
        "full first build rejected")
    try refreshRequire(
        HotCopyRefreshPlan.capacity(wanted: 4 * gb, held: 8 * gb, free: 19 * gb, reserve: 20 * gb) == 4 * gb,
        "explicit reduced size ignored")
}

private func hotCopyRefreshPublicationFailurePreservesBinding() throws {
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let indexURL = folder.appendingPathComponent("index.json")
        let index = try Data(contentsOf: folder.appendingPathComponent("index.json"))
        guard chflags(indexURL.path, UInt32(UF_IMMUTABLE)) == 0 else {
            throw FixtureUnavailable(description: "cannot set fixture immutable flag")
        }
        defer { _ = chflags(indexURL.path, 0) }
        var failed = false
        do { _ = try HotCopy.write(source: experts, keys: [5], folder: folder, identity: "fixture") } catch { failed = true }
        try refreshRequire(failed, "immutable manifest did not stop publication")
        try refreshRequire(try Data(contentsOf: indexURL) == index, "failed publication changed old manifest")
        guard let reopened = HotCopy.open(folder: folder, identity: "fixture", source: experts) else {
            throw EngineError("Failed publication lost reusable old copy")
        }
        try refreshRequire(reopened.keys == [4] && reopened.payload == old.payload, "old keys bound to replacement payload")
        try compareRefreshRecord(reopened, key: 4, experts: experts)
        try compareRefreshRecord(old, key: 4, experts: experts)
    }
}

/// Tiny fake record payloads and empty shared pools only; no model or GPU kernel execution.
private func withRefreshFixture(_ body: (FamilyExperts, URL) throws -> Void) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-refresh-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: root)), device: device)
    let pool = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    try body(pool, root.appendingPathComponent("hot"))
}

private func compareRefreshRecord(_ copy: HotCopy, key: Int, experts: FamilyExperts) throws {
    let count = experts.bank(key / 4)!.bytes.reduce(0, +)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 16384)
    defer { buffer.deallocate() }
    try experts.readRecord(layer: key / 4, expert: key % 4, into: buffer)
    let bytes = try copy.file.read(offset: copy.offsets[key]!, length: count)
    try refreshRequire(bytes.withUnsafeBytes { memcmp(buffer, $0.baseAddress!, count) == 0 }, "record \(key) content changed")
}

private func hotCopyRefreshReusesBytesAndPinsReaders() throws {
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4, 5, 9], folder: folder, identity: "fixture")
        let oldBytes = try old.file.read(offset: 0, length: Int(old.bytes))
        var rejectedIdentity = false
        do {
            _ = try HotCopy.writeIncremental(
                source: experts, keys: [4], folder: folder, identity: "another-representation", reuse: old,
                reserve: 2048, freeBytes: { 4_000_000 })
        } catch { rejectedIdentity = true }
        try refreshRequire(rejectedIdentity, "cache bytes from another representation were reused")
        var sourceReads: [Int] = []
        let next = try HotCopy.writeIncremental(
            source: experts, keys: [5, 9, 6], folder: folder, identity: "fixture", reuse: old, targetBytes: 1_000_000,
            reserve: 2048, freeBytes: { 4_000_000 },
            readMissing: { key, buffer in
                sourceReads.append(key)
                try experts.readRecord(layer: key / 4, expert: key % 4, into: buffer)
            })
        try refreshRequire(sourceReads == [6], "unchanged records were reread from source")
        try refreshRequire(
            next.payload != old.payload && !FileManager.default.fileExists(atPath: old.payload.path), "old generation not retired")
        try refreshRequire(try old.file.read(offset: 0, length: Int(old.bytes)) == oldBytes, "old descriptor changed after unlink")
        for key in next.keys { try compareRefreshRecord(next, key: key, experts: experts) }
        try refreshRequire(
            HotCopy.open(folder: folder, identity: "fixture", source: experts)?.keys == next.keys, "named generation does not reopen"
        )
        try refreshRequire(
            HotCopy.open(folder: folder, identity: "fixture")?.targetBytes == 1_000_000,
            "interrupted migration lost automatic capacity target")
        let before = try next.file.read(offset: 0, length: Int(next.bytes))
        let grown = try HotCopy.writeIncremental(
            source: experts, keys: [10], folder: folder, identity: "fixture", onto: next,
            reserve: 2048, freeBytes: { 4_000_000 })
        try refreshRequire(grown.payload == next.payload && grown.bytes == next.bytes + 114_688, "top-up copied full generation")
        try refreshRequire(try next.file.read(offset: 0, length: Int(next.bytes)) == before, "append changed published prefix")
        for key in grown.keys { try compareRefreshRecord(grown, key: key, experts: experts) }
        try refreshRequire(grown.targetBytes == 1_000_000, "top-up lost migration target")
    }
}

private func hotCopyRefreshInterruptionPreservesManifest() throws {
    struct Stop: Error {}
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let index = try Data(contentsOf: folder.appendingPathComponent("index.json"))
        for append in [false, true] {
            var waits = 0
            var writes = 0
            var stopped = false
            do {
                _ = try HotCopy.writeIncremental(
                    source: experts, keys: [5, 9], folder: folder, identity: "fixture",
                    onto: append ? old : nil, reuse: old, reserve: 2048, freeBytes: { 4_000_000 },
                    wait: {
                        waits += 1
                        if waits == 3 { throw Stop() }
                    }, wrote: { _ in writes += 1 })
            } catch is Stop { stopped = true }
            try refreshRequire(stopped && writes == 1, "interruption fixture failed to write exactly one unpublished record")
            try refreshRequire(try Data(contentsOf: folder.appendingPathComponent("index.json")) == index, "interruption changed manifest")
            try refreshRequire(
                HotCopy.open(folder: folder, identity: "fixture", source: experts)?.keys == [4],
                "published cache lost after interruption")
            try refreshRequire(
                (try FileManager.default.attributesOfItem(atPath: old.payload.path)[.size] as? NSNumber)?.int64Value == old.bytes,
                "unpublished append was not truncated")
            try compareRefreshRecord(old, key: 4, experts: experts)
        }
    }
}

private func hotCopyRefreshReserveFailsClosed() throws {
    struct Unknown: Error {}
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let index = try Data(contentsOf: folder.appendingPathComponent("index.json"))
        let record: UInt64 = 212_992
        let reserve: UInt64 = 2048
        // Unknown space, insufficient space, a stale repeated reading, and an external loss of
        // free space all fail before the next allocation. The old manifest remains exact.
        for mode in 0..<4 {
            var measurements = 0
            var writes = 0
            var failed = false
            do {
                _ = try HotCopy.writeIncremental(
                    source: experts, keys: [5, 6], folder: folder, identity: "fixture", reuse: old,
                    reserve: reserve,
                    freeBytes: {
                        measurements += 1
                        if mode == 0 { throw Unknown() }
                        if mode == 1 { return reserve }
                        if mode == 2 { return reserve + record }
                        return measurements >= 4 ? reserve + record - 1 : 4_000_000
                    }, wrote: { _ in writes += 1 })
            } catch { failed = true }
            try refreshRequire(failed && writes <= 1, "reserve failure did not bound allocated records")
            try refreshRequire(
                try Data(contentsOf: folder.appendingPathComponent("index.json")) == index, "reserve failure changed manifest")
        }
    }
}

private func hotCopyRefreshRecoversOrphans() throws {
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let orphan = folder.appendingPathComponent("experts-generation-\(UUID().uuidString).bin")
        try Data(repeating: 7, count: 16384).write(to: orphan)
        try refreshRequire(HotCopy.open(folder: folder, identity: "fixture")?.keys == [4], "orphan affected published manifest")
        try HotCopy.recoverIncremental(folder: folder, identity: "fixture")
        try refreshRequire(!FileManager.default.fileExists(atPath: orphan.path), "pre-planning recovery left interrupted allocation")
        // The writer also recovers orphans created before its lock acquisition.
        try Data(repeating: 7, count: 16384).write(to: orphan)
        let next = try HotCopy.writeIncremental(
            source: experts, keys: [4, 5], folder: folder, identity: "fixture", reuse: old,
            reserve: 2048,
            freeBytes: {
                FileManager.default.fileExists(atPath: orphan.path) ? 2047 : 4_000_000
            })
        try refreshRequire(!FileManager.default.fileExists(atPath: orphan.path), "interrupted generation not reclaimed")
        try refreshRequire(
            HotCopy.open(folder: folder, identity: "fixture", source: experts)?.keys == next.keys,
            "replacement publication not recoverable")
        var malformed =
            try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("index.json"))) as! [String: Any]
        malformed["payload"] = "../experts.bin"
        try JSONSerialization.data(withJSONObject: malformed).write(to: folder.appendingPathComponent("index.json"))
        try refreshRequire(HotCopy.open(folder: folder, identity: "fixture") == nil, "unsafe manifest path opened")
        try Data([7]).write(to: orphan)
        try HotCopy.recoverIncremental(folder: folder, identity: "fixture")
        try refreshRequire(FileManager.default.fileExists(atPath: orphan.path), "recovery changed invalid-manifest evidence")
        // A first build killed before any manifest can reclaim its generated payloads too.
        try FileManager.default.removeItem(at: folder.appendingPathComponent("index.json"))
        try HotCopy.recoverIncremental(folder: folder, identity: "fixture")
        try refreshRequire(
            !FileManager.default.fileExists(atPath: orphan.path) && !FileManager.default.fileExists(atPath: next.payload.path),
            "first-generation interruption cannot reclaim headroom")
        try compareRefreshRecord(next, key: 4, experts: experts)
    }
}

private func hotCopyRefreshRejectsStaleAppend() throws {
    try withRefreshFixture { experts, folder in
        let first = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let grown = try HotCopy.writeIncremental(
            source: experts, keys: [5], folder: folder, identity: "fixture", onto: first,
            reserve: 2048, freeBytes: { 4_000_000 })
        var rejected = false
        do {
            _ = try HotCopy.writeIncremental(
                source: experts, keys: [6], folder: folder, identity: "fixture", onto: first,
                reserve: 2048, freeBytes: { 4_000_000 })
        } catch { rejected = true }
        try refreshRequire(rejected, "stale append could overwrite a newly published record")
        rejected = false
        do { _ = try HotCopy.write(source: experts, keys: [6], folder: folder, identity: "fixture", onto: first) } catch {
            rejected = true
        }
        try refreshRequire(rejected, "legacy append bypassed stale-manifest guard")
        try refreshRequire(HotCopy.open(folder: folder, identity: "fixture")?.keys == [4, 5], "stale append changed publication")
        try compareRefreshRecord(grown, key: 5, experts: experts)
    }
}

private func hotCopyRefreshSerializesWriters() throws {
    try withRefreshFixture { experts, folder in
        let old = try HotCopy.write(source: experts, keys: [4], folder: folder, identity: "fixture")
        let fd = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
        guard fd >= 0, flock(fd, LOCK_EX | LOCK_NB) == 0 else { throw EngineError("Cannot lock refresh test folder") }
        defer {
            _ = flock(fd, LOCK_UN)
            close(fd)
        }
        let otherPartial = folder.appendingPathComponent("experts.next")
        try Data([11, 12, 13]).write(to: otherPartial)
        var rejected = false
        do { _ = try HotCopy.write(source: experts, keys: [5], folder: folder, identity: "fixture") } catch { rejected = true }
        let remainingPartial = try Data(contentsOf: otherPartial)
        try refreshRequire(
            rejected && remainingPartial == Data([11, 12, 13]), "legacy writer bypassed lock or removed another writer's partial")
        rejected = false
        do {
            _ = try HotCopy.writeIncremental(
                source: experts, keys: [5], folder: folder, identity: "fixture", reuse: old,
                reserve: 2048, freeBytes: { 4_000_000 })
        } catch { rejected = true }
        try refreshRequire(
            rejected && HotCopy.open(folder: folder, identity: "fixture")?.keys == [4], "second writer bypassed ownership")
    }
}

/// Model-free records: key k is `sizes[k]` bytes of a pattern only k has.
private final class PatternRecords: HotCopySource {
    let sizes: [Int: Int]
    init(_ sizes: [Int: Int]) { self.sizes = sizes }
    static func byte(_ key: Int, _ at: Int) -> UInt8 { UInt8(truncatingIfNeeded: key &* 31 &+ at &* 7) }
    func hotCopyRecordBytes(_ key: Int) -> Int? { sizes[key] }
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
        for at in 0..<sizes[key]! { into.storeBytes(of: Self.byte(key, at), toByteOffset: at, as: UInt8.self) }
    }
    func matches(_ key: Int, _ bytes: UnsafeRawBufferPointer) -> Bool { bytes.indices.allSatisfy { bytes[$0] == Self.byte(key, $0) } }
}

/// A copy as Origami 3.0.1 wrote it (index.json with identity, keys, offsets, byteCount, payload, targetBytes and per-key
/// CRC32, a null for a record written before checksums; records padded to 16 KiB in experts-generation-<UUID>.bin), and
/// one from before checksums (experts.bin, identity, keys and offsets only), written here byte by byte, open and read
/// back exactly; a record whose bytes differ from its CRC32 is rejected alone.
private func hotCopyOpens301Layout() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-301-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let source = PatternRecords([3: 20_000, 7: 5_000, 12: 16_384])
    let keys = [7, 3, 12]
    func layout(_ folder: URL, payload: String, index: (_ offsets: [Int], _ sums: [UInt32], _ end: Int) -> [String: Any]) throws {
        var data = Data()
        var offsets: [Int] = []
        var sums: [UInt32] = []
        for key in keys {
            offsets.append(data.count)
            var record = Data(count: source.sizes[key]!)
            record.withUnsafeMutableBytes { try? source.readHotCopyRecord(key, into: $0.baseAddress!) }
            sums.append(record.withUnsafeBytes { UInt32(crc32(0, $0.bindMemory(to: Bytef.self).baseAddress, uInt($0.count))) })
            data += record + Data(count: (16384 - record.count % 16384) % 16384)
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try data.write(to: folder.appendingPathComponent(payload))
        try JSONSerialization.data(withJSONObject: index(offsets, sums, data.count)).write(to: folder.appendingPathComponent("index.json"))
    }
    let current = root.appendingPathComponent("current")
    let payload = "experts-generation-\(UUID().uuidString).bin"
    try layout(current, payload: payload) { offsets, sums, end in
        [
            "identity": "m", "keys": keys, "offsets": offsets, "byteCount": end, "payload": payload, "targetBytes": 1_000_000,
            "checksums": [sums[0], sums[1], NSNull()],
        ]
    }
    let legacy = root.appendingPathComponent("legacy")
    try layout(legacy, payload: "experts.bin") { offsets, _, _ in ["identity": "m", "keys": keys, "offsets": offsets] }
    for folder in [current, legacy] {
        guard let copy = HotCopy.open(folder: folder, identity: "m", source: source) else {
            throw EngineError("Hot copy 3.0.1 layout: \(folder.lastPathComponent) did not open")
        }
        try refreshRequire(copy.keys == keys && HotCopy.open(folder: folder, identity: "other") == nil, "identity or keys")
        for key in keys {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: source.sizes[key]!, alignment: 16384)
            defer { buffer.deallocate() }
            try copy.read(key, into: buffer)
            try refreshRequire(source.matches(key, UnsafeRawBufferPointer(buffer)), "\(folder.lastPathComponent) record \(key) bytes")
        }
    }
    try refreshRequire(HotCopy.open(folder: current, identity: "m")?.targetBytes == 1_000_000, "target bytes")
    let handle = try FileHandle(forWritingTo: current.appendingPathComponent(payload))
    try handle.seek(toOffset: 16384 + 100)
    try handle.write(contentsOf: Data([0xEE]))
    try handle.close()
    let copy = HotCopy.open(folder: current, identity: "m", source: source)!
    let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 20_000, alignment: 16384)
    defer { buffer.deallocate() }
    // One log line for the rejected record, however often it is asked for.
    let lines = HotCopyLog.capture {
        for _ in 0..<3 { _ = try? copy.read(3, into: buffer) }
    }
    try refreshRequire(
        lines.filter { $0.contains("record 3 in \(copy.payload.path) failed its checksum") }.count == 1 && lines.count == 1,
        "rejection log: \(lines)")
    try refreshRequire((try? copy.read(3, into: buffer)) == nil && copy.rejectedRecords == 1 && !copy.disabled, "corrupt record")
    try copy.read(7, into: UnsafeMutableRawBufferPointer(rebasing: buffer[0..<5_000]))
    try refreshRequire(source.matches(7, UnsafeRawBufferPointer(rebasing: buffer[0..<5_000])), "intact record after a corrupt one")
}

/// A read that fails (the file cut short, as an unplugged drive fails) stops every read from the copy, which then throw
/// without touching the file even once it reads again; after `retryDelay` a read reopens the copy off its thread and
/// reads come from it again with the exact bytes.
private func hotCopyReadErrorDisablesThenRevives() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-revive-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let source = PatternRecords([1: 20_000, 2: 9_000])
    let copy = try HotCopy.write(source: source, keys: [1, 2], folder: root, identity: "m")
    copy.retryDelay = 0.3
    let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 9_000, alignment: 16384)
    defer { buffer.deallocate() }
    try copy.read(2, into: buffer)
    let saved = try Data(contentsOf: copy.payload)
    let handle = try FileHandle(forWritingTo: copy.payload)
    try handle.truncate(atOffset: 0)
    let lines = try HotCopyLog.capture {
        try refreshRequire((try? copy.read(2, into: buffer)) == nil && copy.disabled && copy.rejectedRecords == 0, "read error")
        try handle.write(contentsOf: saved)
        try handle.close()
        try refreshRequire((try? copy.read(2, into: buffer)) == nil && copy.disabled, "read during the pause")
        Thread.sleep(forTimeInterval: 0.4)
        let end = Date().addingTimeInterval(10)
        while (try? copy.read(2, into: buffer)) == nil {
            guard Date() < end else { throw EngineError("Hot copy revive: never read again") }
            Thread.sleep(forTimeInterval: 0.05)
        }
    }
    try refreshRequire(!copy.disabled && source.matches(2, UnsafeRawBufferPointer(buffer)), "bytes after revival")
    // One line when the copy pauses and one when it is read again.
    try refreshRequire(
        lines.count == 2 && lines[0].contains("\(root.path) not read for 0 s after a read error") && lines[1].contains("is read again"),
        "pause and revival log: \(lines)")
}

/// The lines hot copies log (`HotCopy.log`) while `body` runs; the checks run one at a time.
final class HotCopyLog: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func add(_ line: String) { lock.withLock { lines.append(line) } }
    var all: [String] { lock.withLock { lines } }
    static func capture(_ body: () throws -> Void) rethrows -> [String] {
        let log = HotCopyLog()
        let saved = HotCopy.log
        HotCopy.log = { log.add($0) }
        defer { HotCopy.log = saved }
        try body()
        return log.all
    }
}
