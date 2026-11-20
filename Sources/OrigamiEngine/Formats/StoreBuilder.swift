import CryptoKit
import Darwin
import Foundation

/// Builds, checks and measures store payloads. Everything is read from the model's own
/// checkpoint files, which are never written. A build writes temporary files and publishes
/// the header last, so an interrupted build leaves no half-valid store behind.
public enum StoreBuilder {
    public typealias Progress = @Sendable (_ done: Int, _ total: Int) -> Void

    /// Bytes a complete pack needs: every expert record plus the PLE table.
    public static func packBytes(_ model: ModelSource) -> UInt64 {
        UInt64(ModelContract.layers * ModelContract.experts * stride(model)) + UInt64(ModelContract.pleRows * 100)
    }
    public static func mirrorBytes(records: Int, _ model: ModelSource) -> UInt64 { UInt64(records * model.quantization.expertRecordBytes) }
    /// Ranked experts available for a mirror.
    public static var maximumMirrorRecords: Int { HotExpertRanking.keys.count }
    /// Experts most used first (the model's own ranking, else the shipped one), less `skipping`.
    public static func ranked(_ ranking: [Int]?, skipping: Set<Int> = []) -> [Int] {
        (ranking ?? HotExpertRanking.keys).filter { !skipping.contains($0) }
    }
    static func stride(_ model: ModelSource) -> Int { (model.quantization.expertRecordBytes + 4095) / 4096 * 4096 }

    /// Writes a complete pack into `folder` (created if needed).
    public static func buildPack(model: ModelSource, folder: URL, progress: Progress? = nil) throws {
        let keys = Array(0..<(ModelContract.layers * ModelContract.experts))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let store = try PackStore(model: model)
        try store.enableConcurrentOriginalSpans(pool: SpanReadPool(threads: 16))
        let total = keys.count + ModelContract.pleShards
        let records = try writeRecords(
            keys, store: store, to: folder.appendingPathComponent("experts.bin"),
            stride: stride(model), header: folder.appendingPathComponent(StoreHeader.fileName(.pack)),
            progress: { done in progress?(done, total) })
        try writePLE(store: store, to: folder.appendingPathComponent("ple.bin")) { done in progress?(keys.count + done, total) }
        try StoreHeader(
            kind: .pack, modelFingerprint: model.fingerprint, quantization: model.quantization, expertFile: "experts.bin",
            recordBytes: model.quantization.expertRecordBytes, stride: stride(model), records: records,
            pleFile: "ple.bin", pleRowBytes: 100, pleRowCount: ModelContract.pleRows, created: timestamp()
        ).write(to: folder)
    }

    /// Re-reads every record of a store and checks its digest; with `compareModel`, also checks
    /// the bytes against the checkpoint (and the PLE table, for a pack). Returns problems found.
    public static func verify(
        model: ModelSource, folder: URL, kind: StoreHeader.Kind,
        compareModel: Bool = false, progress: Progress? = nil
    ) throws -> [String] {
        guard let header = try StoreHeader.load(kind, in: folder) else {
            return ["no \(StoreHeader.fileName(kind))"]
        }
        if let reason = header.mismatch(for: model, folder: folder) { return [reason] }
        let file = try ExactFile(url: folder.appendingPathComponent(header.expertFile))
        let source = try PackStore(model: model)
        let problems = Problems()
        let counter = Counter()
        let total = header.records.count + (compareModel && kind == .pack ? ModelContract.pleShards : 0)
        DispatchQueue.concurrentPerform(iterations: 8) { worker in
            var buffer = [UInt8](repeating: 0, count: header.recordBytes)
            var reference = buffer
            for index in Swift.stride(from: worker, to: header.records.count, by: 8) {
                let record = header.records[index]
                do {
                    try buffer.withUnsafeMutableBytes { try file.read(offset: record.offset, into: $0) }
                    if hex(SHA256.hash(data: buffer)) != record.sha256 { problems.add("record \(record.key) digest differs") }
                    if compareModel {
                        try reference.withUnsafeMutableBytes {
                            try source.readExpert(layer: record.key / 512, expert: record.key % 512, into: $0, source: .original)
                        }
                        if reference != buffer { problems.add("record \(record.key) differs from the model") }
                    }
                } catch { problems.add("record \(record.key): \(error)") }
                progress?(counter.next(), total)
            }
        }
        if compareModel, kind == .pack, let pleFile = header.pleFile {
            let ple = try ExactFile(url: folder.appendingPathComponent(pleFile))
            for shard in 0..<ModelContract.pleShards {
                for row in [0, ModelContract.pleRowsPerShard / 2, ModelContract.pleRowsPerShard - 1] {
                    let global = shard * ModelContract.pleRowsPerShard + row
                    var packed = [UInt8](repeating: 0, count: 100)
                    var original = packed
                    try packed.withUnsafeMutableBytes { try ple.read(offset: Int64(global) * 100, into: $0) }
                    try original.withUnsafeMutableBytes { try source.readPLERowOriginalRace(global, into: $0) }
                    if packed != original { problems.add("PLE row \(global) differs from the model") }
                }
                progress?(counter.next(), total)
            }
        }
        return problems.all
    }

    // MARK: - Writers

    /// Payloads an interrupted build left in a model folder: never valid (the header is written last), so
    /// `origami setup` removes them before it measures free space.
    public static func removePartials(in folder: URL) {
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] where name.hasSuffix(".partial") {
            try? FileManager.default.removeItem(at: folder.appendingPathComponent(name))
        }
    }

    /// A fresh `<url>.partial`: whatever sits at that name (a stale partial, or a planted link) is removed,
    /// never written through.
    private static func createPartial(_ url: URL) throws -> (URL, Int32) {
        let temporary = url.appendingPathExtension("partial")
        unlink(temporary.path)
        let fd = open(temporary.path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0o644)
        guard fd >= 0 else { throw EngineError("Cannot create \(temporary.path): errno \(errno)") }
        return (temporary, fd)
    }

    /// Reads `keys` from `store` (pack if open, else checkpoint spans), writes them in order at
    /// `stride`, and returns their records with digests. Eight workers; files bypass the page cache.
    package static func writeRecords(
        _ keys: [Int], store: PackStore, to url: URL, stride: Int, header: URL, progress: (@Sendable (Int) -> Void)?
    )
        throws -> [StoreHeader.Record]
    {
        let (temporary, fd) = try createPartial(url)
        defer {
            close(fd)
            unlink(temporary.path)
        }  // a no-op once renamed; otherwise no stale partial stays behind
        _ = fcntl(fd, F_NOCACHE, 1)
        guard ftruncate(fd, off_t(keys.count * stride)) == 0 else {
            throw EngineError("Cannot size \(url.lastPathComponent): errno \(errno)")
        }
        let digests = Digests(count: keys.count)
        let problems = Problems()
        let counter = Counter()
        DispatchQueue.concurrentPerform(iterations: 8) { worker in
            var buffer = [UInt8](repeating: 0, count: stride)
            for index in Swift.stride(from: worker, to: keys.count, by: 8) where problems.all.isEmpty {
                do {
                    try buffer.withUnsafeMutableBytes { raw in
                        try store.readExpert(
                            layer: keys[index] / 512, expert: keys[index] % 512,
                            into: UnsafeMutableRawBufferPointer(rebasing: raw[0..<PackStore.expertBytes]))
                        try writeAll(fd, UnsafeRawBufferPointer(raw), at: off_t(index * stride))
                    }
                    digests.set(index, hex(SHA256.hash(data: buffer[0..<PackStore.expertBytes])))
                } catch { problems.add("\(error)") }
                progress?(counter.next())
            }
        }
        if let problem = problems.all.first { throw EngineError(problem) }
        guard fsync(fd) == 0 else { throw EngineError("Writing \(url.lastPathComponent) failed: errno \(errno)") }
        // An old header must never describe the new payload if the process stops before the caller writes the new one.
        if FileManager.default.fileExists(atPath: header.path) { try FileManager.default.removeItem(at: header) }
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: temporary, to: url)
        return keys.enumerated().map { StoreHeader.Record(key: $1, offset: Int64($0 * stride), sha256: digests.value($0)) }
    }

    /// The PLE table, 100 bytes per row (80 weight, 10 scales, 10 biases), shards in order.
    static func writePLE(store: PackStore, to url: URL, progress: @escaping @Sendable (Int) -> Void) throws {
        let (temporary, fd) = try createPartial(url)
        defer {
            close(fd)
            unlink(temporary.path)
        }  // a no-op once renamed; otherwise no stale partial stays behind
        _ = fcntl(fd, F_NOCACHE, 1)
        let rows = ModelContract.pleRowsPerShard
        guard ftruncate(fd, off_t(ModelContract.pleRows * 100)) == 0 else { throw EngineError("Cannot size ple.bin: errno \(errno)") }
        let problems = Problems()
        let counter = Counter()
        // Two shards at a time: each holds its three tensors (250 MB) and the interleaved rows (250 MB).
        DispatchQueue.concurrentPerform(iterations: 2) { worker in
            for shard in Swift.stride(from: worker, to: ModelContract.pleShards, by: 2) where problems.all.isEmpty {
                do {
                    var parts: [[UInt8]] = []
                    for (part, width) in [("weight", 80), ("scales", 10), ("biases", 10)] {
                        var bank = [UInt8](repeating: 0, count: rows * width)
                        try bank.withUnsafeMutableBytes {
                            try store.readTensor("model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard).\(part)", into: $0)
                        }
                        parts.append(bank)
                    }
                    var output = [UInt8](repeating: 0, count: rows * 100)
                    output.withUnsafeMutableBytes { out in
                        parts[0].withUnsafeBytes { w in
                            parts[1].withUnsafeBytes { s in
                                parts[2].withUnsafeBytes { b in
                                    for row in 0..<rows {
                                        let base = out.baseAddress! + row * 100
                                        memcpy(base, w.baseAddress! + row * 80, 80)
                                        memcpy(base + 80, s.baseAddress! + row * 10, 10)
                                        memcpy(base + 90, b.baseAddress! + row * 10, 10)
                                    }
                                }
                            }
                        }
                    }
                    try output.withUnsafeBytes { try writeAll(fd, $0, at: off_t(shard * rows * 100)) }
                } catch { problems.add("\(error)") }
                progress(counter.next())
            }
        }
        if let problem = problems.all.first { throw EngineError(problem) }
        guard fsync(fd) == 0 else { throw EngineError("Writing ple.bin failed: errno \(errno)") }
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: temporary, to: url)
    }

    static func writeAll(_ fd: Int32, _ bytes: UnsafeRawBufferPointer, at offset: off_t) throws {
        var done = 0
        while done < bytes.count {
            let wrote = pwrite(fd, bytes.baseAddress! + done, bytes.count - done, offset + off_t(done))
            if wrote < 0 && errno == EINTR { continue }
            guard wrote > 0 else {
                let code = errno
                throw EngineError(
                    "Write failed: "
                        + (code == ENOSPC
                            ? "the drive is full"
                            : code == ENXIO || code == EIO ? "the drive was disconnected" : "errno \(code)"))
            }
            done += wrote
        }
    }

    static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 { digest.map { String(format: "%02x", $0) }.joined() }
    package static func timestamp() -> String { ISO8601DateFormatter().string(from: Date()) }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func next() -> Int {
            lock.lock()
            defer { lock.unlock() }
            count += 1
            return count
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }
    private final class Problems: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [String] = []
        func add(_ item: String) {
            lock.lock()
            items.append(item)
            lock.unlock()
        }
        var all: [String] {
            lock.lock()
            defer { lock.unlock() }
            return items
        }
    }
    private final class Digests: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [String]
        init(count: Int) { items = Array(repeating: "", count: count) }
        func set(_ index: Int, _ value: String) {
            lock.lock()
            items[index] = value
            lock.unlock()
        }
        func value(_ index: Int) -> String {
            lock.lock()
            defer { lock.unlock() }
            return items[index]
        }
    }
}
