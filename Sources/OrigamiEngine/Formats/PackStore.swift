import CryptoKit
import Foundation

/// One tensor's bytes inside a checkpoint shard. The original spans are always the
/// exact fallback for every packed or mirrored read.
public struct TensorSpan: Decodable, Sendable, Equatable {
    public let file, dtype: String
    public let offset, length: Int64
    public let shape: [Int]
}

/// How packed and mirrored expert reads are checked against their header digests: `sampled` hashes one
/// read in 1024 (across all keys), `sha256` every read.
public enum ReadIntegrity: String, Codable, Sendable { case sampled, sha256 }
/// Where `readExpert` may read a record: a copy then the checkpoint (`automatic`), a copy only (`packed`,
/// which throws when none holds the key) or the checkpoint's nine spans only (`original`).
/// Every source returns the same bytes for a key (BITWISE).
public enum ExpertReadSource: Sendable { case automatic, packed, original }

/// Weight reads for one model: the checkpoint's own spans, plus optional copies of the expert
/// records (one read per expert) and an optional hot-expert mirror. The copies are a pack and
/// further packs or most-used-expert copies on other drives; each packed read uses observed
/// service time and outstanding work to choose a copy holding that expert. A missing or damaged copy is
/// dropped with a log line and the others (finally the checkpoint) keep serving; an unplugged
/// copy is picked up again when it returns. A copy record that fails its digest is dropped from
/// that copy alone, for the life of this store.
/// Caller buffers must remain reserved and GPU-unread until a read returns successfully.
public final class PackStore: @unchecked Sendable {
    /// One expert record of the layout this process runs (`ModelLayout`).
    public static var expertBytes: Int { ModelLayout.expertRecordBytes }
    /// Read counters and copy health for snapshots and `origami status`; a point-in-time copy taken under the lock.
    public struct Status: Codable, Sendable {
        public let packedAvailable: Bool
        public let degradedReason: String?
        public let packedExpertBytes, originalExpertBytes, packedPLEBytes, originalPLEBytes: UInt64
        /// Expert records read from the hot copy, and their bytes.
        package var hotCopyReads: UInt64 = 0
        package var hotCopyBytes: UInt64 = 0
        /// Original nine-span expert records read with concurrent spans (path proof); nil when disabled.
        public var originalConcurrentRecords: UInt64? = nil
        /// Every expert copy the packed reads use, the pack first; nil without a pack folder.
        public var copies: [Copy]? = nil
        /// One expert copy as `Status` reports it.
        public struct Copy: Codable, Sendable {
            public let folder: String
            /// Experts it holds: every one for a pack, the most-used ones for a partial copy (0 while unusable).
            public let experts: Int
            public let reads: UInt64
            /// Why it is not read now, or nil.
            public let problem: String?
        }
    }
    public let model: ModelSource
    public var tensors: [String: TensorSpan] { model.tensors }
    /// Binds caches and mirrors to these weights (config, tensor index, shard sizes).
    public var modelFingerprint: String { model.fingerprint }
    private let integrity: ReadIntegrity
    private let lock = NSLock()
    private var originalFiles: [String: ExactFile] = [:]
    /// The drive the checkpoint (or GGUF) is read from: its files reopen in place when it leaves and returns.
    package let drive: ModelDrive
    /// One copy of expert records: a pack (every record and the PLE table), or a partial copy of the most-used experts
    /// on another drive: the older mirror format (`mirror.json`, read in place) or a `HotCopy` store in the folder's
    /// `copy` subfolder (`store`, checked by its CRC32s). Every copy holds the same bytes for a key. Lock-owned.
    private final class Copy: @unchecked Sendable {
        let folder: URL
        let primary: Bool
        var experts: ExactFile?, ple: ExactFile?
        var records: [StoreHeader.Record?] = []
        /// A further copy in the `HotCopy` format (instead of `experts` and `records`).
        var store: HotCopy?
        /// Keys whose record failed its digest: never read from this copy again, even after a reopen.
        var quarantined = Set<Int>()
        var held = 0
        var lane = ReadLane()
        var problem: String?
        var lastRevival = Date.distantPast, reviving = false
        init(folder: URL, primary: Bool) {
            self.folder = folder
            self.primary = primary
        }
    }
    private var copies: [Copy] = []
    private var serial: UInt64 = 0
    private var placements: UInt64 = 0
    private var packedExpertBytes: UInt64 = 0
    private var originalExpertBytes: UInt64 = 0
    private var spanPool: SpanReadPool?
    /// How much of the model files' expert reading went far below the drive's own healthy rate (`origami refresh`).
    package let modelDriveReads = SlowDriveReads()
    private var originalConcurrentRecords: UInt64 = 0
    private var packedPLEBytes: UInt64 = 0
    private var originalPLEBytes: UInt64 = 0
    /// The pack folder, kept so an unplugged drive can be picked up again.
    public let packFolder: URL?
    /// Logged once per state change, never per read.
    private let log: @Sendable (String) -> Void

    /// The most-used experts' copies (`HotCopy` stores, one per drive, each kept by a `HotCopyKeeper`), read before the
    /// model's own spans through one `HotCopySet`. Swapped by `attachHotCopies` under `hotLock`; a read takes the set
    /// attached when it starts and keeps it to its end.
    private let hotLock = NSLock()
    private var hotCopies: HotCopySet?
    private var hotReads: UInt64 = 0, hotBytes: UInt64 = 0
    private var hotMetadata: (MemoryLedger, MemoryLedger.Reservation)?
    /// Bytes charged for a hot copy's index (at most 48 x 512 records), held while one may be attached.
    package static let hotCopyMetadataBytes: UInt64 = 8 * 1024 * 1024
    deinit { if let (ledger, reservation) = hotMetadata { try? ledger.release(reservation) } }

    /// Charges the hot copy's index to `ledger` for the life of this store. Once, before an executor is published.
    package func reserveHotCopyMetadata(ledger: MemoryLedger) throws {
        guard hotMetadata == nil else { throw EngineError("The hot copy's metadata is already reserved") }
        hotMetadata = (ledger, try ledger.reserve(bytes: Self.hotCopyMetadataBytes, category: "hot-copy-metadata"))
    }
    /// Reads from `copy` (nil: none) from now on; reads already started keep the copy they took.
    package func attachHotCopy(_ copy: HotCopy?) { attachHotCopies(copy.map { [$0] } ?? []) }
    /// Reads from `copies` (one per drive, fastest first) from now on; reads already started keep the set they took.
    package func attachHotCopies(_ copies: [HotCopy]) { hotLock.withLock { hotCopies = copies.isEmpty ? nil : HotCopySet(copies) } }
    package var attachedHotCopies: [HotCopy] { hotLock.withLock { hotCopies?.stores ?? [] } }
    /// The copies reads come from now, with each drive's lane (`HotCopySet.laneRates`).
    package var hotCopySet: HotCopySet? { hotLock.withLock { hotCopies } }
    package var attachedHotCopy: HotCopy? { attachedHotCopies.first }
    package func hasHotExpert(_ key: ExpertKey) -> Bool {
        hotLock.withLock { hotCopies }?.holds(key.layer * ModelContract.experts + key.expert) == true
    }
    /// The record from a hot copy holding it, the one expected to finish first (`HotCopySet`; checked on its first
    /// read), else from the model's spans: the same bytes either way. A copy whose drive is gone gives way to another
    /// copy, then to the spans. A copy holds a GGUF record without the slot's padding (`ModelSource.hotCopyRecordBytes`),
    /// which is zero as `GGUFExperts.prepare` leaves it. A GGUF record's buffer may be a smaller size class's slot.
    package func readInternalExpert(_ key: ExpertKey, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard buffer.count == Self.expertBytes || model.ggufExperts != nil && buffer.count < Self.expertBytes else {
            throw EngineError("Hot copy record buffer differs")
        }
        let index = key.layer * ModelContract.experts + key.expert
        let held = model.ggufExperts == nil ? buffer.count : model.hotCopyRecordBytes(index) ?? 0
        if (1...buffer.count).contains(held), let copies = hotLock.withLock({ hotCopies }), copies.holds(index),
            (try? copies.read(index, into: UnsafeMutableRawBufferPointer(rebasing: buffer[..<held]))) != nil
        {
            memset(buffer.baseAddress! + held, 0, buffer.count - held)
            hotLock.withLock {
                hotReads &+= 1
                hotBytes &+= UInt64(held)
            }
            return
        }
        try readExpert(layer: key.layer, expert: key.expert, into: buffer, source: .original)
    }

    /// `pack` is a pack folder (`StoreHeader.modelFolder`); nil serves from the checkpoint alone. `copies` are
    /// more folders read beside it, each holding a pack or a partial copy (other drives, fastest first).
    public init(
        model: ModelSource, pack: URL? = nil, copies: [URL] = [], integrity: ReadIntegrity = .sampled,
        log: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data("origami: \($0)\n".utf8)) }
    ) throws {
        self.model = model
        self.integrity = integrity
        self.log = log
        drive = ModelDrive(log: log)
        packFolder = model.ggufExperts == nil ? pack : nil
        // GGUF experts are read from the GGUF and its hot copy: packs and further copies hold MLX records.
        for folder in model.ggufExperts == nil ? (pack.map { [$0] } ?? []) + copies : [] {
            let copy = Copy(folder: folder, primary: folder == pack)
            do { try open(copy) } catch {
                copy.problem = "unavailable: \(error)"
                log("\(copy.primary ? "pack" : "expert copy") at \(folder.path) is not used: \(error)")
            }
            self.copies.append(copy)
        }
    }

    private final class TimeoutBox<T>: @unchecked Sendable { var result: Result<T, Error>? }
    /// Touches `url` with a timeout: a server started by launchd that touches a removable volume
    /// without the macOS "Removable Volumes" permission blocks in open() until someone
    /// answers a prompt nobody can see. Skip the store instead of hanging.
    private static func withTimeout<T>(_ url: URL, seconds: Double = 10, _ body: @escaping @Sendable () throws -> T) throws -> T {
        let box = TimeoutBox<T>()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            box.result = Result { try body() }
            done.signal()
        }
        guard done.wait(timeout: .now() + seconds) == .success, let result = box.result else {
            throw EngineError(
                "opening \(url.path) did not finish in \(Int(seconds)) s; if this runs from launchd or another app, "
                    + "grant it access to removable volumes (System Settings > Privacy & Security > Files and Folders) or run `origami setup` once in Terminal",
                code: "volume_permission")
        }
        return try result.get()
    }

    /// The `identity` of this model's copies in the `HotCopy` format (the hot copy, and further copies on other drives).
    package static func hotCopyIdentity(_ model: ModelSource) -> String { "qwen \(model.fingerprint) \(model.quantization.rawValue)" }
    /// The subfolder of a store's model folder holding a further copy in the `HotCopy` format.
    package static let furtherCopyFolder = "copy"

    /// A copy is its folder's pack, or for a folder beside the pack its partial copy: the older mirror format, else a
    /// `HotCopy` store in its `copy` subfolder.
    private func open(_ copy: Copy) throws {
        let folder = copy.folder
        let primary = copy.primary
        let header = try Self.withTimeout(folder) { () throws -> StoreHeader? in
            if let pack = try StoreHeader.load(.pack, in: folder) { return pack }
            return primary ? nil : try StoreHeader.load(.mirror, in: folder)
        }
        if header == nil, !primary {
            let identity = Self.hotCopyIdentity(model)
            let url = folder.appendingPathComponent(Self.furtherCopyFolder, isDirectory: true)
            if let store = try Self.withTimeout(url, { [self] in HotCopy.open(folder: url, identity: identity, source: self) }) {
                lock.lock()
                defer { lock.unlock() }
                copy.store = store
                copy.held = store.keys.count
                copy.problem = nil
                return
            }
        }
        guard let header else { throw EngineError("no \(StoreHeader.fileName(.pack)) in \(folder.path)") }
        if let reason = header.mismatch(for: model, folder: folder) {
            throw EngineError("\(header.kind == .pack ? "pack" : "copy") \(reason); run `origami setup` to rebuild it")
        }
        let expertURL = folder.appendingPathComponent(header.expertFile)
        let experts = try Self.withTimeout(expertURL) { try ExactFile(url: expertURL) }
        let ple = try header.pleFile.map { name in
            let url = folder.appendingPathComponent(name)
            return try Self.withTimeout(url) { try ExactFile(url: url) }
        }
        var ordered = [StoreHeader.Record?](repeating: nil, count: ModelContract.layers * ModelContract.experts)
        for record in header.records { ordered[record.key] = record }
        lock.lock()
        defer { lock.unlock() }
        for key in copy.quarantined { ordered[key] = nil }
        copy.records = ordered
        copy.held = header.records.count
        copy.experts = experts
        copy.ple = ple
        copy.problem = nil
    }

    /// The least busy open copy holding `key`, its in-flight count raised; ties go to the earlier copy (the pack,
    /// then the others fastest first), so a lone read takes the fastest drive and a burst spreads over all of them.
    package static func leastBusy(_ inFlight: [Int?]) -> Int? {
        inFlight.indices.filter { inFlight[$0] != nil }.min { inFlight[$0]! < inFlight[$1]! }
    }
    /// A copy to read `key` from (not one in `tried`), its in-flight count raised: the shared choice among copies on
    /// several drives (`HotCopySet.pick`). Its payload and record (nil for a `HotCopy` store) and whether to hash this read.
    private func claim(_ key: Int, excluding tried: Set<ObjectIdentifier>) -> (Copy, (ExactFile, StoreHeader.Record)?, Bool)? {
        lock.lock()
        defer { lock.unlock() }
        let eligible = copies.filter {
            !tried.contains(ObjectIdentifier($0)) && ($0.store?.holds(key) ?? ($0.experts != nil && $0.records[key] != nil))
        }
        guard !eligible.isEmpty else { return nil }
        placements &+= 1
        // Warm each available copy from ordinary reads, then periodically refresh an idle path.
        // No extra I/O is generated; all candidates hold this exact record and its digest.
        let copy = eligible[HotCopySet.pick(eligible.map(\.lane), placements: placements, bytes: Self.expertBytes)!]
        copy.lane.lastSelected = placements
        copy.lane.inFlight += 1
        guard copy.store == nil else { return (copy, nil, false) }
        serial &+= 1
        return (copy, (copy.experts!, copy.records[key]!), integrity == .sha256 || (integrity == .sampled && serial % 1024 == 1))
    }

    /// A shard of the checkpoint, or (by its absolute path) a GGUF the experts are read from.
    private func originalFile(_ name: String) throws -> ExactFile {
        lock.lock()
        defer { lock.unlock() }
        if let file = originalFiles[name] { return file }
        let gguf = model.ggufExperts?.files.first { $0.path == name }
        guard let size = model.shardSizes[name] ?? gguf?.size else { throw EngineError("Unknown checkpoint shard \(name)") }
        let file = try ExactFile(
            url: gguf.map { URL(fileURLWithPath: $0.path) } ?? model.directory.appendingPathComponent(name), drive: drive)
        // An mtime outside 1678...2262 overflows the nanoseconds (APFS clamps to that range, rounding seconds down):
        // no recorded value can match it, so the shard reads as changed.
        let seconds = file.identity.modificationSeconds.multipliedReportingOverflow(by: 1_000_000_000)
        let modified = seconds.partialValue.addingReportingOverflow(file.identity.modificationNanoseconds)
        guard file.identity.size == size, !seconds.overflow, !modified.overflow,
            (gguf?.modified ?? model.shardModified[name]).map({ $0 == modified.partialValue }) ?? true
        else {
            throw EngineError("Checkpoint shard \(name) changed since Origami started; restart it")
        }
        originalFiles[name] = file
        return file
    }

    public func validateBatchBoundary() throws {
        lock.lock()
        let originals = Array(originalFiles.values)
        let open = copies.filter { $0.experts != nil }.map { ($0, [$0.experts!] + ($0.ple.map { [$0] } ?? [])) }
        lock.unlock()
        for file in originals { try file.validateIdentity() }
        for (copy, files) in open {
            do { for file in files { try file.validateIdentity() } } catch { disable(copy, "changed while serving: \(error)") }
        }
        reviveIfPossible()
    }
    private func disable(_ copy: Copy, _ reason: String) {
        lock.lock()
        let wasOpen = copy.experts != nil
        if wasOpen { copy.problem = reason }  // the first cause, not a later in-flight reader's
        copy.experts = nil
        copy.ple = nil  // Other concurrent readers retain their descriptors until their reads join.
        let others = copies.contains { $0.experts != nil || $0.store != nil }
        lock.unlock()
        if wasOpen {
            log(
                "\(copy.primary ? "pack" : "expert copy") at \(copy.folder.path): \(reason); serving from "
                    + (others ? "the other copies" : "the model checkpoint") + " until it is back")
        }
    }
    /// Re-opens unplugged copies once their drive is back. Throttled, and off the calling (serving) thread:
    /// a slow, returning or permission-blocked drive must not stall a request.
    private func reviveIfPossible() {
        lock.lock()
        let due = copies.filter { $0.experts == nil && $0.store == nil && !$0.reviving && Date().timeIntervalSince($0.lastRevival) >= 30 }
        for copy in due {
            copy.lastRevival = Date()
            copy.reviving = true
        }
        lock.unlock()
        guard !due.isEmpty else { return }
        DispatchQueue.global(qos: .utility).async { [self] in
            for copy in due {
                if [StoreHeader.fileName(.pack), StoreHeader.fileName(.mirror), "\(Self.furtherCopyFolder)/index.json"]
                    .contains(where: {
                        FileManager.default.fileExists(atPath: copy.folder.appendingPathComponent($0).path)
                    })
                {
                    do {
                        try open(copy)
                        log("\(copy.primary ? "pack" : "expert copy") at \(copy.folder.path) is back in use")
                    } catch {
                        lock.lock()
                        copy.problem = "unavailable: \(error)"
                        lock.unlock()
                    }
                }
                lock.lock()
                copy.reviving = false
                lock.unlock()
            }
        }
    }

    public func readTensor(_ name: String, row: Int? = nil, into buffer: UnsafeMutableRawBufferPointer) throws {
        let canonical = name.hasPrefix("language_model.") ? name : "language_model." + name
        guard let span = tensors[canonical] else { throw EngineError("Missing checkpoint tensor: \(canonical)") }
        var length = span.length
        var offset = span.offset
        if let row {
            guard let rows = span.shape.first, rows > 0, (0..<rows).contains(row), length % Int64(rows) == 0 else {
                throw EngineError("Invalid tensor row")
            }
            length /= Int64(rows)
            offset += Int64(row) * length
        }
        guard length == buffer.count else { throw EngineError("Tensor destination length differs from span") }
        let file = try originalFile(span.file)
        try file.validateIdentity()
        try file.read(offset: offset, into: buffer)
        try file.validateIdentity()
        if model.zeroCentredNorms, row == nil, span.dtype == "BF16", span.shape.count == 1,
            CheckpointConverter.plusOneNorms.contains(where: { canonical.hasSuffix($0) })
        {
            // Stored as w: the engine computes with 1 + w, rounded as the converter rounds it.
            let values = buffer.bindMemory(to: UInt16.self)
            for i in values.indices { values[i] = BF16.encode(BF16.decode(values[i]) + 1) }
        }
    }

    public func readExpert(
        layer: Int, expert: Int, into buffer: UnsafeMutableRawBufferPointer,
        source readSource: ExpertReadSource = .automatic
    ) throws {
        // A GGUF record may be read into a smaller size class's slot (`readGGUFExpert` checks it holds the layer's record).
        guard (0..<ModelContract.layers).contains(layer), (0..<ModelContract.experts).contains(expert),
            buffer.count == Self.expertBytes || model.ggufExperts != nil && buffer.count < Self.expertBytes
        else {
            throw EngineError("Invalid expert or record buffer")
        }
        var failure: Error?
        let key = layer * ModelContract.experts + expert
        var tried = Set<ObjectIdentifier>()
        while readSource != .original, let (copy, payload, check) = claim(key, excluding: tried) {
            tried.insert(ObjectIdentifier(copy))
            do {
                defer {
                    lock.lock()
                    copy.lane.inFlight -= 1
                    lock.unlock()
                }
                guard let (file, record) = payload else {
                    // A `HotCopy` store pauses itself after a read error and drops a record failing its CRC32.
                    let started = DispatchTime.now().uptimeNanoseconds
                    do { try copy.store!.read(key, into: buffer) } catch {
                        failure = error
                        continue
                    }
                    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
                    lock.lock()
                    copy.lane.served(seconds: seconds, bytes: buffer.count)
                    packedExpertBytes += UInt64(buffer.count)
                    lock.unlock()
                    return
                }
                try file.validateIdentity()
                let started = DispatchTime.now().uptimeNanoseconds
                try file.read(offset: record.offset, into: buffer)
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
                try file.validateIdentity()
                if check, Self.digest(Data(bytesNoCopy: buffer.baseAddress!, count: buffer.count, deallocator: .none)) != record.sha256 {
                    // The file is unchanged, so this one record is bad: drop the record, not the copy.
                    lock.lock()
                    let first = copy.quarantined.insert(key).inserted
                    copy.records[key] = nil
                    lock.unlock()
                    if first {
                        log(
                            "\(copy.primary ? "pack" : "expert copy") at \(copy.folder.path): expert \(key) failed its checksum; "
                                + "it is read from another copy or the model until restart (run `origami verify`)")
                    }
                    failure = EngineError("Expert payload digest mismatch")
                    continue
                }
                lock.lock()
                copy.lane.served(seconds: seconds, bytes: buffer.count)
                packedExpertBytes += UInt64(buffer.count)
                lock.unlock()
                return
            } catch {
                // Another copy may still hold the record; the checkpoint is the last resort.
                disable(copy, "expert read failed (\(error))")
                failure = error
            }
        }
        if readSource == .packed { throw failure ?? EngineError("Packed expert source is unavailable") }
        modelDriveReads.begin()
        var read = 0
        defer { modelDriveReads.end(bytes: Double(read)) }
        try readOriginalExpert(layer: layer, expert: expert, into: buffer)
        read = buffer.count
    }

    /// The checkpoint's nine expert spans, back to back in its own record format.
    private func readOriginalExpert(layer: Int, expert: Int, into buffer: UnsafeMutableRawBufferPointer) throws {
        if let gguf = model.ggufExperts { return try readGGUFExpert(gguf, layer: layer, expert: expert, into: buffer) }
        var cursor = 0
        var parts: [(name: String, range: Range<Int>)] = []
        for projection in ["gate", "up", "down"] {
            for part in model.quantization.expertParts {
                let name = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)_proj.\(part)"
                guard let span = tensors[name], span.shape.first == ModelContract.experts, span.length % Int64(ModelContract.experts) == 0,
                    span.length / Int64(ModelContract.experts) <= Int64(buffer.count - cursor)
                else { throw EngineError("Missing/inconsistent expert component") }
                let length = Int(span.length / Int64(ModelContract.experts))
                parts.append((name, cursor..<(cursor + length)))
                cursor += length
            }
        }
        guard cursor == buffer.count else { throw EngineError("Incomplete original expert record") }
        lock.lock()
        let pool = spanPool
        lock.unlock()
        if let pool {
            // The same nine validated reads into disjoint ranges, issued together.
            nonisolated(unsafe) let destination = buffer
            try pool.run(
                parts.map { part in
                    { [self] in
                        try readTensor(part.name, row: expert, into: UnsafeMutableRawBufferPointer(rebasing: destination[part.range]))
                    }
                })
        } else {
            for part in parts {
                try readTensor(part.name, row: expert, into: UnsafeMutableRawBufferPointer(rebasing: buffer[part.range]))
            }
        }
        lock.lock()
        originalExpertBytes += UInt64(buffer.count)
        originalConcurrentRecords += pool == nil ? 0 : 1
        lock.unlock()
    }

    /// A GGUF expert record: its header, then gate, up and down, three spans of the GGUF read into their places.
    private func readGGUFExpert(_ gguf: GGUFExperts, layer: Int, expert: Int, into buffer: UnsafeMutableRawBufferPointer) throws {
        // A slot, or the layer's own bytes (a hot copy's record).
        guard expert < gguf.experts, gguf.layers.indices.contains(layer),
            (gguf.layers[layer].bytes...gguf.recordBytes).contains(buffer.count)
        else {
            throw EngineError("Expert \(expert) of layer \(layer) is not in this model's GGUF (\(gguf.experts) experts)")
        }
        gguf.prepare(buffer, layer: layer, expert: expert)
        let parts = try gguf.layers[layer].parts.map { part in
            (
                file: try originalFile(part.path), offset: Int64(part.offset + expert * part.stride),
                range: part.record..<(part.record + part.bytes)
            )
        }
        nonisolated(unsafe) let destination = buffer
        let reads = parts.map { part in
            { () throws -> Void in
                try part.file.validateIdentity()
                try part.file.read(offset: part.offset, into: UnsafeMutableRawBufferPointer(rebasing: destination[part.range]))
                try part.file.validateIdentity()
            }
        }
        lock.lock()
        let pool = spanPool
        lock.unlock()
        if let pool { try pool.run(reads) } else { for read in reads { try read() } }
        lock.lock()
        originalExpertBytes += UInt64(buffer.count)
        originalConcurrentRecords += pool == nil ? 0 : 1
        lock.unlock()
    }

    public func readPLERow(_ row: Int, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard (0..<ModelContract.pleRows).contains(row), buffer.count == 100 else { throw EngineError("Invalid PLE row/destination") }
        lock.lock()
        let copy = copies.first { $0.ple != nil }
        let packed = copy?.ple
        lock.unlock()
        if let copy, let packed {
            do {
                try packed.validateIdentity()
                try packed.read(offset: Int64(row) * 100, into: buffer)
                try packed.validateIdentity()
                lock.lock()
                packedPLEBytes += 100
                lock.unlock()
                return
            } catch { disable(copy, "PLE read failed (\(error))") }
        }
        try readOriginalPLESpans(row, into: buffer)
        lock.lock()
        originalPLEBytes += 100
        lock.unlock()
    }

    /// `decode_ple_internal_race`: one PLE row from the checkpoint spans (the model drive),
    /// the same 100 bytes as the packed row. Not a fallback: it leaves the pack state and
    /// `originalPLEBytes` (the fallback proof) unchanged.
    public func readPLERowOriginalRace(_ row: Int, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard (0..<ModelContract.pleRows).contains(row), buffer.count == 100 else { throw EngineError("Invalid PLE row/destination") }
        try readOriginalPLESpans(row, into: buffer)
    }

    /// True while a pack's PLE table is open.
    public var packedPLEAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return copies.contains { $0.ple != nil }
    }
    /// True while a complete copy (a pack) is open: any expert can then be read in one read.
    public var packedAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return copies.contains { $0.experts != nil && $0.held == ModelContract.layers * ModelContract.experts }
    }

    private func readOriginalPLESpans(_ row: Int, into buffer: UnsafeMutableRawBufferPointer) throws {
        let shard = row / ModelContract.pleRowsPerShard
        let local = row % ModelContract.pleRowsPerShard
        var cursor = 0
        for (part, length) in [("weight", 80), ("scales", 10), ("biases", 10)] {
            let name = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard).\(part)"
            try readTensor(name, row: local, into: UnsafeMutableRawBufferPointer(rebasing: buffer[cursor..<cursor + length]))
            cursor += length
        }
    }

    public func status() -> Status {
        let available = packedAvailable
        lock.lock()
        defer { lock.unlock() }
        var value = Status(
            packedAvailable: available, degradedReason: copies.first { $0.primary }?.problem,
            packedExpertBytes: packedExpertBytes, originalExpertBytes: originalExpertBytes,
            packedPLEBytes: packedPLEBytes, originalPLEBytes: originalPLEBytes)
        (value.hotCopyReads, value.hotCopyBytes) = hotLock.withLock { (hotReads, hotBytes) }
        value.originalConcurrentRecords = spanPool == nil ? nil : originalConcurrentRecords
        if !copies.isEmpty {
            value.copies = copies.map {
                Status.Copy(
                    folder: $0.folder.path, experts: $0.experts == nil && $0.store == nil ? 0 : $0.held, reads: $0.lane.reads,
                    problem: $0.problem)
            }
        }
        return value
    }

    /// Read the nine spans of each original expert record concurrently (once, at startup).
    package func enableConcurrentOriginalSpans(pool: SpanReadPool) throws {
        lock.lock()
        defer { lock.unlock() }
        guard spanPool == nil else { throw EngineError("Concurrent original spans are already enabled") }
        spanPool = pool
    }

    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}

/// Qwen's hot copy holds an expert's record in the layout this process runs, read as `origami setup` read its offline
/// mirror's: from a store opened with the pack and `.sha256` (a pack record failing its digest is read from the spans),
/// else from the checkpoint's spans. A key is layer * 512 + expert. A GGUF expert record is its layer's own bytes
/// (`ModelSource.hotCopyRecordBytes`), read from the GGUF: a copy of a mixed-type GGUF holds and reads no slot padding.
extension PackStore: HotCopySource {
    package func hotCopyRecordBytes(_ key: Int) -> Int? {
        model.ggufExperts == nil
            ? ((0..<(ModelContract.layers * ModelContract.experts)).contains(key) ? Self.expertBytes : nil) : model.hotCopyRecordBytes(key)
    }
    package func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
        if let gguf = model.ggufExperts {
            guard let bytes = model.hotCopyRecordBytes(key) else { throw EngineError("Expert record \(key) is not in this model's GGUF") }
            return try readGGUFExpert(
                gguf, layer: key / ModelContract.experts, expert: key % ModelContract.experts,
                into: UnsafeMutableRawBufferPointer(start: into, count: bytes))
        }
        try readExpert(
            layer: key / ModelContract.experts, expert: key % ModelContract.experts,
            into: UnsafeMutableRawBufferPointer(start: into, count: Self.expertBytes))
    }
}
