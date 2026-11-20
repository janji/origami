import CryptoKit
import Darwin
import Foundation

/// Bounded runtime-only recurrent images. Files are unlinked immediately after
/// creation: the last verified reader owns both their bytes and disk charge,
/// and process exit cannot leave unbounded orphan snapshots. QSA pages and the
/// live sequence remain separate shared owners. This is not the restart codec.
package final class ConversationFrontierStore {
    /// Disk charge and traffic counters for /v1/status and diagnostics.
    package struct Snapshot: Codable, Sendable {
        package let liveBytes, highWaterBytes, writtenBytes, readBytes: UInt64
        package let files, captures, restores: Int
        /// Async writer path counters; absent on the synchronous path.
        package var writeWindow: WriteWindow? = nil
    }
    /// Asynchronous writer counters (prefix_cache.frontier_write_staging_slots). `windowWaitSeconds` is the time a
    /// retain blocked for a free slot (exposed writer lag); `writerSeconds` and `maximumSlotWriteSeconds` are the
    /// serial writer's append+hash time in total and for its slowest single slot; `sealWaitSeconds` is the time
    /// `seal()` waited for queued slots and `fsyncSeconds` the time in its fsync.
    package struct WriteWindow: Codable, Sendable {
        package let stagingSlots, stagedSlots, windowWaits, maximumSlotsInUse: Int
        package let windowWaitSeconds, writerSeconds, maximumSlotWriteSeconds, sealWaitSeconds, fsyncSeconds: Double
    }
    package static let imageBytes: UInt64 = 36 * (48 * 128 * 128 * 4 + 3 * 10240 * 2)
    private let directory: URL, directoryFD: Int32
    package let ledger: MemoryLedger
    package let memoryQuota: CheckpointMemoryQuota?
    package let maximumBytes: UInt64
    private let reserveBytes: UInt64
    private let metadata: MemoryLedger.Reservation
    private let lock = NSLock()
    private var live: UInt64 = 0, high: UInt64 = 0, written: UInt64 = 0, read: UInt64 = 0
    private var files = 0, captures = 0, restores = 0
    /// Asynchronous writes: recurrent images are appended and hashed on one serial writer, off the prefill thread.
    /// Each retain copies its layer into one of `stagingSlots` ledger-reserved slots; the writer
    /// appends the same bytes in the same per-file order, so every image is byte-identical to the
    /// synchronous path. A full window blocks the caller, which bounds staged memory.
    /// The window width is configurable (prefix_cache.frontier_write_staging_slots, default 4; the plan uses 64).
    /// A wider window lets the writer absorb its periodic multi-hundred-millisecond append stalls and
    /// drain during the QSA/MoE stages instead of blocking the GDN tile loop.
    package static let defaultStagingSlots = 4
    package static let stagingSlotRange = 4...96
    package static let stagingSlotBytes = 48 * 128 * 128 * 4 + 3 * 10240 * 2
    package static func stagingReservationBytes(slots: Int) -> UInt64 {
        let page = Int(getpagesize())
        return UInt64((stagingSlotBytes + page - 1) / page * page * slots)
    }
    package static var stagingReservationBytes: UInt64 { stagingReservationBytes(slots: defaultStagingSlots) }
    fileprivate let writer = DispatchQueue(label: "origami.conversation-frontier.writer", qos: .userInitiated)
    fileprivate let window: DispatchSemaphore
    package let asyncWrites: Bool
    package let stagingSlots: Int
    private var stagedSlots = 0, windowWaits = 0, slotsInUse = 0, maximumSlotsInUse = 0
    private var windowWaitSeconds = 0.0, writerSeconds = 0.0, maximumSlotWriteSeconds = 0.0
    private var sealWaitSeconds = 0.0, fsyncSeconds = 0.0
    private let staging: MemoryLedger.Reservation?
    private var freeSlots: [UnsafeMutableRawPointer]
    private let slotLock = NSLock()

    /// - reserveBytes: free disk kept above the drive's reserve (`DiskBudget.reserve`, 10 GB) for the prompt cache's next save.
    package init(
        parent: URL, maximumBytes: UInt64, ledger: MemoryLedger, memoryBytes: UInt64? = nil, asyncWrites: Bool = false,
        stagingSlots: Int = ConversationFrontierStore.defaultStagingSlots, reserveBytes: UInt64 = 0
    ) throws {
        self.reserveBytes = reserveBytes
        guard Self.stagingSlotRange.contains(stagingSlots), asyncWrites || stagingSlots == Self.defaultStagingSlots else {
            throw EngineError("Frontier write staging slots need asynchronous writes and 4...96 slots", code: "invalid_configuration")
        }
        guard maximumBytes >= Self.imageBytes, maximumBytes <= 360 * Self.imageBytes else {
            throw EngineError("Conversation frontier storage needs a bounded image quota")
        }
        let metadata = try ledger.reserve(bytes: 131072, category: "conversation_frontier_store")
        let page = Int(getpagesize())
        let slotBytes = (Self.stagingSlotBytes + page - 1) / page * page
        let staging: MemoryLedger.Reservation?
        do {
            staging =
                asyncWrites
                ? try ledger.reserve(
                    bytes: Self.stagingReservationBytes(slots: stagingSlots), category: "conversation_frontier_write_staging") : nil
        } catch {
            try? ledger.release(metadata)
            throw error
        }
        var slots: [UnsafeMutableRawPointer] = []
        for _ in 0..<(asyncWrites ? stagingSlots : 0) {
            guard let slot = mmap(nil, slotBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), slot != MAP_FAILED else {
                for slot in slots { munmap(slot, slotBytes) }
                if let staging { try? ledger.release(staging) }
                try? ledger.release(metadata)
                throw EngineError("Frontier write staging allocation failed", code: "insufficient_memory")
            }
            slots.append(slot)
        }
        func unwind() {
            for slot in slots { munmap(slot, slotBytes) }
            if let staging { try? ledger.release(staging) }
            try? ledger.release(metadata)
        }
        let directory = parent.appendingPathComponent("frontiers-" + UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        } catch {
            unwind()
            throw error
        }
        let fd = directory.withUnsafeFileSystemRepresentation { Darwin.open($0!, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
        guard fd >= 0 else {
            unwind()
            throw EngineError("Frontier directory open failed: errno=\(errno)")
        }
        self.directory = directory
        directoryFD = fd
        self.maximumBytes = maximumBytes
        self.ledger = ledger
        self.metadata = metadata
        self.staging = staging
        freeSlots = slots
        self.asyncWrites = asyncWrites
        self.stagingSlots = stagingSlots
        window = DispatchSemaphore(value: stagingSlots)
        memoryQuota = memoryBytes.map { CheckpointMemoryQuota(maximumBytes: $0) }
        budgetEntry = DiskBudget.shared.add(
            "conversation frontiers", folder: { [directory] in directory }, bytes: { [weak self] in self?.snapshot().liveBytes ?? 0 })
    }
    /// This store's entry in the drive's account (`DiskBudget`).
    private var budgetEntry: Int?
    deinit {
        if let budgetEntry { DiskBudget.shared.remove(budgetEntry) }
        // Do not remove a replacement directory reached through the same path.
        var owned = stat()
        var current = stat()
        let same =
            fstat(directoryFD, &owned) == 0 && directory.withUnsafeFileSystemRepresentation { lstat($0!, &current) } == 0
            && owned.st_dev == current.st_dev && owned.st_ino == current.st_ino
        if same { _ = directory.withUnsafeFileSystemRepresentation { rmdir($0!) } }
        Darwin.close(directoryFD)
        // Files hold the store strongly, so no write can still be queued here.
        let page = Int(getpagesize())
        let slotBytes = (Self.stagingSlotBytes + page - 1) / page * page
        for slot in freeSlots { munmap(slot, slotBytes) }
        if let staging { try? ledger.release(staging) }
        try? ledger.release(metadata)
    }
    fileprivate func takeSlot() -> UnsafeMutableRawPointer {
        var waited = 0.0
        if window.wait(timeout: .now()) == .timedOut {
            let started = DispatchTime.now().uptimeNanoseconds
            window.wait()
            waited = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        }
        slotLock.lock()
        defer { slotLock.unlock() }
        stagedSlots += 1
        slotsInUse += 1
        maximumSlotsInUse = max(maximumSlotsInUse, slotsInUse)
        if waited > 0 {
            windowWaits += 1
            windowWaitSeconds += waited
        }
        return freeSlots.removeLast()
    }
    fileprivate func returnSlot(_ slot: UnsafeMutableRawPointer, writeSeconds: Double) {
        slotLock.lock()
        freeSlots.append(slot)
        slotsInUse -= 1
        writerSeconds += writeSeconds
        maximumSlotWriteSeconds = max(maximumSlotWriteSeconds, writeSeconds)
        slotLock.unlock()
        window.signal()
    }
    fileprivate func sealed(waitSeconds: Double, fsyncSeconds seconds: Double) {
        slotLock.lock()
        sealWaitSeconds += waitSeconds
        fsyncSeconds += seconds
        slotLock.unlock()
    }
    package func writeWindowSnapshot() -> WriteWindow? {
        guard asyncWrites else { return nil }
        slotLock.lock()
        defer { slotLock.unlock() }
        return WriteWindow(
            stagingSlots: stagingSlots, stagedSlots: stagedSlots, windowWaits: windowWaits,
            maximumSlotsInUse: maximumSlotsInUse, windowWaitSeconds: windowWaitSeconds, writerSeconds: writerSeconds,
            maximumSlotWriteSeconds: maximumSlotWriteSeconds, sealWaitSeconds: sealWaitSeconds, fsyncSeconds: fsyncSeconds)
    }
    package func make(offset: Int) throws -> FrontierRecurrentFile {
        // A full or vanished volume fails only this save; the generation driver skips it under original_spans.
        let free = { [directoryFD] () -> UInt64 in
            var volume = statfs()
            return fstatfs(directoryFD, &volume) == 0 ? UInt64(volume.f_bavail) * UInt64(volume.f_bsize) : 0
        }
        // Keep the drive usable: free space stays above the drive's one reserve (`DiskBudget`) plus room for the prompt
        // cache's next save; when short, the hot copies on this drive give back what is missing; only then is the
        // frontier skipped (the request still runs).
        let need = Self.imageBytes + reserveBytes
        if free() < need + DiskBudget.reserve,
            !DiskBudget.shared.makeRoom(need, in: directory, for: "a conversation frontier save", free: free)
        {
            DiskBudget.shared.skipped(in: directory, for: "a conversation frontier save", reason: "lacks space for one image")
            throw EngineError("Conversation frontier volume lacks space for one image", code: "conversation_frontier_disk_space")
        }
        lock.lock()
        guard Self.imageBytes <= maximumBytes - live else {
            lock.unlock()
            throw EngineError("Pinned recurrent readers exceed disk quota", code: "conversation_frontier_disk_quota")
        }
        live += Self.imageBytes
        high = max(high, live)
        files += 1
        lock.unlock()
        do { return try FrontierRecurrentFile(store: self, directoryFD: directoryFD, offset: offset) } catch {
            retired()
            throw error
        }
    }
    fileprivate func retired() {
        lock.lock()
        live -= Self.imageBytes
        files -= 1
        lock.unlock()
    }
    fileprivate func wrote(_ bytes: Int) {
        lock.lock()
        written += UInt64(bytes)
        lock.unlock()
    }
    fileprivate func readBytes(_ bytes: Int) {
        lock.lock()
        read += UInt64(bytes)
        lock.unlock()
    }
    fileprivate func captured() {
        lock.lock()
        captures += 1
        lock.unlock()
    }
    fileprivate func restored() {
        lock.lock()
        restores += 1
        lock.unlock()
    }
    package func snapshot() -> Snapshot {
        let window = writeWindowSnapshot()
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            liveBytes: live, highWaterBytes: high, writtenBytes: written, readBytes: read,
            files: files, captures: captures, restores: restores, writeWindow: window)
    }
}

/// Sendable by discipline: `sections`, `length` and `failure` are written only on the store's serial writer
/// until `pending` is joined (seal, deinit); `scheduled` is touched only by the capturing thread.
package final class FrontierRecurrentFile: @unchecked Sendable {
    private struct Section { let offset: UInt64, length: Int, sha256: SHA256.Digest }
    package let offset: Int
    private let store: ConversationFrontierStore, fd: Int32, memory: CheckpointMemoryLease
    private var sections: [Int: (Section, Section)] = [:]
    private var length: UInt64 = 0, sealed = false
    private var identity = stat()
    /// Caller-side record of queued layers; `sections`, `length` and `failure` belong to the writer
    /// until `pending` is joined.
    private var scheduled = Set<Int>()
    private let pending = DispatchGroup()
    private var failure: Error?

    fileprivate init(store: ConversationFrontierStore, directoryFD: Int32, offset: Int) throws {
        guard (1...ModelContract.maximumContext).contains(offset) else { throw EngineError("Invalid recurrent file frontier") }
        let memory = try CheckpointMemoryLease(
            ledger: store.ledger, bytes: 262144,
            category: "conversation_frontier_file_metadata", quota: store.memoryQuota)
        let name = ".recurrent-" + UUID().uuidString
        let fd = name.withCString { openat(directoryFD, $0, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600) }
        guard fd >= 0 else { throw EngineError("Recurrent file creation failed: errno=\(errno)") }
        guard name.withCString({ unlinkat(directoryFD, $0, 0) }) == 0 else {
            Darwin.close(fd)
            throw EngineError("Recurrent file unlink ownership failed: errno=\(errno)")
        }
        guard fcntl(fd, F_NOCACHE, 1) == 0 else {
            Darwin.close(fd)
            throw EngineError("Recurrent no-cache I/O unavailable")
        }
        self.store = store
        self.fd = fd
        self.offset = offset
        self.memory = memory
    }
    deinit {
        pending.wait()
        Darwin.close(fd)
        store.retired()
    }
    package var metadataBytes: UInt64 { memory.bytes }

    private func append(_ bytes: UnsafeRawBufferPointer) throws -> Section {
        guard !sealed, UInt64(bytes.count) <= ConversationFrontierStore.imageBytes - length else {
            throw EngineError("Recurrent capture exceeds its admitted immutable image")
        }
        var done = 0
        while done < bytes.count {
            let amount = Darwin.write(fd, bytes.baseAddress!.advanced(by: done), bytes.count - done)
            if amount < 0 && errno == EINTR { continue }
            guard amount > 0 else { throw EngineError("Recurrent capture write failed: errno=\(errno)") }
            done += amount
        }
        var digest = SHA256()
        digest.update(bufferPointer: bytes)
        let result = Section(offset: length, length: bytes.count, sha256: digest.finalize())
        length += UInt64(bytes.count)
        store.wrote(bytes.count)
        return result
    }
    package func retain(layer: Int, state: GDNState) throws {
        try state.validate()
        guard !sealed, (0..<48).contains(layer), layer % 4 != 3,
            !scheduled.contains(layer), state.offset == offset
        else { throw EngineError("Recurrent capture layer/frontier differs") }
        guard store.asyncWrites else {
            let recurrent = try state.withRecurrentBytes { try append($0) }
            let tail = try state.convolutionTail.withUnsafeBytes { try append($0) }
            scheduled.insert(layer)
            sections[layer] = (recurrent, tail)
            return
        }
        let recurrentBytes = 48 * 128 * 128 * 4
        let tailBytes = 3 * 10240 * 2
        let slot = store.takeSlot()
        guard state.copyRecurrentBytes(to: UnsafeMutableRawBufferPointer(start: slot, count: recurrentBytes)),
            state.copyConvolutionBytes(to: UnsafeMutableRawBufferPointer(start: slot.advanced(by: recurrentBytes), count: tailBytes))
        else {
            store.returnSlot(slot, writeSeconds: 0)
            throw EngineError("Recurrent capture staging geometry differs")
        }
        scheduled.insert(layer)
        pending.enter()
        let address = UInt(bitPattern: slot)
        store.writer.async { [self] in
            let slot = UnsafeMutableRawPointer(bitPattern: address)!
            let started = DispatchTime.now().uptimeNanoseconds
            defer {
                store.returnSlot(slot, writeSeconds: Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9)
                pending.leave()
            }
            guard failure == nil else { return }
            do {
                let recurrent = try append(UnsafeRawBufferPointer(start: slot, count: recurrentBytes))
                let tail = try append(UnsafeRawBufferPointer(start: slot.advanced(by: recurrentBytes), count: tailBytes))
                sections[layer] = (recurrent, tail)
            } catch { failure = error }
        }
    }
    package func seal() throws {
        let started = DispatchTime.now().uptimeNanoseconds
        pending.wait()
        let joined = DispatchTime.now().uptimeNanoseconds
        if let failure { throw failure }
        guard !sealed, length == ConversationFrontierStore.imageBytes,
            Set(sections.keys) == Set((0..<48).filter { $0 % 4 != 3 })
        else {
            throw EngineError("Recurrent image is incomplete or failed final publication")
        }
        let synced = fsync(fd) == 0
        if store.asyncWrites {
            store.sealed(
                waitSeconds: Double(joined - started) / 1e9,
                fsyncSeconds: Double(DispatchTime.now().uptimeNanoseconds - joined) / 1e9)
        }
        guard synced, fstat(fd, &identity) == 0, identity.st_size == Int64(length), identity.st_nlink == 0 else {
            throw EngineError("Recurrent image is incomplete or failed final publication")
        }
        sealed = true
        store.captured()
    }
    package func validate(expectedOffset: Int) throws {
        var current = stat()
        guard sealed, expectedOffset == offset, length == ConversationFrontierStore.imageBytes,
            Set(sections.keys) == Set((0..<48).filter { $0 % 4 != 3 }),
            fstat(fd, &current) == 0, current.st_dev == identity.st_dev, current.st_ino == identity.st_ino,
            current.st_size == identity.st_size, current.st_mtimespec.tv_sec == identity.st_mtimespec.tv_sec,
            current.st_mtimespec.tv_nsec == identity.st_mtimespec.tv_nsec,
            current.st_ctimespec.tv_sec == identity.st_ctimespec.tv_sec,
            current.st_ctimespec.tv_nsec == identity.st_ctimespec.tv_nsec
        else {
            throw EngineError("Immutable recurrent image identity changed")
        }
    }
    package func withTensors(_ body: (Int, String, UnsafeRawBufferPointer) throws -> Void) throws {
        try validate(expectedOffset: offset)
        let bytes = 48 * 128 * 128 * 4
        let scratch = try CheckpointMemoryLease(ledger: store.ledger, bytes: UInt64(bytes), category: "conversation_frontier_read_scratch")
        let pointer = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        guard let pointer, pointer != MAP_FAILED else { throw EngineError("Recurrent read scratch allocation failed") }
        defer {
            munmap(pointer, bytes)
            withExtendedLifetime(scratch) {}
        }
        for layer in sections.keys.sorted() {
            let pair = sections[layer]!
            for (name, section) in [("recurrent", pair.0), ("tail", pair.1)] {
                guard section.length <= bytes else { throw EngineError("Recurrent section exceeds scratch") }
                var done = 0
                while done < section.length {
                    let amount = pread(fd, pointer.advanced(by: done), section.length - done, off_t(section.offset) + off_t(done))
                    if amount < 0 && errno == EINTR { continue }
                    guard amount > 0 else { throw EngineError("Recurrent image short read: errno=\(errno)") }
                    done += amount
                }
                let span = UnsafeRawBufferPointer(start: pointer, count: section.length)
                var digest = SHA256()
                digest.update(bufferPointer: span)
                guard digest.finalize() == section.sha256 else { throw EngineError("Recurrent image section checksum differs") }
                store.readBytes(section.length)
                try body(layer, name, span)
            }
        }
        try validate(expectedOffset: offset)
    }
    /// The image's GDN layers read back into host arrays, charged to `memory` while they live.
    package struct Restored {
        package let states: [Int: GDNState]
        private let memory: CheckpointMemoryLease
        fileprivate init(states: [Int: GDNState], memory: CheckpointMemoryLease) {
            self.states = states
            self.memory = memory
        }
    }
    package func restore() throws -> Restored {
        let memory = try CheckpointMemoryLease(ledger: store.ledger, bytes: 65536, category: "conversation_frontier_restore")
        var recurrent: [Int: [Float]] = [:]
        var tails: [Int: [UInt16]] = [:]
        try withTensors { layer, name, bytes in
            if name == "recurrent" {
                recurrent[layer] = try memory.allocateArray(count: 48 * 128 * 128) { count in
                    Array(UnsafeBufferPointer(start: bytes.baseAddress!.assumingMemoryBound(to: Float.self), count: count))
                }
            } else {
                tails[layer] = try memory.allocateArray(count: 3 * 10240) { count in
                    Array(UnsafeBufferPointer(start: bytes.baseAddress!.assumingMemoryBound(to: UInt16.self), count: count))
                }
            }
        }
        var states: [Int: GDNState] = [:]
        for layer in recurrent.keys {
            states[layer] = try GDNState(offset: offset, recurrent: recurrent[layer]!, convolutionTail: tails[layer]!)
        }
        store.restored()
        return Restored(states: states, memory: memory)
    }
}

extension FrontierRecurrentFile {
    /// Only the standalone check in this file can mutate the sealed fixture.
    fileprivate func corruptFixture() throws {
        var byte: UInt8 = 0xff
        guard pwrite(fd, &byte, 1, 0) == 1 else { throw EngineError("Frontier corruption fixture write failed") }
    }
}

/// Model-free checks of the frontier store: window bounds, writer stalls, ownership, faults and corruption.
public enum ConversationFrontierSelfChecks {
    /// A wide staging window absorbs a stalled writer without blocking any retain, keeps every image
    /// byte-identical, charges exactly its slots and reports its path counters. The default window keeps
    /// its four-slot bound; widths outside 4...96 and a widened synchronous store refuse.
    public static func stagingWindowAbsorbsWriterStall() throws {
        func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
        func digest(_ bytes: UnsafeRawBufferPointer) -> String {
            var hash = SHA256()
            hash.update(bufferPointer: bytes)
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        let parent = CheckFixtures.scratch
        let ledger = try MemoryLedger(ceiling: 1_000_000_000, transientMargin: 16_000_000)
        for (slots, asynchronous) in [(3, true), (97, true), (8, false), (0, true)] {
            var refused = false
            do {
                _ = try ConversationFrontierStore(
                    parent: parent, maximumBytes: 1_100_000_000, ledger: ledger,
                    asyncWrites: asynchronous, stagingSlots: slots)
            } catch let error as EngineError { refused = error.code == "invalid_configuration" }
            try require(refused, "Out-of-range frontier staging window \(slots) async=\(asynchronous) was admitted")
        }
        try require(ledger.snapshot().reserved == 0, "Refused staging windows retained ownership")
        let layers = (0..<48).filter { $0 % 4 != 3 }
        var inputs: [(Int, GDNState)] = []
        for layer in layers {
            let salt = layer * 32771 + 7
            let tailSalt = layer * 19 + 7
            let recurrent: [Float] = (0..<48 * 128 * 128).map { Float(bitPattern: 0x3e80_0000 | UInt32(($0 * 65537 + salt) & 0x007f_ffff)) }
            let tail: [UInt16] = (0..<3 * 10240).map { UInt16(($0 + tailSalt) % 65536) }
            inputs.append((layer, try GDNState(offset: 256, recurrent: recurrent, convolutionTail: tail)))
        }
        var expected: [String: String] = [:]
        for (layer, state) in inputs {
            expected["\(layer).recurrent"] = state.withRecurrentBytes(digest)
            expected["\(layer).tail"] = state.convolutionTail.withUnsafeBytes(digest)
        }
        for slots in [ConversationFrontierStore.defaultStagingSlots, 32] {
            try autoreleasepool {
                let store = try ConversationFrontierStore(
                    parent: parent, maximumBytes: 1_100_000_000, ledger: ledger,
                    asyncWrites: true, stagingSlots: slots)
                try require(
                    store.stagingSlots == slots
                        && ledger.snapshot().categories["conversation_frontier_write_staging"]
                            == ConversationFrontierStore.stagingReservationBytes(slots: slots)
                        && ConversationFrontierStore.stagingReservationBytes(slots: slots)
                            == UInt64(slots) * ConversationFrontierStore.stagingReservationBytes
                            / UInt64(ConversationFrontierStore.defaultStagingSlots),
                    "Frontier staging window charge differs from its slots")
                let file = try store.make(offset: 256)
                // Stall the writer: the first `slots` retains must still return (no window wait).
                let stalled = min(slots, inputs.count)
                store.writer.suspend()
                var stallFailure: Error?
                do { for (layer, state) in inputs.prefix(stalled) { try file.retain(layer: layer, state: state) } } catch {
                    stallFailure = error
                }
                let duringStall = store.writeWindowSnapshot()
                store.writer.resume()
                if let stallFailure { throw stallFailure }
                try require(
                    duringStall?.windowWaits == 0 && duringStall?.stagedSlots == stalled
                        && duringStall?.maximumSlotsInUse == stalled, "A stalled writer blocked retains inside the staging window")
                for (layer, state) in inputs.dropFirst(stalled) { try file.retain(layer: layer, state: state) }
                try file.seal()
                try file.withTensors { layer, name, bytes in
                    try require(digest(bytes) == expected["\(layer).\(name)"], "Staging window width changed tensor bits")
                }
                guard let window = store.snapshot().writeWindow else {
                    throw EngineError("Async frontier store omitted its window counters")
                }
                try require(
                    window.stagingSlots == slots && window.stagedSlots == inputs.count
                        && window.maximumSlotsInUse <= slots && window.maximumSlotsInUse >= stalled
                        && window.writerSeconds > 0 && window.maximumSlotWriteSeconds > 0
                        && window.maximumSlotWriteSeconds <= window.writerSeconds
                        && window.sealWaitSeconds >= 0 && window.fsyncSeconds >= 0 && window.windowWaitSeconds >= 0
                        && (window.windowWaits == 0) == (window.windowWaitSeconds == 0),
                    "Frontier write window counters are inconsistent")
            }
            try require(ledger.snapshot().reserved == 0, "Frontier staging window leaked ownership")
        }
        // The synchronous path reports no window.
        try autoreleasepool {
            let store = try ConversationFrontierStore(parent: parent, maximumBytes: 1_100_000_000, ledger: ledger)
            try require(
                store.snapshot().writeWindow == nil && store.writeWindowSnapshot() == nil
                    && ledger.snapshot().categories["conversation_frontier_write_staging"] == nil,
                "Synchronous frontier store reported a write window")
        }
        try require(ledger.snapshot().reserved == 0, "Synchronous frontier store leaked ownership")
    }
    public static func ownershipAndFaults() throws {
        func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
        func rejects(_ body: () throws -> Void) throws {
            do { try body() } catch is EngineError { return }
            throw EngineError("Expected file-backed frontier refusal")
        }
        func digest(_ bytes: UnsafeRawBufferPointer) -> String {
            var hash = SHA256()
            hash.update(bufferPointer: bytes)
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        let ledger = try MemoryLedger(ceiling: 1_000_000_000, transientMargin: 16_000_000)
        let fixture = try ledger.reserve(bytes: 64_000_000, category: "frontier_fixture_inputs")
        var escaped: FrontierRecurrentFile?
        try autoreleasepool {
            let parent = CheckFixtures.scratch
            let store = try ConversationFrontierStore(parent: parent, maximumBytes: 1_100_000_000, ledger: ledger, asyncWrites: true)
            var held = try (0..<9).map { _ in try store.make(offset: 256) }
            try rejects { _ = try store.make(offset: 256) }
            try rejects { try held[0].seal() }
            try rejects { _ = try held[0].restore() }
            held.removeAll()
            try require(store.snapshot().liveBytes == 0, "Unpublished frontier files leaked their disk charge")
            let file = try store.make(offset: 256)
            var expected: [String: String] = [:]
            for layer in (0..<48).filter({ $0 % 4 != 3 }) {
                let state = try GDNState(
                    offset: 256,
                    recurrent: (0..<48 * 128 * 128).map {
                        Float(bitPattern: 0x3e80_0000 | UInt32(($0 * 65537 + layer * 32771) & 0x007f_ffff))
                    },
                    convolutionTail: (0..<3 * 10240).map { UInt16(($0 + layer * 19) % 65536) })
                expected["\(layer).recurrent"] = state.withRecurrentBytes(digest)
                expected["\(layer).tail"] = state.convolutionTail.withUnsafeBytes(digest)
                try file.retain(layer: layer, state: state)
                try rejects { try file.retain(layer: layer, state: state) }
            }
            try file.seal()
            try file.withTensors { layer, name, bytes in
                try require(digest(bytes) == expected["\(layer).\(name)"], "Frontier file changed original tensor bits")
            }
            let current = ledger.snapshot().reserved
            let blocker = try ledger.reserve(bytes: 984_000_000 - current - 1_048_576, category: "frontier_fixture_denial")
            try rejects { _ = try file.restore() }
            try ledger.release(blocker)
            try require(ledger.snapshot().reserved == current, "Failed frontier read leaked transient ownership")
            try autoreleasepool {
                let restored = try file.restore()
                for (layer, state) in restored.states {
                    try require(
                        state.withRecurrentBytes(digest) == expected["\(layer).recurrent"]
                            && state.convolutionTail.withUnsafeBytes(digest) == expected["\(layer).tail"], "Restored recurrent bits differ")
                }
            }
            try require(ledger.snapshot().reserved == current, "Restoration lease survived its reader")
            try autoreleasepool {
                let files = try [512, 768].map { try store.make(offset: $0) }
                var expectedPair: [Int: [String: String]] = [:]
                for layer in (0..<48).filter({ $0 % 4 != 3 }) {
                    for (index, other) in files.enumerated() {
                        let state = try GDNState(
                            offset: other.offset,
                            recurrent: (0..<48 * 128 * 128).map {
                                Float(bitPattern: 0x3f00_0000 | UInt32(($0 * 40503 + layer * 977 + index * 7919) & 0x007f_ffff))
                            },
                            convolutionTail: (0..<3 * 10240).map { UInt16(($0 * 3 + layer * 29 + index) % 65536) })
                        expectedPair[index, default: [:]]["\(layer).recurrent"] = state.withRecurrentBytes(digest)
                        expectedPair[index, default: [:]]["\(layer).tail"] = state.convolutionTail.withUnsafeBytes(digest)
                        try other.retain(layer: layer, state: state)
                    }
                }
                for (index, other) in files.enumerated() {
                    try other.seal()
                    try other.withTensors { layer, name, bytes in
                        try require(
                            digest(bytes) == expectedPair[index]!["\(layer).\(name)"], "Interleaved frontier writer changed tensor bits")
                    }
                }
            }
            try require(ledger.snapshot().reserved == current, "Interleaved frontier files leaked ownership")
            try file.corruptFixture()
            try rejects { _ = try file.restore() }
            escaped = file
        }
        try require(
            ledger.snapshot().reserved == 64_000_000 + 131072 + ConversationFrontierStore.stagingReservationBytes + 262144,
            "Escaped file reader lost its store/allocation charge")
        try autoreleasepool {
            try rejects { try escaped!.validate(expectedOffset: 256) }
            escaped = nil
        }
        try require(ledger.snapshot().reserved == 64_000_000, "Final file reader did not release store metadata")
        try ledger.release(fixture)
        try require(ledger.snapshot().reserved == 0, "Frontier fixture retained memory ownership")
    }
}
