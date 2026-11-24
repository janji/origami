import Foundation
import zlib

/// What a hot copy needs from a model to write and check its records: a record's bytes as a slot holds it (nil for a
/// key the model does not have) and that record read from the model's own files. Keys are the engine's
/// (layer * experts + expert).
package protocol HotCopySource: AnyObject {
    func hotCopyRecordBytes(_ key: Int) -> Int?
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws
}

/// A copy of a model's most-used routed-expert records in one file on another drive (Origami's data folder, normally
/// the internal SSD), so the records a token misses are read from two drives at once. Records are stored as a slot
/// holds them at 16 KiB-aligned offsets; `index.json` names the model (`identity`) and the records in file order. They
/// are chosen from miss counts, most missed first, then by uses (`choose`), within the `hot_copy` size: ranking by uses
/// alone picked the records already resident, which served about a tenth of the misses. The bytes are the model's:
/// replies are unchanged (BITWISE). `HotCopyKeeper` builds and grows it while the model serves. Each record written
/// carries a CRC32 in the index, checked on its first read since open (`read`); a copy written before checksums is
/// read unchecked. A read that fails (a drive gone) stops all reads from the copy for `retryDelay` seconds, after which
/// it is reopened off the reading thread.
package final class HotCopy: @unchecked Sendable {
    /// The descriptor reads use (a new one once the copy is reopened after a read error).
    package var file: ExactFile { checking.withLock { opened } }
    package let payload: URL
    package let identity: String
    /// Record key (layer * experts + expert) to its offset in the file, and the keys in file order (empty slots left out).
    package var offsets: [Int: Int64] { checking.withLock { slotOf.mapValues { slotOffsets[$0] } } }
    package var keys: [Int] { checking.withLock { slotKeys.filter { $0 >= 0 } } }
    package let bytes: Int64
    /// Intended final capacity survives an interrupted bounded migration in automatic sizing.
    package let targetBytes: UInt64
    package let folder: URL
    /// Record key to the CRC32 of its bytes (records written before checksums have none).
    package var checksums: [Int: UInt32] {
        checking.withLock { slotOf.compactMapValues { slotSums[$0] } }
    }
    /// Records checked since open, and those that failed: read from the model's own files from then on.
    private let checking = NSLock()
    private var verified = Set<Int>(), rejected = Set<Int>()
    /// The slots in file order (`replace` rewrites one in place): each one's key (-1: empty, read from the model's files),
    /// offset, padded size and checksum; the slot of each key; reads in flight per slot. `version` 2 once a slot was
    /// replaced (empty slots need it); nil for the original index. A `retired` copy (replaced by another of the same
    /// payload) starts no read.
    private var slotKeys: [Int], slotSums: [UInt32?], slotOf: [Int: Int], pins: [Int: Int] = [:]
    private let slotOffsets: [Int64], slotBytes: [UInt64]
    private var version: Int?, retired = false
    private let attachedHold: Int32
    /// Opened with `attaching` false: the caller holds `.attached.lock` exclusive (`IdleHotCopy`).
    private let callerHoldsAttached: Bool
    /// Slots being changed (`vacate`), each by one owner (a swap or a give-back): only that owner puts it back
    /// (`restore`) or empties it on disk. `key` is what a restore puts back: the old key while nothing on disk has
    /// changed, -1 once the index says the slot is empty.
    private var vacated: [Int: (key: Int, sum: UInt32?, owner: Int)] = [:], owners = 0
    /// The index version this build writes once a slot is replaced; a newer one is refused (`open`).
    package static let slotVersion = 2
    package var rejectedRecords: Int { checking.withLock { rejected.count } }
    package var rejectedKeys: [Int] { checking.withLock { rejected.sorted() } }
    private var opened: ExactFile
    /// How much of the copy's reading went far below its drive's own healthy rate (R15: the internal drive can slow down
    /// on heavily read blocks as the model's drive did).
    package let reads = SlowDriveReads()
    /// The payload's name and birth time: a rebuild writes a new payload, on fresh blocks (`DriveReadHistory.generation`).
    package var generation: String {
        let born = (try? payload.resourceValues(forKeys: [.creationDateKey]))?.creationDate?.timeIntervalSince1970 ?? 0
        return "\(payload.lastPathComponent)@\(Int(born))"
    }
    /// After a read error no read touches the copy until then; the first read after it reopens the copy off its thread.
    private var disabledUntil: Date?, reopening = false
    private var delay: TimeInterval = 30
    /// Seconds a copy that failed a read stays unused before it is tried again (as `PackStore` does with its copies).
    package var retryDelay: TimeInterval {
        get { checking.withLock { delay } }
        set { checking.withLock { delay = newValue } }
    }
    /// Whether reads skip the copy now, after a read error.
    package var disabled: Bool { checking.withLock { disabledUntil != nil } }
    /// Whether a read of `key` would come from the copy now: it holds the record, has not rejected it and is not paused.
    /// A pause that is over starts the reopen here too, so a copy that reads are no longer offered to comes back.
    package func holds(_ key: Int) -> Bool {
        checking.withLock {
            reopenIfDueLocked()
            return slotOf[key] != nil && !retired && disabledUntil == nil && !rejected.contains(key)
        }
    }
    /// Starts the reopen, off the calling thread, once a pause after a read error is over (under `checking`).
    private func reopenIfDueLocked() {
        guard let until = disabledUntil, !reopening, Date() >= until else { return }
        reopening = true
        DispatchQueue.global(qos: .utility).async { [self] in reopen() }
    }

    private struct Index: Codable {
        /// nil: the original index (every key a record); 2: a key may be -1, an empty slot (`replace`).
        var version: Int? = nil
        let identity: String
        var keys: [Int]
        let offsets: [Int64]
        var byteCount: Int64? = nil
        var payload: String? = nil
        var targetBytes: UInt64? = nil
        /// Per key, in order; nil for a record written before checksums.
        var checksums: [UInt32?]? = nil
    }

    private init(
        folder: URL, identity: String, keys: [Int], offsets: [Int64], bytes: Int64, payload: String = "experts.bin",
        targetBytes: UInt64? = nil, checksums: [UInt32?] = [], version: Int? = nil, hold: Int32? = nil, callerHoldsAttached: Bool = false
    ) throws {
        self.identity = identity
        self.payload = folder.appendingPathComponent(payload)
        opened = try ExactFile(url: self.payload)
        self.folder = folder
        self.bytes = bytes
        self.targetBytes = max(UInt64(bytes), targetBytes ?? UInt64(bytes))
        self.version = version
        slotKeys = keys
        slotOffsets = offsets
        slotBytes = offsets.indices.map { UInt64((offsets.indices.contains($0 + 1) ? offsets[$0 + 1] : bytes) - offsets[$0]) }
        slotSums = checksums.count == keys.count ? checksums : Array(repeating: nil, count: keys.count)
        slotOf = Dictionary(uniqueKeysWithValues: keys.indices.filter { keys[$0] >= 0 }.map { (keys[$0], $0) })
        self.callerHoldsAttached = callerHoldsAttached
        // Last: a failed init never owns it (`open` closes what it took).
        attachedHold = hold ?? Self.holdAttached(folder)
    }
    deinit { if attachedHold >= 0 { close(attachedHold) } }

    /// A shared hold (flock) on `folder`'s `.attached.lock` for as long as a process may read the copy there: another
    /// model's server takes slots from an idle copy (`IdleHotCopy`) only with that lock exclusive, so never from a copy
    /// a process has open. -1 where the drive keeps no locks (the copy is then never taken from).
    static func holdAttached(_ folder: URL) -> Int32 {
        let fd = Darwin.open(folder.appendingPathComponent(".attached.lock").path, O_RDONLY | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { return -1 }
        while flock(fd, LOCK_SH) != 0 {
            guard errno == EINTR else {
                close(fd)
                return -1
            }
        }
        return fd
    }
    /// Whether a process other than this one has a payload in `folder` open (one built before `.attached.lock` existed
    /// holds no lock): its reads would not see slots emptied under it. The scan asks about every open file of every
    /// process, and waits for as long as any of them is blocked on its drive (minutes behind a busy USB 2 link), so it
    /// gets `seconds`: no answer by then, or an earlier scan still out, counts as open (fail closed).
    package static func openElsewhere(_ folder: URL, seconds: Double = 2, scan: @escaping @Sendable (String) -> Bool? = scanOpen) -> Bool {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        let payloads = names.filter(HotCopyRefreshPlan.validPayload).map { folder.appendingPathComponent($0).path }
        guard !payloads.isEmpty else { return false }
        guard scanGate.wait(timeout: .now()) == .success else { return true }
        let answer = ScanAnswer()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async {
            // Unknown (an error) counts as open: fail closed.
            answer.open = payloads.contains { scan($0) ?? true }
            scanGate.signal()
            done.signal()
        }
        return done.wait(timeout: .now() + seconds) == .success ? answer.open : true
    }
    private static let scanGate = DispatchSemaphore(value: 1)
    private final class ScanAnswer: @unchecked Sendable { var open = true }
    /// Whether a process other than this one has `path` open; nil when the system cannot say.
    package static let scanOpen: @Sendable (String) -> Bool? = { path in
        var pids = [Int32](repeating: 0, count: 256)
        let bytes = proc_listpidspath(UInt32(PROC_ALL_PIDS), 0, path, 0, &pids, Int32(pids.count * MemoryLayout<Int32>.size))
        guard bytes >= 0 else { return nil }
        return pids.prefix(Int(bytes) / MemoryLayout<Int32>.size).contains { $0 != getpid() && $0 != 0 }
    }
    /// An in-place change refused because another process (or another copy object of this folder) has the copy open.
    package struct SharedElsewhere: Error, CustomStringConvertible {
        package var description: String { "another Origami process has this copy open: it is not changed while that lasts" }
    }
    /// Runs an in-place change (`replace`, `giveBack`: slots rewritten or freed under the published payload) only while
    /// no other process has the copy open: every reader holds `.attached.lock` shared and keeps the index it read, so
    /// this copy's own hold is turned exclusive without waiting (a failed try keeps it shared), and back to shared after;
    /// a reader opening meanwhile waits for the change and reads the new index. A process without that lock (an older
    /// build) is found by its open payload (`openElsewhere`). Whole-file replacement (a new payload) needs neither.
    private func alone<T>(_ change: () throws -> T) throws -> T {
        if !callerHoldsAttached {
            guard attachedHold >= 0, flock(attachedHold, LOCK_EX | LOCK_NB) == 0 else { throw SharedElsewhere() }
        }
        defer { if !callerHoldsAttached { while flock(attachedHold, LOCK_SH) != 0 && errno == EINTR {} } }
        guard !Self.openElsewhere(folder) else { throw SharedElsewhere() }
        return try change()
    }

    static func checksum(_ bytes: UnsafeRawBufferPointer) -> UInt32 {
        UInt32(crc32(0, bytes.baseAddress?.assumingMemoryBound(to: Bytef.self), uInt(bytes.count)))
    }

    /// Record `key` (`into.count` bytes, as a slot holds it), checked against its checksum on its first read since
    /// open. A record that does not match is dropped: this read and every later one throw, so the caller reads the
    /// same bytes from the model's own files. A read that fails stops every read for `retryDelay` seconds (they throw at
    /// once); the first read after that reopens the copy off its thread, and reads use it again once its file reads.
    package func read(_ key: Int, into: UnsafeMutableRawBufferPointer) throws {
        // The slot stays pinned until this read and its check are done: `replace` rewrites a slot only once none is.
        let (slot, expected) = try pinnedRead(key, at: 0, into: into)
        guard let expected else { return checking.withLock { unpinLocked(slot) } }
        try check(key, expected, UnsafeRawBufferPointer(into), unpinning: slot)
    }

    /// Bytes `at ..< at + into.count` of record `key` (one part of a split read), under `read`'s pause, rejection and
    /// slot-pin rules (the slot is pinned while the bytes are read, so a swap never rewrites it under the read); the
    /// checksum the assembled record must match when its first-read check is due (nil: nothing to check), for `verify`.
    @discardableResult
    package func read(_ key: Int, at: Int, into: UnsafeMutableRawBufferPointer) throws -> UInt32? {
        let (slot, expected) = try pinnedRead(key, at: at, into: into)
        checking.withLock { unpinLocked(slot) }
        return expected
    }

    /// A split read's record `key`, assembled from parts read here and from other sources holding the same bytes, checked
    /// against `sum` (what `read(_:at:into:)` returned): a match counts as the record's first-read check; a mismatch
    /// throws without dropping the record (the wrong bytes may be another source's), so the caller reads the record
    /// whole, and that read's own check finds and drops whichever source is wrong.
    package func verify(_ key: Int, sum: UInt32, record: UnsafeRawBufferPointer) throws {
        guard Self.checksum(record) == sum else { throw EngineError("Split hot-copy record \(key) failed its checksum") }
        _ = checking.withLock { verified.insert(key) }
    }

    /// Reads bytes `at ..<` of `key`'s slot with the slot pinned (still pinned on return; unpinned when it throws): the
    /// slot, and its checksum when the key's first-read check is due.
    private func pinnedRead(_ key: Int, at: Int, into: UnsafeMutableRawBufferPointer) throws -> (Int, UInt32?) {
        let (file, slot, offset, expected, unchecked) = try checking.withLock {
            guard let slot = slotOf[key], !retired else { throw EngineError("The hot copy holds no record \(key)") }
            if disabledUntil != nil {
                reopenIfDueLocked()
                throw EngineError("The hot copy at \(folder.path) is not read after a read error")
            }
            guard !rejected.contains(key) else { throw EngineError("Hot-copy record \(key) failed its checksum") }
            pins[slot, default: 0] += 1
            return (opened, slot, slotOffsets[slot], slotSums[slot], slotSums[slot] != nil && !verified.contains(key))
        }
        reads.begin()
        var read = 0.0
        defer { reads.end(bytes: read) }
        do {
            try file.read(offset: offset + Int64(at), into: into)
            read = Double(into.count)
        } catch {
            let first = checking.withLock {
                unpinLocked(slot)
                // A read still in flight on a descriptor already replaced does not stop the new one.
                guard disabledUntil == nil, opened === file else { return false }
                disabledUntil = Date().addingTimeInterval(delay)
                return true
            }
            if first { Self.log("hot copy at \(folder.path) not read for \(Int(retryDelay)) s after a read error (\(error))") }
            throw error
        }
        return (slot, unchecked ? expected : nil)
    }

    private func check(_ key: Int, _ expected: UInt32, _ record: UnsafeRawBufferPointer, unpinning slot: Int?) throws {
        let matches = Self.checksum(record) == expected
        let first = checking.withLock {
            if let slot { unpinLocked(slot) }
            if matches { verified.insert(key) }
            return !matches && rejected.insert(key).inserted
        }
        // Once per record: a rejected record is never read from the copy again.
        if first { Self.log("hot copy record \(key) in \(payload.path) failed its checksum; read from the model's files until repaired") }
        guard matches else { throw EngineError("Hot-copy record \(key) failed its checksum") }
    }

    /// Writes record `key`'s bytes from the model over its place in the copy (after it failed its checksum), when they
    /// match its stored checksum; false otherwise. The index and every offset stay as they are, so a reader that opened
    /// the copy earlier stays valid: one that rejected the record keeps reading it from the model, and any other read
    /// gets the bytes its checksum describes. A fresh `open` reads the record from the copy again.
    package func repair(_ key: Int, source: HotCopySource) throws -> Bool {
        let (offset, sum) = checking.withLock { slotOf[key].map { (slotOffsets[$0], slotSums[$0]) } ?? (0, nil) }
        guard let expected = sum, let count = source.hotCopyRecordBytes(key) else { return false }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 16384)
        defer { buffer.deallocate() }
        try source.readHotCopyRecord(key, into: buffer)
        guard Self.checksum(UnsafeRawBufferPointer(start: buffer, count: count)) == expected else { return false }
        let lock = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
        guard lock >= 0 else { throw EngineError("Cannot open the hot-copy lock: errno \(errno)") }
        defer {
            _ = flock(lock, LOCK_UN)
            close(lock)
        }
        guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { throw EngineError("Another hot-copy writer owns this folder") }
        let descriptor = Darwin.open(payload.path, O_WRONLY)
        guard descriptor >= 0 else { throw EngineError("Cannot open the hot-copy payload: errno \(errno)") }
        defer { close(descriptor) }
        _ = fcntl(descriptor, F_NOCACHE, 1)
        guard pwrite(descriptor, buffer, count, off_t(offset)) == count, PersistentCheckpointStore.fullSync(descriptor) else {
            throw EngineError("Cannot repair hot-copy record \(key): errno \(errno)")
        }
        Self.log("hot copy record \(key) in \(payload.path) repaired from the model's files")
        return true
    }

    /// `origami refresh --copy`: writes the payload back in place, byte for byte (`ModelRefresh.rewrite`: read, written
    /// to the same place, read again, and the SHA-256 of both passes compared), so its drive stores it on fresh blocks
    /// when it reads the copy slowly (R15) without the copy being built again. The index stays as it is: every write
    /// repeats the bytes already there, so stopping it part-way leaves the copy as it was. Only while no other process
    /// has the copy open (`alone`) and no writer owns the folder.
    package func rewriteInPlace(progress: (Int64) -> Void) throws {
        try alone {
            let lock = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
            guard lock >= 0 else { throw EngineError("Cannot open the hot-copy lock: errno \(errno)") }
            defer {
                _ = flock(lock, LOCK_UN)
                close(lock)
            }
            guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { throw EngineError("Another hot-copy writer owns this folder") }
            var info = stat()
            guard stat(payload.path, &info) == 0 else { throw EngineError("Cannot stat \(payload.path)") }
            _ = try ModelRefresh.rewrite(
                payload.path, modified: [Int64(info.st_mtimespec.tv_sec), Int64(info.st_mtimespec.tv_nsec)], hook: nil, progress: progress)
        }
    }

    /// Step 1 of `replace`, which the keeper takes alone while no request runs (then `drain` with no request held): the
    /// slot (held or empty) becomes the caller's to change, and its key leaves the lookup, so no new read pins it. The
    /// owner it returns (nil: another owner is changing the slot) is the only one that may `restore` it, which puts the
    /// key back until the slot is emptied on disk.
    @discardableResult
    package func vacate(_ slot: Int) -> Int? {
        checking.withLock {
            guard slotKeys.indices.contains(slot), vacated[slot] == nil else { return nil }
            owners += 1
            vacated[slot] = (slotKeys[slot], slotSums[slot], owners)
            if slotKeys[slot] >= 0 { slotOf[slotKeys[slot]] = nil }
            slotKeys[slot] = -1
            slotSums[slot] = nil
            return owners
        }
    }
    /// Whether the reads in flight on `slot` all finished within `seconds` (a hung pread cannot be cancelled).
    package func drain(_ slot: Int, within seconds: Double) -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while checking.withLock({ pins[slot] != nil }) {
            guard Date() < end else { return false }
            Thread.sleep(forTimeInterval: 0.001)
        }
        return true
    }
    /// How long `drain` waits: the demand read bound for the slot's bytes (`ExpertReadStall`).
    package func drainBound(_ slot: Int) -> Double {
        ExpertReadStall.bound(bytes: Int(checking.withLock { slotBytes[slot] }), bytesPerSecond: 0)
    }
    /// Ends `owner`'s change of `slot`: its old key and checksum go back while nothing on disk changed for it. Nothing for
    /// any other owner.
    package func restore(_ slot: Int, owner: Int) {
        checking.withLock {
            guard vacated[slot]?.owner == owner, let old = vacated.removeValue(forKey: slot), old.key >= 0, slotKeys[slot] == -1,
                slotOf[old.key] == nil
            else { return }
            slotKeys[slot] = old.key
            slotSums[slot] = old.sum
            slotOf[old.key] = slot
        }
    }
    /// After the index says `owner`'s slots are empty: nothing goes back into the lookup for them any more.
    private func emptiedLocked(_ slots: [(slot: Int, owner: Int)]) {
        for (slot, owner) in slots where vacated[slot]?.owner == owner { vacated[slot] = (-1, nil, owner) }
    }
    /// Pins `key`'s slot as a read in flight does, until the returned closure runs (checks: a read that never returns).
    package func pinRead(_ key: Int) -> (() -> Void)? {
        checking.withLock {
            guard let slot = slotOf[key] else { return nil }
            pins[slot, default: 0] += 1
            return { [self] in checking.withLock { unpinLocked(slot) } }
        }
    }

    private func unpinLocked(_ slot: Int) {
        pins[slot, default: 1] -= 1
        if pins[slot] == 0 { pins[slot] = nil }
    }
    /// Starts no read from now on (reads in flight finish): the keeper attached another copy of the same payload, whose
    /// slots it may rewrite once this one's `readsInFlight` is 0.
    package func retire() { checking.withLock { retired = true } }
    package var readsInFlight: Int { checking.withLock { pins.values.reduce(0, +) } }
    /// The slots in file order: each one's key (-1: empty) and padded size.
    package var slots: [(key: Int, bytes: UInt64)] { checking.withLock { zip(slotKeys, slotBytes).map { ($0, $1) } } }
    /// Whether the index on disk is the slot form (`slotVersion`) rather than the original one.
    package var slotIndex: Bool { checking.withLock { version != nil } }
    /// Bytes the payload takes on its drive now: its length less the slots given back (`giveBack`).
    package var diskBytes: UInt64 {
        var info = stat()
        guard stat(payload.path, &info) == 0 else { return UInt64(bytes) }
        return min(UInt64(bytes), UInt64(info.st_blocks) * 512)
    }
    /// Empty slots (no key, none being swapped) whose bytes are still on the drive (a swap that stopped): the first a
    /// give-back frees, since they serve nothing.
    package var emptyHeld: [Int] {
        let descriptor = Darwin.open(payload.path, O_RDONLY)
        guard descriptor >= 0 else { return [] }
        defer { close(descriptor) }
        let empty = checking.withLock { slotKeys.indices.filter { slotKeys[$0] < 0 && vacated[$0] == nil } }
        return empty.filter { slot in
            let data = lseek(descriptor, off_t(slotOffsets[slot]), SEEK_DATA)
            return data >= 0 && data < off_t(slotOffsets[slot]) + off_t(slotBytes[slot])
        }
    }

    /// Gives the drive space of `chosen` slots back (`HotCopyKeeper.giveBack`, for a save that lacks room), in this order,
    /// so that a stop at any point leaves a copy that opens with every mapped key on its own bytes:
    /// 1. each slot holding a record leaves the lookup (its reads go to the model's files) and its reads in flight finish
    ///    (bounded; a slot whose read does not return is kept);
    /// 2. the index is published with those slots empty (key -1, `slotVersion`);
    /// 3. only then are their blocks freed (`F_PUNCHHOLE`): the file keeps its length and every other slot its offset and
    ///    bytes. A later swap (`replace`) writes a record into such a slot again.
    /// Each slot is this give-back's from step 1 on (`vacate`'s owner): a slot another writer is changing is left alone,
    /// and no other writer can put a key back into one, so a freed slot is never mapped again until a swap has written
    /// and verified it. Returns the slots freed, each with the key it held (-1: it was empty); step 3 failing throws.
    /// Any thread: the lookup and the index change under the copy's locks, its writer lock, and with no other process
    /// having it open (`alone`; else `SharedElsewhere`).
    /// Model-free checks stop a give-back at a step ("vacated", "published", "freeing") as a crash there would.
    nonisolated(unsafe) package static var stopForChecks: ((_ step: String) -> Bool)?
    /// Model-free checks set whether a drive frees part of a file; production probes it (`punchable`).
    nonisolated(unsafe) package static var punchForChecks: Bool?
    private var punchProbe: Bool?
    /// Whether this copy's drive frees part of a file (`F_PUNCHHOLE`; HFS+ and exFAT do not), probed once per copy on a
    /// scratch file beside the payload: where it does not, no record is emptied for a save, since none would free space.
    package var punchable: Bool {
        if let known = checking.withLock({ punchProbe }) { return known }
        let probe = folder.appendingPathComponent(".punch-probe")
        let works: Bool =
            Self.punchForChecks
            ?? {
                let descriptor = Darwin.open(probe.path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR)
                guard descriptor >= 0 else { return false }
                defer {
                    close(descriptor)
                    unlink(probe.path)
                }
                let size = 1 << 20
                let block = [UInt8](repeating: 1, count: size)
                guard block.withUnsafeBytes({ Darwin.write(descriptor, $0.baseAddress, size) }) == size else { return false }
                var hole = fpunchhole_t(fp_flags: 0, reserved: 0, fp_offset: 0, fp_length: off_t(size / 2))
                return fcntl(descriptor, F_PUNCHHOLE, &hole) == 0
            }()
        checking.withLock { punchProbe = works }
        return works
    }
    package func giveBack(_ chosen: [Int]) throws -> [(slot: Int, key: Int)] {
        guard punchable else { throw EngineError("This drive cannot free part of a file; the hot copy gives nothing back here") }
        let lock = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
        guard lock >= 0 else { throw EngineError("Cannot open the hot-copy lock: errno \(errno)") }
        defer {
            _ = flock(lock, LOCK_UN)
            close(lock)
        }
        // A swap holds the writer lock for one record; a build or top-up for its whole run (then nothing is given back).
        let end = Date().addingTimeInterval(2)
        while flock(lock, LOCK_EX | LOCK_NB) != 0 {
            guard Date() < end else { throw EngineError("Another hot-copy writer owns this folder") }
            Thread.sleep(forTimeInterval: 0.01)
        }
        var mine: [(slot: Int, owner: Int, key: Int)] = []
        // Ends this give-back's ownership: before the index the keys go back; after it the slots stay empty.
        defer { for one in mine { restore(one.slot, owner: one.owner) } }
        for slot in Set(chosen).sorted() {
            guard let owner = vacate(slot) else { continue }
            let key = checking.withLock { vacated[slot]?.key ?? -1 }
            if drain(slot, within: drainBound(slot)) { mine.append((slot, owner, key)) } else { restore(slot, owner: owner) }
        }
        guard !mine.isEmpty else { return [] }
        if Self.stopForChecks?("vacated") == true { throw EngineError("stopped for a check before the index") }
        return try alone {
            let emptied = checking.withLock { () -> Index in
                var index = indexLocked()
                index.version = Self.slotVersion
                for one in mine { index.keys[one.slot] = -1 }
                return index
            }
            // Not on disk on a throw: the records stay where they are.
            try Self.publish(emptied, folder: folder) {
                checking.withLock {
                    emptiedLocked(mine.map { ($0.slot, $0.owner) })
                    version = Self.slotVersion
                }
            }
            let descriptor = Darwin.open(payload.path, O_WRONLY)
            guard descriptor >= 0 else { throw EngineError("Cannot open the hot-copy payload to free slots: errno \(errno)") }
            defer { close(descriptor) }
            var freed: [(slot: Int, key: Int)] = []
            for (n, one) in mine.enumerated() {
                if Self.stopForChecks?(n == 0 ? "published" : "freeing") == true { throw EngineError("stopped for a check while freeing") }
                // Only a slot still empty and still this give-back's: one whose key came back keeps its bytes and index.
                guard checking.withLock({ vacated[one.slot]?.owner == one.owner && slotKeys[one.slot] < 0 }) else { continue }
                var hole = fpunchhole_t(
                    fp_flags: 0, reserved: 0, fp_offset: off_t(slotOffsets[one.slot]), fp_length: off_t(slotBytes[one.slot]))
                guard fcntl(descriptor, F_PUNCHHOLE, &hole) == 0 else {
                    throw EngineError("Cannot free hot-copy slot \(one.slot) (errno \(errno)); it stays empty")
                }
                freed.append((one.slot, one.key))
            }
            return freed
        }
    }
    private func indexLocked() -> Index {
        Index(
            version: version, identity: identity, keys: slotKeys, offsets: slotOffsets, byteCount: bytes,
            payload: payload.lastPathComponent, targetBytes: targetBytes, checksums: slotSums)
    }

    /// Rewrites slot `slot` in place with record `key` from the model's files (its padded size must be the slot's), in
    /// this order, so that a stop at any point leaves a copy that opens with every mapped key on its own bytes:
    /// 1. the slot leaves the lookup (reads of its old key go to the model's files) and its reads in flight finish;
    /// 2. the index is published with the slot empty (key -1, `slotVersion`); until then its old bytes are intact;
    /// 3. the record is written and flushed (`F_FULLFSYNC`); 4. read back and checked against its CRC32. The index says
    ///    empty meanwhile, so torn or unflushed bytes are never mapped;
    /// 5. the index is published with the new key and checksum; only then does the lookup hold it.
    /// A failure after 1 leaves the slot empty (a later swap fills it). `owner`: the caller's `vacate` of the slot (the
    /// keeper's, taken between requests); nil: this call takes it. `step` is called after each numbered step (a check
    /// stops there to simulate a kill). Caller: the keeper's thread, serialized with request admission. Only while no
    /// other process has the copy open (`alone`; else `SharedElsewhere`).
    package func replace(
        slot: Int, with key: Int, source: HotCopySource, within seconds: Double? = nil, owner: Int? = nil,
        step: (Int) throws -> Void = { _ in }
    ) throws {
        // Ends the change: until the empty slot is on disk its old key goes back into the lookup; after, it stays empty.
        var owner = owner
        defer { if let owner { restore(slot, owner: owner) } }
        guard key >= 0, let count = source.hotCopyRecordBytes(key), let padded = Self.size(key, source: source),
            checking.withLock({ slotOf[key] == nil && slotBytes.indices.contains(slot) && slotBytes[slot] == padded })
        else { throw EngineError("Hot-copy record \(key) does not fit slot \(slot)") }
        let lock = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
        guard lock >= 0 else { throw EngineError("Cannot open the hot-copy lock: errno \(errno)") }
        defer {
            _ = flock(lock, LOCK_UN)
            close(lock)
        }
        guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { throw EngineError("Another hot-copy writer owns this folder") }
        if owner == nil { owner = vacate(slot) }
        guard let mine = owner, checking.withLock({ vacated[slot]?.owner == mine }) else {
            throw EngineError("Hot-copy slot \(slot) is being changed by another writer")
        }
        guard drain(slot, within: seconds ?? drainBound(slot)) else {
            Self.log("hot copy at \(folder.path): a read of slot \(slot) did not return; its swap is abandoned and its record kept")
            throw EngineError("A read of hot-copy slot \(slot) did not return; the swap is abandoned")
        }
        try alone { try rewrite(slot: slot, with: key, count: count, padded: padded, source: source, owner: mine, step: step) }
    }
    private func rewrite(
        slot: Int, with key: Int, count: Int, padded: UInt64, source: HotCopySource, owner: Int, step: (Int) throws -> Void
    ) throws {
        let emptied = checking.withLock { () -> Index in
            var index = indexLocked()
            index.version = Self.slotVersion
            return index
        }
        try step(1)
        try Self.publish(emptied, folder: folder) {
            checking.withLock {
                emptiedLocked([(slot, owner)])
                version = Self.slotVersion
            }
        }
        try step(2)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: Int(padded), alignment: 16384)
        defer { buffer.deallocate() }
        memset(buffer, 0, Int(padded))
        try source.readHotCopyRecord(key, into: buffer)
        let sum = Self.checksum(UnsafeRawBufferPointer(start: buffer, count: count))
        let descriptor = Darwin.open(payload.path, O_WRONLY)
        guard descriptor >= 0 else { throw EngineError("Cannot open the hot-copy payload: errno \(errno)") }
        defer { close(descriptor) }
        _ = fcntl(descriptor, F_NOCACHE, 1)
        let offset = slotOffsets[slot]
        guard pwrite(descriptor, buffer, Int(padded), off_t(offset)) == Int(padded), PersistentCheckpointStore.fullSync(descriptor)
        else { throw EngineError("Cannot write hot-copy record \(key): errno \(errno)") }
        try step(3)
        let back = UnsafeMutableRawBufferPointer.allocate(byteCount: count, alignment: 16384)
        defer { back.deallocate() }
        try file.read(offset: offset, into: back)
        guard Self.checksum(UnsafeRawBufferPointer(back)) == sum else {
            throw EngineError("Hot-copy record \(key) read back differs from what was written; its slot stays empty")
        }
        try step(4)
        let filled = checking.withLock { () -> Index in
            var index = indexLocked()
            index.keys[slot] = key
            index.checksums![slot] = sum
            return index
        }
        try Self.publish(filled, folder: folder)
        checking.withLock {
            vacated[slot] = nil
            slotKeys[slot] = key
            slotSums[slot] = sum
            slotOf[key] = slot
            verified.insert(key)
            rejected.remove(key)
        }
        try step(5)
    }

    /// The index as `PagedCheckpointStore` publishes: flushed beside, renamed over (then `renamed`), the folder flushed.
    private static func publish(_ index: Index, folder: URL, renamed: () -> Void = {}) throws {
        let staging = folder.appendingPathComponent(".index.json.staging")
        try JSONEncoder().encode(index).write(to: staging)
        let staged = try FileHandle(forWritingTo: staging)
        defer { try? staged.close() }
        guard PersistentCheckpointStore.fullSync(staged.fileDescriptor),
            rename(staging.path, folder.appendingPathComponent("index.json").path) == 0
        else { throw EngineError("Cannot publish the hot-copy index: errno \(errno)") }
        renamed()
        let directory = Darwin.open(folder.path, O_RDONLY)
        defer { if directory >= 0 { close(directory) } }
        guard directory >= 0, PersistentCheckpointStore.fullSync(directory) else {
            throw EngineError("Cannot flush the hot-copy folder: errno \(errno)")
        }
    }

    /// Reopens the payload (the same file, at least as long as the copy) and reads its first block: on success reads use
    /// the copy again, else it waits another `retryDelay`.
    private func reopen() {
        let first = UnsafeMutableRawBufferPointer.allocate(byteCount: 16384, alignment: 16384)
        defer { first.deallocate() }
        let original = checking.withLock { opened.identity.inode }
        let next = try? { () throws -> ExactFile in
            let file = try ExactFile(url: payload)
            guard file.identity.inode == original, file.identity.size >= UInt64(bytes) else {
                throw EngineError("The hot copy's file was replaced")
            }
            try file.read(offset: 0, into: first)
            return file
        }()
        // Told before the first read can use it again.
        if next != nil { Self.log("hot copy at \(folder.path) is read again") }
        checking.withLock {
            reopening = false
            if let next {
                opened = next
                disabledUntil = nil
            } else {
                disabledUntil = Date().addingTimeInterval(delay)
            }
        }
    }
    /// Where a copy's events go (a read error pausing it, its return, a record rejected or repaired): the server's log.
    nonisolated(unsafe) package static var log: @Sendable (String) -> Void = {
        FileHandle.standardError.write(Data("origami: \($0)\n".utf8))
    }

    /// `origami verify`: every record of the copy in `folder` re-read and checked against its CRC32 (`recordBytes`: a
    /// record's bytes, as `HotCopySource` gives them). The records checked, those written before checksums, and the
    /// problems; nil when there is no copy of `identity` there.
    package static func verify(folder: URL, identity: String, recordBytes: (Int) -> Int?) -> (
        checked: Int, unchecked: Int, problems: [String]
    )? {
        guard let copy = open(folder: folder, identity: identity) else { return nil }
        var problems: [String] = []
        var checked = 0
        let (keys, sums, offsets) = (copy.keys, copy.checksums, copy.offsets)
        for key in keys {
            guard let expected = sums[key] else { continue }
            guard let count = recordBytes(key) else {
                problems.append("record \(key): the model has no such record")
                continue
            }
            do {
                var bytes = try copy.file.read(offset: offsets[key]!, length: count)
                if bytes.withUnsafeBytes({ checksum($0) }) != expected {
                    // The server may have swapped this slot since the index was read: a mismatch counts only if the
                    // index read again still maps the key there with that checksum, and the bytes read again differ.
                    guard let again = open(folder: folder, identity: identity), again.offsets[key] == offsets[key],
                        again.checksums[key] == expected
                    else { continue }
                    bytes = try again.file.read(offset: offsets[key]!, length: count)
                    if bytes.withUnsafeBytes({ checksum($0) }) != expected { problems.append("record \(key) differs from its checksum") }
                }
                checked += 1
            } catch { problems.append("record \(key): \(error)") }
        }
        return (checked, keys.count - sums.count, problems)
    }

    /// Use (or miss) counts saved in `url` (empty when there are none or the file is unreadable).
    package static func loadUses(_ url: URL) -> [Int: Int] {
        guard let data = try? Data(contentsOf: url), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            return [:]
        }
        // Distinct JSON names such as "1" and "01" must not become duplicate Int keys and trap.
        // Writers use canonical nonnegative names; ignore malformed entries as for unreadable counts.
        var result: [Int: Int] = [:]
        for (name, count) in object {
            guard let key = Int(name), key >= 0, String(key) == name, count > 0 else { continue }
            result[key] = count
        }
        return result
    }
    /// These cumulative ranking hints must not make a successful reply fail at integer overflow.
    package static func mergeUses(_ earlier: [Int: Int], _ current: [Int: Int]) -> [Int: Int] {
        earlier.merging(current) { a, b in
            let sum = a.addingReportingOverflow(b)
            return sum.overflow ? Int.max : sum.partialValue
        }
    }
    /// Counts decay so a copy follows a changed workload: once a model's counts cover `2 * decayTokens` tokens, every
    /// use and miss halves (rounded up, so a counted key stays), back to `decayTokens`: twice the
    /// `RoutingProfile.minimumTokens` the ranking is trusted from, so a halving never leaves too few counts to rank by.
    /// The halving changes values only: the files stay readable by earlier builds. Counted in tokens (Qwen's layer
    /// windows), or in uses for the family engines (`tokens * usesPerToken`), at any quality level: a fast level
    /// counts the experts it reads, which are the reads a copy serves.
    package static let decayTokens = 2 * RoutingProfile.minimumTokens
    /// `uses` and `misses` halved as often as their uses reach `2 * floor` (nil: below it, unchanged).
    package static func halved(_ uses: [Int: Int], _ misses: [Int: Int], floor: Int) -> (uses: [Int: Int], misses: [Int: Int])? {
        guard floor > 0, uses.values.reduce(0, &+) >= 2 * floor else { return nil }
        var (uses, misses) = (uses, misses)
        while true {
            let total = uses.values.reduce(0, &+)
            guard total >= 2 * floor else { return (uses, misses) }
            uses = uses.mapValues { $0 / 2 + $0 % 2 }
            misses = misses.mapValues { $0 / 2 + $0 % 2 }
            // Only keys counted once are left: halving cannot lower the total further.
            if uses.values.reduce(0, &+) == total { return (uses, misses) }
        }
    }
    /// `now` less `base` per key (a model's counts since its last halving; both only grow).
    package static func added(_ now: [Int: Int], since base: [Int: Int]) -> [Int: Int] {
        now.reduce(into: [:]) { result, entry in
            let more = entry.value - (base[entry.key] ?? 0)
            if more > 0 { result[entry.key] = more }
        }
    }
    package static func saveUses(_ uses: [Int: Int], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let object = Dictionary(uniqueKeysWithValues: uses.map { (String($0.key), $0.value) })
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url, options: .atomic)
    }

    /// A record's bytes in a copy: its size rounded up to 16 KiB (nil for a key the model does not have).
    package static func size(_ key: Int, source: HotCopySource) -> UInt64? {
        source.hotCopyRecordBytes(key).map { UInt64(($0 + 16383) / 16384 * 16384) }
    }

    /// The records a copy of `budget` bytes holds: the most missed first, then the most used (ties by key), each taking
    /// its record size rounded up to 16 KiB, skipping keys the model does not have.
    package static func choose(_ uses: [Int: Int], misses: [Int: Int] = [:], source: HotCopySource, budget: UInt64) -> [Int] {
        var chosen: [Int] = []
        var total: UInt64 = 0
        let rank = Set(uses.keys).union(misses.keys).map { (key: $0, misses: misses[$0] ?? 0, uses: uses[$0] ?? 0) }
        for key in rank.sorted(by: { ($0.misses, $0.uses, $1.key) > ($1.misses, $1.uses, $0.key) }).map(\.key) {
            guard let size = size(key, source: source), total + size <= budget else { continue }
            chosen.append(key)
            total += size
        }
        return chosen
    }

    /// The finished copy in `folder` for the model named `identity`, or nil (none, another model's, or a file shorter
    /// than its index says).
    /// `attaching` false (`IdleHotCopy`, which holds the folder's `.attached.lock` exclusive itself): no shared hold.
    package static func open(folder: URL, identity: String, source: HotCopySource? = nil, attaching: Bool = true) -> HotCopy? {
        // Held before the index is read: a give-back by another model's server (under the exclusive lock) is either
        // wholly before this read or wholly after this copy is closed.
        let hold = attaching ? holdAttached(folder) : -1
        var kept = false
        defer { if !kept, hold >= 0 { close(hold) } }
        guard
            let index = (try? Data(contentsOf: folder.appendingPathComponent("index.json"))).flatMap({
                try? JSONDecoder().decode(Index.self, from: $0)
            }),
            // A newer index is refused (the keeper then builds a new copy); -1 (an empty slot) only in the slot form.
            index.version.map({ $0 == slotVersion }) ?? true, index.identity == identity, !index.keys.isEmpty,
            Set(index.keys.filter { $0 >= 0 }).count == index.keys.filter({ $0 >= 0 }).count,
            index.keys.allSatisfy({ $0 >= 0 || ($0 == -1 && index.version != nil) }),
            index.keys.count == index.offsets.count, index.checksums.map({ $0.count == index.keys.count }) ?? true,
            HotCopyRefreshPlan.validPayload(index.payload ?? "experts.bin"),
            let size =
                ((try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent(index.payload ?? "experts.bin").path))?[
                    .size] as? NSNumber)?.int64Value,
            index.offsets.allSatisfy({ $0 >= 0 && $0 < size })
        else { return nil }
        let held = index.byteCount ?? size
        guard held > 0, held <= size, index.offsets.first == 0,
            index.offsets.allSatisfy({ $0 < held && $0 % 16384 == 0 }),
            zip(index.offsets, index.offsets.dropFirst()).allSatisfy({ $0 < $1 })
        else { return nil }
        if let source {
            var end: UInt64 = 0
            for (n, (key, offset)) in zip(index.keys, index.offsets).enumerated() {
                // An empty slot spans to the next slot's offset (or the end).
                let span = key < 0 ? UInt64((index.offsets.indices.contains(n + 1) ? index.offsets[n + 1] : held) - offset) : nil
                guard let bytes = span ?? Self.size(key, source: source), UInt64(offset) == end,
                    bytes <= UInt64(held) - end
                else { return nil }
                end += bytes
            }
            // A published prefix may have an unfinished append beyond it, but its own records must
            // be contiguous and complete. Monotonic offsets alone admit overlapping, incorrect weights.
            guard end == UInt64(held) else { return nil }
        }
        let copy = try? HotCopy(
            folder: folder, identity: identity, keys: index.keys, offsets: index.offsets, bytes: held,
            payload: index.payload ?? "experts.bin", targetBytes: index.targetBytes, checksums: index.checksums ?? [],
            version: index.version, hold: hold, callerHoldsAttached: !attaching)
        kept = copy != nil
        return copy
    }

    /// Default full build/top-up policy with the same safe publication transaction as bounded
    /// refresh: unique replacement payload, atomic index, immutable old reader extents. The
    /// caller's existing placement policy governs capacity; bounded staging remains opt-in.
    package static func write(
        source: HotCopySource, keys: [Int], folder: URL, identity: String, onto: HotCopy? = nil,
        wait: () throws -> Void = {}, perform: (_ work: () throws -> Void) throws -> Void = { try $0() },
        wrote: (Int64) -> Void = { _ in }
    ) throws -> HotCopy {
        try writeIncremental(
            source: source, keys: keys, folder: folder, identity: identity, onto: onto,
            reserve: 0, freeBytes: { UInt64.max }, wait: wait, perform: perform, wrote: wrote)
    }

    /// Reclaims unpublished generations without allocating, so a killed build cannot permanently
    /// consume the space its retry needs. Existing writer ownership is required; an absent
    /// manifest permits only orphan generated payloads, while a mismatched/invalid manifest is retained.
    package static func recoverIncremental(folder: URL, identity: String) throws {
        let descriptor = Darwin.open(folder.appendingPathComponent(".refresh.lock").path, O_WRONLY)
        guard descriptor >= 0 else { return }
        defer {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { throw EngineError("Another hot-copy writer owns recovery") }
        let indexURL = folder.appendingPathComponent("index.json")
        var kept: String?
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let data = try? Data(contentsOf: indexURL), let index = try? JSONDecoder().decode(Index.self, from: data),
                index.identity == identity, HotCopyRefreshPlan.validPayload(index.payload ?? "experts.bin")
            else { return }
            kept = index.payload ?? "experts.bin"
        }
        for url in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) {
            // Without a manifest, uniquely named payloads are interrupted first generations.
            // Preserve a legacy payload lacking its index for diagnosis/recovery outside here.
            if HotCopyRefreshPlan.validPayload(url.lastPathComponent), url.lastPathComponent != kept,
                kept != nil || url.lastPathComponent != "experts.bin"
            {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    /// Opt-in bounded refresh. A replacement gets a unique immutable payload name and an atomic
    /// manifest; an append writes only beyond the currently published extent. Old descriptors
    /// remain readable after unlink. Caller serializes every `perform` with request acquisition.
    /// `readMissing` and `freeBytes` are injectable for tiny-file, model-free failure tests.
    package static func writeIncremental(
        source: HotCopySource, keys: [Int], folder: URL, identity: String,
        onto: HotCopy? = nil, reuse: HotCopy? = nil, targetBytes: UInt64? = nil,
        reserve: UInt64 = HotCopyKeeper.keepFree,
        freeBytes: (() throws -> UInt64)? = nil,
        wait: () throws -> Void = {},
        perform: (_ work: () throws -> Void) throws -> Void = { try $0() },
        wrote: (Int64) -> Void = { _ in },
        readMissing: ((_ key: Int, _ into: UnsafeMutableRawPointer) throws -> Void)? = nil
    ) throws -> HotCopy {
        guard !keys.isEmpty, Set(keys).count == keys.count, keys.allSatisfy({ $0 >= 0 }),
            onto == nil
                || (onto!.identity == identity && onto!.folder.standardizedFileURL == folder.standardizedFileURL
                    && Set(onto!.keys).isDisjoint(with: keys)),
            reuse == nil || reuse!.identity == identity
        else {
            throw EngineError("Invalid bounded hot-copy records")
        }
        let manager = FileManager.default
        let indexURL = folder.appendingPathComponent("index.json")
        let measure = freeBytes ?? { try HotCopyRefreshPlan.available(folder) }
        let payload = onto?.payload ?? folder.appendingPathComponent("experts-generation-\(UUID().uuidString).bin")
        let start = onto?.bytes ?? 0
        var lockFD: Int32 = -1
        var opened: FileHandle?
        var published = false
        var created = false
        var supersededPayload: URL?
        defer {
            try? opened?.close()
            if lockFD >= 0 {
                _ = flock(lockFD, LOCK_UN)
                close(lockFD)
            }
        }
        do {
            try wait()
            try perform {
                // Existing writer ownership allows orphan cleanup even when an interrupted
                // generation filled the remaining headroom. Creating new metadata needs room.
                let lockURL = folder.appendingPathComponent(".refresh.lock")
                if !manager.fileExists(atPath: lockURL.path),
                    HotCopyRefreshPlan.room(free: try measure(), reserve: reserve) == 0
                {
                    throw EngineError("Hot-copy refresh waits for free space above its reserve")
                }
                try manager.createDirectory(at: folder, withIntermediateDirectories: true)
                lockFD = Darwin.open(lockURL.path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)
                guard lockFD >= 0, flock(lockFD, LOCK_EX | LOCK_NB) == 0 else {
                    throw EngineError("Another hot-copy refresh owns this folder")
                }
                let existing = (try? Data(contentsOf: indexURL)).flatMap { try? JSONDecoder().decode(Index.self, from: $0) }
                if let existing, HotCopyRefreshPlan.validPayload(existing.payload ?? "experts.bin") {
                    supersededPayload = folder.appendingPathComponent(existing.payload ?? "experts.bin")
                }
                if let onto {
                    let held = onto.checking.withLock { onto.indexLocked() }
                    guard let existing, existing.identity == identity, existing.keys == held.keys,
                        existing.offsets == held.offsets, (existing.byteCount ?? onto.bytes) == onto.bytes,
                        (existing.payload ?? "experts.bin") == onto.payload.lastPathComponent
                    else {
                        throw EngineError("The hot-copy manifest changed before append")
                    }
                }
                // Interrupted unique generations are unreferenced. Unlinking never mutates an
                // open reader's inode; blocks pinned by readers remain in measured usage.
                let kept = existing?.payload ?? (existing == nil ? nil : "experts.bin")
                for url in (try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] {
                    if HotCopyRefreshPlan.validPayload(url.lastPathComponent), url.lastPathComponent != kept,
                        url != payload
                    {
                        try? manager.removeItem(at: url)
                    }
                }
                guard HotCopyRefreshPlan.room(free: try measure(), reserve: reserve) > 0 else {
                    throw EngineError("Hot-copy refresh waits for free space above its reserve")
                }
                let flags = onto == nil ? O_WRONLY | O_CREAT | O_EXCL : O_WRONLY
                let descriptor = Darwin.open(payload.path, flags, S_IRUSR | S_IWUSR)
                guard descriptor >= 0 else { throw EngineError("Cannot open bounded hot-copy payload: errno \(errno)") }
                created = onto == nil
                let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
                opened = handle
                _ = fcntl(descriptor, F_NOCACHE, 1)
                // An interrupted append's tail is outside every published record.
                try handle.truncate(atOffset: UInt64(start))
                try handle.seek(toOffset: UInt64(start))
            }
            guard let handle = opened else { throw EngineError("Hot-copy refresh did not open its payload") }
            var allowance = HotCopyRefreshPlan.room(free: try measure(), reserve: reserve)
            // Appended after every slot of `onto`, empty ones included.
            let held = onto.map { copy in copy.checking.withLock { copy.indexLocked() } }
            var offsets = held?.offsets ?? []
            var checksums = held?.checksums ?? []
            var at = start
            let sizes = try keys.map { key -> UInt64 in
                guard let bytes = size(key, source: source), bytes > 0, bytes <= UInt64(Int.max) else {
                    throw EngineError("No routed record \(key)")
                }
                return bytes
            }
            let buffer = UnsafeMutableRawPointer.allocate(byteCount: Int(sizes.max()!), alignment: 16384)
            defer { buffer.deallocate() }
            for (key, padded) in zip(keys, sizes) {
                try wait()
                var sum: UInt32 = 0
                try perform {
                    guard padded <= allowance,
                        padded <= HotCopyRefreshPlan.room(free: try measure(), reserve: reserve)
                    else {
                        throw EngineError("Hot-copy refresh stopped before crossing its free-space reserve")
                    }
                    memset(buffer, 0, Int(padded))
                    let count = source.hotCopyRecordBytes(key)!
                    // Checked reads: a record failing its checksum is read from the source, not carried into the new copy.
                    let copied =
                        reuse.map { (try? $0.read(key, into: UnsafeMutableRawBufferPointer(start: buffer, count: count))) != nil } ?? false
                    if !copied {
                        if let readMissing {
                            try readMissing(key, buffer)
                        } else {
                            try source.readHotCopyRecord(key, into: buffer)
                        }
                    }
                    sum = checksum(UnsafeRawBufferPointer(start: buffer, count: count))
                    try handle.write(contentsOf: Data(bytesNoCopy: buffer, count: Int(padded), deallocator: .none))
                }
                allowance -= padded
                offsets.append(at)
                checksums.append(sum)
                at += Int64(padded)
                wrote(Int64(padded))
            }
            let all = (held?.keys ?? []) + keys
            var finished: HotCopy?
            try wait()
            try perform {
                let free = try measure()
                guard free >= reserve else { throw EngineError("Hot-copy reserve unavailable at publication") }
                // fsync leaves data in the drive's cache: a power cut could publish an index over unwritten records.
                guard PersistentCheckpointStore.fullSync(handle.fileDescriptor) else {
                    throw EngineError("Cannot flush the hot-copy payload: errno \(errno)")
                }
                // Open the completed payload before publication: a descriptor failure cannot
                // leave a new manifest followed by rollback/truncation of its published bytes.
                let next = try HotCopy(
                    folder: folder, identity: identity, keys: all, offsets: offsets, bytes: at, payload: payload.lastPathComponent,
                    targetBytes: targetBytes ?? onto?.targetBytes, checksums: checksums, version: held?.version)
                try publish(
                    Index(
                        version: held?.version, identity: identity, keys: all, offsets: offsets, byteCount: at,
                        payload: payload.lastPathComponent, targetBytes: next.targetBytes, checksums: checksums),
                    folder: folder
                ) {
                    published = true
                    finished = next
                }
                // Safe even while old readers hold descriptors. Their storage is not counted
                // as reclaimed until statfs actually reports it available on a later step.
                if let supersededPayload, supersededPayload != payload { try? manager.removeItem(at: supersededPayload) }
            }
            guard let finished else { throw EngineError("Hot-copy refresh did not publish") }
            return finished
        } catch {
            if !published {
                if created {
                    try? manager.removeItem(at: payload)
                } else if onto != nil, let handle = opened {
                    // Stopping may deny idle ownership. Leave a harmless unpublished tail for
                    // the next attempt to truncate in that case.
                    try? perform { try handle.truncate(atOffset: UInt64(start)) }
                }
            }
            throw error
        }
    }

}
