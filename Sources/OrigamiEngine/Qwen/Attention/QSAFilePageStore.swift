import CExactIO
import CryptoKit
import Darwin
import Foundation

/// Runtime-only immutable KV extents. One unlinked file owns all slots; live page leases prevent reuse, and every
/// returned read unit is SHA256 verified. A slot only grows: a page is a view of its first `count` tokens, so appending
/// to the slot's longest view writes just the new rows after it (the older view still reads what it read), and
/// appending to a shorter one copies it into a new slot. This is not a durable checkpoint: publication requires
/// complete writes but no per-page fsync. Restart images retain their separate journal contract.
package final class QSAFilePageStore: @unchecked Sendable {
    package static let pageBytes = 256 * 512 * 2
    package static let chunkBytes = 8 * 512 * 2
    package static let maximumDiskBytes: UInt64 = 64 * 1024 * 1024 * 1024
    /// Slots of one 256 KiB page each, and the anonymous metadata slab: per slot 32 SHA-256 digests (one per full
    /// 8 KiB chunk = 8 tokens; a view's last partial chunk keeps its own in the page), a generation, the bytes written,
    /// a free-stack entry and the number of live views.
    package struct Layout: Sendable {
        package let slots, slabBytes: Int
        package let metadataBytes, diskBytes: UInt64
    }
    package static func layout(diskBytes: UInt64) throws -> Layout {
        guard diskBytes > 0, diskBytes <= maximumDiskBytes, diskBytes % UInt64(pageBytes) == 0 else {
            throw EngineError("Invalid file-KV extent quota")
        }
        let slots = Int(diskBytes / UInt64(pageBytes))
        // 32 SHA256s, generation, written-byte count, bounded free stack and view count.
        // 256B/slot additionally admits the page lease/immutable-page objects.
        let slab = (slots * (1024 + 8 + 4 + 4 + 4) + pageBytes + 16383) / 16384 * 16384
        return Layout(
            slots: slots, slabBytes: slab,
            metadataBytes: UInt64(slab + slots * 256 + 131072), diskBytes: diskBytes)
    }
    package struct Snapshot: Codable, Sendable {
        package let diskQuotaBytes, liveBytes, highWaterBytes, writtenBytes, readBytes, metadataBytes: UInt64
        package let livePages, allocations, reusedSlots, refusals: Int
    }
    /// A lease on the first `count` tokens of one published slot, which never change; the last view's release
    /// returns the slot to the free stack. `tail` is the digest of its last partial chunk (nil when it ends on one).
    package final class Page: @unchecked Sendable {
        fileprivate let store: QSAFilePageStore, slot: Int, generation: UInt64
        package let count: Int
        fileprivate let tail: Data?
        package var cacheIdentity: (slot: Int, generation: UInt64) { (slot, generation) }
        fileprivate init(store: QSAFilePageStore, slot: Int, generation: UInt64, count: Int, tail: Data?) {
            self.store = store
            self.slot = slot
            self.generation = generation
            self.count = count
            self.tail = tail
        }
        deinit { store.retire(slot: slot, generation: generation) }
    }
    package let layout: Layout
    private let memory: CheckpointMemoryLease
    fileprivate let fd: Int32
    private let parent: URL, identity: stat
    private let slab: UnsafeMutableRawPointer, hashes: UnsafeMutableRawPointer, scratch: UnsafeMutableRawPointer
    private let generations: UnsafeMutablePointer<UInt64>, lengths: UnsafeMutablePointer<UInt32>, free: UnsafeMutablePointer<UInt32>
    private let views: UnsafeMutablePointer<UInt32>
    private let lock = NSLock()
    private var next = 0, freeCount = 0, live = 0, high = 0, allocations = 0, reused = 0, refusals = 0
    private var written: UInt64 = 0, read: UInt64 = 0

    package init(parent: URL, diskBytes: UInt64 = maximumDiskBytes, ledger: MemoryLedger) throws {
        let layout = try Self.layout(diskBytes: diskBytes)
        let memory = try CheckpointMemoryLease(ledger: ledger, bytes: layout.metadataBytes, category: "conversation_file_kv_metadata")
        let directory = parent.appendingPathComponent("kv-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let directoryFD = directory.withUnsafeFileSystemRepresentation { Darwin.open($0!, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
        // The file is unlinked and its folder removed before the store is used, so a killed process leaves
        // nothing behind; the open descriptor keeps the file until the store goes.
        defer {
            if directoryFD >= 0 {
                _ = ".extents".withCString { unlinkat(directoryFD, $0, 0) }
                Darwin.close(directoryFD)
            }
            _ = directory.withUnsafeFileSystemRepresentation { rmdir($0!) }
        }
        var fd: Int32 = -1
        var mapped: UnsafeMutableRawPointer?
        do {
            var directoryStat = stat()
            guard directoryFD >= 0, fstat(directoryFD, &directoryStat) == 0,
                directoryStat.st_uid == geteuid(), directoryStat.st_mode & 0o777 == 0o700
            else {
                throw EngineError("File-KV directory ownership differs")
            }
            fd = ".extents".withCString { openat(directoryFD, $0, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600) }
            guard fd >= 0 else { throw Self.failure("create the context file", in: parent, errno) }
            guard ".extents".withCString({ unlinkat(directoryFD, $0, 0) }) == 0,
                fcntl(fd, F_NOCACHE, 1) == 0, ftruncate(fd, off_t(diskBytes)) == 0
            else {
                throw Self.failure("set up the context file", in: parent, errno)
            }
            var identity = stat()
            guard fstat(fd, &identity) == 0, identity.st_nlink == 0, identity.st_uid == geteuid(),
                identity.st_mode & S_IFMT == S_IFREG, identity.st_mode & 0o777 == 0o600,
                identity.st_size == off_t(diskBytes)
            else { throw EngineError("File-KV descriptor identity differs") }
            mapped = mmap(nil, layout.slabBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
            guard let mapped, mapped != MAP_FAILED else {
                throw EngineError("File-KV metadata allocation failed", code: "insufficient_memory")
            }
            self.layout = layout
            self.memory = memory
            self.parent = parent
            self.fd = fd
            self.identity = identity
            slab = mapped
            hashes = mapped
            generations = mapped.advanced(by: layout.slots * 1024).assumingMemoryBound(to: UInt64.self)
            lengths = mapped.advanced(by: layout.slots * 1032).assumingMemoryBound(to: UInt32.self)
            free = mapped.advanced(by: layout.slots * 1036).assumingMemoryBound(to: UInt32.self)
            views = mapped.advanced(by: layout.slots * 1040).assumingMemoryBound(to: UInt32.self)
            scratch = mapped.advanced(by: layout.slots * 1044)
        } catch {
            if let mapped, mapped != MAP_FAILED { munmap(mapped, layout.slabBytes) }
            if fd >= 0 { Darwin.close(fd) }
            throw error
        }
    }
    deinit {
        Darwin.close(fd)
        munmap(slab, layout.slabBytes)
    }
    /// What failed, in which folder and why, and what to do: a full drive is `insufficient_storage`, any other
    /// error `kv_io_failure` (0: the file ended early).
    fileprivate static func failure(_ action: String, in folder: URL, _ code: Int32) -> EngineError {
        let reason = code == 0 ? "the file is shorter than its pages" : String(cString: strerror(code))
        let hint =
            code == ENOSPC
            ? "Free space on that drive"
            : (code == EFBIG || code == ENOTSUP ? "Use a folder on an APFS or Mac OS Extended drive" : "Check that drive")
        return EngineError(
            "Could not \(action) in \(folder.path): \(reason). \(hint), or set context_dir to another folder.",
            code: code == ENOSPC ? "insufficient_storage" : "kv_io_failure")
    }
    private func validate() throws {
        var current = stat()
        guard fstat(fd, &current) == 0 else { throw Self.failure("check the context file", in: parent, errno) }
        guard current.st_dev == identity.st_dev, current.st_ino == identity.st_ino,
            current.st_size == identity.st_size, current.st_nlink == 0, current.st_uid == identity.st_uid,
            current.st_mode == identity.st_mode
        else { throw EngineError("File-KV descriptor changed") }
        // mtime/ctime legitimately change when other immutable slots publish.
        // Per-chunk hashes plus live generation checks protect existing pages.
    }
    private func validate(_ page: Page) throws {
        guard page.store === self, page.slot >= 0, page.slot < next, page.count > 0, views[page.slot] > 0,
            generations[page.slot] == page.generation, UInt32(page.count * 2) <= lengths[page.slot],
            (page.tail != nil) == (page.count * 2 % Self.chunkBytes != 0)
        else {
            throw EngineError("File-KV page belongs to another owner/generation")
        }
    }
    private func retire(slot: Int, generation: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        precondition(generations[slot] == generation && lengths[slot] > 0 && views[slot] > 0 && live > 0 && freeCount < layout.slots)
        views[slot] -= 1
        guard views[slot] == 0 else { return }
        lengths[slot] = 0
        free[freeCount] = UInt32(slot)
        freeCount += 1
        live -= 1
    }
    /// Whether `digest` is what a view's chunk at `offset` must hash to: the slot's digest for a full chunk, the view's
    /// own for its last partial one.
    private func matches(_ page: Page, offset: Int, _ digest: SHA256.Digest) -> Bool {
        digest.withUnsafeBytes { computed in
            if offset + Self.chunkBytes > page.count * 2, let tail = page.tail {
                return tail.withUnsafeBytes { memcmp(computed.baseAddress!, $0.baseAddress!, 32) == 0 }
            }
            return memcmp(computed.baseAddress!, hashes.advanced(by: page.slot * 1024 + offset / Self.chunkBytes * 32), 32) == 0
        }
    }
    private func hash(_ bytes: UnsafeRawBufferPointer) -> SHA256.Digest {
        var value = SHA256()
        value.update(bufferPointer: bytes)
        return value.finalize()
    }
    private func readLocked(
        _ page: Page, range: Range<Int>, into target: UnsafeMutableRawBufferPointer,
        shouldContinue: () -> Bool
    ) throws {
        try validate(page)
        let valid = page.count * 2
        guard !range.isEmpty, range.lowerBound >= 0, range.upperBound <= valid, range.lowerBound % Self.chunkBytes == 0,
            range.upperBound == valid || range.upperBound % Self.chunkBytes == 0,
            target.count == range.count, let pointer = target.baseAddress
        else { throw EngineError("File-KV read is not a complete verification unit") }
        guard shouldContinue() else { throw IOOperationCancelled() }
        let result = origami_pread_exact(fd, pointer, range.count, Int64(page.slot * Self.pageBytes + range.lowerBound))
        guard result.error_code == 0, result.bytes_read == range.count else {
            throw Self.failure("read the context file", in: parent, result.error_code)
        }
        read += UInt64(range.count)
        for offset in stride(from: range.lowerBound, to: range.upperBound, by: Self.chunkBytes) {
            guard shouldContinue() else { throw IOOperationCancelled() }
            let bytes = UnsafeRawBufferPointer(
                start: pointer.advanced(by: offset - range.lowerBound), count: min(Self.chunkBytes, valid - offset))
            guard matches(page, offset: offset, hash(bytes)) else {
                throw EngineError("File-KV chunk checksum differs", code: "kv_integrity_failure")
            }
        }
    }
    package func read(
        _ page: Page, range: Range<Int>, into target: UnsafeMutableRawBufferPointer,
        shouldContinue: () -> Bool = { true }
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        try validate()
        try readLocked(page, range: range, into: target, shouldContinue: shouldContinue)
        try validate()
    }
    /// A view of `prefix` (if any) plus `suffix`, whole 512-element rows. Appended to the slot's longest view, only the new
    /// rows are written (in place, after the bytes every older view reads); otherwise the prefix is copied into a new
    /// slot. Nothing is published unless every write succeeded: an earlier view stays valid either way.
    package func make(
        prefix: Page? = nil, suffix: UnsafeRawBufferPointer,
        shouldContinue: () -> Bool = { true }
    ) throws -> Page {
        lock.lock()
        defer { lock.unlock() }
        try validate()
        let prior = (prefix?.count ?? 0) * 2
        guard suffix.count > 0, suffix.count % 1024 == 0, prior < Self.pageBytes,
            suffix.count <= Self.pageBytes - prior, let source = suffix.baseAddress
        else { throw EngineError("File-KV append exceeds a256-token page") }
        if let prefix { try validate(prefix) }
        guard shouldContinue() else { throw IOOperationCancelled() }
        // In place: the prefix is the slot's longest view, so nothing after it is read by anyone.
        let inPlace = prefix.map { lengths[$0.slot] == UInt32(prior) } ?? false
        // `start` is the first byte rewritten in `scratch`: the prefix's partial last chunk in place, all of it in a copy.
        let start = inPlace ? prior / Self.chunkBytes * Self.chunkBytes : 0
        if let prefix, prior > start {
            try readLocked(
                prefix, range: start..<prior, into: UnsafeMutableRawBufferPointer(start: scratch, count: prior - start),
                shouldContinue: shouldContinue)
        }
        scratch.advanced(by: prior - start).copyMemory(from: source, byteCount: suffix.count)
        let slot: Int
        if inPlace {
            slot = prefix!.slot
        } else {
            guard live < layout.slots else {
                refusals += 1
                throw EngineError("Pinned file-KV pages exhaust disk quota", code: "conversation_kv_disk_quota")
            }
            if freeCount > 0 {
                freeCount -= 1
                slot = Int(free[freeCount])
                reused += 1
            } else {
                slot = next
                next += 1
            }
        }
        var published = false
        defer {
            if !published && !inPlace {
                free[freeCount] = UInt32(slot)
                freeCount += 1
            }
        }
        guard inPlace || generations[slot] < UInt64.max else { throw EngineError("File-KV slot generation overflow") }
        let length = prior + suffix.count
        var done = inPlace ? prior : 0
        while done < length {
            guard shouldContinue() else { throw IOOperationCancelled() }
            let amount = pwrite(fd, scratch.advanced(by: done - start), length - done, off_t(slot * Self.pageBytes + done))
            if amount < 0 && errno == EINTR { continue }
            guard amount > 0 else { throw Self.failure("write the context file", in: parent, amount < 0 ? errno : EIO) }
            done += amount
            self.written += UInt64(amount)
        }
        // Full chunks get the slot's digest (they never change again); a partial last chunk is the view's own.
        var tail: Data?
        for offset in stride(from: start, to: length, by: Self.chunkBytes) {
            let count = min(Self.chunkBytes, length - offset)
            let digest = hash(UnsafeRawBufferPointer(start: scratch.advanced(by: offset - start), count: count))
            if count == Self.chunkBytes {
                digest.withUnsafeBytes {
                    hashes.advanced(by: slot * 1024 + offset / Self.chunkBytes * 32).copyMemory(from: $0.baseAddress!, byteCount: 32)
                }
            } else {
                tail = Data(digest)
            }
        }
        guard shouldContinue() else { throw IOOperationCancelled() }
        try validate()
        if inPlace {
            views[slot] += 1
        } else {
            generations[slot] += 1
            views[slot] = 1
            live += 1
            high = max(high, live)
        }
        lengths[slot] = UInt32(length)
        allocations += 1
        let page = Page(store: self, slot: slot, generation: generations[slot], count: length / 2, tail: tail)
        published = true
        return page
    }
    package func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            diskQuotaBytes: layout.diskBytes, liveBytes: UInt64(live * Self.pageBytes),
            highWaterBytes: UInt64(high * Self.pageBytes), writtenBytes: written, readBytes: read,
            metadataBytes: layout.metadataBytes, livePages: live, allocations: allocations, reusedSlots: reused, refusals: refusals)
    }
    /// Digest metadata belongs to the immutable generation. Changed pages are
    /// still verified from their bytes before durable publication.
    package func checkpointRoot(_ page: Page) throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        try validate()
        try validate(page)
        var length = UInt64(page.count * 2).littleEndian
        var digest = SHA256()
        withUnsafeBytes(of: &length) { digest.update(bufferPointer: $0) }
        let full = page.count * 2 / Self.chunkBytes
        digest.update(bufferPointer: UnsafeRawBufferPointer(start: hashes.advanced(by: page.slot * 1024), count: full * 32))
        if let tail = page.tail { tail.withUnsafeBytes { digest.update(bufferPointer: $0) } }
        return Data(digest.finalize())
    }
}

/// Fault injection stays in this file so no production API exposes its fd.
package enum QSAFilePageSelfChecks {
    private static func require(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError(message) }
    }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("File-KV check expected an explicit refusal")
    }
    /// A row appended to a slot's longest view is written in place, its bytes alone, while every older view of the slot
    /// still reads exactly its own rows; appending to an older view copies it into a new slot and leaves the longer one
    /// intact. `measurements` (`file_kv_tail_appends`) gives the bytes written and seconds of filling one page with
    /// one-row appends, in place and copying (2.0.5's way).
    package static func appendsInPlace() throws {
        let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 0)
        try autoreleasepool {
            let store = try QSAFilePageStore(
                parent: CheckFixtures.scratch, diskBytes: UInt64(16 * QSAFilePageStore.pageBytes), ledger: ledger)
            func row(_ token: Int) -> [UInt16] { (0..<512).map { UInt16(truncatingIfNeeded: token * 131 + $0) } }
            func read(_ page: QSAFilePageStore.Page) throws -> [UInt16] {
                var values = [UInt16](repeating: 0, count: page.count)
                try values.withUnsafeMutableBytes { try store.read(page, range: 0..<(page.count * 2), into: $0) }
                return values
            }
            var views: [QSAFilePageStore.Page] = []
            var current: QSAFilePageStore.Page?
            var written = store.snapshot().writtenBytes
            var start = DispatchTime.now().uptimeNanoseconds
            for token in 0..<256 {
                let next = try row(token).withUnsafeBytes { try store.make(prefix: current, suffix: $0) }
                views.append(next)
                current = next
            }
            let inPlaceSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let inPlaceBytes = store.snapshot().writtenBytes - written
            try require(inPlaceBytes == 256 * 1024 && Set(views.map(\.slot)).count == 1, "In-place appends wrote \(inPlaceBytes) bytes")
            for (index, view) in views.enumerated() where index % 23 == 0 || index == 255 {
                try require(try read(view) == (0...index).flatMap(row), "View \(index) no longer reads its own rows")
            }
            // A branch from an older view: a copy in a new slot; the longest view is unchanged.
            written = store.snapshot().writtenBytes
            let branch = try row(999).withUnsafeBytes { try store.make(prefix: views[100], suffix: $0) }
            try require(
                branch.slot != views[100].slot && store.snapshot().writtenBytes - written == 102 * 1024
                    && read(branch) == (0...100).flatMap(row) + row(999) && read(views[255]) == (0...255).flatMap(row),
                "A branch from an older view was not a copy, or changed the longer view")
            views.removeAll()
            current = nil
            // Copying, as every one-row append did: each row extends a view another continuation already extended.
            var copyBytes: UInt64 = 0
            var copySeconds = 0.0
            for token in 0..<256 {
                let sibling = try row(token).withUnsafeBytes { try store.make(prefix: current, suffix: $0) }
                written = store.snapshot().writtenBytes
                start = DispatchTime.now().uptimeNanoseconds
                let next = try row(token).withUnsafeBytes { try store.make(prefix: current, suffix: $0) }
                copySeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                copyBytes += store.snapshot().writtenBytes - written
                withExtendedLifetime(sibling) {}
                current = next
            }
            try require(copyBytes == UInt64(256 * 257 / 2 * 1024) && read(current!) == (0..<256).flatMap(row), "Copying appends differ")
            CheckFixtures.measurements.append([
                "name": "file_kv_tail_appends", "rows": 256, "bytes_in_place": inPlaceBytes, "seconds_in_place": inPlaceSeconds,
                "bytes_copying": copyBytes, "seconds_copying": copySeconds,
            ])
        }
        try require(ledger.snapshot().reserved == 0, "File-KV append fixture leaked")
    }
    package static func integrityAndLifetime() throws {
        let maximum = try QSAFilePageStore.layout(diskBytes: QSAFilePageStore.maximumDiskBytes)
        try require(maximum.slots == 262144 && maximum.metadataBytes == 341_180_416, "Maximum file-KV geometry differs")
        try rejected { _ = try QSAFilePageStore.layout(diskBytes: QSAFilePageStore.maximumDiskBytes + 1) }
        let parent = CheckFixtures.scratch
        let quota = UInt64(4 * QSAFilePageStore.pageBytes)
        let layout = try QSAFilePageStore.layout(diskBytes: quota)
        let denied = try MemoryLedger(ceiling: layout.metadataBytes - 1, transientMargin: 0)
        try rejected { _ = try QSAFilePageStore(parent: parent, diskBytes: quota, ledger: denied) }
        try require(denied.snapshot().reserved == 0, "Denied file-KV owner leaked")
        let ledger = try MemoryLedger(ceiling: 2_000_000, transientMargin: 0)
        var store: QSAFilePageStore? = try QSAFilePageStore(parent: parent, diskBytes: quota, ledger: ledger)
        var escaped: QSAFilePageStore.Page?
        try autoreleasepool {
            let owner = store!
            let row = [UInt16](repeating: 0x3f80, count: 512)
            var first: QSAFilePageStore.Page? = try row.withUnsafeBytes { try owner.make(suffix: $0) }
            escaped = first
            let seven = [UInt16](repeating: 0xc001, count: 7 * 512)
            // Appended in place: the same slot, two views.
            var appended: QSAFilePageStore.Page? = try seven.withUnsafeBytes { try owner.make(prefix: first, suffix: $0) }
            let eight = [UInt16](repeating: 0x7fc1, count: 8 * 512)
            var third: QSAFilePageStore.Page? = try eight.withUnsafeBytes { try owner.make(suffix: $0) }
            var fourth: QSAFilePageStore.Page? = try eight.withUnsafeBytes { try owner.make(suffix: $0) }
            var fifth: QSAFilePageStore.Page? = try eight.withUnsafeBytes { try owner.make(suffix: $0) }
            try rejected { _ = try row.withUnsafeBytes { try owner.make(suffix: $0) } }
            try require(fourth != nil && fifth != nil && owner.snapshot().livePages == 4, "File-KV quota refusal changed existing leases")
            var bytes = [UInt16](repeating: 0, count: 8 * 512)
            try bytes.withUnsafeMutableBytes { try owner.read(appended!, range: 0..<$0.count, into: $0) }
            try require(bytes == row + seven && appended!.slot == first!.slot, "File-KV in-place prefix differs")
            first = nil
            try require(owner.snapshot().livePages == 4, "Escaped file-KV reader lost its slot")
            fourth = nil
            var calls = 0
            var cancelled = false
            do {
                _ = try row.withUnsafeBytes {
                    try owner.make(
                        suffix: $0,
                        shouldContinue: {
                            calls += 1
                            return calls < 3
                        })
                }
            } catch is IOOperationCancelled { cancelled = true }
            try require(cancelled && calls == 3 && owner.snapshot().livePages == 3, "Cancelled write published/lost a slot")
            fourth = try row.withUnsafeBytes { try owner.make(suffix: $0) }
            try require(owner.snapshot().reusedSlots >= 2 && owner.snapshot().livePages == 4, "Last-reader slot reuse failed")
            // An owner mismatch, incomplete hash unit and corrupt/truncated
            // underlying descriptor must never return authorized bytes.
            let foreign = try QSAFilePageStore(parent: parent, diskBytes: UInt64(QSAFilePageStore.pageBytes), ledger: ledger)
            try rejected { try bytes.withUnsafeMutableBytes { try foreign.read(appended!, range: 0..<$0.count, into: $0) } }
            try rejected { try bytes.withUnsafeMutableBytes { try owner.read(appended!, range: 2..<$0.count, into: $0) } }
            var bad: UInt8 = 0
            let corrupt = withUnsafePointer(to: &bad) { pwrite(owner.fd, $0, 1, off_t(third!.slot * QSAFilePageStore.pageBytes + 1)) }
            try require(corrupt == 1, "File-KV fault setup failed")
            try rejected { try bytes.withUnsafeMutableBytes { try owner.read(third!, range: 0..<$0.count, into: $0) } }
            // Another slot's corruption cannot retroactively invalidate a
            // verified immutable prefix; its own hash remains authoritative.
            try bytes.withUnsafeMutableBytes { try owner.read(appended!, range: 0..<$0.count, into: $0) }
            try require(bytes == row + seven, "Foreign slot fault changed a retained prefix")
            try require(ftruncate(owner.fd, off_t(quota - 1)) == 0, "File-KV truncation setup failed")
            try rejected { try bytes.withUnsafeMutableBytes { try owner.read(appended!, range: 0..<$0.count, into: $0) } }
            appended = nil
            third = nil
            fourth = nil
            fifth = nil
            try require(owner.snapshot().livePages == 1, "File-KV leases did not retire")
        }
        store = nil
        try require(
            escaped != nil && ledger.snapshot().reserved == layout.metadataBytes, "Escaped file page did not retain complete admission")
        escaped = nil
        try require(ledger.snapshot().reserved == 0, "File-KV last reader leaked metadata admission")
    }
}

extension QSAFilePageSelfChecks {
    /// The context file leaves no folder behind, even when the process is killed, and its I/O errors name the
    /// folder, the reason and what to do, with a code the server maps (a full drive is `insufficient_storage`).
    package static func leavesNoFolderAndNamesErrors() throws {
        let parent = CheckFixtures.scratch.appendingPathComponent("file-kv-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: parent) }
        let ledger = try MemoryLedger(ceiling: 2_000_000, transientMargin: 0)
        try autoreleasepool {
            let store = try QSAFilePageStore(parent: parent, diskBytes: UInt64(2 * QSAFilePageStore.pageBytes), ledger: ledger)
            try require(
                try FileManager.default.contentsOfDirectory(atPath: parent.path).isEmpty, "The context file left its folder in place")
            let row = [UInt16](repeating: 0x3f80, count: 512)
            let page = try row.withUnsafeBytes { try store.make(suffix: $0) }
            var bytes = [UInt16](repeating: 0, count: 512)
            try bytes.withUnsafeMutableBytes { try store.read(page, range: 0..<$0.count, into: $0) }
            try require(bytes == row, "An unlinked context file lost its bytes")
        }
        let full = QSAFilePageStore.failure("write the context file", in: parent, ENOSPC)
        let gone = QSAFilePageStore.failure("read the context file", in: parent, ENXIO)
        try require(
            full.code == "insufficient_storage" && full.description.contains(parent.path)
                && full.description.contains(String(cString: strerror(ENOSPC))) && full.description.contains("Free space"),
            "A full drive is not reported as such: \(full)")
        try require(
            gone.code == "kv_io_failure" && gone.description.contains(parent.path) && gone.description.contains("context_dir"),
            "A drive error is not reported as such: \(gone)")
        try require(ledger.snapshot().reserved == 0, "Context file fixture leaked its metadata")
    }

}
