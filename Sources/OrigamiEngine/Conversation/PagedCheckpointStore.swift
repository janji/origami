import CryptoKit
import Darwin
import Foundation

/// What the last paged publication wrote, cloned and verified, and how far it got (`publicationState`).
public struct PagedCheckpointWriteStatistics: Codable, Sendable {
    public let logicalBytes, allocatedBytes, writtenBytes, verificationReadBytes: UInt64
    public let unchangedPages, changedPages: Int
    public let cloned: Bool
    public let publicationState: String
}

/// A paged head's manifest fields and decoded timeline, charged to the ledger while it lives; only a hint
/// for choosing what to read (the read re-verifies everything).
package final class CheckpointDiskHint {
    package let version: Int
    let byteOrder: String
    let identity: CheckpointIdentity
    package let segments: [CheckpointSegment]
    package let frontier: CheckpointFrontier
    let hasLogits: Bool
    private let memory: CheckpointMemoryLease
    init(header: PagedCheckpointFormat.Header, segments: [CheckpointSegment], memory: CheckpointMemoryLease) {
        version = header.version
        byteOrder = header.byteOrder
        identity = header.identity
        self.segments = segments
        frontier = header.frontier
        hasLogits = header.hasLogits
        self.memory = memory
    }
}

extension PersistentCheckpointStore {
    private func pagedFile() throws -> ExactFile {
        var info = stat()
        guard checkpointURL.withUnsafeFileSystemRepresentation({ lstat($0!, &info) }) == 0,
            (info.st_mode & S_IFMT) == S_IFREG, info.st_uid == geteuid(), info.st_mode & 0o077 == 0,
            info.st_nlink == 1
        else { throw EngineError("Paged checkpoint is missing or has unsafe ownership/type") }
        let file = try ExactFile(url: checkpointURL)
        guard file.identity.device == UInt64(info.st_dev), file.identity.inode == info.st_ino else {
            throw EngineError("Paged checkpoint changed while opening")
        }
        return file
    }
    private func validateHead(_ file: ExactFile) throws {
        try file.validateIdentity()
        guard try pagedFile().identity == file.identity else { throw EngineError("Committed checkpoint name changed during transaction") }
    }
    private func page(
        _ file: ExactFile, manifest: PagedCheckpointFormat.Manifest, section: Int, index: Int,
        into target: UnsafeMutableRawBufferPointer, shouldContinue: () -> Bool
    ) throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        let spec = manifest.header.sections[section]
        let offset = UInt64(index * PagedCheckpointFormat.pageBytes)
        guard index >= 0, index < spec.pages, offset < spec.length,
            target.count == min(PagedCheckpointFormat.pageBytes, Int(spec.length - offset))
        else {
            throw EngineError("Paged checkpoint requested invalid page")
        }
        // A failed read says nothing about the entry: its code keeps the prompt cache from dropping it.
        do { try file.read(offset: Int64(spec.offset + offset), into: target) } catch {
            throw EngineError("Could not read the saved state in \(directory.path): \(error)", code: "kv_io_failure")
        }
        guard PagedCheckpointFormat.root(UnsafeRawBufferPointer(target)) == manifest.root(section: section, page: index) else {
            throw EngineError("Paged checkpoint payload checksum mismatch: \(spec.name) page\(index)")
        }
    }
    /// The saved image's token IDs alone, each page checked against its digest, its tensors left unread (the
    /// dashboard's view of a saved prompt). The store lock keeps a save from replacing the file mid-read.
    package func savedIDs() throws -> [Int64] {
        let memory = try CheckpointMemoryLease(
            ledger: ledger, bytes: UInt64(PagedCheckpointFormat.manifestBytes + capacity * 8), category: "checkpoint_saved_ids")
        defer { withExtendedLifetime(memory) {} }
        return try withStoreLock {
            let file = try pagedFile()
            let manifest = try PagedCheckpointFormat.load(file, capacity: capacity, maximumBytes: maximumBytes)
            guard let spec = manifest.header.sections.first, spec.name == "consumed_ids" else {
                throw EngineError("Paged checkpoint holds no token IDs")
            }
            var ids = [Int64](repeating: 0, count: Int(spec.length) / 8)
            try ids.withUnsafeMutableBytes { bytes in
                for index in 0..<spec.pages {
                    let start = index * PagedCheckpointFormat.pageBytes
                    try page(
                        file, manifest: manifest, section: 0, index: index,
                        into: UnsafeMutableRawBufferPointer(
                            rebasing: bytes[start..<min(bytes.count, start + PagedCheckpointFormat.pageBytes)]), shouldContinue: { true })
                }
            }
            return ids
        }
    }
    private func verifyPayload(
        _ file: ExactFile, manifest: PagedCheckpointFormat.Manifest,
        scratch: CheckpointPageScratch, shouldContinue: () -> Bool
    ) throws -> UInt64 {
        var read: UInt64 = 0
        for (section, spec) in manifest.header.sections.enumerated() {
            for index in 0..<spec.pages {
                let count = min(PagedCheckpointFormat.pageBytes, Int(spec.length) - index * PagedCheckpointFormat.pageBytes)
                try page(
                    file, manifest: manifest, section: section, index: index, into: scratch.buffer(count), shouldContinue: shouldContinue)
                read += UInt64(count)
            }
        }
        try validateHead(file)
        return read
    }
    package func peekPaged() throws -> CheckpointDiskHint {
        let scratch = try CheckpointMemoryLease(
            ledger: ledger, bytes: PagedCheckpointFormat.scratchBytes, category: "checkpoint_manifest_scratch")
        defer { withExtendedLifetime(scratch) {} }
        return try withStoreLock {
            let file = try pagedFile()
            let manifest = try PagedCheckpointFormat.load(file, capacity: capacity, maximumBytes: maximumBytes)
            let count = manifest.timeline.count / PagedCheckpointFormat.segmentBytes
            let memory = try CheckpointMemoryLease(
                ledger: ledger, bytes: UInt64(count * 1024 + 65536), category: "checkpoint_hint_timeline")
            let segments = try PagedCheckpointFormat.decodeTimeline(manifest.timeline)
            try validateHead(file)
            return CheckpointDiskHint(header: manifest.header, segments: segments, memory: memory)
        }
    }
    @discardableResult
    package func writePaged(_ checkpoint: ModelCheckpoint, base: PersistentCheckpointStore? = nil, shouldContinue: () -> Bool = { true })
        throws -> String
    {
        try checkpoint.validate()
        let scratch = try CheckpointMemoryLease(
            ledger: ledger, bytes: PagedCheckpointFormat.scratchBytes, category: "checkpoint_manifest_scratch")
        let buffer = try CheckpointPageScratch(ledger: ledger)
        defer { withExtendedLifetime(scratch) {} }
        let layout = try PagedCheckpointFormat.layout(
            capacity: capacity, tokens: checkpoint.frontier.consumedTokenCount, hasLogits: checkpoint.nextLogits != nil)
        let timeline = try PagedCheckpointFormat.timeline(checkpoint.segments)
        return try withStoreLock {
            lastWriteStatistics = nil
            guard shouldContinue() else { throw IOOperationCancelled() }
            try removeLeftoverStaging()
            let existing = try managedBytes()
            guard existing <= maximumBytes, layout.bytes <= maximumBytes - existing else {
                throw EngineError("Paged checkpoint logical old+staged quota exhausted")
            }
            let volume = try directory.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            guard let available = volume.volumeAvailableCapacity, available >= 0,
                UInt64(available) >= PagedCheckpointFormat.diskBytes(layout, at: directory)
            else {
                throw EngineError("Not enough free disk in \(directory.path) to save this state", code: "insufficient_storage")
            }
            let dirFD = directory.withUnsafeFileSystemRepresentation { open($0!, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
            guard dirFD >= 0 else { throw EngineError("Paged checkpoint directory open failed") }
            defer { close(dirFD) }
            var dirIdentity = stat()
            guard fstat(dirFD, &dirIdentity) == 0, dirIdentity.st_uid == geteuid(), dirIdentity.st_mode & 0o022 == 0 else {
                throw EngineError("Paged checkpoint directory ownership differs")
            }
            func validateDirectory() throws {
                var current = stat()
                guard directory.withUnsafeFileSystemRepresentation({ lstat($0!, &current) }) == 0,
                    current.st_dev == dirIdentity.st_dev, current.st_ino == dirIdentity.st_ino,
                    current.st_mode == dirIdentity.st_mode, current.st_uid == dirIdentity.st_uid
                else {
                    throw EngineError("Paged checkpoint directory identity changed")
                }
            }
            // `head` is the committed file this publication replaces; `oldFile` only when it verified for cloning.
            var oldFile: ExactFile?
            var head: ExactFile?
            var old: PagedCheckpointFormat.Manifest?
            var verificationRead: UInt64 = 0
            var info = stat()
            let headStatus = checkpointURL.withUnsafeFileSystemRepresentation { lstat($0!, &info) }
            if headStatus == 0 {
                let file = try pagedFile()
                head = file
                do {
                    let manifest = try PagedCheckpointFormat.load(file, capacity: capacity, maximumBytes: maximumBytes)
                    // Cold instances verify every live payload page before trusting
                    // a clone. Same-process reuse requires the exact sealed stamp.
                    if trustedPagedIdentity != file.identity {
                        verificationRead = try verifyPayload(file, manifest: manifest, scratch: buffer, shouldContinue: shouldContinue)
                    }
                    oldFile = file
                    old = manifest
                } catch let error as IOOperationCancelled { throw error } catch {
                    // A head failing format or payload verification is never cloned: a complete
                    // fresh image replaces it, otherwise no later save could ever publish.
                }
            } else if errno != ENOENT {
                throw EngineError("Paged checkpoint head stat failed: errno=\(errno)")
            }
            // The base's image is trusted only as this process sealed it (its identity unchanged since its write);
            // saves run one at a time, so it is not being rewritten meanwhile.
            if let base, base !== self, base.capacity == capacity, let file = try? base.pagedFile(),
                base.trustedPagedIdentity == file.identity,
                let manifest = try? PagedCheckpointFormat.load(file, capacity: capacity, maximumBytes: base.maximumBytes)
            {
                oldFile = file
                old = manifest
            }
            let name = ".checkpoint-\(UUID().uuidString).tmp"
            defer { _ = name.withCString { unlinkat(dirFD, $0, 0) } }
            // A volume without clones (Mac OS Extended) writes a complete fresh image instead.
            if let source = oldFile, (try? source.clone(into: dirFD, name: name)) == nil {
                oldFile = nil
                old = nil
            }
            let flags = oldFile == nil ? O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC : O_RDWR | O_NOFOLLOW | O_CLOEXEC
            let fd = name.withCString { openat(dirFD, $0, flags, 0o600) }
            guard fd >= 0 else { throw EngineError("Paged checkpoint staging open failed: errno=\(errno)") }
            defer { close(fd) }
            guard fchmod(fd, 0o600) == 0, fcntl(fd, F_NOCACHE, 1) == 0, ftruncate(fd, off_t(layout.bytes)) == 0 else {
                throw EngineError("Paged checkpoint staging setup failed: errno=\(errno)")
            }
            try pagedFault?("staged")
            var written: UInt64 = 0
            var changed = 0
            var unchanged = 0
            func write(_ bytes: UnsafeRawBufferPointer, offset: UInt64) throws {
                var done = 0
                while done < bytes.count {
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    let count = pwrite(fd, bytes.baseAddress!.advanced(by: done), bytes.count - done, off_t(offset + UInt64(done)))
                    if count < 0 && errno == EINTR { continue }
                    guard count > 0 else { throw EngineError("Paged checkpoint exact write failed: errno=\(errno),partial=\(done)") }
                    done += count
                    written += UInt64(count)
                }
            }
            var roots = Data()
            var sectionIndex = 0
            var pageIndex = 0
            try checkpoint.withCheckpointPages { name, dtype, index, bytes, root, load in
                guard shouldContinue() else { throw IOOperationCancelled() }
                while sectionIndex < layout.sections.count && pageIndex == layout.sections[sectionIndex].pages {
                    sectionIndex += 1
                    pageIndex = 0
                }
                guard sectionIndex < layout.sections.count else { throw EngineError("Paged source has extra tensors") }
                let spec = layout.sections[sectionIndex]
                let length = min(PagedCheckpointFormat.pageBytes, Int(spec.length) - index * PagedCheckpointFormat.pageBytes)
                guard name == spec.name, dtype == spec.dtype, index == pageIndex, bytes == length, root.count == 32 else {
                    throw EngineError("Paged source tensor geometry differs: \(name)")
                }
                roots.append(root)
                if let old, old.header.sections[sectionIndex].pages > index,
                    old.header.sections[sectionIndex].length >= UInt64(index * PagedCheckpointFormat.pageBytes + bytes),
                    old.root(section: sectionIndex, page: index) == root
                {
                    unchanged += 1
                } else {
                    let target = buffer.buffer(bytes)
                    try load(target)
                    guard PagedCheckpointFormat.root(UnsafeRawBufferPointer(target)) == root else {
                        throw EngineError("Paged source changed between digest and copy")
                    }
                    try write(UnsafeRawBufferPointer(target), offset: spec.offset + UInt64(index * PagedCheckpointFormat.pageBytes))
                    changed += 1
                    try pagedFault?("page_written")
                }
                pageIndex += 1
            }
            guard roots.count == layout.sections.reduce(0, { $0 + $1.pages }) * 32 else {
                throw EngineError("Paged source omitted tensor pages")
            }
            let header = PagedCheckpointFormat.Header(
                version: 2, capacity: capacity, byteOrder: "little", identity: checkpoint.identity,
                frontier: checkpoint.frontier, hasLogits: checkpoint.nextLogits != nil, sections: layout.sections)
            let (manifest, digest) = try PagedCheckpointFormat.encode(header: header, timeline: timeline, roots: roots)
            try manifest.withUnsafeBytes { try write($0, offset: 0) }
            try pagedFault?("before_fsync")
            guard Self.fullSync(fd) else { throw EngineError("Paged checkpoint staging fsync failed: errno=\(errno)") }
            if let head {
                try validateHead(head)
            } else {
                var absent = stat()
                guard fstatat(dirFD, "checkpoint.origami", &absent, AT_SYMLINK_NOFOLLOW) != 0, errno == ENOENT else {
                    throw EngineError("Paged checkpoint appeared during initial publication")
                }
            }
            try validateDirectory()
            try pagedFault?("before_rename")
            guard shouldContinue() else { throw IOOperationCancelled() }
            // Fault callbacks may mutate the old file; recheck at the final
            // publication boundary as well as before the injected boundary.
            if let head { try validateHead(head) }
            let renamed = name.withCString { renameat(dirFD, $0, dirFD, "checkpoint.origami") }
            guard renamed == 0 else { throw EngineError("Paged checkpoint rename failed: errno=\(errno)") }
            trustedPagedIdentity = nil
            var finalInfo = stat()
            guard fstat(fd, &finalInfo) == 0, finalInfo.st_size == off_t(layout.bytes), finalInfo.st_nlink == 1 else {
                throw EngineError("Paged checkpoint changed after rename; commit durability unconfirmed")
            }
            func statistics(_ state: String) -> PagedCheckpointWriteStatistics {
                PagedCheckpointWriteStatistics(
                    logicalBytes: layout.bytes, allocatedBytes: UInt64(finalInfo.st_blocks) * 512,
                    writtenBytes: written, verificationReadBytes: verificationRead, unchangedPages: unchanged,
                    changedPages: changed, cloned: oldFile != nil, publicationState: state)
            }
            lastWriteStatistics = statistics("renamed_directory_sync_unconfirmed")
            try pagedFault?("after_rename")
            guard Self.fullSync(dirFD) else { throw EngineError("Paged checkpoint directory fsync failed after commit: errno=\(errno)") }
            try validateDirectory()
            let published = try pagedFile()
            guard published.identity.device == UInt64(finalInfo.st_dev), published.identity.inode == finalInfo.st_ino else {
                throw EngineError("Paged checkpoint head changed after commit")
            }
            trustedPagedIdentity = published.identity
            lastWriteStatistics = statistics("durable")
            return digest
        }
    }
    package func readPaged(
        expectedIdentity: CheckpointIdentity, expectedSegments: [CheckpointSegment], expectedConsumedIDs: [Int64],
        pageArena: QSAPageArena?, shouldContinue: () -> Bool = { true }
    ) throws -> ModelCheckpoint {
        let scratch = try CheckpointMemoryLease(
            ledger: ledger, bytes: PagedCheckpointFormat.scratchBytes, category: "checkpoint_manifest_scratch")
        defer { withExtendedLifetime(scratch) {} }
        return try withStoreLock {
            let file = try pagedFile()
            let manifest = try PagedCheckpointFormat.load(file, capacity: capacity, maximumBytes: maximumBytes)
            let header = manifest.header
            let end = header.frontier.consumedTokenCount
            guard header.identity == expectedIdentity, end == expectedConsumedIDs.count,
                try manifest.timeline == PagedCheckpointFormat.timeline(expectedSegments)
            else {
                throw EngineError("Paged checkpoint artifact, timeline or consumed frontier mismatch")
            }
            let metadata = UInt64(expectedSegments.count * 1024 + 65536)
            let referenceBound = pageArena == nil ? 0 : QSAPagedStorage.maximumReferenceBytes
            let memory = try CheckpointMemoryLease(ledger: ledger, bytes: metadata + referenceBound)
            var sectionIndex = 0
            func readSection(_ into: UnsafeMutableRawBufferPointer, section: Int) throws {
                let spec = header.sections[section]
                guard into.count == spec.length else { throw EngineError("Paged section allocation length differs") }
                for index in 0..<spec.pages {
                    let offset = index * PagedCheckpointFormat.pageBytes
                    let count = min(PagedCheckpointFormat.pageBytes, into.count - offset)
                    try page(
                        file, manifest: manifest, section: section, index: index,
                        into: UnsafeMutableRawBufferPointer(rebasing: into[offset..<offset + count]), shouldContinue: shouldContinue)
                }
            }
            func array<T>(_ zero: T) throws -> [T] {
                let section = sectionIndex
                sectionIndex += 1
                var value: [T] = try memory.allocateArray(count: Int(header.sections[section].length) / MemoryLayout<T>.stride) {
                    [T](repeating: zero, count: $0)
                }
                try value.withUnsafeMutableBytes { try readSection($0, section: section) }
                return value
            }
            func vector<T>(_ initial: T, append: (T, [UInt16]) throws -> T) throws -> T {
                let section = sectionIndex
                sectionIndex += 1
                let spec = header.sections[section]
                let allocation = try CheckpointMemoryLease(
                    ledger: ledger,
                    bytes: CheckpointMemoryLease.arrayAdmissionBytes(count: PagedCheckpointFormat.pageBytes / 2, element: UInt16.self),
                    category: "checkpoint_stream_page_scratch")
                defer { withExtendedLifetime(allocation) {} }
                var result = initial
                for index in 0..<spec.pages {
                    let bytes = min(PagedCheckpointFormat.pageBytes, Int(spec.length) - index * PagedCheckpointFormat.pageBytes)
                    var values = [UInt16](repeating: 0, count: bytes / 2)
                    guard UInt64(values.capacity * 2) <= allocation.bytes else {
                        throw EngineError("Paged restore chunk exceeds admission")
                    }
                    try values.withUnsafeMutableBytes {
                        try page(file, manifest: manifest, section: section, index: index, into: $0, shouldContinue: shouldContinue)
                    }
                    result = try append(result, values)
                }
                return result
            }
            let ids = try array(Int64(0))
            guard ids == expectedConsumedIDs else { throw EngineError("Paged checkpoint token identity mismatch") }
            let history = try array(Int64(0))
            let ple = try array(UInt16(0))
            var gdn: [Int: GDNState] = [:]
            var qsa: [Int: QSAState] = [:]
            for layer in 0..<48 where layer % 4 != 3 {
                gdn[layer] = try GDNState(offset: end, recurrent: array(Float(0)), convolutionTail: array(UInt16(0)))
            }
            for layer in 0..<48 where layer % 4 == 3 {
                if let pageArena {
                    var pages = QSAPagedStorage(arena: pageArena)
                    pages.keys = try vector(pages.keys) { try $0.appending($1) }
                    pages.values = try vector(pages.values) { try $0.appending($1) }
                    pages.pooled = try vector(pages.pooled) { try $0.appending($1) }
                    qsa[layer] = try QSAState(offset: end, pages: pages, rawTail: array(UInt16(0)))
                } else {
                    qsa[layer] = try QSAState(
                        offset: end, keys: array(UInt16(0)), values: array(UInt16(0)), pooledKeys: array(UInt16(0)),
                        rawTail: array(UInt16(0)))
                }
            }
            let logits: [Float]? = header.hasLogits ? try array(Float(0)) : nil
            guard sectionIndex == header.sections.count - (header.hasLogits ? 0 : 1), shouldContinue() else { throw IOOperationCancelled() }
            try validateHead(file)
            let actual =
                gdn.values.reduce(UInt64(0), { $0 + $1.hostCapacityBytes }) + qsa.values.reduce(UInt64(0), { $0 + $1.hostCapacityBytes })
                + UInt64(ids.capacity * 8 + history.capacity * 8 + ple.capacity * 2 + (logits?.capacity ?? 0) * 4) + metadata
            guard qsa.values.reduce(UInt64(0), { $0 + ($1.pages?.referenceCapacityBytes ?? 0) }) <= referenceBound,
                actual <= memory.bytes
            else { throw EngineError("Paged restore retained capacity exceeded admission") }
            let result = try ModelCheckpoint(
                identity: header.identity, segments: expectedSegments, frontier: header.frontier,
                consumedIDs: ids, nextLogits: logits, history: history, gdn: gdn, qsa: qsa, pleConvolution: ple, memory: memory)
            trustedPagedIdentity = file.identity
            return result
        }
    }
}
