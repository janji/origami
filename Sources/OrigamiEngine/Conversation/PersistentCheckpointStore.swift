import Darwin
import Foundation

/// A single snapshot on any volume, in the paged format (`PagedCheckpointFormat`): fixed-capacity slots with per-page
/// digests, written by cloning the previous image and rewriting the pages that changed. Quota includes the old
/// committed file and the new staging file; a save first deletes orphaned staging files under the store lock. No
/// automatic prefix lookup, generation RNG or wire-parser state is exposed.
public final class PersistentCheckpointStore {
    package let capacity: Int
    package var trustedPagedIdentity: FileIdentity?
    public internal(set) var lastWriteStatistics: PagedCheckpointWriteStatistics?
    // Package-only deterministic publication faults; production leaves this nil.
    package var pagedFault: ((String) throws -> Void)?

    public let directory: URL
    public let maximumBytes: UInt64
    public var checkpointURL: URL { directory.appendingPathComponent("checkpoint.origami") }
    package let ledger: MemoryLedger

    public init(directory: URL, maximumBytes: UInt64, ledger: MemoryLedger, capacity: Int = 8192) throws {
        guard maximumBytes > 0, maximumBytes <= 16_000_000_000,
            UInt16(littleEndian: 1) == 1
        else { throw EngineError("Invalid checkpoint quota or byte order") }
        let resolved = directory.standardizedFileURL.resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: resolved, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let layout = try PagedCheckpointFormat.layout(capacity: capacity, tokens: capacity, hasLogits: true)
        guard layout.bytes * 2 <= maximumBytes else { throw EngineError("Paged checkpoint quota cannot hold both logical generations") }
        self.capacity = capacity
        self.directory = resolved
        self.maximumBytes = maximumBytes
        self.ledger = ledger
    }

    /// Returns the image's manifest digest. `base`, another store on the same volume this process wrote last (an image of
    /// the same conversation), is cloned in place of this store's own head when its image is still the one written, so
    /// only the pages that differ are written. A failed pre-rename write leaves the previous file intact. A post-rename
    /// directory-fsync failure is reported as an error; callers must inspect the store instead of assuming rollback.
    @discardableResult
    public func write(_ checkpoint: ModelCheckpoint, base: PersistentCheckpointStore? = nil) throws -> String {
        try writePaged(checkpoint, base: base)
    }

    /// No state is published until every field, page digest and file mutation stamp passes. Identity, segmentation
    /// and token checks happen before tensor allocation or payload reads.
    public func read(
        expectedIdentity: CheckpointIdentity, expectedSegments: [CheckpointSegment],
        expectedConsumedIDs: [Int64]
    ) throws -> ModelCheckpoint {
        try read(
            expectedIdentity: expectedIdentity, expectedSegments: expectedSegments, expectedConsumedIDs: expectedConsumedIDs, pageArena: nil
        )
    }
    package func read(
        expectedIdentity: CheckpointIdentity, expectedSegments: [CheckpointSegment],
        expectedConsumedIDs: [Int64], pageArena: QSAPageArena?
    ) throws -> ModelCheckpoint {
        try readPaged(
            expectedIdentity: expectedIdentity, expectedSegments: expectedSegments,
            expectedConsumedIDs: expectedConsumedIDs, pageArena: pageArena)
    }

    package static func expectedSections(tokens: Int, hasLogits: Bool) -> [(String, String, UInt64)] {
        var output: [(String, String, UInt64)] = [
            ("consumed_ids", "i64", UInt64(tokens) * 8), ("history", "i64", 16),
            ("ple_convolution", "u16", 9 * 10_240 * 2),
        ]
        for layer in 0..<48 where layer % 4 != 3 {
            output.append(("gdn.\(layer).recurrent", "f32", 48 * 128 * 128 * 4))
            output.append(("gdn.\(layer).tail", "u16", 3 * 10_240 * 2))
        }
        for layer in 0..<48 where layer % 4 == 3 {
            output.append(("qsa.\(layer).keys", "u16", UInt64(tokens) * 512 * 2))
            output.append(("qsa.\(layer).values", "u16", UInt64(tokens) * 512 * 2))
            output.append(("qsa.\(layer).pooled", "u16", UInt64(tokens / 4) * 128 * 2))
            output.append(("qsa.\(layer).tail", "u16", UInt64(tokens % 4) * 128 * 2))
        }
        if hasLogits { output.append(("next_logits", "f32", 248_320 * 4)) }
        return output
    }

    package func withStoreLock<T>(_ body: () throws -> T) throws -> T {
        let url = directory.appendingPathComponent(".checkpoint.lock")
        let fd = url.withUnsafeFileSystemRepresentation { Darwin.open($0!, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600) }
        guard fd >= 0 else { throw EngineError("Checkpoint lock open failed: errno=\(errno)") }
        defer { Darwin.close(fd) }
        while flock(fd, LOCK_EX) != 0 { if errno != EINTR { throw EngineError("Checkpoint store lock failed") } }
        defer { flock(fd, LOCK_UN) }
        return try body()
    }
    /// Store lock held. The flock serializes writers, so every staging file is a killed
    /// writer's leftover that would otherwise hold old+staged quota forever. Only regular
    /// files go; lstat/unlink never follow a symlink.
    func removeLeftoverStaging() throws {
        for name in try FileManager.default.contentsOfDirectory(atPath: directory.path)
        where name.hasPrefix(".checkpoint-") && name.hasSuffix(".tmp") || name == ".state.bin.staging" || name == ".state.json.staging" {
            let path = directory.appendingPathComponent(name)
            var info = stat()
            guard path.withUnsafeFileSystemRepresentation({ lstat($0!, &info) }) == 0 else { continue }
            guard (info.st_mode & S_IFMT) == S_IFREG else {
                // Family staging names are fixed and would be opened by the next writer. Never leave a symlink
                // there for createFile/FileHandle to follow; an unexpected object requires explicit repair.
                if name == ".state.bin.staging" || name == ".state.json.staging" {
                    throw EngineError("Unexpected family checkpoint staging file type", code: "storage_unavailable")
                }
                continue
            }
            guard path.withUnsafeFileSystemRepresentation({ unlink($0!) }) == 0 || errno == ENOENT else {
                throw EngineError("Leftover checkpoint staging removal failed: errno=\(errno)")
            }
        }
    }
    /// macOS fsync leaves data in the drive's write cache; F_FULLFSYNC flushes it where supported.
    static func fullSync(_ fd: Int32) -> Bool {
        fcntl(fd, F_FULLFSYNC) == 0 || ([ENOTSUP, EINVAL, ENOTTY].contains(errno) && fsync(fd) == 0)
    }
    package func managedBytes() throws -> UInt64 {
        var total: UInt64 = 0
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey], options: [])
        for entry in entries
        where entry.lastPathComponent == "checkpoint.origami"
            || (entry.lastPathComponent.hasPrefix(".checkpoint-") && entry.pathExtension == "tmp")
        {
            let info = try entry.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard info.isRegularFile == true, let size = info.fileSize, size >= 0,
                UInt64(size) <= maximumBytes, total <= maximumBytes - UInt64(size)
            else {
                throw EngineError("Existing checkpoint artifacts exceed quota or are not regular files")
            }
            total += UInt64(size)
        }
        return total
    }
}
