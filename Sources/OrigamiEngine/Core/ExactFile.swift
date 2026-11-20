import CExactIO
import CryptoKit
import Darwin
import Foundation
import Synchronization

/// The engine's general error: a message for people and an optional stable `code` (`model_incompatible`,
/// `insufficient_memory`, ...) that callers and the server's error mapping branch on.
public struct EngineError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public let code: String?
    public init(_ description: String, code: String? = nil) {
        self.description = description
        self.code = code
    }
}

/// The fstat identity of an open file (device, inode, size, modification and change times): any change means
/// the file was replaced or rewritten since it was opened.
public struct FileIdentity: Equatable, Codable, Sendable {
    public let device, inode, size: UInt64
    public let modificationSeconds, modificationNanoseconds, changeSeconds, changeNanoseconds: Int64
    init(_ raw: origami_file_identity) {
        device = raw.device
        inode = raw.inode
        size = raw.size
        modificationSeconds = raw.modification_seconds
        modificationNanoseconds = raw.modification_nanoseconds
        changeSeconds = raw.change_seconds
        changeNanoseconds = raw.change_nanoseconds
    }
}

/// Persistent descriptor. Shared reads are safe; callers own and synchronize destination buffers. A model's own
/// files are opened with their `ModelDrive`: when the drive goes and comes back, the descriptor is replaced in place
/// (the same file only) and readers go on.
public final class ExactFile: @unchecked Sendable {
    /// One open descriptor and what `fstat` said of it at open; closed when the last holder (a reader) lets go.
    final class Handle: @unchecked Sendable {
        let descriptor: Int32
        let identity: FileIdentity
        let volumeUUID: String?
        /// The `ModelDrive` generation it was opened under (0 without a drive).
        let generation: UInt64
        private let drive: ModelDrive?
        init(url: URL, bypassCache: Bool, drive: ModelDrive?, generation: UInt64) throws {
            var code: Int32 = ENOENT
            // Checks: an unplugged drive's paths do not open.
            let descriptor =
                drive?.openFails() == true
                ? -1 : url.withUnsafeFileSystemRepresentation { origami_open_readonly($0, bypassCache ? 1 : 0, &code) }
            guard descriptor >= 0 else { throw EngineError("open failed: errno=\(code), path=\(url.path)") }
            var raw = origami_file_identity()
            let result = origami_get_identity(descriptor, &raw)
            guard result == 0 else {
                close(descriptor)
                throw EngineError("fstat failed: errno=\(result), path=\(url.path)")
            }
            self.descriptor = descriptor
            identity = FileIdentity(raw)
            var uuid = [CChar](repeating: 0, count: 37)
            let uuidResult = origami_get_volume_uuid(descriptor, &uuid, uuid.count)
            volumeUUID = uuidResult == 0 ? String(decoding: uuid.prefix(36).map { UInt8(bitPattern: $0) }, as: UTF8.self) : nil
            self.generation = generation
            self.drive = drive
            drive?.opened()
        }
        deinit {
            close(descriptor)
            drive?.closed()
        }
    }
    private let handle: Mutex<Handle>
    private let url: URL
    private let bypassCache: Bool
    private let drive: ModelDrive?
    /// The SHA-256 of `ModelDrive.sample` at the first open, which a reopened file must match.
    private let sample: SHA256.Digest?
    /// For error messages: a failed read names the file (a removed drive, a replaced shard).
    private var path: String { url.path }
    /// What `fstat` said at the first open (a reopened descriptor is the same file; only its device may differ).
    public let identity: FileIdentity
    public let volumeUUID: String?
    /// Bytes this descriptor has read (what the drive was asked for), for checks that compare counters with reads.
    private let bytesRead = Atomic<UInt64>(0)
    package var readBytes: UInt64 { bytesRead.load(ordering: .relaxed) }

    public convenience init(url: URL, bypassCache: Bool = true) throws { try self.init(url: url, bypassCache: bypassCache, drive: nil) }
    /// A model file on `drive` (nil: no reopen; a failed read stays failed).
    package init(url: URL, bypassCache: Bool = true, drive: ModelDrive?) throws {
        // A file first opened while its drive is away fails as its open siblings do.
        try drive?.admit()
        self.url = url
        self.bypassCache = bypassCache
        self.drive = drive
        let opened = try Handle(url: url, bypassCache: bypassCache, drive: drive, generation: drive?.generation ?? 0)
        identity = opened.identity
        volumeUUID = opened.volumeUUID
        sample = drive == nil ? nil : ModelDrive.digest(opened.descriptor, size: opened.identity.size)
        guard drive == nil || sample != nil else { throw EngineError("pread failed: sampling \(url.path)") }
        handle = Mutex(opened)
        drive?.register(self)
    }

    /// The descriptor reads use now; throws `drive_away` at once while the model's drive is away.
    private func current() throws -> Handle {
        try drive?.admit()
        return handle.withLock { $0 }
    }
    /// After `code` failed on `used`: the newer descriptor to read again on once, when the drive was swapped meanwhile;
    /// nil when the failure was not the device leaving. Throws `drive_away` when it left and is not back.
    private func retry(after used: Handle, code: Int32) throws -> Handle? {
        guard let drive, drive.gone(used.descriptor, generation: used.generation, code: code) else { return nil }
        drive.lost(path, generation: used.generation, code: code)
        let now = handle.withLock { $0 }
        if now.generation != used.generation { return now }
        throw drive.awayError ?? EngineError("The model drive is not connected (\(path))", code: "drive_away")
    }

    /// `ModelDrive.reopen`: this file opened again by its path, if it is the one first opened.
    func reopened(generation: UInt64) -> ModelDrive.Reopened {
        guard let drive, let sample else { return .missing }
        guard let opened = try? Handle(url: url, bypassCache: bypassCache, drive: drive, generation: generation) else { return .missing }
        if drive.acceptAny { return .same(opened) }
        let now = opened.identity
        let sameVolume = volumeUUID != nil && opened.volumeUUID == volumeUUID
        var differs: [String] = []
        if now.size != identity.size { differs.append("size") }
        if now.modificationSeconds != identity.modificationSeconds || now.modificationNanoseconds != identity.modificationNanoseconds {
            differs.append("modification time")
        }
        if sameVolume && now.inode != identity.inode { differs.append("inode") }
        if differs.isEmpty {
            guard let digest = ModelDrive.digest(opened.descriptor, size: now.size) else { return .missing }
            if digest != sample { differs.append("content") }
        }
        guard differs.isEmpty else {
            return .different(
                "A different file is at \(path) (its \(differs.joined(separator: ", ")) differs from the file this model loaded): "
                    + "Origami does not read it; reconnect the model's own drive, or Reload to load what is there")
        }
        return .same(opened)
    }
    /// `ModelDrive.reopen`: reads use `new` from now on; readers on the old descriptor keep it until they return.
    func swap(_ new: Handle) -> Handle {
        handle.withLock { held in
            let old = held
            held = new
            return old
        }
    }

    public func validateIdentity() throws {
        var used = try current()
        var raw = origami_file_identity()
        var code = stat(used, &raw)
        if code != 0, let newer = try retry(after: used, code: code) {
            used = newer
            code = stat(used, &raw)
        }
        guard code == 0 && FileIdentity(raw) == used.identity else {
            throw EngineError("File identity or mutation stamps changed, path=\(path)")
        }
    }
    private func stat(_ used: Handle, _ raw: inout origami_file_identity) -> Int32 {
        drive?.dead(used.generation) == true ? EBADF : origami_get_identity(used.descriptor, &raw)
    }

    package func clone(into directoryFD: Int32, name: String) throws {
        try validateIdentity()
        // Held for the call: a drive swap meanwhile must not close the descriptor being cloned.
        let used = try current()
        let code = withExtendedLifetime(used) { name.withCString { origami_clone_to(used.descriptor, directoryFD, $0) } }
        guard code == 0 else { throw EngineError("Atomic checkpoint clone unavailable: errno=\(code)") }
        try validateIdentity()
    }

    public func read(offset: Int64, into buffer: UnsafeMutableRawBufferPointer) throws {
        guard offset >= 0, UInt64(offset) <= identity.size,
            UInt64(buffer.count) <= identity.size - UInt64(offset)
        else {
            throw EngineError("Read outside installed file bounds, path=\(path)")
        }
        var used = try current()
        var result = pread(used, offset: offset, into: buffer)
        // A reader on a descriptor the drive swap replaced reads its whole range again, once, on the new one: no read
        // returns bytes from two descriptors.
        if result.error_code != 0, let newer = try retry(after: used, code: result.error_code) {
            used = newer
            drive?.retriedOnce()
            result = pread(used, offset: offset, into: buffer)
            if result.error_code != 0, try retry(after: used, code: result.error_code) != nil {
                throw drive?.awayError ?? EngineError("The model drive is not connected (\(path))", code: "drive_away")
            }
        }
        guard result.error_code == 0 else {
            throw EngineError("pread failed: errno=\(result.error_code), partialBytes=\(result.bytes_read), path=\(path)")
        }
        bytesRead.add(UInt64(buffer.count), ordering: .relaxed)
    }
    private func pread(_ used: Handle, offset: Int64, into buffer: UnsafeMutableRawBufferPointer) -> origami_read_result {
        let result = origami_pread_exact(used.descriptor, buffer.baseAddress, buffer.count, offset)
        drive?.readHook?()
        // Checks: a descriptor that died while this read ran fails it as on a vanished device, its bytes landed.
        if result.error_code == 0, drive?.dead(used.generation) == true { return origami_read_result(bytes_read: 0, error_code: EIO) }
        return result
    }

    public func read(offset: Int64, length: Int) throws -> Data {
        guard length >= 0, offset >= 0, UInt64(offset) <= identity.size,
            UInt64(length) <= identity.size - UInt64(offset)
        else {
            throw EngineError("Read outside installed file bounds, path=\(path)")
        }
        var data = Data(count: length)
        try data.withUnsafeMutableBytes { try read(offset: offset, into: $0) }
        return data
    }
}
