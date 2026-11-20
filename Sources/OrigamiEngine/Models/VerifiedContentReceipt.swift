import CryptoKit
import Darwin
import Foundation

/// Explicit, streamed content verification. This does not replace the serving cache's local-incarnation key.
/// A manifest declares the exact ordered files to verify; it is not proof of publisher authenticity or that all
/// files consumed by a model are listed. Receipts are trusted local records, not signed attestations. Importing a
/// receipt onto another file incarnation always rehashes the bytes against its embedded manifest.
package enum VerifiedContentReceipt {
    package struct File: Codable, Equatable {
        package let path: String
        package let size: UInt64
        package let sha256: String
        package init(path: String, size: UInt64, sha256: String) {
            self.path = path
            self.size = size
            self.sha256 = sha256
        }
    }
    package struct Manifest: Codable, Equatable {
        package let schemaVersion: Int
        package let algorithm: String
        package let files: [File]
        package init(files: [File]) {
            schemaVersion = 1
            algorithm = "sha256"
            self.files = files
        }
    }
    package struct Stamp: Codable, Equatable {
        let device: Int64, inode: UInt64, size: UInt64
        let modifiedSeconds: Int64, modifiedNanoseconds: Int64
        let changedSeconds: Int64, changedNanoseconds: Int64
        let bornSeconds: Int64, bornNanoseconds: Int64

        init(_ info: stat) throws {
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size >= 0 else {
                throw EngineError("Content verification requires a regular file")
            }
            device = Int64(info.st_dev)
            inode = UInt64(info.st_ino)
            size = UInt64(info.st_size)
            modifiedSeconds = Int64(info.st_mtimespec.tv_sec)
            modifiedNanoseconds = Int64(info.st_mtimespec.tv_nsec)
            changedSeconds = Int64(info.st_ctimespec.tv_sec)
            changedNanoseconds = Int64(info.st_ctimespec.tv_nsec)
            bornSeconds = Int64(info.st_birthtimespec.tv_sec)
            bornNanoseconds = Int64(info.st_birthtimespec.tv_nsec)
        }
    }
    package struct Binding: Codable, Equatable {
        let resolvedPath: String
        let stamp: Stamp
    }
    package struct Receipt: Codable {
        package let schemaVersion: Int
        package let verifiedAt: String
        package let contentIdentity: String
        package let manifest: Manifest
        package let root: String
        let bindings: [Binding]
    }

    private static let maximumMetadataBytes = 8 << 20

    package static func readManifest(_ url: URL) throws -> Manifest {
        let manifest = try JSONDecoder().decode(Manifest.self, from: metadata(url))
        try validate(manifest)
        return manifest
    }

    package static func readReceipt(_ url: URL) throws -> Receipt {
        let receipt = try JSONDecoder().decode(Receipt.self, from: metadata(url))
        try validate(receipt)
        return receipt
    }

    /// Ordered, length-prefixed fields make this identity independent of filesystem paths, timestamps and JSON
    /// formatting, while distinguishing shard order and file names. Only a successful verify binds it to bytes.
    package static func identity(_ manifest: Manifest) throws -> String {
        try validate(manifest)
        var hash = SHA256()
        func fields(_ values: [String]) { hash.update(data: Data(values.map { "\($0.utf8.count):\($0)" }.joined().utf8)) }
        fields(["origami-ordered-content-v1", String(manifest.files.count)])
        for (index, file) in manifest.files.enumerated() {
            fields([String(index), file.path, String(file.size), file.sha256])
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Hash all listed files and publish only after every digest and incarnation has been checked. The optional
    /// checkpoint/progress hooks permit cancellation and tiny deterministic race fixtures; there is no model load.
    @discardableResult
    package static func verify(
        _ manifest: Manifest, root inputRoot: URL, writingTo destination: URL,
        chunkBytes: Int = 8 << 20, checkpoint: () throws -> Void = {},
        progress: ((Int, UInt64) throws -> Void)? = nil
    ) throws -> Receipt {
        try validate(manifest)
        guard chunkBytes > 0 && chunkBytes <= 8 << 20 else { throw EngineError("Invalid content verification chunk size") }
        let root = inputRoot.standardizedFileURL.resolvingSymlinksInPath()
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory), directory.boolValue else {
            throw EngineError("Content verification root is not a directory: \(root.path)")
        }
        var bindings: [Binding] = []
        for (index, expected) in manifest.files.enumerated() {
            try checkpoint()
            let logical = root.appendingPathComponent(expected.path)
            let resolved = logical.resolvingSymlinksInPath()
            let fd = Darwin.open(resolved.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
            guard fd >= 0 else { throw EngineError("Cannot open content file: \(expected.path)") }
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            defer { try? handle.close() }
            let before = try stamp(fd)
            let binding = Binding(resolvedPath: resolved.path, stamp: before)
            try check(binding, logical: logical)
            guard before.size == expected.size else { throw EngineError("Content size differs: \(expected.path)") }
            var hash = SHA256()
            var readBytes: UInt64 = 0
            while true {
                try checkpoint()
                guard let bytes = try handle.read(upToCount: chunkBytes), !bytes.isEmpty else { break }
                let (total, overflow) = readBytes.addingReportingOverflow(UInt64(bytes.count))
                guard !overflow, total <= expected.size else { throw EngineError("Content grew during verification: \(expected.path)") }
                readBytes = total
                hash.update(data: bytes)
                try progress?(index, readBytes)
            }
            try checkpoint()
            guard readBytes == expected.size, try stamp(fd) == before else {
                throw EngineError("Content changed during verification: \(expected.path)")
            }
            try check(binding, logical: logical)
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == expected.sha256 else {
                throw EngineError("Content SHA256 differs: \(expected.path)")
            }
            bindings.append(binding)
        }
        let receipt = Receipt(
            schemaVersion: 1, verifiedAt: ISO8601DateFormatter().string(from: Date()),
            contentIdentity: try identity(manifest), manifest: manifest, root: root.path, bindings: bindings)
        // An earlier shard can change while a later shard is being hashed. Recheck the complete set before commit.
        try check(receipt)
        try checkpoint()
        try write(receipt, to: destination, checkpoint: checkpoint)
        return receipt
    }

    /// A cheap revalidation of a trusted local receipt, not a new read/verification of the payload. Copied files,
    /// replacement inodes, ctime changes and symlink retargeting invalidate it even when mtime/size are restored.
    package static func check(_ receipt: Receipt) throws {
        try validate(receipt)
        let root = URL(fileURLWithPath: receipt.root, isDirectory: true)
        guard root.standardizedFileURL.resolvingSymlinksInPath().path == receipt.root else {
            throw EngineError("Content receipt root changed; verify or import again")
        }
        for (file, binding) in zip(receipt.manifest.files, receipt.bindings) {
            try check(binding, logical: root.appendingPathComponent(file.path))
        }
    }

    private static func validate(_ manifest: Manifest) throws {
        guard manifest.schemaVersion == 1, manifest.algorithm == "sha256", !manifest.files.isEmpty,
            manifest.files.count <= 4096
        else { throw EngineError("Unsupported or empty content manifest") }
        var names = Set<String>()
        var total: UInt64 = 0
        for file in manifest.files {
            let parts = file.path.split(separator: "/", omittingEmptySubsequences: false)
            guard !file.path.isEmpty, file.path.utf8.count <= 4096, !file.path.contains("\0"),
                parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }), names.insert(file.path).inserted,
                file.sha256.utf8.count == 64,
                file.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
            else {
                throw EngineError("Content manifest needs unique relative paths and lowercase SHA256 digests")
            }
            let (sum, overflow) = total.addingReportingOverflow(file.size)
            guard !overflow else { throw EngineError("Content manifest byte count overflows") }
            total = sum
        }
    }

    private static func validate(_ receipt: Receipt) throws {
        try validate(receipt.manifest)
        guard receipt.schemaVersion == 1, receipt.root.hasPrefix("/"), !receipt.root.contains("\0"),
            receipt.contentIdentity == (try identity(receipt.manifest)), receipt.bindings.count == receipt.manifest.files.count
        else {
            throw EngineError("Invalid or unsupported content receipt")
        }
        for (file, binding) in zip(receipt.manifest.files, receipt.bindings) {
            guard binding.resolvedPath.hasPrefix("/"), !binding.resolvedPath.contains("\0"), binding.stamp.size == file.size else {
                throw EngineError("Invalid content receipt file binding")
            }
        }
    }

    private static func stamp(_ fd: Int32) throws -> Stamp {
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw EngineError("Cannot inspect content file descriptor") }
        return try Stamp(info)
    }

    private static func check(_ binding: Binding, logical: URL) throws {
        let resolved = logical.standardizedFileURL.resolvingSymlinksInPath()
        var info = stat()
        guard resolved.path == binding.resolvedPath, lstat(resolved.path, &info) == 0,
            try Stamp(info) == binding.stamp
        else {
            throw EngineError("Content file incarnation changed: \(logical.path); verify or import again")
        }
    }

    private static func metadata(_ url: URL) throws -> Data {
        let fd = Darwin.open(url.standardizedFileURL.resolvingSymlinksInPath().path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard fd >= 0 else { throw EngineError("Cannot open content metadata: \(url.path)") }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer { try? handle.close() }
        guard try stamp(handle.fileDescriptor).size <= maximumMetadataBytes else { throw EngineError("Content metadata exceeds 8 MiB") }
        let data = try handle.read(upToCount: maximumMetadataBytes + 1) ?? Data()
        guard data.count <= maximumMetadataBytes else { throw EngineError("Content metadata exceeds 8 MiB") }
        return data
    }

    private static func write(_ receipt: Receipt, to input: URL, checkpoint: () throws -> Void) throws {
        let destination = input.standardizedFileURL
        // A typo must never replace a model payload or arbitrary user file with a receipt.
        let resolved = destination.resolvingSymlinksInPath().path
        guard !receipt.bindings.contains(where: { $0.resolvedPath == resolved }) else {
            throw EngineError("Receipt destination is a verified payload")
        }
        var previous = stat()
        if lstat(destination.path, &previous) == 0 {
            guard (previous.st_mode & S_IFMT) == S_IFREG else { throw EngineError("Receipt destination is not a regular file") }
            guard
                !receipt.bindings.contains(where: { $0.stamp.device == Int64(previous.st_dev) && $0.stamp.inode == UInt64(previous.st_ino) }
                )
            else {
                throw EngineError("Receipt destination aliases a verified payload")
            }
            _ = try readReceipt(destination)  // Existing receipts may be replaced; unrelated files may not.
        } else if errno != ENOENT {
            throw EngineError("Cannot inspect receipt destination")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(receipt)
        guard data.count <= maximumMetadataBytes else { throw EngineError("Content receipt exceeds 8 MiB") }
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".origami-content-\(UUID().uuidString).tmp")
        let fd = Darwin.open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw EngineError("Cannot create receipt staging file; its parent directory must exist") }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer {
            try? handle.close()
            try? FileManager.default.removeItem(at: temporary)
        }
        try handle.write(contentsOf: data)
        try handle.synchronize()
        try checkpoint()
        try check(receipt)
        guard rename(temporary.path, destination.path) == 0 else { throw EngineError("Cannot publish content receipt") }
        // Atomic rename prevents readers from observing a partial receipt. Directory fsync is best effort on
        // filesystems that do not support it; this is not a guarantee against a sudden loss of storage power.
        let directory = Darwin.open(destination.deletingLastPathComponent().path, O_RDONLY | O_CLOEXEC)
        if directory >= 0 {
            _ = fsync(directory)
            Darwin.close(directory)
        }
    }
}
