import Darwin
import Foundation

/// The data folder's state files that Origami writes by itself (tuning results, the drive read tally, the memory
/// history): one that exists but cannot be read is set aside before a new one is written, never overwritten.
public enum StateFile {
    /// Renames `file` to `<name>.unreadable-<date>` and says so on standard error (the server's log); false when it
    /// could not be moved, and then it must not be written either.
    public static func setAside(_ file: URL) -> Bool {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let aside = file.appendingPathExtension("unreadable-" + stamp)
        guard (try? FileManager.default.moveItem(at: file, to: aside)) != nil else {
            FileHandle.standardError.write(Data("note: \(file.path) cannot be read and was left as it is\n".utf8))
            return false
        }
        FileHandle.standardError.write(
            Data("note: \(file.path) could not be read; it is kept as \(aside.lastPathComponent) and a new one is started\n".utf8))
        return true
    }

    /// What a writer starts from: the file's contents decoded by `decode`, `empty` when there is no file, or when it
    /// could not be read and was set aside; nil when it could neither be read nor set aside (write nothing then).
    public static func load<T>(_ file: URL, empty: T, _ decode: (Data) throws -> T) -> T? {
        guard FileManager.default.fileExists(atPath: file.path) else { return empty }
        if let data = try? Data(contentsOf: file), let value = try? decode(data) { return value }
        return setAside(file) ? empty : nil
    }

    /// Runs `body`, a read-modify-write of `file`, holding an advisory lock (`flock`) on its hidden sibling
    /// `.<name>.lock`, so writers in other processes (two `origami config set`, the command line and a server) or on
    /// other threads never lose each other's changes. The write itself stays an atomic replace.
    public static func locked<T>(_ file: URL, _ body: () throws -> T) throws -> T {
        let folder = file.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let path = folder.appendingPathComponent("." + file.lastPathComponent + ".lock").path
        let fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            // A folder that cannot be written is reported as such (the command line names the folder and the fix).
            let code = errno
            if [EACCES, EPERM, EROFS, ENOSPC].contains(code) {
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(code), userInfo: [NSFilePathErrorKey: file.path])
            }
            throw EngineError("Cannot lock \(file.path) (\(String(cString: strerror(code)))); nothing was saved")
        }
        defer { close(fd) }
        while flock(fd, LOCK_EX) != 0 {
            guard errno == EINTR else {
                throw EngineError("Cannot lock \(file.path) (\(String(cString: strerror(errno)))); nothing was saved")
            }
        }
        return try body()
    }
}
