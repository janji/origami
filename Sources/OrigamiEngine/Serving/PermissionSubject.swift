import Darwin
import Foundation

/// The program macOS lists under Privacy & Security > Files & Folders for this process. A privacy permission is stored
/// for the program that started the process tree (a launchd job's own program, a terminal app), so when something
/// else starts `origami` (a launcher, a shell in a terminal) that program is the one to allow.
public enum PermissionSubject {
    /// The topmost ancestor below launchd, by name (an app's name for a program inside its bundle).
    public static let name = name(of: getpid())

    package static func name(of pid: pid_t, parent: (pid_t) -> pid_t? = parent, path: (pid_t) -> String? = path) -> String {
        var top = pid
        for _ in 0..<64 {
            guard let above = parent(top), above > 1 else { break }
            top = above
        }
        guard let file = path(top) else { return "origami" }
        if let bundle = file.range(of: ".app/Contents/MacOS/") { return (String(file[..<bundle.lowerBound]) as NSString).lastPathComponent }
        return (file as NSString).lastPathComponent
    }
    static func parent(_ pid: pid_t) -> pid_t? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        return proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size ? pid_t(info.pbi_ppid) : nil
    }
    static func path(_ pid: pid_t) -> String? {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        return length > 0 ? String(decoding: buffer.prefix(Int(length)), as: UTF8.self) : nil
    }
}
