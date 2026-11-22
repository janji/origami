import Darwin
import Foundation

/// Who holds this Mac's memory: each process's physical footprint (what Activity Monitor shows as its memory, Metal
/// buffers included), named as a person knows it, its app (an app's helpers counted with it). Processes of other users
/// do not answer and are left out.
public enum ProcessMemory {
    /// A process's physical footprint, or nil.
    public static func footprint(_ pid: pid_t) -> UInt64? {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        return status == 0 ? info.ri_phys_footprint : nil
    }

    /// The name a person knows a process by: the outermost `.app` its executable is in, else the executable's name, or
    /// for one named by its version (`…/claude/versions/2.1.3`) the folder that names it.
    public static func name(path: String) -> String {
        let parts = path.split(separator: "/")
        if let app = parts.first(where: { $0.hasSuffix(".app") }) { return String(app.dropLast(4)) }
        let generic: Set<Substring> = ["versions", "bin", "sbin", "libexec", "current"]
        return String(parts.last { !($0.first?.isNumber ?? true) && !generic.contains($0) } ?? parts.last ?? "")
    }

    /// Every process that answers: pid, name and footprint.
    public static func all() -> [(pid: pid_t, name: String, bytes: UInt64)] {
        var pids = [pid_t](repeating: 0, count: Int(max(0, proc_listallpids(nil, 0))) + 64)
        let count = Int(proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size)))
        return pids.prefix(max(0, min(count, pids.count))).compactMap { pid in
            var path = [UInt8](repeating: 0, count: 4096)  // PROC_PIDPATHINFO_MAXSIZE
            guard pid > 0, proc_pidpath(pid, &path, UInt32(path.count)) > 0, let bytes = footprint(pid) else { return nil }
            return (pid, name(path: String(decoding: path.prefix { $0 != 0 }, as: UTF8.self)), bytes)
        }
    }

    /// The `count` largest apps other than `own` (Origami's own processes), by their processes' footprints together.
    public static func others(_ count: Int = 3, own: String = ProcessInfo.processInfo.processName) -> [(name: String, bytes: UInt64)] {
        var apps: [String: UInt64] = [:]
        for process in all() where process.name != own { apps[process.name, default: 0] += process.bytes }
        return apps.sorted { $0.value > $1.value }.prefix(count).map { ($0.key, $0.value) }
    }
}
