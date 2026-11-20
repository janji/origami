import Darwin
import Foundation

/// The hot copy's free-space and payload-name rules (no numerical policy).
package enum HotCopyRefreshPlan {
    /// Initial-build minimum never rejects an existing valid subset. Its missing capacity is
    /// a growth goal, not an admission requirement, even when external writers reduce room.
    package static func capacity(wanted: UInt64, held: UInt64, free: UInt64, reserve: UInt64) -> UInt64? {
        let (sum, overflow) = free.addingReportingOverflow(held)
        let available = overflow ? UInt64.max : sum
        let room = available > reserve ? available - reserve : 0
        let budget = min(wanted, held > 0 ? max(held, room) : room)
        return held > 0 || budget >= min(wanted, 20_000_000_000) ? budget : nil
    }

    package static func room(free: UInt64, reserve: UInt64) -> UInt64 { free > reserve ? free - reserve : 0 }

    package static func validPayload(_ name: String) -> Bool {
        if name == "experts.bin" { return true }
        guard name.hasPrefix("experts-generation-"), name.hasSuffix(".bin") else { return false }
        return UUID(uuidString: String(name.dropFirst("experts-generation-".count).dropLast(4))) != nil
    }

    /// Actual available blocks, excluding reclaimable/purgeable capacity. Unknown fails closed.
    package static func available(_ folder: URL) throws -> UInt64 {
        var path = folder
        while !FileManager.default.fileExists(atPath: path.path), path.path != "/" { path.deleteLastPathComponent() }
        var info = statfs()
        guard statfs(path.path, &info) == 0 else { throw EngineError("Cannot measure hot-copy free space: errno \(errno)") }
        let (bytes, overflow) = UInt64(info.f_bavail).multipliedReportingOverflow(by: UInt64(info.f_bsize))
        guard !overflow else { throw EngineError("Invalid hot-copy free-space measurement") }
        return bytes
    }
}
