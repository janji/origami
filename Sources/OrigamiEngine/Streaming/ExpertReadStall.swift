import Foundation

/// How long a demand read may take before the drive is taken for stopped (ROADMAP R2): `multiple` times the time its
/// bytes take at the drive's healthy (measured) rate, at least `floor` (a drive waking from sleep). A hung pread cannot
/// be cancelled, so the request fails with `drive_stalled` while the reader keeps its slot: the slot stays `reading`
/// (no policy evicts or retires it) until the read returns and its owner frees it.
package enum ExpertReadStall {
    /// Settable for checks.
    nonisolated(unsafe) package static var floor = 60.0
    nonisolated(unsafe) package static var multiple = 20.0

    package static func bound(bytes: Int, bytesPerSecond rate: Double) -> Double {
        max(floor, rate > 0 ? multiple * Double(bytes) / rate : 0)
    }

    package static func error(bytes: Int, seconds: Double, what: String = "expert") -> EngineError {
        EngineError(
            String(
                format: "The model drive stopped answering: %.1f MB of %@ reads did not return in %.1f s", Double(bytes) / 1e6, what,
                seconds),
            code: "drive_stalled")
    }

    /// Joins the readers in `group` (`bytes` of `what` rows; no measured rate, so 1 GB/s), polling `shouldContinue`
    /// every millisecond. A cancel calls `cancel` and throws `IOOperationCancelled` once the readers stop; readers still
    /// out at the bound are left running (each keeps alive what it writes into), `cancel` is called and the wait
    /// throws `drive_stalled` (or `IOOperationCancelled` after a cancel). Returns once they joined.
    package static func join(
        _ group: DispatchGroup, bytes: Int, what: String, shouldContinue: () -> Bool, cancel: () -> Void
    ) throws {
        let bound = Self.bound(bytes: bytes, bytesPerSecond: 1e9)
        let start = DispatchTime.now().uptimeNanoseconds
        var cancelled = false
        while group.wait(timeout: .now() + .milliseconds(1)) == .timedOut {
            if !cancelled, !shouldContinue() {
                cancel()
                cancelled = true
            }
            let waited = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            if waited > bound {
                cancel()
                throw cancelled ? IOOperationCancelled() : error(bytes: bytes, seconds: waited, what: what)
            }
        }
        if cancelled { throw IOOperationCancelled() }
    }
}
