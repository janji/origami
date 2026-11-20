import Foundation

/// Per drive (lane 0 the model's drive, then each copy place), the seconds it had at least one read running (busy) or
/// none (idle) and the bytes its reads finished, each split by what the serving thread was doing then: inside a fetch
/// (`reads`), waiting on the GPU (`gpu`), or anything else (`other`). Counted only while a host state is set (decode);
/// `idle[lane][reads]` is a drive idle while the host waits on the other drives. Timing only: one lock and one clock
/// read per read start, read end and state change, never a wait.
package final class DriveActivity: @unchecked Sendable {
    package enum Host: Int, CaseIterable {
        case reads, gpu, other
        package var name: String { ["reads", "gpu", "other"][rawValue] }
    }
    /// Seconds and bytes since start, `[lane][host]`, and the wall time in each host state.
    package struct Totals: Sendable {
        package var busy: [[Double]], idle: [[Double]], bytes: [[Double]]
        package var wall: [Double]
    }
    private let lock = NSLock()
    private let clock: () -> UInt64
    private var host: Host?
    private var running: [Int]
    private var at: UInt64
    private var totals: Totals

    package init(lanes: Int, clock: @escaping () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) {
        self.clock = clock
        running = Array(repeating: 0, count: lanes)
        at = clock()
        let zero = Array(repeating: Array(repeating: 0.0, count: Host.allCases.count), count: lanes)
        totals = Totals(busy: zero, idle: zero, bytes: zero, wall: Array(repeating: 0, count: Host.allCases.count))
    }

    private func advanceLocked() {
        let now = clock()
        defer { at = now }
        guard let host else { return }
        let seconds = Double(now &- min(now, at)) / 1e9
        totals.wall[host.rawValue] += seconds
        for lane in running.indices {
            if running[lane] > 0 { totals.busy[lane][host.rawValue] += seconds } else { totals.idle[lane][host.rawValue] += seconds }
        }
    }
    /// Starts counting (a decode pass begins) in `other`, or stops (nil).
    package func window(_ open: Bool) {
        lock.withLock {
            advanceLocked()
            host = open ? .other : nil
        }
    }
    /// The serving thread is now in `state` (only while counting): the state it left, for `restore`.
    package func enter(_ state: Host) -> Host? {
        lock.withLock {
            guard let previous = host else { return nil }
            advanceLocked()
            host = state
            return previous
        }
    }
    package func restore(_ previous: Host?) {
        guard let previous else { return }
        lock.withLock {
            guard host != nil else { return }
            advanceLocked()
            host = previous
        }
    }
    package func began(_ lane: Int) {
        lock.withLock {
            guard running.indices.contains(lane) else { return }
            advanceLocked()
            running[lane] += 1
        }
    }
    package func ended(_ lane: Int, bytes: Int) {
        lock.withLock {
            guard running.indices.contains(lane), running[lane] > 0 else { return }
            advanceLocked()
            running[lane] -= 1
            if let host { totals.bytes[lane][host.rawValue] += Double(bytes) }
        }
    }
    /// `count` lanes from now (a copy attached or dropped, while no read runs); the first lanes keep their totals.
    package func setLanes(_ count: Int) {
        lock.withLock {
            advanceLocked()
            let zero = Array(repeating: 0.0, count: Host.allCases.count)
            func fit(_ rows: [[Double]]) -> [[Double]] {
                Array(rows.prefix(count)) + Array(repeating: zero, count: max(0, count - rows.count))
            }
            totals.busy = fit(totals.busy)
            totals.idle = fit(totals.idle)
            totals.bytes = fit(totals.bytes)
            running = Array(running.prefix(count)) + Array(repeating: 0, count: max(0, count - running.count))
        }
    }
    package var snapshot: Totals {
        lock.withLock {
            advanceLocked()
            return totals
        }
    }
    /// `drive_activity.<lane>.{busy,idle}_seconds.<host>`, `.bytes.<host>` and `drive_activity.wall_seconds.<host>`;
    /// lane 0 `model_drive`, the others `copy_place_<n>`.
    package static func notes(_ totals: Totals) -> [String: Double] {
        var notes: [String: Double] = [:]
        for host in Host.allCases {
            notes["drive_activity.wall_seconds." + host.name] = totals.wall[host.rawValue]
            for lane in totals.busy.indices {
                let name = "drive_activity." + (lane == 0 ? "model_drive" : "copy_place_\(lane - 1)")
                notes[name + ".busy_seconds." + host.name] = totals.busy[lane][host.rawValue]
                notes[name + ".idle_seconds." + host.name] = totals.idle[lane][host.rawValue]
                notes[name + ".bytes." + host.name] = totals.bytes[lane][host.rawValue]
            }
        }
        return notes
    }
}
