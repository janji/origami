import Foundation

/// Comparison eligibility, not memory governance. Host counters are system-wide: ordinary background
/// swapping, warning pressure and a stable power/thermal mode do not invalidate useful measurements.
/// A rejected sample never changes the user's limit or disables the governor's pressure response.
enum TuningConditions {
    struct Snapshot: Sendable {
        let monotonic: TimeInterval
        let physicalMemoryBytes: UInt64
        let swapoutBytes: UInt64
        /// macOS: normal 1, warning 2, critical 4. An unavailable reading may be passed as 0.
        let memoryPressure: Int
        /// ProcessInfo.ThermalState.rawValue: nominal 0, fair 1, serious 2, critical 3.
        let thermalState: Int
        let lowPower: Bool
        let budgetBytes: UInt64?

        init(
            monotonic: TimeInterval, physicalMemoryBytes: UInt64, swapoutBytes: UInt64,
            memoryPressure: Int, thermalState: Int, lowPower: Bool, budgetBytes: UInt64? = nil
        ) {
            self.monotonic = monotonic
            self.physicalMemoryBytes = physicalMemoryBytes
            self.swapoutBytes = swapoutBytes
            self.memoryPressure = memoryPressure
            self.thermalState = thermalState
            self.lowPower = lowPower
            self.budgetBytes = budgetBytes
        }
    }

    enum Disposition: String, Sendable { case usable, disrupted, critical }
    struct Assessment: Sendable {
        let disposition: Disposition
        let reason: String?
        let freshSwapBytes: UInt64
        let duration: TimeInterval?
        let budgetChangeFraction: Double?
    }

    /// Thresholds scale with RAM, with floors for small machines and caps for larger ones. These are
    /// screening heuristics, not claims that a particular swap rate caused an engine slowdown.
    struct Thresholds: Sendable {
        let softSwapBytes: UInt64
        let severeSwapBytes: UInt64
    }
    static func thresholds(physicalMemoryBytes ram: UInt64) -> Thresholds {
        let mib: UInt64 = 1 << 20
        // Divide before multiplying: even a malformed UInt64.max host reading cannot overflow.
        return Thresholds(
            softSwapBytes: min(512 * mib, max(128 * mib, ram / 50)),
            severeSwapBytes: min(2_048 * mib, max(512 * mib, ram / 25 * 2)))
    }

    static func assess(_ current: Snapshot, after previous: Snapshot? = nil) -> Assessment {
        let fresh = previous.map { current.swapoutBytes >= $0.swapoutBytes ? current.swapoutBytes - $0.swapoutBytes : 0 } ?? 0
        let elapsed = previous.map { current.monotonic - $0.monotonic }
        let duration = elapsed.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
        let fraction: Double?
        if let before = previous?.budgetBytes, before > 0, let after = current.budgetBytes, after > 0 {
            fraction = abs(Double(after) - Double(before)) / Double(before)
        } else {
            fraction = nil
        }
        func result(_ disposition: Disposition, _ reason: String? = nil) -> Assessment {
            Assessment(
                disposition: disposition, reason: reason, freshSwapBytes: fresh,
                duration: duration, budgetChangeFraction: fraction)
        }
        if current.memoryPressure >= 4 {
            return result(.critical, "macOS reports critical memory pressure (level \(current.memoryPressure))")
        }
        if current.thermalState >= 3 {
            return result(.critical, "macOS reports critical thermal pressure (state \(current.thermalState))")
        }
        guard let previous else { return result(.usable) }
        guard let duration else { return result(.disrupted, "The sample's elapsed-time reading is unavailable or did not advance") }
        if previous.lowPower != current.lowPower {
            return result(
                .disrupted,
                "Low Power Mode changed during the sample (\(previous.lowPower ? "on" : "off") to \(current.lowPower ? "on" : "off"))")
        }
        if current.thermalState >= 2, previous.thermalState < 2 {
            return result(
                .disrupted,
                "The Mac entered serious thermal pressure during the sample (state \(previous.thermalState) to \(current.thermalState))")
        }
        if let fraction, fraction > 0.15, let before = previous.budgetBytes, let after = current.budgetBytes {
            return result(
                .disrupted,
                String(
                    format: "Origami's temporary memory allowance changed from %.2f to %.2f GB (%.1f%%) during the sample",
                    Double(before) / 1e9, Double(after) / 1e9, fraction * 100))
        }
        let limits = thresholds(physicalMemoryBytes: current.physicalMemoryBytes)
        let rate = Double(fresh) / duration
        // A brief large burst or sustained heavy paging can distort a sample. A small cumulative
        // trickle over a long request does not. A global counter reset is not a huge unsigned delta.
        let fastBurst = fresh >= limits.softSwapBytes && rate >= Double(limits.softSwapBytes) / 5
        let sustained = fresh >= limits.severeSwapBytes && rate >= Double(limits.softSwapBytes) / 20
        if fastBurst || sustained {
            return result(
                .disrupted,
                String(
                    format: "Heavy system-wide swapping during the sample: %.0f MiB in %.1f s (%.1f MiB/s)",
                    Double(fresh) / 1_048_576, duration, rate / 1_048_576))
        }
        return result(.usable)
    }

    /// A caller may discard/repeat a disturbed warm-up only within both bounds. This has no clock,
    /// sleep, inference, configuration or governor side effect; the caller supplies observed time.
    struct Settling {
        private let deadline: TimeInterval
        private var remaining: Int
        init(startedAt: TimeInterval, maximumSeconds: TimeInterval = 90, maximumExtraSamples: Int = 2) {
            let end = startedAt + max(0, maximumSeconds)
            deadline = startedAt.isFinite && maximumSeconds.isFinite && end.isFinite ? end : -.infinity
            remaining = max(0, maximumExtraSamples)
        }
        mutating func permitRetry(for assessment: Assessment, at now: TimeInterval) -> Bool {
            guard assessment.disposition == .disrupted, now.isFinite, now < deadline, remaining > 0 else { return false }
            remaining -= 1
            return true
        }
    }
}
