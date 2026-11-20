import Foundation

/// Validation for an explicitly selected Origami memory budget. Automatic planning keeps its existing policy.
/// The macOS wired ceiling and Metal's recommended working set are distinct; neither replaces OS headroom.
public enum WiredMemoryPolicy {
    public static let mebibyte: UInt64 = 1 << 20
    public static let gibibyte: UInt64 = 1 << 30

    /// Leave at least 3 GiB or one eighth of physical RAM for macOS. Round down to the kernel setting's MiB unit
    /// so rounding a requested budget up cannot make its required ceiling cross this bound.
    public static func maximumBudgetBytes(physicalMemoryBytes: UInt64) -> UInt64 {
        let headroom = max(3 * gibibyte, physicalMemoryBytes / 8)
        guard physicalMemoryBytes > headroom else { return 0 }
        return (physicalMemoryBytes - headroom) / mebibyte * mebibyte
    }

    public struct Plan: Sendable, Equatable {
        public let requestedBudgetBytes: UInt64
        public let maximumBudgetBytes: UInt64
        public let currentLimitMiB: UInt64
        /// Positive explicit sysctl value, or Metal's advertised default when the raw value is zero.
        public let effectiveLimitMiB: UInt64
        public let requiredLimitMiB: UInt64
        public var needsRaise: Bool { requiredLimitMiB > effectiveLimitMiB }
    }

    /// Zero is the driver's automatic setting, not a zero-byte limit. Only in this case use the advertised
    /// default supplied by Metal. A missing/negative/unknown value cannot be treated as automatic.
    public static func plan(
        requestedBudgetBytes: UInt64, physicalMemoryBytes: UInt64,
        currentLimitMiB: UInt64?, defaultLimitBytes: UInt64? = nil
    ) throws -> Plan {
        let maximum = maximumBudgetBytes(physicalMemoryBytes: physicalMemoryBytes)
        guard requestedBudgetBytes > 0, requestedBudgetBytes <= maximum else {
            throw EngineError(
                "The memory Origami may keep must be positive and at most \(maximum) bytes on this Mac, leaving memory for macOS",
                code: "invalid_configuration")
        }
        guard let currentLimitMiB, currentLimitMiB <= UInt64(Int32.max) else {
            throw EngineError(
                "macOS did not report a supported GPU memory limit; no system change was made", code: "wired_memory_unavailable")
        }
        let effective = currentLimitMiB == 0 ? (defaultLimitBytes ?? 0) / mebibyte : currentLimitMiB
        guard effective > 0, effective <= UInt64(Int32.max),
            currentLimitMiB != 0 || (defaultLimitBytes ?? 0) <= physicalMemoryBytes
        else {
            throw EngineError(
                "macOS uses its automatic GPU memory limit, but no valid Metal default was reported", code: "wired_memory_unavailable")
        }
        let required = requestedBudgetBytes / mebibyte + (requestedBudgetBytes % mebibyte == 0 ? 0 : 1)
        guard required <= UInt64(Int32.max) else {
            throw EngineError("The requested GPU memory limit exceeds the supported kernel range", code: "invalid_configuration")
        }
        return Plan(
            requestedBudgetBytes: requestedBudgetBytes, maximumBudgetBytes: maximum,
            currentLimitMiB: currentLimitMiB, effectiveLimitMiB: effective, requiredLimitMiB: required)
    }
}
