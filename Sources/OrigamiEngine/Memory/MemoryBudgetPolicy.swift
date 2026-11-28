import Darwin
import Foundation

/// Automatic follows estimated host headroom. An explicit limit is the user's ceiling and starting target;
/// only actual pressure, fresh swapping or a hard physical-ceiling overrun reduces its allowance. In both modes,
/// after a cut the budget regrows to the level that held, then half the way to the cap per 40 s quiet with memory to spare.
public enum MemoryBudgetMode: String, Sendable, Codable {
    case automatic, explicit
}

/// XNU's read-only sysctl exports Dispatch memory-pressure levels, not a free-page estimate.
public enum MemoryPressureLevel: UInt32, Sendable, Codable {
    case normal = 1
    case warning = 2
    case critical = 4
    public var isElevated: Bool { self != .normal }

    /// Nil means unsupported or unrecognized; it must not be interpreted as pressure.
    /// XNU: bsd/kern/kern_memorystatus_notify.c, sysctl_memorystatus_vm_pressure_level.
    public static func current() -> Self? {
        var raw: UInt32 = 0
        var size = MemoryLayout<UInt32>.size
        guard sysctlbyname("kern.memorystatus_vm_pressure_level", &raw, &size, nil, 0) == 0,
            size == MemoryLayout<UInt32>.size
        else { return nil }
        return Self(rawValue: raw)
    }
}

/// Deterministic governor decisions. Inputs are supplied snapshots and monotonic time;
/// this type neither probes the host nor allocates or releases model memory.
package struct MemoryGovernorPolicy: Sendable {
    package enum Reason: Sendable, Equatable {
        case warning, critical, swap, footprint, available, recovery, growthRetry
        /// A cut the Mac asked for (pressure events, fresh swap, short free memory), not the process holding to its own
        /// ceiling: what `origami tune --memory` counts against a limit, the same on every model.
        package var isPressure: Bool { [.warning, .critical, .swap, .available].contains(self) }
    }
    package struct Decision: Sendable, Equatable {
        package let target: UInt64
        package let reason: Reason
        package let freshSwap: UInt64
    }
    private let mode: MemoryBudgetMode
    private let cap, hardCeiling, floor, minimum: UInt64
    private var observedSwapouts: UInt64
    private var sampledSwapouts: UInt64
    private var swapWindowStart: TimeInterval?
    private var swapWindowBytes: UInt64 = 0
    private var swapWindowSamples = 0
    private var pendingSwapRelief = false
    private var lastShrink: TimeInterval = -.infinity
    private var lastShrinkWasUrgent = false
    private var quietSince: TimeInterval?
    private var lastGrowth: TimeInterval?
    private var stableSince: TimeInterval?
    private var recoveryBackoff = 0
    /// The level the last cut left, which regrowth does not pass until 40 s quiet with free memory to spare raise it
    /// half way to the cap (`raised`); nil means the whole cap.
    private var held: UInt64?
    /// Optional allocations stop on real distress even when no further budget reduction is
    /// possible. A quiet recovery interval reopens admission without reducing the saved cap.
    package private(set) var optionalGrowthInhibited = false
    /// Qwen prompt read-ahead (it only fills slots of the cache the current budget already holds) stops only while the
    /// pressure is live: an elevated level, a swap relief, an overrun or short free memory at this step, or a cut still
    /// settling. The recovery after it narrows them through the smaller cache, not through this flag.
    package private(set) var optionalReadsPaused = false
    private var settleUntil: TimeInterval = -.infinity
    private var growthQuietSince: TimeInterval?
    private static let gb: UInt64 = 1_000_000_000
    /// Free memory a growth step keeps above the floor (`fit`'s slack in `step`); a prompt read's arena loan keeps it too.
    package static let growthSlack: UInt64 = gb
    package static let recoveryInterval: TimeInterval = 20
    /// Growth past the learned start (`pastLearned`): the quiet spell before each step, the largest step, how soon after
    /// a step a cut counts against it, and the calm that forgives it.
    package static let pastLearnedQuiet: TimeInterval = 60, pastLearnedStep: UInt64 = 2 * gb
    package static let pastLearnedRegret: TimeInterval = 600, pastLearnedCalm: TimeInterval = 3_600
    /// The doubling of `pastLearnedQuiet` (0 to 2) after cuts that followed a step past the learned start, and that step.
    package private(set) var pastBackoff = 0
    private var lastPastGrowth: TimeInterval?
    // A global counter cannot attribute a small isolated write to this process. Two
    // distinct increases within ten seconds, or a 64 MiB burst, warrant swap relief.
    private static let swapWindow: TimeInterval = 10
    private static let swapBurst: UInt64 = 64 * 1_024 * 1_024

    package init(
        mode: MemoryBudgetMode, cap: UInt64, floor: UInt64, minimum: UInt64, swapouts: UInt64, hardCeiling: UInt64? = nil
    ) {
        let targetCap = min(cap, hardCeiling ?? cap)
        self.mode = mode
        self.cap = targetCap
        self.hardCeiling = hardCeiling ?? cap
        self.floor = floor
        self.minimum = min(minimum, targetCap)
        observedSwapouts = swapouts
        sampledSwapouts = swapouts
    }

    private static func add(_ a: UInt64, _ b: UInt64) -> UInt64 {
        let (value, overflow) = a.addingReportingOverflow(b)
        return overflow ? .max : value
    }
    private static func subtract(_ a: UInt64, _ b: UInt64) -> UInt64 { a > b ? a - b : 0 }
    /// The held level after one quiet recovery step from `level`: half the gap to `cap`, at least 1 GB, at most `spare`
    /// (free memory above the floor and 1 GB); nil once that reaches the cap.
    package static func raised(_ level: UInt64, cap: UInt64, spare: UInt64) -> UInt64? {
        let step = min(max(subtract(cap, level) / 2, gb), spare)
        return add(level, step) < cap ? level + step : nil
    }
    package static func fit(reserved: UInt64, available: UInt64, floor: UInt64, slack: UInt64 = 0) -> UInt64 {
        subtract(add(reserved, available), add(floor, slack))
    }

    package mutating func step(
        now: TimeInterval, available: UInt64, reserved: UInt64, usable: UInt64,
        swapouts: UInt64, pressure: MemoryPressureLevel, footprint: UInt64 = 0,
        learned: UInt64? = nil
    ) -> Decision? {
        // A reset or unavailable counter must not underflow into a fictitious swap burst.
        if swapouts < sampledSwapouts {
            observedSwapouts = swapouts
            sampledSwapouts = swapouts
            swapWindowStart = nil
            swapWindowBytes = 0
            swapWindowSamples = 0
            pendingSwapRelief = false
        }
        let sampled = Self.subtract(swapouts, sampledSwapouts)
        sampledSwapouts = swapouts
        if let start = swapWindowStart, now - start >= Self.swapWindow {
            swapWindowStart = nil
            swapWindowBytes = 0
            swapWindowSamples = 0
        }
        if sampled > 0 {
            if swapWindowStart == nil { swapWindowStart = now }
            swapWindowBytes = Self.add(swapWindowBytes, sampled)
            swapWindowSamples = min(2, swapWindowSamples + 1)
        }
        let fresh = swapouts > observedSwapouts ? swapouts - observedSwapouts : 0
        let overrun = Self.subtract(footprint, hardCeiling)
        let lowAvailable = mode == .automatic && available < floor
        if fresh > 0,
            mode == .automatic || pressure.isElevated || overrun > 0
                || swapWindowBytes >= Self.swapBurst || swapWindowSamples >= 2
        {
            pendingSwapRelief = true
        }
        let swapRelief = fresh > 0 && pendingSwapRelief
        let distressed = fresh > 0 || pressure != .normal || overrun > 0 || lowAvailable
        if distressed {
            quietSince = nil
            stableSince = nil
        }
        // A pressure reversal shortly after attempted recovery calls for a slower pace;
        // a retry may allocate nothing, so this is not causal attribution of pressure.
        // Back off temporarily, never lower the user's immutable target. Count one
        // reversal per growth, not each callback from the same pressure episode.
        if swapRelief || pressure.isElevated || overrun > 0 || lowAvailable,
            let grew = lastGrowth, now - grew <= 2 * Self.recoveryInterval
        {
            recoveryBackoff = min(2, recoveryBackoff + 1)
            lastGrowth = nil
        }

        optionalReadsPaused = swapRelief || pressure.isElevated || overrun > 0 || lowAvailable || now < settleUntil
        if swapRelief || pressure.isElevated || overrun > 0 || lowAvailable {
            optionalGrowthInhibited = true
            growthQuietSince = nil
        } else if optionalGrowthInhibited {
            if distressed {
                growthQuietSince = nil
            } else {
                let since = growthQuietSince ?? now
                growthQuietSince = since
                let interval = Self.recoveryInterval * Double(1 << recoveryBackoff)
                if now - since >= interval {
                    optionalGrowthInhibited = false
                    growthQuietSince = nil
                }
            }
        }

        var target = usable
        var reason = Reason.available
        if swapRelief {
            // A first signal lowers the budget by the swap plus 1 GB (memory held above that level goes back to macOS);
            // a signal within `swapWindow` of the last cut means pressure persists, and the cut is taken from what is
            // held, as before. Either way the budget drops at least 1 GB per settled signal and growth stops at once.
            let step = Self.add(fresh, Self.gb)
            let from = now - lastShrink <= Self.swapWindow ? min(usable, reserved) : usable
            target = min(target, Self.subtract(from, step))
            reason = .swap
        }
        if pressure != .normal {
            let pressureTarget = Self.subtract(reserved, pressure == .critical ? 3 * Self.gb : Self.gb)
            if pressureTarget <= target {
                target = pressureTarget
                reason = pressure == .critical ? .critical : .warning
            }
        }
        if overrun > 0 {
            // Footprint includes unreserved framework/driver memory. Respect the immutable
            // process ceiling (including its declared transient margin), not a temporary
            // lower usable budget whose old cache slabs may still be retiring.
            let footprintTarget = Self.subtract(reserved, Self.add(overrun, Self.gb / 4))
            if footprintTarget <= target {
                target = footprintTarget
                reason = .footprint
            }
        }
        if lowAvailable {
            let availableTarget = Self.fit(reserved: reserved, available: available, floor: floor, slack: Self.gb / 4)
            if availableTarget < target {
                target = availableTarget
                reason = .available
            }
        }
        target = min(cap, max(target, minimum))
        // Repeated callbacks cannot repeatedly trim the same not-yet-released slabs. A new
        // critical state or hard-ceiling overrun can escalate a weaker trim at once;
        // subsequent urgent trims wait for the previously released slabs to retire.
        let urgent = pressure == .critical || overrun > 0
        let interval: TimeInterval = swapRelief || urgent ? 2 : 8
        let escalated = urgent && !lastShrinkWasUrgent
        let settling = now - lastShrink < interval && !escalated
        if !settling || !swapRelief {
            observedSwapouts = swapouts
            pendingSwapRelief = false
        }
        if target < usable, overrun > 0 || usable - target > Self.gb / 4, !settling {
            if let grew = lastPastGrowth, now - grew <= Self.pastLearnedRegret {
                pastBackoff = min(2, pastBackoff + 1)
                lastPastGrowth = nil
            }
            lastShrink = now
            lastShrinkWasUrgent = urgent
            settleUntil = now + interval
            optionalReadsPaused = true
            quietSince = nil
            held = target
            return Decision(target: target, reason: reason, freshSwap: fresh)
        }
        guard !distressed else { return nil }

        // Automatic: while free memory fits no more than the held level, the other app is still there; no probe.
        if mode == .automatic, let level = held, Self.fit(reserved: reserved, available: available, floor: floor, slack: Self.gb) <= level {
            stableSince = now
        }
        let stable = stableSince ?? now
        stableSince = stable
        if recoveryBackoff > 0 {
            if now - stable >= 6 * Self.recoveryInterval {
                recoveryBackoff -= 1
                stableSince = now
            }
        } else if now - stable >= 2 * Self.recoveryInterval, let level = held, usable >= level,
            available >= Self.add(floor, 2 * Self.gb)
        {
            // Quiet this long with memory to spare: raise the held level by half its gap to the cap (at least 1 GB), but
            // never past what free memory holds with 1 GB above the floor, whether or not the caches have filled it (they
            // fill slowly, and a short squeeze by another app must not cost many minutes). A reversal after the growth
            // raises `recoveryBackoff`, which must decay (120 s a step) before the next probe.
            held = Self.raised(level, cap: cap, spare: Self.subtract(available, Self.add(floor, Self.gb)))
            stableSince = now
        }

        let grown =
            mode == .explicit
            ? min(cap, held ?? cap)
            : min(
                cap, held ?? .max, learned ?? .max,
                Self.fit(reserved: reserved, available: available, floor: floor, slack: Self.growthSlack))
        // Cache allocation may stop on a transient pressure event before the governor
        // lowers its allowance. Retry that unfinished growth even if the allowance is
        // already at target; otherwise no later budget increase would wake the owner.
        let retriesGrowth = mode == .explicit && usable >= grown
        // At or past the learned start, with nothing left below it to regain: grow past it, carefully (`pastLearned`).
        if mode == .automatic, let learned, !(grown > usable && grown - usable > Self.gb / 2), Self.add(usable, Self.gb / 4) >= learned {
            return pastLearned(
                now: now, usable: usable, free: Self.fit(reserved: reserved, available: available, floor: floor, slack: Self.gb))
        }
        // Explicit targets include the final partial step; automatic retains its existing
        // hysteresis so harmless fluctuations in estimated headroom do not resize caches.
        guard retriesGrowth || (grown > usable && (mode == .explicit || grown - usable > Self.gb / 2)) else {
            quietSince = nil
            return nil
        }
        let since = quietSince ?? now
        quietSince = since
        let factor = UInt64(1 << recoveryBackoff)
        let recoveryWait = Self.recoveryInterval * Double(factor)
        guard now - since >= recoveryWait else { return nil }
        if retriesGrowth {
            quietSince = now
            lastGrowth = now
            return Decision(target: usable, reason: .growthRetry, freshSwap: 0)
        }
        // Each quiet interval closes half the gap to the cap (at least 1 GB; slower after a reversal), as the held level does.
        let recovered = mode == .explicit ? min(grown, Self.add(usable, max(Self.gb, Self.subtract(cap, usable) / 2) / factor)) : grown
        quietSince = mode == .explicit ? now : nil
        lastGrowth = now
        return Decision(target: recovered, reason: .recovery, freshSwap: 0)
    }

    /// Automatic mode past the learned start. The learned budget is the worst tenth of two weeks of this Mac's memory
    /// use, not what is free now: it sets where the budget starts, never where it must stay. When memory is really free
    /// (the step keeps the floor and 1 GB free, no pressure, no fresh swap: `step` returns before this otherwise) for
    /// `pastLearnedQuiet` seconds, the budget grows by at most `pastLearnedStep` or half its gap to the cap, whichever
    /// is smaller; a cut within `pastLearnedRegret` seconds of such a step doubles the next wait (up to four times), and
    /// `pastLearnedCalm` seconds without a cut restore it. Cuts are taken and recorded as before, so the learned start
    /// stays honest.
    private mutating func pastLearned(now: TimeInterval, usable: UInt64, free: UInt64) -> Decision? {
        if pastBackoff > 0, now - lastShrink >= Self.pastLearnedCalm { pastBackoff = 0 }
        let gap = Self.subtract(cap, usable)
        let least = min(gap, Self.gb / 2)
        let step = min(Self.pastLearnedStep, max(gap / 2, least), Self.subtract(free, usable))
        guard least > 0, step >= least else {
            quietSince = nil
            return nil
        }
        let since = quietSince ?? now
        quietSince = since
        guard now - since >= Self.pastLearnedQuiet * Double(1 << pastBackoff) else { return nil }
        quietSince = nil
        lastPastGrowth = now
        return Decision(target: usable + step, reason: .recovery, freshSwap: 0)
    }
}
