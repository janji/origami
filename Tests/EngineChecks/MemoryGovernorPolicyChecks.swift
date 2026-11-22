import Foundation
import OrigamiEngine

private func memoryPolicyRequire(_ value: Bool, _ message: String) throws {
    if !value { throw EngineError(message) }
}

/// Pure host snapshots and monotonic time: no model, GPU, sysctl, timers, or waiting.
func memoryExplicitTargetPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    let cap: UInt64 = 25_770_000_000
    var policy = MemoryGovernorPolicy(mode: .explicit, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    for time in [0.0, 2, 19.99, 20, 20.25, 40] {
        let decision = policy.step(
            now: time, available: 0, reserved: 24 * gb, usable: cap,
            swapouts: 0, pressure: .normal, learned: 10 * gb)
        if time == 20 || time == 40 {
            try memoryPolicyRequire(
                decision?.target == cap && decision?.reason == .growthRetry,
                "An allocation interrupted below its target had no bounded growth retry")
        } else {
            try memoryPolicyRequire(decision == nil, "Low free estimates reduced the target or repeated callbacks grew too often")
        }
    }
    // Recovery ignores conservative free headroom and a lower learned allowance, but
    // waits for quiet and takes bounded steps, including the last sub-500 MB remainder.
    policy = MemoryGovernorPolicy(mode: .explicit, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    var usable = 24 * gb
    try memoryPolicyRequire(
        policy.step(
            now: 110, available: 0, reserved: usable, usable: usable,
            swapouts: 0, pressure: .normal, learned: 10 * gb) == nil, "Recovery skipped quiet interval")
    try memoryPolicyRequire(
        policy.step(
            now: 129.9, available: 0, reserved: usable, usable: usable,
            swapouts: 0, pressure: .normal) == nil, "Recovery happened before 20 s")
    let first = policy.step(now: 130, available: 0, reserved: usable, usable: usable, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(first?.target == 25 * gb && first?.reason == .recovery, "Explicit recovery did not take a 1 GB step")
    usable = first!.target
    try memoryPolicyRequire(
        policy.step(
            now: 130.1, available: 0, reserved: usable, usable: usable,
            swapouts: 0, pressure: .normal) == nil, "Repeated callbacks grew without another quiet interval")
    let last = policy.step(now: 150, available: 0, reserved: usable, usable: usable, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(last?.target == cap, "Explicit recovery did not reach the exact target")
    let retry = policy.step(now: 200, available: 0, reserved: cap, usable: cap, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(retry?.target == cap && retry?.reason == .growthRetry, "Low free estimate oscillated the recovered target")
    var remainder = MemoryGovernorPolicy(mode: .explicit, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    _ = remainder.step(now: 0, available: 0, reserved: 25_500_000_000, usable: 25_500_000_000, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(
        remainder.step(
            now: 20, available: 0, reserved: 25_500_000_000, usable: 25_500_000_000,
            swapouts: 0, pressure: .normal)?.target == cap, "Final small remainder was stranded by adaptive hysteresis")
}

func memoryPressureRecoveryPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    var policy = MemoryGovernorPolicy(mode: .explicit, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    let warning = policy.step(now: 0, available: 10 * gb, reserved: 25 * gb, usable: 26 * gb, swapouts: 0, pressure: .warning)
    try memoryPolicyRequire(
        warning?.target == 24 * gb && warning?.reason == .warning, "Real warning was ignored despite apparently free memory")
    for time in [0.25, 2, 7.99] {
        try memoryPolicyRequire(
            policy.step(
                now: time, available: 10 * gb, reserved: 24 * gb, usable: 24 * gb,
                swapouts: 0, pressure: .warning) == nil, "Repeated warning callback escaped settling")
    }
    try memoryPolicyRequire(
        policy.step(
            now: 8, available: 10 * gb, reserved: 24 * gb, usable: 24 * gb,
            swapouts: 0, pressure: .warning)?.target == 23 * gb, "Persistent warning did not release more memory")
    let critical = policy.step(now: 8.1, available: 10 * gb, reserved: 23 * gb, usable: 23 * gb, swapouts: 0, pressure: .critical)
    try memoryPolicyRequire(
        critical?.target == 20 * gb && critical?.reason == .critical, "Critical escalation waited behind a weaker warning")
    try memoryPolicyRequire(
        policy.step(
            now: 8.2, available: 10 * gb, reserved: 20 * gb, usable: 20 * gb,
            swapouts: 0, pressure: .critical) == nil, "Critical callbacks repeatedly trimmed unreleased slabs")
    try memoryPolicyRequire(
        policy.step(
            now: 10.2, available: 10 * gb, reserved: 20 * gb, usable: 20 * gb,
            swapouts: 0, pressure: .critical)?.target == 17 * gb, "Persistent critical pressure was ignored")
    try memoryPolicyRequire(
        policy.step(
            now: 100, available: 20 * gb, reserved: 8 * gb, usable: 8 * gb,
            swapouts: 0, pressure: .warning) == nil, "Active pressure regrew or crossed compulsory owners")
    _ = policy.step(now: 101, available: 0, reserved: 8 * gb, usable: 8 * gb, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(
        policy.step(
            now: 120.9, available: 0, reserved: 8 * gb, usable: 8 * gb,
            swapouts: 0, pressure: .normal) == nil, "Normal signal recovered before quiet interval")
    try memoryPolicyRequire(
        policy.step(
            now: 121, available: 0, reserved: 8 * gb, usable: 8 * gb,
            swapouts: 0, pressure: .normal)?.target == 17 * gb, "Normal did not recover half the gap to the cap (to the level that held)")
    try memoryPolicyRequire(
        policy.step(
            now: 121.1, available: 10 * gb, reserved: 9 * gb, usable: 9 * gb,
            swapouts: 0, pressure: .warning)?.target == 8 * gb, "New pressure failed to reverse recovery")
    _ = policy.step(now: 122, available: 0, reserved: 8 * gb, usable: 8 * gb, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(
        policy.step(
            now: 141, available: 0, reserved: 8 * gb, usable: 8 * gb,
            swapouts: 0, pressure: .normal) == nil, "Pressure did not reset the recovery interval")
    try memoryPolicyRequire(
        policy.step(
            now: 162, available: 0, reserved: 8 * gb, usable: 8 * gb,
            swapouts: 0, pressure: .normal
        ).map { $0.reason == .growthRetry && $0.target == 8 * gb } == true,
        "After a pressure reversal recovery passed the level that held, or lost its slower pace")
    try memoryPolicyRequire(
        MemoryPressureLevel(rawValue: 1) == .normal && !MemoryPressureLevel.normal.isElevated
            && MemoryPressureLevel(rawValue: 2)?.isElevated == true && MemoryPressureLevel(rawValue: 4)?.isElevated == true
            && MemoryPressureLevel(rawValue: 0) == nil && MemoryPressureLevel(rawValue: 7) == nil,
        "Kernel pressure levels confused unknown values or normal with distress")
}

func memorySwapRecoveryPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    let mb: UInt64 = 1_000_000
    // Seeded before loading, rather than after: a load-time swap burst is fresh even
    // when pressure has already cleared before the governor's first observation.
    var startup = MemoryGovernorPolicy(mode: .explicit, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 100 * gb)
    let loaded = startup.step(
        now: 10, available: 10 * gb, reserved: 25 * gb, usable: 26 * gb,
        swapouts: 101 * gb, pressure: .normal)
    try memoryPolicyRequire(
        loaded?.target == 24 * gb && loaded?.freshSwap == gb && loaded?.reason == .swap,
        "Swap during loading was lost before governor construction")
    var policy = MemoryGovernorPolicy(mode: .explicit, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 100 * gb)
    try memoryPolicyRequire(
        policy.step(
            now: 0, available: 0, reserved: 25 * gb, usable: 26 * gb,
            swapouts: 100 * gb, pressure: .normal) == nil, "Historical swap was mistaken for fresh swapping")
    let first = policy.step(
        now: 1, available: 10 * gb, reserved: 25 * gb, usable: 26 * gb,
        swapouts: 100 * gb + 100 * mb, pressure: .normal)
    try memoryPolicyRequire(
        first?.target == 24_900 * mb && first?.freshSwap == 100 * mb && first?.reason == .swap,
        "Fresh swap did not temporarily lower explicit budget")
    try memoryPolicyRequire(
        policy.step(
            now: 1.25, available: 10 * gb, reserved: 24_900 * mb, usable: 24_900 * mb,
            swapouts: 100 * gb + 200 * mb, pressure: .normal) == nil, "Swap callbacks escaped settling")
    let second = policy.step(
        now: 3.1, available: 10 * gb, reserved: 24_900 * mb, usable: 24_900 * mb,
        swapouts: 100 * gb + 300 * mb, pressure: .normal)
    try memoryPolicyRequire(
        second?.target == 23_700 * mb && second?.freshSwap == 200 * mb,
        "Swap during settling was lost or counted twice")
    _ = policy.step(
        now: 4, available: 0, reserved: 23_700 * mb, usable: 23_700 * mb,
        swapouts: 100 * gb + 300 * mb, pressure: .normal)
    try memoryPolicyRequire(
        policy.step(
            now: 24, available: 0, reserved: 23_700 * mb, usable: 23_700 * mb,
            swapouts: 100 * gb + 300 * mb, pressure: .normal
        ).map { $0.reason == .growthRetry && $0.target == 23_700 * mb } == true,
        "Old swap prevented recovery to the level that held, or recovery passed it")
    try memoryPolicyRequire(
        policy.step(
            now: 25, available: 0, reserved: 23_700 * mb, usable: 23_700 * mb,
            swapouts: 0, pressure: .normal) == nil, "Counter reset underflowed into a fresh swap burst")
    try memoryPolicyRequire(
        policy.step(
            now: 30, available: .max, reserved: 23_700 * mb, usable: 23_700 * mb,
            swapouts: .max, pressure: .normal)?.target == 8 * gb, "Huge swap wrapped or crossed compulsory memory floor")

    // The cut follows the signal: 0.35 GB swapped at a 24 GB budget with 16 GB held lowers the budget by the swap plus
    // 1 GB, to 22.65 GB (before: to what is held less that, 14.65 GB).
    func squeezed() -> MemoryGovernorPolicy {
        var policy = MemoryGovernorPolicy(mode: .explicit, cap: 24 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
        _ = policy.step(now: 0, available: 6 * gb, reserved: 16 * gb, usable: 24 * gb, swapouts: 0, pressure: .normal)
        return policy
    }
    var small = squeezed()
    let proportionate = small.step(now: 1, available: 6 * gb, reserved: 16 * gb, usable: 24 * gb, swapouts: 350 * mb, pressure: .normal)
    try memoryPolicyRequire(
        proportionate?.target == 22_650 * mb && proportionate?.reason == .swap && small.optionalGrowthInhibited,
        "A 0.35 GB swap did not cut 1.35 GB off the budget and stop growth: \(String(describing: proportionate))")
    // Swapping that persists (0.35 GB a second): every later cut within 10 s is taken from what is held, as before, so the
    // budget reaches the 8 GB floor at 11 s, one 2 s settling interval after the old rule (9 s); each returns 1.7 GB or more.
    var persistent = squeezed()
    var (usable, reserved, swapped) = (24 * gb, 16 * gb, UInt64(0))
    var floorAt: TimeInterval?
    var cuts: [UInt64] = []
    for time in stride(from: 1.0, through: 30, by: 1) {
        swapped += 350 * mb
        if let cut = persistent.step(
            now: time, available: 6 * gb, reserved: reserved, usable: usable, swapouts: swapped, pressure: .normal)
        {
            cuts.append(min(usable, reserved) - min(cut.target, reserved))
            usable = cut.target
            reserved = min(reserved, usable)
        }
        if usable == 8 * gb, floorAt == nil { floorAt = time }
    }
    try memoryPolicyRequire(
        floorAt == 11 && cuts.first == 0 && cuts.dropFirst().dropLast().allSatisfy { $0 >= 1_700 * mb },
        "Persistent swap did not walk the budget to its floor in time: floor at \(String(describing: floorAt)) s, freed \(cuts)")
}

func memoryAutomaticAndLedgerPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    var automatic = MemoryGovernorPolicy(mode: .automatic, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    let low = automatic.step(now: 0, available: 3 * gb, reserved: 24 * gb, usable: 26 * gb, swapouts: 0, pressure: .normal)
    try memoryPolicyRequire(low?.target == 22_750_000_000 && low?.reason == .available, "Automatic lost its headroom policy")
    var learned = MemoryGovernorPolicy(mode: .automatic, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    _ = learned.step(now: 0, available: 10 * gb, reserved: 18 * gb, usable: 20 * gb, swapouts: 0, pressure: .normal, learned: 22 * gb)
    try memoryPolicyRequire(
        learned.step(
            now: 20, available: 10 * gb, reserved: 18 * gb, usable: 20 * gb,
            swapouts: 0, pressure: .normal, learned: 22 * gb)?.target == 22 * gb,
        "Automatic recovery ignored learned allowance")
    try memoryPolicyRequire(
        learned.step(
            now: 40, available: 5 * gb, reserved: 22 * gb, usable: 22 * gb,
            swapouts: 0, pressure: .normal) == nil, "Automatic recovery spent its free-memory cushion")
    try memoryPolicyRequire(
        MemoryGovernorPolicy.fit(reserved: .max, available: 1, floor: 4, slack: 1) == .max - 5
            && MemoryGovernorPolicy.fit(reserved: 1, available: 1, floor: 3) == 0, "Fit arithmetic overflowed or underflowed")

    for mode in [MemoryBudgetMode.automatic, .explicit] {
        let ledger = try MemoryLedger(ceiling: 100, transientMargin: 10, availableFloor: 20, budgetMode: mode)
        try memoryPolicyRequire(ledger.usableBytes == 90 && ledger.maximumUsableBytes == 90, "Mode changed the hard ceiling or margin")
        try memoryPolicyRequire(
            ledger.needsPressureCheck(available: 0, footprint: 90) == (mode == .automatic), "Ledger used free estimates in explicit mode")
        try memoryPolicyRequire(
            !ledger.needsPressureCheck(available: 100, footprint: 91), "Declared transient margin was treated as a hard overrun")
        try memoryPolicyRequire(
            ledger.needsPressureCheck(available: 100, footprint: 101), "Hard footprint overrun stopped reaching the governor")
        let token = try ledger.reserve(bytes: 90, category: "policy-fixture")
        do {
            _ = try ledger.reserve(bytes: 1, category: "overflow")
            throw EngineError("Hard reservation ceiling was bypassed")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try ledger.release(token)
        ledger.setUsable(.max)
        try memoryPolicyRequire(ledger.usableBytes == 90, "Regrowth crossed hard ceiling")
    }
    let zero = try MemoryLedger(ceiling: 100, transientMargin: 10, availableFloor: 0, budgetMode: .explicit)
    try memoryPolicyRequire(!zero.usesAvailableMemoryFloor && zero.usableBytes == 90, "Explicit zero advisory floor was refused")
    do {
        _ = try MemoryLedger(ceiling: 100, transientMargin: 10, availableFloor: 0)
        throw EngineError("Automatic zero floor was accepted", code: "fixture_failed")
    } catch let error as EngineError where error.code != "fixture_failed" {}
}

func memoryHardFootprintPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    let mb: UInt64 = 1_000_000
    var small = MemoryGovernorPolicy(
        mode: .explicit, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb,
        swapouts: 0, hardCeiling: 27 * gb)
    try memoryPolicyRequire(
        small.step(
            now: 0, available: 10 * gb, reserved: 26 * gb, usable: 25_900 * mb,
            swapouts: 0, pressure: .normal, footprint: 27_010 * mb)?.target == 25_740 * mb,
        "Adaptive hysteresis ignored a small but real hard-ceiling violation")
    var policy = MemoryGovernorPolicy(
        mode: .explicit, cap: 26 * gb, floor: 4 * gb, minimum: 8 * gb,
        swapouts: 0, hardCeiling: 27 * gb)
    // A temporarily reduced allowance is not a hard physical ceiling. Pending slab
    // retirement and legitimate declared transient memory cannot cause repeated cuts.
    _ = policy.step(now: 0, available: 0, reserved: 24 * gb, usable: 20 * gb, swapouts: 0, pressure: .normal, footprint: 26_500 * mb)
    let recovery = policy.step(
        now: 20, available: 0, reserved: 24 * gb, usable: 20 * gb,
        swapouts: 0, pressure: .normal, footprint: 26_500 * mb)
    try memoryPolicyRequire(
        recovery?.target == 23 * gb && recovery?.reason == .recovery,
        "Footprint above temporary usable allowance blocked explicit recovery")
    let over = policy.step(
        now: 21, available: 10 * gb, reserved: 21 * gb, usable: 21 * gb,
        swapouts: 0, pressure: .normal, footprint: 28 * gb)
    try memoryPolicyRequire(
        over?.target == 19_750 * mb && over?.reason == .footprint,
        "Hard physical overrun did not release overage plus bounded cushion")
    try memoryPolicyRequire(
        policy.step(
            now: 21.25, available: 10 * gb, reserved: 19_750 * mb, usable: 19_750 * mb,
            swapouts: 0, pressure: .normal, footprint: 28 * gb) == nil,
        "Physical overrun callbacks repeatedly trimmed unreleased slabs")
    try memoryPolicyRequire(
        policy.step(
            now: 23, available: 10 * gb, reserved: 19_750 * mb, usable: 19_750 * mb,
            swapouts: 0, pressure: .normal, footprint: 28 * gb)?.target == 18_500 * mb,
        "Persistent hard overrun stopped receiving relief")
    _ = policy.step(
        now: 24, available: 0, reserved: 18_500 * mb, usable: 18_500 * mb,
        swapouts: 0, pressure: .normal, footprint: 26_500 * mb)
    try memoryPolicyRequire(
        policy.step(
            now: 64, available: 0, reserved: 18_500 * mb, usable: 18_500 * mb,
            swapouts: 0, pressure: .normal, footprint: 26_500 * mb
        ).map { $0.reason == .growthRetry && $0.target == 18_500 * mb } == true,
        "Cleared hard overrun did not resume recovery to the level that held")
    let floor = policy.step(
        now: 70, available: .max, reserved: 18_500 * mb, usable: 18_500 * mb,
        swapouts: 0, pressure: .normal, footprint: .max)
    try memoryPolicyRequire(floor?.target == 8 * gb, "Unreserved memory overrun crossed compulsory owner floor or overflowed")
    for time in [71.0, 100, 200] {
        try memoryPolicyRequire(
            policy.step(
                now: time, available: .max, reserved: 8 * gb, usable: 8 * gb,
                swapouts: 0, pressure: .normal, footprint: 28 * gb) == nil,
            "Recovery grew despite a persistent hard physical overrun")
    }
    let ledger = try MemoryLedger(ceiling: 100, transientMargin: 10, availableFloor: 20, budgetMode: .explicit)
    ledger.setUsable(50)
    try memoryPolicyRequire(
        !ledger.needsPressureCheck(available: 0, footprint: 90)
            && ledger.needsPressureCheck(available: 0, footprint: 101), "Ledger compared physical footprint with a temporary soft allowance"
    )
}

/// An explicit limit is the ceiling: after a swap-forced cut, regrowth stops at the level that held, probes half way
/// to the limit after 40 s quiet with free memory to spare (floor + 2 GB), and backs off again on fresh swap.
/// Without a cut it still reaches the full limit; automatic mode holds and probes the same way; history seeds the start.
func memoryExplicitHeldLevelPolicy() throws {
    let gb: UInt64 = 1_000_000_000
    let cap = 30 * gb
    struct Host {
        var policy: MemoryGovernorPolicy
        var usable: UInt64
        var swap: UInt64 = 0
        mutating func step(
            _ now: TimeInterval, available: UInt64 = 0, reserved: UInt64? = nil, pressure: MemoryPressureLevel = .normal
        ) -> MemoryGovernorPolicy.Decision? {
            let decision = policy.step(
                now: now, available: available, reserved: reserved ?? usable, usable: usable, swapouts: swap, pressure: pressure)
            if let decision { usable = decision.target }
            return decision
        }
        /// Quiet steps every second from `from` to `until`; the time of the first growth above `level`, if any.
        mutating func quiet(
            from: TimeInterval, until: TimeInterval, above level: UInt64, reserved: UInt64? = nil, available: UInt64 = 0
        ) -> TimeInterval? {
            var time = from
            while time <= until {
                _ = step(time, available: available, reserved: reserved)
                if usable > level { return time }
                time += 1
            }
            return nil
        }
    }
    func explicit(usable: UInt64) -> Host {
        Host(policy: MemoryGovernorPolicy(mode: .explicit, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0), usable: usable)
    }

    // (1) A swap burst cuts; quiet without free memory to spare (under floor + 2 GB) regrows only to the level that held.
    var host = explicit(usable: cap)
    _ = host.step(0)
    host.swap = 2 * gb
    let cut = host.step(1)
    try memoryPolicyRequire(
        cut?.reason == .swap && cut?.target == 27 * gb, "Swap did not cut the explicit budget: \(String(describing: cut))")
    try memoryPolicyRequire(
        host.quiet(from: 2, until: 600, above: 27 * gb, available: 5_900_000_000) == nil,
        "Explicit recovery grew past the level that held without free memory to spare")
    // (2) Quiet for 40 s with free memory to spare: one step up at once, whether or not the caches filled the level.
    let probe = host.quiet(from: 601, until: 640, above: 27 * gb, reserved: 20 * gb, available: 10 * gb)
    // Half the 3 GB gap to the 30 GB limit: 28.5 GB.
    try memoryPolicyRequire(probe != nil && host.usable == 28_500_000_000, "The held level never probed upward: \(host.usable)")
    // Fresh swap right after that growth cuts again and backs off: the next probe waits for the back-off to decay.
    host.swap += 2 * gb
    let reversal = probe! + 5
    try memoryPolicyRequire(host.step(reversal)?.target == 25_500_000_000, "Fresh swap after a probe was not cut")
    try memoryPolicyRequire(
        host.quiet(from: reversal + 1, until: reversal + 150, above: 25_500_000_000, available: 10 * gb) == nil,
        "A reversal did not slow the next probe")
    let later = host.quiet(from: reversal + 151, until: reversal + 200, above: 25_500_000_000, available: 10 * gb)
    try memoryPolicyRequire(later != nil && host.usable == 27_750_000_000, "Probing stopped for good after a reversal: \(host.usable)")
    // Probes continue while memory stays free, up to the exact limit and never past it.
    _ = host.quiet(from: later! + 1, until: later! + 2_000, above: cap, available: 10 * gb)
    try memoryPolicyRequire(host.usable == cap, "Probing never returned to the limit: \(host.usable)")

    // (3) Without any cut, explicit mode still reaches the full limit at its normal pace.
    var plain = explicit(usable: 25 * gb)
    _ = plain.quiet(from: 0, until: 120, above: cap - 1)
    try memoryPolicyRequire(plain.usable == cap, "Explicit mode without a cut did not reach its limit: \(plain.usable)")

    // (4) Automatic mode holds too. Another app takes memory: the budget drops to what free memory fits, and stays
    // there while the app runs (no bounce on brief quiet). Once the app has gone for good it comes back one step per
    // 40 s quiet, slower again after a reversal, and never above the cap.
    var automatic = Host(
        policy: MemoryGovernorPolicy(mode: .automatic, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0), usable: 24 * gb)
    _ = automatic.step(0, available: 10 * gb)
    let squeezed = automatic.step(1, available: 2 * gb)
    try memoryPolicyRequire(
        squeezed?.reason == .available && squeezed?.target == 21_750_000_000,
        "Automatic did not drop for another app: \(String(describing: squeezed))")
    // The app keeps its memory: what the cut freed is free, nothing more.
    for time in stride(from: 2.0, through: 300, by: 1) { _ = automatic.step(time, available: 4_250_000_000) }
    try memoryPolicyRequire(
        automatic.usable == 21_750_000_000, "Automatic bounced while the other app held its memory: \(automatic.usable)")
    // The app quits: plenty is free, but regrowth waits 40 s quiet at the held level, then takes one step (20 s later).
    var time = 301.0
    while time < 340, automatic.usable == 21_750_000_000 {
        _ = automatic.step(time, available: 12 * gb)
        time += 1
    }
    try memoryPolicyRequire(automatic.usable == 21_750_000_000, "Automatic regrew at once after the pressure went: \(automatic.usable)")
    while time < 380, automatic.usable == 21_750_000_000 {
        _ = automatic.step(time, available: 12 * gb)
        time += 1
    }
    // Half the 8.25 GB gap to the cap: 25.875 GB (free memory, 12 GB, holds that with 1 GB above the 4 GB floor).
    try memoryPolicyRequire(automatic.usable == 25_875_000_000, "Automatic did not probe one step up: \(automatic.usable)")
    // A warning right after that probe cuts and slows the next one; quiet probes then climb to the cap and stop there.
    let backAgain = time + 5
    try memoryPolicyRequire(
        automatic.step(backAgain, available: 12 * gb, pressure: .warning)?.target == 24_875_000_000, "A warning after a probe was not cut")
    try memoryPolicyRequire(
        automatic.quiet(from: backAgain + 1, until: backAgain + 150, above: 24_875_000_000, available: 12 * gb) == nil,
        "A warning after a probe did not slow the next probe")
    _ = automatic.quiet(from: backAgain + 151, until: backAgain + 5_000, above: cap - 1, available: 40 * gb)
    // Automatic keeps its 0.5 GB hysteresis: within it of the cap is back.
    try memoryPolicyRequire(automatic.usable + gb / 2 >= cap, "Automatic never came back to the cap: \(automatic.usable)")
    for time in stride(from: backAgain + 5_001, through: backAgain + 6_000, by: 1) { _ = automatic.step(time, available: 40 * gb) }
    try memoryPolicyRequire(automatic.usable <= cap, "Automatic passed its cap: \(automatic.usable)")

    // (5) History never lowers a limit's start; it only sets the level its advice suggests (`limitAdvice`).
    let now = Date()
    func cuts(under limit: UInt64?) -> [MemoryHistory.Trim] {
        [21, 22, 23, 24, 31].map { MemoryHistory.Trim(t: now.timeIntervalSince1970 - Double($0) * 60, to: UInt64($0) * gb, limit: limit) }
    }
    let trims = cuts(under: 29 * gb)
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: trims, cap: 29 * gb, minimum: 8 * gb, explicit: true, now: now) == 21 * gb,
        "The held level was not the 20th percentile of cuts below the limit")
    // Cuts are evidence about the ceiling they happened under: one at or below this limit lowers it, a higher one or an
    // unknown one (files written before the ceiling was recorded) does not; unknown ones still teach automatic mode.
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: cuts(under: 26 * gb), cap: 29 * gb, minimum: 8 * gb, explicit: true, now: now) == 21 * gb,
        "Cuts under a lower limit did not lower the held level")
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: cuts(under: 31 * gb), cap: 29 * gb, minimum: 8 * gb, explicit: true, now: now) == nil,
        "Cuts under a higher limit lowered the held level")
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: cuts(under: nil), cap: 29 * gb, minimum: 8 * gb, explicit: true, now: now) == nil,
        "Cuts of unknown limit lowered the held level")
    let hour = (0..<6).map { MemoryHistory.Bucket(t: now.timeIntervalSince1970 - Double($0 + 1) * 600, fit: 28 * gb) }
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: hour, trims: cuts(under: nil), cap: 29 * gb, minimum: 8 * gb, now: now) == 22 * gb
            && MemoryHistory.budget(buckets: hour, trims: cuts(under: 31 * gb), cap: 29 * gb, minimum: 8 * gb, now: now) == 28 * gb,
        "Automatic mode: cuts of unknown limit no longer count, or a higher ceiling's do")
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: Array(trims.suffix(3)), cap: 29 * gb, minimum: 8 * gb, explicit: true, now: now) == nil,
        "Fewer than three cuts below the limit made a held level")
    try memoryPolicyRequire(
        MemoryHistory.budget(buckets: [], trims: trims, cap: 29 * gb, minimum: 8 * gb, now: now) == nil,
        "Automatic learning no longer needs an hour of free-memory history")
    // Cuts are written to the file at once; the governor still starts at the limit, and logs no history start.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-held-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let history = MemoryHistory(file: folder.appendingPathComponent("memory-history.json"), identity: "fixture")
    for (index, level) in [21, 22, 23].enumerated() {
        history.trimmed(to: UInt64(level) * gb, limit: 29 * gb, now: now.addingTimeInterval(Double(index - 3)))
    }
    let reread = MemoryHistory(file: history.file, identity: "fixture")
    try memoryPolicyRequire(reread.budget(cap: 29 * gb, minimum: 8 * gb, explicit: true) == 21 * gb, "Cuts were not written at once")
    // A file written before ceilings were recorded still loads.
    let old = folder.appendingPathComponent("old.json")
    try Data(#"{"identity":"fixture","buckets":[],"trims":[{"t":1,"to":2}]}"#.utf8).write(to: old)
    try memoryPolicyRequire(
        MemoryHistory(file: old, identity: "fixture").summary(cap: 29 * gb, minimum: 0, applied: false)["trims"] as? Int == 1,
        "A history file without ceilings did not load")
    let ledger = try MemoryLedger(ceiling: 30 * gb, transientMargin: gb, availableFloor: 4 * gb, budgetMode: .explicit)
    let lines = HeldCheckLines()
    let governor = MemoryGovernor(
        ledger: ledger, memory: OrigamiMemory(cap: 29 * gb, start: 29 * gb, floor: 4 * gb, minimum: 8 * gb),
        trim: { 0 }, log: { lines.append($0) }, history: reread, initialSwapouts: 0)
    governor.start()
    try memoryPolicyRequire(
        ledger.usableBytes == 29 * gb && !lines.all.contains { $0.contains("starting at") },
        "A limit cut before started lower: \(ledger.usableBytes) \(lines.all)")
    // Cuts on three days speak in the advice, and the limit still starts at itself.
    let days = MemoryHistory(file: folder.appendingPathComponent("days.json"), identity: "fixture")
    for (day, level) in [21, 22, 23].enumerated() {
        days.trimmed(to: UInt64(level) * gb, limit: 29 * gb, now: now.addingTimeInterval(-Double(day) * 2 * 86_400))
    }
    let daysLedger = try MemoryLedger(ceiling: 30 * gb, transientMargin: gb, availableFloor: 4 * gb, budgetMode: .explicit)
    let persistent = MemoryGovernor(
        ledger: daysLedger, memory: OrigamiMemory(cap: 29 * gb, start: 29 * gb, floor: 4 * gb, minimum: 8 * gb),
        trim: { 0 }, log: { _ in }, history: days, initialSwapouts: 0)
    persistent.start()
    try memoryPolicyRequire(
        daysLedger.usableBytes == 29 * gb && persistent.limitAdvice != nil,
        "Cuts on three days: start \(daysLedger.usableBytes), advice \(String(describing: persistent.limitAdvice))")
    // Cuts are per model and context: model a's mark neither b's measured limit stale nor a's at another context;
    // cuts from before a scope was kept count only in automatic mode.
    let scoped = folder.appendingPathComponent("scoped.json")
    let a = MemoryHistory(file: scoped, identity: "fixture", scope: .init(model: "a", context: 8192))
    for day in 0..<3 { a.trimmed(to: 21 * gb, limit: 29 * gb, now: now.addingTimeInterval(-Double(day) * 2 * 86_400)) }
    let since = now.addingTimeInterval(-10 * 86_400)
    let b = MemoryHistory(file: scoped, identity: "fixture", scope: .init(model: "b", context: 8192))
    let a32 = MemoryHistory(file: scoped, identity: "fixture", scope: .init(model: "a", context: 32768))
    try memoryPolicyRequire(
        a.cutDays(below: 29 * gb, since: since) == 3 && b.cutDays(below: 29 * gb, since: since) == 0
            && a32.cutDays(below: 29 * gb, since: since) == 0 && b.limitAdvice(limit: 29 * gb, cap: 29 * gb, minimum: 0) == nil,
        "One model's cuts counted for another: \(b.cutDays(below: 29 * gb, since: since))")
    let unkeyed = MemoryHistory.Trim(t: now.timeIntervalSince1970, to: 21 * gb, limit: 29 * gb)
    let scope = MemoryHistory.Scope(model: "b", context: 8192)
    try memoryPolicyRequire(
        unkeyed.counts(against: 29 * gb, explicit: false, scope: scope) && !unkeyed.counts(against: 29 * gb, explicit: true, scope: scope),
        "A cut of unknown model counts outside automatic mode")
}

/// Time below target counts against `target`: automatic mode at a learned budget below the cap, with no cut, has no
/// deficit; after a cut it grows by (target - budget) per second.
func memoryDeficitAgainstLearnedTarget() throws {
    let gb: UInt64 = 1_000_000_000
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-deficit-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let history = MemoryHistory(file: folder.appendingPathComponent("memory-history.json"), identity: "fixture")
    let now = Date()
    for bucket in 0...6 { history.observe(fit: 22 * gb, now: now.addingTimeInterval(-Double(6 - bucket) * MemoryHistory.bucketSeconds)) }
    let ledger = try MemoryLedger(ceiling: 30 * gb, transientMargin: gb, availableFloor: 4 * gb, budgetMode: .automatic)
    let governor = MemoryGovernor(
        ledger: ledger, memory: OrigamiMemory(cap: 29 * gb, start: 29 * gb, floor: 4 * gb, minimum: 8 * gb),
        trim: { 0 }, log: { _ in }, history: history, learns: true, initialSwapouts: 0)
    try memoryPolicyRequire(
        governor.learned == 22 * gb && governor.target == 29 * gb,
        "No learned start below the cap, or the cap is not the target: \(governor.target)")
    let uptime = ProcessInfo.processInfo.systemUptime
    ledger.setUsable(22 * gb)
    governor.markDeficit(now: uptime - 20)
    try memoryPolicyRequire(
        governor.budgetDeficitGBSeconds() == 0, "At its learned target the budget counted a deficit: \(governor.budgetDeficitGBSeconds())")
    ledger.setUsable(20 * gb)
    governor.markDeficit(now: uptime - 10)
    let deficit = governor.budgetDeficitGBSeconds()
    try memoryPolicyRequire(deficit >= 20 && deficit < 21, "A 2 GB cut for about 10 s counted \(deficit) GB-seconds")
}

private final class HeldCheckLines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.withLock { lines.append(line) } }
    var all: [String] { lock.withLock { lines } }
}

/// The memory_limit suggestion: only a limit cut on three calendar days or more (300 cuts in one evening say nothing);
/// the level that held plus what the limit adds over its budget, down to 0.1 GB, never above the limit; plain text
/// naming the exact setting.
func memoryLimitAdvice() throws {
    let gb: UInt64 = 1_000_000_000
    let now = Date()
    let limit: UInt64 = 30_064_771_072
    // Two days apart: three different calendar days in any time zone.
    let trims = [23_150, 22_060, 24_600, 31_000].enumerated().map {
        MemoryHistory.Trim(
            t: now.timeIntervalSince1970 - 600 - Double($0.offset) * 2 * 86_400, to: UInt64($0.element) * 1_000_000, limit: limit)
    }
    let advice = MemoryHistory.limitAdvice(trims: trims, limit: limit, cap: limit - gb, minimum: 8 * gb, now: now)
    try memoryPolicyRequire(
        advice?["suggested_bytes"] as? UInt64 == 23_000_000_000 && advice?["cuts"] as? Int == 3
            && advice?["text"] as? String
                == "memory_limit 30.1 GB could not be held on this Mac (cut 3 times); about 23.0GB is the most it kept: set memory_limit to 23.0GB (`origami config set memory_limit 23.0GB`) or remove it to let Origami choose (`origami config unset memory_limit`)",
        "Limit advice: \(String(describing: advice))")
    try memoryPolicyRequire(
        MemoryHistory.limitAdvice(trims: Array(trims.prefix(2)), limit: limit, cap: limit - gb, minimum: 8 * gb, now: now) == nil,
        "Two cuts already produced advice")
    let evening = Calendar.current.startOfDay(for: now.addingTimeInterval(-600)).timeIntervalSince1970 + 12 * 3_600
    let oneEvening = (0..<300).map { MemoryHistory.Trim(t: evening + Double($0) * 10, to: UInt64(18 + $0 % 5) * gb, limit: limit) }
    try memoryPolicyRequire(
        MemoryHistory.limitAdvice(trims: oneEvening, limit: limit, cap: limit - gb, minimum: 8 * gb, now: now) == nil
            && MemoryHistory.limitAdvice(
                trims: oneEvening + Array(trims.prefix(2)), limit: limit, cap: limit - gb, minimum: 8 * gb, now: now) == nil,
        "Cuts on fewer than three days produced advice")
    try memoryPolicyRequire(
        MemoryHistory.limitAdvice(trims: trims, limit: limit, cap: limit - gb, minimum: 8 * gb, now: now.addingTimeInterval(15 * 86_400))
            == nil, "Cuts older than 14 days produced advice")
    // Cuts under a higher limit, or of unknown limit, say nothing about this one; a lower limit's count.
    func under(_ ceiling: UInt64?) -> [MemoryHistory.Trim] { trims.map { MemoryHistory.Trim(t: $0.t, to: $0.to, limit: ceiling) } }
    try memoryPolicyRequire(
        MemoryHistory.limitAdvice(trims: under(limit + gb), limit: limit, cap: limit - gb, minimum: 8 * gb, now: now) == nil
            && MemoryHistory.limitAdvice(trims: under(nil), limit: limit, cap: limit - gb, minimum: 8 * gb, now: now) == nil,
        "Cuts under a higher or unknown limit produced advice")
    try memoryPolicyRequire(
        MemoryHistory.limitAdvice(trims: under(limit + gb) + under(limit - gb), limit: limit, cap: limit - gb, minimum: 8 * gb, now: now)?[
            "cuts"] as? Int == 3, "The count of cuts included a higher limit's, or missed a lower one's")
}

/// After a pressure cut, the held level rises half its gap to the ceiling (at least 1 GB, at most free memory above the floor
/// and 1 GB) per 40 s quiet while free memory stays at the floor + 2 GB or more,
/// whether or not the caches fill it (they refill slowly), up to the ceiling; a limit the Mac cannot hold stays down.
/// Floor 4 GB, ceiling 25 GB; each case runs one-second steps through the real policy.
func memoryHeldLevelQuietRecovery() throws {
    let gb: UInt64 = 1_000_000_000
    let cap = 25 * gb
    struct Host {
        var policy: MemoryGovernorPolicy
        var usable: UInt64
        var rises: [(time: TimeInterval, to: UInt64)] = []
        mutating func step(_ now: TimeInterval, available: UInt64, under: UInt64 = 0, pressure: MemoryPressureLevel = .normal)
            -> MemoryGovernorPolicy.Decision?
        {
            let decision = policy.step(
                now: now, available: available, reserved: usable - under, usable: usable, swapouts: 0, pressure: pressure)
            if let decision {
                if decision.target > usable { rises.append((now, decision.target)) }
                usable = decision.target
            }
            return decision
        }
    }
    func host(_ mode: MemoryBudgetMode = .explicit, cap: UInt64 = cap) -> Host {
        Host(policy: MemoryGovernorPolicy(mode: mode, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0), usable: cap)
    }
    /// Today's cut: steady at the ceiling for a minute (recovery retries running), then two warnings 8 s apart.
    func cutTo17_7(_ host: inout Host) throws {
        for time in stride(from: 0.0, through: 60, by: 1) { _ = host.step(time, available: 7 * gb) }
        let first = host.step(61, available: 7 * gb, under: 5_300_000_000, pressure: .warning)
        try memoryPolicyRequire(
            host.step(65, available: 7 * gb, pressure: .warning) == nil, "A second warning escaped settling after a cut")
        let second = host.step(69, available: 7 * gb, pressure: .warning)
        // (6) The cut itself: the same depth (reserved - 1 GB on a warning) and settling (8 s) as before.
        try memoryPolicyRequire(
            first?.target == 18_700_000_000 && first?.reason == .warning && second?.target == 17_700_000_000,
            "The cut changed: \(String(describing: first)) \(String(describing: second))")
    }

    // (1) Quiet with 7 GB free, the caches staying 2 GB under the budget: half the gap to the ceiling per 40 s, at least
    // 1 GB, at most the 2 GB free memory holds above the floor and 1 GB: 19.7, 21.7, 23.35, 24.35, 25 GB.
    var today = host()
    try cutTo17_7(&today)
    for time in stride(from: 70.0, through: 1_000, by: 1) {
        _ = today.step(time, available: 7 * gb, under: 2 * gb)
        try memoryPolicyRequire(today.usable <= cap, "The budget passed the ceiling: \(today.usable)")
    }
    let steps = zip(today.rises, today.rises.dropFirst())
    try memoryPolicyRequire(
        today.usable == cap && today.rises.map(\.to) == [19_700, 21_700, 23_350, 24_350, 25_000].map { $0 * 1_000_000 }
            && steps.allSatisfy { $1.time - $0.time == 40 },
        "Quiet recovery: \(today.rises.map { "\(Int($0.time)) s \(Double($0.to) / 1e9)" })")

    // (2) The same with free memory under floor + 2 GB: no climb.
    var short = host()
    try cutTo17_7(&short)
    for time in stride(from: 70.0, through: 1_000, by: 1) { _ = short.step(time, available: 5_900_000_000, under: 2 * gb) }
    try memoryPolicyRequire(short.usable == 17_700_000_000, "Climbed without free memory to spare: \(short.usable)")

    // (3) A raise cut again within 40 s backs off: the next raise waits for the back-off to decay (120 s), then 40 s.
    var bounced = host()
    try cutTo17_7(&bounced)
    var raised: TimeInterval?
    var time = 70.0
    while raised == nil, time < 1_000 {
        _ = bounced.step(time, available: 7 * gb, under: 2 * gb)
        if bounced.usable > 17_700_000_000 { raised = time }
        time += 1
    }
    try memoryPolicyRequire(
        // The raise went to 19.7 GB; a warning takes 1 GB off what is held.
        bounced.step(raised! + 10, available: 7 * gb, pressure: .warning)?.target == 18_700_000_000, "A raise was not cut")
    let rises = bounced.rises.count
    time = raised! + 11
    while bounced.rises.count == rises, time < raised! + 1_000 {
        _ = bounced.step(time, available: 7 * gb, under: 2 * gb)
        time += 1
    }
    try memoryPolicyRequire(time - raised! - 10 > 160, "The raise after a reversal did not wait: \(time - raised! - 10) s")
    // A limit cut after every raise (here: pressure whenever the budget is above 19 GB) stays down for 10 minutes: a raise
    // passes 19 GB by at most one step (here at most the 2 GB free memory allows) before it is cut again.
    var unheld = host()
    try cutTo17_7(&unheld)
    for time in stride(from: 70.0, through: 70 + 600, by: 1) {
        _ = unheld.step(time, available: 7 * gb, under: 2 * gb, pressure: unheld.usable > 19 * gb ? .warning : .normal)
        try memoryPolicyRequire(unheld.usable <= 21 * gb, "A limit the Mac cannot hold bounced up: \(unheld.usable) at \(time) s")
    }

    // (4) No cut ever: unchanged at the ceiling.
    var steady = host()
    for time in stride(from: 0.0, through: 600, by: 1) {
        _ = steady.step(time, available: 7 * gb, under: 2 * gb)
        try memoryPolicyRequire(steady.usable == cap, "Without a cut the budget moved: \(steady.usable)")
    }

    // (5) A measured limit serves as an explicit one (`memory_limit`): the same climb, capped at the measured 21.7 GB.
    var measured = host(cap: 21_700_000_000)
    _ = measured.step(0, available: 7 * gb)
    _ = measured.step(1, available: 7 * gb, under: 2 * gb, pressure: .warning)
    try memoryPolicyRequire(measured.usable == 18_700_000_000, "Measured limit: the cut changed: \(measured.usable)")
    for time in stride(from: 2.0, through: 300, by: 1) {
        _ = measured.step(time, available: 7 * gb, under: 2 * gb)
        try memoryPolicyRequire(measured.usable <= 21_700_000_000, "Passed the measured limit: \(measured.usable)")
    }
    try memoryPolicyRequire(measured.usable == 21_700_000_000, "Measured limit: no climb back: \(measured.usable)")
    // Plain automatic mode shares the rule when free memory fits more than the held level.
    var automatic = host(.automatic)
    _ = automatic.step(0, available: 12 * gb)
    _ = automatic.step(1, available: 12 * gb, under: 2 * gb, pressure: .warning)
    let held = automatic.usable
    for time in stride(from: 2.0, through: 120, by: 1) { _ = automatic.step(time, available: 12 * gb, under: 2 * gb) }
    try memoryPolicyRequire(
        held == 22 * gb && automatic.rises.count == 2 && automatic.rises[1].time - automatic.rises[0].time == 40,
        "Automatic: \(held) \(automatic.rises.map { "\(Int($0.time)) s \(Double($0.to) / 1e9)" })")

    // (7) From 14.7 GB back to a 24 GB limit with free memory to spare: five quiet intervals of 40 s (19.35, 21.675,
    // 22.84, 23.84, 24 GB), the first 40 s after the cut; it was nine 1 GB steps. Free memory 1 byte under the floor + 2 GB
    // stops it; a new cut during the climb starts it over from the new level.
    func cutTo14_7(_ host: inout Host) throws {
        _ = host.step(0, available: 20 * gb)
        try memoryPolicyRequire(
            host.step(1, available: 20 * gb, under: 8_300_000_000, pressure: .warning)?.target == 14_700_000_000, "Cut to 14.7 GB")
    }
    var back = host(cap: 24 * gb)
    try cutTo14_7(&back)
    for time in stride(from: 2.0, through: 400, by: 1) { _ = back.step(time, available: 20 * gb) }
    let climb = zip(back.rises, back.rises.dropFirst())
    try memoryPolicyRequire(
        back.rises.map(\.to) == [19_350_000_000, 21_675_000_000, 22_837_500_000, 23_837_500_000, 24 * gb]
            && back.rises.first!.time == 42 && climb.allSatisfy { $1.time - $0.time == 40 },
        "Recovery from 14.7 GB: \(back.rises.map { "\(Int($0.time)) s \(Double($0.to) / 1e9)" })")
    var tight = host(cap: 24 * gb)
    try cutTo14_7(&tight)
    for time in stride(from: 2.0, through: 400, by: 1) { _ = tight.step(time, available: 6 * gb - 1) }
    try memoryPolicyRequire(tight.usable == 14_700_000_000, "Recovered without free memory above the floor + 2 GB: \(tight.usable)")
    var again = host(cap: 24 * gb)
    try cutTo14_7(&again)
    for time in stride(from: 2.0, through: 50, by: 1) { _ = again.step(time, available: 20 * gb) }
    _ = again.step(51, available: 20 * gb, under: 2 * gb, pressure: .warning)
    let restart = again.usable
    for time in stride(from: 52.0, through: 400, by: 1) { _ = again.step(time, available: 20 * gb) }
    try memoryPolicyRequire(
        restart == 16_350_000_000 && again.rises.count >= 2 && again.rises[1].to == 20_175_000_000 && again.rises[1].time > 51 + 40,
        "A cut during recovery did not start it over: \(restart) \(again.rises.map { "\(Int($0.time)) s \(Double($0.to) / 1e9)" })")
}
