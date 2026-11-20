import CExactIO
import Foundation
import Metal
import OrigamiEngine

private func growthPressureRequire(_ condition: Bool, _ message: String) throws {
    guard condition else { throw EngineError(message) }
}

/// No host timing or allocations: pressure at a compulsory floor must still close optional
/// admission, then reopen it at the saved cap. Tiny unrelated swap noise must not close it.
func memoryGrowthPressureRecovery() throws {
    let cap: UInt64 = 24_000_000_000
    var policy = MemoryGovernorPolicy(
        mode: .explicit, cap: cap, floor: 4_000_000_000,
        minimum: cap, swapouts: 0)
    @discardableResult func step(
        _ now: Double, _ pressure: MemoryPressureLevel = .normal,
        swap: UInt64 = 0, footprint: UInt64 = 0
    ) -> MemoryGovernorPolicy.Decision? {
        policy.step(
            now: now, available: 0, reserved: cap, usable: cap, swapouts: swap,
            pressure: pressure, footprint: footprint)
    }
    try growthPressureRequire(
        step(0) == nil && !policy.optionalGrowthInhibited,
        "Explicit target was inhibited by an available-page estimate")
    try growthPressureRequire(
        step(1, .warning) == nil && policy.optionalGrowthInhibited,
        "Pressure at the compulsory floor failed to stop optional growth")
    step(2)
    step(21)
    try growthPressureRequire(policy.optionalGrowthInhibited, "Growth reopened before the quiet recovery interval")
    step(22)
    try growthPressureRequire(!policy.optionalGrowthInhibited, "A full explicit cap never reopened after pressure cleared")
    step(23, swap: 16_384)
    try growthPressureRequire(!policy.optionalGrowthInhibited, "One tiny global swap write stopped optional growth")
    step(24, swap: 32_768)
    try growthPressureRequire(policy.optionalGrowthInhibited, "Repeated fresh swap writes did not stop optional growth")
    step(25, swap: 32_768)
    step(64, swap: 32_768)
    try growthPressureRequire(policy.optionalGrowthInhibited, "A pressure reversal ignored recovery backoff")
    step(65, swap: 32_768)
    try growthPressureRequire(!policy.optionalGrowthInhibited, "Swap relief permanently capped optional growth")
    step(66, swap: 32_768, footprint: cap + 1)
    try growthPressureRequire(policy.optionalGrowthInhibited, "A physical-ceiling overrun left optional admission open")

    var automatic = MemoryGovernorPolicy(
        mode: .automatic, cap: cap, floor: 4_000_000_000,
        minimum: 1_000_000_000, swapouts: 0)
    _ = automatic.step(now: 0, available: 0, reserved: cap, usable: cap, swapouts: 0, pressure: .normal)
    try growthPressureRequire(automatic.optionalGrowthInhibited, "Automatic host headroom no longer inhibits growth")
}

/// Exercise real cache/Metal ownership without running a GPU kernel. Blocking optional
/// preparation must preserve compulsory demand, and an escaped raw buffer keeps its charge.
func memoryGrowthPressureOwners() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
    let slab = UInt64(8 * PackStore.expertBytes)
    // Physical admission counts the whole process, which the checks before this one can leave above 2 GB: 2 GB on top.
    let ceiling = origami_physical_footprint() + 2_000_000_000
    let ledger = try MemoryLedger(ceiling: ceiling, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: 2 * slab, maximumByteLimit: 4 * slab)
    cache.setGrowthCeiling(4 * slab)
    ledger.setOptionalGrowthAllowed(false)
    try growthPressureRequire(
        try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false) == 0,
        "Pressure allowed optional preparation")
    var escaped: MTLBuffer?
    try autoreleasepool {
        let acquisition = try cache.acquire(ExpertKey(layer: 0, expert: 0))
        let record = try cache.beginRead(acquisition.ticket)
        record.bytes[0] = 0x5a
        try growthPressureRequire(cache.completeRead(acquisition.ticket, verified: true), "Compulsory demand failed under pressure")
        let held = try cache.pin(acquisition.ticket)
        escaped = held.buffer
        try cache.unpinAfterGPUCompletion(acquisition.ticket)
        ledger.recordCacheReliefRequest(targetBytes: 1)
        try cache.resize(to: 0)
    }
    var relief = ledger.cacheReliefSnapshot
    try growthPressureRequire(
        relief.requestCount == 1 && relief.cacheBytesAtLastRequest == slab
            && relief.cacheBytesReleasedSinceLastRequest == 0 && relief.cacheOwnedBytes == slab,
        "Cache detachment was reported as completed owner retirement")
    try autoreleasepool {
        try growthPressureRequire(escaped?.contents().load(as: UInt8.self) == 0x5a, "Escaped resource became invalid")
        escaped = nil
    }
    relief = ledger.cacheReliefSnapshot
    try growthPressureRequire(
        relief.cacheBytesReleasedSinceLastRequest == slab && relief.cacheOwnedBytes == 0
            && ledger.reservedBytes == 0, "Last resource retirement did not complete the cache charge")
    let unrelated = try ledger.reserve(bytes: 123, category: "unrelated-workspace")
    try ledger.release(unrelated)
    try growthPressureRequire(ledger.cacheReliefSnapshot == relief, "Unrelated workspace retirement counted as expert relief")
    cache.restoreGrowthLimit()
    try growthPressureRequire(cache.snapshot().byteLimit == 0, "Pressure restored a lazy growth allowance")
    ledger.setOptionalGrowthAllowed(true)
    cache.restoreGrowthLimit()
    try growthPressureRequire(
        cache.snapshot().byteLimit == 4 * slab && ledger.maximumUsableBytes == ceiling - 1,
        "Recovery failed to restore the original growth target")
    let snapshot = ledger.snapshot()
    try growthPressureRequire(
        snapshot.optionalGrowthAllowed == true && snapshot.cacheRelief == relief,
        "Boundary diagnostics differ from actual admission/ownership state")
}

/// Qwen's prompt read-ahead follows live pressure, not the recovery: a cut pauses it for its settle window only, while
/// cache growth stays latched for the quiet interval; live pressure pauses it; a second cut pauses it at once. (The
/// family engines' guesses and read-ahead do not read this flag: they only fill slots the pool has, and a refusal of a
/// slab they needed is counted as live while it is closed, `family_pressure_prefetch_floor`.)
func memoryLivePressureReads() throws {
    let gb: UInt64 = 1_000_000_000
    let cap = 24 * gb
    var policy = MemoryGovernorPolicy(mode: .explicit, cap: cap, floor: 4 * gb, minimum: 2 * gb, swapouts: 0)
    var usable = cap
    func step(_ now: Double, _ pressure: MemoryPressureLevel = .normal, swap: UInt64 = 0) {
        let decision = policy.step(
            now: now, available: 0, reserved: usable, usable: usable, swapouts: swap, pressure: pressure)
        if let decision { usable = decision.target }
    }
    step(0)
    try growthPressureRequire(!policy.optionalReadsPaused, "Reads paused with no pressure")
    step(2, swap: 400_000_000)
    try growthPressureRequire(
        usable < cap && policy.optionalReadsPaused && policy.optionalGrowthInhibited, "A swap cut left reads open")
    step(4, swap: 400_000_000)
    try growthPressureRequire(
        !policy.optionalReadsPaused && policy.optionalGrowthInhibited && usable < cap,
        "Reads stayed paused after the cut settled, or growth reopened early")
    let afterFirst = usable
    step(6, .warning, swap: 400_000_000)
    try growthPressureRequire(policy.optionalReadsPaused && usable == afterFirst, "Live pressure left reads open")
    step(12, .warning, swap: 400_000_000)
    try growthPressureRequire(policy.optionalReadsPaused && usable < afterFirst, "A warning cut left reads open")
    step(14, swap: 400_000_000)
    try growthPressureRequire(policy.optionalReadsPaused, "A warning cut's 8 s settle window reopened reads early")
    step(20, swap: 400_000_000)
    try growthPressureRequire(!policy.optionalReadsPaused, "Reads stayed paused once the warning cut settled")
    step(22, swap: 1_400_000_000)
    try growthPressureRequire(policy.optionalReadsPaused, "A second swap cut did not pause reads at once")

    // The ledger's flags as the governor sets them; refusals split by cause, seconds joined within 2 s.
    let ledger = try MemoryLedger(ceiling: 2 * gb, transientMargin: 1)
    ledger.setOptionalGrowthAllowed(!policy.optionalGrowthInhibited)
    ledger.setOptionalReadsAllowed(!policy.optionalReadsPaused)
    try growthPressureRequire(!ledger.optionalReadsAllowed && !ledger.optionalGrowthAllowed, "Ledger flags differ from policy")
    ledger.noteOptionalRefusal(live: true, count: 2, now: 10)
    ledger.noteOptionalRefusal(live: true, now: 11.5)
    ledger.noteOptionalRefusal(live: true, now: 20)
    ledger.noteOptionalRefusal(live: false, now: 21)
    let r = ledger.optionalRefusals
    try growthPressureRequire(
        r.live == 4 && r.liveSeconds == 1.5 && r.belowTarget == 1 && r.belowTargetSeconds == 0,
        "Refusal counters differ: \(r)")
}
