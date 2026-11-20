import Foundation

/// Compile alongside Sources/origami/TuningConditions.swift. No server, Metal, pressure changes or sleeps.
@main
enum TuningConditionsChecks {
    struct Failed: Error { let message: String }
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw Failed(message: message) }
    }
    static func sample(
        _ time: Double, swap: UInt64 = 0, pressure: Int = 1, thermal: Int = 0,
        lowPower: Bool = false, budget: UInt64 = 22 << 30, ram: UInt64 = 32 << 30
    ) -> TuningConditions.Snapshot {
        .init(
            monotonic: time, physicalMemoryBytes: ram, swapoutBytes: swap,
            memoryPressure: pressure, thermalState: thermal, lowPower: lowPower, budgetBytes: budget)
    }
    static func main() throws {
        let mib: UInt64 = 1 << 20
        let gib: UInt64 = 1 << 30
        // A working laptop can keep warning/serious heat and Low Power Mode consistently, and have
        // more than 64 MB of background swapping, without invalidating each balanced comparison.
        let background = TuningConditions.assess(
            sample(40, swap: 400 * mib, pressure: 2, thermal: 2, lowPower: true),
            after: sample(0, pressure: 2, thermal: 2, lowPower: true))
        try require(background.disposition == .usable, "stable real-user conditions were rejected")
        let next = TuningConditions.assess(sample(80, swap: 800 * mib, pressure: 2), after: sample(40, swap: 400 * mib, pressure: 2))
        try require(next.disposition == .usable && next.freshSwapBytes == 400 * mib, "old swapping accumulated across samples")
        let recovery = TuningConditions.assess(sample(40, budget: 22_786_370_560), after: sample(0, budget: 21_786_370_560))
        try require(recovery.disposition == .usable, "ordinary governor recovery was rejected")

        let material = TuningConditions.assess(sample(40, budget: 27 * gib), after: sample(0, budget: 22 * gib))
        let shrinking = TuningConditions.assess(sample(40, budget: 20 * gib), after: sample(0, budget: 28 * gib))
        try require(
            material.disposition == .disrupted && shrinking.disposition == .disrupted
                && material.reason?.contains("allowance") == true, "material allowance changes were not explained")
        let critical = TuningConditions.assess(sample(0, pressure: 4, lowPower: true))
        try require(critical.disposition == .critical && critical.reason?.contains("memory") == true, "critical memory was tolerated")
        try require(TuningConditions.assess(sample(0, thermal: 3)).disposition == .critical, "critical heat was tolerated")
        try require(
            TuningConditions.assess(sample(30, lowPower: true), after: sample(0)).disposition == .disrupted,
            "a power-mode transition was mixed into a timing sample")
        try require(
            TuningConditions.assess(sample(30, thermal: 2), after: sample(0, thermal: 1)).disposition == .disrupted,
            "new serious thermal pressure was accepted")

        let storm = TuningConditions.assess(sample(30, swap: 3 * gib), after: sample(0))
        try require(
            storm.disposition == .disrupted && storm.reason?.contains("system-wide") == true,
            "sustained paging was accepted or incorrectly attributed to the engine")
        let trickle = TuningConditions.assess(sample(300, swap: 3 * gib), after: sample(0))
        try require(trickle.disposition == .usable, "a long background trickle became a cumulative abort")
        let reset = TuningConditions.assess(sample(30, swap: 1), after: sample(0, swap: UInt64.max))
        try require(reset.disposition == .usable && reset.freshSwapBytes == 0, "counter reset underflowed")
        try require(
            TuningConditions.assess(sample(.nan), after: sample(0)).disposition == .disrupted,
            "invalid elapsed time was used to judge swap rate")
        let tiny = TuningConditions.thresholds(physicalMemoryBytes: 1)
        let huge = TuningConditions.thresholds(physicalMemoryBytes: .max)
        try require(
            tiny.softSwapBytes > 0 && tiny.severeSwapBytes > tiny.softSwapBytes
                && huge.softSwapBytes < huge.severeSwapBytes && huge.severeSwapBytes <= 2 * gib,
            "machine scaling has no bounded usable range")
        let smaller = TuningConditions.assess(sample(5, swap: 256 * mib, ram: 8 * gib), after: sample(0, ram: 8 * gib))
        let larger = TuningConditions.assess(sample(5, swap: 256 * mib), after: sample(0))
        try require(smaller.disposition == .disrupted && larger.disposition == .usable, "machine-relative disturbance was ignored")

        var settling = TuningConditions.Settling(startedAt: 0, maximumSeconds: 90, maximumExtraSamples: 2)
        try require(!settling.permitRetry(for: critical, at: 1), "critical pressure allowed another warm-up")
        try require(
            settling.permitRetry(for: storm, at: 10) && settling.permitRetry(for: storm, at: 20)
                && !settling.permitRetry(for: storm, at: 30), "settling exceeded its sample budget")
        var expired = TuningConditions.Settling(startedAt: 0, maximumSeconds: 10)
        try require(!expired.permitRetry(for: storm, at: 10), "settling ignored the time budget")
        var invalid = TuningConditions.Settling(startedAt: .greatestFiniteMagnitude, maximumSeconds: .greatestFiniteMagnitude)
        try require(!invalid.permitRetry(for: storm, at: 1), "an overflowing deadline enabled settling")
        print(
            "Tuning conditions: passed (background activity, recovery, pressure, transitions, paging, counter reset and bounded settling)")
    }
}
