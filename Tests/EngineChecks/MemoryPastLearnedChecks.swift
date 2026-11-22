import Foundation
import OrigamiEngine

/// Automatic mode past the learned start (`MemoryGovernorPolicy.pastLearned`), with fake memory readings every 2 s: a
/// learned 14.7 GB under a 24 GB cap grows step by step to the cap with plenty free (a minute quiet before each step,
/// steps of at most 2 GB or half the gap); it stays with too little free; pressure right after a step cuts and doubles
/// the next wait, and an hour without a cut restores it; an explicit limit is never exceeded; the cut is recorded in
/// the history the learned start comes from.
func memoryGrowsPastLearned() throws {
    let gb: UInt64 = 1_000_000_000
    let learned: UInt64 = 14_700_000_000
    let cap = 24 * gb
    func policy(_ mode: MemoryBudgetMode = .automatic, cap: UInt64 = cap) -> MemoryGovernorPolicy {
        MemoryGovernorPolicy(mode: mode, cap: cap, floor: 4 * gb, minimum: 8 * gb, swapouts: 0)
    }
    /// Runs `seconds` of 2 s ticks from `start`; `free` gives the free memory for the budget (caches full at it), and
    /// `pressure` the level at a time. Returns every change, with the budget it ended at.
    func run(
        _ p: inout MemoryGovernorPolicy, usable: inout UInt64, from start: TimeInterval = 0, seconds: TimeInterval,
        free: (UInt64) -> UInt64, learned: UInt64? = learned, pressure: (TimeInterval) -> MemoryPressureLevel = { _ in .normal }
    ) -> [(at: TimeInterval, from: UInt64, to: UInt64)] {
        var changes: [(TimeInterval, UInt64, UInt64)] = []
        var now = start
        while now <= start + seconds {
            if let change = p.step(
                now: now, available: free(usable), reserved: usable, usable: usable, swapouts: 0, pressure: pressure(now), learned: learned),
                change.reason != .growthRetry
            {
                changes.append((now, usable, change.target))
                usable = change.target
            }
            now += 2
        }
        return changes
    }
    let plenty = { (usable: UInt64) in 40 * gb - usable }

    // Plenty free: a minute quiet, then steps of at most 2 GB or half the gap, each a minute after the last, to the cap.
    var grows = policy()
    var usable = learned
    let steps = run(&grows, usable: &usable, seconds: 900, free: plenty)
    let sizes = steps.map { $0.to - $0.from }
    let apart = zip(steps.dropFirst(), steps).allSatisfy { $0.at - $1.at >= MemoryGovernorPolicy.pastLearnedQuiet }
    let sized = steps.allSatisfy {
        let gap = cap - $0.from
        return $0.to > $0.from && $0.to - $0.from <= min(2 * gb, max(gap / 2, min(gap, gb / 2)))
    }
    try pastLearnedRequire(
        steps.first?.at == 60 && apart && sized && usable == cap && sizes.prefix(3).allSatisfy { $0 == 2 * gb } && steps.count == 7,
        "Past the learned start: \(steps.map { "\($0.at) s \(Double($0.to) / 1e9) GB" })")

    // Too little free (the step would not keep the floor and 1 GB): it stays at the learned start.
    var tight = policy()
    usable = learned
    let none = run(&tight, usable: &usable, seconds: 900, free: { _ in 5_300_000_000 })
    try pastLearnedRequire(none.isEmpty && usable == learned, "Grew without free memory: \(none)")

    // Pressure right after a step cuts, and doubles the next wait; an hour without a cut restores a minute.
    var burnt = policy()
    usable = learned
    let first = run(&burnt, usable: &usable, seconds: 60, free: plenty)
    let cut = run(&burnt, usable: &usable, from: 62, seconds: 0, free: plenty, pressure: { _ in .warning })
    try pastLearnedRequire(
        first.count == 1 && cut.count == 1 && cut[0].to < cut[0].from && burnt.pastBackoff == 1,
        "Pressure after a step: \(first), \(cut), back-off \(burnt.pastBackoff)")
    let slower = run(&burnt, usable: &usable, from: 64, seconds: 300, free: plenty)
    try pastLearnedRequire(
        slower.first.map {
            $0.at - 64 >= 2 * MemoryGovernorPolicy.pastLearnedQuiet && $0.at - 64 < 2 * MemoryGovernorPolicy.pastLearnedQuiet + 4
        }
            == true, "The wait after a regretted step was not doubled: \(slower.map(\.at))")
    _ = run(&burnt, usable: &usable, from: 400, seconds: 0, free: plenty, pressure: { _ in .warning })
    let calm = 400 + MemoryGovernorPolicy.pastLearnedCalm + 10
    _ = run(&burnt, usable: &usable, from: 402, seconds: calm - 402, free: { _ in 5_300_000_000 })
    let restored = run(&burnt, usable: &usable, from: calm + 2, seconds: 200, free: plenty)
    try pastLearnedRequire(
        burnt.pastBackoff == 0 && restored.first.map { $0.at - (calm + 2) <= MemoryGovernorPolicy.pastLearnedQuiet + 2 } == true,
        "An hour without a cut did not restore the minute: back-off \(burnt.pastBackoff), \(restored.map(\.at))")

    // An explicit limit is never exceeded, whatever is free or learned.
    var limited = policy(.explicit, cap: 20 * gb)
    usable = learned
    let explicit = run(&limited, usable: &usable, seconds: 1_800, free: plenty)
    try pastLearnedRequire(usable == 20 * gb && explicit.allSatisfy { $0.to <= 20 * gb }, "An explicit limit was exceeded: \(explicit)")

    // The cut after a step is recorded in the history the learned start comes from, as before.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-past-learned-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let history = MemoryHistory(file: folder.appendingPathComponent("memory-history.json"), identity: "fixture")
    let ledger = try MemoryLedger(ceiling: 25 * gb, transientMargin: gb, availableFloor: 4 * gb, budgetMode: .automatic)
    let governor = MemoryGovernor(
        ledger: ledger, memory: OrigamiMemory(cap: cap, start: learned, floor: 4 * gb, minimum: 8 * gb), trim: { 0 }, log: { _ in },
        history: history, learns: true, initialSwapouts: 0)
    ledger.setUsable(learned)
    var replay = policy()
    var budget = learned
    for now in stride(from: 0.0, through: 64, by: 2) {
        if let change = replay.step(
            now: now, available: plenty(budget), reserved: budget, usable: budget, swapouts: 0, pressure: now == 62 ? .warning : .normal,
            learned: learned)
        {
            governor.apply(change, usable: budget, available: plenty(budget), footprint: 0)
            budget = change.target
        }
    }
    let trims = history.summary(cap: cap, minimum: 8 * gb, applied: true)["trims"] as? Int
    try pastLearnedRequire(
        trims == 1 && ledger.usableBytes == budget && budget < learned + 2 * gb, "The cut was not recorded: \(trims ?? -1)")
}

private func pastLearnedRequire(_ condition: @autoclosure () -> Bool, _ message: @autoclosure () -> String) throws {
    guard condition() else { throw EngineError(message()) }
}
