import Dispatch
import Foundation
import Synchronization

/// Keeps Origami a good neighbour. It watches the Mac's free memory and macOS memory-pressure
/// events and fresh swap. An explicit limit stays the target until those real pressure
/// signals or a hard physical-ceiling overrun require relief; automatic also follows estimated headroom.
/// After pressure clears the cache recovers (Qwen: its trimmed slabs return mid-reply).
public final class MemoryGovernor: @unchecked Sendable {
    private let ledger: MemoryLedger
    private let memory: OrigamiMemory
    private let trim: @Sendable () -> UInt64
    /// After the budget grows back: the engine prepares the cache it trimmed, mid-reply (Qwen; nothing by default).
    private let grow: @Sendable () -> Void
    private let log: @Sendable (String) -> Void
    private let queue = DispatchQueue(label: "origami.memory-governor", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var pressure: DispatchSourceMemoryPressure?
    private var policy: MemoryGovernorPolicy
    private var pressureLevel = MemoryPressureLevel.normal
    private var lastLog = Date.distantPast
    /// A budget change the 10 s limit held back (merged with the next in the same direction): it is never lost.
    private typealias Change = (reason: String, from: UInt64, to: UInt64, retired: UInt64, steps: Int, first: Date, last: Date)
    private var held: Change?
    private let decisionFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
    /// Boundary pokes arrive up to hundreds of times a second while the host is short: at most one check per 0.25 s.
    private var lastPoke = Date.distantPast
    private let gb: UInt64 = 1_000_000_000
    /// This Mac's memory over time (`MemoryHistory`): each check and trim is recorded. `learns`: its learned budget
    /// caps the start and every regrowth (false when the user's memory_limit sets the budget).
    private let history: MemoryHistory?, learns: Bool
    /// The budget's ceiling: the cap, within the ledger's hard limit.
    private let budgetCap: UInt64
    private let pressureCutCount = Atomic<Int>(0)
    /// Budget cuts for real pressure since start (`MemoryGovernorPolicy.Reason.isPressure`), for `/v1/status`.
    public var pressureCuts: Int { pressureCutCount.load(ordering: .relaxed) }
    private let lowestCutTo = Atomic<UInt64>(.max)
    /// The lowest budget a pressure cut set since start, or nil (the budget may have grown back since), for `/v1/status`.
    public var lowestPressureCut: UInt64? {
        let value = lowestCutTo.load(ordering: .relaxed)
        return value == .max ? nil : value
    }
    /// What the budget climbs to: the cap, in either mode (in automatic mode past a learned start too, more carefully:
    /// `MemoryGovernorPolicy.pastLearned`). A budget below it is temporarily reduced.
    public var target: UInt64 { budgetCap }
    /// What a cut is counted against (`budgetDeficitGBSeconds`, a request's "memory was cut"): the learned start in
    /// automatic mode when there is one, else the cap; a budget still growing past its learned start was not cut.
    private var deficitLevel: UInt64 { learned.map { min($0, budgetCap) } ?? budgetCap }
    private let cutLock = NSLock()
    private var lastCutSaid: (reason: String, at: Date)?
    /// The last cut's reason and time, or nil (none since load), for `/v1/status`.
    public var lastCut: (reason: String, at: Date)? { cutLock.withLock { lastCutSaid } }

    /// The budget's gap below its `target` over time: each change with the GB-seconds before it.
    private let deficitLock = NSLock()
    private var deficitMarks: [(at: TimeInterval, total: Double, gap: Double)] = []
    package func markDeficit(now: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        let target = deficitLevel
        let gap = Double(target - min(target, ledger.usableBytes)) / 1e9
        deficitLock.withLock {
            guard deficitMarks.last?.gap != gap else { return }
            deficitMarks.append((now, Self.total(deficitMarks, at: now), gap))
            // Old marks only serve requests older than a few hours; the total since load stays in the first one kept.
            if deficitMarks.count > 4_096 { deficitMarks.removeFirst(deficitMarks.count - 4_096) }
        }
    }
    private static func total(_ marks: [(at: TimeInterval, total: Double, gap: Double)], at time: TimeInterval) -> Double {
        guard let mark = marks.last(where: { $0.at <= time }) else { return marks.first?.total ?? 0 }
        return mark.total + mark.gap * (time - mark.at)
    }
    /// GB-seconds the budget has spent below its target since load, or in the last `seconds` (a request's span).
    public func budgetDeficitGBSeconds(last seconds: TimeInterval = .infinity) -> Double {
        let now = ProcessInfo.processInfo.systemUptime
        return deficitLock.withLock {
            Self.total(deficitMarks, at: now) - (seconds.isFinite ? Self.total(deficitMarks, at: now - seconds) : 0)
        }
    }

    public init(
        ledger: MemoryLedger, memory: OrigamiMemory, trim: @escaping @Sendable () -> UInt64,
        log: @escaping @Sendable (String) -> Void, history: MemoryHistory? = nil, learns: Bool = false,
        initialSwapouts: UInt64? = nil,
        grow: @escaping @Sendable () -> Void = {}
    ) {
        self.ledger = ledger
        self.memory = memory
        self.trim = trim
        self.log = log
        self.history = history
        self.learns = learns
        self.grow = grow
        let cap = min(memory.cap, ledger.maximumUsableBytes)
        budgetCap = cap
        // A limit (memory_limit or measured) always starts at itself: earlier cuts only make advice (`limitAdvice`) or
        // a measured limit stale, so one evening of pressure does not slow the next two weeks.
        policy = MemoryGovernorPolicy(
            mode: ledger.budgetMode, cap: cap,
            floor: memory.floor, minimum: memory.minimum, swapouts: initialSwapouts ?? swapoutBytes(),
            hardCeiling: ledger.ceiling)
    }
    /// Under memory_limit: the suggestion when this Mac could not hold it (`MemoryHistory.limitAdvice`), or nil.
    public var limitAdvice: [String: Any]? {
        ledger.budgetMode == .explicit ? history?.limitAdvice(limit: memory.cap, cap: budgetCap, minimum: memory.minimum) : nil
    }
    /// The learned budget in force, or nil.
    public var learned: UInt64? {
        learns && ledger.budgetMode == .automatic
            ? history?.budget(cap: min(memory.cap, ledger.maximumUsableBytes), minimum: memory.minimum, ceiling: memory.cap) : nil
    }

    public func start() {
        if let start = learned, start + gb / 4 < ledger.usableBytes {
            let from = ledger.usableBytes
            ledger.setUsable(start)
            ledger.recordCacheReliefRequest(targetBytes: start)
            _ = trim()
            log(
                "memory: starting at \(OrigamiPlan.gigabytes(start)) instead of \(OrigamiPlan.gigabytes(from)), learned from this Mac's memory use (`memory_limit` sets it instead)"
            )
        }
        markDeficit()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 2, repeating: 2)
        timer.setEventHandler { [weak self] in self?.tick(event: nil) }
        timer.resume()
        self.timer = timer
        let pressure = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: queue)
        pressure.setEventHandler { [weak self, weak pressure] in self?.tick(event: pressure?.data) }
        pressure.resume()
        self.pressure = pressure
        // Execution boundaries that see the host short of memory ask for an early check.
        ledger.onPressure = { [weak self] in
            guard let self else { return }
            self.queue.async { self.poke() }
        }
        // An already-pressured Mac need not send a new edge after this listener starts.
        queue.async { [weak self] in self?.tick(event: nil) }
    }
    private func poke() {
        guard Date().timeIntervalSince(lastPoke) >= 0.25 else { return }
        lastPoke = Date()
        tick(event: nil)
    }
    deinit {
        timer?.cancel()
        pressure?.cancel()
    }

    /// Targets are absolute (from this process's reservations and the free memory now), so repeated
    /// checks while slabs are still being released do not keep lowering the budget.
    private func tick(event: DispatchSource.MemoryPressureEvent?) {
        defer { markDeficit() }
        let wasAllowed = ledger.optionalGrowthAllowed
        if let change = held, Date().timeIntervalSince(lastLog) >= 10 {
            held = nil
            write(change)
        }
        // The event is also authoritative, including platforms where the sysctl is unavailable.
        if event?.contains(.critical) == true {
            pressureLevel = .critical
        } else if event?.contains(.warning) == true {
            pressureLevel = .warning
        } else if event?.contains(.normal) == true {
            pressureLevel = .normal
        }
        // The fresh snapshot wins over an older notification already queued for delivery.
        if let current = MemoryPressureLevel.current() { pressureLevel = current }
        // Close optional admission before the remaining host probes or any deferred trim.
        if pressureLevel.isElevated {
            ledger.setOptionalReadsAllowed(false)
            ledger.setOptionalGrowthAllowed(false)
        }
        let available = availableMemoryBytes()
        let reserved = ledger.reservedBytes
        let usable = ledger.usableBytes
        let footprint = processFootprintBytes()
        history?.observe(fit: MemoryGovernorPolicy.fit(reserved: reserved, available: available, floor: memory.floor))
        let change = policy.step(
            now: ProcessInfo.processInfo.systemUptime, available: available, reserved: reserved,
            usable: usable, swapouts: swapoutBytes(), pressure: pressureLevel, footprint: footprint, learned: learned)
        ledger.setOptionalGrowthAllowed(!policy.optionalGrowthInhibited)
        ledger.setOptionalReadsAllowed(!policy.optionalReadsPaused)
        guard let change else {
            // Even at the compulsory floor, stop prepared optional owners from sitting
            // charged until a later target reduction. The trim callback preserves that floor.
            if wasAllowed, !ledger.optionalGrowthAllowed {
                ledger.recordCacheReliefRequest(targetBytes: usable)
                _ = trim()
            }
            // Recovery can reopen admission while already at a minimum/cap, with no target change.
            if !wasAllowed, ledger.optionalGrowthAllowed { grow() }
            return
        }
        if change.reason == .growthRetry {
            grow()
            return
        }
        apply(change, usable: usable, available: available, footprint: footprint)
    }

    /// A policy decision carried out: the budget set, a cut recorded in the history (the learned start learns from it,
    /// also after growth past it) and trimmed, a growth handed to the engine; each logged.
    package func apply(_ change: MemoryGovernorPolicy.Decision, usable: UInt64, available: UInt64, footprint: UInt64) {
        ledger.setUsable(change.target)
        if change.target < usable {
            history?.trimmed(to: change.target, limit: memory.cap)
            if change.reason.isPressure {
                pressureCutCount.add(1, ordering: .relaxed)
                lowestCutTo.min(change.target, ordering: .relaxed)
            }
            ledger.recordCacheReliefRequest(targetBytes: change.target)
            let freed = trim()
            let reason: String
            switch change.reason {
            case .warning: reason = "macOS reports memory pressure"
            case .critical: reason = "macOS reports critical memory pressure"
            case .swap: reason = "the system swap counter increased"
            case .footprint: reason = "process memory exceeds its hard ceiling (\(OrigamiPlan.gigabytes(footprint)) used)"
            default: reason = "memory is short (\(OrigamiPlan.gigabytes(available)) free)"
            }
            cutLock.withLock { lastCutSaid = (reason, Date()) }
            note(
                reason
                    + (change.freshSwap > 0
                        ? " (\(String(format: "%.2f", Double(change.freshSwap) / 1_048_576)) MiB globally swapped)" : ""),
                from: usable, to: change.target, retired: freed)
        } else {
            history?.grew()
            grow()
            note(
                ledger.budgetMode == .explicit
                    ? "memory pressure is clear; recovering toward the requested limit"
                    : learned.map { change.target > $0 } == true
                        ? "memory has stayed free; growing past the learned start" : "memory is free again",
                from: usable, to: change.target)
        }
    }

    /// At most one line per 10 s, so a long squeeze does not flood the log; held changes merge into one line
    /// ("budget 24.1 GB -> 17.1 GB in 2 steps"), and a change of direction prints the held one at once.
    private func note(_ reason: String, from: UInt64, to: UInt64, retired: UInt64 = 0) {
        let now = Date()
        var change: Change = (reason, from, to, retired, 1, now, now)
        if let held {
            if (held.to < held.from) == (to < from) {
                change.from = held.from
                change.steps += held.steps
                change.first = held.first
                change.retired += held.retired
            } else {
                write(held)
            }
            self.held = nil
        }
        guard Date().timeIntervalSince(lastLog) >= 10 else {
            held = change
            return
        }
        write(change)
    }
    private func write(_ change: Change) {
        lastLog = Date()
        log(
            "\(change.reason): budget \(OrigamiPlan.gigabytes(change.from)) -> \(OrigamiPlan.gigabytes(change.to))"
                + (change.steps > 1 ? " in \(change.steps) steps" : "")
                + (change.to < change.from
                    ? "; cache ownership retired \(OrigamiPlan.gigabytes(change.retired)) (busy owners may retire later)" : "")
                + "; decision \(decisionFormatter.string(from: change.first))"
                + (change.steps > 1 ? " ... \(decisionFormatter.string(from: change.last))" : ""))
    }
}
