import CExactIO
import Foundation
import Synchronization

/// Reservation accounting, not an allocator. All allocations must first obtain a reservation.
/// A shared CPU/Metal allocation owns a single reservation until all users have retired.
/// Reservations are bounded by `ceiling - transientMargin`. The usable budget moves below that
/// at runtime (`MemoryGovernor`): it is the target the expert cache sizes itself to, shrinking
/// when the Mac needs memory back and growing when memory is free again. Lowering it releases
/// cache slabs; it never refuses another owner's reservation.
public final class MemoryLedger: @unchecked Sendable {
    /// One live charge; release it exactly once (a second release throws).
    public struct Reservation: Hashable, Sendable { fileprivate let id: UUID }
    /// Totals by category plus the process footprint and host free memory at the time of the call.
    public struct Snapshot: Codable, Sendable {
        public let ceiling, transientMargin, reserved, reservationHighWater, physicalFootprint, availableMemoryEstimate: UInt64
        /// The current budget when it is below the maximum (`ceiling - transientMargin`).
        public let usableBytes: UInt64?
        public let categories: [String: UInt64]
        /// Additive diagnostics; older saved receipts need not contain these fields.
        public var optionalGrowthAllowed: Bool? = nil
        public var cacheRelief: MemoryCacheReliefSnapshot? = nil
        /// Measured process bytes outside materialized owners; allowances remain separately charged.
        public var physicalOverheadBytes: UInt64? = nil
        public var prospectiveAllowanceBytes: UInt64? = nil
    }
    private struct Entry {
        let bytes: UInt64
        let category: String
        var allowance: Bool = false
    }
    private let lock = NSCondition()  // broadcast at every release, for `waitForRelease`
    public let ceiling, transientMargin: UInt64
    public let budgetMode: MemoryBudgetMode
    /// An explicit target does not shrink or stop cache admission because free-page estimates are low.
    public var usesAvailableMemoryFloor: Bool { budgetMode == .automatic }
    /// Largest budget the governor may grow to.
    public var maximumUsableBytes: UInt64 { ceiling - transientMargin }
    /// Automatic mode's host headroom. Explicit mode reports it as advisory and follows actual pressure instead.
    public let availableFloorBytes: UInt64
    private var usable: UInt64
    private var entries: [UUID: Entry] = [:]
    private var reserved: UInt64 = 0
    private var highWater: UInt64 = 0
    private var allowances: UInt64 = 0, physicalOverhead: UInt64 = 0
    private let growthAllowed = Atomic<Bool>(true)
    private let readsAllowed = Atomic<Bool>(true)
    private let refusals = Mutex(OptionalRefusals())
    private var cacheOwned: UInt64 = 0
    private var reliefRequests: UInt64 = 0, cacheAtRelief: UInt64 = 0, cacheReleasedAfterRelief: UInt64 = 0
    private var reliefTarget: UInt64?
    private var pressureHook: (@Sendable () -> Void)?
    /// Called (without the ledger lock) when an execution boundary sees the host short of memory.
    public var onPressure: (@Sendable () -> Void)? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return pressureHook
        }
        set {
            lock.lock()
            pressureHook = newValue
            lock.unlock()
        }
    }

    public init(
        ceiling: UInt64 = 22_000_000_000, transientMargin: UInt64 = 1_000_000_000,
        usableLimit: UInt64? = nil, availableFloor: UInt64 = 3 * 1_024 * 1_024 * 1_024,
        budgetMode: MemoryBudgetMode = .automatic
    ) throws {
        guard ceiling > transientMargin else { throw EngineError("Memory ceiling must exceed transient margin") }
        guard availableFloor > 0 || budgetMode == .explicit else { throw EngineError("Available-memory floor must be positive") }
        availableFloorBytes = availableFloor
        self.budgetMode = budgetMode
        let usable = usableLimit ?? (ceiling - transientMargin)
        guard usable > 0, usable <= ceiling - transientMargin else { throw EngineError("Invalid effective memory owner budget") }
        self.ceiling = ceiling
        self.transientMargin = transientMargin
        self.usable = usable
    }

    /// The current owner budget.
    public var usableBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return usable
    }
    public var reservedBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return reserved
    }
    /// A cheap authoritative-pressure latch for optional cache allocation. Mandatory streaming
    /// floors and existing records remain usable; the user's immutable target never changes.
    public var optionalGrowthAllowed: Bool { growthAllowed.load(ordering: .acquiring) }
    package func setOptionalGrowthAllowed(_ allowed: Bool) { growthAllowed.store(allowed, ordering: .releasing) }
    /// Qwen's prompt read-ahead into slots the cache already holds (the family engines' guesses do not stop for it):
    /// closed only while pressure is live (the governor's `optionalReadsPaused`), not for the whole recovery after a cut.
    /// It allocates nothing, so the smaller cache of a lowered budget is its bound; new slabs still wait for `optionalGrowthAllowed`.
    public var optionalReadsAllowed: Bool { readsAllowed.load(ordering: .acquiring) }
    package func setOptionalReadsAllowed(_ allowed: Bool) { readsAllowed.store(allowed, ordering: .releasing) }

    /// Read-ahead scopes and guesses refused for memory, cumulative: while pressure was live, or while the budget was
    /// below its target with no live pressure (a cache too small for a broad scope, a guess that needed a slab).
    /// Seconds join refusals of one cause under 2 s apart (one governor tick); a lone refusal adds none.
    public struct OptionalRefusals: Sendable {
        public var live = 0, belowTarget = 0
        public var liveSeconds = 0.0, belowTargetSeconds = 0.0
        fileprivate var lastLive = -Double.infinity, lastBelow = -Double.infinity
    }
    public var optionalRefusals: OptionalRefusals { refusals.withLock { $0 } }
    package func noteOptionalRefusal(live: Bool, count: Int = 1, now: Double = ProcessInfo.processInfo.systemUptime) {
        guard count > 0 else { return }
        refusals.withLock {
            if live {
                $0.live += count
                if now - $0.lastLive < 2 { $0.liveSeconds += now - $0.lastLive }
                $0.lastLive = now
            } else {
                $0.belowTarget += count
                if now - $0.lastBelow < 2 { $0.belowTargetSeconds += now - $0.lastBelow }
                $0.lastBelow = now
            }
        }
    }

    /// A requested target is distinct from physical owner retirement. The release counter
    /// advances only when a charged expert owner actually retires, including escaped buffers.
    package func recordCacheReliefRequest(targetBytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        reliefRequests &+= 1
        reliefTarget = targetBytes
        cacheAtRelief = cacheOwned
        cacheReleasedAfterRelief = 0
    }
    public var cacheReliefSnapshot: MemoryCacheReliefSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return cacheReliefLocked()
    }
    private func cacheReliefLocked() -> MemoryCacheReliefSnapshot {
        MemoryCacheReliefSnapshot(
            requestCount: reliefRequests, requestedTargetBytes: reliefTarget,
            cacheBytesAtLastRequest: cacheAtRelief, cacheBytesReleasedSinceLastRequest: cacheReleasedAfterRelief,
            cacheOwnedBytes: cacheOwned)
    }
    private static func isExpertOwner(_ category: String) -> Bool { category == "expert-slabs" || category == "family-experts" }

    /// Moves the budget within (0, maximum]. Live reservations are never revoked; a budget
    /// below them only stops new ones until owners shrink.
    public func setUsable(_ bytes: UInt64) {
        lock.lock()
        usable = max(1, min(bytes, maximumUsableBytes))
        lock.unlock()
    }

    public func reserve(bytes: UInt64, category: String) throws -> Reservation {
        lock.lock()
        defer { lock.unlock() }
        return try reserveLocked(bytes: bytes, category: category)
    }

    /// Prospective capacity has no buffer yet. It still limits admission, but must not hide
    /// framework/driver memory when comparing the process footprint with actual owners.
    package func reserveAllowance(bytes: UInt64, category: String) throws -> Reservation {
        lock.lock()
        defer { lock.unlock() }
        return try reserveLocked(bytes: bytes, category: category, allowance: true)
    }

    private func validateReservationLocked(bytes: UInt64, category: String) throws {
        let limit = ceiling - transientMargin
        guard reserved <= limit, bytes <= limit - reserved else {
            throw EngineError("Memory reservation denied: \(bytes) bytes for \(category)", code: "insufficient_memory")
        }
    }
    private func reserveLocked(bytes: UInt64, category: String, allowance: Bool = false) throws -> Reservation {
        try validateReservationLocked(bytes: bytes, category: category)
        let token = Reservation(id: UUID())
        entries[token.id] = Entry(bytes: bytes, category: category, allowance: allowance)
        reserved += bytes
        highWater = max(highWater, reserved)
        if allowance { allowances += bytes }
        if Self.isExpertOwner(category) { cacheOwned += bytes }
        return token
    }

    public func release(_ token: Reservation) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let entry = entries.removeValue(forKey: token.id) else {
            throw EngineError("Unknown or already released memory reservation")
        }
        reserved -= entry.bytes
        if entry.allowance { allowances -= entry.bytes }
        retireCacheBytes(entry.bytes, category: entry.category)
        lock.broadcast()
    }

    /// Return unused admission without releasing the live allocation's token.
    /// Growing a reservation is deliberately unsupported: admission precedes
    /// allocation, and no release/reacquire window may hide retained storage.
    public func shrink(_ token: Reservation, to bytes: UInt64) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let entry = entries[token.id], bytes <= entry.bytes else {
            throw EngineError("Unknown reservation or attempted reservation growth")
        }
        entries[token.id] = Entry(bytes: bytes, category: entry.category, allowance: entry.allowance)
        reserved -= entry.bytes - bytes
        if entry.allowance { allowances -= entry.bytes - bytes }
        retireCacheBytes(entry.bytes - bytes, category: entry.category)
        lock.broadcast()
    }

    private func retireCacheBytes(_ bytes: UInt64, category: String) {
        guard Self.isExpertOwner(category) else { return }
        cacheOwned -= bytes
        if reliefRequests > 0 {
            let (total, overflow) = cacheReleasedAfterRelief.addingReportingOverflow(bytes)
            cacheReleasedAfterRelief = overflow ? .max : total
        }
    }

    /// Waits at most `seconds` for `done(reserved bytes, bytes by category)`, looked at again at every release. A joined
    /// command's buffers, and the charges they carry, go when Metal (or the worker that ran the command) drops its last
    /// reference, which can trail the join by a moment and longer on a busy machine: checks wait for that here.
    package func waitForRelease(seconds: Double = 10, until done: (UInt64, [String: UInt64]) -> Bool) -> Bool {
        let deadline = Date(timeIntervalSinceNow: seconds)
        lock.lock()
        defer { lock.unlock() }
        func holds() -> Bool {
            var categories: [String: UInt64] = [:]
            for entry in entries.values { categories[entry.category, default: 0] += entry.bytes }
            return done(reserved, categories)
        }
        while !holds() { if !lock.wait(until: deadline) { return holds() } }
        return true
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        var categories: [String: UInt64] = [:]
        for entry in entries.values { categories[entry.category, default: 0] += entry.bytes }
        var result = Snapshot(
            ceiling: ceiling, transientMargin: transientMargin, reserved: reserved,
            reservationHighWater: highWater, physicalFootprint: origami_physical_footprint(),
            availableMemoryEstimate: origami_available_memory(),
            usableBytes: usable == ceiling - transientMargin ? nil : usable, categories: categories)
        result.optionalGrowthAllowed = optionalGrowthAllowed
        result.cacheRelief = cacheReliefLocked()
        result.physicalOverheadBytes = physicalOverhead
        result.prospectiveAllowanceBytes = allowances
        return result
    }

    package var physicalOverheadBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return physicalOverhead
    }

    /// A low footprint may mean that already-owned slabs are still lazy, not that process overhead retired.
    /// Raising the observation is always safe. Lowering requires a joined boundary's resident-owner proof;
    /// a partial proof bounds overhead above by footprint minus those known-resident bytes.
    package func reconcilePhysicalOverhead(
        footprint: UInt64? = nil, fullyResident: Bool = false,
        residentOwnedBytes: UInt64? = nil
    ) {
        guard budgetMode == .explicit else { return }
        lock.lock()
        defer { lock.unlock() }
        let current = footprint ?? origami_physical_footprint()
        let owned = reserved - allowances
        let observed = current > owned ? current - owned : 0
        if fullyResident || residentOwnedBytes != nil {
            let resident = fullyResident ? owned : min(owned, residentOwnedBytes!)
            let upper = current > resident ? current - resident : 0
            physicalOverhead = max(observed, min(physicalOverhead, upper))
        } else {
            physicalOverhead = max(physicalOverhead, observed)
        }
    }

    /// Check and reserve under one lock. Two background/demand allocations must not both spend the same
    /// physical headroom while their new slabs have not been faulted in. The existing final-owner token
    /// retains the complete charge, regardless of residency, until the allocation actually retires.
    package func reservePhysical(bytes: UInt64, category: String, footprint: UInt64? = nil) throws -> Reservation {
        guard let reservation = try reservePhysicalIfAvailable(bytes: bytes, category: category, footprint: footprint) else {
            throw EngineError("Physical memory admission denied: \(bytes) bytes for \(category)", code: "insufficient_memory")
        }
        return reservation
    }

    /// Optional cache growth may reuse an existing slot when physical headroom is temporarily exhausted.
    /// An impossible logical reservation is still an error, never permission to evict a valid record.
    package func reservePhysicalIfAvailable(
        bytes: UInt64, category: String,
        footprint: UInt64? = nil
    ) throws -> Reservation? {
        lock.lock()
        defer { lock.unlock() }
        try validateReservationLocked(bytes: bytes, category: category)
        if budgetMode == .explicit {
            let current = footprint ?? origami_physical_footprint()
            // Probe first: a denial must not count pages released but not yet reclaimed as lasting overhead.
            guard physicalShortfallLocked(bytes: bytes, current: current) == 0 else { return nil }
            let owned = reserved - allowances
            physicalOverhead = max(physicalOverhead, current > owned ? current - owned : 0)
        }
        return try reserveLocked(bytes: bytes, category: category)
    }

    /// Optional model caches cannot borrow a mandatory owner's admission policy: honor the moving usable budget
    /// and pressure latch as well as the hard/physical ceiling, under the same lock as reservation publication.
    package func reserveOptionalPhysicalIfAvailable(
        bytes: UInt64, category: String,
        retainingBytes: UInt64 = 0, footprint: UInt64? = nil
    ) throws -> Reservation? {
        lock.lock()
        defer { lock.unlock() }
        guard optionalGrowthAllowed, reserved <= usable, bytes <= usable - reserved,
            retainingBytes <= usable - reserved - bytes
        else { return nil }
        try validateReservationLocked(bytes: bytes, category: category)
        if budgetMode == .explicit {
            let current = footprint ?? origami_physical_footprint()
            // Unmaterialized mandatory owners (for example the two-wave expert floor at startup) must still be
            // fundable after this optional allocation. The usable check above proves the sum cannot overflow.
            guard physicalShortfallLocked(bytes: bytes + retainingBytes, current: current) == 0 else { return nil }
            let owned = reserved - allowances
            physicalOverhead = max(physicalOverhead, current > owned ? current - owned : 0)
        }
        return try reserveLocked(bytes: bytes, category: category)
    }

    /// A non-mutating planning probe. Atomic admission still has to follow it before an allocation.
    package func physicalAllocationShortfall(bytes: UInt64, footprint: UInt64? = nil) -> UInt64 {
        guard budgetMode == .explicit else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        return physicalShortfallLocked(bytes: bytes, current: footprint ?? origami_physical_footprint())
    }
    private func physicalShortfallLocked(bytes: UInt64, current: UInt64) -> UInt64 {
        let owned = reserved - allowances
        let overhead = max(physicalOverhead, current > owned ? current - owned : 0)
        let (committed, baseOverflow) = reserved.addingReportingOverflow(overhead)
        guard !baseOverflow else { return .max }
        let (next, overflow) = max(current, committed).addingReportingOverflow(bytes)
        return overflow ? .max : next > ceiling ? next - ceiling : 0
    }

    /// Counts both prompt and requested continuation before the first state allocation.
    public static func sequenceStateBytes(inputTokens: Int, outputTokens: Int) throws -> UInt64 {
        guard inputTokens >= 0, outputTokens >= 0, inputTokens <= ModelContract.maximumContext,
            outputTokens <= ModelContract.maximumContext - inputTokens
        else {
            throw EngineError("Requested token reservation exceeds checkpoint context")
        }
        return UInt64(inputTokens + outputTokens) * 25_344 + 108 * 1_024 * 1_024
    }

    /// Execution boundaries ask the governor to recheck. Explicit mode leaves pressure and
    /// swapping to its authoritative host signals, without using free memory as a proxy.
    public func checkAdmission(minimumAvailable: UInt64? = nil) {
        let available = usesAvailableMemoryFloor ? origami_available_memory() : 0
        if needsPressureCheck(available: available, footprint: origami_physical_footprint(), minimumAvailable: minimumAvailable) {
            onPressure?()
        }
    }
    package func needsPressureCheck(available: UInt64, footprint: UInt64, minimumAvailable: UInt64? = nil) -> Bool {
        (usesAvailableMemoryFloor && available < (minimumAvailable ?? availableFloorBytes)) || footprint > ceiling
    }
}

/// After a proved cache release, Metal can retire the owners before the GPU driver reclaims their pages (hundreds of
/// ms). Call only while short: waits with backoff, at most 1.023 s, until `satisfied` (re-probed after each wait);
/// cancellation is checked on both sides of each wait, before any probe or allocation. Callers still re-check and fail
/// on a persistent excess. Shared by family scratch and Qwen image admission.
package enum ReclamationWait {
    package static let delays: [Double] = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
    package static func wait(
        shouldContinue: () -> Bool, sleep: (Double) -> Void = { Thread.sleep(forTimeInterval: $0) },
        satisfied: () -> Bool
    ) throws {
        for delay in delays {
            guard shouldContinue() else { throw EngineError("Cancelled", code: "cancelled") }
            sleep(delay)
            guard shouldContinue() else { throw EngineError("Cancelled", code: "cancelled") }
            if satisfied() { return }
        }
    }
}
