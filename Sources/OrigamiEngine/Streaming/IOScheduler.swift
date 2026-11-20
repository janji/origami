import Dispatch
import Foundation

/// The drive a read touches: `internalSSD` is the model's drive (checkpoint spans, hot-expert mirror),
/// `externalSSD` the pack queue (every pack and partial copy, on one or more other drives).
public enum IOSource: String, CaseIterable, Codable, Sendable {
    case internalSSD, externalSSD
}

/// `demand`: the current layer needs the record now; `prefetch`: read-ahead or a prediction, which never
/// takes the capacity reserved for demand and can be promoted when demand adopts it.
public enum IOPriority: String, Codable, Sendable { case demand, prefetch }

/// How one scheduled operation ended; `cancelled` also covers a success that lost to cancellation.
public enum IOOutcome: Equatable, Sendable {
    case succeeded(bytes: Int)
    case cancelled
    case failed(String)
}

/// Thrown by an operation (or a join) that stopped because its ticket or request was cancelled.
public struct IOOperationCancelled: Error, Sendable { public init() {} }

/// One ticket's cancellation flag, outcome and completion wake-up, behind its own condition. Lock order:
/// the scheduler lock may be held while taking this one, never the reverse.
private final class IOTicketState: @unchecked Sendable {
    private let condition = NSCondition()
    private var cancelled = false
    private var completing = false
    private var outcome: IOOutcome?
    private var completionWake: (@Sendable () -> Void)?
    private var wakeRegistered = false

    var isCancelled: Bool {
        condition.lock()
        defer { condition.unlock() }
        return cancelled
    }

    func requestCancellation() -> Bool {
        condition.lock()
        defer { condition.unlock() }
        guard !completing, outcome == nil, !cancelled else { return false }
        cancelled = true
        return true
    }

    /// Linearization point: cancellation before this wins; later cancellation
    /// cannot revoke a successful completion already handed to its consumer.
    func beginCompletion(_ proposed: IOOutcome) -> IOOutcome {
        condition.lock()
        defer { condition.unlock() }
        precondition(!completing && outcome == nil)
        completing = true
        return cancelled ? .cancelled : proposed
    }

    func finish(_ value: IOOutcome) {
        condition.lock()
        defer { condition.unlock() }
        precondition(completing && outcome == nil)
        // The bounded wake-up retires before join can return. It may only
        // signal its owner's condition, never re-enter this ticket/loader.
        completionWake?()
        completionWake = nil
        outcome = value
        condition.broadcast()
    }

    func notifyCompletion(_ wake: @escaping @Sendable () -> Void) throws {
        condition.lock()
        defer { condition.unlock() }
        guard !wakeRegistered else { throw EngineError("I/O ticket already has a completion wake-up") }
        wakeRegistered = true
        if outcome != nil { wake() } else { completionWake = wake }
    }

    func join(until deadline: Date?) -> IOOutcome? {
        condition.lock()
        defer { condition.unlock() }
        while outcome == nil {
            if let deadline {
                if !condition.wait(until: deadline) && outcome == nil { return nil }
            } else {
                condition.wait()
            }
        }
        return outcome
    }
}

/// Operations may check cancellation between exact reads. A blocking pread is
/// allowed to finish; cancellation never makes its destination reusable early.
public struct IOCancellation: Sendable {
    private let state: IOTicketState
    fileprivate init(_ state: IOTicketState) { self.state = state }
    public var isCancelled: Bool { state.isCancelled }
    public func throwIfCancelled() throws {
        if state.isCancelled { throw IOOperationCancelled() }
    }
}

/// `decode_prerouting_read_hygiene`: a prefetch-priority (predicted) read runs
/// with this token as its thread's current read priority, so a shared reader
/// pool (SpanReadPool) can serve demand work first. Demand adopting the read
/// (IOScheduler.promote) clears it, even while the read runs, so a demanded
/// record is never deferred. Only order changes; bytes and spans do not.
package final class IOReadPriority: @unchecked Sendable {
    private static let key: pthread_key_t = {
        var key = pthread_key_t()
        precondition(pthread_key_create(&key, nil) == 0, "Read-priority thread key unavailable")
        return key
    }()
    private let lock = NSLock()
    private var deferrable = true
    private let stats: IOReadHygieneStats
    fileprivate init(stats: IOReadHygieneStats) { self.stats = stats }

    package var isDeferrable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return deferrable
    }
    fileprivate func promote() {
        lock.lock()
        deferrable = false
        lock.unlock()
    }
    /// A reader pool served a demand item ahead of this queued deferrable item.
    package func recordPoolDeferral() { stats.addPoolDeferral() }

    /// The read priority of the operation running on this thread (nil: demand, PLE or hygiene off).
    package static var current: IOReadPriority? {
        guard let pointer = pthread_getspecific(key) else { return nil }
        return Unmanaged<IOReadPriority>.fromOpaque(pointer).takeUnretainedValue()
    }
    /// Run `body` with `token` as this thread's current read priority; restores the previous one.
    package static func run<T>(_ token: IOReadPriority?, _ body: () throws -> T) rethrows -> T {
        guard let token else { return try body() }
        let previous = pthread_getspecific(key)
        pthread_setspecific(key, Unmanaged.passUnretained(token).toOpaque())
        defer { pthread_setspecific(key, previous) }
        return try withExtendedLifetime(token) { try body() }
    }
    /// Standalone checks only: a token outside a scheduler.
    package static func makeForChecks() -> (IOReadPriority, IOReadHygieneStats) {
        let stats = IOReadHygieneStats()
        return (IOReadPriority(stats: stats), stats)
    }
    package func promoteForChecks() { promote() }
}

/// Read-hygiene counters shared by the scheduler's tokens (reader threads may record).
package final class IOReadHygieneStats: @unchecked Sendable {
    private let lock = NSLock()
    private var pool: UInt64 = 0
    fileprivate init() {}
    fileprivate func addPoolDeferral() {
        lock.lock()
        pool &+= 1
        lock.unlock()
    }
    package var poolDeferrals: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return pool
    }
}

/// While any hold is live the scheduler starts no queued prefetch job (demand is
/// unaffected; running reads finish), or with `all` (`holdStarts`) no queued job at all.
/// Released once, explicitly or on deinit.
package final class IOPrefetchHold: @unchecked Sendable {
    private let scheduler: IOScheduler
    private let all: Bool
    private let lock = NSLock()
    private var released = false
    fileprivate init(_ scheduler: IOScheduler, all: Bool = false) {
        self.scheduler = scheduler
        self.all = all
    }
    package func release() {
        lock.lock()
        let first = !released
        released = true
        lock.unlock()
        if first { scheduler.releasePrefetchHold(all: all) }
    }
    deinit { release() }
}

/// Completion state lives in the ticket, so the scheduler retains no unbounded
/// history when a caller forgets to join an already completed request.
public final class IOTicket: @unchecked Sendable {
    public let id: UInt64
    public let generation: IOScheduler.Generation
    public let source: IOSource
    fileprivate let state = IOTicketState()

    fileprivate init(id: UInt64, generation: IOScheduler.Generation, source: IOSource) {
        self.id = id
        self.generation = generation
        self.source = source
    }

    /// Returns after the operation and completion callback have both stopped.
    /// Do not join a ticket from its own operation or completion callback.
    public func join() -> IOOutcome { state.join(until: nil)! }

    /// A timeout does not cancel the request or release its destination buffer.
    public func join(timeout: TimeInterval) -> IOOutcome? {
        state.join(until: Date(timeIntervalSinceNow: max(0, timeout)))
    }

    /// One bounded, signal-only observer for the prefill producer. Runs after
    /// the operation, publication callback and queue-capacity release. The
    /// observer must not re-enter the ticket, scheduler or expert loader.
    package func notifyCompletion(_ wake: @escaping @Sendable () -> Void) throws {
        try state.notifyCompletion(wake)
    }
}

/// Demand-first per-drive admission and dispatch. No hedges, speculative reads,
/// read buffers or extra copies are introduced here. Operations capture already
/// ledger-owned destinations (for example ExpertBuffer) until their callback
/// finishes. The source identifies the drive the operation actually accesses;
/// callers must resubmit on the other source when changing placement.
public final class IOScheduler: @unchecked Sendable {
    /// One request's submission epoch; only the current generation of this scheduler may submit.
    public struct Generation: Hashable, Sendable {
        fileprivate let owner: UUID
        public let value: UInt64
    }

    /// Per-drive bounds: concurrent reads, admitted jobs and admitted bytes, each with a share that
    /// only demand may use (prefetch admission stops `reserved…` short of the maximum).
    public struct Limits: Sendable {
        public let maximumConcurrent, reservedDemandConcurrent: Int
        public let maximumOutstanding, reservedDemandOutstanding: Int
        public let maximumOutstandingBytes, reservedDemandBytes: UInt64

        public init(
            maximumConcurrent: Int = 4, reservedDemandConcurrent: Int = 1,
            maximumOutstanding: Int = 64, reservedDemandOutstanding: Int = 16,
            maximumOutstandingBytes: UInt64 = 256 * 1_024 * 1_024,
            reservedDemandBytes: UInt64
        ) {
            self.maximumConcurrent = maximumConcurrent
            self.reservedDemandConcurrent = reservedDemandConcurrent
            self.maximumOutstanding = maximumOutstanding
            self.reservedDemandOutstanding = reservedDemandOutstanding
            self.maximumOutstandingBytes = maximumOutstandingBytes
            self.reservedDemandBytes = reservedDemandBytes
        }

        fileprivate func validate() throws {
            guard (1...64).contains(maximumConcurrent),
                (1...maximumConcurrent).contains(reservedDemandConcurrent),
                maximumOutstanding >= maximumConcurrent, maximumOutstanding <= 65_536,
                reservedDemandOutstanding >= reservedDemandConcurrent,
                reservedDemandOutstanding <= maximumOutstanding,
                maximumOutstandingBytes > 0, reservedDemandBytes > 0,
                reservedDemandBytes <= maximumOutstandingBytes
            else {
                throw EngineError("Invalid bounded I/O queue limits")
            }
        }
    }

    /// One drive's queues and lifetime counters at a point in time.
    public struct DriveSnapshot: Codable, Sendable {
        public let source: IOSource
        public let queuedDemand, queuedPrefetch, runningDemand, runningPrefetch, outstanding: Int
        public let outstandingBytes, successfulBytes, succeeded, failed, cancelled: UInt64
    }

    /// Both drives' snapshots under one lock, with the current generation.
    public struct Snapshot: Codable, Sendable {
        public let generation: UInt64
        public let accepting: Bool
        public let drives: [DriveSnapshot]
    }

    private final class Job {
        enum Phase { case queued, running, cancelledQueued, completing }
        let ticket: IOTicket
        let byteCount: Int
        let operation: @Sendable (IOCancellation) throws -> Void
        let completion: @Sendable (IOTicket, IOOutcome) -> Void
        var priority: IOPriority
        var phase: Phase = .queued
        var started = false
        /// Read hygiene only: the prefetch job's reader-pool priority (immutable reference) and
        /// whether a prefetch hold has already deferred its start once (counted once).
        let token: IOReadPriority?
        var deferred = false

        init(
            ticket: IOTicket, byteCount: Int, priority: IOPriority, token: IOReadPriority? = nil,
            operation: @escaping @Sendable (IOCancellation) throws -> Void,
            completion: @escaping @Sendable (IOTicket, IOOutcome) -> Void
        ) {
            self.ticket = ticket
            self.byteCount = byteCount
            self.priority = priority
            self.token = token
            self.operation = operation
            self.completion = completion
        }
    }

    private final class Drive {
        let limits: Limits
        var demand: [Job] = [], prefetch: [Job] = []
        var runningDemand = 0, runningPrefetch = 0, outstanding = 0
        var outstandingBytes: UInt64 = 0, successfulBytes: UInt64 = 0
        var succeeded: UInt64 = 0, failed: UInt64 = 0, cancelled: UInt64 = 0
        init(_ limits: Limits) { self.limits = limits }
    }

    private let owner = UUID()
    private let lock = NSLock()
    private let workers = DispatchQueue(
        label: "origami.io.workers", qos: .userInitiated,
        attributes: .concurrent, autoreleaseFrequency: .workItem)
    private let ledger: MemoryLedger
    private let metadataReservation: MemoryLedger.Reservation
    private var drives: [IOSource: Drive]
    private var jobs: [UInt64: Job] = [:]
    private var sequence: UInt64 = 0
    private var nextID: UInt64 = 0
    private var accepting = true
    /// `decode_prerouting_read_hygiene` (nil when off): token statistics, live prefetch holds,
    /// prefetch jobs whose start a hold deferred, and queued prefetch jobs cancelled before start.
    private let hygiene: IOReadHygieneStats?
    private var prefetchHolds = 0
    /// Checks only (`holdStarts`): live holds that stop every queued job, demand included, from starting.
    private var startHolds = 0
    private var holdDeferrals: UInt64 = 0, cancelledQueued: UInt64 = 0
    /// Told under the lock as a job starts (nil) and as a started job ends (its successful bytes, else 0); must not call back.
    private var runObserver: (@Sendable (IOSource, Int?) -> Void)?
    package func observeRuns(_ observer: @escaping @Sendable (IOSource, Int?) -> Void) { lock.withLock { runObserver = observer } }

    /// Metadata uses a conservative 2 KiB allowance per admitted job. Destination
    /// allocations and captured model state remain owned by their original ledger
    /// reservations and are not double-counted as I/O buffers. `readHygiene` adds
    /// reader-pool priority tokens, prefetch holds and queued-only cancellation;
    /// off, admission and dispatch are unchanged.
    public init(
        ledger: MemoryLedger, internalLimits: Limits, externalLimits: Limits,
        readHygiene: Bool = false
    ) throws {
        try internalLimits.validate()
        try externalLimits.validate()
        let count = UInt64(internalLimits.maximumOutstanding + externalLimits.maximumOutstanding)
        metadataReservation = try ledger.reserve(bytes: count * 2_048, category: "io-scheduler-metadata")
        self.ledger = ledger
        drives = [.internalSSD: Drive(internalLimits), .externalSSD: Drive(externalLimits)]
        hygiene = readHygiene ? IOReadHygieneStats() : nil
    }

    /// Read-hygiene path counters (see `hygieneCounts`).
    package struct HygieneCounts: Sendable {
        /// Reader-pool picks that served a demand item ahead of a queued predicted item.
        package let poolDeferrals: UInt64
        /// Predicted (prefetch) jobs whose start waited at least once for a prefetch hold (decode PLE rows).
        package let holdDeferrals: UInt64
        /// Predicted jobs cancelled while still queued (never started).
        package let cancelledQueued: UInt64
        package let liveHolds: Int
    }
    /// Nil when read hygiene is off.
    package func hygieneCounts() -> HygieneCounts? {
        lock.lock()
        defer { lock.unlock() }
        guard let hygiene else { return nil }
        return HygieneCounts(
            poolDeferrals: hygiene.poolDeferrals, holdDeferrals: holdDeferrals,
            cancelledQueued: cancelledQueued, liveHolds: prefetchHolds)
    }

    /// Read hygiene: no queued prefetch job starts until the hold is released (nil when off).
    package func holdPrefetch() -> IOPrefetchHold? {
        lock.lock()
        defer { lock.unlock() }
        guard hygiene != nil else { return nil }
        prefetchHolds += 1
        return IOPrefetchHold(self)
    }
    /// Checks only: no queued job starts, demand or prefetch, until the hold is released (a suspended reader, so a
    /// check can hold a promoted prediction queued). Running reads finish; cancellation still completes at once.
    package func holdStarts() -> IOPrefetchHold {
        lock.lock()
        defer { lock.unlock() }
        startHolds += 1
        return IOPrefetchHold(self, all: true)
    }
    fileprivate func releasePrefetchHold(all: Bool) {
        lock.lock()
        let next: [Job]
        if all {
            precondition(startHolds > 0, "Start hold released twice")
            startHolds -= 1
            next = pumpLocked()
        } else {
            precondition(prefetchHolds > 0, "Prefetch hold released twice")
            prefetchHolds -= 1
            next = prefetchHolds == 0 ? pumpLocked() : []
        }
        lock.unlock()
        launch(next)
    }

    /// Read hygiene: cancel a prefetch job only while it is still queued (never
    /// started, never promoted to demand). Its completion runs as `.cancelled`
    /// (the caller's ordinary producer cleanup releases the cache slot and the
    /// outstanding admission); join the ticket to observe that. A started read
    /// is never interrupted: false. `anyPriority` (a replica won the race, read
    /// hygiene on or off): the losing primary, demand or prefetch, the same way.
    @discardableResult package func cancelQueued(_ ticket: IOTicket, anyPriority: Bool = false) -> Bool {
        lock.lock()
        guard ticket.generation.owner == owner, let job = jobs[ticket.id], job.ticket === ticket, job.phase == .queued,
            anyPriority || hygiene != nil && job.priority == .prefetch, ticket.state.requestCancellation()
        else {
            lock.unlock()
            return false
        }
        let removed = removeCancelledQueuedLocked(job)
        precondition(removed, "Queued job left its queue")
        if !anyPriority { cancelledQueued &+= 1 }
        let next = pumpLocked()
        lock.unlock()
        launchCancelled([job])
        launch(next)
        return true
    }

    deinit { try? ledger.release(metadataReservation) }

    public var currentGeneration: Generation {
        lock.lock()
        defer { lock.unlock() }
        return Generation(owner: owner, value: sequence)
    }

    /// Start the next request and cancel every older request. Old destinations
    /// remain retained until their tickets join; old generation handles reject
    /// all future submissions. Single-sequence serving needs no generation map.
    public func beginGeneration() throws -> Generation {
        lock.lock()
        guard accepting, sequence < UInt64.max else {
            lock.unlock()
            throw EngineError("I/O scheduler closed or generation exhausted")
        }
        sequence += 1
        let cancelled = cancelLocked(Array(jobs.values))
        let next = pumpLocked()
        let generation = Generation(owner: owner, value: sequence)
        lock.unlock()
        launchCancelled(cancelled)
        launch(next)
        return generation
    }

    @discardableResult
    public func submit(
        source: IOSource, priority: IOPriority, generation: Generation,
        byteCount: Int,
        operation: @escaping @Sendable (IOCancellation) throws -> Void,
        completion: @escaping @Sendable (IOTicket, IOOutcome) -> Void = { _, _ in }
    ) throws -> IOTicket {
        lock.lock()
        do {
            try validateSubmissionLocked(generation: generation, byteCount: byteCount)
            guard canAdmitLocked(source: source, priority: priority, byteCount: byteCount) else {
                throw EngineError("I/O queue admission denied; demand capacity is reserved")
            }
            let (ticket, ready) = enqueueLocked(
                source: source, priority: priority, generation: generation,
                byteCount: byteCount, operation: operation, completion: completion)
            lock.unlock()
            launch(ready)
            return ticket
        } catch {
            lock.unlock()
            throw error
        }
    }

    /// A drive's expected seconds to finish its queue with one more read: (source, outstanding bytes and jobs
    /// including the new read, the drive's concurrent-read bound).
    public typealias Estimate = (
        _ source: IOSource, _ bytesIncludingNew: UInt64, _ jobsIncludingNew: Int, _ maximumConcurrent: Int
    ) -> Double

    /// Choose and charge one actual source atomically. No destination, ticket,
    /// speculative read, or capacity claim is created for the losing source.
    /// Availability reflects PackStore's currently verified exact sources; loss
    /// after selection still uses the loader's ordinary joined fallback.
    /// `estimate` runs under the scheduler lock and must not call back into it; `ties`: the source an exact tie goes to.
    @discardableResult
    public func submitEarliest(
        availableSources: [IOSource], estimate: Estimate, ties: IOSource = .externalSSD, idleProbeSource: IOSource? = nil,
        priority: IOPriority, generation: Generation, byteCount: Int,
        operation: @escaping @Sendable (IOSource, IOCancellation) throws -> Void,
        completion: @escaping @Sendable (IOTicket, IOOutcome) -> Void = { _, _ in }
    ) throws -> IOTicket {
        try validateSources(availableSources)
        lock.lock()
        do {
            try validateSubmissionLocked(generation: generation, byteCount: byteCount)
            guard
                let source = earliestLocked(
                    availableSources, estimate: estimate, ties: ties, idleProbeSource: idleProbeSource, priority: priority,
                    byteCount: byteCount, pendingBytes: [:], pendingJobs: [:])
            else { throw EngineError("I/O queue admission denied for every available exact source") }
            let (ticket, ready) = enqueueLocked(
                source: source, priority: priority, generation: generation,
                byteCount: byteCount, operation: { cancellation in try operation(source, cancellation) }, completion: completion)
            lock.unlock()
            launch(ready)
            return ticket
        } catch {
            lock.unlock()
            throw error
        }
    }

    /// `submitEarliest`'s choice without submitting: nil when no available source would admit the read. `pendingBytes`
    /// and `pendingJobs`: reads a caller has placed on a source but not submitted yet (a batch being planned), counted
    /// as if outstanding there.
    package func earliestSource(
        availableSources: [IOSource], estimate: Estimate, ties: IOSource = .externalSSD, idleProbeSource: IOSource? = nil,
        priority: IOPriority, byteCount: Int, pendingBytes: [IOSource: UInt64] = [:], pendingJobs: [IOSource: Int] = [:]
    ) throws -> IOSource? {
        try validateSources(availableSources)
        lock.lock()
        defer { lock.unlock() }
        return earliestLocked(
            availableSources, estimate: estimate, ties: ties, idleProbeSource: idleProbeSource, priority: priority,
            byteCount: byteCount, pendingBytes: pendingBytes, pendingJobs: pendingJobs)
    }

    /// The source choice for any number of sources holding the same bytes: the index of the smallest estimated finish
    /// time (nil: that source is unavailable or would not admit the read), the first on an exact tie; nil when none is
    /// finite. The scheduler's two queues and the copies on several drives (`HotCopySet.pick`) choose by it.
    package static func earliest(_ seconds: [Double?]) -> Int? {
        var selected: Int?
        var best = Double.infinity
        for (index, value) in seconds.enumerated() {
            if let value, value < best {
                selected = index
                best = value
            }
        }
        return selected
    }

    private func validateSources(_ available: [IOSource]) throws {
        guard !available.isEmpty, available.count <= 2, Set(available).count == available.count else {
            throw EngineError("Invalid exact-source availability")
        }
    }

    /// The admissible available source with the smallest estimate (exact ties to `ties`); an idle admissible
    /// `idleProbeSource` wins outright.
    private func earliestLocked(
        _ available: [IOSource], estimate: Estimate, ties: IOSource, idleProbeSource: IOSource?, priority: IOPriority,
        byteCount: Int, pendingBytes: [IOSource: UInt64], pendingJobs: [IOSource: Int]
    ) -> IOSource? {
        func admits(_ source: IOSource) -> Bool {
            canAdmitLocked(
                source: source, priority: priority, byteCount: byteCount,
                pendingBytes: pendingBytes[source] ?? 0, pendingJobs: pendingJobs[source] ?? 0)
        }
        // Stable order: the first source wins an exact tie.
        let order: [IOSource] = [ties, ties == .externalSSD ? .internalSSD : .externalSSD]
        var selected = Self.earliest(
            order.map { source -> Double? in
                guard available.contains(source), admits(source) else { return nil }
                let drive = drives[source]!
                return estimate(
                    source, drive.outstandingBytes + (pendingBytes[source] ?? 0) + UInt64(byteCount),
                    drive.outstanding + (pendingJobs[source] ?? 0) + 1, drive.limits.maximumConcurrent)
            }
        ).map { order[$0] }
        if let probe = idleProbeSource, available.contains(probe), drives[probe]!.outstanding + (pendingJobs[probe] ?? 0) == 0,
            admits(probe)
        {
            selected = probe
        }
        return selected
    }

    /// Reads currently executing on one drive (demand and prefetch), including
    /// the caller's own when called from inside an operation.
    package func runningCount(_ source: IOSource) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let drive = drives[source]!
        return drive.runningDemand + drive.runningPrefetch
    }

    private func validateSubmissionLocked(generation: Generation, byteCount: Int) throws {
        guard accepting, generation.owner == owner, generation.value == sequence,
            byteCount > 0, nextID < UInt64.max
        else {
            throw EngineError("I/O submission is closed, stale, or invalid")
        }
    }

    private func canAdmitLocked(
        source: IOSource, priority: IOPriority, byteCount: Int, pendingBytes: UInt64 = 0, pendingJobs: Int = 0
    ) -> Bool {
        let drive = drives[source]!
        let limits = drive.limits
        let outstanding = drive.outstanding + pendingJobs
        let outstandingBytes = drive.outstandingBytes + pendingBytes
        let countLimit =
            priority == .demand
            ? limits.maximumOutstanding
            : limits.maximumOutstanding - limits.reservedDemandOutstanding
        let byteLimit =
            priority == .demand
            ? limits.maximumOutstandingBytes
            : limits.maximumOutstandingBytes - limits.reservedDemandBytes
        return outstanding < countLimit && outstandingBytes <= byteLimit
            && UInt64(byteCount) <= byteLimit - outstandingBytes
            && (priority == .demand || limits.maximumConcurrent > limits.reservedDemandConcurrent)
    }

    private func enqueueLocked(
        source: IOSource, priority: IOPriority, generation: Generation, byteCount: Int,
        operation: @escaping @Sendable (IOCancellation) throws -> Void,
        completion: @escaping @Sendable (IOTicket, IOOutcome) -> Void
    ) -> (IOTicket, [Job]) {
        nextID += 1
        let ticket = IOTicket(id: nextID, generation: generation, source: source)
        let token = hygiene != nil && priority == .prefetch ? IOReadPriority(stats: hygiene!) : nil
        let job = Job(
            ticket: ticket, byteCount: byteCount, priority: priority, token: token,
            operation: operation, completion: completion)
        jobs[ticket.id] = job
        let drive = drives[source]!
        drive.outstanding += 1
        drive.outstandingBytes += UInt64(byteCount)
        if priority == .demand { drive.demand.append(job) } else { drive.prefetch.append(job) }
        return (ticket, pumpLocked())
    }

    /// A prefetched expert that becomes demand can use reserved demand capacity
    /// immediately. An already executing operation needs no queue promotion.
    @discardableResult public func promote(_ ticket: IOTicket) -> Bool {
        lock.lock()
        guard ticket.generation.owner == owner, let job = jobs[ticket.id], job.ticket === ticket else {
            lock.unlock()
            return false
        }
        // Read hygiene: a demanded read is never deferred by a reader pool again, queued or running.
        job.token?.promote()
        guard job.phase == .queued, job.priority == .prefetch else {
            lock.unlock()
            return false
        }
        let drive = drives[ticket.source]!
        drive.prefetch.removeAll { $0 === job }
        job.priority = .demand
        drive.demand.append(job)
        let ready = pumpLocked()
        lock.unlock()
        launch(ready)
        return true
    }

    /// Returns false if the ticket belongs elsewhere or its completion has
    /// already committed. Cancellation itself never implies that writers joined.
    @discardableResult public func cancel(_ ticket: IOTicket) -> Bool {
        lock.lock()
        guard ticket.generation.owner == owner, let job = jobs[ticket.id], job.ticket === ticket,
            job.phase != .completing, ticket.state.requestCancellation()
        else {
            lock.unlock()
            return false
        }
        let removed = removeCancelledQueuedLocked(job)
        let next = pumpLocked()
        lock.unlock()
        launchCancelled(removed ? [job] : [])
        launch(next)
        return true
    }

    /// Invalidates a current generation before cancelling it, so a late producer
    /// cannot submit more work into the generation being drained.
    /// With `deadline`, a job still running then is left running and has no outcome here (a drive that stopped answering).
    public func cancelAndJoin(_ generation: Generation, until deadline: Date? = nil) throws -> [IOOutcome] {
        lock.lock()
        guard generation.owner == owner else {
            lock.unlock()
            throw EngineError("Foreign I/O generation")
        }
        if generation.value == sequence {
            guard sequence < UInt64.max else {
                lock.unlock()
                throw EngineError("I/O generation exhausted")
            }
            sequence += 1
        }
        let selected = jobs.values.filter { $0.ticket.generation == generation }
        let cancelled = cancelLocked(selected)
        let next = pumpLocked()
        lock.unlock()
        launchCancelled(cancelled)
        launch(next)
        return selected.compactMap { deadline == nil ? $0.ticket.join() : $0.ticket.join(timeout: deadline!.timeIntervalSinceNow) }
    }

    /// Explicit shutdown joins both active preads and completion callbacks.
    /// Do not call from one of this scheduler's own operations or callbacks.
    public func shutdown() {
        lock.lock()
        accepting = false
        let selected = Array(jobs.values)
        let cancelled = cancelLocked(selected)
        lock.unlock()
        launchCancelled(cancelled)
        for job in selected { _ = job.ticket.join() }
    }

    private func cancelLocked(_ selected: [Job]) -> [Job] {
        var queued: [Job] = []
        for job in selected where job.phase != .completing {
            if job.ticket.state.requestCancellation(), removeCancelledQueuedLocked(job) { queued.append(job) }
        }
        return queued
    }

    private func removeCancelledQueuedLocked(_ job: Job) -> Bool {
        guard job.phase == .queued else { return false }
        let drive = drives[job.ticket.source]!
        drive.demand.removeAll { $0 === job }
        drive.prefetch.removeAll { $0 === job }
        job.phase = .cancelledQueued
        return true
    }

    private func pumpLocked() -> [Job] {
        var ready: [Job] = []
        guard accepting, startHolds == 0 else { return ready }
        for source in IOSource.allCases {
            let drive = drives[source]!
            let limits = drive.limits
            while drive.runningDemand + drive.runningPrefetch < limits.maximumConcurrent {
                let job: Job
                if !drive.demand.isEmpty {
                    job = drive.demand.removeFirst()
                    drive.runningDemand += 1
                } else if !drive.prefetch.isEmpty,
                    drive.runningPrefetch < limits.maximumConcurrent - limits.reservedDemandConcurrent
                {
                    // Read hygiene: a live hold (decode PLE rows in flight) defers every prefetch start.
                    if prefetchHolds > 0 {
                        for queued in drive.prefetch where !queued.deferred {
                            queued.deferred = true
                            holdDeferrals &+= 1
                        }
                        break
                    }
                    job = drive.prefetch.removeFirst()
                    drive.runningPrefetch += 1
                } else {
                    break
                }
                job.phase = .running
                job.started = true
                runObserver?(source, nil)
                ready.append(job)
            }
        }
        return ready
    }

    private func launch(_ ready: [Job]) {
        for job in ready {
            let box = JobBox(job)
            workers.async { [self, box] in
                let job = box.job
                let result: IOOutcome
                do {
                    let cancellation = IOCancellation(job.ticket.state)
                    try cancellation.throwIfCancelled()
                    try IOReadPriority.run(job.token) { try job.operation(cancellation) }
                    result = .succeeded(bytes: job.byteCount)
                } catch is IOOperationCancelled { result = .cancelled } catch { result = .failed(String(describing: error)) }
                finish(job, proposed: result)
            }
        }
    }

    // All mutable Job fields are scheduler-lock-owned. Workers access only the
    // immutable ticket, operation, byte count and read-priority token before
    // returning under that lock.
    private final class JobBox: @unchecked Sendable {
        let job: Job
        init(_ job: Job) { self.job = job }
    }

    private func launchCancelled(_ queued: [Job]) {
        for job in queued {
            let box = JobBox(job)
            workers.async { [self, box] in finish(box.job, proposed: .cancelled) }
        }
    }

    private func finish(_ job: Job, proposed: IOOutcome) {
        lock.lock()
        precondition(jobs[job.ticket.id] === job && job.phase != .completing)
        job.phase = .completing
        let outcome = job.ticket.state.beginCompletion(proposed)
        lock.unlock()
        // The producer is finished before this callback may publish or free its
        // cache destination. Admission remains charged until the callback ends.
        job.completion(job.ticket, outcome)
        lock.lock()
        let drive = drives[job.ticket.source]!
        jobs.removeValue(forKey: job.ticket.id)
        drive.outstanding -= 1
        drive.outstandingBytes -= UInt64(job.byteCount)
        if job.started {
            if job.priority == .demand { drive.runningDemand -= 1 } else { drive.runningPrefetch -= 1 }
            if case .succeeded(let bytes) = outcome { runObserver?(job.ticket.source, bytes) } else { runObserver?(job.ticket.source, 0) }
        }
        switch outcome {
        case .succeeded(let bytes):
            drive.succeeded &+= 1
            drive.successfulBytes &+= UInt64(bytes)
        case .cancelled: drive.cancelled &+= 1
        case .failed: drive.failed &+= 1
        }
        let next = pumpLocked()
        lock.unlock()
        job.ticket.state.finish(outcome)
        launch(next)
    }

    /// Reads finished on either drive so far (succeeded, failed or cancelled): a slot wait's sign of progress.
    package var finishedReads: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return drives.values.reduce(0) { $0 &+ $1.succeeded &+ $1.failed &+ $1.cancelled }
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            generation: sequence, accepting: accepting,
            drives: IOSource.allCases.map { source in
                let drive = drives[source]!
                return DriveSnapshot(
                    source: source, queuedDemand: drive.demand.count,
                    queuedPrefetch: drive.prefetch.count, runningDemand: drive.runningDemand,
                    runningPrefetch: drive.runningPrefetch, outstanding: drive.outstanding,
                    outstandingBytes: drive.outstandingBytes, successfulBytes: drive.successfulBytes,
                    succeeded: drive.succeeded, failed: drive.failed, cancelled: drive.cancelled)
            })
    }
}
