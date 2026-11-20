import Dispatch
import Foundation
import Metal
import ObjectiveC

/// Replica-arena occupancy and race counters for the experts snapshot (a replica winner is consumed in
/// place, never copied into the cache slot).
public struct ExpertReplicaSnapshot: Codable, Sendable {
    public let arenaBytes: UInt64
    public let leasedRecords, highWaterRecords: Int
    public let primaryRecords, replicaRecords, internalWinners, externalWinners: UInt64
    public let replicaWinners: UInt64?
    public let consumerPins: Int?
}

/// Associated with the arena's MTLBuffer so the ledger charge is released only when the last
/// reference to the buffer (arena or an escaped GPU user) goes away.
private final class ReplicaAllocationOwner {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, reservation: MemoryLedger.Reservation) {
        self.ledger = ledger
        self.reservation = reservation
    }
    deinit { try? ledger.release(reservation) }
}
private enum ReplicaAllocationAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Staging for hedge reads: one shared MTLBuffer of `capacity` records of up to `recordStride` bytes each (a 16 KiB
/// multiple), zeroed once. A replica read lands here, never in the cache slot the primary may still be writing.
/// The Metal resource carries the ledger charge even if it escapes the arena. Records stay leased while requests,
/// writers or explicitly pinned GPU consumers live. A `lazily` made arena charges and allocates nothing until its
/// first `acquire` (which returns nil if the ledger refuses it).
package final class ExpertReplicaArena: @unchecked Sendable {
    /// Records, the bytes from one record to the next, and the bytes charged to the ledger once allocated.
    package let capacity: Int, recordStride: Int
    package let chargedBytes: UInt64
    private let ledger: MemoryLedger
    private let device: MTLDevice
    private var buffer: MTLBuffer?
    /// Nil until a lazy arena is allocated.
    package var metalBuffer: MTLBuffer? {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }
    private let lock = NSLock()
    private var free: [Int]
    private var consumers: [Int: (lease: ExpertReplicaLease, count: Int)] = [:]
    private var highWater = 0
    private var primaries: UInt64 = 0, replicas: UInt64 = 0
    private var internalWins: UInt64 = 0, externalWins: UInt64 = 0, replicaWins: UInt64 = 0

    /// The bytes an arena of `records` records of up to `recordBytes` bytes charges.
    package static func chargedBytes(records: Int, recordBytes: Int) -> UInt64 {
        UInt64(records * ((recordBytes + 16_383) / 16_384 * 16_384) + 65_536)
    }

    package init(ledger: MemoryLedger, device: MTLDevice, records: Int, recordBytes: Int, lazily: Bool = false) throws {
        capacity = records
        recordStride = (recordBytes + 16_383) / 16_384 * 16_384
        chargedBytes = Self.chargedBytes(records: records, recordBytes: recordBytes)
        self.ledger = ledger
        self.device = device
        free = Array(0..<records)
        if !lazily { buffer = try allocate() }
    }

    private func allocate() throws -> MTLBuffer {
        let reservation = try ledger.reserve(bytes: chargedBytes, category: "expert-replica-staging")
        guard let buffer = device.makeBuffer(length: capacity * recordStride, options: .storageModeShared) else {
            try ledger.release(reservation)
            throw EngineError("Cannot allocate expert replica arena")
        }
        objc_setAssociatedObject(
            buffer, &ReplicaAllocationAssociation.key,
            ReplicaAllocationOwner(ledger: ledger, reservation: reservation), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        return buffer
    }

    /// A free record for `bytes` (at most `recordStride`), nil when every record is leased (or a lazy arena cannot be
    /// allocated).
    package func acquire(bytes: Int) -> ExpertReplicaLease? {
        precondition(bytes > 0 && bytes <= recordStride, "Replica record larger than the arena's stride")
        lock.lock()
        defer { lock.unlock() }
        if buffer == nil { buffer = try? allocate() }
        guard let buffer, let index = free.popLast() else { return nil }
        highWater = max(highWater, capacity - free.count)
        return ExpertReplicaLease(arena: self, buffer: buffer, offset: index * recordStride, index: index, length: bytes)
    }
    package func didSubmit(replica: Bool) {
        lock.lock()
        defer { lock.unlock() }
        if replica { replicas &+= 1 } else { primaries &+= 1 }
    }
    fileprivate func release(_ index: Int) {
        lock.lock()
        defer { lock.unlock() }
        precondition(!free.contains(index) && consumers[index] == nil)
        free.append(index)
    }
    fileprivate func pin(_ lease: ExpertReplicaLease) {
        lock.lock()
        defer { lock.unlock() }
        precondition(!free.contains(lease.index))
        consumers[lease.index] = (lease, (consumers[lease.index]?.count ?? 0) + 1)
    }
    fileprivate func unpin(_ lease: ExpertReplicaLease) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let current = consumers[lease.index], current.lease === lease else {
            throw EngineError("Replica consumer has no live GPU pin")
        }
        if current.count == 1 { consumers.removeValue(forKey: lease.index) } else { consumers[lease.index] = (lease, current.count - 1) }
    }
    package func didPublish(_ source: IOSource, replica: Bool) {
        lock.lock()
        defer { lock.unlock() }
        if source == .internalSSD { internalWins &+= 1 } else { externalWins &+= 1 }
        if replica { replicaWins &+= 1 }
    }
    package func snapshot() -> ExpertReplicaSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return .init(
            arenaBytes: buffer == nil ? 0 : chargedBytes, leasedRecords: capacity - free.count,
            highWaterRecords: highWater, primaryRecords: primaries, replicaRecords: replicas,
            internalWinners: internalWins, externalWinners: externalWins, replicaWinners: replicaWins,
            consumerPins: consumers.values.reduce(0) { $0 + $1.count })
    }
}

/// One arena record of `length` bytes at `offset` in `metalBuffer`, owned until the lease is dropped and no consumer
/// pin remains.
package final class ExpertReplicaLease: @unchecked Sendable {
    private let arena: ExpertReplicaArena
    fileprivate let index: Int
    package let metalBuffer: MTLBuffer
    package let offset: Int, length: Int
    package var bytes: UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: metalBuffer.contents() + offset, count: length)
    }
    fileprivate init(arena: ExpertReplicaArena, buffer: MTLBuffer, offset: Int, index: Int, length: Int) {
        self.arena = arena
        metalBuffer = buffer
        self.offset = offset
        self.index = index
        self.length = length
    }
    package func pinConsumer() { arena.pin(self) }
    package func retireConsumer() throws { try arena.unpin(self) }
    // A live consumer pin retains this lease in the arena. Dropping a caller's
    // unretired LoadedExpert cannot silently recycle GPU-visible data.
    deinit { arena.release(index) }
}

/// A race between one record's primary read and, once the primary failed or is older than its hedge delay, a replica
/// read of the same bytes from the other drive into an arena lease; the first success publishes. The primary is the
/// sole writer of its cache slot: a replica winner supplies its own lease and never writes into that slot. A replica win
/// cancels a still-queued primary; a started one finishes, and its owner joins it before the slot is reused.
/// The primary is one read `start` submits, or `parts` reads its owner submits (`expect`, `addPrimary`, `finishPart`).
package final class ExpertReadReplica: @unchecked Sendable {
    private let condition = NSCondition()
    private let scheduler: IOScheduler
    private let arena: ExpertReplicaArena
    private let generation: IOScheduler.Generation
    private let recordBytes: Int
    /// Reads the record from a drive into `recordBytes` bytes.
    private let read: @Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void
    private let completePrimary: @Sendable (IOOutcome) -> IOOutcome
    private let releaseEntry: @Sendable () -> Void
    /// Called once a replica's outcome is recorded (a waiter's wake-up); must not re-enter the race.
    private let wake: (@Sendable () -> Void)?
    private var delay = 0.0
    private let created = DispatchTime.now().uptimeNanoseconds
    private var primaryTickets: [IOTicket] = []
    private var replicaTicket: IOTicket?
    private var partsLeft = 1
    private var partsOutcome: IOOutcome?
    private var primarySource: IOSource?
    private var alternateSource: IOSource?
    private var primaryOutcome: IOOutcome?
    private var replicaOutcome: IOOutcome?
    private var replicaSubmitted = false
    private var cancelled = false
    private var published: IOOutcome?
    private var replicaWinner: ExpertReplicaLease?
    private var droppedPrimary = false
    private var entryReleased = false
    /// A pre-routed (speculative) primary gets no hedge replica until the
    /// authoritative demand shares its entry; the read itself is unchanged.
    private var speculative = false

    /// `read`: the record from a drive (the primary's from `start`, and the replica's); `completePrimary` runs on the
    /// primary's completion before the race records it (its cache publication); `releaseEntry` once the primary ended
    /// and an outcome is published.
    package init(
        scheduler: IOScheduler, arena: ExpertReplicaArena, generation: IOScheduler.Generation, recordBytes: Int,
        read: @escaping @Sendable (IOSource, UnsafeMutableRawBufferPointer) throws -> Void,
        completePrimary: @escaping @Sendable (IOOutcome) -> IOOutcome = { $0 },
        releaseEntry: @escaping @Sendable () -> Void = {}, wake: (@Sendable () -> Void)? = nil
    ) {
        self.scheduler = scheduler
        self.arena = arena
        self.generation = generation
        self.recordBytes = recordBytes
        self.read = read
        self.completePrimary = completePrimary
        self.releaseEntry = releaseEntry
        self.wake = wake
    }

    package var isCancelled: Bool {
        condition.lock()
        defer { condition.unlock() }
        return cancelled
    }
    package var winningReplica: ExpertReplicaLease? {
        condition.lock()
        defer { condition.unlock() }
        return cancelled ? nil : replicaWinner
    }
    /// Whether every primary read ended: none writes into its slot any more.
    package var primaryFinished: Bool {
        condition.lock()
        defer { condition.unlock() }
        return primaryOutcome != nil
    }
    /// Whether a replica win cancelled every primary read before it began: none ever wrote into its slot.
    package var primaryDropped: Bool {
        condition.lock()
        defer { condition.unlock() }
        return droppedPrimary
    }
    /// Whether the primary ended with the whole record read.
    package var primarySucceeded: Bool {
        condition.lock()
        defer { condition.unlock() }
        if case .succeeded? = primaryOutcome { return true }
        return false
    }
    /// Whether the replica has ended in a failure.
    package var replicaFailed: Bool {
        condition.lock()
        defer { condition.unlock() }
        if case .failed? = replicaOutcome { return true }
        return false
    }
    /// When `tryReplica` submits the replica (uptime nanoseconds; at once after a failed primary), nil if it never will.
    package var hedgeDeadline: UInt64? {
        condition.lock()
        defer { condition.unlock() }
        guard !cancelled, !speculative, published == nil, !replicaSubmitted, alternateSource != nil else { return nil }
        return primaryOutcome != nil ? created : created + UInt64(max(0, delay) * 1e9)
    }

    /// Submits the primary (`submit` chooses its drive) into `destination`: the other of `available` is the replica's,
    /// after `delay(primary's drive)` seconds.
    package func start(
        destination: UnsafeMutableRawBufferPointer, available: [IOSource], delay: (IOSource) -> Double,
        submit: (
            _ operation: @escaping @Sendable (IOSource, IOCancellation) throws -> Void,
            _ completion: @escaping @Sendable (IOTicket, IOOutcome) -> Void
        ) throws -> IOTicket
    ) throws -> IOTicket {
        condition.lock()
        defer { condition.unlock() }
        precondition(primaryTickets.isEmpty)
        nonisolated(unsafe) let target = destination
        let operation: @Sendable (IOSource, IOCancellation) throws -> Void = { [read] source, cancellation in
            try cancellation.throwIfCancelled()
            try read(source, target)
            try cancellation.throwIfCancelled()
        }
        let ticket = try submit(operation) { [self] ticket, outcome in finishPrimary(ticket.source, outcome: outcome) }
        primaryTickets.append(ticket)
        let other: IOSource = ticket.source == .externalSSD ? .internalSSD : .externalSSD
        expectLocked(parts: 1, source: ticket.source, alternate: available.contains(other) ? other : nil, delay: delay(ticket.source))
        return ticket
    }

    /// A primary of `parts` reads on `source` that its owner submits: each ticket goes to `addPrimary` (all before the
    /// first `tryReplica`) and each read's completion calls `finishPart` after its last write. The replica reads from
    /// `alternate` (nil: never) after `delay` seconds.
    package func expect(parts: Int, source: IOSource, alternate: IOSource?, delay: Double) {
        condition.lock()
        defer { condition.unlock() }
        expectLocked(parts: parts, source: source, alternate: alternate, delay: delay)
    }
    private func expectLocked(parts: Int, source: IOSource, alternate: IOSource?, delay: Double) {
        partsLeft = parts
        primarySource = source
        alternateSource = alternate
        self.delay = delay
        arena.didSubmit(replica: false)
    }
    package func addPrimary(_ ticket: IOTicket) {
        condition.lock()
        primaryTickets.append(ticket)
        condition.unlock()
    }
    /// One primary read ended (`failed` also when its operation reported an error): the primary ends with the last one,
    /// failed if any failed, else cancelled if any was.
    package func finishPart(_ outcome: IOOutcome) {
        func rank(_ outcome: IOOutcome) -> Int {
            switch outcome {
            case .succeeded: 0
            case .cancelled: 1
            case .failed: 2
            }
        }
        condition.lock()
        partsLeft -= 1
        if partsOutcome.map({ rank($0) < rank(outcome) }) ?? true { partsOutcome = outcome }
        let ended = partsLeft == 0 ? partsOutcome : nil
        let source = primarySource
        condition.unlock()
        guard let ended, let source else { return }
        if case .succeeded = ended {
            finishPrimary(source, outcome: .succeeded(bytes: recordBytes))
        } else {
            finishPrimary(source, outcome: ended)
        }
    }

    /// Runs `body` (a write into the primary's own destination) only while the record is still the primary's: no read
    /// has published it and the race is not cancelled. Under the lock a replica's win is published under, so a win and
    /// such a write never overlap; false (nothing ran) once the primary lost the record.
    package func whileOwned(_ body: () -> Void) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        guard !cancelled, published == nil else { return false }
        body()
        return true
    }

    package func markSpeculative() {
        condition.lock()
        speculative = true
        condition.unlock()
    }
    package func markDemanded() {
        condition.lock()
        speculative = false
        condition.unlock()
    }
    package var isSpeculative: Bool {
        condition.lock()
        defer { condition.unlock() }
        return speculative
    }
    /// True once a replica read was submitted (its failure then means both sources failed).
    package var isHedged: Bool {
        condition.lock()
        defer { condition.unlock() }
        return replicaSubmitted
    }

    package func tryReplica() {
        condition.lock()
        guard !cancelled, !speculative, published == nil, !replicaSubmitted, let alternateSource,
            primaryOutcome != nil || Double(DispatchTime.now().uptimeNanoseconds - created) / 1e9 >= delay
        else {
            condition.unlock()
            return
        }
        guard let lease = arena.acquire(bytes: recordBytes) else {
            // Every staging record is pinned, possibly by the waiter's own window: a failed primary must not wait
            // for one. Publish its failure; the owner treats it as an unhedged read's.
            guard let primaryOutcome else {
                condition.unlock()
                return
            }
            published = primaryOutcome
            let retire = retireEntryIfReady()
            condition.broadcast()
            condition.unlock()
            if retire { releaseEntry() }
            return
        }
        defer { condition.unlock() }
        do {
            replicaTicket = try scheduler.submit(
                source: alternateSource, priority: .demand, generation: generation, byteCount: recordBytes,
                operation: { [read, lease] cancellation in
                    try cancellation.throwIfCancelled()
                    try read(alternateSource, lease.bytes)
                    try cancellation.throwIfCancelled()
                }, completion: { [self, lease] ticket, outcome in finishReplica(outcome, source: ticket.source, lease: lease) })
            replicaSubmitted = true
            arena.didSubmit(replica: true)
        } catch {
            // Bounded duplicate admission may defer; the primary stays owned. A closed scheduler or a cancelled
            // generation never admits it: a failed primary's outcome is then final, as when no staging record is free.
            guard let primaryOutcome, generation != scheduler.currentGeneration || !scheduler.snapshot().accepting else { return }
            published = primaryOutcome
            let retire = retireEntryIfReady()
            condition.broadcast()
            if retire {
                condition.unlock()
                releaseEntry()
                condition.lock()
            }
        }
    }

    // Called with the condition locked. The entry may leave the loader only
    // after its cache writer completed and a logical result became available.
    private func retireEntryIfReady() -> Bool {
        guard !entryReleased, primaryOutcome != nil, published != nil else { return false }
        entryReleased = true
        return true
    }

    private func finishPrimary(_ source: IOSource, outcome: IOOutcome) {
        // Only this callback may finish the cache producer. The race lock must
        // not be held while calling the loader's ownership lock.
        let actual = completePrimary(outcome)
        condition.lock()
        primaryOutcome = actual
        var cancelReplica = false
        // A still-speculative race never gets a hedge, so its failed primary is final and must retire.
        if published == nil {
            if cancelled {
                published = .cancelled
            } else if case .succeeded = actual {
                published = actual
                arena.didPublish(source, replica: false)
                cancelReplica = true
            } else if alternateSource == nil || replicaOutcome != nil || speculative {
                published = actual
            }
        }
        let retire = retireEntryIfReady()
        let other = cancelReplica ? replicaTicket : nil
        condition.broadcast()
        condition.unlock()
        if let other { _ = scheduler.cancel(other) }
        if retire { releaseEntry() }
    }

    private func finishReplica(_ outcome: IOOutcome, source: IOSource, lease: ExpertReplicaLease) {
        condition.lock()
        replicaOutcome = outcome
        var won = false
        if published == nil {
            if cancelled || scheduler.currentGeneration != generation {
                published = .cancelled
            } else if case .succeeded = outcome {
                replicaWinner = lease
                published = outcome
                arena.didPublish(source, replica: true)
                won = true
            } else if primaryOutcome != nil {
                published = outcome
            }
        }
        let retire = retireEntryIfReady()
        let queuedPrimary = won && primaryOutcome == nil ? primaryTickets : []
        condition.broadcast()
        condition.unlock()
        // A replica win must not cancel or complete a started cache writer. A still-queued one never touched its slot:
        // drop it, so a slow drive's queue gives back its cache slot and this lease (its cancelled completion retires).
        var dropped = true
        for ticket in queuedPrimary { dropped = scheduler.cancelQueued(ticket, anyPriority: true) && dropped }
        if !queuedPrimary.isEmpty && dropped {
            condition.lock()
            droppedPrimary = true
            condition.unlock()
        }
        if retire { releaseEntry() }
        wake?()
    }

    package func wait(timeout: TimeInterval) -> IOOutcome? {
        condition.lock()
        defer { condition.unlock() }
        if published == nil { _ = condition.wait(until: Date(timeIntervalSinceNow: timeout)) }
        return published
    }

    package func cancel() {
        condition.lock()
        cancelled = true
        if published == nil { published = .cancelled }
        let current = tickets
        let retire = retireEntryIfReady()
        condition.broadcast()
        condition.unlock()
        for ticket in current { _ = scheduler.cancel(ticket) }
        if retire { releaseEntry() }
    }

    /// Read hygiene: cancel a still-speculative race only while its primary is queued (never
    /// started) and no replica exists; the same terminal state as `cancel()`. The primary has
    /// no outcome yet, so the entry is not retired here (the caller may hold the loader lock):
    /// the primary's cancellation completion (`finishPrimary`) frees the cache slot and then
    /// retires the loader entry. Lock order race -> scheduler, as in `tryReplica`.
    package func cancelIfQueued() -> Bool {
        condition.lock()
        defer { condition.unlock() }
        guard !cancelled, speculative, published == nil, primaryOutcome == nil, !replicaSubmitted, primaryTickets.count == 1,
            scheduler.cancelQueued(primaryTickets[0])
        else { return false }
        cancelled = true
        published = .cancelled
        condition.broadcast()
        return true
    }

    /// Every read submitted, primary first (condition locked).
    private var tickets: [IOTicket] { primaryTickets + (replicaTicket.map { [$0] } ?? []) }

    /// With `deadline`, a read still running then is left running (its destination stays held: a teardown that gives up).
    package func joinAll(until deadline: Date? = nil) {
        condition.lock()
        let current = tickets
        condition.unlock()
        for ticket in current {
            if let deadline { _ = ticket.join(timeout: deadline.timeIntervalSinceNow) } else { _ = ticket.join() }
        }
    }

    package var isRetired: Bool {
        condition.lock()
        let current = tickets
        let terminal = published != nil
        condition.unlock()
        return terminal && current.allSatisfy { $0.join(timeout: 0) != nil }
    }
}
