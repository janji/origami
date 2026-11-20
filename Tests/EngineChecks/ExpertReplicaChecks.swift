import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func replicaRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw EngineError(message) }
}
private func replicaWait(_ gate: DispatchSemaphore) throws {
    guard gate.wait(timeout: .now() + 5) == .success else { throw EngineError("Replica fixture gate timed out") }
}
private func replicaEventually(_ condition: () -> Bool) throws {
    let deadline = Date(timeIntervalSinceNow: 1)
    while !condition() && Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
    try replicaRequire(condition(), "Joined replica references did not retire")
}
private final class ReplicaFixtureState: @unchecked Sendable {
    private let lock = NSLock()
    private var errors: [String] = []
    private var running = true
    private var counts: [IOSource: Int] = [:]
    func fail(_ error: Error) {
        lock.lock()
        errors.append(String(describing: error))
        lock.unlock()
    }
    func stop() {
        lock.lock()
        running = false
        lock.unlock()
    }
    var shouldContinue: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }
    var failures: [String] {
        lock.lock()
        defer { lock.unlock() }
        return errors
    }
    func read(_ source: IOSource) {
        lock.lock()
        counts[source, default: 0] += 1
        lock.unlock()
    }
    var reads: [IOSource: Int] {
        lock.lock()
        defer { lock.unlock() }
        return counts
    }
}
private func replicaOwners(limits: IOScheduler.Limits? = nil) throws -> (MemoryLedger, ExpertCache, IOScheduler, ExpertReplicaArena) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(
        device: device, ledger: ledger,
        byteLimit: UInt64(8 * PackStore.expertBytes), slotsPerSlab: 8)
    let limits =
        limits
        ?? IOScheduler.Limits(
            maximumConcurrent: 4, reservedDemandConcurrent: 4,
            maximumOutstanding: 16, reservedDemandOutstanding: 16,
            maximumOutstandingBytes: UInt64(16 * PackStore.expertBytes), reservedDemandBytes: UInt64(16 * PackStore.expertBytes))
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    return (ledger, cache, scheduler, try ExpertReplicaArena(ledger: ledger, device: device))
}

func expertReplicaPublishesBeforeBlockedLoserAndJoinsRetirement() throws {
    let (_, cache, scheduler, arena) = try replicaOwners()
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let state = ReplicaFixtureState()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, source, destination in
        state.read(source)
        if source == .externalSSD {
            started.signal()
            try replicaWait(release)
            // Deliberately different late bytes expose any illegal shared
            // destination or second publication after the internal win.
            destination.initializeMemory(as: UInt8.self, repeating: 0xB5)
        } else {
            destination.initializeMemory(as: UInt8.self, repeating: 0x4D)
        }
    }
    defer {
        release.signal()
        scheduler.shutdown()
    }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    var request: ExpertLoadRequest? = try loader.request(ExpertKey(layer: 0, expert: 0), generation: generation, source: .externalSSD)
    try replicaWait(started)
    var result: LoadedExpert? = try loader.joinAndPin(request!)
    defer { try? result?.retireAfterGPUCompletion() }
    try replicaRequire(result!.buffer.bytes.allSatisfy { $0 == 0x4D }, "Incomplete or incorrect replica winner")
    try replicaRequire(
        arena.snapshot().leasedRecords == 1 && arena.snapshot().internalWinners == 1,
        "Losing destination retired before its reader")
    try replicaRequire(
        cache.snapshot().inFlightReads == 1 && arena.snapshot().consumerPins == 1,
        "Replica win prematurely published the still-written cache slot or omitted its GPU pin")
    let joined = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        loader.joinRetainedReads()
        joined.signal()
    }
    try replicaRequire(
        joined.wait(timeout: .now() + .milliseconds(30)) == .timedOut,
        "Retirement returned while the original writer was blocked")
    release.signal()
    try replicaWait(joined)
    try replicaRequire(result!.buffer.bytes.allSatisfy { $0 == 0x4D }, "Late original changed published bytes")
    try replicaRequire(
        cache.snapshot().inFlightReads == 0 && arena.snapshot().leasedRecords == 1,
        "Replica GPU pin was released by the unrelated original completion")
    request = nil
    try result!.retireAfterGPUCompletion()
    result = nil
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
    try replicaRequire(
        arena.snapshot().replicaWinners == 1
            && state.reads == [.externalSSD: 1, .internalSSD: 1],
        "Replica storage leaked or duplicate limit exceeded")
    try loader.cancelAndJoin(generation)
}

func expertReplicaCancellationRetainsBothWriters() throws {
    let (_, cache, scheduler, arena) = try replicaOwners()
    let external = DispatchSemaphore(value: 0)
    let internalRead = DispatchSemaphore(value: 0)
    let releaseExternal = DispatchSemaphore(value: 0)
    let releaseInternal = DispatchSemaphore(value: 0)
    let done = DispatchSemaphore(value: 0)
    let state = ReplicaFixtureState()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, source, destination in
        if source == .externalSSD {
            external.signal()
            try replicaWait(releaseExternal)
        } else {
            internalRead.signal()
            try replicaWait(releaseInternal)
        }
        destination.initializeMemory(as: UInt8.self, repeating: 0x7C)
    }
    defer {
        releaseExternal.signal()
        releaseInternal.signal()
        scheduler.shutdown()
    }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    let request = try loader.request(ExpertKey(layer: 1, expert: 1), generation: generation, source: .externalSSD)
    try replicaWait(external)
    DispatchQueue.global().async {
        defer { done.signal() }
        do {
            let result = try loader.joinAndPin(request, shouldContinue: { state.shouldContinue })
            try result.retireAfterGPUCompletion()
            state.fail(EngineError("Cancelled replica published a consumer"))
        } catch is IOOperationCancelled {} catch { state.fail(error) }
    }
    try replicaWait(internalRead)
    state.stop()
    try replicaRequire(
        done.wait(timeout: .now() + .milliseconds(30)) == .timedOut,
        "Cancellation returned before writers joined")
    try replicaRequire(
        arena.snapshot().leasedRecords == 1 && cache.snapshot().inFlightReads == 1,
        "Cancellation recycled a live staging or direct cache destination")
    releaseExternal.signal()
    releaseInternal.signal()
    try replicaWait(done)
    try loader.cancelAndJoin(generation)
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
    try replicaRequire(
        state.failures.isEmpty,
        "Cancelled replica did not retire cleanly: \(state.failures)")
    try replicaRequire(
        cache.snapshot().pinnedSlots == 0 && cache.snapshot().inFlightReads == 0,
        "Cancelled replica left a published pin or writer")
}

func expertReplicaDualFailuresAndArenaBoundAreExplicit() throws {
    let experts = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 3424,
        slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4, prefillCacheSlots: 1024, tailReplicas: true)
    for context in [8192, 32768] {
        for input in [0, 1, 255] {
            let fixed = try ConversationGeometry(context: context).maximumFixedBytes(
                input: input,
                residentBytes: 2_962_292_736)
            let expected = min(
                experts.cacheBytes,
                (15_000_000_000 - fixed - 88_670_208 - 16_777_216) / 22_118_400 * 22_118_400)
            let actual = try ConversationPhaseBudget.expertBytes(
                context: context, input: input,
                residentBytes: 2_962_292_736, experts: experts, usable: 15_000_000_000,
                additionalOwners: 16_777_216)
            try replicaRequire(actual == expected, "Replica arena omitted or double-charged in phase admission")
        }
    }
    let (_, cache, scheduler, arena) = try replicaOwners()
    var leases: [ExpertReplicaLease] = []
    for _ in 0..<32 {
        guard let lease = arena.acquire() else { throw EngineError("Incomplete 32-record arena") }
        leases.append(lease)
    }
    try replicaRequire(
        arena.acquire() == nil && arena.snapshot().highWaterRecords == 32
            && arena.snapshot().arenaBytes == 88_670_208, "Replica arena exceeded its independent complete bound")
    leases.removeAll()
    let state = ReplicaFixtureState()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, source, _ in
        state.read(source)
        throw EngineError("injected source failure")
    }
    defer { scheduler.shutdown() }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    let request = try loader.request(ExpertKey(layer: 2, expert: 2), generation: generation, source: .externalSSD)
    var rejected = false
    do {
        let result = try loader.joinAndPin(request)
        try result.retireAfterGPUCompletion()
    } catch is EngineError { rejected = true }
    try loader.cancelAndJoin(generation)
    try replicaRequire(
        rejected && state.reads == [.externalSSD: 1, .internalSSD: 1],
        "Dual failure silently substituted bytes or exceeded one replica")
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
    try replicaRequire(
        cache.snapshot().inFlightReads == 0,
        "Failure left a live staging/cache producer")
}

func expertReplicaDirectCacheAndEscapedMetalOwnership() throws {
    try autoreleasepool {
        let (_, cache, scheduler, arena) = try replicaOwners()
        let loader = SourceExpertLoader(
            cache: cache, scheduler: scheduler, replicaArena: arena,
            packedAvailable: { true }
        ) { _, _, destination in
            destination.initializeMemory(as: UInt8.self, repeating: 0x2C)
        }
        defer { scheduler.shutdown() }
        try loader.setReplicasEnabled(true)
        let generation = scheduler.currentGeneration
        let request = try loader.request(ExpertKey(layer: 3, expert: 3), generation: generation, source: .internalSSD)
        let result = try loader.joinAndPin(request)
        try replicaRequire(result.buffer.bytes.allSatisfy { $0 == 0x2C }, "Direct primary did not fill the cache record")
        try replicaRequire(
            arena.snapshot().highWaterRecords == 0 && arena.snapshot().replicaRecords == 0,
            "Healthy primary used replica storage or publication copies")
        try result.retireAfterGPUCompletion()
        try loader.cancelAndJoin(generation)
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    var escaped: MTLBuffer?
    try autoreleasepool {
        var arena: ExpertReplicaArena? = try ExpertReplicaArena(ledger: ledger, device: device)
        var lease = arena!.acquire()
        weak let staged = lease
        escaped = lease!.expertBuffer.buffer
        lease!.pinConsumer()
        lease = nil
        try replicaRequire(
            staged != nil && arena!.snapshot().consumerPins == 1 && arena!.snapshot().leasedRecords == 1,
            "Dropping a live consumer released a GPU-visible record")
        try staged!.retireConsumer()
        try replicaEventually { arena!.snapshot().leasedRecords == 0 }
        arena = nil
    }
    try replicaRequire(
        ledger.snapshot().categories["expert-replica-staging"] == 88_670_208,
        "Escaped Metal resource lost its reservation")
    try replicaRequire(escaped?.length == 88_604_672, "Escaped replica arena has the wrong physical size")
    escaped = nil
    try replicaEventually { ledger.snapshot().reserved == 0 }
}

func expertReplicaFailedPredictionRetiresAndDemandRereads() throws {
    // Predictions run at prefetch priority, which needs capacity beyond the demand reservation.
    let (_, cache, scheduler, arena) = try replicaOwners(
        limits: IOScheduler.Limits(
            maximumConcurrent: 4, reservedDemandConcurrent: 2,
            maximumOutstanding: 16, reservedDemandOutstanding: 8,
            maximumOutstandingBytes: UInt64(16 * PackStore.expertBytes), reservedDemandBytes: UInt64(8 * PackStore.expertBytes)))
    let state = ReplicaFixtureState()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, source, destination in
        state.read(source)
        guard source == .internalSSD else { throw EngineError("injected external failure") }
        destination.initializeMemory(as: UInt8.self, repeating: 0x5A)
    }
    defer { scheduler.shutdown() }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    let key = try ExpertKey(layer: 4, expert: 4)
    guard case .issued(let predicted) = try loader.requestSpeculative(key, generation: generation, source: .externalSSD) else {
        throw EngineError("Speculative fixture read was not issued")
    }
    loader.joinDiscarding(predicted)
    // An unhedged prediction's failure must retire, or the next phase change refuses.
    loader.joinRetainedReads()
    try loader.setReplicasEnabled(true)
    var request: ExpertLoadRequest? = try loader.request(key, generation: generation, source: .externalSSD)
    var result: LoadedExpert? = try loader.joinAndPin(request!)
    loader.joinRetainedReads()
    try replicaRequire(
        result!.buffer.bytes.allSatisfy { $0 == 0x5A } && state.reads == [.externalSSD: 2, .internalSSD: 1],
        "Demand after a failed prediction did not start a fresh hedged read")
    request = nil
    try result!.retireAfterGPUCompletion()
    result = nil
    try loader.cancelAndJoin(generation)
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
}

/// Every staging record pinned: a failed external primary does not wait for one, it is retried on the model
/// drive like an unhedged read; a failed model-drive primary is reported as the original read failing.
func expertReplicaWithoutStagingRetriesOnModelDrive() throws {
    let (_, cache, scheduler, arena) = try replicaOwners()
    let state = ReplicaFixtureState()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { key, source, destination in
        state.read(source)
        if source == .externalSSD || key.expert == 6 { throw EngineError("injected \(source) failure") }
        destination.initializeMemory(as: UInt8.self, repeating: 0x3E)
    }
    defer { scheduler.shutdown() }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    var leases: [ExpertReplicaLease] = []
    while let lease = arena.acquire() { leases.append(lease) }
    try replicaRequire(leases.count == ExpertReplicaArena.recordCount, "Fixture did not pin every staging record")
    let deadline = Date(timeIntervalSinceNow: 5)
    var result: LoadedExpert? = try loader.joinAndPin(
        loader.request(
            ExpertKey(layer: 5, expert: 5), generation: generation,
            source: .externalSSD), shouldContinue: { Date() < deadline })
    try replicaRequire(
        result!.buffer.bytes.allSatisfy { $0 == 0x3E } && state.reads == [.externalSSD: 1, .internalSSD: 1],
        "Failed primary without a free staging record was not retried on the model drive: \(state.reads)")
    try result!.retireAfterGPUCompletion()
    result = nil
    var message = ""
    do { _ = try loader.joinAndPin(loader.request(ExpertKey(layer: 6, expert: 6), generation: generation, source: .internalSSD)) } catch let
        error as EngineError
    { message = error.description }
    try replicaRequire(message.contains("Original expert read failed"), "Model-drive failure reported as: \(message)")
    leases.removeAll()
    try loader.cancelAndJoin(generation)
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
}

/// Predictions can evict every victim the row prepared; with all 24 transient
/// slots busy, exact demand must rescan and borrow a still-eligible record.
func expertReuseDemandRescansDrainedVictims() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(40 * PackStore.expertBytes), reuseAdmission: true)
    try cache.setReuseDecode(true)
    // Sixteen verified retained records; equal scores keep slots 24...33 as the prepared ten.
    for expert in 0..<16 {
        let a = try cache.acquire(ExpertKey(layer: 0, expert: expert))
        _ = try cache.beginRead(a.ticket)
        try replicaRequire(cache.completeRead(a.ticket, verified: true), "Retained fixture producer failed")
    }
    for expert in 0..<24 { _ = try cache.beginRead(cache.acquire(ExpertKey(layer: 1, expert: expert)).ticket) }
    for expert in 0..<10 {
        guard case .overridden = try cache.acquireSpeculative(ExpertKey(layer: 5, expert: expert), overrideScore: true) else {
            throw EngineError("Prediction fixture did not take a prepared victim")
        }
    }
    let demand = try cache.acquire(ExpertKey(layer: 2, expert: 0))
    try replicaRequire(
        demand.newlyReserved && cache.snapshot().reuseBorrowedBypasses == 1,
        "Exact demand did not borrow a record outside the drained victim list")
}

/// Decode demand runs while its own layer is held for the GPU. When that layer fills the ten lowest scores and slow
/// reads keep all 24 transient slots busy, exact demand must still borrow an unheld record, after a trim that drops
/// nothing (every slab is busy or held).
func expertReuseDemandSkipsGPUHeldVictims() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(40 * PackStore.expertBytes), reuseAdmission: true)
    try cache.setReuseDecode(true)
    _ = try cache.gpuTable()
    // Ten layer-20 records (slots 24...33, the prepared ten on equal scores), then six of layer 19.
    for (layer, expert) in (0..<10).map({ (20, $0) }) + (0..<6).map({ (19, $0) }) {
        let a = try cache.acquire(ExpertKey(layer: layer, expert: expert))
        _ = try cache.beginRead(a.ticket)
        try replicaRequire(cache.completeRead(a.ticket, verified: true), "Retained fixture producer failed")
    }
    _ = cache.holdForGPU(layer: 20)
    defer { cache.releaseForGPU(layer: nil) }
    for expert in 0..<24 { _ = try cache.beginRead(cache.acquire(ExpertKey(layer: 21, expert: expert)).ticket) }
    try cache.resize(to: UInt64(32 * PackStore.expertBytes))
    let demand = try cache.acquire(ExpertKey(layer: 21, expert: 24))
    try replicaRequire(
        demand.newlyReserved && cache.snapshot().reuseBorrowedBypasses == 1
            && (try cache.draftResidentExperts(layer: 20).count) == 10 && (try cache.draftResidentExperts(layer: 19).count) == 5,
        "Exact demand did not borrow an unheld record past the GPU-held layer")
}

/// A demand that adopts a dropped prediction between its publication (no handle left, so no handoff pin) and its
/// retirement from the loader re-pins the verified record instead of failing as discarded. That moment is short, so
/// every round races adoption against the read's completion.
func expertReplicaLateAdopterRepinsPublishedRecord() throws {
    let (_, cache, scheduler, arena) = try replicaOwners(
        limits: IOScheduler.Limits(
            maximumConcurrent: 4, reservedDemandConcurrent: 2,
            maximumOutstanding: 16, reservedDemandOutstanding: 8,
            maximumOutstandingBytes: UInt64(16 * PackStore.expertBytes), reservedDemandBytes: UInt64(8 * PackStore.expertBytes)))
    let gate = DispatchSemaphore(value: 0)
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, _, destination in
        try replicaWait(gate)
        destination.initializeMemory(as: UInt8.self, repeating: 0x6C)
    }
    defer { scheduler.shutdown() }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    for round in 0..<100 {
        let key = try ExpertKey(layer: round % 8, expert: round / 8)
        guard case .issued = try loader.requestSpeculative(key, generation: generation, source: .externalSSD) else {
            throw EngineError("Speculative fixture read \(round) was not issued")
        }
        gate.signal()
        while true {
            let request = try loader.request(key, generation: generation, source: .externalSSD)
            guard request.residentAtRequest || loader.readFinished(request) else { continue }
            let result = try loader.joinAndPin(request)
            try replicaRequire(result.buffer.bytes.allSatisfy { $0 == 0x6C }, "Round \(round) pinned the wrong bytes")
            try result.retireAfterGPUCompletion()
            break
        }
    }
    try loader.cancelAndJoin(generation)
    try replicaEventually { arena.snapshot().leasedRecords == 0 }
}

/// A replica that wins while its external primary is still queued (a slow drive's backlog) cancels that primary: its
/// cache slot and staging lease come back without waiting for the drive. A started primary still finishes (above).
func expertReplicaWinCancelsQueuedPrimary() throws {
    let bytes = UInt64(16 * PackStore.expertBytes)
    let (_, cache, scheduler, arena) = try replicaOwners(
        limits: IOScheduler.Limits(
            maximumConcurrent: 1, reservedDemandConcurrent: 1,
            maximumOutstanding: 16, reservedDemandOutstanding: 16, maximumOutstandingBytes: bytes, reservedDemandBytes: bytes))
    let state = ReplicaFixtureState()
    let release = DispatchSemaphore(value: 0)
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena,
        packedAvailable: { true }
    ) { _, source, destination in
        state.read(source)
        destination.initializeMemory(as: UInt8.self, repeating: source == .internalSSD ? 0x4D : 0xB5)
    }
    defer {
        release.signal()
        scheduler.shutdown()
    }
    try loader.setReplicasEnabled(true)
    let generation = scheduler.currentGeneration
    // The one external lane is busy with an unrelated slow read, so the primary stays queued behind it.
    let blocker = try scheduler.submit(
        source: .externalSSD, priority: .demand, generation: generation,
        byteCount: PackStore.expertBytes, operation: { _ in _ = release.wait(timeout: .now() + 5) })
    var request: ExpertLoadRequest? = try loader.request(ExpertKey(layer: 0, expert: 0), generation: generation, source: .externalSSD)
    var result: LoadedExpert? = try loader.joinAndPin(request!)
    try replicaRequire(result!.buffer.bytes.allSatisfy { $0 == 0x4D }, "Incomplete or incorrect replica winner")
    try replicaEventually { cache.snapshot().inFlightReads == 0 }
    request = nil
    try result!.retireAfterGPUCompletion()
    result = nil
    loader.joinRetainedReads()
    try replicaRequire(
        arena.snapshot().leasedRecords == 0 && state.reads == [.internalSSD: 1],
        "The queued losing primary kept its staging lease or read the slow drive: \(state.reads)")
    release.signal()
    _ = blocker.join()
}
