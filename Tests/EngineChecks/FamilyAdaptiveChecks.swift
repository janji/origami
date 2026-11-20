import Foundation
import Metal
import OrigamiEngine

private func familyAdaptiveRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError("Family adaptive admission: " + message) }
}

/// Poor ranks recover from real router observations without speculative probe reads. Layer histories are separate,
/// and a confident prediction is still refused when the queue or measured arrival deadline cannot accommodate it.
func familyAdaptivePrefetchPolicy() throws {
    var policy = FamilyPrefetchPolicy()
    func choice(_ layer: Int = 1, _ rank: Int = 0, queued: Int = 0, wait: Double = 0, service: Double = 0.01)
        -> FamilyPrefetchPolicy.Decision
    {
        policy.decide(layer: layer, rank: rank, queuedRecords: queued, maximumRecords: 8, queuedSeconds: wait, serviceSeconds: service)
    }
    try familyAdaptiveRequire(choice(1, 0, queued: 2) == .queue, "cold policy flooded the queue")
    for _ in 0..<8 { policy.observe(layer: 1, predicted: [4, 5], needed: [4], seconds: 0.1) }
    try familyAdaptiveRequire(choice() == .admit && choice(1, 1) == .confidence, "rank usefulness was not learned")
    try familyAdaptiveRequire(choice(2, 1) == .admit, "a poor layer suppressed a different layer")
    // A read that can begin before the router is admitted even if it ends after it; one that cannot begin is refused.
    try familyAdaptiveRequire(choice(1, 0, wait: 0.095) == .admit, "work that starts before the router was refused")
    try familyAdaptiveRequire(choice(1, 0, wait: 0.105) == .deadline, "late queued work was admitted")
    try familyAdaptiveRequire(choice(1, 0, queued: 8) == .queue, "mature queue cap was ignored")
    for _ in 0..<32 { policy.observe(layer: 1, predicted: [4, 5], needed: [4, 5], seconds: 0.1) }
    try familyAdaptiveRequire(choice(1, 1) == .admit, "declined ranks could not recover without probe I/O")
}

/// Simulated wire outcomes exercise accounting deterministically. The buffers are tiny, and the hooks do not alter
/// the host's wired-memory limit or pin pages. The ledger charge follows an escaped Metal view after cache retirement.
func familySlabWiringLifetime() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let saved = (FamilySlabStorage.wire, FamilySlabStorage.unwire)
    defer { (FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    let bytes = Int(getpagesize())
    let ledger = try MemoryLedger(ceiling: UInt64(8 * bytes), transientMargin: 1)
    let accounting = FamilySlabWiring()
    var escaped: MTLBuffer?
    var wireCalls = 0
    var unwireCalls = 0
    FamilySlabStorage.wire = { _, _ in
        wireCalls += 1
        return wireCalls != 2
    }
    FamilySlabStorage.unwire = { _, _ in
        unwireCalls += 1
        return true
    }
    try autoreleasepool {
        let firstBuffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        let secondBuffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        var first: FamilySlabStorage? = FamilySlabStorage(
            buffer: firstBuffer, ledger: ledger,
            reservation: try ledger.reserve(bytes: UInt64(bytes), category: "family-experts"), wiring: accounting)
        var second: FamilySlabStorage? = FamilySlabStorage(
            buffer: secondBuffer, ledger: ledger,
            reservation: try ledger.reserve(bytes: UInt64(bytes), category: "family-experts"), wiring: accounting)
        escaped = first!.buffer
        try familyAdaptiveRequire(
            second != nil && accounting.snapshot.wiredBytes == bytes && accounting.snapshot.failedBytes == bytes,
            "a failed wire changed another slab's wired bytes")
        second = nil
        try familyAdaptiveRequire(
            unwireCalls == 0 && accounting.snapshot.wiredBytes == bytes,
            "an unwired slab was passed to munlock")
        first = nil
        try familyAdaptiveRequire(
            unwireCalls == 1 && accounting.snapshot.wiredBytes == 0 && ledger.reservedBytes == UInt64(2 * bytes),
            "cache retirement lost the retained Metal reservation")
    }
    try familyAdaptiveRequire(
        accounting.snapshot.ownedBytes == bytes && accounting.snapshot.unwiredOwnedBytes == bytes
            && ledger.reservedBytes == UInt64(bytes), "escaped allocation was not reported as retained and unwired")
    withExtendedLifetime(escaped) {}
    escaped = nil
    try familyAdaptiveRequire(
        accounting.snapshot.ownedBytes == 0 && ledger.reservedBytes == 0, "last Metal owner did not retire its charge")

    FamilySlabStorage.wire = { _, _ in true }
    FamilySlabStorage.unwire = { _, _ in false }
    try autoreleasepool {
        let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        var slab: FamilySlabStorage? = FamilySlabStorage(
            buffer: buffer, ledger: ledger,
            reservation: try ledger.reserve(bytes: UInt64(bytes), category: "family-experts"), wiring: accounting)
        escaped = slab!.buffer
        slab = nil
    }
    try familyAdaptiveRequire(
        accounting.snapshot.unwireFailures == 1 && accounting.snapshot.wiredBytes == bytes,
        "failed unwire falsely reported memory relief")
    withExtendedLifetime(escaped) {}
    escaped = nil
    try familyAdaptiveRequire(
        accounting.snapshot.wiredBytes == 0 && ledger.reservedBytes == 0, "destroyed failed-unwire allocation stayed charged")
}

/// Pressure does not prevent the mandatory two-wave bootstrap or reuse of its existing slots. At that floor a guess or
/// read-ahead has no slot to take and needs a new slab, so it is refused at once and nothing waits; a guess already
/// queued in an existing slot is kept and lands. Each refusal is counted once, as live while pressure is live. No system
/// pressure is induced; only the ledger's latches are changed.
func familyPressurePrefetchAndFloor() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-adaptive-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let saved = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 1
    defer { FamilyExperts.slabBytes = saved }
    let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
    let pool = try FamilyExperts(weights: weights, budget: 4 * (212_992 + 114_688))
    pool.ledger = ledger
    ledger.setOptionalGrowthAllowed(false)
    _ = try pool.fetch(layer: 1, experts: [0, 1], decode: true)
    let floor = pool.residentBytes
    _ = try pool.fetch(layer: 1, experts: [2, 3], decode: true)
    try familyAdaptiveRequire(
        pool.residentBytes == floor && pool.resident(layer: 1) == [2, 3], "pressure blocked reuse or admitted optional growth")
    pool.prefetch(layer: 2, experts: [0, 1])
    pool.readAhead(layer: 2, experts: [0, 1])
    try familyAdaptiveRequire(
        pool.counters.prefetches == 0 && pool.counters.readAheads == 0 && pool.counters.guessRejectedPressure == 2
            && pool.pendingReads == 0 && pool.residentBytes == floor
            && ledger.optionalRefusals.belowTarget == 4 && ledger.optionalRefusals.live == 0,
        "a guess at the floor read or grew the cache, or was counted as live pressure")
    // The same refusal while pressure is live is counted as live: it is the slab, not the read, that pressure refuses.
    ledger.setOptionalReadsAllowed(false)
    pool.prefetch(layer: 2, experts: [0, 1])
    ledger.setOptionalReadsAllowed(true)
    try familyAdaptiveRequire(
        pool.counters.prefetches == 0 && pool.counters.guessRejectedPressure == 4 && ledger.optionalRefusals.live == 2
            && ledger.optionalRefusals.belowTarget == 4, "a refusal under live pressure was not counted as live")
    ledger.setOptionalGrowthAllowed(true)
    let hold = pool.holdGuesses()
    defer {
        hold.release()
        pool.waitForReads()
    }
    pool.prefetch(layer: 2, experts: [0, 1])
    let grown = pool.residentBytes
    ledger.setOptionalGrowthAllowed(false)
    try familyAdaptiveRequire(
        pool.resident(layer: 2).isEmpty && pool.counters.prefetchCancelled == 0 && pool.pendingReads > 0,
        "pressure cancelled guesses queued in existing slots")
    hold.release()
    let reads = pool.counters.readBytes
    _ = try pool.fetch(layer: 2, experts: [0, 1], decode: true)
    try familyAdaptiveRequire(
        pool.resident(layer: 2) == [0, 1] && pool.counters.prefetchUsed == 2 && pool.counters.readBytes == reads
            && pool.residentBytes == grown, "queued guesses were not used under pressure")
}

/// A queued guess read still owns its entered completion group and destination until its job ends. Destroy the cache
/// with held reads queued, then release the separately retained scheduler: every group part must leave and every
/// final Metal allocation charge retire, with no job/guess retain cycle. Guesses cancelled while queued (the scheduler
/// drops them at once) complete their cleanup and leave the charges to the cache, which retires them when destroyed.
func familyQueuedPrefetchOwnerTeardown() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-queued-owner-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    for cancelled in [false, true] {
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        var scheduler: IOScheduler!
        var hold: IOPrefetchHold?
        var issued = 0
        try autoreleasepool {
            var pool: FamilyExperts? = try FamilyExperts(weights: weights, budget: 4 * (212_992 + 114_688))
            pool!.ledger = ledger
            scheduler = pool!.reads
            hold = pool!.holdGuesses()
            pool!.prefetch(layer: 1, experts: [0, 1])
            issued = pool!.pendingReads
            try familyAdaptiveRequire(issued > 0, "teardown fixture did not queue a read")
            if cancelled {
                _ = try scheduler.cancelAndJoin(scheduler.currentGeneration)  // the scheduler drops the queued reads
                _ = pool!.resident(layer: 1)
                pool!.waitForReads()
                let drives = scheduler.snapshot().drives
                try familyAdaptiveRequire(
                    drives.reduce(UInt64(0)) { $0 + $1.cancelled } == UInt64(issued) && drives.allSatisfy { $0.outstanding == 0 },
                    "cancelled queued cleanup never completed")
            }
            pool = nil
        }
        if !cancelled {
            try familyAdaptiveRequire(ledger.reservedBytes > 0, "queued destination lost its ledger charge before cleanup")
            hold?.release()
            let end = Date(timeIntervalSinceNow: 30)
            while scheduler.snapshot().drives.contains(where: { $0.outstanding > 0 }) && Date() < end { usleep(200) }
            try familyAdaptiveRequire(
                scheduler.snapshot().drives.reduce(UInt64(0)) { $0 + $1.succeeded } == UInt64(issued), "queued reads never completed")
        }
        autoreleasepool {
            hold = nil
            scheduler = nil
        }
        // A completion still running on its own thread holds its buffer for a moment after the scheduler counts the read
        // done (seen on a loaded 16 GB Mac); a retain cycle would hold it for good.
        let retired = Date(timeIntervalSinceNow: 5)
        while ledger.reservedBytes != 0 && Date() < retired { usleep(200) }
        try familyAdaptiveRequire(ledger.reservedBytes == 0, "completed queued reads retained slab charges")
    }
}

/// Real prompt hashes are known work. Pressure defers a whole batch until its consumer, and later recovery allows
/// overlap again; no speculative token is introduced and only one batch can be outstanding.
func familyActualWorkspaceOwnership() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-workspace-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 64)
    let page = try FamilyRunner.scratchLength(1)
    let ledger = try MemoryLedger(ceiling: UInt64(16 * page), transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    runner.ledger = ledger
    var escaped: MTLBuffer?
    try autoreleasepool {
        try runner.prepareScratch(["x": page / 4], footprint: { 0 })
        escaped = try runner.work("x", 1)
        try runner.prepareScratch(["x": page / 2], footprint: { 0 })
    }
    try familyAdaptiveRequire(
        runner.workingAllocatedBytes == UInt64(2 * page) && runner.workingRetainedBytes == UInt64(3 * page)
            && ledger.reservedBytes == UInt64(3 * page), "scratch replacement lost the old Metal owner's charge")
    withExtendedLifetime(escaped) {}
    escaped = nil
    var probes = 0
    try runner.prepareScratch(
        ["x": 1],
        footprint: {
            probes += 1
            return 0
        })
    try familyAdaptiveRequire(
        runner.workingBytes == UInt64(2 * page) && ledger.reservedBytes == UInt64(2 * page) && probes == 0,
        "actual scratch reuse retained an idle maximum allowance or probed")
    runner.beginDecode()
    try familyAdaptiveRequire(runner.workingBytes == 0 && ledger.reservedBytes == 0, "prompt retirement left idle scratch charged")
}
