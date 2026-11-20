import Foundation
import Metal
import OrigamiEngine

/// Repeated request plans under a synthetic governor target: reclaim optional slots using all ledger owners,
/// preserve escaped scratch, then permit mandatory work when only the two-wave floor remains. No GPU dispatch,
/// large allocation or real host pressure is needed; both model families use the production scratch boundary.
func familyScratchPressureRecovery() throws {
    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw EngineError("Family scratch pressure: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-pressure-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let settings = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = settings }
    let page = UInt64(try FamilyRunner.scratchLength(1))
    for family in [ModelFamily.glm, .deepseek] {
        // Automatic mode skips real-process footprint sampling; the separate physical-denial check below
        // supplies a deterministic footprint to an explicit ledger. Neither changes system pressure.
        let ledger = try MemoryLedger(ceiling: FamilyRunner.baseWorkingBytes + (64 << 20), transientMargin: 1, availableFloor: 1)
        let maximum = ledger.maximumUsableBytes
        try autoreleasepool {
            let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(family, in: folder, experts: 8)), device: device)
            let model = try makeFamilyModel(
                weights, expertBudget: 1 << 20, memoryLimit: ledger.ceiling,
                context: 64, ledger: ledger
            ) { _ in }
            let runner = model as! FamilyRunner
            ledger.setUsable(maximum)
            try runner.prepareScratch(["value": Int(page / 4)], footprint: { 0 })
            try autoreleasepool {
                _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8)
                _ = try runner.experts.fetch(layer: 0, experts: Array(0..<8))
            }
            let floor = runner.experts.leastAllocatedBytes
            let fullCache = UInt64(runner.experts.residentBytes)
            try require(fullCache > floor, "\(family): fixture has no optional cache")
            let fixed = ledger.reservedBytes - fullCache - runner.workingRetainedBytes
            let unrelated = try ledger.reserve(bytes: 5 * page, category: "scratch-pressure-unrelated-owner")
            defer { try? ledger.release(unrelated) }
            let target = fixed + 5 * page + 3 * page + floor
            ledger.setUsable(target)
            ledger.setOptionalGrowthAllowed(false)
            try require(ledger.reservedBytes > target, "\(family): target did not fall below existing owners")
            weak var oldOwner: (any MTLBuffer)?
            var escaped: MTLBuffer?
            try autoreleasepool {
                let old = try runner.work("value", 1)
                old.contents().storeBytes(of: UInt32(0x7654_3210), as: UInt32.self)
                escaped = old
                oldOwner = old
                try runner.prepareScratch(["value": Int(page / 2)], footprint: { 0 })
                try require(
                    runner.workingAllocatedBytes == 2 * page && runner.workingRetainedBytes == 3 * page,
                    "\(family): escaped replacement owner lost its charge")
                try require(
                    old.contents().load(as: UInt32.self) == 0x7654_3210,
                    "\(family): replacement overwrote escaped scratch")
                try require(
                    UInt64(runner.experts.residentBytes) == floor && ledger.reservedBytes <= target,
                    "\(family): actual-ledger shortfall did not reclaim optional cache")
                try require(
                    ledger.snapshot().categories["scratch-pressure-unrelated-owner"] == 5 * page,
                    "\(family): unrelated owner was hidden to satisfy admission")
            }
            withExtendedLifetime(escaped) {}
            escaped = nil
            try require(
                oldOwner == nil && runner.workingRetainedBytes == 2 * page,
                "\(family): retired scratch stayed charged")
            // A target below the irreducible owner set is a request to trim optional memory, not permission
            // to fail required scratch while the immutable limit has ample room.
            ledger.setUsable(1)
            try runner.prepareScratch(["value": Int(3 * page / 4)], footprint: { 0 })
            try require(
                runner.workingAllocatedBytes == 3 * page && runner.workingRetainedBytes == 3 * page
                    && UInt64(runner.experts.residentBytes) == floor && ledger.reservedBytes > ledger.usableBytes,
                "\(family): essential work failed or grew cache below its temporary target")
            try require(
                ledger.maximumUsableBytes == maximum && ledger.usableBytes == 1 && !ledger.optionalGrowthAllowed,
                "\(family): admission changed the user cap, governor target or pressure latch")
            ledger.setUsable(maximum)
            ledger.setOptionalGrowthAllowed(true)
            runner.fitExperts()
            _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8)
            try require(
                UInt64(runner.experts.residentBytes) == fullCache,
                "\(family): recovery did not restore optional capacity")
            // No optional slab is available to reclaim now. The immutable maximum must still reject growth,
            // preserving the previous pointer, bytes and ledger charge, and permit a retry once the owner retires.
            autoreleasepool { runner.experts.setLimit(bytes: 0) }
            let occupied = try ledger.reserve(bytes: maximum - ledger.reservedBytes, category: "scratch-pressure-hard-owner")
            let old = try runner.work("value", 1)
            let before = ledger.reservedBytes
            old.contents().storeBytes(of: UInt32(0x1020_3040), as: UInt32.self)
            do {
                try runner.prepareScratch(["value": Int(page)], footprint: { 0 })
                throw EngineError("Family scratch pressure: immutable maximum admitted growth", code: "fixture_unexpected")
            } catch let error as EngineError where error.code == "insufficient_memory" {}
            try require(
                ledger.reservedBytes == before && runner.workingRetainedBytes == 3 * page
                    && (try runner.work("value", 1)) === old && old.contents().load(as: UInt32.self) == 0x1020_3040,
                "\(family): denied replacement changed live bytes or owner accounting")
            try ledger.release(occupied)
            try runner.prepareScratch(["value": Int(page)], footprint: { 0 })
            try require(
                runner.workingAllocatedBytes == 4 * page && ledger.maximumUsableBytes == maximum,
                "\(family): hard-owner retirement did not permit retry")
            withExtendedLifetime(old) {}
        }
        try require(ledger.reservedBytes == 0, "\(family): final owners leaked ledger reservations")
    }
}

/// Mandatory progress never overrides physical admission, even when the temporary target itself is ignored.
func familyScratchPressurePhysicalDenial() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-pressure-physical-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 64)
    let page = UInt64(try FamilyRunner.scratchLength(1))
    let ledger = try MemoryLedger(ceiling: 16 * page, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    runner.ledger = ledger
    try runner.prepareScratch(["value": Int(page / 4)], footprint: { 0 })
    let old = try runner.work("value", 1)
    let before = ledger.reservedBytes
    old.contents().storeBytes(of: UInt32(0x3141_5926), as: UInt32.self)
    ledger.setUsable(1)
    ledger.setOptionalGrowthAllowed(false)
    do {
        try runner.prepareScratch(["value": Int(page / 2)], footprint: { ledger.ceiling - page })
        throw EngineError("Family scratch pressure: physical ceiling admitted growth", code: "fixture_unexpected")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    guard ledger.reservedBytes == before, runner.workingRetainedBytes == page,
        try runner.work("value", 1) === old, old.contents().load(as: UInt32.self) == 0x3141_5926
    else {
        throw EngineError("Family scratch pressure: physical denial changed a live owner")
    }
}

/// A final Metal owner can retire before the process-footprint sample reflects its pages. Inject that lag at
/// the real scratch boundary: no replacement may allocate until the measured physical envelope fits. A
/// persistent high sample still rejects after the bounded retries, with every old buffer/charge preserved.
func familyScratchRetirementRecheck() throws {
    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw EngineError("Family scratch retirement: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-retirement-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let settings = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = settings }
    let page = UInt64(try FamilyRunner.scratchLength(1))
    // The large LIMIT lets ordinary tiny expert allocations admit against the real check-process footprint;
    // no large buffer is allocated. Only prepareScratch receives the synthetic physical-pressure samples.
    let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    try autoreleasepool {
        let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(.deepseek, in: folder, experts: 8)), device: device)
        let model = try makeFamilyModel(
            weights, expertBudget: 8 * page, memoryLimit: ledger.ceiling,
            context: 64, ledger: ledger
        ) { _ in }
        let runner = model as! FamilyRunner
        // makeExperts starts at the two-wave floor when macOS reports pressure; this fixture models a calm host.
        ledger.setUsable(ledger.maximumUsableBytes)
        runner.fitExperts()
        try runner.prepareScratch(["value": Int(page / 4)], footprint: { 0 })
        try autoreleasepool { _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8) }
        let beforeCache = ledger.cacheReliefSnapshot.cacheOwnedBytes
        try require(
            beforeCache == 8 * page && runner.experts.allocatedSlots(layer: 0) == 8,
            "fixture did not allocate optional single-page slabs")
        // Pool slots include the constructor's alignment; raw gate/up/down payload size is not the slot charge.
        // This one-expert-per-token, one-pool fixture must have a two-slot floor and six optional single-page
        // slabs. Drain population's temporary Metal references so a later removal can retire its final owner.
        let slotBytes = runner.experts.leastBytes / 2
        try require(
            slotBytes == page && runner.experts.leastAllocatedBytes == 2 * page,
            "unexpected aligned slot/floor: slot \(slotBytes), floor \(runner.experts.leastAllocatedBytes), page \(page)")
        // Prove the trim-to-owner mapping before injecting lag. A failed population lifetime or geometry
        // precondition must report here, rather than masquerading as a broken physical-admission retry.
        autoreleasepool { runner.experts.setLimit(bytes: 6 * slotBytes) }
        try require(
            runner.experts.allocatedSlots(layer: 0) == 6
                && ledger.cacheReliefSnapshot.cacheOwnedBytes == 6 * page,
            "trim precondition: slots \(runner.experts.allocatedSlots(layer: 0)), owned \(ledger.cacheReliefSnapshot.cacheOwnedBytes), expected \(6 * page)"
        )
        runner.fitExperts()
        try autoreleasepool { _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8) }
        try require(
            runner.experts.allocatedSlots(layer: 0) == 8
                && ledger.cacheReliefSnapshot.cacheOwnedBytes == beforeCache,
            "optional slabs did not regrow after the trim precondition")
        let successfulLag = ledger.ceiling - 4 * page + (beforeCache - 6 * slotBytes)
        let persistentLag = ledger.ceiling - 5 * page + (beforeCache - 5 * slotBytes)
        let before = ledger.reservedBytes
        var probes = 0
        var grewBeforePassingSample = false
        try runner.prepareScratch(
            ["value": Int(page)],
            footprint: {
                probes += 1
                // GPU-backed pages can remain charged beyond the former 31 ms window. Keep the first
                // eight samples high: the replacement must wait for an actual passing sample, not subtract
                // logically retired bytes from a still-high footprint.
                if probes <= 9 && runner.workingAllocatedBytes != page { grewBeforePassingSample = true }
                return probes <= 8 ? successfulLag : 0
            })
        try require(
            probes >= 10 && !grewBeforePassingSample && runner.workingAllocatedBytes == 4 * page,
            "a lagged sample failed mandatory work or allocated before a fresh physical pass")
        try require(
            ledger.cacheReliefSnapshot.cacheOwnedBytes == 6 * page
                && runner.experts.allocatedSlots(layer: 0) == 6
                && ledger.reservedBytes == before + page && runner.workingRetainedBytes == 4 * page,
            "retirement retry hid an owner or trimmed beyond the measured shortfall")
        runner.fitExperts()
        try autoreleasepool { _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8) }
        let old = try runner.work("value", 1)
        old.contents().storeBytes(of: UInt32(0x5566_7788), as: UInt32.self)
        let beforeDenial = ledger.reservedBytes
        probes = 0
        var diagnostic = ""
        do {
            try runner.prepareScratch(
                ["value": Int(5 * page / 4)],
                footprint: {
                    probes += 1
                    return persistentLag
                })
            throw EngineError("Family scratch retirement: persistent physical excess admitted growth", code: "fixture_unexpected")
        } catch let error as EngineError where error.code == "insufficient_memory" { diagnostic = error.description }
        let unchanged = try runner.work("value", 1)
        try require(
            probes == 12 && diagnostic.contains("physical shortfall") && diagnostic.contains("footprint"),
            "persistent deficit did not exhaust exactly the bounded physical rechecks")
        try require(
            unchanged === old && old.contents().load(as: UInt32.self) == 0x5566_7788
                && runner.workingRetainedBytes == 4 * page && runner.workingAllocatedBytes == 4 * page
                && runner.experts.allocatedSlots(layer: 0) == 5
                && ledger.reservedBytes == beforeDenial - 3 * page,
            "physical refusal replaced live scratch, hid a charge or lost proven retirement")
        runner.fitExperts()
        try autoreleasepool { _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8) }
        let beforeCancellation = ledger.reservedBytes
        var cancellationChecks = 0
        probes = 0
        // The request is live immediately before sleeping and cancelled when it wakes. The following physical
        // sample would pass, so cancellation must be rechecked before that successful probe can admit allocation.
        runner.shouldContinue = {
            cancellationChecks += 1
            return cancellationChecks < 2
        }
        do {
            try runner.prepareScratch(
                ["value": Int(5 * page / 4)],
                footprint: {
                    probes += 1
                    return probes <= 2 ? persistentLag : 0
                })
            throw EngineError("Family scratch retirement: cancelled wait admitted growth", code: "fixture_unexpected")
        } catch let error as EngineError where error.code == "cancelled" {}
        runner.shouldContinue = { true }
        let afterCancellation = try runner.work("value", 1)
        try require(
            cancellationChecks == 2 && probes == 2 && afterCancellation === old
                && old.contents().load(as: UInt32.self) == 0x5566_7788
                && runner.experts.allocatedSlots(layer: 0) == 5
                && runner.workingRetainedBytes == 4 * page && ledger.reservedBytes == beforeCancellation - 3 * page,
            "cancellation after the wait allocated scratch or lost a retained owner")
        try runner.prepareScratch(["value": Int(5 * page / 4)], footprint: { 0 })
        try require(
            runner.workingAllocatedBytes == 5 * page && runner.workingRetainedBytes == 9 * page
                && old.contents().load(as: UInt32.self) == 0x5566_7788,
            "a later valid physical sample could not retry while preserving the escaped old buffer")
        // With rows in the conversation the same wait finishes despite a cancel (at most 1.023 s), so a cancelled
        // decode step never throws and resets the conversation.
        runner.fitExperts()
        try autoreleasepool { _ = try runner.experts.prepareWave(layer: 0, maximumExperts: 8) }
        model.restored([1])
        runner.shouldContinue = { false }
        probes = 0
        let waited = Date()
        try runner.prepareScratch(
            ["value": Int(6 * page / 4)],
            footprint: {
                probes += 1
                return probes <= 2 ? persistentLag : 0
            })
        runner.shouldContinue = { true }
        model.restored([])
        try require(
            probes >= 3 && runner.workingAllocatedBytes == 6 * page && Date().timeIntervalSince(waited) < 1.1,
            "a cancel during the wait with live rows threw or did not finish (probes \(probes))")
        withExtendedLifetime(old) {}
    }
    try require(ledger.reservedBytes == 0, "final scratch/cache owners leaked after the request scope")
}
