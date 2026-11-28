import Foundation
import Metal
import OrigamiEngine

/// Tiny allocations exercise the real admission boundary; supplied process footprints keep pressure deterministic.
/// No large model, disk payload, timing threshold or host-pressure manipulation is needed.
func familyScratchAdmission() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family scratch admission: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-admission-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 64)
    let page = try FamilyRunner.scratchLength(1)
    let ceiling = UInt64(8 * page)
    runner.ledger = try MemoryLedger(ceiling: ceiling, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    var probes = 0
    func footprint() -> UInt64 {
        probes += 1
        return 0
    }
    do {
        try runner.prepareScratch(["denied": page / 4], allowance: UInt64(page - 1), footprint: footprint)
        throw EngineError("Family scratch admission: low working allowance admitted growth")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    try require(probes == 0 && runner.workingAllocatedBytes == 0, "owner denial probed or allocated")
    do {
        try runner.prepareScratch(
            ["denied": page / 4],
            footprint: {
                probes += 1
                return ceiling
            })
        throw EngineError("Family scratch admission: physical overrun admitted growth")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    try require(probes > 0 && runner.workingAllocatedBytes == 0, "physical denial published a buffer")
    for count in [-1, Int.max] {
        do {
            _ = try FamilyRunner.scratchLength(count)
            throw EngineError("Family scratch admission: invalid capacity accepted")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
    }
    try runner.prepareScratch(["kept": page / 4], footprint: footprint)
    let allocated = runner.workingAllocatedBytes
    do {
        _ = try runner.work("unplanned", 1)
        throw EngineError("Family scratch admission: unplanned growth succeeded")
    } catch let error as EngineError where error.code == "scratch_plan_mismatch" {}
    try require(runner.workingAllocatedBytes == allocated, "unplanned lookup changed ownership")
    runner.begin()
    do {
        try runner.prepareScratch(["during-encoding": 1], footprint: footprint)
        throw EngineError("Family scratch admission: grew inside a caller-held encoder")
    } catch let error as EngineError where error.code == "scratch_plan_mismatch" {}
    runner.end()
    try require(runner.workingAllocatedBytes == allocated, "in-encoder denial allocated")
}

/// A replacement charges old + new while they coexist, and retires dead scratch only at the completed boundary.
/// Reusing any admitted capacity performs neither an allocation/physical probe nor a command-buffer flush.
func familyScratchReplacementReuse() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family scratch replacement: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-reuse-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 64)
    let page = try FamilyRunner.scratchLength(1)
    runner.ledger = try MemoryLedger(ceiling: UInt64(16 * page), transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    var probes = 0
    func footprint() -> UInt64 {
        probes += 1
        return 0
    }
    try runner.prepareScratch(["value": page / 4], footprint: footprint)
    weak var retired: (any MTLBuffer)?
    try autoreleasepool {
        let old = try runner.work("value", 1)
        old.contents().storeBytes(of: UInt32(0x7654_3210), as: UInt32.self)
        retired = old
        let before = probes
        do {
            try runner.prepareScratch(["value": page / 2], allowance: UInt64(page), footprint: footprint)
            throw EngineError("Family scratch replacement: replacement exceeded the working owner")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        let unchanged = try runner.work("value", 1)
        try require(
            unchanged === old && unchanged.contents().load(as: UInt32.self) == 0x7654_3210 && probes == before,
            "denied replacement changed the old buffer or probed")
        do {
            // With room for coexistence, a physical denial must count the complete new allocation, not its delta.
            let ceiling = runner.ledger!.ceiling
            try runner.prepareScratch(
                ["value": page / 2], allowance: UInt64(3 * page),
                footprint: {
                    probes += 1
                    return ceiling - UInt64(page)
                })
            throw EngineError("Family scratch replacement: physical admission used only the size difference")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try require(try runner.work("value", 1) === old, "physical denial retired a buffer which could coexist")
        try runner.prepareScratch(["value": page / 2], allowance: UInt64(3 * page), footprint: footprint)
        let replacement = try runner.work("value", page / 2)
        try require(replacement !== old && runner.workingAllocatedBytes == UInt64(2 * page), "replacement was not published")
        try require(old.contents().load(as: UInt32.self) == 0x7654_3210, "old buffer was mutated before retirement")
    }
    try require(retired == nil, "old buffer remained retained after boundary publication")
    // Near the owner limit, only final storage fits. Scratch contents are dead: retire before probing/allocating,
    // instead of rejecting legitimate growth or borrowing uncharged space for old + new.
    autoreleasepool { retired = try? runner.work("value", 1) }
    var retiredBeforeProbe = false
    try runner.prepareScratch(
        ["value": 3 * page / 4], allowance: UInt64(3 * page),
        footprint: {
            probes += 1
            retiredBeforeProbe = retired == nil
            return 0
        })
    try require(
        retiredBeforeProbe && runner.workingAllocatedBytes == UInt64(3 * page), "near-cap growth did not retire scratch before admission")
    let before = probes
    let commits = runner.timing.commits
    let bytes = runner.workingAllocatedBytes
    for count in [1, page / 2, 3 * page / 4] {
        try runner.prepareScratch(["value": count], footprint: footprint)
        _ = try runner.work("value", count)
    }
    try require(
        probes == before && runner.timing.commits == commits && runner.workingAllocatedBytes == bytes,
        "hot reuse probed, flushed or changed capacity")
}

/// Model plans use actual row/chunk geometry. Decode does not materialize a prompt or the full context's scores;
/// a multi-chunk forward then exercises every scratch lookup in the tiny pooled-attention family fixtures.
func familyScratchPlanBounds() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family scratch plan: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-plans-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for family in [ModelFamily.deepseek, .glm] {
        let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(family, in: folder)), device: device)
        let model = try makeFamilyModel(weights, expertBudget: 1 << 20, context: 8192) { _ in }
        let runner = model as! FamilyRunner
        let decode = runner.scratchPlan(rows: 1)
        let prompt = runner.scratchPlan(rows: 269)
        try require(prompt["streams-a"] == 269 * decode["streams-a"]!, "\(family): streams do not use actual rows")
        let indexName = family == .deepseek ? "ds-iscores" : "idx-scores"
        try require((decode[indexName] ?? Int.max) <= 256, "\(family): decode planned the full context")
        let small = try decode.values.reduce(UInt64(0)) { $0 + UInt64(try FamilyRunner.scratchLength($1)) }
        let large = try prompt.values.reduce(UInt64(0)) { $0 + UInt64(try FamilyRunner.scratchLength($1)) }
        try require(small < large && large < FamilyRunner.baseWorkingBytes, "\(family): row plan exceeds the prompt phase's working limit")
        let result = try model.forward((0..<269).map { 1 + $0 % 7 })
        try require(result.allSatisfy(\.isFinite), "\(family): planned grouped forward was not finite")
        runner.beginDecode()
        _ = try model.forward([9])
        try require(runner.workingAllocatedBytes < large, "\(family): decode retained prompt capacities")
    }
}
