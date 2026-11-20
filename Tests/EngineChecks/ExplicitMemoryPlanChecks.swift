import Foundation
import OrigamiEngine

/// Explicit targets ignore free-page estimates, include transient overhead, and start small only on real pressure.
/// Simulated host and pressure snapshots keep this check independent of the test machine's load; no Metal is used.
func planExplicitMemoryTarget() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Explicit memory plan: " + what) } }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-explicit-plan-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let model = try ModelSource(fixture: folder, spans: [:])
    let gib: UInt64 = 1 << 30
    let host = OrigamiHost(physicalMemory: 32 * gib, workingSet: 28 * gib, gpuName: "simulated", bfloatMatrices: true)
    func plan(_ limit: UInt64?, pressure: MemoryPressureLevel? = .normal, experts: UInt64? = nil) throws -> OrigamiPlan {
        try OrigamiPlan.make(
            model: model, host: host, context: 8192, memoryLimit: limit, expertMemory: experts,
            promptCache: folder, availableNow: 1, footprintNow: 0, pressureNow: pressure)
    }
    let explicit = try plan(28 * gib)
    let ledger = try explicit.configuration.makeLedger()
    try require(
        explicit.memory.cap == 28 * gib && ledger.ceiling == explicit.memory.cap,
        "chosen process bound excludes its transient margin")
    try require(
        ledger.budgetMode == .explicit && !ledger.usesAvailableMemoryFloor
            && ledger.usableBytes == ledger.maximumUsableBytes && explicit.memory.start == ledger.maximumUsableBytes,
        "low free-memory estimate reduced the explicit target")
    try require(
        ledger.maximumUsableBytes + ledger.transientMargin == explicit.memory.cap,
        "owner and transient allocations exceed the explicit process bound")
    for pressure in [MemoryPressureLevel.warning, .critical] {
        let pressured = try plan(28 * gib, pressure: pressure)
        try require(
            pressured.memory.start == pressured.memory.minimum && pressured.memory.start < ledger.maximumUsableBytes,
            "already elevated pressure did not start at the compulsory minimum")
    }
    let unknown = try plan(28 * gib, pressure: nil)
    try require(unknown.memory.start == explicit.memory.start, "unsupported pressure query reduced the target")
    let automatic = try plan(nil)
    let automaticLedger = try automatic.configuration.makeLedger()
    try require(
        automaticLedger.budgetMode == .automatic && automaticLedger.usesAvailableMemoryFloor
            && automatic.memory.start == automatic.memory.minimum && automatic.memory.start < automatic.memory.cap,
        "automatic busy-host fit changed")
    let capped = try plan(28 * gib, experts: 1_000_000_000)
    let cappedLedger = try capped.configuration.makeLedger()
    try require(
        cappedLedger.ceiling <= explicit.memory.cap && cappedLedger.maximumUsableBytes >= capped.memory.minimum
            && capped.configuration.experts.cacheSlots < explicit.configuration.experts.cacheSlots,
        "explicit expert cap or compulsory cache floor lost")
    // Test the physical next-allocation guard, including exact equality and overflow, without sampling this process.
    try require(
        ledger.physicalAllocationShortfall(bytes: 16, footprint: ledger.ceiling - 16) == 0
            && ledger.physicalAllocationShortfall(bytes: 17, footprint: ledger.ceiling - 16) == 1
            && ledger.physicalAllocationShortfall(bytes: .max, footprint: 1) == .max
            && automaticLedger.physicalAllocationShortfall(bytes: .max, footprint: .max) == 0,
        "physical allocation bounds or automatic compatibility")
}

/// The shared reclamation wait (family scratch and Qwen image admission): stops at the first satisfied re-probe,
/// gives up after the bounded schedule and throws on cancellation before probing again.
func reclamationWaitIsBoundedAndCancellable() throws {
    func require(_ ok: Bool, _ message: String) throws { if !ok { throw EngineError("Reclamation wait: " + message) } }
    var slept: [Double] = []
    var checks = 0
    try ReclamationWait.wait(shouldContinue: { true }, sleep: { slept.append($0) }) {
        checks += 1
        return checks >= 3
    }
    try require(slept == Array(ReclamationWait.delays.prefix(3)), "did not stop once memory was reclaimed: \(slept)")
    slept = []
    try ReclamationWait.wait(shouldContinue: { true }, sleep: { slept.append($0) }) { false }
    try require(slept == ReclamationWait.delays && abs(slept.reduce(0, +) - 1.023) < 1e-9, "unbounded or wrong schedule")
    var calls = 0
    var probes = 0
    do {
        try ReclamationWait.wait(
            shouldContinue: {
                calls += 1
                return calls < 2
            }, sleep: { _ in }
        ) {
            probes += 1
            return true
        }
        throw EngineError("Reclamation wait: cancellation ignored")
    } catch let error as EngineError where error.code == "cancelled" {}
    try require(probes == 0, "probed or admitted after cancellation")
}
