import Foundation
import Metal
import OrigamiEngine
import Synchronization

func demandReadLaneProgress() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError("Demand lanes: " + message) } }
    final class State: @unchecked Sendable {
        let lock = NSLock(), mainStarted = DispatchSemaphore(value: 0), releaseMain = DispatchSemaphore(value: 0)
        let hotFinished = DispatchSemaphore(value: 0), failed = Atomic<Bool>(false)
        var visits: [Int], active = [0, 0], maximum = [0, 0], finished = [0, 0]
        init(_ count: Int) { visits = Array(repeating: 0, count: count) }
        func enter(_ job: Int, _ source: Int) {
            lock.lock()
            defer { lock.unlock() }
            visits[job] += 1
            active[source] += 1
            maximum[source] = max(maximum[source], active[source])
        }
        func leave(_ source: Int) {
            lock.lock()
            defer { lock.unlock() }
            active[source] -= 1
            finished[source] += 1
            if source == 1 && finished[source] == 3 { hotFinished.signal() }
        }
    }
    // Hold both main workers. All three hot jobs must still finish before those workers are released.
    // This proves independent progress and bounded worker ownership, not a timing-based speed assertion.
    let held = try familyCheckScheduler(jobs: 10, lanes: (2, 1))
    for spin: Double? in [nil, 0.001] {
        let state = State(10)
        let sources = Array(repeating: false, count: 7) + Array(repeating: true, count: 3)
        var submitted = 0
        var meanwhile = 0
        try FamilyExperts.perform(
            sources.count, on: held, source: { sources[$0] ? .internalSSD : .externalSSD }, spin: spin,
            submitted: { submitted += 1 },
            meanwhile: {
                meanwhile += 1
                for _ in 0..<2 {
                    if state.mainStarted.wait(timeout: .now() + 5) == .timedOut { state.failed.store(true, ordering: .releasing) }
                }
                if state.hotFinished.wait(timeout: .now() + 5) == .timedOut { state.failed.store(true, ordering: .releasing) }
                for _ in 0..<7 { state.releaseMain.signal() }
            }
        ) { job in
            let source = sources[job] ? 1 : 0
            state.enter(job, source)
            if source == 0 {
                state.mainStarted.signal()
                if state.releaseMain.wait(timeout: .now() + 5) == .timedOut { state.failed.store(true, ordering: .releasing) }
            }
            state.leave(source)
        }
        try require(
            !state.failed.load(ordering: .acquiring) && submitted == 1 && meanwhile == 1,
            "blocked main reads starved hot reads or callback count changed")
        try require(
            state.visits.allSatisfy { $0 == 1 } && state.finished == [7, 3] && state.active == [0, 0],
            "missing, duplicate or unjoined work")
        try require(state.maximum == [2, 1], "worker limits were not enforced")
    }
}

/// Real tiny GGUF and hot-copy reads: compare every returned byte, then fail a source read and recover.
/// No model inference is needed to test the scheduler's publication and buffer-ownership contract.
func demandReadLaneMixedSources() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError("Mixed demand lanes: " + message) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-demand-lanes-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try expertFixtureGGUF(in: folder)
    let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
    func pool() throws -> FamilyExperts { try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688))) }
    let copy = try HotCopy.write(source: pool(), keys: [6], folder: folder.appendingPathComponent("hot"), identity: "lanes")
    let saved = (FamilyExperts.checkReadLimits, FamilyExperts.spins)
    defer { (FamilyExperts.checkReadLimits, FamilyExperts.spins) = saved }
    FamilyExperts.spins = false
    let partBytes = GGUFQuantize.bytes(8, count: 256 * 256)
    func verify(_ slots: [FamilyExperts.Slot]) throws {
        try require(slots.count == 4, "missing records")
        for (expert, slot) in slots.enumerated() {
            for part in 0..<3 {
                let data = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: partBytes)
                try require(data.allSatisfy { $0 == UInt8(50 + expert * 10 + part) }, "incomplete or changed record")
            }
        }
    }
    for limits: (main: Int, hot: Int)? in [nil, (2, 1)] {
        FamilyExperts.checkReadLimits = limits
        let experts = try pool()
        experts.attach(copy)
        try verify(experts.fetch(layer: 1, experts: [0, 1, 2, 3], decode: false))
        try require(
            experts.counters.demandHotJobs == 1 && experts.counters.demandMainJobs == 9, "mixed-source path or counters were not exercised")
        let before = experts.counters.readBytes
        try verify(experts.fetch(layer: 1, experts: [0, 1, 2, 3], decode: false))
        try require(experts.counters.readBytes == before, "completed records were not retained")
    }
    let failed = try pool()
    failed.attach(copy)
    let original = try Data(contentsOf: path)
    let writer = try FileHandle(forWritingTo: path)
    defer { try? writer.close() }
    try writer.truncate(atOffset: 0)
    var rejected = false
    do { _ = try failed.fetch(layer: 1, experts: [0, 1, 2, 3], decode: false) } catch { rejected = true }
    try writer.seek(toOffset: 0)
    try writer.write(contentsOf: original)
    try require(rejected && failed.resident(layer: 1).isEmpty, "failed batch published partial records")
    try verify(failed.fetch(layer: 1, experts: [0, 1, 2, 3], decode: false))
}
