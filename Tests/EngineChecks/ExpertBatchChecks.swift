import Darwin
import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func batchRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func batchReject(_ code: String? = nil, _ body: () throws -> Void) throws {
    do { try body() } catch let error as EngineError {
        if let code { try batchRequire(error.code == code, "Expected \(code), received \(String(describing: error.code)): \(error)") }
        return
    }
    throw EngineError("Invalid expert batch unexpectedly succeeded")
}
private func batchDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}
private func batchInputs(_ rows: Int, salt: Int = 0) -> [Float] {
    (0..<(rows * 2560)).map { index in
        let numerator = (index * 13 + index / 2560 * 7 + salt * 5) % 31 - 15
        return BF16.decode(BF16.encode(Float(numerator) / 32))
    }
}
private func batchRecord(_ expert: Int, _ destination: UnsafeMutableRawBufferPointer) {
    for kind in ExpertProjection.allCases {
        let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
        let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
        let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
        for index in 0..<(819_200 / 4) { words[index] = 0x7654_3210 ^ UInt32((index + expert) & 15) }
        for index in 0..<(51_200 / 2) {
            scales[index] = BF16.encode(Float((index + expert) % 7 + 1) / 128)
            biases[index] = BF16.encode(Float((index + expert) % 9 - 4) / 64)
        }
    }
}
private func batchEngine(_ ledger: MemoryLedger, slots: Int = 8, window: Int = 4, failureExpert: Int? = nil) throws -> ExpertExecution {
    let configuration = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: slots,
        slotsPerSlab: 2, demandWindow: window, readConcurrency: min(window, 2))
    return try ExpertExecution(device: batchDevice(), ledger: ledger, configuration: configuration, packedAvailable: { false }) {
        key, _, destination in
        if key.expert == failureExpert { throw EngineError("Injected batch read failure") }
        batchRecord(key.expert, destination)
    }
}
private func batchRetired(_ engine: ExpertExecution, _ ledger: MemoryLedger) throws {
    let snapshot = engine.snapshot()
    try batchRequire(
        snapshot.cache.pinnedSlots == 0 && snapshot.cache.inFlightReads == 0 && snapshot.cache.reservedSlots == 0
            && snapshot.io.drives.allSatisfy { $0.outstanding == 0 }, "Expert batch retained a producer or cache pin")
    try batchRequire(
        ledger.snapshot().categories["fused-expert-workspace"] == nil && ledger.snapshot().categories["fused-expert-batch-metadata"] == nil
            && ledger.snapshot().categories["expert-consumer-batch-metadata"] == nil,
        "Expert batch retained command/window workspace or metadata")
}

func expertCommandBatchConfigurationBounds() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-expert-batch-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("runtime.json")
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    try batchRequire(try load([:]).expertCommandBatch == 1, "Default configuration changed expert submission grouping")
    let valid: [String: Any] = [
        "expert_mode": "slabPackedPreferred", "fused_experts": true,
        "expert_cache_slots": 4, "expert_slots_per_slab": 2, "expert_window": 2, "io_queue_depth": 2, "expert_command_batch": 8,
    ]
    try batchRequire(try load(valid).expertCommandBatch == 8, "Command grouping was incorrectly clamped to record-window size")
    for invalid in [0, 9, true, "2"] as [Any] {
        var changed = valid
        changed["expert_command_batch"] = invalid
        try batchReject { _ = try load(changed) }
    }
    var unfused = valid
    unfused["fused_experts"] = false
    try batchReject("invalid_configuration") { _ = try load(unfused) }
    var legacy = valid
    legacy["expert_mode"] = "legacyOriginal"
    try batchReject { _ = try load(legacy) }
}

func expertCommandBatchPreservesMixedSchedulesAndTiles() throws {
    let ledger = try MemoryLedger(ceiling: 250_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        let engine = try batchEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1, 2])
        try window.consumeBatch(experts: [0, 1, 2]) { buffers in
            try batchRequire(
                buffers[1].offset > 0 && buffers[2].offset == 0, "Batch fixture did not span nonzero offsets and distinct slabs")
            let cases: [(Bool, Bool, Int)] = [
                (false, false, 1), (false, true, 11), (false, false, 256),
                (true, false, 1), (true, true, 7), (true, true, 9), (true, true, 256), (true, false, 7),
            ]
            let jobs = cases.enumerated().map { index, item in
                FusedExpertExecution.Job(
                    input: batchInputs(item.2, salt: index), buffer: buffers[index % buffers.count],
                    groupedExpert: item.0, simdMatrix: item.1)
            }
            let expected = try jobs.map { job in
                try ExpertExecution.evaluate(
                    job.input, buffer: job.buffer, kernels: kernels,
                    groupedExpert: job.groupedExpert, simdMatrix: job.simdMatrix)
            }
            let before = fused.snapshot()
            var visited: [Int] = []
            try fused.evaluateBatch(jobs) { index, output in
                visited.append(index)
                try batchRequire(
                    output.map(\.bitPattern) == expected[index].map(\.bitPattern),
                    "Expert command batch changed job \(index)'s projection schedule or output tile")
                try batchRequire(engine.snapshot().cache.pinnedSlots == 3, "A record pin retired before all batch outputs were delivered")
            }
            let after = fused.snapshot()
            try batchRequire(
                visited == Array(0..<8) && after.submittedCommands == before.submittedCommands + 1
                    && after.completedCommands == before.completedCommands + 1 && after.submittedJobs == before.submittedJobs + 8,
                "Eight jobs did not execute and join as one command in job order")
            try batchRequire(
                after.submittedRows == before.submittedRows + UInt64(cases.reduce(0) { $0 + $1.2 })
                    && after.gpuTimedCommands <= after.submittedCommands && after.gpuSeconds.isFinite && after.hostJoinSeconds.isFinite
                    && after.gpuSeconds >= before.gpuSeconds && after.hostJoinSeconds >= before.hostJoinSeconds,
                "Command batch timing/row totals are invalid")
        }
        try batchRetired(engine, ledger)
        try batchRequire(engine.snapshot().consumedExperts == 3 && window.isClosed, "GPU job count leaked into consumed expert count")
        try engine.finishSequence()
    }
    // Scheduler callback ARC cleanup may follow ticket completion briefly;
    // physical resources must stay charged until that final reference retires.
    for _ in 0..<1000 {
        if ledger.snapshot().reserved == 0 { break }
        Thread.sleep(forTimeInterval: 0.001)
    }
    try batchRequire(ledger.snapshot().reserved == 0, "Joined batch resources outlived their owner/autorelease pool")
}

func expertCommandBatchWindowOrderBoundsAndRefill() throws {
    let ledger = try MemoryLedger(ceiling: 80_000_000, transientMargin: 1_000_000)
    let engine = try batchEngine(ledger, slots: 4, window: 4)
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 1, experts: Array((0..<10).reversed()))
    for invalid in [[], [1, 0], [0, 0], [0, 2], [0, 1, 2, 3, 4]] {
        try batchReject { try window.consumeBatch(experts: invalid) { _ in throw EngineError("Invalid batch entered its body") } }
        try batchRequire(
            !window.isClosed && window.remainingCount == 10 && engine.snapshot().consumedExperts == 0,
            "Invalid expert batch mutated the consumption frontier")
    }
    for start in stride(from: 0, to: 10, by: 3) {
        let ids = Array(start..<min(10, start + 3))
        try window.consumeBatch(experts: ids) { buffers in
            try batchRequire(
                buffers.count == ids.count && engine.snapshot().cache.pinnedSlots <= 4,
                "Batch pin count exceeded its demand window")
            for (expert, buffer) in zip(ids, buffers) {
                try batchRequire(buffer.bytes[0] == UInt8(0x10 ^ expert), "Sorted expert batch supplied the wrong record")
            }
            try batchReject("expert_phase_busy") { try engine.beginDecodePhase() }
            try batchReject { _ = try window.consume(expert: ids[0]) { $0.bytes[0] } }
        }
        try batchRequire(
            engine.snapshot().consumedExperts == UInt64(start + ids.count),
            "Batch consumed counters do not match the committed union prefix")
    }
    try batchRequire(
        window.isClosed && window.remainingCount == 0 && engine.snapshot().demandedExperts == 10,
        "Final expert batch left pending union entries")
    try batchRetired(engine, ledger)
    try engine.beginDecodePhase()
    try engine.finishSequence()
}

func expertCommandBatchCancellationAndCallbackFailureRetirePins() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger)
        let engine = try batchEngine(ledger, slots: 2, window: 2)
        for allowed in [0, 1, 2] {
            try engine.beginSequence()
            let ids = [allowed * 2, allowed * 2 + 1]
            let window = try engine.makeWindow(layer: 2, experts: ids)
            var checks = 0
            var delivered = 0
            let before = fused.snapshot()
            do {
                try window.consumeBatch(experts: ids) { buffers in
                    let jobs = buffers.enumerated().map {
                        FusedExpertExecution.Job(input: batchInputs(7, salt: $0.offset), buffer: $0.element)
                    }
                    try fused.evaluateBatch(
                        jobs,
                        shouldContinue: {
                            checks += 1
                            return checks <= allowed
                        }
                    ) { _, _ in delivered += 1 }
                }
                throw EngineError("Cancelled command batch returned")
            } catch is IOOperationCancelled {}
            let after = fused.snapshot()
            let expectedSubmitted = UInt64(allowed == 2 ? 1 : 0)
            try batchRequire(
                checks == allowed + 1 && delivered == 0 && window.isClosed
                    && after.submittedCommands == before.submittedCommands + expectedSubmitted
                    && after.completedCommands == before.completedCommands + expectedSubmitted,
                "Cancellation did not exercise the expected pre/post-commit join boundary")
            try batchRetired(engine, ledger)
            try engine.finishSequence()
        }
        try engine.beginSequence()
        let failing = try engine.makeWindow(layer: 2, experts: [8, 9])
        var callbacks = 0
        do {
            try failing.consumeBatch(experts: [8, 9]) { buffers in
                try fused.evaluateBatch(buffers.map { FusedExpertExecution.Job(input: batchInputs(1), buffer: $0) }) { _, _ in
                    callbacks += 1
                    throw EngineError("Injected batch scatter failure")
                }
            }
            throw EngineError("Failed scatter returned")
        } catch let error as EngineError { try batchRequire(error.description == "Injected batch scatter failure", "Wrong scatter error") }
        try batchRequire(
            callbacks == 1 && failing.isClosed && engine.snapshot().consumedExperts == 0,
            "Partial scatter published expert-consumption counters")
        try batchRetired(engine, ledger)
        try engine.finishSequence()
    }
}

func expertCommandBatchPartialJoinAndAdmissionFailureCleanup() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let failed = try batchEngine(ledger, slots: 4, window: 3, failureExpert: 1)
    try failed.beginSequence()
    let partial = try failed.makeWindow(layer: 3, experts: [0, 1, 2])
    var entered = false
    try batchReject { try partial.consumeBatch(experts: [0, 1, 2]) { _ in entered = true } }
    try batchRequire(
        !entered && partial.isClosed && failed.snapshot().consumedExperts == 0,
        "Failed second producer reached a batch body or committed consumption")
    try batchRetired(failed, ledger)
    try failed.finishSequence()

    let engine = try batchEngine(ledger, slots: 2, window: 2)
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 4, experts: [3, 4])
    let page = UInt64(getpagesize())
    let tableBytes: UInt64 = 131_072
    let metadataBytes: UInt64 = 4096 + 2 * 1024
    let small = try MemoryLedger(ceiling: tableBytes + metadataBytes + 6 * page + 1, transientMargin: 1)
    try autoreleasepool {
        let denied = try FusedExpertExecution(arithmetic: .hostReference, ledger: small)
        try batchReject("insufficient_memory") {
            try window.consumeBatch(experts: [3, 4]) { buffers in
                try denied.evaluateBatch(buffers.map { FusedExpertExecution.Job(input: batchInputs(1), buffer: $0) }) { _, _ in
                    throw EngineError("Denied batch published output")
                }
            }
        }
        try batchRequire(
            denied.snapshot().submittedCommands == 0 && window.isClosed,
            "Partial second-job allocation submitted an incomplete command")
        try batchRequire(small.snapshot().reserved == tableBytes, "Denied command batch leaked private buffers/metadata")
    }
    try batchRequire(small.snapshot().reserved == 0, "Denied batch helper leaked its table")
    try batchRetired(engine, ledger)
    try engine.finishSequence()

    // Also deny host batch metadata before any handle is removed. Closing the
    // pending window still joins producers and drops every handoff pin.
    try engine.beginSequence()
    let metadataWindow = try engine.makeWindow(layer: 4, experts: [5, 6])
    let deadline = Date().addingTimeInterval(2)
    while engine.snapshot().io.drives.contains(where: { $0.outstanding != 0 }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.001)
    }
    try batchRequire(engine.snapshot().io.drives.allSatisfy { $0.outstanding == 0 }, "Fixture producers did not complete")
    let snapshot = ledger.snapshot()
    let deniedBytes = snapshot.ceiling - snapshot.transientMargin - snapshot.reserved
    let reservation = try ledger.reserve(bytes: deniedBytes, category: "batch-metadata-denial-fixture")
    do {
        try batchReject("insufficient_memory") { try metadataWindow.consumeBatch(experts: [5, 6]) { _ in entered = true } }
    } catch {
        try? ledger.release(reservation)
        throw error
    }
    try ledger.release(reservation)
    try batchRequire(!entered && metadataWindow.isClosed, "Denied batch metadata mutated consumer execution")
    try batchRetired(engine, ledger)
    try engine.finishSequence()
}

func expertCommandBatchInvalidJobsAllocateNothing() throws {
    let ledger = try MemoryLedger(ceiling: 80_000_000, transientMargin: 1_000_000)
    let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger)
    let engine = try batchEngine(ledger, slots: 2, window: 2)
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 5, experts: [0, 1])
    try window.consumeBatch(experts: [0, 1]) { buffers in
        let good = FusedExpertExecution.Job(input: batchInputs(1), buffer: buffers[0])
        for jobs in [
            [], Array(repeating: good, count: 9),
            [good, FusedExpertExecution.Job(input: [], buffer: buffers[1])],
            [good, FusedExpertExecution.Job(input: [0], buffer: buffers[1])],
            [good, FusedExpertExecution.Job(input: batchInputs(257), buffer: buffers[1])],
        ] {
            try batchReject { try fused.evaluateBatch(jobs) { _, _ in throw EngineError("Invalid batch delivered output") } }
            try batchRequire(
                fused.snapshot().submittedCommands == 0 && ledger.snapshot().categories["fused-expert-workspace"] == nil
                    && ledger.snapshot().categories["fused-expert-batch-metadata"] == nil,
                "Invalid job batch submitted work or allocated private buffers")
        }
    }
    try batchRetired(engine, ledger)
    try engine.finishSequence()
}
