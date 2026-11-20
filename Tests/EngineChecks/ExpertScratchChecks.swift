import Darwin
import Foundation
import OrigamiEngine

private func scratchRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func scratchReject(_ body: () throws -> Void, code: String? = nil) throws {
    do { try body() } catch let error as EngineError {
        if let code { try scratchRequire(error.code == code, "Expert scratch error classification changed") }
        return
    }
    throw EngineError("Invalid expert scratch operation succeeded")
}
private struct ScratchInjectedFailure: Error {}
func expertScratchConfiguration() throws {
    let legacy = Data(
        #"{"submittedCommands":2,"completedCommands":2,"submittedJobs":10,"submittedRows":10,"gpuTimedCommands":2,"gpuSeconds":0.001,"hostJoinSeconds":0.002}"#
            .utf8)
    let decoded = try JSONDecoder().decode(FusedExpertExecution.Snapshot.self, from: legacy)
    try scratchRequire(
        decoded.submittedJobs == 10 && !decoded.scratch.enabled && decoded.scratch.retainedBytes == 0 && decoded.scratch.capacityBytes == 0,
        "Old expert telemetry no longer decodes as pool-disabled")
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("origami-scratch-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: file) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: file)
        return try RuntimeConfiguration.load(file)
    }
    try scratchRequire(try !load([:]).expertScratchReuse, "Expert scratch must default off")
    let valid: [String: Any] = ["expert_mode": "slabPackedPreferred", "fused_experts": true, "expert_scratch_reuse": true]
    try scratchRequire(try load(valid).expertScratchReuse, "Expert scratch flag lost")
    try scratchReject { _ = try load(["expert_scratch_reuse": true]) }
    for value in [1, "true"] as [Any] {
        var input = valid
        input["expert_scratch_reuse"] = value
        try scratchReject { _ = try load(input) }
    }
}
func expertScratchReuseAndRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let baseline = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        let pooled = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let engine = try groupedTiledEngine(ledger)
        let page = UInt64(getpagesize())
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1])
        try window.consumeBatch(experts: [0, 1]) { buffers in
            for pass in 0..<2 {
                let jobs = (0..<8).map { index in
                    FusedExpertExecution.Job(input: groupedTiledInput(1, salt: pass * 8 + index), buffer: buffers[index % 2])
                }
                var expected: [[UInt32]] = []
                var stages: [[String: [UInt32]]] = []
                for job in jobs {
                    var captured: [String: [UInt32]] = [:]
                    let output = try baseline.evaluate(job.input, buffer: job.buffer, capture: { captured[$0] = $1.map(\.bitPattern) })
                    expected.append(output.map(\.bitPattern))
                    stages.append(captured)
                }
                var equal = true
                var count = 0
                let before = pooled.snapshot()
                try pooled.evaluateBatch(
                    jobs,
                    capture: { index, name, values in
                        equal = equal && values.map(\.bitPattern) == stages[index][name]
                    },
                    consumeOutput: { index, values in
                        try scratchRequire(values.map(\.bitPattern) == expected[index], "Expert scratch changed cold/warm output")
                        // All forty buffers remain leased through every callback.
                        try scratchRequire(pooled.snapshot().scratch.leasedBuffers == 40, "Expert scratch retired before output callback")
                        if index == 0 {
                            let held = pooled.snapshot().scratch.retainedBytes
                            try scratchReject({ try pooled.trimScratch() }, code: "expert_scratch_busy")
                            try scratchReject(
                                { _ = try pooled.evaluate(jobs[0].input, buffer: jobs[0].buffer) }, code: "expert_scratch_busy")
                            try scratchRequire(pooled.snapshot().scratch.retainedBytes == held, "Busy scratch rejection mutated its pool")
                        }
                        count += 1
                    })
                let after = pooled.snapshot()
                let scratch = after.scratch
                let roundtrip = try JSONDecoder().decode(FusedExpertExecution.Snapshot.self, from: JSONEncoder().encode(after))
                try scratchRequire(
                    roundtrip.scratch.enabled && roundtrip.scratch.hits == scratch.hits
                        && roundtrip.scratch.retainedBytes == scratch.retainedBytes,
                    "New expert scratch telemetry failed its Codable round trip")
                try scratchRequire(
                    equal && count == 8 && after.completedCommands == before.completedCommands + 1 && scratch.misses == 40
                        && scratch.hits == UInt64(pass * 40) && scratch.retainedBuffers == 40 && scratch.leasedBuffers == 0
                        && scratch.retainedBytes == 40 * page && scratch.capacityBytes == 40 * page,
                    "Expert scratch reuse counts/cap/single-command retirement changed")
                try scratchRequire(
                    ledger.snapshot().categories["fused-expert-scratch-retained"] == 40 * page
                        && ledger.snapshot().categories["fused-expert-workspace"] == nil, "Retained scratch lost its ledger owner")
            }
            let before = pooled.snapshot().scratch
            for (rows, grouped) in [(2, false), (1, true)] {
                let input = groupedTiledInput(rows)
                let a = try baseline.evaluate(input, buffer: buffers[1], groupedExpert: grouped)
                let b = try pooled.evaluate(input, buffer: buffers[1], groupedExpert: grouped)
                try scratchRequire(a.map(\.bitPattern) == b.map(\.bitPattern), "Scratch flag changed an ineligible job")
            }
            try scratchRequire(
                pooled.snapshot().scratch.hits == before.hits && pooled.snapshot().scratch.misses == before.misses,
                "Nondecode job borrowed tiny expert scratch")
        }
        try engine.finishSequence()
        try pooled.trimScratch()
        try scratchRequire(
            pooled.snapshot().scratch.retainedBytes == 0 && ledger.snapshot().categories["fused-expert-scratch-retained"] == nil,
            "Expert scratch trim retained Metal storage")
    }
    try scratchRequire(ledger.snapshot().reserved == 0, "Expert scratch helper destruction retained resources")
}
func expertScratchCancellationAndAdmission() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let pooled = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger, scratchReuse: true)
        let engine = try groupedTiledEngine(ledger)
        for allowed in [0, 1, 2] {
            try engine.beginSequence()
            let window = try engine.makeWindow(layer: 0, experts: [0])
            var calls = 0
            do {
                try window.consume(expert: 0) { buffer in
                    _ = try pooled.evaluate(
                        groupedTiledInput(1), buffer: buffer,
                        shouldContinue: {
                            calls += 1
                            return calls <= allowed
                        })
                }
                throw EngineError("Expert scratch cancellation succeeded")
            } catch is IOOperationCancelled {}
            let stats = pooled.snapshot()
            try scratchRequire(
                calls == allowed + 1 && stats.completedCommands == stats.submittedCommands && stats.scratch.retainedBuffers == 0
                    && stats.scratch.leasedBuffers == 0 && window.isClosed && engine.snapshot().cache.pinnedSlots == 0
                    && ledger.snapshot().categories["fused-expert-scratch-retained"] == nil,
                "Expert scratch cancellation leaked new pages or returned before join/pin retirement")
            try engine.finishSequence()
        }
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0])
        try window.consume(expert: 0) { buffer in
            let before = pooled.snapshot()
            let held = try ledger.reserve(
                bytes: 99_000_000 - ledger.snapshot().reserved - 5_120 - 2 * UInt64(getpagesize()),
                category: "scratch-fixture-partial-denial")
            do {
                defer { try? ledger.release(held) }
                try scratchReject({ _ = try pooled.evaluate(groupedTiledInput(1), buffer: buffer) }, code: "insufficient_memory")
            }
            let after = pooled.snapshot()
            try scratchRequire(
                after.submittedCommands == before.submittedCommands && after.scratch.misses == before.scratch.misses + 2
                    && after.scratch.retainedBytes == 0 && after.scratch.leasedBuffers == 0,
                "Expert scratch partial admission did not retire its two admitted pages")
            // Warm five pages, then throw from delivery of an eight-job command.
            // Only newly admitted pages may retire; the previous warm five remain.
            _ = try pooled.evaluate(groupedTiledInput(1), buffer: buffer)
            let warm = pooled.snapshot()
            let reserved = ledger.snapshot().reserved
            do {
                let jobs = (0..<8).map { FusedExpertExecution.Job(input: groupedTiledInput(1, salt: $0), buffer: buffer) }
                try pooled.evaluateBatch(jobs, consumeOutput: { _, _ in throw ScratchInjectedFailure() })
                throw EngineError("Injected scratch callback failure was swallowed")
            } catch is ScratchInjectedFailure {}
            let failed = pooled.snapshot()
            try scratchRequire(
                failed.completedCommands == warm.completedCommands + 1 && failed.completedCommands == failed.submittedCommands
                    && failed.scratch.retainedBuffers == 5 && failed.scratch.leasedBuffers == 0 && ledger.snapshot().reserved == reserved,
                "Scratch callback failure changed warm storage or leaked new buffers")
            try pooled.trimScratch()
            try scratchRequire(pooled.snapshot().scratch.retainedBytes == 0, "Explicit trim after failure did not retire")
        }
        try engine.finishSequence()
    }
    try scratchRequire(ledger.snapshot().reserved == 0, "Expert scratch failures outlived their owners")
}
