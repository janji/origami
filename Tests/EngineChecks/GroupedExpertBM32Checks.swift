import Foundation
import OrigamiEngine

private func grouped32Require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func grouped32Reject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid grouped BM32 configuration accepted")
}
func groupedBM32Configuration() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("origami-grouped-bm32-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: file) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: file)
        return try RuntimeConfiguration.load(file)
    }
    let defaults = try load([:])
    try grouped32Require(!defaults.groupedExpertBM32 && defaults.groupedExpertBM32MinRows == 128, "Grouped BM32 default changed")
    let flags: [String: Any] = [
        "expert_mode": "slabPackedPreferred", "fused_experts": true, "simd_matrix": true,
        "grouped_expert_prefill": true, "grouped_expert_tiled": true, "grouped_expert_bm32": true,
    ]
    try grouped32Require(try load(flags).groupedExpertBM32, "Grouped BM32 selection ignored")
    for key in ["fused_experts", "simd_matrix", "grouped_expert_prefill", "grouped_expert_tiled"] {
        var input = flags
        input[key] = false
        try grouped32Reject { _ = try load(input) }
    }
    for floor in [0, 257, 1.5, true, "128"] as [Any] {
        var input = flags
        input["grouped_expert_bm32_min_rows"] = floor
        try grouped32Reject { _ = try load(input) }
    }
    for value in [1, "true"] as [Any] {
        var input = flags
        input["grouped_expert_bm32"] = value
        try grouped32Reject { _ = try load(input) }
    }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try grouped32Reject { _ = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger, groupedBM32: true) }
    try grouped32Reject {
        _ = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger, groupedTiled: true, groupedBM32MinRows: 0)
    }
    try grouped32Require(ledger.snapshot().reserved == 0, "Invalid grouped BM32 config allocated resources")
}
func groupedBM32BoundariesAndRetirement() throws {
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let baseline = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true)
        let next = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true)
        let engine = try groupedTiledEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1])
        try window.consumeBatch(experts: [0, 1]) { buffers in
            let shapes = [33, 127, 128, 149, 150, 151, 255, 256]
            let jobs = shapes.enumerated().map { index, rows in
                FusedExpertExecution.Job(
                    input: groupedTiledInput(rows, salt: index), buffer: buffers[index % 2],
                    groupedExpert: true, simdMatrix: true)
            }
            var expected: [[UInt32]] = []
            var stages: [[String: [UInt32]]] = []
            for job in jobs {
                var captured: [String: [UInt32]] = [:]
                let output = try baseline.evaluate(
                    job.input, buffer: job.buffer, groupedExpert: true, simdMatrix: true,
                    capture: { captured[$0] = $1.map(\.bitPattern) })
                expected.append(output.map(\.bitPattern))
                stages.append(captured)
            }
            var capturesEqual = true
            var outputs = 0
            let before = next.snapshot()
            let encodedBefore = next.groupedBM32EncodedProjections
            try next.evaluateBatch(
                jobs,
                capture: { index, name, values in
                    capturesEqual = capturesEqual && values.map(\.bitPattern) == stages[index][name]
                },
                consumeOutput: { index, values in
                    try grouped32Require(values.map(\.bitPattern) == expected[index], "Grouped BM32 differs at M\(shapes[index])")
                    outputs += 1
                })
            try grouped32Require(
                capturesEqual && outputs == jobs.count && next.snapshot().completedCommands == before.completedCommands + 1
                    && next.groupedBM32EncodedProjections == encodedBefore + 18,
                "Grouped BM32 boundaries, floor or single-command batch changed")
            let selected = next.groupedBM32EncodedProjections
            // Cached jobs retain QMV regardless of how many rows their input has.
            for (grouped, simd) in [(false, true), (true, false)] {
                let input = groupedTiledInput(128)
                let a = try baseline.evaluate(input, buffer: buffers[1], groupedExpert: grouped, simdMatrix: simd)
                let b = try next.evaluate(input, buffer: buffers[1], groupedExpert: grouped, simdMatrix: simd)
                try grouped32Require(
                    a.map(\.bitPattern) == b.map(\.bitPattern) && next.groupedBM32EncodedProjections == selected,
                    "Grouped BM32 changed an ineligible schedule")
            }
        }
        try engine.finishSequence()
        try engine.beginSequence()
        let cancelled = try engine.makeWindow(layer: 0, experts: [0])
        var calls = 0
        do {
            try cancelled.consume(expert: 0) { buffer in
                _ = try next.evaluate(
                    groupedTiledInput(150), buffer: buffer, groupedExpert: true, simdMatrix: true,
                    shouldContinue: {
                        calls += 1
                        return calls <= 2
                    })
            }
            throw EngineError("Grouped BM32 ignored cancellation after submission")
        } catch is IOOperationCancelled {}
        try grouped32Require(
            calls == 3 && cancelled.isClosed && engine.snapshot().cache.pinnedSlots == 0
                && next.snapshot().completedCommands == next.snapshot().submittedCommands
                && ledger.snapshot().categories["fused-expert-workspace"] == nil,
            "Grouped BM32 cancellation returned before join/retirement")
        try engine.finishSequence()
        try engine.beginSequence()
        let denied = try engine.makeWindow(layer: 0, experts: [0])
        do {
            try denied.consume(expert: 0) { buffer in
                let held = try ledger.reserve(
                    bytes: 199_000_000 - ledger.snapshot().reserved - 8_192, category: "grouped-bm32-fixture-denial")
                defer { try? ledger.release(held) }
                _ = try next.evaluate(groupedTiledInput(150), buffer: buffer, groupedExpert: true, simdMatrix: true)
            }
            throw EngineError("Grouped BM32 bypassed admission")
        } catch let error as EngineError {
            try grouped32Require(error.code == "insufficient_memory", "Grouped BM32 admission lost classification")
        }
        try grouped32Require(
            denied.isClosed && engine.snapshot().cache.pinnedSlots == 0 && ledger.snapshot().categories["fused-expert-workspace"] == nil
                && ledger.snapshot().categories["fused-expert-batch-metadata"] == nil, "Grouped BM32 admission leaked resources")
        try engine.finishSequence()
    }
    try grouped32Require(ledger.snapshot().reserved == 0, "Grouped BM32 test retained ledger reservations")
}
