import Darwin
import Foundation
import Metal
import OrigamiEngine

private func groupedTiledRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func groupedTiledReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid grouped tiled expert configuration succeeded")
}
func groupedTiledInput(_ rows: Int, salt: Int = 0) -> [Float] {
    (0..<(rows * 2560)).map { BF16.decode(BF16.encode(Float(($0 * 103 + salt * 71) % 991 - 495) / 193)) }
}
func groupedTiledEngine(_ ledger: MemoryLedger) throws -> ExpertExecution {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing grouped fixture Metal device") }
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 2, slotsPerSlab: 2,
        demandWindow: 2, readConcurrency: 1)
    return try ExpertExecution(device: device, ledger: ledger, configuration: config, packedAvailable: { false }) { key, _, destination in
        for kind in ExpertProjection.allCases {
            let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
            let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
            let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(819_200 / 4) { words[i] = UInt32(truncatingIfNeeded: UInt64(i + key.expert + 1) &* 0x9e37_79b9_7f4a_7c15) }
            for i in 0..<((kind.biasOffset - kind.scaleOffset) / 2) {
                scales[i] = BF16.encode(Float((i + key.expert) % 19 + 1) / 257)
                biases[i] = BF16.encode(Float((i + key.expert) % 23 - 11) / 131)
            }
        }
    }
}

func groupedExpertTiledConfigurationRequiresItsExecutionPath() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-grouped-tiled-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    try groupedTiledRequire(try !load([:]).groupedExpertTiled, "Grouped tiled kernel must default off")
    let valid: [String: Any] = [
        "expert_mode": "slabPackedPreferred", "fused_experts": true, "simd_matrix": true,
        "grouped_expert_prefill": true, "grouped_expert_tiled": true,
    ]
    try groupedTiledRequire(try load(valid).groupedExpertTiled, "Grouped tiled option was lost")
    for key in ["fused_experts", "simd_matrix", "grouped_expert_prefill"] {
        var invalid = valid
        invalid[key] = false
        try groupedTiledReject { _ = try load(invalid) }
    }
    for value in [1, "true"] as [Any] {
        var invalid = valid
        invalid["grouped_expert_tiled"] = value
        try groupedTiledReject { _ = try load(invalid) }
    }
}

func groupedExpertTiledPreservesEveryBoundaryAcrossTailAndBatchShapes() throws {
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let baseline = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger)
        let tiled = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true)
        let engine = try groupedTiledEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1])
        try window.consumeBatch(experts: [0, 1]) { buffers in
            try groupedTiledRequire(buffers[1].offset != 0, "Grouped fixture must cover nonzero record offsets")
            let shapes = [1, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 255, 256]
            for base in stride(from: 0, to: shapes.count, by: 8) {
                let cases = Array(shapes[base..<min(base + 8, shapes.count)])
                let jobs = cases.enumerated().map { index, rows in
                    FusedExpertExecution.Job(
                        input: groupedTiledInput(rows, salt: index), buffer: buffers[index % 2],
                        groupedExpert: true, simdMatrix: true)
                }
                var expectedOutputs: [[UInt32]] = []
                var expectedStages: [[String: [UInt32]]] = []
                for job in jobs {
                    var stages: [String: [UInt32]] = [:]
                    let output = try baseline.evaluate(
                        job.input, buffer: job.buffer, groupedExpert: true, simdMatrix: true,
                        capture: { stages[$0] = $1.map(\.bitPattern) })
                    expectedOutputs.append(output.map(\.bitPattern))
                    expectedStages.append(stages)
                }
                var stagesMatch = true
                var consumed = 0
                let before = tiled.snapshot()
                try tiled.evaluateBatch(
                    jobs,
                    capture: { index, name, values in
                        stagesMatch = stagesMatch && values.map(\.bitPattern) == expectedStages[index][name]
                    },
                    consumeOutput: { index, values in
                        try groupedTiledRequire(
                            values.map(\.bitPattern) == expectedOutputs[index], "Grouped tiled output differs at M=\(cases[index])")
                        consumed += 1
                    })
                let after = tiled.snapshot()
                try groupedTiledRequire(
                    stagesMatch && consumed == cases.count, "Grouped tiled gate/up/product boundary or output delivery changed")
                try groupedTiledRequire(
                    after.completedCommands == before.completedCommands + 1
                        && after.submittedJobs == before.submittedJobs + UInt64(jobs.count), "Grouped tiled jobs escaped one joined command"
                )
                try groupedTiledRequire(
                    ledger.snapshot().categories["fused-expert-workspace"] == nil, "Grouped tiled command retained workspace")
            }
            // Enabling the pipeline must leave cached QMV and scalar grouped
            // diagnostics on their original schedules.
            for (grouped, simd) in [(false, false), (false, true), (true, false)] {
                let input = groupedTiledInput(9)
                let a = try baseline.evaluate(input, buffer: buffers[1], groupedExpert: grouped, simdMatrix: simd)
                let b = try tiled.evaluate(input, buffer: buffers[1], groupedExpert: grouped, simdMatrix: simd)
                try groupedTiledRequire(a.map(\.bitPattern) == b.map(\.bitPattern), "Grouped tiled flag affected an ineligible schedule")
            }
        }
        try engine.finishSequence()
        try groupedTiledRequire(engine.snapshot().cache.pinnedSlots == 0, "Grouped tiled parity check retained a pin")
    }
}

func groupedExpertTiledCancellationAndAdmissionRetireBeforeReuse() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let tiled = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger, groupedTiled: true)
        let engine = try groupedTiledEngine(ledger)
        for allowedChecks in [0, 1, 2] {
            try engine.beginSequence()
            let window = try engine.makeWindow(layer: 0, experts: [0, 1])
            var calls = 0
            var outputs = 0
            do {
                try window.consumeBatch(experts: [0, 1]) { buffers in
                    let jobs = buffers.enumerated().map { index, buffer in
                        FusedExpertExecution.Job(
                            input: groupedTiledInput(index == 0 ? 17 : 9), buffer: buffer,
                            groupedExpert: true, simdMatrix: true)
                    }
                    try tiled.evaluateBatch(
                        jobs,
                        shouldContinue: {
                            calls += 1
                            return calls <= allowedChecks
                        },
                        consumeOutput: { _, _ in outputs += 1 })
                }
                throw EngineError("Cancelled tiled expert returned outputs")
            } catch is IOOperationCancelled {}
            try groupedTiledRequire(
                calls == allowedChecks + 1 && outputs == 0 && window.isClosed && engine.snapshot().cache.pinnedSlots == 0
                    && engine.snapshot().cache.inFlightReads == 0 && ledger.snapshot().categories["fused-expert-workspace"] == nil,
                "Tiled cancellation escaped its joined consumer scope")
            try engine.finishSequence()
        }
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [2])
        do {
            try window.consume(expert: 2) { buffer in
                let before = ledger.snapshot().reserved
                let held = try ledger.reserve(bytes: 99_000_000 - before - 8_192, category: "grouped-tiled-fixture-denial")
                defer { try? ledger.release(held) }
                _ = try tiled.evaluate(groupedTiledInput(17), buffer: buffer, groupedExpert: true, simdMatrix: true)
            }
            throw EngineError("Tiled expert silently bypassed ledger admission")
        } catch let error as EngineError {
            try groupedTiledRequire(error.code == "insufficient_memory", "Tiled admission failure lost its memory classification")
        }
        try groupedTiledRequire(
            window.isClosed && engine.snapshot().cache.pinnedSlots == 0 && ledger.snapshot().categories["fused-expert-workspace"] == nil
                && ledger.snapshot().categories["fused-expert-batch-metadata"] == nil, "Tiled allocation denial leaked resources")
        try engine.finishSequence()
    }
}
