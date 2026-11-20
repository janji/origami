import Foundation
import OrigamiEngine

// `quantized_mma_bf16`, `grouped_expert_mma_min_rows` and `gdn_exact_kernels` are default-off,
// refuse their missing prerequisites, and the grouped bfloat-MMA path is bit-identical to the prefetch
// BM32 path with every selection counted.

private func exactKernelRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func exactKernelReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid exact-kernel configuration accepted")
}
private func exactKernelLoad(_ execution: [String: Any]) throws -> RuntimeConfiguration {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("origami-exact-kernels-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: file) }
    try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(to: file)
    return try RuntimeConfiguration.load(file)
}

func quantizedMMAConfiguration() throws {
    let defaults = try exactKernelLoad([:])
    try exactKernelRequire(!defaults.quantizedMMABF16 && defaults.groupedExpertMMAMinRows == 0, "bfloat-MMA defaults changed")
    let dense: [String: Any] = [
        "simd_matrix": true, "quantized_tiled": true, "quantized_tiled_bm32": true, "quantized_tiled_prefetch": true,
        "quantized_mma_bf16": true,
    ]
    try exactKernelRequire(try exactKernelLoad(dense).quantizedMMABF16, "bfloat-MMA selection ignored")
    for key in ["quantized_tiled_prefetch", "quantized_tiled_bm32"] {
        var input = dense
        input[key] = false
        try exactKernelReject { _ = try exactKernelLoad(input) }
    }
    for value in [1, "true"] as [Any] {
        var input = dense
        input["quantized_mma_bf16"] = value
        try exactKernelReject { _ = try exactKernelLoad(input) }
    }
    var grouped = dense
    for (key, value) in [
        "expert_mode": "slabPackedPreferred", "fused_experts": true, "grouped_expert_prefill": true, "grouped_expert_tiled": true,
        "grouped_expert_bm32": true, "grouped_expert_bm32_min_rows": 32, "grouped_expert_mma_min_rows": 192,
    ] as [String: Any] {
        grouped[key] = value
    }
    let configured = try exactKernelLoad(grouped)
    try exactKernelRequire(
        configured.groupedExpertMMAMinRows == 192 && configured.groupedExpertBM32MinRows == 32, "grouped MMA floor ignored")
    for (key, value) in ["quantized_mma_bf16": false, "grouped_expert_bm32": false] as [String: Any] {
        var input = grouped
        input[key] = value
        try exactKernelReject { _ = try exactKernelLoad(input) }
    }
    for floor in [1, 32, 63, 257, 1.5, true, "192"] as [Any] {
        var input = grouped
        input["grouped_expert_mma_min_rows"] = floor
        try exactKernelReject { _ = try exactKernelLoad(input) }
    }
    var off = grouped
    off["grouped_expert_mma_min_rows"] = 0
    try exactKernelRequire(try exactKernelLoad(off).groupedExpertMMAMinRows == 0, "grouped MMA zero floor is off")
    try exactKernelReject { _ = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true, quantizedMMABF16: true) }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    try exactKernelReject {
        _ = try FusedExpertExecution(
            arithmetic: .hostReference, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedMMAMinRows: 192)
    }
    try exactKernelReject {
        _ = try FusedExpertExecution(
            arithmetic: .hostReference, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedPrefetch: true, groupedMMAMinRows: 32)
    }
    try exactKernelRequire(ledger.snapshot().reserved == 0, "Invalid grouped MMA configuration allocated resources")
}

func gdnExactKernelsConfiguration() throws {
    let defaults = try exactKernelLoad([:])
    try exactKernelRequire(!defaults.gdnExactKernels, "gdn_exact_kernels default changed")
    let flags: [String: Any] = ["pinned_reductions": true, "fused_gdn_preparation": true, "gdn_exact_kernels": true]
    try exactKernelRequire(try exactKernelLoad(flags).gdnExactKernels, "gdn_exact_kernels selection ignored")
    try exactKernelReject { _ = try exactKernelLoad(["gdn_exact_kernels": true]) }
    try exactKernelReject { _ = try exactKernelLoad(["pinned_reductions": true, "gdn_exact_kernels": true]) }
    for value in [1, "true"] as [Any] {
        var input = flags
        input["gdn_exact_kernels"] = value
        try exactKernelReject { _ = try exactKernelLoad(input) }
    }
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1)
    try autoreleasepool {
        let reductions = try PinnedReductions(ledger: ledger)
        let helper = try FusedGDNPreparation(arithmetic: .hostReference, reductions: reductions, ledger: ledger, exactKernels: true)
        try exactKernelRequire(helper.exactKernels && helper.exactConvTiles == 0, "exact GDN helper state differs")
        let plain = try FusedGDNPreparation(arithmetic: .hostReference, reductions: reductions, ledger: ledger)
        try exactKernelRequire(!plain.exactKernels, "GDN helper default is the reference kernel")
    }
    try exactKernelRequire(ledger.snapshot().reserved == 0, "GDN exact helper leaked allocations")
}

func groupedMMABoundariesAndCounters() throws {
    let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let baseline = try FusedExpertExecution(
            arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedPrefetch: true)
        let next = try FusedExpertExecution(
            arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true,
            groupedBM32MinRows: 32, groupedPrefetch: true, groupedMMAMinRows: 192)
        try exactKernelRequire(
            next.groupedMMAMinRows == 192 && next.groupedBM32MinRows == 32 && baseline.groupedMMAMinRows == 0, "grouped MMA floors differ")
        let engine = try groupedTiledEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: [0, 1])
        try window.consumeBatch(experts: [0, 1]) { buffers in
            // At most eight jobs share one expert command.
            let shapes = [31, 32, 127, 128, 191, 192, 200, 256]
            let jobs = shapes.enumerated().map { index, rows in
                FusedExpertExecution.Job(
                    input: groupedTiledInput(rows, salt: index + 3), buffer: buffers[index % 2],
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
            try next.evaluateBatch(
                jobs,
                capture: { index, name, values in
                    capturesEqual = capturesEqual && values.map(\.bitPattern) == stages[index][name]
                },
                consumeOutput: { index, values in
                    try exactKernelRequire(values.map(\.bitPattern) == expected[index], "Grouped bfloat-MMA differs at M\(shapes[index])")
                    outputs += 1
                })
            // Three projections per job: 3 jobs at or above 192 rows on the MMA kernel, 4 jobs in 32...191 on
            // prefetch BM32 (counted below the MMA floor), 1 job below 32 rows on BM16; the baseline (floor
            // 128) put 5 jobs on prefetch BM32.
            try exactKernelRequire(
                capturesEqual && outputs == jobs.count && next.groupedMMAEncodedProjections == 9
                    && next.groupedMMABelowMinRowsProjections == 12 && next.groupedBM32EncodedProjections == 21
                    && next.groupedPrefetchEncodedProjections == 12 && baseline.groupedBM32EncodedProjections == 15
                    && baseline.groupedMMAEncodedProjections == 0 && baseline.groupedMMABelowMinRowsProjections == 0,
                "Grouped bfloat-MMA selection counters differ")
            // Job-row histogram (16-row buckets, one count per tiled job): 31, 32, 127, 128, 191, 192 and 200, 256.
            let buckets: [UInt64] = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 1]
            try exactKernelRequire(
                next.groupedTiledJobRows16 == buckets && baseline.groupedTiledJobRows16 == buckets,
                "Grouped job-row histogram differs: \(next.groupedTiledJobRows16) / \(baseline.groupedTiledJobRows16)")
        }
        try engine.finishSequence()
    }
    try exactKernelRequire(ledger.snapshot().reserved == 0, "Grouped MMA check retained ledger reservations")
}
