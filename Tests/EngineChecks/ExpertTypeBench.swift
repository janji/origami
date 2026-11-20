import Foundation
import Metal
import OrigamiEngine

/// `engine-checks --bench-expert-types [rows,rows,...]`: model-free GPU time of Qwen's routed experts per record format,
/// through the engine's own paths: decode is `GatheredExpertDecode.encodeRegion` (ten experts of one token and the
/// ten-rank reduction), prefill is `FusedExpertExecution.evaluateDeviceLayer` with 64 experts of `rows` rows each.
/// Formats: Origami's 4-bit records, and GGUF records (`gguf-experts`) whose gate, up and down are one
/// ggml type, plus the mixes the GSQ-RCO files use. Random weights, 80 distinct records rotated so the system cache
/// does not hold them. Reports GPU ms per expert and the record size.
func expertTypeBench(_ arguments: [String]) throws {
    let prefillRows = arguments.first.map { $0.split(separator: ",").compactMap { Int($0) } } ?? [64, 256]
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("No Metal device")
    }
    // name, gate/up/down ggml types (none: Origami's 4-bit)
    let formats: [(String, [Int])] = [
        ("4-bit (exact)", []),
        ("Q2_0", [42, 42, 42]), ("IQ2_XXS", [16, 16, 16]), ("IQ2_XS", [17, 17, 17]), ("IQ2_S", [22, 22, 22]),
        ("IQ3_XXS", [18, 18, 18]), ("IQ3_S", [21, 21, 21]), ("IQ4_NL", [20, 20, 20]), ("IQ4_XS", [23, 23, 23]),
        ("Q4_K", [12, 12, 12]),
        ("IQ2_XS/IQ2_XS/Q2_0", [17, 17, 42]), ("IQ3_S/IQ3_S/IQ4_NL", [21, 21, 20]),
        ("IQ2_XXS/IQ2_XXS/Q2_0", [16, 16, 42]), ("IQ3_XXS/IQ3_XXS/IQ4_NL", [18, 18, 20]),
    ]
    let only = ProcessInfo.processInfo.environment["BENCH_FORMATS"].map { Set($0.split(separator: ";").map(String.init)) }
    func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
    print(
        "format                    record bytes  decode ms/10 experts  ms/expert"
            + prefillRows.map { "  prefill \($0) rows ms/expert" }.joined())
    for (name, types) in formats where only?.contains(name) ?? true {
        let gguf = types.isEmpty ? nil : GGUFExperts(experts: 512, types: [types])
        if let gguf {
            ModelLayout.use(.ggufExperts, gguf: gguf)
        } else {
            ModelLayout.use(.affine4Group64)
        }
        defer { ModelLayout.use(.affine4Group64) }
        let bytes = PackStore.expertBytes
        try autoreleasepool {
            let ledger = try MemoryLedger(ceiling: 24_000_000_000, transientMargin: 1_000_000)
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true, quantizedTiled: true, quantizedTiledBM32: true,
                quantizedBM32MinRows: 256, quantizedTiledPrefetch: true, quantizedMMABF16: true, gdnExactKernels: true,
                hcExactFusion: true)
            let reductions = try PinnedReductions(ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 32,
                scratchReuse: true, groupedPrefetch: true, groupedMMAMinRows: 192)
            let gathered = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reductions)
            // 80 records with random codes and sane scales.
            let count = 80
            guard let buffer = device.makeBuffer(length: count * bytes, options: .storageModeShared) else {
                throw EngineError("Bench buffer unavailable")
            }
            var random = MatrixRandom(state: 0x7E57)
            for i in 0..<count {
                let base = buffer.contents().advanced(by: i * bytes)
                if let gguf {
                    gguf.prepare(UnsafeMutableRawBufferPointer(start: base, count: bytes), layer: 0, expert: i)
                    for part in gguf.layers[0].parts {
                        let raw = randomGGMLBlocks(part.type, values: part.outputs * part.width, &random)
                        raw.withUnsafeBytes { base.advanced(by: part.record).copyMemory(from: $0.baseAddress!, byteCount: part.bytes) }
                    }
                } else {
                    let words = base.assumingMemoryBound(to: UInt32.self)
                    for w in 0..<(bytes / 4) { words[w] = random.next() }
                    for kind in ExpertProjection.allCases {
                        let groups = kind.width * kind.outputs / ModelLayout.experts.group
                        let scales = base.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
                        let biases = base.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
                        for g in 0..<groups {
                            scales[g] = BF16.encode(Float((g + i) % 7 + 1) / 1024)
                            biases[g] = BF16.encode(Float((g + i) % 9 - 4) / 256)
                        }
                    }
                }
            }
            let records = (0..<count).map { ExpertBuffer(replicaBuffer: buffer, offset: $0 * bytes) }
            func zeroed(_ n: Int) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: n, options: .storageModeShared) else {
                    throw EngineError("Bench buffer unavailable")
                }
                memset(b.contents(), 0, n)
                return b
            }
            // Decode: ten experts of one token, 8 tokens per command buffer, 5 timed runs.
            let input = try zeroed(2560 * 2)
            let x = input.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<2560 { x[i] = BF16.encode(Float(i * 103 % 991 - 495) / 4000) }
            let region = try GatheredExpertDecode.RegionBuffers(
                gate: zeroed(10 * 640 * 4), up: zeroed(10 * 640 * 4), product: zeroed(10 * 2560 * 4),
                down: zeroed(10 * 2560 * 4), descriptors: zeroed(gathered.regionArgumentBufferBytes), ranks: zeroed(40))
            let weights = try zeroed(40)
            for i in 0..<10 { weights.contents().assumingMemoryBound(to: Float.self)[i] = 0.1 }
            let output = try zeroed(2560 * 4)
            var decode: [Double] = []
            for run in 0...5 {
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Bench command unavailable") }
                for copy in 0..<8 {
                    try gathered.encodeRegion(
                        command, inputBF16: input, weights: weights,
                        records: (0..<10).map {
                            GatheredExpertDecode.Record(
                                buffer: records[(copy * 10 + $0) % count].buffer, offset: records[(copy * 10 + $0) % count].offset)
                        },
                        canonicalRanks: Array(0..<10), output: output, buffers: region)
                }
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed else { throw EngineError.gpu("Bench command failed", command.error) }
                if run > 0 { decode.append((command.gpuEndTime - command.gpuStartTime) / 8) }
            }
            // Prefill: 64 experts with `rows` rows each (tokens = 64 * rows / 10, every token on ten distinct experts).
            var prefill: [Double] = []
            let transfer = try PrefillMoETransfer()
            for rows in prefillRows {
                let experts = 64
                let tokens = experts * rows / 10
                var selected: [Int: [(Int, Int)]] = [:]
                for token in 0..<tokens {
                    for rank in 0..<10 { selected[(token * 10 + rank) % experts, default: []].append((token, rank)) }
                }
                let ids = selected.keys.sorted()
                let inputs = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 4000 }
                var gpu: [Double] = []
                for run in 0...3 {
                    try autoreleasepool {
                        let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                        let workspace = try PrefillMoEWorkspace(input: inputs, selectedRows: selected, budget: budget, transfer: transfer)
                        let before = fused.snapshot().gpuSeconds
                        try fused.evaluateDeviceLayer(
                            experts: ids, buffers: ids.map { records[$0 % count] }, selectedRows: selected, workspace: workspace,
                            simdMatrix: true, waves: 8)
                        if run > 0 { gpu.append(fused.snapshot().gpuSeconds - before) }
                    }
                }
                prefill.append(median(gpu) / Double(experts))
            }
            let d = median(decode)
            print(
                name.padding(toLength: 25, withPad: " ", startingAt: 0)
                    + String(format: " %12d  %20.4f  %9.4f", bytes, d * 1e3, d * 1e2)
                    + prefill.map { String(format: "  %26.4f", $0 * 1e3) }.joined())
        }
    }
}
