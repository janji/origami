import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of the production Qwen layout and HC encoders. Compare live phase outputs after both HCs
/// execute in one command, so a shared temporary is necessarily overwritten before attention outputs are read.
package enum ScratchCompactionSelfChecks {
    private static func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Qwen scratch compaction: " + message) }
    }
    private static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("Qwen scratch fixture allocation failed")
        }
        return result
    }
    private static func bytes(_ buffer: MTLBuffer, _ count: Int) -> Data { Data(bytes: buffer.contents(), count: count) }

    package static func qwen() throws {
        let ledger = try MemoryLedger(ceiling: 512 << 20, transientMargin: 1)
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let dense = try PinnedDenseLinear(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let preparation = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let gdn = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: preparation, ledger: ledger, simdMatrix: true)
            let hc = try FusedHCDecode(
                arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels,
                prefillSIMDMatrix: true, decodeFusion: true)
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            let device = kernels.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("Qwen scratch fixture queue unavailable") }
            let host = try ledger.reserve(bytes: 32 << 20, category: "scratch_compaction_fixture")
            defer { try? ledger.release(host) }
            func layout(_ rows: Int, _ compact: Bool, capture: Bool = false, gdnOnly: Bool = false) throws -> WholeExecutionWorkspace.Layout
            {
                try WholeExecutionWorkspace.layout(
                    rows: rows, capacity: 8192, capture: capture, argumentBytes: 4096,
                    kernels: kernels, simdMatrix: true, hc: hc, gdn: gdn, attention: attention, gdnOnly: gdnOnly, compact: compact)
            }
            // The actual maximum tile, ordinary short tiles, MTP pair and GDN-only second generation retain their
            // logical capacities. Captures deliberately retain distinct temporaries for post-command inspection.
            for rows in [1, 2, 3, 41, 256] {
                let plain = try layout(rows, false)
                let compact = try layout(rows, true)
                try require(
                    plain.sizes == compact.sizes && compact.savedBytes > 0
                        && plain.bufferBytes == compact.bufferBytes + compact.savedBytes, "logical geometry or saving differs")
                let captured = try layout(rows, true, capture: true)
                try require(
                    !captured.compact && captured.savedBytes == 0
                        && captured.storageIndices == Array(captured.sizes.indices), "capture retained overwritten intermediates")
                if rows == 256 {
                    let second = try layout(rows, true, gdnOnly: true)
                    try require(second.gdnOnly && second.savedBytes == compact.savedBytes, "second GDN generation lost compaction")
                }
            }
            var random: UInt64 = 0xC0DE_CAFE
            func next() -> UInt64 {
                random = random &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return random >> 33
            }
            func projection(_ width: Int, _ outputs: Int) throws -> FusedHCDecode.Projection {
                let packed = (0..<(width * outputs / 8)).map { _ in UInt32(truncatingIfNeeded: next() &* 0x9E37_79B9) }
                let scales = (0..<(width * outputs / ModelLayout.group)).map { _ in BF16.encode(Float(next() % 11 + 1) / 1024) }
                let biases = (0..<(width * outputs / ModelLayout.group)).map { _ in BF16.encode(Float(Int(next() % 17) - 8) / 512) }
                return .init(words: try buffer(packed, device), scales: try buffer(scales, device), biases: try buffer(biases, device))
            }
            let down = try projection(10240, 320)
            let up = try projection(320, 10240)
            var norms = [UInt16](repeating: BF16.encode(1), count: 10240)
            norms[5] = 0x7FC0  // Preserve separate exception bitmaps even when the temporary that produced one is dead.
            let norm = try buffer(norms, device)
            let injection = try buffer((0..<(4 * 10240)).map { BF16.decode(BF16.encode(Float($0 % 31 - 15) / 512)) }, device)
            for rows in [1, 7] {
                let compactLayout = try layout(rows, true)
                let before = ledger.reservedBytes
                var escaped: MTLBuffer?
                try autoreleasepool {
                    let plain = try WholeExecutionWorkspace(device: device, ledger: ledger, layout: layout(rows, false))
                    let compact = try WholeExecutionWorkspace(device: device, ledger: ledger, layout: compactLayout)
                    let all = WholeExecutionWorkspace.Slot.allCases.compactMap { compact.optional($0) }
                    let unique = Dictionary(all.map { (ObjectIdentifier($0), $0) }, uniquingKeysWith: { first, _ in first })
                    try require(unique.count == compactLayout.allocationSizes.filter { $0 > 0 }.count, "unexpected physical aliases")
                    let charged = unique.values.reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1.length) }
                    try require(charged == compactLayout.bufferBytes, "physical charge did not match unique buffers")
                    for (first, second) in WholeExecutionWorkspace.compactPairs
                    where compact.optional(first) != nil && compact.optional(second) != nil {
                        try require(compact[first] === compact[second], "declared HC alias was not installed")
                    }
                    let retained: [WholeExecutionWorkspace.Slot] = [
                        .hcAMixed, .hcAFlags, .hcARawInjection, .hcAInjection,
                        .hcMMixed, .hcMFlags, .hcMRawInjection, .hcMInjection, .gdnMixed, .gdnNext,
                        .expertGate, .expertUp, .expertProduct, .expertDown, .routed, .result,
                    ]
                    let live = retained.compactMap { compact.optional($0) }
                    try require(Set(live.map { ObjectIdentifier($0) }).count == live.count, "persistent/canonical/expert roles alias")
                    for workspace in [plain, compact] {
                        var poisoned = Set<ObjectIdentifier>()
                        for slot in WholeExecutionWorkspace.Slot.allCases {
                            if let b = workspace.optional(slot), poisoned.insert(ObjectIdentifier(b)).inserted {
                                memset(b.contents(), 0xA5, b.length)
                            }
                        }
                        workspace.clearFlags()
                        let a = (0..<(rows * 10240)).map { BF16.encode(Float($0 % 79 - 39) / 19) }
                        let m = (0..<(rows * 10240)).map { BF16.encode(Float($0 % 113 - 56) / 37) }
                        a.withUnsafeBytes { workspace[.residualBF16].contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                        m.withUnsafeBytes {
                            workspace[.afterAttentionBF16].contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
                        }
                        guard let command = queue.makeCommandBuffer() else { throw EngineError("Qwen scratch command unavailable") }
                        for isAttention in [true, false] {
                            let input = workspace[isAttention ? .residualBF16 : .afterAttentionBF16]
                            let buffers = isAttention ? workspace.attentionHC : workspace.moeHC
                            if rows == 1 {
                                try hc.encodeRow(
                                    command, residualBF16: input, normWeights: norm, downWeights: down, upWeights: up,
                                    injectionWeights: injection, buffers: buffers)
                            } else {
                                try hc.encodeTile(
                                    command, rows: rows, residualBF16: input, normWeights: norm, downWeights: down, upWeights: up,
                                    injectionWeights: injection, buffers: buffers,
                                    downPartials: workspace.optional(isAttention ? .hcADownPartial : .hcMDownPartial),
                                    upPartials: workspace.optional(isAttention ? .hcAUpPartial : .hcMUpPartial),
                                    injectionPartials: workspace.optional(isAttention ? .hcAInjectionPartial : .hcMInjectionPartial))
                            }
                        }
                        command.commit()
                        command.waitUntilCompleted()
                        try require(command.status == .completed, "HC command failed: \(String(describing: command.error))")
                    }
                    let compared: [WholeExecutionWorkspace.Slot] = [
                        .hcAMixed, .hcAFlags, .hcARawInjection, .hcAInjection,
                        .hcMNorm, .hcMNormBF16, .hcMDown, .hcMActivation, .hcMUp, .hcMMixed, .hcMFlags, .hcMRawInjection, .hcMInjection,
                    ]
                    for slot in compared {
                        try require(
                            bytes(plain[slot], compactLayout.sizes[slot.rawValue])
                                == bytes(compact[slot], compactLayout.sizes[slot.rawValue]),
                            "C\(rows) live \(slot) bytes changed after reused HC projection storage")
                    }
                    // A logical alias escaping the workspace keeps the entire shared allocation owner alive.
                    escaped = compact[.hcMNorm]
                }
                try require(
                    escaped != nil && ledger.reservedBytes == before + compactLayout.admittedBytes, "escaped alias lost its workspace owner"
                )
                autoreleasepool { escaped = nil }
                try require(ledger.reservedBytes == before, "workspace aliases leaked after final retirement")
            }
        }
        try require(ledger.reservedBytes == 0, "fixture retained owners after completion")
    }
}
