import Darwin
import Foundation
import Metal
import OrigamiEngine

/// Focused, model-free checks of the Nov 5 paths. These entry points are used only by engine-checks.
package enum FollowupPagePrefillSelfChecks {
    private static func require(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError(message) }
    }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Follow-up fixture expected admission or extent refusal")
    }

    private struct Page {
        let index, count: Int
        let root: Data
        let load: (UnsafeMutableRawBufferPointer) throws -> Void
    }
    private static func pages(_ vector: QSAPagedVector) throws -> [Page] {
        var result: [Page] = []
        try vector.withCheckpointPages { index, count, root, load in
            result.append(Page(index: index, count: count, root: root, load: load))
        }
        return result
    }
    private static func verify(_ pages: [Page], values: [UInt16]) throws {
        try values.withUnsafeBytes { bytes in
            let size = PagedCheckpointFormat.pageBytes
            try require(pages.count == (bytes.count + size - 1) / size, "Resident checkpoint page count differs")
            for (index, page) in pages.enumerated() {
                let expected = UnsafeRawBufferPointer(rebasing: bytes[index * size..<min(bytes.count, (index + 1) * size)])
                try require(
                    page.index == index && page.count == expected.count && page.root == PagedCheckpointFormat.root(expected),
                    "Cached resident digest differs from the unchanged paged-v2 root")
                var loaded = Data(count: page.count)
                try loaded.withUnsafeMutableBytes { try page.load($0) }
                try require(loaded == Data(expected), "Escaped resident page loader changed payload bytes")
            }
        }
    }

    /// Cold/warm roots, COW tails, 8 KiB chunk and 256 KiB disk-page boundaries, both physical
    /// geometries, optional cache-admission failure, and loaders outliving their vector/arena owner.
    package static func residentDigestBoundaries() throws {
        for elements in [256 * 512, 64 * 128] {
            let ledger = try MemoryLedger(ceiling: 32_000_000, transientMargin: 0)
            try autoreleasepool {
                let arena = try QSAPageArena(ledger: ledger, maximumBytes: 8_000_000)
                for count in [0, 4095, 4096, 4097, 8191, 8192, 8193, 131071, 131072, 131073] {
                    try autoreleasepool {
                        let values = (0..<count).map { UInt16(truncatingIfNeeded: $0 * 313 + 0x8000) }
                        let vector = try QSAPagedVector(arena: arena, pageElements: elements).appending(values)
                        let before = ledger.snapshot().reserved
                        let cold = try pages(vector)
                        try verify(cold, values: values)
                        let cached = ledger.snapshot().reserved
                        try require(count == 0 ? cached == before : cached > before, "Resident digest cache was not admitted")
                        let warm = try pages(vector)
                        try require(
                            warm.map(\.root) == cold.map(\.root) && ledger.snapshot().reserved == cached,
                            "Warm resident traversal changed roots or duplicated the cache charge")
                        let suffix: [UInt16] = [0x8000, 0x0001, 0x7fc1]
                        let appended = try vector.appending(suffix)
                        try verify(pages(appended), values: values + suffix)
                        try verify(pages(vector), values: values)
                        try verify(cold, values: values)
                    }
                    try require(
                        arena.snapshot().liveBytes == 0 && ledger.snapshot().reserved == arena.maximumBytes,
                        "Resident page or digest cache outlived its last reader")
                }
            }
            try require(ledger.snapshot().reserved == 0, "Resident digest arena did not retire")

            // The page cache must remain optional when the entire ledger belongs to its arena.
            let denied = try MemoryLedger(ceiling: 1_000_000, transientMargin: 0)
            try autoreleasepool {
                let arena = try QSAPageArena(ledger: denied, maximumBytes: 1_000_000)
                let values = (0..<131073).map { UInt16(truncatingIfNeeded: $0 * 31) }
                let vector = try QSAPagedVector(arena: arena, pageElements: elements).appending(values)
                try verify(pages(vector), values: values)
                try verify(pages(vector), values: values)
                try require(denied.snapshot().reserved == arena.maximumBytes, "Denied digest cache changed its arena charge")
            }
            try require(denied.snapshot().reserved == 0, "Denied digest cache retained its owner")

            let lifetime = try MemoryLedger(ceiling: 2_000_000, transientMargin: 0)
            let values = (0..<8193).map { UInt16(truncatingIfNeeded: $0 * 71) }
            var escaped: [Page] = []
            weak var arenaReader: QSAPageArena?
            try autoreleasepool {
                let arena = try QSAPageArena(ledger: lifetime, maximumBytes: 1_000_000)
                arenaReader = arena
                let vector = try QSAPagedVector(arena: arena, pageElements: elements).appending(values)
                escaped = try pages(vector)
            }
            try require(
                arenaReader != nil && lifetime.snapshot().reserved > 1_000_000,
                "Escaped checkpoint loader lost its pages or digest charge")
            try verify(escaped, values: values)
            autoreleasepool { escaped.removeAll() }
            try require(arenaReader == nil && lifetime.snapshot().reserved == 0, "Escaped checkpoint loader did not retire")
        }
    }

    /// Actual-row scope/workspace copies and publication: includes the observed 236-token suffix,
    /// minimum row counts, the C256 boundary and a second C256 tile. No model weights are read.
    package static func shortSuffixScopeRows() throws {
        let ledger = try MemoryLedger(ceiling: 1_400_000_000, transientMargin: 16_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let dense = try PinnedDenseLinear(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let preparation = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let gdn = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: preparation, ledger: ledger, simdMatrix: true)
            let hc = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true)
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            guard let queue = kernels.device.makeCommandQueue() else { throw EngineError("No short-suffix command queue") }
            let base = ledger.snapshot().reserved
            for (tokens, rows) in [(1, 1), (2, 2), (236, 236), (255, 255), (512, 256)] {
                var escaped: MTLBuffer?
                var scopeCharge: UInt64 = 0
                try autoreleasepool {
                    try require(
                        WholePrefillScope.layout(tokens: tokens, allowTail: true) == nil,
                        "Small-scope execution changed the historical checkpoint scheduling decision")
                    guard let layout = try WholePrefillScope.layout(tokens: tokens, allowTail: true, admitSmallScope: true) else {
                        throw EngineError("Explicit short-suffix scope admission was not selected")
                    }
                    scopeCharge = layout.originalExtra + layout.supplementalBytes
                    try require(
                        layout.supplementalBytes > 0 && scopeCharge == layout.deviceBytes + layout.remainingHostBytes,
                        "Short-suffix supplement does not cover all device and host owners")
                    let denied = try MemoryLedger(ceiling: scopeCharge - 1, transientMargin: 0)
                    try rejected { _ = try WholePrefillScope(device: kernels.device, ledger: denied, layout: layout) }
                    try require(denied.snapshot().reserved == 0, "Partial short-suffix scope admission leaked")
                    let scope = try WholePrefillScope(device: kernels.device, ledger: ledger, layout: layout)
                    let tile = try WholeExecutionWorkspace.layout(
                        rows: rows, capacity: 131072, capture: false, argumentBytes: 4096,
                        kernels: kernels, simdMatrix: true, hc: hc, gdn: gdn, attention: attention)
                    let workspace = try WholeExecutionWorkspace(device: kernels.device, ledger: ledger, layout: tile)
                    try require(
                        tile.rows == rows && tile.sizes[WholeExecutionWorkspace.Slot.residual.rawValue] == rows * 10240 * 4,
                        "Short-suffix workspace padded its arithmetic rows")
                    try require(
                        ledger.snapshot().reserved == base + scopeCharge + tile.admittedBytes,
                        "Short-suffix scope or workspace is uncharged")
                    let embeddings = (0..<tokens * 2560).map { (i: Int) in Float((i * 17) % 1024 - 512) / 16 }
                    try scope.stageEmbeddings(embeddings)
                    let start = tokens - rows
                    guard let input = queue.makeCommandBuffer() else { throw EngineError("No short-suffix input command") }
                    try scope.encodeTileInput(input, start: start, workspace: workspace)
                    try DecodeFrontierSelfChecks.joined(input)
                    let residual = workspace[.residual].contents().assumingMemoryBound(to: Float.self)
                    for row in 0..<rows {
                        for branch in 0..<4 {
                            for column in 0..<2560 {
                                try require(
                                    residual[row * 10240 + branch * 2560 + column].bitPattern
                                        == embeddings[(start + row) * 2560 + column].bitPattern,
                                    "Actual-row input copy changed an embedding branch")
                            }
                        }
                    }
                    let outputs: [(WholeExecutionWorkspace.Slot, WholePrefillScope.Slot, Int)] = [
                        (.afterAttention, .residual, 10240), (.hcMMixed, .input, 2560), (.hcMInjection, .injection, 4),
                        (.routerWeights, .routingWeights, 10), (.sharedOutput, .shared, 2560), (.sharedSigmoid, .sharedGate, 1),
                    ]
                    for (index, entry) in outputs.enumerated() {
                        workspace[entry.0].contents().initializeMemory(
                            as: UInt8.self, repeating: UInt8(0x31 + index), count: rows * entry.2 * 4)
                        scope[entry.1].contents().initializeMemory(as: UInt8.self, repeating: 0xa5, count: tokens * entry.2 * 4)
                    }
                    guard let output = queue.makeCommandBuffer() else { throw EngineError("No short-suffix output command") }
                    try scope.encodeTileOutputs(output, start: start, workspace: workspace)
                    try DecodeFrontierSelfChecks.joined(output)
                    for entry in outputs {
                        try require(
                            memcmp(
                                scope[entry.1].contents().advanced(by: start * entry.2 * 4),
                                workspace[entry.0].contents(), rows * entry.2 * 4) == 0, "Actual-row scope output differs")
                        if start > 0 {
                            let prefix = UnsafeBufferPointer(
                                start: scope[entry.1].contents().assumingMemoryBound(to: UInt8.self), count: start * entry.2 * 4)
                            try require(prefix.allSatisfy { $0 == 0xa5 }, "Scope output overwrote another tile")
                        }
                    }
                    scope[.routed].contents().initializeMemory(as: UInt8.self, repeating: 0x41, count: tokens * 2560 * 4)
                    guard let mix = queue.makeCommandBuffer() else { throw EngineError("No short-suffix mix command") }
                    try scope.encodeMixInput(mix, start: start, workspace: workspace)
                    workspace[.result].contents().initializeMemory(as: UInt8.self, repeating: 0x42, count: rows * 10240 * 4)
                    try require(
                        !scope.publishMix(start: start, workspace: workspace, command: mix), "Unjoined short-suffix output was published")
                    try DecodeFrontierSelfChecks.joined(mix)
                    for (source, target, width) in [
                        (WholePrefillScope.Slot.residual, WholeExecutionWorkspace.Slot.afterAttention, 10240),
                        (.injection, .hcMInjection, 4), (.shared, .sharedOutput, 2560), (.sharedGate, .sharedSigmoid, 1),
                        (.routed, .routed, 2560),
                    ] {
                        try require(
                            memcmp(workspace[target].contents(), scope[source].contents().advanced(by: start * width * 4), rows * width * 4)
                                == 0,
                            "Actual-row scope mix input differs")
                    }
                    try require(
                        scope.publishMix(start: start, workspace: workspace, command: mix), "Joined short-suffix output was refused")
                    try require(
                        memcmp(scope[.residual].contents().advanced(by: start * 10240 * 4), workspace[.result].contents(), rows * 10240 * 4)
                            == 0,
                        "Short-suffix publication changed result bytes")
                    guard let invalid = queue.makeCommandBuffer() else { throw EngineError("No short-suffix rejection command") }
                    try rejected { try scope.encodeTileInput(invalid, start: tokens, workspace: workspace) }
                    try require(
                        !scope.publishMix(start: tokens, workspace: workspace, command: mix),
                        "Out-of-range short-suffix output was published")
                    escaped = scope[.input]
                }
                try require(
                    escaped != nil && ledger.snapshot().reserved == base + scopeCharge,
                    "Escaped short-suffix buffer lost its full admission or retained its workspace")
                autoreleasepool { escaped = nil }
                try require(ledger.snapshot().reserved == base, "Short-suffix scope failed final retirement")
            }
        }
        try require(ledger.snapshot().reserved == 0, "Short-suffix fixture retained GPU owners")
    }

    /// Exercise the actual paged encoder, not the host-array reference API, which deliberately
    /// always uses the old two-dispatch implementation. Require proof of the fused dispatch.
    package static func longFusedAttention() throws {
        for keys in [65_536, 65_537, 131_072, 262_144] {
            let ledger = try MemoryLedger(ceiling: 3_000_000_000, transientMargin: 16_000_000)
            try autoreleasepool {
                let scratch = try ledger.reserve(
                    bytes: UInt64(keys) * 512 * 4 + 64_000_000,
                    category: "long_fused_attention_fixture_arrays")
                defer { try? ledger.release(scratch) }
                let reference = try PinnedAttention(ledger: ledger, architecture: .medium, registerDecode: false)
                let candidate = try PinnedAttention(ledger: ledger, architecture: .medium, registerDecode: true)
                let query = (0..<6144).map { Float(($0 * 7) % 31 - 15) / 32 }
                let oldKeys = (0..<((keys - 1) * 512)).map { BF16.encode(Float(($0 * 13) % 61 - 30) / 64) }
                let oldValues = (0..<((keys - 1) * 512)).map { BF16.encode(Float(($0 * 17) % 67 - 33) / 32) }
                let newKeys = (0..<512).map { BF16.encode(Float(($0 * 19) % 37 - 18) / 64) }
                let newValues = (0..<512).map { BF16.encode(Float(($0 * 23) % 41 - 20) / 32) }
                let selected = Array(Set((0..<2048).map { UInt32(($0 * (keys - 1)) / 2047) })).sorted()
                let expected = try reference.attention(
                    queries: query, oldKeys: oldKeys, newKeys: newKeys,
                    oldValues: oldValues, newValues: newValues, selectedPositions: [selected])
                let arena = try QSAPageArena(ledger: ledger, maximumBytes: UInt64(keys) * 2112 + 16_777_216)
                var state = QSAState(arena: arena)
                try state.append(
                    keys: oldKeys, values: oldValues,
                    pooledKeys: [UInt16](repeating: 0, count: (keys - 1) / 4 * 128),
                    rawTail: [UInt16](repeating: 0, count: (keys - 1) % 4 * 128), tokens: keys - 1)
                let storage = try candidate.makePagedStorage(capacityTokens: keys)
                let actual = try candidate.attentionPaged(
                    queries: query, state: state, newKeys: newKeys,
                    newValues: newValues, selectedPositions: [selected], storage: storage)
                try require(
                    candidate.longFusedDecodeDispatches == 1 && candidate.registerDecodeDispatches == 0
                        && reference.longFusedDecodeDispatches == 0,
                    "Long fused attention did not dispatch its actual kernel at \(keys) keys")
                try require(
                    expected.count == actual.count
                        && zip(expected, actual).allSatisfy {
                            $0.isFinite && $1.isFinite && $0.bitPattern == $1.bitPattern
                        }, "Long fused attention differs from the unfused reference at \(keys) keys")
                try require(storage.snapshot().committedTokens == keys, "Long fused attention did not publish its joined table")
            }
            try require(ledger.snapshot().reserved == 0, "Long fused attention retained request or page owners")
        }
    }
}
