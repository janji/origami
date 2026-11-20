import Darwin
import Foundation
import Metal
import OrigamiEngine

/// Checks of `expert_prefill_layer_command` (one command per prefill layer for all its experts): raw contributions
/// and the canonical reduction are bit-identical to the eight-job batch control on whole, partial, exceptional and
/// mixed-tail scopes, with and without waves; cancellation and failures join before returning and publish nothing;
/// the owner's bytes are part of memory admission; the final layer's row skip keeps its kept rows bit-identical.
package enum PrefillLayerCommandSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: @autoclosure () -> String) throws {
        guard try value() else { throw EngineError(message()) }
    }
    private static func rejects(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Invalid layer command was admitted")
    }
    private static func selected(_ tokens: Int, maximum: Bool) -> [Int: [(Int, Int)]] {
        var rows: [Int: [(Int, Int)]] = [:]
        for token in 0..<tokens {
            for rank in 0..<10 {
                let expert = maximum ? (rank == 0 ? 0 : 1 + (token * 9 + rank - 1) % 511) : (rank + token) % 10
                rows[expert, default: []].append((token, rank))
            }
        }
        return rows
    }
    private static func engine(_ ledger: MemoryLedger) throws -> ExpertExecution {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
        return try ExpertExecution(
            device: device, ledger: ledger,
            configuration: ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 528, slotsPerSlab: 8,
                demandWindow: 16, readConcurrency: 4, prefillReadAhead: true, prefillLayerCommand: true), packedAvailable: { false }
        ) { key, _, destination in
            for kind in ExpertProjection.allCases {
                let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
                let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
                let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
                for i in 0..<(819200 / 4) { words[i] = UInt32(truncatingIfNeeded: UInt64(i + key.expert + 1) &* 0x9e37_79b9_7f4a_7c15) }
                for i in 0..<(51200 / 2) {
                    scales[i] = BF16.encode(Float((i + key.expert) % 7 + 1) / 1024)
                    biases[i] = BF16.encode(Float((i + key.expert) % 9 - 4) / 256)
                }
            }
        }
    }
    private static func retired(_ engine: ExpertExecution) throws {
        let s = engine.snapshot()
        try require(
            s.cache.pinnedSlots == 0 && s.cache.inFlightReads == 0 && s.cache.reservedSlots == 0
                && s.io.drives.allSatisfy { $0.outstanding == 0 }, "Layer command retained a pin or writer")
    }
    package static func exactActualScope(mixedOnly: Bool = false, waves: Int = 1) throws {
        let ledger = try MemoryLedger(ceiling: 7_000_000_000, transientMargin: 100_000_000)
        try autoreleasepool {
            let host = try ledger.reserve(bytes: 400_000_000, category: "layer_command_fixture_host")
            defer { try? ledger.release(host) }
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 128)
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let executor = try engine(ledger)
            var admitted: [Int] = []
            for tokens in stride(from: 256, through: 8192, by: 256) {
                if try WholePrefillScope.layout(tokens: tokens) != nil { admitted.append(tokens) }
            }
            try require(admitted.contains(7424) && admitted.contains(8192), "Actual whole-prefill layouts were not admitted")
            // Exercise the smallest admitted whole scope and the 7,424-token scope of an
            // 8K prompt, in addition to exceptional partial tiles and the maximum.
            let cases: [(Int, Bool, Int?)] =
                mixedOnly
                ? [(7425, false, 7424), (7678, false, 7424), (8191, false, 7936)]
                : [
                    (257, false, nil), (257, true, nil), (admitted[0], false, nil),
                    (7424, false, nil), (7678, false, nil), (8192, false, nil),
                ]
            for (tokens, exceptional, tailStart) in cases {
                try autoreleasepool {
                    var input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
                    if exceptional {
                        for (i, bits) in [UInt32(0x7fc3_0000), 0xffc7_0000, 0x7f80_0000, 0xff80_0000, 1, 0x8000_0000].enumerated() {
                            input[i * 2560 + 17] = Float(bitPattern: bits)
                        }
                    }
                    let fullUnion = try WholePrefillScope.layout(tokens: tokens, allowTail: true) != nil
                    let rows = selected(tokens, maximum: fullUnion)
                    let experts = rows.keys.sorted()
                    let controlBudget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                    let candidateBudget = try PrefillMoEBudget(
                        ledger: ledger,
                        bytes: PrefillMoEBudget.bytesForScope(tokens: tokens)
                            + (tailStart == nil ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes))
                    let control = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: controlBudget, transfer: transfer)
                    let candidate = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: candidateBudget, transfer: transfer)
                    try executor.beginSequence()
                    defer { try? executor.finishSequence() }
                    // The independent control retains the existing eight-job
                    // batches, separate encoders and per-job private buffers.
                    let referenceWindow = try executor.makeWindow(layer: 0, experts: experts)
                    for first in stride(from: 0, to: experts.count, by: 8) {
                        let group = Array(experts[first..<min(first + 8, experts.count)])
                        try referenceWindow.consumeBatch(experts: group) { buffers in
                            var jobs: [FusedExpertExecution.DeviceJob] = []
                            func flush() throws {
                                if jobs.isEmpty { return }
                                try fused.evaluateDeviceBatch(jobs, workspace: control)
                                jobs.removeAll(keepingCapacity: true)
                            }
                            for (index, expert) in group.enumerated() {
                                let positions = rows[expert]!
                                // Independent legacy construction: the old full scope and
                                // short scope are separately selected and chunked.
                                let partitions: [([(Int, Int)], Bool)] =
                                    tailStart.map { tail in
                                        [(positions.filter { $0.0 < tail }, true), (positions.filter { $0.0 >= tail }, false)]
                                    } ?? [(positions, true)]
                                for (selected, grouped) in partitions {
                                    let chunk = 256
                                    for lower in stride(from: 0, to: selected.count, by: chunk) {
                                        jobs.append(
                                            FusedExpertExecution.DeviceJob(
                                                expert: expert,
                                                slots: selected[lower..<min(lower + chunk, selected.count)].map {
                                                    UInt32($0.0 * 10 + $0.1)
                                                },
                                                buffer: buffers[index], groupedExpert: grouped, simdMatrix: true))
                                        if jobs.count == 8 { try flush() }
                                    }
                                }
                            }
                            try flush()
                        }
                    }
                    if tokens == 7678, tailStart != nil {
                        // Reproduce the earlier all-grouped command (wrong for a mixed
                        // tail) against the mixed control, so this fixture cannot pass
                        // by comparing two copies of the same changed arithmetic.
                        try autoreleasepool {
                            let oldBudget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                            let old = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: oldBudget, transfer: transfer)
                            try executor.beginPrefillReadAhead(layers: 1)
                            let oldWindow = try executor.makeWindow(layer: 0, experts: experts)
                            try oldWindow.consumeLayer { buffers in
                                try rejects {
                                    try fused.evaluateDeviceLayer(
                                        experts: experts, buffers: buffers, selectedRows: rows,
                                        workspace: old, simdMatrix: true, ungroupedTailStart: tailStart)
                                }
                                try fused.evaluateDeviceLayer(
                                    experts: experts, buffers: buffers, selectedRows: rows,
                                    workspace: old, simdMatrix: true)
                            }
                            let a = control.contributions.contents().assumingMemoryBound(to: UInt32.self)
                            let b = old.contributions.contents().assumingMemoryBound(to: UInt32.self)
                            try require(
                                (0..<(tokens * 10 * 2560)).contains { a[$0] != b[$0] },
                                "Mixed-tail fixture failed to discriminate L's all-grouped arithmetic defect")
                            try executor.finishPrefillReadAheadLayer(0)
                            executor.finishPrefillReadAhead()
                        }
                    }
                    try executor.beginPrefillReadAhead(layers: 1)
                    let window = try executor.makeWindow(layer: 0, experts: experts)
                    let before = fused.snapshot()
                    try window.consumeLayer { buffers in
                        if fullUnion {
                            try require(
                                buffers.count == 512 && executor.snapshot().cache.pinnedSlots == 512,
                                "Actual full union was not simultaneously pinned")
                        }
                        try rejects { try executor.beginDecodePhase() }
                        if tailStart != nil {
                            for invalid in [0, 1, tokens] {
                                try rejects {
                                    try fused.evaluateDeviceLayer(
                                        experts: experts, buffers: buffers, selectedRows: rows,
                                        workspace: candidate, simdMatrix: true, ungroupedTailStart: invalid)
                                }
                            }
                        }
                        try fused.evaluateDeviceLayer(
                            experts: experts, buffers: buffers, selectedRows: rows,
                            workspace: candidate, simdMatrix: true, ungroupedTailStart: tailStart, waves: waves)
                    }
                    let after = fused.snapshot()
                    try require(waves == 1 || fused.layerWaveCommands > 0, "Wave layer command fell back to the serial chain")
                    try require(
                        after.submittedCommands == before.submittedCommands + 1
                            && after.completedCommands == after.submittedCommands && candidate.completedSlots == tokens * 10,
                        "Layer command did not publish complete coverage after one joined command")
                    if let tail = tailStart {
                        try require(
                            after.mixedTailCommands == before.mixedTailCommands + 1
                                && after.mixedTailQMVRows == before.mixedTailQMVRows + UInt64((tokens - tail) * 10)
                                && after.mixedTailQMVTiles > before.mixedTailQMVTiles
                                && after.mixedTailQMVTiles - before.mixedTailQMVTiles <= 512,
                            "Mixed tail did not publish exact QMV coverage after its single joined command")
                    }
                    let a = control.contributions.contents().assumingMemoryBound(to: UInt32.self)
                    let b = candidate.contributions.contents().assumingMemoryBound(to: UInt32.self)
                    for i in 0..<(tokens * 10 * 2560) {
                        try require(
                            a[i] == b[i], "Layer command changed raw contribution at \(i), tokens=\(tokens), exceptional=\(exceptional)")
                    }
                    let weights = (0..<(tokens * 10)).map { Float($0 % 17 + 1) / 128 }
                    let expected = try control.reduce(weights: weights, using: reductions)
                    let actual = try candidate.reduce(weights: weights, using: reductions)
                    for i in actual.indices {
                        try require(actual[i].bitPattern == expected[i].bitPattern, "Layer command changed canonical reduction")
                    }
                    try executor.finishPrefillReadAheadLayer(0)
                    executor.finishPrefillReadAhead()
                    try retired(executor)
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Layer parity fixture retained allocations")
    }

    /// The release plan's experts (prefetch BM32 from 32 rows, the MMA tile from 192): the layer command, whose
    /// prefetch-BM32 tiles run gate, up and the SiLU product in one dispatch, against the separate gate, up and
    /// product dispatches of the eight-job batches, bit for bit, at awkward per-expert row counts (1 to 300, the
    /// 32/192/256 boundaries and partial tiles) and with exceptional inputs.
    package static func gateUpFusion() throws {
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 50_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 32,
                groupedPrefetch: true, groupedMMAMinRows: 192)
            let transfer = try PrefillMoETransfer()
            let executor = try engine(ledger)
            let counts = [1, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 127, 128, 129, 160, 191, 192, 193, 255, 256, 257, 300]
            let tokens = 1100
            // Each rank's tokens split into contiguous runs of awkward lengths, a disjoint expert set per rank.
            var rows: [Int: [(Int, Int)]] = [:]
            var expert = 0
            for rank in 0..<10 {
                var token = 0
                var i = rank * 3
                while token < tokens {
                    let n = min(counts[i % counts.count], tokens - token)
                    rows[expert] = (token..<(token + n)).map { ($0, rank) }
                    expert += 1
                    token += n
                    i += 1
                }
            }
            try require(expert <= 512, "Gate/up fixture used too many experts")
            let experts = rows.keys.sorted()
            var input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
            for (i, bits) in [UInt32(0x7fc3_0000), 0xffc7_0000, 0x7f80_0000, 0xff80_0000, 1, 0x8000_0000].enumerated() {
                input[(i * 211) * 2560 + 17 + i] = Float(bitPattern: bits)
            }
            let budget = { try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens)) }
            let control = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: budget(), transfer: transfer)
            let candidate = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: budget(), transfer: transfer)
            try executor.beginSequence()
            defer { try? executor.finishSequence() }
            // The control: eight-job batches with separate gate, up and product dispatches.
            let referenceWindow = try executor.makeWindow(layer: 0, experts: experts)
            for first in stride(from: 0, to: experts.count, by: 8) {
                let group = Array(experts[first..<min(first + 8, experts.count)])
                try referenceWindow.consumeBatch(experts: group) { buffers in
                    var jobs: [FusedExpertExecution.DeviceJob] = []
                    for (index, expert) in group.enumerated() {
                        let positions = rows[expert]!
                        for lower in stride(from: 0, to: positions.count, by: 256) {
                            jobs.append(
                                FusedExpertExecution.DeviceJob(
                                    expert: expert,
                                    slots: positions[lower..<min(lower + 256, positions.count)].map { UInt32($0.0 * 10 + $0.1) },
                                    buffer: buffers[index], groupedExpert: true, simdMatrix: true))
                        }
                    }
                    for lower in stride(from: 0, to: jobs.count, by: 8) {
                        try fused.evaluateDeviceBatch(Array(jobs[lower..<min(lower + 8, jobs.count)]), workspace: control)
                    }
                }
            }
            let separate = fused.gateUpFusedTiles
            try executor.beginPrefillReadAhead(layers: 1)
            let window = try executor.makeWindow(layer: 0, experts: experts)
            try window.consumeLayer { buffers in
                try fused.evaluateDeviceLayer(
                    experts: experts, buffers: buffers, selectedRows: rows, workspace: candidate, simdMatrix: true, waves: 8)
            }
            let expected = experts.reduce(UInt64(0)) { total, e in
                let n = rows[e]!.count
                return total + UInt64(stride(from: 0, to: n, by: 256).filter { (32..<192).contains(min(256, n - $0)) }.count)
            }
            try require(
                separate == 0 && fused.gateUpFusedTiles == expected && expected > 0,
                "Gate/up fusion ran on \(fused.gateUpFusedTiles) tiles, expected \(expected) (control \(separate))")
            let a = control.contributions.contents().assumingMemoryBound(to: UInt32.self)
            let b = candidate.contributions.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<(tokens * 10 * 2560) where a[i] != b[i] {
                throw EngineError("Gate/up fusion changed contribution \(i) (token \(i / 25600), rank \(i / 2560 % 10))")
            }
            try executor.finishPrefillReadAheadLayer(0)
            executor.finishPrefillReadAhead()
            try retired(executor)
        }
        try require(ledger.snapshot().reserved == 0, "Gate/up fixture retained allocations")
    }

    /// The final layer's row skip (`SerialExecutor.keptExperts`, a `stale` workspace): for random routings, the kept
    /// rows' union runs with complete row sets on layer 1 of a read-ahead that ends at layer 1 (its demand reads past
    /// the end); the kept rows' ten contributions and reductions are bitwise the full union's, and skipped slots keep
    /// the previous values.
    package static func lastLayerRowSkip() throws {
        let ledger = try MemoryLedger(ceiling: 3_000_000_000, transientMargin: 50_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 128)
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let executor = try engine(ledger)
            var seed: UInt64 = 0x5eed
            func random(_ bound: Int) -> Int {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Int(seed >> 33) % bound
            }
            for (tokens, tail, kept) in [(256, nil as Int?, 1), (256, nil, 0), (300, 256, 3), (300, 256, 64)] {
                try autoreleasepool {
                    let extra = tail == nil ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes
                    let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens) + extra)
                    // The adopted workspace passes its buffer on (kept by the budget) once it retires: drop it after.
                    try autoreleasepool {
                        // Random distinct top-ten routing per row.
                        var rows: [Int: [(Int, Int)]] = [:]
                        var routes: [[Int]] = []
                        for token in 0..<tokens {
                            var chosen: [Int] = []
                            while chosen.count < 10 {
                                let e = random(512)
                                if !chosen.contains(e) { chosen.append(e) }
                            }
                            routes.append(chosen)
                            for (rank, expert) in chosen.enumerated() { rows[expert, default: []].append((token, rank)) }
                        }
                        let from = tokens - kept
                        let union = SerialExecutor.keptExperts(rows, from: from)
                        try require(
                            Set(union.keys) == Set(routes[from...].flatMap { $0 })
                                && union.allSatisfy({ rows[$0.key]!.elementsEqual($0.value, by: ==) }),
                            "Kept union differs from the kept rows' experts")
                        let input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
                        let controlBudget = try PrefillMoEBudget(
                            ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens) + extra)
                        let control = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: controlBudget, transfer: transfer)
                        let scopeInput = try budget.buffer(bytes: tokens * 2560 * 4, device: transfer.device)
                        input.withUnsafeBytes { scopeInput.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                        try rejects {
                            _ = try PrefillMoEWorkspace(
                                tokens: tokens, input: scopeInput, selectedRows: union, budget: budget, transfer: transfer)
                        }
                        let candidate = try PrefillMoEWorkspace(
                            tokens: tokens, input: scopeInput, selectedRows: union, budget: budget,
                            transfer: transfer, stale: true)
                        let previous = candidate.contributions.contents().assumingMemoryBound(to: UInt32.self)
                        for i in 0..<(tokens * 10 * 2560) { previous[i] = 0x3f80_0000 &+ UInt32(i % 977) }
                        try executor.beginSequence()
                        defer { try? executor.finishSequence() }
                        try executor.beginPrefillReadAhead(layers: 1)
                        let full = try executor.makeWindow(layer: 0, experts: rows.keys.sorted())
                        try full.consumeLayer { buffers in
                            try fused.evaluateDeviceLayer(
                                experts: full.keys.map(\.expert), buffers: buffers, selectedRows: rows,
                                workspace: control, simdMatrix: true, ungroupedTailStart: tail)
                        }
                        try executor.finishPrefillReadAheadLayer(0)
                        if !union.isEmpty {
                            // Layer 1 is past the read-ahead's end: its records come from demand reads (same bytes per expert).
                            let window = try executor.makeWindow(layer: 1, experts: Array(union.keys))
                            try window.consumeLayer { buffers in
                                try fused.evaluateDeviceLayer(
                                    experts: window.keys.map(\.expert), buffers: buffers, selectedRows: union,
                                    workspace: candidate, simdMatrix: true, ungroupedTailStart: tail, waves: 8)
                            }
                        }
                        try executor.finishPrefillReadAheadLayer(1)
                        executor.finishPrefillReadAhead()
                        let ahead = executor.snapshot().prefillReadAhead
                        try require(
                            (ahead?.submitted ?? .max) <= ModelLayout.routedExperts && ahead?.closed == true, "Read-ahead read past its end"
                        )
                        try require(candidate.completedSlots == tokens * 10, "Skipped layer did not count its stale slots complete")
                        let a = control.contributions.contents().assumingMemoryBound(to: UInt32.self)
                        for token in 0..<tokens {
                            for rank in 0..<10 {
                                let slot = token * 10 + rank
                                let fresh = union[routes[token][rank]] != nil
                                try require(token < from ? true : fresh, "A kept row's expert is outside the union")
                                for i in (slot * 2560)..<((slot + 1) * 2560) {
                                    try require(
                                        previous[i] == (fresh ? a[i] : 0x3f80_0000 &+ UInt32(i % 977)),
                                        "Row skip changed slot \(slot) (fresh \(fresh)), tokens=\(tokens), kept=\(kept)")
                                }
                            }
                        }
                        let weights = (0..<(tokens * 10)).map { Float($0 % 17 + 1) / 128 }
                        let expected = try control.reduce(weights: weights, using: reductions)
                        let actual = try candidate.reduce(weights: weights, using: reductions)
                        for i in (from * 2560)..<(tokens * 2560) {
                            try require(actual[i].bitPattern == expected[i].bitPattern, "Row skip changed a kept row's reduction")
                        }
                        try retired(executor)
                    }
                    budget.dropSpare()
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Row skip fixture retained allocations")
    }

    /// The layer command's buffers (`CommandBudget`) on the largest fixture, the 8,192-token full union, and a mixed tail:
    /// no buffer holds more than the limit, a buffer at every wave boundary (with and without waves) and at
    /// the start limit give the one-buffer command's contributions bit for bit, and a faked abort is retried through
    /// `GPURetry` with the limit halved and the same contributions.
    package static func commandBuffers() throws {
        let ledger = try MemoryLedger(ceiling: 7_000_000_000, transientMargin: 100_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 128)
            let transfer = try PrefillMoETransfer()
            let executor = try engine(ledger)
            defer { fused.commandBudget.limit = CommandBudget.work }
            for (tokens, tail) in [(8192, nil as Int?), (7678, 7424)] {
                let input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
                let rows = selected(tokens, maximum: true)
                let experts = rows.keys.sorted()
                let extra = tail == nil ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes
                func run(limit: Double?, waves: Int = 1) throws -> PrefillMoEWorkspace {
                    let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens) + extra)
                    let workspace = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: budget, transfer: transfer)
                    if let limit { fused.commandBudget.limit = limit }
                    try executor.beginSequence()
                    defer { try? executor.finishSequence() }
                    try executor.beginPrefillReadAhead(layers: 1)
                    defer { executor.finishPrefillReadAhead() }
                    let window = try executor.makeWindow(layer: 0, experts: experts)
                    try window.consumeLayer { buffers in
                        try fused.evaluateDeviceLayer(
                            experts: experts, buffers: buffers, selectedRows: rows, workspace: workspace, simdMatrix: true,
                            ungroupedTailStart: tail, waves: waves)
                    }
                    try executor.finishPrefillReadAheadLayer(0)
                    try require(workspace.completedSlots == tokens * 10, "Layer command left slots uncovered")
                    return workspace
                }
                let bytes = tokens * 10 * 2560 * 4
                try autoreleasepool {
                    let reference = try run(limit: .infinity)
                    // Each candidate is dropped (with its escrow) before the next.
                    func matches(_ what: String, _ candidate: () throws -> PrefillMoEWorkspace) throws {
                        try autoreleasepool {
                            let workspace = try candidate()
                            try require(
                                memcmp(workspace.contributions.contents(), reference.contributions.contents(), bytes) == 0,
                                "\(what) changed contributions, tokens=\(tokens)")
                        }
                    }
                    let start = fused.layerBuffers
                    fused.largestLayerBufferWork = 0
                    try matches("The start limit") { try run(limit: CommandBudget.work) }
                    // A buffer passes the limit only with a single wave larger than it; these waves are far smaller.
                    try require(
                        fused.layerBuffers - start > 1 && fused.largestLayerBufferWork <= CommandBudget.work,
                        "Layer command held \(fused.largestLayerBufferWork) in a buffer (\(fused.layerBuffers - start) buffers)")
                    for waves in [1, 8] {
                        let jobs = fused.snapshot().submittedJobs
                        let buffers = fused.layerBuffers
                        try matches("A buffer at every wave of \(waves)") { try run(limit: 1, waves: waves) }
                        let tiles = Int(fused.snapshot().submittedJobs - jobs)
                        try require(
                            Int(fused.layerBuffers - buffers) == (tiles + waves - 1) / waves,
                            "Layer command did not end a buffer at every wave (\(fused.layerBuffers - buffers) for \(tiles) tiles)")
                    }
                    // A faked abort: the attempt fails as gpu_interrupted, publishes nothing, and the retry runs half the buffer.
                    fused.commandBudget.limit = CommandBudget.work
                    fused.interruptWait = 0
                    let retry = GPURetry(log: nil)
                    try matches("The retried layer command") { try retry.run { try autoreleasepool { try run(limit: nil) } } }
                    try require(
                        retry.attempt == 1 && fused.interruptWait == nil && fused.commandBudget.limit <= CommandBudget.work / 2,
                        "The faked abort was not retried once with smaller buffers (limit \(fused.commandBudget.limit))")
                }
                try retired(executor)
            }
        }
        try require(ledger.snapshot().reserved == 0, "Bounded layer fixture retained allocations")
    }

    package static func cancellationAndPublication() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 10_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 128)
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let executor = try engine(ledger)
            for (tokens, tail) in [(17, nil as Int?), (273, 256)] {
                let rows = selected(tokens, maximum: false)
                let experts = rows.keys.sorted()
                for mode in 0..<3 {
                    try autoreleasepool {
                        let budget = try PrefillMoEBudget(
                            ledger: ledger,
                            bytes: PrefillMoEBudget.bytesForScope(tokens: tokens)
                                + (tail == nil ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes))
                        let workspace = try PrefillMoEWorkspace(
                            input: .init(repeating: 0.5, count: tokens * 2560),
                            selectedRows: rows, budget: budget, transfer: transfer)
                        try executor.beginSequence()
                        try executor.beginPrefillReadAhead(layers: 1)
                        defer { try? executor.finishSequence() }
                        let window = try executor.makeWindow(layer: 0, experts: experts)
                        let before = fused.snapshot()
                        var caught = false
                        var calls = 0
                        do {
                            try window.consumeLayer { buffers in
                                if mode == 2 {
                                    var invalid = rows
                                    invalid[0]![0] = invalid[0]![1]
                                    try rejects {
                                        try fused.evaluateDeviceLayer(
                                            experts: experts, buffers: buffers,
                                            selectedRows: invalid, workspace: workspace, simdMatrix: true, ungroupedTailStart: tail)
                                    }
                                    // Rejected validation must roll its staged bits
                                    // back so the unchanged valid map still works.
                                    try fused.evaluateDeviceLayer(
                                        experts: experts, buffers: buffers,
                                        selectedRows: rows, workspace: workspace, simdMatrix: true, ungroupedTailStart: tail)
                                    throw EngineError("Injected callback failure after joined computation")
                                }
                                try fused.evaluateDeviceLayer(
                                    experts: experts, buffers: buffers,
                                    selectedRows: rows, workspace: workspace, simdMatrix: true, ungroupedTailStart: tail,
                                    shouldContinue: {
                                        calls += 1
                                        return mode == 0 ? calls < 4 : fused.snapshot().completedCommands == before.completedCommands
                                    })
                            }
                        } catch is IOOperationCancelled { caught = true } catch is EngineError {
                            if mode == 2 { caught = true } else { throw EngineError("Unexpected cancellation error") }
                        }
                        try require(caught, "Layer cancellation/throw did not propagate")
                        let after = fused.snapshot()
                        try require(
                            after.submittedCommands == before.submittedCommands + (mode == 0 ? 0 : 1)
                                && after.completedCommands == after.submittedCommands, "Cancellation returned before GPU join")
                        if mode != 2 {
                            try require(workspace.completedSlots == 0, "Cancelled layer published coverage")
                            try rejects { _ = try workspace.reduce(weights: .init(repeating: 0.1, count: tokens * 10), using: reductions) }
                        }
                        executor.finishPrefillReadAhead()
                        try retired(executor)
                    }
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Cancelled layer retained allocations")
    }

    package static func completeAdmission() throws {
        let url = CheckFixtures.file("config/prefill-layer-command-8192.json")
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: path) }
        let config = try RuntimeConfiguration.load(url)
        let fixed = try ConversationGeometry(context: 8192).maximumFixedBytes(
            input: 8192,
            residentBytes: config.residentBytes + config.denseResidentBytes)
        let owners = fixed + config.experts.replicaOwnerBytes + config.experts.readAheadOwnerBytes + config.experts.layerCommandOwnerBytes
        // The fixture stays below its 15 GB usable ceiling (16 GB less a 1 GB margin); the first prefill cache in
        // whole 8-record slabs that does not fit beside the owners, the layer command's included, is refused.
        let slab = UInt64(8 * PackStore.expertBytes)
        let refused = Int((15_000_000_000 - owners) / slab) * 8 + 8
        try require(
            config.experts.prefillLayerCommand && owners + config.experts.prefillCacheBytes <= 15_000_000_000,
            "Layer command owner is absent from admission")
        for (key, value) in [
            ("prefill_expert_cache_slots", refused as Any), ("expert_prefill_read_ahead", false),
            ("moe_device_prefill", false), ("expert_prefill_layer_command", 1),
        ] {
            var object = original
            var execution = original["execution"] as! [String: Any]
            execution[key] = value
            object["execution"] = execution
            try JSONSerialization.data(withJSONObject: object).write(to: path)
            try rejects { _ = try RuntimeConfiguration.load(path) }
        }
    }
}
