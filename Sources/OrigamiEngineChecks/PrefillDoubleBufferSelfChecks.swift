import Foundation
import Metal
import OrigamiEngine

/// Standalone checks for `prefill_double_buffer`, without original weights:
/// - the tile pipeline issues tile t+1 before finishing tile t, finishes in order, discards and
///   re-issues the successor of a replayed tile, and reports a failure found at a deferred join;
/// - the discarded successor's scope input is restored, so its re-issue reads the original rows;
/// - the GPU narrow with an input offset equals `BF16.encode` bit for bit over exceptional values;
/// - a GDN tile chained on the GPU from the previous tile's provisional recurrence and projected
///   rows (`WholeLayerExecution.GDNChain` primitives) yields the same bytes as the sequential tile
///   encoded from the host publication of that previous tile;
/// - the GDN-only second workspace keeps every non-QSA slot and allocates no QSA data slot.
package enum PrefillDoubleBufferSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        if try !value() { throw EngineError(message) }
    }

    package static func pipeline() throws {
        enum Event: Equatable { case issue(Int, Int, Int?), finish(Int), discard(Int), replay(Int) }
        struct Fixture {
            var events: [Event] = []
            var invalid: Set<Int> = [], failing: Set<Int> = []
        }
        func run(_ tiles: Int, invalid: Set<Int> = [], failing: Set<Int> = []) throws -> (
            events: [Event], counts: PrefillTilePipeline.Counts
        ) {
            var fixture = Fixture(invalid: invalid, failing: failing)
            var finished: [Int] = []
            let counts = try PrefillTilePipeline.run(
                tiles: tiles,
                issue: { tile, slot, predecessor -> Int in
                    fixture.events.append(.issue(tile, slot, predecessor))
                    return tile
                },
                finish: { tile, ticket -> Bool in
                    try require(ticket == tile, "Pipeline finished a ticket of another tile")
                    fixture.events.append(.finish(tile))
                    if fixture.failing.contains(tile) { throw EngineError("deferred join failure \(tile)") }
                    finished.append(tile)
                    if fixture.invalid.contains(tile) {
                        fixture.invalid.remove(tile)
                        return false
                    }
                    return true
                },
                discard: { tile, ticket in
                    try require(ticket == tile, "Pipeline discarded a ticket of another tile")
                    fixture.events.append(.discard(tile))
                }, replay: { tile in fixture.events.append(.replay(tile)) })
            try require(
                finished == finished.sorted() && Set(finished).count == finished.count || !invalid.isEmpty,
                "Pipeline finished tiles out of order")
            return (fixture.events, counts)
        }
        // Every tile is finished only after its successor was issued, in order, alternating slots.
        let plain = try run(5)
        try require(
            plain.events == [
                .issue(0, 0, nil), .issue(1, 1, 0), .finish(0), .issue(2, 0, 1), .finish(1), .issue(3, 1, 2),
                .finish(2), .issue(4, 0, 3), .finish(3), .finish(4),
            ], "Pipeline order differs: \(plain.events)")
        try require(plain.counts == .init(deferred: 4, replays: 0, reissued: 0), "Pipeline counts differ: \(plain.counts)")
        // A replayed tile: its already issued successor is discarded before the replay and re-issued unchained.
        let replayed = try run(5, invalid: [2])
        try require(
            replayed.events == [
                .issue(0, 0, nil), .issue(1, 1, 0), .finish(0), .issue(2, 0, 1), .finish(1), .issue(3, 1, 2),
                .finish(2), .discard(3), .replay(2), .issue(3, 1, nil), .issue(4, 0, 3), .finish(3), .finish(4),
            ],
            "Pipeline replay sequence differs: \(replayed.events)")
        try require(replayed.counts == .init(deferred: 3, replays: 1, reissued: 1), "Pipeline replay counts differ: \(replayed.counts)")
        // A replayed last tile has no successor to discard.
        let last = try run(2, invalid: [1])
        try require(
            last.events == [.issue(0, 0, nil), .issue(1, 1, 0), .finish(0), .finish(1), .replay(1)]
                && last.counts == .init(deferred: 1, replays: 1, reissued: 0), "Pipeline last-tile replay differs: \(last.events)")
        // Two consecutive replays, including the re-issued tile itself.
        let twice = try run(4, invalid: [1, 2])
        try require(
            twice.events == [
                .issue(0, 0, nil), .issue(1, 1, 0), .finish(0), .issue(2, 0, 1), .finish(1), .discard(2), .replay(1),
                .issue(2, 0, nil), .issue(3, 1, 2), .finish(2), .discard(3), .replay(2), .issue(3, 1, nil), .finish(3),
            ],
            "Pipeline consecutive replays differ: \(twice.events)")
        try require(twice.counts == .init(deferred: 1, replays: 2, reissued: 2), "Pipeline consecutive replay counts differ")
        // A failure found at the deferred join is reported from the finishing call, after the successor was issued.
        var reported: String?
        do { _ = try run(3, failing: [1]) } catch let error as EngineError { reported = error.description }
        try require(reported?.contains("deferred join failure 1") == true, "Deferred join failure was not reported")
        var observed: [Event] = []
        _ = try? PrefillTilePipeline.run(
            tiles: 3,
            issue: { tile, slot, predecessor -> Int in
                observed.append(.issue(tile, slot, predecessor))
                return tile
            },
            finish: { tile, _ in
                observed.append(.finish(tile))
                if tile == 1 { throw EngineError("x") }
                return true
            },
            discard: { _, _ in throw EngineError("unexpected discard") }, replay: { _ in throw EngineError("unexpected replay") })
        try require(
            observed == [.issue(0, 0, nil), .issue(1, 1, 0), .finish(0), .issue(2, 0, 1), .finish(1)],
            "Deferred failure was not discovered after the successor's issue: \(observed)")
        // Degenerate counts.
        let single = try run(1)
        try require(single.events == [.issue(0, 0, nil), .finish(0)] && single.counts == .init(), "Single tile pipeline differs")
        try require(try run(0).events.isEmpty, "Empty pipeline issued work")
        var refused = false
        do { _ = try run(-1) } catch is EngineError { refused = true }
        try require(refused, "Negative tile count was accepted")
    }

    private static func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
        guard
            let result = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
            )
        else {
            throw EngineError("Double-buffer check allocation failed")
        }
        return result
    }
    private static func zeroed(_ bytes: Int, _ device: MTLDevice) throws -> MTLBuffer {
        guard let result = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Double-buffer check allocation failed")
        }
        result.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        return result
    }
    private static func bytes(_ buffer: MTLBuffer, _ count: Int, offset: Int = 0) -> [UInt8] {
        Array(UnsafeRawBufferPointer(start: buffer.contents().advanced(by: offset), count: count))
    }
    private static func joined(_ command: MTLCommandBuffer) throws {
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw EngineError.gpu("Double-buffer fixture command failed", command.error) }
    }

    package static func exactness() throws {
        let ledger = try MemoryLedger(ceiling: 1_200_000_000, transientMargin: 1_000_000)
        let charge = try ledger.reserve(bytes: 500_000_000, category: "prefill_double_buffer_fixture")
        defer { try? ledger.release(charge) }
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let device = kernels.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("Double-buffer fixture queue unavailable") }
            let dense = try PinnedDenseLinear(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let preparation = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let gdn = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: preparation, ledger: ledger, simdMatrix: true)
            let scalars = try WholeExecutionScalars(
                device: device, sigmoidTable: zeroed(65536 * 2, device), siluTable: zeroed(65536 * 2, device))

            // 1. GPU narrow with an input offset equals BF16.encode bit for bit, including every
            //    exceptional class, and leaves the bytes before the offset unread and the output exact.
            let special: [UInt32] = [
                0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001, 0xffa1_2345, 0x7fff_ffff,
                0x0000_0001, 0x8000_0001, 0x007f_ffff, 0x0080_0000, 0x3f80_7fff, 0x3f80_8000, 0x3f80_8001,
                0x3f81_7fff, 0x3f81_8000, 0x3f81_8001, 0xbf80_8000, 0x7f7f_ffff, 0x7f7f_8000, 0x4b00_ffff,
            ]
            let count = 3 * 10240
            let lead = 5 * 10240
            let input: [UInt32] = (0..<(lead + count)).map { i in
                i % 7 == 0 ? special[i % special.count] : (UInt32(truncatingIfNeeded: i) &* 2_654_435_761) ^ UInt32(i << 3)
            }
            let source = try buffer(input, device)
            let narrowed = try zeroed(count * 2 + 64, device)
            narrowed.contents().advanced(by: count * 2).initializeMemory(as: UInt8.self, repeating: 0xa5, count: 64)
            guard let command = queue.makeCommandBuffer() else { throw EngineError("Narrow fixture command unavailable") }
            try scalars.encodeNarrow(command, input: source, inputOffset: lead * 4, output: narrowed, count: count)
            try joined(command)
            let expected = input[lead...].map { BF16.encode(Float(bitPattern: $0)) }
            let actual = Array(UnsafeBufferPointer(start: narrowed.contents().assumingMemoryBound(to: UInt16.self), count: count))
            try require(actual == expected, "GPU narrow with offset differs from BF16.encode")
            try require(bytes(narrowed, 64, offset: count * 2).allSatisfy { $0 == 0xa5 }, "GPU narrow wrote past its extent")
            for (offset, label) in [
                (2, "Unaligned narrow offset"), (lead * 4 + 4, "Narrow offset past the input extent"), (-4, "Negative narrow offset"),
            ] {
                guard let probe = queue.makeCommandBuffer() else { throw EngineError("Narrow probe command unavailable") }
                var refused = false
                do {
                    try scalars.encodeNarrow(probe, input: source, inputOffset: offset, output: narrowed, count: count)
                } catch is EngineError { refused = true }
                try require(refused && probe.status == .notEnqueued, label + " was accepted")
            }

            // 2. A chained GDN tile equals the sequential tile. Tile 0 runs once; tile 1 runs twice
            //    with other inputs: from the host publication of tile 0 (recurrence copied, history
            //    BF16-encoded on the host) and chained on the GPU from tile 0's buffers, submitted
            //    before tile 0 completes.
            func projection(_ width: Int, _ outputs: Int, _ seed: UInt32) throws -> GDNDeviceBlock.Projection {
                let words = (0..<(outputs * width / 8)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
                let scales = (0..<(outputs * width / 64)).map { BF16.encode(Float($0 % 5 + 1) / 2048) }
                let biases = (0..<(outputs * width / 64)).map { BF16.encode(-Float($0 % 5 + 1) * 7.5 / 2048) }
                return try .init(words: buffer(words, device), scales: buffer(scales, device), biases: buffer(biases, device))
            }
            let qkv = try projection(2560, 10240, 17)
            let z = try projection(2560, 6144, 731)
            let out = try projection(6144, 2560, 83)
            let b = try buffer((0..<(48 * 2560)).map { (i: Int) in LayerArithmetic.round(Float((i * 31) % 127 - 63) / 4096) }, device)
            let a = try buffer((0..<(48 * 2560)).map { (i: Int) in LayerArithmetic.round(Float((i * 73) % 109 - 54) / 4096) }, device)
            let convolution = (0..<40960).map { (i: Int) in LayerArithmetic.round(Float((i * 17 + 7) % 251 - 125) / 128) }
            let bias = (0..<48).map { Float($0 % 5 - 2) / 16 }
            let logs = (0..<48).map { Float($0 % 7 - 3) / 16 }
            let norm = (0..<128).map { Float($0 % 7 + 1) / 8 }
            struct Tile {
                let buffers: GDNDeviceBlock.RowBuffers, partials: GDNDeviceBlock.TilePartials
                let input, bits, recurrence: MTLBuffer
            }
            func tile(rows: Int, seed: Int, history: [UInt16]?) throws -> Tile {
                let p = try gdn.tilePartialBytes(rows: rows)
                func partial(_ bytes: Int) throws -> MTLBuffer? { bytes == 0 ? nil : try zeroed(bytes, device) }
                let x: [Float] = (0..<(rows * 2560)).map { Float(($0 * 73 + seed) % 1009 - 504) / 512 }
                let buffers = try GDNDeviceBlock.RowBuffers(
                    mixed: zeroed(rows * 10240 * 4, device), z: zeroed(rows * 6144 * 4, device),
                    b: zeroed(rows * 48 * 4, device), a: zeroed(rows * 48 * 4, device),
                    history: history.map { try buffer($0, device) } ?? zeroed(3 * 10240 * 2, device),
                    convolution: buffer(convolution, device), bias: buffer(bias, device), logs: buffer(logs, device),
                    prepared: zeroed(rows * 10240 * 4, device), beta: zeroed(rows * 48 * 4, device), decay: zeroed(rows * 48 * 4, device),
                    prepareFlags: zeroed(4, device), mask: buffer([UInt32](repeating: 1, count: rows), device),
                    normWeights: buffer(norm, device),
                    normed: zeroed(rows * 6144 * 4, device), gated: zeroed(rows * 6144 * 4, device), gateFlags: zeroed(4, device),
                    narrowed: zeroed(rows * 6144 * 2, device), output: zeroed(rows * 2560 * 4, device), convolved: nil)
                return try Tile(
                    buffers: buffers,
                    partials: .init(
                        qkv: partial(p.qkv), z: partial(p.z), b: partial(p.dense), a: partial(p.dense), output: partial(p.output)),
                    input: buffer(x, device), bits: buffer(x.map(BF16.encode), device),
                    recurrence: zeroed((rows * 6144 + 48 * 128 * 128) * 4, device))
            }
            func encode(_ command: MTLCommandBuffer, _ t: Tile, rows: Int, initial: MTLBuffer, offset: Int) throws {
                try gdn.encodeTile(
                    command, rows: rows, inputFP32: t.input, inputBF16: t.bits, qkv: qkv, z: z, b: b, a: a, out: out,
                    initial: initial, initialOffset: offset, recurrence: t.recurrence, buffers: t.buffers, partials: t.partials)
            }
            for rows in [3, 256] {
                try autoreleasepool {
                    let initialState = (0..<(48 * 128 * 128)).map { Float(($0 % 13 - 6) + rows) / 1024 }
                    let initialTail = (0..<30720).map { BF16.encode(Float($0 % 17 - 8) / 64) }
                    let first = try tile(rows: rows, seed: 5, history: initialTail)
                    let chained = try tile(rows: rows, seed: 977, history: nil)
                    guard let firstCommand = queue.makeCommandBuffer(), let chainCommand = queue.makeCommandBuffer() else {
                        throw EngineError("GDN chain fixture commands unavailable")
                    }
                    try encode(firstCommand, first, rows: rows, initial: buffer(initialState, device), offset: 0)
                    // The chained tile is encoded and committed while tile 0 is still running: its
                    // history narrows tile 0's last three projected rows and its initial state is
                    // tile 0's provisional recurrence, both read in queue order.
                    try scalars.encodeNarrow(
                        chainCommand, input: first.buffers.mixed, inputOffset: (rows - 3) * 10240 * 4,
                        output: chained.buffers.history, count: 3 * 10240)
                    try encode(chainCommand, chained, rows: rows, initial: first.recurrence, offset: rows * 6144 * 4)
                    firstCommand.commit()
                    chainCommand.commit()
                    firstCommand.waitUntilCompleted()
                    chainCommand.waitUntilCompleted()
                    try require(firstCommand.status == .completed && chainCommand.status == .completed, "GDN chain fixture commands failed")
                    try require(
                        first.buffers.prepareFlags.contents().load(as: UInt32.self) == 0
                            && first.buffers.gateFlags.contents().load(as: UInt32.self) == 0,
                        "GDN chain fixture tile 0 requested repair")
                    // The sequential tile 1: the host publication of tile 0, exactly as WholeLayerExecution.Pending.publish derives it.
                    let mixed = first.buffers.mixed.contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 10240 - 30720)
                    let publishedTail = (0..<30720).map { BF16.encode(mixed[$0]) }
                    let publishedState = Array(
                        UnsafeBufferPointer(
                            start: first.recurrence.contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 6144),
                            count: 48 * 128 * 128))
                    let sequential = try tile(rows: rows, seed: 977, history: publishedTail)
                    guard let sequentialCommand = queue.makeCommandBuffer() else {
                        throw EngineError("GDN sequential fixture command unavailable")
                    }
                    try encode(sequentialCommand, sequential, rows: rows, initial: buffer(publishedState, device), offset: 0)
                    try joined(sequentialCommand)
                    try require(
                        bytes(chained.buffers.history, 61440) == bytes(sequential.buffers.history, 61440),
                        "Chained GDN history differs from the host publication (rows \(rows))")
                    let s = sequential.buffers
                    let c = chained.buffers
                    for (name, x, y, count) in [
                        ("output", c.output, s.output, rows * 2560 * 4),
                        ("recurrence", chained.recurrence, sequential.recurrence, (rows * 6144 + 48 * 128 * 128) * 4),
                        ("mixed", c.mixed, s.mixed, rows * 10240 * 4), ("prepared", c.prepared, s.prepared, rows * 10240 * 4),
                        ("normed", c.normed, s.normed, rows * 6144 * 4), ("gated", c.gated, s.gated, rows * 6144 * 4),
                        ("beta", c.beta, s.beta, rows * 48 * 4), ("decay", c.decay, s.decay, rows * 48 * 4),
                        ("flags", c.prepareFlags, s.prepareFlags, 4), ("gateFlags", c.gateFlags, s.gateFlags, 4),
                    ] {
                        try require(
                            bytes(x, count) == bytes(y, count), "Chained GDN tile \(name) differs from the sequential tile (rows \(rows))")
                    }
                    try require(
                        s.prepareFlags.contents().load(as: UInt32.self) == 0 && s.gateFlags.contents().load(as: UInt32.self) == 0,
                        "GDN chain fixture tile 1 requested repair")
                    // The comparison is not vacuous: both tiles produced finite non-zero outputs, tile 0
                    // moved the recurrence away from the initial state, and tile 1 moved it again.
                    let chainedOutput = Array(
                        UnsafeBufferPointer(start: c.output.contents().assumingMemoryBound(to: Float.self), count: rows * 2560))
                    let chainedState = Array(
                        UnsafeBufferPointer(
                            start: chained.recurrence.contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 6144),
                            count: 48 * 128 * 128))
                    try require(
                        chainedOutput.allSatisfy(\.isFinite) && chainedOutput.contains { $0 != 0 } && publishedState != initialState
                            && chainedState != publishedState && publishedTail != initialTail,
                        "GDN chain fixture produced trivial tiles (rows \(rows))")
                    // Tile 0's buffers are unchanged by the chained reader.
                    try require(
                        Array(
                            UnsafeBufferPointer(
                                start: first.recurrence.contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 6144),
                                count: 48 * 128 * 128)) == publishedState,
                        "Chained reader changed tile 0's recurrence")
                }
            }

            // 3. The GDN-only second workspace: every non-QSA slot keeps the ordinary C256 size,
            //    no QSA data slot is allocated, the flag slots remain, and it admits fewer bytes.
            let hc = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true)
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            for capacity in [8192, 32768] {
                let full = try WholeExecutionWorkspace.layout(
                    rows: 256, capacity: capacity, capture: false, argumentBytes: 4096,
                    kernels: kernels, simdMatrix: true, hc: hc, gdn: gdn, attention: attention)
                let reduced = try WholeExecutionWorkspace.layout(
                    rows: 256, capacity: capacity, capture: false, argumentBytes: 4096,
                    kernels: kernels, simdMatrix: true, hc: hc, gdn: gdn, attention: attention, gdnOnly: true)
                let qsa = Set(WholeExecutionWorkspace.qsaDataSlots.map(\.rawValue))
                for slot in WholeExecutionWorkspace.Slot.allCases {
                    let expectedSize = qsa.contains(slot.rawValue) ? 0 : full.sizes[slot.rawValue]
                    try require(
                        reduced.sizes[slot.rawValue] == expectedSize, "GDN-only layout slot \(slot) differs at capacity \(capacity)")
                }
                try require(
                    !full.gdnOnly && reduced.gdnOnly && reduced.admittedBytes < full.admittedBytes
                        && reduced.sizes[WholeExecutionWorkspace.Slot.qsaGateFlags.rawValue] == 4
                        && reduced.sizes[WholeExecutionWorkspace.Slot.gdnNext.rawValue] == (256 * 6144 + 48 * 128 * 128) * 4,
                    "GDN-only layout admission or kept slots differ at capacity \(capacity)")
                let before = ledger.snapshot().reserved
                try autoreleasepool {
                    let workspace = try WholeExecutionWorkspace(device: device, ledger: ledger, layout: reduced)
                    workspace.clearFlags()
                    try require(
                        workspace.optional(.qsaScores) == nil && workspace.optional(.qsaAttentionScores) == nil
                            && workspace.optional(.gdnHistory)?.length == 61440 && workspace.optional(.result)?.length == 256 * 10240 * 4
                            && ledger.snapshot().reserved == before + reduced.admittedBytes, "GDN-only workspace allocation differs")
                }
                try require(ledger.snapshot().reserved == before, "GDN-only workspace did not retire")
            }
            var refusedLayout = false
            do {
                _ = try WholeExecutionWorkspace.layout(
                    rows: 1, capacity: 8192, capture: false, argumentBytes: 4096,
                    kernels: kernels, simdMatrix: true, hc: hc, gdn: gdn, attention: attention, gdnOnly: true)
            } catch is EngineError { refusedLayout = true }
            try require(refusedLayout, "A GDN-only row-one layout was accepted")
        }
    }

    /// Regression: a replayed tile's discarded successor had already copied its provisional
    /// after-attention rows over its own scope input, so the unchained re-issue read them as its
    /// layer input. Real scope and workspaces (only the slots the scope blits bind; attention
    /// simulated) driven by the pipeline: every issued tile must read the original input.
    package static func replayRestore() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Metal unavailable")
        }
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let scope = try WholePrefillScope(device: device, ledger: ledger, layout: WholePrefillScope.layout(tokens: 1536)!)
            var sizes = [Int](repeating: 0, count: WholeExecutionWorkspace.Slot.allCases.count)
            for (slot, width) in [
                (WholeExecutionWorkspace.Slot.residual, 10240), (.afterAttention, 10240), (.hcMMixed, 2560),
                (.hcMInjection, 4), (.routerWeights, 10), (.sharedOutput, 2560), (.sharedSigmoid, 1),
            ] {
                sizes[slot.rawValue] = 256 * width * 4
            }
            let total = sizes.reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1) }
            let layout = WholeExecutionWorkspace.Layout(
                rows: 256, capacity: 8192, capture: false, sizes: sizes, bufferBytes: total, admittedBytes: total)
            let workspaces = try (0..<2).map { _ in try WholeExecutionWorkspace(device: device, ledger: ledger, layout: layout) }
            let n = 256 * 10240
            func original(_ tile: Int) -> [UInt32] {
                (tile * n..<(tile + 1) * n).map { UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) & 0x7fff_ffff }
            }
            func filled(_ tag: UInt32) -> [UInt32] { [UInt32](repeating: tag, count: n) }
            func words(_ buffer: MTLBuffer, tile: Int = 0) -> [UInt32] {
                Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self).advanced(by: tile * n), count: n))
            }
            func store(_ values: [UInt32], _ buffer: MTLBuffer, tile: Int = 0) {
                values.withUnsafeBytes {
                    buffer.contents().advanced(by: tile * n * 4).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
                }
            }
            for tile in 0..<6 { store(original(tile), scope[.residual], tile: tile) }

            // Refusals leave the scope unchanged: an unjoined command and starts outside the tile grid.
            guard let idle = queue.makeCommandBuffer() else { throw EngineError("Restore fixture command unavailable") }
            try require(
                !scope.restoreTileInput(start: 256, workspace: workspaces[1], command: idle), "Restore accepted an unjoined command")
            try joined(idle)
            for start in [-256, 1, 1536] {
                try require(
                    !scope.restoreTileInput(start: start, workspace: workspaces[1], command: idle), "Restore accepted start \(start)")
            }
            try require((0..<6).allSatisfy { words(scope[.residual], tile: $0) == original($0) }, "Refused restore changed scope bytes")

            // Tile 0 requires the replay; its successor, tile 1, is discarded and re-issued.
            struct Ticket { let start: Int, workspace: WholeExecutionWorkspace, command: MTLCommandBuffer }
            var issued: [Int] = []
            let counts = try PrefillTilePipeline.run(
                tiles: 3,
                issue: { tile, slot, _ -> Ticket in
                    let w = workspaces[slot]
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("Restore fixture command unavailable") }
                    store(filled(0x8000_0000 | UInt32(tile)), w[.afterAttention])  // the attention's provisional output
                    try scope.encodeTileInput(command, start: tile * 256, workspace: w)
                    try scope.encodeTileOutputs(command, start: tile * 256, workspace: w)
                    command.commit()
                    issued.append(tile)
                    return Ticket(start: tile * 256, workspace: w, command: command)
                },
                finish: { tile, ticket -> Bool in
                    ticket.command.waitUntilCompleted()
                    try require(ticket.command.status == .completed, "Restore fixture command failed")
                    try require(words(ticket.workspace[.residual]) == original(tile), "Issued tile \(tile) read a provisional layer input")
                    return tile != 0
                },
                discard: { tile, ticket in
                    ticket.command.waitUntilCompleted()
                    try require(ticket.command.status == .completed, "Restore fixture command failed")
                    try require(
                        words(scope[.residual], tile: tile) == filled(0x8000_0000 | UInt32(tile)),
                        "Discarded successor did not overwrite its scope input")
                    try require(
                        scope.restoreTileInput(start: ticket.start, workspace: ticket.workspace, command: ticket.command),
                        "Joined restore was refused")
                    try require(words(scope[.residual], tile: tile) == original(tile), "Restore did not return the original layer input")
                }, replay: { tile in store(filled(0x4000_0000 | UInt32(tile)), scope[.residual], tile: tile) })
            try require(
                counts == .init(deferred: 1, replays: 1, reissued: 1) && issued == [0, 1, 1, 2],
                "Restore fixture pipeline differs: \(counts) \(issued)")
            let expected = [filled(0x4000_0000), filled(0x8000_0001), filled(0x8000_0002), original(3), original(4), original(5)]
            try require(
                (0..<6).allSatisfy { words(scope[.residual], tile: $0) == expected[$0] }, "Scope rows after the replayed pipeline differ")
        }
        try require(ledger.snapshot().reserved == 0, "Restore fixture retained its owners")
    }
}
