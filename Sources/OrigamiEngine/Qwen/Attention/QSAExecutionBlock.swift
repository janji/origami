import Foundation
import Metal

/// One attention layer's pooled index as an argument table of its BF16 pages (64 blocks each), which the indexer
/// reads in place instead of a whole-index fp32 copy every token. Only bindings whose page changed are rewritten
/// (the last, partial page every fourth token, a new page every 256 tokens, everything after a conversation switch).
package final class QSAPooledPageTable {
    static let maximumPages = 1024
    package let table: MTLBuffer
    private let arguments: MTLArgumentEncoder, ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    private var bound: [(buffer: MTLBuffer, offset: Int)] = []
    private var last: MTLCommandBuffer?

    package init(device: MTLDevice, ledger: MemoryLedger, arguments: MTLArgumentEncoder) throws {
        let bytes = (arguments.encodedLength + 16383) / 16384 * 16384
        guard device.argumentBuffersSupport == .tier2, arguments.encodedLength > 0, bytes <= 131072 else {
            throw EngineError("Invalid pooled QSA page-table geometry")
        }
        // The table, plus the bound list at 1,024 pages.
        reservation = try ledger.reserve(bytes: UInt64(bytes + 32768), category: "qsa_pooled_page_table")
        guard let table = device.makeBuffer(length: bytes, options: .storageModeShared), table.allocatedSize <= bytes else {
            try? ledger.release(reservation)
            throw EngineError("Pooled QSA page table allocation failed", code: "insufficient_memory")
        }
        table.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        self.table = table
        self.arguments = arguments
        self.ledger = ledger
    }
    deinit { try? ledger.release(reservation) }

    /// Points the table at `pages` for `command` and returns the buffers it must make resident.
    package func bind(_ pages: [(buffer: MTLBuffer, offset: Int)], command: MTLCommandBuffer) throws -> [MTLResource] {
        guard pages.count <= Self.maximumPages, pages.allSatisfy({ $0.buffer.device.registryID == table.device.registryID }) else {
            throw EngineError("Pooled QSA pages exceed their table")
        }
        // Layers publish in order, so the previous token's reader has joined; a running one is waited for.
        if let last, [.committed, .scheduled].contains(last.status) { try last.waitBounded() }
        last = command
        arguments.setArgumentBuffer(table, offset: 0)
        defer { arguments.setArgumentBuffer(nil, offset: 0) }
        for index in 0..<max(pages.count, bound.count) {
            let next = index < pages.count ? pages[index] : nil
            if index >= bound.count || next == nil || next!.buffer !== bound[index].buffer || next!.offset != bound[index].offset {
                arguments.setBuffer(next?.buffer, offset: next?.offset ?? 0, index: index)
            }
        }
        bound = pages
        // Arena pages are distinct buffers; a host-held index repeats one buffer in adjacent entries.
        var resources: [MTLResource] = []
        resources.reserveCapacity(pages.count)
        for page in pages where resources.last !== page.buffer { resources.append(page.buffer) }
        return resources
    }
}

/// Encodes the complete original QSA arithmetic into an enclosing command.
/// The command may also contain the preceding MoE and following HC/router;
/// publication is therefore a separate, explicitly joined transaction step.
/// Order: Q/gate, K, V, index projections; split; Q/K RMSNorm; RoPE; BF16 narrowing; index
/// queries and 4-token pooled block keys; indexer scores (keys > 2048); top-512 block selection;
/// attention over <= 2051 keys; sigmoid gate; output projection. Pool, selection and gate kernels
/// set flags on values the host path handles differently; a flagged layer is replayed on the host.
package final class QSAExecutionBlock {
    private let kernels: ReferenceKernels, rope: PinnedRoPE, indexer: PinnedIndexer
    private let dense: PinnedDenseLinear
    private let attention: PinnedAttention, scalar: QSAExecutionScalars, selection: QSADeviceSelection
    private let simdMatrix: Bool
    private let pooledTables: [Int: QSAPooledPageTable]

    init(
        kernels: ReferenceKernels, dense: PinnedDenseLinear, rope: PinnedRoPE, indexer: PinnedIndexer, attention: PinnedAttention,
        scalar: QSAExecutionScalars, selection: QSADeviceSelection, simdMatrix: Bool, ledger: MemoryLedger
    ) throws {
        self.kernels = kernels
        self.dense = dense
        self.rope = rope
        self.indexer = indexer
        self.attention = attention
        self.scalar = scalar
        self.selection = selection
        self.simdMatrix = simdMatrix
        var tables: [Int: QSAPooledPageTable] = [:]
        for layer in stride(from: 3, to: ModelContract.layers, by: 4) {
            tables[layer] = try QSAPooledPageTable(device: kernels.device, ledger: ledger, arguments: indexer.pooledArguments)
        }
        pooledTables = tables
    }

    /// One encoded layer awaiting its command: `publish` appends K/V/pooled rows to the state and commits
    /// the mirror or page table; `discard` retires them. One of the two must run on every path.
    final class Pending {
        let command: MTLCommandBuffer
        fileprivate let workspace: WholeExecutionWorkspace
        fileprivate let weights: WholeLayerWeights, rope: PinnedRoPE, tables: [PinnedRoPE.EncodedTables]
        fileprivate let mirror: QSADecodeStorage?, append: QSADecodeStorage.EncodedAppend?
        fileprivate let pagedRead: QSAPagedAttentionStorage.Read?
        fileprivate let offset, rows: Int
        private var finished = false
        /// The router signal of this command fired (queued-experts decode), so its flags are final although later
        /// work queued in it runs on.
        var routerSignaled = false
        fileprivate init(
            command: MTLCommandBuffer, workspace: WholeExecutionWorkspace, weights: WholeLayerWeights,
            rope: PinnedRoPE, tables: [PinnedRoPE.EncodedTables], mirror: QSADecodeStorage?,
            append: QSADecodeStorage.EncodedAppend?, pagedRead: QSAPagedAttentionStorage.Read?, offset: Int, rows: Int
        ) {
            self.command = command
            self.workspace = workspace
            self.weights = weights
            self.rope = rope
            self.tables = tables
            self.mirror = mirror
            self.append = append
            self.pagedRead = pagedRead
            self.offset = offset
            self.rows = rows
        }
        /// Inspect only after the enclosing command joins. The caller must also
        /// validate earlier layers' flags before accepting any dependent work.
        func requiresReplay() throws -> Bool {
            guard !finished, command.status == .completed || routerSignaled else {
                throw EngineError("QSA transaction has not joined or already retired")
            }
            return [WholeExecutionWorkspace.Slot.qsaPoolFlags, .qsaSelectionFlags, .qsaGateFlags].contains {
                workspace[$0].contents().load(as: UInt32.self) != 0
            }
        }
        func publish(to state: inout QSAState) throws {
            guard try !requiresReplay(), state.offset == offset else {
                throw EngineError("QSA publication needs replay or frontier differs")
            }
            func bits(_ role: WholeExecutionWorkspace.Slot, _ count: Int) -> [UInt16] {
                Array(UnsafeBufferPointer(start: workspace[role].contents().assumingMemoryBound(to: UInt16.self), count: count))
            }
            // All validation in append precedes mutation. No provisional CPU
            // state copy is retained across this call, avoiding needless COW.
            try state.append(
                keys: bits(.qsaKeyBF16, rows * 512), values: bits(.qsaValueBF16, rows * 512),
                pooledKeys: bits(.qsaPoolBF16, ((offset + rows) / 4 - offset / 4) * 128),
                rawTail: bits(.qsaTail, (offset + rows) % 4 * 128), tokens: rows)
            if let append { try mirror!.finishEncoded(append, success: true) }
            if let pagedRead { try pagedRead.owner.finish(pagedRead, success: true) }
            finished = true
            for table in tables { try rope.publishTables(table) }
        }
        func discard() throws {
            guard !finished else { return }
            if let append { try mirror!.finishEncoded(append, success: false) }
            if let pagedRead { try pagedRead.owner.finish(pagedRead, success: false) }
            finished = true
        }
    }

    /// Keep this copy outside the rethrowing Void closure ABI: the installed
    /// optimizer otherwise tail-calls memmove with a live buffer address in
    /// Swift's error register. The explicit Bool result prevents that return.
    @inline(never)
    package static func stageOldTail(_ values: [UInt16], into buffer: MTLBuffer) -> Bool {
        guard values.count <= 384, values.count % 128 == 0,
            buffer.storageMode == .shared, values.count * 2 <= buffer.length
        else { return false }
        return values.withUnsafeBytes { source -> Bool in
            if !source.isEmpty { buffer.contents().copyMemory(from: source.baseAddress!, byteCount: source.count) }
            return true
        }
    }

    func encode(
        _ initialCommand: MTLCommandBuffer, inputFP32: MTLBuffer, inputBF16: MTLBuffer, weights owner: WholeLayerWeights,
        state: QSAState, mirror: QSADecodeStorage?, paged: QSAPagedAttentionStorage? = nil,
        workspace s: WholeExecutionWorkspace, joinSelection: ((MTLCommandBuffer) throws -> MTLCommandBuffer)? = nil
    ) throws -> Pending {
        var command = initialCommand
        try state.validate()
        let rows = s.layout.rows
        let keys = state.offset + rows
        let blocks = keys / 4
        let newBlocks = blocks - state.offset / 4
        guard let w = owner.qsa, owner.layer % 4 == 3, keys <= s.layout.capacity,
            (mirror == nil) != (paged == nil), keys <= (mirror?.capacityTokens ?? paged!.capacityTokens),
            (mirror?.capacityTokens ?? paged!.capacityTokens) <= s.layout.capacity, inputBF16.length >= rows * 2560 * 2,
            command.commandQueue.device.registryID == kernels.device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Whole QSA layer/command/workspace geometry differs")
        }
        let positions = Array(state.offset..<keys)
        let table = try rope.encodeTables(command, positions: positions)
        var tables = [table]
        func project(
            _ p: GDNDeviceBlock.Projection, input: MTLBuffer, output: WholeExecutionWorkspace.Slot,
            partial: WholeExecutionWorkspace.Slot, width: Int, count: Int
        ) throws {
            let plan = try kernels.pinnedProjectionPlan(rows: rows, width: width, outputs: count, simdMatrix: simdMatrix)
            try kernels.encodePinnedProjection(
                command, input: input, words: p.words, scales: p.scales, biases: p.biases,
                output: s[output], partials: s.optional(partial), plan: plan, simdMatrix: simdMatrix)
        }
        try project(w.queryGate, input: inputBF16, output: .qsaQG, partial: .qsaQPartial, width: 2560, count: 12288)
        try project(w.key, input: inputBF16, output: .qsaKey, partial: .qsaKPartial, width: 2560, count: 512)
        try project(w.value, input: inputBF16, output: .qsaValue, partial: .qsaVPartial, width: 2560, count: 512)
        try dense.encodeRows(
            command, input: inputFP32, weights: w.index, output: s[.qsaIndexQK], partials: s.optional(.qsaIndexPartial),
            rows: rows, width: 2560, outputs: 640, outputBF16: true)
        try scalar.encodeSplit(command, qg: s[.qsaQG], queries: s[.qsaQuery], gates: s[.qsaGates], rows: rows)
        try kernels.encodeQSARMS(
            command, input: s[.qsaQuery], weights: w.queryNorm, output: s[.qsaQueryNorm], groups: rows * 24, width: 256)
        try kernels.encodeQSARMS(command, input: s[.qsaKey], weights: w.keyNorm, output: s[.qsaKeyNorm], groups: rows * 2, width: 256)
        try rope.encodeRotation(command, input: s[.qsaQueryNorm], output: s[.qsaQueryRotated], width: 256, heads: 24, tables: table)
        try rope.encodeRotation(command, input: s[.qsaKeyNorm], output: s[.qsaKeyRotated], width: 256, heads: 2, tables: table)
        try scalar.encodeNarrow(command, input: s[.qsaQueryRotated], output: s[.qsaQueryBF16], count: rows * 6144)
        try scalar.encodeNarrow(command, input: s[.qsaKeyRotated], output: s[.qsaKeyBF16], count: rows * 512)
        try scalar.encodeNarrow(command, input: s[.qsaValue], output: s[.qsaValueBF16], count: rows * 512)
        guard Self.stageOldTail(state.rawTail, into: s[.qsaOldTail]) else { throw EngineError("QSA old tail exceeds its admitted span") }
        try scalar.encodeIndex(
            command, qk: s[.qsaIndexQK], oldTail: s[.qsaOldTail], queries: s[.qsaIndexQuery],
            raw: s[.qsaRaw], rows: rows, tailRows: state.offset % 4)
        try scalar.encodePool(
            command, raw: s[.qsaRaw], pooled: s[.qsaPooled], tail: s[.qsaTail],
            exceptions: s[.qsaPoolFlags], rows: rows, tailRows: state.offset % 4)
        if newBlocks > 0 {
            try kernels.encodeQSARMS(
                command, input: s[.qsaPooled], weights: w.indexKeyNorm, output: s[.qsaPoolNorm], groups: newBlocks, width: 128)
            let poolTable = try rope.encodeTables(command, positions: (0..<newBlocks).map { (state.offset / 4 + $0) * 4 })
            tables.append(poolTable)
            try rope.encodeRotation(command, input: s[.qsaPoolNorm], output: s[.qsaPoolRotated], width: 128, heads: 1, tables: poolTable)
            try scalar.encodeNarrow(command, input: s[.qsaPoolRotated], output: s[.qsaPoolBF16], count: newBlocks * 128)
        }
        if keys > 2048 {
            // The indexer reads the existing immutable pooled pages in place; this step's new blocks stay in FP32.
            guard let pooledTable = pooledTables[owner.layer] else { throw EngineError("QSA layer has no pooled page table") }
            let pages = try state.pooledPages(device: kernels.device, host: indexer.copyPooled)
            let paged = PinnedIndexer.PagedKeys(
                table: pooledTable.table, resources: try pooledTable.bind(pages, command: command),
                fresh: s[.qsaPoolRotated], oldBlocks: state.offset / 4)
            try kernels.encodeQSARMS(
                command, input: s[.qsaIndexQuery], weights: w.indexQueryNorm, output: s[.qsaIndexNorm], groups: rows * 4, width: 128)
            try rope.encodeRotation(command, input: s[.qsaIndexNorm], output: s[.qsaIndexRotated], width: 128, heads: 4, tables: table)
            let stripe = try PinnedIndexer.tileRows(rows: rows, blocks: blocks)
            for first in stride(from: 0, to: rows, by: stripe) {
                try indexer.encodeScores(
                    command, query: s[.qsaIndexRotated], queryOffset: first * 512 * 4,
                    paged: paged, dots: s[.qsaDots], output: s[.qsaScores], outputOffset: first * blocks * 4,
                    rows: min(stripe, rows - first), blocks: blocks)
            }
        }
        let selected = try selection.encode(
            command, positions: positions, keys: keys, positionsBuffer: s[.qsaPositions],
            scores: s[.qsaScores], masked: s[.qsaMasked], first: s[.qsaSortFirst], second: s.optional(.qsaSortSecond),
            ids: s[.qsaIDs], counts: s[.qsaCounts], exceptions: s[.qsaSelectionFlags])
        let positionsTicket: PinnedAttention.EncodedPositions
        var fileSelection: [[UInt32]]?
        if state.pages?.arena.fileStore != nil, rows <= 2 {
            guard let joinSelection else { throw EngineError("File-KV decode requires an owned selection dependency boundary") }
            command = try joinSelection(command)
            let rebound = try attention.joinedPositions(selected, command: command)
            positionsTicket = rebound.0
            fileSelection = rebound.1
        } else {
            positionsTicket = try attention.encodedPositions(selected)
        }
        let append = try mirror?.stageEncoded(
            command, state: state,
            newKeys: s[.qsaKeyBF16], newValues: s[.qsaValueBF16], rows: rows)
        let pagedRead = try paged?.stage(
            command, state: state,
            newKeys: s[.qsaKeyBF16], newValues: s[.qsaValueBF16], rows: rows, selected: fileSelection)
        do {
            do {
                if let pagedRead {
                    try attention.encodePagedAttention(
                        command, query: s[.qsaQueryBF16], read: pagedRead,
                        positions: positionsTicket, output: s[.qsaAttended], scratch: s.attentionScratch)
                } else {
                    try attention.encodeAttention(
                        command, query: s[.qsaQueryBF16], key: mirror!.keys, value: mirror!.values,
                        positions: positionsTicket, output: s[.qsaAttended], scratch: s.attentionScratch)
                }
            }
            try scalar.encodeGate(
                command, attended: s[.qsaAttended], gates: s[.qsaGates], output: s[.qsaGated], exceptions: s[.qsaGateFlags], rows: rows)
            try scalar.encodeNarrow(command, input: s[.qsaGated], output: s[.qsaNarrowed], count: rows * 6144)
            try project(w.output, input: s[.qsaNarrowed], output: .qsaOutput, partial: .qsaOutputPartial, width: 6144, count: 2560)
        } catch {
            // encode never submits. A failure discards this entire command;
            // subsequent callers must create another mirror before replay.
            // Both owners are retired and the original failure is reported, not a secondary retirement error.
            if let append { try? mirror!.finishEncoded(append, success: false) }
            if let pagedRead { try? pagedRead.owner.finish(pagedRead, success: false) }
            throw error
        }
        return Pending(
            command: command, workspace: s, weights: owner, rope: rope, tables: tables,
            mirror: mirror, append: append, pagedRead: pagedRead, offset: state.offset, rows: rows)
    }
}
