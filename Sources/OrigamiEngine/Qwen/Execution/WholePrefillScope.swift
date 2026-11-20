import Foundation
import Metal
import ObjectiveC

/// The host part of a scope's allowance, associated with every scope buffer so it retires with the last one.
private final class WholeScopeHostAllocation {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "read_scope_frontier_bound")
    }
    deinit { try? ledger.release(reservation) }
}
private enum WholeScopeAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The device frontiers of one whole-prefill scope of R <= 8192 tokens: the residual stream [R][4][2560] FP32 (four
/// hyper-connection branches per token), the MoE input [R][2560], injection gates [R][4], routing weights [R][10],
/// shared-expert output [R][2560] and gate [R], and the routed output [R][2560]. They share one escrow carved from
/// the per-scope allowance max(0, R - 256) * 240,000 bytes that the per-layer path also uses (the fixed 2 GB phase
/// reserve is untouched); its owner lives as long as any buffer, including an adopted MoE input. `layout` is nil
/// when the scope does not fit that allowance (below 1,515 tokens with a ragged tail, 1,536 in C256 multiples) and
/// the caller then takes the per-layer path. Execution may explicitly admit a small scope's shortfall separately;
/// scheduling never uses that extra allowance to merge tiles or change checkpoint arithmetic. Copies only.
package final class WholePrefillScope {
    /// Scope buffers in allocation order; `Layout.sizes` holds each one's bytes.
    package enum Slot: Int, CaseIterable { case residual, input, injection, routingWeights, shared, sharedGate, routed }
    /// An admitted scope: buffer sizes, the original allowance plus any small-scope supplement, and its device/host split.
    package struct Layout {
        package let tokens: Int, sizes: [Int]
        package let originalExtra, supplementalBytes, deviceBytes, remainingHostBytes: UInt64
    }
    package let layout: Layout, budget: PrefillMoEBudget
    private let buffers: [MTLBuffer]

    package static func layout(tokens: Int, allowTail: Bool = false, admitSmallScope: Bool = false) throws -> Layout? {
        guard (1...8192).contains(tokens), tokens % 256 == 0 || allowTail else {
            throw EngineError("Whole prefill requires declared C256 or mixed-tail scope geometry")
        }
        let sizes = [
            tokens * 10240 * 4, tokens * 2560 * 4, tokens * 4 * 4, tokens * 10 * 4,
            tokens * 2560 * 4, tokens * 4, tokens * 2560 * 4,
        ]
        // Keep the historical scheduling allowance: callers that only plan checkpoint segments must not
        // start merging previously separate small scopes because execution can now admit their buffers.
        let old = UInt64(max(0, tokens - 256)) * 240_000
        // bytesForScope already includes input, canonical ten-expert output,
        // batch maps, coverage and producer metadata. Count input exactly once.
        let additional = sizes.enumerated().filter { $0.offset != Slot.input.rawValue }
            .reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1.element) }
        let mixedMetadata = tokens % 256 == 0 ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes
        let device = try PrefillMoEBudget.bytesForScope(tokens: tokens) + additional + 131072 + mixedMetadata
        // Remaining host frontiers: embeddings + serial route/index maps and
        // their allocator slack. At 32 bytes/route (two Ints plus 2x capacity),
        // routing needs 320 B/token. Keep a conservative additional 4 MiB metadata
        // bound. The unchanged phase reserve covers at most two C256 original
        // replay tiles/PLE work and captures, never full scope duplications.
        let host =
            PrefillMoEBudget.pageRounded(tokens * 2560 * 4 * 2)
            + PrefillMoEBudget.pageRounded(tokens * 320) + 4 * 1024 * 1024
        let required = device + host
        let supplement = required > old ? required - old : 0
        guard supplement == 0 || (admitSmallScope && tokens <= 1536) else { return nil }
        return Layout(
            tokens: tokens, sizes: sizes, originalExtra: old, supplementalBytes: supplement,
            deviceBytes: device, remainingHostBytes: old + supplement - device)
    }
    package init(device: MTLDevice, ledger: MemoryLedger, layout: Layout) throws {
        let host = try WholeScopeHostAllocation(ledger: ledger, bytes: layout.remainingHostBytes)
        let budget = try PrefillMoEBudget(ledger: ledger, bytes: layout.deviceBytes)
        let buffers = try layout.sizes.map { try budget.buffer(bytes: $0, device: device) }
        for buffer in buffers {
            objc_setAssociatedObject(buffer, &WholeScopeAssociation.key, host, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        }
        self.layout = layout
        self.budget = budget
        self.buffers = buffers
    }
    // The last layer's contribution buffer holds the escrow through its permit; it goes with the scope.
    deinit { budget.dropSpare() }
    package subscript(_ role: Slot) -> MTLBuffer { buffers[role.rawValue] }

    /// Input Float rows are copied directly into the four residual branches;
    /// no full-scope host residual Array or transient duplicated frontier.
    package func stageEmbeddings(_ values: [Float]) throws {
        guard values.count == layout.tokens * 2560 else { throw EngineError("Whole scope embedding shape differs") }
        values.withUnsafeBytes { source in
            for row in 0..<layout.tokens {
                for branch in 0..<4 {
                    self[.residual].contents().advanced(by: (row * 10240 + branch * 2560) * 4)
                        .copyMemory(from: source.baseAddress!.advanced(by: row * 2560 * 4), byteCount: 2560 * 4)
                }
            }
        }
    }
    package func encodeTileInput(_ command: MTLCommandBuffer, start: Int, workspace: WholeExecutionWorkspace) throws {
        guard (1...256).contains(workspace.layout.rows), start >= 0, start % workspace.layout.rows == 0,
            start <= layout.tokens - workspace.layout.rows,
            command.status == .notEnqueued, let e = command.makeBlitCommandEncoder()
        else { throw EngineError("Whole scope tile extent/command differs") }
        e.copy(
            from: self[.residual], sourceOffset: start * 10240 * 4, to: workspace[.residual], destinationOffset: 0,
            size: workspace.layout.rows * 10240 * 4)
        e.endEncoding()
    }
    /// Provisional outputs remain private to this scope. The original input
    /// stays in workspace.residual until all flags/state validate. No following
    /// layer or demand owner may observe these slices before that point.
    package func encodeTileOutputs(_ command: MTLCommandBuffer, start: Int, workspace s: WholeExecutionWorkspace) throws {
        guard (1...256).contains(s.layout.rows), start >= 0, start % s.layout.rows == 0, start <= layout.tokens - s.layout.rows,
            command.status == .notEnqueued, let e = command.makeBlitCommandEncoder()
        else { throw EngineError("Whole scope output extent/command differs") }
        for (from, to, width) in [
            (WholeExecutionWorkspace.Slot.afterAttention, Slot.residual, 10240),
            (.hcMMixed, .input, 2560), (.hcMInjection, .injection, 4), (.routerWeights, .routingWeights, 10),
            (.sharedOutput, .shared, 2560), (.sharedSigmoid, .sharedGate, 1),
        ] {
            e.copy(from: s[from], sourceOffset: 0, to: self[to], destinationOffset: start * width * 4, size: s.layout.rows * width * 4)
        }
        e.endEncoding()
    }
    package func encodeMixInput(_ command: MTLCommandBuffer, start: Int, workspace s: WholeExecutionWorkspace) throws {
        guard (1...256).contains(s.layout.rows), start >= 0, start % s.layout.rows == 0, start <= layout.tokens - s.layout.rows,
            command.status == .notEnqueued, let e = command.makeBlitCommandEncoder()
        else { throw EngineError("Whole scope mix extent/command differs") }
        for (from, to, width) in [
            (Slot.residual, WholeExecutionWorkspace.Slot.afterAttention, 10240),
            (.injection, .hcMInjection, 4), (.shared, .sharedOutput, 2560), (.sharedGate, .sharedSigmoid, 1), (.routed, .routed, 2560),
        ] {
            e.copy(from: self[from], sourceOffset: start * width * 4, to: s[to], destinationOffset: 0, size: s.layout.rows * width * 4)
        }
        e.endEncoding()
    }
    /// Joined validated tile publication. A shared-memory copy keeps the old
    /// scope residual available for repair without adding another GPU join.
    // Keep this raw copy outside Swift's throwing ABI. CLT 6.3.3 generated a
    // tail memmove with its source pointer in x21 (the error result register)
    // for the throwing Void method; the success return appeared to throw a
    // buffer address. A non-inlined Bool result also prevents that tail call.
    @inline(never)
    package func publishMix(start: Int, workspace s: WholeExecutionWorkspace, command: MTLCommandBuffer) -> Bool {
        guard command.status == .completed, (1...256).contains(s.layout.rows), start >= 0, start % s.layout.rows == 0,
            start <= layout.tokens - s.layout.rows
        else {
            return false
        }
        self[.residual].contents().advanced(by: start * 10240 * 4).copyMemory(
            from: s[.result].contents(), byteCount: s.layout.rows * 10240 * 4)
        return true
    }
    /// Joined discard of a double-buffered successor tile: its encodeTileOutputs already copied
    /// provisional after-attention rows over its own layer input, which the re-issue reads again.
    /// The workspace still holds that input (encodeTileInput). Raw copy for the reason above.
    @inline(never)
    package func restoreTileInput(start: Int, workspace s: WholeExecutionWorkspace, command: MTLCommandBuffer) -> Bool {
        guard command.status == .completed, s.layout.rows == 256, start >= 0, start % s.layout.rows == 0,
            start <= layout.tokens - s.layout.rows
        else {
            return false
        }
        self[.residual].contents().advanced(by: start * 10240 * 4).copyMemory(
            from: s[.residual].contents(), byteCount: s.layout.rows * 10240 * 4)
        return true
    }
}
