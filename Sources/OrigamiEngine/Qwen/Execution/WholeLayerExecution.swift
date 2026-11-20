import Foundation
import Metal

/// The fixed whole-schedule arithmetic region. Submission, two-generation
/// scheduling, demand pins and dependent replay belong to its caller. Every
/// return here is provisional; an encoded router never authorizes reads.
package final class WholeLayerExecution {
    private let kernels: ReferenceKernels, dense: PinnedDenseLinear, hc: FusedHCDecode
    private let gdn: GDNDeviceBlock, qsa: QSAExecutionBlock
    private let selector: BatchedTopIndices, reductions: PinnedReductions
    private let scalars: WholeExecutionScalars, injection: FusedResidualInjection
    private let ledger: MemoryLedger
    private let simdMatrix: Bool
    private let deviceConvolutionTail: Bool
    /// The MTP verify step's BF16 copy of a GDN layer's two new rows (reused; queue order serializes its users).
    private var pairTail: MTLBuffer?

    init(
        kernels: ReferenceKernels, dense: PinnedDenseLinear, hc: FusedHCDecode, gdn: GDNDeviceBlock,
        qsa: QSAExecutionBlock, selector: BatchedTopIndices, reductions: PinnedReductions,
        scalars: WholeExecutionScalars, injection: FusedResidualInjection, ledger: MemoryLedger, simdMatrix: Bool,
        deviceConvolutionTail: Bool = false
    ) {
        self.kernels = kernels
        self.dense = dense
        self.hc = hc
        self.gdn = gdn
        self.qsa = qsa
        self.selector = selector
        self.reductions = reductions
        self.scalars = scalars
        self.injection = injection
        self.ledger = ledger
        self.simdMatrix = simdMatrix
        self.deviceConvolutionTail = deviceConvolutionTail
    }
    /// A tile's checked top-10 expert IDs per row (rank order) and their probabilities [rows][10].
    struct Routing {
        let ids: [[Int]], probabilities: [Float]
    }
    /// One encoded attention/router transaction: its command, weights, workspace and unpublished GDN or QSA state.
    /// After the join it is either published (`routing`, `publish`) or discarded; until then nothing it computed is visible.
    final class Pending {
        let command: MTLCommandBuffer, weights: WholeLayerWeights, workspace: WholeExecutionWorkspace
        let oldGDN: GDNState?, nextGDN: GDNRecurrentStorage?, qsa: QSAExecutionBlock.Pending?
        private var finished = false
        /// Pre-encoded decode: the router signal encoded after this layer's attention and router fired. Flags and routing are final
        /// then; the shared expert and pre-routing queued behind it may still run, and publication waits for them.
        private(set) var routerSignaled = false
        func markRouterSignaled() {
            routerSignaled = true
            qsa?.routerSignaled = true
        }
        fileprivate init(
            command: MTLCommandBuffer, weights: WholeLayerWeights, workspace: WholeExecutionWorkspace,
            oldGDN: GDNState?, nextGDN: GDNRecurrentStorage?, qsa: QSAExecutionBlock.Pending?
        ) {
            self.command = command
            self.weights = weights
            self.workspace = workspace
            self.oldGDN = oldGDN
            self.nextGDN = nextGDN
            self.qsa = qsa
        }
        func requiresReplay() throws -> Bool {
            guard !finished, command.status == .completed || routerSignaled else {
                throw EngineError("Whole attention/router transaction has not joined or retired")
            }
            let s = workspace
            for slot in [
                WholeExecutionWorkspace.Slot.hcAFlags, .hcMFlags, .attentionInjectionFlags, .routerFlags,
                .gdnPrepareFlags, .gdnGateFlags,
            ] {
                let p = s[slot].contents().assumingMemoryBound(to: UInt32.self)
                let count = (slot == .hcAFlags || slot == .hcMFlags) ? s.layout.rows * 80 : 1
                if (0..<count).contains(where: { p[$0] != 0 }) { return true }
            }
            return try qsa?.requiresReplay() ?? false
        }
        /// Only checked, joined IDs can be returned to a demand owner. Calling
        /// this after speculative encoding but before completion is rejected.
        func routing() throws -> Routing {
            guard try !requiresReplay() else { throw EngineError("Whole routing needs original-path replay") }
            let rows = workspace.layout.rows
            let pairs = workspace[.routerPairs].contents().assumingMemoryBound(to: UInt32.self)
            let probability = workspace[.routerWeights].contents().assumingMemoryBound(to: Float.self)
            var ids: [[Int]] = []
            var probabilities: [Float] = []
            ids.reserveCapacity(rows)
            probabilities.reserveCapacity(rows * 10)
            for row in 0..<rows {
                var selected: [Int] = []
                selected.reserveCapacity(10)
                var sum: Float = 0
                for rank in 0..<10 {
                    let id = Int(pairs[(row * 10 + rank) * 2 + 1])
                    let weight = probability[row * 10 + rank]
                    guard (0..<ModelLayout.routedExperts).contains(id), !selected.contains(id), weight.isFinite, weight >= 0 else {
                        throw EngineError("Whole routing IDs/probabilities are invalid")
                    }
                    selected.append(id)
                    probabilities.append(weight)
                    sum += weight
                }
                guard sum.isFinite, sum > 0 else { throw EngineError("Whole router has no finite positive probability mass") }
                ids.append(selected)
            }
            return Routing(ids: ids, probabilities: probabilities)
        }
        /// Caller first validates preceding MoE/dependent flags and cancellation.
        /// This publishes only this attention state, never the residual stream.
        func publish(state: SequenceState) throws {
            guard try !requiresReplay() else { throw EngineError("Whole state publication requires replay") }
            let s = workspace
            let rows = s.layout.rows
            let layer = weights.layer
            if let oldGDN {
                guard state.gdn[layer]?.offset == oldGDN.offset else { throw EngineError("Whole GDN publication frontier differs") }
                var next = oldGDN
                if let nextGDN, nextGDN.convolutionByteOffset != nil {
                    next.publishConvolution(nextGDN)
                } else {
                    let retained = min(3, rows) * 10240
                    let mixed = s[.gdnMixed].contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 10240 - retained)
                    next.convolutionTail =
                        oldGDN.convolutionTail.suffix(3 * 10240 - retained).map { BF16.encode(BF16.decode($0)) }
                        + (0..<retained).map { BF16.encode(mixed[$0]) }
                }
                if let nextGDN {
                    next.publishRecurrent(nextGDN)
                } else {
                    let p = s[.gdnNext].contents().assumingMemoryBound(to: Float.self).advanced(by: rows * 6144)
                    next.recurrent = Array(UnsafeBufferPointer(start: p, count: 48 * 128 * 128))
                }
                next.offset += rows
                try next.validate()
                state.gdn[layer] = next
            } else {
                guard let qsa, var value = state.qsa.removeValue(forKey: layer) else {
                    throw EngineError("Missing whole QSA publication state")
                }
                do {
                    try qsa.publish(to: &value)
                    state.qsa[layer] = value
                } catch {
                    state.qsa[layer] = value
                    throw error
                }
            }
            finished = true
        }
        func discard() throws {
            guard !finished else { return }
            guard command.status == .notEnqueued || command.status == .completed || command.status == .error else {
                throw EngineError("Cannot discard a whole transaction with an unjoined writer")
            }
            // GDN uses distinct unpublished storage. QSA additionally has a
            // request-owned mutable suffix whose mirror must be invalidated.
            try qsa?.discard()
            finished = true
        }
    }

    /// `hc_exact_fusion`: tile HCs take their FP32 input and narrow it inside the fused entry.
    private func fusedHCEntry(_ s: WholeExecutionWorkspace) -> Bool { s.layout.rows > 1 && hc.fusedEntryEnabled }
    /// `hc_decode_fusion`: row-one HCs take their FP32 input (narrowed inside the fused entry) and write the BF16
    /// narrowing of their mixed output themselves; the caller encodes neither narrowing.
    private func fusedHCRow(_ s: WholeExecutionWorkspace) -> Bool { s.layout.rows == 1 && hc.decodeFusion }
    /// `inputFP32` (fused entry only): the FP32 rows whose BF16 narrowing the entry writes to `input`.
    /// `mixedBF16` (fused row only): the destination of the mixed output's BF16 narrowing.
    private func encodeHC(
        _ command: MTLCommandBuffer, weights w: DecodeFrontierWeights.HC,
        input: MTLBuffer, inputFP32: MTLBuffer? = nil, mixedBF16: MTLBuffer? = nil,
        workspace s: WholeExecutionWorkspace, attention: Bool
    ) throws {
        let buffers = attention ? s.attentionHC : s.moeHC
        if s.layout.rows == 1 {
            guard (inputFP32 == nil && mixedBF16 == nil) || fusedHCRow(s) else {
                throw EngineError("The fused HC row entry requires hc_decode_fusion")
            }
            try hc.encodeRow(
                command, residualBF16: input, normWeights: w.norm, downWeights: w.down, upWeights: w.up,
                injectionWeights: w.injection, buffers: buffers, residualFP32: inputFP32, mixedBF16: mixedBF16)
        } else {
            try hc.encodeTile(
                command, rows: s.layout.rows, residualBF16: input, normWeights: w.norm,
                downWeights: w.down, upWeights: w.up, injectionWeights: w.injection, buffers: buffers,
                downPartials: s.optional(attention ? .hcADownPartial : .hcMDownPartial),
                upPartials: s.optional(attention ? .hcAUpPartial : .hcMUpPartial),
                injectionPartials: s.optional(attention ? .hcAInjectionPartial : .hcMInjectionPartial), residualFP32: inputFP32)
        }
    }
    /// One row of a layer's hyper-connection: the real layer applies its MoE HC
    /// to its own post-attention row; decode pre-routing applies the next
    /// layer's to the current one, through this same encoder.
    func encodeHCRow(
        _ command: MTLCommandBuffer, residualBF16: MTLBuffer, weights w: DecodeFrontierWeights.HC,
        buffers: FusedHCDecode.RowBuffers
    ) throws {
        try Self.encodeHCRow(hc, command, residualBF16: residualBF16, weights: w, buffers: buffers)
    }
    static func encodeHCRow(
        _ hc: FusedHCDecode, _ command: MTLCommandBuffer, residualBF16: MTLBuffer,
        weights w: DecodeFrontierWeights.HC, buffers: FusedHCDecode.RowBuffers
    ) throws {
        try hc.encodeRow(
            command, residualBF16: residualBF16, normWeights: w.norm,
            downWeights: w.down, upWeights: w.up, injectionWeights: w.injection, buffers: buffers)
    }
    /// The router over the mixed HC output and its top-ten selection; shared by
    /// the real layer path and decode pre-routing.
    func encodeRouterSelection(
        _ command: MTLCommandBuffer, mixed: MTLBuffer, router: MTLBuffer, logits: MTLBuffer,
        partial: MTLBuffer?, pairs: MTLBuffer, rows: Int
    ) throws {
        try Self.encodeRouterSelection(
            dense, selector, command, mixed: mixed, router: router, logits: logits, partial: partial, pairs: pairs, rows: rows)
    }
    package static func encodeRouterSelection(
        _ dense: PinnedDenseLinear, _ selector: BatchedTopIndices, _ command: MTLCommandBuffer,
        mixed: MTLBuffer, router: MTLBuffer, logits: MTLBuffer,
        partial: MTLBuffer?, pairs: MTLBuffer, rows: Int
    ) throws {
        try dense.encodeRows(
            command, input: mixed, weights: router, output: logits, partials: partial,
            rows: rows, width: 2560, outputs: 512, outputBF16: false)
        if ModelLayout.routedExperts < 512 {
            try selector.encodeMaskColumns(command, scores: logits, rows: rows, columns: 512, eligible: ModelLayout.routedExperts)
        }
        _ = try selector.encodeTopIndices(command, scores: logits, first: pairs, second: nil, rows: rows, columns: 512, count: 10)
    }
    /// Double-buffered prefill (`prefill_double_buffer`): the previous tile's provisional recurrence
    /// and its last three projected rows are this tile's initial state and convolution history, read
    /// on the GPU behind that tile's still-running command (same queue, commit order) instead of
    /// copied from host state after its join. `state` is that tile's old GDN state advanced by its
    /// rows; publication of this tile still validates it against the published frontier.
    struct GDNChain {
        let previous: WholeExecutionWorkspace, state: GDNState
    }
    /// residual may be the previous generation's final result. Preserve an
    /// independent old copy for replay before any layer operation reads it.
    func encodeAttentionRouter(
        _ command: inout MTLCommandBuffer, residual: MTLBuffer, weights w: WholeLayerWeights,
        state: SequenceState, mirror: QSADecodeStorage?, paged: QSAPagedAttentionStorage? = nil,
        workspace s: WholeExecutionWorkspace,
        joinSelection: ((MTLCommandBuffer) throws -> MTLCommandBuffer)? = nil,
        gdnChain: GDNChain? = nil,
        afterAttention: ((inout MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> Pending {
        let rows = s.layout.rows
        guard (0..<48).contains(w.layer), command.status == .notEnqueued,
            command.commandQueue.device.registryID == kernels.device.registryID,
            residual.length >= rows * 10240 * 4, residual.device.registryID == kernels.device.registryID
        else {
            throw EngineError("Whole attention/router input geometry differs")
        }
        guard
            gdnChain == nil
                || (w.gdn != nil && w.qsa == nil && rows >= 3 && gdnChain!.previous !== s
                    && gdnChain!.previous.layout.rows == rows && gdnChain!.previous.layout.capacity == s.layout.capacity)
        else {
            throw EngineError("Whole GDN chain requires two distinct equal C256 tiles of one GDN layer")
        }
        guard !s.layout.gdnOnly || (w.gdn != nil && w.qsa == nil) else {
            throw EngineError("A GDN-only prefill workspace cannot encode a QSA layer")
        }
        if residual !== s[.residual] {
            guard let e = command.makeBlitCommandEncoder() else { throw EngineError("Whole residual snapshot encoder unavailable") }
            e.copy(from: residual, sourceOffset: 0, to: s[.residual], destinationOffset: 0, size: rows * 10240 * 4)
            e.endEncoding()
        }
        if fusedHCRow(s) {
            try encodeHC(
                command, weights: w.attentionHC, input: s[.residualBF16], inputFP32: s[.residual],
                mixedBF16: s[.mixedAttentionBF16], workspace: s, attention: true)
        } else {
            if fusedHCEntry(s) {
                try encodeHC(
                    command, weights: w.attentionHC, input: s[.residualBF16], inputFP32: s[.residual], workspace: s, attention: true)
            } else {
                try scalars.encodeNarrow(command, input: s[.residual], output: s[.residualBF16], count: rows * 10240)
                try encodeHC(command, weights: w.attentionHC, input: s[.residualBF16], workspace: s, attention: true)
            }
            try scalars.encodeNarrow(command, input: s[.hcAMixed], output: s[.mixedAttentionBF16], count: rows * 2560)
        }
        var oldGDN: GDNState?
        var nextGDN: GDNRecurrentStorage?
        var pendingQSA: QSAExecutionBlock.Pending?
        let attentionOutput: MTLBuffer
        if let gw = w.gdn {
            guard let old = gdnChain?.state ?? state.gdn[w.layer], w.qsa == nil, old.offset + rows <= min(state.capacity, s.layout.capacity)
            else { throw EngineError("Whole GDN old state differs") }
            try old.validate()
            oldGDN = old
            if let chain = gdnChain {
                // The chained history is exactly the host publication's tail: BF16.encode of the
                // previous tile's last three projected rows (hc_decode_key is bit-identical).
                try scalars.encodeNarrow(
                    command, input: chain.previous[.gdnMixed], inputOffset: (rows - 3) * 10240 * 4,
                    output: s[.gdnHistory], count: 3 * 10240)
            } else if deviceConvolutionTail, rows == 1, let tail = old.convolutionStorage {
                guard let encoder = command.makeBlitCommandEncoder(), let offset = tail.convolutionByteOffset else {
                    throw EngineError("Whole convolution history copy unavailable")
                }
                encoder.copy(from: tail.buffer, sourceOffset: offset, to: s[.gdnHistory], destinationOffset: 0, size: 61440)
                encoder.endEncoding()
            } else {
                // Not a Void copy closure through `withConvolutionBytes`: built with Swift 6.4 that returned here with
                // the buffer's address in the error register (see `copyRecurrentBytes`; Tests/error-register-check.py).
                guard
                    old.copyConvolutionBytes(
                        to: UnsafeMutableRawBufferPointer(start: s[.gdnHistory].contents(), count: 3 * 10240 * 2))
                else {
                    throw EngineError("Whole convolution history copy extent differs")
                }
            }
            s[.gdnMask].contents().assumingMemoryBound(to: UInt32.self).initialize(repeating: 1, count: rows)
            let initial: MTLBuffer
            let offset: Int
            if let chain = gdnChain {
                initial = chain.previous[.gdnNext]
                offset = rows * 6144 * 4
            } else if let retained = old.recurrentStorage {
                initial = retained.buffer
                offset = retained.byteOffset
            } else {
                guard
                    old.copyRecurrentBytes(
                        to: UnsafeMutableRawBufferPointer(start: s[.gdnColdInitial].contents(), count: 48 * 128 * 128 * 4))
                else {
                    throw EngineError("Whole GDN initial state copy extent differs")
                }
                initial = s[.gdnColdInitial]
                offset = 0
            }
            if rows == 1 {
                let next = try GDNRecurrentStorage(
                    device: kernels.device, outputCount: 6144, ledger: ledger, retainConvolutionTail: deviceConvolutionTail)
                nextGDN = next
                try gdn.encodeRow(
                    command, inputFP32: s[.hcAMixed], inputBF16: s[.mixedAttentionBF16], qkv: gw.qkv, z: gw.z,
                    b: gw.b, a: gw.a, out: gw.output, initial: initial, initialOffset: offset, next: next, buffers: s.gdn(weights: gw))
                if deviceConvolutionTail {
                    try scalars.encodeConvolutionTail(command, history: s[.gdnHistory], mixed: s[.gdnMixed], next: next)
                }
            } else {
                try gdn.encodeTile(
                    command, rows: rows, inputFP32: s[.hcAMixed], inputBF16: s[.mixedAttentionBF16], qkv: gw.qkv, z: gw.z,
                    b: gw.b, a: gw.a, out: gw.output, initial: initial, initialOffset: offset, recurrence: s[.gdnNext],
                    buffers: s.gdn(weights: gw), partials: s.gdnPartials)
                if rows == 2 {
                    // The MTP verify step keeps the recurrence and the convolution tail (the history's last row, then both
                    // new rows in BF16) on the GPU, as a decode row does, so publishing reads nothing from the workspace.
                    let next = try GDNRecurrentStorage(
                        device: kernels.device, outputCount: 6144, ledger: ledger, retainConvolutionTail: true)
                    nextGDN = next
                    if pairTail == nil { pairTail = kernels.device.makeBuffer(length: 2 * 10240 * 2, options: .storageModeShared) }
                    guard let tail = pairTail, let tailOffset = next.convolutionByteOffset else {
                        throw EngineError("Pair GDN tail unavailable")
                    }
                    try scalars.encodeNarrow(command, input: s[.gdnMixed], output: tail, count: 2 * 10240)
                    guard let blit = command.makeBlitCommandEncoder() else { throw EngineError("Pair GDN state copy unavailable") }
                    blit.copy(
                        from: s[.gdnNext], sourceOffset: rows * 6144 * 4, to: next.buffer, destinationOffset: next.byteOffset,
                        size: next.count * 4)
                    blit.copy(
                        from: s[.gdnHistory], sourceOffset: 2 * 10240 * 2, to: next.buffer, destinationOffset: tailOffset, size: 10240 * 2)
                    blit.copy(from: tail, sourceOffset: 0, to: next.buffer, destinationOffset: tailOffset + 10240 * 2, size: 2 * 10240 * 2)
                    blit.endEncoding()
                }
            }
            attentionOutput = s[.gdnOutput]
        } else {
            guard let old = state.qsa[w.layer], mirror != nil || paged != nil else {
                throw EngineError("Whole QSA state/storage unavailable")
            }
            pendingQSA = try qsa.encode(
                command, inputFP32: s[.hcAMixed], inputBF16: s[.mixedAttentionBF16], weights: w,
                state: old, mirror: mirror, paged: paged, workspace: s, joinSelection: joinSelection)
            command = pendingQSA!.command
            attentionOutput = s[.qsaOutput]
        }
        do {
            try injection.encodeTile(
                command, rows: rows, output: attentionOutput, residual: s[.residual], gates: s[.hcAInjection],
                result: s[.afterAttention], exceptions: s[.attentionInjectionFlags])
            try afterAttention?(&command)
            if fusedHCRow(s) {
                try encodeHC(
                    command, weights: w.moeHC, input: s[.afterAttentionBF16], inputFP32: s[.afterAttention],
                    mixedBF16: s[.mixedMoEBF16], workspace: s, attention: false)
            } else {
                if fusedHCEntry(s) {
                    try encodeHC(
                        command, weights: w.moeHC, input: s[.afterAttentionBF16], inputFP32: s[.afterAttention], workspace: s,
                        attention: false)
                } else {
                    try scalars.encodeNarrow(command, input: s[.afterAttention], output: s[.afterAttentionBF16], count: rows * 10240)
                    try encodeHC(command, weights: w.moeHC, input: s[.afterAttentionBF16], workspace: s, attention: false)
                }
                try scalars.encodeNarrow(command, input: s[.hcMMixed], output: s[.mixedMoEBF16], count: rows * 2560)
            }
            try encodeRouterSelection(
                command, mixed: s[.hcMMixed], router: w.router, logits: s[.routerLogits],
                partial: s.optional(.routerPartial), pairs: s[.routerPairs], rows: rows)
            try scalars.encodeRouterGather(
                command, logits: s[.routerLogits], pairs: s[.routerPairs], selected: s[.routerSelected], exceptions: s[.routerFlags],
                rows: rows)
            try reductions.encodeRouterRows(command, input: s[.routerSelected], output: s[.routerWeights], rows: rows)
        } catch {
            try pendingQSA?.discard()
            throw error
        }
        return Pending(command: command, weights: w, workspace: s, oldGDN: oldGDN, nextGDN: nextGDN, qsa: pendingQSA)
    }

    func encodeShared(_ command: MTLCommandBuffer, weights w: WholeLayerWeights, workspace s: WholeExecutionWorkspace) throws {
        let rows = s.layout.rows
        func project(
            _ p: GDNDeviceBlock.Projection, input: MTLBuffer, output: WholeExecutionWorkspace.Slot,
            partial: WholeExecutionWorkspace.Slot, width: Int, count: Int
        ) throws {
            let plan = try kernels.pinnedProjectionPlan(rows: rows, width: width, outputs: count, simdMatrix: simdMatrix)
            try kernels.encodePinnedProjection(
                command, input: input, words: p.words, scales: p.scales, biases: p.biases,
                output: s[output], partials: s.optional(partial), plan: plan, simdMatrix: simdMatrix)
        }
        try project(w.sharedGate, input: s[.mixedMoEBF16], output: .sharedGate, partial: .sharedGatePartial, width: 2560, count: 640)
        try project(w.sharedUp, input: s[.mixedMoEBF16], output: .sharedUp, partial: .sharedUpPartial, width: 2560, count: 640)
        try scalars.encodeProduct(
            command, gate: s[.sharedGate], up: s[.sharedUp], output: s[.sharedProduct], exceptions: s[.sharedFlags], rows: rows)
        try project(w.sharedDown, input: s[.sharedProduct], output: .sharedOutput, partial: .sharedDownPartial, width: 640, count: 2560)
        try dense.encodeSharedGateRows(command, input: s[.hcMMixed], weights: w.sharedOutputGate, output: s[.sharedRawGate], rows: rows)
        try scalars.encodeSigmoid(command, input: s[.sharedRawGate], output: s[.sharedSigmoid], rows: rows)
    }

    func encodeMix(_ command: MTLCommandBuffer, workspace s: WholeExecutionWorkspace) throws {
        try scalars.encodeCombine(
            command, routed: s[.routed], shared: s[.sharedOutput], gates: s[.sharedSigmoid],
            output: s[.combined], exceptions: s[.combinationFlags], rows: s.layout.rows)
        try injection.encodeTile(
            command, rows: s.layout.rows, output: s[.combined], residual: s[.afterAttention],
            gates: s[.hcMInjection], result: s[.result], exceptions: s[.finalInjectionFlags])
    }
}
