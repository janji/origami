import Darwin
import Foundation
import Metal
import ObjectiveC

/// The ledger charge of one workspace, associated with each of its buffers so it retires with the last one.
private final class WholeWorkspaceAllocation {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "whole_schedule_tile_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum WholeWorkspaceAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// A fixed generation of tile storage: one shared buffer per live role for an actual 1...256-row prefill tile, 2 rows (MTP verify)
/// or 1 row (decode). Prefill has one actual-row generation (two for C256 with `prefill_double_buffer`, the second GDN-only);
/// overlapping decode has two disjoint row-one generations. Scope frontiers, weight readers, recurrent state, KV
/// mirrors and RoPE tables have their own charges. This never borrows the existing 2 GB phase reserve.
package final class WholeExecutionWorkspace {
    /// Every intermediate of one layer (HC, GDN, QSA, router, shared expert, experts, mix) and its exception flags.
    package enum Slot: Int, CaseIterable {
        case residual, residualBF16, afterAttention, afterAttentionBF16, result, mixedAttentionBF16, mixedMoEBF16
        case hcANorm, hcANormBF16, hcADown, hcAActivation, hcAUp, hcAMixed, hcAFlags, hcARawInjection, hcAInjection, hcACapture
        case hcADownPartial, hcAUpPartial, hcAInjectionPartial
        case hcMNorm, hcMNormBF16, hcMDown, hcMActivation, hcMUp, hcMMixed, hcMFlags, hcMRawInjection, hcMInjection, hcMCapture
        case hcMDownPartial, hcMUpPartial, hcMInjectionPartial
        case gdnMixed, gdnZ, gdnB, gdnA, gdnHistory, gdnPrepared, gdnBeta, gdnDecay, gdnPrepareFlags, gdnMask
        case gdnNormed, gdnGated, gdnGateFlags, gdnNarrowed, gdnOutput, gdnConvolved, gdnColdInitial, gdnNext
        case gdnQPartial, gdnZPartial, gdnBPartial, gdnAPartial, gdnOutputPartial
        case qsaQG, qsaQuery, qsaGates, qsaQueryNorm, qsaQueryRotated, qsaQueryBF16
        case qsaKey, qsaKeyNorm, qsaKeyRotated, qsaKeyBF16, qsaValue, qsaValueBF16
        case qsaIndexQK, qsaOldTail, qsaIndexQuery, qsaRaw, qsaPooled, qsaPoolNorm, qsaPoolRotated, qsaPoolBF16, qsaTail
        case qsaIndexNorm, qsaIndexRotated, qsaDots, qsaScores, qsaMasked, qsaSortFirst, qsaSortSecond
        case qsaPositions, qsaIDs, qsaCounts, qsaPoolFlags, qsaSelectionFlags, qsaAttended, qsaGated, qsaGateFlags, qsaNarrowed, qsaOutput
        case qsaAttentionPartial, qsaAttentionSums, qsaAttentionMaxima, qsaAttentionScores
        case qsaQPartial, qsaKPartial, qsaVPartial, qsaIndexPartial, qsaOutputPartial
        case routerLogits, routerPairs, routerSelected, routerWeights, routerFlags, routerPartial
        case sharedGate, sharedUp, sharedProduct, sharedOutput, sharedRawGate, sharedSigmoid, sharedFlags
        case sharedGatePartial, sharedUpPartial, sharedDownPartial, sharedRawGatePartial
        case expertGate, expertUp, expertProduct, expertDown, expertDescriptors, expertRanks, expertMissingDescriptors, expertMissingRanks,
            routed, combined
        case attentionInjectionFlags, combinationFlags, finalInjectionFlags
    }

    /// Slot sizes for a row count and context capacity; `admittedBytes` adds 128 KiB of bookkeeping.
    package struct Layout {
        package let rows, capacity: Int
        let capture: Bool
        package let sizes: [Int]
        /// Logical roles resolve to physical allocations; aliases never cross workspace generations.
        package let storageIndices, allocationSizes: [Int]
        package let compact: Bool
        package let savedBytes: UInt64
        package let bufferBytes, admittedBytes: UInt64
        /// The second double-buffered prefill workspace: GDN attention, router, shared branch and
        /// canonical mix only. QSA data slots are unallocated; the flag slots are kept so the
        /// ordinary clear/check loops stay uniform.
        package var gdnOnly = false

        package init(
            rows: Int, capacity: Int, capture: Bool, sizes: [Int], storageIndices: [Int]? = nil,
            allocationSizes: [Int]? = nil, compact: Bool = false, savedBytes: UInt64 = 0,
            bufferBytes: UInt64, admittedBytes: UInt64, gdnOnly: Bool = false
        ) {
            self.rows = rows
            self.capacity = capacity
            self.capture = capture
            self.sizes = sizes
            self.storageIndices = storageIndices ?? Array(sizes.indices)
            self.allocationSizes = allocationSizes ?? sizes
            self.compact = compact
            self.savedBytes = savedBytes
            self.bufferBytes = bufferBytes
            self.admittedBytes = admittedBytes
            self.gdnOnly = gdnOnly
        }
    }
    package let layout: Layout
    private let buffers: [MTLBuffer?]

    /// The attention HC finishes all projection consumers before the MoE HC starts in encodeAttentionRouter.
    /// Its mixed value/injection, exception bitmap and capture must remain independently inspectable, so only
    /// internal norm/down/up scratch (including split-K partials) can be overwritten by the second HC.
    /// These are ordinary tracked MTLBuffers on one ordered queue, not untracked heap aliases.
    package static let compactPairs: [(Slot, Slot)] = [
        (.hcANorm, .hcMNorm), (.hcANormBF16, .hcMNormBF16), (.hcADown, .hcMDown),
        (.hcAActivation, .hcMActivation), (.hcAUp, .hcMUp), (.hcADownPartial, .hcMDownPartial),
        (.hcAUpPartial, .hcMUpPartial), (.hcAInjectionPartial, .hcMInjectionPartial),
    ]

    /// Slots a GDN-only prefill workspace never binds (QSA projections, index, selection, attention and partials).
    package static let qsaDataSlots: [Slot] = [
        .qsaQG, .qsaQuery, .qsaGates, .qsaQueryNorm, .qsaQueryRotated, .qsaQueryBF16,
        .qsaKey, .qsaKeyNorm, .qsaKeyRotated, .qsaKeyBF16, .qsaValue, .qsaValueBF16,
        .qsaIndexQK, .qsaOldTail, .qsaIndexQuery, .qsaRaw, .qsaPooled, .qsaPoolNorm, .qsaPoolRotated, .qsaPoolBF16, .qsaTail,
        .qsaIndexNorm, .qsaIndexRotated, .qsaDots, .qsaScores, .qsaMasked, .qsaSortFirst, .qsaSortSecond,
        .qsaPositions, .qsaIDs, .qsaCounts, .qsaAttended, .qsaGated, .qsaNarrowed, .qsaOutput,
        .qsaAttentionPartial, .qsaAttentionSums, .qsaAttentionMaxima, .qsaAttentionScores,
        .qsaQPartial, .qsaKPartial, .qsaVPartial, .qsaIndexPartial, .qsaOutputPartial,
    ]

    package static func layout(
        rows: Int, capacity: Int, capture: Bool, argumentBytes: Int,
        kernels: ReferenceKernels, simdMatrix: Bool, separateExpertBindings: Bool = false, hc: FusedHCDecode,
        gdn: GDNDeviceBlock, attention: PinnedAttention, gdnOnly: Bool = false,
        compact: Bool = ScratchLifetimeCompaction.enabledByDefault
    ) throws -> Layout {
        guard !gdnOnly || (rows == 256 && !capture) else {
            throw EngineError("A GDN-only prefill workspace is the ordinary C256 tile only")
        }
        // Exact row counts retain the original projection/reduction schedules, including short cached suffixes.
        guard (1...256).contains(rows) else {
            throw EngineError("Whole tile requires declared geometry")
        }
        guard ConversationGeometry.contexts.contains(capacity), (1...16384).contains(argumentBytes) else {
            throw EngineError("Whole tile supports only a declared conversation capacity and bounded arguments")
        }
        func partial(_ width: Int, _ outputs: Int) throws -> Int {
            try kernels.pinnedProjectionPlan(rows: rows, width: width, outputs: outputs, simdMatrix: simdMatrix).partialBytes
        }
        func densePartial(_ width: Int, _ outputs: Int) throws -> Int {
            let p = try PinnedDenseLinear.plan(rows: rows, width: width, outputs: outputs)
            return p.kind == .splitK ? rows * outputs * p.splitKPartitions * 4 : 0
        }
        let h = rows == 1 ? (down: 0, up: 0, injection: 0) : try hc.tilePartialBytes(rows: rows)
        let g = rows == 1 ? (qkv: 0, z: 0, dense: 0, output: 0) : try gdn.tilePartialBytes(rows: rows)
        let blocks = capacity / 4
        let poolRows = (rows + 3) / 4
        let sort = try BatchedTopIndices.bufferPlan(rows: rows, columns: blocks, count: 512)
        let attn = try attention.bufferScratchPlan(rows: rows, keys: capacity)
        // The batch stripe byte count is not monotonic with key count: rounding
        // the number of units can leave a smaller final allocation at capacity.
        // Its documented 32 MiB limit safely covers every intervening frontier.
        let attentionScores =
            rows == 1 ? attn.scoreBytes : (attention.batched ? PinnedAttention.maximumScoreScratchBytes : 8 * capacity * 2)
        let hcSizes = [
            rows * 10240 * 4, rows * 10240 * 2, rows * 320 * 4, rows * 320 * 2,
            rows * 10240 * 4, rows * 2560 * 4, rows * 80 * 4, rows * 16, rows * 16,
            capture ? rows * 10240 * 4 : 0, h.down, h.up, h.injection,
        ]
        var sizes = [
            rows * 10240 * 4, rows * 10240 * 2, rows * 10240 * 4, rows * 10240 * 2,
            rows * 10240 * 4, rows * 2560 * 2, rows * 2560 * 2,
        ]
        sizes += hcSizes
        sizes += hcSizes
        sizes += [
            rows * 10240 * 4, rows * 6144 * 4, rows * 48 * 4, rows * 48 * 4, 3 * 10240 * 2,
            rows * 10240 * 4, rows * 48 * 4, rows * 48 * 4, 4, rows * 4,
            rows * 6144 * 4, rows * 6144 * 4, 4, rows * 6144 * 2, rows * 2560 * 4,
            capture ? rows * 10240 * 4 : 0, 48 * 128 * 128 * 4, (rows * 6144 + 48 * 128 * 128) * 4,
            g.qkv, g.z, g.dense, g.dense, g.output,
        ]
        sizes += [
            rows * 12288 * 4, rows * 6144 * 4, rows * 6144 * 4, rows * 6144 * 4, rows * 6144 * 4, rows * 6144 * 2,
            rows * 512 * 4, rows * 512 * 4, rows * 512 * 4, rows * 512 * 2, rows * 512 * 4, rows * 512 * 2,
            rows * 640 * 4, 3 * 128 * 2, rows * 512 * 4, (rows + 3) * 128 * 4,
            poolRows * 128 * 4, poolRows * 128 * 4, poolRows * 128 * 4, poolRows * 128 * 2, 3 * 128 * 2,
            rows * 512 * 4, rows * 512 * 4, min(32 * 1024 * 1024, rows * 4 * blocks * 4),
            rows * blocks * 4, rows * blocks * 4, sort.firstBytes, sort.secondBytes,
            rows * 4, rows * 2051 * 4, rows * 4, 4, 4, rows * 6144 * 4, rows * 6144 * 4, 4, rows * 6144 * 2, rows * 2560 * 4,
            attn.intermediateBytes, attn.sumsBytes, attn.maximaBytes, attentionScores,
        ]
        sizes += try [partial(2560, 12288), partial(2560, 512), partial(2560, 512), densePartial(2560, 640), partial(6144, 2560)]
        sizes += [rows * 512 * 4, rows * 512 * 8, rows * 10 * 4, rows * 10 * 4, 4, try densePartial(2560, 512)]
        sizes += [rows * 640 * 4, rows * 640 * 4, rows * 640 * 2, rows * 2560 * 4, rows * 4, rows * 4, 4]
        sizes += try [partial(2560, 640), partial(2560, 640), partial(640, 2560), densePartial(2560, 1)]
        sizes += rows == 1 ? [10 * 640 * 4, 10 * 640 * 4, 10 * 640 * 2, 10 * 2560 * 4, argumentBytes, 10 * 4] : [0, 0, 0, 0, 0, 0]
        sizes += rows == 1 && separateExpertBindings ? [argumentBytes, 10 * 4] : [0, 0]
        sizes += [rows * 2560 * 4, rows * 2560 * 4, 4, 4, 4]
        guard sizes.count == Slot.allCases.count, sizes.allSatisfy({ $0 >= 0 }) else {
            throw EngineError("Whole tile layout differs from declared roles")
        }
        if gdnOnly { for slot in qsaDataSlots { sizes[slot.rawValue] = 0 } }
        // Captures can inspect intermediate values after both phases; retain the full diagnostic layout.
        let compact = compact && !capture
        var storageIndices = Array(sizes.indices)
        var allocationSizes = sizes
        if compact {
            for (first, second) in compactPairs where sizes[first.rawValue] > 0 && sizes[second.rawValue] > 0 {
                storageIndices[second.rawValue] = first.rawValue
                allocationSizes[first.rawValue] = max(sizes[first.rawValue], sizes[second.rawValue])
                allocationSizes[second.rawValue] = 0
            }
        }
        let plainBytes = sizes.reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1) }
        let bufferBytes = allocationSizes.reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1) }
        // Bounded owner/arrays/dispatch metadata; immutable weights, CPU state,
        // captures and scope maps are independently charged by their owners.
        return Layout(
            rows: rows, capacity: capacity, capture: capture, sizes: sizes,
            storageIndices: storageIndices, allocationSizes: allocationSizes, compact: compact,
            savedBytes: plainBytes - bufferBytes,
            bufferBytes: bufferBytes, admittedBytes: bufferBytes + 131072, gdnOnly: gdnOnly)
    }

    package init(device: MTLDevice, ledger: MemoryLedger, layout: Layout) throws {
        let owner = try WholeWorkspaceAllocation(ledger: ledger, bytes: layout.admittedBytes)
        let allocations: [MTLBuffer?] = try layout.allocationSizes.map { bytes in
            if bytes == 0 { return nil }
            guard bytes <= device.maxBufferLength, let buffer = device.makeBuffer(length: bytes, options: .storageModeShared),
                UInt64(buffer.allocatedSize) <= PrefillMoEBudget.pageRounded(bytes)
            else {
                throw EngineError("Whole tile allocation exceeds admitted page bound", code: "insufficient_memory")
            }
            objc_setAssociatedObject(buffer, &WholeWorkspaceAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return buffer
        }
        buffers = layout.storageIndices.map { allocations[$0] }
        self.layout = layout
    }
    package subscript(_ slot: Slot) -> MTLBuffer {
        guard let buffer = buffers[slot.rawValue] else { preconditionFailure("Whole tile slot \(slot) is not allocated in this layout") }
        return buffer
    }
    package func optional(_ slot: Slot) -> MTLBuffer? { buffers[slot.rawValue] }
    package var attentionHC: FusedHCDecode.RowBuffers { hc(start: Slot.hcANorm.rawValue) }
    package var moeHC: FusedHCDecode.RowBuffers { hc(start: Slot.hcMNorm.rawValue) }
    private func hc(start: Int) -> FusedHCDecode.RowBuffers {
        .init(
            norm: buffers[start]!, normBF16: buffers[start + 1]!, down: buffers[start + 2]!, activation: buffers[start + 3]!,
            rawUp: buffers[start + 4]!, mixed: buffers[start + 5]!, exceptions: buffers[start + 6]!,
            rawInjection: buffers[start + 7]!, injection: buffers[start + 8]!, capturedUp: buffers[start + 9])
    }
    func gdn(weights: WholeLayerWeights.GDN) -> GDNDeviceBlock.RowBuffers {
        .init(
            mixed: self[.gdnMixed], z: self[.gdnZ], b: self[.gdnB], a: self[.gdnA], history: self[.gdnHistory],
            convolution: weights.convolution, bias: weights.bias, logs: weights.logs, prepared: self[.gdnPrepared], beta: self[.gdnBeta],
            decay: self[.gdnDecay], prepareFlags: self[.gdnPrepareFlags], mask: self[.gdnMask], normWeights: weights.norm,
            normed: self[.gdnNormed], gated: self[.gdnGated], gateFlags: self[.gdnGateFlags], narrowed: self[.gdnNarrowed],
            output: self[.gdnOutput], convolved: optional(.gdnConvolved))
    }
    var gdnPartials: GDNDeviceBlock.TilePartials {
        .init(
            qkv: optional(.gdnQPartial), z: optional(.gdnZPartial), b: optional(.gdnBPartial), a: optional(.gdnAPartial),
            output: optional(.gdnOutputPartial))
    }
    var attentionScratch: PinnedAttention.BufferScratch {
        .init(
            intermediate: optional(.qsaAttentionPartial), sums: optional(.qsaAttentionSums),
            maxima: optional(.qsaAttentionMaxima), scores: optional(.qsaAttentionScores))
    }
    var missingExperts: GatheredExpertDecode.RegionBuffers {
        .init(
            gate: self[.expertGate], up: self[.expertUp], product: self[.expertProduct], down: self[.expertDown],
            descriptors: self[.expertMissingDescriptors], ranks: self[.expertMissingRanks])
    }
    var experts: GatheredExpertDecode.RegionBuffers {
        .init(
            gate: self[.expertGate], up: self[.expertUp], product: self[.expertProduct], down: self[.expertDown],
            descriptors: self[.expertDescriptors], ranks: self[.expertRanks])
    }
    package func clearFlags() {
        for slot in [
            Slot.hcAFlags, .hcMFlags, .gdnPrepareFlags, .gdnGateFlags, .qsaPoolFlags, .qsaSelectionFlags, .qsaGateFlags,
            .routerFlags, .sharedFlags, .attentionInjectionFlags, .combinationFlags, .finalInjectionFlags,
        ] {
            self[slot].contents().initializeMemory(as: UInt8.self, repeating: 0, count: layout.sizes[slot.rawValue])
        }
    }
}
