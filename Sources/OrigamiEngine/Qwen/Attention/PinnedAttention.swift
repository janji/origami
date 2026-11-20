import Darwin
import Foundation
import Metal
import ObjectiveC

private final class PinnedAttentionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "pinned_qsa_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum PinnedAttentionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The QSA layers' sparse attention (24 query heads, 2 KV heads, head dim 256, scale 1/16) over a
/// sorted causal selection of at most 2,051 keys per query, reproducing MLX's kernels on the M2 Pro
/// ("medium" architecture) bit for bit (BITWISE). Production always builds it as `.medium`, so every
/// Mac runs the same reduction schedule.
/// - Decode (<= 2 queries): MLX `sdpa_vector` (keys < 1024) or its two-pass form (partition count from
///   `vectorPartitions`), optionally the fused register kernel; BF16 partials, FP32 online softmax.
/// - Prefill (>= 3 queries): Steel-GEMM-order QK (8x8 FP32 MMA, K-eight steps), in-place BF16 softmax,
///   PV with the partial BK=16 tail first. Unbatched holds one 8-row BF16 stripe (at most 4 MiB);
///   batched holds at most 32 MiB of stripes or compact selected-key storage (16K+ keys); the reuse,
///   compact-reuse and exact variants keep every output's MMA order and are checked against the
///   original batched control. It never builds the complete [24, queries, keys] score tensor.
/// K/V are token-major BF16 [token][kv head][256]; queries BF16 [row][head][256]; outputs FP32
/// holding BF16-rounded values. Caller-owned arrays belong to the admitted workspace.
public final class PinnedAttention {
    public static let maximumScoreScratchBytes = 32 * 1024 * 1024
    public static let maximumQueries = 256
    public static let maximumSelectedKeys = 2051
    /// MLX's GPU size class; it sets the decode partition count and so the reduction order.
    public enum Architecture: String, Codable { case small, medium, large }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let ledger: MemoryLedger
    public let architecture: Architecture
    public let batched: Bool
    private let batchedQK, batchedSoftmaxShort, batchedSoftmaxLong, batchedPV: MTLComputePipelineState
    private let prefillQK, prefillPV: MTLComputePipelineState
    private let fusedPrefill: Bool
    private let compactUnionPrepare, compactReuseQK, compactReuseSoftmax, compactReusePV, compactReusePVGrouped: MTLComputePipelineState?
    /// `compact_pv_grouped`: compact PV with one threadgroup per (query pair, KV head) sharing V tiles
    /// across its three 8-row units; bit-identical by construction (checked). Default off; the 32K
    /// release configuration turns it on, and `qsa_exact_kernels` then replaces it with the direct PV.
    package var compactPVGrouped = false
    /// `qsa_exact_kernels`: the BF16 double-buffered compact QK and the direct register-fragment PV on the
    /// compact-reuse path, the direct stripe PV on the prefill-reuse path. Each keeps the replaced kernel's
    /// per-element FP32 MMA order (BITWISE, checked by `QSAExactKernelSelfChecks`); batched prefill calls
    /// outside those two paths keep their kernels and are counted as fallbacks. Default off; every release
    /// configuration turns it on.
    package var qsaExactKernels = false { didSet { pagedVariant?.qsaExactKernels = qsaExactKernels } }
    private let library: MTLLibrary
    private var exactStripePV, exactCompactQK, exactCompactPV: MTLComputePipelineState?
    private var exactPVUnits: UInt64 = 0, exactQKGroups: UInt64 = 0, exactFallbackCalls: UInt64 = 0
    /// Path proof: 8-row PV units encoded with a direct PV kernel (both paths).
    package var qsaExactPVTiles: UInt64 { exactPVUnits + (pagedVariant?.qsaExactPVTiles ?? 0) }
    /// Path proof: compact QK threadgroups encoded with the double-buffered kernel (compact-reuse path only).
    package var qsaExactQKTiles: UInt64 { exactQKGroups + (pagedVariant?.qsaExactQKTiles ?? 0) }
    /// Batched prefill calls made with the flag on that the exact kernels do not serve (counted, never silent).
    package var qsaExactFallbacks: UInt64 { exactFallbackCalls + (pagedVariant?.qsaExactFallbacks ?? 0) }
    /// Diagnostic for the standalone byte comparison: the compact-reuse encoder stops after its QK
    /// dispatch so the compact score region can be compared before the in-place softmax.
    package var diagnosticCompactReuseQKOnly = false
    private var compactReuseCalls = 0
    private var fusedPrefillCounters: MTLBuffer?
    /// Prefill path counters for diagnostics. The compact counters are 64-bit values the GPU keeps as
    /// low-word plus carry in `fusedPrefillCounters`.
    public struct PrefillPathSnapshot: Codable, Sendable {
        public let strategy: String
        public let compactReuseDispatches, compactQKGroupsCompleted, compactQKGroupsSkipped: UInt64
        public let compactSoftmaxGroupsCompleted, compactPVGroupsCompleted: UInt64
        public let compactUnionGroupsCompleted, compactUnionTiles, compactSelectedTiles: UInt64
    }
    /// `qsa_exact_kernels` path proof (both libraries): direct PV units, double-buffered compact QK
    /// threadgroups, and batched prefill calls the exact kernels did not serve. Reported under
    /// wholeExecution; the prefill path snapshot keeps its schema.
    public struct QSAExactKernelCounts: Codable, Sendable {
        public let pvTiles, qkTiles, fallbacks: UInt64
    }
    package var qsaExactKernelCounts: QSAExactKernelCounts {
        .init(pvTiles: qsaExactPVTiles, qkTiles: qsaExactQKTiles, fallbacks: qsaExactFallbacks)
    }
    public var prefillPathSnapshot: PrefillPathSnapshot {
        let own = fusedPrefillCounters?.contents().assumingMemoryBound(to: UInt32.self)
        let other = pagedVariant?.prefillPathSnapshot
        // GPU writers publish low-word additions plus carry. Read only at the
        // existing joined request boundaries, so neither half is in flight.
        func count(_ index: Int) -> UInt64 { UInt64(own?[index] ?? 0) | (UInt64(own?[index + 1] ?? 0) << 32) }
        return .init(
            strategy: fusedPrefill ? "compactReuse" : "off",
            compactReuseDispatches: UInt64(compactReuseCalls) + (other?.compactReuseDispatches ?? 0),
            compactQKGroupsCompleted: count(2) + (other?.compactQKGroupsCompleted ?? 0),
            compactQKGroupsSkipped: count(4) + (other?.compactQKGroupsSkipped ?? 0),
            compactSoftmaxGroupsCompleted: count(8) + (other?.compactSoftmaxGroupsCompleted ?? 0),
            compactPVGroupsCompleted: count(6) + (other?.compactPVGroupsCompleted ?? 0),
            compactUnionGroupsCompleted: count(10) + (other?.compactUnionGroupsCompleted ?? 0),
            compactUnionTiles: count(12) + (other?.compactUnionTiles ?? 0),
            compactSelectedTiles: count(14) + (other?.compactSelectedTiles ?? 0))
    }
    private var prefillReuseCount = 0
    package var prefillReuseDispatches: Int { prefillReuseCount + (pagedVariant?.prefillReuseDispatches ?? 0) }
    private let compactPrepare, compactQK, compactSoftmax, compactPV: MTLComputePipelineState
    private let vector, partial, finish, qk, softmaxShort, softmaxLong, pv: MTLComputePipelineState
    private let partitionPositions, selectedPartial: MTLComputePipelineState
    private let pagedKV: Bool
    private let registerDecode: Bool
    private let registerLibrary: MTLLibrary?
    private var registerPipelines: [Int: MTLComputePipelineState] = [:]
    private var registerDispatches = 0
    package var registerDecodeDispatches: Int { registerDispatches + (pagedVariant?.registerDecodeDispatches ?? 0) }
    private var longFusedDispatches = 0
    package var longFusedDecodeDispatches: Int { longFusedDispatches + (pagedVariant?.longFusedDecodeDispatches ?? 0) }
    private let pageArguments: MTLArgumentEncoder?
    private var pagedVariant: PinnedAttention?
    private var indirectResources: [MTLResource]?

    /// The architecture is explicit because MLX branches on the final character
    /// of MTLDevice.architecture.name: s=medium, d=large, otherwise small.
    public convenience init(
        ledger: MemoryLedger, architecture: Architecture, batched: Bool = false, registerDecode: Bool = false, fusedPrefill: Bool = false
    ) throws {
        try self.init(
            ledger: ledger, architecture: architecture, batched: batched, pagedKV: false, registerDecode: registerDecode,
            fusedPrefill: fusedPrefill)
    }
    private init(ledger: MemoryLedger, architecture: Architecture, batched: Bool, pagedKV: Bool, registerDecode: Bool, fusedPrefill: Bool)
        throws
    {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Pinned attention Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(
            source: "#define ORIGAMI_PAGED_KV \(pagedKV ? 1 : 0)\n" + QSAPagedAttentionStorage.preamble + "\n" + Self.source + "\n"
                + CompactSelectedAttention.source + "\n" + PrefillAttentionReuseKernel.source + "\n" + QSAExactPrefillKernels.stripeSource
                + "\n" + (fusedPrefill ? CompactPrefillReuse.source + "\n" + QSAExactPrefillKernels.compactSource : "") + "\n"
                + (registerDecode ? RegisterAttentionKernel.source : ""), options: options)
        self.library = library
        self.fusedPrefill = fusedPrefill
        self.pagedKV = pagedKV
        self.registerDecode = registerDecode
        self.registerLibrary = registerDecode ? library : nil
        if pagedKV {
            guard let function = library.makeFunction(name: "origami_paged_argument_layout") else {
                throw EngineError("Missing shared QSA argument layout")
            }
            pageArguments = function.makeArgumentEncoder(bufferIndex: 0)
        } else {
            pageArguments = nil
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try Self.makePipeline(library, device, name, pagedKV: pagedKV)
        }
        vector = try pipeline("origami_sdpa_vector")
        partial = try pipeline("origami_sdpa_partial")
        partitionPositions = try pipeline("origami_sdpa_partition_positions")
        selectedPartial = try pipeline("origami_sdpa_selected_partial")
        finish = try pipeline("origami_sdpa_finish")
        qk = try pipeline("origami_qk_mma")
        softmaxShort = try pipeline("origami_softmax_short")
        softmaxLong = try pipeline("origami_softmax_long")
        pv = try pipeline("origami_pv_mma")
        batchedQK = try pipeline("origami_qk_mma_batched")
        batchedSoftmaxShort = try pipeline("origami_softmax_short_batched")
        batchedSoftmaxLong = try pipeline("origami_softmax_long_batched")
        batchedPV = try pipeline("origami_pv_mma_batched")
        prefillQK = try pipeline("origami_qk_gqa_reuse_k16")
        prefillPV = try pipeline("origami_pv_full_width")
        compactPrepare = try pipeline("origami_compact_prepare")
        compactQK = try pipeline("origami_compact_qk")
        compactSoftmax = try pipeline("origami_compact_softmax")
        compactPV = try pipeline("origami_compact_pv")
        compactUnionPrepare = try fusedPrefill ? pipeline("origami_compact_union_prepare") : nil
        compactReuseQK = try fusedPrefill ? pipeline("origami_compact_qk_reuse") : nil
        compactReuseSoftmax = try fusedPrefill ? pipeline("origami_compact_softmax_reuse") : nil
        compactReusePV = try fusedPrefill ? pipeline("origami_compact_pv_reuse") : nil
        compactReusePVGrouped = try fusedPrefill ? pipeline("origami_compact_pv_reuse_kv") : nil
        if let compactReusePVGrouped {
            guard compactReusePVGrouped.maxTotalThreadsPerThreadgroup >= 768,
                compactReusePVGrouped.staticThreadgroupMemoryLength <= 16384
            else {
                throw EngineError(
                    "Grouped compact PV exceeded its threadgroup owner: \(compactReusePVGrouped.staticThreadgroupMemoryLength) bytes, \(compactReusePVGrouped.maxTotalThreadsPerThreadgroup) threads"
                )
            }
        }
        if let compactReuseQK, let compactReusePV {
            guard compactReuseQK.staticThreadgroupMemoryLength <= 15360,
                compactReusePV.staticThreadgroupMemoryLength <= 11264
            else {
                throw EngineError("Compact prefill reuse exceeded its threadgroup owner")
            }
        }
        guard vector.maxTotalThreadsPerThreadgroup >= 1024,
            finish.maxTotalThreadsPerThreadgroup >= 1024,
            softmaxLong.maxTotalThreadsPerThreadgroup >= 1024,
            batchedSoftmaxLong.maxTotalThreadsPerThreadgroup >= 1024,
            partial.maxTotalThreadsPerThreadgroup >= 768,
            selectedPartial.maxTotalThreadsPerThreadgroup >= 768,
            prefillQK.maxTotalThreadsPerThreadgroup >= 384,
            prefillPV.maxTotalThreadsPerThreadgroup >= 256,
            prefillQK.staticThreadgroupMemoryLength <= 15360,
            prefillPV.staticThreadgroupMemoryLength <= 11264
        else {
            throw EngineError("Pinned attention requires its exact 1024-thread source reduction schedule")
        }
        self.device = device
        self.queue = queue
        self.ledger = ledger
        self.architecture = architecture
        self.batched = batched
        if fusedPrefill {
            let counters = try allocate(bytes: 256)
            memset(counters.contents(), 0, 256)
            fusedPrefillCounters = counters
        }
        // Compiled here so an unavailable geometry fails at startup, not in the first long decode.
        if registerDecode { for partitions in [128, 256] { _ = try registerPipeline(partitions: partitions) } }
        // Optional long variants keep the original two-pass path on a device that cannot admit their
        // threadgroup geometry. Resolve availability once, never compile/retry on the decode hot path.
        if registerDecode { for partitions in [512, 1024] { _ = try? registerPipeline(partitions: partitions) } }
    }

    private static func makePipeline(
        _ library: MTLLibrary, _ device: MTLDevice, _ name: String, pagedKV: Bool,
        threads explicitThreads: Int? = nil
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else { throw EngineError("Missing pinned attention kernel \(name)") }
        if name.hasPrefix("origami_compact_") || explicitThreads != nil {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.label = name
            let threads: Int
            if let explicitThreads {
                threads = explicitThreads
            } else if name.contains("softmax") {
                threads = 1024
            } else if name == "origami_compact_qk_reuse" || name == "origami_compact_qk_reuse_dbuf" {
                threads = 384
            } else if name == "origami_compact_pv_reuse_kv" {
                threads = 768
            } else if ["origami_compact_prepare", "origami_compact_union_prepare", "origami_compact_pv_reuse"].contains(name) {
                threads = 256
            } else {
                threads = 32
            }
            descriptor.maxTotalThreadsPerThreadgroup = threads
            let result = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            guard result.maxTotalThreadsPerThreadgroup >= threads,
                result.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Compact attention arithmetic geometry unavailable")
            }
            return result
        }
        if pagedKV, ["origami_sdpa_vector", "origami_sdpa_partial", "origami_sdpa_selected_partial"].contains(name) {
            // Indirect page loads otherwise make this compiler report a
            // 640-thread limit. Declare the original reduction geometry
            // so register allocation accommodates it; never alter the tree.
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.label = name
            descriptor.maxTotalThreadsPerThreadgroup = name == "origami_sdpa_vector" ? 1024 : 768
            return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = name
        return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    }

    /// The direct stripe PV (8K prefill-reuse path): one SIMD group per unit x 64 columns, 128 threads,
    /// no threadgroup memory. Created on first use; an unavailable geometry is an error, never a fallback.
    private func exactStripePipeline() throws -> MTLComputePipelineState {
        if let exactStripePV { return exactStripePV }
        let made = try Self.makePipeline(library, device, "origami_pv_direct_stripe_c64", pagedKV: pagedKV, threads: 128)
        guard made.threadExecutionWidth == 32, made.maxTotalThreadsPerThreadgroup >= 128, made.staticThreadgroupMemoryLength == 0 else {
            throw EngineError(
                "Direct stripe PV exceeded its 128-thread register geometry: \(made.staticThreadgroupMemoryLength) bytes, \(made.maxTotalThreadsPerThreadgroup) threads"
            )
        }
        exactStripePV = made
        return made
    }
    /// The compact-reuse pair: the double-buffered QK (384 threads, 14,912 bytes under the 15,360-byte
    /// compact QK owner) and the direct compact PV (128 threads, no threadgroup memory).
    private func exactCompactPipelines() throws -> (qk: MTLComputePipelineState, pv: MTLComputePipelineState) {
        if let exactCompactQK, let exactCompactPV { return (exactCompactQK, exactCompactPV) }
        guard fusedPrefill else { throw EngineError("Exact compact kernels require compact-reuse fused prefill") }
        let qk = try Self.makePipeline(library, device, "origami_compact_qk_reuse_dbuf", pagedKV: pagedKV)
        guard qk.threadExecutionWidth == 32, qk.maxTotalThreadsPerThreadgroup >= 384, qk.staticThreadgroupMemoryLength <= 15360 else {
            throw EngineError(
                "Double-buffered compact QK exceeded its threadgroup owner: \(qk.staticThreadgroupMemoryLength) bytes, \(qk.maxTotalThreadsPerThreadgroup) threads"
            )
        }
        let pv = try Self.makePipeline(library, device, "origami_pv_direct_compact_c64", pagedKV: pagedKV, threads: 128)
        guard pv.threadExecutionWidth == 32, pv.maxTotalThreadsPerThreadgroup >= 128, pv.staticThreadgroupMemoryLength == 0 else {
            throw EngineError(
                "Direct compact PV exceeded its 128-thread register geometry: \(pv.staticThreadgroupMemoryLength) bytes, \(pv.maxTotalThreadsPerThreadgroup) threads"
            )
        }
        exactCompactQK = qk
        exactCompactPV = pv
        return (qk, pv)
    }
    /// Creates every exact-kernel pipeline this library can select, so an unavailable geometry fails at
    /// configuration time rather than at the first prefill tile. The paged variant prepares itself when made.
    package func prepareExactKernels() throws {
        guard qsaExactKernels else { throw EngineError("qsa_exact_kernels is not enabled on this attention") }
        _ = try exactStripePipeline()
        if fusedPrefill { _ = try exactCompactPipelines() }
        if let pagedVariant { try pagedVariant.prepareExactKernels() }
    }

    private func sharedVariant() throws -> PinnedAttention {
        guard !pagedKV else { return self }
        if let pagedVariant { return pagedVariant }
        let made = try PinnedAttention(
            ledger: ledger, architecture: architecture, batched: batched, pagedKV: true, registerDecode: registerDecode,
            fusedPrefill: fusedPrefill)
        made.compactPVGrouped = compactPVGrouped
        made.qsaExactKernels = qsaExactKernels
        if qsaExactKernels { try made.prepareExactKernels() }
        pagedVariant = made
        return made
    }
    package func makePagedStorage(capacityTokens: Int) throws -> QSAPagedAttentionStorage {
        let variant = try sharedVariant()
        return try QSAPagedAttentionStorage(
            device: device, ledger: ledger, arguments: variant.pageArguments!, capacityTokens: capacityTokens)
    }

    private func registerPipeline(partitions: Int) throws -> MTLComputePipelineState {
        if let cached = registerPipelines[partitions] { return cached }
        guard registerDecode, [128, 256, 512, 1024].contains(partitions), let registerLibrary else {
            throw EngineError("Unsupported register attention partition schedule")
        }
        var count = UInt32(partitions)
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&count, type: .uint, index: 0)
        let descriptor = MTLComputePipelineDescriptor()
        if partitions <= 256 {
            descriptor.computeFunction = try registerLibrary.makeFunction(name: "origami_sdpa_register_fused", constantValues: constants)
        } else {
            guard let function = registerLibrary.makeFunction(name: "origami_sdpa_long_fused_\(partitions)") else {
                throw EngineError("Long-context fused attention function unavailable")
            }
            descriptor.computeFunction = function
        }
        descriptor.maxTotalThreadsPerThreadgroup = 1024
        let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 1024,
            pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
        else {
            throw EngineError("Register attention resource geometry differs")
        }
        registerPipelines[partitions] = pipeline
        return pipeline
    }

    func encodePagedAttention(
        _ command: MTLCommandBuffer, query: MTLBuffer, read: QSAPagedAttentionStorage.Read,
        positions: EncodedPositions, output: MTLBuffer, scratch: BufferScratch
    ) throws {
        guard positions.owner === self, read.command === command,
            read.tokens == positions.keys
        else {
            throw EngineError("Shared QSA positions or reader belong to a different transaction")
        }
        if let lease = read.fileLease, lease.sparse {
            guard positions.hostPrepared else { throw EngineError("Sparse file-KV reads require a joined/host-validated selection ticket") }
            var selected: [[UInt32]] = []
            for row in 0..<positions.rows {
                let count = Int(positions.counts.contents().assumingMemoryBound(to: UInt32.self)[row])
                guard count <= positions.stride else { throw EngineError("File-KV selection count changed after preparation") }
                selected.append(
                    Array(
                        UnsafeBufferPointer(
                            start: positions.ids.contents().assumingMemoryBound(to: UInt32.self)
                                .advanced(by: row * positions.stride), count: count)))
            }
            try lease.validateSelection(selected, totalTokens: positions.keys)
        }
        let variant = try sharedVariant()
        guard variant.indirectResources == nil else { throw EngineError("Shared QSA encoder is reentrant") }
        variant.indirectResources = read.resources
        defer { variant.indirectResources = nil }
        let encoded = EncodedPositions(
            owner: variant, command: command, ids: positions.ids, counts: positions.counts,
            rows: positions.rows, keys: positions.keys, stride: positions.stride, hostPrepared: positions.hostPrepared)
        try variant.encodeAttention(
            command, query: query, key: read.owner.table, value: read.owner.table,
            positions: encoded, output: output, scratch: scratch)
    }

    /// How many 8-row score stripes (units) fit in the 32 MiB batched scratch, and the batches that follow.
    public struct UnfusedBatchPlan: Sendable {
        public let units, unitsPerBatch, batches, scoreScratchBytes: Int
        public var encoders: Int { batches * 3 }
    }

    /// One unit contains two query tokens and four heads sharing one KV head.
    /// The final unit group can contain fewer query rows without changing its
    /// score stride. Geometry is checked before multiplication or admission.
    public static func unfusedBatchPlan(queries: Int, keys: Int) throws -> UnfusedBatchPlan {
        guard (3...maximumQueries).contains(queries), keys >= queries,
            keys <= ModelContract.maximumContext
        else {
            throw EngineError("Invalid batched pinned attention geometry")
        }
        let units = ((queries + 1) / 2) * 6
        let stripeBytes = 8 * keys * 2
        let perBatch = min(units, maximumScoreScratchBytes / stripeBytes)
        guard perBatch > 0 else { throw EngineError("Pinned attention stripe exceeds scratch limit") }
        return UnfusedBatchPlan(
            units: units, unitsPerBatch: perBatch,
            batches: (units + perBatch - 1) / perBatch, scoreScratchBytes: perBatch * stripeBytes)
    }

    public static func vectorPartitions(keys: Int, queries: Int, architecture: Architecture) throws -> Int {
        guard (1...ModelContract.maximumContext).contains(keys), (1...2).contains(queries), keys >= queries else {
            throw EngineError("Invalid pinned vector attention geometry")
        }
        if keys < (architecture == .small ? 4096 : 1024) { return 0 }
        switch architecture {
        case .small: return 64
        case .medium:
            if keys <= 1024 { return 64 }
            if keys <= 8192 { return 128 }
            if keys <= 32768 { return 256 }
            if keys <= 65536 { return 512 }
            return 1024
        case .large:
            if keys < 16384 { return 128 }
            if keys < 65536 { return 512 }
            return 1024
        }
    }

    /// Positions are full-domain, sorted, unique, and already causally filtered.
    /// An empty selection preserves the source behavior: zero for fused vector
    /// attention, uniform probabilities over ALL keys for unfused attention.
    public func attention(
        queries: [Float], oldKeys: [UInt16], newKeys: [UInt16],
        oldValues: [UInt16], newValues: [UInt16],
        selectedPositions: [[UInt32]], decodeStorage: QSADecodeStorage? = nil
    ) throws -> [Float] {
        guard !queries.isEmpty, queries.count % 6144 == 0,
            queries.count / 6144 <= Self.maximumQueries,
            oldKeys.count % 512 == 0, newKeys.count % 512 == 0,
            oldKeys.count == oldValues.count, newKeys.count == newValues.count,
            oldKeys.count / 512 <= ModelContract.maximumContext,
            newKeys.count / 512 <= ModelContract.maximumContext - oldKeys.count / 512
        else {
            throw EngineError("Invalid pinned attention input geometry")
        }
        let rows = queries.count / 6144
        let keys = (oldKeys.count + newKeys.count) / 512
        guard keys >= rows, selectedPositions.count == rows else { throw EngineError("Invalid pinned attention key domain") }
        let causalOffset = keys - rows
        var stride = 1
        for (row, ids) in selectedPositions.enumerated() {
            guard ids.count <= Self.maximumSelectedKeys else { throw EngineError("Pinned attention selection exceeds budget") }
            var previous: UInt32?
            for id in ids {
                guard Int(id) <= causalOffset + row, previous == nil || previous! < id else {
                    throw EngineError("Pinned attention positions must be sorted unique causal key positions")
                }
                previous = id
            }
            stride = max(stride, ids.count)
        }
        guard decodeStorage == nil || rows == 1 else { throw EngineError("Decode storage requires exactly one chronological query") }
        return try autoreleasepool {
            if let decodeStorage {
                try decodeStorage.stage(oldKeys: oldKeys, newKeys: newKeys, oldValues: oldValues, newValues: newValues)
            }
            var joinedSuccess = false
            defer { decodeStorage?.complete(success: joinedSuccess) }
            func buffer(_ name: String, _ bytes: Int) throws -> MTLBuffer {
                if let decodeStorage { return try decodeStorage.buffer(name: name, bytes: bytes) { try allocate(bytes: $0) } }
                return try allocate(bytes: bytes)
            }
            let query = try buffer("query", queries.count * 2)
            let queryPointer = query.contents().assumingMemoryBound(to: UInt16.self)
            for index in queries.indices { queryPointer[index] = BF16.encode(queries[index]) }
            let key = try decodeStorage?.keys ?? concatenateBuffer(oldKeys, newKeys)
            let value = try decodeStorage?.values ?? concatenateBuffer(oldValues, newValues)
            let ids = try buffer("ids", rows * stride * 4)
            let counts = try buffer("counts", rows * 4)
            let idsPointer = ids.contents().assumingMemoryBound(to: UInt32.self)
            let countsPointer = counts.contents().assumingMemoryBound(to: UInt32.self)
            for row in 0..<rows {
                countsPointer[row] = UInt32(selectedPositions[row].count)
                selectedPositions[row].withUnsafeBytes { source in
                    if !source.isEmpty {
                        UnsafeMutableRawPointer(idsPointer.advanced(by: row * stride)).copyMemory(
                            from: source.baseAddress!, byteCount: source.count)
                    }
                }
            }
            let output = try buffer("output", queries.count * 4)
            guard let command = queue.makeCommandBuffer() else {
                throw EngineError("Pinned attention command allocation failed", code: "insufficient_memory")
            }
            command.label = "pinned-attention"
            if rows <= 2 {
                let partitions = try Self.vectorPartitions(keys: keys, queries: rows, architecture: architecture)
                let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(partitions)]
                if partitions == 0 {
                    try encode(
                        command, vector, [query, key, value, ids, counts, output], shape,
                        grid: MTLSize(width: 24, height: rows, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
                } else {
                    let intermediate = try buffer("intermediate", rows * 24 * partitions * 256 * 2)
                    let sums = try buffer("sums", rows * 24 * partitions * 4)
                    let maxima = try buffer("maxima", rows * 24 * partitions * 4)
                    try encode(
                        command, partial, [query, key, value, ids, counts, intermediate, sums, maxima], shape,
                        grid: MTLSize(width: 2, height: 1, depth: partitions), group: MTLSize(width: 32, height: 12, depth: rows))
                    try encode(
                        command, finish, [intermediate, sums, maxima, output], shape,
                        grid: MTLSize(width: 24, height: rows, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
                }
            } else if batched {
                let scores = try allocate(bytes: batchedScratchBytes(rows: rows, keys: keys))
                try encodeBatched(
                    command, query: query, key: key, value: value, ids: ids, counts: counts,
                    output: output, scores: scores, rows: rows, keys: keys, stride: stride)
            } else {
                let stripe = try allocate(bytes: 8 * keys * 2)
                let softmax = keys <= 4096 ? softmaxShort : softmaxLong
                let softmaxThreads = keys <= 4096 ? ((keys + 127) / 128) * 32 : 1024
                for rowBase in Swift.stride(from: 0, to: rows, by: 8) {
                    let stripeRows = min(8, rows - rowBase)
                    for head in 0..<24 {
                        let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(rowBase), UInt32(stripeRows), UInt32(head)]
                        try encode(
                            command, qk, [query, key, ids, counts, stripe], shape,
                            grid: MTLSize(width: (keys + 7) / 8, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
                        try encode(
                            command, softmax, [stripe], shape,
                            grid: MTLSize(width: stripeRows, height: 1, depth: 1),
                            group: MTLSize(width: softmaxThreads, height: 1, depth: 1))
                        try encode(
                            command, pv, [stripe, value, output], shape,
                            grid: MTLSize(width: 32, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
                    }
                }
            }
            command.commit()
            try command.waitBounded()
            guard command.status == .completed else { throw EngineError.gpu("Pinned attention dispatch failed", command.error) }
            let result = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: queries.count))
            joinedSuccess = true
            return result
        }
    }

    /// Original host arithmetic also reads the shared prefix. Only the new
    /// projection and bounded query/scratch are uploaded for short append scopes
    /// and exceptional original-path repair.
    package func attentionPaged(
        queries: [Float], state: QSAState, newKeys: [UInt16], newValues: [UInt16],
        selectedPositions: [[UInt32]], storage provided: QSAPagedAttentionStorage? = nil
    ) throws -> [Float] {
        let rows = queries.count / 6144
        let keys = state.offset + rows
        guard state.pages != nil, !queries.isEmpty, queries.count % 6144 == 0, (1...256).contains(rows),
            newKeys.count == rows * 512, newValues.count == newKeys.count, selectedPositions.count == rows
        else {
            throw EngineError("Invalid shared-prefix host attention geometry")
        }
        return try autoreleasepool {
            let storage = try provided ?? makePagedStorage(capacityTokens: keys)
            let query = try allocate(bytes: queries.count * 2)
            let newK = try allocate(bytes: newKeys.count * 2)
            let newV = try allocate(bytes: newValues.count * 2)
            let ids = try allocate(bytes: rows * Self.maximumSelectedKeys * 4)
            let counts = try allocate(bytes: rows * 4)
            let output = try allocate(bytes: queries.count * 4)
            for index in queries.indices { query.contents().assumingMemoryBound(to: UInt16.self)[index] = BF16.encode(queries[index]) }
            newKeys.withUnsafeBytes { newK.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            newValues.withUnsafeBytes { newV.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            let plan = try bufferScratchPlan(rows: rows, keys: keys)
            func scratch(_ bytes: Int) throws -> MTLBuffer? { bytes == 0 ? nil : try allocate(bytes: bytes) }
            let buffers = try BufferScratch(
                intermediate: scratch(plan.intermediateBytes), sums: scratch(plan.sumsBytes),
                maxima: scratch(plan.maximaBytes), scores: scratch(plan.scoreBytes))
            guard let command = queue.makeCommandBuffer() else { throw EngineError("Shared-prefix attention command unavailable") }
            let positions = try encodeHostPositions(command, selected: selectedPositions, keys: keys, ids: ids, counts: counts)
            let read = try storage.stage(
                command, state: state, newKeys: newK, newValues: newV, rows: rows,
                selected: rows <= 2 ? selectedPositions : nil)
            do {
                try encodePagedAttention(command, query: query, read: read, positions: positions, output: output, scratch: buffers)
                command.commit()
                try command.waitBounded()
                guard command.status == .completed else { throw EngineError.gpu("Shared-prefix attention command failed", command.error) }
                try storage.finish(read, success: true)
            } catch {
                // Report the original failure, not a secondary retirement error.
                try? storage.finish(read, success: false)
                throw error
            }
            return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: queries.count))
        }
    }

    /// A bounded causal selection prepared for this command. Only factories in
    /// this type construct the ticket; attention never accepts unchecked counts.
    package struct EncodedPositions {
        fileprivate let owner: PinnedAttention
        fileprivate let command: MTLCommandBuffer
        fileprivate let ids, counts: MTLBuffer
        fileprivate let rows, keys, stride: Int
        fileprivate let hostPrepared: Bool
    }
    /// Caller-owned scratch: two-pass decode partials (BF16 outputs, FP32 sums and maxima) and the
    /// prefill score region, or the decode bucketed-position region in `scores`.
    package struct BufferScratch {
        package let intermediate, sums, maxima, scores: MTLBuffer?
        package init(intermediate: MTLBuffer?, sums: MTLBuffer?, maxima: MTLBuffer?, scores: MTLBuffer?) {
            self.intermediate = intermediate
            self.sums = sums
            self.maxima = maxima
            self.scores = scores
        }
    }
    /// Only the bounded producer can mint this ticket. Its repair flag remains
    /// part of the enclosing transaction; invalid results must never publish
    /// state or reach a later routing/I/O decision.
    func encodedPositions(_ selection: QSADeviceSelection.Encoded) throws -> EncodedPositions {
        guard selection.command.commandQueue.device.registryID == device.registryID,
            selection.command.status == .notEnqueued
        else { throw EngineError("QSA device selection command/device differs") }
        return EncodedPositions(
            owner: self, command: selection.command, ids: selection.ids, counts: selection.counts,
            rows: selection.rows, keys: selection.keys, stride: Self.maximumSelectedKeys, hostPrepared: false)
    }
    /// A real dependency boundary for file-KV vector reads. The old producer
    /// must be joined; the new host ticket revalidates sorted causal IDs. A
    /// producer repair flag stays set and forces the enclosing original replay.
    func joinedPositions(_ selection: QSADeviceSelection.Encoded, command: MTLCommandBuffer) throws -> (EncodedPositions, [[UInt32]]) {
        guard selection.command.status == .completed, selection.command.commandQueue.device.registryID == device.registryID,
            command.status == .notEnqueued, (1...2).contains(selection.rows),
            selection.ids.storageMode == .shared, selection.counts.storageMode == .shared,
            selection.exceptions.storageMode == .shared, selection.exceptions.length >= 4,
            selection.ids.length >= selection.rows * Self.maximumSelectedKeys * 4,
            selection.counts.length >= selection.rows * 4
        else { throw EngineError("File-KV selection producer is not joined/bounded") }
        let flagged = selection.exceptions.contents().load(as: UInt32.self) != 0
        var selected: [[UInt32]] = []
        for row in 0..<selection.rows {
            let count = flagged ? 0 : Int(selection.counts.contents().assumingMemoryBound(to: UInt32.self)[row])
            guard count <= Self.maximumSelectedKeys else { throw EngineError("Joined file-KV selection exceeds the native bound") }
            selected.append(
                Array(
                    UnsafeBufferPointer(
                        start: selection.ids.contents().assumingMemoryBound(to: UInt32.self)
                            .advanced(by: row * Self.maximumSelectedKeys), count: count)))
        }
        return (
            try encodeHostPositions(command, selected: selected, keys: selection.keys, ids: selection.ids, counts: selection.counts),
            selected
        )
    }
    /// The byte sizes `BufferScratch` must provide for one (rows, keys) geometry.
    package struct BufferScratchPlan {
        package let intermediateBytes, sumsBytes, maximaBytes, scoreBytes: Int
    }
    package func bufferScratchPlan(rows: Int, keys: Int) throws -> BufferScratchPlan {
        guard (1...Self.maximumQueries).contains(rows), (rows...ModelContract.maximumContext).contains(keys) else {
            throw EngineError("Invalid encoded attention geometry")
        }
        if rows <= 2 {
            let partitions = try Self.vectorPartitions(keys: keys, queries: rows, architecture: architecture)
            // One count and ceil(keys/partitions) absolute IDs per partition.
            // Sorted input IDs preserve each original modulo-partition order.
            let selectedTraversal = architecture == .medium && partitions >= 256
            let positionBytes = selectedTraversal ? rows * partitions * (1 + (keys + partitions - 1) / partitions) * 4 : 0
            return BufferScratchPlan(
                intermediateBytes: rows * 24 * partitions * 256 * 2,
                sumsBytes: rows * 24 * partitions * 4, maximaBytes: rows * 24 * partitions * 4, scoreBytes: positionBytes)
        }
        return BufferScratchPlan(
            intermediateBytes: 0, sumsBytes: 0, maximaBytes: 0,
            scoreBytes: try batched ? batchedScratchBytes(rows: rows, keys: keys) : 8 * keys * 2)
    }

    /// Existing host selection control for interface checks and explicit capture
    /// boundaries. The buffers may not be overwritten until this command joins.
    package func encodeHostPositions(
        _ command: MTLCommandBuffer, selected: [[UInt32]], keys: Int,
        ids: MTLBuffer, counts: MTLBuffer
    ) throws -> EncodedPositions {
        let rows = selected.count
        guard (1...Self.maximumQueries).contains(rows), (rows...ModelContract.maximumContext).contains(keys),
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            ids !== counts, ids.device.registryID == device.registryID, counts.device.registryID == device.registryID
        else {
            throw EngineError("Invalid encoded attention selection geometry")
        }
        var stride = 1
        for (row, positions) in selected.enumerated() {
            guard positions.count <= Self.maximumSelectedKeys else { throw EngineError("Encoded attention selection exceeds bound") }
            var previous: UInt32?
            for id in positions {
                guard Int(id) <= keys - rows + row, previous == nil || previous! < id else {
                    throw EngineError("Encoded attention selection is not sorted, unique and causal")
                }
                previous = id
            }
            stride = max(stride, positions.count)
        }
        guard ids.length >= rows * stride * 4, counts.length >= rows * 4,
            ids.storageMode == .shared, counts.storageMode == .shared
        else { throw EngineError("Selection backing too small/not host visible") }
        for row in 0..<rows {
            counts.contents().assumingMemoryBound(to: UInt32.self)[row] = UInt32(selected[row].count)
            selected[row].withUnsafeBytes { source in
                if !source.isEmpty {
                    ids.contents().advanced(by: row * stride * 4).copyMemory(from: source.baseAddress!, byteCount: source.count)
                }
            }
        }
        return EncodedPositions(
            owner: self, command: command, ids: ids, counts: counts, rows: rows, keys: keys, stride: stride, hostPrepared: true)
    }

    /// Buffer-only entry. Two-pass vector attention visits stable selected-key
    /// buckets; its original per-partition arithmetic and final pass are intact.
    /// Query and K/V storage is BF16. Provisional selections and all scratch
    /// remain owned by the caller until joined completion and validation.
    package func encodeAttention(
        _ command: MTLCommandBuffer, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
        positions: EncodedPositions, output: MTLBuffer, scratch: BufferScratch
    ) throws {
        guard positions.owner === self, positions.command === command,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Encoded attention selection/command identity differs")
        }
        let rows = positions.rows
        let keys = positions.keys
        let stride = positions.stride
        let ids = positions.ids
        let counts = positions.counts
        let p = try bufferScratchPlan(rows: rows, keys: keys)
        var spans: [(MTLBuffer, Int)] = [
            (query, rows * 6144 * 2),
            (ids, rows * stride * 4), (counts, rows * 4), (output, rows * 6144 * 4),
        ]
        if pagedKV {
            guard key === value, indirectResources?.isEmpty == false, let pageArguments else {
                throw EngineError("Shared QSA indirect readers are absent")
            }
            spans.append((key, pageArguments.encodedLength))
        } else {
            spans += [(key, keys * 512 * 2), (value, keys * 512 * 2)]
        }
        for (buffer, bytes) in [
            (scratch.intermediate, p.intermediateBytes), (scratch.sums, p.sumsBytes),
            (scratch.maxima, p.maximaBytes), (scratch.scores, p.scoreBytes),
        ] where bytes > 0 {
            guard let buffer else { throw EngineError("Encoded attention scratch missing") }
            spans.append((buffer, bytes))
        }
        guard spans.allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(spans.map { ObjectIdentifier($0.0) }).count == spans.count
        else { throw EngineError("Encoded attention spans overlap or exceed buffers") }
        if registerDecode && architecture == .medium && rows == 1 && (1025...32768).contains(keys) {
            let partitions = try Self.vectorPartitions(keys: keys, queries: rows, architecture: architecture)
            try encode(
                command, registerPipeline(partitions: partitions), [query, key, value, ids, counts, output],
                [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(partitions)],
                grid: MTLSize(width: 24, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
            registerDispatches += 1
            return
        }
        if registerDecode && architecture == .medium && rows == 1 && keys > 32768 {
            let partitions = try Self.vectorPartitions(keys: keys, queries: rows, architecture: architecture)
            if let pipeline = registerPipelines[partitions] {
                // These buffers are already validated and charged by the ordinary two-pass workspace.
                // The fused long path retains its BF16 partial writes, without global position buckets.
                try encode(
                    command, pipeline,
                    [
                        query, key, value, ids, counts,
                        scratch.intermediate!, scratch.sums!, scratch.maxima!, output,
                    ],
                    [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(partitions)],
                    grid: MTLSize(width: 24, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
                longFusedDispatches += 1
                return
            }
        }
        if rows <= 2 {
            let partitions = try Self.vectorPartitions(keys: keys, queries: rows, architecture: architecture)
            let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(partitions)]
            if partitions == 0 {
                try encode(
                    command, vector, [query, key, value, ids, counts, output], shape,
                    grid: MTLSize(width: 24, height: rows, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
            } else {
                let intermediate = scratch.intermediate!
                let sums = scratch.sums!
                let maxima = scratch.maxima!
                if p.scoreBytes > 0 {
                    let partitioned = scratch.scores!
                    try encode(
                        command, partitionPositions, [ids, counts, partitioned], shape,
                        grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 1, height: 1, depth: 1))
                    try encode(
                        command, selectedPartial, [query, key, value, partitioned, counts, intermediate, sums, maxima], shape,
                        grid: MTLSize(width: 2, height: 1, depth: partitions), group: MTLSize(width: 32, height: 12, depth: rows))
                } else {
                    try encode(
                        command, partial, [query, key, value, ids, counts, intermediate, sums, maxima], shape,
                        grid: MTLSize(width: 2, height: 1, depth: partitions), group: MTLSize(width: 32, height: 12, depth: rows))
                }
                try encode(
                    command, finish, [intermediate, sums, maxima, output], shape,
                    grid: MTLSize(width: 24, height: rows, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
            }
        } else if batched {
            try encodeBatched(
                command, query: query, key: key, value: value, ids: ids, counts: counts,
                output: output, scores: scratch.scores!, rows: rows, keys: keys, stride: stride)
        } else {
            let stripe = scratch.scores!
            let softmax = keys <= 4096 ? softmaxShort : softmaxLong
            let softmaxThreads = keys <= 4096 ? ((keys + 127) / 128) * 32 : 1024
            for rowBase in Swift.stride(from: 0, to: rows, by: 8) {
                let stripeRows = min(8, rows - rowBase)
                for head in 0..<24 {
                    let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(rowBase), UInt32(stripeRows), UInt32(head)]
                    try encode(
                        command, qk, [query, key, ids, counts, stripe], shape,
                        grid: MTLSize(width: (keys + 7) / 8, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
                    try encode(
                        command, softmax, [stripe], shape,
                        grid: MTLSize(width: stripeRows, height: 1, depth: 1), group: MTLSize(width: softmaxThreads, height: 1, depth: 1))
                    try encode(
                        command, pv, [stripe, value, output], shape,
                        grid: MTLSize(width: 32, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
                }
            }
        }
    }

    private func batchedScratchBytes(rows: Int, keys: Int) throws -> Int {
        if fusedPrefill { return try CompactPrefillReuse.layout(rows: rows, keys: keys).bytes }
        return try keys >= CompactSelectedAttention.minimumKeys
            ? CompactSelectedAttention.layout(rows: rows).bytes
            : Self.unfusedBatchPlan(queries: rows, keys: keys).scoreScratchBytes
    }

    private func encodeBatched(
        _ command: MTLCommandBuffer, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
        ids: MTLBuffer, counts: MTLBuffer, output: MTLBuffer, scores: MTLBuffer,
        rows: Int, keys: Int, stride: Int
    ) throws {
        if fusedPrefill {
            try encodeCompactReuse(
                command, query: query, key: key, value: value, ids: ids, counts: counts,
                output: output, scores: scores, rows: rows, keys: keys, stride: stride)
            compactReuseCalls += 1
            return
        }
        if rows == 256 && (256...7680).contains(keys) {
            try encodePrefillReuse(
                command, query: query, key: key, value: value, ids: ids, counts: counts,
                output: output, scores: scores, rows: rows, keys: keys, stride: stride)
            prefillReuseCount += 1
            return
        }
        // Outside the prefill-reuse geometry (ragged rows, keys above 7680 without fused prefill) the
        // direct stripe PV has no checked form: these kernels stay, and the call is counted as a fallback.
        if qsaExactKernels { exactFallbackCalls += 1 }
        guard keys >= CompactSelectedAttention.minimumKeys else {
            try encodeOriginalBatched(
                command, query: query, key: key, value: value, ids: ids, counts: counts,
                output: output, scores: scores, rows: rows, keys: keys, stride: stride)
            return
        }
        let layout = try CompactSelectedAttention.layout(rows: rows)
        let shape = [UInt32(rows), UInt32(keys), UInt32(stride)]
        let one = MTLSize(width: 32, height: 1, depth: 1)
        try encode(
            command, compactPrepare, [ids, counts, scores], shape,
            grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        try encode(
            command, compactQK, [query, key, ids, counts, scores, scores], shape,
            grid: MTLSize(width: Self.maximumSelectedKeys, height: rows, depth: 4), group: one,
            offsets: [0, 0, 0, 0, 0, layout.probabilitiesOffset])
        try encode(
            command, compactSoftmax, [scores, ids, counts, scores, scores], shape,
            grid: MTLSize(width: rows, height: 24, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1),
            offsets: [layout.probabilitiesOffset, 0, 0, 0, layout.defaultsOffset])
        try encode(
            command, compactPV, [scores, value, ids, counts, scores, scores, output], shape,
            grid: MTLSize(width: 32, height: ((rows + 1) / 2) * 6, depth: 1), group: one,
            offsets: [layout.probabilitiesOffset, 0, 0, 0, 0, layout.defaultsOffset, 0])
    }

    // Shared by production and the actual maximum packed-union fixture.
    package func encodeCompactUnion(
        _ command: MTLCommandBuffer, ids: MTLBuffer, counts: MTLBuffer,
        scores: MTLBuffer, rows: Int, keys: Int, stride: Int
    ) throws -> CompactPrefillReuse.Layout {
        let layout = try CompactPrefillReuse.layout(rows: rows, keys: keys)
        guard fusedPrefill, stride > 0 && stride <= Self.maximumSelectedKeys,
            scores.length >= layout.bytes, ids.length >= rows * stride * 4, counts.length >= rows * 4
        else {
            throw EngineError("Compact union inputs exceed their admitted owner")
        }
        let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(layout.unionWordsPerGroup), UInt32(layout.maximumUnionTiles)]
        try encode(
            command, compactPrepare, [ids, counts, scores], shape,
            grid: .init(width: rows, height: 1, depth: 1), group: .init(width: 256, height: 1, depth: 1))
        try encode(
            command, compactUnionPrepare!, [scores, scores, scores, fusedPrefillCounters!], shape,
            grid: .init(width: (rows + 7) / 8, height: 1, depth: 1), group: .init(width: 256, height: 1, depth: 1),
            offsets: [0, layout.unionIDsOffset, layout.unionCountsOffset, 0])
        return layout
    }

    private func encodeCompactReuse(
        _ command: MTLCommandBuffer, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
        ids: MTLBuffer, counts: MTLBuffer, output: MTLBuffer, scores: MTLBuffer,
        rows: Int, keys: Int, stride: Int
    ) throws {
        let layout = try encodeCompactUnion(command, ids: ids, counts: counts, scores: scores, rows: rows, keys: keys, stride: stride)
        let compact = layout.compact
        let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(layout.unionWordsPerGroup), UInt32(layout.maximumUnionTiles)]
        // `qsa_exact_kernels`: the same dispatch grids and buffers; the double-buffered QK keeps 384 threads,
        // the direct PV runs one 128-thread group per 8-row unit (one SIMD group per 64 value columns).
        let exact = qsaExactKernels ? try exactCompactPipelines() : nil
        let qkGrid = MTLSize(width: (layout.maximumUnionTiles + 15) / 16, height: (rows + 7) / 8, depth: 2)
        try encode(
            command, exact?.qk ?? compactReuseQK!, [query, key, ids, counts, scores, scores, fusedPrefillCounters!, scores, scores], shape,
            grid: qkGrid, group: .init(width: 384, height: 1, depth: 1),
            offsets: [0, 0, 0, 0, 0, compact.probabilitiesOffset, 0, layout.unionIDsOffset, layout.unionCountsOffset])
        if exact != nil { exactQKGroups += UInt64(qkGrid.width * qkGrid.height * qkGrid.depth) }
        if diagnosticCompactReuseQKOnly { return }
        try encode(
            command, compactReuseSoftmax!, [scores, ids, counts, scores, scores, fusedPrefillCounters!], shape,
            grid: .init(width: rows, height: 24, depth: 1), group: .init(width: 1024, height: 1, depth: 1),
            offsets: [compact.probabilitiesOffset, 0, 0, 0, compact.defaultsOffset, 0])
        let pvPipeline: MTLComputePipelineState
        let pvGroups: Int
        let pvThreads: Int
        if let exact {
            pvPipeline = exact.pv
            pvGroups = ((rows + 1) / 2) * 6
            pvThreads = 128
        } else {
            pvPipeline = compactPVGrouped ? compactReusePVGrouped! : compactReusePV!
            pvGroups = ((rows + 1) / 2) * (compactPVGrouped ? 2 : 6)
            pvThreads = compactPVGrouped ? 768 : 256
        }
        try encode(
            command, pvPipeline, [scores, value, ids, counts, scores, scores, output, fusedPrefillCounters!], shape,
            grid: .init(width: 1, height: pvGroups, depth: 1), group: .init(width: pvThreads, height: 1, depth: 1),
            offsets: [compact.probabilitiesOffset, 0, 0, 0, 0, compact.defaultsOffset, 0, 0])
        if exact != nil { exactPVUnits += UInt64(pvGroups) }
    }

    /// Fixed C256 composition. Align existing score stripes to eight queries;
    /// retain their original unit addresses and the unchanged softmax kernel.
    private func encodePrefillReuse(
        _ command: MTLCommandBuffer, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
        ids: MTLBuffer, counts: MTLBuffer, output: MTLBuffer, scores: MTLBuffer,
        rows: Int, keys: Int, stride: Int
    ) throws {
        let plan = try Self.unfusedBatchPlan(queries: rows, keys: keys)
        let perBatch = (plan.unitsPerBatch / 24) * 24
        guard rows == 256, (256...7680).contains(keys), perBatch >= 24,
            scores.length >= plan.scoreScratchBytes
        else { throw EngineError("Invalid prefill reuse stripe geometry") }
        let softmax = keys <= 4096 ? batchedSoftmaxShort : batchedSoftmaxLong
        let softmaxThreads = keys <= 4096 ? ((keys + 127) / 128) * 32 : 1024
        // `qsa_exact_kernels`: the direct stripe PV replaces the full-width PV (same grid, 128 threads).
        let exactPV = qsaExactKernels ? try exactStripePipeline() : nil
        for firstUnit in Swift.stride(from: 0, to: plan.units, by: perBatch) {
            let count = min(perBatch, plan.units - firstUnit)
            let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(firstUnit), UInt32(count)]
            try encode(
                command, prefillQK, [query, key, ids, counts, scores], shape,
                grid: MTLSize(width: (keys + 127) / 128, height: (count + 23) / 24, depth: 2),
                group: MTLSize(width: 384, height: 1, depth: 1))
            try encode(
                command, softmax, [scores], shape,
                grid: MTLSize(width: 8, height: count, depth: 1),
                group: MTLSize(width: softmaxThreads, height: 1, depth: 1))
            try encode(
                command, exactPV ?? prefillPV, [scores, value, output], shape,
                grid: MTLSize(width: 1, height: count, depth: 1),
                group: MTLSize(width: exactPV != nil ? 128 : 256, height: 1, depth: 1))
            if exactPV != nil { exactPVUnits += UInt64(count) }
        }
    }

    // Active smaller-domain schedule; also the independent original-batched
    // control for the compact primitive's query-pair exceptional semantics.
    package func encodeOriginalBatched(
        _ command: MTLCommandBuffer, query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
        ids: MTLBuffer, counts: MTLBuffer, output: MTLBuffer, scores: MTLBuffer,
        rows: Int, keys: Int, stride: Int
    ) throws {
        let plan = try Self.unfusedBatchPlan(queries: rows, keys: keys)
        let softmax = keys <= 4096 ? batchedSoftmaxShort : batchedSoftmaxLong
        let softmaxThreads = keys <= 4096 ? ((keys + 127) / 128) * 32 : 1024
        for firstUnit in Swift.stride(from: 0, to: plan.units, by: plan.unitsPerBatch) {
            let count = min(plan.unitsPerBatch, plan.units - firstUnit)
            let shape = [UInt32(rows), UInt32(keys), UInt32(stride), UInt32(firstUnit), UInt32(count)]
            try encode(
                command, batchedQK, [query, key, ids, counts, scores], shape,
                grid: MTLSize(width: (keys + 7) / 8, height: count, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
            try encode(
                command, softmax, [scores], shape,
                grid: MTLSize(width: 8, height: count, depth: 1), group: MTLSize(width: softmaxThreads, height: 1, depth: 1))
            try encode(
                command, batchedPV, [scores, value, output], shape,
                grid: MTLSize(width: 32, height: count, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    public func makeDecodeStorage(capacityTokens: Int) throws -> QSADecodeStorage {
        guard (1...ModelContract.maximumContext).contains(capacityTokens) else { throw EngineError("Invalid decode mirror capacity") }
        let keys = try allocate(bytes: capacityTokens * 512 * 2)
        let values = try allocate(bytes: capacityTokens * 512 * 2)
        return try QSADecodeStorage(capacityTokens: capacityTokens, keys: keys, values: values)
    }

    private func allocate(bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength else {
            throw EngineError("Pinned attention buffer exceeds device allocation limit")
        }
        let page = UInt64(getpagesize())
        let charged = (UInt64(bytes) + page - 1) / page * page
        let owner = try PinnedAttentionAllocation(ledger: ledger, bytes: charged)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Pinned attention buffer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &PinnedAttentionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard UInt64(buffer.allocatedSize) <= charged else { throw EngineError("Pinned attention allocation exceeded page reservation") }
        return buffer
    }
    private func concatenateBuffer(_ a: [UInt16], _ b: [UInt16]) throws -> MTLBuffer {
        let buffer = try allocate(bytes: (a.count + b.count) * 2)
        a.withUnsafeBytes { if !$0.isEmpty { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
        b.withUnsafeBytes {
            if !$0.isEmpty { buffer.contents().advanced(by: a.count * 2).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        return buffer
    }
    private func encode(
        _ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
        _ buffers: [MTLBuffer], _ shape: [UInt32], grid: MTLSize, group: MTLSize, offsets: [Int]? = nil
    ) throws {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Pinned attention encoder allocation failed", code: "insufficient_memory")
        }
        encoder.label = "attention/" + (pipeline.label ?? "unlabelled")
        encoder.setComputePipelineState(pipeline)
        if let indirectResources { encoder.useResources(indirectResources, usage: .read) }
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: offsets?[index] ?? 0, index: index) }
        if !shape.isEmpty { shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) } }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }

    private static let source = #"""
        #include <metal_stdlib>
        #include <metal_simdgroup>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        // BF16 helpers: exact widening, round-to-nearest-even narrowing with NaN quieted (as MLX's bfloat16).
        inline float load_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
        inline ushort store_bf16(float x) {
            uint b = as_type<uint>(x);
            if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) return ushort((b >> 16) | 0x0040u);
            return ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
        }
        inline float round_bf16(float x) { return load_bf16(store_bf16(x)); }
        // Membership of `key` in one row's sorted selection (binary search).
        inline bool selected(device const uint *ids, uint count, uint key) {
            uint lo = 0, hi = count;
            while (lo < hi) { uint middle = (lo + hi) / 2; if (ids[middle] < key) lo = middle + 1; else hi = middle; }
            return lo < count && ids[lo] == key;
        }

        // Adapted from pinned MLX sdpa_vector.h. Lane assignment, online updates,
        // reduction trees and BF16 boundaries are retained; addresses use the native
        // token-major K/V layout. No sinks or additive masks exist in this model.
        kernel void origami_sdpa_vector(device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], ORIGAMI_KV_ARGS values [[buffer(2)]],
            device const uint *ids [[buffer(3)]], device const uint *counts [[buffer(4)]],
            device float *out [[buffer(5)]], constant uint *shape [[buffer(6)]],
            uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]]) {
            uint head = group.x, row = group.y, kv = head / 12, n = shape[1];
            float q[8], o[8] = {0};
            threadgroup float outputs[1024], maxima[32], sums[32];
            for (uint j = 0; j < 8; ++j) q[j] = 0.0625f * load_bf16(queries[(row * 24 + head) * 256 + lane * 8 + j]);
            float maximum = -FLT_MAX, sum = 0;
            for (uint key = sg; key < n; key += 32) {
                if (selected(ids + row * shape[2], counts[row], key)) {
                    float score = 0;
                    for (uint j = 0; j < 8; ++j) score += q[j] * load_bf16(origami_key_bits(keys, key * 512 + kv * 256 + lane * 8 + j));
                    score = simd_sum(score);
                    float next = max(maximum, score), factor = fast::exp(maximum - next), e = fast::exp(score - next);
                    maximum = next; sum = sum * factor + e;
                    for (uint j = 0; j < 8; ++j) o[j] = o[j] * factor + e * load_bf16(origami_value_bits(values, key * 512 + kv * 256 + lane * 8 + j));
                }
            }
            if (lane == 0) { maxima[sg] = maximum; sums[sg] = sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum = maxima[lane];
            float next = simd_max(maximum), factor = fast::exp(maximum - next);
            sum = simd_sum(sums[lane] * factor);
            for (uint j = 0; j < 8; ++j) {
                outputs[lane * 32 + sg] = o[j];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[j] = simd_sum(outputs[sg * 32 + lane] * factor);
                o[j] = sum == 0 ? o[j] : o[j] / sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) for (uint j = 0; j < 8; ++j) out[(row * 24 + head) * 256 + sg * 8 + j] = round_bf16(o[j]);
        }

        // MLX sdpa_vector_2pass_1: grid (2 KV heads, 1, P partitions), threadgroup (32 lanes, 12 heads,
        // rows). Partition `block` visits keys block, block+P, ... in ascending order (selected keys only),
        // with the same online softmax as origami_sdpa_vector. Writes BF16 partial outputs and FP32 sum/max.
        kernel void origami_sdpa_partial(device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], ORIGAMI_KV_ARGS values [[buffer(2)]],
            device const uint *ids [[buffer(3)]], device const uint *counts [[buffer(4)]],
            device ushort *out [[buffer(5)]], device float *sums [[buffer(6)]], device float *maxima [[buffer(7)]],
            constant uint *shape [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
            uint3 local [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            uint kv = group.x, block = group.z, row = local.z, head = kv * 12 + local.y, blocks = shape[3];
            float q[8], o[8] = {0};
            for (uint j = 0; j < 8; ++j) q[j] = 0.0625f * load_bf16(queries[(row * 24 + head) * 256 + lane * 8 + j]);
            float maximum = -FLT_MAX, sum = 0;
            for (uint key = block; key < shape[1]; key += blocks) {
                if (selected(ids + row * shape[2], counts[row], key)) {
                    float score = 0;
                    for (uint j = 0; j < 8; ++j) score += q[j] * load_bf16(origami_key_bits(keys, key * 512 + kv * 256 + lane * 8 + j));
                    score = simd_sum(score);
                    float next = max(maximum, score), factor = fast::exp(maximum - next), e = fast::exp(score - next);
                    maximum = next; sum = sum * factor + e;
                    for (uint j = 0; j < 8; ++j) o[j] = o[j] * factor + e * load_bf16(origami_value_bits(values, key * 512 + kv * 256 + lane * 8 + j));
                }
            }
            uint partial = (head * shape[0] + row) * blocks + block;
            if (lane == 0) { sums[partial] = sum; maxima[partial] = maximum; }
            for (uint j = 0; j < 8; ++j) out[partial * 256 + lane * 8 + j] = store_bf16(o[j]);
        }

        // One writer per row. No atomics, sorting or changed reduction order:
        // each partition receives sorted absolute keys in its original traversal.
        kernel void origami_sdpa_partition_positions(device const uint *ids [[buffer(0)]],
            device const uint *counts [[buffer(1)]], device uint *buckets [[buffer(2)]],
            constant uint *shape [[buffer(3)]], uint row [[thread_position_in_grid]]) {
            uint blocks = shape[3], stride = (shape[1] + blocks - 1) / blocks;
            uint base = row * blocks * (stride + 1);
            for (uint block = 0; block < blocks; ++block) buckets[base + block] = 0;
            uint previous = 0xffffffffu;
            for (uint i = 0; i < min(counts[row], shape[2]); ++i) {
                uint key = ids[row * shape[2] + i];
                // A malformed device selection is flagged by its producer and
                // repaired by the transaction. Keep its dependent work in bounds.
                if (key >= shape[1] || key == previous) continue;
                previous = key;
                uint block = key % blocks, index = buckets[base + block];
                if (index < stride) {
                    buckets[base + block] = index + 1;
                    buckets[base + blocks + block * stride + index] = key;
                }
            }
        }

        // origami_sdpa_partial over the buckets built above: the same keys in the same order, without testing
        // every key of the domain. BITWISE equal to origami_sdpa_partial (medium, P >= 256).
        kernel void origami_sdpa_selected_partial(device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], ORIGAMI_KV_ARGS values [[buffer(2)]],
            device const uint *ids [[buffer(3)]], device const uint *counts [[buffer(4)]],
            device ushort *out [[buffer(5)]], device float *sums [[buffer(6)]], device float *maxima [[buffer(7)]],
            constant uint *shape [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
            uint3 local [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            uint kv = group.x, block = group.z, row = local.z, head = kv * 12 + local.y, blocks = shape[3];
            float q[8], o[8] = {0};
            for (uint j = 0; j < 8; ++j) q[j] = 0.0625f * load_bf16(queries[(row * 24 + head) * 256 + lane * 8 + j]);
            float maximum = -FLT_MAX, sum = 0;
            uint bucketStride = (shape[1] + blocks - 1) / blocks;
            uint base = row * blocks * (bucketStride + 1), count = ids[base + block];
            for (uint index = 0; index < count; ++index) {
                uint key = ids[base + blocks + block * bucketStride + index];
                {
                    float score = 0;
                    for (uint j = 0; j < 8; ++j) score += q[j] * load_bf16(origami_key_bits(keys, key * 512 + kv * 256 + lane * 8 + j));
                    score = simd_sum(score);
                    float next = max(maximum, score), factor = fast::exp(maximum - next), e = fast::exp(score - next);
                    maximum = next; sum = sum * factor + e;
                    for (uint j = 0; j < 8; ++j) o[j] = o[j] * factor + e * load_bf16(origami_value_bits(values, key * 512 + kv * 256 + lane * 8 + j));
                }
            }
            uint partial = (head * shape[0] + row) * blocks + block;
            if (lane == 0) { sums[partial] = sum; maxima[partial] = maximum; }
            for (uint j = 0; j < 8; ++j) out[partial * 256 + lane * 8 + j] = store_bf16(o[j]);
        }

        // MLX sdpa_vector_2pass_2: one 1024-thread group per (head, row). Global max and rescaled sum over
        // the P partials (lane-strided, then simd), outputs rescaled per partition (SIMD-group-strided),
        // transposed through threadgroup memory, divided by the sum (left as is when it is zero), BF16-rounded.
        kernel void origami_sdpa_finish(device const ushort *partials [[buffer(0)]],
            device const float *sums [[buffer(1)]], device const float *maxima [[buffer(2)]],
            device float *out [[buffer(3)]], constant uint *shape [[buffer(4)]],
            uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]]) {
            uint head = group.x, row = group.y, blocks = shape[3], base = (head * shape[0] + row) * blocks;
            float o[8] = {0}; threadgroup float outputs[1024];
            float maximum = -FLT_MAX, sum = 0;
            for (uint b = 0; b < blocks / 32; ++b) maximum = max(maximum, maxima[base + lane + 32 * b]);
            maximum = simd_max(maximum);
            for (uint b = 0; b < blocks / 32; ++b) sum += fast::exp(maxima[base + lane + 32 * b] - maximum) * sums[base + lane + 32 * b];
            sum = simd_sum(sum);
            for (uint b = 0; b < blocks / 32; ++b) {
                float factor = fast::exp(maxima[base + sg + 32 * b] - maximum);
                for (uint j = 0; j < 8; ++j) o[j] += factor * load_bf16(partials[(base + sg + 32 * b) * 256 + lane * 8 + j]);
            }
            for (uint j = 0; j < 8; ++j) {
                outputs[lane * 32 + sg] = o[j]; threadgroup_barrier(mem_flags::mem_threadgroup);
                o[j] = simd_sum(outputs[sg * 32 + lane]); o[j] = sum == 0 ? o[j] : o[j] / sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) for (uint j = 0; j < 8; ++j) out[(row * 24 + head) * 256 + sg * 8 + j] = round_bf16(o[j]);
        }

        // Steel GEMM's float-accumulator 8x8 MMA fragments, advancing K by eight.
        // BF16 Q scale, BF16 scores, and BF16 PV output are distinct boundaries.
        kernel void origami_qk_mma(device const ushort *queries [[buffer(0)]], ORIGAMI_KV_ARGS keys [[buffer(1)]],
            device const uint *ids [[buffer(2)]], device const uint *counts [[buffer(3)]], device ushort *scores [[buffer(4)]],
            constant uint *shape [[buffer(5)]], uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint n = shape[1], rowBase = shape[3], rows = shape[4], head = shape[5], keyBase = block * 8;
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 256; d += 8) {
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    as[index] = row < rows ? round_bf16(0.0625f * load_bf16(queries[((rowBase + row) * 24 + head) * 256 + d + col])) : 0.0f;
                    bs[index] = keyBase + col < n ? load_bf16(origami_key_bits(keys, (keyBase + col) * 512 + (head / 12) * 256 + d + row)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) {
                uint row = index / 8, key = keyBase + index % 8;
                if (row < rows && key < n) scores[row * n + key] = selected(ids + (rowBase + row) * shape[2], counts[rowBase + row], key) ? store_bf16(cs[index]) : ushort(0xff7f);
            }
        }

        // MLX softmax for rows of at most 4096: ceil(n/128)*32 threads, four elements each; max and sum
        // reduce through simd then threadgroup; in-place BF16 probabilities. Masked scores (0xff7f, the
        // lowest finite BF16) underflow to zero unless the whole row is masked, which gives a uniform row.
        kernel void origami_softmax_short(device ushort *scores [[buffer(0)]], constant uint *shape [[buffer(1)]],
            uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], offset = row * n + tid * 4;
            threadgroup float maxima[32], sums[32]; float x[4];
            for (uint i = 0; i < 4; ++i) x[i] = tid * 4 + i < n ? load_bf16(scores[offset + i]) : -INFINITY;
            if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maximum = -FLT_MAX;
            for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
            maximum = simd_max(maximum); if (lane == 0) maxima[sg] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) { maximum = simd_max(maxima[lane]); if (lane == 0) maxima[0] = maximum; }
            threadgroup_barrier(mem_flags::mem_threadgroup); maximum = maxima[0];
            float sum = 0; for (uint i = 0; i < 4; ++i) { x[i] = fast::exp(x[i] - maximum); sum += x[i]; }
            sum = simd_sum(sum); if (lane == 0) sums[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) { sum = simd_sum(sums[lane]); if (lane == 0) sums[0] = sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup); float inverse = 1.0f / sums[0];
            for (uint i = 0; i < 4; ++i) if (tid * 4 + i < n) scores[offset + i] = store_bf16(x[i] * inverse);
        }

        // MLX looped softmax for longer rows: 1024 threads, 4096-element rounds with an online max/sum per
        // thread, then simd and threadgroup reductions; a second pass writes BF16 probabilities in place.
        kernel void origami_softmax_long(device ushort *scores [[buffer(0)]], constant uint *shape [[buffer(1)]],
            uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], base = row * n; threadgroup float maxima[32], sums[32];
            float previous, maximum = -FLT_MAX, sum = 0;
            for (uint r = 0; r < (n + 4095) / 4096; ++r) {
                uint offset = r * 4096 + tid * 4; float x[4];
                for (uint i = 0; i < 4; ++i) x[i] = offset + i < n ? load_bf16(scores[base + offset + i]) : -INFINITY;
                previous = maximum;
                for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
                sum *= fast::exp(previous - maximum);
                for (uint i = 0; i < 4; ++i) sum += fast::exp(x[i] - maximum);
            }
            previous = maximum; maximum = simd_max(maximum); sum *= fast::exp(previous - maximum); sum = simd_sum(sum);
            previous = maximum; if (lane == 0) maxima[sg] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum = simd_max(maxima[lane]); sum *= fast::exp(previous - maximum);
            if (lane == 0) sums[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup); sum = simd_sum(sums[lane]); float inverse = 1.0f / sum;
            // All source reads used for the global reduction finish before in-place stores.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < (n + 4095) / 4096; ++r) {
                uint offset = r * 4096 + tid * 4;
                for (uint i = 0; i < 4; ++i) if (offset + i < n) scores[base + offset + i] = store_bf16(fast::exp(load_bf16(scores[base + offset + i]) - maximum) * inverse);
            }
        }

        // Unbatched P x V: one SIMD group per 8 value columns of an 8-row stripe and head; FP32 8x8 MMA over
        // keys, BF16-rounded output.
        kernel void origami_pv_mma(device const ushort *probabilities [[buffer(0)]], ORIGAMI_KV_ARGS values [[buffer(1)]],
            device float *out [[buffer(2)]], constant uint *shape [[buffer(3)]],
            uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint n = shape[1], rowBase = shape[3], rows = shape[4], head = shape[5], column = block * 8;
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            // The dispatched Steel GEMM wrapper reduces a partial BK=16 tile
            // FIRST, then the aligned prefix. Preserve both 8-wide MMAs in that
            // partial tile, including its zero padding, and their FP32 order.
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16;
            for (uint step = 0; step < alignedKeys + tailSteps; step += 8) {
                uint key = step < tailSteps ? alignedKeys + step : step - tailSteps;
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    as[index] = row < rows && key + col < n ? load_bf16(probabilities[row * n + key + col]) : 0.0f;
                    bs[index] = key + row < n ? load_bf16(origami_value_bits(values, (key + row) * 512 + (head / 12) * 256 + column + col)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) if (index / 8 < rows) out[((rowBase + index / 8) * 24 + head) * 256 + column + index % 8] = round_bf16(cs[index]);
        }

        // Batched prefill groups two query tokens across four heads of one KV head.
        // Every logical output keeps its original MMA/reduction order; only
        // independent rows are regrouped so sparse blocks can be skipped together.
        kernel void origami_qk_mma_batched(device const ushort *queries [[buffer(0)]], ORIGAMI_KV_ARGS keys [[buffer(1)]],
            device const uint *ids [[buffer(2)]], device const uint *counts [[buffer(3)]], device ushort *scores [[buffer(4)]],
            constant uint *shape [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint n = shape[1], unit = shape[3] + group.y;
            uint rowBase = (unit / 6) * 2, rows = min(2u, shape[0] - rowBase) * 4, head = (unit % 6) * 4, keyBase = group.x * 8;
            scores += group.y * 8 * n;
            // All 32 lanes agree before bypassing an entirely masked 8-key tile.
            bool anySelected = false;
            for (uint index = lane; index < 64; index += 32) {
                uint row = index / 8, key = keyBase + index % 8;
                anySelected |= row < rows && key < n && selected(ids + (rowBase + row / 4) * shape[2], counts[rowBase + row / 4], key);
            }
            if (!simd_any(anySelected)) {
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, key = keyBase + index % 8;
                    if (row < rows && key < n) scores[row * n + key] = ushort(0xff7f);
                }
                return;
            }
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 256; d += 8) {
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    as[index] = row < rows ? round_bf16(0.0625f * load_bf16(queries[((rowBase + row / 4) * 24 + head + row % 4) * 256 + d + col])) : 0.0f;
                    bs[index] = keyBase + col < n ? load_bf16(origami_key_bits(keys, (keyBase + col) * 512 + (head / 12) * 256 + d + row)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) {
                uint row = index / 8, key = keyBase + index % 8;
                if (row < rows && key < n) scores[row * n + key] = selected(ids + (rowBase + row / 4) * shape[2], counts[rowBase + row / 4], key) ? store_bf16(cs[index]) : ushort(0xff7f);
            }
        }

        // origami_softmax_short on the batched layout: group (row of the unit, unit); rows past a ragged final
        // query pair return early.
        kernel void origami_softmax_short_batched(device ushort *scores [[buffer(0)]], constant uint *shape [[buffer(1)]],
            uint3 group [[threadgroup_position_in_grid]], uint3 localID [[thread_position_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint tid = localID.x;
            uint n = shape[1], unit = shape[3] + group.y, row = group.x;
            uint rowBase = (unit / 6) * 2;
            if (row >= min(2u, shape[0] - rowBase) * 4) return;
            scores += group.y * 8 * n;
            uint offset = row * n + tid * 4;
            threadgroup float maxima[32], sums[32]; float x[4];
            for (uint i = 0; i < 4; ++i) x[i] = tid * 4 + i < n ? load_bf16(scores[offset + i]) : -INFINITY;
            if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maximum = -FLT_MAX;
            for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
            maximum = simd_max(maximum); if (lane == 0) maxima[sg] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) { maximum = simd_max(maxima[lane]); if (lane == 0) maxima[0] = maximum; }
            threadgroup_barrier(mem_flags::mem_threadgroup); maximum = maxima[0];
            float sum = 0; for (uint i = 0; i < 4; ++i) { x[i] = fast::exp(x[i] - maximum); sum += x[i]; }
            sum = simd_sum(sum); if (lane == 0) sums[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) { sum = simd_sum(sums[lane]); if (lane == 0) sums[0] = sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup); float inverse = 1.0f / sums[0];
            for (uint i = 0; i < 4; ++i) if (tid * 4 + i < n) scores[offset + i] = store_bf16(x[i] * inverse);
        }

        // origami_softmax_long on the batched layout.
        kernel void origami_softmax_long_batched(device ushort *scores [[buffer(0)]], constant uint *shape [[buffer(1)]],
            uint3 group [[threadgroup_position_in_grid]], uint3 localID [[thread_position_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint tid = localID.x;
            uint n = shape[1], unit = shape[3] + group.y, row = group.x;
            uint rowBase = (unit / 6) * 2;
            if (row >= min(2u, shape[0] - rowBase) * 4) return;
            scores += group.y * 8 * n;
            uint base = row * n; threadgroup float maxima[32], sums[32];
            float previous, maximum = -FLT_MAX, sum = 0;
            for (uint r = 0; r < (n + 4095) / 4096; ++r) {
                uint offset = r * 4096 + tid * 4; float x[4];
                for (uint i = 0; i < 4; ++i) x[i] = offset + i < n ? load_bf16(scores[base + offset + i]) : -INFINITY;
                previous = maximum;
                for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
                sum *= fast::exp(previous - maximum);
                for (uint i = 0; i < 4; ++i) sum += fast::exp(x[i] - maximum);
            }
            previous = maximum; maximum = simd_max(maximum); sum *= fast::exp(previous - maximum); sum = simd_sum(sum);
            previous = maximum; if (lane == 0) maxima[sg] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum = simd_max(maxima[lane]); sum *= fast::exp(previous - maximum);
            if (lane == 0) sums[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup); sum = simd_sum(sums[lane]); float inverse = 1.0f / sum;
            // All source reads used for the global reduction finish before in-place stores.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < (n + 4095) / 4096; ++r) {
                uint offset = r * 4096 + tid * 4;
                for (uint i = 0; i < 4; ++i) if (offset + i < n) scores[base + offset + i] = store_bf16(fast::exp(load_bf16(scores[base + offset + i]) - maximum) * inverse);
            }
        }

        // origami_pv_mma on the batched layout (unit = 2 queries x 4 heads); an 8-key tile whose 64
        // probabilities are all zero is skipped, the surviving MMAs keep their absolute order.
        kernel void origami_pv_mma_batched(device const ushort *probabilities [[buffer(0)]], ORIGAMI_KV_ARGS values [[buffer(1)]],
            device float *out [[buffer(2)]], constant uint *shape [[buffer(3)]],
            uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint n = shape[1], unit = shape[3] + group.y;
            uint rowBase = (unit / 6) * 2, rows = min(2u, shape[0] - rowBase) * 4, head = (unit % 6) * 4, column = group.x * 8;
            probabilities += group.y * 8 * n;
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            // The dispatched Steel GEMM wrapper reduces a partial BK=16 tile
            // FIRST, then the aligned prefix. Preserve both 8-wide MMAs in that
            // partial tile, including its zero padding, and their FP32 order.
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16;
            for (uint step = 0; step < alignedKeys + tailSteps; step += 8) {
                uint key = step < tailSteps ? alignedKeys + step : step - tailSteps;
                bool anyProbability = false;
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    float probability = row < rows && key + col < n ? load_bf16(probabilities[row * n + key + col]) : 0.0f;
                    as[index] = probability;
                    anyProbability |= probability != 0.0f;
                }
                // Keep every surviving 8-key MMA in its original absolute order.
                if (!simd_any(anyProbability)) continue;
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    bs[index] = key + row < n ? load_bf16(origami_value_bits(values, (key + row) * 512 + (head / 12) * 256 + column + col)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) if (index / 8 < rows) out[((rowBase + (index / 8) / 4) * 24 + head + (index / 8) % 4) * 256 + column + index % 8] = round_bf16(cs[index]);
        }
        """#
}
