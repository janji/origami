import Darwin
import Foundation
import Metal
import ObjectiveC

private final class FusedExpertAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64, category: String) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: category)
    }
    deinit { try? ledger.release(reservation) }
}
private enum FusedExpertAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The owner is serial, like FusedExpertExecution. A lease remains active through
/// its complete joined command AND all output callbacks. Each take returns a
/// distinct page; no buffer can alias another job or stage in the same scope.
private final class ExpertDecodeScratchPool {
    static let capacity = 40  // Eight jobs, five private buffers per decode job.
    let pageBytes = Int(getpagesize())
    private let metadata: FusedExpertAllocation
    private let allocate: (Int) throws -> MTLBuffer
    private var buffers: [MTLBuffer] = []
    private(set) var active = false
    private(set) var leased = 0
    private(set) var hits: UInt64 = 0, misses: UInt64 = 0
    var retained: Int { buffers.count }
    init(ledger: MemoryLedger, allocate: @escaping (Int) throws -> MTLBuffer) throws {
        guard getpagesize() >= 10_240 else { throw EngineError("Expert decode scratch page cannot fit its bounded output") }
        metadata = try FusedExpertAllocation(ledger: ledger, bytes: UInt64(getpagesize()), category: "fused-expert-scratch-metadata")
        self.allocate = allocate
        buffers.reserveCapacity(Self.capacity)
    }
    func withScope(_ body: (ExpertDecodeScratchPool) throws -> Void) throws {
        guard !active else { throw EngineError("Expert scratch scope is already active", code: "expert_scratch_busy") }
        active = true
        leased = 0
        let previousCount = buffers.count
        defer {
            active = false
            leased = 0
        }
        do { try body(self) } catch {
            // The caller unwinds only after its autoreleasepool has joined or
            // discarded an unsubmitted command. Retire newly admitted pages on
            // failure; previously warm pages remain valid, idle and charged.
            if buffers.count > previousCount { buffers.removeSubrange(previousCount..<buffers.count) }
            throw error
        }
    }
    func take(bytes: Int) throws -> MTLBuffer {
        guard active, bytes > 0, bytes <= pageBytes, leased < Self.capacity else {
            throw EngineError("Invalid expert scratch lease geometry")
        }
        let index = leased
        if index < buffers.count {
            hits += 1
        } else {
            let buffer = try allocate(pageBytes)  // Admission errors always propagate.
            buffers.append(buffer)
            misses += 1
        }
        leased += 1
        return buffers[index]
    }
    func trim() throws {
        guard !active else { throw EngineError("Cannot trim an active expert scratch scope", code: "expert_scratch_busy") }
        buffers.removeAll(keepingCapacity: true)
    }
}

/// Opt-in command batching for one to eight already pinned expert jobs. Gate, up,
/// table-SiLU/product, and down execute in one command and join exactly once.
/// Quantized kernels come directly from ReferenceKernels' source, not a copy:
/// cached experts keep the row-one QMV schedule for every input row, while
/// grouped prefill preserves the scalar/SIMD QMM no-split schedule.
///
/// Call inside ExpertDemandWindow.consume/consumeBatch (or valid consumer leases).
/// Neither a raw MTLBuffer nor ExpertBuffer alone pins a recyclable cache slot.
/// Every submitted command is joined before success OR cancellation returns,
/// so existing consume-scope retirement remains valid without API changes.
public final class FusedExpertExecution {
    public let unaryPolicy: String
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let qmvPipeline, multirowQMVPipeline, qmmPipeline, simdPipeline, productPipeline: MTLComputePipelineState
    private let tiledPipeline, tiled32Pipeline, mmaPipeline: MTLComputePipelineState?
    /// Gate and up of a prefetch-BM32 tile in one dispatch with the product (`qmm_gate_up_tiled32_prefetch_product`).
    private let gateUpPipeline: MTLComputePipelineState?
    package private(set) var gateUpFusedTiles: UInt64 = 0
    public let groupedTiled: Bool
    public let groupedBM32: Bool
    public let groupedBM32MinRows: Int
    /// Encoded projections, including commands cancelled before submission.
    /// Startup/diagnostic checks use this to reject a silent BM16 fallback.
    public private(set) var groupedBM32EncodedProjections: UInt64 = 0
    /// Grouped BM32 projections encoded with the load-scheduled prefetch kernel.
    public let groupedPrefetch: Bool
    public private(set) var groupedPrefetchEncodedProjections: UInt64 = 0
    /// `grouped_expert_mma_min_rows` (0 = off): grouped BM32 jobs with at least this many rows use the
    /// BM64 x BN64 bfloat-MMA kernel (expert N = 640 and 2560 are multiples of 64); jobs below it stay on
    /// the prefetch BM32 kernel and are counted separately.
    public let groupedMMAMinRows: Int
    public private(set) var groupedMMAEncodedProjections: UInt64 = 0
    public private(set) var groupedMMABelowMinRowsProjections: UInt64 = 0
    /// Grouped tiled jobs (counted on their gate projection) by row count in 16-row buckets: index
    /// rows / 16, so index 16 holds the full 256-row jobs. Measures the job-row mix the MMA floor
    /// (`grouped_expert_mma_min_rows`) acts on; no arithmetic depends on it.
    public private(set) var groupedTiledJobRows16 = [UInt64](repeating: 0, count: 17)
    private let scratchPool: ExpertDecodeScratchPool?
    private let silu: MTLBuffer
    private let ledger: MemoryLedger
    /// GGUF experts (`ModelLayout.ggufExperts`): every job and layer command runs on ggml's kernels instead.
    let gguf: GGUFExpertKernels?

    public init(
        arithmetic: LayerUnaryArithmetic, ledger: MemoryLedger, groupedTiled: Bool = false,
        groupedBM32: Bool = false, groupedBM32MinRows: Int = 128, scratchReuse: Bool = false,
        groupedPrefetch: Bool = false, groupedMMAMinRows: Int = 0
    ) throws {
        // The prefetch and MMA loaders read affine 4-bit words; NVFP4 takes the BM32 kernel.
        let affine4 = ModelLayout.experts.bits == 4 && !ModelLayout.experts.fp4
        let groupedPrefetch = groupedPrefetch && affine4
        let groupedMMAMinRows = affine4 ? groupedMMAMinRows : 0
        guard (1...256).contains(groupedBM32MinRows), !groupedBM32 || groupedTiled else {
            throw EngineError("Grouped BM32 requires grouped tiling and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard !groupedPrefetch || groupedBM32 else {
            throw EngineError("Grouped prefetch requires grouped BM32", code: "invalid_configuration")
        }
        guard groupedMMAMinRows == 0 || ((64...256).contains(groupedMMAMinRows) && groupedPrefetch) else {
            throw EngineError(
                "grouped_expert_mma_min_rows requires grouped prefetch BM32 and a floor in 64...256", code: "invalid_configuration")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Fused expert Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(
            source: ModelLayout.expertMetalDefines + ReferenceKernels.expertFusionMetalSource
                + (groupedTiled ? GroupedExpertKernel.source : "") + (groupedBM32 ? QuantizedTiled32Kernel.source : "")
                + (groupedPrefetch ? QuantizedTiled32PrefetchKernel.source : "")
                + ExpertMultirowQMV.source + Self.productSource,
            options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw EngineError("Missing fused expert kernel: \(name)") }
            return try device.makeComputePipelineState(function: function)
        }
        qmvPipeline = try pipeline("qmv_pinned_reference")
        multirowQMVPipeline = try pipeline("expert_qmv_rows8")
        guard multirowQMVPipeline.threadExecutionWidth == 32,
            multirowQMVPipeline.maxTotalThreadsPerThreadgroup >= 64
        else {
            throw EngineError("Shared-weight expert QMV exceeds the fixed SIMD geometry")
        }
        qmmPipeline = try pipeline("qmm_pinned_reference")
        simdPipeline = try pipeline("qmm_pinned_simd_partitions")
        productPipeline = try pipeline("expert_table_silu_product")
        tiledPipeline = groupedTiled ? try pipeline("qmm_grouped_tiled_16_32_32") : nil
        if let tiledPipeline {
            guard tiledPipeline.threadExecutionWidth == 32, tiledPipeline.maxTotalThreadsPerThreadgroup >= 64,
                tiledPipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Grouped tiled expert pipeline exceeds the 64-thread device contract")
            }
        }
        tiled32Pipeline =
            groupedBM32 ? try pipeline(groupedPrefetch ? "qmm_pinned_tiled32_prefetch_partitions" : "qmm_pinned_tiled32_partitions") : nil
        if let tiled32Pipeline {
            guard tiled32Pipeline.threadExecutionWidth == 32, tiled32Pipeline.maxTotalThreadsPerThreadgroup >= 128,
                tiled32Pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Grouped BM32 pipeline exceeds the 128-thread device contract")
            }
        }
        gateUpPipeline = groupedPrefetch ? try pipeline("qmm_gate_up_tiled32_prefetch_product") : nil
        if let gateUpPipeline {
            guard gateUpPipeline.threadExecutionWidth == 32, gateUpPipeline.maxTotalThreadsPerThreadgroup >= 128,
                gateUpPipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Grouped gate/up pipeline exceeds the 128-thread device contract")
            }
        }
        mmaPipeline =
            groupedMMAMinRows > 0
            ? try QuantizedTensorOpsKernel.pipeline(device: device, defines: ModelLayout.expertMetalDefines, options: options) : nil
        if let mmaPipeline {
            guard mmaPipeline.threadExecutionWidth == 32, mmaPipeline.maxTotalThreadsPerThreadgroup >= 128,
                mmaPipeline.staticThreadgroupMemoryLength == QuantizedMMABF16Kernel.threadgroupBytes,
                mmaPipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Grouped MMA pipeline exceeds the 128-thread device contract")
            }
        }
        self.groupedBM32 = groupedBM32
        self.groupedBM32MinRows = groupedBM32MinRows
        self.groupedPrefetch = groupedPrefetch
        self.groupedMMAMinRows = groupedMMAMinRows
        self.groupedTiled = groupedTiled
        self.device = device
        self.queue = queue
        self.ledger = ledger
        unaryPolicy = arithmetic.policy
        gguf = ModelLayout.ggufExperts == nil ? nil : try GGUFExpertKernels(device: device)
        scratchPool =
            scratchReuse
            ? try ExpertDecodeScratchPool(ledger: ledger) { bytes in
                try Self.allocate(device: device, ledger: ledger, bytes: bytes, category: "fused-expert-scratch-retained")
            } : nil
        silu = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 2, category: "fused-expert-unary-table")
        let table = silu.contents().assumingMemoryBound(to: UInt16.self)
        for bits in 0..<65_536 { table[bits] = BF16.encode(arithmetic.silu(BF16.decode(UInt16(bits)))) }
    }

    /// Each job owns one input tile and retains a record view. Its caller must
    /// separately own the record's cache pin until evaluateBatch returns.
    public struct Job {
        public let input: [Float]
        public let buffer: ExpertBuffer
        public let groupedExpert, simdMatrix: Bool
        public init(input: [Float], buffer: ExpertBuffer, groupedExpert: Bool = false, simdMatrix: Bool = false) {
            self.input = input
            self.buffer = buffer
            self.groupedExpert = groupedExpert
            self.simdMatrix = simdMatrix
        }
    }
    /// One expert's prefill rows gathered on the GPU from a PrefillMoEWorkspace (`slots` are its token-rank rows);
    /// the output is scattered back through the workspace map and never read by the host.
    public struct DeviceJob {
        public let expert: Int
        public let slots: [UInt32]
        public let buffer: ExpertBuffer
        public let groupedExpert, simdMatrix: Bool
        public init(expert: Int, slots: [UInt32], buffer: ExpertBuffer, groupedExpert: Bool, simdMatrix: Bool) {
            self.expert = expert
            self.slots = slots
            self.buffer = buffer
            self.groupedExpert = groupedExpert
            self.simdMatrix = simdMatrix
        }
    }
    /// Hits count leased warm pages; misses count successfully created pages,
    /// including pages later rolled back after cancellation/callback failure.
    public struct ScratchSnapshot: Codable, Sendable {
        public let enabled: Bool
        public let hits, misses, retainedBytes, capacityBytes: UInt64
        public let retainedBuffers, leasedBuffers: Int
        static let disabled = ScratchSnapshot(
            enabled: false, hits: 0, misses: 0, retainedBytes: 0,
            capacityBytes: 0, retainedBuffers: 0, leasedBuffers: 0)
    }
    /// Cumulative executor counters: commands, jobs, rows, GPU and host time, and the scratch pool.
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, submittedJobs, submittedRows, gpuTimedCommands: UInt64
        public let mixedTailCommands, mixedTailQMVTiles, mixedTailQMVRows: UInt64
        public let multirowQMVEncodedProjections: UInt64
        public let gpuSeconds, hostJoinSeconds: Double
        public let hostPrepareSeconds: Double?
        public let scratch: ScratchSnapshot
        private enum CodingKeys: String, CodingKey {
            case submittedCommands, completedCommands, submittedJobs, submittedRows, gpuTimedCommands
            case gpuSeconds, hostJoinSeconds, hostPrepareSeconds, scratch
            case mixedTailCommands, mixedTailQMVTiles, mixedTailQMVRows
            case multirowQMVEncodedProjections
        }
    }
    private var submittedCommands: UInt64 = 0, completedCommands: UInt64 = 0, submittedJobs: UInt64 = 0
    private var submittedRows: UInt64 = 0, gpuTimedCommands: UInt64 = 0
    private var mixedTailCommands: UInt64 = 0, mixedTailQMVTiles: UInt64 = 0, mixedTailQMVRows: UInt64 = 0
    private var multirowQMVEncodedProjections: UInt64 = 0
    private var gpuSeconds: Double = 0, hostJoinSeconds: Double = 0
    private var hostPrepareSeconds = 0.0
    /// Cumulative executor-lifetime totals; queried at request completion, never
    /// per expert. GPU duration is included only for valid Metal timestamps.
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submittedCommands, completedCommands: completedCommands, submittedJobs: submittedJobs,
            submittedRows: submittedRows, gpuTimedCommands: gpuTimedCommands,
            mixedTailCommands: mixedTailCommands, mixedTailQMVTiles: mixedTailQMVTiles, mixedTailQMVRows: mixedTailQMVRows,
            multirowQMVEncodedProjections: multirowQMVEncodedProjections,
            gpuSeconds: gpuSeconds, hostJoinSeconds: hostJoinSeconds, hostPrepareSeconds: hostPrepareSeconds,
            scratch: ScratchSnapshot(
                enabled: scratchPool != nil, hits: scratchPool?.hits ?? 0, misses: scratchPool?.misses ?? 0,
                retainedBytes: UInt64((scratchPool?.retained ?? 0) * (scratchPool?.pageBytes ?? 0)),
                capacityBytes: scratchPool == nil ? 0 : UInt64(ExpertDecodeScratchPool.capacity * Int(getpagesize())),
                retainedBuffers: scratchPool?.retained ?? 0, leasedBuffers: scratchPool?.leased ?? 0))
    }
    /// Call at a quiescent request boundary. Idle pages keep their actual ledger
    /// charge until this trim or helper destruction; busy trims fail unchanged.
    public func trimScratch() throws { try scratchPool?.trim() }
    private func withScratchScope(_ pool: ExpertDecodeScratchPool?, _ body: (ExpertDecodeScratchPool?) throws -> Void) throws {
        if let pool { try pool.withScope { try body($0) } } else { try body(nil) }
    }
    private struct GatheredInput {
        let workspace: PrefillMoEWorkspace
        let slots: [UInt32]
    }
    private struct RawJob {
        let input: [Float]
        let weights: MTLBuffer
        let recordOffset: Int
        let groupedExpert, simdMatrix: Bool
        var gathered: GatheredInput? = nil
        var rows: Int { gathered?.slots.count ?? input.count / 2560 }
    }
    private struct PreparedJob {
        let rows: Int
        let input, gate, up, product, output: MTLBuffer
        let gatherMap: MTLBuffer?
    }

    public func evaluate(
        _ x: [Float], buffer: ExpertBuffer, groupedExpert: Bool = false, simdMatrix: Bool = false,
        shouldContinue: () -> Bool = { true },
        capture: ((String, [Float]) -> Void)? = nil
    ) throws -> [Float] {
        try withExtendedLifetime(buffer) {
            try evaluate(
                x, weights: buffer.buffer, recordOffset: buffer.offset, groupedExpert: groupedExpert,
                simdMatrix: simdMatrix, shouldContinue: shouldContinue, capture: capture)
        }
    }

    /// Buffer-level entry point retains the same caller-owned pin requirement.
    /// `recordOffset` addresses one complete verified ExpertProjection layout.
    public func evaluate(
        _ x: [Float], weights: MTLBuffer, recordOffset: Int,
        groupedExpert: Bool = false, simdMatrix: Bool = false,
        shouldContinue: () -> Bool = { true },
        capture: ((String, [Float]) -> Void)? = nil
    ) throws -> [Float] {
        let job = RawJob(
            input: x, weights: weights, recordOffset: recordOffset,
            groupedExpert: groupedExpert, simdMatrix: simdMatrix)
        var output: [Float] = []
        let batchCapture: ((Int, String, [Float]) -> Void)?
        if let capture { batchCapture = { _, name, values in capture(name, values) } } else { batchCapture = nil }
        try evaluateJobs(
            [job], shouldContinue: shouldContinue, capture: batchCapture,
            consumeOutput: { _, values in output = values })
        return output
    }

    /// One command / one joined wait for 1...8 jobs, each with at most 256 rows.
    /// Every job preserves its original QMV or grouped no-split QMM schedule.
    /// Outputs are delivered one tile at a time after the complete command has
    /// joined, so the caller can scatter without retaining all host output tiles.
    /// On allocation, encode, GPU, callback, or cancellation failure, no command
    /// remains in flight when the caller regains ownership of its cache pins.
    public func evaluateBatch(
        _ jobs: [Job], shouldContinue: () -> Bool = { true },
        capture: ((Int, String, [Float]) -> Void)? = nil,
        consumeOutput: (Int, [Float]) throws -> Void
    ) throws {
        guard (1...8).contains(jobs.count) else { throw EngineError("Expert command batch must contain one to eight jobs") }
        try withExtendedLifetime(jobs) {
            try evaluateJobs(
                jobs.map {
                    RawJob(
                        input: $0.input, weights: $0.buffer.buffer, recordOffset: $0.buffer.offset,
                        groupedExpert: $0.groupedExpert, simdMatrix: $0.simdMatrix)
                },
                shouldContinue: shouldContinue, capture: capture, consumeOutput: consumeOutput)
        }
    }

    /// Same projection kernels and joined cache-pin lifetime as evaluateBatch.
    /// The workspace owns a complete exact-once destination map; no accumulation
    /// occurs during scatter and no output is read back to the host.
    public func evaluateDeviceBatch(
        _ jobs: [DeviceJob], workspace: PrefillMoEWorkspace,
        shouldContinue: () -> Bool = { true }
    ) throws {
        guard (1...8).contains(jobs.count), jobs.capacity <= 128,
            MemoryLayout<DeviceJob>.stride <= 256, MemoryLayout<RawJob>.stride <= 256,
            MemoryLayout<PreparedJob>.stride <= 256
        else {
            throw EngineError("Device MoE job metadata exceeds its admitted capacity")
        }
        try workspace.beginBatch(experts: jobs.map(\.expert), slots: jobs.map(\.slots))
        var success = false
        defer { workspace.completeBatch(success: success) }
        try withExtendedLifetime((jobs, workspace)) {
            let raw = jobs.map {
                RawJob(
                    input: [], weights: $0.buffer.buffer, recordOffset: $0.buffer.offset,
                    groupedExpert: $0.groupedExpert, simdMatrix: $0.simdMatrix,
                    gathered: GatheredInput(workspace: workspace, slots: $0.slots))
            }
            try evaluateJobs(
                raw, shouldContinue: shouldContinue, capture: nil,
                consumeOutput: { _, _ in
                    throw EngineError("Device MoE unexpectedly requested a host output")
                })
        }
        success = true
    }

    /// Conservative prospective charge in addition to the unchanged phase
    /// allowance: metadata, one C256 scratch set, the full 8192 x 10 map and 512 pins.
    package static let layerCommandOwnerBytes: UInt64 = 10_620_928
    /// One more page-rounded C256 scratch set (input, gate, up, product, output) per extra wave slot.
    package static let layerWaveScratchBytes: UInt64 = 256 * 2560 * 2 + 2 * 256 * 640 * 4 + 256 * 640 * 2 + 256 * 2560 * 4
    package private(set) var layerWaveCommands: UInt64 = 0
    /// The layer command's buffer limit (`CommandBudget`). An expert tile row (gate, up and down, 3 x 2,560 x 640
    /// multiply-adds) counts `rowWork` `mm` multiply-adds by time, a row of an ungrouped tail (one QMV a row) `qmvRowWork`:
    /// on an M2 Pro an 8,192-token union (81,920 rows) took 0.43 s in one buffer, a tail row about ten grouped rows.
    package let commandBudget = CommandBudget()
    package static let rowWork = 9.2e6
    package static let qmvRowWork = 1e8
    /// Buffers the layer command committed, and the most work one held (settable for checks).
    package private(set) var layerBuffers: UInt64 = 0
    package var largestLayerBufferWork = 0.0
    /// Checks: the layer command after this many more fails as macOS's Impacting Interactivity abort would.
    package var interruptWait: Int?

    /// Whole-layer command composition. Call under consumeLayer's synchronous
    /// pin scope. No slot map mutation or scratch readback after commit.
    package func evaluateDeviceLayer(
        experts: [Int], buffers: [ExpertBuffer],
        selectedRows: [Int: [(Int, Int)]], workspace: PrefillMoEWorkspace,
        simdMatrix: Bool, ungroupedTailStart: Int? = nil, waves: Int = 1,
        groupedExpert: Bool = true, partial: Bool = false,
        shouldContinue: () -> Bool = { true }
    ) throws {
        let prepareStart = DispatchTime.now().uptimeNanoseconds
        guard (1...16).contains(waves) else { throw EngineError("Invalid expert layer wave width") }
        guard shouldContinue() else { throw IOOperationCancelled() }
        let arithmeticBoundary = ungroupedTailStart
        guard groupedExpert || ungroupedTailStart == nil else { throw EngineError("Ungrouped prefill cannot declare a grouped prefix") }
        guard (1...512).contains(experts.count), experts.count == buffers.count,
            experts.capacity <= 1024, buffers.capacity <= 1024,
            MemoryLayout<ExpertBuffer>.stride <= 256
        else {
            throw EngineError("Layer command has invalid expert metadata")
        }
        for buffer in buffers {
            guard buffer.buffer.device.registryID == device.registryID, buffer.offset >= 0,
                buffer.offset % 4 == 0, buffer.offset <= buffer.buffer.length,
                PackStore.expertBytes <= buffer.buffer.length - buffer.offset
            else {
                throw EngineError("Layer command record span is invalid")
            }
        }
        try workspace.withLayerMetadata(bytes: ungroupedTailStart == nil ? 0 : PrefillMoEBudget.mixedTailLayerMetadataBytes) {
            let metadata = try FusedExpertAllocation(
                ledger: ledger, bytes: 4 * 1024 * 1024,
                category: "fused-expert-layer-metadata")
            try withExtendedLifetime((metadata, buffers, workspace)) {
                try autoreleasepool {
                    func scratch(_ bytes: Int) throws -> MTLBuffer {
                        try Self.allocate(device: device, ledger: ledger, bytes: bytes, category: "fused-expert-layer-workspace")
                    }
                    let map = try scratch(workspace.tokens * 10 * 4)
                    let tiles = try workspace.beginLayer(
                        experts: experts, selectedRows: selectedRows, map: map,
                        ungroupedTailStart: ungroupedTailStart, partial: partial)
                    var success = false
                    defer { workspace.completeLayer(success: success) }
                    struct Scratch { let input, gate, up, product, output: MTLBuffer }
                    struct Job {
                        let buffer: ExpertBuffer
                        let rows, mapOffset: Int
                        let gateSchedule, downSchedule: QuantizedSchedule
                        let grouped: Bool
                    }
                    var jobs: [Job] = []
                    var mapRow = 0
                    var encodedTiles = 0
                    var tailTiles = 0
                    var tailRows = 0
                    for (index, expert) in experts.enumerated() {
                        let positions = selectedRows[expert]!
                        let count = positions.count
                        let buffer = buffers[index]
                        let split = arithmeticBoundary.map { tail in positions.firstIndex { $0.0 >= tail } ?? count } ?? count
                        // Preserve both the old prefix's grouped chunk boundaries
                        // (including BM32 selection) and C<256's per-row QMV.
                        // The map is still one expert-major union and one command.
                        for part in 0..<(arithmeticBoundary == nil ? 1 : 2) {
                            let first = part == 0 ? 0 : split
                            let end = part == 0 ? split : count
                            let grouped = groupedExpert && part == 0
                            for lower in stride(from: first, to: end, by: 256) {
                                guard shouldContinue() else { throw IOOperationCancelled() }
                                let rows = min(256, end - lower)
                                let mapOffset = mapRow * 4
                                let gateSchedule = try QuantizedSchedule(
                                    rows: grouped ? rows : 1, width: 2560, outputs: 640, groupedExpert: grouped)
                                let downSchedule = try QuantizedSchedule(
                                    rows: grouped ? rows : 1, width: 640, outputs: 2560, groupedExpert: grouped)
                                guard gateSchedule.splitK == 1, downSchedule.splitK == 1 else {
                                    throw EngineError("Layer command changed projection partitioning")
                                }
                                jobs.append(
                                    Job(
                                        buffer: buffer, rows: rows, mapOffset: mapOffset, gateSchedule: gateSchedule,
                                        downSchedule: downSchedule, grouped: grouped))
                                mapRow += rows
                                encodedTiles += 1
                                if !grouped {
                                    tailTiles += 1
                                    tailRows += rows
                                }
                            }
                        }
                    }
                    guard partial || mapRow + workspace.staleSlots == workspace.tokens * 10, encodedTiles == tiles else {
                        throw EngineError("Layer command dispatch coverage differs from its immutable map")
                    }
                    // Waves (`expert_layer_waves`): each stage is encoded for up to `waves` expert tiles, each with its own scratch
                    // set, between full barriers. Every tile computes exactly as before and scatters to its own
                    // (token, rank) rows, so only scheduling changes. One wave reproduces the original chain.
                    let sets = try (0..<min(waves, max(1, jobs.count))).map { _ in
                        Scratch(
                            input: try scratch(256 * 2560 * 2), gate: try scratch(256 * 640 * 4), up: try scratch(256 * 640 * 4),
                            product: try scratch(256 * 640 * 2), output: try scratch(256 * 2560 * 4))
                    }
                    // Buffers of at most `commandBudget.limit` (`CommandBudget`), cut between waves: a wave ends in a
                    // full barrier, and a buffer boundary orders as one, so every tile computes as in one buffer. Only
                    // the last is waited for.
                    let label = sets.count > 1 ? "prefill-expert-layer-waves-one-join" : "prefill-expert-layer-one-encoder-one-join"
                    let limit = commandBudget.limit
                    var committed: [(command: MTLCommandBuffer, work: Double)] = []
                    var command: MTLCommandBuffer!
                    var encoder: MTLComputeCommandEncoder!
                    var work = 0.0
                    var ended = true
                    defer {
                        if !ended { encoder.endEncoding() }
                        // A throw after a commit: no reader of the pinned records or scratch may outlive this call.
                        try? committed.last?.command.waitBounded(work: committed.reduce(0) { $0 + $1.work }, budget: commandBudget)
                    }
                    func open() throws {
                        guard let next = queue.makeCommandBuffer(),
                            let nextEncoder = next.makeComputeCommandEncoder(dispatchType: sets.count > 1 ? .concurrent : .serial)
                        else {
                            throw EngineError("Layer command/encoder allocation failed", code: "insufficient_memory")
                        }
                        next.label = label
                        command = next
                        encoder = nextEncoder
                        work = 0
                        ended = false
                    }
                    func commit() {
                        encoder.endEncoding()
                        ended = true
                        command.commit()
                        committed.append((command, work))
                        largestLayerBufferWork = max(largestLayerBufferWork, work)
                    }
                    try open()
                    for first in stride(from: 0, to: jobs.count, by: sets.count) {
                        guard shouldContinue() else { throw IOOperationCancelled() }
                        let wave = Array(zip(jobs[first..<min(first + sets.count, jobs.count)], sets))
                        let waveWork = wave.reduce(0.0) { $0 + Double($1.0.rows) * ($1.0.grouped ? Self.rowWork : Self.qmvRowWork) }
                        if work > 0, work + waveWork > limit {
                            commit()
                            try open()
                        }
                        work += waveWork
                        for (job, s) in wave {
                            try workspace.encodeGather(
                                command, map: map, output: s.input, rows: job.rows, mapOffset: job.mapOffset, encoder: encoder)
                        }
                        encoder.memoryBarrier(scope: .buffers)
                        if let gguf, let layout = ModelLayout.ggufExperts {
                            try gguf.encodeTiles(
                                encoder,
                                wave.map { job, s in
                                    GGUFExpertKernels.Tile(
                                        record: (job.buffer.buffer, job.buffer.offset), rows: job.rows,
                                        input: s.input, gate: s.gate, up: s.up, output: s.output)
                                }, layout: layout)
                        } else {
                            // Prefetch-BM32 tiles take gate, up and the product in one dispatch (same bits); the others
                            // keep the separate projections and the product stage.
                            let fused = wave.map { fusesGateUp(rows: $0.0.rows, schedule: $0.0.gateSchedule, simdMatrix: simdMatrix) }
                            for ((job, s), fused) in zip(wave, fused) {
                                if fused {
                                    try encodeGateUpProduct(
                                        input: s.input, weights: job.buffer.buffer, offset: job.buffer.offset, rows: job.rows,
                                        schedule: job.gateSchedule, product: s.product, encoder: encoder)
                                    continue
                                }
                                try encodeProjection(
                                    command, input: s.input, weights: job.buffer.buffer, offset: job.buffer.offset,
                                    kind: .gate, rows: job.rows, schedule: job.gateSchedule, simdMatrix: simdMatrix, output: s.gate,
                                    encoder: encoder)
                                try encodeProjection(
                                    command, input: s.input, weights: job.buffer.buffer, offset: job.buffer.offset,
                                    kind: .up, rows: job.rows, schedule: job.gateSchedule, simdMatrix: simdMatrix, output: s.up,
                                    encoder: encoder)
                            }
                            encoder.memoryBarrier(scope: .buffers)
                            if fused.contains(false) {
                                for ((job, s), fused) in zip(wave, fused) where !fused {
                                    try encodeProduct(
                                        command, gate: s.gate, up: s.up, product: s.product, count: job.rows * 640, encoder: encoder)
                                }
                                encoder.memoryBarrier(scope: .buffers)
                            }
                            for (job, s) in wave {
                                try encodeProjection(
                                    command, input: s.product, weights: job.buffer.buffer, offset: job.buffer.offset,
                                    kind: .down, rows: job.rows, schedule: job.downSchedule, simdMatrix: simdMatrix, output: s.output,
                                    encoder: encoder)
                            }
                        }
                        encoder.memoryBarrier(scope: .buffers)
                        for (job, s) in wave {
                            try workspace.encodeScatter(
                                command, map: map, input: s.output, rows: job.rows, mapOffset: job.mapOffset, encoder: encoder)
                        }
                        // Scatter must finish reading output before its next
                        // overwrite; all other scratch consumers also retire.
                        encoder.memoryBarrier(scope: .buffers)
                    }
                    if sets.count > 1 { layerWaveCommands += 1 }
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    hostPrepareSeconds += Double(DispatchTime.now().uptimeNanoseconds - prepareStart) / 1e9
                    submittedCommands += 1
                    submittedJobs += UInt64(tiles)
                    submittedRows += UInt64(mapRow)
                    commit()
                    let joinStart = DispatchTime.now().uptimeNanoseconds
                    // One queue, in order: the last buffer's end is every buffer's end, so their statuses are final.
                    try committed.last!.command.waitBounded(work: committed.reduce(0) { $0 + $1.work }, budget: commandBudget)
                    hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - joinStart) / 1e9
                    layerBuffers += UInt64(committed.count)
                    let stopped = committed.first { $0.command.status != .completed }?.command
                    var failed = stopped != nil
                    var error = stopped?.error
                    if let left = interruptWait {
                        interruptWait = left > 0 ? left - 1 : nil
                        if left == 0 { (failed, error) = (true, EngineError.fakeInterruption) }
                    }
                    for (done, work) in committed where done.status == .completed {
                        let start = done.gpuStartTime
                        let end = done.gpuEndTime
                        if start.isFinite, end.isFinite, start > 0, end >= start {
                            gpuSeconds += end - start
                            commandBudget.fit(work: work, seconds: end - start)
                        }
                    }
                    if committed.contains(where: { $0.command.gpuStartTime > 0 }) { gpuTimedCommands += 1 }
                    if failed {
                        if EngineError.interrupted(error) { commandBudget.interrupted() }
                        throw EngineError.gpu("Layer expert dispatch failed", error)
                    }
                    completedCommands += 1
                    // Report cancellation only after the command's every reader has
                    // joined; uncommitted/failed work never publishes coverage.
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    if let tail = ungroupedTailStart, !partial {
                        guard tailRows + workspace.staleSlots(from: tail) == (workspace.tokens - tail) * 10, tailTiles <= 512 else {
                            throw EngineError("Mixed tail did not preserve its complete QMV coverage")
                        }
                        mixedTailCommands += 1
                        mixedTailQMVTiles += UInt64(tailTiles)
                        mixedTailQMVRows += UInt64(tailRows)
                    }
                    success = true
                }
            }
        }
    }

    private func evaluateJobs(
        _ jobs: [RawJob], shouldContinue: () -> Bool,
        capture: ((Int, String, [Float]) -> Void)?,
        consumeOutput: (Int, [Float]) throws -> Void
    ) throws {
        let prepareStart = DispatchTime.now().uptimeNanoseconds
        guard shouldContinue() else { throw IOOperationCancelled() }
        guard (1...8).contains(jobs.count) else { throw EngineError("Expert command batch must contain one to eight jobs") }
        for job in jobs {
            guard (1...256).contains(job.rows),
                job.gathered != nil ? job.input.isEmpty : (!job.input.isEmpty && job.input.count % 2560 == 0),
                job.weights.device.registryID == device.registryID, job.recordOffset >= 0, job.recordOffset % 4 == 0,
                job.recordOffset <= job.weights.length, PackStore.expertBytes <= job.weights.length - job.recordOffset
            else {
                throw EngineError("Invalid fused expert activation or verified record span")
            }
        }
        guard capture == nil || gguf == nil else { throw EngineError("Expert captures read the affine intermediates") }
        let metadata = try FusedExpertAllocation(
            ledger: ledger, bytes: UInt64(4096 + jobs.count * 1024),
            category: "fused-expert-batch-metadata")
        let pool = jobs.contains { !$0.groupedExpert && $0.input.count == 2560 } ? scratchPool : nil
        try withExtendedLifetime(metadata) {
            try withScratchScope(pool) { scratch in
                try autoreleasepool {
                    guard let command = queue.makeCommandBuffer() else {
                        throw EngineError("Fused expert command allocation failed", code: "insufficient_memory")
                    }
                    command.label = "expert-batch-\(jobs.count)-gate-up-table-product-down-one-join"
                    var prepared: [PreparedJob] = []
                    prepared.reserveCapacity(jobs.count)
                    for job in jobs {
                        let jobScratch = !job.groupedExpert && job.input.count == 2560 ? scratch : nil
                        prepared.append(try encodeJob(job, command: command, scratch: jobScratch))
                    }
                    // No work is submitted before every allocation and encoder has
                    // succeeded. Private buffers may be freed on a pre-commit error.
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    hostPrepareSeconds += Double(DispatchTime.now().uptimeNanoseconds - prepareStart) / 1e9
                    submittedCommands += 1
                    submittedJobs += UInt64(jobs.count)
                    submittedRows += UInt64(prepared.reduce(0) { $0 + $1.rows })
                    command.commit()
                    let joinStart = DispatchTime.now().uptimeNanoseconds
                    try command.waitBounded()
                    hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - joinStart) / 1e9
                    let gpuStart = command.gpuStartTime
                    let gpuEnd = command.gpuEndTime
                    if gpuStart.isFinite, gpuEnd.isFinite, gpuStart > 0, gpuEnd >= gpuStart {
                        gpuSeconds += gpuEnd - gpuStart
                        gpuTimedCommands += 1
                    }
                    guard command.status == .completed else {
                        throw EngineError.gpu("Fused expert dispatch failed", command.error)
                    }
                    completedCommands += 1
                    // Cancellation during the command is reported only after every
                    // job completed. The enclosing batch can then retire all pins.
                    guard shouldContinue() else { throw IOOperationCancelled() }
                    for (index, item) in prepared.enumerated() {
                        if let capture {
                            capture(index, "gate", Self.floats(item.gate, count: item.rows * 640))
                            capture(index, "up", Self.floats(item.up, count: item.rows * 640))
                            let bits = item.product.contents().assumingMemoryBound(to: UInt16.self)
                            capture(index, "product", (0..<(item.rows * 640)).map { BF16.decode(bits[$0]) })
                        }
                        if jobs[index].gathered == nil {
                            try consumeOutput(index, Self.floats(item.output, count: item.rows * 2560))
                        }
                    }
                }
            }
        }
    }

    private func encodeJob(_ job: RawJob, command: MTLCommandBuffer, scratch: ExpertDecodeScratchPool?) throws -> PreparedJob {
        let rows = job.rows
        if let gguf, let layout = ModelLayout.ggufExperts {
            // BF16 rows as the affine path stages them, then f32 throughout (`product` is `gate`, holding the f32 product).
            let input = try workspace(bytes: rows * 2560 * 2, scratch: scratch)
            let gatherMap: MTLBuffer?
            if let gathered = job.gathered {
                let map = try gathered.workspace.mapBuffer(gathered.slots)
                try gathered.workspace.encodeGather(command, map: map, output: input, rows: rows)
                gatherMap = map
            } else {
                let inputBits = input.contents().assumingMemoryBound(to: UInt16.self)
                for index in job.input.indices { inputBits[index] = BF16.encode(job.input[index]) }
                gatherMap = nil
            }
            let gate = try workspace(bytes: rows * 640 * 4, scratch: scratch)
            let up = try workspace(bytes: rows * 640 * 4, scratch: scratch)
            let output = try workspace(bytes: rows * 2560 * 4, scratch: scratch)
            guard let encoder = command.makeComputeCommandEncoder() else {
                throw EngineError("GGUF expert encoder allocation failed", code: "insufficient_memory")
            }
            do {
                try gguf.encodeTiles(
                    encoder, [.init(record: (job.weights, job.recordOffset), rows: rows, input: input, gate: gate, up: up, output: output)],
                    layout: layout)
            } catch {
                encoder.endEncoding()
                throw error
            }
            encoder.endEncoding()
            if let gathered = job.gathered, let gatherMap {
                try gathered.workspace.encodeScatter(command, map: gatherMap, input: output, rows: rows)
            }
            return PreparedJob(rows: rows, input: input, gate: gate, up: up, product: gate, output: output, gatherMap: gatherMap)
        }
        let gateSchedule = try QuantizedSchedule(
            rows: job.groupedExpert ? rows : 1, width: 2560, outputs: 640,
            groupedExpert: job.groupedExpert)
        let downSchedule = try QuantizedSchedule(
            rows: job.groupedExpert ? rows : 1, width: 640, outputs: 2560,
            groupedExpert: job.groupedExpert)
        guard gateSchedule.splitK == 1, downSchedule.splitK == 1 else {
            throw EngineError("Fused experts require cached QMV or grouped no-split QMM")
        }
        let input = try workspace(bytes: rows * 2560 * 2, scratch: scratch)
        let gatherMap: MTLBuffer?
        if let gathered = job.gathered {
            let map = try gathered.workspace.mapBuffer(gathered.slots)
            try gathered.workspace.encodeGather(command, map: map, output: input, rows: rows)
            gatherMap = map
        } else {
            let inputBits = input.contents().assumingMemoryBound(to: UInt16.self)
            for index in job.input.indices { inputBits[index] = BF16.encode(job.input[index]) }
            gatherMap = nil
        }
        let gate = try workspace(bytes: rows * 640 * 4, scratch: scratch)
        let up = try workspace(bytes: rows * 640 * 4, scratch: scratch)
        let product = try workspace(bytes: rows * 640 * 2, scratch: scratch)
        let output = try workspace(bytes: rows * 2560 * 4, scratch: scratch)
        try encodeProjection(
            command, input: input, weights: job.weights, offset: job.recordOffset, kind: .gate,
            rows: rows, schedule: gateSchedule, simdMatrix: job.simdMatrix, output: gate)
        try encodeProjection(
            command, input: input, weights: job.weights, offset: job.recordOffset, kind: .up,
            rows: rows, schedule: gateSchedule, simdMatrix: job.simdMatrix, output: up)
        try encodeProduct(command, gate: gate, up: up, product: product, count: rows * 640)
        try encodeProjection(
            command, input: product, weights: job.weights, offset: job.recordOffset, kind: .down,
            rows: rows, schedule: downSchedule, simdMatrix: job.simdMatrix, output: output)
        if let gathered = job.gathered, let gatherMap {
            try gathered.workspace.encodeScatter(command, map: gatherMap, input: output, rows: rows)
        }
        return PreparedJob(rows: rows, input: input, gate: gate, up: up, product: product, output: output, gatherMap: gatherMap)
    }

    /// Immutable table shared by the explicitly selected wider decode region.
    /// Its existing allocation owner/charge stays alive with this helper.
    package var decodeRegionSiLUTable: MTLBuffer { silu }

    /// Same compiled lookup/product pipeline and boundaries for a retained
    /// canonical gathered buffer. Caller owns buffers through joined completion.
    func encodeProduct(
        _ command: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer, product: MTLBuffer, count: Int,
        encoder supplied: MTLComputeCommandEncoder? = nil
    ) throws {
        guard count > 0, gate.length >= count * 4, up.length >= count * 4, product.length >= count * 2,
            let encoder = supplied ?? command.makeComputeCommandEncoder()
        else {
            throw EngineError("Fused expert product encoder allocation/geometry failed", code: "insufficient_memory")
        }
        encoder.setComputePipelineState(productPipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(silu, offset: 0, index: 2)
        encoder.setBuffer(product, offset: 0, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, productPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        if supplied == nil { encoder.endEncoding() }
    }

    /// Whether `encodeProjection` would run this job's gate and up on the prefetch BM32 kernel, so that
    /// `encodeGateUpProduct` can take both and the product in one dispatch.
    private func fusesGateUp(rows: Int, schedule: QuantizedSchedule, simdMatrix: Bool) -> Bool {
        gateUpPipeline != nil && simdMatrix && schedule.kind == "qmm_grouped_expert_no_split" && schedule.splitK == 1
            && schedule.partitionSize == 2560
            && rows >= groupedBM32MinRows && !(groupedMMAMinRows > 0 && rows >= groupedMMAMinRows)
    }
    /// Gate, up and the SiLU-table product of a grouped tile (BITWISE: the prefetch BM32 projections and
    /// `expert_table_silu_product`); counted as the two projections it replaces.
    private func encodeGateUpProduct(
        input: MTLBuffer, weights: MTLBuffer, offset: Int, rows: Int, schedule: QuantizedSchedule, product: MTLBuffer,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let gateUpPipeline, (offset + ExpertProjection.gate.weightOffset) % 8 == 0,
            (offset + ExpertProjection.up.weightOffset) % 8 == 0,
            product.length >= rows * 640 * 2
        else {
            throw EngineError("Invalid grouped gate/up geometry")
        }
        encoder.setComputePipelineState(gateUpPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        for (kind, base) in [(ExpertProjection.gate, 1), (.up, 6)] {
            encoder.setBuffer(weights, offset: offset + kind.weightOffset, index: base)
            encoder.setBuffer(weights, offset: offset + kind.scaleOffset, index: base + 1)
            encoder.setBuffer(weights, offset: offset + kind.biasOffset, index: base + 2)
        }
        let shape = [
            2560, 640, rows, ModelLayout.experts.group, schedule.splitK, schedule.partitionSize, schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.setBuffer(product, offset: 0, index: 5)
        encoder.setBuffer(silu, offset: 0, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: 640 / 32, height: (rows + 31) / 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        groupedTiledJobRows16[min(rows, 256) / 16] += 1
        gateUpFusedTiles += 1
        groupedBM32EncodedProjections += 2
        groupedPrefetchEncodedProjections += 2
        if groupedMMAMinRows > 0 { groupedMMABelowMinRowsProjections += 2 }
    }

    private func encodeProjection(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        offset: Int, kind: ExpertProjection, rows: Int,
        schedule: QuantizedSchedule, simdMatrix: Bool, output: MTLBuffer,
        encoder supplied: MTLComputeCommandEncoder? = nil
    ) throws {
        let matrix = schedule.kind.hasPrefix("qmm")
        let useSIMD = matrix && simdMatrix
        let shareRows = !matrix && rows > 1
        guard !shareRows || (schedule.laneCount == 32 && schedule.splitK == 1 && schedule.valuesPerLane == (kind == .down ? 8 : 16)) else {
            throw EngineError("Shared-weight expert QMV changed the one-row arithmetic schedule")
        }
        let useTiled = useSIMD && groupedTiled && schedule.kind == "qmm_grouped_expert_no_split"
        let useBM32 = useTiled && groupedBM32 && rows >= groupedBM32MinRows
        let useMMA = useBM32 && groupedMMAMinRows > 0 && rows >= groupedMMAMinRows && kind.outputs % 64 == 0
        let pipeline: MTLComputePipelineState
        if useTiled {
            guard let tiledPipeline, schedule.splitK == 1, schedule.partitionSize == kind.width,
                kind.width % 64 == 0, kind.outputs % 32 == 0
            else {
                throw EngineError("Invalid grouped tiled expert geometry")
            }
            if useBM32 {
                guard let tiled32Pipeline else { throw EngineError("Missing grouped BM32 pipeline") }
                // The prefetch and MMA kernels' vector loads need 8-byte aligned weight bindings.
                guard !groupedPrefetch || (offset + kind.weightOffset) % 8 == 0 else {
                    throw EngineError("Grouped prefetch weight binding is not 8-byte aligned")
                }
                if useMMA {
                    guard let mmaPipeline else { throw EngineError("Missing grouped MMA pipeline") }
                    pipeline = mmaPipeline
                } else {
                    pipeline = tiled32Pipeline
                }
            } else {
                pipeline = tiledPipeline
            }
        } else {
            pipeline = useSIMD ? simdPipeline : (matrix ? qmmPipeline : (shareRows ? multirowQMVPipeline : qmvPipeline))
        }
        guard let encoder = supplied ?? command.makeComputeCommandEncoder() else {
            throw EngineError("Fused expert projection encoder allocation failed", code: "insufficient_memory")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: offset + kind.weightOffset, index: 1)
        encoder.setBuffer(weights, offset: offset + kind.scaleOffset, index: 2)
        encoder.setBuffer(weights, offset: offset + kind.biasOffset, index: 3)
        let shape = [
            kind.width, kind.outputs, rows, ModelLayout.experts.group, schedule.splitK, schedule.partitionSize,
            schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.setBuffer(output, offset: 0, index: 5)
        if useTiled {
            let blockRows = useMMA ? 64 : useBM32 ? 32 : 16
            let blockOutputs = useMMA ? 64 : 32
            encoder.dispatchThreadgroups(
                MTLSize(width: kind.outputs / blockOutputs, height: (rows + blockRows - 1) / blockRows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: useBM32 ? 4 : 2, depth: 1))
        } else {
            let threads =
                useSIMD
                ? ((rows + 7) / 8) * ((kind.outputs + 7) / 8) * 32
                : (shareRows ? (rows + 7) / 8 : rows) * kind.outputs * 32
            encoder.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        }
        if supplied == nil { encoder.endEncoding() }
        if useTiled && kind == .gate { groupedTiledJobRows16[min(rows, 256) / 16] += 1 }
        if useBM32 {
            groupedBM32EncodedProjections += 1
            if useMMA {
                groupedMMAEncodedProjections += 1
            } else {
                if groupedPrefetch { groupedPrefetchEncodedProjections += 1 }
                if groupedMMAMinRows > 0 { groupedMMABelowMinRowsProjections += 1 }
            }
        }
        if shareRows { multirowQMVEncodedProjections += 1 }
    }

    private func workspace(bytes: Int, scratch: ExpertDecodeScratchPool?) throws -> MTLBuffer {
        if let scratch { return try scratch.take(bytes: bytes) }
        return try Self.allocate(device: device, ledger: ledger, bytes: bytes, category: "fused-expert-workspace")
    }
    private static func allocate(device: MTLDevice, ledger: MemoryLedger, bytes: Int, category: String) throws -> MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength else { throw EngineError("Invalid fused expert buffer size") }
        let page = UInt64(getpagesize())
        let charged = (UInt64(bytes) + page - 1) / page * page
        let owner = try FusedExpertAllocation(ledger: ledger, bytes: charged, category: category)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Fused expert buffer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &FusedExpertAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard UInt64(buffer.allocatedSize) <= charged else { throw EngineError("Fused expert allocation exceeded its page reservation") }
        return buffer
    }
    private static func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    private static let productSource = #"""

        // product[i] = bf16(silu_table[bf16 bits of gate[i]] * up[i]), one thread per element of a
        // rows x 640 gate/up pair (FP32 in, BF16 out). Both roundings are round_bf16 (nearest even, NaN
        // quieted). BITWISE: equal to the host reference round(silu(gate) * up) for the BF16-valued gates the
        // projections produce (fusedExpertPreservesIntermediateBF16AndUnaryBoundaries).
        // The table is derived from the caller's immutable LayerUnaryArithmetic.
        // This library keeps safe algebra/precise functions like its unchanged
        // quantized kernels; no global fast-math policy change is introduced.
        kernel void expert_table_silu_product(device const float *gate [[buffer(0)]],
            device const float *up [[buffer(1)]], device const ushort *silu [[buffer(2)]],
            device ushort *product [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            uint gateBits = as_type<uint>(round_bf16(gate[i])) >> 16;
            float activated = from_bf16(silu[gateBits]);
            product[i] = ushort(as_type<uint>(round_bf16(activated * up[i])) >> 16);
        }
        """#
}

// Older generation/phase results predate scratch reuse. Their missing field
// denotes the disabled zero-capacity pool; malformed present fields still fail.
extension FusedExpertExecution.Snapshot {
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            submittedCommands: try values.decode(UInt64.self, forKey: .submittedCommands),
            completedCommands: try values.decode(UInt64.self, forKey: .completedCommands),
            submittedJobs: try values.decode(UInt64.self, forKey: .submittedJobs),
            submittedRows: try values.decode(UInt64.self, forKey: .submittedRows),
            gpuTimedCommands: try values.decode(UInt64.self, forKey: .gpuTimedCommands),
            mixedTailCommands: try values.decodeIfPresent(UInt64.self, forKey: .mixedTailCommands) ?? 0,
            mixedTailQMVTiles: try values.decodeIfPresent(UInt64.self, forKey: .mixedTailQMVTiles) ?? 0,
            mixedTailQMVRows: try values.decodeIfPresent(UInt64.self, forKey: .mixedTailQMVRows) ?? 0,
            multirowQMVEncodedProjections: try values.decodeIfPresent(UInt64.self, forKey: .multirowQMVEncodedProjections) ?? 0,
            gpuSeconds: try values.decode(Double.self, forKey: .gpuSeconds),
            hostJoinSeconds: try values.decode(Double.self, forKey: .hostJoinSeconds),
            hostPrepareSeconds: try values.decodeIfPresent(Double.self, forKey: .hostPrepareSeconds),
            scratch: values.contains(.scratch) ? try values.decode(FusedExpertExecution.ScratchSnapshot.self, forKey: .scratch) : .disabled)
    }
}
