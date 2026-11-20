import Darwin
import Foundation
import Metal
import ObjectiveC

private final class HCDecodeAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    let bytes: UInt64
    init(ledger: MemoryLedger, bytes: Int) throws {
        self.ledger = ledger
        let page = UInt64(getpagesize())
        self.bytes = (UInt64(bytes) + page - 1) / page * page
        reservation = try ledger.reserve(bytes: self.bytes, category: "hc_decode_pipeline")
    }
    deinit { try? ledger.release(reservation) }
}
private enum HCDecodeAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The hyper-connection entry on the GPU, BITWISE against LayerMath.hyperConnection's host arithmetic:
/// norm = grouped RMS of the BF16 residual (4 groups of 2560, weights hc_norm), down = W_down (320 x 10240,
/// affine4) . BF16(norm), activation = SiLU(BF16(down / 4)) (table), rawUp = W_up (10240 x 320) . activation,
/// mixed = BF16(sum over 4 streams, in order, of BF16(sigmoid(rawUp) * norm)) / 4 (FusedHyperConnection's
/// epilogue, host-repaired where flagged) and, with injection weights, gates = BF16(2 * sigmoid(BF16((W_inject
/// (4 x 10240) . norm) / 4))). One joined command per call (`evaluate`: 1 or 3...256 rows), or encoded into a
/// caller's command (`encodeRow`, `encodeTile`). GPU handoffs preserve every BF16 boundary. Prefill keeps
/// the actual-row QMM/SIMD/tiled/BM32/split-K schedules of the owning ReferenceKernels.
public final class FusedHCDecode {
    public static let policy = "hc_decode_one_command_existing_primitives_v1"
    public static let prefillPolicy = "hc_prefill_one_command_existing_primitives_v1"
    public struct Projection {
        public let words, scales, biases: MTLBuffer
        public init(words: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer) {
            self.words = words
            self.scales = scales
            self.biases = biases
        }
    }
    public struct Result {
        public let mixed, injection: [Float]
        public let norm, down, activated, rawUp, sigmoidUp, rawInjection: [Float]?
        public let repairedOutputs: Int
    }
    /// Caller-owned, page-charged row-one scratch for a wider joined region.
    /// Two HC stages need distinct instances if their captures remain live.
    package struct RowBuffers {
        package let norm, normBF16, down, activation, rawUp, mixed, exceptions: MTLBuffer
        package let rawInjection, injection: MTLBuffer
        package let capturedUp: MTLBuffer?
        package init(
            norm: MTLBuffer, normBF16: MTLBuffer, down: MTLBuffer, activation: MTLBuffer, rawUp: MTLBuffer, mixed: MTLBuffer,
            exceptions: MTLBuffer, rawInjection: MTLBuffer, injection: MTLBuffer, capturedUp: MTLBuffer?
        ) {
            self.norm = norm
            self.normBF16 = normBF16
            self.down = down
            self.activation = activation
            self.rawUp = rawUp
            self.mixed = mixed
            self.exceptions = exceptions
            self.rawInjection = rawInjection
            self.injection = injection
            self.capturedUp = capturedUp
        }
        package var all: [MTLBuffer] {
            [norm, normBF16, down, activation, rawUp, mixed, exceptions, rawInjection, injection]
                + (capturedUp.map { [$0] } ?? [])
        }
    }
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, repairedOutputs: UInt64
        public let gpuSeconds, hostJoinSeconds: Double
        public let prefillSubmittedCommands, prefillCompletedCommands: UInt64
        public let lastPrefillRows: Int?
        public let lastPrefillDown, lastPrefillUp: ReferenceKernels.PinnedProjectionPlan?
        public let lastPrefillInjection: PinnedDenseLinear.Plan?
    }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let ledger: MemoryLedger
    private let reductions: PinnedReductions
    private let dense: PinnedDenseLinear
    private let epilogue: FusedHyperConnection
    private let prefillKernels: ReferenceKernels?
    private let prefillSIMDMatrix: Bool
    private var prefillSubmitted: UInt64 = 0, prefillCompleted: UInt64 = 0
    private var lastPrefillRows: Int?
    private var lastPrefillDown, lastPrefillUp: ReferenceKernels.PinnedProjectionPlan?
    private var lastPrefillInjection: PinnedDenseLinear.Plan?
    private let qmv, narrow, lookupNarrow, lookupFloat: MTLComputePipelineState
    private let activationTable, injectionTable: MTLBuffer
    private var submitted: UInt64 = 0, completed: UInt64 = 0, repaired: UInt64 = 0
    private var gpuSeconds: Double = 0, hostJoinSeconds: Double = 0
    private var active = false
    /// `hc_exact_fusion` (from the prefill kernels' flag): tile entries encoded as the single fused
    /// narrow + grouped RMS + narrow dispatch instead of three (path proof for the whole snapshot).
    public private(set) var fusedEntryTiles: UInt64 = 0
    package var fusedEntryEnabled: Bool { prefillKernels?.hcExactFusion == true && reductions.hcFusedEntryPrepared }
    /// `hc_decode_fusion`: row-one HCs encode four exact fused dispatches instead of eight
    /// (HCDecodeFusionKernels); callers that pass the FP32 residual and the mixed BF16 destination also drop
    /// their own two narrowings (ten encoders per row HC become four). Path proofs below; nil pipelines = off.
    /// On only when requested and this GPU's pipelines hold the fixed threadgroups; otherwise rows take the
    /// bit-identical eight-dispatch route.
    public var decodeFusion: Bool { decodeFusionEnabled }
    private let fusedDown, fusedUp, fusedInjection: MTLComputePipelineState?
    /// Row HCs encoded on the fused kernels, of which with the fused FP32 entry and with the fused mixed narrowing.
    public private(set) var decodeFusedRows: UInt64 = 0, decodeFusedEntries: UInt64 = 0, decodeFusedMixedNarrows: UInt64 = 0
    /// One-row layer HCs (with injection) evaluated on the synchronous eight-dispatch route (`evaluate`) while
    /// `hc_decode_fusion` is on: decode that left the encoded whole/frontier paths (must stay zero).
    public private(set) var decodeFusionFallbacks: UInt64 = 0
    /// One-row HCs without injection on the synchronous route by design: the final readout mixer
    /// (`model.hyper_connection_mixer`, once per decode token, attributed to lm_head_sampling). Reported only.
    public private(set) var decodeFusionUnfusedByDesign: UInt64 = 0
    package var decodeFusionEnabled: Bool { fusedDown != nil && fusedUp != nil && fusedInjection != nil && reductions.hcFusedEntryPrepared }

    public init(
        arithmetic: LayerUnaryArithmetic, ledger: MemoryLedger,
        prefillKernels: ReferenceKernels? = nil, prefillSIMDMatrix: Bool = false, decodeFusion: Bool = false
    ) throws {
        self.prefillKernels = prefillKernels
        self.prefillSIMDMatrix = prefillSIMDMatrix
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("HC decode Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        self.device = device
        self.queue = queue
        self.ledger = ledger
        reductions = try PinnedReductions(ledger: ledger)
        dense = try PinnedDenseLinear(ledger: ledger)
        epilogue = try FusedHyperConnection(arithmetic: arithmetic, ledger: ledger, minimumRows: 1)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        // Flag off: the library source is unchanged. On: the fused row kernels follow the same text and options.
        let library = try device.makeLibrary(
            source: ModelLayout.metalDefines + ReferenceKernels.expertFusionMetalSource + Self.handoffSource
                + (decodeFusion ? HCDecodeFusionKernels.referenceSource : ""), options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw EngineError("HC decode kernel missing: " + name) }
            return try device.makeComputePipelineState(function: function)
        }
        qmv = try pipeline("qmv_pinned_reference")
        narrow = try pipeline("hc_decode_narrow")
        lookupNarrow = try pipeline("hc_decode_lookup_narrow")
        lookupFloat = try pipeline("hc_decode_lookup_float")
        if decodeFusion {
            let down = try pipeline(HCDecodeFusionKernels.downV2Name)
            let up = try pipeline(HCDecodeFusionKernels.upV2Name)
            // PinnedDenseLinear's own source and options (safe math, fast functions), BF16 output constant.
            let denseOptions = MTLCompileOptions()
            denseOptions.languageVersion = .version3_1
            denseOptions.mathMode = .safe
            denseOptions.mathFloatingPointFunctions = .fast
            let denseLibrary = try device.makeLibrary(
                source: ModelLayout.metalDefines + PinnedDenseLinear.metalSource + HCDecodeFusionKernels.denseSource, options: denseOptions)
            let constants = MTLFunctionConstantValues()
            var bf16 = true
            constants.setConstantValue(&bf16, type: .bool, index: 0)
            let injectionPipeline = try device.makeComputePipelineState(
                function: denseLibrary.makeFunction(name: HCDecodeFusionKernels.injectionV2Name, constantValues: constants))
            // Register pressure differs per GPU family; a pipeline that cannot hold its threadgroup leaves fusion off.
            let fits =
                down.threadExecutionWidth == 32 && up.threadExecutionWidth == 32 && injectionPipeline.threadExecutionWidth == 32
                && down.maxTotalThreadsPerThreadgroup >= 32 * HCDecodeFusionKernels.downSIMDGroups
                && up.maxTotalThreadsPerThreadgroup >= 32 * HCDecodeFusionKernels.upSIMDGroupsPerThreadgroup
                && injectionPipeline.maxTotalThreadsPerThreadgroup >= 256
            fusedDown = fits ? down : nil
            fusedUp = fits ? up : nil
            fusedInjection = fits ? injectionPipeline : nil
        } else {
            fusedDown = nil
            fusedUp = nil
            fusedInjection = nil
        }
        activationTable = try Self.allocate(device, ledger, 65536 * 2)
        injectionTable = try Self.allocate(device, ledger, 65536 * 2)
        let activation = activationTable.contents().assumingMemoryBound(to: UInt16.self)
        let injection = injectionTable.contents().assumingMemoryBound(to: UInt16.self)
        // These exact CPU expression boundaries are evaluated once over the
        // complete BF16 domain, including subnormals, signed zeros and NaNs.
        for key in 0..<65536 {
            let divided = LayerArithmetic.round(BF16.decode(UInt16(key)) / 4)
            activation[key] = BF16.encode(arithmetic.silu(divided))
            injection[key] = BF16.encode(LayerArithmetic.round(2 * arithmetic.sigmoid(divided)))
        }
        if prefillKernels?.hcExactFusion == true || fusedDown != nil {
            // A GPU that cannot hold the fused entry's 640 threads keeps the bit-identical separate dispatches.
            do { try reductions.prepareHCFusedEntry() } catch let error as EngineError where error.code == "invalid_configuration" {}
        }
    }
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submitted, completedCommands: completed, repairedOutputs: repaired,
            gpuSeconds: gpuSeconds, hostJoinSeconds: hostJoinSeconds,
            prefillSubmittedCommands: prefillSubmitted, prefillCompletedCommands: prefillCompleted,
            lastPrefillRows: lastPrefillRows, lastPrefillDown: lastPrefillDown,
            lastPrefillUp: lastPrefillUp, lastPrefillInjection: lastPrefillInjection)
    }

    private static func allocate(_ device: MTLDevice, _ ledger: MemoryLedger, _ bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 256 * 10240 * 4 else {
            throw EngineError("HC pipeline allocation exceeds the fixed 256-row tile")
        }
        return try allocateChecked(device, ledger, bytes)
    }
    private static func allocateChecked(_ device: MTLDevice, _ ledger: MemoryLedger, _ bytes: Int) throws -> MTLBuffer {
        let owner = try HCDecodeAllocation(ledger: ledger, bytes: bytes)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared), UInt64(buffer.allocatedSize) <= owner.bytes else {
            throw EngineError("HC decode buffer allocation failed or exceeded admitted charge", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &HCDecodeAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }
    private func allocate(_ bytes: Int) throws -> MTLBuffer { try Self.allocate(device, ledger, bytes) }
    private func validate(_ projection: Projection, width: Int, outputs: Int) throws {
        guard projection.words.device.registryID == device.registryID, projection.scales.device.registryID == device.registryID,
            projection.biases.device.registryID == device.registryID, projection.words.length >= outputs * width / 8 * 4,
            projection.scales.length >= outputs * width / ModelLayout.group * 2,
            projection.biases.length >= outputs * width / ModelLayout.group * 2
        else {
            throw EngineError("HC decode quantized projection buffers differ from fixed shape")
        }
    }
    private func project(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: Projection,
        output: MTLBuffer, width: Int, outputs: Int
    ) throws {
        let schedule = try QuantizedSchedule(rows: 1, width: width, outputs: outputs)
        guard !schedule.kind.hasPrefix("qmm"), let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("HC decode requires the row-one QMV schedule")
        }
        let shape = [
            width, outputs, 1, ModelLayout.group, schedule.splitK, schedule.partitionSize, schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        encoder.setComputePipelineState(qmv)
        for (index, buffer) in [input, weights.words, weights.scales, weights.biases].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.setBuffer(output, offset: 0, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: outputs * 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, qmv.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }
    private func handoff(
        _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState,
        input: MTLBuffer, output: MTLBuffer, count: Int, table: MTLBuffer? = nil
    ) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw EngineError("HC decode handoff encoder allocation failed") }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var count = UInt32(count)
        encoder.setBytes(&count, length: 4, index: 2)
        if let table { encoder.setBuffer(table, offset: 0, index: 3) }
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Existing row-one schedules and handoffs, encoded into an uncommitted
    /// caller command. No state/capture/host result is published. The caller
    /// zeroes exceptions beforehand, retains all buffers and weight pins, and
    /// checks every exception word after join before consuming the result.
    /// `residualFP32` / `mixedBF16` (only with `hc_decode_fusion`): the caller omits its own narrowing of the FP32
    /// row into `residualBF16` (the fused entry writes it, as hc_decode_narrow would) and of `buffers.mixed` into
    /// `mixedBF16` (the fused up/epilogue kernel writes it from the stored mixed value).
    package func encodeRow(
        _ command: MTLCommandBuffer, residualBF16: MTLBuffer, normWeights: MTLBuffer,
        downWeights: Projection, upWeights: Projection, injectionWeights: MTLBuffer,
        buffers: RowBuffers, residualFP32: MTLBuffer? = nil, mixedBF16: MTLBuffer? = nil
    ) throws {
        let sizes =
            [10240 * 4, 10240 * 2, 320 * 4, 320 * 2, 10240 * 4, 2560 * 4, 80 * 4, 16, 16]
            + (buffers.capturedUp == nil ? [] : [10240 * 4])
        let all = buffers.all
        guard !active, all.count == sizes.count,
            zip(all, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(all.map { ObjectIdentifier($0) }).count == all.count,
            !all.contains(where: { $0 === residualBF16 || $0 === normWeights || $0 === injectionWeights }),
            residualBF16.device.registryID == device.registryID, residualBF16.length >= 10240 * 2,
            normWeights.device.registryID == device.registryID, normWeights.length >= 10240 * 2,
            injectionWeights.device.registryID == device.registryID, injectionWeights.length >= 4 * 10240 * 4,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid caller-owned HC row-one region")
        }
        try validate(downWeights, width: 10240, outputs: 320)
        try validate(upWeights, width: 320, outputs: 10240)
        let reads = [downWeights.words, downWeights.scales, downWeights.biases, upWeights.words, upWeights.scales, upWeights.biases]
        guard !all.contains(where: { output in reads.contains(where: { $0 === output }) }) else {
            throw EngineError("HC scratch aliases a retained weight buffer")
        }
        if decodeFusion {
            try encodeFusedRow(
                command, residualBF16: residualBF16, normWeights: normWeights, downWeights: downWeights,
                upWeights: upWeights, injectionWeights: injectionWeights, buffers: buffers, residualFP32: residualFP32, mixedBF16: mixedBF16
            )
            return
        }
        guard residualFP32 == nil, mixedBF16 == nil else {
            throw EngineError("A fused HC row entry/mixed narrowing requires hc_decode_fusion")
        }
        try reductions.encodeGroupedRMS(
            command, input: residualBF16, weights: normWeights, output: buffers.norm,
            rows: 1, width: 10240, groupSize: 2560)
        try handoff(command, pipeline: narrow, input: buffers.norm, output: buffers.normBF16, count: 10240)
        try project(command, input: buffers.normBF16, weights: downWeights, output: buffers.down, width: 10240, outputs: 320)
        try handoff(command, pipeline: lookupNarrow, input: buffers.down, output: buffers.activation, count: 320, table: activationTable)
        try project(command, input: buffers.activation, weights: upWeights, output: buffers.rawUp, width: 320, outputs: 10240)
        try epilogue.encode(
            command, norm: buffers.norm, rawUp: buffers.rawUp, mixed: buffers.mixed,
            captured: buffers.capturedUp, exceptions: buffers.exceptions, rows: 1)
        try dense.encodeRows(
            command, input: buffers.norm, weights: injectionWeights, output: buffers.rawInjection,
            partials: nil, rows: 1, width: 10240, outputs: 4, outputBF16: true)
        try handoff(command, pipeline: lookupFloat, input: buffers.rawInjection, output: buffers.injection, count: 4, table: injectionTable)
    }

    /// `hc_decode_fusion` row HC: entry (fused narrow + grouped RMS + narrow when `residualFP32` is given, else the
    /// current two dispatches), QMV down + activation lookup, QMV up + epilogue (+ mixed narrowing), dense injection
    /// + lookup. Every stored word equals the eight-dispatch route's (HCDecodeFusionSelfChecks).
    private func encodeFusedRow(
        _ command: MTLCommandBuffer, residualBF16: MTLBuffer, normWeights: MTLBuffer,
        downWeights: Projection, upWeights: Projection, injectionWeights: MTLBuffer,
        buffers: RowBuffers, residualFP32: MTLBuffer?, mixedBF16: MTLBuffer?
    ) throws {
        guard let fusedDown, let fusedUp, let fusedInjection else { throw EngineError("hc_decode_fusion pipelines are unavailable") }
        let all = buffers.all
        for extra in [residualFP32, mixedBF16].compactMap({ $0 }) {
            guard extra.device.registryID == device.registryID, extra !== residualBF16, extra !== normWeights, extra !== injectionWeights,
                !all.contains(where: { $0 === extra }),
                ![downWeights.words, downWeights.scales, downWeights.biases, upWeights.words, upWeights.scales, upWeights.biases].contains(
                    where: { $0 === extra })
            else {
                throw EngineError("Invalid fused HC row entry/mixed buffer (aliased or foreign)")
            }
        }
        if let residualFP32, let mixedBF16 {
            guard residualFP32 !== mixedBF16 else { throw EngineError("Fused HC row entry aliases its mixed narrowing") }
        }
        guard (residualFP32?.length ?? Int.max) >= 10240 * 4, (mixedBF16?.length ?? Int.max) >= 2560 * 2 else {
            throw EngineError("Fused HC row entry/mixed buffer is too short")
        }
        let down = try QuantizedSchedule(rows: 1, width: 10240, outputs: 320)
        let up = try QuantizedSchedule(rows: 1, width: 320, outputs: 10240)
        // The fused kernels replay exactly these row-one QMV lane maps; anything else must not reach them.
        guard down.kind == "qmv_fast", down.laneCount == 32, down.valuesPerLane == 16, down.splitK == 1,
            up.kind == "qmv", up.laneCount == 32, up.valuesPerLane == 8, up.splitK == 1,
            10240 / (32 * down.valuesPerLane) <= 32
        else {
            throw EngineError("hc_decode_fusion requires the current row-one QMV schedules")
        }
        if let residualFP32 {
            try reductions.encodeHCFusedEntry(
                command, residual: residualFP32, weights: normWeights, residualBF16: residualBF16,
                output: buffers.norm, outputBF16: buffers.normBF16, rows: 1)
        } else {
            try reductions.encodeGroupedRMS(
                command, input: residualBF16, weights: normWeights, output: buffers.norm,
                rows: 1, width: 10240, groupSize: 2560)
            try handoff(command, pipeline: narrow, input: buffers.norm, output: buffers.normBF16, count: 10240)
        }
        guard let downEncoder = command.makeComputeCommandEncoder() else { throw EngineError("HC fused down encoder allocation failed") }
        downEncoder.setComputePipelineState(fusedDown)
        for (index, buffer) in [buffers.normBF16, downWeights.words, downWeights.scales, downWeights.biases].enumerated() {
            downEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        // v2 kernels have the literal row-one geometry (checked above); shape carries only the launch choice.
        let rowsPerGroup = HCDecodeFusionKernels.downRowsPerThreadgroup
        guard (1...8).contains(rowsPerGroup), 320 % rowsPerGroup == 0 else { throw EngineError("Invalid fused HC down geometry") }
        let downShape = [UInt32(rowsPerGroup)]
        downShape.withUnsafeBytes { downEncoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        downEncoder.setBuffer(buffers.down, offset: 0, index: 5)
        downEncoder.setBuffer(buffers.activation, offset: 0, index: 6)
        downEncoder.setBuffer(activationTable, offset: 0, index: 7)
        downEncoder.dispatchThreadgroups(
            MTLSize(width: 320 / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * HCDecodeFusionKernels.downSIMDGroups, height: 1, depth: 1))
        downEncoder.endEncoding()

        guard let upEncoder = command.makeComputeCommandEncoder() else { throw EngineError("HC fused up encoder allocation failed") }
        upEncoder.setComputePipelineState(fusedUp)
        for (index, buffer) in [buffers.activation, upWeights.words, upWeights.scales, upWeights.biases].enumerated() {
            upEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        let columns = HCDecodeFusionKernels.upColumnsPerSIMDGroup
        let simdGroups = HCDecodeFusionKernels.upSIMDGroupsPerThreadgroup
        guard (1...32).contains(columns), 2560 % columns == 0, simdGroups >= 1 else { throw EngineError("Invalid fused HC up geometry") }
        let upShape: [UInt32] = [buffers.capturedUp == nil ? 0 : 1, mixedBF16 == nil ? 0 : 1, UInt32(columns)]
        upShape.withUnsafeBytes { upEncoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        // Disabled capture / mixed narrowing bind an existing buffer that the runtime flags forbid writing.
        for (index, buffer) in [
            buffers.rawUp, buffers.norm, epilogue.sigmoidTable, buffers.mixed, buffers.capturedUp ?? buffers.mixed,
            buffers.exceptions, mixedBF16 ?? buffers.mixed,
        ].enumerated() {
            upEncoder.setBuffer(buffer, offset: 0, index: 5 + index)
        }
        let columnsPerGroup = columns * simdGroups
        upEncoder.dispatchThreadgroups(
            MTLSize(width: (2560 + columnsPerGroup - 1) / columnsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * simdGroups, height: 1, depth: 1))
        upEncoder.endEncoding()

        let plan = try dense.executionPlan(rows: 1, width: 10240, outputs: 4)
        guard plan.kind == .gemv, !plan.matrixIsInput, plan.reductionSIMDGroups == 8 else {
            throw EngineError("hc_decode_fusion requires the current row-one injection GEMV plan")
        }
        guard let injectionEncoder = command.makeComputeCommandEncoder() else {
            throw EngineError("HC fused injection encoder allocation failed")
        }
        injectionEncoder.setComputePipelineState(fusedInjection)
        for (index, buffer) in [injectionWeights, buffers.norm, buffers.rawInjection].enumerated() {
            injectionEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        let injectionShape = [UInt32(10240), 4, 1, UInt32(plan.reductionSIMDGroups)]
        injectionShape.withUnsafeBytes { injectionEncoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        injectionEncoder.setBuffer(buffers.injection, offset: 0, index: 4)
        injectionEncoder.setBuffer(injectionTable, offset: 0, index: 5)
        injectionEncoder.dispatchThreadgroups(
            MTLSize(width: 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: plan.reductionSIMDGroups, depth: 1))
        injectionEncoder.endEncoding()
        decodeFusedRows += 1
        if residualFP32 != nil { decodeFusedEntries += 1 }
        if mixedBF16 != nil { decodeFusedMixedNarrows += 1 }
    }

    package func tilePartialBytes(rows: Int) throws -> (down: Int, up: Int, injection: Int) {
        // Two rows: the MTP verify step.
        guard (2...256).contains(rows), let kernels = prefillKernels else {
            throw EngineError("HC tile requires enabled actual-row kernels")
        }
        let down = try kernels.pinnedProjectionPlan(rows: rows, width: 10240, outputs: 320, simdMatrix: prefillSIMDMatrix)
        let up = try kernels.pinnedProjectionPlan(rows: rows, width: 320, outputs: 10240, simdMatrix: prefillSIMDMatrix)
        let injection = try dense.executionPlan(rows: rows, width: 10240, outputs: 4)
        return (down.partialBytes, up.partialBytes, injection.kind == .splitK ? rows * 4 * injection.splitKPartitions * 4 : 0)
    }

    /// The existing prefill schedule, encoded without crossing a host Array
    /// boundary. Each logical role has private storage, including both HCs and
    /// any required split-K partials. Exception repair/publication belongs to
    /// the joined enclosing transaction. No small-row dispatch is enabled.
    /// `residualFP32` (only with `hc_exact_fusion`): the caller omits its own narrowing of these FP32 rows;
    /// the fused entry writes `residualBF16` (as hc_decode_narrow would), the norm and its BF16 narrowing.
    package func encodeTile(
        _ command: MTLCommandBuffer, rows: Int, residualBF16: MTLBuffer, normWeights: MTLBuffer,
        downWeights: Projection, upWeights: Projection, injectionWeights: MTLBuffer,
        buffers s: RowBuffers, downPartials: MTLBuffer?, upPartials: MTLBuffer?, injectionPartials: MTLBuffer?,
        residualFP32: MTLBuffer? = nil
    ) throws {
        let partial = try tilePartialBytes(rows: rows)
        guard let kernels = prefillKernels, !active,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid caller-owned HC tile command")
        }
        let sizes =
            [
                rows * 10240 * 4, rows * 10240 * 2, rows * 320 * 4, rows * 320 * 2,
                rows * 10240 * 4, rows * 2560 * 4, rows * 80 * 4, rows * 16, rows * 16,
            ]
            + (s.capturedUp == nil ? [] : [rows * 10240 * 4])
        var spans = Array(zip(s.all, sizes))
        for (buffer, bytes) in [(downPartials, partial.down), (upPartials, partial.up), (injectionPartials, partial.injection)]
        where bytes > 0 {
            guard let buffer else { throw EngineError("Missing HC tile partial storage") }
            spans.append((buffer, bytes))
        }
        try validate(downWeights, width: 10240, outputs: 320)
        try validate(upWeights, width: 320, outputs: 10240)
        let reads = [
            residualBF16, normWeights, injectionWeights, downWeights.words, downWeights.scales,
            downWeights.biases, upWeights.words, upWeights.scales, upWeights.biases,
        ]
        guard spans.count >= sizes.count, spans.allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(spans.map { ObjectIdentifier($0.0) }).count == spans.count,
            reads.allSatisfy({ $0.device.registryID == device.registryID }),
            !spans.contains(where: { s in reads.contains(where: { $0 === s.0 }) }),
            residualBF16.length >= rows * 10240 * 2, normWeights.length >= 10240 * 2,
            injectionWeights.length >= 4 * 10240 * 4
        else { throw EngineError("Invalid HC tile buffer spans") }
        if let residualFP32 {
            guard fusedEntryEnabled, residualFP32.device.registryID == device.registryID, residualFP32.length >= rows * 10240 * 4,
                residualFP32 !== residualBF16, !reads.dropFirst().contains(where: { $0 === residualFP32 }),
                !spans.contains(where: { $0.0 === residualFP32 })
            else {
                throw EngineError("Invalid HC fused tile entry (flag off or aliased FP32 residual)")
            }
        }
        let downPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 10240, outputs: 320, simdMatrix: prefillSIMDMatrix)
        let upPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 320, outputs: 10240, simdMatrix: prefillSIMDMatrix)
        if let residualFP32 {
            try reductions.encodeHCFusedEntry(
                command, residual: residualFP32, weights: normWeights, residualBF16: residualBF16,
                output: s.norm, outputBF16: s.normBF16, rows: rows)
            fusedEntryTiles += 1
        } else {
            try reductions.encodeGroupedRMS(
                command, input: residualBF16, weights: normWeights, output: s.norm, rows: rows, width: 10240, groupSize: 2560)
            try handoff(command, pipeline: narrow, input: s.norm, output: s.normBF16, count: rows * 10240)
        }
        try kernels.encodePinnedProjection(
            command, input: s.normBF16, words: downWeights.words, scales: downWeights.scales,
            biases: downWeights.biases, output: s.down, partials: downPartials, plan: downPlan, simdMatrix: prefillSIMDMatrix)
        try handoff(command, pipeline: lookupNarrow, input: s.down, output: s.activation, count: rows * 320, table: activationTable)
        try kernels.encodePinnedProjection(
            command, input: s.activation, words: upWeights.words, scales: upWeights.scales,
            biases: upWeights.biases, output: s.rawUp, partials: upPartials, plan: upPlan, simdMatrix: prefillSIMDMatrix)
        try epilogue.encode(
            command, norm: s.norm, rawUp: s.rawUp, mixed: s.mixed, captured: s.capturedUp, exceptions: s.exceptions, rows: rows)
        try dense.encodeRows(
            command, input: s.norm, weights: injectionWeights, output: s.rawInjection, partials: injectionPartials,
            rows: rows, width: 10240, outputs: 4, outputBF16: true)
        try handoff(command, pipeline: lookupFloat, input: s.rawInjection, output: s.injection, count: rows * 4, table: injectionTable)
    }

    public func evaluate(
        residual: [Float], normWeights: MTLBuffer, downWeights: Projection, upWeights: Projection,
        injectionWeights: MTLBuffer?, capture: Bool = false, shouldContinue: () -> Bool = { true }
    ) throws -> Result {
        let rows = residual.count / 10240
        guard !residual.isEmpty, residual.count % 10240 == 0,
            rows == 1 || ((3...256).contains(rows) && prefillKernels != nil), !active,
            normWeights.device.registryID == device.registryID, normWeights.length >= 10240 * 2,
            injectionWeights == nil
                || (injectionWeights!.device.registryID == device.registryID && injectionWeights!.length >= 4 * 10240 * 4)
        else {
            throw EngineError("HC pipeline requires an enabled fixed tile, fixed weights and an idle owner")
        }
        try validate(downWeights, width: 10240, outputs: 320)
        try validate(upWeights, width: 320, outputs: 10240)
        guard shouldContinue() else { throw IOOperationCancelled() }
        active = true
        defer { active = false }
        if decodeFusion && rows == 1 {
            if injectionWeights == nil { decodeFusionUnfusedByDesign += 1 } else { decodeFusionFallbacks += 1 }
        }
        let downPlan =
            rows == 1
            ? nil : try prefillKernels!.pinnedProjectionPlan(rows: rows, width: 10240, outputs: 320, simdMatrix: prefillSIMDMatrix)
        let upPlan =
            rows == 1
            ? nil : try prefillKernels!.pinnedProjectionPlan(rows: rows, width: 320, outputs: 10240, simdMatrix: prefillSIMDMatrix)
        let densePlan = injectionWeights == nil ? nil : try dense.executionPlan(rows: rows, width: 10240, outputs: 4)
        return try autoreleasepool {
            // All fallible allocation/admission, including exact source split-K
            // partials, precedes commit. At M256 capture-off scratch is bounded
            // to about 37 MiB; capture adds one 10 MiB output, all page charged.
            let input = try allocate(rows * 10240 * 2)
            let norm = try allocate(rows * 10240 * 4)
            let normBF16 = try allocate(rows * 10240 * 2)
            let down = try allocate(rows * 320 * 4)
            let activation = try allocate(rows * 320 * 2)
            let up = try allocate(rows * 10240 * 4)
            let mixed = try allocate(rows * 2560 * 4)
            let exceptions = try allocate(rows * 80 * 4)
            let rawInjection = injectionWeights == nil ? nil : try allocate(rows * 4 * 4)
            let injection = injectionWeights == nil ? nil : try allocate(rows * 4 * 4)
            let capturedUp = capture ? try allocate(rows * 10240 * 4) : nil
            let downPartials = (downPlan?.partialBytes ?? 0) == 0 ? nil : try allocate(downPlan!.partialBytes)
            let upPartials = (upPlan?.partialBytes ?? 0) == 0 ? nil : try allocate(upPlan!.partialBytes)
            let densePartials = densePlan?.kind == .splitK ? try allocate(rows * 4 * densePlan!.splitKPartitions * 4) : nil
            let typed = input.contents().assumingMemoryBound(to: UInt16.self)
            for index in residual.indices { typed[index] = BF16.encode(residual[index]) }
            exceptions.contents().initializeMemory(as: UInt8.self, repeating: 0, count: rows * 80 * 4)
            guard let command = queue.makeCommandBuffer() else { throw EngineError("HC pipeline command allocation failed") }
            command.label = rows == 1 ? Self.policy : Self.prefillPolicy
            try reductions.encodeGroupedRMS(
                command, input: input, weights: normWeights, output: norm, rows: rows, width: 10240, groupSize: 2560)
            try handoff(command, pipeline: narrow, input: norm, output: normBF16, count: rows * 10240)
            if let downPlan {
                try prefillKernels!.encodePinnedProjection(
                    command, input: normBF16, words: downWeights.words,
                    scales: downWeights.scales, biases: downWeights.biases, output: down, partials: downPartials,
                    plan: downPlan, simdMatrix: prefillSIMDMatrix)
            } else {
                try project(command, input: normBF16, weights: downWeights, output: down, width: 10240, outputs: 320)
            }
            try handoff(command, pipeline: lookupNarrow, input: down, output: activation, count: rows * 320, table: activationTable)
            if let upPlan {
                try prefillKernels!.encodePinnedProjection(
                    command, input: activation, words: upWeights.words,
                    scales: upWeights.scales, biases: upWeights.biases, output: up, partials: upPartials,
                    plan: upPlan, simdMatrix: prefillSIMDMatrix)
            } else {
                try project(command, input: activation, weights: upWeights, output: up, width: 320, outputs: 10240)
            }
            try epilogue.encode(command, norm: norm, rawUp: up, mixed: mixed, captured: capturedUp, exceptions: exceptions, rows: rows)
            if let injectionWeights, let rawInjection, let injection {
                try dense.encodeRows(
                    command, input: norm, weights: injectionWeights, output: rawInjection, partials: densePartials,
                    rows: rows, width: 10240, outputs: 4, outputBF16: true)
                try handoff(command, pipeline: lookupFloat, input: rawInjection, output: injection, count: rows * 4, table: injectionTable)
            }
            guard shouldContinue() else { throw IOOperationCancelled() }
            let began = DispatchTime.now().uptimeNanoseconds
            command.commit()
            submitted += 1
            if rows > 1 {
                prefillSubmitted += 1
                lastPrefillRows = rows
                lastPrefillDown = downPlan
                lastPrefillUp = upPlan
                lastPrefillInjection = densePlan
            }
            try command.waitBounded()
            hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
            guard command.status == .completed else { throw EngineError.gpu("HC pipeline GPU failure", command.error) }
            completed += 1
            if rows > 1 { prefillCompleted += 1 }
            if command.gpuStartTime > 0, command.gpuEndTime >= command.gpuStartTime {
                gpuSeconds += command.gpuEndTime - command.gpuStartTime
            }
            guard shouldContinue() else { throw IOOperationCancelled() }
            let resolved = try epilogue.resolve(
                norm: norm, rawUp: up, mixed: mixed, captured: capturedUp, exceptions: exceptions, rows: rows)
            repaired += UInt64(resolved.repairedOutputs)
            func floats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
                Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
            }
            let activated =
                capture ? (0..<(rows * 320)).map { BF16.decode(activation.contents().assumingMemoryBound(to: UInt16.self)[$0]) } : nil
            return Result(
                mixed: resolved.mixed, injection: injection.map { floats($0, rows * 4) } ?? [],
                norm: capture ? floats(norm, rows * 10240) : nil, down: capture ? floats(down, rows * 320) : nil, activated: activated,
                rawUp: capture ? floats(up, rows * 10240) : nil, sigmoidUp: resolved.sigmoidUp,
                rawInjection: capture ? rawInjection.map { floats($0, rows * 4) } : nil, repairedOutputs: resolved.repairedOutputs)
        }
    }

    /// Bounded direct testing of the complete BF16 handoff domain using the
    /// same tables and kernels that the production one-row pipeline encodes.
    public func diagnosticHandoffs(_ input: [Float]) throws -> (activation: [Float], injection: [Float]) {
        guard !active, !input.isEmpty, input.count <= 10240 else {
            throw EngineError("HC unary diagnostic requires an idle owner and bounded tile")
        }
        active = true
        defer { active = false }
        return try autoreleasepool {
            let source = try allocate(input.count * 4)
            let activated = try allocate(input.count * 2)
            let injected = try allocate(input.count * 4)
            input.withUnsafeBytes { source.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            guard let command = queue.makeCommandBuffer() else { throw EngineError("HC unary diagnostic command unavailable") }
            try handoff(command, pipeline: lookupNarrow, input: source, output: activated, count: input.count, table: activationTable)
            try handoff(command, pipeline: lookupFloat, input: source, output: injected, count: input.count, table: injectionTable)
            command.commit()
            try command.waitBounded()
            guard command.status == .completed else { throw EngineError("HC unary diagnostic GPU failure") }
            return (
                (0..<input.count).map { BF16.decode(activated.contents().assumingMemoryBound(to: UInt16.self)[$0]) },
                Array(UnsafeBufferPointer(start: injected.contents().assumingMemoryBound(to: Float.self), count: input.count))
            )
        }
    }

    static let handoffSource = #"""
        // Source BF16 encoding, including quieting every NaN exactly like BF16.encode.
        inline ushort hc_decode_key(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
        // Elementwise handoffs, one thread per value. narrow: FP32 -> BF16 bits. lookup_narrow: BF16 bits of
        // table[BF16(x)] (the SiLU(x / 4) activation). lookup_float: the same lookup widened to FP32 (the
        // 2 * sigmoid(x / 4) injection gate). Integer-only, so math mode cannot change them.
        kernel void hc_decode_narrow(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]],
            constant uint &count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            if (index < count) y[index] = hc_decode_key(x[index]);
        }
        kernel void hc_decode_lookup_narrow(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]],
            constant uint &count [[buffer(2)]], device const ushort *table [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index < count) y[index] = table[hc_decode_key(x[index])];
        }
        kernel void hc_decode_lookup_float(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
            constant uint &count [[buffer(2)]], device const ushort *table [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index < count) y[index] = as_type<float>(uint(table[hc_decode_key(x[index])]) << 16);
        }
        """#
}
