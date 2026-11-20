import Foundation
import Metal

/// The engine's pinned-arithmetic Metal library for the dense (non-expert) affine 4-bit/64 layout. It compiles the
/// reference kernels below plus the optional exact families the configuration selects (tiled, BM32 and prefetch
/// projections, the GDN eight-row recurrence, the HC serial split reduce) into one library (the 64 x 64 MMA
/// projection compiles alone: `QuantizedTensorOpsKernel.pipeline`), then either runs
/// one kernel per synchronous command (`run`: checks, the reference layer path, small helpers) or encodes
/// projections, RMS and the GDN recurrence into a caller-owned command. The same source (`expertFusionMetalSource`)
/// also serves FusedHCDecode and FusedExpertExecution, which compiles it with the expert layout's defines.
/// Exactness: every kernel replays the pinned MLX/SlotStream rounding boundaries (explicit BF16 rounding, fixed K
/// order, fixed lane and SIMD reduction trees, no FMA contraction); each optional family states its own contract at
/// its flag below. MIT notices in THIRD_PARTY_NOTICES.md. Not thread-safe: one serving thread owns an instance.
public final class ReferenceKernels {
    // One-expert command batching shares the exact quantized kernel source.
    static var expertFusionMetalSource: String { source }
    static var quantizedBF16Preamble: String { preamble }
    package let device: MTLDevice
    private let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var decodeProjectionKernel: DecodeProjectionKernel?
    public private(set) var arithmetic: LayerUnaryArithmetic = .hostReference
    public let pinnedRMS: Bool
    public let quantizedTiled: Bool
    public let quantizedTiledBM32: Bool
    public let quantizedBM32MinRows: Int
    /// BM32 projections use the load-scheduled prefetch kernel (same arithmetic contract).
    public let quantizedTiledPrefetch: Bool
    public private(set) var prefetchTiledProjections: UInt64 = 0
    /// `quantized_mma_bf16`: dense BM32-prefetch projections with rows >= 64 and N % 64 == 0 use the
    /// BM64 x BN64 bfloat-fragment MMA kernel; the rest stay on the prefetch kernel and are counted.
    public let quantizedMMABF16: Bool
    public private(set) var quantizedMMAProjections: UInt64 = 0
    public private(set) var quantizedMMAFallbacks: UInt64 = 0
    /// `gdn_exact_kernels`: the encoded recurrence uses the eight-row register-tiled kernel when the
    /// q/k bindings are 16-byte aligned; otherwise the reference kernel runs and the fallback is counted.
    public let gdnExactKernels: Bool
    public private(set) var gdnExactRecurrenceTiles: UInt64 = 0
    public private(set) var gdnExactRecurrenceFallbacks: UInt64 = 0
    /// `hc_exact_fusion`: encoded split-K projections with fewer than 32 partitions reduce with
    /// one thread per output (`qmm_split_reduce_serial4`, four outputs per thread when rows * outputs % 4 == 0)
    /// replaying the current kernel's lane order exactly; 32 or more partitions keep `qmm_pinned_split_reduce`
    /// (its simd_sum order) and are counted as fallbacks. The synchronous route is unchanged. The owning
    /// FusedHCDecode also enables its fused tile entry from this flag.
    public let hcExactFusion: Bool
    public private(set) var serialSplitReduces: UInt64 = 0
    public private(set) var serialSplitReduceFallbacks: UInt64 = 0

    /// Reduction algebra remains safe by default. Pinned unary tables use a
    /// separate library with safe algebra and fast elementary functions.
    public convenience init(
        fastMath: Bool = false, fastFunctions: Bool? = nil,
        pinnedUnary: Bool = false, ledger: MemoryLedger? = nil, pinnedRMS: Bool = false, quantizedTiled: Bool = false,
        quantizedTiledBM32: Bool = false, quantizedBM32MinRows: Int = 32,
        quantizedTiledPrefetch: Bool = false, quantizedMMABF16: Bool = false, gdnExactKernels: Bool = false,
        hcExactFusion: Bool = false
    ) throws {
        guard (1...256).contains(quantizedBM32MinRows), !quantizedTiledBM32 || quantizedTiled else {
            throw EngineError("BM32 requires quantized_tiled and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard !quantizedTiledPrefetch || quantizedTiledBM32 else {
            throw EngineError("quantized_tiled_prefetch requires quantized_tiled_bm32", code: "invalid_configuration")
        }
        guard !quantizedMMABF16 || quantizedTiledPrefetch else {
            throw EngineError("quantized_mma_bf16 requires quantized_tiled_prefetch", code: "invalid_configuration")
        }
        if pinnedUnary && ledger == nil { throw EngineError("Pinned unary tables require a memory ledger") }
        try self.init(
            fastMath: fastMath, fastFunctions: fastFunctions ?? fastMath,
            source: Self.source + (quantizedTiled ? GroupedExpertKernel.source : "")
                + (quantizedTiledBM32 ? QuantizedTiled32Kernel.source : "")
                + (quantizedTiledPrefetch ? QuantizedTiled32PrefetchKernel.source : "")
                + (gdnExactKernels ? GDNExactKernels.recurrenceSource : "")
                + (hcExactFusion ? "\n" + HCExactFusionKernels.splitReduceSource : ""),
            pinnedRMS: pinnedRMS, quantizedTiled: quantizedTiled, quantizedTiledBM32: quantizedTiledBM32,
            quantizedBM32MinRows: quantizedBM32MinRows,
            quantizedTiledPrefetch: quantizedTiledPrefetch, quantizedMMABF16: quantizedMMABF16, gdnExactKernels: gdnExactKernels,
            hcExactFusion: hcExactFusion)
        if pinnedUnary { arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger!) }
    }

    static func pinnedUnaryKernels() throws -> ReferenceKernels {
        try ReferenceKernels(fastMath: false, fastFunctions: true, source: Self.preamble + Self.unarySource)
    }

    private init(
        fastMath: Bool, fastFunctions: Bool, source: String, pinnedRMS: Bool = false, quantizedTiled: Bool = false,
        quantizedTiledBM32: Bool = false, quantizedBM32MinRows: Int = 32,
        quantizedTiledPrefetch: Bool = false, quantizedMMABF16: Bool = false, gdnExactKernels: Bool = false,
        hcExactFusion: Bool = false
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Metal reference runtime unavailable")
        }
        try ModelContract.validate(device: device)
        self.device = device
        self.queue = queue
        self.pinnedRMS = pinnedRMS
        self.quantizedTiled = quantizedTiled
        self.quantizedTiledBM32 = quantizedTiledBM32
        self.quantizedBM32MinRows = quantizedBM32MinRows
        self.quantizedTiledPrefetch = quantizedTiledPrefetch
        self.quantizedMMABF16 = quantizedMMABF16
        self.gdnExactKernels = gdnExactKernels
        self.hcExactFusion = hcExactFusion
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = fastMath ? .fast : .safe
        options.mathFloatingPointFunctions = fastFunctions ? .fast : .precise
        library = try device.makeLibrary(source: ModelLayout.metalDefines + source, options: options)
        if quantizedMMABF16 {
            pipelines[QuantizedMMABF16Kernel.name] = try QuantizedTensorOpsKernel.pipeline(
                device: device, defines: ModelLayout.metalDefines, options: options)
        }
    }

    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty,
            let result = values.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
        else { throw EngineError("Reference buffer allocation failed or empty input") }
        return result
    }

    func run(
        name: String, inputs: [MTLBuffer], count: Int, threads: Int, offsets: [Int]? = nil,
        threadgroupWidth: Int? = nil, groupCount: Int? = nil, threadgroups: MTLSize? = nil,
        output suppliedOutput: MTLBuffer? = nil, onCommit: (() -> Void)? = nil,
        onRetire: (() -> Void)? = nil
    ) throws -> MTLBuffer {
        let bindingOffsets = offsets ?? [Int](repeating: 0, count: inputs.count)
        guard bindingOffsets.count == inputs.count,
            zip(inputs, bindingOffsets).allSatisfy({ $1 >= 0 && $1 <= $0.length && $1 % 4 == 0 })
        else {
            throw EngineError("Invalid Metal buffer binding offsets")
        }
        guard count > 0, count <= device.maxBufferLength / MemoryLayout<UInt32>.stride else {
            throw EngineError("Invalid reference output extent")
        }
        if let suppliedOutput {
            guard suppliedOutput.device.registryID == device.registryID,
                suppliedOutput.storageMode == .shared, suppliedOutput.length >= count * 4
            else {
                throw EngineError("Invalid admitted reference output buffer")
            }
        }
        guard let output = suppliedOutput ?? device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
        else {
            throw EngineError("Invalid reference dispatch or Metal allocation failed")
        }
        let pipeline: MTLComputePipelineState
        if let cached = pipelines[name] {
            pipeline = cached
        } else {
            guard let function = library.makeFunction(name: name) else { throw EngineError("Missing reference kernel \(name)") }
            pipeline = try device.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
        }
        let groupWidth =
            threadgroupWidth
            ?? (groupCount == nil ? min(64, pipeline.maxTotalThreadsPerThreadgroup) : pipeline.maxTotalThreadsPerThreadgroup)
        guard groupWidth > 0, groupWidth <= pipeline.maxTotalThreadsPerThreadgroup,
            groupCount == nil || groupCount! > 0
        else { throw EngineError("Invalid reference threadgroup geometry") }
        if let threadgroups {
            guard groupCount == nil, threadgroups.width > 0, threadgroups.height > 0, threadgroups.depth > 0,
                [64, 128].contains(groupWidth), pipeline.threadExecutionWidth == 32,
                pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Invalid tiled quantized threadgroup geometry or resource bound")
            }
        }
        encoder.setComputePipelineState(pipeline)
        for (index, input) in inputs.enumerated() { encoder.setBuffer(input, offset: bindingOffsets[index], index: index) }
        encoder.setBuffer(output, offset: 0, index: inputs.count)
        if let threadgroups {
            encoder.dispatchThreadgroups(
                threadgroups,
                threadsPerThreadgroup: MTLSize(width: 32, height: groupWidth / 32, depth: 1))
        } else if let groupCount {
            encoder.dispatchThreadgroups(
                MTLSize(width: groupCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
        } else {
            encoder.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
        }
        encoder.endEncoding()
        onCommit?()
        command.commit()
        try command.waitBounded()
        onRetire?()
        guard command.status == .completed else { throw EngineError.gpu("Reference dispatch failed", command.error) }
        return output
    }

    func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    public func qmatvec(
        x: [Float], words: [UInt32], scales: [UInt16], biases: [UInt16], groupSize: Int = ModelLayout.group,
        dequantizedWeightBF16: Bool = false
    ) throws -> [Float] {
        guard [32, 64].contains(groupSize), !x.isEmpty, x.count % groupSize == 0,
            words.count % (x.count / 8) == 0
        else { throw EngineError("Invalid affine matvec geometry") }
        let rows = words.count / (x.count / 8)
        guard rows > 0, scales.count == rows * (x.count / groupSize), biases.count == scales.count else {
            throw EngineError("Affine scale/bias geometry mismatch")
        }
        let output = try run(
            name: "qmatvec_reference",
            inputs: [
                buffer(x), buffer(words), buffer(scales), buffer(biases),
                buffer([UInt32(x.count), UInt32(groupSize), dequantizedWeightBF16 ? 1 : 0]),
            ], count: rows, threads: rows)
        return floats(output, count: rows)
    }

    /// Reproduces pinned QMM/QMV rounding boundaries and shape dispatch.
    /// Scalar partition dot products are a reference; the SIMD-matrix kernels' parity with them is checked separately.
    public func qlinearPinnedBF16(
        x: [Float], width: Int, words: [UInt32], scales: [UInt16], biases: [UInt16],
        groupSize: Int = ModelLayout.group, simdMatrix: Bool = false, groupedExpert: Bool = false
    ) throws -> (output: [Float], schedule: QuantizedSchedule, implementation: String) {
        guard width > 0, width % 8 == 0, !x.isEmpty, x.count % width == 0,
            words.count % (width / 8) == 0
        else { throw EngineError("Invalid pinned quantized input") }
        let outputs = words.count / (width / 8)
        _ = try QuantizedSchedule(rows: x.count / width, width: width, outputs: outputs, groupSize: groupSize, groupedExpert: groupedExpert)
        guard scales.count == outputs * (width / groupSize), biases.count == scales.count else {
            throw EngineError("Invalid pinned quantized metadata")
        }
        return try qlinearPinnedBF16(
            x: x, width: width, outputs: outputs,
            words: buffer(words), scales: buffer(scales), biases: buffer(biases),
            groupSize: groupSize, simdMatrix: simdMatrix, groupedExpert: groupedExpert)
    }

    /// Buffer-native fixed-weight projection. Each immutable tensor can have a
    /// distinct Metal backing and offset. Actual input rows select the pinned
    /// QMM/QMV schedule; no weight data passes through host arrays here.
    /// The final GPU command is joined before success or failure is returned.
    public func qlinearPinnedBF16(
        x: [Float], width: Int, outputs: Int,
        words: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
        wordOffset: Int = 0, scaleOffset: Int = 0, biasOffset: Int = 0,
        groupSize: Int = ModelLayout.group, simdMatrix: Bool = false, groupedExpert: Bool = false
    )
        throws -> (output: [Float], schedule: QuantizedSchedule, implementation: String)
    {
        guard width > 0, !x.isEmpty, x.count % width == 0 else { throw EngineError("Invalid resident quantized input") }
        let rows = x.count / width
        let schedule = try QuantizedSchedule(rows: rows, width: width, outputs: outputs, groupSize: groupSize, groupedExpert: groupedExpert)
        let wordBytes = outputs * (width / 8) * MemoryLayout<UInt32>.stride
        let metadataBytes = outputs * (width / groupSize) * MemoryLayout<UInt16>.stride
        for (tensor, offset, size) in [
            (words, wordOffset, wordBytes), (scales, scaleOffset, metadataBytes), (biases, biasOffset, metadataBytes),
        ] {
            guard tensor.device.registryID == device.registryID, offset >= 0, offset % 4 == 0,
                offset <= tensor.length, size <= tensor.length - offset
            else {
                throw EngineError("Resident quantized tensor has an invalid device, alignment or byte span")
            }
        }
        // The synchronous readout/ordinary-linear entry must use the same
        // activation-reuse route as encoded layer projections. Offset/grouped
        // and other source schedules retain their original implementation.
        if quantizedTiled && !groupedExpert && groupSize == ModelLayout.group && rows == 1 && schedule.kind == "qmv_fast" && wordOffset == 0
            && scaleOffset == 0 && biasOffset == 0 && [words, scales, biases].allSatisfy({ $0.storageMode == .shared })
            && DecodeProjectionKernel.supports(width: width, outputs: outputs)
        {
            let input = try buffer(x.map(BF16.encode))
            guard let output = device.makeBuffer(length: outputs * 4, options: .storageModeShared),
                let command = queue.makeCommandBuffer()
            else {
                throw EngineError("Synchronous tile8 projection allocation failed")
            }
            try decodeProjection().encode(
                command, input: input, width: width,
                first: .init(words: words, scales: scales, biases: biases, rows: outputs), firstOutput: output)
            command.commit()
            try command.waitBounded()
            guard command.status == .completed else {
                throw EngineError.gpu("Synchronous tile8 projection failed", command.error)
            }
            return (floats(output, count: outputs), schedule, "pinned_qmv_tile8_shared_activation")
        }
        let shape = [
            width, outputs, rows, groupSize, schedule.splitK, schedule.partitionSize,
            schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        let parameters = try buffer(shape)
        let inputs = try [buffer(x.map(BF16.encode)), words, scales, biases, parameters]
        let bindingOffsets = [0, wordOffset, scaleOffset, biasOffset, 0]
        let result: MTLBuffer
        let implementation: String
        if simdMatrix && schedule.kind.hasPrefix("qmm") {
            let tiled =
                quantizedTiled && !groupedExpert && groupSize == ModelLayout.group && outputs >= 32 && outputs % 32 == 0
                && schedule.partitionSize % 64 == 0
            let partials: MTLBuffer
            if tiled {
                let bm32 = quantizedTiledBM32 && rows >= quantizedBM32MinRows
                // The prefetch and MMA kernels' vector loads need 8-byte aligned weight bindings.
                let prefetch = bm32 && quantizedTiledPrefetch && bindingOffsets.allSatisfy { $0 % 8 == 0 }
                let mma = prefetch && quantizedMMABF16 && Self.mmaEligible(rows: rows, outputs: outputs)
                if mma {
                    quantizedMMAProjections += 1
                } else {
                    if prefetch { prefetchTiledProjections += 1 }
                    if quantizedMMABF16 && bm32 { quantizedMMAFallbacks += 1 }
                }
                let blockRows = mma ? 64 : bm32 ? 32 : 16
                let blockOutputs = mma ? 64 : 32
                implementation =
                    mma
                    ? Self.mmaImplementation
                    : bm32 ? "simdgroup_tiled32x32_k32_bf16_partitions" : "simdgroup_tiled16x32_k32_bf16_partitions"
                partials = try run(
                    name: mma
                        ? QuantizedMMABF16Kernel.name
                        : prefetch
                            ? "qmm_pinned_tiled32_prefetch_partitions"
                            : bm32 ? "qmm_pinned_tiled32_partitions" : "qmm_pinned_tiled_partitions", inputs: inputs,
                    count: schedule.splitK * rows * outputs, threads: 0, offsets: bindingOffsets,
                    threadgroupWidth: bm32 ? 128 : 64,
                    threadgroups: MTLSize(width: outputs / blockOutputs, height: (rows + blockRows - 1) / blockRows, depth: schedule.splitK)
                )
            } else {
                implementation = groupedExpert ? "grouped_expert_simdgroup_float8x8_k8_no_split" : "simdgroup_float8x8_k8"
                let tiles = ((rows + 7) / 8) * ((outputs + 7) / 8)
                partials = try run(
                    name: "qmm_pinned_simd_partitions", inputs: inputs,
                    count: schedule.splitK * rows * outputs, threads: schedule.splitK * tiles * 32, offsets: bindingOffsets)
            }
            if schedule.splitK == 1 {
                result = partials
            } else {
                result = try run(
                    name: "qmm_pinned_split_reduce", inputs: [partials, parameters],
                    count: rows * outputs, threads: rows * outputs * 32)
            }
        } else {
            implementation =
                groupedExpert
                ? "grouped_expert_scalar_float32_k_order_no_split"
                : (schedule.kind.hasPrefix("qmm") ? "scalar_float32_k_order" : "pinned_qmv_lane_schedule")
            result = try run(
                name: schedule.kind.hasPrefix("qmm") ? "qmm_pinned_reference" : "qmv_pinned_reference",
                inputs: inputs, count: rows * outputs, threads: rows * outputs * 32, offsets: bindingOffsets)
        }
        return (floats(result, count: rows * outputs), schedule, implementation)
    }

    static let mmaImplementation = "simdgroup_mma_bf16_64x64_k32_bf16_partitions"
    /// The BM64 x BN64 tile pays off from 64 rows and needs N % 64 == 0 (true for every production shape).
    static func mmaEligible(rows: Int, outputs: Int) -> Bool { rows >= 64 && outputs % 64 == 0 }

    /// Host dispatch plan for an ordinary affine4 projection. The optional HC
    /// command pipeline uses the exact same actual-row policies as qlinear.
    public struct PinnedProjectionPlan: Codable, Sendable {
        public let rows, width, outputs: Int
        public let schedule: QuantizedSchedule
        public let implementation, kernel: String
        public let tileRows, threadgroupWidth: Int
        public let partialBytes: Int
    }
    public func pinnedProjectionPlan(rows: Int, width: Int, outputs: Int, simdMatrix: Bool) throws -> PinnedProjectionPlan {
        guard (1...256).contains(rows) else { throw EngineError("Encoded projection rows exceed the bounded tile") }
        let schedule = try QuantizedSchedule(rows: rows, width: width, outputs: outputs, groupSize: ModelLayout.group)
        let implementation: String
        let kernel: String
        let tileRows: Int
        let groupWidth: Int
        let partialBytes: Int
        if simdMatrix && schedule.kind.hasPrefix("qmm") {
            let tiled = quantizedTiled && outputs >= 32 && outputs % 32 == 0 && schedule.partitionSize % 64 == 0
            if tiled {
                let bm32 = quantizedTiledBM32 && rows >= quantizedBM32MinRows
                let mma = bm32 && quantizedTiledPrefetch && quantizedMMABF16 && Self.mmaEligible(rows: rows, outputs: outputs)
                implementation =
                    mma
                    ? Self.mmaImplementation
                    : bm32 ? "simdgroup_tiled32x32_k32_bf16_partitions" : "simdgroup_tiled16x32_k32_bf16_partitions"
                kernel =
                    mma
                    ? QuantizedMMABF16Kernel.name
                    : bm32
                        ? (quantizedTiledPrefetch ? "qmm_pinned_tiled32_prefetch_partitions" : "qmm_pinned_tiled32_partitions")
                        : "qmm_pinned_tiled_partitions"
                tileRows = mma ? 64 : bm32 ? 32 : 16
                groupWidth = bm32 ? 128 : 64
            } else {
                implementation = "simdgroup_float8x8_k8"
                kernel = "qmm_pinned_simd_partitions"
                tileRows = 8
                groupWidth = 64
            }
            partialBytes = schedule.splitK > 1 ? schedule.splitK * rows * outputs * 4 : 0
        } else if quantizedTiled
            && ((rows == 1 && schedule.kind == "qmv_fast" && DecodeProjectionKernel.supports(width: width, outputs: outputs))
                || (rows == 2 && schedule.kind.hasPrefix("qmv")
                    && DecodeProjectionKernel.supports(width: width, outputs: outputs, inputRows: 2)))
        {
            // Two rows (the MTP verify step) share each weight load; each row's arithmetic is the one-row kernel's.
            implementation = "pinned_qmv_tile8_shared_activation"
            kernel = "projection_tile8"
            tileRows = rows
            groupWidth = 256
            partialBytes = 0
        } else {
            implementation = schedule.kind.hasPrefix("qmm") ? "scalar_float32_k_order" : "pinned_qmv_lane_schedule"
            kernel = schedule.kind.hasPrefix("qmm") ? "qmm_pinned_reference" : "qmv_pinned_reference"
            tileRows = 0
            groupWidth = 64
            partialBytes = 0
        }
        return PinnedProjectionPlan(
            rows: rows, width: width, outputs: outputs, schedule: schedule,
            implementation: implementation, kernel: kernel, tileRows: tileRows, threadgroupWidth: groupWidth, partialBytes: partialBytes)
    }

    /// Existing projection shaders and geometry, encoded into a caller-owned
    /// command. Output/partials are admitted before entry and stay owned through
    /// final GPU completion. No arithmetic or partial narrowing is combined.
    package func encodePinnedProjection(
        _ command: MTLCommandBuffer, input: MTLBuffer, words: MTLBuffer,
        scales: MTLBuffer, biases: MTLBuffer, output: MTLBuffer,
        partials: MTLBuffer?, plan: PinnedProjectionPlan, simdMatrix: Bool
    ) throws {
        let actual = try pinnedProjectionPlan(rows: plan.rows, width: plan.width, outputs: plan.outputs, simdMatrix: simdMatrix)
        guard actual.schedule == plan.schedule, actual.kernel == plan.kernel, actual.implementation == plan.implementation,
            actual.partialBytes == plan.partialBytes, actual.tileRows == plan.tileRows, actual.threadgroupWidth == plan.threadgroupWidth,
            command.commandQueue.device.registryID == device.registryID,
            [input, words, scales, biases, output].allSatisfy({ $0.device.registryID == device.registryID && $0.storageMode == .shared }),
            input.length >= plan.rows * plan.width * 2, words.length >= plan.outputs * plan.width / 8 * 4,
            scales.length >= plan.outputs * plan.width / ModelLayout.group * 2,
            biases.length >= plan.outputs * plan.width / ModelLayout.group * 2,
            output.length >= plan.rows * plan.outputs * 4,
            plan.partialBytes == 0
                || (partials != nil && partials!.device.registryID == device.registryID && partials!.storageMode == .shared
                    && partials!.length >= plan.partialBytes)
        else {
            throw EngineError("Encoded affine4 plan or buffer differs from source contract")
        }
        if plan.kernel == "projection_tile8" {
            try decodeProjection().encode(
                command, input: input, width: plan.width,
                first: .init(words: words, scales: scales, biases: biases, rows: plan.outputs), firstOutput: output, inputRows: plan.rows)
            return
        }
        let schedule = plan.schedule
        let shape = [
            plan.width, plan.outputs, plan.rows, ModelLayout.group, schedule.splitK, schedule.partitionSize,
            schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        func encoder(_ name: String, inputs: [MTLBuffer], result: MTLBuffer) throws -> (MTLComputeCommandEncoder, MTLComputePipelineState) {
            let state = try projectionPipeline(named: name)
            guard let encoder = command.makeComputeCommandEncoder() else { throw EngineError("Encoded projection encoder unavailable") }
            encoder.setComputePipelineState(state)
            for (index, buffer) in inputs.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
            shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: inputs.count) }
            encoder.setBuffer(result, offset: 0, index: inputs.count + 1)
            return (encoder, state)
        }
        let target = plan.partialBytes == 0 ? output : partials!
        let (first, state) = try encoder(plan.kernel, inputs: [input, words, scales, biases], result: target)
        if plan.kernel == "qmm_pinned_tiled_partitions" || plan.kernel == "qmm_pinned_tiled32_partitions"
            || plan.kernel == "qmm_pinned_tiled32_prefetch_partitions" || plan.kernel == QuantizedMMABF16Kernel.name
        {
            let mma = plan.kernel == QuantizedMMABF16Kernel.name
            if mma {
                quantizedMMAProjections += 1
            } else if plan.kernel == "qmm_pinned_tiled32_prefetch_partitions" {
                prefetchTiledProjections += 1
                if quantizedMMABF16 { quantizedMMAFallbacks += 1 }
            }
            guard state.threadExecutionWidth == 32, state.maxTotalThreadsPerThreadgroup >= plan.threadgroupWidth,
                state.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
                !mma || (plan.tileRows == 64 && plan.outputs % 64 == 0)
            else {
                first.endEncoding()
                throw EngineError("Encoded tiled projection resource bound differs")
            }
            first.dispatchThreadgroups(
                MTLSize(
                    width: plan.outputs / (mma ? 64 : 32), height: (plan.rows + plan.tileRows - 1) / plan.tileRows, depth: schedule.splitK),
                threadsPerThreadgroup: MTLSize(width: 32, height: plan.threadgroupWidth / 32, depth: 1))
        } else {
            let threads =
                plan.kernel == "qmm_pinned_simd_partitions"
                ? schedule.splitK * ((plan.rows + 7) / 8) * ((plan.outputs + 7) / 8) * 32
                : plan.rows * plan.outputs * 32
            first.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, state.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        }
        first.endEncoding()
        if plan.partialBytes > 0 {
            if hcExactFusion && schedule.splitK < 32 {
                // Same partials, shape and output; one thread replays one output's (or four outputs') lane order.
                let count = plan.rows * plan.outputs
                let vector = count % 4 == 0
                let (reduce, reducer) = try encoder(
                    vector ? HCExactFusionKernels.splitReduceVectorName : HCExactFusionKernels.splitReduceName,
                    inputs: [target], result: output)
                reduce.dispatchThreads(
                    MTLSize(width: vector ? count / 4 : count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, reducer.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                reduce.endEncoding()
                serialSplitReduces += 1
            } else {
                if hcExactFusion { serialSplitReduceFallbacks += 1 }
                let (reduce, reducer) = try encoder("qmm_pinned_split_reduce", inputs: [target], result: output)
                reduce.dispatchThreads(
                    MTLSize(width: plan.rows * plan.outputs * 32, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, reducer.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                reduce.endEncoding()
            }
        }
    }
    private func decodeProjection() throws -> DecodeProjectionKernel {
        if let cached = decodeProjectionKernel { return cached }
        let kernel = try DecodeProjectionKernel(device: device)
        decodeProjectionKernel = kernel
        return kernel
    }
    /// Fuses two row-one projections sharing an input, preserving separate
    /// admitted output buffers. Returns false for the original reference path.
    package func encodePairedDecodeProjection(
        _ command: MTLCommandBuffer, input: MTLBuffer, width: Int,
        first: DecodeProjectionKernel.Weights, firstOutput: MTLBuffer,
        second: DecodeProjectionKernel.Weights, secondOutput: MTLBuffer
    ) throws -> Bool {
        guard quantizedTiled, DecodeProjectionKernel.supports(width: width, outputs: first.rows),
            DecodeProjectionKernel.supports(width: width, outputs: second.rows)
        else { return false }
        try decodeProjection().encode(
            command, input: input, width: width, first: first, firstOutput: firstOutput,
            second: second, secondOutput: secondOutput)
        return true
    }
    private func projectionPipeline(named name: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] { return cached }
        guard let function = library.makeFunction(name: name) else { throw EngineError("Missing reference kernel \(name)") }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }

    /// Expert-slab QMV: every input row retains the pinned ONE-ROW dispatch
    /// schedule, even when several rows share this command. The weight slab is
    /// bound directly at validated offsets and never copied through host Data.
    /// Return/throw occurs only after a submitted command has completed, so the
    /// caller can retire its consumer lease afterward on either path.
    public func qmvPinnedBF16(
        x: [Float], width: Int, outputs: Int, weights: MTLBuffer,
        weightOffset: Int, scaleOffset: Int, biasOffset: Int,
        groupSize: Int = ModelLayout.group
    ) throws -> [Float] {
        guard width > 0, !x.isEmpty, x.count % width == 0, x.count / width <= 256,
            weights.device.registryID == device.registryID
        else { throw EngineError("Invalid expert QMV input or device") }
        let schedule = try QuantizedSchedule(rows: 1, width: width, outputs: outputs, groupSize: groupSize)
        guard schedule.kind.hasPrefix("qmv") else { throw EngineError("Expert projection requires the one-row QMV schedule") }
        let wordBytes = outputs * (width / 8) * MemoryLayout<UInt32>.stride
        let metadataBytes = outputs * (width / groupSize) * MemoryLayout<UInt16>.stride
        for (offset, size) in [(weightOffset, wordBytes), (scaleOffset, metadataBytes), (biasOffset, metadataBytes)] {
            guard offset >= 0, offset % 4 == 0, offset <= weights.length, size <= weights.length - offset else {
                throw EngineError("Expert QMV projection span exceeds its Metal slab")
            }
        }
        let rows = x.count / width
        let shape = [
            width, outputs, rows, groupSize, schedule.splitK, schedule.partitionSize,
            schedule.valuesPerLane, schedule.laneCount,
        ].map(UInt32.init)
        let result = try run(
            name: "qmv_pinned_reference",
            inputs: [buffer(x.map(BF16.encode)), weights, weights, weights, buffer(shape)],
            count: rows * outputs, threads: rows * outputs * 32,
            offsets: [0, weightOffset, scaleOffset, biasOffset, 0])
        return floats(result, count: rows * outputs)
    }

    public func rmsnorm(
        x: [Float], weights: [Float], epsilon: Float = 1e-6, groupSize: Int? = nil,
        pinnedReduction: Bool? = nil
    ) throws -> [Float] {
        guard !weights.isEmpty, x.count % weights.count == 0, epsilon > 0, epsilon.isFinite else {
            throw EngineError("Invalid RMS norm geometry")
        }
        let width = weights.count
        let group = groupSize ?? width
        guard group > 0, width % group == 0 else { throw EngineError("Invalid RMS norm group") }
        if (pinnedReduction ?? pinnedRMS) && groupSize == nil {
            let groupThreads: Int? = group <= 4096 ? ((group + 127) / 128) * 32 : nil
            let output = try run(
                name: "rmsnorm_pinned",
                inputs: [
                    buffer(x), buffer(weights),
                    buffer([UInt32(width), UInt32(group)]), buffer([epsilon]),
                ], count: x.count, threads: 0,
                threadgroupWidth: groupThreads, groupCount: x.count / group)
            return floats(output, count: x.count)
        }
        let output = try run(
            name: "rmsnorm_reference",
            inputs: [
                buffer(x), buffer(weights),
                buffer([UInt32(width), UInt32(group), groupSize == nil ? 0 : 1]), buffer([epsilon]),
            ],
            count: x.count, threads: x.count / group)
        return floats(output, count: x.count)
    }

    /// The unchanged pinned RMS128/256 QSA reduction. Groups are vectors, not
    /// token rows; QSA can have 24 such vectors per token. Buffers stay owned
    /// by the enclosing unpublished transaction until its command joins.
    package func encodeQSARMS(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, groups: Int, width: Int
    ) throws {
        guard pinnedRMS, [128, 256].contains(width), (1...(256 * 24)).contains(groups),
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            input !== weights, input !== output, weights !== output
        else { throw EngineError("Invalid QSA RMS geometry/command") }
        for (buffer, bytes) in [(input, groups * width * 4), (weights, width * 4), (output, groups * width * 4)] {
            guard buffer.device.registryID == device.registryID, buffer.length >= bytes else {
                throw EngineError("QSA RMS span exceeds buffer")
            }
        }
        let pipeline = try projectionPipeline(named: "rmsnorm_pinned")
        let threads = ((width + 127) / 128) * 32
        guard pipeline.maxTotalThreadsPerThreadgroup >= threads, let e = command.makeComputeCommandEncoder() else {
            throw EngineError("QSA RMS encoder unavailable")
        }
        e.setComputePipelineState(pipeline)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(weights, offset: 0, index: 1)
        let shape = [UInt32(width), UInt32(width)]
        var epsilon: Float = 1e-6
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        e.setBytes(&epsilon, length: 4, index: 3)
        e.setBuffer(output, offset: 0, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        e.endEncoding()
    }

    public func rope(x: [Float], width: Int, position: Int, dimensions: Int = 64) throws -> [Float] {
        guard width > 0, !x.isEmpty, x.count % width == 0, dimensions > 0,
            dimensions <= width, dimensions % 2 == 0, position >= 0, position < ModelContract.maximumContext
        else {
            throw EngineError("Invalid RoPE geometry or position")
        }
        var cosines: [Float] = []
        var sines: [Float] = []
        for index in 0..<(dimensions / 2) {
            let inv = powf(10_000_000, -Float(index * 2) / Float(dimensions))
            let angle = Float(position) * inv
            cosines.append(BF16.decode(BF16.encode(cosf(angle))))
            sines.append(BF16.decode(BF16.encode(sinf(angle))))
        }
        let output = try run(
            name: "rope_reference",
            inputs: [
                buffer(x), buffer(cosines), buffer(sines),
                buffer([UInt32(width), UInt32(dimensions)]),
            ], count: x.count, threads: x.count)
        return floats(output, count: x.count)
    }

    /// Independent chronological rows, with the exact scalar path's host
    /// trigonometric table and BF16 arithmetic. Only dispatch/addressing change.
    public func ropeBatch(x: [Float], width: Int, heads: Int, positions: [Int], dimensions: Int = 64) throws -> [Float] {
        guard [128, 256].contains(width), [1, 2, 4, 24].contains(heads),
            (1...256).contains(positions.count), dimensions == 64,
            x.count == positions.count * heads * width,
            positions.allSatisfy({ (0..<ModelContract.maximumContext).contains($0) })
        else {
            throw EngineError("Invalid batched RoPE geometry or position")
        }
        var cosines: [Float] = []
        var sines: [Float] = []
        cosines.reserveCapacity(positions.count * dimensions / 2)
        sines.reserveCapacity(positions.count * dimensions / 2)
        for position in positions {
            for index in 0..<(dimensions / 2) {
                let inv = powf(10_000_000, -Float(index * 2) / Float(dimensions))
                let angle = Float(position) * inv
                cosines.append(BF16.decode(BF16.encode(cosf(angle))))
                sines.append(BF16.decode(BF16.encode(sinf(angle))))
            }
        }
        let output = try run(
            name: "rope_batched",
            inputs: [
                buffer(x), buffer(cosines), buffer(sines),
                buffer([UInt32(width), UInt32(dimensions), UInt32(heads)]),
            ], count: x.count, threads: x.count)
        return floats(output, count: x.count)
    }

    public func topIndices(scores: [Float], count: Int) throws -> [UInt32] {
        guard !scores.isEmpty, scores.count <= 65_536, count >= 1, count <= min(scores.count, 512) else {
            throw EngineError("Invalid selection domain")
        }
        let output = try run(
            name: "top_indices_reference",
            inputs: [
                buffer(scores),
                buffer([UInt32(scores.count), UInt32(count)]),
            ], count: count, threads: 1)
        return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: count))
    }

    public func silu(x: [Float]) throws -> [Float] {
        let output = try run(name: "silu_reference", inputs: [buffer(x)], count: x.count, threads: x.count)
        return floats(output, count: x.count)
    }

    /// The pinned unary operations `bf16_unary_pinned` evaluates on BF16 inputs (raw values are the kernel's selector).
    public enum BF16Unary: UInt32 {
        case sigmoid = 0
        case silu = 1
        case softplus = 2
        case sigmoidFP32 = 3
    }

    /// Pinned unary primitive dtype boundaries. `native` independently checks
    /// MSL integer/bfloat promotion against the explicit rounding schedule.
    public func pinnedUnary(x: [Float], operation: BF16Unary, native: Bool = false) throws -> [Float] {
        let output = try run(
            name: "bf16_unary_pinned",
            inputs: [
                buffer(x.map(BF16.encode)),
                buffer([operation.rawValue, native ? 1 : 0]),
            ], count: x.count, threads: x.count)
        return floats(output, count: x.count)
    }

    /// Existing recurrence encoder for a caller-owned, precharged command.
    /// Uses this instance's unchanged library, math flags and dispatch geometry.
    func encodeGDNRecurrence(
        _ command: MTLCommandBuffer, inputs: [MTLBuffer], initialOffset: Int,
        output: MTLBuffer, tokens: Int, contractUpdates: Bool, qkvOffsets: [Int] = [0, 0, 0]
    ) throws {
        let stateCount = 48 * 128 * 128
        guard (1...256).contains(tokens), inputs.count == 7, qkvOffsets.count == 3, qkvOffsets.allSatisfy({ $0 >= 0 && $0 % 4 == 0 }),
            initialOffset >= 0, initialOffset % 4 == 0
        else {
            throw EngineError("Invalid encoded GDN recurrence geometry")
        }
        let sizes = [
            tokens * 2048 * 4, tokens * 2048 * 4, tokens * 6144 * 4,
            tokens * 48 * 4, tokens * 48 * 4, stateCount * 4, tokens * 4,
        ]
        for (index, buffer) in inputs.enumerated() {
            let offset = index < 3 ? qkvOffsets[index] : (index == 5 ? initialOffset : 0)
            guard buffer.device.registryID == device.registryID, buffer.storageMode == .shared,
                offset <= buffer.length, sizes[index] <= buffer.length - offset
            else {
                throw EngineError("Invalid encoded GDN recurrence buffer")
            }
        }
        guard output.device.registryID == device.registryID, output.storageMode == .shared,
            output.length >= (tokens * 6144 + stateCount) * 4
        else { throw EngineError("Invalid encoded GDN output") }
        // The eight-row kernel loads q and k as float4: its bindings must be 16-byte aligned.
        let exact = gdnExactKernels && qkvOffsets.allSatisfy { $0 % 16 == 0 }
        if gdnExactKernels { if exact { gdnExactRecurrenceTiles += 1 } else { gdnExactRecurrenceFallbacks += 1 } }
        let pipeline = try projectionPipeline(named: exact ? GDNExactKernels.recurrenceName : "gdn_reference")
        guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 32,
            let encoder = command.makeComputeCommandEncoder()
        else { throw EngineError("GDN recurrence encoder allocation failed") }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in inputs.enumerated() {
            encoder.setBuffer(buffer, offset: index < 3 ? qkvOffsets[index] : (index == 5 ? initialOffset : 0), index: index)
        }
        let shape: [UInt32] = [UInt32(tokens), contractUpdates ? 1 : 0]
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 7) }
        encoder.setBuffer(output, offset: 0, index: 8)
        // One SIMD group per value row (reference) or per eight value rows of one head (exact kernel).
        encoder.dispatchThreads(
            MTLSize(width: exact ? stateCount / 32 : stateCount / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The same pinned RMS128 tree and Float storage as rmsnorm(), with no
    /// host materialization between recurrence and normalization.
    func encodeGDNRMS(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, tokens: Int
    ) throws {
        guard pinnedRMS, (1...256).contains(tokens) else { throw EngineError("GDN region requires pinned RMS128") }
        for (buffer, bytes) in [(input, tokens * 6144 * 4), (weights, 128 * 4), (output, tokens * 6144 * 4)] {
            guard buffer.device.registryID == device.registryID, buffer.storageMode == .shared, buffer.length >= bytes else {
                throw EngineError("Invalid encoded GDN RMS buffer")
            }
        }
        let pipeline = try projectionPipeline(named: "rmsnorm_pinned")
        guard pipeline.maxTotalThreadsPerThreadgroup >= 32, let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("GDN RMS encoder allocation failed")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        let shape: [UInt32] = [128, 128]
        var epsilon: Float = 1e-6
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        encoder.setBytes(&epsilon, length: 4, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens * 48, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    public func gdn(
        q: [Float], k: [Float], v: [Float], decay: [Float], beta: [Float],
        state: [Float], mask: [UInt32]? = nil, contractUpdates: Bool = true
    ) throws -> (output: [Float], state: [Float]) {
        let keyRow = 16 * 128
        let valueRow = 48 * 128
        let stateCount = 48 * 128 * 128
        guard !q.isEmpty, q.count % keyRow == 0, q.count / keyRow <= 4096 else {
            throw EngineError("Invalid fixed-shape GDN query geometry")
        }
        let tokens = q.count / keyRow
        guard k.count == q.count, v.count == tokens * valueRow, decay.count == tokens * 48,
            beta.count == decay.count, state.count == stateCount, mask == nil || mask!.count == tokens
        else {
            throw EngineError("Invalid fixed-shape GDN state geometry")
        }
        let output = try run(
            name: "gdn_reference",
            inputs: [
                buffer(q), buffer(k), buffer(v), buffer(decay),
                buffer(beta), buffer(state), buffer(mask ?? [UInt32](repeating: 1, count: tokens)),
                buffer([UInt32(tokens), contractUpdates ? 1 : 0]),
            ], count: v.count + stateCount, threads: stateCount / 4)
        let values = floats(output, count: v.count + stateCount)
        return (Array(values[..<v.count]), Array(values[v.count...]))
    }

    public func indexerScores(query: [Float], pooledKeys: [Float]) throws -> [Float] {
        guard query.count == 4 * 128, !pooledKeys.isEmpty, pooledKeys.count % 128 == 0,
            pooledKeys.count / 128 <= ModelContract.maximumContext / 4
        else {
            throw EngineError("Invalid fixed-shape indexer inputs")
        }
        let blocks = pooledKeys.count / 128
        let result = try run(name: "indexer_scores_reference", inputs: [buffer(query), buffer(pooledKeys)], count: blocks, threads: blocks)
        return floats(result, count: blocks)
    }

    /// Full-domain selection precedes the visibility filter, including exceptional-score behavior.
    public func selectedPositions(scores: [Float], position: Int, kvLength: Int) throws -> [UInt32] {
        guard position >= 0, position < kvLength, kvLength <= ModelContract.maximumContext,
            scores.count == kvLength / 4
        else { throw EngineError("Invalid causal indexer domain") }
        if kvLength <= 2048 { return (0...position).map(UInt32.init) }
        let masked = scores.enumerated().map { index, score in index * 4 + 3 <= position ? score : -Float.infinity }
        let top = try topIndices(scores: masked, count: min(512, scores.count))
        var result: [UInt32] = []
        for block in top where Int(block) * 4 + 3 <= position {
            result.append(contentsOf: (0..<4).map { block * 4 + UInt32($0) })
        }
        let tailStart = (position + 1) / 4 * 4
        if tailStart <= position { result.append(contentsOf: (tailStart...position).map(UInt32.init)) }
        return result.sorted()
    }

    /// One D256 query, GQA-12. Scores and BF16 probabilities are bounded to one query.
    /// This reproduces the unfused arithmetic boundaries, not an MLX reduction tree.
    public func attention(
        query: [Float], keys: [Float], values: [Float], kvLength: Int,
        position: Int, selectedPositions: [UInt32]? = nil
    ) throws -> [Float] {
        guard kvLength > 0, kvLength <= ModelContract.maximumContext, position >= 0, position < kvLength,
            query.count == 24 * 256, keys.count == 2 * kvLength * 256, values.count == keys.count
        else {
            throw EngineError("Invalid fixed-shape QSA attention inputs")
        }
        let positions = selectedPositions ?? (0...position).map(UInt32.init)
        guard positions.count <= 2051, Set(positions).count == positions.count,
            positions.allSatisfy({ Int($0) <= position })
        else {
            throw EngineError("Attention requires unique visible positions within the bounded selection budget")
        }
        // A fully masked row in the pinned fallback is uniform over the full stored key domain.
        // The empty-position flag implements that without allocating an unbounded score array.
        let ids = positions.isEmpty ? [UInt32(0)] : positions
        let output = try run(
            name: "attention_reference",
            inputs: [
                buffer(query), buffer(keys), buffer(values),
                buffer(ids), buffer([UInt32(kvLength), UInt32(positions.count)]),
            ], count: query.count, threads: 24)
        return floats(output, count: query.count)
    }

    private static let preamble = #"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        // No FMA contraction: every multiply and add rounds as written; fma() appears only where the baseline fuses.
        #pragma clang fp contract(off)
        // from_bf16 widens a BF16 bit pattern. round_bf16 rounds FP32 to the nearest BF16 (ties to even, NaN kept
        // quiet) and returns it widened: the explicit "cast to bfloat" boundary of the pinned schedules.
        inline float from_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
        inline float round_bf16(float x) {
            uint b = as_type<uint>(x);
            if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u)
                return as_type<float>((b & 0xffff0000u) | 0x00400000u);
            return as_type<float>((b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u);
        }
        """# + "\n"

    private static let unarySource = #"""
        // Primitive formulas and dtype boundaries adapted from pinned MLX
        // unary_ops.h, binary_ops.h, bf16_math.h and utils.h; MIT notice retained.
        inline bfloat native_bf_exp(bfloat x) { return bfloat(exp(float(x))); }
        inline bfloat native_bf_sigmoid(bfloat x) {
            auto y = 1 / (1 + native_bf_exp(bfloat(abs(float(x)))));
            return bfloat(x < 0 ? y : 1 - y);
        }
        inline float explicit_bf_sigmoid(float x) {
            float e = round_bf16(exp(abs(x)));
            float y = round_bf16(1.0f / round_bf16(1.0f + e));
            return x < 0 ? y : round_bf16(1.0f - y);
        }
        // bf16_unary_pinned: one element per thread. shape[0] selects sigmoid (0), silu (1), softplus (2) or the FP32
        // sigmoid of a widened BF16 input (3); shape[1] = 1 uses MSL bfloat arithmetic, 0 the explicit round_bf16
        // schedule, and the two must agree bit for bit. The reference for the pinned unary tables (LayerUnaryArithmetic).
        kernel void bf16_unary_pinned(device const ushort *input [[buffer(0)]],
            constant uint *shape [[buffer(1)]], device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            float x = from_bf16(input[i]);
            if (shape[0] == 3) {
                // The baseline explicitly widens the BF16 z projection before sigmoid.
                float y = 1.0f / (1.0f + exp(abs(x)));
                out[i] = x < 0 ? y : 1.0f - y;
            } else if (shape[0] == 2) {
                if (isnan(x) || x == INFINITY) { out[i] = round_bf16(x); return; }
                if (x == -INFINITY) { out[i] = 0; return; }
                if (shape[1]) {
                    bfloat mx = bfloat(max(x, 0.0f)), mn = bfloat(min(x, 0.0f));
                    bfloat e = native_bf_exp(mn - mx);
                    float xp1 = 1.0f + float(e);
                    bfloat correction = xp1 == 1.0f ? e : bfloat(float(e) * (log(xp1) / (xp1 - 1.0f)));
                    out[i] = float(bfloat(mx + correction));
                } else {
                    float mx = max(x, 0.0f), mn = min(x, 0.0f);
                    float e = round_bf16(exp(round_bf16(mn - mx)));
                    float xp1 = 1.0f + e;
                    float correction = xp1 == 1.0f ? e : round_bf16(e * (log(xp1) / (xp1 - 1.0f)));
                    out[i] = round_bf16(mx + correction);
                }
            } else if (shape[1]) {
                bfloat bx = as_type<bfloat>(input[i]);
                bfloat sigmoid = native_bf_sigmoid(bx);
                out[i] = float(shape[0] == 0 ? sigmoid : bfloat(bx * sigmoid));
            } else {
                float sigmoid = explicit_bf_sigmoid(x);
                out[i] = shape[0] == 0 ? sigmoid : round_bf16(x * sigmoid);
            }
        }

        """# + "\n"

    private static let source =
        preamble + unarySource + #"""
            // qmatvec_reference: one thread per output row; a plain FP32 K-order dot of x with the dequantized affine 4-bit
            // row (shape[2] = 1 rounds each weight to BF16 first), output rounded to BF16. Check-only; no pinned schedule.
            kernel void qmatvec_reference(device const float *x [[buffer(0)]],
                device const uint *words [[buffer(1)]], device const ushort *scales [[buffer(2)]],
                device const ushort *biases [[buffer(3)]], constant uint *shape [[buffer(4)]],
                device float *out [[buffer(5)]], uint row [[thread_position_in_grid]]) {
                uint width = shape[0], group = shape[1];
                float total = 0.0f;
                for (uint i = 0; i < width; ++i) {
                    uint index = row * (width / group) + i / group;
                    uint nibble = (words[row * (width / 8) + i / 8] >> ((i % 8) * 4)) & 15;
                    float w = float(nibble) * from_bf16(scales[index]) + from_bf16(biases[index]);
                    if (shape[2]) w = round_bf16(w); // Explicit QMM tile-rounding variant.
                    total = total + x[i] * w;
                }
                out[row] = round_bf16(total);
            }
            // qmm_pinned_reference: the pinned QMM split-K schedule (QuantizedSchedule), scalar. One SIMD group per
            // (row, output); x is BF16 [rows][width]. Lane j takes partitions j, j + lanes, ...: each partition is an FP32
            // K-order dot with BF16-rounded dequantized weights, rounded to BF16 and added to the lane total with BF16
            // rounding. Lanes combine in lane order with BF16 rounding below 32 splits, by simd_sum from 32.
            kernel void qmm_pinned_reference(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
                device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
                constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
                uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
                uint outputIndex = tid / 32, width = shape[0], outputs = shape[1], group = shape[3];
                uint token = outputIndex / outputs, row = outputIndex % outputs;
                uint splits = shape[4], partition = shape[5], lanes = shape[7];
                float laneTotal = 0.0f;
                if (lane < lanes) {
                    for (uint part = lane; part < splits; part += lanes) {
                        float total = 0.0f;
                        for (uint i = part * partition; i < (part + 1) * partition; ++i) {
                            uint index = row * (width / group) + i / group;
                            uint nibble = (words[row * (width / 8) + i / 8] >> ((i % 8) * 4)) & 15;
                            float w = round_bf16(QVAL(nibble) * QSCALE(scales, biases, index) + QBIAS(biases, index));
                            total = total + from_bf16(x[token*width+i]) * w;
                        }
                        laneTotal = round_bf16(laneTotal + round_bf16(total));
                    }
                }
                float sum = 0.0f;
                if (splits < 32) {
                    for (uint j = 0; j < lanes; ++j) sum = round_bf16(sum + simd_shuffle(laneTotal, j));
                } else {
                    sum = simd_sum(laneTotal);
                }
                if (lane == 0) out[outputIndex] = round_bf16(sum);
            }
            // Arithmetic fragments follow pinned MLX steel/gemm/mma.h (MIT). Each
            // SIMDgroup owns one 8x8 output fragment, with two elements per lane.
            // BF16 storage widens to FLOAT matrices, exactly as BlockMMA<...,AccumType=float>.
            // Output: BF16-rounded partials [splitK][rows][outputs]; qmm_pinned_split_reduce combines them when splitK > 1.
            kernel void qmm_pinned_simd_partitions(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
                device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
                constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
                uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
                uint width = shape[0], outputs = shape[1], rows = shape[2], group = shape[3], partition = shape[5];
                uint tilesN = (outputs + 7) / 8, tilesM = (rows + 7) / 8;
                uint fragmentIndex = tid / 32, part = fragmentIndex / (tilesN * tilesM);
                uint tile = fragmentIndex % (tilesN * tilesM), m = (tile / tilesN) * 8, n = (tile % tilesN) * 8;
                uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
                simdgroup_float8x8 a, b, accumulator;
                accumulator.thread_elements()[0] = 0.0f;
                accumulator.thread_elements()[1] = 0.0f;
                for (uint start = part * partition; start < (part + 1) * partition; start += 8) {
                    for (uint j = 0; j < 2; ++j) {
                        a.thread_elements()[j] = m + fm < rows ? from_bf16(x[(m + fm) * width + start + fn + j]) : 0.0f;
                        float weight = 0.0f;
                        if (n + fn + j < outputs) {
                            uint row = n + fn + j, column = start + fm;
                            uint metadata = row * (width / group) + column / group;
                            uint nibble = (words[row * (width / 8) + column / 8] >> ((column % 8) * 4)) & 15;
                            weight = round_bf16(QVAL(nibble) * QSCALE(scales, biases, metadata) + QBIAS(biases, metadata));
                        }
                        b.thread_elements()[j] = weight;
                    }
                    simdgroup_multiply_accumulate(accumulator, a, b, accumulator);
                }
                if (m + fm < rows) for (uint j = 0; j < 2; ++j) {
                    if (n + fn + j < outputs) {
                        out[(part * rows + m + fm) * outputs + n + fn + j] = round_bf16(accumulator.thread_elements()[j]);
                    }
                }
            }
            // qmm_pinned_split_reduce: one SIMD group per output; combines the [splitK][rows][outputs] partials with the
            // same lane assignment and combination order as qmm_pinned_reference (BF16-rounded adds; simd_sum from 32 splits).
            kernel void qmm_pinned_split_reduce(device const float *partials [[buffer(0)]],
                constant uint *shape [[buffer(1)]], device float *out [[buffer(2)]],
                uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
                uint index = tid / 32, count = shape[1] * shape[2], splits = shape[4], lanes = shape[7];
                float total = 0.0f;
                if (lane < lanes) for (uint part = lane; part < splits; part += lanes)
                    total = round_bf16(total + partials[part * count + index]);
                float sum = 0.0f;
                if (splits < 32) {
                    for (uint j = 0; j < lanes; ++j) sum = round_bf16(sum + simd_shuffle(total, j));
                } else { sum = simd_sum(total); }
                if (lane == 0) out[index] = round_bf16(sum);
            }
            // qmv_pinned_reference: the pinned MLX qmv lane schedule. One SIMD group per (row, output); each lane step takes
            // `perLane` consecutive K values inside one quantization group, sums code products in fours (FP32) and x in fours
            // with BF16 rounding (for the bias), then adds fma(scale, codeSum, xSum * bias); lanes combine by simd_sum and the
            // output is rounded to BF16. Also the one-row expert QMV (qmvPinnedBF16).
            kernel void qmv_pinned_reference(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
                device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
                constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
                uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
                uint outputIndex = tid / 32, width = shape[0], outputs = shape[1], group = shape[3];
                uint token = outputIndex / outputs, row = outputIndex % outputs, perLane = shape[6], lanes = shape[7];
                float result = 0.0f;
                if (lane < lanes) for (uint lo = lane*perLane; lo < width; lo += lanes*perLane) {
                    float quantSum = 0.0f, biasSum = 0.0f;
                    for (uint i = lo; i < min(lo+perLane, width); i += 4) {
                        float values[4], products[4];
                        for (uint j = 0; j < 4; ++j) {
                            values[j] = from_bf16(x[token*width+i+j]);
                            uint nibble = (words[row*(width/8)+(i+j)/8] >> (((i+j)%8)*4)) & 15;
                            products[j] = values[j] * QVAL(nibble);
                        }
                        biasSum = biasSum + round_bf16(round_bf16(round_bf16(values[0]+values[1])+values[2])+values[3]);
                        quantSum = quantSum + (((products[0]+products[1])+products[2])+products[3]);
                    }
                    uint index = row*(width/group)+lo/group;
                    float dot = fma(QSCALE(scales, biases, index), quantSum, biasSum*QBIAS(biases, index));
                    result = result + dot;
                }
                result = simd_sum(result);
                if (lane == 0) out[outputIndex] = round_bf16(result);
            }
            // Pinned rms_norm.metal (Apple MIT): N_READS=4 contiguous values per
            // lane, then SIMD reduction and a second SIMD reduction of group totals.
            // The looped variant uses the pipeline's maximum threadgroup width.
            // One threadgroup per group of `group` values (weights indexed modulo width); the output rounds to BF16 twice,
            // after normalizing and after the weight, as the pinned kernel does.
            kernel void rmsnorm_pinned(device const float *x [[buffer(0)]], device const float *weights [[buffer(1)]],
                constant uint *shape [[buffer(2)]], constant float *epsilon [[buffer(3)]], device float *out [[buffer(4)]],
                uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                uint groupSize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint simdGroup [[simdgroup_index_in_threadgroup]]) {
                threadgroup float localSums[32];
                threadgroup float inverse;
                uint width = shape[0], group = shape[1], start = row * group;
                float total = 0.0f;
                for (uint r = 0; r < group; r += groupSize * 4)
                    for (uint i = 0; i < 4; ++i) {
                        uint column = r + lid * 4 + i;
                        if (column < group) { float value = x[start + column]; total += value * value; }
                    }
                total = simd_sum(total);
                if (simdGroup == 0) localSums[lane] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) localSums[simdGroup] = total;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simdGroup == 0) {
                    total = simd_sum(localSums[lane]);
                    if (lane == 0) inverse = metal::precise::rsqrt(total / float(group) + epsilon[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint r = 0; r < group; r += groupSize * 4)
                    for (uint i = 0; i < 4; ++i) {
                        uint column = r + lid * 4 + i;
                        if (column < group) out[start + column] = round_bf16(round_bf16(x[start + column] * inverse) * weights[(start + column) % width]);
                    }
            }
            // rmsnorm_reference: one thread per group; FP32 sum of squares in index order, precise rsqrt, then the same two
            // BF16 roundings as rmsnorm_pinned. Used for grouped norms and when pinned RMS is off.
            kernel void rmsnorm_reference(device const float *x [[buffer(0)]], device const float *weights [[buffer(1)]],
                constant uint *shape [[buffer(2)]], constant float &epsilon [[buffer(3)]],
                device float *out [[buffer(4)]], uint g [[thread_position_in_grid]]) {
                uint width = shape[0], group = shape[1], start = g * group;
                float sum = 0.0f;
                for (uint i = 0; i < group; ++i) sum = sum + x[start+i] * x[start+i];
                float inverse = metal::precise::rsqrt(sum / float(group) + epsilon);
                for (uint i = 0; i < group; ++i) {
                    // Pinned rms_single_row and rms_looped both cast to T BEFORE
                    // multiplying by the learned weight, including ordinary RMSNorm.
                    float v = round_bf16(x[start+i] * inverse);
                    out[start+i] = round_bf16(v * weights[(start+i) % width]);
                }
            }
            // rope_reference: one thread per element; rotates the first shape[1] values of each shape[0]-wide row as split
            // halves (x[i], x[i + half]) with host-computed, BF16-rounded cos/sin, rounding each product and the sum to BF16.
            kernel void rope_reference(device const float *x [[buffer(0)]], device const float *c [[buffer(1)]],
                device const float *s [[buffer(2)]], constant uint *shape [[buffer(3)]],
                device float *out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
                uint d = i % shape[0], halfDim = shape[1] / 2;
                if (d >= shape[1]) { out[i] = x[i]; return; }
                uint freq = d % halfDim;
                float other = d < halfDim ? -x[i+halfDim] : x[i-halfDim];
                out[i] = round_bf16(round_bf16(x[i] * c[freq]) + round_bf16(other * s[freq]));
            }
            // rope_batched: rope_reference over tokens of shape[2] heads each; tables are [token][half]. Same arithmetic.
            kernel void rope_batched(device const float *x [[buffer(0)]], device const float *c [[buffer(1)]],
                device const float *s [[buffer(2)]], constant uint *shape [[buffer(3)]],
                device float *out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
                uint d = i % shape[0], halfDim = shape[1] / 2;
                if (d >= shape[1]) { out[i] = x[i]; return; }
                uint token = i / (shape[0] * shape[2]);
                uint freq = token * halfDim + d % halfDim;
                float other = d < halfDim ? -x[i+halfDim] : x[i-halfDim];
                out[i] = round_bf16(round_bf16(x[i] * c[freq]) + round_bf16(other * s[freq]));
            }
            // selection_key: an order-preserving uint key for a float score (larger score, larger key). NaN takes the lowest
            // key and signed zeros and subnormals compare equal, the pinned selection's tie semantics.
            inline uint selection_key(float value) {
                uint bits = as_type<uint>(value), mag = bits & 0x7fffffffu;
                if (mag > 0x7f800000u) return 0;
                if (mag < 0x00800000u) bits = 0; // signed zero/subnormal tie semantics
                return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
            }
            // top_indices_reference: one thread keeps a descending list of the k <= 512 largest keys by binary insertion; an
            // equal key goes after the earlier index. Router top-10 and indexer top-512 on the unbatched path; the batched
            // selector's reference.
            kernel void top_indices_reference(device const float *scores [[buffer(0)]], constant uint *shape [[buffer(1)]],
                device uint *out [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
                if (tid != 0) return;
                uint selected[512], keys[512], used = 0, k = shape[1];
                for (uint i = 0; i < shape[0]; ++i) {
                    uint key = selection_key(scores[i]);
                    if (used == k && key <= keys[k-1]) continue;
                    uint lo = 0, hi = used;
                    while (lo < hi) {
                        uint mid = (lo + hi) / 2;
                        if (keys[mid] >= key) lo = mid + 1; else hi = mid;
                    }
                    if (lo >= k) continue;
                    for (uint j = min(used, k-1); j > lo; --j) {
                        selected[j] = selected[j-1]; keys[j] = keys[j-1];
                    }
                    selected[lo] = i; keys[lo] = key; used = min(used+1, k);
                }
                for (uint rank = 0; rank < k; ++rank) out[rank] = selected[rank];
            }
            // silu_reference: x * sigmoid(x) in FP32 with the overflow-safe sigmoid form, rounded to BF16.
            kernel void silu_reference(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
                float e = exp(-abs(x[i]));
                float sigmoid = x[i] >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
                out[i] = round_bf16(x[i] * sigmoid);
            }

            // Adapted recurrence semantics from SlotStream Vendored/GatedDelta.swift (MIT;
            // Copyright 2026 Carlos Galarza). Preserve four contiguous Dk lanes per thread,
            // Kahan memory accumulation and SIMD-group reduction; FP32 state never narrows.
            // Layout: q, k [tokens][16 key heads][128]; v [tokens][48 value heads][128]; decay, beta [tokens][48]; state
            // [48][128 dv][128 dk]. One SIMD group per (value head, dv) row, four dk values per lane; value head h reads key
            // head h / 3. Output: values [tokens][48][128] rounded to BF16 (0 for masked tokens), then the final state.
            // shape[1] = 1 uses fma for the state and output updates, 0 separate multiply and add.
            kernel void gdn_reference(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                device const float *v [[buffer(2)]], device const float *decay [[buffer(3)]],
                device const float *beta [[buffer(4)]], device const float *initial [[buffer(5)]],
                device const uint *mask [[buffer(6)]], constant uint *shape [[buffer(7)]],
                device float *out [[buffer(8)]], uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
                uint row = tid / 32, hv = row / 128, dv = row % 128;
                uint stateBase = row * 128 + lane * 4, tokens = shape[0];
                float state[4];
                for (uint j = 0; j < 4; ++j) state[j] = initial[stateBase+j];
                for (uint t = 0; t < tokens; ++t) {
                    uint qBase = t * 16 * 128 + (hv / 3) * 128 + lane * 4;
                    uint valueIndex = t * 48 * 128 + hv * 128 + dv;
                    if (mask[t]) {
                        float memory = 0.0f;
                        {
                            #pragma clang fp reassociate(off)
                            #pragma clang fp contract(off)
                            float compensation = 0.0f;
                            for (uint j = 0; j < 4; ++j) {
                                state[j] = state[j] * decay[t*48+hv];
                                float product = state[j] * k[qBase+j];
                                float corrected = product - compensation;
                                float next = memory + corrected;
                                compensation = (next - memory) - corrected;
                                memory = next;
                            }
                        }
                        memory = simd_sum(memory);
                        float delta = (v[valueIndex] - memory) * beta[t*48+hv];
                        float value = 0.0f;
                        for (uint j = 0; j < 4; ++j) {
                            state[j] = shape[1] ? fma(k[qBase+j], delta, state[j]) : state[j] + k[qBase+j] * delta;
                            value = shape[1] ? fma(state[j], q[qBase+j], value) : value + state[j] * q[qBase+j];
                        }
                        value = simd_sum(value);
                        if (lane == 0) out[valueIndex] = round_bf16(value);
                    } else if (lane == 0) { out[valueIndex] = 0.0f; }
                }
                for (uint j = 0; j < 4; ++j) out[tokens*48*128+stateBase+j] = state[j];
            }
            // indexer_scores_reference: one thread per pooled key block; the sum over four query heads of relu(q . k) in
            // FP32 K order (a NaN dot passes through), divided by sqrt(128).
            kernel void indexer_scores_reference(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                device float *out [[buffer(2)]], uint block [[thread_position_in_grid]]) {
                float sum = 0.0f;
                for (uint h = 0; h < 4; ++h) {
                    float dot = 0.0f;
                    for (uint d = 0; d < 128; ++d) dot = dot + q[h*128+d] * k[block*128+d];
                    sum = sum + (isnan(dot) ? dot : max(dot, 0.0f));
                }
                out[block] = sum / sqrt(128.0f);
            }
            // attention_reference: one thread per query head (24 heads of 256 over 2 KV heads, GQA 12). Keys and values are
            // [kvHead][n][256]. Over the selected positions (at most 2051): dots of BF16-rounded q / 16 with k, rounded to
            // BF16; FP32 softmax with BF16-rounded probabilities; BF16-rounded weighted value sums. No selection means a
            // uniform average over all n stored keys, the pinned fully-masked fallback.
            kernel void attention_reference(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                device const float *v [[buffer(2)]], device const uint *ids [[buffer(3)]], constant uint *shape [[buffer(4)]],
                device float *out [[buffer(5)]], uint head [[thread_position_in_grid]]) {
                uint n = shape[0], selected = shape[1], kv = head / 12;
                float probabilities[2051];
                if (selected == 0) {
                    float p = round_bf16(1.0f / float(n));
                    for (uint d = 0; d < 256; ++d) {
                        float total = 0;
                        for (uint i = 0; i < n; ++i) total = total + p * v[(kv*n+i)*256+d];
                        out[head*256+d] = round_bf16(total);
                    }
                    return;
                }
                float maximum = -INFINITY;
                for (uint i = 0; i < selected; ++i) {
                    float dot = 0.0f;
                    for (uint d = 0; d < 256; ++d)
                        dot = dot + round_bf16(q[head*256+d] * 0.0625f) * k[(kv*n+ids[i])*256+d];
                    probabilities[i] = round_bf16(dot);
                    maximum = max(maximum, probabilities[i]);
                }
                float denominator = 0.0f;
                for (uint i = 0; i < selected; ++i) {
                    probabilities[i] = exp(probabilities[i] - maximum);
                    denominator = denominator + probabilities[i];
                }
                for (uint i = 0; i < selected; ++i) probabilities[i] = round_bf16(probabilities[i] / denominator);
                for (uint d = 0; d < 256; ++d) {
                    float total = 0.0f;
                    for (uint i = 0; i < selected; ++i) total = total + probabilities[i] * v[(kv*n+ids[i])*256+d];
                    out[head*256+d] = round_bf16(total);
                }
            }
            """#
}
