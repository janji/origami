import Darwin
import Foundation
import Metal
import ObjectiveC

private final class PinnedReductionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "pinned_reduction_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum PinnedReductionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Reduction schedules transcribed from MLX core 1f8e74e3f12f31365464a6867c6579f0e9b29d85: grouped RMS, row softmax,
/// the ten-expert weighted sum, GDN L2 norm and decay, the PLE dot and the HC fused entry. The host-joined layer path
/// uses them when `pinned_reductions` is set; the fused HC, PLE, GDN and decode paths always do. Separate from ordinary
/// fused RMSNorm. Each kernel reproduces MLX's reduction order and rounding boundaries bit for bit (BITWISE); safe
/// algebra, fast elementary functions except where MLX picks precise.
/// CPU inputs/results belong to the caller's admitted workspace. Each temporary
/// Metal buffer retains its own page-rounded reservation until actual retirement.
/// Calls join submitted GPU work before returning or throwing.
public final class PinnedReductions {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let groupNormPipeline, softmaxPipeline, expertReducePipeline, pleDotPipeline: MTLComputePipelineState
    private let l2Norm128Pipeline, gdnDecayPipeline: MTLComputePipelineState
    /// `hc_exact_fusion`: the fused HC tile entry, compiled on request in its own library (this file's
    /// source plus HCExactFusionKernels.entrySource, same options), so the default library is unchanged.
    private var hcFusedEntryPipeline: MTLComputePipelineState?
    private let ledger: MemoryLedger

    public init(ledger: MemoryLedger) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Pinned reductions Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw EngineError("Missing pinned reduction kernel: \(name)")
            }
            return try device.makeComputePipelineState(function: function)
        }
        groupNormPipeline = try pipeline("pinned_grouped_rms")
        softmaxPipeline = try pipeline("pinned_softmax_rows")
        expertReducePipeline = try pipeline("pinned_expert_reduce10")
        l2Norm128Pipeline = try pipeline("pinned_l2_norm128")
        gdnDecayPipeline = try pipeline("pinned_gdn_decay")
        pleDotPipeline = try pipeline("pinned_ple_dot")
        self.device = device
        self.queue = queue
        self.ledger = ledger
    }

    /// BF16 input/weight boundaries, FP32 square and general row sum, FP32
    /// multiply by Float(1 / Double(groupSize)), precise rsqrt, BF16 normalized
    /// value, then BF16 weighted output. Weights repeat for every model row.
    /// Widths below 65 use another core reduction specialization and are rejected.
    public func groupedRMS(
        x: [Float], weights: [Float], groupSize: Int,
        epsilon: Float = 1e-6
    ) throws -> [Float] {
        guard groupSize >= 65, groupSize <= 10_240, !weights.isEmpty,
            weights.count % groupSize == 0, !x.isEmpty,
            x.count % weights.count == 0, x.count <= Int(UInt32.max),
            epsilon.isFinite, epsilon > 0
        else {
            throw EngineError("Invalid pinned grouped RMS geometry or epsilon")
        }
        let threadCount: Int
        if groupSize <= 512 {
            threadCount = 32
        } else if groupSize <= 1024 {
            threadCount = 128
        } else {
            threadCount = min(1024, ((groupSize + 127) / 128) * 32)
        }
        guard threadCount <= groupNormPipeline.maxTotalThreadsPerThreadgroup else {
            throw EngineError("Pinned grouped RMS schedule exceeds this pipeline's thread limit")
        }
        return try autoreleasepool {
            let input = try copyBuffer(x.map(BF16.encode))
            let weight = try copyBuffer(weights.map(BF16.encode))
            let output = try allocate(bytes: x.count * MemoryLayout<Float>.stride)
            let shape = [UInt32(weights.count), UInt32(groupSize), UInt32(x.count / groupSize)]
            // number_of_elements(... inverted=true ...) computes a double
            // reciprocal before constructing the FP32 scalar, not GPU division.
            let constants: [Float] = [Float(1.0 / Double(groupSize)), epsilon]
            let command = try commandBuffer("pinned-grouped-rms")
            let encoder = try computeEncoder(command)
            encoder.setComputePipelineState(groupNormPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(weight, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
            constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
            encoder.dispatchThreadgroups(
                MTLSize(width: x.count / groupSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
            encoder.endEncoding()
            try complete(command)
            return read(output, count: x.count)
        }
    }

    /// Encode the existing grouped-RMS primitive into an owner's command.
    /// Inputs use BF16 storage; output preserves the original Float BF16 values.
    /// Caller retains every buffer/lease until the submitted command is joined.
    func encodeGroupedRMS(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, rows: Int, width: Int, groupSize: Int,
        epsilon: Float = 1e-6
    ) throws {
        guard (1...256).contains(rows), width == 10_240, groupSize == 2560,
            epsilon.isFinite, epsilon > 0, command.commandQueue.device.registryID == device.registryID,
            input.device.registryID == device.registryID, weights.device.registryID == device.registryID,
            output.device.registryID == device.registryID,
            input.length >= rows * width * 2, weights.length >= width * 2, output.length >= rows * width * 4
        else {
            throw EngineError("Invalid encoded grouped RMS fixed geometry or buffers")
        }
        let threads = min(1024, ((groupSize + 127) / 128) * 32)
        guard threads <= groupNormPipeline.maxTotalThreadsPerThreadgroup else {
            throw EngineError("Encoded grouped RMS exceeds threadgroup limit")
        }
        let shape = [UInt32(width), UInt32(groupSize), UInt32(rows * width / groupSize)]
        let constants: [Float] = [Float(1.0 / Double(groupSize)), epsilon]
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(groupNormPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.dispatchThreadgroups(
            MTLSize(width: rows * width / groupSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Compiles the `hc_exact_fusion` HC tile entry (idempotent). The kernel keeps 640 threads per
    /// 2560-element group (four elements per thread), exactly encodeGroupedRMS's geometry.
    func prepareHCFusedEntry() throws {
        guard hcFusedEntryPipeline == nil else { return }
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(source: Self.source + "\n" + HCExactFusionKernels.entrySource, options: options)
        guard let function = library.makeFunction(name: HCExactFusionKernels.entryName) else {
            throw EngineError("Missing HC fused entry kernel")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 640 else {
            throw EngineError("HC fused entry pipeline cannot hold 640 threads", code: "invalid_configuration")
        }
        hcFusedEntryPipeline = pipeline
    }
    var hcFusedEntryPrepared: Bool { hcFusedEntryPipeline != nil }

    /// `hc_exact_fusion` tile entry: from the FP32 residual rows, writes (1) the BF16 narrowing that
    /// hc_decode_narrow writes, (2) the grouped RMS output that encodeGroupedRMS writes from (1), and
    /// (3) the BF16 narrowing of (2), in one dispatch with encodeGroupedRMS's threadgroup geometry.
    func encodeHCFusedEntry(
        _ command: MTLCommandBuffer, residual: MTLBuffer, weights: MTLBuffer, residualBF16: MTLBuffer,
        output: MTLBuffer, outputBF16: MTLBuffer, rows: Int, epsilon: Float = 1e-6
    ) throws {
        let width = 10_240
        let groupSize = 2560
        let threads = 640
        guard let pipeline = hcFusedEntryPipeline else { throw EngineError("HC fused entry was not prepared") }
        let buffers = [residual, weights, residualBF16, output, outputBF16]
        guard (1...256).contains(rows), command.commandQueue.device.registryID == device.registryID,
            buffers.allSatisfy({ $0.device.registryID == device.registryID }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            residual.length >= rows * width * 4, weights.length >= width * 2, residualBF16.length >= rows * width * 2,
            output.length >= rows * width * 4, outputBF16.length >= rows * width * 2,
            threads <= pipeline.maxTotalThreadsPerThreadgroup, groupSize == 4 * threads
        else {
            throw EngineError("Invalid encoded HC fused entry geometry or buffers")
        }
        let shape = [UInt32(width), UInt32(groupSize), UInt32(rows * width / groupSize)]
        let constants: [Float] = [Float(1.0 / Double(groupSize)), epsilon]
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.setBuffer(residualBF16, offset: 0, index: 5)
        encoder.setBuffer(outputBF16, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows * width / groupSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// SOURCE PLE multiplies BF16 key/query, then performs a BF16 row sum.
    /// Each of 640 threads adds four contiguous products with BF16 rounding
    /// after each addition. Its 20 native BF16 SIMD sums narrow before the
    /// final native BF16 SIMD sum. Row-simple (>=32 rows) and row-looped use
    /// this same per-row arithmetic. Up to 256 tokens, four branches each.
    public func pleDot(key: [Float], query: [Float]) throws -> [Float] {
        guard !key.isEmpty, key.count == query.count, key.count % 2560 == 0,
            key.count <= 256 * 4 * 2560
        else {
            throw EngineError("Invalid pinned PLE dot geometry")
        }
        guard pleDotPipeline.maxTotalThreadsPerThreadgroup >= 640 else {
            throw EngineError("Pinned PLE dot exceeds this pipeline's thread limit")
        }
        return try autoreleasepool {
            let left = try copyBuffer(key.map(BF16.encode))
            let right = try copyBuffer(query.map(BF16.encode))
            let count = key.count / 2560
            let output = try allocate(bytes: count * MemoryLayout<Float>.stride)
            let command = try commandBuffer("pinned-ple-dot")
            let encoder = try computeEncoder(command)
            encoder.setComputePipelineState(pleDotPipeline)
            encoder.setBuffer(left, offset: 0, index: 0)
            encoder.setBuffer(right, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var rowCount = UInt32(count)
            encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 640, height: 1, depth: 1))
            encoder.endEncoding()
            try complete(command)
            return read(output, count: count)
        }
    }

    /// The pleDot dispatch encoded into an owner's command: identical pipeline, 640-thread
    /// rows and BF16 inputs. `rows` counts token-branch rows of 2560 values.
    func encodePLEDot(_ command: MTLCommandBuffer, key: MTLBuffer, query: MTLBuffer, output: MTLBuffer, rows: Int) throws {
        guard (1...(256 * 4)).contains(rows), pleDotPipeline.maxTotalThreadsPerThreadgroup >= 640,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            [key, query, output].allSatisfy({ $0.device.registryID == device.registryID }),
            key.length >= rows * 2560 * 2, query.length >= rows * 2560 * 2, output.length >= rows * 4
        else {
            throw EngineError("Invalid encoded PLE dot geometry or buffers")
        }
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(pleDotPipeline)
        encoder.setBuffer(key, offset: 0, index: 0)
        encoder.setBuffer(query, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var rowCount = UInt32(rows)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 640, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Pinned softmax_single_row with FP32 accumulation, explicit fast::exp,
    /// reads4 SIMD reduction and reciprocal multiplication. BF16 output mode
    /// first narrows the supplied scores to BF16, as unfused BF16 SDPA does.
    /// Long-axis looped softmax (>4096) is a distinct schedule, not implemented.
    public func softmax(rows: [Float], width: Int, outputBF16: Bool = false) throws -> [Float] {
        guard width > 0, width <= 4096, !rows.isEmpty, rows.count % width == 0,
            rows.count <= Int(UInt32.max)
        else {
            throw EngineError("Invalid pinned row softmax geometry")
        }
        let threadCount = ((width + 127) / 128) * 32
        guard threadCount <= softmaxPipeline.maxTotalThreadsPerThreadgroup else {
            throw EngineError("Pinned softmax schedule exceeds this pipeline's thread limit")
        }
        return try autoreleasepool {
            let input = try copyBuffer(rows)
            let output = try allocate(bytes: rows.count * MemoryLayout<Float>.stride)
            let shape = [UInt32(width), UInt32(outputBF16 ? 1 : 0)]
            let command = try commandBuffer("pinned-softmax")
            let encoder = try computeEncoder(command)
            encoder.setComputePipelineState(softmaxPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            encoder.dispatchThreadgroups(
                MTLSize(width: rows.count / width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
            encoder.endEncoding()
            try complete(command)
            return read(output, count: rows.count)
        }
    }

    /// Identical width10 row-one softmax dispatch for a caller-owned decode
    /// region. No input conversion or altered reduction/unary policy occurs.
    /// The caller joins and validates finite nonnegative weights before use.
    package func encodeRouterSoftmax10(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer) throws {
        guard input.device.registryID == device.registryID, output.device.registryID == device.registryID,
            input.length >= 40, output.length >= 40, input !== output,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            softmaxPipeline.maxTotalThreadsPerThreadgroup >= 32
        else {
            throw EngineError("Invalid encoded router softmax geometry")
        }
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(softmaxPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let shape: [UInt32] = [10, 0]
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Same per-row width10 reduction as softmax(rows:width:), with the
    /// original32-thread group and no host activation boundary.
    package func encodeRouterRows(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, rows: Int) throws {
        guard (1...256).contains(rows), input.device.registryID == device.registryID, output.device.registryID == device.registryID,
            input.length >= rows * 40, output.length >= rows * 40, input !== output,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            softmaxPipeline.maxTotalThreadsPerThreadgroup >= 32
        else { throw EngineError("Invalid encoded router tile geometry") }
        let e = try computeEncoder(command)
        e.setComputePipelineState(softmaxPipeline)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(output, offset: 0, index: 1)
        let shape: [UInt32] = [10, 0]
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        e.endEncoding()
    }

    /// Experts are canonical [tokens,10,hidden], weights are FP32 [tokens,10].
    /// SOURCE cached/sweep/workspace paths all materialize that FP32 product and
    /// reduce axis -2. col_reduce_small uses 8 lanes: p0+p8, p1+p9, p2...p7,
    /// then a sequential reduction of those 8 partials. Final routed output is BF16.
    public func weightedExpertReduce10(experts: [Float], weights: [Float], hiddenSize: Int = 2560) throws -> [Float] {
        guard hiddenSize > 0, hiddenSize <= 10_240, !experts.isEmpty,
            experts.count % (10 * hiddenSize) == 0, experts.count <= Int(UInt32.max),
            weights.count == experts.count / hiddenSize
        else {
            throw EngineError("Invalid pinned ten-expert reduction geometry")
        }
        return try autoreleasepool {
            let input = try copyBuffer(experts)
            return try weightedExpertReduce10(
                buffer: input, tokens: experts.count / (10 * hiddenSize),
                weights: weights, hiddenSize: hiddenSize)
        }
    }

    /// Direct buffer input preserves the exact existing canonical reduction
    /// pipeline, tree, BF16 boundary and joined output read. The caller retains
    /// the contribution owner until this command has joined.
    public func weightedExpertReduce10(buffer input: MTLBuffer, tokens: Int, weights: [Float], hiddenSize: Int = 2560) throws -> [Float] {
        guard (1...10240).contains(hiddenSize), tokens > 0, tokens <= Int(UInt32.max) / (10 * hiddenSize),
            input.device.registryID == device.registryID, input.length >= tokens * 10 * hiddenSize * 4,
            weights.count == tokens * 10
        else { throw EngineError("Invalid direct ten-expert reduction geometry") }
        let count = tokens * hiddenSize
        return try withExtendedLifetime(input) {
            try autoreleasepool {
                let weight = try copyBuffer(weights)
                let output = try allocate(bytes: count * MemoryLayout<Float>.stride)
                let command = try commandBuffer("pinned-expert-reduce10")
                try encodeWeightedExpertReduce10(
                    command, input: input, weights: weight, output: output, tokens: tokens, hiddenSize: hiddenSize)
                try complete(command)
                return read(output, count: count)
            }
        }
    }

    /// Composes the unchanged independently compiled reduction into the caller's
    /// command. GPU readers and all buffers must be joined before scope exit.
    package func encodeWeightedExpertReduce10(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, tokens: Int, hiddenSize: Int = 2560
    ) throws {
        guard tokens > 0, hiddenSize > 0, hiddenSize <= 10240,
            tokens <= Int(UInt32.max) / (10 * hiddenSize), input.length >= tokens * 10 * hiddenSize * 4,
            weights.length >= tokens * 10 * 4, output.length >= tokens * hiddenSize * 4
        else {
            throw EngineError("Invalid composed expert reduction geometry")
        }
        let count = tokens * hiddenSize
        let shape = [UInt32(hiddenSize), UInt32(tokens * hiddenSize)]
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(expertReducePipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, expertReducePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// General FP32 square/sum128 using four contiguous values per SIMD lane,
    /// explicit precise rsqrt(sum + epsilon), and BF16 normalization. The optional
    /// query scale is first narrowed to BF16, matching MLXArray * Float semantics.
    public func l2Norm128(x: [Float], scale: Float? = nil) throws -> [Float] {
        guard !x.isEmpty, x.count % 128 == 0, x.count <= 256 * 2048,
            scale == nil || scale!.isFinite
        else {
            throw EngineError("Invalid pinned GDN L2 geometry or scale")
        }
        return try autoreleasepool {
            let input = try copyBuffer(x.map(BF16.encode))
            let output = try allocate(bytes: x.count * MemoryLayout<Float>.stride)
            let command = try commandBuffer("pinned-gdn-l2norm128")
            try encodeL2Norm128(command, input: input, output: output, count: x.count, scale: scale)
            try complete(command)
            return read(output, count: x.count)
        }
    }

    /// The baseline first widens BF16 A_log, applies FP32 precise exp and negate,
    /// multiplies the supplied BF16 softplus, then applies FP32 precise exp.
    public func gdnDecay(aLog: [Float], softplus: [Float]) throws -> [Float] {
        guard aLog.count == 48, !softplus.isEmpty, softplus.count % 48 == 0,
            softplus.count <= 256 * 48
        else {
            throw EngineError("Invalid pinned GDN decay geometry")
        }
        return try autoreleasepool {
            let logarithms = try copyBuffer(aLog.map(BF16.encode))
            let factors = try copyBuffer(softplus.map(BF16.encode))
            let output = try allocate(bytes: softplus.count * MemoryLayout<Float>.stride)
            let command = try commandBuffer("pinned-gdn-decay")
            try encodeGDNDecay(
                command, aLog: logarithms, softplus: factors, output: output,
                count: softplus.count)
            try complete(command)
            return read(output, count: softplus.count)
        }
    }

    /// One joined command buffer prepares the two 16-head norms and 48-head
    /// decay for 1...256 chronological GDN rows. All arithmetic boundaries match
    /// the separate graph operations; the submitted GPU work has no CPU gaps.
    public func prepareGDN(query: [Float], keys: [Float], aLog: [Float], softplus: [Float]) throws
        -> (query: [Float], keys: [Float], decay: [Float])
    {
        guard !query.isEmpty, query.count % 2048 == 0, query.count <= 256 * 2048,
            keys.count == query.count, aLog.count == 48,
            softplus.count == query.count / 2048 * 48
        else {
            throw EngineError("Invalid pinned GDN preparation geometry")
        }
        return try autoreleasepool {
            let queryInput = try copyBuffer(query.map(BF16.encode))
            let keyInput = try copyBuffer(keys.map(BF16.encode))
            let logarithms = try copyBuffer(aLog.map(BF16.encode))
            let factors = try copyBuffer(softplus.map(BF16.encode))
            let queryOutput = try allocate(bytes: query.count * MemoryLayout<Float>.stride)
            let keyOutput = try allocate(bytes: keys.count * MemoryLayout<Float>.stride)
            let decayOutput = try allocate(bytes: softplus.count * MemoryLayout<Float>.stride)
            let command = try commandBuffer("pinned-gdn-preparation")
            // This is the exact SOURCE host scalar expression, before BF16 typing.
            let queryScale = Float(pow(Double(128), -0.5))
            try encodeL2Norm128(command, input: queryInput, output: queryOutput, count: query.count, scale: queryScale)
            try encodeL2Norm128(command, input: keyInput, output: keyOutput, count: keys.count, scale: nil)
            try encodeGDNDecay(
                command, aLog: logarithms, softplus: factors, output: decayOutput,
                count: softplus.count)
            try complete(command)
            return (
                read(queryOutput, count: query.count), read(keyOutput, count: keys.count),
                read(decayOutput, count: softplus.count)
            )
        }
    }

    private func encodeL2Norm128(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        count: Int, scale: Float?
    ) throws {
        guard l2Norm128Pipeline.threadExecutionWidth == 32,
            l2Norm128Pipeline.maxTotalThreadsPerThreadgroup >= 32
        else {
            throw EngineError("Pinned GDN normalization requires a 32-lane SIMD group")
        }
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(l2Norm128Pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let constants: [Float] = [1e-6, BF16.decode(BF16.encode(scale ?? 1))]
        var applyScale = UInt32(scale == nil ? 0 : 1)
        constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        encoder.setBytes(&applyScale, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: count / 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeGDNDecay(
        _ command: MTLCommandBuffer, aLog: MTLBuffer, softplus: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        let encoder = try computeEncoder(command)
        encoder.setComputePipelineState(gdnDecayPipeline)
        encoder.setBuffer(aLog, offset: 0, index: 0)
        encoder.setBuffer(softplus, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var shape = UInt32(count)
        encoder.setBytes(&shape, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, gdnDecayPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func allocate(bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength else {
            throw EngineError("Pinned reduction buffer exceeds Metal allocation bound")
        }
        let page = UInt64(getpagesize())
        let charged = (UInt64(bytes) + page - 1) / page * page
        let owner = try PinnedReductionAllocation(ledger: ledger, bytes: charged)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Pinned reduction allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &PinnedReductionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard UInt64(buffer.allocatedSize) <= charged else {
            throw EngineError("Pinned reduction allocation exceeded its admitted page charge")
        }
        return buffer
    }
    private func copyBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        let buffer = try allocate(bytes: values.count * MemoryLayout<T>.stride)
        values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return buffer
    }
    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
    private func commandBuffer(_ label: String) throws -> MTLCommandBuffer {
        guard let command = queue.makeCommandBuffer() else {
            throw EngineError("Pinned reduction command allocation failed", code: "insufficient_memory")
        }
        command.label = label
        return command
    }
    private func computeEncoder(_ command: MTLCommandBuffer) throws -> MTLComputeCommandEncoder {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Pinned reduction encoder allocation failed", code: "insufficient_memory")
        }
        return encoder
    }
    private func complete(_ command: MTLCommandBuffer) throws {
        command.commit()
        try command.waitBounded()
        guard command.status == .completed else {
            throw EngineError.gpu("Pinned reduction dispatch failed", command.error)
        }
    }

    // Adapted from MLX core reduce_row.h, softmax.h and col_reduce_small dispatch
    // schedules (Copyright 2023-2024 Apple Inc., MIT; THIRD_PARTY_NOTICES.md).
    // Safe algebra preserves distinct graph operation rounding. Elementary
    // functions retain the source's Fast floating-point function policy.
    private static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        inline float from_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
        inline float round_bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
            return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
        }

        // Layers.swift l2normQK explicitly widens input, squares and general-sums.
        // Width128 is one SIMD group; the source simple/looped reducers both add
        // four contiguous elements per lane before their SIMD sum. Rsqrt's unary
        // functor explicitly uses precise::rsqrt regardless of library defaults.
        kernel void pinned_l2_norm128(device const ushort *input [[buffer(0)]],
            device float *output [[buffer(1)]], constant float *constants [[buffer(2)]],
            constant uint &applyScale [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]]) {
            uint base = row * 128 + lane * 4;
            float values[4];
            float total = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                values[i] = from_bf16(input[base + i]);
                float squared = values[i] * values[i];
                total = squared + total;
            }
            total = simd_sum(total);
            float inverse = metal::precise::rsqrt(total + constants[0]);
            for (uint i = 0; i < 4; ++i) {
                float normalized = round_bf16(values[i] * inverse);
                output[base + i] = applyScale ? round_bf16(normalized * constants[1]) : normalized;
            }
        }

        // unary_ops.h Exp explicitly selects precise::exp (both exponentials).
        kernel void pinned_gdn_decay(device const ushort *aLog [[buffer(0)]],
            device const ushort *softplus [[buffer(1)]], device float *output [[buffer(2)]],
            constant uint *shape [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index >= shape[0]) return;
            float logarithm = from_bf16(aLog[index % 48]);
            float factor = from_bf16(softplus[index]);
            float inner = metal::precise::exp(logarithm);
            float product = -inner * factor;
            output[index] = metal::precise::exp(product);
        }

        // Grouped RMS: one threadgroup per group of shape[1] BF16 values; FP32 squares in reads4 lane order, SIMD then
        // threadgroup sum, times the host's double-computed reciprocal, precise rsqrt, BF16 normalized value times the BF16
        // weight (weights repeat every shape[0] values), BF16-rounded output in FP32.
        kernel void pinned_grouped_rms(device const ushort *x [[buffer(0)]], device const ushort *weights [[buffer(1)]],
            device float *output [[buffer(2)]], constant uint *shape [[buffer(3)]], constant float *constants [[buffer(4)]],
            uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
            uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simdGroup [[simdgroup_index_in_threadgroup]]) {
            uint width = shape[0], group = shape[1], start = row * group;
            threadgroup float partials[32];
            threadgroup float inverse;
            if (simdGroup == 0) partials[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0.0f;
            // Contiguous reads4 then SIMD sum, as row_reduce_simple / looped.
            for (uint base = tid * 4; base < group; base += threads * 4) {
                for (uint i = 0; i < 4; ++i) {
                    if (base + i < group) {
                        float value = from_bf16(x[start + base + i]);
                        float squared = value * value;
                        total = squared + total;
                    }
                }
            }
            total = simd_sum(total);
            if (lane == 0) partials[simdGroup] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                // Source skips the second reduction when one SIMD group suffices.
                float sum = threads == 32 ? total : simd_sum(partials[lane]);
                if (lane == 0) {
                    float mean = sum * constants[0];
                    // unary_ops.h Rsqrt explicitly selects precise, overriding
                    // the library's default Fast elementary-function policy.
                    inverse = metal::precise::rsqrt(mean + constants[1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint column = tid; column < group; column += threads) {
                float normalized = round_bf16(from_bf16(x[start + column]) * inverse);
                output[start + column] = round_bf16(normalized * from_bf16(weights[(start + column) % width]));
            }
        }

        // reduction/ops.h Sum<bfloat>, reduce_row.h per_thread_row_reduce and
        // threadgroup_reduce, reduce.cpp threadgroup_size_from_row_size(2560).
        // Use the native bfloat SIMD intrinsic, preserving its internal widening
        // and BF16 return exactly as the pinned source does.
        kernel void pinned_ple_dot(device const ushort *key [[buffer(0)]],
            device const ushort *query [[buffer(1)]], device float *output [[buffer(2)]],
            constant uint &rowCount [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint simdGroup [[simdgroup_index_in_threadgroup]]) {
            threadgroup bfloat partials[32];
            if (simdGroup == 0) partials[lane] = bfloat(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint start = row * 2560 + tid * 4;
            bfloat total = bfloat(0);
            for (uint i = 0; i < 4; ++i) {
                bfloat product = bfloat(as_type<bfloat>(key[start + i]) * as_type<bfloat>(query[start + i]));
                total = bfloat(product + total);
            }
            // row_reduce_looped combines its one row subtotal into an initial
            // zero; preserve that explicit operand order for the short-row path.
            if (rowCount < 32) total = bfloat(bfloat(0) + total);
            // MLX bf16_math.h widens its bfloat SIMD reduction to float and
            // narrows the result. Metal's standard library has no bfloat overload.
            total = bfloat(simd_sum(float(total)));
            if (lane == 0) partials[simdGroup] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                total = bfloat(simd_sum(float(partials[lane])));
                if (lane == 0) output[row] = float(total);
            }
        }

        // Row softmax as MLX softmax_single_row (width <= 4096): one threadgroup per row, four values per thread, padding
        // -inf; max, fast::exp, SIMD then threadgroup sums, reciprocal multiply. shape[1] narrows input and output to BF16.
        kernel void pinned_softmax_rows(device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
            constant uint *shape [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simdGroup [[simdgroup_index_in_threadgroup]]) {
            uint width = shape[0], narrow = shape[1], start = row * width + tid * 4;
            threadgroup float maxima[32], denominators[32];
            float values[4];
            for (uint i = 0; i < 4; ++i) {
                float value = tid * 4 + i < width ? input[start + i] : -INFINITY;
                values[i] = narrow ? round_bf16(value) : value;
            }
            if (simdGroup == 0) { maxima[lane] = -INFINITY; denominators[lane] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maximum = -FLT_MAX;
            for (uint i = 0; i < 4; ++i) maximum = maximum < values[i] ? values[i] : maximum;
            maximum = simd_max(maximum);
            if (lane == 0) maxima[simdGroup] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                maximum = simd_max(maxima[lane]);
                if (lane == 0) maxima[0] = maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum = maxima[0];
            float denominator = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                values[i] = fast::exp(values[i] - maximum);
                denominator += values[i];
            }
            denominator = simd_sum(denominator);
            if (lane == 0) denominators[simdGroup] = denominator;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                denominator = simd_sum(denominators[lane]);
                if (lane == 0) denominators[0] = denominator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float reciprocal = 1.0f / denominators[0];
            for (uint i = 0; i < 4; ++i) if (tid * 4 + i < width) {
                float value = values[i] * reciprocal;
                output[start + i] = narrow ? round_bf16(value) : value;
            }
        }

        // Ten-expert weighted sum, one thread per output: FP32 products, MLX col_reduce_small order (p0+p8, p1+p9, p2..p7,
        // then a sequential sum of the eight partials), BF16-rounded output. Input [tokens][10][hidden], weights [tokens][10].
        kernel void pinned_expert_reduce10(device const float *experts [[buffer(0)]], device const float *weights [[buffer(1)]],
            device float *output [[buffer(2)]], constant uint *shape [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index >= shape[1]) return;
            uint hidden = shape[0], row = index / hidden, column = index % hidden;
            uint start = row * 10 * hidden + column;
            float parts[8];
            for (uint rank = 0; rank < 8; ++rank) {
                float product = experts[start + rank * hidden] * weights[row * 10 + rank];
                parts[rank] = product + 0.0f;
            }
            float p8 = experts[start + 8 * hidden] * weights[row * 10 + 8];
            float p9 = experts[start + 9 * hidden] * weights[row * 10 + 9];
            parts[0] = p8 + parts[0];
            parts[1] = p9 + parts[1];
            float total = parts[0];
            for (uint rank = 1; rank < 8; ++rank) total = parts[rank] + total;
            output[index] = round_bf16(total);
        }
        """#
}
