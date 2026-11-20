import Darwin
import Foundation
import Metal
import ObjectiveC

private final class GDNPreparationAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    let charged: UInt64
    init(ledger: MemoryLedger, bytes: Int) throws {
        self.ledger = ledger
        let page = UInt64(getpagesize())
        charged = (UInt64(bytes) + page - 1) / page * page
        reservation = try ledger.reserve(bytes: charged, category: "fused_gdn_preparation")
    }
    deinit { try? ledger.release(reservation) }
}
private enum GDNPreparationAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// GDN preparation on the GPU, BITWISE against the host path (LayerMath / `hostPreparation`): the
/// causal depthwise convolution over [3 tail rows | new rows] with four ascending taps, SiLU (table),
/// the L2 norms of q (times BF16(128^-0.5)) and k per 128-wide head, beta = sigmoid(b) and
/// decay = exp(-exp(BF16(A_log)) * softplus(BF16(a + dt_bias))) in FP32 with precise exp; also the
/// output gate round(normed * sigmoidFP32(z)).
/// Output layout `qkv`: planar [rows x 2048 q | rows x 2048 k | rows x 6144 v]. Any operand outside
/// conservative finite bounds sets an exception word and the host arithmetic recomputes the whole
/// tile. `prepare`/`gate` are synchronous (per-layer path, rows >= `minimumRows`); `encodePreparation`
/// and `encodeGate` add the same dispatches to a caller's command (GDNDeviceBlock, GDNBufferRegion).
/// Recurrence, output RMS, projections and sequence-state publication remain with the callers.
public final class FusedGDNPreparation {
    public let minimumRows: Int
    /// `gdn_exact_kernels`: the convolution runs the vector-load/store kernel `gdn_conv_exact_vec4`
    /// (same grid, same arithmetic and checks); every dispatch of it is counted.
    public let exactKernels: Bool
    public private(set) var exactConvTiles: UInt64 = 0
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let preparation, factors, gating: MTLComputePipelineState
    private let siluTable, sigmoidTable, softplusTable, wideSigmoidTable: MTLBuffer
    private let ledger: MemoryLedger
    private let arithmetic: LayerUnaryArithmetic
    private let reductions: PinnedReductions
    public private(set) var submittedCommands: UInt64 = 0
    public private(set) var completedCommands: UInt64 = 0
    public private(set) var preparationFallbacks: UInt64 = 0
    public private(set) var gatingFallbacks: UInt64 = 0
    public private(set) var embeddedGateEncoders: UInt64 = 0
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, preparationFallbacks, gatingFallbacks: UInt64
        /// Embedded in GDNBufferRegion's command; not an additional submission.
        public let embeddedGateEncoders: UInt64
    }
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submittedCommands, completedCommands: completedCommands,
            preparationFallbacks: preparationFallbacks, gatingFallbacks: gatingFallbacks, embeddedGateEncoders: embeddedGateEncoders)
    }

    public init(
        arithmetic: LayerUnaryArithmetic, reductions: PinnedReductions, ledger: MemoryLedger, minimumRows: Int = 8,
        exactKernels: Bool = false
    ) throws {
        guard (1...256).contains(minimumRows) else {
            throw EngineError("fused_gdn_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Fused GDN Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(
            source: Self.source + (exactKernels ? GDNExactKernels.convolutionSource : ""), options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw EngineError("Missing fused GDN kernel") }
            return try device.makeComputePipelineState(function: function)
        }
        preparation = try pipeline(exactKernels ? GDNExactKernels.convolutionName : "gdn_conv_normalize")
        factors = try pipeline("gdn_factors")
        gating = try pipeline("gdn_gate")
        guard preparation.threadExecutionWidth == 32, preparation.maxTotalThreadsPerThreadgroup >= 32 else {
            throw EngineError("Fused GDN preparation requires a 32-lane SIMD group")
        }
        self.exactKernels = exactKernels
        self.device = device
        self.queue = queue
        self.ledger = ledger
        self.arithmetic = arithmetic
        self.reductions = reductions
        self.minimumRows = minimumRows
        siluTable = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 2)
        sigmoidTable = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 2)
        softplusTable = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 2)
        wideSigmoidTable = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 4)
        let silu = siluTable.contents().assumingMemoryBound(to: UInt16.self)
        let sigmoid = sigmoidTable.contents().assumingMemoryBound(to: UInt16.self)
        let softplus = softplusTable.contents().assumingMemoryBound(to: UInt16.self)
        let wide = wideSigmoidTable.contents().assumingMemoryBound(to: Float.self)
        for bits in 0..<65_536 {
            let value = BF16.decode(UInt16(bits))
            silu[bits] = BF16.encode(arithmetic.silu(value))
            sigmoid[bits] = BF16.encode(arithmetic.sigmoid(value))
            softplus[bits] = BF16.encode(arithmetic.softplus(value))
            wide[bits] = arithmetic.sigmoidFP32(value)
        }
    }

    public func usesGPU(rows: Int) -> Bool { (minimumRows...256).contains(rows) }
    public struct Prepared {
        public let query, keys, values, beta, decay: [Float]
        public let convolved: [Float]?
        public let nextTail: [UInt16]
    }

    public func prepare(
        mixed: [Float], tail: [UInt16], convolution: [Float],
        a: [Float], b: [Float], dtBias: [Float], aLog: [Float],
        captureConvolved: Bool = false, shouldContinue: () -> Bool = { true }
    ) throws -> Prepared {
        guard !mixed.isEmpty, mixed.count % 10_240 == 0, (1...256).contains(mixed.count / 10_240),
            tail.count == 3 * 10_240, convolution.count == 4 * 10_240,
            a.count == mixed.count / 10_240 * 48, b.count == a.count,
            dtBias.count == 48, aLog.count == 48
        else { throw EngineError("Invalid fused GDN preparation geometry") }
        guard shouldContinue() else { throw IOOperationCancelled() }
        let rows = mixed.count / 10_240
        return try autoreleasepool {
            let input = try copy(mixed)
            let history = try copy(tail)
            let weights = try copy(convolution)
            let aInput = try copy(a)
            let bInput = try copy(b)
            let bias = try copy(dtBias)
            let logs = try copy(aLog)
            let qkv = try allocate(mixed.count * 4)
            let beta = try allocate(a.count * 4)
            let decay = try allocate(a.count * 4)
            let convolved = captureConvolved ? try allocate(mixed.count * 4) : nil
            let exceptions = try copy([UInt32(0)])
            let command = try commandBuffer("fused-gdn-convolution-normalization-factors")
            let convEncoder = try encoder(command)
            convEncoder.setComputePipelineState(preparation)
            for (index, buffer) in [input, history, weights, siluTable, qkv, convolved ?? qkv].enumerated() {
                convEncoder.setBuffer(buffer, offset: 0, index: index)
            }
            let shape = [UInt32(rows), captureConvolved ? 1 : 0]
            // Exact SOURCE host scale, narrowed to the BF16 array dtype.
            let constants: [Float] = [1e-6, BF16.decode(BF16.encode(Float(pow(Double(128), -0.5))))]
            shape.withUnsafeBytes { convEncoder.setBytes($0.baseAddress!, length: $0.count, index: 6) }
            constants.withUnsafeBytes { convEncoder.setBytes($0.baseAddress!, length: $0.count, index: 7) }
            convEncoder.setBuffer(exceptions, offset: 0, index: 8)
            convEncoder.dispatchThreadgroups(
                MTLSize(width: rows * 80, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            convEncoder.endEncoding()
            if exactKernels { exactConvTiles += 1 }
            let factorEncoder = try encoder(command)
            factorEncoder.setComputePipelineState(factors)
            for (index, buffer) in [aInput, bInput, bias, logs, softplusTable, sigmoidTable, beta, decay].enumerated() {
                factorEncoder.setBuffer(buffer, offset: 0, index: index)
            }
            var factorCount = UInt32(a.count)
            factorEncoder.setBytes(&factorCount, length: MemoryLayout<UInt32>.size, index: 8)
            factorEncoder.setBuffer(exceptions, offset: 0, index: 9)
            factorEncoder.dispatchThreads(
                MTLSize(width: a.count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, factors.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            factorEncoder.endEncoding()
            try complete(command, shouldContinue: shouldContinue)
            if exceptions.contents().load(as: UInt32.self) != 0 {
                // Keep the existing CPU exception behavior and existing pinned
                // GPU norm/decay graph. No payload or subnormal mismatch is hidden.
                let result = try hostPreparation(
                    mixed: mixed, tail: tail, convolution: convolution,
                    a: a, b: b, dtBias: dtBias, aLog: aLog, captureConvolved: captureConvolved)
                guard shouldContinue() else { throw IOOperationCancelled() }
                preparationFallbacks += 1
                return result
            }
            return Prepared(
                query: read(qkv, count: rows * 2048), keys: read(qkv, count: rows * 2048, offset: rows * 2048),
                values: read(qkv, count: rows * 6144, offset: rows * 4096), beta: read(beta, count: a.count),
                decay: read(decay, count: a.count),
                convolved: convolved.map { read($0, count: mixed.count) }, nextTail: Self.nextTail(tail: tail, mixed: mixed))
        }
    }

    /// Buffer entry point for GDNDeviceBlock (any 1...256 rows, no row floor). The caller zeroes
    /// `exceptions`, joins, and replays the layer on the host path when it is set.
    func encodePreparation(
        _ command: MTLCommandBuffer, input: MTLBuffer, history: MTLBuffer,
        weights: MTLBuffer, aInput: MTLBuffer, bInput: MTLBuffer,
        bias: MTLBuffer, logs: MTLBuffer, qkv: MTLBuffer, beta: MTLBuffer,
        decay: MTLBuffer, convolved: MTLBuffer?, exceptions: MTLBuffer, rows: Int
    ) throws {
        guard (1...256).contains(rows), command.device.registryID == device.registryID else {
            throw EngineError("Invalid device GDN preparation geometry")
        }
        let spans =
            [
                (input, rows * 10240 * 4), (history, 3 * 10240 * 2), (weights, 4 * 10240 * 4),
                (aInput, rows * 48 * 4), (bInput, rows * 48 * 4), (bias, 48 * 4), (logs, 48 * 4),
                (qkv, rows * 10240 * 4), (beta, rows * 48 * 4), (decay, rows * 48 * 4), (exceptions, 4),
            ]
            + (convolved.map { [($0, rows * 10240 * 4)] } ?? [])
        for (buffer, bytes) in spans {
            guard buffer.device.registryID == device.registryID, buffer.storageMode == .shared, buffer.length >= bytes else {
                throw EngineError("Invalid device GDN preparation span")
            }
        }
        let convEncoder = try encoder(command)
        convEncoder.setComputePipelineState(preparation)
        for (index, buffer) in [input, history, weights, siluTable, qkv, convolved ?? qkv].enumerated() {
            convEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        let shape = [UInt32(rows), convolved != nil ? 1 : 0]
        // Exact SOURCE host scale, narrowed to the BF16 array dtype.
        let constants: [Float] = [1e-6, BF16.decode(BF16.encode(Float(pow(Double(128), -0.5))))]
        shape.withUnsafeBytes { convEncoder.setBytes($0.baseAddress!, length: $0.count, index: 6) }
        constants.withUnsafeBytes { convEncoder.setBytes($0.baseAddress!, length: $0.count, index: 7) }
        convEncoder.setBuffer(exceptions, offset: 0, index: 8)
        convEncoder.dispatchThreadgroups(
            MTLSize(width: rows * 80, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        convEncoder.endEncoding()
        if exactKernels { exactConvTiles += 1 }
        let factorEncoder = try encoder(command)
        factorEncoder.setComputePipelineState(factors)
        for (index, buffer) in [aInput, bInput, bias, logs, softplusTable, sigmoidTable, beta, decay].enumerated() {
            factorEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        var factorCount = UInt32(rows * 48)
        factorEncoder.setBytes(&factorCount, length: MemoryLayout<UInt32>.size, index: 8)
        factorEncoder.setBuffer(exceptions, offset: 0, index: 9)
        factorEncoder.dispatchThreads(
            MTLSize(width: rows * 48, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, factors.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        factorEncoder.endEncoding()
    }

    /// Equivalent to round(normed * arithmetic.sigmoidFP32(z)). The FP32 table
    /// deliberately retains the wide sigmoid, unlike the BF16 beta table.
    public func gate(normed: [Float], z: [Float], shouldContinue: () -> Bool = { true }) throws -> [Float] {
        guard !normed.isEmpty, normed.count == z.count, normed.count % 6144 == 0,
            (1...256).contains(normed.count / 6144)
        else { throw EngineError("Invalid fused GDN gating geometry") }
        guard shouldContinue() else { throw IOOperationCancelled() }
        return try autoreleasepool {
            let input = try copy(normed)
            let projected = try copy(z)
            let output = try allocate(normed.count * 4)
            let exceptions = try copy([UInt32(0)])
            let command = try commandBuffer("fused-gdn-wide-sigmoid-gate")
            let compute = try encoder(command)
            compute.setComputePipelineState(gating)
            for (index, buffer) in [input, projected, wideSigmoidTable, output].enumerated() {
                compute.setBuffer(buffer, offset: 0, index: index)
            }
            var count = UInt32(normed.count)
            compute.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            compute.setBuffer(exceptions, offset: 0, index: 5)
            compute.dispatchThreads(
                MTLSize(width: normed.count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, gating.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            compute.endEncoding()
            try complete(command, shouldContinue: shouldContinue)
            if exceptions.contents().load(as: UInt32.self) != 0 {
                let result = zip(normed, z).map { LayerArithmetic.round($0 * arithmetic.sigmoidFP32($1)) }
                guard shouldContinue() else { throw IOOperationCancelled() }
                gatingFallbacks += 1
                return result
            }
            return read(output, count: normed.count)
        }
    }

    /// Exposes the existing gate pipeline/table without changing the ordinary
    /// gate() control. The region owner joins and checks exceptions before use.
    func encodeGate(
        _ command: MTLCommandBuffer, normed: MTLBuffer, z: MTLBuffer,
        output: MTLBuffer, exceptions: MTLBuffer, tokens: Int, completeBlock: Bool = false
    ) throws {
        guard completeBlock ? (1...256).contains(tokens) : usesGPU(rows: tokens) else {
            throw EngineError("GDN region cannot enable a gate below its configured row floor")
        }
        for (buffer, bytes) in [
            (normed, tokens * 6144 * 4), (z, tokens * 6144 * 4),
            (output, tokens * 6144 * 4), (exceptions, 4),
        ] {
            guard buffer.device.registryID == device.registryID, buffer.storageMode == .shared, buffer.length >= bytes else {
                throw EngineError("Invalid embedded GDN gate buffer")
            }
        }
        let compute = try encoder(command)
        compute.setComputePipelineState(gating)
        for (index, buffer) in [normed, z, wideSigmoidTable, output].enumerated() { compute.setBuffer(buffer, offset: 0, index: index) }
        var count = UInt32(tokens * 6144)
        compute.setBytes(&count, length: 4, index: 4)
        compute.setBuffer(exceptions, offset: 0, index: 5)
        compute.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, gating.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        compute.endEncoding()
        embeddedGateEncoders += 1
    }

    func repairEmbeddedGate(normed: [Float], z: [Float]) -> [Float] {
        gatingFallbacks += 1
        return zip(normed, z).map { LayerArithmetic.round($0 * arithmetic.sigmoidFP32($1)) }
    }

    private static func nextTail(tail: [UInt16], mixed: [Float]) -> [UInt16] {
        let retained = min(3 * 10_240, mixed.count)
        var result = tail.suffix(3 * 10_240 - retained).map { BF16.encode(BF16.decode($0)) }
        result.reserveCapacity(3 * 10_240)
        result.append(contentsOf: mixed.suffix(retained).map(BF16.encode))
        return result
    }
    private func hostPreparation(
        mixed: [Float], tail: [UInt16], convolution: [Float], a: [Float], b: [Float],
        dtBias: [Float], aLog: [Float], captureConvolved: Bool
    ) throws -> Prepared {
        let rows = mixed.count / 10_240
        let raw = tail.map(BF16.decode) + mixed
        var conv = [Float](repeating: 0, count: mixed.count)
        for token in 0..<rows {
            for channel in 0..<10_240 {
                var sum: Float = 0
                for tap in 0..<4 { sum += raw[(token + tap) * 10_240 + channel] * convolution[channel * 4 + tap] }
                conv[token * 10_240 + channel] = arithmetic.silu(LayerArithmetic.round(sum))
            }
        }
        var query: [Float] = []
        var keys: [Float] = []
        var values: [Float] = []
        query.reserveCapacity(rows * 2048)
        keys.reserveCapacity(rows * 2048)
        values.reserveCapacity(rows * 6144)
        for token in 0..<rows {
            let start = token * 10_240
            query.append(contentsOf: conv[start..<(start + 2048)])
            keys.append(contentsOf: conv[(start + 2048)..<(start + 4096)])
            values.append(contentsOf: conv[(start + 4096)..<(start + 10_240)])
        }
        let beta = b.map(arithmetic.sigmoid)
        let softplus = a.enumerated().map { index, value in arithmetic.softplus(LayerArithmetic.round(value + dtBias[index % 48])) }
        let result = try reductions.prepareGDN(query: query, keys: keys, aLog: aLog, softplus: softplus)
        return Prepared(
            query: result.query, keys: result.keys, values: values, beta: beta, decay: result.decay,
            convolved: captureConvolved ? conv : nil, nextTail: raw.suffix(3 * 10_240).map(BF16.encode))
    }
    private func allocate(_ bytes: Int) throws -> MTLBuffer { try Self.allocate(device: device, ledger: ledger, bytes: bytes) }
    private static func allocate(device: MTLDevice, ledger: MemoryLedger, bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 256 * 10_240 * 4 else { throw EngineError("Fused GDN allocation exceeds one bounded tile") }
        let owner = try GDNPreparationAllocation(ledger: ledger, bytes: bytes)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared), UInt64(buffer.allocatedSize) <= owner.charged
        else {
            throw EngineError("Fused GDN allocation failed or exceeded admitted charge", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &GDNPreparationAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }
    private func copy<T>(_ values: [T]) throws -> MTLBuffer {
        let result = try allocate(values.count * MemoryLayout<T>.stride)
        values.withUnsafeBytes { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return result
    }
    private func read(_ buffer: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self).advanced(by: offset), count: count))
    }
    private func commandBuffer(_ label: String) throws -> MTLCommandBuffer {
        guard let result = queue.makeCommandBuffer() else {
            throw EngineError("Fused GDN command allocation failed", code: "insufficient_memory")
        }
        result.label = label
        return result
    }
    private func encoder(_ command: MTLCommandBuffer) throws -> MTLComputeCommandEncoder {
        guard let result = command.makeComputeCommandEncoder() else {
            throw EngineError("Fused GDN encoder allocation failed", code: "insufficient_memory")
        }
        return result
    }
    private func complete(_ command: MTLCommandBuffer, shouldContinue: () -> Bool) throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        command.commit()
        submittedCommands += 1
        try command.waitBounded()
        guard command.status == .completed else { throw EngineError.gpu("Fused GDN GPU failure", command.error) }
        completedCommands += 1
        guard shouldContinue() else { throw IOOperationCancelled() }
    }

    // Convolution: pinned conv.metal depthwise_conv_1d, four ascending taps.
    // Q/K: unchanged pinned_l2_norm128 reads4+SIMD sum and precise rsqrt.
    // Decay: unchanged pinned_gdn_decay precise exp graph. MIT attribution is
    // retained in THIRD_PARTY_NOTICES.md. Every unary table comes from the exact
    // selected LayerUnaryArithmetic policy, never a newly approximated function.
    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        inline float gdn_from_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
        inline float gdn_bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
            return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
        }
        inline uint gdn_key(float x) { return as_type<uint>(gdn_bf16(x)) >> 16; }
        // Conservative bounds send nonfinite/extreme tiles through the original
        // CPU arithmetic. The low bound also covers cancellation/product cases
        // that could underflow before a GPU result can reveal the lost subnormal.
        inline void gdn_check(float x, device atomic_uint *exceptions) {
            uint magnitude = as_type<uint>(x) & 0x7fffffffu;
            if ((magnitude != 0 && magnitude < 0x0d800000u) || magnitude >= 0x5d800000u)
                atomic_fetch_or_explicit(exceptions, 1u, memory_order_relaxed);
        }
        inline void gdn_check_product(float a, float b, device atomic_uint *exceptions) {
            uint aa = as_type<uint>(a) & 0x7fffffffu, bb = as_type<uint>(b) & 0x7fffffffu;
            if (aa != 0 && bb != 0 && ((aa >> 23) + (bb >> 23) <= 128u))
                atomic_fetch_or_explicit(exceptions, 1u, memory_order_relaxed);
        }
        // One 32-lane threadgroup per (token, 128-column head) of the 10240 mixed channels: rows * 80 groups,
        // lane L owns columns head * 128 + 4L ... +3. Heads 0-15 are q (L2-normalized, times constants[1]),
        // 16-31 k (L2-normalized), 32-79 v (SiLU output only). The L2 total adds the lane's four squares in
        // order, then simd_sum, then precise rsqrt(total + 1e-6). shape = [rows, write capture].
        kernel void gdn_conv_normalize(device const float *mixed [[buffer(0)]], device const ushort *tail [[buffer(1)]],
            device const float *weights [[buffer(2)]], device const ushort *silu [[buffer(3)]], device float *qkv [[buffer(4)]],
            device float *capture [[buffer(5)]], constant uint *shape [[buffer(6)]], constant float *constants [[buffer(7)]],
            device atomic_uint *exceptions [[buffer(8)]],
            uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint token = group / 80, head = group % 80, column = head * 128 + lane * 4;
            float values[4];
            for (uint j = 0; j < 4; ++j) {
                float sum = 0.0f;
                for (uint tap = 0; tap < 4; ++tap) {
                    uint row = token + tap;
                    float value = row < 3 ? gdn_from_bf16(tail[row * 10240 + column + j]) : mixed[(row - 3) * 10240 + column + j];
                    float weight = weights[(column + j) * 4 + tap]; gdn_check(value, exceptions); gdn_check(weight, exceptions);
                    gdn_check_product(value, weight, exceptions);
                    float product = value * weight;
                    gdn_check(product, exceptions);
                    sum = sum + product;
                    gdn_check(sum, exceptions);
                }
                values[j] = gdn_from_bf16(silu[gdn_key(sum)]);
                gdn_check(sum, exceptions); gdn_check(values[j], exceptions);
                if (shape[1]) capture[token * 10240 + column + j] = values[j];
            }
            if (head < 32) {
                float total = 0.0f;
                for (uint j = 0; j < 4; ++j) { float squared = values[j] * values[j]; total = squared + total; }
                total = simd_sum(total);
                float inverse = metal::precise::rsqrt(total + constants[0]);
                uint base = (head < 16 ? 0 : shape[0] * 2048) + token * 2048 + (head % 16) * 128 + lane * 4;
                for (uint j = 0; j < 4; ++j) {
                    float normalized = gdn_bf16(values[j] * inverse);
                    qkv[base + j] = head < 16 ? gdn_bf16(normalized * constants[1]) : normalized;
                }
            } else {
                uint base = shape[0] * 4096 + token * 6144 + (head - 32) * 128 + lane * 4;
                for (uint j = 0; j < 4; ++j) qkv[base + j] = values[j];
            }
        }
        // One thread per (token, value head): beta = sigmoid table[BF16(b)], decay = exp(-exp(BF16(A_log)) *
        // softplus table[BF16(a + dt_bias)]) with precise exp; both FP32 outputs, rows * 48 each.
        kernel void gdn_factors(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            device const float *bias [[buffer(2)]], device const float *logs [[buffer(3)]],
            device const ushort *softplus [[buffer(4)]], device const ushort *sigmoid [[buffer(5)]],
            device float *beta [[buffer(6)]], device float *decay [[buffer(7)]], constant uint &count [[buffer(8)]],
            device atomic_uint *exceptions [[buffer(9)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            gdn_check(a[index], exceptions); gdn_check(b[index], exceptions); gdn_check(bias[index % 48], exceptions); gdn_check(logs[index % 48], exceptions);
            beta[index] = gdn_from_bf16(sigmoid[gdn_key(b[index])]);
            float factor = gdn_from_bf16(softplus[gdn_key(a[index] + bias[index % 48])]);
            float logarithm = gdn_bf16(logs[index % 48]);
            float inner = metal::precise::exp(logarithm);
            float product = -inner * factor;
            decay[index] = metal::precise::exp(product);
        }
        // One thread per element: BF16(normed * sigmoidFP32 table[BF16(z)]); tiny or nonfinite operands flag.
        kernel void gdn_gate(device const float *normed [[buffer(0)]], device const float *z [[buffer(1)]],
            device const float *sigmoid [[buffer(2)]], device float *out [[buffer(3)]], constant uint &count [[buffer(4)]],
            device atomic_uint *exceptions [[buffer(5)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            float weight = sigmoid[gdn_key(z[index])];
            gdn_check(normed[index], exceptions); gdn_check(z[index], exceptions); gdn_check(weight, exceptions);
            gdn_check_product(normed[index], weight, exceptions);
            out[index] = gdn_bf16(normed[index] * weight);
        }
        """#
}
