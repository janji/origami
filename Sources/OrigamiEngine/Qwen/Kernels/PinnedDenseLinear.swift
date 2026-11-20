import Foundation
import Metal
import ObjectiveC

private final class PinnedDenseAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "pinned_dense_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum PinnedDenseAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The checkpoint's plain BF16 dense projections, with FP32 weights resident: router 2560 -> 512, shared-expert gate
/// 2560 -> 1, GDN a/b 2560 -> 48, QSA index 2560 -> 640, HC injection 10240 -> 4. Plans and kernels transcribe pinned
/// MLX 1f8e74e3's dispatch (scalar dot, GEMV, split-K) as MLX selects it on an M2-class GPU; every Mac runs the same
/// plan, because the plan fixes the rounding order. BF16 mode narrows inputs and the output to BF16; split-K narrows
/// only after the sequential FP32 sum of its partials. The router keeps FP32 inputs, weights and output.
public final class PinnedDenseLinear {
    /// One dense dispatch: scalar dot (one row, one output), GEMV (`matrixIsInput` when the activation rows are the
    /// matrix, i.e. one output) or split-K MMA with `splitKPartitions` partitions of `splitKPartitionSize` columns.
    public struct Plan: Codable, Equatable, Sendable {
        public enum Kind: String, Codable, Sendable { case dot, gemv, splitK }
        public let kind: Kind
        public let matrixIsInput: Bool
        public let reductionSIMDGroups, splitKPartitions, splitKPartitionSize: Int
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let ledger: MemoryLedger
    private let dot, gemv, splitK, accumulate: [MTLComputePipelineState]

    public init(ledger: MemoryLedger) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Pinned dense Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipelines(_ name: String) throws -> [MTLComputePipelineState] {
            try [false, true].map { enabled in
                let constants = MTLFunctionConstantValues()
                var enabled = enabled
                constants.setConstantValue(&enabled, type: .bool, index: 0)
                return try device.makeComputePipelineState(function: library.makeFunction(name: name, constantValues: constants))
            }
        }
        dot = try pipelines("origami_dense_dot")
        gemv = try pipelines("origami_dense_mv")
        splitK = try pipelines("origami_dense_splitk")
        accumulate = try pipelines("origami_dense_accumulate")
        guard gemv.allSatisfy({ $0.maxTotalThreadsPerThreadgroup >= 256 }) else {
            throw EngineError("Pinned dense requires the source 256-thread narrow GEMV reduction")
        }
        guard dot.allSatisfy({ $0.maxTotalThreadsPerThreadgroup >= 512 }) else {
            throw EngineError("Pinned scalar dot requires the source 512-thread reduction")
        }
        self.device = device
        self.queue = queue
        self.ledger = ledger
    }

    /// The five plain dense shapes actually present in the fixed text model.
    /// Unknown shapes fail closed rather than silently choosing another order.
    public static func plan(rows: Int, width: Int, outputs: Int) throws -> Plan {
        guard (1...256).contains(rows),
            (width == 2560 && [1, 48, 512, 640].contains(outputs)) || (width == 10240 && outputs == 4)
        else {
            throw EngineError("Pinned dense projection is outside the fixed checkpoint geometry")
        }
        if rows == 1 && outputs == 1 {
            // Contiguous M=N=1 precedes GEMV in pinned matmul.cpp. At K2560
            // there is one FP32 partial, followed only by the output cast.
            return Plan(
                kind: .dot, matrixIsInput: true, reductionSIMDGroups: 16,
                splitKPartitions: 0, splitKPartitionSize: 0)
        }
        if min(rows, outputs) == 1 {
            let matrixIsInput = outputs == 1
            let matrixRows = matrixIsInput ? rows : outputs
            return Plan(
                kind: .gemv, matrixIsInput: matrixIsInput,
                reductionSIMDGroups: width >= 16 * matrixRows ? 8 : 1,
                splitKPartitions: 0, splitKPartitionSize: 0)
        }
        // All remaining supported M/N/K meet the M2 medium-device split-K gate:
        // ceil(M/16)*ceil(N/16)<=2048, K/16>=8, and K>=max(M,N).
        let tileRows = (rows + 31) / 32
        let tileColumns = (outputs + 31) / 32
        let target = (width / 16) / (tileRows * tileColumns)
        var power = 1
        while power < target { power *= 2 }
        let partitions = min(max(2, power), 32)
        return Plan(
            kind: .splitK, matrixIsInput: false, reductionSIMDGroups: 0,
            splitKPartitions: partitions, splitKPartitionSize: ((width / 16) / partitions) * 16)
    }

    package func executionPlan(rows: Int, width: Int, outputs: Int) throws -> Plan {
        try Self.plan(rows: rows, width: width, outputs: outputs)
    }

    public func linear(_ input: [Float], weights: [Float], width: Int, outputs: Int, outputBF16: Bool) throws -> [Float] {
        guard width > 0, outputs > 0, input.count % width == 0 else { throw EngineError("Invalid pinned dense arrays") }
        _ = try Self.plan(rows: input.count / width, width: width, outputs: outputs)
        guard weights.count == width * outputs else { throw EngineError("Pinned dense weight geometry mismatch") }
        return try autoreleasepool {
            let buffer = try copy(weights)
            return try linear(input, weights: buffer, width: width, outputs: outputs, outputBF16: outputBF16)
        }
    }

    /// Uses the existing widened resident weight allocation directly. The caller
    /// retains its lease through this synchronous command's completed return.
    public func linear(_ input: [Float], weights: MTLBuffer, width: Int, outputs: Int, outputBF16: Bool) throws -> [Float] {
        guard width > 0, outputs > 0, input.count % width == 0 else { throw EngineError("Invalid pinned dense input geometry") }
        let rows = input.count / width
        let plan = try executionPlan(rows: rows, width: width, outputs: outputs)
        guard weights.device.registryID == device.registryID, weights.length >= width * outputs * 4 else {
            throw EngineError("Pinned dense resident buffer is too short or belongs to another device")
        }
        return try autoreleasepool {
            let x = try copy(input)
            let result = try allocate(rows * outputs * 4)
            guard let command = queue.makeCommandBuffer() else {
                throw EngineError("Pinned dense command allocation failed", code: "insufficient_memory")
            }
            command.label = "pinned-dense-" + plan.kind.rawValue
            let dtype = outputBF16 ? 1 : 0
            if plan.kind == .dot {
                try encode(
                    command, pipeline: dot[dtype], buffers: [x, weights, result], shape: [UInt32(width)],
                    grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 512, height: 1, depth: 1))
            } else if plan.kind == .splitK {
                let partials = try allocate(rows * outputs * plan.splitKPartitions * 4)
                let shape = [UInt32(rows), UInt32(outputs), UInt32(width), UInt32(plan.splitKPartitions), UInt32(plan.splitKPartitionSize)]
                try encode(
                    command, pipeline: splitK[dtype], buffers: [x, weights, partials], shape: shape,
                    grid: MTLSize(width: (outputs + 7) / 8, height: (rows + 7) / 8, depth: plan.splitKPartitions),
                    group: MTLSize(width: 32, height: 1, depth: 1))
                try encode(
                    command, pipeline: accumulate[dtype], buffers: [partials, result], shape: shape,
                    grid: MTLSize(width: (rows * outputs + 255) / 256, height: 1, depth: 1),
                    group: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                let matrixRows = plan.matrixIsInput ? rows : outputs
                let vectorRows = plan.matrixIsInput ? 1 : rows
                let shape = [UInt32(width), UInt32(matrixRows), UInt32(vectorRows), UInt32(plan.reductionSIMDGroups)]
                try encode(
                    command, pipeline: gemv[dtype], buffers: plan.matrixIsInput ? [x, weights, result] : [weights, x, result], shape: shape,
                    grid: MTLSize(width: matrixRows, height: vectorRows, depth: 1),
                    group: MTLSize(width: 32, height: plan.reductionSIMDGroups, depth: 1))
            }
            command.commit()
            try command.waitBounded()
            guard command.status == .completed else { throw EngineError.gpu("Pinned dense dispatch failed", command.error) }
            return Array(UnsafeBufferPointer(start: result.contents().assumingMemoryBound(to: Float.self), count: rows * outputs))
        }
    }

    private func allocate(_ bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 32 * 1024 * 1024 else { throw EngineError("Pinned dense allocation exceeds bounded workspace") }
        let page = Int(getpagesize())
        let charged = (bytes + page - 1) / page * page
        let allocation = try PinnedDenseAllocation(ledger: ledger, bytes: UInt64(charged))
        guard let result = device.makeBuffer(length: bytes, options: .storageModeShared), result.allocatedSize <= charged else {
            throw EngineError("Pinned dense buffer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(result, &PinnedDenseAssociation.key, allocation, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return result
    }

    /// The scalar dot and multi-row shared gate share linear()'s dispatch.
    package func encodeSharedGateRow(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer
    ) throws {
        try encodeSharedGateRows(command, input: input, weights: weights, output: output, rows: 1)
    }

    /// Scalar dot at one row, otherwise output-width-one GEMV including its
    /// 160/161-row switch. Only the enclosing command/buffer owner changes.
    package func encodeSharedGateRows(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, rows: Int
    ) throws {
        let plan = try executionPlan(rows: rows, width: 2560, outputs: 1)
        guard plan.matrixIsInput, plan.kind == .dot || plan.kind == .gemv,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            [input, weights, output].allSatisfy({ $0.device.registryID == device.registryID && $0.storageMode == .shared }),
            input.length >= rows * 2560 * 4, weights.length >= 2560 * 4, output.length >= rows * 4,
            output !== input, output !== weights
        else {
            throw EngineError("Invalid shared gate actual-row dense buffers/schedule")
        }
        if plan.kind == .dot {
            try encode(
                command, pipeline: dot[1], buffers: [input, weights, output], shape: [2560],
                grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 512, height: 1, depth: 1))
        } else {
            let shape: [UInt32] = [2560, UInt32(rows), 1, UInt32(plan.reductionSIMDGroups)]
            try encode(
                command, pipeline: gemv[1], buffers: [input, weights, output], shape: shape,
                grid: MTLSize(width: rows, height: 1, depth: 1),
                group: MTLSize(width: 32, height: plan.reductionSIMDGroups, depth: 1))
        }
    }
    /// Existing dense projection encoded into a larger joined command. Caller
    /// owns the unchanged Float split-K partials until their final source-ordered
    /// accumulation and BF16 boundary complete.
    @discardableResult
    package func encodeRows(
        _ command: MTLCommandBuffer, input: MTLBuffer, weights: MTLBuffer, output: MTLBuffer,
        partials: MTLBuffer?, rows: Int, width: Int, outputs: Int, outputBF16: Bool
    ) throws -> Plan {
        let plan = try executionPlan(rows: rows, width: width, outputs: outputs)
        guard !plan.matrixIsInput, command.commandQueue.device.registryID == device.registryID,
            [input, weights, output].allSatisfy({ $0.device.registryID == device.registryID && $0.storageMode == .shared }),
            input.length >= rows * width * 4, weights.length >= width * outputs * 4, output.length >= rows * outputs * 4
        else {
            throw EngineError("Encoded dense buffer/geometry mismatch")
        }
        let dtype = outputBF16 ? 1 : 0
        if plan.kind == .splitK {
            guard let partials, partials.device.registryID == device.registryID, partials.storageMode == .shared,
                partials.length >= rows * outputs * plan.splitKPartitions * 4
            else {
                throw EngineError("Encoded dense split-K workspace is not admitted")
            }
            let shape = [UInt32(rows), UInt32(outputs), UInt32(width), UInt32(plan.splitKPartitions), UInt32(plan.splitKPartitionSize)]
            try encode(
                command, pipeline: splitK[dtype], buffers: [input, weights, partials], shape: shape,
                grid: MTLSize(width: (outputs + 7) / 8, height: (rows + 7) / 8, depth: plan.splitKPartitions),
                group: MTLSize(width: 32, height: 1, depth: 1))
            try encode(
                command, pipeline: accumulate[dtype], buffers: [partials, output], shape: shape,
                grid: MTLSize(width: (rows * outputs + 255) / 256, height: 1, depth: 1),
                group: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            let shape = [UInt32(width), UInt32(outputs), UInt32(rows), UInt32(plan.reductionSIMDGroups)]
            try encode(
                command, pipeline: gemv[dtype], buffers: [weights, input, output], shape: shape,
                grid: MTLSize(width: outputs, height: rows, depth: 1),
                group: MTLSize(width: 32, height: plan.reductionSIMDGroups, depth: 1))
        }
        return plan
    }
    private func copy(_ values: [Float]) throws -> MTLBuffer {
        let result = try allocate(values.count * 4)
        values.withUnsafeBytes { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return result
    }
    private func encode(
        _ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: [MTLBuffer],
        shape: [UInt32], grid: MTLSize, group: MTLSize
    ) throws {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Pinned dense encoder allocation failed", code: "insufficient_memory")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }

    /// `hc_decode_fusion` compiles its fused injection after this exact text with the same options.
    static var metalSource: String { source }

    private static let source = #"""
        #include <metal_stdlib>
        #include <metal_simdgroup>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        constant bool origami_dense_bf16 [[function_constant(0)]];
        inline float typed(float value) { return origami_dense_bf16 ? float(bfloat(value)) : value; }

        // Pinned dot.h: 32 items/thread, 512 threads, dtype-dependent vector width,
        // four independent accumulators and two SIMD sums. K2560 has one partial;
        // its subsequent single-element FP32 reduction/copy adds no arithmetic.
        kernel void origami_dense_dot(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            device float *output [[buffer(2)]], constant uint *shape [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
            uint row [[threadgroup_position_in_grid]]) {
            const uint vec = origami_dense_bf16 ? 8 : 4;
            uint start = sg * 32 * 32 + lane * vec, n = shape[0];
            a += row * n;
            float4 c = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 32; i += vec) {
                uint idx = start + i * 32;
                if (idx + vec <= n) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < vec; j += 4) {
                        float4 av = float4(typed(a[idx+j]), typed(a[idx+j+1]), typed(a[idx+j+2]), typed(a[idx+j+3]));
                        float4 bv = float4(typed(b[idx+j]), typed(b[idx+j+1]), typed(b[idx+j+2]), typed(b[idx+j+3]));
                        c += av * bv;
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < vec; ++j) {
                        if (idx + j < n) c[j & 3] += typed(a[idx+j]) * typed(b[idx+j]);
                    }
                }
            }
            threadgroup float sums[16];
            float sum = c[0] + c[1] + c[2] + c[3];
            sum = simd_sum(sum);
            if (lane == 0) sums[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16) {
                sum = sums[tid]; sum = simd_sum(sum);
                if (tid == 0) output[row] = typed(sum);
            }
        }

        // One output per threadgroup preserves GEMVKernel's exact four-adjacent-K
        // lane accumulation, descending SIMD shuffles, and ascending BN sum. BM/TM
        // affect only independent output packing in the pinned kernel.
        kernel void origami_dense_mv(device const float *matrix [[buffer(0)]], device const float *vectors [[buffer(1)]],
            device float *output [[buffer(2)]], constant uint *shape [[buffer(3)]],
            uint3 block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint k = shape[0], n = shape[1], groups = shape[3];
            uint index = sg * 128 + lane * 4;
            float sum = 0;
            for (uint base = 0; base < k; base += groups * 128) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    uint column = base + index + i;
                    float coefficient = column < k ? typed(vectors[block.y * k + column]) : 0;
                    float value = column < k ? typed(matrix[block.x * k + column]) : 0;
                    sum += value * coefficient;
                }
            }
            #pragma clang loop unroll(full)
            for (ushort offset = 16; offset >= 1; offset >>= 1) sum += simd_shuffle_down(sum, offset);
            threadgroup float partials[8];
            if (lane == 0) partials[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && lane == 0) {
                for (uint part = 1; part < groups; ++part) sum += partials[part];
                output[block.y * n + block.x] = typed(sum);
            }
        }

        // Steel BlockMMA uses float 8x8 fragments even for BF16 inputs. Each output
        // fragment advances K by eight in order; BM/BN only pack independent fragments.
        kernel void origami_dense_splitk(device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
            device float *partials [[buffer(2)]], constant uint *shape [[buffer(3)]],
            uint3 block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint m = shape[0], n = shape[1], k = shape[2], partitions = shape[3], span = shape[4];
            uint rowBase = block.y * 8, colBase = block.x * 8, start = block.z * span;
            uint end = block.z + 1 == partitions ? k : start + span;
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = start; kk < end; kk += 8) {
                for (uint index = lane; index < 64; index += 32) {
                    uint row = index / 8, col = index % 8;
                    as[index] = rowBase + row < m ? typed(input[(rowBase + row) * k + kk + col]) : 0;
                    bs[index] = colBase + col < n ? typed(weights[(colBase + col) * k + kk + row]) : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) {
                uint row = rowBase + index / 8, col = colBase + index % 8;
                if (row < m && col < n) partials[(block.z * m + row) * n + col] = cs[index];
            }
        }
        // Split-K finish: one thread per output sums the partitions in ascending order in FP32, then the output boundary.
        kernel void origami_dense_accumulate(device const float *partials [[buffer(0)]], device float *output [[buffer(1)]],
            constant uint *shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            uint size = shape[0] * shape[1]; if (index >= size) return;
            float sum = 0;
            for (uint part = 0; part < shape[3]; ++part) sum += partials[part * size + index];
            output[index] = typed(sum);
        }
        """#
}
