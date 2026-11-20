import Foundation
import Metal

/// Fixed QSA assembly boundaries, each equal to the host path's scalar arithmetic (BITWISE; fp contract
/// off, precise math). No allocation, commit, state publication or
/// CPU readback occurs here. Flags require joined original-path replay; they
/// cannot be ignored because a downstream result happens to be finite.
final class QSAExecutionScalars {
    private let device: MTLDevice
    private let split, index, pool, gate, narrow: MTLComputePipelineState
    private let sigmoid: MTLBuffer

    init(device: MTLDevice, sigmoidBF16: MTLBuffer) throws {
        guard sigmoidBF16.device.registryID == device.registryID, sigmoidBF16.length >= 65536 * 2 else {
            throw EngineError("QSA scalar sigmoid domain/device differs")
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw EngineError("Missing QSA scalar kernel") }
            return try device.makeComputePipelineState(function: f)
        }
        split = try pipeline("qsa_split")
        index = try pipeline("qsa_index_split")
        pool = try pipeline("qsa_pool_tail")
        gate = try pipeline("qsa_gate")
        narrow = try pipeline("qsa_narrow")
        self.device = device
        sigmoid = sigmoidBF16
    }

    private func encode(
        _ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
        _ buffers: [MTLBuffer], sizes: [Int], shape: [UInt32], count: Int
    ) throws {
        guard count > 0, buffers.count == sizes.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.1 >= 0 && $0.0.length >= $0.1 }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            let e = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid QSA scalar buffer/command geometry") }
        e.setComputePipelineState(pipeline)
        for (i, buffer) in buffers.enumerated() { e.setBuffer(buffer, offset: 0, index: i) }
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
        e.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
    }

    func encodeSplit(_ command: MTLCommandBuffer, qg: MTLBuffer, queries: MTLBuffer, gates: MTLBuffer, rows: Int) throws {
        guard (1...256).contains(rows) else { throw EngineError("Invalid QSA split rows") }
        try encode(
            command, split, [qg, queries, gates], sizes: [rows * 12288 * 4, rows * 6144 * 4, rows * 6144 * 4],
            shape: [UInt32(rows * 6144)], count: rows * 6144)
    }

    /// Old raw tail is BF16; the new projected rows are FP32. Preserve that
    /// distinction until the four-row FP32 sum and final BF16 rounding.
    func encodeIndex(
        _ command: MTLCommandBuffer, qk: MTLBuffer, oldTail: MTLBuffer,
        queries: MTLBuffer, raw: MTLBuffer, rows: Int, tailRows: Int
    ) throws {
        guard (1...256).contains(rows), (0...3).contains(tailRows) else { throw EngineError("Invalid QSA index assembly rows") }
        try encode(
            command, index, [qk, oldTail, queries, raw],
            sizes: [rows * 640 * 4, tailRows * 128 * 2, rows * 512 * 4, (tailRows + rows) * 128 * 4],
            shape: [UInt32(rows), UInt32(tailRows)], count: rows * 512)
    }

    func encodePool(
        _ command: MTLCommandBuffer, raw: MTLBuffer, pooled: MTLBuffer,
        tail: MTLBuffer, exceptions: MTLBuffer, rows: Int, tailRows: Int
    ) throws {
        guard (1...256).contains(rows), (0...3).contains(tailRows) else { throw EngineError("Invalid QSA pool rows") }
        let total = rows + tailRows
        let blocks = total / 4
        let tailCount = total % 4 * 128
        // Empty outputs have zero logical extent and a distinct admitted dummy
        // backing. The shader never accesses their payload.
        try encode(
            command, pool, [raw, pooled, tail, exceptions],
            sizes: [total * 128 * 4, blocks * 128 * 4, tailCount * 2, 4],
            shape: [UInt32(blocks), UInt32(tailCount)], count: max(blocks * 128, tailCount))
    }

    func encodeGate(
        _ command: MTLCommandBuffer, attended: MTLBuffer, gates: MTLBuffer,
        output: MTLBuffer, exceptions: MTLBuffer, rows: Int
    ) throws {
        guard (1...256).contains(rows) else { throw EngineError("Invalid QSA gating rows") }
        try encode(
            command, gate, [attended, gates, sigmoid, output, exceptions],
            sizes: [rows * 6144 * 4, rows * 6144 * 4, 65536 * 2, rows * 6144 * 4, 4],
            shape: [UInt32(rows * 6144)], count: rows * 6144)
    }

    func encodeNarrow(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int) throws {
        guard count > 0, count <= 256 * 6144 else { throw EngineError("Invalid QSA narrowing extent") }
        try encode(command, narrow, [input, output], sizes: [count * 4, count * 2], shape: [UInt32(count)], count: count)
    }

    private static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp contract(off)
        inline ushort qsa_bf16(uint bits) {
            if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x40u);
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
        inline bool qsa_pool_exception(uint bits) {
            uint magnitude = bits & 0x7fffffffu, exponent = magnitude >> 23;
            // Conservative replay also covers cancellation/underflow of two
            // opposite normal values, and overflow in the sequential four adds.
            return magnitude != 0u && (exponent <= 26u || exponent >= 252u);
        }
        // Splits the interleaved Q/gate projection: per head 256 query values then 256 gate values.
        kernel void qsa_split(device const uint *qg [[buffer(0)]], device uint *q [[buffer(1)]],
            device uint *g [[buffer(2)]], constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint source = (i / 256u) * 512u + i % 256u;
            q[i] = qg[source]; g[i] = qg[source + 256u];
        }
        // Index projection row (640) -> 4 x 128 index queries and one 128-wide raw block key, appended after
        // the 0-3 BF16 raw rows left over from the previous step.
        kernel void qsa_index_split(device const uint *qk [[buffer(0)]], device const ushort *old [[buffer(1)]],
            device uint *q [[buffer(2)]], device uint *raw [[buffer(3)]], constant uint *shape [[buffer(4)]],
            uint i [[thread_position_in_grid]]) {
            uint rows = shape[0], tails = shape[1];
            if (i < rows * 512u) q[i] = qk[(i / 512u) * 640u + i % 512u];
            if (i < tails * 128u) raw[i] = uint(old[i]) << 16;
            if (i < rows * 128u) raw[tails * 128u + i] = qk[(i / 128u) * 640u + 512u + i % 128u];
        }
        // Pooled block key = BF16((r0 + r1 + r2 + r3) / 4), summed in order in FP32; leftover rows become the
        // new BF16 tail. Magnitudes near underflow or overflow set the flag (host replay).
        kernel void qsa_pool_tail(device const uint *raw [[buffer(0)]], device uint *pooled [[buffer(1)]],
            device ushort *tail [[buffer(2)]], device atomic_uint &flag [[buffer(3)]],
            constant uint *shape [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            uint blocks = shape[0], tailCount = shape[1];
            if (i < tailCount) tail[i] = qsa_bf16(raw[blocks * 512u + i]);
            if (i >= blocks * 128u) return;
            float total = 0.0f; bool exceptional = false;
            for (uint row = 0; row < 4; ++row) {
                uint bits = raw[(i / 128u * 4u + row) * 128u + i % 128u];
                exceptional |= qsa_pool_exception(bits);
                total = total + as_type<float>(bits);
                exceptional |= qsa_pool_exception(as_type<uint>(total));
            }
            if (exceptional) { atomic_store_explicit(&flag, 1u, memory_order_relaxed); pooled[i] = 0u; }
            else pooled[i] = uint(qsa_bf16(as_type<uint>(total / 4.0f))) << 16;
        }
        // output = BF16(attended x sigmoid), sigmoid from the 65,536-entry BF16 table indexed by BF16(gate);
        // non-finite, subnormal or range-edge operands set the flag (host replay).
        kernel void qsa_gate(device const uint *attended [[buffer(0)]], device const uint *gates [[buffer(1)]],
            device const ushort *sigmoid [[buffer(2)]], device uint *output [[buffer(3)]],
            device atomic_uint &flag [[buffer(4)]], constant uint &count [[buffer(5)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint a = attended[i], g = gates[i], b = uint(sigmoid[qsa_bf16(g)]) << 16;
            uint ea = (a >> 23) & 255u, eb = (b >> 23) & 255u, eg = (g >> 23) & 255u;
            bool na = (a & 0x7fffffffu) != 0u, nb = (b & 0x7fffffffu) != 0u;
            bool special = ea == 255u || eb == 255u || eg == 255u ||
                (na && ea == 0u) || (nb && eb == 0u) ||
                (na && nb && (ea + eb <= 128u || ea + eb >= 380u));
            if (special) { atomic_store_explicit(&flag, 1u, memory_order_relaxed); output[i] = 0u; }
            else output[i] = uint(qsa_bf16(as_type<uint>(as_type<float>(a) * as_type<float>(b)))) << 16;
        }
        // FP32 -> BF16, round to nearest even, NaN quieted.
        kernel void qsa_narrow(device const uint *input [[buffer(0)]], device ushort *output [[buffer(1)]],
            constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < count) output[i] = qsa_bf16(input[i]);
        }
        """#
}
