import Foundation
import Metal

/// Multi-row (C256, two-row and row-one) versions of the decode scalar kernels, built by patching the
/// DecodeFrontierScalars and FusedResidualInjection sources at fixed anchor strings; each anchor must occur exactly
/// once or init throws, so a comment that repeats an anchor in those sources stops the engine from starting. Only
/// addresses, names and the work count change: every FP32/BF16 expression and exceptional-value flag (a flagged tile
/// takes the host replay) is the row-one original's, so each element is bit-identical (BITWISE). Kernels:
/// hc_decode_narrow (FP32 -> BF16), hc_decode_lookup_float (BF16 sigmoid table), whole_router_gather (the logits of
/// each row's top-10 pairs), whole_shared_product (SiLU table x up, BF16), whole_shared_combine (routed + shared x
/// gate per token, the injection kernel's expression) and whole_convolution_tail (below).
package final class WholeExecutionScalars {
    private let device: MTLDevice
    private let narrow, sigmoid, gather, product, combine, convolutionTail: MTLComputePipelineState
    private let sigmoidTable, siluTable: MTLBuffer

    package init(device: MTLDevice, sigmoidTable: MTLBuffer, siluTable: MTLBuffer) throws {
        guard [sigmoidTable, siluTable].allSatisfy({ $0.device.registryID == device.registryID && $0.length >= 65536 * 2 }) else {
            throw EngineError("Whole scalar unary table domain/device differs")
        }
        func replace(_ source: String, _ old: String, _ new: String) throws -> String {
            guard source.components(separatedBy: old).count == 2 else { throw EngineError("Whole scalar original source seam changed") }
            return source.replacingOccurrences(of: old, with: new)
        }
        var shared = DecodeFrontierScalars.source
        shared = try replace(shared, "kernel void frontier_router_gather", "kernel void whole_router_gather")
        shared = try replace(
            shared, "&exceptional [[buffer(3)]], uint i", "&exceptional [[buffer(3)]], constant uint &count [[buffer(4)]], uint i")
        shared = try replace(shared, "if (i >= 10) return;", "if (i >= count) return;")
        shared = try replace(shared, "else selected[i] = logits[index];", "else selected[i] = logits[(i / 10u) * 512u + index];")
        shared = try replace(shared, "kernel void frontier_shared_product", "kernel void whole_shared_product")
        shared = try replace(
            shared, "&exceptional [[buffer(4)]],\n    uint i", "&exceptional [[buffer(4)]], constant uint &count [[buffer(5)]],\n    uint i"
        )
        shared = try replace(shared, "if (i >= 640) return;", "if (i >= count) return;")
        var combined = FusedResidualInjection.source
        combined = try replace(combined, "kernel void residual_injection", "kernel void whole_shared_combine")
        combined = try replace(
            combined, "uint token = i / 10240, branch = (i % 10240) / 2560, column = i % 2560;",
            "uint token = i / 2560, column = i % 2560;")
        combined = try replace(
            combined, "float a = x[token * 2560 + column], b = g[token * 4 + branch], c = r[i];",
            "float a = x[token * 2560 + column], b = g[token], c = r[i];")
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(
            source: ReferenceKernels.quantizedBF16Preamble + "\n" + FusedHCDecode.handoffSource
                + "\n" + shared + "\n" + combined + "\n" + Self.convolutionSource, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw EngineError("Missing whole scalar pipeline") }
            return try device.makeComputePipelineState(function: f)
        }
        narrow = try pipeline("hc_decode_narrow")
        sigmoid = try pipeline("hc_decode_lookup_float")
        gather = try pipeline("whole_router_gather")
        product = try pipeline("whole_shared_product")
        combine = try pipeline("whole_shared_combine")
        convolutionTail = try pipeline("whole_convolution_tail")
        self.device = device
        self.sigmoidTable = sigmoidTable
        self.siluTable = siluTable
    }
    private func encoder(_ command: MTLCommandBuffer, buffers: [MTLBuffer], sizes: [Int]) throws -> MTLComputeCommandEncoder {
        guard buffers.count == sizes.count, Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            let e = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid whole scalar buffer spans/command") }
        return e
    }
    private func dispatch(_ e: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, count: Int, width: Int = 256) {
        e.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
    }
    /// `inputOffset` (bytes, a multiple of 4) narrows a trailing span of `input`: the double-buffered
    /// prefill chain reads the previous tile's last projected rows as the next tile's history.
    package func encodeNarrow(_ command: MTLCommandBuffer, input: MTLBuffer, inputOffset: Int = 0, output: MTLBuffer, count: Int) throws {
        guard (1...(256 * 10240)).contains(count), inputOffset >= 0, inputOffset % 4 == 0 else {
            throw EngineError("Whole narrowing extent invalid")
        }
        let e = try encoder(command, buffers: [input, output], sizes: [inputOffset + count * 4, count * 2])
        e.setComputePipelineState(narrow)
        e.setBuffer(input, offset: inputOffset, index: 0)
        e.setBuffer(output, offset: 0, index: 1)
        var n = UInt32(count)
        e.setBytes(&n, length: 4, index: 2)
        dispatch(e, narrow, count: count)
    }
    func encodeSigmoid(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, rows: Int) throws {
        guard (1...256).contains(rows) else { throw EngineError("Whole sigmoid rows invalid") }
        let e = try encoder(command, buffers: [input, output, sigmoidTable], sizes: [rows * 4, rows * 4, 65536 * 2])
        e.setComputePipelineState(sigmoid)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(output, offset: 0, index: 1)
        var count = UInt32(rows)
        e.setBytes(&count, length: 4, index: 2)
        e.setBuffer(sigmoidTable, offset: 0, index: 3)
        dispatch(e, sigmoid, count: rows)
    }
    package func encodeProduct(
        _ command: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer, output: MTLBuffer, exceptions: MTLBuffer, rows: Int
    )
        throws
    {
        guard (1...256).contains(rows) else { throw EngineError("Whole shared product rows invalid") }
        let e = try encoder(
            command, buffers: [gate, up, siluTable, output, exceptions],
            sizes: [rows * 640 * 4, rows * 640 * 4, 65536 * 2, rows * 640 * 2, 4])
        e.setComputePipelineState(product)
        for (i, b) in [gate, up, siluTable, output, exceptions].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var count = UInt32(rows * 640)
        e.setBytes(&count, length: 4, index: 5)
        dispatch(e, product, count: Int(count), width: 64)
    }
    package func encodeRouterGather(
        _ command: MTLCommandBuffer, logits: MTLBuffer, pairs: MTLBuffer, selected: MTLBuffer,
        exceptions: MTLBuffer, rows: Int
    ) throws {
        guard (1...256).contains(rows) else { throw EngineError("Whole router gather rows invalid") }
        let e = try encoder(
            command, buffers: [logits, pairs, selected, exceptions], sizes: [rows * 512 * 4, rows * 10 * 8, rows * 10 * 4, 4])
        e.setComputePipelineState(gather)
        for (i, b) in [logits, pairs, selected, exceptions].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var count = UInt32(rows * 10)
        e.setBytes(&count, length: 4, index: 4)
        dispatch(e, gather, count: Int(count))
    }
    func encodeCombine(
        _ command: MTLCommandBuffer, routed: MTLBuffer, shared: MTLBuffer, gates: MTLBuffer,
        output: MTLBuffer, exceptions: MTLBuffer, rows: Int
    ) throws {
        guard (1...256).contains(rows) else { throw EngineError("Whole shared combination rows invalid") }
        let e = try encoder(
            command, buffers: [shared, routed, gates, output, exceptions],
            sizes: [rows * 2560 * 4, rows * 2560 * 4, rows * 4, rows * 2560 * 4, 4])
        e.setComputePipelineState(combine)
        for (i, b) in [shared, routed, gates, output].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var count = UInt32(rows * 2560)
        e.setBytes(&count, length: 4, index: 4)
        e.setBuffer(exceptions, offset: 0, index: 5)
        dispatch(e, combine, count: Int(count))
    }
    /// Bit-identical to BF16.encode(BF16.decode(oldTail)) followed by
    /// BF16.encode(currentMixed); quiet NaN payloads and ties-even are explicit.
    package func encodeConvolutionTail(
        _ command: MTLCommandBuffer, history: MTLBuffer, mixed: MTLBuffer,
        next: GDNRecurrentStorage
    ) throws {
        guard let offset = next.convolutionByteOffset, offset >= 0, offset + 61440 <= next.buffer.length else {
            throw EngineError("Convolution tail output was not admitted")
        }
        let e = try encoder(command, buffers: [history, mixed, next.buffer], sizes: [61440, 40960, offset + 61440])
        e.setComputePipelineState(convolutionTail)
        e.setBuffer(history, offset: 0, index: 0)
        e.setBuffer(mixed, offset: 0, index: 1)
        e.setBuffer(next.buffer, offset: offset, index: 2)
        dispatch(e, convolutionTail, count: 30720)
    }
    private static let convolutionSource = #"""
        // whole_convolution_tail: a decode row's next three-row BF16 convolution history (30,720 threads): rows 1 and 2
        // of the old history re-encoded (this only quiets NaNs), then the new FP32 row narrowed with ties-to-even.
        inline ushort whole_tail_bf16(uint bits) {
            if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
                return ushort(bits >> 16) | 0x0040u;
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
        kernel void whole_convolution_tail(device const ushort *history [[buffer(0)]],
            device const float *mixed [[buffer(1)]], device ushort *tail [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= 30720u) return;
            uint bits = i < 20480u ? uint(history[i + 10240u]) << 16 : as_type<uint>(mixed[i - 20480u]);
            tail[i] = whole_tail_bf16(bits);
        }
        """#

}
