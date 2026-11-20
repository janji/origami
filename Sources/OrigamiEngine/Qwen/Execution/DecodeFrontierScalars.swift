import Darwin
import Foundation
import Metal
import ObjectiveC

private final class FrontierScalarAllocation {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(_ ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "decode_frontier_sigmoid_table")
    }
    deinit { try? ledger.release(reservation) }
}
private enum FrontierScalarAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Row-one (decode) scalar kernels of the whole schedule: FP32-to-BF16 narrowing, the shared
/// expert's sigmoid gate and SiLU-times-up product through pinned 65,536-entry BF16 tables, and
/// the router's top-ten logit gather. Each is the host expression evaluated per element (no
/// reduction), so results are BITWISE equal to it; exceptional operands raise an exception word
/// and the caller replays on the host. No command is committed here; the caller owns flags,
/// buffers and joined original-path replay. WholeExecutionSchedule's scalars patch `source`
/// textually (WholeExecutionScalars), so its kernel text is a seam.
package final class DecodeFrontierScalars {
    private let device: MTLDevice
    private let narrow, sigmoid, gather, product: MTLComputePipelineState
    private let sigmoidTable, siluTable: MTLBuffer

    // Shared immutable owner: the whole-schedule QSA gate uses this same BF16
    // sigmoid domain, with no duplicate table or uncharged allocation.
    package var bf16SigmoidTable: MTLBuffer { sigmoidTable }

    package init(
        device: MTLDevice, ledger: MemoryLedger, arithmetic: LayerUnaryArithmetic,
        fused: FusedExpertExecution
    ) throws {
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(
            source: ReferenceKernels.quantizedBF16Preamble
                + "\n" + FusedHCDecode.handoffSource + "\n" + Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw EngineError("Missing decode frontier scalar kernel") }
            return try device.makeComputePipelineState(function: function)
        }
        narrow = try pipeline("hc_decode_narrow")
        sigmoid = try pipeline("hc_decode_lookup_float")
        gather = try pipeline("frontier_router_gather")
        product = try pipeline("frontier_shared_product")
        let bytes = 65536 * 2
        let page = Int(getpagesize())
        let charge = (bytes + page - 1) / page * page
        let owner = try FrontierScalarAllocation(ledger, bytes: UInt64(charge))
        guard let table = device.makeBuffer(length: bytes, options: .storageModeShared), table.allocatedSize <= charge else {
            throw EngineError("Decode frontier sigmoid table exceeds admitted allocation", code: "insufficient_memory")
        }
        objc_setAssociatedObject(table, &FrontierScalarAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        let values = table.contents().assumingMemoryBound(to: UInt16.self)
        for bits in 0..<65536 { values[bits] = BF16.encode(arithmetic.sigmoid(BF16.decode(UInt16(bits)))) }
        let silu = fused.decodeRegionSiLUTable
        guard silu.device.registryID == device.registryID, silu.length >= bytes else {
            throw EngineError("Decode frontier shared unary table differs from device/domain")
        }
        self.device = device
        sigmoidTable = table
        siluTable = silu
    }

    private func encoder(_ command: MTLCommandBuffer, buffers: [MTLBuffer], sizes: [Int]) throws -> MTLComputeCommandEncoder {
        guard buffers.count == sizes.count,
            zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            let encoder = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid decode scalar buffers/command") }
        return encoder
    }

    func encodeNarrow(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int) throws {
        guard [640, 2560, 10240].contains(count) else { throw EngineError("Unexpected decode narrowing extent") }
        let e = try encoder(command, buffers: [input, output], sizes: [count * 4, count * 2])
        e.setComputePipelineState(narrow)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(output, offset: 0, index: 1)
        var count = UInt32(count)
        e.setBytes(&count, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, narrow.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
    }

    func encodeSharedSigmoid(_ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer) throws {
        let e = try encoder(command, buffers: [input, output, sigmoidTable], sizes: [4, 4, 65536 * 2])
        e.setComputePipelineState(sigmoid)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(output, offset: 0, index: 1)
        var count: UInt32 = 1
        e.setBytes(&count, length: 4, index: 2)
        e.setBuffer(sigmoidTable, offset: 0, index: 3)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.endEncoding()
    }

    package func encodeRouterGather(
        _ command: MTLCommandBuffer, logits: MTLBuffer, pairs: MTLBuffer,
        selected: MTLBuffer, exceptions: MTLBuffer
    ) throws {
        let e = try encoder(command, buffers: [logits, pairs, selected, exceptions], sizes: [512 * 4, 10 * 8, 10 * 4, 4])
        e.setComputePipelineState(gather)
        for (index, buffer) in [logits, pairs, selected, exceptions].enumerated() { e.setBuffer(buffer, offset: 0, index: index) }
        e.dispatchThreads(MTLSize(width: 10, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 10, height: 1, depth: 1))
        e.endEncoding()
    }

    package func encodeSharedProduct(
        _ command: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer,
        output: MTLBuffer, exceptions: MTLBuffer
    ) throws {
        let e = try encoder(
            command, buffers: [gate, up, siluTable, output, exceptions],
            sizes: [640 * 4, 640 * 4, 65536 * 2, 640 * 2, 4])
        e.setComputePipelineState(product)
        for (index, buffer) in [gate, up, siluTable, output, exceptions].enumerated() { e.setBuffer(buffer, offset: 0, index: index) }
        e.dispatchThreads(
            MTLSize(width: 640, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, product.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
    }

    static let source = #"""
        #pragma clang fp contract(off)
        // Router gather (row one): each of ten threads copies the logit of its selected expert,
        // pairs[i].y, to selected[i]; an index past 511 writes 0 and raises the exception word.
        // A copy, so exact.
        kernel void frontier_router_gather(device const float *logits [[buffer(0)]],
            device const uint2 *pairs [[buffer(1)]], device float *selected [[buffer(2)]],
            device atomic_uint &exceptional [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            if (i >= 10) return;
            uint index = pairs[i].y;
            if (index >= 512u) { atomic_store_explicit(&exceptional, 1u, memory_order_relaxed); selected[i] = 0.0f; }
            else selected[i] = logits[index];
        }
        // Shared-expert product (row one, 640 columns, one thread each): the BF16 key of gate[i]
        // indexes the pinned SiLU table and that value times up[i] is rounded to BF16, as the host
        // expression does with fp contract off, so BITWISE. Inf/NaN, subnormal or underflowing
        // operands write 0 and raise the exception word; the caller replays the layer on the host.
        kernel void frontier_shared_product(device const float *gate [[buffer(0)]],
            device const float *up [[buffer(1)]], device const ushort *silu [[buffer(2)]],
            device ushort *out [[buffer(3)]], device atomic_uint &exceptional [[buffer(4)]],
            uint i [[thread_position_in_grid]]) {
            if (i >= 640) return;
            uint gateBits = uint(hc_decode_key(gate[i]));
            uint activationBits = uint(silu[gateBits]) << 16;
            uint upBits = as_type<uint>(up[i]);
            uint ea = (activationBits >> 23) & 255u, eb = (upBits >> 23) & 255u;
            uint eg = (as_type<uint>(gate[i]) >> 23) & 255u;
            bool nonzeroA = (activationBits & 0x7fffffffu) != 0u, nonzeroB = (upBits & 0x7fffffffu) != 0u;
            bool special = ea == 255u || eb == 255u || eg == 255u ||
                (ea == 0u && nonzeroA) || (eb == 0u && nonzeroB) ||
                (nonzeroA && nonzeroB && ea + eb <= 128u);
            if (special) { atomic_store_explicit(&exceptional, 1u, memory_order_relaxed); out[i] = 0; }
            else {
                // Same pinned table, multiplication and BF16 boundary as the
                // original shared-expert host expression and fused expert product.
                float activated = as_type<float>(activationBits);
                out[i] = ushort(as_type<uint>(round_bf16(activated * up[i])) >> 16);
            }
        }
        """#
}
