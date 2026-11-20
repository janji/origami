import Darwin
import Foundation
import Metal
import ObjectiveC

private final class ResidualInjectionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    let bytes: UInt64
    init(ledger: MemoryLedger, bytes: Int) throws {
        self.ledger = ledger
        let page = Int(getpagesize())
        self.bytes = UInt64((bytes + page - 1) / page * page)
        reservation = try ledger.reserve(bytes: self.bytes, category: "residual_injection_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum ResidualInjectionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Writes a sublayer's output back into the four residual streams: residual[s] = BF16(residual[s] +
/// BF16(output * gate[s])) for s = 0...3, gates from the HC entry's injection. BITWISE against
/// `reference` (the host loop); a flagged tile (nonfinite or possibly-flushed operands) is recomputed
/// by `reference` in full. Float-backed transfers avoid host conversion passes. Large read scopes are
/// striped into 256-row commands to bound transient RAM. `encodeSharedCombine` reuses the same kernel
/// for the MoE output: BF16(routed + BF16(shared * gate)).
public final class FusedResidualInjection {
    public let minimumRows: Int
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let ledger: MemoryLedger
    public private(set) var submittedCommands: UInt64 = 0
    public private(set) var completedCommands: UInt64 = 0
    public private(set) var repairedTiles: UInt64 = 0
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, repairedTiles: UInt64
    }
    public func snapshot() -> Snapshot {
        Snapshot(submittedCommands: submittedCommands, completedCommands: completedCommands, repairedTiles: repairedTiles)
    }

    public init(ledger: MemoryLedger, minimumRows: Int = 8) throws {
        guard (1...256).contains(minimumRows) else {
            throw EngineError("fused_residual_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Residual injection Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "residual_injection") else {
            throw EngineError("Missing residual injection kernel")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        self.device = device
        self.queue = queue
        self.ledger = ledger
        self.minimumRows = minimumRows
    }

    public func usesGPU(rows: Int) -> Bool { (minimumRows...8192).contains(rows) }

    /// Original stripe kernel at the declared actual row count. No host copy,
    /// submission, repair or change of the normal dispatch floor is implied.
    func encodeTile(
        _ command: MTLCommandBuffer, rows: Int, output: MTLBuffer, residual: MTLBuffer,
        gates: MTLBuffer, result: MTLBuffer, exceptions: MTLBuffer
    ) throws {
        let buffers = [output, residual, gates, result, exceptions]
        let sizes = [rows * 2560 * 4, rows * 10240 * 4, rows * 4 * 4, rows * 10240 * 4, 4]
        guard (1...256).contains(rows),
            zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            let e = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid residual injection tile buffers") }
        e.setComputePipelineState(pipeline)
        for (i, buffer) in [output, residual, gates, result].enumerated() { e.setBuffer(buffer, offset: 0, index: i) }
        var count = UInt32(rows * 10240)
        e.setBytes(&count, length: 4, index: 4)
        e.setBuffer(exceptions, offset: 0, index: 5)
        e.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
    }

    /// Explicit row-one frontier geometry, independently characterized before
    /// selection. This does not change usesGPU() or the ordinary dispatch floor.
    /// The caller zeroes its private flag, joins, and replays the original layer
    /// on any exception; downstream outputs must remain unpublished until then.
    func encodeDecode(
        _ command: MTLCommandBuffer, output: MTLBuffer, residual: MTLBuffer,
        gates: MTLBuffer, result: MTLBuffer, exceptions: MTLBuffer
    ) throws {
        let buffers = [output, residual, gates, result, exceptions]
        let sizes = [2560 * 4, 10240 * 4, 4 * 4, 10240 * 4, 4]
        guard zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            let encoder = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid decode residual injection buffers") }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in [output, residual, gates, result].enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var count: UInt32 = 10240
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.setBuffer(exceptions, offset: 0, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: 10240, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The same shader at its first 2560 elements has one branch and one gate:
    /// BF16(routed + BF16(shared * gate)). The routed producer must already
    /// publish BF16-valued FP32, as encodeWeightedExpertReduce10 does. This is
    /// the separately frozen shared/routed boundary, not a new reduction tree.
    func encodeSharedCombine(
        _ command: MTLCommandBuffer, routedBF16: MTLBuffer, shared: MTLBuffer,
        gate: MTLBuffer, result: MTLBuffer, exceptions: MTLBuffer
    ) throws {
        let buffers = [shared, routedBF16, gate, result, exceptions]
        let sizes = [2560 * 4, 2560 * 4, 4, 2560 * 4, 4]
        guard zip(buffers, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            let encoder = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid shared/routed combination buffers") }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in [shared, routedBF16, gate, result].enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var count: UInt32 = 2560
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.setBuffer(exceptions, offset: 0, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: 2560, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    public func evaluate(
        output: [Float], residual: [Float], gates: [Float],
        shouldContinue: () -> Bool = { true }
    ) throws -> [Float] {
        guard !residual.isEmpty, residual.count % 10240 == 0,
            (1...8192).contains(residual.count / 10240),
            output.count == residual.count / 4, gates.count == residual.count / 2560
        else {
            throw EngineError("Invalid residual injection geometry")
        }
        guard shouldContinue() else { throw IOOperationCancelled() }
        let rows = residual.count / 10240
        var result = [Float]()
        result.reserveCapacity(residual.count)
        for start in stride(from: 0, to: rows, by: 256) {
            try autoreleasepool {
                guard shouldContinue() else { throw IOOperationCancelled() }
                let count = min(256, rows - start)
                let elements = count * 10240
                let x = try copy(output, range: (start * 2560)..<((start + count) * 2560))
                let r = try copy(residual, range: (start * 10240)..<((start + count) * 10240))
                let g = try copy(gates, range: (start * 4)..<((start + count) * 4))
                let y = try allocate(bytes: elements * 4)
                let exceptional = try allocate(bytes: 4)
                exceptional.contents().storeBytes(of: UInt32(0), as: UInt32.self)
                guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                    throw EngineError("Residual injection command allocation failed", code: "insufficient_memory")
                }
                encoder.setComputePipelineState(pipeline)
                for (index, buffer) in [x, r, g, y].enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
                var n = UInt32(elements)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBuffer(exceptional, offset: 0, index: 5)
                encoder.dispatchThreads(
                    MTLSize(width: elements, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                encoder.endEncoding()
                guard shouldContinue() else { throw IOOperationCancelled() }
                command.commit()
                submittedCommands += 1
                try command.waitBounded()
                guard command.status == .completed else {
                    throw EngineError.gpu("Residual injection GPU failure", command.error)
                }
                completedCommands += 1
                guard shouldContinue() else { throw IOOperationCancelled() }
                if exceptional.contents().load(as: UInt32.self) != 0 {
                    // Metal flushes FP32 subnormals and may canonicalize NaN
                    // payloads. An exceptional tile uses the original host
                    // operations in full, with no change to their evaluation.
                    repairedTiles += 1
                    result.append(
                        contentsOf: try Self.reference(
                            output: Array(output[(start * 2560)..<((start + count) * 2560)]),
                            residual: Array(residual[(start * 10240)..<((start + count) * 10240)]),
                            gates: Array(gates[(start * 4)..<((start + count) * 4)])))
                } else {
                    result.append(contentsOf: UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float.self), count: elements))
                }
            }
        }
        return result
    }

    public static func reference(output: [Float], residual: [Float], gates: [Float]) throws -> [Float] {
        let tokens = residual.count / 10240
        guard !residual.isEmpty, residual.count % 10240 == 0, output.count == tokens * 2560, gates.count == tokens * 4 else {
            throw EngineError("Invalid hyper injection geometry")
        }
        var result = residual
        for token in 0..<tokens {
            for branch in 0..<4 {
                for column in 0..<2560 {
                    let index = token * 10240 + branch * 2560 + column
                    result[index] = LayerArithmetic.round(
                        residual[index] + LayerArithmetic.round(output[token * 2560 + column] * gates[token * 4 + branch]))
                }
            }
        }
        return result
    }

    private func copy(_ values: [Float], range: Range<Int>) throws -> MTLBuffer {
        let buffer = try allocate(bytes: range.count * 4)
        values.withUnsafeBytes {
            buffer.contents().copyMemory(from: $0.baseAddress!.advanced(by: range.lowerBound * 4), byteCount: range.count * 4)
        }
        return buffer
    }
    private func allocate(bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 256 * 10240 * 4 else { throw EngineError("Residual injection tile allocation out of bounds") }
        let owner = try ResidualInjectionAllocation(ledger: ledger, bytes: bytes)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared), UInt64(buffer.allocatedSize) <= owner.bytes else {
            throw EngineError("Residual injection Metal allocation failed or exceeded charge", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &ResidualInjectionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }

    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp contract(off)
        inline float injection_bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
                return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
            return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
        }
        // One thread per residual element i (token, stream, column): y = BF16(r[i] + BF16(x[token, column] *
        // g[token, stream])). Any special operand stores 0 and sets the single `exceptional` word.
        kernel void residual_injection(device const float *x [[buffer(0)]], device const float *r [[buffer(1)]],
            device const float *g [[buffer(2)]], device float *y [[buffer(3)]], constant uint &count [[buffer(4)]],
            device atomic_uint &exceptional [[buffer(5)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint token = i / 10240, branch = (i % 10240) / 2560, column = i % 2560;
            float a = x[token * 2560 + column], b = g[token * 4 + branch], c = r[i];
            uint ea = (as_type<uint>(a) >> 23) & 255u, eb = (as_type<uint>(b) >> 23) & 255u;
            uint ec = (as_type<uint>(c) >> 23) & 255u;
            // Conservative integer-exponent checks do not themselves flush tiny
            // values. Normal multiplication is safe only above the FP32 underflow
            // boundary. A small cancellation can underflow at up to 24 exponent
            // steps above it when arbitrary Float inputs use all mantissa bits.
            bool nonzeroA = (as_type<uint>(a) & 0x7fffffffu) != 0u;
            bool nonzeroB = (as_type<uint>(b) & 0x7fffffffu) != 0u;
            bool tinyInput = (ea == 0u && nonzeroA) || (eb == 0u && nonzeroB) ||
                             (ec == 0u && (as_type<uint>(c) & 0x7fffffffu) != 0u);
            bool special = ea == 255u || eb == 255u || ec == 255u || tinyInput ||
                           (nonzeroA && nonzeroB && ea + eb <= 128u);
            float product = injection_bf16(a * b);
            uint ep = (as_type<uint>(product) >> 23) & 255u;
            special = special || (max(ec, ep) <= 24u &&
                ((as_type<uint>(c) & 0x7fffffffu) != 0u || (as_type<uint>(product) & 0x7fffffffu) != 0u));
            if (special) { atomic_store_explicit(&exceptional, 1u, memory_order_relaxed); y[i] = 0.0f; }
            else y[i] = injection_bf16(c + product);
        }
        """#
}
