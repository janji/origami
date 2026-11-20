import Darwin
import Foundation
import Metal
import ObjectiveC

private final class HyperConnectionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    let bytes: UInt64
    init(ledger: MemoryLedger, bytes: Int, category: String) throws {
        self.ledger = ledger
        let page = Int(getpagesize())
        self.bytes = UInt64((bytes + page - 1) / page * page)
        reservation = try ledger.reserve(bytes: self.bytes, category: category)
    }
    deinit { try? ledger.release(reservation) }
}
private enum HyperConnectionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The HC epilogue that mixes the four residual streams into the layer input: per output column,
/// BF16(sum over streams of BF16(sigmoid(rawUp) * norm)) / 4, BITWISE against LayerMath's host loop.
/// Preserves the table sigmoid, BF16 products, four sequential BF16 additions from +0, and final
/// BF16 divide-by-4.
/// Inputs are BF16 values widened to Float by the existing norm/up projections.
/// A bounded bitmap flags nonfinite or potentially flushed columns; after join,
/// only those columns use the original CPU operations, preserving their payloads
/// and gradual underflow. Normalization/projections/injection stay unchanged.
/// Every actual Metal buffer owns a page-rounded reservation through retirement;
/// the submitted command is joined on success, error and cancellation.
public final class FusedHyperConnection {
    public let minimumRows: Int
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let sigmoid: MTLBuffer
    private let ledger: MemoryLedger
    private let arithmetic: LayerUnaryArithmetic
    public private(set) var submittedCommands: UInt64 = 0
    public private(set) var completedCommands: UInt64 = 0

    public init(arithmetic: LayerUnaryArithmetic, ledger: MemoryLedger, minimumRows: Int = 8) throws {
        guard (1...256).contains(minimumRows) else {
            throw EngineError("fused_hc_min_rows must be in 1...256", code: "invalid_configuration")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Fused HC Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "hc_sigmoid_sum4") else {
            throw EngineError("Missing fused HC kernel")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        self.device = device
        self.queue = queue
        self.ledger = ledger
        self.minimumRows = minimumRows
        self.arithmetic = arithmetic
        sigmoid = try Self.allocate(device: device, ledger: ledger, bytes: 65_536 * 2, category: "hc_sigmoid_table")
        let table = sigmoid.contents().assumingMemoryBound(to: UInt16.self)
        for bits in 0..<65_536 { table[bits] = BF16.encode(arithmetic.sigmoid(BF16.decode(UInt16(bits)))) }
    }

    /// The BF16 sigmoid table this epilogue reads (`hc_decode_fusion` binds the same table in its fused kernel).
    var sigmoidTable: MTLBuffer { sigmoid }

    public func usesGPU(rows: Int) -> Bool { (minimumRows...256).contains(rows) }

    public struct Result {
        public let mixed: [Float]
        /// Present only for the existing `.wup` activation capture.
        public let sigmoidUp: [Float]?
        /// Exceptional columns recomputed by the unchanged CPU arithmetic.
        public let repairedOutputs: Int
    }

    public func evaluate(
        norm: [Float], rawUp: [Float], captureUp: Bool = false,
        shouldContinue: () -> Bool = { true }
    ) throws -> Result {
        guard !norm.isEmpty, norm.count == rawUp.count, norm.count % 10_240 == 0,
            (1...256).contains(norm.count / 10_240)
        else {
            throw EngineError("Invalid fused HC geometry")
        }
        guard shouldContinue() else { throw IOOperationCancelled() }
        return try autoreleasepool {
            // All fallible buffer admission precedes command submission.
            let normalized = try copy(norm)
            let up = try copy(rawUp)
            let count = norm.count / 4
            let mixed = try Self.allocate(device: device, ledger: ledger, bytes: count * 4, category: "hc_epilogue_buffers")
            let bitmapWords = (count + 31) / 32
            let exceptions = try Self.allocate(
                device: device, ledger: ledger, bytes: bitmapWords * 4,
                category: "hc_epilogue_exceptions")
            exceptions.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bitmapWords * 4)
            // A disabled capture binds the mixed buffer to the unused argument;
            // the runtime branch forbids writes through that binding.
            let captured =
                captureUp
                ? try Self.allocate(
                    device: device, ledger: ledger, bytes: norm.count * 4,
                    category: "hc_epilogue_capture") : nil
            guard let command = queue.makeCommandBuffer() else {
                throw EngineError("Fused HC command allocation failed", code: "insufficient_memory")
            }
            command.label = "hc-table-sigmoid-sequential-bf16-sum4"
            try encode(
                command, norm: normalized, rawUp: up, mixed: mixed, captured: captured,
                exceptions: exceptions, rows: norm.count / 10240)
            guard shouldContinue() else { throw IOOperationCancelled() }
            command.commit()
            submittedCommands += 1
            try command.waitBounded()
            guard command.status == .completed else {
                throw EngineError.gpu("Fused HC GPU failure", command.error)
            }
            completedCommands += 1
            guard shouldContinue() else { throw IOOperationCancelled() }
            return try resolve(
                norm: normalized, rawUp: up, mixed: mixed, captured: captured,
                exceptions: exceptions, rows: norm.count / 10240)
        }
    }

    private func copy(_ values: [Float]) throws -> MTLBuffer {
        let buffer = try Self.allocate(device: device, ledger: ledger, bytes: values.count * 4, category: "hc_epilogue_buffers")
        values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return buffer
    }

    /// Reuses this exact epilogue in a command owned by a larger pipeline.
    /// The caller zeroes the exception bitmap and joins before resolving it.
    func encode(
        _ command: MTLCommandBuffer, norm: MTLBuffer, rawUp: MTLBuffer,
        mixed: MTLBuffer, captured: MTLBuffer?, exceptions: MTLBuffer, rows: Int
    ) throws {
        let count = rows * 2560
        guard (1...256).contains(rows), command.commandQueue.device.registryID == device.registryID,
            [norm, rawUp, mixed, exceptions].allSatisfy({ $0.device.registryID == device.registryID }),
            norm.length >= rows * 10240 * 4, rawUp.length >= rows * 10240 * 4,
            mixed.length >= count * 4, exceptions.length >= ((count + 31) / 32) * 4,
            captured == nil || (captured!.device.registryID == device.registryID && captured!.length >= rows * 10240 * 4),
            let encoder = command.makeComputeCommandEncoder()
        else {
            throw EngineError("Encoded HC epilogue buffer/geometry mismatch")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in [norm, rawUp, sigmoid, mixed, captured ?? mixed, exceptions].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        let shape: [UInt32] = [UInt32(count), captured == nil ? 0 : 1]
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 6) }
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Only call after the owning command completed. Exceptional products/sums
    /// run the original host arithmetic, including payload and underflow rules.
    func resolve(
        norm: MTLBuffer, rawUp: MTLBuffer, mixed: MTLBuffer,
        captured: MTLBuffer?, exceptions: MTLBuffer, rows: Int
    ) throws -> Result {
        let count = rows * 2560
        guard (1...256).contains(rows), norm.length >= rows * 10240 * 4, rawUp.length >= rows * 10240 * 4,
            mixed.length >= count * 4, exceptions.length >= ((count + 31) / 32) * 4,
            captured == nil || captured!.length >= rows * 10240 * 4
        else { throw EngineError("HC resolution geometry mismatch") }
        let normalized = norm.contents().assumingMemoryBound(to: Float.self)
        let up = rawUp.contents().assumingMemoryBound(to: Float.self)
        var output = Array(UnsafeBufferPointer(start: mixed.contents().assumingMemoryBound(to: Float.self), count: count))
        let bitmap = exceptions.contents().assumingMemoryBound(to: UInt32.self)
        var repaired = 0
        for word in 0..<((count + 31) / 32) {
            var flags = bitmap[word]
            while flags != 0 {
                let index = word * 32 + flags.trailingZeroBitCount
                guard index < count else { throw EngineError("Encoded HC bitmap exceeded output") }
                let token = index / 2560
                let column = index % 2560
                var sum: Float = 0
                for branch in 0..<4 {
                    let source = token * 10240 + branch * 2560 + column
                    let weight = arithmetic.sigmoid(up[source])
                    sum = LayerArithmetic.round(sum + LayerArithmetic.round(weight * normalized[source]))
                }
                output[index] = LayerArithmetic.round(sum / 4)
                repaired += 1
                flags &= flags - 1
            }
        }
        let capture = captured.map {
            Array(UnsafeBufferPointer(start: $0.contents().assumingMemoryBound(to: Float.self), count: rows * 10240))
        }
        return Result(mixed: output, sigmoidUp: capture, repairedOutputs: repaired)
    }
    private static func allocate(device: MTLDevice, ledger: MemoryLedger, bytes: Int, category: String) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 256 * 10_240 * 4 else { throw EngineError("Fused HC allocation outside bounded tile") }
        let owner = try HyperConnectionAllocation(ledger: ledger, bytes: bytes, category: category)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Fused HC Metal allocation failed", code: "insufficient_memory")
        }
        guard UInt64(buffer.allocatedSize) <= owner.bytes else {
            throw EngineError("Fused HC allocation exceeded its admitted byte charge", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &HyperConnectionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }

    // BF16 helpers match ReferenceKernels. The table supplies the exact selected
    // unary policy; compiler elementary functions cannot change the sigmoid.
    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        #pragma clang fp contract(off)
        inline float hc_bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
                return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
            return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
        }
        // One thread per mixed output (token, column of 2560): for the four streams in order, weight =
        // sigmoid table[BF16(rawUp)], sum = BF16(sum + BF16(weight * norm)) from +0, then BF16(sum / 4).
        // A product or sum that could be nonfinite or lose a subnormal sets the column's bit in
        // `exceptions` (one bit per output); `resolve` recomputes those columns on the host.
        kernel void hc_sigmoid_sum4(device const float *norm [[buffer(0)]],
            device const float *rawUp [[buffer(1)]], device const ushort *sigmoid [[buffer(2)]],
            device float *mixed [[buffer(3)]], device float *captured [[buffer(4)]],
            device atomic_uint *exceptions [[buffer(5)]],
            constant uint *shape [[buffer(6)]], uint index [[thread_position_in_grid]]) {
            if (index >= shape[0]) return;
            const uint token = index / 2560, column = index % 2560;
            float sum = 0.0f;
            bool repair = false;
            #pragma clang loop unroll(full)
            for (uint branch = 0; branch < 4; ++branch) {
                const uint source = token * 10240 + branch * 2560 + column;
                const uint key = as_type<uint>(hc_bf16(rawUp[source])) >> 16;
                const float weight = as_type<float>(uint(sigmoid[key]) << 16);
                if (shape[1]) captured[source] = weight;
                const uint a = as_type<uint>(weight) & 0x7fffffffu, b = as_type<uint>(norm[source]) & 0x7fffffffu;
                const uint ea = a >> 23, eb = b >> 23;
                // Nonfinite operands need host NaN propagation. Tiny products can
                // flush even when the rounded BF16 result remains nonzero.
                repair |= ea == 255 || eb == 255 ||
                    (a != 0 && b != 0 && (ea == 0 || eb == 0 || ea + eb <= 128));
                const float product = hc_bf16(weight * norm[source]);
                const uint c = as_type<uint>(sum) & 0x7fffffffu, d = as_type<uint>(product) & 0x7fffffffu;
                const uint ec = c >> 23, ed = d >> 23;
                // BF16 addends with exponents >=8 cannot cancel to a nonzero FP32
                // subnormal. Include any tiny operand and the lower exponent band.
                repair |= c != 0 && d != 0 && (ec == 0 || ed == 0 || max(ec, ed) <= 7);
                sum = hc_bf16(sum + product);
                repair |= ((as_type<uint>(sum) >> 23) & 255u) == 255;
            }
            const uint absolute = as_type<uint>(sum) & 0x7fffffffu;
            repair |= absolute != 0 && (absolute >> 23) <= 2;
            mixed[index] = hc_bf16(sum / 4.0f);
            if (repair) atomic_fetch_or_explicit(&exceptions[index >> 5], 1u << (index & 31), memory_order_relaxed);
        }
        """#
}
