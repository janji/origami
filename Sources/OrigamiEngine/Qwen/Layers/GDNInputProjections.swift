import Foundation
import Metal
import ObjectiveC

private final class GDNProjectionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "gdn_input_projections")
    }
    deinit { try? ledger.release(reservation) }
}
private enum GDNProjectionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The four GDN input projections in one joined command: in_proj_qkv [10240] and in_proj_z [6144]
/// (affine4, BF16 input) and in_proj_b / in_proj_a [48 each] (dense, FP32 input, BF16 output), BITWISE
/// against four `linear` calls: the same actual-row plans, weight geometry, compilation policy and
/// BF16 boundaries. Used by LayerMath.gdn when the device block declines.
/// The caller owns every weight-cache pin until this synchronous call returns.
/// No recurrent state or preparation/exception boundary crosses this command.
public final class GDNInputProjections {
    public struct Projection {
        public let words, scales, biases: MTLBuffer
        public init(words: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer) {
            self.words = words
            self.scales = scales
            self.biases = biases
        }
    }
    public struct Result {
        public let qkv, z, b, a: [Float]
    }
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, submittedRows: UInt64
        /// Logical host copies, including a subsequently cancelled preparation.
        public let inputUploadCalls, inputUploadBytes: UInt64
        /// GPU time is nested within joined host time, not an additive bucket.
        public let gpuSeconds, hostJoinSeconds: Double
    }

    private let kernels: ReferenceKernels
    private let dense: PinnedDenseLinear
    private let ledger: MemoryLedger
    private let queue: MTLCommandQueue
    private let simdMatrix: Bool
    private var submittedCommands: UInt64 = 0, completedCommands: UInt64 = 0, submittedRows: UInt64 = 0
    private var inputUploadCalls: UInt64 = 0, inputUploadBytes: UInt64 = 0
    private var gpuSeconds: Double = 0, hostJoinSeconds: Double = 0

    public init(kernels: ReferenceKernels, dense: PinnedDenseLinear, ledger: MemoryLedger, simdMatrix: Bool) throws {
        guard let queue = kernels.device.makeCommandQueue() else {
            throw EngineError("GDN input composition requires source dense plans and a Metal queue", code: "invalid_configuration")
        }
        self.kernels = kernels
        self.dense = dense
        self.ledger = ledger
        self.queue = queue
        self.simdMatrix = simdMatrix
    }

    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submittedCommands, completedCommands: completedCommands, submittedRows: submittedRows,
            inputUploadCalls: inputUploadCalls, inputUploadBytes: inputUploadBytes,
            gpuSeconds: gpuSeconds, hostJoinSeconds: hostJoinSeconds)
    }

    public func evaluate(
        _ x: [Float], qkv: Projection, z: Projection, b: MTLBuffer, a: MTLBuffer,
        shouldContinue: () -> Bool = { true }
    ) throws -> Result {
        guard shouldContinue() else { throw IOOperationCancelled() }
        let rows = x.count / 2560
        let device = kernels.device
        guard (1...256).contains(rows), x.count == rows * 2560 else {
            throw EngineError("GDN input composition requires 1...256 fixed-width rows")
        }
        func valid(_ buffer: MTLBuffer, bytes: Int) -> Bool {
            buffer.device.registryID == device.registryID && buffer.storageMode == .shared && buffer.length >= bytes
        }
        for (projection, outputs) in [(qkv, 10240), (z, 6144)] {
            guard valid(projection.words, bytes: outputs * 2560 / 8 * 4),
                valid(projection.scales, bytes: outputs * 2560 / 64 * 2),
                valid(projection.biases, bytes: outputs * 2560 / 64 * 2)
            else {
                throw EngineError("GDN resident affine4 buffers differ from the fixed checkpoint")
            }
        }
        guard valid(b, bytes: 48 * 2560 * 4), valid(a, bytes: 48 * 2560 * 4) else {
            throw EngineError("GDN resident dense buffers differ from the fixed checkpoint")
        }
        let qkvPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let zPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 6144, simdMatrix: simdMatrix)
        let densePlan = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 48)
        let denseBytes = densePlan.kind == .splitK ? rows * 48 * densePlan.splitKPartitions * 4 : 0
        let sizes = [
            x.count * 2, x.count * 4, rows * 10240 * 4, rows * 6144 * 4,
            rows * 48 * 4, rows * 48 * 4, qkvPlan.partialBytes, zPlan.partialBytes, denseBytes, denseBytes,
        ]
        func charged(_ bytes: Int) -> UInt64 { UInt64((bytes + 16_383) / 16_384 * 16_384) }
        // Admit the entire Metal region before allocating any part. Host input
        // and returned arrays remain covered by the executor's phase workspace.
        let owner = try GDNProjectionAllocation(ledger: ledger, bytes: sizes.reduce(0) { $0 + charged($1) })
        return try withExtendedLifetime(owner) {
            try autoreleasepool {
                func allocate(_ bytes: Int) throws -> MTLBuffer {
                    guard bytes > 0, let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                        throw EngineError("GDN input composition allocation failed", code: "insufficient_memory")
                    }
                    objc_setAssociatedObject(buffer, &GDNProjectionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                    guard UInt64(buffer.allocatedSize) <= charged(bytes) else {
                        throw EngineError("GDN projection storage exceeded its admitted page bound")
                    }
                    return buffer
                }
                let buffers = try sizes.map { $0 == 0 ? nil : try allocate($0) }
                let bf16 = buffers[0]!
                let fp32 = buffers[1]!
                let bits = bf16.contents().assumingMemoryBound(to: UInt16.self)
                for index in x.indices { bits[index] = BF16.encode(x[index]) }
                x.withUnsafeBytes { fp32.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                inputUploadCalls += 2
                inputUploadBytes += UInt64(x.count * 6)
                guard let command = queue.makeCommandBuffer() else {
                    throw EngineError("GDN input command allocation failed", code: "insufficient_memory")
                }
                command.label = "gdn-four-input-projections-shared-views-one-join"
                try kernels.encodePinnedProjection(
                    command, input: bf16, words: qkv.words, scales: qkv.scales,
                    biases: qkv.biases, output: buffers[2]!, partials: buffers[6], plan: qkvPlan, simdMatrix: simdMatrix)
                try kernels.encodePinnedProjection(
                    command, input: bf16, words: z.words, scales: z.scales,
                    biases: z.biases, output: buffers[3]!, partials: buffers[7], plan: zPlan, simdMatrix: simdMatrix)
                try dense.encodeRows(
                    command, input: fp32, weights: b, output: buffers[4]!, partials: buffers[8],
                    rows: rows, width: 2560, outputs: 48, outputBF16: true)
                try dense.encodeRows(
                    command, input: fp32, weights: a, output: buffers[5]!, partials: buffers[9],
                    rows: rows, width: 2560, outputs: 48, outputBF16: true)
                guard shouldContinue() else { throw IOOperationCancelled() }
                submittedCommands += 1
                submittedRows += UInt64(rows)
                command.commit()
                let start = DispatchTime.now().uptimeNanoseconds
                try command.waitBounded()
                hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if command.gpuStartTime.isFinite, command.gpuEndTime.isFinite,
                    command.gpuStartTime > 0, command.gpuEndTime >= command.gpuStartTime
                {
                    gpuSeconds += command.gpuEndTime - command.gpuStartTime
                }
                guard command.status == .completed else {
                    throw EngineError.gpu("GDN input composition failed", command.error)
                }
                completedCommands += 1
                // Every buffer and caller-held weight pin survives the join,
                // including cancellation observed while the GPU was executing.
                guard shouldContinue() else { throw IOOperationCancelled() }
                func values(_ index: Int, _ count: Int) -> [Float] {
                    Array(UnsafeBufferPointer(start: buffers[index]!.contents().assumingMemoryBound(to: Float.self), count: count))
                }
                return Result(
                    qkv: values(2, rows * 10240), z: values(3, rows * 6144),
                    b: values(4, rows * 48), a: values(5, rows * 48))
            }
        }
    }
}
