import Darwin
import Foundation
import Metal
import ObjectiveC

private final class GDNRegionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "gdn_buffer_region")
    }
    deinit { try? ledger.release(reservation) }
}
private enum GDNRegionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The middle of a GDN layer in one joined command: recurrence, per-head RMS norm (weights norm.weight,
/// 128 wide) and, at rows >= the gate's floor, the FP32-sigmoid output gate; BITWISE against
/// `ReferenceKernels.gdn` + `rmsnorm` + LayerMath's gate (a flagged gate is recomputed on the host).
/// Used by LayerMath.gdn when the device block declines. No shader, rounding, preparation repair or
/// output-projection boundary changes.
/// A one-row result may retain distinct immutable state storage; prefill keeps
/// the original host state so a C256 output never becomes persistent storage.
public final class GDNBufferRegion {
    public struct Result {
        public let gated: [Float]
        public let recurrence, normed: [Float]?
        public let state: GDNState
    }
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, submittedRows, wideGateCommands: UInt64
        /// Explicit shared-buffer copies, including weights/mask and cancelled preparations.
        public let bufferUploadCalls, bufferUploadBytes: UInt64
        /// Float payload copies only; excludes reading the four-byte exception flag.
        public let floatReadbackBytes: UInt64
        /// GPU interval is nested within joined host time. The decode counters (`decodeSnapshot`) overlap these
        /// and must not be added to them.
        public let gpuSeconds, hostJoinSeconds: Double
    }
    /// Row-one (decode) recurrences: commands, and the full-state uploads a state without a resident owner needed.
    public struct DecodeSnapshot: Sendable {
        public let submittedCommands, completedCommands, inputStateUploads: UInt64
        public let inputStateUploadBytes, outputReadbackBytes: UInt64
    }
    private let kernels: ReferenceKernels
    private let ledger: MemoryLedger
    private let queue: MTLCommandQueue
    private let gate: FusedGDNPreparation?
    private var submitted: UInt64 = 0, completed: UInt64 = 0, rowsSubmitted: UInt64 = 0, wideGates: UInt64 = 0
    private var uploadCalls: UInt64 = 0, uploadBytes: UInt64 = 0, readbackBytes: UInt64 = 0
    private var gpuSeconds: Double = 0, hostJoinSeconds: Double = 0
    private var decodeSubmitted: UInt64 = 0, decodeCompleted: UInt64 = 0
    private var stateUploads: UInt64 = 0, stateUploadBytes: UInt64 = 0, decodeOutputReadback: UInt64 = 0

    public init(kernels: ReferenceKernels, ledger: MemoryLedger, gate: FusedGDNPreparation? = nil) throws {
        guard kernels.pinnedRMS, let queue = kernels.device.makeCommandQueue() else {
            throw EngineError("GDN buffer region requires pinned RMS and a Metal queue", code: "invalid_configuration")
        }
        self.kernels = kernels
        self.ledger = ledger
        self.gate = gate
        self.queue = queue
    }
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submitted, completedCommands: completed, submittedRows: rowsSubmitted,
            wideGateCommands: wideGates, bufferUploadCalls: uploadCalls, bufferUploadBytes: uploadBytes,
            floatReadbackBytes: readbackBytes, gpuSeconds: gpuSeconds, hostJoinSeconds: hostJoinSeconds)
    }
    public func decodeSnapshot() -> DecodeSnapshot {
        DecodeSnapshot(
            submittedCommands: decodeSubmitted, completedCommands: decodeCompleted,
            inputStateUploads: stateUploads, inputStateUploadBytes: stateUploadBytes, outputReadbackBytes: decodeOutputReadback)
    }

    public func evaluate(
        q: [Float], k: [Float], v: [Float], decay: [Float], beta: [Float],
        state: GDNState, normWeights: [Float], z: [Float], mask: [UInt32]? = nil,
        contractUpdates: Bool = true, capture: Bool = false,
        shouldContinue: () -> Bool = { true }
    ) throws -> Result {
        guard shouldContinue() else { throw IOOperationCancelled() }
        try state.validate()
        let tokens = q.count / 2048
        let stateCount = 48 * 128 * 128
        let device = kernels.device
        guard (1...256).contains(tokens), q.count == tokens * 2048, k.count == q.count,
            v.count == tokens * 6144, z.count == v.count, decay.count == tokens * 48,
            beta.count == decay.count, normWeights.count == 128, mask == nil || mask!.count == tokens
        else {
            throw EngineError("Invalid bounded GDN buffer region geometry")
        }
        if let old = state.recurrentStorage {
            guard old.buffer.device.registryID == device.registryID, old.buffer.storageMode == .shared,
                old.byteOffset >= 0, old.byteOffset % 4 == 0, old.byteOffset <= old.buffer.length,
                stateCount * 4 <= old.buffer.length - old.byteOffset
            else { throw EngineError("Invalid GDN region state owner") }
        }
        let wideGate = gate.flatMap { $0.usesGPU(rows: tokens) ? $0 : nil }
        let sizes = [
            q.count * 4, k.count * 4, v.count * 4, decay.count * 4, beta.count * 4,
            state.recurrentStorage == nil ? stateCount * 4 : 0, tokens * 4, 128 * 4,
            tokens == 1 ? 0 : (v.count + stateCount) * 4, v.count * 4,
            wideGate == nil ? 0 : z.count * 4, wideGate == nil ? 0 : v.count * 4, wideGate == nil ? 0 : 4,
        ]
        let page = Int(getpagesize())
        func charge(_ bytes: Int) -> UInt64 { UInt64((bytes + page - 1) / page * page) }
        // All transient Metal storage is admitted together; retained state has
        // its own lease. No command is submitted until both admissions succeed.
        let owner = try GDNRegionAllocation(ledger: ledger, bytes: sizes.reduce(0) { $0 + charge($1) })
        let next = tokens == 1 ? try GDNRecurrentStorage(device: device, outputCount: v.count, ledger: ledger) : nil
        return try withExtendedLifetime((owner, state, next)) {
            try autoreleasepool {
                func allocate(_ bytes: Int) throws -> MTLBuffer {
                    guard bytes > 0, bytes <= device.maxBufferLength,
                        let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
                    else {
                        throw EngineError("GDN region allocation failed", code: "insufficient_memory")
                    }
                    objc_setAssociatedObject(buffer, &GDNRegionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                    guard UInt64(buffer.allocatedSize) <= charge(bytes) else {
                        throw EngineError("GDN region exceeded admitted page bound")
                    }
                    return buffer
                }
                let buffers = try sizes.map { $0 == 0 ? nil : try allocate($0) }
                func copy<T>(_ values: [T], to buffer: MTLBuffer) {
                    values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                    uploadCalls += 1
                    uploadBytes += UInt64(values.count * MemoryLayout<T>.stride)
                }
                for (index, values) in [q, k, v, decay, beta].enumerated() { copy(values, to: buffers[index]!) }
                let initial: MTLBuffer
                let initialOffset: Int
                if let old = state.recurrentStorage {
                    initial = old.buffer
                    initialOffset = old.byteOffset
                } else {
                    initial = buffers[5]!
                    initialOffset = 0
                    copy(state.recurrent, to: initial)
                    if tokens == 1 {
                        stateUploads += 1
                        stateUploadBytes += UInt64(stateCount * 4)
                    }
                }
                copy(mask ?? [UInt32](repeating: 1, count: tokens), to: buffers[6]!)
                copy(normWeights, to: buffers[7]!)
                let recurrence = next?.buffer ?? buffers[8]!
                let normed = buffers[9]!
                if wideGate != nil {
                    copy(z, to: buffers[10]!)
                    copy([UInt32(0)], to: buffers[12]!)
                }
                guard let command = queue.makeCommandBuffer() else { throw EngineError("GDN region command allocation failed") }
                command.label = "gdn-recurrence-rms-existing-gate-one-join"
                try kernels.encodeGDNRecurrence(
                    command,
                    inputs: [
                        buffers[0]!, buffers[1]!, buffers[2]!, buffers[3]!,
                        buffers[4]!, initial, buffers[6]!,
                    ], initialOffset: initialOffset, output: recurrence,
                    tokens: tokens, contractUpdates: contractUpdates)
                try kernels.encodeGDNRMS(command, input: recurrence, weights: buffers[7]!, output: normed, tokens: tokens)
                if let wideGate {
                    try wideGate.encodeGate(
                        command, normed: normed, z: buffers[10]!, output: buffers[11]!,
                        exceptions: buffers[12]!, tokens: tokens)
                }
                guard shouldContinue() else { throw IOOperationCancelled() }
                submitted += 1
                rowsSubmitted += UInt64(tokens)
                if tokens == 1 { decodeSubmitted += 1 }
                if wideGate != nil { wideGates += 1 }
                command.commit()
                let start = DispatchTime.now().uptimeNanoseconds
                try command.waitBounded()
                hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if command.gpuStartTime.isFinite, command.gpuEndTime.isFinite,
                    command.gpuStartTime > 0, command.gpuEndTime >= command.gpuStartTime
                {
                    gpuSeconds += command.gpuEndTime - command.gpuStartTime
                }
                guard command.status == .completed else { throw EngineError.gpu("GDN region GPU failure", command.error) }
                completed += 1
                if tokens == 1 { decodeCompleted += 1 }
                guard shouldContinue() else { throw IOOperationCancelled() }
                func read(_ buffer: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
                    readbackBytes += UInt64(count * 4)
                    return Array(
                        UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self).advanced(by: offset), count: count)
                    )
                }
                let repair = wideGate != nil && buffers[12]!.contents().load(as: UInt32.self) != 0
                let normValues = capture || wideGate == nil || repair ? read(normed, count: v.count) : nil
                let gated: [Float]
                if let wideGate {
                    gated = repair ? wideGate.repairEmbeddedGate(normed: normValues!, z: z) : read(buffers[11]!, count: v.count)
                } else {
                    gated = zip(normValues!, z).map { LayerArithmetic.round($0 * kernels.arithmetic.sigmoidFP32($1)) }
                }
                let recurrenceValues = capture ? read(recurrence, count: v.count) : nil
                if capture && tokens == 1 { decodeOutputReadback += UInt64(v.count * 4) }
                var updated = state
                if let next {
                    updated.publishRecurrent(next)
                } else {
                    updated.recurrent = read(recurrence, count: stateCount, offset: v.count)
                }
                guard shouldContinue() else { throw IOOperationCancelled() }
                return Result(gated: gated, recurrence: recurrenceValues, normed: capture ? normValues : nil, state: updated)
            }
        }
    }
}
