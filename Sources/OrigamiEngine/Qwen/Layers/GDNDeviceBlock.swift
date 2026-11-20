import Darwin
import Foundation
import Metal
import ObjectiveC

private final class GDNBlockAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "gdn_device_block")
    }
    deinit { try? ledger.release(reservation) }
}
private enum GDNBlockAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// A complete GDN layer in one command: in_proj_qkv and in_proj_z (affine4 QMV/QMM), in_proj_b and
/// in_proj_a (dense FP32 weights, BF16 output), FusedGDNPreparation (causal conv + SiLU + L2 norm +
/// beta/decay), the recurrence (`gdn_reference` or `gdn_exact_rows8`), the per-head RMS norm, the
/// FP32-sigmoid gate, a BF16 narrowing and out_proj. BITWISE against LayerMath.gdn's host-joined path
/// (`gdn_device_block_composition_ownership`). Three entry points: `evaluate` (synchronous, private
/// buffers), `encodeRow` (decode row into a caller's command) and `encodeTile` (2...256 prefill rows).
/// Every intermediate and next-state buffer is private until the command joins and both numerical
/// exception flags have been checked. Exceptional arithmetic cannot affect addresses; a flagged region
/// returns nil for host-path replay, without publishing state or captures. GPU failures and
/// cancellation throw. Caller-held cache pins survive this synchronous call.
public final class GDNDeviceBlock {
    public typealias Projection = GDNInputProjections.Projection
    public struct Result {
        public let output: [Float]
        public let state: GDNState
        public let captures: [String: [Float]]
    }
    /// Fixed row-one workspace owned by a wider decode region. Constants and
    /// convolution history are filled while idle; next recurrent storage is a
    /// separate immutable-state owner and never part of reusable scratch.
    package struct RowBuffers {
        package let mixed, z, b, a, history, convolution, bias, logs, prepared, beta, decay: MTLBuffer
        package let prepareFlags, mask, normWeights, normed, gated, gateFlags, narrowed, output: MTLBuffer
        package let convolved: MTLBuffer?
        package init(
            mixed: MTLBuffer, z: MTLBuffer, b: MTLBuffer, a: MTLBuffer, history: MTLBuffer, convolution: MTLBuffer, bias: MTLBuffer,
            logs: MTLBuffer, prepared: MTLBuffer, beta: MTLBuffer, decay: MTLBuffer, prepareFlags: MTLBuffer, mask: MTLBuffer,
            normWeights: MTLBuffer, normed: MTLBuffer, gated: MTLBuffer, gateFlags: MTLBuffer, narrowed: MTLBuffer, output: MTLBuffer,
            convolved: MTLBuffer?
        ) {
            self.mixed = mixed
            self.z = z
            self.b = b
            self.a = a
            self.history = history
            self.convolution = convolution
            self.bias = bias
            self.logs = logs
            self.prepared = prepared
            self.beta = beta
            self.decay = decay
            self.prepareFlags = prepareFlags
            self.mask = mask
            self.normWeights = normWeights
            self.normed = normed
            self.gated = gated
            self.gateFlags = gateFlags
            self.narrowed = narrowed
            self.output = output
            self.convolved = convolved
        }
        var all: [MTLBuffer] {
            [
                mixed, z, b, a, history, convolution, bias, logs, prepared, beta, decay,
                prepareFlags, mask, normWeights, normed, gated, gateFlags, narrowed, output,
            ]
                + (convolved.map { [$0] } ?? [])
        }
    }
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, submittedRows, exceptionReplays: UInt64
        public let inputStateUploads, inputStateUploadBytes, bufferUploadCalls, bufferUploadBytes: UInt64
        public let floatReadbackBytes, maximumTransientBytes: UInt64
        public let gpuSeconds, hostJoinSeconds: Double
    }
    private let kernels: ReferenceKernels
    private let dense: PinnedDenseLinear
    private let preparation: FusedGDNPreparation
    private let ledger: MemoryLedger
    private let simdMatrix: Bool
    private let queue: MTLCommandQueue
    private let narrow: MTLComputePipelineState
    private var submitted: UInt64 = 0, completed: UInt64 = 0, rowsSubmitted: UInt64 = 0, replays: UInt64 = 0
    private var stateUploads: UInt64 = 0, stateBytes: UInt64 = 0, uploads: UInt64 = 0, uploadBytes: UInt64 = 0
    private var readBytes: UInt64 = 0, maximumTransient: UInt64 = 0
    private var gpuSeconds: Double = 0, hostJoinSeconds: Double = 0

    public init(
        kernels: ReferenceKernels, dense: PinnedDenseLinear, preparation: FusedGDNPreparation,
        ledger: MemoryLedger, simdMatrix: Bool
    ) throws {
        guard kernels.pinnedRMS, let queue = kernels.device.makeCommandQueue() else {
            throw EngineError("GDN device block requires pinned RMS, source dense schedules and a queue", code: "invalid_configuration")
        }
        self.kernels = kernels
        self.dense = dense
        self.preparation = preparation
        self.ledger = ledger
        self.simdMatrix = simdMatrix
        self.queue = queue
        // FusedHCDecode's integer-only BF16 narrowing (the bits of BF16.encode), compiled precise.
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let source = "#include <metal_stdlib>\nusing namespace metal;\n" + FusedHCDecode.handoffSource
        let library = try kernels.device.makeLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: "hc_decode_narrow") else { throw EngineError("GDN BF16 handoff unavailable") }
        narrow = try kernels.device.makeComputePipelineState(function: function)
    }
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submitted, completedCommands: completed, submittedRows: rowsSubmitted,
            exceptionReplays: replays, inputStateUploads: stateUploads, inputStateUploadBytes: stateBytes,
            bufferUploadCalls: uploads, bufferUploadBytes: uploadBytes, floatReadbackBytes: readBytes,
            maximumTransientBytes: maximumTransient, gpuSeconds: gpuSeconds, hostJoinSeconds: hostJoinSeconds)
    }
    private enum Slot: Int, CaseIterable {
        case bf16, fp32, mixed, z, b, a, qkvPartials, zPartials, bPartials, aPartials
        case history, convolution, bias, logs, prepared, beta, decay, convolved, prepareFlags
        case initial, mask, normWeights, recurrence, normed, gated, gateFlags, narrowed, output, outPartials
    }

    /// The decode-row graph with caller-owned buffers (whole-execution decode). qkv and z share one
    /// paired QMV when DecodeProjectionKernel supports it (bit-identical per output). It neither
    /// commits nor publishes state; `next` receives the recurrent output and state. All addresses
    /// remain fixed even on exceptional values; the caller inspects both flags after join and replays
    /// from the old state before exposing any result.
    package func encodeRow(
        _ command: MTLCommandBuffer, inputFP32: MTLBuffer, inputBF16: MTLBuffer,
        qkv: Projection, z: Projection, b: MTLBuffer, a: MTLBuffer, out: Projection,
        initial: MTLBuffer, initialOffset: Int, next: GDNRecurrentStorage,
        buffers s: RowBuffers
    ) throws {
        let sizes =
            [
                10240 * 4, 6144 * 4, 48 * 4, 48 * 4, 3 * 10240 * 2, 40960 * 4, 48 * 4, 48 * 4,
                10240 * 4, 48 * 4, 48 * 4, 4, 4, 128 * 4, 6144 * 4, 6144 * 4, 4, 6144 * 2, 2560 * 4,
            ]
            + (s.convolved == nil ? [] : [10240 * 4])
        let device = kernels.device
        let all = s.all
        let reads = [
            inputFP32, inputBF16, initial, qkv.words, qkv.scales, qkv.biases,
            z.words, z.scales, z.biases, b, a, out.words, out.scales, out.biases,
        ]
        guard all.count == sizes.count,
            zip(all, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(all.map { ObjectIdentifier($0) }).count == all.count,
            !all.contains(where: { scratch in reads.contains(where: { $0 === scratch }) || scratch === next.buffer }),
            reads.allSatisfy({ $0.device.registryID == device.registryID && $0 !== next.buffer }),
            inputFP32.length >= 2560 * 4, inputBF16.length >= 2560 * 2,
            initialOffset >= 0, initialOffset % 4 == 0, initialOffset <= initial.length,
            48 * 128 * 128 * 4 <= initial.length - initialOffset,
            next.buffer.device.registryID == device.registryID, next.buffer.length >= (6144 + 48 * 128 * 128) * 4,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid caller-owned GDN row-one region")
        }
        let qPlan = try kernels.pinnedProjectionPlan(rows: 1, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let zPlan = try kernels.pinnedProjectionPlan(rows: 1, width: 2560, outputs: 6144, simdMatrix: simdMatrix)
        let outPlan = try kernels.pinnedProjectionPlan(rows: 1, width: 6144, outputs: 2560, simdMatrix: simdMatrix)
        let densePlan = try PinnedDenseLinear.plan(rows: 1, width: 2560, outputs: 48)
        guard qPlan.partialBytes == 0, zPlan.partialBytes == 0, outPlan.partialBytes == 0, densePlan.kind != .splitK else {
            throw EngineError("GDN row-one schedules unexpectedly require partial storage")
        }
        if try !kernels.encodePairedDecodeProjection(
            command, input: inputBF16, width: 2560,
            first: .init(words: qkv.words, scales: qkv.scales, biases: qkv.biases, rows: 10240), firstOutput: s.mixed,
            second: .init(words: z.words, scales: z.scales, biases: z.biases, rows: 6144), secondOutput: s.z)
        {
            try kernels.encodePinnedProjection(
                command, input: inputBF16, words: qkv.words, scales: qkv.scales,
                biases: qkv.biases, output: s.mixed, partials: nil, plan: qPlan, simdMatrix: simdMatrix)
            try kernels.encodePinnedProjection(
                command, input: inputBF16, words: z.words, scales: z.scales,
                biases: z.biases, output: s.z, partials: nil, plan: zPlan, simdMatrix: simdMatrix)
        }
        try dense.encodeRows(
            command, input: inputFP32, weights: b, output: s.b, partials: nil,
            rows: 1, width: 2560, outputs: 48, outputBF16: true)
        try dense.encodeRows(
            command, input: inputFP32, weights: a, output: s.a, partials: nil,
            rows: 1, width: 2560, outputs: 48, outputBF16: true)
        try preparation.encodePreparation(
            command, input: s.mixed, history: s.history, weights: s.convolution,
            aInput: s.a, bInput: s.b, bias: s.bias, logs: s.logs, qkv: s.prepared,
            beta: s.beta, decay: s.decay, convolved: s.convolved, exceptions: s.prepareFlags, rows: 1)
        try kernels.encodeGDNRecurrence(
            command, inputs: [s.prepared, s.prepared, s.prepared, s.decay, s.beta, initial, s.mask],
            initialOffset: initialOffset, output: next.buffer, tokens: 1, contractUpdates: true,
            qkvOffsets: [0, 2048 * 4, 4096 * 4])
        try kernels.encodeGDNRMS(command, input: next.buffer, weights: s.normWeights, output: s.normed, tokens: 1)
        try preparation.encodeGate(
            command, normed: s.normed, z: s.z, output: s.gated, exceptions: s.gateFlags,
            tokens: 1, completeBlock: true)
        guard let encoder = command.makeComputeCommandEncoder() else { throw EngineError("GDN row-one narrow encoder unavailable") }
        encoder.setComputePipelineState(narrow)
        encoder.setBuffer(s.gated, offset: 0, index: 0)
        encoder.setBuffer(s.narrowed, offset: 0, index: 1)
        var count: UInt32 = 6144
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: 6144, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, narrow.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        try kernels.encodePinnedProjection(
            command, input: s.narrowed, words: out.words, scales: out.scales,
            biases: out.biases, output: s.output, partials: nil, plan: outPlan, simdMatrix: simdMatrix)
    }

    /// Caller-owned split-K partial storage for a tile; nil where the row count's plan needs none
    /// (sizes from `tilePartialBytes`).
    package struct TilePartials {
        let qkv, z, b, a, output: MTLBuffer?
        package init(qkv: MTLBuffer?, z: MTLBuffer?, b: MTLBuffer?, a: MTLBuffer?, output: MTLBuffer?) {
            self.qkv = qkv
            self.z = z
            self.b = b
            self.a = a
            self.output = output
        }
    }
    package func tilePartialBytes(rows: Int) throws -> (qkv: Int, z: Int, dense: Int, output: Int) {
        guard (2...256).contains(rows) else { throw EngineError("Invalid GDN tile rows") }
        let q = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let z = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 6144, simdMatrix: simdMatrix)
        let out = try kernels.pinnedProjectionPlan(rows: rows, width: 6144, outputs: 2560, simdMatrix: simdMatrix)
        let d = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 48)
        return (q.partialBytes, z.partialBytes, d.kind == .splitK ? rows * 48 * d.splitKPartitions * 4 : 0, out.partialBytes)
    }

    /// A prefill tile (2...256 rows, the 256-row compute tile or a scope's tail) with the plans
    /// `evaluate` selects for the same rows; `recurrence` receives [rows x 6144 output | FP32 state].
    /// The owner retains old state separately, checks both flags after joining, and publishes or
    /// replays through the host prefill path. Two rows: the MTP verify step.
    package func encodeTile(
        _ command: MTLCommandBuffer, rows: Int, inputFP32: MTLBuffer, inputBF16: MTLBuffer,
        qkv: Projection, z: Projection, b: MTLBuffer, a: MTLBuffer, out: Projection,
        initial: MTLBuffer, initialOffset: Int, recurrence: MTLBuffer,
        buffers s: RowBuffers, partials p: TilePartials
    ) throws {
        let partial = try tilePartialBytes(rows: rows)
        let device = kernels.device
        let sizes =
            [
                rows * 10240 * 4, rows * 6144 * 4, rows * 48 * 4, rows * 48 * 4, 3 * 10240 * 2,
                40960 * 4, 48 * 4, 48 * 4, rows * 10240 * 4, rows * 48 * 4, rows * 48 * 4,
                4, rows * 4, 128 * 4, rows * 6144 * 4, rows * 6144 * 4, 4, rows * 6144 * 2, rows * 2560 * 4,
            ]
            + (s.convolved == nil ? [] : [rows * 10240 * 4])
        var spans = Array(zip(s.all, sizes))
        spans.append((recurrence, (rows * 6144 + 48 * 128 * 128) * 4))
        for (buffer, bytes) in [
            (p.qkv, partial.qkv), (p.z, partial.z), (p.b, partial.dense),
            (p.a, partial.dense), (p.output, partial.output),
        ] where bytes > 0 {
            guard let buffer else { throw EngineError("Missing GDN tile partial storage") }
            spans.append((buffer, bytes))
        }
        let reads = [
            inputFP32, inputBF16, initial, qkv.words, qkv.scales, qkv.biases, z.words, z.scales, z.biases,
            b, a, out.words, out.scales, out.biases,
        ]
        guard command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            spans.allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(spans.map { ObjectIdentifier($0.0) }).count == spans.count,
            reads.allSatisfy({ $0.device.registryID == device.registryID }),
            !spans.contains(where: { s in reads.contains(where: { $0 === s.0 }) }),
            inputFP32.length >= rows * 2560 * 4, inputBF16.length >= rows * 2560 * 2,
            initialOffset >= 0, initialOffset % 4 == 0, initialOffset <= initial.length,
            48 * 128 * 128 * 4 <= initial.length - initialOffset
        else { throw EngineError("Invalid GDN tile buffer spans") }
        let qPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let zPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 6144, simdMatrix: simdMatrix)
        let outPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 6144, outputs: 2560, simdMatrix: simdMatrix)
        do {
            try kernels.encodePinnedProjection(
                command, input: inputBF16, words: qkv.words, scales: qkv.scales,
                biases: qkv.biases, output: s.mixed, partials: p.qkv, plan: qPlan, simdMatrix: simdMatrix)
            try kernels.encodePinnedProjection(
                command, input: inputBF16, words: z.words, scales: z.scales,
                biases: z.biases, output: s.z, partials: p.z, plan: zPlan, simdMatrix: simdMatrix)
        }
        try dense.encodeRows(
            command, input: inputFP32, weights: b, output: s.b, partials: p.b, rows: rows, width: 2560, outputs: 48, outputBF16: true)
        try dense.encodeRows(
            command, input: inputFP32, weights: a, output: s.a, partials: p.a, rows: rows, width: 2560, outputs: 48, outputBF16: true)
        try preparation.encodePreparation(
            command, input: s.mixed, history: s.history, weights: s.convolution,
            aInput: s.a, bInput: s.b, bias: s.bias, logs: s.logs, qkv: s.prepared, beta: s.beta, decay: s.decay,
            convolved: s.convolved, exceptions: s.prepareFlags, rows: rows)
        do {
            try kernels.encodeGDNRecurrence(
                command, inputs: [s.prepared, s.prepared, s.prepared, s.decay, s.beta, initial, s.mask],
                initialOffset: initialOffset, output: recurrence, tokens: rows, contractUpdates: true,
                qkvOffsets: [0, rows * 2048 * 4, rows * 4096 * 4])
        }
        try kernels.encodeGDNRMS(command, input: recurrence, weights: s.normWeights, output: s.normed, tokens: rows)
        try preparation.encodeGate(
            command, normed: s.normed, z: s.z, output: s.gated, exceptions: s.gateFlags, tokens: rows, completeBlock: true)
        guard let e = command.makeComputeCommandEncoder() else { throw EngineError("GDN tile narrow encoder unavailable") }
        e.setComputePipelineState(narrow)
        e.setBuffer(s.gated, offset: 0, index: 0)
        e.setBuffer(s.narrowed, offset: 0, index: 1)
        var count = UInt32(rows * 6144)
        e.setBytes(&count, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, narrow.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding()
        try kernels.encodePinnedProjection(
            command, input: s.narrowed, words: out.words, scales: out.scales,
            biases: out.biases, output: s.output, partials: p.output, plan: outPlan, simdMatrix: simdMatrix)
    }

    public func evaluate(
        _ x: [Float], qkv: Projection, z: Projection, b: MTLBuffer, a: MTLBuffer,
        out: Projection, convolution: [Float], dtBias: [Float], aLog: [Float],
        normWeights: [Float], state: GDNState, mask: [UInt32]? = nil,
        capture: Bool = false, shouldContinue: () -> Bool = { true }
    ) throws -> Result? {
        guard shouldContinue() else { throw IOOperationCancelled() }
        try state.validate()
        let rows = x.count / 2560
        let stateCount = 48 * 128 * 128
        let device = kernels.device
        guard (1...256).contains(rows), x.count == rows * 2560, convolution.count == 40960,
            dtBias.count == 48, aLog.count == 48, normWeights.count == 128,
            mask == nil || mask!.count == rows, rows <= ModelContract.maximumContext - state.offset
        else {
            throw EngineError("Invalid bounded GDN device block geometry")
        }
        if let old = state.recurrentStorage {
            guard old.buffer.device.registryID == device.registryID, old.buffer.storageMode == .shared,
                old.byteOffset >= 0, old.byteOffset % 4 == 0, old.byteOffset <= old.buffer.length,
                stateCount * 4 <= old.buffer.length - old.byteOffset
            else { throw EngineError("Invalid GDN block state owner") }
        }
        let qPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let zPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 6144, simdMatrix: simdMatrix)
        let outPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 6144, outputs: 2560, simdMatrix: simdMatrix)
        let dPlan = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 48)
        let dBytes = dPlan.kind == .splitK ? rows * 48 * dPlan.splitKPartitions * 4 : 0
        let sizes = [
            x.count * 2, x.count * 4, rows * 10240 * 4, rows * 6144 * 4, rows * 48 * 4, rows * 48 * 4,
            qPlan.partialBytes, zPlan.partialBytes, dBytes, dBytes, 3 * 10240 * 2, 40960 * 4, 48 * 4, 48 * 4,
            rows * 10240 * 4, rows * 48 * 4, rows * 48 * 4, capture ? rows * 10240 * 4 : 0, 4,
            state.recurrentStorage == nil ? stateCount * 4 : 0, rows * 4, 128 * 4,
            rows == 1 ? 0 : (rows * 6144 + stateCount) * 4, rows * 6144 * 4, rows * 6144 * 4, 4,
            rows * 6144 * 2, rows * 2560 * 4, outPlan.partialBytes,
        ]
        guard sizes.count == Slot.allCases.count else { throw EngineError("GDN block admission map differs") }
        let page = Int(getpagesize())
        func charge(_ bytes: Int) -> UInt64 { UInt64((bytes + page - 1) / page * page) }
        let transient = sizes.reduce(0) { $0 + charge($1) }
        let owner = try GDNBlockAllocation(ledger: ledger, bytes: transient)
        let next = rows == 1 ? try GDNRecurrentStorage(device: device, outputCount: 6144, ledger: ledger) : nil
        maximumTransient = max(maximumTransient, transient)
        return try withExtendedLifetime((owner, next, state)) {
            try autoreleasepool {
                let buffers: [MTLBuffer?] = try sizes.map { bytes in
                    if bytes == 0 { return nil }
                    guard bytes <= device.maxBufferLength, let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                        throw EngineError("GDN block allocation failed", code: "insufficient_memory")
                    }
                    objc_setAssociatedObject(buffer, &GDNBlockAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                    guard UInt64(buffer.allocatedSize) <= charge(bytes) else { throw EngineError("GDN block exceeded admitted page bound") }
                    return buffer
                }
                func buffer(_ slot: Slot) -> MTLBuffer { buffers[slot.rawValue]! }
                func optional(_ slot: Slot) -> MTLBuffer? { buffers[slot.rawValue] }
                func copy<T>(_ values: [T], _ slot: Slot) {
                    values.withUnsafeBytes { buffer(slot).contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                    uploads += 1
                    uploadBytes += UInt64(values.count * MemoryLayout<T>.stride)
                }
                let bits = buffer(.bf16).contents().assumingMemoryBound(to: UInt16.self)
                for index in x.indices { bits[index] = BF16.encode(x[index]) }
                uploads += 1
                uploadBytes += UInt64(x.count * 2)
                copy(x, .fp32)
                copy(state.convolutionTail, .history)
                copy(convolution, .convolution)
                copy(dtBias, .bias)
                copy(aLog, .logs)
                copy(mask ?? [UInt32](repeating: 1, count: rows), .mask)
                copy(normWeights, .normWeights)
                copy([UInt32(0)], .prepareFlags)
                copy([UInt32(0)], .gateFlags)
                let initial: MTLBuffer
                let initialOffset: Int
                if let old = state.recurrentStorage {
                    initial = old.buffer
                    initialOffset = old.byteOffset
                } else {
                    copy(state.recurrent, .initial)
                    initial = buffer(.initial)
                    initialOffset = 0
                    stateUploads += 1
                    stateBytes += UInt64(stateCount * 4)
                }
                let recurrence = next?.buffer ?? buffer(.recurrence)
                guard let command = queue.makeCommandBuffer() else { throw EngineError("GDN block command allocation failed") }
                command.label = "gdn-complete-device-block-one-join"
                try kernels.encodePinnedProjection(
                    command, input: buffer(.bf16), words: qkv.words, scales: qkv.scales,
                    biases: qkv.biases, output: buffer(.mixed), partials: optional(.qkvPartials), plan: qPlan, simdMatrix: simdMatrix)
                try kernels.encodePinnedProjection(
                    command, input: buffer(.bf16), words: z.words, scales: z.scales,
                    biases: z.biases, output: buffer(.z), partials: optional(.zPartials), plan: zPlan, simdMatrix: simdMatrix)
                try dense.encodeRows(
                    command, input: buffer(.fp32), weights: b, output: buffer(.b), partials: optional(.bPartials),
                    rows: rows, width: 2560, outputs: 48, outputBF16: true)
                try dense.encodeRows(
                    command, input: buffer(.fp32), weights: a, output: buffer(.a), partials: optional(.aPartials),
                    rows: rows, width: 2560, outputs: 48, outputBF16: true)
                try preparation.encodePreparation(
                    command, input: buffer(.mixed), history: buffer(.history), weights: buffer(.convolution),
                    aInput: buffer(.a), bInput: buffer(.b), bias: buffer(.bias), logs: buffer(.logs), qkv: buffer(.prepared),
                    beta: buffer(.beta), decay: buffer(.decay), convolved: optional(.convolved), exceptions: buffer(.prepareFlags),
                    rows: rows)
                try kernels.encodeGDNRecurrence(
                    command,
                    inputs: [
                        buffer(.prepared), buffer(.prepared), buffer(.prepared),
                        buffer(.decay), buffer(.beta), initial, buffer(.mask),
                    ], initialOffset: initialOffset, output: recurrence,
                    tokens: rows, contractUpdates: true, qkvOffsets: [0, rows * 2048 * 4, rows * 4096 * 4])
                try kernels.encodeGDNRMS(command, input: recurrence, weights: buffer(.normWeights), output: buffer(.normed), tokens: rows)
                try preparation.encodeGate(
                    command, normed: buffer(.normed), z: buffer(.z), output: buffer(.gated),
                    exceptions: buffer(.gateFlags), tokens: rows, completeBlock: true)
                guard let narrowEncoder = command.makeComputeCommandEncoder() else {
                    throw EngineError("GDN narrow encoder allocation failed")
                }
                narrowEncoder.setComputePipelineState(narrow)
                narrowEncoder.setBuffer(buffer(.gated), offset: 0, index: 0)
                narrowEncoder.setBuffer(buffer(.narrowed), offset: 0, index: 1)
                var count = UInt32(rows * 6144)
                narrowEncoder.setBytes(&count, length: 4, index: 2)
                narrowEncoder.dispatchThreads(
                    MTLSize(width: Int(count), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, narrow.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                narrowEncoder.endEncoding()
                try kernels.encodePinnedProjection(
                    command, input: buffer(.narrowed), words: out.words, scales: out.scales,
                    biases: out.biases, output: buffer(.output), partials: optional(.outPartials), plan: outPlan, simdMatrix: simdMatrix)
                guard shouldContinue() else { throw IOOperationCancelled() }
                submitted += 1
                rowsSubmitted += UInt64(rows)
                command.commit()
                let start = DispatchTime.now().uptimeNanoseconds
                try command.waitBounded()
                hostJoinSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if command.gpuStartTime.isFinite, command.gpuEndTime.isFinite, command.gpuStartTime > 0,
                    command.gpuEndTime >= command.gpuStartTime
                {
                    gpuSeconds += command.gpuEndTime - command.gpuStartTime
                }
                guard command.status == .completed else { throw EngineError.gpu("GDN block GPU failure", command.error) }
                completed += 1
                guard shouldContinue() else { throw IOOperationCancelled() }
                if buffer(.prepareFlags).contents().load(as: UInt32.self) != 0 || buffer(.gateFlags).contents().load(as: UInt32.self) != 0 {
                    replays += 1
                    return nil
                }
                func read(_ source: MTLBuffer, _ count: Int, offset: Int = 0) -> [Float] {
                    readBytes += UInt64(count * 4)
                    return Array(
                        UnsafeBufferPointer(start: source.contents().assumingMemoryBound(to: Float.self).advanced(by: offset), count: count)
                    )
                }
                let output = read(buffer(.output), rows * 2560)
                var captures: [String: [Float]] = [:]
                if capture {
                    for (name, slot, count) in [
                        ("mixed", Slot.mixed, rows * 10240), ("z", .z, rows * 6144),
                        ("a", .a, rows * 48), ("b", .b, rows * 48), ("convolved", .convolved, rows * 10240),
                        ("beta", .beta, rows * 48), ("decay", .decay, rows * 48), ("normed", .normed, rows * 6144),
                        ("gated", .gated, rows * 6144),
                    ] {
                        captures[name] = read(buffer(slot), count)
                    }
                    captures["query"] = read(buffer(.prepared), rows * 2048)
                    captures["keys"] = read(buffer(.prepared), rows * 2048, offset: rows * 2048)
                    captures["values"] = read(buffer(.prepared), rows * 6144, offset: rows * 4096)
                    captures["recurrence"] = read(recurrence, rows * 6144)
                    captures["output"] = output
                }
                let retained = min(3, rows) * 10240
                let tail =
                    state.convolutionTail.suffix(3 * 10240 - retained).map { BF16.encode(BF16.decode($0)) }
                    + read(buffer(.mixed), retained, offset: rows * 10240 - retained).map(BF16.encode)
                var updated = state
                if let next {
                    updated.publishRecurrent(next)
                } else {
                    updated.recurrent = read(recurrence, stateCount, offset: rows * 6144)
                }
                updated.convolutionTail = tail
                updated.offset += rows
                guard shouldContinue() else { throw IOOperationCancelled() }
                return Result(output: output, state: updated, captures: captures)
            }
        }
    }
}
