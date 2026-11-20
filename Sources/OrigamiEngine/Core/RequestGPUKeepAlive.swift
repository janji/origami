import CExactIO
import Foundation
import Metal
import ObjectiveC

private final class KeepAliveCharge {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(_ ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "request_gpu_keepalive")
    }
    deinit { try? ledger.release(reservation) }
}
private enum KeepAliveAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Keeps the GPU clocked up for one request (`gpu_keepalive`): a worker keeps at most two one-thread commands
/// queued that spin on a shared flag (bounded by `pollIterations`) until the session closes, so the GPU does not
/// drop its clock between the engine's own short commands. It never reads model memory, and every submitted
/// command is joined before `close()` returns. Adapted from SlotStream's GPUKeepAlive (MIT, see
/// THIRD_PARTY_NOTICES.md); no process-global worker survives a request and no power, driver or system setting
/// is changed.
public final class RequestGPUKeepAlive {
    /// `original` polls up to 200,000 iterations per command; `shortRegisterAttention` 1,024, the bound used with
    /// register QSA decode, which cannot share the long poll (GenerationDriver refuses that pairing).
    public enum Policy: String, Codable, Sendable {
        case original, shortRegisterAttention
        public var pollIterations: Int { self == .original ? 200_000 : 1_024 }
    }
    public let policy: Policy
    /// What one session did: commands submitted, the most in flight, its duration and any GPU failure.
    public struct Snapshot: Codable, Sendable {
        public let policy: Policy
        public let pollIterations: Int
        public let submittedCommands: Int
        public let maximumInFlight: Int
        public let seconds: Double
        public let joined: Bool
        public let error: String?
    }
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let ledger: MemoryLedger
    private let lock = NSLock()
    private weak var active: Session?

    public init(ledger: MemoryLedger, policy: Policy = .original) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Keepalive Metal device unavailable") }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(
            source: """
                #include <metal_stdlib>
                using namespace metal;
                // One thread spins on the session flag until it is set or the bound runs out; no other memory is read.
                kernel void origami_request_keepalive(device atomic_uint *flag [[buffer(0)]]) {
                    for (uint iteration = 0; iteration < \(policy.pollIterations)u; ++iteration) {
                        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
                    }
                }
                """, options: options)
        guard let function = library.makeFunction(name: "origami_request_keepalive") else { throw EngineError("Keepalive kernel missing") }
        pipeline = try device.makeComputePipelineState(function: function)
        self.device = device
        self.ledger = ledger
        self.policy = policy
    }
    deinit { _ = active?.close() }

    public func begin() throws -> Session {
        lock.lock()
        defer { lock.unlock() }
        guard active == nil else { throw EngineError("Keepalive already has an active request") }
        guard let queue = device.makeCommandQueue() else {
            throw EngineError("Keepalive queue allocation failed", code: "insufficient_memory")
        }
        let charge = try KeepAliveCharge(ledger, bytes: 16384)
        guard let flag = device.makeBuffer(length: 16, options: .storageModeShared) else {
            throw EngineError("Keepalive flag allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(flag, &KeepAliveAssociation.key, charge, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard flag.allocatedSize <= 16384 else { throw EngineError("Keepalive flag exceeds charged page") }
        origami_atomic_store_u32(flag.contents(), 0)
        let state = RunningState(queue: queue, pipeline: pipeline, flag: flag, policy: policy)
        let session = Session(state: state) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.active = nil
            self.lock.unlock()
        }
        active = session
        state.group.enter()
        DispatchQueue.global(qos: .userInitiated).async { state.run() }
        return session
    }

    /// One request's keep-alive; closing (or dropping) it stops and joins the worker and its commands.
    public final class Session {
        private let state: RunningState
        private let closeLock = NSLock()
        private var result: Snapshot?
        private var didClose: (() -> Void)?
        fileprivate init(state: RunningState, didClose: @escaping () -> Void) {
            self.state = state
            self.didClose = didClose
        }
        deinit { _ = close() }
        /// Idempotent. The GPU and worker have stopped when this returns, even
        /// after immediate cancellation or a Metal allocation/dispatch failure.
        @discardableResult public func close() -> Snapshot {
            closeLock.lock()
            defer { closeLock.unlock() }
            if let result { return result }
            state.lock.lock()
            state.stopping = true
            origami_atomic_store_u32(state.flag.contents(), 1)
            state.lock.unlock()
            state.group.wait()
            let value = Snapshot(
                policy: state.policy, pollIterations: state.policy.pollIterations, submittedCommands: state.submitted,
                maximumInFlight: state.highWater,
                seconds: Double(DispatchTime.now().uptimeNanoseconds - state.started) / 1e9,
                joined: true, error: state.failure)
            result = value
            didClose?()
            didClose = nil
            return value
        }
    }

    fileprivate final class RunningState: @unchecked Sendable {
        let queue: MTLCommandQueue, pipeline: MTLComputePipelineState, flag: MTLBuffer
        let policy: Policy
        let lock = NSLock(), group = DispatchGroup()
        let started = DispatchTime.now().uptimeNanoseconds
        var stopping = false
        // Worker owns these until group completion synchronizes the final read.
        var submitted = 0, highWater = 0
        var failure: String?
        init(queue: MTLCommandQueue, pipeline: MTLComputePipelineState, flag: MTLBuffer, policy: Policy) {
            self.queue = queue
            self.pipeline = pipeline
            self.flag = flag
            self.policy = policy
        }
        func shouldStop() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return stopping
        }
        func run() {
            defer { group.leave() }
            var pending: [MTLCommandBuffer] = []
            defer {
                origami_atomic_store_u32(flag.contents(), 1)
                for command in pending {
                    try? command.waitBounded()
                    if command.status != .completed { failure = failure ?? "Keepalive GPU completion failed" }
                }
            }
            while !shouldStop() {
                let submittedOK: Bool = autoreleasepool {
                    while pending.count < 2 && !shouldStop() {
                        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                            failure = "Keepalive command allocation failed"
                            return false
                        }
                        command.label = "request-keepalive-bounded-poll"
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(flag, offset: 0, index: 0)
                        encoder.dispatchThreads(
                            MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                        encoder.endEncoding()
                        command.commit()
                        pending.append(command)
                        submitted += 1
                        highWater = max(highWater, pending.count)
                    }
                    return true
                }
                guard submittedOK, let first = pending.first else { break }
                try? first.waitBounded()
                pending.removeFirst()
                guard first.status == .completed else {
                    failure = "Keepalive GPU completion failed"
                    break
                }
            }
        }
    }
}
