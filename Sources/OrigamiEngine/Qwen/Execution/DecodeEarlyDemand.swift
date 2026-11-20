import Foundation
import Metal

/// One event, one serial listener, and at most one unadopted ten-record batch.
/// The GPU signal is encoded AFTER router/preceding validity writes and BEFORE
/// the independent shared branch. No callback writes model or workspace state.
package final class DecodeEarlyDemand {
    package static let metadataBytes: UInt64 = 2 * 1024 * 1024
    fileprivate final class Allocation: @unchecked Sendable {
        let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
        init(_ ledger: MemoryLedger) throws {
            self.ledger = ledger
            reservation = try ledger.reserve(bytes: DecodeEarlyDemand.metadataBytes, category: "decode_early_demand_metadata")
        }
        deinit { try? ledger.release(reservation) }
    }
    /// Immutable GPU-written views. The owning batch must finish its listener
    /// before either workspace is reused or its transaction is discarded.
    struct RouterView: @unchecked Sendable {
        let layer: Int
        let pairs, weights: MTLBuffer
        let flags: [(MTLBuffer, Int)]
        init(next: WholeExecutionWorkspace, preceding: WholeExecutionWorkspace, layer: Int) {
            self.layer = layer
            pairs = next[.routerPairs]
            weights = next[.routerWeights]
            var flags: [(MTLBuffer, Int)] = []
            for role in [
                WholeExecutionWorkspace.Slot.hcAFlags, .hcMFlags, .attentionInjectionFlags,
                .routerFlags, .gdnPrepareFlags, .gdnGateFlags, .qsaPoolFlags, .qsaSelectionFlags, .qsaGateFlags,
            ] {
                flags.append((next[role], role == .hcAFlags || role == .hcMFlags ? 80 : 1))
            }
            for role in [WholeExecutionWorkspace.Slot.sharedFlags, .combinationFlags, .finalInjectionFlags] {
                flags.append((preceding[role], 1))
            }
            self.flags = flags
        }
        // Used only by a reached-event callback. This never reads mutable
        // Pending.finished or calls its command-completion-only routing API.
        func ids() throws -> [Int] {
            for (buffer, count) in flags {
                let words = buffer.contents().assumingMemoryBound(to: UInt32.self)
                guard (0..<count).allSatisfy({ words[$0] == 0 }) else { throw EngineError("Signalled routing needs replay") }
            }
            let p = pairs.contents().assumingMemoryBound(to: UInt32.self)
            let w = weights.contents().assumingMemoryBound(to: Float.self)
            var ids: [Int] = []
            var sum: Float = 0
            for rank in 0..<10 {
                let id = Int(p[rank * 2 + 1])
                let weight = w[rank]
                guard (0..<512).contains(id), !ids.contains(id), weight.isFinite, weight >= 0 else {
                    throw EngineError("Signalled routing IDs/probabilities are invalid")
                }
                ids.append(id)
                sum += weight
            }
            guard sum.isFinite, sum > 0 else { throw EngineError("Signalled router has no probability mass") }
            return ids.sorted()
        }
    }
    /// Whether the callback started and finished before its command completed, and the time spent.
    package struct Observation {
        package let beganBeforeCompletion, finishedBeforeCompletion: Bool
        package let callbackNanoseconds, adoptionWaitNanoseconds: UInt64
    }
    /// The next layer's exact demand, issued by a shared-event callback as soon as the GPU router has written
    /// its ten experts (before the command completes); adopted by the serial owner only after completion and
    /// only if the keys equal the joined routing, otherwise closed and its requests discarded.
    package final class Batch: @unchecked Sendable {
        private let lock = NSLock(), done = DispatchGroup()
        private let event: MTLSharedEvent, value: UInt64, command: MTLCommandBuffer
        private let allocation: Allocation
        private var cancelled = false, retired = false
        private var result: Result<[ExpertLoadRequest], Error>?
        private var beganBeforeCompletion = false, finishedBeforeCompletion = false
        private var callbackNanoseconds: UInt64 = 0, adoptionWaitNanoseconds: UInt64 = 0
        package var observation: Observation {
            lock.lock()
            defer { lock.unlock() }
            precondition(retired, "Observe only after listener adoption/retirement")
            return Observation(
                beganBeforeCompletion: beganBeforeCompletion, finishedBeforeCompletion: finishedBeforeCompletion,
                callbackNanoseconds: callbackNanoseconds, adoptionWaitNanoseconds: adoptionWaitNanoseconds)
        }
        private let discard: @Sendable ([ExpertLoadRequest]) -> Void
        fileprivate init(
            event: MTLSharedEvent, value: UInt64, command: MTLCommandBuffer,
            listener: MTLSharedEventListener, allocation: Allocation,
            operation: @escaping @Sendable (@Sendable () -> Bool) throws -> [ExpertLoadRequest],
            discard: @escaping @Sendable ([ExpertLoadRequest]) -> Void
        ) {
            self.event = event
            self.value = value
            self.command = command
            self.allocation = allocation
            self.discard = discard
            done.enter()
            event.notify(listener, atValue: value) { [self] _, _ in
                let began = DispatchTime.now().uptimeNanoseconds
                let before = self.command.status != .completed && self.command.status != .error
                let result = Result { () throws -> [ExpertLoadRequest] in
                    guard keepGoing() else { throw IOOperationCancelled() }
                    return try operation { self.keepGoing() }
                }
                let finished = DispatchTime.now().uptimeNanoseconds
                let earlyFinish = self.command.status != .completed && self.command.status != .error
                lock.lock()
                self.result = result
                beganBeforeCompletion = before
                finishedBeforeCompletion = earlyFinish
                callbackNanoseconds = finished - began
                lock.unlock()
                done.leave()
            }
        }
        private func keepGoing() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return !cancelled
        }
        fileprivate var isRetired: Bool {
            lock.lock()
            defer { lock.unlock() }
            return retired
        }
        /// Only after the GPU owner joins: force notification for a command
        /// cancelled before submission or failed before reaching its signal.
        private func drainNotification(cancel: Bool) {
            precondition(
                [MTLCommandBufferStatus.notEnqueued, .completed, .error].contains(command.status),
                "Early demand retirement preceded GPU completion")
            lock.lock()
            if cancel { cancelled = true }
            lock.unlock()
            if event.signaledValue < value { event.signaledValue = value }
            done.wait()
        }
        package func adopt(expected: [ExpertKey]) throws -> [ExpertLoadRequest] {
            guard command.status == .completed else {
                close()
                throw EngineError("Early demand adoption requires completed GPU writer")
            }
            let beganWait = DispatchTime.now().uptimeNanoseconds
            drainNotification(cancel: false)
            let waited = DispatchTime.now().uptimeNanoseconds - beganWait
            lock.lock()
            adoptionWaitNanoseconds = waited
            guard !retired else {
                lock.unlock()
                throw EngineError("Early demand already retired")
            }
            let result = self.result!
            self.result = nil
            retired = true
            lock.unlock()
            let requests = try result.get()
            guard requests.map(\.key) == expected else {
                discard(requests)
                throw EngineError("Early demand keys differ from authoritative joined routing")
            }
            return requests
        }
        package func close() {
            if isRetired { return }
            drainNotification(cancel: true)
            lock.lock()
            let result = self.result
            self.result = nil
            retired = true
            lock.unlock()
            if case .success(let requests) = result { discard(requests) }
        }
    }
    /// Serial-host publication boundary. Notification owns only its bounded
    /// group/metadata. Polling the command on a timed wait also handles GPU
    /// failure before the encoded signal, without trusting unsignalled buffers.
    package final class Frontier {
        private let command: MTLCommandBuffer, event: MTLSharedEvent, value: UInt64
        private let allocation: Allocation, done = DispatchGroup()
        private(set) var isClosed = false
        private(set) var waitSeconds = 0.0
        fileprivate init(
            command: MTLCommandBuffer, event: MTLSharedEvent, value: UInt64,
            listener: MTLSharedEventListener, allocation: Allocation
        ) {
            self.command = command
            self.event = event
            self.value = value
            self.allocation = allocation
            done.enter()
            event.notify(listener, atValue: value) { [done, allocation] _, _ in
                withExtendedLifetime(allocation) { done.leave() }
            }
        }
        package func waitUntilReady(shouldContinue: () -> Bool) throws -> Bool {
            guard !isClosed, command.status != .notEnqueued else { throw EngineError("Publication checkpoint is not submitted") }
            let began = DispatchTime.now().uptimeNanoseconds
            defer { waitSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9 }
            while true {
                guard shouldContinue() else { throw IOOperationCancelled() }
                if command.status == .error { throw EngineError.gpu("Publication GPU failed before checkpoint", command.error) }
                // Completion is independently sufficient to read the flags.
                // Do not wait for a late CPU notification before publication.
                if command.status == .completed { return false }
                if done.wait(timeout: .now() + .milliseconds(1)) == .success {
                    guard command.status != .error else { throw EngineError.gpu("Publication GPU failed after checkpoint", command.error) }
                    return command.status != .completed
                }
            }
        }
        package func close() {
            guard !isClosed else { return }
            precondition(
                [MTLCommandBufferStatus.notEnqueued, .completed, .error].contains(command.status),
                "Publication event retired before GPU writers")
            if event.signaledValue < value { event.signaledValue = value }
            done.wait()
            isClosed = true
        }
    }
    private let allocation: Allocation, event: MTLSharedEvent
    /// One serial listener queue, so a pre-routing callback encoded after this
    /// owner's demand signal also runs after that demand was issued.
    let listener: MTLSharedEventListener
    private var value: UInt64 = 0
    private weak var active: Batch?
    private weak var publication: Frontier?
    package init(device: MTLDevice, ledger: MemoryLedger) throws {
        allocation = try Allocation(ledger)
        guard let event = device.makeSharedEvent() else { throw EngineError("Router-ready shared event unavailable") }
        self.event = event
        listener = MTLSharedEventListener(dispatchQueue: DispatchQueue(label: "origami.decode.exact-demand", qos: .userInitiated))
    }
    package func checkpoint(command: MTLCommandBuffer) throws -> Frontier {
        try makeCheckpoint(command: command, encodeSignal: true)
    }
    /// Fault fixture: a completed command without its expected notification.
    package func unsignalledCheckpointForSelfCheck(command: MTLCommandBuffer) throws -> Frontier {
        try makeCheckpoint(command: command, encodeSignal: false)
    }
    private func makeCheckpoint(command: MTLCommandBuffer, encodeSignal: Bool) throws -> Frontier {
        guard command.status == .notEnqueued, publication == nil || publication!.isClosed, value < UInt64.max else {
            throw EngineError("Publication checkpoint exceeds its one-frontier boundary")
        }
        value += 1
        let frontier = Frontier(command: command, event: event, value: value, listener: listener, allocation: allocation)
        publication = frontier
        if encodeSignal { command.encodeSignalEvent(event, value: value) }
        return frontier
    }
    package func stage(
        command: MTLCommandBuffer,
        operation: @escaping @Sendable (@Sendable () -> Bool) throws -> [ExpertLoadRequest],
        discard: @escaping @Sendable ([ExpertLoadRequest]) -> Void
    ) throws -> Batch {
        guard command.status == .notEnqueued, active == nil || active!.isRetired, value < UInt64.max else {
            throw EngineError("Router-ready demand exceeded its one-batch boundary")
        }
        value += 1
        let batch = Batch(
            event: event, value: value, command: command, listener: listener,
            allocation: allocation, operation: operation, discard: discard)
        active = batch
        command.encodeSignalEvent(event, value: value)
        return batch
    }
}
