import Foundation
import Metal
import ObjectiveC

/// Path proof and accounting for decode next-layer pre-routing:
/// every prediction is staged, adopted or closed, and every predicted expert is
/// gated, resident, issued, dropped, refused or failed. Nothing falls back
/// silently; the authoritative router of the next layer is never replaced.
public struct DecodePreroutingCounts: Codable, Sendable {
    /// Prediction batches encoded (one per target layer 1...47 of a decode token).
    public var batchesStaged: UInt64 = 0
    /// Batches compared against the authoritative routing of their target layer.
    public var batchesAdopted: UInt64 = 0
    /// Batches closed by a replay, an error or sequence end before adoption.
    public var batchesClosedUnadopted: UInt64 = 0
    /// Predicted top-ten entries over adopted batches (ten per batch).
    public var predictedExperts: UInt64 = 0
    /// Predicted entries below the margin threshold: no read considered.
    public var predictionsGated: UInt64 = 0
    /// Entries above the margin that were already resident or in flight: no read.
    public var predictedResident: UInt64 = 0
    /// Reads issued for predicted entries.
    public var predictionsIssued: UInt64 = 0
    /// Issued reads whose expert the target layer then actually demanded.
    public var predictedHits: UInt64 = 0
    /// Issued reads the target layer did not demand (the record stays evictable).
    public var predictedFalse: UInt64 = 0
    /// Actual target-layer experts that were predicted above the margin (resident or issued).
    public var coveredDemand: UInt64 = 0
    /// Entries above the margin the cache could not place without a forced eviction.
    public var predictionsDroppedNoSlot: UInt64 = 0
    /// Entries above the margin the I/O scheduler refused (prefetch admission).
    public var predictionsRefusedQueue: UInt64 = 0
    /// Batches whose callback failed (HC exceptions, invalid logits, cancellation, loader errors).
    public var predictionsFailed: UInt64 = 0
    /// Targets skipped because the token weight plan had no weights for them.
    public var weightsUnavailable: UInt64 = 0
    /// Bytes of issued predicted reads, and of those not demanded, that reached a drive (read
    /// hygiene's cancelled-while-queued reads excluded).
    public var bytesPredictedRead: UInt64 = 0, bytesPredictedFalse: UInt64 = 0
    public var callbackSeconds = 0.0, adoptionWaitSeconds = 0.0
    public var marginThreshold: Double
    /// `decode_prerouting_slot_override` (all nil when off): the per-target-layer width; reads
    /// admitted past the reuse-score rule by evicting a cold record (also in predictionsIssued);
    /// those the target layer then demanded; and the entries still dropped (also in
    /// predictionsDroppedNoSlot) for want of an eligible cold record or after the width was used.
    public var overrideWidth: Int? = nil
    public var overrideAdmitted: UInt64? = nil, overrideHits: UInt64? = nil
    public var overrideNoVictim: UInt64? = nil, overrideWidthExhausted: UInt64? = nil
    /// `decode_prerouting_read_hygiene` (both nil when off): issued predictions the target layer
    /// did not demand that were cancelled at adoption while still queued (never read; also in
    /// predictionsIssued and predictedFalse), and cancellations that found the read shared with
    /// another request or still in flight after their join (consistency; expected 0).
    public var readHygieneCancelled: UInt64? = nil, readHygieneCancelRaces: UInt64? = nil
    public init(marginThreshold: Double, overrideWidth: Int = 0, readHygiene: Bool = false) {
        self.marginThreshold = marginThreshold
        if overrideWidth > 0 {
            self.overrideWidth = overrideWidth
            overrideAdmitted = 0
            overrideHits = 0
            overrideNoVictim = 0
            overrideWidthExhausted = 0
        }
        if readHygiene {
            readHygieneCancelled = 0
            readHygieneCancelRaces = 0
        }
    }
}

/// Decode next-layer pre-routing: layer L+1's MoE hyper-connection and router
/// (the engine's own row kernels, shared with the real layer path) applied to
/// layer L's post-attention residual inside the command that computes it, a
/// shared-event callback that gates the predicted top ten by logit margin and
/// issues prefetch-priority reads through the ordinary loader, and a serial
/// adoption that counts the prediction against the authoritative routing.
/// Predictions change only when a record is read, never which records execute.
package final class DecodePrerouting {
    /// Extra per-drive scheduler lanes and outstanding records that only
    /// prefetch-priority (predicted) reads may use; demand keeps its reserve.
    static let readLanes = 2, outstandingRecords = 20
    package static let metadataBytes: UInt64 = 2 * 1024 * 1024
    /// Metadata plus two page-rounded row workspaces (about 130 KiB each).
    package static let ownerBytes: UInt64 = 4 * 1024 * 1024

    fileprivate final class Allocation: @unchecked Sendable {
        let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
        init(_ ledger: MemoryLedger, bytes: UInt64, category: String) throws {
            self.ledger = ledger
            reservation = try ledger.reserve(bytes: bytes, category: category)
        }
        deinit { try? ledger.release(reservation) }
    }
    private enum Association { nonisolated(unsafe) static var key: UInt8 = 0 }

    /// One row of MoE HC scratch, 512 FP32 logits and the selected pairs. The
    /// callback reads it only after the GPU signal; the serial owner reuses it
    /// only after that callback retired.
    final class Workspace: @unchecked Sendable {
        let hc: FusedHCDecode.RowBuffers
        let logits, pairs: MTLBuffer
        let partial: MTLBuffer?
        let bytes: UInt64
        init(device: MTLDevice, ledger: MemoryLedger) throws {
            let selection = try BatchedTopIndices.bufferPlan(rows: 1, columns: 512, count: 10)
            let plan = try PinnedDenseLinear.plan(rows: 1, width: 2560, outputs: 512)
            let partialBytes = plan.kind == .splitK ? 512 * plan.splitKPartitions * 4 : 0
            let sizes = [
                10240 * 4, 10240 * 2, 320 * 4, 320 * 2, 10240 * 4, 2560 * 4, 80 * 4, 16, 16,
                512 * 4, selection.firstBytes, partialBytes,
            ]
            let total = sizes.reduce(UInt64(0)) { $0 + PrefillMoEBudget.pageRounded($1) }
            let owner = try Allocation(ledger, bytes: total, category: "decode_prerouting_workspace")
            var buffers: [MTLBuffer?] = []
            for size in sizes {
                if size == 0 {
                    buffers.append(nil)
                    continue
                }
                guard let buffer = device.makeBuffer(length: size, options: .storageModeShared),
                    UInt64(buffer.allocatedSize) <= PrefillMoEBudget.pageRounded(size)
                else {
                    throw EngineError("Pre-routing workspace allocation exceeds its admitted page bound", code: "insufficient_memory")
                }
                objc_setAssociatedObject(buffer, &Association.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                buffers.append(buffer)
            }
            hc = .init(
                norm: buffers[0]!, normBF16: buffers[1]!, down: buffers[2]!, activation: buffers[3]!, rawUp: buffers[4]!,
                mixed: buffers[5]!, exceptions: buffers[6]!, rawInjection: buffers[7]!, injection: buffers[8]!, capturedUp: nil)
            logits = buffers[9]!
            pairs = buffers[10]!
            partial = buffers[11]
            bytes = total
        }
        func clearExceptions() {
            hc.exceptions.contents().initializeMemory(as: UInt8.self, repeating: 0, count: 80 * 4)
        }
    }

    /// What the callback issued for one target layer.
    package struct Issue: Sendable {
        package var requests: [ExpertLoadRequest] = []
        package var issued = 0, resident = 0, noSlot = 0, refused = 0
        /// Slot override: experts admitted past the reuse-score rule (their requests are in `requests`).
        package var overrides: [Int] = []
        package var overrideNoVictim = 0, overrideWidthExhausted = 0
        package init() {}
    }
    package typealias Factory = @Sendable ([Int], Int, @Sendable () -> Bool) throws -> Issue
    /// What the callback saw and did for one batch; read by the serial owner at adoption only.
    package struct Outcome: Sendable {
        /// Predicted top ten in rank order, each entry's logit margin over the eleventh logit, and the gated subset.
        package let predicted: [Int], margins: [Float], gated: [Int]
        let issue: Issue
    }
    /// An adopted batch's outcome with its callback time and the host's wait for the callback.
    package struct Observation {
        package let outcome: Result<Outcome, Error>
        let callbackNanoseconds, adoptionWaitNanoseconds: UInt64
    }

    /// One staged prediction for target layer `target`: a shared-event notification that runs the gate and
    /// issues reads on the listener queue, adopted or closed exactly once by the serial owner after its
    /// command has completed (closing first cancels the callback, then discards any issued requests).
    package final class Batch: @unchecked Sendable {
        let target: Int
        private let lock = NSLock(), done = DispatchGroup()
        private let event: MTLSharedEvent, value: UInt64, command: MTLCommandBuffer
        private let allocation: Allocation, workspace: Workspace
        private var cancelled = false, retired = false
        private var result: Result<Outcome, Error>?
        private var callbackNanoseconds: UInt64 = 0
        private let discard: @Sendable ([ExpertLoadRequest]) -> Void
        fileprivate init(
            target: Int, event: MTLSharedEvent, value: UInt64, command: MTLCommandBuffer,
            listener: MTLSharedEventListener, allocation: Allocation, workspace: Workspace, threshold: Float,
            dial: QualityDial, operation: @escaping Factory, discard: @escaping @Sendable ([ExpertLoadRequest]) -> Void
        ) {
            self.target = target
            self.event = event
            self.value = value
            self.command = command
            self.allocation = allocation
            self.workspace = workspace
            self.discard = discard
            done.enter()
            event.notify(listener, atValue: value) { [self] _, _ in
                let began = DispatchTime.now().uptimeNanoseconds
                let result = Result { () throws -> Outcome in
                    guard keepGoing() else { throw IOOperationCancelled() }
                    let selected = try Self.read(workspace)
                    var gated: [Int] = []
                    // A quality dial that drops experts predicts only those it would keep (the router's softmax weights
                    // follow from the margins); a dropped rank's read would be wasted.
                    let top = selected.margins.max() ?? 0
                    let kept = dial.dropsExperts ? Set(dial.keptRanks(selected.margins.map { expf($0 - top) })) : nil
                    for (rank, (id, margin)) in zip(selected.ids, selected.margins).enumerated()
                    where margin >= threshold && kept?.contains(rank) != false { gated.append(id) }
                    guard keepGoing() else { throw IOOperationCancelled() }
                    var issue = Issue()
                    if !gated.isEmpty { issue = try operation(gated, target) { self.keepGoing() } }
                    return Outcome(predicted: selected.ids, margins: selected.margins, gated: gated, issue: issue)
                }
                let finished = DispatchTime.now().uptimeNanoseconds
                lock.lock()
                self.result = result
                callbackNanoseconds = finished - began
                lock.unlock()
                done.leave()
            }
        }
        /// Only the signalled values are read; exception words must be zero.
        private static func read(_ workspace: Workspace) throws -> (ids: [Int], margins: [Float]) {
            let flags = workspace.hc.exceptions.contents().assumingMemoryBound(to: UInt32.self)
            guard (0..<80).allSatisfy({ flags[$0] == 0 }) else { throw EngineError("Pre-routing hyper-connection raised exceptions") }
            let logits = workspace.logits.contents().assumingMemoryBound(to: Float.self)
            let pairs = workspace.pairs.contents().assumingMemoryBound(to: UInt32.self)
            // A pruned GGUF's columns past its experts are masked (NaN) and never routed.
            let routed = ModelLayout.routedExperts
            guard (0..<routed).allSatisfy({ logits[$0].isFinite }) else { throw EngineError("Pre-routing router logits are not finite") }
            var ids: [Int] = []
            ids.reserveCapacity(10)
            for rank in 0..<10 {
                let id = Int(pairs[rank * 2 + 1])
                guard (0..<routed).contains(id), !ids.contains(id) else { throw EngineError("Pre-routing selection is invalid") }
                ids.append(id)
            }
            var eleventh = -Float.infinity
            for id in 0..<routed where !ids.contains(id) { eleventh = max(eleventh, logits[id]) }
            let margins = ids.map { logits[$0] - eleventh }
            guard margins.allSatisfy({ $0 >= 0 }) else { throw EngineError("Pre-routing selection is not the top ten") }
            return (ids, margins)
        }
        private func keepGoing() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return !cancelled
        }
        var isRetired: Bool {
            lock.lock()
            defer { lock.unlock() }
            return retired
        }
        private func drainNotification(cancel: Bool) {
            precondition(
                [MTLCommandBufferStatus.notEnqueued, .completed, .error].contains(command.status),
                "Pre-routing retirement preceded GPU completion")
            lock.lock()
            if cancel { cancelled = true }
            lock.unlock()
            if event.signaledValue < value { event.signaledValue = value }
            done.wait()
        }
        /// Serial owner, after the target layer's window holds its own handles.
        /// Prediction handles are never cancelled here: a false read completes into an
        /// evictable verified record; a hit is already shared by demand. (Read hygiene
        /// cancels only still-queued unrouted reads, in DecodePrerouting.adopt.)
        package func adopt() throws -> Observation {
            guard command.status == .completed else {
                _ = close()
                throw EngineError("Pre-routing adoption requires completed GPU writer")
            }
            let beganWait = DispatchTime.now().uptimeNanoseconds
            drainNotification(cancel: false)
            let waited = DispatchTime.now().uptimeNanoseconds - beganWait
            lock.lock()
            guard !retired else {
                lock.unlock()
                throw EngineError("Pre-routing batch already retired")
            }
            let result = self.result!
            self.result = nil
            retired = true
            let callback = callbackNanoseconds
            lock.unlock()
            return Observation(outcome: result, callbackNanoseconds: callback, adoptionWaitNanoseconds: waited)
        }
        func close() -> Bool {
            if isRetired { return false }
            drainNotification(cancel: true)
            lock.lock()
            let result = self.result
            self.result = nil
            retired = true
            lock.unlock()
            if case .success(let outcome) = result { discard(outcome.issue.requests) }
            return true
        }
    }

    private let allocation: Allocation, event: MTLSharedEvent, listener: MTLSharedEventListener
    private let hc: FusedHCDecode, dense: PinnedDenseLinear, selector: BatchedTopIndices
    private let workspaces: [Workspace]
    private weak var first: Batch?
    private weak var second: Batch?
    private var value: UInt64 = 0
    package private(set) var counts: DecodePreroutingCounts
    let threshold: Float

    /// `hc`, `dense` and `selector` are the schedule's own helpers, so the
    /// prediction encodes the kernels the real layer path encodes.
    /// `overrideWidth` > 0 only reports the slot-override counters; the override
    /// itself reuses admitted expert-cache slots and charges no new owner.
    package init(
        device: MTLDevice, ledger: MemoryLedger, listener: MTLSharedEventListener,
        hc: FusedHCDecode, dense: PinnedDenseLinear, selector: BatchedTopIndices, margin: Double, overrideWidth: Int = 0,
        readHygiene: Bool = false
    ) throws {
        guard margin.isFinite, margin >= 0, margin <= 64 else {
            throw EngineError("Pre-routing margin must be in 0...64", code: "invalid_configuration")
        }
        guard (0...10).contains(overrideWidth) else {
            throw EngineError("Pre-routing override width must be in 0...10", code: "invalid_configuration")
        }
        allocation = try Allocation(ledger, bytes: Self.metadataBytes, category: "decode_prerouting_metadata")
        guard let event = device.makeSharedEvent() else { throw EngineError("Pre-routing shared event unavailable") }
        let first = try Workspace(device: device, ledger: ledger)
        let second = try Workspace(device: device, ledger: ledger)
        guard Self.metadataBytes + first.bytes + second.bytes <= Self.ownerBytes else {
            throw EngineError("Pre-routing owners exceed their declared admission")
        }
        self.event = event
        self.listener = listener
        self.hc = hc
        self.dense = dense
        self.selector = selector
        workspaces = [first, second]
        counts = DecodePreroutingCounts(marginThreshold: margin, overrideWidth: overrideWidth, readHygiene: readHygiene)
        threshold = Float(margin)
    }

    /// Encodes the target layer's MoE HC + router over `afterAttentionBF16`
    /// (the current layer's post-attention row) into this owner's workspace and
    /// signals the callback. The caller has already encoded every exact-demand
    /// signal of this command, so the prediction never delays exact demand.
    package func stage(
        command: MTLCommandBuffer, afterAttentionBF16: MTLBuffer,
        moeHC: DecodeFrontierWeights.HC, router: MTLBuffer, target: Int, dial: QualityDial = .exact,
        operation: @escaping Factory, discard: @escaping @Sendable ([ExpertLoadRequest]) -> Void
    ) throws -> Batch {
        guard (1..<48).contains(target), command.status == .notEnqueued, value < UInt64.max else {
            throw EngineError("Pre-routing target/command is invalid")
        }
        let slot = target % 2
        let previous = slot == 0 ? first : second
        guard previous == nil || previous!.isRetired else { throw EngineError("Pre-routing exceeded its two-batch boundary") }
        let workspace = workspaces[slot]
        workspace.clearExceptions()
        try WholeLayerExecution.encodeHCRow(hc, command, residualBF16: afterAttentionBF16, weights: moeHC, buffers: workspace.hc)
        try WholeLayerExecution.encodeRouterSelection(
            dense, selector, command, mixed: workspace.hc.mixed, router: router,
            logits: workspace.logits, partial: workspace.partial, pairs: workspace.pairs, rows: 1)
        value += 1
        let batch = Batch(
            target: target, event: event, value: value, command: command, listener: listener,
            allocation: allocation, workspace: workspace, threshold: threshold, dial: dial, operation: operation, discard: discard)
        if slot == 0 { first = batch } else { second = batch }
        command.encodeSignalEvent(event, value: value)
        counts.batchesStaged += 1
        return batch
    }

    func recordWeightsUnavailable() { counts.weightsUnavailable += 1 }

    /// Serial owner: compare the prediction with the authoritative sorted ten
    /// keys of its target layer. With read hygiene, `cancelUnrouted` first receives
    /// the issued requests outside the route (after the target window already holds
    /// its own handles); it cancels the still-queued ones and joins them
    /// (ExpertExecution.preroutingCancelUnrouted). Returns the requests outside the
    /// route: a started read runs on, so the caller keeps them until a read boundary
    /// joins them (ExpertExecution.retainPredictions).
    @discardableResult
    package func adopt(_ batch: Batch, actual: [Int], cancelUnrouted: (([ExpertLoadRequest]) -> (cancelled: Int, races: Int))? = nil)
        -> [ExpertLoadRequest]
    {
        let observation: Observation
        do { observation = try batch.adopt() } catch {
            counts.predictionsFailed += 1
            counts.batchesClosedUnadopted += 1
            return []
        }
        counts.batchesAdopted += 1
        counts.callbackSeconds += Double(observation.callbackNanoseconds) / 1e9
        counts.adoptionWaitSeconds += Double(observation.adoptionWaitNanoseconds) / 1e9
        switch observation.outcome {
        case .failure:
            counts.predictionsFailed += 1
            return []
        case .success(let outcome):
            let demanded = Set(actual)
            counts.predictedExperts += UInt64(outcome.predicted.count)
            counts.predictionsGated += UInt64(outcome.predicted.count - outcome.gated.count)
            counts.coveredDemand += UInt64(outcome.gated.filter { demanded.contains($0) }.count)
            counts.predictedResident += UInt64(outcome.issue.resident)
            counts.predictionsDroppedNoSlot += UInt64(outcome.issue.noSlot)
            counts.predictionsRefusedQueue += UInt64(outcome.issue.refused)
            counts.predictionsIssued += UInt64(outcome.issue.issued)
            counts.bytesPredictedRead += UInt64(outcome.issue.issued * PackStore.expertBytes)
            let hits = outcome.issue.requests.filter { demanded.contains($0.key.expert) }.count
            counts.predictedHits += UInt64(hits)
            counts.predictedFalse += UInt64(outcome.issue.requests.count - hits)
            counts.bytesPredictedFalse += UInt64((outcome.issue.requests.count - hits) * PackStore.expertBytes)
            let unrouted = outcome.issue.requests.filter { $0.key.layer == batch.target && !demanded.contains($0.key.expert) }
            if let cancelUnrouted {
                let result = cancelUnrouted(unrouted)
                counts.readHygieneCancelled? += UInt64(result.cancelled)
                counts.readHygieneCancelRaces? += UInt64(result.races)
                // Cancelled while still queued: never read from a drive (all were counted false above).
                counts.bytesPredictedRead -= UInt64(result.cancelled * PackStore.expertBytes)
                counts.bytesPredictedFalse -= UInt64(result.cancelled * PackStore.expertBytes)
            }
            counts.overrideAdmitted? += UInt64(outcome.issue.overrides.count)
            counts.overrideHits? += UInt64(outcome.issue.overrides.filter { demanded.contains($0) }.count)
            counts.overrideNoVictim? += UInt64(outcome.issue.overrideNoVictim)
            counts.overrideWidthExhausted? += UInt64(outcome.issue.overrideWidthExhausted)
            return unrouted
        }
    }
    package func close(_ batch: Batch) {
        if batch.close() { counts.batchesClosedUnadopted += 1 }
    }
}
