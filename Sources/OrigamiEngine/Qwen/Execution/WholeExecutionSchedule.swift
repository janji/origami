import Foundation
import Metal
import QuartzCore

/// Serial owner of the whole-execution schedule (`whole_execution_schedule`): one command queue, the actual-row prefill
/// workspace (plus the GDN-only second one of `prefill_double_buffer`), two row-one decode workspaces, the two-row MTP
/// pair workspace, the in-flight command list (at most three, four with pre-encoded decode layers) and the counters.
/// It submits and joins commands and never publishes model state itself. The two decode generations are admitted
/// together and recycled only after joined readers. Wall times include exposed reads and host work; they must not be
/// added to or subtracted from overlapping GPU intervals. Used from the serving (executor) thread only.
public final class WholeExecutionSchedule {
    /// Command, replay and transfer counters of one phase (prefill or decode), cumulative over scopes.
    public struct Counts: Codable, Sendable {
        public var registerAttentionDispatches: UInt64 = 0
        /// Long-context fusion keeps the charged BF16 partials; separate from the register-only short path.
        public var longContextAttentionDispatches: UInt64? = nil
        public var mixedTailTiles: UInt64 = 0, mixedTailRows: UInt64 = 0
        public var scopes: UInt64 = 0, layers: UInt64 = 0, originalReplays: UInt64 = 0
        /// Scopes whose final layer ran routed experts for the read rows only, and the replays that layer took (expected 0).
        public var lastLayerRowSkips: UInt64 = 0, lastLayerSkipReplays: UInt64 = 0
        public var submitted: UInt64 = 0, joined: UInt64 = 0, failed: UInt64 = 0, timed: UInt64 = 0, missingTimestamps: UInt64 = 0
        public var orchestratorUploadBytes: UInt64 = 0, orchestratorReadbackBytes: UInt64 = 0
        public var tokenPlanLayers: UInt64 = 0, tokenPlanReplays: UInt64 = 0, emptyCommandsElided: UInt64 = 0,
            emptyCommandsElidedOnMissLayers: UInt64 = 0
        public var earlyDemandStaged: UInt64 = 0, earlyDemandAdopted: UInt64 = 0, earlyDemandKeys: UInt64 = 0,
            residentJoinsSkipped: UInt64 = 0
        public var earlyDemandBeganBeforeCompletion: UInt64 = 0, earlyDemandFinishedBeforeCompletion: UInt64 = 0
        public var publicationCheckpoints: UInt64 = 0, publicationsBeforeCompletion: UInt64 = 0
        public var publicationCheckpointWaitSeconds = 0.0
        public var earlyDemandCallbackSeconds = 0.0, earlyDemandAdoptionWaitSeconds = 0.0
        /// `decode_preencoded_layers` path proof: decode layers whose experts were queued before their routing was known.
        public var preencodedLayers: UInt64? = nil
        public var wallSeconds = 0.0, gpuSeconds = 0.0, explicitJoinSeconds = 0.0
        /// Where a scope's wall goes: the union of its command intervals on the GPU; the part of queued experts'
        /// intervals before the host's late signal (the GPU waiting for late reads); the host spinning for router
        /// signals and waiting for late expert reads (layers and experts it waited for); the token's readout tail.
        public var gpuUnionSeconds = 0.0, armedStallSeconds = 0.0, routerWaitSeconds = 0.0, lateWaitSeconds = 0.0
        public var lateWaitLayers: UInt64 = 0, lateWaitExperts: UInt64 = 0, tokenTailSeconds = 0.0
        /// Host work: demand windows, encoding the next layer, and residency-set rebuilds (count and time).
        public var windowSeconds = 0.0, encodeNextSeconds = 0.0, residencyRebuilds: UInt64 = 0, residencySeconds = 0.0
        /// The phase change's residency-set build, run on another thread (`prepareDecodeResidency`): off the token path.
        public var residencyPhaseBuilds: UInt64? = nil, residencyPhaseSeconds: Double? = nil
    }
    /// The schedule's counters plus the optional per-switch counters the executor fills in; each optional is nil
    /// when its switch is off, so reports only carry the paths that ran.
    public struct Snapshot: Codable, Sendable {
        public let prefill, decode: Counts
        public let prefillTileBytes, decodeGenerationBytes, metadataBytes: UInt64
        public let accounting: String
        package init(
            prefill: Counts, decode: Counts, prefillTileBytes: UInt64, decodeGenerationBytes: UInt64,
            metadataBytes: UInt64, accounting: String
        ) {
            self.prefill = prefill
            self.decode = decode
            self.prefillTileBytes = prefillTileBytes
            self.decodeGenerationBytes = decodeGenerationBytes
            self.metadataBytes = metadataBytes
            self.accounting = accounting
        }
    }
    let layer: WholeLayerExecution, attention: PinnedAttention, scalars: WholeExecutionScalars
    let gathered: GatheredExpertDecode, reductions: PinnedReductions
    let earlyDemand: DecodeEarlyDemand?
    let prerouting: DecodePrerouting?
    private let kernels: ReferenceKernels, hc: FusedHCDecode, gdn: GDNDeviceBlock
    private let ledger: MemoryLedger, queue: MTLCommandQueue
    private let simdMatrix: Bool, capacity: Int
    private let metadata: MemoryLedger.Reservation
    private var prefillWorkspace: WholeExecutionWorkspace?
    /// The second (GDN-only) C256 workspace of `prefill_double_buffer`; admitted on first use.
    private var prefillSecondWorkspace: WholeExecutionWorkspace?
    private var decodeWorkspaces: [WholeExecutionWorkspace] = []
    private var inFlight: [MTLCommandBuffer] = []
    /// Pre-encoded decode layers (`decode_preencoded_layers`): each decode layer's experts, reduce and mix are queued behind its attention before
    /// the routing is known; the GPU waits on `expertsEvent` while the host fills the records and grids of two groups
    /// (resident at routing, then late). The 96 grid bytes sit inside the metadata reservation.
    let preencoded: Bool
    private let expertsEvent: MTLSharedEvent?, grids: MTLBuffer?, residency: MTLResidencySet?
    private var expertsValue: UInt64 = 0, residencyGeneration: UInt64?
    /// The phase change's build of the residency set, still running or not yet joined.
    private var residencyBuild: ResidencyBuild?
    /// Signalled after each pre-encoded layer's router: its flags and routing are final from then on.
    private let routerEvent: MTLSharedEvent?
    private var routerValue: UInt64 = 0
    /// A submitted pre-encoded experts command and the event value its late group waits for.
    private struct Armed {
        let command: MTLCommandBuffer
        let tile: Int
        let late: UInt64
        var signaled: UInt64
        /// Host time (the GPU timestamps' clock) of the late signal.
        var signaledAt = 0.0
    }
    private var armed: [Armed] = []
    private var prefill = Counts(), decode = Counts(), decoding = false
    private var started: UInt64?
    private var intervals: [(start: Double, end: Double)] = []
    private var registerDispatchesAtStart = 0
    private var longFusedDispatchesAtStart = 0

    package init(
        weights: ReferenceWeights, math: LayerMath, ledger: MemoryLedger, context: Int, gdn: GDNDeviceBlock,
        gathered: GatheredExpertDecode, selector: BatchedTopIndices, reductions: PinnedReductions, injection: FusedResidualInjection,
        decodeEarlyDemand: Bool = false, decodeTokenPlan: Bool = false,
        decodePrerouting: Bool = false, preroutingMargin: Double = 0.25, preroutingOverrideWidth: Int = 0,
        preroutingReadHygiene: Bool = false, preencodedLayers: Bool = false
    ) throws {
        guard ConversationGeometry.contexts.contains(context), let queue = weights.kernels.device.makeCommandQueue() else {
            throw EngineError("Whole execution requires declared conversation geometry", code: "invalid_configuration")
        }
        let helpers = try weights.decodeFrontierHelpers()
        let qsa = try math.wholeQSAHelpers()
        let tables = try DecodeFrontierScalars(
            device: weights.kernels.device, ledger: ledger,
            arithmetic: weights.kernels.arithmetic, fused: helpers.fused)
        let scalar = try WholeExecutionScalars(
            device: weights.kernels.device, sigmoidTable: tables.bf16SigmoidTable,
            siluTable: helpers.fused.decodeRegionSiLUTable)
        let qsaScalar = try QSAExecutionScalars(device: weights.kernels.device, sigmoidBF16: tables.bf16SigmoidTable)
        let selection = try QSADeviceSelection(device: weights.kernels.device, selector: selector)
        let qsaBlock = try QSAExecutionBlock(
            kernels: weights.kernels, dense: helpers.dense, rope: qsa.rope, indexer: qsa.indexer, attention: qsa.attention,
            scalar: qsaScalar, selection: selection, simdMatrix: weights.simdMatrix, ledger: ledger)
        layer = WholeLayerExecution(
            kernels: weights.kernels, dense: helpers.dense, hc: helpers.hc, gdn: gdn,
            qsa: qsaBlock, selector: selector, reductions: reductions, scalars: scalar, injection: injection,
            ledger: ledger, simdMatrix: weights.simdMatrix, deviceConvolutionTail: decodeTokenPlan)
        self.ledger = ledger
        self.queue = queue
        self.kernels = weights.kernels
        hc = helpers.hc
        self.gdn = gdn
        self.gathered = gathered
        self.reductions = reductions
        attention = qsa.attention
        scalars = scalar
        capacity = context
        simdMatrix = weights.simdMatrix
        earlyDemand = decodeEarlyDemand ? try DecodeEarlyDemand(device: weights.kernels.device, ledger: ledger) : nil
        if decodePrerouting {
            guard let earlyDemand, decodeTokenPlan else {
                throw EngineError("Decode pre-routing requires exact early demand and the whole token plan", code: "invalid_configuration")
            }
            prerouting = try DecodePrerouting(
                device: weights.kernels.device, ledger: ledger, listener: earlyDemand.listener,
                hc: helpers.hc, dense: helpers.dense, selector: selector, margin: preroutingMargin, overrideWidth: preroutingOverrideWidth,
                readHygiene: preroutingReadHygiene)
        } else {
            guard preroutingOverrideWidth == 0 else {
                throw EngineError("Slot override requires decode pre-routing", code: "invalid_configuration")
            }
            guard !preroutingReadHygiene else {
                throw EngineError("Read hygiene requires decode pre-routing", code: "invalid_configuration")
            }
            prerouting = nil
        }
        preencoded = preencodedLayers
        if preencodedLayers {
            let device = weights.kernels.device
            guard decodeEarlyDemand, let event = device.makeSharedEvent(),
                let grids = device.makeBuffer(length: 4 * GatheredExpertDecode.deferredGridBytes, options: .storageModeShared)
            else {
                throw EngineError(
                    "Pre-encoded decode layers need exact early demand, a shared event and their grids", code: "invalid_configuration")
            }
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "decode-expert-records"
            descriptor.initialCapacity = 1024
            residency = try device.makeResidencySet(descriptor: descriptor)
            guard let router = device.makeSharedEvent() else { throw EngineError("Router signal event unavailable") }
            expertsEvent = event
            self.grids = grids
            routerEvent = router
        } else {
            expertsEvent = nil
            grids = nil
            residency = nil
            routerEvent = nil
        }
        metadata = try ledger.reserve(bytes: 131072, category: "whole_schedule_metadata")
    }
    deinit {
        // Defensive joined retirement even if a caller fails to finish after a
        // thrown error. Normal paths join explicitly and expose error counters.
        for command in inFlight { try? command.waitBounded() }
        residencyBuild?.done.wait()
        try? ledger.release(metadata)
    }
    private func record(_ change: (inout Counts) -> Void) {
        if decoding { change(&decode) } else { change(&prefill) }
    }
    func recordMixedTail(rows: Int) {
        precondition(!decoding && (1..<256).contains(rows))
        record {
            $0.mixedTailTiles += 1
            $0.mixedTailRows += UInt64(rows)
        }
    }
    package func snapshot() -> Snapshot {
        Snapshot(
            prefill: prefill, decode: decode,
            prefillTileBytes: (prefillWorkspace?.layout.admittedBytes ?? 0) + (prefillSecondWorkspace?.layout.admittedBytes ?? 0),
            decodeGenerationBytes: decodeWorkspaces.reduce(0) { $0 + $1.layout.admittedBytes }, metadataBytes: 131072,
            accounting:
                "Whole scope wall includes host work and exposed reads. Command counts/GPU intervals cover this owner's queue only; original replay, PLE and readout retain separate helpers. Orchestrator transfer bytes exclude internal state, pooled-index, mirror and constant copies. GPU intervals may overlap reads; no additive remainder attribution. Workspace bytes are current live admission, not high-water."
        )
    }
    package func beginScope(decode: Bool) throws {
        guard started == nil, inFlight.isEmpty else { throw EngineError("Whole schedule scope reentry with live work") }
        decoding = decode
        started = DispatchTime.now().uptimeNanoseconds
        registerDispatchesAtStart = attention.registerDecodeDispatches
        longFusedDispatchesAtStart = attention.longFusedDecodeDispatches
    }
    package func endScope() {
        precondition(inFlight.isEmpty, "Whole schedule returned before its GPU readers joined")
        if let start = started {
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let registers = attention.registerDecodeDispatches - registerDispatchesAtStart
            let longFused = attention.longFusedDecodeDispatches - longFusedDispatchesAtStart
            precondition(registers >= 0)
            precondition(longFused >= 0)
            var union = 0.0
            var reached = -Double.infinity
            for interval in intervals.sorted(by: { $0.start < $1.start }) {
                union += max(0, interval.end - max(interval.start, reached))
                reached = max(reached, interval.end)
            }
            intervals.removeAll(keepingCapacity: true)
            record {
                $0.scopes += 1
                $0.wallSeconds += wall
                $0.gpuUnionSeconds += union
                $0.registerAttentionDispatches += UInt64(registers)
                if longFused > 0 { $0.longContextAttentionDispatches = ($0.longContextAttentionDispatches ?? 0) + UInt64(longFused) }
            }
        }
        started = nil
    }
    package func trim() throws {
        guard started == nil, inFlight.isEmpty else { throw EngineError("Whole schedule cannot trim live readers") }
        prefillWorkspace = nil
        prefillSecondWorkspace = nil
        pairWorkspace = nil
        decodeWorkspaces.removeAll()
        // The residency set keeps every slab it holds alive (and charged): empty it so a governor trim between
        // requests frees memory. The next phase change (or pre-encoded layer) rebuilds it.
        joinResidencyBuild()
        residency?.removeAllAllocations()
        residency?.commit()
        residencyGeneration = nil
    }
    package func prepareDecode() throws {
        guard started == nil, inFlight.isEmpty else { throw EngineError("Whole phase change requires joined readers") }
        prefillWorkspace = nil
        prefillSecondWorkspace = nil
    }
    private func makeWorkspace(rows: Int, gdnOnly: Bool = false) throws -> WholeExecutionWorkspace {
        let layout = try WholeExecutionWorkspace.layout(
            rows: rows, capacity: capacity, capture: false,
            argumentBytes: gathered.regionArgumentBufferBytes, kernels: kernels, simdMatrix: simdMatrix,
            separateExpertBindings: earlyDemand != nil, hc: hc, gdn: gdn, attention: attention, gdnOnly: gdnOnly)
        return try WholeExecutionWorkspace(device: kernels.device, ledger: ledger, layout: layout)
    }
    /// The two C256 prefill workspaces of `prefill_double_buffer`: the ordinary tile first, then a
    /// GDN-only tile. Both are admitted before any tile of the scope is submitted.
    func prefillTilePair() throws -> [WholeExecutionWorkspace] {
        let first = try prefillTile()
        guard first.layout.rows == 256, !first.layout.gdnOnly else {
            throw EngineError("Double-buffered prefill requires the ordinary C256 tile")
        }
        if let second = prefillSecondWorkspace { return [first, second] }
        guard inFlight.isEmpty else { throw EngineError("Cannot admit the second prefill tile during execution") }
        let second = try makeWorkspace(rows: 256, gdnOnly: true)
        prefillSecondWorkspace = second
        return [first, second]
    }
    func prefillTile(rows: Int = 256) throws -> WholeExecutionWorkspace {
        guard inFlight.isEmpty else { throw EngineError("Cannot replace prefill tile during execution") }
        guard (1...256).contains(rows) else { throw EngineError("Invalid actual-row prefill tile") }
        if let prefillWorkspace, prefillWorkspace.layout.rows == rows { return prefillWorkspace }
        prefillWorkspace = nil
        prefillSecondWorkspace = nil  // Previous generation has no GPU readers at this boundary.
        let value = try makeWorkspace(rows: rows)
        prefillWorkspace = value
        return value
    }
    /// The two-row workspace of the MTP verify step, admitted on first use.
    func pairTile() throws -> WholeExecutionWorkspace {
        if let pairWorkspace { return pairWorkspace }
        guard inFlight.isEmpty else { throw EngineError("Pair admission occurred after submission") }
        let value = try makeWorkspace(rows: 2)
        pairWorkspace = value
        return value
    }
    private var pairWorkspace: WholeExecutionWorkspace?
    package func decodeTile(_ index: Int) throws -> WholeExecutionWorkspace {
        guard (0...1).contains(index) else { throw EngineError("Whole decode generation index exceeds two") }
        if decodeWorkspaces.isEmpty {
            guard inFlight.isEmpty else { throw EngineError("Decode admission occurred after submission") }
            let first = try makeWorkspace(rows: 1)
            let second = try makeWorkspace(rows: 1)
            decodeWorkspaces = [first, second]
        }
        return decodeWorkspaces[index]
    }
    package func command(_ label: String) throws -> MTLCommandBuffer {
        guard started != nil, let value = queue.makeCommandBuffer() else {
            throw EngineError("Whole command outside an active scope/unavailable")
        }
        value.label = label
        return value
    }
    package func submit(_ command: MTLCommandBuffer, shouldContinue: () -> Bool) throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        guard started != nil, command.commandQueue === queue, command.status == .notEnqueued,
            inFlight.count < (preencoded ? 4 : 3), !inFlight.contains(where: { $0 === command })
        else {
            throw EngineError("Whole command admission/queue differs")
        }
        inFlight.append(command)
        record { $0.submitted += 1 }
        command.commit()
    }
    /// Joins even after cancellation. A caller may inspect flags/cancel only
    /// afterward; nothing in this method publishes model or mirror state.
    package func join(_ command: MTLCommandBuffer) throws {
        guard let index = inFlight.firstIndex(where: { $0 === command }) else {
            throw EngineError("Whole command is unsubmitted or already accounted")
        }
        if armed.contains(where: { $0.command === command && $0.signaled < $0.late }) { releaseArmed() }
        let signaledAt = armed.first { $0.command === command }?.signaledAt
        armed.removeAll { $0.command === command }
        let begin = DispatchTime.now().uptimeNanoseconds
        try command.waitBounded()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e9
        inFlight.remove(at: index)
        record { c in
            c.joined += 1
            c.explicitJoinSeconds += wall
            if command.status != .completed { c.failed += 1 }
            if command.gpuStartTime.isFinite, command.gpuEndTime.isFinite, command.gpuStartTime > 0,
                command.gpuEndTime >= command.gpuStartTime
            {
                c.timed += 1
                c.gpuSeconds += command.gpuEndTime - command.gpuStartTime
                intervals.append((command.gpuStartTime, command.gpuEndTime))
                if let signaledAt {
                    c.armedStallSeconds += max(0, min(signaledAt, command.gpuEndTime) - command.gpuStartTime)
                }
            } else {
                c.missingTimestamps += 1
            }
        }
        guard command.status == .completed else {
            throw EngineError.gpu("Whole command failed after joined completion", command.error)
        }
    }
    package func joinAll() throws {
        releaseArmed()
        var first: Error?
        while let command = inFlight.first {
            do { try join(command) } catch { if first == nil { first = error } }
        }
        if let first { throw first }
    }
    package func joined(_ command: MTLCommandBuffer, shouldContinue: () -> Bool) throws {
        try submit(command, shouldContinue: shouldContinue)
        try join(command)
        guard shouldContinue() else { throw IOOperationCancelled() }
    }
    func joinedFileSelection(_ command: MTLCommandBuffer, shouldContinue: () -> Bool) throws -> MTLCommandBuffer {
        try joined(command, shouldContinue: shouldContinue)
        return try self.command("whole-file-kv-after-selection")
    }
    /// Makes `residency` hold exactly `records` before a command that uses it is encoded: joins the phase change's build
    /// (instant once it finished; a caller that did not check `residencyBuilding` waits here, on the token path, as
    /// before), then rebuilds only when the slabs changed since. Metal makes resident only the allocations a commit adds
    /// (measured), so a rebuild after a slab came or went costs that slab, not the set.
    package func syncResidency(_ records: (generation: UInt64, buffers: [MTLBuffer])) {
        guard let residency else { return }
        let began = DispatchTime.now().uptimeNanoseconds
        let joined = joinResidencyBuild()
        let rebuilt = residencyGeneration != records.generation
        if rebuilt {
            ResidencyBuild.rebuild(residency, records.buffers)
            residencyGeneration = records.generation
        }
        guard joined || rebuilt else { return }
        assert(residencyHolds(records.buffers), "The residency set differs from the record buffers")
        record {
            if rebuilt { $0.residencyRebuilds += 1 }
            $0.residencySeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
        }
    }
    /// The phase change (the prompt read's end, after the decode slabs are prepared): builds the residency set for
    /// `records` on another thread, so the reply's first tokens do not wait for it (seconds when the decode cache
    /// was made resident afresh). Until it finishes the token loop takes the joined path (byte-identical).
    package func prepareDecodeResidency(_ records: (generation: UInt64, buffers: [MTLBuffer])) {
        guard let residency, residencyBuild == nil, residencyGeneration != records.generation else { return }
        residencyBuild = ResidencyBuild(residency, records)
    }
    /// True while the phase change's build runs.
    package var residencyBuilding: Bool { residencyBuild.map { $0.done.wait(timeout: .now()) == .timedOut } ?? false }
    /// Whether the residency set holds exactly `buffers`: every buffer a pre-encoded command can use when it is committed,
    /// and no retired slab (which it would keep alive).
    package func residencyHolds(_ buffers: [MTLBuffer]) -> Bool {
        guard let residency else { return buffers.isEmpty }
        let held = Set(residency.allAllocations.map { ObjectIdentifier($0) })
        return held == Set(buffers.map { ObjectIdentifier($0) })
    }
    @discardableResult private func joinResidencyBuild() -> Bool {
        guard let build = residencyBuild else { return false }
        build.done.wait()
        residencyBuild = nil
        residencyGeneration = build.generation
        decode.residencyPhaseBuilds = (decode.residencyPhaseBuilds ?? 0) + 1
        decode.residencyPhaseSeconds = (decode.residencyPhaseSeconds ?? 0) + build.seconds
        return true
    }
    /// Pre-encoded decode: queue `s`'s experts and submit. Group one is the routed experts the cache's residency table holds from before
    /// `epoch`, looked up and computed on the GPU without the host; group two (the rest) waits for the host's records and
    /// grid; then the canonical reduce and the mix, as the finish command encodes them. Every slab a record can live in is
    /// resident through the residency set; the caller holds slab growth and the layer's records while it is queued.
    func submitPreencodedExperts(
        tile: Int, layer: Int, workspace s: WholeExecutionWorkspace, records: (generation: UInt64, buffers: [MTLBuffer]),
        table: (keys: MTLBuffer, slots: MTLBuffer), epoch: UInt32, shouldContinue: () -> Bool
    ) throws -> MTLCommandBuffer {
        guard preencoded, let event = expertsEvent, let grids, let residency, (0...1).contains(tile),
            !armed.contains(where: { $0.tile == tile })
        else { throw EngineError("Pre-encoded experts outside their two-tile boundary") }
        syncResidency(records)
        let command = try self.command("whole-decode-preencoded-experts")
        command.useResidencySet(residency)
        try gathered.encodeResidentLookup(
            command, pairs: s[.routerPairs], table: table, layer: layer, epoch: epoch,
            buffers: s.experts, grids: grids, offset: tile * 48)
        try gathered.encodeDeferredGroup(
            command, layer: layer, inputBF16: s[.mixedMoEBF16], buffers: s.experts, grids: grids, offset: tile * 48)
        let late = expertsValue + 1
        command.encodeWaitForEvent(event, value: late)
        try gathered.encodeDeferredGroup(
            command, layer: layer, inputBF16: s[.mixedMoEBF16], buffers: s.missingExperts, grids: grids, offset: tile * 48 + 24)
        try reductions.encodeWeightedExpertReduce10(
            command, input: s.experts.down, weights: s[.routerWeights], output: s[.routed], tokens: 1)
        try self.layer.encodeMix(command, workspace: s)
        try submit(command, shouldContinue: shouldContinue)
        expertsValue = late
        armed.append(Armed(command: command, tile: tile, late: late, signaled: late - 1))
        record { $0.preencodedLayers = ($0.preencodedLayers ?? 0) + 1 }
        return command
    }
    /// Pre-encoded decode: writes the late group of `tile`'s armed command (the experts the GPU did not take) and lets it run.
    func fillPreencoded(tile: Int, workspace s: WholeExecutionWorkspace, records: [GatheredExpertDecode.Record], ranks: [Int]) throws {
        guard let event = expertsEvent, let grids, let index = armed.firstIndex(where: { $0.tile == tile }) else {
            throw EngineError("Pre-encoded experts are not armed")
        }
        let value = armed[index].late
        guard armed[index].signaled == value - 1, event.signaledValue == value - 1 else {
            throw EngineError("Pre-encoded experts signalled out of order")
        }
        try gathered.fillDeferredGroup(
            records: records, canonicalRanks: ranks, buffers: s.missingExperts, grids: grids, offset: tile * 48 + 24)
        armed[index].signaled = value
        armed[index].signaledAt = CACurrentMediaTime()
        event.signaledValue = value
    }
    /// Pre-encoded decode: signals the router event after the attention and router encoded so far in `command`.
    func encodeRouterSignal(_ command: MTLCommandBuffer) throws -> (event: MTLSharedEvent, value: UInt64) {
        guard let routerEvent else { throw EngineError("Router signal outside pre-encoded decode") }
        routerValue += 1
        command.encodeSignalEvent(routerEvent, value: routerValue)
        return (routerEvent, routerValue)
    }
    /// MTP verify step without the host: `tile`'s routed experts run from the residency table (records published
    /// before `epoch`), weighted and reduced into its `.routed`; `grids` at `offset` records how many were found.
    /// False when pre-encoded decode (the residency set) is off.
    func encodeResidentExperts(
        _ command: MTLCommandBuffer, tile s: WholeExecutionWorkspace, layer: Int,
        records: (generation: UInt64, buffers: [MTLBuffer]), table: (keys: MTLBuffer, slots: MTLBuffer),
        epoch: UInt32, grids: MTLBuffer, offset: Int
    ) throws -> Bool {
        guard let residency else { return false }
        syncResidency(records)
        command.useResidencySet(residency)
        try gathered.encodeResidentLookup(
            command, pairs: s[.routerPairs], table: table, layer: layer, epoch: epoch,
            buffers: s.experts, grids: grids, offset: offset)
        try gathered.encodeDeferredGroup(
            command, layer: layer, inputBF16: s[.mixedMoEBF16], buffers: s.experts, grids: grids, offset: offset)
        try reductions.encodeWeightedExpertReduce10(
            command, input: s.experts.down, weights: s[.routerWeights], output: s[.routed], tokens: 1)
        return true
    }
    func armedExperts(tile: Int) -> MTLCommandBuffer? { armed.first { $0.tile == tile }?.command }
    func joinIfInFlight(_ command: MTLCommandBuffer) throws {
        if inFlight.contains(where: { $0 === command }) { try join(command) }
    }
    /// Joins every in-flight command except the experts armed for `tile` (the next layer's, which read none of this
    /// layer's records and wait for this host).
    func joinAll(keepingArmed tile: Int) throws {
        var first: Error?
        for command in inFlight where !armed.contains(where: { $0.command === command && $0.tile == tile }) {
            do { try join(command) } catch { if first == nil { first = error } }
        }
        if let first { throw first }
    }
    /// Lets every armed command run with an empty late group (its reduce and mix then write rows the caller discards),
    /// so no join waits on this host.
    private func releaseArmed() {
        guard let event = expertsEvent, let grids else { return }
        for entry in armed where entry.signaled < entry.late {
            let grid = grids.contents().advanced(by: entry.tile * 48 + 24).assumingMemoryBound(to: UInt32.self)
            (grid[0], grid[1], grid[2], grid[3], grid[4], grid[5]) = (20, 0, 1, 80, 0, 1)
        }
        if let last = armed.map(\.late).max(), event.signaledValue < last { event.signaledValue = last }
        for i in armed.indices where armed[i].signaled < armed[i].late {
            armed[i].signaled = armed[i].late
            armed[i].signaledAt = CACurrentMediaTime()
        }
    }
    func layerCompleted(replayed: Bool) {
        record {
            $0.layers += 1
            if replayed { $0.originalReplays += 1 }
        }
    }
    func publicationCheckpoint(early: Bool, waitSeconds: Double) {
        record {
            $0.publicationCheckpoints += 1
            if early { $0.publicationsBeforeCompletion += 1 }
            $0.publicationCheckpointWaitSeconds += waitSeconds
        }
    }
    func recordDecodeWait(
        router: Double = 0, late: Double = 0, lateExperts: Int = 0, tail: Double = 0, window: Double = 0, encode: Double = 0
    ) {
        record {
            $0.windowSeconds += window
            $0.encodeNextSeconds += encode
            $0.routerWaitSeconds += router
            $0.lateWaitSeconds += late
            if lateExperts > 0 {
                $0.lateWaitLayers += 1
                $0.lateWaitExperts += UInt64(lateExperts)
            }
            $0.tokenTailSeconds += tail
        }
    }
    func stagedDemand() { record { $0.earlyDemandStaged += 1 } }
    func adoptedDemand(_ observation: DecodeEarlyDemand.Observation) {
        record {
            $0.earlyDemandAdopted += 1
            $0.earlyDemandKeys += 10
            if observation.beganBeforeCompletion { $0.earlyDemandBeganBeforeCompletion += 1 }
            if observation.finishedBeforeCompletion { $0.earlyDemandFinishedBeforeCompletion += 1 }
            $0.earlyDemandCallbackSeconds += Double(observation.callbackNanoseconds) / 1e9
            $0.earlyDemandAdoptionWaitSeconds += Double(observation.adoptionWaitNanoseconds) / 1e9
        }
    }
    func tokenPlanLayer(replayed: Bool) {
        record {
            $0.tokenPlanLayers += 1
            if replayed { $0.tokenPlanReplays += 1 }
        }
    }
    package func independentCommand(sharedPreencoded: Bool, readyCount: Int, coalesce: Bool, missing: Bool) throws -> MTLCommandBuffer? {
        guard (0...9).contains(readyCount) else { throw EngineError("Independent decode group exceeds resident/miss split") }
        if coalesce && sharedPreencoded && readyCount == 0 {
            elidedEmptyCommand(missing: missing)
            return nil
        }
        return try command("whole-decode-independent-during-demand")
    }
    private func elidedEmptyCommand(missing: Bool) {
        record {
            $0.emptyCommandsElided += 1
            if missing { $0.emptyCommandsElidedOnMissLayers += 1 }
        }
    }
    func skippedResidentJoin() { record { $0.residentJoinsSkipped += 1 } }
    func replay() { record { $0.originalReplays += 1 } }
    var prefillReplays: UInt64 { prefill.originalReplays }
    func skippedLastLayerRows(replays: UInt64) {
        record {
            $0.lastLayerRowSkips += 1
            $0.lastLayerSkipReplays += replays
        }
    }
    func uploaded(_ bytes: Int) { record { $0.orchestratorUploadBytes += UInt64(bytes) } }
    func copy<T>(_ values: [T], to buffer: MTLBuffer, offset: Int = 0) throws {
        let bytes = values.count * MemoryLayout<T>.stride
        guard offset >= 0, offset <= buffer.length, bytes <= buffer.length - offset, buffer.storageMode == .shared else {
            throw EngineError("Whole host copy exceeds admitted span")
        }
        values.withUnsafeBytes {
            if !$0.isEmpty { buffer.contents().advanced(by: offset).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        uploaded(bytes)
    }
    func read(_ buffer: MTLBuffer, count: Int, offset: Int = 0) throws -> [Float] {
        guard count >= 0, offset >= 0, offset <= buffer.length / 4, count <= buffer.length / 4 - offset, buffer.storageMode == .shared
        else {
            throw EngineError("Whole readback exceeds admitted span")
        }
        record { $0.orchestratorReadbackBytes += UInt64(count * 4) }
        return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self).advanced(by: offset), count: count))
    }
    func flagged(_ workspace: WholeExecutionWorkspace, _ slots: [WholeExecutionWorkspace.Slot]) -> Bool {
        slots.contains { workspace[$0].contents().load(as: UInt32.self) != 0 }
    }
}

/// One residency-set build on a global queue; the schedule reads `seconds` and touches the set again only after `done`.
private final class ResidencyBuild: @unchecked Sendable {
    let generation: UInt64
    let done = DispatchGroup()
    private(set) var seconds = 0.0
    private let set: MTLResidencySet, buffers: [MTLBuffer]
    init(_ set: MTLResidencySet, _ records: (generation: UInt64, buffers: [MTLBuffer])) {
        generation = records.generation
        self.set = set
        buffers = records.buffers
        done.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let began = DispatchTime.now().uptimeNanoseconds
            Self.rebuild(self.set, self.buffers)
            seconds = Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
            done.leave()
        }
    }
    static func rebuild(_ set: MTLResidencySet, _ buffers: [MTLBuffer]) {
        set.removeAllAllocations()
        set.addAllocations(buffers)
        set.commit()
        set.requestResidency()
    }
}
