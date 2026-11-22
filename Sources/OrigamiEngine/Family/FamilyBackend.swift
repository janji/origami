import CryptoKit
import Foundation
import Metal

/// The forward pass a family engine provides to the serving backend.
package protocol FamilyModel: AnyObject {
    /// Logits after running `tokens` on top of the conversation so far.
    func forward(_ tokens: [Int]) throws -> [Float]
    /// Rows before each point a prompt's state is kept at (a saved mark, its end) that need every layer; rows further
    /// back are read with `feed`. nil: every row needs every layer.
    var promptTail: Int? { get }
    /// Reads `tokens` like `forward`, but through only the layers later rows read them through, so the rows must lie more
    /// than `promptTail` before the next kept point. No logits.
    func feed(_ tokens: [Int]) throws
    /// `feed` of the first `fed` rows and `forward` of the rest, in one pass where the model can (each layer's experts
    /// read once for both).
    func forward(_ tokens: [Int], fed: Int) throws -> [Float]
    /// Prompt rows one `forward` or `feed` takes: whole slices of `FamilyRunner.promptSlice` rows.
    var promptRows: Int { get }
    func reset() throws
    var position: Int { get }
    var contextLimit: Int { get }
    var shouldContinue: () -> Bool { get set }
    var routing: QualityDial { get set }
    /// Experts per token the model routes, and its vocabulary.
    var routedExperts: Int { get }
    var vocabulary: Int { get }
    /// Decode uses per routed-expert record since the model loaded (`FamilyExperts.uses`).
    var expertUses: [Int: Int] { get }
    /// Misses per record since the model loaded, in quarters (`FamilyExperts.missed`).
    var expertMisses: [Int: Int] { get }
    var timing: FamilyRunner.Timing { get }
    func statusNotes() -> [String: Any]
    /// The conversation state a saved prompt keeps at `position` tokens: byte ranges from the start of the state
    /// buffers, in a fixed order. Rows that are only ever appended count up to the position; running state (recurrent
    /// layers, the window ring) counts whole.
    func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)]
    /// After a reset and the parts written back: the conversation is `tokens`.
    func restored(_ tokens: [Int])
    /// Positions the per-position state holds now (`FamilyRunner.stateCapacity`), and making it hold `positions`
    /// before the conversation reaches them, between forward passes (`FamilyRunner.reserveState`).
    var stateCapacity: Int { get }
    func reserveState(_ positions: Int) throws
    /// Memory the model holds whatever the conversation (the dense weights and working buffers), its expert slots'
    /// pool and how many slots it holds, the routed experts in all, and the longest context the model takes.
    var memory: (fixed: UInt64, experts: UInt64, slots: Int, totalExperts: Int, maximumContext: Int) { get }
    /// Whether the per-position state is kept in files on disk.
    var contextOnDisk: Bool { get }
}
extension FamilyModel {
    package var promptTail: Int? { nil }
    package var promptRows: Int { FamilyRunner.promptSlice }
    package func feed(_ tokens: [Int]) throws { _ = try forward(tokens) }
    package func forward(_ tokens: [Int], fed: Int) throws -> [Float] {
        if fed > 0 { try feed(Array(tokens[..<fed])) }
        return try forward(Array(tokens[fed...]))
    }
    package var memory: (fixed: UInt64, experts: UInt64, slots: Int, totalExperts: Int, maximumContext: Int) { (0, 0, 0, 0, contextLimit) }
    package var contextOnDisk: Bool { false }
    package var stateCapacity: Int { contextLimit }
    package func reserveState(_ positions: Int) throws {}
    /// The conversation state at `tokens` (a size estimate past the allocated context).
    package func stateBytes(at tokens: Int) -> UInt64 { UInt64(stateParts(at: tokens).reduce(0) { $0 + $1.bytes }) }
}

extension FamilyRunner {
    package var memory: (fixed: UInt64, experts: UInt64, slots: Int, totalExperts: Int, maximumContext: Int) {
        (
            UInt64(weights.residentBytes) + workingBytes + optionalModelCacheBytes, UInt64(experts.poolBytes), experts.slotCount,
            g.experts * g.moeLayers.count, g.context
        )
    }
    package var contextOnDisk: Bool { contextFolder != nil }
    package var routedExperts: Int { g.expertsPerToken }
    package var vocabulary: Int { g.vocab }
    package var expertUses: [Int: Int] { experts.uses }
    package var expertMisses: [Int: Int] { experts.missed }
    /// The read counters, which status also takes while a request runs (`FamilyServingBackend.publishCounters`, on the
    /// request's thread between steps): where the asked records came from, and the bytes read per source.
    package func readCounterNotes() -> [String: Any] {
        let c = experts.counters
        let split = c.split
        var notes: [String: Any] = [
            "expert_hits": c.hits, "expert_misses": c.misses, "expert_hot_reads": c.hotReads, "expert_lookups": split.lookups,
            "expert_read_split": ["memory": split.memory, "hot_copy": split.hot, "model_drive": split.drive],
            "expert_hot_read_gb": Double(c.copyBytes) / 1e9, "expert_gguf_read_gb": Double(c.modelBytes) / 1e9,
            "expert_read_gb": Double(c.readBytes) / 1e9, "expert_read_seconds": c.readSeconds,
            "prompt_read_ahead": c.readAheads, "prompt_read_ahead_used": c.readAheadUsed, "prompt_read_ahead_late": c.readAheadLate,
            "expert_prefetches": c.prefetches, "expert_prefetch_used": c.prefetchUsed,
            "split_reads": experts.splitReads.report,
        ]
        if let shrank = experts.shrank { notes["expert_cache_shrank"] = shrank }
        return notes
    }
    package func statusNotes() -> [String: Any] {
        let c = experts.counters
        let t = timing
        let wiring = experts.wiring
        var notes: [String: Any] = [
            "family": g.family.rawValue, "expert_slots": experts.slotCount,
            "family_dense_wired_bytes": weights.wiredBytes,
            "family_idle_overhead_refresh": refreshesIdleOverhead,
            "family_idle_overhead_refreshes": idleOverheadRefreshes,
            "expert_prefetch_late": c.prefetchLate,
            "expert_prefetch_cancelled": c.prefetchCancelled, "expert_prefetch_read_gb": Double(c.prefetchBytes) / 1e9,
            "expert_prefetch_wasted_gb": Double(c.prefetchWastedBytes) / 1e9,
            "expert_replicas": [
                "replica_records": c.hedges, "replica_winners": c.hedgeReplicaWins, "primary_winners": c.hedgePrimaryWins,
                "replica_failures": c.hedgeReplicaFailures, "replica_read_bytes": c.hedgeBytes,
            ],
            "expert_prefetch_rejected_confidence": c.guessRejectedConfidence, "expert_prefetch_rejected_queue": c.guessRejectedQueue,
            "expert_prefetch_rejected_deadline": c.guessRejectedDeadline, "expert_prefetch_rejected_pressure": c.guessRejectedPressure,
            "expert_cache_active_owned_bytes": experts.residentBytes, "expert_cache_owned_bytes": wiring.ownedBytes,
            "expert_cache_user_wired_bytes": wiring.wiredBytes, "expert_cache_unwired_owned_bytes": wiring.unwiredOwnedBytes,
            "expert_wire_attempts": wiring.attempts, "expert_wire_failures": wiring.failures,
            "expert_wire_failed_bytes_total": wiring.failedBytes, "expert_unwire_attempts": wiring.unwireAttempts,
            "expert_allocation_seconds": c.allocationSeconds, "expert_wire_seconds": wiring.wireSeconds,
            "expert_unwire_failures": wiring.unwireFailures,
            "working_admitted_bytes": workingBytes, "working_allocated_bytes": workingAllocatedBytes,
            "working_retained_bytes": workingRetainedBytes,
            "lazy_startup_workspace": lazyStartupWorkspace,
            "scratch_compaction_enabled": scratchCompactionEnabled, "scratch_compaction_saved_bytes": scratchCompactionSavedBytes,
            "family_grouped_decode": Self.groupsDecodeExperts, "family_grouped_gate_up": Self.groupsGateUp,
            "expert_prefetch_recall": c.predicted > 0 ? Double(c.predictedRight) / Double(c.predicted) : 0,
            "expert_prefetch_issued_by_rank": c.guessIssued, "expert_prefetch_used_by_rank": c.guessUsed, "gpu_seconds": t.gpu,
            "gpu_wait_seconds": t.gpuWait, "expert_wait_seconds": t.reads, "gpu_longest_buffer_seconds": t.longest,
            "gpu_longest_buffer_step": t.longestStep,
            "gpu_seconds_by_command_label": t.gpuByCommandLabels, "gpu_time_attribution": "command_buffer_open_label_not_exclusive_phase",
        ]
        notes.merge(readCounterNotes()) { $1 }
        notes["resident_precision"] = weights.residentPack?.status ?? ["enabled": false]
        notes["prefetch_observations"] = experts.observationNotes
        notes["model_drive_slow_reads"] = experts.modelDriveReads.snapshot().report
        // Since start: decode's drive activity, guesses per drive, router-to-read times and the split rule's state.
        notes["family_overlap"] = DriveActivity.notes(experts.driveActivity.snapshot).merging(c.overlapTiming) { $1 }
            .merging(experts.splitPlannerNotes) { $1 }.merging(["router_to_demand_max_seconds": c.routerToDemandMaximum]) { $1 }
        if wantsRouteTrace { notes["route_trace"] = routeTrace?.snapshot ?? ["enabled": false, "admission_refused": true] }
        return notes
    }
}
extension FamilyRunner {
    /// The tuner's switches for one request (`TuneOverrides`), and what puts the server's own back: requests run one at a
    /// time, so no other reply sees them.
    package func tune(_ overrides: TuneOverrides?) throws -> (() throws -> Void)? {
        guard let overrides,
            overrides.familyPrefetch != nil || overrides.familyPrefetchDepth != nil || overrides.familySpin != nil
                || overrides.familyPrefillSlices != nil
                || overrides.familyEarlyDelayedReads != nil
        else { return nil }
        let saved = (
            guessesNext, experts.guessDepth, FamilyExperts.spins, prefillSlices
        )
        let savedEarlyDelayedReads = earlyDelayedReads
        if let value = overrides.familyPrefetch { guessesNext = value }
        if let depth = overrides.familyPrefetchDepth { experts.guessDepth = depth < 0 ? nil : depth }
        if let value = overrides.familySpin { FamilyExperts.spins = value }
        if let value = overrides.familyEarlyDelayedReads { earlyDelayedReads = value }
        if let value = overrides.familyPrefillSlices { prefillSlices = value }
        return { [self] in
            guessesNext = saved.0
            experts.guessDepth = saved.1
            FamilyExperts.spins = saved.2
            prefillSlices = saved.3
            earlyDelayedReads = savedEarlyDelayedReads
        }
    }
}
extension GLMModel: FamilyModel {}
extension DeepSeekModel: FamilyModel {}

/// The family engine for a GGUF of a served family; `ledger` (the memory governor's) holds the expert slots to its budget.
package func makeFamilyModel(
    _ weights: FamilyWeights, expertBudget: UInt64?, memoryLimit: UInt64? = nil, context: Int,
    contextFile: ((UInt64) -> URL?)? = nil, ledger: MemoryLedger? = nil, log: @escaping (String) -> Void
) throws -> FamilyModel {
    switch weights.geometry.family {
    case .glm:
        return try GLMModel(
            weights: weights, expertBudget: expertBudget, memoryLimit: memoryLimit, context: context, contextFile: contextFile,
            ledger: ledger, log: log)
    case .deepseek:
        return try DeepSeekModel(
            weights: weights, expertBudget: expertBudget, memoryLimit: memoryLimit, context: context, contextFile: contextFile,
            ledger: ledger, log: log)
    case .qwen: throw EngineError("Qwen3.8-Flash-Next runs on Origami's own engine, not a family engine")
    }
}

/// Serves a family engine (GLM-5.3-Flash, DeepSeek-V4.1-Flash) through the same server, wires, think and tool
/// splitting as Qwen. One conversation state: a prompt that extends what the model has read continues from it; any
/// other starts from the longest saved prompt it begins with (`promptCache`, Qwen's own: the end of a system prompt,
/// and of each whole prompt with its logits), else from nothing (recurrent layers cannot rewind). The quality dial acts
/// on decode rows only, as for Qwen: a prompt is read exactly, and a conversation continues only from rows of the same
/// level (or from an exact one).
/// Value-only telemetry captured while the backend exclusively owns an idle model. The dictionary contains
/// scalar/String/Array/Dictionary values, never live cache owners. Copy-on-write preserves each published value.
package struct FamilyServingStatusSnapshot: @unchecked Sendable {
    package private(set) var notes: [String: Any]
    package let memory: (fixed: UInt64, experts: UInt64, slots: Int, totalExperts: Int, maximumContext: Int)
    package let contextLimit, stateCapacity, routedExperts: Int
    package let contextOnDisk: Bool
    package private(set) var capturedAt: TimeInterval
    package private(set) var boundary: String
    fileprivate init(model: FamilyModel, boundary: String) {
        notes = model.statusNotes()
        memory = model.memory
        contextLimit = model.contextLimit
        stateCapacity = model.stateCapacity
        routedExperts = model.routedExperts
        contextOnDisk = model.contextOnDisk
        capturedAt = Date().timeIntervalSince1970
        self.boundary = boundary
    }
    /// This snapshot with a running request's fresh read counters (`FamilyRunner.readCounterNotes`).
    fileprivate func with(_ fresh: [String: Any]) -> Self {
        var copy = self
        copy.notes.merge(fresh) { $1 }
        copy.capturedAt = Date().timeIntervalSince1970
        copy.boundary = "request"
        return copy
    }
}

public final class FamilyServingBackend: ServingBackendStopping, ServingPromptCounting, @unchecked Sendable {
    /// Its lock; a request counts as arriving at it, so the hot copy's builder stops taking turns (`ServingGate`).
    private let gate = ServingGate()
    private var lock: NSLock { gate.lock }
    /// Published values are immutable. Readers must not wait for hot-copy disk I/O under `lock`.
    private let telemetryLock = NSLock()
    private let model: FamilyModel
    package let tokenizer: TokenizerStack
    private let stops: Set<Int>
    private let version: String
    private let fastAllowed: Bool
    private let hostPhaseTiming: Bool
    private var publishedStatus: FamilyServingStatusSnapshot
    private var busy = false
    /// Every token the model has read, in order (prompt and reply), and the dial its reply rows were decoded with.
    private var history: [Int] = [] { didSet { historyFed = nil } }
    /// Rows of `history` below this were only fed (`FamilyModel.feed`): it continues only into a prompt whose next kept
    /// point lies at least the model's tail past it, as the cancelled read it was kept from would have (`fedReach`).
    private var historyFed: Int?
    private var historyQuality = QualityDial.exact
    /// Whether the request that left `history` saves prompts (not `no_store`, nor teacher-forced, and saves on).
    private var historySaves = false
    /// One reply's tool boundary. Only fixed-size mutable state is copied: append-only rows stay in the live model
    /// until the following request consumes or drops this anchor, before any reset or branch can overwrite them.
    private final class ToolAnchor {
        let tokens: [Int], parts: [(index: Int, data: Data)], bytes: UInt64, fed: Int?
        let ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?
        init(
            tokens: [Int], parts: [(buffer: MTLBuffer, bytes: Int)], bytes: UInt64, fed: Int?,
            ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?
        ) {
            self.tokens = tokens
            self.fed = fed
            self.bytes = bytes
            self.ledger = ledger
            self.reservation = reservation
            self.parts = parts.enumerated().compactMap { index, part in
                part.bytes > 0 ? (index, Data(bytes: part.buffer.contents(), count: part.bytes)) : nil
            }
        }
        deinit { if let reservation { try? ledger?.release(reservation) } }
    }
    private var toolAnchor: ToolAnchor?, trimToolAnchor = false
    package static let maximumToolAnchorBytes: UInt64 = 256 << 20
    public var log: (@Sendable (String) -> Void)?
    /// Checks: runs as a restore begins writing a saved state back into the model.
    package var restoreProbe: (() -> Void)?
    /// A log line the dashboard's events list shows too (a GPU interruption's retry); nil: `log`.
    package var event: (@Sendable (String) -> Void)?
    /// Where decode use counts are kept (added to `earlierUses`, the counts saved before this start) after each reply,
    /// for the next start's hot copy, with miss counts in `misses.json` beside it (added to `earlierMisses`); nil keeps none.
    package var usesFile: URL?
    package var earlierUses: [Int: Int] = [:], earlierMisses: [Int: Int] = [:]
    /// The model's own counts when the merged counts last halved (`HotCopy.halved`): only those since then are added.
    fileprivate var halvedAt: (uses: [Int: Int], misses: [Int: Int]) = ([:], [:])
    /// Saved prompts, in the prompt cache Qwen uses; `epoch` binds them to this build, model, tokenizer and context.
    package var promptCache: PromptCache?
    package var epoch = ""
    /// The GPU keep-alive each reply's decode runs under (`gpu_keepalive`), so the GPU keeps its clock across expert-read
    /// gaps: a request's `origami_tune` wins, then `keepAliveSetting` (the setting or tuned), else auto: on unless Low
    /// Power Mode is on, read per reply. Outputs are unchanged: it reads only its own flag. Prompts run without it.
    package var keepAlive: RequestGPUKeepAlive?
    package var keepAliveSetting: Bool?
    package var lowPowerMode: () -> Bool = { ProcessInfo.processInfo.isLowPowerModeEnabled }
    /// The start's warm-up (`warm`): `warmStop` ends it after its current batch, and `warming` is left when it ends
    /// (`warmingNow` until then).
    private var warmStop = false, warmingNow = false
    private let warming = DispatchGroup()
    /// A tuner's request (`origami_tune`, even empty) holds the hot copy's builder and swaps while it runs and this long
    /// after it ends, so no arm reads the drive while a copy is written or reads from a copy the arm before it did not
    /// have. Uptime, not the clock: a clock change cannot stretch it, and an aborted tune frees it this long after.
    package static let tuneHold: TimeInterval = 120
    private var tuning = false, heldUntil = 0.0
    private var held: Bool { tuning || ProcessInfo.processInfo.systemUptime < heldUntil }
    /// Builds and grows the hot copy while serving; told the counts after each reply.
    package var keeper: HotCopyPlaces?

    package init(
        model: FamilyModel, tokenizer: TokenizerStack, stops: Set<Int>, version: String, fastAllowed: Bool,
        hostPhaseTiming: Bool? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.stops = stops
        self.version = version
        self.fastAllowed = fastAllowed
        self.hostPhaseTiming = hostPhaseTiming ?? FamilyHostPhaseTiming.enabled()
        publishedStatus = FamilyServingStatusSnapshot(model: model, boundary: "load")
        if let ledger = (model as? FamilyRunner)?.ledger { tokenizer.admitPromptEncodingCache(ledger: ledger) }
    }

    /// Never samples mutable cache metadata on a status thread, including during a delayed expert fetch.
    /// Readers take only the publication lock; inference adds no per-token or per-expert synchronization.
    package var telemetrySnapshot: FamilyServingStatusSnapshot { telemetryLock.withLock { publishedStatus } }
    /// Status's read counters while a request runs (ROADMAP R14): taken on the request's own thread between steps, where
    /// nothing else changes them, at most once a second; the rest of the snapshot stays the last boundary's.
    private var countersPublishedAt = 0.0
    /// Seconds between two such updates (settable for checks).
    nonisolated(unsafe) package static var countersInterval = 1.0
    /// Whether it published (the caller then updates the request's phase counters too).
    fileprivate func publishCounters(_ runner: FamilyRunner) -> Bool {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - countersPublishedAt >= Self.countersInterval else { return false }
        countersPublishedAt = now
        let fresh = runner.readCounterNotes()
        telemetryLock.withLock { publishedStatus = publishedStatus.with(fresh) }
        return true
    }
    /// Caller holds `lock` and no forward/fetch/warm worker can mutate the model.
    private func publishStatus(_ boundary: String) {
        let snapshot = FamilyServingStatusSnapshot(model: model, boundary: boundary)
        telemetryLock.withLock { publishedStatus = snapshot }
    }

    public func status() -> ServingStatus {
        var status = ServingStatus(version: version, contextTier: model.contextLimit, ready: true)
        status.fastOptIn = fastAllowed
        return status
    }

    public func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        try gate.arrive { try admit(request, cancellation: cancellation) }
    }
    private func admit(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        do { try stopWarming() } catch { throw ServingFailure.from(error) }
        var request = request
        guard request.quality.isExact || fastAllowed else {
            throw ServingFailure(FastLateMissSkip.notAllowed, status: 400, code: "fast_not_allowed")
        }
        do { _ = try request.quality.validated(routedExperts: model.routedExperts) } catch {
            throw ServingFailure(String(describing: error))
        }
        if let teacher = request.teacherTokens, teacher.contains(where: { !(0..<model.vocabulary).contains($0) }) {
            throw ServingFailure("origami_teacher_tokens holds an ID outside the vocabulary")
        }
        if request.tune?.qwenPairProbe == true {
            throw ServingFailure("qwen_pair_probe is supported only by the Qwen backend")
        }
        let prompt: [Int]
        var images: [String] = []
        let encodingStarted = ProcessInfo.processInfo.systemUptime
        do { prompt = try ServingPrompt.encode(request, tokenizer: tokenizer, images: &images) } catch {
            throw ServingFailure.promptEncoding(error)
        }
        let encodingSeconds = ProcessInfo.processInfo.systemUptime - encodingStarted
        guard images.isEmpty else {
            throw ServingFailure("This model reads text only; the request holds images", code: "images_unsupported")
        }
        request.maximumTokens = try request.outputBudget(promptTokens: prompt.count)
        let system = promptCache == nil ? nil : systemEnd(request, prompt: prompt)
        do { try lockBounded() } catch { throw ServingFailure.from(error) }
        defer { lock.unlock() }
        guard !busy else { throw ServingFailure("Engine is busy", status: 503, type: "engine_busy") }
        (model as? FamilyRunner)?.refreshIdleOverhead()
        publishStatus("request_start")
        busy = true
        tuning = request.tune != nil
        return Prepared(
            owner: self, prompt: prompt, request: request, cancellation: cancellation, systemEnd: system, encodingSeconds: encodingSeconds)
    }

    private func finish() {
        lock.lock()
        busy = false
        if tuning {
            tuning = false
            heldUntil = ProcessInfo.processInfo.systemUptime + Self.tuneHold
        }
        if trimToolAnchor {
            clearToolAnchor()
            trimToolAnchor = false
        }
        // A trim left while the request ran applies now, not at the next request's start.
        (model as? FamilyRunner)?.fitIfTrimmed()
        publishStatus("request_end")
        lock.unlock()
    }

    /// The memory governor's trim (`MemoryGovernor`, on its own thread): the expert slots lowered to what its budget
    /// leaves, at once when no request holds the engine, else where the running one next can (`FamilyRunner.trimExperts`).
    /// The bytes released.
    package func trimExperts() -> UInt64 {
        // A drive that stopped answering holds the warm-up or the keeper: nothing is trimmed this time.
        guard (try? stopWarming()) != nil, (try? lockBounded()) != nil else { return 0 }
        defer { lock.unlock() }
        var released: UInt64 = 0
        if busy {
            trimToolAnchor = true
        } else {
            released = toolAnchor?.bytes ?? 0
            clearToolAnchor()
        }
        released += (model as? FamilyRunner)?.trimExperts(idle: !busy) ?? 0
        if !busy { publishStatus("idle_trim") }
        return released
    }

    /// Called on the request's thread (or under `lock` while idle). The reservation follows the immutable bytes'
    /// lifetime; the runner then lends their share of memory back to the expert pool at its next fitting boundary.
    private func clearToolAnchor() {
        toolAnchor = nil
        (model as? FamilyRunner)?.toolAnchorBytes = 0
    }
    private func releaseToolAnchorIfTrimmed() {
        guard toolAnchor != nil else { return }
        lock.lock()
        defer { lock.unlock() }
        if trimToolAnchor {
            clearToolAnchor()
            trimToolAnchor = false
            (model as? FamilyRunner)?.fitExperts()
        }
    }

    /// At a token boundary after GPU completion. `stateParts(at: 0)` names whole recurrent/window buffers and zero
    /// bytes for append-only rows, so capture costs no context-sized copy and no disk I/O.
    private func captureToolAnchor() {
        guard history.count >= GenerationDriver.minimumSavedTokens, history.count == model.position, historyQuality.isExact else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !trimToolAnchor else { return }
        let parts = model.stateParts(at: 0)
        guard parts.allSatisfy({ $0.bytes >= 0 && $0.bytes <= $0.buffer.length }) else { return }
        let bytes = UInt64(parts.reduce(0) { $0 + $1.bytes }) + UInt64(history.count * MemoryLayout<Int>.stride)
        guard bytes <= Self.maximumToolAnchorBytes else { return }
        let runner = model as? FamilyRunner
        let ledger = runner?.ledger
        if let runner {
            guard runner.poolTarget() >= bytes + runner.experts.leastBytes else { return }
            runner.toolAnchorBytes = bytes
            runner.fitExperts()
        }
        let reservation: MemoryLedger.Reservation?
        if let ledger {
            let held = ledger.reservedBytes
            let usable = ledger.usableBytes
            guard held <= usable, bytes <= usable - held,
                let charge = try? ledger.reserve(bytes: bytes, category: "family-tool-anchor")
            else {
                runner?.toolAnchorBytes = 0
                runner?.fitExperts()
                return
            }
            reservation = charge
        } else {
            reservation = nil
        }
        toolAnchor = ToolAnchor(tokens: history, parts: parts, bytes: bytes, fed: historyFed, ledger: ledger, reservation: reservation)
    }

    /// The first position a kept point may lie at after rows fed up to `fed`: no fed row within the model's tail of it.
    private func fedReach(_ fed: Int?) -> Int { fed.map { $0 + (model.promptTail ?? 0) } ?? 0 }

    /// Restores only while the unchanged live prefix still backs this anchor, and only for a strict token prefix.
    /// No logits are needed: at least the next prompt token is always run after the restored boundary.
    private func restoreToolAnchor(_ prompt: [Int]) -> Int {
        defer { clearToolAnchor() }
        guard let anchor = toolAnchor, historyQuality.isExact, model.position == history.count,
            anchor.tokens.count < prompt.count, history.starts(with: anchor.tokens), prompt.starts(with: anchor.tokens),
            fedReach(anchor.fed) <= prompt.count
        else { return 0 }
        let parts = model.stateParts(at: 0)
        guard
            anchor.parts.allSatisfy({
                $0.index < parts.count && parts[$0.index].bytes == $0.data.count
                    && $0.data.count <= parts[$0.index].buffer.length
            })
        else { return 0 }
        for part in anchor.parts {
            part.data.withUnsafeBytes { parts[part.index].buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        model.restored(anchor.tokens)
        history = anchor.tokens
        historyFed = anchor.fed
        historyQuality = .exact
        return anchor.tokens.count
    }

    /// After a start, while no request has come: the empty slots of the model's `experts` filled with the records of
    /// `keys` (most wanted first, `FamilyExperts.warm`) on a utility thread, so the first reply's decode does not miss
    /// almost every record. A request or a trim stops it for good, waiting for at most the batch being read before it
    /// touches the model. `done`: the records placed and the seconds taken.
    package func warm(_ experts: FamilyExperts, keys: [Int], done: @escaping @Sendable (Int, Double) -> Void = { _, _ in }) {
        warming.enter()
        lock.withLock { warmingNow = true }
        DispatchQueue.global(qos: .utility).async { [self] in
            defer {
                lock.withLock {
                    warmingNow = false
                    publishStatus("warm_end")
                }
                warming.leave()
            }
            let started = Date()
            let placed = experts.warm(keys: keys) {
                lock.lock()
                defer { lock.unlock() }
                return warmStop
            }
            done(placed, Date().timeIntervalSince(started))
        }
    }
    private func stopWarming() throws {
        try lockBounded()
        warmStop = true
        lock.unlock()
        let bound = stallBound
        if warming.wait(timeout: .now() + bound) == .timedOut { throw stalled("the warm-up's", after: bound) }
    }
    /// How long a request waits for background reads that hold the model (ROADMAP R2): the warm-up's batch, a record
    /// the hot copy's keeper reads under `lock`.
    private var stallBound: Double { (model as? FamilyRunner)?.experts?.batchStall ?? ExpertReadStall.floor }
    /// Takes `lock` within `stallBound`, else fails as a drive that stopped answering.
    private func lockBounded() throws {
        let bound = stallBound
        guard lock.lock(before: Date() + bound) else { throw stalled("the hot copy's", after: bound) }
    }
    private func stalled(_ whose: String, after seconds: Double) -> EngineError {
        EngineError(
            String(format: "The model drive stopped answering: %@ reads did not return in %.0f s", whose, seconds), code: "drive_stalled")
    }

    /// Whether no request holds the engine or arrives, the warm-up is over and no tuner holds it: when the hot copy's
    /// builder reads the model's drive.
    package var idle: Bool { gate.idle { !busy && !warmingNow && !held } }
    /// An explicit Build now action ends the between-trial hold once the engine is idle.
    /// It cannot interrupt a request or warm-up; ordinary idle maintenance keeps the hold.
    @discardableResult
    package func releaseIdleHotCopyHold() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !busy && !warmingNow && !tuning else { return false }
        heldUntil = 0
        return true
    }
    /// One bounded copy record or publication, serialized with request acquisition. The work
    /// must not call back into this backend; the keeper pauses outside this lock between records.
    package func withIdleHotCopy(_ work: () throws -> Void) rethrows -> Bool {
        try gate.whileIdle({ !busy && !warmingNow && !held }, work)
    }
    /// `experts` reads from `copy` (nil: none) from now on, swapped while no request runs nor the warm-up; false
    /// (nothing changed) while one does.
    package func attachHotCopy(_ copy: HotCopy?, to experts: FamilyExperts) -> Bool {
        attachHotCopies(copy.map { [$0] } ?? [], to: experts)
    }
    /// The same with a copy on each of several drives, fastest first (`HotCopyPlaces`; `FamilyExperts.attach(copies:)`).
    package func attachHotCopies(_ copies: [HotCopy], to experts: FamilyExperts) -> Bool {
        gate.whileIdle({ !busy && !warmingNow && !held }) {
            experts.attach(copies: copies)
            publishStatus("hot_copy_attach")
        }
    }

    /// Where the system prompt ends in `prompt`: the tokens it shares with the same chat rendered with its system
    /// messages and a placeholder user turn, so the boundary is found under any chat template; nil without one.
    private func systemEnd(_ request: ServingRequest, prompt: [Int]) -> Int? {
        var probe = request
        probe.messages = request.messages.filter { $0["role"] == .string("system") || $0["role"] == .string("developer") }
        guard !probe.messages.isEmpty else { return nil }
        probe.messages.append(["role": .string("user"), "content": .string("\u{1}")])
        guard let tokens = try? ServingPrompt.encode(probe, tokenizer: tokenizer) else { return nil }
        let shared = TokenizerStack.commonTokenPrefix(tokens, prompt)
        return shared > 0 ? shared : nil
    }

    /// The longest saved prefix of `prompt` written back into the model: its tokens and, for a whole prompt, its logits.
    /// A saved state that fails its checks is removed and the next one tried. A restore once begun finishes (a read of
    /// the state's bytes) and a cancel takes effect after it, so the cancelled request keeps the restored state as its
    /// conversation (`keep`) instead of a half-written one that must be reset.
    private func restore(_ prompt: [Int], cancellation: GenerationControl) throws -> (tokens: Int, logits: [Float]?) {
        guard let cache = promptCache else { return (0, nil) }
        for candidate in cache.candidates(for: prompt.map(Int64.init)) {
            let folder = cache.stateFolder(candidate.slot)
            do {
                try cancellation.check()
                let header = try FamilyStateFile.header(folder, epoch: epoch, prompt: prompt)
                // A whole prompt starts the reply from its logits; one saved without them is no use here.
                if header.tokens.count == prompt.count && header.logits == 0 { continue }
                try model.reset()
                restoreProbe?()
                let logits = try FamilyStateFile.read(
                    folder, header: header, into: model.stateParts(at: header.tokens.count),
                    expectedLogits: model.vocabulary)
                model.restored(header.tokens)
                cache.touch(candidate.slot)
                return (header.tokens.count, logits)
            } catch {
                // Cancellation says nothing about the saved state's integrity: the entry survives for the next request.
                if cancellation.isCancelled {
                    try? model.reset()
                    try cancellation.check()
                }
                log?("prompt cache: a saved state was not used (\(error)); it is removed")
                cache.forget(candidate.slot)
                try? model.reset()
            }
        }
        return (0, nil)
    }

    /// Saves the model's state at `tokens` (its position now), with `logits` for a whole prompt. A save that fails
    /// costs only the reuse, never the reply. Request cancellation is propagated rather than swallowed. Only exact
    /// states are ever saved: rows read after fast reply rows hold their bits, and a saved state has no quality to say so.
    private func save(_ tokens: [Int], logits: [Float]?, kind: String?, replacing: Int? = nil, cancellation: GenerationControl) throws {
        try cancellation.check()
        guard historyQuality.isExact, let cache = promptCache else { return }
        let parts = model.stateParts(at: tokens.count)
        let started = Date()
        do {
            try cache.writeState(tokens: tokens, kind: kind, replacing: replacing, bytes: FamilyStateFile.bytes(parts, logits: logits)) {
                folder in
                try FamilyStateFile.write(
                    parts, logits: logits, tokens: tokens, epoch: epoch, to: folder,
                    checkCancellation: cancellation.check)
            }
            log?(
                String(
                    format: "prompt cache: saving %d tokens%@ took %.1f s", tokens.count, kind == "system" ? " (system prompt)" : "",
                    Date().timeIntervalSince(started)))
        } catch {
            try cancellation.check()
            if cache.savesPaused == nil { log?("prompt cache: not saved (\(error))") }
        }
    }

    /// At `origami stop`, an unload or a model switch (`LocalHTTPServer.stopAndJoin`): the live conversation saved as
    /// leaving it for another prompt saves it (`prepare`), so the next start restores it instead of reading it again.
    /// Only exact history that a save does not hold already. A save still writing after the save bound for its bytes
    /// (`PrefixReuseController.saveBound`) is cancelled and logged, so a slow drive does not hold the stop up.
    public func serverStopped() { saveLiveConversation() }
    public func promptTokens(_ request: ServingRequest) throws -> Int { try ServingPrompt.encode(request, tokenizer: tokenizer).count }
    @discardableResult package func saveLiveConversation() -> Bool {
        guard (try? stopWarming()) != nil, (try? lockBounded()) != nil else { return false }
        defer { lock.unlock() }
        guard !busy, historySaves, historyQuality.isExact, historyFed == nil, model.position == history.count,
            history.count >= GenerationDriver.minimumSavedTokens, let cache = promptCache, cache.savesOff == nil,
            case let ids = history.map(Int64.init), !cache.contains(ids[...])
        else { return false }
        let bound = PrefixReuseController.saveBound(bytes: FamilyStateFile.bytes(model.stateParts(at: history.count), logits: nil))
        let control = GenerationControl()
        let timer = DispatchWorkItem { control.cancel() }
        DispatchQueue.global().asyncAfter(deadline: .now() + bound, execute: timer)
        defer { timer.cancel() }
        do {
            try save(history, logits: nil, kind: nil, replacing: cache.extended(by: ids)?.slot, cancellation: control)
            return cache.contains(ids[...])
        } catch {
            log?(
                String(
                    format: "stop: the conversation (%d tokens) was not saved within %.0f s; the next start reads it again", history.count,
                    bound))
            return false
        }
    }

    /// `origami save [name]`: the chat's system prompt read (unless saved already) and pinned under `name`.
    public func saveSystemPrompt(_ request: ServingRequest, name: String?, cancellation: GenerationControl) throws -> String {
        if let name, !ChatCommand.isName(name) { throw ServingFailure("The saved name " + ChatCommand.nameRule) }
        guard let cache = promptCache, cache.savesOff == nil else { throw ServingFailure("Saved prompts are off on this server") }
        var probe = request
        probe.messages =
            request.messages.filter { $0["role"] == .string("system") || $0["role"] == .string("developer") }
            + [["role": .string("user"), "content": .string("Hi")]]
        probe.maximumTokens = 1
        probe.command = nil
        probe.noStore = false
        probe.stream = false
        probe.quality = .exact
        probe.teacherTokens = nil
        probe.keep = nil
        let prompt: [Int]
        do { prompt = try ServingPrompt.encode(probe, tokenizer: tokenizer) } catch { throw ServingFailure.promptEncoding(error) }
        guard let end = systemEnd(probe, prompt: prompt) else { throw ServingFailure("This chat has no system prompt to save") }
        guard end >= GenerationDriver.minimumSavedTokens else {
            throw ServingFailure("The system prompt is too short to be worth saving (\(end) tokens; it is read in moments)")
        }
        let ids = prompt.prefix(end).map(Int64.init)[...]
        let label = name ?? String(PromptCache.prefixHash(ids).prefix(8))
        let started = Date()
        let prefilled = !cache.contains(ids)
        if prefilled { _ = try prepare(probe, cancellation: cancellation).run { _ in true } }
        guard cache.pin(ids, name: label) else {
            throw ServingFailure("Could not save the system prompt; try again in a new chat", status: 409)
        }
        return
            "saved the system prompt as **\(label)** (\(end) tokens\(prefilled ? String(format: ", read in %.0f s", Date().timeIntervalSince(started)) : ", already cached")). New chats with it start from there."
    }

    /// The saved prompts' canary (`PromptCanary.familyProbe`): the fixed prompt read on a reset model, then its decode
    /// tokens one at a time, exact; the bits of every step's logits and of the final state. Nothing is restored, and the
    /// decode uses it counts are left out of the usage counts. While loading, or (`yielding`) as idle work: the conversation
    /// in memory is saved first (as at a stop), and a request arriving stops the probe at its next layer step
    /// (`IOOperationCancelled`).
    package func canary(yielding: Bool = false) throws -> String {
        if yielding { saveLiveConversation() }
        var result = ""
        let run = { [self] in
            let probe = PromptCanary.familyProbe(vocabulary: model.vocabulary)
            try model.reset()
            let previous = model.shouldContinue
            if yielding { model.shouldContinue = { [gate] in !gate.waiting } }
            defer {
                model.shouldContinue = previous
                try? model.reset()
                history = []
                halvedAt = (model.expertUses, model.expertMisses)
            }
            model.routing = .exact
            try model.reserveState(probe.prompt.count + probe.decode.count)
            var logits = [try model.forward(probe.prompt)]
            for token in probe.decode { logits.append(try model.forward([token])) }
            let state = model.stateParts(at: model.position).map { UnsafeRawBufferPointer(start: $0.buffer.contents(), count: $0.bytes) }
            result = PromptCanary.digest(logits: logits, state: state)
        }
        guard try gate.whileIdle({ !busy && !warmingNow && !held }, run) else { throw IOOperationCancelled() }
        return result
    }

    /// Saved prompts for `/v1/status`, as Qwen's server reports them.
    package func promptCacheReport() -> [String: Any]? { promptCache?.report }

    /// Debug (`ORIGAMI_FAMILY_PHASE_TIMING`): the first 12 hex digits of the SHA-256 of `bytes`, to compare runs in the log.
    private static func digest(_ bytes: UnsafeRawBufferPointer) -> String {
        SHA256.hash(data: bytes).prefix(6).map { String(format: "%02x", $0) }.joined()
    }
    private static func digest(_ floats: [Float]) -> String { floats.withUnsafeBytes { digest($0) } }
    /// The state's digest, then each part's (`stateParts` order), at the model's position.
    private func stateDigests() -> String {
        var all = SHA256()
        let parts = model.stateParts(at: model.position).map { part -> String in
            let bytes = UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes)
            all.update(bufferPointer: bytes)
            return Self.digest(bytes)
        }
        let whole = all.finalize().prefix(6).map { String(format: "%02x", $0) }.joined()
        return "\(whole) (parts \(parts.joined(separator: ",")))"
    }

    /// Holds the engine from `prepare` until it is dropped, so a request abandoned before `run` (a client that left,
    /// a stream that failed to start) releases it too.
    private final class Prepared: PreparedGeneration, @unchecked Sendable {
        let owner: FamilyServingBackend, prompt: [Int], request: ServingRequest, cancellation: GenerationControl, systemEnd: Int?
        private let lock = NSLock()
        private var consumed = false
        private let encodingSeconds: Double
        init(
            owner: FamilyServingBackend, prompt: [Int], request: ServingRequest, cancellation: GenerationControl, systemEnd: Int?,
            encodingSeconds: Double
        ) {
            self.owner = owner
            self.prompt = prompt
            self.request = request
            self.cancellation = cancellation
            self.systemEnd = systemEnd
            self.encodingSeconds = encodingSeconds
        }
        deinit { owner.finish() }
        func run(emit: @escaping @Sendable (String) throws -> Bool) throws -> GenerationStatistics {
            lock.lock()
            guard !consumed else {
                lock.unlock()
                throw EngineError("Prepared request already consumed")
            }
            consumed = true
            lock.unlock()
            // One seed for every attempt, so a retried reply samples as the first did.
            let retry = GPURetry(log: owner.event ?? owner.log)
            let seed = request.seed ?? UInt64.random(in: .min ... .max)
            let emit = retry.passing(emit)
            do {
                var stats = try retry.run {
                    try cancellation.withAttempt(
                        number: retry.attempt + 1, replayedTokens: retry.replay.count,
                        preparationSeconds: ["prompt_encoding": encodingSeconds]
                    ) {
                        try attempt(retry, seed: seed, emit: emit)
                    }
                }
                if retry.attempt > 0 { stats.gpuRetries = retry.attempt }
                return stats
            } catch { throw GPURetry.isInterruption(error) ? ServingFailure.from(error) : error }
        }
        private func attempt(_ retry: GPURetry, seed: UInt64, emit: @escaping @Sendable (String) throws -> Bool) throws
            -> GenerationStatistics
        {
            let untune = try (owner.model as? FamilyRunner)?.tune(request.tune)
            let result = Result { try attemptBody(retry, seed: seed, emit: emit) }
            do { try untune?() } catch {
                var original = ""
                if case .failure(let cause) = result { original = "; request failed: \(cause)" }
                throw EngineError("Could not restore request cache/tuning settings: \(error)\(original)")
            }
            return try result.get()
        }
        private func attemptBody(_ retry: GPURetry, seed: UInt64, emit: @escaping @Sendable (String) throws -> Bool) throws
            -> GenerationStatistics
        {
            let setupStarted = ProcessInfo.processInfo.systemUptime
            let model = owner.model
            let tokenizer = owner.tokenizer
            let hostTiming = FamilyHostPhaseTiming(enabled: owner.hostPhaseTiming)
            var completed = false
            defer { if !completed { owner.clearToolAnchor() } }
            // An attempt's partial work remains observable if prefill, decode or the client fails. Counters are read
            // on the owning model thread; this never inserts a GPU barrier. withAttempt preserves each attempt.
            var partialStages = ["prompt_encoding": encodingSeconds]
            var partialStage = "restore_and_setup"
            var partialStarted = setupStarted
            var partialCounters: (() -> [String: Double])?
            cancellation.reportGeneratedTokens(0)
            cancellation.report {
                $0.stageSeconds = partialStages
                $0.phaseCounters = [:]
            }
            defer {
                if !completed {
                    partialStages[partialStage] = max(0, ProcessInfo.processInfo.systemUptime - partialStarted)
                    let counters = (partialCounters?() ?? [:]).merging(hostTiming.counters) { $1 }
                    cancellation.report {
                        $0.stageSeconds = partialStages
                        $0.phaseCounters = counters
                    }
                }
            }
            model.shouldContinue = { [cancellation] in !cancellation.isCancelled }
            (model as? FamilyRunner)?.routeTrace?.reset()
            defer { (model as? FamilyRunner)?.routeTrace?.capturesDecode = false }
            // The read counters move in status during the request, split by phase in its phase counters.
            func live() {
                guard let runner = model as? FamilyRunner, owner.publishCounters(runner) else { return }
                if let counters = partialCounters?() { cancellation.report { $0.phaseCounters = counters } }
            }
            if let runner = model as? FamilyRunner {
                runner.activityChanged = { [cancellation] activity in
                    cancellation.report { if $0.phase == "prefill" { $0.activity = activity } }
                    live()
                }
            }
            defer {
                (model as? FamilyRunner)?.activityChanged = nil
                cancellation.report { $0.activity = nil }
            }
            try (model as? FamilyRunner)?.prepareDelayedExpertStorage(for: request.quality)
            defer { (model as? FamilyRunner)?.releaseDelayedExpertStorage() }
            // Continue the conversation when this prompt extends it from rows of this level (or exact ones); else start
            // from the longest saved prompt it begins with, or over.
            var reused = 0
            var restored = false
            var logits: [Float] = []
            // The per-position state holds the prompt and the reply's first 1,024 tokens (a longer reply grows it in the
            // forward pass), the rest lent to the expert slots; a restore writes at most the prompt's rows.
            let reach = min(model.contextLimit, prompt.count + min(request.maximumTokens, 1024))
            let hadToolAnchor = owner.toolAnchor != nil
            // The expert slots regrow to what the governor's budget leaves once memory is free again (`fitExperts`).
            (model as? FamilyRunner)?.fitExperts()
            let continuesLive =
                !owner.history.isEmpty && owner.history.count < prompt.count && model.position == owner.history.count
                && (owner.historyQuality.isExact || owner.historyQuality == request.quality) && prompt.starts(with: owner.history)
                && owner.fedReach(owner.historyFed) <= prompt.count
            if continuesLive {
                owner.clearToolAnchor()
                reused = owner.history.count
                try model.reserveState(reach)
            } else if case let anchored = owner.restoreToolAnchor(prompt), anchored > 0 {
                reused = anchored
                restored = true
                try model.reserveState(reach)
            } else {
                // Leaving the conversation in memory: its state is saved first, in the slot of the saved end it extends
                // (its prompt's), so a side request or a later mismatch costs it no re-read of its reply when it comes
                // back. Not when this prompt branches from that saved end (a regenerated or edited turn): restoring
                // it serves this prompt, and the branch left is not coming back.
                if owner.historySaves, owner.historyQuality.isExact, owner.historyFed == nil, model.position == owner.history.count,
                    owner.history.count >= GenerationDriver.minimumSavedTokens, let cache = owner.promptCache, cache.savesOff == nil,
                    case let ids = owner.history.map(Int64.init), !cache.contains(ids[...])
                {
                    let extended = cache.extended(by: ids)
                    if extended.map({ !prompt.starts(with: owner.history.prefix($0.tokens)) }) ?? true {
                        try owner.save(owner.history, logits: nil, kind: nil, replacing: extended?.slot, cancellation: cancellation)
                    }
                }
                // A model that has read nothing is clean already.
                if model.position != 0 || !owner.history.isEmpty { try model.reset() }
                owner.history = []
                owner.historyQuality = .exact
                try model.reserveState(reach)
                let saved = try owner.restore(prompt, cancellation: cancellation)
                reused = saved.tokens
                restored = saved.tokens > 0
                logits = saved.logits ?? []
                owner.history = Array(prompt.prefix(reused))
            }
            if hadToolAnchor { (model as? FamilyRunner)?.fitExperts() }
            // Rows the conversation continued from were only fed below this (a cancelled read kept at a fed point).
            let fedBefore = owner.historyFed
            retry.note(from: reused)
            let setupEnded = ProcessInfo.processInfo.systemUptime
            let beforePrefillIO = (model as? FamilyRunner)?.experts.counters
            let beforeRefusals = (model as? FamilyRunner)?.experts.ledger?.optionalRefusals
            partialStages[partialStage] = setupEnded - partialStarted
            partialStage = "prefill_and_save"
            partialStarted = setupEnded
            if let before = beforePrefillIO, let runner = model as? FamilyRunner {
                partialCounters = { [runner] in
                    runner.experts.counters.reads(since: before, phase: "prefill").merging([
                        "prefill_expert_read_bytes": Double(runner.experts.counters.readBytes - before.readBytes)
                    ]) { $1 }
                }
            }
            cancellation.report {
                $0.phase = "prefill"
                $0.activity = nil
                $0.promptTokens = prompt.count
                $0.restoredTokens = reused
                $0.completedPromptTokens = reused
            }
            owner.promptCache?.explainRestore(prompt, restored: reused)
            let started = Date()
            model.routing = .exact
            // What to save, each when long enough and not saved yet: the end of the system prompt (and the part a changed
            // one shares with an earlier one), and the whole prompt. A teacher-forced or no-store request saves nothing.
            let cache = request.noStore || request.teacherTokens != nil ? nil : owner.promptCache.flatMap { $0.savesOff == nil ? $0 : nil }
            let minimum = GenerationDriver.minimumSavedTokens
            var marks: [Int: String] = [:]
            if let cache, let end = systemEnd {
                let shared = cache.rememberSystemBlock(Array(prompt.prefix(end)))
                // Not within the tail of rows fed before: those would hold a state some rows of which lack later layers.
                for offset in [end, shared]
                where offset > reused && offset >= minimum && offset < prompt.count && offset >= owner.fedReach(fedBefore)
                    && !cache.contains(prompt.prefix(offset).map(Int64.init)[...])
                { marks[offset] = "system" }
            }
            (model as? FamilyRunner)?.planPrompt(rows: prompt.count - reused)
            if prompt.count - reused > 1 { try (model as? FamilyRunner)?.beginPrefill() }
            cache?.noteRead(prompt.count - reused)
            do {
                // In slices, so prompt scratch (streams, router rows, candidates) stays bounded at any prompt length; a
                // slice also ends where a state is saved. A whole slice more than the model's tail before that end (a
                // mark or the prompt's end) is only fed, so every kept state holds what its rows need at every layer; the
                // slices stay where they were, so a cut never reads a layer's experts once more. Consecutive slices run as
                // one call of up to `promptRows` (each layer's experts read once for them), fed ones before whole ones,
                // every row in the slice and of the kind it had alone.
                var first = reused
                // A cancel goes back to the last slice end read instead of resetting the conversation: its running state
                // is copied there (`captureToolAnchor`; rows below it are never written again) and written back. A point
                // is whole when no fed row lies within the model's tail before it; one that is not (a long fresh DeepSeek
                // prompt) is kept with where its fed rows end, and continues only into prompts it is whole for.
                var fedTo = fedBefore
                func keep() {
                    guard first < prompt.count else { return }
                    owner.history = Array(prompt.prefix(first))
                    if owner.fedReach(fedTo) > first { owner.historyFed = fedTo }
                    owner.clearToolAnchor()
                    owner.captureToolAnchor()
                }
                keep()
                while first < prompt.count {
                    try cancellation.check()
                    try ServingFailure.checkTimeLimit(request)
                    let end = marks.keys.filter { $0 > first }.min() ?? prompt.count
                    let slice = FamilyRunner.promptSlice
                    func fed(_ last: Int) -> Bool { model.promptTail.map { last + $0 <= end } ?? false }
                    var last = min(end, first + slice)
                    var fedEnd = fed(last) ? last : first
                    while last < end, last - first < model.promptRows {
                        last = min(end, last + slice)
                        if fed(last) { fedEnd = last }
                    }
                    if fedEnd == last {
                        try hostTiming.measure(.prefillFeed) { try model.feed(Array(prompt[first..<last])) }
                    } else {
                        logits = try hostTiming.measure(.prefillForward) {
                            try model.forward(Array(prompt[first..<last]), fed: fedEnd - first)
                        }
                    }
                    if fedEnd > first { fedTo = fedEnd }
                    first = last
                    keep()
                    // For /v1/status and stream progress: one locked store per slice.
                    cancellation.report { $0.completedPromptTokens = first }
                    live()
                    if let kind = marks[first] {
                        try hostTiming.measure(.prefillSave) {
                            try owner.save(Array(prompt.prefix(first)), logits: nil, kind: kind, cancellation: cancellation)
                        }
                    }
                }
            } catch {
                // The prompt's working memory, and what the expert pool lent it (`planPrompt`), go back now: the pass
                // has ended its buffers, so nothing waits for the next request to return them.
                defer { (model as? FamilyRunner)?.beginDecode() }
                // A forward pass that throws leaves the position where it was.
                owner.history = Array(prompt.prefix(model.position))
                if cancellation.isCancelled || ServingFailure.isTimeLimit(error), !GPURetry.isInterruption(error),
                    owner.restoreToolAnchor(prompt) > 0
                {
                    (model as? FamilyRunner)?.fitExperts()
                    throw ServingFailure.from(error)
                }
                try? model.reset()
                owner.history = []
                throw GPURetry.isInterruption(error) ? error : ServingFailure.from(error)
            }
            if owner.toolAnchor != nil {
                owner.clearToolAnchor()
                (model as? FamilyRunner)?.fitExperts()
            }
            owner.history = prompt
            owner.historySaves = cache != nil
            // Saved even when this prompt has an entry: this request read some of it, so the entry may hold other bits (a
            // turn continued live where an earlier one read the whole prompt), and a restart must continue this state.
            // With no slot free, in the slot of the saved end it extends (its conversation's), as the leave-save does.
            if let cache, prompt.count >= minimum, reused < prompt.count {
                // An entry for this very prompt (another process's bits) is overwritten in its own slot instead.
                let ids = prompt.map(Int64.init)
                let replacing = cache.contains(ids[...]) ? nil : cache.extended(by: ids)?.slot
                try hostTiming.measure(.prefillSave) {
                    try owner.save(prompt, logits: logits, kind: nil, replacing: replacing, cancellation: cancellation)
                }
            }
            if let name = request.keep, let cache, cache.keep(prompt.map(Int64.init)[...], name: name) == false {
                let why = owner.historyQuality.isExact ? "" : " (it continues fast reply rows, and only exact states are saved)"
                owner.log?("keep: no saved state holds this prompt (\(prompt.count) tokens)\(why), so \(name) was not kept")
            }
            let prefillEnded = ProcessInfo.processInfo.systemUptime
            (model as? FamilyRunner)?.activityChanged = nil
            cancellation.report {
                $0.phase = "decode"
                $0.activity = nil
            }
            (model as? FamilyRunner)?.routeTrace?.capturesDecode = true
            (model as? FamilyRunner)?.experts.resetRequestMaximum()
            let afterPrefillIO = (model as? FamilyRunner)?.experts.counters
            let promptRefusals = (model as? FamilyRunner)?.experts.ledger?.optionalRefusals
            let afterPrefillActivity = (model as? FamilyRunner).map { DriveActivity.notes($0.experts.driveActivity.snapshot) }
            partialStages[partialStage] = prefillEnded - partialStarted
            partialStage = "generation"
            partialStarted = prefillEnded
            if let before = beforePrefillIO, let prompt = afterPrefillIO, let runner = model as? FamilyRunner {
                partialCounters = { [runner] in
                    let now = runner.experts.counters
                    return prompt.reads(since: before, phase: "prefill").merging(now.reads(since: prompt, phase: "decode")) { $1 }
                        .merging([
                            "prefill_expert_read_bytes": Double(prompt.readBytes - before.readBytes),
                            "decode_expert_read_bytes": Double(now.readBytes - prompt.readBytes),
                        ]) { $1 }
                }
            }
            owner.log?(
                String(
                    format: "prompt: %d tokens (%d %@) in %.1f s", prompt.count, reused, restored ? "restored" : "continued",
                    Date().timeIntervalSince(started)))
            // Debug: what the reply starts from, to compare a continued and a restored conversation (ROADMAP R19).
            var decodeDigests: [String] = []
            if owner.hostPhaseTiming {
                owner.log?("digests at \(model.position): logits \(FamilyServingBackend.digest(logits)) state \(owner.stateDigests())")
            }
            let stops = try StopSequenceFilter(
                stops: request.stopSequences, reasoningOpen: !["none", "minimal"].contains(request.reasoningEffort))
            var sampler = ReferenceSampler(seed: seed)
            var output: [Int] = []
            var visible = ""
            var finish = "length"
            var consumerStopped = false
            var reply = ReplyText(tokenizer)
            var exactFrom: Int?
            // Times each token was emitted, for the presence and frequency penalties. Keep the exact emitted (possibly
            // forced/replayed) counts rather than rebuilding them from the whole reply at every sample.
            var generated: [Int: Int] = [:]
            let thinking = !["none", "minimal"].contains(request.reasoningEffort)
            // As Qwen's: the think block closes at the budget, or where only the answer room is left, so a reply that
            // runs out of tokens while thinking still answers. This family's own control token closes it.
            let closer =
                thinking && request.teacherTokens == nil
                ? tokenizer.thinkEnd.map {
                    ThinkCloser(
                        open: true, maximumTokens: request.maximumTokens, budget: request.thinkingBudget,
                        end: $0, closing: [$0] + tokenizer.encode("\n\n"))
                } : nil
            let constraint = request.decodeGrammar.map {
                JSONConstraint(
                    grammar: $0, tokenizer: tokenizer, vocabularySize: model.vocabulary, stops: owner.stops.sorted(), thinking: thinking,
                    toolOpener: (model as? FamilyRunner).flatMap { request.toolOpener(.of($0.weights.geometry.family)) })
            }
            var forced: [Int] = []
            var teacher = request.teacherTokens.map { _ in TeacherAgreement() }
            var toolCalls = ToolCallAnchorTracker(openers: tokenizer.toolCallOpeners, markers: tokenizer.toolCallTextMarkers)
            let savesToolAnchor = cache != nil && request.quality.isExact
            (model as? FamilyRunner)?.beginDecode()
            let decodeStart = Date()
            // Closed when the reply ends, is cancelled or fails; every queued spin command is joined first.
            let keepsAlive = request.tune?.gpuKeepAlive ?? owner.keepAliveSetting ?? !owner.lowPowerMode()
            let keepAlive = keepsAlive ? try owner.keepAlive?.begin() : nil
            defer { keepAlive?.close() }
            while output.count < request.maximumTokens {
                try cancellation.check()
                owner.releaseToolAnchorIfTrimmed()
                let token = try hostTiming.measure(.sampleForceReplay) {
                    if forced.isEmpty, let closing = closer?.next(output) { forced = closing }
                    var picked: Int
                    if forced.isEmpty {
                        picked = try sampler.next(
                            logits: constraint?.masked(logits) ?? logits, parameters: request.sampling, generated: generated)
                        // Teacher forcing: score this step's own pick, then continue from the teacher's token.
                        if let teacherTokens = request.teacherTokens, output.count < teacherTokens.count {
                            teacher?.record(picked, teacherTokens[output.count], position: output.count, logits: logits)
                            picked = teacherTokens[output.count]
                        }
                    } else {
                        picked = forced.removeFirst()
                    }
                    let token = retry.token(output.count, picked: picked)
                    try constraint?.accept(token)
                    return token
                }
                if owner.stops.contains(token) {
                    finish = "stop"
                    break
                }
                output.append(token)
                retry.made(output.count - 1, token)
                generated[token, default: 0] += 1
                cancellation.reportGeneratedTokens(output.count)
                live()
                let (delta, openedTool) = try hostTiming.measure(.detokenizeTools) {
                    var decoded = try reply.text(output)
                    while decoded.last == "\u{FFFD}" { decoded.removeLast() }
                    guard let delta = GenerationDriver.streamDelta(decoded, after: visible) else {
                        throw EngineError("Tokenizer streaming prefix changed")
                    }
                    visible = decoded
                    return (delta, toolCalls.observe(token: token, delta: delta))
                }
                if openedTool {
                    if !request.quality.isExact { exactFrom = output.count }
                    if savesToolAnchor { hostTiming.measure(.toolAnchorCopy) { owner.captureToolAnchor() } }
                }
                let shown = hostTiming.measure(.stopFilter) { stops.push(delta) }
                if !shown.isEmpty, try !hostTiming.measure(.emitCallback, { try emit(shown) }) {
                    consumerStopped = true
                    finish = "tool_calls"
                    break
                }
                if stops.didStop {
                    finish = "stop"
                    break
                }
                // The time limit ends the reply as `max_tokens` does: this token sent, not yet read.
                guard output.count < request.maximumTokens, !ServingFailure.pastTimeLimit(request) else { break }
                // A dynamic dial picks each step's knobs from the confidence in the token just produced; a tool call
                // decodes exact from its opener on.
                model.routing = exactFrom == nil ? request.quality.step(after: logits) : .exact
                do { logits = try hostTiming.measure(.decodeForward) { try model.forward([token]) } } catch {
                    try? model.reset()
                    owner.history = []
                    throw GPURetry.isInterruption(error) ? error : ServingFailure.from(error)
                }
                owner.history.append(token)
                owner.historyQuality = request.quality
                if owner.hostPhaseTiming { decodeDigests.append(String(FamilyServingBackend.digest(logits).prefix(8))) }
            }
            if owner.hostPhaseTiming {
                owner.log?(
                    "digests at \(model.position): decode logits \(decodeDigests.joined(separator: " ")) state \(owner.stateDigests())")
            }
            if !consumerStopped && !stops.didStop {
                let tail = hostTiming.measure(.stopFilter) { stops.finish() }
                if !tail.isEmpty { _ = try hostTiming.measure(.emitCallback) { try emit(tail) } }
            }
            // The reply is complete; a keep-alive failure only cost speed.
            if let error = keepAlive?.close().error { owner.log?("GPU keepalive failed: \(error)") }
            let seconds = Date().timeIntervalSince(decodeStart)
            let finalizationStarted = ProcessInfo.processInfo.systemUptime
            partialStages[partialStage] = finalizationStarted - partialStarted
            partialStage = "finalization"
            partialStarted = finalizationStarted
            owner.log?(
                String(
                    format: "reply: %d tokens in %.1f s (%.2f tok/s) %@", output.count, seconds,
                    output.count > 1 ? Double(output.count - 1) / seconds : 0, "\(model.statusNotes())"))
            // The counts the hot copy is chosen from, read here on the model's thread: saved for the next start (a failed
            // save costs only that choice, not this reply) and handed to the keeper, which builds once this reply is done.
            var uses = HotCopy.mergeUses(owner.earlierUses, HotCopy.added(model.expertUses, since: owner.halvedAt.uses))
            var misses = HotCopy.mergeUses(owner.earlierMisses, HotCopy.added(model.expertMisses, since: owner.halvedAt.misses))
            let perToken = (model as? FamilyRunner).map { $0.weights.geometry.moeLayers.count * model.routedExperts } ?? 0
            if let halved = HotCopy.halved(uses, misses, floor: HotCopy.decayTokens * perToken) {
                (uses, misses) = halved
                (owner.earlierUses, owner.earlierMisses) = halved
                owner.halvedAt = (model.expertUses, model.expertMisses)
            }
            if let file = owner.usesFile {
                try? HotCopy.saveUses(uses, to: file)
                try? HotCopy.saveUses(misses, to: file.deletingLastPathComponent().appendingPathComponent("misses.json"))
            }
            owner.keeper?.trigger(counts: (uses, misses))
            var statistics = GenerationStatistics(
                promptTokens: prompt.count, completionTokens: output.count, cachedTokens: reused,
                finishReason: finish, quality: request.quality)
            if request.returnTokens { statistics.outputTokens = output }
            statistics.teacher = teacher
            statistics.exactFrom = exactFrom
            statistics.stopSequence = stops.matched
            if thinking, let end = tokenizer.thinkEnd {
                statistics.reasoningTokens = output.firstIndex(of: end) ?? output.count
                statistics.reasoningClosedBy = closer?.closedBy
            }
            statistics.stageSeconds = [
                "prompt_encoding": encodingSeconds, "restore_and_setup": setupEnded - setupStarted,
                "prefill_and_save": prefillEnded - setupEnded,
                "generation": finalizationStarted - prefillEnded,
                "finalization": ProcessInfo.processInfo.systemUptime - finalizationStarted,
            ]
            if let before = beforePrefillIO, let prompt = afterPrefillIO, let runner = model as? FamilyRunner {
                let end = runner.experts.counters
                // Logical source bytes, including admitted read-ahead; these deltas add no reads or GPU barriers.
                // Optional host wall observations are merged separately below.
                statistics.phaseCounters = [
                    "prefill_expert_read_bytes": Double(prompt.readBytes - before.readBytes),
                    "decode_expert_read_bytes": Double(end.readBytes - prompt.readBytes),
                    "decode_prefetch_read_bytes": Double(end.prefetchBytes - prompt.prefetchBytes),
                    "decode_prefetch_wasted_bytes": Double(end.prefetchWastedBytes - prompt.prefetchWastedBytes),
                    "decode_prefetch_used": Double(end.prefetchUsed - prompt.prefetchUsed),
                    "decode_prefetch_cancelled": Double(end.prefetchCancelled - prompt.prefetchCancelled),
                    "decode_prefetch_rejected_confidence": Double(end.guessRejectedConfidence - prompt.guessRejectedConfidence),
                    "decode_prefetch_rejected_queue": Double(end.guessRejectedQueue - prompt.guessRejectedQueue),
                    "decode_prefetch_rejected_deadline": Double(end.guessRejectedDeadline - prompt.guessRejectedDeadline),
                    "decode_prefetch_rejected_pressure": Double(end.guessRejectedPressure - prompt.guessRejectedPressure),
                ]
                statistics.phaseCounters.merge(prompt.reads(since: before, phase: "prefill")) { $1 }
                if let first = beforeRefusals, let middle = promptRefusals, let last = runner.experts.ledger?.optionalRefusals {
                    for (phase, from, to) in [("prefill", first, middle), ("decode", middle, last)] {
                        statistics.phaseCounters.merge([
                            phase + ".memory_refused_live": Double(to.live - from.live),
                            phase + ".memory_refused_live_seconds": to.liveSeconds - from.liveSeconds,
                            phase + ".memory_refused_below_target": Double(to.belowTarget - from.belowTarget),
                            phase + ".memory_refused_below_target_seconds": to.belowTargetSeconds - from.belowTargetSeconds,
                        ]) { $1 }
                    }
                }
                statistics.phaseCounters.merge(end.reads(since: prompt, phase: "decode")) { $1 }
                // Existing demand timestamps, summed after reads finish; no added sampling or synchronization.
                // Concurrent record seconds and per-source batch spans must not be added as exclusive phases.
                let initialDemand = before.demandTiming
                let promptDemand = prompt.demandTiming
                for (key, value) in end.demandTiming {
                    statistics.phaseCounters["prefill." + key] = (promptDemand[key] ?? 0) - (initialDemand[key] ?? 0)
                    statistics.phaseCounters["decode." + key] = value - (promptDemand[key] ?? 0)
                }
                // The decode overlap counters (A, C, D, E): sums since the prompt ended, and the split rule's state now.
                let promptOverlap = prompt.overlapTiming
                for (key, value) in end.overlapTiming { statistics.phaseCounters["decode." + key] = value - (promptOverlap[key] ?? 0) }
                statistics.phaseCounters["decode.router_to_demand_max_seconds"] = end.routerToDemandMaximum
                let promptActivity = afterPrefillActivity ?? [:]
                for (key, value) in DriveActivity.notes(runner.experts.driveActivity.snapshot) {
                    statistics.phaseCounters["decode." + key] = value - (promptActivity[key] ?? 0)
                }
                for (key, value) in runner.experts.splitPlannerNotes { statistics.phaseCounters["decode." + key] = value }
            }
            if let runner = model as? FamilyRunner {
                let delayed = runner.delayedExpertCounters
                if delayed.deferredExperts > 0 || delayed.capacityRefusals > 0 {
                    statistics.phaseCounters.merge([
                        "decode.delayed_experts": Double(delayed.deferredExperts), "decode.delayed_mass": delayed.deferredMass,
                        "decode.delayed_deadline_joins": Double(delayed.deadlineJoins),
                        "decode.delayed_capacity_refusals": Double(delayed.capacityRefusals),
                        "decode.delayed_applied_contributions": Double(delayed.appliedContributions),
                        "decode.delayed_early_read_submissions": Double(delayed.earlyReadSubmissions),
                        "decode.delayed_early_submissions_with_demand": Double(delayed.earlySubmissionsWithDemand),
                    ]) { $1 }
                }
            }
            statistics.phaseCounters.merge(hostTiming.counters) { $1 }
            cancellation.report {
                $0.stageSeconds = statistics.stageSeconds
                $0.phaseCounters = statistics.phaseCounters
            }
            completed = true
            return statistics
        }
    }
}
