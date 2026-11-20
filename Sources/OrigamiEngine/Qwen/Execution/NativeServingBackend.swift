import CryptoKit
import Foundation

/// One serial executor behind the transport's prepare/run contract.
public final class NativeServingBackend: ServingBackend, @unchecked Sendable {
    private let gate = ServingGate()
    private var lock: NSLock { gate.lock }
    private let driver: GenerationDriver
    private var admission = ServingAdmission()
    private let version: String
    private let computeTile: Int
    private let readScope: Int
    /// The applied prompt group and the allocation's ceiling; the compute tile is unchanged.
    public let prefillReadScope: Int
    public var prefillReadScopeMaximum: Int { readScope }
    public var tunesPrefillScope: Bool { computeTile == 256 }
    private var nextOrdinal: UInt64 = 0
    private var startupPreparation: [String: Any] = [:]

    /// Called once while loading, before the backend is published. Fixed caches
    /// enforce admission; the governor can inhibit optional growth during this loop.
    /// No dummy forward, prompt state or full expert-pool allocation is required.
    package func prepareReadiness() throws {
        lock.lock()
        defer { lock.unlock() }
        guard admission.isIdle, startupPreparation.isEmpty else { return }
        let started = ProcessInfo.processInfo.systemUptime
        var layers = 0
        var reads = 0
        var reason = "complete"
        defer {
            startupPreparation = [
                "seconds": ProcessInfo.processInfo.systemUptime - started,
                "fixed_layers": layers, "tensor_reads": reads,
                "complete": layers == ModelContract.layers, "reason": reason,
            ]
        }
        for layer in 0..<ModelContract.layers {
            guard driver.executor.ledger.optionalGrowthAllowed,
                MemoryPressureLevel.current().map({ $0 == .normal }) ?? true
            else {
                reason = "memory_pressure"
                return
            }
            do {
                let result = try driver.executor.weights.prefetchWholeLayer(layer)
                reads += result.reads
                guard result.complete else {
                    reason = "resident_capacity"
                    return
                }
                layers += 1
            } catch let error as EngineError where error.code == "insufficient_memory" {
                reason = "memory_admission"
                return
            }
        }
    }

    /// Loading time is kept separate from first-request work and remains visible.
    public var readinessPreparation: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        return startupPreparation
    }

    public init(driver: GenerationDriver, version: String, computeTile: Int = 256, readScope: Int = 256, prefillReadScope: Int? = nil) {
        self.driver = driver
        self.version = version
        self.computeTile = computeTile
        self.readScope = readScope
        self.prefillReadScope = computeTile == 256 ? min(readScope, max(256, (prefillReadScope ?? readScope) / 256 * 256)) : readScope
        driver.tokenizer.admitPromptEncodingCache(ledger: driver.executor.ledger)
    }
    /// True when a pack store was configured, so its absence is worth reporting.
    public var expectsPack = false
    /// One line per request about the prompt cache.
    public var log: (@Sendable (String) -> Void)?
    /// The longest context a request may use: the user's `context`, which may be below its tier's geometry.
    public var servedContext: Int?
    /// The checkpoint, when it has a vision tower: prompts with images need it.
    public var visionModel: ModelSource?
    /// Where the routing counts are kept; saved at most once a minute, after a request.
    public var routingFile: URL?
    private var routingSaved = Date.distantPast  // under lock
    /// Saves the routing counts now (at shutdown, after the last request).
    public func flushRouting() {
        lock.lock()
        routingSaved = .distantPast
        lock.unlock()
        saveRouting()
    }
    /// Request thread or shutdown; under the backend lock so two saves never interleave.
    fileprivate func saveRouting() {
        lock.lock()
        defer { lock.unlock() }
        guard let routingFile, let profile = driver.executor.routingProfile, Date().timeIntervalSince(routingSaved) >= 60 else { return }
        routingSaved = Date()
        try? profile.save(to: routingFile)
    }

    /// The hot copy's keeper, handed the routing counts after each reply (`hotCopyCounts`); it builds while no request
    /// runs and attaches the copy with `attachHotCopy`.
    package var hotCopyKeeper: HotCopyPlaces?
    /// The routing counts a hot copy is ranked from (none until `RoutingProfile.minimumTokens`).
    /// The prompt cache (nil without one), for the request rows' restore account (`PromptCache.takeRestore`).
    package var promptCache: PromptCache? { driver.prefixReuse?.cache }
    package var hotCopyCounts: (uses: [Int: Int], misses: [Int: Int]) { driver.executor.routingProfile?.hotCopyCounts ?? ([:], [:]) }
    /// The tokens those counts cover (`RoutingProfile.tokens`).
    package var routingTokens: Int { driver.executor.routingProfile?.tokens ?? 0 }
    /// No request holds the engine nor waits for it (`ServingGate`); the startup preparation runs under its lock too.
    package var hotCopyIdle: Bool { gate.idle { admission.isIdle } }
    /// One record of a hot copy build, or its publication, while no request runs or arrives (`prepare` waits on `lock`).
    package func withIdleHotCopy(_ work: () throws -> Void) rethrows -> Bool { try gate.whileIdle({ admission.isIdle }, work) }
    /// Expert reads use `copy` (nil: none) from the next request on; false, with nothing changed, while one runs. Reads
    /// already started keep the copy they took (`PackStore.attachHotCopy`).
    package func attachHotCopy(_ copy: HotCopy?) -> Bool { attachHotCopies(copy.map { [$0] } ?? []) }
    /// The same with a copy on each of several drives, fastest first (`HotCopyPlaces`).
    package func attachHotCopies(_ copies: [HotCopy]) -> Bool {
        gate.whileIdle({ admission.isIdle }) { driver.executor.weights.store.attachHotCopies(copies) }
    }

    public func status() -> ServingStatus {
        var status = ServingStatus(
            version: version, contextTier: min(servedContext ?? .max, driver.executor.contextLimit), ready: driver.executor.isHealthy)
        if expectsPack && !driver.executor.packedSourceAvailable {
            status.degraded = "pack store unavailable; reading the model checkpoint directly (slower)"
        }
        status.fastOptIn = driver.executor.fastAllowed
        status.vision = visionModel != nil
        return status
    }

    /// `origami save`: the chat's system prompt gets a snapshot (prefilled once if there is none yet),
    /// pinned so it is never evicted, and named. Call while holding the server's generation slot.
    public func saveSystemPrompt(_ request: ServingRequest, name: String?, cancellation: GenerationControl) throws -> String {
        if let name, !ChatCommand.isName(name) { throw ServingFailure("The saved name " + ChatCommand.nameRule) }
        var probe = request
        probe.messages =
            request.messages.filter { $0["role"] == .string("system") || $0["role"] == .string("developer") }
            + [["role": .string("user"), "content": .string("Hi")]]
        // The probe is not the chat: an X-Origami-Keep on the command must not keep it.
        probe.maximumTokens = 1
        probe.command = nil
        probe.noStore = false
        probe.stream = false
        probe.quality = .exact
        probe.teacherTokens = nil
        probe.keep = nil
        let prompt: [Int]
        do { prompt = try ServingPrompt.encode(probe, tokenizer: driver.tokenizer) } catch { throw ServingFailure.promptEncoding(error) }
        guard let end = driver.tokenizer.systemBlockEnd(prompt) else { throw ServingFailure("This chat has no system prompt to save") }
        let offset = end / 256 * 256
        guard offset >= GenerationDriver.minimumSavedTokens else {
            throw ServingFailure("The system prompt is too short to be worth saving (\(end) tokens; it prefills in moments)")
        }
        let label =
            name
            ?? String(
                SHA256.hash(data: Data(prompt.prefix(offset).map(String.init).joined(separator: ",").utf8))
                    .map { String(format: "%02x", $0) }.joined().prefix(8))
        let started = Date()
        var prefilled = false
        try driver.prefixReuse?.finishSaves(cancellation: cancellation)
        if !(driver.prefixReuse?.hasSaved(prompt.prefix(offset)) ?? false) {
            _ = try prepare(probe, cancellation: cancellation).run { _ in true }
            try driver.prefixReuse?.finishSaves(cancellation: cancellation)
            prefilled = true
        }
        guard driver.prefixReuse?.pin(prompt.prefix(offset), name: label) == true else {
            throw ServingFailure(
                "Could not save the system prompt: a longer saved state already covered it; try again in a new chat", status: 409)
        }
        return
            "saved the system prompt as **\(label)** (\(offset) tokens\(prefilled ? String(format: ", prefilled in %.0f s", Date().timeIntervalSince(started)) : ", already cached")). New chats with it start from there."
    }

    /// The model's tokenizer (the dashboard's view of a saved prompt decodes its tokens).
    public var tokenizer: TokenizerStack { driver.tokenizer }

    /// Waits for prompt-cache saves still being written (cache commands, shutdown). Any thread.
    public func drainSaves(cancellation: GenerationControl? = nil) throws { try driver.prefixReuse?.drainSaves(cancellation: cancellation) }

    /// Memory and expert-cache counters for /v1/status. Any thread.
    public var memorySnapshot: MemoryLedger.Snapshot { driver.executor.ledger.snapshot() }
    public var expertCacheSnapshot: ExpertCache.Snapshot? { driver.executor.expertCacheSnapshot }
    /// The model files' slow reading (`SlowDriveReads`), for the Experts card's model row.
    public var modelDriveSlowReads: [String: Any] { modelDriveReads.snapshot().report }
    /// `split_reads` in /v1/status (`SplitReadCounters`), nil without streamed experts. Any thread.
    public var splitReads: [String: Any]? { driver.executor.splitReads?.report }
    package var modelDriveReads: SlowDriveReads { driver.executor.weights.store.modelDriveReads }
    /// Expert reads since start from the pack's drive, the model's drive and the hot copy: count and seconds each.
    package var expertReadTimes: [(count: UInt64, seconds: Double, buckets: [UInt64])]? { driver.executor.decodeTiming?.reads }
    /// The copies reads come from now (`HotCopySet.laneRates`).
    package var hotCopySet: HotCopySet? { driver.executor.weights.store.hotCopySet }
    /// The online source estimator's counters (two drives), nil without one. Any thread.
    public var sourceOnline: ExpertSourceOnlineSnapshot? { driver.executor.expertExecutionSnapshot?.expertSourceOnline }
    /// Expert records read so far from the hot copy, from each pack or copy (by folder) and from the model's own files.
    public var storeReads: (hot: Int, copies: [String: Int], model: Int) {
        let status = driver.executor.weights.store.status()
        return (
            Int(status.hotCopyReads),
            Dictionary((status.copies ?? []).map { ($0.folder, Int($0.reads)) }, uniquingKeysWith: +),
            Int(status.originalExpertBytes / UInt64(max(1, PackStore.expertBytes)))
        )
    }

    /// Releases expert-cache memory down to the ledger's budget (memory governor). Any thread.
    public func trimToMemoryBudget() -> UInt64 { driver.executor.trimToMemoryBudget() }
    public func growToMemoryBudget() { driver.executor.growToMemoryBudget() }

    public func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        try gate.arrive { try admit(request, cancellation: cancellation) }
    }
    private func admit(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        var request = request
        // The transport hands out one slot at a time; a cancelled predecessor may still be unwinding.
        let deadline = Date(timeIntervalSinceNow: 30)
        // The hot copy's keeper copies a record under the lock; a drive that stopped answering would hold every request
        // here (ROADMAP R2). Arriving, this request lets the keeper take no new turn.
        guard lock.lock(before: Date(timeIntervalSinceNow: ExpertReadStall.floor)) else {
            throw EngineError(
                String(format: "The model drive stopped answering: the hot copy's reads did not return in %.0f s", ExpertReadStall.floor),
                code: "drive_stalled")
        }
        defer { lock.unlock() }
        while !admission.isIdle {
            lock.unlock()
            defer { lock.lock() }
            try cancellation.check()
            guard Date() < deadline else { throw ServingFailure("Engine is busy", status: 503, type: "engine_busy") }
            Thread.sleep(forTimeInterval: 0.01)
        }
        guard request.quality.isExact || driver.executor.fastAllowed else {
            throw ServingFailure(FastLateMissSkip.notAllowed, status: 400, code: "fast_not_allowed")
        }
        do { _ = try request.quality.validated() } catch { throw ServingFailure(String(describing: error)) }
        if let teacher = request.teacherTokens, teacher.contains(where: { $0 >= ModelContract.geometry["vocab_size"]! }) {
            throw ServingFailure("origami_teacher_tokens holds an ID outside the vocabulary")
        }
        try cancellation.check()
        var prompt: [Int]
        var urls: [String] = []
        let encodingStarted = ProcessInfo.processInfo.systemUptime
        do { prompt = try ServingPrompt.encode(request, tokenizer: driver.tokenizer, images: &urls) } catch {
            throw ServingFailure.promptEncoding(error)
        }
        let encodingEnded = ProcessInfo.processInfo.systemUptime
        // CPU-only prompt preparation can overlap the preceding immutable save.
        // Drain before touching live model/cache state, keeping retained readers
        // within the existing conversation quota and avoiding competing SSD reads.
        try driver.prefixReuse?.finishSaves(cancellation: cancellation)
        let savesEnded = ProcessInfo.processInfo.systemUptime
        var images: PromptImages?
        var decoded: [PromptImage] = []
        if !urls.isEmpty {
            guard visionModel != nil, let pad = driver.tokenizer.imagePad else {
                throw ServingFailure("This model reads text only: its checkpoint has no vision tower")
            }
            do {
                // A link may take up to a minute to fetch; a client that left stops the rest. The files wait in memory
                // until prefill knows which ones it reads: bounded like 32 of the largest links.
                var bytes = 0
                decoded = try urls.map {
                    try cancellation.check()
                    let image = try PromptImage(url: $0)
                    bytes += image.data.count
                    guard bytes <= PromptImage.maximumDecoded * PromptImage.maximumLinkBytes else {
                        throw ServingFailure(
                            "The images of this request take more than \(PromptImage.maximumDecoded * PromptImage.maximumLinkBytes / 1_048_576) MB",
                            code: "too_many_images")
                    }
                    return image
                }
                let expanded = try PromptImage.expand(prompt, pad: pad, images: decoded)
                prompt = expanded.prompt
                images = try PromptImages(map: expanded.map, digests: decoded.map(\.digest))
            } catch { throw ServingFailure.promptEncoding(error) }
        }
        request.maximumTokens = try request.outputBudget(promptTokens: prompt.count)
        try cancellation.check()
        let state: SequenceState
        let stateStarted = ProcessInfo.processInfo.systemUptime
        do {
            state = try driver.prepareState(
                prompt: prompt, maximumOutput: request.maximumTokens, quality: request.quality, images: images?.keys ?? [],
                isolatedPrefill: request.tune?.qwenPrefillScope != nil || request.teacherTokens != nil)
        } catch { throw ServingFailure.from(error) }
        state.images = images
        let stages = [
            "prompt_encoding": encodingEnded - encodingStarted, "prior_save_wait": savesEnded - encodingEnded,
            "state_prepare": ProcessInfo.processInfo.systemUptime - stateStarted,
        ]
        let token = try admission.begin()
        nextOrdinal &+= 1
        return Prepared(
            owner: self, driver: driver, state: state, prompt: prompt, request: request, cancellation: cancellation, ordinal: nextOrdinal,
            admission: token, decoded: decoded, preparationSeconds: stages)
    }

    /// Runs the vision tower on the images whose tokens prefill still reads from `offset` (those inside restored
    /// state need no encode). The tower (about 0.9 GB) and its working memory are borrowed from the expert cache
    /// for the encode and handed back after it.
    fileprivate func encodeImages(_ decoded: [PromptImage], into state: SequenceState, from offset: Int) throws {
        guard let model = visionModel, let images = state.images else { return }
        let needed = images.unencoded(from: offset)
        guard !needed.isEmpty else {
            log?("images: \(decoded.count) restored with the prompt, none encoded")
            return
        }
        try PromptImage.checkDecoding(needed.count, of: decoded.count)
        let ledger = driver.executor.ledger
        let working = needed.map { VisionEncoder.workingBytes(patches: decoded[$0].rows * decoded[$0].columns) }.max()!
        let need = VisionEncoder.weightBytes(model) + working
        let budget = ledger.usableBytes
        // Room is needed in the ledger; automatic mode also reserves free-memory headroom.
        let reserved = ledger.reservedBytes
        let available = availableMemoryBytes()
        let shortfall = max(
            reserved + need > ledger.maximumUsableBytes ? reserved + need - ledger.maximumUsableBytes : 0,
            ledger.usesAvailableMemoryFloor && available < ledger.availableFloorBytes + need
                ? ledger.availableFloorBytes + need - available : 0,
            ledger.physicalAllocationShortfall(bytes: need))
        var lowered: UInt64?
        var released: UInt64 = 0
        if shortfall > 0 {
            lowered = min(budget, reserved > shortfall ? reserved - shortfall : 1)
            ledger.setUsable(lowered!)
            released = driver.executor.trimToMemoryBudget()
        }
        // The governor may have moved the budget meanwhile; only an untouched lowering is undone.
        defer { if let lowered, ledger.usableBytes == lowered { ledger.setUsable(budget) } }
        do {
            // As the family scratch admission (R22): only after a proved cache release may the reclamation wait run.
            if released > 0, ledger.physicalAllocationShortfall(bytes: need) > 0 {
                try ReclamationWait.wait(shouldContinue: driver.executor.shouldContinue) {
                    ledger.physicalAllocationShortfall(bytes: need) == 0
                }
            }
            guard ledger.physicalAllocationShortfall(bytes: need) == 0 else {
                throw EngineError("Image encoding would exceed memory_limit", code: "insufficient_memory")
            }
            let reservation = try ledger.reserve(bytes: working, category: "vision-working")
            defer { try? ledger.release(reservation) }
            let started = Date()
            let encoder = try VisionEncoder(model: model, ledger: ledger)
            var tokens = 0
            for index in needed {
                let image = decoded[index]
                let features = try encoder.encode(patches: image.patches(), rows: image.rows, columns: image.columns)
                try state.images?.setFeatures(features, image: index)
                tokens += features.count / VisionEncoder.output
            }
            log?(
                String(
                    format: "images: %d of %d (%d tokens) encoded in %.1f s", needed.count, decoded.count, tokens,
                    Date().timeIntervalSince(started)))
        } catch { throw ServingFailure.from(error) }
    }

    private func retire(_ state: SequenceState, admission token: UUID) {
        lock.lock()
        defer { lock.unlock() }
        admission.retire(token) {
            driver.executor.finish(state)
            driver.executor.shouldContinue = { true }
        }
    }

    private final class Prepared: PreparedGeneration, @unchecked Sendable {
        private let owner: NativeServingBackend, driver: GenerationDriver
        /// A retry after a GPU interruption prepares a new one (the failed attempt abandoned its own).
        private var state: SequenceState
        private let prompt: [Int], request: ServingRequest, cancellation: GenerationControl
        private let lock = NSLock()
        private var consumed = false
        private let ordinal: UInt64
        private let admission: UUID
        /// The prompt's images, decoded; encoded inside `generate` once restored state is known.
        private let decoded: [PromptImage]
        private let preparationSeconds: [String: Double]
        init(
            owner: NativeServingBackend, driver: GenerationDriver, state: SequenceState, prompt: [Int], request: ServingRequest,
            cancellation: GenerationControl, ordinal: UInt64, admission: UUID,
            decoded: [PromptImage], preparationSeconds: [String: Double]
        ) {
            self.owner = owner
            self.driver = driver
            self.state = state
            self.prompt = prompt
            self.request = request
            self.cancellation = cancellation
            self.ordinal = ordinal
            self.admission = admission
            self.decoded = decoded
            self.preparationSeconds = preparationSeconds
        }
        deinit { owner.retire(state, admission: admission) }
        func run(emit: @escaping @Sendable (String) throws -> Bool) throws -> GenerationStatistics {
            lock.lock()
            guard !consumed else {
                lock.unlock()
                throw EngineError("Prepared request already consumed")
            }
            consumed = true
            lock.unlock()
            defer { owner.retire(state, admission: admission) }
            try cancellation.check()
            driver.executor.shouldContinue = { [cancellation] in !cancellation.isCancelled }
            // One seed for every attempt, so a retried reply samples as the first did.
            let retry = GPURetry(log: owner.log)
            let seed = request.seed ?? UInt64.random(in: .min ... .max)
            let emit = retry.passing(emit)
            let result = Result {
                var stats = try retry.run {
                    try cancellation.withAttempt(
                        number: retry.attempt + 1, replayedTokens: retry.replay.count,
                        preparationSeconds: preparationSeconds
                    ) {
                        if retry.attempt > 0 {
                            let images = state.images
                            state = try driver.prepareState(
                                prompt: prompt, maximumOutput: request.maximumTokens, quality: request.quality, images: images?.keys ?? [],
                                isolatedPrefill: request.tune?.qwenPrefillScope != nil || request.teacherTokens != nil)
                            state.images = images
                        }
                        return try attempt(retry, seed: seed, emit: emit)
                    }
                }
                if retry.attempt > 0 { stats.gpuRetries = retry.attempt }
                return stats
            }
            return try result.get()
        }
        private func attempt(_ retry: GPURetry, seed: UInt64, emit: @escaping @Sendable (String) throws -> Bool) throws
            -> GenerationStatistics
        {
            let stops = try StopSequenceFilter(
                stops: request.stopSequences, reasoningOpen: !["none", "minimal"].contains(request.reasoningEffort))
            var consumerStopped = false
            let thinking = !["none", "minimal"].contains(request.reasoningEffort)
            let closing = driver.tokenizer.thinkClosing
            let closer =
                thinking && request.teacherTokens == nil && !closing.isEmpty
                ? ThinkCloser(
                    open: true, maximumTokens: request.maximumTokens,
                    budget: request.thinkingBudget, end: driver.tokenizer.thinkEnd!, closing: closing) : nil
            let constraint = request.decodeGrammar.map {
                JSONConstraint(
                    grammar: $0, tokenizer: driver.tokenizer, vocabularySize: ModelContract.geometry["vocab_size"]!,
                    stops: TokenizerStack.eos.sorted(),
                    thinking: thinking, toolOpener: request.toolOpener(.qwen))
            }
            cancellation.report {
                $0.phase = "prefill"
                $0.promptTokens = prompt.count
                $0.restoredTokens = nil
            }
            cancellation.reportGeneratedTokens(0)
            let attemptCounters = RequestPhaseCounters(executor: driver.executor)
            let attemptStarted = ProcessInfo.processInfo.systemUptime
            var observedStages = preparationSeconds
            var observedPhase = "restore_prefill_and_save"
            var observedStart = attemptStarted
            var fullyReported = false
            defer {
                if !fullyReported {
                    observedStages[observedPhase] = max(0, ProcessInfo.processInfo.systemUptime - observedStart)
                    let counters = RequestPhaseCounters(executor: driver.executor).delta(from: attemptCounters, phase: "partial_attempt")
                    cancellation.report {
                        $0.stageSeconds = observedStages
                        $0.phaseCounters = counters
                    }
                }
            }
            let result = try driver.generate(
                prompt: prompt, maximumOutput: request.maximumTokens,
                seed: seed, parameters: request.sampling,
                computeTile: owner.computeTile,
                readScope: owner.tunesPrefillScope
                    ? min(owner.readScope, request.tune?.qwenPrefillScope ?? owner.prefillReadScope) : owner.readScope,
                quality: request.quality, preparedState: state,
                requestOrdinal: ordinal, persist: !request.noStore, forceTokens: closer.map { closer in { closer.next($0) } },
                teacherTokens: request.teacherTokens, retry: retry, gpuKeepAlive: request.tune?.gpuKeepAlive,
                isolatedPrefill: request.tune?.qwenPrefillScope != nil || request.teacherTokens != nil,
                qwenPairProbe: request.tune?.qwenPairProbe == true, constraint: constraint, deadline: request.deadline,
                prepareImages: decoded.isEmpty
                    ? nil : { [owner, decoded, state] offset in try owner.encodeImages(decoded, into: state, from: offset) },
                onProgress: { [cancellation, prompt, driver] event in
                    if event.event == "prefill_complete", observedPhase != "generation" {
                        let now = ProcessInfo.processInfo.systemUptime
                        observedStages[observedPhase] = max(0, now - observedStart)
                        observedStart = now
                        observedPhase = "generation"
                    }
                    // The first scope starts (or prefill completes) where restored state ended.
                    if event.event == "prefill_scope_started" || event.event == "prefill_complete" {
                        var first = false
                        cancellation.report {
                            if $0.restoredTokens == nil {
                                $0.restoredTokens = event.completedPromptTokens
                                first = true
                            }
                        }
                        if first { driver.prefixReuse?.cache?.explainRestore(prompt, restored: event.completedPromptTokens) }
                        retry.note(from: event.completedPromptTokens)
                    }
                    guard event.phase == "prefill", let tokens = event.scopeTokens else { return }
                    // Within a read scope, layers complete in order: count the scope's share of finished layers.
                    let done = event.completedPromptTokens + tokens * (event.layersComplete ?? 0) / ModelContract.layers
                    cancellation.report { $0.completedPromptTokens = min(max($0.completedPromptTokens, done), prompt.count) }
                },
                onSample: { [cancellation] in cancellation.reportGeneratedTokens($0) },
                onToken: { [cancellation] _, delta in
                    try cancellation.check()
                    let visible = stops.push(delta)
                    if !visible.isEmpty, try !emit(visible) { consumerStopped = true }
                    return !consumerStopped && !stops.didStop
                })
            if !consumerStopped && !stops.didStop {
                let tail = stops.finish()
                if !tail.isEmpty, try !emit(tail) { consumerStopped = true }
            }
            let finish = stops.didStop ? "stop" : (consumerStopped ? "tool_calls" : result.finishReason)
            if let name = request.keep, request.tune?.qwenPrefillScope == nil {
                // The prompt's entry is written on the save queue, so the pin goes there too.
                let reuse = driver.prefixReuse
                let keys = state.images?.keys ?? []
                let prompt = prompt
                let log = owner.log
                let keep: @Sendable () -> Void = {
                    let kept = reuse?.keep(prompt, name: name, images: keys) == true
                    log?(
                        kept
                            ? "kept the session as \(name) (\(prompt.count) tokens)"
                            : "could not keep \(name): its prompt was not saved (under \(GenerationDriver.minimumSavedTokens) tokens or no_store)"
                    )
                }
                if let reuse { reuse.afterSaves(keep) } else { keep() }
            }
            let cached = min(result.conversation?.lastReusedTokens ?? 0, result.promptIDs.count)
            owner.log?(
                "prompt cache: \(cached) of \(result.promptIDs.count) prompt tokens restored"
                    + (result.snapshots.map { "; saving " + $0.joined(separator: ", ") } ?? "") + (request.noStore ? "; no_store" : ""))
            var stats = GenerationStatistics(
                promptTokens: result.promptIDs.count, completionTokens: result.outputIDs.count,
                cachedTokens: min(result.conversation?.lastReusedTokens ?? 0, result.promptIDs.count),
                finishReason: finish, quality: request.quality)
            stats.outputTokens = request.returnTokens ? result.outputIDs : nil
            stats.teacher = result.teacher
            stats.exactFrom = result.exactFrom
            stats.stageSeconds = preparationSeconds.merging(result.stageSeconds) { $1 }
            stats.phaseCounters = result.phaseCounters
            cancellation.report {
                $0.stageSeconds = stats.stageSeconds
                $0.phaseCounters = stats.phaseCounters
            }
            fullyReported = true
            if thinking, let end = driver.tokenizer.thinkEnd {
                stats.reasoningTokens = result.outputIDs.firstIndex(of: end) ?? result.outputIDs.count
                stats.reasoningClosedBy = closer?.closedBy
            }
            stats.stopSequence = stops.matched
            owner.saveRouting()
            owner.hotCopyKeeper?.trigger(counts: owner.hotCopyCounts)
            return stats
        }
    }
}
