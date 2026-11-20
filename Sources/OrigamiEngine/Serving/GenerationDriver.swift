import Foundation

/// One request's outcome: the reply's text and token IDs, its finish reason, the conversation's reuse counters, the
/// quality level that produced it, a teacher-forcing score and the snapshots saved beside the prompt and conversation.
public struct GenerationResult: Sendable {
    public let text: String
    public let promptIDs, outputIDs: [Int]
    public let finishReason: String
    public let conversation: PersistentConversation.Statistics?
    /// The quality dial that produced this response (`origami_mode`).
    public var quality: QualityDial = .exact
    /// Teacher forcing (`teacherTokens`): positions compared, positions whose own greedy pick was the teacher's token,
    /// and the first position that differed (nil: none did).
    public var teacher: TeacherAgreement? = nil
    /// Snapshots saved to the prompt cache besides the prompt and conversation images ("6912 system").
    public var snapshots: [String]? = nil
    /// The first reply token decoded exact because a tool call opened before it (`QualityDial.opensToolCall`).
    public var exactFrom: Int? = nil
    public var stageSeconds: [String: Double] = [:]
    public var phaseCounters: [String: Double] = [:]
}

extension GenerationDriver {
    /// What `text` adds after the already sent `sent`, compared by Unicode scalars, not characters: a token that adds a
    /// combining mark or a variation selector (or "\n" after "\r") extends the character sent last. Nil when `sent` is no
    /// longer a prefix.
    package static func streamDelta(_ text: String, after sent: String) -> String? {
        guard text.unicodeScalars.starts(with: sent.unicodeScalars) else { return nil }
        return String(Substring(text.unicodeScalars.dropFirst(sent.unicodeScalars.count)))
    }
}

/// Teacher-forced agreement of a decode with a reference reply (top-1 agreement of greedy picks).
public struct TeacherAgreement: Codable, Sendable, Equatable {
    public var positions = 0, agreed = 0
    public var firstDifference: Int?
    /// Diagnostic requests only: raw model log-probability at each forced position.
    /// Nil entries expose invalid distributions without emitting non-JSON infinities.
    /// Each arm advances its own recurrent state using the same teacher tokens.
    public var tokenLogProbabilities: [Double?]?
    public init() {}
    package mutating func record(_ own: Int, _ teacher: Int, position: Int, logits: [Float]? = nil) {
        positions += 1
        if own == teacher { agreed += 1 } else if firstDifference == nil { firstDifference = position }
        if let logits {
            if tokenLogProbabilities == nil { tokenLogProbabilities = [] }
            tokenLogProbabilities!.append(Self.logProbability(teacher, logits: logits))
        }
    }
    package static func logProbability(_ token: Int, logits: [Float]) -> Double? {
        guard logits.indices.contains(token), logits[token].isFinite,
            logits.allSatisfy({ $0.isFinite || $0 == -.infinity }),
            let top = logits.max(), top.isFinite
        else { return nil }
        let maximum = Double(top)
        let normalizer = logits.reduce(0.0) { $0 + exp(Double($1) - maximum) }
        guard normalizer > 0, normalizer.isFinite else { return nil }
        let result = Double(logits[token]) - maximum - log(normalizer)
        return result.isFinite ? result : nil
    }
}

/// Runs one request end to end on the SerialExecutor: restore from the prompt cache or the live
/// conversation, prefill by read scopes (capturing frontiers), decode with sampling and streaming
/// (optionally with MTP speculation), then capture and queue the saves. One request at a time.
public final class GenerationDriver {
    public let executor: SerialExecutor
    public let tokenizer: TokenizerStack
    public let prefixReuse: PrefixReuseController?
    public let keepAlive: RequestGPUKeepAlive?
    /// Whether a reply runs the keep-alive when its request does not say (`gpu_keepalive`, the plan's or tuned).
    public var keepAliveByDefault = true
    /// Prompts and conversations shorter than this are not saved to the disk prompt cache.
    public static let minimumSavedTokens = 512

    /// Where to keep system-prompt snapshots for a prefill that starts at `start`: the last frontier
    /// (every 256 tokens from `start`) inside the system block, and inside the part a changed system
    /// block shares with an earlier one. Each needs at least `minimumSavedTokens` and must end before the prompt.
    public static func systemSnapshotOffsets(start: Int, systemEnd: Int, shared: Int, promptTokens: Int) -> [Int] {
        var offsets: [Int] = []
        for limit in [systemEnd, shared] where limit > start {
            let offset = start + (limit - start) / 256 * 256
            if offset > start, offset >= minimumSavedTokens, offset < promptTokens, !offsets.contains(offset) { offsets.append(offset) }
        }
        return offsets
    }
    /// Opt-in MTP speculation (`speculation: mtp`): the checkpoint's multi-token-prediction head drafts one token, a two-row
    /// step verifies it, and a per-request controller keeps speculating only while it measures faster than plain decode.
    package var speculation: MTPHead? {
        didSet { executor.captureHiddenRows = speculation == nil ? 0 : Self.mtpPromptSlots }
    }
    static let mtpPromptSlots = 64
    /// Whether speculating beats plain decode: (1 + a) tokens per pair step plus a plain redo for a rejected draft
    /// (1 - a), against one token per plain step. Nil until both costs and eight attempts are measured.
    package static func speculationPays(plain: Double, pair: Double, acceptance: Double, attempts: Int) -> Bool? {
        guard plain > 0, pair > 0, attempts >= 8 else { return nil }
        return (1 + acceptance) * plain > pair + (1 - acceptance) * plain
    }
    /// Requests that speculated, and their attempts / accepted drafts (status and diagnostics).
    package private(set) var speculationCounts = (requests: 0, attempts: 0, accepted: 0, fallbacks: 0)
    public init(
        executor: SerialExecutor, tokenizer: TokenizerStack, prefixReuse: PrefixReuseController? = nil,
        keepAlive: RequestGPUKeepAlive? = nil
    ) {
        self.executor = executor
        self.tokenizer = tokenizer
        self.prefixReuse = prefixReuse
        self.keepAlive = keepAlive
    }

    /// Whether a request of `quality` reuses only exact timelines: exactness alone, never the configuration's
    /// allowance, since saved images made while fast was allowed share the prompt cache's folder.
    package static func exactTimeline(_ quality: QualityDial) -> Bool { quality.isExact }

    /// `quality` (the request's `origami_mode`): a non-exact level needs the configuration's allowance (refused otherwise).
    /// An exact request never continues from non-exact conversation state.
    /// `images`: the prompt's image keys, which reused state must match as well as the token IDs.
    package func prepareState(
        prompt: [Int], maximumOutput: Int, quality: QualityDial = .exact, images: [ImageKey] = [],
        isolatedPrefill: Bool = false
    ) throws -> SequenceState {
        guard quality.isExact || executor.fastAllowed else { throw EngineError(FastLateMissSkip.notAllowed, code: "fast_not_allowed") }
        if let conversation = prefixReuse?.conversation {
            return try conversation.prepare(
                prompt: prompt, maximumOutput: maximumOutput, executor: executor,
                exactTimeline: Self.exactTimeline(quality), images: images, reuse: !isolatedPrefill)
        }
        return try executor.begin(inputTokens: prompt.count, outputTokens: maximumOutput)
    }

    public func generate(
        prompt: [Int], maximumOutput: Int, seed: UInt64 = 42, parameters: SamplingParameters = .init(),
        computeTile: Int = 256, readScope: Int = 256, quality: QualityDial = .exact, preparedState: SequenceState? = nil,
        requestOrdinal: UInt64 = 0, persist: Bool = true, forceTokens: (([Int]) -> [Int]?)? = nil,
        teacherTokens: [Int]? = nil, retry: GPURetry? = nil, gpuKeepAlive: Bool? = nil,
        isolatedPrefill: Bool = false, qwenPairProbe: Bool = false, constraint: JSONConstraint? = nil, deadline: Date? = nil,
        prepareImages: ((Int) throws -> Void)? = nil,
        onProgress: ((GenerationProgress) -> Void)? = nil,
        onSample: ((Int) -> Void)? = nil,
        onToken: ((Int, String) throws -> Bool)? = nil
    ) throws -> GenerationResult {
        guard !qwenPairProbe || (quality.isExact && speculation == nil && (teacherTokens?.count ?? 0) >= 2 && maximumOutput >= 3) else {
            throw EngineError(
                "Verifier probe requires exact teacher forcing, at least three output slots and no MTP head", code: "invalid_configuration")
        }
        // Per request: this request's dial only; the executor returns to exact afterwards.
        try executor.selectRequestMode(quality)
        timeLimit = deadline
        defer { timeLimit = nil }
        defer {
            try? executor.selectRequestMode(.exact)
            executor.releaseDelayedExpertStorage()
        }
        let exactTimeline = Self.exactTimeline(quality)
        // Saved state is matched on the images' keys too (their tokens do not identify their pixels). Without a
        // conversation nothing is reused or saved. `prepareImages` encodes the images prefill still reads, once
        // restored state is known.
        let images = preparedState?.images
        let imageKeys = images?.keys ?? []
        try executor.setPositionMap(images?.map)
        defer { try? executor.setPositionMap(nil) }
        let conversation = prefixReuse?.conversation
        let state =
            try preparedState
            ?? prepareState(prompt: prompt, maximumOutput: maximumOutput, quality: quality, isolatedPrefill: isolatedPrefill)
        let stores = persist && !isolatedPrefill
        guard state.valid, state.capacity >= prompt.count + maximumOutput,
            !isolatedPrefill || state.tokenCount == 0,
            conversation != nil || (state.tokenCount == 0 && state.capacity == prompt.count + maximumOutput)
        else { throw EngineError("Prepared admission does not match request") }
        var successful = false
        var failureCode: String?
        var progressPhase = "prefill"
        var emittedTokens = 0
        var decodeForwardScopes = 0
        let progress = onProgress.map {
            GenerationProgressRecorder(requestOrdinal: requestOrdinal, promptTokens: prompt.count, publish: $0)
        }
        let previousProgressObserver = executor.progressObserver
        if let progress {
            executor.progressObserver = { offset, count, layers in
                previousProgressObserver?(offset, count, layers)
                // Decode layers are not recorded: nothing reads them, and 48 records per token would fill the recorder.
                guard progressPhase == "prefill" else { return }
                progress.record(
                    "prefill_layer_complete", phase: progressPhase, completedPromptTokens: min(offset, prompt.count),
                    outputTokens: emittedTokens,
                    decodeForwardScopes: decodeForwardScopes, scopeStart: offset, scopeTokens: count, layersComplete: layers)
            }
        }
        defer {
            executor.progressObserver = previousProgressObserver
            if !successful {
                progress?.record(
                    "request_abandoned", phase: progressPhase, completedPromptTokens: min(state.tokenCount, prompt.count),
                    outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes,
                    failure: GenerationFailureObservation(engineCode: failureCode))
            }
            if let conversation { if !successful { conversation.abandon(state, executor: executor) } } else { executor.finish(state) }
        }
        do {
            let setupStarted = ProcessInfo.processInfo.systemUptime
            let captureStarted = conversation?.captureWallSeconds ?? 0
            let countersStarted = RequestPhaseCounters(executor: executor)
            var restoreEnded = setupStarted
            let keepsAlive = gpuKeepAlive ?? keepAliveByDefault
            guard !keepsAlive || !executor.registerQSADecode || keepAlive?.policy != .original else {
                throw EngineError("Register QSA cannot share the original long-poll keepalive", code: "invalid_configuration")
            }
            let keepAliveSession = keepsAlive ? try keepAlive?.begin() : nil
            defer { _ = keepAliveSession?.close() }
            progress?.record("request_started", phase: progressPhase)
            var logits: [Float] = []
            // The prompt-end image is saved to disk after the response, next to the conversation image:
            // a chat follow-up re-renders the prompt but not the generated thinking, so it matches this one.
            var promptImage: ModelCheckpoint?
            var snapshots: [(ModelCheckpoint, String)] = []  // frontiers to save after the request, with their kind
            let schedule = try PrefillReadSchedule(
                promptTokens: prompt.count, computeTile: computeTile, readScope: readScope, mergeTerminalTile: executor.raggedPrefillScope)
            restoreEnded = ProcessInfo.processInfo.systemUptime
            if let conversation {
                try readPrompt(
                    prompt, conversation: conversation, state: state, images: imageKeys, exactTimeline: exactTimeline,
                    isolatedPrefill: isolatedPrefill, stores: stores, computeTile: computeTile, readScope: readScope,
                    progress: progress, progressPhase: progressPhase, prepareImages: prepareImages, logits: &logits,
                    restoreEnded: &restoreEnded, promptImage: &promptImage, snapshots: &snapshots)
            } else {
                try readPrompt(
                    prompt, schedule: schedule, state: state, progress: progress, progressPhase: progressPhase,
                    prepareImages: prepareImages, logits: &logits)
            }
            // Full prefill and complete-prefix hits use the same committed boundary.
            // Grow lazily before sampling; failure follows the normal request cleanup.
            let prefillWorkEnded = ProcessInfo.processInfo.systemUptime
            let promptCaptureSeconds = max(0, (conversation?.captureWallSeconds ?? 0) - captureStarted)
            try executor.prepareDecode(
                state, consumedPromptTokens: prompt.count,
                retainedFrontierBytes: conversation?.retainedFrontierBytes)
            progressPhase = "decode"
            let prefillEnded = ProcessInfo.processInfo.systemUptime
            let countersPrefilled = RequestPhaseCounters(executor: executor)
            var verifierProbe: [String: Double] = [:]
            if qwenPairProbe {
                try conversation?.admitGrowth(tokens: 2)
                verifierProbe = try executor.probeVerifierCost(tokens: Array(teacherTokens!.prefix(2)), state: state)
            }
            progress?.record("prefill_complete", phase: progressPhase, completedPromptTokens: prompt.count)
            var sampler = ReferenceSampler(seed: seed)
            var output = try GeneratedTokenSequence(maximumOutput: maximumOutput)
            var visible = ""
            var finishReason = "length"
            var reply = ReplyText(tokenizer)
            var exactFrom: Int?
            var toolAnchor = ToolCallAnchorTracker(openers: tokenizer.toolCallOpeners, markers: tokenizer.toolCallTextMarkers)
            let savesToolAnchor = stores && quality.isExact && teacherTokens == nil && conversation != nil
            func emitSample(_ token: Int) throws -> Bool {
                try constraint?.accept(token)
                // As in the reference implementation, EOS is sampled but never appended, emitted, penalized or forwarded.
                // The sampler has already consumed exactly its ordinary draw.
                guard try output.recordSample(token) else {
                    finishReason = "stop"
                    return false
                }
                retry?.made(output.ids.count - 1, token)
                emittedTokens = output.ids.count
                onSample?(emittedTokens)
                state.emittedButUnconsumed = token
                var decoded = try reply.text(output.ids)
                // Byte-level BPE may end in incomplete UTF-8. Hold that suffix until
                // more tokens complete it; never publish text that must be revised.
                while decoded.last == "\u{FFFD}" { decoded.removeLast() }
                guard let delta = Self.streamDelta(decoded, after: visible) else { throw EngineError("Tokenizer streaming prefix changed") }
                visible = decoded
                if exactFrom == nil, !quality.isExact, QualityDial.opensToolCall(token, text: visible, openers: tokenizer.toolCallOpeners) {
                    exactFrom = output.ids.count
                    try executor.selectRequestMode(.exact)
                }
                if savesToolAnchor, toolAnchor.observe(token: token, delta: delta),
                    state.tokenCount >= Self.minimumSavedTokens, let conversation
                {
                    // Plain decode is just before the opener's last token. An accepted MTP
                    // draft is already consumed: its frontier must not replay that token.
                    let pending = state.emittedButUnconsumed
                    if state.tokenCount == prompt.count + output.ids.count { state.emittedButUnconsumed = nil }
                    defer { state.emittedButUnconsumed = pending }
                    _ = try prefixReuse!.skippingSaveFailure("tool_anchor") {
                        try executor.reclaimFrontierLoan(retainedFrontierBytes: conversation.toolAnchorAllowance(state))
                        return try conversation.capture(state, produced: output.ids.count, tool: true)
                    }
                }
                progress?.record(
                    "decode_token_sampled", phase: progressPhase, completedPromptTokens: prompt.count,
                    outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes)
                if try onToken?(token, delta) == false {
                    finishReason = "tool_calls"
                    return false
                }
                return true
            }
            var forced: [Int] = []
            // MTP speculation: the head's cache starts from the prompt's last rows; `chosen` is a token already decided by
            // a rejected draft; the controller compares measured costs (seconds) and acceptance.
            // A retry replaying reply tokens decodes them plainly.
            // JSON mode samples plainly: a draft's check would need the mask of the step after it.
            let head =
                quality.isExact && teacherTokens == nil && retry?.replay.isEmpty != false && constraint == nil ? speculation : nil
            var teacher = teacherTokens.map { _ in TeacherAgreement() }
            var hidden = executor.lastHidden.last
            var chosen: Int?
            // The first two reply tokens decode plainly (the first step after a prompt read carries one-off costs), so the
            // controller has a plain step to compare with.
            var plainSeconds = 0.0
            var pairSeconds = 0.0
            var acceptance = 0.8
            var attempts = 0
            var pausedUntil = 3
            var backoff = 32
            if let head {
                head.reset()
                speculationCounts.requests += 1
                let rows = executor.lastHidden
                let first = state.tokenCount - rows.count
                for (offset, row) in rows.dropLast().enumerated() where first + offset + 1 < prompt.count {
                    _ = try head.step(
                        embedding: executor.embeddingRow(prompt[first + offset + 1]), hidden: row,
                        position: first + offset, cacheOnly: true
                    ) { _, _, _ in [] }
                }
            }
            func elapsed(_ since: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - since) / 1e9 }
            while output.ids.count < maximumOutput {
                guard executor.shouldContinue() else { throw IOOperationCancelled() }
                // Tokens the caller writes instead of sampling (closing a think block early).
                if forced.isEmpty, let next = forceTokens?(output.ids) { forced = next }
                var token: Int
                if let decided = chosen, forced.isEmpty {
                    token = decided
                    chosen = nil
                } else if forced.isEmpty {
                    var sampled = try sampler.next(
                        logits: constraint?.masked(logits) ?? logits, parameters: parameters, generated: output.generated)
                    // Teacher forcing: score this step's own pick, then continue from the teacher's token.
                    if let teacherTokens, output.ids.count < teacherTokens.count {
                        teacher?.record(sampled, teacherTokens[output.ids.count], position: output.ids.count, logits: logits)
                        sampled = teacherTokens[output.ids.count]
                    }
                    token = sampled
                } else {
                    token = forced.removeFirst()
                    chosen = nil
                }  // a forced token replaces a decided one too
                if let retry { token = retry.token(output.ids.count, picked: token) }
                guard try emitSample(token) else { break }
                // Speculate when the controller expects it to pay (always for the first attempts, then by measurement,
                // probing again after `backoff` tokens).
                if let head, let last = hidden, forced.isEmpty, output.ids.count + 1 < maximumOutput, output.ids.count >= pausedUntil {
                    let started = DispatchTime.now().uptimeNanoseconds
                    let offset = state.tokenCount
                    let draftHidden = try head.step(
                        embedding: executor.embeddingRow(token), hidden: last, position: offset - 1, cacheOnly: false
                    ) {
                        try executor.mtpExperts($0, head: head, ids: $1, weights: $2)
                    }!
                    let draftLogits = try executor.lmHead(draftHidden)
                    let draft = draftLogits.indices.max { draftLogits[$0] < draftLogits[$1] }!
                    let snapshot = executor.pairSnapshot(state)
                    var verified: (logits: [[Float]], hidden: [[Float]])?
                    // An end-of-turn draft is never forwarded (plain decode never consumes EOS); the plain step decides.
                    if !TokenizerStack.eos.contains(draft) {
                        do {
                            try conversation?.admitGrowth(tokens: 2)
                            verified = try executor.forwardDecodePair([token, draft], state: state)
                        } catch let error as EngineError where error.code == "pair_unsupported" {
                            speculationCounts.fallbacks += 1
                            pausedUntil = maximumOutput
                            FileHandle.standardError.write(Data("origami: speculation off for this reply: \(error)\n".utf8))
                        }
                    }
                    if let verified {
                        attempts += 1
                        speculationCounts.attempts += 1
                        let check = try sampler.verify(
                            draft: draft, logits: verified.logits[0], parameters: parameters, generated: output.generated)  // `token` is already counted
                        if check.accepted {
                            speculationCounts.accepted += 1
                            pairSeconds = attempts == 1 ? elapsed(started) : 0.8 * pairSeconds + 0.2 * elapsed(started)
                            acceptance = 0.9 * acceptance + 0.1
                            decodeForwardScopes += 2  // both tokens consumed
                            if let conversation {
                                let epoch = FastLateMissSkip.timelineEpoch(conversation.identity.arithmeticEpoch, quality: quality)
                                try conversation.committed(
                                    [token, draft],
                                    producing: [
                                        try PrefillReadSchedule.decodeSegment(start: offset, arithmeticEpoch: epoch),
                                        try PrefillReadSchedule.decodeSegment(start: offset + 1, arithmeticEpoch: epoch),
                                    ], state: state, logits: verified.logits[1], replay: false)
                            }
                            let more = try emitSample(draft)
                            state.emittedButUnconsumed = nil
                            _ = try head.step(
                                embedding: executor.embeddingRow(draft), hidden: verified.hidden[0], position: offset, cacheOnly: true
                            ) { _, _, _ in [] }
                            logits = verified.logits[1]
                            hidden = verified.hidden[1]
                            if !more { break }
                            if Self.speculationPays(plain: plainSeconds, pair: pairSeconds, acceptance: acceptance, attempts: attempts)
                                == false
                            {
                                pausedUntil = output.ids.count + backoff
                                backoff = min(512, backoff * 2)
                            }
                            continue
                        }
                        // Rejected: back to before the pair, consume the token alone, emit the corrected token next.
                        try executor.restore(snapshot, into: state)
                        state.emittedButUnconsumed = token
                        acceptance = 0.9 * acceptance
                        chosen = check.token
                        pairSeconds = attempts == 1 ? elapsed(started) : 0.8 * pairSeconds + 0.2 * elapsed(started)
                    }
                    // Without a verify step nothing was measured, and speculation may be off for this reply.
                    switch verified == nil
                        ? nil : Self.speculationPays(plain: plainSeconds, pair: pairSeconds, acceptance: acceptance, attempts: attempts)
                    {
                    case false?:
                        pausedUntil = output.ids.count + backoff
                        backoff = min(512, backoff * 2)
                    case true?: backoff = 32
                    case nil: break
                    }
                }
                // The time limit ends the reply as `maximumOutput` does: this token sent, not yet read.
                if timeLimit.map({ Date() >= $0 }) == true { break }
                if output.ids.count < maximumOutput {
                    let offset = state.tokenCount
                    let plainStart = DispatchTime.now().uptimeNanoseconds
                    try conversation?.admitGrowth(tokens: 1)
                    // Dynamic dial: this step's knobs follow the model's confidence in the token it just produced
                    // (until a tool call opens: exact from there).
                    if quality.confidentAbove != nil, exactFrom == nil { try executor.selectRequestMode(quality.step(after: logits)) }
                    logits = try executor.forward([token], state: state)
                    if head != nil {
                        hidden = executor.lastHidden.last
                        if output.ids.count > 1 {
                            plainSeconds = plainSeconds == 0 ? elapsed(plainStart) : 0.8 * plainSeconds + 0.2 * elapsed(plainStart)
                        }
                    }
                    decodeForwardScopes += 1
                    if let conversation {
                        // Fast rows carry their own epoch, so no exact request continues from them (FastLateMissSkip).
                        let segment = try PrefillReadSchedule.decodeSegment(
                            start: offset,
                            arithmeticEpoch: FastLateMissSkip.timelineEpoch(conversation.identity.arithmeticEpoch, quality: quality))
                        try conversation.committed([token], producing: [segment], state: state, logits: logits, replay: false)
                    }
                    progress?.record(
                        "decode_scope_complete", phase: progressPhase, completedPromptTokens: prompt.count,
                        outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes, scopeStart: offset, scopeTokens: 1)
                }
            }
            if head != nil {
                logSpeculation(
                    hidden: hidden, attempts: attempts, acceptance: acceptance, pairSeconds: pairSeconds,
                    plainSeconds: plainSeconds, pausedUntil: pausedUntil, output: output)
            }
            progressPhase = "finalization"
            let finalizationStarted = ProcessInfo.processInfo.systemUptime
            let countersGenerated = RequestPhaseCounters(executor: executor)
            progress?.record(
                "finalization_started", phase: progressPhase, completedPromptTokens: prompt.count,
                outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes)
            let finalText = try reply.text(output.ids)
            guard let tail = Self.streamDelta(finalText, after: visible) else {
                throw EngineError("Tokenizer final streaming prefix changed")
            }
            if !tail.isEmpty, let last = output.ids.last, try onToken?(last, tail) == false { finishReason = "tool_calls" }
            if !isolatedPrefill, let conversation {
                try saveReply(
                    conversation: conversation, state: state, output: output, snapshots: snapshots, promptImage: promptImage,
                    stores: stores, prompt: prompt, progress: progress, progressPhase: progressPhase, emittedTokens: emittedTokens,
                    decodeForwardScopes: decodeForwardScopes)
            }
            // The reply is complete; a keepalive failure only cost speed.
            if let error = keepAliveSession?.close().error {
                FileHandle.standardError.write(Data("origami: GPU keepalive failed: \(error)\n".utf8))
            }
            var result = GenerationResult(
                text: finalText, promptIDs: prompt, outputIDs: output.ids, finishReason: finishReason,
                conversation: conversation?.statistics())
            result.quality = quality
            result.teacher = teacher
            result.exactFrom = exactFrom
            result.snapshots = snapshots.isEmpty ? nil : snapshots.map { "\($0.0.frontier.consumedTokenCount) \($0.1)" }
            if isolatedPrefill, let conversation {
                conversation.abandon(state, executor: executor)
            } else {
                try conversation?.complete(state, executor: executor)
            }
            result.stageSeconds = [
                "restore_and_setup": restoreEnded - setupStarted,
                "prefill_and_capture": prefillEnded - restoreEnded,
                // Components of the legacy aggregate above, not additional wall time.
                // Execution includes fixed-weight I/O and CPU/GPU work; it is not GPU time.
                "prefill_execution": max(0, prefillWorkEnded - restoreEnded - promptCaptureSeconds),
                "prompt_capture": promptCaptureSeconds,
                "decode_prepare": prefillEnded - prefillWorkEnded,
                "generation": finalizationStarted - prefillEnded,
                "finalization": ProcessInfo.processInfo.systemUptime - finalizationStarted,
            ]
            result.phaseCounters = countersPrefilled.delta(from: countersStarted, phase: "prefill_and_prepare")
                .merging(countersGenerated.delta(from: countersPrefilled, phase: "generation")) { $1 }
            for (key, value) in verifierProbe { result.phaseCounters["verifier_probe." + key] = value }
            if quality.delayedExperts != nil {
                result.phaseCounters["request.delayed_capacity_refusals"] = Double(executor.delayedExpertCounters.capacityRefusals)
            }
            successful = true
            progress?.record(
                "request_complete", phase: "complete", completedPromptTokens: prompt.count,
                outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes)
            return result
        } catch {
            failureCode = (error as? EngineError)?.code
            throw error
        }
    }

    /// `generate`'s prompt read in a conversation: restore saved state, replay a pending output, then read the rest
    /// in scopes, capturing frontiers and keeping the snapshots to save after the request.
    private func readPrompt(
        _ prompt: [Int], conversation: PersistentConversation, state: SequenceState, images imageKeys: [ImageKey],
        exactTimeline: Bool, isolatedPrefill: Bool, stores: Bool, computeTile: Int, readScope: Int,
        progress: GenerationProgressRecorder?, progressPhase: String, prepareImages: ((Int) throws -> Void)?,
        logits: inout [Float], restoreEnded: inout TimeInterval, promptImage: inout ModelCheckpoint?,
        snapshots: inout [(ModelCheckpoint, String)]
    ) throws {
        if !isolatedPrefill {
            try prefixReuse!.restoreConversation(
                requestIDs: prompt.map(Int64.init), images: imageKeys, into: state, exactTimeline: exactTimeline)
        }
        restoreEnded = ProcessInfo.processInfo.systemUptime
        conversation.discardPendingIfAbsent(state, prompt: prompt)
        // A matching pending output is a distinct row-one prefill segment
        // in the new request's recorded timeline; never replay it twice.
        if let pending = state.emittedButUnconsumed {
            try conversation.admitGrowth(tokens: 1)
            let offset = state.tokenCount
            let part = try PrefillReadSchedule(promptTokens: offset + 1, computeTile: 1, readScope: 1, startOffset: offset)
            progress?.record(
                "prefill_scope_started", phase: progressPhase, completedPromptTokens: offset, scopeStart: offset, scopeTokens: 1)
            logits = try executor.forward([pending], state: state)
            try conversation.committed(
                [pending], producing: part.checkpointSegments(arithmeticEpoch: conversation.identity.arithmeticEpoch),
                state: state, logits: logits, replay: true)
            conversation.consumedPending()
            progress?.record(
                "prefill_scope_complete", phase: progressPhase, completedPromptTokens: state.tokenCount, scopeStart: offset,
                scopeTokens: 1)
        }
        // Snapshots: the frontiers at these offsets are captured anyway (every 256 tokens of a scope);
        // keep them and save them after the request. The system block (and the part a changed one
        // shares), and where this turn's reply begins, which the chat's next turn re-renders.
        var snapshotOffsets: [Int: String] = [:]
        prefixReuse?.noteRead(prompt.count - state.tokenCount)
        if stores, state.tokenCount < prompt.count {
            if let systemEnd = tokenizer.systemBlockEnd(prompt) {
                let shared = prefixReuse!.rememberSystemBlock(Array(prompt.prefix(systemEnd)))
                for offset in Self.systemSnapshotOffsets(
                    start: state.tokenCount, systemEnd: systemEnd, shared: shared, promptTokens: prompt.count)
                {
                    snapshotOffsets[offset] = "system"
                }
            }
            if let turn = tokenizer.turnStart(prompt) {
                for offset in Self.systemSnapshotOffsets(
                    start: state.tokenCount, systemEnd: turn, shared: 0, promptTokens: prompt.count)
                where snapshotOffsets[offset] == nil { snapshotOffsets[offset] = "turn" }
            }
            snapshotOffsets = snapshotOffsets.filter { !prefixReuse!.hasSaved(prompt.prefix($0.key), images: imageKeys) }
        }
        try prepareImages?(state.tokenCount)
        if state.tokenCount < prompt.count {
            let suffix = try PrefillReadSchedule(
                promptTokens: prompt.count, computeTile: computeTile, readScope: readScope, startOffset: state.tokenCount,
                mergeTerminalTile: executor.raggedPrefillScope)
            let timeline = try suffix.checkpointSegments(arithmeticEpoch: conversation.identity.arithmeticEpoch)
            for (index, part) in suffix.parts.enumerated() {
                try checkTimeLimit()
                executor.nextPrefillScope = index + 1 < suffix.parts.count ? Array(prompt[suffix.parts[index + 1].range]) : nil
                progress?.record(
                    "prefill_scope_started", phase: progressPhase, completedPromptTokens: state.tokenCount,
                    scopeStart: part.range.lowerBound, scopeTokens: part.range.count)
                try conversation.admitGrowth(tokens: part.range.count)
                let producing = timeline.filter {
                    $0.start >= part.range.lowerBound && $0.start + $0.count <= part.range.upperBound
                }
                // A skipped stage/publication/capture leaves the live state valid; only that frontier is lost.
                let captured =
                    !isolatedPrefill && part.computeTile == 256
                    ? try prefixReuse!.skippingSaveFailure("scope_stage") {
                        try conversation.stageScope(part.range, ids: Array(prompt[part.range]), producing: producing)
                    } : nil
                if let captured {
                    logits = try executor.forwardConversationScope(
                        Array(prompt[part.range]), state: state,
                        returnLogits: part.range.upperBound == prompt.count, capture: captured)
                    let published = try prefixReuse!.skippingSaveFailure("scope_publish") {
                        try conversation.publishScope(captured, state: state)
                    }
                    snapshots += (published ?? []).compactMap { image in
                        snapshotOffsets[image.frontier.consumedTokenCount].map { (image, $0) }
                    }
                } else {
                    logits = try executor.forwardScope(
                        Array(prompt[part.range]), computeTile: part.computeTile,
                        state: state, returnLogits: part.range.upperBound == prompt.count)
                }
                try conversation.committed(
                    Array(prompt[part.range]), producing: producing, state: state,
                    logits: part.range.upperBound == prompt.count ? logits : nil, replay: true)
                let image: ModelCheckpoint? =
                    isolatedPrefill
                    ? nil
                    : try prefixReuse!.skippingSaveFailure("prompt_capture") {
                        try conversation.capture(state, prompt: part.range.upperBound == prompt.count)
                    }
                if part.range.upperBound == prompt.count { promptImage = image }
                if let image, let kind = snapshotOffsets[part.range.upperBound] { snapshots.append((image, kind)) }
                progress?.record(
                    "prefill_scope_complete", phase: progressPhase, completedPromptTokens: state.tokenCount,
                    scopeStart: part.range.lowerBound, scopeTokens: part.range.count)
            }
        } else if logits.isEmpty {
            guard let saved = conversation.logits else { throw EngineError("Persistent complete prompt has no logits") }
            logits = saved
        }
    }

    /// The running request's `origami_time_limit`; past it at a prompt scope's start, the request ends as a cancel does
    /// (the conversation keeps the frontiers read) with `ServingFailure.timeLimit`.
    private var timeLimit: Date?
    private func checkTimeLimit() throws { if timeLimit.map({ Date() >= $0 }) == true { throw ServingFailure.timeLimit() } }

    /// `generate`'s prompt read without a conversation: every scope of `schedule`.
    private func readPrompt(
        _ prompt: [Int], schedule: PrefillReadSchedule, state: SequenceState, progress: GenerationProgressRecorder?,
        progressPhase: String, prepareImages: ((Int) throws -> Void)?, logits: inout [Float]
    ) throws {
        try prepareImages?(0)
        for (index, part) in schedule.parts.enumerated() {
            try checkTimeLimit()
            executor.nextPrefillScope = index + 1 < schedule.parts.count ? Array(prompt[schedule.parts[index + 1].range]) : nil
            progress?.record(
                "prefill_scope_started", phase: progressPhase, completedPromptTokens: state.tokenCount,
                scopeStart: part.range.lowerBound, scopeTokens: part.range.count)
            logits = try executor.forwardScope(
                Array(prompt[part.range]), computeTile: part.computeTile,
                state: state, returnLogits: part.range.upperBound == prompt.count)
            progress?.record(
                "prefill_scope_complete", phase: progressPhase, completedPromptTokens: state.tokenCount,
                scopeStart: part.range.lowerBound, scopeTokens: part.range.count)
        }
    }

    /// `generate`'s summary of a reply's speculation, on standard error.
    private func logSpeculation(
        hidden: [Float]?, attempts: Int, acceptance: Double, pairSeconds: Double, plainSeconds: Double, pausedUntil: Int,
        output: GeneratedTokenSequence
    ) {
        if executor.pairResidentMisses > 0 {
            FileHandle.standardError.write(
                Data(
                    "origami: speculation: \(executor.pairResidentMisses) steps had experts outside memory (host-joined step)\n"
                        .utf8))
            executor.pairResidentMisses = 0
        }
        if hidden == nil { FileHandle.standardError.write(Data("origami: speculation: no hidden row from the prompt read\n".utf8)) }
        FileHandle.standardError.write(
            Data(
                String(
                    format: "origami: speculation: %d drafts, acceptance %.2f, pair step %.0f ms, plain step %.0f ms%@\n",
                    attempts, acceptance, pairSeconds * 1000, plainSeconds * 1000, pausedUntil > output.ids.count ? " (paused)" : ""
                ).utf8))
    }

    /// `generate`'s capture of the finished reply, then its snapshots, conversation and prompt saved.
    private func saveReply(
        conversation: PersistentConversation, state: SequenceState, output: GeneratedTokenSequence,
        snapshots: [(ModelCheckpoint, String)], promptImage: ModelCheckpoint?, stores: Bool, prompt: [Int],
        progress: GenerationProgressRecorder?, progressPhase: String, emittedTokens: Int, decodeForwardScopes: Int
    ) throws {
        try executor.reclaimFrontierLoan()
        let image = try prefixReuse!.skippingSaveFailure("final_capture") {
            try conversation.capture(state, produced: output.ids.count)
        }
        progress?.record(
            "capture_complete", phase: progressPhase, completedPromptTokens: prompt.count,
            outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes)
        // The snapshots, the conversation, then the prompt: with one free slot the prompt's image survives, and
        // `X-Origami-Keep` pins it. Short prompts prefill in moments; saving them would only push useful entries out.
        for (snapshot, kind) in snapshots { try prefixReuse!.persistConversation(snapshot, kind: kind) }
        for saved in [image, promptImage] where stores {
            if let saved, saved.consumedIDs.count >= GenerationDriver.minimumSavedTokens {
                try prefixReuse!.persistConversation(saved)
            }
        }
        progress?.record(
            "publication_complete", phase: progressPhase, completedPromptTokens: prompt.count,
            outputTokens: emittedTokens, decodeForwardScopes: decodeForwardScopes)
    }
}
