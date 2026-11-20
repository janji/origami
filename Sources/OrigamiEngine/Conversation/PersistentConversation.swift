import Foundation

/// One serial conversation owner. Reusing a producing timeline is an explicit
/// numerical epoch, separate from a cold-prefill cache hit. This type never
/// asserts equivalence to another batching schedule from token equality alone.
public final class PersistentConversation {
    /// Reuse, capture and page counters for diagnostics and /v1/status.
    public struct Statistics: Codable, Sendable {
        public internal(set) var liveHits = 0, frontierHits = 0, coldStarts = 0, captures = 0, evictions = 0
        public internal(set) var toolAnchors = 0
        public internal(set) var reusedTokens = 0, replayedTokens = 0, pendingConsumed = 0, pendingDiscarded = 0
        public internal(set) var captureSeconds = 0.0, restoreSeconds = 0.0
        public internal(set) var scopeCaptureSeconds = 0.0
        public internal(set) var scopeDraftsRequested = 0, scopeDraftsAdmitted = 0
        public internal(set) var lastScopeRequestedBytes: UInt64 = 0, lastScopeAdmittedBytes: UInt64 = 0
        public internal(set) var lastScopeAdmissionReason = "unused"
        public internal(set) var maximumFrontiers = 0, frontierSpacingTokens = 0
        public internal(set) var frontierMemoryLimitBytes: UInt64 = 0, frontierMemoryLiveBytes: UInt64 = 0
        public internal(set) var reasons: [String: Int] = [:]
        public internal(set) var lastDecision = "unused", lastReusedTokens = 0
        public internal(set) var retainedFrontiers = 0
        public internal(set) var frontierOffsets: [Int] = []
        public internal(set) var pageLiveBytes: UInt64 = 0, pageHighWaterBytes: UInt64 = 0, pageCopiedBytes: UInt64 = 0
        /// The arena charge held; the rest of `ConversationGeometry.arenaBytes` is lent to the expert cache.
        public internal(set) var pageHeldBytes: UInt64 = 0
        package var frontierFiles: ConversationFrontierStore.Snapshot?
        package var kvFiles: QSAFilePageStore.Snapshot?
        package var kvStaging: QSAFileStaging.Snapshot?
    }
    public let identity: CheckpointIdentity
    package let arena: QSAPageArena
    private let ledger: MemoryLedger
    private let frontierStore: ConversationFrontierStore?
    private let maximumFrontiers: Int
    private let frontiers = ConversationFrontierIndex()
    private weak var protectedPrompt: ModelCheckpoint?
    private weak var toolAnchor: ModelCheckpoint?
    private let frontierSpacing: Int
    private let geometry: ConversationGeometry, frontierMemoryBytes: UInt64
    // Includes escaped readers and unpublished drafts, not merely index entries.
    package var retainedFrontierBytes: UInt64? { frontierStore?.memoryQuota?.liveBytes }
    private var live: SequenceState?
    package private(set) var consumedIDs: [Int64] = []
    /// The images of the request that last extended the timeline: reuse must match them as well as the IDs.
    private var timelineImages: [ImageKey] = []
    package private(set) var segments: [CheckpointSegment] = []
    package private(set) var logits: [Float]?
    private var counters = Statistics()

    /// Serial-owner boundary counter without materializing page/frontier snapshots.
    /// Includes synchronous scope publication and capture retention inside forward.
    package var captureWallSeconds: Double { counters.captureSeconds + counters.scopeCaptureSeconds }

    public convenience init(identity: CheckpointIdentity, ledger: MemoryLedger, maximumBytes: UInt64, contextLimit: Int) throws {
        try self.init(identity: identity, ledger: ledger, maximumBytes: maximumBytes, contextLimit: contextLimit, frontierStore: nil)
    }
    package init(
        identity: CheckpointIdentity, ledger: MemoryLedger, maximumBytes: UInt64, contextLimit: Int,
        frontierStore: ConversationFrontierStore?, fileParent: URL? = nil, fileBacked: Bool? = nil
    ) throws {
        try identity.validate()
        let geometry = try ConversationGeometry(context: contextLimit, fileBacked: fileBacked)
        guard maximumBytes == geometry.conversationBytes,
            !geometry.fileBacked || frontierStore != nil
        else {
            throw EngineError(
                "Persistent conversation requires its declared state and external-frontier owners", code: "invalid_configuration")
        }
        self.geometry = geometry
        frontierMemoryBytes = frontierStore == nil ? maximumBytes - geometry.arenaBytes - 120_000_000 : try geometry.frontierMemoryBytes
        self.identity = identity
        self.ledger = ledger
        self.frontierStore = frontierStore
        frontierSpacing = frontierStore == nil ? contextLimit / 8 : 256
        maximumFrontiers = frontierStore == nil ? 8 : geometry.maximumFrontiers
        // Complete continuation plus a bounded divergent branch and COW growth.
        // Recurrent snapshots are charged separately by their actual owners.
        arena = try geometry.makeArena(ledger: ledger, parent: fileParent)
    }

    public func statistics() -> Statistics {
        var result = counters
        let pages = arena.snapshot()
        result.retainedFrontiers = frontiers.count
        result.pageLiveBytes = pages.liveBytes
        result.maximumFrontiers = maximumFrontiers
        result.frontierSpacingTokens = frontierSpacing
        result.frontierMemoryLimitBytes = frontierMemoryBytes
        result.frontierMemoryLiveBytes = frontierStore?.memoryQuota?.liveBytes ?? frontiers.reservedBytes
        result.frontierOffsets = frontiers.images.map { $0.consumedIDs.count }
        result.pageHighWaterBytes = pages.highWaterBytes
        result.pageCopiedBytes = pages.copiedBytes
        result.pageHeldBytes = pages.heldBytes
        result.frontierFiles = frontierStore?.snapshot()
        result.kvFiles = arena.fileStore?.snapshot()
        result.kvStaging = arena.stagingSnapshot()
        return result
    }
    private func record(_ reason: String) {
        counters.reasons[reason, default: 0] += 1
        counters.lastDecision = reason
    }

    package func adoptedDisk(_ image: ModelCheckpoint, state: SequenceState) throws {
        try state.validate()
        guard image.identity == identity, state.tokenCount == image.consumedIDs.count, live === state else {
            throw EngineError("Persistent disk restore has a different live owner or epoch")
        }
        consumedIDs = image.consumedIDs
        segments = image.segments
        logits = image.nextLogits
        counters.coldStarts -= 1
        counters.lastReusedTokens = state.tokenCount
        counters.reusedTokens += state.tokenCount
        record("verified_disk_timeline")
    }

    /// The input is the complete newly rendered token sequence, never a textual
    /// suffix supplied on trust. New sampling/parser state is owned by the driver.
    /// `exactTimeline` (an exact request): reuse only timelines without fast decode rows
    /// (FastLateMissSkip.isExactTimeline); otherwise the selection is unchanged.
    package func prepare(
        prompt: [Int], maximumOutput: Int, executor: SerialExecutor, exactTimeline: Bool = false,
        images: [ImageKey] = [], reuse: Bool = true
    ) throws -> SequenceState {
        // In memory, fast rows exist only where fast is allowed (fixed at load): every request is exact otherwise, and
        // the one way in from disk (`PrefixReuseController.restoreConversation`) tests every exact request's image.
        // This spares exact requests the per-decode-row rehash on configurations that never run fast.
        let memoryExact = exactTimeline && executor.fastAllowed
        let started = DispatchTime.now().uptimeNanoseconds
        defer { counters.restoreSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 }
        guard !prompt.isEmpty, maximumOutput > 0, prompt.count <= executor.contextLimit - maximumOutput else {
            throw EngineError("Persistent request exceeds admitted context")
        }
        let ids = prompt.map(Int64.init)
        if reuse, let state = live, state.valid, state.tokenCount == consumedIDs.count,
            consumedIDs.count <= ids.count, consumedIDs.elementsEqual(ids.prefix(consumedIDs.count)),
            ImageKey.below(timelineImages, consumedIDs.count) == ImageKey.below(images, consumedIDs.count),
            state.tokenCount < ids.count || logits != nil,
            try !memoryExact || FastLateMissSkip.isExactTimeline(segments, arithmeticEpoch: identity.arithmeticEpoch)
        {
            try state.validate()
            try executor.resume(
                state, upcomingInput: frontierStore == nil ? nil : prompt.count - state.tokenCount,
                growth: prompt.count - state.tokenCount + maximumOutput)
            counters.liveHits += 1
            counters.lastReusedTokens = state.tokenCount
            counters.reusedTokens += state.tokenCount
            timelineImages = images
            record("live_timeline")
            return state
        }
        leaveTimeline(for: ids, images: images)
        // An isolated tuning run cannot borrow a matching prompt's pages; let admission evict its protection if needed.
        if !reuse { protectedPrompt = nil }
        let selected = try reusableFrontier(for: ids, images: images, exactTimeline: memoryExact, reuse: reuse)
        let state = try executor.begin(
            inputTokens: prompt.count, outputTokens: maximumOutput, pageArena: arena,
            upcomingInput: frontierStore == nil ? nil : prompt.count - (selected?.consumedIDs.count ?? 0))
        do {
            if let selected {
                // The saved complete timeline is the authoritative producing
                // schedule. No cold-prefill segmentation guard is removed.
                _ = try selected.restore(
                    into: state, expectedIdentity: identity, expectedSegments: selected.segments,
                    expectedConsumedIDs: Array(ids.prefix(selected.consumedIDs.count)),
                    intent: selected.frontier.emittedButUnconsumed == nil ? .appendInput : .consumePending)
                consumedIDs = selected.consumedIDs
                segments = selected.segments
                logits = selected.nextLogits
                counters.frontierHits += 1
                record("complete_frontier_timeline")
            } else {
                counters.coldStarts += 1
                record("cold_no_compatible_frontier")
            }
            counters.lastReusedTokens = state.tokenCount
            counters.reusedTokens += state.tokenCount
            live = state
            return state
        } catch {
            executor.finish(state)
            throw error
        }
    }

    /// The request continues this conversation from memory, at or after its last prompt: the live timeline or the
    /// last prompt's image is a prefix of it. Such a request needs no saved prompt from disk, and the frontiers its
    /// growth may evict are earlier states of the conversation it saves again at its end.
    package func continues(_ prompt: [Int], images: [ImageKey] = []) -> Bool {
        let ids = prompt.map(Int64.init)
        func matches(_ saved: [Int64], _ keys: [ImageKey]) -> Bool {
            !saved.isEmpty && ids.starts(with: saved) && ImageKey.below(keys, saved.count) == ImageKey.below(images, saved.count)
        }
        if let state = live, state.valid, state.tokenCount == consumedIDs.count, matches(consumedIDs, timelineImages) { return true }
        return protectedPrompt.map { matches($0.consumedIDs, $0.images) } ?? false
    }

    /// Isolated prefill tuning bypasses even a complete matching image, without removing unrelated cached images.
    package func reusableFrontier(for ids: [Int64], images: [ImageKey] = [], exactTimeline: Bool = false, reuse: Bool = true) throws
        -> ModelCheckpoint?
    {
        guard reuse else { return nil }
        let compatible = frontiers.images.filter { image in
            let n = image.consumedIDs.count
            return image.identity == identity && n <= ids.count && image.consumedIDs.elementsEqual(ids.prefix(n))
                && image.images == ImageKey.below(images, n) && (n < ids.count || image.nextLogits != nil)
        }
        var selected = compatible.max { $0.consumedIDs.count < $1.consumedIDs.count }
        if exactTimeline, let longest = selected,
            try !FastLateMissSkip.isExactTimeline(longest.segments, arithmeticEpoch: identity.arithmeticEpoch)
        {
            record("fast_timeline_excluded")
            selected = try compatible.filter {
                try FastLateMissSkip.isExactTimeline($0.segments, arithmeticEpoch: identity.arithmeticEpoch)
            }
            .max { $0.consumedIDs.count < $1.consumedIDs.count }
        }
        return selected
    }

    /// Drops the live timeline for a request that does not continue it. Releasing the sequence retires its active
    /// reservation; immutable checkpoint readers and their arena/Metal ownership remain charged. The previous
    /// prompt's image stays protected only while this request extends it, so another chat's growth may evict it.
    package func leaveTimeline(for ids: [Int64], images: [ImageKey]) {
        live = nil
        logits = nil
        consumedIDs = []
        segments = []
        timelineImages = images
        if let kept = protectedPrompt,
            !(ids.starts(with: kept.consumedIDs) && kept.images == ImageKey.below(images, kept.consumedIDs.count))
        {
            protectedPrompt = nil
        }
    }

    /// Reserve growth before any layer starts. Evict only our unpinned index;
    /// external immutable readers keep pages charged and can force a refusal.
    package func admitGrowth(tokens: Int, retainingOffsets: [Int] = []) throws {
        guard (1...geometry.context).contains(tokens), retainingOffsets.count <= 31,
            retainingOffsets.allSatisfy({ (1...geometry.context).contains($0) })
        else {
            throw EngineError("Invalid conversation growth extent")
        }
        // Each unaligned staged frontier can retain one partial page per QSA
        // vector after the live sequence replaces it. These copies coexist
        // with incoming pages until the complete scope publishes or retires.
        let copies = retainingOffsets.filter { $0 % 256 != 0 }.count
        let pages = UInt64((tokens + 255) / 256 + 1 + copies)
        let growth = pages * 256 * geometry.pooledBytesPerToken
        let fileGrowth = geometry.fileBacked ? pages * 24 * UInt64(QSAFilePageStore.pageBytes) : 0
        func fits() -> Bool {
            guard growth <= arena.maximumBytes - arena.snapshot().liveBytes else { return false }
            guard let file = arena.fileStore?.snapshot() else { return fileGrowth == 0 }
            return fileGrowth <= file.diskQuotaBytes - file.liveBytes
        }
        while !fits(), evictRedundant() {
            record("page_growth_eviction")
        }
        guard fits() else {
            record("page_growth_refused")
            throw EngineError("Retained QSA readers prevent admitted conversation growth", code: "conversation_page_quota")
        }
    }

    package func discardPendingIfAbsent(_ state: SequenceState, prompt: [Int]) {
        guard let pending = state.emittedButUnconsumed else { return }
        if state.tokenCount >= prompt.count || prompt[state.tokenCount] != pending {
            state.emittedButUnconsumed = nil
            counters.pendingDiscarded += 1
        }
    }

    package func committed(_ ids: [Int], producing: [CheckpointSegment], state: SequenceState, logits: [Float]?, replay: Bool) throws {
        try state.validate()
        guard state.tokenCount == consumedIDs.count + ids.count,
            producing.first?.start == consumedIDs.count, producing.last.map({ $0.start + $0.count }) == state.tokenCount
        else {
            throw EngineError("Conversation publication is not a complete producing frontier")
        }
        var end = consumedIDs.count
        for segment in producing {
            try segment.validate()
            guard segment.start == end else { throw EngineError("Conversation timeline has a gap") }
            end += segment.count
        }
        consumedIDs.append(contentsOf: ids.map(Int64.init))
        segments.append(contentsOf: producing)
        self.logits = logits
        if replay { counters.replayedTokens += ids.count }
    }
    package func consumedPending() { counters.pendingConsumed += 1 }

    package static func scopeOffsets(_ range: Range<Int>, spacing: Int) -> [Int] {
        guard range.count >= 512, spacing >= 256 else { return [] }
        // Relative to this actual producing tile sequence; a resumed scope may
        // begin at an unaligned absolute token offset.
        return stride(from: range.lowerBound + spacing, to: range.upperBound, by: spacing).map { $0 }
    }

    package func stageScope(_ range: Range<Int>, ids: [Int], producing: [CheckpointSegment]) throws -> ConversationScopeCapture? {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { counters.scopeCaptureSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 }
        let offsets = Self.scopeOffsets(range, spacing: frontierSpacing)
        guard !offsets.isEmpty else { return nil }
        guard consumedIDs.count == range.lowerBound, ids.count == range.count else {
            throw EngineError("Staged scope has a different consumed prefix")
        }
        let timeline = segments + producing
        let requested = try offsets.map {
            try ConversationScopeCapture.admission(offset: $0, segments: timeline, fileBacked: frontierStore != nil)
        }
        counters.scopeDraftsRequested += requested.count
        counters.lastScopeRequestedBytes = requested.reduce(0) { $0 + $1.quotaBytes }
        counters.lastScopeAdmissionReason = "complete_requested_set"
        // Cheapest first maximizes the admitted count for an unusually long
        // timeline. Prefer the later offset when costs tie, then publish in order.
        let selection = ConversationScopeCapture.select(requested, refusal: { frontierRefusal(bytes: $0, count: $1) }) {
            guard evictRedundant() else { return false }
            record("scope_staging_eviction")
            return true
        }
        counters.lastScopeAdmissionReason = selection.reason
        if selection.drafts.count < requested.count { record("scope_draft_quota_reduction_" + selection.reason) }
        counters.scopeDraftsAdmitted += selection.drafts.count
        counters.lastScopeAdmittedBytes = selection.bytes
        guard !selection.drafts.isEmpty else { return nil }
        try admitGrowth(tokens: range.count, retainingOffsets: selection.drafts.map(\.offset))
        return try ConversationScopeCapture(
            offsets: selection.drafts.map(\.offset).sorted(), scope: range, consumed: consumedIDs + ids.map(Int64.init),
            segments: timeline, identity: identity, ledger: ledger, frontierStore: frontierStore)
    }

    @discardableResult
    package func publishScope(_ capture: ConversationScopeCapture, state: SequenceState) throws -> [ModelCheckpoint] {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { counters.scopeCaptureSeconds += capture.retentionSeconds + Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 }
        let images = try capture.finish(state: state)
        let available = frontierMemoryBytes
        guard frontiers.count + images.count <= maximumFrontiers,
            frontiers.reservedBytes + images.reduce(0, { $0 + $1.reservedBytes }) <= available
        else {
            throw EngineError("Completed intra-scope checkpoints exceed admission")
        }
        for image in images {
            image.images = ImageKey.below(timelineImages, image.consumedIDs.count)
            frontiers.insert(image)
        }
        counters.captures += images.count
        return images
    }

    @discardableResult
    private func evictRedundant() -> Bool {
        // Prefer ordinary edit frontiers; memory pressure may still evict the tool anchor.
        guard
            frontiers.evict(protecting: protectedPrompt, also: toolAnchor)
                || frontiers.evict(protecting: protectedPrompt)
        else { return false }
        counters.evictions += 1
        record("frontier_quota_eviction")
        return true
    }
    private func fitsFrontiers(bytes: UInt64, count: Int, needsDisk: Bool = true) -> Bool {
        frontierRefusal(bytes: bytes, count: count, needsDisk: needsDisk) == nil
    }
    private func frontierRefusal(bytes: UInt64, count: Int, needsDisk: Bool = true) -> String? {
        let live = frontierStore?.memoryQuota?.liveBytes ?? frontiers.reservedBytes
        guard live <= frontierMemoryBytes, bytes <= frontierMemoryBytes - live else { return "memory_quota_with_retained_readers" }
        guard frontiers.count + count <= maximumFrontiers else { return "protected_index_count" }
        guard needsDisk, let store = frontierStore else { return nil }
        return UInt64(count) * ConversationFrontierStore.imageBytes <= store.maximumBytes - store.snapshot().liveBytes
            ? nil : "disk_quota_with_retained_readers"
    }

    private func captureEstimate(_ state: SequenceState, inMemory: Bool) -> UInt64 {
        (inMemory || frontierStore == nil ? state.gdn.values.reduce(UInt64(0)) { $0 + $1.capacityBytes } : 262144)
            + state.qsa.values.reduce(UInt64(0)) { $0 + $1.hostCapacityBytes }
            + UInt64(
                state.ple.convolution.capacity * 2 + state.history.capacity * 8
                    + segments.count * 1024 + consumedIDs.capacity * 8 + (logits?.capacity ?? 0) * 4 + 65_536)
    }

    /// Reclaim only the prospective anchor's share of a decode loan. Older escaped
    /// readers stay counted; a full quota falls back to the existing eviction path.
    package func toolAnchorAllowance(_ state: SequenceState) -> UInt64? {
        guard !geometry.fileBacked, let quota = frontierStore?.memoryQuota else { return nil }
        return min(frontierMemoryBytes, quota.liveBytes + captureEstimate(state, inMemory: true))
    }

    package func capture(_ state: SequenceState, produced: Int = 0, prompt: Bool = false, tool: Bool = false) throws -> ModelCheckpoint {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { counters.captureSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 }
        try state.validate()
        guard state.tokenCount == consumedIDs.count else { throw EngineError("Conversation capture frontier differs") }
        let estimate = captureEstimate(state, inMemory: tool)
        let available = frontierMemoryBytes
        if tool, let previous = toolAnchor { frontiers.remove(previous) }
        // Deduplicate before reserving. Repeated identical requests must not
        // evict unrelated edit frontiers to recreate the same immutable image.
        frontiers.removeDuplicates(ids: consumedIDs, segments: segments)
        if prompt { protectedPrompt = nil }
        while !fitsFrontiers(bytes: estimate, count: 1, needsDisk: !tool), evictRedundant() {}
        guard fitsFrontiers(bytes: estimate, count: 1, needsDisk: !tool) else {
            record("frontier_capture_refused")
            throw EngineError("Conversation frontier exceeds quota", code: "conversation_frontier_quota")
        }
        let image = try ModelCheckpoint.capture(
            state: state, consumedIDs: consumedIDs, identity: identity,
            segments: segments, producedTokenCount: produced, nextLogits: logits, ledger: ledger, frontierStore: frontierStore,
            inMemoryRecurrent: tool)
        guard frontiers.reservedBytes + image.reservedBytes <= available else {
            record("frontier_measured_quota_refused")
            throw EngineError("Measured conversation frontier exceeds admission", code: "conversation_frontier_quota")
        }
        image.images = ImageKey.below(timelineImages, image.consumedIDs.count)
        frontiers.insert(image)
        counters.captures += 1
        if prompt { protectedPrompt = image }
        if tool {
            toolAnchor = image
            counters.toolAnchors += 1
        }
        return image
    }

    package func complete(_ state: SequenceState, executor: SerialExecutor) throws {
        guard live === state, state.valid else { throw EngineError("Conversation completion lost its valid live owner") }
        do { try executor.suspend(state) } catch {
            record("cleanup_failure")
            abandon(state, executor: executor)
            throw error
        }
    }

    package func abandon(_ state: SequenceState, executor: SerialExecutor) {
        guard live === state else {
            executor.finish(state)
            return
        }
        executor.finish(state)
        live = nil
        consumedIDs = []
        segments = []
        logits = nil
        timelineImages = []
        record("live_invalidated")
    }
}
