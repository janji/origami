import Darwin
import Foundation

/// Prefix-reuse and prompt-cache counters for /v1/status; owned by the generation thread.
public struct PrefixReuseStatistics: Codable, Sendable {
    public fileprivate(set) var memoryHits: UInt64 = 0
    public fileprivate(set) var diskHits: UInt64 = 0
    public fileprivate(set) var misses: UInt64 = 0
    public fileprivate(set) var saves: UInt64 = 0
    public fileprivate(set) var diskSaveFailures: UInt64 = 0
    /// Conversation frontier/snapshot captures skipped after a failure (storage.missing_drive original_spans).
    public fileprivate(set) var frontierSaveFailures: UInt64 = 0
    public fileprivate(set) var reasons: [String: UInt64] = [:]
    public fileprivate(set) var lastDecision = "unused"
    public fileprivate(set) var lastDetail: String?
}

/// New-request prefix reuse only, never suspended-request or transport resume. A served configuration carries a
/// `conversation`: requests restore through `restoreConversation` and save through `persistConversation`; without
/// one (only an `ORIGAMI_EXECUTION` override turns it off) nothing is reused or saved. The owner serializes calls at
/// committed whole-model scope boundaries; images live in the conversation's memory and on disk (`PromptCache`), all
/// charged to the shared memory ledger.
/// Unchecked Sendable: the save queue touches only `promptCache` (locked) and `finished` (under `saveLock`).
public final class PrefixReuseController: @unchecked Sendable {
    public let identity: CheckpointIdentity
    public let conversation: PersistentConversation?
    public let memoryBytes: UInt64
    public private(set) var statistics = PrefixReuseStatistics()
    private let ledger: MemoryLedger
    private let promptCache: PromptCache?
    // Asynchronous writes: prompt-cache saves run on one serial queue after the response has ended. The next
    // request, cache commands and shutdown wait for them first (`finishSaves`, `drainSaves`), so a save never
    // overlaps another request's work and the memory plan is unchanged.
    private let saveQueue = DispatchQueue(label: "origami.prompt-cache.saves", qos: .userInitiated)
    private let pendingSaves = DispatchGroup()
    private let saveLock = NSLock()
    private var finished: (saves: UInt64, failures: UInt64) = (0, 0)
    /// The save the queue runs now: when it started and how long it may take (under `saveLock`).
    private var running: (started: DispatchTime, bound: Double)?
    /// A captured image is immutable; the queue only reads it.
    private struct QueuedSave: @unchecked Sendable { let image: ModelCheckpoint, kind: String? }

    /// The last decision's name in `statistics.reasons` / `lastDecision`.
    private enum Reason: String {
        case diskUnavailable, diskVerificationFailed, saveQueued, diskSaveFailed, miss
        case liveConversationHit, diskTimelineMismatch, diskTimelineHit
        case fastTimelineExcluded, frontierSaveSkipped
    }

    public init(
        identity: CheckpointIdentity, ledger: MemoryLedger, memoryBytes: UInt64,
        promptCache: PromptCache? = nil, conversationContext: Int? = nil,
        conversationFileBacked: Bool? = nil,
        frontierDiskBytes: UInt64 = 0, frontierAsyncWrites: Bool = false, frontierWriteStagingSlots: Int = 4
    ) throws {
        try identity.validate()
        self.identity = identity
        self.ledger = ledger
        self.memoryBytes = memoryBytes
        // The 262,144 tier keeps its pages in a file unless it was planned in memory.
        let fileBacked = conversationFileBacked ?? (conversationContext == ConversationGeometry.maximumFileContext)
        if fileBacked, let conversationContext {
            guard promptCache != nil,
                frontierDiskBytes == (try ConversationGeometry(context: conversationContext, fileBacked: true).frontierDiskBytes)
            else {
                throw EngineError("262K conversation requires its complete external quotas", code: "invalid_configuration")
            }
        }
        let frontierStore: ConversationFrontierStore?
        var geometry: ConversationGeometry?
        if frontierDiskBytes > 0 {
            guard let conversationContext else { throw EngineError("Conversation frontiers need a context", code: "invalid_configuration") }
            geometry = try ConversationGeometry(context: conversationContext, fileBacked: fileBacked)
            guard frontierDiskBytes == geometry!.frontierDiskBytes else {
                throw EngineError("Conversation frontier quota differs from its geometry")
            }
            if let promptCache {
                frontierStore = try ConversationFrontierStore(
                    parent: promptCache.frontierParent, maximumBytes: frontierDiskBytes,
                    ledger: ledger, memoryBytes: geometry!.frontierMemoryBytes, asyncWrites: frontierAsyncWrites,
                    stagingSlots: frontierWriteStagingSlots,
                    reserveBytes: PagedCheckpointFormat.layout(capacity: conversationContext, tokens: conversationContext, hasLogits: true)
                        .bytes)
            } else if !geometry!.fileBacked {
                frontierStore = nil
            } else {
                throw EngineError("262K conversations need a prompt cache folder", code: "invalid_configuration")
            }
        } else {
            frontierStore = nil
        }
        let conversation = try conversationContext.map {
            try PersistentConversation(
                identity: identity, ledger: ledger, maximumBytes: memoryBytes,
                contextLimit: $0, frontierStore: frontierStore, fileParent: promptCache?.frontierParent, fileBacked: fileBacked)
        }
        self.conversation = conversation
        self.promptCache = promptCache
    }

    /// A failed frontier/snapshot save (drive lost, volume full, quota) is counted, reported in one
    /// stderr line and skipped; the save never mutated the live state. Cancellation still ends the request.
    package func skippingSaveFailure<T>(_ stage: String, _ save: () throws -> T?) throws -> T? {
        do { return try save() } catch {
            guard !(error is IOOperationCancelled) else { throw error }
            statistics.frontierSaveFailures &+= 1
            record(.frontierSaveSkipped, detail: stage + ": " + String(describing: error))
            Self.report("conversation frontier save skipped (\(stage))", error)
            return nil
        }
    }

    private static func report(_ what: String, _ error: Error) {
        let line = "origami: \(what): \(error)".replacingOccurrences(of: "\n", with: " ") + "\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func record(_ reason: Reason, detail: String? = nil) {
        statistics.reasons[reason.rawValue, default: 0] &+= 1
        statistics.lastDecision = reason.rawValue
        statistics.lastDetail = detail.map { String($0.prefix(512)) }
    }

    /// Separate from strict cold-prefix restoration: restore the complete saved
    /// producing timeline under the explicitly selected conversation epoch.
    /// Header data is only a hint; read() verifies sections, payload, token IDs,
    /// full digest and file identity before any state is published.
    /// `exactTimeline` (an exact request, whether or not fast is allowed): an image holding fast decode rows is a miss.
    package func restoreConversation(requestIDs: [Int64], images: [ImageKey] = [], into target: SequenceState, exactTimeline: Bool = false)
        throws
    {
        guard let conversation else { throw EngineError("Conversation restoration is disabled") }
        if target.tokenCount > 0 {
            statistics.memoryHits += 1
            record(.liveConversationHit)
            return
        }
        guard let promptCache else {
            statistics.misses += 1
            record(.diskUnavailable)
            return
        }
        let candidates = promptCache.candidates(for: requestIDs, images: images)
        guard !candidates.isEmpty else {
            statistics.misses += 1
            record(.miss)
            return
        }
        for (slot, persistent) in candidates {
            let image: ModelCheckpoint
            do {
                let hint = try peek(persistent)
                let n = hint.frontier.consumedTokenCount
                guard hint.version == 2, hint.byteOrder == "little", hint.identity == identity,
                    n > 0, n <= requestIDs.count, n < requestIDs.count || hint.hasLogits
                else {
                    record(.diskTimelineMismatch)
                    continue
                }
                if exactTimeline, try !FastLateMissSkip.isExactTimeline(hint.segments, arithmeticEpoch: identity.arithmeticEpoch) {
                    record(.fastTimelineExcluded)
                    continue
                }
                // No retained frontier is a prefix of this request (prepare found none): they give way to its pages.
                try conversation.admitGrowth(tokens: n)
                image = try persistent.read(
                    expectedIdentity: identity, expectedSegments: hint.segments,
                    expectedConsumedIDs: Array(requestIDs.prefix(n)), pageArena: conversation.arena)
            } catch {
                if Self.isMemoryError(error) || error is IOOperationCancelled { throw error }
                // A damaged or stale entry fails a check (an error without a code) and is dropped so it is not tried
                // again; a full arena, a quota or a drive that cannot be read leaves the entry for a later request.
                guard (error as? EngineError)?.code == nil else {
                    record(.diskUnavailable, detail: String(describing: error))
                    continue
                }
                promptCache.forget(slot)
                record(.diskVerificationFailed, detail: String(describing: error))
                continue
            }
            _ = try image.restore(
                into: target, expectedIdentity: identity, expectedSegments: image.segments,
                expectedConsumedIDs: Array(requestIDs.prefix(image.consumedIDs.count)),
                intent: image.frontier.emittedButUnconsumed == nil ? .appendInput : .consumePending)
            try conversation.adoptedDisk(image, state: target)
            promptCache.touch(slot)
            statistics.diskHits += 1
            record(.diskTimelineHit)
            // The reader's host QSA arrays die here. The target owns arena pages;
            // its active allowance covers borrowed host recurrent/logits arrays.
            return
        }
        statistics.misses += 1
    }

    /// The prompt cache holds an entry for exactly these tokens.
    package func hasSaved(_ ids: ArraySlice<Int>, images: [ImageKey] = []) -> Bool {
        promptCache?.contains(ids.map(Int64.init)[...], images: images) ?? false
    }
    package func pin(_ ids: ArraySlice<Int>, name: String) -> Bool { promptCache?.pin(ids.map(Int64.init)[...], name: name) ?? false }
    package func keep(_ ids: [Int], name: String, images: [ImageKey] = []) -> Bool {
        promptCache?.keep(ids.map(Int64.init)[...], name: name, images: images) ?? false
    }
    /// Counts prompt tokens a request had to read, for the prompt cache's eviction rule.
    package func noteRead(_ tokens: Int) { promptCache?.noteRead(tokens) }
    /// The prompt cache, for the request's restore account (`PromptCache.explainRestore`) and status.
    package var cache: PromptCache? { promptCache }
    /// See `PromptCache.rememberSystemBlock`.
    package func rememberSystemBlock(_ ids: [Int]) -> Int { promptCache?.rememberSystemBlock(ids) ?? 0 }

    package func persistConversation(_ image: ModelCheckpoint, kind: String? = nil) throws {
        guard conversation != nil, image.identity == identity else { throw EngineError("Persistent image epoch differs") }
        guard promptCache != nil else {
            record(.diskUnavailable)
            return
        }
        let save = QueuedSave(image: image, kind: kind)
        let tokens = image.frontier.consumedTokenCount
        let bytes = (try? PagedCheckpointFormat.layout(capacity: tokens, tokens: tokens, hasLogits: image.nextLogits != nil).bytes) ?? 0
        afterSaves(bytes: UInt64(bytes)) { [self] in
            do {
                try promptCache!.write(save.image, kind: save.kind)
                saveLock.lock()
                finished.saves += 1
                saveLock.unlock()
            } catch {
                saveLock.lock()
                finished.failures += 1
                saveLock.unlock()
                // Paused for disk space: the cache logged that once.
                if promptCache!.savesPaused == nil { Self.report("durable conversation save failed", error) }
            }
        }
        record(.saveQueued)
    }

    /// Runs `work` on the save queue after every save queued before it (X-Origami-Keep pins an entry once it is written).
    /// `bytes`: what it writes, which bounds how long a waiter waits for it (`drainSaves`).
    package func afterSaves(bytes: UInt64 = 0, _ work: @escaping @Sendable () -> Void) {
        pendingSaves.enter()
        saveQueue.async { [self] in
            saveLock.withLock { running = (.now(), Self.saveBound(bytes: bytes)) }
            work()
            saveLock.withLock { running = nil }
            pendingSaves.leave()
        }
    }
    /// How long a save of `bytes` may run before its drive is taken for stopped (ROADMAP R2): as a demand read of as
    /// many bytes at 1 GB/s (`ExpertReadStall`: 20 times that, at least its floor), so only a drive writing under
    /// 50 MB/s trips it.
    package static func saveBound(bytes: UInt64) -> Double { ExpertReadStall.bound(bytes: Int(clamping: bytes), bytesPerSecond: 1e9) }
    /// Waits for the queued saves; a cancelled waiter leaves them running. A save running past its bound fails the
    /// waiter with `drive_stalled` (and every later one at once until it ends): a write cannot be cancelled, and the
    /// next request must not run beside it. Any thread.
    package func drainSaves(cancellation: GenerationControl? = nil) throws {
        while pendingSaves.wait(timeout: .now() + 0.2) == .timedOut {
            try cancellation?.check()
            let late = saveLock.withLock {
                running.flatMap {
                    DispatchTime.now().uptimeNanoseconds - $0.started.uptimeNanoseconds > UInt64($0.bound * 1e9) ? $0.bound : nil
                }
            }
            if let late {
                throw EngineError(
                    String(
                        format: "A prompt-cache save has not finished in %.0f s: the drive holding saved prompts stopped answering", late),
                    code: "drive_stalled")
            }
        }
    }
    /// The next request's first step, on the thread that owns `statistics`: waits for the saves, then counts them.
    package func finishSaves(cancellation: GenerationControl? = nil) throws {
        try drainSaves(cancellation: cancellation)
        saveLock.lock()
        let done = finished
        finished = (0, 0)
        saveLock.unlock()
        statistics.saves &+= done.saves
        statistics.diskSaveFailures &+= done.failures
        if done.failures > 0 { record(.diskSaveFailed) }
    }

    /// An untrusted manifest of a saved image, used only to pick the expected segments and frontier; the store's
    /// full read re-verifies everything before any state is published.
    private func peek(_ store: PersistentCheckpointStore) throws -> CheckpointDiskHint { try store.peekPaged() }

    private static func isMemoryError(_ error: Error) -> Bool {
        if (error as? EngineError)?.code == "insufficient_memory" { return true }
        let ns = error as NSError
        return ns.domain == NSPOSIXErrorDomain && ns.code == Int(ENOMEM)
    }
}
