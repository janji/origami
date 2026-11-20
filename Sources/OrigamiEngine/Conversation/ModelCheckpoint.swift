import CryptoKit
import Darwin
import Foundation

/// Compatibility is explicit: callers bind this to installed artifact hashes and
/// the complete arithmetic schedule, including device/compiler/reduction rules.
public struct CheckpointIdentity: Codable, Equatable, Sendable {
    public let modelSHA256, tokenizerSHA256, templateSHA256, quantizationSHA256: String
    public let arithmeticEpoch: String

    public init(
        modelSHA256: String, tokenizerSHA256: String, templateSHA256: String,
        quantizationSHA256: String, arithmeticEpoch: String
    ) throws {
        self.modelSHA256 = modelSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.templateSHA256 = templateSHA256
        self.quantizationSHA256 = quantizationSHA256
        self.arithmeticEpoch = arithmeticEpoch
        try validate()
    }
    public func validate() throws {
        guard [modelSHA256, tokenizerSHA256, templateSHA256, quantizationSHA256].allSatisfy(checkpointHash),
            !arithmeticEpoch.isEmpty, arithmeticEpoch.utf8.count <= 256
        else {
            throw EngineError("Invalid checkpoint artifact or arithmetic identity")
        }
    }
}

func checkpointHash(_ text: String) -> Bool {
    text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

/// Describes each *producing* forward. An equal token prefix with different
/// projection padding, query tiles or dispatch schedule is not an exact hit.
public struct CheckpointSegment: Codable, Equatable, Sendable {
    public enum Phase: String, Codable, Sendable { case prefill, decode }
    public let start, count, projectionRows, queryTile, keyExtent: Int
    public let phase: Phase
    public let dispatchSHA256: String
    public init(
        start: Int, count: Int, projectionRows: Int, queryTile: Int,
        keyExtent: Int, phase: Phase, dispatchSHA256: String
    ) throws {
        self.start = start
        self.count = count
        self.projectionRows = projectionRows
        self.queryTile = queryTile
        self.keyExtent = keyExtent
        self.phase = phase
        self.dispatchSHA256 = dispatchSHA256
        try validate()
    }
    public func validate() throws {
        guard start >= 0, start <= ModelContract.maximumContext, (1...256).contains(count),
            count <= ModelContract.maximumContext - start,
            (count...256).contains(projectionRows), (1...256).contains(queryTile),
            keyExtent == start + count, phase != .decode || count == 1,
            checkpointHash(dispatchSHA256)
        else { throw EngineError("Invalid checkpoint forward segment") }
    }
}

/// What the caller will do right after a restore; `CheckpointFrontier.validate(intent:hasLogits:)` refuses a
/// restore whose frontier cannot support it.
public enum CheckpointRestoreIntent: Sendable {
    /// Extend a consumed prefix; this does not restart a suspended sampler.
    case appendInput
    /// Sample saved logits only when no token was already produced from them.
    case sampleFrontier
    /// The caller must forward the returned pending token exactly once next.
    case consumePending
}

/// Where a checkpoint stands: tokens forwarded into the state, completion tokens sampled, and a sampled token
/// not yet forwarded (at most one).
public struct CheckpointFrontier: Codable, Equatable, Sendable {
    public let consumedTokenCount: Int
    /// Number of completion tokens sampled, including an unconsumed token.
    /// This count is supplied by the generation owner, never inferred from IDs.
    public let producedTokenCount: Int
    public let emittedButUnconsumed: Int?

    public init(consumedTokenCount: Int, producedTokenCount: Int, emittedButUnconsumed: Int?) throws {
        self.consumedTokenCount = consumedTokenCount
        self.producedTokenCount = producedTokenCount
        self.emittedButUnconsumed = emittedButUnconsumed
        try validate()
    }
    public func validate() throws {
        guard (0...ModelContract.maximumContext).contains(consumedTokenCount), producedTokenCount >= 0,
            producedTokenCount <= consumedTokenCount + (emittedButUnconsumed == nil ? 0 : 1),
            emittedButUnconsumed == nil || (producedTokenCount > 0 && (0..<248_320).contains(emittedButUnconsumed!))
        else {
            throw EngineError("Invalid consumed/produced checkpoint frontier")
        }
    }
    public func validate(intent: CheckpointRestoreIntent, hasLogits: Bool) throws {
        try validate()
        switch intent {
        case .appendInput:
            guard emittedButUnconsumed == nil else { throw EngineError("Consume the already emitted token before appending input") }
        case .sampleFrontier:
            guard emittedButUnconsumed == nil, hasLogits, consumedTokenCount > 0 else {
                throw EngineError("Frontier cannot be sampled: pending token or missing saved logits")
            }
        case .consumePending:
            guard emittedButUnconsumed != nil else { throw EngineError("No emitted token is pending consumption") }
        }
    }
}

/// A conversation's subordinate quota follows readers, not index entries.
/// The process ledger still owns the one actual memory charge.
package final class CheckpointMemoryQuota {
    package let maximumBytes: UInt64
    private let lock = NSLock()
    private var live: UInt64 = 0
    package init(maximumBytes: UInt64) { self.maximumBytes = maximumBytes }
    package var liveBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return live
    }
    package func reserve(_ bytes: UInt64) throws {
        lock.lock()
        defer { lock.unlock() }
        guard bytes <= maximumBytes - live else {
            throw EngineError("Pinned checkpoint readers exceed conversation memory quota", code: "conversation_frontier_quota")
        }
        live += bytes
    }
    package func release(_ bytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        precondition(bytes <= live)
        live -= bytes
    }
}

/// The memory-ledger charge (and optional conversation quota share) of one checkpoint's host arrays, released
/// when the checkpoint and every reader of it are gone.
package final class CheckpointMemoryLease {
    let ledger: MemoryLedger
    private let category: String
    private var reservations: [MemoryLedger.Reservation]
    private let quota: CheckpointMemoryQuota?
    private let extraQuotaBytes: UInt64
    package private(set) var bytes: UInt64
    package init(
        ledger: MemoryLedger, bytes: UInt64, category: String = "model_checkpoint", quota: CheckpointMemoryQuota? = nil,
        extraQuotaBytes: UInt64 = 0
    ) throws {
        self.ledger = ledger
        self.bytes = bytes
        self.category = category
        self.quota = quota
        self.extraQuotaBytes = extraQuotaBytes
        let (charged, overflow) = bytes.addingReportingOverflow(extraQuotaBytes)
        guard !overflow else { throw EngineError("Checkpoint quota size overflows") }
        try quota?.reserve(charged)
        do { reservations = [try ledger.reserve(bytes: bytes, category: category)] } catch {
            quota?.release(charged)
            throw error
        }
    }

    /// Convert a prospective single-owner admission to measured retained bytes.
    package func shrink(to actual: UInt64) throws {
        guard reservations.count == 1, actual <= bytes else { throw EngineError("Invalid checkpoint reservation shrink") }
        try ledger.shrink(reservations[0], to: actual)
        quota?.release(bytes - actual)
        bytes = actual
    }

    /// Swift's native Array derives capacity from malloc_size, so a reused
    /// Darwin allocation can exceed malloc_good_size. On this arm64
    /// target runtime (macOS 26.3; inspected Swift 6.3.2 native Array ABI),
    /// libmalloc's magazine and xzone caches both
    /// require cachedBytes < 2 * roundedRequestBytes. Verified in Apple source
    /// libmalloc-812.100.31: magazine_large.c/large_malloc_best_fit_in_cache and
    /// xzone_segment.c/_xzm_segment_group_alloc_huge_chunk_from_cache.
    /// The 256-byte allowance covers the native Array header/alignment (32 bytes
    /// for these scalar types). Requalify this bound when changing the runtime;
    /// every measured capacity is still checked before payload I/O/publication.
    package static func arrayAdmissionBytes<T>(count: Int, element: T.Type) throws -> UInt64 {
        let stride = MemoryLayout<T>.stride
        guard count >= 0, [2, 4, 8].contains(stride), MemoryLayout<T>.alignment <= 8,
            count <= (Int.max / 2 - 256) / stride
        else {
            throw EngineError("Checkpoint array allocation geometry exceeds bounds")
        }
        if count == 0 { return 0 }
        let request = count * stride + 256
        let rounded = malloc_good_size(request)
        guard rounded >= request, rounded <= Int.max / 2 else {
            throw EngineError("Checkpoint allocator size exceeds bounds")
        }
        return UInt64(rounded) * 2
    }

    /// Precharge only the next array's worst case, then retain its measured
    /// capacity. Earlier arrays remain charged throughout; a 32K checkpoint
    /// never needs a second whole-checkpoint reservation for allocator slack.
    /// The closure is also used by the package regression to deterministically
    /// model an oversized cached block without changing global malloc settings.
    package func allocateArray<T>(count: Int, _ allocate: (Int) throws -> [T]) throws -> [T] {
        let bound = try Self.arrayAdmissionBytes(count: count, element: T.self)
        try quota?.reserve(bound)
        let reservation: MemoryLedger.Reservation
        do { reservation = try ledger.reserve(bytes: bound, category: category) } catch {
            quota?.release(bound)
            throw error
        }
        var retained = false
        defer {
            if !retained {
                try? ledger.release(reservation)
                quota?.release(bound)
            }
        }
        let result = try allocate(count)
        guard result.count == count, result.capacity <= Int.max / MemoryLayout<T>.stride else {
            throw EngineError("Checkpoint array allocator returned invalid geometry")
        }
        let actual = UInt64(result.capacity * MemoryLayout<T>.stride)
        guard actual <= bound else {
            throw EngineError(
                "Checkpoint array capacity exceeded admission bound: actual=\(actual) admitted=\(bound) count=\(count) stride=\(MemoryLayout<T>.stride)"
            )
        }
        try ledger.shrink(reservation, to: actual)
        quota?.release(bound - actual)
        reservations.append(reservation)
        bytes += actual
        retained = true
        return result
    }
    deinit {
        for reservation in reservations { try? ledger.release(reservation) }
        quota?.release(bytes + extraQuotaBytes)
    }
}

/// Immutable value-semantic image, independent of SequenceState's validity and
/// active-executor UUID. COW arrays retain their ledger reservation until every
/// checkpoint reader retires. This is not a sampler, parser or SSE checkpoint.
public final class ModelCheckpoint {
    public let identity: CheckpointIdentity
    public let segments: [CheckpointSegment]
    public let frontier: CheckpointFrontier
    public let consumedIDs: [Int64]
    public let nextLogits: [Float]?
    /// The images among `consumedIDs`; set by the conversation that captured it, not stored in the file.
    package var images: [ImageKey] = []
    // Cache quota includes retained Metal storage, but its shared owner already
    // holds the ledger lease. Capturing another reader never charges it twice.
    public var reservedBytes: UInt64 {
        memory.bytes + gdn.values.reduce(UInt64(0)) { $0 + $1.residentCapacityBytes } + (recurrentFile?.metadataBytes ?? 0)
    }
    let history: [Int64]
    package let gdn: [Int: GDNState]
    private let recurrentFile: FrontierRecurrentFile?
    package let qsa: [Int: QSAState]
    package let pleConvolution: [UInt16]
    private let memory: CheckpointMemoryLease

    public static func capture(
        state: SequenceState, consumedIDs: [Int64], identity: CheckpointIdentity,
        segments: [CheckpointSegment], producedTokenCount: Int = 0,
        nextLogits: [Float]? = nil, ledger: MemoryLedger
    ) throws -> ModelCheckpoint {
        try capture(
            state: state, consumedIDs: consumedIDs, identity: identity, segments: segments,
            producedTokenCount: producedTokenCount, nextLogits: nextLogits, ledger: ledger, frontierStore: nil)
    }
    package static func capture(
        state: SequenceState, consumedIDs: [Int64], identity: CheckpointIdentity,
        segments: [CheckpointSegment], producedTokenCount: Int = 0,
        nextLogits: [Float]? = nil, ledger: MemoryLedger,
        frontierStore: ConversationFrontierStore?, inMemoryRecurrent: Bool = false
    ) throws -> ModelCheckpoint {
        try state.validate()
        guard segments.count <= state.tokenCount else { throw EngineError("Checkpoint has more producing segments than consumed tokens") }
        let frontier = try CheckpointFrontier(
            consumedTokenCount: state.tokenCount,
            producedTokenCount: producedTokenCount, emittedButUnconsumed: state.emittedButUnconsumed)
        let file = try inMemoryRecurrent ? nil : frontierStore?.make(offset: state.tokenCount)
        if let file {
            for layer in state.gdn.keys.sorted() { try file.retain(layer: layer, state: state.gdn[layer]!) }
            try file.seal()
        }
        let gdn = file == nil ? state.gdn : [:]
        let bytes =
            gdn.values.reduce(UInt64(0)) { $0 + $1.hostCapacityBytes }
            + state.qsa.values.reduce(UInt64(0)) { $0 + $1.hostCapacityBytes }
            + UInt64(
                state.ple.convolution.capacity * 2 + state.history.capacity * 8 + consumedIDs.capacity * 8
                    + (nextLogits?.capacity ?? 0) * 4 + segments.count * 1024 + 65_536)
        // A tool anchor retains the already immutable GPU state instead of writing it to disk.
        // Its buffers own their ledger charge; the frontier quota still counts their full size.
        let resident = gdn.values.reduce(UInt64(0)) { $0 + $1.residentCapacityBytes }
        let memory = try CheckpointMemoryLease(
            ledger: ledger, bytes: bytes, quota: frontierStore?.memoryQuota,
            extraQuotaBytes: resident)
        return try ModelCheckpoint(
            identity: identity, segments: segments, frontier: frontier,
            consumedIDs: consumedIDs, nextLogits: nextLogits, history: state.history,
            gdn: gdn, qsa: state.qsa, pleConvolution: state.ple.convolution, memory: memory, recurrentFile: file)
    }

    init(
        identity: CheckpointIdentity, segments: [CheckpointSegment], frontier: CheckpointFrontier,
        consumedIDs: [Int64], nextLogits: [Float]?, history: [Int64], gdn: [Int: GDNState],
        qsa: [Int: QSAState], pleConvolution: [UInt16], memory: CheckpointMemoryLease,
        recurrentFile: FrontierRecurrentFile? = nil
    ) throws {
        self.identity = identity
        self.segments = segments
        self.frontier = frontier
        self.consumedIDs = consumedIDs
        self.nextLogits = nextLogits
        self.history = history
        self.gdn = gdn
        self.qsa = qsa
        self.pleConvolution = pleConvolution
        self.memory = memory
        self.recurrentFile = recurrentFile
        try validate()
    }

    package func validate() throws {
        try identity.validate()
        try frontier.validate()
        guard consumedIDs.count == frontier.consumedTokenCount,
            consumedIDs.allSatisfy({ (0..<248_320).contains($0) }),
            history == Array(([Int64(248_044), 248_044] + consumedIDs).suffix(2)),
            pleConvolution.count == 9 * 10_240,
            recurrentFile == nil ? Set(gdn.keys) == Set((0..<48).filter { $0 % 4 != 3 }) : gdn.isEmpty,
            Set(qsa.keys) == Set((0..<48).filter { $0 % 4 == 3 }),
            nextLogits == nil || nextLogits!.count == 248_320,
            segments.count <= ModelContract.maximumContext
        else { throw EngineError("Incomplete or inconsistent checkpoint image") }
        var end = 0
        for segment in segments {
            try segment.validate()
            guard segment.start == end else { throw EngineError("Checkpoint segments contain a gap or overlap") }
            end += segment.count
        }
        guard end == frontier.consumedTokenCount else { throw EngineError("Checkpoint segmentation does not reach frontier") }
        try recurrentFile?.validate(expectedOffset: end)
        for state in gdn.values {
            try state.validate()
            guard state.offset == end else { throw EngineError("Checkpoint GDN frontier mismatch") }
        }
        for state in qsa.values {
            try state.validate()
            guard state.offset == end else { throw EngineError("Checkpoint QSA frontier mismatch") }
        }
    }

    /// Bounded per-section raw-bit fingerprints (the checks compare restored state with them).
    /// This does not allocate a tensor-sized copy or expose mutable state.
    public func tensorSHA256() throws -> [String: String] {
        var result: [String: String] = [:]
        try withTensors { name, _, bytes in
            var hash = SHA256()
            hash.update(bufferPointer: bytes)
            result[name] = hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        return result
    }

    /// Validates everything before changing an admitted *blank* sequence. The
    /// target keeps its executor UUID and capacity reservation. Returned logits
    /// remain owned by this checkpoint; keep the checkpoint alive while using
    /// them. Restoring does not establish generation RNG or output-parser state.
    @discardableResult
    public func restore(
        into target: SequenceState, expectedIdentity: CheckpointIdentity,
        expectedSegments: [CheckpointSegment], expectedConsumedIDs: [Int64],
        intent: CheckpointRestoreIntent
    ) throws -> CheckpointFrontier {
        try validate()
        try target.validate()
        guard identity == expectedIdentity, segments == expectedSegments, consumedIDs == expectedConsumedIDs else {
            throw EngineError("Checkpoint artifact, arithmetic, segmentation or token identity mismatch")
        }
        try frontier.validate(intent: intent, hasLogits: nextLogits != nil)
        guard target.tokenCount == 0, target.emittedButUnconsumed == nil,
            frontier.consumedTokenCount < target.capacity
        else { throw EngineError("Checkpoint restore needs a blank admitted target with continuation capacity") }
        // Preallocate all QSA capacities before publication. SequenceState's
        // existing two-copy reservation covers these mutable continuation copies.
        var restoredQSA = qsa
        for layer in restoredQSA.keys {
            if let arena = target.pageArena {
                restoredQSA[layer] = try restoredQSA[layer]!.usingPages(arena)
            } else {
                try restoredQSA[layer]!.reserveCapacity(tokens: target.capacity)
            }
        }
        let restored = try recurrentFile?.restore()
        defer { withExtendedLifetime(restored) {} }
        let restoredGDN = restored?.states ?? gdn
        // Both representations of the recurrent state (in memory, or restored from the frontier file) stay under
        // 238 MB of host capacity; a larger restore is not a state this engine produced.
        guard restoredGDN.values.reduce(UInt64(0), { $0 + $1.hostCapacityBytes }) <= 238_000_000 else {
            throw EngineError("Restored recurrent allocation exceeds active state ownership")
        }
        target.gdn = restoredGDN
        target.qsa = restoredQSA
        target.ple.convolution = pleConvolution
        target.history = history
        target.tokenCount = frontier.consumedTokenCount
        target.emittedButUnconsumed = frontier.emittedButUnconsumed
        return frontier
    }

    var checkpointLedger: MemoryLedger { memory.ledger }

    // Little-endian host is required by the persistent codec. Buffers are only
    // valid synchronously inside body; no tensor-sized serialization copy.
    package func withTensors(_ body: (String, String, UnsafeRawBufferPointer) throws -> Void) throws {
        // Legacy codec exposes one contiguous section at a time. Paged state
        // admits the allocator's complete one-section bound before flattening.
        let largest = qsa.values.filter { $0.pages != nil }.map { $0.keyCount }.max() ?? 0
        let scratch =
            largest == 0
            ? nil
            : try CheckpointMemoryLease(
                ledger: memory.ledger,
                bytes: CheckpointMemoryLease.arrayAdmissionBytes(count: largest, element: UInt16.self),
                category: "checkpoint_page_section_scratch")
        defer { withExtendedLifetime(scratch) {} }
        try withPrefixTensors(body)
        for layer in qsa.keys.sorted() {
            try qsa[layer]!.keys.withUnsafeBytes { try body("qsa.\(layer).keys", "u16", $0) }
            try qsa[layer]!.values.withUnsafeBytes { try body("qsa.\(layer).values", "u16", $0) }
            try qsa[layer]!.pooledKeys.withUnsafeBytes { try body("qsa.\(layer).pooled", "u16", $0) }
            try qsa[layer]!.rawTail.withUnsafeBytes { try body("qsa.\(layer).tail", "u16", $0) }
        }
        if let nextLogits { try nextLogits.withUnsafeBytes { try body("next_logits", "f32", $0) } }
    }
    func withPrefixTensors(_ body: (String, String, UnsafeRawBufferPointer) throws -> Void) throws {
        try consumedIDs.withUnsafeBytes { try body("consumed_ids", "i64", $0) }
        try history.withUnsafeBytes { try body("history", "i64", $0) }
        try pleConvolution.withUnsafeBytes { try body("ple_convolution", "u16", $0) }
        if let recurrentFile {
            try recurrentFile.withTensors { layer, name, bytes in
                try body("gdn.\(layer).\(name)", name == "recurrent" ? "f32" : "u16", bytes)
            }
        } else {
            for layer in gdn.keys.sorted() {
                try gdn[layer]!.withRecurrentBytes { try body("gdn.\(layer).recurrent", "f32", $0) }
                try gdn[layer]!.convolutionTail.withUnsafeBytes { try body("gdn.\(layer).tail", "u16", $0) }
            }
        }
    }

}
