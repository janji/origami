import Foundation

/// An unpublished full-model frontier assembled along the existing layer-major
/// read scope. Each attention tile has joined before its immutable state enters
/// this owner. Publication waits for every layer and the entire scope to succeed.
/// The enclosing R dispatch identity is retained even when the image ends inside
/// R; it does not assert equivalence to cold prefill with a shorter read scope.
package final class ConversationScopeCapture {
    /// The prospective memory and file-metadata cost of one staged frontier at `offset`, before any layer runs.
    package struct Admission {
        package let offset: Int, timelineCount: Int
        package let metadataBytes, tensorBytes, fileMetadataBytes: UInt64
        var memoryBytes: UInt64 { metadataBytes + tensorBytes }
        package var quotaBytes: UInt64 { memoryBytes + fileMetadataBytes }
    }
    /// The drafts admitted (cheapest first, later offset on ties), their total cost and why selection stopped.
    package struct Selection {
        package let drafts: [Admission]
        package let bytes: UInt64, reason: String
    }
    package static func select(_ requested: [Admission], refusal: (UInt64, Int) -> String?, evict: () -> Bool) -> Selection {
        let ranked = requested.sorted { $0.quotaBytes == $1.quotaBytes ? $0.offset > $1.offset : $0.quotaBytes < $1.quotaBytes }
        var admitted: [Admission] = []
        var needed: UInt64 = 0
        for candidate in ranked {
            let bytes = needed + candidate.quotaBytes
            let count = admitted.count + 1
            while refusal(bytes, count) != nil, evict() {}
            if let reason = refusal(bytes, count) { return Selection(drafts: admitted, bytes: needed, reason: reason) }
            admitted.append(candidate)
            needed = bytes
        }
        return Selection(drafts: admitted, bytes: needed, reason: "complete_requested_set")
    }
    /// Before a layer executes, account for the *actual producing history* and
    /// the allocator's token-array bound. Long one-row histories are legal.
    package static func admission(offset: Int, segments: [CheckpointSegment], fileBacked: Bool) throws -> Admission {
        guard (1...ModelContract.maximumContext).contains(offset),
            let end = segments.lastIndex(where: { $0.start + $0.count <= offset }),
            segments[end].start + segments[end].count == offset
        else {
            throw EngineError("Staged frontier is not a complete producing boundary")
        }
        let metadata =
            try CheckpointMemoryLease.arrayAdmissionBytes(count: offset, element: Int64.self)
            + UInt64((end + 1) * 1024 + 65_536)
        return Admission(
            offset: offset, timelineCount: end + 1, metadataBytes: metadata,
            // PLE/tails plus 36 page-reference arrays, including growth and
            // allocator slack at 1024 pages/vector. Payload pages own separate leases.
            tensorBytes: fileBacked ? QSAPagedStorage.maximumReferenceBytes + 1_048_576 : 130_000_000,
            fileMetadataBytes: fileBacked ? 262_144 : 0)
    }
    /// One frontier being assembled: its lease, its recurrent file (file frontiers) and the layers retained so far.
    private final class Draft {
        let offset: Int
        let admission: Admission
        let memory: CheckpointMemoryLease
        let recurrentFile: FrontierRecurrentFile?
        var gdn: [Int: GDNState] = [:], qsa: [Int: QSAState] = [:]
        var ple: [UInt16]?
        var hostBytes: UInt64
        var retainedBytes: UInt64 { hostBytes + gdn.values.reduce(0) { $0 + $1.residentCapacityBytes } }
        init(admission: Admission, ledger: MemoryLedger, frontierStore: ConversationFrontierStore?) throws {
            self.admission = admission
            offset = admission.offset
            hostBytes = admission.metadataBytes
            memory = try CheckpointMemoryLease(
                ledger: ledger, bytes: admission.memoryBytes,
                category: "conversation_scope_checkpoint", quota: frontierStore?.memoryQuota)
            recurrentFile = try frontierStore?.make(offset: offset)
        }
        func retain(layer: Int, state: SequenceState) throws {
            guard gdn[layer] == nil, qsa[layer] == nil else { throw EngineError("Duplicate staged checkpoint layer") }
            var host: UInt64 = 0
            var resident: UInt64 = 0
            if layer % 4 == 3 {
                guard let value = state.qsa[layer], value.offset == offset, value.pages != nil else {
                    throw EngineError("Staged QSA checkpoint has wrong frontier or ownership")
                }
                try value.validate()
                host = value.hostCapacityBytes
            } else {
                guard let value = state.gdn[layer], value.offset == offset else {
                    throw EngineError("Staged GDN checkpoint has wrong frontier")
                }
                try value.validate()
                if let recurrentFile {
                    try recurrentFile.retain(layer: layer, state: value)
                } else {
                    host = value.hostCapacityBytes
                    resident = value.residentCapacityBytes
                }
            }
            if layer == 1 { host += UInt64(state.ple.convolution.capacity * 2) }
            guard retainedBytes + host + resident <= memory.bytes else {
                throw EngineError("Staged checkpoint exceeds its prospective bound", code: "conversation_frontier_quota")
            }
            hostBytes += host
            if layer % 4 == 3 { qsa[layer] = state.qsa[layer] } else if recurrentFile == nil { gdn[layer] = state.gdn[layer] }
            if layer == 1 { ple = state.ple.convolution }
        }
    }
    package let offsets: [Int]
    package private(set) var retentionSeconds = 0.0
    private let identity: CheckpointIdentity, consumed: [Int64], segments: [CheckpointSegment]
    private let scopeEnd: Int
    private var drafts: [Int: Draft] = [:]
    private var published = false

    package init(
        offsets: [Int], scope: Range<Int>, consumed: [Int64], segments: [CheckpointSegment],
        identity: CheckpointIdentity, ledger: MemoryLedger, frontierStore: ConversationFrontierStore? = nil
    ) throws {
        guard !offsets.isEmpty, offsets.count <= 31, offsets == offsets.sorted(), Set(offsets).count == offsets.count,
            scope.count <= 8192,
            offsets.allSatisfy({ $0 > scope.lowerBound && $0 < scope.upperBound && ($0 - scope.lowerBound) % 256 == 0 }),
            consumed.count == scope.upperBound, segments.last.map({ $0.start + $0.count }) == scope.upperBound
        else {
            throw EngineError("Invalid intra-scope checkpoint placement")
        }
        self.offsets = offsets
        scopeEnd = scope.upperBound
        self.consumed = consumed
        self.segments = segments
        self.identity = identity
        for offset in offsets {
            drafts[offset] = try Draft(
                admission: Self.admission(
                    offset: offset, segments: segments,
                    fileBacked: frontierStore != nil), ledger: ledger, frontierStore: frontierStore)
        }
    }

    package func retain(layer: Int, offset: Int, state: SequenceState) throws {
        guard !published, (0..<48).contains(layer) else { throw EngineError("Invalid staged checkpoint publication") }
        let started = DispatchTime.now().uptimeNanoseconds
        defer { retentionSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 }
        // The executor offers every tile end; only the admitted frontier offsets have a draft.
        try drafts[offset]?.retain(layer: layer, state: state)
    }

    package func finish(state: SequenceState) throws -> [ModelCheckpoint] {
        try state.validate()
        guard !published, state.tokenCount == scopeEnd else { throw EngineError("Staged checkpoints need a complete successful scope") }
        var images: [ModelCheckpoint] = []
        for offset in offsets {
            guard let draft = drafts[offset], let ple = draft.ple else { throw EngineError("Staged checkpoint omitted PLE") }
            let ids = Array(consumed.prefix(offset))
            let timeline = Array(segments.prefix(draft.admission.timelineCount))
            let metadata = UInt64(ids.capacity * 8 + timeline.count * 1024 + 65_536)
            guard metadata <= draft.admission.metadataBytes,
                UInt64(timeline.capacity * MemoryLayout<CheckpointSegment>.stride + timeline.count * 256)
                    <= UInt64(timeline.count * 1024 + 65_536)
            else {
                throw EngineError("Staged checkpoint metadata exceeds admission")
            }
            try draft.memory.shrink(to: draft.hostBytes - draft.admission.metadataBytes + metadata)
            try draft.recurrentFile?.seal()
            images.append(
                try ModelCheckpoint(
                    identity: identity, segments: timeline,
                    frontier: CheckpointFrontier(consumedTokenCount: offset, producedTokenCount: 0, emittedButUnconsumed: nil),
                    consumedIDs: ids, nextLogits: nil, history: Array(([Int64(248044), 248044] + ids).suffix(2)),
                    gdn: draft.gdn, qsa: draft.qsa, pleConvolution: ple, memory: draft.memory, recurrentFile: draft.recurrentFile))
        }
        published = true
        drafts.removeAll()
        return images
    }
}
