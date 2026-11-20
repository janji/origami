import Foundation
import OrigamiEngine

/// No model weights or inference. These exercise actual allocation geometry,
/// complete state ownership and denial, not numerical model qualification.
package enum PersistentConversationSelfChecks {
    nonisolated(unsafe) package private(set) static var reports: [[String: UInt64]] = []
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError(message) }
    }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Expected conversation admission refusal")
    }
    package static func pages() throws {
        let ledger = try MemoryLedger(ceiling: 14_000_000, transientMargin: 1_000_000)
        var arena: QSAPageArena? = try QSAPageArena(ledger: ledger, maximumBytes: 12_000_000)
        var current: QSAState? = QSAState(arena: arena!)
        var retained: [QSAState] = []
        for end in [3, 4, 5, 255, 256, 257, 2047, 2048, 2049] {
            let n = end - current!.offset
            let old = current!.offset
            try current!.append(
                keys: (old * 512..<end * 512).map { UInt16($0 % 65521) },
                values: (old * 512..<end * 512).map { UInt16(($0 + 19) % 65521) },
                pooledKeys: (old / 4 * 128..<end / 4 * 128).map { UInt16(($0 + 3) % 65521) },
                rawTail: [UInt16](repeating: 7, count: end % 4 * 128), tokens: n)
            try current!.validate()
            retained.append(current!)
        }
        for image in retained {
            try require(image.keys == (0..<image.offset * 512).map { UInt16($0 % 65521) }, "Published QSA prefix mutated")
            try require(image.values == (0..<image.offset * 512).map { UInt16(($0 + 19) % 65521) }, "Published QSA values mutated")
            try require(image.pooledKeys == (0..<image.offset / 4 * 128).map { UInt16(($0 + 3) % 65521) }, "Published pooled index mutated")
        }
        try require(
            !retained[5].pages!.keys.pageIdentities.intersection(retained.last!.pages!.keys.pageIdentities).isEmpty,
            "Complete prefix pages were copied")
        try require(
            retained[7].rawTail.isEmpty && retained[7].hostCapacityBytes >= 3 * 8 * 8,
            "Aligned snapshot omitted its retained page-reference arrays")
        let before = try current!.keys
        try rejected { try current!.append(keys: [], values: [], pooledKeys: [], rawTail: [], tokens: 1) }
        try require(current!.keys == before, "Invalid append changed published state")
        retained.removeAll()
        current = nil
        try require(arena!.snapshot().liveBytes == 0, "QSA pages leaked after final reader")
        arena = nil
        try require(ledger.snapshot().reserved == 0, "QSA arena charge outlived all owners")

        // A partial append needs a second immutable page while its reader lives.
        let tiny = try QSAPageArena(ledger: ledger, maximumBytes: 16_384)
        let original = try QSAPagedVector(arena: tiny, pageElements: 8192).appending([1, 2, 3])
        try rejected { _ = try original.appending([4]) }
        try require(
            original.array() == [1, 2, 3] && tiny.snapshot().liveBytes == 16_384 && tiny.snapshot().refusals == 1,
            "Quota failure changed reader or leaked provisional allocation")
    }

    /// The previous prompt's image stays protected while later prompts extend it; a prompt that does not lets its growth
    /// evict that image (on the 32K tier a 12K-token chat followed by any long other prompt used to be refused).
    package static func unrelatedPromptReleasesProtection() throws {
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let identity = try CheckpointIdentity(
                modelSHA256: String(repeating: "1", count: 64),
                tokenizerSHA256: String(repeating: "2", count: 64), templateSHA256: String(repeating: "3", count: 64),
                quantizationSHA256: String(repeating: "4", count: 64), arithmeticEpoch: "protected-prompt-fixture-v1")
            let owner = try PersistentConversation(
                identity: identity, ledger: ledger,
                maximumBytes: try ConversationGeometry(context: 32768).conversationBytes, contextLimit: 32768)
            let tokens = 12288
            let ids = [Int64](repeating: 17, count: tokens)
            try autoreleasepool {
                let state = try SequenceState(
                    inputTokens: tokens, outputTokens: 1, ledger: ledger, contextLimit: 32768, pageArena: owner.arena)
                for layer in 0..<48 {
                    if layer % 4 == 3 {
                        for _ in stride(from: 0, to: tokens, by: 256) {
                            try state.qsa[layer]!.append(
                                keys: [UInt16](repeating: 1, count: 256 * 512), values: [UInt16](repeating: 2, count: 256 * 512),
                                pooledKeys: [UInt16](repeating: 3, count: 64 * 128), rawTail: [], tokens: 256)
                        }
                    } else {
                        state.gdn[layer]!.offset = tokens
                    }
                }
                state.tokenCount = tokens
                state.history = [17, 17]
                let segments = try PrefillReadSchedule(promptTokens: tokens, computeTile: 256, readScope: 8192)
                    .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
                try owner.committed([Int](repeating: 17, count: tokens), producing: segments, state: state, logits: nil, replay: false)
                _ = try owner.capture(state, prompt: true)
            }
            // The image's pages (311 MB of the 1.1 GB arena) leave less than a whole-context growth (837 MB).
            try require(owner.arena.snapshot().liveBytes == UInt64(tokens) * 25_344, "Prompt image fixture differs")
            try rejected { try owner.admitGrowth(tokens: 32768) }
            owner.leaveTimeline(for: ids + [18], images: [])
            try rejected { try owner.admitGrowth(tokens: 32768) }
            owner.leaveTimeline(for: [Int64](repeating: 19, count: 24576), images: [])
            try owner.admitGrowth(tokens: 32768)
            try require(
                owner.arena.snapshot().liveBytes == 0 && owner.statistics().retainedFrontiers == 0,
                "An unrelated prompt could not evict the previous prompt's image")
        }
        try require(ledger.snapshot().reserved == 0, "Protected prompt fixture leaked a reservation")
    }

    package static func shape8K() throws { try shape(context: 8192) }
    package static func toolAnchorOwnership() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-anchor-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try autoreleasepool {
            let hash = String(repeating: "a", count: 64)
            let identity = try CheckpointIdentity(
                modelSHA256: hash, tokenizerSHA256: hash, templateSHA256: hash,
                quantizationSHA256: hash, arithmeticEpoch: "tool-anchor-fixture")
            let geometry = try ConversationGeometry(context: 8192)
            let store = try ConversationFrontierStore(
                parent: folder, maximumBytes: geometry.frontierDiskBytes,
                ledger: ledger, memoryBytes: geometry.frontierMemoryBytes)
            let owner = try PersistentConversation(
                identity: identity, ledger: ledger, maximumBytes: geometry.conversationBytes,
                contextLimit: 8192, frontierStore: store)
            let state = try SequenceState(inputTokens: 512, outputTokens: 4, ledger: ledger, contextLimit: 8192, pageArena: owner.arena)
            for layer in 0..<48 {
                if layer % 4 == 3 {
                    try state.qsa[layer]!.append(
                        keys: [UInt16](repeating: 1, count: 512 * 512),
                        values: [UInt16](repeating: 2, count: 512 * 512), pooledKeys: [UInt16](repeating: 3, count: 128 * 128),
                        rawTail: [], tokens: 512)
                } else {
                    state.gdn[layer]!.offset = 512
                }
            }
            state.tokenCount = 512
            state.history = [17, 17]
            state.emittedButUnconsumed = 49
            let segments = try PrefillReadSchedule(promptTokens: 512, computeTile: 256, readScope: 512)
                .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
            try owner.committed([Int](repeating: 17, count: 512), producing: segments, state: state, logits: nil, replay: false)
            let allowance = owner.toolAnchorAllowance(state)
            try require(
                allowance != nil && allowance! < geometry.frontierMemoryBytes,
                "A small tool anchor reclaimed the whole frontier allowance")
            let anchor = try owner.capture(state, produced: 1, tool: true)
            let matching = anchor.consumedIDs + [49]
            try require(try owner.reusableFrontier(for: matching) === anchor, "Matching prompt did not find its retained frontier")
            try require(
                try owner.reusableFrontier(for: matching, reuse: false) == nil,
                "Isolated prefill tuning reused a matching frontier")
            try require(owner.statistics().retainedFrontiers == 1, "Bypassing reuse discarded an unrelated saved frontier")
            try require(
                store.snapshot().liveBytes == 0 && store.memoryQuota!.liveBytes == anchor.reservedBytes,
                "Tool anchor wrote recurrent state to disk or escaped its memory quota")
            state.gdn[0]!.recurrent[0] = 7
            state.ple.convolution[0] = 9
            state.emittedButUnconsumed = nil
            let target = try SequenceState(inputTokens: 512, outputTokens: 4, ledger: ledger, contextLimit: 8192, pageArena: owner.arena)
            try rejected {
                _ = try anchor.restore(
                    into: target, expectedIdentity: identity, expectedSegments: segments,
                    expectedConsumedIDs: [Int64](repeating: 18, count: 512), intent: .consumePending)
            }
            let pending = try anchor.restore(
                into: target, expectedIdentity: identity, expectedSegments: segments,
                expectedConsumedIDs: anchor.consumedIDs, intent: .consumePending)
            try require(
                pending.emittedButUnconsumed == 49 && target.gdn[0]!.recurrent[0] == 0 && target.ple.convolution[0] == 0,
                "Tool anchor did not preserve mutable recurrence or pending opener")
            _ = try owner.capture(target, produced: 1, tool: true)
            try require(
                owner.statistics().retainedFrontiers == 1 && owner.statistics().toolAnchors == 2,
                "A second tool anchor retained the previous indexed anchor")
        }
        try require(ledger.reservedBytes == 0, "Tool anchor retained its state after the final reader retired")
    }
    package static func denseMetadataAndPinnedQuota() throws {
        let hash = String(repeating: "a", count: 64)
        let segments = try (0..<262144).map {
            try CheckpointSegment(
                start: $0, count: 1, projectionRows: 1, queryTile: 1,
                keyExtent: $0 + 1, phase: .decode, dispatchSHA256: hash)
        }
        let admission = try ConversationScopeCapture.admission(offset: 261888, segments: segments, fileBacked: true)
        try require(
            admission.timelineCount == 261888 && admission.metadataBytes > 268_000_000,
            "Legal one-row history retained the obsolete1MiB allowance")
        let short = try ConversationScopeCapture.admission(offset: 256, segments: segments, fileBacked: true)
        try require(
            short.metadataBytes < 1_048_576 && admission.quotaBytes > short.quotaBytes,
            "Draft metadata is not derived from its actual prefix")
        let requested = try PersistentConversation.scopeOffsets(253952..<262144, spacing: 256).map {
            try ConversationScopeCapture.admission(offset: $0, segments: segments, fileBacked: true)
        }
        let available: UInt64 = 1_171_363_584
        let selection = ConversationScopeCapture.select(
            requested, refusal: { bytes, _ in bytes > available ? "fixture_memory" : nil }, evict: { false })
        try require(
            selection.drafts.count == 4 && selection.reason == "fixture_memory" && selection.bytes <= available
                && requested.map(\.quotaBytes).min()! * 5 > available,
            "Largest legal history did not admit the maximum fitting draft set")
        let denied = ConversationScopeCapture.select(requested, refusal: { _, _ in "pinned_fixture" }, evict: { false })
        try require(
            denied.drafts.isEmpty && denied.bytes == 0 && denied.reason == "pinned_fixture", "Pinned quota silently admitted a draft")
        let quota = CheckpointMemoryQuota(maximumBytes: admission.quotaBytes)
        let ledger = try MemoryLedger(ceiling: admission.quotaBytes + 1, transientMargin: 0)
        var escaped: CheckpointMemoryLease? = try CheckpointMemoryLease(ledger: ledger, bytes: admission.quotaBytes, quota: quota)
        try rejected { _ = try CheckpointMemoryLease(ledger: ledger, bytes: 1, quota: quota) }
        try require(
            quota.liveBytes == admission.quotaBytes && ledger.snapshot().reserved == admission.quotaBytes,
            "Refused pinned owner changed accounting")
        try escaped!.shrink(to: short.quotaBytes)
        try require(quota.liveBytes == short.quotaBytes, "Measured metadata shrink lost quota ownership")
        escaped = nil
        try require(quota.liveBytes == 0 && ledger.snapshot().reserved == 0, "Final pinned reader leaked its charge")
        let tiny = try MemoryLedger(ceiling: 10, transientMargin: 0)
        try rejected { _ = try CheckpointMemoryLease(ledger: tiny, bytes: 11, quota: quota) }
        try require(quota.liveBytes == 0 && tiny.snapshot().reserved == 0, "Ledger refusal leaked subordinate quota")
    }
    package static func shape32K() throws { try shape(context: 32768) }
    package static func shape8KFiles() throws { try shape(context: 8192, fileBacked: true) }
    package static func shape32KFiles() throws { try shape(context: 32768, fileBacked: true) }
    package static func scopeAssembly() throws { try scopeAssembly(fileBacked: false) }
    package static func scopeAssemblyFiles() throws { try scopeAssembly(fileBacked: true) }
    private static func scopeAssembly(fileBacked: Bool) throws {
        let offsets = PersistentConversation.scopeOffsets(0..<7680, spacing: 1024)
        try require(offsets == [1024, 2048, 3072, 4096, 5120, 6144, 7168], "8K edit placement missed useful inner frontiers")
        try require(
            PersistentConversation.scopeOffsets(327..<8519, spacing: 4096) == [4423], "Resumed scope changed producing tile alignment")
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
        let parent = CheckFixtures.scratch
        let scopeEnd = fileBacked ? 8192 : 512
        let selected = PersistentConversation.scopeOffsets(0..<scopeEnd, spacing: 256)
        var frontierStore =
            fileBacked
            ? try ConversationFrontierStore(
                parent: parent, maximumBytes: 35 * ConversationFrontierStore.imageBytes, ledger: ledger, memoryBytes: 427_819_008) : nil
        let identity = try CheckpointIdentity(
            modelSHA256: String(repeating: "1", count: 64),
            tokenizerSHA256: String(repeating: "2", count: 64), templateSHA256: String(repeating: "3", count: 64),
            quantizationSHA256: String(repeating: "4", count: 64), arithmeticEpoch: "scope-fixture-v1")
        var arena: QSAPageArena? = try QSAPageArena(ledger: ledger, maximumBytes: 268_435_456)
        var state: SequenceState? = try SequenceState(
            inputTokens: scopeEnd - 1, outputTokens: 1, ledger: ledger, contextLimit: 8192, pageArena: arena)
        let segments = try PrefillReadSchedule(promptTokens: scopeEnd, computeTile: 256, readScope: scopeEnd)
            .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
        var draft: ConversationScopeCapture? = try ConversationScopeCapture(
            offsets: selected, scope: 0..<scopeEnd,
            consumed: [Int64](repeating: 17, count: scopeEnd), segments: segments, identity: identity, ledger: ledger,
            frontierStore: frontierStore)
        try rejected { _ = try draft!.finish(state: state!) }
        for layer in 0..<48 {
            for end in stride(from: 256, through: scopeEnd, by: 256) {
                if layer % 4 == 3 {
                    try state!.qsa[layer]!.append(
                        keys: [UInt16](repeating: UInt16(end), count: 256 * 512),
                        values: [UInt16](repeating: UInt16(layer), count: 256 * 512),
                        pooledKeys: [UInt16](repeating: UInt16(end + 1), count: 64 * 128), rawTail: [], tokens: 256)
                } else {
                    state!.gdn[layer] = try GDNState(
                        offset: end,
                        recurrent: [Float](repeating: Float(layer + end), count: 48 * 128 * 128),
                        convolutionTail: [UInt16](repeating: UInt16(end + layer), count: 3 * 10240))
                }
                if layer == 1 { state!.ple.convolution = [UInt16](repeating: UInt16(end), count: 9 * 10240) }
                try draft!.retain(layer: layer, offset: end, state: state!)
            }
        }
        state!.tokenCount = scopeEnd
        state!.history = [17, 17]
        var images = try draft!.finish(state: state!)
        try rejected { _ = try draft!.finish(state: state!) }
        let shortened = try PrefillReadSchedule(promptTokens: 256, computeTile: 256, readScope: 256)
            .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
        try require(
            images.count == selected.count && images[0].segments == Array(segments.prefix(1)) && images[0].segments != shortened,
            "Intra-scope image lost the original enclosing dispatch")
        try require(
            (fileBacked ? images[0].gdn.isEmpty : (images[0].gdn[0]!.recurrent.first == 256 && images[0].gdn[46]!.recurrent.first == 302))
                && images[0].pleConvolution.first == 256 && images[0].qsa[47]!.offset == 256,
            "Intra-scope assembly mixed different frontiers")
        try require(
            images[0].qsa[47]!.pages!.keys.pageIdentities.isSubset(of: state!.qsa[47]!.pages!.keys.pageIdentities),
            "Staging copied complete QSA pages")
        if fileBacked {
            try require(images.count == 31, "Dense scope omitted a C256 boundary")
            for image in images {
                try image.validate()
                try require(
                    image.segments == Array(segments.prefix(image.consumedIDs.count / 256)), "Dense frontier lost producing timeline")
                try require(image.pleConvolution.first == UInt16(image.consumedIDs.count), "Dense PLE frontier differs")
            }
            let index = ConversationFrontierIndex()
            for image in images.reversed() { index.insert(image) }
            try require(index.evict(protecting: images[0]), "Dense index failed to evict")
            try require(index.images.map { $0.consumedIDs.count } == selected.filter { $0 != 512 }, "Cached predecessor eviction differs")
            try require(index.evict(protecting: images[0]), "Dense index failed its second eviction")
            try require(
                index.images.map { $0.consumedIDs.count } == selected.filter { $0 != 512 && $0 != 1024 },
                "Eviction did not repair affected predecessor")
        }
        state = nil
        draft = nil
        state = try SequenceState(inputTokens: scopeEnd - 1, outputTokens: 1, ledger: ledger, contextLimit: 8192, pageArena: arena)
        try images[0].restore(
            into: state!, expectedIdentity: identity, expectedSegments: images[0].segments,
            expectedConsumedIDs: [Int64](repeating: 17, count: 256), intent: .appendInput)
        try state!.validate()
        try require(
            state!.gdn[0]!.recurrent.first == 256 && state!.gdn[46]!.recurrent.first == 302, "File-backed restore changed recurrent layers")
        try autoreleasepool {
            let restored = try ModelCheckpoint.capture(
                state: state!, consumedIDs: images[0].consumedIDs,
                identity: identity, segments: images[0].segments, ledger: ledger)
            try require(try restored.tensorSHA256() == images[0].tensorSHA256(), "Complete frontier roundtrip changed tensor bytes")
        }
        state = nil
        images.removeAll()
        arena = nil
        if fileBacked {
            draft = try ConversationScopeCapture(
                offsets: selected, scope: 0..<scopeEnd,
                consumed: [Int64](repeating: 17, count: scopeEnd), segments: segments, identity: identity, ledger: ledger,
                frontierStore: frontierStore)
            try require(frontierStore!.snapshot().files == 31, "Cancellation fixture omitted unpublished owners")
            draft = nil
            try require(frontierStore!.memoryQuota!.liveBytes == 0, "Cancelled scope leaked its metadata quota")
        }
        try require(frontierStore == nil || frontierStore!.snapshot().liveBytes == 0, "File-backed checkpoint survived its final reader")
        frontierStore = nil
        try require(ledger.snapshot().reserved == 0, "Staged checkpoint ownership leaked")
    }
    package static func pendingAndSchedules() throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
        let identity = try CheckpointIdentity(
            modelSHA256: String(repeating: "1", count: 64),
            tokenizerSHA256: String(repeating: "2", count: 64), templateSHA256: String(repeating: "3", count: 64),
            quantizationSHA256: String(repeating: "4", count: 64), arithmeticEpoch: "pending-fixture-v1")
        let owner = try PersistentConversation(identity: identity, ledger: ledger, maximumBytes: 2_200_000_000, contextLimit: 8192)
        let state = try SequenceState(inputTokens: 1, outputTokens: 8, ledger: ledger, contextLimit: 8192, pageArena: owner.arena)
        state.emittedButUnconsumed = 43
        owner.discardPendingIfAbsent(state, prompt: [43, 44])
        try require(state.emittedButUnconsumed == 43, "Matching pending output was discarded")
        owner.discardPendingIfAbsent(state, prompt: [44, 43])
        try require(state.emittedButUnconsumed == nil && state.tokenCount == 0, "Mismatching pending output was consumed")
        state.emittedButUnconsumed = 43
        owner.discardPendingIfAbsent(state, prompt: [])
        try require(state.emittedButUnconsumed == nil, "Regeneration inherited a pending output")
        var output = try GeneratedTokenSequence(maximumOutput: 8)
        try require(try !output.recordSample(248046), "EOS unexpectedly became visible output")
        try require(output.ids.isEmpty, "EOS entered consumed output history")
        let suffix = try PrefillReadSchedule(promptTokens: 770, computeTile: 256, readScope: 512, startOffset: 257)
        try require(suffix.parts.map(\.range) == [257..<769, 769..<770], "Append schedule changed absolute offsets")
        let segments = try suffix.checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
        try require(segments.map(\.start) == [257, 513, 769] && segments.last!.keyExtent == 770, "Append timeline has a gap")
        let other = try suffix.checkpointSegments(arithmeticEpoch: "different")
        try require(segments != other, "Append schedule omitted its arithmetic epoch")
        try rejected { _ = try PrefillReadSchedule(promptTokens: 10, computeTile: 256, readScope: 512, startOffset: 10) }
        state.gdn.removeValue(forKey: 0)
        try rejected { try state.validate() }
    }
    private static func shape(context: Int, fileBacked: Bool = false) throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
        let parent = CheckFixtures.scratch
        let geometry = try ConversationGeometry(context: context)
        if fileBacked {
            // The frontier store refuses an image once free disk would drop under 10 GB; a full drive is not a failure.
            let needed = UInt64(geometry.maximumFrontiers + 1) * ConversationFrontierStore.imageBytes + 10_000_000_000
            let free = UInt64(max(0, try parent.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity ?? 0))
            if free < needed {
                throw FixtureUnavailable(
                    description: "needs \(needed / 1_000_000_000) GB free disk in \(parent.path), has \(free / 1_000_000_000) GB")
            }
        }
        var frontierStore =
            fileBacked
            ? try ConversationFrontierStore(
                parent: parent, maximumBytes: geometry.frontierDiskBytes, ledger: ledger, memoryBytes: geometry.frontierMemoryBytes) : nil
        let identity = try CheckpointIdentity(
            modelSHA256: String(repeating: "1", count: 64),
            tokenizerSHA256: String(repeating: "2", count: 64), templateSHA256: String(repeating: "3", count: 64),
            quantizationSHA256: String(repeating: "4", count: 64), arithmeticEpoch: "synthetic-conversation-v1")
        var arena: QSAPageArena? = try QSAPageArena(ledger: ledger, maximumBytes: UInt64(context) * 25_344 + 268_435_456)
        var state: SequenceState? = try SequenceState(
            inputTokens: context - 64, outputTokens: 64,
            ledger: ledger, contextLimit: context, pageArena: arena)
        var images: [ModelCheckpoint] = []
        let boundaryCount = fileBacked ? context / 256 : 8
        let strideTokens = context / boundaryCount
        for boundary in 1...boundaryCount {
            let end = boundary * strideTokens
            let old = state!.tokenCount
            for layer in (0..<48).filter({ $0 % 4 == 3 }) {
                for start in stride(from: old, to: end, by: 256) {
                    let n = min(256, end - start)
                    try state!.qsa[layer]!.append(
                        keys: [UInt16](repeating: UInt16(0x3f00 + boundary), count: n * 512),
                        values: [UInt16](repeating: UInt16(0x4000 + layer), count: n * 512),
                        pooledKeys: [UInt16](repeating: 0x3f80, count: n / 4 * 128), rawTail: [], tokens: n)
                }
            }
            for layer in (0..<48).filter({ $0 % 4 != 3 }) {
                // Distinct nonzero recurrent states, not repeated references to a
                // single zero allocation. Every physical page is touched.
                state!.gdn[layer] = try GDNState(
                    offset: end,
                    recurrent: [Float](repeating: Float(boundary * 48 + layer) / 4096, count: 48 * 128 * 128),
                    convolutionTail: [UInt16](repeating: UInt16(0x3f00 + boundary), count: 3 * 10240))
            }
            state!.tokenCount = end
            state!.history = [17, 17]
            state!.ple.convolution = [UInt16](repeating: UInt16(0x3f00 + boundary), count: 9 * 10240)
            let segments = try PrefillReadSchedule(promptTokens: end, computeTile: 256, readScope: 256)
                .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
            images.append(
                try ModelCheckpoint.capture(
                    state: state!, consumedIDs: [Int64](repeating: 17, count: end),
                    identity: identity, segments: segments, ledger: ledger, frontierStore: frontierStore))
        }
        if fileBacked {
            // Prompt/generated or divergent histories can add two images beyond
            // the ordinary aligned set. Their payload files are populated and
            // charged even when the synthetic fixture shares immutable KV.
            for last in [18, 19] {
                var ids = [Int64](repeating: 17, count: context)
                ids[context - 1] = Int64(last)
                state!.history = [17, Int64(last)]
                images.append(
                    try ModelCheckpoint.capture(
                        state: state!, consumedIDs: ids, identity: identity,
                        segments: images.last!.segments, ledger: ledger, frontierStore: frontierStore))
            }
        }
        try require(images.count == (fileBacked ? geometry.maximumFrontiers : 8), "Actual-shape fixture omitted frontiers")
        for (i, image) in images.enumerated() {
            try image.validate()
            if fileBacked {
                try require(image.gdn.isEmpty && image.reservedBytes < 2_000_000, "File-backed image retained recurrent payload RAM")
                try image.withTensors { name, _, bytes in
                    if name == "gdn.0.recurrent" {
                        let values = bytes.bindMemory(to: Float.self)
                        try require(
                            values.first == Float(min(i + 1, boundaryCount) * 48) / 4096 && values.last == values.first,
                            "Earlier file-backed recurrent state changed")
                    }
                }
            } else {
                try require(
                    image.gdn[0]!.recurrent.first == Float(min(i + 1, boundaryCount) * 48) / 4096, "Earlier recurrent state changed")
            }
        }
        let arenaBefore = arena!.snapshot()
        let fullShapeBytes = UInt64(context) * 25_344
        try require(arenaBefore.liveBytes == fullShapeBytes, "Aligned snapshots duplicated immutable QSA pages")
        // Old complete readers survive an edited branch and its partial tail.
        state = nil
        state = try SequenceState(inputTokens: context - 64, outputTokens: 64, ledger: ledger, contextLimit: context, pageArena: arena)
        var selected: ModelCheckpoint? = images[boundaryCount / 2 - 1]
        try rejected {
            try selected!.restore(
                into: state!, expectedIdentity: identity, expectedSegments: Array(selected!.segments.dropLast()),
                expectedConsumedIDs: selected!.consumedIDs, intent: .appendInput)
        }
        try rejected {
            try selected!.restore(
                into: state!, expectedIdentity: identity, expectedSegments: selected!.segments,
                expectedConsumedIDs: [18] + selected!.consumedIDs.dropFirst(), intent: .appendInput)
        }
        try require(state!.tokenCount == 0, "Refused restore published partial state")
        try selected!.restore(
            into: state!, expectedIdentity: identity, expectedSegments: selected!.segments,
            expectedConsumedIDs: selected!.consumedIDs, intent: .appendInput)
        for layer in (0..<48).filter({ $0 % 4 == 3 }) {
            try state!.qsa[layer]!.append(
                keys: [UInt16](repeating: 0x4123, count: 512),
                values: [UInt16](repeating: 0x4321, count: 512), pooledKeys: [], rawTail: [UInt16](repeating: 0x3f80, count: 128), tokens: 1
            )
        }
        try require(
            images.last!.qsa[3]!.offset == context && state!.qsa[3]!.offset == context / 2 + 1,
            "Branch modified old reader")
        // Complete conservative ledger categories for the shared-page subject.
        // Other payloads are reserved here, not physically allocated.
        let common: UInt64 =
            3_450_000_000 + 14_352_384 + 128_000_000 + 406_061_056 + 120_000_000
            + 2 * (UInt64(context) * 512 * 2 + 16_384)
        let prefill = try ledger.reserve(
            bytes: common + 2_831_155_200 + 2_000_000_000 + 1_904_640_000 + 512_000_000,
            category: "synthetic_other_prefill_owners")
        let prefillReserved = ledger.snapshot().reserved
        try ledger.release(prefill)
        let expertBytes: UInt64 = UInt64(fileBacked ? (context == 8192 ? 2192 : 2128) : (context == 8192 ? 2560 : 2496)) * 2_764_800
        let decode = try ledger.reserve(
            bytes: common + expertBytes + 536_870_912 + 128_000_000,
            category: "synthetic_other_decode_owners")
        let decodeReserved = ledger.snapshot().reserved
        try ledger.release(decode)
        let peak = ledger.snapshot()
        let pagePeak = arena!.snapshot()
        reports.append([
            "context": UInt64(context), "frontiers": UInt64(images.count), "file_backed": fileBacked ? 1 : 0,
            "recurrent_file_bytes": frontierStore?.snapshot().liveBytes ?? 0, "other_payloads_physically_allocated": 0,
            "prefill_total_reserved": prefillReserved, "decode_total_reserved": decodeReserved, "page_live_bytes": pagePeak.liveBytes,
            "page_high_water": pagePeak.highWaterBytes, "ledger_high_water": peak.reservationHighWater,
            "physical_footprint": peak.physicalFootprint, "page_copied_bytes": pagePeak.copiedBytes,
        ])
        state = nil
        images.removeAll()
        // selected intentionally remains a pinned reader until this scope exits;
        // it must keep its pages and reservation alive after index eviction.
        try require(
            arena!.snapshot().liveBytes == UInt64(context / 2) * 25_344,
            "Eviction released live reader or retained unreferenced branch")
        selected = nil
        try require(arena!.snapshot().liveBytes == 0, "Actual-shape final reader leaked pages")
        try require(frontierStore == nil || frontierStore!.snapshot().liveBytes == 0, "Actual-shape final reader leaked files")
        frontierStore = nil
        arena = nil
        try require(ledger.snapshot().reserved == 0, "Actual-shape cleanup leaked reservations")
    }
}
