import Foundation
import OrigamiEngine

/// Package-only fixture access. This file lives in OrigamiEngine so it can seed
/// internal state without adding public production setters. EngineChecks can
/// invoke these package entrypoints.
package enum PrefixReuseSelfChecks {
    private static func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try condition() else { throw EngineError(message) }
    }
    private static func rejects(_ body: () throws -> Void, memory: Bool = false) throws {
        do { try body() } catch {
            if memory { try require((error as? EngineError)?.code == "insufficient_memory", "Memory denial was hidden as a cache miss") }
            return
        }
        throw EngineError("Invalid prefix operation did not reject")
    }
    private static func identity(_ epoch: String = "prefix-fixture-v1") throws -> CheckpointIdentity {
        let hash = String(repeating: "a", count: 64)
        return try CheckpointIdentity(
            modelSHA256: hash, tokenizerSHA256: hash, templateSHA256: hash,
            quantizationSHA256: hash, arithmeticEpoch: epoch)
    }
    private static func blank(_ ledger: MemoryLedger, input: Int) throws -> SequenceState {
        try SequenceState(inputTokens: input, outputTokens: 2, ledger: ledger, contextLimit: input + 2)
    }
    private static func fixture(_ ledger: MemoryLedger, ids: [Int64]) throws -> SequenceState {
        let state = try blank(ledger, input: ids.count)
        state.tokenCount = ids.count
        state.history = Array(([Int64(248_044), 248_044] + ids).suffix(2))
        for layer in 0..<48 {
            if layer % 4 == 3 {
                state.qsa[layer] = try QSAState(
                    offset: ids.count,
                    keys: [UInt16](repeating: UInt16(100 + layer), count: ids.count * 512),
                    values: [UInt16](repeating: UInt16(200 + layer), count: ids.count * 512),
                    pooledKeys: [UInt16](repeating: UInt16(300 + layer), count: ids.count / 4 * 128),
                    rawTail: [UInt16](repeating: UInt16(400 + layer), count: ids.count % 4 * 128))
            } else {
                state.gdn[layer]!.offset = ids.count
                state.gdn[layer]!.recurrent[0] = Float(layer) + 0.25
                state.gdn[layer]!.convolutionTail[0] = UInt16(500 + layer)
            }
        }
        state.ple.convolution[0] = 0x3f80
        try state.validate()
        return state
    }

    /// Saves run in order on the queue, the next request's barrier waits for them, and a cancelled waiter
    /// returns while the save keeps running.
    package static func saveQueueOrderAndBarrier() throws {
        final class Order: @unchecked Sendable {
            private let lock = NSLock()
            private var items: [Int] = []
            func add(_ item: Int) {
                lock.lock()
                items.append(item)
                lock.unlock()
            }
            var values: [Int] {
                lock.lock()
                defer { lock.unlock() }
                return items
            }
        }
        let controller = try PrefixReuseController(identity: identity(), ledger: try MemoryLedger(), memoryBytes: 1)
        let order = Order()
        let started = Date()
        controller.afterSaves {
            Thread.sleep(forTimeInterval: 0.2)
            order.add(1)
        }
        controller.afterSaves { order.add(2) }
        try controller.finishSaves()
        try require(
            order.values == [1, 2] && Date().timeIntervalSince(started) >= 0.15,
            "The save barrier returned before the queued saves ran in order")
        let control = GenerationControl()
        control.cancel()
        controller.afterSaves {
            Thread.sleep(forTimeInterval: 0.4)
            order.add(3)
        }
        var cancelled = false
        do { try controller.drainSaves(cancellation: control) } catch { cancelled = true }
        try require(cancelled && order.values == [1, 2], "A cancelled waiter did not return while the save ran")
        try controller.drainSaves()
        try require(order.values == [1, 2, 3], "A save was lost after a cancelled waiter")
    }

    /// A save that hangs (its drive stopped answering) fails the next request's wait for it with `drive_stalled` at its
    /// bound, and a later wait at once, until it ends; then waits succeed again (ROADMAP R2).
    package static func saveDrainBounded() throws {
        let controller = try PrefixReuseController(identity: identity(), ledger: try MemoryLedger(), memoryBytes: 1)
        let saved = ExpertReadStall.floor
        ExpertReadStall.floor = 0.3
        defer { ExpertReadStall.floor = saved }
        let gate = DispatchSemaphore(value: 0)
        var opened = false
        // Whatever fails below, the hung save returns.
        defer { if !opened { gate.signal() } }
        controller.afterSaves { gate.wait() }
        func code(_ body: () throws -> Void) -> String? {
            do { try body() } catch { return (error as? EngineError)?.code ?? "\(error)" }
            return nil
        }
        var started = Date()
        try require(code { try controller.finishSaves() } == "drive_stalled", "A hung save was waited for past its bound")
        var seconds = Date().timeIntervalSince(started)
        try require(seconds >= 0.3 && seconds < 2, "The hung save's bound took \(seconds) s")
        started = Date()
        try require(code { try controller.drainSaves() } == "drive_stalled", "A later wait for the hung save did not fail")
        seconds = Date().timeIntervalSince(started)
        try require(seconds < 0.5, "A later wait for the hung save took \(seconds) s")
        opened = true
        gate.signal()
        try require(code { try controller.finishSaves() } == nil, "The wait failed after the save ended")
        ExpertReadStall.floor = saved
        try require(PrefixReuseController.saveBound(bytes: 0) == saved, "The save bound's floor")
        try require(PrefixReuseController.saveBound(bytes: 100_000_000_000) == 2_000, "The save bound's multiple")
    }

    // MARK: prompt cache on disk

    private static func image(_ ledger: MemoryLedger, ids: [Int64]) throws -> ModelCheckpoint {
        let identity = try identity()
        let segments = try PrefillReadSchedule(promptTokens: ids.count, computeTile: 256, readScope: 8192)
            .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
        return try ModelCheckpoint.capture(
            state: fixture(ledger, ids: ids), consumedIDs: ids, identity: identity, segments: segments,
            nextLogits: [Float](repeating: 0, count: 248_320), ledger: ledger)
    }
    private static func cache(_ root: URL, _ ledger: MemoryLedger, capacity: Int = 8192, minimumFreeBytes: UInt64 = 0) throws -> PromptCache
    {
        try PromptCache(
            root: root, epoch: "fixture", entries: 1, systemEntries: 0,
            entryBytes: PagedCheckpointFormat.layout(capacity: capacity, tokens: capacity, hasLogits: true).bytes * 2,
            ledger: ledger, capacity: capacity, minimumFreeBytes: minimumFreeBytes)
    }
    private static func folder() throws -> URL {
        let url = CheckFixtures.scratch.appendingPathComponent("prompt-cache-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        return url
    }

    /// A saved prompt that does not fit the conversation's arena (or meets a drive error) stays saved for a later
    /// request; one that fails its checks is dropped.
    /// A saved image holding fast decode rows (left by a server that allowed fast) is never reused by an exact request,
    /// whatever the configuration allows now: the driver passes exactness alone, the disk restore records
    /// `fastTimelineExcluded` and keeps the entry.
    package static func exactRequestSkipsFastImage() throws {
        try require(
            GenerationDriver.exactTimeline(.exact) && !GenerationDriver.exactTimeline(.lateMissSkip),
            "The driver's exact-timeline rule is not exactness alone")
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let cache = try cache(parent.appendingPathComponent("0123456789abcdef"), ledger)
            let ids = (0..<600).map { Int64($0 % 1000 + 1) }
            let identity = try identity()
            let fast = FastLateMissSkip.timelineEpoch(identity.arithmeticEpoch, quality: .lateMissSkip)
            let segments =
                try PrefillReadSchedule(promptTokens: 598, computeTile: 256, readScope: 8192)
                .checkpointSegments(arithmeticEpoch: identity.arithmeticEpoch)
                + [598, 599].map { try PrefillReadSchedule.decodeSegment(start: $0, arithmeticEpoch: fast) }
            try cache.write(
                ModelCheckpoint.capture(
                    state: fixture(ledger, ids: ids), consumedIDs: ids, identity: identity, segments: segments,
                    nextLogits: [Float](repeating: 0, count: 248_320), ledger: ledger))
            let controller = try PrefixReuseController(
                identity: identity, ledger: ledger,
                memoryBytes: try ConversationGeometry(context: 8192).conversationBytes, promptCache: cache, conversationContext: 8192)
            let request = ids + [7]
            let target = try SequenceState(
                inputTokens: request.count, outputTokens: 1, ledger: ledger, contextLimit: 8192, pageArena: controller.conversation!.arena)
            try controller.restoreConversation(requestIDs: request, into: target, exactTimeline: GenerationDriver.exactTimeline(.exact))
            try require(
                target.tokenCount == 0 && controller.statistics.lastDecision == "fastTimelineExcluded" && cache.contains(ids[...]),
                "An exact request restored fast decode rows (\(controller.statistics.lastDecision))")
        }
        try require(ledger.snapshot().reserved == 0, "Fast image fixture leaked a reservation")
    }

    package static func restoreKeepsEntryOnQuota() throws {
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let cache = try cache(parent.appendingPathComponent("0123456789abcdef"), ledger)
            let ids = (0..<600).map { Int64($0 % 1000 + 1) }
            try cache.write(image(ledger, ids: ids))
            let controller = try PrefixReuseController(
                identity: identity(), ledger: ledger,
                memoryBytes: try ConversationGeometry(context: 8192).conversationBytes, promptCache: cache, conversationContext: 8192)
            let arena = controller.conversation!.arena
            // Readers outside the conversation's index hold the arena, so nothing can give way to the saved prompt's pages.
            var held: [QSAPagedVector] = []
            while arena.maximumBytes - arena.snapshot().liveBytes > 16 * 262_144 {
                held.append(try QSAPagedVector(arena: arena, pageElements: 256 * 512).appending([1]))
            }
            let request = ids + [7]
            let target = try SequenceState(
                inputTokens: request.count, outputTokens: 1, ledger: ledger, contextLimit: 8192, pageArena: arena)
            try controller.restoreConversation(requestIDs: request, into: target)
            try require(
                target.tokenCount == 0 && controller.statistics.lastDecision == "diskUnavailable" && cache.contains(ids[...]),
                "A saved prompt that did not fit the arena was dropped as damaged")
            held.removeAll()
            let file = parent.appendingPathComponent("0123456789abcdef/entry-0/checkpoint.origami").path
            let fd = open(file, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0 else { throw EngineError("Saved prompt fixture missing") }
            var byte: UInt8 = 0
            let offset = off_t(PagedCheckpointFormat.manifestBytes)
            let damaged =
                pread(fd, &byte, 1, offset) == 1
                && {
                    byte ^= 0xff
                    return pwrite(fd, &byte, 1, offset) == 1
                }()
            close(fd)
            try require(damaged, "Saved prompt corruption setup failed")
            try controller.restoreConversation(requestIDs: request, into: target)
            try require(
                target.tokenCount == 0 && controller.statistics.lastDecision == "diskVerificationFailed" && !cache.contains(ids[...]),
                "A damaged saved prompt was kept")
        }
        try require(ledger.snapshot().reserved == 0, "Saved prompt restore fixture leaked a reservation")
    }

    /// Entry sizes are the blocks on disk (within the save's estimate), a save into a pool whose slots are all pinned fails instead of passing
    /// as saved, and low disk ends even when another build's folder cannot be removed.
    package static func promptCacheDiskRules() throws {
        final class Attempt: @unchecked Sendable {
            let cache: PromptCache, image: ModelCheckpoint, done = DispatchSemaphore(value: 0)
            var error: Error?
            init(_ cache: PromptCache, _ image: ModelCheckpoint) {
                self.cache = cache
                self.image = image
            }
        }
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        let stuck = parent.appendingPathComponent("fedcba9876543210/locked", isDirectory: true)
        defer {
            chmod(stuck.path, 0o700)
            try? FileManager.default.removeItem(at: parent)
        }
        let root = parent.appendingPathComponent("0123456789abcdef")
        try autoreleasepool {
            // 32K: its key/value slots leave holes APFS keeps (16 MiB or more), so the blocks are well below the file's length.
            let cache = try cache(root, ledger, capacity: 32768)
            let first = (0..<600).map { Int64($0 + 1) }
            let second = first + [5]
            try cache.write(image(ledger, ids: first))
            let layout = try PagedCheckpointFormat.layout(capacity: 32768, tokens: 600, hasLogits: true)
            let sparse = (try? root.resourceValues(forKeys: [.volumeSupportsSparseFilesKey]))?.volumeSupportsSparseFiles == true
            let bytes = cache.entries.first?.bytes ?? 0
            let estimate = PagedCheckpointFormat.diskBytes(layout, at: root)
            try require(
                cache.entries.count == 1 && bytes > 100_000_000 && bytes <= estimate && (!sparse || estimate < layout.bytes / 2),
                "Entry size is not its blocks on disk, or exceeds the save's estimate (\(bytes), estimate \(estimate), length \(layout.bytes))"
            )
            try require(cache.keep(first[...], name: "session"), "Keeping the saved prompt failed")
            try rejects { try cache.write(image(ledger, ids: second)) }
            try require(cache.entries.count == 1 && cache.entries[0].tokens == 600, "A save into a pinned pool changed the pool")
            try FileManager.default.createDirectory(at: stuck, withIntermediateDirectories: true)
            try Data([1]).write(to: stuck.appendingPathComponent("file"))
            try require(chmod(stuck.path, 0o500) == 0, "Locked folder setup failed")
            let low = try Self.cache(root, ledger, capacity: 32768, minimumFreeBytes: UInt64.max / 2)
            _ = low.forget(named: "session")
            let attempt = Attempt(low, try image(ledger, ids: second))
            Thread {
                do { try attempt.cache.write(attempt.image) } catch { attempt.error = error }
                attempt.done.signal()
            }.start()
            try require(attempt.done.wait(timeout: .now() + 120) == .success, "Low disk kept retrying a folder it cannot remove")
            try require(
                (attempt.error as? EngineError)?.description.contains("Too little free disk") == true,
                "Low disk did not refuse the save: \(String(describing: attempt.error))")
        }
        try require(ledger.snapshot().reserved == 0, "Prompt cache disk fixture leaked a reservation")
    }

    /// Another build's folder unused for 14 days goes at start, but not one whose index is there and cannot be read (it may
    /// name or pin prompts): only `origami cache prune` removes that.
    package static func unreadableSiblingIndexKept() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        let unreadable = parent.appendingPathComponent("1111111111111111", isDirectory: true)
        let unused = parent.appendingPathComponent("2222222222222222", isDirectory: true)
        for sibling in [unreadable, unused] { try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: false) }
        let index = unreadable.appendingPathComponent("index.json")
        try Data("{\"layout\": 9}".utf8).write(to: index)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-20 * 86_400)], ofItemAtPath: index.path)
        try autoreleasepool { _ = try cache(parent.appendingPathComponent("0123456789abcdef"), ledger) }
        try require(!FileManager.default.fileExists(atPath: unused.path), "A build unused for 14 days was not removed")
        try require(FileManager.default.fileExists(atPath: index.path), "A build whose index could not be read was removed")
    }

    /// An index that cannot be read is rebuilt from Qwen checkpoint slots too (each slot holds its token IDs): the entry
    /// comes back and matches only its own prompt (the family-slot case: `prompt_cache_unreadable_index_second_server`).
    package static func unreadableIndexFromQwenSlot() throws {
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        let root = parent.appendingPathComponent("0123456789abcdef")
        let ids = (0..<600).map { Int64($0 % 1000 + 1) }
        try autoreleasepool { try cache(root, ledger).write(image(ledger, ids: ids)) }
        try Data("{\"epoch\":".utf8).write(to: root.appendingPathComponent("index.json"))
        try autoreleasepool {
            let reread = try cache(root, ledger)
            try require(
                reread.entries.map(\.tokens) == [600] && reread.contains(ids[...]) && !reread.contains((ids.dropLast() + [7])[...])
                    && FileManager.default.fileExists(atPath: root.appendingPathComponent("index.json.unreadable").path),
                "An unreadable index lost its Qwen slot: \(reread.entries)")
        }
        try require(ledger.snapshot().reserved == 0, "Qwen index rebuild fixture leaked a reservation")
    }

    /// What a cache hit costs on the request's path (`touch`; its durable index write of a few ms goes off the path, and
    /// hits close together share one), on the scratch folder's drive; printed under `measurements`. The queued write
    /// still reaches the disk.
    package static func hitIndexWriteTime() throws {
        let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let cache = try cache(parent.appendingPathComponent("0123456789abcdef"), ledger)
            let ids = (0..<600).map { Int64($0 % 1000 + 1) }
            try cache.write(image(ledger, ids: ids))
            let hits = 40
            let index = cache.root.appendingPathComponent("index.json")
            let before = try Data(contentsOf: index)
            var nanoseconds: UInt64 = 0
            for _ in 0..<hits {
                let start = DispatchTime.now().uptimeNanoseconds
                cache.touch(0)
                nanoseconds += DispatchTime.now().uptimeNanoseconds - start
                Thread.sleep(forTimeInterval: 0.002)  // apart, as hits are: most queue a write of their own
            }
            let seconds = Double(nanoseconds) / 1e9 / Double(hits)
            // The queued write lands: the index on disk holds the entry's new last use.
            var written = false
            for _ in 0..<200 where !written {
                Thread.sleep(forTimeInterval: 0.01)
                written = (try? Data(contentsOf: index)).map { $0 != before } ?? false
            }
            try require(cache.entries.count == 1 && written, "A cache hit's last use never reached the index on disk")
            CheckFixtures.measurements.append(["name": "prompt_cache_hit_index_write", "hits": hits, "seconds_per_hit": seconds])
        }
    }

    /// At start: a killed save's staging file and a context-file folder go, a damaged index cannot address a
    /// slot that does not exist, and a folder without permissions (exFAT, FAT) or on a missing drive is refused.
    package static func promptCacheStartRules() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        let root = parent.appendingPathComponent("0123456789abcdef", isDirectory: true)
        let slot = root.appendingPathComponent("entry-0", isDirectory: true)
        let staging = slot.appendingPathComponent(".checkpoint-killed.tmp")
        try FileManager.default.createDirectory(at: slot, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.createDirectory(at: root.appendingPathComponent("kv-killed"), withIntermediateDirectories: false)
        try Data([1]).write(to: staging)
        let entry = #"{"tokens":600,"prefix":"x","used":0}"#
        try Data(#"{"epoch":"fixture","entries":{"-1":\#(entry),"0":\#(entry)}}"#.utf8).write(to: root.appendingPathComponent("index.json"))
        let cache = try cache(root, ledger)
        try require(
            !FileManager.default.fileExists(atPath: staging.path)
                && !FileManager.default.fileExists(atPath: root.appendingPathComponent("kv-killed").path),
            "Leftovers of a killed process stayed")
        try require(cache.entries.count == 1 && cache.summary.entries == 1, "A damaged index kept an entry for a slot that does not exist")
        // Fewer regular entries (`prompt_cache_entries` 5, then 3) move the system slots: entries no longer in their
        // pool's slots and slot folders past the last slot go.
        let resized = parent.appendingPathComponent("00112233445566ff", isDirectory: true)
        let past = resized.appendingPathComponent("entry-8")
        try FileManager.default.createDirectory(at: past, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let system = #"{"tokens":600,"prefix":"y","used":0,"kind":"system"}"#
        try Data(#"{"epoch":"fixture","entries":{"0":\#(entry),"4":\#(entry),"5":\#(entry),"6":\#(system),"8":\#(system)}}"#.utf8)
            .write(to: resized.appendingPathComponent("index.json"))
        let smaller = try PromptCache(
            root: resized, epoch: "fixture", entries: 3,
            entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2, ledger: ledger,
            capacity: 8192)
        try require(
            smaller.entries.map(\.kind).sorted() == ["prompt", "system"] && !FileManager.default.fileExists(atPath: past.path),
            "A resized prompt cache kept entries outside their pools: \(smaller.entries.map(\.kind))")
        func refused(_ root: URL, _ text: String) throws {
            do {
                _ = try Self.cache(root, ledger)
                throw EngineError("A data folder was accepted: \(root.path)")
            } catch let error as EngineError where error.code == "invalid_configuration" {
                try require(error.description.contains(text), "Refusal does not say what to do: \(error)")
            }
        }
        // A folder others can write serves without saves, and says what to do, instead of refusing to start.
        let open = parent.appendingPathComponent("fedcba9876543210/entry-0", isDirectory: true)
        try FileManager.default.createDirectory(at: open, withIntermediateDirectories: true)
        try require(chmod(open.path, 0o777) == 0, "Open folder setup failed")
        let unsaved = try Self.cache(open.deletingLastPathComponent(), ledger)
        try require(
            unsaved.savesOff?.contains("not private") == true && unsaved.candidates(for: [1, 2, 3]).isEmpty,
            "An open data folder kept saving")
        _ = unsaved.rememberSystemBlock([7, 8, 9])
        try require(
            !FileManager.default.fileExists(atPath: open.deletingLastPathComponent().appendingPathComponent("system-blocks.json").path),
            "An open data folder got system-prompt token IDs")
        let missing = URL(fileURLWithPath: "/Volumes/origami-missing-\(UUID().uuidString)/data", isDirectory: true)
        try refused(missing, "not connected")
        try require(PromptCache.missingDrive(parent) == nil, "A connected folder reads as a missing drive")
    }

    /// The dashboard's saved prompts (`/v1/prompts`): an entry's tokens read back from Qwen's checkpoint and from a family
    /// engine's state header (and refused once the slot holds others), rename to a valid name no other entry has, keep and
    /// release, delete by id with its files (a stale or malformed id refused), and a system prompt's own text found under
    /// any chat template.
    package static func promptCacheEdits() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let root = try folder()
        defer { try? FileManager.default.removeItem(at: root) }
        try autoreleasepool {
            let cache = try PromptCache(
                root: root, epoch: "fixture", entries: 1, systemEntries: 1,
                entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2, ledger: ledger,
                capacity: 8192,
                minimumFreeBytes: 0)
            let qwen = (0..<600).map { Int64($0 + 1) }
            let family = Array(700..<1300)
            func header(_ tokens: [Int], _ folder: URL) throws {
                try JSONEncoder().encode(FamilyStateFile.Header(epoch: "fixture", tokens: tokens, parts: [], logits: 0, sha256: ""))
                    .write(to: FamilyStateFile.headerURL(folder))
            }
            try cache.write(image(ledger, ids: qwen))
            try cache.writeState(tokens: family, kind: "system", bytes: 0) { try header(family, $0) }
            let ids = Dictionary(uniqueKeysWithValues: cache.entries.map { ($0.kind, $0.id) })
            guard let prompt = ids["prompt"], let system = ids["system"] else {
                throw EngineError("Saved entries missing: \(cache.entries)")
            }
            try require(
                try cache.saved(prompt).tokens == qwen.map(Int.init) && cache.saved(system).tokens == family,
                "Saved tokens do not read back")
            try require(try cache.update(system, name: "coder", keep: true).pinned, "Keeping a system prompt failed")
            try rejects { _ = try cache.update(prompt, name: "coder", keep: nil) }
            try rejects { _ = try cache.update(prompt, name: "../x", keep: nil) }
            let kept = try cache.update(prompt, name: "chat", keep: true)
            try require(
                kept.pinned && kept.name == "chat" && cache.entries.first { $0.id == system }?.name == "coder",
                "Rename or keep differs: \(cache.entries)")
            try require(try !cache.update(prompt, name: nil, keep: false).pinned, "Release kept the entry pinned")
            let folder = cache.stateFolder(1)
            try header(Array(1..<601), folder)
            try rejects { _ = try cache.saved(system) }
            for stale in ["1-000000000000", "../1", "", "x", "+" + system, system + "0"] { try rejects { try cache.remove(stale) } }
            try cache.remove(system)
            try require(
                cache.entries.map(\.id) == [prompt] && !FileManager.default.fileExists(atPath: FamilyStateFile.headerURL(folder).path)
                    && FileManager.default.fileExists(atPath: folder.path), "Delete left the entry or its files, or took its folder")
            try rejects { _ = try cache.saved(system) }
        }
        let qwenProbe = "<|im_start|>system\n\u{2}<|im_end|>\n<|im_start|>user\n\u{1}<|im_end|>\n<|im_start|>assistant\n"
        let texts = [
            PromptCache.systemText("<|im_start|>system\nYou code.<|im_end|>\n<|im_start|>user\n", probe: qwenProbe),
            PromptCache.systemText("<|im_start|>system\nYou co", probe: qwenProbe),
            PromptCache.systemText(
                "[gMASK]<sop><|system|>\nBe brief.<|user|>", probe: "[gMASK]<sop><|system|>\n\u{2}<|user|>\n\u{1}<|assistant|>"),
            PromptCache.systemText("plain", probe: "no placeholder"),
        ]
        try require(
            texts.map(\.text) == ["You code.", "You co", "Be brief.", "plain"] && texts.map(\.whole) == [true, false, true, true],
            "System text differs: \(texts)")
        try require(ledger.snapshot().reserved == 0, "Saved prompt edits leaked a reservation")
    }

    /// Each reply saves where its turn starts, its reply and then its prompt; with two kept sessions and an unkept chat
    /// taking turns, every save finds a slot and each kept name moves to its session's newest prompt, at every entry
    /// count `prompt_cache_entries` allows.
    package static func keptSessionsAdvance() throws {
        for context in ConversationGeometry.contexts {
            for count in context <= 8192 ? [PromptCache.defaultEntries] : Array(PromptCache.entryRange) {
                let regular = PromptCache.regularEntries(context: context, entries: count)
                try require(regular == (context <= 8192 ? 8 : count), "Regular entries at \(context): \(regular)")
                var entries: [String: PromptCache.Entry] = [:]
                var read = 0
                var clock = 0.0
                func save(_ prefix: String, _ tokens: Int) throws -> String {
                    guard
                        let slot = PromptCache.slot(
                            for: prefix, tokens: tokens, kind: nil, entries: entries, regular: regular,
                            total: regular + 4, read: read)
                    else { throw EngineError("A regular save found every slot pinned at \(context)") }
                    let key = String(slot)
                    let previous = entries.removeValue(forKey: key)
                    let same = previous?.prefix == prefix
                    clock += 1
                    entries[key] = PromptCache.Entry(
                        tokens: tokens, prefix: prefix, used: clock, name: same ? previous?.name : nil,
                        pinned: same ? previous?.pinned : nil, mark: read)
                    return key
                }
                for turn in 0..<6 {
                    for session in ["a", "b", "chat"] {
                        read += 1000
                        _ = try save("\(session)\(turn)t", 1900 + 100 * turn)
                        _ = try save("\(session)\(turn)r", 2050 + 100 * turn)
                        let prompt = try save("\(session)\(turn)p", 2000 + 100 * turn)
                        guard session != "chat" else { continue }
                        try require(
                            entries[prompt]?.prefix == "\(session)\(turn)p", "A kept session's prompt was evicted before keep at \(context)"
                        )
                        entries = PromptCache.kept(entries, key: prompt, name: session)
                    }
                }
                try require(
                    entries.values.filter { $0.pinned == true }.map(\.prefix).sorted() == ["a5p", "b5p"],
                    "Kept sessions did not advance at \(context), \(count) entries")
            }
        }
    }

    /// Low disk pauses saves visibly: a save that dropping every entry could not make room for drops none, and sets the
    /// status note with the real numbers, logged once; the next save that fits resumes them, logged once.
    /// Records of 4 MiB whose every byte depends on the key and its position (a hot copy's model, for `diskBudgetGiveBack`).
    private final class Records: HotCopySource, @unchecked Sendable {
        static let bytes = 4 << 20
        static func byte(_ key: Int, _ at: Int) -> UInt8 { UInt8(truncatingIfNeeded: key &* 31 &+ at &* 7 &+ 1) }
        func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<40).contains(key) ? Self.bytes : nil }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
            for at in 0..<Self.bytes { into.storeBytes(of: Self.byte(key, at), toByteOffset: at, as: UInt8.self) }
        }
        /// Every record `copy` maps reads back as the model's bytes.
        func check(_ copy: HotCopy) throws -> Bool {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: Self.bytes, alignment: 16384)
            defer { buffer.deallocate() }
            for key in copy.keys {
                try copy.read(key, into: buffer)
                guard buffer.indices.allSatisfy({ buffer[$0] == Self.byte(key, $0) }) else { return false }
            }
            return true
        }
    }

    /// One disk budget per drive (`DiskBudget`): a prompt-cache save short of room with a hot copy on its drive makes the
    /// copy give back exactly the whole records it lacks, a duplicate first, then the least used; the save succeeds, the
    /// copy still serves every other record with the model's bytes (also opened again from disk). A save that fits takes
    /// nothing; without a copy a save short of room pauses as before and counts as skipped. Once the drive has room above
    /// the copy's reserve again, its swaps fill the freed slots.
    package static func diskBudgetGiveBack() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer {
            DiskBudget.shared.freeForChecks = nil
            try? FileManager.default.removeItem(at: parent)
        }
        let floor: UInt64 = 1_000_000_000
        func skipped() -> Int {
            (DiskBudget.shared.report.first { $0["volume"] as? String == DiskBudget.volume(parent) })?["saves_skipped"] as? Int ?? 0
        }
        // No copy on the drive: a save short of room pauses as before, and counts as skipped.
        try autoreleasepool {
            let lone = try PromptCache(
                root: parent.appendingPathComponent("fedcba9876543210"), epoch: "fixture", entries: 2, systemEntries: 0,
                entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2,
                ledger: ledger, capacity: 8192, minimumFreeBytes: floor)
            let before = skipped()
            lone.freeBytesForChecks = { floor / 2 }
            try rejects { try lone.write(try Self.image(ledger, ids: (0..<600).map { Int64($0 + 3) })) }
            try require(lone.savesPaused != nil && skipped() == before + 1, "A save without room was not paused and counted")
        }
        try? FileManager.default.removeItem(at: parent.appendingPathComponent("fedcba9876543210"))
        let records = Records()
        let copyFolder = parent.appendingPathComponent("hot")
        let built = try HotCopy.write(source: records, keys: Array(0..<40), folder: copyFolder, identity: "fixture")
        final class Lines: @unchecked Sendable {
            let lock = NSLock()
            var all: [String] = []
        }
        let lines = Lines()
        // Key k is used (k + 1) times; key 7 is also on a faster drive.
        let keeper = HotCopyKeeper(
            source: records, identity: "fixture", copy: built,
            counts: (Dictionary(uniqueKeysWithValues: (0..<40).map { ($0, $0 + 1) }), [:]),
            place: { _ in .folder(copyFolder, drive: "fixture", budget: UInt64(built.bytes), free: 0) },
            attach: { _ in true }, idle: { true }, event: { line in lines.lock.withLock { lines.all.append(line) } })
        defer { keeper.stop(waiting: 3) }
        keeper.redundant = { [7] }
        try autoreleasepool {
            let cache = try PromptCache(
                root: parent.appendingPathComponent("0123456789abcdef"), epoch: "fixture", entries: 2, systemEntries: 0,
                entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2,
                ledger: ledger, capacity: 8192, minimumFreeBytes: floor)
            let image = try Self.image(ledger, ids: (0..<600).map { Int64($0 + 1) })
            // The save's need as the cache counts it, in its own slot folder; a folder not made yet counts the same.
            let layout = try PagedCheckpointFormat.layout(capacity: 8192, tokens: 600, hasLogits: true)
            let slot = parent.appendingPathComponent("0123456789abcdef/entry-1")
            let bytes = PagedCheckpointFormat.diskBytes(layout, at: slot)
            try require(
                PagedCheckpointFormat.diskBytes(layout, at: slot.appendingPathComponent("not-made-yet")) == bytes,
                "A first save into a folder not made yet counted another need")
            // The drive, all injected: a fixed base plus the slots the copy has emptied (the drive's own block count
            // after a punched hole depends on the volume: local snapshots, shared extents).
            func emptied() -> UInt64 { built.slots.filter { $0.key < 0 }.reduce(UInt64(0)) { $0 + $1.bytes } }
            var base = floor + bytes + 100 << 20
            cache.freeBytesForChecks = { base + emptied() }
            // A save that fits: nothing given back.
            try cache.write(image)
            try require(built.keys.count == 40 && lines.lock.withLock { lines.all.isEmpty }, "A save that fit made the copy give back")
            // 10 MiB short: three 4 MiB records go, the duplicate (7), then the least used (0, 1).
            base = floor + bytes - 10 << 20
            let skippedBefore = skipped()
            func given() -> UInt64 {
                (DiskBudget.shared.report.first { $0["volume"] as? String == DiskBudget.volume(parent) })?["given_back_bytes"] as? UInt64
                    ?? 0
            }
            let givenBefore = given()
            try cache.write(try Self.image(ledger, ids: (0..<600).map { Int64($0 + 9) }))
            try require(
                emptied() == 3 * UInt64(Records.bytes) && Set(built.keys) == Set(0..<40).subtracting([7, 0, 1])
                    && cache.entries.count == 2,
                "The copy did not give back exactly the 3 least-used records: \(emptied()) bytes, \(built.keys.count) kept")
            try require(
                try records.check(built) && records.check(HotCopy.open(folder: copyFolder, identity: "fixture", source: records)!),
                "A record kept after the give-back read other bytes")
            try require(
                lines.lock.withLock { lines.all.count == 1 && lines.all[0].contains("(3 least-used experts) for a prompt-cache save") },
                "The give-back was not said once: \(lines.lock.withLock { lines.all })")
            let volume = DiskBudget.shared.report.first { $0["volume"] as? String == DiskBudget.volume(parent) }
            try require(
                given() - givenBefore == 3 * UInt64(Records.bytes)
                    && volume?["saves_skipped"] as? Int == skippedBefore
                    && keeper.report["given_back_records"] as? Int == 3,
                "The disk budget did not show the give-back: \(String(describing: volume))")
            // Space returns: the copy's swaps fill the three slots again (none while the drive lacks room above its reserve).
            DiskBudget.shared.freeForChecks = { _ in 0 }
            keeper.trigger()
            Thread.sleep(forTimeInterval: 0.5)
            try require(built.keys.count == 37, "The copy grew back into space a save had just needed")
            DiskBudget.shared.freeForChecks = { _ in 100_000_000_000 }
            keeper.trigger()
            let end = Date().addingTimeInterval(10)
            while built.keys.count < 40, Date() < end { Thread.sleep(forTimeInterval: 0.05) }
            try require(
                Set(built.keys) == Set(0..<40) && emptied() == 0 && (try records.check(built)),
                "The copy did not grow back once space returned: \(built.keys.count) records")
        }
        try require(ledger.snapshot().reserved == 0, "Disk-budget fixture leaked a reservation")
    }

    /// The restore account (`PromptCache.explainRestore`): for each reason a request restored fewer tokens than a save could
    /// have given, the reason and the tokens wanted; none when it restored all a save holds.
    package static func restoreReasons() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let cache = try PromptCache(
                root: parent.appendingPathComponent("0123456789abcdef"), epoch: "fixture", entries: 3, systemEntries: 0,
                entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2,
                ledger: ledger, capacity: 8192, minimumFreeBytes: 0)
            func ids(_ from: Int, _ count: Int = 600) -> [Int] { (0..<count).map { $0 + from } }
            func reason(_ request: [Int], restored: Int) -> (wanted: Int, reason: String?) {
                cache.explainRestore(request, restored: restored)
                let row = cache.takeRestore()
                return (row?["wanted"] as? Int ?? -1, row?["reason"] as? String)
            }
            // No entry: the previous request's prompt was never saved.
            _ = reason(ids(1), restored: 0)
            let none = reason(ids(1, 700), restored: 0)
            try require(none.wanted == 600 && none.reason == "no_entry", "no entry: \(none)")
            // A saved entry the request extends, restored whole: nothing to explain; not restored: entry_not_used.
            try cache.write(image(ledger, ids: ids(1).map(Int64.init)))
            let whole = reason(ids(1) + ids(9_000, 100), restored: 600)
            try require(whole.wanted == 600 && whole.reason == nil, "restored whole: \(whole)")
            let unused = reason(ids(1) + ids(9_100, 100), restored: 0)
            try require(unused.wanted == 600 && unused.reason == "entry_not_used", "entry not used: \(unused)")
            // A different prefix: the request leaves the saved tokens at 300.
            let differs = reason(ids(1, 300) + ids(5_000, 400), restored: 0)
            try require(differs.wanted == 300 && differs.reason == "prefix_differs_at=300", "different prefix: \(differs)")
            // The entry evicted.
            cache.forget(cache.candidates(for: ids(1).map(Int64.init)).first!.slot)
            let evicted = reason(ids(1) + ids(9_200, 100), restored: 0)
            try require(evicted.wanted == 600 && evicted.reason == "entry_evicted", "entry evicted: \(evicted)")
            // A save skipped for space.
            cache.freeBytesForChecks = { 0 }
            try rejects { try cache.write(image(ledger, ids: ids(20_000).map(Int64.init))) }
            let skipped = reason(ids(20_000) + ids(9_300, 100), restored: 0)
            try require(skipped.wanted == 600 && skipped.reason == "save_skipped_disk", "save skipped: \(skipped)")
            // A save still running when the request looks.
            let gate = DispatchSemaphore(value: 0)
            let entered = DispatchSemaphore(value: 0)
            cache.freeBytesForChecks = {
                entered.signal()
                gate.wait()
                return .max / 4
            }
            let save = ids(30_000).map(Int64.init)
            nonisolated(unsafe) let late = try image(ledger, ids: save)
            nonisolated(unsafe) let pending = cache
            let done = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                _ = try? pending.write(late)
                done.signal()
            }
            entered.wait()
            let running = reason(ids(30_000) + ids(9_400, 100), restored: 0)
            gate.signal()
            while done.wait(timeout: .now() + 0.01) == .timedOut { gate.signal() }
            try require(running.wanted == 600 && running.reason == "save_not_finished", "save not finished: \(running)")
            let counted = cache.report["restore_fallbacks"] as? [String: Int] ?? [:]
            try require(
                counted == [
                    "no_entry": 1, "entry_not_used": 1, "prefix_differs": 1, "entry_evicted": 1, "save_skipped_disk": 1,
                    "save_not_finished": 1,
                ] && cache.report["restore_lost_tokens"] as? Int == 600 * 5 + 300,
                "restore counters: \(counted) \(String(describing: cache.report["restore_lost_tokens"]))")
        }
        try require(ledger.snapshot().reserved == 0, "Restore-reason fixture leaked a reservation")
    }

    package static func savesPauseOnLowDisk() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let cache = try PromptCache(
                root: parent.appendingPathComponent("0123456789abcdef"), epoch: "fixture", entries: 2, systemEntries: 0,
                entryBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2,
                ledger: ledger, capacity: 8192, minimumFreeBytes: 10_000_000_000)
            var lines: [String] = []
            var free: UInt64 = 1_000_000_000_000
            cache.log = { lines.append($0) }
            cache.freeBytesForChecks = { free }
            try cache.write(image(ledger, ids: (0..<600).map { Int64($0 + 1) }))
            free = 4_000_000_000
            for offset in [7, 9] { try rejects { try cache.write(image(ledger, ids: (0..<600).map { Int64($0 + offset) })) } }
            let note = cache.report["saves_paused"] as? String ?? ""
            try require(cache.entries.count == 1, "A save that could not fit dropped the saved prompt")
            try require(
                note.hasPrefix("conversation saves are paused: 4.0 GB free, and a save keeps 10.0 GB free plus its image (")
                    && lines == ["prompt cache: " + note], "Paused saves were not said once with their numbers: \(lines)")
            free = 1_000_000_000_000
            try cache.write(image(ledger, ids: (0..<600).map { Int64($0 + 7) }))
            try require(
                cache.savesPaused == nil && cache.report["saves_paused"] is NSNull
                    && lines.last == "prompt cache: conversation saves resumed" && lines.count == 2, "Resumed saves were not said: \(lines)"
            )
        }
        try require(ledger.snapshot().reserved == 0, "Paused-save fixture leaked a reservation")
    }

    /// A save clones the image the cache wrote last (`base`) when that image is still the one this process wrote, so a
    /// reply's prompt image after its conversation image writes only the pages that differ, and reads back exactly;
    /// a base rewritten since, or one in another capacity, is not used. `measurements` (`prompt_cache_clone_saves`)
    /// reports the bytes and seconds of the same save without and with the base.
    /// A save that fails before publishing (here: no room) keeps the entry and image it would have replaced.
    package static func failedSaveKeepsPrevious() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        let root = parent.appendingPathComponent("0123456789abcdef")
        try autoreleasepool {
            let first = (0..<600).map { Int64($0 + 1) }
            let other = (0..<600).map { Int64($0 + 7) }
            try cache(root, ledger).write(image(ledger, ids: first))
            let low = try cache(root, ledger, minimumFreeBytes: UInt64.max / 2)
            try rejects { try low.write(image(ledger, ids: other)) }
            try require(low.entries.count == 1 && low.entries[0].tokens == 600, "A failed save dropped the entry it would have replaced")
            let segments = try PrefillReadSchedule(promptTokens: first.count, computeTile: 256, readScope: 8192)
                .checkpointSegments(arithmeticEpoch: identity().arithmeticEpoch)
            let store = try PersistentCheckpointStore(
                directory: low.stateFolder(0),
                maximumBytes: PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2, ledger: ledger,
                capacity: 8192)
            _ = try store.read(expectedIdentity: identity(), expectedSegments: segments, expectedConsumedIDs: first)
        }
        try require(ledger.snapshot().reserved == 0, "Failed-save fixture leaked a reservation")
    }

    package static func cloneBaseSaves() throws {
        let ledger = try MemoryLedger(ceiling: 3_000_000_000, transientMargin: 0)
        let parent = try folder()
        defer { try? FileManager.default.removeItem(at: parent) }
        try autoreleasepool {
            let bytes = try PagedCheckpointFormat.layout(capacity: 8192, tokens: 8192, hasLogits: true).bytes * 2
            func store(_ name: String) throws -> PersistentCheckpointStore {
                try PersistentCheckpointStore(
                    directory: parent.appendingPathComponent(name), maximumBytes: bytes, ledger: ledger, capacity: 8192)
            }
            let conversation = (0..<8000).map { Int64($0 % 997 + 1) }
            let prompt = Array(conversation.prefix(7680))
            let base = try store("base")
            let plain = try store("plain")
            let cloned = try store("cloned")
            try base.write(image(ledger, ids: conversation))
            let promptImage = try image(ledger, ids: prompt)
            let expected = try promptImage.tensorSHA256()
            var start = DispatchTime.now().uptimeNanoseconds
            try plain.write(promptImage)
            let plainSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            start = DispatchTime.now().uptimeNanoseconds
            try cloned.write(promptImage, base: base)
            let clonedSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let full = plain.lastWriteStatistics!
            let delta = cloned.lastWriteStatistics!
            try require(
                !full.cloned && delta.cloned && delta.unchangedPages > 0 && delta.writtenBytes * 2 < full.writtenBytes,
                "A save after its conversation rewrote its shared pages: \(delta.writtenBytes) of \(full.writtenBytes) bytes")
            let segments = try PrefillReadSchedule(promptTokens: prompt.count, computeTile: 256, readScope: 8192)
                .checkpointSegments(arithmeticEpoch: identity().arithmeticEpoch)
            let restored = try cloned.read(expectedIdentity: identity(), expectedSegments: segments, expectedConsumedIDs: prompt)
            try require(try restored.tensorSHA256() == expected, "A cloned save reads back other tensors")
            // A base whose file was replaced behind the store is not trusted as a clone source.
            try FileManager.default.removeItem(at: base.checkpointURL)
            try FileManager.default.copyItem(at: plain.checkpointURL, to: base.checkpointURL)
            let again = try store("again")
            try again.write(promptImage, base: base)
            try require(again.lastWriteStatistics?.cloned == false, "A replaced base was cloned without verification")
            CheckFixtures.measurements.append([
                "name": "prompt_cache_clone_saves", "image_bytes": full.logicalBytes, "written_bytes_without_base": full.writtenBytes,
                "written_bytes_with_base": delta.writtenBytes, "seconds_without_base": plainSeconds, "seconds_with_base": clonedSeconds,
            ])
        }
        try require(ledger.snapshot().reserved == 0, "Clone-base fixture leaked a reservation")
    }
}
