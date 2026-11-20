import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

/// Model-free checks of the paged-v2 format and store: maximum geometry, clone-and-delta publication,
/// injected faults at each publication boundary, cancellation, corruption and truncation.
package enum PagedCheckpointSelfChecks {
    private static func require(_ condition: Bool, _ message: String) throws { if !condition { throw EngineError(message) } }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return } catch is IOOperationCancelled { return }
        throw EngineError("Paged checkpoint check expected refusal")
    }
    private static func cancelled(_ body: () throws -> Void) throws {
        do { try body() } catch is IOOperationCancelled { return }
        throw EngineError("Paged checkpoint check expected cancellation")
    }
    package static func maximumFormat() throws {
        let maximum = try PagedCheckpointFormat.layout(capacity: 262144, tokens: 262144, hasLogits: true)
        try require(
            maximum.bytes * 2 <= ConversationGeometry.durableDiskBytes && maximum.sections.count == 124,
            "Maximum paged layout exceeds quota or changed tensor schema")
        let earlier = try PagedCheckpointFormat.layout(capacity: 262144, tokens: 262143, hasLogits: false)
        try require(
            zip(maximum.sections, earlier.sections).allSatisfy { $0.offset == $1.offset } && maximum.bytes == earlier.bytes,
            "Paged layout shifted unchanged tensors")
        let ledger = try MemoryLedger(ceiling: 700_000_000, transientMargin: 0)
        try autoreleasepool {
            let memory = try CheckpointMemoryLease(ledger: ledger, bytes: UInt64(2 * 262144 * 1024) + PagedCheckpointFormat.scratchBytes)
            defer { withExtendedLifetime(memory) {} }
            let hash = String(repeating: "c", count: 64)
            let segments = try (0..<262144).map {
                try CheckpointSegment(
                    start: $0, count: 1, projectionRows: 1, queryTile: 1,
                    keyExtent: $0 + 1, phase: .decode, dispatchSHA256: hash)
            }
            let encoded = try PagedCheckpointFormat.timeline(segments)
            let decoded = try PagedCheckpointFormat.decodeTimeline(encoded)
            try require(encoded.count == 262144 * 44 && decoded == segments, "Maximum binary producing timeline changed")
        }
        try require(ledger.snapshot().reserved == 0, "Maximum timeline admission leaked")
    }
    /// A save needs the disk its pages take, not the capacity-sized file: a 600-token image of the 256K tier asks for
    /// about 165 MB where the volume keeps sparse files (the whole 6.8 GB where it does not); at 8K the gaps between its
    /// pages are under 16 MiB, which APFS fills, so it asks for the whole file. `prompt_cache_disk_rules` compares the
    /// estimate with a written entry's blocks.
    package static func diskBytesOfWrittenPages() throws {
        let layout = try PagedCheckpointFormat.layout(capacity: 262144, tokens: 600, hasLogits: true)
        let small = try PagedCheckpointFormat.layout(capacity: 8192, tokens: 600, hasLogits: true)
        let scratch = CheckFixtures.scratch
        let needed = PagedCheckpointFormat.diskBytes(layout, at: scratch)
        let sparse = (try? scratch.resourceValues(forKeys: [.volumeSupportsSparseFilesKey]))?.volumeSupportsSparseFiles == true
        try require(
            layout.bytes > 6_000_000_000 && PagedCheckpointFormat.diskBytes(small, at: scratch) == small.bytes
                && (sparse ? needed < 200_000_000 && needed > 130_000_000 : needed == layout.bytes),
            "Save size differs: \(needed) of \(layout.bytes) (sparse \(sparse))")
    }
    package static func cloneDeltaAndRollback() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 16_000_000)
        let parent = CheckFixtures.scratch
        let root = parent.appendingPathComponent("paged-checkpoint-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: root) }
        try autoreleasepool {
            let identity = try CheckpointIdentity(
                modelSHA256: String(repeating: "1", count: 64), tokenizerSHA256: String(repeating: "2", count: 64),
                templateSHA256: String(repeating: "3", count: 64), quantizationSHA256: String(repeating: "4", count: 64),
                arithmeticEpoch: "paged-checkpoint-fixture-v2")
            let sourceStore = try QSAFilePageStore(parent: root, diskBytes: 256 * 1024 * 1024, ledger: ledger)
            let arena = try QSAPageArena(ledger: ledger, maximumBytes: 8_000_000, fileStore: sourceStore)
            let state = try SequenceState(inputTokens: 513, outputTokens: 4, ledger: ledger, contextLimit: 8192, pageArena: arena)
            for layer in 0..<48 {
                if layer % 4 == 3 {
                    for start in stride(from: 0, to: 513, by: 256) {
                        let count = min(256, 513 - start)
                        let end = start + count
                        try state.qsa[layer]!.append(
                            keys: (0..<count * 512).map { UInt16(truncatingIfNeeded: layer + start * 512 + $0) },
                            values: (0..<count * 512).map { UInt16(truncatingIfNeeded: layer * 7 + start * 512 + $0) },
                            pooledKeys: [UInt16](repeating: UInt16(layer), count: (end / 4 - start / 4) * 128),
                            rawTail: [UInt16](repeating: 0x8000, count: end % 4 * 128), tokens: count)
                    }
                } else {
                    state.gdn[layer] = try GDNState(
                        offset: 513, recurrent: [Float](repeating: Float(layer), count: 48 * 128 * 128),
                        convolutionTail: [UInt16](repeating: UInt16(layer), count: 3 * 10240))
                }
            }
            state.tokenCount = 513
            state.history = [17, 17]
            var ids = [Int64](repeating: 17, count: 513)
            var segments = try PrefillReadSchedule(promptTokens: 513, computeTile: 256, readScope: 512).checkpointSegments(
                arithmeticEpoch: identity.arithmeticEpoch)
            let logits = [Float](repeating: 0.125, count: 248320)
            func capture() throws -> ModelCheckpoint {
                try ModelCheckpoint.capture(
                    state: state, consumedIDs: ids, identity: identity, segments: segments, nextLogits: logits, ledger: ledger)
            }
            let first = try capture()
            let expected = try first.tensorSHA256()
            let location = root.appendingPathComponent("paged")
            let disk = try PersistentCheckpointStore(directory: location, maximumBytes: 1_100_000_000, ledger: ledger, capacity: 8192)
            let digest = try disk.write(first)
            let oldFile = try ExactFile(url: disk.checkpointURL)
            let oldManifest = try PagedCheckpointFormat.load(oldFile, capacity: 8192, maximumBytes: disk.maximumBytes)
            let oldEncoded = try oldFile.read(offset: 0, length: oldManifest.bytes)
            try require(
                disk.lastWriteStatistics?.cloned == false && disk.lastWriteStatistics?.unchangedPages == 0,
                "First paged image was not complete")
            let restored = try disk.read(expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: arena)
            try require(try restored.tensorSHA256() == expected, "Paged image differs from the captured tensors")
            let beforeIdentical = sourceStore.snapshot().readBytes
            try require(try disk.write(first) == digest, "Identical image changed manifest digest")
            let identical = disk.lastWriteStatistics!
            try require(
                identical.cloned && identical.changedPages == 0 && identical.verificationReadBytes == 0 && identical.writtenBytes < 65536,
                "Identical paged image rewrote payload or reread durable bytes")
            try require(sourceStore.snapshot().readBytes == beforeIdentical, "Identical publication reread file-KV pages")
            // Real appended QSA page identities: only IDs, K/V tails and raw
            // indexer tails change; recurrent test values deliberately do not.
            for layer in 0..<48 {
                if layer % 4 == 3 {
                    try state.qsa[layer]!.append(
                        keys: [UInt16](repeating: UInt16(layer), count: 512),
                        values: [UInt16](repeating: UInt16(layer + 1), count: 512), pooledKeys: [],
                        rawTail: [UInt16](repeating: 0x8000, count: 2 * 128), tokens: 1)
                } else {
                    let previous = state.gdn[layer]!
                    state.gdn[layer] = try GDNState(offset: 514, recurrent: previous.recurrent, convolutionTail: previous.convolutionTail)
                }
            }
            state.tokenCount = 514
            ids.append(17)
            segments.append(
                try CheckpointSegment(
                    start: 513, count: 1, projectionRows: 1, queryTile: 1, keyExtent: 514,
                    phase: .prefill, dispatchSHA256: String(repeating: "a", count: 64)))
            let append = try capture()
            try disk.write(append)
            let appended = disk.lastWriteStatistics!
            try require(
                appended.cloned && appended.changedPages == 37 && appended.writtenBytes < 200_000 && appended.verificationReadBytes == 0,
                "One-token append rewrote unchanged payload")
            // The old descriptor still returns its original immutable manifest
            // and payload after two later atomic heads replace its pathname.
            // Replacing the pathname legitimately changes the detached old
            // inode's ctime/nlink. Production committed-head checks stay strict;
            // this escaped-reader check verifies every immutable byte by root.
            try require(try oldFile.read(offset: 0, length: oldManifest.bytes) == oldEncoded, "APFS clone mutated old manifest")
            let oldScratch = try CheckpointPageScratch(ledger: ledger)
            for (section, spec) in oldManifest.header.sections.enumerated() {
                for index in 0..<spec.pages {
                    let count = min(PagedCheckpointFormat.pageBytes, Int(spec.length) - index * PagedCheckpointFormat.pageBytes)
                    let bytes = oldScratch.buffer(count)
                    try oldFile.read(offset: Int64(spec.offset) + Int64(index * PagedCheckpointFormat.pageBytes), into: bytes)
                    try require(
                        PagedCheckpointFormat.root(UnsafeRawBufferPointer(bytes)) == oldManifest.root(section: section, page: index),
                        "APFS clone mutated retained old payload")
                }
            }
            let oldSection = oldManifest.header.sections.first { $0.name == "qsa.3.keys" }!
            let oldBytes = try oldFile.read(offset: Int64(oldSection.offset + 2 * UInt64(PagedCheckpointFormat.pageBytes)), length: 1024)
            try require(
                oldBytes.withUnsafeBytes { PagedCheckpointFormat.root($0) } == oldManifest.root(section: 75, page: 2),
                "APFS clone changed old-reader KV tail")
            let appendedIdentity = try ExactFile(url: disk.checkpointURL).identity
            for point in ["staged", "page_written", "before_fsync", "before_rename"] {
                var reached = false
                disk.pagedFault = {
                    if $0 == point {
                        reached = true
                        throw EngineError("injected pre-publication failure")
                    }
                }
                try rejected { try disk.write(first) }
                disk.pagedFault = nil
                try require(reached, "Injected boundary was never reached: \(point)")
                try require(try ExactFile(url: disk.checkpointURL).identity == appendedIdentity, "Failed \(point) replaced old head")
            }
            var calls = 0
            try cancelled {
                try disk.writePaged(
                    first,
                    shouldContinue: {
                        calls += 1
                        return calls < 40
                    })
            }
            try require(try ExactFile(url: disk.checkpointURL).identity == appendedIdentity, "Cancelled publication replaced old head")
            let expectedAppend = try append.tensorSHA256()
            let beforeRead = ledger.snapshot().reserved
            let beforePages = sourceStore.snapshot().livePages
            calls = 0
            try cancelled {
                _ = try disk.readPaged(
                    expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids,
                    pageArena: arena,
                    shouldContinue: {
                        calls += 1
                        return calls < 550
                    })
            }
            try require(
                ledger.snapshot().reserved == beforeRead && sourceStore.snapshot().livePages == beforePages,
                "Cancelled streamed read leaked pages or admission")
            // A fresh store must prove the old payload before its first delta.
            let cold = try PersistentCheckpointStore(directory: location, maximumBytes: disk.maximumBytes, ledger: ledger, capacity: 8192)
            try cold.write(append)
            try require(
                cold.lastWriteStatistics!.verificationReadBytes > 120_000_000 && cold.lastWriteStatistics!.changedPages == 0,
                "Fresh instance trusted unverified old payload")
            // A divergent edit changes one earlier page and one token-ID page.
            var editedKeys = try state.qsa[3]!.keys
            editedKeys[0] ^= 1
            let previousQSA = state.qsa[3]!
            state.qsa[3] = try QSAState(
                offset: 514, keys: editedKeys, values: previousQSA.values,
                pooledKeys: previousQSA.pooledKeys, rawTail: previousQSA.rawTail
            ).usingPages(arena)
            ids[0] = 18
            let edit = try capture()
            try cold.write(edit)
            try require(
                cold.lastWriteStatistics!.changedPages == 2 && cold.lastWriteStatistics!.writtenBytes < 400_000,
                "Divergent edit rewrote unchanged pages")
            let editRestored = try cold.read(
                expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: arena)
            try require(try editRestored.tensorSHA256() == edit.tensorSHA256(), "Edited paged image changed bytes")
            cold.pagedFault = { if $0 == "after_rename" { throw EngineError("injected post-rename failure") } }
            try rejected { try cold.write(append) }
            cold.pagedFault = nil
            try require(
                cold.lastWriteStatistics?.publicationState == "renamed_directory_sync_unconfirmed",
                "Post-rename error falsely claimed rollback/durability")
            ids[0] = 17
            let afterRename = try cold.read(
                expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: arena)
            try require(try afterRename.tensorSHA256() == expectedAppend, "Post-rename error hid actually published head")
            // An 8K/32K restart publishes resident shared pages directly,
            // with no contiguous full KV image in the returned checkpoint.
            try autoreleasepool {
                let ram = try QSAPageArena(ledger: ledger, maximumBytes: 40_000_000)
                let resident = try cold.read(
                    expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: ram)
                try require(
                    try resident.tensorSHA256() == expectedAppend && resident.qsa.values.allSatisfy { $0.pages != nil },
                    "Resident paged restart changed tensors or flattened KV")
            }
            let hint = try cold.peekPaged()
            try require(
                hint.version == 2 && hint.segments == segments && hint.frontier == append.frontier, "Paged restart hint changed timeline")
            var mismatch = ids
            mismatch[0] = 22
            try rejected {
                _ = try cold.read(expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: mismatch, pageArena: arena)
            }
            let mutationFD = cold.checkpointURL.withUnsafeFileSystemRepresentation { open($0!, O_RDWR | O_NOFOLLOW | O_CLOEXEC) }
            guard mutationFD >= 0 else { throw EngineError("Paged manifest fault setup failed") }
            defer { close(mutationFD) }
            var original: UInt8 = 0
            try require(pread(mutationFD, &original, 1, 32) == 1, "Paged manifest digest setup failed")
            var damaged = original ^ 1
            try require(pwrite(mutationFD, &damaged, 1, 32) == 1, "Paged manifest digest corruption failed")
            try rejected { _ = try cold.peekPaged() }
            try require(pwrite(mutationFD, &original, 1, 32) == 1, "Paged manifest digest restoration failed")
            // A digest-correct file changing after clone must be rejected at
            // final publication even when every new payload page is unchanged.
            let mutationHead = try ExactFile(url: cold.checkpointURL).identity
            cold.pagedFault = { point in
                if point == "before_rename" {
                    guard pwrite(mutationFD, &damaged, 1, 32) == 1 else { throw EngineError("Mutation injection failed") }
                }
            }
            try rejected { try cold.write(append) }
            cold.pagedFault = nil
            try require(try ExactFile(url: cold.checkpointURL).identity.inode == mutationHead.inode, "Mutated source allowed a new head")
            try require(pwrite(mutationFD, &original, 1, 32) == 1, "Mutated source restoration failed")
            // Corrupt a late KV page: read must retire all partially loaded
            // pages and keep prior valid readers unchanged. The writer must not
            // clone it, even though its previous manifest is cached: a complete
            // fresh image replaces the corrupt head (likewise a truncated one).
            func replacesCorruptHead() throws {
                try cold.write(append)
                try require(
                    cold.lastWriteStatistics?.cloned == false && cold.lastWriteStatistics?.publicationState == "durable",
                    "Corrupt head was not replaced by a complete fresh image")
                try autoreleasepool {
                    let recovered = try cold.read(
                        expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: arena)
                    try require(try recovered.tensorSHA256() == expectedAppend, "Fresh replacement image does not verify")
                }
            }
            let beforeCorrupt = ledger.snapshot().reserved
            let pagesBeforeCorrupt = sourceStore.snapshot().livePages
            let head = try ExactFile(url: cold.checkpointURL)
            let manifest = try PagedCheckpointFormat.load(head, capacity: 8192, maximumBytes: cold.maximumBytes)
            let late = manifest.header.sections.first { $0.name == "qsa.47.values" }!
            let fd = cold.checkpointURL.withUnsafeFileSystemRepresentation { open($0!, O_RDWR | O_NOFOLLOW | O_CLOEXEC) }
            guard fd >= 0 else { throw EngineError("Paged corruption setup failed") }
            defer { close(fd) }
            var byte: UInt8 = 0
            try require(pread(fd, &byte, 1, off_t(late.offset)) == 1, "Paged fault read failed")
            byte ^= 0xff
            try require(pwrite(fd, &byte, 1, off_t(late.offset)) == 1, "Paged fault write failed")
            try rejected {
                _ = try cold.read(expectedIdentity: identity, expectedSegments: segments, expectedConsumedIDs: ids, pageArena: arena)
            }
            try require(
                ledger.snapshot().reserved == beforeCorrupt && sourceStore.snapshot().livePages == pagesBeforeCorrupt,
                "Corrupt paged read leaked partial state")
            try replacesCorruptHead()
            try require(try restored.tensorSHA256() == expected, "Later corrupt publication changed retained reader")
            let replacedFD = cold.checkpointURL.withUnsafeFileSystemRepresentation { open($0!, O_RDWR | O_NOFOLLOW | O_CLOEXEC) }
            guard replacedFD >= 0 else { throw EngineError("Paged truncation setup failed") }
            defer { close(replacedFD) }
            try require(ftruncate(replacedFD, off_t(head.identity.size - 1)) == 0, "Paged truncation setup failed")
            try rejected { _ = try cold.peekPaged() }
            try replacesCorruptHead()
            let leftovers = try FileManager.default.contentsOfDirectory(atPath: location.path).filter { $0.hasSuffix(".tmp") }
            try require(leftovers.isEmpty, "Failed clone publications left owned staging files")
        }
        try require(ledger.snapshot().reserved == 0, "Paged checkpoint final owners leaked")
    }
}
