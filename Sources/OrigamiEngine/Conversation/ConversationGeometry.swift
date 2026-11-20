import Foundation
import Metal

/// The selected conversation representations and their simultaneous owners.
/// These are admission bounds, never a physical-host or quality qualification.
/// Per token the QSA layers keep 25,344 bytes (12 layers x (512 key + 512 value) bf16 + the pooled index), so an
/// in-memory arena holds one full context plus 256 MiB of divergence; the file tier keeps only 768 bytes per token
/// in memory and its key/value pages in an unlinked file sized for two complete histories.
package struct ConversationGeometry {
    package static let contexts = [8192, 32768, 65536, 131072, 262144]
    /// Tiers that always keep every key/value page in memory; 262,144 keeps them in a file unless planned in memory.
    package static let inMemoryContexts = [8192, 32768, 65536, 131072]
    package static func inMemory(_ context: Int, fileBacked: Bool) -> Bool { contexts.contains(context) && !fileBacked }
    package static let maximumFileContext = 262144
    // Two complete divergent histories and one R8192 growth/COW admission.
    package static let fileDiskBytes: UInt64 = 2 * 262144 * 24576 + 33 * 24 * 262144
    package static let durableDiskBytes: UInt64 = 14_000_000_000
    package static let aggregateDiskLimit: UInt64 = 64 * 1024 * 1024 * 1024
    package let context: Int
    /// 8K and 32K keep their fixed 2.2 GB; a longer in-memory context adds its pages to the same 1.1 GB for the rest.
    package var conversationBytes: UInt64 { fileBacked ? 2_900_000_000 : max(2_200_000_000, arenaBytes + 1_100_000_000) }
    package var maximumFrontiers: Int {
        fileBacked
            ? Int(
                (Self.aggregateDiskLimit - Self.fileDiskBytes - Self.durableDiskBytes)
                    / ConversationFrontierStore.imageBytes) - 1 : min(context / 256 + 2, 130)  // 32K's 15 GB of frontiers at most
    }
    package var frontierDiskBytes: UInt64 { UInt64(maximumFrontiers + 1) * ConversationFrontierStore.imageBytes }
    /// The key/value pages live in a file: the 262,144 tier unless it was planned in memory.
    package let fileBacked: Bool
    package var arenaBytes: UInt64 { UInt64(context) * (fileBacked ? 768 : 25_344) + 268_435_456 }
    package var fileMetadataBytes: UInt64 {
        get throws { fileBacked ? try QSAFilePageStore.layout(diskBytes: Self.fileDiskBytes).metadataBytes : 0 }
    }
    package var stagingBytes: UInt64 { fileBacked ? QSAFileStaging.admittedBytes : 0 }
    package var frontierMemoryBytes: UInt64 {
        get throws {
            let residual = conversationBytes - arenaBytes - (try fileMetadataBytes) - stagingBytes - 120_000_000
            return min(residual, UInt64(maximumFrontiers) * (UInt64(context) * 1024 + 4 * 1024 * 1024))
        }
    }
    package var pooledBytesPerToken: UInt64 { fileBacked ? 768 : 25_344 }
    /// The arena charge a request of `growth` more tokens can reach beside `live` page bytes: its pages with
    /// admitGrowth's extra page set, and the arena's 256 MiB of divergence. The rest is lent to the expert cache.
    package func heldArenaBytes(live: UInt64, growth: Int) -> UInt64 {
        min(arenaBytes, live + UInt64(max(0, growth) / 256 + 2) * 256 * pooledBytesPerToken + 268_435_456)
    }
    package var http: LocalHTTPConfiguration {
        var value = LocalHTTPConfiguration()
        value.applyContextLimits(context)
        return value
    }
    package init(context: Int, fileBacked: Bool? = nil) throws {
        guard Self.contexts.contains(context), fileBacked != true || context == Self.maximumFileContext else {
            throw EngineError("Unsupported conversation geometry", code: "invalid_configuration")
        }
        self.context = context
        self.fileBacked = fileBacked ?? (context == Self.maximumFileContext)
    }
    package func makeArena(ledger: MemoryLedger, parent: URL?) throws -> QSAPageArena {
        let store: QSAFilePageStore?
        if fileBacked {
            guard let parent else { throw EngineError("262K conversation requires its external KV owner", code: "invalid_configuration") }
            store = try QSAFilePageStore(parent: parent, diskBytes: Self.fileDiskBytes, ledger: ledger)
        } else {
            store = nil
        }
        let arena = try QSAPageArena(ledger: ledger, maximumBytes: arenaBytes, fileStore: store)
        // The planned 1 GiB staging is reserved now: a Mac short of memory refuses at start, not in the first long prompt.
        if fileBacked, let device = MTLCreateSystemDefaultDevice() { _ = try arena.staging(device: device) }
        return arena
    }

    // Maximum C256 allocation, checked against the actual Metal layout by the
    // workspace fixture. Two decode generations can coexist with checkpoint
    // serialization; restoration ends before those generations are allocated.
    // Neither holds a 32 MiB fp32 copy of the pooled index (the indexer reads the pages in place).
    package static let maximumPrefillTileBytes: UInt64 = 556_171_264
    package static let maximumDecodeTileBytes: UInt64 = 24_559_616
    // Two 6 MiB maximum original-rotation buffers, bounded cached/provisional
    // tables and page rounding. Tables contain <=256 positions, not all keys.
    package static let maximumRoPEBytes: UInt64 = 16 * 1024 * 1024
    package var maximumCheckpointScratchBytes: UInt64 {
        get throws {
            max(
                try CheckpointMemoryLease.arrayAdmissionBytes(count: context * 512, element: UInt16.self)
                    + 32 * 1024 * 1024,
                PagedCheckpointFormat.scratchBytes + UInt64(context * 1024 + 65_536))
        }
    }
    /// `heldArenaBytes`: the arena charge held now (nil: the whole arena).
    package func maximumFixedBytes(
        input: Int, residentBytes: UInt64,
        retainedFrontierBytes: UInt64? = nil, heldArenaBytes: UInt64? = nil
    ) throws -> UInt64 {
        guard (0...8192).contains(input) else { throw EngineError("Conversation phase geometry differs") }
        let frontier = try frontierMemoryBytes
        guard retainedFrontierBytes == nil || (input == 0 && !fileBacked && retainedFrontierBytes! <= frontier) else {
            throw EngineError("Frontier loan requires bounded8K/32K decode owners", code: "invalid_configuration")
        }
        let host = input == 0 ? ReferencePhaseWorkspace.decodeBytes : ReferencePhaseWorkspace.fullBytes
        // A scope of up to `input` rows (prepareUpcomingInput rounds the prompt's real rows up to whole tiles): the
        // historical 240 KB per row past 256, plus what a small scope's buffers need beyond it (none from 1,537 rows).
        // A tail one row shorter carries mixed-tail metadata a whole-tile scope does not, so the larger of the two.
        func scopeBytes(_ rows: Int) throws -> UInt64 {
            guard rows > 0 else { return 0 }
            let layout = try WholePrefillScope.layout(tokens: rows, allowTail: true, admitSmallScope: true)
            return UInt64(max(0, rows - 256)) * 240_000 + (layout?.supplementalBytes ?? 0)
        }
        let scope = max(try scopeBytes(input), try scopeBytes(input - 1))
        // The complete 262K decode allowance also covers both smaller layouts.
        // Retain its slack for synchronous readout and joined phase bridges.
        let gpu = input == 0 ? 2 * Self.maximumDecodeTileBytes : (context > 32768 ? Self.maximumPrefillTileBytes : 512_000_000)
        let phase = input == 0 ? gpu + (try maximumCheckpointScratchBytes) : max(gpu, 238_000_000)
        return residentBytes + min(heldArenaBytes ?? arenaBytes, arenaBytes) + (try fileMetadataBytes) + stagingBytes
            + 240_000_000 + 120_000_000 + (retainedFrontierBytes ?? frontier) + 128_000_000 + http.admittedBytes
            + host + scope + phase + 14_352_384 + Self.maximumRoPEBytes
    }
}
