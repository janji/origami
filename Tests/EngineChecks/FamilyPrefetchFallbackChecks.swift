import Foundation
import Metal
import OrigamiEngine

/// A guess whose lazy slab is denied (a full ledger here, memory pressure at the end) reads into a slot that already has
/// one, never one the current wave holds or one being read, and allocates nothing. Exercises the real record reader.
func familyPrefetchAllocatedFallback() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Family prefetch fallback: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prefetch-fallback-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let saved = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    for type in [8, 12] {
        let partBytes = GGUFQuantize.bytes(type, count: 256 * 256)
        let slotBytes = (3 * partBytes + 16383) / 16384 * 16384
        var tensors: [(String, [Int], Int)] = []
        var data: [String: Data] = [:]
        for (part, name) in ["gate", "up", "down"].enumerated() {
            let tensor = "blk.1.ffn_\(name)_exps.weight"
            tensors.append((tensor, [6, 256, 256], type))
            data[tensor] = Data(
                (0..<6).flatMap { expert in
                    [UInt8](repeating: UInt8(50 + expert * 10 + part), count: partBytes)
                })
        }
        let file = folder.appendingPathComponent("experts-\(type).gguf")
        try familyGGUF(
            [
                ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256),
                ("glm5-next.vocab_size", 512), ("glm5-next.expert_count", 6),
                ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 256),
                ("glm5-next.leading_dense_block_count", 1), ("glm5-next.block_count", 2),
                ("glm5-next.attention.head_count_kv", [0, 64]), ("glm5-next.hyper_connection.count", 4),
            ],
            tensors: tensors, data: data, to: file)
        for ahead in [false, true] {
            let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
            try autoreleasepool {
                let weights = try FamilyWeights(gguf: GGUFSource(file), device: device)
                let pool = try FamilyExperts(weights: weights, budget: UInt64(5 * slotBytes))
                pool.ledger = ledger
                pool.adaptiveGuesses = false
                var hold: IOPrefetchHold?
                defer {
                    hold?.release()
                    pool.waitForReads()
                }
                func verify(_ slots: [FamilyExperts.Slot], _ experts: [Int]) throws {
                    try require(slots.count == experts.count, "missing requested expert")
                    for (slot, expert) in zip(slots, experts) {
                        for part in 0..<3 {
                            let bytes = UnsafeBufferPointer(
                                start: (slot.buffer.contents() + slot.offsets[part])
                                    .assumingMemoryBound(to: UInt8.self), count: partBytes)
                            try require(
                                bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) },
                                "type \(type), ahead \(ahead), expert \(expert), projection \(part) overwritten")
                        }
                    }
                }
                // Slot 0/expert 5 is oldest and Passed's furthest-ahead candidate.
                // If lazy slot 4 fails, its fallback must not be selected again
                // from that stale prompt-victim snapshot for the second guess.
                _ = try pool.fetch(layer: 1, experts: [5, 4])
                let current = try pool.fetch(layer: 1, experts: [2, 3])
                try require(
                    pool.slots(layer: 1) == 5 && pool.allocatedSlots(layer: 1) == 4,
                    "fixture did not create five logical and four allocated slots")
                let occupied = try ledger.reserve(
                    bytes: ledger.maximumUsableBytes - ledger.reservedBytes,
                    category: "prefetch-fallback-denial")
                defer { try? ledger.release(occupied) }
                let reserved = ledger.reservedBytes
                let residentBytes = pool.residentBytes
                try require(ledger.optionalGrowthAllowed, "fixture accidentally inhibited all guesses")
                hold = pool.holdGuesses()
                func predict(_ experts: [Int]) {
                    if ahead { pool.readAhead(layer: 1, experts: experts) } else { pool.prefetch(layer: 1, experts: experts) }
                }
                predict([0, 1])
                let issued = ahead ? pool.counters.readAheads : pool.counters.prefetches
                try require(
                    issued == 2, "type \(type), ahead \(ahead): issued \(issued), expected two allocated-slot reads after ledger denial")
                try require(pool.pendingReads == 6, "did not enqueue exactly three projections per record")
                let drives = pool.reads.snapshot().drives
                try require(
                    drives.reduce(0) { $0 + $1.queuedPrefetch } == 6 && drives.allSatisfy { $0.queuedDemand == 0 },
                    "speculation priority changed")
                // Both fallback slots are busy; the other two are this wave's GPU
                // owners. Another prediction must neither wait nor overwrite them.
                predict([4, 5])
                try require(
                    (ahead ? pool.counters.readAheads : pool.counters.prefetches) == issued
                        && pool.pendingReads == 6, "busy/current-wave slots admitted another guess")
                try verify(current, [2, 3])
                hold?.release()
                pool.waitForReads()
                try verify(current, [2, 3])
                let reads = pool.counters.readBytes
                let demanded = try pool.fetch(layer: 1, experts: [0, 1], decode: true)
                try verify(demanded, [0, 1])
                try verify(current, [2, 3])
                try require(pool.counters.readBytes == reads, "completed fallback records were read again on demand")
                try require(
                    (ahead ? pool.counters.readAheadUsed : pool.counters.prefetchUsed) == 2,
                    "fallback records were not consumed as prefetched hits")
                try require(
                    pool.allocatedSlots(layer: 1) == 4 && pool.residentBytes == residentBytes
                        && ledger.reservedBytes == reserved, "fallback grew or lost physical/ledger ownership")
                // Expert 1 is now the coldest non-current owner, but it is also
                // already resident in the next prediction. Keep it while the
                // other two old slots satisfy that prediction's missing records.
                _ = try pool.fetch(layer: 1, experts: [2, 3], decode: true)
                _ = try pool.fetch(layer: 1, experts: [0], decode: true)
                hold = pool.holdGuesses()
                predict([1, 4, 5])
                let allIssued = ahead ? pool.counters.readAheads : pool.counters.prefetches
                try require(
                    allIssued == issued + 2 && pool.pendingReads == 6,
                    "kept/current owner protection lost available fallback capacity")
                hold?.release()
                pool.waitForReads()
                try verify(demanded, [0, 1])
                let following = try pool.fetch(layer: 1, experts: [4, 5], decode: true)
                try verify(following, [4, 5])
                try require(
                    pool.allocatedSlots(layer: 1) == 4 && pool.residentBytes == residentBytes
                        && ledger.reservedBytes == reserved, "kept-owner fallback changed allocation ownership")
                // Memory pressure on top: the guess still reads into the two slots the current wave does not hold.
                ledger.setOptionalGrowthAllowed(false)
                predict([2, 3])
                pool.waitForReads()
                try require(
                    (ahead ? pool.counters.readAheads : pool.counters.prefetches) == allIssued + 2,
                    "pressure refused a guess that needed no new slab")
                try verify(following, [4, 5])
                let wave = try pool.fetch(layer: 1, experts: [2, 3], decode: true)
                try verify(wave, [2, 3])
                try require(
                    pool.allocatedSlots(layer: 1) == 4 && pool.residentBytes == residentBytes
                        && ledger.reservedBytes == reserved, "a guess under pressure changed allocation ownership")
                // Live pressure too: a guess an existing slot can take still reads (skipping it frees nothing), never
                // into the current wave's slots and without a new slab; nothing is counted as refused.
                let missing = (0..<6).filter { !pool.resident(layer: 1).contains($0) }
                let refusals = ledger.optionalRefusals
                let before = ahead ? pool.counters.readAheads : pool.counters.prefetches
                let rejected = pool.counters.guessRejectedPressure
                ledger.setOptionalReadsAllowed(false)
                predict(missing)
                pool.waitForReads()
                ledger.setOptionalReadsAllowed(true)
                try require(
                    missing.count == 2 && (ahead ? pool.counters.readAheads : pool.counters.prefetches) == before + 2
                        && pool.counters.guessRejectedPressure == rejected && ledger.optionalRefusals.live == refusals.live
                        && ledger.optionalRefusals.belowTarget == refusals.belowTarget,
                    "live pressure refused a guess that needed no new slab, or counted one that proceeded")
                try verify(wave, [2, 3])
                try require(
                    pool.allocatedSlots(layer: 1) == 4 && pool.residentBytes == residentBytes
                        && ledger.reservedBytes == reserved, "a guess under live pressure changed allocation ownership")
                ledger.setOptionalGrowthAllowed(true)
            }
            try require(ledger.reservedBytes == 0, "completed records retained ledger ownership")
        }
    }
}

/// A queued guess that cancels itself after `settle` kept it and before it looked at the read's end read nothing: its
/// slot still holds the evicted record's bytes. Landing it forgets it (counted as cancelled, never used or wasted),
/// and a later demand reads it, never serving the stale slot as resident.
func familyPrefetchSelfCancelledLanding() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Family self-cancelled guess: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-self-cancel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let partBytes = GGUFQuantize.bytes(8, count: 256 * 256)
    let slotBytes = (3 * partBytes + 16383) / 16384 * 16384
    var tensors: [(String, [Int], Int)] = []
    var data: [String: Data] = [:]
    for (part, name) in ["gate", "up", "down"].enumerated() {
        let tensor = "blk.1.ffn_\(name)_exps.weight"
        tensors.append((tensor, [6, 256, 256], 8))
        data[tensor] = Data((0..<6).flatMap { expert in [UInt8](repeating: UInt8(50 + expert * 10 + part), count: partBytes) })
    }
    let file = folder.appendingPathComponent("experts.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256),
            ("glm5-next.vocab_size", 512), ("glm5-next.expert_count", 6),
            ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 256),
            ("glm5-next.leading_dense_block_count", 1), ("glm5-next.block_count", 2),
            ("glm5-next.attention.head_count_kv", [0, 64]), ("glm5-next.hyper_connection.count", 4),
        ],
        tensors: tensors, data: data, to: file)
    for ahead in [false, true] {
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        let weights = try FamilyWeights(gguf: GGUFSource(file), device: device)
        let pool = try FamilyExperts(weights: weights, budget: UInt64(4 * slotBytes))
        pool.ledger = ledger
        pool.adaptiveGuesses = false
        var hold: IOPrefetchHold?
        defer {
            pool.settleProbe = nil
            hold?.release()
            pool.waitForReads()
        }
        _ = try pool.fetch(layer: 1, experts: [0, 1], decode: true)
        _ = try pool.fetch(layer: 1, experts: [2, 3], decode: true)
        let before = pool.counters
        hold = pool.holdGuesses()
        if ahead { pool.readAhead(layer: 1, experts: [4]) } else { pool.prefetch(layer: 1, experts: [4]) }
        try require(pool.pendingReads == 3, "ahead \(ahead): the guess did not queue its three reads")
        // The scheduler cancels the queued reads in the window, after settle kept the guess and before it looks at their end.
        pool.settleProbe = {
            _ = try? pool.reads.cancelAndJoin(pool.reads.currentGeneration)
            hold?.release()
            pool.waitForReads()
        }
        let resident = pool.resident(layer: 1)
        pool.settleProbe = nil
        try require(!resident.contains(4), "ahead \(ahead): a guess that read nothing became resident \(resident)")
        let c = pool.counters
        try require(
            ahead
                ? c.readAheads == before.readAheads && c.readBytes == before.readBytes
                : c.prefetches == before.prefetches + 1 && c.prefetchCancelled == before.prefetchCancelled + 1
                    && c.prefetchBytes == before.prefetchBytes,
            "ahead \(ahead): not counted as cancelled \(c)")
        let slot = try pool.fetch(layer: 1, experts: [4], decode: true)[0]
        for part in 0..<3 {
            let bytes = UnsafeBufferPointer(
                start: (slot.buffer.contents() + slot.offsets[part]).assumingMemoryBound(to: UInt8.self), count: partBytes)
            try require(bytes.allSatisfy { $0 == UInt8(90 + part) }, "ahead \(ahead): projection \(part) holds stale bytes")
        }
        try require(
            pool.counters.misses == c.misses + 1 && pool.counters.prefetchUsed == 0 && pool.counters.readAheadUsed == 0
                && pool.counters.prefetchWastedBytes == 0,
            "ahead \(ahead): not read on demand, or counted as used or wasted \(pool.counters)")
    }
}

/// A guess whose reads are cancelled while still queued in the scheduler read nothing: by the scheduler itself
/// (`cancelAndJoin`, mapped onto the guess as `settle` cancels one) or by `settle` (`cancelQueued`). Its slot, which
/// held another record, is not served as loaded: it counts as cancelled, never used, and a following demand fetch
/// of that key reads the file's bytes.
func familyPrefetchQueuedCancelRereads() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Family queued-cancelled guess: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-queued-cancel-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let per = GGUFQuantize.bytes(8, count: 256 * 256)
    func verify(_ slots: [FamilyExperts.Slot], _ experts: [Int]) throws {
        try require(slots.count == experts.count, "missing records")
        for (slot, expert) in zip(slots, experts) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
                try require(bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) }, "expert \(expert) part \(part) is not the file's")
            }
        }
    }
    for byScheduler in [true, false] {
        let pool = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))  // 3 slots per layer
        pool.adaptiveGuesses = false
        try verify(pool.fetch(layer: 1, experts: [0, 1, 2], decode: true), [0, 1, 2])
        try verify(pool.fetch(layer: 1, experts: [0], decode: true), [0])  // every slot holds a record; 1 and 2 may go
        let hold = pool.holdGuesses()
        defer {
            hold.release()
            pool.waitForReads()
        }
        pool.prefetch(layer: 1, experts: [3])
        let queued = pool.reads.snapshot().drives
        try require(
            pool.counters.prefetches == 1 && queued.reduce(0) { $0 + $1.queuedPrefetch } == 3,
            "the guess did not queue its three reads")
        let before = pool.counters
        if byScheduler {
            let outcomes = try pool.reads.cancelAndJoin(pool.reads.currentGeneration)
            try require(outcomes.count == 3 && outcomes.allSatisfy { $0 == .cancelled }, "the scheduler did not cancel the queued reads")
        } else {
            try verify(pool.fetch(layer: 1, experts: [0], decode: true), [0])  // settle cancels the guess it does not ask for
            let drives = pool.reads.snapshot().drives
            try require(
                drives.reduce(UInt64(0)) { $0 + $1.cancelled } == 3 && drives.allSatisfy { $0.outstanding == 0 },
                "settle left the cancelled reads queued")
        }
        try verify(pool.fetch(layer: 1, experts: [3], decode: true), [3])
        let c = pool.counters
        try require(
            c.prefetchCancelled == before.prefetchCancelled + 1 && c.prefetchUsed == 0 && c.misses == before.misses + 1
                && c.readBytes == before.readBytes + 3 * per && c.prefetchBytes == 0,
            "by scheduler \(byScheduler): a cancelled guess was served or miscounted \(c)")
    }
}
