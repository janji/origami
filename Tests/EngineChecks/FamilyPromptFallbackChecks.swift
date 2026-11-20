import Foundation
import Metal
import OrigamiEngine

/// The static prompt victim queue must be revalidated after a lazy slab allocation
/// fails and falls back to an existing slot. The old queue still contains that slot.
/// This exercises actual fetch/read bytes, using synthetic pressure rather than host pressure.
func familyPromptFallbackDistinctOwners() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Family prompt fallback: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prompt-fallback-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let saved = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    // Both family expert layouts: distinct payload per expert and projection,
    // five logical slots, four allocated slots and six possible authoritative experts.
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
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        try autoreleasepool {
            let weights = try FamilyWeights(gguf: GGUFSource(file), device: device)
            let pool = try FamilyExperts(weights: weights, budget: UInt64(5 * slotBytes))
            pool.ledger = ledger
            func verify(_ slots: [FamilyExperts.Slot], _ experts: [Int]) throws {
                try require(slots.count == experts.count, "missing requested expert")
                for (slot, expert) in zip(slots, experts) {
                    for part in 0..<3 {
                        let bytes = UnsafeBufferPointer(
                            start: (slot.buffer.contents() + slot.offsets[part])
                                .assumingMemoryBound(to: UInt8.self), count: partBytes)
                        try require(
                            bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) },
                            "expert \(expert), projection \(part) overwritten after allocation fallback")
                    }
                }
            }
            // Expert5 is oldest/coldest and also Passed's furthest-ahead candidate.
            // Two ascending prompt requests are enough to create this causal history.
            _ = try pool.fetch(layer: 1, experts: [5])
            _ = try pool.fetch(layer: 1, experts: [2, 3, 4])
            try require(
                pool.slots(layer: 1) == 5 && pool.allocatedSlots(layer: 1) == 4,
                "fixture did not create exactly one unallocated logical slot")
            ledger.setOptionalGrowthAllowed(false)
            try require(
                try pool.prepareWave(layer: 1, maximumExperts: 2) == 2,
                "physical two-wave planning did not permit the bounded demand")
            // Miss0 first chooses lazy slot4, is denied, then takes slot0/expert5.
            // Without revalidation, miss1's stale Passed entry chooses slot0 again.
            let current = try pool.fetch(layer: 1, experts: [0, 1])
            try require(
                ObjectIdentifier(current[0].buffer) != ObjectIdentifier(current[1].buffer)
                    || current[0].offsets[0] != current[1].offsets[0], "two demanded experts alias one slot")
            try verify(current, [0, 1])
            try require(pool.allocatedSlots(layer: 1) == 4, "denied optional growth unexpectedly allocated")
            // Protect both current owners while the next wave uses the other two slots.
            let following = try pool.fetch(layer: 1, experts: [4, 5], keepingLast: true)
            try verify(current, [0, 1])
            try verify(following, [4, 5])
            // A full current fetch is also protected after stale candidates are exhausted.
            let full = try pool.fetch(layer: 1, experts: [0, 1, 4, 5])
            var refused = false
            do { _ = try pool.fetch(layer: 1, experts: [2], keepingLast: true) } catch { refused = true }
            try require(refused, "exhausted protected pool accepted an overwrite")
            try verify(full, [0, 1, 4, 5])
            pool.waitForReads()
        }
        try require(ledger.reservedBytes == 0, "joined records retained their ledger ownership")
    }
}
