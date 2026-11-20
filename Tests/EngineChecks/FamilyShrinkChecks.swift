import Foundation
import Metal
import OrigamiEngine

func familyShrinkChecks() -> [(String, () throws -> Void)] {
    [
        ("family_shrink_preserves_record_bytes", familyShrinkPreservesRecordBytes),
        ("family_shrink_cancels_unstarted_guesses", familyShrinkCancelsUnstartedGuesses),
    ]
}

private func shrinkRequire(_ ok: @autoclosure () -> Bool, _ message: String) throws {
    if !ok() { throw EngineError("Family shrink: " + message) }
}

private func withShrinkFixture(_ body: (FamilyExperts) throws -> Void) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-shrink-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let saved = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: root)), device: device)
    let pool = try FamilyExperts(weights: weights, budget: UInt64(4 * (212_992 + 114_688)))
    pool.guessDepth = nil
    try body(pool)
}

private func warmTail(_ pool: FamilyExperts, layer: Int) throws {
    try autoreleasepool {
        _ = try pool.fetch(layer: layer, experts: [0, 1, 2, 3], decode: true)
        for _ in 0..<8 { _ = try pool.fetch(layer: layer, experts: [2, 3], decode: true) }
    }
}

private func checkShrinkBytes(_ slots: [FamilyExperts.Slot], layer: Int, experts: [Int]) throws {
    let type = layer == 1 ? 8 : 12
    let bytes = GGUFQuantize.bytes(type, count: 256 * 256)
    for (slot, expert) in zip(slots, experts) {
        try shrinkRequire(slot.types == [type, type, type], "moved record type changed")
        for part in 0..<3 {
            let payload = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: bytes)
            try shrinkRequire(payload.allSatisfy { $0 == UInt8(layer * 50 + expert * 10 + part) }, "moved projection bytes changed")
        }
    }
}

/// Tiny fake expert records and shared buffers only: no model inference, command queues or GPU kernels.
private func familyShrinkPreservesRecordBytes() throws {
    try withShrinkFixture { pool in
        try warmTail(pool, layer: 1)
        let reads = pool.counters.readBytes
        let allocated = pool.residentBytes
        pool.setLimit(bytes: UInt64(2 * (212_992 + 114_688)))
        try shrinkRequire(pool.resident(layer: 1) == [0, 1], "shrink retained wrong identities")
        try shrinkRequire(pool.residentBytes < allocated && pool.allocatedSlots(layer: 1) == 2, "slabs not retired")
        let slots = try pool.fetch(layer: 1, experts: [2, 3], decode: true)
        try checkShrinkBytes(slots, layer: 1, experts: [2, 3])
        try shrinkRequire(pool.counters.readBytes > reads, "records past the limit were not read again")
        // A genuinely evicted expert is fetched again, with its own full projection bytes.
        let cold = try pool.fetch(layer: 1, experts: [0, 1], decode: true)
        try checkShrinkBytes(cold, layer: 1, experts: [0, 1])
    }
}

private func familyShrinkCancelsUnstartedGuesses() throws {
    try withShrinkFixture { pool in
        _ = try pool.fetch(layer: 1, experts: [0, 1], decode: true)
        pool.adaptiveGuesses = false
        let hold = pool.holdGuesses()
        defer {
            hold.release()
            pool.waitForReads()
        }
        pool.prefetch(layer: 1, experts: [2, 3])
        try shrinkRequire(pool.counters.prefetches == 2, "fixture did not queue both guesses")
        pool.setLimit(bytes: UInt64(2 * (212_992 + 114_688)))
        try shrinkRequire(
            pool.counters.prefetchCancelled == 2 && pool.counters.prefetchBytes == 0 && pool.counters.prefetchWastedBytes == 0,
            "cancelled predictions counted as copied/read/wasted")
        try shrinkRequire(pool.resident(layer: 1) == [0, 1], "unread cancelled bytes became resident")
        try checkShrinkBytes(try pool.fetch(layer: 1, experts: [0, 1], decode: true), layer: 1, experts: [0, 1])
    }
}
