import Foundation
import Metal
import OrigamiEngine

/// A pool of three slots per layer over the two-layer expert fixture (layer 1's records 212,992 bytes, layer 2's
/// 114,688), and a check that a returned slot holds expert `expert`'s bytes.
private func slotCorePool(_ name: String) throws -> (FamilyExperts, URL, URL) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-\(name)-\(getpid())")
    let gguf = try expertFixtureGGUF(in: folder)
    let experts = try FamilyExperts(
        weights: try FamilyWeights(gguf: GGUFSource(gguf), device: device), budget: UInt64(3 * (212_992 + 114_688)))
    experts.adaptiveGuesses = false
    return (experts, folder, gguf)
}
private func holds(_ slot: FamilyExperts.Slot, layer: Int, expert: Int) -> Bool {
    let per = GGUFQuantize.bytes(layer == 1 ? 8 : 12, count: 256 * 256)
    return (0..<3).allSatisfy { part in
        UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per).allSatisfy {
            $0 == UInt8(layer * 50 + expert * 10 + part)
        }
    }
}

/// A fetch that adopts a guess still queued, which the scheduler then cancels, reads the record itself: it returns the
/// file's bytes, never the slot the cancelled guess held (nothing was read into it).
func familyAdoptedGuessCancelled() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Adopted guess cancelled: " + what) } }
    let (experts, folder, _) = try slotCorePool("adopted-guess")
    defer { try? FileManager.default.removeItem(at: folder) }
    _ = try experts.fetch(layer: 1, experts: [0], decode: true)
    // Every start held: the guess the fetch adopts (promoted to demand) stays queued until the scheduler cancels it.
    let held = experts.reads.holdStarts()
    var released = false
    defer { if !released { held.release() } }
    experts.prefetch(layer: 1, experts: [1])
    var delivered: [Int] = []
    let slots = try experts.fetch(
        layer: 1, experts: [0, 1], decode: true,
        ready: { indices, _ in
            delivered = indices
            _ = try experts.reads.cancelAndJoin(experts.reads.currentGeneration)
            held.release()
            released = true
        })
    try require(delivered == [0], "delivered \(delivered) before the adopted guess ended")
    try require(holds(slots[0], layer: 1, expert: 0) && holds(slots[1], layer: 1, expert: 1), "a returned slot holds other bytes")
    let c = experts.counters
    try require(c.prefetchCancelled == 1 && c.misses == 2 && c.readBytes == 2 * experts.bank(1)!.bytes.reduce(0, +), "counters \(c)")
    try require(experts.resident(layer: 1) == [0, 1] && experts.phase(layer: 1, expert: 1) == .pinned, "\(experts.resident(layer: 1))")
    try requireFamilyMirror(experts, "Adopted guess cancelled")
}

/// A ticket of a record whose slot was taken by another record (a new generation) can neither publish nor pin it, and
/// the slot keeps the new record's bytes.
func familyStaleTicketRejected() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family stale ticket: " + what) } }
    let (experts, folder, _) = try slotCorePool("stale-ticket")
    defer { try? FileManager.default.removeItem(at: folder) }
    let core = experts.slotCore
    _ = try experts.fetch(layer: 2, experts: [0], decode: true)
    let stale = try core.acquire(2 * 4 + 0).ticket
    for expert in 1...3 { _ = try experts.fetch(layer: 2, experts: [expert], decode: true) }  // the third takes expert 0's slot
    try require(experts.resident(layer: 2) == [1, 2, 3] && core.lookup[2 * 4 + 3] == stale.slot, "expert 0's slot was not reused")
    let rejected = core.rejected
    try require(!core.completeRead(stale, verified: true) && core.rejected == rejected + 1, "a stale ticket published")
    try require((try? core.pin(stale)) == nil && (try? core.beginRead(stale)) == nil, "a stale ticket pinned or began a read")
    try require(holds(try experts.fetch(layer: 2, experts: [3], decode: true)[0], layer: 2, expert: 3), "expert 3's bytes changed")
    try require(holds(try experts.fetch(layer: 2, experts: [0], decode: true)[0], layer: 2, expert: 0), "expert 0 read again")
}

/// A record whose read fails is never returned, resident, ready or pending: a demand read that fails throws, a guess
/// that fails is dropped when it lands, and a fetch that adopted a failing guess reads the record itself and throws.
func familyFailedReadNeverResident() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family failed read: " + what) } }
    let (experts, folder, gguf) = try slotCorePool("failed-read")
    defer { try? FileManager.default.removeItem(at: folder) }
    let full = try Data(contentsOf: gguf)
    _ = try experts.fetch(layer: 2, experts: [1], decode: true)  // opens the file at its full size
    // Layer 2's down projection is the file's last tensor: cut its experts 2 and 3.
    let handle = try FileHandle(forWritingTo: gguf)
    try handle.truncate(atOffset: UInt64(full.count - 2 * GGUFQuantize.bytes(12, count: 256 * 256)))
    // `resident` first: it lands the guesses that ended.
    func nothing(_ expert: Int) -> Bool {
        !experts.resident(layer: 2).contains(expert) && experts.phase(layer: 2, expert: expert) == nil
            && !experts.observedAvailableOrPendingExperts(layer: 2).contains(expert)
    }
    try require((try? experts.fetch(layer: 2, experts: [0, 3], decode: true)) == nil, "a failed read returned")
    try require(nothing(3) && nothing(0), "a failed fetch left \(experts.resident(layer: 2))")
    experts.prefetch(layer: 2, experts: [2])
    experts.waitForReads()
    try require(!experts.observedReadyExperts(layer: 2).contains(2) && nothing(2), "a failed guess is ready")
    _ = try experts.fetch(layer: 1, experts: [0], decode: true)  // the slots layer 2's calls took are no longer the last call's
    let guesses = experts.holdGuesses()
    experts.prefetch(layer: 2, experts: [3])
    try require(experts.phase(layer: 2, expert: 3) == .reading && !experts.resident(layer: 2).contains(3), "a guess being read is resident")
    try require((try? experts.fetch(layer: 2, experts: [3], decode: true)) == nil, "a fetch returned a failed guess")
    guesses.release()
    experts.waitForReads()
    try require(nothing(3), "a failed adopted guess left a record")
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: full)
    try handle.close()
    let slots = try experts.fetch(layer: 2, experts: [2, 3], decode: true)
    try require(holds(slots[0], layer: 2, expert: 2) && holds(slots[1], layer: 2, expert: 3), "records read after the file came back")
    try requireFamilyMirror(experts, "Failed read never resident")
}

/// The records a fetch returns stay pinned until the next fetch begins, or the one after when that one is `keepingLast`
/// (the GPU may still read them); a guess keeps only the last fetch's pins, and a limit change releases them all.
func familyPinLifetime() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family pins: " + what) } }
    let (experts, folder, _) = try slotCorePool("pins")
    defer { try? FileManager.default.removeItem(at: folder) }
    func pinned(_ layer: Int = 1) -> [Int] { (0..<4).filter { experts.phase(layer: layer, expert: $0) == .pinned } }
    _ = try experts.fetch(layer: 1, experts: [0], decode: true)
    try require(pinned() == [0], "first call \(pinned())")
    _ = try experts.fetch(layer: 1, experts: [1], decode: true)
    try require(pinned() == [1] && experts.phase(layer: 1, expert: 0) == .evictable, "next call \(pinned())")
    _ = try experts.fetch(layer: 1, experts: [2], keepingLast: true)
    try require(pinned() == [1, 2], "keepingLast \(pinned())")
    _ = try experts.fetch(layer: 1, experts: [3], keepingLast: true)
    try require(pinned() == [2, 3], "keepingLast again \(pinned())")
    _ = try experts.fetch(layer: 1, experts: [2], decode: true)
    try require(pinned() == [2], "a call that keeps nothing \(pinned())")
    _ = try experts.fetch(layer: 1, experts: [3], decode: true, keepingLast: true)
    try require(pinned() == [2, 3], "keepingLast after decode \(pinned())")
    experts.prefetch(layer: 2, experts: [0])
    experts.waitForReads()
    try require(pinned() == [3] && pinned(2).isEmpty, "a guess kept \(pinned())")
    experts.setLimit(bytes: experts.leastBytes)
    try require(pinned().isEmpty, "a limit change kept \(pinned())")
}
