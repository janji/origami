import Foundation
import Metal
import OrigamiEngine

/// The delayed record is enqueued only after the immediate records own their slots. Exercise that
/// existing submission boundary with queued/completed reads, pressure cancellation and a real read error.
/// No full model, timing-based speed assertion or extra record allocation is needed.
func familyEarlyDelayedReadOwnership() throws {
    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw EngineError("Early delayed read: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-early-delay-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let savedSpins = FamilyExperts.spins
    FamilyExperts.spins = false
    defer { FamilyExperts.spins = savedSpins }
    let partBytes = GGUFQuantize.bytes(8, count: 256 * 256)
    func verify(_ slots: [FamilyExperts.Slot], _ experts: [Int]) throws {
        try require(slots.count == experts.count, "missing record")
        for (slot, expert) in zip(slots, experts) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: partBytes)
                try require(
                    bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) },
                    "expert \(expert) projection \(part) overwritten or incomplete")
            }
        }
    }
    for scenario in ["completed", "queued", "pressure", "failed"] {
        let path = try expertFixtureGGUF(in: root.appendingPathComponent(scenario))
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        // The delayed policy requires one spare slot beyond all four selected experts.
        let pool = try FamilyExperts(weights: weights, budget: UInt64(5 * (212_992 + 114_688)))
        pool.ledger = ledger
        _ = try pool.prepareWave(layer: 1, maximumExperts: 5)
        let owned = pool.residentBytes
        let reserved = ledger.reservedBytes
        let hold = pool.holdGuesses()
        defer {
            hold.release()
            pool.waitForReads()
        }
        var submitted = 0
        let immediate = try pool.fetch(
            layer: 1, experts: [0, 1, 2], decode: true,
            submitted: {
                submitted += 1
                pool.readAhead(layer: 1, experts: [3])
            })
        try require(
            submitted == 1 && pool.counters.readAheads == 1 && pool.pendingReads == 3,
            "immediate fetch joined, cancelled or failed to submit the deferred record")
        try verify(immediate, [0, 1, 2])
        try require(
            pool.residentBytes == owned && ledger.reservedBytes == reserved,
            "submission allocated another record")
        let original = scenario == "failed" ? try Data(contentsOf: path) : nil
        if scenario == "failed" {
            let writer = try FileHandle(forWritingTo: path)
            try writer.truncate(atOffset: 0)
            try writer.close()
        }
        if scenario == "pressure" { ledger.setOptionalGrowthAllowed(false) }
        hold.release()
        if scenario != "queued" { pool.waitForReads() }
        ledger.setOptionalGrowthAllowed(true)
        let before = pool.counters.readBytes
        if let original {
            var failed = false
            do { _ = try pool.fetch(layer: 1, experts: [3], decode: true, keepingLast: true) } catch { failed = true }
            try require(failed, "failed deferred record was published")
            // Restore this disposable fixture in place: the reader owns its existing descriptor.
            let writer = try FileHandle(forWritingTo: path)
            try writer.write(contentsOf: original)
            try writer.close()
        }
        let deferred = try pool.fetch(layer: 1, experts: [3], decode: true, keepingLast: true)
        try verify(deferred, [3])
        try verify(immediate, [0, 1, 2])
        if scenario == "completed" || scenario == "queued" {
            try require(
                pool.counters.readBytes == before && pool.counters.readAheadUsed == 1,
                "early record was read twice instead of consumed")
        }
        try require(
            pool.residentBytes == owned && ledger.reservedBytes == reserved,
            "joining or fallback changed allocated ownership")

        let runner = try FamilyRunner(weights: weights, context: 64)
        runner.experts = pool
        let override = TuneOverrides(familyEarlyDelayedReads: true)
        try require(try TuneOverrides(json: override.wire) == override, "tuning round trip lost the switch")
        let restore = try runner.tune(override)
        try require(runner.earlyDelayedReads, "request override did not apply")
        try restore?()
        try require(!runner.earlyDelayedReads, "request override leaked into later requests")
    }
}

/// A consumer error must wait for both ordinary reads and an already requested prediction, suppress later callbacks,
/// and leave complete cache records reusable. A failed read must instead remove every fresh record before returning.
/// Uses tiny fixture records and shared buffers only; no model or GPU commands are needed.
func familyArrivalErrorsJoinAndRecover() throws {
    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw EngineError("Family arrival cleanup: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-arrival-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let savedSpins = FamilyExperts.spins
    defer { FamilyExperts.spins = savedSpins }
    enum ConsumerError: Error { case stopped }
    let perExpert = GGUFQuantize.bytes(8, count: 256 * 256)
    func verify(_ slots: [FamilyExperts.Slot], experts: [Int]) throws {
        try require(slots.count == experts.count, "missing complete records")
        for (slot, expert) in zip(slots, experts) {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: perExpert)
                try require(
                    bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) },
                    "expert \(expert) part \(part) was published before all of its bytes arrived")
            }
        }
    }
    for spins in [false, true] {
        FamilyExperts.spins = spins
        let folder = root.appendingPathComponent(spins ? "spin" : "blocking")
        let path = try expertFixtureGGUF(in: folder)
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let pool = try FamilyExperts(weights: weights, budget: UInt64(4 * (212_992 + 114_688)))
        var suspended: IOPrefetchHold?
        defer {
            suspended?.release()
            pool.waitForReads()
        }
        _ = try pool.prepareWave(layer: 2, maximumExperts: 1)
        _ = try pool.fetch(layer: 1, experts: [0], decode: true)
        // Hold every start: the fetch promotes the prediction it asks for, and that must stay queued until the callback.
        suspended = pool.reads.holdStarts()
        pool.prefetch(layer: 1, experts: [1])
        try require(pool.resident(layer: 1) == [0], "an unstarted prediction was already resident")
        var callbacks = 0
        var stopped = false
        do {
            _ = try pool.fetch(
                layer: 1, experts: [0, 1, 2], decode: true,
                submitted: { pool.prefetch(layer: 2, experts: [0]) }
            ) { indices, slots in
                callbacks += 1
                try require(indices == [0], "the resident callback included an incomplete record")
                try verify(slots, experts: [0])
                // The prediction is deterministically still pending when the callback begins, so a thrown consumer
                // cannot bypass the later awaited-prediction join merely because all fixture reads were fast.
                suspended?.release()
                throw ConsumerError.stopped
            }
        } catch ConsumerError.stopped { stopped = true }
        try require(
            stopped && callbacks == 1 && pool.resident(layer: 1) == [0, 1, 2],
            "consumer failure escaped before every requested producer joined")
        let readBefore = pool.counters.readBytes
        let missBefore = pool.counters.misses
        try verify(pool.fetch(layer: 1, experts: [0, 1, 2], decode: true), experts: [0, 1, 2])
        try require(
            pool.counters.readBytes == readBefore && pool.counters.misses == missBefore,
            "a consumer failure discarded successfully joined records")

        // Corrupt only this disposable fixture after its file descriptor and first resident record are installed.
        // The real pread path fails; no fresh record may be handed to the caller or retained as a cache hit.
        let failed = try FamilyExperts(weights: weights, budget: UInt64(4 * (212_992 + 114_688)))
        defer { failed.waitForReads() }
        _ = try failed.prepareWave(layer: 2, maximumExperts: 1)
        _ = try failed.fetch(layer: 1, experts: [0], decode: true)
        let original = try Data(contentsOf: path)
        let writer = try FileHandle(forWritingTo: path)
        defer { try? writer.close() }
        try writer.truncate(atOffset: 0)
        callbacks = 0
        var readFailed = false
        do {
            _ = try failed.fetch(
                layer: 1, experts: [0, 1, 2], decode: true,
                submitted: { failed.prefetch(layer: 2, experts: [0]) }
            ) { indices, slots in
                callbacks += 1
                try require(indices == [0], "a failed record reached an arrival callback")
                try verify(slots, experts: [0])
            }
        } catch let error as EngineError { readFailed = String(describing: error).contains("pread failed") }
        try require(
            readFailed && callbacks == 1 && failed.resident(layer: 1) == [0],
            "read failure retained a partial record or lost the resident one")
        try writer.seek(toOffset: 0)
        try writer.write(contentsOf: original)
        try writer.synchronize()
        let beforeRetry = failed.counters.misses
        try verify(failed.fetch(layer: 1, experts: [0, 1, 2], decode: true), experts: [0, 1, 2])
        try require(failed.counters.misses == beforeRetry + 2, "retry trusted a failed record instead of reading it again")
    }
}
