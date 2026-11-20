import Foundation
import Metal
import OrigamiEngine

private func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw EngineError(message) }
}
private func mustFail(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Expected invalid cache/storage transition to fail")
}

func cacheCancellationPinsAndLateCompletions() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let bytes = UInt64(2 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: bytes + 1_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: bytes, slotsPerSlab: 2)
        let a = try ExpertKey(layer: 0, expert: 0)
        let b = try ExpertKey(layer: 0, expert: 1)
        let c = try ExpertKey(layer: 1, expert: 0)
        let ta = try cache.acquire(a).ticket
        try mustFail { _ = try cache.pin(ta) }
        var retainedWriter: ExpertBuffer? = try cache.beginRead(ta)
        var rawBuffer: MTLBuffer? = retainedWriter!.buffer
        retainedWriter!.bytes[0] = 0x37
        try mustFail { try cache.cancel(ta) }
        try require(try cache.acquire(a).phase == .reading, "A record being read was reserved again")
        let tb = try cache.acquire(b).ticket
        _ = try cache.beginRead(tb)
        try require(cache.completeRead(tb, verified: true), "Valid read failed publication")
        _ = try cache.pin(tb)
        try mustFail { _ = try cache.acquire(c) }
        try require(try cache.resize(to: 0) == bytes, "Busy slab was released")
        try require(!cache.completeRead(ta, verified: false), "Unverified read was published")
        let tc = try cache.acquire(c).ticket
        try require(tc.generation != ta.generation, "Generation reused")
        _ = try cache.beginRead(tc)
        try require(!cache.completeRead(ta, verified: true), "Late completion accepted into recycled slot")
        try require(cache.completeRead(tc, verified: true), "New generation damaged by stale completion")
        try cache.unpinAfterGPUCompletion(tb)
        try mustFail { try cache.unpinAfterGPUCompletion(tb) }
        try require(try cache.resize(to: 0) == 0, "Idle slabs did not shrink")
        try require(ledger.snapshot().reserved == bytes, "Live buffer escaped memory accounting")
        retainedWriter = nil
        try require(
            rawBuffer!.length == Int(bytes) && ledger.snapshot().reserved == bytes, "Escaped Metal buffer escaped memory accounting")
        rawBuffer = nil
        try require(cache.snapshot().rejectedCompletions == 1, "Late completion not counted")
        try mustFail { _ = try cache.pin(tc) }
    }
    try require(ledger.snapshot().reserved == 0, "Released buffer kept reservation after Metal autorelease drain")
}

func cacheClockEvictsOnlyUnpinnedRecords() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let bytes = UInt64(2 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: bytes + 100, transientMargin: 1)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: bytes, slotsPerSlab: 2)
    let a = try cache.acquire(ExpertKey(layer: 0, expert: 4)).ticket
    let b = try cache.acquire(ExpertKey(layer: 0, expert: 5)).ticket
    _ = try cache.beginRead(a)
    _ = try cache.beginRead(b)
    try require(cache.completeRead(a, verified: true), "A did not verify")
    try require(cache.completeRead(b, verified: true), "B did not verify")
    _ = try cache.pin(a)
    let c = try cache.acquire(ExpertKey(layer: 0, expert: 6)).ticket
    try require(c.generation != b.generation, "Eviction kept stale generation")
    try require(cache.snapshot().evictions == 1 && cache.snapshot().pinnedSlots == 1, "CLOCK evicted pinned data")
    try mustFail { _ = try cache.pin(b) }
    try cache.cancel(c)
    try cache.unpinAfterGPUCompletion(a)
    try require(try cache.resize(to: 0) == 0, "Cache failed final shrink")
}

/// Replays prefill sweeps (ascending keys, the last two requests handoff-pinned like the read-ahead) through an 8-slot
/// cache: with the prefill scan rule the two records decode kept survive the sweep and are hit, and a second sweep
/// hits every record the first one left; plain CLOCK (scan off) hits nothing in either.
func cachePrefillScanEvictsPassedRecords() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let bytes = UInt64(8 * PackStore.expertBytes)
    let ledger = try MemoryLedger(ceiling: bytes + 100, transientMargin: 1)
    func replay(scan: Bool) throws -> (first: UInt64, second: UInt64) {
        let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: bytes, slotsPerSlab: 2)
        defer { _ = try? cache.resize(to: 0) }
        var window: [ExpertTicket] = []
        func touch(_ key: ExpertKey, hold: Int) throws {
            let acquisition = try cache.acquire(key)
            if acquisition.newlyReserved {
                _ = try cache.beginRead(acquisition.ticket)
                try require(cache.completeRead(acquisition.ticket, verified: true), "Replay read did not verify")
            }
            _ = try cache.pin(acquisition.ticket)
            window.append(acquisition.ticket)
            while window.count > hold { try cache.unpinAfterGPUCompletion(window.removeFirst()) }
        }
        let hot = try [ExpertKey(layer: 5, expert: 0), ExpertKey(layer: 5, expert: 1)]
        for key in hot { try touch(key, hold: 0) }  // decode's records: not scan
        cache.setPrefillScan(scan)
        let sweep = try (0..<6).flatMap { layer in try (0..<4).map { try ExpertKey(layer: layer, expert: $0) } }
        for key in sweep { try touch(key, hold: 2) }
        while let ticket = window.popLast() { try cache.unpinAfterGPUCompletion(ticket) }
        let first = cache.snapshot().hits
        for key in sweep { try touch(key, hold: 2) }
        while let ticket = window.popLast() { try cache.unpinAfterGPUCompletion(ticket) }
        return (first, cache.snapshot().hits - first)
    }
    let clock = try replay(scan: false)
    let scan = try replay(scan: true)
    try require(clock.first == 0 && clock.second == 0, "CLOCK replay changed: \(clock) (scan rule \(scan))")
    // Sweep 1 hits both decode records; sweep 2 hits all 8 records the cache held, the most any policy can.
    try require(scan.first == 2 && scan.second == 8, "Prefill scan rule lost kept records: \(scan)")
    try require(ledger.snapshot().reserved == 0, "Replay caches kept slabs")
}

func originalSpanFallbackAndMutation() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-storage-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let payload = directory.appendingPathComponent("tensor.bin")
    try Data([7, 11, 13, 17]).write(to: payload)
    let spans: [String: Any] = ["language_model.test": ["file": "tensor.bin", "offset": 0, "length": 4, "dtype": "U8", "shape": [2, 2]]]
    let store = try PackStore(
        model: ModelSource(fixture: directory, spans: spans), pack: directory.appendingPathComponent("absent"), log: { _ in })
    try require(!store.status().packedAvailable, "Missing packs incorrectly enabled")
    var values = [UInt8](repeating: 0, count: 2)
    try values.withUnsafeMutableBytes { try store.readTensor("test", row: 1, into: $0) }
    try require(values == [13, 17], "Original row fallback returned wrong bytes")
    try mustFail { try values.withUnsafeMutableBytes { try store.readTensor("test", row: 2, into: $0) } }
    try mustFail { try values.withUnsafeMutableBytes { try store.readTensor("test", into: $0) } }
    let writer = try FileHandle(forWritingTo: payload)
    try writer.write(contentsOf: Data([99]))
    try writer.close()
    try mustFail { try store.validateBatchBoundary() }
    try mustFail { try values.withUnsafeMutableBytes { try store.readTensor("test", row: 1, into: $0) } }
    // A shard whose mtime moves before 1678 (APFS keeps -9223372037 s, beyond Int64 nanoseconds) reads as changed, not a trap.
    let later = try PackStore(
        model: ModelSource(fixture: directory, spans: spans), pack: directory.appendingPathComponent("absent"), log: { _ in })
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: -1e10)], ofItemAtPath: payload.path)
    try mustFail { try values.withUnsafeMutableBytes { try later.readTensor("test", row: 1, into: $0) } }
}
