import Foundation
import Metal
import OrigamiEngine

/// Gates on `FamilyExperts.readProbe`: a closed gate holds every read of its kind (drive, primary or replica) at its
/// start, a failing one makes it fail; each kind counts the reads that reached it.
private final class ReadGates: @unchecked Sendable {
    private let condition = NSCondition()
    private var closed: Set<String> = [], failing: Set<String> = []
    private var delays: [String: ClosedRange<Double>] = [:]
    private var entered: [String: Int] = [:]
    static func kind(_ source: IOSource, replica: Bool) -> String {
        (replica ? "replica-" : "primary-") + (source == .internalSSD ? "copy" : "gguf")
    }
    func close(_ kinds: String...) { change { closed.formUnion(kinds) } }
    func open(_ kinds: String...) { change { closed.subtract(kinds) } }
    func fail(_ kinds: String...) { change { failing.formUnion(kinds) } }
    func heal() { change { failing.removeAll() } }
    /// Each read of `kinds` sleeps a random time in `seconds` before it reads (nil: none).
    func delay(_ seconds: ClosedRange<Double>?, _ kinds: String...) { change { for kind in kinds { delays[kind] = seconds } } }
    func count(_ kind: String) -> Int { change { entered[kind] ?? 0 } }
    /// Waits up to five seconds for a read of `kind` to reach its gate.
    func awaitEntered(_ kind: String) -> Bool {
        let end = Date(timeIntervalSinceNow: 5)
        condition.lock()
        defer { condition.unlock() }
        while entered[kind] == nil { if !condition.wait(until: end) { return false } }
        return true
    }
    func probe(_ source: IOSource, _ replica: Bool) throws {
        let kind = Self.kind(source, replica: replica)
        if let delay = change({ delays[kind] }) { Thread.sleep(forTimeInterval: .random(in: delay)) }
        condition.lock()
        defer { condition.unlock() }
        entered[kind, default: 0] += 1
        condition.broadcast()
        while closed.contains(kind) { condition.wait() }
        if failing.contains(kind) { throw EngineError("injected \(kind) read failure") }
    }
    private func change<T>(_ body: () -> T) -> T {
        condition.lock()
        defer {
            condition.broadcast()
            condition.unlock()
        }
        return body()
    }
}

/// The two-layer expert fixture (layer 1's records 212,992 bytes, layer 2's 114,688; three slots a layer) with a hot
/// copy of `keys` (layer * 4 + expert) attached and its reads gated. `lanes`: (GGUF, copy) reads at once per drive.
private func hedgeFixture(
    _ name: String, keys: [Int] = Array(4..<12), lanes: (Int, Int)? = nil
) throws -> (FamilyExperts, ReadGates, URL) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hedge-\(name)-\(getpid())")
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let saved = FamilyExperts.checkReadLimits
    defer { FamilyExperts.checkReadLimits = saved }
    FamilyExperts.checkReadLimits = lanes
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    experts.adaptiveGuesses = false
    // A ledger of the fixture's own: the default one refuses the hedge arena while this Mac has under 3 GiB available.
    experts.ledger = try MemoryLedger(ceiling: 64 << 20, transientMargin: 1, availableFloor: 1)
    experts.attach(try HotCopy.write(source: experts, keys: keys, folder: folder.appendingPathComponent("hot"), identity: "h"))
    let gates = ReadGates()
    experts.readProbe = { [gates] source, replica in try gates.probe(source, replica) }
    return (experts, gates, folder)
}
private func holdsRecord(_ slot: FamilyExperts.Slot, layer: Int, expert: Int) -> Bool {
    let per = GGUFQuantize.bytes(layer == 1 ? 8 : 12, count: 256 * 256)
    return (0..<3).allSatisfy { part in
        UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per).allSatisfy {
            $0 == UInt8(layer * 50 + expert * 10 + part)
        }
    }
}
/// Waits up to five seconds for `condition`.
private func eventually(_ condition: () -> Bool) -> Bool {
    let end = Date(timeIntervalSinceNow: 5)
    while !condition() && Date() < end { Thread.sleep(forTimeInterval: 0.001) }
    return condition()
}
/// Opens `kinds` of `gates` after `seconds`, from another thread.
private func open(_ gates: ReadGates, _ kinds: String..., after seconds: Double) {
    DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { for kind in kinds { gates.open(kind) } }
}
/// A Family check's closing probe: the slot phases `FamilyEviction` mirrors agree with the core's.
func requireFamilyMirror(_ experts: FamilyExperts, _ what: String) throws {
    if let mismatch = experts.evictionMirrorMismatch() { throw EngineError("\(what): eviction mirror differs from the core: \(mismatch)") }
}

/// A demand read stalled on one drive is hedged on the other: the replica wins long before the stall ends and the fetch
/// returns the record's exact bytes. Copy stalled while reading: the replica's lease is used in place, the slot stays its
/// primary's (reading) until that read ends, then holds the same bytes. GGUF stalled while queued (one GGUF lane, held):
/// the replica win cancels the three part reads, its bytes go into the slot, which is returned verified.
func familyHedgedReadFiresOnStall() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedged read on stall: " + what) } }
    do {
        let (experts, gates, folder) = try hedgeFixture("copy")
        defer { try? FileManager.default.removeItem(at: folder) }
        let record = experts.bank(1)!.bytes.reduce(0, +)
        gates.close("primary-copy")
        open(gates, "primary-copy", after: 3)  // a fetch that never hedges ends, and fails below
        let begun = Date()
        let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
        let waited = Date().timeIntervalSince(begun)
        try require(gates.count("primary-copy") == 1 && gates.count("replica-gguf") == 1, "reads \(gates.count("replica-gguf"))")
        try require(waited < 2, "the fetch waited \(waited) s for a stalled drive")
        try require(holdsRecord(slots[0], layer: 1, expert: 2), "the replica's bytes differ")
        try require(
            experts.phase(layer: 1, expert: 2) == .reading,
            "the stalled read's slot is \(String(describing: experts.phase(layer: 1, expert: 2)))")
        let c = experts.counters
        try require(
            c.hedges == 1 && c.hedgeReplicaWins == 1 && c.hedgePrimaryWins == 0 && c.hedgeReplicaFailures == 0
                && c.hedgeBytes == record && c.misses == 1 && c.hotReads == 0,
            "counters \(c)")
        try require(experts.replicaArena?.snapshot().leasedRecords == 1, "the returned lease is not held")
        try requireFamilyMirror(experts, "copy stalled")
        gates.open("primary-copy")
        try require(eventually { experts.resident(layer: 1).contains(2) }, "the stalled read never landed")
        let again = try experts.fetch(layer: 1, experts: [2], decode: true)
        try require(holdsRecord(again[0], layer: 1, expert: 2) && experts.counters.misses == 1, "the landed slot")
        try require(experts.replicaArena?.snapshot().leasedRecords == 0, "the lease outlived its fetch's pins")
        try requireFamilyMirror(experts, "copy stall landed")
    }
    do {
        let (experts, gates, folder) = try hedgeFixture("gguf", lanes: (1, 2))
        defer { try? FileManager.default.removeItem(at: folder) }
        // The GGUF drive's one lane is held, so expert 2's three part reads queue behind it; expert 0 comes from the copy.
        let held = DispatchSemaphore(value: 0)
        let blocker = try experts.reads.submit(
            source: .externalSSD, priority: .demand, generation: experts.reads.currentGeneration, byteCount: 1,
            operation: { _ in held.wait() }, completion: { _, _ in })
        let begun = Date()
        let slots = try experts.fetch(layer: 1, experts: [0, 2], decode: true)
        let waited = Date().timeIntervalSince(begun)
        try require(waited < 2, "the fetch waited \(waited) s for a stalled drive")
        try require(holdsRecord(slots[0], layer: 1, expert: 0) && holdsRecord(slots[1], layer: 1, expert: 2), "bytes")
        try require(gates.count("primary-gguf") == 0 && gates.count("replica-copy") == 1, "the queued parts ran")
        try require(
            experts.phase(layer: 1, expert: 2) == .pinned && slots[1].buffer !== experts.replicaArena?.metalBuffer,
            "the replica's bytes were not placed in the slot")
        let c = experts.counters
        try require(c.hedges == 1 && c.hedgeReplicaWins == 1 && c.hotReads == 2 && c.misses == 2, "counters \(c)")
        try require(experts.replicaArena?.snapshot().leasedRecords == 0, "a copied lease is still held")
        held.signal()
        _ = blocker.join()
        try requireFamilyMirror(experts, "GGUF stalled")
    }
}

/// The primary ends just as its hedge began: the primary's bytes are published, the replica (still reading) is the loser
/// and its lease comes back once it ends.
func familyHedgedReadPrimaryWins() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge primary wins: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("primary-wins")
    defer { try? FileManager.default.removeItem(at: folder) }
    gates.close("primary-copy", "replica-gguf")
    open(gates, "primary-copy", "replica-gguf", after: 3)
    DispatchQueue.global().async {
        if gates.awaitEntered("replica-gguf") { gates.open("primary-copy") }
    }
    let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
    try require(holdsRecord(slots[0], layer: 1, expert: 2) && experts.phase(layer: 1, expert: 2) == .pinned, "bytes")
    let c = experts.counters
    try require(c.hedges == 1 && c.hedgePrimaryWins == 1 && c.hedgeReplicaWins == 0 && c.hotReads == 1, "counters \(c)")
    gates.open("replica-gguf")
    try require(eventually { experts.replicaArena?.snapshot().leasedRecords == 0 }, "the losing replica kept its lease")
    try requireFamilyMirror(experts, "primary wins")
}

/// Both drives fail the record: the fetch throws as it does today and nothing is resident; once healed it reads again.
func familyHedgedReadBothFail() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge both fail: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("both-fail")
    defer { try? FileManager.default.removeItem(at: folder) }
    gates.fail("primary-copy", "replica-gguf")
    var failed = false
    do { _ = try experts.fetch(layer: 1, experts: [2], decode: true) } catch { failed = true }
    try require(failed && experts.phase(layer: 1, expert: 2) == nil && experts.resident(layer: 1).isEmpty, "a failed record stayed")
    let c = experts.counters
    try require(c.hedges == 1 && c.hedgeReplicaFailures == 1 && c.hedgeReplicaWins == 0, "counters \(c)")
    try requireFamilyMirror(experts, "both failed")
    gates.heal()
    let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
    try require(holdsRecord(slots[0], layer: 1, expert: 2) && experts.counters.hedges == 1, "the healed read")
}

/// What never hedges: a record only the GGUF holds (it waits out its stall), a guess (prefetch) stalled behind a held
/// copy lane, and a normal fetch.
func familyHedgedReadExclusions() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge exclusions: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("exclusions", keys: [6, 9], lanes: (2, 1))
    defer { try? FileManager.default.removeItem(at: folder) }
    // Normal fetches, from both drives.
    for expert in [0, 1, 2] { _ = try experts.fetch(layer: 1, experts: [expert], decode: true) }
    try require(experts.counters.hedges == 0 && experts.replicaArena?.snapshot().arenaBytes == 0, "a healthy read hedged")
    // GGUF only (layer 2 expert 0).
    gates.close("primary-gguf")
    open(gates, "primary-gguf", after: 0.1)
    let begun = Date()
    let slots = try experts.fetch(layer: 2, experts: [0], decode: true)
    try require(Date().timeIntervalSince(begun) >= 0.09 && holdsRecord(slots[0], layer: 2, expert: 0), "the GGUF-only read")
    try require(experts.counters.hedges == 0, "a GGUF-only record hedged")
    // A guess of a copy-held record (layer 2 expert 1), queued behind the copy's one held lane.
    let held = DispatchSemaphore(value: 0)
    let blocker = try experts.reads.submit(
        source: .internalSSD, priority: .demand, generation: experts.reads.currentGeneration, byteCount: 1,
        operation: { _ in held.wait() }, completion: { _, _ in })
    experts.prefetch(layer: 2, experts: [1])
    Thread.sleep(forTimeInterval: 0.05)
    try require(experts.counters.prefetches == 1 && experts.counters.hedges == 0, "a guess hedged \(experts.counters)")
    held.signal()
    _ = blocker.join()
    let guessed = try experts.fetch(layer: 2, experts: [1], decode: true)
    try require(holdsRecord(guessed[0], layer: 2, expert: 1) && experts.counters.prefetchUsed == 1, "the guess")
    try require(experts.counters.hedges == 0 && experts.replicaArena?.snapshot().arenaBytes == 0, "hedges \(experts.counters)")
    try requireFamilyMirror(experts, "exclusions")
}

/// More stalled records than staging records (the arena holds two: a token's experts, at least two): at most two are
/// hedged, which of them the rule's estimates on this Mac's drive decide; the third waits out its stall, and the fetch
/// succeeds with exact bytes.
func familyHedgedReadArenaExhausted() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge arena exhausted: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("arena")
    defer { try? FileManager.default.removeItem(at: folder) }
    gates.close("primary-copy", "primary-gguf")
    open(gates, "primary-copy", "primary-gguf", after: 0.2)
    let begun = Date()
    let slots = try experts.fetch(layer: 1, experts: [0, 1, 2], decode: true)
    try require(Date().timeIntervalSince(begun) >= 0.19, "the unhedged record did not wait")
    try require((0..<3).allSatisfy { holdsRecord(slots[$0], layer: 1, expert: $0) }, "bytes")
    let staged = experts.replicaArena?.snapshot().highWaterRecords ?? 0
    try require((1...2).contains(staged) && experts.counters.hedges == staged, "\(staged) staged, \(experts.counters)")
    try require(eventually { experts.resident(layer: 1) == [0, 1, 2] }, "the records did not land")
    try requireFamilyMirror(experts, "arena exhausted")
}

/// A prompt wave read while both drives stall its primary reads, with the previous wave kept (`keepingLast`): the record is
/// hedged on the other drive, both waves' records stay exact, and the stalled read lands later.
func familyHedgedReadPromptWave() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge prompt wave: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("wave")
    defer { try? FileManager.default.removeItem(at: folder) }
    let first = try experts.fetch(layer: 1, experts: [0, 2])
    // Whichever drive the split gives it, its primary read stalls; its replica does not.
    gates.close("primary-copy", "primary-gguf")
    open(gates, "primary-copy", "primary-gguf", after: 3)
    let begun = Date()
    let second = try experts.fetch(layer: 1, experts: [1], keepingLast: true)
    try require(Date().timeIntervalSince(begun) < 2, "the wave waited for its stalled drive")
    try require(
        holdsRecord(first[0], layer: 1, expert: 0) && holdsRecord(first[1], layer: 1, expert: 2)
            && holdsRecord(second[0], layer: 1, expert: 1),
        "a wave's bytes")
    try require(experts.counters.hedges == 1 && experts.counters.hedgeReplicaWins == 1, "\(experts.counters)")
    try requireFamilyMirror(experts, "prompt wave")
    gates.open("primary-copy", "primary-gguf")
    try require(eventually { experts.resident(layer: 1) == [0, 1, 2] }, "the stalled read never landed")
    try require(holdsRecord(first[0], layer: 1, expert: 0) && holdsRecord(first[1], layer: 1, expert: 2), "the kept wave changed")
    try requireFamilyMirror(experts, "prompt wave landed")
}

/// The hedge rule's estimates. Healthy reads late by 20-40 ms (scheduling jitter) never hedge. A stalled copy read whose
/// other drive is busier than the copy's wait so far is not hedged (the estimate says the replica would lose); a failed
/// one still hedges at once, however busy the other drive.
func familyHedgedReadEstimates() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge estimates: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("estimates")
    defer { try? FileManager.default.removeItem(at: folder) }
    // The GGUF's drive has 8 GiB outstanding (seconds of reads): every record is read from the copy, and a replica on
    // the GGUF's drive would lose to any copy read that is merely late.
    let held = DispatchSemaphore(value: 0)
    let blocker = try experts.reads.submit(
        source: .externalSSD, priority: .demand, generation: experts.reads.currentGeneration, byteCount: 1 << 33,
        operation: { _ in held.wait() }, completion: { _, _ in })
    defer {
        held.signal()
        _ = blocker.join()
    }
    do {
        // Copy reads late by 20-40 ms (more where the OS coalesces timers): none is hedged.
        gates.delay(0.02...0.04, "primary-copy")
        for round in 0..<12 {
            for layer in [1, 2] { _ = try experts.fetch(layer: layer, experts: [round % 4, (round + 1) % 4], decode: true) }
        }
        gates.delay(nil, "primary-copy")
        let c = experts.counters
        try require(c.misses >= 24 && c.hotReads == c.misses, "the reads did not use the copy: \(c)")
        try require(c.hedges == 0 && experts.replicaArena?.snapshot().arenaBytes == 0, "jitter hedged: \(c)")
    }
    do {
        // Each layer has three slots for its four experts: one of each is not resident.
        let absent = [1, 2].map { layer in (0..<4).first { !experts.resident(layer: layer).contains($0) }! }
        gates.close("primary-copy")
        open(gates, "primary-copy", after: 0.3)
        let begun = Date()
        let slots = try experts.fetch(layer: 1, experts: [absent[0]], decode: true)
        try require(Date().timeIntervalSince(begun) >= 0.29 && holdsRecord(slots[0], layer: 1, expert: absent[0]), "the held read")
        try require(experts.counters.hedges == 0 && gates.count("replica-gguf") == 0, "a losing hedge fired: \(experts.counters)")
        let decision = experts.hedgeDecisions.last
        try require(
            decision?["decision"] as? String == "held: primary sooner" && decision?["winner"] as? String == "primary",
            "trace \(experts.hedgeDecisions)")
        // A failed copy read hedges at once to the busy drive, and its replica delivers the record.
        gates.fail("primary-copy")
        let failedAt = Date()
        let again = try experts.fetch(layer: 2, experts: [absent[1]], decode: true)
        try require(Date().timeIntervalSince(failedAt) < 1 && holdsRecord(again[0], layer: 2, expert: absent[1]), "the failed read")
        let c = experts.counters
        try require(c.hedges == 1 && c.hedgeReplicaWins == 1, "counters \(c)")
        try require(experts.hedgeDecisions.last?["decision"] as? String == "fired: primary failed", "trace \(experts.hedgeDecisions)")
        try requireFamilyMirror(experts, "estimates")
    }
}

/// The GGUF's drive in a slow period (every part read takes a second): a copy-held record read from it is hedged to the
/// copy at the minimum delay and the replica wins; the stall then counts toward the drive's current rate, so the next
/// copy-held records are read from the copy, unhedged; a record only the GGUF holds waits as before.
func familyHedgedReadSlowPeriod() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Hedge slow period: " + what) } }
    let (experts, gates, folder) = try hedgeFixture("slow", keys: Array(4..<11))
    defer { try? FileManager.default.removeItem(at: folder) }
    gates.delay(1...1, "primary-gguf")
    // Equal estimates: the first record goes to the copy, the second to the GGUF's drive.
    let begun = Date()
    let slots = try experts.fetch(layer: 1, experts: [0, 1], decode: true)
    let waited = Date().timeIntervalSince(begun)
    try require(holdsRecord(slots[0], layer: 1, expert: 0) && holdsRecord(slots[1], layer: 1, expert: 1), "bytes")
    try require(waited < 0.6, "the slow read was not rescued (\(waited) s)")
    try require(eventually { gates.count("primary-gguf") == 3 }, "the GGUF's part reads did not all begin")
    var c = experts.counters
    try require(c.hedges == 1 && c.hedgeReplicaWins == 1, "counters \(c)")
    let fired = experts.hedgeDecisions.last
    try require(
        fired?["decision"] as? String == "fired: replica sooner" && fired?["primary"] as? String == "original"
            && fired?["winner"] as? String == "replica",
        "trace \(experts.hedgeDecisions)")
    // The split now sees the stall: both records come from the copy.
    let next = try experts.fetch(layer: 1, experts: [2, 3], decode: true)
    try require(holdsRecord(next[0], layer: 1, expert: 2) && holdsRecord(next[1], layer: 1, expert: 3), "bytes")
    c = experts.counters
    try require(gates.count("primary-gguf") == 3 && c.hedges == 1, "the stalled drive was chosen again: \(c)")
    // Only the GGUF holds layer 2's expert 3: it waits out the slow read.
    let slowBegun = Date()
    let only = try experts.fetch(layer: 2, experts: [3], decode: true)
    try require(Date().timeIntervalSince(slowBegun) >= 0.99 && holdsRecord(only[0], layer: 2, expert: 3), "the GGUF-only read")
    try require(experts.counters.hedges == 1, "a GGUF-only record hedged")
    try require(eventually { experts.resident(layer: 1).contains(1) }, "the lost read never landed")
    try requireFamilyMirror(experts, "slow period")
}
