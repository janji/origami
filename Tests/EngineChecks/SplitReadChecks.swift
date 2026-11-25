import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func splitRequire(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Split reads: " + what) } }

private func splitSource(
    _ rate: Double, latency: Double = 1e-4, healthy: Double? = nil, busy: Double = 0, age: Double = 0, wearing: Bool = false
) -> SplitReadSource {
    SplitReadSource(
        bytesPerSecond: rate, latencySeconds: latency, healthyBytesPerSecond: healthy ?? rate, busySeconds: busy, measuredSecondsAgo: age,
        wearing: wearing)
}

/// Whether `parts` cover `record` bytes once, in order, one per source, every boundary on the alignment.
private func splitCovers(_ parts: [SplitReadPart], _ record: Int) -> Bool {
    var at = 0
    for part in parts.sorted(by: { $0.offset < $1.offset }) {
        guard part.offset == at, part.length > 0, part.offset % SplitReads.alignment == 0 else { return false }
        at += part.length
    }
    return at == record && Set(parts.map(\.source)).count == parts.count
}

/// Bytes no two nearby offsets share, so a misplaced range shows.
private func splitPattern(_ count: Int) -> [UInt8] { (0..<count).map { UInt8(truncatingIfNeeded: ($0 &* 2_654_435_761) >> 11 ^ $0) } }

/// The planner's gates and shapes: which reads are split, why the others are not, and parts in proportion to speed.
func splitReadPlanner() throws {
    let qwen = 2_764_800
    let family = 18_415_616
    // Lone reads as measured (a fast internal drive, a slower external one): both sizes split, fastest first, every part
    // predicted to end within one alignment unit's time of the others.
    for record in [qwen, family, 1_000_003] {
        let sources = [splitSource(3.2e9, latency: 1.5e-4), splitSource(2.4e9, latency: 2e-4)]
        guard case .split(let parts) = SplitReads.plan(recordBytes: record, sources: sources, waited: true) else {
            throw EngineError("Split reads: a lone \(record)-byte read on two idle drives was not split")
        }
        try splitRequire(splitCovers(parts, record) && parts[0].source == 0, "parts \(parts)")
        let ends = parts.map { sources[$0.source].seconds($0.length) }
        // Rounding to pages (and the record's partial last page) moves a part's end by at most two pages' time.
        try splitRequire(ends.max()! - ends.min()! <= 2 * Double(SplitReads.alignment) / 2.4e9, "ends \(ends)")
        try splitRequire(ends.max()! <= (1 - SplitReads.margin) * sources[0].seconds(record), "no clear gain")
    }
    let two = [splitSource(2.5e9), splitSource(2.5e9)]
    let notEligible = SplitReadPlan.whole(source: nil, nil)
    try splitRequire(SplitReads.plan(recordBytes: qwen, sources: two, waited: false) == notEligible, "read-ahead was split")
    try splitRequire(SplitReads.plan(recordBytes: qwen, sources: two, waited: true, enabled: false) == notEligible, "switch off")
    try splitRequire(SplitReads.plan(recordBytes: qwen, sources: [two[0]], waited: true) == notEligible, "one source")
    try splitRequire(SplitReads.plan(recordBytes: 20_000, sources: two, waited: true) == notEligible, "a record under two pages")
    // A fresh drive below a third of its healthy rate (a throttled one at 0.5 of 2 GB/s), or one wearing: no part, and the
    // record is read whole from the other.
    try splitRequire(
        SplitReads.plan(recordBytes: family, sources: [two[0], splitSource(0.5e9, healthy: 2e9)], waited: true)
            == .whole(source: 0, .unhealthy), "a stalled drive took a part")
    try splitRequire(
        SplitReads.plan(recordBytes: family, sources: [two[0], splitSource(2.4e9, wearing: true)], waited: true)
            == .whole(source: 0, .wear),
        "a wearing drive took a part")
    try splitRequire(
        SplitReads.plan(recordBytes: family, sources: [splitSource(2.5e9, wearing: true), two[1]], waited: true)
            == .whole(source: 1, .wear), "the wearing fastest drive split, or was read")
    // A healthy but much slower drive: the split would gain under the margin.
    try splitRequire(
        SplitReads.plan(recordBytes: family, sources: [two[0], splitSource(0.3e9)], waited: true) == .whole(source: 0, .smallGain),
        "small gain")
    // Three and four sources: parts in proportion to speed; the healthy idle ones all take a part.
    for rates in [[3e9, 2e9, 1e9], [4e9, 3e9, 2e9, 2e9]] {
        let sources = rates.map { splitSource($0, latency: 0) }
        guard case .split(let parts) = SplitReads.plan(recordBytes: family, sources: sources, waited: true) else {
            throw EngineError("Split reads: \(rates.count) sources were not split")
        }
        try splitRequire(splitCovers(parts, family) && parts.count == rates.count, "\(rates.count) sources: \(parts)")
        for part in parts {
            let share = Double(part.length) / Double(family)
            let expected = rates[part.source] / rates.reduce(0, +)
            try splitRequire(abs(share - expected) <= Double(SplitReads.alignment) / Double(family) + 1e-9, "share \(share) of \(expected)")
        }
    }
    // A source whose fixed cost per read alone outlasts the others' parts, or whose part would be under a page: dropped,
    // its bytes go to the faster ones.
    for slow in [splitSource(2.5e9, latency: 0.01), splitSource(5e6, latency: 0)] {
        guard case .split(let parts) = SplitReads.plan(recordBytes: qwen, sources: [two[0], two[1], slow], waited: true) else {
            throw EngineError("Split reads: a useless third source stopped the split")
        }
        try splitRequire(splitCovers(parts, qwen) && Set(parts.map(\.source)) == [0, 1], "small part kept: \(parts)")
    }
}

/// Synthetic sources over one pattern: each reads its range after its delay (or fails), writing `poison` instead when set
/// (a late staged read must never reach the destination), and counts the reads still writing into `destination`.
private final class SplitSources: @unchecked Sendable {
    let pattern: [UInt8]
    private let lock = NSCondition()
    var delays: [Double]
    var failing: Set<Int> = []
    var poison: Set<Int> = []
    var destination: UnsafeMutableRawBufferPointer?
    private(set) var inPlace = 0, running = 0, reads: [Int] = []
    /// Where each read wrote (its first byte).
    private(set) var addresses: [UInt] = []
    /// Bytes per second each source reads at (a sleep of bytes / rate per read); nil: no sleep.
    var rates: [Double]?
    init(_ count: Int, sources: Int) {
        pattern = splitPattern(count)
        delays = Array(repeating: 0, count: sources)
    }
    func read(_ source: Int, _ offset: Int, _ into: UnsafeMutableRawBufferPointer) throws {
        let placed = destination.map { into.baseAddress! >= $0.baseAddress! && into.baseAddress! < $0.baseAddress! + $0.count } ?? false
        let (delay, fails, poisoned) = lock.withLock {
            reads.append(source)
            addresses.append(UInt(bitPattern: into.baseAddress))
            running += 1
            if placed { inPlace += 1 }
            return (
                delays[source] + (rates.map { Double(into.count) / $0[source] } ?? 0), failing.contains(source), poison.contains(source)
            )
        }
        defer {
            lock.withLock {
                running -= 1
                if placed { inPlace -= 1 }
            }
        }
        if rates != nil {
            // A drive at a given speed: spun, since a sleep of milliseconds can overshoot many times over.
            let end = DispatchTime.now().uptimeNanoseconds + UInt64(delay * 1e9)
            while DispatchTime.now().uptimeNanoseconds < end {}
        } else if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        if fails { throw EngineError("injected failure of source \(source)") }
        if poisoned {
            into.initializeMemory(as: UInt8.self, repeating: 0xEE)
        } else {
            pattern.withUnsafeBytes { into.copyMemory(from: UnsafeRawBufferPointer(rebasing: $0[offset..<(offset + into.count)])) }
        }
    }
    var placedReads: Int { lock.withLock { inPlace } }
    var runningReads: Int { lock.withLock { running } }
    var readCount: Int { lock.withLock { reads.count } }
}

/// Runs `parts` over `sources` into a fresh destination (staged or not); the destination's bytes and the outcome.
private func splitRun(
    _ parts: [SplitReadPart], _ sources: SplitSources, staged: Bool, late: Double = 10, order: [Int]? = nil,
    cancelled: @escaping () -> Bool = { false }
) throws -> ([UInt8], SplitReadOutcome) {
    let count = sources.pattern.count
    let destination = UnsafeMutableRawBufferPointer.allocate(byteCount: count, alignment: SplitReads.alignment)
    defer { destination.deallocate() }
    destination.initializeMemory(as: UInt8.self, repeating: 0)
    sources.destination = destination
    let staging = staged ? UnsafeMutableRawBufferPointer.allocate(byteCount: count, alignment: SplitReads.alignment) : nil
    final class Keep {
        let buffer: UnsafeMutableRawBufferPointer?
        init(_ buffer: UnsafeMutableRawBufferPointer?) { self.buffer = buffer }
        deinit { buffer?.deallocate() }
    }
    let outcome = try SplitReads.execute(
        parts, into: destination, order: order ?? Array(Set(parts.map(\.source))).sorted(), staging: staging, keep: Keep(staging),
        lateAfter: { _ in late }, cancelled: cancelled, read: sources.read)
    try splitRequire(sources.placedReads == 0, "a read still writes into the destination after the split returned")
    return (Array(destination), outcome)
}

/// Execution: every shape gives the whole record's bytes; a late staged part is raced and the first complete version
/// wins (the late one never reaches the destination); a failed part is read again from another source; a range every
/// source failed fails the read; cancellation returns only once nothing writes into the destination.
func splitReadExecution() throws {
    for record in [2_764_800, 18_415_616, 3 * SplitReads.alignment + 7, 1_000_003] {
        for count in [2, 3, 4] {
            let rates = (0..<count).map { 3e9 - Double($0) * 0.5e9 }
            guard
                case .split(let parts) = SplitReads.plan(
                    recordBytes: record, sources: rates.map { splitSource($0, latency: 0) }, waited: true)
            else { throw EngineError("Split reads: \(record) bytes over \(count) sources were not split") }
            for staged in [false, true] {
                let sources = SplitSources(record, sources: count)
                let (bytes, outcome) = try splitRun(parts, sources, staged: staged)
                try splitRequire(
                    bytes == sources.pattern, "\(record) bytes over \(count) sources (staged \(staged)) differ from a whole read")
                try splitRequire(outcome.guards == 0 && outcome.fallbacks == 0 && sources.readCount == parts.count, "extra reads")
                try splitRequire(outcome.bytesBySource.values.reduce(0, +) == record, "bytes by source \(outcome.bytesBySource)")
            }
            // A delay of infinity, a huge one or NaN never makes a part late (and must not trap).
            for late in [Double.infinity, 1e300, .nan] {
                let sources = SplitSources(record, sources: count)
                let (bytes, outcome) = try splitRun(parts, sources, staged: true, late: late)
                try splitRequire(bytes == sources.pattern && outcome.guards == 0, "late after \(late): guards \(outcome.guards)")
            }
        }
    }
    let record = 2_764_800
    let parts = [SplitReadPart(source: 0, offset: 0, length: 1_392_640), SplitReadPart(source: 1, offset: 1_392_640, length: 1_372_160)]
    do {
        // Source 1 stalls and would write garbage: its range is read again from source 0 after 20 ms and wins.
        let sources = SplitSources(record, sources: 2)
        sources.delays[1] = 0.4
        sources.poison = [1]
        let begun = Date()
        let (bytes, outcome) = try splitRun(parts, sources, staged: true, late: 0.02)
        try splitRequire(Date().timeIntervalSince(begun) < 0.3, "the guard did not end the wait")
        try splitRequire(bytes == sources.pattern && outcome.guards == 1, "guarded bytes (guards \(outcome.guards))")
        try splitRequire(outcome.bytesBySource == [0: record], "the late part was counted: \(outcome.bytesBySource)")
    }
    do {
        // Its first guard (source 2) fails while the late part still reads: the next source guards it at once, without
        // waiting for the part.
        let sources = SplitSources(record, sources: 3)
        sources.delays[1] = 0.4
        sources.poison = [1]
        sources.failing = [2]
        let begun = Date()
        let (bytes, outcome) = try splitRun(parts, sources, staged: true, late: 0.02, order: [2, 0, 1])
        try splitRequire(Date().timeIntervalSince(begun) < 0.3, "a failed guard waited for the late part")
        try splitRequire(bytes == sources.pattern && outcome.guards == 2, "second guard's bytes (guards \(outcome.guards))")
    }
    do {
        // A late part that ends while its guard still reads: copied in only after the guard ended (same bytes).
        let sources = SplitSources(record, sources: 2)
        sources.delays = [0.15, 0.05]
        let (bytes, outcome) = try splitRun(parts, sources, staged: true, late: 0.01)
        try splitRequire(bytes == sources.pattern, "a part racing its guard changed the bytes (guards \(outcome.guards))")
    }
    for failing in [Set([1]), Set([0])] {
        // A failed part (staged, or the fastest one in place) is read again from the other source.
        let sources = SplitSources(record, sources: 2)
        sources.failing = failing
        let (bytes, outcome) = try splitRun(parts, sources, staged: true)
        try splitRequire(bytes == sources.pattern && outcome.fallbacks == 1, "failed \(failing): fallbacks \(outcome.fallbacks)")
    }
    do {
        let sources = SplitSources(record, sources: 2)
        sources.failing = [0, 1]
        var threw = false
        do { _ = try splitRun(parts, sources, staged: true) } catch is EngineError { threw = true }
        try splitRequire(threw, "a range no source read did not fail the read")
    }
    do {
        // Cancelled while both parts read into place: it waits for them, then throws.
        let sources = SplitSources(record, sources: 2)
        sources.delays = [0.1, 0.1]
        let begun = Date()
        var threw = false
        do {
            _ = try splitRun(parts, sources, staged: false, cancelled: { Date().timeIntervalSince(begun) > 0.02 })
        } catch is IOOperationCancelled {
            threw = true
        }
        try splitRequire(threw && Date().timeIntervalSince(begun) >= 0.09, "cancellation returned before its writers ended")
    }
    // The pieces of a range of a record kept as spans back to back.
    let pieces = SplitReads.pieces(of: 5..<23, spans: [(100, 10), (200, 4), (300, 20)])
    try splitRequire(
        pieces.map { [$0.span, Int($0.offset), $0.length, $0.at] } == [[0, 105, 5, 0], [1, 200, 4, 5], [2, 300, 9, 9]], "pieces \(pieces)")
}

/// GLM and DeepSeek (one read path, `FamilyExperts.fetch`): a lone decode miss the copy holds is split across both
/// drives, lands with the exact bytes, verified against the copy's checksum, and is counted; with splits off the same
/// fetch reads whole (the hedged-read checks run that way); a guess in flight keeps it whole (busy).
func splitReadFamilyFetch() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    for on in [true, false] {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-\(on)-\(getpid())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
        let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
        experts.adaptiveGuesses = false
        experts.splitsReads = on
        experts.disk = { _ in nil }
        let record = experts.bank(1)!.bytes.reduce(0, +)
        experts.attach(try HotCopy.write(source: experts, keys: Array(4..<12), folder: folder.appendingPathComponent("hot"), identity: "h"))
        let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
        let per = GGUFQuantize.bytes(8, count: 256 * 256)
        let exact = (0..<3).allSatisfy { part in
            UnsafeRawBufferPointer(start: slots[0].buffer.contents() + slots[0].offsets[part], count: per).allSatisfy {
                $0 == UInt8(50 + 20 + part)
            }
        }
        let report = experts.splitReads.report
        let c = experts.counters
        try splitRequire(exact, "the fetched record's bytes (splits \(on))")
        try splitRequire(report["lone_demand_waits"] as? Int == 1, "the lone wait was not counted: \(report)")
        try splitRequire(c.copyBytes + c.modelBytes == record && c.misses == 1, "bytes per source \(c.copyBytes) + \(c.modelBytes)")
        if on {
            let bytes = report["part_bytes"] as? [String: Int] ?? [:]
            try splitRequire(
                report["split_reads"] as? Int == 1 && bytes["copy0", default: 0] + bytes["model", default: 0] == record
                    && bytes["copy0"] == c.copyBytes, "not split: \(report)")
            // Another read in flight on a drive no longer keeps a lone miss whole: its part starts after that read's bytes.
            let held = DispatchSemaphore(value: 0)
            let blocker = try experts.reads.submit(
                source: .externalSSD, priority: .demand, generation: experts.reads.currentGeneration, byteCount: 1,
                operation: { _ in held.wait() }, completion: { _, _ in })
            _ = try experts.fetch(layer: 1, experts: [3], decode: true)
            held.signal()
            _ = blocker.join()
            let after = experts.splitReads.report
            let sources = after["sources"] as? [String: [String: Any]] ?? [:]
            try splitRequire(
                after["split_reads"] as? Int == 2 && (after["read_whole"] as? [String: Int])?["busy"] == nil
                    && sources["copy0"]?["bytes_per_second"] is Double && sources["model"]?["measured_seconds_ago"] != nil,
                "busy drive: \(after)")
        } else {
            try splitRequire(report["split_reads"] as? Int == 0 && experts.counters.hotReads == 1, "switch off split: \(report)")
        }
        try requireFamilyMirror(experts, "split reads")
    }
}

/// Waits up to two seconds for `condition`.
private func splitEventually(_ condition: () -> Bool) -> Bool {
    let end = Date(timeIntervalSinceNow: 2)
    while !condition() && Date() < end { Thread.sleep(forTimeInterval: 0.001) }
    return condition()
}

/// A Qwen loader as decode runs it (replica race and arena, decoding), with part reads from `sources`, whole reads
/// from `whole`, an online estimator when `estimator` is given, and `concurrent` reads at once per drive.
private func splitQwenLoader(
    sources: SplitSources, whole: SplitSources, estimator: ExpertSourceOnlineEstimator? = nil, split: Bool = true, concurrent: Int = 4,
    replicas: Bool = true,
    measured: @escaping @Sendable (IOSource) -> (bytesPerSecond: Double, secondsAgo: Double, healthy: Double) = { _ in (0, .infinity, 0) }
) throws -> (SourceExpertLoader, IOScheduler, ExpertCache, ExpertReplicaArena) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(4 * PackStore.expertBytes), slotsPerSlab: 4)
    let limits = IOScheduler.Limits(
        maximumConcurrent: concurrent, reservedDemandConcurrent: min(2, concurrent), maximumOutstanding: 16,
        reservedDemandOutstanding: 8, maximumOutstandingBytes: 64_000_000, reservedDemandBytes: 32_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    let arena = try ExpertReplicaArena(ledger: ledger, device: device)
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: arena, estimator: estimator, packedAvailable: { true }
    ) { _, source, destination in try whole.read(source == .internalSSD ? 0 : 1, 0, destination) }
    if split {
        loader.enableSplitReads(
            .init(
                available: { _ in true }, wearing: { _, _ in false }, measured: measured,
                read: { _, source, offset, into, _ in try sources.read(source == .internalSSD ? 0 : 1, offset, into) },
                verify: { _, record, _ in
                    guard record.elementsEqual(sources.pattern) else { throw EngineError("assembled record differs") }
                }))
        loader.splitsReads = true
    }
    try loader.setReplicasEnabled(replicas)
    loader.setDecoding(true)
    return (loader, scheduler, cache, arena)
}
private let splitCalibration = ExpertSourceCalibration(
    queueDepths: [1], internalBytesPerSecond: [2.5e9], externalBytesPerSecond: [2.4e9], internalLatencySeconds: 1e-4,
    externalLatencySeconds: 1e-4, label: "split-fixture", replicaDelaySeconds: 1)

/// Qwen (`SourceExpertLoader`, decode with the replica race): a lone demand read both drives hold is split and published
/// as one record with the exact bytes, and its wait is counted; read-ahead and the switch off read whole; while a split
/// runs, its parts count on both drives, so a read planned then is `busy`; cancelling a split, cancelling its
/// generation or shutting the scheduler down leaves no reader running and every staging lease back in the arena.
func splitReadQwenLoader() throws {
    let sources = SplitSources(PackStore.expertBytes, sources: 2)
    let whole = SplitSources(PackStore.expertBytes, sources: 2)
    let (loader, scheduler, cache, arena) = try splitQwenLoader(sources: sources, whole: whole)
    defer { scheduler.shutdown() }
    let generation = scheduler.currentGeneration
    func request(_ expert: Int) throws -> ExpertLoadRequest {
        try loader.request(
            try ExpertKey(layer: 0, expert: expert), generation: generation, policy: .earliestCompletion, calibration: splitCalibration)
    }
    let loaded = try loader.joinAndPin(try request(1))
    try splitRequire(Array(loaded.buffer.bytes) == sources.pattern, "the split record's bytes")
    try loaded.retireAfterGPUCompletion()
    let report = loader.splitReads.report
    try splitRequire(
        report["split_reads"] as? Int == 1 && sources.readCount == 2 && whole.readCount == 0 && report["lone_demand_waits"] as? Int == 1,
        "not split: \(report), \(sources.reads) part and \(whole.reads) whole reads")
    // Read-ahead never splits.
    if case .issued(let guess) = try loader.requestSpeculative(
        try ExpertKey(layer: 0, expert: 2), generation: generation, policy: .earliestCompletion, calibration: splitCalibration)
    {
        loader.joinDiscarding(guess)
    }
    try splitRequire(whole.readCount == 1 && sources.readCount == 2, "a guess was split")
    // The switch off: today's whole read.
    loader.splitsReads = false
    let off = try loader.joinAndPin(try request(3))
    loader.splitsReads = true
    try off.retireAfterGPUCompletion()
    try splitRequire(
        whole.readCount == 2 && loader.splitReads.splitCount == 1 && loader.splitReads.report["lone_demand_waits"] as? Int == 2,
        "the switch off still split, or its lone wait was not counted")
    // During a split its parts' bytes are planned on its sources (what a read planned then finds them busy with); none
    // once it ended.
    sources.delays = [0.15, 0.15]
    let leases = arena.snapshot().leasedRecords
    let running = try request(0)
    try splitRequire(splitEventually { sources.runningReads == 2 }, "the split's parts did not start")
    try splitRequire(loader.plannedBytes["model"] > 0 && loader.plannedBytes["pack"] > 0, "the parts' bytes are not planned")
    try (try loader.joinAndPin(running)).retireAfterGPUCompletion()
    try splitRequire(splitEventually { loader.plannedBytes.total == 0 }, "planned bytes left: \(loader.plannedBytes.total)")
    // Cancelled while its parts read: the wait ends once they have, and the slot is free again.
    sources.destination = nil
    // Parts a second long: the cancel at 20 ms comes while they read, also on a loaded Mac or under a sanitizer.
    sources.delays = [1, 1]
    let begun = Date()
    let slow = try request(7)
    var threw = false
    do { _ = try loader.joinAndPin(slow) { Date().timeIntervalSince(begun) < 0.02 } } catch is IOOperationCancelled { threw = true }
    loader.joinDiscarding(slow)
    try splitRequire(
        threw && sources.runningReads == 0,
        "a cancelled split's reader outlived its join: threw \(threw), running \(sources.runningReads), reads \(sources.reads)")
    try splitRequire(
        splitEventually { cache.snapshot().inFlightReads == 0 && arena.snapshot().leasedRecords == leases }, "slot or lease kept")
    // Its generation cancelled, then the scheduler shut down, with a slow staged part running: joined, leases back.
    sources.delays = [0, 0.3]
    let staged = try request(5)
    try splitRequire(splitEventually { sources.runningReads >= 1 }, "no part started")
    try loader.cancelAndJoin(generation)
    try splitRequire(sources.runningReads == 0, "cancelAndJoin left a part reader running")
    withExtendedLifetime(staged) {}
    try splitRequire(splitEventually { arena.snapshot().leasedRecords == leases }, "a lease outlived its cancelled split")
    let next = scheduler.currentGeneration
    let last = try loader.request(
        try ExpertKey(layer: 0, expert: 6), generation: next, policy: .earliestCompletion, calibration: splitCalibration)
    try splitRequire(splitEventually { sources.runningReads >= 1 }, "no part started before shutdown")
    scheduler.shutdown()
    try splitRequire(sources.runningReads == 0, "shutdown left a part reader running")
    withExtendedLifetime(last) {}
    try splitRequire(splitEventually { arena.snapshot().leasedRecords == leases }, "a lease outlived shutdown")
}

/// Every part trains its drive's live estimate as a whole read does: 100 split reads with the pack at half speed leave
/// the online estimator's multiples where 100 whole reads at those speeds leave them (Qwen); a family drive's moving
/// rate learns the same from parts as from whole records (`FamilyExperts.learn`).
func splitReadTrainsEstimates() throws {
    // A quick average: the first reads into fresh slots are slow (page faults), and the planner judges health on it.
    let settings = ExpertSourceOnlineCalibration(smoothing: 0.2, warmupObservations: 8)
    let calibration = ExpertSourceCalibration(
        queueDepths: [1], internalBytesPerSecond: [0.25e9], externalBytesPerSecond: [0.25e9], label: "split-training",
        replicaDelaySeconds: 1)
    let rates = [0.25e9, 0.125e9]
    func multiples(split: Bool) throws -> (Double, Double, Int) {
        let estimator = try ExpertSourceOnlineEstimator(calibration: calibration, settings: settings, maximumConcurrent: 4)
        let sources = SplitSources(PackStore.expertBytes, sources: 2)
        let whole = SplitSources(PackStore.expertBytes, sources: 2)
        sources.rates = rates
        whole.rates = rates
        let (loader, scheduler, _, _) = try splitQwenLoader(sources: sources, whole: whole, estimator: estimator, split: split)
        defer { scheduler.shutdown() }
        let generation = scheduler.currentGeneration
        for n in 0..<100 {
            let key = try ExpertKey(layer: n % 40, expert: n / 40)
            let request =
                split
                ? try loader.request(key, generation: generation, policy: .earliestCompletion, calibration: calibration)
                : try loader.request(key, generation: generation, source: n % 2 == 0 ? .internalSSD : .externalSSD)
            try (try loader.joinAndPin(request)).retireAfterGPUCompletion()
            // One read at a time, as the estimator is told: the previous job has fully ended before the next starts.
            _ = splitEventually { scheduler.snapshot().drives.allSatisfy { $0.runningDemand + $0.runningPrefetch == 0 } }
        }
        let snapshot = estimator.snapshot()
        return (snapshot.internalSSD.smoothedMultiple, snapshot.externalSSD.smoothedMultiple, loader.splitReads.splitCount)
    }
    let parts = try multiples(split: true)
    let wholes = try multiples(split: false)
    try splitRequire(
        parts.2 >= 80 && abs(parts.0 / wholes.0 - 1) < 0.25 && abs(parts.1 / wholes.1 - 1) < 0.25 && parts.1 > 1.5 * parts.0,
        "estimates from parts \(parts) against whole reads \(wholes)")
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-learn-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let a = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    let b = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    var (fromParts, fromWholes) = (0.0, 0.0)
    for _ in 0..<100 {
        fromParts = a.learnForChecks(drive: 1, bytes: 81_920, seconds: 81_920 / 1e9)
        fromWholes = b.learnForChecks(drive: 1, bytes: 208_896, seconds: 208_896 / 1e9)
    }
    try splitRequire(abs(fromParts / fromWholes - 1) < 0.01 && abs(fromParts / 1e9 - 1) < 0.01, "family rates \(fromParts), \(fromWholes)")
}

/// GLM and DeepSeek: a hedge replica that wins while a split still reads publishes the model's bytes, and the split
/// never writes the record afterwards, even with a staged part holding wrong bytes (200 runs, random delays); with a
/// slow staged part, shutting the scheduler down leaves no reader running and the staging lease back in the arena.
func splitReadFamilyHedgeWins() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-hedge-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let per = GGUFQuantize.bytes(8, count: 256 * 256)
    func exact(_ slot: FamilyExperts.Slot) -> Bool {
        (0..<3).allSatisfy { part in
            UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per).allSatisfy {
                $0 == UInt8(50 + 20 + part)
            }
        }
    }
    final class Probe: @unchecked Sendable {
        let lock = NSLock()
        var inPlace = 0, running = 0
        var stagedDelay = 0.0, guardDelay = 0.0
        var outstanding: [Int] = []
    }
    var replicaWins = 0
    for run in 0..<200 {
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
        experts.adaptiveGuesses = false
        experts.splitsReads = true
        experts.disk = { _ in nil }
        experts.attach(
            try HotCopy.write(source: experts, keys: Array(4..<12), folder: folder.appendingPathComponent("hot\(run)"), identity: "h"))
        let probe = Probe()
        probe.stagedDelay = .random(in: 0...0.12)
        probe.guardDelay = .random(in: 0...0.12)
        let reads = experts.reads
        experts.splitPartProbe = { _, staged, into in
            let delay = probe.lock.withLock { () -> Double in
                probe.running += 1
                if staged { return probe.stagedDelay }
                probe.inPlace += 1
                return probe.inPlace > 1 ? probe.guardDelay : 0
            }
            if staged { probe.lock.withLock { probe.outstanding = reads.snapshot().drives.map(\.outstanding) } }
            Thread.sleep(forTimeInterval: delay)
            if staged { into.initializeMemory(as: UInt8.self, repeating: 0xEE) }
            probe.lock.withLock { probe.running -= 1 }
        }
        let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
        try splitRequire(
            exact(slots[0]), "run \(run): the published record differs (staged \(probe.stagedDelay), guard \(probe.guardDelay))")
        if experts.counters.hedgeReplicaWins > 0 { replicaWins += 1 }
        try splitRequire(
            splitEventually {
                probe.lock.withLock({ probe.running }) == 0 && experts.reads.snapshot().drives.allSatisfy { $0.outstanding == 0 }
            },
            "run \(run): the split never ended: running \(probe.lock.withLock { probe.running }), drives \(experts.reads.snapshot().drives.map { ($0.queuedDemand, $0.runningDemand, $0.outstanding) }), hedge wins \(experts.counters.hedgeReplicaWins)"
        )
        try splitRequire(exact(slots[0]), "run \(run): the record changed after the split ended")
        let again = try experts.fetch(layer: 1, experts: [2], decode: true)
        try splitRequire(exact(again[0]), "run \(run): the record read again differs")
        if run == 0, let seen = probe.lock.withLock({ probe.outstanding.isEmpty ? nil : probe.outstanding }) {
            try splitRequire(seen.allSatisfy { $0 >= 1 }, "a staged part is not charged to its drive: \(seen)")
        }
        try requireFamilyMirror(experts, "split hedge run \(run)")
    }
    try splitRequire(replicaWins > 0, "no run had the hedge win mid-split")
    // A slow staged part, its range won by the guard: the fetch returns; shutdown joins the staged reader.
    let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    experts.adaptiveGuesses = false
    experts.splitsReads = true
    experts.disk = { _ in nil }
    experts.attach(try HotCopy.write(source: experts, keys: Array(4..<12), folder: folder.appendingPathComponent("hotB"), identity: "h"))
    let probe = Probe()
    experts.splitPartProbe = { _, staged, _ in
        guard staged else { return }
        probe.lock.withLock { probe.running += 1 }
        Thread.sleep(forTimeInterval: 0.4)
        probe.lock.withLock { probe.running -= 1 }
    }
    let leases = experts.replicaArena?.snapshot().leasedRecords ?? 0
    let slots = try experts.fetch(layer: 1, experts: [2], decode: true)
    try splitRequire(exact(slots[0]) && probe.lock.withLock({ probe.running }) == 1, "the guarded split")
    experts.reads.shutdown()
    try splitRequire(probe.lock.withLock { probe.running } == 0, "shutdown left a staged reader running")
    // A hedge replica that won while the split still read stays pinned for its slot: that lease is not the staging's.
    let pinned = experts.counters.hedgeReplicaWins
    try splitRequire(
        splitEventually { (experts.replicaArena?.snapshot().leasedRecords ?? 0) <= leases + pinned },
        "the staging lease was kept (\(experts.replicaArena?.snapshot().leasedRecords ?? 0) leased, \(leases) before, \(pinned) hedge wins)"
    )
}

/// GLM and DeepSeek with two copy places (`attach(copies:)`): a record both places hold splits three ways with the
/// GGUF, one a single place holds two ways; bytes exact; parts are no records in the demand timing counters.
func splitReadFamilyPlaces() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-places-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    experts.adaptiveGuesses = false
    experts.splitsReads = true
    experts.disk = { _ in nil }
    let a = try HotCopy.write(source: experts, keys: [4, 5, 6], folder: folder.appendingPathComponent("a"), identity: "h")
    let b = try HotCopy.write(source: experts, keys: [4, 5], folder: folder.appendingPathComponent("b"), identity: "h")
    experts.attach(copies: [a, b])
    let per = GGUFQuantize.bytes(8, count: 256 * 256)
    func exact(_ slot: FamilyExperts.Slot, _ expert: Int) -> Bool {
        (0..<3).allSatisfy { part in
            UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per).allSatisfy {
                $0 == UInt8(50 + expert * 10 + part)
            }
        }
    }
    func used() -> Int { (experts.splitReads.report["part_bytes"] as? [String: Int] ?? [:]).filter { $0.value > 0 }.count }
    try splitRequire(
        exact(try experts.fetch(layer: 1, experts: [0], decode: true)[0], 0) && used() == 3, "three ways: \(experts.splitReads.report)")
    let c = experts.counters
    // A fresh cache (its drive estimates untrained by the first split): a record one place holds splits two ways.
    let single = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    single.adaptiveGuesses = false
    single.splitsReads = true
    single.disk = { _ in nil }
    single.attach(copies: [a, b])
    try splitRequire(exact(try single.fetch(layer: 1, experts: [2], decode: true)[0], 2), "one place")
    let parts = single.splitReads.report["part_bytes"] as? [String: Int] ?? [:]
    try splitRequire(
        single.splitReads.splitCount == 1 && parts["copy1", default: 0] == 0 && parts.filter({ $0.value > 0 }).count == 2,
        "one place: \(single.splitReads.report)")
    try splitRequire(
        c.demandHotRecords + c.demandMainRecords == 0, "parts counted as records: \(c.demandHotRecords), \(c.demandMainRecords)")
}

/// Runs `body` on another thread: whether it returned within `seconds`.
private func splitPromptly(_ seconds: Double = 2, _ body: @escaping @Sendable () -> Void) -> Bool {
    let done = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        body()
        done.signal()
    }
    return done.wait(timeout: .now() + seconds) == .success
}

/// Holds `scheduler`'s starts until `queued` (the split's own job) waits on one drive, puts a slow read (it ends only
/// when cancelled, or after 5 s) on the other drive, then lets both start: the split's part on that drive queues behind
/// the slow read (one read at a time per drive). Then cancels the generation or shuts the scheduler down.
private func splitQueuePartThenCancel(
    _ scheduler: IOScheduler, shutdown: Bool, hold: IOPrefetchHold, cancel: @escaping @Sendable () throws -> Void
) throws {
    try splitRequire(splitEventually { scheduler.snapshot().drives.contains { $0.queuedDemand == 1 } }, "the split's job did not queue")
    let parent = scheduler.snapshot().drives.first { $0.queuedDemand == 1 }!.source
    let other: IOSource = parent == .internalSSD ? .externalSSD : .internalSSD
    try scheduler.submit(
        source: other, priority: .demand, generation: scheduler.currentGeneration, byteCount: 1,
        operation: { cancellation in
            let end = Date(timeIntervalSinceNow: 5)
            while Date() < end {
                try cancellation.throwIfCancelled()
                Thread.sleep(forTimeInterval: 0.002)
            }
        })
    hold.release()
    try splitRequire(
        splitEventually { scheduler.snapshot().drives.first { $0.source == other }!.queuedDemand == 1 },
        "the part did not queue behind the slow read: \(scheduler.snapshot().drives)")
    try splitRequire(
        splitPromptly { if shutdown { scheduler.shutdown() } else { try? cancel() } },
        "shutdown or the cancel waited on a split whose part was queued")
}

/// Qwen's split parts hold no queue slot (a part waiting for a slot that split reads' own jobs hold would never start):
/// with the other drive's only slot taken by a slow read, a split still ends at once, exact. The families' parts are
/// jobs on their drives (only a lone miss is split there): one still queued behind a slow read when its generation is
/// cancelled or its scheduler shut down ends without running, and the part ends with it, so the split returns at once
/// (not at the stall bound), no reader is left, the staging lease is back and `shutdown` returns promptly.
func splitReadQueuedPartCancel() throws {
    do {
        let sources = SplitSources(PackStore.expertBytes, sources: 2)
        let whole = SplitSources(PackStore.expertBytes, sources: 2)
        let (loader, scheduler, _, _) = try splitQwenLoader(sources: sources, whole: whole, concurrent: 1)
        let hold = scheduler.holdStarts()
        let request = try loader.request(
            try ExpertKey(layer: 0, expert: 1), generation: scheduler.currentGeneration, policy: .earliestCompletion,
            calibration: splitCalibration)
        try splitRequire(splitEventually { scheduler.snapshot().drives.contains { $0.queuedDemand == 1 } }, "the split's job did not queue")
        let other: IOSource =
            scheduler.snapshot().drives.first { $0.queuedDemand == 1 }!.source == .internalSSD ? .externalSSD : .internalSSD
        try scheduler.submit(
            source: other, priority: .demand, generation: scheduler.currentGeneration, byteCount: 1,
            operation: { cancellation in
                while true {
                    try cancellation.throwIfCancelled()
                    Thread.sleep(forTimeInterval: 0.002)
                }
            })
        hold.release()
        let begun = Date()
        try (try loader.joinAndPin(request)).retireAfterGPUCompletion()
        let seconds = Date().timeIntervalSince(begun)
        try splitRequire(
            seconds < 1 && loader.splitReads.splitCount == 1 && whole.readCount == 0,
            "Qwen: the split waited \(seconds) s for a queue slot: \(loader.splitReads.report)")
        try splitRequire(splitPromptly { scheduler.shutdown() }, "Qwen: shutdown did not return")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-queued-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    for shutdown in [false, true] {
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let saved = FamilyExperts.checkReadLimits
        FamilyExperts.checkReadLimits = (1, 1)
        let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
        FamilyExperts.checkReadLimits = saved
        experts.adaptiveGuesses = false
        experts.splitsReads = true
        experts.disk = { _ in nil }
        experts.attach(
            try HotCopy.write(source: experts, keys: Array(4..<12), folder: folder.appendingPathComponent("hot\(shutdown)"), identity: "h"))
        let leases = experts.replicaArena?.snapshot().leasedRecords ?? 0
        let reads = experts.reads
        let hold = reads.holdStarts()
        final class Done: @unchecked Sendable {
            var error: Error?
            var at: Date?
        }
        let done = Done()
        let cancelling = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            do {
                try splitQueuePartThenCancel(reads, shutdown: shutdown, hold: hold) { _ = try reads.cancelAndJoin(reads.currentGeneration) }
            } catch { done.error = error }
            done.at = Date()
            cancelling.signal()
        }
        var cancelled = false
        do { _ = try experts.fetch(layer: 1, experts: [2], decode: true) } catch { cancelled = true }
        let ended = Date()
        _ = cancelling.wait(timeout: .now() + 5)
        if let error = done.error { throw error }
        let seconds = done.at.map { ended.timeIntervalSince($0) } ?? .infinity
        try splitRequire(
            cancelled && seconds < 1 && experts.splitReads.splitCount == 0,
            "family (shutdown \(shutdown)): the split waited \(seconds) s on its queued part (cancelled \(cancelled))")
        try splitRequire(
            splitEventually { (experts.replicaArena?.snapshot().leasedRecords ?? 0) == leases }, "family: the staging lease was kept")
        try splitRequire(splitPromptly { reads.shutdown() }, "family: shutdown did not return")
    }
}

/// Qwen without the replica race (one drive with hot copies): the fastest part is read once, straight into the slot
/// (inline, before the wait that guards late parts begins, so a staged copy of it could never be guarded); only the
/// other part is read into staging and copied in. The record is exact.
func splitReadFirstPartInPlace() throws {
    let sources = SplitSources(PackStore.expertBytes, sources: 2)
    let whole = SplitSources(PackStore.expertBytes, sources: 2)
    let (loader, scheduler, _, _) = try splitQwenLoader(sources: sources, whole: whole, replicas: false)
    defer { scheduler.shutdown() }
    let loaded = try loader.joinAndPin(
        try loader.request(
            try ExpertKey(layer: 0, expert: 1), generation: scheduler.currentGeneration, policy: .earliestCompletion,
            calibration: splitCalibration))
    defer { try? loaded.retireAfterGPUCompletion() }
    let slot = UInt(bitPattern: loaded.buffer.bytes.baseAddress)
    let inSlot = sources.addresses.filter { $0 >= slot && $0 < slot + UInt(PackStore.expertBytes) }.count
    try splitRequire(
        Array(loaded.buffer.bytes) == sources.pattern && loader.splitReads.splitCount == 1 && sources.readCount == 2 && inSlot == 1,
        "\(sources.readCount) part reads, \(inSlot) into the slot, \(loader.splitReads.splitCount) splits")
}

/// A seeded generator (SplitMix64), so a failing random plan can be run again.
private struct SplitRandom: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// The planner with busy and stale sources: a busy source gets a smaller, later part instead of none; a deep equal
/// backlog gives a whole read from the source that ends it first; a stale source (even one stalled when last measured) is
/// measured again with a part that is not the first, and leads only when every candidate is stale; over 400 seeded random
/// source sets the parts cover the record once on aligned boundaries, the lead's at its end, only candidates take parts
/// and a whole read goes to the candidate ending first. One source per disk; a lane's and the slow-read tally's age.
func splitReadPlannerLive() throws {
    let record = 2_764_800
    func plan(_ sources: [SplitReadSource]) -> SplitReadPlan { SplitReads.plan(recordBytes: record, sources: sources, waited: true) }
    let idle = splitSource(2.5e9)
    for (sources, idleAt) in [([idle, splitSource(2.5e9, busy: 0.5e-3)], 0), ([splitSource(2.5e9, busy: 0.5e-3), idle], 1)] {
        guard case .split(let parts) = plan(sources), parts.count == 2, splitCovers(parts, record) else {
            throw EngineError("Split reads: a busy source took no part: \(plan(sources))")
        }
        let busyPart = parts.first { $0.source != idleAt }!
        try splitRequire(parts[0].source == idleAt && busyPart.length < record / 2, "the busy source's part: \(parts)")
        let ends = parts.map { sources[$0.source].seconds($0.length) }
        try splitRequire(abs(ends[0] - ends[1]) <= 2 * Double(SplitReads.alignment) / 2.5e9, "busy parts end apart: \(ends)")
    }
    try splitRequire(plan([splitSource(2.5e9, busy: 1), splitSource(2.5e9, busy: 1.001)]) == .whole(source: 0, .smallGain), "backlog")
    try splitRequire(plan([splitSource(2.5e9, busy: 1.001), splitSource(2.5e9, busy: 1)]) == .whole(source: 1, .smallGain), "backlog 2")
    // Stalled when last measured a minute ago: planned at its healthy rate, staged behind the fresh lead.
    for sources in [[idle, splitSource(0.3e9, healthy: 2.4e9, age: 60)], [splitSource(0.3e9, healthy: 3e9, age: .infinity), idle]] {
        guard case .split(let parts) = plan(sources), parts.count == 2, sources[parts[0].source].fresh else {
            throw EngineError("Split reads: a stale source was not measured again behind a fresh lead: \(plan(sources))")
        }
    }
    try splitRequire(
        plan([idle, splitSource(0.3e9, healthy: 2.4e9, age: 1)]) == .whole(source: 0, .unhealthy)
            && plan([idle, splitSource(2.5e9, age: 60, wearing: true)]) == .whole(source: 0, .wear), "a fresh stalled or wearing source")
    // Every candidate stale: one may lead. A fresh one deep in a backlog with stale idle ones: no stale lead, so whole.
    guard case .split(let stale) = plan([splitSource(2.5e9, age: 30), splitSource(2.4e9, age: .infinity)]), stale[0].source == 0 else {
        throw EngineError("Split reads: stale sources alone were not split")
    }
    try splitRequire(
        plan([splitSource(2.5e9, busy: 1), splitSource(2.5e9, age: 60), splitSource(2.4e9, age: 60)]) == .whole(source: 1, .smallGain),
        "a stale source led while a fresh one was a candidate")
    var random = SplitRandom(state: 1)
    for run in 0..<400 {
        let count = Int.random(in: 2...5, using: &random)
        let sources = (0..<count).map { _ -> SplitReadSource in
            let rate = Double.random(in: 0.05e9...5e9, using: &random)
            return splitSource(
                rate, latency: Double.random(in: 0...1e-3, using: &random),
                healthy: [rate, 2 * rate, 4 * rate, 0].randomElement(using: &random)!,
                busy: [0, 0, 1e-4, 1e-3, 0.1].randomElement(using: &random)!,
                age: [0, 5, 30, .infinity].randomElement(using: &random)!, wearing: Int.random(in: 0..<10, using: &random) == 0)
        }
        let bytes = Int.random(in: (2 * SplitReads.alignment)...20_000_000, using: &random)
        let candidates = sources.indices.filter { !sources[$0].stalled && !sources[$0].wearing }
        let first = candidates.map { sources[$0].seconds(bytes) }.min()
        switch SplitReads.plan(recordBytes: bytes, sources: sources, waited: true) {
        case .split(let parts):
            try splitRequire(
                parts.count >= 2 && splitCovers(parts, bytes) && parts[0].offset + parts[0].length == bytes
                    && parts.allSatisfy { candidates.contains($0.source) }
                    && (sources[parts[0].source].fresh || candidates.allSatisfy { !sources[$0].fresh })
                    && parts.map({ sources[$0.source].seconds($0.length) }).max()! <= (1 - SplitReads.margin) * first!,
                "run \(run): parts \(parts) of \(bytes) bytes over \(sources)")
        case .whole(let source?, _):
            try splitRequire(candidates.contains(source) && sources[source].seconds(bytes) == first, "run \(run): whole from \(source)")
        case .whole(nil, _):
            try splitRequire(candidates.isEmpty, "run \(run): no source for a whole read with candidates \(candidates)")
        }
    }
    try splitRequire(
        SplitReads.onePerDisk(["disk3", "disk3", nil, "disk4", nil], rates: [1, 2, 0, 1, 0]) == [1, 2, 3, 4]
            && SplitReads.onePerDisk(["disk3", "disk3"], rates: [0, 0]) == [1], "one source per disk")
    // A lane trained by parts and whole records alike: bytes over time, aged from its last read.
    var lane = ReadLane()
    for n in 0..<16 { lane.served(seconds: 1e-3 * Double(n % 2 + 1), bytes: 1_000_000 * (n % 2 + 1)) }
    let measured = lane.measured
    try splitRequire(
        measured.map { abs($0.bytesPerSecond / 1e9 - 1) < 0.01 && $0.secondsAgo < 1 } == true && ReadLane().measured == nil,
        "lane rate \(String(describing: measured))")
    final class Clock: @unchecked Sendable { var now = 100.0 }
    let clock = Clock()
    let reads = SlowDriveReads(clock: { clock.now })
    try splitRequire(reads.age == .infinity, "an unfed tally has an age")
    reads.add(bytes: 1e6, seconds: 1e-3)
    clock.now = 112
    try splitRequire(abs(reads.age - 12) < 1e-9, "tally age \(reads.age)")
}

/// Qwen: a read whose normal path is the pack, with the pack stalled, goes whole to the model's drive as one part (no
/// staging lease), exact, counted as `whole_from_other_source` and not as a split, and leaves no planned bytes behind.
func splitReadQwenRouted() throws {
    let sources = SplitSources(PackStore.expertBytes, sources: 2)
    let whole = SplitSources(PackStore.expertBytes, sources: 2)
    let (routed, routedScheduler, _, arena) = try splitQwenLoader(
        sources: sources, whole: whole, measured: { $0 == .externalSSD ? (0.5e9, 0, 2.4e9) : (2.5e9, 0, 2.5e9) })
    defer { routedScheduler.shutdown() }
    let leases = arena.snapshot().leasedRecords
    let loaded = try routed.joinAndPin(
        try routed.request(
            try ExpertKey(layer: 0, expert: 1), generation: routedScheduler.currentGeneration, calibration: splitCalibration))
    defer { try? loaded.retireAfterGPUCompletion() }
    let report = routed.splitReads.report
    try splitRequire(
        Array(loaded.buffer.bytes) == sources.pattern && sources.reads == [0] && whole.readCount == 0
            && report["whole_from_other_source"] as? Int == 1 && routed.splitReads.splitCount == 0
            && arena.snapshot().leasedRecords == leases,
        "routed whole read: \(sources.reads) part and \(whole.reads) whole reads, \(report)")
    try splitRequire(splitEventually { routed.plannedBytes.total == 0 }, "planned bytes left after a routed read")
}

/// GLM and DeepSeek: a lone miss whose copy place is wearing is read whole from the GGUF (`whole_from_other_source`),
/// exact, though `copyShare` gave it the copy; with neither wearing, several lone misses in a row split, exact.
func splitReadFamilyRouted() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-split-routed-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(3 * (212_992 + 114_688)))
    experts.adaptiveGuesses = false
    experts.splitsReads = true
    experts.disk = { _ in nil }
    let copy = try HotCopy.write(source: experts, keys: Array(4..<12), folder: folder.appendingPathComponent("hot"), identity: "h")
    experts.attach(copy)
    let per = GGUFQuantize.bytes(8, count: 256 * 256)
    func exact(_ slot: FamilyExperts.Slot, _ expert: Int) -> Bool {
        (0..<3).allSatisfy { part in
            UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per).allSatisfy {
                $0 == UInt8(50 + expert * 10 + part)
            }
        }
    }
    copy.reads.markWear(true)
    let hotReads = experts.counters.hotReads
    try splitRequire(exact(try experts.fetch(layer: 1, experts: [2], decode: true)[0], 2), "the routed record's bytes")
    let report = experts.splitReads.report
    try splitRequire(
        report["whole_from_other_source"] as? Int == 1 && experts.splitReads.splitCount == 0 && experts.counters.hotReads == hotReads,
        "a wearing copy was read: \(report), hot reads \(experts.counters.hotReads)")
    copy.reads.markWear(false)
    for expert in [0, 1, 3] {
        try splitRequire(exact(try experts.fetch(layer: 1, experts: [expert], decode: true)[0], expert), "expert \(expert)'s bytes")
    }
    try splitRequire(experts.splitReads.splitCount >= 1, "no lone miss split: \(experts.splitReads.report)")
    try requireFamilyMirror(experts, "split routed")
}
