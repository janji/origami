import Dispatch
import Foundation

/// One source holding a record (a drive with the model's files or a copy), as the split planner sees it now: its live
/// read rate and fixed cost per read (from that drive's own completed reads, whole and part) and how long ago a read
/// last fed that rate, its own healthy rate, how long until the work it already has is done, and whether its read tally
/// says it is being worn into its slow state (`DriveReadHistory.high` or `due`).
package struct SplitReadSource: Sendable, Equatable {
    package var bytesPerSecond: Double
    package var latencySeconds: Double
    package var healthyBytesPerSecond: Double
    /// Seconds until the reads already on this source are done (0: idle).
    package var busySeconds: Double
    /// Seconds since a completed read last fed `bytesPerSecond` (`.infinity`: never).
    package var measuredSecondsAgo: Double
    package var wearing: Bool
    package init(
        bytesPerSecond: Double, latencySeconds: Double, healthyBytesPerSecond: Double, busySeconds: Double = 0,
        measuredSecondsAgo: Double = 0, wearing: Bool = false
    ) {
        self.bytesPerSecond = bytesPerSecond
        self.latencySeconds = latencySeconds
        self.healthyBytesPerSecond = healthyBytesPerSecond
        self.busySeconds = busySeconds
        self.measuredSecondsAgo = measuredSecondsAgo
        self.wearing = wearing
    }
    /// Its rate is recent enough to trust (`SplitReads.freshSeconds`).
    package var fresh: Bool { measuredSecondsAgo <= SplitReads.freshSeconds }
    /// The rate a plan uses: the live one while fresh; a stale one is measured again, planned at the healthy rate when
    /// that is known (else at the last one).
    package var plannedBytesPerSecond: Double { max(1, fresh || healthyBytesPerSecond <= 0 ? bytesPerSecond : healthyBytesPerSecond) }
    /// When a read issued now would start delivering: after the work already there and the fixed cost per read.
    package var startSeconds: Double { max(0, busySeconds) + max(0, latencySeconds) }
    /// When a read of `bytes` issued now would end on this source.
    package func seconds(_ bytes: Int) -> Double { startSeconds + Double(bytes) / plannedBytesPerSecond }
    /// Below a third of its own healthy rate, measured recently: the hedge rule's and `SlowDriveReads`' stall (a stale
    /// rate says nothing about now).
    package var stalled: Bool { fresh && healthyBytesPerSecond > 0 && bytesPerSecond < healthyBytesPerSecond / 3 }
}

/// One part of a split read: bytes `offset ..< offset + length` of the record from source `source`.
package struct SplitReadPart: Sendable, Equatable {
    package let source: Int, offset: Int, length: Int
    package init(source: Int, offset: Int, length: Int) {
        self.source = source
        self.offset = offset
        self.length = length
    }
}

/// Why a demand read that more than one source holds was read whole.
package enum SplitReadSkip: String, CaseIterable, Sendable {
    case unhealthy, wear
    case smallGain = "small_gain"
}

/// The planner's answer: the read whole from `source`, the one whose read would end first (nil: not eligible, read as
/// before), with why it was not split when it was eligible; or its parts.
package enum SplitReadPlan: Equatable, Sendable {
    case whole(source: Int?, SplitReadSkip?)
    case split([SplitReadPart])
}

/// Split reads (measured 2026-11-17 with `Tools/bench/split_reads.py`): a demand read that a layer waits on and several
/// healthy sources hold is read in parts, one per source, sized by each one's live speed and the work it already has so
/// they end together; else whole from the source that would end it first. Every source holds the same bytes and the
/// assembled record is verified as a whole read's is, so output cannot change (BITWISE); only which drive reads which
/// bytes does. Shared by all three engines.
package enum SplitReads {
    /// A split must be predicted to end at least this much sooner than the best whole read: the measurement gained
    /// 25–40% on lone reads, with about 10% run-to-run noise, so 15% keeps only clear wins.
    package static let margin = 0.15
    /// Part boundaries fall on multiples of this many bytes from the record's start: the 16 KiB page of Apple silicon,
    /// at which every record, slot and staging record starts (so uncached reads land whole pages).
    package static let alignment = 16_384
    /// Split reads are off unless `ORIGAMI_SPLIT_READS=1` (an experiment: no gain measured yet; the default or deleted
    /// by 3.2). Each engine reads it once at start (`splitsReads`).
    package static let enabled = ProcessInfo.processInfo.environment["ORIGAMI_SPLIT_READS"] == "1"
    /// A source's rate older than this many seconds is measured again instead of trusted (`SplitReadSource.fresh`): a
    /// drive's state (thermal throttling, another app reading it) changes over tens of seconds, and measuring it again
    /// costs one guarded part.
    package static let freshSeconds = 10.0

    /// How to read a record of `recordBytes` that `sources` hold (the same bytes each). `waited`: a demand read a layer
    /// waits on (never read-ahead, a guess, warm-up or a keeper read); otherwise, with one source or a record under two
    /// pages, whole with no source (read as before). Of the candidates (not wearing, not stalled while its rate is
    /// fresh), each source can deliver rate x (T - busy - latency) bytes by time T: T is where the candidates together
    /// deliver the record (sources added by when they could start, while that is before T). A source whose part would
    /// be under one alignment unit takes none and its bytes go to the others; boundaries are on `alignment`. Split when
    /// the latest part's predicted end is at least `margin` sooner than the best whole read (the candidate whose read
    /// would end first); else whole from that candidate.
    /// The first part (read in place with no guard of its own) is the fastest fresh source's and lies at the record's
    /// end; a stale source takes it only when every candidate is stale (it is staged, so a wrong guess costs one guarded
    /// re-read). Together the parts cover the record once.
    package static func plan(recordBytes: Int, sources: [SplitReadSource], waited: Bool, enabled: Bool = true) -> SplitReadPlan {
        guard enabled, waited, sources.count >= 2, recordBytes >= 2 * alignment else { return .whole(source: nil, nil) }
        let ends = sources.map { $0.seconds(recordBytes) }
        func excluded(_ i: Int) -> SplitReadSkip? { sources[i].stalled ? .unhealthy : sources[i].wearing ? .wear : nil }
        let candidates = sources.indices.filter { excluded($0) == nil }
        guard let best = candidates.min(by: { ends[$0] < ends[$1] }) else {
            return .whole(source: nil, excluded(sources.indices.min { ends[$0] < ends[$1] }!))
        }
        guard candidates.count >= 2 else {
            let order = SplitReadSkip.allCases
            return .whole(source: best, sources.indices.compactMap(excluded).min { order.firstIndex(of: $0)! < order.firstIndex(of: $1)! })
        }
        // When the sources in `chosen` together deliver the record: the sum of rate x (T - start) is the record.
        func end(_ chosen: [Int]) -> Double {
            let rates = chosen.map { sources[$0].plannedBytesPerSecond }
            return (Double(recordBytes) + zip(rates, chosen).reduce(0) { $0 + $1.0 * sources[$1.1].startSeconds }) / rates.reduce(0, +)
        }
        var chosen: [Int] = []
        for i in candidates.sorted(by: { sources[$0].startSeconds < sources[$1].startSeconds }) {
            if !chosen.isEmpty && sources[i].startSeconds >= end(chosen) { break }
            chosen.append(i)
        }
        let fresh = chosen.filter { sources[$0].fresh }
        guard chosen.count >= 2,
            let lead = (fresh.isEmpty && candidates.allSatisfy { !sources[$0].fresh } ? chosen : fresh).min(by: { ends[$0] < ends[$1] })
        else { return .whole(source: best, .smallGain) }
        chosen = [lead] + chosen.filter { $0 != lead }.sorted()
        let units = (recordBytes + alignment - 1) / alignment
        var shares: [Int] = []
        while true {
            let at = end(chosen)
            // Whole alignment units per part (the record's partial last page counted whole); the lead takes what rounding
            // leaves.
            shares = chosen.map { Int((sources[$0].plannedBytesPerSecond * (at - sources[$0].startSeconds) / Double(alignment)).rounded()) }
            shares[0] = units - shares.dropFirst().reduce(0, +)
            guard let drop = shares.indices.dropFirst().last(where: { shares[$0] < 1 }) else { break }
            chosen.remove(at: drop)
            if chosen.count < 2 { return .whole(source: best, .smallGain) }
        }
        guard shares[0] >= 1 else { return .whole(source: best, .smallGain) }
        // The lead part takes the record's partial last page and lies at its end, so every boundary is on a page.
        var lengths = shares.map { $0 * alignment }
        lengths[0] -= units * alignment - recordBytes
        let predicted = chosen.indices.map { sources[chosen[$0]].seconds(lengths[$0]) }.max()!
        guard lengths[0] > 0, predicted <= (1 - margin) * ends[best] else { return .whole(source: best, .smallGain) }
        var offsets = Array(repeating: 0, count: chosen.count)
        var offset = 0
        for n in chosen.indices.dropFirst() + [0] {
            offsets[n] = offset
            offset += lengths[n]
        }
        return .split(chosen.indices.map { SplitReadPart(source: chosen[$0], offset: offsets[$0], length: lengths[$0]) })
    }

    /// Reads `parts` (covering the record once) in parallel into `destination` and returns once every range holds its
    /// bytes and no read writes into `destination` any more. The first part (the fastest source's) is read on this
    /// thread, straight into place, before the wait begins (so it has no guard of its own: only a record-level hedge
    /// covers it, as a whole read); the others are `submit`ted (each its own job on its source's queue, so it counts as
    /// that drive's work while it runs) and, with `staging` (at least record-sized, the same offsets), read there and
    /// copied in, so a late one can be raced: once a staged part outlives `lateAfter(part)` seconds, its range is read
    /// again straight into `destination` from the fastest other source of `order` (the sources holding the record,
    /// fastest first), and the first complete version wins (a staged part that ends first waits for that read to end
    /// before it is copied, so two writers never write the same bytes at once). A staged part is copied in only inside
    /// `owns`, which runs it only while this read still owns the record (no other read published it): once it says no,
    /// no staged byte is copied, the reads into place are waited for and `SplitReadOrphaned` is thrown. A part that fails
    /// is read again into place from the next source of `order`; a range every source failed fails the whole read. A
    /// staged read still running when this returns writes only into `staging`, which `keep` (held by its job) keeps
    /// until it ends. `cancelled` is asked while waiting: no read starts after it says yes, and the reads into place
    /// are waited for before it throws `IOOperationCancelled`; so is a part whose job was cancelled (its generation or
    /// its scheduler ended). `submit` must call its job exactly as the scheduler ends it: with nil from the job's
    /// operation (the read runs), and with an error when the job ends without running it (cancelled while queued, or
    /// before it started); the part ends once, whichever comes first, so the read never waits on a part that cannot end.
    package static func execute(
        _ parts: [SplitReadPart], into destination: UnsafeMutableRawBufferPointer, order: [Int],
        staging: UnsafeMutableRawBufferPointer? = nil, keep: AnyObject? = nil, lateAfter: (SplitReadPart) -> Double,
        cancelled: () -> Bool = { false },
        owns: (() -> Void) -> Bool = {
            $0()
            return true
        },
        submit: (_ source: Int, _ bytes: Int, _ job: @escaping @Sendable (_ skipped: Error?) -> Void) throws -> Void = { _, _, job in
            queue.async { job(nil) }
        },
        read: @escaping @Sendable (_ source: Int, _ offset: Int, _ into: UnsafeMutableRawBufferPointer) throws -> Void
    ) throws -> SplitReadOutcome {
        precondition(!parts.isEmpty && parts.reduce(0) { $0 + $1.length } == destination.count, "Split parts must cover the record")
        precondition(staging.map { $0.count >= destination.count } ?? true, "Staging smaller than the record")
        let state = SplitReadState(parts.map(\.length))
        let started = DispatchTime.now().uptimeNanoseconds
        let stagedParts = staging != nil
        nonisolated(unsafe) let target = destination
        nonisolated(unsafe) let stage = staging
        let readers = SplitStagingReaders(keep)
        // One read of part `k`'s range from `source`: into staging (`staged`) or into place; on this thread (`inline`)
        // or as a job of its own. A job that cannot be submitted is a failed read.
        func launch(_ k: Int, source: Int, staged: Bool, kind: SplitReadState.Kind, inline: Bool = false) {
            let part = parts[k]
            nonisolated(unsafe) let range = UnsafeMutableRawBufferPointer(
                rebasing: (staged ? stage! : target)[part.offset..<(part.offset + part.length)])
            state.begin(k, kind: kind, source: source)
            if staged { readers.begin() }
            let once = SplitPartOnce()
            // Ends the part once: after its read (`skipped` nil), or at once with `skipped` (its job never ran it).
            let job: @Sendable (Error?) -> Void = { skipped in
                guard once.claim() else { return }
                let begun = DispatchTime.now().uptimeNanoseconds
                var error = skipped
                if skipped == nil {
                    do { try read(source, part.offset, range) } catch let failure { error = failure }
                }
                if staged { readers.end() }
                state.end(k, kind: kind, error: error, seconds: Double(DispatchTime.now().uptimeNanoseconds - begun) / 1e9)
            }
            if inline { return job(nil) }
            do { try submit(source, part.length, { skipped in withExtendedLifetime(readers) { job(skipped) } }) } catch { job(error) }
        }
        // A staged part's bytes into place (only once no other read writes its range).
        func copy(_ k: Int) {
            let part = parts[k]
            (target.baseAddress! + part.offset).copyMemory(from: stage!.baseAddress! + part.offset, byteCount: part.length)
        }
        for k in parts.indices.dropFirst() { launch(k, source: parts[k].source, staged: stagedParts, kind: .part) }
        launch(0, source: parts[0].source, staged: false, kind: .part, inline: true)
        // A delay that is not finite, or a billion seconds or more, never makes a part late.
        let late = parts.map { part -> UInt64 in
            let seconds = lateAfter(part)
            guard seconds.isFinite, seconds < 1e9 else { return .max }
            let (at, overflow) = started.addingReportingOverflow(UInt64(max(0, seconds) * 1e9))
            return overflow ? .max : at
        }
        var failure: Error?
        state.lock.lock()
        while true {
            let stop = cancelled()
            var next = UInt64.max
            for k in parts.indices where failure == nil && !stop && !state.ranges[k].resolved {
                let r = state.ranges[k]
                let staged = k > 0 && stagedParts
                if r.other == .succeeded || (r.part == .succeeded && r.other != .running) {
                    // A staged part that ended first is copied in, while the record is still this read's; a guard or
                    // fallback that ended first is in place.
                    if staged && r.part == .succeeded && r.other != .succeeded && !owns({ copy(k) }) {
                        failure = SplitReadOrphaned()
                        break
                    }
                    state.ranges[k].resolved = true
                    continue
                }
                if r.other == .running { continue }
                let failed = r.part == .failed
                // A cancelled job: its generation or scheduler ended, so no other read of the range would start.
                if failed && r.error is IOOperationCancelled {
                    failure = IOOperationCancelled()
                    break
                }
                // A late part is guarded, and a guard that failed while the part still runs is followed by the next.
                let due =
                    staged && r.part == .running
                    && (r.other == .failed || (r.other == .none && DispatchTime.now().uptimeNanoseconds >= late[k]))
                if failed || due {
                    // Into place: the part's own read has ended (failed) or writes only into staging.
                    if let source = order.first(where: { !r.tried.contains($0) }) {
                        state.lock.unlock()
                        launch(k, source: source, staged: false, kind: due ? .guarded : .fallback)
                        state.lock.lock()
                    } else if failed {
                        failure = r.error ?? EngineError("Every source of a split read part failed")
                    }
                } else if staged && r.part == .running && r.other == .none {
                    next = min(next, late[k])
                }
            }
            // Done once every range is resolved; after a failure or cancellation, once nothing writes into place.
            if failure != nil || stop {
                if !state.writingInPlace(staged: stagedParts) {
                    if failure == nil && !state.ranges.allSatisfy(\.resolved) { failure = IOOperationCancelled() }
                    break
                }
            } else if state.ranges.allSatisfy(\.resolved) {
                break
            }
            let now = DispatchTime.now().uptimeNanoseconds
            let wakeAt = min(next, now + 2_000_000)
            _ = state.lock.wait(until: Date(timeIntervalSinceNow: Double(wakeAt - min(wakeAt, now)) / 1e9))
        }
        let outcome = SplitReadOutcome(
            guards: state.guards, fallbacks: state.fallbacks, bytesBySource: state.bytesBySource, secondsBySource: state.secondsBySource,
            parts: state.completed)
        state.lock.unlock()
        if let failure { throw failure }
        return outcome
    }

    private static let queue = DispatchQueue(label: "origami.split-reads", qos: .userInitiated, attributes: .concurrent)

    /// The reads that put bytes `range` of a record laid out as `spans` back to back (each an offset in its file and a
    /// length) at the start of an output of `range.count` bytes: (span, offset in its file, length, offset in the output).
    package static func pieces(of range: Range<Int>, spans: [(offset: Int64, length: Int)]) -> [(
        span: Int, offset: Int64, length: Int, at: Int
    )] {
        var pieces: [(span: Int, offset: Int64, length: Int, at: Int)] = []
        var start = 0
        for (index, span) in spans.enumerated() {
            let lower = max(range.lowerBound, start)
            let upper = min(range.upperBound, start + span.length)
            if lower < upper { pieces.append((index, span.offset + Int64(lower - start), upper - lower, lower - range.lowerBound)) }
            start += span.length
        }
        return pieces
    }

    /// Of sources on `disks` (each one's physical disk, `DriveSpeed.disk`; nil: unknown, always kept), the ones a split
    /// may use: one per disk, the fastest by `rates` (the later one on a tie). Two sources on one disk share its
    /// bandwidth, so each one's own rate would claim the whole disk.
    package static func onePerDisk(_ disks: [String?], rates: [Double]) -> [Int] {
        disks.indices.filter { i in
            !disks.indices.contains { j in
                j != i && disks[i] != nil && disks[j] == disks[i] && (rates[j] > rates[i] || (rates[j] == rates[i] && j > i))
            }
        }
    }
}

/// What a split read did: guard reads of late parts, reads again after a failed part, per source the bytes it
/// delivered and the seconds its successful reads took, and each successful read (source, bytes, seconds) so the
/// engine trains each source's live estimate with it as with a whole read.
package struct SplitReadOutcome: Sendable {
    package let guards, fallbacks: Int
    package let bytesBySource: [Int: Int]
    package let secondsBySource: [Int: Double]
    package let parts: [(source: Int, bytes: Int, seconds: Double)]
}

/// Thrown by `SplitReads.execute` when another read published the record first: no staged byte was copied in, and the
/// caller must not read the record again into its slot (it is no longer the record's writer).
package struct SplitReadOrphaned: Error, Sendable { package init() {} }

/// The staged readers of one split read: they hold it (and with it the caller's `keep`, the staging lease) until each
/// has returned, so the lease goes back to its arena only after its last reader.
private final class SplitStagingReaders: @unchecked Sendable {
    private let lock = NSLock()
    private var running = 0
    private let keep: AnyObject?
    init(_ keep: AnyObject?) { self.keep = keep }
    func begin() { lock.withLock { running += 1 } }
    func end() { lock.withLock { running -= 1 } }
    deinit { assert(running == 0, "A split read's staging was released while a reader still wrote into it") }
}

/// Whether a split part has ended: its job ends it once, from its operation or from its completion.
private final class SplitPartOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var ended = false
    func claim() -> Bool {
        lock.withLock {
            defer { ended = true }
            return !ended
        }
    }
}

/// `SplitReads.execute`'s per-range state, under `lock` (an `NSCondition` the reads signal).
private final class SplitReadState: @unchecked Sendable {
    enum Kind { case part, guarded, fallback }
    enum Read { case none, running, succeeded, failed }
    struct Range {
        /// The part's own read, and the latest guard or fallback read of its range (into place).
        var part = Read.running, other = Read.none
        /// Sources that read (or are reading) this range, and the source of the part's own read and of the other.
        var tried: Set<Int> = []
        var partSource = 0, otherSource = 0
        var error: Error?
        var resolved = false
    }
    let lock = NSCondition()
    var ranges: [Range]
    var guards = 0, fallbacks = 0
    var bytesBySource: [Int: Int] = [:], secondsBySource: [Int: Double] = [:]
    var completed: [(source: Int, bytes: Int, seconds: Double)] = []
    private let lengths: [Int]
    init(_ lengths: [Int]) {
        self.lengths = lengths
        ranges = Array(repeating: Range(), count: lengths.count)
    }

    func begin(_ k: Int, kind: Kind, source: Int) {
        lock.lock()
        ranges[k].tried.insert(source)
        switch kind {
        case .part: ranges[k].partSource = source
        case .guarded, .fallback:
            ranges[k].other = .running
            ranges[k].otherSource = source
            if kind == .guarded { guards += 1 } else { fallbacks += 1 }
        }
        lock.unlock()
    }

    func end(_ k: Int, kind: Kind, error: Error?, seconds: Double) {
        lock.lock()
        let result: Read = error == nil ? .succeeded : .failed
        if kind == .part { ranges[k].part = result } else { ranges[k].other = result }
        if let error { ranges[k].error = error }
        if error == nil {
            let source = kind == .part ? ranges[k].partSource : ranges[k].otherSource
            secondsBySource[source, default: 0] += seconds
            bytesBySource[source, default: 0] += lengths[k]
            completed.append((source, lengths[k], seconds))
        }
        lock.broadcast()
        lock.unlock()
    }
    /// Whether a read into place still runs: the first part's (every part's when unstaged), or a guard or fallback.
    func writingInPlace(staged: Bool) -> Bool {
        ranges.indices.contains { k in ranges[k].other == .running || ((k == 0 || !staged) && ranges[k].part == .running) }
    }
}

/// Split-read counters (any thread), in `/v1/status` and each request's notes as `split_reads`: split reads, the bytes
/// each source delivered to them, guard re-reads of late parts and re-reads of failed parts, the demand reads that more
/// than one source held but were read whole and why, the whole reads sent to another source than the default one, per
/// source the rate and its age the last plan used, and the seconds decode waited on lone reads (split or not, so the
/// benefit can be judged).
package final class SplitReadCounters: @unchecked Sendable {
    private let lock = NSLock()
    private var splits = 0, guards = 0, fallbacks = 0, failures = 0, routed = 0
    private var partBytes: [String: Int] = [:]
    private var skipped: [String: Int] = [:]
    private var planned: [String: (rate: Double, age: Double, at: UInt64)] = [:]
    private var loneWaits = 0, loneWaitSeconds = 0.0
    package init() {}

    /// A plan, made from `sources` (named `names`, each source's status name).
    package func record(_ plan: SplitReadPlan, sources: [SplitReadSource] = [], names: [String] = []) {
        let now = DispatchTime.now().uptimeNanoseconds
        lock.withLock {
            for (name, source) in zip(names, sources) { planned[name] = (source.plannedBytesPerSecond, source.measuredSecondsAgo, now) }
            if case .whole(_, let reason?) = plan { skipped[reason.rawValue, default: 0] += 1 }
        }
    }
    /// A whole read went to the source the plan found best instead of the default one.
    package func routedWhole() { lock.withLock { routed += 1 } }
    /// A split read ended (`outcome` nil: it failed and was read again whole); `names`: each source's status name.
    package func record(_ outcome: SplitReadOutcome?, names: [String]) {
        lock.withLock {
            guard let outcome else {
                failures += 1
                return
            }
            splits += 1
            guards += outcome.guards
            fallbacks += outcome.fallbacks
            for (source, bytes) in outcome.bytesBySource {
                partBytes[names.indices.contains(source) ? names[source] : "\(source)", default: 0] += bytes
            }
        }
    }
    /// Decode waited `seconds` on a lone demand read that more than one source held.
    package func waited(_ seconds: Double) {
        lock.withLock {
            loneWaits += 1
            loneWaitSeconds += max(0, seconds)
        }
    }
    package var report: [String: Any] {
        lock.withLock {
            [
                "enabled": SplitReads.enabled, "split_reads": splits, "part_bytes": partBytes, "guard_rereads": guards,
                "failed_part_rereads": fallbacks, "failed_splits": failures,
                "read_whole": Dictionary(uniqueKeysWithValues: SplitReadSkip.allCases.map { ($0.rawValue, skipped[$0.rawValue] ?? 0) }),
                "lone_demand_waits": loneWaits, "lone_demand_wait_seconds": loneWaitSeconds, "whole_from_other_source": routed,
                "sources": planned.mapValues { source -> [String: Any] in
                    let age = source.age + Double(DispatchTime.now().uptimeNanoseconds - source.at) / 1e9
                    return ["bytes_per_second": source.rate, "measured_seconds_ago": age.isFinite ? age : NSNull()]
                },
            ]
        }
    }
    package var splitCount: Int { lock.withLock { splits } }
    package var guardCount: Int { lock.withLock { guards } }
    package var loneWaitTotal: Double { lock.withLock { loneWaitSeconds } }
}

/// The bytes planned on each source and not yet read (any thread), by a key per physical source: with the source's
/// rate, how long until the work it already has is done (`SplitReadSource.busySeconds`).
package final class SplitPlannedBytes: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes: [AnyHashable: Int] = [:]
    package init() {}
    package subscript(_ source: AnyHashable) -> Int { lock.withLock { bytes[source] ?? 0 } }
    /// Every source's bytes together: 0 once every planned read has ended (checks).
    package var total: Int { lock.withLock { bytes.values.reduce(0, +) } }
    fileprivate func add(_ source: AnyHashable, _ count: Int) { lock.withLock { bytes[source, default: 0] += count } }
    fileprivate func remove(_ source: AnyHashable, _ count: Int) {
        lock.withLock {
            let left = bytes[source, default: 0] - count
            assert(left >= 0, "More bytes ended on a split source than were planned on it")
            bytes[source] = left > 0 ? left : nil
        }
    }
}

/// The bytes one read planned on its sources (`SplitPlannedBytes`, added at once), each item given back once: when its
/// part ends (`end`), when the read ends (`endAll`), at the latest when this is released.
package final class SplitPlannedRead: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [(source: AnyHashable, bytes: Int)?]
    private let ledger: SplitPlannedBytes
    package init(_ items: [(AnyHashable, Int)], ledger: SplitPlannedBytes) {
        self.items = items.map { (source: $0.0, bytes: $0.1) }
        self.ledger = ledger
        for (source, bytes) in items { ledger.add(source, bytes) }
    }
    deinit { endAll() }
    package func end(_ index: Int) {
        let item = lock.withLock { () -> (source: AnyHashable, bytes: Int)? in
            defer { items[index] = nil }
            return items[index]
        }
        if let item { ledger.remove(item.source, item.bytes) }
    }
    package func endAll() { for index in lock.withLock({ items.indices }) { end(index) } }
}
