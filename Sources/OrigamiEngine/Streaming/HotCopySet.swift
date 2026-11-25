import Foundation

/// One source of a record among several holding the same bytes (a copy on one drive): its reads in flight, its reads
/// done, its smoothed time per read, when it was last chosen and when it last served a read. Lock-owned by whoever keeps
/// the lanes.
package struct ReadLane: Sendable {
    package var inFlight = 0
    package var reads: UInt64 = 0
    package var serviceSeconds = 0.0
    /// Bytes per read, smoothed as `serviceSeconds` is: with it, the drive's rate per read (`HotCopySet.laneRates`).
    package var serviceBytes = 0.0
    package var lastSelected: UInt64 = 0
    /// When it last served a read (uptime seconds).
    package var lastServed = 0.0
    package init() {}

    /// Its drive's rate per read (smoothed bytes over smoothed time, so parts and whole records train it alike) and the
    /// seconds since a read last fed it, once it has 8 reads; nil before.
    package var measured: (bytesPerSecond: Double, secondsAgo: Double)? {
        guard reads >= 8, serviceSeconds > 0 else { return nil }
        return (serviceBytes / serviceSeconds, max(0, Double(DispatchTime.now().uptimeNanoseconds) / 1e9 - lastServed))
    }

    /// A read of `bytes` done in `seconds`: the smoothed time moves an eighth of the way to it (each sample clamped to
    /// a quarter to four times the estimate, so one stall or one cached read does not swing it).
    package mutating func served(seconds: Double, bytes: Int) {
        let prior = serviceSeconds > 0 ? serviceSeconds : Double(bytes) / 2e9
        let sample = min(1, max(1e-6, min(prior * 4, max(prior / 4, seconds))))
        serviceSeconds = reads == 0 ? sample : prior + (sample - prior) / 8
        serviceBytes = reads == 0 ? Double(bytes) : serviceBytes + (Double(bytes) - serviceBytes) / 8
        reads &+= 1
        lastServed = Double(DispatchTime.now().uptimeNanoseconds) / 1e9
    }
}

/// A model's copies on several drives (one `HotCopy` store per drive, kept by its own `HotCopyKeeper`) read as one
/// source: a record comes from the store holding it that is expected to finish first (`pick`), else from the next one
/// holding it; when none reads it the caller reads the model's own files. Every store holds the model's bytes, each
/// record checked against its CRC32 on its first read (`HotCopy.read`), so the bytes, and the replies, are the same
/// whichever store serves (BITWISE). A store whose drive fails a read pauses itself (`HotCopy.retryDelay`) and a
/// record failing its checksum is dropped from that store alone.
package final class HotCopySet: @unchecked Sendable {
    package let stores: [HotCopy]
    private let lock = NSLock()
    private var lanes: [ReadLane]
    private var placements: UInt64 = 0
    /// Each store's reads in `activity`, store i as its lane `first + i` (`DriveActivity`).
    private let activity: (counter: DriveActivity, first: Int)?

    package init(_ stores: [HotCopy], activity: (counter: DriveActivity, first: Int)? = nil) {
        self.stores = stores
        self.activity = activity
        lanes = Array(repeating: ReadLane(), count: stores.count)
    }

    /// Whether a read of `key` would come from one of the stores now.
    package func holds(_ key: Int) -> Bool { stores.contains { $0.holds(key) } }
    /// The records some store holds and has not rejected, each with its offset in the first such store: one store's
    /// own `offsets` less its rejected records.
    package var heldOffsets: [Int: Int64] {
        var held: [Int: Int64] = [:]
        for store in stores.reversed() {
            let rejected = Set(store.rejectedKeys)
            for (key, offset) in store.offsets where !rejected.contains(key) { held[key] = offset }
        }
        return held
    }
    /// Records read from each store, in `stores` order.
    package var reads: [UInt64] { lock.withLock { lanes.map(\.reads) } }

    /// The lane to read from among `lanes` (each holding the record), as of placement number `placements`: an idle lane
    /// with fewer than 8 reads, or not chosen in the last 1,024 placements, is probed first (the one chosen longest
    /// ago), so every drive gets measured from ordinary reads; otherwise the one whose reads in flight plus this one
    /// finish first at its smoothed time per read (`IOScheduler.earliest`; `bytes` / 2 GB/s until measured), the
    /// earlier lane on a tie. Nil for no lanes.
    package static func pick(_ lanes: [ReadLane], placements: UInt64, bytes: Int) -> Int? {
        let probe = lanes.indices.filter {
            lanes[$0].inFlight == 0 && (lanes[$0].reads < 8 || placements &- lanes[$0].lastSelected >= 1024)
        }
        if let first = probe.min(by: { lanes[$0].lastSelected < lanes[$1].lastSelected }) { return first }
        let prior = Double(bytes) / 2e9
        return IOScheduler.earliest(lanes.map { Double($0.inFlight + 1) * ($0.reads >= 8 ? $0.serviceSeconds : prior) })
    }

    /// The stores (indices in `stores`) that would read `key` now: every drive holding the record, not paused and not
    /// having rejected it. The one list of a record's sources the read choice (`read`) works from, for all three engines.
    package func sources(_ key: Int) -> [Int] { stores.indices.filter { stores[$0].holds(key) } }
    /// Each store's folder and its drive's smoothed rate per read (bytes over one read's time), for stores with at least
    /// 8 reads: what serving sees of each copy place's drive (`DriveSpeed.perReadRatio`).
    package var laneRates: [(folder: URL, bytesPerSecond: Double)] {
        lock.withLock {
            stores.indices.filter { lanes[$0].reads >= 8 && lanes[$0].serviceSeconds > 0 }.map {
                (stores[$0].folder, lanes[$0].serviceBytes / lanes[$0].serviceSeconds)
            }
        }
    }
    /// Store `index`'s lane now (reads in flight, reads done, smoothed time per read): a split's view of that drive.
    package func lane(_ index: Int) -> ReadLane { lock.withLock { lanes[index] } }

    private func claim(_ key: Int, bytes: Int, excluding tried: Set<Int>) -> Int? {
        lock.withLock {
            let eligible = sources(key).filter { !tried.contains($0) }
            guard !eligible.isEmpty else { return nil }
            placements &+= 1
            guard let chosen = Self.pick(eligible.map { lanes[$0] }, placements: placements, bytes: bytes) else { return nil }
            let index = eligible[chosen]
            lanes[index].lastSelected = placements
            lanes[index].inFlight += 1
            activity?.counter.began(activity!.first + index)
            return index
        }
    }

    /// Record `key` into `into` from the store expected to finish first, then from each other store holding it; the
    /// index of the store that read it. Throws when none did (none holds it, or each failed): the caller reads the
    /// model's own files.
    @discardableResult
    package func read(_ key: Int, into: UnsafeMutableRawBufferPointer) throws -> Int {
        try read(key, at: nil, into: into).store
    }

    /// Bytes `at ..< at + into.count` of record `key` (a split read's part), as `read` picks a store, unchecked: the
    /// store that read them and the checksum `verify` checks the assembled record against (nil: none due). A part's
    /// time trains its store's lane as a whole read's does, with its own bytes.
    package func read(_ key: Int, at: Int?, store only: Int? = nil, into: UnsafeMutableRawBufferPointer) throws -> (
        store: Int, sum: UInt32?
    ) {
        // `only`: a split part planned on that store; it fails (and fails over) once the store no longer holds the key.
        var tried = Set(only.map { only in stores.indices.filter { $0 != only } } ?? [])
        var failure: Error?
        while let index = claim(key, bytes: into.count, excluding: tried) {
            tried.insert(index)
            let started = DispatchTime.now().uptimeNanoseconds
            do {
                var sum: UInt32?
                if let at { sum = try stores[index].read(key, at: at, into: into) } else { try stores[index].read(key, into: into) }
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
                lock.withLock {
                    lanes[index].inFlight -= 1
                    lanes[index].served(seconds: seconds, bytes: into.count)
                    activity?.counter.ended(activity!.first + index, bytes: into.count)
                }
                return (index, sum)
            } catch {
                lock.withLock {
                    lanes[index].inFlight -= 1
                    activity?.counter.ended(activity!.first + index, bytes: 0)
                }
                failure = error
            }
        }
        throw failure ?? EngineError("No copy holds record \(key)")
    }

    /// Checks the assembled record `key` against each store's checksum that `read(_:at:into:)` returned (`HotCopy.verify`).
    package func verify(_ key: Int, sums: [Int: UInt32], record: UnsafeRawBufferPointer) throws {
        for (index, sum) in sums.sorted(by: { $0.key < $1.key }) { try stores[index].verify(key, sum: sum, record: record) }
    }
}
