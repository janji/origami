import Foundation

/// A model's hot copies on every drive the owner chose (F10): one `HotCopyKeeper` per place, the ranking split over the
/// places fastest drive first. The fastest place holds the first ranks, the next the following ones, and so on: each
/// place's keeper ranks only the records no faster place holds now (its attached copy), so a record leaves a slower
/// place only once a faster one has written and attached it (no gap in coverage). Only one keeper writes at a time, so
/// a build never loads two drives at once. Every attached copy is read through one `HotCopySet` (`attach` gets them
/// fastest first), which picks per record the store expected to finish first; a place whose drive is gone, full or
/// not permitted is detached by its own keeper and attached again when it comes back. Every place holds the model's
/// own bytes, so the replies are the same whichever serves.
package final class HotCopyPlaces: @unchecked Sendable {
    /// In the order the places were given (the first is `hot_copy_dir`'s).
    package let keepers: [HotCopyKeeper]
    /// `lock` guards the order and the attached copies; `attachLock` serializes the engine's swaps (taken without `lock`,
    /// so a status read during a swap never waits on it); `writerLock` guards `writer`.
    private let lock = NSLock(), attachLock = NSLock(), writerLock = NSLock()
    /// Place indices, fastest drive first.
    private var order: [Int]
    private var attached: [HotCopy?]
    private var writer: Int?, stopping = false

    /// `places`: each one's `HotCopyKeeper.Place` and the copy attached at start (already given to the engine);
    /// `speeds`: each place's measured drive rate (nil: unmeasured, after the measured ones in the given order).
    /// `attach` gives the engine every attached copy, fastest first, false while a request holds it. The rest is as
    /// for one `HotCopyKeeper`; `choose` (with `.max`) ranks every record.
    package init(
        source: HotCopySource, identity: String, places: [(place: (_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place, copy: HotCopy?)],
        speeds: [Double?], counts: (uses: [Int: Int], misses: [Int: Int]),
        choose: ((_ counts: (uses: [Int: Int], misses: [Int: Int]), _ budget: UInt64) -> [Int])? = nil,
        attach: @escaping ([HotCopy]) -> Bool, idle: @escaping () -> Bool,
        withIdle: ((_ work: () throws -> Void) throws -> Bool)? = nil, retire: ((_ folder: URL) -> Void)? = nil,
        event: @escaping (String) -> Void
    ) {
        let rank = choose ?? { counts, budget in HotCopy.choose(counts.uses, misses: counts.misses, source: source, budget: budget) }
        let withIdle =
            withIdle ?? { work in
                guard idle() else { return false }
                try work()
                return true
            }
        order = Self.order(speeds)
        attached = places.map(\.copy)
        var made: [HotCopyKeeper] = []
        weak var weakSelf: HotCopyPlaces?
        for (index, place) in places.enumerated() {
            made.append(
                HotCopyKeeper(
                    source: source, identity: identity, copy: place.copy, counts: counts,
                    choose: { counts, budget in
                        let held = weakSelf?.heldFaster(than: index) ?? []
                        return Self.fit(rank(counts, .max), excluding: held, budget: budget) { HotCopy.size($0, source: source) ?? 0 }
                    },
                    place: place.place,
                    attach: { copy in weakSelf?.attach(copy, at: index, with: attach) ?? false },
                    idle: { idle() && weakSelf?.claim(index) == true },
                    // Waits while another place writes: a swap batch ends at the first false (it means a request), and
                    // nothing would start it again until the next reply.
                    withIdle: { work in weakSelf?.await(index) == true ? try withIdle(work) : false },
                    retire: retire,
                    event: places.count > 1
                        ? { line in
                            event("\(index == 0 ? "" : "Place \(index + 1): ")\(line)")
                            // A copy that changed (built, grown, swapped in place) re-splits the slower places.
                            weakSelf?.slower(than: index).forEach { weakSelf?.keepers[$0].trigger() }
                        } : event))
        }
        keepers = made
        weakSelf = self
        // A swap in a slower place replaces what a faster place also holds first.
        for (index, keeper) in made.enumerated() { keeper.redundant = { weakSelf?.heldFaster(than: index) ?? [] } }
    }

    /// Place indices fastest first: measured ones by rate, then unmeasured ones, each group in the given order.
    package static func order(_ speeds: [Double?]) -> [Int] {
        speeds.indices.sorted { a, b in
            let (x, y) = (speeds[a] ?? -1, speeds[b] ?? -1)
            return x != y ? x > y : a < b
        }
    }
    /// The records of `ranked` (best first) a place of `budget` holds when `excluding` is held by faster places: first
    /// the next ranks in order (coverage), each taking its `size`, skipping one that does not fit; then, once every
    /// ranked record outside `excluding` is in, the room left holds the top ranks again (redundancy: a popular record
    /// on several drives, read from whichever delivers soonest). Duplicates come last, so a smaller cap drops them first.
    package static func fit(_ ranked: [Int], excluding: Set<Int>, budget: UInt64, size: (Int) -> UInt64) -> [Int] {
        var chosen: [Int] = []
        var total: UInt64 = 0
        var covered = true
        for key in ranked where !excluding.contains(key) {
            let bytes = size(key)
            guard bytes > 0 else { continue }
            guard total + bytes <= budget else {
                covered = false
                continue
            }
            chosen.append(key)
            total += bytes
        }
        guard covered else { return chosen }
        for key in ranked where excluding.contains(key) {
            let bytes = size(key)
            guard bytes > 0, total + bytes <= budget else { continue }
            chosen.append(key)
            total += bytes
        }
        return chosen
    }

    /// Re-orders the places after a drive was measured again; the next refresh moves records to follow.
    package func reorder(_ speeds: [Double?]) { lock.withLock { order = Self.order(speeds) } }
    /// Place indices, fastest first.
    package var ranking: [Int] { lock.withLock { order } }
    /// Records place `index` holds that a faster place holds too (duplicates for redundancy, `fit`).
    package func duplicates(_ index: Int) -> Int {
        let faster = heldFaster(than: index)
        return keepers[index].current?.keys.filter(faster.contains).count ?? 0
    }
    /// The copy attached at each place (nil: none), in the given order.
    package var copies: [HotCopy?] { lock.withLock { attached } }

    /// Every keeper plans again (after a reply, or a change from the dashboard): `places` replaces the places' rules
    /// (same count, else ignored).
    package func trigger(
        counts: (uses: [Int: Int], misses: [Int: Int])? = nil, places: [(_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place]? = nil,
        force: Bool = false
    ) {
        let places = places?.count == keepers.count ? places : nil
        for (index, keeper) in keepers.enumerated() { keeper.trigger(counts: counts, place: places?[index], force: force) }
    }
    package func stop(waiting seconds: Double = 0) {
        writerLock.withLock { stopping = true }
        for keeper in keepers { keeper.stop() }
        for keeper in keepers { keeper.stop(waiting: seconds) }
    }

    /// The places slower than `index`, fastest first.
    private func slower(than index: Int) -> [Int] { Array(ranking.drop { $0 != index }.dropFirst()) }
    /// The records attached at places faster than `index`.
    private func heldFaster(than index: Int) -> Set<Int> {
        lock.withLock {
            Set(order.prefix { $0 != index }.flatMap { attached[$0]?.keys ?? [] })
        }
    }
    /// The engine reads `copy` at place `index` from now on, with every other place's: false while a request runs. The
    /// slower places then plan again, so the records this one now holds move off them (the re-split).
    private func attach(_ copy: HotCopy?, at index: Int, with engine: ([HotCopy]) -> Bool) -> Bool {
        let order: [Int]? = attachLock.withLock {
            let (order, next) = lock.withLock { () -> ([Int], [HotCopy?]) in
                var next = attached
                next[index] = copy
                return (self.order, next)
            }
            guard engine(order.compactMap { next[$0] }) else { return nil }
            lock.withLock { attached[index] = copy }
            return order
        }
        guard order != nil else { return false }
        for slower in slower(than: index) { keepers[slower].trigger() }
        return true
    }
    /// Keeper `index` may write once no other keeper is building (false once stopping).
    private func await(_ index: Int) -> Bool {
        while !claim(index) {
            if writerLock.withLock({ stopping }) { return false }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return true
    }
    /// Whether keeper `index` may write now: no other keeper is building.
    private func claim(_ index: Int) -> Bool {
        writerLock.withLock {
            if let writer, writer != index, keepers[writer].building { return false }
            writer = index
            return true
        }
    }
}
