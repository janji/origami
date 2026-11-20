import Foundation

/// GLM's and DeepSeek's victim choice over the slot core (`FamilyExperts`' policy, and its observer). Slots are numbered
/// across pools: a pool's `first` slot plus the slot's index in its size class. Per slot: its heat, when it was last
/// brought up to date (`stamp`, in accesses of all pools), its `level` (log2 heat + stamp / halfLife), whether a prompt
/// placed its record and decode has not used it since (`scan`), and the storage cost of its record (`cost`). A victim is
/// the first empty slot, else the coolest record by heat decayed to `clock`; a prompt first takes the records its sweep
/// has passed (`Passed`). Never a slot stamped after the request's `start` (taken by this call, or by the call before
/// with `keepingLast`), nor one reserved, being read or pinned. Every method runs under the core lock.
package final class FamilyEviction: ExpertSlotPolicy, ExpertSlotObserver {
    package typealias Observer = FamilyEviction
    package typealias Core = ExpertSlotCore<FamilyEviction>

    /// The victims a prompt's reads take first, collected once per call from the slots `victim` may take: empty ones in
    /// slot order, then records a prompt placed (`scan`), the nearest before `bound` first, else the furthest ahead. A
    /// prompt sweeps the keys in order, so the coolest record is the oldest and the next sweep's first need, and the one
    /// just passed is needed last (Belady's choice); decode's records stay. nil once none is left.
    struct Passed {
        var empty: [Int], records: [(key: Int, slot: Int)], lo = 0, hi: Int, below: [Int] = []
        mutating func next(before bound: Int) -> Int? {
            if let slot = empty.popLast() { return slot }
            while lo < hi && records[lo].key < bound {
                below.append(records[lo].slot)
                lo += 1
            }
            if let slot = below.popLast() { return slot }
            guard lo < hi else { return nil }
            hi -= 1
            return records[hi].slot
        }
    }
    /// One miss's victim search (`FamilyExperts.fetch`, `guess`): its pool, `start`, the slots it may use and the
    /// prompt's `Passed`; `stage` and `failure` carry it through the retries a demand needs (guesses landed first).
    struct Request {
        enum Stage { case first, landed, allocatedLanded }
        enum Failure {
            case land, exhausted
            case denied(Error)
        }
        var pool: Int, start: UInt64, usable: Range<Int>, limit: Int
        var decode = false
        var prompt: Passed?
        /// A guess: its `Passed` bound, the slots it may take, and whether it may add a slab (not while the ledger
        /// refuses optional growth).
        var guess = false, bound = 0, eligible: Range<Int> = 0..<0, grows = true
        var stage = Stage.first
        var failure: Failure?, denied: Error?
    }

    /// Each pool's first slot number and slot bytes; experts per layer and per token.
    let first: [Int], slotBytes: [Int]
    let experts: Int, perToken: Int
    /// Accesses over which a slot's heat halves.
    let halfLife: Double
    package var heat: [Double], level: [Double], stamp: [UInt64], scan: [Bool]
    /// Each slot's log2 storage cost (0 unless the hot copy holds its record), so a miss does no hash lookup per slot,
    /// and `level + cost`, the order a victim scan compares (kept with both, the same sum bit for bit).
    private(set) var cost: [Double], rank: [Double]
    /// Each slot's phase as a victim scan needs it, kept from the core's callbacks: `empty` (free or no slab), `idle` (a
    /// verified record, unpinned) or `busy` (reserved, being read or pinned).
    private(set) var state: [UInt8]
    static let empty: UInt8 = 0, idle: UInt8 = 1, busy: UInt8 = 2
    package var clock: UInt64 = 0
    /// `clock` when the last `fetch` began: slots stamped after it are the ones that call returned (`keepingLast`).
    var lastStart: UInt64 = 0
    /// The attached hot copy's records and the per-layer log2 copy cost, frozen at idle/request boundaries.
    private(set) var hotOffsets: [Int: Int64]?
    private(set) var storageLogCost: [Int: Double] = [:]
    var request: Request?
    /// The owner's bookkeeping for a key leaving the lookup (`FamilyExperts`: unread guesses).
    var forgetting: (Int) -> Void = { _ in }
    /// Allocates pool `p`'s slabs up to slot `s` (`FamilyExperts.allocateLocked`).
    var allocate: (Int, Int) throws -> Void = { _, _ in }
    /// Per pool, the slot index from which whole slabs retire (`orderRetirement`).
    var retireFrom: [Int]

    package init(first: [Int], slotBytes: [Int], slots: Int, experts: Int, perToken: Int, halfLife: Double) {
        self.first = first
        self.slotBytes = slotBytes
        self.experts = experts
        self.perToken = perToken
        self.halfLife = halfLife
        heat = Array(repeating: 0, count: slots)
        level = Array(repeating: 0, count: slots)
        stamp = Array(repeating: 0, count: slots)
        scan = Array(repeating: false, count: slots)
        cost = Array(repeating: 0, count: slots)
        rank = Array(repeating: 0, count: slots)
        state = Array(repeating: Self.empty, count: slots)
        retireFrom = Array(repeating: .max, count: first.count)
    }

    func touch(_ slot: Int, heat value: Double) {
        heat[slot] = value
        stamp[slot] = clock
        level[slot] = log2(value) + Double(clock) / halfLife
        rank[slot] = level[slot] + cost[slot]
    }
    /// `key`'s record now lives in `slot` (its `level` set): its storage cost.
    func placed(_ key: Int, at slot: Int) {
        cost[slot] = storageCost(key)
        rank[slot] = level[slot] + cost[slot]
    }
    private func storageCost(_ key: Int) -> Double {
        hotOffsets?[key] == nil ? 0 : storageLogCost[key / experts] ?? 0
    }
    /// A new hot copy or new copy costs: every slot's cost again (and its rank, from its `level`).
    package func setStorage(hot: [Int: Int64]?, logCost: [Int: Double], _ core: Core) {
        hotOffsets = hot
        storageLogCost = logCost
        for (p, cls) in core.classes.enumerated() {
            for (i, slot) in cls.slots.enumerated() { cost[first[p] + i] = slot?.key.map(storageCost) ?? 0 }
        }
        for s in rank.indices { rank[s] = level[s] + cost[s] }
    }
    func storageMultiplier(_ key: Int) -> Double {
        hotOffsets?[key] == nil ? 1 : exp2(storageLogCost[key / experts] ?? 0)
    }

    /// The key in slot `s` of pool `p` (nil: empty, or not allocated).
    func owner(_ s: Int, pool p: Int, _ core: Core) -> Int? {
        let i = s - first[p]
        let all = core.classes[p].slots
        return i < all.count ? all[i]?.key : nil
    }
    /// Not reserved, being read or pinned (an empty slot is idle).
    func idle(_ s: Int) -> Bool { state[s] != Self.busy }
    /// Whether slot `s` of pool `p` has a slab.
    func allocated(_ s: Int, pool p: Int, _ core: Core) -> Bool {
        let i = s - first[p]
        let all = core.classes[p].slots
        return i < all.count && all[i] != nil
    }
    /// Pool `p`'s slots with a slab (slabs are added and retired at the end).
    func allocatedCount(_ p: Int, _ core: Core) -> Int { core.classes[p].slabs.values.reduce(0) { $0 + $1.slots.count } }

    /// Every slot's key and whether it is reserved, being read or pinned, as the opt-in variants take them.
    func table(_ core: Core) -> (owner: [Int?], busy: [Bool]) {
        var owner = [Int?](repeating: nil, count: heat.count)
        for (p, cls) in core.classes.enumerated() {
            for (i, slot) in cls.slots.enumerated() { owner[first[p] + i] = slot?.key }
        }
        return (owner, state.map { $0 == Self.busy })
    }

    /// The slot of pool `p`'s `slots` a miss reuses: the first empty one, else the coolest by heat decayed to `clock` and
    /// scaled by its storage cost (the first of equals), never one stamped after `start`, reserved, being read or pinned.
    /// `level` (log2 heat + stamp / halfLife) orders slots as their decayed heat does without an exp2 per slot; decayed
    /// heat itself is compared only for slots within rounding of the lowest level, or so cold it may underflow, so the
    /// choice is the full scan's bit for bit (`family_victim_choice`). -1: every slot is taken.
    package func storageVictim(_ slots: Range<Int>, start: UInt64) -> Int {
        var lowest = Double.infinity
        for s in slots where stamp[s] <= start && state[s] != Self.busy {
            if state[s] == Self.empty { return s }
            lowest = min(lowest, rank[s])
        }
        // Rounding of level and decayed heat is ~1e-15 in log2 units; below 2^-1000 the product may lose precision.
        let near = lowest + 1e-9 * max(1, lowest.magnitude)
        let cold = Double(clock) / halfLife - 1000
        var victim = -1
        var coolest = Double.infinity
        for s in slots where stamp[s] <= start && state[s] == Self.idle && (rank[s] <= near || rank[s] < cold) {
            let h = heat[s] * exp2(cost[s] - Double(clock - stamp[s]) / halfLife)
            if h < coolest {
                coolest = h
                victim = s
            }
        }
        return victim
    }

    /// `Passed` over pool `p`'s slots `slots` stamped at or before `start` and idle.
    func passed(_ slots: Range<Int>, start: UInt64, pool p: Int, _ core: Core) -> Passed {
        var empty: [Int] = []
        var records: [(key: Int, slot: Int)] = []
        let all = core.classes[p].slots
        for s in slots where stamp[s] <= start && idle(s) {
            if state[s] == Self.empty {
                empty.append(s)
            } else if scan[s] {
                records.append((all[s - first[p]]!.key!, s))
            }
        }
        records.sort { $0.key < $1.key }
        return Passed(empty: empty.reversed(), records: records, hi: records.count)
    }

    // MARK: ExpertSlotPolicy

    /// The slot `request` takes for `key`, its record (if any) evicted; nil with `request.failure` set when none may be
    /// had now. A slot without a slab gets one first (`allocate`); a slab denied for memory falls back to the slots
    /// already allocated, as `FamilyExperts.fetch` describes.
    package func victim(for key: Int, sizeClass p: Int, in core: Core) throws -> Int? {
        guard var r = request else { return nil }
        request = nil  // `r` alone owns the request, so taking from its `Passed` copies nothing
        defer { request = r }
        let base = first[p]
        func allocatedSlots() -> Range<Int> { base..<(base + min(allocatedCount(p, core), r.limit)) }
        func take(_ s: Int) -> Int {
            if owner(s, pool: p, core) != nil { core.evictLocked(s - base, sizeClass: p) }
            return s - base
        }
        func nextPromptVictim(before bound: Int) -> Int? {
            // A denied lazy slab may have fallen back to a slot still in Passed's snapshot. Never select that
            // now-current-wave owner a second time. Each stale entry is consumed once, so exhaustion reaches the live scan.
            while let candidate = r.prompt?.next(before: bound) {
                if stamp[candidate] <= r.start && idle(candidate) { return candidate }
            }
            return nil
        }
        if r.guess {
            var victim = nextPromptVictim(before: r.bound) ?? storageVictim(r.eligible, start: r.start)
            guard victim >= 0 else { return nil }
            // Speculation never waits for busy slots. Without a new slab (memory pressure, a full ledger) the guess takes
            // a slot that already has one, by the same rules, so a smaller cache still reads ahead.
            if !allocated(victim, pool: p, core), !r.grows || (try? allocate(p, victim)) == nil {
                repeat { victim = nextPromptVictim(before: r.bound) ?? -1 } while victim >= 0 && !allocated(victim, pool: p, core)
                if victim < 0 { victim = storageVictim(r.eligible.clamped(to: allocatedSlots()), start: r.start) }
                guard victim >= 0 else { return nil }
            }
            return take(victim)
        }
        func select(_ slots: Range<Int>) -> Int { storageVictim(slots, start: r.start) }
        var victim: Int
        switch r.stage {
        case .first:
            victim = nextPromptVictim(before: key) ?? storageVictim(r.usable, start: r.start)
            guard victim >= 0 else {
                // Guesses still read hold their slots (a `keepingLast` wave has no room to spare): land them first.
                r.stage = .landed
                r.failure = .land
                return nil
            }
        case .landed:
            victim = select(r.usable)
            guard victim >= 0 else {
                r.failure = .exhausted
                return nil
            }
        case .allocatedLanded:
            victim = select(allocatedSlots())
            guard victim >= 0 else {
                r.failure = .denied(r.denied!)
                return nil
            }
            return take(victim)
        }
        if !allocated(victim, pool: p, core) {
            // Under a hard physical or ledger bound, keep serving from existing slots. Admission happens before
            // evicting the chosen record, and the fallback preserves the same busy/current-wave exclusions.
            do { try allocate(p, victim) } catch let error as EngineError where error.code == "insufficient_memory" {
                victim = select(allocatedSlots())
                guard victim >= 0 else {
                    r.stage = .allocatedLanded
                    r.denied = error
                    r.failure = .land
                    return nil
                }
            }
        }
        return take(victim)
    }
    package func didHit(_ slot: Int, sizeClass: Int) {}
    /// Also called by the owner for a slot it reserves itself (`FamilyExperts.warm`).
    package func didReserve(_ slot: Int, sizeClass p: Int) { state[first[p] + slot] = Self.busy }
    package func didChange(_ slot: Int, key: Int, sizeClass p: Int, to phase: ExpertSlotPhase) {
        state[first[p] + slot] = phase == .pinned ? Self.busy : Self.idle
    }
    package func releasesOnLastUnpin(_ slot: Int, sizeClass: Int) -> Bool { false }
    package func didForget(_ key: Int) {}
    package func didInsertSlab(_ slots: [Int], sizeClass p: Int) { for i in slots { state[first[p] + i] = Self.empty } }
    /// Only whole slabs at or past `retireFrom` retire, the last first.
    package func orderRetirement(_ slabs: inout [Int], sizeClass p: Int, in core: Core) {
        let start = { (id: Int) in core.classes[p].slabs[id]!.slots.min() ?? 0 }
        slabs = slabs.filter { start($0) >= retireFrom[p] }.sorted { start($0) > start($1) }
    }

    /// Checks only: the first slot whose `state` is not what the core's phase makes it, nil when all agree.
    func stateMismatch(_ core: Core) -> String? {
        for p in first.indices {
            let end = p + 1 < first.count ? first[p + 1] : state.count
            for s in first[p]..<end {
                let local = s - first[p]
                let phase = local < core.classes[p].slots.count ? core.classes[p].slots[local]?.phase : nil
                let want: UInt8
                switch phase {
                case nil, .free?: want = Self.empty
                case .verified?, .evictable?: want = Self.idle
                case .reserved?, .reading?, .pinned?: want = Self.busy
                }
                if state[s] != want { return "pool \(p) slot \(local): mirror \(state[s]), core \(phase.map(\.rawValue) ?? "none")" }
            }
        }
        return nil
    }

    // MARK: ExpertSlotObserver

    package func isHeld(_ slot: Int, key: Int?, sizeClass: Int) -> Bool { false }
    package func didVerify(_ slot: Int, sizeClass: Int, in slots: ExpertSlotClass) {}
    /// An empty slot has no heat.
    package func willForget(_ key: Int, slot: Int?, sizeClass p: Int) {
        if let slot { (heat[first[p] + slot], state[first[p] + slot]) = (0, Self.empty) }
        forgetting(key)
    }
    package func didInsertSlab(_ id: Int, sizeClass: Int, in slots: ExpertSlotClass) {}
}
