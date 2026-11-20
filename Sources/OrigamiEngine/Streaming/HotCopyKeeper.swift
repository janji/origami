import Foundation

/// Keeps a model's hot copy while it serves (`FamilyServingBackend.keeper`): after each reply (with the counts
/// it saved), after a change from the dashboard (`POST v1/hot-copy`) and once at start, it plans from the counts and
/// builds, tops up or rebuilds the copy on its own thread, one record at a time, pausing while a request holds the
/// engine (so a reply never shares the model's drive with it); the finished copy is attached between requests
/// (`attach`), with no restart. Reads in flight keep the copy they started with; a replaced copy's file is removed
/// after the swap (reads open on it still complete). A copy at its size follows usage in place: between requests the
/// least-used records inside are swapped for clearly more-used ones outside (`swaps`, `HotCopy.replace`), a record at a
/// time, with no second file. `place` says where the copy goes and how big it may be.
package final class HotCopyKeeper: @unchecked Sendable {
    /// Where the copy goes: its folder and drive, its size (`budget`, within the room the drive has) and the drive's
    /// free bytes; or why there is none (`state`: off, not_possible, not_connected, not_allowed) in the card's words.
    /// `auto` (an automatic size): the budget may grow from the usage curve (`HotCopyCurve.automatic`) up to `room`
    /// (the drive's free bytes above `keepFree`, the copy's own counted) and `whole` (the model's expert bytes).
    package enum Place: Sendable {
        case folder(URL, drive: String, budget: UInt64, free: UInt64, target: UInt64? = nil, auto: Auto? = nil)
        case none(state: String, detail: String)
    }
    package struct Auto: Sendable {
        package let room: UInt64, whole: UInt64
        package init(room: UInt64, whole: UInt64) {
            self.room = room
            self.whole = whole
        }
        /// GLM's and DeepSeek's first automatic size: half the routed experts' bytes (`whole`), at most 40 GB and the
        /// `room` above `keepFree`, at least 20 GB; the curve grows it from there.
        package var familyStart: UInt64 { max(20_000_000_000, min(whole / 2, 40_000_000_000, room)) }
    }
    package enum Plan: Equatable {
        case none
        case build([Int])
        case topUp([Int])
        case rebuild([Int])
    }
    /// A top-up waits for this much more to add unless `Build now` asks.
    package static let topUpBytes: UInt64 = 1_000_000_000
    /// Bytes a copy leaves free on its drive.
    package static let keepFree: UInt64 = 20_000_000_000

    /// What to do with a copy holding `held` (`heldBytes`) when `chosen` is what a copy of `budget` would hold: build it
    /// when there is none; rebuild it once the size is lowered below it (by 10% and 1 GB, or with under 80% of its
    /// records still chosen); else top up (the chosen records it lacks, up to `budget`) once they come to `topUpBytes`
    /// or `force` asks. Nothing without counts. A copy at its size whose records are no longer chosen follows usage by
    /// swaps in place (`swaps`), not by a rebuild.
    package static func plan(held: [Int], heldBytes: UInt64, chosen: [Int], budget: UInt64, force: Bool = false, size: (Int) -> UInt64)
        -> Plan
    {
        guard !chosen.isEmpty else { return .none }
        guard !held.isEmpty else { return .build(chosen) }
        let have = Set(held)
        // Above its size, a copy whose records are under 80% chosen is rebuilt smaller (as before swaps existed).
        if heldBytes > budget, have.intersection(chosen).count * 5 < held.count * 4 { return .rebuild(chosen) }
        if heldBytes > budget && heldBytes - budget >= max(1_000_000_000, heldBytes / 10) { return .rebuild(chosen) }
        let add = topUp(have: have, heldBytes: heldBytes, chosen: chosen, budget: budget, size: size)
        return !add.keys.isEmpty && (force || add.bytes >= topUpBytes) ? .topUp(add.keys) : .none
    }
    /// Bytes swaps may read from the model's drive per hour (that drive slows under heavy reading): a top-up's amount.
    /// (A variable only for model-free checks, which lower it.)
    nonisolated(unsafe) package static var swapBytesPerHour = topUpBytes

    /// Whether counts `b` (misses, uses) are clearly above `a`'s in the ranking's order (misses first, then uses): by
    /// more than the fifth of drift a copy tolerates before a re-rank pays (`plan`'s 80% rule), so two records of
    /// similar use never trade places back and forth. It implies `b` ranks above `a`.
    package static func clearlyMore(_ b: (misses: Int, uses: Int), than a: (misses: Int, uses: Int)) -> Bool {
        Double(b.misses) * 4 > Double(a.misses) * 5 || (b.misses >= a.misses && Double(b.uses) * 4 > Double(a.uses) * 5)
    }
    /// In-place swaps for a copy with `slots` (key, -1 for empty, and padded bytes, file order): each chosen record it
    /// lacks (in `chosen` order, less `skip`, the ones a top-up adds) takes the slot of the same size holding the least
    /// used record not chosen (an empty slot first), when it is `clearlyMore` used; at most `limit` bytes of records
    /// replaced (the hourly allowance: re-ranking). Empty slots take at most `fillable` bytes (the drive's room above
    /// `keepFree`: a slot given back fills again once space has returned) and no allowance: refilling is growth.
    package static func swaps(
        slots: [(key: Int, bytes: UInt64)], chosen: [Int], skip: Set<Int> = [], counts: (uses: [Int: Int], misses: [Int: Int]),
        size: (Int) -> UInt64, limit: UInt64, redundant: Set<Int> = [], fillable: UInt64 = .max
    ) -> [(slot: Int, key: Int)] {
        let want = Set(chosen)
        let held = Set(slots.map(\.key))
        // A record another drive holds too (`redundant`, `HotCopyPlaces`) counts as unused here: duplicates go first.
        func count(_ key: Int) -> (misses: Int, uses: Int) {
            key < 0 || redundant.contains(key) ? (0, 0) : (counts.misses[key] ?? 0, counts.uses[key] ?? 0)
        }
        func rank(_ slot: Int) -> (Int, Int, Int) { (count(slots[slot].key).misses, count(slots[slot].key).uses, slots[slot].key) }
        // Per slot size, the victims least used last (an empty slot is the least).
        var victims: [UInt64: [Int]] = [:]
        var empty: UInt64 = 0
        for slot in slots.indices where !want.contains(slots[slot].key) {
            if slots[slot].key < 0 {
                guard empty + slots[slot].bytes <= fillable else { continue }
                empty += slots[slot].bytes
            }
            victims[slots[slot].bytes, default: []].append(slot)
        }
        for bytes in victims.keys { victims[bytes]!.sort { rank($0) > rank($1) } }
        var out: [(slot: Int, key: Int)] = []
        var total: UInt64 = 0
        for key in chosen where !held.contains(key) && !skip.contains(key) {
            let bytes = size(key)
            guard let slot = victims[bytes]?.last else { continue }
            let victim = slots[slot].key
            guard victim < 0 || (total + bytes <= limit && clearlyMore(count(key), than: count(victim))) else { continue }
            victims[bytes]!.removeLast()
            out.append((slot, key))
            if victim >= 0 { total += bytes }
        }
        return out
    }

    /// The slots a copy gives back first for a save (`giveBack`): empty slots still on the drive (`emptyHeld`), then
    /// records another drive's copy holds too (`redundant`), then the least used (fewest misses, then uses, as the
    /// ranking orders them); whole records, as few as cover `bytes`.
    package static func giveBackOrder(
        slots: [(key: Int, bytes: UInt64)], emptyHeld: [Int], counts: (uses: [Int: Int], misses: [Int: Int]), redundant: Set<Int>,
        bytes: UInt64
    ) -> [Int] {
        let held = slots.indices.filter { slots[$0].key >= 0 }.sorted {
            let (a, b) = (slots[$0].key, slots[$1].key)
            return (redundant.contains(a) ? 0 : 1, counts.misses[a] ?? 0, counts.uses[a] ?? 0, a)
                < (redundant.contains(b) ? 0 : 1, counts.misses[b] ?? 0, counts.uses[b] ?? 0, b)
        }
        var out: [Int] = []
        var total: UInt64 = 0
        for slot in emptyHeld + held where total < bytes {
            out.append(slot)
            total += slots[slot].bytes
        }
        return out
    }

    private static func topUp(have: Set<Int>, heldBytes: UInt64, chosen: [Int], budget: UInt64, size: (Int) -> UInt64) -> (
        keys: [Int], bytes: UInt64
    ) {
        var room = budget > heldBytes ? budget - heldBytes : 0
        var keys: [Int] = []
        var bytes: UInt64 = 0
        for key in chosen where !have.contains(key) {
            let s = size(key)
            if s <= room {
                keys.append(key)
                room -= s
                bytes += s
            }
        }
        return (keys, bytes)
    }

    private let source: HotCopySource, identity: String
    private let choose: (_ counts: (uses: [Int: Int], misses: [Int: Int]), _ budget: UInt64) -> [Int]
    private let attach: (HotCopy?) -> Bool, idle: () -> Bool
    private let withIdle: (_ work: () throws -> Void) throws -> Bool
    private let event: (String) -> Void
    private let retire: ((_ folder: URL) -> Void)?
    private var retired = Set<URL>()
    private let lock = NSLock()
    private var place: (_ heldHere: (URL) -> UInt64) -> Place
    private var counts: (uses: [Int: Int], misses: [Int: Int])
    private var copy: HotCopy?
    private var rebuildReason: String?
    private var state = "learning", detail: String?, folder: URL?, drive: String?, budget: UInt64 = 0, since: Date?, topUpDue: UInt64 = 0
    private var progress: (done: Int64, total: Int64, rate: Double)?
    private var swapping = false
    /// Another process has the copy open (`HotCopy.SharedElsewhere`): no swap or give-back changes it meanwhile.
    private var elsewhere = false
    private var curve: HotCopyCurve?, limit: HotCopyCurve.Limit?
    private var repaired = 0
    /// Copies replaced by another of the same payload: swaps wait until their reads in flight are done.
    private var stale: [HotCopy] = []
    /// Swaps: the bytes read for them in the last hour, and the experts swapped today (`day`).
    private var swapReads: [(at: Date, bytes: UInt64)] = [], swappedToday = 0, day = Calendar.current.startOfDay(for: Date())
    private var pending = false, forced = false, running = false, stopped = false
    /// A failed build is planned again after a reply once this much time has passed (`retryAt`).
    package var retryDelay: TimeInterval = 300
    private var retryAt = Date.distantFuture
    /// This copy's entry in the drive's account (`DiskBudget`), and the bytes it gave back for saves since start.
    private var budgetEntry: Int?
    private var givenBack: (records: Int, bytes: UInt64) = (0, 0)
    /// The model this copy is of, as the disk budget names it (`DiskBudget.take`: idle models' copies give to it).
    package var model = ""
    /// What idle models' copies gave this one to grow (per model: bytes, last time).
    private var took: [String: (bytes: UInt64, at: Date)] = [:]
    /// The swaps the ranking wants now that the hourly allowance holds back (records, bytes, the uses those records
    /// count), and when that was last logged.
    private var backlog: (records: Int, bytes: UInt64, uses: Int) = (0, 0, 0), backlogLogged = Date.distantPast

    /// `source` the model's records, `identity` the model a copy belongs to, `copy` the one attached at start. `choose`
    /// ranks the records a copy of a budget holds from the counts (default `HotCopy.choose`: most missed, then most
    /// used). `attach` swaps the copy the engine reads from, returning false while a request holds the engine; `idle`
    /// says no request does (nor the start's warm-up). `place` gets the bytes a copy already in a folder holds (counted
    /// as room there). `retire` runs once per folder as the first build or rebuild there starts: it removes files an
    /// earlier copy of this model left that the new one replaces (Qwen's offline mirror).
    package init(
        source: HotCopySource, identity: String, copy: HotCopy?, counts: (uses: [Int: Int], misses: [Int: Int]),
        choose: ((_ counts: (uses: [Int: Int], misses: [Int: Int]), _ budget: UInt64) -> [Int])? = nil,
        place: @escaping (_ heldHere: (URL) -> UInt64) -> Place, attach: @escaping (HotCopy?) -> Bool,
        idle: @escaping () -> Bool,
        withIdle: ((_ work: () throws -> Void) throws -> Bool)? = nil, retire: ((_ folder: URL) -> Void)? = nil,
        event: @escaping (String) -> Void
    ) {
        self.retire = retire
        self.source = source
        self.identity = identity
        self.choose = choose ?? { counts, budget in HotCopy.choose(counts.uses, misses: counts.misses, source: source, budget: budget) }
        self.copy = copy
        self.counts = counts
        self.place = place
        self.attach = attach
        self.idle = idle
        self.event = event
        self.withIdle =
            withIdle ?? { work in
                guard idle() else { return false }
                try work()
                return true
            }
        if let copy {
            state = "ready"
            folder = copy.folder
            since = Date()
        }
        budgetEntry = DiskBudget.shared.add(
            "hot copy", folder: { [weak self] in self?.current?.folder }, bytes: { [weak self] in self?.current?.diskBytes ?? 0 },
            about: { [weak self] in
                self.map { keeper in keeper.lock.withLock { ["model": keeper.model, "loaded": true, "took": keeper.tookLocked] } } ?? [:]
            },
            giveBack: { [weak self] bytes, reason in self?.giveBack(bytes, for: reason) ?? 0 })
    }
    deinit { if let budgetEntry { DiskBudget.shared.remove(budgetEntry) } }

    /// For a save that lacks `bytes` of room on this copy's drive (`DiskBudget`): the attached copy empties its least-used
    /// whole records (`giveBackOrder`), only as many as cover `bytes`, and frees their space on the drive
    /// (`HotCopy.giveBack`); one line in the log. Its swaps fill those slots again once the drive has room above
    /// `keepFree`. The only case a working copy shrinks without the owner asking; it is bounded and partial. The bytes
    /// freed.
    package func giveBack(_ bytes: UInt64, for reason: String) -> UInt64 {
        // A copy this one replaced may still finish a read of a slot about to be freed: wait briefly for it.
        let end = Date().addingTimeInterval(2)
        while !lock.withLock({ () -> Bool in
            stale.removeAll { $0.readsInFlight == 0 }
            return stale.isEmpty
        }) {
            guard Date() < end else { return 0 }
            Thread.sleep(forTimeInterval: 0.01)
        }
        // A drive that cannot free part of a file: no record is emptied for nothing (the save is counted as skipped).
        guard let copy = lock.withLock({ copy }), copy.punchable else { return 0 }
        let counts = lock.withLock { self.counts }
        let slots = copy.slots
        let order = Self.giveBackOrder(slots: slots, emptyHeld: copy.emptyHeld, counts: counts, redundant: redundant(), bytes: bytes)
        var failure: Error?
        var emptied: [(slot: Int, key: Int)] = []
        do {
            emptied = try copy.giveBack(order)
        } catch is HotCopy.SharedElsewhere {
            // Another process reads this copy by the index it opened: nothing is freed under it (the save asks the next).
            lock.withLock { elsewhere = true }
            return 0
        } catch { failure = error }
        // The records' slot bytes, not the drive's block count: what a punched slot really frees depends on the volume
        // (local snapshots, shared extents); the saver measures its free space again and asks once more while still
        // short. Slots that were empty already count nothing, so asking again always takes a record or ends. Counted
        // from what was freed, not from the order asked.
        let records = emptied.filter { $0.key >= 0 }.count
        let freed = emptied.filter { $0.key >= 0 }.reduce(UInt64(0)) { $0 + slots[$1.slot].bytes }
        lock.withLock { elsewhere = false }
        guard freed > 0 || failure != nil else { return 0 }
        lock.withLock { givenBack = (givenBack.records + records, givenBack.bytes + freed) }
        event(
            "Hot copy gave back \(OrigamiPlan.gigabytes(freed)) (\(records) least-used experts) for \(reason), which lacked "
                + "\(OrigamiPlan.gigabytes(bytes)) of room; it grows back once the drive has room again."
                + (failure.map { " Stopped: \($0)." } ?? ""))
        return freed
    }

    /// Plans again (after a reply with its `counts`, or a change of `place`), building on the keeper's thread; `force`
    /// tops up whatever the amount (`Build now`).
    package func trigger(
        counts: (uses: [Int: Int], misses: [Int: Int])? = nil, place: ((_ heldHere: (URL) -> UInt64) -> Place)? = nil, force: Bool = false
    ) {
        let start: Bool = lock.withLock {
            if let counts { self.counts = counts }
            if let place { self.place = place }
            // A failed build is tried again on `Build now`, a change of place, or a reply once `retryDelay` has passed.
            guard place != nil || force || state != "failed" || Date() >= retryAt else { return false }
            pending = true
            forced = forced || force
            guard !running else { return false }
            running = true
            return true
        }
        guard start else { return }
        Thread.detachNewThread { [self] in
            while let force = lock.withLock({ () -> Bool? in
                guard pending, !stopped else {
                    running = false
                    return nil
                }
                pending = false
                defer { forced = false }
                return forced
            }) { step(force: force) }
        }
    }
    /// Ends a build at its next record (its partial `experts.next` is removed), waiting up to `seconds` for that; one cut
    /// off anyway (the process gone first) is overwritten by the next build in that folder.
    package func stop(waiting seconds: Double = 0) {
        lock.withLock { stopped = true }
        let end = Date().addingTimeInterval(seconds)
        while lock.withLock({ running }), Date() < end { Thread.sleep(forTimeInterval: 0.05) }
    }

    private struct Stopped: Error {}
    private func set(_ state: String, _ detail: String? = nil) {
        lock.withLock {
            self.state = state
            self.detail = detail
        }
    }

    private func step(force: Bool) {
        let (attached, counts, place) = lock.withLock { (copy, self.counts, self.place) }
        var current = attached
        // Room held by the copy in a folder: the attached one's, or with none attached a finished one there (kept while its drive was gone).
        let spot = place { folder in
            if let current, current.folder.standardizedFileURL == folder.standardizedFileURL { return current.diskBytes }
            return HotCopy.open(folder: folder, identity: identity, source: source)?.diskBytes ?? 0
        }
        guard case .folder(let folder, let drive, let capacity, let free, _, let auto) = spot else {
            guard case .none(let state, let detail) = spot else { return }
            // Unavailable or disabled: detach the copy, retaining its files for reuse.
            if let current, detach(current) {
                event(
                    "Hot copy detached (\(current.keys.count) experts, \(OrigamiPlan.gigabytes(UInt64(current.bytes)))): \(detail) Its files are kept."
                )
            }
            lock.withLock {
                self.state = state
                self.detail = detail
                self.budget = 0
                topUpDue = 0
                limit = nil
            }
            return
        }
        var budget = capacity
        let size = { (key: Int) in HotCopy.size(key, source: self.source) ?? 0 }
        let counted = !counts.uses.isEmpty || !counts.misses.isEmpty
        let curve = counted ? HotCopyCurve(uses: counts.uses, order: choose(counts, .max), size: size) : nil
        if let auto {
            let held = current.map { $0.folder.standardizedFileURL == folder.standardizedFileURL ? UInt64($0.bytes) : 0 } ?? 0
            let pick = curve?.automatic(floor: budget, room: auto.room, whole: auto.whole) ?? (min(budget, auto.room), .noCounts)
            // Grows by top-ups as the curve pays; neither a shifting ranking nor falling free space shrinks a copy.
            budget = max(budget, pick.bytes, held)
            lock.withLock { limit = pick.limit }
        } else {
            lock.withLock { limit = nil }
        }
        lock.withLock {
            self.folder = folder
            self.drive = drive
            self.budget = budget
            self.curve = curve
        }
        if current?.folder.standardizedFileURL != folder.standardizedFileURL,
            let found = HotCopy.open(folder: folder, identity: identity, source: source), swap(found)
        {
            lock.withLock {
                if let current { retireLocked(current) }
                copy = found
                since = Date()
            }
            event("Hot copy attached again: \(found.keys.count) experts, \(OrigamiPlan.gigabytes(UInt64(found.bytes))) in \(folder.path).")
            current = found
        }
        // Records that failed their checksum are written again from the model while idle, then a fresh open reads them.
        if let damaged = current, damaged.folder.standardizedFileURL == folder.standardizedFileURL, !damaged.rejectedKeys.isEmpty {
            var fixed = 0
            _ = try? withIdle { fixed = try damaged.rejectedKeys.filter { try damaged.repair($0, source: source) }.count }
            if fixed > 0, let fresh = HotCopy.open(folder: folder, identity: identity, source: source), swap(fresh) {
                lock.withLock {
                    if copy === damaged {
                        retireLocked(damaged)
                        copy = fresh
                    }
                    repaired += fixed
                }
                current = fresh
            }
        }
        guard counted else {
            set(current == nil ? "learning" : "ready", following)
            return
        }
        let chosen = choose(counts, budget)
        let here = current.map { $0.folder.standardizedFileURL == folder.standardizedFileURL } ?? false
        let plan =
            current != nil && !here
            ? (chosen.isEmpty ? .none : .build(chosen))
            : Self.plan(
                held: current?.keys ?? [], heldBytes: UInt64(current?.bytes ?? 0), chosen: chosen, budget: budget, force: force, size: size)
        let due =
            current.map { Self.topUp(have: Set($0.keys), heldBytes: UInt64($0.bytes), chosen: chosen, budget: budget, size: size).bytes }
            ?? 0
        lock.withLock { topUpDue = plan == .none ? due : 0 }
        let keys: [Int]
        let onto: HotCopy?
        switch plan {
        case .none:
            if let current, here {
                let skip = Set(
                    Self.topUp(have: Set(current.keys), heldBytes: UInt64(current.bytes), chosen: chosen, budget: budget, size: size).keys)
                follow(current, chosen: chosen, skip: skip, counts: counts, size: size)
            }
            set(current == nil ? "learning" : "ready", following)
            return
        case .topUp(let add):
            keys = add
            onto = current
        case .build(let all):
            keys = all
            onto = nil
            lock.withLock { rebuildReason = current == nil ? "No compatible copy" : "Location changed" }
        case .rebuild(let all):
            keys = all
            onto = nil
            // Only a lowered size rebuilds a copy here (`plan`); usage shifts are followed by swaps in place.
            let reason = "Size limit reduced"
            lock.withLock { rebuildReason = reason }
            // The replacement is written beside the copy in use, which serves until it is attached; never delete-then-build.
            if let current, all.reduce(0, { $0 + size($1) }) + Self.keepFree > free {
                // No room beside it: the copy in use shrinks in place to the new size instead, its least-used records
                // given back to the drive (`giveBack`), and serves on.
                let held = current.slots.filter { $0.key >= 0 }.reduce(UInt64(0)) { $0 + $1.bytes }
                let given = held > budget ? giveBack(held - budget, for: "a smaller hot_copy size") : 0
                set(
                    "ready",
                    lock.withLock({ elsewhere }) && given == 0
                        ? "\(reason): \(Self.sharedText)"
                        : given > 0 || held <= budget
                            ? "\(reason): shrunk in place (its least-used experts given back); a compact copy is written once there is room beside it."
                            : "\(reason): the replacement waits for room beside the current copy, which stays in use; delete the copy in the dashboard's Models view to free its space now (turning the hot copy off keeps its files)."
                )
                return
            }
        }
        if onto == nil, let retire, lock.withLock({ retired.insert(folder.standardizedFileURL).inserted }) { retire(folder) }
        let total = keys.reduce(Int64(0)) { $0 + Int64(size($1)) }
        grow(UInt64(total), in: folder)
        let started = Date()
        lock.withLock {
            progress = (0, total, 0)
            state = "building"
            detail = nil
        }
        var mark = Date()
        var written: Int64 = 0
        var checked: Int64 = 0
        do {
            let wait: () throws -> Void = { [self] in
                while true {
                    if lock.withLock({ stopped }) { throw Stopped() }
                    // Other writers may fill the drive meanwhile: the build stops above the 10 GB floor. The shared
                    // reading: exFAT and other non-APFS drives report 0 for important usage with plain free space there.
                    if written - checked >= 1_000_000_000 {
                        checked = written
                        let left = DiskBudget.shared.free(at: folder)
                        if left < 10_000_000_000 { throw EngineError("\(shown(drive)) has under 10 GB free; the hot copy stopped") }
                    }
                    if idle() { break }
                    set("paused")
                    Thread.sleep(forTimeInterval: 0.2)
                    mark = Date()
                }
                set("building")
            }
            let perform: (_ work: () throws -> Void) throws -> Void = { [self] work in
                while true {
                    if lock.withLock({ stopped }) { throw Stopped() }
                    if try withIdle(work) { return }
                    set("paused")
                    Thread.sleep(forTimeInterval: 0.2)
                    mark = Date()
                }
            }
            let wrote: (Int64) -> Void = { [self] bytes in
                let seconds = max(1e-3, Date().timeIntervalSince(mark))
                mark = Date()
                written += bytes
                lock.withLock {
                    guard let p = progress else { return }
                    let rate = Double(bytes) / seconds
                    progress = (p.done + bytes, p.total, p.rate == 0 ? rate : p.rate * 0.9 + rate * 0.1)
                }
            }
            let built = try HotCopy.write(
                source: source, keys: keys, folder: folder, identity: identity, onto: onto, wait: wait, perform: perform, wrote: wrote)
            guard swap(built) else { throw Stopped() }
            let old = lock.withLock { () -> HotCopy? in
                let old = copy
                if let old { retireLocked(old) }
                copy = built
                state = "ready"
                detail = nil
                progress = nil
                since = Date()
                return old
            }
            if let old, old.folder.standardizedFileURL != folder.standardizedFileURL {
                try? FileManager.default.removeItem(at: old.folder)
                event(
                    "Hot copy at \(old.folder.path) removed (\(OrigamiPlan.gigabytes(UInt64(old.bytes)))): replaced by the one in \(folder.path)."
                )
            }
            event(
                onto == nil
                    ? String(
                        format: "Hot copy ready: %@ experts, %@ on %@ (%.0f s).",
                        built.keys.count.formatted(.number.locale(Locale(identifier: "en_US"))),
                        OrigamiPlan.gigabytes(UInt64(built.bytes)), shown(drive), Date().timeIntervalSince(started))
                        + (old.map {
                            " It replaces the previous copy (\(OrigamiPlan.gigabytes(UInt64($0.bytes)))): \(lock.withLock { rebuildReason } ?? "")."
                        } ?? "")
                    : "Hot copy grew by \(OrigamiPlan.gigabytes(UInt64(total))).")
        } catch is Stopped {
            lock.withLock {
                progress = nil
                state = copy == nil ? "learning" : "ready"
            }
        } catch {
            let delay = lock.withLock {
                progress = nil
                state = "failed"
                detail = "\(error)"
                retryAt = Date().addingTimeInterval(retryDelay)
                return retryDelay
            }
            event(
                "Hot copy could not be built (\(OrigamiPlan.gigabytes(UInt64(written))) of \(OrigamiPlan.gigabytes(UInt64(total))) written; "
                    + (current.map { "the copy in use, \(OrigamiPlan.gigabytes(UInt64($0.bytes))), stays" } ?? "no copy in use")
                    + "): \(error). Planned again after a reply in \(Int(delay / 60)) min.")
        }
    }

    /// The ready state's detail once swaps ran today (nil before), or why the copy is not changed in place now.
    private var following: String? {
        lock.withLock {
            elsewhere ? Self.sharedText : swappedToday > 0 ? "Following usage: \(swappedToday) experts swapped today." : nil
        }
    }
    private static let sharedText = "Another Origami process has this copy open: it is not changed while that lasts."
    private func retireLocked(_ old: HotCopy) {
        old.retire()
        stale.append(old)
    }

    /// One batch of swaps in `copy` (attached, at its folder), a record at a time while no request runs: it ends at a
    /// request (after the record in progress), at the hourly read bound, or when no swap is clearly worth it.
    private func follow(
        _ copy: HotCopy, chosen: [Int], skip: Set<Int>, counts: (uses: [Int: Int], misses: [Int: Int]), size: (Int) -> UInt64
    ) {
        let now = Date()
        let allowance = lock.withLock { () -> UInt64? in
            if Calendar.current.startOfDay(for: now) != day {
                day = Calendar.current.startOfDay(for: now)
                swappedToday = 0
            }
            swapReads.removeAll { now.timeIntervalSince($0.at) >= 3600 }
            // A copy replaced by this one may still finish a read of a slot about to be rewritten.
            stale.removeAll { $0.readsInFlight == 0 }
            let used = swapReads.reduce(UInt64(0)) { $0 + $1.bytes }
            return stale.isEmpty ? (used < Self.swapBytesPerHour ? Self.swapBytesPerHour - used : 0) : nil
        }
        guard let allowance else { return }
        let slots = copy.slots
        let duplicates = redundant()
        func fillable() -> UInt64 {
            let free = DiskBudget.shared.free(at: copy.folder)
            return free > Self.keepFree ? free - Self.keepFree : 0
        }
        // Empty slots refill into the drive's room and what idle models' copies on it may give (`grow`).
        let spare = DiskBudget.shared.spare(on: copy.folder, for: model)
        let wanted = Self.swaps(
            slots: slots, chosen: chosen, skip: skip, counts: counts, size: size, limit: .max, redundant: duplicates,
            fillable: fillable() + spare)
        let fills = wanted.filter { slots[$0.slot].key < 0 }.reduce(UInt64(0)) { $0 + size($1.key) }
        if fills > fillable() { grow(fills, in: copy.folder) }
        let pairs = Self.swaps(
            slots: slots, chosen: chosen, skip: skip, counts: counts, size: size, limit: allowance, redundant: duplicates,
            fillable: fillable())
        // What the allowance holds back: the record swaps the ranking wants now beyond it.
        let held = Set(pairs.map(\.key))
        let waiting = wanted.filter { slots[$0.slot].key >= 0 && !held.contains($0.key) }
        let left = (
            records: waiting.count, bytes: waiting.reduce(UInt64(0)) { $0 + size($1.key) },
            uses: waiting.reduce(0) { $0 + (counts.uses[$1.key] ?? 0) }
        )
        let due = lock.withLock { () -> Bool in
            backlog = left
            guard left.records > 0, now.timeIntervalSince(backlogLogged) >= 3600 else { return false }
            backlogLogged = now
            return true
        }
        if due {
            event(
                "Hot copy backlog: \(left.records) swaps (\(OrigamiPlan.gigabytes(left.bytes)), \(left.uses) counted uses) wait for the "
                    + "\(OrigamiPlan.gigabytes(Self.swapBytesPerHour)) per hour swap allowance.")
        }
        guard !pairs.isEmpty else {
            lock.withLock { elsewhere = false }
            return
        }
        lock.withLock { swapping = true }
        defer { lock.withLock { swapping = false } }
        func ends() -> String {
            func count(_ key: Int) -> (misses: Int, uses: Int) { (counts.misses[key] ?? 0, counts.uses[key] ?? 0) }
            let held = Set(copy.keys)
            let inside = held.min { count($0) < count($1) }.map(count)
            let outside = Set(counts.uses.keys).union(counts.misses.keys).subtracting(held).max { count($0) < count($1) }.map(count)
            func shown(_ c: (misses: Int, uses: Int)?) -> String { c.map { "\($0.misses) misses/\($0.uses) uses" } ?? "none" }
            return "least used inside \(shown(inside)), most used outside \(shown(outside))"
        }
        func count(_ key: Int) -> (misses: Int, uses: Int) { (counts.misses[key] ?? 0, counts.uses[key] ?? 0) }
        func shown(_ c: (misses: Int, uses: Int)) -> String { "\(c.misses) misses/\(c.uses) uses" }
        let before = ends()
        var done = 0
        var bytes: UInt64 = 0
        var failure: Error?
        for (slot, key) in pairs {
            if lock.withLock({ stopped }) { break }
            // This swap's hold on the slot: only it puts the slot's key back (a give-back may own the slot instead).
            var owner: Int?
            defer { if let owner { copy.restore(slot, owner: owner) } }
            do {
                // The slot leaves the lookup between requests; its reads in flight are waited for with no request held
                // (bounded: a read that never returns abandons the swap); then the record is written between requests.
                guard try withIdle({ owner = copy.vacate(slot) }) else { break }
                guard let owner else {
                    failure = EngineError("slot \(slot) is being given back; that swap waits")
                    break
                }
                guard copy.drain(slot, within: copy.drainBound(slot)) else {
                    failure = EngineError("a read of slot \(slot) did not return; that swap is abandoned and its record kept")
                    break
                }
                guard try withIdle({ try copy.replace(slot: slot, with: key, source: source, within: 0, owner: owner) }) else { break }
            } catch is HotCopy.SharedElsewhere {
                lock.withLock { elsewhere = true }
                break
            } catch {
                failure = error
                break
            }
            lock.withLock { elsewhere = false }
            done += 1
            bytes += size(key)
            lock.withLock {
                // Refilling an empty slot is growth, not re-ranking: it takes nothing from the hourly allowance.
                if slots[slot].key >= 0 { swapReads.append((Date(), size(key))) }
                swappedToday += 1
            }
        }
        guard done > 0 || failure != nil else { return }
        // What the done pairs took: empty slots, records a faster drive's copy also holds, and records outranked by the
        // margin, of which the closest pair (the most used record taken, with what replaced it) shows the margin held.
        let taken = pairs.prefix(done).map { (victim: slots[$0.slot].key, key: $0.key) }
        let empty = taken.filter { $0.victim < 0 }.count
        let doubled = taken.filter { $0.victim >= 0 && duplicates.contains($0.victim) }.count
        let closest = taken.filter { $0.victim >= 0 && !duplicates.contains($0.victim) }.max { count($0.victim) < count($1.victim) }
        let parts =
            [
                done - empty - doubled > 0 ? "\(done - empty - doubled) swapped" : nil, empty > 0 ? "\(empty) empty slots filled" : nil,
                doubled > 0 ? "\(doubled) records a faster drive also holds replaced" : nil,
            ]
            .compactMap { $0 }
        event(
            "Hot copy follows usage: \(parts.isEmpty ? "nothing written" : parts.joined(separator: ", ")) (\(OrigamiPlan.gigabytes(bytes)))"
                + (closest.map { "; closest swap: \(shown(count($0.key))) in for \(shown(count($0.victim))) out" } ?? "")
                + "; copy before: \(before); after: \(ends())." + (failure.map { " Stopped: \($0)." } ?? ""))
    }

    private func shown(_ drive: String) -> String { drive == "internal drive" ? "This Mac" : drive }
    /// Before writing `bytes` into `folder`: when the drive lacks them above `keepFree`, the idle models' copies on its
    /// drive give what is missing (`DiskBudget.take`).
    private func grow(_ bytes: UInt64, in folder: URL) {
        let free = DiskBudget.shared.free(at: folder)
        guard bytes + Self.keepFree > free else { return }
        let given = DiskBudget.shared.take(bytes + Self.keepFree - free, in: folder, for: model) { [self] line in event(line) }
        lock.withLock {
            for (from, bytes) in given { took[from] = ((took[from]?.bytes ?? 0) + bytes, Date()) }
        }
    }
    /// What idle models' copies gave this one, per model (under `lock`).
    private var tookLocked: [[String: Any]] {
        let format = ISO8601DateFormatter()
        return took.sorted { $0.value.at < $1.value.at }.map {
            ["from": $0.key, "bytes": $0.value.bytes, "at": format.string(from: $0.value.at)] as [String: Any]
        }
    }
    /// The experts read from `to` (nil: no copy) from a point no request runs; false when stopped first.
    private func swap(_ to: HotCopy?) -> Bool {
        while !attach(to) {
            if lock.withLock({ stopped }) { return false }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return true
    }
    /// The experts stop reading `current` (its files stay); false when stopped first.
    private func detach(_ current: HotCopy) -> Bool {
        guard swap(nil) else { return false }
        lock.withLock {
            if copy === current {
                retireLocked(current)
                copy = nil
                since = nil
            }
        }
        return true
    }

    /// For `/v1/status` (`experts_from`): the state (learning, building, paused, ready, off, not_possible,
    /// not_connected, not_allowed, failed), its detail, the copy's folder, drive, bytes and records, its size limit, a
    /// build's progress, the top-up waiting below `topUpBytes`, and since when the copy serves; `coverage`, the share of
    /// counted expert uses a copy of the size limit holds, and `size_limit`, what set an automatic size (null: set by the user).
    package var report: [String: Any] {
        lock.withLock {
            var out: [String: Any] = [
                "state": state, "detail": detail ?? NSNull(), "path": (copy?.folder ?? folder)?.path ?? NSNull(),
                "drive": drive ?? NSNull(), "bytes": copy?.bytes ?? 0, "records": copy?.keys.count ?? 0,
                "budget_bytes": budget, "topup_bytes": topUpDue, "progress": NSNull(),
                "since": since.map { ISO8601DateFormatter().string(from: $0) } ?? NSNull(),
                "has_counts": !counts.uses.isEmpty || !counts.misses.isEmpty, "build_reason": rebuildReason ?? NSNull(),
                "swapped_today": swappedToday, "rejected_records": copy?.rejectedRecords ?? 0,
                "repaired_records": repaired, "coverage": curve.map { $0.coverage(budget) as Any } ?? NSNull(),
                "size_limit": limit?.rawValue ?? NSNull(),
                "given_back_records": givenBack.records, "given_back_bytes": givenBack.bytes,
                "model": model, "took": tookLocked, "backlog_records": backlog.records, "backlog_bytes": backlog.bytes,
                "backlog_uses": backlog.uses,
            ]
            if let p = progress {
                out["progress"] = [
                    "done_bytes": p.done, "total_bytes": p.total,
                    "seconds_left": p.rate > 0 ? Int((Double(p.total - p.done) / p.rate).rounded(.up)) as Any : NSNull(),
                ]
            }
            return out
        }
    }
    /// The usage curve from the counts the last plan used (nil before any are counted).
    package var usage: HotCopyCurve? { lock.withLock { curve } }
    /// The records the attached copy holds (none: 0).
    package var records: Int { lock.withLock { copy?.keys.count ?? 0 } }
    /// The copy reads come from now.
    package var current: HotCopy? { lock.withLock { copy } }
    /// Whether a build, top-up or rebuild is under way (`HotCopyPlaces`: one keeper writes at a time).
    package var building: Bool { lock.withLock { progress != nil || swapping } }
    /// Records a faster drive's copy holds too (`HotCopyPlaces`): the first a swap replaces.
    package var redundant: () -> Set<Int> = { [] }
}
