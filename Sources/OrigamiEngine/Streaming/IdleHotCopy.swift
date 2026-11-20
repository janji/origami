import Foundation

/// A hot copy place of a model this server has not loaded, on a drive the loaded model's copy shares: a holder in the
/// drive's account (`DiskBudget`) that gives space back to the loaded model's copy (`DiskBudget.take`) and to saves, in
/// whole records from its least valuable end by its own model's counts (empty slots holding blocks, records an earlier
/// place of its model holds too, then the least missed and used), never below its floor (`floor`). Only while no
/// process has it open: the folder's `.attached.lock` is taken exclusive for the whole give-back (a process reading the
/// copy holds it shared, `HotCopy.holdAttached`), and no other process may have its payload open (a build without that
/// lock). The give-back is `HotCopy.giveBack`'s: under `.refresh.lock`, the index is published with the slots empty,
/// then their blocks are freed in place with the file layout unchanged. The copy grows back when its model is loaded
/// again (its keeper refills empty slots between requests).
package final class IdleHotCopy: @unchecked Sendable {
    /// The share of its model's counted uses an idle copy keeps: the most valuable records it holds that reach it. No
    /// rule of the usage curve names a share (its automatic size rule names a size), so half.
    package static let floorShare = 0.5

    package let model: String, folder: URL, identity: String
    /// The owner's size for this place (nil: automatic) and whether they asked to keep it while idle (`hot_copy_keep`):
    /// both set, the copy gives nothing.
    package let cap: UInt64?, keep: Bool
    /// When its model last served (its counts file's time; nil: never): the longest unused gives first.
    package let lastUsed: Date?
    private let counts: (uses: [Int: Int], misses: [Int: Int])
    /// Records an earlier place of the same model holds: given back first, and not counted toward the floor.
    private let redundant: () -> Set<Int>
    private let lock = NSLock()
    private var slots: [(key: Int, bytes: UInt64)] = []
    private var budgetEntry: Int?
    private var gave: [String: (bytes: UInt64, records: Int, at: Date)] = [:]

    package init(
        model: String, folder: URL, identity: String, cap: UInt64?, keep: Bool, lastUsed: Date?,
        counts: (uses: [Int: Int], misses: [Int: Int]), redundant: @escaping () -> Set<Int> = { [] }
    ) {
        self.model = model
        self.folder = folder
        self.identity = identity
        self.cap = cap
        self.keep = keep
        self.lastUsed = lastUsed
        self.counts = counts
        self.redundant = redundant
        refresh()
    }
    deinit { if let budgetEntry { DiskBudget.shared.remove(budgetEntry) } }

    /// Reads the copy's slots again (none: no copy of `identity` there).
    package func refresh() {
        let now = HotCopy.open(folder: folder, identity: identity, attaching: false)?.slots ?? []
        lock.withLock { slots = now }
    }
    /// Counts this copy on its drive's account; it stays there while this object lives.
    package func register() {
        guard lock.withLock({ budgetEntry == nil }) else { return }
        let entry = DiskBudget.shared.add(
            "hot copy of \(model)", folder: { [weak self] in self.flatMap { $0.held > 0 ? $0.folder : nil } },
            bytes: { [weak self] in self?.held ?? 0 }, model: model, lastUsed: lastUsed,
            spare: { [weak self] in self?.spare ?? 0 }, about: { [weak self] in self?.about ?? [:] },
            giveBack: { [weak self] bytes, reason in self?.giveBack(bytes, for: reason) ?? 0 })
        lock.withLock { budgetEntry = entry }
    }

    /// The keys it holds.
    package var keys: Set<Int> { lock.withLock { Set(slots.map(\.key).filter { $0 >= 0 }) } }
    /// Bytes its records hold (empty slots left out).
    package var held: UInt64 { lock.withLock { slots.filter { $0.key >= 0 }.reduce(0) { $0 + $1.bytes } } }
    /// The least it keeps: the smaller of what it holds and its most valuable records (by its model's misses, then
    /// uses; none another place holds) that reach `floorShare` of the counted uses. All it holds without counts, or when
    /// its records reach less.
    package var floor: UInt64 {
        let slots = lock.withLock { self.slots }
        return Self.floor(slots: slots, counts: counts, redundant: redundant())
    }
    package static func floor(slots: [(key: Int, bytes: UInt64)], counts: (uses: [Int: Int], misses: [Int: Int]), redundant: Set<Int>)
        -> UInt64
    {
        let records = slots.filter { $0.key >= 0 }
        let all = records.reduce(UInt64(0)) { $0 + $1.bytes }
        let counted = counts.uses.values.reduce(0, +)
        guard counted > 0 else { return all }
        func rank(_ key: Int) -> (Int, Int, Int) { (counts.misses[key] ?? 0, counts.uses[key] ?? 0, key) }
        var bytes: UInt64 = 0
        var uses = 0
        for record in records.filter({ !redundant.contains($0.key) }).sorted(by: { rank($0.key) > rank($1.key) }) {
            bytes += record.bytes
            uses += counts.uses[record.key] ?? 0
            if Double(uses) >= floorShare * Double(counted) { return min(bytes, all) }
        }
        return all
    }
    /// What it may give now: its records above the floor; none when kept by the owner or open in any process.
    package var spare: UInt64 {
        guard !(keep && cap != nil), !inUse else { return 0 }
        let held = held
        let floor = floor
        return held > floor ? held - floor : 0
    }
    /// Whether a process (this one included) has the copy open, or the drive keeps no locks.
    package var inUse: Bool {
        guard let fd = Self.exclusive(folder) else { return true }
        close(fd)
        return HotCopy.openElsewhere(folder)
    }
    /// `.attached.lock` held exclusive (nil: someone holds it, or the drive keeps no locks).
    private static func exclusive(_ folder: URL) -> Int32? {
        let fd = Darwin.open(folder.appendingPathComponent(".attached.lock").path, O_RDONLY | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { return nil }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            return nil
        }
        return fd
    }

    /// Frees at least `bytes` (fewer when its spare ends first) in whole records, least valuable first, for `who` (the
    /// loaded model's copy, or a save); the record bytes freed. Nothing while any process has the copy open.
    package func giveBack(_ bytes: UInt64, for who: String) -> UInt64 {
        guard !(keep && cap != nil), let fd = Self.exclusive(folder) else { return 0 }
        defer {
            _ = flock(fd, LOCK_UN)
            close(fd)
        }
        guard !HotCopy.openElsewhere(folder), let copy = HotCopy.open(folder: folder, identity: identity, attaching: false) else {
            return 0
        }
        let slots = copy.slots
        let redundant = redundant()
        let held = slots.filter { $0.key >= 0 }.reduce(UInt64(0)) { $0 + $1.bytes }
        let floor = Self.floor(slots: slots, counts: counts, redundant: redundant)
        let spare = held > floor ? held - floor : 0
        // Empty slots still holding blocks first (they serve nothing and count toward no floor), then records.
        var order: [Int] = []
        var records: UInt64 = 0
        for slot in HotCopyKeeper.giveBackOrder(
            slots: slots, emptyHeld: copy.emptyHeld, counts: counts, redundant: redundant, bytes: min(bytes, spare))
        {
            if slots[slot].key >= 0 {
                guard records + slots[slot].bytes <= spare else { break }
                records += slots[slot].bytes
            }
            order.append(slot)
        }
        guard !order.isEmpty, let emptied = try? copy.giveBack(order) else {
            refresh()
            return 0
        }
        let freed = emptied.filter { $0.key >= 0 }.reduce(UInt64(0)) { $0 + slots[$1.slot].bytes }
        let count = emptied.filter { $0.key >= 0 }.count
        lock.withLock {
            self.slots = copy.slots
            let before = gave[who]
            if freed > 0 { gave[who] = ((before?.bytes ?? 0) + freed, (before?.records ?? 0) + count, Date()) }
        }
        return freed
    }

    /// Its row in the disk budget's holders: its model, cap, floor, what it may still give, whether it is kept or open,
    /// when its model last served, and what it gave to whom and when.
    package var about: [String: Any] {
        let gave = lock.withLock { self.gave }
        let format = ISO8601DateFormatter()
        return [
            "model": model, "loaded": false, "cap": cap.map { $0 as Any } ?? "auto", "keep_when_idle": keep,
            "floor_bytes": floor, "spare_bytes": spare, "last_used": lastUsed.map { format.string(from: $0) } ?? NSNull(),
            "gave": gave.sorted { $0.value.at < $1.value.at }.map {
                ["to": $0.key, "bytes": $0.value.bytes, "records": $0.value.records, "at": format.string(from: $0.value.at)]
                    as [String: Any]
            },
        ]
    }
}
