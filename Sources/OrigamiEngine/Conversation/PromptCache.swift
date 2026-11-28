import CryptoKit
import Foundation

/// Saved prompts and conversations on disk, so a repeated prompt or a follow-up starts
/// from its saved state even after a restart. Each entry is one verified checkpoint
/// (`PersistentCheckpointStore`, paged format) in its own slot folder; a small index maps
/// slots to their token prefix. Lookup takes the longest saved prefix of a new request;
/// saving replaces the same prompt's entry, a free slot, or the least recently used one.
/// Entries from another epoch (another arithmetic, model or settings) are deleted at start. Builds of one arithmetic
/// share a folder once their canary matches (`PromptCanary`, `settle`).
public final class PromptCache {
    /// One saved image in `index.json`, keyed by its slot number. The tokens themselves are not stored here; the
    /// slot's checkpoint holds them and its read verifies them against the request.
    package struct Entry: Codable {
        let tokens: Int
        /// SHA-256 of the entry's token IDs, little-endian Int64.
        package let prefix: String
        var used: Double
        /// "system" for a system-prompt snapshot; nil (also older indexes) for prompt and conversation images.
        package var kind: String? = nil
        package var name: String? = nil
        package var pinned: Bool? = nil
        /// `Index.read` when the entry was last saved or used.
        var mark: Int? = nil
        /// Digest of the images among the tokens (`PromptCache.digest`); nil for text.
        var images: String? = nil
        package init(
            tokens: Int, prefix: String, used: Double, kind: String? = nil, name: String? = nil, pinned: Bool? = nil, mark: Int? = nil,
            images: String? = nil
        ) {
            self.tokens = tokens
            self.prefix = prefix
            self.used = used
            self.kind = kind
            self.name = name
            self.pinned = pinned
            self.mark = mark
            self.images = images
        }
        /// Holds exactly these tokens and images (keys of the whole request; those inside the entry count).
        package func holds(_ ids: ArraySlice<Int64>, images keys: [ImageKey]) -> Bool {
            tokens == ids.count && prefix == PromptCache.prefixHash(ids) && images == PromptCache.digest(ImageKey.below(keys, tokens))
        }
    }
    /// `index.json`: entries of this epoch only (another epoch's index is discarded at start).
    struct Index: Codable {
        var epoch: String
        var entries: [String: Entry]
        /// Prompt tokens requests had to read since this cache began.
        var read: Int? = nil
    }
    /// The folder in use: the epoch's own, or this build's after the epoch's folder refused its canary (`settle`).
    public private(set) var root: URL
    /// Parent of the per-process conversation frontier and context-file folders: `root`, or a private temporary folder
    /// while another server holds `root` (`shared`), whose start would remove them. That folder goes at exit, or at a
    /// later start once its process is gone (`temporaryFolder`).
    public let frontierParent: URL
    /// Another server process holds `root` (its `.lock`): this one reads, writes and removes nothing there, nor in the
    /// data folder's other shared files (the usage counters), which only the folder's owner writes.
    public private(set) var shared: Bool
    /// Serializes index writes, so the last one written holds the latest index.
    private let indexWrites = NSLock()
    private var slots: [PersistentCheckpointStore]
    /// Slots 0..<regular hold prompt and conversation images; the rest hold system-prompt snapshots,
    /// so a day of one-off chats cannot evict a harness's system prompt.
    private let regular: Int
    /// Token IDs of recently seen system blocks, to find the part a changing system prompt keeps.
    private var systemBlocks: [[Int]]
    private var index: Index
    private let minimumFreeBytes: UInt64
    private let sameModel: ((URL) -> Bool?)?
    private let epoch: String, arithmetic: String, entryBytes: UInt64, ledger: MemoryLedger, capacity: Int
    /// This executable's SHA-256 when the folder is shared by builds of one arithmetic (`PromptCanary`); nil: not shared.
    private let build: String?
    /// The folder is held closed until `settle` ran this build's canary against it: nothing read, written or removed.
    private var pending: (savesOff: String?, savesNote: String?)?
    /// Entries of sibling folders this build cannot read (`PromptCanary.whyUnreadable`), with why, for `explainRestore`.
    private var unreadable: [(tokens: Int, prefix: String, why: String)] = []
    /// The slot of the last image written, the clone base of the next save.
    private var lastWritten: Int?
    /// Why saving is off (a data folder that cannot be kept private, such as one on exFAT); nil when saves work. The
    /// server still serves, without reuse across restarts; the runtime says so once at start.
    package private(set) var savesOff: String?
    /// Saves are on, on a drive with file permissions that is mounted with ownership ignored (an external APFS or
    /// Mac OS Extended drive, by macOS's default): what the start says once about who can read them.
    package private(set) var savesNote: String?
    /// Saved prompts of this model in folders this build cannot read (`PromptCanary.whyUnreadable`): what the start says once.
    package private(set) var unusableNote: String?

    /// Why prompts are not saved in `root`, a folder with `owner` and `mode` on a drive mounted at `drive` with
    /// `mountFlags` (`statfs`), or nil: saves trust only a folder this user owns and no one else can write (as
    /// `writePaged` checks), on a drive that has permissions. exFAT and FAT have none: they mount with ownership ignored
    /// (`MNT_IGNORE_OWNERSHIP`, `noowners`) and show every folder as the user's own and 0700, while every user of this Mac
    /// can read and write it (ROADMAP R8). There is no setting to save prompts there anyway. A drive with permissions
    /// that is mounted with ownership ignored (`filesystem` apfs or hfs: macOS's default for an external drive) keeps
    /// saving, as before, with `savesNote`.
    package static func savesOff(root: URL, owner: uid_t, mode: mode_t, mountFlags: UInt32, filesystem: String, drive: String) -> String? {
        if mountFlags & UInt32(MNT_IGNORE_OWNERSHIP) != 0, !keepsPermissions(filesystem) {
            return "saved prompts are off on \(drive) because every user of this Mac can read it (the drive has no file "
                + "ownership, as exFAT and FAT do); move the data folder to a drive of yours on APFS or Mac OS Extended "
                + "(origami config set context_dir <folder>) to save them"
        }
        return owner == geteuid() && mode & 0o022 == 0 ? nil : notPrivate(root)
    }
    package static func savesNote(mountFlags: UInt32, filesystem: String, drive: String) -> String? {
        guard mountFlags & UInt32(MNT_IGNORE_OWNERSHIP) != 0, keepsPermissions(filesystem) else { return nil }
        return "saved prompts on \(drive) can be read and changed by every user of this Mac: the drive ignores file ownership "
            + "(\"Ignore ownership on this volume\" in its Get Info window, on by default for an external drive); turn that "
            + "off, or move the data folder (origami config set context_dir <folder>), to keep them private"
    }
    private static func keepsPermissions(_ filesystem: String) -> Bool { ["apfs", "hfs"].contains(filesystem) }
    /// Whether the drive that holds `folder` (or will: its nearest existing parent) has file ownership. False for exFAT
    /// and FAT, where no folder is private: prompts are not saved there (`savesOff`), so the command-line tool keeps them
    /// in the Mac's own data folder unless `context_dir` says otherwise.
    package static func driveKeepsOwnership(_ folder: URL) -> Bool {
        var probe = folder.standardizedFileURL
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 { probe.deleteLastPathComponent() }
        var volume = statfs()
        guard statfs(probe.path, &volume) == 0 else { return true }
        let filesystem = withUnsafeBytes(of: volume.f_fstypename) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        return volume.f_flags & UInt32(MNT_IGNORE_OWNERSHIP) == 0 || keepsPermissions(filesystem)
    }
    private static func notPrivate(_ root: URL) -> String {
        "the data folder \(root.path) is not private to this user (another user's folder, or a drive without file "
            + "permissions), so prompts are not saved; set context_dir to a folder of yours on an APFS or Mac OS Extended "
            + "drive (origami config set context_dir <folder>)"
    }
    private let lock = NSLock()
    /// Existing family entries remain readable while replacements stage, but their metadata cannot change until
    /// publication. Otherwise a concurrent dashboard pin/rename could be acknowledged and then overwritten.
    private var writingStateSlots = Set<Int>()
    /// Deterministic disk-admission fixtures; production always probes the actual filesystem.
    package var freeBytesForChecks: (() -> UInt64)?
    /// Why saves are paused for disk space (the free bytes, the floor and the image), from a save refused for it until one
    /// fits; `log` (the server's) says so once as they pause and once as they resume, the dashboard while it lasts.
    private var paused: String?
    package var log: ((String) -> Void)?
    package var savesPaused: String? { lock.withLock { paused } }

    /// - Parameters:
    ///   - epoch: hash of everything a saved state depends on (engine build, model, settings).
    ///   - entryBytes: per-entry quota (old plus staged image while it is replaced).
    ///   - minimumFreeBytes: a save that would leave less free disk than this evicts old entries first, or is skipped.
    ///   - sameModel: whether a sibling epoch folder is another build of this model (true), another model's (false), or
    ///     cannot say (nil: an older build's folder without a marker). Without it every sibling is this model's, as in
    ///     Qwen's `<fingerprint12>-<context>` folders.
    ///   - build: this executable's SHA-256 when `epoch` is keyed on the arithmetic digest (`PromptCanary.epoch`): the
    ///     folder opens at once when its canary lists this build, else stays closed until `settle`.
    ///   - arithmetic: that digest (its engine's `ArithmeticDigest`), which the folder's marker records.
    public init(
        root: URL, epoch: String, entries: Int, systemEntries: Int = 4, entryBytes: UInt64, ledger: MemoryLedger,
        capacity: Int, minimumFreeBytes: UInt64 = 10_000_000_000, sameModel: ((URL) -> Bool?)? = nil, build: String? = nil,
        arithmetic: String = ArithmeticDigest.qwen.value
    ) throws {
        if let drive = Self.missingDrive(root) {
            throw EngineError(
                "The data folder \(root.path) is on the drive \(drive), which is not connected; connect it or set context_dir to another folder",
                code: "invalid_configuration")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        shared = !Self.hold(self.root)
        Self.removeStaleTemporaryFolders()
        frontierParent = shared ? try Self.temporaryFolder() : self.root
        self.minimumFreeBytes = minimumFreeBytes
        self.sameModel = sameModel
        (self.epoch, self.entryBytes, self.ledger, self.capacity, self.build) = (epoch, entryBytes, ledger, capacity, build)
        self.arithmetic = arithmetic
        regular = max(1, entries)
        index = Index(epoch: epoch, entries: [:])
        systemBlocks = []
        slots = try Self.slots(in: root, count: regular + max(0, systemEntries), entryBytes: entryBytes, ledger: ledger, capacity: capacity)
        var folder = stat()
        var volume = statfs()
        let looked = stat(slots[0].directory.path, &folder) == 0 && statfs(slots[0].directory.path, &volume) == 0
        let drive = withUnsafeBytes(of: volume.f_mntonname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let filesystem = withUnsafeBytes(of: volume.f_fstypename) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        savesOff =
            shared
            ? "another Origami server is using the data folder \(root.path), so this one saves and reuses no prompts and keeps no usage counts; give "
                + "each server its own folder (origami config set context_dir <folder>)"
            : !looked
                ? Self.notPrivate(root)
                : Self.savesOff(
                    root: root, owner: folder.st_uid, mode: folder.st_mode, mountFlags: volume.f_flags, filesystem: filesystem,
                    drive: drive)
        savesNote = looked && savesOff == nil ? Self.savesNote(mountFlags: volume.f_flags, filesystem: filesystem, drive: drive) : nil
        // Frontier and context-file folders of a process that did not exit cleanly hold no live files (they are unlinked).
        // They go now, before this process makes its own beside them (`frontierParent`), never after: a folder held
        // closed for the canary opens (`open`) once the conversation's frontier folder exists.
        if !shared { Self.remove(in: self.root) { $0.hasPrefix("frontiers-") || $0.hasPrefix("kv-") } }
        if let build, savesOff == nil {
            let marker = PromptCanary.marker(self.root)
            let forced = ProcessInfo.processInfo.environment["ORIGAMI_TEST_CANARY_AT_LOAD"] == "1"
            switch PromptCanary.known(self.root, build: build) {
            case .listed where !forced: break
            case .refused(let own) where !forced: try use(own)
            case _ where !PromptCanary.holdsEntries(self.root) && !forced:
                // Nothing to restore: the folder is this build's, its canary written while idle (`canaryWhenIdle`). An earlier
                // build's canary there is dropped with its list: this build has not reproduced it, and keeping it would
                // vouch for saves it did not check; the next upgrade checks the canary this build writes.
                guard PromptCanary.write(.init(arithmetic: arithmetic, canary: nil, builds: [build]), to: self.root) else {
                    savesOff = "the saved prompts' marker could not be written in \(self.root.path)"
                    break
                }
            case _ where marker?.canary == nil && !forced:
                // Saves whose canary their build has not written yet: nothing to check them against.
                canaryNote =
                    "the saved prompts in \(self.root.path) have no canary yet (their build writes it while idle), so this build "
                    + "cannot check them and saves its own"
                try use(PromptCanary.ownFolder(self.root, build: build))
                if savesOff == nil {
                    _ = PromptCanary.write(
                        .init(arithmetic: arithmetic, canary: nil, builds: [build], refused: self.root.lastPathComponent),
                        to: self.root)
                }
            default:
                pending = (savesOff, savesNote)
                savesOff = Self.checking
                return
            }
        }
        try open()
    }
    /// What a cache held closed says while the probe runs (`settle`), in the log, the load progress and `/health`.
    package static let checking = "checking this build against the saved prompts' canary"
    /// The folder this build could not check (`init`), said once at start.
    package private(set) var canaryNote: String?

    /// The slot stores of a folder.
    private static func slots(in root: URL, count: Int, entryBytes: UInt64, ledger: MemoryLedger, capacity: Int) throws
        -> [PersistentCheckpointStore]
    {
        try (0..<count).map { slot in
            try PersistentCheckpointStore(
                directory: root.appendingPathComponent("entry-\(slot)", isDirectory: true),
                maximumBytes: entryBytes, ledger: ledger, capacity: capacity)
        }
    }

    /// Moves to this build's own folder `folder` (the epoch's refused its canary) before anything is read there.
    private func use(_ folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        root = folder.standardizedFileURL.resolvingSymlinksInPath()
        slots = try Self.slots(in: root, count: slots.count, entryBytes: entryBytes, ledger: ledger, capacity: capacity)
        if !Self.hold(root) {
            shared = true
            savesOff =
                "another Origami server is using the data folder \(root.path), so this one saves and reuses no prompts; give "
                + "each server its own folder (origami config set context_dir <folder>)"
        }
    }

    /// Runs this build's canary against a folder held closed (`build`): `probe` returns the SHA-256 of the fixed probe's
    /// bits (`PromptCanary`), run before any saved prompt is read or written. Equal to the folder's: this build joins its
    /// list and the folder opens; a new folder takes this canary. Different: one loud line, nothing of that folder is
    /// read, written or removed, and this build uses its own folder (`PromptCanary.ownFolder`). A probe that fails leaves
    /// saves off for this run. Returns at once when there is nothing to check (the canary lists this build, the folder
    /// is not shared by builds, or saves are off).
    package func settle(probe: () throws -> String) {
        guard let build, let opened = pending else { return }
        log?("saved prompts: \(Self.checking) (a fixed probe, before any saved prompt is read)")
        let started = Date()
        let canary: String
        do { canary = try probe() } catch {
            log?("saved prompts: off for this run: the canary probe failed (\(error))")
            savesOff = "the canary probe failed: \(error)"
            return
        }
        let seconds = String(format: "%.1f", Date().timeIntervalSince(started))
        (savesOff, savesNote) = opened
        pending = nil
        var marker = PromptCanary.marker(root)
        if marker?.canary == nil {
            // A folder without a canary holds nothing this build can vouch for (only `ORIGAMI_TEST_CANARY_AT_LOAD` gets here).
            try? FileManager.default.removeItem(at: root.appendingPathComponent("index.json"))
            marker = PromptCanary.Marker(arithmetic: arithmetic, canary: canary, builds: [])
            log?("saved prompts: this build wrote the folder's canary in \(seconds) s")
        } else if marker!.canary == canary {
            log?("saved prompts: this build was checked against the folder's canary in \(seconds) s")
        } else {
            let refused = root
            log?(
                "saved prompts: WARNING: this build computes different bits than the builds that saved the prompts in "
                    + "\(refused.path) (canary \(marker!.canary!.prefix(12)) there, \(canary.prefix(12)) here, checked in \(seconds) s): "
                    + "nothing is restored from that folder; this build saves its own")
            do { try use(PromptCanary.ownFolder(refused, build: build)) } catch {
                savesOff = "this build's own saved-prompt folder could not be made: \(error)"
                return
            }
            marker = PromptCanary.Marker(
                arithmetic: arithmetic, canary: canary, builds: [], refused: refused.lastPathComponent)
        }
        if !marker!.builds.contains(build) { marker!.builds.append(build) }
        if savesOff == nil, !PromptCanary.write(marker!, to: root) {
            savesOff = "the saved prompts' canary could not be written in \(root.path)"
            return
        }
        try? open()
    }

    /// Seconds without a request before the idle probe runs (`canaryWhenIdle`); settable for checks.
    package var canaryIdle: TimeInterval = 120
    /// Since when no request came, whether the idle probe runs, and whether it has nothing (more) to do this run.
    private var quietSince = Date(), canaryRunning = false, canaryDone = false

    /// Writes the canary of the folder this build made (its marker lists only this build and has none), once something
    /// is saved there and the server has had no request for `canaryIdle`. Called at each look of the server's loop with
    /// whether the engine is idle and whether other work blocks it (a hot copy being built); never while macOS reports
    /// critical memory pressure (`critical`; the warning level is this Mac's ordinary state on a busy day, and the probe
    /// reads into the caches the model already has).
    /// `probe` runs as a request any arriving request cancels (it then throws): tried again at the next idle spell. `wait`
    /// runs it on the caller's thread.
    package func canaryWhenIdle(
        idle: Bool, blocked: Bool, wait: Bool = false, critical: () -> Bool = { MemoryPressureLevel.current() == .critical },
        probe: @escaping () throws -> String
    ) {
        let now = Date()
        let start: Bool = lock.withLock {
            if !idle { quietSince = now }
            guard idle, !blocked, !canaryRunning, !canaryDone, pending == nil, !shared, savesOff == nil, !index.entries.isEmpty,
                now.timeIntervalSince(quietSince) >= canaryIdle
            else { return false }
            canaryRunning = true
            return true
        }
        guard start else { return }
        let finish = { (done: Bool) in self.lock.withLock { (self.canaryRunning, self.canaryDone) = (false, done) } }
        guard let build, !critical() else { return finish(false) }
        guard let marker = PromptCanary.marker(root), marker.canary == nil, marker.refused == nil, marker.builds == [build] else {
            return finish(true)
        }
        nonisolated(unsafe) let me = self  // its state is read and written under its lock only
        let work = {
            let started = Date()
            let canary = Result { try probe() }
            let seconds = String(format: "%.1f", Date().timeIntervalSince(started))
            guard case .success(let value) = canary else {
                me.lock.withLock { me.quietSince = Date() }
                me.log?("saved prompts: the idle canary probe gave way to a request; it runs again at the next idle spell")
                return finish(false)
            }
            // Written only if the folder is still this build's alone and without a canary.
            if var now = PromptCanary.marker(me.root), now.canary == nil, now.builds == [build] {
                now.canary = value
                if PromptCanary.write(now, to: me.root) {
                    me.log?("saved prompts: this build wrote the folder's canary while idle in \(seconds) s")
                }
            }
            finish(true)
        }
        nonisolated(unsafe) let detached = work  // touches the cache only under its lock
        if wait { work() } else { Thread.detachNewThread { detached() } }
    }

    /// Reads the folder's index as this epoch's and applies the start rules (slots, leftovers, other builds' folders).
    private func open() throws {
        let root = self.root
        let epoch = self.epoch
        let indexURL = root.appendingPathComponent("index.json")
        let indexData = shared ? nil : try? Data(contentsOf: indexURL)
        let saved = indexData.flatMap { try? JSONDecoder().decode(Index.self, from: $0) }
        index = Index(epoch: epoch, entries: saved?.epoch == epoch ? saved!.entries : [:])
        systemBlocks =
            (try? Data(contentsOf: root.appendingPathComponent("system-blocks.json"))).flatMap {
                try? JSONDecoder().decode([[Int]].self, from: $0)
            } ?? []
        // An index that cannot be read (a crash or a full drive while it was written) is set aside, and its entries are
        // read back from their slots, which hold their tokens: the folder is this epoch's, and every read verifies its
        // slot again. Names, pins and image digests are lost (an entry holding images then serves no request).
        if indexData != nil, saved == nil {
            let aside = root.appendingPathComponent("index.json.unreadable")
            try? FileManager.default.removeItem(at: aside)
            try? FileManager.default.moveItem(at: indexURL, to: aside)
            for (slot, store) in slots.enumerated() {
                let header = (try? Data(contentsOf: FamilyStateFile.headerURL(store.directory))).flatMap {
                    try? JSONDecoder().decode(FamilyStateFile.Header.self, from: $0)
                }
                let tokens = header.map { $0.epoch == epoch ? $0.tokens : [] } ?? (try? store.savedIDs())?.map(Int.init) ?? []
                guard !tokens.isEmpty else { continue }
                index.entries[String(slot)] = Entry(
                    tokens: tokens.count, prefix: Self.prefixHash(tokens.map(Int64.init)[...]), used: 0,
                    kind: slot >= regular ? "system" : nil)
            }
        }
        // Another number of regular entries (`prompt_cache_entries`) moves the system slots: an entry past the last slot,
        // or in a slot of the other pool, goes (its file below, its folder with the leftovers).
        let regular = self.regular
        let count = slots.count
        index.entries = index.entries.filter { key, entry in
            Int(key).map { (0..<count).contains($0) && (entry.kind == "system") == ($0 >= regular) } ?? false
        }
        guard !shared else { return }
        for (slot, store) in slots.enumerated() {
            if index.entries[String(slot)] == nil { Self.removeSaved(store) }
            // A save interrupted by a kill leaves its staging file (up to one image) until the slot's next save.
            _ = try? store.withStoreLock { try store.removeLeftoverStaging() }
        }
        // Other builds' folders of this model stay for a rollback; those unused for 14 days go.
        for (folder, used) in otherBuilds(lowDisk: false) where Date().timeIntervalSince(used) > 14 * 86_400 {
            try? FileManager.default.removeItem(at: folder)
        }
        unreadable =
            build.map { PromptCanary.unreadableEntries(beside: root, build: $0, arithmetic: arithmetic, sameModel: sameModel) } ?? []
        let kinds = Set(unreadable.map(\.why)).sorted()
        if !kinds.isEmpty {
            unusableNote =
                "\(unreadable.count) saved prompt\(unreadable.count == 1 ? "" : "s") of this model beside \(root.path) cannot be used by "
                + "this build (\(kinds.joined(separator: "; "))): it starts without them; unless kept, they go after 14 days unused"
        }
        // The slot folders past the last slot, from a run with more entries, go (leftover frontiers went at `init`).
        Self.remove(in: root) { $0.hasPrefix("entry-") && Int($0.dropFirst(6)).map { $0 >= count } ?? false }
        saveIndex()
        let entries = slots.indices
        budgetEntry = DiskBudget.shared.add(
            "prompt cache", folder: { root },
            bytes: { [weak self] in self.map { cache in entries.reduce(UInt64(0)) { $0 + cache.diskBytes($1) } } ?? 0 })
    }
    deinit { if let budgetEntry { DiskBudget.shared.remove(budgetEntry) } }
    /// This cache's entry in the drive's account (`DiskBudget`).
    private var budgetEntry: Int?

    package static func prefixHash(_ ids: ArraySlice<Int64>) -> String {
        var hash = SHA256()
        ids.withUnsafeBufferPointer { hash.update(bufferPointer: UnsafeRawBufferPointer($0)) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Digest of the images inside a saved prefix; nil when it holds none, so text entries are unchanged.
    package static func digest(_ keys: [ImageKey]) -> String? {
        keys.isEmpty
            ? nil
            : SHA256.hash(data: Data(keys.map { "\($0.start):\($0.digest)" }.joined(separator: ",").utf8))
                .map { String(format: "%02x", $0) }.joined()
    }

    /// Entries whose tokens (and images) are a prefix of the request's, longest first. Each still needs the
    /// store's full verification when read.
    package func candidates(for ids: [Int64], images: [ImageKey] = []) -> [(slot: Int, store: PersistentCheckpointStore)] {
        guard savesOff == nil else { return [] }
        lock.lock()
        defer { lock.unlock() }
        return index.entries.compactMap { key, entry -> (Int, Int)? in
            guard let slot = Int(key), entry.tokens > 0, entry.tokens <= ids.count, entry.holds(ids.prefix(entry.tokens), images: images)
            else { return nil }
            return (slot, entry.tokens)
        }.sorted { $0.1 > $1.1 }.map { (slot: $0.0, store: slots[$0.0]) }
    }

    /// A cache hit: the entry's last use is written with the next index write, off the request's path (a durable write
    /// takes a few ms). Hits close together share one write; a crash before it loses only the entry's recency.
    package func touch(_ slot: Int) {
        lock.lock()
        let read = index.read ?? 0  // read before the entry is modified: one access to `index` at a time
        index.entries[String(slot)]?.used = Date().timeIntervalSince1970
        index.entries[String(slot)]?.mark = read
        let first = !touched
        touched = true
        lock.unlock()
        guard first else { return }
        nonisolated(unsafe) let me = self  // its index is read and written under its locks only
        DispatchQueue.global(qos: .utility).async {
            me.lock.withLock { me.touched = false }
            me.saveIndex()
        }
    }
    /// A hit's index write is queued (`touch`).
    private var touched = false
    package func noteRead(_ tokens: Int) {
        lock.lock()
        index.read = (index.read ?? 0) + max(0, tokens)
        lock.unlock()
    }

    /// Sibling folders of this model's other engine builds (16 hex characters, `sameModel`), each with when it was last
    /// used; for `lowDisk`, only those known to be this model's (a folder without a marker may be another model's).
    /// Never a running server's, another model's (only `origami cache prune` removes those) or one holding a pinned or
    /// named entry.
    private func otherBuilds(lowDisk: Bool) -> [(URL, Date)] {
        func used(_ folder: URL) -> Date {
            (try? folder.appendingPathComponent("index.json").resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
        }
        return Self.children(root.deletingLastPathComponent()).filter { folder in
            guard folder.standardizedFileURL != root.standardizedFileURL, Self.epoch(folder), !Self.inUse(folder),
                !Self.holdsKept(folder)
            else { return false }
            let same: Bool? = sameModel.map { $0(folder) } ?? true
            return same ?? !lowDisk
        }.map { ($0, used($0)) }
    }
    /// The folder's index names or pins an entry (a kept session, a saved system prompt), or is there but cannot be read
    /// (another build's layout, a damaged file): it may, so only `origami cache prune` removes it.
    private static func holdsKept(_ folder: URL) -> Bool {
        let url = folder.appendingPathComponent("index.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        guard let data = try? Data(contentsOf: url), let index = try? JSONDecoder().decode(Index.self, from: data) else { return true }
        return index.entries.values.contains { $0.pinned == true || $0.name != nil }
    }
    private static func children(_ folder: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
    }
    private static func epoch(_ folder: URL) -> Bool {
        folder.lastPathComponent.count == 16 && folder.lastPathComponent.allSatisfy(\.isHexDigit)
    }
    /// The fingerprint of Qwen's `<fingerprint12>-<context>` folder name; nil for any other name.
    package static func modelFolder(_ name: String) -> Substring? {
        name.range(of: "^[0-9a-f]{12}-[0-9]+$", options: .regularExpression) == nil ? nil : name.prefix(12)
    }
    /// A running server holds the epoch folder, or one of the epoch folders in a Qwen `<fingerprint12>-<context>` (`.lock`).
    package static func inUse(_ folder: URL) -> Bool {
        ([folder] + children(folder).filter(epoch)).contains { epoch in
            let fd = Darwin.open(epoch.appendingPathComponent(".lock").path, O_RDONLY | O_EXLOCK | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
            if fd >= 0 { Darwin.close(fd) }
            return fd < 0 && errno == EWOULDBLOCK
        }
    }
    package func forget(_ slot: Int) {
        lock.lock()
        index.entries.removeValue(forKey: String(slot))
        for n in attempts.indices where attempts[n].slot == slot && attempts[n].fate == .saved { attempts[n].fate = .evicted }
        lock.unlock()
        Self.removeSaved(slots[slot])
        saveIndex()
    }
    /// A slot's saved state of either kind: Qwen's checkpoint, or a family engine's state (`FamilyStateFile`).
    private static func removeSaved(_ store: PersistentCheckpointStore) {
        for url in [store.checkpointURL, FamilyStateFile.headerURL(store.directory), FamilyStateFile.dataURL(store.directory)] {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// The folder a family engine's state for `slot` goes in (`FamilyStateFile`).
    package func stateFolder(_ slot: Int) -> URL { slots[slot].directory }
    /// The longest regular, unpinned prompt or conversation entry that `ids` extends: a chat's earlier end, the first
    /// one a save of its later end replaces when no slot is free (`writeState(replacing:)`).
    package func extended(by ids: [Int64]) -> (slot: Int, tokens: Int)? {
        lock.lock()
        defer { lock.unlock() }
        return index.entries.compactMap { key, entry -> (slot: Int, tokens: Int)? in
            guard let slot = Int(key), slot < regular, entry.kind == nil, entry.pinned != true, entry.tokens < ids.count,
                entry.holds(ids.prefix(entry.tokens), images: [])
            else { return nil }
            return (slot, entry.tokens)
        }.max { $0.tokens < $1.tokens }
    }

    /// Saves a family engine's state for `tokens` as `kind` (as `write` saves a checkpoint): the same slot policy, or
    /// `replacing`'s slot while it is still regular and unpinned, room made for `bytes` first, then `write` fills the
    /// slot's folder.
    package func writeState(tokens: [Int], kind: String? = nil, replacing: Int? = nil, bytes: UInt64, write: (URL) throws -> Void) throws {
        guard savesOff == nil else { return }
        try Self.refuseAfterLostGPU()
        let ids = tokens.map(Int64.init)[...]
        let hash = Self.prefixHash(ids)
        lock.lock()
        let replaced = replacing.flatMap { slot in
            index.entries[String(slot)].flatMap { slot < regular && $0.kind == nil && $0.pinned != true ? slot : nil }
        }
        let chosen = Self.slot(
            for: hash, tokens: ids.count, kind: kind, entries: index.entries, regular: regular, total: slots.count, read: index.read ?? 0)
        // `replacing` (the conversation's earlier, shorter save) is the preferred victim, not an unconditional one: a free
        // slot or this very prompt's entry is taken first, so the earlier save stays while there is room (an edited last
        // message still finds it); with none free it goes before any other conversation's entry.
        let free = chosen.map { slot in index.entries[String(slot)].map { $0.prefix == hash && $0.tokens == ids.count } ?? true }
        guard let slot = free == true ? chosen : replaced ?? chosen else {
            lock.unlock()
            throw EngineError(
                "Every \(kind == "system" ? "system-prompt" : "prompt-cache") slot is pinned, so this state was not saved; `origami forget <name>` releases one"
            )
        }
        let previous = index.entries[String(slot)]
        guard writingStateSlots.insert(slot).inserted else {
            lock.unlock()
            throw EngineError("This prompt-cache slot is already being written", code: "prompt_cache_busy")
        }
        let attempt = attemptLocked(ids, slot: slot)
        lock.unlock()
        defer {
            lock.lock()
            writingStateSlots.remove(slot)
            lock.unlock()
        }
        // FamilyStateFile stages before invalidating the old header. Keep its index and payload available during
        // that work too; a cancelled/failed save must not evict the state it was intended to replace. The staging
        // allowance is additional free space, and this slot cannot be selected as a space-making victim.
        do {
            try slots[slot].withStoreLock {
                // A killed family writer can retain a whole staging payload. Reclaim it before planning free
                // space, or the protected old entry plus that orphan can make every later replacement fail.
                try slots[slot].removeLeftoverStaging()
                try makeRoom(for: bytes, excluding: slot)
                try write(slots[slot].directory)
                // Its buffers were read while a stuck command may still have written them: never published as a hit.
                if GPUWait.lost {
                    Self.removeSaved(slots[slot])
                    try Self.refuseAfterLostGPU()
                }
            }
            pause(nil)
            settle(attempt, slot: slot, nil)
        } catch {
            settle(attempt, slot: slot, error)
            // A failure during the two-rename commit may have removed the publication marker. Such an entry must
            // not keep claiming a hit; failures before commit leave the prior header and index untouched.
            if !FileManager.default.fileExists(atPath: FamilyStateFile.headerURL(slots[slot].directory).path) {
                lock.lock()
                index.entries.removeValue(forKey: String(slot))
                lock.unlock()
                Self.removeSaved(slots[slot])
                saveIndex()
            }
            throw error
        }
        try? FileManager.default.removeItem(at: slots[slot].checkpointURL)
        let same = previous?.prefix == hash
        lock.lock()
        let read = index.read ?? 0
        index.entries[String(slot)] = Entry(
            tokens: ids.count, prefix: hash, used: Date().timeIntervalSince1970, kind: same ? previous?.kind : kind,
            name: same ? previous?.name : nil, pinned: same ? previous?.pinned : nil, mark: read)
        lock.unlock()
        saveIndex()
    }

    /// Regular slots above 8K unless set (`prompt_cache_entries`), and the range a setting may choose: one reply saves
    /// up to three regular images (where its turn starts, its reply, then its prompt, so the prompt survives) beside
    /// the pins of two kept sessions, so three let `keep` still pin the prompt.
    public static let defaultEntries = 3, entryRange = 3...8
    /// Regular slots for a tier: 8K keeps eight (its images are small), larger contexts `entries`.
    package static func regularEntries(context: Int, entries: Int = defaultEntries) -> Int { context <= 8192 ? 8 : entries }
    /// The most disk one saved image takes at `context` (a full context's prompt and logits).
    public static func entryDiskBytes(context: Int) -> UInt64 {
        (try? PagedCheckpointFormat.layout(capacity: context, tokens: context, hasLogits: true).bytes) ?? 0
    }

    /// The slot a new entry of `kind` takes: the same tokens' entry, else a free slot of its pool, else the
    /// pool's unpinned entry with the most prompt tokens read since its last use per token of its own,
    /// so a long conversation outlives many short chats; nil when every one is pinned.
    package static func slot(
        for prefix: String, tokens: Int, images: String? = nil, kind: String?, entries: [String: Entry], regular: Int,
        total: Int, read: Int = 0
    ) -> Int? {
        if let same = entries.first(where: { $0.value.tokens == tokens && $0.value.prefix == prefix && $0.value.images == images })
            .flatMap({ Int($0.key) })
        {
            return same
        }
        let pool = kind == "system" ? regular..<total : 0..<regular
        if let free = pool.first(where: { entries[String($0)] == nil }) { return free }
        func staleness(_ entry: Entry) -> Double { Double(read - (entry.mark ?? 0)) / Double(max(1, entry.tokens)) }
        return entries.filter { pool.contains(Int($0.key) ?? -1) && $0.value.pinned != true }
            .max { (staleness($0.value), -$0.value.used) < (staleness($1.value), -$1.value.used) }.flatMap { Int($0.key) }
    }

    /// Saves `image` as `kind` ("system" for a system-prompt snapshot; nil for prompt and conversation images).
    /// `shouldContinue` false stops the save at its next page (`IOOperationCancelled`), the slot's old image kept.
    package func write(_ image: ModelCheckpoint, kind: String? = nil, shouldContinue: () -> Bool = { true }) throws {
        guard savesOff == nil else { return }
        try Self.refuseAfterLostGPU()
        let ids = image.consumedIDs[...]
        let hash = Self.prefixHash(ids)
        let images = Self.digest(image.images)
        lock.lock()
        guard
            let slot = Self.slot(
                for: hash, tokens: ids.count, images: images, kind: kind, entries: index.entries, regular: regular,
                total: slots.count, read: index.read ?? 0)
        else {
            lock.unlock()
            throw EngineError(
                "Every \(kind == "system" ? "system-prompt" : "prompt-cache") slot is pinned, so this state was not saved; `origami forget <name>` releases one"
            )
        }
        // The entry is hidden while its slot is rewritten; the old image stays on disk until the staged one publishes,
        // so a failed or cancelled save keeps the state it would have replaced (and a same-slot save clones its pages).
        let previous = index.entries.removeValue(forKey: String(slot))
        let attempt = attemptLocked(ids, slot: slot)
        lock.unlock()
        func headIdentity() -> [Int] {
            var info = stat()
            return lstat(slots[slot].checkpointURL.path, &info) == 0 ? [Int(info.st_dev), Int(info.st_ino)] : []
        }
        let head = headIdentity()
        do {
            try makeRoom(
                for: PagedCheckpointFormat.diskBytes(
                    PagedCheckpointFormat.layout(
                        capacity: slots[slot].capacity,
                        tokens: image.frontier.consumedTokenCount, hasLogits: image.nextLogits != nil), at: slots[slot].directory),
                excluding: slot)
            // Clone the image this cache wrote last (usually the same conversation's), so only its changed pages are written.
            lock.lock()
            let base = lastWritten.flatMap { $0 != slot ? slots[$0] : nil }
            lock.unlock()
            try slots[slot].writePaged(image, base: base, shouldContinue: shouldContinue)
            pause(nil)
            settle(attempt, slot: slot, nil)
        } catch {
            settle(attempt, slot: slot, error)
            if let previous, !head.isEmpty, headIdentity() == head {
                lock.lock()
                index.entries[String(slot)] = previous
                lock.unlock()
            }
            throw error
        }
        let same = previous?.prefix == hash
        lock.lock()
        let read = index.read ?? 0
        index.entries[String(slot)] = Entry(
            tokens: ids.count, prefix: hash, used: Date().timeIntervalSince1970, kind: same ? previous?.kind : kind,
            name: same ? previous?.name : nil, pinned: same ? previous?.pinned : nil, mark: read, images: images)
        lastWritten = slot
        lock.unlock()
        saveIndex()
    }

    /// A state read after the GPU was taken for lost (`GPUWait`) may hold what a stuck command wrote: it is not saved.
    private static func refuseAfterLostGPU() throws {
        if GPUWait.lost { throw EngineError("The GPU stopped answering, so this state was not saved", code: "gpu_lost") }
    }

    /// Pins and names the entry holding exactly these tokens (`origami save`); false when there is none.
    package func pin(_ ids: ArraySlice<Int64>, name: String) -> Bool {
        lock.lock()
        guard let key = index.entries.first(where: { $0.value.holds(ids, images: []) })?.key else {
            lock.unlock()
            return false
        }
        index.entries[key]?.pinned = true
        index.entries[key]?.name = name
        lock.unlock()
        saveIndex()
        return true
    }
    /// Pins the entry holding exactly these tokens under `name` for a client that asked to keep its session
    /// (`X-Origami-Keep`); false when there is none.
    package func keep(_ ids: ArraySlice<Int64>, name: String, images: [ImageKey] = []) -> Bool {
        lock.lock()
        guard let key = index.entries.first(where: { $0.value.holds(ids, images: images) })?.key else {
            lock.unlock()
            return false
        }
        index.entries = Self.kept(index.entries, key: key, name: name)
        lock.unlock()
        saveIndex()
        return true
    }
    /// A kept session's name moves to its newest state, and at most `limit` sessions stay pinned (the least recently
    /// used gives way), so they cannot fill the pool that chats share.
    package static func kept(_ entries: [String: Entry], key: String, name: String, limit: Int = 2) -> [String: Entry] {
        var entries = entries
        for (other, entry) in entries where other != key && entry.kind != "system" && entry.name == name {
            entries[other]?.pinned = nil
            entries[other]?.name = nil
        }
        entries[key]?.pinned = true
        entries[key]?.name = name
        let sessions = entries.filter { $0.value.kind != "system" && $0.value.pinned == true }.sorted { $0.value.used > $1.value.used }
        for (other, _) in sessions.dropFirst(limit) where other != key {
            entries[other]?.pinned = nil
            entries[other]?.name = nil
        }
        return entries
    }
    /// Kept sessions: name, tokens, last use.
    public var keptSessions: [(name: String, tokens: Int, used: Date)] {
        lock.lock()
        defer { lock.unlock() }
        return index.entries.values.filter { $0.kind != "system" && $0.pinned == true }
            .map { ($0.name ?? "unnamed", $0.tokens, Date(timeIntervalSince1970: $0.used)) }.sorted { $0.used > $1.used }
    }
    /// Removes system-prompt snapshots and kept sessions by name, or every one (`origami forget`); returns the names removed.
    package func forget(named name: String?) -> [String] {
        lock.lock()
        let doomed = index.entries.filter { name == nil ? ($0.value.kind == "system" || $0.value.pinned == true) : $0.value.name == name }
        lock.unlock()
        for key in doomed.keys.compactMap(Int.init) { forget(key) }
        return doomed.values.map { $0.name ?? "unnamed (\($0.tokens) tokens)" }
    }
    /// System-prompt snapshots: name, tokens, pinned, last use, bytes.
    public var systemSnapshots: [(name: String?, tokens: Int, pinned: Bool, used: Date, bytes: UInt64)] {
        lock.lock()
        let saved = index.entries.filter { $0.value.kind == "system" }
        lock.unlock()
        return saved.compactMap { key, entry in
            Int(key).map { slot in
                (entry.name, entry.tokens, entry.pinned == true, Date(timeIntervalSince1970: entry.used), diskBytes(slot))
            }
        }.sorted { $0.used > $1.used }
    }

    /// An entry holds exactly these tokens and images.
    package func contains(_ ids: ArraySlice<Int64>, images: [ImageKey] = []) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return index.entries.values.contains { $0.holds(ids, images: images) }
    }

    /// Remembers a system block and returns the longest prefix it shares with a different one seen before
    /// (0 when none): a system prompt that changes keeps the part before the change cacheable. Written only while saves are on.
    package func rememberSystemBlock(_ ids: [Int]) -> Int {
        lock.lock()
        let shared = Self.sharedPrefix(ids, among: systemBlocks)
        let changed = !systemBlocks.contains(ids)
        if changed { systemBlocks = Array((systemBlocks + [ids]).suffix(8)) }
        let data = changed && savesOff == nil ? try? JSONEncoder().encode(systemBlocks) : nil
        lock.unlock()
        try? data?.write(to: root.appendingPathComponent("system-blocks.json"), options: .atomic)
        return shared
    }
    package static func sharedPrefix(_ ids: [Int], among blocks: [[Int]]) -> Int {
        blocks.filter { $0 != ids }.map { TokenizerStack.commonTokenPrefix($0, ids) }.max() ?? 0
    }

    /// Low disk: keep at least `minimumFreeBytes` free after the write (a 256K image is 6.8 GB). This model's other builds'
    /// folders go first (never another model's, one without a marker or one holding a kept entry: `otherBuilds`); then the hot copies on this drive give back the room still missing (`DiskBudget`: a save the next request
    /// may need is worth more than a copy's least-used records); then this cache's oldest entries. When dropping every
    /// entry it may drop would not make that room, none is dropped and saves pause (`savesPaused`).
    private func makeRoom(for bytes: UInt64, excluding protectedSlot: Int? = nil) throws {
        let (required, overflow) = minimumFreeBytes.addingReportingOverflow(bytes)
        guard !overflow else { throw EngineError("Prompt-cache disk allowance overflow") }
        var stuck = Set<URL>()  // folders that would not go (read-only, locked or another user's): skipped, never retried
        var asked = false
        while case let free = freeBytes(), free < required {
            // Another build's folder of this model (kept for a rollback) goes before this build's entries.
            if let oldest = otherBuilds(lowDisk: true).filter({ !stuck.contains($0.0) }).min(by: { $0.1 < $1.1 }) {
                if (try? FileManager.default.removeItem(at: oldest.0)) == nil { stuck.insert(oldest.0) }
                continue
            }
            if !asked {
                asked = true
                _ = DiskBudget.shared.makeRoom(bytes, in: root, for: "a prompt-cache save", floor: minimumFreeBytes, free: freeBytes)
                continue
            }
            lock.lock()
            // Unpinned prompt and conversation images go first, system snapshots last.
            let victims = index.entries.filter { $0.value.pinned != true && Int($0.key) != protectedSlot }
            let oldest = victims.min {
                ($0.value.kind == "system" ? 1 : 0, $0.value.used) < ($1.value.kind == "system" ? 1 : 0, $1.value.used)
            }
            .flatMap { Int($0.key) }
            lock.unlock()
            let reclaimable = victims.keys.compactMap(Int.init).reduce(UInt64(0)) { $0 + diskBytes($1) }
            guard let oldest, free + reclaimable >= required else {
                let note = Self.pausedNote(free: free, floor: minimumFreeBytes, image: bytes)
                DiskBudget.shared.skipped(in: root, for: "a prompt-cache save", reason: note)
                pause(note)
                throw EngineError("Too little free disk for the prompt cache at \(root.path): " + note, code: "prompt_cache_disk_space")
            }
            forget(oldest)
        }
    }
    package static func pausedNote(free: UInt64, floor: UInt64, image: UInt64) -> String {
        "conversation saves are paused: \(OrigamiPlan.gigabytes(free)) free, and a save keeps \(OrigamiPlan.gigabytes(floor)) free "
            + "plus its image (\(OrigamiPlan.gigabytes(image))); free disk space or lower the context"
    }
    /// Removes the entries of `folder` whose names `which` picks.
    private static func remove(in folder: URL, _ which: (String) -> Bool) {
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] where which(name) {
            try? FileManager.default.removeItem(at: folder.appendingPathComponent(name))
        }
    }
    /// A conversation save that failed on the file system (a folder gone, a drive error): shown as saves paused in the
    /// status and said once in the log until a save works again, not only in a line of its own each time.
    package func saveFailed(_ what: String) { pause("conversation saves are failing: " + what) }
    private func pause(_ note: String?) {
        let changed = lock.withLock {
            defer { paused = note }
            return (paused == nil) != (note == nil)
        }
        if changed { log?("prompt cache: " + (note ?? "conversation saves resumed")) }
    }
    private func freeBytes() -> UInt64 { freeBytesForChecks?() ?? Self.freeDiskBytes(at: root) }
    /// The drive a folder under /Volumes is on when that drive is not mounted, else nil.
    package static func missingDrive(_ url: URL) -> String? {
        let parts = url.standardizedFileURL.pathComponents
        guard parts.count > 2, parts[1] == "Volumes", !FileManager.default.fileExists(atPath: "/Volumes/" + parts[2]) else { return nil }
        return parts[2]
    }
    /// Free space on the drive that holds `url`, or will (its nearest existing parent).
    package static func freeDiskBytes(at url: URL) -> UInt64 {
        var probe = url.standardizedFileURL
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 { probe.deleteLastPathComponent() }
        let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        // Some volumes (disk images among them) report 0 for important usage while plain free space is there.
        return UInt64(max(0, values?.volumeAvailableCapacityForImportantUsage ?? 0, Int64(values?.volumeAvailableCapacity ?? 0)))
    }

    private func saveIndex() {
        guard !shared, pending == nil else { return }
        indexWrites.lock()
        defer { indexWrites.unlock() }
        lock.lock()
        let data = try? JSONEncoder().encode(index)
        lock.unlock()
        guard let data else { return }
        // As durable as the checkpoints (PagedCheckpointStore): staged, flushed to the drive, renamed over the old index,
        // then the folder flushed, so a crash or power loss leaves the old index or the new one, whole.
        let staging = root.appendingPathComponent(".index.json.staging")
        let fd = Darwin.open(staging.path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { return }
        let written = (try? FileHandle(fileDescriptor: fd, closeOnDealloc: false).write(contentsOf: data)) != nil
        let synced = written && PersistentCheckpointStore.fullSync(fd)
        Darwin.close(fd)
        guard synced, Darwin.rename(staging.path, root.appendingPathComponent("index.json").path) == 0 else {
            unlink(staging.path)
            return
        }
        let folder = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard folder >= 0 else { return }
        _ = PersistentCheckpointStore.fullSync(folder)
        Darwin.close(folder)
    }

    /// Folders this process holds the `.lock` of (flock, exclusive): another server process finds the folder held and
    /// leaves it alone; caches of one process (the checks open several) share it. The descriptor stays open, since
    /// closing it would release the lock, until the process exits.
    nonisolated(unsafe) private static var held = Set<String>()
    private static let heldLock = NSLock()
    private static func hold(_ folder: URL) -> Bool {
        heldLock.withLock {
            guard !held.contains(folder.path) else { return true }
            let fd = Darwin.open(folder.appendingPathComponent(".lock").path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
            // The storage map's probe (`inUse`) takes the lock for a moment: a folder still held after 0.1 s is another server's.
            var locked = fd >= 0 ? flock(fd, LOCK_EX | LOCK_NB) : 0
            for _ in 0..<5 where locked != 0 && errno == EWOULDBLOCK {
                usleep(20_000)
                locked = flock(fd, LOCK_EX | LOCK_NB)
            }
            if fd >= 0, locked != 0 {
                let busy = errno == EWOULDBLOCK
                Darwin.close(fd)
                // A drive without locks serves as before.
                if busy { return false }
            }
            held.insert(folder.path)
            return true
        }
    }

    /// A second server's frontier folder, `<temporary folder>/origami-<UUID>`, held through its `.lock` (flock, taken
    /// atomically as the file is created) until the process ends, and removed at exit.
    private static func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let fd = Darwin.open(
            folder.appendingPathComponent(".lock").path, O_RDWR | O_CREAT | O_EXCL | O_EXLOCK | O_NONBLOCK | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            try? FileManager.default.removeItem(at: folder)
            throw EngineError("Cannot lock the temporary folder \(folder.path): errno \(errno)")
        }
        heldLock.withLock {
            if temporary.isEmpty { atexit { PromptCache.removeTemporaryFolders() } }
            temporary.append(folder)
        }
        return folder
    }
    nonisolated(unsafe) private static var temporary: [URL] = []
    /// This process's temporary folders go (their files are unlinked already). At exit, and for checks.
    package static func removeTemporaryFolders() {
        for folder in heldLock.withLock({ temporary }) { try? FileManager.default.removeItem(at: folder) }
    }
    /// `origami-<UUID>` folders in the temporary folder that are this user's and whose `.lock` no process holds: left by
    /// a second server that was killed. A folder without a lock, or a lock held (a live server's), stays.
    package static func removeStaleTemporaryFolders() {
        let parent = FileManager.default.temporaryDirectory
        for name in (try? FileManager.default.contentsOfDirectory(atPath: parent.path)) ?? []
        where name.hasPrefix("origami-") && UUID(uuidString: String(name.dropFirst(8))) != nil {
            let folder = parent.appendingPathComponent(name, isDirectory: true)
            var info = stat()
            guard lstat(folder.path, &info) == 0, info.st_mode & S_IFMT == S_IFDIR, info.st_uid == geteuid() else { continue }
            let fd = Darwin.open(folder.appendingPathComponent(".lock").path, O_RDONLY | O_EXLOCK | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0 else { continue }
            try? FileManager.default.removeItem(at: folder)
            Darwin.close(fd)
        }
    }

    /// One saved entry as listed: its id (`id(slot:prefix:)`), kind, name, pinned, tokens, bytes on disk, last use.
    public typealias Item = (id: String, kind: String, name: String?, pinned: Bool, tokens: Int, bytes: UInt64, used: Date)
    /// Saved entries, most recently used first.
    public var entries: [Item] {
        lock.lock()
        let saved = index.entries
        lock.unlock()
        return saved.compactMap { key, entry in Int(key).map { item($0, entry) } }.sorted { $0.used > $1.used }
    }
    private func item(_ slot: Int, _ entry: Entry) -> Item {
        (
            Self.id(slot: slot, prefix: entry.prefix), entry.kind ?? "prompt", entry.name, entry.pinned == true, entry.tokens,
            diskBytes(slot),
            Date(timeIntervalSince1970: entry.used)
        )
    }
    /// `prompt_cache` in `/v1/status`, for every engine: the entries and the folder; since load, the requests that restored
    /// fewer prompt tokens than a save could have given, by reason, and the tokens they read again (`explainRestore`).
    public var report: [String: Any] {
        let (reasons, lost) = lock.withLock { (restoreReasons, restoreLostTokens) }
        return [
            "entries": entries.map(Self.wire), "folder": root.path, "saves_paused": savesPaused ?? NSNull(),
            "restore_fallbacks": reasons, "restore_lost_tokens": lost,
        ]
    }

    /// What became of a save this cache was asked for (`explainRestore`).
    private enum SaveFate: String {
        case pending = "save_not_finished"
        case saved
        case skipped = "save_skipped_disk"
        case failed = "save_failed"
        case evicted = "entry_evicted"
    }
    /// The last saves tried (their tokens, slot and fate), the previous request's prompt, and the restore counters.
    private var attempts: [(ids: [Int32], slot: Int, fate: SaveFate)] = []
    private var previousPrompt: [Int32] = []
    private var restoreReasons: [String: Int] = [:], restoreLostTokens = 0, lastRestore: [String: Any]?
    /// A save into `slot` starts: an entry it replaces is evicted. Its index in `attempts` (under `lock`).
    private func attemptLocked(_ ids: ArraySlice<Int64>, slot: Int) -> [Int32] {
        for n in attempts.indices where attempts[n].slot == slot && attempts[n].fate == .saved { attempts[n].fate = .evicted }
        let tokens = ids.map { Int32(truncatingIfNeeded: $0) }
        attempts.append((tokens, slot, .pending))
        if attempts.count > 8 { attempts.removeFirst(attempts.count - 8) }
        return tokens
    }
    private func settle(_ tokens: [Int32], slot: Int, _ error: Error?) {
        lock.withLock {
            // A save put off stays pending (`save_not_finished`) until it runs again.
            guard !(error is IOOperationCancelled),
                let n = attempts.lastIndex(where: { $0.slot == slot && $0.fate == .pending && $0.ids == tokens })
            else { return }
            attempts[n].fate =
                error == nil ? .saved : (error as? EngineError)?.code == "prompt_cache_disk_space" ? .skipped : .failed
        }
    }

    /// A request's restore, once known (`restored` of its `ids` came from saved state, live or on disk): `wanted` is the
    /// longest prefix of it a save tried since load holds, or the previous request's prompt shares; when more was wanted
    /// than restored, why: `no_entry` (that prompt was never saved), `entry_evicted`, `save_skipped_disk` (no room even
    /// after the hot copies gave back), `save_not_finished`, `save_failed`, `prefix_differs_at=K` (the request leaves
    /// the saved tokens at K), `entry_not_used` (whole, but not restored), or `saved_by_other_build` (a save in a folder
    /// this build cannot read; `reason_words` say why: `PromptCanary.whyUnreadable`). For the request's row (`takeRestore`).
    package func explainRestore(_ ids: [Int], restored: Int) {
        let request = ids.map { Int32(truncatingIfNeeded: $0) }
        func common(_ saved: [Int32]) -> Int {
            var n = 0
            let end = min(saved.count, request.count)
            while n < end, saved[n] == request[n] { n += 1 }
            return n
        }
        lock.withLock {
            quietSince = Date()
            var wanted = restored
            var reason: String?
            var words: String?
            for attempt in attempts.reversed() {
                let shared = common(attempt.ids)
                guard shared > wanted else { continue }
                wanted = shared
                // A request that leaves the saved tokens could not have restored them whatever became of the save.
                reason =
                    shared < attempt.ids.count
                    ? "prefix_differs_at=\(shared)" : attempt.fate != .saved ? attempt.fate.rawValue : "entry_not_used"
            }
            let previous = common(previousPrompt)
            if previous > wanted {
                wanted = previous
                reason = "no_entry"
            }
            // A save in a folder this build cannot read (another build's arithmetic, or one from before upgrades kept them).
            for entry in unreadable where entry.tokens > wanted && entry.tokens <= ids.count {
                guard Self.prefixHash(ids.prefix(entry.tokens).map(Int64.init)[...]) == entry.prefix else { continue }
                wanted = entry.tokens
                reason = PromptCanary.savedByOtherBuild
                words = entry.why
            }
            previousPrompt = request
            if let reason {
                restoreReasons[reason.hasPrefix("prefix_differs") ? "prefix_differs" : reason, default: 0] += 1
                restoreLostTokens += wanted - restored
            }
            lastRestore = ["wanted": wanted, "restored": restored, "reason": reason ?? NSNull()]
            if let words { lastRestore?["reason_words"] = words }
        }
    }
    /// The last request's restore (`explainRestore`), once: nil when none was explained since.
    package func takeRestore() -> [String: Any]? {
        lock.withLock {
            defer { lastRestore = nil }
            return lastRestore
        }
    }
    package static func wire(_ item: Item) -> [String: Any] {
        var row: [String: Any] = [
            "id": item.id, "kind": item.kind, "tokens": item.tokens, "bytes": item.bytes, "pinned": item.pinned,
            "last_used": ISO8601DateFormatter().string(from: item.used),
        ]
        if let name = item.name { row["name"] = name }
        return row
    }

    // MARK: the dashboard's saved prompts (`/v1/prompts`)

    /// An entry's id: its slot and the start of its tokens' hash, so an id names one saved prompt and goes stale, rather
    /// than naming another, once its slot is reused.
    package static func id(slot: Int, prefix: String) -> String { "\(slot)-\(prefix.prefix(12))" }
    /// The slot and entry `id` names; under the lock.
    private func entry(_ id: String) throws -> (slot: Int, entry: Entry) {
        guard let slot = id.split(separator: "-", maxSplits: 1).first.flatMap({ Int($0) }), let entry = index.entries[String(slot)],
            Self.id(slot: slot, prefix: entry.prefix) == id
        else {
            throw ServingFailure("No saved prompt has this id (it may have been replaced since); reload the list", status: 404)
        }
        return (slot, entry)
    }
    /// The entry `id` names and its tokens, read from its slot (a family engine's state header, else Qwen's checkpoint)
    /// and checked against the entry's hash, so a save into the slot meanwhile is never shown as this entry.
    package func saved(_ id: String) throws -> (item: Item, tokens: [Int]) {
        lock.lock()
        let found = Result { try entry(id) }
        lock.unlock()
        let (slot, saved) = try found.get()
        let folder = slots[slot].directory
        let tokens =
            try (try? Data(contentsOf: FamilyStateFile.headerURL(folder))).map {
                try JSONDecoder().decode(FamilyStateFile.Header.self, from: $0).tokens
            }
            ?? slots[slot].savedIDs().map(Int.init)
        guard Self.prefixHash(tokens.map(Int64.init)[...]) == saved.prefix else {
            throw ServingFailure("This saved prompt changed while it was read; reload the list", status: 409)
        }
        return (item(slot, saved), tokens)
    }
    /// Renames the entry `id` names (a name no other entry has) and keeps or releases it. A kept prompt or conversation
    /// is a kept session (`kept`: at most two, the least recently used giving way), so they cannot fill the chats' pool.
    package func update(_ id: String, name: String?, keep: Bool?) throws -> Item {
        if let name, !ChatCommand.isName(name) { throw ServingFailure("The saved name " + ChatCommand.nameRule) }
        lock.lock()
        let changed = Result { () throws -> Item in
            let (slot, found) = try self.entry(id)
            let key = String(slot)
            guard !writingStateSlots.contains(slot) else {
                throw ServingFailure("This saved prompt is being replaced; retry after the request finishes", status: 409)
            }
            if let name, index.entries.contains(where: { $0.key != key && $0.value.name == name }) {
                throw ServingFailure("Another saved prompt is named \(name)", status: 409)
            }
            if let name { index.entries[key]?.name = name }
            if keep == false { index.entries[key]?.pinned = nil }
            if keep == true, found.kind == "system" { index.entries[key]?.pinned = true }
            if keep == true, found.kind != "system" {
                index.entries = Self.kept(index.entries, key: key, name: index.entries[key]?.name ?? String(found.prefix.prefix(8)))
            }
            return item(slot, index.entries[key]!)
        }
        lock.unlock()
        let updated = try changed.get()
        saveIndex()
        return updated
    }
    /// Removes the entry `id` names and its files, which are always inside its slot's folder. The caller holds the
    /// generation slot with saves drained, so no request reads or writes the slot meanwhile.
    package func remove(_ id: String) throws {
        lock.lock()
        let found = Result { try entry(id) }
        lock.unlock()
        forget(try found.get().slot)
    }
    /// A saved system prompt's own text, to edit: `saved` (its tokens decoded) without what the chat template writes
    /// around a system message, read from `probe` (a chat of system "\u{2}" and user "\u{1}", decoded): all before the
    /// placeholder, and the longest start of what follows it that `saved` ends with. `saved` as it is when the template
    /// does not match. `whole`: `saved` reaches the template's close of the system message (its first line after the
    /// placeholder); a Qwen save stops at a 256-token boundary, and a shared part ends where two prompts part, so most
    /// hold only the prompt's start.
    package static func systemText(_ saved: String, probe: String) -> (text: String, whole: Bool) {
        guard let mark = probe.firstIndex(of: "\u{2}"), saved.hasPrefix(probe[..<mark]) else { return (saved, true) }
        var text = saved.dropFirst(probe.distance(from: probe.startIndex, to: mark))
        let after = probe[probe.index(after: mark)...].prefix { $0 != "\u{1}" }
        let cut = stride(from: after.count, to: 0, by: -1).first(where: { text.hasSuffix(after.prefix($0)) }) ?? 0
        text = text.dropLast(cut)
        return (String(text), cut >= max(1, after.prefix { $0 != "\n" }.count))
    }

    /// Entries and their total size, for `origami status`.
    public var summary: (entries: Int, bytes: UInt64) {
        lock.lock()
        let used = index.entries.keys.compactMap(Int.init)
        lock.unlock()
        return (used.count, used.reduce(UInt64(0)) { $0 + diskBytes($1) })
    }
    /// Disk an entry takes: its allocated blocks, not the sparse file's capacity-sized length.
    private func diskBytes(_ slot: Int) -> UInt64 {
        [slots[slot].checkpointURL, FamilyStateFile.dataURL(slots[slot].directory)].reduce(UInt64(0)) { sum, url in
            sum + UInt64((try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize ?? 0) ?? 0)
        }
    }
}
