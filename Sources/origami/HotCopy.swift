import Foundation
import OrigamiEngine

/// Where a family model's hot copy goes (`HotCopyKeeper.Place`), the drives the dashboard's Experts card offers for it,
/// and saving a change from the card (`POST v1/hot-copy`). The card names drives, not folders: Origami's data folder
/// on its own drive, `<volume>/Origami/hot-copy` on any other.
enum HotCopy {
    /// A same-drive copy cannot add another drive's bandwidth, but its contiguous records may
    /// reduce read/dispatch overhead. Keep this experiment explicit; capacity rules still apply.
    static var sameDriveExperiment: Bool {
        ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_SAME_DRIVE_HOT_COPY"] == "1"
    }
    /// The card's name for a drive: "This Mac" for the internal one.
    static func shown(_ volume: Stores.Volume) -> String { volume.name == "internal drive" ? "This Mac" : volume.name }
    /// A drive's name when it is not connected: its folder under /Volumes.
    static func name(_ url: URL) -> String {
        let parts = url.pathComponents
        return parts.count > 2 && parts[1] == "Volumes" ? parts[2] : shown(Stores.volume(url))
    }

    /// The copy's place for the GGUF at `model` (`routed`: its routed experts' bytes), from `hot_copy` and
    /// `hot_copy_dir`. Automatic, it starts at half the routed experts' bytes, at most 40 GB and what leaves 20 GB free, at
    /// least 20 GB, and grows from the usage counts while the curve pays (`HotCopyCurve.automatic`); by default it is possible only off the model's drive and with room for 20 GB (or the size asked, if smaller)
    /// that still leaves 20 GB free. A copy already there counts as room (`heldHere`).
    static func place(_ settings: Settings, model: URL, key: String, routed: UInt64) -> (_ heldHere: (URL) -> UInt64) ->
        HotCopyKeeper.Place
    {
        { heldHere in
            if settings.hotCopy == 0 { return .none(state: "off", detail: "Turned off in the settings.") }
            if let dir = settings.hotCopyDir {
                if !Stores.volume(dir).mounted {
                    return .none(state: "not_connected", detail: "\(name(dir)) is not connected; reading from the model's drive only.")
                }
                if !Stores.answers(dir, within: 3) {
                    return .none(
                        state: "not_allowed", detail: "macOS has not let Origami use \(name(dir)) yet; reading from the model's drive only."
                    )
                }
            }
            let folder = FamilySetup.hotFolder(settings, key: key)
            let identity = LocalModelIdentity.familyRepresentation(source: key)
            let volume = Stores.volume(settings.hotCopyDir ?? settings.dataDir)
            let modelVolume = Stores.volume(model)
            if volume.disk == modelVolume.disk && !sameDriveExperiment {
                return .none(state: "not_possible", detail: "A copy on the model's own drive is disabled by default.")
            }
            let held = heldHere(folder)
            // What idle models' copies on this drive may give counts as room (`IdleHotCopy`, taken as the copy grows).
            let spare = DiskBudget.shared.spare(on: folder, for: loadedModel)
            let free = volume.free + held + spare
            let keep = HotCopyKeeper.keepFree
            // A bounded migration can temporarily hold less than its intended size. Keep its
            // recorded target across restarts without treating unwritten bytes as free capacity.
            let resumed = held > 0 ? OrigamiEngine.HotCopy.open(folder: folder, identity: identity)?.targetBytes ?? held : 0
            let automatic = HotCopyKeeper.Auto(room: free > keep ? free - keep : 0, whole: routed)
            let wanted = settings.hotCopy ?? (held > 0 ? max(held, min(resumed, routed)) : automatic.familyStart)
            // An existing copy is reusable even when another app consumes the remaining free space.
            let least = min(wanted, 20_000_000_000)
            guard let budget = HotCopyRefreshPlan.capacity(wanted: wanted, held: held, free: volume.free + spare, reserve: keep) else {
                return .none(
                    state: "not_possible",
                    detail: "\(shown(volume))'s drive has \(OrigamiPlan.gigabytes(volume.free)) free; "
                        + "the copy needs \(OrigamiPlan.gigabytes(least)) and keeps \(OrigamiPlan.gigabytes(keep)) free.")
            }
            let auto = settings.hotCopy == nil ? automatic : nil
            return .folder(folder, drive: volume.name, budget: budget, free: volume.free, target: wanted, auto: auto)
        }
    }

    /// Qwen's hot copy place, from `hot_copy` and `hot_copy_dir` as `origami setup` placed its mirror: in this model's
    /// folder of the store `Stores.hotCopyFolder` picks (any drive, the model's included: a copy is one read per expert
    /// instead of nine). Automatic, it starts at the shipped ranking's experts (`StoreBuilder.maximumMirrorRecords`) and
    /// grows from the usage counts while the curve pays (`HotCopyCurve.automatic`); at least 512 records, leaving `HotCopyKeeper.keepFree` free. A copy already there, and the offline mirror it replaces, count as room.
    static func qwenPlace(_ settings: Settings, model: ModelSource) -> (_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place {
        { heldHere in
            if settings.hotCopy == 0 { return .none(state: "off", detail: "Turned off in the settings.") }
            if let dir = settings.hotCopyDir {
                if !Stores.volume(dir).mounted {
                    return .none(state: "not_connected", detail: "\(name(dir)) is not connected; reading from the model's drive only.")
                }
                if !Stores.answers(dir, within: 3) {
                    return .none(
                        state: "not_allowed", detail: "macOS has not let Origami use \(name(dir)) yet; reading from the model's drive only."
                    )
                }
            }
            let found = Stores.scan(model: model, settings: settings)
            guard let folder = Stores.plan(model: model, settings: settings, found: found).hotCopyFolder else {
                return .none(
                    state: "not_possible",
                    detail: model.ggufExperts == nil
                        ? "The experts are read from a full copy on the model's own drive; no hot copy."
                        : "The experts file is on this drive already: a copy here would not be faster. Set hot_copy_dir to another drive."
                )
            }
            let legacy = Stores.legacyMirror(model: model, settings: settings, found: found).flatMap { item in
                item.folder.standardizedFileURL == folder.deletingLastPathComponent().standardizedFileURL ? item.mirror : nil
            }
            let volume = Stores.volume(folder)
            let held = heldHere(folder) + (legacy.map { StoreBuilder.mirrorBytes(records: $0.records.count, model) } ?? 0)
            let keep = HotCopyKeeper.keepFree
            // A record in the copy, padded to 16 KiB (GGUF experts: the average, a layer's records have its own size).
            let whole = model.hotCopyWholeBytes
            let record = whole / UInt64(ModelContract.layers * (model.ggufExperts?.experts ?? ModelContract.experts))
            // What idle models' copies on this drive may give counts as room (`IdleHotCopy`, taken as the copy grows).
            let spare = DiskBudget.shared.spare(on: folder, for: loadedModel)
            let room = volume.free + held + spare > keep ? volume.free + held + spare - keep : 0
            // The family copies' rule: falling free space only stops a copy growing; only a lower `hot_copy` shrinks it.
            // Nil (no copy held, under 20 GB of room) leaves Qwen's own least size, 512 records, to decide below.
            let wanted = settings.hotCopy ?? max(heldHere(folder), UInt64(StoreBuilder.maximumMirrorRecords) * record)
            let budget =
                HotCopyRefreshPlan.capacity(wanted: wanted, held: held, free: volume.free + spare, reserve: keep) ?? min(wanted, room)
            guard heldHere(folder) > 0 || budget >= 512 * record else {
                return .none(
                    state: "not_possible",
                    detail: "\(shown(volume))'s drive has \(OrigamiPlan.gigabytes(volume.free)) free; "
                        + "the copy needs \(OrigamiPlan.gigabytes(512 * record)) and keeps \(OrigamiPlan.gigabytes(keep)) free.")
            }
            return .folder(
                folder, drive: volume.name, budget: budget, free: volume.free + held - heldHere(folder),
                auto: settings.hotCopy == nil ? HotCopyKeeper.Auto(room: room, whole: whole) : nil)
        }
    }

    /// Qwen's keepers for the loaded `backend`, one per place (`qwenPlaces`): a finished copy in each is attached at once
    /// (no request runs yet), then each keeper builds, grows and attaches its copy between replies from the routing
    /// counts (`NativeServingBackend.hotCopyCounts`), ranked as `RoutingProfile.ranking` and split over the places fastest
    /// drive first (`HotCopyPlaces`). Records are read for a build as `origami setup` read its mirror's: from a store with
    /// the pack, every read digest-checked; GGUF experts from the GGUF, each record its layer's own size.
    static func qwenKeeper(
        _ settings: Settings, model: ModelSource, pack: URL?, backend: NativeServingBackend, event: @escaping @Sendable (String) -> Void
    ) throws -> HotCopyPlaces? {
        let source = try PackStore(model: model, pack: pack, integrity: .sha256)
        let identity = Stores.hotCopyIdentity(model)
        let places = qwenPlaces(settings, model: model)
        let found = places.map { place -> OrigamiEngine.HotCopy? in
            guard
                case .folder(let folder, _, _, _, _, _) = place({
                    UInt64(OrigamiEngine.HotCopy.open(folder: $0, identity: identity)?.bytes ?? 0)
                })
            else { return nil }
            return OrigamiEngine.HotCopy.open(folder: folder, identity: identity, source: source)
        }
        let speeds = qwenFolders(settings, model: model).map { Calibrations.speed(settings, $0)?.rate(depth: 4) }
        let start = HotCopyPlaces.order(speeds).compactMap { found[$0] }
        let attached = !start.isEmpty && backend.attachHotCopies(start)
        if attached {
            for copy in start {
                event("hot copy: \(copy.keys.count) expert records (\(OrigamiPlan.gigabytes(UInt64(copy.bytes)))) from \(copy.folder.path)")
            }
        }
        let keeper = HotCopyPlaces(
            source: source, identity: identity, places: places.indices.map { (places[$0], attached ? found[$0] : nil) }, speeds: speeds,
            counts: backend.hotCopyCounts,
            choose: { counts, budget in RoutingProfile.hotCopyChoice(counts, budget: budget, record: PackStore.expertBytes) },
            attach: { [weak backend] copies in backend?.attachHotCopies(copies) ?? false },
            idle: { [weak backend] in backend?.hotCopyIdle ?? false },
            withIdle: { [weak backend] work in try backend?.withIdleHotCopy(work) ?? false },
            retire: { folder in
                if let bytes = Stores.retireLegacyMirror(model: model, settings: settings, hotCopy: folder) {
                    event(
                        "Hot copy: removed the offline mirror an earlier version built (\(OrigamiPlan.gigabytes(bytes))); the new copy replaces it."
                    )
                }
            }, event: event)
        backend.hotCopyKeeper = keeper
        keeper.trigger()
        return keeper
    }

    /// Every hot copy place's settings (F10): the first is `hot_copy` and `hot_copy_dir` as they are, then one per
    /// `hot_copy_more` entry, with its folder as `hot_copy_dir` and its size as `hot_copy`, so each place follows the
    /// one-place rules (its drive's reserve, automatic size, not connected, not allowed, the model's own drive).
    static func placeSettings(_ settings: Settings) -> [Settings] {
        [settings]
            + settings.hotCopyMore.map { place in
                var one = settings
                one.hotCopy = place.size
                one.hotCopyDir = place.folder
                one.hotCopyMore = []
                return one
            }
    }
    /// Whether a saved change of the places needs the model restarted (a place added, removed or moved: one keeper per
    /// place); a size changes while serving.
    static func restartNeeded(_ fresh: Settings, started: Settings) -> Bool {
        fresh.hotCopyMore.map(\.folder) != started.hotCopyMore.map(\.folder)
    }
    /// `place` for a further place in `folder`, refused while an earlier place (`earlier`) is on its drive: two places on
    /// one drive would share its bandwidth and its reserve.
    static func alone(_ place: @escaping (_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place, folder: URL, earlier: [URL]) -> (
        _ heldHere: (URL) -> UInt64
    ) -> HotCopyKeeper.Place {
        { held in
            let volume = Stores.volume(folder)
            if volume.mounted, let other = earlier.first(where: { Stores.volume($0).mounted && Stores.volume($0).disk == volume.disk }) {
                return .none(
                    state: "not_possible", detail: "\(name(folder)) already holds the hot copy place \(other.path); one place per drive.")
            }
            return place(held)
        }
    }
    /// The family places' folders, on whose drive each copy goes: `hot_copy_dir` (else the data folder), then `hot_copy_more`'s.
    static func folders(_ settings: Settings) -> [URL] { placeSettings(settings).map { $0.hotCopyDir ?? $0.dataDir } }
    /// Every place of a family model's hot copy (`place` for each of `placeSettings`).
    static func places(_ settings: Settings, model: URL, key: String, routed: UInt64) -> [(_ heldHere: (URL) -> UInt64) ->
        HotCopyKeeper.Place]
    {
        let all = placeSettings(settings)
        let bases = folders(settings)
        return all.indices.map { index in
            let one = place(all[index], model: model, key: key, routed: routed)
            return index == 0 ? one : alone(one, folder: bases[index], earlier: Array(bases[..<index]))
        }
    }
    /// Qwen's places' folders, on whose drive each copy goes: the hot copy's store, then `hot_copy_more`'s.
    static func qwenFolders(_ settings: Settings, model: ModelSource) -> [URL] {
        let first =
            Stores.hotCopyFolder(model: model, settings: settings, found: Stores.scan(model: model, settings: settings))
            ?? settings.homeStore
        return [first] + settings.hotCopyMore.map(\.folder)
    }
    /// Every place of Qwen's hot copy (`qwenPlace` for each of `placeSettings`).
    static func qwenPlaces(_ settings: Settings, model: ModelSource) -> [(_ heldHere: (URL) -> UInt64) -> HotCopyKeeper.Place] {
        let all = placeSettings(settings)
        let bases = qwenFolders(settings, model: model)
        return all.indices.map { index in
            let one = qwenPlace(all[index], model: model)
            return index == 0 ? one : alone(one, folder: bases[index], earlier: Array(bases[..<index]))
        }
    }

    /// The Experts card's rows for every place of `places` (`row` for each, role `hot`): `place` (its index: 0 is
    /// `hot_copy_dir`'s), `rank` (0 holds the most-used experts: the fastest drive), `cap` (its size setting: bytes,
    /// `auto` or `off`), `free_bytes` and `reserve_bytes` on its drive, `speed_bytes_per_second` (measured, at `depth`
    /// readers; null until measured), `copy_read_share` (its share of the bytes read from the copies since each was
    /// attached), `read_share` (that part of `hotShare`, the copies' share of expert reads) and `duplicates` (its records a
    /// faster place holds too: spare room kept for redundancy, `HotCopyPlaces.fit`).
    static func rows(
        _ places: HotCopyPlaces, settings: Settings, bases: [URL], paths: [URL], reads: URL?, hotShare: Double?, depth: Int,
        tokens: Int = 0
    ) -> [[String: Any]] {
        let each = placeSettings(settings)
        let ranking = places.ranking
        let copies = places.keepers.map(\.current)
        // Every place's graph is the one ranking (the fastest place's curve: it excludes nothing), each with where its
        // share begins: the bytes and records the faster places hold, and which drive holds them.
        let global = ranking.first.flatMap { places.keepers[$0].usage }
        let bytes = copies.map { $0?.reads.snapshot().bytes ?? 0 }
        let total = bytes.reduce(0, +)
        let held = copies.compactMap { $0 }.count
        return places.keepers.indices.compactMap { index in
            guard index < each.count, index < bases.count, index < paths.count else { return nil }
            let keeper = places.keepers[index]
            let share = copies[index] == nil ? 0 : total > 0 ? bytes[index] / total : 1 / Double(max(1, held))
            var row = Self.row(
                keeper.report, settings: each[index], base: bases[index], path: paths[index], reads: reads,
                readShare: hotShare.map { $0 * share }, usage: global, tokens: tokens)
            let faster = ranking.prefix { $0 != index }
            let before = faster.reduce(UInt64(0)) { $0 + UInt64(copies[$1]?.bytes ?? 0) }
            if before > 0, let global, var popularity = row["popularity"] as? [String: Any] {
                popularity["before_bytes"] = before
                popularity["before_experts"] = faster.reduce(0) { $0 + (copies[$1]?.keys.count ?? 0) }
                popularity["before"] = faster.compactMap { place -> [String: Any]? in
                    copies[place].map { ["drive": places.keepers[place].report["drive"] ?? NSNull(), "bytes": $0.bytes, "place": place] }
                }
                popularity["duplicate_experts"] = places.duplicates(index)
                row["popularity"] = popularity
                // The share of uses this place adds after the faster ones, for each size.
                row["coverage"] = (row["coverage"] as? [[String: Any]] ?? []).map { entry in
                    let bytes = entry["bytes"] as? UInt64 ?? 0
                    return ["bytes": bytes, "share": global.coverage(before + bytes) - global.coverage(before)] as [String: Any]
                }
            }
            let volume = Stores.volume(bases[index])
            row["place"] = index
            row["place_folder"] = index == 0 ? NSNull() : bases[index].path as Any
            row["rank"] = ranking.firstIndex(of: index) ?? index
            row["cap"] = each[index].hotCopy.map { $0 == 0 ? "off" as Any : $0 as Any } ?? "auto"
            row["free_bytes"] = volume.mounted ? volume.free as Any : NSNull()
            row["reserve_bytes"] = HotCopyKeeper.keepFree
            row["speed_bytes_per_second"] = Calibrations.speed(settings, bases[index])?.rate(depth: depth) ?? NSNull()
            row["copy_read_share"] = share
            row["duplicates"] = places.duplicates(index)
            return row
        }
    }
    /// The drives macOS shows that could hold a copy but no place uses (`used`: the places' folders), for the card to
    /// offer: name, path, the folder a place there would use, free bytes, measured speed (null until measured) and why
    /// one cannot (`why_not`: the model's own drive, too little room above the reserve; null when it can).
    static func unusedDrives(_ settings: Settings, reads: URL?, used: [URL], depth: Int) -> [[String: Any]] {
        let taken = Set(used.filter { Stores.volume($0).mounted }.map { Stores.volume($0).disk })
        let why = ["holds all experts": "the model's own drive", "too small": "too little free space above the reserve"]
        return choices(settings, reads: reads, current: nil, held: 0, sizes: false).filter { row in
            !taken.contains(Stores.volume(URL(fileURLWithPath: row["path"] as? String ?? "/")).disk)
        }.map { row in
            let path = URL(fileURLWithPath: row["path"] as? String ?? "/")
            return [
                "drive": row["drive"] ?? NSNull(), "path": path.path, "free_bytes": row["free_bytes"] ?? 0,
                "folder": placeFolder(forDrive: path, settings: settings).path,
                "speed_bytes_per_second": Calibrations.speed(settings, path)?.rate(depth: depth) ?? NSNull(),
                "why_not": why[row["status"] as? String ?? ""] ?? NSNull(),
            ]
        }
    }

    /// The Experts card's hot copy row for either engine, from its keeper's `report`: the state with its detail in the
    /// card's words, where it may go (`choices`; `reads`, the drive all experts are read from, is offered for none),
    /// whether `Build now` applies, and the share of reads it served. `base` is the folder whose drive it goes on, `path`
    /// its folder while the keeper has none. With the keeper's `usage` curve, `coverage` gives the share of counted expert
    /// uses for the size limit and each size offered, `popularity` the graph of that curve (with `tokens`, the tokens its
    /// counts cover, and what the copy holds now), and `size_note` why an automatic size is what it is.
    static func row(
        _ report: [String: Any], settings: Settings, base: URL, path fallback: URL, reads: URL?, readShare: Double?,
        usage: HotCopyCurve? = nil, tokens: Int = 0
    ) -> [String: Any] {
        func gb(_ bytes: UInt64) -> String { OrigamiPlan.gigabytes(bytes).replacingOccurrences(of: ".0 GB", with: " GB") }
        var row = report
        row["role"] = "hot"
        let state = row["state"] as? String ?? "learning"
        let path = (row["path"] as? String).map { URL(fileURLWithPath: $0) } ?? fallback
        let volume = Stores.volume(base)
        let bytes = UInt64(row["bytes"] as? Int64 ?? 0)
        if row["drive"] is NSNull { row["drive"] = volume.mounted ? volume.name : name(base) }
        row["path"] = path.path
        if row["detail"] is NSNull {
            let budget = row["budget_bytes"] as? UInt64 ?? 0
            let topUp = row["topup_bytes"] as? UInt64 ?? 0
            switch state {
            case "learning": row["detail"] = "Uses up to \(gb(budget)) of \(gb(volume.free + bytes)) free. Replies are unchanged."
            case "building", "paused": row["detail"] = "Pauses during replies, warm-up, and performance comparisons."
            case "ready" where topUp > 0: row["detail"] = "\(gb(topUp)) more to add after this reply"
            default: break
            }
        }
        row["read_share"] = readShare.map { $0 as Any } ?? NSNull()
        row["can_build"] = row["has_counts"] as? Bool == true && ["learning", "ready", "failed"].contains(state)
        let offered = choices(
            settings, reads: reads, current: ["off", "not_possible", "not_connected"].contains(state) ? nil : path, held: bytes)
        row["choices"] = offered
        if let usage {
            let sizes = offered.flatMap { ($0["sizes"] as? [Any] ?? []).compactMap { ($0 as? Int).map(UInt64.init) } }
            row["coverage"] = Set(sizes + [row["budget_bytes"] as? UInt64 ?? 0]).filter { $0 > 0 }.sorted().map {
                ["bytes": $0, "share": usage.coverage($0)] as [String: Any]
            }
            row["popularity"] = usage.popularity().merging(["tokens": tokens, "held_experts": row["records"] ?? 0, "held_bytes": bytes]) {
                $1
            }
        }
        let drive = (row["drive"] as? String).map { $0 == "internal drive" ? "This Mac" : $0 } ?? "its drive"
        row["size_note"] =
            [
                "free_space": "Automatic size: limited by free space on \(drive) (\(gb(HotCopyKeeper.keepFree)) is kept free).",
                "flat": "Automatic size: the usage curve is flat beyond this.",
                "whole_set": "Automatic size: all of the model's experts.",
                "no_counts": "Automatic size: the starting size, until replies are counted.",
            ][row["size_limit"] as? String ?? ""] ?? NSNull()
        return row
    }

    /// `hot_copy_dir` for a copy on the drive of `path`: none (automatic, the data folder) on the data folder's drive,
    /// `/Volumes/<drive>/Origami/hot-copy` on another, the default data folder's `hot-copy` on the boot drive. Qwen's
    /// automatic mirror goes beside the model, so on the data folder's drive Qwen's is named: the data folder's store.
    static func folder(forDrive path: URL, settings: Settings, qwen: Bool = false) -> URL? {
        if Stores.volume(path).disk == Stores.volume(settings.dataDir).disk { return qwen ? settings.homeStore : nil }
        let parts = path.pathComponents
        if parts.count > 2 && parts[1] == "Volumes" {
            return URL(fileURLWithPath: "/Volumes/\(parts[2])/Origami/hot-copy", isDirectory: true)
        }
        return Settings.defaultDataDir.appendingPathComponent("hot-copy", isDirectory: true)
    }
    /// The store a pack or a further copy goes to on the drive of `path`: the data folder's own store on its drive,
    /// `/Volumes/<drive>/Origami/store` on another, the default data folder's store on the boot drive.
    static func store(forDrive path: URL, settings: Settings) -> URL {
        if Stores.volume(path).disk == Stores.volume(settings.dataDir).disk { return settings.homeStore }
        let parts = path.pathComponents
        if parts.count > 2 && parts[1] == "Volumes" { return URL(fileURLWithPath: "/Volumes/\(parts[2])/Origami/store", isDirectory: true) }
        return Settings.defaultDataDir.appendingPathComponent("store", isDirectory: true)
    }

    /// The drives the card offers: the data folder's, then each connected writable drive, one row per disk, with its free
    /// bytes, its word (`holds all experts`: `reads`' drive, `goes here`: `current`'s, with `folder`, the place itself;
    /// `has room` for `need` bytes that leave `keep` free; `too small`) and, with `sizes`, the hot copy sizes that fit.
    static func choices(
        _ settings: Settings, reads: URL?, current: URL?, held: UInt64, need: UInt64 = 20_000_000_000,
        keep: UInt64 = HotCopyKeeper.keepFree, sizes: Bool = true
    ) -> [[String: Any]] {
        let mounted =
            (FileManager.default.mountedVolumeURLs(
                includingResourceValuesForKeys: [.volumeIsReadOnlyKey, .volumeIsLocalKey],
                options: [.skipHiddenVolumes]) ?? [])
            .filter { url in
                let values = try? url.resourceValues(forKeys: [.volumeIsReadOnlyKey, .volumeIsLocalKey])
                return url.path.hasPrefix("/Volumes/") && values?.volumeIsReadOnly != true && values?.volumeIsLocal != false
            }
        let modelDisk = reads.map { Stores.volume($0).disk }
        let here = current.map { Stores.volume($0).disk }
        var disks = Set<String>()
        var out: [[String: Any]] = []
        for root in [settings.dataDir] + mounted {
            let volume = Stores.volume(root)
            guard volume.mounted, disks.insert(volume.disk).inserted else { continue }
            let free = volume.free + (volume.disk == here ? held : 0)
            let room = free > keep ? free - keep : 0
            let status =
                volume.disk == modelDisk && !sameDriveExperiment
                ? "holds all experts" : volume.disk == here ? "goes here" : room >= need ? "has room" : "too small"
            var row: [String: Any] = [
                "drive": volume.name, "path": root.path, "free_bytes": volume.free, "room_bytes": room, "status": status,
            ]
            if volume.disk == here, let current { row["folder"] = current.path }
            if sizes, !["holds all experts", "too small"].contains(status) {
                row["sizes"] =
                    ["auto"]
                    + [20_000_000_000, 40_000_000_000, 80_000_000_000, 120_000_000_000, 160_000_000_000].filter { UInt64($0) <= room }.map {
                        $0 as Any
                    } + ["off"]
            }
            out.append(row)
        }
        return out
    }

    /// Saves a change from the card: `drive` (a drive's path, or `auto`) as `hot_copy_dir`, `size` (`40GB`, `auto`,
    /// `off`) as `hot_copy`. A family model's own drive is refused unless its layout experiment is enabled. The keys saved.
    /// For Qwen (`qwen`), `copy` (a drive's path) adds that drive's store (`store(forDrive:)`) to `stores`, where
    /// `origami setup` writes all experts or, beside a pack elsewhere, a further copy.
    static func save(_ body: [String: Any], settings: Settings, model: URL?, qwen: Bool = false) throws -> [String] {
        var changes: [String: Any?] = [:]
        func connected(_ key: String, in object: [String: Any]? = nil) throws -> URL? {
            guard let drive = (object ?? body)[key] as? String, drive != "auto" || key != "drive" else { return nil }
            guard drive.hasPrefix("/"), !drive.contains("\0"), FileManager.default.fileExists(atPath: drive),
                Stores.volume(URL(fileURLWithPath: drive)).mounted
            else {
                throw ServingFailure(
                    "\(key) must be the full path of a connected drive, such as /Volumes/Fast" + (key == "drive" ? ", or auto" : ""))
            }
            return URL(fileURLWithPath: drive, isDirectory: true)
        }
        if body["drive"] as? String == "auto" {
            changes.updateValue(nil, forKey: "hot_copy_dir")
        } else if let url = try connected("drive") {
            if let model, Stores.volume(model).disk == Stores.volume(url).disk, !sameDriveExperiment {
                throw ServingFailure(onModelDrive, status: 409)
            }
            changes.updateValue(folder(forDrive: url, settings: settings, qwen: qwen)?.path, forKey: "hot_copy_dir")
        }
        if let url = try connected("copy") {
            guard qwen else { throw ServingFailure("Further copies are for Qwen models") }
            let store = store(forDrive: url, settings: settings)
            guard store != settings.homeStore else { throw ServingFailure("Origami always uses its data folder's drive") }
            try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
            let listed = (try Settings.readConfig(settings.configFile)["stores"] as? [[String: Any]]) ?? []
            if !listed.contains(where: { $0["path"] as? String == store.path }) {
                changes.updateValue(listed + [["path": store.path]], forKey: "stores")
            }
        }
        if let size = body["size"] as? String {
            do { changes.updateValue(try Settings.storedValue("hot_copy", size, current: settings), forKey: "hot_copy") } catch let error
                as CLIError
            { throw ServingFailure(error.description) }
        }
        // The places after the first (`hot_copy_more`), in full: each `{"drive": path}` (its folder there, `placeFolder`)
        // or `{"folder": path}`, with `size` (`40GB`, `auto`, `off`; default auto). A place left out keeps its files unless
        // `delete_removed` is true.
        var removed: [URL] = []
        var added: [URL] = []
        if let list = body["places"] as? [[String: Any]] {
            var entries: [[String: Any]] = []
            for entry in list {
                let folder: URL
                if let text = entry["folder"] as? String {
                    guard text.hasPrefix("/"), !text.contains("\0") else { throw ServingFailure("folder must be a full path") }
                    folder = URL(fileURLWithPath: text, isDirectory: true)
                } else if let url = try connected("drive", in: entry) {
                    folder = placeFolder(forDrive: url, settings: settings, qwen: qwen)
                } else {
                    throw ServingFailure("each place is {\"drive\": \"/Volumes/<drive>\"} or {\"folder\": \"<full path>\"}, with a size")
                }
                if let model, Stores.volume(model).disk == Stores.volume(folder).disk, !sameDriveExperiment {
                    throw ServingFailure(onModelDrive, status: 409)
                }
                entries.append(["folder": folder.path, "size": entry["size"] as? String ?? "auto"])
            }
            let text = String(data: try JSONSerialization.data(withJSONObject: entries), encoding: .utf8) ?? "[]"
            do {
                changes.updateValue(try Settings.storedValue("hot_copy_more", text, current: settings), forKey: "hot_copy_more")
            } catch let error as CLIError { throw ServingFailure(error.description, status: 409) }
            let kept = Set(((try? Settings.copyPlaces(text)) ?? []).map(\.folder))
            removed = settings.hotCopyMore.map(\.folder).filter { !kept.contains($0) }
            added = kept.filter { !settings.hotCopyMore.map(\.folder).contains($0) }.sorted { $0.path < $1.path }
        }
        // The copy control's "keep this size when idle" box (`hot_copy_keep`): this model's, or another registered
        // `model`'s (its idle copy's row).
        if let keep = body["keep_when_idle"] as? Bool {
            if let other = body["model"] as? String, other != settings.model {
                guard ModelRegistry.entry(other, settings: settings) != nil else {
                    throw ServingFailure("No model named \(other)", status: 404)
                }
                var target = settings
                target.model = other
                var one: [String: Any?] = ["hot_copy_keep": keep]
                try Commands.modelSettings(&one, target: target)
            } else {
                changes.updateValue(keep, forKey: "hot_copy_keep")
            }
        }
        if !changes.keys.filter({ ["hot_copy_dir", "hot_copy_more", "stores"].contains($0) }).isEmpty { Storage.remember(settings) }
        let saved = changes.keys.sorted()
        // The first place as this save leaves it: `drive` may name a removed place's folder, whose copy stays.
        var after = settings
        if let dir = changes["hot_copy_dir"] { after.hotCopyDir = (dir as? String).map { URL(fileURLWithPath: $0, isDirectory: true) } }
        try Commands.modelSettings(&changes, target: settings)
        if !changes.isEmpty { try Settings.write(changes, to: settings.configFile) }
        if body["delete_removed"] as? Bool == true { _ = removeCopies(removed, settings: after) }
        for folder in added { Calibrations.measurePlace(folder, settings: settings, qwen: qwen, log: OrigamiEngine.HotCopy.log) }
        return saved
    }

    /// The folder a further place uses on the drive of `path` (`folder(forDrive:)`; on the data folder's drive its
    /// `hot-copy` folder, since a further place names its folder).
    static func placeFolder(forDrive path: URL, settings: Settings, qwen: Bool = false) -> URL {
        if Stores.volume(path).disk == Stores.volume(settings.dataDir).disk {
            return settings.dataDir.appendingPathComponent("hot-copy", isDirectory: true)
        }
        return folder(forDrive: path, settings: settings, qwen: qwen)
            ?? settings.dataDir.appendingPathComponent("hot-copy", isDirectory: true)
    }
    /// The configured model's copy in the place `folder`: `family-<key>/hot` for GLM and DeepSeek, `<model folder>/hot`
    /// for Qwen; nil when the model cannot be read now.
    static func copyFolder(_ settings: Settings, place folder: URL) -> URL? {
        var one = settings
        one.hotCopyDir = folder
        if let url = FamilySetup.gguf(settings) {
            return (try? ExpertVariant.source(url, settings: settings)).map { FamilySetup.hotFolder(one, key: FamilySupport.key($0)) }
        }
        return (try? Commands.model(settings)).map {
            StoreHeader.modelFolder(store: folder, model: $0).appendingPathComponent("hot", isDirectory: true)
        }
    }
    /// Deletes the configured model's copy in each of the places `removed` (asked explicitly: `delete_removed`, or
    /// `origami config set hot_copy_more … --yes`); only that `hot` folder. The folders removed.
    static func removeCopies(_ removed: [URL], settings: Settings) -> [URL] {
        // Never the first place's copy, should `hot_copy_dir` now name a removed place's folder.
        removed.filter { $0 != settings.hotCopyDir }.compactMap { copyFolder(settings, place: $0) }.filter {
            FileManager.default.fileExists(atPath: $0.path) && (try? FileManager.default.removeItem(at: $0)) != nil
        }
    }
    static let onModelDrive = "a hot copy on the model's own drive is disabled by default"

    /// The model this server loaded, as the disk budget names it (`HotCopyKeeper.model`), once `shareDrives` ran.
    nonisolated(unsafe) static var loadedModel = ""
    nonisolated(unsafe) private static var idle: [IdleHotCopy] = []
    private static let idleLock = NSLock()
    /// The loaded model's copies (`keeper`, of `identity`) may grow at the cost of idle models' copies on their drives:
    /// every other registered model's places (`idleCopies`) join the disk budget, then the keepers plan again. Off the
    /// caller's thread (a model's drive may be slow to answer).
    static func shareDrives(_ settings: Settings, flags: [String: [String]], identity: String, keeper: HotCopyPlaces) {
        let name = ModelFront.servedID(settings)
        loadedModel = name
        for one in keeper.keepers { one.model = name }
        Thread.detachNewThread {
            let found = idleCopies(settings, flags: flags, except: identity)
            for copy in found { copy.register() }
            idleLock.withLock { idle = found }
            keeper.trigger()
        }
    }
    /// Every hot copy place of each registered model but the one of `identity`, with its counts, its owner's size and
    /// `hot_copy_keep`, and when its model last served (its counts file's time). Models whose files cannot be read now
    /// (a drive not connected) are left out.
    static func idleCopies(_ settings: Settings, flags: [String: [String]], except identity: String) -> [IdleHotCopy] {
        var out: [IdleHotCopy] = []
        for entry in (try? ModelRegistry.entries(settings)) ?? [] {
            guard let name = entry["name"] as? String, let path = entry["path"] as? String,
                Stores.answers(URL(fileURLWithPath: path), within: 3)
            else { continue }
            var named = flags
            named["model"] = [name]
            guard let one = try? Settings.load(flags: named) else { continue }
            let places = placeSettings(one)
            var copy: (identity: String, folders: [URL], counts: (uses: [Int: Int], misses: [Int: Int]), file: URL)?
            if let url = FamilySetup.gguf(one) {
                if let source = try? ExpertVariant.source(url, settings: one) {
                    let key = FamilySupport.key(source)
                    let counted = one.dataDir.appendingPathComponent("family/\(key)", isDirectory: true)
                    copy = (
                        LocalModelIdentity.familyRepresentation(source: key), places.map { FamilySetup.hotFolder($0, key: key) },
                        (
                            OrigamiEngine.HotCopy.loadUses(counted.appendingPathComponent("uses.json")),
                            OrigamiEngine.HotCopy.loadUses(counted.appendingPathComponent("misses.json"))
                        ), counted.appendingPathComponent("uses.json")
                    )
                }
            } else if let model = try? Commands.model(one) {
                let found = Stores.scan(model: model, settings: one)
                let file = one.routingFile(model)
                copy = (
                    Stores.hotCopyIdentity(model), places.compactMap { Stores.hotCopyFolder(model: model, settings: $0, found: found) },
                    RoutingProfile.load(file)?.hotCopyCounts ?? ([:], [:]), file
                )
            }
            guard let copy, copy.identity != identity, !out.contains(where: { $0.identity == copy.identity }) else { continue }
            let used = (try? copy.file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            var earlier: [IdleHotCopy] = []
            for (index, folder) in copy.folders.enumerated() where index < places.count {
                let before = earlier
                let idle = IdleHotCopy(
                    model: name, folder: folder, identity: copy.identity, cap: places[index].hotCopy, keep: one.hotCopyKeep,
                    lastUsed: used, counts: copy.counts, redundant: { before.reduce(into: Set<Int>()) { $0.formUnion($1.keys) } })
                earlier.append(idle)
                if idle.held > 0 { out.append(idle) }
            }
        }
        return out
    }
}
