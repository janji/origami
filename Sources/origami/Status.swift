import Foundation
import OrigamiEngine

/// What /v1/status reports beyond the server's own numbers: the facts fixed at start (model, Mac,
/// settings, stores) and, once the model has loaded, live memory, expert-cache and prompt-cache numbers.
final class StatusSource: @unchecked Sendable {
    private let fixed: [String: Any]
    private let lock = NSLock()
    private var runtime: OrigamiRuntime.Loaded?
    private let events: EventLog
    private let settings: Settings, model: ModelSource, host: OrigamiHost, found: [Stores.Found]
    /// Qwen's storage plan, fixed at start: where the experts are read from (`experts_from`).
    private let storage: OrigamiStorage
    private var hotCopyKeeper: HotCopyPlaces?
    /// Qwen's hot copy keeper, once the model has loaded (`HotCopy.qwenKeeper`).
    var keeper: HotCopyPlaces? {
        get { lock.withLock { hotCopyKeeper } }
        set { lock.withLock { hotCopyKeeper = newValue } }
    }

    init(settings: Settings, model: ModelSource, host: OrigamiHost, found: [Stores.Found], storage: OrigamiStorage, events: EventLog) {
        self.events = events
        self.settings = settings
        self.model = model
        self.host = host
        self.found = found
        var stores: [[String: Any]] = found.map { item in
            [
                "path": item.store.path, "drive": item.volume.name, "free_bytes": item.volume.free, "mounted": item.volume.mounted,
                "pack": item.pack != nil, "pack_in_use": item.folder == storage.packFolder,
                "copy_in_use": storage.copyFolders.contains(item.folder),
                "mirror_experts": item.mirror?.records.count ?? item.further,
                "hot_copy_in_use": item.folder == storage.hotCopyFolder?.deletingLastPathComponent(),
                "problems": item.problems,
                "permission": item.problems.contains(Stores.noAnswer),
            ]
        }
        if let calibration = storage.calibration, let index = stores.firstIndex(where: { $0["pack_in_use"] as? Bool == true }) {
            stores[index]["measured_gbps"] = calibration.atFourReaders.pack / 1e9
        }
        self.storage = storage
        fixed = [
            "model": [
                "name": model.displayName, "path": model.directory.path, "fingerprint": model.fingerprint,
                "quantization": model.quantization.rawValue, "bytes": model.totalBytes,
                "experts": ExpertVariant.note(model).map { $0 as Any } ?? NSNull(),
            ],
            "mac": ["gpu": host.gpuName, "memory_bytes": host.physicalMemory, "notes": host.notes],
            "settings": settings.report, "config_file": settings.configFile.path, "stores": stores,
            "reads": Stores.reads(storage),
            "quality": Self.qualityReport(settings),
            "context_plan": OrigamiPlan.contextPlan(
                model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit, expertMemory: settings.expertMemory,
                storage: storage, promptCache: settings.promptCacheRoot
            ).report,
        ]
    }

    /// The Experts card's rows for Qwen, in the family engines' shape (`FamilyStatus.expertsFrom`): the model (read, or
    /// not when a pack is), the pack (`pack`, with the drives it may move to), the
    /// hot copy (its keeper's row, `hotRow`, once the model has loaded) with the drives it may go
    /// to, and the further copies (`more`). Each row's share of reads is filled in by `report` once the model runs.
    static func sources(
        settings: Settings, model: ModelSource, found: [Stores.Found], storage: OrigamiStorage, hot: [[String: Any]] = []
    ) -> [[String: Any]] {
        func row(_ role: String, _ folder: URL, _ bytes: UInt64, _ state: String, _ detail: String? = nil) -> [String: Any] {
            [
                "role": role, "drive": Stores.volume(folder).name, "path": folder.path, "bytes": bytes, "state": state,
                "detail": detail ?? NSNull(), "read_share": NSNull(),
            ]
        }
        let packDrive = storage.packFolder.map { HotCopy.shown(Stores.volume($0)) } ?? ""
        var rows = [
            row(
                "all", model.directory, model.totalBytes, storage.packFolder == nil ? "reading" : "not_read",
                storage.packFolder == nil ? nil : "The 4-bit copy on \(packDrive) is read instead.")
        ]
        // A low-bit variant's experts are read from its experts file; its own folder is links on the data folder's drive.
        if let file = model.ggufExperts?.files.first { rows[0]["drive"] = Stores.driveName(file.path) }
        if let pack = storage.packFolder {
            let bytes = StoreBuilder.packBytes(model)
            var packRow = row("pack", pack, bytes, "reading")
            packRow["bits"] = 4
            packRow["moves"] = HotCopy.choices(
                settings, reads: nil, current: pack, held: bytes, need: bytes, keep: 10_000_000_000, sizes: false)
            rows.append(packRow)
        }
        rows += hot
        for folder in storage.copyFolders where folder != storage.packFolder { rows.append(row("more", folder, 0, "reading")) }
        return rows
    }

    /// The Experts card's Clear for a retired `expert_bits` saved in the settings file or the model's entry (flags and
    /// the environment are not cleared): the option is gone and the model's own experts run.
    static func retiredSettings(_ settings: Settings) -> [[String: String]] {
        guard let value = settings.retiredExpertBits, !["flag", "env"].contains(settings.sources["expert_bits"] ?? "") else { return [] }
        return [
            [
                "key": "expert_bits", "label": "expert_bits " + value,
                "why":
                    "no longer an option; for smaller experts add a published low-bit file: origami model add <experts.gguf> --base <model>",
            ]
        ]
    }

    /// Drives a further copy may go to (`POST v1/hot-copy` `copy`): connected ones with room that hold none of this
    /// model's experts, nor the data folder's (always a store). Without a pack, setup writes the whole pack there, so the
    /// drive needs room for it plus the 10 GB setup keeps free; beside a pack, the most-used experts that fit. `bytes`:
    /// what setup writes, at most.
    static func copyDrives(settings: Settings, model: ModelSource, storage: OrigamiStorage) -> [[String: Any]] {
        guard model.ggufExperts == nil else { return [] }
        let held = Set(
            ([model.directory, settings.dataDir, storage.packFolder, storage.hotCopyFolder].compactMap { $0 } + storage.copyFolders).map {
                Stores.volume($0).disk
            })
        let bytes = StoreBuilder.packBytes(model)
        let drives =
            storage.packFolder == nil
            ? HotCopy.choices(settings, reads: nil, current: nil, held: 0, need: bytes, keep: 10_000_000_000, sizes: false)
            : HotCopy.choices(settings, reads: nil, current: nil, held: 0, sizes: false)
        return drives.filter { row in
            row["status"] as? String == "has room"
                && !held.contains(Stores.volume(URL(fileURLWithPath: row["path"] as? String ?? "/")).disk)
        }.map { $0.merging(["bytes": bytes]) { a, _ in a } }
    }

    /// GET /v1/plan: the memory plan other settings would give, computed without applying them, with how
    /// RAM is split and what each choice is measured or estimated to cost.
    func preview(_ query: [String: String]) throws -> [String: Any] {
        var changed = settings
        func size(_ key: String, off: Bool = false) throws -> UInt64?? {
            guard let text = query[key], !text.isEmpty else { return .none }
            if text == "auto" { return .some(nil) }
            if off, text == "off" { return .some(0) }
            guard let bytes = Settings.parseBytes(text) else {
                throw ServingFailure("\(key) is a size such as 20GB\(off ? ", off" : "") or auto")
            }
            return .some(bytes)
        }
        if let text = query["context"], !text.isEmpty {
            guard let tokens = Settings.parseContext(text), tokens <= OrigamiPlan.maximumContext else {
                throw ServingFailure("context is a token count up to 262144")
            }
            changed.context = tokens
        }
        if case .some(let value) = try size("memory_limit") {
            _ = try Settings.storedValue("memory_limit", query["memory_limit"]!, current: settings)
            changed.memoryLimit = value
        }
        if case .some(let value) = try size("expert_memory") { changed.expertMemory = value }
        if case .some(let value) = try size("hot_copy", off: true) { changed.hotCopy = value }
        if let text = query["context_dir"], !text.isEmpty {
            // A relative path would resolve against the server's working folder (/ under launchd).
            guard text.hasPrefix("/") || text.hasPrefix("~") else {
                throw ServingFailure("context_dir needs a full path, such as /Volumes/Fast/origami")
            }
            changed.contextDir = URL(fileURLWithPath: Settings.expand(text), isDirectory: true)
        }
        if let text = query["quality"], !text.isEmpty, text != settings.quality.name {
            guard let level = QualityDial.level(text) else {
                throw ServingFailure("quality is one of \(QualityDial.offered(.qwen).joined(separator: ", "))")
            }
            changed.quality = level
        }
        // Preview the unified control's result after authorization; it does not change the live system ceiling.
        let currentHost = (try? OrigamiHost.current()) ?? self.host
        let host = OrigamiHost(
            physicalMemory: currentHost.physicalMemory,
            workingSet: max(currentHost.workingSet, changed.memoryLimit ?? 0),
            gpuName: currentHost.gpuName, bfloatMatrices: currentHost.bfloatMatrices, notes: currentHost.notes)
        let storage = Stores.plan(model: model, settings: changed, found: found)
        let chosen: OrigamiPlan
        do {
            chosen = try OrigamiPlan.make(
                model: model, host: host, context: changed.context, memoryLimit: changed.memoryLimit, expertMemory: changed.expertMemory,
                storage: Stores.plan(model: model, settings: settings, found: found),
                promptCache: changed.promptCacheRoot)
        } catch { throw ServingFailure("No plan for these settings: \(error)") }
        let memory = chosen.memory
        let budget = chosen.configuration.experts.cacheBytes
        let perSlab = chosen.configuration.experts.slotsPerSlab
        let record = model.expertRecordBytes
        let held = min(OrigamiPlan.maximumSlots, Int(budget / UInt64(record)) / perSlab * perSlab)
        let context = OrigamiPlan.contextPlan(
            model: model, host: host, context: changed.context, memoryLimit: changed.memoryLimit, expertMemory: changed.expertMemory,
            promptCache: changed.promptCacheRoot)
        // How RAM is split: Origami's fixed parts, the cached experts, the context, and what macOS and other apps keep.
        let experts = UInt64(held * record)
        let contextBytes = OrigamiPlan.contextMemoryBytes(context: context.served, fileBacked: context.paged)
        let split: [[String: Any]] = [
            [
                "part": "fixed", "label": "Origami's fixed parts",
                "bytes": memory.cap > experts + contextBytes ? memory.cap - experts - contextBytes : 0,
            ],
            ["part": "experts", "label": "experts in memory", "bytes": experts],
            ["part": "context", "label": context.paged ? "context (the rest in a file)" : "context", "bytes": contextBytes],
            [
                "part": "others", "label": "macOS and other apps",
                "bytes": host.physicalMemory > memory.cap ? host.physicalMemory - memory.cap : 0,
            ],
        ]
        // The hot copy (built while serving, as `HotCopy.qwenPlace` sizes it) and the drive of the context file and saves.
        let recordBytes = model.hotCopyWholeBytes / UInt64(ModelContract.layers * (model.ggufExperts?.experts ?? ModelContract.experts))
        let hotCopy = changed.hotCopy == 0 ? 0 : changed.hotCopy.map { Int($0 / recordBytes) } ?? StoreBuilder.maximumMirrorRecords
        let hotNow = keeper?.keepers.reduce(0) { $0 + $1.records } ?? 0
        var apply: [String] = []
        func set(_ key: String, _ value: String) { apply.append("origami config set \(key) \(value)") }
        if changed.context != settings.context {
            set("context", changed.context % 1024 == 0 ? "\(changed.context / 1024)k" : "\(changed.context)")
        }
        func gb(_ bytes: UInt64?) -> String { bytes.map { String(format: "%.0fGB", Double($0) / 1e9) } ?? "auto" }
        if changed.memoryLimit != settings.memoryLimit { set("memory_limit", gb(changed.memoryLimit)) }
        if changed.expertMemory != settings.expertMemory { set("expert_memory", gb(changed.expertMemory)) }
        if changed.hotCopy != settings.hotCopy { set("hot_copy", changed.hotCopy.map(Settings.hotCopyName) ?? "auto") }
        if changed.promptCacheRoot.path != settings.promptCacheRoot.path { set("context_dir", changed.promptCacheRoot.path) }
        if changed.quality != settings.quality { set("quality", changed.quality.name) }
        if !apply.isEmpty { apply.append("then restart origami serve") }
        // Drives: what each holds, and where a paged context keeps its file and saves.
        let legacy = Stores.legacyMirror(model: model, settings: settings, found: found)?.folder
        let hotHere = storage.hotCopyFolder?.deletingLastPathComponent()
        func holds(_ item: Stores.Found) -> [String] {
            let pack = item.pack.map { _ in
                "4-bit pack" + (item.folder == storage.packFolder || storage.copyFolders.contains(item.folder) ? " (read)" : "")
            }
            // The scan at start: an offline mirror the keeper has since retired is gone from the drive.
            let mirror = item.mirror.flatMap { header in
                FileManager.default.fileExists(atPath: item.folder.appendingPathComponent(StoreHeader.fileName(.mirror)).path)
                    ? header : nil
            }.map {
                item.folder == legacy && settings.hotCopy != 0
                    ? "offline mirror of an earlier version (not read)"
                    : "copy of \($0.records.count.grouped) most-used experts" + (storage.copyFolders.contains(item.folder) ? " (read)" : "")
            }
            let further =
                item.further > 0
                ? "copy of \(item.further.grouped) most-used experts" + (storage.copyFolders.contains(item.folder) ? " (read)" : "") : nil
            let hot = item.folder == hotHere && hotNow > 0 ? "hot copy of \(hotNow.grouped) experts" : nil
            return [pack, mirror, further, hot].compactMap { $0 }
        }
        var drives: [[String: Any]] = found.map { item in
            ["path": item.store.path, "drive": item.volume.name, "free_bytes": item.volume.free, "holds": holds(item)]
        }
        let contextDrive = Stores.volume(changed.promptCacheRoot)
        drives.append([
            "path": changed.promptCacheRoot.path, "drive": contextDrive.name, "free_bytes": contextDrive.free,
            "holds": [context.paged ? "the context file and saved prompts" : "saved prompts"],
            // The file tier is one 262,144-token geometry whatever the context, so its need is fixed.
            "needs_bytes": context.paged ? 37_000_000_000 : 0,
        ])
        func orAuto(_ bytes: UInt64?) -> Any { bytes.map { $0 as Any } ?? "auto" }
        return [
            "settings": [
                "context": changed.context, "memory_limit": orAuto(changed.memoryLimit),
                "expert_memory": orAuto(changed.expertMemory),
                "hot_copy": changed.hotCopy == 0 ? "off" : orAuto(changed.hotCopy), "context_dir": changed.promptCacheRoot.path,
                "quality": changed.quality.offered(by: .qwen).name,
            ],
            "quality": Self.qualityReport(changed),
            "memory": [
                "cap_bytes": memory.cap, "floor_bytes": memory.floor, "start_bytes": memory.start, "ram_bytes": host.physicalMemory,
                "most_bytes": WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: host.physicalMemory), "split": split,
            ],
            "experts": [
                "cache_experts": held, "cache_bytes": experts,
                "total": OrigamiPlan.maximumSlots, "all_resident": held >= OrigamiPlan.maximumSlots,
                "hot_copy_experts": hotCopy, "hot_copy_now": hotNow,
                "hot_copy_coverage": keeper?.keepers.first?.usage.map { usage in
                    // The size asked (`hot_copy=`), or the automatic one the keeper chose.
                    usage.coverage(changed.hotCopy ?? keeper?.keepers.first?.report["budget_bytes"] as? UInt64 ?? 0) as Any
                } ?? NSNull(),
            ],
            "context": context.report, "notes": chosen.notes, "apply": apply, "drives": drives,
            "impact": Self.impacts(
                held: held, context: context, expertsHeld: held, automaticHeld: automaticHeld(changed, record: record),
                hotCopy: (hotCopy, hotNow), quality: changed.quality.offered(by: .qwen)),
        ]
    }

    /// Experts the cache would hold for `changed` without its expert_memory, for the estimate of what that share costs.
    private func automaticHeld(_ changed: Settings, record: Int) -> Int? {
        guard changed.expertMemory != nil else { return nil }
        var automatic = changed
        automatic.expertMemory = nil
        let plan = try? OrigamiPlan.make(
            model: model, host: host, context: automatic.context, memoryLimit: automatic.memoryLimit,
            storage: Stores.plan(model: model, settings: settings, found: found),
            promptCache: automatic.promptCacheRoot)
        return plan.map { Int($0.configuration.experts.cacheBytes / UInt64(record)) }
    }

    /// Each choice's effect: M2 Pro 32 GB measurements (the README's figures); "estimate" rows are scaled from them.
    static func impacts(
        held: Int, context: OrigamiPlan.ContextPlan, expertsHeld: Int = 0, automaticHeld: Int? = nil,
        hotCopy: (experts: Int, now: Int) = (0, 0), quality: QualityDial = .exact
    ) -> [[String: String]] {
        var rows: [[String: String]] = [
            [
                "choice": "4-bit experts (the model's own)",
                "decode": "7.1 tok/s at 8K, 6.3 at 32K; \(held.grouped) experts in memory here",
                "prompt": "7.7K tokens in 70 s (first token)", "quality": "the model's own replies",
                "basis": "measured on an M2 Pro with 32 GB",
            ]
        ]
        if context.paged {
            let c = Double(context.served)
            rows.append([
                "choice": "\(context.served / 1024)K context, kept in a file on disk",
                "decode": String(format: "about %.1f tok/s (2.6 after 30K, 1.8 at 256K)", 2.6 - 0.8 * max(0, c - 30_000) / 229_200),
                "prompt": String(
                    format: "about %.0f minutes for a full prompt (67 for 256K)", 259_200.0 / 25 * log(80 / (80 - 25 * c / 259_200)) / 60),
                "quality": "unchanged", "basis": "measured on an M2 Pro with 32 GB at 30K and 256K; this size estimated",
            ])
        } else {
            // In memory: 8K and 32K from the 2.0 runs; 64K, 128K and 256K from 46K-, 101K- and 157K-token conversations.
            rows.append([
                "choice": "\(context.served / 1024)K context, in memory",
                "decode": context.tier <= 8192
                    ? "7.1 tok/s"
                    : context.tier <= 32768
                        ? "6.3 tok/s at 32K"
                        : context.tier <= 65536
                            ? "5.2 tok/s at 46K (5.6 at 31K)"
                            : context.tier <= 131072
                                ? "4.5 tok/s at 101K (5.2 at 46K)"
                                : "4.3 tok/s at 157K (4.5 at 101K)",
                "prompt": context.tier <= 32768 ? "about 110 tokens/s" : "about 100 tokens/s (157K tokens in 27 minutes)",
                "quality": "unchanged", "basis": "measured on an M2 Pro with 32 GB",
            ])
        }
        if let automaticHeld, automaticHeld > 0, expertsHeld != automaticHeld {
            // Misses fall as experts^-1.085 (traces); about a third of decode waits for misses on the M2 Pro.
            let ratio = Double(expertsHeld) / Double(automaticHeld)
            let change = 0.35 * (pow(ratio, -1.085) - 1)
            rows.append([
                "choice": "\(expertsHeld.grouped) experts in memory (\(automaticHeld.grouped) without expert_memory)",
                "decode": String(format: "about %.0f%% %@", abs(change) * 100, change > 0 ? "slower" : "faster"),
                "prompt": change > 0 ? "slower when a prompt needs more experts than fit" : "slightly faster",
                "quality": "unchanged", "basis": "estimate (misses fall as experts^-1.085)",
            ])
        } else {
            rows.append([
                "choice": "More memory for Origami", "decode": "about 3–4% faster for each 10% more experts held in memory",
                "prompt": "slightly faster", "quality": "unchanged", "basis": "estimate (misses fall as experts^-1.085)",
            ])
        }
        // The quality dial: each level's measured cost (ballpark cases at 8K, no pack).
        if let measured = qualityCost[quality.name] {
            rows.append([
                "choice": "Quality " + quality.name, "decode": measured.decode, "prompt": "unchanged (prompts are always read exactly)",
                "quality": measured.quality, "basis": measured.basis,
            ])
        } else {
            rows.append([
                "choice": "Quality custom (" + quality.signature + ")", "decode": "faster than exact; not measured as a level",
                "prompt": "unchanged (prompts are always read exactly)", "quality": "replies differ from exact", "basis": "estimate",
            ])
        }
        if hotCopy.experts != hotCopy.now {
            rows.append([
                "choice": hotCopy.experts == 0 ? "No hot copy" : "Hot copy of \(hotCopy.experts.grouped) experts next to the model",
                "decode": hotCopy.experts > hotCopy.now
                    ? "first reads of more experts come from the faster drive" : "more first reads from the slower drive",
                "prompt": hotCopy.experts > hotCopy.now ? "slightly faster on long prompts" : "slightly slower on long prompts",
                "quality": "unchanged", "basis": "estimate; on an M2 Pro with 32 GB one drive was about as fast as two",
            ])
        }
        return rows
    }

    /// The quality dial for /v1/status and /v1/plan: the level the default runs, its knobs, each level `family` offers
    /// with its knobs (and, for Qwen, measured cost), and whether levels other than exact are allowed.
    static func qualityReport(_ settings: Settings, family: ModelFamily = .qwen) -> [String: Any] {
        let level = settings.quality.offered(by: family)
        return [
            "level": level.name, "knobs": level.knobs, "allowed": settings.allowFast,
            "levels": QualityDial.offered(family).map { name -> [String: Any] in
                var row: [String: Any] = ["name": name, "knobs": QualityDial.level(name)!.knobs]
                if family == .qwen, let cost = qualityCost[name] {
                    row["decode"] = cost.decode
                    row["quality"] = cost.quality
                    row["basis"] = cost.basis
                }
                return row
            },
        ]
    }
    /// Each level Qwen offers, its measured effect on an M2 Pro with 32 GB.
    static let qualityCost: [String: (decode: String, quality: String, basis: String)] = [
        "exact": ("the model's own speed", "the model's own replies (bitwise)", "measured on an M2 Pro with 32 GB"),
        "faster": ("about 1.7× decode", "94.8% of next tokens as exact; ballpark answers right", "measured on an M2 Pro with 32 GB"),
        "fastest": (
            "about 2.1× decode", "90% of next tokens as exact; slips on strict instructions (3 of 4 ballpark answers right)",
            "measured on an M2 Pro with 32 GB"
        ),
    ]

    func loaded(_ runtime: OrigamiRuntime.Loaded) {
        lock.lock()
        self.runtime = runtime
        lock.unlock()
    }
    var loadedRuntime: OrigamiRuntime.Loaded? {
        lock.lock()
        defer { lock.unlock() }
        return runtime
    }

    /// The Experts card's hot copy rows, one per place: each keeper's state as the family engines show it
    /// (`HotCopy.rows`), with what the copy means for Qwen; none before the model has loaded or for a model without one.
    /// `fresh`: settings changed since.
    func hotRows(_ fresh: Settings? = nil) -> [[String: Any]] {
        guard let keeper else { return [] }
        let settings = fresh ?? self.settings
        let first = Stores.hotCopyFolder(model: model, settings: settings, found: found) ?? settings.homeStore
        let more = settings.hotCopyMore.map(\.folder)
        let paths = [first] + more.map { StoreHeader.modelFolder(store: $0, model: model).appendingPathComponent("hot", isDirectory: true) }
        return HotCopy.rows(
            keeper, settings: settings, bases: [first] + more, paths: paths, reads: model.ggufExperts == nil ? nil : model.expertsLocation,
            hotShare: nil, depth: 4,
            tokens: loadedRuntime?.backend.routingTokens ?? 0
        ).map {
            var row = $0
            // A copy helps even on the drive all experts are read from (one read each instead of nine): no drive is refused.
            // GGUF experts (three reads each) get one only on another drive (`Stores.plan`).
            row["meaning"] =
                "Hot copy: the experts replies use most, one read each instead of \(model.ggufExperts == nil ? "nine" : "three"), best on a drive other than the one all experts are read from. Replies are unchanged."
            return row
        }
    }

    /// The model row's `slow_reads` (`SlowDriveReads`), and while it is high its `detail` says so and what cures it.
    /// `experts`: a low-bit variant's experts file parts, named with their drive and counted in the time a refresh takes.
    static func slowReads(_ row: inout [String: Any], _ report: [String: Any], bytes: UInt64, experts: [String] = []) {
        row["slow_reads"] = report
        let file = ExpertVariant.filesSummary(experts)
        if let file { row["experts_file"] = ["path": experts[0], "drive": file.drive, "bytes": file.bytes] }
        if let advice = SlowDriveReads.advice(report, modelBytes: bytes + (file?.bytes ?? 0), experts: file?.text) {
            row["detail"] = advice
        }
    }

    /// What a request's reads added to the drive history (`DriveReadLog`): the model's files (`model`, nil while
    /// another copy replaces them) and the hot copy. Once per request, after it ends.
    static func recordReads(_ log: DriveReadLog, model: URL?, reads: SlowDriveReads?, keeper: HotCopyPlaces?) {
        if let model, let reads { log.record(model, reads: reads) }
        for copy in keeper?.keepers.compactMap(\.current) ?? [] { log.record(copy.folder, reads: copy.reads, generation: copy.generation) }
    }

    /// `since_rewrite` on the model and hot copy rows: their files' reading since last written (`DriveReadHistory`).
    static func sinceRewrite(_ sources: inout [[String: Any]], data: URL) {
        let all = DriveReadLog.read(DriveReadLog.file(data))
        for index in sources.indices {
            let role = sources[index]["role"] as? String
            guard role == "all" || role == "hot", let path = sources[index]["path"] as? String,
                let history = all[DriveReadLog.key(URL(fileURLWithPath: path))]
            else { continue }
            sources[index]["since_rewrite"] = history.report(hot: role == "hot")
        }
    }

    func report() -> [String: Any] {
        lock.lock()
        let runtime = self.runtime
        lock.unlock()
        var report = fixed
        report["events"] = events.recent
        report["drive_speeds"] = Calibrations.report(settings)
        var sources = Self.sources(settings: settings, model: model, found: found, storage: storage, hot: hotRows())
        Self.sinceRewrite(&sources, data: settings.dataDir)
        var from: [String: Any] = [
            "released": model.ggufExperts?.typeSummary ?? (model.quantization.experts.fp4 ? "NVFP4" : "4-bit"),
            "split": ["reads": 0, "memory": 0, "hot_copy": NSNull(), "model_drive": 0], "sources": sources,
            "copy_to": Self.copyDrives(settings: settings, model: model, storage: storage),
            "unused_settings": Self.retiredSettings(settings),
            "unused_drives": HotCopy.unusedDrives(
                settings, reads: storage.packFolder == nil ? model.expertsLocation : nil,
                used: [model.expertsLocation, Stores.hotCopyFolder(model: model, settings: settings, found: found) ?? settings.homeStore]
                    + settings.hotCopyMore.map(\.folder), depth: 4),
        ]
        report["experts_from"] = from
        guard let runtime else { return report }
        let memory = runtime.backend.memorySnapshot
        let plan = runtime.plan.memory
        let target = memory.ceiling - memory.transientMargin
        let usable = memory.usableBytes ?? target
        let mode = runtime.plan.configuration.memoryBudgetMode
        report["memory"] = [
            "cap_bytes": plan.cap, "floor_bytes": plan.floor, "start_bytes": plan.start, "reserved_bytes": memory.reserved,
            "budget_bytes": usable, "budget_mode": mode.rawValue, "target_bytes": target,
            "temporarily_reduced": mode == .explicit && usable < target,
            "footprint_bytes": memory.physicalFootprint, "available_bytes": memory.availableMemoryEstimate,
        ]
        report["readiness_preparation"] = runtime.backend.readinessPreparation
        report["memory_admission"] = memoryAdmissionReport(memory)
        if let cache = runtime.backend.expertCacheSnapshot {
            let lookups = cache.hits + cache.misses
            report["expert_cache"] = [
                "slots": cache.activeSlots, "bytes": cache.allocatedBytes, "limit_bytes": cache.byteLimit,
                "hits": cache.hits, "misses": cache.misses, "hit_rate": lookups == 0 ? 0 : Double(cache.hits) / Double(lookups),
            ]
            // The misses split as the store counted its reads: the hot copy, each pack or copy, the model's files.
            let reads = runtime.backend.storeReads
            let missShare = lookups == 0 ? 0 : Double(cache.misses) / Double(lookups)
            let copies = reads.copies.values.reduce(0, +)
            let disk = max(1, reads.hot + copies + reads.model)
            let share = { (count: Int) in missShare * Double(count) / Double(disk) }
            // The hot copy's reads are counted since load: none is shown while no copy is attached (its row says why).
            let hotShare = keeper?.copies.contains { $0 != nil } != true ? 0 : share(reads.hot)
            for index in sources.indices {
                let role = sources[index]["role"] as? String ?? ""
                let path = sources[index]["path"] as? String ?? ""
                sources[index]["read_share"] =
                    role == "hot"
                    ? hotShare * (sources[index]["copy_read_share"] as? Double ?? 1)
                    : role == "all" ? share(reads.model) : share(reads.copies[path] ?? 0)
                if role == "all", storage.packFolder == nil {
                    Self.slowReads(
                        &sources[index], runtime.backend.modelDriveSlowReads, bytes: model.totalBytes,
                        experts: model.ggufExperts?.files.map(\.path) ?? [])
                }
            }
            let hot = keeper == nil ? nil : hotShare
            from["sources"] = sources
            from["split"] = [
                "reads": lookups, "memory": lookups == 0 ? 0 : Double(cache.hits) / Double(lookups),
                "hot_copy": hot.map { $0 as Any } ?? NSNull(),
                "model_drive": missShare - (keeper == nil ? 0 : share(reads.hot)),
            ]
            report["experts_from"] = from
        }
        if let split = runtime.backend.splitReads { report["split_reads"] = split }
        report["template"] = runtime.template
        report["prompt_cache"] = runtime.promptCache.report
        return report
    }
}

/// Whether a read counter moved since the last ask: one ask as each request finishes (requests run one at a time), so
/// it says whether the hot copy served reads during that reply.
final class ReadsSince: @unchecked Sendable {
    private let lock = NSLock()
    private var last = 0
    func moved(_ now: Int) -> Bool {
        lock.withLock {
            defer { last = now }
            return now > last
        }
    }
}

/// The server's last log lines, for the dashboard's events list.
final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [(Date, String)] = []
    func add(_ text: String) {
        lock.lock()
        lines.append((Date(), text))
        if lines.count > 50 { lines.removeFirst() }
        lock.unlock()
    }
    var recent: [[String: String]] {
        lock.lock()
        defer { lock.unlock() }
        let formatter = ISO8601DateFormatter()
        return lines.map { ["time": formatter.string(from: $0.0), "text": $0.1] }
    }
}

extension Int {
    /// "6,536": grouped as the dashboard groups numbers, whatever the server's locale.
    var grouped: String { formatted(.number.locale(Locale(identifier: "en_US"))) }
}
