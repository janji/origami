import Foundation
import OrigamiEngine

/// `/v1/status` and `/v1/plan` for the family engines (GLM-5.3-Flash, DeepSeek-V4.1-Flash), in the fields Qwen's
/// `StatusSource` reports, so the dashboard shows them the same memory strip, dials, drives and saved prompts. Their
/// memory: the dense weights and working buffers (fixed), the conversation state (context; its size at any length
/// from the model's own state parts) and the expert slots (experts), which take what `expert_memory` gives, else what
/// `memory_limit` leaves, else what was free at start.
final class FamilyStatus: @unchecked Sendable {
    private let settings: Settings, model: FamilyModel, backend: FamilyServingBackend, host: OrigamiHost?
    private let url: URL, name: String, family: ModelFamily, key: String
    private let gguf: GGUFSource, geometry: ModelGeometry
    /// Its experts' own types.
    private let released: String
    /// The GGUF's files' bytes.
    private let modelBytes: UInt64
    /// The stores (`stores`, set in the file), which this model never writes to, that hold nothing of Origami's
    /// (another model's pack keeps its store), read once at start: what the Experts card offers to clear.
    private let idleStores: [String]
    private let keeper: HotCopyPlaces?, events: EventLog
    /// The settings as last saved: the hot copy's drive and size change while serving (`update`).
    private let lock = NSLock()
    private var current: Settings

    init(
        settings: Settings, model: FamilyModel, backend: FamilyServingBackend, url: URL, gguf: GGUFSource, geometry: ModelGeometry,
        keeper: HotCopyPlaces? = nil, events: EventLog = EventLog()
    ) {
        self.settings = settings
        self.model = model
        self.backend = backend
        self.url = url
        self.family = geometry.family
        self.key = FamilySupport.key(gguf)
        self.gguf = gguf
        self.geometry = geometry
        self.keeper = keeper
        self.events = events
        current = settings
        host = try? OrigamiHost.current()
        name =
            ModelRegistry.entry(settings.model, settings: settings) != nil ? settings.model! : url.deletingPathExtension().lastPathComponent
        released = Self.released(gguf, geometry)
        modelBytes = gguf.files.reduce(UInt64(0)) {
            $0 + (Models.fileSize(gguf.folder.appendingPathComponent($1)).map { UInt64(max(0, $0)) } ?? 0)
        }
        idleStores =
            settings.sources["stores"] != "file"
            ? []
            : settings.stores.map(\.path).filter { path in
                let store = URL(fileURLWithPath: path, isDirectory: true)
                // A drive that does not answer may hold something: it is kept.
                return store != settings.homeStore && Stores.answers(store, within: 3) && Commands.origamiFolders(in: store).isEmpty
                    && !((try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []).contains { $0.hasPrefix("family-") }
            }
    }

    /// The routed experts' own types, as `IQ2_XXS + IQ1_S`.
    static func released(_ gguf: GGUFSource, _ geometry: ModelGeometry) -> String {
        expertTypes(gguf, geometry).sorted().compactMap { GGUFQuant.names[$0] }.joined(separator: " + ")
    }
    static func expertTypes(_ gguf: GGUFSource, _ geometry: ModelGeometry) -> Set<Int> {
        Set(geometry.moeLayers.flatMap { l in ["gate", "up", "down"].compactMap { gguf.tensors["blk.\(l).ffn_\($0)_exps.weight"]?.type } })
    }
    /// The settings saved since start (the hot copy's drive and size), for the Experts card.
    func update(_ settings: Settings) { lock.withLock { current = settings } }

    /// What the settings mean for a family engine (`settings` in `/v1/status`).
    static let meanings: [String: String] = [
        "hot_copy":
            "size of the hot copy, e.g. 40GB, or off (default: automatic, from 20–40 GB growing with the usage curve, when the copy's drive is not the model's); built while serving after the first reply",
        "hot_copy_dir": "folder for the hot copy, on a drive other than the model's (default: the data folder)",
        "stores": "folders where Qwen's expert copies may be written; GLM and DeepSeek write nothing there",
        "data_dir": "Origami's settings, usage counts, hot copy and saved prompts",
    ]

    private static func gb(_ bytes: UInt64) -> String { OrigamiPlan.gigabytes(bytes).replacingOccurrences(of: ".0 GB", with: " GB") }

    /// Expert records read from the hot copy since start.
    var hotReads: Int { backend.telemetrySnapshot.notes["expert_hot_reads"] as? Int ?? 0 }

    /// Expert lookups since start: from memory, and the rest in the shares of bytes read from the hot copy and the
    /// model's drive (`FamilyExperts.Counters.split`; guessed records included).
    private func split(notes supplied: [String: Any]? = nil) -> (reads: Int, memory: Double, hot: Double, drive: Double) {
        let notes = supplied ?? backend.telemetrySnapshot.notes
        let split = notes["expert_read_split"] as? [String: Double] ?? [:]
        // The hot copy's reads are counted since load: none is shown while no copy is attached (its row says why).
        let hot = keeper?.copies.contains { $0 != nil } != true ? 0 : split["hot_copy"] ?? 0
        return (notes["expert_lookups"] as? Int ?? 0, split["memory"] ?? 0, hot, split["model_drive"] ?? 0)
    }

    /// The Experts card's hot copy row: the first place's (`hot_copy_dir`), as `hotSources` gives it.
    func hotSource(notes: [String: Any]? = nil) -> [String: Any] { hotSources(notes: notes)[0] }
    /// The Experts card's hot copy rows, one per place (`HotCopy.rows`): each keeper's state with its detail in the
    /// card's words, where it may go (`choices`), whether `Build now` applies, and the share of reads it served.
    func hotSources(notes: [String: Any]? = nil) -> [[String: Any]] {
        let settings = lock.withLock { current }
        guard let keeper else {
            return [
                [
                    "role": "hot", "state": "not_possible",
                    "detail": "This model reads all its experts from one file; it has no hot copy yet.",
                ]
            ]
        }
        return HotCopy.rows(
            keeper, settings: settings, bases: HotCopy.folders(settings),
            paths: HotCopy.placeSettings(settings).map { FamilySetup.hotFolder($0, key: key) }, reads: url,
            hotShare: split(notes: notes).hot, depth: geometry.expertsPerToken,
            // Uses are counted per decode token, layer and routed expert.
            tokens: Int(keeper.keepers[0].usage?.counted ?? 0) / max(1, geometry.moeLayers.count * geometry.expertsPerToken))
    }

    /// `experts_from`: where the running model's experts are read from, for the dashboard's Experts card: the model's
    /// drive, then one row per hot copy place (`hot`: the rows given, else `hotSources`), and the drives that could hold a
    /// copy but are not used (`unused_drives`, `HotCopy.unusedDrives`).
    func expertsFrom(hot: [[String: Any]]? = nil, notes: [String: Any]? = nil) -> [String: Any] {
        let reads = split(notes: notes)
        let volume = Stores.volume(url)
        var sources: [[String: Any]] = [
            [
                "role": "all", "drive": volume.name, "path": url.deletingLastPathComponent().path, "bytes": modelBytes,
                "state": "reading", "read_share": reads.drive, "detail": NSNull(),
            ]
        ]
        if let slow = (notes ?? backend.telemetrySnapshot.notes)["model_drive_slow_reads"] as? [String: Any] {
            StatusSource.slowReads(&sources[0], slow, bytes: modelBytes, experts: gguf.expertFiles)
        }
        if keeper != nil { sources += hot ?? hotSources(notes: notes) }
        let settings = lock.withLock { current }
        StatusSource.sinceRewrite(&sources, data: settings.dataDir)
        return [
            "released": released,
            "split": ["reads": reads.reads, "memory": reads.memory, "hot_copy": reads.hot, "model_drive": reads.drive],
            "sources": sources, "unused_settings": unusedSettings(),
            "unused_drives": HotCopy.unusedDrives(
                settings, reads: url, used: keeper == nil ? [] : HotCopy.folders(settings), depth: geometry.expertsPerToken),
        ]
    }

    /// Settings saved in the config file that do nothing for this model, for the card's Clear: a retired `expert_bits`,
    /// and `stores`, where it writes nothing. Flags and environment are not cleared.
    func unusedSettings() -> [[String: String]] {
        let settings = lock.withLock { current }
        var out = StatusSource.retiredSettings(settings)
        // All of them, or none: Clear empties `stores`.
        if !idleStores.isEmpty, idleStores.count == settings.stores.filter({ $0.path != settings.homeStore.path }).count {
            out.append([
                "key": "stores", "label": "Expert folders " + idleStores.joined(separator: ", "),
                "why": "this model writes nothing there",
            ])
        }
        return out
    }

    /// Whether `bytes` of per-position state go to disk: more than a quarter of what Origami may use on this Mac.
    static func contextOnDisk(_ bytes: UInt64) -> Bool { bytes > diskThreshold }
    static var diskThreshold: UInt64 {
        ((try? OrigamiHost.current()).map(OrigamiPlan.memoryCeiling(host:)) ?? ProcessInfo.processInfo.physicalMemory / 2) / 4
    }

    /// Only bounded diagnostic dictionaries from the same published boundary as the rest of the report.
    /// Preserve explicit disabled/refused states; an unsupported model's absent notes remain absent.
    /// Never inspect a live model or expose unrelated internal notes on a status-reader thread.
    static func appendObservationNotes(_ notes: [String: Any], to report: inout [String: Any]) {
        for key in ["prefetch_observations", "resident_precision", "route_trace"] {
            if let value = notes[key] as? [String: Any] { report[key] = value }
        }
    }

    func report() -> [String: Any] {
        let snapshot = backend.telemetrySnapshot
        let notes = snapshot.notes
        let m = snapshot.memory
        let hits = notes["expert_hits"] as? Int ?? 0
        let misses = notes["expert_misses"] as? Int ?? 0
        // The state held now: the rest of the context's is lent to the expert slots (their bytes).
        let state = model.stateBytes(at: snapshot.stateCapacity)
        // The settings as saved since start, with what they mean for this engine.
        let settings = lock.withLock { current }
        var shown: [String: Any] = settings.report.mapValues { $0 as Any }
        for (key, meaning) in Self.meanings {
            if var entry = settings.report[key] {
                entry["meaning"] = meaning
                shown[key] = entry
            }
        }
        let hots = keeper.map { _ in hotSources(notes: notes) }
        let hot = hots?.first
        let modelDrive = Stores.volume(url)
        let hotState = hot?["state"] as? String ?? ""
        let hotDetail = hot?["detail"] as? String
        var stores: [[String: Any]] = [
            [
                "path": url.deletingLastPathComponent().path, "drive": modelDrive.name, "free_bytes": modelDrive.free, "mounted": true,
                "model": true,
                "pack": false, "pack_in_use": false, "copy_in_use": false, "mirror_experts": 0, "mirror_in_use": false,
                "problems": [String](), "permission": false,
            ]
        ]
        if let hot {
            let path = hot["path"] as? String ?? ""
            let records = hot["records"] as? Int ?? 0
            stores.append([
                "path": path, "drive": hot["drive"] ?? "", "free_bytes": Stores.volume(URL(fileURLWithPath: path)).free,
                "mounted": hotState != "not_connected", "pack": false, "pack_in_use": false, "copy_in_use": false,
                "mirror_experts": records, "mirror_in_use": records > 0,
                "problems": ["not_connected", "not_allowed", "failed"].contains(hotState) ? [hotDetail ?? hotState] : [String](),
                "permission": hotState == "not_allowed",
            ])
        }
        var report: [String: Any] = [
            "settings": shown, "config_file": settings.configFile.path, "experts_from": expertsFrom(hot: hots, notes: notes),
            "events": events.recent,
            "model": [
                "name": name, "path": url.path, "family": family.rawValue, "quantization": "gguf", "fingerprint": key,
                "experts": ExpertVariant.note(gguf).map { $0 as Any } ?? NSNull(),
                "context": snapshot.contextLimit,
            ],
            "telemetry": [
                "kind": "boundary_snapshot", "boundary": snapshot.boundary, "sampled_at_unix": snapshot.capturedAt,
                "age_seconds": max(0, Date().timeIntervalSince1970 - snapshot.capturedAt),
            ],
            "scratch": [
                "compaction_enabled": notes["scratch_compaction_enabled"] ?? false,
                "compaction_saved_bytes": notes["scratch_compaction_saved_bytes"] ?? UInt64(0),
                "allocated_bytes": notes["working_allocated_bytes"] ?? UInt64(0),
            ],
            "memory": ["budget_bytes": m.fixed + state + m.experts, "cap_bytes": m.fixed + state + m.experts],
            "expert_cache": [
                "slots": m.slots, "bytes": m.experts, "hits": hits, "misses": misses,
                "hit_rate": hits + misses == 0 ? 0 : Double(hits) / Double(hits + misses),
                "shrank": notes["expert_cache_shrank"] ?? NSNull(),
                "prefetch_issued_by_rank": notes["expert_prefetch_issued_by_rank"] ?? [Int](),
                "prefetch_used_by_rank": notes["expert_prefetch_used_by_rank"] ?? [Int](),
            ],
            "stores": stores,
            "prompt_cache": backend.promptCacheReport().map { $0 as Any } ?? NSNull(),
        ]
        Self.appendObservationNotes(notes, to: &report)
        report["drive_speeds"] = Calibrations.report(settings)
        if let host { report["mac"] = ["gpu": host.gpuName, "memory_bytes": host.physicalMemory, "notes": host.notes] }
        return report
    }

    /// The plan another setting would give, as `StatusSource.preview` answers for Qwen: the memory split, the experts
    /// held, the context and the drives. Estimates past the running setting scale the state and the expert slots.
    func preview(_ query: [String: String]) throws -> [String: Any] {
        var changed = settings
        func size(_ key: String) throws -> UInt64?? {
            guard let text = query[key], !text.isEmpty else { return .none }
            if text == "auto" { return .some(nil) }
            guard let bytes = Settings.parseBytes(text) else { throw ServingFailure("\(key) is a size such as 20GB or auto") }
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
        if let text = query["quality"], !text.isEmpty, text != settings.quality.name {
            guard let level = QualityDial.level(text) else {
                throw ServingFailure("quality is one of \(QualityDial.offered(family).joined(separator: ", "))")
            }
            changed.quality = level
        }
        let snapshot = backend.telemetrySnapshot
        let m = snapshot.memory
        let ram = host?.physicalMemory ?? ProcessInfo.processInfo.physicalMemory
        let context = min(changed.context, m.maximumContext)
        // The state in memory: all of it, or without its per-position part when that goes to disk.
        let fixedState = model.stateBytes(at: 0)
        func inMemory(_ tokens: Int) -> (bytes: UInt64, paged: Bool) {
            let all = model.stateBytes(at: tokens)
            return (settings.contextOnDisk ?? Self.contextOnDisk(all - fixedState)) ? (fixedState, true) : (all, false)
        }
        let stateNow = snapshot.contextOnDisk ? fixedState : model.stateBytes(at: snapshot.contextLimit)
        let (state, paged) = context == snapshot.contextLimit ? (stateNow, snapshot.contextOnDisk) : inMemory(context)
        // The slots hold the state lent to them now; the plan counts the whole context's state.
        let heldNow = snapshot.contextOnDisk ? fixedState : model.stateBytes(at: snapshot.stateCapacity)
        // The longest context whose per-position state stays in memory here.
        let perToken = Double(model.stateBytes(at: snapshot.contextLimit) - fixedState) / Double(max(1, snapshot.contextLimit))
        let inMemoryUpTo = min(m.maximumContext, perToken > 0 ? Int(Double(Self.diskThreshold) / perToken) : m.maximumContext)
        // Automatic: what was free at start, with the context's difference moved to or from the expert slots.
        let automatic = m.experts + heldNow > state ? m.experts + heldNow - state : 0
        let experts = changed.expertMemory ?? changed.memoryLimit.map { $0 > m.fixed + state ? $0 - m.fixed - state : 0 } ?? automatic
        // A record's size: the running slots' mean.
        let record = m.slots > 0 ? max(1, m.experts / UInt64(m.slots)) : 1
        let held = min(m.totalExperts, Int(experts / record))
        let cap = m.fixed + state + experts
        var notes: [String] = []
        if changed.context > m.maximumContext {
            notes.append("\(family.displayName) takes up to \(m.maximumContext / 1024)K tokens of context")
        }
        if held < 2 * snapshot.routedExperts { notes.append("that leaves too little memory for the experts: this model would not start") }
        let split: [[String: Any]] = [
            ["part": "fixed", "label": "Origami", "bytes": m.fixed], ["part": "experts", "label": "Experts held", "bytes": experts],
            ["part": "context", "label": "Context", "bytes": state],
            ["part": "others", "label": "macOS and other apps", "bytes": ram > cap ? ram - cap : 0],
        ]
        func orAuto(_ bytes: UInt64?) -> Any { bytes.map { $0 as Any } ?? "auto" }
        // The model's drive holds all its experts, the hot copy's drive the copy.
        let hotNow = keeper?.keepers.first?.report
        // The size asked (`hot_copy=`), or the automatic one the keeper chose, for the share of uses it would cover.
        let hotAsked = changed.hotCopy ?? hotNow?["budget_bytes"] as? UInt64 ?? 0
        var modelDrives: [[String: Any]] = [
            [
                "path": url.deletingLastPathComponent().path, "drive": Stores.volume(url).name,
                "free_bytes": Stores.volume(url).free, "holds": ["all experts"],
            ]
        ]
        if let hotNow, let path = hotNow["path"] as? String, (hotNow["records"] as? Int ?? 0) > 0 {
            modelDrives.append([
                "path": path, "drive": hotNow["drive"] ?? "", "free_bytes": Stores.volume(URL(fileURLWithPath: path)).free,
                "holds": ["hot copy"],
            ])
        }
        return [
            "settings": [
                "context": changed.context,
                "memory_limit": orAuto(changed.memoryLimit), "expert_memory": orAuto(changed.expertMemory),
                "hot_copy": changed.hotCopy == 0 ? "off" : orAuto(changed.hotCopy), "context_dir": changed.promptCacheRoot.path,
                "quality": changed.quality.offered(by: family).name,
            ],
            "quality": StatusSource.qualityReport(changed, family: family),
            "memory": [
                "cap_bytes": cap, "floor_bytes": ram > cap ? ram - cap : 0, "start_bytes": cap, "ram_bytes": ram,
                "most_bytes": WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ram), "split": split,
            ],
            "experts": [
                "cache_experts": held, "cache_bytes": experts, "total": m.totalExperts, "all_resident": held >= m.totalExperts,
                "released": released,
                "estimated": changed.context != settings.context || changed.memoryLimit != settings.memoryLimit
                    || changed.expertMemory != settings.expertMemory,
            ],
            "context": ["tier": context, "served": context, "in_memory_up_to": inMemoryUpTo, "limit": m.maximumContext, "paged": paged],
            "notes": notes,
            "apply": [String](),
            "hot_copy": [
                "bytes": hotNow?["bytes"] ?? 0, "drive": hotNow?["drive"] ?? NSNull(), "state": hotNow?["state"] ?? "not_possible",
                "asked_bytes": hotAsked, "coverage": keeper?.keepers.first?.usage.map { $0.coverage(hotAsked) as Any } ?? NSNull(),
            ],
            "drives": modelDrives + [
                [
                    "path": changed.promptCacheRoot.path, "drive": Stores.volume(changed.promptCacheRoot).name,
                    "free_bytes": Stores.volume(changed.promptCacheRoot).free, "holds": ["saved prompts"], "needs_bytes": 0,
                ]
            ],
        ]
    }
}
