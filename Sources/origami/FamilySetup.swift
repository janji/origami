import Foundation
import OrigamiEngine

/// The family engines' (GLM-5.3-Flash, DeepSeek-V4.1-Flash) setup, drives, hot copy folder and `doctor` rows. They
/// serve their GGUF's own experts (or a low-bit variant's, `model add --base`).
enum FamilySetup {
    /// The configured model when it is a GLM or DeepSeek GGUF.
    static func gguf(_ settings: Settings) -> URL? {
        guard let spec = settings.model else { return nil }
        let path = ModelRegistry.entry(spec, settings: settings)?["path"] as? String ?? Settings.expand(spec)
        guard Models.isGGUF(path), let source = try? GGUFSource(URL(fileURLWithPath: path)),
            let family = ModelFamily.detect(gguf: source.metadata), family != .qwen
        else { return nil }
        return URL(fileURLWithPath: path)
    }
    /// The GGUF the `model` setting names (a registered name or a path), as `serve` finds it, unless it is a Qwen one: by
    /// its path alone, so a GGUF that does not open (a part missing) is still found and its problem told (`doctor`, `status`).
    static func named(_ settings: Settings) -> URL? {
        guard let spec = settings.model else { return nil }
        let path = ModelRegistry.entry(spec, settings: settings)?["path"] as? String ?? Settings.expand(spec)
        guard path.lowercased().hasSuffix(".gguf") || Models.isGGUF(path) else { return nil }
        if let source = try? GGUFSource(URL(fileURLWithPath: path)), ModelFamily.detect(gguf: source.metadata) == .qwen { return nil }
        return URL(fileURLWithPath: path)
    }
    static func stores(_ settings: Settings) -> [URL] {
        var stores = settings.stores.map { URL(fileURLWithPath: $0.path, isDirectory: true) }
        if !stores.contains(settings.homeStore) { stores.append(settings.homeStore) }
        return stores
    }
    /// The hot copy's folder for the GGUF keyed `key`: `<hot_copy_dir>/family-<key>/hot`, else `<data>/family/<key>/hot`.
    /// Only this `hot` folder is replaced or removed: `family-<key>` may also hold what an earlier Origami made.
    static func hotFolder(_ settings: Settings, key: String) -> URL {
        (settings.hotCopyDir?.appendingPathComponent("family-\(key)", isDirectory: true)
            ?? settings.dataDir.appendingPathComponent("family/\(key)", isDirectory: true))
            .appendingPathComponent("hot", isDirectory: true)
    }
    /// Measures the drives a family model reads that are unmeasured or stale (all with `force`; `Calibrations`): the
    /// GGUF's, and each hot copy place's (`hot`, then `more`) once it holds a copy. Returns the copies' and the model's
    /// measurements, which seed the engine's drive estimates and order the places.
    @discardableResult
    static func measureDrives(
        _ gguf: GGUFSource, geometry: ModelGeometry, hot: URL, more: [URL] = [], settings: Settings, force: Bool = false,
        log: (String) -> Void
    ) -> (copy: DriveSpeed?, model: DriveSpeed?, more: [DriveSpeed?]) {
        let list = { (folder: URL) in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] }
        // A variant's experts are read from their own file.
        let experts = gguf.expertFiles.map { URL(fileURLWithPath: $0) }
        let model = experts.first?.deletingLastPathComponent() ?? gguf.folder
        let modelFiles = experts.isEmpty ? gguf.files.map { gguf.folder.appendingPathComponent($0) } : experts
        // A place whose drive is not connected is not listed (nor touched): its folder would resolve to the boot drive.
        let copies = ([hot] + more).map { folder in Stores.volume(folder).mounted ? list(folder) : [] }
        // The average routed record: every `ffn_*_exps` tensor over its layers' experts.
        let parts = gguf.tensors.values.filter { $0.name.hasPrefix("blk.") && $0.name.contains("_exps.") }
        let layers = Set(parts.map { $0.name.split(separator: ".")[1] }).count
        let record = Int(parts.reduce(Int64(0)) { $0 + $1.length }) / max(1, layers * geometry.experts)
        let all = DriveSpeeds.ensure(
            settings.driveSpeedsFile,
            drives: [(model, modelFiles, Stores.volume(model).name)]
                + zip([hot] + more, copies).filter { !$0.1.isEmpty }.map { ($0.0, $0.1, Stores.volume($0.0).name) },
            recordBytes: max(4096, record), force: force, log: log)
        let speed = { (url: URL) in DriveSpeeds.key(url).flatMap { all[$0] } }
        let rates = zip([hot] + more, copies).map { $0.1.isEmpty ? nil : speed($0.0) }
        return (rates[0], speed(model), Array(rates.dropFirst()))
    }
    /// `origami setup` for a family GGUF: measures its drives (the hot copy is built while serving).
    static func setup(_ url: URL, settings: Settings) throws {
        let gguf = try ExpertVariant.source(url, settings: settings)
        let geometry = try ModelGeometry(gguf: gguf)
        let size = gguf.files.reduce(UInt64(0)) { $0 + (Models.fileSize(gguf.folder.appendingPathComponent($1)).map(UInt64.init) ?? 0) }
        print("Model   \(geometry.family.displayName) (\(Commands.gigabytes(size)), GGUF)\n        \(url.path)")
        print("Hot copy: built while serving, after the first reply (see the dashboard's Experts card).")
        if let note = settings.expertBitsNote { print("        \(note)") }
        print("Drives  measuring the read speed of each drive holding the model or a copy…")
        let hot = HotCopy.placeSettings(settings).map { hotFolder($0, key: FamilySupport.key(gguf)) }
        measureDrives(gguf, geometry: geometry, hot: hot[0], more: Array(hot.dropFirst()), settings: settings, force: true) {
            print("        \($0)")
        }
    }
}

extension FamilySetup {
    /// The file naming the model of a family epoch folder (`<context_dir>/<epoch16>`, whose name does not show it), so the
    /// storage map and `origami cache` can list another model's or build's saved prompts on their own.
    static let epochMarker = "origami-epoch.json"
    /// This executable's build as the markers name it: the first 16 hex digits of its SHA-256 (read once, when first asked).
    static let build = String(((try? OrigamiRuntime.executableDigest()) ?? "unknown").prefix(16))
    /// Writes the marker into the epoch folder `folder` unless it has one (a folder made by an earlier start gets it now).
    static func markEpoch(_ folder: URL, model: String, key: String, context: Int) {
        let file = folder.appendingPathComponent(epochMarker)
        guard FileManager.default.fileExists(atPath: folder.path), !FileManager.default.fileExists(atPath: file.path) else { return }
        let marker: [String: Any] = [
            "model": model, "family_key": key, "context": context, "version": version, "build": build,
            "created": ISO8601DateFormatter().string(from: Date()),
        ]
        try? JSONSerialization.data(withJSONObject: marker, options: [.sortedKeys]).write(to: file, options: .atomic)
    }
    /// The marker of the family epoch folder `folder`, nil when it has none.
    static func epoch(_ folder: URL) -> [String: Any]? {
        guard folder.lastPathComponent.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil else { return nil }
        return (try? Data(contentsOf: folder.appendingPathComponent(epochMarker))).flatMap {
            (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any]
        }
    }

    /// One row of `doctor` and `status` for a family model: a problem when not `ok`; `more` are lines under it.
    struct Row {
        let ok: Bool, label: String
        var text: String
        var more: [String] = []
    }

    /// Record sizes from the GGUF's headers: enough to open a hot copy and rank its records without loading the model.
    private final class RecordSizes: HotCopySource {
        let bytes: [Int: Int], experts: Int
        init(_ bytes: [Int: Int], experts: Int) {
            self.bytes = bytes
            self.experts = experts
        }
        func hotCopyRecordBytes(_ key: Int) -> Int? { bytes[key / experts] }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws { throw CLIError("The model is not loaded") }
    }

    /// What `doctor` and `status` say of the GLM or DeepSeek GGUF at `url`, from its headers and Origami's files (no
    /// weights loaded): the model and its files, the experts it reads, the hot copy with the share of expert uses it and
    /// other sizes would hold (from the counts serving saved), and the memory it may use.
    static func rows(_ url: URL, settings: Settings, host: OrigamiHost?) -> (rows: [Row], report: [String: Any]) {
        let gguf: GGUFSource
        let geometry: ModelGeometry
        do {
            gguf = try ExpertVariant.source(url, settings: settings)
            geometry = try ModelGeometry(gguf: gguf)
        } catch { return ([Row(ok: false, label: "model", text: "\(url.path): \(error)")], ["model": ["path": url.path]]) }
        let key = FamilySupport.key(gguf)
        let name =
            ModelRegistry.entry(settings.model, settings: settings) != nil ? settings.model! : url.deletingPathExtension().lastPathComponent
        let size = gguf.files.reduce(UInt64(0)) {
            $0 + (Models.fileSize(gguf.folder.appendingPathComponent($1)).map { UInt64(max(0, $0)) } ?? 0)
        }
        // A low-bit variant: its experts file with its size and drive, beside the base's files.
        let expertsFile = ExpertVariant.filesSummary(gguf.expertFiles)
        var rows: [Row] = []
        var report: [String: Any] = [
            "model": [
                "path": url.path, "name": name, "family": geometry.family.rawValue, "key": key, "bytes": size,
                "files": gguf.files.count, "experts": ExpertVariant.note(gguf).map { $0 as Any } ?? NSNull(),
                "experts_file": expertsFile.map { ["paths": gguf.expertFiles, "drive": $0.drive, "bytes": $0.bytes] as Any } ?? NSNull(),
            ]
        ]
        let problem = ModelCompatibility.familyProblem(gguf, geometry)
        rows.append(
            Row(
                ok: problem == nil, label: "model",
                text: problem.map { "\(url.lastPathComponent): Origami cannot run it: \($0)" }
                    ?? "\(name), \(geometry.family.displayName) (GGUF, \(Commands.gigabytes(size)) in \(gguf.files.count) file\(gguf.files.count == 1 ? "" : "s")"
                    + (expertsFile.map { " + \(Commands.gigabytes($0.bytes)) low-bit experts file" } ?? "") + ")",
                more: [gguf.folder.path]))
        if let note = ExpertVariant.note(gguf) {
            rows.append(
                Row(
                    ok: true, label: "experts", text: note,
                    more: expertsFile.map { ["experts file: \($0.text)"] + gguf.expertFiles.dropFirst() } ?? gguf.expertFiles))
        }
        // The hot copy: what its folder holds, and whether serving would read it (`HotCopy.place` decides as serving does).
        // shortcut: without the loaded model this does not know whether every expert fits in memory, so on such a Mac it
        // says a copy is built where serving says none is needed; pass `resident` once the plan here has the experts' bytes.
        let bytes = FamilyExperts.recordBytes(gguf, geometry: geometry)
        let sizes = RecordSizes(bytes, experts: geometry.experts)
        let folder = hotFolder(settings, key: key)
        let identity = LocalModelIdentity.familyRepresentation(source: key)
        let copy = OrigamiEngine.HotCopy.open(folder: folder, identity: identity, source: sizes)
        let held = copy.map { "\($0.keys.count) experts (\(Commands.gigabytes(UInt64($0.bytes)))) in \(folder.path)" }
        let routed = bytes.values.reduce(UInt64(0)) { $0 + UInt64($1 * geometry.experts) }
        var hot: Row
        switch HotCopy.place(
            settings, model: gguf.expertFiles.first.map { URL(fileURLWithPath: $0) } ?? url, key: key, routed: routed)(
                {
                    $0.standardizedFileURL == folder.standardizedFileURL ? UInt64(copy?.bytes ?? 0) : 0
                })
        {
        case .folder:
            hot = Row(
                ok: true, label: "hot copy", text: held ?? "not built yet: the server builds it in \(folder.path) after the first reply")
        case .none(let state, let detail):
            hot = Row(
                ok: state != "not_connected" && state != "not_allowed", label: "hot copy", text: held.map { "\($0), not read: " } ?? "")
            hot.text += detail
        }
        report["hot_copy"] = ["path": folder.path, "records": copy?.keys.count ?? 0, "bytes": copy?.bytes ?? 0]
        // The usage curve from the counts serving saved, ranked as the keeper ranks them.
        let counted = settings.dataDir.appendingPathComponent("family/\(key)", isDirectory: true)
        let uses = OrigamiEngine.HotCopy.loadUses(counted.appendingPathComponent("uses.json"))
        let misses = OrigamiEngine.HotCopy.loadUses(counted.appendingPathComponent("misses.json"))
        if !uses.isEmpty || !misses.isEmpty {
            let curve = HotCopyCurve(uses: uses, misses: misses, source: sizes)
            let candidates = Set([UInt64(copy?.bytes ?? 0), 20_000_000_000, 40_000_000_000, 80_000_000_000]).filter { $0 > 0 }.sorted()
            hot.more.append(
                "a copy of "
                    + candidates.map { "\(Commands.gigabytes($0)) would cover \(Int((curve.coverage($0) * 100).rounded()))%" }
                    .joined(separator: ", ") + " of expert uses counted")
            report["hot_copy_coverage"] = candidates.map { ["bytes": $0, "share": curve.coverage($0)] as [String: Any] }
        }
        rows.append(hot)
        // Each further place (`hot_copy_more`), one line each, as serving would use it (`HotCopy.places`).
        let places = HotCopy.places(
            settings, model: gguf.expertFiles.first.map { URL(fileURLWithPath: $0) } ?? url, key: key, routed: routed)
        var placeReports: [[String: Any]] = [report["hot_copy"] as? [String: Any] ?? [:]]
        for (index, one) in HotCopy.placeSettings(settings).enumerated().dropFirst() {
            let folder = hotFolder(one, key: key)
            let reachable = Stores.volume(one.hotCopyDir ?? folder).mounted
            let copy = reachable ? OrigamiEngine.HotCopy.open(folder: folder, identity: identity, source: sizes) : nil
            let held = copy.map { "\($0.keys.count) experts (\(Commands.gigabytes(UInt64($0.bytes)))) in \(folder.path)" }
            let size = ", cap " + (one.hotCopy.map(Settings.hotCopyName) ?? "auto")
            switch places[index]({ $0.standardizedFileURL == folder.standardizedFileURL ? UInt64(copy?.bytes ?? 0) : 0 }) {
            case .folder:
                rows.append(
                    Row(
                        ok: true, label: "hot copy \(index + 1)",
                        text: (held ?? "not built yet: the server builds it in \(folder.path)") + size))
            case .none(let state, let detail):
                rows.append(
                    Row(
                        ok: state != "not_connected" && state != "not_allowed", label: "hot copy \(index + 1)",
                        text: (held.map { "\($0), not read: " } ?? "") + detail))
            }
            placeReports.append(["path": folder.path, "records": copy?.keys.count ?? 0, "bytes": copy?.bytes ?? 0])
        }
        report["hot_copies"] = placeReports
        // Memory as serving's ledger allows it (memory_limit, else 70% of this Mac's memory), less the other weights.
        let physical = host?.physicalMemory ?? ProcessInfo.processInfo.physicalMemory
        let ceiling = min(settings.memoryLimit ?? physical / 10 * 7, host.map(OrigamiPlan.memoryCeiling(host:)) ?? .max)
        let dense = UInt64(gguf.tensors.values.filter { !$0.name.contains("_exps.") }.reduce(Int64(0)) { $0 + $1.length })
        let floor = physical - WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: physical)
        rows.append(
            Row(
                ok: dense < ceiling, label: "memory",
                text: dense < ceiling
                    ? "up to \(Commands.gigabytes(ceiling)) (\(settings.memoryLimit == nil ? "70% of this Mac's memory" : "memory_limit")), "
                        + "\(Commands.gigabytes(dense)) of it for the weights other than the routed experts, the rest for the context "
                        + "and cached experts; \(Commands.gigabytes(floor)) always left for other apps"
                    : "the weights other than the routed experts take \(Commands.gigabytes(dense)), more than the "
                        + "\(Commands.gigabytes(ceiling)) Origami may use here: raise memory_limit, or use a smaller model"))
        report["memory"] = ["cap": ceiling, "dense": dense, "floor": floor]
        return (rows, report)
    }
}
