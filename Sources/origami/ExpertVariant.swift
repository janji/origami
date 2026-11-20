import Foundation
import OrigamiEngine

/// `origami model add <experts.gguf> --base <model>`: a Qwen model whose routed experts are read from a published low-bit
/// GGUF and everything else (dense weights, router, PLE table, tokenizer) from an exact 4-bit model already added. Its
/// folder is generated in `<data_dir>/variants/<name>`: links to the base's files, the base's config marked
/// `expert_format: gguf`, and `gguf-experts.json` (`GGUFExperts.variant` checks the pairing). Nothing is copied or
/// converted; removing the variant removes only that folder.
enum ExpertVariant {
    static func folder(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("variants", isDirectory: true) }

    /// Checks `experts` against the base model `spec` and writes the variant's folder `name`; returns it.
    static func make(experts: URL, base spec: String, name: String, settings: Settings) throws -> URL {
        let (base, manifest) = try pairing(experts: experts, base: spec, settings: settings)
        guard
            var config = (try? JSONSerialization.jsonObject(with: Data(contentsOf: base.directory.appendingPathComponent("config.json"))))
                as? [String: Any]
        else { throw CLIError("The base's config.json cannot be read") }
        for key in ["quantization", "quantization_config"] {
            if var quantization = config[key] as? [String: Any] {
                quantization["expert_format"] = "gguf"
                config[key] = quantization
            }
        }
        let manager = FileManager.default
        let target = folder(settings).appendingPathComponent(name, isDirectory: true)
        // Built beside it and swapped in, so a refusal leaves an earlier variant of that name as it was.
        let staging = folder(settings).appendingPathComponent(".\(name)-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: staging) }
        for item in try manager.contentsOfDirectory(atPath: base.directory.path)
        where !item.hasPrefix(".") && !["config.json", GGUFExperts.file].contains(item) {
            try manager.createSymbolicLink(
                at: staging.appendingPathComponent(item), withDestinationURL: base.directory.appendingPathComponent(item))
        }
        try ModelSource.configJSON(config, options: [.prettyPrinted, .sortedKeys]).write(to: staging.appendingPathComponent("config.json"))
        try manifest.write(to: staging.appendingPathComponent(GGUFExperts.file))
        _ = try ModelSource(directory: staging)
        // Only links and two small files: removing it never touches the base's files or the GGUF.
        if manager.fileExists(atPath: target.path) { try manager.removeItem(at: target) }
        try manager.moveItem(at: staging, to: target)
        return target
    }

    /// The Qwen pairing `make` checks: the base model and the variant's `gguf-experts.json`; throws a refusal.
    private static func pairing(experts: URL, base spec: String, settings: Settings) throws -> (ModelSource, Data) {
        guard experts.pathExtension.lowercased() == "gguf", FileManager.default.fileExists(atPath: experts.path) else {
            throw CLIError("--base takes a .gguf file of experts; \(experts.path) is not one")
        }
        guard let url = Models.locate(spec, settings: settings), !Models.isGGUF(url.path) else {
            throw CLIError("No exact model \(spec) to base it on; `origami model list` shows the added ones")
        }
        let base = try ModelSource(directory: url)
        let gguf: GGUFSource
        do { gguf = try GGUFSource(experts, tokenizer: true, partial: true) } catch {
            throw CLIError("Cannot read \(experts.lastPathComponent) as a GGUF (\(error)); is the download complete?")
        }
        return (base, try refusal { try GGUFExperts.variant(gguf: gguf, base: base) })
    }

    /// A pairing the engine refuses, with what to do about it.
    private static func refusal<T>(_ pair: () throws -> T) throws -> T {
        do { return try pair() } catch let error as EngineError where error.code == "model_incompatible" {
            throw CLIError("\(error).\nNothing was added: these experts need the exact model they were made from as their base.")
        }
    }

    /// The check `model add <experts> --base <spec>` makes before it adds anything, with nothing written (the dashboard
    /// asks it before its add job): throws a refusal; returns what an experts-only file cannot show, for the person to
    /// acknowledge, or nil when the file pairs fully.
    static func check(experts: URL, base spec: String, settings: Settings) throws -> String? {
        if let family = try familyPairing(experts: experts, base: spec, settings: settings) { return family.unverified }
        _ = try pairing(experts: experts, base: spec, settings: settings)
        return nil
    }

    /// The family pairing `family` checks, nil when `spec` names no family GGUF: the base's GGUF, its family, and what an
    /// experts-only file cannot show (nil when it pairs fully); throws a refusal.
    private static func familyPairing(experts: URL, base spec: String, settings: Settings) throws -> (
        url: URL, family: ModelFamily, unverified: String?
    )? {
        guard let url = Models.locate(spec, settings: settings) ?? URL?(URL(fileURLWithPath: Settings.expand(spec))),
            Models.isGGUF(url.path), let base = try? GGUFSource(url, tokenizer: true),
            let family = ModelFamily.detect(gguf: base.metadata), family != .qwen
        else { return nil }
        guard experts.pathExtension.lowercased() == "gguf", FileManager.default.fileExists(atPath: experts.path) else {
            throw CLIError("--base takes a .gguf file of experts; \(experts.path) is not one")
        }
        if ModelRegistry.entry(spec, settings: settings)?["experts"] != nil {
            throw CLIError("\(spec) has low-bit experts itself; pass the exact model it was added on")
        }
        let gguf: GGUFSource
        do { gguf = try GGUFSource(experts, tokenizer: true, partial: true) } catch {
            throw CLIError("Cannot read \(experts.lastPathComponent) as a GGUF (\(error)); is the download complete?")
        }
        let pairing = try refusal { try FamilyExpertVariant.pair(gguf, base: base) }
        let report = ModelCompatibility.check(gguf: pairing.source, source: url.path)
        guard report.verdict == .runs else { throw CLIError("Origami cannot run it: \(report.reason)") }
        let unverified =
            pairing.unverified.isEmpty
            ? nil
            : "\(experts.lastPathComponent) \(pairing.unverified.joined(separator: ", ")): only its banks' shapes say it is \(spec) "
                + "quantized another way. Add it only if its publisher says so."
        return (url, family, unverified)
    }

    /// A model added again under its name keeps the settings saved for it (`Settings.modelKeys` in its old entry) when it
    /// is the same file or folder (and the same experts file); another one under that name starts without them. The line
    /// saying which, or nil when the old entry kept none.
    static func keepSettings(_ entry: inout [String: Any], from entries: [[String: Any]]) -> String? {
        guard let name = entry["name"] as? String, let old = entries.first(where: { $0["name"] as? String == name }) else { return nil }
        let keys = Settings.modelKeys.filter { old[$0] != nil && !(old[$0] is NSNull) }
        guard !keys.isEmpty else { return nil }
        guard old["path"] as? String == entry["path"] as? String, old["experts"] as? String == entry["experts"] as? String else {
            return "The settings saved for \(name) (\(keys.joined(separator: ", "))) were for \(old["path"] as? String ?? "another model")"
                + " and are not kept: set them again with `origami config set <key> <value> --model \(name)`."
        }
        for key in keys { entry[key] = old[key] }
        return "Kept the settings saved for \(name): \(keys.joined(separator: ", "))."
    }

    /// GLM-5.3-Flash and DeepSeek-V4.1-Flash: the models.json entry of a variant of the family GGUF `spec` names (its path
    /// the base's GGUF, `experts` the file its routed experts are read from; nothing is written), or nil when `spec` names
    /// no family GGUF. `FamilyExpertVariant.pair` checks the pairing; what an experts-only file cannot show is acknowledged.
    static func family(experts: URL, base spec: String, name: String, settings: Settings, yes: Bool) throws -> [String: Any]? {
        guard let (url, family, unverified) = try familyPairing(experts: experts, base: spec, settings: settings) else { return nil }
        if let unverified {
            print(unverified)
            guard Terminal.confirm("Add it?", yes: yes) else {
                throw CLIError("Not added: pass --yes once you know \(experts.lastPathComponent) holds \(spec)'s experts")
            }
        }
        return [
            "name": name, "path": url.path, "family": family.rawValue, "base": spec, "experts": experts.path,
            "added": ISO8601DateFormatter().string(from: Date()),
        ]
    }

    /// The GGUF `url` as the configured model reads it: a family variant's routed experts from its experts file.
    static func source(_ url: URL, settings: Settings) throws -> GGUFSource {
        let base = try GGUFSource(url)
        guard let entry = ModelRegistry.entry(settings.model, settings: settings), entry["path"] as? String == url.path,
            let experts = entry["experts"] as? String
        else { return base }
        do {
            return try FamilyExpertVariant.pair(GGUFSource(URL(fileURLWithPath: experts), partial: true), base: base, verify: false).source
        } catch {
            let name = entry["name"] as? String ?? "this model"
            throw CLIError(
                "The low-bit experts of \(name) in \(experts) cannot be read (\(error)); put the file back, "
                    + "or `origami model remove \(name)` and add it again")
        }
    }
    /// A family variant's experts file parts (absolute paths), for its key and the files kept; none for another entry.
    static func expertFiles(_ entry: [String: Any]) -> [String] {
        guard entry["family"] != nil, let path = entry["experts"] as? String, let parts = try? GGUFSource.parts(URL(fileURLWithPath: path))
        else { return [] }
        return parts.files.map { parts.folder.appendingPathComponent($0).standardizedFileURL.path }
    }

    /// Seeds the usage counts of the variant `name` from its base's: its router is the base's, so they describe the same
    /// routing, for every layer it has (Qwen: `routing-<fingerprint>.json`; GLM and DeepSeek: `family/<key>/uses.json`
    /// and `misses.json`). Only when the variant has none and the base has some; copied, so each keeps its own from then
    /// on. At `model add` and at the variant's first load; returns the line to say, nil when nothing was seeded.
    @discardableResult
    static func seedCounts(_ name: String?, settings: Settings) -> String? {
        guard let entry = ModelRegistry.entry(name, settings: settings), let base = entry["base"] as? String,
            let path = entry["path"] as? String
        else { return nil }
        let data = settings.dataDir
        let pairs: [(from: URL, to: URL)]
        if entry["family"] != nil {
            let experts = expertFiles(entry)
            guard !experts.isEmpty, let parts = try? GGUFSource.parts(URL(fileURLWithPath: path)) else { return nil }
            let from = data.appendingPathComponent("family/\(FamilySupport.key(folder: parts.folder, files: parts.files))")
            let to = data.appendingPathComponent("family/\(FamilySupport.key(folder: parts.folder, files: parts.files, experts: experts))")
            pairs = ["uses.json", "misses.json"].map { (from.appendingPathComponent($0), to.appendingPathComponent($0)) }
        } else {
            guard owns(path, settings: settings), let folder = Models.locate(base, settings: settings),
                let from = try? ModelSource.fingerprint(directory: folder),
                let to = try? ModelSource.fingerprint(directory: URL(fileURLWithPath: path))
            else { return nil }
            pairs = [
                (
                    data.appendingPathComponent("routing-\(from.prefix(12)).json"),
                    data.appendingPathComponent("routing-\(to.prefix(12)).json")
                )
            ]
        }
        let manager = FileManager.default
        guard pairs.allSatisfy({ !manager.fileExists(atPath: $0.to.path) }),
            pairs.contains(where: { manager.fileExists(atPath: $0.from.path) })
        else { return nil }
        for pair in pairs where manager.fileExists(atPath: pair.from.path) {
            guard (try? manager.createDirectory(at: pair.to.deletingLastPathComponent(), withIntermediateDirectories: true)) != nil,
                (try? manager.copyItem(at: pair.from, to: pair.to)) != nil
            else { return nil }
        }
        return "usage counts of \(name!) seeded from \(base)'s (the same router), for its hot copy; each keeps its own from now on"
    }

    /// The experts file parts a variant entry reads (a family entry's, or a Qwen variant folder's manifest's), for
    /// `model list`; none for another entry.
    static func readFiles(_ entry: [String: Any]) -> [String] {
        guard entry["family"] == nil else { return expertFiles(entry) }
        guard entry["base"] != nil, let path = entry["path"] as? String,
            let data = try? Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent(GGUFExperts.file)),
            let files = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["files"] as? [[String: Any]]
        else { return [] }
        return files.compactMap { $0["path"] as? String }
    }

    /// A variant's experts file parts as status, doctor, `model list` and the slow-read advice name them: their bytes, the
    /// drive, and "<first part> on <drive>"; nil for none.
    static func filesSummary(_ paths: [String]) -> (bytes: UInt64, drive: String, text: String)? {
        guard let first = paths.first else { return nil }
        let bytes = paths.reduce(UInt64(0)) { $0 + UInt64(max(0, Models.fileSize(URL(fileURLWithPath: $1)) ?? 0)) }
        let drive = Stores.driveName(first)
        return (bytes, drive, "\(first) on \(drive)")
    }

    /// What status, doctor, the dashboard, the server log and the request log say of a family variant.
    static func note(_ gguf: GGUFSource) -> String? {
        guard !gguf.expertFiles.isEmpty else { return nil }
        let banks = gguf.tensors.values.filter { $0.file.hasPrefix("/") }
        var counts: [Int: Int] = [:]
        for bank in banks { counts[bank.type, default: 0] += 1 }
        let types = counts.sorted { ($0.value, $1.key) > ($1.value, $0.key) }.map { "\(GGUFQuant.name($0.key)) ×\($0.value)" }
        let bits = Double(banks.reduce(Int64(0)) { $0 + $1.length }) * 8 / Double(max(1, banks.reduce(0) { $0 + $1.shape.reduce(1, *) }))
        let layers = Set(banks.compactMap { Int($0.name.split(separator: ".")[1]) }).sorted()
        let all = gguf.tensors.keys.filter { $0.hasSuffix(".ffn_gate_exps.weight") }.count
        let file = URL(fileURLWithPath: gguf.expertFiles[0]).lastPathComponent
        let from = gguf.expertFiles.count == 1 ? file : "\(GGUFSource.modelPrefix(file)) (\(gguf.expertFiles.count) parts)"
        return "low-bit experts from \(from) (" + types.joined(separator: ", ") + String(format: "; %.2f bits per weight", bits)
            + (layers.count == all || layers.isEmpty
                ? "" : "; layers \(layers.first!)–\(layers.last!), the other \(all - layers.count) the base's")
            + "): replies differ from the exact model"
    }

    /// Whether `path` is a folder `make` wrote.
    static func owns(_ path: String, settings: Settings) -> Bool {
        URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL.path == folder(settings).standardizedFileURL.path
    }

    /// What status, doctor, the dashboard, the server log and the request log say of a model whose experts are in a GGUF.
    static func note(_ model: ModelSource) -> String? {
        guard let gguf = model.ggufExperts else { return nil }
        return "low-bit experts from \(gguf.files.map { URL(fileURLWithPath: $0.path).lastPathComponent }.joined(separator: ", ")) ("
            + "\(gguf.typeSummary); \(String(format: "%.2f", gguf.bitsPerWeight)) bits per weight): replies differ from the exact model; "
            + "no MTP in this mode"
    }
}
