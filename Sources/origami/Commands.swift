import CryptoKit
import Darwin
import Foundation
import Metal
import OrigamiEngine

/// The commands that set up, serve and manage Origami: setup, serve, status, model, store, config, cache, verify,
/// uninstall. Each loads `Settings` itself; failures are `CLIError`s that `main` prints and maps to an exit status.
enum Commands {
    /// Stable provenance for HTTP observations, computed once at startup. Credentials are excluded before hashing;
    /// settings identify the startup configuration, not later per-request overrides or governor decisions.
    static func observationTags(
        _ settings: Settings, version: String, identity: String, context: Int,
        startupTuning: [String: Any]? = nil, tuningScope: String = "front; no worker tuning selected"
    ) throws -> [String: String] {
        func value<T>(_ item: T?) -> Any { item.map { $0 as Any } ?? NSNull() }
        // Settings.report is for people: its rounded GB amounts and joined path lists are not identities.
        let values: [String: Any] = [
            "model": value(settings.model), "stores": settings.stores.map { ["path": $0.path, "size_bytes": value($0.size)] },
            "data_dir": settings.dataDir.path, "memory_limit_bytes": value(settings.memoryLimit),
            "expert_memory_bytes": value(settings.expertMemory), "context_dir": value(settings.contextDir?.path),
            "hot_copy_bytes": value(settings.hotCopy), "hot_copy_dir": value(settings.hotCopyDir?.path),
            "models_dirs": settings.modelsDirs.map(\.path), "context": settings.context, "port": settings.port,
            "allow_fast": settings.allowFast, "quality": try JSONSerialization.jsonObject(with: JSONEncoder().encode(settings.quality)),
            "reasoning_effort": value(settings.reasoningEffort), "max_thinking_tokens": value(settings.maxThinkingTokens),
            "commands": settings.commands, "list_models": settings.listModels, "stopped": settings.stopped,
            "context_on_disk": value(settings.contextOnDisk), "status_messages": settings.statusMessages,
            "gpu_keepalive": value(settings.gpuKeepAlive), "speculation": value(settings.mtpWeights?.path),
            "request_log": value(settings.requestLog?.path), "listen_lan": settings.listenLAN, "lan_open": settings.lanOpen,
            "prompt_cache_entries": settings.promptCacheEntries,
            "sampling_custom": settings.samplingCustom, "sampling_values": settings.samplingValues,
            "sampling_override": settings.samplingOverride,
        ]
        var tags = try ObservationProvenance.tags(
            settings: values, environment: ProcessInfo.processInfo.environment,
            startupTuning: startupTuning, tuningScope: tuningScope)
        tags.merge([
            "engine_version": version, "executable_sha256": try OrigamiRuntime.executableDigest(),
            "model_identity": identity, "model_identity_scope": "local_incarnation", "context_tier": String(context),
        ]) { $1 }
        return tags
    }
    static func gigabytes(_ bytes: UInt64) -> String { OrigamiPlan.gigabytes(bytes) }
    /// Installed RAM the way Apple sells it (binary gigabytes: 32 GB, not 34.4).
    static func ram(_ bytes: UInt64) -> String { String(format: "%.0f GB", Double(bytes) / 1_073_741_824) }

    /// The model folder from the settings, or a clear next step.
    static func model(_ settings: Settings) throws -> ModelSource {
        guard let url = Models.locate(settings.model, settings: settings) else {
            if let spec = settings.model {
                if Models.isGGUF(Settings.expand(spec)), FileManager.default.fileExists(atPath: Settings.expand(spec)) {
                    throw CLIError("\(spec) is a GGUF; `origami model add \(spec)` converts it once into the layout Origami runs.")
                }
                let why = missingModel(spec, settings: settings)
                throw CLIError(why.prefix(1).uppercased() + why.dropFirst())
            }
            throw CLIError("No model found. Run `origami setup` to download \(Models.defaultRepo), or `origami setup --model <folder>`.")
        }
        try requirePresent(url, settings: settings)
        return try ModelSource(directory: url)
    }
    /// A registered model whose folder is gone says why (its drive not connected) rather than "No config.json".
    static func requirePresent(_ url: URL, settings: Settings) throws {
        guard !FileManager.default.fileExists(atPath: url.path), let spec = settings.model else { return }
        let why = missingModel(spec, settings: settings)
        throw CLIError(why.prefix(1).uppercased() + why.dropFirst())
    }

    // MARK: setup

    static func setup(flags: [String: [String]], switches: Set<String>) async throws {
        var settings = try Settings.load(flags: flags)
        if let added = flags["store"] {
            // Merged into the stores this config file lists (a --store flag alone would replace them), and tried first
            // in this run: the dashboard's conversion passes the folder the user chose.
            var stored = try Settings.load(flags: flags.filter { $0.key != "store" }).stores
            let first = added.map(Settings.expand)
            for path in first where !stored.contains(where: { $0.path == path }) { stored.append(.init(path: path)) }
            settings.stores = stored
            try settings.saveStores()
            settings.stores = stored.filter { first.contains($0.path) } + stored.filter { !first.contains($0.path) }
        }
        if let gguf = FamilySetup.gguf(settings) {
            // A GGUF given by path becomes the model, as a Qwen folder does: `status`, `doctor` and a service then use it;
            // only once its setup succeeded, so a GGUF Origami refuses never replaces the default that worked.
            try FamilySetup.setup(gguf, settings: settings)
            if try Settings.readConfig(settings.configFile)["model"] as? String != gguf.path,
                ModelRegistry.entry(settings.model, settings: settings) == nil
            {
                try Settings.write(["model": gguf.path], to: settings.configFile)
            }
            return
        }
        let yes = switches.contains("yes")
        // The Mac first: one too small learns it before any download (README: 24 GB or more; 16 GB fits only an 8K context).
        let host = try OrigamiHost.current()
        print("Mac     \(host.gpuName), \(ram(host.physicalMemory)) memory")
        for note in host.notes { print("        \(note)") }
        guard host.physicalMemory >= 16 << 30 else {
            throw CLIError(
                "This Mac has \(ram(host.physicalMemory)) of memory; Origami needs 24 GB or more (16 GB runs only an 8K context).")
        }
        if host.physicalMemory < 24 << 30 {
            print("        Not recommended: 24 GB or more is needed for more than an 8K context.")
        }
        var url = Models.locate(settings.model, settings: settings)
        if let url { try requirePresent(url, settings: settings) }
        if url == nil {
            let spec = settings.model
            let repo = spec == nil ? Models.defaultRepo : (spec!.contains("/") && !"/~.".contains(spec!.first!) ? spec! : nil)
            guard let repo else {
                throw CLIError(
                    "No model folder at \(spec!) (it needs config.json): pass a folder, an owner/name Hugging Face repo, or a name from `origami model list`"
                )
            }
            // Another repo is checked before a byte is fetched (GGUF files, a model Origami does not run as it is).
            if repo != Models.defaultRepo {
                let report = try Models.checkRepo(repo)
                guard report.verdict == .runs else {
                    throw CLIError(
                        "\(repo) is converted once before it runs: `origami model add \(repo)` does that, then `origami setup --model <its name>`."
                    )
                }
            }
            print(
                "Origami needs the model weights\(repo == Models.defaultRepo ? " (about 104 GB)" : ""). None were found on this Mac.")
            guard Terminal.confirm("Download \(repo) from Hugging Face to \(settings.modelsDir.path)?", yes: yes) else {
                throw CLIError("No model. Pass --model <folder> if you already have it, or run `origami setup --yes` to download.")
            }
            url = try await Models.download(repo: repo, settings: settings)
        }
        let model = try ModelSource(directory: url!)
        print(
            "Model   \(model.displayName) (\(gigabytes(model.totalBytes)), \(model.quantization.rawValue))\n        \(model.directory.path)"
        )
        // A named model keeps its name (and the expert width it was added with); a found or downloaded folder becomes the default.
        if settings.model != model.directory.path, ModelRegistry.entry(settings.model, settings: settings) == nil {
            try Settings.write(["model": model.directory.path], to: settings.configFile)
            settings.model = model.directory.path
        }
        if let note = settings.expertBitsNote { print("        \(note)") }
        let resident = try OrigamiPlan.make(
            model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit, expertMemory: settings.expertMemory,
            promptCache: settings.promptCacheRoot
        ).allResident
        if model.ggufExperts != nil {
            print("Stores  no packs: this model's experts are read from its GGUF as they are; packs are for MLX experts")
            if Stores.plan(model: model, settings: settings, found: Stores.scan(model: model, settings: settings)).hotCopyFolder != nil {
                print("Hot copy built while serving, after the first reply (see the dashboard's Health).")
            }
        } else if resident && !switches.contains("full-pack") {
            print("Stores  not needed: this Mac keeps all \(OrigamiPlan.maximumSlots) experts in memory once they are read")
        } else {
            try prepareStores(
                model: model, settings: settings, noPack: switches.contains("no-pack"), fullPack: switches.contains("full-pack"))
        }
        let storage = Stores.plan(model: model, settings: settings, found: Stores.scan(model: model, settings: settings))
        let plan = try OrigamiPlan.make(
            model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit, expertMemory: settings.expertMemory,
            storage: storage, promptCache: settings.promptCacheRoot)
        print(
            "Memory  up to \(gigabytes(plan.memory.cap)) (now \(gigabytes(plan.memory.start)); leaves \(gigabytes(plan.memory.floor)) for other apps)"
        )
        let context = OrigamiPlan.contextPlan(
            model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit, expertMemory: settings.expertMemory,
            storage: storage, promptCache: settings.promptCacheRoot)
        print(
            "Context \(context.served) tokens " + (context.paged ? "kept in a file on disk" : "in memory")
                + "; this Mac keeps up to \(context.inMemory / 1024)K in memory, more is paged to disk")
        if let note = context.note { print("        \(note)") }
        if !settings.promptsBesideData {
            print(
                "Prompts saved in \(settings.promptCacheRoot.path): \(Stores.driveName(settings.dataDir.path)) has no file ownership "
                    + "(exFAT or FAT), so they could not be kept private or reused there")
        }
        print("\nReady. Start the server with `origami serve`, then point any OpenAI client at http://127.0.0.1:\(settings.port)/v1")
    }

    /// Builds what is missing: a pack on another drive (or on the model's drive with --full-pack),
    /// further copies on other drives, and drive measurements for two-drive routing (the hot copy is built while serving).
    /// Fails, once the rest is done, when a further copy could not be built for lack of room, saying how much (`noRoom`).
    static func prepareStores(model: ModelSource, settings: Settings, noPack: Bool, fullPack: Bool) throws {
        ModelLayout.use(model)
        let modelDisk = Stores.volume(model.directory).disk
        for store in settings.storeFolders {
            // Creating the folder once from Terminal makes macOS ask for removable-volume access here,
            // not later in a background server where nobody sees the prompt.
            try? FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
            StoreBuilder.removePartials(in: StoreHeader.modelFolder(store: store, model: model))  // an interrupted build's payloads
        }
        var found = Stores.scan(model: model, settings: settings)
        for item in found { for problem in item.problems { print("Store   \(item.store.path): \(problem)") } }
        let reserve: UInt64 = 10_000_000_000  // keep the drive usable
        func limit(_ item: Stores.Found) -> UInt64 {
            let free = item.volume.free > reserve ? item.volume.free - reserve : 0
            let size = settings.stores.first { URL(fileURLWithPath: $0.path) == item.store }?.size
            return min(free, size ?? .max)
        }
        // The smallest copy worth making (512 records) that its drive has no room for, said with the numbers.
        var short: [String] = []
        let least = StoreBuilder.mirrorBytes(records: 512, model)
        func roomless(_ what: String, _ item: Stores.Found) -> Bool {
            guard item.volume.free < least + reserve else { return false }
            short.append(noRoom(what, need: least, keep: reserve, item: item, settings: settings))
            return true
        }
        let packBytes = StoreBuilder.packBytes(model)
        if !noPack, !found.contains(where: { $0.pack != nil && ($0.volume.disk != modelDisk || fullPack) }) {
            let candidates = found.filter {
                $0.volume.mounted && ($0.volume.disk != modelDisk || fullPack) && settings.holdsPacks($0.store)
            }
            if let target = candidates.first(where: { limit($0) >= packBytes }) {
                print("Pack    building a full expert pack on \(target.volume.name) (\(gigabytes(packBytes)); one read per expert)")
                try StoreBuilder.buildPack(model: model, folder: target.folder, progress: Terminal.steps("pack"))
                found = Stores.scan(model: model, settings: settings)
            } else if let target = candidates.first {
                print(
                    "Pack    not built: \(target.volume.name) has \(gigabytes(limit(target))) usable, the pack needs \(gigabytes(packBytes))"
                )
            }
        }
        let pack = found.first { $0.pack != nil }
        // Once enough of this model's routing is counted (`RoutingProfile`), further copies hold its most missed experts.
        let ranking = RoutingProfile.load(settings.routingFile(model))?.ranking()
        let recordBytes = UInt64(model.expertRecordBytes)
        // The hot copy is built while serving (`HotCopyKeeper`): further copies skip the experts it holds now.
        let hotFolder = Stores.plan(model: model, settings: settings, found: found).hotCopyFolder
        let hotKeys = Set(
            hotFolder.flatMap { OrigamiEngine.HotCopy.open(folder: $0, identity: Stores.hotCopyIdentity(model))?.keys } ?? [])
        if hotFolder != nil {
            print("Hot copy built while serving, after the first reply (see the dashboard's Health).")
        }
        // Several drives: every other store in `stores` off the model's drive gets a copy, the whole pack where it
        // fits, else the most-used experts after the hot copy's that fit; each read then goes to the least busy one.
        var drives: Set<String> = [modelDisk]
        for item in found where item.pack != nil || item.mirror != nil || item.further > 0 { drives.insert(item.volume.disk) }
        if let hotFolder, !hotKeys.isEmpty { drives.insert(Stores.volume(hotFolder).disk) }
        let listed = Set(settings.stores.map { URL(fileURLWithPath: $0.path, isDirectory: true) })
        for item in found
        where pack != nil && item.volume.mounted && listed.contains(item.store) && drives.insert(item.volume.disk).inserted {
            if limit(item) >= packBytes {
                print("Copy    building another full expert pack on \(item.volume.name) (\(gigabytes(packBytes))); reads use every drive")
                try StoreBuilder.buildPack(model: model, folder: item.folder, progress: Terminal.steps("copy"))
            } else {
                let hot = hotKeys
                let order = StoreBuilder.ranked(ranking, skipping: hot)
                // A further copy is a hot-copy store (`copy/`), its records padded to 16 KiB as the hot copy's are.
                let padded = (recordBytes + 16383) / 16384 * 16384
                let records = min(order.count, Int(limit(item) / padded))
                guard records >= 512 else {
                    if !roomless("Copy on \(item.volume.name)", item) {
                        print("Copy    skipped: \(item.volume.name) has too little room (\(gigabytes(limit(item))))")
                    }
                    continue
                }
                print(
                    "Copy    copying the next \(records) most-used experts to \(item.volume.name) (\(gigabytes(UInt64(records) * padded))); reads use every drive"
                )
                // Read as the 3.0 builder read its mirror's: from the pack, every read digest-checked (a bad pack record
                // is read from the checkpoint), each record then written with its CRC32.
                let source = try PackStore(model: model, pack: pack?.folder, integrity: .sha256)
                try source.enableConcurrentOriginalSpans(pool: SpanReadPool(threads: 16))
                let step = Terminal.steps("copy")
                var written: UInt64 = 0
                _ = try OrigamiEngine.HotCopy.write(
                    source: source, keys: Array(order.prefix(records)),
                    folder: item.folder.appendingPathComponent(PackStore.furtherCopyFolder, isDirectory: true),
                    identity: Stores.hotCopyIdentity(model),
                    wrote: { bytes in
                        written += UInt64(bytes)
                        step(Int(written / padded), records)
                    })
            }
        }
        found = Stores.scan(model: model, settings: settings)
        print("Drives  measuring the read speed of each drive holding the model or a copy…")
        Calibrations.measure(settings: settings, model: model, found: found, force: true) { print("        \($0)") }
        for item in found where item.pack != nil || item.mirror != nil || item.further > 0 {
            let kinds = [
                item.pack.map { _ in "pack" }, item.mirror.map { "copy (\($0.records.count) most-used experts)" },
                item.further > 0 ? "copy (\(item.further) most-used experts)" : nil,
            ].compactMap { $0 }
            print("Store   \(item.store.path): \(kinds.joined(separator: ", "))")
        }
        if !short.isEmpty { throw CLIError(short.joined(separator: ". ")) }
    }

    /// Why `what` was not built on `item`'s drive: what it needs, what is kept free and what is free, and what copies an
    /// earlier Origami version left there hold (this version never reads them; the dashboard's Models & Storage deletes them).
    static func noRoom(_ what: String, need: UInt64, keep: UInt64, item: Stores.Found, settings: Settings) -> String {
        let held = Storage.items(settings, found: ([], "setup does not search")).filter {
            $0.earlier != nil && $0.readable && Stores.volume(URL(fileURLWithPath: $0.path)).disk == item.volume.disk
        }.reduce(UInt64(0)) { $0 + $1.allocated }
        let drive = item.volume.name == "internal drive" ? "the internal drive" : item.volume.name
        return "\(what) not built: it needs \(gigabytes(need)) plus \(gigabytes(keep)) kept free, and \(drive) has "
            + "\(gigabytes(item.volume.free)) free"
            + (held > 0
                ? "; \(gigabytes(held)) there is held by copies an earlier Origami version made, which this version does not "
                    + "use: delete them in the dashboard's Models view" : "")
    }

    // MARK: serve

    /// The dashboard's Apply (`POST v1/config`): each value checked as `origami config set` checks it, all or nothing,
    /// written to the config file; `POST v1/reload` on the front then restarts the model with it.
    static func configWriter(_ settings: Settings, coordinateWiredMemory: Bool = false)
        -> @Sendable ([String: String]) throws -> [String: Any]
    {
        { values in
            let allowed =
                [
                    "model", "context", "expert_bits", "memory_limit", "expert_memory", "hot_copy", "context_dir", "quality",
                    "reasoning_effort",
                    "stores", "hot_copy_dir", "models_dirs", "listen", "lan_key", "sampling", "sampling_override",
                    "memory_auto_measure",
                ] + SamplingDefaults.keys
            var stored: [String: Any?] = [:]
            for (key, text) in values.sorted(by: { $0.key < $1.key }) {
                guard allowed.contains(key) else { throw ServingFailure("\(key) is not set from the dashboard") }
                // An empty model is the automatic choice, an empty sampling value the model's own: the setting is removed. A
                // retired `expert_bits` is only ever removed (the Clear of a setting the model does not use).
                if (["model", "expert_bits"] + SamplingDefaults.keys).contains(key) && text.isEmpty {
                    stored.updateValue(nil, forKey: key)
                    continue
                }
                // A relative path would resolve against the server's working folder (/ under launchd).
                let folders =
                    ["stores", "models_dirs"].contains(key)
                    ? ((try? JSONDecoder().decode([String].self, from: Data(text.utf8)))
                        ?? text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
                    : ["context_dir", "hot_copy_dir"].contains(key) && !["auto", "off"].contains(text) ? [text] : []
                if let relative = folders.first(where: { !$0.hasPrefix("/") && !$0.hasPrefix("~") }) {
                    throw ServingFailure("\(key) needs full paths, such as /Volumes/Fast/origami (not \(relative))")
                }
                do { stored.updateValue(try Settings.storedValue(key, text, current: settings), forKey: key) } catch let error as CLIError {
                    throw ServingFailure(error.description)
                }
            }
            // Checked against the model these settings run (the one in this save, else the configured one).
            var target = settings
            if let model = stored["model"] { target.model = model as? String }
            if !HotCopy.sameDriveExperiment, let dir = stored["hot_copy_dir"].flatMap({ $0 }) as? String,
                let gguf = FamilySetup.gguf(target),
                Stores.volume(gguf).disk == Stores.volume(URL(fileURLWithPath: dir)).disk
            {
                throw ServingFailure(HotCopy.onModelDrive, status: 409)
            }
            // Validate the entire save before requesting administrator authorization or changing either setting.
            let previousModels = try ModelRegistry.entries(settings)
            var ceilingRaised = false
            if coordinateWiredMemory {
                guard let text = values["memory_limit"] else { throw ServingFailure("Choose a memory limit") }
                guard !["flag", "env"].contains(settings.sources["memory_limit"]) else {
                    throw ServingFailure("The memory limit is set by the command or environment that started Origami", status: 409)
                }
                if text != "auto", let bytes = Settings.parseBytes(text) {
                    let snapshot = WiredMemory.snapshot()
                    let plan = try WiredMemoryPolicy.plan(
                        requestedBudgetBytes: bytes, physicalMemoryBytes: snapshot.physicalMemoryBytes,
                        currentLimitMiB: snapshot.currentLimitMiB, defaultLimitBytes: snapshot.metalRecommendedBytes)
                    ceilingRaised = try WiredMemory.ensureCeiling(for: plan).raised
                }
            }
            let saved = stored.keys.sorted()
            do {
                try modelSettings(&stored, target: target)
                // What Origami left in the folders these named stays on the storage map.
                if !Set(saved).isDisjoint(with: ["stores", "hot_copy_dir", "context_dir"]) { Storage.remember(settings) }
                // A level means what it says: knobs set alone before are reset with it.
                if values["quality"] != nil { for knob in Settings.qualityKnobs { stored.updateValue(nil, forKey: knob) } }
                stored.merge(settings.lanKeyFor(stored)) { $1 }
                try Settings.write(stored, to: settings.configFile)
            } catch {
                let saveError = error
                do { try ModelRegistry.update(settings) { $0 = previousModels } } catch {
                    throw ServingFailure(
                        "Settings could not be saved or fully restored: \(saveError). Review the saved values before reloading."
                            + (ceilingRaised ? " The system GPU ceiling was raised; the running budget is unchanged." : ""), status: 500)
                }
                if ceilingRaised {
                    throw ServingFailure(
                        "The system GPU ceiling was raised, but settings could not be saved: \(saveError). The running budget is unchanged; retry the save.",
                        status: 500)
                }
                throw saveError
            }
            return [
                "saved": saved, "config_file": settings.configFile.path, "restart": true,
                "overridden": saved.filter { ["flag", "env"].contains(settings.sources[$0]) },
            ]
        }
    }

    /// After a `listen` or key change: where other devices open the dashboard, with the key, and when it applies.
    static func printLAN(flags: [String: [String]]) throws {
        let fresh = try Settings.load(flags: flags)
        guard fresh.listenLAN else { return print("Only this Mac reaches Origami (listen local) once origami serve restarts.") }
        let urls = Access.urls(port: fresh.port)
        print(
            "Other devices on this network open " + (urls.first ?? "this Mac's address") + (fresh.remoteKey.map { "#key=" + $0 } ?? "")
                + (urls.count > 1 ? " (or " + urls.dropFirst().joined(separator: ", ") + ")" : ""))
        print(
            fresh.remoteKey == nil
                ? "They need no key (lan_key off): anything on the network can use Origami."
                : "They need the key after #key= (\(fresh.apiKey != nil ? "api_key" : "lan_key")); programs on this Mac need \(fresh.apiKey != nil ? "it too (api_key)" : "none")."
                    + " The key travels in clear over plain HTTP: for TLS, see docs/integration.md#tls."
        )
        print("This takes effect when origami serve restarts; the dashboard on this Mac shows it with a QR code under Connect.")
    }

    /// A named model keeps its own width and sampling (`Settings.modelKeys` in its entry, which beat the file's keys): a
    /// change of them in `changes` goes there; independent global defaults remain in the file.
    static func modelSettings(_ changes: inout [String: Any?], target: Settings) throws {
        let keys = Settings.modelKeys.filter(changes.keys.contains)
        guard !keys.isEmpty, let name = target.model, ModelRegistry.entry(name, settings: target) != nil else { return }
        try ModelRegistry.update(target) { entries in
            for i in entries.indices where entries[i]["name"] as? String == name {
                for key in keys { entries[i][key] = (changes[key] ?? nil) ?? NSNull() }
            }
        }
        for key in keys { changes.removeValue(forKey: key) }
    }

    /// `serve`: the front that lists every model and runs the one asked for (`ModelFront`); with `single`
    /// (`--single-model`, what the front starts for each model) this one model in this process.
    static func serve(flags: [String: [String]], single: Bool) async throws {
        let settings = try Settings.load(flags: flags)
        guard single else { return try await ModelFront.serve(settings: settings, flags: flags) }
        // A GPU that stopped answering (`GPUWait`): its buffers may still be written, so this process ends once the
        // failed request has had its reply; the front starts the model again for the next request.
        GPUWait.log = { FileHandle.standardError.write(Data("\(ISO8601DateFormatter().string(from: Date())) origami: \($0)\n".utf8)) }
        GPUWait.onLost = { reason in
            FileHandle.standardError.write(Data("origami: \(reason); restarting the model\n".utf8))
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                // An exit that waits on the stuck GPU work is cut short.
                DispatchQueue.global().asyncAfter(deadline: .now() + 10) { _exit(75) }
                exit(75)
            }
        }
        // A refresh of this model that has not finished (`origami refresh`): its copies are between two identities.
        if let pending = ModelRefresh.unfinished(data: settings.dataDir), Models.locate(settings.model, settings: settings)?.path == pending
        {
            throw CLIError(
                "`origami refresh` of \(pending) is unfinished: run `origami refresh --model \(pending)` to finish it, then serve the model."
                    + " If it cannot finish (a file it cannot write), delete \(ModelRefresh.journalURL(data: settings.dataDir).path) to give it up:"
                    + " the files keep their bytes; copies made for the files already rewritten are rebuilt")
        }
        // Started by a front that has gone (killed): stop rather than hold the model's memory.
        let parent = getppid()
        // A GLM or DeepSeek GGUF, given as a path or a registered name, runs on its family engine.
        if let spec = settings.model {
            let path =
                Models.isGGUF(Settings.expand(spec))
                ? Settings.expand(spec)
                : (ModelRegistry.entry(spec, settings: settings)?["path"] as? String)
                    ?? (Settings.expand(spec).lowercased().hasSuffix(".gguf") ? Settings.expand(spec) : nil)
            if let path, path.lowercased().hasSuffix(".gguf"), !FileManager.default.fileExists(atPath: path) {
                let why = missingModel(path, settings: settings)
                throw CLIError(why.prefix(1).uppercased() + why.dropFirst())
            }
            if let path, Models.isGGUF(path), let family = ModelFamily.detect(gguf: try GGUFSource(URL(fileURLWithPath: path)).metadata),
                family != .qwen
            {
                return try await serveFamily(settings: settings, flags: flags, gguf: URL(fileURLWithPath: path), parent: parent)
            }
        }
        let model = try model(settings)
        let host = try OrigamiHost.current()
        let events = EventLog()
        let log: @Sendable (String) -> Void = { line in
            let stamp = ISO8601DateFormatter().string(from: Date())
            FileHandle.standardError.write(Data("\(stamp) origami: \(line)\n".utf8))
            events.add(line)
        }
        let found = Stores.scan(model: model, settings: settings)
        // Drives not measured yet (or stale) are measured now, before the server listens: never during a request.
        Calibrations.measure(settings: settings, model: model, found: found, log: log)
        let storage = Stores.plan(model: model, settings: settings, found: found)
        log("model \(model.displayName) at \(model.directory.path)")
        for item in found where !item.problems.isEmpty { log("store \(item.store.path) skipped: \(item.problems.joined(separator: "; "))") }
        if let note = settings.expertBitsNote { log("warning: " + note) }
        if let pack = storage.packFolder { log("pack \(pack.path)" + (storage.packSharesModelDrive ? " (same drive as the model)" : "")) }
        if let hot = storage.hotCopyFolder { log("hot copy \(hot.path) (built and kept while serving)") }
        for copy in storage.copyFolders { log("expert copy \(copy.path)") }
        // Tuned values for this model and context on this Mac (`origami tune`), and what this Mac's memory taught.
        let tuning = Tuning.load(
            settings, model: model.fingerprint, context: settings.context, gpu: host.gpuName,
            drives: [model.directory, storage.packFolder, storage.hotCopyFolder].compactMap { $0 })
        let history = Tuning.memoryHistory(settings)
        // A measured memory limit (`origami tune --memory`): shown, and without memory_limit served with unless stale.
        nonisolated(unsafe) let measuredMemory = Tuning.memoryAdvice(settings, key: tuning.key, history: history)
        let memoryLimit = Tuning.memoryLimit(settings, advice: measuredMemory)
        log(Tuning.memoryLine(settings, advice: measuredMemory))
        // The tier serving `context`: in memory when this Mac has room, the file tier above (the plan's own choice).
        let contextPlan = OrigamiPlan.contextPlan(
            model: model, host: host, context: settings.context, memoryLimit: memoryLimit, expertMemory: settings.expertMemory,
            storage: storage, promptCache: settings.promptCacheRoot)
        if let note = contextPlan.note { log(note) }
        if let note = ExpertVariant.note(model) {
            log("experts: \(note)")
            if let line = ExpertVariant.seedCounts(settings.model, settings: settings) { log(line) }
        } else if storage.packFolder == nil && storage.hotCopyFolder == nil {
            log("no expert copies: reading the checkpoint directly (run `origami setup` for faster reads)")
        }
        if storage.packFolder != nil && !storage.packSharesModelDrive && storage.calibration == nil {
            log("drive speeds not measured yet; reads split evenly between the drives until they are")
        }

        let context = contextPlan.tier
        var http = LocalHTTPConfiguration()  // buffers sized for the tier; requests are held to the user's context
        http.port = UInt16(settings.port)
        http.applyContextLimits(context)
        http.model = ModelFront.listedID(settings)
        http.log = log
        http.apiKey = settings.apiKey
        http.lan = settings.listenLAN
        if let effort = settings.reasoningEffort { http.defaults.reasoningEffort = effort }
        http.defaults.maxThinkingTokens = settings.maxThinkingTokens
        http.defaults.recommendedSampling = SamplingDefaults.recommended(
            folder: model.directory, gguf: model.ggufExperts?.files.first?.path)
        http.defaults.sampling = settings.sampling(http.defaults.recommendedSampling)
        http.defaults.samplingOverride = settings.samplingOverride
        // The server's buffers are counted in the engine plan; this ledger only bounds them.
        let source = StatusSource(settings: settings, model: model, host: host, found: found, storage: storage, events: events)
        http.observationTags = try observationTags(
            settings, version: version, identity: model.fingerprint, context: context,
            startupTuning: ["overrides": tuning.tuned.wire],
            tuningScope: "Qwen startup tuning overrides; settings/environment may override; excludes deferred plan and governor decisions")
        if tuning.state != "none" {
            log(
                "tuning: \(tuning.state)" + (tuning.reason.map { " (\($0))" } ?? "")
                    + (tuning.tuned.wire.isEmpty ? "" : ", applying \(tuning.tuned.wire)"))
        }
        let keepAliveSet =
            settings.gpuKeepAlive != nil || (ProcessInfo.processInfo.environment["ORIGAMI_EXECUTION"] ?? "").contains("\"gpu_keepalive\"")
        let execution =
            (try? JSONSerialization.jsonObject(with: Data((ProcessInfo.processInfo.environment["ORIGAMI_EXECUTION"] ?? "{}").utf8)))
            as? [String: Any]
        let prefillScopeSet = (execution?["prefill"] as? [String: Any])?["read_scope"] != nil
        http.statusReport = {
            var report = source.report()
            guard let loaded = source.loadedRuntime else { return report }
            if let memory = report["memory"] as? [String: Any] {
                report["memory"] = memory.merging(Self.cutReport(loaded.governor, usable: memory["budget_bytes"] as? UInt64)) { $1 }
            }
            let on = loaded.plan.configuration.gpuKeepAlive
            var knobs = [
                "gpu_keepalive": Tuning.knob(
                    on, source: keepAliveSet ? "setting" : tuning.tuned.gpuKeepAlive != nil ? "tuned" : "default", default: true)
            ]
            if loaded.backend.tunesPrefillScope {
                knobs["qwen_prefill_scope"] = Tuning.knob(
                    loaded.backend.prefillReadScope,
                    source: prefillScopeSet ? "setting" : tuning.tuned.qwenPrefillScope != nil ? "tuned" : "default",
                    default: loaded.backend.prefillReadScopeMaximum
                )
                .merging(["maximum": loaded.backend.prefillReadScopeMaximum, "step": 256]) { $1 }
            }
            report["tuning"] = tuning.report(knobs: knobs)
            report["memory_learned"] = history.summary(
                cap: loaded.plan.memory.cap, minimum: loaded.plan.memory.minimum, applied: memoryLimit == nil,
                advice: measuredMemory ?? loaded.governor.limitAdvice
            ).merging([
                // As a memory_limit (`tune --memory` tries it), not the owners' budget the governor keeps.
                "minimum_bytes": loaded.plan.minimumLimit,
                "in_force": Tuning.memoryLine(settings, advice: measuredMemory),
                "auto_measure": Tuning.autoMeasureLine(settings) ?? NSNull(),
            ]) { $1 }
            report["memory_pressure_cuts"] = loaded.governor.pressureCuts
            report["memory_pressure_cut_to"] = loaded.governor.lowestPressureCut ?? NSNull()
            report["memory_budget_deficit_gb_seconds"] = loaded.governor.budgetDeficitGBSeconds()
            report["disk_budget"] = DiskBudget.shared.report
            return report
        }
        http.actions["/v1/memory/learned"] = { _ in
            history.reset()
            return ["reset": true]
        }
        http.planReport = { try source.preview($0) }
        let hotReads = ReadsSince()
        let review = DriveReview()
        let driveReads = DriveReadLog(data: settings.dataDir)
        http.requestTags = { seconds in
            // Serving's reads against each measured drive: the model's, a pack's and every copy place (`DriveReview`).
            if let backend = source.loadedRuntime?.backend {
                review.qwen(
                    settings, backend: backend, model: model.directory, pack: storage.packFolder, online: storage.calibration != nil,
                    log: log)
            }
            StatusSource.recordReads(
                driveReads, model: storage.packFolder == nil ? model.directory : nil, reads: source.loadedRuntime?.backend.modelDriveReads,
                keeper: source.keeper)
            return [
                "hot_copy": hotReads.moved(source.loadedRuntime?.backend.storeReads.hot ?? 0),
                "prompt_restore": source.loadedRuntime?.backend.promptCache?.takeRestore() ?? NSNull(),
                "memory_budget_deficit_gb_seconds": source.loadedRuntime?.governor.budgetDeficitGBSeconds(last: seconds) ?? 0,
            ]
        }
        // The Experts card: a drive or size saved (also from a front that forwarded it) and applied at once.
        http.hotCopy = { body in
            guard let keeper = source.keeper else { throw ServingFailure("This model has no hot copy", status: 404) }
            // GGUF experts: their file's drive is refused, as a family model's is (no copy is built there).
            _ = try HotCopy.save(
                body, settings: (try? Settings.load(flags: flags)) ?? settings,
                model: model.ggufExperts == nil ? nil : model.expertsLocation, qwen: true)
            var fresh = (try? Settings.load(flags: flags)) ?? settings
            // A place added, removed or moved takes a restart (one keeper per place); a size changes at once.
            let restart = HotCopy.restartNeeded(fresh, started: settings)
            if restart { fresh.hotCopyMore = settings.hotCopyMore }
            keeper.trigger(places: HotCopy.qwenPlaces(fresh, model: model), force: body["build"] as? Bool == true)
            if body["keep_when_idle"] != nil {
                HotCopy.shareDrives(fresh, flags: flags, identity: Stores.hotCopyIdentity(model), keeper: keeper)
            }
            let rows = source.hotRows(fresh)
            return ["hot_copy": rows.first ?? NSNull(), "hot_copies": rows, "restart": restart]
        }
        http.configWriter = configWriter(settings)
        let deferred = DeferredBackend(version: version, context: min(settings.context, context))
        if let note = settings.qualityNote { log(note) }
        // Forgiving: a stored level this model does not offer serves as its nearest offered one.
        http.defaults.quality = settings.quality.offered(by: .qwen)
        if http.defaults.quality != settings.quality {
            log("quality \(settings.quality.name) is not offered by \(ModelFamily.qwen.displayName); serving \(http.defaults.quality.name)")
        }
        let sticky = StickyQuality(file: settings.dataDir.appendingPathComponent("fast-mode.json"))
        http.stickyQuality = { sticky.level($0) }
        http.prompts = ChatCommands.prompts(keeper: { source.loadedRuntime.map(RuntimeKeeper.init) })
        if settings.commands {
            http.commands = ChatCommands.handler(
                report: { source.report() }, serving: { deferred.status() },
                keeper: { source.loadedRuntime.map(RuntimeKeeper.init) }, sticky: sticky, defaultQuality: http.defaults.quality,
                family: .qwen)
        }
        http.defaults.commands = settings.commands
        http.defaults.statusMessages = settings.statusMessages
        if let folder = settings.requestLog {
            let recorder = try RequestRecorder(folder: folder)
            let experts = ExpertVariant.note(model)
            http.recordRequest = { row, units in recorder.record(experts.map { row.merging(["experts": $0]) { $1 } } ?? row, units: units) }
            log("request records in \(folder.path)")
        }
        let httpLedger = try MemoryLedger(ceiling: http.admittedBytes + 2, transientMargin: 1, availableFloor: 1)
        let server = try LocalHTTPServer(configuration: http, backend: deferred, ledger: httpLedger)
        do { try server.start() } catch {
            throw CLIError("Cannot listen on 127.0.0.1:\(settings.port) (\(error)); is another server using it? Pick one with --port.")
        }
        log("listening on http://127.0.0.1:\(server.boundPort ?? 0) — loading the model")

        let started = Date()
        let loader = Task.detached { () -> OrigamiRuntime.Loaded? in
            do {
                let loaded = try await OrigamiRuntime.load(
                    model: model, host: host, storage: storage, context: settings.context,
                    memoryLimit: memoryLimit, expertMemory: settings.expertMemory, tier: context, inMemory: !contextPlan.paged,
                    allowFast: settings.allowFast,
                    promptCacheRoot: settings.promptCacheRoot.appendingPathComponent(
                        "\(model.fingerprint.prefix(12))-\(context)", isDirectory: true),
                    version: version, routingFile: settings.routingFile(model),
                    mtpWeights: settings.mtpWeights, promptCacheEntries: settings.promptCacheEntries,
                    gpuKeepAlive: settings.gpuKeepAlive ?? tuning.tuned.gpuKeepAlive,
                    qwenPrefillScope: tuning.tuned.qwenPrefillScope,
                    memoryHistory: history, learnsMemory: memoryLimit == nil,
                    log: log)
                // The hot copy: attached now when one is finished, then built, grown and attached between replies.
                source.keeper = try? HotCopy.qwenKeeper(
                    settings, model: model, pack: storage.packFolder, backend: loaded.backend, event: log)
                if let keeper = source.keeper {
                    HotCopy.shareDrives(settings, flags: flags, identity: Stores.hotCopyIdentity(model), keeper: keeper)
                }
                deferred.ready(loaded.backend)
                source.loaded(loaded)
                let memory = loaded.plan.memory
                let served = min(settings.context, context)
                log(
                    String(
                        format: "ready in %.0f s: %@ context, memory %@ now (up to %@), %@", Date().timeIntervalSince(started),
                        served % 1024 == 0 ? "\(served / 1024)K" : "\(served)-token",
                        gigabytes(memory.start), gigabytes(memory.cap),
                        settings.allowFast
                            ? "quality \(settings.display("quality")) unless a request chooses another" : "exact only (allow_fast false)"))
                return loaded
            } catch {
                log("the model failed to load: \(error)")
                deferred.failed(String(describing: error))
                return nil
            }
        }
        let previousINT = signal(SIGINT, SIG_IGN)
        let previousTERM = signal(SIGTERM, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        // A second Ctrl-C quits at once, without waiting for a request or a long save.
        interrupt.setEventHandler {
            if server.isStopped {
                log("second Ctrl-C: quitting without waiting; a save in progress is lost")
                exit(130)
            }
            server.stop()
        }
        terminate.setEventHandler { server.stop() }
        interrupt.resume()
        terminate.resume()
        defer {
            interrupt.cancel()
            terminate.cancel()
            signal(SIGINT, previousINT)
            signal(SIGTERM, previousTERM)
        }
        while !server.isStopped {
            if getppid() != parent {
                log("the process that started this server is gone; stopping")
                server.stop()
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        log("stopping (finishing prompt-cache saves; Ctrl-C again quits at once)")
        let joined = server.stopAndJoin()
        if !joined { log("request cleanup or metrics persistence was still pending after 30 s; finishing the saves already queued") }
        let loaded = await loader.value
        do { try loaded?.backend.drainSaves() }  // prompt-cache saves still being written
        catch { log("a prompt-cache save did not finish: \(error)") }
        loaded?.backend.flushRouting()
        source.keeper?.stop(waiting: 3)
        withExtendedLifetime(loaded) {}
        if !joined { throw CLIError("Request cleanup or metrics persistence was still pending at shutdown") }
    }

    /// `serve` for a GLM-5.3-Flash or DeepSeek-V4.1-Flash GGUF: the family engine reads the file as stored (dense
    /// weights resident, routed experts read on demand into the expert memory); the tokenizer and chat template
    /// come from the GGUF.
    /// The epoch of a family model's saved prompts (`PromptCache`): this build, the hot copy's identity (the GGUF's key and
    /// the expert representation), a resident pack's, the GGUF's tensor table, the context and the decode path
    /// (`ORIGAMI_FAMILY_GROUPED_DECODE=0`: no check shows the two give the same bits end to end, so states never cross it).
    static func familyEpoch(_ gguf: GGUFSource, identity: String, resident: String, context: Int) throws -> String {
        let table = gguf.tensors.values.sorted { $0.name < $1.name }.map { "\($0.name):\($0.type):\($0.file):\($0.offset):\($0.length)" }
            .joined(separator: ",")
        let path = FamilyRunner.groupsDecodeExperts ? "" : "|ungrouped-decode"
        return OrigamiRuntime.hex(
            SHA256.hash(data: Data((try OrigamiRuntime.executableDigest() + identity + resident + table + "\(context)" + path).utf8)))
    }

    static func serveFamily(settings: Settings, flags: [String: [String]], gguf url: URL, parent: pid_t) async throws {
        let initialSwapouts = swapoutBytes()
        let events = EventLog()
        let log: @Sendable (String) -> Void = { line in
            FileHandle.standardError.write(Data("\(ISO8601DateFormatter().string(from: Date())) origami: \(line)\n".utf8))
        }
        // What the dashboard's events list shows: the lines a person acts on, as they happen.
        let event: @Sendable (String) -> Void = { line in
            log(line)
            events.add(line)
        }
        let started = Date()
        let gguf = try ExpertVariant.source(url, settings: settings)
        let geometry = try ModelGeometry(gguf: gguf)
        if let problem = ModelCompatibility.familyProblem(gguf, geometry) {
            throw CLIError("Origami cannot run \(url.lastPathComponent): \(problem)")
        }
        if let note = settings.qualityNote { log(note) }
        let support = settings.dataDir.appendingPathComponent("support/\(FamilySupport.key(gguf))", isDirectory: true)
        try FamilySupport.prepare(url, into: support)
        let tokenizer = try await TokenizerStack(modelDirectory: support)
        guard let device = MTLCreateSystemDefaultDevice() else { throw CLIError("No Metal device") }
        let residentPack = try ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_RESIDENT_Q4_PACK"].map {
            try FamilyResidentPack.open(URL(fileURLWithPath: $0, isDirectory: true), gguf: gguf, geometry: geometry)
        }
        let weights = try FamilyWeights(gguf: gguf, device: device, residentPack: residentPack)
        if let residentPack {
            log(
                "EXPERIMENTAL non-exact resident Q4 weights: \(OrigamiPlan.gigabytes(UInt64(residentPack.savedBytes))) fewer payload bytes; answer quality is not validated"
            )
        }
        let context = min(settings.context, geometry.context)
        log(
            "\(geometry.family.displayName) from \(url.path): \(geometry.layers) layers, \(geometry.experts) experts (\(geometry.expertsPerToken) per token)"
        )
        // A variant's (`model add <experts.gguf> --base`) experts are read from their file: the hot copy's drive is judged by it.
        let expertsNote = ExpertVariant.note(gguf)
        let expertsAt = gguf.expertFiles.first.map { URL(fileURLWithPath: $0) } ?? url
        if let expertsNote { event("experts: \(expertsNote)") }
        // Expert memory: the setting, else what memory_limit leaves, else what this Mac has free once the rest is loaded.
        if let note = settings.expertBitsNote { event("warning: " + note) }
        // The context's per-position state: in memory while it takes at most a quarter of what Origami may use on this Mac,
        // else in files on the context drive (with room for it and 10 GB more), which macOS pages like any file.
        let contextFolder = settings.promptCacheRoot.appendingPathComponent("family-context", isDirectory: true)
        let onDisk: (UInt64) -> Bool = { [setting = settings.contextOnDisk] in setting ?? FamilyStatus.contextOnDisk($0) }
        // The memory governor's ledger: memory_limit, else 70% of the Mac's memory, with OrigamiPlan's floor left free.
        let physical = ProcessInfo.processInfo.physicalMemory
        let hostCeiling = OrigamiPlan.memoryCeiling(host: try OrigamiHost.current())
        // A measured memory limit (`origami tune --memory`, keyed as `Tuning.load` keys it): shown, and without memory_limit
        // served with unless stale.
        let history = Tuning.memoryHistory(settings)
        let tuningModel =
            residentPack.map { OrigamiRuntime.hex(SHA256.hash(data: Data((FamilySupport.key(gguf) + $0.representationIdentity).utf8))) }
            ?? FamilySupport.key(gguf)
        nonisolated(unsafe) let measuredMemory = Tuning.memoryAdvice(
            settings, key: Tuning.key(model: tuningModel, context: context), history: history)
        let memoryLimit = Tuning.memoryLimit(settings, advice: measuredMemory)
        log(Tuning.memoryLine(settings, advice: measuredMemory))
        let ceiling = min(memoryLimit ?? physical / 10 * 7, hostCeiling)
        let floor = physical - WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: physical)
        let ledger = try MemoryLedger(
            ceiling: ceiling, transientMargin: 1, availableFloor: floor,
            budgetMode: memoryLimit == nil ? .automatic : .explicit)
        let model = try makeFamilyModel(
            weights, expertBudget: settings.expertMemory, memoryLimit: ceiling, context: context,
            contextFile: { bytes in
                guard onDisk(bytes) else { return nil }
                guard (try? FileManager.default.createDirectory(at: contextFolder, withIntermediateDirectories: true)) != nil,
                    Stores.volume(contextFolder).free >= bytes + 10_000_000_000
                else {
                    event(
                        "context: \(OrigamiPlan.gigabytes(bytes)) would go to disk, but \(contextFolder.path) has too little room; kept in memory"
                    )
                    return nil
                }
                event(
                    "context: \(OrigamiPlan.gigabytes(bytes)) kept in files in \(contextFolder.path) ("
                        + (settings.contextOnDisk == true ? "context_on_disk" : "more than a quarter of what Origami may use here") + ")")
                return contextFolder
            }, ledger: ledger, log: log)
        // The hot copy: the most-missed expert records (counted while serving, kept per model in the data folder) copied
        // to the data folder's drive (or `hot_copy_dir`), read alongside the model's (`HotCopy.place` sizes it). Nothing is
        // built before the server listens: a finished copy is attached now, and `HotCopyKeeper` builds or grows one in the
        // background after replies, attaching it between them.
        let key = FamilySupport.key(gguf)
        let family = settings.dataDir.appendingPathComponent("family/\(key)", isDirectory: true)
        let usesFile = family.appendingPathComponent("uses.json")
        if let line = ExpertVariant.seedCounts(settings.model, settings: settings) { event(line) }
        let uses = OrigamiEngine.HotCopy.loadUses(usesFile)
        let misses = OrigamiEngine.HotCopy.loadUses(family.appendingPathComponent("misses.json"))
        // A hot copy holds records as the slots read them.
        let identity = LocalModelIdentity.familyRepresentation(source: key)
        let routed =
            (model as? FamilyRunner).map { runner in
                geometry.moeLayers.reduce(UInt64(0)) { $0 + UInt64((runner.experts.bank($1)?.bytes.reduce(0, +) ?? 0) * geometry.experts) }
            } ?? 0
        // One place per drive chosen (`hot_copy_dir`, then `hot_copy_more`), each with its own copy and keeper (`HotCopyPlaces`).
        let placeSettings = HotCopy.placeSettings(settings)
        let hotFolders = placeSettings.map { FamilySetup.hotFolder($0, key: key) }
        let hotFolder = hotFolders[0]
        // The expert memory holds every routed expert (a Mac with more memory than the model's experts): no automatic copy.
        let resident = (model as? FamilyRunner).map { routed > 0 && $0.experts.budget >= routed } ?? false
        let places = HotCopy.places(settings, model: expertsAt, key: key, routed: routed, resident: resident)
        // A finished copy counts as room on its drive: else a copy that filled the drive would be removed at each start.
        var startCopies: [OrigamiEngine.HotCopy?] = []
        for (index, folder) in hotFolders.enumerated() {
            let found = OrigamiEngine.HotCopy.open(folder: folder, identity: identity, source: (model as? FamilyRunner)?.experts)
            switch places[index]({ $0.standardizedFileURL == folder.standardizedFileURL ? UInt64(found?.bytes ?? 0) : 0 }) {
            case .folder: startCopies.append(found)
            case .none(let state, let detail):
                startCopies.append(nil)
                // Disabled or temporarily unsuitable copies remain available for reuse or explicit deletion.
                if state == "not_connected" {
                    event("Hot copy: \(HotCopy.name(placeSettings[index].hotCopyDir ?? folder)) not connected.")
                } else {
                    log("hot copy\(index == 0 ? "" : " place \(index + 1)"): \(detail)")
                }
            }
        }
        // Drives not measured yet (or stale) are measured now, before the server listens: never during a request.
        // Seeded at a token's experts read at once, the copies' at the fastest place's rate.
        let speeds = FamilySetup.measureDrives(
            gguf, geometry: geometry, hot: hotFolder, more: Array(hotFolders.dropFirst()), settings: settings, log: log)
        let placeSpeeds = ([speeds.copy] + speeds.more).map { $0?.rate(depth: geometry.expertsPerToken) }
        let order = HotCopyPlaces.order(placeSpeeds)
        (model as? FamilyRunner)?.experts.seed(
            copy: order.lazy.compactMap { placeSpeeds[$0] }.first, model: speeds.model?.rate(depth: geometry.expertsPerToken),
            places: Dictionary(zip(hotFolders, placeSpeeds).compactMap { folder, rate in rate.map { (folder, $0) } }) { a, _ in a })
        let start = order.compactMap { startCopies[$0] }
        if !start.isEmpty, let runner = model as? FamilyRunner {
            runner.experts.attach(copies: start)
            for copy in start {
                log("hot copy: \(copy.keys.count) expert records (\(OrigamiPlan.gigabytes(UInt64(copy.bytes)))) from \(copy.folder.path)")
            }
        }
        let backend = FamilyServingBackend(
            model: model, tokenizer: tokenizer, stops: FamilySupport.stopTokens(gguf), version: version,
            fastAllowed: settings.allowFast)
        backend.log = log
        backend.event = event
        backend.usesFile = usesFile
        backend.earlierUses = uses
        backend.earlierMisses = misses
        let keeper = (model as? FamilyRunner).map { runner in
            HotCopyPlaces(
                source: runner.experts, identity: identity, places: places.indices.map { (places[$0], startCopies[$0]) },
                speeds: placeSpeeds, counts: (uses, misses),
                attach: { [weak backend] copies in backend?.attachHotCopies(copies, to: runner.experts) ?? false },
                idle: { [weak backend] in backend?.idle ?? false },
                withIdle: { [weak backend] work in try backend?.withIdleHotCopy(work) ?? false }, event: event)
        }
        backend.keeper = keeper
        if let keeper { HotCopy.shareDrives(settings, flags: flags, identity: identity, keeper: keeper) }
        // Tuned values for this model and context on this Mac (`origami tune`) where no ORIGAMI_FAMILY_* variable sets them.
        // Tuning truncates its model key: `tuningModel` hashes the whole representation before that truncation.
        let tuning = Tuning.load(
            settings, model: tuningModel, context: context, gpu: (try? OrigamiHost.current())?.gpuName ?? device.name,
            drives: [url, hotFolder])
        if tuning.state != "none" {
            log(
                "tuning: \(tuning.state)" + (tuning.reason.map { " (\($0))" } ?? "")
                    + (tuning.tuned.wire.isEmpty ? "" : ", applying \(tuning.tuned.wire)"))
        }
        let environment = ProcessInfo.processInfo.environment
        let fixed = Set(
            ["PREFETCH", "PREFETCH_DEPTH", "SPIN", "PREFILL_SLICES"].filter {
                environment["ORIGAMI_FAMILY_" + $0] != nil
            })
        // The GPU keep-alive (`gpu_keepalive`): the setting, else tuned, else auto (on outside Low Power Mode, read per reply).
        // Built either way, so a request can turn it on; one that cannot be made fails the start only when it is chosen.
        backend.keepAliveSetting = settings.gpuKeepAlive ?? tuning.tuned.gpuKeepAlive
        backend.keepAlive =
            backend.keepAliveSetting != false ? try RequestGPUKeepAlive(ledger: ledger) : try? RequestGPUKeepAlive(ledger: ledger)
        // Auto reads Low Power Mode per reply, so the status reads it too: off in Low Power Mode, not the default's "on".
        let keepAliveSource = settings.gpuKeepAlive != nil ? "setting" : tuning.tuned.gpuKeepAlive != nil ? "tuned" : "default"
        let keepAliveKnob = { [backend] in
            Tuning.knob(backend.keepAliveSetting ?? !backend.lowPowerMode(), source: keepAliveSource, default: true)
        }
        let knobs = (model as? FamilyRunner).map {
            Tuning.FamilyKnobs(runner: $0, fixed: fixed, tuned: tuning.tuned, keepAlive: keepAliveKnob)
        }
        if let runner = model as? FamilyRunner {
            if !fixed.contains("PREFETCH"), let value = tuning.tuned.familyPrefetch { runner.guessesNext = value }
            if !fixed.contains("PREFETCH_DEPTH"), let depth = tuning.tuned.familyPrefetchDepth {
                runner.experts.guessDepth = depth < 0 ? nil : depth
            }
            if !fixed.contains("SPIN"), let value = tuning.tuned.familySpin { FamilyExperts.spins = value }
            if let value = tuning.tuned.familyEarlyDelayedReads { runner.earlyDelayedReads = value }
            if let text = environment["ORIGAMI_FAMILY_PREFILL_SLICES"] {
                guard let value = Int(text), (1...4).contains(value) else { throw CLIError("ORIGAMI_FAMILY_PREFILL_SLICES must be 1 to 4") }
                runner.prefillSlices = value
            } else if let value = tuning.tuned.familyPrefillSlices {
                runner.prefillSlices = value
            }
        }
        // Automatic mode only: the budget this Mac's memory use taught (`MemoryHistory`); the pool starts there. A limit
        // (memory_limit or measured) always starts at itself.
        let learnsMemory = memoryLimit == nil
        let measured = settings.memoryLimit == nil && memoryLimit != nil
        let leastBytes = (model as? FamilyRunner)?.minimumLimit ?? 0
        if let runner = model as? FamilyRunner, learnsMemory,
            let learned = history.budget(cap: min(ceiling, ledger.maximumUsableBytes), minimum: leastBytes, ceiling: ceiling),
            learned + 250_000_000 < ledger.usableBytes
        {
            event(
                "Memory: starting at \(OrigamiPlan.gigabytes(learned)) instead of \(OrigamiPlan.gigabytes(ledger.usableBytes)), learned from this Mac's memory use."
            )
            ledger.setUsable(learned)
            runner.fitExperts()
        }
        // The empty pool warmed from the saved decode uses, most used first, until the first request (`FamilyServingBackend.warm`).
        if let runner = model as? FamilyRunner {
            backend.warm(runner.experts, keys: uses.sorted { ($0.value, $1.key) > ($1.value, $0.key) }.map(\.key)) { placed, seconds in
                if placed > 0 { log(String(format: "experts: %d records warmed from earlier replies in %.1f s", placed, seconds)) }
            }
            event(
                String(
                    format: "Expert cache target: %@ slots (%@), within the %@ budget.", runner.experts.slotCount.grouped,
                    OrigamiPlan.gigabytes(UInt64(runner.experts.poolBytes)),
                    settings.memoryLimit != nil ? "chosen" : measured ? "measured" : "automatic"))
        }
        // Counts kept from earlier runs: plan once now (the build waits for the warm-up and any request).
        keeper?.trigger()
        // Pressure or fresh swap temporarily lowers the chosen budget; automatic mode also follows free headroom.
        // After 20 quiet seconds it recovers, and the pool admits growth at the next safe layer boundary, even mid-reply.
        let governor = (model as? FamilyRunner).map { runner in
            MemoryGovernor(
                ledger: ledger,
                memory: OrigamiMemory(
                    cap: ceiling, start: ledger.usableBytes, floor: ledger.availableFloorBytes,
                    minimum: leastBytes),
                trim: { backend.trimExperts() }, log: log, history: history, learns: learnsMemory,
                initialSwapouts: initialSwapouts)
        }
        governor?.start()
        // Saved prompts in Qwen's prompt cache, one folder per epoch: this build, the GGUF (its tokenizer and template
        // included), the effective expert representation and the context. The support files written from the GGUF
        // at each start do not enter it.
        let epoch = try familyEpoch(gguf, identity: identity, resident: residentPack?.representationIdentity ?? "", context: context)
        let epochFolder = settings.promptCacheRoot.appendingPathComponent(String(epoch.prefix(16)), isDirectory: true)
        let cacheLedger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
        let cache = try PromptCache(
            root: epochFolder, epoch: epoch,
            entries: PromptCache.regularEntries(context: context, entries: settings.promptCacheEntries),
            // The slots' Qwen checkpoints are never written here; the smallest store holds the folders.
            entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2, ledger: cacheLedger, capacity: 8192,
            // Sibling epoch folders are any family model's: only this key's and context's are its other builds.
            sameModel: { folder in
                FamilySetup.epoch(folder).map { $0["family_key"] as? String == key && $0["context"] as? Int == context }
            })
        if let off = cache.savesOff { log("prompt cache: off (\(off))") }
        if let note = cache.savesNote { log("prompt cache: \(note)") }
        FamilySetup.markEpoch(epochFolder, model: ModelFront.listedID(settings), key: key, context: context)
        // Only the server that owns the data folder writes the usage counts back.
        if cache.shared { backend.usesFile = nil }
        cache.log = log
        backend.promptCache = cache
        backend.epoch = epoch
        var http = LocalHTTPConfiguration()
        http.port = UInt16(settings.port)
        http.applyContextLimits(context)
        http.log = log
        http.apiKey = settings.apiKey
        http.lan = settings.listenLAN
        http.model = ModelFront.listedID(settings)
        http.family = geometry.family
        http.observationTags = try observationTags(
            settings, version: version, identity: identity, context: context,
            startupTuning: [
                "overrides": tuning.tuned.wire, "resolved_family_knobs": knobs?.report.mapValues { $0["value"] ?? NSNull() } ?? [:],
            ],
            tuningScope: "family startup tuning overrides and resolved tuner knobs; excludes per-request switches and governor decisions")
        http.configWriter = configWriter(settings)
        // `/v1/status` and `/v1/plan` as Qwen's server answers them, with where the experts are read from (`experts_from`).
        let status = FamilyStatus(
            settings: settings, model: model, backend: backend, url: url, gguf: gguf, geometry: geometry, keeper: keeper,
            events: events)
        // The Experts card: a drive or size saved (also from a front that forwarded it) and applied at once.
        http.hotCopy = { body in
            guard let keeper else { throw ServingFailure("This model has no hot copy", status: 404) }
            _ = try HotCopy.save(body, settings: (try? Settings.load(flags: flags)) ?? settings, model: expertsAt)
            var fresh = (try? Settings.load(flags: flags)) ?? settings
            // A place added, removed or moved takes a restart (one keeper per place); a size changes at once.
            let restart = HotCopy.restartNeeded(fresh, started: settings)
            if restart { fresh.hotCopyMore = settings.hotCopyMore }
            status.update(fresh)
            if body["build"] as? Bool == true { backend.releaseIdleHotCopyHold() }
            keeper.trigger(
                places: HotCopy.places(fresh, model: expertsAt, key: key, routed: routed, resident: resident),
                force: body["build"] as? Bool == true)
            if body["keep_when_idle"] != nil { HotCopy.shareDrives(fresh, flags: flags, identity: identity, keeper: keeper) }
            let rows = status.hotSources()
            return ["hot_copy": rows[0], "hot_copies": rows, "restart": restart]
        }
        http.statusReport = {
            var report = status.report()
            let memory = ledger.snapshot()
            let target = ledger.maximumUsableBytes
            let usable = memory.usableBytes ?? target
            report["memory"] = [
                "cap_bytes": ceiling, "budget_bytes": usable,
                "budget_mode": ledger.budgetMode.rawValue, "target_bytes": target,
                "temporarily_reduced": ledger.budgetMode == .explicit && usable < target,
                "reserved_bytes": memory.reserved, "footprint_bytes": memory.physicalFootprint,
                "available_bytes": memory.availableMemoryEstimate, "floor_bytes": ledger.availableFloorBytes,
            ].merging(Self.cutReport(governor, usable: usable)) { $1 }
            report["memory_admission"] = memoryAdmissionReport(memory)
            if let knobs { report["tuning"] = tuning.report(knobs: knobs.report) }
            report["memory_learned"] = history.summary(
                cap: ceiling, minimum: leastBytes, applied: learnsMemory, advice: measuredMemory ?? governor?.limitAdvice
            ).merging([
                "in_force": Tuning.memoryLine(settings, advice: measuredMemory),
                "auto_measure": Tuning.autoMeasureLine(settings) ?? NSNull(),
            ]) { $1 }
            report["memory_pressure_cuts"] = governor?.pressureCuts ?? 0
            report["memory_pressure_cut_to"] = governor?.lowestPressureCut ?? NSNull()
            report["memory_budget_deficit_gb_seconds"] = governor?.budgetDeficitGBSeconds() ?? 0
            report["disk_budget"] = DiskBudget.shared.report
            return report
        }
        http.actions["/v1/memory/learned"] = { _ in
            history.reset()
            return ["reset": true]
        }
        http.planReport = { try status.preview($0) }
        let hotReads = ReadsSince()
        let review = DriveSpeedReview()
        let experts = (model as? FamilyRunner)?.experts
        let driveReads = DriveReadLog(data: settings.dataDir)
        // The prompt cache locks its own state; a row reads its last restore once (`takeRestore`).
        nonisolated(unsafe) let restores = cache
        http.requestTags = { seconds in
            // The engine's smoothed rate of each measured drive against the measured range (one reader to eight): a
            // drive read one record at a time is no contradiction of its deeper rates.
            if let experts {
                let now = experts.driveBandwidth
                let ratio = { (rate: Double, speed: DriveSpeed) in rate / min(max(rate, speed.rate(depth: 1)), speed.bytesPerSecond.max()!)
                }
                // Each copy place from its own lane (`DriveReview.places`), the model's drive from its smoothed rate.
                Calibrations.review(
                    review, settings: settings,
                    drives: [(gguf.folder, speeds.model.map { ratio(now.model, $0) })].compactMap { url, ratio in ratio.map { (url, $0) } }
                        + DriveReview.places(experts.hotCopySet, settings), log: log)
            }
            StatusSource.recordReads(
                driveReads, model: url.deletingLastPathComponent(), reads: experts?.modelDriveReads, keeper: keeper)
            return [
                "hot_copy": hotReads.moved(status.hotReads), "prompt_restore": restores.takeRestore() ?? NSNull(),
                "memory_budget_deficit_gb_seconds": governor?.budgetDeficitGBSeconds(last: seconds) ?? 0,
            ]
        }
        if let effort = settings.reasoningEffort { http.defaults.reasoningEffort = effort }
        http.defaults.maxThinkingTokens = settings.maxThinkingTokens
        http.defaults.commands = settings.commands
        http.defaults.statusMessages = settings.statusMessages
        // Forgiving: a stored level this model does not offer serves as its nearest offered one.
        http.defaults.quality = settings.quality.offered(by: geometry.family)
        if http.defaults.quality != settings.quality {
            log("quality \(settings.quality.name) is not offered by \(geometry.family.displayName); serving \(http.defaults.quality.name)")
        }
        http.defaults.recommendedSampling = SamplingDefaults.recommended(gguf.metadata, splitModes: false)
        http.defaults.sampling = settings.sampling(http.defaults.recommendedSampling)
        http.defaults.samplingOverride = settings.samplingOverride
        let sticky = StickyQuality(file: settings.dataDir.appendingPathComponent("fast-mode.json"))
        http.stickyQuality = { sticky.level($0) }
        http.prompts = ChatCommands.prompts(keeper: { backend })
        if settings.commands {
            http.commands = ChatCommands.handler(
                report: { status.report() }, serving: { backend.status() }, keeper: { backend }, sticky: sticky,
                defaultQuality: http.defaults.quality, family: geometry.family)
        }
        if let folder = settings.requestLog {
            let recorder = try RequestRecorder(folder: folder)
            http.recordRequest = { row, units in
                recorder.record(expertsNote.map { row.merging(["experts": $0]) { $1 } } ?? row, units: units)
            }
            log("request records in \(folder.path)")
        }
        let httpLedger = try MemoryLedger(ceiling: http.admittedBytes + 2, transientMargin: 1, availableFloor: 1)
        let server = try LocalHTTPServer(configuration: http, backend: backend, ledger: httpLedger)
        do { try server.start() } catch {
            throw CLIError("Cannot listen on 127.0.0.1:\(settings.port) (\(error)); is another server using it? Pick one with --port.")
        }
        log(
            String(
                format: "ready in %.0f s on http://127.0.0.1:%d: %d-token context, quality %@", Date().timeIntervalSince(started),
                server.boundPort ?? 0, context,
                settings.allowFast ? "\(settings.display("quality")) unless a request chooses another" : "exact only (allow_fast false)"))
        let previousINT = signal(SIGINT, SIG_IGN)
        let previousTERM = signal(SIGTERM, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        interrupt.setEventHandler { server.stop() }
        terminate.setEventHandler { server.stop() }
        interrupt.resume()
        terminate.resume()
        defer {
            interrupt.cancel()
            terminate.cancel()
            signal(SIGINT, previousINT)
            signal(SIGTERM, previousTERM)
        }
        while !server.isStopped {
            if getppid() != parent {
                log("the process that started this server is gone; stopping")
                server.stop()
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        log("stopping")
        keeper?.stop(waiting: 3)
        if !server.stopAndJoin() { log("request cleanup or metrics persistence was still pending at shutdown") }
        withExtendedLifetime((model, governor)) {}
    }

    /// `model add` for a family engine's model: nil when `url` is not GLM or DeepSeek. A GGUF is checked (headers
    /// only) and registered as stored; a GLM safetensors checkpoint is converted once into a GGUF in `folder` (`--to`, else the models folder),
    /// its experts in Q2_K (`--expert-bits 2`, the default) or Q4_K (`--expert-bits 4`).
    static func addFamilyModel(
        _ url: URL, gguf: Bool, name given: String?, derived: (String) throws -> String, into folder: URL,
        flags: [String: [String]], yes: Bool, taken: (_ name: String, _ path: String) throws -> Void
    ) throws -> [String: Any]? {
        if gguf {
            let source = try GGUFSource(url)
            guard let family = ModelFamily.detect(gguf: source.metadata), family != .qwen else { return nil }
            let report = ModelCompatibility.check(gguf: source, source: url.path)
            guard report.verdict == .runs else { throw CLIError("Origami cannot run \(url.lastPathComponent): \(report.reason)") }
            let name = try given ?? derived(GGUFSource.modelPrefix(url.lastPathComponent).replacingOccurrences(of: ".gguf", with: ""))
            try taken(name, url.path)
            return ["name": name, "path": url.path, "family": family.rawValue, "added": ISO8601DateFormatter().string(from: Date())]
        }
        let config = (try? Data(contentsOf: url.appendingPathComponent("config.json"))).flatMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        }
        guard let config, let family = ModelFamily.detect(config: config), family != .qwen else { return nil }
        guard family == .glm else {
            throw CLIError(
                "Origami converts GLM-5.3-Flash checkpoints; for \(family.displayName) use a GGUF (`origami inspect --family` shows which run)"
            )
        }
        let converter = try Models.familyConverter(url, bits: flags["expert-bits"]?.last ?? "2")
        let type = converter.expertType
        let name = try given ?? derived(url.lastPathComponent + "-" + GGUFQuant.name(type).lowercased())
        let target = folder.appendingPathComponent(name + ".gguf")
        try taken(name, target.path)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let free = Stores.volume(folder).free
        let need = Models.conversionNeeds(UInt64(converter.outputBytes))
        print(
            "\(url.lastPathComponent) is \(family.displayName). Origami converts it once into a GGUF with \(GGUFQuant.name(type)) experts and Q8_0 dense "
                + "matrices: \(gigabytes(UInt64(converter.outputBytes))) in \(target.path). The source is only read.")
        guard free >= need else {
            throw CLIError(
                "Converting needs \(gigabytes(need)) free there, 10 GB to spare included (\(gigabytes(free)) now). "
                    + "Pass --to with a folder on a bigger drive: `origami model add <source> --to /Volumes/…`.")
        }
        guard Terminal.confirm("Convert it now? This reads the whole checkpoint once.", yes: yes) else {
            throw CLIError("Not added: \(url.lastPathComponent) was not converted.")
        }
        let steps = Terminal.steps("converting")
        let total = converter.outputBytes
        do {
            try interruptible { cancelled in try converter.convert(to: target, progress: { steps($0, total) }, cancelled: cancelled) }
        } catch let error as EngineError where error.code == "cancelled" {
            throw CLIError(
                "Stopped: the unfinished conversion \(target.lastPathComponent).partial was deleted (a GLM conversion cannot be resumed; the same command starts it again)"
            )
        }
        return [
            "name": name, "path": target.path, "family": family.rawValue, "converted_from": url.path,
            "added": ISO8601DateFormatter().string(from: Date()),
        ]
    }

    // MARK: status

    /// A cut in either mode for `/v1/status`'s `memory`: `temporarily_reduced` when the budget is below what the governor
    /// climbs back to (`recovery_target_bytes`: the learned budget in automatic mode, else the cap), and the last cut's
    /// reason and time (`last_cut`).
    static func cutReport(_ governor: MemoryGovernor?, usable: UInt64?) -> [String: Any] {
        guard let governor, let usable else { return [:] }
        return [
            "recovery_target_bytes": governor.target, "temporarily_reduced": usable < governor.target,
            "last_cut": governor.lastCut.map { ["reason": $0.reason, "at": ISO8601DateFormatter().string(from: $0.at)] } ?? NSNull(),
        ]
    }

    /// What runs now, from a server's `/v1/status` (`live`): the loaded model and its queue, memory now against its
    /// target with the cuts since load, the expert cache's hits and where misses were read, each copy's state, and the
    /// last request's prompt restore. None when no server answers.
    static func liveLines(_ live: [String: Any]) -> [String] {
        let state = live["status"] as? String ?? "unavailable"
        guard state != "unavailable" else { return [] }
        func percent(_ value: Any?) -> String { "\(Int((((value as? NSNumber)?.doubleValue ?? 0) * 100).rounded()))%" }
        func size(_ value: Any?) -> String { (value as? NSNumber).map { gigabytes($0.uint64Value) } ?? "?" }
        let model = (live["model"] as? [String: Any])?["name"] as? String ?? live["model"] as? String ?? "the model"
        let id = (live["model_id"] as? String).flatMap { $0 == ModelFront.setupID ? nil : $0 }
        let queue = live["queue"] as? [String: Any] ?? [:]
        let waiting = queue["queued"] as? Int ?? 0
        // `degraded` is the machine value; a person reads the dashboard's name for the state.
        let shown = state != "degraded" ? state : "drive gone"
        var lines = [
            (id.map { "\($0) (\(model))" } ?? model) + " loaded, \(shown)" + (queue["busy"] as? Bool == true ? ", replying" : "")
                + (waiting > 0 ? ", \(waiting) waiting" : "")
        ]
        switch state {
        case "loading":
            let loading = live["loading"] as? [String: Any]
            return [
                "loading \(model)" + ((loading?["seconds"] as? Int).map { " for \($0) s" } ?? "")
                    + ((loading?["line"] as? String).map { ": " + $0 } ?? "")
            ]
        case "stopped", "stopping": return ["\(state): no model loaded (`origami start` loads \(model))"]
        case "failed": return ["\(model) did not start: \(live["failure"] as? String ?? "see the server log")"]
        default: break
        }
        if let degraded = live["degraded"] as? String { lines.append(degraded + " (`origami stop` then `origami start` reloads it)") }
        if let memory = live["memory"] as? [String: Any], let budget = memory["budget_bytes"] {
            let cuts = live["memory_pressure_cuts"] as? Int ?? 0
            let cut = memory["last_cut"] as? [String: Any]
            let since = (cut?["at"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) }
            lines.append(
                "memory \(size(budget)) of \(size(memory["recovery_target_bytes"] ?? memory["target_bytes"]))"
                    + (memory["temporarily_reduced"] as? Bool == true ? " (cut for now)" : "") + "; "
                    + (cuts == 0 ? "no cuts since load" : "\(cuts) cut\(cuts == 1 ? "" : "s") since load")
                    + ((cut?["reason"] as? String).map { ", the last \(since.map { AutoMeasure.shown($0) } ?? ""): \($0)" } ?? ""))
        }
        let from = live["experts_from"] as? [String: Any] ?? [:]
        let split = from["split"] as? [String: Any] ?? [:]
        let sources = from["sources"] as? [[String: Any]] ?? []
        if (split["reads"] as? Int ?? 0) > 0 {
            let copies = sources.filter {
                ["hot", "more", "copy", "pack"].contains($0["role"] as? String ?? "") && $0["read_share"] != nil
            }
            lines.append(
                "expert reads: memory \(percent(split["memory"]))"
                    + copies.map { " · \($0["drive"] as? String ?? "copy") copy \(percent($0["read_share"]))" }.joined()
                    + " · model drive \(percent(split["model_drive"]))")
        }
        let copies = sources.filter { ["hot", "more"].contains($0["role"] as? String ?? "") }
        if !copies.isEmpty {
            lines.append(
                "copies: "
                    + copies.map {
                        "\($0["drive"] as? String ?? "?") \($0["state"] as? String ?? "?")"
                            + (($0["bytes"] as? NSNumber).map { " " + gigabytes($0.uint64Value) } ?? "")
                    }.joined(separator: ", "))
        }
        if let last = ((live["requests"] as? [String: Any])?["recent"] as? [[String: Any]])?.last,
            let restore = last["prompt_restore"] as? [String: Any], let wanted = restore["wanted"] as? Int
        {
            let restored = restore["restored"] as? Int ?? 0
            lines.append(
                "last request: \(restored) of \(wanted) prompt tokens restored" + ((restore["reason"] as? String).map { " (\($0))" } ?? ""))
        }
        if let last = ((live["requests"] as? [String: Any])?["recent"] as? [[String: Any]])?.last {
            let verdict = SlowReason.verdict(request: last, status: live)
            if verdict != "nothing unusual" { lines.append("last request: " + verdict) }
        }
        return lines
    }

    static func status(flags: [String: [String]], json: Bool) throws {
        let (lines, report) = try statusReport(flags: flags)
        // Whether it runs, what it serves and with how much memory first; each row keeps its indented lines.
        var blocks: [[String]] = []
        for line in lines {
            if line.hasPrefix(" "), !blocks.isEmpty { blocks[blocks.count - 1].append(line) } else { blocks.append([line]) }
        }
        let rank = ["Server": 0, "Live": 1, "Model": 2, "Memory": 3]
        let ordered = blocks.enumerated().sorted {
            let (a, b) = (rank[String($0.element[0].prefix { $0 != " " })] ?? 4, rank[String($1.element[0].prefix { $0 != " " })] ?? 4)
            return a != b ? a < b : $0.offset < $1.offset
        }
        emit(ordered.flatMap(\.element), report, json)
    }

    static func statusReport(flags: [String: [String]]) throws -> (lines: [String], report: [String: Any]) {
        let settings = try Settings.load(flags: flags)
        var report: [String: Any] = ["version": version, "config_file": settings.configFile.path, "data_dir": settings.dataDir.path]
        report.merge(Tuning.reportFields(settings)) { $1 }
        let history = Tuning.memoryHistory(settings)
        // shortcut: without a loaded model the limit's margin over its budget is unknown, so this suggestion can be up to
        // that margin (Qwen: 1 GB) lower than the server's; pass the plan's budget cap if that ever matters.
        // The entry the running server looks up (its tuning key, which a move or refresh changes); with no server, the
        // newest for the configured model and context.
        let live = DashboardState.status(port: UInt16(clamping: settings.port), key: settings.apiKey)
        let running = (live["tuning"] as? [String: Any])?["key"] as? String
        let measured = Tuning.memoryAdvice(settings, key: running, history: history)
        let line = Tuning.memoryLine(settings, advice: measured)
        // The loaded model's own line when a server answers (it may not be the configured model).
        let auto = (live["memory_learned"] as? [String: Any])?["auto_measure"] as? String ?? Tuning.autoMeasureLine(settings)
        let learned = history.summary(
            cap: .max, minimum: 0, applied: Tuning.memoryLimit(settings, advice: measured) == nil,
            advice: measured ?? settings.memoryLimit.flatMap { history.limitAdvice(limit: $0, cap: $0, minimum: 0) }
        ).merging(["in_force": line, "auto_measure": auto ?? NSNull()]) { $1 }
        report["memory_learned"] = learned
        var lines = ["Memory    " + line.dropFirst("memory: ".count)]
        if let auto { lines.append("Memory    " + auto.dropFirst("memory: ".count)) }
        let now = liveLines(live)
        lines += now.map { "Live      " + $0 }
        if !now.isEmpty { report["live"] = now }
        if let advice = (learned["limit_advice"] as? [String: Any])?["text"] as? String { lines.append("Memory    " + advice) }
        let host = try? OrigamiHost.current()
        if let host {
            lines.append("Mac       \(host.gpuName), \(ram(host.physicalMemory)) memory")
            lines += host.notes.map { "          \($0)" }
            report["mac"] = ["gpu": host.gpuName, "memory": host.physicalMemory]
        }
        report["drive_speeds"] = Calibrations.report(settings)
        lines += Calibrations.lines(settings).map { "Drive     " + $0 }
        // A retired `expert_bits` still set, and the smaller experts an earlier Origami made, which nothing reads now.
        if let note = settings.expertBitsNote { lines.append("Settings  " + note) }
        report["expert_bits_retired"] = settings.expertBitsNote ?? NSNull()
        let retired = Stores.retiredExperts(settings)
        for item in retired {
            lines.append(
                "Unused    \(item.path) (\(gigabytes(item.bytes))): smaller experts an earlier Origami made; nothing reads them now")
            lines.append("          delete: \(item.delete)   (or in the dashboard's Models view)")
        }
        report["unused_expert_files"] = retired.map { ["path": $0.path, "bytes": $0.bytes, "delete": $0.delete] }
        // The saved prompts, the settings file and the server, for every model.
        func finish() -> (lines: [String], report: [String: Any]) {
            let cacheRoot = settings.promptCacheRoot
            let cacheBytes =
                (FileManager.default.enumerator(at: cacheRoot, includingPropertiesForKeys: [.fileSizeKey])?.compactMap {
                    (try? ($0 as? URL)?.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                }.reduce(0, +)) ?? 0
            lines.append("Cache     prompt cache \(gigabytes(UInt64(cacheBytes))) in \(cacheRoot.path)")
            lines.append(
                "Settings  \(settings.configFile.path)\(FileManager.default.fileExists(atPath: settings.configFile.path) ? "" : " (none; defaults)")"
            )
            let server = serverState(settings)
            lines.append("Server    " + server + (server.hasPrefix("not running") ? ": `origami serve` starts it" : ""))
            report["server"] = server
            // A save skipped for space, or a hot copy that gave space back for one (`DiskBudget`), on the running server.
            if !server.hasPrefix("not ") {
                let budget = DashboardState.status(port: UInt16(settings.port), key: settings.apiKey)["disk_budget"]
                if let disk = DiskBudget.line(budget) { lines.append("Disk      " + disk) }
                if let grew = DiskBudget.growthLine(budget) { lines.append("Disk      " + grew) }
            }
            return (lines, report)
        }
        // A GLM or DeepSeek GGUF: its rows as `doctor` gives them (`FamilySetup.rows`).
        if let family = FamilySetup.named(settings) {
            let (rows, fields) = FamilySetup.rows(family, settings: settings, host: host)
            for row in rows {
                let label = row.label.prefix(1).uppercased() + row.label.dropFirst()
                lines.append(label.padding(toLength: 10, withPad: " ", startingAt: 0) + row.text)
                lines += row.more.map { "          " + $0 }
            }
            report.merge(fields) { $1 }
            return finish()
        }
        guard let url = Models.locate(settings.model, settings: settings) else {
            let added = ((try? ModelRegistry.entries(settings)) ?? []).compactMap { $0["name"] as? String }
            lines.append(
                "Model     "
                    + (settings.model.map { missingModel($0, settings: settings) }
                        ?? (added.isEmpty
                            ? "none found: run `origami setup` (it can download the model)"
                            : "no default model; added: \(added.joined(separator: ", ")) (pass --model <name>, or `origami model default <name>`)"))
            )
            return (lines, report)
        }
        let model: ModelSource
        do { model = try ModelSource(directory: url) } catch {
            lines.append("Model     \(url.path): \(error)")
            return (lines, report)
        }
        // A low-bit variant: the base's files and its experts file, each with its size and the file's drive.
        let expertsPaths = model.ggufExperts?.files.map(\.path) ?? []
        let expertsFile = ExpertVariant.filesSummary(expertsPaths)
        lines.append(
            "Model     \(model.displayName) (\(model.quantization.rawValue), \(gigabytes(model.totalBytes))"
                + (expertsFile.map { " + \(gigabytes($0.bytes)) low-bit experts file" } ?? "") + ")")
        lines.append("          \(model.directory.path)")
        if let expertsFile { lines.append("          experts file: \(expertsFile.text)") }
        report["model"] = [
            "path": model.directory.path, "name": model.displayName, "fingerprint": model.fingerprint,
            "quantization": model.quantization.rawValue, "experts": ExpertVariant.note(model).map { $0 as Any } ?? NSNull(),
            "bytes": model.totalBytes,
            "experts_file": expertsFile.map { ["paths": expertsPaths, "drive": $0.drive, "bytes": $0.bytes] as Any } ?? NSNull(),
        ]
        if let note = ExpertVariant.note(model) { lines.append("          " + note) }
        let found = Stores.scan(model: model, settings: settings)
        let storage = Stores.plan(model: model, settings: settings, found: found)
        var stores: [[String: Any]] = []
        for item in found {
            var contents: [String] = []
            let copy = storage.copyFolders.contains(item.folder)
            if item.pack != nil { contents.append(item.folder == storage.packFolder || copy ? "pack (in use)" : "pack") }
            if let mirror = item.mirror {
                contents.append(
                    item.folder == Stores.legacyMirror(model: model, settings: settings, found: found)?.folder && settings.hotCopy != 0
                        ? "offline mirror of an earlier version (\(mirror.records.count) experts; not read, removed when the hot copy is built)"
                        : "copy (\(copy ? "in use, " : "")\(mirror.records.count) most-used experts)")
            }
            if item.further > 0 { contents.append("copy (\(copy ? "in use, " : "")\(item.further) most-used experts)") }
            contents += [Stores.hotCopyHeld(item, model: model)].compactMap { $0 } + item.problems
            lines.append(
                "Store     \(item.store.path) on \(item.volume.name), \(gigabytes(item.volume.free)) free: \(contents.isEmpty ? "empty" : contents.joined(separator: ", "))"
            )
            stores.append(["path": item.store.path, "volume": item.volume.name, "free": item.volume.free, "contents": contents])
        }
        report["stores"] = stores
        let routing =
            storage.packFolder == nil || storage.packSharesModelDrive
            ? Stores.reads(storage)
            : (Stores.reads(storage)
                + (storage.calibration.map {
                    String(
                        format: " (measured: model drive %.1f GB/s, pack drive %.1f GB/s)", $0.atFourReaders.model / 1e9,
                        $0.atFourReaders.pack / 1e9)
                }
                    ?? " (not measured yet: split evenly; measured at the next start)"))
        lines.append("Reads     \(routing)")
        if let folder = storage.hotCopyFolder {
            let copy = OrigamiEngine.HotCopy.open(folder: folder, identity: Stores.hotCopyIdentity(model))
            lines.append(
                "Hot copy  "
                    + (copy.map { "\($0.keys.count) experts (\(gigabytes(UInt64($0.bytes)))) in \(folder.path)" }
                        ?? "not built yet: the server builds it in \(folder.path) after the first reply"))
            report["hot_copy"] = ["path": folder.path, "records": copy?.keys.count ?? 0, "bytes": copy?.bytes ?? 0]
            // The usage curve from the saved routing counts, ranked as the server's keeper ranks them.
            if let counts = RoutingProfile.load(settings.routingFile(model))?.hotCopyCounts, !counts.uses.isEmpty {
                let curve = HotCopyCurve(
                    uses: counts.uses, order: RoutingProfile.hotCopyChoice(counts, budget: .max, record: model.expertRecordBytes),
                    size: { UInt64(((model.hotCopyRecordBytes($0) ?? 0) + 16383) / 16384 * 16384) })
                let sizes = Set([UInt64(copy?.bytes ?? 0), 20_000_000_000, 40_000_000_000, 80_000_000_000]).filter { $0 > 0 }.sorted()
                let shares = sizes.map { ["bytes": $0, "share": curve.coverage($0)] as [String: Any] }
                lines.append(
                    "          a copy of "
                        + sizes.map { "\(gigabytes($0)) would cover \(Int((curve.coverage($0) * 100).rounded()))%" }.joined(separator: ", ")
                        + " of expert uses counted")
                report["hot_copy_coverage"] = shares
            }
        }
        // Whose routing ranks the hot copy and further copies: this model's own counts once enough are counted.
        let counted = RoutingProfile.load(settings.routingFile(model))?.tokens ?? 0
        lines.append(
            "Ranking   "
                + (counted >= RoutingProfile.minimumTokens
                    ? "the hot copy holds the experts this model used most (\(counted) tokens counted)"
                    : "the hot copy holds a built-in list of experts until this model has \(RoutingProfile.minimumTokens) tokens counted (\(counted) so far)")
        )
        report["routing_tokens"] = counted
        let plan: OrigamiPlan?
        do {
            plan = try host.map {
                try OrigamiPlan.make(
                    model: model, host: $0, context: settings.context, memoryLimit: settings.memoryLimit,
                    expertMemory: settings.expertMemory,
                    storage: storage, promptCache: settings.promptCacheRoot)
            }
        } catch {
            plan = nil
            lines.append("Memory    cannot plan: \(error)")
        }
        if let host, let plan {
            let experts = plan.configuration.experts
            lines.append(
                "Memory    up to \(gigabytes(plan.memory.cap)), \(gigabytes(plan.memory.start)) if started now; keeps \(gigabytes(plan.memory.floor)) free for other apps"
            )
            lines.append(
                "          expert cache up to \(experts.cacheSlots) experts (\(gigabytes(experts.cacheBytes))), "
                    + (plan.allResident
                        ? "all of them: once read, an expert stays in memory"
                        : "at least \(experts.effectivePrefillCacheSlots) while a prompt is read (more for shorter prompts)"))
            lines += plan.notes.map { "          \($0)" }
            report["memory"] = [
                "cap": plan.memory.cap, "start": plan.memory.start, "floor": plan.memory.floor, "cache_experts": experts.cacheSlots,
                "prefill_cache_experts": experts.effectivePrefillCacheSlots, "all_resident": plan.allResident,
            ]
            let context = OrigamiPlan.contextPlan(
                model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit, expertMemory: settings.expertMemory,
                storage: storage, promptCache: settings.promptCacheRoot)
            lines.append(
                "Context   \(context.served) tokens " + (context.paged ? "kept in a file on disk" : "in memory")
                    + "; this Mac keeps up to \(context.inMemory / 1024)K in memory")
            if let note = context.note { lines.append("          \(note)") }
            report["context_plan"] = context.report
        }
        return finish()
    }

    // MARK: model

    /// `origami model list|add|remove|default`: named models (`ModelRegistry`); `--model <name>` and the `model`
    /// setting accept a name. Removing a name never touches the model's folder.
    static func model(_ arguments: [String], flags: [String: [String]], switches: Set<String>) async throws {
        let settings = try Settings.load(flags: flags)
        let entries = try ModelRegistry.entries(settings)
        let usage =
            "Usage: origami model list | add <folder|org/repo> [--name <name>] [--to <folder>] [--expert-bits 2|4|8] | add <experts.gguf> --base <model> [--name <name>] | remove <name> | move <name> <folder> | default <name>"
        func named(_ name: String) throws -> [String: Any] {
            guard let entry = entries.first(where: { $0["name"] as? String == name }) else {
                throw CLIError("No model named \(name); `origami model list` shows them")
            }
            return entry
        }
        switch arguments.first ?? "list" {
        case "list":
            var lines: [String] = []
            let mounted = Storage.volumes()
            Storage.noteVolumes(settings, volumes: mounted)
            let relinks = Storage.relinks(settings, volumes: mounted)
            for entry in entries {
                let name = entry["name"] as? String ?? "?"
                let path = entry["path"] as? String ?? "?"
                var line = (name == settings.model ? "* " : "  ") + name + "  " + path
                var bytes = ((entry["manifest"] as? [String: Any])?["bytes"] as? Int64).map { UInt64(max(0, $0)) }
                // A low-bit variant: its base's files and its experts file, each with its size.
                let experts = ExpertVariant.filesSummary(ExpertVariant.readFiles(entry))
                if experts != nil, bytes == nil, let parts = try? GGUFSource.parts(URL(fileURLWithPath: path)) {
                    bytes = parts.files.reduce(UInt64(0)) {
                        $0 + UInt64(max(0, Models.fileSize(parts.folder.appendingPathComponent($1)) ?? 0))
                    }
                }
                if let bytes { line += ", " + gigabytes(bytes) }
                if let experts { line += " + \(gigabytes(experts.bytes)) experts file \(experts.text)" }
                if let base = entry["base"] as? String { line += ", \(base) with low-bit experts (replies differ from exact)" }
                let gguf = Models.isGGUF(path)
                if let moved = relinks.first(where: { $0.setting == "model " + name }) {
                    line += "  (missing; at \(moved.to) on a renamed or remounted drive: `origami relink`)"
                } else if let drive = driveNotConnected(path) {
                    line += "  (drive \(drive) not connected)"
                } else if !FileManager.default.fileExists(atPath: gguf ? path : path + "/config.json") {
                    line += gguf ? "  (file missing)" : "  (folder missing)"
                }
                lines.append(line)
            }
            lines =
                lines.isEmpty
                ? [
                    "No models added; `origami model add <folder|org/repo>` names one. Origami serves \(settings.model ?? Models.defaultRepo)."
                ]
                : lines + ["", "* the default (`origami model default <name>` changes it)"]
            emit(lines, ["default": settings.model ?? NSNull(), "models": entries], switches.contains("json"))
        case "add":
            guard arguments.count == 2 else { throw CLIError.usage(usage) }
            func valid(_ name: String) throws -> String {
                guard name.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", options: .regularExpression) != nil else {
                    throw CLIError.usage("A model name is letters, digits, '.', '_' and '-' (up to 64), e.g. --name flash-3bit")
                }
                return name
            }
            let given = try flags["name"]?.last.map(valid)
            // Where a download or conversion is written: --to <folder> (the dashboard's choice), else the models folder.
            let destination = flags["to"]?.last.map { URL(fileURLWithPath: Settings.expand($0), isDirectory: true) } ?? settings.modelsDir
            if destination.path.hasPrefix("/Volumes/"), !Stores.volume(destination).mounted {
                throw CLIError("\(Stores.driveName(destination.path)) is not connected; connect it or pass another --to folder")
            }
            let spec = arguments[1]
            func derived(_ text: String) throws -> String {
                let base = (text.split(separator: "/").last.map(String.init) ?? "model")
                    .map { $0.isASCII && ($0.isLetter || $0.isNumber) || "._-".contains($0) ? $0 : "-" }
                return try valid(String(String(base).prefix(64)))
            }
            var downloaded = false
            /// A name derived again for another copy (another path) would silently re-point the first: refused without --name,
            /// before anything is converted.
            func sameName(_ name: String, _ path: String) throws {
                guard given == nil, let earlier = entries.first(where: { $0["name"] as? String == name })?["path"] as? String,
                    earlier != path
                else { return }
                throw CLIError(
                    "\(name) already names \(earlier); pass --name <another name> to add this copy too, or `origami model remove \(name)` first"
                )
            }
            let gguf = Models.isGGUF(Settings.expand(spec))
            var url =
                gguf
                ? URL(fileURLWithPath: Settings.expand(spec))
                : FileManager.default.fileExists(atPath: Settings.expand(spec) + "/config.json")
                    ? URL(fileURLWithPath: Settings.expand(spec), isDirectory: true) : Models.locate(spec, settings: settings)
            if spec.lowercased().hasSuffix(".gguf"), !FileManager.default.fileExists(atPath: Settings.expand(spec)) {
                let why = missingModel(spec, settings: settings)
                throw CLIError(why.prefix(1).uppercased() + why.dropFirst())
            }
            // Only `owner/repo` is offered as a download, and not when `owner` is a folder here: then both readings are
            // printed and nothing is assumed.
            let local = FileManager.default.currentDirectoryPath + "/" + spec
            if url == nil, spec.contains("/"), !"/~.".contains(spec.first!),
                spec.split(separator: "/", omittingEmptySubsequences: false).count != 2
                    || FileManager.default.fileExists(atPath: String(spec.prefix { $0 != "/" }))
            {
                throw CLIError(
                    "No model folder at \(local) (it needs config.json)"
                        + (spec.split(separator: "/", omittingEmptySubsequences: false).count == 2
                            ? "; \(spec) may also mean the Hugging Face repo \(spec), not downloaded because ./\(spec.prefix { $0 != "/" }) is a folder here. Fix the path, or run the command from another folder to download that repo"
                            : ": fix the path"))
            }
            if url == nil, spec.contains("/"), !"/~.".contains(spec.first!) {
                let folder = destination.appendingPathComponent(spec.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
                // Known before a byte is fetched: a repo of GGUF files, a model Origami cannot use, a drive too small to
                // hold the download and its conversion (what an interrupted download or conversion holds there counts).
                let report = try Models.checkRepo(spec)
                if report.verdict == .converts, let family = report.family {
                    let base = String(report.source.dropFirst(spec.count + 1))
                    let need = Models.conversionNeeds(
                        UInt64(max(0, report.totalBytes))
                            + (try Models.conversionBytes(hub: HubRepo(spec), base: base, family: family, settings: settings)))
                    let name = try given ?? derived(folder.lastPathComponent)
                    let held = [folder.path, destination.appendingPathComponent(name + "-4bit.partial").path].reduce(UInt64(0)) {
                        $0 + Storage.size($1).allocated
                    }
                    let free = Stores.volume(destination).free
                    guard free + held >= need else {
                        throw CLIError(
                            "Downloading and converting \(spec) needs \(gigabytes(need)) free in \(destination.path), 10 GB to spare included (\(gigabytes(free)) now). "
                                + "Nothing was downloaded; pass --to with a folder on a bigger drive.")
                    }
                }
                guard Terminal.confirm("\(spec) is not on this Mac. Download it from Hugging Face?", yes: switches.contains("yes")) else {
                    throw CLIError("Not added: \(spec) was not downloaded.")
                }
                url = try await Models.download(repo: spec, settings: settings, into: flags["to"] == nil ? nil : folder)
                downloaded = true
            }
            guard var url else { throw CLIError("No model folder at \(spec): it needs config.json and .safetensors files") }
            // Low-bit experts from a GGUF on an exact model already here: a generated folder, nothing copied or converted.
            var variant: [String: Any]?
            if let base = flags["base"]?.last {
                let name = try given ?? derived(GGUFSource.modelPrefix(url.lastPathComponent).replacingOccurrences(of: ".gguf", with: ""))
                if let earlier = entries.first(where: { $0["name"] as? String == name }), earlier["base"] == nil {
                    throw CLIError("\(name) already names a model; pass --name <another name>")
                }
                variant = ["base": base, "experts": url.path]
                // A GLM or DeepSeek GGUF base: only an entry, its path the base's GGUF.
                if var entry = try ExpertVariant.family(
                    experts: url, base: base, name: name, settings: settings, yes: switches.contains("yes"))
                {
                    let kept = ExpertVariant.keepSettings(&entry, from: entries)
                    let replaced = try ModelRegistry.put(entry, settings: settings)
                    var probe = settings
                    probe.model = name
                    let note = ExpertVariant.note(
                        try ExpertVariant.source(URL(fileURLWithPath: entry["path"] as! String), settings: probe))!
                    print("\(replaced ? "Updated" : "Added") \(name): \(base) with \(note).")
                    if let kept { print(kept) }
                    print("Opt-in: \(base) itself stays exact. Serve it with `origami serve --model \(name)`.")
                    if let line = ExpertVariant.seedCounts(name, settings: settings) {
                        print(line.prefix(1).uppercased() + line.dropFirst() + ".")
                    }
                    return
                }
                url = try ExpertVariant.make(experts: url, base: base, name: name, settings: settings)
            }
            // GLM-5.3-Flash and DeepSeek-V4.1-Flash: a GGUF runs as stored; a GLM safetensors checkpoint converts once to a GGUF.
            if var entry = try addFamilyModel(
                url, gguf: gguf && variant == nil, name: given, derived: derived, into: destination, flags: flags,
                yes: switches.contains("yes"), taken: sameName)
            {
                let kept = ExpertVariant.keepSettings(&entry, from: entries)
                let replaced = try ModelRegistry.put(entry, settings: settings)
                print(
                    "\(replaced ? "Updated" : "Added") \(entry["name"]!): \(entry["path"]!). Serve it with `origami serve --model \(entry["name"]!)`."
                )
                if let kept { print(kept) }
                return
            }
            if flags["expert-bits"] != nil {
                throw CLIError("--expert-bits is for converting a GLM-5.3-Flash checkpoint; otherwise it " + Settings.expertBitsRetired)
            }
            let model: ModelSource
            let source = url
            // A split GGUF is named after its parts without the part number.
            let sourceName =
                gguf
                ? url.deletingPathExtension().lastPathComponent.replacingOccurrences(
                    of: "-\\d{5}-of-\\d{5}$", with: "", options: .regularExpression)
                : url.lastPathComponent
            var converted = false
            do { model = try ModelSource(directory: url) } catch {
                // An unquantized checkpoint (the official bf16 release, most fine-tunes), the FP8 release or a GGUF is converted once.
                let converter: CheckpointConverter
                if gguf {
                    let source = try GGUFSource(url)
                    if let problem = CheckpointConverter.ggufProblem(source) {
                        throw CLIError("Origami cannot convert this GGUF: \(problem)")
                    }
                    converter = try CheckpointConverter(gguf: source, support: try await Models.ggufSupport(settings: settings))
                } else {
                    // The converter's own refusal names what it cannot convert; for a checkpoint it does not convert in kind
                    // (one already quantized), the loader's error says more.
                    do { converter = try CheckpointConverter(source: url) } catch let refusal as EngineError
                        where !refusal.description.contains("already quantized")
                    { throw refusal } catch _ { throw error }
                }
                let name = try given ?? derived(sourceName)
                let target = destination.appendingPathComponent(name + "-4bit", isDirectory: true)
                try sameName(name, target.path)
                let manifest = (try? Data(contentsOf: target.appendingPathComponent("origami-model.json")))
                    .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
                if manifest?["source_identity"] as? String == converter.identity, let earlier = try? ModelSource(directory: target) {
                    print("Using the earlier conversion in \(target.path).")
                    url = target
                    converted = true
                    model = earlier
                } else {
                    // A resumed conversion's own partial folder is room it already holds.
                    let free = Stores.volume(destination).free + Storage.size(target.path + ".partial").allocated
                    let need = Models.conversionNeeds(converter.outputBytes)
                    print(
                        "\(spec) is \(converter.sourceKind) (\(gigabytes(converter.sourceBytes))). Origami converts it once into the 4-bit layout it runs: "
                            + "\(gigabytes(converter.outputBytes)) in \(target.path). The source is only read."
                            + (converter.expertManifest == nil
                                ? "" : " Its routed experts stay in the GGUF and are read from it while serving: keep it where it is."))
                    if manifest != nil {
                        print(
                            "That folder holds an earlier conversion of another source, which this replaces; pass --name <name> to keep both."
                        )
                    }
                    guard free >= need else {
                        // The download stays where it is and is named, so the next run converts it instead of fetching it again.
                        throw CLIError(
                            "Converting needs \(gigabytes(need)) free there, 10 GB to spare included (\(gigabytes(free)) now). "
                                + (downloaded
                                    ? "The download is kept in \(source.path): run `origami model add \(source.path) --to <a folder on a bigger drive>`."
                                    : "Pass --to with a folder on a bigger drive: `origami model add <source> --to /Volumes/…`.")
                        )
                    }
                    guard Terminal.confirm("Convert it now? This reads the whole checkpoint once.", yes: switches.contains("yes")) else {
                        throw CLIError("Not added: \(spec) was not converted.")
                    }
                    let steps = Terminal.steps("converting")
                    // Stopped, it keeps the shards it finished and resumes from them: kept, and said so.
                    let partial = target.path + ".partial"
                    let keep = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
                        signal(number, SIG_IGN)
                        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
                        source.setEventHandler {
                            Terminal.progress(nil)
                            FileHandle.standardError.write(
                                Data(
                                    "origami: stopped; the unfinished conversion is kept in \(partial): run the same command again to resume it, or delete it\n"
                                        .utf8))
                            exit(130)
                        }
                        source.resume()
                        return source
                    }
                    defer {
                        for source in keep { source.cancel() }
                        signal(SIGINT, SIG_DFL)
                        signal(SIGTERM, SIG_DFL)
                    }
                    try converter.convert(to: target) { done, total in steps(Int(done), Int(total)) }
                    url = target
                    converted = true
                    model = try ModelSource(directory: target)
                }
            }
            let name = try given ?? derived(converted ? sourceName : model.displayName)
            var entry: [String: Any] = ["name": name, "path": url.path, "added": ISO8601DateFormatter().string(from: Date())]
            entry.merge(variant ?? [:]) { $1 }
            var manifest = try Diagnostics.inspectReport(url).report
            manifest["fingerprint"] = model.fingerprint
            // What stays when the folder moves (Locate compares it): its config, index and shard sizes.
            manifest["earlier_fingerprint"] = ModelSource.earlierFingerprint(directory: url) ?? NSNull()
            if converted {
                manifest["converted_from"] = source.path
                manifest["chain"] =
                    (try? Data(contentsOf: url.appendingPathComponent("origami-model.json")))
                    .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["chain"] ?? "converted"
            }
            manifest["template_sha256"] =
                (try? Data(contentsOf: url.appendingPathComponent("chat_template.jinja")))
                .map { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() } ?? NSNull()
            entry["manifest"] = manifest
            try sameName(name, url.path)
            let kept = ExpertVariant.keepSettings(&entry, from: entries)
            let replaced = try ModelRegistry.put(entry, settings: settings)
            print(
                "\(replaced ? "Updated" : "Added") \(name): \(url.path) (\(model.quantization.rawValue))."
            )
            if let kept { print(kept) }
            if variant != nil, let note = ExpertVariant.note(model) {
                print("It runs \(variant!["base"]!) with \(note). Opt-in: \(variant!["base"]!) itself stays exact.")
                if let line = ExpertVariant.seedCounts(name, settings: settings) {
                    print(line.prefix(1).uppercased() + line.dropFirst() + ".")
                }
            }
            print(
                "Serve it with `origami serve --model \(name)`, or make it the default: `origami model default \(name)`.")
        case "remove":
            guard arguments.count == 2 else { throw CLIError.usage(usage) }
            let entry = try named(arguments[1])
            let path = entry["path"] as? String ?? "?"
            try ModelRegistry.update(settings) { $0.removeAll { $0["name"] as? String == arguments[1] } }
            if settings.model == arguments[1] { try Settings.write(["model": nil], to: settings.configFile) }
            if entry["base"] != nil, ExpertVariant.owns(path, settings: settings) {
                // Links and two small files that `model add --base` wrote; the links' targets stay.
                try? FileManager.default.removeItem(atPath: path)
                print(
                    "Removed \(arguments[1]) and its generated folder; \(entry["base"]!) and \(entry["experts"] ?? "the experts file") are untouched."
                )
            } else if entry["base"] != nil {
                print("Removed \(arguments[1]); \(entry["base"]!) and \(entry["experts"] ?? "the experts file") are untouched.")
            } else {
                print(
                    "Removed \(arguments[1]); its folder \(path) is untouched. Its hot copy and saved prompts stay too; the dashboard's Models view lists and deletes them."
                )
            }
        case "move":
            guard arguments.count == 3 else { throw CLIError.usage(usage) }
            _ = try named(arguments[1])
            try requireNoServer(settings, "model move")
            let all = Storage.items(settings, found: ([], "the command line does not search for models"))
            guard let item = all.first(where: { $0.role == "model" && $0.model == arguments[1] }) else {
                throw CLIError("\(arguments[1]) has no files of its own to move (a low-bit variant moves with its base)")
            }
            print(try move(item, to: Settings.expand(arguments[2]), settings: settings, items: all))
        case "default":
            guard arguments.count == 2 else { throw CLIError.usage(usage) }
            _ = try named(arguments[1])
            try Settings.write(["model": arguments[1]], to: settings.configFile)
            print("\(arguments[1]) is the default model; a running server loads it after a restart.")
        default: throw CLIError.usage(usage)
        }
    }

    static func emit(_ lines: [String], _ report: [String: Any], _ json: Bool) {
        if json, let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) {
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(lines.joined(separator: "\n"))
        }
    }

    /// Asks a running server on the settings' port for its health, without blocking for long.
    static func serverState(_ settings: Settings) -> String {
        let port = settings.port
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/health")!)
        request.timeoutInterval = 1
        if let key = settings.apiKey { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        // Written by the URLSession callback before it signals; read only after that signal, never after a timeout.
        final class Box: @unchecked Sendable { var text = "not running on port" }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { data, response, _ in
            if (response as? HTTPURLResponse)?.statusCode == 401 {
                box.text = "running with another API key (api_key) on port"
            } else if let data, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                box.text = "\(object["status"] ?? "?") on port"
            }
            done.signal()
        }.resume()
        guard done.wait(timeout: .now() + 1.5) == .success else { return "not answering on port \(port)" }
        return "\(box.text) \(port)"
    }

    /// For a command that needs a ready server: the state the server is in, and what to do, when it is not ready.
    static func requireReady(_ settings: Settings) throws {
        let state = serverState(settings)
        guard !state.hasPrefix("ready") else { return }
        let next =
            state.hasPrefix("not running")
            ? "start `origami serve` first" : state.hasPrefix("stopped") ? "run `origami start`" : "see `origami status`"
        throw CLIError("No ready Origami server: \(state); \(next)")
    }

    /// `origami stop` and `origami start`: the running server's Stop and unload and Start (`POST v1/stop`, `v1/start`),
    /// waiting until the model is unloaded, or runs (or fails to start) again.
    static func power(start: Bool, flags: [String: [String]]) async throws {
        let settings = try Settings.load(flags: flags)
        let port = settings.port
        func ask(_ path: String, post: Bool = false) async throws -> (code: Int, body: [String: Any]) {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/" + path)!)
            request.timeoutInterval = 10
            if let key = settings.apiKey { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
            if post {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = Data("{}".utf8)
            }
            let (data, response) = try await URLSession.shared.data(for: request)
            return (
                (response as? HTTPURLResponse)?.statusCode ?? 0, (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            )
        }
        let answer: (code: Int, body: [String: Any])
        do { answer = try await ask(start ? "v1/start" : "v1/stop", post: true) } catch {
            throw CLIError("No Origami server answers on port \(port); origami serve starts one")
        }
        guard [200, 202].contains(answer.code) else {
            if answer.code == 401 { throw CLIError("The server on port \(port) has another API key (api_key)") }
            if [404, 405].contains(answer.code) {
                throw CLIError(
                    "The server on port \(port) has no \(start ? "Start" : "Stop and unload") (an older Origami, or one serving --single-model)"
                )
            }
            throw CLIError(((answer.body["error"] as? [String: Any])?["message"] as? String) ?? "The server answered HTTP \(answer.code)")
        }
        let model = answer.body["model"] as? String ?? "the model"
        if start && answer.code == 200 { return print("Origami is not stopped (\(model)); origami doctor shows how it is.") }
        if start, answer.body["retried"] as? Bool == true {
            print("The last start failed (\(answer.body["failure"] as? String ?? "see the server's log")); starting \(model) again.")
        }
        if !start && answer.body["status"] as? String == "stopping" {
            print("Stopping: a running request gets up to \(answer.body["wait_seconds"] as? Int ?? 30) s to finish, then it is cancelled.")
        }
        // Until it is unloaded (a few seconds, or the wait), or runs or fails again (a load takes about a minute).
        let deadline = Date().addingTimeInterval(start ? 900 : 300)
        while Date() < deadline {
            try await Task.sleep(for: .milliseconds(500))
            guard let state = try? await ask("health").body else { continue }
            switch (start, state["status"] as? String) {
            case (false, "stopped"):
                print(
                    "Stopped: \(model) is unloaded"
                        + ((state["freed_bytes"] as? NSNumber).map { ", about \(OrigamiPlan.gigabytes($0.uint64Value)) back to macOS" }
                            ?? "")
                        + ". The dashboard still answers; origami start, or Start there, loads it again.")
                if let note = state["note"] as? String { print(note) }
                return
            case (true, "ready"): return print("Ready: \(model) runs again.")
            case (true, "failed"): throw CLIError(state["failure"] as? String ?? "\(model) did not start; the server's log says why")
            default: continue
            }
        }
        throw CLIError(
            start
                ? "\(model) is still loading; the dashboard shows when it is ready"
                : "The model is still unloading; origami doctor shows when it has stopped")
    }

    // MARK: store

    static func store(_ arguments: [String], flags: [String: [String]], switches: Set<String>) async throws {
        // The stores the config file lists: a --store flag would replace that list for this run only.
        var settings = try Settings.load(flags: flags.filter { $0.key != "store" })
        switch arguments.first {
        case "add":
            guard arguments.count == 2 else { throw CLIError.usage("Usage: origami store add <folder> [--size 200GB]") }
            let path = Settings.expand(arguments[1])
            var size: UInt64?
            if let text = flags["size"]?.last {
                guard let bytes = Settings.parseBytes(text) else { throw CLIError("--size \(text) is not a size such as 200GB") }
                size = bytes
            }
            do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true) } catch {
                throw CLIError("Cannot use \(path): \(error)")
            }
            settings.stores.removeAll { $0.path == path }
            settings.stores.append(.init(path: path, size: size))
            try settings.saveStores()
            print("Added store \(path)\(size.map { " (up to \(gigabytes($0)))" } ?? "")")
            if let model = try? model(settings) {
                try prepareStores(
                    model: model, settings: settings, noPack: switches.contains("no-pack"), fullPack: switches.contains("full-pack"))
            }
        case "remove":
            guard arguments.count == 2 else { throw CLIError.usage("Usage: origami store remove <folder>") }
            let path = Settings.expand(arguments[1])
            guard settings.stores.contains(where: { $0.path == path }) else {
                throw CLIError("\(path) is not a store (see `origami store list`)")
            }
            try requireNoServer(settings, "store remove")
            // The storage map keeps listing what is left there.
            Storage.remember(settings)
            let payloads = origamiFolders(in: URL(fileURLWithPath: path))
            if !payloads.isEmpty {
                print("Origami's expert copies in \(path):")
                for folder in payloads { print("  \(folder.lastPathComponent)") }
                guard Terminal.confirm("Delete these \(payloads.count) folder(s)?", yes: switches.contains("yes")) else {
                    throw CLIError("Nothing was deleted (not confirmed).")
                }
                for folder in payloads { try FileManager.default.removeItem(at: folder) }
            }
            settings.stores.removeAll { $0.path == path }
            try settings.saveStores()
            print("Removed store \(path)")
        case "list", nil:
            // Conversions stopped part way, which can be large: with their size and what to do.
            defer {
                for item in Storage.items(settings, found: ([], "not searched")) where item.path.hasSuffix(".partial") {
                    print("\(item.path)  \(gigabytes(item.allocated))  \(item.note ?? "")")
                }
            }
            let model: ModelSource
            do { model = try Commands.model(settings) } catch {
                for path in settings.stores.map(\.path) + [settings.homeStore.path] {
                    let volume = Stores.volume(URL(fileURLWithPath: path))
                    print("\(path)  [\(volume.name), \(gigabytes(volume.free)) free]")
                }
                print("Contents not checked: \(error)")
                return
            }
            for item in Stores.scan(model: model, settings: settings) {
                let contents =
                    [
                        item.pack.map { _ in "pack" }, item.mirror.map { "copy (\($0.records.count) most-used experts)" },
                        item.further > 0 ? "copy (\(item.further) most-used experts)" : nil,
                        Stores.hotCopyHeld(item, model: model),
                    ].compactMap { $0 } + item.problems
                print(
                    "\(item.store.path)  [\(item.volume.name), \(gigabytes(item.volume.free)) free]  \(contents.isEmpty ? "empty" : contents.joined(separator: ", "))"
                )
            }
        default: throw CLIError.usage("Usage: origami store add|remove|list")
        }
    }

    /// Moves a storage map item as the dashboard's Move does (`Storage.plan`, then `Storage.execute` with its journal), with
    /// progress on the terminal; Ctrl-C cancels a copy under way and puts everything back.
    static func move(_ item: Storage.Item, to folder: String, settings: Settings, items: [Storage.Item]) throws -> String {
        let plan = try Storage.plan(item, to: folder, settings: settings, items: items)
        print(
            plan.renames
                ? "Moving \(item.name) to \(plan.folder) (the same drive: renamed, nothing copied)."
                : "Moving \(item.name) to \(plan.folder): \(gigabytes(plan.bytes)) copied, read back and compared, then the original deleted. Ctrl-C cancels."
        )
        defer { Terminal.progress(nil) }
        return try interruptible { cancelled in
            try Storage.execute(
                plan, settings: settings,
                step: { text, fraction in Terminal.progress(text + (fraction.map { String(format: " %.0f%%", $0 * 100) } ?? "")) },
                cancelled: cancelled)
        }
    }
    /// Runs `body` with Ctrl-C (and a stop, SIGTERM) turned into a request it polls: what it does then is its own to undo.
    static func interruptible<T>(_ body: (@escaping @Sendable () -> Bool) throws -> T) rethrows -> T {
        let stop = StopFlag()
        let sources = [SIGINT, SIGTERM].map { number in
            let previous = signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { stop.raise() }
            source.resume()
            return (number, previous, source)
        }
        defer {
            for (number, previous, source) in sources {
                source.cancel()
                signal(number, previous)
            }
        }
        return try body { stop.isRaised }
    }

    final class StopFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var raised = false
        func raise() { lock.withLock { raised = true } }
        var isRaised: Bool { lock.withLock { raised } }
    }

    /// `origami move [list]` and `origami move recover [<id>]`: the moves a stop interrupted (their journals in
    /// `<data_dir>/moves`), and finishing or undoing them as the dashboard's Recover does.
    static func moves(_ arguments: [String], flags: [String: [String]]) throws {
        let settings = try Settings.load(flags: flags)
        let pending = MoveJournal.pending(settings).sorted { ($0["created"] as? Double ?? 0) < ($1["created"] as? Double ?? 0) }
        func line(_ record: [String: Any]) -> String {
            let from = ((record["pairs"] as? [[String: Any]])?.first?["from"] as? String) ?? "?"
            return
                "\(record["id"] as? String ?? "?")  \(record["name"] as? String ?? "?")  \(from) → \(record["folder"] as? String ?? "?")  ("
                + (record["phase"] as? String == "committed"
                    ? "moved; " + (record["rebind"] != nil ? "carry-over or " : "") + "cleanup left)"
                    : "stopped while \(record["phase"] as? String ?? "copying"))")
        }
        switch arguments.first ?? "list" {
        case "list":
            if pending.isEmpty { print("No interrupted moves.") }
            for record in pending { print(line(record)) }
            if !pending.isEmpty { print("`origami move recover` finishes or undoes them, with the server quit.") }
        case "recover":
            guard arguments.count <= 2 else { throw CLIError.usage("Usage: origami move recover [<id>]") }
            let chosen = pending.filter { arguments.count < 2 || $0["id"] as? String == arguments[1] }
            guard !chosen.isEmpty else {
                if arguments.count == 2 { throw CLIError("No interrupted move \(arguments[1]); `origami move list` shows them") }
                print("No interrupted moves.")
                return
            }
            try requireNoServer(settings, "move recover")
            var failed = 0
            for record in chosen {
                let name = record["name"] as? String ?? "?"
                do { print("\(name): " + (try MoveJournal.recover(record["id"] as? String ?? "", settings: settings)) + ".") } catch {
                    print("\(name): not recovered: \(error)")
                    failed += 1
                }
            }
            if failed > 0 { throw CLIError("\(failed) of \(chosen.count) moves not recovered; their journals are kept") }
        default: throw CLIError.usage("Usage: origami move list | recover [<id>]")
        }
    }

    /// `origami service install|remove`: the LaunchAgent of docs/integration.md, written to
    /// `~/Library/LaunchAgents/<label>.plist` and loaded (`launchctl bootstrap`), or unloaded and deleted. It runs this
    /// binary's `serve` with the setting options given and the `ORIGAMI_*` variables of this shell, logs to
    /// `~/Library/Logs/Origami/<label>.log`, and prints exactly what it wrote; a different plist there is replaced only
    /// with --yes.
    static func service(_ arguments: [String], flags: [String: [String]], yes: Bool) throws {
        let label = flags["label"]?.last ?? "com.origami.serve"
        guard label.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil else {
            throw CLIError.usage("A label is letters, digits, '.', '_' and '-', such as com.origami.serve")
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let plist = home.appendingPathComponent("Library/LaunchAgents/\(label).plist")
        let domain = "gui/\(getuid())"
        func launchctl(_ arguments: [String]) -> (status: Int32, output: String) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            process.arguments = arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            guard (try? process.run()) != nil else { return (-1, "cannot run launchctl") }
            let output = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            process.waitUntilExit()
            return (process.terminationStatus, output.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        switch arguments.first {
        case "install":
            let settings = try Settings.load(flags: flags)
            guard let binary = Bundle.main.executableURL.map({ $0.resolvingSymlinksInPath().path }) else {
                throw CLIError("Cannot tell where this origami binary is")
            }
            let logs = home.appendingPathComponent("Library/Logs/Origami", isDirectory: true)
            let environment = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("ORIGAMI_") }
            // Every setting option given is carried into the service's arguments, its folders made absolute the way
            // they read here (launchd starts the service in /).
            let folders: Set = ["model", "data-dir", "context-dir", "hot-copy-dir", "request-log", "speculation", "models-dirs"]
            var program = [binary, "serve"]
            for option in settingOptions {
                for value in (flags[option] ?? []).suffix(option == "store" ? Int.max : 1) {
                    let key = option.replacingOccurrences(of: "-", with: "_")
                    let stored = try folders.contains(option) ? Settings.storedValue(key, value, current: settings) as? String : nil
                    program += ["--" + option, option == "store" || option == "config" ? Settings.expand(value) : stored ?? value]
                }
            }
            let contents: [String: Any] = [
                "Label": label, "ProgramArguments": program, "RunAtLoad": true, "KeepAlive": true, "ExitTimeOut": 180,
                "StandardErrorPath": logs.appendingPathComponent(label + ".log").path,
            ].merging(environment.isEmpty ? [:] : ["EnvironmentVariables": environment]) { $1 }
            let data = try PropertyListSerialization.data(fromPropertyList: contents, format: .xml, options: 0)
            if let existing = try? Data(contentsOf: plist), existing != data, !yes {
                throw CLIError("\(plist.path) holds a different service; `origami service install --yes` replaces it")
            }
            // launchd would create the log folder itself, readable by others.
            try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.createDirectory(at: plist.deletingLastPathComponent(), withIntermediateDirectories: true)
            _ = launchctl(["bootout", domain + "/" + label])
            // launchd keeps the old job until its server has exited (up to ExitTimeOut); a bootstrap meanwhile fails.
            if launchctl(["print", domain + "/" + label]).status == 0 {
                print("Waiting for the running service to stop…")
                let deadline = Date().addingTimeInterval(190)
                while launchctl(["print", domain + "/" + label]).status == 0 {
                    guard Date() < deadline else {
                        throw CLIError("The running service \(label) did not stop within 190 s; run `origami service install` again")
                    }
                    Thread.sleep(forTimeInterval: 0.5)
                }
            }
            try data.write(to: plist, options: .atomic)
            // It may hold a key from the shell: readable by this user only, and shown as "(set)".
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: plist.path)
            let secrets = Settings.secretKeys.map { "ORIGAMI_" + $0.uppercased() }.filter { environment[$0] != nil }
            var shown = contents
            if !secrets.isEmpty { shown["EnvironmentVariables"] = environment.merging(secrets.map { ($0, "(set)") }) { $1 } }
            let text = try PropertyListSerialization.data(fromPropertyList: shown, format: .xml, options: 0)
            print("Wrote \(plist.path):\n" + String(decoding: text, as: UTF8.self))
            if !secrets.isEmpty { print("It holds \(secrets.joined(separator: " and ")) from this shell, so only you can read it.") }
            let loaded = launchctl(["bootstrap", domain, plist.path])
            guard loaded.status == 0 else { throw CLIError("launchctl bootstrap \(domain) \(plist.path) failed: \(loaded.output)") }
            print(
                "Loaded: it starts now and at each login. `origami service remove\(flags["label"] == nil ? "" : " --label \(label)")` stops and deletes it."
            )
            let external =
                ([Models.locate(settings.model, settings: settings)?.path, settings.dataDir.path, settings.promptCacheRoot.path]
                + settings.stores.map(\.path)).compactMap { $0 }.filter { $0.hasPrefix("/Volumes/") }
            if !external.isEmpty {
                print(
                    "Its files are on an external drive (\(external[0])): macOS may ask once to allow Removable Volumes for origami; "
                        + "allow it (System Settings › Privacy & Security › Files and Folders), and again after replacing the binary.")
            }
        case "remove":
            let unloaded = launchctl(["bootout", domain + "/" + label])
            let existed = FileManager.default.fileExists(atPath: plist.path)
            if existed { try FileManager.default.removeItem(at: plist) }
            print(
                existed || unloaded.status == 0
                    ? "Removed the service \(label)\(existed ? " (\(plist.path))" : "")\(unloaded.status == 0 ? "; stopped" : "")."
                    : "No service \(label) is installed.")
        default: throw CLIError.usage("Usage: origami service install|remove [--label <label>] [--yes]")
        }
    }

    /// `origami relink`: the folders settings name on a drive renamed or mounted under another name (`Storage.relinks`),
    /// shown, then used once confirmed.
    static func relink(flags: [String: [String]], yes: Bool) throws {
        let settings = try Settings.load(flags: flags)
        let volumes = Storage.volumes()
        Storage.noteVolumes(settings, volumes: volumes)
        let found = Storage.relinks(settings, volumes: volumes)
        guard !found.isEmpty else {
            print("Nothing to relink: every folder the settings name is where they say, or no connected drive holds it.")
            return
        }
        print("On a drive renamed or mounted under another name:")
        for item in found { print("  \(item.setting): \(item.from)\n    is at \(item.to) (\(item.how))") }
        try requireNoServer(settings, "relink")
        guard Terminal.confirm("Use the new places?", yes: yes) else { throw CLIError("Nothing was changed (not confirmed).") }
        for line in try Storage.relink(found, settings: settings) { print(line) }
        Storage.noteVolumes(try Settings.load(flags: flags), volumes: volumes)
    }

    /// Subfolders of a store that hold an Origami pack, further copy, Qwen hot copy (`hot/index.json`; any model) or the
    /// 3- or 2-bit pack an earlier Origami made (`Stores.retiredExperts`), known by a header in
    /// Origami's store format: a folder that merely holds a file of that name is the user's.
    static func origamiFolders(in store: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: store, includingPropertiesForKeys: nil)) ?? []).filter { folder in
            ["pack.json", "mirror.json", "pack-3bit.json", "pack-2bit.json"].contains { name in
                (try? Data(contentsOf: folder.appendingPathComponent(name)))
                    .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["format"] as? String == StoreHeader.format
            } || interruptedBuild(folder)
                || (folder.lastPathComponent.range(of: "-[0-9a-f]{12}$", options: .regularExpression) != nil
                    && FileManager.default.fileExists(atPath: folder.appendingPathComponent("hot/index.json").path))
        }
    }
    /// A `<name>-<fingerprint12>` folder holding only an interrupted build's payloads (headers are written last).
    static func interruptedBuild(_ folder: URL) -> Bool {
        guard folder.lastPathComponent.range(of: "-[0-9a-f]{12}$", options: .regularExpression) != nil,
            let names = try? FileManager.default.contentsOfDirectory(atPath: folder.path), !names.isEmpty
        else { return false }
        return names.allSatisfy {
            $0 == "hot" || $0.range(of: "^(experts(-[23]bit)?|ple|mirror)\\.bin(\\.partial)?$", options: .regularExpression) != nil
        }
    }

    /// Origami's `<fingerprint12>-<context>` folders in the prompt-cache root; anything else there is the user's
    /// (context_dir may be a shared folder).
    static func cacheFolders(_ settings: Settings) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: settings.promptCacheRoot, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.range(of: "^[0-9a-f]{12}-[0-9]+$", options: .regularExpression) != nil }.sorted {
                $0.path < $1.path
            }
    }

    /// Refuses to delete files a running server may be using.
    /// Why the model `spec` names is not there, and what to do: its drive is not connected, no model has that name, or
    /// nothing is at that path.
    static func missingModel(_ spec: String, settings: Settings) -> String {
        let path = ModelRegistry.entry(spec, settings: settings)?["path"] as? String ?? Settings.expand(spec)
        if let drive = driveNotConnected(path) { return "no model at \(path): the drive \(drive) is not connected; connect it" }
        if !spec.contains("/"), !spec.hasPrefix("~"), !spec.hasPrefix(".") {
            return "no model named \(spec): `origami model list` shows the names, `origami model add <folder>` adds one"
        }
        return spec.lowercased().hasSuffix(".gguf")
            ? "no model file at \(spec): fix the path"
            : "no model at \(spec): fix the path, or run `origami setup` (it can download the model)"
    }

    /// The name of the drive `path` is on when that drive is not mounted (`/Volumes/<name>` is not there).
    static func driveNotConnected(_ path: String) -> String? {
        let parts = URL(fileURLWithPath: path).pathComponents
        guard path.hasPrefix("/Volumes/"), parts.count > 2, !FileManager.default.fileExists(atPath: "/Volumes/" + parts[2]) else {
            return nil
        }
        return parts[2]
    }

    static func requireNoServer(_ settings: Settings, _ command: String) throws {
        let state = serverState(settings)
        guard state.hasPrefix("not running") else {
            throw CLIError(
                "An Origami server is \(state) and uses these files.\n"
                    + "Quit it (Ctrl-C where `origami serve` runs, or stop the service that runs it; `origami stop` leaves it running),\n"
                    + "then run `origami \(command)` again.")
        }
    }

    // MARK: config

    /// When a saved change to `key` applies (`Settings.appliesNow`, `appliesAtRestart`, else at the model's next load),
    /// for which model (`model`: saved in its entry), and what overrides it: an option or variable of this command, or of
    /// the server that answers on the port.
    static func appliesLine(_ key: String, model: String?, settings: Settings) -> String {
        var line =
            Settings.appliesNow.contains(key)
            ? "Applies now"
            : Settings.appliesAtRestart.contains(key)
                ? "Applies when `origami serve` starts again"
                : "Applies when the model next loads (the dashboard's Reload, or `origami stop` then `origami start`)"
        if let model { line += ", to \(model) only (saved for it in models.json)" }
        let option = "--" + key.replacingOccurrences(of: "_", with: "-")
        let variable = "ORIGAMI_" + key.uppercased()
        let server =
            (DashboardState.status(port: UInt16(clamping: settings.port), key: settings.apiKey)["settings"] as? [String: Any])?[key]
            as? [String: Any]
        if ["flag", "env"].contains(settings.sources[key]) {
            line += "; \(settings.sources[key] == "flag" ? option : variable) given here overrides it"
        } else if let source = server?["source"] as? String, ["flag", "env"].contains(source) {
            line +=
                "; the running server was started with \(source == "flag" ? option : variable), which overrides it until it starts without"
        }
        return line + "."
    }

    static func config(_ arguments: [String], flags: [String: [String]], json: Bool = false, yes: Bool = false) throws {
        let settings: Settings
        do { settings = try Settings.load(flags: flags) } catch let error as CLIError
            where arguments.first == "unset" && arguments.count == 2
        {
            // A value the settings refuse must not keep `unset` from removing it: the file's key goes without a full load.
            let url = Settings.configFile(flags: flags)
            guard try Settings.readConfig(url)[arguments[1]] != nil else { throw error }
            try Settings.write([arguments[1]: nil], to: url)
            return print("\(arguments[1]) removed from \(url.path)")
        }
        let file = try Settings.readConfig(settings.configFile)
        if !json, let note = settings.unknownKeysNote { FileHandle.standardError.write(Data("origami: warning: \(note)\n".utf8)) }
        let keys = Settings.keys.map(\.0)
        let effective = settings.display
        switch arguments.first {
        case nil:
            if json { return emit([], ["settings": settings.report, "config_file": settings.configFile.path], true) }
            print("Settings file: \(settings.configFile.path)\(file.isEmpty ? " (none; defaults)" : "")")
            print("`origami config set <key> <value>` changes a setting, `origami config unset <key>` resets it.\n")
            let from = [
                "file": "set in the settings file", "flag": "set by an option", "env": "set by an ORIGAMI_ environment variable",
                "model": "saved for this model (models.json)",
            ]
            for (key, meaning) in Settings.keys {
                print("\(key) = \(effective(key))" + (settings.sources[key].flatMap { from[$0] }.map { "   (\($0))" } ?? ""))
                // The meaning, wrapped under its key.
                var line = "   "
                for word in meaning.split(separator: " ") {
                    if line.count + word.count >= 110 {
                        print(line)
                        line = "   "
                    }
                    line += " " + word
                }
                print(line)
            }
        case "get":
            guard arguments.count == 2, keys.contains(arguments[1]) else {
                throw CLIError.usage("Usage: origami config get <\(keys.joined(separator: "|"))>")
            }
            print(effective(arguments[1]))
        case "set":
            if arguments.count >= 2, arguments[1] == "expert_bits" { throw CLIError("expert_bits " + Settings.expertBitsRetired) }
            guard arguments.count >= 3, keys.contains(arguments[1]) else {
                throw CLIError.usage("Usage: origami config set <\(keys.joined(separator: "|"))> <value>")
            }
            let key = arguments[1]
            let text = arguments[2...].joined(separator: " ")
            let value = try Settings.storedValue(key, text, current: settings)
            // The configured model's widths, when it can be read now; any width is kept for one that cannot.
            if key == "hot_copy_dir", !HotCopy.sameDriveExperiment, let dir = value as? String, let gguf = FamilySetup.gguf(settings),
                Stores.volume(gguf).disk == Stores.volume(URL(fileURLWithPath: dir)).disk
            {
                FileHandle.standardError.write(
                    Data("origami: note: \(HotCopy.onModelDrive); it is kept, and the hot copy stays off while it is set\n".utf8))
            }
            if ["stores", "hot_copy_dir", "hot_copy_more", "context_dir"].contains(key) { Storage.remember(settings) }
            // A new data folder leaves the old one's models, counts, tuning and saved prompts behind, unseen.
            let old = settings.dataDir
            let leaves =
                key == "data_dir" && URL(fileURLWithPath: value as? String ?? old.path).standardizedFileURL != old.standardizedFileURL
            if leaves, FileManager.default.fileExists(atPath: old.appendingPathComponent("models.json").path), !yes {
                throw CLIError(
                    "\(old.path) holds models.json: its models, counts, tuning and saved prompts stay there and Origami would no longer "
                        + "see them. To put a model on another drive, use `origami model add <source> --to /Volumes/…`; to change the "
                        + "data folder anyway, run this again with --yes")
            }
            var changes: [String: Any?] = [key: value]
            // A level means what it says: knobs set alone before are reset with it.
            if key == "quality" { for knob in Settings.qualityKnobs { changes.updateValue(nil, forKey: knob) } }
            try modelSettings(&changes, target: settings)
            let forModel = changes.keys.contains(key) ? nil : settings.model
            changes.merge(settings.lanKeyFor(changes)) { $1 }
            try Settings.write(changes, to: settings.configFile)
            print("\(key) = \(Settings.secretKeys.contains(key) && !Settings.isOff(text) ? "(set)" : text)")
            print(appliesLine(key, model: forModel, settings: settings))
            if key == "hot_copy_more" {
                let kept = Set(try Settings.copyPlaces(text).map(\.folder))
                removedPlaces(settings.hotCopyMore.map(\.folder).filter { !kept.contains($0) }, settings: settings, yes: yes)
            }
            if ["listen", "lan_key"].contains(key) || key == "api_key" && settings.listenLAN { try printLAN(flags: flags) }
            if leaves {
                let left = ((try? FileManager.default.contentsOfDirectory(atPath: old.path)) ?? []).filter { !$0.hasPrefix(".") }.sorted()
                if !left.isEmpty {
                    print(
                        "Left in \(old.path): \(left.joined(separator: ", ")); `origami config set data_dir \(old.path)` returns to them.")
                }
            }
        case "unset":
            // `expert_bits` is retired but can still be removed (from the file and the named model's entry).
            guard arguments.count == 2,
                keys.contains(arguments[1]) || arguments[1] == "expert_bits" || settings.unknownKeys.contains(arguments[1])
            else {
                throw CLIError.usage("Usage: origami config unset <key>")
            }
            if arguments[1] == "expert_bits", let name = settings.model, ModelRegistry.entry(name, settings: settings) != nil {
                try ModelRegistry.update(settings) { entries in
                    for i in entries.indices where entries[i]["name"] as? String == name { entries[i]["expert_bits"] = nil }
                }
            }
            if ["stores", "hot_copy_dir", "hot_copy_more", "context_dir"].contains(arguments[1]) { Storage.remember(settings) }
            var changes: [String: Any?] = [arguments[1]: nil]
            try modelSettings(&changes, target: settings)
            try Settings.write(changes, to: settings.configFile)
            print("\(arguments[1]) reset to its default")
            if arguments[1] == "hot_copy_more" { removedPlaces(settings.hotCopyMore.map(\.folder), settings: settings, yes: yes) }
        default: throw CLIError.usage("Usage: origami config [get|set|unset] <key> [value]")
        }
    }

    /// After `hot_copy_more` dropped the places `removed`: their copies of the configured model are deleted with `--yes`
    /// (the server stops reading one at its next start, or at once if its files go), else kept and named.
    private static func removedPlaces(_ removed: [URL], settings: Settings, yes: Bool) {
        guard !removed.isEmpty else { return }
        if yes {
            for folder in HotCopy.removeCopies(removed, settings: settings) { print("deleted the hot copy in \(folder.path)") }
        } else {
            let held = removed.compactMap { HotCopy.copyFolder(settings, place: $0) }.filter {
                FileManager.default.fileExists(atPath: $0.path)
            }
            for folder in held { print("kept the hot copy in \(folder.path); run the same command with --yes to delete it") }
        }
    }

    // MARK: cache

    /// `<data_dir>/prompt-cache/<model>-<context>/<build>/`; 1.0.0 kept its entries in `<model>-<context>/` itself.
    static func cache(_ arguments: [String], flags: [String: [String]], all: Bool, yes: Bool = false) throws {
        let settings = try Settings.load(flags: flags)
        let manager = FileManager.default
        func children(_ url: URL) -> [URL] {
            ((try? manager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
                .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }.sorted { $0.path < $1.path }
        }
        func bytes(_ urls: [URL]) -> UInt64 {
            urls.reduce(0) { total, url in
                total
                    + ((manager.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey])?.compactMap {
                        (try? ($0 as? URL)?.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize
                    }.reduce(0, +)).map(UInt64.init) ?? 0)
            }
        }
        func used(_ folder: URL) -> Date {
            (try? folder.appendingPathComponent("index.json").resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
        }
        // (model folder, files that make one build's cache, label, last use)
        var builds: [(model: URL, items: [URL], label: String, used: Date)] = []
        for model in cacheFolders(settings) {
            for build in children(model) where build.lastPathComponent.count == 16 && build.lastPathComponent.allSatisfy(\.isHexDigit) {
                builds.append((model, [build], model.lastPathComponent + "/" + build.lastPathComponent, used(build)))
            }
            let legacy = ((try? manager.contentsOfDirectory(at: model, includingPropertiesForKeys: nil)) ?? [])
                .filter {
                    $0.lastPathComponent.hasPrefix("entry-") || $0.lastPathComponent == "index.json"
                        || $0.lastPathComponent.hasPrefix("frontiers-")
                }
            if !legacy.isEmpty { builds.append((model, legacy, model.lastPathComponent + " (1.0.0)", used(model))) }
        }
        // GLM's and DeepSeek's `<epoch16>` folders, each its own build, named by its marker when it has one.
        for epoch in children(settings.promptCacheRoot)
        where epoch.lastPathComponent.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil {
            let marker = FamilySetup.epoch(epoch)
            let label =
                epoch.lastPathComponent
                + (marker.map {
                    " (\($0["model"] as? String ?? "?"), \($0["context"] as? Int ?? 0) context, version \($0["version"] as? String ?? "?"))"
                }
                    ?? "")
            builds.append((epoch, [epoch], label, used(epoch)))
        }
        // A GLM or DeepSeek context kept in files while serving (`family-context`): nothing reads it once no model runs.
        let context = settings.promptCacheRoot.appendingPathComponent("family-context", isDirectory: true)
        let scratch = manager.fileExists(atPath: context.path) ? [context] : []
        builds += scratch.map { ($0, [$0], "family-context (a GLM or DeepSeek context kept on disk while serving)", used($0)) }
        // Folders of a model file no registered model has (an earlier version's identity, a replaced file) or, for a family
        // epoch, of another build or GGUF, as the storage map finds them.
        let notes = Dictionary(
            Storage.items(settings, found: ([], nil)).filter { $0.role == "prompts" && $0.unused }.map {
                (($0.path as NSString).lastPathComponent, $0.note ?? "")
            }
        ) { a, _ in a }
        let unreachable = Set(notes.keys)
        switch arguments.first {
        case nil:
            if builds.isEmpty { print("No saved prompts in \(settings.promptCacheRoot.path)") }
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            for build in builds {
                print(
                    "\(build.label)  \(gigabytes(bytes(build.items)))  last used \(formatter.string(from: build.used))"
                        + (!unreachable.contains(build.model.lastPathComponent)
                            ? ""
                            : build.items == [build.model]
                                ? "  (\(notes[build.model.lastPathComponent]!))"
                                : "  (from an earlier Origami version or another model file; safe to delete)"))
            }
        case "prune":
            // A server with no model loaded (`origami stop`) reads none of them, and loads none until `origami start`.
            let state = serverState(settings)
            guard state.hasPrefix("not running") || state.hasPrefix("stopped") else {
                throw CLIError(
                    "An Origami server is \(state), and its model uses these files or may load and use them.\n"
                        + "Run `origami stop` (no model loads until `origami start`), or quit it, then `origami cache prune` again.")
            }
            let newest = Dictionary(grouping: builds, by: \.model).compactMapValues { $0.max { $0.used < $1.used }?.label }
            /// The names of the conversations and prompts a build keeps by name or pin (`PromptCache` never drops them itself).
            func kept(_ build: URL) -> [String] {
                let entries = (try? Settings.readConfig(build.appendingPathComponent("index.json")))?["entries"] as? [String: [String: Any]]
                return (entries ?? [:]).values.compactMap {
                    $0["name"] as? String ?? ($0["pinned"] as? Bool == true ? "a pinned prompt" : nil)
                }
                .sorted()
            }
            let chosen = builds.filter {
                all || newest[$0.model] != $0.label || unreachable.contains($0.model.lastPathComponent) || scratch.contains($0.model)
            }
            /// An index that cannot be read: the server keeps such a build (it may name or pin prompts), and so does prune
            /// unless --yes.
            func unreadable(_ build: URL) -> Bool { (try? Settings.readConfig(build.appendingPathComponent("index.json"))) == nil }
            let held = chosen.filter { !yes && (!$0.items.flatMap(kept).isEmpty || $0.items.contains(where: unreadable)) }
            let going = chosen.filter { build in !held.contains { $0.label == build.label } }
            for build in going {
                print(
                    "Removes \(build.label)  \(gigabytes(bytes(build.items)))"
                        + (build.items.contains(where: unreadable)
                            ? "  (its index could not be read, so any named or pinned prompts in it go too)" : ""))
            }
            for build in held {
                let names = build.items.flatMap(kept)
                let why =
                    (names.isEmpty ? [] : ["it holds " + names.joined(separator: ", ")])
                    + (build.items.contains(where: unreadable)
                        ? ["its index could not be read, so it may hold named or pinned prompts"] : [])
                print(
                    "Keeps \(build.label)  \(gigabytes(bytes(build.items))): \(why.joined(separator: "; ")); "
                        + "`origami cache prune\(all ? " --all" : "") --yes` removes it too")
            }
            var freed: UInt64 = 0
            for build in going {
                freed += bytes(build.items)
                for item in build.items { try manager.removeItem(at: item) }
            }
            print(going.isEmpty ? "Nothing removed." : "Freed \(gigabytes(freed))")
        default: throw CLIError.usage("Usage: origami cache [prune [--all] [--yes]]")
        }
    }

    // MARK: verify

    static func verify(flags: [String: [String]], deep: Bool) throws {
        let settings = try Settings.load(flags: flags)
        var failures = 0
        /// A hot copy (either engine's): every record against its CRC32 (`HotCopy.verify`).
        func hotCopy(_ folder: URL, identity: String, recordBytes: (Int) -> Int?) {
            guard let result = OrigamiEngine.HotCopy.verify(folder: folder, identity: identity, recordBytes: recordBytes) else { return }
            let unchecked = result.unchecked > 0 ? " (\(result.unchecked) written before checksums, not checked)" : ""
            if result.problems.isEmpty {
                print("Hot     \(folder.path): \(result.checked) records OK\(unchecked)")
            } else {
                failures += result.problems.count
                print(
                    "Hot     \(folder.path): \(result.problems.count) problem(s), first: \(result.problems[0]). Those records are read from the model instead; the server rebuilds the copy when its experts change."
                )
            }
        }
        if let url = FamilySetup.gguf(settings) {
            let gguf = try ExpertVariant.source(url, settings: settings)
            let geometry = try ModelGeometry(gguf: gguf)
            let key = FamilySupport.key(gguf)
            let bytes = FamilyExperts.recordBytes(gguf, geometry: geometry)
            hotCopy(
                FamilySetup.hotFolder(settings, key: key), identity: LocalModelIdentity.familyRepresentation(source: key),
                recordBytes: { bytes[$0 / geometry.experts] })
            if failures > 0 { throw CLIError("verification found \(failures) problem(s)") }
            return
        }
        let model = try model(settings)
        ModelLayout.use(model)
        print(
            "Model   \(model.displayName): \(model.tensors.count) tensors in \(model.shardSizes.count) shards, layout \(model.quantization.rawValue) — OK"
        )
        let found = Stores.scan(model: model, settings: settings)
        for item in found {
            for (kind, header) in [(StoreHeader.Kind.pack, item.pack), (.mirror, item.mirror)] where header != nil {
                let label = kind.rawValue
                let problems = try StoreBuilder.verify(
                    model: model, folder: item.folder, kind: kind, compareModel: deep,
                    progress: Terminal.steps("\(label) on \(item.volume.name)"))
                if problems.isEmpty {
                    print("Store   \(item.folder.path) \(label): OK")
                } else {
                    failures += problems.count
                    print(
                        "Store   \(item.folder.path) \(label): \(problems.count) problem(s), first: \(problems[0]). Rebuild it: delete the folder and run `origami setup`."
                    )
                }
            }
        }
        // The hot copy, then each further copy in the hot-copy format: every record against its CRC32.
        // shortcut: `--deep` does not compare these records with the checkpoint; add it with R12's adoption.
        let copies =
            [Stores.plan(model: model, settings: settings, found: found).hotCopyFolder].compactMap { $0 }
            + found.filter { $0.further > 0 }.map { $0.folder.appendingPathComponent(PackStore.furtherCopyFolder, isDirectory: true) }
        for folder in copies {
            hotCopy(folder, identity: Stores.hotCopyIdentity(model)) { model.hotCopyRecordBytes($0) }
        }
        if failures > 0 { throw CLIError("verification found \(failures) problem(s)") }
    }

    // MARK: uninstall

    static func uninstall(flags: [String: [String]], yes: Bool) throws {
        let settings = try Settings.load(flags: flags)
        let manager = FileManager.default
        try requireNoServer(settings, "uninstall")
        // The default data folder is Origami's own; elsewhere (data_dir or context_dir may be a shared folder) only what it made goes.
        let own = settings.dataDir.path == Settings.defaultDataDir.path
        // Every folder that may hold Origami's copies: the stores and copy places, each model's own copy places, and the
        // folders named before (`former-folders.json`, deleted below).
        let entries = (try? ModelRegistry.entries(settings)) ?? []
        let named =
            entries.flatMap { entry in
                [entry["hot_copy_dir"] as? String].compactMap { $0 }
                    + ((entry["hot_copy_more"] as? String).flatMap { try? Settings.copyPlaces($0) } ?? []).map(\.folder.path)
            } + Storage.formerFolders(settings)
        var places = settings.storeFolders.filter { $0 != settings.homeStore }
        for path in named where !places.contains(where: { $0.path == path }) { places.append(URL(fileURLWithPath: path)) }
        var targets = places.flatMap(origamiFolders(in:))
        targets += own ? [settings.homeStore] : origamiFolders(in: settings.homeStore)
        targets += own && settings.contextDir == nil ? [settings.promptCacheRoot] : cacheFolders(settings)
        targets += [
            settings.calibrationFile, settings.driveSpeedsFile, settings.configFile, Settings.logDir, ModelRegistry.file(settings),
            settings.dataDir.appendingPathComponent("fast-mode.json"),
        ]
        targets += ((try? manager.contentsOfDirectory(at: settings.dataDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("routing-") && $0.pathExtension == "json" }
        targets += ["former-folders.json", "moving.json"].map { settings.dataDir.appendingPathComponent($0) }
        targets.append(ExpertVariant.folder(settings))  // links and manifests only (`model add --base`)
        // Origami's own data folder: everything in it but the downloaded models.
        if own {
            targets += ((try? manager.contentsOfDirectory(at: settings.dataDir, includingPropertiesForKeys: nil)) ?? []).filter {
                $0.standardizedFileURL.path != settings.modelsDir.standardizedFileURL.path
            }
        }
        // GLM and DeepSeek: `family-<key>/hot` and `family-<key>/pack-<type>` in a store or hot_copy_dir, their counts and
        // default hot copy in the data folder's `family/<key>`.
        func keyed(_ folder: URL, _ pattern: String) -> [URL] {
            ((try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []).filter {
                $0.lastPathComponent.range(of: pattern, options: .regularExpression) != nil
            }
        }
        let familyFolders = (places + (own ? [] : [settings.homeStore])).flatMap { keyed($0, "^family-[0-9a-f]{16}$") }
        targets += familyFolders.flatMap { base in
            keyed(base, "^(hot|pack-.+)$").filter {
                $0.lastPathComponent == "hot" || manager.fileExists(atPath: $0.appendingPathComponent("pack.json").path)
            }
        }
        targets += keyed(settings.dataDir.appendingPathComponent("family"), "^[0-9a-f]{16}$")
        // A hot_copy_dir (or home store) that is also a store lists its folders twice: a second delete would fail.
        var seen = Set<String>()
        targets = targets.filter { manager.fileExists(atPath: $0.path) && seen.insert($0.standardizedFileURL.path).inserted }
        // One inside another goes with it.
        targets = targets.filter { target in !seen.contains { target.standardizedFileURL.path.hasPrefix($0 + "/") } }
        guard !targets.isEmpty else {
            print("Nothing to remove.")
            return
        }
        print("This deletes Origami's expert copies, prompt cache, measurements and settings:")
        for target in targets { print("  \(target.path)") }
        print(
            "Model weights are kept\(FileManager.default.fileExists(atPath: settings.modelsDir.path) ? " (downloads stay in \(settings.modelsDir.path))" : "")."
        )
        guard Terminal.confirm("Delete these?", yes: yes) else { throw CLIError("Nothing was deleted (not confirmed).") }
        var failed: [String] = []
        for target in targets {
            do { try manager.removeItem(at: target) } catch { failed.append("\(target.path) (\(error.localizedDescription))") }
        }
        // Their parent folders too, once empty (rmdir leaves a folder that still holds anything).
        for folder in familyFolders + [settings.dataDir.appendingPathComponent("family")] { rmdir(folder.path) }
        rmdir(settings.homeStore.path)
        if settings.contextDir == nil { rmdir(settings.promptCacheRoot.path) }
        guard failed.isEmpty else {
            throw CLIError(
                "Could not delete:\n  " + failed.joined(separator: "\n  ") + "\nCheck their permissions and delete them by hand.")
        }
        // What this leaves: the binary and the Hub bundle beside it, a LaunchAgent if one was made, the downloaded models.
        let binary = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent()
        let agent = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/LaunchAgents/com.origami.serve.plist")
        print(
            "Done. Left in place: the origami binary"
                + (binary.map { " and \($0.appendingPathComponent("swift-transformers_Hub.bundle").path)" } ?? " and its Hub bundle")
                + (manager.fileExists(atPath: agent.path) ? ", the LaunchAgent \(agent.path) (unload it first)" : "")
                + (manager.fileExists(atPath: settings.modelsDir.path) ? ", the downloaded models in \(settings.modelsDir.path)" : "")
                + ". Delete them by hand to finish.")
    }
}
