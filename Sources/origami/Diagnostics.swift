import Darwin
import Foundation
import OrigamiEngine

/// `origami inspect`, `doctor`, `bench` and `dashboard`: diagnosis and measurement without Python.
enum Diagnostics {
    static func gigabytes(_ bytes: Int64) -> String { OrigamiPlan.gigabytes(UInt64(max(0, bytes))) }

    // MARK: inspect

    /// What a checkpoint holds per tensor family, from its config and safetensors headers only,
    /// so a checkpoint Origami refuses can still be described.
    static func inspect(_ arguments: [String], flags: [String: [String]], json: Bool) throws {
        let settings = try Settings.load(flags: flags)
        let folder: URL
        let gguf = arguments.first.map { Models.isGGUF(Settings.expand($0)) } ?? false
        if let spec = arguments.first {
            let path = Settings.expand(spec)
            if gguf {
                folder = URL(fileURLWithPath: path)
            } else if FileManager.default.fileExists(atPath: path + "/config.json") {
                folder = URL(fileURLWithPath: path, isDirectory: true)
            } else if let found = Models.locate(spec, settings: settings) {
                folder = found
            } else {
                throw CLIError("No model folder at \(spec): it needs config.json and .safetensors files")
            }
        } else {
            guard let found = Models.locate(settings.model, settings: settings) else {
                throw CLIError("No model found; name one: origami inspect <folder>")
            }
            folder = found
        }
        // An if, not a ternary: Swift 6.4 cannot type-check the ternary over these two tuple-returning calls.
        let (report, families, verdict) = if gguf { try ggufReport(folder) } else { try inspectReport(folder) }
        if json { return Commands.emit([], report, true) }
        print(
            "Model    \(folder.path)\n         \(report["model_type"] ?? "?"), \(gigabytes(report["bytes"] as! Int64)) in \(report["shards"] ?? 0) shard\(report["shards"] as? Int == 1 ? "" : "s")\n"
        )
        print(
            "Family".padding(toLength: 34, withPad: " ", startingAt: 0) + "Tensors".padding(toLength: 9, withPad: " ", startingAt: 0)
                + "Size".padding(toLength: 10, withPad: " ", startingAt: 0) + "Format")
        for (family, entry) in families {
            let formats = entry.formats.sorted { $0.value > $1.value }.map { entry.formats.count == 1 ? $0.key : "\($0.key) ×\($0.value)" }
                .joined(separator: ", ")
            print(
                family.padding(toLength: 34, withPad: " ", startingAt: 0)
                    + "\(entry.tensors)".padding(toLength: 9, withPad: " ", startingAt: 0)
                    + gigabytes(entry.bytes).padding(toLength: 10, withPad: " ", startingAt: 0) + formats)
        }
        print("\n\(verdict)")
    }

    /// `inspect --family`: any checkpoint, local or on the Hugging Face Hub, judged from its headers alone (no weights are
    /// downloaded). A repo id lists every model in it.
    static func inspectFamily(_ arguments: [String], json: Bool) throws {
        guard let spec = arguments.first else {
            throw CLIError.usage("inspect --family needs a folder, a .gguf file or a Hugging Face repo id")
        }
        let path = Settings.expand(spec)
        let reports: [CompatibilityReport]
        if FileManager.default.fileExists(atPath: path) {
            reports = [try ModelCompatibility.check(local: URL(fileURLWithPath: path))]
        } else if spec.split(separator: "/").count == 2, !spec.hasPrefix("/"), !spec.hasPrefix(".") {
            reports = try ModelCompatibility.check(hub: HubRepo(spec))
        } else {
            throw CLIError("No model at \(spec), and it is not a Hugging Face repo id (owner/name)")
        }
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(reports), as: UTF8.self))
            return
        }
        if reports.isEmpty { print("\(spec) holds no GGUF or safetensors model") }
        for report in reports {
            print(
                "\(report.source)\n  \(report.family?.displayName ?? "unknown family") (\(report.format), \(report.architecture)), \(gigabytes(report.totalBytes))"
            )
            if let g = report.geometry {
                print(
                    "  \(g.layers) layers, hidden \(g.hidden), \(g.experts) experts (\(g.expertsPerToken) per token, width \(g.expertWidth)), "
                        + "\(g.mixers.filter { $0 == .recurrent }.count) recurrent, vocabulary \(g.vocab)")
            }
            for (role, types) in report.types.sorted(by: { (report.bytes[$0.key] ?? 0) > (report.bytes[$1.key] ?? 0) }) {
                let list = types.sorted { $0.value > $1.value }.map { types.count == 1 ? $0.key : "\($0.key) ×\($0.value)" }.joined(
                    separator: ", ")
                print(
                    "    " + role.padding(toLength: 24, withPad: " ", startingAt: 0)
                        + gigabytes(report.bytes[role] ?? 0).padding(toLength: 10, withPad: " ", startingAt: 0) + list)
            }
            print("  \(report.verdict.rawValue): \(report.reason)\n")
        }
    }

    /// What a model folder holds per tensor family, and whether Origami runs it (`inspect`, and the manifest of `model add`).
    static func inspectReport(_ folder: URL) throws -> (
        report: [String: Any], families: [(key: String, value: (tensors: Int, bytes: Int64, formats: [String: Int]))], verdict: String
    ) {
        let config =
            (try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("config.json"))) as? [String: Any])
            ?? [:]
        let quantization = (config["quantization"] ?? config["quantization_config"]) as? [String: Any] ?? [:]
        let index =
            (try? Data(contentsOf: folder.appendingPathComponent("model.safetensors.index.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["weight_map"] as? [String: String]
        let shards = index.map { Array(Set($0.values)).sorted() } ?? ["model.safetensors"]
        // The index is untrusted: a shard name may not leave the folder.
        guard shards.allSatisfy({ !$0.isEmpty && !$0.contains("/") && !$0.hasPrefix(".") }) else {
            throw CLIError(
                "\(folder.path)/model.safetensors.index.json names a shard outside the folder; the checkpoint is damaged or unsafe, so download it again"
            )
        }
        var tensors: [String: TensorSpan] = [:]
        for shard in shards {
            let url = folder.appendingPathComponent(shard)
            var info = stat()
            guard stat(url.path, &info) == 0 else {
                throw CLIError("Missing shard \(shard) in \(folder.path); the download may be incomplete")
            }
            for (name, span) in try ModelSource.header(of: url, file: shard) { tensors[name] = span }
        }
        var families: [String: (tensors: Int, bytes: Int64, formats: [String: Int])] = [:]
        for (name, span) in tensors where !name.hasSuffix(".scales") && !name.hasSuffix(".biases") {
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            var bytes = span.length
            var format = span.dtype.lowercased()
            // A quantized weight carries scales (and biases); its bits and group come from the config, per module or global.
            if let scales = tensors[base + ".scales"] {
                bytes += scales.length + (tensors[base + ".biases"]?.length ?? 0)
                let module = base.hasPrefix("language_model.") ? String(base.dropFirst("language_model.".count)) : base
                let rule = quantization[module] as? [String: Any] ?? quantization[base] as? [String: Any] ?? quantization
                format = "\(rule["bits"].map { "\($0)" } ?? "?")-bit g\(rule["group_size"].map { "\($0)" } ?? "?")"
                if let mode = rule["mode"] as? String, mode != "affine" { format += " " + mode }
            }
            let family = Self.family(name)
            var entry = families[family] ?? (0, 0, [:])
            entry.tensors += 1
            entry.bytes += bytes
            entry.formats[format, default: 0] += 1
            families[family] = entry
        }
        let verdict: String
        do {
            let model = try ModelSource(directory: folder)
            verdict =
                "Origami runs this checkpoint."
                + (model.ggufExperts == nil ? "" : " Its routed experts are read from the GGUF as they are.")
            if let gguf = model.ggufExperts {
                var formats: [String: Int] = [:]
                for layer in gguf.layers { for part in layer.parts { formats[GGUFQuant.name(part.type), default: 0] += 1 } }
                families["experts (in the GGUF, \(gguf.experts) per layer)"] = (gguf.layers.count * 3, Int64(gguf.bankBytes), formats)
            }
        } catch {
            if let converter = try? CheckpointConverter(source: folder) {
                verdict =
                    "Origami converts it once into the 4-bit layout it runs (\(OrigamiPlan.gigabytes(converter.outputBytes))): origami model add \(folder.path)"
            } else {
                verdict = "Origami refuses it: \(error)"
            }
        }
        let total = tensors.values.reduce(Int64(0)) { $0 + $1.length }
        let ordered = families.sorted { $0.value.bytes > $1.value.bytes }
        let report: [String: Any] = [
            "folder": folder.path, "model_type": config["model_type"] ?? NSNull(), "bytes": total, "shards": shards.count,
            "runs": verdict.hasPrefix("Origami runs"),
            "verdict": verdict,
            "families": ordered.map {
                ["family": $0.key, "tensors": $0.value.tensors, "bytes": $0.value.bytes, "formats": $0.value.formats]
            },
        ]
        return (report, ordered, verdict)
    }

    /// A GGUF's tensors per family and encoding, and whether Origami converts it (`inspect`).
    static func ggufReport(_ url: URL) throws -> (
        report: [String: Any], families: [(key: String, value: (tensors: Int, bytes: Int64, formats: [String: Int]))], verdict: String
    ) {
        let gguf = try GGUFSource(url)
        var families: [String: (tensors: Int, bytes: Int64, formats: [String: Int])] = [:]
        for tensor in gguf.tensors.values {
            let name = tensor.name
            let family: String
            if name.contains("nextn") {
                family = "multi-token prediction (not used)"
            } else if name.contains("_exps") {
                family = "experts"
            } else if name.contains("_shexp") {
                family = "shared expert"
            } else if name.contains("ffn_gate_inp") {
                family = "router"
            } else if name.contains("ssm_") || name.contains("attn_qkv") || name.contains("attn_gate") {
                family = "linear attention"
            } else if name.contains("attn_") || name.contains("indexer") {
                family = "attention"
            } else if name.contains("hc_") {
                family = "hyper-connections"
            } else if name.contains("ple") {
                family = "per-layer embeddings"
            } else if name.hasPrefix("token_embd") || name.hasPrefix("output.") {
                family = "token embeddings and output"
            } else {
                family = "other"
            }
            var entry = families[family] ?? (0, 0, [:])
            entry.tensors += 1
            entry.bytes += max(0, tensor.length)
            entry.formats[GGUFQuant.name(tensor.type), default: 0] += 1
            families[family] = entry
        }
        let verdict =
            CheckpointConverter.ggufProblem(gguf).map { "Origami refuses it: \($0)" }
            ?? (CheckpointConverter.keepsExpertsNative(gguf)
                ? "Origami converts it once into the 4-bit layout it runs, all but its routed experts (about 37 GB), and reads those from the GGUF as they are: origami model add \(url.path)"
                : "Origami converts it once into the 4-bit layout it runs (about 105 GB): origami model add \(url.path)")
        let ordered = families.sorted { $0.value.bytes > $1.value.bytes }
        var architecture = "?"
        if case .string(let value)? = gguf.metadata["general.architecture"] { architecture = value }
        let report: [String: Any] = [
            "folder": url.path, "model_type": architecture + " (GGUF)", "bytes": Int64(gguf.bytes), "shards": gguf.files.count,
            "runs": false, "converts": verdict.hasPrefix("Origami converts"), "verdict": verdict,
            "families": ordered.map {
                ["family": $0.key, "tensors": $0.value.tensors, "bytes": $0.value.bytes, "formats": $0.value.formats]
            },
        ]
        return (report, ordered, verdict)
    }

    static func family(_ name: String) -> String {
        if name.hasPrefix("vision_tower.") || name.contains(".visual.") { return "vision (not used yet)" }
        if name.hasPrefix("mtp.") || name.contains(".mtp.") { return "multi-token prediction (not used)" }
        if name.contains(".switch_mlp.") || name.contains(".experts.") { return "experts" }
        if name.contains(".shared_expert") { return "shared expert" }
        if name.contains(".mlp.gate.") { return "router" }
        if name.contains(".linear_attn.") { return "linear attention" }
        if name.contains(".self_attn.") { return "attention" }
        if name.contains("hyper_connection") { return "hyper-connections" }
        if name.contains(".ple.") { return "per-layer embeddings" }
        if name.contains("embed_tokens") || name.contains("lm_head") { return "token embeddings and output" }
        return "other"
    }

    // MARK: doctor

    /// Whether two paths name the same file or folder (a link, /tmp for /private/tmp).
    static func same(_ a: String, _ b: String) -> Bool {
        URL(fileURLWithPath: a).resolvingSymlinksInPath().standardizedFileURL.path
            == URL(fileURLWithPath: b).resolvingSymlinksInPath().standardizedFileURL.path
    }

    /// Checks what can go wrong before the model loads: settings, Mac, model, template, stores, memory, disk.
    static func doctor(flags: [String: [String]]) async throws {
        var problems = 0
        func report(_ ok: Bool, _ text: String) {
            print((ok ? "ok    " : "fix   ") + text)
            if !ok { problems += 1 }
        }
        let settings = try Settings.load(flags: flags)
        report(
            true,
            "settings: " + settings.configFile.path
                + (FileManager.default.fileExists(atPath: settings.configFile.path) ? "" : " (none; defaults)"))
        let host: OrigamiHost?
        do {
            let current = try OrigamiHost.current()
            report(true, "Mac: \(current.gpuName), \(Commands.ram(current.physicalMemory)) memory")
            for note in current.notes { print("      " + note) }
            host = current
        } catch {
            host = nil
            report(false, "Mac: \(error)")
        }
        let free = Stores.volume(settings.dataDir).free
        report(
            free >= 10_000_000_000,
            "disk: \(OrigamiPlan.gigabytes(free)) free in \(settings.dataDir.path)"
                + (free >= 10_000_000_000
                    ? "" : "; saved prompts need at least 10 GB free there (move it with `origami config set data_dir …`)"))
        let entries = PromptCache.regularEntries(context: settings.context, entries: settings.promptCacheEntries)
        print(
            "info  saved prompts: \(entries) at a \(settings.display("context")) context, each up to "
                + OrigamiPlan.gigabytes(PromptCache.entryDiskBytes(context: settings.context)) + " on disk"
                + (settings.context > 8192 ? " (prompt_cache_entries sets how many)" : "") + ", in \(settings.promptCacheRoot.path)")
        // An explicit context_dir on a drive without file ownership: the server saves and reuses no prompt there.
        let prompts = settings.promptCacheRoot
        if settings.contextDir != nil, Folders.timed(2, on: prompts.path, { PromptCache.driveKeepsOwnership(prompts) }) == false {
            report(
                false,
                "saved prompts: off in \(prompts.path): \(Stores.driveName(prompts.path)) has no file ownership (exFAT or FAT), so every "
                    + "follow-up reads its whole conversation again; `origami config unset context_dir` keeps them on this Mac's own drive")
        }
        let server = Commands.serverState(settings)
        print("info  server: " + server)
        if settings.listenLAN && settings.remoteKey != nil {
            print("warn  network: listen lan sends the key in clear over plain HTTP; for TLS, see docs/integration.md#tls")
        }
        // A retired `expert_bits` still set, and the smaller experts an earlier Origami made: told, never deleted here.
        if let note = settings.expertBitsNote { print("warn  " + note) }
        if let note = settings.unknownKeysNote { print("warn  settings: " + note) }
        let mounted = Storage.volumes()
        Storage.noteVolumes(settings, volumes: mounted)
        for item in Storage.relinks(settings, volumes: mounted) {
            report(false, "\(item.setting): \(item.from) is missing; it is at \(item.to) (\(item.how)): run `origami relink`")
        }
        for item in Stores.retiredExperts(settings) {
            print(
                "warn  unused: \(item.path) (\(Commands.gigabytes(item.bytes))): smaller experts an earlier Origami made; nothing reads them now"
            )
            print("      delete: \(item.delete)   (or in the dashboard's Models view)")
        }
        // A GLM or DeepSeek GGUF (named or registered, as `serve` finds it), else a Qwen model folder.
        let family = FamilySetup.named(settings)
        let asked = family ?? Models.locate(settings.model, settings: settings)
        // The running model's drive: how much of its reading went far below its own speed (`SlowDriveReads`), when the
        // server runs the model asked about.
        let status = server.hasPrefix("not ") ? [:] : DashboardState.status(port: UInt16(settings.port), key: settings.apiKey)
        // A save skipped for space is a problem (a copy could not give enough back); a copy that gave back is news.
        if let disk = DiskBudget.line(status["disk_budget"]) {
            let skipped = ((status["disk_budget"] as? [[String: Any]]) ?? []).contains { ($0["saves_skipped"] as? Int ?? 0) > 0 }
            if skipped { report(false, disk) } else { print("info  " + disk) }
        }
        if let running = status["model"] as? [String: Any], let path = running["path"] as? String,
            asked.map({ Self.same($0.path, path) }) != true
        {
            print("info  a server on port \(settings.port) is serving \(running["name"] as? String ?? path)")
        } else if (status["model"] as? [String: Any])?["path"] != nil {
            let row = ((status["experts_from"] as? [String: Any])?["sources"] as? [[String: Any]])?.first { $0["role"] as? String == "all" }
            if let row, let slow = row["slow_reads"] as? [String: Any] {
                let bytes = (row["bytes"] as? NSNumber)?.uint64Value ?? 0
                let file = row["experts_file"] as? [String: Any]
                let experts = file.map { "\($0["path"] as? String ?? "?") on \($0["drive"] as? String ?? "?")" }
                let total = bytes + ((file?["bytes"] as? NSNumber)?.uint64Value ?? 0)
                if let advice = SlowDriveReads.advice(slow, modelBytes: total, experts: experts) {
                    report(false, "model drive: " + advice)
                } else {
                    report(
                        true,
                        String(
                            format:
                                "model drive: %.0f%% of %.0f s of reading below a third of its own speed (more than %.0f%% would call for `origami refresh`)",
                            100 * (slow["share"] as? Double ?? 0), slow["busy_seconds"] as? Double ?? 0, 100 * SlowDriveReads.threshold))
                }
            }
        }
        // Each drive's reading since its files were last written, from the servers' counts (`DriveReadLog`); only hot
        // copies carry a generation.
        for (path, history) in DriveReadLog.read(DriveReadLog.file(settings.dataDir)).sorted(by: { $0.key < $1.key }) {
            let line = "drive reads of \(path): " + history.text(hot: history.generation != nil)
            if history.high && history.settled {
                report(false, line)
            } else {
                print((history.high || history.due ? "warn  " : "info  ") + line)
            }
        }
        if let family {
            let (rows, _) = FamilySetup.rows(family, settings: settings, host: host)
            for row in rows {
                report(row.ok, "\(row.label): \(row.text)")
                for line in row.more { print("      " + line) }
            }
            for line in Calibrations.lines(settings) { report(true, "drive speed: " + line) }
            if problems > 0 { throw CLIError("\(problems) problem\(problems == 1 ? "" : "s") to fix") }
            return
        }
        guard let url = asked else {
            report(
                false,
                settings.model.map { "model: " + Commands.missingModel($0, settings: settings) }
                    ?? "model: none found; run `origami setup` (it can download the model) or pass --model <folder>")
            throw CLIError("\(problems) problem\(problems == 1 ? "" : "s") to fix")
        }
        do {
            let model = try ModelSource(directory: url)
            report(true, "model: \(model.displayName) (\(model.quantization.rawValue)) in \(model.directory.path)")
            if let note = ExpertVariant.note(model) { print("      " + note) }
            if let file = ExpertVariant.filesSummary(model.ggufExperts?.files.map(\.path) ?? []) {
                print(
                    "      \(Commands.gigabytes(model.totalBytes)) of the base's files + \(Commands.gigabytes(file.bytes)) low-bit experts file: \(file.text)"
                )
            }
            let probe = try await TokenizerStack(modelDirectory: model.directory).templateCheck()
            report(
                probe?.thinkingSwitch ?? false,
                probe == nil
                    ? "chat template: cannot render a test conversation; the model folder's chat_template.jinja or tokenizer_config.json is missing or broken: download the model again (`origami inspect` shows what the folder holds)"
                    : probe!.thinkingSwitch
                        ? "chat template: thinking turns on and off"
                        : "chat template: ignores the thinking switch, so reasoning_effort none still thinks; set max_thinking_tokens to bound it, or use the model's original template"
            )
            let found = Stores.scan(model: model, settings: settings)
            for item in found {
                let contents = [
                    item.pack.map { _ in "full expert copy" }, item.mirror.map { "\($0.records.count) most-used experts" },
                    item.further > 0 ? "\(item.further) most-used experts" : nil,
                    Stores.hotCopyHeld(item, model: model),
                ].compactMap { $0 }
                if item.problems.isEmpty {
                    report(true, "store \(item.store.path): " + (contents.isEmpty ? "empty" : contents.joined(separator: ", ")))
                }
                let listed = settings.stores.contains { $0.path == item.store.path }
                for problem in item.problems {
                    report(
                        false,
                        "store \(item.store.path): \(problem)"
                            + (problem == "drive not connected"
                                ? "; connect it" : problem.hasPrefix("drive") ? "" : "; run `origami setup` to rebuild it")
                            + (listed ? ", or stop using it: `origami store remove \(item.store.path)`" : ""))
                }
            }
            let storage = Stores.plan(model: model, settings: settings, found: found)
            var plan: OrigamiPlan?
            if let host {
                do {
                    plan = try OrigamiPlan.make(
                        model: model, host: host, context: settings.context, memoryLimit: settings.memoryLimit,
                        expertMemory: settings.expertMemory,
                        storage: storage, promptCache: settings.promptCacheRoot)
                } catch { report(false, "memory: \(error)") }
            }
            if storage.packFolder == nil && storage.hotCopyFolder == nil {
                // Copies cannot help when every expert stays in memory, and hot_copy off asks for none.
                if plan?.allResident == true {
                    report(true, "reads: from the model files (every expert stays in memory once read)")
                } else if settings.hotCopy == 0 {
                    report(true, "reads: from the model files (hot_copy is off)")
                } else {
                    report(false, "reads: straight from the model files; run `origami setup` to build faster expert copies")
                }
            } else if storage.packFolder != nil && !storage.packSharesModelDrive && storage.calibration == nil {
                report(
                    false, "reads: \(Stores.reads(storage)), drive speeds not measured yet; the next start or `origami setup` measures them"
                )
            } else {
                report(
                    true,
                    "reads: "
                        + (storage.packFolder == nil
                            ? "model files and the hot copy (built and kept while serving)"
                            : storage.packSharesModelDrive && storage.packDrives == 1
                                ? "expert copy on the model's drive" : "\(Stores.reads(storage)), measured"))
            }
            for line in Calibrations.lines(settings) { report(true, "drive speed: " + line) }
            if let plan {
                report(
                    true,
                    "memory: up to \(OrigamiPlan.gigabytes(plan.memory.cap)), \(OrigamiPlan.gigabytes(plan.memory.start)) if started now; \(OrigamiPlan.gigabytes(plan.memory.floor)) always left for other apps"
                )
                for note in plan.notes { print("      " + note) }
            }
        } catch { report(false, "model at \(url.path): \(error)") }
        if problems > 0 { throw CLIError("\(problems) problem\(problems == 1 ? "" : "s") to fix") }
    }

    // MARK: bench

    /// Times three fixed requests against a running server: a short question, a story, and a document
    /// with a fresh prefix so no saved prompt can answer it.
    static func bench(flags: [String: [String]]) async throws {
        let settings = try Settings.load(flags: flags)
        try Commands.requireReady(settings)
        let base = URL(string: "http://127.0.0.1:\(settings.port)")!
        let document = (0..<230).map { "Line \($0): the warehouse inventory lists \($0 * 7 % 101) crates of item \($0)." }.joined(
            separator: " ")
        let cases: [(String, String, Int)] = [
            ("short question", "Explain in three sentences why the sky is blue.", 160),
            ("story", "Write a story of about 250 words about a paper crane that learns to fly.", 400),
            (
                "6K-token document",
                "Reference \(UUID().uuidString).\n" + document + "\nHow many crates does line 117 list? Answer with the number only.", 16
            ),
        ]
        print(
            "Benchmark against \(base.absoluteString) (temperature 0, thinking off). The first run after a start is slower while memory fills.\n"
        )
        print("Request".padding(toLength: 20, withPad: " ", startingAt: 0) + "Prompt  Reused  First word  Reading     Writing")
        for (label, prompt, maximum) in cases {
            let run = try await timed(base, prompt: prompt, maximumTokens: maximum, apiKey: settings.apiKey)
            // Short prompts and short answers measure start-up and saving, not reading or writing speed.
            let reading =
                run.prompt - run.cached >= 1000 && run.ttft > 0
                ? String(format: "%5.0f tok/s", Double(run.prompt - run.cached) / run.ttft) : "    –      "
            let writing =
                run.completion >= 16 && run.total > run.ttft
                ? String(format: "%5.1f tok/s", Double(run.completion - 1) / (run.total - run.ttft)) : "    –"
            print(
                label.padding(toLength: 20, withPad: " ", startingAt: 0)
                    + String(format: "%6d  %6d  %8.1f s  ", run.prompt, run.cached, run.ttft) + reading + "  " + writing)
        }
    }

    /// `extra`: more request fields (the tuner's `origami_tune`); `text` is the reply; `gaps` the seconds between its text events.
    static func timed(_ base: URL, prompt: String, maximumTokens: Int, apiKey: String?, extra: [String: Any] = [:], deadline: Date? = nil)
        async throws -> (ttft: Double, total: Double, prompt: Int, cached: Int, completion: Int, text: String, gaps: [Double])
    {
        var request = URLRequest(url: base.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        if let apiKey { request.setValue("Bearer " + apiKey, forHTTPHeaderField: "Authorization") }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 3600
        request.httpBody = try JSONSerialization.data(
            withJSONObject: [
                "model": "origami", "stream": true, "stream_options": ["include_usage": true], "temperature": 0, "reasoning_effort": "none",
                // A benchmark saves nothing: each run would otherwise take a prompt-cache slot (one of two at 256K).
                // Exact, whatever level the server defaults to, so the numbers are comparable.
                "prompt_cache_policy": "no_store", "origami_mode": "exact", "max_tokens": maximumTokens,
                "messages": [["role": "user", "content": prompt]],
            ]
            .merging(extra) { $1 })
        let start = Date()
        var first: Date?
        var last: Date?
        var gaps: [Double] = []
        var usage: [String: Any]?
        var text = ""
        let session: URLSession
        if let deadline {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw CLIError("Tuning time limit reached") }
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForResource = remaining
            configuration.timeoutIntervalForRequest = min(remaining, 60)
            request.timeoutInterval = remaining
            session = URLSession(configuration: configuration)
        } else {
            session = .shared
        }
        defer { if deadline != nil { session.invalidateAndCancel() } }
        let (bytes, response) = try await session.bytes(for: request)
        guard let status = (response as? HTTPURLResponse)?.statusCode, status == 200 else {
            var body = ""
            for try await line in bytes.lines { body += line }
            let message =
                ((try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])?["error"] as? [String: Any])?["message"]
                as? String
            throw CLIError(
                "The server refused the benchmark request (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)): \(message ?? body)")
        }
        for try await line in bytes.lines where line.hasPrefix("data: ") && line != "data: [DONE]" {
            guard let event = try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any] else { continue }
            if let error = event["error"] as? [String: Any] { throw CLIError("The server reported: \(error["message"] ?? "an error")") }
            if let value = event["usage"] as? [String: Any] { usage = value }
            let delta = (event["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any]
            if ["content", "reasoning_content"].contains(where: { !((delta?[$0] as? String) ?? "").isEmpty }) {
                let now = Date()
                if let last { gaps.append(now.timeIntervalSince(last)) }
                if first == nil { first = now }
                last = now
            }
            text += (delta?["reasoning_content"] as? String ?? "") + (delta?["content"] as? String ?? "")
        }
        var total = Date().timeIntervalSince(start)
        // The tuning-flow check's replies report their own first-token and writing seconds (`reported_timing`), so its
        // decisions do not depend on how fast the fixture really ran.
        if Tuning.checkHost?["reported_timing"] as? Bool == true, let timing = usage?["check_timing"] as? [String: Double],
            let ttft = timing["ttft"], let decode = timing["decode"]
        {
            (first, total, gaps) = (start.addingTimeInterval(ttft), ttft + decode, [])
        }
        return (
            (first ?? Date()).timeIntervalSince(start), total, usage?["prompt_tokens"] as? Int ?? 0,
            (usage?["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int ?? 0, usage?["completion_tokens"] as? Int ?? 0,
            text, gaps
        )
    }

    // MARK: report

    /// A paste-ready block for the support table: this Mac, the memory plan, the stores and the benchmark.
    static func report(flags: [String: [String]]) async throws {
        let settings = try Settings.load(flags: flags)
        let status = try Commands.statusReport(flags: flags).report
        let json = String(
            decoding: try JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]),
            as: UTF8.self)
        print(
            "Paste everything from the first ``` to the last one into a GitHub issue. Home-folder paths are shortened to ~; "
                + "paths on other drives are shown as they are, so edit them out if you like.\n")
        print("```")
        print("origami \(version), macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
        // Whole path components only: /Users/alex must not rewrite /Users/alexandra.
        let home = NSHomeDirectory()
        print(json.replacingOccurrences(of: home + "/", with: "~/").replacingOccurrences(of: "\"" + home + "\"", with: "\"~\""))
        if Commands.serverState(settings).hasPrefix("ready") {
            try await bench(flags: flags)
        } else {
            print("Speed not measured: start `origami serve`, then run `origami report` again.")
        }
        print("```")
    }

    // MARK: dashboard

    static func dashboard(flags: [String: [String]]) throws {
        let settings = try Settings.load(flags: flags)
        let url = "http://127.0.0.1:\(settings.port)/dashboard"
        if Commands.serverState(settings).hasPrefix("not running") {
            print("Origami is not running on port \(settings.port); the page connects once `origami serve` starts.")
        }
        let open = Process()
        // The page reads the key from the fragment, which the browser never sends.
        let key = settings.apiKey.map { "#key=" + ($0.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? "") } ?? ""
        open.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        open.arguments = [url + key]
        try open.run()
        open.waitUntilExit()
        print(url)
    }
}
