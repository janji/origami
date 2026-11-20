import Darwin
import Foundation
import OrigamiEngine

/// The self-tuner: `origami tune` (and the dashboard's Tune, Re-check and Undo, which run it as a job) measures the
/// running model's execution switches against its current settings, request by request in the one loaded model
/// (`origami_tune`), and keeps the single winner in `<data_dir>/tuning.json`, per model and context, bound to this Mac,
/// its drives and `SelfTuning.version`. A serving model applies a matching result at start: the user's own settings
/// beat it, and a missing, broken or stale file only means the defaults.
enum Tuning {
    static func file(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning.json") }
    static func proposalFile(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning-proposal.json") }
    static func historyFile(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning-history.jsonl") }
    static func memoryHistory(_ settings: Settings) -> MemoryHistory {
        // A model loaded at a trial limit (`origami tune --memory`) neither reads nor teaches this Mac's history: its cuts
        // are the measurement, kept in a throwaway file.
        let trialHistory = ProcessInfo.processInfo.environment[trialVariable].map { _ in
            FileManager.default.temporaryDirectory.appendingPathComponent("origami-memory-trial-\(getpid()).json")
        }
        if let trialHistory { try? FileManager.default.removeItem(at: trialHistory) }
        return MemoryHistory(
            file: trialHistory ?? settings.dataDir.appendingPathComponent("memory-history.json"),
            identity: "\(macModel) \(ProcessInfo.processInfo.physicalMemory)",
            scope: MemoryHistory.Scope(model: settings.model ?? "", context: settings.context))
    }

    /// Removes the throwaway histories of trial models that have ended (`memoryHistory`); a running one keeps its own.
    static func removeTrialHistories() {
        let folder = FileManager.default.temporaryDirectory
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] {
            guard name.hasPrefix("origami-memory-trial-"), name.hasSuffix(".json"),
                let pid = pid_t(name.dropFirst("origami-memory-trial-".count).dropLast(".json".count)),
                kill(pid, 0) != 0 && errno == ESRCH
            else { continue }
            try? FileManager.default.removeItem(at: folder.appendingPathComponent(name))
        }
    }

    static var macModel: String {
        var size = 0
        guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// What a result is bound to: the Mac, its GPU and memory, the tuning version, the model and the drives it reads.
    static func identity(model: String, gpu: String, drives: [URL]) -> [String: Any] {
        let uuids = Set(drives.compactMap { try? $0.resourceValues(forKeys: [.volumeUUIDStringKey]).volumeUUIDString }).sorted()
        return [
            "tuning_version": SelfTuning.version, "mac": macModel, "gpu": gpu, "memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "model": String(model.prefix(12)), "drives": uuids.joined(separator: ","),
            "macos": ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
        ]
    }

    /// A serving model's tuning, read once at start: its key, identity, saved entry and whether it applies (`state`:
    /// `none`, `current`, `recheck`, `stale` or `unreadable`), and the values applied.
    struct State: @unchecked Sendable {
        let key: String, identity: [String: Any]
        var entry: [String: Any]?, state = "none", reason: String?
        var tuned = TuneOverrides()
        func report(knobs: [String: [String: Any]]) -> [String: Any] {
            [
                "version": SelfTuning.version, "key": key, "identity": identity, "entry": entry.map { $0 as Any } ?? NSNull(),
                "state": state,
                "reason": reason.map { $0 as Any } ?? NSNull(), "knobs": knobs,
            ]
        }
    }

    static func entries(_ settings: Settings) throws -> [String: Any]? {
        guard let data = try? Data(contentsOf: file(settings)) else { return nil }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], let entries = object["entries"] as? [String: Any]
        else {
            throw CLIError("tuning.json holds no entries")
        }
        return entries
    }

    /// A result's key in tuning.json: the model's identity (truncated) and the context.
    static func key(model: String, context: Int) -> String { "\(model.prefix(12))@\(context)" }

    /// Never throws: a file that cannot be read serves the defaults, saying so.
    static func load(_ settings: Settings, model: String, context: Int, gpu: String, drives: [URL]) -> State {
        var state = State(key: key(model: model, context: context), identity: identity(model: model, gpu: gpu, drives: drives))
        do {
            guard let entry = try entries(settings)?[state.key] as? [String: Any] else { return state }
            state.entry = entry
            let date = (entry["date"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast
            (state.state, state.reason) = SelfTuning.staleness(
                saved: entry["identity"] as? [String: Any] ?? [:], current: state.identity, date: date)
            if state.state != "stale" { state.tuned = try TuneOverrides(json: entry["values"] ?? [String: Any]()) }
        } catch {
            state.state = "unreadable"
            state.reason = "\(error)"
            state.tuned = TuneOverrides()
        }
        return state
    }

    /// Writes (or with nil removes) one entry, keeping the others.
    static func save(_ settings: Settings, key: String, entry: [String: Any]?) throws {
        try StateFile.locked(file(settings)) {
            // An unreadable tuning.json is set aside, never rewritten from empty (`StateFile`).
            var all: [String: Any]
            do { all = try entries(settings) ?? [:] } catch {
                guard StateFile.setAside(file(settings)) else {
                    throw CLIError("\(file(settings).path) cannot be read and was left as it is; nothing was saved")
                }
                all = [:]
            }
            all[key] = entry
            try JSONSerialization.data(withJSONObject: ["entries": all], options: [.prettyPrinted, .sortedKeys]).write(
                to: file(settings), options: .atomic)
        }
    }

    /// The knob's value and its source for /v1/status, with the plan's default the re-check compares against.
    static func knob(_ value: Any, source: String, default fallback: Any) -> [String: Any] {
        ["value": value, "source": source, "default": fallback]
    }

    /// `origami report`: the saved results and the last run.
    static func reportFields(_ settings: Settings) -> [String: Any] {
        let last = (try? String(contentsOf: historyFile(settings), encoding: .utf8))?.split(separator: "\n").last
            .flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) }
        return ["tuning": ((try? entries(settings)) ?? nil).map { $0 as Any } ?? NSNull(), "tuning_last_run": last ?? NSNull()]
    }

    // MARK: origami tune

    final class Aborted: Error, CustomStringConvertible {
        let description: String
        init(_ text: String) { description = text }
    }

    /// The front's model did not start (`/v1/status` says `failed`), with its reason.
    struct NotStarted: Error, CustomStringConvertible { let description: String }

    struct Client {
        let base: URL, key: String?
        var deadline: Date? = nil
        func call(_ path: String, body: [String: Any]? = nil) async throws -> [String: Any] {
            var request = URLRequest(url: base.appendingPathComponent(path))
            request.timeoutInterval = min(30, max(0.01, deadline?.timeIntervalSinceNow ?? 30))
            if let deadline, Date() >= deadline { throw Aborted("Time limit reached; no settings changed") }
            if let key { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data, response): (Data, URLResponse)
            do { (data, response) = try await URLSession.shared.data(for: request) } catch {
                // Cut short at the limit (its timeout is the time left): the limit is the reason.
                if let deadline, Date() >= deadline { throw Aborted("Time limit reached; no settings changed") }
                throw Aborted("Origami stopped answering (it may be restarting)")
            }
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                if object["status"] as? String == "failed" { throw NotStarted(description: object["failure"] as? String ?? "failed") }
                throw CLIError((object["error"] as? [String: Any])?["message"] as? String ?? "Server unavailable")
            }
            return object
        }
    }

    static let prompts = [
        "Write a story of about 250 words about a paper crane that learns to fly.",
        "Explain in about 200 words how a refrigerator keeps food cold.",
    ]
    /// About 18 tokens a line (75 lines: about 1,400) with a fresh first line, so no saved prompt matches it.
    static func document(lines: Int) -> String {
        "Reference \(UUID().uuidString).\n"
            + (0..<lines).map { "Line \($0): the warehouse inventory lists \($0 * 7 % 101) crates of item \($0)." }.joined(separator: " ")
            + "\nHow many crates does line \(min(41, max(0, lines - 1))) list? Answer with the number only."
    }

    static func pressureLevel() -> Int32 {
        var level: Int32 = 1
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0 ? level : 1
    }

    static func label(_ arm: String) -> String {
        [
            ("gpu_keepalive=false", "GPU keep-alive off"), ("gpu_keepalive=true", "GPU keep-alive on"),
            ("family_prefetch=false", "next-layer guesses off"),
            ("family_prefetch=true", "next-layer guesses on"), ("family_spin=false", "spinning off"), ("family_spin=true", "spinning on"),
            ("defaults", "the defaults"),
        ].first { $0.0 == arm }?.1
            ?? arm.replacingOccurrences(of: "family_prefetch_depth=", with: "guess depth ")
            .replacingOccurrences(of: "qwen_prefill_scope=", with: "prompt group ")
            .replacingOccurrences(of: "family_prefill_slices=", with: "prompt slices ")
    }

    static func run(flags: [String: [String]], recheck: Bool, undo: Bool, memory: Bool = false) async throws {
        let settings = try Settings.load(flags: flags)
        // Strict: a value that is not a number or true/false is a usage error, never a normal tune that applies itself.
        let minutesText = flags["minutes"]?.last
        let givenMinutes = try minutesText.map { text -> Double in
            guard let value = Double(text), value.isFinite else { throw CLIError.usage("--minutes is a number of minutes, such as 8") }
            guard (3...(memory ? 60 : 30)).contains(value) else {
                throw CLIError.usage(memory ? "--minutes is 3 to 60 with --memory" : "--minutes is 3 to 30")
            }
            return value
        }
        func flag(_ name: String) throws -> Bool {
            guard let text = flags[name]?.last else { return false }
            guard let value = Settings.parseBool(text) else { throw CLIError.usage("--\(name) is true or false") }
            return value
        }
        let review = try flag("review")
        let tradeoffs = try flag("tradeoffs")
        try Commands.requireReady(settings)
        var client = Client(base: URL(string: "http://127.0.0.1:\(settings.port)")!, key: settings.apiKey)
        // From the command line, the dashboard's rule and words (a job the server started has passed them already).
        if ProcessInfo.processInfo.environment["ORIGAMI_JOB"] == nil,
            let block = ((try? await client.call("v1/dashboard"))?["dashboard"] as? [String: Any])?["tune_block"] as? String, !block.isEmpty
        {
            throw CLIError(block)
        }
        let status = try await client.call("v1/status")
        guard let tuning = status["tuning"] as? [String: Any], let key = tuning["key"] as? String,
            let identity = tuning["identity"] as? [String: Any],
            let knobs = tuning["knobs"] as? [String: [String: Any]]
        else { throw CLIError("The model on port \(settings.port) cannot be tuned (it runs an older Origami)") }
        if memory {
            let minutes = givenMinutes ?? 30
            return try await measureMemory(settings, client: client, status: status, key: key, identity: identity, minutes: minutes)
        }
        let entry = tuning["entry"] as? [String: Any]
        let applies = ["current", "recheck"].contains(tuning["state"] as? String ?? "")
        if undo {
            guard entry != nil else {
                print("Nothing is tuned for this model and context.")
                outcome("unchanged")
                return
            }
            try save(settings, key: key, entry: nil)
            _ = try await client.call("v1/reload", body: [:])
            print("Undone: the tuned settings are removed; Origami reloads with its defaults.")
            outcome("undone")
            return
        }
        let minutes = givenMinutes ?? 8
        // The default limit grows to what this model's measurements need; an owner's own limit is kept.
        let ownLimit = flags["minutes"] != nil
        let ownership = RequestOwnership(status)
        if tradeoffs {
            return try await explore(
                settings, client: client, key: key, minutes: minutes, offered: status["quality_levels"] as? [String] ?? ["exact"],
                floor: flags["quality-floor"]?.last, ownership: ownership)
        }
        guard (3...30).contains(minutes) else { throw CLIError.usage("--minutes is 3 to 30") }
        // The drives again too, between requests (a busy drive is skipped); serving uses them from its next start.
        Calibrations.measureConfigured(settings, force: true) { print("Drives  " + $0) }
        let free = knobs.filter { $0.value["source"] as? String != "setting" }
        let savedValues = applies ? entry?["values"] as? [String: Any] ?? [:] : [:]
        // The arms: the base (as the server runs now), its repeat, and each candidate as its own switches.
        var arms: [(name: String, values: [String: Any])] = [("base", [:]), ("control", [:])]
        if recheck {
            guard !savedValues.isEmpty else { throw CLIError("Nothing tuned to re-check here; run origami tune") }
            arms.append(("defaults", savedValues.keys.reduce(into: [String: Any]()) { $0[$1] = knobs[$1]?["default"] ?? NSNull() }))
        } else {
            for name in [
                "gpu_keepalive", "family_prefetch", "family_spin",
            ] {
                if let value = free[name]?["value"] as? Bool { arms.append(("\(name)=\(!value)", [name: !value])) }
            }
            for name in ["qwen_prefill_scope", "family_prefill_slices"] {
                guard let knob = free[name], let current = knob["value"] as? Int,
                    let maximum = knob["maximum"] as? Int, let step = knob["step"] as? Int
                else { continue }
                for value in SelfTuning.prefillCandidates(current: current, maximum: maximum, step: step) {
                    arms.append(("\(name)=\(value)", [name: value]))
                }
            }
        }
        guard arms.count > 2 || free["family_prefetch_depth"] != nil else {
            throw CLIError(
                "Nothing to tune: \(knobs.isEmpty ? "this model has no timing switches" : "your settings fix every timing switch")")
        }
        let started = Date()
        let revision = DashboardState.revision(settings)
        var deadline = started.addingTimeInterval(minutes * minute)
        client.deadline = deadline
        var sampleSeconds = 0.0
        /// Past the limit, the round in progress still finishes (requests may run this long), then the run decides.
        var hard: Date { deadline.addingTimeInterval(6 * sampleSeconds + 30) }
        try? FileManager.default.removeItem(at: proposalFile(settings))
        var stalls = 0.0
        var stalled = 0
        var warming = true
        var comparisonAnchor: TuningConditions.Snapshot?
        /// Shared host policy and ownership checks apply to Qwen, GLM and DeepSeek alike.
        func one(_ prompt: String, _ tokens: Int, _ values: [String: Any], writing: Bool = false) async throws -> TimedReply {
            try await measuredRequest(
                client, key: key, prompt: prompt, tokens: tokens,
                extra: ["origami_tune": values], deadline: hard, warmup: warming,
                ownership: ownership, anchor: comparisonAnchor,
                minimumCompletionTokens: writing ? 2 : 1
            ).run
        }
        /// Writing speed over both prompts, including the stalls the Mac added (`SelfTuning.stallSeconds`).
        func decode(_ values: [String: Any], tokens maximum: Int) async throws -> (rate: Double, ttft: Double, replies: [String]) {
            var tokens = 0
            var seconds = 0.0
            var ttft = 0.0
            var replies: [String] = []
            for prompt in prompts {
                let run = try await one(prompt, maximum, values, writing: true)
                let stall = SelfTuning.stallSeconds(gaps: run.gaps)
                if stall > 0 {
                    stalls += stall
                    stalled += 1
                }
                tokens += max(0, run.completion - 1)
                seconds += max(0.001, run.total - run.ttft)
                ttft += run.ttft
                replies.append("\(run.completion):\(run.text)")
            }
            return (Double(tokens) / seconds, ttft / Double(prompts.count), replies)
        }
        var done = 0
        var planned = 1
        func progress(_ stage: String) {
            done += 1
            print("\(stage)  \(min(99, done * 100 / max(1, planned)))%")
        }
        var dTokens = 16
        var lines = 12
        var prefillLines = 12
        var sampledPromptTokens = 0
        func objective(_ name: String) -> SelfTuning.Objective {
            name.hasPrefix("qwen_prefill_scope=") || name.hasPrefix("family_prefill_slices=")
                || name == "defaults" && (savedValues["qwen_prefill_scope"] != nil || savedValues["family_prefill_slices"] != nil)
                ? .prompt : .decode
        }
        func measure(_ values: [String: Any], stage: String, objective: SelfTuning.Objective = .decode) async throws -> SelfTuning.Sample {
            let began = Date()
            var values = values
            // Explicit scope requests bypass all prompt reuse and capture. Apply that policy to their controls too.
            if objective == .prompt, values["qwen_prefill_scope"] == nil, let scope = knobs["qwen_prefill_scope"]?["value"] as? Int {
                values["qwen_prefill_scope"] = scope
            }
            let written = try await decode(values, tokens: dTokens)
            let read = try await one(document(lines: objective == .prompt ? prefillLines : lines), objective == .prompt ? 8 : 1, values)
            sampledPromptTokens = read.prompt - read.cached
            sampleSeconds = max(sampleSeconds * 0.8, Date().timeIntervalSince(began))
            progress(stage)
            return SelfTuning.Sample(
                decode: written.rate, prompt: read.ttft > 0 ? Double(read.prompt - read.cached) / read.ttft : 0,
                replies: written.replies + (objective == .prompt ? ["\(read.completion):\(read.text)"] : []))
        }
        print(
            "Tuning \(Tuning.label(arms.last!.name) == "the defaults" ? "(re-check)" : "for this Mac"): about \(Int(minutes)) minutes\(ownLimit ? "" : ", longer if this model needs it"); candidate replies are checked; a request from an app stops it, keeping what is measured."
        )
        var kept = Kept()
        do {
            // Warm-up (discarded): the first replies after a start are slower while memory and the expert cache fill. It
            // runs until two passes in a row write within 5%, and sizes the replies and the document.
            print("Warming up")
            let read = try await one(document(lines: lines), 1, [:])
            var passes: [Double] = []
            var warm = try await decode([:], tokens: 12)
            passes.append(warm.rate)
            repeat {
                warm = try await decode([:], tokens: 12)
                passes.append(warm.rate)
            } while passes.count < 4 && deadline.timeIntervalSinceNow > minutes * minute * 0.75
                && !SelfTuning.settled(passes[passes.count - 2], passes.last!)
            let warmNote = "warm-up wrote " + passes.map { String(format: "%.2f", $0) }.joined(separator: ", ") + " tok/s"
            if !SelfTuning.settled(passes[passes.count - 2], passes.last!) {
                print("Warm-up varied; paired controls will determine whether a gain is reliable")
            }
            warming = false
            let rate = max(0.5, warm.rate)
            // The document is read in about 8 s (8 to 75 lines): a guard against slower reading, not the score.
            let perLine = Double(max(1, read.prompt)) / Double(lines)
            let readRate = read.ttft > 0 ? Double(read.prompt - read.cached) / read.ttft : 0
            lines = readRate > 0 ? min(75, max(8, Int(readRate * 8 / perLine))) : lines
            // A group cannot be evaluated with the short decode guard's document. Fit a longer prompt inside this
            // context, leaving room for template tokens and more digits in later inventory lines.
            let context = status["context"] as? Int ?? settings.context
            let qwenGroup = knobs["qwen_prefill_scope"]?["maximum"] as? Int ?? 0
            let familyGroup = (knobs["family_prefill_slices"]?["maximum"] as? Int ?? 0) * 2048
            // An extra family slice allows the feed/forward boundary to fall between groups.
            let target = min(max(256, context - 512), max(2048, qwenGroup + 256, familyGroup + 2048))
            prefillLines = max(lines, Int(Double(target) / max(1, perLine * 1.25)))
            // Family models: a guess depth from what each guessed rank was used for (the deepest rank used at least half the time).
            let guesses = free["family_prefetch"]?["value"] as? Bool ?? false
            let allGuesses = free["family_prefetch_depth"]?["value"] as? String == "all"
            if !recheck, guesses, allGuesses {
                let cache = try await client.call("v1/status")["expert_cache"] as? [String: Any] ?? [:]
                let issued = cache["prefetch_issued_by_rank"] as? [Int] ?? []
                let used = cache["prefetch_used_by_rank"] as? [Int] ?? []
                if let depth = SelfTuning.guessDepth(issued: issued, used: used) {
                    arms.append(("family_prefetch_depth=\(depth)", ["family_prefetch_depth": depth]))
                }
            }
            guard arms.count > 2 else { throw CLIError("Nothing to tune: your settings fix every timing switch") }
            let requests = (arms.count - 2) * 9 + 4
            let fixed = 2 * warm.ttft + read.ttft * Double(lines) / 12
            let budget = (ownLimit ? minutes : SelfTuning.maximumMinutes) * minute - (Date().timeIntervalSince(started))
            dTokens = min(64, max(8, Int(rate * 6)))
            if Double(requests) * (fixed + 2 * Double(dTokens) / rate) > budget {
                dTokens = max(8, Int(rate * ((budget / Double(requests) - fixed) / 2)))
            }
            // Measurements an interrupted run kept for this model, context and settings: continued, at their sizes.
            let measuring = knobs.mapValues { "\($0["value"] ?? "")" }
            kept = Kept(settings, key: key, identity: identity, measuring: measuring) ?? Kept()
            /// The samples a candidate still needs: none once decided, else what its three rounds lack.
            func need(_ name: String) -> Int {
                guard let arm = kept.arms[name] else { return 9 }
                return arm.judged || arm.rounds.count >= 3 ? 0 : (3 - arm.rounds.count) * 3
            }
            if kept.count > 0, let tokens = kept.shape["tokens"], let keptLines = kept.shape["lines"],
                let keptPrefill = kept.shape["prefill_lines"]
            {
                (dTokens, lines, prefillLines) = (tokens, keptLines, keptPrefill)
                let total = kept.count + arms.dropFirst(2).reduce(0) { $0 + need($1.name) }
                print(
                    "continuing: \(kept.count) of \(total) measurements kept from \(kept.date.formatted(date: .abbreviated, time: .shortened))"
                )
            } else {
                kept = Kept()
            }
            kept.shape = ["tokens": dTokens, "lines": lines, "prefill_lines": prefillLines]
            planned = requests
            sampleSeconds = fixed + Double(2 * dTokens + 1) / rate
            // Kept rounds count only while this Mac still runs as it did then: a fresh baseline must agree with theirs
            // within the drift test's tolerance (`SelfTuning.driftLimit`), else they are dropped and measured again.
            if kept.count > 0 {
                let then = SelfTuning.median(kept.arms.values.flatMap { $0.rounds.compactMap { $0["base"]?.decode } })
                comparisonAnchor = conditions(try await readyToMeasure(client, key: key, deadline: hard, ownership: ownership))
                let now = try await measure([:], stage: "Baseline check", objective: .decode).decode
                if then > 0, abs(now / then - 1) > SelfTuning.driftLimit {
                    print(
                        String(
                            format:
                                "Kept measurements dropped: the baseline now writes %.0f%% %@ than when they were measured (more than %.0f%%); measuring again.",
                            abs(now / then - 1) * 100, now > then ? "faster" : "slower", SelfTuning.driftLimit * 100))
                    kept.arms = [:]
                }
            }
            // Each candidate has adjacent controls, limiting cache/thermal drift between comparisons.
            // An inconclusive candidate gets up to two more pairs within the same deadline.
            var decision = SelfTuning.Decision(outcome: "already_fastest")
            var sawNoise = false
            var tested = 0
            // The candidates closest to a decision first: kept measurements are finished before new ones start.
            let candidates = arms.dropFirst(2).enumerated().sorted {
                (need($0.element.name), $0.offset) < (need($1.element.name), $1.offset)
            }
            .map(\.element)
            client.deadline = nil
            var keep = candidates.count
            var plannedTime = false
            /// After the first baseline, the plan from its measured time: the default limit grows (to its maximum) to fit
            /// every candidate's three rounds and a confirmation; an owner's limit keeps the candidates that fit.
            func planTime(from position: Int) {
                plannedTime = true
                let more = ownLimit ? 0 : max(0, started.addingTimeInterval(SelfTuning.maximumMinutes * minute).timeIntervalSince(deadline))
                let fit = SelfTuning.fit(
                    needs: candidates[position...].map { need($0.name) }, confirm: 4, sampleSeconds: sampleSeconds,
                    left: deadline.timeIntervalSinceNow, more: more)
                if fit.extra > 0 {
                    deadline = deadline.addingTimeInterval(fit.extra)
                    let total = Int(ceil(deadline.timeIntervalSince(started) / minute))
                    print(
                        "Time limit: this model needs about \(total) minutes for a reliable comparison; the default limit grows to \(total)."
                    )
                }
                keep = position + fit.count
                if keep < candidates.count {
                    let left = candidates[keep...].map { label($0.name) }.joined(separator: ", ")
                    print(
                        "Time limit: \(ownLimit ? "your \(Int(minutes))-minute limit fits" : "even \(Int(SelfTuning.maximumMinutes)) minutes fit only") \(keep) of \(candidates.count) candidates; left out: \(left)."
                    )
                }
            }
            /// A candidate's three rounds or more, judged; the best winner so far is the decision. Whether to stop measuring it.
            func judge(_ rounds: [SelfTuning.Round], _ name: String, _ metric: SelfTuning.Objective) -> Bool {
                let result = SelfTuning.decide(rounds, candidates: [name], objective: metric)
                if result.outcome == "too_noisy" { sawNoise = true }
                if result.winner != nil && result.gain > decision.gain || recheck { decision = result }
                return result.outcome != "too_noisy" || deadline.timeIntervalSinceNow < sampleSeconds * 7
            }
            var timeUp = false
            /// Why a candidate was left after two disrupted rounds: the ending's reason when the limit did not cut.
            var disrupted: String?
            for (position, arm) in candidates.enumerated() {
                let metric = objective(arm.name)
                var (rounds, judged) = kept.arms[arm.name] ?? ([], false)
                if judged || rounds.count >= 3 {
                    // Decided by an earlier run: judged again from its kept rounds, nothing measured.
                    if rounds.count >= 3 { _ = judge(rounds, arm.name, metric) }
                    tested += 1
                    continue
                }
                guard position < keep, !timeUp else {
                    decision.notes.append("Time limit: tested \(tested) of \(candidates.count) candidates")
                    break
                }
                var disruptedRounds = 0
                var attempts = rounds.count
                while attempts < 5 {
                    attempts += 1
                    // A round starts only if it ends within the limit; one in progress at the limit finishes (`hard`).
                    if plannedTime, deadline.timeIntervalSinceNow < sampleSeconds * 3 {
                        timeUp = true
                        break
                    }
                    comparisonAnchor = conditions(try await readyToMeasure(client, key: key, deadline: hard, ownership: ownership))
                    do {
                        let base = try await measure([:], stage: "Baseline · " + label(arm.name), objective: metric)
                        if !plannedTime { planTime(from: position) }
                        if metric == .prompt {
                            let name = arm.values["qwen_prefill_scope"] != nil ? "qwen_prefill_scope" : "family_prefill_slices"
                            let unit = name == "qwen_prefill_scope" ? 1 : 2048
                            let smaller = min(knobs[name]?["value"] as? Int ?? 0, arm.values[name] as? Int ?? 0) * unit
                            if sampledPromptTokens <= smaller {
                                decision.notes.append("\(label(arm.name)): prompt too short to exercise different groups")
                                judged = true
                                break
                            }
                        }
                        let candidate = try await measure(arm.values, stage: label(arm.name), objective: metric)
                        let control = try await measure([:], stage: "Confirming baseline", objective: metric)
                        rounds.append(["base": base, "control": control, arm.name: candidate])
                        // Kept as each round ends: an interruption from here on loses nothing measured.
                        judged = rounds.count == 1 && metric.speed(candidate) < metric.speed(base) * 0.90
                        kept.arms[arm.name] = (rounds, judged)
                        kept.save(settings, key: key, identity: identity, measuring: measuring)
                        if judged || rounds.count >= 3 && judge(rounds, arm.name, metric) { break }
                    } catch let disruption as Disrupted {
                        disruptedRounds += 1
                        sawNoise = true
                        print(
                            "Discarded comparison: \(disruption.description). \(disruptedRounds == 1 ? "Retrying within your time limit." : "Moving on without applying it.")"
                        )
                        if disruptedRounds > 1 {
                            disrupted = disruption.description
                            break
                        }
                    }
                }
                if judged {
                    kept.arms[arm.name] = (rounds, true)
                    kept.save(settings, key: key, identity: identity, measuring: measuring)
                }
                if judged || rounds.count >= 3 { tested += 1 }
            }
            if decision.winner == nil && tested < candidates.count {
                // At the limit: decided from what it has, and nothing is said about the candidates not measured.
                decision.outcome = "incomplete"
                if let disrupted, !timeUp, !decision.notes.contains(where: { $0.hasPrefix("Time limit") }) {
                    decision.notes.insert(
                        "Discarded comparisons: \(disrupted); measured \(tested) of \(candidates.count) candidates", at: 0)
                } else if !decision.notes.contains(where: { $0.hasPrefix("Time limit") }) {
                    decision.notes.insert("Time limit: tested \(tested) of \(candidates.count) candidates", at: 0)
                }
            } else if decision.winner == nil && sawNoise {
                decision.outcome = "too_noisy"
                decision.notes.append("Measurements varied too much to confirm an improvement")
            }
            decision.notes.append(warmNote)
            // Every candidate decided by kept rounds: the default limit is sized for the confirmation alone.
            if decision.winner != nil, !plannedTime { planTime(from: candidates.count) }
            // The single winner once more against the base, judged with the same threshold.
            if let winner = decision.winner, deadline.timeIntervalSinceNow < sampleSeconds * 4 {
                decision.outcome = "incomplete"
                decision.winner = nil
                decision.gain = 0
                decision.notes.insert("Time limit: \(label(winner)) was faster, with no time left to confirm it", at: 0)
            }
            if let winner = decision.winner, let values = arms.first(where: { $0.name == winner })?.values {
                var confirm: [SelfTuning.Round] = []
                for index in 0..<2 {
                    comparisonAnchor = conditions(try await readyToMeasure(client, key: key, deadline: hard, ownership: ownership))
                    var round: SelfTuning.Round = [:]
                    for arm in index == 0 ? [("base", [String: Any]()), (winner, values)] : [(winner, values), ("base", [:])] {
                        round[arm.0] = try await measure(arm.1, stage: "Confirming", objective: decision.objective)
                    }
                    confirm.append(round)
                }
                let again = SelfTuning.decide(confirm, candidates: [winner], threshold: decision.threshold, objective: decision.objective)
                if again.winner == nil {
                    decision.outcome = "too_noisy"
                    decision.winner = nil
                    decision.gain = 0
                    decision.notes.insert("\(label(winner)) was not faster again when confirmed", at: 0)
                }
            }
            if stalled > 0 { decision.notes.append(String(format: "%.1f s of stalls included in %d replies", stalls, stalled)) }
            _ = try await idleState(client, key: key, ownership: ownership)
            guard revision == DashboardState.revision(settings) else { throw Aborted("Settings changed during tuning") }
            // Every candidate decided: nothing left to continue. Otherwise the kept rounds wait for the next run.
            if decision.outcome != "incomplete" && !kept.arms.isEmpty { try? save(settings, key: Kept.key(key), entry: nil) }
            let summary = "Measured \(tested) of \(candidates.count) candidates (\(kept.count) measurements)"
            try await finish(
                settings, client: client, key: key, identity: identity, decision: decision, arms: arms, saved: savedValues,
                recheck: recheck, minutes: Date().timeIntervalSince(started) / 60, review: review, summary: summary,
                next: decision.notes.first?.hasPrefix("Discarded") == true
                    ? "run tuning again when Origami is idle; it continues from them"
                    : ownLimit
                        ? "run tuning again (it continues from them), or choose a longer limit" : "run tuning again; it continues from them"
            )
        } catch  where error is Aborted || error is Disrupted {
            // Whatever stopped it, the rounds finished so far are already kept (`Kept`), and no setting changed.
            let stop = (reason: "\(error)", aborted: error is Aborted)
            let what = "\(stop.aborted ? "Stopped" : "Incomplete"): \(stop.reason)."
            let text =
                kept.count > 0
                ? "\(what) Kept: \(kept.count) measurements; no setting changed. Next: run tuning again when Origami is idle; it continues from them."
                : "\(what) Nothing was measured yet; no setting changed. Next: run tuning again when Origami is idle."
            record(settings, key: key, ["outcome": stop.aborted ? "aborted" : "incomplete", "reason": stop.reason, "kept": kept.count])
            print(text)
            outcome(stop.aborted ? "aborted" : "incomplete", reason: text)
        }
    }

    static func record(_ settings: Settings, key: String, _ fields: [String: Any]) {
        var line = fields
        line["key"] = key
        line["date"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        let url = historyFile(settings)
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data + Data("\n".utf8))
            try? handle.close()
        }
    }

    /// Saves the result (a too-noisy run changes nothing) and reloads the model when its values changed.
    static func finish(
        _ settings: Settings, client: Client, key: String, identity: [String: Any], decision: SelfTuning.Decision,
        arms: [(name: String, values: [String: Any])], saved: [String: Any], recheck: Bool, minutes: Double, review: Bool = false,
        summary: String, next: String
    ) async throws {
        let measured: [String: Any] = [
            "threshold": decision.threshold, "noise": decision.noise, "drift": decision.drift, "ratios": decision.ratios,
            "minutes": minutes, "notes": decision.notes, "objective": decision.objective.rawValue,
        ]
        record(
            settings, key: key,
            ["outcome": decision.outcome, "winner": decision.winner ?? NSNull(), "gain": decision.gain, "measured": measured])
        for note in decision.notes { print("note: \(note)") }
        // Every ending says what was measured, what was kept and what to do next.
        guard decision.outcome != "too_noisy" && decision.outcome != "incomplete" else {
            let reason = decision.notes.first ?? "The speeds varied too much to confirm an improvement"
            let text =
                decision.outcome == "incomplete"
                ? "Incomplete: \(reason). \(summary). Kept: those measurements; no setting changed. Next: \(next)."
                : "Inconclusive: \(reason). \(summary). Kept: nothing to continue (every candidate was measured); no setting changed. Next: nothing needed (the current settings stay); run tuning again when the Mac is quieter to try once more."
            print(text)
            outcome(decision.outcome, reason: text)
            return
        }
        var values = saved
        var gain = decision.gain
        var label = saved.isEmpty ? "" : (entryLabel(saved))
        if recheck {
            // The defaults won: the tuned values go. Otherwise they stay, with their measured lead over the defaults.
            if decision.winner == "defaults" {
                values = [:]
                label = ""
            } else {
                let ratios = decision.ratios["defaults"] ?? []
                gain = ratios.isEmpty ? 0 : max(0, 1 / max(0.01, SelfTuning.median(ratios)) - 1)
            }
        } else if let winner = decision.winner, let won = arms.first(where: { $0.name == winner }) {
            values.merge(won.values) { $1 }
            label = entryLabel(values)
        }
        let outcome = values.isEmpty ? "already_fastest" : decision.winner != nil || recheck ? "faster" : "already_fastest"
        let entry: [String: Any] = [
            "values": values, "outcome": outcome, "gain": outcome == "faster" ? gain : 0, "label": label,
            "date": ISO8601DateFormatter().string(from: Date()), "identity": identity, "measured": measured,
        ]
        let fastest =
            (values.isEmpty ? "Tuned: the defaults are fastest on this Mac. " : "Tuned: \(label) stays fastest on this Mac. ") + summary
        // Qwen's prompt group is part of its saved prompts' identity: a new one starts their cache afresh.
        let promptEntries = ((try? await client.call("v1/status"))?["prompt_cache"] as? [String: Any])?["entries"] as? [[String: Any]] ?? []
        let setsAside =
            values["qwen_prefill_scope"].map { "\($0)" } != saved["qwen_prefill_scope"].map { "\($0)" } && !promptEntries.isEmpty
            ? "Applying it sets aside this model's \(promptEntries.count) saved prompt\(promptEntries.count == 1 ? "" : "s") ("
                + OrigamiPlan.gigabytes(UInt64(promptEntries.reduce(0) { $0 + ($1["bytes"] as? Int ?? 0) }))
                + "): the new prompt group starts their cache afresh, so the next prompts are read in full." : nil
        let text: String
        if review {
            if decision.winner != nil || recheck {
                try Settings.write(
                    [
                        "kind": "performance", "key": key, "revision": DashboardState.revision(settings), "entry": entry,
                        "sets_aside": setsAside ?? NSNull(),
                    ], to: proposalFile(settings))
                text =
                    String(format: "Improvement found: +%.0f%% with %@. ", gain * 100, label)
                    + "\(summary). Kept: the result, not applied yet. " + (setsAside.map { $0 + " " } ?? "")
                    + "Next: review and apply it in the dashboard."
            } else {
                // Nothing beat what runs: a result to keep (the page shows it), with no setting changed.
                try save(settings, key: key, entry: entry)
                text = fastest + ". Kept: this result; no setting changed. Next: nothing."
            }
        } else {
            try save(settings, key: key, entry: entry)
            let changed = !NSDictionary(dictionary: values).isEqual(to: saved)
            if changed { _ = try? await client.call("v1/reload", body: [:]) }
            text =
                outcome == "faster"
                ? String(format: "Tuned: \(decision.objective == .prompt ? "reading" : "writing") +%.0f%% with %@. ", gain * 100, label)
                    + "\(summary). Kept: this setting for this model and context\(changed ? "; Origami reloads to use it" : "")."
                    + (setsAside.map { " " + $0.replacingOccurrences(of: "Applying it sets", with: "It sets") } ?? "") + " Next: nothing."
                : fastest + ". Kept: this result\(changed ? "; Origami reloads with its defaults" : "; no setting changed"). Next: nothing."
        }
        print(text)
        self.outcome(outcome, reason: text)
    }

    /// The family knobs for /v1/status, read live (a request's own switches are put back before it ends).
    final class FamilyKnobs: @unchecked Sendable {
        let runner: FamilyRunner, fixed: Set<String>, tuned: TuneOverrides, keepAlive: () -> [String: Any]
        init(runner: FamilyRunner, fixed: Set<String>, tuned: TuneOverrides, keepAlive: @escaping () -> [String: Any]) {
            self.runner = runner
            self.fixed = fixed
            self.tuned = tuned
            self.keepAlive = keepAlive
        }
        var report: [String: [String: Any]] {
            func source(_ name: String, _ tuned: Any?) -> String { fixed.contains(name) ? "setting" : tuned != nil ? "tuned" : "default" }
            let depth: Any = runner.experts.guessDepth.map { $0 as Any } ?? "all"
            return [
                "gpu_keepalive": keepAlive(),
                "family_prefetch": knob(
                    runner.guessesNext, source: source("PREFETCH", tuned.familyPrefetch), default: type(of: runner).guessesByDefault),
                "family_prefetch_depth": knob(depth, source: source("PREFETCH_DEPTH", tuned.familyPrefetchDepth), default: "all"),
                "family_spin": knob(FamilyExperts.spins, source: source("SPIN", tuned.familySpin), default: true),
                "family_early_delayed_reads": knob(
                    runner.earlyDelayedReads, source: tuned.familyEarlyDelayedReads != nil ? "tuned" : "default", default: false),
                "family_prefill_slices": knob(
                    runner.prefillSlices, source: source("PREFILL_SLICES", tuned.familyPrefillSlices), default: runner.promptSlices
                )
                .merging(["maximum": runner.promptSlices, "step": 1]) { $1 },
            ]
        }
    }

    static func entryLabel(_ values: [String: Any]) -> String {
        // A JSON boolean reads back as a number: shown as true or false, never 1 or 0.
        func text(_ value: Any?) -> String {
            if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
            return value.map { "\($0)" } ?? ""
        }
        return values.keys.sorted().map { label("\($0)=\(text(values[$0]))") }.joined(separator: ", ")
    }
}
