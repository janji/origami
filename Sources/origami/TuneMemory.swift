import Darwin
import Foundation
import OrigamiEngine

/// `origami tune --memory` (the dashboard's tuning mode "Fastest memory limit"): reloads the running model at a few
/// memory limits (`SelfTuning.memoryCandidates`), each through the front as a trial that leaves the saved settings and the
/// memory history alone, and times a long prompt's first reply and a repeat. A limit the Mac cut or swapped for is out;
/// the highest kept is suggested (when even the lowest is out, up to three lower limits are tried first, from the level the
/// cuts left; `SelfTuning.chosenMemory`; `memory@<key>` in tuning.json, shown by `origami status`,
/// `/v1/status` and the dashboard). The saved limit is reloaded at the end, also after a failure, Ctrl-C or the job's Cancel.
extension Tuning {
    /// Set by the front on a model it loads at a trial limit: its memory history is a throwaway file (`memoryHistory`).
    static let trialVariable = "ORIGAMI_MEMORY_CALIBRATION"

    /// Whether a trial limit may be loaded now; read by the signal handler.
    final class TrialState: @unchecked Sendable {
        private let lock = NSLock()
        private var active = false, done = 0
        var loaded: Bool {
            get { lock.withLock { active } }
            set { lock.withLock { active = newValue } }
        }
        /// Trials measured (and kept) so far.
        var measured: Int {
            get { lock.withLock { done } }
            set { lock.withLock { done = newValue } }
        }
    }

    /// The trials an interrupted `tune --memory` kept (`memory-partial@<key>`, each saved as it ends) for the same build
    /// and Mac, for a day (`AutoMeasure.backOff`), as `origami tune` keeps its rounds (`Kept`).
    static func keptMemoryTrials(_ settings: Settings, key: String, identity: [String: Any]) -> (
        trials: [SelfTuning.MemoryTrial], date: Date
    )? {
        guard let entry = ((try? entries(settings)) ?? nil)?["memory-partial@" + key] as? [String: Any],
            entry["build"] as? String == version,
            let date = (entry["date"] as? String).flatMap({ ISO8601DateFormatter().date(from: $0) }),
            Date().timeIntervalSince(date) < AutoMeasure.backOff,
            SelfTuning.staleness(saved: entry["identity"] as? [String: Any] ?? [:], current: identity, date: date).state == "current",
            let data = try? JSONSerialization.data(withJSONObject: entry["trials"] ?? []),
            let trials = try? JSONDecoder().decode([SelfTuning.MemoryTrial].self, from: data), !trials.isEmpty
        else { return nil }
        return (trials, date)
    }

    static func measureMemory(
        _ settings: Settings, client base: Client, status: [String: Any], key: String, identity: [String: Any], minutes: Double
    ) async throws {
        var client = base
        let started = Date()
        let deadline = started.addingTimeInterval(minutes * minute)
        client.deadline = deadline
        _ = try await idleState(client, key: key, ownership: RequestOwnership(status))
        let learned = status["memory_learned"] as? [String: Any] ?? [:]
        let held =
            ((learned["limit_advice"] as? [String: Any])?["suggested_bytes"] as? NSNumber)?.uint64Value
            ?? (learned["applied"] as? Bool == true ? (learned["bytes"] as? NSNumber)?.uint64Value : nil)
        let host = try OrigamiHost.current()
        let automatic = OrigamiPlan.automaticMemory(host: host)
        let ceiling = max(automatic, OrigamiPlan.memoryCeiling(host: host))
        // The least limit the model starts with at this context: no limit under it is tried.
        let least = (((learned["minimum_bytes"] as? NSNumber)?.uint64Value ?? 0) + 99_999_999) / 100_000_000 * 100_000_000
        var candidates = Array(
            Set(
                SelfTuning.memoryCandidates(automatic: automatic, ceiling: ceiling, current: settings.memoryLimit, held: held).map {
                    max($0, least)
                })
        ).sorted()
        let swapLimit = TuningConditions.thresholds(physicalMemoryBytes: host.physicalMemory).softSwapBytes
        print(
            "Measuring memory limits \(candidates.map(SelfTuning.memorySize).joined(separator: ", ")): each reloads the model; a request from an app stops it, and your saved limit is reloaded at the end."
        )

        // Ctrl-C or the job's Cancel: the saved limit comes back before this ends.
        let trial = TrialState()
        let restoreRequest: @Sendable () -> Void = {
            var request = URLRequest(url: base.base.appendingPathComponent("v1/reload"))
            request.httpMethod = "POST"
            request.timeoutInterval = 10
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let key = base.key { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
            request.httpBody = Data(#"{"restore_memory": true}"#.utf8)
            let done = DispatchSemaphore(value: 0)
            URLSession.shared.dataTask(with: request) { _, _, _ in done.signal() }.resume()
            _ = done.wait(timeout: .now() + 10)
        }
        let signals = [SIGINT, SIGTERM].map { number in
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler {
                if trial.loaded { restoreRequest() }
                let text = "Stopped: cancelled; your saved memory limit is reloaded. " + keptText(trial.measured)
                print(text)
                outcome("aborted", reason: text)
                exit(1)
            }
            source.resume()
            return source
        }
        defer { signals.forEach { $0.cancel() } }

        /// Reloads at `limit` (nil: the saved settings) and waits for the model.
        func load(_ limit: UInt64?) async throws -> [String: Any] {
            if limit != nil { trial.loaded = true }
            // The restore: a no-op once an app's request has restored the limit itself (the front does, before admitting it).
            _ = try await client.call("v1/reload", body: limit.map { ["memory_limit_trial": $0] } ?? ["restore_memory": true])
            if limit == nil { trial.loaded = false }
            let until = Date().addingTimeInterval(ModelFront.startWait + 60)
            while Date() < until {
                try await Task.sleep(for: .seconds(2))
                let state = try await client.call("v1/status")
                if state["status"] as? String == "ready", state["pending"] == nil { return state }
            }
            throw Aborted("The model did not load at \(limit.map(SelfTuning.memorySize) ?? "your saved limit")")
        }
        /// One reply, refused if an app's request came or went: writing tokens a second and the first token's seconds.
        func one(_ prompt: String, _ tokens: Int, _ ownership: RequestOwnership) async throws -> (rate: Double, ttft: Double) {
            _ = try await idleState(client, key: key, ownership: ownership)
            let run = try await Diagnostics.timed(
                client.base, prompt: prompt, maximumTokens: tokens, apiKey: client.key,
                // Marks the search's own requests: the front ends a trial for any other.
                extra: ["origami_tune": [String: Any]()], deadline: deadline)
            _ = try await idleState(client, key: key, ownership: ownership, finishing: true)
            return (Double(max(0, run.completion - 1)) / max(0.001, run.total - run.ttft), run.ttft)
        }
        /// The lowest budget a cut set in the last trial (it may have grown back by its end); nil without a cut.
        var settled: UInt64?
        /// The lowest budget a cut set in any trial: about what macOS left the model.
        var lowestCut: UInt64?
        // What an interrupted run kept: continued once a fresh trial of its first kept limit agrees with it (`continuing`).
        let partialKey = "memory-partial@" + key
        let earlier = keptMemoryTrials(settings, key: key, identity: identity)
        var kept: [SelfTuning.MemoryTrial] = []
        /// Every trial measured or taken from `kept` in this run, saved as each ends so any interruption keeps them.
        var done: [SelfTuning.MemoryTrial] = []
        func keep(_ result: SelfTuning.MemoryTrial) {
            guard !done.contains(result) else { return }
            done.append(result)
            trial.measured = done.count
            let encoded = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(done))) ?? []
            try? save(
                settings, key: partialKey,
                entry: ["date": ISO8601DateFormatter().string(from: Date()), "build": version, "identity": identity, "trials": encoded])
        }
        func measure(_ limit: UInt64) async throws -> SelfTuning.MemoryTrial {
            if let index = kept.firstIndex(where: { $0.limit == limit }) {
                let result = kept.remove(at: index)
                settled = nil  // where a kept cut settled is not known: the next limit down is the plain step
                print(row(result) + "  (kept)")
                keep(result)
                return result
            }
            let result = try await measureNow(limit)
            keep(result)
            return result
        }
        func measureNow(_ limit: UInt64) async throws -> SelfTuning.MemoryTrial {
            settled = nil
            print("Loading at \(SelfTuning.memorySize(limit))")
            var result = SelfTuning.MemoryTrial(limit: limit)
            let ownership: RequestOwnership
            do { ownership = RequestOwnership(try await load(limit)) } catch let error as NotStarted {
                // Too small for this model (its fixed memory and two layers' experts): out, and the next limit is loaded.
                print("  \(error)")
                result.excluded = SelfTuning.notStarted
                print(row(result))
                return result
            }
            let swap = swapoutBytes()
            var critical = false
            // A long fresh prompt fills the expert cache; then the first reply after it, and a repeat.
            _ = try await one(document(lines: 75), 1, ownership)
            critical = critical || pressureLevel() >= 4
            (result.first, result.firstToken) = try await one(prompts[0], 128, ownership)
            critical = critical || pressureLevel() >= 4
            result.warm = try await one(prompts[1], 128, ownership).rate
            let after = try await client.call("v1/status")
            let memory = after["memory"] as? [String: Any] ?? [:]
            // Only cuts the Mac asked for count; the process holding to the limit itself is the limit working.
            result.cuts = after["memory_pressure_cuts"] as? Int ?? 0
            if result.cuts > 0 { settled = (after["memory_pressure_cut_to"] as? NSNumber)?.uint64Value }
            if let settled { lowestCut = min(lowestCut ?? settled, settled) }
            result.cache =
                ((after["expert_cache"] as? [String: Any])?["bytes"] as? NSNumber)?.uint64Value
                ?? (memory["reserved_bytes"] as? NSNumber)?.uint64Value ?? 0
            let now = swapoutBytes()
            result.swapped = now > swap ? now - swap : 0
            result.excluded =
                result.cuts > 0
                ? "cut \(result.cuts) time\(result.cuts == 1 ? "" : "s")"
                : result.swapped > swapLimit
                    ? "\(result.swapped / 1_000_000) MB swapped" : critical ? "critical memory pressure" : nil
            print(row(result))
            return result
        }

        var trials: [SelfTuning.MemoryTrial] = []
        var discarded: SelfTuning.MemoryTrial?
        /// Whether the search stopped with the lowest limit out and the next one at the model's minimum.
        var atMinimum = false
        do {
            // Kept limits count only while this Mac runs as it did then: a fresh trial of the first one that passed must
            // pass again and agree within the drift test's tolerance (`SelfTuning.driftLimit`), else all are measured again.
            if let earlier, let anchor = earlier.trials.first(where: { $0.excluded == nil }) {
                print(
                    "continuing: \(earlier.trials.count) limit\(earlier.trials.count == 1 ? "" : "s") kept from \(earlier.date.formatted(date: .abbreviated, time: .shortened)); checking \(SelfTuning.memorySize(anchor.limit)) again first"
                )
                let fresh = try await measure(anchor.limit)
                if fresh.excluded == nil, anchor.speed > 0, abs(fresh.speed / anchor.speed - 1) <= SelfTuning.driftLimit {
                    kept = earlier.trials.filter { $0.limit != anchor.limit } + [fresh]
                } else {
                    print(
                        "Kept measurements dropped: \(SelfTuning.memorySize(anchor.limit)) "
                            + (fresh.excluded.map { "is now out (\($0))" }
                                ?? String(
                                    format: "now runs %.0f%% %@ than then", abs(fresh.speed / anchor.speed - 1) * 100,
                                    fresh.speed > anchor.speed ? "faster" : "slower"))
                            + "; measuring again.")
                    kept = [fresh]
                }
            } else if earlier != nil {
                print("Kept measurements dropped: none of their limits passed, so none can be checked; measuring again.")
            }
            for limit in candidates {
                // Above a limit the Mac could not keep, other apps would only be squeezed harder.
                if SelfTuning.memorySkipsHigher(trials) {
                    trials.append(.init(limit: limit, excluded: "not tried: a lower limit was cut or swapped"))
                    continue
                }
                trials.append(try await measure(limit))
            }
            // The lowest limit cut or swapped: other apps hold memory, so go on downwards from the level the governor settled
            // at (else 1.5 GB lower), at most three more, never below the least the model runs with.
            var lower = 0
            if let lowest = trials.first, SelfTuning.memorySkipsHigher([lowest]) {
                if lowest.cuts > 0 { print(otherMemory()) }
                var from = lowest
                // The next limit below `from`, never under the least the model starts with (`minimum_bytes`).
                func next() -> UInt64 {
                    let step = settled.flatMap { $0 + 100_000_000 <= from.limit ? $0 : nil } ?? from.limit - min(from.limit, 1_500_000_000)
                    return max(least, step / 100_000_000 * 100_000_000)
                }
                while lower < 3, SelfTuning.memorySkipsHigher([from]) {
                    let limit = next()
                    guard limit > 0, limit < from.limit else { break }
                    print("\(SelfTuning.memorySize(from.limit)) was out; trying \(SelfTuning.memorySize(limit))")
                    from = try await measure(limit)
                    trials.append(from)
                    lower += 1
                }
                atMinimum = least > 0 && SelfTuning.memorySkipsHigher([from]) && next() >= from.limit
            }
            let measured = trials.filter { $0.excluded?.hasPrefix("not tried") != true && $0.excluded != SelfTuning.notStarted }.count
            let perTrial = Date().timeIntervalSince(started) / Double(max(1, measured))
            if lower == 0, measured < 4, deadline.timeIntervalSinceNow > perTrial * 1.5 + 60,
                let limit = SelfTuning.memoryRefinement(trials),
                limit >= least, !candidates.contains(limit)
            {
                candidates.append(limit)
                trials.append(try await measure(limit))
            }
            // Speed only guards: a kept limit about to be passed over for a lower one is measured once more, and the better
            // of its two results counts (a cut or swap the second time is the Mac's answer, and counts too).
            if let limit = SelfTuning.memoryPassedOver(trials), let index = trials.firstIndex(where: { $0.limit == limit }) {
                print("Measuring \(SelfTuning.memorySize(limit)) again: it was over 15% slower than a lower limit")
                let again = try await measure(limit)
                if again.excluded != nil || again.speed > trials[index].speed {
                    discarded = trials[index]
                    trials[index] = again
                } else {
                    discarded = again
                }
            }
        } catch {
            print("Reloading your saved memory limit")
            client.deadline = nil
            _ = try? await load(nil)
            let reason = ((error as? Aborted)?.description ?? "\(error)").trimmingCharacters(in: CharacterSet(charactersIn: " ."))
            let text = "Stopped: \(reason). " + keptText(done.count, limits: done.map(\.limit))
            print(text)
            outcome(reason.lowercased().contains("time limit") ? "incomplete" : "aborted", reason: text)
            return
        }
        try? save(settings, key: partialKey, entry: nil)  // finished: the result below replaces what was kept
        trials.sort { $0.limit < $1.limit }
        let encoded = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(trials))) ?? []
        let saved: Void? = try? save(
            settings, key: "memory@" + key,
            entry: [
                "date": ISO8601DateFormatter().string(from: Date()), "identity": identity, "automatic_bytes": automatic,
                "model_setting": settings.model ?? "", "context": settings.context, "trials": encoded,
                "minutes": Date().timeIntervalSince(started) / 60,
            ])
        // Saved first, so the model reloaded with your settings shows the result.
        print("Reloading your saved memory limit")
        client.deadline = nil
        _ = try await load(nil)
        if saved == nil { print("note: the result could not be saved to tuning.json") }
        print("Limit    First reply  Repeat      First token  Cache    Cuts  Swap")
        trials.forEach { print(row($0)) }
        if let discarded { print(row(discarded) + "  (the other of its two tries)") }
        var text = SelfTuning.memoryText(trials, current: settings.memoryLimit, automatic: automatic)
        if atMinimum, SelfTuning.chosenMemory(trials) == nil {
            text +=
                ". The model needs at least \(SelfTuning.memorySize(least)) to start with this context"
                + (lowestCut.map { " and macOS left it about \(SelfTuning.memorySize($0))" } ?? "")
                + ": use a smaller context (`origami config set context \(settings.context / 2)`) or close other apps"
        }
        let chosen = SelfTuning.chosenMemory(trials) != nil
        text +=
            (text.hasSuffix(".") ? "" : ".") + " Measured: \(done.count) trial\(done.count == 1 ? "" : "s") of "
            + Set(trials.filter { $0.excluded?.hasPrefix("not tried") != true }.map(\.limit)).sorted().map(SelfTuning.memorySize)
            .joined(separator: ", ") + ". Kept: this result for this model and context (`origami status` and the dashboard show it)"
            + (saved == nil ? ", except that tuning.json could not be written" : "") + "; no setting changed. Next: "
            + (chosen
                ? (settings.memoryLimit == nil
                    ? "nothing (automatic memory starts from it)" : "set it as your limit with the command above if you want it")
                : "close other apps or use a smaller context, then run it again") + "."
        print(text + String(format: " (%.0f minutes)", Date().timeIntervalSince(started) / 60))
        outcome("comparison", reason: text)
    }

    /// What a stopped memory search measured and kept, and what to do next.
    static func keptText(_ count: Int, limits: [UInt64] = []) -> String {
        guard count > 0 else {
            return "Measured: nothing yet. Kept: nothing; no setting changed. Next: run `origami tune --memory` again when the Mac is idle."
        }
        return "Measured: \(count) trial\(count == 1 ? "" : "s")"
            + (limits.isEmpty ? "" : " (" + limits.map(SelfTuning.memorySize).joined(separator: ", ") + ")")
            + ". Kept: \(count == 1 ? "it" : "them") for a day; no setting changed. Next: run `origami tune --memory` again to continue from \(count == 1 ? "it" : "them")."
    }

    /// "Other apps hold memory: …; swap 8.1 of 9.2 GB used.": the three largest other processes by physical footprint
    /// (not Origami's own), and swap.
    static func otherMemory() -> String {
        var pids = [pid_t](repeating: 0, count: Int(max(0, proc_listallpids(nil, 0))) + 64)
        let count = Int(proc_listallpids(&pids, Int32(pids.count * MemoryLayout<pid_t>.size)))
        let own = ProcessInfo.processInfo.processName
        let largest = pids.prefix(max(0, count)).compactMap { pid -> (String, UInt64)? in
            var path = [UInt8](repeating: 0, count: 4096)  // PROC_PIDPATHINFO_MAXSIZE
            guard pid > 0, proc_pidpath(pid, &path, UInt32(path.count)) > 0, let bytes = ModelFront.footprint(pid) else { return nil }
            let text = (String(decoding: path.prefix { $0 != 0 }, as: UTF8.self) as NSString).lastPathComponent
            return text == own ? nil : (text, bytes)
        }
        .sorted { $0.1 > $1.1 }.prefix(3)
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        let swapText =
            sysctlbyname("vm.swapusage", &swap, &size, nil, 0) == 0
            ? String(format: "; swap %.1f of %.1f GB used", Double(swap.xsu_used) / 1e9, Double(swap.xsu_total) / 1e9) : ""
        return "Other apps hold memory: "
            + largest.map { String(format: "%@ %.1f GB", $0.0, Double($0.1) / 1e9) }.joined(separator: ", ") + swapText + "."
    }

    static func row(_ trial: SelfTuning.MemoryTrial) -> String {
        guard trial.excluded?.hasPrefix("not tried") != true, trial.excluded != SelfTuning.notStarted else {
            return SelfTuning.memorySize(trial.limit).padding(toLength: 9, withPad: " ", startingAt: 0) + (trial.excluded ?? "")
        }
        return String(
            format: "%@ %5.1f tok/s  %5.1f tok/s  %6.1f s     %@  %4d  %d MB%@",
            SelfTuning.memorySize(trial.limit).padding(toLength: 8, withPad: " ", startingAt: 0), trial.first, trial.warm,
            trial.firstToken, SelfTuning.memorySize(trial.cache).padding(toLength: 7, withPad: " ", startingAt: 0), trial.cuts,
            trial.swapped / 1_000_000, trial.excluded.map { "  (out: \($0))" } ?? "")
    }

    /// The measured suggestion for `origami status`, `/v1/status` and the dashboard (`limit_advice`, before the unmeasured
    /// one): this model's `memory@<key>` entry (without a key, the newest for the configured model and context), its text
    /// for the limit set now, and whether it is stale (`SelfTuning.memoryStaleness`, counting the days `history` cut it since) or
    /// `applied`: with no memory_limit, a current result is the limit automatic mode serves with.
    static func memoryAdvice(_ settings: Settings, key: String? = nil, history: MemoryHistory? = nil) -> [String: Any]? {
        let all = ((try? entries(settings)) ?? nil) ?? [:]
        let entry =
            key.map { all["memory@" + $0] as? [String: Any] }
            ?? all.filter { $0.key.hasPrefix("memory@") }.compactMap { $0.value as? [String: Any] }
            .filter { $0["model_setting"] as? String == settings.model ?? "" && $0["context"] as? Int == settings.context }
            .max { ($0["date"] as? String ?? "") < ($1["date"] as? String ?? "") }
        guard let entry, let data = try? JSONSerialization.data(withJSONObject: entry["trials"] ?? []),
            let trials = try? JSONDecoder().decode([SelfTuning.MemoryTrial].self, from: data)
        else { return nil }
        let automatic = (entry["automatic_bytes"] as? NSNumber)?.uint64Value ?? 0
        let best = SelfTuning.chosenMemory(trials)?.limit
        let date = (entry["date"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast
        let stale = SelfTuning.memoryStaleness(
            saved: entry["identity"] as? [String: Any] ?? [:],
            current: ["mac": macModel, "memory_bytes": ProcessInfo.processInfo.physicalMemory], savedContext: entry["context"] as? Int,
            context: settings.context, cutDays: best.flatMap { history?.cutDays(below: $0, since: date) } ?? 0, date: date)
        let applied = settings.memoryLimit == nil && stale == nil && best != nil
        return [
            "measured": entry["date"] ?? NSNull(), "limit_bytes": settings.memoryLimit.map { $0 as Any } ?? NSNull(),
            "suggested_bytes": best as Any? ?? NSNull(), "stale": stale as Any? ?? NSNull(), "applied": applied,
            "text": SelfTuning.memoryText(trials, current: settings.memoryLimit, automatic: automatic, applied: applied),
        ]
    }

    /// The limit a serving model starts with: the user's memory_limit, else the measured one `memoryAdvice` applies.
    static func memoryLimit(_ settings: Settings, advice: [String: Any]?) -> UInt64? {
        settings.memoryLimit ?? (advice?["applied"] as? Bool == true ? (advice?["suggested_bytes"] as? NSNumber)?.uint64Value : nil)
    }

    /// The one line saying which memory rule is in force (startup log, `origami status`, the dashboard).
    static func memoryLine(_ settings: Settings, advice: [String: Any]?) -> String {
        if let limit = settings.memoryLimit { return "memory: your limit \(OrigamiPlan.gigabytes(limit)) (ceiling)" }
        if let limit = memoryLimit(settings, advice: advice) {
            return "memory: automatic, using the measured \(OrigamiPlan.gigabytes(limit))"
        }
        if let stale = advice?["stale"] as? String {
            return
                "memory: automatic; this Mac's memory use has changed since the memory limit was measured (\(stale)): run `origami tune --memory`"
        }
        return "memory: automatic (not measured: run `origami tune --memory`)"
    }

    /// `memory_auto_measure`'s last try (`AutoMeasure`): one tuning.json entry per model and context (`settings.model`, the
    /// model a server runs), its date and line, and why it waits (`waiting`, since `waiting_since`) while it does.
    static func autoMeasureKey(_ settings: Settings) -> String { "memory-auto@\(settings.model ?? "")@\(settings.context)" }

    /// The last automatic try for this model and context, or nil.
    static func autoMeasureRecord(_ settings: Settings) -> [String: Any]? {
        ((try? entries(settings)) ?? nil)?[autoMeasureKey(settings)] as? [String: Any]
    }

    /// Its line for `origami status`, `/v1/status` and the dashboard, or nil: why it waits, else how the last try ended.
    static func autoMeasureLine(_ settings: Settings) -> String? {
        let record = autoMeasureRecord(settings)
        // During the back-off the last try's own line says how it ended and that it tries again.
        if let reason = record?["waiting"] as? String, !(reason.hasPrefix("tried at") && record?["line"] is String),
            let since = (record?["waiting_since"] as? String).flatMap({ ISO8601DateFormatter().date(from: $0) })
        {
            return AutoMeasure.waitingLine(reason, since: since)
        }
        return record?["line"] as? String
    }

    static func storeAutoMeasure(_ settings: Settings, _ record: [String: Any]) {
        try? save(
            settings, key: autoMeasureKey(settings),
            entry: record.merging(["model_setting": settings.model ?? "", "context": settings.context]) { $1 })
    }

    /// Keeps why it waits (nil: it does not), written only when that changes; the time is when this reason began.
    static func noteAutoMeasureWaiting(_ settings: Settings, _ reason: String?) {
        let record = autoMeasureRecord(settings) ?? [:]
        guard record["waiting"] as? String != reason else { return }
        var next = record.filter { $0.key != "waiting" && $0.key != "waiting_since" }
        if let reason { (next["waiting"], next["waiting_since"]) = (reason, ISO8601DateFormatter().string(from: Date())) }
        storeAutoMeasure(settings, next)
    }

    /// Every stored try still saying it measures, settled (`AutoMeasure.settled`): the server that ran it has stopped.
    static func settleAutoMeasures(_ settings: Settings) {
        for (key, value) in ((try? entries(settings)) ?? nil) ?? [:] where key.hasPrefix("memory-auto@") {
            if let record = AutoMeasure.settled(value as? [String: Any]) { try? save(settings, key: key, entry: record) }
        }
    }
}
