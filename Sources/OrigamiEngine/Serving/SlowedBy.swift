import Foundation

/// What slows Origami down now, each with what a person can do about it: `slowed_by` in `/v1/status` and `/health`,
/// the Now tab's notice and the "Slowed by" lines of `origami status`. One rule per condition, in one table, read from
/// the status the server already builds (no polling of its own); the order of the table is the order of the list, the
/// costliest first, and settings that could be better come last, as advice. Checked against
/// `Tests/Fixtures/status/slowed-by.json`, whose lists the dashboard's checks show as the page does.
///
/// The front adds to the model's status, before the table reads it: `mac_now` (`thermal_state`, `low_power`) and
/// `memory_holders` (the largest other apps, `ProcessMemory.others`, while `memoryShort`), both kept in the status; and
/// for the table alone `job` (a job running) and `last_tuning` (the last tuning run, while none is applied). Every field
/// a rule reads is listed with it (`reads`) and checked against a status as the servers build it.
public enum SlowedBy {
    typealias Found = (text: String, fix: String, since: String?)

    /// The list: `[{id, text, fix, since}]`, `since` (ISO 8601) when the status knows when it began, else null.
    public static func list(_ s: [String: Any]) -> [[String: Any]] {
        rules.compactMap { rule in
            rule.test(s).map { ["id": rule.id, "text": $0.text, "fix": $0.fix, "since": $0.since ?? NSNull()] as [String: Any] }
        }
    }

    /// Fills each item's missing `since` with when it was first seen (`seen`, by id, kept by the caller); conditions no
    /// longer listed are forgotten.
    public static func stamp(_ items: [[String: Any]], seen: inout [String: String], now: Date = Date()) -> [[String: Any]] {
        let ids = Set(items.compactMap { $0["id"] as? String })
        seen = seen.filter { ids.contains($0.key) }
        return items.map { item in
            var item = item
            let id = item["id"] as? String ?? ""
            if let since = item["since"] as? String { seen[id] = since }
            item["since"] = seen[id] ?? ISO8601DateFormatter().string(from: now)
            seen[id] = item["since"] as? String
            return item
        }
    }

    /// Every condition's id, in the list's order.
    public static var ids: [String] { rules.map(\.id) }
    /// Every status field a rule reads (`a.b`, `a[].b` for any element of an array), for the check that each is in the
    /// status as the servers build it.
    public static var reads: [String] { rules.flatMap(\.reads) }

    private static let source = "experts_from.sources[]."
    private static let advice = "memory_learned.limit_advice."
    private static let waits = ["requests.recent[].stage_seconds.generation", "requests.recent[].phase_counters"]
    static let rules: [(id: String, reads: [String], test: @Sendable ([String: Any]) -> Found?)] = [
        ("drive_gone", ["status"], driveGone),
        ("job", ["job.state", "job.title", "job.kind", "job.started"], job),
        (
            "memory_low",
            [
                "memory.budget_bytes", "memory.cap_bytes", "memory.target_bytes", "memory.recovery_target_bytes", "memory.budget_mode",
                "memory.last_cut",
                "memory.last_cut.reason", "memory.last_cut.at", "memory_learned.applied", "memory_learned.bytes",
                "memory_learned.days", "memory_holders[].name", "memory_holders[].bytes",
            ], memoryLow
        ),
        (
            "memory_limit_not_kept", ["memory.budget_mode", advice + "limit_bytes", advice + "suggested_bytes", advice + "cuts?"],
            limitNotKept
        ),
        ("warming_up", ["expert_cache.limit_bytes", "requests.completed"], warmingUp),
        (
            "memory_full",
            [
                "requests.completed", "memory.cap_bytes", advice + "applied", advice + "limit_bytes", advice + "suggested_bytes",
                "mac.memory_bytes",
            ]
                + waits, memoryFull
        ),
        (
            "model_drive_slow",
            [
                source + "role", source + "detail", source + "slow_reads.high", source + "slow_reads.share",
                source + "slow_reads.hour_share",
                source + "since_rewrite.high", source + "since_rewrite.share",
            ], modelDriveSlow
        ),
        (
            "model_drive_due",
            [source + "since_rewrite.due", source + "since_rewrite.bytes", source + "since_rewrite.last_slow_after_bytes"], modelDriveDue
        ),
        ("drive_not_allowed", [source + "state", source + "drive"], driveNotAllowed),
        ("copy_drive_gone", [source + "state", source + "drive"], copyDriveGone),
        (
            "hot_copy_building",
            [source + "progress", source + "progress.done_bytes", source + "progress.total_bytes", source + "progress.seconds_left"],
            hotCopyBuilding
        ),
        ("hot_copy_no_room", [source + "detail"], hotCopyNoRoom),
        ("hot_copy_failed", [source + "detail"], hotCopyFailed),
        ("hot_copy_short", [source + "bytes", source + "budget_bytes"], hotCopyShort),
        ("hot_copy_drive_slow", [source + "since_rewrite.high", source + "since_rewrite.share"], hotCopyDriveSlow),
        ("mac_hot", ["mac_now.thermal_state"], macHot),
        ("low_power", ["mac_now.low_power", "tuning.knobs.gpu_keepalive.value", "tuning.knobs.gpu_keepalive.source"], lowPower),
        ("saves_paused", ["prompt_cache.saves_paused"], savesPaused),
        (
            "prompt_read_again",
            [
                "requests.recent[].time", "requests.recent[].prompt_restore.wanted", "requests.recent[].prompt_restore.restored",
                "requests.recent[].prompt_restore.reason",
            ], promptReadAgain
        ),
        ("queue", ["queue.queued"], queue),
        ("memory_limit_low", ["memory.budget_mode", advice + "limit_bytes", advice + "suggested_bytes", advice + "stale"], limitLow),
        ("not_tuned", ["tuning.state", "tuning.entry", "tuning.key", "tuning.knobs", "last_tuning.key"], notTuned),
    ]

    // MARK: the rules

    /// The model's drive stopped answering (the front's `degraded`).
    static func driveGone(_ s: [String: Any]) -> Found? {
        guard s["status"] as? String == "degraded" else { return nil }
        return ("The model's drive is not answering: replies wait for it", "Reconnect it; reading resumes by itself, with no reload.", nil)
    }

    /// A job of Origami's own (tuning, a memory search, a move, an add, copying experts).
    static func job(_ s: [String: Any]) -> Found? {
        guard let job = s["job"] as? [String: Any], job["state"] as? String == "running" else { return nil }
        let title = job["title"] as? String ?? "A job"
        let since = number(job["started"]).map { ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: $0)) }
        let what =
            job["kind"] as? String == "tune"
            ? "it reloads the model and times it, so requests wait or stop it" : "it shares the Mac's drives and memory with replies"
        return ("\(title) is running: \(what)", "Nothing to do: it ends by itself; Cancel on the dashboard stops it.", since)
    }

    /// Whether the memory budget is at least 1 GB under what Origami may use (`memory.target_bytes`: the cap, or your
    /// limit): the front then names the largest other apps (`memory_holders`).
    public static func memoryShort(_ s: [String: Any]) -> Bool {
        guard let m = s["memory"] as? [String: Any], let budget = number(m["budget_bytes"]), let may = number(m["target_bytes"])
        else { return false }
        return may - budget >= 1e9
    }

    /// The budget at least 1 GB under what Origami may use, in one of three cases (`MemoryGovernor`): cut below what it
    /// climbs back to (`last_cut`, `recovery_target_bytes`), which it regains by itself; started at the budget learned
    /// from this Mac's memory use (`memory_learned`; automatic mode), which it grows past by itself as memory frees; or
    /// started small because little memory was free, which it grows out of too.
    static func memoryLow(_ s: [String: Any]) -> Found? {
        guard memoryShort(s), let m = s["memory"] as? [String: Any], let budget = number(m["budget_bytes"]),
            let may = number(m["target_bytes"])
        else { return nil }
        let holders = (s["memory_holders"] as? [[String: Any]] ?? []).prefix(3).compactMap { h -> String? in
            guard let name = h["name"] as? String, let bytes = number(h["bytes"]) else { return nil }
            return "\(name) \(size(bytes))"
        }
        let apps = holders.isEmpty ? "" : "Other apps hold " + holders.joined(separator: ", ") + ". "
        let have = "Memory: Origami holds itself to \(size(budget)) of the \(size(number(m["cap_bytes"]) ?? may)) it may use"
        let back = number(m["recovery_target_bytes"]) ?? may
        let cut = m["last_cut"] as? [String: Any]
        let learned = s["memory_learned"] as? [String: Any]
        let explicit = m["budget_mode"] as? String == "explicit"
        let toLimit =
            "it climbs back to your limit by itself, half the remaining way every 20 s while macOS is not short of memory; no reload needed"
        if let cut, back - budget >= 1e9 {
            let grows = explicit ? toLimit : "it grows back by itself as memory stays free, a step every 20 to 60 s; no reload needed"
            return (
                have + ": macOS took memory back" + ((cut["reason"] as? String).map { " because " + SlowReason.cutWhy($0) } ?? ""),
                apps + "Quit what you do not need: " + grows + ".", cut["at"] as? String
            )
        }
        // Started at the budget learned from this Mac's memory use, at or past it: it grows past it by itself as memory
        // stays free (`MemoryGovernorPolicy.pastLearned`: a minute quiet before each step of up to 2 GB).
        if learned?["applied"] as? Bool == true, let bytes = number(learned?["bytes"]), bytes + 1e9 <= may, budget + 2.5e8 >= bytes {
            let days = Int((number(learned?["days"]) ?? 0).rounded())
            return (
                have + ": it started at what this Mac usually had free over the last \(days) days and grows as memory frees",
                apps + "Quitting them helps: once memory stays free for a minute it grows by up to 2 GB a step, a minute or more apart;"
                    + " no reload needed.", nil
            )
        }
        return (
            have + ": it started small because " + (explicit ? "macOS reported memory pressure" : "little memory was free"),
            apps + "Quit what you do not need: "
                + (explicit ? toLimit : "it grows into free memory by itself about 20 s later; no reload needed") + ".", nil
        )
    }

    /// `memory_limit` set above what this Mac kept (`limit_advice` from the cuts it took).
    static func limitNotKept(_ s: [String: Any]) -> Found? {
        guard let (limit, suggested, advice) = limitAdvice(s), suggested < limit else { return nil }
        let cuts = (advice["cuts"] as? Int).map { " (cut \($0) times)" } ?? ""
        let value = String(format: "%.1fGB", suggested / 1e9)
        return (
            "Your memory limit of \(size(limit)) is more than this Mac keeps free\(cuts): macOS takes it back, and the experts are read again",
            "Set it to \(size(suggested)) (Use \(size(suggested)) on the Memory line, or `origami config set memory_limit \(value)`), or remove it to let Origami choose.",
            nil
        )
    }

    /// The share of their writing time the last replies (up to five) waited for drives to read parts of the model
    /// (`SlowReason.parts`); nil before any timed reply.
    static func driveWait(_ s: [String: Any]) -> Double? {
        let recent = ((s["requests"] as? [String: Any])?["recent"] as? [[String: Any]] ?? []).map(SlowReason.parts).filter { $0.write > 0 }
        let rows = recent.suffix(5)
        let write = rows.reduce(0) { $0 + $1.write }
        guard write > 0 else { return nil }
        return min(1, rows.reduce(0) { $0 + $1.waits.reduce(0) { $0 + $1.seconds } } / write)
    }

    private static func finished(_ s: [String: Any]) -> Double? { number((s["requests"] as? [String: Any])?["completed"]) }

    /// Qwen just loaded and no reply finished yet: its memory for the model's parts starts empty and fills while the
    /// first reply is written (measured: full after one reply). GLM and DeepSeek fill theirs while loading and report
    /// no `limit_bytes`.
    static func warmingUp(_ s: [String: Any]) -> Found? {
        guard (s["expert_cache"] as? [String: Any])?["limit_bytes"] != nil, finished(s) == 0 else { return nil }
        return (
            "Origami has just loaded the model and is warming up: the first reply starts later and is written slower,"
                + " while the parts of the model it needs are read into memory",
            "Nothing to do: it is over after the first reply, when this line is gone.", nil
        )
    }

    /// Warmed up, at its full memory, and the last replies still waited on drives for a fifth of their writing time or
    /// more: the conversation uses more of the model than the memory holds. Whether this Mac can keep more.
    static func memoryFull(_ s: [String: Any]) -> Found? {
        guard finished(s) ?? 1 > 0, !memoryShort(s), let wait = driveWait(s), wait >= 0.2, let m = s["memory"] as? [String: Any],
            let may = number(m["cap_bytes"])
        else { return nil }
        let text =
            "The last replies spent about \(percent(wait)) of their writing time waiting for the drive:"
            + " this conversation uses more of the model than fits in Origami's memory"
        let advice = (s["memory_learned"] as? [String: Any])?["limit_advice"] as? [String: Any]
        let most = advice?["applied"] as? Bool == true || limitAdvice(s).map { $0.1 <= $0.0 + 5e8 } == true
        if most {
            return (
                text,
                "Nothing to change: \(size(may)) is the most this Mac kept for Origami without macOS taking memory back, which is slower still."
                    + " It differs from one conversation to the next. Quitting other apps and measuring again (Memory line) may find more.",
                nil
            )
        }
        let left = number((s["mac"] as? [String: Any])?["memory_bytes"]).map { " (they have \(size(max(0, $0 - may))) now)" } ?? ""
        return (
            text,
            "A higher memory limit helps only while macOS and other apps can spare the memory\(left): when they run short macOS takes it"
                + " back, and replies get slower. Measure on the Memory line finds the most this Mac keeps.", nil
        )
    }

    /// The model's drive reading its files far below its own speed (the Health row's `slow_reads` or `since_rewrite`).
    static func modelDriveSlow(_ s: [String: Any]) -> Found? {
        guard let row = sources(s).first(where: { $0["role"] as? String == "all" }) else { return nil }
        let slow = row["slow_reads"] as? [String: Any]
        let since = row["since_rewrite"] as? [String: Any]
        let share: Double
        if slow?["high"] as? Bool == true, row["detail"] is String {
            share = max(number(slow?["share"]) ?? 0, number(slow?["hour_share"]) ?? 0)
        } else if since?["high"] as? Bool == true {
            share = number(since?["share"]) ?? 0
        } else {
            return nil
        }
        let minutes = (row["detail"] as? String).flatMap { detail in
            detail.range(of: #"at least \d+ min"#, options: .regularExpression).map { " (" + detail[$0] + ")" }
        }
        return (
            "The model's drive reads slowly: \(Int((100 * share).rounded()))% of its reading time is far below its own speed",
            "Stop Origami and run `origami refresh`: it rewrites the model's files in place, which usually cures this\(minutes ?? "").", nil
        )
    }

    /// The model's drive near or past where it slowed down last time (`since_rewrite.due`, as the Health row warns): a
    /// share of its reading is slow already, and more will be.
    static func modelDriveDue(_ s: [String: Any]) -> Found? {
        guard let since = sources(s).first(where: { $0["role"] as? String == "all" })?["since_rewrite"] as? [String: Any],
            since["due"] as? Bool == true, let bytes = number(since["bytes"])
        else { return nil }
        let last = number(since["last_slow_after_bytes"]).map { "; last time it slowed down after \(amount($0))" } ?? ""
        return (
            "The model's drive is due for a refresh: \(Int((100 * (number(since["share"]) ?? 0)).rounded()))% of its reading was slow"
                + " in the \(amount(bytes)) read since its files were rewritten\(last)",
            "When Origami can stop for a while, run `origami refresh`: it rewrites the model's files in place.", nil
        )
    }

    /// A drive macOS has not let Origami read (Removable Volumes).
    static func driveNotAllowed(_ s: [String: Any]) -> Found? {
        guard let row = sources(s).first(where: { $0["state"] as? String == "not_allowed" }) else { return nil }
        return (
            "macOS has not let Origami use \(drive(row)): its experts are read from the model's drive instead",
            "In System Settings › Privacy & Security › Files & Folders, turn on Removable Volumes under origami (or under the program that starts it, such as a launcher), then press Reload.",
            nil
        )
    }

    /// A copy's drive (a hot copy, a further copy or a pack) unplugged.
    static func copyDriveGone(_ s: [String: Any]) -> Found? {
        guard
            let row = sources(s).first(where: {
                $0["role"] as? String != "all" && $0["state"] as? String == "not_connected"
            })
        else { return nil }
        return (
            "\(drive(row)) is not connected: the experts copied there are read from the model's drive",
            "Connect it again and press Reload, or choose another drive under Models › Where the experts are read from.", nil
        )
    }

    /// The hot copy being built (or paused while replies run): until it is done more experts come from the model's drive.
    static func hotCopyBuilding(_ s: [String: Any]) -> Found? {
        guard let hot = hot(s), ["building", "paused"].contains(hot["state"] as? String ?? "") else { return nil }
        let p = hot["progress"] as? [String: Any]
        let done = number(p?["done_bytes"])
        let total = number(p?["total_bytes"])
        let part = done.flatMap { d in total.map { " (\(size(d)) of \(size($0)))" } } ?? ""
        let left = number(p?["seconds_left"]).map { " in about \(max(1, Int(($0 / 60).rounded()))) min" } ?? ""
        return (
            "The hot copy is still being built\(part): until it is done, more experts are read from the model's drive",
            "Nothing to do: it finishes by itself\(left), pausing while replies are written.", nil
        )
    }

    /// No room for the hot copy on its drive.
    static func hotCopyNoRoom(_ s: [String: Any]) -> Found? {
        guard let hot = hot(s), hot["state"] as? String == "not_possible", let detail = hot["detail"] as? String,
            detail.contains(" free")
        else { return nil }
        return (
            "No hot copy, for lack of room: " + detail.trimmingCharacters(in: CharacterSet(charactersIn: ".")),
            "Free space on that drive, or choose another under Models › Where the experts are read from.", nil
        )
    }

    /// The hot copy failed to build.
    static func hotCopyFailed(_ s: [String: Any]) -> Found? {
        guard let hot = hot(s), hot["state"] as? String == "failed" else { return nil }
        return (
            "The hot copy could not be built" + ((hot["detail"] as? String).map { ": " + $0 } ?? ""),
            "It tries again by itself later; Build now under Models › Where the experts are read from tries at once.", nil
        )
    }

    /// The hot copy holding at least 2 GB less than its size.
    static func hotCopyShort(_ s: [String: Any]) -> Found? {
        guard let hot = hot(s), hot["state"] as? String == "ready", let bytes = number(hot["bytes"]),
            let budget = number(hot["budget_bytes"]), budget - bytes >= 2e9
        else { return nil }
        return (
            "The hot copy holds \(size(bytes)) of its \(size(budget))",
            "Nothing to do: it adds the rest by itself after replies.", nil
        )
    }

    /// The hot copy's own files read slowly (`since_rewrite` on its row).
    static func hotCopyDriveSlow(_ s: [String: Any]) -> Found? {
        guard let hot = hot(s), let since = hot["since_rewrite"] as? [String: Any], since["high"] as? Bool == true else { return nil }
        return (
            "The hot copy's drive reads it slowly: \(Int((100 * (number(since["share"]) ?? 0)).rounded()))% of its reading time",
            "Quit `origami serve` and run `origami refresh --copy`: it rewrites the copy in place, on fresh blocks.",
            nil
        )
    }

    /// macOS slowing the chip for heat (`ProcessInfo.thermalState` serious or critical).
    static func macHot(_ s: [String: Any]) -> Found? {
        guard let state = (s["mac_now"] as? [String: Any])?["thermal_state"] as? Int, state >= 2 else { return nil }
        return (
            "The Mac is \(state >= 3 ? "very " : "")hot: macOS slows its chip down",
            "Nothing to do: it recovers by itself as the Mac cools; keep its vents clear.", nil
        )
    }

    /// Low Power Mode: a slower chip, and for GLM and DeepSeek no GPU keep-alive (its automatic setting turns it off).
    static func lowPower(_ s: [String: Any]) -> Found? {
        guard (s["mac_now"] as? [String: Any])?["low_power"] as? Bool == true else { return nil }
        let knob = ((s["tuning"] as? [String: Any])?["knobs"] as? [String: Any])?["gpu_keepalive"] as? [String: Any]
        let keepAlive = knob?["value"] as? Bool == false && knob?["source"] as? String == "default"
        return (
            "Low Power Mode is on: macOS slows the chip" + (keepAlive ? ", and Origami turns its GPU keep-alive off" : ""),
            "Turn Low Power Mode off in System Settings › Battery (Energy on a desktop Mac).", nil
        )
    }

    /// Conversation saves paused for lack of disk (`prompt_cache.saves_paused`, `PromptCache.pausedNote`).
    static func savesPaused(_ s: [String: Any]) -> Found? {
        guard let note = (s["prompt_cache"] as? [String: Any])?["saves_paused"] as? String, !note.isEmpty else { return nil }
        let facts = (note.components(separatedBy: "; free disk space").first ?? note).components(separatedBy: "paused: ").last ?? ""
        return (
            "Conversation saves are paused, so the next turn of a conversation reads its whole prompt again: " + facts,
            "Free disk space where saved prompts are kept, or lower the context; saves resume by themselves.", nil
        )
    }

    /// The last request read at least 1,000 prompt tokens again that a saved prompt should have given
    /// (`prompt_restore`, its reason from `PromptCache.explainRestore`).
    static func promptReadAgain(_ s: [String: Any]) -> Found? {
        guard let last = ((s["requests"] as? [String: Any])?["recent"] as? [[String: Any]])?.last,
            let restore = last["prompt_restore"] as? [String: Any]
        else { return nil }
        let lost = Int(number(restore["wanted"]) ?? 0) - Int(number(restore["restored"]) ?? 0)
        guard lost >= 1000 else { return nil }
        let reason = restore["reason"] as? String ?? ""
        let fix: String
        switch reason {
        case "save_skipped_disk": fix = "Free disk space where saved prompts are kept; the next save goes through."
        case "save_not_finished":
            fix = "Nothing to do: it happens when a turn follows the last reply at once; later turns reuse it."
        case "entry_evicted":
            fix = "Keep the conversation (Saved prompts › Keep) so newer ones do not replace it, or give saved prompts more disk."
        case "save_failed": fix = "The server log says why the save failed; later turns are saved again."
        case "saved_by_other_build": fix = "Nothing to do: this version saves it again, and later turns reuse it."
        case _ where reason.hasPrefix("prefix_differs_at="):
            fix =
                "Nothing Origami can change: the app changed the start of the prompt (a date or tool list in the system prompt often does)."
        default: fix = "Nothing to do: later turns of the conversation reuse what this one saved."
        }
        // As the last request's line says it (`SlowReason`); a reason this build has no words for, as the general case.
        let why = reason.isEmpty ? "" : " because " + SlowReason.why(reason)
        return (
            "\(SlowReason.grouped(lost)) tokens of the last prompt had to be read again\(why)", fix, last["time"] as? String
        )
    }

    /// Requests waiting for the one being answered.
    static func queue(_ s: [String: Any]) -> Found? {
        guard let waiting = (s["queue"] as? [String: Any])?["queued"] as? Int, waiting > 0 else { return nil }
        return (
            "\(waiting) request\(waiting == 1 ? " waits" : "s wait") for the one being answered",
            "Nothing to do: Origami answers one request at a time, and they run in turn.", nil
        )
    }

    /// Advice: `memory_limit` set below what the memory search measured holds on this Mac (its result not stale).
    static func limitLow(_ s: [String: Any]) -> Found? {
        guard let (limit, suggested, advice) = limitAdvice(s), suggested >= limit + 5e8, !(advice["stale"] is String) else { return nil }
        let value = String(format: "%.1fGB", suggested / 1e9)
        return (
            "Your memory limit of \(size(limit)) is under the \(size(suggested)) a measurement found this Mac keeps for Origami",
            "Use \(size(suggested)) on the Memory line, or `origami config set memory_limit \(value)`.", nil
        )
    }

    /// Advice: never tuned for this Mac (no entry, and no run that found the defaults fastest).
    static func notTuned(_ s: [String: Any]) -> Found? {
        guard let t = s["tuning"] as? [String: Any], t["state"] as? String == "none", t["entry"] == nil || t["entry"] is NSNull,
            !((t["knobs"] as? [String: Any]) ?? [:]).isEmpty
        else { return nil }
        if let last = s["last_tuning"] as? [String: Any], last["key"] as? String == t["key"] as? String { return nil }
        return (
            "Not tuned for this Mac yet: tuning may find faster settings with the same replies",
            "Tune for this Mac in Setup (about 8 minutes, while nothing else runs).", nil
        )
    }

    // MARK: helpers

    static func percent(_ share: Double) -> String { "\(Int((100 * share).rounded()))%" }
    static func number(_ value: Any?) -> Double? { (value as? NSNumber)?.doubleValue }
    static func amount(_ bytes: Double) -> String { DriveReadHistory.amount(bytes) }
    static func size(_ bytes: Double) -> String { OrigamiPlan.gigabytes(UInt64(max(0, bytes))) }
    static func sources(_ s: [String: Any]) -> [[String: Any]] {
        (s["experts_from"] as? [String: Any])?["sources"] as? [[String: Any]] ?? []
    }
    static func hot(_ s: [String: Any]) -> [String: Any]? { sources(s).first { $0["role"] as? String == "hot" } }
    static func drive(_ row: [String: Any]) -> String {
        (row["drive"] as? String).map { $0 == "internal drive" ? "this Mac's drive" : $0 } ?? "a drive"
    }
    /// The limit in force and a suggestion for it: `memory_learned.limit_advice` while a limit of yours is set.
    static func limitAdvice(_ s: [String: Any]) -> (Double, Double, [String: Any])? {
        guard (s["memory"] as? [String: Any])?["budget_mode"] as? String == "explicit",
            let advice = (s["memory_learned"] as? [String: Any])?["limit_advice"] as? [String: Any],
            let limit = number(advice["limit_bytes"]), let suggested = number(advice["suggested_bytes"])
        else { return nil }
        return (limit, suggested, advice)
    }
}
