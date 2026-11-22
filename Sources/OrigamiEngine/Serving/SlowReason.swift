import Foundation

/// Why a finished request (a `/v1/status` `requests.recent` row) was slow, in one line of plain words picked by fixed
/// rules: `origami status` prints it, and the dashboard's Now tab (`slowReason` in its script) applies the same rules
/// (both checked against `Tests/Fixtures/status/slow-reasons.json`). Rules, first that holds:
/// memory held at least 1 GB under its limit on average over the request; else the larger of a prompt read again
/// (at least 1,000 tokens a saved prompt should have given) and the longest wait on one drive's reads, when it took at
/// least 2 s and a fifth of the request; else nothing unusual.
public enum SlowReason {
    /// A restore's `reason` (`PromptCache.explainRestore`) in words.
    static let restoreWhy = [
        "save_skipped_disk": "the saved copy was skipped for disk space", "save_failed": "saving it had failed",
        "save_not_finished": "its save had not finished", "entry_evicted": "its saved copy had been replaced by newer ones",
        "entry_not_used": "the saved copy was not used", "no_entry": "it had not been saved",
        "saved_by_other_build": "it was saved by another Origami build",
    ]
    /// A restore's `reason` in words; a reason without words shows as its code.
    static func why(_ reason: String) -> String {
        reason.hasPrefix("prefix_differs_at=")
            ? "the prompt differed from the saved one at token " + reason.dropFirst(18) : restoreWhy[reason] ?? reason
    }

    /// Where the request's time went: reading the prompt and writing (stage seconds), the GPU's share of writing, and
    /// each drive's wait while the reply waited on reads; the prompt tokens read new and restored, and those a saved
    /// prompt should have given (`lost`, with its reason).
    public struct Parts {
        public var read = 0.0, write = 0.0, gpu = 0.0
        public var waits: [(drive: String, seconds: Double)] = []
        public var fresh = 0, restored = 0, lost = 0
        public var why: String?
    }

    public static func parts(_ r: [String: Any]) -> Parts {
        func number(_ value: Any?) -> Double { (value as? NSNumber)?.doubleValue ?? 0 }
        let counters = r["phase_counters"] as? [String: Any] ?? [:]
        let stages = r["stage_seconds"] as? [String: Any] ?? [:]
        var p = Parts()
        p.read = number(stages["prefill_and_capture"] ?? stages["prefill_and_save"])
        p.write = number(stages["generation"])
        // GLM and DeepSeek: each drive's busy time while the reply waited on reads.
        for (key, value) in counters {
            let parts = key.split(separator: ".")
            guard parts.count == 5, parts[0] == "decode", parts[1] == "drive_activity", parts[3] == "busy_seconds", parts[4] == "reads"
            else { continue }
            let lane = String(parts[2])
            let name = lane == "model_drive" ? "the model's drive" : "copy " + String((Int(lane.dropFirst("copy_place_".count)) ?? 0) + 1)
            p.waits.append((name, number(value)))
        }
        // Qwen: the reply's wait on late reads, split by each drive's read time.
        let late = number(counters["generation.decode_late_wait_seconds"])
        let times = [
            ("the model's drive", number(counters["generation.reads_internal_seconds"])),
            ("the pack's drive", number(counters["generation.reads_external_seconds"])),
            ("the hot copy", number(counters["generation.reads_hot_seconds"])),
        ]
        let all = times.reduce(0) { $0 + $1.1 }
        if late > 0, all > 0 { p.waits += times.filter { $0.1 > 0 }.map { ($0.0, late * $0.1 / all) } }
        p.waits.sort { $0.seconds > $1.seconds }
        p.gpu = number(counters["generation.decode_gpu_seconds"] ?? counters["decode.drive_activity.wall_seconds.gpu"])
        let prompt = Int(number(r["prompt_tokens"]))
        p.restored = Int(number(r["cached_tokens"]))
        p.fresh = max(0, prompt - p.restored)
        if let restore = r["prompt_restore"] as? [String: Any] {
            p.lost = max(0, Int(number(restore["wanted"])) - Int(number(restore["restored"])))
            p.why = restore["reason"] as? String
        }
        return p
    }

    /// The verdict: "slow because: …" or "nothing unusual". `status` (the `/v1/status` it came with) adds the hot copy's
    /// swaps waiting to a wait on the model's drive.
    public static func verdict(request r: [String: Any], status s: [String: Any] = [:]) -> String {
        let p = parts(r)
        let seconds = (r["seconds"] as? NSNumber)?.doubleValue ?? 0
        guard seconds > 0 else { return "nothing unusual" }
        let short = ((r["memory_budget_deficit_gb_seconds"] as? NSNumber)?.doubleValue ?? 0) / seconds
        if short >= 1 {
            return
                "slow because: memory was cut by about \(String(format: "%.1f", short)) GB for this request's \(Int(seconds.rounded())) s"
        }
        let reread = p.lost >= 1000 && p.fresh > 0 ? p.read * min(1, Double(p.lost) / Double(p.fresh)) : 0
        let wait = p.waits.first
        let enough = { (cost: Double) in cost >= 2 && cost >= 0.2 * seconds }
        if enough(reread), reread >= wait?.seconds ?? 0 {
            let why = p.why.map(Self.why)
            return "slow because: \(grouped(p.lost)) prompt tokens were read again" + (why.map { " — " + $0 } ?? "")
        }
        if let wait, enough(wait.seconds) {
            let sources = (s["experts_from"] as? [String: Any])?["sources"] as? [[String: Any]] ?? []
            let backlog = sources.filter { $0["role"] as? String == "hot" }.reduce(0) {
                $0 + ((($1["backlog_records"] as? NSNumber)?.intValue) ?? 0)
            }
            return "slow because: \(wait.drive) kept the reply waiting for \(Int(wait.seconds.rounded())) s"
                + (wait.drive == "the model's drive" && backlog > 0 ? " — the hot copy has \(grouped(backlog)) swaps waiting" : "")
        }
        return "nothing unusual"
    }

    /// 6100 as "6,100", as the dashboard writes numbers.
    static func grouped(_ n: Int) -> String {
        let digits = String(abs(n))
        let head = digits.count % 3 == 0 ? 3 : digits.count % 3
        var out = String(digits.prefix(head))
        var at = digits.index(digits.startIndex, offsetBy: head)
        while at < digits.endIndex {
            let next = digits.index(at, offsetBy: 3)
            out += "," + digits[at..<next]
            at = next
        }
        return (n < 0 ? "-" : "") + out
    }
}
