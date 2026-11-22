import Foundation

/// `origami_tune`: execution switches a request from this Mac's own tuner may set for
/// itself, so `origami tune` compares settings request by request in the one loaded model. Only these keys, only with
/// `origami_mode: exact` except explicit delayed-expert experiments;
/// the front refuses the field from other devices.
public struct TuneOverrides: Sendable, Equatable {
    /// Qwen: the GPU keep-alive spin for this reply.
    public var gpuKeepAlive: Bool?
    /// GLM and DeepSeek: decode reads the next layer's guessed experts; how many of them (-1: all); spins through reads.
    public var familyPrefetch: Bool?
    public var familyPrefetchDepth: Int?
    public var familySpin: Bool?
    /// Start a selected delayed expert's read during the immediate batch. Has no effect without delayedExperts.
    public var familyEarlyDelayedReads: Bool?
    /// Prompt grouping within already reserved memory; compute tiles and family slices keep their existing size.
    public var qwenPrefillScope: Int?
    public var familyPrefillSlices: Int?
    /// Local diagnostic only: target verifier cost without loading a drafter.
    public var qwenPairProbe: Bool?
    /// Approximate-only replacement for eligible skipped contributions.
    public var delayedExperts: DelayedExpertPolicy?
    public var delayedExpertsOff = false
    public init(
        gpuKeepAlive: Bool? = nil, familyPrefetch: Bool? = nil, familyPrefetchDepth: Int? = nil, familySpin: Bool? = nil,
        qwenPrefillScope: Int? = nil, familyPrefillSlices: Int? = nil,
        familyEarlyDelayedReads: Bool? = nil
    ) {
        self.gpuKeepAlive = gpuKeepAlive
        self.familyPrefetch = familyPrefetch
        self.familyPrefetchDepth = familyPrefetchDepth
        self.familySpin = familySpin
        self.qwenPrefillScope = qwenPrefillScope
        self.familyPrefillSlices = familyPrefillSlices
        self.familyEarlyDelayedReads = familyEarlyDelayedReads
    }
    public static let keys = [
        "gpu_keepalive", "family_prefetch", "family_prefetch_depth", "family_spin",
        "family_early_delayed_reads", "qwen_prefill_scope", "family_prefill_slices",
        "qwen_pair_probe", "delayed_experts",
    ]
    /// Switches the H3 screen retired: accepted with any value and ignored, so a saved tuning result or a request that
    /// still names one keeps working.
    public static let retiredKeys = ["family_arriving_first", "family_engram_overlap", "expert_admission", "qwen_arriving_first"]
    /// Whether a request body sets `origami_tune` (the front refuses it from other devices). Parsed as the server will parse
    /// it: a JSON escape spells the key without its bytes (`origami\u005ftune`).
    public static func requested(in body: Data) -> Bool {
        guard body.range(of: Data("origami_tune".utf8)) != nil || body.range(of: Data("\\u".utf8)) != nil else { return false }
        return ((try? JSONSerialization.jsonObject(with: body)) as? [String: Any])?["origami_tune"] != nil
    }

    /// The field as sent, or tuning.json's `values`: booleans, and a depth of 0...64 or "all".
    public init(json: Any) throws {
        guard let object = json as? [String: Any] else { throw ServingFailure("origami_tune must be an object") }
        if let key = object.keys.first(where: { !Self.keys.contains($0) && !Self.retiredKeys.contains($0) }) {
            throw ServingFailure("origami_tune does not take \(key)")
        }
        func flag(_ key: String) throws -> Bool? {
            guard let value = object[key] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw ServingFailure("origami_tune.\(key) must be true or false")
            }
            return number.boolValue
        }
        gpuKeepAlive = try flag("gpu_keepalive")
        familyPrefetch = try flag("family_prefetch")
        familySpin = try flag("family_spin")
        familyEarlyDelayedReads = try flag("family_early_delayed_reads")
        qwenPairProbe = try flag("qwen_pair_probe")
        if let value = object["delayed_experts"] {
            if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() {
                if number.boolValue { delayedExperts = try DelayedExpertPolicy().validated() } else { delayedExpertsOff = true }
            } else if let fields = value as? [String: Any] {
                let keys: Set<String> = ["version", "max_mass", "max_expert_mass", "max_experts"]
                guard Set(fields.keys).isSubset(of: keys) else { throw ServingFailure("Unknown delayed expert policy field") }
                func number(_ name: String, _ fallback: Double) throws -> Double {
                    guard let value = fields[name] else { return fallback }
                    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else {
                        throw ServingFailure("delayed_experts.\(name) must be a finite number")
                    }
                    return n.doubleValue
                }
                let version = try number("version", 1)
                let count = try number("max_experts", 1)
                guard let version = Int(exactly: version), let count = Int(exactly: count) else {
                    throw ServingFailure("Delayed expert version and count must be integers")
                }
                do {
                    delayedExperts = try DelayedExpertPolicy(
                        version: version,
                        maximumMass: number("max_mass", 0.05), maximumExpertMass: number("max_expert_mass", 0.05),
                        maximumExperts: count
                    ).validated()
                } catch { throw ServingFailure(String(describing: error)) }
            } else {
                throw ServingFailure("origami_tune.delayed_experts must be boolean or a bounded policy object")
            }
        }
        if let value = object["family_prefetch_depth"] {
            if value as? String == "all" {
                familyPrefetchDepth = -1
            } else if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let depth = Int(exactly: number),
                (0...64).contains(depth)
            {
                familyPrefetchDepth = depth
            } else {
                throw ServingFailure("origami_tune.family_prefetch_depth must be 0 to 64 or \"all\"")
            }
        }
        func integer(_ key: String, range: ClosedRange<Int>, multiple: Int = 1) throws -> Int? {
            guard let value = object[key] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let value = Int(exactly: number),
                range.contains(value), value % multiple == 0
            else {
                throw ServingFailure("origami_tune.\(key) must be \(range.lowerBound) to \(range.upperBound) in steps of \(multiple)")
            }
            return value
        }
        qwenPrefillScope = try integer("qwen_prefill_scope", range: 256...8192, multiple: 256)
        familyPrefillSlices = try integer("family_prefill_slices", range: 1...4)
    }
    public var wire: [String: Any] {
        var object: [String: Any] = [:]
        if let gpuKeepAlive { object["gpu_keepalive"] = gpuKeepAlive }
        if let familyPrefetch { object["family_prefetch"] = familyPrefetch }
        if let familyPrefetchDepth { object["family_prefetch_depth"] = familyPrefetchDepth < 0 ? "all" as Any : familyPrefetchDepth }
        if let familySpin { object["family_spin"] = familySpin }
        if let familyEarlyDelayedReads { object["family_early_delayed_reads"] = familyEarlyDelayedReads }
        if let qwenPrefillScope { object["qwen_prefill_scope"] = qwenPrefillScope }
        if let familyPrefillSlices { object["family_prefill_slices"] = familyPrefillSlices }
        if let qwenPairProbe { object["qwen_pair_probe"] = qwenPairProbe }
        if delayedExpertsOff {
            object["delayed_experts"] = false
        } else if let policy = delayedExperts {
            object["delayed_experts"] =
                [
                    "version": policy.version, "max_mass": policy.maximumMass,
                    "max_expert_mass": policy.maximumExpertMass, "max_experts": policy.maximumExperts,
                ] as [String: Any]
        }
        return object
    }
}

/// The self-tuner's rules, kept apart from its HTTP client so the checks run them without a model: the decision over
/// rotated rounds, when a saved result goes stale, and which value wins.
public enum SelfTuning {
    /// Bumped whenever a tuned setting's code or default changes: results of another version are not applied.
    public static let version = 5
    public static let minimumGain = 0.03, noiseLimit = 0.15, promptLoss = 0.02, earlyDrop = 0.05, driftLimit = 0.03
    /// The warm-up ends when two passes in a row decode within this of each other.
    public static let settleLimit = 0.05

    /// Seconds of a reply's stalls: what its gaps between tokens spend beyond four times their median (none with fewer
    /// than 8 gaps). A stall the Mac adds now and then (a drive pausing, macOS reclaiming memory: DeepSeek on the M2 Pro
    /// showed ~8.5 s about every 90 s) lands in one arm's reply and not its repeat; a setting that slows every token stays.
    public static func stallSeconds(gaps: [Double]) -> Double {
        guard gaps.count >= 8 else { return 0 }
        let limit = 4 * median(gaps)
        return limit > 0 ? gaps.reduce(0) { $0 + max(0, $1 - limit) } : 0
    }
    /// Whether two warm-up passes' decode speeds are within `settleLimit` of each other.
    public static func settled(_ previous: Double, _ next: Double) -> Bool { previous > 0 && abs(next / previous - 1) <= settleLimit }
    /// The longest `origami tune --minutes`: a run on the default limit grows to it when this model's measurements need it.
    public static let maximumMinutes = 30.0
    /// The comparison that fits in `left` seconds, plus at most `more`: how many of the candidates, in order (`needs`: the
    /// samples each still needs, its whole set), and the seconds to add. A winner's confirmation (`confirm` samples) and
    /// one round (3) of margin are kept; at least one candidate is planned, so a short limit still measures and keeps.
    public static func fit(needs: [Int], confirm: Int, sampleSeconds: Double, left: Double, more: Double) -> (
        count: Int, extra: Double
    ) {
        func seconds(_ count: Int) -> Double { Double(needs.prefix(count).reduce(0, +) + confirm + 3) * sampleSeconds }
        let count = needs.isEmpty ? 0 : max(1, (0...needs.count).last { seconds($0) <= left + more } ?? 0)
        return (count, min(more, max(0, seconds(count) - left)))
    }
    /// Soft staleness: applied, but the dashboard offers a re-check.
    public static let recheckAge: TimeInterval = 90 * 86_400

    /// One arm in one round: decode and prompt tokens a second, and its replies (text with token count) in order.
    public struct Sample: Sendable {
        public var decode: Double, prompt: Double, replies: [String]
        public init(decode: Double, prompt: Double, replies: [String]) {
            self.decode = decode
            self.prompt = prompt
            self.replies = replies
        }
    }
    /// Arm name to sample: "base" (the current settings), "control" (the base again) and each candidate.
    public typealias Round = [String: Sample]

    /// Prefill groups compete on prompt speed; decode switches keep their original objective.
    public enum Objective: String, Sendable {
        case decode, prompt
        public func speed(_ sample: Sample) -> Double { self == .decode ? sample.decode : sample.prompt }
    }

    public struct Decision: Sendable {
        /// `faster`, `already_fastest` or `too_noisy`.
        public var outcome: String
        public init(outcome: String) { self.outcome = outcome }
        public var objective = Objective.decode
        public var winner: String?
        /// The winner's median ratio less one (0 without one), the gain an arm needed, the control gap and the base's drift.
        public var gain = 0.0, threshold = 1.0, noise = 0.0, drift = 0.0
        /// Each arm's objective speed ratio to its round's base, round by round.
        public var ratios: [String: [Double]] = [:]
        public var notes: [String] = []
    }

    public static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        let n = s.count
        return n == 0 ? 0 : n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// Arms more than `earlyDrop` slower than the round's base: not run again (a pathological keep-alive shows at once).
    public static func dropped(after round: Round) -> Set<String> {
        guard let base = round["base"], base.decode > 0 else { return [] }
        return Set(round.filter { $0.key != "base" && $0.key != "control" && $0.value.decode < base.decode * (1 - earlyDrop) }.keys)
    }

    /// Accepts a candidate only if it was faster than its round's base by max(3%, twice the control's gap) in every
    /// round, its best other speed is at most 2% below the base's best (a stall only ever slows a reading), and its
    /// replies equal the base's; the single best one wins. A control gap
    /// above 15%, or a base that drifted more than 3% and more than half the gain, changes nothing. `threshold` (a confirm)
    /// judges against the gain the first rounds asked for instead of measuring the noise again.
    public static func decide(_ rounds: [Round], candidates: [String], threshold given: Double? = nil, objective: Objective = .decode)
        -> Decision
    {
        var decision = Decision(outcome: "already_fastest")
        decision.objective = objective
        let usable = rounds.filter { $0["base"].map { objective.speed($0) > 0 } ?? false }
        guard !usable.isEmpty else {
            decision.outcome = "too_noisy"
            decision.notes.append("no round measured the base")
            return decision
        }
        func ratio(_ arm: String, _ round: Round, _ speed: (Sample) -> Double) -> Double? {
            guard let sample = round[arm], let base = round["base"], speed(base) > 0 else { return nil }
            return speed(sample) / speed(base)
        }
        let control = usable.compactMap { ratio("control", $0, objective.speed) }
        decision.noise = median(control.map { abs($0 - 1) })
        decision.threshold = given ?? 1 + max(minimumGain, 2 * decision.noise)
        let bases = usable.map { objective.speed($0["base"]!) }
        decision.drift = abs(bases.last! / bases.first! - 1)
        if given == nil, decision.noise > noiseLimit {
            decision.outcome = "too_noisy"
            decision.notes.append(
                String(format: "the base and its repeat differed by %.1f%% (more than %.0f%%)", decision.noise * 100, noiseLimit * 100))
            return decision
        }
        // Unstable baseline output cannot establish output equivalence.
        let reference = usable[0]["base"]!.replies
        let baseSteady = usable.allSatisfy { $0["base"]!.replies == reference && ($0["control"].map { $0.replies == reference } ?? true) }
        if !baseSteady {
            decision.outcome = "too_noisy"
            decision.notes.append("Baseline replies varied; output equivalence could not be verified")
            return decision
        }
        var best: (arm: String, gain: Double)?
        for arm in candidates {
            let ratios = usable.compactMap { ratio(arm, $0, objective.speed) }
            guard ratios.count == usable.count else { continue }
            decision.ratios[arm] = ratios
            if baseSteady, usable.contains(where: { $0[arm]!.replies != $0["base"]!.replies }) {
                decision.notes.append("\(arm): replies differed from the base; the setting was not accepted")
                continue
            }
            guard ratios.allSatisfy({ $0 >= decision.threshold }) else { continue }
            let secondary: Objective = objective == .decode ? .prompt : .decode
            let bestOther = { (name: String) in usable.compactMap { $0[name].map(secondary.speed) }.max() ?? 0 }
            let other = bestOther("base") > 0 ? bestOther(arm) / bestOther("base") : 0
            guard other == 0 || other >= 1 - promptLoss else {
                decision.notes.append(
                    String(format: "\(arm): \(secondary == .prompt ? "reading" : "writing") %.0f%% slower", (1 - other) * 100))
                continue
            }
            let gain = median(ratios) - 1
            if best.map({ gain > $0.gain }) ?? true { best = (arm, gain) }
        }
        guard let best else {
            if given == nil && decision.noise > 0.04 {
                decision.outcome = "too_noisy"
                decision.notes.append("Measured gains did not clear twice the control variation")
            }
            return decision
        }
        if given == nil, decision.drift > driftLimit && decision.drift > best.gain / 2 {
            decision.outcome = "too_noisy"
            decision.notes.append(String(format: "the base drifted %.1f%% from the first round to the last", decision.drift * 100))
            return decision
        }
        decision.outcome = "faster"
        decision.winner = best.arm
        decision.gain = best.gain
        return decision
    }

    /// A bounded neighboring search, performed only by `origami tune`, never on the serving path.
    /// Larger groups cannot exceed the existing working-memory allowance.
    public static func prefillCandidates(current: Int, maximum: Int, step: Int) -> [Int] {
        guard step > 0, maximum >= step else { return [] }
        let cap = maximum / step * step
        let current = max(step, min(current / step * step, cap))
        let larger = current > cap / 2 ? cap : current * 2
        return Array(Set([max(step, current / 2 / step * step), larger])).filter { $0 != current }.sorted()
    }

    /// A family model's guess depth to try: the deepest rank of guessed experts used at least half the time, or nil
    /// when every rank is (or nothing was guessed yet).
    public static func guessDepth(issued: [Int], used: [Int]) -> Int? {
        let ranks = zip(issued, used).map { $0.0 > 0 && 2 * $0.1 >= $0.0 }
        guard issued.contains(where: { $0 > 0 }), let deepest = ranks.lastIndex(of: true) else { return nil }
        return deepest + 1 < issued.count ? deepest + 1 : nil
    }

    /// Whether a saved result fits this Mac and model now: `current` (not applied, re-tune), `recheck` (applied, re-check
    /// offered: macOS's major version changed or it is over 90 days old) or `stale` (another Mac, memory size, GPU, model,
    /// drive set or tuning version), with the reason.
    public static func staleness(saved: [String: Any], current: [String: Any], date: Date, now: Date = Date()) -> (
        state: String, reason: String?
    ) {
        let hard: [(String, String)] = [
            ("tuning_version", "Origami's tuning changed"), ("mac", "another Mac"), ("gpu", "another GPU"),
            ("memory_bytes", "another memory size"), ("model", "another model"), ("drives", "other drives"),
        ]
        func text(_ value: Any?) -> String { value.map { "\($0)" } ?? "" }
        for (key, reason) in hard where text(saved[key]) != text(current[key]) { return ("stale", reason) }
        if text(saved["macos"]) != text(current["macos"]) { return ("recheck", "macOS was updated") }
        if now.timeIntervalSince(date) > recheckAge { return ("recheck", "it is over 90 days old") }
        return ("current", nil)
    }

    /// A setting's value and where it came from: the user's own (config, flag or environment) beats a tuned one, which
    /// beats the plan's default.
    public static func resolve<T>(explicit: T?, tuned: T?, fallback: T) -> (value: T, source: String) {
        if let explicit { return (explicit, "setting") }
        if let tuned { return (tuned, "tuned") }
        return (fallback, "default")
    }
}

/// What this Mac's other apps leave Origami over time, kept in the data folder (`memory-history.json`): per 10 minutes
/// the least budget that would have kept free memory at the governor's floor, and every trim with the budget it cut to.
/// From those it learns a start budget that would have avoided most trims (`budget`), never above today's ceiling.
public final class MemoryHistory: @unchecked Sendable {
    public struct Bucket: Codable, Sendable, Equatable {
        /// Start of the 10 minutes (seconds since 1970), the least fitting budget seen in them, and the regrowths.
        public var t: Double, fit: UInt64, grows: Int
        public init(t: Double, fit: UInt64, grows: Int = 0) {
            self.t = t
            self.fit = fit
            self.grows = grows
        }
    }
    public struct Trim: Codable, Sendable, Equatable {
        /// `limit`: the ceiling in force when it was cut (the memory_limit, measured or automatic cap); `model` and
        /// `context`: what was loaded. Each is nil in files written before it was recorded.
        public var t: Double, to: UInt64, limit: UInt64?
        public var model: String?, context: Int?
        public init(t: Double, to: UInt64, limit: UInt64? = nil, model: String? = nil, context: Int? = nil) {
            self.t = t
            self.to = to
            self.limit = limit
            self.model = model
            self.context = context
        }
        /// A cut is evidence against `ceiling` only if it happened under it or a lower one; one of unknown ceiling only
        /// in automatic mode (`explicit` false), never against a memory_limit or a measured one. With a `scope` (the
        /// model and context loaded now), only that model's cuts at that context count, and one of unknown model only in
        /// automatic mode.
        public func counts(against ceiling: UInt64, explicit: Bool, scope: Scope? = nil) -> Bool {
            if let scope {
                guard let model else { return !explicit && (limit.map { $0 <= ceiling } ?? true) }
                guard model == scope.model && context == scope.context else { return false }
            }
            return limit.map { $0 <= ceiling } ?? !explicit
        }
    }
    /// The model and context a history learns for (`Trim.counts`).
    public struct Scope: Sendable, Equatable {
        public let model: String, context: Int
        public init(model: String, context: Int) {
            self.model = model
            self.context = context
        }
    }
    struct File: Codable {
        var identity: String
        var buckets: [Bucket]
        var trims: [Trim]
    }
    public static let bucketSeconds: Double = 600, window: Double = 14 * 86_400, minimumBuckets = 6

    public let file: URL, identity: String
    /// The model and context loaded: stamped on each cut, and only their cuts count (`Trim.counts`); nil counts every cut.
    public let scope: Scope?
    private let lock = NSLock()
    private var open: Bucket?
    private var buckets: [Bucket] = [], trims: [Trim] = []

    /// A file kept for another Mac (or memory size), or one that cannot be read, starts over.
    public init(file: URL, identity: String, scope: Scope? = nil) {
        self.file = file
        self.identity = identity
        self.scope = scope
        // Unreadable: set aside before this one writes (`StateFile`); another Mac's is replaced on purpose.
        if let saved = StateFile.load(file, empty: nil, { try JSONDecoder().decode(File?.self, from: $0) }) ?? nil,
            saved.identity == identity
        {
            buckets = saved.buckets
            trims = saved.trims
        }
    }

    /// The governor's check: `fit`, the budget that would leave free memory exactly at its floor now.
    public func observe(fit: UInt64, now: Date = Date()) {
        let start = (now.timeIntervalSince1970 / Self.bucketSeconds).rounded(.down) * Self.bucketSeconds
        lock.withLock {
            if let current = open, current.t != start {
                buckets.append(current)
                open = nil
                save()
            }
            if open == nil { open = Bucket(t: start, fit: fit) } else { open!.fit = min(open!.fit, fit) }
        }
    }
    public func trimmed(to budget: UInt64, limit: UInt64, now: Date = Date()) {
        lock.withLock {
            trims.append(Trim(t: now.timeIntervalSince1970, to: budget, limit: limit, model: scope?.model, context: scope?.context))
            if let fit = open?.fit { open!.fit = min(fit, budget) }
            // Written at once: a server stopped within 10 minutes still remembers its cuts.
            save()
        }
    }
    public func grew() { lock.withLock { open?.grows += 1 } }
    /// The dashboard's reset: nothing learned, the file gone.
    public func reset() {
        lock.withLock {
            buckets = []
            trims = []
            open = nil
            try? FileManager.default.removeItem(at: file)
        }
    }

    /// Under `lock`: the last 14 days written (failures cost only the learning).
    private func save() {
        let cut = Date().timeIntervalSince1970 - Self.window
        buckets.removeAll { $0.t < cut }
        trims.removeAll { $0.t < cut }
        guard let data = try? JSONEncoder().encode(File(identity: identity, buckets: buckets, trims: trims)) else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: file, options: .atomic)
    }

    /// The start budget of automatic mode: the 10th percentile of the 10-minute least fits, and with three trims or more
    /// at most the 20th percentile of what they cut to (so about four in five would not have happened), at least
    /// `minimum`; nil with less than an hour of history or nothing below `cap`, today's ceiling. `explicit` (a limit of
    /// `cap`): free memory does not count, only a limit cut three times or more, to that 20th percentile; it is the level
    /// `limitAdvice` suggests, never a start (a limit always starts at itself). Only cuts that count against `ceiling`
    /// (`Trim.counts`; default `cap`), the limit in force now, are used.
    public static func budget(
        buckets: [Bucket], trims: [Trim], cap: UInt64, minimum: UInt64, explicit: Bool = false, ceiling: UInt64? = nil,
        scope: Scope? = nil, now: Date = Date()
    ) -> UInt64? {
        let cut = now.timeIntervalSince1970 - window
        let fits = buckets.filter { $0.t >= cut }.map(\.fit).sorted()
        let cuts = trims.filter {
            $0.t >= cut && (!explicit || $0.to < cap) && $0.counts(against: ceiling ?? cap, explicit: explicit, scope: scope)
        }
        .map(\.to).sorted()
        guard explicit ? cuts.count >= 3 : fits.count >= minimumBuckets else { return nil }
        var learned = explicit ? UInt64.max : fits[fits.count / 10]
        if cuts.count >= 3 { learned = min(learned, cuts[cuts.count / 5]) }
        learned = max(learned, minimum)
        return learned + 250_000_000 < cap ? learned : nil
    }

    public func budget(cap: UInt64, minimum: UInt64, explicit: Bool = false, ceiling: UInt64? = nil, now: Date = Date()) -> UInt64? {
        lock.withLock {
            Self.budget(
                buckets: buckets + (open.map { [$0] } ?? []), trims: trims, cap: cap, minimum: minimum, explicit: explicit,
                ceiling: ceiling, scope: scope, now: now)
        }
    }

    /// The calendar days `trims` fell on. Cuts on three days or more are a Mac that cannot hold a limit; more cuts on
    /// fewer days are one bad evening (other jobs at once), which must not change the next two weeks.
    public static func days(_ trims: [Trim]) -> Int {
        Set(trims.map { Calendar.current.startOfDay(for: Date(timeIntervalSince1970: $0.t)) }).count
    }

    /// The suggestion for a memory_limit of `limit` (its budget at most `cap`) cut on three days or more in 14 days: the
    /// limit this Mac kept (the level that held plus what the limit adds over its budget, down to 0.1 GB), the cuts and
    /// the advice. Origami never changes the setting itself, nor starts below it.
    public static func limitAdvice(
        trims: [Trim], limit: UInt64, cap: UInt64, minimum: UInt64, scope: Scope? = nil, now: Date = Date()
    ) -> [String: Any]? {
        let counted = trims.filter {
            $0.t >= now.timeIntervalSince1970 - window && $0.to < cap && $0.counts(against: limit, explicit: true, scope: scope)
        }
        guard days(counted) >= 3,
            let held = budget(buckets: [], trims: trims, cap: cap, minimum: minimum, explicit: true, ceiling: limit, scope: scope, now: now)
        else { return nil }
        let cuts = counted.count
        let kept = min(limit, held + (limit > cap ? limit - cap : 0)) / 100_000_000
        let size = "\(kept / 10).\(kept % 10)GB"
        return [
            "limit_bytes": limit, "suggested_bytes": kept * 100_000_000, "cuts": cuts,
            "text":
                "memory_limit \(OrigamiPlan.gigabytes(limit)) could not be held on this Mac (cut \(cuts) times); about \(size) is the most it kept: set memory_limit to \(size) (`origami config set memory_limit \(size)`) or remove it to let Origami choose (`origami config unset memory_limit`)",
        ]
    }

    /// The calendar days with cuts below `limit` under it or a lower ceiling since `date` (within the 14 days kept):
    /// three or more make a measured limit stale.
    public func cutDays(below limit: UInt64, since date: Date) -> Int {
        lock.withLock {
            Self.days(
                trims.filter {
                    $0.t >= date.timeIntervalSince1970 && $0.to < limit && $0.counts(against: limit, explicit: true, scope: scope)
                })
        }
    }

    public func limitAdvice(limit: UInt64, cap: UInt64, minimum: UInt64) -> [String: Any]? {
        lock.withLock { Self.limitAdvice(trims: trims, limit: limit, cap: cap, minimum: minimum, scope: scope) }
    }

    /// For /v1/status and the dashboard: the learned budget (nil: none yet), the days it is learned from, the trims in them;
    /// and `limit_advice` (`limitAdvice`) when a memory_limit could not be held.
    public func summary(cap: UInt64, minimum: UInt64, applied: Bool, advice: [String: Any]? = nil) -> [String: Any] {
        let learned = budget(cap: cap, minimum: minimum)
        return lock.withLock {
            let all = buckets + (open.map { [$0] } ?? [])
            let days = all.isEmpty ? 0 : (all.last!.t - all.first!.t + Self.bucketSeconds) / 86_400
            return [
                "bytes": learned.map { $0 as Any } ?? NSNull(), "days": days, "trims": trims.count,
                "regrowths": all.reduce(0) { $0 + $1.grows },
                "applied": applied && learned != nil, "cap_bytes": cap == .max ? NSNull() : cap as Any, "minimum_bytes": minimum,
                "limit_advice": advice ?? NSNull(),
            ]
        }
    }
}

// MARK: origami tune --memory

extension SelfTuning {
    /// One memory_limit as `origami tune --memory` measured it: writing speed (tokens a second) on the first reply after a
    /// long prompt and on a repeat, the first reply's first-token seconds, the expert cache, and what it cost the Mac:
    /// budget cuts and fresh swap. `excluded`: why it cannot be chosen (cut, swapped, not tried), or nil.
    public struct MemoryTrial: Codable, Sendable, Equatable {
        public var limit: UInt64, first = 0.0, warm = 0.0, firstToken = 0.0, cache: UInt64 = 0, cuts = 0, swapped: UInt64 = 0
        public var excluded: String?
        public init(limit: UInt64, excluded: String? = nil) {
            self.limit = limit
            self.excluded = excluded
        }
        public var speed: Double { (first + warm) / 2 }
    }

    /// "25.5GB": a limit as `origami config set memory_limit` takes it, to 0.1 GB.
    public static func memorySize(_ bytes: UInt64) -> String { "\(bytes / 1_000_000_000).\(bytes / 100_000_000 % 10)GB" }

    /// The limits to measure, lowest first: the user's limit, the level that held, the automatic budget, and a step halfway
    /// from it to `ceiling` (the most macOS lets Origami have); a step below automatic (so a lowest limit that is out
    /// still leaves one lower to keep), and a second while that leaves fewer than three.
    /// Each within `ceiling`, to 0.1 GB, at least 0.5 GB from one kept earlier; at most four.
    public static func memoryCandidates(automatic: UInt64, ceiling: UInt64, current: UInt64?, held: UInt64?) -> [UInt64] {
        let gb: UInt64 = 1_000_000_000
        let step = max(gb, ceiling > automatic ? (ceiling - automatic) / 2 : 0)
        var kept: [UInt64] = []
        func add(_ value: UInt64) {
            let value = min(value, ceiling) / 100_000_000 * 100_000_000
            guard value > 0, kept.count < 4, !kept.contains(where: { max($0, value) - min($0, value) < gb / 2 }) else { return }
            kept.append(value)
        }
        [current, held, automatic].compactMap { $0 }.forEach(add)
        if automatic + step <= ceiling { add(automatic + step) }
        for steps: UInt64 in 1...2 where kept.count < (steps == 1 ? 4 : 3) && automatic > steps * step { add(automatic - steps * step) }
        return kept.sorted()
    }

    /// Why a limit is out when the model refused to start at it (too small for its fixed memory and two layers' experts):
    /// that says nothing of higher limits, which are still measured.
    public static let notStarted = "the model did not start at this limit"

    /// Whether the limits above are skipped: one measured was cut for pressure or swapped, and above it other apps would only
    /// be squeezed harder. A limit too small to start at, or one merely slow, does not stop the higher ones.
    public static func memorySkipsHigher(_ trials: [MemoryTrial]) -> Bool {
        trials.contains { trial in trial.excluded.map { !$0.hasPrefix("not tried") && $0 != notStarted } ?? false }
    }

    /// How much slower, in both the first reply and the repeat, a kept limit must be than the next lower kept one to be
    /// passed over for it: one trial's tok/s is noisy, so speed only guards against something being wrong.
    public static let memorySlowerGuard = 0.15

    /// The limits kept (no cuts, no swap, started), lowest first.
    static func keptMemory(_ trials: [MemoryTrial]) -> [MemoryTrial] {
        trials.filter { $0.excluded == nil && $0.speed > 0 }.sorted { $0.limit < $1.limit }
    }

    /// The kept limits the speed guard leaves, lowest first: the highest is passed over for the next lower one while both
    /// its first reply and its repeat are more than `memorySlowerGuard` slower than that one's.
    static func guardedMemory(_ trials: [MemoryTrial]) -> [MemoryTrial] {
        var kept = keptMemory(trials)
        while kept.count > 1, slower(kept[kept.count - 1], than: kept[kept.count - 2]) { kept.removeLast() }
        return kept
    }

    /// The limit chosen: more memory is faster unless the Mac cuts or swaps, and that is measured reliably while one
    /// trial's speed is not, so the highest kept the speed guard leaves (`guardedMemory`, `memoryPassedOver`). More memory
    /// is only faster while it holds more experts: where a lower kept limit held the same expert cache (a Mac that keeps
    /// every expert in memory), the lowest of those is chosen, and their speeds, one trial each, do not decide.
    public static func chosenMemory(_ trials: [MemoryTrial]) -> MemoryTrial? {
        var kept = guardedMemory(trials)
        while kept.count > 1, kept[kept.count - 1].cache > 0, kept[kept.count - 1].cache == kept[kept.count - 2].cache { kept.removeLast() }
        return kept.last
    }

    /// The highest kept limit the guard would pass over, measured once more before that (`origami tune --memory`), or nil.
    public static func memoryPassedOver(_ trials: [MemoryTrial]) -> UInt64? {
        guard let guarded = guardedMemory(trials).last, let top = keptMemory(trials).last, top.limit != guarded.limit else { return nil }
        return top.limit
    }

    static func slower(_ high: MemoryTrial, than low: MemoryTrial) -> Bool {
        high.first < low.first * (1 - memorySlowerGuard) && high.warm < low.warm * (1 - memorySlowerGuard)
    }

    /// One more limit halfway between the highest kept (none kept: the highest the model did not start at) and the next
    /// above it (cut, swapped or not tried), when those are 1.5 GB apart or more; chosen only if it is kept too.
    public static func memoryRefinement(_ trials: [MemoryTrial]) -> UInt64? {
        guard
            let low = keptMemory(trials).last?.limit ?? trials.filter({ $0.excluded == notStarted }).map(\.limit).max(),
            let above = trials.map(\.limit).filter({ $0 > low }).min(), above - low >= 1_500_000_000
        else { return nil }
        return (low + above) / 2 / 100_000_000 * 100_000_000
    }

    /// Why a measured memory limit no longer applies, or nil: another Mac or memory size, another model or context, cut
    /// on three days or more since (`cutDays`, the rule that makes a memory_limit's advice), or older than `recheckAge`.
    public static func memoryStaleness(
        saved: [String: Any], current: [String: Any], savedContext: Int?, context: Int?, cutDays: Int, date: Date, now: Date = Date()
    ) -> String? {
        func text(_ value: Any?) -> String { value.map { "\($0)" } ?? "" }
        for (key, reason) in [("mac", "another Mac"), ("memory_bytes", "another memory size"), ("model", "another model")]
        where current[key] != nil && text(saved[key]) != text(current[key]) {
            return reason
        }
        if let context, savedContext != context { return "another context size" }
        if cutDays >= 3 { return "cut on \(cutDays) days since" }
        if now.timeIntervalSince(date) > recheckAge { return "over \(Int(recheckAge / 86_400)) days old" }
        return nil
    }

    /// The plain-language result: the limit chosen with its repeat's speed, the current limit (or the automatic budget) as
    /// measured, and the command that sets it (`applied`: automatic mode serves with it already).
    public static func memoryText(_ trials: [MemoryTrial], current: UInt64?, automatic: UInt64, applied: Bool = false) -> String {
        func rate(_ trial: MemoryTrial) -> String { String(format: "%.1f tok/s on a repeat", trial.warm) }
        func near(_ a: UInt64, _ b: UInt64) -> Bool { max(a, b) - min(a, b) < 100_000_000 }
        guard let best = chosenMemory(trials) else {
            let lowest = trials.filter { $0.excluded?.hasPrefix("not tried") != true }.map(\.limit).min()
            return "No memory limit measured ran without cuts or swap on this Mac"
                + (lowest.map { ", down to \(memorySize($0))" } ?? "") + "; nothing to suggest"
        }
        let now = current ?? automatic
        let top = keptMemory(trials).last
        var text =
            "Chosen: \(memorySize(best.limit)) — \(rate(best)) ("
            + (best.limit == top?.limit
                ? "highest limit this Mac kept)"
                : best.cache > 0 && top?.cache == best.cache
                    ? "the higher limits kept hold the same experts in memory, so they add nothing)"
                    : "a higher limit kept was over \(Int(memorySlowerGuard * 100))% slower)")
        if let mine = trials.first(where: { near($0.limit, now) }), mine.limit != best.limit {
            text +=
                "; \(current == nil ? "the automatic budget" : "your current limit") \(memorySize(now)) — \(mine.excluded ?? rate(mine))"
        }
        return text
            + (applied
                ? "; automatic mode uses it"
                : current.map { near($0, best.limit) } == true
                    ? "; it is your limit already" : "; set it with `origami config set memory_limit \(memorySize(best.limit))`")
    }
}
