import Foundation

/// The quality dial: one setting from `exact` (the default, bitwise the model's own replies) through `dynamic` and
/// `fast` to a deliberately aggressive `fastest`, which only models it stayed usable on offer (`offered(_:)`). A named
/// level is a set of knobs; a model may run a level with other knobs (`offered(by:)`, DeepSeek's), and users see the
/// name on that model's ladder (`name(_:)`). Each knob is also settable alone (`origami config set
/// experts_per_token 6` and so on). The knobs are plain values so any family backend can read them and apply those it
/// supports; Qwen applies the expert knobs in its fused decode (`SerialExecutor`) and
/// `confidentAbove` between decode steps (`GenerationDriver`).
///
/// The knobs act on decode rows only: a prompt is always read exactly, so saved prompt state is shared by every level,
/// and decode rows of a non-exact level carry their own timeline epoch (`timelineEpoch`), so an exact request never
/// continues from them. Every non-exact level is NUMERIC, and a configuration without the fast allowance
/// (`allow_fast false`) refuses all of them. With `confidentAbove` the dial is dynamic: each decode step uses the knobs
/// only after a token the model was sure of, and decodes exact after one it was not (`step(after:)`).
public struct QualityDial: Codable, Sendable, Equatable {
    /// What a decode layer does with a routed expert whose read has not finished when the layer needs it.
    public enum LateMiss: String, Codable, Sendable, CaseIterable {
        /// Wait for the read (exact).
        case wait
        /// The expert contributes nothing; the other gates are rescaled to the routed mass (v1 fast mode).
        case skip
        /// The expert in memory with the highest router score outside the routed set takes its place, weighted by its
        /// own score; skipped as above when the layer has none in memory.
        case substitute
    }
    /// Limit of routed experts per token, with gates renormalised over the kept ones (nil: all the model routes).
    /// Selection follows the engine: family routing uses probability plus selection bias; keptRanks uses gate mass.
    /// Those orders can differ. Dropped experts are neither read nor waited for.
    public var expertsPerToken: Int?
    /// Routed experts whose share of the routed gate mass is below this are dropped (nil: none). The heaviest is kept.
    public var tailSkip: Double?
    /// The tail skip drops only experts that are not in memory, so it saves reads and keeps every free contribution
    /// (no effect without `tailSkip`).
    public var tailSkipMissesOnly: Bool
    /// What a late routed expert does (`wait` is exact).
    public var lateMiss: LateMiss
    /// Dynamic: the other knobs apply to a decode step only when the token just produced had a top-1 probability of
    /// at least this in the distribution it was sampled from (the model's own confidence, as a proxy for how hard the
    /// next token is); other steps decode exact. nil: every decode step uses the knobs.
    public var confidentAbove: Double?
    /// Experimental same-token delayed contributions. Nil preserves every existing mode and timeline.
    public var delayedExperts: DelayedExpertPolicy?

    public init(
        expertsPerToken: Int? = nil, tailSkip: Double? = nil, tailSkipMissesOnly: Bool = false, lateMiss: LateMiss = .wait,
        confidentAbove: Double? = nil, delayedExperts: DelayedExpertPolicy? = nil
    ) {
        self.expertsPerToken = expertsPerToken
        self.tailSkip = tailSkip
        self.tailSkipMissesOnly = tailSkipMissesOnly
        self.lateMiss = lateMiss
        self.confidentAbove = confidentAbove
        self.delayedExperts = delayedExperts
    }

    public static let exact = QualityDial()
    /// v1's fast mode, unchanged: late misses are skipped. Not a named level; the knob `late_miss skip` alone still
    /// reaches it, and its decode rows keep v1's timeline epoch.
    public static let lateMissSkip = QualityDial(lateMiss: .skip)
    /// Every named knob set, the names settings files and requests use (`offered(by:)` turns one into what a model
    /// runs). The knobs came from a screen of every knob (an M2 Pro with 32 GB, 8K context): the misses-only tail skip
    /// beat experts per token and substitution on speed and agreement at once.
    public static let levels: [(name: String, dial: QualityDial)] = [
        ("exact", .exact),
        // fast's knobs after a token the model gave at least 0.9, exact after any other.
        ("dynamic", QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true, confidentAbove: 0.9)),
        ("fast", QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true)),
        ("fastest", QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true, lateMiss: .skip)),
        // fastest's knobs after a token the model gave at least 0.9, exact after any other. No model lists it: it is
        // what DeepSeek's dynamic runs.
        ("dynamic-fastest", QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true, lateMiss: .skip, confidentAbove: 0.9)),
    ]
    /// The ladder a model offers: exact, dynamic (shortcuts only after a token the model was sure of), fast, and fastest
    /// where an even faster level stayed usable on short checkable tasks (answers scored right or wrong, one Mac).
    /// `dynamic` stayed right everywhere on Qwen and GLM and was slower than `fast` there.
    public static func offered(_ family: ModelFamily) -> [String] {
        switch family {
        // Qwen: fast and fastest saved 62–71% of the time with every task right.
        case .qwen: ["exact", "dynamic", "fast", "fastest"]
        // GLM: fast saved 34–42% with every task right; fastest rambled and wrote reasoning after </think>.
        case .glm, .deepseek: ["exact", "dynamic", "fast"]
        }
    }
    /// The knob set `family` runs for a named level, where it is not the level's own. DeepSeek's levels use stronger
    /// knobs, as the plain ones saved it no time: its dynamic runs dynamic-fastest (about 30% less time, every task
    /// right) and its fast runs fastest (faster still; one task of 14 wrong, reasoning after </think> twice). Every
    /// target here is its own family's fixed point, so `offered(by:)` is idempotent.
    private static func runs(_ family: ModelFamily) -> [String: String] {
        switch family {
        case .qwen: ["dynamic-fastest": "dynamic"]
        case .glm: ["fastest": "fast", "dynamic-fastest": "dynamic"]
        case .deepseek: ["dynamic": "dynamic-fastest", "fast": "fastest"]
        }
    }
    /// Each model's levels besides exact, for help and messages that cannot name the served one.
    public static var offeredByModel: String {
        let models = ModelFamily.allCases.map { "\($0.displayName): " + offered($0).dropFirst().joined(separator: ", ") }
        return "exact or a quicker level the model offers (" + models.joined(separator: "; ") + ")"
    }
    /// Names accepted for a level besides its own (`origami_mode`, the `-<name>` model ids, settings and chat commands):
    /// `faster`, the name of `fast` until 3.1.8.
    public static let aliases = ["faster": "fast"]
    public static func level(_ name: String) -> QualityDial? {
        let name = aliases[name] ?? name
        return levels.first { $0.name == name }?.dial
    }
    /// This dial as `family` runs it. Forgiving: a named level the model does not offer runs its nearest offered one;
    /// exact and custom knobs are unchanged.
    public func offered(by family: ModelFamily) -> QualityDial {
        guard let target = Self.runs(family)[name] else { return self }
        var dial = Self.level(target)!
        dial.delayedExperts = delayedExperts
        return dial
    }
    /// The level's name on `family`'s ladder, as users see it (usage, headers, status, logs): DeepSeek's run of the
    /// fastest knobs is its `fast`. "custom" for knobs set alone.
    public func name(_ family: ModelFamily) -> String {
        let served = offered(by: family).name
        return Self.offered(family).first { Self.level($0)!.offered(by: family).name == served } ?? served
    }

    public var isExact: Bool { self == .exact }
    /// The dial drops routed experts before any read starts (experts per token or tail skip).
    public var dropsExperts: Bool { delayedExperts == nil && (expertsPerToken != nil || tailSkip != nil) }
    /// The knob set's name, as settings files store it, or "custom" for knobs set alone (`name(_:)` is the user's).
    public var name: String {
        var base = self
        base.delayedExperts = nil
        return Self.levels.first { $0.dial == base }?.name ?? "custom"
    }
    /// The knobs as `origami config` and /v1/status show them.
    public var knobs: [String: String] {
        var values = [
            "experts_per_token": expertsPerToken.map(String.init) ?? "all", "tail_skip": tailSkip.map { "\($0)" } ?? "off",
            "tail_skip_misses_only": "\(tailSkipMissesOnly)", "late_miss": lateMiss.rawValue,
            "confident_above": confidentAbove.map { "\($0)" } ?? "off",
        ]
        if let delayedExperts { values["delayed_experts"] = delayedExperts.signature }
        return values
    }
    /// A short stable label of the knobs, for timeline epochs and logs ("k5-t0.05m-skip").
    public var signature: String {
        [
            expertsPerToken.map { "k\($0)" }, tailSkip.map { "t\($0)" + (tailSkipMissesOnly ? "m" : "") },
            lateMiss == .wait ? nil : lateMiss.rawValue, confidentAbove.map { "c\($0)" }, delayedExperts?.signature,
        ].compactMap { $0 }.joined(separator: "-")
    }

    /// The dial of one decode step, given `logits`, the distribution the token just produced was sampled from: the
    /// knobs (without `confidentAbove`) when the model gave its top token at least `confidentAbove`, else exact.
    public func step(after logits: [Float]) -> QualityDial {
        guard let confidentAbove else { return self }
        var knobs = self
        knobs.confidentAbove = nil
        return Self.topProbability(logits) >= confidentAbove ? knobs : .exact
    }
    /// Tool calls decode exact: once a reply opens one, the rest of it decodes exact at every level, so a call's name
    /// and arguments are the model's own. Prose before it, in a request with tools too, keeps the level's speed; the
    /// opener itself comes from the step before it, which may have used the knobs. `token`: the token just produced;
    /// `text`: the reply so far; `openers`: the tokenizer's single-token openers (`TokenizerStack.toolCallOpeners`), one
    /// integer compare a step. A vocabulary with none of them as one token is checked on the reply's tail instead.
    public static func opensToolCall(_ token: Int, text: String, openers: Set<Int>) -> Bool {
        guard openers.isEmpty else { return openers.contains(token) }
        // shortcut: a token longer than 64 characters after the marker hides it; no served tokenizer takes this path.
        let tail = text.suffix(64)
        return TokenizerStack.toolCallMarkers.contains { tail.contains($0) }
    }
    /// The largest probability of softmax(logits) (0 when there is no finite logit).
    public static func topProbability(_ logits: [Float]) -> Double {
        guard let top = logits.max(), top.isFinite else { return 0 }
        var sum = 0.0
        for logit in logits { sum += Double(expf(logit - top)) }
        return 1 / sum
    }

    /// Refuses knob values no family can honour, and writes knobs that change nothing as their defaults (all routed
    /// experts per token; no misses-only without a tail skip), so such a dial is exact.
    public func validated(routedExperts: Int = 10) throws -> QualityDial {
        if let expertsPerToken, !(1...routedExperts).contains(expertsPerToken) {
            throw EngineError("experts per token \(expertsPerToken) is not in 1...\(routedExperts)", code: "invalid_configuration")
        }
        if let tailSkip, !(tailSkip > 0 && tailSkip < 1) {
            throw EngineError("tail skip \(tailSkip) is not a share of the gate mass in (0, 1)", code: "invalid_configuration")
        }
        if let confidentAbove, !(confidentAbove > 0 && confidentAbove < 1) {
            throw EngineError("confident above \(confidentAbove) is not a probability in (0, 1)", code: "invalid_configuration")
        }
        var dial = self
        if let delayedExperts { dial.delayedExperts = try delayedExperts.validated() }
        if expertsPerToken == routedExperts { dial.expertsPerToken = nil }
        if tailSkip == nil { dial.tailSkipMissesOnly = false }
        if dial.delayedExperts != nil {
            var base = dial
            base.delayedExperts = nil
            base.confidentAbove = nil
            guard !base.isExact else {
                throw EngineError("Delayed contributions require an approximate quality mode", code: "invalid_configuration")
            }
        }
        // A threshold over knobs that change nothing is exact too.
        var knobs = dial
        knobs.confidentAbove = nil
        if knobs.isExact { dial.confidentAbove = nil }
        return dial
    }

    /// The routed ranks one token keeps, ascending, from its routed gates (rank order, any positive scale).
    /// `resident(rank)`: that rank's expert is in memory (asked only for a misses-only tail skip).
    public func keptRanks(_ weights: [Float], resident: (Int) -> Bool = { _ in false }) -> [Int] {
        if delayedExperts != nil { return Array(weights.indices) }
        let mass = weights.reduce(Float(0), +)
        let order = weights.indices.sorted { weights[$0] > weights[$1] || (weights[$0] == weights[$1] && $0 < $1) }
        let top = Set(order.prefix(max(1, expertsPerToken ?? weights.count)))
        return weights.indices.filter { rank in
            guard top.contains(rank) else { return false }
            guard let tailSkip, rank != order.first else { return true }
            return Double(weights[rank] / mass) >= tailSkip || (tailSkipMissesOnly && resident(rank))
        }
    }

    /// The gates of the routed ranks: `active` weights (0 where an expert is dropped) scaled so they carry the routed
    /// mass of `weights` again, as the trained top-k normalisation does over the experts that remain.
    public static func gates(_ weights: [Float], active: [Float]) -> [Float] {
        let mass = weights.reduce(Float(0), +)
        let remaining = active.reduce(Float(0), +)
        return active.map { $0 * (mass / remaining) }
    }
    /// `weights` with only the `kept` ranks, renormalised.
    public static func gates(_ weights: [Float], kept: [Int]) -> [Float] {
        gates(weights, active: weights.indices.map { kept.contains($0) ? weights[$0] : 0 })
    }

    /// Late-miss substitutes: for each late rank (heaviest first), the expert in memory outside `routed` with the
    /// highest router logit, each used once. Returns rank → expert; ranks without a candidate are absent.
    public static func substitutes(late ranks: [Int], weights: [Float], routed: [Int], logits: [Float], resident: [Int]) -> [Int: Int] {
        var candidates = resident.filter { !routed.contains($0) && logits.indices.contains($0) }
            .sorted { logits[$0] > logits[$1] || (logits[$0] == logits[$1] && $0 < $1) }
        var chosen: [Int: Int] = [:]
        for rank in ranks.sorted(by: { weights[$0] > weights[$1] || (weights[$0] == weights[$1] && $0 < $1) }) where !candidates.isEmpty {
            chosen[rank] = candidates.removeFirst()
        }
        return chosen
    }
    /// A substitute's gate weight on the scale of the routed ones: the router's softmax weight it would have had beside
    /// the replaced expert (whose weight is `weight`): weight * exp(substitute logit - replaced logit).
    public static func substituteWeight(_ weight: Float, replacedLogit: Float, substituteLogit: Float) -> Float {
        weight * expf(substituteLogit - replacedLogit)
    }
}
