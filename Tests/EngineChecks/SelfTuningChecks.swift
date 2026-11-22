import Foundation
import OrigamiEngine

private func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }

/// The self-tuner's decision over rotated rounds: the control's gap sets the gain needed, a candidate must clear it in
/// every round with the base's replies, the single best wins, and noise or drift changes nothing.
func selfTuningDecisionRule() throws {
    func sample(_ decode: Double, prompt: Double = 300, replies: [String] = ["a", "b"]) -> SelfTuning.Sample {
        SelfTuning.Sample(decode: decode, prompt: prompt, replies: replies)
    }
    func rounds(_ control: [Double], _ arms: [String: [Double]], replies: [String: [String]] = [:], prompt: [String: Double] = [:])
        -> [SelfTuning.Round]
    {
        control.indices.map { i in
            var round: SelfTuning.Round = ["base": sample(10), "control": sample(10 * control[i])]
            for (arm, ratios) in arms {
                round[arm] = sample(10 * ratios[i], prompt: 300 * (prompt[arm] ?? 1), replies: replies[arm] ?? ["a", "b"])
            }
            return round
        }
    }
    // Quiet rounds: 1% noise asks for 3%; +10% in every round wins over +5%.
    var d = SelfTuning.decide(rounds([1.01, 0.99, 1.0], ["a": [1.10, 1.11, 1.09], "b": [1.05, 1.06, 1.05]]), candidates: ["a", "b"])
    try require(
        d.outcome == "faster" && d.winner == "a" && abs(d.threshold - 1.03) < 1e-9 && abs(d.gain - 0.10) < 1e-9, "best arm wins: \(d)")
    // Faster on the median but not in one round: not accepted.
    d = SelfTuning.decide(rounds([1, 1, 1], ["a": [1.20, 1.01, 1.20]]), candidates: ["a"])
    try require(d.outcome == "already_fastest" && d.winner == nil, "a round it did not win: \(d)")
    // 2.5% noise asks for 5%: +4% everywhere is not enough.
    d = SelfTuning.decide(rounds([1.025, 0.975, 1.025], ["a": [1.04, 1.04, 1.04]]), candidates: ["a"])
    try require(d.winner == nil && abs(d.threshold - 1.05) < 1e-9, "twice the noise: \(d)")
    // Severe control variation: too noisy, nothing chosen.
    d = SelfTuning.decide(rounds([1.20, 0.80, 1.20], ["a": [1.5, 1.5, 1.5]]), candidates: ["a"])
    try require(d.outcome == "too_noisy" && d.winner == nil, "noise veto: \(d)")
    d = SelfTuning.decide(rounds([1.05, 0.95, 1.05], ["a": [1.20, 1.20, 1.20]]), candidates: ["a"])
    try require(d.winner == "a" && d.threshold >= 1.10, "paired improvement can exceed moderate noise")
    // Different replies from a steady base: refused and reported.
    d = SelfTuning.decide(rounds([1, 1, 1], ["a": [1.2, 1.2, 1.2]], replies: ["a": ["a", "c"]]), candidates: ["a"])
    try require(d.winner == nil && d.notes.contains { $0.contains("replies differed") }, "identical replies: \(d)")
    // A base whose replies differ cannot establish unchanged output; no setting is accepted.
    var shaky = rounds([1, 1, 1], ["a": [1.2, 1.2, 1.2]], replies: ["a": ["x", "y"]])
    shaky[1]["base"]!.replies = ["z"]
    d = SelfTuning.decide(shaky, candidates: ["a"])
    try require(d.winner == nil && d.outcome == "too_noisy", "unsteady base: \(d)")
    // Writing faster but reading more than 2% slower: not accepted.
    d = SelfTuning.decide(rounds([1, 1, 1], ["a": [1.2, 1.2, 1.2]], prompt: ["a": 0.95]), candidates: ["a"])
    try require(d.winner == nil, "prompt guard: \(d)")
    // A reading stall in one round of the candidate: judged on each arm's best round, so it still wins.
    var slowRead = rounds([1, 1, 1], ["a": [1.2, 1.2, 1.2]])
    slowRead[1]["a"]!.prompt = 300 * 0.7
    d = SelfTuning.decide(slowRead, candidates: ["a"])
    try require(d.winner == "a", "prompt guard on best rounds: \(d)")
    // Stalls: an 8.75 s gap among 0.25 s ones is left out beyond 4x the median; a reply slower on every token keeps its
    // time; fewer than 8 gaps are not judged.
    let steady = [Double](repeating: 0.25, count: 31)
    var stalled = steady
    stalled[12] = 8.75
    try require(SelfTuning.stallSeconds(gaps: steady) == 0 && abs(SelfTuning.stallSeconds(gaps: stalled) - 7.75) < 1e-9, "stall left out")
    try require(
        SelfTuning.stallSeconds(gaps: steady.map { $0 * 2 }) == 0 && SelfTuning.stallSeconds(gaps: [9, 0.25, 0.25]) == 0, "slow tokens stay"
    )
    try require(SelfTuning.settled(4.0, 4.15) && !SelfTuning.settled(4.0, 3.7) && !SelfTuning.settled(0, 1), "warm-up settles within 5%")
    // A base drifting 6% over the run, more than half a 10% gain: too noisy.
    var drifting = rounds([1, 1, 1], ["a": [1.1, 1.1, 1.1]])
    for i in drifting.indices {
        let f = 1 + 0.03 * Double(i)
        for k in drifting[i].keys { drifting[i][k]!.decode *= f }
    }
    d = SelfTuning.decide(drifting, candidates: ["a"])
    try require(d.outcome == "too_noisy", "drift veto: \(d)")
    // The confirm reuses the first threshold.
    d = SelfTuning.decide(rounds([1.0, 1.0], ["a": [1.04, 1.04]]), candidates: ["a"], threshold: 1.05)
    try require(d.winner == nil, "confirm keeps the threshold")
    // Early drop: more than 5% slower than the round's base.
    try require(SelfTuning.dropped(after: rounds([1], ["slow": [0.9], "ok": [0.97]])[0]) == ["slow"], "early drop")
    // Guess depth: the deepest rank used at least half the time.
    try require(SelfTuning.guessDepth(issued: [100, 80, 60, 40], used: [90, 50, 20, 5]) == 2, "depth 2")
    try require(
        SelfTuning.guessDepth(issued: [100, 80], used: [90, 60]) == nil && SelfTuning.guessDepth(issued: [0, 0], used: [0, 0]) == nil,
        "no depth")
    // A faster prefill group can win with unchanged decode; decode regressions and changed long-prompt replies veto it.
    let promptRounds = rounds([1, 1, 1], ["group": [1, 1, 1]], prompt: ["group": 1.2])
    d = SelfTuning.decide(promptRounds, candidates: ["group"], objective: .prompt)
    try require(d.winner == "group" && d.objective == .prompt && abs(d.gain - 0.2) < 1e-9, "prompt objective wins independently")
    try require(SelfTuning.decide(promptRounds, candidates: ["group"]).winner == nil, "decode objective unchanged")
    d = SelfTuning.decide(
        rounds([1, 1, 1], ["group": [0.9, 0.9, 0.9]], prompt: ["group": 1.2]), candidates: ["group"], objective: .prompt)
    try require(d.winner == nil && d.notes.contains { $0.contains("writing") }, "prompt winner must preserve decode")
    d = SelfTuning.decide(
        rounds([1, 1, 1], ["group": [1, 1, 1]], replies: ["group": ["changed"]], prompt: ["group": 1.2]), candidates: ["group"],
        objective: .prompt)
    try require(d.winner == nil, "prompt winner must preserve tested replies")
    var noisyPrefill = promptRounds
    for i in noisyPrefill.indices { noisyPrefill[i]["control"]!.prompt *= 1.2 }
    try require(
        SelfTuning.decide(noisyPrefill, candidates: ["group"], objective: .prompt).outcome == "too_noisy", "prompt control determines noise"
    )
    // The search never asks for more working memory or a different compute tile.
    try require(SelfTuning.prefillCandidates(current: 8192, maximum: 8192, step: 256) == [4096], "Qwen bounded neighbor")
    try require(SelfTuning.prefillCandidates(current: 2, maximum: 3, step: 1) == [1, 3], "family capacity bound")
    try require(SelfTuning.prefillCandidates(current: 1, maximum: 1, step: 1).isEmpty, "no extra family allocation")
    for current in stride(from: 256, through: 8192, by: 256) {
        let candidates = SelfTuning.prefillCandidates(current: current, maximum: 8192, step: 256)
        try require(
            candidates.count <= 2 && candidates.allSatisfy { (256...8192).contains($0) && $0 % 256 == 0 && $0 != current },
            "candidate geometry")
    }
}

/// Saved results: the user's setting beats a tuned value, which beats the default; another Mac, GPU, memory, model,
/// drives or tuning version are not applied; a macOS update or 90 days only ask for a re-check. The request field takes
/// only its allow-list, on exact requests.
/// The time plan: everything fits as it is; a default limit grows to fit (never past its maximum); an owner's limit keeps
/// the candidates that fit whole, at least one.
func selfTuningTimePlan() throws {
    func fit(_ needs: [Int], left: Double, more: Double) -> (count: Int, extra: Double) {
        SelfTuning.fit(needs: needs, confirm: 4, sampleSeconds: 10, left: left, more: more)
    }
    // Two candidates of 9 samples, a confirmation of 4 and a round of margin: 25 samples, 250 s.
    var plan = fit([9, 9], left: 300, more: 0)
    try require(plan.count == 2 && plan.extra == 0, "fits as it is: \(plan)")
    plan = fit([9, 9], left: 100, more: 600)
    try require(plan.count == 2 && plan.extra == 150, "default limit grows: \(plan)")
    plan = fit([9, 9], left: 100, more: 100)
    try require(plan.count == 1 && plan.extra == 60, "grows to its maximum, then fewer candidates: \(plan)")
    plan = fit([9, 9, 9], left: 200, more: 0)
    try require(plan.count == 1 && plan.extra == 0, "an owner's limit keeps whole candidates only: \(plan)")
    plan = fit([9, 9], left: 30, more: 0)
    try require(plan.count == 1 && plan.extra == 0, "at least one candidate, measured and kept: \(plan)")
    plan = fit([0, 3, 9], left: 120, more: 0)
    try require(plan.count == 2, "kept measurements need less time: \(plan)")
}

func selfTuningStalenessAndPrecedence() throws {
    try require(SelfTuning.resolve(explicit: false, tuned: true, fallback: true) == (false, "setting"), "setting wins")
    try require(SelfTuning.resolve(explicit: nil, tuned: false, fallback: true) == (false, "tuned"), "tuned beats default")
    try require(SelfTuning.resolve(explicit: nil as Bool?, tuned: nil, fallback: true) == (true, "default"), "default")
    let identity: [String: Any] = [
        "tuning_version": SelfTuning.version, "mac": "Mac14,12", "gpu": "Apple M2 Pro", "memory_bytes": 34_359_738_368,
        "model": "abc", "drives": "U1,U2", "macos": 26,
    ]
    let now = Date()
    try require(SelfTuning.staleness(saved: identity, current: identity, date: now, now: now).state == "current", "same")
    for (key, value) in [
        ("tuning_version", SelfTuning.version + 1), ("mac", "Mac15,1"), ("memory_bytes", 1), ("model", "other"), ("drives", "U1"),
    ] as [(String, Any)] {
        var changed = identity
        changed[key] = value
        try require(SelfTuning.staleness(saved: identity, current: changed, date: now, now: now).state == "stale", "\(key) makes it stale")
    }
    var updated = identity
    updated["macos"] = 27
    try require(SelfTuning.staleness(saved: identity, current: updated, date: now, now: now) == ("recheck", "macOS was updated"), "macOS")
    try require(
        SelfTuning.staleness(saved: identity, current: identity, date: now.addingTimeInterval(-91 * 86_400), now: now).state == "recheck",
        "age")
    // The request field.
    let all = try TuneOverrides(json: [
        "gpu_keepalive": false, "family_prefetch": true, "family_prefetch_depth": "all", "family_spin": false,
        "qwen_prefill_scope": 4096, "family_prefill_slices": 2, "family_arriving_first": true, "family_engram_overlap": true,
        "qwen_arriving_first": true, "expert_admission": "margin", "family_early_delayed_reads": true,
    ])
    try require(
        all
            == TuneOverrides(
                gpuKeepAlive: false, familyPrefetch: true, familyPrefetchDepth: -1, familySpin: false,
                qwenPrefillScope: 4096, familyPrefillSlices: 2,
                familyEarlyDelayedReads: true), "parsed")
    try require(
        (try? TuneOverrides(json: ["io_queue_depth": 8])) == nil && (try? TuneOverrides(json: ["family_prefetch_depth": 65])) == nil
            && (try? TuneOverrides(json: ["gpu_keepalive": 1])) == nil
            && (try? TuneOverrides(json: ["family_spin": 1])) == nil, "allow-list and types")
    for key in TuneOverrides.retiredKeys {
        try require(try TuneOverrides(json: [key: "any"]) == TuneOverrides(), "retired \(key) was not ignored")
    }
    for bad in [0, 255, 257, 8193, true, 256.5, "4096"] as [Any] {
        try require((try? TuneOverrides(json: ["qwen_prefill_scope": bad])) == nil, "invalid prefill scope \(bad)")
    }
    for bad in [0, 5, true, 1.5, "2"] as [Any] {
        try require((try? TuneOverrides(json: ["family_prefill_slices": bad])) == nil, "invalid family slices \(bad)")
    }
    try require(try TuneOverrides(json: all.wire) == all, "wire round trip")
    func parse(_ extra: [String: Any]) throws -> ServingRequest {
        try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: ["messages": [["role": "user", "content": "hi"]]].merging(extra) { $1 }),
            contextTier: 8192)
    }
    try require(
        try parse(["origami_tune": ["gpu_keepalive": false]]).tune?.gpuKeepAlive == false && (try parse([:])).tune == nil, "request field")
    var refused = false
    do { _ = try parse(["origami_tune": ["gpu_keepalive": false], "origami_mode": "faster"]) } catch let failure as ServingFailure {
        refused = failure.status == 400
    }
    try require(refused, "origami_tune needs an exact request")
    try require(
        try parse(["origami_tune": [String: Any](), "origami_mode": "faster"]).tune != nil,
        "tradeoff benchmark may hold the cache without exact-only overrides")
    // The front's remote refusal sees the key however it is spelled, and only as the key the server reads.
    let escaped = Data(#"{"messages":[{"role":"user","content":"hi"}],"origami\u005ftune":{"gpu_keepalive":false}}"#.utf8)
    try require(try ServingRequest.parse(escaped, contextTier: 8192).tune != nil && TuneOverrides.requested(in: escaped), "escaped key")
    try require(
        TuneOverrides.requested(in: Data(#"{"origami_tune":{}}"#.utf8))
            && !TuneOverrides.requested(in: Data(#"{"messages":[{"role":"user","content":"\"origami_tune\" \u00e9"}]}"#.utf8)), "key only")
}

/// The learned memory budget from a synthetic history: a Mac whose other apps leave about 18 GB most of the time and
/// trim a 24 GB budget about once per reply starts at what four in five trims cut to, never above today's ceiling or
/// below the least the model runs with; the file is per Mac and a reset forgets it.
func memoryBudgetLearnedFromHistory() throws {
    let gb: UInt64 = 1_000_000_000
    let now = Date()
    // Two days of 10-minute buckets: fits mostly 20 GB, one in ten dips to 18 GB.
    let buckets = (0..<288).map {
        MemoryHistory.Bucket(t: now.timeIntervalSince1970 - Double(288 - $0) * 600, fit: $0 % 10 == 0 ? 18 * gb : 20 * gb)
    }
    // 200 trims, cutting to 17 to 21 GB.
    let trims = (0..<200).map { MemoryHistory.Trim(t: now.timeIntervalSince1970 - Double($0) * 800, to: (17 + UInt64($0 % 5)) * gb) }
    let learned = MemoryHistory.budget(buckets: buckets, trims: trims, cap: 24 * gb, minimum: 6 * gb, now: now)
    try require(learned == 18 * gb, "learned \(String(describing: learned))")
    // Of the trims, those that cut to the learned budget or more would not have happened: most of them.
    try require(trims.filter { $0.to >= learned! }.count * 5 >= trims.count * 4, "avoids most trims")
    // Without trims, the budget fitting nine in ten 10-minute windows.
    try require(MemoryHistory.budget(buckets: buckets, trims: [], cap: 24 * gb, minimum: 6 * gb, now: now) == 18 * gb, "buckets alone")
    // Never above the ceiling (nothing to lower), never below the minimum, nothing from under an hour, nothing older than 14 days.
    try require(MemoryHistory.budget(buckets: buckets, trims: trims, cap: 18 * gb, minimum: 6 * gb, now: now) == nil, "at the ceiling")
    try require(MemoryHistory.budget(buckets: buckets, trims: trims, cap: 24 * gb, minimum: 19 * gb, now: now) == 19 * gb, "minimum")
    try require(
        MemoryHistory.budget(buckets: Array(buckets.prefix(5)), trims: trims, cap: 24 * gb, minimum: 0, now: now) == nil,
        "too little history")
    try require(
        MemoryHistory.budget(buckets: buckets, trims: trims, cap: 24 * gb, minimum: 0, now: now.addingTimeInterval(15 * 86_400)) == nil,
        "window")
    // The recorder: checks fold into 10-minute buckets written as each closes; another Mac's file starts over; reset forgets.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-memory-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("memory-history.json")
    let history = MemoryHistory(file: file, identity: "Mac14,12 32")
    let start = (now.timeIntervalSince1970 / 600).rounded(.down) * 600 - 7 * 600
    for i in 0..<7 {
        history.observe(fit: 20 * gb, now: Date(timeIntervalSince1970: start + Double(i) * 600))
        history.observe(fit: 19 * gb, now: Date(timeIntervalSince1970: start + Double(i) * 600 + 300))
        history.trimmed(to: 18 * gb, limit: 24 * gb, now: Date(timeIntervalSince1970: start + Double(i) * 600 + 400))
    }
    history.observe(fit: 20 * gb, now: Date(timeIntervalSince1970: start + 7 * 600))
    let reread = MemoryHistory(file: file, identity: "Mac14,12 32")
    try require(reread.budget(cap: 24 * gb, minimum: 0, now: now) == 18 * gb, "written and read back")
    try require(
        MemoryHistory(file: file, identity: "Mac15,1 64").budget(cap: 24 * gb, minimum: 0, now: now) == nil, "another Mac starts over")
    let summary = reread.summary(cap: 24 * gb, minimum: 0, applied: true)
    try require(
        summary["trims"] as? Int == 7 && summary["applied"] as? Bool == true && (summary["days"] as? Double ?? 0) > 0, "summary \(summary)")
    reread.reset()
    try require(!FileManager.default.fileExists(atPath: file.path) && reread.budget(cap: 24 * gb, minimum: 0, now: now) == nil, "reset")
}

/// `origami tune --memory`: candidates from the automatic budget, the user's limit and the level that held, within the
/// ceiling; a cut or swapped limit is never chosen; the highest kept wins unless both its speeds are over 15% below the
/// next lower kept one's; one refinement halfway up; plain text.
func selfTuningMemoryCalibration() throws {
    let automatic: UInt64 = 24_051_816_858
    let ceiling: UInt64 = 30_064_771_072
    let gb: UInt64 = 1_000_000_000
    func limits(_ values: [UInt64]) -> String { values.map(SelfTuning.memorySize).joined(separator: ",") }
    let set = SelfTuning.memoryCandidates(automatic: automatic, ceiling: ceiling, current: ceiling, held: nil)
    try require(limits(set) == "21.0GB,24.0GB,27.0GB,30.0GB", "with a limit: \(limits(set))")
    let auto = SelfTuning.memoryCandidates(automatic: automatic, ceiling: ceiling, current: nil, held: nil)
    try require(limits(auto) == "21.0GB,24.0GB,27.0GB", "automatic: \(limits(auto))")
    let all = SelfTuning.memoryCandidates(automatic: automatic, ceiling: ceiling, current: 31 * gb, held: 22_000_000_000)
    try require(limits(all) == "22.0GB,24.0GB,27.0GB,30.0GB", "held and a limit over the ceiling: \(limits(all))")
    let tight = SelfTuning.memoryCandidates(automatic: automatic, ceiling: automatic, current: nil, held: 23_800_000_000)
    try require(limits(tight) == "22.0GB,23.0GB,23.8GB", "no room above, close values merged: \(limits(tight))")

    func trial(_ limit: UInt64, _ first: Double, _ warm: Double, _ excluded: String? = nil) -> SelfTuning.MemoryTrial {
        var trial = SelfTuning.MemoryTrial(limit: limit, excluded: excluded)
        (trial.first, trial.warm) = (first, warm)
        return trial
    }
    // The highest kept wins, not the fastest: a cut limit is out, the refinement halfway up counts only if kept too.
    var trials = [trial(24 * gb, 9.5, 9.7), trial(27 * gb, 12, 12, "cut 2 times"), trial(30 * gb, 0, 0, "not tried")]
    try require(SelfTuning.chosenMemory(trials)?.limit == 24 * gb, "a cut limit chosen")
    try require(
        SelfTuning.memoryRefinement(trials) == 25_500_000_000, "refinement \(String(describing: SelfTuning.memoryRefinement(trials)))")
    try require(
        SelfTuning.chosenMemory(trials + [trial(25_500_000_000, 9.8, 10, "700 MB swapped")])?.limit == 24 * gb,
        "a refinement not kept chosen")
    trials.append(trial(25_500_000_000, 9.0, 9.2))
    try require(SelfTuning.chosenMemory(trials)?.limit == 25_500_000_000, "a kept refinement, a little slower, not chosen")
    trials[3] = trial(25_500_000_000, 9.8, 10)
    let text = SelfTuning.memoryText(trials, current: 30 * gb, automatic: automatic)
    try require(
        text
            == "Chosen: 25.5GB — 10.0 tok/s on a repeat (highest limit this Mac kept); your current limit 30.0GB — not tried; set it with `origami config set memory_limit 25.5GB`",
        text)
    try require(
        SelfTuning.memoryText(trials, current: nil, automatic: automatic).contains(
            "; the automatic budget 24.0GB — 9.7 tok/s on a repeat; set it"), "automatic text")
    try require(
        SelfTuning.memoryText(trials, current: 25_500_000_000, automatic: automatic).hasSuffix("; it is your limit already"), "already set")
    // A noisy middle trial neither wins nor loses anything (a run on a 32 GB Mac: 21.0 GB 8.5, 22.5 GB 5.0, 24.0 GB 9.0).
    let noisy = [
        trial(21 * gb, 8.55, 8.51), trial(22_500_000_000, 4.96, 5.82), trial(24 * gb, 9.01, 6.0), trial(27 * gb, 4.7, 8.5, "cut 1 time"),
    ]
    try require(
        SelfTuning.chosenMemory(noisy)?.limit == 24 * gb && SelfTuning.memoryPassedOver(noisy) == nil
            && SelfTuning.chosenMemory(noisy.filter { $0.limit != 22_500_000_000 })?.limit == 24 * gb, "a noisy middle trial decided")
    // The guard: passed over only when both speeds are over 15% below the next lower kept limit's, after one more try.
    let slow = [trial(21 * gb, 8.5, 8.5), trial(24 * gb, 6.0, 7.0)]
    try require(
        SelfTuning.memoryPassedOver(slow) == 24 * gb && SelfTuning.chosenMemory(slow)?.limit == 21 * gb
            && SelfTuning.memoryText(slow, current: nil, automatic: automatic).contains("(a higher limit kept was over 15% slower)"),
        "both speeds over 15% slower kept")
    try require(
        SelfTuning.chosenMemory([trial(21 * gb, 8.5, 8.5), trial(24 * gb, 6.0, 8.0)])?.limit == 24 * gb,
        "one speed alone passed it over")
    let retried = [trial(21 * gb, 8.5, 8.5), trial(24 * gb, 8.0, 8.1)]
    try require(
        SelfTuning.chosenMemory(retried)?.limit == 24 * gb && SelfTuning.memoryPassedOver(retried) == nil, "a better second try lost")
    try require(
        SelfTuning.chosenMemory([trial(24 * gb, 9, 9, "800 MB swapped")]) == nil
            && SelfTuning.memoryText([], current: nil, automatic: automatic).hasPrefix("No memory limit"), "nothing kept")

    // A Mac that keeps every expert in memory (a 128 GB one: 112.4, 113.4 and 114.4 GB all held 67.9 GB of experts and
    // wrote 31.6, 29.6 and 31.2 tok/s): the lowest of the limits with the same cache, said as such, and nothing measured
    // again; a limit whose cache is larger still wins, and trials without a cache size decide as before.
    func held(_ limit: UInt64, _ warm: Double, _ cache: UInt64) -> SelfTuning.MemoryTrial {
        var one = trial(limit, warm - 6, warm)
        one.cache = cache
        return one
    }
    let full = [
        held(112_400_000_000, 31.6, 67_947_724_800), held(113_400_000_000, 29.6, 67_947_724_800),
        held(114_400_000_000, 31.2, 67_947_724_800),
    ]
    let fullText = SelfTuning.memoryText(full, current: nil, automatic: 113_400_000_000, applied: true)
    try require(
        SelfTuning.chosenMemory(full)?.limit == 112_400_000_000 && SelfTuning.memoryPassedOver(full) == nil
            && fullText
                == "Chosen: 112.4GB — 31.6 tok/s on a repeat (the higher limits kept hold the same experts in memory, so they add nothing); the automatic budget 113.4GB — 29.6 tok/s on a repeat; automatic mode uses it",
        "every expert in memory: \(fullText)")
    // Two kept limits with the same cache end the search: nothing higher, and nothing between, is measured.
    let skipped = Array(full.prefix(2)) + [trial(116 * gb, 0, 0, "not tried: the lower limits hold every expert in memory")]
    try require(
        SelfTuning.memoryHoldsAll(skipped) && !SelfTuning.memoryHoldsAll(Array(full.prefix(1)))
            && SelfTuning.memoryRefinement(skipped) == nil
            && SelfTuning.chosenMemory(skipped)?.limit == 112_400_000_000
            && !SelfTuning.memoryHoldsAll([held(22 * gb, 10.2, 15 * gb), held(24 * gb, 10.4, 17 * gb)]),
        "every expert in memory: higher limits not skipped")
    try require(
        SelfTuning.chosenMemory([held(22 * gb, 10.2, 15 * gb), held(24 * gb, 10.4, 17 * gb), held(25 * gb, 10.3, 17 * gb)])?.limit == 24
            * gb
            && SelfTuning.chosenMemory([held(22 * gb, 10.2, 15 * gb), held(24 * gb, 10.4, 17 * gb)])?.limit == 24 * gb,
        "the cache rule passed over a limit that held more")

    // Higher limits are skipped only after one cut for pressure or swapped; not after one merely slow or too small to start.
    try require(
        !SelfTuning.memorySkipsHigher([trial(20 * gb, 1.9, 1.8), trial(21 * gb, 0, 0, SelfTuning.notStarted)])
            && SelfTuning.memorySkipsHigher([trial(20 * gb, 1.9, 1.8, "cut 1 time")])
            && SelfTuning.memorySkipsHigher([trial(20 * gb, 1.9, 1.8, "800 MB swapped")]), "skip rule")
    let unstarted = [trial(20 * gb, 0, 0, SelfTuning.notStarted), trial(24 * gb, 6, 6), trial(27 * gb, 7, 7)]
    try require(SelfTuning.chosenMemory(unstarted)?.limit == 27 * gb, "a limit that did not start chosen")
    let squeezed = [
        trial(20 * gb, 0, 0, SelfTuning.notStarted), trial(24 * gb, 1.3, 1.3, "682 MB swapped"), trial(27 * gb, 0, 0, "not tried"),
    ]
    try require(
        SelfTuning.memoryRefinement(squeezed) == 22 * gb,
        "between too small and swapped: \(String(describing: SelfTuning.memoryRefinement(squeezed)))")

    // A cut counts only for pressure: the process overrunning its own limit is the limit working (every model's governor).
    var policy = MemoryGovernorPolicy(mode: .explicit, cap: 20 * gb, floor: 4 * gb, minimum: 8 * gb, swapouts: 0, hardCeiling: 20 * gb)
    _ = policy.step(now: 0, available: 8 * gb, reserved: 19 * gb, usable: 19 * gb, swapouts: 0, pressure: .normal)
    let own = policy.step(
        now: 1, available: 8 * gb, reserved: 19 * gb, usable: 19 * gb, swapouts: 0, pressure: .normal, footprint: 20_500_000_000)
    try require(own?.reason == .footprint && own?.reason.isPressure == false, "own ceiling: \(String(describing: own))")
    let swap = policy.step(now: 10, available: 8 * gb, reserved: 18 * gb, usable: 18 * gb, swapouts: 200 << 20, pressure: .normal)
    let warning = policy.step(now: 20, available: 8 * gb, reserved: 16 * gb, usable: 16 * gb, swapouts: 200 << 20, pressure: .warning)
    try require(
        swap?.reason == .swap && swap?.reason.isPressure == true && warning?.reason == .warning && warning?.reason.isPressure == true,
        "pressure cuts: \(String(describing: swap)) \(String(describing: warning))")
}

/// When a measured memory limit stops applying: another Mac, memory size, model or context; cut on three calendar days
/// or more since it was measured (only cuts below it, and only after it); older than `recheckAge`. Applied, its text
/// says so.
func selfTuningMemoryStaleness() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError("memory staleness: " + message) } }
    let gb: UInt64 = 1_000_000_000
    let now = Date()
    let saved: [String: Any] = ["mac": "Mac14,12", "memory_bytes": 34_359_738_368, "model": "abc"]
    func stale(
        _ current: [String: Any] = ["mac": "Mac14,12", "memory_bytes": 34_359_738_368], context: Int = 8192, cutDays: Int = 0,
        age: Double = 0
    )
        -> String?
    {
        SelfTuning.memoryStaleness(
            saved: saved, current: current, savedContext: 8192, context: context, cutDays: cutDays, date: now.addingTimeInterval(-age),
            now: now)
    }
    try require(stale() == nil && stale(cutDays: 2) == nil && stale(age: 89 * 86_400) == nil, "a current result was stale")
    try require(stale(["mac": "Mac15,1", "memory_bytes": 34_359_738_368]) == "another Mac", "another Mac")
    try require(stale(["mac": "Mac14,12", "memory_bytes": 68_719_476_736]) == "another memory size", "another memory size")
    try require(stale(["mac": "Mac14,12", "model": "def"]) == "another model", "another model")
    try require(stale(context: 32768) == "another context size", "another context")
    try require(stale(cutDays: 3) == "cut on 3 days since", "cuts on three days")
    try require(stale(age: 91 * 86_400) == "over 90 days old", "age")

    // Only cuts below the measured limit, and after it was measured, count, by calendar day.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-stale-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let history = MemoryHistory(file: folder.appendingPathComponent("memory-history.json"), identity: "fixture")
    let day = 86_400.0
    let measured = now.addingTimeInterval(-5 * day)
    // Only cuts under the measured limit or a lower one count, not those under a higher limit. Two days apart: distinct
    // calendar days in any time zone.
    for (offset, level, limit) in [(-7 * day, 20, 22.5), (-4 * day, 21, 22.5), (-2 * day, 23, 22.5), (-2 * day, 21, 20), (-30, 19, 30)] {
        history.trimmed(to: UInt64(level) * gb, limit: UInt64(limit * 1e9), now: now.addingTimeInterval(offset))
    }
    try require(
        history.cutDays(below: 22_500_000_000, since: measured) == 2,
        "cuts before the measurement, above it or under a higher limit counted")
    // 300 cuts in one evening are one day: not stale.
    let noon = Calendar.current.startOfDay(for: now.addingTimeInterval(-4 * day)).timeIntervalSince1970 + 12 * 3_600
    for index in 0..<300 { history.trimmed(to: 20 * gb, limit: 22_500_000_000, now: Date(timeIntervalSince1970: noon + Double(index))) }
    try require(history.cutDays(below: 22_500_000_000, since: measured) == 2, "one evening's cuts counted as more than a day")
    history.trimmed(to: 20 * gb, limit: 22_500_000_000, now: now)
    try require(history.cutDays(below: 22_500_000_000, since: measured) == 3, "a third day since was lost")

    let trials = [
        SelfTuning.MemoryTrial(limit: 22_500_000_000), SelfTuning.MemoryTrial(limit: 24 * gb, excluded: "cut 2 times"),
    ].map { trial -> SelfTuning.MemoryTrial in
        var trial = trial
        (trial.first, trial.warm) = (9, 9)
        return trial
    }
    try require(
        SelfTuning.memoryText(trials, current: nil, automatic: 24 * gb, applied: true).hasSuffix("; automatic mode uses it"),
        "the applied text still asked for a setting")
}
