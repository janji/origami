import Dispatch
import Foundation
import Metal
import OrigamiEngine

// The quality dial: levels and knobs, kept ranks and renormalised gates, the late-miss substitute choice, every
// request surface (origami_mode, -<level> model ids, chat command, usage, teacher tokens, timeline epochs), and the
// decode window that never reads dropped experts and pins a substitute in a skipped record's place.
func qualityDialChecks() -> [(String, () throws -> Void)] {
    [
        ("quality_dial_levels_and_knobs", qualityDialLevelsAndKnobs),
        ("quality_dial_offered_per_model", qualityDialOfferedPerModel),
        ("quality_dial_dynamic_steps", qualityDialDynamicSteps),
        ("quality_dial_tool_call_opener", qualityDialToolCallOpener),
        ("quality_dial_kept_ranks_and_gates", qualityDialKeptRanksAndGates),
        ("quality_dial_substitute_choice", qualityDialSubstituteChoice),
        ("quality_dial_request_surfaces", qualityDialRequestSurfaces),
        ("quality_dial_window_drops_and_substitutes", qualityDialWindowDropsAndSubstitutes),
        ("streamed_text_extends_last_character", streamedTextExtendsLastCharacter),
    ]
}

/// A token may extend the character streamed last (a combining mark, a variation selector, "\n" after "\r"); the next
/// delta is then that scalar alone, never a refused request. A changed prefix is still refused.
private func streamedTextExtendsLastCharacter() throws {
    try require(
        GenerationDriver.streamDelta("e\u{301}x", after: "e") == "\u{301}x" && GenerationDriver.streamDelta("a\r\n", after: "a\r") == "\n"
            && GenerationDriver.streamDelta("☺\u{FE0F}", after: "☺") == "\u{FE0F}"
            && GenerationDriver.streamDelta("abc", after: "abc") == ""
            && GenerationDriver.streamDelta("abc", after: "") == "abc", "Stream deltas differ")
    try require(
        GenerationDriver.streamDelta("ab", after: "b") == nil && GenerationDriver.streamDelta("a", after: "ab") == nil,
        "A changed prefix was accepted")
}

private func require(_ condition: Bool, _ message: @autoclosure () -> String) throws {
    if !condition { throw EngineError(message()) }
}
private func refuses<T>(_ body: () throws -> T) -> Bool {
    do {
        _ = try body()
        return false
    } catch { return true }
}

/// A tool call decodes exact from its opener on: the opener is one integer compare against the tokenizer's single-token
/// markers, or, for a vocabulary with none of them, found on the reply's tail. With `ORIGAMI_MODEL`, that model's
/// tokenizer has one of the markers as a single token, the one its chat format writes.
private func qualityDialToolCallOpener() throws {
    try require(
        QualityDial.opensToolCall(7, text: "", openers: [7, 9]) && !QualityDial.opensToolCall(8, text: "<tool_call>", openers: [7, 9]),
        "Single-token openers differ")
    try require(
        QualityDial.opensToolCall(1, text: "Let me look.\n<tool_call>", openers: [])
            && QualityDial.opensToolCall(1, text: "x\n\n<｜DSML｜", openers: [])
            && !QualityDial.opensToolCall(1, text: "<tool_cal", openers: []) && !QualityDial.opensToolCall(1, text: "", openers: []),
        "Text openers differ")
    let tiny = try tinyFamilyTokenizer("{{ messages }}")
    try require(tiny.toolCallOpeners.isEmpty, "A vocabulary without markers has openers \(tiny.toolCallOpeners)")
    guard let folder = ProcessInfo.processInfo.environment["ORIGAMI_MODEL"] else { return }
    final class Box: @unchecked Sendable {
        var stack: TokenizerStack?
        var error: Error?
    }
    let box = Box()
    let done = DispatchSemaphore(value: 0)
    Task {
        do { box.stack = try await TokenizerStack(modelDirectory: URL(fileURLWithPath: folder)) } catch { box.error = error }
        done.signal()
    }
    done.wait()
    if let error = box.error { throw error }
    let model = box.stack!
    let single = TokenizerStack.toolCallMarkers.filter { model.encode($0).count == 1 }
    try require(
        !single.isEmpty && Set(single.map { model.encode($0)[0] }) == model.toolCallOpeners,
        "Openers \(model.toolCallOpeners) differ from the single-token markers \(single)")
}

private func qualityDialLevelsAndKnobs() throws {
    let names = QualityDial.levels.map(\.name)
    try require(
        names == ["exact", "dynamic", "faster", "fastest", "dynamic-fastest"] && Set(names).count == names.count,
        "Levels differ: \(names)")
    // `fast` names `faster`; v1's late-miss skip is an internal dial without a name.
    try require(
        QualityDial.level("exact") == .exact && QualityDial.exact.isExact && QualityDial.level("fast") == QualityDial.level("faster")
            && QualityDial.lateMissSkip == QualityDial(lateMiss: .skip) && QualityDial.lateMissSkip.name == "custom"
            && QualityDial.level("FAST") == nil && QualityDial.level("custom") == nil, "Level lookup differs")
    try require(QualityDial.levels.dropFirst().allSatisfy { !$0.dial.isExact }, "A non-exact level is exact")
    try require(QualityDial.levels.allSatisfy { $0.dial.name == $0.name }, "A level's dial does not name it")
    let custom = QualityDial(expertsPerToken: 6, tailSkip: 0.05, tailSkipMissesOnly: true, lateMiss: .substitute)
    try require(
        custom.name == "custom" && custom.signature == "k6-t0.05m-substitute" && custom.dropsExperts
            && !QualityDial.lateMissSkip.dropsExperts && QualityDial.exact.signature.isEmpty,
        "Custom dial name or signature differs: \(custom.signature)")
    try require(
        custom.knobs == [
            "experts_per_token": "6", "tail_skip": "0.05", "tail_skip_misses_only": "true", "late_miss": "substitute",
            "confident_above": "off",
        ]
            && QualityDial.exact.knobs["experts_per_token"] == "all" && QualityDial.exact.knobs["tail_skip"] == "off", "Knob report differs"
    )
    for bad in [QualityDial(expertsPerToken: 0), QualityDial(expertsPerToken: 11), QualityDial(tailSkip: 0), QualityDial(tailSkip: 1)] {
        try require(refuses { try bad.validated() }, "Invalid knobs accepted: \(bad)")
    }
    try require(
        !refuses { try custom.validated() } && !refuses { try QualityDial(tailSkipMissesOnly: true).validated() },
        "Valid knobs refused")
    // Knobs that change nothing are exact (all ten routed experts; misses-only without a tail skip).
    try require(
        try QualityDial(expertsPerToken: 10).validated().isExact && QualityDial(tailSkipMissesOnly: true).validated().isExact
            && QualityDial(expertsPerToken: 8).validated(routedExperts: 8).isExact && custom.validated() == custom,
        "No-op knobs are not exact")
    // A dial survives the JSON of a published result.
    let decoded = try JSONDecoder().decode(QualityDial.self, from: JSONEncoder().encode(custom))
    try require(decoded == custom, "Dial JSON round trip differs")
}

/// Each model offers exact and only the fast levels that clearly helped it; every other named level runs as the model's
/// mildest offered fast level, keeping an experimental delayed policy, and custom knobs run as set.
private func qualityDialOfferedPerModel() throws {
    try require(
        QualityDial.offered(.qwen) == ["exact", "faster", "fastest"] && QualityDial.offered(.glm) == ["exact", "faster"]
            && QualityDial.offered(.deepseek) == ["exact", "dynamic-fastest"], "Offered levels differ")
    let runs: [ModelFamily: [String]] = [  // what exact, dynamic, faster, fastest and dynamic-fastest run as
        .qwen: ["exact", "faster", "faster", "fastest", "faster"], .glm: ["exact", "faster", "faster", "faster", "faster"],
        .deepseek: ["exact", "dynamic-fastest", "dynamic-fastest", "dynamic-fastest", "dynamic-fastest"],
    ]
    for (family, expected) in runs {
        let ran = QualityDial.levels.map { $0.dial.offered(by: family).name }
        try require(ran == expected, "\(family.displayName) runs the levels as \(ran)")
    }
    let custom = QualityDial(expertsPerToken: 6, lateMiss: .skip)
    var delayed = QualityDial.level("fastest")!
    delayed.delayedExperts = try DelayedExpertPolicy().validated()
    var expected = QualityDial.level("faster")!
    expected.delayedExperts = delayed.delayedExperts
    try require(
        custom.offered(by: .glm) == custom && delayed.offered(by: .glm) == expected && delayed.offered(by: .qwen) == delayed,
        "Custom knobs or a delayed policy changed")
    try require(
        QualityDial.offeredByModel
            == "exact or a fast level the model offers (Qwen3.8-Flash-Next: faster, fastest; GLM-5.3-Flash: faster; DeepSeek-V4.1-Flash: dynamic-fastest)",
        "Per-model summary differs: \(QualityDial.offeredByModel)")
}

/// Dynamic: the knobs after a confident token, exact after any other; the threshold is a probability, a threshold over
/// exact knobs is exact, and a dial saved before the knob existed still decodes.
private func qualityDialDynamicSteps() throws {
    let dynamic = QualityDial.level("dynamic")!
    let faster = QualityDial.level("faster")!
    let fastest = QualityDial.level("fastest")!
    try require(
        dynamic.confidentAbove == 0.9 && !dynamic.isExact && dynamic.signature == faster.signature + "-c0.9"
            && dynamic.knobs["confident_above"] == "0.9", "Dynamic level differs: \(dynamic.signature)")
    // Softmax top-1: [ln 9, 0] gives 0.9; equal logits split evenly; no finite logit is no confidence.
    try require(
        abs(QualityDial.topProbability([logf(9), 0]) - 0.9) < 1e-6 && abs(QualityDial.topProbability([1, 1, 1, 1]) - 0.25) < 1e-12
            && QualityDial.topProbability([]) == 0 && QualityDial.topProbability([-.infinity, -.infinity]) == 0
            && QualityDial.topProbability([.nan, 1]) <= 1, "Top probability differs")
    // dynamic-fastest: fastest's knobs (late misses skipped too) after a token of at least 0.9, exact after any other.
    let dynamicFastest = QualityDial.level("dynamic-fastest")!
    try require(
        dynamicFastest == QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true, lateMiss: .skip, confidentAbove: 0.9)
            && dynamicFastest.signature == fastest.signature + "-c0.9" && dynamicFastest.step(after: [10, 0, 0]) == fastest
            && dynamicFastest.step(after: [1, 1, 0]) == .exact && dynamicFastest.validated() == dynamicFastest,
        "Dynamic-fastest level differs: \(dynamicFastest.signature)")
    try require(
        dynamic.step(after: [10, 0, 0]) == faster && dynamic.step(after: [1, 1, 0]) == .exact
            && dynamic.step(after: [logf(8.9), 0]) == .exact && fastest.step(after: [0, 0]) == fastest, "Dynamic step differs")
    for bad in [0, 1, -0.5, 2, Double.nan] {
        try require(refuses { try QualityDial(tailSkip: 0.1, confidentAbove: bad).validated() }, "Threshold \(bad) accepted")
    }
    try require(try QualityDial(confidentAbove: 0.9).validated().isExact && dynamic.validated() == dynamic, "Dynamic validation differs")
    let saved = #"{"tailSkipMissesOnly":true,"tailSkip":0.1,"lateMiss":"skip"}"#
    try require(
        try JSONDecoder().decode(QualityDial.self, from: Data(saved.utf8)) == fastest
            && JSONDecoder().decode(QualityDial.self, from: JSONEncoder().encode(dynamic)) == dynamic, "Dial JSON differs")
}

private func qualityDialKeptRanksAndGates() throws {
    let weights: [Float] = [0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
    func sum(_ values: [Float]) -> Float { values.reduce(0, +) }
    try require(
        QualityDial.exact.keptRanks(weights) == Array(0..<10) && QualityDial.lateMissSkip.keptRanks(weights) == Array(0..<10),
        "Exact or the late-miss skip dropped a routed expert")
    let five = QualityDial(expertsPerToken: 5)
    try require(five.keptRanks(weights) == [0, 1, 2, 3, 4], "Top-5 keeps \(five.keptRanks(weights))")
    // Ranks need not arrive heaviest first: the heaviest are kept wherever they sit.
    let shuffled: [Float] = [0.02, 0.30, 0.04, 0.20, 0.05, 0.12, 0.03, 0.10, 0.06, 0.08]
    try require(five.keptRanks(shuffled) == [1, 3, 5, 7, 9], "Top-5 of unordered ranks keeps \(five.keptRanks(shuffled))")
    let gates = QualityDial.gates(weights, kept: five.keptRanks(weights))
    try require(
        abs(sum(gates) - sum(weights)) < 1e-6 && gates[5...].allSatisfy { $0 == 0 }
            && abs(gates[0] / gates[1] - 1.5) < 1e-5, "Top-5 gates are not renormalised over the kept: \(gates)")
    // Tail skip by share of the routed mass (here the mass is 1): under 5.5% dropped, the heaviest always kept.
    let tail = QualityDial(tailSkip: 0.055)
    try require(tail.keptRanks(weights) == [0, 1, 2, 3, 4, 5], "Tail skip keeps \(tail.keptRanks(weights))")
    try require(QualityDial(tailSkip: 0.5).keptRanks(weights) == [0], "A tail skip above the heaviest dropped it")
    let scaled = weights.map { $0 * 4 }
    try require(tail.keptRanks(scaled) == tail.keptRanks(weights), "Tail skip depends on the gate scale")
    // Misses only: experts in memory under the threshold stay.
    let misses = QualityDial(tailSkip: 0.055, tailSkipMissesOnly: true)
    try require(
        misses.keptRanks(weights) { [7, 9].contains($0) } == [0, 1, 2, 3, 4, 5, 7, 9], "Misses-only tail skip dropped a resident expert")
    // Both: the top five, then the tail among them.
    try require(
        QualityDial(expertsPerToken: 5, tailSkip: 0.09).keptRanks(weights) == [0, 1, 2, 3], "Top-k and tail skip combine differently")
    let active: [Float] = [0.30, 0, 0.12, 0.10, 0, 0, 0, 0, 0, 0.01]
    let renormalised = QualityDial.gates(weights, active: active)
    try require(
        abs(sum(renormalised) - 1) < 1e-6 && renormalised[1] == 0 && abs(renormalised[0] / renormalised[9] - 30) < 1e-3,
        "Active gates are not renormalised to the routed mass: \(renormalised)")
}

private func qualityDialSubstituteChoice() throws {
    var logits = [Float](repeating: -10, count: 512)
    let routed = [40, 3, 17, 250, 9, 11, 500, 64, 128, 7]
    for (rank, id) in routed.enumerated() { logits[id] = 5 - Float(rank) * 0.5 }
    logits[300] = 0.4
    logits[301] = 0.3
    logits[302] = 0.2
    logits[17] = 3.9
    let weights: [Float] = [0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
    // Late ranks 9 and 6: the heavier (6) takes the best resident unrouted expert (300), 9 the next (301); routed and
    // non-resident experts are never chosen.
    let chosen = QualityDial.substitutes(
        late: [9, 6], weights: weights, routed: routed, logits: logits, resident: [17, 301, 300, 40, 999])
    try require(chosen == [6: 300, 9: 301], "Substitutes differ: \(chosen)")
    try require(
        QualityDial.substitutes(late: [9, 6, 8], weights: weights, routed: routed, logits: logits, resident: [302]) == [6: 302],
        "Each substitute must be used once")
    try require(
        QualityDial.substitutes(late: [9], weights: weights, routed: routed, logits: logits, resident: routed).isEmpty,
        "A routed expert was offered as a substitute")
    // Its weight is the router's softmax weight beside the replaced expert: w * exp(l_sub - l_late).
    let weight = QualityDial.substituteWeight(0.05, replacedLogit: logits[500], substituteLogit: logits[300])
    try require(abs(weight - 0.05 * expf(0.4 - 2.0)) < 1e-7 && weight < 0.05, "Substitute weight differs: \(weight)")
}

private func qualityDialRequestSurfaces() throws {
    func parse(_ extra: [String: Any], defaults: ServingDefaults = ServingDefaults()) throws -> ServingRequest {
        var body: [String: Any] = ["model": "qwen3.8-flash-next", "messages": [["role": "user", "content": "hi"]]]
        for (key, value) in extra { body[key] = value }
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192, defaults: defaults)
    }
    for level in QualityDial.levels {
        let named = try parse(["origami_mode": level.name])
        try require(named.quality == level.dial && named.explicitMode, "origami_mode \(level.name) did not select it")
        guard level.name != "exact" else { continue }
        let byID = try parse(["model": "qwen3.8-flash-next-" + level.name])
        try require(
            byID.quality == level.dial && byID.explicitMode && byID.model == "qwen3.8-flash-next-" + level.name,
            "The -\(level.name) model id did not select it")
        try require(
            refuses { try parse(["model": "qwen3.8-flash-next-" + level.name, "origami_mode": "exact"]) },
            "A level id with origami_mode exact was not refused")
    }
    try require(
        try parse(["model": "origami-fastest", "origami_mode": "fastest"]).quality == QualityDial.level("fastest"),
        "A matching id and mode were refused")
    // `fast` (v1's name) is `faster` as a mode and as a model id, and a matching pair is not a conflict.
    try require(
        try parse(["origami_mode": "fast"]).quality == QualityDial.level("faster")
            && parse(["model": "qwen3.8-flash-next-fast"]).quality == QualityDial.level("faster")
            && parse(["model": "qwen3.8-flash-next-fast", "origami_mode": "faster"]).quality == QualityDial.level("faster")
            && parse(["model": "qwen3.8-flash-next-faster"]).quality == QualityDial.level("faster"), "fast does not name faster")
    // J61: only the served model's id (or `origami`) with a level is a level; another name ending in one is just a name.
    try require(
        try parse(["model": "something-fast"]).quality == .exact && parse(["model": "x-fastest"]).quality == .exact
            && parse(["model": "origami-faster"]).quality == QualityDial.level("faster"), "A name merely ending in a level selected it")
    try require(
        refuses { try parse(["origami_mode": "turbo"]) } && refuses { try parse(["origami_mode": 1]) }, "An unknown level was accepted")
    // The server's default applies to a request that names no level, and only to it.
    var defaults = ServingDefaults()
    defaults.quality = QualityDial.level("fastest")!
    try require(
        try parse([:], defaults: defaults).quality == QualityDial.level("fastest") && !parse([:], defaults: defaults).explicitMode
            && parse(["origami_mode": "exact"], defaults: defaults).quality == .exact && parse([:]).quality == .exact,
        "The default quality level is not applied as the request's absent level")
    // Teacher tokens and returned tokens are measurement fields; usage carries them only when asked.
    let teacher = try parse(["origami_teacher_tokens": [1, 2, 3], "origami_return_tokens": true])
    try require(
        teacher.teacherTokens == [1, 2, 3] && teacher.returnTokens && parse([:]).teacherTokens == nil, "Teacher tokens were not parsed")
    for bad in [[] as [Any], [1, -2], [1, true], ["1"], [1.5]] as [Any] {
        try require(refuses { try parse(["origami_teacher_tokens": bad]) }, "Teacher tokens \(bad) were accepted")
    }
    var stats = GenerationStatistics(promptTokens: 5, completionTokens: 3, quality: QualityDial.level("fastest")!)
    stats.outputTokens = [7, 8, 9]
    var agreement = TeacherAgreement()
    agreement.positions = 3
    agreement.agreed = 2
    agreement.firstDifference = 1
    stats.teacher = agreement
    let usage = try ServingWire.usage(stats)
    try require(
        usage["origami_mode"] as? String == "fastest" && usage["origami_tokens"] as? [Int] == [7, 8, 9]
            && (usage["origami_teacher"] as? [String: Any])?["agreed"] as? Int == 2, "Usage does not report the level or teacher result")
    let custom = try ServingWire.usage(GenerationStatistics(promptTokens: 5, completionTokens: 3, quality: QualityDial(expertsPerToken: 6)))
    try require(custom["origami_mode"] as? String == "custom", "A custom dial is not reported as custom")
    // Chat commands.
    try require(
        ChatCommand.parse("origami quality Fastest") == ChatCommand(verb: "quality", arguments: ["fastest"])
            && ChatCommand.parse("origami quality") == ChatCommand(verb: "quality")
            && ChatCommand.parse("origami quality turbo") == nil && ChatCommand.parse("origami quality fast now") == nil
            && ChatCommand.parse("origami fast on") == ChatCommand(verb: "fast", arguments: ["on"]), "Quality chat command parsing differs")
    // /health, /v1/models: the levels the served model offers, where allowed.
    var status = ServingStatus(version: "x", ready: true)
    try require(status.wire["quality_levels"] as? [String] == ["exact"], "Exact-only status lists other levels")
    status.fastOptIn = true
    for family in ModelFamily.allCases {
        status.family = family
        try require(status.wire["quality_levels"] as? [String] == QualityDial.offered(family), "Status does not list \(family)'s levels")
    }
    // Every non-exact dial has its own timeline epoch; v1's late-miss skip keeps its own, so its saved conversations continue.
    let epoch = "e"
    let epochs = QualityDial.levels.map { FastLateMissSkip.timelineEpoch(epoch, quality: $0.dial) }
    try require(
        epochs[0] == epoch && Set(epochs).count == epochs.count
            && FastLateMissSkip.timelineEpoch(epoch, quality: .lateMissSkip) == epoch + ":fast-late-miss-skip-v1",
        "Timeline epochs differ from the rule: \(epochs)")
    try require(
        try FastLateMissSkip.requestTau(.exact, allowedTau: nil) == nil
            && FastLateMissSkip.requestTau(QualityDial(expertsPerToken: 5), allowedTau: 1) == 1
            && refuses { try FastLateMissSkip.requestTau(QualityDial(expertsPerToken: 5), allowedTau: nil) },
        "A non-exact dial was not gated by the fast allowance")
}

/// Kept experts 0-7 of ten (8 and 9 dropped, never requested); 0-5 and 20 in memory, 6 and 7 still being read at the
/// join point. Skipping 7 with expert 20 as its substitute: finish gets [6, 7, 8, 9] with 6's own record, 20's record
/// for 7 and the first pinned record for the dropped 8 and 9; nothing reads 8, 9 or 20 again, 7's read is not cancelled.
private func qualityDialWindowDropsAndSubstitutes() throws {
    final class Reads: @unchecked Sendable {
        let lock = NSLock()
        var semaphores: [Int: DispatchSemaphore] = [:], counts: [Int: Int] = [:]
        func semaphore(_ expert: Int) -> DispatchSemaphore {
            lock.lock()
            defer { lock.unlock() }
            if let value = semaphores[expert] { return value }
            let value = DispatchSemaphore(value: 0)
            semaphores[expert] = value
            return value
        }
        func read(_ expert: Int) {
            lock.lock()
            counts[expert, default: 0] += 1
            lock.unlock()
        }
        func count(_ expert: Int) -> Int {
            lock.lock()
            defer { lock.unlock() }
            return counts[expert] ?? 0
        }
    }
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let reads = Reads()
        let configuration = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 32, slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4)
        let execution = try ExpertExecution(
            device: device, ledger: ledger, configuration: configuration,
            fastSkipReserve: FastLateMissSkip.readReserve, packedAvailable: { false }
        ) { key, _, bytes in
            reads.read(key.expert)
            if [6, 7].contains(key.expert), reads.semaphore(key.expert).wait(timeout: .now() + 10) != .success {
                throw EngineError("Quality dial fixture read timed out")
            }
            bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
        }
        try execution.beginSequence()
        let warm = try execution.makeWindow(layer: 0, experts: [0, 1, 2, 3, 4, 5, 20])
        for expert in [0, 1, 2, 3, 4, 5, 20] { try warm.consume(expert: expert) { _ in } }
        try execution.beginDecodePhase()
        try require(try Set(execution.residentExperts(layer: 0)) == [0, 1, 2, 3, 4, 5, 20], "Resident experts differ")
        let window = try execution.makeWindow(layer: 0, experts: Array(0..<8))
        try require(window.residentAtRequestCount == 6, "Quality dial fixture warm records were not resident")
        var late: [Int] = []
        var offered: [Int] = []
        var readyIDs: [Int] = []
        var readyBuffers: [ExpertBuffer] = []
        var finishIDs: [Int] = []
        var finishBuffers: [ExpertBuffer] = []
        var finishBytes: [UInt8] = []
        try window.consumeTenReadyFirst(
            experts: Array(0..<8),
            skipLate: { seen, _ in
                late = seen
                reads.semaphore(6).signal()  // the kept late record may finish
                return [7]
            },
            substitute: { skipped in
                offered = skipped
                return [7: 20]
            }, dropped: [8, 9],
            ready: { ids, buffers in
                readyIDs = ids
                readyBuffers = buffers
            },
            finish: { (ids: [Int], buffers: [ExpertBuffer]) -> Void in
                finishIDs = ids
                finishBuffers = buffers
                finishBytes = buffers.map { $0.bytes[0] }
            })
        try require(
            readyIDs == Array(0..<6) && late == [6, 7] && offered == [7] && finishIDs == [6, 7, 8, 9],
            "Window groups differ: ready \(readyIDs), late \(late), finish \(finishIDs)")
        try require(
            finishBuffers.count == 4 && finishBytes == [6, 20, 0, 0] && finishBuffers[2] === readyBuffers[0]
                && finishBuffers[3] === readyBuffers[0], "Substitute or dropped records bound wrongly: \(finishBytes)")
        try require(
            reads.count(8) == 0 && reads.count(9) == 0 && reads.count(20) == 1 && reads.count(7) == 1,
            "Dropped or substitute experts were read")
        reads.semaphore(7).signal()
        try execution.joinReadProducers()
        let joined = execution.snapshot()
        try require(
            joined.cache.inFlightReads == 0 && joined.cache.pinnedSlots == 0 && joined.io.drives.allSatisfy { $0.cancelled == 0 },
            "The window left reads or pins: \(joined.cache)")
        // A dropped expert inside the window is refused; closing the window leaves no pin.
        let again = try execution.makeWindow(layer: 0, experts: Array(0..<8))
        try require(
            refuses {
                try again.consumeTenReadyFirst(
                    experts: Array(0..<8), dropped: [3], ready: { _, _ in }, finish: { (_: [Int], _: [ExpertBuffer]) -> Void in })
            }, "A dropped expert inside the window was accepted")
        again.close()
        try require(execution.snapshot().cache.pinnedSlots == 0, "A refused window leaked a pin")
        try execution.finishSequence()
    }
    for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
    try require(ledger.snapshot().reserved == 0, "Quality dial fixture leaked its reservations")
}
