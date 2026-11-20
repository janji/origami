import Foundation
import Metal
import OrigamiEngine

/// Exercises the actual family serving loop with a CPU-only model and a tiny tokenizer: budgets, natural closure,
/// total caps, teacher diagnostics and cancelled partial metrics require no model files or GPU execution.
// Swift 6.3.3's release CopyPropagation pass crashes on this fixture's optional TeacherAgreement borrow.
// Keep only this untimed test body unoptimized; the serving implementation and performance probes remain optimized.
@_optimize(none)
func familyServingThinkingBudgets() throws {
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Family thinking budget: " + message) }
    }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 8192, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 6 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        var generated: [Int] = [], prefilled = false, naturalCloseAt: Int?, ordinary = false
        var fixedLogits: [Float]?
        func statusNotes() -> [String: Any] { [:] }
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [] }
        func restored(_ tokens: [Int]) { position = tokens.count }
        func reset() throws {
            position = 0
            generated = []
            prefilled = false
        }
        func forward(_ tokens: [Int]) throws -> [Float] {
            position += tokens.count
            if prefilled { generated += tokens } else { prefilled = true }
            if let fixedLogits { return fixedLogits }
            let token: Int
            if generated.last == 65 {
                token = 259
            }  // A followed by EOS.
            else if ordinary || generated.contains(258) {
                token = 65
            } else if naturalCloseAt == generated.count {
                token = 258
            } else {
                token = 72
            }  // H inside the open think block.
            var logits = [Float](repeating: -100, count: vocabulary)
            logits[token] = 100
            return logits
        }
    }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}{% if thinking %}<think>{% endif %}")
    func request(
        maximum: Int = 2048, budget: Int? = 3, effort: String = "high", teacher: [Int]? = nil, presence: Float = 0,
        defaults: ServingDefaults = ServingDefaults()
    ) throws -> ServingRequest {
        var body: [String: Any] = [
            "model": "family", "messages": [["role": "user", "content": "Hi"]],
            "max_tokens": maximum, "temperature": 0, "reasoning_effort": effort,
            "origami_return_tokens": true, "prompt_cache_policy": "no_store",
        ]
        body["presence_penalty"] = presence
        if let budget { body["thinking_budget"] = budget }
        if let teacher { body["origami_teacher_tokens"] = teacher }
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192, defaults: defaults)
    }
    func backend(_ model: Stub) -> FamilyServingBackend {
        FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
    }
    let control = GenerationControl()
    let bounded = backend(Stub())
    let budgeted = try bounded.prepare(request(), cancellation: control).run { _ in true }
    let closing = [258] + tokenizer.encode("\n\n")
    try require(budgeted.outputTokens == [72, 72, 72] + closing + [65], "did not force the family closing at its budget")
    try require(
        budgeted.reasoningTokens == 3 && budgeted.reasoningClosedBy == "budget" && budgeted.finishReason == "stop",
        "budget/answer usage is incorrect")
    try require(
        control.progress.generatedTokens == budgeted.completionTokens && control.progress.firstGeneratedAt != nil,
        "actual generated-token progress differs from completed usage")
    try require(control.progress.stageSeconds == budgeted.stageSeconds, "completed stages were not published")

    var defaults = ServingDefaults()
    defaults.maxThinkingTokens = 2
    let inherited = try backend(Stub()).prepare(request(budget: nil, defaults: defaults), cancellation: GenerationControl()).run { _ in true
    }
    try require(inherited.reasoningTokens == 2 && inherited.reasoningClosedBy == "budget", "inherited budget was ignored")

    let natural = Stub()
    natural.naturalCloseAt = 2
    let naturallyClosed = try backend(natural).prepare(request(), cancellation: GenerationControl()).run { _ in true }
    try require(
        naturallyClosed.outputTokens == [72, 72, 258, 65] && naturallyClosed.reasoningTokens == 2
            && naturallyClosed.reasoningClosedBy == nil, "natural closing was forced again")

    // No budget: as Qwen's, the think block closes where only the answer room is left (`ThinkCloser.limit`), so a reply
    // that would run out of tokens while thinking answers within `max_tokens`; one token leaves room only for the closing.
    for maximum in [16, 1026] {
        let unbudgeted = try backend(Stub()).prepare(
            request(maximum: maximum, budget: nil),
            cancellation: GenerationControl()
        ).run { _ in true }
        let limit = ThinkCloser.limit(maximumTokens: maximum, budget: nil)
        try require(
            unbudgeted.outputTokens == [Int](repeating: 72, count: limit) + closing + [65]
                && unbudgeted.completionTokens <= maximum && unbudgeted.reasoningTokens == limit
                && unbudgeted.reasoningClosedBy == "answer_room" && unbudgeted.finishReason == "stop",
            "a reply out of tokens while thinking did not close at the answer room (\(maximum))")
    }
    let tiny = try backend(Stub()).prepare(request(maximum: 1, budget: nil), cancellation: GenerationControl()).run { _ in true }
    try require(
        tiny.outputTokens == [258] && tiny.completionTokens == 1 && tiny.finishReason == "length", "the closing exceeded a 1-token cap")
    // A reply that closes its think block itself, well before the answer room, is unchanged: tokens and usage.
    for maximum in [16, 2048] {
        let unbudgetedNatural = Stub()
        unbudgetedNatural.naturalCloseAt = 2
        let naturallyUnbudgeted = try backend(unbudgetedNatural).prepare(
            request(maximum: maximum, budget: nil),
            cancellation: GenerationControl()
        ).run { _ in true }
        try require(
            naturallyUnbudgeted.outputTokens == [72, 72, 258, 65] && naturallyUnbudgeted.completionTokens == 4
                && naturallyUnbudgeted.reasoningTokens == 2 && naturallyUnbudgeted.reasoningClosedBy == nil
                && naturallyUnbudgeted.finishReason == "stop", "unbudgeted natural closure changed")
    }
    // Thinking off: no closing, however long the reply runs.
    let unthinking = try backend(Stub()).prepare(request(maximum: 16, budget: nil, effort: "none"), cancellation: GenerationControl())
        .run { _ in true }
    try require(
        unthinking.outputTokens == [Int](repeating: 72, count: 16) && unthinking.reasoningTokens == nil
            && unthinking.reasoningClosedBy == nil && unthinking.finishReason == "length", "thinking off was closed")

    let room = try backend(Stub()).prepare(request(maximum: 1026, budget: 999), cancellation: GenerationControl()).run { _ in true }
    // An explicit budget keeps at most half the output (513 of 1026) as answer room, so the requested thinking still happens.
    try require(room.reasoningTokens == 513 && room.reasoningClosedBy == "answer_room", "explicit budget lost answer-room reservation")
    let capped = try backend(Stub()).prepare(request(maximum: 1), cancellation: GenerationControl()).run { _ in true }
    try require(
        capped.outputTokens == [258] && capped.completionTokens == 1 && capped.finishReason == "length",
        "forced closing exceeded total output cap")

    let ordinary = Stub()
    ordinary.ordinary = true
    let plain = try backend(ordinary).prepare(request(effort: "none"), cancellation: GenerationControl()).run { _ in true }
    try require(
        plain.outputTokens == [65] && plain.reasoningTokens == nil && plain.reasoningClosedBy == nil,
        "disabled thinking injected a closure or reported reasoning")

    let teacher = [72, 72, 72, 72, 258, 65]
    let diagnostic = try backend(Stub()).prepare(
        request(maximum: teacher.count, budget: 1, teacher: teacher),
        cancellation: GenerationControl()
    ).run { _ in true }
    try require(
        diagnostic.outputTokens == teacher && diagnostic.teacher?.positions == teacher.count
            && diagnostic.reasoningClosedBy == nil, "thinking policy replaced diagnostic teacher tokens")

    // The family loop's incremental membership includes the emitted teacher/replay token, never the overridden
    // sampler proposal, and repeated tokens receive the presence penalty only once.
    let penalties = Stub()
    var fixed = [Float](repeating: -100, count: penalties.vocabulary)
    fixed[65] = 3
    fixed[66] = 2.8
    fixed[67] = 2.6
    penalties.fixedLogits = fixed
    let penalized = try backend(penalties).prepare(
        request(maximum: 6, effort: "none", presence: 0.5),
        cancellation: GenerationControl()
    ).run { _ in true }
    try require(penalized.outputTokens == [65, 66, 67, 65, 65, 65], "presence membership changed repeated-token sampling")
    let forcing = Stub()
    forcing.fixedLogits = fixed
    let forcedMembership = try backend(forcing).prepare(
        request(maximum: 6, effort: "none", teacher: [65, 65, 66], presence: 0.5),
        cancellation: GenerationControl()
    ).run { _ in true }
    try require(forcedMembership.outputTokens == [65, 65, 66, 67, 65, 65], "presence membership followed a replaced proposal")

    let cancelled = GenerationControl()
    do {
        _ = try backend(Stub()).prepare(request(), cancellation: cancelled).run { _ in
            cancelled.cancel()
            return true
        }
        throw EngineError("Family thinking budget: cancelled request completed")
    } catch { if !cancelled.isCancelled { throw error } }
    try require(
        cancelled.progress.generatedTokens == 1 && cancelled.progress.firstGeneratedAt != nil,
        "cancelled partial token count was lost")
    try require(
        cancelled.progress.stageSeconds["generation"] != nil
            && cancelled.progress.stageSeconds["prefill_and_save"] != nil,
        "cancelled partial stage timing was lost")
}
