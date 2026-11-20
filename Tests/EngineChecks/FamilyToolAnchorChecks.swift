import Foundation
import Metal
import OrigamiEngine

/// A rewritten tool call reuses its exact prefix while recurrent/ring state rewinds and append-only rows stay live.
/// A stub records both kinds of state, so restoring just a position or using overwritten rows fails the check.
func familyToolAnchors() throws {
    // The stub model's state lives in Metal buffers.
    guard MTLCreateSystemDefaultDevice() != nil else { throw FixtureUnavailable(description: "no Metal device") }
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Family tool anchor: " + message) }
    }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 8192, resets = 0, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 8 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        func statusNotes() -> [String: Any] { [:] }
        let mutable = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16, options: .storageModeShared)!
        let rows = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 8192 * 8, options: .storageModeShared)!
        var reply: [Int] = [], replyStart = 0, restoredTokens: [Int]?, badRestore = false
        var zeroStateReads = 0, interruptPosition: Int?
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] {
            if position == 0 { zeroStateReads += 1 }
            return [(mutable, 16), (rows, position * 8)]
        }
        func reset() throws {
            position = 0
            resets += 1
            memset(mutable.contents(), 0, mutable.length)
            memset(rows.contents(), 0, rows.length)
        }
        func restored(_ tokens: [Int]) {
            let state = mutable.contents().assumingMemoryBound(to: Int64.self)
            let appended = rows.contents().assumingMemoryBound(to: Int64.self)
            badRestore =
                badRestore || state[0] != Int64(tokens.count) || state[1] != Int64(tokens.reduce(0, +))
                || tokens.enumerated().contains { appended[$0.offset] != Int64($0.element) }
            restoredTokens = tokens
            position = tokens.count
        }
        func forward(_ tokens: [Int]) throws -> [Float] {
            if let at = interruptPosition, position >= at {
                interruptPosition = nil
                throw EngineError.gpu("tool anchor check", EngineError.fakeInterruption)
            }
            let state = mutable.contents().assumingMemoryBound(to: Int64.self)
            let appended = rows.contents().assumingMemoryBound(to: Int64.self)
            for token in tokens {
                appended[position] = Int64(token)
                state[1] += Int64(token)
                position += 1
            }
            state[0] = Int64(position)
            let step = position - replyStart
            let token = reply.indices.contains(step) ? reply[step] : 72
            var logits = [Float](repeating: 0, count: vocabulary)
            logits[token] = 100
            return logits
        }
    }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}")
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-tool-anchors-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    var count = 0
    func setup() throws -> (FamilyServingBackend, Stub) {
        count += 1
        let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
        let cache = try PromptCache(
            root: root.appendingPathComponent("\(count)"), epoch: "anchor", entries: 3,
            entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2, ledger: ledger, capacity: 8192)
        let stub = Stub()
        try stub.reset()
        let backend = FamilyServingBackend(model: stub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
        backend.promptCache = cache
        backend.epoch = "anchor"
        return (backend, stub)
    }
    let prompt = String(repeating: "H", count: 600)
    let reply = "thinking<tool_call>{\"x\":1}<tool_call>{\"x\":2}"
    let boundaryText = prompt + "thinking<tool_call"
    func request(_ text: String, maximum: Int, extra: [String: Any] = [:]) throws -> ServingRequest {
        // The reply part comes back as the assistant's turn, as clients send it: typed user text has its markers escaped.
        let rest = text.count > prompt.count && text.hasPrefix(prompt) ? String(text.dropFirst(prompt.count)) : nil
        var body: [String: Any] = [
            "model": "family", "max_tokens": maximum, "temperature": 0, "reasoning_effort": "none",
            "messages": [["role": "user", "content": rest == nil ? text : prompt]]
                + (rest.map { [["role": "assistant", "content": $0]] } ?? []),
        ]
        for (key, value) in extra { body[key] = value }
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
    }
    func run(
        _ backend: FamilyServingBackend, _ stub: Stub, text: String, output: String, extra: [String: Any] = [:],
        control: GenerationControl = GenerationControl(), emit: @escaping @Sendable (String) throws -> Bool = { _ in true }
    ) throws -> GenerationStatistics {
        let req = try request(text, maximum: output.utf8.count, extra: extra)
        stub.replyStart = try ServingPrompt.encode(req, tokenizer: tokenizer).count
        stub.reply = Array(output.utf8).map(Int.init)
        return try backend.prepare(req, cancellation: control).run(emit: emit)
    }
    let changed = prompt + "thinking<tool_call>{ \"x\": 1 }"
    let boundary = try ServingPrompt.encode(request(boundaryText, maximum: 1), tokenizer: tokenizer).count
    let (backend, stub) = try setup()
    let initial = try run(backend, stub, text: prompt, output: reply)
    let entries = backend.promptCache!.entries
    try require(
        stub.zeroStateReads == 1 && entries.count == 1 && entries[0].tokens == initial.promptTokens,
        "capture was repeated or wrote a decode snapshot to disk")
    let resets = stub.resets
    let resumed = try run(backend, stub, text: changed, output: "HH", extra: ["prompt_cache_policy": "no_store"])
    try require(
        resumed.cachedTokens == boundary && stub.restoredTokens?.count == boundary && !stub.badRestore && stub.resets == resets,
        "rewritten JSON reused \(resumed.cachedTokens), expected \(boundary), bad state \(stub.badRestore)")
    try require(backend.promptCache!.entries.map(\.id) == entries.map(\.id), "read-only anchor reuse wrote a prompt")

    // Exact token equality is required, and only the following request can borrow the original append-only rows.
    let (mismatch, mismatchStub) = try setup()
    _ = try run(mismatch, mismatchStub, text: prompt, output: reply)
    let altered = try run(mismatch, mismatchStub, text: prompt + "different<tool_call>{}", output: "HH")
    try require(altered.cachedTokens == initial.promptTokens, "changed reasoning reused a tool anchor")
    let (side, sideStub) = try setup()
    _ = try run(side, sideStub, text: prompt, output: reply)
    _ = try run(side, sideStub, text: "unrelated", output: "HH")
    let afterSide = try run(side, sideStub, text: changed, output: "HH")
    try require(afterSide.cachedTokens < boundary && !sideStub.badRestore, "a side request left an anchor backed by overwritten rows")

    // Requests that cannot save exact model output never capture. Teacher forcing is explicitly checked even though
    // parsing already makes it no_store; internal callers may construct requests themselves.
    for extra: [String: Any] in [
        ["prompt_cache_policy": "no_store"], ["origami_mode": "faster"],
        ["origami_teacher_tokens": Array(reply.utf8).map(Int.init)],
    ] {
        let (skipped, skippedStub) = try setup()
        _ = try run(skipped, skippedStub, text: prompt, output: reply, extra: extra)
        try require(skippedStub.zeroStateReads == 0, "captured disallowed request \(extra)")
    }
    let (trimmed, trimmedStub) = try setup()
    _ = try run(trimmed, trimmedStub, text: prompt, output: reply)
    try require(trimmed.trimExperts() > 0, "idle trim retained the anchor")
    let afterTrim = try run(trimmed, trimmedStub, text: changed, output: "HH")
    try require(afterTrim.cachedTokens < boundary, "trimmed anchor remained reusable")

    let (cancelled, cancelledStub) = try setup()
    let control = GenerationControl()
    do {
        _ = try run(cancelled, cancelledStub, text: prompt, output: reply, control: control) { delta in
            if delta.contains("1") { control.cancel() }
            return true
        }
        throw EngineError("Family tool anchor: cancellation completed")
    } catch { if !control.isCancelled { throw error } }
    let afterCancel = try run(cancelled, cancelledStub, text: changed, output: "HH")
    try require(afterCancel.cachedTokens < boundary, "cancelled request retained its anchor")

    // A GPU retry must drop the first attempt's anchor, replay safely, and capture fresh state once for that attempt.
    let (retrying, retryStub) = try setup()
    retryStub.interruptPosition = boundary + 3
    let retried = try run(retrying, retryStub, text: prompt, output: reply)
    let afterRetry = try run(retrying, retryStub, text: changed, output: "HH")
    try require(
        retried.gpuRetries == 1 && afterRetry.cachedTokens == boundary && !retryStub.badRestore,
        "retry did not recapture a valid tool boundary")
}
