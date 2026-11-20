import Foundation
import Metal
import OrigamiEngine

/// A cancel while a follow-up prompt is read keeps the conversation: inside a slice or between slices, the state goes
/// back to the last slice end read (with a model tail, a point after fed rows too, kept for the prompt it was read
/// for), and the request that resends the prompt continues from there with the same reply and state bytes as a run
/// never cancelled. The stub's running state depends on every row in order, so a wrong rewind or a reset shows.
func familyPromptCancelKeepsConversation() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw FixtureUnavailable(description: "no Metal device") }
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Family prompt cancel: " + message) }
    }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 16384, resets = 0, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var promptTail: Int?
        var routedExperts: Int { 8 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        func statusNotes() -> [String: Any] { [:] }
        let running = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16, options: .storageModeShared)!
        let rows = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16384 * 8, options: .storageModeShared)!
        /// Cancels inside the call reading row `inside` (after writing the rows before it), or once a call ends at or after `after`.
        var cancel: (() -> Void)?, inside: Int?, after: Int?
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [(running, 16), (rows, position * 8)] }
        var state: Data { Data(bytes: running.contents(), count: 16) + Data(bytes: rows.contents(), count: position * 8) }
        func reset() throws {
            position = 0
            resets += 1
            memset(running.contents(), 0, running.length)
            memset(rows.contents(), 0, rows.length)
        }
        func restored(_ tokens: [Int]) { position = tokens.count }
        func feed(_ tokens: [Int]) throws { _ = try forward(tokens) }
        func forward(_ tokens: [Int]) throws -> [Float] {
            let state = running.contents().assumingMemoryBound(to: Int64.self)
            let appended = rows.contents().assumingMemoryBound(to: Int64.self)
            for (i, token) in tokens.enumerated() {
                if let at = inside, position + i == at {
                    inside = nil
                    cancel?()
                    // A real pass checks between layers: the running state is already part written.
                    guard shouldContinue() else { throw EngineError("Cancelled", code: "cancelled") }
                }
                appended[position + i] = Int64(token) &+ state[0]
                state[0] = state[0] &* 31 &+ Int64(token)
                state[1] &+= 1
            }
            position += tokens.count
            if let at = after, position >= at {
                after = nil
                cancel?()
            }
            var logits = [Float](repeating: 0, count: vocabulary)
            logits[65 + Int(UInt64(bitPattern: state[0]) % 26)] = 100
            return logits
        }
    }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}")
    let opening = String(repeating: "H", count: 1000)
    func request(_ messages: [[String: String]]) throws -> ServingRequest {
        let body: [String: Any] = [
            "model": "family", "max_tokens": 6, "temperature": 0, "reasoning_effort": "none", "messages": messages,
        ]
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 16384)
    }
    final class Text: @unchecked Sendable { var value = "" }
    func say(_ backend: FamilyServingBackend, _ request: ServingRequest) throws -> (String, GenerationStatistics) {
        let text = Text()
        let stats = try backend.prepare(request, cancellation: GenerationControl()).run {
            text.value += $0
            return true
        }
        return (text.value, stats)
    }
    for tail in [nil, 3000] {
        // An opening turn, then a follow-up of about 6,000 tokens read in slices of 2,048 (fed while 3,000 or more
        // before its end when the model has a tail).
        func conversation() throws -> (FamilyServingBackend, Stub, followUp: ServingRequest, reused: Int) {
            let stub = Stub()
            stub.promptTail = tail
            try stub.reset()
            let backend = FamilyServingBackend(model: stub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
            let (reply, _) = try say(backend, try request([["role": "user", "content": opening]]))
            let followUp = try request([
                ["role": "user", "content": opening], ["role": "assistant", "content": reply],
                ["role": "user", "content": String(repeating: "H", count: 6000)],
            ])
            return (backend, stub, followUp, stub.position)
        }
        let (plain, plainStub, followUp, reused) = try conversation()
        let (want, wanted) = try say(plain, followUp)
        let wantState = plainStub.state
        try require(wanted.cachedTokens == reused && reused > GenerationDriver.minimumSavedTokens, "fixture did not continue live")
        // Inside the third slice, and between the first and second.
        let slice = FamilyRunner.promptSlice
        for (inside, after) in [(reused + 2 * slice + 7, nil), (nil, reused + slice)] as [(Int?, Int?)] {
            let (backend, stub, followUp, _) = try conversation()
            let control = GenerationControl()
            stub.cancel = { control.cancel() }
            (stub.inside, stub.after) = (inside, after)
            do {
                _ = try backend.prepare(followUp, cancellation: control).run { _ in true }
                throw EngineError("Family prompt cancel: the cancelled request ran to its end")
            } catch { if !control.isCancelled { throw error } }
            // The slice boundary before the cancel, after fed rows or not.
            let kept = inside.map { $0 - 7 } ?? after!
            try require(
                stub.resets == 1 && stub.position == kept,
                "tail \(tail ?? 0), cancel at \(inside ?? after!): kept \(stub.position), not \(kept) (resets \(stub.resets))")
            let (got, stats) = try say(backend, followUp)
            try require(
                got == want && stats.cachedTokens == kept && stats.completionTokens == wanted.completionTokens && stub.state == wantState
                    && stub.resets == 1,
                "tail \(tail ?? 0), cancel at \(inside ?? after!): continued \(stats.cachedTokens), reply \(got) vs \(want)")
        }
    }
}
