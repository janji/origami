import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// A follow-up gives the same reply whether the conversation goes on in the same server or after a restart (ROADMAP
/// R19), through the serving backend over the synthetic model: request A (a 2,470-token prompt, 13 tokens), then B (A's
/// chat, its reply and a new turn, 40 tokens); then a new backend and model over the same saved prompts serve B again,
/// restoring all of it. B's reply, the state after it and the logits of one more token must be bitwise the first
/// server's. Three ways B is reached: A's reply re-encodes to other tokens (B restores A's prompt and reads the rest);
/// it re-encodes to the same tokens (B continues the live state); and that, with B read cold by an earlier server, so
/// its prompt was saved already. Before the fix the last one restored the earlier server's state, not the live one.
func familyServingRestoreMatchesLive(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let name = family == .glm ? "glm" : "deepseek"
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-serving-restore-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try syntheticFamilyGGUF(family, in: folder)
    func text(_ count: Int, seed: UInt64) -> String {
        var state = seed
        return String(
            (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Character(UnicodeScalar(UInt8(97 + (state >> 33) % 26)))
            })
    }
    let a = [["role": "user", "content": text(2_470, seed: 11)]]
    let turn = ["role": "user", "content": text(12, seed: 12)]
    for (scenario, fillerType, earlier) in [("restored", 1, false), ("live", 4, false), ("live over an earlier save", 4, true)] {
        func require(_ condition: Bool, _ what: String) throws {
            if !condition { throw EngineError("\(name) serving restore, \(scenario): " + what) }
        }
        // The synthetic vocabulary is 2,048: fillers past the tokenizer's 261 so every reply decodes.
        let tokenizer = try tinyFamilyTokenizer(
            "{% for m in messages %}{{ m.content }}{% endfor %}", fillers: 2_048 - 261, fillerType: fillerType)
        let cacheRoot = folder.appendingPathComponent("cache-\(fillerType)-\(earlier)", isDirectory: true)
        func session() throws -> (FamilyServingBackend, FamilyRunner) {
            let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 1)
            let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
            let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: 8 << 30, context: 8_192, ledger: ledger) { _ in }
            let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
            backend.promptCache = try PromptCache(
                root: cacheRoot, epoch: "serving-restore", entries: 8, entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2,
                ledger: ledger, capacity: 8_192)
            backend.epoch = "serving-restore"
            return (backend, model as! FamilyRunner)
        }
        func serve(_ backend: FamilyServingBackend, _ messages: [[String: String]], maximumTokens: Int, noStore: Bool = false) throws
            -> GenerationStatistics
        {
            let body: [String: Any] = [
                "model": "family", "messages": messages, "max_tokens": maximumTokens, "temperature": 0, "reasoning_effort": "none",
                "origami_return_tokens": true, "prompt_cache_policy": noStore ? "no_store" : "default",
            ]
            let request = try ServingRequest.parse(
                JSONSerialization.data(withJSONObject: body), contextTier: 8_192, defaults: ServingDefaults())
            return try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
        }
        func followUp(_ reply: GenerationStatistics) throws -> [[String: String]] {
            a + [["role": "assistant", "content": try tokenizer.decode(reply.outputTokens ?? [])], turn]
        }
        /// The state's digest and the logits after one more token.
        func after(_ runner: FamilyRunner) throws -> (String, [UInt32]) {
            var hash = SHA256()
            for part in runner.stateParts(at: runner.position) {
                hash.update(bufferPointer: UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes))
            }
            return (hash.finalize().description, try (runner as! FamilyModel).forward([97]).map(\.bitPattern))
        }
        if earlier {
            // B read whole by another server (A's reply from a server that saves nothing).
            let reply = try autoreleasepool { try serve(session().0, a, maximumTokens: 13, noStore: true) }
            _ = try autoreleasepool { try serve(session().0, followUp(reply), maximumTokens: 1) }
        }
        let (replyA, liveB, liveAfter) = try autoreleasepool {
            let (backend, runner) = try session()
            let replyA = try serve(backend, a, maximumTokens: 13)
            let replyB = try serve(backend, try followUp(replyA), maximumTokens: 40)
            return (replyA, replyB, try after(runner))
        }
        let (restartB, restartAfter) = try autoreleasepool {
            let (backend, runner) = try session()
            let replyB = try serve(backend, try followUp(replyA), maximumTokens: 40)
            return (replyB, try after(runner))
        }
        // Restored: A's prompt; live: past it, A's reply as decoded.
        try require(
            (liveB.cachedTokens == replyA.promptTokens) == (fillerType == 1) && liveB.cachedTokens < liveB.promptTokens,
            "B reused \(liveB.cachedTokens) of \(liveB.promptTokens) tokens (A's prompt \(replyA.promptTokens))")
        try require(
            restartB.cachedTokens == restartB.promptTokens,
            "the restart did not restore B's whole prompt (\(restartB.cachedTokens) of \(restartB.promptTokens))")
        try require(
            restartB.outputTokens == liveB.outputTokens,
            "the reply after a restart differs: \(restartB.outputTokens ?? []) vs \(liveB.outputTokens ?? [])")
        try require(restartAfter.0 == liveAfter.0, "the state after the reply differs")
        try require(restartAfter.1 == liveAfter.1, "the logits after the reply differ")
    }
}

/// Only exact states are saved: a fast turn A, a fast turn B continuing it live (B's prompt read after A's fast reply
/// rows), then an exact B must not restore past A's prompt, in the same server or after a restart (a new backend on the
/// same cache folder), and must reply and end bitwise as a cold exact B. `keep` on the fast B pins nothing. The expert
/// budget is held under the routed experts' size so fast levels skip misses. Before the fix B's whole prompt was saved
/// over A's fast rows and the exact request restored it.
func familyExactNeverRestoresFastState(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let name = family == .glm ? "glm" : "deepseek"
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-fast-state-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try syntheticFamilyGGUF(family, in: folder)
    var seed: UInt64 = 21
    let text = String(
        (0..<2_470).map { _ in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Character(UnicodeScalar(UInt8(97 + (seed >> 33) % 26)))
        })
    let a = [["role": "user", "content": text]]
    // fillerType 4: a reply re-encodes to its own tokens, so B continues A live.
    let tokenizer = try tinyFamilyTokenizer(
        "{% for m in messages %}{{ m.content }}{% endfor %}", fillers: 2_048 - 261, fillerType: 4)
    func session(_ cache: String) throws -> (FamilyServingBackend, FamilyRunner) {
        let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 1)
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 8_192, ledger: ledger) { _ in }
        let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
        backend.promptCache = try PromptCache(
            root: folder.appendingPathComponent(cache, isDirectory: true), epoch: "fast-state", entries: 8,
            entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger, capacity: 8_192)
        backend.epoch = "fast-state"
        return (backend, model as! FamilyRunner)
    }
    func serve(
        _ backend: FamilyServingBackend, _ messages: [[String: String]], mode: String, keep: String? = nil, teacher: [Int]? = nil,
        maximumTokens: Int = 13
    ) throws -> GenerationStatistics {
        var body: [String: Any] = [
            "model": "family", "messages": messages, "max_tokens": maximumTokens, "temperature": 0, "reasoning_effort": "none",
            "origami_return_tokens": true, "origami_mode": mode,
        ]
        if let teacher { body["origami_teacher_tokens"] = teacher }
        var request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: body), contextTier: 8_192, defaults: ServingDefaults())
        request.keep = keep
        return try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
    }
    func digest(_ runner: FamilyRunner) -> String {
        var hash = SHA256()
        for part in runner.stateParts(at: runner.position) {
            hash.update(bufferPointer: UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes))
        }
        return hash.finalize().description
    }
    func after(_ runner: FamilyRunner) throws -> (String, [UInt32]) {
        (digest(runner), try (runner as! FamilyModel).forward([97]).map(\.bitPattern))
    }
    // Every requirement is judged on its own, so each hole shows by its own message.
    var problems: [String] = []
    for restart in [false, true] {
        func require(_ condition: Bool, _ what: String) {
            if !condition { problems.append("\(name) exact after fast turns\(restart ? " and a restart" : ""): " + what) }
        }
        var fastPrompt = ""
        var kept = true
        let cacheName = "cache-\(restart)"
        var promptA = 0
        var b: [[String: String]] = []
        var exact: (GenerationStatistics, (String, [UInt32]))?
        try autoreleasepool {
            let (backend, runner) = try session(cacheName)
            // A's reply forced to fillers that re-encode as themselves (a sampled one may hold a lone byte), decoded fast.
            let replyA = try serve(backend, a, mode: "fastest", teacher: Array(400..<413))
            promptA = replyA.promptTokens
            b =
                a + [
                    ["role": "assistant", "content": try tokenizer.decode(replyA.outputTokens ?? [])], ["role": "user", "content": "xyz"],
                ]
            // One token: the state after it is B's prompt read after A's fast rows (the reply's token is not read).
            let fastB = try serve(backend, b, mode: "fastest", keep: "fast", maximumTokens: 1)
            require(fastB.cachedTokens > promptA, "the fast B did not continue A live (\(fastB.cachedTokens) of \(promptA))")
            fastPrompt = digest(runner)
            kept = backend.promptCache?.keptSessions.isEmpty == true
            if !restart { exact = (try serve(backend, b, mode: "exact"), try after(runner)) }
        }
        if restart {
            try autoreleasepool {
                let (backend, runner) = try session(cacheName)
                exact = (try serve(backend, b, mode: "exact"), try after(runner))
            }
        }
        let (cold, coldAfter) = try autoreleasepool {
            let (backend, runner) = try session("cold-\(restart)")
            return (try serve(backend, b, mode: "exact"), try after(runner))
        }
        // The fixture tests the hole only while the fast rows give B's prompt other bits than an exact read.
        let coldPrompt = try autoreleasepool {
            let (backend, runner) = try session("cold-prompt-\(restart)")
            _ = try serve(backend, b, mode: "exact", maximumTokens: 1)
            return digest(runner)
        }
        require(fastPrompt != coldPrompt, "the fast turns left B's prompt state bitwise exact, so nothing is tested")
        guard let exact else { throw EngineError("\(name): no exact reply") }
        let (reply, state) = exact
        require(reply.cachedTokens <= promptA, "the exact B restored \(reply.cachedTokens) tokens (A's prompt \(promptA))")
        require(cold.cachedTokens == 0, "the cold B reused \(cold.cachedTokens) tokens")
        require(reply.outputTokens == cold.outputTokens, "the reply differs from a cold exact one")
        require(state.0 == coldAfter.0 && state.1 == coldAfter.1, "the state after the reply differs from a cold exact one")
        require(kept, "keep pinned a state read after fast rows")
    }
    if !problems.isEmpty { throw EngineError(problems.joined(separator: "; ")) }
}
