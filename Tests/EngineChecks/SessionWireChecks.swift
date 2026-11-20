import Foundation
import Metal
import OrigamiEngine

/// Sessions and the wires: tool choice, reasoning levels, wire gaps and replies that used to be empty.
func sessionWireChecks() -> [(String, () throws -> Void)] {
    [
        ("tool_choice_prompt_prefix_stays", toolChoicePromptPrefixStays),
        ("tool_choice_required_decodes_a_call", toolChoiceRequiredDecodesACall),
        ("tool_choice_unmet_is_a_client_error", toolChoiceUnmetIsAClientError),
        ("empty_replies_answer_with_reasoning", emptyRepliesAnswerWithReasoning),
        ("reasoning_levels_distinct_per_model", reasoningLevelsDistinctPerModel),
        ("family_two_chats_keep_their_saves", familyTwoChatsKeepTheirSaves),
        ("family_stop_saves_the_conversation", familyStopSavesTheConversation),
        ("wire_count_tokens", wireCountTokens),
        ("wire_responses_reasoning_and_custom_tools", wireResponsesReasoningAndCustomTools),
        ("wire_tool_result_images", wireToolResultImages),
        ("wire_stop_sequence_reported", wireStopSequenceReported),
        ("wire_invalid_tool_named", wireInvalidToolNamed),
    ]
}

private func check(_ condition: Bool, _ what: @autoclosure () -> String) throws {
    if !condition { throw EngineError("Sessions and wires: " + what()) }
}
private func request(_ fields: [String: Any], context: Int = 8_192) throws -> ServingRequest {
    var body: [String: Any] = ["model": "m", "messages": [["role": "user", "content": "Hi"]]]
    for (key, value) in fields { body[key] = value }
    return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: context)
}
private func tool(_ name: String) -> [String: Any] {
    [
        "type": "function",
        "function": [
            "name": name, "description": name,
            "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]],
        ],
    ]
}
private func failure(_ operation: () throws -> Void) -> ServingFailure? {
    do {
        try operation()
        return nil
    } catch {
        return ServingFailure.from(error)
    }
}

/// J29: `tool_choice` required or named changes only the end of the last user turn: the system prompt, the tools and
/// every earlier turn are token-identical to `auto`, so a saved prefix stays valid when a client toggles it.
private func toolChoicePromptPrefixStays() throws {
    let tokenizer = try tinyFamilyTokenizer(
        "{% for t in tools %}[{{ t.function.name }}]{% endfor %}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>"
    )
    let earlier: [[String: Any]] = [
        ["role": "system", "content": "Be brief."], ["role": "user", "content": "Weather in Rome?"],
        ["role": "assistant", "content": "Sunny."],
    ]
    let messages = earlier + [["role": "user", "content": "And Oslo?"]]
    func ids(_ choice: Any?) throws -> [Int] {
        var fields: [String: Any] = ["messages": messages, "tools": [tool("weather"), tool("lookup")], "think": false]
        if let choice { fields["tool_choice"] = choice }
        return try ServingPrompt.encode(request(fields), tokenizer: tokenizer)
    }
    let plain = try ids(nil)
    let start =
        try ServingPrompt.encode(
            request(["messages": earlier + [["role": "user", "content": ""]], "tools": [tool("weather"), tool("lookup")], "think": false]),
            tokenizer: tokenizer
        ).count - tokenizer.encode("<assistant>").count
    for choice in ["required", ["type": "function", "function": ["name": "lookup"]]] as [Any] {
        let chosen = try ids(choice)
        let shared = TokenizerStack.commonTokenPrefix(plain, chosen)
        try check(shared >= start && chosen.count > plain.count, "\(choice): shares \(shared) of \(start) tokens before the last user turn")
        let text = try tokenizer.decode(chosen)
        try check(
            text.hasSuffix(
                "And Oslo?\n\n"
                    + (choice is String ? "You must call one of the provided tools." : "You must call the provided tool named lookup.")
                    + "<assistant>") && text.contains("[weather][lookup]"), "\(choice): \(text)")
    }
    // No user turn at all: the instruction is a user turn of its own at the end.
    let only = try ServingPrompt.encode(
        request(["messages": [["role": "system", "content": "S"]], "tools": [tool("weather")], "tool_choice": "required"]),
        tokenizer: tokenizer)
    try check(try tokenizer.decode(only).hasSuffix("<user>You must call one of the provided tools.<assistant>"), "system-only chat")
}

/// J29 with N7: on the synthetic GLM and DeepSeek models, `tool_choice` required makes every reply open a tool call,
/// and a named choice opens that tool's call, even beside JSON mode.
private func toolChoiceRequiredDecodesACall() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-tool-choice-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for family in [ModelFamily.glm, .deepseek] {
        let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
        let tokenizer = try tinyFamilyTokenizer(
            "{% for m in messages %}{{ m.content }}{% endfor %}", fillerTexts: jsonFillers(weights.geometry.vocab - 261))
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed), context: 2_048) { _ in }
        let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
        let grammar = ToolCallGrammar.of(family)
        for (choice, opener) in [
            ("required" as Any, try request(["tools": [tool("weather")], "tool_choice": "required"]).toolOpener(grammar)!),
            (
                ["type": "function", "function": ["name": "lookup"]],
                try request([
                    "tools": [tool("weather"), tool("lookup")], "tool_choice": ["type": "function", "function": ["name": "lookup"]],
                ]).toolOpener(grammar)!
            ),
        ] {
            try check(
                opener.hasPrefix(family == .glm ? "<tool_call>" : "<｜DSML｜ calls>")
                    && (choice is String || opener.hasSuffix("lookup") || opener.hasSuffix("lookup\">")),
                "\(family) opener \(opener)")
            for seed in 0..<4 {
                var fields: [String: Any] = [
                    "tools": [tool("weather"), tool("lookup")], "tool_choice": choice, "max_tokens": 96, "reasoning_effort": "none",
                    "seed": seed, "temperature": 1,
                ]
                if seed == 3 { fields["response_format"] = ["type": "json_object"] }
                final class Text: @unchecked Sendable { var value = "" }
                let text = Text()
                _ = try backend.prepare(request(fields, context: 2_048), cancellation: GenerationControl()).run {
                    text.value += $0
                    return true
                }
                try check(text.value.hasPrefix(opener), "\(family) seed \(seed) \(choice): \(text.value.prefix(80))")
            }
        }
    }
}

/// J29: a reply that holds no valid call (text, or a call the parser could not read) under a required or named choice is a 422
/// naming what was asked, not a 500; a reply cut by `max_tokens` still finishes as `length`.
private func toolChoiceUnmetIsAClientError() throws {
    let named = try request([
        "tools": [tool("weather"), tool("lookup")], "tool_choice": ["type": "function", "function": ["name": "lookup"]], "think": false,
    ])
    try check(named.tools.count == 2 && named.definitions.map(\.name) == ["lookup"], "a named choice hid the other tools from the prompt")
    let stream = ServingStream(request: named)
    _ = try stream.push("Sure.")
    let error = failure { _ = try stream.finish(reason: "stop") }
    try check(
        error?.status == 422 && error?.code == "tool_choice_unmet" && error?.description.contains("the tool lookup") == true,
        "named: \(String(describing: error))")
    let required = ServingStream(request: try request(["tools": [tool("weather")], "tool_choice": "required", "think": false]))
    _ = try required.push("No call")
    let other = failure { _ = try required.finish(reason: "stop") }
    try check(other?.status == 422 && other?.description.contains("one of the tools") == true, "required: \(String(describing: other))")
    let cut = ServingStream(request: try request(["tools": [tool("weather")], "tool_choice": "required", "think": false]))
    _ = try cut.push("<tool_call>\n<function=wea")
    try check(try cut.finish(reason: "length").reason == "length", "a cut reply failed")
}

/// J62: a turn that ends inside its think block, or whose only output is an empty tool-call block, answers with its
/// reasoning and says so (`empty_answer`); with no reasoning either it stays empty and says `nothing`; a cut reply and
/// a reply with text are as before.
private func emptyRepliesAnswerWithReasoning() throws {
    func finish(_ pieces: [String], reason: String = "stop", effort: String = "medium") throws -> (ServingStream, String, String) {
        let stream = ServingStream(request: try request(["tools": [tool("weather")], "reasoning_effort": effort]))
        var content = ""
        for piece in pieces { for delta in try stream.push(piece) { content += delta["content"] as? String ?? "" } }
        let end = try stream.finish(reason: reason)
        for delta in end.deltas { content += delta["content"] as? String ?? "" }
        return (stream, content, end.reason)
    }
    let unclosed = try finish(["Thinking about ", "Rome."])
    try check(
        unclosed.1 == "Thinking about Rome." && unclosed.0.text == unclosed.1 && unclosed.0.emptyAnswer == "reasoning"
            && unclosed.0.message["content"] as? String == unclosed.1 && unclosed.2 == "stop", "unclosed think: \(unclosed.1)")
    let emptyCall = try finish(["Plan</think>\n\n", "<tool_call>\n", "\n</tool_call>"])
    try check(
        emptyCall.1 == "Plan" && emptyCall.0.calls.isEmpty && emptyCall.0.emptyAnswer == "reasoning", "empty call: \(emptyCall.1)")
    let nothing = try finish(["<tool_call>\n</tool_call>"], effort: "none")
    try check(nothing.1.isEmpty && nothing.0.emptyAnswer == "nothing", "empty call without reasoning: \(nothing.1)")
    let cut = try finish(["Still thinking"], reason: "length")
    try check(cut.1.isEmpty && cut.0.emptyAnswer == nil && cut.2 == "length", "a cut reply changed")
    let answered = try finish(["Plan</think>\n\nSunny."])
    try check(answered.1 == "Sunny." && answered.0.emptyAnswer == nil, "an answered reply changed: \(answered.1)")
}

/// J56: each family's listed levels are exactly the distinct renders of its own chat template (the repo's copies of the
/// models' templates), first name of each; every accepted name renders (DeepSeek's template refuses `xhigh` by name).
private func reasoningLevelsDistinctPerModel() throws {
    for (model, family) in [("qwen3.8-flash-next", ModelFamily.qwen), ("glm-5.3-flash", .glm), ("deepseek-v4.1-flash", .deepseek)] {
        let template = try String(contentsOf: CheckFixtures.file("models/\(model)/chat_template.jinja"), encoding: .utf8)
        let tokenizer = try templateTokenizer(template, specials: templateSpecials(template))
        var seen: [[Int]] = []
        var distinct: [String] = []
        for effort in ServingDefaults.efforts {
            let ids = try tokenizer.encodeChat(messages: [["role": .string("user"), "content": .string("Hi")]], reasoningEffort: effort)
            if !seen.contains(ids) {
                seen.append(ids)
                distinct.append(effort)
            }
        }
        // Each listed name is one distinct render; the name of a group may be a later alias (GLM lists `max` for the Max
        // that `medium` also renders).
        let listed = ServingDefaults.efforts(for: family)
        let renders = try listed.map {
            try tokenizer.encodeChat(messages: [["role": .string("user"), "content": .string("Hi")]], reasoningEffort: $0)
        }
        try check(
            listed.count == distinct.count && Set(renders).count == renders.count && renders.allSatisfy(seen.contains),
            "\(model): renders \(distinct) apart, lists \(listed)")
        let models = LocalHTTPServer.modelEntry("m", family: family, vision: false, context: 8_192, origami: [:])
        try check(
            models["reasoning_efforts"] as? [String] == listed,
            "\(model): /v1/models lists \(String(describing: models["reasoning_efforts"]))")
    }
}

/// A synthetic Family serving session over `cacheRoot` (`entries` regular save slots), as `family_serving_restore_*`
/// builds it; replies decode to text that encodes back to the same tokens.
private final class FamilyChatFixture {
    let family: ModelFamily
    let folder: URL
    let path: URL
    let tokenizer: TokenizerStack
    init(_ family: ModelFamily) throws {
        self.family = family
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-chats-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        path = try syntheticFamilyGGUF(family, in: folder)
        // Turn markers keep a reply's first and last tokens from merging with the text around it when the chat is encoded again.
        tokenizer = try tinyFamilyTokenizer(
            "{% for m in messages %}{% if m.role == 'user' %}U:{{ m.content }}A:{% else %}{{ m.content }}{% endif %}{% endfor %}",
            fillers: 2_048 - 261, fillerType: 4)
    }
    deinit { try? FileManager.default.removeItem(at: folder) }
    func session(entries: Int, device: MTLDevice, cache: String = "cache") throws -> FamilyServingBackend {
        let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 1)
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: 8 << 30, context: 8_192, ledger: ledger) { _ in }
        let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
        backend.promptCache = try PromptCache(
            root: folder.appendingPathComponent(cache, isDirectory: true), epoch: "family-chats", entries: entries,
            entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger, capacity: 8_192)
        backend.epoch = "family-chats"
        return backend
    }
    static func text(_ count: Int, seed: UInt64) -> String {
        var state = seed
        return String(
            (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Character(UnicodeScalar(UInt8(97 + (state >> 33) % 26)))
            })
    }
    /// One turn: `messages` plus a new user turn, served; the reply appended. The prompt tokens the first try restored
    /// or continued. The synthetic model can write byte tokens that do not decode to text, so a reply that would not
    /// encode back to its own tokens is drawn again with the next seed (the prompt is whole in memory by then).
    func turn(_ backend: FamilyServingBackend, _ messages: inout [[String: String]], seed: UInt64) throws -> (cached: Int, before: Int) {
        let before = messages.isEmpty ? 0 : try ServingPrompt.encode(request(messages), tokenizer: tokenizer).count
        messages.append(["role": "user", "content": Self.text(600, seed: seed)])
        let sent = try ServingPrompt.encode(request(messages), tokenizer: tokenizer)
        var cached: Int?
        for draw in 0..<64 {
            let stats = try backend.prepare(request(messages, seed: draw), cancellation: GenerationControl()).run { _ in true }
            cached = cached ?? stats.cachedTokens
            let reply = (stats.outputTokens ?? []).filter { $0 != 259 }
            let text = try tokenizer.decode(reply)
            let again = try ServingPrompt.encode(
                request(messages + [["role": "assistant", "content": text], ["role": "user", "content": "x"]]), tokenizer: tokenizer)
            if again.starts(with: sent + reply) {
                messages.append(["role": "assistant", "content": text])
                return (cached!, before)
            }
        }
        throw EngineError("Sessions and wires: no reply of 64 encodes back to its tokens")
    }
    func request(_ messages: [[String: String]], seed: Int = 0) throws -> ServingRequest {
        let body: [String: Any] = [
            "model": "family", "messages": messages, "max_tokens": 8, "temperature": 1, "seed": seed, "reasoning_effort": "none",
            "origami_return_tokens": true,
        ]
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8_192, defaults: ServingDefaults())
    }
}

/// J31: a conversation's earlier save is the preferred victim of its next save, not an unconditional one. Three slots:
/// two or three conversations alternating four turns each re-read none of their turns from the start; with a slot free,
/// an edited last user message still restores the conversation's earlier prompt; four conversations in turn lose only
/// the least recently used one, never another conversation's newest save.
private func familyTwoChatsKeepTheirSaves() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try FamilyChatFixture(.glm)
    var results: [String] = []
    var missed = 0
    for count in [2, 3] {
        let backend = try fixture.session(entries: 3, device: device, cache: "chats-\(count)")
        var chats = [[[String: String]]](repeating: [], count: count)
        var misses: [String] = []
        for round in 0..<4 {
            for chat in 0..<count {
                let (cached, before) = try fixture.turn(backend, &chats[chat], seed: UInt64(100 * round + chat))
                // A conversation's last reply token was never read, so a whole restore is one short of the history.
                if round > 0 && cached < before - 1 { misses.append("chat \(chat) turn \(round + 1): \(cached) of \(before)") }
            }
        }
        results.append("\(count) chats: \(misses.count) of \(3 * count) turns missed \(misses)")
        missed += misses.count
    }
    try check(missed == 0, results.joined(separator: "; "))

    // An edited last message: the second turn's save took a free slot, so the first turn's prompt is still there.
    let edits = try fixture.session(entries: 3, device: device, cache: "edit")
    var chat: [[String: String]] = []
    _ = try fixture.turn(edits, &chat, seed: 1)
    let first = try ServingPrompt.encode(fixture.request(Array(chat.prefix(1))), tokenizer: fixture.tokenizer).count
    _ = try fixture.turn(edits, &chat, seed: 2)
    chat.removeLast(2)
    let (edited, _) = try fixture.turn(edits, &chat, seed: 3)
    try check(edited >= first, "an edited last message restored \(edited) tokens, not the first turn's \(first)")

    // Four conversations, three slots: after one turn each, the next turns of the three most recent restore whole and
    // only the least recently used one reads again.
    let four = try fixture.session(entries: 3, device: device, cache: "four")
    var chats = [[[String: String]]](repeating: [], count: 4)
    for n in 0..<4 { _ = try fixture.turn(four, &chats[n], seed: UInt64(500 + n)) }
    var lost: [Int] = []
    for n in [1, 2, 3, 0] {
        let (cached, before) = try fixture.turn(four, &chats[n], seed: UInt64(600 + n))
        if cached < before - 1 { lost.append(n) }
    }
    try check(lost == [0], "four conversations, three slots: \(lost) read again, want only the least recently used (0)")
}

/// J60: GLM and DeepSeek save the live conversation when their server stops (`origami stop`, an unload and a model
/// switch all stop it): after a turn and a stop, a new process over the same saved prompts restores the whole
/// conversation, where without the stop's save it restores only the turn's prompt. The server calls the backend's stop
/// work once its requests have ended.
private func familyStopSavesTheConversation() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    for family in [ModelFamily.glm, .deepseek] {
        let fixture = try FamilyChatFixture(family)
        var restored: [Bool: (cached: Int, before: Int)] = [:]
        for stops in [false, true] {
            var chat: [[String: String]] = []
            let cache = "stop-\(stops)"
            try autoreleasepool {
                let backend = try fixture.session(entries: 3, device: device, cache: cache)
                _ = try fixture.turn(backend, &chat, seed: 7)
                if stops { backend.serverStopped() }
            }
            restored[stops] = try autoreleasepool {
                try fixture.turn(try fixture.session(entries: 3, device: device, cache: cache), &chat, seed: 8)
            }
        }
        let saved = restored[true]!
        let unsaved = restored[false]!
        try check(
            saved.cached >= saved.before - 1 && unsaved.cached < unsaved.before - 1,
            "\(family): after a stop the next start restored \(saved.cached) of \(saved.before) (without the save \(unsaved.cached))")
    }
    final class Stopping: ServingBackendStopping, @unchecked Sendable {
        var stopped = 0
        func status() -> ServingStatus { ServingStatus(version: "check", ready: true) }
        func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
            throw ServingFailure("unused")
        }
        func serverStopped() { stopped += 1 }
    }
    let backend = Stopping()
    var configuration = LocalHTTPConfiguration()
    configuration.port = 0
    let server = try LocalHTTPServer(
        configuration: configuration, backend: backend, ledger: try MemoryLedger(ceiling: 1 << 30, transientMargin: 1 << 20))
    try server.start()
    try check(server.stopAndJoin() && backend.stopped == 1, "the server's stop did not call the backend's stop work")
}

private func translated(_ body: [String: Any], _ format: WireFormat, vision: Bool = false) throws -> [String: Any] {
    let data = try WireTranslation.chatBody(JSONSerialization.data(withJSONObject: body), format: format, vision: vision)
    return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
}

/// J61: `POST /v1/messages/count_tokens` answers Anthropic's `input_tokens`: the prompt tokens the same body renders to
/// through `/v1/messages` (template, tools, system), with no reply; its errors, a template's refusal too, are
/// Anthropic-shaped 400s.
private func wireCountTokens() throws {
    final class Counting: ServingPromptCounting, @unchecked Sendable {
        let tokenizer: TokenizerStack
        init(_ tokenizer: TokenizerStack) { self.tokenizer = tokenizer }
        func status() -> ServingStatus { ServingStatus(version: "check", ready: true) }
        func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
            throw ServingFailure("unused")
        }
        func promptTokens(_ request: ServingRequest) throws -> Int { try ServingPrompt.encode(request, tokenizer: tokenizer).count }
    }
    let tokenizer = try tinyFamilyTokenizer(
        "{% for t in tools %}[{{ t.function.name }}]{% endfor %}{% for m in messages %}"
            + "{% if m.content == 'boom' %}{{ raise_exception('No user query found in messages.') }}{% endif %}"
            + "<{{ m.role }}>{{ m.content }}{% endfor %}")
    var configuration = LocalHTTPConfiguration()
    configuration.port = 0
    let server = try LocalHTTPServer(
        configuration: configuration, backend: Counting(tokenizer), ledger: try MemoryLedger(ceiling: 1 << 30, transientMargin: 1 << 20))
    try server.start()
    defer { _ = server.stopAndJoin() }
    func post(_ body: [String: Any]) throws -> (Int, [String: Any]) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(server.boundPort ?? 0)/v1/messages/count_tokens")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        final class Box: @unchecked Sendable { var data = Data(), status = 0 }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { data, response, _ in
            box.data = data ?? Data()
            box.status = (response as? HTTPURLResponse)?.statusCode ?? 0
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 10)
        return (box.status, (try? JSONSerialization.jsonObject(with: box.data)) as? [String: Any] ?? [:])
    }
    let body: [String: Any] = [
        "model": "m", "system": "Be brief.", "messages": [["role": "user", "content": "Weather in Rome?"]],
        "tools": [["name": "weather", "description": "w", "input_schema": ["type": "object"]]],
    ]
    let expected = try ServingPrompt.encode(
        ServingRequest.parse(
            WireTranslation.chatBody(
                JSONSerialization.data(withJSONObject: body.merging(["max_tokens": 1]) { $1 }),
                format: .anthropic), contextTier: 8_192), tokenizer: tokenizer
    ).count
    let (status, answer) = try post(body)
    try check(
        status == 200 && answer["input_tokens"] as? Int == expected && expected > 0, "count_tokens: \(status) \(answer), want \(expected)")
    let (bad, error) = try post(["model": "m", "messages": "x"])
    try check(bad == 400 && error["type"] as? String == "error", "count_tokens error: \(bad) \(error)")
    // A conversation the template refuses is the client's mistake: a 400 in Anthropic's shape, not a server fault.
    let (refused, refusal) = try post(["model": "m", "messages": [["role": "user", "content": "boom"]]])
    try check(
        refused == 400 && refusal["type"] as? String == "error"
            && ((refusal["error"] as? [String: Any])?["message"] as? String)?.hasPrefix("Invalid conversation") == true,
        "count_tokens template refusal: \(refused) \(refusal)")
}

/// J61: Responses `reasoning` input items are the next assistant turn's reasoning, as the chat wire's
/// `reasoning_content` (content text, else the summary Origami writes), and no later turn's after a user message or a call
/// output (a turn stopped while thinking); a `custom` tool is a 400 that names it.
private func wireResponsesReasoningAndCustomTools() throws {
    let body: [String: Any] = [
        "model": "m",
        "input": [
            ["role": "user", "content": "Hi"],
            ["type": "reasoning", "summary": [["type": "summary_text", "text": "Plan it"]]],
            ["type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "c1", "output": "sun"],
            ["type": "reasoning", "content": [["type": "reasoning_text", "text": "Now answer"]], "summary": []],
            ["role": "assistant", "content": [["type": "output_text", "text": "Sunny"]]],
            ["role": "user", "content": "Thanks"],
        ],
        "tools": [["type": "function", "name": "weather", "parameters": ["type": "object"]]],
    ]
    let messages = try translated(body, .responses)["messages"] as? [[String: Any]] ?? []
    let assistants = messages.filter { $0["role"] as? String == "assistant" }
    try check(
        assistants.map { $0["reasoning_content"] as? String } == ["Plan it", "Now answer"]
            && (assistants.first?["tool_calls"] as? [[String: Any]])?.count == 1, "reasoning items: \(assistants)")
    _ = try ServingRequest.parse(JSONSerialization.data(withJSONObject: translated(body, .responses)), contextTier: 8_192)
    var custom = body
    custom["tools"] = [["type": "custom", "name": "apply_patch", "description": "free-form"]]
    let refused = failure { _ = try translated(custom, .responses) }
    try check(
        refused?.status == 400 && refused?.description.contains("custom tool apply_patch") == true,
        "custom tool: \(String(describing: refused))")
    let interruptions: [[String: Any]] = [
        ["role": "user", "content": "Stop"], ["type": "function_call_output", "call_id": "c0", "output": "late"],
    ]
    for interruption in interruptions {
        let replay: [String: Any] = [
            "model": "m",
            "input": [
                ["role": "user", "content": "Hi"], ["type": "reasoning", "summary": [["type": "summary_text", "text": "Half a thought"]]],
                interruption, ["role": "assistant", "content": "Ok"],
            ],
        ]
        let turns = (try translated(replay, .responses)["messages"] as? [[String: Any]] ?? []).filter {
            $0["role"] as? String == "assistant"
        }
        try check(turns.count == 1 && turns[0]["reasoning_content"] == nil, "stopped reasoning carried over: \(turns)")
    }
}

/// J61: an Anthropic `tool_result` holding images keeps them for a model with vision (a user turn after the results);
/// for one without, the result says one was left out.
private func wireToolResultImages() throws {
    let image: [String: Any] = ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": "iVBOR"]]
    let body: [String: Any] = [
        "model": "m", "max_tokens": 8,
        "messages": [
            ["role": "user", "content": "Look"],
            ["role": "assistant", "content": [["type": "tool_use", "id": "t1", "name": "shot", "input": [:]]]],
            [
                "role": "user",
                "content": [["type": "tool_result", "tool_use_id": "t1", "content": [["type": "text", "text": "Here"], image]]],
            ],
        ],
    ]
    let seeing = try translated(body, .anthropic, vision: true)["messages"] as? [[String: Any]] ?? []
    let tail = Array(seeing.suffix(2))
    try check(
        tail.first?["role"] as? String == "tool" && tail.first?["content"] as? String == "Here" && tail.last?["role"] as? String == "user"
            && ((tail.last?["content"] as? [[String: Any]])?.first?["type"] as? String) == "image_url", "vision: \(tail)")
    let blind = try translated(body, .anthropic)["messages"] as? [[String: Any]] ?? []
    try check(
        blind.last?["role"] as? String == "tool" && (blind.last?["content"] as? String)?.contains("an image was left out") == true,
        "text only: \(String(describing: blind.last))")
}

/// J61: a reply ended by a `stop` string says so on the Anthropic wire (`stop_reason` `stop_sequence` with the string),
/// streamed and whole; the filter names the string; other wires keep `stop`.
private func wireStopSequenceReported() throws {
    let filter = try StopSequenceFilter(stops: ["END", "HALT"], reasoningOpen: false)
    let shown = filter.push("one HALT two END")
    try check(shown == "one " && filter.matched == "HALT", "the filter: \(shown) \(String(describing: filter.matched))")
    let usage: [String: Any] = [
        "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5, "prompt_tokens_details": ["cached_tokens": 0],
    ]
    let anthropic = WireWriter(format: .anthropic, id: "msg_1", created: 0, model: "m", includeUsage: false)
    let body = anthropic.body(message: ["role": "assistant", "content": "one "], reason: "stop", usage: usage, stopSequence: "HALT")
    try check(body["stop_reason"] as? String == "stop_sequence" && body["stop_sequence"] as? String == "HALT", "whole: \(body)")
    let plain = anthropic.body(message: ["role": "assistant", "content": "one"], reason: "stop", usage: usage)
    try check(plain["stop_reason"] as? String == "end_turn" && plain["stop_sequence"] is NSNull, "end of turn: \(plain)")
    let frames = try anthropic.close(reason: "stop", usage: usage, stopSequence: "HALT").map { String(decoding: $0, as: UTF8.self) }
        .joined()
    try check(frames.contains(#""stop_reason":"stop_sequence""#) && frames.contains(#""stop_sequence":"HALT""#), "streamed: \(frames)")
    let chat = WireWriter(format: .openAI, id: "c", created: 0, model: "m", includeUsage: false)
    let choice =
        (chat.body(message: ["role": "assistant", "content": "one "], reason: "stop", usage: usage, stopSequence: "HALT")["choices"]
        as? [[String: Any]])?.first
    try check(choice?["finish_reason"] as? String == "stop", "chat: \(String(describing: choice))")
}

/// J61: a refused tool declaration names the tool and what is wrong.
private func wireInvalidToolNamed() throws {
    let tools = [tool("weather"), tool("weather")]
    let repeated = failure { _ = try request(["tools": tools]) }
    try check(repeated?.status == 400 && repeated?.description.contains("\"weather\" (tools 2)") == true, "\(String(describing: repeated))")
    let unnamed = failure { _ = try request(["tools": [["type": "function", "function": ["description": "x"]]]]) }
    try check(unnamed?.description.contains("(tools 1)") == true, "\(String(describing: unnamed))")
}
