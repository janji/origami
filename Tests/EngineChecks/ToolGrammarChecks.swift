import CryptoKit
import Foundation
import OrigamiEngine

private func grammarAssert(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else { throw EngineError("Tool grammar: " + message) }
}

private func grammarTools() -> [[String: Any]] {
    [
        [
            "type": "function",
            "function": [
                "name": "read_file", "description": "Read a file",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"], "start_line": ["type": "integer"],
                    ], "required": ["path"],
                ],
            ],
        ],
        [
            "type": "function",
            "function": [
                "name": "patch", "description": "Patch",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "options": ["type": "object"], "tags": ["type": "array"], "dry": ["type": "boolean"], "note": ["type": "string"],
                        "code": ["type": "string"],
                    ],
                ],
            ],
        ],
    ]
}

private func grammarRequest(_ additions: [String: Any] = [:]) throws -> ServingRequest {
    var body: [String: Any] = [
        "model": "qwen3.8-flash-next:4bit", "messages": [["role": "user", "content": "Hello"]], "tools": grammarTools(), "think": false,
    ]
    for (key, value) in additions { body[key] = value }
    return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
}

/// The two calls every fixture below makes, as the client must receive them.
private let grammarExpected: [(String, [String: JSONValue], [String])] = [
    ("read_file", ["path": .string("src/雪 \"a\".txt"), "start_line": .int(12)], ["path", "start_line"]),
    (
        "patch",
        [
            "options": .object(["mode": .string("fast"), "nested": .object(["a": .array([.int(1), .double(2.5), .null])])]),
            "tags": .array([.string("x"), .string("ü")]), "dry": .bool(true), "note": .string("line1\nline2 "), "code": .string("123"),
        ],
        ["options", "tags", "dry", "note", "code"]
    ),
]

/// One reply pushed in `pieces`: its visible text, its calls, each call's streamed arguments and the finish reason.
private func grammarRun(_ family: ModelFamily, _ pieces: [String], reason: String = "stop", request: ServingRequest? = nil) throws
    -> (text: String, calls: [ParsedToolCall], streamed: [String], reason: String)
{
    let stream = ServingStream(request: try request ?? grammarRequest(), family: family)
    var deltas: [[String: Any]] = []
    for piece in pieces { deltas += try stream.push(piece) }
    let end = try stream.finish(reason: reason)
    deltas += end.deltas
    var text = ""
    var streamed: [String] = []
    for delta in deltas {
        text += delta["content"] as? String ?? ""
        for call in delta["tool_calls"] as? [[String: Any]] ?? [] {
            let index = call["index"] as! Int
            if index == streamed.count { streamed.append("") }
            streamed[index] += (call["function"] as? [String: Any])?["arguments"] as? String ?? ""
        }
    }
    try grammarAssert(text == stream.text, "streamed text differs from the reply's")
    return (text, stream.calls, streamed, end.reason)
}

/// GLM-5.3's `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` and DeepSeek-V4.1's DSML calls
/// become the same tool calls Qwen's do, typed as JSON (strings stay strings, JSON values parse), whole or split at
/// any character, several per reply; a block naming an undeclared tool or not in the grammar stays text, and so does
/// an unterminated or unreadable one, the reply finishing as usual.
func toolCallGrammarsParse() throws {
    let glm =
        "I'll read it.<tool_call>read_file<arg_key>path</arg_key><arg_value>src/雪 \"a\".txt</arg_value>"
        + "<arg_key>start_line</arg_key><arg_value>12</arg_value></tool_call>"
        + "<tool_call>patch\n<arg_key>options</arg_key>\n<arg_value>{\"mode\": \"fast\", \"nested\": {\"a\": [1, 2.5, null]}}</arg_value>\n"
        + "<arg_key>tags</arg_key><arg_value>[\"x\", \"ü\"]</arg_value><arg_key>dry</arg_key><arg_value>true</arg_value>"
        + "<arg_key>note</arg_key><arg_value>line1\nline2 </arg_value><arg_key>code</arg_key><arg_value>123</arg_value></tool_call>"
    func dsml(_ name: String, _ params: [(String, Bool, String)]) -> String {
        "<｜DSML｜ invoke name=\"\(name)\">\n"
            + params.map { "<｜DSML｜ parameter name=\"\($0.0)\" string=\"\($0.1)\">\($0.2)</｜DSML｜ parameter>" }
            .joined(separator: "\n") + "\n</｜DSML｜ invoke>"
    }
    let deepseek =
        "I'll read it.\n\n<｜DSML｜ calls>\n"
        + dsml("read_file", [("path", true, "src/雪 \"a\".txt"), ("start_line", false, "12")]) + "\n"
        + dsml(
            "patch",
            [
                ("options", false, "{\"mode\": \"fast\", \"nested\": {\"a\": [1, 2.5, null]}}"), ("tags", false, "[\"x\", \"ü\"]"),
                ("dry", false, "true"), ("note", true, "line1\nline2 "), ("code", true, "123"),
            ]) + "\n</｜DSML｜ calls>"
    for (family, raw, prose) in [(ModelFamily.glm, glm, "I'll read it."), (.deepseek, deepseek, "I'll read it.")] {
        let characters = raw.map(String.init)
        var splits: [[String]] = [[raw], characters]
        for cut in 1..<characters.count { splits.append([characters[..<cut].joined(), characters[cut...].joined()]) }
        for pieces in splits {
            let run = try grammarRun(family, pieces)
            try grammarAssert(
                run.text == prose && run.reason == "tool_calls" && run.calls.count == 2,
                "\(family) in \(pieces.count) pieces: \(run.text) \(run.calls)")
            for (call, (name, arguments, order)) in zip(run.calls, grammarExpected) {
                try grammarAssert(call.name == name && call.arguments == arguments && call.order == order, "\(family) call \(call)")
            }
            try grammarAssert(
                run.streamed == run.calls.map(\.inputJSON), "\(family) streamed \(run.streamed) for \(run.calls.map(\.inputJSON))")
        }
    }
    // A call without arguments, then text after it.
    let bare = try grammarRun(.glm, ["<tool_call>read_file</tool_call>ok"])
    try grammarAssert(bare.calls.map(\.inputJSON) == ["{}"] && bare.text == "ok", "GLM call without arguments: \(bare)")
    // An undeclared tool or a JSON-form block is the model's text; an empty block vanishes; a declared call after them parses.
    let undeclared = "<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>"
    let glmText = try grammarRun(
        .glm,
        [
            "a", undeclared, "<tool_call>{\"name\":\"read_file\"}</tool_call><tool_call>\n</tool_call>",
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>p</arg_value></tool_call>",
        ])
    try grammarAssert(
        glmText.text == "a" + undeclared + "<tool_call>{\"name\":\"read_file\"}</tool_call>" && glmText.calls.map(\.name) == ["read_file"],
        "GLM fallbacks: \(glmText.text) \(glmText.calls)")
    let dsUndeclared = "<｜DSML｜ calls>\n" + dsml("lookup", [("q", true, "x")]) + "\n</｜DSML｜ calls>"
    let dsText = try grammarRun(
        .deepseek, [dsUndeclared, "\n<｜DSML｜ calls>\n" + dsml("read_file", [("path", true, "p")]) + "\n</｜DSML｜ calls>"])
    try grammarAssert(
        dsText.text == dsUndeclared + "\n" && dsText.calls.map(\.inputJSON) == ["{\"path\":\"p\"}"],
        "DeepSeek fallbacks: \(dsText.text) \(dsText.calls)")
    // The `\n\n` before a block that turns out to be text stays in it.
    let dsSeparated = try grammarRun(.deepseek, ["x\n\n" + dsUndeclared])
    try grammarAssert(dsSeparated.text == "x\n\n" + dsUndeclared && dsSeparated.calls.isEmpty, "DeepSeek text block: \(dsSeparated.text)")
    // Without `string=` the declared type decides; a reply that ends after a whole invoke keeps its call.
    let unmarked = try grammarRun(
        .deepseek,
        [
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n<｜DSML｜ parameter name=\"start_line\">7</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        ])
    try grammarAssert(
        unmarked.calls.map(\.inputJSON) == ["{\"start_line\":7}"] && unmarked.reason == "tool_calls", "DeepSeek unmarked: \(unmarked)")
    // A marker contradicting the declared type loses to it (the client's schema holds); an untyped parameter keeps its marker.
    let marked = try grammarRun(
        .deepseek,
        [
            "<｜DSML｜ calls>\n"
                + dsml(
                    "patch",
                    [
                        ("note", false, "123"), ("dry", true, "true"), ("code", false, "\"q\""),
                        ("x", true, "12"), ("y", false, "2.5"),
                    ]) + "\n</｜DSML｜ calls>"
        ].joined().map(String.init))
    try grammarAssert(
        marked.calls.map(\.inputJSON) == ["{\"note\":\"123\",\"dry\":true,\"code\":\"\\\"q\\\"\",\"x\":\"12\",\"y\":2.5}"]
            && marked.streamed == marked.calls.map(\.inputJSON), "DeepSeek markers: \(marked.calls.map(\.inputJSON)) \(marked.streamed)")
    // One call when parallel calls are off.
    let single = try grammarRun(.deepseek, [deepseek], request: try grammarRequest(["parallel_tool_calls": false]))
    try grammarAssert(single.calls.map(\.name) == ["read_file"], "DeepSeek single call: \(single.calls)")
    // Unterminated: the block is the reply's text, and the reply ends as it ended.
    for (family, cut) in [(ModelFamily.glm, String(glm.prefix(60))), (.deepseek, String(deepseek.prefix(80)))] {
        for reason in ["stop", "length"] {
            let open = try grammarRun(family, [cut], reason: reason)
            try grammarAssert(
                open.text == cut && open.calls.isEmpty && open.streamed.isEmpty && open.reason == reason, "\(family) unterminated: \(open)")
        }
    }
    // A call the grammar cannot read (a repeated parameter) is text too, and none of it streamed as a call (its start and
    // arguments wait for the block to parse); a later call is still a call, at index 0. Text before a call streams at once.
    let repeated =
        "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>"
    let valid = "<tool_call>\n<function=read_file>\n<parameter=path>\nc\n</parameter>\n</function>\n</tool_call>"
    for pieces in [["Sure. " + repeated + " Then:" + valid], ("Sure. " + repeated + " Then:" + valid).map(String.init)] {
        let twice = try grammarRun(.qwen, pieces)
        try grammarAssert(
            twice.text == "Sure. " + repeated + " Then:" && twice.calls.map(\.inputJSON) == ["{\"path\":\"c\"}"]
                && twice.streamed == ["{\"path\":\"c\"}"] && twice.reason == "tool_calls",
            "Repeated parameter: \(twice)")
    }
    let early = ServingStream(request: try grammarRequest(), family: .qwen)
    let first = try early.push("Sure. <tool_call>\n<function=read_file>\n<parameter=path>\na")
    try grammarAssert(
        first.count == 1 && first[0]["content"] as? String == "Sure. ",
        "Text before a call waited, or a call streamed before it parsed: \(first)")
    // Qwen's grammar is unchanged by the others' tags.
    let qwen = try grammarRun(.qwen, ["<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>"])
    try grammarAssert(qwen.calls.map(\.inputJSON) == ["{\"path\":\"a\"}"], "Qwen: \(qwen.calls)")
}

/// The real GLM-5.3 and DeepSeek-V4.1 chat templates (Tests/Fixtures/models) render a tool round trip: the tools,
/// the assistant's call in the family's own syntax with typed arguments, and the result. The rendered call parses
/// back into the same call, so what the model is shown and what the server reads are one grammar.
func familyToolTemplateRoundTrip() throws {
    let arguments = "{\"path\":\"src/雪.txt\",\"start_line\":12,\"options\":{\"deep\":[1,true]}}"
    let messages: [[String: Any]] = [
        ["role": "system", "content": "Be brief."], ["role": "user", "content": "Read it"],
        [
            "role": "assistant", "content": "",
            "tool_calls": [["id": "call_1", "type": "function", "function": ["name": "read_file", "arguments": arguments]]],
        ],
        ["role": "tool", "tool_call_id": "call_1", "content": "file body ü"], ["role": "user", "content": "Thanks"],
    ]
    let tools: [[String: Any]] = [
        [
            "type": "function",
            "function": [
                "name": "read_file", "description": "Read a file",
                "parameters": [
                    "type": "object",
                    "properties": ["path": ["type": "string"], "start_line": ["type": "integer"], "options": ["type": "object"]],
                ],
            ],
        ]
    ]
    let decoder = JSONDecoder()
    let templateMessages = try decoder.decode([[String: TemplateValue]].self, from: JSONSerialization.data(withJSONObject: messages))
    let templateTools = try decoder.decode([[String: TemplateValue]].self, from: JSONSerialization.data(withJSONObject: tools))
    let cases: [(ModelFamily, String, [String], String, String)] = [
        // Arguments render in key order; the values' JSON spacing is the template engine's, so the parse back checks them.
        (
            .glm, "glm-5.3-flash",
            [
                "<tool_call>read_file<arg_key>", "<arg_key>path</arg_key><arg_value>src/雪.txt</arg_value>",
                "<arg_key>start_line</arg_key><arg_value>12</arg_value>", "<arg_key>options</arg_key><arg_value>{",
                "<|observation|><tool_response>file body ü</tool_response>", "\"name\": \"read_file\"",
            ], "<tool_call>", "</tool_call>"
        ),
        (
            .deepseek, "deepseek-v4.1-flash",
            [
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n<｜DSML｜ parameter name=\"",
                "<｜DSML｜ parameter name=\"path\" string=\"true\">src/雪.txt</｜DSML｜ parameter>",
                "<｜DSML｜ parameter name=\"start_line\" string=\"false\">12</｜DSML｜ parameter>", "<tool_result>file body ü</tool_result>",
                "<｜DSML｜ parameter name=\"options\" string=\"false\">{", "\"name\": \"read_file\"",
            ],
            "<｜DSML｜ calls>", "</｜DSML｜ calls>"
        ),
    ]
    for (family, folder, expected, open, close) in cases {
        let template = try String(contentsOf: CheckFixtures.file("models/\(folder)/chat_template.jinja"), encoding: .utf8)
        let stack = try tinyFamilyTokenizer(template)
        let rendered = try stack.decode(try stack.encodeChat(messages: templateMessages, tools: templateTools, reasoningEffort: "high"))
        for part in expected { try grammarAssert(rendered.contains(part), "\(family) template lacks \(part) in:\n\(rendered)") }
        // The last block: the system prompt's description of the syntax holds an example of its own.
        guard let start = rendered.range(of: open, options: .backwards),
            let end = rendered.range(of: close, range: start.upperBound..<rendered.endIndex)
        else {
            throw EngineError("Tool grammar: \(family) template rendered no call block")
        }
        let run = try grammarRun(
            family, [String(rendered[start.lowerBound..<end.upperBound])], request: try grammarRequest(["tools": tools]))
        try grammarAssert(
            run.calls.count == 1
                && JSONValue.object(run.calls[0].arguments)
                    == JSONValue.object([
                        "path": .string("src/雪.txt"), "start_line": .int(12), "options": .object(["deep": .array([.int(1), .bool(true)])]),
                    ]),
            "\(family) rendered call parses as \(run.calls)")
    }
}

/// The next turn continues what the model generated: a reply in the family's own format, served (think and tool
/// splitting, the OpenAI message a client stores) and sent back with the next turn, renders as the prompt plus that
/// reply, token for token, so the next request reads only what is new. Per real template (Tests/Fixtures/models): a tool
/// turn whose arguments are out of alphabetical order (a string, an array of objects, an integer) after content, the
/// reply served whole and a character at a time; and a user turn after a reply with reasoning, without tools (where
/// GLM's `clear_thinking` and DeepSeek's `drop_thinking` would rewrite it). With ORIGAMI_MODEL (a model folder, or a
/// family GGUF whose header gives the tokenizer and template), on that tokenizer and template too.
func templateContinuationRoundTrip() throws {
    // Nested values render through the template's `tojson`, Python's: the model's key order, `", "` and `": "`.
    let edits = #"[{"oldText": "a/b", "newText": "b"}]"#
    let replies: [ModelFamily: (tool: String, plain: String)] = [
        .qwen: (
            "Looking.\n</think>\n\nI will edit it.\n\n<tool_call>\n<function=edit>\n<parameter=path>\nsrc/雪.txt\n</parameter>\n"
                + "<parameter=edits>\n\(edits)\n</parameter>\n<parameter=line>\n12\n</parameter>\n</function>\n</tool_call>",
            "Looking.\n</think>\n\nDone."
        ),
        .glm: (
            "Looking.</think>I will edit it.<tool_call>edit<arg_key>path</arg_key><arg_value>src/雪.txt</arg_value>"
                + "<arg_key>edits</arg_key><arg_value>\(edits)</arg_value><arg_key>line</arg_key><arg_value>12</arg_value></tool_call>",
            "Looking.</think>Done."
        ),
        .deepseek: (
            "Looking.</think>I will edit it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n"
                + "<｜DSML｜ parameter name=\"path\" string=\"true\">src/雪.txt</｜DSML｜ parameter>\n"
                + "<｜DSML｜ parameter name=\"edits\" string=\"false\">\(edits)</｜DSML｜ parameter>\n"
                + "<｜DSML｜ parameter name=\"line\" string=\"false\">12</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            "Looking.</think>Done."
        ),
    ]
    let tools: [[String: Any]] = [
        [
            "type": "function",
            "function": [
                "name": "edit", "description": "Edit a file",
                "parameters": [
                    "type": "object",
                    "properties": ["path": ["type": "string"], "edits": ["type": "array"], "line": ["type": "integer"]],
                ],
            ],
        ]
    ]
    func check(_ family: ModelFamily, _ stack: TokenizerStack, _ name: String) throws {
        let reply = replies[family]!
        for (text, withTools) in [(reply.tool, true), (reply.plain, false)] {
            var body: [String: Any] = [
                "model": "m", "reasoning_effort": "high",
                "messages": [["role": "system", "content": "Be brief."], ["role": "user", "content": "Fix it"]],
            ]
            if withTools { body["tools"] = tools }
            let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
            let prompt = try ServingPrompt.encode(request, tokenizer: stack)
            for pieces in [[text], text.map(String.init)] {
                let stream = ServingStream(request: request, family: family)
                for piece in pieces { _ = try stream.push(piece) }
                _ = try stream.finish(reason: "stop")
                var next = body
                var messages = body["messages"] as! [[String: Any]]
                messages.append(stream.message)
                if let call = stream.calls.first {
                    try grammarAssert(
                        stream.calls.count == 1 && call.order == ["path", "edits", "line"], "\(name): served \(stream.calls)")
                    messages.append(["role": "tool", "tool_call_id": call.id, "content": "done"])
                } else {
                    messages.append(["role": "user", "content": "Next"])
                }
                next["messages"] = messages
                let continued = try ServingPrompt.encode(
                    ServingRequest.parse(JSONSerialization.data(withJSONObject: next), contextTier: 8192), tokenizer: stack)
                let generated = prompt + stack.encode(text)
                let shared = TokenizerStack.commonTokenPrefix(generated, continued)
                try grammarAssert(
                    shared == generated.count,
                    "\(name) \(withTools ? "tool" : "user") turn (\(pieces.count) pieces) diverges at token "
                        + "\(shared) of \(generated.count): generated\n\(try stack.decode(Array(generated[shared...])))\nre-rendered\n"
                        + "\(try stack.decode(Array(continued[shared...].prefix(40))))")
            }
        }
    }
    for (family, folder) in [(ModelFamily.qwen, "qwen3.8-flash-next"), (.glm, "glm-5.3-flash"), (.deepseek, "deepseek-v4.1-flash")] {
        try check(
            family,
            try tinyFamilyTokenizer(String(contentsOf: CheckFixtures.file("models/\(folder)/chat_template.jinja"), encoding: .utf8)), folder
        )
    }
    guard let path = ProcessInfo.processInfo.environment["ORIGAMI_MODEL"] else { return }
    var folder = URL(fileURLWithPath: path)
    let support = FileManager.default.temporaryDirectory.appendingPathComponent("origami-continuation-\(getpid())-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: support) }
    if folder.pathExtension == "gguf" { folder = try FamilySupport.prepare(folder, into: support) }
    let model = folder
    final class Box: @unchecked Sendable {
        var stack: TokenizerStack?
        var error: Error?
    }
    let box = Box()
    let done = DispatchSemaphore(value: 0)
    Task {
        do { box.stack = try await TokenizerStack(modelDirectory: model) } catch { box.error = error }
        done.signal()
    }
    done.wait()
    if let error = box.error { throw error }
    let template = try String(contentsOf: model.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
    let family: ModelFamily = template.contains("｜DSML｜") ? .deepseek : template.contains("<arg_key>") ? .glm : .qwen
    try check(family, box.stack!, "ORIGAMI_MODEL (\(family))")
}

/// `tojson` renders as transformers' does (`json.dumps(x, ensure_ascii=False)`): the three real templates
/// (Tests/Fixtures/models) render a tool schema and a past call with nested, out-of-order, escaped and float arguments
/// exactly as Python does, and the filter's options match it too. The expected texts and hashes come from jinja2 3.x with
/// transformers' environment (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, its `tojson`),
/// rendering the same messages with `enable_thinking`, `thinking`, `reasoning_effort="medium"` and
/// `clear_thinking`/`drop_thinking` false. Tool schemas reach the template with sorted keys (the request body is read
/// with JSONSerialization), so the schema here is written sorted.
func templateToJSONMatchesPython() throws {
    let arguments =
        #"{"path": "src/雪.txt", "edits": [{"oldText": "a/b <x> & 'q'", "newText": "ü\n\"z\"\t"}], "line": 12, "ratio": 1.0, "#
        + #""tiny": 1e-05, "flags": {"dry": false, "note": null, "on": true}}"#
    let schema =
        #"{"description": "Edit a file under src/ (ü)", "name": "edit", "parameters": {"properties": {"edits": {"items": "#
        + #"{"properties": {"newText": {"type": "string"}, "oldText": {"type": "string"}}, "type": "object"}, "type": "array"}, "#
        + #""path": {"description": "Like src/a.txt", "type": "string"}}, "required": ["path", "edits"], "type": "object"}}"#
    let body: [String: Any] = [
        "model": "m", "reasoning_effort": "medium",
        "tools": [["type": "function", "function": try JSONSerialization.jsonObject(with: Data(schema.utf8))]],
        "messages": [
            ["role": "system", "content": "Be brief."], ["role": "user", "content": "Fix it"],
            [
                "role": "assistant", "content": "I will edit it.", "reasoning_content": "Looking.",
                "tool_calls": [["id": "call_1", "type": "function", "function": ["name": "edit", "arguments": arguments]]],
            ],
            ["role": "tool", "tool_call_id": "call_1", "content": "done"], ["role": "user", "content": "Next"],
        ],
    ]
    let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
    let parts = [
        schema, #"[{"oldText": "a/b <x> & 'q'", "newText": "ü\n\"z\"\t"}]"#, #"{"dry": false, "note": null, "on": true}"#, "1.0",
        "1e-05",
    ]
    for (folder, digest) in [
        ("qwen3.8-flash-next", "6e89d12c560c30a3e9abb690472dee17bf33618bc3c2161a10ee3a0730cce42e"),
        ("glm-5.3-flash", "d79ab3c7cb5b4e351b18851a5dbc4c751fe9262d0a6ffb592216ac126228aec1"),
        ("deepseek-v4.1-flash", "76455cb575297a92e2f354f6c55f0ce868c09c16f29162bb8212cf130605d047"),
    ] {
        let stack = try tinyFamilyTokenizer(String(contentsOf: CheckFixtures.file("models/\(folder)/chat_template.jinja"), encoding: .utf8))
        let rendered = try stack.decode(ServingPrompt.encode(request, tokenizer: stack))
        for part in parts { try grammarAssert(rendered.contains(part), "\(folder) lacks Python's \(part) in:\n\(rendered)") }
        let hash = SHA256.hash(data: Data(rendered.utf8)).map { String(format: "%02x", $0) }.joined()
        try grammarAssert(hash == digest, "\(folder) renders other than Python (sha256 \(hash)):\n\(rendered)")
    }
    // The options, on arguments out of key order.
    let options = try tinyFamilyTokenizer(
        "{% set x = messages[0].tool_calls[0].function.arguments %}{{ x | tojson(indent=2) }}\n"
            + "{{ x | tojson(ensure_ascii=True, sort_keys=True) }}\n{{ x | tojson(separators=[\",\", \":\"]) }}\n"
            + "{{ x | tojson(indent=\"\\t\") }}\n{{ x.b | tojson(indent=0) }}")
    let messages: [[String: TemplateValue]] = [
        [
            "role": .string("assistant"), "content": .string(""),
            "tool_calls": .array([
                .object([
                    "id": .string("c"), "type": .string("function"),
                    "function": .object([
                        "name": .string("f"),
                        "arguments": .string(#"{"d": {"y": null, "x": true}, "b": [1, 2.5, "é/<\u007f😀"], "a": {}, "c": []}"#),
                    ]),
                ])
            ]),
        ],
        ["role": .string("tool"), "tool_call_id": .string("c"), "content": .string("")],
    ]
    let expected =
        "{\n  \"d\": {\n    \"y\": null,\n    \"x\": true\n  },\n  \"b\": [\n    1,\n    2.5,\n    \"é/<\u{7F}😀\"\n  ],\n  \"a\": {},\n"
        + "  \"c\": []\n}\n{\"a\": {}, \"b\": [1, 2.5, \"\\u00e9/<\\u007f\\ud83d\\ude00\"], \"c\": [], \"d\": {\"x\": true, \"y\": null}}\n"
        + "{\"d\":{\"y\":null,\"x\":true},\"b\":[1,2.5,\"é/<\u{7F}😀\"],\"a\":{},\"c\":[]}\n{\n\t\"d\": {\n\t\t\"y\": null,\n\t\t\"x\": true\n\t},\n"
        + "\t\"b\": [\n\t\t1,\n\t\t2.5,\n\t\t\"é/<\u{7F}😀\"\n\t],\n\t\"a\": {},\n\t\"c\": []\n}\n[\n1,\n2.5,\n\"é/<\u{7F}😀\"\n]"
    let shown = try options.decode(options.encodeChat(messages: messages, reasoningEffort: "none"))
    try grammarAssert(shown == expected, "tojson options render\n\(shown)\nnot Python's\n\(expected)")
}
