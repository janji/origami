import Foundation
import OrigamiEngine

func tokenizerNormalizationPreservesToolSemantics() throws {
    let input: [[String: TemplateValue]] = [
        ["role": .string("system"), "content": .string("First")],
        ["role": .string("developer"), "content": .string("Second")],
        [
            "role": .string("user"),
            "content": .array([
                .object(["type": .string("text"), "text": .string("ab")]), .object(["type": .string("input_text"), "text": .string("cd")]),
            ]),
        ],
        [
            "role": .string("assistant"), "content": .null, "reasoning": .string("saved thought"),
            "tool_calls": .array([
                .object([
                    "id": .string("a"), "type": .string("function"),
                    "function": .object(["name": .string("alpha"), "arguments": .string("{\"n\":9007199254740993,\"optional\":null}")]),
                ]),
                .object([
                    "id": .string("b"), "type": .string("function"),
                    "function": .object(["name": .string("beta"), "arguments": .string("{}")]),
                ]),
            ]),
        ],
        ["role": .string("tool"), "tool_call_id": .string("b"), "content": .string("second")],
        ["role": .string("tool"), "tool_call_id": .string("a"), "content": .string("first")],
    ]
    let decoded = try JSONDecoder().decode([[String: TemplateValue]].self, from: JSONEncoder().encode(input))
    let result = try TokenizerStack.normalizeMessages(decoded)
    guard result.count == 5, result[0]["content"] == .string("First\n\nSecond"), result[1]["content"] == .string("abcd"),
        result[2]["reasoning_content"] == .string("saved thought"), result[3]["content"] == .string("first"),
        result[4]["content"] == .string("second"),
        case .array(let calls) = result[2]["tool_calls"], case .object(let call) = calls[0], case .object(let function) = call["function"],
        function["arguments"] == .string("{\"n\":9007199254740993,\"optional\":null}")
    else {
        throw EngineError("Tokenizer normalization changed message or tool semantics")
    }
    // Arguments sent as a JSON string render in its key order, at every depth, with the values typed as JSON has them
    // (an integer beyond a double's precision stays exact); arguments sent as an object render as before.
    let keys = try tinyFamilyTokenizer(
        "{% macro show(v) %}{% if v is mapping %}{{ '{' }}{% for k, x in v|items %}{{ k }}={{ show(x) }};{% endfor %}{{ '}' }}"
            + "{% elif v is string %}'{{ v }}'{% elif v is iterable %}[{% for x in v %}{{ show(x) }},{% endfor %}]{% elif v is none %}null{% else %}{{ v }}{% endif %}{% endmacro %}"
            + "{% for m in messages %}{% for c in m.tool_calls or [] %}{{ show(c.function.arguments) }}|{% endfor %}{% endfor %}")
    func rendered(_ arguments: TemplateValue) throws -> String {
        try keys.decode(
            keys.encodeChat(messages: [
                ["role": .string("user"), "content": .string("x")],
                [
                    "role": .string("assistant"), "content": .null,
                    "tool_calls": .array([
                        .object([
                            "id": .string("c"), "type": .string("function"),
                            "function": .object(["name": .string("f"), "arguments": arguments]),
                        ])
                    ]),
                ],
                ["role": .string("tool"), "tool_call_id": .string("c"), "content": .string("r")],
            ]))
    }
    let ordered = try rendered(
        .string(" {\"z\": \"a\\\"\\u00e9\\/\", \"b\":[1, 2.5, true, null, {\"y\":{}, \"a\":[]}], \"n\":9007199254740993 } "))
    let object = try rendered(.object(["z": .string("q"), "b": .integer(1)]))
    let repeated = try rendered(.string("{\"b\":1,\"a\":2,\"b\":3}"))
    guard ordered == "{z='a\"é/';b=[1,2.5,true,null,{y={};a=[];},];n=9007199254740993;}|", object == "{b=1;z='q';}|",
        repeated == "{b=1;a=2;}|"
    else {
        throw EngineError("Tool arguments rendered as \(ordered), \(object) and \(repeated)")
    }
    for bad in ["{\"a\":1", "{\"a\" 1}", "[1]", "{\"a\":1} x"] {
        do {
            _ = try TokenizerStack.normalizeMessages([
                ["role": .string("user"), "content": .string("x")],
                [
                    "role": .string("assistant"), "content": .null,
                    "tool_calls": .array([
                        .object([
                            "id": .string("c"), "type": .string("function"),
                            "function": .object(["name": .string("f"), "arguments": .string(bad)]),
                        ])
                    ]),
                ],
                ["role": .string("tool"), "tool_call_id": .string("c"), "content": .string("r")],
            ])
            throw EngineError("Tool arguments \(bad) were accepted")
        } catch is DecodingError {} catch let error as EngineError where !error.description.contains("accepted") {}
    }
    // The same thinking under both field names is one field (as clients echo a reply back); different texts are refused.
    let echoed = try TokenizerStack.normalizeMessages([
        ["role": .string("user"), "content": .string("hi")],
        [
            "role": .string("assistant"), "content": .string("Hi!"), "reasoning": .string("a greeting"),
            "reasoning_content": .string("a greeting"),
        ],
        ["role": .string("user"), "content": .string("again")],
    ])
    guard echoed.count == 3, echoed[1]["reasoning_content"] == .string("a greeting"), echoed[1]["reasoning"] == nil else {
        throw EngineError("The same reasoning under both names was not taken as one field")
    }
    do {
        _ = try TokenizerStack.normalizeMessages([
            ["role": .string("user"), "content": .string("hi")],
            ["role": .string("assistant"), "content": .string("Hi!"), "reasoning": .string("one"), "reasoning_content": .string("two")],
        ])
        throw TokenizerCheckFailure()
    } catch is EngineError {} catch { throw EngineError("Two different reasoning texts were accepted") }
    let late = try TokenizerStack.normalizeMessages([decoded[2], decoded[0]])
    guard late.count == 2, late[1]["role"] == .string("user"), late[1]["content"] == .string("First") else {
        throw EngineError("A system message after the conversation began did not become a user turn")
    }
    for invalid in [
        Array(input.dropLast()), [input[2], input[4]],
        [["role": .string("user"), "content": .array([.object(["type": .string("image_url"), "text": .string("reject")])])]],
    ] {
        do {
            _ = try TokenizerStack.normalizeMessages(invalid)
            throw TokenizerCheckFailure()
        } catch is EngineError {} catch { throw EngineError("Invalid chat was accepted") }
    }
    guard TokenizerStack.commonTokenPrefix([1, 2, 3], [1, 2, 4]) == 2 else { throw EngineError("Incorrect token prefix") }
    // The template's range() is Jinja's up to 65,536 values; a model file's range(9999999) (found by the GGUF fuzzer:
    // 1.5 GB on every prompt) and a step past Int.max are refused.
    let ranges = try tinyFamilyTokenizer(
        "{% for i in range(3) %}{{ i }}{% endfor %}|{{ range(5, 0, -2)|join(',') }}|{{ range(2, 9, 3)|join(',') }}|"
            + "{{ range(4, 4)|length }}{% if messages[0].content == 'big' %}{{ range(9999999)|length }}{% endif %}"
            + "{% if messages[0].content == 'step' %}{{ range(9223372036854775806, 9223372036854775807, 5)|join(',') }}{% endif %}")
    func render(_ content: String) throws -> String {
        try ranges.decode(ranges.encodeChat(messages: [["role": .string("user"), "content": .string(content)]]))
    }
    guard try render("x") == "012|5,3,1|2,5,8|0", try render("step") == "012|5,3,1|2,5,8|09223372036854775806" else {
        throw EngineError("range() rendered \(try render("x")) and \(try render("step"))")
    }
    do {
        _ = try render("big")
        throw EngineError("range(9999999) was rendered")
    } catch let error as EngineError where error.description.contains("65,536") {}
}
private struct TokenizerCheckFailure: Error {}

/// `ReplyText` is the whole decode of every longer prefix of a reply, byte for byte: on a byte-level tokenizer (the
/// families', no clean-up) over random replies whose characters split across tokens, with broken bytes (U+FFFD),
/// combining marks and special tokens, and with the whole-decode fallback. With ORIGAMI_MODEL, also on that model's
/// tokenizer over an 8,192-token reply, timing both (`measurements`: `reply_text`).
func replyTextMatchesWholeDecode() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let tokenizer = try tinyFamilyTokenizer("{{ messages }}")
    try require(tokenizer.decodesBytewise, "The family tokenizer is not byte-level without clean-up")
    let pieces: [[Int]] =
        ["a", " ", "é", "€", "😀", "\u{0301}", "\n"].map { Array($0.utf8).map(Int.init) }
        + [[0x80], [0xff], [0xe2, 0x82], [0xf0, 0x9f]] + [[256], [257], [258], [259]]
    var state: UInt64 = 0x9e37_79b9_7f4a_7c15
    func next(_ bound: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int(state >> 33) % bound
    }
    for trial in 0..<300 {
        var ids: [Int] = []
        var fast = ReplyText(tokenizer)
        var whole = ReplyText(tokenizer, bytewise: false)
        while ids.count < 96 {
            ids += pieces[next(pieces.count)]
            let text = try tokenizer.decode(ids)
            try require(
                Array(try fast.text(ids).utf8) == Array(text.utf8) && Array(try whole.text(ids).utf8) == Array(text.utf8),
                "Reply text differs from the whole decode in trial \(trial) at \(ids.count) tokens")
        }
    }
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
    try require(model.decodesBytewise, "The model's tokenizer is not byte-level without clean-up")
    let paragraph = "The paper crane unfolds its wings — 纸鹤展开翅膀，Ἀρχὴ ἥμισυ παντός; naïve café, 東京 🌸 x² ≥ ∑ αᵢ. "
    var ids: [Int] = []
    while ids.count < 8192 { ids += model.encode(paragraph) }
    ids = Array(ids.prefix(8192))
    var reply = ReplyText(model)
    var wholeSeconds = 0.0
    var replySeconds = 0.0
    for count in 1...ids.count {
        let prefix = Array(ids.prefix(count))
        var start = DispatchTime.now().uptimeNanoseconds
        let whole = try model.decode(prefix)
        wholeSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        start = DispatchTime.now().uptimeNanoseconds
        let text = try reply.text(prefix)
        replySeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        try require(Array(text.utf8) == Array(whole.utf8), "Reply text differs from the model's whole decode at \(count) tokens")
    }
    CheckFixtures.measurements.append([
        "name": "reply_text", "tokens": ids.count, "whole_decode_seconds": wholeSeconds, "reply_text_seconds": replySeconds,
    ])
}
