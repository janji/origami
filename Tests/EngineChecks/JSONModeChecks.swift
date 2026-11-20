import Foundation
import Metal
import OrigamiEngine

/// JSON mode (ROADMAP N7): the acceptor on a table of documents, the token mask on a small vocabulary and on a real
/// tokenizer's (when its support folder is on this Mac), the mask's cost, the wires, and constrained replies from the
/// synthetic GLM and DeepSeek models.
func jsonModeChecks() -> [(String, () throws -> Void)] {
    [
        ("json_mode_acceptor", jsonModeAcceptor),
        ("json_mode_schema_refusals", jsonModeSchemaRefusals),
        ("json_mode_enum_beside_type", jsonModeEnumBesideType),
        ("json_mode_key_escape", jsonModeKeyEscape),
        ("json_mode_forbidden_property", jsonModeForbiddenProperty),
        ("json_mode_mask", jsonModeMask),
        ("json_mode_mask_real_vocabulary", jsonModeMaskRealVocabulary),
        ("json_mode_wires", jsonModeWires),
        ("json_mode_synthetic_replies", jsonModeSyntheticReplies),
    ]
}

private func require(_ condition: Bool, _ what: @autoclosure () -> String) throws {
    if !condition { throw EngineError("JSON mode: " + what()) }
}
private func allows(_ bits: [UInt64], _ id: Int) -> Bool { bits[id >> 6] >> UInt64(id & 63) & 1 == 1 }

/// Feeds `text` byte by byte: alive throughout, and complete at its end.
private func feed(_ grammar: JSONGrammar, _ text: [UInt8], leading: Int = 0, opener: String? = nil) -> (alive: Bool, complete: Bool) {
    let rules = JSONRules(grammar: grammar, opener: opener, leadingNewlines: leading)
    var state = JSONAcceptor(rules)
    for byte in text where !state.step(byte, rules) { return (false, false) }
    return (true, state.complete(rules))
}

private let validObjects = [
    "{}", #"{"a":1}"#, #"{ "a" : [ ] , "b" : { } }"#, "{\n  \"a\": [1, 2.5, -0, 0e5, 1E+2, -3.25e-7],\n  \"b\": {\"c\": null}\n}",
    #"{"s":"\"\\\/\b\f\n\r\t","u":"\u00e9\ud83d\ude00\u0041"}"#, #"{"ü":"日本語 €","e":"🎉"}"#,
    #"{"a":[[[[]]]],"b":[{},{"c":[true,false,null]}]}"#,
    #"{"n":-0.0e0,"m":12345678901234567890,"k":0.000001}"#, #"{"a":"","":""}"#,
]
private let invalidObjects = [
    "[]", "\"x\"", "1", " {}", "{} ", "{}x", #"{"a":01}"#, #"{"a":1.}"#, #"{"a":.5}"#, #"{"a":+1}"#, "{,}", #"{"a"}"#, #"{"a":1,}"#,
    #"{'a':1}"#, #"{"a":tru}"#, #"{"a":nul}"#, #"{"a":1e}"#, #"{"a":-}"#, #"{"a":"\x"}"#, #"{"a":"\ud800"}"#, #"{"a":"\udc00"}"#,
    #"{"a":"\ud800\u0041"}"#, #"{"a":"\u12g4"}"#, #"{"a":1E812}"#, #"{"a":1e-400}"#, "{\"a\":\"\u{01}\"}", #"{"a":[1,]}"#, #"{"a":[,1]}"#,
    #"{"a" 1}"#, "{\"a\":1}\n",
]

/// The acceptor against a table: valid documents (also as JSONSerialization reads them) accepted whole and complete only
/// at their end; invalid ones refused or left incomplete; byte splits; invalid UTF-8; schemas; thinking's leading
/// newlines; a tool-call opener.
private func jsonModeAcceptor() throws {
    for text in validObjects {
        let bytes = Array(text.utf8)
        try require(
            (try? JSONSerialization.jsonObject(with: Data(bytes))) is [String: Any], "the table's valid document \(text) does not parse")
        try require(feed(.object, bytes) == (true, true), "valid \(text) refused")
        for cut in 1..<bytes.count where feed(.object, Array(bytes.prefix(cut))).complete {
            throw EngineError("JSON mode: the prefix \(String(decoding: bytes.prefix(cut), as: UTF8.self)) of \(text) is complete")
        }
    }
    for text in invalidObjects {
        let result = feed(.object, Array(text.utf8))
        try require(!(result.alive && result.complete), "invalid \(text) accepted")
    }
    for bytes: [UInt8] in [
        [0xC0, 0x80], [0xED, 0xA0, 0x80], [0xF5, 0x80, 0x80, 0x80], [0xE0, 0x80, 0x80], [0xF4, 0x90, 0x80, 0x80], [0x80],
    ] {
        let text = Array(#"{"a":""#.utf8) + bytes + Array(#""}"#.utf8)
        try require(!feed(.object, text).alive, "invalid UTF-8 \(bytes) accepted")
    }
    // Thinking: up to two newlines before the JSON (the think splitter drops them); never spaces.
    try require(feed(.object, Array("\n\n{}".utf8), leading: 2) == (true, true), "two newlines after thinking refused")
    try require(!feed(.object, Array("\n\n\n{}".utf8), leading: 2).alive, "three leading newlines accepted")
    try require(!feed(.object, Array(" {}".utf8), leading: 2).alive, "a leading space accepted")
    // A tool call instead of the JSON: once its opener is whole, nothing is constrained.
    let tool = feed(.object, Array("<tool_call>anything {".utf8), opener: "<tool_call>")
    try require(tool.alive && !tool.complete, "a tool call was not left free")
    try require(!feed(.object, Array("<tool_call>".utf8)).alive, "a tool opener accepted without tools")

    // Schemas: required, additional, nested $ref, arrays with bounds, enum, anyOf, integer, duplicates.
    let schema: [String: Any] = [
        "type": "object",
        "properties": [
            "name": ["type": "string"], "age": ["type": "integer"],
            "tags": ["type": "array", "items": ["enum": ["x", "y", "日"]], "minItems": 1, "maxItems": 3],
            "kind": ["anyOf": [["type": "string"], ["type": "null"], ["const": "fixed"]]],
            "pos": ["$ref": "#/$defs/point"], "ratio": ["type": ["number", "boolean"]],
            "child": ["$ref": "#"],
        ],
        "required": ["name", "age"], "additionalProperties": false,
        "$defs": [
            "point": [
                "type": "object", "properties": ["x": ["type": "number"], "y": ["type": "number"]], "required": ["x", "y"],
                "additionalProperties": false, "description": "a point",
            ]
        ],
    ]
    let grammar = try JSONGrammar(schema: schema)
    for text in [
        #"{"name":"a","age":3}"#, #"{"age":-1,"name":"","tags":["x","日"],"kind":null,"pos":{"y":2.5,"x":1}}"#,
        #"{"name":"b","age":0,"kind":"fixed","ratio":true,"child":{"name":"c","age":1,"child":{"name":"d","age":2}}}"#,
        #"{"name":"q","age":7,"kind":"anything","ratio":-1.5e3}"#,
    ] {
        try require(feed(grammar, Array(text.utf8)) == (true, true), "schema refused \(text)")
    }
    for text in [
        #"{"name":"a"}"#, #"{"name":"a","age":1,"z":1}"#, #"{"name":"a","age":1.5}"#, #"{"name":"a","age":1,"tags":[]}"#,
        #"{"name":"a","age":1,"tags":["x","x","x","x"]}"#, #"{"name":"a","age":1,"tags":["q"]}"#, #"{"name":"a","name":"b","age":1}"#,
        #"{"name":"a","age":1,"pos":{"x":1}}"#, #"{"name":"a","age":1,"kind":3}"#, #"{"name":1,"age":1}"#,
        #"{"name":"a","age":1,"ratio":"1"}"#,
        #"{"name":"a","age":1,"child":{"age":1}}"#,
    ] {
        let result = feed(grammar, Array(text.utf8))
        try require(!(result.alive && result.complete), "schema accepted \(text)")
    }
    // Additional properties allowed: any key but a repeat of a declared one; a scalar root; an empty object schema.
    let open = try JSONGrammar(schema: ["type": "object", "properties": ["a": ["type": "boolean"]]])
    try require(feed(open, Array(#"{"b":[1],"a":true,"ab":"x"}"#.utf8)) == (true, true), "an additional key refused")
    try require(!feed(open, Array(#"{"a":true,"a":false}"#.utf8)).alive, "a repeated key accepted")
    try require(!feed(open, Array(#"{"a":1}"#.utf8)).alive, "a declared key's schema ignored beside additional keys")
    try require(feed(try JSONGrammar(schema: ["type": "integer"]), Array("-12".utf8)) == (true, true), "an integer root refused")
    try require(!feed(try JSONGrammar(schema: ["type": "integer"]), Array("1.0".utf8)).alive, "an integer took a fraction")
    try require(feed(try JSONGrammar(schema: [:]), Array("[1,\"a\"]".utf8)) == (true, true), "an empty schema refused an array")
    // An optional property with no finite value is left out; a recursion with no way out is refused.
    let pruned = try JSONGrammar(schema: [
        "type": "object", "properties": ["a": false, "b": ["type": "null"]], "additionalProperties": false,
    ])
    try require(!feed(pruned, Array(#"{"a""#.utf8)).alive && feed(pruned, Array(#"{"b":null}"#.utf8)) == (true, true), "pruning")
}

/// Unsupported keywords and impossible schemas are refused with a 400 naming them.
private func jsonModeSchemaRefusals() throws {
    let cases: [(Any, String)] = [
        (["type": "string", "pattern": "^a"], "pattern"), (["type": "string", "minLength": 2], "minLength"),
        (["oneOf": [["type": "string"]]], "oneOf"), (["type": "number", "maximum": 3], "maximum"),
        (["type": "object", "properties": ["a": ["type": "array", "items": [["type": "string"]]]]], "items"),
        (["$ref": "https://example.com/schema"], "$ref"), (["type": "object", "required": ["a"]], "required"),
        (["type": "object", "properties": ["a": ["$ref": "#"]], "required": ["a"]], "no finite"),
        (["type": "date"], "date"), (["type": "array", "minItems": 3, "maxItems": 2], "minItems"), ("x", "schema must be"),
        (["type": "object", "patternProperties": [:]], "patternProperties"), (["enum": []], "enum"),
    ]
    for (schema, named) in cases {
        do {
            _ = try JSONGrammar(schema: schema)
            throw EngineError("JSON mode: the schema \(schema) was accepted")
        } catch let failure as ServingFailure {
            try require(
                failure.status == 400 && failure.description.contains(named), "unclear refusal for \(named): \(failure.description)")
        }
    }
}

/// Outside review (R33): `type` beside `enum` narrows the values; an enum with none of the type, or beside another
/// keyword, is refused by name.
private func jsonModeEnumBesideType() throws {
    let grammar = try JSONGrammar(schema: ["type": "integer", "enum": [1, "a", 2.5]])
    try require(feed(grammar, Array("1".utf8)) == (true, true), "an integer enum value refused")
    for text in [#""a""#, "2.5"] { try require(!feed(grammar, Array(text.utf8)).complete, "\(text) accepted for an integer") }
    let either = try JSONGrammar(schema: ["type": ["string", "null"], "enum": ["a", 1, NSNull()]])
    try require(feed(either, Array(#""a""#.utf8)).complete && !feed(either, Array("1".utf8)).complete, "a type list did not narrow")
    try require(feed(either, Array("null".utf8)).complete, "null refused for a null type")
    for (schema, named) in [
        (["type": "integer", "enum": ["a"]] as [String: Any], "no enum or const value has the type integer"),
        (["enum": [["a": 1]], "properties": ["a": ["type": "string"]]], "beside properties"),
    ] {
        do {
            _ = try JSONGrammar(schema: schema)
            throw EngineError("JSON mode: \(schema) was accepted")
        } catch let failure as ServingFailure {
            try require(failure.status == 400 && failure.description.contains(named), "unclear refusal: \(failure.description)")
        }
    }
}

/// Outside review (R33): a declared key spelled with an escape (`\u0078` for `x`) is not an additional key that skips
/// the property's schema; additional keys without escapes, and declared keys that need one, still work.
private func jsonModeKeyEscape() throws {
    let grammar = try JSONGrammar(schema: ["type": "object", "properties": ["x": ["type": "integer"], "q\"": ["type": "integer"]]])
    for text in [#"{"\u0078":"s"}"#, #"{"\u0078":1}"#, #"{"a\/b":1}"#, #"{"q\u0022":"s"}"#] {
        try require(!feed(grammar, Array(text.utf8)).complete, "\(text) accepted")
    }
    for text in [#"{"x":1}"#, #"{"q\"":2,"y":"free"}"#, #"{"x":1,"日本":[]}"#] {
        try require(feed(grammar, Array(text.utf8)) == (true, true), "\(text) refused")
    }
    try require(feed(.object, Array(#"{"\u0078":"s"}"#.utf8)) == (true, true), "json_object refused an escaped key")
}

/// Outside review (R33): `"x": false` forbids `x` even when additional properties are allowed; other keys still pass.
private func jsonModeForbiddenProperty() throws {
    for additional in [true, false] {
        let grammar = try JSONGrammar(schema: [
            "type": "object", "properties": ["x": false, "y": ["type": "integer"]], "additionalProperties": additional,
        ])
        try require(!feed(grammar, Array(#"{"x":1}"#.utf8)).complete, "a forbidden x accepted (additional \(additional))")
        try require(feed(grammar, Array(#"{"y":1}"#.utf8)) == (true, true), "y refused (additional \(additional))")
        try require(feed(grammar, Array(#"{"xy":1}"#.utf8)).complete == additional, "xy wrong (additional \(additional))")
    }
}

/// A synthetic vocabulary: every byte, JSON-ish pieces spanning several lexemes, pieces ending inside a UTF-8 character,
/// a tool opener and a stop token.
private let syntheticPieces: [[UInt8]] = {
    var pieces = (0..<256).map { [UInt8($0)] }
    for text in [
        #"{""#, #"":"#, #"","#, #""}"#, #""]"#, "},", "}}", "]}", "[{", "true", "false", "null", "123", ".5", "e-", #"\u"#, "00", "d8",
        "de", #"\n"#, #"\""#, " \"", "\n  ", "  ", "日本", "ab", "name", #""name":"#, ",\"", "0,", "1]", "-0", "e+1", "[]", "{}", "\":[",
        "true}", "null,", #"\ud83d"#, #"\ude00"#, "🎉", #"","age":"#,
    ] { pieces.append(Array(text.utf8)) }
    pieces += [[0xE6, 0x97], [0xA5, 0x22], [0xF0, 0x9F], [0x8E, 0x89, 0x22, 0x7D], [0xC3]]
    return pieces
}()

private func syntheticVocabulary(extra: [[UInt8]] = []) -> (JSONVocabulary, tool: Int, stop: Int) {
    let pieces = syntheticPieces + extra
    let tool = pieces.count
    let stop = pieces.count + 1
    let vocabulary = JSONVocabulary(size: pieces.count + 2) { id in
        id == tool ? (Array("<tool_call>".utf8), true) : id == stop ? (Array("<|end|>".utf8), true) : (pieces[id], false)
    }
    return (vocabulary, tool, stop)
}

/// Longest-match tokenization over the synthetic pieces.
private func tokenize(_ text: [UInt8], _ pieces: [[UInt8]]) -> [Int] {
    var ids: [Int] = []
    var at = 0
    while at < text.count {
        let best = pieces.indices.filter { pieces[$0].count <= text.count - at && Array(text[at..<at + pieces[$0].count]) == pieces[$0] }
            .max { pieces[$0].count < pieces[$1].count }!
        ids.append(best)
        at += pieces[best].count
    }
    return ids
}

/// The mask never refuses a valid document's next token (whole, longest-match tokenized, split in two at every byte),
/// allows the stop token exactly at its end, and random walks over allowed tokens never reach a state with nothing
/// allowed; those that stop parse. Masking puts -infinity on refused tokens and fails when nothing allowed is finite.
private func jsonModeMask() throws {
    let (vocabulary, tool, stop) = syntheticVocabulary()
    func constraint(_ grammar: JSONGrammar = .object, vocabulary: JSONVocabulary, tool: String? = nil, stops: [Int]) -> JSONConstraint {
        JSONConstraint(grammar: grammar, vocabulary: vocabulary, stops: stops, thinkEnd: nil, thinking: false, toolOpener: tool)
    }
    func follow(_ ids: [Int], _ c: JSONConstraint, stop: Int, _ what: String) throws {
        for (index, id) in ids.enumerated() {
            let bits = c.allowedBits()
            try require(allows(bits, id), "\(what): token \(index) refused")
            try require(!allows(bits, stop), "\(what): stop allowed before the end")
            try c.accept(id)
        }
        try require(allows(c.allowedBits(), stop), "\(what): stop refused at the end")
    }
    for text in validObjects {
        let bytes = Array(text.utf8)
        try follow(
            tokenize(bytes, syntheticPieces), constraint(vocabulary: vocabulary, stops: [stop]), stop: stop, "longest match of \(text)")
        for cut in 1..<bytes.count {
            let (split, _, splitStop) = syntheticVocabulary(extra: [Array(bytes[..<cut]), Array(bytes[cut...])])
            let first = syntheticPieces.count
            let second = first + 1
            // Byte pieces share the split's bytes; the check is that the two (possibly mid-character) pieces are allowed.
            try follow([first, second], constraint(vocabulary: split, stops: [splitStop]), stop: splitStop, "\(text) split at \(cut)")
        }
    }
    // The tool opener is allowed only at the start, and only when tools are.
    try require(!allows(constraint(vocabulary: vocabulary, stops: [stop]).allowedBits(), tool), "tool opener allowed without tools")
    let tools = constraint(vocabulary: vocabulary, tool: "<tool_call>", stops: [stop])
    try require(allows(tools.allowedBits(), tool), "tool opener refused with tools")
    try tools.accept(tool)
    try require(!tools.constrains, "a tool call stayed constrained")

    // Random walks: never a dead end; a walk that stops parses (as the schema says, for the schema).
    let schema = try JSONGrammar(schema: [
        "type": "object", "properties": ["name": ["type": "string"], "n": ["type": "array", "items": ["type": "integer"], "maxItems": 2]],
        "required": ["name"], "additionalProperties": false,
    ])
    var stream = SplitMixStream(seed: 7)
    var stopped = 0
    for walk in 0..<300 {
        let grammar = walk % 2 == 0 ? JSONGrammar.object : schema
        let c = constraint(grammar, vocabulary: vocabulary, stops: [stop])
        var text: [UInt8] = []
        for _ in 0..<200 {
            let bits = c.allowedBits()
            let allowed = (0..<vocabulary.size).filter { allows(bits, $0) }
            try require(!allowed.isEmpty, "walk \(walk) reached a dead end after \(String(decoding: text, as: UTF8.self))")
            // Stop when allowed; otherwise prefer closing pieces half the time so walks end.
            if allowed.contains(stop) {
                stopped += 1
                let parsed = try? JSONSerialization.jsonObject(with: Data(text)) as? [String: Any]
                try require(parsed != nil, "walk \(walk) stopped on unparsable \(String(decoding: text, as: UTF8.self))")
                if walk % 2 == 1 {
                    try require(
                        parsed!["name"] is String && Set(parsed!.keys).isSubset(of: ["name", "n"]), "walk \(walk) broke the schema")
                }
                break
            }
            let closing = allowed.filter { vocabulary.bytes(of: $0).contains { [0x22, 0x5D, 0x7D].contains($0) } }
            let pool = !closing.isEmpty && stream.uniform() < 0.5 ? closing : allowed
            let id = pool[Int(stream.uniform() * Double(pool.count))]
            try c.accept(id)
            text += vocabulary.bytes(of: id)
        }
    }
    try require(stopped >= 150, "only \(stopped) of 300 random walks completed")
    // Masking: refused tokens at -infinity, and a dead end is an error, not invalid JSON.
    let c = constraint(vocabulary: vocabulary, stops: [stop])
    let logits = (0..<vocabulary.size).map { Float($0 % 7) }
    let masked = try c.masked(logits)
    try require(masked[0x7B] == logits[0x7B] && masked[0x5B] == -.infinity && masked[stop] == -.infinity, "masking")
    var dead = logits
    dead[0x7B] = -.infinity
    for (id, piece) in syntheticPieces.enumerated() where piece.first == 0x7B { dead[id] = .nan }
    do {
        _ = try c.masked(dead)
        throw EngineError("JSON mode: a mask with no usable token was sampled")
    } catch let error as EngineError where error.code == "json_mode" {}
}

/// A tokenizer for the real-vocabulary check: a support folder on this Mac when there is one (read only), else a
/// tiny family tokenizer whose fillers are JSON pieces. Returns the tokenizer, its vocabulary size and end token.
private func realTokenizer() throws -> (TokenizerStack, size: Int, stop: Int, name: String) {
    let support = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Origami/support")
    let folders = (try? FileManager.default.contentsOfDirectory(at: support, includingPropertiesForKeys: nil)) ?? []
    for folder in folders.sorted(by: { $0.path < $1.path }) {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("tokenizer.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let vocab = (object["model"] as? [String: Any])?["vocab"] as? [String: Int],
            let config = try? JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json")))
                as? [String: Any]
        else { continue }
        let added = object["added_tokens"] as? [[String: Any]] ?? []
        let eos = config["eos_token"] as? String ?? (config["eos_token"] as? [String: Any])?["content"] as? String
        guard let stop = added.first(where: { $0["content"] as? String == eos })?["id"] as? Int else { continue }
        let size = max(vocab.values.max() ?? 0, added.compactMap { $0["id"] as? Int }.max() ?? 0) + 1
        final class Box: @unchecked Sendable { var result: Result<TokenizerStack, Error>? }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        Task {
            do { box.result = .success(try await TokenizerStack(modelDirectory: folder)) } catch { box.result = .failure(error) }
            done.signal()
        }
        done.wait()
        if case .success(let tokenizer) = box.result! { return (tokenizer, size, stop, "support \(folder.lastPathComponent)") }
    }
    let tokenizer = try tinyFamilyTokenizer("{{ messages }}", fillerTexts: jsonFillers(1_787))
    return (tokenizer, 2_048, 259, "tiny fixture")
}

/// The real vocabulary's mask: each corpus document's own tokens are allowed at every step and the end token only at
/// the end; random walks never dead-end. Measures a mask's first computation and a cached one (`measurements`), and
/// the same on a synthetic 248,320-token vocabulary (Qwen's size).
private func jsonModeMaskRealVocabulary() throws {
    let (tokenizer, size, stop, name) = try realTokenizer()
    let built = DispatchTime.now().uptimeNanoseconds
    let vocabulary = tokenizer.jsonVocabulary(size: size)
    let buildSeconds = Double(DispatchTime.now().uptimeNanoseconds - built) / 1e9
    var misses: [Double] = []
    var hits: [Double] = []
    var applies: [Double] = []
    func timed(_ c: JSONConstraint) -> [UInt64] {
        let before = c.computeSeconds
        let count = c.computed
        let bits = c.allowedBits()
        if c.computed > count { misses.append(c.computeSeconds - before) }
        let start = DispatchTime.now().uptimeNanoseconds
        _ = c.allowedBits()
        hits.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        return bits
    }
    let logits = [Float](repeating: 0, count: size)
    for text in validObjects + [#"{"answer": 42, "reason": "The sum of 40 and 2.", "steps": ["add", "check"]}"#] {
        let ids = tokenizer.encode(text)
        try require(ids.flatMap { vocabulary.bytes(of: $0) } == Array(text.utf8), "\(name): \(text) does not round-trip through its tokens")
        let c = JSONConstraint(grammar: .object, vocabulary: vocabulary, stops: [stop], thinkEnd: nil, thinking: false, toolOpener: nil)
        for (index, id) in ids.enumerated() {
            let bits = timed(c)
            try require(allows(bits, id), "\(name): \(text) token \(index) (\(id)) refused")
            try require(!allows(bits, stop), "\(name): the end allowed inside \(text)")
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try c.masked(logits)
            applies.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            try c.accept(id)
        }
        try require(allows(timed(c), stop), "\(name): the end refused after \(text)")
    }
    var stream = SplitMixStream(seed: 11)
    for walk in 0..<8 {
        let c = JSONConstraint(grammar: .object, vocabulary: vocabulary, stops: [stop], thinkEnd: nil, thinking: false, toolOpener: nil)
        for _ in 0..<40 {
            let bits = timed(c)
            let allowed = (0..<size).filter { allows(bits, $0) }
            try require(!allowed.isEmpty, "\(name): walk \(walk) reached a dead end")
            if allowed == [stop] { break }
            let id = allowed.filter { $0 != stop }[Int(stream.uniform() * Double(allowed.count - (allowed.contains(stop) ? 1 : 0)))]
            try c.accept(id)
        }
    }
    func milliseconds(_ values: [Double]) -> [String: Double] {
        ["mean": values.reduce(0, +) / Double(max(values.count, 1)) * 1000, "max": (values.max() ?? 0) * 1000]
    }
    // Qwen's vocabulary size, made of random byte pieces (words, punctuation runs, UTF-8 fragments).
    var random = SplitMixStream(seed: 3)
    let alphabet = Array("etaoinshrdlucmfwypvbgkqjxz".utf8)
    let punctuation = Array("{}[]\":,. \n0123456789-_\\/'".utf8)
    let big = JSONVocabulary(size: 248_320) { id in
        if id < 256 { return ([UInt8(id)], false) }
        if id >= 248_000 { return (Array("<|c\(id)|>".utf8), true) }
        let length = 1 + Int(random.uniform() * 9)
        let kind = random.uniform()
        let bytes: [UInt8] = (0..<length).map { _ in
            kind < 0.7
                ? alphabet[Int(random.uniform() * Double(alphabet.count))]
                : kind < 0.9 ? punctuation[Int(random.uniform() * Double(punctuation.count))] : UInt8(0x80 + Int(random.uniform() * 0x70))
        }
        return (kind < 0.5 ? [0x20] + bytes : bytes, false)
    }
    var bigMisses: [Double] = []
    for prefix in ["", "{", #"{"key"#, #"{"key":"some text"#, #"{"key":12"#, #"{"a":[1,{"b":"#] {
        let c = JSONConstraint(grammar: .object, vocabulary: big, stops: [248_319], thinkEnd: nil, thinking: false, toolOpener: nil)
        for byte in prefix.utf8 { try c.accept(Int(byte)) }
        let before = c.computeSeconds
        _ = c.allowedBits()
        bigMisses.append(c.computeSeconds - before)
    }
    CheckFixtures.measurements.append([
        "name": "json_mode_mask", "tokenizer": name, "vocabulary": size, "trie_nodes": vocabulary.trieNodes,
        "vocabulary_build_ms": buildSeconds * 1000, "first_visit_ms": milliseconds(misses), "cached_ms": milliseconds(hits),
        "apply_ms": milliseconds(applies), "first_visits": misses.count,
        "synthetic_248320": ["trie_nodes": big.trieNodes, "first_visit_ms": milliseconds(bigMisses)],
    ])
    FileHandle.standardError.write(
        Data(
            String(
                format:
                    "json mode mask (%@, %d tokens): first visit %.1f ms mean / %.1f max, cached %.3f ms, apply %.2f ms; 248,320 tokens: first visit %.1f ms mean / %.1f max\n",
                name, size, milliseconds(misses)["mean"]!, milliseconds(misses)["max"]!, milliseconds(hits)["mean"]!,
                milliseconds(applies)["mean"]!, milliseconds(bigMisses)["mean"]!, milliseconds(bigMisses)["max"]!
            ).utf8))
}

private func chatBody(_ fields: [String: Any]) throws -> Data {
    var body: [String: Any] = ["model": "m", "messages": [["role": "user", "content": "hi"]]]
    for (key, value) in fields { body[key] = value }
    return try JSONSerialization.data(withJSONObject: body)
}

/// The three wires reach the same grammar; text formats stay unconstrained; bad formats and teacher forcing are 400s.
private func jsonModeWires() throws {
    let schema: [String: Any] = [
        "type": "object", "properties": ["a": ["type": "integer"]], "required": ["a"], "additionalProperties": false,
    ]
    let expected = try JSONGrammar(schema: schema)
    func parse(_ data: Data, _ format: WireFormat = .openAI) throws -> ServingRequest {
        try ServingRequest.parse(WireTranslation.chatBody(data, format: format), contextTier: 8_192)
    }
    let chat: [(Any, JSONGrammar?)] = [
        (["type": "text"], nil), (["type": "json_object"], .object),
        (["type": "json_schema", "json_schema": ["name": "x", "schema": schema, "strict": true]], expected),
        (["type": "json_schema", "json_schema": ["name": "x"]], .object),
    ]
    for (format, grammar) in chat {
        let parsed = try parse(chatBody(["response_format": format]))
        try require(parsed.responseFormat == grammar, "chat response_format \(format)")
    }
    let responses: [(Any, JSONGrammar?)] = [
        (["type": "text"], nil), (["type": "json_object"], .object),
        (["type": "json_schema", "name": "x", "schema": schema, "strict": true], expected),
    ]
    for (format, grammar) in responses {
        let body = try JSONSerialization.data(withJSONObject: ["model": "m", "input": "hi", "text": ["format": format]])
        let parsed = try parse(body, .responses)
        try require(parsed.responseFormat == grammar, "Responses text.format \(format)")
    }
    let ollama: [(Any, JSONGrammar?)] = [("", nil), ("json", .object), (schema, expected)]
    for (format, grammar) in ollama {
        let body = try JSONSerialization.data(withJSONObject: [
            "model": "m", "messages": [["role": "user", "content": "hi"]], "format": format,
        ])
        let parsed = try parse(body, .ollama)
        try require(parsed.responseFormat == grammar, "Ollama format \(format)")
    }
    let refused: [(Data, WireFormat, String)] = [
        (try chatBody(["response_format": ["type": "xml"]]), .openAI, "type"),
        (try chatBody(["response_format": ["type": "json_schema"]]), .openAI, "json_schema"),
        (
            try chatBody(["response_format": ["type": "json_schema", "json_schema": ["schema": ["type": "string", "pattern": "a"]]]]),
            .openAI, "pattern"
        ),
        (
            try chatBody(["response_format": ["type": "json_schema", "json_schema": ["schema": schema, "strict": "yes"]]]), .openAI,
            "strict"
        ),
        (try chatBody(["response_format": ["type": "json_object"], "origami_teacher_tokens": [1, 2]]), .openAI, "teacher"),
        (
            try JSONSerialization.data(withJSONObject: ["model": "m", "input": "hi", "text": ["format": ["type": "yaml"]]]), .responses,
            "text.format"
        ),
        (try JSONSerialization.data(withJSONObject: ["model": "m", "messages": [], "format": "yaml"]), .ollama, "format"),
        (
            try JSONSerialization.data(withJSONObject: ["model": "m", "messages": [], "format": ["type": "object", "minProperties": 1]]),
            .ollama, "minProperties"
        ),
    ]
    for (body, format, named) in refused {
        do {
            _ = try parse(body, format)
            throw EngineError("JSON mode: \(String(decoding: body, as: UTF8.self)) was accepted")
        } catch let failure as ServingFailure {
            try require(failure.status == 400 && failure.description.contains(named), "unclear refusal (\(named)): \(failure.description)")
        }
    }
}

/// Filler tokens for a synthetic model's vocabulary that make JSON likely to close: words ending in a quote and a
/// delimiter, and numbers, all printable ASCII (byte-level BPE writes those as themselves).
func jsonFillers(_ count: Int) -> [String] {
    let letters = Array("abcdefghijklmnopqrstuvwxyz")
    let endings = ["\"", "\",", "\":", "\"}", "\"]", "\"},", ""]
    return (0..<count).map { index in
        if index % 3 == 2 { return "\(index)" + ["", ",", "}", "]"][index / 3 % 4] }
        var word = ""
        var value = index + 1
        while value > 0 {
            word.append(letters[(value - 1) % 26])
            value = (value - 1) / 26
        }
        return word + endings[index % endings.count]
    }
}

/// The synthetic GLM and DeepSeek models through the serving backend, with a vocabulary of JSON pieces (`jsonFillers`):
/// 100 seeds at temperature 1 in json_object mode all parse; schema replies validate; with thinking (closed by a
/// budget) the think block is free and the answer after it is JSON; the same request without JSON mode is unchanged by
/// the constraint code (its tokens equal a run on a backend that never saw JSON mode); a cut reply finishes `length`.
private func jsonModeSyntheticReplies() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-json-mode-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let schema: [String: Any] = [
        "type": "object",
        "properties": [
            "ok": ["type": "boolean"], "n": ["type": "integer"], "tag": ["enum": ["red", "green"]],
            "items": ["type": "array", "items": ["type": "number"], "maxItems": 3],
        ],
        "required": ["ok", "n", "tag"], "additionalProperties": false,
    ]
    for family in [ModelFamily.glm, .deepseek] {
        let name = family == .glm ? "glm" : "deepseek"
        let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
        let tokenizer = try tinyFamilyTokenizer(
            "{% for m in messages %}{{ m.content }}{% endfor %}", fillerTexts: jsonFillers(weights.geometry.vocab - 261))
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed), context: 2_048) { _ in }
        let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
        func run(_ fields: [String: Any]) throws -> (text: String, finish: String, tokens: [Int]) {
            var body: [String: Any] = [
                "model": "synthetic", "messages": [["role": "user", "content": "Reply in JSON."]], "max_tokens": 160,
                "reasoning_effort": "none", "origami_return_tokens": true,
            ]
            for (key, value) in fields { body[key] = value }
            let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 2_048)
            final class Text: @unchecked Sendable { var value = "" }
            let text = Text()
            let stats = try backend.prepare(request, cancellation: GenerationControl()).run {
                text.value += $0
                return true
            }
            return (text.value, stats.finishReason, stats.outputTokens ?? [])
        }
        var lengths = 0
        for seed in 0..<100 {
            let reply = try run(["response_format": ["type": "json_object"], "seed": seed, "temperature": 1])
            if reply.finish == "length" {
                lengths += 1
                continue
            }
            try require(
                reply.finish == "stop" && (try? JSONSerialization.jsonObject(with: Data(reply.text.utf8))) is [String: Any],
                "\(name) seed \(seed): \(reply.finish) \(reply.text)")
        }
        try require(lengths == 0, "\(name): \(lengths) of 100 json_object replies ran out of tokens")
        for seed in 0..<20 {
            let reply = try run([
                "response_format": ["type": "json_schema", "json_schema": ["name": "s", "schema": schema]], "seed": seed, "temperature": 1,
            ])
            let value = try? JSONSerialization.jsonObject(with: Data(reply.text.utf8)) as? [String: Any]
            // JSONSerialization gives NSNumbers: a boolean is CFBoolean; an integer may be too large for Int.
            func boolean(_ x: Any?) -> Bool { (x as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false }
            func whole(_ x: Any?) -> Bool { (x as? NSNumber).map { !boolean($0) && $0.doubleValue == $0.doubleValue.rounded() } ?? false }
            try require(
                reply.finish == "stop" && boolean(value?["ok"]) && whole(value?["n"])
                    && ["red", "green"].contains(value?["tag"] as? String)
                    && Set(value!.keys).isSubset(of: ["ok", "n", "tag", "items"]) && ((value?["items"] as? [Any])?.count ?? 0) <= 3,
                "\(name) schema seed \(seed): \(reply.finish) \(reply.text)")
        }
        for seed in 0..<10 {
            let reply = try run([
                "response_format": ["type": "json_object"], "seed": seed, "reasoning_effort": "medium", "max_thinking_tokens": 12,
            ])
            guard let end = reply.text.range(of: "</think>") else { throw EngineError("JSON mode: \(name) thinking never closed") }
            let answer = reply.text[end.upperBound...].drop { $0 == "\n" }
            try require(
                reply.finish == "stop" && (try? JSONSerialization.jsonObject(with: Data(answer.utf8))) is [String: Any],
                "\(name) thinking seed \(seed): \(reply.text)")
            try require(reply.tokens.firstIndex(of: 258) == 12, "\(name) thinking seed \(seed): the think block was constrained")
        }
        // A cut reply: length, and what came is a prefix of valid JSON.
        let cut = try run(["response_format": ["type": "json_object"], "seed": 1, "max_tokens": 2])
        try require(cut.finish == "length" && cut.tokens.count == 2, "\(name): a cut reply finished \(cut.finish)")
        // Without JSON mode: the same tokens as a fresh backend that never ran JSON mode.
        let plain = try run(["seed": 5, "temperature": 1, "max_tokens": 24])
        let freshModel = try makeFamilyModel(weights, expertBudget: UInt64(routed), context: 2_048) { _ in }
        let fresh = FamilyServingBackend(model: freshModel, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
        let request = try ServingRequest.parse(
            JSONSerialization.data(withJSONObject: [
                "model": "synthetic", "messages": [["role": "user", "content": "Reply in JSON."]], "max_tokens": 24,
                "reasoning_effort": "none", "origami_return_tokens": true, "seed": 5, "temperature": 1,
            ]), contextTier: 2_048)
        let reference = try fresh.prepare(request, cancellation: GenerationControl()).run { _ in true }
        try require(reference.outputTokens == plain.tokens, "\(name): a reply without JSON mode changed")
    }
}
