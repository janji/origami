import Darwin
import Foundation
import OrigamiEngine

// The serving fuzz targets (`Fuzz.swift`): raw HTTP requests through the ingress decoder and every endpoint's JSON
// decoding, the reply path (tool-call splitter and wire writers), the chat templates' request-borne inputs, and a live
// loopback server whose clients send mutated requests and disconnect mid-reply.

private func httpRequest(_ method: String, _ target: String, _ body: String? = nil, _ extra: String = "") -> Data {
    var head = "\(method) \(target) HTTP/1.1\r\nHost: 127.0.0.1\r\n" + extra
    if let body { head += "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n" }
    return Data((head + "\r\n" + (body ?? "")).utf8)
}

private let fuzzSchema =
    #"{"type":"object","properties":{"city":{"type":"string","description":"City"},"days":{"type":"integer"},"#
    + #""opts":{"type":"object","properties":{"unit":{"enum":["c","f"]}}}},"required":["city"]}"#
/// Model replies in each family's tool-call markup, as user text: the reply path feeds every message's text through the
/// splitter as if the model had written it.
private let fuzzReplies = [
    "Sure.<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    "<think>hmm</think>Ok<tool_call>weather<arg_key>city</arg_key><arg_value>Rome</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
    "Fine\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
]

/// Real request bodies of every endpoint (the API checks' shapes), wrapped as HTTP requests.
private let httpSeeds: [Data] = {
    let tools = #"[{"type":"function","function":{"name":"weather","description":"Weather <b>now</b>","parameters":\#(fuzzSchema)}}]"#
    let history =
        #"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Rome\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"sun"}"#
    var seeds = [
        httpRequest("POST", "/v1/chat/completions", #"{"model":"m","messages":[{"role":"user","content":"Hi"}]}"#),
        httpRequest(
            "POST", "/v1/chat/completions",
            #"{"model":"m-faster","stream":true,"stream_options":{"include_usage":true},"temperature":0.7,"top_p":0.9,"top_k":20,"#
                + #""min_p":0.05,"presence_penalty":1.5,"seed":42,"stop":["END","\n\n"],"max_tokens":300,"response_format":{"type":"text"},"#
                + #""reasoning_effort":"low","tool_choice":{"type":"function","function":{"name":"weather"}},"parallel_tool_calls":false,"#
                + #""tools":\#(tools),"messages":[{"role":"system","content":"Rules"},{"role":"user","content":"Weather?"},\#(history),"#
                + #"{"role":"user","content":[{"type":"text","text":"And"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]}"#
        ),
        httpRequest(
            "POST", "/v1/chat/completions",
            #"{"messages":[{"role":"user","content":"x"}],"reasoning":{"effort":"high","max_tokens":64,"exclude":true},"#
                + #""chat_template_kwargs":{"enable_thinking":true},"options":{"num_ctx":4096},"origami_mode":"exact","metadata":{"a":"b"},"#
                + #""origami_tune":{},"prompt_cache_policy":"no_store","status_messages":true,"return_progress":true,"logit_bias":{}}"#),
        httpRequest(
            "POST", "/v1/messages",
            #"{"model":"claude","max_tokens":300,"system":[{"type":"text","text":"Rules"}],"stop_sequences":["END"],"#
                + #""thinking":{"type":"enabled","budget_tokens":1024},"tool_choice":{"type":"any"},"#
                + #""tools":[{"name":"weather","description":"Weather","input_schema":\#(fuzzSchema)},{"type":"web_search_20250305","name":"web_search"}],"#
                + #""messages":[{"role":"user","content":[{"type":"text","text":"Look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}]},"#
                + #"{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},{"type":"tool_use","id":"t1","name":"weather","input":{"city":"Rome"}}]},"#
                + #"{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sun"},{"type":"text","text":"And?"}]}]}"#),
        httpRequest(
            "POST", "/v1/responses",
            #"{"model":"gpt","instructions":"Rules","max_output_tokens":200,"reasoning":{"effort":"low","summary":"auto"},"stream":true,"#
                + #""tools":[{"type":"function","name":"weather","description":"Weather","parameters":\#(fuzzSchema)},{"type":"web_search"}],"#
                + #""input":[{"role":"user","content":[{"type":"input_text","text":"Hi"}]},"#
                + #"{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking"}]},"#
                + #"{"type":"function_call","call_id":"c1","name":"weather","arguments":"{\"city\":\"Rome\"}"},"#
                + #"{"type":"function_call_output","call_id":"c1","output":"sun"}]}"#),
        httpRequest(
            "POST", "/api/chat",
            #"{"model":"m","stream":true,"think":true,"options":{"num_ctx":2048,"temperature":0.2},"tools":\#(tools),"#
                + #""messages":[{"role":"user","content":"Hi","images":[]}]}"#),
        httpRequest("POST", "/api/generate", #"{"model":"m","prompt":"Once","system":"Be brief","stream":false,"options":{"seed":3}}"#),
        httpRequest("POST", "/api/generate", #"{"model":"m","prompt":""}"#),
        // JSON mode on every wire.
        httpRequest(
            "POST", "/v1/chat/completions",
            #"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"r","strict":true,"#
                + #""schema":{"type":"object","properties":{"a":{"type":"integer"},"b":{"anyOf":[{"type":"string","enum":["x","y\u00e9"]},"#
                + ##"{"type":"null"},{"$ref":"#/$defs/p"}]},"c":{"type":"array","items":{"type":["number","boolean"]},"minItems":1,"maxItems":3},"##
                + ##""d":{"const":{"k":[1,true]}},"e":{"$ref":"#"}},"required":["a","b"],"additionalProperties":false,"##
                + #""$defs":{"p":{"type":"object","properties":{"x":{"type":"string","description":"d"}},"additionalProperties":{"type":"boolean"}}}}}}}"#
        ),
        httpRequest(
            "POST", "/v1/chat/completions",
            #"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"},"tools":\#(tools),"reasoning_effort":"high"}"#
        ),
        httpRequest(
            "POST", "/v1/responses",
            #"{"input":"x","text":{"format":{"type":"json_schema","name":"r","schema":{"type":"object","properties":{"a":{"type":"string"}}},"strict":false}}}"#
        ),
        httpRequest("POST", "/v1/responses", #"{"input":"x","text":{"format":{"type":"json_object"}}}"#),
        httpRequest("POST", "/api/chat", #"{"model":"m","messages":[{"role":"user","content":"x"}],"format":"json"}"#),
        httpRequest(
            "POST", "/api/generate",
            #"{"model":"m","prompt":"x","format":{"type":"object","properties":{"n":{"type":"number"},"l":{"type":"array"}},"required":["n"]}}"#
        ),
        httpRequest("POST", "/v1/config", #"{"expert_bits":"3","context":"16384"}"#, "Origin: http://127.0.0.1\r\n"),
        httpRequest("POST", "/v1/hot-copy", #"{"build":true,"size":"40GB","drive":"/Volumes/X"}"#, "Origin: http://127.0.0.1\r\n"),
        httpRequest("POST", "/v1/prompts", #"{"text":"You are terse.","name":"terse"}"#),
        httpRequest("PATCH", "/v1/prompts/abc", #"{"name":"x"}"#),
        httpRequest("GET", "/v1/models/qwen%20x?y=1"),
        httpRequest("GET", "/v1/plan?context=8192&quality=exact&x=%ZZ"),
        httpRequest("GET", "/health", nil, "Authorization: Bearer k\r\nX-Origami-Keep: notes\r\n"),
        httpRequest("HEAD", "/v1/dashboard", nil, "Accept: text/html\r\n"),
        httpRequest("GET", "/api/tags"),
        // A trusted proxy's client address (`HTTPIngressPolicy.forwardedPeer`), repeated fields joined by commas.
        httpRequest("GET", "/v1/models", nil, "X-Forwarded-For: 192.168.1.20\r\n"),
        httpRequest("GET", "/v1/models", nil, "X-Forwarded-For: 203.0.113.9, [2001:db8::7]\r\nX-Forwarded-For: 10.0.0.1\r\n"),
        httpRequest("GET", "/v1/models", nil, "X-Forwarded-For: unknown, 10.0.0.300\r\n"),
    ]
    for reply in fuzzReplies {
        let body = try! JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": reply]], "tools": try! JSONSerialization.jsonObject(with: Data(tools.utf8)),
            "stream": true,
        ])
        seeds.append(httpRequest("POST", "/v1/chat/completions", String(decoding: body, as: UTF8.self)))
    }
    return seeds
}()

/// Head and body of a request, when the input has the usual shape.
private func httpSplit(_ input: Data) -> (head: [String], body: Data)? {
    guard let end = input.range(of: Data("\r\n\r\n".utf8)) else { return nil }
    let head = String(decoding: input[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
    return (head, input.subdata(in: end.upperBound..<input.count))
}

private func httpMutate(_ rng: inout FuzzRNG, _ input: Data, _ corpus: [Data]) -> Data {
    guard var (head, body) = httpSplit(input), !head.isEmpty else { return input }
    var fixLength = rng.chance(85)
    switch rng.below(9) {
    case 0, 1, 2, 3:
        if let tree = fuzzParseJSON(body) { body = Data(fuzzJSONText(fuzzMutateJSON(&rng, tree)).utf8) }
    case 4:
        // The request line: method, target and version.
        var parts = head[0].components(separatedBy: " ")
        let i = rng.below(parts.count)
        parts[i] = rng.pick([
            "GET", "POST", "HEAD", "PATCH", "DELETE", "get", "", "HTTP/1.0", "HTTP/1.1 ", "/v1/chat/completions", "/v1/messages",
            "/v1/responses", "/api/chat", "/api/generate", "/v1/config", "/v1/hot-copy", "/v1/prompts/", "/v1/prompts/%00",
            "/v1/models/%E2%82", "/v1/plan?a=%", "*", "http://127.0.0.1/health", "/health?" + String(repeating: "a", count: 9000),
            "/\u{7f}", "/é", "//", "/v1/status", "/",
        ])
        head[0] = parts.joined(separator: rng.chance(90) ? " " : "  ")
    case 5:
        // A header added, repeated or changed.
        let line = rng.pick([
            "Content-Length: 0", "Content-Length: -1", "Content-Length: +5", "Content-Length: 99999999999999999999",
            "Content-Length: 1, 1", "Content-Length:  ", "Transfer-Encoding: chunked", "Transfer-Encoding: identity",
            "Expect: 100-continue",
            "Host: evil.example", "Host: 127.0.0.1:1", "Host: [::1]", "Host: localhost.", "Host:", "Origin: null",
            "Origin: http://evil.example", "Origin: http://127.0.0.1", "Content-Type: text/plain",
            "Content-Type: application/json; charset=x",
            "X-Origami-Keep: a b", "X-Origami-Keep: notes", "Authorization: Bearer x", "x-api-key: y", " folded: line", "\tfold",
            "Bad Header: x", ": empty", "Accept: text/html", "Connection: keep-alive", "X: \u{80}\u{ff}", "X: a\rb",
        ])
        if line.hasPrefix("Content-Length") { fixLength = false }
        head.insert(line, at: 1 + rng.below(head.count))
    case 6 where head.count > 1:
        let i = 1 + rng.below(head.count - 1)
        if rng.chance(50) { head.remove(at: i) } else { head.insert(head[i], at: i) }
    case 7:
        // A chunked body, as clients that do not know the length send it.
        let chunked = String(body.count, radix: 16) + "\r\n" + String(decoding: body, as: UTF8.self) + "\r\n0\r\n\r\n"
        body = Data(chunked.utf8)
        head.append("Transfer-Encoding: chunked")
        head.removeAll { $0.lowercased().hasPrefix("content-length") }
        fixLength = false
    default:
        fixLength = false
        body.append(Data(rng.pick(["x", "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", ""]).utf8))
    }
    if fixLength {
        head.removeAll { $0.lowercased().hasPrefix("content-length:") }
        if !body.isEmpty || head[0].hasPrefix("POST") { head.append("Content-Length: \(body.count)") }
    }
    return Data(head.joined(separator: "\r\n").utf8) + Data("\r\n\r\n".utf8) + body
}

/// A deterministic hash of the input: its chunking and the server's knobs come from it, so a replay repeats the run.
private func fuzzHash(_ data: Data) -> UInt64 {
    var hash: UInt64 = 0xCBF2_9CE4_8422_2325
    for byte in data { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01B3 }
    return hash
}

/// Every message's text, as the model's reply for the splitter.
private func replyText(_ request: ServingRequest) -> String {
    request.messages.map { message -> String in
        if case .string(let text)? = message["content"] { return text }
        return ""
    }.joined(separator: "\n")
}

/// The decoded request through the routing the server does before a backend runs: query forms, model ids, control
/// bodies, the wire translation and `ServingRequest.parse`, then the reply path for each family.
private func httpServe(_ request: BoundedHTTPRequest) throws {
    let path = String(request.target.split(separator: "?").first ?? "")
    _ = formQueryItems(request.target)
    _ = HTTPIngressPolicy.isControl(method: request.method, target: request.target)
    _ = LocalHTTPServer.isDashboard(path: path, headers: request.headers)
    _ = LocalHTTPServer.carriesKey("k", headers: request.headers)
    _ = HostGuard.ownPage(origin: request.headers["origin"], host: request.headers["host"] ?? "")
    _ = String(path.dropFirst("/v1/models/".count)).removingPercentEncoding
    if let keep = request.headers["x-origami-keep"] { _ = ChatCommand.isName(keep) }
    guard (try? HTTPIngressPolicy.checkJSON(request.body)) != nil else { return }
    _ = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: String]
    let formats: [String: WireFormat] = [
        "/v1/chat/completions": .openAI, "/v1/messages": .anthropic, "/v1/responses": .responses, "/api/chat": .ollama,
        "/api/generate": .ollama,
    ]
    let format = formats[path] ?? .openAI
    let generate = path == "/api/generate"
    let parsed: ServingRequest
    do {
        parsed = try ServingRequest.parse(
            WireTranslation.chatBody(request.body, format: format, generate: generate), contextTier: 8192, maximumStructureBytes: 256 << 10)
    } catch { return }
    guard parsed.contextLimit >= 1, parsed.contextLimit <= 8192, parsed.maximumTokens >= 1, parsed.maximumTokens <= parsed.contextLimit,
        !parsed.messages.isEmpty
    else {
        throw FuzzFinding(description: "parse accepted context \(parsed.contextLimit), output budget \(parsed.maximumTokens)")
    }
    var images: [String] = []
    _ = try? TokenizerStack.normalizeMessages(parsed.messages, images: &images)
    _ = try? parsed.outputBudget(promptTokens: Int(fuzzHash(request.body) % 9000))
    let text = replyText(parsed)
    if let grammar = parsed.responseFormat { try jsonModeFuzz(grammar, text: Array(text.utf8), body: request.body) }
    for family in [ModelFamily.qwen, .glm, .deepseek] {
        let stream = ServingStream(request: parsed, family: family, maximumBytes: 1 << 20)
        let writer = WireWriter(
            format: format, id: "fuzz", created: 0, model: parsed.model, includeUsage: parsed.includeUsage, generate: generate,
            request: parsed)
        do {
            _ = try writer.open()
            var deltas: [[String: Any]] = []
            var rest = Substring(text)
            var cut = fuzzHash(Data(text.utf8))
            while !rest.isEmpty {
                let size = 1 + Int(cut % 7)
                cut = cut >> 3 | cut << 61
                deltas += try stream.push(String(rest.prefix(size)))
                rest = rest.dropFirst(size)
            }
            let (last, reason) = try stream.finish(reason: "stop")
            for delta in deltas + last { _ = try writer.frames(delta) }
            let usage = try ServingWire.usage(GenerationStatistics(promptTokens: 3, completionTokens: 2))
            _ = try writer.close(reason: reason, usage: usage)
            _ = writer.body(message: ["role": "assistant", "content": text], reason: reason, usage: usage)
        } catch is ServingFailure {} catch is EngineError {}
    }
}

/// Bytes, a small vocabulary and the end token for JSON mode's fuzzing.
private let fuzzJSONVocabulary: JSONVocabulary = {
    let pieces =
        (0..<256).map { [UInt8($0)] }
        + [#"{""#, #"":"#, #"","#, #""}"#, "true", "null", "12", #"\u00"#, #"\ud83d\ude00"#, "  "].map { Array($0.utf8) }
    return JSONVocabulary(size: pieces.count + 1) { $0 < pieces.count ? (pieces[$0], false) : (Array("<end>".utf8), true) }
}()

/// The parsed grammar against the acceptor: the reply text and the body fed byte by byte (a complete value must parse),
/// and a walk over allowed tokens chosen by the input's hash, which must never find nothing allowed.
private func jsonModeFuzz(_ grammar: JSONGrammar, text: [UInt8], body: Data) throws {
    let rules = JSONRules(grammar: grammar, leadingNewlines: 2)
    for bytes in [text, Array(body)] {
        var state = JSONAcceptor(rules)
        var taken = 0
        for byte in bytes {
            guard state.step(byte, rules) else { break }
            taken += 1
        }
        if state.complete(rules), (try? JSONSerialization.jsonObject(with: Data(bytes.prefix(taken)), options: .fragmentsAllowed)) == nil {
            throw FuzzFinding(description: "JSON mode accepted \(String(decoding: bytes.prefix(taken), as: UTF8.self))")
        }
    }
    let end = fuzzJSONVocabulary.size - 1
    let constraint = JSONConstraint(
        grammar: grammar, vocabulary: fuzzJSONVocabulary, stops: [end], thinkEnd: nil, thinking: false, toolOpener: nil)
    var walked: [UInt8] = []
    var cut = fuzzHash(body)
    for _ in 0..<48 {
        let bits = constraint.allowedBits()
        let allowed = (0..<fuzzJSONVocabulary.size).filter { bits[$0 >> 6] >> UInt64($0 & 63) & 1 == 1 }
        guard !allowed.isEmpty else {
            throw FuzzFinding(description: "JSON mode dead end after \(String(decoding: walked, as: UTF8.self))")
        }
        if allowed.contains(end) {
            guard (try? JSONSerialization.jsonObject(with: Data(walked), options: .fragmentsAllowed)) != nil else {
                throw FuzzFinding(description: "JSON mode completed unparsable \(String(decoding: walked, as: UTF8.self))")
            }
            if cut & 3 == 0 { break }
        }
        let id =
            allowed.filter { $0 != end }.isEmpty
            ? end : allowed.filter { $0 != end }[Int(cut % UInt64(allowed.count - (allowed.contains(end) ? 1 : 0)))]
        cut = cut >> 7 | cut << 57
        if id == end { break }
        try constraint.accept(id)
        walked += fuzzJSONVocabulary.bytes(of: id)
    }
}

private func httpRun(_ input: Data) throws {
    var cut = fuzzHash(input)
    var decoder = BoundedHTTPRequestDecoder(maximumHeaderBytes: 32 << 10, maximumBodyBytes: 1 << 20, lan: cut & 1 == 1)
    var request: BoundedHTTPRequest?
    var offset = 0
    do {
        // The bytes arrive in pieces of 1 byte to 4 KiB, as a slow or fast client sends them.
        while offset < input.count, request == nil {
            let size = [1, 2, 7, 64, 1_000, 4_096][Int(cut % 6)]
            cut = cut >> 5 | cut << 59
            let end = min(input.count, offset + size)
            request = try decoder.append(input.subdata(in: offset..<end))
            offset = end
        }
        if request != nil, offset < input.count { _ = try decoder.append(input.subdata(in: offset..<input.count)) }
    } catch let failure as ServingFailure {
        guard (400..<600).contains(failure.status) else { throw FuzzFinding(description: "ingress failure status \(failure.status)") }
        if request == nil { return }
    }
    guard let request else { return }
    let declared = Int(request.headers["content-length"] ?? "0") ?? -1
    guard request.body.count == declared, request.target.hasPrefix("/"),
        request.headers.values.allSatisfy({ !$0.contains("\r") && !$0.contains("\n") }), request.headers["transfer-encoding"] == nil
    else {
        throw FuzzFinding(
            description: "ingress accepted \(request.method) \(request.target) with \(request.body.count) of \(declared) body bytes")
    }
    // A forwarded address is refused with a 400 or comes back canonical: parsing its own output gives itself.
    do {
        if let peer = try HTTPIngressPolicy.forwardedPeer(request.headers),
            try HTTPIngressPolicy.forwardedPeer(["x-forwarded-for": peer]) != peer
        {
            throw FuzzFinding(description: "forwarded address \(peer) is not canonical")
        }
    } catch let failure as ServingFailure {
        guard failure.status == 400 else { throw FuzzFinding(description: "forwarded address failure status \(failure.status)") }
    }
    try httpServe(request)
}

func httpFuzzTarget() -> FuzzTarget { FuzzTarget(name: "http", seeds: { httpSeeds }, mutate: httpMutate, run: httpRun) }

// MARK: chat templates

/// The control tokens each family's template writes; typed into a request they must stay text.
func templateSpecials(_ template: String) -> [String] {
    var found = Set<String>()
    let pattern = try! NSRegularExpression(pattern: #"<\|[A-Za-z_▁]{1,40}\|>|<｜[^｜<>]{1,40}｜>"#)
    for match in pattern.matches(in: template, range: NSRange(template.startIndex..., in: template)) {
        found.insert(String(template[Range(match.range, in: template)!]))
    }
    for tag in [
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<arg_key>", "</arg_key>",
        "<arg_value>",
        "</arg_value>", "[gMASK]", "<sop>",
    ] where template.contains(tag) {
        found.insert(tag)
    }
    return found.sorted()
}

/// A tokenizer for `template` whose vocabulary is the 256 byte symbols, one merge, and `specials` as control tokens.
func templateTokenizer(_ template: String, specials: [String]) throws -> TokenizerStack {
    let folder = CheckFixtures.scratch.appendingPathComponent("fuzz-template-\(getpid())-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    var shifted = 0
    let alphabet = (0..<256).map { b -> String in
        if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) { return String(UnicodeScalar(UInt8(b))) }
        shifted += 1
        return String(UnicodeScalar(UInt32(255 + shifted))!)
    }
    let gguf = folder.appendingPathComponent("t.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "glm4"),
            ("tokenizer.ggml.tokens", alphabet + ["ab"] + specials), ("tokenizer.ggml.merges", ["a b"]),
            ("tokenizer.ggml.token_type", [Int](repeating: 1, count: 257) + [Int](repeating: 3, count: specials.count)),
            ("tokenizer.ggml.eos_token_id", 256), ("tokenizer.chat_template", template),
        ], tensors: [], to: gguf)
    let support = try FamilySupport.prepare(gguf, into: folder.appendingPathComponent("support"))
    final class Box: @unchecked Sendable {
        var stack: TokenizerStack?
        var error: Error?
    }
    let box = Box()
    let done = DispatchSemaphore(value: 0)
    Task {
        do { box.stack = try await TokenizerStack(modelDirectory: support) } catch { box.error = error }
        done.signal()
    }
    done.wait()
    if let error = box.error { throw error }
    return box.stack!
}

nonisolated(unsafe) private var templateStacks: [(name: String, stack: TokenizerStack, specials: Set<Int>)]?
private func templateFamilies() throws -> [(name: String, stack: TokenizerStack, specials: Set<Int>)] {
    if let templateStacks { return templateStacks }
    var result: [(name: String, stack: TokenizerStack, specials: Set<Int>)] = []
    for model in ["qwen3.8-flash-next", "glm-5.3-flash", "deepseek-v4.1-flash"] {
        let template = try String(contentsOf: CheckFixtures.file("models/\(model)/chat_template.jinja"), encoding: .utf8)
        let specials = templateSpecials(template)
        result.append((model, try templateTokenizer(template, specials: specials), Set(257..<(257 + specials.count))))
    }
    templateStacks = result
    return result
}

/// Request strings the client controls outside the assistant's own history, with every marker character replaced: the
/// same request without anything that could spell a control token.
private func scrub(_ value: Any, inAssistant: Bool = false) -> Any {
    func clean(_ s: String) -> String {
        s.replacingOccurrences(of: "<", with: "‹").replacingOccurrences(of: "｜", with: "¦").replacingOccurrences(of: "|", with: "¦")
            .replacingOccurrences(of: "[", with: "⟦")
    }
    switch value {
    case let object as [String: Any]:
        let assistant = inAssistant || object["role"] as? String == "assistant"
        return Dictionary(
            uniqueKeysWithValues: object.map { (assistant ? $0.key : clean($0.key), scrub($0.value, inAssistant: assistant)) })
    case let array as [Any]: return array.map { scrub($0, inAssistant: inAssistant) }
    case let string as String: return inAssistant ? string : clean(string)
    default: return value
    }
}

/// The prompt's control tokens for `body` in each family, nil where the request or its template refuses it.
private func templateCounts(_ body: Data) throws -> [Int?]? {
    guard let parsed = try? ServingRequest.parse(body, contextTier: 8192, maximumStructureBytes: 256 << 10),
        let normalized = try? TokenizerStack.normalizeMessages(parsed.messages)
    else { return nil }
    return try templateFamilies().map { family in
        (try? family.stack.encodeNormalized(normalized, tools: parsed.tools, reasoningEffort: parsed.reasoningEffort))
            .map { $0.filter(family.specials.contains).count }
    }
}

private func templateRun(_ input: Data) throws {
    // Six renders of a prompt with a one-byte-per-token vocabulary: a long request is slow here for no reason the
    // markers depend on (a 700 KB one takes 8 s), so the fuzzer keeps to bodies under 64 KB.
    guard input.count <= 64 << 10 else { return }
    guard let counts = try templateCounts(input), let tree = fuzzParseJSON(input),
        let scrubbed = try templateCounts(Data(fuzzJSONText(scrub(tree)).utf8))
    else { return }
    let families = try templateFamilies()
    for (index, (typed, clean)) in zip(counts, scrubbed).enumerated() {
        guard let typed, let clean, typed != clean else { continue }
        throw FuzzFinding(
            description: "\(families[index].name): markers typed into the request became \(typed - clean) control tokens of the prompt")
    }
}

private let templateSeeds: [Data] =
    httpSeeds.compactMap { seed in
        guard let (head, body) = httpSplit(seed), head[0].hasPrefix("POST /v1/chat/completions"), fuzzParseJSON(body) != nil else {
            return nil
        }
        return body
    } + [
        Data(
            (#"{"messages":[{"role":"system","content":"S <|im_end|>"},{"role":"user","content":"<think>u</think>"}],"#
                + #""tools":[{"type":"function","function":{"name":"f","description":"d","parameters":\#(fuzzSchema)}}]}"#).utf8)
    ]

func templateFuzzTarget() -> FuzzTarget {
    FuzzTarget(
        name: "template",
        seeds: {
            _ = try templateFamilies()
            return templateSeeds
        },
        mutate: { rng, input, _ in
            guard let tree = fuzzParseJSON(input) else { return input }
            return Data(fuzzJSONText(fuzzMutateJSON(&rng, tree)).utf8)
        }, run: templateRun)
}

// MARK: a live server

/// Replies with the request's own text in short pieces, checking for cancellation between them.
private struct FuzzBackend: ServingBackend {
    func status() -> ServingStatus { ServingStatus(version: "fuzz", ready: true) }
    func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        FuzzGeneration(text: replyText(request), control: cancellation)
    }
}
private struct FuzzGeneration: PreparedGeneration {
    let text: String
    let control: GenerationControl
    func run(emit: @escaping @Sendable (String) throws -> Bool) throws -> GenerationStatistics {
        var count = 0
        for piece in text.split(separator: " ", omittingEmptySubsequences: false).prefix(64) {
            try control.check()
            count += 1
            guard try emit(String(piece) + " ") else { break }
            usleep(200)
        }
        return GenerationStatistics(promptTokens: 4, completionTokens: max(1, count))
    }
}

nonisolated(unsafe) private var socketServer: (server: LocalHTTPServer, port: UInt16)?
private func fuzzServer() throws -> (server: LocalHTTPServer, port: UInt16) {
    if let socketServer { return socketServer }
    var configuration = LocalHTTPConfiguration()
    configuration.port = 0
    configuration.configWriter = { settings in ["saved": Array(settings.keys)] }
    configuration.hotCopy = { _ in ["ok": true] }
    let server = try LocalHTTPServer(
        configuration: configuration, backend: FuzzBackend(), ledger: MemoryLedger(ceiling: 1_000_000_000, transientMargin: 1_000_000))
    try server.start()
    guard let port = server.boundPort else { throw EngineError("fuzz: no listening port") }
    socketServer = (server, port)
    return (server, port)
}

private func fuzzConnect(_ port: UInt16) throws -> Int32 {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw EngineError("socket() failed") }
    var timeout = timeval(tv_sec: 2, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    var on: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    let connected = withUnsafePointer(to: address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard connected == 0 else {
        close(fd)
        throw EngineError("connect() failed")
    }
    return fd
}

/// GET /v1/status's queue, read whole.
private func fuzzQueueIdle(_ port: UInt16) throws -> Bool {
    let fd = try fuzzConnect(port)
    defer { close(fd) }
    let request = "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    _ = request.withCString { write(fd, $0, strlen($0)) }
    var reply = Data()
    var buffer = [UInt8](repeating: 0, count: 8_192)
    while true {
        let count = read(fd, &buffer, buffer.count)
        if count <= 0 { break }
        reply.append(contentsOf: buffer[..<count])
    }
    guard let split = reply.range(of: Data("\r\n\r\n".utf8)),
        let object = (try? JSONSerialization.jsonObject(with: reply.subdata(in: split.upperBound..<reply.count))) as? [String: Any],
        let queue = object["queue"] as? [String: Any]
    else {
        throw FuzzFinding(description: "the server stopped answering /v1/status: \(String(decoding: reply.prefix(200), as: UTF8.self))")
    }
    return queue["busy"] as? Bool != true && (queue["queued"] as? Int ?? 0) == 0
}

/// Sends the input in pieces, then reads all, some or none of the reply and closes (sometimes with a reset); the server
/// must stay up and free its slot within 2 s.
private func socketRun(_ input: Data) throws {
    let (_, port) = try fuzzServer()
    var cut = fuzzHash(input)
    let fd = try fuzzConnect(port)
    var closed = false
    defer { if !closed { close(fd) } }
    var offset = 0
    while offset < input.count {
        let end = min(input.count, offset + [7, 300, 4_096, 65_536][Int(cut % 4)])
        cut = cut >> 3 | cut << 61
        let sent = input.subdata(in: offset..<end).withUnsafeBytes { send(fd, $0.baseAddress, $0.count, 0) }
        guard sent > 0 else { break }
        offset += sent
    }
    // The client has said all it will: a request the bytes leave incomplete must end at once, not at the read deadline.
    shutdown(fd, SHUT_WR)
    let keep = [0, 1, 100, 400, Int.max][Int(cut % 5)]
    var received = 0
    var buffer = [UInt8](repeating: 0, count: 4_096)
    while received < keep {
        let count = read(fd, &buffer, min(buffer.count, keep - received))
        if count <= 0 { break }
        received += count
    }
    if cut & 32 != 0 {
        var reset = linger(l_onoff: 1, l_linger: 0)
        setsockopt(fd, SOL_SOCKET, SO_LINGER, &reset, socklen_t(MemoryLayout<linger>.size))
    }
    close(fd)
    closed = true
    let deadline = DispatchTime.now().uptimeNanoseconds + 2_000_000_000
    while try !fuzzQueueIdle(port) {
        guard DispatchTime.now().uptimeNanoseconds < deadline else {
            throw FuzzFinding(description: "the request slot was still taken 2 s after the client left")
        }
        usleep(2_000)
    }
}

func socketFuzzTarget() -> FuzzTarget {
    FuzzTarget(
        name: "socket",
        seeds: {
            _ = try fuzzServer()
            return httpSeeds
        }, mutate: httpMutate, run: socketRun)
}
