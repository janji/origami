import Foundation

/// The wire formats clients speak besides OpenAI chat. Anthropic Messages (`/v1/messages`, what Claude Code uses),
/// OpenAI Responses (`/v1/responses`, Codex) and Ollama (`/api/chat`). A request is translated into an OpenAI chat body,
/// so every check and feature of the chat path applies; the reply is written in the client's format from the same
/// OpenAI-shaped deltas.
public enum WireFormat: String, Sendable {
    case openAI = "openai"
    case anthropic, responses, ollama
}

/// Request side of the wires: turns an Anthropic, Responses or Ollama body into the OpenAI chat body
/// `ServingRequest.parse` validates. It parses the client's JSON once and re-serializes it; fields it does not
/// map are dropped, and anything it cannot express (raw prompts, `previous_response_id`) is a 400.
package enum WireTranslation {
    static func failure(_ message: String) -> ServingFailure { ServingFailure(message) }

    /// The OpenAI chat body for a request in `format` (`generate`: Ollama's `/api/generate`).
    /// `vision`: the model reads images (a tool result's images then reach it; else the result says one was left out).
    package static func chatBody(_ data: Data, format: WireFormat, generate: Bool = false, vision: Bool = false) throws -> Data {
        guard format != .openAI else { return data }
        guard let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw failure("Request must be a JSON object")
        }
        let translated: [String: Any]
        switch format {
        case .anthropic: translated = try anthropic(body, vision: vision)
        case .responses: translated = try responses(body)
        case .ollama: translated = try ollama(generate ? ollamaGenerate(body) : body)
        case .openAI: translated = body
        }
        return try JSONSerialization.data(withJSONObject: translated)
    }

    /// `/api/generate` as an `/api/chat` body: the system text, then the prompt with its images. Raw prompts and
    /// fill-in-the-middle need a completion model; Origami serves chat, so they are refused. A custom template is ignored.
    static func ollamaGenerate(_ body: [String: Any]) throws -> [String: Any] {
        if body["raw"] as? Bool == true { throw failure("raw prompts are not supported; Origami applies the model's chat template") }
        if let suffix = body["suffix"] as? String, !suffix.isEmpty { throw failure("suffix (fill in the middle) is not supported") }
        var chat = body
        var user: [String: Any] = ["role": "user", "content": body["prompt"] as? String ?? ""]
        if let images = body["images"] { user["images"] = images }
        let system = (body["system"] as? String).map { [["role": "system", "content": $0]] } ?? []
        chat["messages"] = system + [user]
        for key in ["prompt", "system", "images", "raw", "suffix", "context", "template"] { chat.removeValue(forKey: key) }
        return chat
    }

    private static func json(_ value: Any?) throws -> String {
        guard let value, JSONSerialization.isValidJSONObject(value) else { return "{}" }
        return String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self)
    }
    private static func text(_ content: Any?) -> String {
        if let text = content as? String { return text }
        return (content as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
    }
    /// An Anthropic image block as a chat image part.
    private static func imagePart(_ block: [String: Any]) throws -> [String: Any] {
        let source = block["source"] as? [String: Any] ?? [:]
        let url: String
        switch source["type"] as? String {
        case "base64": url = "data:\(source["media_type"] as? String ?? "image/png");base64,\(source["data"] as? String ?? "")"
        case "url": url = source["url"] as? String ?? ""
        default: throw failure("Images are base64 or url sources")
        }
        return ["type": "image_url", "image_url": ["url": url]]
    }
    private static func copy(_ keys: [String], from body: [String: Any], into out: inout [String: Any]) {
        for key in keys { if let value = body[key] { out[key] = value } }
    }

    static func anthropic(_ body: [String: Any], vision: Bool = false) throws -> [String: Any] {
        var out: [String: Any] = [:]
        var messages: [[String: Any]] = []
        if let system = body["system"] { messages.append(["role": "system", "content": text(system)]) }
        guard let list = body["messages"] as? [[String: Any]] else { throw failure("messages must be an array") }
        for message in list {
            guard let role = message["role"] as? String, ["user", "assistant", "system"].contains(role) else {
                throw failure("Message roles are user, assistant or system")
            }
            // Claude Code sends its environment as a system message after the user's.
            if role == "system" {
                messages.append(["role": "system", "content": text(message["content"])])
                continue
            }
            if let content = message["content"] as? String {
                messages.append(["role": role, "content": content])
                continue
            }
            guard let blocks = message["content"] as? [[String: Any]] else {
                throw failure("Message content must be text or content blocks")
            }
            if role == "assistant" {
                var content = ""
                var reasoning = ""
                var calls: [[String: Any]] = []
                for block in blocks {
                    switch block["type"] as? String {
                    case "text": content += block["text"] as? String ?? ""
                    case "thinking": reasoning += block["thinking"] as? String ?? ""
                    case "redacted_thinking": break
                    case "tool_use":
                        guard let id = block["id"] as? String, let name = block["name"] as? String else {
                            throw failure("tool_use needs an id and a name")
                        }
                        calls.append([
                            "id": id, "type": "function",
                            "function": ["name": name, "arguments": try json(block["input"] ?? [String: Any]())],
                        ])
                    default: throw failure("Unsupported assistant content block \(block["type"] ?? "?")")
                    }
                }
                var turn: [String: Any] = ["role": "assistant", "content": calls.isEmpty || !content.isEmpty ? content : NSNull()]
                if !reasoning.isEmpty { turn["reasoning_content"] = reasoning }
                if !calls.isEmpty { turn["tool_calls"] = calls }
                messages.append(turn)
            } else {
                var parts: [[String: Any]] = []
                for block in blocks {
                    switch block["type"] as? String {
                    case "text": parts.append(["type": "text", "text": block["text"] as? String ?? ""])
                    case "image": parts.append(try imagePart(block))
                    case "tool_result":
                        guard let id = block["tool_use_id"] as? String else { throw failure("tool_result needs a tool_use_id") }
                        var result = text(block["content"])
                        // A tool message holds text: its images follow the results as the user's, when the model sees them.
                        for image in (block["content"] as? [[String: Any]] ?? []) where image["type"] as? String == "image" {
                            if vision {
                                parts.append(try imagePart(image))
                            } else {
                                result += "\n[an image was left out: this model reads text only]"
                            }
                        }
                        messages.append([
                            "role": "tool", "tool_call_id": id, "content": block["is_error"] as? Bool == true ? "Error: " + result : result,
                        ])
                    default: throw failure("Unsupported content block \(block["type"] ?? "?")")
                    }
                }
                if !parts.isEmpty { messages.append(["role": "user", "content": parts]) }
            }
        }
        out["messages"] = messages
        if let tools = body["tools"] as? [[String: Any]] {
            // Anthropic's server tools (web search and the like) run on Anthropic's side; only custom tools reach the model.
            out["tools"] = tools.filter { ($0["type"] as? String ?? "custom") == "custom" }.map { tool in
                [
                    "type": "function",
                    "function": [
                        "name": tool["name"] ?? "", "description": tool["description"] ?? "",
                        "parameters": tool["input_schema"] ?? ["type": "object"],
                    ],
                ] as [String: Any]
            }
        }
        if let choice = body["tool_choice"] as? [String: Any] {
            switch choice["type"] as? String {
            case "any": out["tool_choice"] = "required"
            case "tool": out["tool_choice"] = ["type": "function", "function": ["name": choice["name"] ?? ""]]
            case "none": out["tool_choice"] = "none"
            default: out["tool_choice"] = "auto"
            }
            if choice["disable_parallel_tool_use"] as? Bool == true { out["parallel_tool_calls"] = false }
        }
        copy(["model", "max_tokens", "temperature", "top_p", "top_k", "thinking", "stream"], from: body, into: &out)
        // `output_config.effort` (Claude Code sends it with adaptive thinking) is the thinking effort unless thinking is off.
        if let effort = (body["output_config"] as? [String: Any])?["effort"] as? String, ServingDefaults.efforts.contains(effort),
            (body["thinking"] as? [String: Any])?["type"] as? String != "disabled"
        {
            out["reasoning_effort"] = effort
        }
        if let stops = body["stop_sequences"] { out["stop"] = stops }
        if out["stream"] as? Bool == true { out["stream_options"] = ["include_usage": true] }
        return out
    }

    static func responses(_ body: [String: Any]) throws -> [String: Any] {
        if let previous = body["previous_response_id"], !(previous is NSNull) {
            throw failure("previous_response_id is not supported: Origami keeps no responses; send the whole conversation as input")
        }
        var out: [String: Any] = [:]
        var messages: [[String: Any]] = []
        if let instructions = body["instructions"] as? String { messages.append(["role": "system", "content": instructions]) }
        func parts(_ content: Any?) throws -> Any {
            if let text = content as? String { return text }
            guard let list = content as? [[String: Any]] else { throw failure("Message content must be text or parts") }
            return try list.map { part -> [String: Any] in
                switch part["type"] as? String {
                case "input_text", "output_text", "text": return ["type": "text", "text": part["text"] as? String ?? ""]
                case "refusal": return ["type": "text", "text": part["refusal"] as? String ?? ""]
                case "input_image":
                    let url = (part["image_url"] as? String) ?? ((part["image_url"] as? [String: Any])?["url"] as? String)
                    guard let url else { throw failure("input_image needs an image_url (files are not supported)") }
                    return ["type": "image_url", "image_url": ["url": url]]
                default: throw failure("Unsupported content part \(part["type"] ?? "?")")
                }
            }
        }
        if let input = body["input"] as? String {
            messages.append(["role": "user", "content": input])
        } else if let items = body["input"] as? [[String: Any]] {
            var reasoning = ""
            for item in items {
                switch item["type"] as? String ?? "message" {
                case "message":
                    guard let role = item["role"] as? String else { throw failure("Input messages need a role") }
                    if role == "assistant" {
                        var turn: [String: Any] = ["role": "assistant", "content": text(item["content"])]
                        if !reasoning.isEmpty { turn["reasoning_content"] = reasoning }
                        reasoning = ""
                        messages.append(turn)
                    } else {
                        // Reasoning not followed by its turn (a turn stopped while thinking) is no later turn's.
                        reasoning = ""
                        messages.append(["role": role, "content": try parts(item["content"])])
                    }
                case "function_call":
                    guard let id = item["call_id"] as? String, let name = item["name"] as? String else {
                        throw failure("function_call needs a call_id and a name")
                    }
                    let call: [String: Any] = [
                        "id": id, "type": "function", "function": ["name": name, "arguments": item["arguments"] as? String ?? "{}"],
                    ]
                    // Calls that follow an assistant message (or each other) belong to that turn.
                    if reasoning.isEmpty, var last = messages.last, last["role"] as? String == "assistant" {
                        last["tool_calls"] = (last["tool_calls"] as? [[String: Any]] ?? []) + [call]
                        if (last["content"] as? String)?.isEmpty ?? true { last["content"] = NSNull() }
                        messages[messages.count - 1] = last
                    } else {
                        var turn: [String: Any] = ["role": "assistant", "content": NSNull(), "tool_calls": [call]]
                        if !reasoning.isEmpty { turn["reasoning_content"] = reasoning }
                        reasoning = ""
                        messages.append(turn)
                    }
                case "function_call_output":
                    guard let id = item["call_id"] as? String else { throw failure("function_call_output needs a call_id") }
                    reasoning = ""
                    messages.append(["role": "tool", "tool_call_id": id, "content": text(item["output"])])
                // The reasoning before an assistant message or call is that turn's, as the chat wire's reasoning_content.
                case "reasoning":
                    let pieces = (item["content"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }
                    reasoning +=
                        (pieces.isEmpty ? (item["summary"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String } : pieces)
                        .joined(separator: "\n")
                default: throw failure("Unsupported input item \(item["type"] ?? "?")")
                }
            }
        } else {
            throw failure("input must be text or a list of items")
        }
        out["messages"] = messages
        if let tools = body["tools"] as? [[String: Any]] {
            // Built-in tools (web search, file search) run on OpenAI's side; function tools reach the model. A custom tool
            // (free-form text input, no JSON schema) is not a function call the models' templates can express.
            if let custom = tools.first(where: { $0["type"] as? String == "custom" }) {
                throw failure("The custom tool \(custom["name"] as? String ?? "?") is not supported: declare it as a function tool")
            }
            out["tools"] = tools.filter { $0["type"] as? String == "function" }.map { tool in
                [
                    "type": "function",
                    "function": [
                        "name": tool["name"] ?? "", "description": tool["description"] ?? "",
                        "parameters": tool["parameters"] ?? ["type": "object"],
                    ],
                ] as [String: Any]
            }
        }
        if let choice = body["tool_choice"] as? String {
            out["tool_choice"] = choice
        } else if let choice = body["tool_choice"] as? [String: Any], let name = choice["name"] {
            out["tool_choice"] = ["type": "function", "function": ["name": name]]
        }
        // JSON mode: text.format carries the schema's fields beside its type; chat nests them under json_schema.
        if let format = (body["text"] as? [String: Any])?["format"] {
            guard let format = format as? [String: Any], let type = format["type"] as? String else {
                throw failure("text.format must be an object with a type")
            }
            switch type {
            case "text": break
            case "json_object": out["response_format"] = ["type": "json_object"]
            case "json_schema":
                var specification: [String: Any] = [:]
                for key in ["name", "schema", "strict", "description"] where format[key] != nil { specification[key] = format[key] }
                out["response_format"] = ["type": "json_schema", "json_schema": specification]
            default: throw failure("text.format type must be \"text\", \"json_object\" or \"json_schema\"")
            }
        }
        if let effort = (body["reasoning"] as? [String: Any])?["effort"] { out["reasoning_effort"] = effort }
        copy(["model", "max_output_tokens", "temperature", "top_p", "parallel_tool_calls", "stream"], from: body, into: &out)
        if out["stream"] as? Bool == true { out["stream_options"] = ["include_usage": true] }
        return out
    }

    static func ollama(_ body: [String: Any]) throws -> [String: Any] {
        var out: [String: Any] = [:]
        var messages: [[String: Any]] = []
        var pending: [String] = []
        var calls = 0
        guard let list = body["messages"] as? [[String: Any]] else { throw failure("messages must be an array") }
        for message in list {
            guard let role = message["role"] as? String else { throw failure("Messages need a role") }
            var turn: [String: Any] = ["role": role, "content": message["content"] as? String ?? ""]
            if let images = message["images"] as? [String], !images.isEmpty {
                turn["content"] =
                    [["type": "text", "text": message["content"] as? String ?? ""]]
                    + images.map { ["type": "image_url", "image_url": ["url": $0.hasPrefix("data:") ? $0 : "data:image/*;base64," + $0]] }
            }
            if let thinking = message["thinking"] as? String, role == "assistant" { turn["reasoning_content"] = thinking }
            if let list = message["tool_calls"] as? [[String: Any]], role == "assistant" {
                // Ollama's calls carry no ids; tool results answer them in order.
                turn["tool_calls"] = try list.map { call -> [String: Any] in
                    let function = call["function"] as? [String: Any] ?? [:]
                    let id = "call_\(calls)"
                    calls += 1
                    pending.append(id)
                    let arguments =
                        function["arguments"] is String
                        ? function["arguments"] as! String : try json(function["arguments"] ?? [String: Any]())
                    return ["id": id, "type": "function", "function": ["name": function["name"] ?? "", "arguments": arguments]]
                }
                if (turn["content"] as? String)?.isEmpty ?? false { turn["content"] = NSNull() }
            }
            if role == "tool" {
                guard !pending.isEmpty else { throw failure("A tool message answers no call") }
                turn["tool_call_id"] = pending.removeFirst()
            }
            messages.append(turn)
        }
        out["messages"] = messages
        // JSON mode: "json" for any JSON object, or a JSON schema.
        switch body["format"] {
        case nil, is NSNull: break
        case let name as String where name.isEmpty: break
        case let name as String where name == "json": out["response_format"] = ["type": "json_object"]
        case let schema as [String: Any]: out["response_format"] = ["type": "json_schema", "json_schema": ["schema": schema]]
        default: throw failure("format must be \"json\" or a JSON schema")
        }
        if let think = body["think"] as? String {
            out["reasoning_effort"] = think
        } else if let think = body["think"] {
            out["think"] = think
        }
        var options = body["options"] as? [String: Any] ?? [:]
        for key in ["temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "seed", "stop"] {
            if let value = options.removeValue(forKey: key) { out[key] = value }
        }
        if let tokens = options.removeValue(forKey: "num_predict") as? Int, tokens > 0 { out["max_tokens"] = tokens }
        if let context = options["num_ctx"] { out["options"] = ["num_ctx": context] }
        copy(["model", "tools"], from: body, into: &out)
        out["stream"] = body["stream"] as? Bool ?? true
        out["stream_options"] = ["include_usage": true]
        return out
    }
}

/// Writes one reply in a wire format from the OpenAI-shaped deltas `ServingStream` produces. The OpenAI writer
/// produces the bytes the chat path always sent. Thread-safe: status lines come from the keepalive timer.
package final class WireWriter: @unchecked Sendable {
    package let format: WireFormat
    private let lock = NSLock()
    private let id: String, created: Int, model: String, includeUsage: Bool
    private var sequence = 0
    // Anthropic and Responses: the open block or item and what it has gathered.
    private var openKind: String?, openIndex = -1
    private var items: [[String: Any]] = [], itemText = "", itemArguments = ""
    private var ollamaCalls: [[String: Any]] = []
    /// Ollama's `/api/generate`: lines carry `response` text instead of a `message`.
    private let generate: Bool
    /// Ollama's `created_at`, formatted once.
    private let timestamp: String
    /// The Responses object's echo of the request's tool settings (as applied: `tool_choice` none hides the tools).
    private let toolEcho: [String: Any]

    package init(
        format: WireFormat, id: String, created: Int, model: String, includeUsage: Bool, generate: Bool = false,
        request: ServingRequest? = nil
    ) {
        self.format = format
        self.id = id
        self.created = created
        self.model = model
        self.includeUsage = includeUsage
        self.generate = generate
        timestamp = format == .ollama ? ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: TimeInterval(created))) : ""
        guard format == .responses else {
            toolEcho = [:]
            return
        }
        let choice: Any
        switch request?.toolChoice ?? .auto {
        case .auto: choice = "auto"
        case .none: choice = "none"
        case .required: choice = "required"
        case .named(let name): choice = ["type": "function", "name": name]
        }
        toolEcho = [
            "parallel_tool_calls": request?.parallelToolCalls ?? true, "tool_choice": choice,
            "tools": (request?.definitions ?? []).map {
                [
                    "type": "function", "name": $0.name, "description": $0.description,
                    "parameters": WireWriter.object($0.parameters.jsonText), "strict": false,
                ] as [String: Any]
            },
        ]
    }
    package var contentType: String { format == .ollama ? "application/x-ndjson" : "text/event-stream; charset=utf-8" }
    /// An idle-time frame the client ignores; NDJSON has none.
    package var keepAlive: Data? { format == .ollama ? nil : ServingWire.keepAlive }

    private func chunk(_ delta: [String: Any], finish: String?) -> [String: Any] {
        [
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]],
        ]
    }
    private func event(_ type: String, _ payload: [String: Any]) throws -> Data {
        var payload = payload
        payload["type"] = type
        if format == .responses {
            payload["sequence_number"] = sequence
            sequence += 1
        }
        var data = Data("event: \(type)\ndata: ".utf8)
        data.append(try ServingWire.json(payload))
        data.append(Data("\n\n".utf8))
        return data
    }
    private func line(_ object: [String: Any]) throws -> Data {
        var data = try ServingWire.json(object)
        data.append(0x0a)
        return data
    }

    /// The stream's first frames.
    package func open() throws -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        switch format {
        case .openAI: return [try ServingWire.sse(chunk(["role": "assistant"], finish: nil))]
        case .anthropic:
            return [
                try event(
                    "message_start",
                    [
                        "message": [
                            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
                            "stop_reason": NSNull(), "stop_sequence": NSNull(), "usage": ["input_tokens": 0, "output_tokens": 0],
                        ]
                    ])
            ]
        case .responses:
            let response = responseObject(status: "in_progress", usage: nil, incomplete: false)
            return [try event("response.created", ["response": response]), try event("response.in_progress", ["response": response])]
        case .ollama: return []
        }
    }

    /// Frames for one delta: `content`, `reasoning_content` or `tool_calls` (a start with id and name, then arguments).
    package func frames(_ delta: [String: Any]) throws -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        switch format {
        case .openAI: return [try ServingWire.sse(chunk(delta, finish: nil))]
        case .ollama:
            if let call = (delta["tool_calls"] as? [[String: Any]])?.first {
                let function = call["function"] as? [String: Any] ?? [:]
                if call["id"] != nil {
                    ollamaCalls.append(["name": function["name"] ?? "", "arguments": ""])
                } else if !ollamaCalls.isEmpty {
                    ollamaCalls[ollamaCalls.count - 1]["arguments"] =
                        (ollamaCalls.last!["arguments"] as? String ?? "") + (function["arguments"] as? String ?? "")
                }
                return []
            }
            var message: [String: Any] = ["role": "assistant", "content": delta["content"] as? String ?? ""]
            if let thought = delta["reasoning_content"] as? String { message["thinking"] = thought }
            return [try line(["model": model, "created_at": timestamp, "done": false].merging(ollamaReply(message)) { $1 })]
        case .anthropic, .responses:
            var frames: [Data] = []
            if let thought = delta["reasoning_content"] as? String {
                frames += try enter("thinking", start: [:]) + [try textDelta(thought, kind: "thinking")]
            } else if let text = delta["content"] as? String {
                frames += try enter("text", start: [:]) + [try textDelta(text, kind: "text")]
            } else if let call = (delta["tool_calls"] as? [[String: Any]])?.first {
                let function = call["function"] as? [String: Any] ?? [:]
                if let callID = call["id"] as? String {
                    frames += try enter("tool", start: ["id": callID, "name": function["name"] as? String ?? ""], force: true)
                } else if let arguments = function["arguments"] as? String, !arguments.isEmpty {
                    itemArguments += arguments
                    frames.append(
                        format == .anthropic
                            ? try event(
                                "content_block_delta",
                                ["index": openIndex, "delta": ["type": "input_json_delta", "partial_json": arguments]])
                            : try event(
                                "response.function_call_arguments.delta",
                                ["item_id": items[openIndex]["id"]!, "output_index": openIndex, "delta": arguments]))
                }
            }
            return frames
        }
    }

    /// Opens a block (Anthropic) or output item (Responses) of `kind` unless one is open already.
    private func enter(_ kind: String, start: [String: Any], force: Bool = false) throws -> [Data] {
        if openKind == kind && !force { return [] }
        var frames = try leave()
        openKind = kind
        openIndex += 1
        itemText = ""
        itemArguments = ""
        if format == .anthropic {
            let block: [String: Any]
            switch kind {
            case "thinking": block = ["type": "thinking", "thinking": "", "signature": ""]
            case "text": block = ["type": "text", "text": ""]
            default: block = ["type": "tool_use", "id": start["id"] ?? "", "name": start["name"] ?? "", "input": [String: Any]()]
            }
            frames.append(try event("content_block_start", ["index": openIndex, "content_block": block]))
        } else {
            let prefix = kind == "thinking" ? "rs" : kind == "text" ? "msg" : "fc"
            var item: [String: Any] = ["id": "\(prefix)_\(id.suffix(24))_\(openIndex)"]
            switch kind {
            case "thinking": item.merge(["type": "reasoning", "summary": []]) { $1 }
            case "text": item.merge(["type": "message", "status": "in_progress", "role": "assistant", "content": []]) { $1 }
            default:
                item.merge([
                    "type": "function_call", "status": "in_progress", "call_id": start["id"] ?? "", "name": start["name"] ?? "",
                    "arguments": "",
                ]) { $1 }
            }
            items.append(item)
            frames.append(try event("response.output_item.added", ["output_index": openIndex, "item": item]))
            if kind == "text" {
                frames.append(
                    try event(
                        "response.content_part.added",
                        [
                            "item_id": item["id"]!, "output_index": openIndex, "content_index": 0,
                            "part": ["type": "output_text", "text": "", "annotations": []],
                        ]))
            } else if kind == "thinking" {
                frames.append(
                    try event(
                        "response.reasoning_summary_part.added",
                        [
                            "item_id": item["id"]!, "output_index": openIndex, "summary_index": 0,
                            "part": ["type": "summary_text", "text": ""],
                        ]))
            }
        }
        return frames
    }
    private func textDelta(_ text: String, kind: String) throws -> Data {
        itemText += text
        if format == .anthropic {
            return try event(
                "content_block_delta",
                [
                    "index": openIndex,
                    "delta": kind == "thinking" ? ["type": "thinking_delta", "thinking": text] : ["type": "text_delta", "text": text],
                ])
        }
        let itemID = items[openIndex]["id"]!
        return kind == "thinking"
            ? try event(
                "response.reasoning_summary_text.delta", ["item_id": itemID, "output_index": openIndex, "summary_index": 0, "delta": text])
            : try event("response.output_text.delta", ["item_id": itemID, "output_index": openIndex, "content_index": 0, "delta": text])
    }
    /// Closes the open block or item.
    private func leave() throws -> [Data] {
        guard let kind = openKind else { return [] }
        openKind = nil
        if format == .anthropic { return [try event("content_block_stop", ["index": openIndex])] }
        var item = items[openIndex]
        var frames: [Data] = []
        let itemID = item["id"]!
        switch kind {
        case "thinking":
            frames.append(
                try event(
                    "response.reasoning_summary_text.done",
                    ["item_id": itemID, "output_index": openIndex, "summary_index": 0, "text": itemText]))
            frames.append(
                try event(
                    "response.reasoning_summary_part.done",
                    ["item_id": itemID, "output_index": openIndex, "summary_index": 0, "part": ["type": "summary_text", "text": itemText]]))
            item["summary"] = [["type": "summary_text", "text": itemText]]
        case "text":
            let part: [String: Any] = ["type": "output_text", "text": itemText, "annotations": []]
            frames.append(
                try event(
                    "response.output_text.done", ["item_id": itemID, "output_index": openIndex, "content_index": 0, "text": itemText]))
            frames.append(
                try event("response.content_part.done", ["item_id": itemID, "output_index": openIndex, "content_index": 0, "part": part]))
            item["content"] = [part]
            item["status"] = "completed"
        default:
            frames.append(
                try event(
                    "response.function_call_arguments.done", ["item_id": itemID, "output_index": openIndex, "arguments": itemArguments]))
            item["arguments"] = itemArguments
            item["status"] = "completed"
        }
        items[openIndex] = item
        frames.append(try event("response.output_item.done", ["output_index": openIndex, "item": item]))
        return frames
    }

    /// The stream's last frames, after the final deltas went through `frames`.
    /// `stopSequence`: the request's `stop` string that ended the reply (Anthropic says so; the other wires' `stop` is it).
    package func close(reason: String, usage: [String: Any], stopSequence: String? = nil) throws -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        switch format {
        case .openAI:
            var frames = [try ServingWire.sse(chunk([:], finish: reason))]
            if includeUsage {
                frames.append(
                    try ServingWire.sse([
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage,
                    ]))
            }
            return frames + [ServingWire.done]
        case .anthropic:
            return try leave() + [
                try event(
                    "message_delta",
                    [
                        "delta": ["stop_reason": stopReason(reason, stopSequence), "stop_sequence": stopSequence ?? NSNull()],
                        "usage": anthropicUsage(usage),
                    ]),
                try event("message_stop", [:]),
            ]
        case .responses:
            let frames = try leave()
            return frames + [
                try event(
                    reason == "length" ? "response.incomplete" : "response.completed",
                    [
                        "response": responseObject(
                            status: reason == "length" ? "incomplete" : "completed", usage: usage, incomplete: reason == "length")
                    ])
            ]
        case .ollama:
            var frames: [Data] = []
            if !ollamaCalls.isEmpty {
                let calls = ollamaCalls.map {
                    ["function": ["name": $0["name"]!, "arguments": Self.object($0["arguments"] as? String ?? "")]]
                }
                frames.append(
                    try line([
                        "model": model, "created_at": timestamp, "message": ["role": "assistant", "content": "", "tool_calls": calls],
                        "done": false,
                    ]))
            }
            return frames + [try line(ollamaFinal(message: ["role": "assistant", "content": ""], reason: reason, usage: usage))]
        }
    }

    /// The complete reply for a request that did not stream.
    package func body(message: [String: Any], reason: String, usage: [String: Any], stopSequence: String? = nil) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        let calls = message["tool_calls"] as? [[String: Any]] ?? []
        func function(_ call: [String: Any]) -> [String: Any] { call["function"] as? [String: Any] ?? [:] }
        switch format {
        case .openAI:
            return [
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [["index": 0, "message": message, "finish_reason": reason]], "usage": usage,
            ]
        case .anthropic:
            var content: [[String: Any]] = []
            if let thought = message["reasoning_content"] as? String {
                content.append(["type": "thinking", "thinking": thought, "signature": ""])
            }
            if let text = message["content"] as? String, !text.isEmpty || calls.isEmpty { content.append(["type": "text", "text": text]) }
            for call in calls {
                content.append([
                    "type": "tool_use", "id": call["id"] ?? "", "name": function(call)["name"] ?? "",
                    "input": Self.object(function(call)["arguments"] as? String ?? ""),
                ])
            }
            return [
                "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
                "stop_reason": stopReason(reason, stopSequence), "stop_sequence": stopSequence ?? NSNull(), "usage": anthropicUsage(usage),
            ]
        case .responses:
            if let thought = message["reasoning_content"] as? String {
                items.append([
                    "id": "rs_\(id.suffix(24))_\(items.count)", "type": "reasoning",
                    "summary": [["type": "summary_text", "text": thought]],
                ])
            }
            if let text = message["content"] as? String, !text.isEmpty || calls.isEmpty {
                items.append([
                    "id": "msg_\(id.suffix(24))_\(items.count)", "type": "message", "status": "completed", "role": "assistant",
                    "content": [["type": "output_text", "text": text, "annotations": []]],
                ])
            }
            for call in calls {
                items.append([
                    "id": "fc_\(id.suffix(24))_\(items.count)", "type": "function_call", "status": "completed", "call_id": call["id"] ?? "",
                    "name": function(call)["name"] ?? "", "arguments": function(call)["arguments"] ?? "{}",
                ])
            }
            return responseObject(status: reason == "length" ? "incomplete" : "completed", usage: usage, incomplete: reason == "length")
        case .ollama:
            var reply: [String: Any] = ["role": "assistant", "content": message["content"] as? String ?? ""]
            if let thought = message["reasoning_content"] as? String { reply["thinking"] = thought }
            if !calls.isEmpty {
                reply["tool_calls"] = calls.map {
                    [
                        "function": [
                            "name": function($0)["name"] ?? "", "arguments": Self.object(function($0)["arguments"] as? String ?? ""),
                        ]
                    ]
                }
            }
            return ollamaFinal(message: reply, reason: reason, usage: usage)
        }
    }

    /// An error as this format reports it: the body of a refused request, or the payload of a stream's error frame.
    package func error(_ failure: ServingFailure) -> [String: Any] {
        switch format {
        case .openAI, .responses: return ["error": failure.wire]
        case .anthropic:
            // Anthropic's types; clients retry `api_error`, so every other client error is `invalid_request_error`.
            let type =
                [
                    401: "authentication_error", 403: "permission_error", 404: "not_found_error", 413: "request_too_large",
                    503: "overloaded_error",
                ][failure.status]
                ?? (failure.status < 500 ? "invalid_request_error" : "api_error")
            return ["type": "error", "error": ["type": type, "message": failure.description]]
        case .ollama: return ["error": failure.description]
        }
    }
    package func errorFrame(_ failure: ServingFailure) throws -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        switch format {
        case .openAI: return [try ServingWire.sse(["error": failure.wire]), ServingWire.done]
        case .responses:
            // A typed event Codex reads (its `code` decides, e.g. compaction on `context_length_exceeded`); no `[DONE]`.
            var failed = responseObject(status: "failed", usage: nil, incomplete: false)
            failed["error"] = ["code": failure.code, "message": failure.description]
            return [try event("response.failed", ["response": failed])]
        case .anthropic: return [try event("error", ["error": (error(failure)["error"] as? [String: Any]) ?? [:]])]
        case .ollama: return [try line(error(failure))]
        }
    }

    private func stopReason(_ reason: String, _ sequence: String? = nil) -> String {
        if reason == "stop", sequence != nil { return "stop_sequence" }
        return reason == "length" ? "max_tokens" : reason == "tool_calls" ? "tool_use" : "end_turn"
    }
    private func anthropicUsage(_ usage: [String: Any]) -> [String: Any] {
        let prompt = usage["prompt_tokens"] as? Int ?? 0
        let cached = (usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int ?? 0
        return [
            "input_tokens": prompt - cached, "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0,
            "output_tokens": usage["completion_tokens"] as? Int ?? 0,
        ]
    }
    private func responseObject(status: String, usage: [String: Any]?, incomplete: Bool) -> [String: Any] {
        var response: [String: Any] = [
            "id": id, "object": "response", "created_at": created, "status": status, "model": model, "output": items,
            "error": NSNull(), "incomplete_details": incomplete ? ["reason": "max_output_tokens"] : NSNull(),
        ]
        response.merge(toolEcho) { $1 }
        if let usage {
            let details = usage["completion_tokens_details"] as? [String: Any] ?? [:]
            response["usage"] = [
                "input_tokens": usage["prompt_tokens"] ?? 0, "output_tokens": usage["completion_tokens"] ?? 0,
                "total_tokens": usage["total_tokens"] ?? 0,
                "input_tokens_details": ["cached_tokens": (usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] ?? 0],
                "output_tokens_details": ["reasoning_tokens": details["reasoning_tokens"] ?? 0],
            ]
        }
        return response
    }
    private func ollamaFinal(message: [String: Any], reason: String, usage: [String: Any]) -> [String: Any] {
        let elapsed = Int64(max(0, Date().timeIntervalSince1970 - TimeInterval(created)) * 1e9)
        return [
            "model": model, "created_at": timestamp, "done": true, "done_reason": reason == "length" ? "length" : "stop",
            "total_duration": elapsed, "prompt_eval_count": usage["prompt_tokens"] ?? 0, "eval_count": usage["completion_tokens"] ?? 0,
        ]
        .merging(ollamaReply(message)) { $1 }
    }
    /// A chat line's `message`, or for `/api/generate` its text as `response` (and `thinking`).
    private func ollamaReply(_ message: [String: Any]) -> [String: Any] {
        guard generate else { return ["message": message] }
        var reply: [String: Any] = ["response": message["content"] as? String ?? ""]
        if let thought = message["thinking"] { reply["thinking"] = thought }
        return reply
    }
    /// Tool arguments as a JSON object (they arrive as text).
    static func object(_ text: String) -> Any {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8))).flatMap { $0 as? [String: Any] } ?? [String: Any]()
    }
}
