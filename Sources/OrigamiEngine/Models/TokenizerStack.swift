import CryptoKit
import Foundation
import Jinja
import Tokenizers

/// A JSON value as the chat template sees it (messages, tool calls and tool schemas). Decoding tries Bool, String,
/// Int, Double, array, object in that order, so integers stay integers (`9007199254740993` is not rounded).
public indirect enum TemplateValue: Codable, Sendable, Equatable {
    case string(String)
    case integer(Int)
    case number(Double)
    case bool(Bool)
    case array([TemplateValue])
    case object([String: TemplateValue])
    case null
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let value = try? c.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? c.decode(String.self) {
            self = .string(value)
        } else if let value = try? c.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? c.decode(Double.self) {
            self = .number(value)
        } else if let value = try? c.decode([TemplateValue].self) {
            self = .array(value)
        } else {
            self = .object(try c.decode([String: TemplateValue].self))
        }
    }
    public func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .integer(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }
    var templateValue: any Sendable {
        switch self {
        case .string(let v): return v
        case .integer(let v): return v
        case .number(let v): return v
        case .bool(let v): return v
        case .array(let v): return v.map(\.templateValue)
        case .object(let v): return v.mapValues(\.templateValue)
        case .null: return Optional<String>.none as any Sendable
        }
    }
}

/// Tokenizer and chat template from the model folder, through the same pinned
/// swift-transformers/swift-jinja libraries as the reference. The identity hashes the
/// three files so a cached prompt never survives a tokenizer or template change.
public final class TokenizerStack: Sendable {
    private let tokenizer: any Tokenizers.Tokenizer
    public let identity: String
    private let template: String
    private let promptEncoding: PromptEncodingCache
    public static let eos: Set<Int> = [248_046, 248_044]
    /// `</think>` and `<think>`, when the vocabulary has them.
    public let thinkEnd: Int?, thinkStart: Int?
    /// `<|im_start|>`, `<|im_end|>` and the `system` role, to find where the system block ends.
    private let imStart: Int?, imEnd: Int?, systemRole: [Int]
    /// `<|image_pad|>`: one per image in a rendered prompt, expanded to the image's tokens.
    public let imagePad: Int?
    /// How the served chat formats open a tool call: Qwen's and GLM's `<tool_call>`, DeepSeek-V4.1's `<｜DSML｜ calls>`
    /// (whose marker token comes first: its closing tags follow the opening one) and earlier DeepSeeks'
    /// `<｜tool▁calls▁begin｜>`. `toolCallOpeners` holds those the vocabulary has as one token (`QualityDial.opensToolCall`).
    public static let toolCallMarkers = ["<tool_call>", "｜DSML｜", "<｜tool▁calls▁begin｜>"]
    /// Single-token control strings of the served formats that typed text must not spell (a zero-width space goes after
    /// the first character): Qwen's and GLM's `<|…|>`, DeepSeek's fullwidth `<｜…｜>` and its DSML marker, the think tags,
    /// the tool-call and tool-result tags a result could close early, GLM's argument tags and its prompt prefix.
    static let typedMarkers = [
        "<|", "<｜", "｜DSML｜", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "[gMASK]", "<sop>",
    ]
    public let toolCallOpeners: Set<Int>
    /// Only markers absent as single vocabulary tokens need incremental text matching.
    package let toolCallTextMarkers: [String]
    /// What an image part becomes in the chat template's text.
    public static let imageMarker = "<|vision_start|><|image_pad|><|vision_end|>"
    /// The decoder only maps tokens to bytes and joins them (byte-level BPE without space clean-up, as Qwen's and the
    /// GGUF families' tokenizers are), so a token sequence's text is its parts' texts joined wherever the left part
    /// ends in a whole character (`ReplyText`).
    public let decodesBytewise: Bool
    /// The decoder maps each token's byte symbols to bytes (`byteSymbols`), and the ids of the added (control) tokens.
    private let byteLevel: Bool, addedIDs: Set<Int>
    private let jsonVocabularies = JSONVocabularyCache()

    public init(modelDirectory: URL) async throws {
        var bound: [String] = []
        var template: Data?
        var bytewise = (decoder: false, cleanUp: true)
        var addedTokens: [String] = []
        var addedIDs = Set<Int>()
        var sectionSafe = false
        var configuration: [String: TemplateValue] = [:]
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            var data = try? Data(contentsOf: modelDirectory.appendingPathComponent(name))
            if data == nil, name == "chat_template.jinja",
                let config = try? Data(contentsOf: modelDirectory.appendingPathComponent("tokenizer_config.json")),
                let text = ((try? JSONSerialization.jsonObject(with: config)) as? [String: Any])?["chat_template"] as? String
            {
                // Older tooling keeps the template only in tokenizer_config.json; hashed as if it were the file.
                data = Data(text.utf8)
            }
            guard let data else { throw EngineError("The model folder lacks \(name)", code: "model_incomplete") }
            bound.append(name + ":" + SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            if name == "chat_template.jinja" { template = data }
            let object = name.hasSuffix(".json") ? (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] : nil
            if name == "tokenizer.json" {
                bytewise.decoder = (object?["decoder"] as? [String: Any])?["type"] as? String == "ByteLevel"
                let added = object?["added_tokens"] as? [[String: Any]] ?? []
                addedTokens = added.compactMap { $0["content"] as? String }.filter { !$0.isEmpty }
                addedIDs = Set(added.compactMap { $0["id"] as? Int })
                sectionSafe =
                    (object?["model"] as? [String: Any])?["type"] as? String == "BPE"
                    && !addedTokens.isEmpty && addedTokens.count == added.count
                    && added.allSatisfy {
                        ($0["id"] as? Int).map { $0 >= 0 } == true
                            && ($0["lstrip"] as? Bool ?? false) == false && ($0["rstrip"] as? Bool ?? false) == false
                    }
                    && PromptEncodingCache.identityPostprocessor(object?["post_processor"])
                try Self.requireCompleteVocabulary(object)
            }
            // swift-transformers cleans up spaces unless the configuration says false.
            if name == "tokenizer_config.json" {
                bytewise.cleanUp = object?["clean_up_tokenization_spaces"] as? Bool ?? true
                configuration = try JSONDecoder().decode([String: TemplateValue].self, from: data)
                // The pinned library's Llama override rewrites the whole input.
                if (object?["tokenizer_class"] as? String)?.hasPrefix("Llama") == true { sectionSafe = false }
            }
        }
        decodesBytewise = bytewise.decoder && !bytewise.cleanUp
        byteLevel = bytewise.decoder
        self.addedIDs = addedIDs
        identity = SHA256.hash(data: Data(bound.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
        guard let text = template.flatMap({ String(data: $0, encoding: .utf8) }) else {
            throw EngineError("The model's chat template is not UTF-8 text", code: "model_incompatible")
        }
        self.template = Self.pythonDictCalls(text)
        promptEncoding = try PromptEncodingCache(
            template: self.template, configuration: configuration, tokens: addedTokens, sectionSafe: sectionSafe)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        self.tokenizer = tokenizer
        thinkEnd = tokenizer.convertTokenToId("</think>")
        thinkStart = tokenizer.convertTokenToId("<think>")
        imStart = tokenizer.convertTokenToId("<|im_start|>")
        imEnd = tokenizer.convertTokenToId("<|im_end|>")
        systemRole = tokenizer.encode(text: "system", addSpecialTokens: false)
        imagePad = tokenizer.convertTokenToId("<|image_pad|>")
        // Round-tripped, so an unknown-token fallback is never taken for a marker.
        toolCallOpeners = Set(
            Self.toolCallMarkers.compactMap { marker in
                tokenizer.convertTokenToId(marker).flatMap { tokenizer.convertIdToToken($0) == marker ? $0 : nil }
            })
        toolCallTextMarkers = Self.toolCallMarkers.filter { marker in
            guard let id = tokenizer.convertTokenToId(marker) else { return true }
            return tokenizer.convertIdToToken(id) != marker
        }
    }

    /// The 256 symbols byte-level BPE writes bytes as (GPT-2's table): printable Latin-1 as itself, the rest from U+0100.
    static let byteSymbols: [String] = {
        var shifted = 0
        return (0..<256).map { b in
            if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) { return String(UnicodeScalar(UInt8(b))) }
            shifted += 1
            return String(UnicodeScalar(UInt32(255 + shifted))!)
        }
    }()

    /// swift-transformers unwraps the id of every piece a prompt splits into, so a BPE vocabulary without an unknown token
    /// must hold every merge's parts and result, and a byte-level one every byte symbol; with a byte-level decoder every
    /// token but the added ones must be made of byte symbols. A model file that breaks this is refused here instead of
    /// trapping on the first prompt or reply that needs the token.
    static func requireCompleteVocabulary(_ tokenizer: [String: Any]?) throws {
        guard let model = tokenizer?["model"] as? [String: Any], model["type"] as? String == "BPE",
            model["unk_token"] == nil || model["unk_token"] is NSNull
        else { return }
        func refuse(_ why: String) -> EngineError { EngineError("The model's tokenizer is incomplete: \(why)", code: "model_incompatible") }
        guard let vocab = model["vocab"] as? [String: Any] else { throw refuse("it has no vocabulary") }
        func byteLevel(_ value: Any?) -> Bool {
            guard let object = value as? [String: Any] else { return false }
            return object["type"] as? String == "ByteLevel" || (object["pretokenizers"] as? [Any] ?? []).contains(where: byteLevel)
        }
        // shortcut: a vocabulary that is not byte-level is not checked for every character it may meet; the served
        // models' are all byte-level.
        if byteLevel(tokenizer?["pre_tokenizer"]), let missing = byteSymbols.firstIndex(where: { vocab[$0] == nil }) {
            throw refuse("byte \(missing) has no token")
        }
        // The byte-level decoder unwraps the byte of every character of a token that is not an added one.
        if (tokenizer?["decoder"] as? [String: Any])?["type"] as? String == "ByteLevel" {
            let added = Set((tokenizer?["added_tokens"] as? [[String: Any]] ?? []).compactMap { $0["content"] as? String })
            let symbols = Set(byteSymbols)
            if let token = vocab.keys.first(where: { !added.contains($0) && !$0.allSatisfy { symbols.contains(String($0)) } }) {
                throw refuse("the token \(token.debugDescription) is not made of byte symbols")
            }
        }
        for merge in model["merges"] as? [Any] ?? [] {
            let pair = (merge as? String).map { $0.split(separator: " ", maxSplits: 1).map(String.init) } ?? (merge as? [String] ?? [])
            guard pair.count == 2, vocab[pair[0]] != nil, vocab[pair[1]] != nil, vocab[pair[0] + pair[1]] != nil else {
                throw refuse("the merge \(merge) is not in its vocabulary")
            }
        }
    }

    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    package func admitPromptEncodingCache(ledger: MemoryLedger) { promptEncoding.admit(ledger: ledger) }
    /// What the server writes to end a think block early: Qwen's own early-exit sentence, then `</think>`.
    public var thinkClosing: [Int] {
        guard let thinkEnd else { return [] }
        return encode("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n")
            + [thinkEnd] + encode("\n\n")
    }
    /// Each token's bytes for JSON mode (`JSONVocabulary`), built on first use for a vocabulary size and kept.
    package func jsonVocabulary(size: Int) -> JSONVocabulary {
        jsonVocabularies.get(size) {
            let symbols = Dictionary(uniqueKeysWithValues: Self.byteSymbols.enumerated().map { ($1.unicodeScalars.first!, UInt8($0)) })
            return JSONVocabulary(size: size) { id in
                guard let piece = tokenizer.convertIdToToken(id) else { return nil }
                if addedIDs.contains(id) { return (Array(piece.utf8), true) }
                guard byteLevel else {
                    // shortcut: a decoder that is not byte-level loses a token's partial UTF-8; no served tokenizer is one.
                    return (Array(tokenizer.decode(tokens: [id], skipSpecialTokens: false).utf8), false)
                }
                var bytes: [UInt8] = []
                for scalar in piece.unicodeScalars {
                    guard let byte = symbols[scalar] else { return nil }
                    bytes.append(byte)
                }
                return (bytes, false)
            }
        }
    }
    public func decode(_ tokens: [Int]) throws -> String {
        guard tokens.allSatisfy({ (0..<248_320).contains($0) }) else { throw EngineError("Invalid token ID") }
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
    }

    public static func normalizeMessages(_ messages: [[String: TemplateValue]]) throws -> [[String: TemplateValue]] {
        var images: [String] = []
        return try normalizeMessages(messages, images: &images)
    }
    /// Image parts of user messages become `imageMarker`; `images` receives their URLs in prompt order.
    public static func normalizeMessages(_ messages: [[String: TemplateValue]], images: inout [String]) throws -> [[String: TemplateValue]]
    {
        guard !messages.isEmpty else { throw EngineError("Chat requires at least one message") }
        var normalized: [[String: TemplateValue]] = []
        var systems: [String] = []
        var pending: [String: String] = [:]
        var order: [String] = []
        var ids = Set<String>()
        var results: [String: [String: TemplateValue]] = [:]
        for var message in messages {
            guard case .string(let role) = message["role"], ["system", "developer", "user", "assistant", "tool"].contains(role) else {
                throw EngineError("Unsupported or absent message role")
            }
            if role != "tool" && !pending.isEmpty { throw EngineError("Outstanding tool calls need results before the next message") }
            if let attached = message["images"], attached != .null {
                guard case .array(let list) = attached, list.isEmpty else { throw EngineError("Send images as image_url content parts") }
            }
            // Chat markers typed into a message are text, not control tokens: `<|im_end|>` in a user message must
            // not end the turn. A zero-width space keeps them from matching; the assistant's own history keeps them.
            func typed(_ text: String) -> String { role == "assistant" ? text : Self.typedText(text) }
            switch message["content"] {
            case .string(let text): message["content"] = .string(typed(text))
            case .null, .none:
                guard role == "assistant" else { throw EngineError("Message content is required") }
                message["content"] = .string("")
            case .array(let parts):
                // Text between images is joined before neutralizing, so split parts cannot spell a marker.
                var content = ""
                var run = ""
                for part in parts {
                    guard case .object(let value) = part else { throw EngineError("Unsupported content part") }
                    if value["type"] == .string("image_url") {
                        guard role == "user" else { throw EngineError("Images belong in user messages") }
                        guard let url = PromptImage.url(of: value) else { throw EngineError("Image part without a URL") }
                        images.append(url)
                        content += typed(run) + Self.imageMarker
                        run = ""
                        continue
                    }
                    guard case .string(let piece) = value["text"] else { throw EngineError("Unsupported content part") }
                    if let kind = value["type"] {
                        guard case .string(let label) = kind, ["text", "input_text"].contains(label) else {
                            throw EngineError("Unsupported content part")
                        }
                    }
                    run += piece
                }
                message["content"] = .string(content + typed(run))
            default: throw EngineError("Invalid chat content")
            }
            if role == "system" || role == "developer" {
                guard case .string(let text) = message["content"] else { throw EngineError("System/developer messages must be text") }
                // The template takes system text only first. A later one (OpenAI allows them anywhere; Claude Code sends its
                // environment that way) is a user turn where it stands, so earlier tokens and their saved state still match.
                guard normalized.isEmpty else {
                    message["role"] = .string("user")
                    normalized.append(message)
                    continue
                }
                systems.append(text)
                continue
            }
            if role == "assistant" {
                // Some clients send a reply's thinking back under both names with the same text: that is one field.
                if let reasoning = message["reasoning"], let content = message["reasoning_content"], reasoning != content {
                    throw EngineError("The assistant message's reasoning and reasoning_content differ: send one")
                }
                if let reasoning = message.removeValue(forKey: "reasoning") { message["reasoning_content"] = reasoning }
                // Status lines Origami streamed as reasoning are not the model's; a client that sends them back gets them dropped.
                if case .string(let text) = message["reasoning_content"], text.contains(ChatCommand.marker) {
                    message["reasoning_content"] = .string(
                        text.split(separator: "\n", omittingEmptySubsequences: false)
                            .filter { !$0.hasPrefix(ChatCommand.marker) }.joined(separator: "\n"))
                }
                if let callsValue = message["tool_calls"] {
                    guard case .array(let calls) = callsValue else { throw EngineError("Tool calls must be an array") }
                    message["tool_calls"] = .array(
                        try calls.map { call in
                            guard case .object(let object) = call, case .string("function") = object["type"],
                                case .string(let id) = object["id"], !id.isEmpty, ids.insert(id).inserted,
                                case .object(let function) = object["function"], case .string(let name) = function["name"], !name.isEmpty
                            else {
                                throw EngineError("Tool calls need unique IDs and function names")
                            }
                            // A JSON string is kept as sent (checked here): `encodeNormalized` renders its keys in its order.
                            let arguments = try function["arguments"].map { value -> TemplateValue in
                                guard case .string(let serialized) = value else { return value }
                                return try JSONDecoder().decode(TemplateValue.self, from: Data(serialized.utf8))
                            }
                            guard case .object? = arguments else { throw EngineError("Tool arguments must be an object") }
                            pending[id] = name
                            order.append(id)
                            return .object(object)
                        })
                }
            } else if role == "tool" {
                guard case .string(let id) = message["tool_call_id"], let name = pending.removeValue(forKey: id) else {
                    throw EngineError("Tool result must answer an outstanding call exactly once")
                }
                if let supplied = message["name"] {
                    guard case .string(name) = supplied else { throw EngineError("Tool result name does not match its call") }
                }
                message["name"] = .string(name)
                results[id] = message
                if pending.isEmpty {
                    normalized.append(contentsOf: order.map { results[$0]! })
                    order.removeAll()
                    results.removeAll()
                }
                continue
            }
            normalized.append(message)
        }
        guard pending.isEmpty else { throw EngineError("Messages are missing tool results") }
        if !systems.isEmpty {
            normalized.insert(["role": .string("system"), "content": .string(systems.joined(separator: "\n\n"))], at: 0)
        }
        return normalized
    }

    /// `text` with every typed marker broken by a zero-width space after its first character.
    static func typedText(_ text: String) -> String {
        typedMarkers.reduce(text) { text, marker in
            text.contains(marker)
                ? text.replacingOccurrences(of: marker, with: String(marker.first!) + "\u{200B}" + marker.dropFirst()) : text
        }
    }

    /// A tool declaration as the template shows it: its names, descriptions and schema are the client's text like a system
    /// message's, so markers in them stay text too.
    package static func typedTool(_ value: TemplateValue) -> TemplateValue {
        switch value {
        case .string(let text): return .string(typedText(text))
        case .array(let items): return .array(items.map(typedTool))
        case .object(let fields): return .object(Dictionary(fields.map { (typedText($0.key), typedTool($0.value)) }) { first, _ in first })
        default: return value
        }
    }

    /// What a thinking level becomes in a template that refuses its name: its number in Unsloth's DeepSeek-V4.1 map, then
    /// the nearest names.
    static let effortSubstitutes: [String: [any Sendable]] = [
        "low": [50, "medium", "high"], "medium": [62, "high", "low"], "high": [75, "max", "medium"],
        "xhigh": [100, "max", "high"], "max": [100, "xhigh", "high"],
    ]

    public func encodeChat(messages: [[String: TemplateValue]], tools: [[String: TemplateValue]] = [], reasoningEffort: String = "medium")
        throws -> [Int]
    {
        try encodeNormalized(Self.normalizeMessages(messages), tools: tools, reasoningEffort: reasoningEffort)
    }
    /// `normalized` comes from `normalizeMessages`, once: normalizing again would turn image markers into text.
    public func encodeNormalized(
        _ normalized: [[String: TemplateValue]], tools: [[String: TemplateValue]] = [], reasoningEffort: String = "medium"
    ) throws -> [Int] {
        guard ["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(reasoningEffort) else {
            throw EngineError("Unsupported reasoning effort")
        }
        let thinking = !["none", "minimal"].contains(reasoningEffort)
        // `thinking` is DeepSeek's templates' switch; Qwen's and GLM's read `enable_thinking` (or none).
        // GLM's `clear_thinking` and DeepSeek's `drop_thinking` off: past reasoning stays rendered as it was generated, so a
        // user turn continues the conversation instead of re-reading the earlier replies without it.
        var context: [String: any Sendable] = [
            "enable_thinking": thinking, "thinking": thinking, "clear_thinking": false, "drop_thinking": false,
            "_python_dict": Self.pythonDict,
            "_bounded_range": Self.boundedRange,
        ]
        // Qwen's template names its top effort `xhigh` (`high` and `max` are it); a template without it takes `high` and
        // `max` as asked and `xhigh` as `max` (DeepSeek's refuses the name; GLM's reads any unknown level as Max).
        if thinking {
            let xhigh = template.contains("xhigh")
            context["reasoning_effort"] =
                ["high", "max"].contains(reasoningEffort) && xhigh
                ? "xhigh" : reasoningEffort == "xhigh" && !xhigh ? "max" : reasoningEffort
        }
        let tools = tools.map { $0.mapValues(Self.typedTool) }
        var ids: [Int]
        do {
            ids = try promptEncoding.encode(normalized, tools: tools, context: context, tokenizer: tokenizer)
        } catch  where thinking {
            // A template that refuses the level's name (ISTA-DASLab's DeepSeek one takes low, high, max or 1–100) gets the
            // level on DeepSeek's 1–100 scale, then the nearest names; one that takes none of them keeps its own error.
            var found: [Int]?
            for substitute in Self.effortSubstitutes[reasoningEffort] ?? [] where found == nil {
                context["reasoning_effort"] = substitute
                found = try? promptEncoding.encode(normalized, tools: tools, context: context, tokenizer: tokenizer)
            }
            guard let found else { throw error }
            ids = found
        }
        // A template without a thinking switch (GLM-5.3's) always opens `<think>`: thinking off closes it at once.
        if !thinking, let thinkStart, let thinkEnd, ids.last == thinkStart { ids.append(thinkEnd) }
        return ids
    }

    /// A normalized message for the template. A tool call's arguments sent as a JSON string become an object whose keys
    /// keep the string's order (a Swift dictionary's are sorted on the way in), so the call renders as the model wrote it
    /// and the next prompt continues the conversation.
    static func templateMessage(_ message: [String: TemplateValue]) throws -> [String: any Sendable] {
        var values = message.mapValues(\.templateValue)
        guard case .array(let calls)? = message["tool_calls"] else { return values }
        values["tool_calls"] = try calls.map { call -> any Sendable in
            guard case .object(let object) = call, case .object(let function)? = object["function"],
                case .string(let serialized)? = function["arguments"]
            else { return call.templateValue }
            var rendered = function.mapValues(\.templateValue)
            var outer = object.mapValues(\.templateValue)
            rendered["arguments"] = try orderedJSON(serialized)
            outer["function"] = rendered
            return outer
        }
        return values
    }
    /// Valid JSON as a template value whose objects keep their keys in the text's order. Every string, number and literal
    /// is read as `TemplateValue` reads it, so values are typed as before; only the order is new.
    static func orderedJSON(_ text: String) throws -> Jinja.Value {
        let bytes = Array(text.utf8)
        var at = 0
        func peek() -> UInt8 {
            while at < bytes.count, [0x20, 0x09, 0x0a, 0x0d].contains(bytes[at]) { at += 1 }
            return at < bytes.count ? bytes[at] : 0
        }
        func scalar() throws -> TemplateValue {
            guard at < bytes.count else { throw EngineError("Tool arguments are not JSON") }
            let start = at
            if bytes[at] == UInt8(ascii: "\"") {
                at += 1
                while at < bytes.count, bytes[at] != UInt8(ascii: "\"") { at += bytes[at] == UInt8(ascii: "\\") ? 2 : 1 }
                at += 1
            } else {
                while at < bytes.count, !Array(",:]} \t\n\r".utf8).contains(bytes[at]) { at += 1 }
            }
            return try JSONDecoder().decode(TemplateValue.self, from: Data(bytes[start..<min(at, bytes.count)]))
        }
        func value() throws -> Jinja.Value {
            let open = peek()
            guard open == UInt8(ascii: "{") || open == UInt8(ascii: "[") else { return try Jinja.Value(any: scalar().templateValue) }
            at += 1
            let close = open == UInt8(ascii: "{") ? UInt8(ascii: "}") : UInt8(ascii: "]")
            var object: OrderedDictionary<ObjectKey, Jinja.Value> = [:]
            var array: [Jinja.Value] = []
            if peek() != close {
                while true {
                    if open == UInt8(ascii: "{") {
                        _ = peek()
                        guard case .string(let key) = try scalar(), peek() == UInt8(ascii: ":") else {
                            throw EngineError("Tool arguments are not JSON")
                        }
                        at += 1
                        // A repeated key keeps its first value, as JSONDecoder (and so `TemplateValue`, before) reads it.
                        let item = try value()
                        if object[.string(key)] == nil { object[.string(key)] = item }
                    } else {
                        array.append(try value())
                    }
                    guard peek() == UInt8(ascii: ",") else { break }
                    at += 1
                }
            }
            guard peek() == close else { throw EngineError("Tool arguments are not JSON") }
            at += 1
            return open == UInt8(ascii: "{") ? .object(object) : .array(array)
        }
        let result = try value()
        guard peek() == 0, at == bytes.count else { throw EngineError("Tool arguments are not JSON") }
        return result
    }

    /// Jinja's `dict()` as Python has it: a mapping's (or a list of pairs') entries, then the keyword arguments.
    /// swift-jinja's built-in keeps only the keywords, so DeepSeek-V4.1's `dict(msg, content_blocks=…)` lost every
    /// message's role. Without a positional argument it is the built-in. swift-jinja's globals cannot be replaced from
    /// the context, so a template's `dict(` calls become `_python_dict(` when it is loaded (`pythonDictCalls`).
    static func pythonDictCalls(_ template: String) -> String {
        template.replacingOccurrences(of: #"(?<![A-Za-z0-9_.])dict\("#, with: "_python_dict(", options: .regularExpression)
            .replacingOccurrences(of: #"(?<![A-Za-z0-9_.])range\("#, with: "_bounded_range(", options: .regularExpression)
    }
    /// Jinja's `range()` with at most 65,536 values: the chat template comes from the model file, and the built-in made
    /// `range(9999999)` a gigabyte list on every prompt (and trapped on a step past Int.max). Templates loop over
    /// messages and tools, far fewer.
    static let boundedRange = Jinja.Value.function { arguments, _, _ in
        let values = arguments.map { value -> Int? in
            if case .int(let number) = value { return number }
            return nil
        }
        guard (1...3).contains(values.count), values.allSatisfy({ $0 != nil }) else {
            throw EngineError("range() takes one to three integers")
        }
        let (start, stop) = values.count == 1 ? (0, values[0]!) : (values[0]!, values[1]!)
        let step = values.count == 3 ? values[2]! : 1
        guard step != 0 else { throw EngineError("range() step must not be zero") }
        let (span, overflow) = stop.subtractingReportingOverflow(start)
        guard !overflow, span / step <= 1 << 16 else { throw EngineError("range() of more than 65,536 values") }
        let count = span.signum() == step.signum() ? (span - step.signum()) / step + 1 : 0
        return .array((0..<count).map { .int(start + $0 * step) })
    }
    static let pythonDict = Jinja.Value.function { arguments, keywords, _ in
        guard arguments.count <= 1 else { throw EngineError("dict() takes at most one positional argument") }
        var merged: OrderedDictionary<ObjectKey, Jinja.Value> = [:]
        switch arguments.first {
        case nil: break
        case .object(let entries)?: merged = entries
        case .array(let pairs)?:
            for pair in pairs {
                guard case .array(let kv) = pair, kv.count == 2, let key = ObjectKey(kv[0]) else {
                    throw EngineError("dict() takes a mapping or a list of key-value pairs")
                }
                merged[key] = kv[1]
            }
        default: throw EngineError("dict() takes a mapping or a list of key-value pairs")
        }
        for (key, value) in keywords.sorted(by: { $0.key < $1.key }) { merged[.string(key)] = value }
        return .object(merged)
    }

    /// Where this turn's reply begins (the last `<|im_start|>`, the generation prompt's): the next turn
    /// of the chat re-renders everything before it.
    public func turnStart(_ ids: [Int]) -> Int? { imStart.flatMap { ids.lastIndex(of: $0) } }

    /// Startup probe of the chat template: does thinking off change the prompt, and does a follow-up
    /// turn (thinking off) extend the previous prompt plus its reply?
    public func templateCheck() -> (thinkingSwitch: Bool, followUpExtends: Bool)? {
        let user: [[String: TemplateValue]] = [["role": .string("user"), "content": .string("Hi")]]
        guard let off = try? encodeChat(messages: user, reasoningEffort: "none"),
            let on = try? encodeChat(messages: user, reasoningEffort: "medium")
        else { return nil }
        let turn =
            user + [["role": .string("assistant"), "content": .string("Hello.")], ["role": .string("user"), "content": .string("Again")]]
        let next = (try? encodeChat(messages: turn, reasoningEffort: "none")) ?? []
        return (off != on, next.starts(with: off + encode("Hello.")))
    }

    /// Tokens up to and including the `<|im_end|>` of a leading system block; nil without one.
    public func systemBlockEnd(_ ids: [Int]) -> Int? {
        guard let imStart, let imEnd else { return nil }
        return Self.systemBlockEnd(ids, start: imStart, end: imEnd, role: systemRole)
    }
    public static func systemBlockEnd(_ ids: [Int], start: Int, end: Int, role: [Int]) -> Int? {
        guard ids.first == start, !role.isEmpty, ids.count > role.count, Array(ids[1...role.count]) == role,
            let close = ids.firstIndex(of: end)
        else { return nil }
        return close + 1
    }

    /// Re-render first, then compare token IDs. BPE boundaries make character
    /// prefix lengths unsafe after edits or a changed conversation template.
    public static func commonTokenPrefix(_ left: [Int], _ right: [Int]) -> Int {
        var length = 0
        while length < min(left.count, right.count), left[length] == right[length] { length += 1 }
        return length
    }
}

/// A reply's text as its tokens arrive: `text(ids)` is `tokenizer.decode(ids)` for every longer prefix of one reply
/// (BITWISE). With a byte-level decoder (`decodesBytewise`) it decodes only the tokens since the text last ended in a
/// whole character and joins that to the text before, which is exact: those bytes start a fresh UTF-8 sequence and the
/// decoder adds nothing between tokens. So a long reply costs each token a few tokens' decode instead of the whole
/// reply's. Other decoders decode the whole reply each time.
public struct ReplyText {
    private let tokenizer: TokenizerStack, bytewise: Bool
    private var stable = "", from = 0
    public init(_ tokenizer: TokenizerStack) { self.init(tokenizer, bytewise: tokenizer.decodesBytewise) }
    package init(_ tokenizer: TokenizerStack, bytewise: Bool) {
        self.tokenizer = tokenizer
        self.bytewise = bytewise
    }
    public mutating func text(_ ids: [Int]) throws -> String {
        guard bytewise, from <= ids.count else { return try tokenizer.decode(ids) }
        let tail = try tokenizer.decode(Array(ids[from...]))
        let text = stable + tail
        if let last = tail.unicodeScalars.last, last != "\u{FFFD}" {
            stable = text
            from = ids.count
        }
        return text
    }
}

/// The JSON vocabularies a tokenizer has built, by vocabulary size.
private final class JSONVocabularyCache: @unchecked Sendable {
    private let lock = NSLock()
    private var built: [Int: JSONVocabulary] = [:]
    func get(_ size: Int, build: () -> JSONVocabulary) -> JSONVocabulary {
        lock.lock()
        defer { lock.unlock() }
        if let vocabulary = built[size] { return vocabulary }
        let vocabulary = build()
        built[size] = vocabulary
        return vocabulary
    }
}
