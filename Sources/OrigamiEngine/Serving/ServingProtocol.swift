/*
MIT License

Copyright (c) 2026 Carlos Galarza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// ToolCallSplitter and ThinkSplitter reused from pinned SlotStream source
// a8a5294a8a826ee9356c900df76614646e756bba. No MLX dependency.
// JSON bridging retains the same values with explicit Sendable existentials.

// Native tool-call parsing, in the model's own grammar.
//
// Qwen3.8-Flash-Next does not emit JSON tool calls. Its chat template teaches
// it an XML-ish form, and that is what the weights actually produce:
//
//     <tool_call>
//     <function=read_file>
//     <parameter=path>
//     hello.txt
//     </parameter>
//     </function>
//     </tool_call>
//
// Prose may precede the first call and follow the last one; several calls may
// appear in one turn. Nothing else is in the grammar.
//
// Why a splitter rather than a regex over the finished text. The API streams,
// so the parser is fed token deltas and must never emit half of a tag as text:
// a client that sees "<tool" in its transcript has already been given a wrong
// answer, and no later correction removes it. `ThinkSplitter` solves the same
// problem for `</think>` by withholding the tail of its buffer; this holds back
// the longest suffix that could still become an opening tag, which is a shorter
// hold than a fixed one and lets ordinary prose stream immediately.
//
// Why coercion needs the schema. The grammar is untyped: every parameter
// arrives as raw text between two tags. The client asked for typed JSON, so
// `{"start_line": 1}` and `{"start_line": "1"}` are different answers and only
// one of them satisfies the tool. Agent harnesses' schemas routinely use `enum`,
// `anyOf`, or nested objects, so a string-only passthrough is not an option.
// The declared type decides; anything that fails to parse falls back to the
// raw string rather than dropping the parameter, because a tool reporting a
// bad argument is a better failure than a call that silently lost one.

import CoreFoundation
import CryptoKit
import Foundation

/// A JSON value, restricted to what a tool argument can be. Deliberately a
/// concrete enum and not `Any`: the parser's output is compared in checks, and
/// `Any` is neither `Equatable` nor `Sendable`.
public enum JSONValue: Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    /// The Foundation value, for `JSONSerialization` and for the chat template.
    ///
    /// A JSON null crosses as an EMPTY OPTIONAL, never as `NSNull`. This is not
    /// a style choice: swift-jinja's `Value(any:)` maps Swift `nil` to Jinja
    /// null and *throws* on `NSNull` — "Cannot convert value of type NSNull to
    /// Jinja Value" — so an `NSNull` anywhere in a tool schema, a replayed tool
    /// call's arguments, or a tool result fails the whole turn with a 400. Agents
    /// send `"default": null` inside tool schemas and nulls for unset optional
    /// arguments, so this is the common case, not an exotic one. A boxed empty
    /// Optional reaches the bridge's Mirror path and degrades to null, which
    /// renders as `null` and keeps array positions intact.
    public var any: any Sendable {
        switch self {
        case .string(let s): return s
        case .int(let i): return i
        case .double(let d): return d
        case .bool(let b): return b
        case .array(let a): return a.map { $0.any }
        case .object(let o): return o.mapValues { $0.any }
        case .null: return Self.templateNull
        }
    }

    /// An empty `Optional` boxed as `Any`. See `any`.
    public static let templateNull: any Sendable = Optional<String>.none as any Sendable

    /// Compact JSON for one value, used to build the streamed input deltas.
    /// Key order inside an object is sorted so a delta stream is reproducible
    /// across processes; Swift dictionaries have no stable order of their own.
    public var jsonText: String {
        switch self {
        case .string(let s): return Self.quote(s)
        case .int(let i): return String(i)
        case .double(let d):
            // A whole double must not print as "1.0" where the schema said
            // number and the tool expects 1; JSON has one numeric type.
            if d == d.rounded(), abs(d) < 1e15 { return String(Int(d)) }
            return String(d)
        case .bool(let b): return b ? "true" : "false"
        case .array(let a): return "[" + a.map { $0.jsonText }.joined(separator: ",") + "]"
        case .object(let o):
            let body = o.keys.sorted().map { Self.quote($0) + ":" + o[$0]!.jsonText }
            return "{" + body.joined(separator: ",") + "}"
        case .null: return "null"
        }
    }

    public static func quote(_ s: String) -> String {
        var out = "\""
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if ch.value < 0x20 {
                    out += String(format: "\\u%04x", ch.value)
                } else {
                    out.unicodeScalars.append(ch)
                }
            }
        }
        return out + "\""
    }

    /// Parse text as JSON, or return nil. `fragmentsAllowed` so a bare number
    /// or string parses too.
    static func parse(_ text: String) -> JSONValue? {
        guard let data = text.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        return from(obj)
    }

    static func from(_ obj: Any) -> JSONValue {
        switch obj {
        case let n as NSNumber:
            // NSNumber erases Bool into 0/1; the objCType tells them apart.
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return .bool(n.boolValue) }
            if let i = Int(exactly: n) { return .int(i) }
            return .double(n.doubleValue)
        case let s as String: return .string(s)
        case let a as [Any]: return .array(a.map { from($0) })
        case let o as [String: Any]: return .object(o.mapValues { from($0) })
        default: return .null
        }
    }
}

/// The declared type of one tool parameter. `unknown` covers unresolved unions
/// and a parameter the schema does not mention at all.
public enum ToolParamKind: String, Sendable, Equatable {
    case string, integer, number, boolean, array, object, unknown
}

/// One tool as the parser needs it: a name and its parameters' declared types.
public struct ToolSchema: Sendable, Equatable {
    public let name: String
    public let params: [String: ToolParamKind]

    public init(name: String, params: [String: ToolParamKind]) {
        self.name = name
        self.params = params
    }
}

/// One parsed call.
public struct ParsedToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: [String: JSONValue]
    /// Parameter names in the order the model emitted them.
    ///
    /// Not decoration: the input is streamed as `tool-input-delta` fragments as
    /// each parameter closes, and the client concatenates them. If the final
    /// `input` were key-sorted instead, the accumulated deltas and the final
    /// object would disagree whenever the model emitted out of alphabetical
    /// order, which is most calls with more than one parameter. Emission order
    /// is also stable run to run, where Swift's dictionary order is not.
    public let order: [String]

    public init(id: String, name: String, arguments: [String: JSONValue], order: [String] = []) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.order = order.isEmpty ? arguments.keys.sorted() : order
    }

    /// The compact JSON object the wire carries as `input`.
    public var inputJSON: String {
        let body = order.compactMap { k in
            arguments[k].map { JSONValue.quote(k) + ":" + $0.jsonText }
        }
        return "{" + body.joined(separator: ",") + "}"
    }
}

/// What the parser emits, in order.
public enum ToolStreamEvent: Sendable, Equatable {
    case text(String)
    case toolInputStart(id: String, name: String)
    case toolInputDelta(id: String, delta: String)
    case toolInputEnd(id: String)
    case toolCall(ParsedToolCall)
    /// A `<tool_call>` block that never closed before the sequence ended. The
    /// text is returned verbatim so nothing the model produced is lost, and the
    /// turn finishes as ordinary text.
    case malformed(String)
}

/// The tool-call syntax a family's chat template teaches its model. The three
/// differ only in their tags, in where a call's name and a parameter's name
/// sit, and in how a value says its type, so one state machine reads them all:
///
///     Qwen      <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
///     GLM-5.3   <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
///     DeepSeek  <｜DSML｜ calls>\n<｜DSML｜ invoke name="NAME">\n<｜DSML｜ parameter name="K" string="true">V</｜DSML｜ parameter>
///               \n</｜DSML｜ invoke>\n…more invokes…\n</｜DSML｜ calls>
///
/// GLM's template writes a string as is and anything else as JSON, so the
/// declared type decides, as for Qwen; DeepSeek's says which with `string=`,
/// which decides only for a parameter declared without a type.
public struct ToolCallGrammar: Sendable, Equatable {
    let family: ModelFamily
    let open, close: String
    /// Nil: the name follows `open` directly and ends at the first parameter or `close` (GLM).
    let callOpen: String?
    let callNameEnd: String
    /// Nil: the block's close ends the call (GLM).
    let callClose: String?
    let paramOpen, paramNameEnd, paramClose: String
    /// Several calls in one block (DeepSeek's invokes), each emitted at its own close.
    let manyPerBlock: Bool
    /// One newline after a parameter's open tag and one before its close are the template's framing (Qwen).
    let framedValues: Bool
    /// What the template writes between a message's content and its block (DeepSeek's `\n\n`, after content it does not
    /// trim): left out of the content sent to the client, so the next prompt renders it once, as the model wrote it.
    var separator = ""

    public static let qwen = ToolCallGrammar(
        family: .qwen, open: "<tool_call>", close: "</tool_call>", callOpen: "<function=",
        callNameEnd: ">", callClose: "</function>", paramOpen: "<parameter=", paramNameEnd: ">", paramClose: "</parameter>",
        manyPerBlock: false, framedValues: true)
    public static let glm = ToolCallGrammar(
        family: .glm, open: "<tool_call>", close: "</tool_call>", callOpen: nil,
        callNameEnd: "", callClose: nil, paramOpen: "<arg_key>", paramNameEnd: "<arg_value>", paramClose: "</arg_value>",
        manyPerBlock: false, framedValues: false)
    public static let deepseek = ToolCallGrammar(
        family: .deepseek, open: "<｜DSML｜ calls>", close: "</｜DSML｜ calls>",
        callOpen: "<｜DSML｜ invoke name=\"", callNameEnd: "\">", callClose: "</｜DSML｜ invoke>",
        paramOpen: "<｜DSML｜ parameter name=\"", paramNameEnd: "\">", paramClose: "</｜DSML｜ parameter>",
        manyPerBlock: true, framedValues: false, separator: "\n\n")

    public static func of(_ family: ModelFamily) -> ToolCallGrammar {
        switch family {
        case .qwen: return .qwen
        case .glm: return .glm
        case .deepseek: return .deepseek
        }
    }

    /// A parameter's name, and whether its value is a string (`true`), JSON (`false`) or as declared (nil), from the
    /// text between `paramOpen` and `paramNameEnd`; nil when that text is not a parameter header.
    func parameter(_ header: Substring) -> (name: String, literal: Bool?)? {
        switch family {
        case .qwen: return (String(header), nil)
        case .glm:
            // `K</arg_key>`, and any whitespace the model put before `<arg_value>`.
            guard let end = header.range(of: "</arg_key>"), header[end.upperBound...].allSatisfy(\.isWhitespace) else { return nil }
            return (header[..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines), nil)
        case .deepseek:
            // `K" string="true`: the name, then its attributes.
            let quote = header.firstIndex(of: "\"") ?? header.endIndex
            let attributes = header[quote...]
            return (
                String(header[..<quote]), attributes.contains("string=\"true") ? true : attributes.contains("string=\"false") ? false : nil
            )
        }
    }
}

/// Splits a generated stream into text and tool calls.
///
/// Feed it the deltas that survive `ThinkSplitter` (reasoning is never scanned
/// for calls) and it returns the events for each. `flush()` closes the stream.
/// The tags are the family's (`ToolCallGrammar`); the comments name Qwen's.
public final class ToolCallSplitter {
    private enum State {
        case text
        case call  // inside <tool_call>, before <function=…> (DeepSeek: also between invokes)
        case params  // inside <function=…>, between parameters
        case value(String)  // inside <parameter=NAME>
        case closing  // after </function>, before </tool_call>
    }

    private let grammar: ToolCallGrammar
    private let schemas: [String: ToolSchema]
    private let nextID: @Sendable () -> String

    private var buf = ""
    private var state: State = .text
    private var currentID = ""
    private var currentName = ""
    private var currentArgs: [String: JSONValue] = [:]
    private var currentOrder: [String] = []
    private var emittedAnyArg = false
    private var valueRaw = ""
    private var streamsString = false
    private var valueLiteral: Bool?
    private var valueAtStart = true
    /// Calls finished inside the open block (DeepSeek's several invokes).
    private var blockCalls = 0
    /// Everything consumed since `<tool_call>`, so a block that never closes
    /// can be returned verbatim rather than vanishing.
    private var rawCall = ""

    public init(tools: [ToolSchema], grammar: ToolCallGrammar = .qwen, idFactory: (@Sendable () -> String)? = nil) {
        var byName: [String: ToolSchema] = [:]
        for t in tools { byName[t.name] = t }
        self.schemas = byName
        self.grammar = grammar
        self.nextID = idFactory ?? { ToolCallSplitter.randomID() }
    }

    /// `call_` plus 24 hex characters (96 random bits). A request refuses a
    /// repeated ID anywhere in its history, so one collision would break every
    /// later turn of that conversation. The chat template never renders IDs.
    public static func randomID() -> String {
        "call_" + (0..<3).map { _ in String(format: "%08x", UInt32.random(in: 0...UInt32.max)) }.joined()
    }

    public func push(_ s: String) -> [ToolStreamEvent] {
        buf += s
        var out: [ToolStreamEvent] = []
        loop: while true {
            switch state {
            case .text:
                if let r = buf.range(of: grammar.open) {
                    var before = String(buf[..<r.lowerBound])
                    var separator = ""
                    // Back in the block's text if it is not a call after all.
                    if before.hasSuffix(grammar.separator) {
                        before.removeLast(grammar.separator.count)
                        separator = grammar.separator
                    }
                    if !before.isEmpty { out.append(.text(before)) }
                    rawCall = separator + grammar.open
                    buf = String(buf[r.upperBound...])
                    blockCalls = 0
                    state = .call
                    continue loop
                }
                // No opening tag yet: emit everything that cannot become one (or its separator and one).
                let keep = max(Self.holdback(buf, grammar.open), Self.holdback(buf, grammar.separator + grammar.open))
                let emit = String(buf.dropLast(keep))
                if !emit.isEmpty { out.append(.text(emit)) }
                buf = String(buf.suffix(keep))
                break loop

            case .call:
                // Between <tool_call> and <function=NAME>. Only whitespace is
                // expected here, and an empty call closes straight away.
                //
                // Both tags must be located before either is acted on. Taking
                // whichever one happened to be in the buffer swallowed a whole
                // call when the complete text arrived in one push: `</tool_call>`
                // was already present, so the empty-call path consumed the
                // function and its parameters as if nothing had been called.
                guard let callOpen = grammar.callOpen else {
                    // GLM: the name runs from <tool_call> to the first <arg_key> or </tool_call>, neither consumed here.
                    let pOpen = buf.range(of: grammar.paramOpen)
                    let cTag = buf.range(of: grammar.close)
                    guard let end = [pOpen, cTag].compactMap({ $0 }).min(by: { $0.lowerBound < $1.lowerBound }) else { break loop }
                    let name = buf[..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                    rawCall += String(buf[..<end.lowerBound])
                    buf = String(buf[end.lowerBound...])
                    if end == cTag && name.isEmpty {
                        // An empty block closes without a call, as in the function form.
                        rawCall += grammar.close
                        buf = String(buf.dropFirst(grammar.close.count))
                        resetCall()
                        state = .text
                    } else {
                        startCall(name, into: &out)
                    }
                    continue loop
                }
                let fOpen = buf.range(of: callOpen)
                let cTag = buf.range(of: grammar.close)
                if let ct = cTag, fOpen == nil || ct.lowerBound < fOpen!.lowerBound {
                    // A block holding anything but a <function=…> (a JSON-form call, say) is not in the
                    // grammar: it stays the model's text instead of vanishing.
                    let inside = buf[..<ct.lowerBound]
                    rawCall += String(buf[..<ct.upperBound])
                    if !inside.allSatisfy(\.isWhitespace) { out.append(.text(rawCall)) }
                    buf = String(buf[ct.upperBound...])
                    resetCall()
                    state = .text
                    continue loop
                }
                guard let r = fOpen else { break loop }
                guard let gt = buf.range(of: grammar.callNameEnd, range: r.upperBound..<buf.endIndex) else { break loop }
                rawCall += String(buf[..<gt.upperBound])
                let name = String(buf[r.upperBound..<gt.lowerBound])
                buf = String(buf[gt.upperBound...])
                startCall(name, into: &out)
                continue loop

            case .params:
                let pOpen = buf.range(of: grammar.paramOpen)
                let fClose = grammar.callClose.flatMap { buf.range(of: $0) }
                let cTag = buf.range(of: grammar.close)
                // Whichever terminator comes first decides.
                let first = [pOpen, fClose, cTag].compactMap { $0 }
                    .min { $0.lowerBound < $1.lowerBound }
                guard let hit = first else { break loop }
                if hit == pOpen {
                    guard let gt = buf.range(of: grammar.paramNameEnd, range: hit.upperBound..<buf.endIndex) else { break loop }
                    guard let header = grammar.parameter(buf[hit.upperBound..<gt.lowerBound]), currentArgs[header.name] == nil else {
                        // The rest is read as text again, so a later call in the same reply is still a call.
                        out.append(.malformed(rawCall))
                        resetCall()
                        state = .text
                        continue loop
                    }
                    let name = header.name
                    rawCall += String(buf[..<gt.upperBound])
                    buf = String(buf[gt.upperBound...])
                    valueRaw = ""
                    valueAtStart = true
                    valueLiteral = header.literal
                    // The declared type decides; DeepSeek's `string=` only for a parameter declared without one (a
                    // quantized model can mark `12` a string for an integer, which the client's schema would refuse).
                    let declared = schemas[currentName]?.params[name] ?? .unknown
                    streamsString = declared == .string || (declared == .unknown && header.literal == true)
                    if streamsString {
                        out.append(
                            .toolInputDelta(
                                id: currentID,
                                delta: (emittedAnyArg ? "," : "{") + JSONValue.quote(name) + ":\""))
                    }
                    state = .value(name)
                    continue loop
                }
                // </function> ends the parameters but not the block: the
                // call is emitted only once </tool_call> is consumed, so a
                // truncated block is reported malformed instead of being
                // dispatched. A bare </tool_call> here completes it directly.
                rawCall += String(buf[..<hit.upperBound])
                buf = String(buf[hit.upperBound...])
                if hit == cTag {
                    out.append(contentsOf: finishCall())
                    state = .text
                } else if grammar.manyPerBlock {
                    // The invoke is whole: emit it, and read on for another or the block's close.
                    out.append(contentsOf: finishCall())
                    blockCalls += 1
                    state = .call
                } else {
                    state = .closing
                }
                continue loop

            case .value(let param):
                guard let r = buf.range(of: grammar.paramClose) else {
                    // Search only a bounded tail on the next push. Keeping a
                    // closing tag's length also withholds its framing newline.
                    // The old code rescanned the entire growing argument on
                    // every token and never streamed a long file's contents.
                    let keep = min(buf.count, grammar.paramClose.count)
                    consumeValue(String(buf.dropLast(keep)), final: false, into: &out)
                    buf = String(buf.suffix(keep))
                    break loop
                }
                consumeValue(String(buf[..<r.lowerBound]), final: true, into: &out)
                rawCall += grammar.paramClose
                buf = String(buf[r.upperBound...])
                let kind = schemas[currentName]?.params[param] ?? .unknown
                let framed = grammar.framedValues
                // Untyped and marked: a string is one, JSON is parsed (else read as an untyped value).
                let marked =
                    kind != .unknown
                    ? nil
                    : valueLiteral.map { literal in
                        literal
                            ? .string(valueRaw)
                            : JSONValue.parse(valueRaw.trimmingCharacters(in: .whitespacesAndNewlines))
                                ?? Self.coerce(valueRaw, as: kind, framed: framed)
                    }
                let value = marked ?? Self.coerce(valueRaw, as: kind, framed: framed)
                if currentArgs[param] == nil { currentOrder.append(param) }
                currentArgs[param] = value
                out.append(
                    .toolInputDelta(
                        id: currentID,
                        delta: streamsString
                            ? "\""
                            : (emittedAnyArg ? "," : "{") + JSONValue.quote(param) + ":" + value.jsonText))
                emittedAnyArg = true
                state = .params
                continue loop

            case .closing:
                guard let r = buf.range(of: grammar.close) else { break loop }
                rawCall += String(buf[..<r.upperBound])
                buf = String(buf[r.upperBound...])
                out.append(contentsOf: finishCall())
                state = .text
                continue loop
            }
        }
        return out
    }

    /// Close the stream. Anything still buffered is text; an unterminated call
    /// is returned verbatim as `malformed` and the turn ends as text.
    public func flush() -> [ToolStreamEvent] {
        var out: [ToolStreamEvent] = []
        switch state {
        case .text:
            if !buf.isEmpty { out.append(.text(buf)) }
        case .value:
            // A length-limited reply still owns the stable argument text it
            // generated. Do not lose the decoder's final holdback on flush.
            let keep = Self.holdback(buf, grammar.paramClose)
            let tail = String(buf.suffix(keep))
            consumeValue(String(buf.dropLast(keep)), final: false, into: &out)
            out.append(.malformed(rawCall + tail))
        case .call where blockCalls > 0 && (rawCall + buf).allSatisfy(\.isWhitespace):
            break  // the reply ended after a whole invoke, before its block's close: the calls stand
        case .call, .params, .closing:
            out.append(.malformed(rawCall + buf))
        }
        buf = ""
        rawCall = ""
        state = .text
        return out
    }

    /// A call is named: a declared tool opens it; any other name sends the block through as the model's text.
    private func startCall(_ name: String, into out: inout [ToolStreamEvent]) {
        currentName = name
        guard schemas[currentName] != nil else {
            // A tool the request did not declare: the block is the model's text, streamed through as it comes.
            out.append(.text(rawCall))
            resetCall()
            state = .text
            return
        }
        currentID = nextID()
        currentArgs = [:]
        currentOrder = []
        emittedAnyArg = false
        out.append(.toolInputStart(id: currentID, name: currentName))
        state = .params
    }

    private func finishCall() -> [ToolStreamEvent] {
        var out: [ToolStreamEvent] = []
        // A call with no parameters still needs a syntactically whole input.
        out.append(.toolInputDelta(id: currentID, delta: emittedAnyArg ? "}" : "{}"))
        out.append(.toolInputEnd(id: currentID))
        out.append(
            .toolCall(
                ParsedToolCall(
                    id: currentID, name: currentName, arguments: currentArgs,
                    order: currentOrder)))
        resetCall()
        return out
    }

    private func resetCall() {
        currentID = ""
        currentName = ""
        currentArgs = [:]
        currentOrder = []
        emittedAnyArg = false
        rawCall = ""
        valueRaw = ""
        valueLiteral = nil
    }

    private func consumeValue(_ raw: String, final: Bool, into out: inout [ToolStreamEvent]) {
        valueRaw += raw
        rawCall += raw
        guard streamsString else { return }
        var value = raw
        if valueAtStart, !value.isEmpty {
            if grammar.framedValues, value.hasPrefix("\n") { value.removeFirst() }
            valueAtStart = false
        }
        if final, grammar.framedValues, value.hasSuffix("\n") { value.removeLast() }
        if !value.isEmpty {
            // Escape each stable span, without adding another pair of quotes.
            out.append(
                .toolInputDelta(
                    id: currentID,
                    delta: String(JSONValue.quote(value).dropFirst().dropLast())))
        }
    }

    /// How many trailing characters of `s` must be withheld because they could
    /// still grow into `tag`. Zero for ordinary prose, so text streams at once.
    public static func holdback(_ s: String, _ tag: String) -> Int {
        let tail = s.suffix(tag.count - 1)
        let maxLen = tail.count
        if maxLen <= 0 { return 0 }
        for len in stride(from: maxLen, through: 1, by: -1) {
            if tag.hasPrefix(tail.suffix(len)) { return len }
        }
        return 0
    }

    /// The declared type decides how a raw span becomes JSON.
    ///
    /// Qwen's template writes a newline after `<parameter=NAME>` and before
    /// `</parameter>`; both are framing, not content, and exactly one of each
    /// is removed (`framed`). A value that itself ends in a blank line therefore survives.
    public static func coerce(_ raw: String, as kind: ToolParamKind, framed: Bool = true) -> JSONValue {
        var v = raw
        if framed, v.hasPrefix("\n") { v.removeFirst() }
        if framed, v.hasSuffix("\n") { v.removeLast() }
        let trimmed = v.trimmingCharacters(in: .whitespacesAndNewlines)
        switch kind {
        case .string:
            return .string(v)
        case .integer:
            // `Int(d)` traps outside Int's range, and a model can write 1e20
            // or inf; those stay numbers, or text, rather than end the process.
            if let i = Int(trimmed) { return .int(i) }
            if let d = Double(trimmed), d.isFinite { return Int(exactly: d).map(JSONValue.int) ?? .double(d) }
            return .string(v)
        case .number:
            if let i = Int(trimmed) { return .int(i) }
            // JSON has no infinity or NaN.
            if let d = Double(trimmed), d.isFinite { return .double(d) }
            return .string(v)
        case .boolean:
            if trimmed == "true" { return .bool(true) }
            if trimmed == "false" { return .bool(false) }
            return .string(v)
        case .array, .object:
            if let parsed = JSONValue.parse(trimmed) { return parsed }
            return .string(v)
        case .unknown:
            // `anyOf` and undeclared names: take JSON when it is unambiguous,
            // and a bare word stays a string rather than becoming a number.
            if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") {
                if let parsed = JSONValue.parse(trimmed) { return parsed }
            }
            if trimmed == "true" { return .bool(true) }
            if trimmed == "false" { return .bool(false) }
            if let i = Int(trimmed) { return .int(i) }
            return .string(v)
        }
    }
}

/// A tool as the client declared it: the raw JSON Schema the template renders,
/// plus the reduced type table the parser coerces with.
///
/// One type holds both because they must not drift. The schema the model is
/// shown and the schema its answer is typed against are the same document; a
/// second, hand-maintained copy of the parameter types would be a silent way
/// for a call to be parsed against a tool the model never saw.
public struct ToolDefinition: Sendable {
    public let name: String
    public let description: String
    /// The JSON Schema object for the parameters, verbatim from the request.
    public let parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// The parameter types, including a single non-null type expressed through
    /// `type: ["string", "null"]` or `anyOf`, and same-document JSON Pointer
    /// references. Genuine unions and unsupported references stay `.unknown`.
    /// The original schema remains intact for the prompt.
    public var schema: ToolSchema {
        var params: [String: ToolParamKind] = [:]
        var inference = SchemaInference(document: parameters)
        if let (root, _, referencesAllowed) = inference.resolve(parameters, isRoot: true),
            case .object(let props)? = root["properties"]
        {
            for (key, value) in props {
                // A shared definition used by another property is not a cycle.
                var fieldInference = SchemaInference(document: parameters)
                let types = fieldInference.types(value, referencesAllowed: referencesAllowed) ?? []
                let concrete = types.filter { $0 != "null" }
                params[key] =
                    concrete.count == 1
                    ? ToolParamKind(rawValue: concrete[0]) ?? .unknown : .unknown
            }
        }
        return ToolSchema(name: name, params: params)
    }

    /// Type inference only: no validation, remote retrieval, anchors or resource
    /// rebasing. A nested `$id` can change what `#` means, so it is not traversed.
    private struct SchemaInference {
        let document: JSONValue
        // Bound schema visits AND pointer steps per property, including branching
        // anyOf/ref graphs. Exhaustion keeps the existing unknown-type fallback;
        // this is a defensive work limit, not a supported schema-depth promise.
        var remaining = 128
        private static let refSiblings: Set<String> = [
            "$ref", "$defs", "definitions", "$schema", "$id", "$comment", "$anchor",
            "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
        ]

        private static func hasResourceID(_ field: [String: JSONValue]) -> Bool {
            // In a properties/$defs map, a key named "$id" holds a schema,
            // not a URI. Do not mistake it for a resource identifier.
            ["$id", "id"].contains {
                if case .string? = field[$0] { return true }
                return false
            }
        }

        mutating func resolve(
            _ value: JSONValue, visited: Set<[String]> = [],
            referencesAllowed: Bool = true, isRoot: Bool = false
        )
            -> ([String: JSONValue], Set<[String]>, Bool)?
        {
            guard remaining > 0, case .object(let field) = value else { return nil }
            remaining -= 1
            let canReference = referencesAllowed && (isRoot || !Self.hasResourceID(field))
            // A resource ID does not affect a directly declared scalar type,
            // but references anywhere beneath it need unsupported rebasing.
            guard let reference = field["$ref"] else { return (field, visited, canReference) }
            // Ref siblings have different semantics across schema drafts. Keep
            // the existing inference from this field's own declarations when
            // constraints are present, rather than merging them with the target.
            guard Set(field.keys).isSubset(of: Self.refSiblings) else {
                return (field, visited, canReference)
            }
            guard canReference, case .string(let ref) = reference, ref.unicodeScalars.first == "#",
                let pointer = String(ref.unicodeScalars.dropFirst()).removingPercentEncoding,
                pointer.isEmpty || pointer.unicodeScalars.first == "/"
            else { return nil }
            var tokens: [String] = []
            for raw in pointer.unicodeScalars.split(separator: "/", omittingEmptySubsequences: false).dropFirst() {
                var token = ""
                var characters = raw.makeIterator()
                while let ch = characters.next() {
                    if ch == "~" {
                        guard let escaped = characters.next(), escaped == "0" || escaped == "1" else { return nil }
                        token.unicodeScalars.append(escaped == "0" ? "~" : "/")
                    } else {
                        token.unicodeScalars.append(ch)
                    }
                }
                tokens.append(token)
            }
            guard !visited.contains(tokens) else { return nil }
            var target = document
            for (index, token) in tokens.enumerated() {
                guard remaining > 0 else { return nil }
                remaining -= 1
                switch target {
                case .object(let object):
                    // Do not cross an embedded resource on the way to a target.
                    guard index == 0 || !Self.hasResourceID(object),
                        let entryIndex = object.index(forKey: token)
                    else { return nil }
                    let (key, next) = object[entryIndex]
                    // Swift String equality normalizes Unicode; JSON Pointer
                    // requires the same code points, without normalization.
                    guard key.utf8.elementsEqual(token.utf8) else { return nil }
                    target = next
                case .array(let array):
                    guard let index = Int(token), String(index) == token,
                        array.indices.contains(index)
                    else { return nil }
                    target = array[index]
                default: return nil
                }
            }
            if !tokens.isEmpty, case .object(let object) = target, Self.hasResourceID(object) { return nil }
            return resolve(target, visited: visited.union([tokens]), isRoot: tokens.isEmpty)
        }

        mutating func types(_ value: JSONValue, visited: Set<[String]> = [], referencesAllowed: Bool = true) -> [String]? {
            guard let (field, path, canReference) = resolve(value, visited: visited, referencesAllowed: referencesAllowed)
            else { return nil }
            if let type = field["type"] {
                if case .string(let name) = type { return [name] }
                guard case .array(let options) = type else { return nil }
                let names = options.compactMap {
                    if case .string(let name) = $0 { return name }
                    return nil
                }
                return names.count == options.count ? names : nil
            }
            guard case .array(let options)? = field["anyOf"], !options.isEmpty else { return nil }
            var names: [String] = []
            for option in options {
                guard let branch = types(option, visited: path, referencesAllowed: canReference) else { return nil }
                names += branch
            }
            return names
        }
    }

    /// The value the chat template expects in its `tools` list.
    public var templateValue: [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": parameters.any,
            ] as [String: any Sendable],
        ]
    }
}

/// Splits a thinking reply at the first `</think>` into reasoning and answer, streamed: until the tag arrives it
/// withholds the last seven characters so a tag split across deltas never leaks into the reasoning.
private final class ServingThinkSplitter {
    private static let tag = "</think>"
    private var buf = ""
    private var closed = false
    private var answered = false

    init() {}

    /// Splits one delta into (thinking, content). Until the tag arrives the
    /// last few characters are withheld, so a tag straddling two deltas is
    /// never emitted as reasoning text.
    func push(_ s: String) -> (String, String) {
        if closed { return ("", answer(s)) }
        buf += s
        if let r = buf.range(of: Self.tag) {
            let think = String(buf[..<r.lowerBound])
            let rest = String(buf[r.upperBound...])
            buf = ""
            closed = true
            return (think, answer(rest))
        }
        let keep = min(buf.count, Self.tag.count - 1)
        let emit = String(buf.dropLast(keep))
        buf = String(buf.suffix(keep))
        return (emit, "")
    }

    /// The answer without the newlines the model puts after `</think>`,
    /// which can arrive in the deltas after the tag.
    private func answer(_ s: String) -> String {
        if answered { return s }
        let rest = s.drop { $0 == "\n" }
        if !rest.isEmpty { answered = true }
        return String(rest)
    }

    /// Whatever is still withheld when generation ends. A response that never
    /// closed its reasoning is all thinking and no answer, and says so.
    func flush() -> (String, String) {
        let rest = buf
        buf = ""
        return closed ? ("", answer(rest)) : (rest, "")
    }
}

/// An error a client may see: HTTP status, message, and OpenAI-style `type`/`code`. Each wire maps it to its own
/// error shape; `from` turns any other error into a generic 500 (503 for memory, 507 for storage) and logs the cause to
/// stderr.
public struct ServingFailure: Error, CustomStringConvertible, Sendable {
    public let status: Int
    public let description: String
    public let type, code: String
    public init(_ description: String, status: Int = 400, type: String = "invalid_request_error", code: String? = nil) {
        self.description = description
        self.status = status
        self.type = type
        self.code = code ?? type
    }
    public var wire: [String: Any] { ["message": description, "type": type, "code": code] }
    public static func from(_ error: Error) -> Self {
        if let failure = error as? Self { return failure }
        // A client that left (or a server stopping) cancels its request: not a failure worth a log line.
        if error is IOOperationCancelled { return Self("The request was cancelled", status: 503, type: "cancelled") }
        if let engine = error as? EngineError, let code = engine.code,
            [
                "insufficient_memory", "insufficient_storage", "kv_io_failure", "gpu_interrupted", "gpu_stalled", "gpu_lost",
                "drive_stalled", "drive_away",
                "json_mode",
            ].contains(code)
        {
            FileHandle.standardError.write(Data("origami: \(engine.description)\n".utf8))
            switch code {
            // Ledger numbers describe the host; the client gets the code only.
            case "insufficient_memory": return Self("Insufficient memory for this request", status: 503, type: code)
            // A full disk or a failed context-file read or write: the message says which, and what to do.
            case "insufficient_storage": return Self(engine.description, status: 507, type: code)
            // macOS stopped the GPU work (`GPURetry` already ran it again): a 503 clients and agents retry.
            case "gpu_interrupted": return Self(GPURetry.message, status: 503, type: "server_busy", code: code)
            // A GPU step outran its bound three times (`GPUWait`), or never ended: the model process restarts.
            case "gpu_stalled":
                return Self(
                    "The GPU stopped answering in time (\(engine.description)); please retry your request", status: 503,
                    type: "server_busy", code: code)
            case "gpu_lost":
                return Self(
                    "The GPU stopped answering (\(engine.description)); the model restarts, please retry your request in a minute",
                    status: 503,
                    type: "engine_unavailable", code: code)
            // A demand read with no second source did not return within its bound (`ExpertReadStall`).
            case "drive_stalled": return Self(engine.description, status: 503, type: "engine_unavailable", code: code)
            // The model's drive is gone (`ModelDrive`): the same error to clients, at once instead of after a bound.
            case "drive_away": return Self(engine.description, status: 503, type: "engine_unavailable", code: "drive_stalled")
            default: return Self(engine.description, status: 500, type: code)
            }
        }
        // The client gets a generic message; the log keeps the cause.
        FileHandle.standardError.write(Data("origami: request failed: \(error)\n".utf8))
        return Self("Inference request failed", status: 500, type: "server_error")
    }
    /// A request's `origami_time_limit` passed while its prompt was read: no reply yet, so the answer the front gives a
    /// request that outlives its deadline (408 `request_timeout`). The conversation keeps what was read, as on a cancel.
    public static func timeLimit() -> Self {
        Self("The request's time limit (origami_time_limit) passed while its prompt was read", status: 408, type: "request_timeout")
    }
    public static func isTimeLimit(_ error: Error) -> Bool { (error as? Self)?.code == "request_timeout" }
    public static func pastTimeLimit(_ request: ServingRequest) -> Bool { request.deadline.map { Date() >= $0 } ?? false }
    public static func checkTimeLimit(_ request: ServingRequest) throws { if pastTimeLimit(request) { throw timeLimit() } }
    /// Prompt encoding only normalizes the conversation and renders the chat
    /// template, so its failures (a template `raise_exception` such as a
    /// system-only chat, two reasoning fields, a mismatched tool name) describe
    /// the client's request, not an engine fault.
    public static func promptEncoding(_ error: Error) -> Self {
        if error is Self || error is IOOperationCancelled || (error as? EngineError)?.code == "insufficient_memory" { return from(error) }
        return Self("Invalid conversation: \((error as? LocalizedError)?.errorDescription ?? String(describing: error))")
    }
}

/// OpenAI `tool_choice`. `none` hides the tools from the prompt; `required` and `named` add an instruction to the end
/// of the last user turn, and decode allows only the tool call (`ServingRequest.decodeGrammar`).
public enum ServingToolChoice: Equatable, Sendable {
    case auto, none, required
    case named(String)
}

/// An in-chat command: the whole last user message is `origami <verb> [args]`, in any case. Anything
/// else (an unknown verb, extra words, a trailing `?`) is an ordinary chat message.
public struct ChatCommand: Sendable, Equatable {
    /// The first line of every reply, and how an earlier reply is recognised in the history.
    public static let marker = "Origami ·"
    public let verb: String
    public let arguments: [String]
    public init(verb: String, arguments: [String] = []) {
        self.verb = verb
        self.arguments = arguments
    }

    public static func parse(_ text: String) -> ChatCommand? {
        // hermes-agent appends "\n\n<memory-context>…" to the user's turn when stored facts match it.
        let typed = text.components(separatedBy: "\n\n<memory-context>").first ?? text
        let words = typed.split(whereSeparator: \.isWhitespace).map(String.init)
        guard words.first?.lowercased() == "origami" else { return nil }
        guard words.count > 1 else { return ChatCommand(verb: "help") }
        let verb = words[1].lowercased()
        let arguments = Array(words.dropFirst(2))
        switch verb {
        case "help", "status", "prompts": guard arguments.isEmpty else { return nil }
        case "save": guard arguments.count <= 1 else { return nil }
        case "forget": guard arguments.count == 1 else { return nil }
        case "fast": guard arguments.count == 1, ["on", "off"].contains(arguments[0].lowercased()) else { return nil }
        case "quality":
            guard arguments.count <= 1, arguments.first.map({ QualityDial.level($0.lowercased()) != nil }) ?? true else { return nil }
        default: return nil
        }
        return ChatCommand(verb: verb, arguments: ["fast", "quality"].contains(verb) ? arguments.map { $0.lowercased() } : arguments)
    }
    /// A saved-state name (`origami save <name>`, `X-Origami-Keep`). `all` is refused: `origami forget all` means everything.
    public static func isName(_ text: String) -> Bool {
        text.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", options: .regularExpression) != nil && text.lowercased() != "all"
    }
    public static let nameRule = "is a name of letters, digits, '.', '_' and '-' (up to 64), other than \"all\""
    /// A message's text: a string, or its text parts joined.
    public static func text(_ content: Any?) -> String {
        if let text = content as? String { return text }
        return (content as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
    }
}

/// Server-side defaults for fields a request omits (`origami config set reasoning_effort …`).
public struct ServingDefaults: Sendable {
    public static let efforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    /// The levels a family's chat template tells apart, each once: every name in `efforts` is still accepted and
    /// renders as one of these. Qwen: `minimal` is `none`, `high`, `xhigh` and `max` are one (the template's xhigh).
    /// GLM-5.3: `minimal` is `none`, `medium`, `xhigh` and `max` are one (the template's Max). DeepSeek-V4.1:
    /// `minimal` is `none`, `xhigh` is `max`.
    public static func efforts(for family: ModelFamily) -> [String] {
        switch family {
        case .qwen: return ["none", "low", "medium", "high"]
        case .glm: return ["none", "low", "high", "max"]
        case .deepseek: return ["none", "low", "medium", "high", "max"]
        }
    }
    /// The effort of a request that sends no thinking control: harnesses rarely send one.
    public var reasoningEffort = "none"
    /// Thinking budget of a request that sends none; nil adds no explicit cap. Every model still closes the think
    /// block where only the answer room is left (`ThinkCloser`).
    public var maxThinkingTokens: Int?
    /// `origami <verb>` messages are commands (`commands` setting).
    public var commands = true
    /// Streamed replies carry "Origami ·" status lines as reasoning while they wait or read the prompt.
    public var statusMessages = false
    /// The quality dial of a request that names no level (`origami config set quality …`; exact unless set).
    public var quality = QualityDial.exact
    /// Sampling for the fields a request omits, per mode: the model's recommendation, or custom values (`sampling`).
    public var sampling = SamplingDefaults.fallback
    /// The model's own recommendation, which `sampling` is unless set custom (for /v1/status and the dashboard).
    public var recommendedSampling = SamplingDefaults.fallback
    /// `sampling_override`: `sampling` even over a request's own values.
    public var samplingOverride = false
    public init() {}
}

/// One validated OpenAI chat request (every other wire is translated into this shape first): messages and tools as
/// template values, sampling, budgets, thinking control, and Origami's own fields. `parse` refuses deep or oversized
/// JSON before any recursive consumer sees it.
public struct ServingRequest: Sendable {
    public let model: String
    public var messages: [[String: TemplateValue]]
    public let tools: [[String: TemplateValue]]
    public let definitions: [ToolDefinition]
    public let sampling: SamplingParameters
    public let seed: UInt64?
    public var maximumTokens: Int
    public let contextLimit: Int
    /// False when max_tokens/max_completion_tokens were omitted; see `outputBudget`.
    public let explicitOutputBudget: Bool
    public let reasoningEffort: String
    public let stopSequences: [String]
    public var stream: Bool
    public let includeUsage, parallelToolCalls: Bool
    public let toolChoice: ServingToolChoice
    /// `origami_mode`: a quality level (any of `QualityDial.levels`; "fast" is "faster"); absent, the server's default
    /// (exact unless set). A non-exact level runs only where the configuration allows it, and the server runs a level its
    /// model does not offer as the model's nearest one (`QualityDial.offered(by:)`). The model id `<alias>-<level>`
    /// (`-faster`, `-fastest`, `-dynamic-fastest`, `-fast`…) selects one too, for clients that cannot send body fields.
    public var quality: QualityDial = .exact
    /// `origami_teacher_tokens`: decode compares its own greedy picks with these tokens and continues from them
    /// (measurement: top-1 agreement with a reference reply, reported in usage). Nil decodes normally.
    public var teacherTokens: [Int]?
    /// `origami_return_tokens`: usage carries the reply's token IDs (`origami_tokens`).
    public var returnTokens = false
    /// The last message is a tool result: an agent's turn, which waits behind a person's message.
    public var agentTurn = false
    /// OpenRouter `reasoning.exclude`: think, but return only the answer.
    public var excludeReasoning = false
    /// Most tokens inside the think block before the server closes it; nil leaves only the answer room.
    public var thinkingBudget: Int?
    /// `prompt_cache_policy: "no_store"`: reuse saved state but save nothing (title generators, summarizers).
    public var noStore = false
    /// `X-Origami-Keep: <name>`: keep this request's saved prompt pinned under a name.
    public var keep: String?
    /// The last message is an in-chat command (answered without the model, or for `save` by a prefill).
    public var command: ChatCommand?
    /// The request chose its quality level (`origami_mode` or a `-<level>` id): a sticky `origami quality` does not apply.
    public var explicitMode = false
    /// SHA-256 of the system prompt's text: the harness's identity for `origami fast on` and `origami quality`.
    public var systemKey: String?
    /// `status_messages`: this request's choice; nil takes the server's.
    public var statusMessages: Bool?
    /// `return_progress`: llama.cpp-style `prompt_progress` frames while the prompt is read.
    public var returnProgress = false
    /// `origami_time_limit` (seconds, from arrival): the engine ends the reply at the first token or prompt-slice
    /// boundary past it, with the tokens so far and finish reason `length`, and keeps the conversation. Nil: no limit.
    public var deadline: Date?
    /// `origami_tune`: execution switches for this request alone (`origami tune`; exact requests only).
    public var tune: TuneOverrides?
    /// `response_format` `json_object` or `json_schema` (Responses' `text.format`, Ollama's `format`): the answer is
    /// sampled under this grammar (`JSONConstraint`). Nil: sampling is untouched.
    public var responseFormat: JSONGrammar?
    /// The reply may open a tool call: tools are declared and `tool_choice` is not `none`.
    public var allowsToolCalls: Bool { !definitions.isEmpty && toolChoice != .none }
    /// What decode follows: with `tool_choice` required or named only the tool call (JSON mode's branch is left out),
    /// else the request's JSON mode. A teacher-forced reply is not constrained.
    package var decodeGrammar: JSONGrammar? {
        guard teacherTokens == nil else { return responseFormat }
        switch toolChoice {
        case .required, .named: return allowsToolCalls ? .toolCallOnly : responseFormat
        default: return responseFormat
        }
    }
    /// The bytes a reply opens its tool call with in `grammar`: for a named choice through the tool's name.
    package func toolOpener(_ grammar: ToolCallGrammar) -> String? {
        guard allowsToolCalls else { return nil }
        guard case .named(let name) = toolChoice else { return grammar.open }
        return grammar.open + (grammar.callOpen.map { "\n" + $0 + name + grammar.callNameEnd } ?? name)
    }

    /// The grammar of an OpenAI `response_format`; nil for `{"type": "text"}`.
    package static func responseFormat(_ value: Any) throws -> JSONGrammar? {
        guard let format = value as? [String: Any], let type = format["type"] as? String else {
            throw ServingFailure("response_format must be an object with a type")
        }
        switch type {
        case "text": return nil
        case "json_object": return .object
        case "json_schema":
            guard let specification = format["json_schema"] as? [String: Any] else {
                throw ServingFailure("response_format json_schema needs a json_schema object holding the schema")
            }
            if let strict = specification["strict"], !(strict is NSNull) {
                guard let flag = strict as? NSNumber, CFGetTypeID(flag) == CFBooleanGetTypeID() else {
                    throw ServingFailure("response_format json_schema.strict must be true or false")
                }
            }
            // Strict or not, the schema is enforced. Without one, any JSON object.
            guard let schema = specification["schema"], !(schema is NSNull) else { return .object }
            return try JSONGrammar(schema: schema)
        default: throw ServingFailure("response_format type must be \"text\", \"json_object\" or \"json_schema\"")
        }
    }

    /// The output budget for a tokenized prompt: the limit, explicit or not, shrunk to the room the prompt leaves.
    public func outputBudget(promptTokens: Int) throws -> Int {
        let room = contextLimit - promptTokens
        // Anthropic's wording, which Claude Code reads: a prompt that leaves less room than the reply may need (what it
        // asks for, at most 512 tokens) is compacted, never answered with a reply cut short.
        let needed = max(1, min(maximumTokens, 512))
        guard room >= needed else {
            throw ServingFailure(
                "prompt is too long: \(promptTokens) tokens > \(contextLimit - needed) maximum", status: 400,
                type: "context_length_exceeded")
        }
        return min(maximumTokens, room)
    }

    /// Refuses arrays/objects nested deeper than 64 before anything parses
    /// them: the parser, JSONValue, Codable TemplateValue and swift-jinja all
    /// recurse per level, and requests run on 512 KiB GCD worker stacks.
    /// Also refuses a number beyond a double's range (`1e400`): Foundation reads it as infinity and then raises an
    /// exception, ending the server, when the request is written as JSON again (wire translation, template values).
    /// Returns the bytes outside strings.
    @discardableResult package static func checkNesting(_ json: some Sequence<UInt8>) throws -> Int {
        var depth = 0
        var inString = false
        var escaped = false
        var structure = 0
        var number: [UInt8] = []
        func finishNumber() throws {
            defer { number.removeAll(keepingCapacity: true) }
            guard let value = Double(String(decoding: number, as: UTF8.self)), !value.isFinite else { return }
            throw ServingFailure("JSON number \(String(decoding: number.prefix(32), as: UTF8.self)) is out of range")
        }
        for byte in json {
            if !inString {
                if (48...57).contains(byte) || byte == UInt8(ascii: "-") || byte == UInt8(ascii: "+") || byte == UInt8(ascii: ".")
                    || byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E")
                {
                    if number.count < 4_096 { number.append(byte) }
                } else if !number.isEmpty {
                    try finishNumber()
                }
            }
            if inString {
                if escaped {
                    escaped = false
                } else if byte == UInt8(ascii: "\\") {
                    escaped = true
                } else if byte == UInt8(ascii: "\"") {
                    inString = false
                }
                continue
            }
            structure += 1
            if byte == UInt8(ascii: "\"") {
                inString = true
            } else if byte == UInt8(ascii: "[") || byte == UInt8(ascii: "{") {
                depth += 1
                guard depth <= 64 else { throw ServingFailure("JSON nests deeper than 64 levels") }
            } else if byte == UInt8(ascii: "]") || byte == UInt8(ascii: "}") {
                depth -= 1
            }
        }
        if !number.isEmpty { try finishNumber() }
        return structure
    }

    public static func parse(
        _ data: Data, model: String = "qwen3.8-flash-next:4bit", contextTier: Int,
        defaults: ServingDefaults = ServingDefaults(), maximumStructureBytes: Int = 1_024 * 1_024
    ) throws -> Self {
        guard contextTier > 0 else { throw ServingFailure("Invalid context tier", status: 503) }
        guard try checkNesting(data) <= maximumStructureBytes else {
            throw ServingFailure("Request JSON is too large outside its strings", status: 413)
        }
        let object: Any
        do { object = try JSONSerialization.jsonObject(with: data) } catch { throw ServingFailure("Malformed JSON request") }
        // An explicit top-level null means the field is absent (OpenAI clients send `"stop": null`).
        guard let body = (object as? [String: Any])?.filter({ !($0.value is NSNull) }) else {
            throw ServingFailure("Request must be a JSON object")
        }
        // Fields this server does not use are ignored, as clients send many vendor extensions.
        // One model is served under any name (wrappers such as llama-swap rename it); a name
        // ending in "-<level>" ("-faster", "-fastest", "-fast"…) selects that quality level for clients that cannot send body fields.
        let requestedModel = body["model"] as? String ?? model
        // `origami_mode` (a level name). The old field name is refused, not ignored: ignoring it would run exact
        // where a client asked for fast.
        guard body["qfs_mode"] == nil else { throw ServingFailure("qfs_mode is now origami_mode (\"exact\", \"faster\" or \"fastest\")") }
        let modeField = body["origami_mode"]
        let levelNames = QualityDial.levels.map(\.name)
        var named: QualityDial?
        if let modeField {
            guard let level = (modeField as? String).flatMap(QualityDial.level) else {
                throw ServingFailure("origami_mode must be " + QualityDial.offeredByModel)
            }
            named = level
        }
        // Only the documented form: this model's own id (or `origami`, the setup's name) and "-<level>", so a model whose
        // name ends in a level's ("something-fast") is not read as one. The longest match, so "-fastest" is not "-fast".
        let bases = [model, String(model.split(separator: ":", maxSplits: 1).first ?? ""), "origami"]
        let asked = requestedModel.hasSuffix(":latest") ? String(requestedModel.dropLast(":latest".count)) : requestedModel
        if let suffix = (levelNames.dropFirst() + QualityDial.aliases.keys).filter({ level in
            bases.contains { asked == $0 + "-" + level }
        }).max(by: { $0.count < $1.count }) {
            guard named == nil || named == QualityDial.level(suffix) else {
                throw ServingFailure("origami_mode \"\(named!.name)\" conflicts with the model id's -\(suffix)")
            }
            named = QualityDial.level(suffix)
        }
        var teacher: [Int]?
        if let raw = body["origami_teacher_tokens"] {
            guard let list = raw as? [Any], !list.isEmpty, list.count <= 1_048_576,
                case let ids = list.compactMap({
                    ($0 as? NSNumber).flatMap {
                        CFGetTypeID($0) == CFBooleanGetTypeID()
                            || !$0.doubleValue.isFinite ? nil : Int(exactly: $0)
                    }
                }),
                ids.count == list.count, ids.allSatisfy({ $0 >= 0 })
            else {
                throw ServingFailure("origami_teacher_tokens must be a list of token IDs")
            }
            teacher = ids
        }
        func number(_ key: String, default fallback: Double, range: ClosedRange<Double>) throws -> Double {
            guard let raw = body[key] else { return fallback }
            guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                value.doubleValue.isFinite, range.contains(value.doubleValue)
            else { throw ServingFailure("Invalid \(key)") }
            return value.doubleValue
        }
        func integer(_ key: String, default fallback: Int, range: ClosedRange<Int>) throws -> Int {
            guard let raw = body[key] else { return fallback }
            guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                let result = Int(exactly: value), range.contains(result)
            else { throw ServingFailure("Invalid \(key)") }
            return result
        }
        func boolean(_ value: Any?, default fallback: Bool) throws -> Bool {
            guard let value else { return fallback }
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw ServingFailure("Expected a JSON boolean")
            }
            return number.boolValue
        }
        guard try integer("n", default: 1, range: 1...1) == 1 else { throw ServingFailure("Only one sequence is supported") }
        if let value = body["logprobs"], try boolean(value, default: false) { throw ServingFailure("logprobs are unsupported") }
        if body["top_logprobs"] != nil { throw ServingFailure("top_logprobs are unsupported") }
        if let value = body["logit_bias"] {
            guard let map = value as? [String: Any], map.isEmpty else { throw ServingFailure("Only empty logit_bias is supported") }
        }
        let responseFormat = try body["response_format"].flatMap(Self.responseFormat)
        if responseFormat != nil, teacher != nil { throw ServingFailure("origami_teacher_tokens cannot be combined with JSON mode") }
        if let value = body["store"], try boolean(value, default: false) { throw ServingFailure("Stored completions are unsupported") }
        if let value = body["metadata"] {
            guard let map = value as? [String: Any], map.values.allSatisfy({ $0 is String }) else {
                throw ServingFailure("metadata must contain text values")
            }
        }
        guard body["prompt_cache_policy"] == nil || ["default", "no_store"].contains(body["prompt_cache_policy"] as? String ?? "") else {
            throw ServingFailure("prompt_cache_policy must be \"default\" or \"no_store\"")
        }
        for key in ["prompt_cache_key", "prompt_cache_retention", "safety_identifier", "service_tier"] {
            if let value = body[key], !(value is String) { throw ServingFailure("\(key) must be text") }
        }
        let stops: [String]
        if let value = body["stop"] {
            if let string = value as? String {
                stops = [string]
            } else if let strings = value as? [String] {
                stops = strings
            } else {
                throw ServingFailure("stop must be text or an array of text")
            }
        } else {
            stops = []
        }
        _ = try StopSequenceFilter(stops: stops, reasoningOpen: false)
        var contextLimit = contextTier
        if let raw = body["options"] {
            guard let options = raw as? [String: Any] else { throw ServingFailure("options must be an object") }
            if let value = options["num_ctx"] {
                guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                    let count = Int(exactly: number), (1...contextTier).contains(count)
                else { throw ServingFailure("options.num_ctx exceeds served context or is invalid") }
                contextLimit = count
            }
        }
        let defaultBudget = max(1, min(min(8192, max(256, contextTier / 4)), contextTier - 1))
        // One field under three names; -1 asks for the most the context leaves (like an omitted one, unshrunk by the default).
        var budget: (key: String, value: Int)?
        for key in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
            guard let raw = body[key] else { continue }
            guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let exact = Int(exactly: number),
                exact == -1 || exact >= 1
            else {
                throw ServingFailure("Invalid \(key)")
            }
            // More than the context holds is clamped to it, as the prompt's room clamps it later (`outputBudget`).
            let value = min(exact, contextLimit)
            if let budget, budget.value != value {
                throw ServingFailure("Conflicting output token limits: \(budget.key) \(budget.value), \(key) \(value)")
            }
            budget = (key, value)
        }
        let completionTokens = budget.map { $0.value == -1 ? contextLimit : $0.value } ?? min(defaultBudget, contextLimit)
        var seed: UInt64?
        if let raw = body["seed"] {
            guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
                throw ServingFailure("Invalid integer seed")
            }
            if let signed = Int64(exactly: number) {
                seed = UInt64(bitPattern: signed)
            } else if let unsigned = UInt64(exactly: number) {
                seed = unsigned
            } else {
                throw ServingFailure("Invalid integer seed")
            }
        }
        // Thinking control under every name harnesses send; two of them that disagree are a 400.
        // A field the request sends wins; the server default fills a request that sends none.
        var level: (field: String, value: String)?
        var toggle: (field: String, value: Bool)?
        func setLevel(_ field: String, _ raw: Any?) throws {
            guard let raw else { return }
            guard let value = raw as? String, ServingDefaults.efforts.contains(value) else {
                throw ServingFailure("Unsupported \(field); use one of \(ServingDefaults.efforts.joined(separator: ", "))")
            }
            if let level, level.value != value { throw ServingFailure("\(field) conflicts with \(level.field)") }
            level = (field, value)
        }
        func setToggle(_ field: String, _ value: Bool?) throws {
            guard let value else { return }
            if let toggle, toggle.value != value { throw ServingFailure("\(field) conflicts with \(toggle.field)") }
            toggle = (field, value)
        }
        func objectField(_ key: String) throws -> [String: Any]? {
            guard let raw = body[key] else { return nil }
            guard let value = raw as? [String: Any] else { throw ServingFailure("\(key) must be an object") }
            return value.filter { !($0.value is NSNull) }
        }
        // The thinking budget, under the names harnesses send it.
        var thinkingBudget: (field: String, value: Int)?
        func setBudget(_ field: String, _ raw: Any?) throws {
            guard let raw else { return }
            guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let value = Int(exactly: number), value >= 1
            else {
                throw ServingFailure("Invalid \(field)")
            }
            if let thinkingBudget, thinkingBudget.value != value { throw ServingFailure("\(field) conflicts with \(thinkingBudget.field)") }
            thinkingBudget = (field, value)
        }
        for key in ["max_thinking_tokens", "thinking_budget_tokens", "thinking_budget", "thinking_token_budget"] {
            try setBudget(key, body[key])
        }
        try setLevel("reasoning_effort", body["reasoning_effort"])
        try setToggle("think", body["think"].map { try boolean($0, default: true) })  // Ollama
        var excludeReasoning = false
        if let reasoning = try objectField("reasoning") {  // OpenRouter: hermes-agent, aider
            try setLevel("reasoning.effort", reasoning["effort"])
            try setToggle("reasoning.enabled", reasoning["enabled"].map { try boolean($0, default: true) })
            excludeReasoning = try boolean(reasoning["exclude"], default: false)
            try setBudget("reasoning.max_tokens", reasoning["max_tokens"])
            if reasoning["max_tokens"] != nil { try setToggle("reasoning.max_tokens", true) }  // OpenRouter: a budget enables reasoning
        }
        // OpenRouter's older field: false is reasoning.exclude, true turns reasoning on (as `reasoning: {}`) unless another field sets it.
        let includeReasoning = try body["include_reasoning"].map { try boolean($0, default: true) }
        if includeReasoning == false { excludeReasoning = true }
        if let thinking = try objectField("thinking") {  // Anthropic: aider --thinking-tokens
            guard let type = thinking["type"] as? String, ["enabled", "adaptive", "disabled"].contains(type) else {
                throw ServingFailure("thinking.type must be \"enabled\" or \"disabled\"")
            }
            try setToggle("thinking.type", type != "disabled")
            try setBudget("thinking.budget_tokens", thinking["budget_tokens"])
        }
        if let kwargs = try objectField("chat_template_kwargs") {  // vLLM, llama.cpp
            try setToggle("chat_template_kwargs.enable_thinking", kwargs["enable_thinking"].map { try boolean($0, default: true) })
            try setLevel("chat_template_kwargs.reasoning_effort", kwargs["reasoning_effort"])
        }
        if includeReasoning == true, level == nil, toggle == nil { toggle = ("include_reasoning", true) }
        func thinks(_ effort: String) -> Bool { !["none", "minimal"].contains(effort) }
        if let level, let toggle, thinks(level.value) != toggle.value {
            throw ServingFailure("\(toggle.field) conflicts with \(level.field)")
        }
        let effort =
            level?.value
            ?? toggle.map { $0.value ? (thinks(defaults.reasoningEffort) ? defaults.reasoningEffort : "medium") : "none" }
            ?? defaults.reasoningEffort
        if let user = body["user"], !(user is String) { throw ServingFailure("user must be a string") }
        var includeUsage = false
        if let raw = body["stream_options"] {
            guard let options = raw as? [String: Any] else { throw ServingFailure("stream_options must be an object") }
            includeUsage = try boolean(options["include_usage"], default: false)
        }
        guard let original = body["messages"] as? [[String: Any]], !original.isEmpty, original.count <= 4096 else {
            throw ServingFailure("messages must be a nonempty array of at most 4096 messages")
        }
        // In-chat commands: the last message may be one. Earlier ones, and Origami's replies to them, are
        // dropped, so the model never sees them and saved prefixes keep matching.
        func command(_ message: [String: Any]) -> ChatCommand? {
            defaults.commands && message["role"] as? String == "user" ? ChatCommand.parse(ChatCommand.text(message["content"])) : nil
        }
        var messages: [[String: Any]] = []
        var dropReply = false
        for (index, message) in original.enumerated() {
            if dropReply, message["role"] as? String == "assistant", ChatCommand.text(message["content"]).hasPrefix(ChatCommand.marker) {
                dropReply = false
                continue
            }
            dropReply = index < original.count - 1 && command(message) != nil
            if !dropReply { messages.append(message) }
        }
        var pending = Set<String>()
        var seenCalls = Set<String>()
        var images = 0
        for message in messages {
            guard let role = message["role"] as? String, ["system", "developer", "user", "assistant", "tool"].contains(role) else {
                throw ServingFailure("Unsupported message role")
            }
            if role != "tool" && !pending.isEmpty { throw ServingFailure("Supply outstanding tool results before another message") }
            if let name = message["name"] {
                guard let value = name as? String, !value.isEmpty, value.utf8.count <= 128 else {
                    throw ServingFailure("Invalid message name")
                }
            }
            for key in ["reasoning", "reasoning_content"] {
                if let value = message[key] {
                    guard role == "assistant", value is String || value is NSNull else {
                        throw ServingFailure("Reasoning history must be assistant text")
                    }
                }
            }
            if role != "assistant" {
                guard let content = message["content"], !(content is NSNull) else { throw ServingFailure("Message content is required") }
            }
            if role != "tool" && message["tool_call_id"] != nil { throw ServingFailure("tool_call_id belongs only to tool results") }
            if let content = message["content"], !(content is NSNull), !(content is String) {
                guard let parts = content as? [[String: Any]] else { throw ServingFailure("Invalid message content") }
                for part in parts {
                    if part["type"] as? String == "image_url" {
                        guard role == "user" else { throw ServingFailure("Images belong in user messages") }
                        guard let url = ((part["image_url"] as? [String: Any])?["url"] ?? part["image_url"]) as? String,
                            ["data:", "http://", "https://"].contains(where: url.prefix(8).lowercased().hasPrefix)
                        else {
                            throw ServingFailure("Send images as base64 data: URLs or http(s) links")
                        }
                        images += 1
                        continue
                    }
                    guard ["text", "input_text"].contains(part["type"] as? String ?? "text"), part["text"] is String else {
                        throw ServingFailure("Origami reads text and images; remove files or audio from the message")
                    }
                }
            }
            if let raw = message["tool_calls"] {
                guard role == "assistant", let calls = raw as? [[String: Any]], calls.count <= 64 else {
                    throw ServingFailure("Invalid assistant tool_calls")
                }
                for call in calls {
                    if let arguments = (call["function"] as? [String: Any])?["arguments"] as? String { try checkNesting(arguments.utf8) }
                    guard (call["type"] as? String ?? "function") == "function", let id = call["id"] as? String, !id.isEmpty,
                        seenCalls.insert(id).inserted, let function = call["function"] as? [String: Any],
                        let name = function["name"] as? String, !name.isEmpty,
                        let arguments = function["arguments"] as? String,
                        case .object? = JSONValue.parse(arguments)
                    else { throw ServingFailure("Tool calls require unique IDs and JSON object argument strings") }
                    pending.insert(id)
                }
            }
            if role == "tool" {
                guard let id = message["tool_call_id"] as? String, pending.remove(id) != nil else {
                    throw ServingFailure("Tool result must reference an outstanding call exactly once")
                }
            }
        }
        guard pending.isEmpty else { throw ServingFailure("Missing tool results") }
        // Images are fetched before the prompt's length is known, so their number is bounded here: each takes at least
        // 64 tokens (the smallest image). The 32 a request may decode count later, once restored state is known.
        let imageLimit = (contextLimit - 1) / (PromptImage.minimumPixels / (PromptImage.factor * PromptImage.factor))
        guard images <= imageLimit else {
            throw ServingFailure(
                "This request has \(images) images; at most \(imageLimit) fit (each takes at least 64 of the \(contextLimit)-token context)"
            )
        }
        var rawTools: [[String: Any]] = []
        if let raw = body["tools"] {
            guard let values = raw as? [[String: Any]], values.count <= 512 else {
                throw ServingFailure(
                    "tools must be an array of at most 512 tools" + ((raw as? [Any]).map { " (this request has \($0.count))" } ?? ""))
            }
            rawTools = values
        }
        var definitions: [ToolDefinition] = []
        var names = Set<String>()
        for (index, tool) in rawTools.enumerated() {
            // Other keys (`strict`, a client's own) are ignored: the schema is shown to the model, not enforced.
            guard tool["type"] as? String == "function", let function = tool["function"] as? [String: Any],
                function["description"] == nil || function["description"] is String,
                let name = function["name"] as? String, !name.isEmpty, name.utf8.count <= 128,
                !name.contains(where: { $0.isWhitespace || "<>=/".contains($0) }), names.insert(name).inserted
            else {
                let name = (tool["function"] as? [String: Any])?["name"] as? String
                throw ServingFailure(
                    "Invalid or repeated tool function"
                        + (name.map {
                            " \"\($0.prefix(128))\" (tools \(index + 1)): a function names a declared tool once, in up to 128 characters without spaces or <>=/"
                        }
                            ?? " (tools \(index + 1)): each tool needs type function and a function name"))
            }
            let rawParameters = function["parameters"] ?? ["type": "object", "properties": [:]] as [String: Any]
            guard let parameters = rawParameters as? [String: Any] else {
                throw ServingFailure("Tool parameters must be a JSON Schema object")
            }
            rawTools[index] = [
                "type": "function",
                "function": ["name": name, "description": function["description"] as? String ?? "", "parameters": parameters],
            ]
            definitions.append(
                ToolDefinition(name: name, description: function["description"] as? String ?? "", parameters: JSONValue.from(parameters)))
        }
        let choice: ServingToolChoice
        if let raw = body["tool_choice"] {
            if let label = raw as? String {
                switch label {
                case "auto": choice = .auto
                case "none": choice = .none
                case "required": choice = .required
                default: throw ServingFailure("Unsupported tool_choice")
                }
            } else if let object = raw as? [String: Any], Set(object.keys).isSubset(of: ["type", "function"]),
                object["type"] as? String == "function",
                let function = object["function"] as? [String: Any], Set(function.keys) == ["name"], let name = function["name"] as? String,
                names.contains(name)
            {
                choice = .named(name)
            } else {
                throw ServingFailure("Invalid named tool_choice")
            }
        } else {
            choice = .auto
        }
        if choice == .required && definitions.isEmpty { throw ServingFailure("Required tool_choice needs declared tools") }
        if choice == .none {
            rawTools = []
            definitions = []
        }
        // A named choice keeps every tool in the prompt (so the prompt's start does not change) and reads only its call.
        if case .named(let name) = choice { definitions = definitions.filter { $0.name == name } }
        let thinking = !["none", "minimal"].contains(effort)
        // The server's values for this mode, unless the request sends its own (and `sampling_override` is off).
        var sampling = defaults.sampling.mode(thinking: thinking, tools: !definitions.isEmpty)
        if !defaults.samplingOverride || body["origami_tune"] != nil {
            sampling.temperature = Float(try number("temperature", default: Double(sampling.temperature), range: 0...100))
            sampling.topP = Float(try number("top_p", default: Double(sampling.topP), range: 0...1))
            // top_p 0 keeps only the most likely token: greedy.
            if sampling.topP == 0 { (sampling.temperature, sampling.topP) = (0, 1) }
            sampling.topK = try integer("top_k", default: sampling.topK, range: 0...248_320)
            sampling.minP = Float(try number("min_p", default: Double(sampling.minP), range: 0...1))
            sampling.presencePenalty = Float(try number("presence_penalty", default: Double(sampling.presencePenalty), range: -2...2))
            sampling.frequencyPenalty = Float(try number("frequency_penalty", default: 0, range: -2...2))
        }
        let decoder = JSONDecoder()
        let tuning = try tune(body["origami_tune"], quality: named ?? defaults.quality)
        var resolvedQuality = named ?? defaults.quality
        if tuning?.delayedExpertsOff == true {
            resolvedQuality.delayedExperts = nil
        } else if let policy = tuning?.delayedExperts {
            resolvedQuality.delayedExperts = policy
        }
        if tuning?.qwenPairProbe == true, (teacher?.count ?? 0) < 2 {
            throw ServingFailure("qwen_pair_probe requires at least two origami_teacher_tokens")
        }
        var request = try Self(
            model: requestedModel,
            messages: decoder.decode([[String: TemplateValue]].self, from: JSONSerialization.data(withJSONObject: messages)),
            tools: decoder.decode([[String: TemplateValue]].self, from: JSONSerialization.data(withJSONObject: rawTools)),
            definitions: definitions, sampling: sampling, seed: seed, maximumTokens: completionTokens, contextLimit: contextLimit,
            explicitOutputBudget: budget != nil && budget?.value != -1,
            reasoningEffort: effort, stopSequences: stops.filter { !$0.isEmpty }, stream: boolean(body["stream"], default: false),
            includeUsage: includeUsage,
            parallelToolCalls: boolean(body["parallel_tool_calls"], default: true), toolChoice: choice, quality: resolvedQuality,
            teacherTokens: teacher, returnTokens: try boolean(body["origami_return_tokens"], default: false),
            agentTurn: messages.last?["role"] as? String == "tool", excludeReasoning: excludeReasoning,
            thinkingBudget: thinkingBudget?.value ?? defaults.maxThinkingTokens,
            noStore: teacher != nil || body["prompt_cache_policy"] as? String == "no_store",  // a teacher-forced reply is not the model's
            command: messages.last.flatMap(command), explicitMode: named != nil,
            systemKey: systemKey(messages), statusMessages: try body["status_messages"].map { try boolean($0, default: false) },
            returnProgress: try boolean(body["return_progress"], default: false), tune: tuning)
        request.responseFormat = responseFormat
        if body["origami_time_limit"] != nil {
            request.deadline = Date(timeIntervalSinceNow: try number("origami_time_limit", default: 0, range: 0...604_800))
        }
        return request
    }

    /// Most tuner switches remain exact-only. The explicitly approximate local
    /// experiment can be selected only on an already non-exact quality mode.
    private static func tune(_ raw: Any?, quality: QualityDial) throws -> TuneOverrides? {
        guard let raw else { return nil }
        let overrides = try TuneOverrides(json: raw)
        // An empty marker pauses background copying for a local quality comparison; no timing switches change.
        if quality.isExact, overrides.delayedExperts != nil {
            throw ServingFailure("delayed_experts requires dynamic, fast/faster or fastest mode")
        }
        // Prefill remains exact in approximate decode modes. A bounded prefill scope also forces
        // isolated prompt state, so qualification requests cannot reuse another mode's frontier.
        let approximateKeys: Set<String> = ["delayed_experts", "family_early_delayed_reads", "qwen_prefill_scope"]
        guard quality.isExact || Set(overrides.wire.keys).isSubset(of: approximateKeys) else {
            throw ServingFailure("These origami_tune switches need origami_mode \"exact\"")
        }
        return overrides
    }

    /// SHA-256 of the leading system and developer messages' text (later ones are user turns), nil without one.
    private static func systemKey(_ messages: [[String: Any]]) -> String? {
        let system = messages.prefix { ["system", "developer"].contains($0["role"] as? String ?? "") }.map {
            ChatCommand.text($0["content"])
        }
        guard !system.isEmpty else { return nil }
        return SHA256.hash(data: Data(system.joined(separator: "\n\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Per-request cancellation flag and progress, shared (under its lock) by the HTTP connection, the keepalive timer
/// and the backend. Closing the connection cancels; the backend checks at bounded points and throws.
public final class GenerationControl: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    /// Where a request is, for /v1/status: the backend reports, the server reads.
    public struct Progress: Sendable {
        public var phase = "queued"
        /// Current bounded layer/subphase label, not an exclusive timing attribution.
        public var activity: String?
        public var promptTokens = 0, completedPromptTokens = 0
        /// Prompt tokens restored from saved state, once known.
        public var restoredTokens: Int?
        /// Real sampled output tokens; nil means this backend has not reported a count.
        public var generatedTokens: Int?
        public var firstGeneratedAt: Date?
        public var stageSeconds: [String: Double] = [:]
        public var phaseCounters: [String: Double] = [:]
        public var attemptTokens = 0
        public var preparationStageSeconds: [String: Double] = [:]
        public var attempts: [Attempt] = []
        public var omittedAttempts = 0
    }
    /// Work for one execution attempt, including replayed tokens. These counters must not be confused with
    /// unique request output, or summed with the last-attempt compatibility fields on the request row.
    public struct Attempt: Sendable {
        public let number, replayedTokens, generatedTokens: Int
        public let outcome: String
        public let seconds: Double
        public let stageSeconds, phaseCounters: [String: Double]
        public var wire: [String: Any] {
            [
                "attempt": number, "replay_tokens_at_start": replayedTokens, "attempt_output_tokens": generatedTokens,
                "outcome": outcome, "seconds": seconds, "stage_seconds": stageSeconds, "phase_counters": phaseCounters,
            ]
        }
    }
    private var current = Progress()
    public init() {}
    public var progress: Progress {
        lock.lock()
        defer { lock.unlock() }
        return current
    }
    public func report(_ change: (inout Progress) -> Void) {
        lock.lock()
        change(&current)
        lock.unlock()
    }
    public func reportGeneratedTokens(_ count: Int) {
        report {
            $0.attemptTokens = max(0, count)
            $0.generatedTokens = max($0.generatedTokens ?? 0, count)
            if count > 0 && $0.firstGeneratedAt == nil { $0.firstGeneratedAt = Date() }
        }
    }
    /// Each backend wraps its complete retry attempt, including setup that can itself fail. Preparation before
    /// the retry loop is stored once; the per-attempt stages exclude it. Histories are bounded independently of
    /// a backend's retry policy (GPURetry currently has at most three attempts).
    public func withAttempt<T>(
        number: Int, replayedTokens: Int, preparationSeconds: [String: Double] = [:],
        _ body: () throws -> T
    ) throws -> T {
        let started = ProcessInfo.processInfo.systemUptime
        report {
            $0.attemptTokens = 0
            $0.stageSeconds = [:]
            $0.phaseCounters = [:]
            $0.preparationStageSeconds = preparationSeconds
        }
        var outcome = "completed"
        defer {
            let elapsed = max(0, ProcessInfo.processInfo.systemUptime - started)
            report {
                let stages = $0.stageSeconds.filter { preparationSeconds[$0.key] == nil }
                $0.attempts.append(
                    Attempt(
                        number: number, replayedTokens: replayedTokens, generatedTokens: $0.attemptTokens,
                        outcome: outcome, seconds: elapsed, stageSeconds: stages, phaseCounters: $0.phaseCounters))
                if $0.attempts.count > 8 {
                    $0.attempts.removeFirst()
                    $0.omittedAttempts += 1
                }
            }
        }
        do { return try body() } catch {
            outcome = isCancelled ? "cancelled" : (GPURetry.isInterruption(error) ? "gpu_interrupted" : "failed")
            throw error
        }
    }
    public func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
    public func check() throws { if isCancelled { throw IOOperationCancelled() } }
}

/// What a finished generation reports for usage, finish reason and the quality dial that produced it.
public struct GenerationStatistics: Sendable {
    public let promptTokens, completionTokens, cachedTokens: Int
    public let finishReason: String
    /// The quality dial that produced the response.
    public let quality: QualityDial
    /// The reply's token IDs (`origami_return_tokens`) and its teacher-forced agreement (`origami_teacher_tokens`).
    public var outputTokens: [Int]?
    public var teacher: TeacherAgreement?
    /// Tokens inside the think block, and why the server closed it ("budget", "answer_room"), if it did.
    public var reasoningTokens: Int?
    public var reasoningClosedBy: String?
    /// The request's `stop` string that ended the reply, if one did (Anthropic's `stop_sequence`).
    public var stopSequence: String?
    /// The first reply token a non-exact dial decoded exact because a tool call opened before it (`QualityDial.opensToolCall`).
    public var exactFrom: Int?
    /// How many times `GPURetry` ran the request again after macOS interrupted it, when it did.
    public var gpuRetries: Int?
    /// Request-boundary wall times, never GPU-only rates. Retry results describe
    /// the successful attempt; no per-token polling or extra synchronization.
    /// Image expansion, transport/admission and deferred cleanup are outside these
    /// stages, so their sum need not equal the complete HTTP request duration.
    public var stageSeconds: [String: Double] = [:]
    /// Completed logical I/O/scheduling counters sampled at phase boundaries.
    public var phaseCounters: [String: Double] = [:]
    public init(
        promptTokens: Int, completionTokens: Int, cachedTokens: Int = 0, finishReason: String = "stop", quality: QualityDial = .exact
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.finishReason = finishReason
        self.quality = quality
    }
}

/// The backend's state for /health, /v1/status, /v1/models and the response headers.
public struct ServingStatus: Sendable {
    public let version: String
    public let contextTier: Int
    /// False while the model loads (or after it failed to load).
    public var ready: Bool
    /// Why the model could not load; the server keeps answering /health with it.
    public var failure: String? = nil
    /// Something a user may want to know, e.g. a pack drive that is unplugged.
    public var degraded: String? = nil
    /// The model's own drive is gone (`ModelDrive`): requests fail until it is back, then reads resume by themselves.
    package var driveAway = false
    /// The configuration allows per-request non-exact quality levels (`origami_mode` or a `-<level>` model id).
    public var fastOptIn: Bool = false
    /// The served model's family, whose offered levels `quality_levels` lists (`LocalHTTPServer` sets it).
    public var family = ModelFamily.qwen
    /// The checkpoint has a vision tower: prompts may hold images (Ollama's `vision` capability).
    public var vision = false
    public init(version: String, contextTier: Int = 8192, ready: Bool) {
        self.version = version
        self.contextTier = contextTier
        self.ready = ready
    }
    public var wire: [String: Any] {
        var value: [String: Any] = [
            "engine": "origami", "version": version, "context": contextTier,
            "status": ready ? "ready" : (failure == nil ? "loading" : "failed"),
        ]
        if let failure { value["error"] = failure }
        if let degraded { value["note"] = degraded }
        if ready, driveAway, let degraded {
            value["status"] = "degraded"
            value["degraded"] = degraded
        }
        value["fast_mode"] = fastOptIn
        value["quality_levels"] = fastOptIn ? QualityDial.offered(family) : ["exact"]
        return value
    }
}

/// prepare must tokenize and complete ledger admission before response headers.
/// run emits only complete UTF-8 strings, counts reasoning in completionTokens,
/// checks cancellation at bounded execution boundaries, and stops when emit
/// returns false (for example a requested single tool call has completed).
public protocol PreparedGeneration: Sendable {
    func run(emit: @escaping @Sendable (String) throws -> Bool) throws -> GenerationStatistics
}
public protocol ServingBackend: Sendable {
    func status() -> ServingStatus
    func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration
}
/// A backend with work left once its server has stopped and every request has ended (`LocalHTTPServer.stopAndJoin`):
/// `origami stop`, an unload and a model switch all stop the serving process that way.
public protocol ServingBackendStopping: ServingBackend {
    func serverStopped()
}
/// A backend that can say how many prompt tokens a request renders to (`/v1/messages/count_tokens`): its tokenizer
/// and chat template, as the request would be read (`ServingPrompt.encode`).
public protocol ServingPromptCounting: ServingBackend {
    func promptTokens(_ request: ServingRequest) throws -> Int
}

/// Per-response parser. All state is accessed on one generation worker. The
/// Sendable wrapper permits the synchronous injected emitter closure to capture
/// it; a backend must never invoke emit concurrently.
public final class ServingStream: @unchecked Sendable {
    private let request: ServingRequest
    private let thinker: ServingThinkSplitter?
    /// Nil when the request declares no tools: a `<tool_call>` then is the model's text (it can echo an earlier
    /// call's form from the history), as other servers return it.
    private let tools: ToolCallSplitter?
    private let maximumBytes: Int
    private var receivedBytes = 0
    private var pendingCall: String?
    /// The pending call's start and argument deltas, held back until its block parses: a call the model wrote badly then
    /// streams as text only, never as a call the client cannot take back. Text before it streams as it comes.
    private var heldCall: [[String: Any]] = []
    public private(set) var text = "", reasoning = ""
    public private(set) var calls: [ParsedToolCall] = []
    public var finishedSingleCall: Bool { !request.parallelToolCalls && !calls.isEmpty }
    /// Set when a finished turn had neither text nor a call: `reasoning` when its reasoning became the answer (a think
    /// block the turn never closed, an empty tool-call block), `nothing` when there was none either.
    public private(set) var emptyAnswer: String?

    /// `family` picks the tool-call syntax its chat template teaches (`ToolCallGrammar`).
    public init(request: ServingRequest, family: ModelFamily = .qwen, maximumBytes: Int = 8 * 1_024 * 1_024) {
        precondition(maximumBytes >= 0)
        self.request = request
        self.maximumBytes = maximumBytes
        thinker = ["none", "minimal"].contains(request.reasoningEffort) ? nil : ServingThinkSplitter()
        tools = request.definitions.isEmpty ? nil : ToolCallSplitter(tools: request.definitions.map(\.schema), grammar: .of(family))
    }
    public func push(_ piece: String) throws -> [[String: Any]] {
        guard piece.utf8.count <= maximumBytes - receivedBytes else {
            throw ServingFailure("Generated output exceeded its byte budget", status: 507, type: "resource_error")
        }
        receivedBytes += piece.utf8.count
        let (thought, answer) = thinker?.push(piece) ?? ("", piece)
        var deltas: [[String: Any]] = []
        if !thought.isEmpty {
            reasoning += thought
            if !request.excludeReasoning { deltas.append(["reasoning_content": thought]) }
        }
        deltas += try consume(tools?.push(answer) ?? [.text(answer)])
        return deltas
    }
    public func finish(reason: String) throws -> (deltas: [[String: Any]], reason: String) {
        var deltas: [[String: Any]] = []
        if let thinker {
            let (thought, answer) = thinker.flush()
            if !thought.isEmpty {
                reasoning += thought
                if !request.excludeReasoning { deltas.append(["reasoning_content": thought]) }
            }
            deltas += try consume(tools?.push(answer) ?? [.text(answer)])
        }
        deltas += try consume(tools?.flush() ?? [])
        // Decode allows only the call, so what is left is a call the parser could not read (it is the reply's text).
        if calls.isEmpty && reason != "length" {
            let wanted: String?
            switch request.toolChoice {
            case .required: wanted = "one of the tools"
            case .named(let name): wanted = "the tool \(name)"
            default: wanted = nil
            }
            if let wanted {
                throw ServingFailure(
                    "tool_choice asks for a call to \(wanted), and the model's reply holds no valid call (its text: \(text.prefix(200)))",
                    status: 422, type: "invalid_request_error", code: "tool_choice_unmet")
            }
        }
        // A turn that ended with neither text nor a call answers with its reasoning rather than nothing (a cut reply
        // keeps its usual form: the think closer answers before the limit).
        if text.isEmpty && calls.isEmpty && reason != "length" {
            if reasoning.contains(where: { !$0.isWhitespace }) {
                text = reasoning
                deltas.append(["content": reasoning])
                emptyAnswer = "reasoning"
            } else {
                emptyAnswer = "nothing"
            }
        }
        return (deltas, calls.isEmpty ? reason : "tool_calls")
    }
    private func consume(_ events: [ToolStreamEvent]) throws -> [[String: Any]] {
        var deltas: [[String: Any]] = []
        for event in events {
            if finishedSingleCall { break }
            switch event {
            case .text(let value):
                if !value.isEmpty {
                    text += value
                    deltas.append(["content": value])
                }
            case .toolInputStart(let id, let name):
                pendingCall = id
                heldCall = [
                    [
                        "tool_calls": [
                            ["index": calls.count, "id": id, "type": "function", "function": ["name": name, "arguments": ""]]
                        ]
                    ]
                ]
            case .toolInputDelta(let id, let value):
                guard pendingCall == id else { throw ServingFailure("Tool parser lost call identity", status: 500) }
                if !value.isEmpty { heldCall.append(["tool_calls": [["index": calls.count, "function": ["arguments": value]]]]) }
            case .toolInputEnd: break
            case .toolCall(let call):
                calls.append(call)
                pendingCall = nil
                deltas += heldCall
                heldCall = []
            case .malformed(let raw):
                // A call the grammar cannot read (a repeated parameter, a block left open) is the model's text, and the
                // reply finishes as usual. Nothing of it was streamed as a call.
                pendingCall = nil
                heldCall = []
                if !raw.isEmpty {
                    text += raw
                    deltas.append(["content": raw])
                }
            }
        }
        return deltas
    }
    public var message: [String: Any] {
        var result: [String: Any] = ["role": "assistant", "content": text]
        if !reasoning.isEmpty && !request.excludeReasoning { result["reasoning_content"] = reasoning }
        if !calls.isEmpty {
            result["tool_calls"] = calls.map {
                ["id": $0.id, "type": "function", "function": ["name": $0.name, "arguments": $0.inputJSON]] as [String: Any]
            }
            if text.isEmpty { result["content"] = NSNull() }
        }
        return result
    }
}

/// JSON and SSE framing shared by every wire: sorted-key JSON (refusing values JSON cannot hold instead of
/// aborting), `data:` frames, `[DONE]`, the keepalive comment, and OpenAI-shaped usage.
public enum ServingWire {
    public static func json(_ object: [String: Any]) throws -> Data {
        // JSONSerialization aborts the process on a non-JSON value; refuse it as an error instead.
        guard JSONSerialization.isValidJSONObject(object) else { throw EngineError("Response holds a value JSON cannot represent") }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .fragmentsAllowed])
    }
    public static func sse(_ object: [String: Any]) throws -> Data {
        var value = Data("data: ".utf8)
        value.append(try json(object))
        value.append(Data("\n\n".utf8))
        return value
    }
    public static let done = Data("data: [DONE]\n\n".utf8)
    /// An SSE comment: keeps idle timers from firing, ignored by every client.
    public static let keepAlive = Data(": keepalive\n\n".utf8)
    public static func usage(_ stats: GenerationStatistics) throws -> [String: Any] {
        guard stats.promptTokens >= 0, stats.completionTokens >= 0, stats.cachedTokens >= 0,
            stats.cachedTokens <= stats.promptTokens, stats.promptTokens <= Int.max - stats.completionTokens
        else { throw EngineError("Invalid backend usage") }
        var usage: [String: Any] = [
            "prompt_tokens": stats.promptTokens, "completion_tokens": stats.completionTokens,
            "total_tokens": stats.promptTokens + stats.completionTokens,
            "prompt_tokens_details": ["cached_tokens": stats.cachedTokens],
        ]
        // Exact usage keeps its exact-only bytes; any other level says which (X-Origami-Mode reports every level).
        if !stats.quality.isExact { usage["origami_mode"] = stats.quality.name }
        if let tokens = stats.outputTokens { usage["origami_tokens"] = tokens }
        // Explicit diagnostic requests reuse already collected boundary data;
        // ordinary traffic adds neither measurement payload nor hot-path work.
        if stats.outputTokens != nil || stats.teacher != nil {
            if !stats.stageSeconds.isEmpty { usage["origami_stage_seconds"] = stats.stageSeconds }
            if !stats.phaseCounters.isEmpty { usage["origami_phase_counters"] = stats.phaseCounters }
        }
        if let teacher = stats.teacher {
            var report: [String: Any] = [
                "positions": teacher.positions, "agreed": teacher.agreed,
                "first_difference": teacher.firstDifference.map { $0 as Any } ?? NSNull(),
            ]
            if let probabilities = teacher.tokenLogProbabilities {
                report["token_log_probabilities"] = probabilities.map { $0.map { $0 as Any } ?? NSNull() }
            }
            usage["origami_teacher"] = report
        }
        return usage
    }
}
