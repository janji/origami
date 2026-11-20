import Foundation

/// JSON mode (`response_format` `json_object` or `json_schema`, Responses' `text.format`, Ollama's `format`): the
/// grammar a reply's answer must follow. `object` takes any JSON object; `init(schema:)` compiles the supported part
/// of JSON Schema and refuses the rest with a 400 naming the keyword. A node no finite JSON value satisfies is pruned
/// (an optional property whose schema has none is forbidden, an `anyOf` branch is left out), so every state the
/// acceptor reaches can still be completed.
public struct JSONGrammar: Sendable, Equatable {
    package enum Node: Sendable, Equatable {
        /// `keys` are the names as JSON writes them (escaped, without quotes); `additional` nil allows no other key.
        case object(keys: [[UInt8]], values: [Int], required: UInt64, additional: Int?)
        case array(item: Int, minimum: Int, maximum: Int)
        case string
        case number(integer: Bool)
        /// `enum`, `const`, booleans and null: the values' compact JSON text.
        case literal([[UInt8]])
        case union([Int])
    }
    package private(set) var nodes: [Node]
    package private(set) var root: Int
    /// The non-union nodes each node can be, for the acceptor.
    package private(set) var choices: [[Int]] = []

    /// Nodes every grammar starts with: any value, any object, any array, string, number, true/false/null.
    fileprivate static let base: [Node] = [
        .union([1, 2, 3, 4, 5]), .object(keys: [], values: [], required: 0, additional: 0), .array(item: 0, minimum: 0, maximum: .max),
        .string, .number(integer: false), .literal(["true", "false", "null"].map { Array($0.utf8) }),
    ]
    /// `json_object`: one JSON object, anything inside.
    public static let object: JSONGrammar = try! JSONGrammar(nodes: base, root: 1)
    /// No JSON value at all: the reply must open a tool call (`tool_choice` required or named).
    package static let toolCallOnly: JSONGrammar = {
        var grammar = object
        grammar.nodes.append(.union([]))
        grammar.choices.append([])
        grammar.root = grammar.nodes.count - 1
        return grammar
    }()

    private init(nodes: [Node], root: Int) throws {
        self.nodes = nodes
        self.root = root
        try prune()
    }

    /// A `json_schema` schema (the decoded JSON). Supported: `type` (one or a list), `properties`, `required`,
    /// `additionalProperties`, `items`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `$ref` to the document
    /// (`#`, `#/$defs/…`), and annotations (`title`, `description`, `default`, `examples`, `format`, …), which are not
    /// enforced.
    public init(schema: Any) throws {
        var compiler = SchemaCompiler(document: schema)
        let root = try compiler.compile(schema, path: "")
        try self.init(nodes: compiler.nodes, root: root)
    }

    /// Drops what has no finite value (fixpoint from the leaves), then flattens unions into `choices`.
    private mutating func prune() throws {
        var finite = [Bool](repeating: false, count: nodes.count)
        var changed = true
        while changed {
            changed = false
            for (index, node) in nodes.enumerated() where !finite[index] {
                let value: Bool
                switch node {
                case .object(_, let values, let required, _):
                    value = values.indices.allSatisfy { required & (1 << UInt64($0)) == 0 || finite[values[$0]] }
                case .array(let item, let minimum, _): value = minimum == 0 || finite[item]
                case .string, .number: value = true
                case .literal(let texts): value = !texts.isEmpty
                case .union(let options): value = options.contains { finite[$0] }
                }
                if value {
                    finite[index] = true
                    changed = true
                }
            }
        }
        guard finite[root] else { throw ServingFailure("response_format schema allows no finite JSON value") }
        // An optional property with no value stays declared, pointing at `never`, so additional keys cannot admit it.
        let never = nodes.count
        nodes.append(.union([]))
        for (index, node) in nodes.enumerated() {
            switch node {
            case .object(let keys, let values, let required, let additional):
                nodes[index] = .object(
                    keys: keys, values: values.map { finite[$0] ? $0 : never }, required: required,
                    additional: additional.flatMap { finite[$0] ? $0 : nil })
            case .array(let item, let minimum, _) where !finite[item]: nodes[index] = .array(item: item, minimum: minimum, maximum: 0)
            case .union(let options): nodes[index] = .union(options.filter { finite[$0] })
            default: break
            }
        }
        choices = nodes.indices.map { start in
            var seen = Set<Int>()
            var found: [Int] = []
            var pending = [start]
            while let next = pending.popLast() {
                guard seen.insert(next).inserted else { continue }
                if case .union(let options) = nodes[next] { pending += options.reversed() } else { found.append(next) }
            }
            return found
        }
    }
}

/// Turns a JSON Schema into grammar nodes; `$ref`s are compiled once each, so a recursive schema is a cycle of nodes.
private struct SchemaCompiler {
    let document: Any
    var nodes = JSONGrammar.base
    var refs: [String: Int] = [:]
    static let annotations: Set<String> = [
        "title", "description", "default", "examples", "$schema", "$id", "$comment", "format", "deprecated", "readOnly", "writeOnly",
        "$defs", "definitions",
    ]
    static let structural: Set<String> = [
        "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const", "anyOf", "$ref",
    ]

    /// Whether `value` is of the JSON Schema `type` (an integer is a number without a fraction).
    static func value(_ value: JSONValue, has type: String) -> Bool {
        switch (value, type) {
        case (.string, "string"), (.bool, "boolean"), (.null, "null"), (.array, "array"), (.object, "object"), (.int, "number"),
            (.double, "number"), (.int, "integer"):
            return true
        case (.double(let number), "integer"): return number.rounded() == number
        default: return false
        }
    }
    func refuse(_ what: String, _ path: String) -> ServingFailure {
        ServingFailure("response_format schema\(path.isEmpty ? "" : " at \(path)"): \(what)")
    }
    mutating func add(_ node: JSONGrammar.Node) -> Int {
        nodes.append(node)
        return nodes.count - 1
    }
    static func boolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }
    func count(_ value: Any?, _ name: String, _ path: String) throws -> Int? {
        guard let value else { return nil }
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let count = Int(exactly: number), count >= 0
        else { throw refuse("\(name) must be a whole number of at least 0", path) }
        return count
    }

    mutating func compile(_ value: Any, path: String) throws -> Int {
        if let flag = Self.boolean(value) { return flag ? 0 : add(.union([])) }
        guard let schema = value as? [String: Any] else { throw refuse("a schema must be an object or a boolean", path) }
        if let key = schema.keys.sorted().first(where: { !Self.annotations.contains($0) && !Self.structural.contains($0) }) {
            throw refuse("the keyword \"\(key)\" is not supported", path)
        }
        let present = Set(schema.keys).intersection(Self.structural)
        if let reference = schema["$ref"] {
            guard present == ["$ref"] else { throw refuse("$ref beside other keywords is not supported", path) }
            guard let pointer = reference as? String else { throw refuse("$ref must be text", path) }
            if let node = refs[pointer] { return node }
            let target = try resolve(pointer, path)
            let node = add(.union([]))
            refs[pointer] = node
            nodes[node] = .union([try compile(target, path: pointer)])
            return node
        }
        if let options = schema["anyOf"] {
            guard present == ["anyOf"] else { throw refuse("anyOf beside other keywords is not supported", path) }
            guard let list = options as? [Any], !list.isEmpty else { throw refuse("anyOf must be a non-empty list", path) }
            var alternatives: [Int] = []
            for (index, option) in list.enumerated() { alternatives.append(try compile(option, path: path + "/anyOf/\(index)")) }
            return add(.union(alternatives))
        }
        let types: [String]
        switch schema["type"] {
        case nil:
            if !present.isDisjoint(with: ["properties", "required", "additionalProperties"]) {
                types = ["object"]
            } else if !present.isDisjoint(with: ["items", "minItems", "maxItems"]) {
                types = ["array"]
            } else {
                types = []
            }
        case let name as String: types = [name]
        case let list as [Any]:
            let names = list.compactMap { $0 as? String }
            guard !names.isEmpty, names.count == list.count, Set(names).count == names.count else {
                throw refuse("type must be a type name or a list of distinct names", path)
            }
            types = names
        default: throw refuse("type must be a type name or a list of names", path)
        }
        if schema["const"] != nil || schema["enum"] != nil {
            guard schema["const"] == nil || schema["enum"] == nil else { throw refuse("enum beside const is not supported", path) }
            // `type` narrows the values; any other keyword beside them is a second constraint the literal would skip.
            if let other = present.subtracting(["enum", "const", "type"]).sorted().first {
                throw refuse("enum or const beside \(other) is not supported", path)
            }
            let values = schema["const"].map { [$0] } ?? (schema["enum"] as? [Any] ?? [])
            guard (1...64).contains(values.count) else { throw refuse("enum must list 1 to 64 values", path) }
            var texts: [[UInt8]] = []
            for value in values {
                let json = JSONValue.from(value)
                guard schema["type"] == nil || types.contains(where: { Self.value(json, has: $0) }) else { continue }
                let text = Array(json.jsonText.utf8)
                if !texts.contains(text) { texts.append(text) }
            }
            guard !texts.isEmpty else { throw refuse("no enum or const value has the type \(types.joined(separator: " or "))", path) }
            return add(.literal(texts))
        }
        if types.isEmpty { return 0 }
        var alternatives: [Int] = []
        for type in types { alternatives.append(try typed(type, schema, path)) }
        return alternatives.count == 1 ? alternatives[0] : add(.union(alternatives))
    }

    mutating func typed(_ type: String, _ schema: [String: Any], _ path: String) throws -> Int {
        switch type {
        case "object":
            guard let properties = (schema["properties"] ?? [String: Any]()) as? [String: Any] else {
                throw refuse("properties must be an object", path)
            }
            guard properties.count <= 64 else { throw refuse("more than 64 properties are not supported", path) }
            let names = properties.keys.sorted()
            guard let required = (schema["required"] ?? [String]()) as? [String] else {
                throw refuse("required must be a list of names", path)
            }
            var mask: UInt64 = 0
            for name in required {
                guard let index = names.firstIndex(of: name) else {
                    throw refuse("required names \"\(name)\", which properties does not define", path)
                }
                mask |= 1 << UInt64(index)
            }
            var additional: Int? = 0
            if let value = schema["additionalProperties"] {
                if let flag = Self.boolean(value) {
                    additional = flag ? 0 : nil
                } else {
                    additional = try compile(value, path: path + "/additionalProperties")
                }
            }
            var values: [Int] = []
            for name in names { values.append(try compile(properties[name]!, path: path + "/properties/" + name)) }
            let keys = names.map { Array(JSONValue.quote($0).utf8.dropFirst().dropLast()) }
            return add(.object(keys: keys, values: values, required: mask, additional: additional))
        case "array":
            let item: Int
            switch schema["items"] {
            case nil: item = 0
            case let value? where value is [String: Any] || Self.boolean(value) != nil: item = try compile(value, path: path + "/items")
            default: throw refuse("items must be one schema (a list of item schemas is not supported)", path)
            }
            let minimum = try count(schema["minItems"], "minItems", path) ?? 0
            let maximum = try count(schema["maxItems"], "maxItems", path) ?? .max
            guard minimum <= maximum else { throw refuse("minItems is larger than maxItems", path) }
            return add(.array(item: item, minimum: minimum, maximum: maximum))
        case "string": return 3
        case "number": return 4
        case "integer": return add(.number(integer: true))
        case "boolean": return add(.literal([Array("true".utf8), Array("false".utf8)]))
        case "null": return add(.literal([Array("null".utf8)]))
        default: throw refuse("the type \"\(type)\" is unknown", path)
        }
    }

    /// A JSON pointer into the schema document: `#`, `#/$defs/name`, `#/definitions/name`, any local path.
    func resolve(_ pointer: String, _ path: String) throws -> Any {
        guard pointer.hasPrefix("#") else { throw refuse("only $ref into this schema (#/$defs/…) is supported", path) }
        var value: Any = document
        for part in pointer.dropFirst().split(separator: "/", omittingEmptySubsequences: true) {
            let name = part.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
            if let object = value as? [String: Any], let next = object[name] {
                value = next
            } else if let list = value as? [Any], let index = Int(name), list.indices.contains(index) {
                value = list[index]
            } else {
                throw refuse("$ref \(pointer) points at nothing", path)
            }
        }
        return value
    }
}

/// What a constraint's acceptor follows: the grammar in flat tables, the tool-call opener a reply may start with
/// instead of JSON, and how many newlines may come before the JSON (the ones after a think block, which the reply's
/// think splitter drops).
package final class JSONRules: @unchecked Sendable {
    let grammar: JSONGrammar
    let opener: [UInt8]
    let leading: UInt8
    /// Per node: 0 object, 1 array, 2 string, 3 number, 4 integer, 5 literal, 6 union.
    let kind: [UInt8]
    let keys: [[[UInt8]]], values: [[Int32]], required: [UInt64], additional: [Int32], allKeys: [UInt64]
    /// Per object: the declared keys whose value nothing satisfies (`"x": false`); never taken, never additional.
    let forbidden: [UInt64]
    let item: [Int32], minimum: [Int], maximum: [Int]
    let literals: [[[UInt8]]]
    let choices: [[Int32]]
    static let maximumDepth: Int32 = 32
    /// Whitespace bytes in a row inside a container `depth` frames deep: a newline and four spaces an indent level, so
    /// pretty-printed JSON fits and a reply cannot fill itself with whitespace.
    static func maximumSpaces(depth: Int32) -> Int { 2 + 4 * Int(depth) }

    package init(grammar: JSONGrammar, opener: String? = nil, leadingNewlines: Int = 0) {
        self.grammar = grammar
        self.opener = opener.map { Array($0.utf8) } ?? []
        leading = UInt8(min(max(leadingNewlines, 0), 8))
        let count = grammar.nodes.count
        var kind = [UInt8](repeating: 6, count: count)
        var keys = [[[UInt8]]](repeating: [], count: count)
        var values = [[Int32]](repeating: [], count: count)
        var required = [UInt64](repeating: 0, count: count)
        var additional = [Int32](repeating: -1, count: count)
        var item = [Int32](repeating: 0, count: count)
        var minimum = [Int](repeating: 0, count: count)
        var maximum = [Int](repeating: 0, count: count)
        var literals = [[[UInt8]]](repeating: [], count: count)
        for (index, node) in grammar.nodes.enumerated() {
            switch node {
            case .object(let k, let v, let r, let a):
                kind[index] = 0
                keys[index] = k
                values[index] = v.map(Int32.init)
                required[index] = r
                additional[index] = a.map(Int32.init) ?? -1
            case .array(let i, let low, let high):
                kind[index] = 1
                item[index] = Int32(i)
                minimum[index] = low
                maximum[index] = high
            case .string: kind[index] = 2
            case .number(let integer): kind[index] = integer ? 4 : 3
            case .literal(let texts):
                kind[index] = 5
                literals[index] = texts
            case .union: kind[index] = 6
            }
        }
        self.kind = kind
        self.keys = keys
        self.values = values
        self.required = required
        self.additional = additional
        allKeys = keys.map { $0.count == 64 ? ~0 : (1 << UInt64($0.count)) - 1 }
        forbidden = values.map { list in
            list.indices.reduce(0) { mask, k in grammar.choices[Int(list[k])].isEmpty ? mask | 1 << UInt64(k) : mask }
        }
        self.item = item
        self.minimum = minimum
        self.maximum = maximum
        self.literals = literals
        choices = grammar.choices.map { $0.map(Int32.init) }
    }
}

/// An incremental JSON acceptor (a pushdown automaton) over the UTF-8 bytes of a reply. A byte-level token may end
/// inside a character or span several JSON tokens, so it is fed byte by byte; strings check UTF-8 (shortest form, no
/// surrogates), escapes and paired `\u` surrogates. `anyOf` branches that start alike run side by side (`alternatives`).
/// Value semantics and `Hashable`, so a state keys the mask cache. Bounds that keep a reply from running on inside
/// the grammar: 32 levels of nesting, numbers of about 32 characters, whitespace runs of a newline and an indent.
package struct JSONAcceptor: Hashable {
    enum Kind: UInt8 { case root, object, key, array, string, number, literal }
    struct Frame: Hashable {
        var kind: Kind
        var phase: UInt8 = 0
        var node: Int32
        /// Array: items; key, literal and number: bytes read; object: the key being read (-1: an additional one).
        var count: Int32 = 0
        /// Object: keys seen; key and literal: candidates still matching; number: 1 for an integer.
        var mask: UInt64 = 0
        /// Key: its object's keys seen.
        var seen: UInt64 = 0
        /// String and key: 0 plain, 1 after `\`, 2–5 `\u` digits, 6 `\` and 7 `u` of a low surrogate, 8–11 its digits.
        var escape: UInt8 = 0
        /// UTF-8 continuation bytes still due, and the range the next one must fall in.
        var due: UInt8 = 0
        var low: UInt8 = 0x80, high: UInt8 = 0xBF
        var hex: UInt16 = 0
    }
    final class Link: Hashable {
        let frame: Frame
        let below: Link?
        let digest: Int
        init(_ frame: Frame, _ below: Link?) {
            self.frame = frame
            self.below = below
            var hasher = Hasher()
            hasher.combine(frame)
            hasher.combine(below?.digest ?? 0)
            digest = hasher.finalize()
        }
        static func == (a: Link, b: Link) -> Bool { a === b || (a.digest == b.digest && a.frame == b.frame && a.below == b.below) }
        func hash(into hasher: inout Hasher) { hasher.combine(digest) }
    }
    struct Alternative: Hashable {
        var top: Frame
        var below: Link?
        var depth: Int32 = 1
        mutating func push(_ frame: Frame) {
            below = Link(top, below)
            top = frame
            depth += 1
        }
        mutating func pop() {
            top = below!.frame
            below = below!.below
            depth -= 1
        }
    }
    /// Usually one; more only while `anyOf` branches that begin alike are both alive. Empty: no JSON can follow.
    struct Alternatives: Hashable {
        var first: Alternative?
        var more: [Alternative] = []
        mutating func add(_ alternative: Alternative) {
            if first == nil {
                first = alternative
            } else if first != alternative, !more.contains(alternative) {
                more.append(alternative)
            }
        }
    }
    var alternatives: Alternatives
    var spaces: UInt8 = 0
    /// Bytes of the tool-call opener matched so far; -1 once the reply cannot be a tool call.
    var tool: Int16
    var started = false
    /// The reply opened a tool call: nothing is constrained from here.
    package private(set) var free = false

    package init(_ rules: JSONRules) {
        alternatives = Alternatives(first: Alternative(top: Frame(kind: .root, node: Int32(rules.grammar.root))))
        tool = rules.opener.isEmpty ? -1 : 0
    }

    /// Takes one byte of the reply; false when nothing valid can follow (the state is then of no use). `special`: the byte
    /// belongs to a control token, which only a tool-call opener may use.
    @inline(__always) package mutating func step(_ byte: UInt8, special: Bool = false, _ rules: JSONRules) -> Bool {
        if free { return true }
        let space = byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
        if !started {
            if space {
                guard byte == 0x0A, spaces < rules.leading else { return false }
                spaces += 1
                return true
            }
            started = true
            spaces = 0
        } else {
            spaces = space ? min(spaces, 254) + 1 : 0
        }
        if tool >= 0 {
            if Int(tool) < rules.opener.count, rules.opener[Int(tool)] == byte {
                tool += 1
                if Int(tool) == rules.opener.count {
                    free = true
                    alternatives = Alternatives()
                    return true
                }
            } else {
                tool = -1
            }
        }
        if special {
            alternatives = Alternatives()
        } else if let first = alternatives.first {
            var next = Alternatives()
            Self.step(first, byte, rules, spaces, &next)
            for other in alternatives.more { Self.step(other, byte, rules, spaces, &next) }
            alternatives = next
        }
        return alternatives.first != nil || tool > 0
    }

    /// Whether the reply may end here: one whole JSON value, nothing after it.
    package func complete(_ rules: JSONRules) -> Bool {
        guard !free, tool <= 0 else { return false }
        if let first = alternatives.first, Self.ends(first, rules) { return true }
        return alternatives.more.contains { Self.ends($0, rules) }
    }
    /// Whether a control token can still be taken (only as part of the tool-call opener).
    var opensTool: Bool { !free && tool >= 0 }

    private static func ends(_ alternative: Alternative, _ r: JSONRules) -> Bool {
        var a = alternative
        while true {
            switch a.top.kind {
            case .root: return a.top.phase == 1
            case .number: guard numberEnds(a.top) else { return false }
            case .literal: guard literalEnds(a.top, r) else { return false }
            default: return false
            }
            a.pop()
        }
    }

    private static func step(_ alternative: Alternative, _ byte: UInt8, _ r: JSONRules, _ spaces: UInt8, _ out: inout Alternatives) {
        var a = alternative
        let space = byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
        while true {
            let node = Int(a.top.node)
            switch a.top.kind {
            case .string:
                switch stringByte(&a.top, byte) {
                case 0: return
                case 1: out.add(a)
                default:
                    a.pop()
                    out.add(a)
                }
                return
            case .key:
                let keys = r.keys[node]
                let position = Int(a.top.count)
                let escaped = a.top.escape != 0
                switch stringByte(&a.top, byte) {
                case 0: return
                case 1:
                    var matching: UInt64 = 0
                    var rest = a.top.mask
                    while rest != 0 {
                        let k = rest.trailingZeroBitCount
                        rest &= rest - 1
                        if keys[k].count > position, keys[k][position] == byte { matching |= 1 << UInt64(k) }
                    }
                    if r.additional[node] < 0, matching & ~a.top.seen & ~r.forbidden[node] == 0 { return }
                    // An additional key is written without escapes once it leaves every declared key: `\u0078` or
                    // `\/` could otherwise spell a declared key and skip its schema.
                    if matching == 0, !keys.isEmpty, escaped || a.top.escape != 0 { return }
                    a.top.mask = matching
                    a.top.count += 1
                    out.add(a)
                default:
                    var property: Int32 = -1
                    var rest = a.top.mask
                    while rest != 0 {
                        let k = rest.trailingZeroBitCount
                        rest &= rest - 1
                        if keys[k].count == position { property = Int32(k) }
                    }
                    if property >= 0 {
                        if (a.top.seen | r.forbidden[node]) & (1 << UInt64(property)) != 0 { return }
                    } else if r.additional[node] < 0 {
                        return
                    }
                    a.pop()
                    a.top.phase = 2
                    a.top.count = property
                    if property >= 0 { a.top.mask |= 1 << UInt64(property) }
                    out.add(a)
                }
                return
            case .number:
                if numberByte(&a.top, byte) {
                    out.add(a)
                    return
                }
                guard numberEnds(a.top) else { return }
                a.pop()
            case .literal:
                if literalContinues(&a.top, byte, r) {
                    out.add(a)
                    return
                }
                guard literalEnds(a.top, r) else { return }
                a.pop()
            case .object:
                if space {
                    if Int(spaces) <= JSONRules.maximumSpaces(depth: a.depth) { out.add(a) }
                    return
                }
                let seen = a.top.mask
                let complete = r.required[node] & ~seen == 0
                let another = r.allKeys[node] & ~r.forbidden[node] & ~seen != 0 || r.additional[node] >= 0
                switch a.top.phase {
                case 0, 1:
                    if byte == 0x22, a.top.phase == 1 || another {
                        a.push(Frame(kind: .key, node: Int32(node), mask: r.allKeys[node], seen: seen))
                    } else if byte == 0x7D, a.top.phase == 0, complete {
                        a.pop()
                    } else {
                        return
                    }
                case 2:
                    guard byte == 0x3A else { return }
                    a.top.phase = 3
                case 3:
                    a.top.phase = 4
                    let property = Int(a.top.count)
                    begin(property >= 0 ? Int(r.values[node][property]) : Int(r.additional[node]), byte, a, r, &out)
                    return
                default:
                    if byte == 0x2C, another {
                        a.top.phase = 1
                    } else if byte == 0x7D, complete {
                        a.pop()
                    } else {
                        return
                    }
                }
                out.add(a)
                return
            case .array:
                if space {
                    if Int(spaces) <= JSONRules.maximumSpaces(depth: a.depth) { out.add(a) }
                    return
                }
                switch a.top.phase {
                case 0:
                    if byte == 0x5D, r.minimum[node] == 0 {
                        a.pop()
                        out.add(a)
                    } else if r.maximum[node] > 0 {
                        a.top.phase = 2
                        a.top.count = 1
                        begin(Int(r.item[node]), byte, a, r, &out)
                    }
                case 1:
                    a.top.phase = 2
                    a.top.count += 1
                    begin(Int(r.item[node]), byte, a, r, &out)
                default:
                    if byte == 0x2C, Int(a.top.count) < r.maximum[node] {
                        a.top.phase = 1
                        out.add(a)
                    } else if byte == 0x5D, Int(a.top.count) >= r.minimum[node] {
                        a.pop()
                        out.add(a)
                    }
                }
                return
            case .root:
                guard a.top.phase == 0 else { return }
                a.top.phase = 1
                begin(node, byte, a, r, &out)
                return
            }
        }
    }

    /// The first byte of a value of `node` (every type it may be): the new frame on top of `parent`.
    private static func begin(_ node: Int, _ byte: UInt8, _ parent: Alternative, _ r: JSONRules, _ out: inout Alternatives) {
        for choice in r.choices[node] {
            let c = Int(choice)
            var frame: Frame?
            switch r.kind[c] {
            case 0: if byte == 0x7B, parent.depth < JSONRules.maximumDepth { frame = Frame(kind: .object, node: choice) }
            case 1: if byte == 0x5B, parent.depth < JSONRules.maximumDepth { frame = Frame(kind: .array, node: choice) }
            case 2: if byte == 0x22 { frame = Frame(kind: .string, node: choice) }
            case 3, 4:
                let phase: UInt8? = byte == 0x2D ? 0 : byte == 0x30 ? 1 : (0x31...0x39).contains(byte) ? 2 : nil
                if let phase { frame = Frame(kind: .number, phase: phase, node: choice, count: 1, mask: r.kind[c] == 4 ? 1 : 0) }
            case 5:
                var matching: UInt64 = 0
                for (k, text) in r.literals[c].enumerated() where text.first == byte { matching |= 1 << UInt64(k) }
                if matching != 0 { frame = Frame(kind: .literal, node: choice, count: 1, mask: matching) }
            default: break
            }
            if let frame {
                var next = parent
                next.push(frame)
                out.add(next)
            }
        }
    }

    private static func literalContinues(_ frame: inout Frame, _ byte: UInt8, _ r: JSONRules) -> Bool {
        let texts = r.literals[Int(frame.node)]
        let position = Int(frame.count)
        var matching: UInt64 = 0
        var rest = frame.mask
        while rest != 0 {
            let k = rest.trailingZeroBitCount
            rest &= rest - 1
            if texts[k].count > position, texts[k][position] == byte { matching |= 1 << UInt64(k) }
        }
        guard matching != 0 else { return false }
        frame.mask = matching
        frame.count += 1
        return true
    }
    private static func literalEnds(_ frame: Frame, _ r: JSONRules) -> Bool {
        let texts = r.literals[Int(frame.node)]
        var rest = frame.mask
        while rest != 0 {
            let k = rest.trailingZeroBitCount
            rest &= rest - 1
            if texts[k].count == Int(frame.count) { return true }
        }
        return false
    }

    /// Number phases: 0 after `-`, 1 a leading `0`, 2 integer digits, 3 after `.`, 4 fraction digits, 5 after `e`,
    /// 6 after the exponent's sign, 7 exponent digits (at most two, so every number a double can hold; `hex` counts them).
    private static func numberByte(_ f: inout Frame, _ byte: UInt8) -> Bool {
        let digit = (0x30...0x39).contains(byte)
        let integer = f.mask == 1
        let room = f.count < 32
        let next: UInt8?
        switch f.phase {
        case 0: next = byte == 0x30 ? 1 : digit ? 2 : nil
        case 1, 2:
            if digit && f.phase == 2 && room {
                next = 2
            } else if byte == 0x2E && !integer && f.count < 30 {
                next = 3
            } else if (byte == 0x65 || byte == 0x45) && !integer && f.count < 30 {
                next = 5
            } else {
                next = nil
            }
        case 3: next = digit ? 4 : nil
        case 4: next = digit && room ? 4 : (byte == 0x65 || byte == 0x45) && f.count < 30 ? 5 : nil
        case 5: next = byte == 0x2B || byte == 0x2D ? 6 : digit ? 7 : nil
        case 6: next = digit ? 7 : nil
        default: next = digit && f.hex < 2 ? 7 : nil
        }
        guard let next else { return false }
        f.hex = next == 7 ? f.hex + 1 : 0
        f.phase = next
        f.count += 1
        return true
    }
    private static func numberEnds(_ f: Frame) -> Bool { [1, 2, 4, 7].contains(f.phase) }

    /// 0 rejected, 1 taken, 2 the closing quote.
    @inline(__always) private static func stringByte(_ f: inout Frame, _ byte: UInt8) -> UInt8 {
        if f.due > 0 {
            guard byte >= f.low, byte <= f.high else { return 0 }
            f.due -= 1
            f.low = 0x80
            f.high = 0xBF
            return 1
        }
        func hexValue(_ b: UInt8) -> UInt16? {
            switch b {
            case 0x30...0x39: return UInt16(b - 0x30)
            case 0x41...0x46: return UInt16(b - 0x41 + 10)
            case 0x61...0x66: return UInt16(b - 0x61 + 10)
            default: return nil
            }
        }
        switch f.escape {
        case 0:
            switch byte {
            case 0x22: return 2
            case 0x5C: f.escape = 1
            case 0..<0x20: return 0
            case 0x20..<0x80: break
            case 0xC2...0xDF: f.due = 1
            case 0xE0:
                f.due = 2
                f.low = 0xA0
            case 0xED:
                f.due = 2
                f.high = 0x9F
            case 0xE1...0xEF: f.due = 2
            case 0xF0:
                f.due = 3
                f.low = 0x90
            case 0xF1...0xF3: f.due = 3
            case 0xF4:
                f.due = 3
                f.high = 0x8F
            default: return 0
            }
        case 1:
            switch byte {
            case 0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74: f.escape = 0
            case 0x75:
                f.escape = 2
                f.hex = 0
            default: return 0
            }
        case 2...5:
            guard let value = hexValue(byte) else { return 0 }
            f.hex = f.hex << 4 | value
            // A lone low surrogate (DC00–DFFF) is refused at its second digit, before it could only end badly.
            if f.escape == 3, (0xDC...0xDF).contains(f.hex) { return 0 }
            f.escape += 1
            if f.escape == 6 { f.escape = (0xD800...0xDBFF).contains(f.hex) ? 6 : 0 }
        case 6:
            guard byte == 0x5C else { return 0 }
            f.escape = 7
        case 7:
            guard byte == 0x75 else { return 0 }
            f.escape = 8
            f.hex = 0
        default:
            guard let value = hexValue(byte) else { return 0 }
            f.hex = f.hex << 4 | value
            if f.escape == 8, value != 0xD { return 0 }
            if f.escape == 9, !(0xDC...0xDF).contains(f.hex) { return 0 }
            f.escape = f.escape == 11 ? 0 : f.escape + 1
        }
        return 1
    }
}

/// Each token's bytes for one tokenizer and vocabulary size, and a byte trie over them, built once per loaded model
/// (`TokenizerStack.jsonVocabulary`). Control tokens (the tokenizer's added tokens) are kept apart: JSON never takes
/// them; a tool-call opener may.
package final class JSONVocabulary: @unchecked Sendable {
    package let size: Int
    private let bytes: [UInt8], offsets: [Int32]
    private let special: [Bool]
    let specials: [Int]
    private let label: [UInt8], firstChild: [Int32], sibling: [Int32], ending: [Int32], nextEnding: [Int32]

    /// `token(id)`: the token's bytes and whether it is a control token; nil for an id the tokenizer cannot decode.
    package init(size: Int, token: (Int) -> (bytes: [UInt8], special: Bool)?) {
        self.size = size
        var bytes: [UInt8] = []
        var offsets: [Int32] = [0]
        var special = [Bool](repeating: false, count: size)
        var specials: [Int] = []
        var label: [UInt8] = [0]
        var firstChild: [Int32] = [-1]
        var sibling: [Int32] = [-1]
        var ending: [Int32] = [-1]
        var nextEnding = [Int32](repeating: -1, count: size)
        var rootChild = [Int32](repeating: -1, count: 256)
        for id in 0..<size {
            guard let (piece, control) = token(id), !piece.isEmpty else {
                special[id] = true
                offsets.append(Int32(bytes.count))
                continue
            }
            bytes += piece
            offsets.append(Int32(bytes.count))
            if control {
                special[id] = true
                specials.append(id)
                continue
            }
            var node: Int32 = 0
            for (index, byte) in piece.enumerated() {
                var child = index == 0 ? rootChild[Int(byte)] : firstChild[Int(node)]
                if index > 0 { while child >= 0, label[Int(child)] != byte { child = sibling[Int(child)] } }
                if child < 0 {
                    child = Int32(label.count)
                    label.append(byte)
                    firstChild.append(-1)
                    ending.append(-1)
                    if index == 0 {
                        sibling.append(firstChild[0])
                        firstChild[0] = child
                        rootChild[Int(byte)] = child
                    } else {
                        sibling.append(firstChild[Int(node)])
                        firstChild[Int(node)] = child
                    }
                }
                node = child
            }
            nextEnding[id] = ending[Int(node)]
            ending[Int(node)] = Int32(id)
        }
        self.bytes = bytes
        self.offsets = offsets
        self.special = special
        self.specials = specials
        self.label = label
        self.firstChild = firstChild
        self.sibling = sibling
        self.ending = ending
        self.nextEnding = nextEnding
    }

    package func bytes(of id: Int) -> ArraySlice<UInt8> { bytes[Int(offsets[id])..<Int(offsets[id + 1])] }
    package func isSpecial(_ id: Int) -> Bool { special[id] }
    package var trieNodes: Int { label.count }

    /// The tokens that keep `state` alive, as bits (token i: word i / 64, bit i % 64), plus `stops` when the JSON is
    /// complete. A depth-first walk of the trie, pruned where the acceptor dies, so tokens sharing a prefix share its work.
    package func allowed(from state: JSONAcceptor, rules: JSONRules, stops: [Int]) -> [UInt64] {
        var bits = [UInt64](repeating: 0, count: (size + 63) / 64)
        var pending: [(Int32, JSONAcceptor)] = [(0, state)]
        while let (node, current) = pending.popLast() {
            var child = firstChild[Int(node)]
            while child >= 0 {
                var next = current
                if next.step(label[Int(child)], rules) {
                    var id = ending[Int(child)]
                    while id >= 0 {
                        bits[Int(id) >> 6] |= 1 << UInt64(id & 63)
                        id = nextEnding[Int(id)]
                    }
                    if firstChild[Int(child)] >= 0 { pending.append((child, next)) }
                }
                child = sibling[Int(child)]
            }
        }
        if state.opensTool {
            for id in specials where !stops.contains(id) {
                var next = state
                if bytes(of: id).allSatisfy({ next.step($0, special: true, rules) }) { bits[id >> 6] |= 1 << UInt64(id & 63) }
            }
        }
        if state.complete(rules) { for id in stops where id < size { bits[id >> 6] |= 1 << UInt64(id & 63) } }
        return bits
    }
}

/// One request's JSON mode: masks each step's logits to the tokens that keep the answer valid JSON (before the sampler
/// applies temperature, top-k/p, min-p and penalties), and follows the tokens the reply takes. With thinking on it starts
/// after the think block's end token; a reply that opens a tool call (when the request allows tools) is not constrained
/// from there. Per request: nothing of it enters saved state. Masks are cached per acceptor state.
public final class JSONConstraint {
    package let rules: JSONRules
    private let vocabulary: JSONVocabulary
    private let stops: [Int]
    private let thinkEnd: Int?
    private var thinking: Bool
    package private(set) var state: JSONAcceptor
    private var cache: [JSONAcceptor: [UInt64]] = [:]
    /// Masks computed (cache misses) and the seconds they took, for measurement.
    package private(set) var computed = 0, computeSeconds = 0.0

    /// `thinking`: the reply opens in a think block that `thinkEnd` closes. `toolOpener`: the served format's tool-call
    /// opening, when the request allows tools.
    package init(
        grammar: JSONGrammar, vocabulary: JSONVocabulary, stops: [Int], thinkEnd: Int?, thinking: Bool, toolOpener: String?
    ) {
        let thinks = thinking && thinkEnd != nil
        rules = JSONRules(grammar: grammar, opener: toolOpener, leadingNewlines: thinks ? 2 : 0)
        self.vocabulary = vocabulary
        self.stops = stops
        self.thinkEnd = thinkEnd
        self.thinking = thinks
        state = JSONAcceptor(rules)
    }
    package convenience init(
        grammar: JSONGrammar, tokenizer: TokenizerStack, vocabularySize: Int, stops: [Int], thinking: Bool, toolOpener: String?
    ) {
        self.init(
            grammar: grammar, vocabulary: tokenizer.jsonVocabulary(size: vocabularySize), stops: stops, thinkEnd: tokenizer.thinkEnd,
            thinking: thinking, toolOpener: toolOpener)
    }

    /// Whether this step's logits are masked (not inside thinking, not inside a tool call).
    package var constrains: Bool { !thinking && !state.free }

    /// The allowed tokens from the current state (cached).
    package func allowedBits() -> [UInt64] {
        if let hit = cache[state] { return hit }
        let started = DispatchTime.now().uptimeNanoseconds
        let bits = vocabulary.allowed(from: state, rules: rules, stops: stops)
        computeSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        computed += 1
        if cache.count >= 1_024 { cache.removeAll(keepingCapacity: true) }
        cache[state] = bits
        return bits
    }

    /// `logits` with every token the JSON does not allow at -infinity; unchanged while nothing is constrained.
    package func masked(_ logits: [Float]) throws -> [Float] {
        guard constrains else { return logits }
        guard logits.count == vocabulary.size else { throw EngineError("JSON mode: the logits do not match the vocabulary") }
        let bits = allowedBits()
        var out = logits
        var usable = false
        out.withUnsafeMutableBufferPointer { values in
            for index in values.indices {
                if bits[index >> 6] >> UInt64(index & 63) & 1 == 0 {
                    values[index] = -.infinity
                } else if values[index].isFinite {
                    usable = true
                }
            }
        }
        guard usable else {
            throw EngineError("JSON mode: no token that keeps the reply valid JSON has a usable score", code: "json_mode")
        }
        return out
    }

    /// Follows a token the reply took (sampled or written by the server).
    package func accept(_ token: Int) throws {
        if thinking {
            if token == thinkEnd { thinking = false }
            return
        }
        guard !state.free, !stops.contains(token) else { return }
        guard (0..<vocabulary.size).contains(token) else { throw EngineError("JSON mode: token \(token) is outside the vocabulary") }
        var next = state
        let special = vocabulary.isSpecial(token)
        for byte in vocabulary.bytes(of: token) {
            guard next.step(byte, special: special, rules) else {
                throw EngineError("JSON mode: token \(token) would make the reply invalid JSON", code: "json_mode")
            }
        }
        if special && vocabulary.bytes(of: token).isEmpty {
            throw EngineError("JSON mode: token \(token) would make the reply invalid JSON", code: "json_mode")
        }
        state = next
    }
}
