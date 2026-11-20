import Foundation
import Jinja
import Tokenizers

/// Per-tokenizer, bounded prompt preparation reuse for all three serving engines.
/// The template still sees the complete current conversation: arbitrary Jinja may
/// inspect later messages. Only byte-identical converted values and independently
/// tokenized added-token sections are reused, never separately encoded messages.
final class PromptEncodingCache: @unchecked Sendable {
    private struct Message {
        let key: Data, value: Jinja.Value, bytes: Int
    }
    private struct Piece {
        let ids: [Int], bytes: Int
        var used: UInt64
    }
    private let lock = NSLock()
    private let template: Jinja.Template
    private let specials: [String: Jinja.Value]
    private let boundaries: NSRegularExpression?
    private var memory: CheckpointMemoryLease?
    private var messages: [Message] = []
    private var pieces: [Data: Piece] = [:]
    private var pieceBytes = 0, clock: UInt64 = 0
    private static let messageLimit = 8 << 20, pieceLimit = 8 << 20

    init(template: String, configuration: [String: TemplateValue], tokens: [String], sectionSafe: Bool) throws {
        self.template = try Jinja.Template(template, with: .init(lstripBlocks: true, trimBlocks: true))
        var specials: [String: Jinja.Value] = [:]
        for key in [
            "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "additional_special_tokens",
        ] {
            guard let value = configuration[key], value != .null else { continue }
            if case .object(let fields) = value {
                if case .string(let content)? = fields["content"] { specials[key] = .string(content) }
            } else {
                specials[key] = try Jinja.Value(any: value.templateValue)
            }
        }
        self.specials = specials
        // Match the pinned tokenizer's longest-first added-token splitting. Any
        // whitespace-stripping token or nonidentity postprocessor disables this
        // optimization for that tokenizer; the canonical full encode remains.
        boundaries =
            sectionSafe && !tokens.isEmpty
            ? try NSRegularExpression(
                pattern:
                    tokens.sorted { $0.count > $1.count }.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")) : nil
    }

    func admit(ledger: MemoryLedger) {
        lock.lock()
        defer { lock.unlock() }
        guard memory == nil else { return }
        // Payload limits include conservative per-entry object/collection costs;
        // the remaining 8 MiB covers cache tables and transient replacement keys.
        memory = try? CheckpointMemoryLease(ledger: ledger, bytes: 24 << 20, category: "prompt_encoding_cache")
    }

    func encode(
        _ normalized: [[String: TemplateValue]], tools: [[String: TemplateValue]],
        context additional: [String: any Sendable], tokenizer: any Tokenizers.Tokenizer
    ) throws -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        var values: [Jinja.Value] = []
        var retained: [Message] = []
        var retainedBytes = 0
        for (index, message) in normalized.enumerated() {
            // Data equality is intentional: Swift String equality treats Unicode
            // normalization variants alike, whereas byte-level BPE does not.
            let key: Data? = memory == nil ? nil : Self.messageKey(message)
            let value: Jinja.Value
            if let key, index < messages.count, messages[index].key == key {
                value = messages[index].value
            } else {
                value = try Jinja.Value(any: TokenizerStack.templateMessage(message))
            }
            values.append(value)
            if let key {
                let cost = key.count * 8 + 1024
                if retained.count == index, index < 4096, cost <= Self.messageLimit - retainedBytes {
                    retained.append(Message(key: key, value: value, bytes: cost))
                    retainedBytes += cost
                }
            }
        }
        messages = retained
        var context: [String: Jinja.Value] = ["messages": .array(values), "add_generation_prompt": .boolean(true)]
        if !tools.isEmpty { context["tools"] = try .array(tools.map { try Jinja.Value(any: $0.mapValues(\.templateValue)) }) }
        for (key, value) in additional { context[key] = try Jinja.Value(any: value) }
        for (key, value) in specials { context[key] = value }
        let rendered = try template.render(context)
        guard memory != nil, let boundaries else { return tokenizer.encode(text: rendered, addSpecialTokens: false) }
        let matches = boundaries.matches(in: rendered, range: NSRange(rendered.startIndex..<rendered.endIndex, in: rendered))
        guard !matches.isEmpty else { return piece(rendered, tokenizer: tokenizer) }
        // Every noninitial piece starts with an added token. Its following text
        // therefore keeps the upstream tokenizer's non-first-section semantics.
        var start = rendered.startIndex
        var ids: [Int] = []
        for match in matches {
            guard let range = Range(match.range, in: rendered), start < range.lowerBound else { continue }
            ids += piece(String(rendered[start..<range.lowerBound]), tokenizer: tokenizer)
            start = range.lowerBound
        }
        if start < rendered.endIndex { ids += piece(String(rendered[start...]), tokenizer: tokenizer) }
        return ids
    }

    private func piece(_ text: String, tokenizer: any Tokenizers.Tokenizer) -> [Int] {
        guard text.utf8.count <= Self.pieceLimit else { return tokenizer.encode(text: text, addSpecialTokens: false) }
        let key = Data(text.utf8)
        clock &+= 1
        if var cached = pieces[key] {
            cached.used = clock
            pieces[key] = cached
            return cached.ids
        }
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let cost = key.count + ids.count * MemoryLayout<Int>.stride + 256
        guard cost <= Self.pieceLimit else { return ids }
        while pieceBytes + cost > Self.pieceLimit || pieces.count >= 4096 {
            guard let oldest = pieces.min(by: { $0.value.used < $1.value.used }) else { break }
            pieceBytes -= oldest.value.bytes
            pieces.removeValue(forKey: oldest.key)
        }
        pieces[key] = Piece(ids: ids, bytes: cost, used: clock)
        pieceBytes += cost
        return ids
    }

    /// Bounded typed keys preserve distinctions JSON serialization may erase
    /// (integer 1 versus floating 1.0, signed zero, and exact Unicode bytes).
    private static func messageKey(_ message: [String: TemplateValue]) -> Data? {
        let limit = 1 << 20
        var data = Data()
        func word(_ number: UInt64) -> Bool {
            guard data.count <= limit - 8 else { return false }
            var number = number.littleEndian
            withUnsafeBytes(of: &number) { data.append(contentsOf: $0) }
            return true
        }
        func string(_ text: String) -> Bool {
            let count = text.utf8.count
            guard count <= limit - data.count, word(UInt64(count)), count <= limit - data.count else { return false }
            data.append(contentsOf: text.utf8)
            return true
        }
        func value(_ item: TemplateValue) -> Bool {
            guard data.count < limit else { return false }
            switch item {
            case .null:
                data.append(0)
                return true
            case .bool(let flag):
                data.append(flag ? 1 : 2)
                return true
            case .integer(let number):
                data.append(3)
                return word(UInt64(bitPattern: Int64(number)))
            case .number(let number):
                data.append(4)
                return word(number.bitPattern)
            case .string(let text):
                data.append(5)
                return string(text)
            case .array(let items):
                data.append(6)
                guard word(UInt64(items.count)) else { return false }
                for item in items { guard value(item) else { return false } }
                return true
            case .object(let fields):
                data.append(7)
                guard word(UInt64(fields.count)) else { return false }
                for key in fields.keys.sorted() {
                    guard string(key), value(fields[key]!) else { return false }
                }
                return true
            }
        }
        return value(.object(message)) ? data : nil
    }

    /// Only processors that are identity for a single sequence with special-token
    /// insertion disabled may be applied independently to added-token sections.
    static func identityPostprocessor(_ value: Any?) -> Bool {
        guard let value, !(value is NSNull) else { return true }
        guard let object = value as? [String: Any], let type = object["type"] as? String else { return false }
        switch type {
        case "ByteLevel", "BertProcessing": return true
        case "Sequence":
            guard let processors = object["processors"] as? [Any] else { return false }
            return processors.allSatisfy { identityPostprocessor($0) }
        case "TemplateProcessing":
            guard let single = object["single"] as? [[String: Any]] else { return false }
            let sequences = single.compactMap { ($0["Sequence"] as? [String: Any])?["id"] as? String }
            return sequences == ["A"] && single.allSatisfy { $0["Sequence"] != nil || $0["SpecialToken"] != nil }
        default: return false
        }
    }
}
