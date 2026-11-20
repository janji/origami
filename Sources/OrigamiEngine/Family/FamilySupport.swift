import CryptoKit
import Foundation

/// What a GGUF model needs beside its weights to serve chats: a Hugging Face tokenizer (tokenizer.json,
/// tokenizer_config.json) and the chat template, written from the GGUF's own vocabulary, merges, token types and
/// template, so a fine-tune with added tokens or its own template is served as it was trained. The pre-tokenizer
/// comes from `tokenizer.ggml.pre` (llama.cpp's names); an unknown one is refused, naming it.
package enum FamilySupport {
    /// Hugging Face pre-tokenizers for llama.cpp's `tokenizer.ggml.pre` names used by these families.
    static func preTokenizer(_ name: String) -> [String: Any]? {
        func split(_ regex: String) -> [String: Any] {
            ["type": "Split", "pattern": ["Regex": regex], "behavior": "Isolated", "invert": false]
        }
        let byteLevel: [String: Any] = ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
        switch name {
        case "gpt-4o":  // o200k: case-aware words, contractions after them
            return [
                "type": "Sequence",
                "pretokenizers": [
                    split(
                        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                            + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                            + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
                    byteLevel,
                ],
            ]
        case "glm4", "llama-bpe", "llama3":
            return [
                "type": "Sequence",
                "pretokenizers": [
                    split(
                        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    ),
                    byteLevel,
                ],
            ]
        case "joyai-llm", "deepseek-v3", "deepseek3-llm", "hy-v4":
            return [
                "type": "Sequence",
                "pretokenizers": [
                    split("\\p{N}{1,3}"), split("[一-龥぀-ゟ゠-ヿ]+"),
                    split(
                        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    ),
                    byteLevel,
                ],
            ]
        default: return nil
        }
    }

    /// Whether a word found whole in the vocabulary skips the merges: GLM's and Llama 3's tokenizers do (their
    /// tokenizer.json sets it, as llama.cpp does for them); DeepSeek's and o200k's do not.
    static func ignoreMerges(_ name: String) -> Bool { ["glm4", "llama-bpe", "llama3"].contains(name) }

    /// End-of-turn tokens: the GGUF's eos, eot and eom ids.
    package static func stopTokens(_ gguf: GGUFSource) -> Set<Int> {
        Set(
            ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]
                .compactMap { gguf.metadata[$0]?.integer.map(Int.init) })
    }

    /// Writes the support files into `folder` (created; `key` names it by the GGUF's header identity); returns the folder.
    @discardableResult
    package static func prepare(_ url: URL, into folder: URL) throws -> URL {
        let gguf = try GGUFSource(url, tokenizer: true)
        guard case .string(let model)? = gguf.metadata["tokenizer.ggml.model"], model == "gpt2" else {
            throw EngineError(
                "This GGUF's tokenizer is not byte-level BPE (tokenizer.ggml.model \(gguf.metadata["tokenizer.ggml.model"].map { "\($0)" } ?? "missing"))",
                code: "model_incompatible")
        }
        var pre = "default"
        if case .string(let value)? = gguf.metadata["tokenizer.ggml.pre"] { pre = value }
        guard let preTokenizer = preTokenizer(pre) else {
            throw EngineError("This GGUF's pre-tokenizer \"\(pre)\" is unknown to Origami", code: "model_incompatible")
        }
        let tokens = (gguf.tokenizer["tokenizer.ggml.tokens"] ?? []).compactMap { value -> String? in
            if case .string(let s) = value { return s }
            return nil
        }
        let merges = (gguf.tokenizer["tokenizer.ggml.merges"] ?? []).compactMap { value -> [String]? in
            guard case .string(let s) = value, let space = s.firstIndex(of: " ") else { return nil }
            return [String(s[..<space]), String(s[s.index(after: space)...])]
        }
        let types = (gguf.tokenizer["tokenizer.ggml.token_type"] ?? []).compactMap(\.integer)
        guard !tokens.isEmpty, !merges.isEmpty else {
            throw EngineError("This GGUF has no tokenizer vocabulary or merges", code: "model_incompatible")
        }
        var template = ""
        if case .string(let value)? = gguf.metadata["tokenizer.chat_template"] { template = value }
        guard !template.isEmpty else {
            throw EngineError("This GGUF has no chat template (tokenizer.chat_template)", code: "model_incompatible")
        }
        template = jinjaCompatible(template)

        var vocab: [String: Int] = [:]
        var added: [[String: Any]] = []
        for (id, token) in tokens.enumerated() {
            // llama.cpp token types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.
            let type = id < types.count ? types[id] : 1
            if type == 3 || type == 4 {
                added.append([
                    "id": id, "content": token, "single_word": false, "lstrip": false, "rstrip": false,
                    "normalized": false, "special": type == 3,
                ])
            }
            if vocab[token] == nil { vocab[token] = id }
        }
        func name(_ key: String) -> String? {
            gguf.metadata[key]?.integer.flatMap { id in (0..<Int64(tokens.count)).contains(id) ? tokens[Int(id)] : nil }
        }
        let tokenizer: [String: Any] = [
            "version": "1.0", "truncation": NSNull(), "padding": NSNull(), "added_tokens": added, "normalizer": NSNull(),
            "pre_tokenizer": preTokenizer,
            "post_processor": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true],
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": NSNull(),
                "end_of_word_suffix": NSNull(), "fuse_unk": false, "byte_fallback": false, "ignore_merges": ignoreMerges(pre),
                "vocab": vocab, "merges": merges,
            ],
        ]
        var config: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizerFast", "clean_up_tokenization_spaces": false,
            "model_max_length": 1_048_576, "chat_template": template,
        ]
        if let eos = name("tokenizer.ggml.eos_token_id") { config["eos_token"] = eos }
        if let bos = name("tokenizer.ggml.bos_token_id") { config["bos_token"] = bos }
        if let pad = name("tokenizer.ggml.padding_token_id") { config["pad_token"] = pad }
        config["add_bos_token"] = gguf.metadata["tokenizer.ggml.add_bos_token"].map { $0 == .bool(true) } ?? false

        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys]).write(
            to: folder.appendingPathComponent("tokenizer.json"), options: .atomic)
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys, .prettyPrinted]).write(
            to: folder.appendingPathComponent("tokenizer_config.json"), options: .atomic)
        try Data(template.utf8).write(to: folder.appendingPathComponent("chat_template.jinja"), options: .atomic)
        return folder
    }

    /// Jinja's numeric attribute access (`m.content.0.type`, as GLM's template has) written as item access
    /// (`m.content[0].type`), which the template engine parses; float literals such as `1.0` are left alone.
    package static func jinjaCompatible(_ template: String) -> String {
        template.replacingOccurrences(of: #"(?<=[A-Za-z_\])\)])\.(\d+)\b"#, with: "[$1]", options: .regularExpression)
    }

    /// A folder name for a GGUF's local file incarnation. Same-shaped different weights, replacements and ordinary
    /// in-place edits cannot reuse a previous model's support, hot copy, packs or saved prompts. Not a content digest.
    /// A low-bit variant (`GGUFSource.withExperts`) has a key of its own: its usage counts, hot copy and saved prompts are
    /// never the base's.
    package static func key(_ gguf: GGUFSource) -> String { key(folder: gguf.folder, files: gguf.files, experts: gguf.expertFiles) }
    /// The same from the parts' names (`GGUFSource.parts`), without reading a header; `experts`: a variant's experts file parts.
    package static func key(folder: URL, files: [String], experts: [String] = []) -> String {
        var hash = SHA256()
        for name in files.sorted() { hash.update(data: LocalModelIdentity.file(folder.appendingPathComponent(name))) }
        LocalModelIdentity.updateSupport(&hash, directory: folder)
        for path in experts.sorted() {
            hash.update(data: Data("experts".utf8))
            hash.update(data: LocalModelIdentity.file(URL(fileURLWithPath: path)))
        }
        return hash.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }
    /// The key Origami 2.x gave these parts (their names and sizes alone), only to recognise the copies it made.
    package static func earlierKey(folder: URL, files: [String]) -> String {
        let identity = files.map { name -> String in
            let path = folder.appendingPathComponent(name).resolvingSymlinksInPath().path
            let size = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.uint64Value ?? 0
            return "\(name):\(size)"
        }.joined(separator: "\n")
        return SHA256.hash(data: Data(identity.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}
