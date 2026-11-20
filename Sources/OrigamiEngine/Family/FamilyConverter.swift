import Foundation

/// Converts a GLM-5.3-Flash safetensors checkpoint (FP8 E4M3 with block scales, or BF16/F32) into a GGUF the family
/// engine runs: DwarfStar's tensor names (which `GLMModel` reads), the checkpoint's config.json embedded as
/// `glm5-next.config` (its geometry), its tokenizer and chat template, dense matrices in Q8_0, norms and small vectors
/// in F32, the hyper-connection and indexer projections in BF16, and the routed experts in `expertType` (Q2_K by
/// default, a small streaming pack; Q4_K or Q8_0). The plan follows DwarfStar's glm53_quantize.py (MIT), including its kv_b split into
/// per-head k_b (transposed) and v_b. The source is only read.
package struct FamilyConverter {
    struct Item {
        let name: String
        /// GGUF dims, fastest first.
        let shape: [Int]
        let type: Int
        /// Source tensor names (one, or one per expert).
        let sources: [String]
        let transform: Transform?
        var bytes: Int { GGUFQuantize.bytes(type, count: shape.reduce(1, *)) }
    }
    enum Transform { case kvbK, kvbV }

    package let source: URL
    let config: [String: Any]
    let configText: String
    let headers: [String: TensorSpan]
    let items: [Item]
    package let expertType: Int
    /// The chat template (chat_template.jinja, or tokenizer_config.json's), required before anything is converted.
    let template: String
    /// Bytes of the GGUF's tensor data.
    package var outputBytes: Int { items.reduce(0) { $0 + ($1.bytes + 31) / 32 * 32 } }

    package init(source: URL, expertType: Int = 10, layers only: Set<Int>? = nil) throws {
        guard [10, 12, 8].contains(expertType) else {
            throw EngineError("Experts convert to Q2_K, Q4_K or Q8_0", code: "invalid_configuration")
        }
        self.source = source
        self.expertType = expertType
        let configURL = source.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any], ModelFamily.detect(config: config) == .glm
        else {
            throw EngineError("\(source.path) is not a GLM-5.3-Flash checkpoint", code: "model_incompatible")
        }
        self.config = config
        configText = String(decoding: configData, as: UTF8.self)
        let tokenizerConfig = (try? Data(contentsOf: source.appendingPathComponent("tokenizer_config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        guard
            let template = (try? String(contentsOf: source.appendingPathComponent("chat_template.jinja"), encoding: .utf8))
                ?? tokenizerConfig?["chat_template"] as? String, !template.isEmpty
        else {
            throw EngineError(
                "\(source.path) has no chat template (chat_template.jinja or tokenizer_config.json)", code: "model_incompatible")
        }
        self.template = template
        let geo = try ModelGeometry(config: config)
        var headers: [String: TensorSpan] = [:]
        for shard in try FileManager.default.contentsOfDirectory(atPath: source.path) where shard.hasSuffix(".safetensors") {
            for (name, span) in try ModelSource.header(of: try LocalBytes(source.appendingPathComponent(shard)), file: shard) {
                headers[name] = span
            }
        }
        self.headers = headers
        // Hugging Face names lose a `language_model.` level in some exports.
        let root = headers.keys.contains("model.language_model.embed_tokens.weight") ? "model.language_model" : "model"
        func has(_ name: String) -> Bool { headers[name] != nil }
        func shape(_ name: String) throws -> [Int] {
            guard let span = headers[name] else { throw EngineError("The checkpoint lacks \(name)", code: "model_incompatible") }
            return span.shape.reversed()
        }
        var items: [Item] = []
        func add(_ name: String, _ source: String, _ type: Int, shape override: [Int]? = nil, transform: Transform? = nil) throws {
            items.append(Item(name: name, shape: try override ?? shape(source), type: type, sources: [source], transform: transform))
        }
        let q8 = 8
        let f32 = 0
        let bf16 = 30
        try add("token_embd.weight", "\(root).embed_tokens.weight", q8)
        try add("output.weight", has("lm_head.weight") ? "lm_head.weight" : "\(root).embed_tokens.weight", q8)
        try add("output_norm.weight", "\(root).norm.weight", f32)
        for l in 0..<(geo.layers + geo.mtpLayers) where only?.contains(l) ?? true {
            let p = "\(root).layers.\(l)"
            let trunk = l < geo.layers
            try add("blk.\(l).attn_norm.weight", "\(p).input_layernorm.weight", f32)
            try add("blk.\(l).ffn_norm.weight", "\(p).post_attention_layernorm.weight", f32)
            if trunk {
                for site in ["attn", "ffn"] {
                    try add("blk.\(l).hc_\(site)_fn.weight", "\(p).hc_\(site)_fn", bf16)
                    try add("blk.\(l).hc_\(site)_base.weight", "\(p).hc_\(site)_base", f32)
                    try add("blk.\(l).hc_\(site)_scale.weight", "\(p).hc_\(site)_scale", f32)
                }
            }
            let a = "\(p).self_attn"
            if trunk && geo.mixers[l] == .recurrent {
                for (target, name, type) in [
                    ("q", "q_proj.weight", q8), ("k", "k_proj.weight", q8), ("v", "v_proj.weight", q8), ("q_conv", "q_conv1d.weight", f32),
                    ("k_conv", "k_conv1d.weight", f32), ("v_conv", "v_conv1d.weight", f32), ("f_a", "f_a_proj.weight", q8),
                    ("f_b", "f_b_proj.weight", q8), ("dt_bias", "dt_bias", f32), ("a_log", "A_log", f32), ("beta", "b_proj.weight", q8),
                    ("g_a", "g_a_proj.weight", q8), ("g_b", "g_b_proj.weight", q8), ("o_norm", "o_norm.weight", f32),
                    ("output", "o_proj.weight", q8),
                ] {
                    try add("blk.\(l).kda_\(target).weight", "\(a).\(name)", type)
                }
            } else {
                for (target, name, type) in [
                    ("attn_q_a.weight", "q_a_proj.weight", q8), ("attn_q_a_norm.weight", "q_a_layernorm.weight", f32),
                    ("attn_q_b.weight", "q_b_proj.weight", q8), ("attn_kv_a_mqa.weight", "kv_a_proj_with_mqa.weight", q8),
                    ("attn_kv_a_norm.weight", "kv_a_layernorm.weight", f32), ("attn_output.weight", "o_proj.weight", q8),
                    ("indexer.attn_q_b.weight", "indexer.wq_b.weight", bf16), ("indexer.attn_k.weight", "indexer.wk.weight", bf16),
                    ("indexer.k_norm.weight", "indexer.k_norm.weight", f32), ("indexer.k_norm.bias", "indexer.k_norm.bias", f32),
                    ("indexer.proj.weight", "indexer.weights_proj.weight", bf16),
                    ("indexer.pool_ape.weight", "indexer.index_kpool_compress_ape", bf16),
                    ("indexer.pool_gate.weight", "indexer.index_kpool_compress_gate", bf16),
                ] {
                    try add("blk.\(l).\(target)", "\(a).\(name)", type)
                }
                let heads = geo.int("num_attention_heads") ?? 64
                let rank = geo.int("kv_lora_rank") ?? 512
                let kDim = geo.int("qk_head_dim") ?? 256
                let vDim = geo.int("v_head_dim") ?? 256
                guard try shape("\(a).kv_b_proj.weight") == [rank, heads * (kDim + vDim)] else {
                    throw EngineError("Layer \(l)'s kv_b_proj is not \(heads) x (\(kDim) + \(vDim)) by \(rank)", code: "model_incompatible")
                }
                try add("blk.\(l).attn_k_b.weight", "\(a).kv_b_proj.weight", q8, shape: [kDim, rank, heads], transform: .kvbK)
                try add("blk.\(l).attn_v_b.weight", "\(a).kv_b_proj.weight", q8, shape: [rank, vDim, heads], transform: .kvbV)
            }
            let m = "\(p).mlp"
            if trunk && !geo.moe[l] {
                for part in ["gate", "up", "down"] { try add("blk.\(l).ffn_\(part).weight", "\(m).\(part)_proj.weight", q8) }
            } else {
                try add("blk.\(l).ffn_gate_inp.weight", "\(m).gate.weight", f32)
                try add("blk.\(l).exp_probs_b.bias", "\(m).gate.e_score_correction_bias", f32)
                for part in ["gate", "up", "down"] {
                    let names = (0..<geo.experts).map { "\(m).experts.\($0).\(part)_proj.weight" }
                    let first = try shape(names[0])
                    for name in names where try shape(name) != first {
                        throw EngineError("\(name) differs in shape from expert 0", code: "model_incompatible")
                    }
                    items.append(
                        Item(
                            name: "blk.\(l).ffn_\(part)_exps.weight", shape: first + [geo.experts], type: expertType, sources: names,
                            transform: nil))
                    try add("blk.\(l).ffn_\(part)_shexp.weight", "\(m).shared_experts.\(part)_proj.weight", q8)
                }
            }
            if !trunk {
                for (target, name) in [
                    ("nextn.eh_proj.weight", "\(p).eh_proj.weight"), ("nextn.enorm.weight", "\(p).enorm.weight"),
                    ("nextn.hnorm.weight", "\(p).hnorm.weight"), ("nextn.shared_head_norm.weight", "\(p).shared_head.norm.weight"),
                ] where has(name) {
                    try add("blk.\(l).\(target)", name, target.hasSuffix("norm.weight") ? f32 : bf16)
                }
            }
        }
        for item in items {
            let (block, _) = GGUFQuant.sizes[item.type]!
            guard item.shape[0] % block == 0 else {
                throw EngineError(
                    "\(item.name)'s rows of \(item.shape[0]) are not whole \(GGUFQuant.name(item.type)) blocks", code: "model_incompatible")
            }
        }
        self.items = items
    }

    /// A source tensor's values as f32 (FP8 times its block scales, BF16, F16 or F32), rows `rows` of a 2-D tensor.
    package func floats(_ name: String, files: inout [String: ExactFile]) throws -> (values: [Float], shape: [Int]) {
        guard let span = headers[name] else { throw EngineError("The checkpoint lacks \(name)", code: "model_incompatible") }
        func open(_ file: String) throws -> ExactFile {
            if let f = files[file] { return f }
            let f = try ExactFile(url: source.appendingPathComponent(file))
            files[file] = f
            return f
        }
        let raw = try open(span.file).read(offset: span.offset, length: Int(span.length))
        let count = span.shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        switch span.dtype {
        case "F32": raw.withUnsafeBytes { values = Array($0.bindMemory(to: Float.self)) }
        case "BF16": raw.withUnsafeBytes { values = $0.bindMemory(to: UInt16.self).map(BF16.decode) }
        case "F16": raw.withUnsafeBytes { values = $0.bindMemory(to: UInt16.self).map { Float(Float16(bitPattern: $0)) } }
        case "F8_E4M3":
            guard let scale = headers[name + "_scale_inv"], span.shape.count == 2, scale.shape.count == 2,
                ["F32", "BF16"].contains(scale.dtype)
            else {
                throw EngineError("\(name) is FP8 without F32 or BF16 block scales", code: "model_incompatible")
            }
            let rows = span.shape[0]
            let columns = span.shape[1]
            let quantization = config["quantization_config"] as? [String: Any]
            let block = quantization?["weight_block_size"] as? [Int] ?? [128, 128]
            guard block.count == 2, (rows + block[0] - 1) / block[0] == scale.shape[0],
                (columns + block[1] - 1) / block[1] == scale.shape.last
            else {
                throw EngineError("\(name)'s scales \(scale.shape) do not tile it in blocks of \(block)", code: "model_incompatible")
            }
            let blockRows = block[0]
            let blockColumns = block[1]
            let s = try open(scale.file).read(offset: scale.offset, length: Int(scale.length))
            let scales: [Float] = s.withUnsafeBytes {
                scale.dtype == "F32" ? Array($0.bindMemory(to: Float.self)) : $0.bindMemory(to: UInt16.self).map(BF16.decode)
            }
            raw.withUnsafeBytes {
                CheckpointConverter.decodeFP8(
                    $0.baseAddress!, rows: 0..<rows, columns: columns, scales: scales, block: [blockRows, blockColumns], into: &values)
            }
        default: throw EngineError("\(name) is \(span.dtype), which the converter does not read", code: "model_incompatible")
        }
        // The quantizers turn NaN or infinity into garbage (or trap, for Q8_0's integer conversion).
        guard values.allSatisfy(\.isFinite) else { throw EngineError("\(name) holds NaN or infinite values", code: "model_incompatible") }
        return (values, span.shape)
    }

    /// The values of one item in GGUF order (rows of shape[0]), after its transform.
    func values(_ item: Item, source index: Int, files: inout [String: ExactFile]) throws -> [Float] {
        let (v, shape) = try floats(item.sources[index], files: &files)
        guard let transform = item.transform else { return v }
        // kv_b: [heads * (k + v), rank] -> per head: k_b transposed [rank][k] (GGUF [k, rank, heads]) or v_b [v][rank].
        let rank = shape[1]
        let heads = item.shape[2]
        let combined = shape[0] / heads
        let k = transform == .kvbK ? item.shape[0] : combined - item.shape[1]
        let vDim = combined - k
        var out = [Float](repeating: 0, count: item.shape.reduce(1, *))
        for h in 0..<heads {
            switch transform {
            case .kvbK: for r in 0..<rank { for j in 0..<k { out[(h * rank + r) * k + j] = v[(h * combined + j) * rank + r] } }
            case .kvbV: for i in 0..<vDim { for r in 0..<rank { out[(h * vDim + i) * rank + r] = v[(h * combined + k + i) * rank + r] } }
            }
        }
        return out
    }

    /// Writes the GGUF (to a partial name, renamed when complete; removed on failure and when `cancelled`, polled between
    /// tensors: it cannot be resumed). `progress` gets bytes written. Only an earlier Origami conversion at `destination`
    /// is replaced.
    package func convert(to destination: URL, progress: (Int) -> Void = { _ in }, cancelled: () -> Bool = { false }) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            var ours = false
            if let existing = try? GGUFSource(destination), case .string(let by)? = existing.metadata["general.converter"] {
                ours = by.hasPrefix("origami ")
            }
            guard ours else {
                throw EngineError(
                    "\(destination.path) already exists and is not an Origami conversion; move it or choose another --name",
                    code: "destination_exists")
            }
        }
        var header = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func string(_ s: String) {
            u64(UInt64(s.utf8.count))
            header.append(contentsOf: s.utf8)
        }
        var kv: [(String, () -> Void)] = []
        func kvString(_ key: String, _ value: String) {
            kv.append(
                (
                    key,
                    {
                        u32(8)
                        string(value)
                    }
                ))
        }
        func kvU32(_ key: String, _ value: Int) {
            kv.append(
                (
                    key,
                    {
                        u32(4)
                        u32(UInt32(value))
                    }
                ))
        }
        func kvStrings(_ key: String, _ values: [String]) {
            kv.append(
                (
                    key,
                    {
                        u32(9)
                        u32(8)
                        u64(UInt64(values.count))
                        values.forEach(string)
                    }
                ))
        }
        func kvInts(_ key: String, _ values: [Int32]) {
            kv.append(
                (
                    key,
                    {
                        u32(9)
                        u32(5)
                        u64(UInt64(values.count))
                        values.forEach { u32(UInt32(bitPattern: $0)) }
                    }
                ))
        }
        kvString("general.architecture", "glm5-next")
        kvString("general.name", source.lastPathComponent)
        kvU32("general.alignment", 32)
        kvString("glm5-next.config", configText)
        kvString("general.converter", "origami expert type \(GGUFQuant.name(expertType)); dense Q8_0; from \(source.lastPathComponent)")
        // Tokenizer from tokenizer.json: ids in order (vocab plus added tokens), merges, token types, template.
        let tokenizerURL = source.appendingPathComponent("tokenizer.json")
        guard let tok = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerURL)) as? [String: Any],
            let model = tok["model"] as? [String: Any], let vocab = model["vocab"] as? [String: Int]
        else {
            throw EngineError("\(tokenizerURL.path) is not a BPE tokenizer.json", code: "model_incompatible")
        }
        let added = (tok["added_tokens"] as? [[String: Any]] ?? []).compactMap { e -> (Int, String, Bool)? in
            guard let id = e["id"] as? Int, let content = e["content"] as? String else { return nil }
            return (id, content, e["special"] as? Bool ?? false)
        }
        let text = config["text_config"] as? [String: Any] ?? config
        let vocabSize = text["vocab_size"] as? Int ?? max(vocab.values.max() ?? 0, added.map(\.0).max() ?? 0) + 1
        var tokens = [String](repeating: "", count: vocabSize)
        var types = [Int32](repeating: 1, count: vocabSize)
        for (t, id) in vocab where id < vocabSize { tokens[id] = t }
        for (id, content, special) in added where id < vocabSize {
            tokens[id] = content
            types[id] = special ? 3 : 4
        }
        for id in 0..<vocabSize where tokens[id].isEmpty {
            tokens[id] = "[PAD\(id)]"
            types[id] = 5
        }
        let merges = (model["merges"] as? [Any] ?? []).compactMap { m -> String? in (m as? [String])?.joined(separator: " ") ?? m as? String
        }
        kvString("tokenizer.ggml.model", "gpt2")
        kvString("tokenizer.ggml.pre", "glm4")
        kvStrings("tokenizer.ggml.tokens", tokens)
        kvInts("tokenizer.ggml.token_type", types)
        kvStrings("tokenizer.ggml.merges", merges)
        // A list of end tokens (GLM: end of text, <|user|>, <|observation|>) keeps its first three as eos, eot and eom,
        // the stops `FamilySupport.stopTokens` reads.
        let ends = (text["eos_token_id"] as? [Int]) ?? (text["eos_token_id"] as? Int).map { [$0] } ?? []
        for (key, id) in zip(["eos", "eot", "eom"], ends) { kvU32("tokenizer.ggml.\(key)_token_id", id) }
        if let pad = text["pad_token_id"] as? Int { kvU32("tokenizer.ggml.padding_token_id", pad) }
        kvString("tokenizer.chat_template", template)

        header.append(contentsOf: "GGUF".utf8)
        u32(3)
        u64(UInt64(items.count))
        u64(UInt64(kv.count))
        for (key, write) in kv {
            string(key)
            write()
        }
        var at: UInt64 = 0
        for item in items {
            string(item.name)
            u32(UInt32(item.shape.count))
            item.shape.forEach { u64(UInt64($0)) }
            u32(UInt32(item.type))
            u64(at)
            at += UInt64((item.bytes + 31) / 32 * 32)
        }
        header.append(Data(count: (32 - header.count % 32) % 32))

        let partial = destination.appendingPathExtension("partial")
        fm.createFile(atPath: partial.path, contents: nil)
        let out = try FileHandle(forWritingTo: partial)
        var finished = false
        defer {
            try? out.close()
            if !finished { try? fm.removeItem(at: partial) }
        }
        try out.write(contentsOf: header)
        var files: [String: ExactFile] = [:]
        var written = header.count
        for item in items {
            if cancelled() { throw EngineError("Conversion stopped", code: "cancelled") }
            var encoded = Data(count: (item.bytes + 31) / 32 * 32)
            if item.sources.count == 1 {
                let v = try values(item, source: 0, files: &files)
                guard v.count == item.shape.reduce(1, *) else {
                    throw EngineError("\(item.name): \(v.count) values for shape \(item.shape)")
                }
                v.withUnsafeBufferPointer { src in
                    encoded.withUnsafeMutableBytes {
                        GGUFQuantize.encode(src.baseAddress!, count: v.count, type: item.type, into: $0.baseAddress!)
                    }
                }
            } else {
                // Experts: one source tensor each, quantized in parallel into its slice.
                let perExpert = item.shape.dropLast().reduce(1, *)
                let expertBytes = GGUFQuantize.bytes(item.type, count: perExpert)
                final class Box: @unchecked Sendable {
                    var error: Error?
                    let lock = NSLock()
                }
                let box = Box()
                let sources = item.sources
                encoded.withUnsafeMutableBytes { dst in
                    nonisolated(unsafe) let base = dst.baseAddress!
                    nonisolated(unsafe) let me = self  // each expert writes its own part of base; me is only read
                    DispatchQueue.concurrentPerform(iterations: sources.count) { e in
                        do {
                            var local: [String: ExactFile] = [:]
                            let v = try me.values(item, source: e, files: &local)
                            v.withUnsafeBufferPointer {
                                GGUFQuantize.encode($0.baseAddress!, count: perExpert, type: item.type, into: base + e * expertBytes)
                            }
                        } catch {
                            box.lock.lock()
                            box.error = error
                            box.lock.unlock()
                        }
                    }
                }
                if let error = box.error { throw error }
            }
            try out.write(contentsOf: encoded)
            written += encoded.count
            progress(written)
        }
        try out.synchronize()
        try? fm.removeItem(at: destination)
        try fm.moveItem(at: partial, to: destination)
        finished = true
    }
}
