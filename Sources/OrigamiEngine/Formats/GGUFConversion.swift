import CryptoKit
import Foundation

/// A llama.cpp GGUF of Qwen3.8-Flash-Next converted back into Origami's MLX 4-bit layout. Each tensor is
/// dequantized and requantized (a 4-bit GGUF is quantized twice; Q8_0 is close to lossless) and llama.cpp's conversion
/// (`conversion/qwen4exp.py` over its Qwen3-Next base) is undone: value heads of the linear attention tiled across key
/// heads go back to grouped order (rows of in_proj_qkv's V part, in_proj_z, in_proj_a/b and the conv channels, columns
/// of out_proj, elements of A_log and dt_bias), A_log stored as -exp(A_log) becomes log(-x), the conv kernels get their
/// last dimension back, merged pairs ([qkv | gate], [beta | alpha], [hc down | inject]) split, the indexer's q and k join,
/// and the PLE table splits back into its 128 shards. Norms are stored the same way in both (1 + w). The GGUF keeps no
/// tokenizer files Origami reads, so `support` supplies the official config.json and tokenizer; the GGUF's own chat
/// template wins over the official one. A GGUF whose routed experts are below Origami's 4-bit size keeps them as they are
/// (`keepsExpertsNative`, `GGUFExperts`).
extension CheckpointConverter {
    /// Whether this GGUF's routed experts stay in its own encodings, read from it while serving (`GGUFExperts`), instead
    /// of being requantized: their banks average under 4.5 bits per weight (Origami's 4-bit records are 4.5, so
    /// requantizing would cost quality and size), or it keeps fewer than the model's `experts` (a pruned model, which the
    /// 4-bit layout cannot hold). ggml's kernels must run every bank's type; otherwise the banks are requantized.
    package static func keepsExpertsNative(_ gguf: GGUFSource, experts total: Int = ModelContract.experts) -> Bool {
        let banks = gguf.tensors.values.filter {
            $0.name.hasPrefix("blk.") && $0.name.hasSuffix("_exps.weight") && !$0.name.contains("nextn")
        }
        let weights = banks.reduce(0.0) { $0 + Double($1.shape.reduce(1, *)) }
        let bytes = banks.reduce(0.0) { $0 + Double($1.length) }
        guard !banks.isEmpty, weights > 0,
            banks.allSatisfy({ GGMLMatrixKernels.types.contains($0.type) && (GGUFQuant.sizes[$0.type]?.block ?? 1) > 1 })
        else { return false }
        let pruned = (gguf.metadata["qwen4exp.expert_count"]?.integer).map { $0 < Int64(total) } ?? false
        return bytes * 8 / weights < 4.5 || pruned
    }

    /// Why a GGUF would not convert, told without the support files: another architecture, folded rotations, or an
    /// encoding Origami does not read; nil when it converts.
    package static func ggufProblem(_ gguf: GGUFSource) -> String? {
        guard case .string("qwen4exp")? = gguf.metadata["general.architecture"] else { return "it is not Qwen3.8-Flash-Next (qwen4exp)" }
        if gguf.metadata.keys.contains(where: { $0.hasPrefix("prism.hadamard.") }) {
            return "Hadamard rotations are folded into its weights"
        }
        let unread = Set(
            gguf.tensors.values.filter { !$0.name.contains("nextn") && !GGUFQuant.readable.contains($0.type) }.map {
                GGUFQuant.name($0.type)
            })
        return unread.isEmpty
            ? nil
            : "its \(unread.sorted().joined(separator: ", ")) tensors use encodings Origami does not read; a Q8_0, Q6_K, Q5_K or Q4_K build converts"
    }

    package init(gguf: GGUFSource, support: URL, checkContract: Bool = true) throws {
        func refuse(_ why: String) -> EngineError { EngineError("Origami cannot convert this GGUF: \(why)", code: "model_incompatible") }
        let meta = gguf.metadata
        guard case .string(let architecture)? = meta["general.architecture"], architecture == "qwen4exp" else {
            throw refuse(
                "it is not Qwen3.8-Flash-Next (architecture \(meta["general.architecture"].map { "\($0)" } ?? "missing"), not qwen4exp)")
        }
        if meta.keys.contains(where: { $0.hasPrefix("prism.hadamard.") }) {
            throw refuse("Hadamard rotations are folded into its weights, which Origami does not undo")
        }
        func int(_ key: String) throws -> Int {
            // Positive and small (the real values are at most a few thousand): they size the reorders and ranges below.
            guard let value = meta["qwen4exp." + key]?.integer, (1...65_536).contains(value) else {
                throw refuse("its metadata lacks a usable qwen4exp.\(key)")
            }
            return Int(value)
        }
        let kHeads = try int("ssm.group_count")
        let vHeads = try int("ssm.time_step_rank")
        let keyDim = try int("ssm.state_size")
        let inner = try int("ssm.inner_size")
        let interval = try int("full_attention_interval")
        let lowRank = try int("hyper_connection.low_rank")
        let streams = try int("hyper_connection.count")
        let qk = 2 * kHeads * keyDim
        // The reorders are as long as layer 0's conv channels, a tensor whose size the file bounds.
        guard vHeads % kHeads == 0, inner % vHeads == 0, gguf.tensors["blk.0.ssm_conv1d.weight"]?.shape.first == qk + inner else {
            throw refuse("its linear-attention geometry is inconsistent")
        }
        let valueDim = inner / vHeads
        let perK = vHeads / kHeads
        let pleLayers = meta["qwen4exp.ple.layers"]?.integers?.map(Int.init) ?? []

        let configData = try Data(contentsOf: support.appendingPathComponent("config.json"))
        guard var config = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any] else {
            throw refuse("the support config.json is not JSON")
        }
        let text = config["text_config"] as? [String: Any] ?? config
        let layers = text["num_hidden_layers"] as? Int ?? ModelContract.layers
        let experts = try int("expert_count")
        let configExperts = text["num_experts"] as? Int ?? 0
        let native = Self.keepsExpertsNative(gguf, experts: configExperts)
        guard try int("block_count") >= layers, try int("embedding_length") == text["hidden_size"] as? Int ?? 0, experts <= configExperts
        else {
            throw refuse("its geometry differs from Qwen3.8-Flash-Next's")
        }
        guard experts == configExperts || native else {
            throw refuse(
                "it keeps \(experts) of \(configExperts) experts, which Origami runs only in the GGUF's own encodings, and its expert banks use one ggml's kernels do not run"
            )
        }
        // Native experts: per layer, gate, up and down as (path, offset, stride, type, rows, columns) for the manifest.
        var banks: [[(path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int)]] = []

        var shards: [Shard] = []
        var overrides: [String: Any] = [:]
        var used = Set<String>()
        func tensor(_ name: String) throws -> GGUFSource.Tensor {
            guard let t = gguf.tensors[name] else { throw refuse("it lacks \(name)") }
            guard GGUFQuant.readable.contains(t.type) else {
                throw refuse("\(name) is \(GGUFQuant.name(t.type)), which Origami does not read; a Q8_0, Q6_K, Q5_K or Q4_K build converts")
            }
            used.insert(name)
            return t
        }
        /// The tensor, or `rows` of it seen as [rows, columns].
        func span(_ t: GGUFSource.Tensor, rows: Range<Int>? = nil) throws -> TensorSpan {
            let columns = t.shape.last ?? 1
            let size = GGUFQuant.sizes[t.type]!
            let rowBytes = Int64(columns / size.block * size.bytes)
            if let rows, t.shape.count != 2 || rows.upperBound > t.shape[0] { throw refuse("\(t.name) has no rows \(rows)") }
            guard let rows else {
                return TensorSpan(file: t.file, dtype: "GGUF:\(t.type)", offset: t.offset, length: t.length, shape: t.shape)
            }
            return TensorSpan(
                file: t.file, dtype: "GGUF:\(t.type)", offset: t.offset + Int64(rows.lowerBound) * rowBytes,
                length: Int64(rows.count) * rowBytes, shape: [rows.count, columns])
        }
        func quantize(
            _ name: String, _ out: String, rows: Range<Int>? = nil, rowOrder: [Int]? = nil, columnOrder: [Int]? = nil,
            group: Int = 64, into shard: inout Shard
        ) throws {
            let source = try span(try tensor(name), rows: rows)
            try check(source, name, rowOrder, columnOrder)
            if group != 64 { overrides[String(out.dropFirst("language_model.".count))] = ["group_size": group, "bits": 4] }
            shard.works.append(
                Work(
                    source: source, kind: .quantize(group: group),
                    outputs: try Self.quantized(out, source.shape, group, of: name, to: &shard),
                    rowOrder: rowOrder, columnOrder: columnOrder))
        }
        func floats(
            _ name: String, _ out: String, shape: [Int]? = nil, rows: Range<Int>? = nil, rowOrder: [Int]? = nil,
            columnOrder: [Int]? = nil, negativeLog: Bool = false, into shard: inout Shard
        ) throws {
            let source = try span(try tensor(name), rows: rows)
            try check(source, name, rowOrder, columnOrder)
            shard.works.append(
                Work(
                    source: source, kind: .floats, outputs: [Self.add(out, "BF16", shape ?? source.shape, to: &shard)],
                    rowOrder: rowOrder, columnOrder: columnOrder, negativeLog: negativeLog))
        }
        /// A reorder covers exactly the rows (columns) it permutes.
        func check(_ source: TensorSpan, _ name: String, _ rowOrder: [Int]?, _ columnOrder: [Int]?) throws {
            guard rowOrder.map({ $0.count == source.shape.dropLast().reduce(1, *) }) ?? true,
                columnOrder.map({ $0.count == source.shape.last }) ?? true
            else { throw refuse("\(name) does not match its linear-attention geometry") }
        }
        /// Grouped position r of `count` values in heads of `dim` comes from llama.cpp's tiled position.
        func grouped(_ count: Int, _ dim: Int) -> [Int] {
            (0..<count).map { r in
                let head = r / dim
                return ((head % perK) * kHeads + head / perK) * dim + r % dim
            }
        }
        let channels = Array(0..<qk) + grouped(inner, valueDim).map { $0 + qk }

        for i in 0..<layers {
            var shard = Shard(file: String(format: "model-layer-%02d.safetensors", i))
            let b = "blk.\(i)."
            let l = "language_model.model.layers.\(i)."
            for (side, hc) in [("attn", "attn_hyper_connection."), ("ffn", "mlp_hyper_connection.")] {
                try floats(b + "hc_\(side)_norm.weight", l + hc + "hc_norm.weight", into: &shard)
                if gguf.tensors[b + "hc_\(side)_down_inject.weight"] != nil {
                    try quantize(b + "hc_\(side)_down_inject.weight", l + hc + "input_mix_weight_down", rows: 0..<lowRank, into: &shard)
                    try floats(
                        b + "hc_\(side)_down_inject.weight", l + hc + "block_inject_weight.weight", rows: lowRank..<(lowRank + streams),
                        into: &shard)
                } else {
                    try quantize(b + "hc_\(side)_down.weight", l + hc + "input_mix_weight_down", into: &shard)
                    try floats(b + "hc_\(side)_inject.weight", l + hc + "block_inject_weight.weight", into: &shard)
                }
                try quantize(b + "hc_\(side)_up.weight", l + hc + "input_mix_weight_up", into: &shard)
            }
            if (i + 1) % interval != 0 {
                let a = l + "linear_attn."
                if gguf.tensors[b + "attn_qkv_gate.weight"] != nil {
                    try quantize(b + "attn_qkv_gate.weight", a + "in_proj_qkv", rows: 0..<(qk + inner), rowOrder: channels, into: &shard)
                    try quantize(
                        b + "attn_qkv_gate.weight", a + "in_proj_z", rows: (qk + inner)..<(qk + 2 * inner),
                        rowOrder: grouped(inner, valueDim), into: &shard)
                } else {
                    try quantize(b + "attn_qkv.weight", a + "in_proj_qkv", rowOrder: channels, into: &shard)
                    try quantize(b + "attn_gate.weight", a + "in_proj_z", rowOrder: grouped(inner, valueDim), into: &shard)
                }
                if gguf.tensors[b + "ssm_beta_alpha.weight"] != nil {
                    try floats(
                        b + "ssm_beta_alpha.weight", a + "in_proj_b.weight", rows: 0..<vHeads, rowOrder: grouped(vHeads, 1), into: &shard)
                    try floats(
                        b + "ssm_beta_alpha.weight", a + "in_proj_a.weight", rows: vHeads..<(2 * vHeads), rowOrder: grouped(vHeads, 1),
                        into: &shard)
                } else {
                    try floats(b + "ssm_beta.weight", a + "in_proj_b.weight", rowOrder: grouped(vHeads, 1), into: &shard)
                    try floats(b + "ssm_alpha.weight", a + "in_proj_a.weight", rowOrder: grouped(vHeads, 1), into: &shard)
                }
                try floats(b + "ssm_a", a + "A_log", columnOrder: grouped(vHeads, 1), negativeLog: true, into: &shard)
                try floats(b + "ssm_dt.bias", a + "dt_bias", columnOrder: grouped(vHeads, 1), into: &shard)
                try floats(
                    b + "ssm_conv1d.weight", a + "conv1d.weight", shape: try tensor(b + "ssm_conv1d.weight").shape + [1],
                    rowOrder: channels, into: &shard)
                try floats(b + "ssm_norm.weight", a + "norm.weight", into: &shard)
                try quantize(b + "ssm_out.weight", a + "out_proj", columnOrder: grouped(inner, valueDim), into: &shard)
            } else {
                let s = l + "self_attn."
                for (from, to) in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")] {
                    try quantize(b + from + ".weight", s + to, into: &shard)
                }
                try floats(b + "attn_q_norm.weight", s + "q_norm.weight", into: &shard)
                try floats(b + "attn_k_norm.weight", s + "k_norm.weight", into: &shard)
                let q = try tensor(b + "indexer.q_proj.weight")
                let k = try tensor(b + "indexer.k_proj.weight")
                guard q.shape.count == 2, k.shape.count == 2, q.shape[1] == k.shape[1] else {
                    throw refuse("its indexer projections are not matrices of one width")
                }
                let joined = Self.add(s + "indexer.index_qk_proj.weight", "BF16", [q.shape[0] + k.shape[0], q.shape[1]], to: &shard)
                shard.works.append(Work(source: try span(q), kind: .floats, outputs: [joined]))
                shard.works.append(Work(source: try span(k), kind: .floats, outputs: [joined], row: q.shape[0]))
                try floats(b + "indexer.q_norm.weight", s + "indexer.q_layernorm.weight", into: &shard)
                try floats(b + "indexer.k_norm.weight", s + "indexer.k_layernorm.weight", into: &shard)
            }
            let m = l + "mlp."
            if experts < configExperts {
                // A pruned GGUF's router, padded to the engine's experts with zero rows (`ModelLayout.routedExperts` masks them).
                let router = try tensor(b + "ffn_gate_inp.weight")
                let hidden = router.shape.last ?? 1
                guard router.shape == [experts, hidden] else { throw refuse("\(router.name) is not \(experts) router rows") }
                try floats(router.name, m + "gate.weight", shape: [configExperts, hidden], into: &shard)
                let zeros = Data(count: (configExperts - experts) * hidden * 2)
                shard.works.append(
                    Work(
                        source: TensorSpan(
                            file: "", dtype: "BF16", offset: 0, length: Int64(zeros.count), shape: [configExperts - experts, hidden]),
                        kind: .literal(zeros), outputs: [shard.tensors.count - 1], byteOffset: Int64(experts * hidden * 2)))
            } else {
                try floats(b + "ffn_gate_inp.weight", m + "gate.weight", into: &shard)
            }
            try floats(
                b + "ffn_gate_inp_shexp.weight", m + "shared_expert_gate.weight",
                shape: [1, try tensor(b + "ffn_gate_inp_shexp.weight").shape.last ?? 1], into: &shard)
            for p in ["gate", "up", "down"] { try quantize(b + "ffn_\(p)_shexp.weight", m + "shared_expert.\(p)_proj", into: &shard) }
            if native {
                // The banks stay in the GGUF: part `part` of `parts` stacked row blocks of each expert's [rows, columns],
                // `stride` bytes after the previous expert's (a fused gate/up bank is two parts).
                func bank(_ name: String, part: Int = 0, of parts: Int = 1) throws -> (
                    path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int
                ) {
                    let t = try tensor(name)
                    let size = GGUFQuant.sizes[t.type]!
                    guard t.shape.count == 3, t.shape[0] == experts, t.shape[1] % parts == 0, t.shape[2] % size.block == 0,
                        Int64(experts * t.shape[1] * (t.shape[2] / size.block * size.bytes)) == t.length
                    else {
                        throw refuse("\(name) is not a bank of \(experts) experts")
                    }
                    let rows = t.shape[1] / parts
                    let rowBytes = t.shape[2] / size.block * size.bytes
                    return (
                        gguf.folder.appendingPathComponent(t.file).standardizedFileURL.path, Int(t.offset) + part * rows * rowBytes,
                        t.shape[1] * rowBytes, t.type, rows, t.shape[2]
                    )
                }
                let fused = b + "ffn_gate_up_exps.weight"
                banks.append(
                    gguf.tensors[fused] != nil
                        ? [try bank(fused, part: 0, of: 2), try bank(fused, part: 1, of: 2)]
                        : [try bank(b + "ffn_gate_exps.weight"), try bank(b + "ffn_up_exps.weight")])
                banks[banks.count - 1].append(try bank(b + "ffn_down_exps.weight"))
            } else if gguf.tensors[b + "ffn_gate_up_exps.weight"] != nil {
                let fused = try tensor(b + "ffn_gate_up_exps.weight")
                guard fused.shape.count == 3, fused.shape[1] % 2 == 0 else {
                    throw refuse("\(fused.name) is not a stack of gate and up rows")
                }
                let half = [fused.shape[0], fused.shape[1] / 2, fused.shape[2]]
                shard.works.append(
                    Work(
                        source: try span(fused), kind: .gateUp(group: 64),
                        outputs: try Self.quantized(m + "switch_mlp.gate_proj", half, 64, of: fused.name, to: &shard)
                            + Self.quantized(m + "switch_mlp.up_proj", half, 64, of: fused.name, to: &shard)))
            } else {
                try quantize(b + "ffn_gate_exps.weight", m + "switch_mlp.gate_proj", into: &shard)
                try quantize(b + "ffn_up_exps.weight", m + "switch_mlp.up_proj", into: &shard)
            }
            if !native { try quantize(b + "ffn_down_exps.weight", m + "switch_mlp.down_proj", into: &shard) }
            if pleLayers.contains(i) {
                let p = l + "ple."
                try floats(
                    b + "ple_conv1d.weight", p + "conv1d.weight", shape: try tensor(b + "ple_conv1d.weight").shape + [1], into: &shard)
                try quantize(b + "ple_key.weight", p + "key_proj", into: &shard)
                try quantize(b + "ple_value.weight", p + "value_proj", into: &shard)
                for n in ["conv", "key", "query"] { try floats(b + "ple_norm_\(n).weight", p + "norm_\(n).weight", into: &shard) }
                // The hash constants are int64 tensors in the checkpoint; llama.cpp keeps them as metadata.
                for (key, out) in [
                    ("layer_multipliers", "layer_multipliers"), ("head_offsets", "ngram_heads_offsets"),
                    ("head_vocab_sizes", "ngram_heads_vocab_sizes"),
                ] {
                    guard let values = meta["qwen4exp.ple." + key]?.integers, !values.isEmpty else {
                        throw refuse("its metadata lacks qwen4exp.ple.\(key)")
                    }
                    let data = values.map(\.littleEndian).withUnsafeBytes { Data($0) }
                    shard.works.append(
                        Work(
                            source: TensorSpan(file: "", dtype: "I64", offset: 0, length: Int64(data.count), shape: [values.count]),
                            kind: .literal(data), outputs: [Self.add(p + "ple_embedding." + out, "I64", [values.count], to: &shard)]))
                }
            }
            shards.append(shard)
        }
        var globals = Shard(file: "model-globals.safetensors")
        let g = "language_model.model."
        try quantize("token_embd.weight", g + "embed_tokens", into: &globals)
        try quantize("output.weight", "language_model.lm_head", into: &globals)
        try floats("output_hc_norm.weight", g + "hyper_connection_mixer.hc_norm.weight", into: &globals)
        try quantize("output_hc_down.weight", g + "hyper_connection_mixer.input_mix_weight_down", into: &globals)
        try quantize("output_hc_up.weight", g + "hyper_connection_mixer.input_mix_weight_up", into: &globals)
        shards.append(globals)
        if let layer = pleLayers.first {
            // The 128 n-gram shards, one table in the GGUF: eight files of sixteen.
            let table = try tensor("per_layer_token_embd.weight")
            let rows = table.shape.first ?? 0
            let parts = ModelContract.pleShards
            guard rows > 0, rows % parts == 0 else { throw refuse("its PLE table of \(rows) rows does not split into \(parts) shards") }
            for file in 0..<8 {
                var shard = Shard(file: "model-ple-\(file).safetensors")
                for k in (file * parts / 8)..<((file + 1) * parts / 8) {
                    try quantize(
                        "per_layer_token_embd.weight", "language_model.model.layers.\(layer).ple.ple_embedding.ngram_embedding.shard_\(k)",
                        rows: (k * rows / parts)..<((k + 1) * rows / parts), group: 32, into: &shard)
                }
                shards.append(shard)
            }
        }
        // Nothing may be left behind unnoticed; the MTP head (blocks past the model's, `nextn`) is dropped on purpose.
        let unknown = gguf.tensors.keys.filter { name in
            guard !used.contains(name), !name.contains("nextn") else { return false }
            let block = name.hasPrefix("blk.") ? Int(name.dropFirst(4).prefix { $0.isNumber }) : nil
            return block.map { $0 < layers } ?? true
        }
        guard unknown.isEmpty else { throw refuse("Origami does not know \(unknown.sorted().prefix(5).joined(separator: ", "))") }
        for i in shards.indices { try Self.finish(&shards[i]) }

        var quantization: [String: Any] = ["group_size": 64, "bits": 4]
        quantization.merge(overrides) { $1 }
        if native {
            quantization["expert_format"] = "gguf"
            let files = Set(banks.joined().map(\.path)).sorted().map { path -> (path: String, size: UInt64) in
                var info = stat()
                return (path, stat(path, &info) == 0 ? UInt64(info.st_size) : 0)
            }
            expertManifest = try GGUFExperts.manifest(experts: experts, files: files, layers: banks)
            if experts != configExperts {
                if var nested = config["text_config"] as? [String: Any] {
                    nested["num_experts"] = experts
                    config["text_config"] = nested
                } else {
                    config["num_experts"] = experts
                }
            }
        }
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        self.config = try ModelSource.configJSON(config, options: [.prettyPrinted, .sortedKeys])
        if checkContract { _ = try ModelContract.validate(config: self.config) }
        self.shards = shards
        source = gguf.folder
        extras = support
        if case .string(let template)? = meta["tokenizer.chat_template"] { self.template = template } else { template = nil }
        var hash = SHA256()
        hash.update(data: configData)
        // Each part's modification time: a GGUF rebuilt in place with the same recipe has the same names and offsets.
        for file in gguf.files {
            hash.update(data: Data("\(file):\(ModelSource.modificationNanoseconds(gguf.folder.appendingPathComponent(file)) ?? 0)".utf8))
        }
        for t in gguf.tensors.values.sorted(by: { $0.name < $1.name }) {
            hash.update(data: Data("\(t.name):\(t.type):\(t.file):\(t.offset)".utf8))
        }
        identity = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let types = Set(used.compactMap { gguf.tensors[$0]?.type }).map(GGUFQuant.name).sorted().joined(separator: "/")
        sourceKind = "GGUF (\(types))"
        chain =
            "gguf \(types) -> affine 4-bit group 64 (PLE n-gram tables group 32), MLX rounding"
            + (native ? "; routed experts (\(experts)) kept in the GGUF as they are" : "")
        sourceBytes = gguf.bytes
        outputBytes = UInt64(shards.reduce(Int64(0)) { $0 + $1.bytes })
    }
}
