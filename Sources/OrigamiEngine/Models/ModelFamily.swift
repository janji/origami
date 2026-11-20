import CoreFoundation
import Foundation

/// The architectures Origami serves. A checkpoint's family is what its own files say: `model_type` or
/// `architectures` in config.json, `general.architecture` in a GGUF. It is never inferred from a repo name, a hash, a
/// publisher's tensor types or shard layout, so fine-tunes, abliterations and requantizations of an architecture are
/// the same family, and its geometry is read from the file (`ModelGeometry`).
public enum ModelFamily: String, Codable, Sendable, CaseIterable {
    case qwen = "qwen3.8-flash-next"
    case glm = "glm-5.3-flash"
    case deepseek = "deepseek-v4.1-flash"

    public var displayName: String {
        switch self {
        case .qwen: return "Qwen3.8-Flash-Next"
        case .glm: return "GLM-5.3-Flash"
        case .deepseek: return "DeepSeek-V4.1-Flash"
        }
    }

    static let configTypes: [String: ModelFamily] = [
        "qwen4_exp": .qwen, "qwen4_exp_text": .qwen, "glm5_next": .glm, "glm5_next_text": .glm,
        "deepseek_v41": .deepseek, "deepseek_v41_text": .deepseek,
    ]
    static let configArchitectures: [String: ModelFamily] = [
        "Qwen4ExpForConditionalGeneration": .qwen, "Qwen4ExpForCausalLM": .qwen,
        "Glm5NextForConditionalGeneration": .glm, "Glm5NextForCausalLM": .glm,
        "DeepseekV41ForCausalLM": .deepseek, "DeepseekV41ForConditionalGeneration": .deepseek,
    ]
    /// `general.architecture` as converters write it: llama.cpp's DeepSeek 4.1 branch and DwarfStar (`glm5-next`), mainline
    /// llama.cpp (`glm5next`) and others (`glm5_next`).
    static let ggufArchitectures: [String: ModelFamily] = [
        "qwen4exp": .qwen, "glm5-next": .glm, "glm5next": .glm, "glm5_next": .glm,
        "deepseek41": .deepseek,
    ]

    /// The family a config.json describes (`model_type` at the top or in `text_config`, else `architectures`).
    public static func detect(config: [String: Any]) -> ModelFamily? {
        let text = config["text_config"] as? [String: Any] ?? [:]
        for type in [config["model_type"], text["model_type"]].compactMap({ $0 as? String }) {
            if let family = configTypes[type] { return family }
        }
        return (config["architectures"] as? [String] ?? []).lazy.compactMap { configArchitectures[$0] }.first
    }

    package static func detect(gguf metadata: [String: GGUFSource.Value]) -> ModelFamily? {
        guard case .string(let architecture)? = metadata["general.architecture"] else { return nil }
        return ggufArchitectures[architecture]
    }

    /// What a config or GGUF calls its architecture, for a refusal that names it.
    static func describe(config: [String: Any]) -> String {
        let text = config["text_config"] as? [String: Any] ?? [:]
        return (config["model_type"] as? String ?? text["model_type"] as? String).map { "model_type \($0)" }
            ?? (config["architectures"] as? [String])?.first.map { "architecture \($0)" } ?? "no model_type"
    }
}

/// One checkpoint's shape, read from its config.json or its GGUF metadata (llama.cpp's and DwarfStar's key names).
/// Values keep their config.json names (`values`, `lists`) so a family's kernels read the same keys whichever
/// file they came from; the common ones are typed.
public struct ModelGeometry: Codable, Sendable, Equatable {
    /// The token mixer of a trunk layer: recurrent (Qwen GDN, GLM KDA) or attention (Qwen QSA, GLM DSA, DeepSeek
    /// compressed sparse attention).
    public enum Mixer: String, Codable, Sendable { case recurrent, attention }

    public let family: ModelFamily
    /// Trunk layers, without multi-token-prediction blocks.
    public let layers: Int
    public let hidden: Int
    public let vocab: Int
    /// Routed experts per MoE layer, how many a token uses, and the shared experts beside them.
    public let experts: Int
    public let expertsPerToken: Int
    public let sharedExperts: Int
    /// Inner width of one routed expert (gate and up are hidden x width, down width x hidden).
    public let expertWidth: Int
    /// Per trunk layer.
    public let mixers: [Mixer]
    /// Per trunk layer: true for routed experts, false for a dense feed-forward block.
    public let moe: [Bool]
    public let hyperConnections: Int
    public let mtpLayers: Int
    public let context: Int
    /// The family's other numbers and lists by their config.json names.
    public let values: [String: Double]
    public let lists: [String: [Int]]

    public var moeLayers: [Int] { moe.indices.filter { moe[$0] } }
    /// Weights in one routed expert (gate, up and down).
    public var expertParameters: Int { 3 * hidden * expertWidth }
    public func int(_ key: String) -> Int? { values[key].map { Int($0) } }

    /// A missing or out-of-range value in a checkpoint of a known family: it cannot be planned or run.
    /// A whole-number setting a family engine sizes or divides by: `fallback` when absent, refused outside `range`.
    package func int(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
        guard let value = values[key] else { return fallback }
        guard value == value.rounded(), value >= Double(range.lowerBound), value <= Double(range.upperBound) else {
            throw Self.incompatible(family, "\(key) is \(value); the engine reads \(range.lowerBound)...\(range.upperBound)")
        }
        return Int(value)
    }

    static func incompatible(_ family: ModelFamily, _ what: String) -> EngineError {
        EngineError("This \(family.displayName) checkpoint cannot run: \(what)", code: "model_incompatible")
    }

    /// From a config.json (a `text_config` block is read when present).
    public init(config: [String: Any]) throws {
        guard let family = ModelFamily.detect(config: config) else {
            throw EngineError(
                "Origami does not know this architecture (\(ModelFamily.describe(config: config)))", code: "model_incompatible")
        }
        let text = config["text_config"] as? [String: Any] ?? config
        func number(_ key: String) -> Double? {
            var node: Any? = text
            for part in key.split(separator: ".") { node = (node as? [String: Any])?[String(part)] }
            guard let value = node as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else {
                return (node as? Bool).map { $0 ? 1 : 0 }
            }
            return value.doubleValue
        }
        func list(_ key: String) -> [Int]? {
            var node: Any? = text
            for part in key.split(separator: ".") { node = (node as? [String: Any])?[String(part)] }
            return (node as? [NSNumber])?.map(\.intValue)
        }
        var values: [String: Double] = [:]
        var lists: [String: [Int]] = [:]
        for key in Self.configNumbers[family]! { values[key] = number(key) }
        for key in Self.configLists[family]! { lists[key] = list(key) }
        // Layer kinds as config.json names them, normalised to the two mixers.
        let kinds = (text["layer_types"] as? [String])?.map { $0 == "linear_attention" ? Mixer.recurrent : .attention }
        let mlp = (text["mlp_layer_types"] as? [String])?.map { $0 != "dense" }
        try Self.bound(values, family)
        try self.init(family: family, values: values, lists: lists, mixers: kinds, moe: mlp)
    }

    /// From a GGUF: its metadata, with the shapes llama.cpp does not always write as keys (vocabulary, width, experts)
    /// taken from the embedding and expert tensors.
    package init(gguf: GGUFSource) throws {
        var shapes: [String: Double] = [:]
        if let embed = gguf.tensors["token_embd.weight"], embed.shape.count == 2 {
            shapes["vocab_size"] = Double(embed.shape[0])
            shapes["hidden_size"] = Double(embed.shape[1])
        }
        if let bank = gguf.tensors.values.first(where: { $0.name.hasSuffix(".ffn_gate_exps.weight") }), bank.shape.count == 3 {
            shapes["experts"] = Double(bank.shape[0])
            shapes["moe_intermediate_size"] = Double(bank.shape[1])
        }
        // Each layer's mixer from its tensors, for headers that mark it nowhere else (a recurrent KDA layer has
        // ssm_*/kda_* tensors, an attention layer the latent q/kv projections).
        // DeepSeek's source layers likewise, for headers without the keys (vcruz305's converter): a key/value source
        // carries the compressor, an index source the indexer's query.
        var tensorMixers: [Int: Mixer] = [:]
        var tensorLists: [String: [Int]] = [:]
        for name in gguf.tensors.keys where name.hasPrefix("blk.") {
            let parts = name.split(separator: ".", maxSplits: 2)
            guard parts.count == 3, let layer = Int(parts[1]) else { continue }
            if ["ssm_a", "ssm_conv1d_q.weight", "kda_q.weight", "kda_q_conv.weight"].contains(parts[2]) {
                tensorMixers[layer] = .recurrent
            } else if ["attn_q_a.weight", "attn_kv_a_mqa.weight"].contains(parts[2]) {
                tensorMixers[layer] = .attention
            }
            if parts[2] == "attn_compressor_kv.weight" { tensorLists["kv_source_layer_ids", default: []].append(layer) }
            if parts[2] == "indexer.attn_q_b.weight" { tensorLists["index_source_layer_ids", default: []].append(layer) }
        }
        try self.init(gguf: gguf.metadata, shapes: shapes, tensorMixers: tensorMixers, tensorLists: tensorLists.mapValues { $0.sorted() })
    }

    /// From GGUF metadata; llama.cpp and DwarfStar spell several keys differently, so each value has a few names.
    /// `shapes` and `tensorLists` fill values the metadata lacks.
    package init(
        gguf metadata: [String: GGUFSource.Value], shapes: [String: Double] = [:], tensorMixers: [Int: Mixer] = [:],
        tensorLists: [String: [Int]] = [:]
    ) throws {
        guard let family = ModelFamily.detect(gguf: metadata) else {
            var architecture = "missing"
            if case .string(let value)? = metadata["general.architecture"] { architecture = value }
            throw EngineError(
                "Origami does not know this architecture (GGUF general.architecture \(architecture))", code: "model_incompatible")
        }
        var prefix = ""
        if case .string(let value)? = metadata["general.architecture"] { prefix = value + "." }
        // DwarfStar embeds the checkpoint's config.json: the most faithful source when present.
        if case .string(let text)? = metadata[prefix + "config"],
            let config = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
            ModelFamily.detect(config: config) == family, let geometry = try? ModelGeometry(config: config)
        {
            self = geometry
            return
        }
        func number(_ names: [String]) -> Double? {
            for name in names {
                switch metadata[prefix + name] {
                case .int(let v)?: return Double(v)
                case .uint(let v)?: return Double(v)
                case .float(let v)?: return v
                case .bool(let v)?: return v ? 1 : 0
                case .array(let items)?:
                    if let first = items.first?.integer { return Double(first) }
                    if case .float(let v)? = items.first { return v }
                default: continue
                }
            }
            return nil
        }
        func list(_ names: [String]) -> [Int]? {
            for name in names { if let values = metadata[prefix + name]?.integers { return values.map { Int($0) } } }
            return nil
        }
        var values: [String: Double] = [:]
        var lists: [String: [Int]] = [:]
        // Each value under its GGUF names, then under its config.json name (DwarfStar writes those as keys).
        for (key, names) in Self.ggufNumbers[family]! { values[key] = number(names + [key]) }
        for key in Self.configNumbers[family]! where values[key] == nil { values[key] = number([key]) }
        for key in ["vocab_size", "hidden_size", "moe_intermediate_size"] where values[key] == nil { values[key] = shapes[key] }
        let expertsKey = family == .qwen ? "num_experts" : "n_routed_experts"
        if values[expertsKey] == nil { values[expertsKey] = shapes["experts"] }
        for (key, names) in Self.ggufLists[family]! { lists[key] = list(names + [key]) }
        for key in Self.configLists[family]! where lists[key] == nil { lists[key] = list([key]) ?? tensorLists[key] }
        if let blocks = number(["block_count"]) {
            // llama.cpp's block_count counts the multi-token-prediction blocks after the trunk; DwarfStar adds trunk_block_count.
            values["num_hidden_layers"] = number(["trunk_block_count"]) ?? blocks - (values["num_nextn_predict_layers"] ?? 0)
        }
        try Self.bound(values, family)
        let layers = Int(values["num_hidden_layers"] ?? 0)
        guard (0...Self.countLimit).contains(layers) else { throw Self.incompatible(family, "num_hidden_layers is \(layers)") }
        var mixers: [Mixer]?
        if let kinds = list(["layer_types"]) {
            mixers = kinds.prefix(layers).map { $0 == 0 ? .recurrent : .attention }
        } else if let kv = list(["attention.head_count_kv"]), kv.count >= layers {
            mixers = kv.prefix(layers).map { $0 == 0 ? .recurrent : .attention }  // llama.cpp: recurrent layers have no KV heads
        } else if layers > 0, (0..<layers).allSatisfy({ tensorMixers[$0] != nil }) {
            mixers = (0..<layers).map { tensorMixers[$0]! }
        }
        var moe: [Bool]?
        if let dense = values["first_k_dense_replace"] { moe = (0..<layers).map { $0 >= Int(dense) } }
        try self.init(family: family, values: values, lists: lists, mixers: mixers, moe: moe)
    }

    /// Header and config numbers are untrusted: every one must be finite and of a size no model has outgrown, so the
    /// `Int(...)` conversions after this cannot trap.
    static func bound(_ values: [String: Double], _ family: ModelFamily) throws {
        if let (key, value) = values.first(where: { !$0.value.isFinite || abs($0.value) > 1e12 }) {
            throw incompatible(family, "\(key) is \(value)")
        }
    }
    /// The largest layer, expert, width or vocabulary count a header may give.
    static let countLimit = 1 << 24

    private init(family: ModelFamily, values: [String: Double], lists: [String: [Int]], mixers: [Mixer]?, moe: [Bool]?) throws {
        func need(_ key: String) throws -> Int {
            guard let value = values[key], value > 0, value == value.rounded(), value <= Double(Self.countLimit) else {
                throw Self.incompatible(family, "\(key) is \(values[key].map { "\($0)" } ?? "missing")")
            }
            return Int(value)
        }
        self.family = family
        layers = try need("num_hidden_layers")
        hidden = try need("hidden_size")
        vocab = try need("vocab_size")
        experts = try need(family == .qwen ? "num_experts" : "n_routed_experts")
        expertsPerToken = try need("num_experts_per_tok")
        sharedExperts = family == .qwen ? 1 : Int(values["n_shared_experts"] ?? 1)
        expertWidth = try need("moe_intermediate_size")
        hyperConnections = Int(values[family == .qwen ? "hc_count" : "hc_mult"] ?? 1)
        mtpLayers = Int(values["num_nextn_predict_layers"] ?? 0)
        context = Int(values["max_position_embeddings"] ?? Double(ModelContract.maximumContext))
        guard expertsPerToken <= experts else { throw Self.incompatible(family, "\(expertsPerToken) experts per token of \(experts)") }
        switch family {
        case .qwen:
            let interval = Int(values["full_attention_interval"] ?? 4)
            guard interval >= 1 else { throw Self.incompatible(family, "full_attention_interval is \(interval)") }
            self.mixers = mixers ?? (0..<layers).map { $0 % interval == interval - 1 ? .attention : .recurrent }
            self.moe = moe ?? Array(repeating: true, count: layers)
        case .glm:
            guard let mixers else { throw Self.incompatible(family, "its layer types are missing") }
            self.mixers = mixers
            self.moe = moe ?? (0..<layers).map { $0 >= Int(values["first_k_dense_replace"] ?? 0) }
        case .deepseek:
            self.mixers = Array(repeating: .attention, count: layers)
            self.moe = moe ?? Array(repeating: true, count: layers)
        }
        guard self.mixers.count == layers, self.moe.count == layers else {
            throw Self.incompatible(family, "it lists \(self.mixers.count) layer types for \(layers) layers")
        }
        self.values = values
        self.lists = lists
    }

    /// Qwen3.8-Flash-Next as `ModelContract` pins it (the kernels are specialised to these numbers).
    public static let qwen: ModelGeometry = {
        var config: [String: Any] = ["model_type": "qwen4_exp"]
        for (key, value) in ModelContract.geometry { config[key] = value }
        return try! ModelGeometry(config: config)
    }()

    /// config.json keys per family (dotted keys are nested).
    static let configNumbers: [ModelFamily: [String]] = [
        .qwen: Array(ModelContract.geometry.keys) + ["max_position_embeddings", "num_nextn_predict_layers"],
        .glm: [
            "hidden_size", "num_hidden_layers", "vocab_size", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
            "moe_intermediate_size", "intermediate_size", "first_k_dense_replace", "num_attention_heads", "q_lora_rank",
            "kv_lora_rank", "qk_head_dim", "qk_rope_head_dim", "v_head_dim", "index_n_heads", "index_head_dim", "index_topk",
            "index_kpool", "linear_attn_config.num_heads", "linear_attn_config.head_dim", "linear_attn_config.short_conv_kernel_size",
            "linear_attn_config.gate_lower_bound", "hc_mult", "hc_sinkhorn_iters", "hc_eps", "routed_scaling_factor",
            "swiglu_limit", "rms_norm_eps", "num_nextn_predict_layers", "max_position_embeddings", "norm_topk_prob",
        ],
        .deepseek: [
            "hidden_size", "num_hidden_layers", "vocab_size", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
            "moe_intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "qk_rope_head_dim",
            "q_lora_rank", "o_lora_rank", "o_groups", "sliding_window", "compress_rope_theta", "rope_theta",
            "rope_scaling.factor", "rope_scaling.beta_fast", "rope_scaling.beta_slow", "rope_scaling.original_max_position_embeddings",
            "index_n_heads", "index_head_dim", "index_topk", "candidate_source_layer_id", "candidate_topk_blocks",
            "candidate_block_size", "hc_mult", "hc_sinkhorn_iters", "hc_eps", "engram_max_ngram_size", "engram_n_heads",
            "engram_head_dim", "engram_pad_token_id", "routed_scaling_factor", "swiglu_limit", "rms_norm_eps",
            "num_nextn_predict_layers", "max_position_embeddings", "norm_topk_prob",
        ],
    ]
    static let configLists: [ModelFamily: [String]] = [
        .qwen: ["ple_layer_ids"], .glm: [],
        .deepseek: ["compress_ratios", "kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"],
    ]
    /// GGUF keys (after `<architecture>.`) for each config.json key; the first present wins.
    static let ggufNumbers: [ModelFamily: [String: [String]]] = [
        .qwen: [
            "hidden_size": ["embedding_length"], "vocab_size": ["vocab_size"], "num_experts": ["expert_count"],
            "num_experts_per_tok": ["expert_used_count"], "moe_intermediate_size": ["expert_feed_forward_length"],
            "num_attention_heads": ["attention.head_count"], "hc_count": ["hyper_connection.count"],
            "max_position_embeddings": ["context_length"], "num_nextn_predict_layers": ["nextn_predict_layers"],
        ],
        .glm: [
            "hidden_size": ["embedding_length"], "vocab_size": ["vocab_size"], "n_routed_experts": ["expert_count"],
            "num_experts_per_tok": ["expert_used_count"], "n_shared_experts": ["expert_shared_count"],
            "moe_intermediate_size": ["expert_feed_forward_length"], "intermediate_size": ["feed_forward_length"],
            "first_k_dense_replace": ["leading_dense_block_count"], "num_attention_heads": ["attention.head_count"],
            "q_lora_rank": ["attention.q_lora_rank"], "kv_lora_rank": ["attention.kv_lora_rank"],
            "qk_head_dim": ["attention.key_length_mla", "attention.key_length"], "qk_rope_head_dim": ["rope.dimension_count"],
            "v_head_dim": ["attention.value_length_mla", "attention.value_length"],
            "index_n_heads": ["attention.indexer.head_count"], "index_head_dim": ["attention.indexer.key_length"],
            "index_topk": ["attention.indexer.top_k"],
            "index_kpool": ["attention.indexer.kpool", "attention.indexer.pool_size", "attention.indexer.block_size"],
            "linear_attn_config.num_heads": ["linear_attention.head_count", "attention.head_count"],
            "linear_attn_config.head_dim": ["linear_attention.head_dimension", "kda.head_dim", "ssm.state_size"],
            "linear_attn_config.short_conv_kernel_size": ["linear_attention.conv_kernel", "ssm.conv_kernel"],
            "linear_attn_config.gate_lower_bound": ["linear_attention.gate_lower_bound", "kda.gate_lower_bound"],
            "hc_mult": ["hyper_connection.count"], "hc_sinkhorn_iters": ["hyper_connection.sinkhorn_iterations"],
            "hc_eps": ["hyper_connection.epsilon"], "routed_scaling_factor": ["expert_weights_scale"],
            "swiglu_limit": ["swiglu_limit", "swiglu_clamp_exp"], "rms_norm_eps": ["attention.layer_norm_rms_epsilon"],
            "num_nextn_predict_layers": ["nextn_predict_layers"], "max_position_embeddings": ["context_length"],
            "norm_topk_prob": ["expert_weights_norm"],
        ],
        .deepseek: [
            "hidden_size": ["embedding_length"], "vocab_size": ["vocab_size"], "n_routed_experts": ["expert_count"],
            "num_experts_per_tok": ["expert_used_count"], "n_shared_experts": ["expert_shared_count"],
            "moe_intermediate_size": ["expert_feed_forward_length"], "num_attention_heads": ["attention.head_count"],
            "num_key_value_heads": ["attention.head_count_kv"], "head_dim": ["attention.key_length"],
            "qk_rope_head_dim": ["rope.dimension_count"], "q_lora_rank": ["attention.q_lora_rank"],
            "o_lora_rank": ["attention.output_lora_rank"], "o_groups": ["attention.output_group_count"],
            "sliding_window": ["attention.sliding_window"], "compress_rope_theta": ["attention.compress_rope_freq_base"],
            "rope_theta": ["rope.freq_base"], "rope_scaling.factor": ["rope.scaling.factor"],
            "rope_scaling.beta_fast": ["rope.scaling.yarn_beta_fast"], "rope_scaling.beta_slow": ["rope.scaling.yarn_beta_slow"],
            "rope_scaling.original_max_position_embeddings": ["rope.scaling.original_context_length"],
            "index_n_heads": ["attention.indexer.head_count"], "index_head_dim": ["attention.indexer.key_length"],
            "index_topk": ["attention.indexer.top_k"], "candidate_source_layer_id": ["attention.indexer.candidate_source_layer"],
            "candidate_topk_blocks": ["attention.indexer.candidate_top_k_blocks"],
            "candidate_block_size": ["attention.indexer.candidate_block_size"], "hc_mult": ["hyper_connection.count"],
            "hc_sinkhorn_iters": ["hyper_connection.sinkhorn_iterations"], "hc_eps": ["hyper_connection.epsilon"],
            "engram_max_ngram_size": ["ple.ngram_size", "engram.max_ngram_size"],
            "engram_n_heads": ["ple.heads_per_ngram", "engram.head_count"],
            "engram_head_dim": ["embedding_length_per_layer_input", "engram.key_length"],
            "engram_pad_token_id": ["engram.pad_token_id", "engram.pad_id"],
            "routed_scaling_factor": ["expert_weights_scale"], "swiglu_limit": ["swiglu_clamp_exp"],
            "rms_norm_eps": ["attention.layer_norm_rms_epsilon"], "num_nextn_predict_layers": ["nextn_predict_layers"],
            "max_position_embeddings": ["context_length"], "norm_topk_prob": ["expert_weights_norm"],
        ],
    ]
    static let ggufLists: [ModelFamily: [String: [String]]] = [
        .qwen: ["ple_layer_ids": ["ple.layers"]], .glm: [:],
        .deepseek: [
            "compress_ratios": ["attention.compress_ratios"], "kv_source_layer_ids": ["attention.kv_source_layers"],
            "index_source_layer_ids": ["attention.indexer.source_layers"], "engram_layer_ids": ["ple.layers", "engram.layer_ids"],
        ],
    ]
}

/// What a tensor is for, whatever the naming (Hugging Face, MLX, llama.cpp's GGUF or DwarfStar's): the compatibility
/// check reads tensor types by role and refuses a type only where no kernel reads it.
public enum TensorRole: String, Codable, Sendable, CaseIterable {
    case experts = "routed experts"
    case sharedExpert = "shared expert"
    case router = "router"
    case denseFFN = "dense feed-forward"
    case recurrent = "recurrent mixer"
    case attention = "attention"
    case hyperConnections = "hyper-connections"
    case table = "n-gram table"
    case embeddings = "token embeddings"
    case output = "output head"
    case norms = "norms"
    case mtp = "multi-token prediction"
    case vision = "vision"
    case other = "other"

    /// Shapes in GGUF, Hugging Face and MLX naming; DwarfStar's names are llama.cpp's except for the recurrent mixer
    /// (`kda_*`) and the indexer pool (`indexer.pool_*`).
    public static func of(_ name: String) -> TensorRole {
        let n = name.lowercased()
        func has(_ parts: String...) -> Bool { parts.contains { n.contains($0) } }
        if has("visual", "vision_tower", "vision_model", "mm_projector", "mmproj") || n.hasPrefix("v.") || n.hasPrefix("mm.") {
            return .vision
        }
        if has("nextn", ".mtp.", "mtp.", "eh_proj", ".enorm", ".hnorm", "shared_head", "dspark") { return .mtp }
        if has("_exps", ".experts.", "switch_mlp") { return .experts }
        if has("_shexp", "shared_expert") { return .sharedExpert }
        if has("ffn_gate_inp", "exp_probs_b", "mlp.gate.", "e_score_correction", ".router") { return .router }
        if has("per_layer_token_embd", "engram", ".ple.", "ngram_embedding") { return .table }
        if has("hc_", "hyper_connection") { return .hyperConnections }
        if has("kda_", "ssm_", "linear_attn", "a_log", "dt_bias", "conv1d", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", ".b_proj") {
            return .recurrent
        }
        if has("attn_q", "attn_k", "attn_v", "attn_o", "attn_kv", "attn_sinks", "attn_compressor", "self_attn", "indexer", ".attn.") {
            return .attention
        }
        if has("token_embd", "embed_tokens", "wte") { return .embeddings }
        if n.hasPrefix("output.") || has("lm_head") { return .output }
        if has("norm") { return .norms }
        if has("ffn_gate", "ffn_up", "ffn_down", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj") { return .denseFFN }
        return .other
    }
}
