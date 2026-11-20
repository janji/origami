import CoreFoundation
import Foundation
import Metal

/// The Qwen3.8-Flash-Next architecture the kernels are specialized for. Any checkpoint
/// of this architecture (the original release, its fine-tunes, other conversions)
/// can run if its quantization layout has kernels (`QuantizationScheme`).
public enum ModelContract {
    public static let maximumContext = 262_144
    public static let layers = 48
    public static let experts = 512
    public static let pleShards = 128
    public static let pleRowsPerShard = 2_500_012
    public static let pleRows = pleShards * pleRowsPerShard
    public static let geometry: [String: Int] = [
        "hidden_size": 2_560, "num_hidden_layers": layers, "vocab_size": 248_320,
        "num_experts": experts, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640, "num_attention_heads": 24,
        "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
        "indexer_compress_ratio": 4, "indexer_budget": 2_048,
        "hc_count": 4, "hc_lowrank": 320, "heads_per_ngram": 8,
    ]

    /// Checks the architecture and returns the quantization scheme. Throws only for a
    /// checkpoint these kernels cannot run correctly; the message says what differs.
    @discardableResult
    public static func validate(config: Data) throws -> QuantizationScheme {
        func incompatible(_ what: String) -> EngineError {
            EngineError("This checkpoint is not a supported Qwen3.8-Flash-Next build: \(what)", code: "model_incompatible")
        }
        guard let root = try? JSONSerialization.jsonObject(with: config) as? [String: Any] else {
            throw incompatible("config.json is not JSON")
        }
        guard root["model_type"] as? String == "qwen4_exp" else {
            throw incompatible("model_type is \(root["model_type"] ?? "missing"), expected qwen4_exp")
        }
        let text = root["text_config"] as? [String: Any] ?? root
        let quantization = root["quantization"] as? [String: Any] ?? root["quantization_config"] as? [String: Any]
        for (key, expected) in geometry {
            guard let value = text[key] as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else {
                throw incompatible("\(key) is \(text[key] ?? "missing"), expected \(expected)")
            }
            // A pruned GGUF keeps fewer routed experts: the engine keeps 512 per layer and never routes the others.
            if key == "num_experts", quantization?["expert_format"] as? String == "gguf", (10...experts).contains(value.intValue),
                value.doubleValue == Double(value.intValue)
            {
                continue
            }
            guard value.doubleValue == Double(expected) else { throw incompatible("\(key) is \(value), expected \(expected)") }
        }
        let expectedLayers = (0..<layers).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        guard (text["layer_types"] as? [String]).map({ $0 == expectedLayers }) ?? true,
            (text["output_gate_type"] as? String ?? "sigmoid") == "sigmoid",
            (text["partial_rotary_factor"] as? Double ?? 0.25) == 0.25,
            ((rope["rope_theta"] ?? text["rope_theta"]) as? NSNumber)?.doubleValue ?? 10_000_000 == 10_000_000
        else {
            throw incompatible("attention layout or rotary embedding differs")
        }
        // Config's PLE layer ID is one-based; the tensors live under zero-based layer 1.
        guard (text["ple_layer_ids"] as? [Int]).map({ $0 == [2] }) ?? true else { throw incompatible("PLE layer differs") }
        return try scheme(quantization)
    }

    /// Reads the MLX quantization block: 4-bit affine with group 64 or 32 and a 4-bit group-32 PLE
    /// table. Overrides on the matrices kept as floats (`floatModules`) are dequantized at load; any
    /// other override that differs from the global layout needs its own kernels.
    static func scheme(_ quantization: [String: Any]?) throws -> QuantizationScheme {
        guard let quantization else {
            throw EngineError(
                "This checkpoint is unquantized; `origami model add <folder>` converts it once to 4-bit", code: "model_incompatible")
        }
        if let kind = ["fp8": "FP8", "modelopt": "NVIDIA's NVFP4 (ModelOpt) release"][quantization["quant_method"] as? String ?? ""] {
            throw EngineError(
                "This checkpoint is \(kind); `origami model add <folder>` converts it once to 4-bit", code: "model_incompatible")
        }
        let bits = quantization["bits"] as? Int
        let group = quantization["group_size"] as? Int
        let mode = quantization["mode"] as? String ?? "affine"
        // Keyed by canonical module names (mlx-community's PLE shards `shards.<n>` read as `shard_<n>`).
        let overrides = Dictionary(quantization.compactMapValues { $0 as? [String: Any] }.map { (ModelSource.canonical($0.key), $0.value) })
        { a, _ in a }
        func format(_ entry: [String: Any]) -> (Int?, Int?, String) {
            (entry["bits"] as? Int, entry["group_size"] as? Int, entry["mode"] as? String ?? "affine")
        }
        let ple = overrides.filter { $0.key.contains("ple_embedding.ngram_embedding.shard_") }
        // The PLE kernels read 4-bit group 32: overridden on every shard, or the global layout itself.
        let pleIsG32 = ple.isEmpty ? group == 32 : ple.count == pleShards && ple.values.allSatisfy { format($0) == (4, 32, "affine") }
        // NVFP4 experts (converted from NVIDIA's ModelOpt release): every routed expert bank, nothing else.
        let fp4 = overrides.filter { format($0.value).2 == "nvfp4" }
        let fp4Experts =
            fp4.count == layers * 3
            && fp4.allSatisfy { name, entry in
                format(entry) == (4, 16, "nvfp4")
                    && ["gate_proj", "up_proj", "down_proj"].contains { name.hasSuffix(".mlp.switch_mlp." + $0) }
            }
        let unsupported = overrides.filter { name, entry in
            if ple[name] != nil || (fp4Experts && fp4[name] != nil) { return false }
            if floatModules.contains(where: { name.hasSuffix($0) }) { return !dequantizable(entry) }
            return format(entry) != (bits, group, mode)
        }
        // `expert_format: gguf`: the routed experts are in a GGUF (`GGUFExperts`), not among the checkpoint's tensors.
        let ggufExperts = quantization["expert_format"] as? String == "gguf"
        guard quantization["expert_format"] == nil || ggufExperts else {
            throw EngineError(
                "This checkpoint's expert_format \(quantization["expert_format"]!) is not one Origami reads", code: "model_incompatible")
        }
        if ggufExperts, bits == 4, mode == "affine", group == 64, pleIsG32, unsupported.isEmpty, !fp4Experts { return .ggufExperts }
        if !ggufExperts, bits == 4, mode == "affine", group == 64 || group == 32, pleIsG32, unsupported.isEmpty, !fp4Experts || group == 64
        {
            return fp4Experts ? .nvfp4Experts : group == 64 ? .affine4Group64 : .affine4Group32
        }
        let layout =
            "\(bits.map(String.init) ?? "?")-bit group \(group.map(String.init) ?? "?") (\(mode))"
            + (unsupported.isEmpty ? "" : " with \(unsupported.count) per-tensor overrides")
        // Widths the converter reads (`CheckpointConverter`'s MLX affine sources) convert once.
        if mode == "affine", let bits, [2, 4, 8].contains(bits), !ggufExperts, fp4.isEmpty {
            throw EngineError(
                "This checkpoint is MLX \(layout); `origami model add <folder>` converts it once to 4-bit group 64",
                code: "model_incompatible")
        }
        throw EngineError(
            "Quantization \(layout) has no Origami kernels yet. Supported: 4-bit group 64 or 32 with a 4-bit "
                + "group-32 PLE table, as in carloslfu/Qwen3.8-Flash-Next-MLX-4bit and mlx-community/Qwen3.8-Flash-Next-4bit.",
            code: "model_incompatible")
    }

    /// Modules the engine keeps as float matrices; a quantized copy is dequantized when it is loaded.
    static let floatModules = [
        ".mlp.gate", ".mlp.shared_expert_gate", ".block_inject_weight",
        ".linear_attn.in_proj_a", ".linear_attn.in_proj_b", ".indexer.index_qk_proj",
    ]
    static func dequantizable(_ entry: [String: Any]) -> Bool {
        [2, 4, 8].contains(entry["bits"] as? Int ?? 0) && (entry["group_size"] as? Int ?? 0) > 0
            && (entry["mode"] as? String ?? "affine") == "affine"
    }

    /// The global affine format and the per-module ones the config names (canonical module names).
    public static func formats(config: Data) -> (global: AffineFormat, modules: [String: AffineFormat]) {
        let root = (try? JSONSerialization.jsonObject(with: config)) as? [String: Any] ?? [:]
        let block = root["quantization"] as? [String: Any] ?? root["quantization_config"] as? [String: Any] ?? [:]
        let global = AffineFormat(bits: block["bits"] as? Int ?? 4, group: block["group_size"] as? Int ?? 64)
        var modules: [String: AffineFormat] = [:]
        for (name, value) in block {
            guard let entry = value as? [String: Any] else { continue }
            modules[ModelSource.canonical(name)] = AffineFormat(
                bits: entry["bits"] as? Int ?? global.bits,
                group: entry["group_size"] as? Int ?? global.group)
        }
        return (global, modules)
    }

    /// Feature probe instead of a chip-name check. Refuses only what the kernels cannot do:
    /// Apple-silicon unified memory, SIMD-group matrices (Apple7, M1 and later) and 32 KiB
    /// threadgroup memory. Returns notes for `origami status` and the startup log.
    @discardableResult
    public static func validate(device: MTLDevice) throws -> [String] {
        guard device.hasUnifiedMemory, device.supportsFamily(.apple7), device.maxThreadgroupMemoryLength >= 32_768 else {
            throw EngineError(
                "\(device.name) is not supported: Origami needs an Apple-silicon GPU (M1 or later)", code: "device_unsupported")
        }
        var notes: [String] = []
        if !device.supportsFamily(.apple8) {
            notes.append("\(device.name): bfloat16 matrix kernels disabled (Apple8 or later needed); prefill is slower")
        }
        return notes
    }

    /// True when the bfloat16 SIMD-group matrix kernels are known to compile and run (M2 and later).
    public static func supportsBFloatMatrices(_ device: MTLDevice) -> Bool { device.supportsFamily(.apple8) }
}
