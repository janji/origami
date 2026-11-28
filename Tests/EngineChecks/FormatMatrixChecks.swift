import Foundation
import OrigamiEngine

/// docs/models.md's format matrix, one safetensors class per row, from headers alone: what `model add` does with it
/// (runs as stored, converts, refused) and whether the conversion quantizes weights a second time (`lossy`, which
/// `model add` names and needs `--lossy` for). The GGUF rows are in `converter_gguf` (Qwen) and
/// `model_family_geometry` (GLM, DeepSeek). Also: re-quantizing MLX 4-bit group-64 codes onto the same grid
/// reproduces them, which is why such tensors do not count as lossy.
func modelFormatMatrix() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Format matrix: " + what) } }
    func config(_ name: String) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/\(name)/config.json"))) as! [String: Any]
    }
    func spans(_ tensors: [String: String]) throws -> [String: TensorSpan] {
        let json = tensors.mapValues { ["file": "a.safetensors", "dtype": $0, "offset": 0, "length": 2, "shape": [1]] as [String: Any] }
        return try JSONDecoder().decode([String: TensorSpan].self, from: JSONSerialization.data(withJSONObject: json))
    }
    let l = "language_model.model.layers.0."
    /// MLX codes of three matrices, the experts and the router (kept in floats).
    let mlx = [
        l + "self_attn.q_proj.weight": "U32", l + "self_attn.q_proj.scales": "BF16", l + "self_attn.q_proj.biases": "BF16",
        l + "mlp.switch_mlp.up_proj.weight": "U32", l + "mlp.switch_mlp.up_proj.scales": "BF16",
        l + "mlp.switch_mlp.up_proj.biases": "BF16", l + "mlp.gate.weight": "U32", l + "mlp.gate.scales": "BF16",
        l + "mlp.gate.biases": "BF16", l + "input_layernorm.weight": "BF16",
    ]
    let native = try config("mlx-community-4bit")
    func affine(_ bits: Int, _ group: Int, overrides: [String: Any] = [:]) -> [String: Any] {
        var c = native
        c["quantization"] = ["bits": bits, "group_size": group, "mode": "affine"].merging(overrides) { $1 }
        c["quantization_config"] = c["quantization"]
        return c
    }
    var autoRound = native
    autoRound["quantization"] = nil
    autoRound["quantization_config"] = ["quant_method": "auto-round", "bits": 4, "group_size": 128]
    var bf16 = try config("official-bf16")
    bf16["quantization_config"] = nil
    var fp8 = bf16
    fp8["quantization_config"] = ["quant_method": "fp8", "weight_block_size": [128, 128]]
    var nvfp4 = bf16
    nvfp4["quantization_config"] = ["quant_method": "modelopt", "quant_algo": "NVFP4"]
    var glmBF16 = try config("glm-5.3-flash")
    glmBF16["quantization_config"] = nil
    var glmMLX = glmBF16
    glmMLX["quantization_config"] = ["bits": 4, "group_size": 64]
    var exl3 = try config("deepseek-v4.1-flash")
    exl3["quantization_config"] = ["quant_method": "exl3"]
    let floats = ["model.language_model.layers.0.self_attn.q_proj.weight": "BF16", "lm_head.weight": "BF16"]
    // (class, config, tensors, verdict, words the lossy line has; nil: not lossy)
    let rows: [(String, [String: Any], [String: String], CompatibilityReport.Verdict, String?)] = [
        ("Qwen MLX 4-bit group 32 (mlx-community)", native, mlx, .runs, nil),
        ("Qwen BF16", bf16, floats, .converts, nil),
        ("Qwen FP8", fp8, ["model.language_model.layers.0.self_attn.q_proj.weight": "F8_E4M3"], .converts, "FP8 E4M3 ×1"),
        (
            "Qwen NVFP4 (ModelOpt)", nvfp4, ["model.language_model.layers.1.ple.ple_embedding.shard_0": "F8_E4M3"], .converts,
            "NVFP4 routed experts stay"
        ),
        ("Qwen MLX 8-bit", affine(8, 64), mlx, .converts, "re-quantizes 2 tensors stored as MLX 8-bit group 64 ×2"),
        ("Qwen MLX 2-bit group 32", affine(2, 32), mlx, .converts, "MLX 2-bit group 32 ×2"),
        ("Qwen MLX 4-bit group 128", affine(4, 128), mlx, .converts, "MLX 4-bit group 128 ×2"),
        (
            "Qwen MLX 4-bit group 64, 8-bit attention", affine(4, 64, overrides: [l + "self_attn.q_proj": ["bits": 8, "group_size": 64]]),
            mlx, .converts, "re-quantizes 1 tensors stored as MLX 8-bit group 64 ×1"
        ),
        ("Qwen MLX 3-bit", affine(3, 64), mlx, .refused, nil),
        (
            "Qwen MLX 4-bit group 64, 6-bit attention", affine(4, 64, overrides: [l + "self_attn.q_proj": ["bits": 6, "group_size": 64]]),
            mlx, .refused, nil
        ),
        ("Qwen AutoRound (GPTQ packing)", autoRound, ["x.qweight": "I32", "x.scales": "F16"], .refused, nil),
        ("GLM BF16", glmBF16, ["model.layers.0.mlp.experts.0.up_proj.weight": "BF16"], .converts, nil),
        (
            "GLM FP8", try config("glm-5.3-flash"), ["model.layers.0.mlp.experts.0.up_proj.weight": "F8_E4M3", "x": "F32"], .converts,
            "re-quantizes 1 FP8 E4M3 tensors"
        ),
        ("GLM MLX 4-bit", glmMLX, ["model.layers.0.mlp.experts.0.up_proj.weight": "U32"], .refused, nil),
        ("DeepSeek FP8", try config("deepseek-v4.1-flash"), ["x": "F8_E4M3"], .refused, nil),
        ("DeepSeek EXL3", exl3, ["x": "I16"], .refused, nil),
    ]
    for (what, checkpoint, tensors, verdict, lossy) in rows {
        let report = ModelCompatibility.check(config: checkpoint, headers: try spans(tensors), source: what)
        try require(report.verdict == verdict, "\(what) is \(report.verdict): \(report.reason)")
        try require(
            lossy.map { report.lossy?.contains($0) ?? false } ?? (report.lossy == nil), "\(what) lossy: \(report.lossy ?? "none")")
        try require(
            (ModelCompatibility.lossLine(report) != nil) == (lossy != nil)
                && (ModelCompatibility.lossLine(report)?.hasSuffix("Measured cost: not measured.") ?? true),
            "\(what) line: \(ModelCompatibility.lossLine(report) ?? "none")")
    }

    // MLX 4-bit group 64 codes decoded and quantized again with MLX's rule come back bit for bit.
    var seed: UInt64 = 0x5EED
    let rows64 = 8
    let columns = 256
    let values = (0..<(rows64 * columns)).map { _ -> Float in
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(Int64(bitPattern: seed >> 11) % 2_000_001) / 1_000_000 - 1
    }
    func quantize(_ x: [Float]) -> ([UInt32], [UInt16], [UInt16]) {
        var codes = [UInt32](repeating: 0, count: x.count / 8)
        var scales = [UInt16](repeating: 0, count: x.count / 64)
        var biases = scales
        x.withUnsafeBytes { v in
            codes.withUnsafeMutableBufferPointer { c in
                scales.withUnsafeMutableBufferPointer { s in
                    biases.withUnsafeMutableBufferPointer { b in
                        CheckpointConverter.quantize(
                            v.baseAddress!, dtype: "F32", rows: rows64, columns: columns, group: 64, codes: c.baseAddress!,
                            scales: s.baseAddress!, biases: b.baseAddress!)
                    }
                }
            }
        }
        return (codes, scales, biases)
    }
    let first = quantize(values)
    var decoded = [Float](repeating: 0, count: values.count)
    first.0.withUnsafeBytes { raw in
        CheckpointConverter.decodeAffine(
            raw.baseAddress!, rows: 0..<rows64, columns: columns, bits: 4, group: 64, scales: first.1.map(BF16.decode),
            biases: first.2.map(BF16.decode), into: &decoded)
    }
    let again = quantize(decoded)
    try require(again.0 == first.0 && again.1 == first.1 && again.2 == first.2, "4-bit group 64 codes changed when quantized again")
}
