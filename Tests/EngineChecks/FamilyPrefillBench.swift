import Foundation
import Metal
import OrigamiEngine

/// `engine-checks --bench-family-prefill <glm|deepseek> [tokens] [runs] [profile] [pairs]`: a prompt of `tokens` (default
/// 2,500) read through the family engine's own prompt path (`forward` in calls of `promptRows`, as serving runs a prompt
/// that is not fed) on a few layers at the real model's widths, from `Tests/Fixtures/models/<model>/config.json`, with the
/// real models' expert types (GLM Q4_K gate and up, Q5_K down; DeepSeek MXFP4). Dense weights are Q8_0 and random; routed
/// experts are zeros (a hole in the file: no disk space, and a kernel's time does not depend on its values), read into a
/// pool of 35% of the Mac's memory. GLM: a dense KDA layer, a KDA and a DSA MoE layer, and a last KDA layer (a prompt runs
/// only its last row's experts). DeepSeek: window-only with Engram, a ratio-2 key/value and index source, a ratio-2 reader,
/// a ratio-1 source with the candidate blocks, a ratio-1 index source, a ratio-1 reader, and a last layer.
/// Prints, per run after a warm-up, wall time and the engine's counters (GPU busy, GPU and read waits, command buffers);
/// with `profile` ("fine", or "layer" for a class per layer and step) one more run with `FamilyGPUProfile`: GPU seconds,
/// dispatches, host encode seconds, weight bytes and rows per dispatch by kernel class. With `decode` (or `decode-layer`)
/// instead, decode tokens after the prompt (`familyDecodeBench`). `pairs` (GLM, default 1): KDA and DSA MoE layer pairs
/// (more layers outgrow the pool, so decode then reads). The model is deleted at the end.
func familyPrefillBench(_ arguments: [String]) throws {
    guard let name = arguments.first, ["glm", "deepseek"].contains(name) else {
        throw EngineError(
            "Usage: engine-checks --bench-family-prefill <glm|deepseek> [tokens] [runs] [fine|layer|decode|decode-layer][,gap=ms][,spin][,keepalive|keepalive-short] [GLM KDA+DSA pairs]"
        )
    }
    let tokens = arguments.count > 1 ? Int(arguments[1]) ?? 2_500 : 2_500
    let runs = arguments.count > 2 ? Int(arguments[2]) ?? 3 : 3
    let profiling = arguments.count > 3 ? arguments[3] : nil
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prefill-bench-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let family: ModelFamily = name == "glm" ? .glm : .deepseek
    let path = try realWidthFamilyGGUF(family, in: folder, pairs: arguments.count > 4 ? max(1, Int(arguments[4]) ?? 1) : 1)
    try autoreleasepool {
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let budget = min(UInt64(routed), ProcessInfo.processInfo.physicalMemory * 35 / 100)
        let model = try makeFamilyModel(weights, expertBudget: budget, context: (tokens + 2_047) / 2_048 * 2_048 + 2_048) { _ in }
        let runner = model as! FamilyRunner
        var rng = SyntheticBenchRNG(state: 0xBE7C_0001)
        let prompt = (0..<tokens).map { _ in Int(rng.next() % UInt64(weights.geometry.vocab)) }
        func read() throws -> (wall: Double, timing: FamilyRunner.Timing) {
            try model.reset()
            try runner.beginPrefill()
            let before = runner.timing
            let started = Date()
            var first = 0
            while first < prompt.count {
                let last = min(prompt.count, first + runner.promptRows)
                _ = try model.forward(Array(prompt[first..<last]))
                first = last
            }
            let wall = Date().timeIntervalSince(started)
            runner.beginDecode()
            var t = runner.timing
            t.gpu -= before.gpu
            t.gpuWait -= before.gpuWait
            t.reads -= before.reads
            t.commits -= before.commits
            t.overBound -= before.overBound
            return (wall, t)
        }
        print(
            "\(name): \(weights.geometry.layers) layers, \(tokens) tokens in calls of \(runner.promptRows), "
                + "pool \(runner.experts.slotCount) slots (\(runner.experts.poolBytes >> 20) MB) for \(routed >> 20) MB of experts")
        _ = try read()
        for run in 0..<runs {
            runner.resetLongestBuffer()
            let (wall, t) = try read()
            print(
                String(
                    format: "run %d: wall %.3f s, gpu busy %.3f s, gpu wait %.3f s, read wait %.3f s, buffers %d, longest %.1f ms",
                    run, wall, t.gpu, t.gpuWait, t.reads, t.commits, t.longest * 1000)
                    + " (\(t.longestStep)), \(t.overBound) over \(Int(CommandBudget.seconds * 1000)) ms")
        }
        guard let profiling else { return }
        if profiling.hasPrefix("decode") {
            // `decode,gap=17,keepalive`: a 17 ms pause before each layer's experts (as a read leaves; `spin`: spun through), under
            // the request keep-alive (`keepalive-short` its short poll).
            let options = profiling.split(separator: ",").map(String.init)
            runner.benchReadGap = (options.first { $0.hasPrefix("gap=") }.flatMap { Double($0.dropFirst(4)) } ?? 0) / 1000
            runner.benchGapSpins = options.contains("spin")
            let policy = options.compactMap {
                ["keepalive": RequestGPUKeepAlive.Policy.original, "keepalive-short": .shortRegisterAttention][$0]
            }.first
            // The owner must outlive the session: dropping it closes the session (its deinit), as it did here before.
            let owner = try policy.map { try RequestGPUKeepAlive(ledger: MemoryLedger(ceiling: 1 << 30, transientMargin: 1), policy: $0) }
            let keepAlive = try owner?.begin()
            defer {
                withExtendedLifetime(owner) {
                    if let snapshot = keepAlive?.close() {
                        print("keep-alive: \(snapshot.submittedCommands) commands in \(snapshot.seconds) s")
                    }
                }
            }
            print(
                "decode: gap \(runner.benchReadGap * 1000) ms a layer\(runner.benchGapSpins ? " (spun)" : ""), keep-alive \(policy.map(\.rawValue) ?? "off")"
            )
            try familyDecodeBench(model: model, runner: runner, device: device, tokens: 32, byLayer: options[0] == "decode-layer")
            return
        }
        let profile = try FamilyGPUProfile(device: device) { step, kernel, matrix in
            let words = step.split(separator: " ")
            let layer = words.count > 1 && words[0] == "layer" ? "L\(words[1]) " : ""
            let part = words.filter { Int($0) == nil && !$0.contains("+") && $0 != "rows" && $0 != "layer" }.joined(separator: " ")
            let role =
                matrix.map { m in
                    m.split(separator: ".").filter { Int($0) == nil && $0 != "blk" && $0 != "weight" }.joined(separator: ".")
                } ?? ""
            let what = role.isEmpty ? kernel : "\(kernel) \(role)"
            return profiling == "layer" ? "\(layer)\(part)" : "\(layer)\(part) | \(what)"
        }
        let (cpu0, gpu0) = device.sampleTimestamps()
        runner.profile = profile
        let (wall, t) = try read()
        runner.profile = nil
        let (cpu1, gpu1) = device.sampleTimestamps()
        // Both timestamps are in nanoseconds on Apple GPUs; the ratio corrects a GPU clock that is not.
        try profile.collect(nanosecondsPerTick: Double(cpu1 - cpu0) / Double(gpu1 - gpu0))
        let total = profile.gpu.values.reduce(0, +)
        print(
            String(
                format:
                    "profiled run: wall %.3f s, gpu busy %.3f s (in classes %.3f s), gpu wait %.3f s, read wait %.3f s, buffers %d, host encode %.3f s",
                wall, t.gpu, total, t.gpuWait, t.reads, t.commits, profile.host.values.reduce(0, +)))
        print("class\tgpu s\tshare\tdispatches\thost s\tweight MB\trows (dispatches)")
        for (key, seconds) in profile.gpu.sorted(by: { $0.value > $1.value }) {
            let rows = (profile.rows[key] ?? [:]).sorted { $0.key < $1.key }
            let shown =
                rows.count <= 6
                ? rows.map { "\($0.key)x\($0.value)" }.joined(separator: " ")
                : "\(rows.first!.key)-\(rows.last!.key), median \(median(rows))"
            print(
                String(
                    format: "%@\t%.4f\t%.1f%%\t%d\t%.4f\t%.0f\t%@", key, seconds, 100 * seconds / max(total, 1e-9),
                    profile.dispatches[key] ?? 0, profile.host[key] ?? 0, Double(profile.weightBytes[key] ?? 0) / 1e6, shown))
        }
    }
}

/// `decode` / `decode-layer` in place of a profile: after the prompt, `tokens` decode tokens (one row each, every
/// expert resident in the bench's pool) with the engine's counters per token, then one profiled pass by class (by layer
/// with `decode-layer`). Wall less GPU busy is the GPU's idle time a token.
private func familyDecodeBench(model: FamilyModel, runner: FamilyRunner, device: MTLDevice, tokens: Int, byLayer: Bool) throws {
    var rng = SyntheticBenchRNG(state: 0xBE7C_0003)
    // A few tokens only, so their experts stay resident after the warm-up (a fully resident decode).
    func token() -> Int { Int(rng.next() % 3) }
    for _ in 0..<12 { _ = try model.forward([token()]) }
    var t0 = runner.timing
    var walls: [Double] = []
    var gpu: [Double] = []
    var wait: [Double] = []
    var reads: [Double] = []
    var commits: [Int] = []
    for _ in 0..<tokens {
        let started = Date()
        _ = try model.forward([token()])
        walls.append(Date().timeIntervalSince(started))
        let t = runner.timing
        gpu.append(t.gpu - t0.gpu)
        wait.append(t.gpuWait - t0.gpuWait)
        reads.append(t.reads - t0.reads)
        commits.append(t.commits - t0.commits)
        t0 = t
    }
    func med(_ x: [Double]) -> Double { x.sorted()[x.count / 2] * 1000 }
    print(
        String(
            format:
                "decode (median of %d tokens): wall %.2f ms, gpu busy %.2f ms, idle %.2f ms, gpu wait %.2f ms, read wait %.2f ms, buffers %d",
            tokens, med(walls), med(gpu), med(zip(walls, gpu).map { $0 - $1 }), med(wait), med(reads), commits.sorted()[tokens / 2]))
    let profile = try FamilyGPUProfile(device: device) { step, kernel, matrix in
        let words = step.split(separator: " ")
        let layer = byLayer && words.count > 1 && words[0] == "layer" ? "L\(words[1]) " : ""
        let part = words.filter { Int($0) == nil && !$0.contains("+") && $0 != "rows" && $0 != "layer" }.joined(separator: " ")
        let role = matrix.map { m in
            m.split(separator: ".").filter { Int($0) == nil && $0 != "blk" && $0 != "weight" }.joined(separator: ".")
        }
        return "\(layer)\(part) | \(role.map { "\(kernel) \($0)" } ?? kernel)"
    }
    let (cpu0, gpu0) = device.sampleTimestamps()
    runner.profile = profile
    let profiled = 8
    let started = Date()
    for _ in 0..<profiled { _ = try model.forward([token()]) }
    let wall = Date().timeIntervalSince(started)
    runner.profile = nil
    let (cpu1, gpu1) = device.sampleTimestamps()
    try profile.collect(nanosecondsPerTick: Double(cpu1 - cpu0) / Double(gpu1 - gpu0))
    let total = profile.gpu.values.reduce(0, +)
    let per = 1000 / Double(profiled)
    print(
        String(
            format: "profiled decode: %.2f ms a token wall, %.2f ms in classes, host encode %.2f ms",
            wall * per, total * per, profile.host.values.reduce(0, +) * per))
    print("class\tgpu ms/token\tshare\tdispatches/token\thost ms/token")
    for (key, seconds) in profile.gpu.sorted(by: { $0.value > $1.value }).prefix(24) {
        print(
            String(
                format: "%@\t%.3f\t%.1f%%\t%d\t%.3f", key, seconds * per, 100 * seconds / max(total, 1e-9),
                (profile.dispatches[key] ?? 0) / profiled, (profile.host[key] ?? 0) * per))
    }
    let idle = profile.idleBefore.values.reduce(0, +)
    print(String(format: "GPU idle between timed spans: %.3f ms a token, before:", idle * per))
    for (key, seconds) in profile.idleBefore.sorted(by: { $0.value > $1.value }).prefix(12) {
        print(String(format: "  %.3f ms  %@", seconds * per, key))
    }
}

private func median(_ histogram: [(key: Int, value: Int)]) -> Int {
    let count = histogram.reduce(0) { $0 + $1.value }
    var seen = 0
    for (rows, n) in histogram {
        seen += n
        if 2 * seen >= count { return rows }
    }
    return 0
}

private struct SyntheticBenchRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// The benchmark's model: the fixture config's widths on a few layers (see `familyPrefillBench`), a 4,096-token vocabulary.
private func realWidthFamilyGGUF(_ family: ModelFamily, in folder: URL, pairs: Int = 1) throws -> URL {
    let fixture = family == .glm ? "glm-5.3-flash" : "deepseek-v4.1-flash"
    let json = try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/\(fixture)/config.json")))
    guard var config = (json as? [String: Any])?["text_config"] as? [String: Any] else { throw EngineError("No text_config in \(fixture)") }
    func int(_ key: String) -> Int { (config[key] as? NSNumber)?.intValue ?? 0 }
    let hidden = int("hidden_size")
    let experts = int("n_routed_experts")
    let expertWidth = int("moe_intermediate_size")
    let vocab = 4_096
    var tensors: [(String, [Int], Int)] = []
    // Random values from a pool, quantized as the synthetic models' are (uniform within ±sqrt(3 / width) around `center`).
    var rng = SyntheticBenchRNG(state: 0xBE7C_0002)
    let pool = (0..<(1 << 20) + 7).map { _ in Float(rng.next() >> 40) / Float(1 << 23) - 1 }
    var fills: [String: (scale: Float?, center: Float)] = [:]
    func add(_ name: String, _ shape: [Int], type requested: Int? = nil, scale: Float? = nil, center: Float = 0, zeros: Bool = false) {
        let width = shape.last!
        tensors.append((name, shape, requested ?? (shape.count == 2 && shape[0] >= 32 && width % 32 == 0 ? 8 : 0)))
        if !zeros { fills[name] = (scale, center) }
    }
    add("token_embd.weight", [vocab, hidden], type: 8, scale: 1)
    add("output.weight", [vocab, hidden], type: 8)
    add("output_norm.weight", [hidden], scale: 0.1, center: 1)
    let layers: Int
    let moeLayers: [Int]
    if family == .glm {
        // A dense KDA layer, `pairs` KDA and DSA MoE layers, then a last KDA layer.
        layers = 2 + 2 * pairs
        let types =
            ["linear_attention"] + Array(repeating: ["linear_attention", "deepseek_sparse_attention"], count: pairs).flatMap { $0 }
            + ["linear_attention"]
        moeLayers = Array(1..<layers)
        var linear = config["linear_attn_config"] as? [String: Any] ?? [:]
        linear["kda_layers"] = types.indices.filter { types[$0] == "linear_attention" }
        linear["full_attn_layers"] = types.indices.filter { types[$0] != "linear_attention" }
        config.merge(
            [
                "num_hidden_layers": layers, "layer_types": types,
                "mlp_layer_types": ["dense"] + Array(repeating: "sparse", count: layers - 1),
                "first_k_dense_replace": 1, "indexer_types": Array(repeating: "full", count: layers), "linear_attn_config": linear,
                "num_nextn_predict_layers": 0, "vocab_size": vocab,
            ]
        ) { $1 }
        let (heads, qk, v, qRank, kvRank) = (
            int("num_attention_heads"), int("qk_head_dim"), int("v_head_dim"), int("q_lora_rank"), int("kv_lora_rank")
        )
        let kdaHeads = (linear["num_heads"] as? NSNumber)?.intValue ?? 0
        let kdaDim = (linear["head_dim"] as? NSNumber)?.intValue ?? 0
        let (indexHeads, indexDim, pool) = (int("index_n_heads"), int("index_head_dim"), int("index_kpool"))
        let c = kdaHeads * kdaDim
        for (l, kind) in types.enumerated() {
            if kind == "linear_attention" {
                for part in ["q", "k", "v"] {
                    add("blk.\(l).kda_\(part).weight", [c, hidden])
                    add("blk.\(l).kda_\(part)_conv.weight", [4, c], scale: 0.4)
                }
                for part in ["f", "g"] {
                    add("blk.\(l).kda_\(part)_a.weight", [kdaDim, hidden])
                    add("blk.\(l).kda_\(part)_b.weight", [c, kdaDim])
                }
                add("blk.\(l).kda_beta.weight", [kdaHeads, hidden])
                add("blk.\(l).kda_output.weight", [hidden, c])
                add("blk.\(l).kda_dt_bias.weight", [c], scale: 0.5)
                add("blk.\(l).kda_a_log.weight", [kdaHeads], scale: 0.5, center: -0.5)
                add("blk.\(l).kda_o_norm.weight", [kdaDim], scale: 0.1, center: 1)
            } else {
                add("blk.\(l).attn_q_a.weight", [qRank, hidden])
                add("blk.\(l).attn_q_a_norm.weight", [qRank], scale: 0.1, center: 1)
                add("blk.\(l).attn_q_b.weight", [heads * qk, qRank])
                add("blk.\(l).attn_kv_a_mqa.weight", [kvRank, hidden])
                add("blk.\(l).attn_kv_a_norm.weight", [kvRank], scale: 0.1, center: 1)
                add("blk.\(l).attn_k_b.weight", [heads * kvRank, qk])
                add("blk.\(l).attn_v_b.weight", [heads * v, kvRank])
                add("blk.\(l).attn_output.weight", [hidden, heads * v])
                add("blk.\(l).indexer.attn_q_b.weight", [indexHeads * indexDim, qRank])
                add("blk.\(l).indexer.attn_k.weight", [indexDim, hidden])
                add("blk.\(l).indexer.proj.weight", [indexHeads, hidden])
                add("blk.\(l).indexer.pool_gate.weight", [indexDim, hidden])
                add("blk.\(l).indexer.k_norm.weight", [indexDim], scale: 0.1, center: 1)
                add("blk.\(l).indexer.k_norm.bias", [indexDim], scale: 0.1)
                add("blk.\(l).indexer.pool_ape.weight", [pool, indexDim], scale: 0.1)
            }
        }
    } else {
        // Window only with Engram; ratio 2: a key/value and index source, a reader; ratio 1: a source with the candidate
        // blocks, an index source, a reader; a last layer.
        layers = 7
        moeLayers = Array(0..<layers)
        let ratios = [0, 2, 2, 1, 1, 1, 1]
        let kvSources = [1, 3]
        let indexSources = [1, 3, 4]
        let engramLayers = [0]
        config = config.filter { !$0.key.hasPrefix("dspark") && !$0.key.hasPrefix("engram_num") && !$0.key.contains("vocab_size") }
        config.merge(
            [
                "num_hidden_layers": layers, "compress_ratios": ratios, "kv_source_layer_ids": kvSources,
                "index_source_layer_ids": indexSources, "candidate_source_layer_id": 3, "engram_layer_ids": engramLayers,
                "num_nextn_predict_layers": 0, "vocab_size": vocab,
            ]
        ) { $1 }
        let (heads, headDim, qRank, groups, oRank) = (
            int("num_attention_heads"), int("head_dim"), int("q_lora_rank"), int("o_groups"), int("o_lora_rank")
        )
        let (indexHeads, indexDim) = (int("index_n_heads"), int("index_head_dim"))
        let (ngram, engramHeads, engramDim) = (int("engram_max_ngram_size"), int("engram_n_heads"), int("engram_head_dim"))
        let cols = (ngram - 1) * engramHeads
        let primes = (0..<cols).map { [1_009, 1_013, 1_019, 1_021, 1_031, 1_033, 1_039, 1_049][$0 % 8] + 100 * ($0 / 8) }
        for l in 0..<layers {
            add("blk.\(l).attn_q_a.weight", [qRank, hidden])
            add("blk.\(l).attn_q_a_norm.weight", [qRank], scale: 0.1, center: 1)
            add("blk.\(l).attn_q_b.weight", [heads * headDim, qRank])
            add("blk.\(l).attn_kv.weight", [headDim, hidden])
            add("blk.\(l).attn_kv_norm.weight", [headDim], scale: 0.1, center: 1)
            add("blk.\(l).attn_sinks.weight", [heads], scale: 0.5)
            add("blk.\(l).attn_output_a.weight", [groups * oRank, heads / groups * headDim])
            add("blk.\(l).attn_output_b.weight", [hidden, groups * oRank])
            if kvSources.contains(l) {
                add("blk.\(l).attn_compressor_kv.weight", [headDim, hidden])
                if ratios[l] > 1 { add("blk.\(l).attn_compressor_gate.weight", [headDim, hidden]) }
                add("blk.\(l).attn_compressor_norm.weight", [headDim], scale: 0.1, center: 1)
                add("blk.\(l).indexer.attn_k.weight", [indexDim, headDim])
                add("blk.\(l).indexer.k_norm.weight", [indexDim], scale: 0.1, center: 1)
            }
            if indexSources.contains(l) {
                add("blk.\(l).indexer.attn_q_b.weight", [indexHeads * indexDim, qRank])
                add("blk.\(l).indexer.proj.weight", [indexHeads, hidden])
            }
            if engramLayers.contains(l) {
                add("blk.\(l).engram_embd.weight", [primes.reduce(0, +), engramDim], type: 8, scale: 1)
                add("blk.\(l).engram_kv.weight", [5 * hidden, cols * engramDim])
                add("blk.\(l).engram_gate.weight", [4 * hidden], scale: 0.1, center: 1)
            }
        }
        config["_engram"] = ["primes": primes, "ngram": ngram]
    }
    // Layers after the attention: norms, mixes, router, experts (holes), shared expert or dense FFN.
    let expertTypes = family == .glm ? [12, 12, 13] : [39, 39, 39]
    for l in 0..<layers {
        add("blk.\(l).attn_norm.weight", [hidden], scale: 0.1, center: 1)
        add("blk.\(l).ffn_norm.weight", [hidden], scale: 0.1, center: 1)
        for part in ["attn", "ffn"] {
            add("blk.\(l).hc_\(part)_fn.weight", [24, 4 * hidden], type: 0, scale: 0.05)
            add("blk.\(l).hc_\(part)_scale.weight", [3], scale: 0.2, center: 0.8)
            add("blk.\(l).hc_\(part)_base.weight", [24], scale: 0.2)
        }
        if moeLayers.contains(l) {
            // Unit-variance logits, so rows spread over the experts as a real router's do.
            add("blk.\(l).ffn_gate_inp.weight", [experts, hidden], type: 0)
            add("blk.\(l).exp_probs_b.bias", [experts], scale: 0.05)
            add("blk.\(l).ffn_gate_exps.weight", [experts, expertWidth, hidden], type: expertTypes[0], zeros: true)
            add("blk.\(l).ffn_up_exps.weight", [experts, expertWidth, hidden], type: expertTypes[1], zeros: true)
            add("blk.\(l).ffn_down_exps.weight", [experts, hidden, expertWidth], type: expertTypes[2], zeros: true)
            add("blk.\(l).ffn_gate_shexp.weight", [expertWidth, hidden])
            add("blk.\(l).ffn_up_shexp.weight", [expertWidth, hidden])
            add("blk.\(l).ffn_down_shexp.weight", [hidden, expertWidth])
        } else {
            let width = int("intermediate_size")
            add("blk.\(l).ffn_gate.weight", [width, hidden])
            add("blk.\(l).ffn_up.weight", [width, hidden])
            add("blk.\(l).ffn_down.weight", [hidden, width])
        }
    }
    let architecture = family == .deepseek ? "deepseek41" : "glm5-next"
    var metadata: [(String, Any)] = [("general.architecture", architecture)]
    if let engram = config.removeValue(forKey: "_engram") as? [String: Any], let primes = engram["primes"] as? [Int],
        let ngram = engram["ngram"] as? Int
    {
        metadata += [
            ("deepseek41.engram.token_map", (0..<vocab).map { ($0 * 7 + 3) % 1_000 }),
            ("deepseek41.engram.multipliers", (0..<ngram).map { 2 * $0 * 104_729 + 1 }),
            ("deepseek41.engram.primes", primes),
        ]
    }
    let text = String(decoding: try JSONSerialization.data(withJSONObject: config, options: .sortedKeys), as: UTF8.self)
    let path = folder.appendingPathComponent("bench-\(architecture).gguf")
    let shapes = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, ($0.1, $0.2)) })
    var offset = 0
    try familyGGUF(
        metadata + [(architecture + ".config", text)], tensors: tensors,
        fill: { name, _ in
            guard let (shape, type) = shapes[name], let (scale, center) = fills[name] else { return nil }
            let count = shape.reduce(1, *)
            let amplitude = scale ?? (3 / Float(shape.last!)).squareRoot()
            var values = [Float](repeating: 0, count: count)
            for i in 0..<count { values[i] = center + amplitude * pool[(i + offset) & ((1 << 20) - 1)] }
            offset += 7
            var bytes = Data(count: GGUFQuantize.bytes(type, count: count))
            values.withUnsafeBufferPointer { source in
                bytes.withUnsafeMutableBytes { GGUFQuantize.encode(source.baseAddress!, count: count, type: type, into: $0.baseAddress!) }
            }
            return bytes
        }, to: path)
    return path
}
