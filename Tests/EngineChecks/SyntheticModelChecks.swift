import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// The model-free bitwise referee (ROADMAP S3): a tiny GLM-5.3-Flash- and DeepSeek-V4.1-Flash-shaped GGUF, generated
/// here from a fixed seed, runs a 2,100-token prompt and a greedy reply through the family engine's own entry points
/// (`makeFamilyModel`, `feed`/`forward` in the serving backend's prompt slices, `FamilyStateFile` for the saved state),
/// with an expert budget two fifths of the routed experts, with and without a hot copy of half of them, and with decode prefetch switched. Every token,
/// the logits' bits and the saved state's bytes must match `Tests/Fixtures/synthetic/<family>.json`.
/// Regenerate the references (after a change that is meant to alter output): run the checks with
/// ORIGAMI_SYNTHETIC_WRITE_REFERENCE=1; they then write the files instead of comparing.
/// Qwen3.8-Flash-Next has no synthetic model: its kernels and `ModelContract.validate` fix every dimension at the real
/// model's (48 layers, 512 experts, hidden 2,560, 248,320 vocabulary, a 320M-row PLE table), so the smallest model it
/// accepts is the real one.
func syntheticModelChecks() -> [(String, () throws -> Void)] {
    [
        ("synthetic_glm_end_to_end", { try syntheticEndToEnd(.glm) }),
        ("synthetic_deepseek_end_to_end", { try syntheticEndToEnd(.deepseek) }),
        ("synthetic_family_canary", { for family in [ModelFamily.glm, .deepseek] { try syntheticCanary(family) } }),
        ("synthetic_glm_prompt_buffers", { try syntheticPromptBuffers(.glm) }),
        ("synthetic_deepseek_prompt_buffers", { try syntheticPromptBuffers(.deepseek) }),
        ("synthetic_prompt_cancel_resumes", { for family in [ModelFamily.glm, .deepseek] { try syntheticPromptCancelResumes(family) } }),
        ("synthetic_restore_cancel_keeps", { for family in [ModelFamily.glm, .deepseek] { try syntheticRestoreCancelKeeps(family) } }),
        ("synthetic_prompt_loan", { for family in [ModelFamily.glm, .deepseek] { try syntheticPromptLoan(family) } }),
        ("synthetic_prompt_loan_returned", { for family in [ModelFamily.glm, .deepseek] { try syntheticPromptLoanReturned(family) } }),
        (
            "synthetic_read_ahead_under_pressure",
            { for family in [ModelFamily.glm, .deepseek] { try syntheticReadAheadUnderPressure(family) } }
        ),
    ]
}

private struct SyntheticRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 23) - 1 }
}

private let syntheticPromptTokens = 2_100
private let syntheticDecodeTokens = 24
private let syntheticContinueTokens = 8

/// A GGUF in the family's layout with every tensor role its engine reads, at a small width: Q8_0 dense matrices, F32
/// norms, routers and hyper-connection mixes, routed experts cycling Q4_K, Q2_K and Q8_0 by layer (several record sizes).
/// Without `weights` the tensors are zero bytes (the header is the same; the GGUF fuzzer's seed). `expertTypes` gives
/// some layers' banks other types (gate, up, down; the same values encoded, or for a type without an encoder here, Q3_K
/// and IQ2_XXS, bytes from the tensor's name with small block scales), every other tensor unchanged; `tokens` a tokenizer.
func syntheticFamilyGGUF(
    _ family: ModelFamily, in folder: URL, weights: Bool = true, expertTypes: [Int: [Int]] = [:], tokens: [String] = []
) throws -> URL {
    var rng = SyntheticRNG(state: family == .glm ? 0x5EED_0001 : 0x5EED_0002)
    let hidden = 256
    let vocab = 2_048
    let experts = 16
    let expertWidth = 256
    // GLM's dense first layer leaves six MoE layers too: two of each expert type, as each type's pool needs.
    let layers = family == .glm ? 7 : 6
    let architecture = family == .deepseek ? "deepseek41" : "glm5-next"
    // Eight attention heads in both, so decode takes the eight-heads-a-thread attention kernels the real models take.
    var tensors: [(String, [Int], Int)] = []
    var data: [String: Data] = [:]
    /// A tensor of shape [outputs, width] (or a vector): values uniform within ±scale around `center`.
    func add(_ name: String, _ shape: [Int], type requested: Int? = nil, scale: Float? = nil, center: Float = 0) {
        let count = shape.reduce(1, *)
        let width = shape.last!
        let type = requested ?? (shape.count == 2 && shape[0] >= 32 && width % 32 == 0 ? 8 : 0)
        guard weights else { return tensors.append((name, shape, type)) }
        let amplitude = scale ?? (3 / Float(width)).squareRoot()
        let values = (0..<count).map { _ in center + amplitude * rng.unit() }
        var bytes = Data(count: GGUFQuantize.bytes(type, count: count))
        if let scale = [11: 108, 16: 0][type] {
            // Q3_K's d is its block's last half, IQ2_XXS's its first: 2^-11 keeps every value within about ±0.08.
            var fill = SyntheticRNG(state: name.utf8.reduce(UInt64(0x51DE)) { $0 &* 31 &+ UInt64($1) })
            let block = GGUFQuant.sizes[type]!.bytes
            bytes.withUnsafeMutableBytes { out in
                for i in 0..<out.count { out[i] = UInt8(truncatingIfNeeded: fill.next()) }
                for b in stride(from: 0, to: out.count, by: block) {
                    out.storeBytes(of: Float16(0x1p-11).bitPattern, toByteOffset: b + scale, as: UInt16.self)
                }
            }
        } else {
            values.withUnsafeBufferPointer { source in
                bytes.withUnsafeMutableBytes { GGUFQuantize.encode(source.baseAddress!, count: count, type: type, into: $0.baseAddress!) }
            }
        }
        tensors.append((name, shape, type))
        data[name] = bytes
    }
    add("token_embd.weight", [vocab, hidden], type: 8, scale: 1)
    add("output.weight", [vocab, hidden], type: 8)
    add("output_norm.weight", [hidden], scale: 0.1, center: 1)
    var config: [String: Any] = [
        "num_hidden_layers": layers, "hidden_size": hidden, "vocab_size": vocab, "n_routed_experts": experts,
        "num_experts_per_tok": 4, "moe_intermediate_size": expertWidth, "n_shared_experts": 1, "hc_mult": 4, "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6, "swiglu_limit": 10.0, "norm_topk_prob": true, "max_position_embeddings": 8_192, "index_n_heads": 2,
        "index_head_dim": 32,
    ]
    let moeLayers = family == .glm ? Array(1..<layers) : Array(0..<layers)
    for l in 0..<layers {
        add("blk.\(l).attn_norm.weight", [hidden], scale: 0.1, center: 1)
        add("blk.\(l).ffn_norm.weight", [hidden], scale: 0.1, center: 1)
        for part in ["attn", "ffn"] {
            add("blk.\(l).hc_\(part)_fn.weight", [24, 4 * hidden], type: 0, scale: 0.05)
            add("blk.\(l).hc_\(part)_scale.weight", [3], scale: 0.2, center: 0.8)
            add("blk.\(l).hc_\(part)_base.weight", [24], scale: 0.2)
        }
        if let n = moeLayers.firstIndex(of: l) {
            add("blk.\(l).ffn_gate_inp.weight", [experts, hidden], type: 0, scale: 0.3)
            add("blk.\(l).exp_probs_b.bias", [experts], scale: 0.05)
            let types = expertTypes[l] ?? Array(repeating: [12, 10, 8][n % 3], count: 3)
            add("blk.\(l).ffn_gate_exps.weight", [experts, expertWidth, hidden], type: types[0])
            add("blk.\(l).ffn_up_exps.weight", [experts, expertWidth, hidden], type: types[1])
            add("blk.\(l).ffn_down_exps.weight", [experts, hidden, expertWidth], type: types[2])
            add("blk.\(l).ffn_gate_shexp.weight", [expertWidth, hidden])
            add("blk.\(l).ffn_up_shexp.weight", [expertWidth, hidden])
            add("blk.\(l).ffn_down_shexp.weight", [hidden, expertWidth])
        } else {
            add("blk.\(l).ffn_gate.weight", [512, hidden])
            add("blk.\(l).ffn_up.weight", [512, hidden])
            add("blk.\(l).ffn_down.weight", [hidden, 512])
        }
    }
    var metadata: [(String, Any)] = [("general.architecture", architecture)]
    if family == .glm {
        // Three KDA layers, then sparse attention (DSA with its pooled indexer), as GLM-5.3-Flash repeats them; layer 0's
        // FFN is dense (first_k_dense_replace).
        let types = (0..<layers).map { $0 % 4 == 3 || $0 == layers - 1 ? "full_attention" : "linear_attention" }
        let (heads, qk, v, qRank, kvRank, kdaHeads) = (8, 64, 64, 128, 64, 2)
        config.merge(
            [
                "model_type": "glm5_next", "layer_types": types, "first_k_dense_replace": 1, "intermediate_size": 512,
                "num_attention_heads": heads, "qk_head_dim": qk, "v_head_dim": v, "q_lora_rank": qRank, "kv_lora_rank": kvRank,
                "qk_rope_head_dim": 0, "index_topk": 64, "index_kpool": 4, "rms_norm_eps": 1e-5, "routed_scaling_factor": 2.5,
                "linear_attn_config": ["num_heads": kdaHeads, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0],
            ]
        ) { $1 }
        let c = kdaHeads * 128
        for (l, kind) in types.enumerated() {
            if kind == "linear_attention" {
                for part in ["q", "k", "v"] {
                    add("blk.\(l).kda_\(part).weight", [c, hidden])
                    add("blk.\(l).kda_\(part)_conv.weight", [4, c], scale: 0.4)
                }
                for part in ["f", "g"] {
                    add("blk.\(l).kda_\(part)_a.weight", [128, hidden])
                    add("blk.\(l).kda_\(part)_b.weight", [c, 128])
                }
                add("blk.\(l).kda_beta.weight", [kdaHeads, hidden])
                add("blk.\(l).kda_output.weight", [hidden, c])
                add("blk.\(l).kda_dt_bias.weight", [c], scale: 0.5)
                add("blk.\(l).kda_a_log.weight", [kdaHeads], scale: 0.5, center: -0.5)
                add("blk.\(l).kda_o_norm.weight", [128], scale: 0.1, center: 1)
            } else {
                add("blk.\(l).attn_q_a.weight", [qRank, hidden])
                add("blk.\(l).attn_q_a_norm.weight", [qRank], scale: 0.1, center: 1)
                add("blk.\(l).attn_q_b.weight", [heads * qk, qRank])
                add("blk.\(l).attn_kv_a_mqa.weight", [kvRank, hidden])
                add("blk.\(l).attn_kv_a_norm.weight", [kvRank], scale: 0.1, center: 1)
                add("blk.\(l).attn_k_b.weight", [heads * kvRank, qk])
                add("blk.\(l).attn_v_b.weight", [heads * v, kvRank])
                add("blk.\(l).attn_output.weight", [hidden, heads * v])
                add("blk.\(l).indexer.attn_q_b.weight", [2 * 32, qRank])
                add("blk.\(l).indexer.attn_k.weight", [32, hidden])
                add("blk.\(l).indexer.proj.weight", [2, hidden])
                add("blk.\(l).indexer.pool_gate.weight", [32, hidden])
                add("blk.\(l).indexer.k_norm.weight", [32], scale: 0.1, center: 1)
                add("blk.\(l).indexer.k_norm.bias", [32], scale: 0.1)
                add("blk.\(l).indexer.pool_ape.weight", [4, 32], scale: 0.1)
            }
        }
    } else {
        // Every attention kind DeepSeek-V4.1-Flash has: window only (layer 0), pairs compressed (ratio 2) and every token
        // (ratio 1) as key/value sources, a layer reading an earlier source, an index-only source, the candidate-block
        // layer, and Engram on two layers.
        let ratios = [0, 2, 2, 1, 2, 2]
        let kvSources = [1, 3, 4]
        let indexSources = [1, 2, 3, 4]
        let engramLayers = [0, 3]
        let (heads, headDim, qRank, groups, oRank) = (8, 64, 64, 2, 32)
        let (ngram, engramHeads, engramDim) = (3, 2, 32)
        config.merge(
            [
                "model_type": "deepseek_v41", "num_attention_heads": heads, "head_dim": headDim, "qk_rope_head_dim": 16,
                "q_lora_rank": qRank, "o_groups": groups, "o_lora_rank": oRank, "sliding_window": 16, "compress_ratios": ratios,
                "kv_source_layer_ids": kvSources, "index_source_layer_ids": indexSources, "index_topk": 32,
                "candidate_source_layer_id": 4, "candidate_topk_blocks": 16, "candidate_block_size": 8, "rope_theta": 10_000,
                "compress_rope_theta": 160_000,
                "rope_scaling": ["factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 1_024],
                "rms_norm_eps": 1e-6, "routed_scaling_factor": 1.5, "engram_layer_ids": engramLayers, "engram_max_ngram_size": ngram,
                "engram_n_heads": engramHeads, "engram_head_dim": engramDim, "engram_pad_token_id": 2,
            ]
        ) { $1 }
        let cols = (ngram - 1) * engramHeads
        let primes = [31, 37, 41, 43, 47, 53, 59, 61]
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
                add("blk.\(l).indexer.attn_k.weight", [32, headDim])
                add("blk.\(l).indexer.k_norm.weight", [32], scale: 0.1, center: 1)
            }
            if indexSources.contains(l) {
                add("blk.\(l).indexer.attn_q_b.weight", [2 * 32, qRank])
                add("blk.\(l).indexer.proj.weight", [2, hidden])
            }
            if let e = engramLayers.firstIndex(of: l) {
                let rows = primes[(e * cols)..<((e + 1) * cols)].reduce(0, +)
                add("blk.\(l).engram_embd.weight", [rows, engramDim], type: 8, scale: 1)
                add("blk.\(l).engram_kv.weight", [5 * hidden, cols * engramDim])
                add("blk.\(l).engram_gate.weight", [4 * hidden], scale: 0.1, center: 1)
            }
        }
        metadata += [
            ("deepseek41.engram.token_map", (0..<vocab).map { ($0 * 7 + 3) % 1_000 }),
            ("deepseek41.engram.multipliers", (0..<(engramLayers.count * ngram)).map { 2 * $0 * 104_729 + 1 }),
            ("deepseek41.engram.primes", primes),
        ]
    }
    let path = folder.appendingPathComponent("synthetic-\(architecture).gguf")
    let json = String(decoding: try JSONSerialization.data(withJSONObject: config, options: .sortedKeys), as: UTF8.self)
    if !tokens.isEmpty { metadata.append(("tokenizer.ggml.tokens", tokens)) }
    try familyGGUF(metadata + [(architecture + ".config", json)], tensors: tensors, data: data, to: path)
    return path
}

struct SyntheticRun: Equatable {
    var reply: [Int] = []
    var continued: [Int] = []
    var digests: [String: String] = [:]
    var counters: [String: Int] = [:]
}

private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private func sha256(_ floats: [Float]) -> String { floats.withUnsafeBytes { sha256(Data($0)) } }
/// Greedy: the first of the largest logits.
private func greedy(_ logits: [Float]) -> Int {
    var best = 0
    for i in logits.indices where logits[i] > logits[best] { best = i }
    return best
}

/// One conversation as the serving backend runs it: the prompt in its slices (`feed` for rows more than the model's tail
/// before the end), a greedy reply, its state saved with `FamilyStateFile`, a continuation on the live state, then the
/// same continuation from the saved state after a reset.
func syntheticConversation(
    _ path: URL, source: GGUFSource? = nil, hotCopy: Bool, flipPrefetch: Bool, in folder: URL, device: MTLDevice, hotDrives: Int = 1
) throws -> SyntheticRun {
    try autoreleasepool {
        var run = SyntheticRun()
        let weights = try FamilyWeights(gguf: source ?? GGUFSource(path), device: device)
        let geometry = weights.geometry
        // Two fifths of the routed experts' bytes (GLM's least: two layers' experts per token in each pool), so records are
        // evicted and read again.
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 4_096) { _ in }
        let runner = model as! FamilyRunner
        // Decode prefetch is off by default for GLM and on for DeepSeek (tuning may change either): both ways give one output.
        if flipPrefetch { runner.guessesNext.toggle() }
        if hotCopy {
            // Half of every layer's experts in a hot copy beside the model: those records come from it, the others from the GGUF.
            let keys = geometry.moeLayers.flatMap { l in stride(from: 0, to: geometry.experts, by: 2).map { l * geometry.experts + $0 } }
            if hotDrives == 1 {
                let copy = try HotCopy.write(
                    source: runner.experts, keys: keys, folder: folder.appendingPathComponent("hot", isDirectory: true),
                    identity: "synthetic")
                runner.experts.attach(copy)
            } else {
                // The same records spread over `hotDrives` stores (one per "drive"), every fourth one on all of them.
                let copies = try (0..<hotDrives).map { drive in
                    try HotCopy.write(
                        source: runner.experts,
                        keys: keys.enumerated().filter { $0.offset % 4 == 0 || $0.offset % hotDrives == drive }.map(\.element),
                        folder: folder.appendingPathComponent("hot-drive-\(drive)", isDirectory: true), identity: "synthetic")
                }
                runner.experts.attach(copies: copies)
            }
        }
        let prompt = syntheticPrompt(vocabulary: geometry.vocab)
        var logits = try readPrompt(prompt, into: model)
        guard logits.count == geometry.vocab, logits.allSatisfy(\.isFinite), Set(logits).count > geometry.vocab / 2 else {
            throw EngineError("\(geometry.family): the synthetic model's logits are degenerate")
        }
        run.digests["prompt_logits"] = sha256(logits)
        runner.beginDecode()
        for _ in 0..<syntheticDecodeTokens {
            run.reply.append(greedy(logits))
            logits = try model.forward([run.reply.last!])
        }
        let tokens = prompt + run.reply
        let state = folder.appendingPathComponent("state-\(hotCopy)-\(flipPrefetch)", isDirectory: true)
        try FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)
        try FamilyStateFile.write(model.stateParts(at: tokens.count), logits: logits, tokens: tokens, epoch: "synthetic", to: state)
        run.digests["reply_logits"] = sha256(logits)
        run.digests["state"] = sha256(try Data(contentsOf: FamilyStateFile.dataURL(state)))
        func continueReply(from start: [Float]) throws -> ([Int], [Float]) {
            var logits = start
            var out: [Int] = []
            for _ in 0..<syntheticContinueTokens {
                out.append(greedy(logits))
                logits = try model.forward([out.last!])
            }
            return (out, logits)
        }
        let live = try continueReply(from: logits)
        run.continued = live.0
        run.digests["continued_logits"] = sha256(live.1)
        try model.reset()
        let header = try FamilyStateFile.header(state, epoch: "synthetic", prompt: tokens)
        let saved = try FamilyStateFile.read(
            state, header: header, into: model.stateParts(at: tokens.count), expectedLogits: model.vocabulary)
        model.restored(tokens)
        runner.beginDecode()
        let restored = try continueReply(from: saved ?? [])
        if saved != logits || restored.0 != live.0 || restored.1.map(\.bitPattern) != live.1.map(\.bitPattern) {
            throw EngineError("\(geometry.family): the continuation from the saved state differs from the live one")
        }
        let c = runner.experts.counters
        try requireFamilyMirror(runner.experts, "\(geometry.family)")
        run.counters = [
            "prefetch": runner.guessesNext ? 1 : 0, "slots": runner.experts.slotCount, "misses": c.misses, "hits": c.hits,
            "hot_reads": c.hotReads, "hedges": c.hedges,
            "prefetches": c.prefetches, "read_aheads": c.readAheads, "routed_bytes": routed, "pool_bytes": runner.experts.poolBytes,
        ]
        return run
    }
}

private func syntheticPrompt(vocabulary: Int) -> [Int] {
    var rng = SyntheticRNG(state: 0x9_0000 + UInt64(vocabulary))
    return (0..<syntheticPromptTokens).map { _ in Int(rng.next() % UInt64(vocabulary)) }
}

/// The prompt as the serving backend reads it: in its slices, `feed` for rows more than the model's tail before the end.
private func readPrompt(_ prompt: [Int], into model: FamilyModel, plan: Bool = true) throws -> [Float] {
    if plan { (model as! FamilyRunner).planPrompt(rows: prompt.count) }
    try (model as! FamilyRunner).beginPrefill()
    var logits: [Float] = []
    var first = 0
    while first < prompt.count {
        let end = prompt.count
        let slice = FamilyRunner.promptSlice
        func fed(_ last: Int) -> Bool { model.promptTail.map { last + $0 <= end } ?? false }
        var last = min(end, first + slice)
        var fedEnd = fed(last) ? last : first
        while last < end, last - first < model.promptRows {
            last = min(end, last + slice)
            if fed(last) { fedEnd = last }
        }
        if fedEnd == last {
            try model.feed(Array(prompt[first..<last]))
        } else {
            logits = try model.forward(Array(prompt[first..<last]), fed: fedEnd - first)
        }
        first = last
    }
    return logits
}

/// A prompt of more slices than a call takes by default (`promptSlices`) runs as one call with the working memory lent
/// by the expert pool (`planPrompt`): every record is asked for once (the default's two calls ask again for those both
/// use; DeepSeek's fed slices run in the same pass as its whole one), the logits and state are the default's bits, the
/// ledger stays within its limit with the pool shrunk by the loan, and decode gets the pool back.
private func syntheticPromptLoan(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-loan-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) prompt loan: " + what) }
    }
    let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
    let limit: UInt64 = 8 << 30
    // One slice a call by default, as on a Mac whose reserve holds no second one.
    FamilyRunner.promptReserve = 0
    defer { FamilyRunner.promptReserve = nil }
    func read(plan: Bool) throws -> (bits: [String], asked: Int, calls: Int, runner: FamilyRunner, ledger: MemoryLedger, pool: UInt64) {
        try autoreleasepool {
            let ledger = try MemoryLedger(ceiling: limit + 1, transientMargin: 1, availableFloor: 1)
            let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: limit, context: 16_384, ledger: ledger) { _ in }
            let runner = model as! FamilyRunner
            ledger.setUsable(limit)
            runner.fitExperts()
            var rng = SyntheticRNG(state: 0x10A2)
            let prompt = (0..<(runner.promptSlices * FamilyRunner.promptSlice + 100)).map { _ in
                Int(rng.next() % UInt64(weights.geometry.vocab))
            }
            let pool = runner.experts.budget
            let before = runner.experts.counters
            let calls = runner.timing.calls
            let logits = try readPrompt(prompt, into: model, plan: plan)
            let after = runner.experts.counters
            let bits =
                [sha256(logits)] + model.stateParts(at: prompt.count).map { sha256(Data(bytes: $0.buffer.contents(), count: $0.bytes)) }
            return (bits, after.hits + after.misses - before.hits - before.misses, runner.timing.calls - calls, runner, ledger, pool)
        }
    }
    let base = try read(plan: false)
    let lent = try read(plan: true)
    let runner = lent.runner
    let records = weights.geometry.moeLayers.count * weights.geometry.experts
    try require(
        runner.prefillSlices == runner.promptSlices + 1 && lent.calls == 1 && base.calls == 2, "calls \(base.calls) -> \(lent.calls)")
    try require(lent.asked <= records && base.asked > lent.asked, "records asked for \(base.asked) -> \(lent.asked) of \(records)")
    try require(lent.bits == base.bits, "the lent call changed the logits or state")
    let maximum = runner.poolMaximum + runner.workingBytes
    try require(
        runner.workingBytes > base.runner.workingBytes && lent.ledger.reservedBytes <= lent.ledger.maximumUsableBytes
            && runner.experts.budget <= runner.poolTarget(),
        "working \(runner.workingBytes), reserved \(lent.ledger.reservedBytes) of \(lent.ledger.maximumUsableBytes)")
    runner.beginDecode()
    try require(
        runner.prefillSlices == runner.promptSlices && runner.workingBytes == 0 && runner.poolMaximum == maximum
            && runner.experts.budget == runner.poolTarget() && runner.experts.budget >= lent.pool,
        "decode did not get the pool back: \(runner.experts.budget) after \(lent.pool)")
}

/// The working memory a long prompt borrows from the expert pool (`planPrompt`) is back as soon as a served prompt ends
/// without a reply, as after one with a reply: cancelled inside its first call, and failed (the GPU interrupted on every
/// attempt), the prompt's working allowance is gone, its grouping is the default again and the pool holds its whole
/// target, not at the next request.
private func syntheticPromptLoanReturned(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-loan-back-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) prompt loan returned: " + what) }
    }
    let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: weights.geometry.vocab - 261)
    let limit: UInt64 = 8 << 30
    FamilyRunner.promptReserve = 0
    defer { FamilyRunner.promptReserve = nil }
    var rng = SyntheticRNG(state: 0xB0_44)
    let letters = Array("abcdefghjklmnopqrstuvwxyz0123456789,.;:-+=")
    let text = String((0..<(3 * FamilyRunner.promptSlice + 100)).map { _ in letters[Int(rng.next() % UInt64(letters.count))] })
    for how in ["replied", "cancelled", "failed"] {
        try autoreleasepool {
            let ledger = try MemoryLedger(ceiling: limit + 1, transientMargin: 1, availableFloor: 1)
            let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: limit, context: 16_384, ledger: ledger) { _ in }
            let runner = model as! FamilyRunner
            ledger.setUsable(limit)
            runner.fitExperts()
            let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
            let request = try ServingRequest.parse(
                JSONSerialization.data(withJSONObject: [
                    "model": "synthetic", "temperature": 0, "reasoning_effort": "none", "max_tokens": 2,
                    "messages": [["role": "user", "content": text]],
                ]), contextTier: 16_384)
            let budget = runner.experts.budget
            try require(runner.promptSlices == 1, "the reserve holds \(runner.promptSlices) slices")
            let control = GenerationControl()
            final class Seen: @unchecked Sendable {
                let lock = NSLock()
                var done = false, lent = false
            }
            let seen = Seen()
            Thread.detachNewThread {
                let until = Date().addingTimeInterval(60)
                var step: String?
                while !seen.lock.withLock({ seen.done }), Date() < until {
                    // The prompt planned more slices a call than the reserve holds: the pool lends their memory.
                    if runner.prefillSlices > runner.promptSlices, !seen.lock.withLock({ seen.lent }) {
                        seen.lock.withLock { seen.lent = true }
                        step = control.progress.activity
                    }
                    // Cancelled once the lent call's work has begun.
                    if how == "cancelled", seen.lock.withLock({ seen.lent }), control.progress.activity != step { control.cancel() }
                    if how == "failed", runner.interruptWait == nil { runner.interruptWait = 0 }
                    usleep(50)
                }
            }
            defer { seen.lock.withLock { seen.done = true } }
            do {
                _ = try backend.prepare(request, cancellation: control).run { _ in true }
                if how == "replied" {
                    try require(seen.lock.withLock { seen.lent }, "the prompt borrowed nothing")
                    return
                }
                throw EngineError("synthetic \(name) prompt loan returned: the \(how) request ran to its end")
            } catch {
                if how == "replied" || "\(error)".contains("ran to its end") { throw error }
            }
            seen.lock.withLock { seen.done = true }
            usleep(2_000)
            runner.interruptWait = nil
            try require(seen.lock.withLock { seen.lent }, "\(how): the prompt borrowed nothing (working \(runner.workingBytes))")
            try require(
                runner.workingBytes == 0 && runner.prefillSlices == runner.promptSlices && runner.experts.budget >= budget
                    && runner.experts.budget == runner.poolTarget(),
                "\(how): working \(runner.workingBytes), \(runner.prefillSlices) slices a call, budget "
                    + "\(runner.experts.budget) after \(budget)")
        }
    }
}

/// Where a prompt's command buffers end changes nothing: the synthetic model's prompt read with a buffer committed at
/// every point the engine can commit, with a limit of four of the largest steps between two such points (no buffer then
/// holds more than the limit and one step), and with the starting limit gives the same logits and state bits; and an
/// abort faked mid-prompt (macOS's Impacting Interactivity), retried as serving retries a request, gives them too and
/// halves the limit for the retry.
private func syntheticPromptBuffers(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) prompt buffers: " + what) }
    }
    try autoreleasepool {
        let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 4_096) { _ in }
        let runner = model as! FamilyRunner
        let prompt = syntheticPrompt(vocabulary: weights.geometry.vocab)
        func read() throws -> [String] {
            try model.reset()
            let logits = try readPrompt(prompt, into: model)
            return [sha256(logits)] + model.stateParts(at: prompt.count).map { sha256(Data(bytes: $0.buffer.contents(), count: $0.bytes)) }
        }
        runner.commitLimit = 1
        var commits = runner.timing.commits
        let split = try read()
        let step = runner.timing.largestWork
        let splitBuffers = runner.timing.commits - commits
        runner.commitLimit = 4 * step
        let bounded = try read()
        try require(
            step > 0 && runner.timing.largestWork <= 5 * step,
            "a buffer held \(runner.timing.largestWork) multiply-adds at a limit of \(4 * step)")
        runner.commitLimit = FamilyRunner.commitWork
        commits = runner.timing.commits
        let whole = try read()
        try require(runner.timing.commits - commits < splitBuffers, "\(splitBuffers) buffers split, \(runner.timing.commits - commits) not")
        try require(split == whole && bounded == whole, "the buffer boundaries changed the output")
        let retry = GPURetry(log: nil)
        runner.interruptWait = 3
        let retried = try retry.run { try read() }
        try require(retry.attempt == 1 && runner.interruptWait == nil, "the faked abort was not retried once")
        try require(runner.commitLimit <= FamilyRunner.commitWork / 2, "the retry ran buffers of \(runner.commitLimit)")
        try require(retried == whole, "the retried prompt's logits or state differ")
    }
}

private func syntheticEndToEnd(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try syntheticFamilyGGUF(family, in: folder)
    let plain = try syntheticConversation(path, hotCopy: false, flipPrefetch: false, in: folder, device: device)
    let hot = try syntheticConversation(path, hotCopy: true, flipPrefetch: false, in: folder, device: device)
    let flipped = try syntheticConversation(path, hotCopy: true, flipPrefetch: true, in: folder, device: device)
    let drives = try syntheticConversation(path, hotCopy: true, flipPrefetch: false, in: folder, device: device, hotDrives: 3)
    let name = family == .glm ? "glm" : "deepseek"
    func fail(_ what: String) -> EngineError { EngineError("synthetic \(name): " + what) }
    // The budget must make the cache work: misses beyond the slots (records read again), hot-copy reads only with a copy,
    // prefetches only with prefetch on, prompt read-ahead.
    for (run, copy) in [(plain, false), (hot, true), (flipped, true), (drives, true)] {
        let c = run.counters
        guard c["pool_bytes"]! < c["routed_bytes"]!, c["misses"]! > c["slots"]!, (c["hot_reads"]! > 0) == copy,
            (c["prefetches"]! > 0) == (c["prefetch"] == 1), c["read_aheads"]! > 0, c["hedges"] == 0
        else { throw fail("the expert cache was not exercised as intended: \(c)") }
    }
    for (other, what) in [(hot, "a hot copy"), (flipped, "switching prefetch"), (drives, "copies on three drives")]
    where plain.reply != other.reply || plain.continued != other.continued || plain.digests != other.digests {
        throw fail("\(what) changed the output: \(plain.reply) \(plain.digests) vs \(other.reply) \(other.digests)")
    }
    let reference: [String: Any] = ["reply": plain.reply, "continued": plain.continued, "digests": plain.digests]
    let file = CheckFixtures.file("synthetic/\(name).json")
    if ProcessInfo.processInfo.environment["ORIGAMI_SYNTHETIC_WRITE_REFERENCE"] == "1" {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: reference, options: [.prettyPrinted, .sortedKeys]).write(to: file)
        FileHandle.standardError.write(
            Data("synthetic \(name): wrote \(file.path) (\(plain.counters), hot \(hot.counters), flipped \(flipped.counters))\n".utf8))
        return
    }
    let stored = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
    let digests = stored["digests"] as? [String: String] ?? [:]
    let differ =
        (stored["reply"] as? [Int] == plain.reply ? [] : ["reply tokens"])
        + (stored["continued"] as? [Int] == plain.continued ? [] : ["continued tokens"])
        + plain.digests.keys.sorted().filter { digests[$0] != plain.digests[$0] }
    guard differ.isEmpty else { throw fail("\(differ.joined(separator: ", ")) differ from \(file.path)") }
}

/// A cancel while the serving backend reads a follow-up turn's prompt, on the synthetic model's real forward pass: the
/// conversation goes back to the last point read whole (any slice end for GLM; for DeepSeek one with no fed row within
/// its tail, else the turn's start) and the resent request reads only the rows after it. Its reply tokens, the logits and
/// state saved at the prompt's end and the whole state after the reply are the bits of the same two turns never
/// cancelled. Cancels land inside a slice and at the last slice (once its layer steps have begun while the position is that slice's
/// start) and between two slices (as the position reaches the boundary). A DeepSeek turn too short to keep anything
/// still resets: the same reply, with the bits of the follow-up read cold. Each case also runs with a hot copy and prefetch switched.
private func syntheticPromptCancelResumes(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-cancel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) prompt cancel: " + what) }
    }
    let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
    let geometry = weights.geometry
    let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
    // Every id the model can pick decodes; the turns' text is one byte a token ("H" and "i", the one merge, left out).
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: geometry.vocab - 261)
    var rng = SyntheticRNG(state: 0xCA_0CE1)
    let letters = Array("abcdefghjklmnopqrstuvwxyz0123456789,.;:-+=")
    func text(_ count: Int) -> String { String((0..<count).map { _ in letters[Int(rng.next() % UInt64(letters.count))] }) }
    let slice = FamilyRunner.promptSlice
    let hotKeys = geometry.moeLayers.flatMap { l in stride(from: 0, to: geometry.experts, by: 2).map { l * geometry.experts + $0 } }
    var hotFolder: URL?
    func request(_ body: [String: Any]) throws -> ServingRequest {
        var body = body
        body["model"] = "synthetic"
        body["temperature"] = 0
        body["reasoning_effort"] = "none"
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8_192)
    }
    struct Result: Equatable {
        var reply: [Int] = [], promptLogits = "", promptState = "", state = "", reused = 0, kept: Int?
    }
    /// The two turns; with `cancel`, the follow-up is cancelled once, then resent (as `resend` when given).
    func conversation(
        opening: String, followUp: String, hot: Bool, cancel: ((FamilyRunner, GenerationControl) -> Void)?, opened: Bool = true,
        resend: String? = nil
    ) throws -> Result {
        try autoreleasepool {
            let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 8_192) { _ in }
            let runner = model as! FamilyRunner
            // One slice a pass (the default groups several when memory allows), so every slice end is a point to keep.
            runner.prefillSlices = 1
            if hot {
                runner.guessesNext.toggle()
                if hotFolder == nil {
                    hotFolder = folder.appendingPathComponent("hot", isDirectory: true)
                    _ = try HotCopy.write(source: runner.experts, keys: hotKeys, folder: hotFolder!, identity: "synthetic")
                }
                runner.experts.attach(HotCopy.open(folder: hotFolder!, identity: "synthetic"))
            }
            let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
            let cache = try PromptCache(
                root: folder.appendingPathComponent("cache-\(UUID().uuidString)"), epoch: "synthetic", entries: 3,
                entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger, capacity: 8_192)
            let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
            backend.promptCache = cache
            backend.epoch = "synthetic"
            backend.keepAlive = try RequestGPUKeepAlive(ledger: ledger)  // on, as served: the references hold with it
            // The opening reply is forced ("XYZ") so the follow-up's text holds its exact ids; it is read as decode rows.
            let first = try request([
                "messages": [["role": "user", "content": opening]], "max_tokens": 3, "origami_teacher_tokens": [88, 89, 90],
            ])
            if opened { _ = try backend.prepare(first, cancellation: GenerationControl()).run { _ in true } }
            func turn(_ followUp: String) throws -> ServingRequest {
                try request([
                    "messages": [
                        ["role": "user", "content": opening], ["role": "assistant", "content": "XYZ"],
                        ["role": "user", "content": followUp],
                    ],
                    "max_tokens": 6, "origami_return_tokens": true,
                ])
            }
            let second = try turn(resend ?? followUp)
            var result = Result()
            if let cancel {
                let control = GenerationControl()
                cancel(runner, control)
                do {
                    _ = try backend.prepare(turn(followUp), cancellation: control).run { _ in true }
                    throw EngineError("synthetic \(name) prompt cancel: the cancelled request ran to its end")
                } catch { if !control.isCancelled { throw error } }
                result.kept = runner.position
            }
            let stats = try backend.prepare(second, cancellation: GenerationControl()).run { _ in true }
            result.reply = stats.outputTokens ?? []
            result.reused = stats.cachedTokens
            var whole = SHA256()
            for part in model.stateParts(at: model.position) {
                whole.update(bufferPointer: UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes))
            }
            result.state = whole.finalize().map { String(format: "%02x", $0) }.joined()
            let ids = try ServingPrompt.encode(second, tokenizer: tokenizer).map(Int64.init)
            let saved = try cache.candidates(for: ids).map { try Data(contentsOf: FamilyStateFile.dataURL(cache.stateFolder($0.slot))) }
            guard let entry = saved.max(by: { $0.count < $1.count }), entry.count > geometry.vocab * 4 else {
                throw EngineError("synthetic \(name) prompt cancel: the follow-up's prompt end was not saved")
            }
            result.promptLogits = sha256(entry.suffix(geometry.vocab * 4))
            result.promptState = sha256(entry.prefix(entry.count - geometry.vocab * 4))
            return result
        }
    }
    /// Cancels inside the slice starting at `start`: once the prompt's reported layer step changes while the position is
    /// there (the slice's work has begun, and a slice of 2,048 rows outlasts the polling by far).
    func inside(_ start: Int) -> (FamilyRunner, GenerationControl) -> Void {
        { runner, control in
            Thread.detachNewThread {
                let until = Date().addingTimeInterval(60)
                while runner.position < start, Date() < until { usleep(50) }
                let step = control.progress.activity
                while runner.position == start, control.progress.activity == step, Date() < until { usleep(50) }
                control.cancel()
            }
        }
    }
    /// Cancels as soon as the position reaches `boundary` (at the next check: before the next slice or inside it).
    func between(_ boundary: Int) -> (FamilyRunner, GenerationControl) -> Void {
        { runner, control in
            Thread.detachNewThread {
                let until = Date().addingTimeInterval(60)
                while runner.position < boundary, Date() < until, !control.isCancelled { usleep(50) }
                control.cancel()
            }
        }
    }
    // The opening turn's rows: the opening's tokens and two of the forced reply's.
    let opening = text(700)
    let start = 700 + 2
    // The follow-up's new rows: "Z" and its text. GLM: four slices. DeepSeek: a fed slice, then two read whole, the last
    // shorter than the tail, so the second's end is a whole point and the first's a fed one (kept for prompts that reach
    // the tail past it).
    let deepSeekTail =
        family == .deepseek
        ? try autoreleasepool {
            try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 8_192) { _ in }.promptTail ?? 0
        } : 0
    let followUp = family == .glm ? text(3 * slice + 500) : text(2 * slice + max(1, deepSeekTail / 2) - 1)
    var cases: [(String, (FamilyRunner, GenerationControl) -> Void, kept: Int)] = []
    if family == .glm {
        cases = [
            ("inside the second slice", inside(start + slice), start + slice),
            ("between the second and third slices", between(start + 2 * slice), start + 2 * slice),
            ("inside the last slice", inside(start + 3 * slice), start + 3 * slice),
        ]
    } else {
        cases = [
            ("inside the second slice (its start is fed)", inside(start + slice), start + slice),
            ("between the fed slice and the next", between(start + slice), start + slice),
            ("inside the last slice", inside(start + 2 * slice), start + 2 * slice),
        ]
    }
    let reference = try conversation(opening: opening, followUp: followUp, hot: false, cancel: nil)
    try require(reference.reused == start && reference.reply.count == 6, "the follow-up did not continue live: \(reference)")
    for hot in [false, true] {
        for (what, cancel, kept) in cases {
            var got = try conversation(opening: opening, followUp: followUp, hot: hot, cancel: cancel)
            try require(
                got.kept == kept && got.reused == kept,
                "\(what)\(hot ? " (hot copy)" : ""): kept \(got.kept ?? -1), reused \(got.reused), not \(kept)")
            got.kept = nil
            got.reused = reference.reused
            try require(got == reference, "\(what)\(hot ? " (hot copy)" : ""): not bitwise: \(got) vs \(reference)")
        }
    }
    guard family == .deepseek else { return }
    // Kept at the fed slice's end, a prompt ending within the tail past it does not continue from there: its last rows
    // would read the fed rows' missing layers.
    let within = String(followUp.prefix(slice + deepSeekTail / 2))
    let shorter = try conversation(opening: opening, followUp: followUp, hot: false, cancel: between(start + slice), resend: within)
    try require(
        shorter.kept == start + slice && shorter.reused < start + slice,
        "a prompt within the tail of the fed point continued from it: kept \(shorter.kept ?? -1), reused \(shorter.reused)")
    // A turn that starts too early to keep (under 512 rows) and is cancelled inside its first slice: the reset.
    let short = text(300)
    // A reset reads the whole follow-up again, its opening rows in other slices than the opening turn had: the reply is
    // the same, its bits are those of the follow-up sent first (read cold), not of the two turns, the saved prompt end
    // included (the reset clears the running buffers the aborted read wrote).
    let twoTurns = try conversation(opening: short, followUp: followUp, hot: false, cancel: nil)
    let cold = try conversation(opening: short, followUp: followUp, hot: false, cancel: nil, opened: false)
    for hot in [false, true] {
        let got = try conversation(opening: short, followUp: followUp, hot: hot, cancel: inside(300 + 2))
        let label = "short turn\(hot ? " (hot copy)" : "")"
        try require(got.kept == 0 && got.reused == 0, "\(label): kept \(got.kept ?? -1), reused \(got.reused)")
        try require(
            got.reply == cold.reply && got.reply == twoTurns.reply && got.promptLogits == cold.promptLogits
                && got.promptState == cold.promptState && got.state == cold.state,
            "\(label): not bitwise after the reset: \(got) vs \(cold)")
    }
}

/// A request cancelled while it restores a saved prompt (after a restart) keeps the restored state instead of resetting:
/// the restore finishes, the request answers cancelled, and the resent request continues from that state, its reply,
/// state and saved prompt end the bits of the same request never cancelled.
private func syntheticRestoreCancelKeeps(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-restore-cancel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) restore cancel: " + what) }
    }
    let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
    let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: weights.geometry.vocab - 261)
    var rng = SyntheticRNG(state: 0x5E_57C0)
    let letters = Array("abcdefghjklmnopqrstuvwxyz0123456789,.;:-+=")
    func text(_ count: Int) -> String { String((0..<count).map { _ in letters[Int(rng.next() % UInt64(letters.count))] }) }
    let opening = text(900)
    let followUp = text(FamilyRunner.promptSlice + 300)
    let cacheRoot = folder.appendingPathComponent("cache")
    func request(_ messages: [[String: String]], tokens: Int) throws -> ServingRequest {
        let body: [String: Any] = [
            "model": "synthetic", "temperature": 0, "reasoning_effort": "none", "messages": messages, "max_tokens": tokens,
            "origami_return_tokens": true,
        ]
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8_192)
    }
    let first = try request([["role": "user", "content": opening]], tokens: 3)
    let second = try request(
        [["role": "user", "content": opening], ["role": "assistant", "content": "XYZ"], ["role": "user", "content": followUp]], tokens: 6)
    /// A new process on the shared saved prompts: the model, its backend and their cache.
    func server() throws -> (FamilyModel, FamilyServingBackend, PromptCache) {
        let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 8_192) { _ in }
        let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
        let cache = try PromptCache(
            root: cacheRoot, epoch: "synthetic", entries: 3, entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger,
            capacity: 8_192)
        let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
        backend.promptCache = cache
        backend.epoch = "synthetic"
        return (model, backend, cache)
    }
    struct Result: Equatable { var reply: [Int] = [], reused = 0, state = "", saved = "" }
    func finish(_ model: FamilyModel, _ backend: FamilyServingBackend, _ cache: PromptCache) throws -> Result {
        let stats = try backend.prepare(second, cancellation: GenerationControl()).run { _ in true }
        var whole = SHA256()
        for part in model.stateParts(at: model.position) {
            whole.update(bufferPointer: UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes))
        }
        let ids = try ServingPrompt.encode(second, tokenizer: tokenizer).map(Int64.init)
        let saved = try cache.candidates(for: ids).map { try Data(contentsOf: FamilyStateFile.dataURL(cache.stateFolder($0.slot))) }
        return Result(
            reply: stats.outputTokens ?? [], reused: stats.cachedTokens,
            state: whole.finalize().map { String(format: "%02x", $0) }.joined(),
            saved: sha256(saved.max { $0.count < $1.count } ?? Data()))
    }
    let ids = try ServingPrompt.encode(second, tokenizer: tokenizer).map(Int64.init)
    let opened = try autoreleasepool { () throws -> Set<Int> in
        let (_, backend, cache) = try server()
        _ = try backend.prepare(first, cancellation: GenerationControl()).run { _ in true }
        return Set(cache.candidates(for: ids).map(\.slot))
    }
    let restored = try ServingPrompt.encode(first, tokenizer: tokenizer).count
    let reference = try autoreleasepool { () throws -> Result in
        let (model, backend, cache) = try server()
        return try finish(model, backend, cache)
    }
    try require(reference.reused == restored && reference.reply.count == 6, "the follow-up did not restore the opening: \(reference)")
    // The reference saved the follow-up's prompt end in the opening's slot (one chat, one slot): remove it and read the
    // opening again, so the cancelled request restores the opening as the reference did.
    try autoreleasepool {
        let (_, _, cache) = try server()
        for candidate in cache.candidates(for: ids) { cache.forget(candidate.slot) }
    }
    try autoreleasepool {
        let (_, backend, cache) = try server()
        _ = try backend.prepare(first, cancellation: GenerationControl()).run { _ in true }
        try require(Set(cache.candidates(for: ids).map(\.slot)).count == opened.count, "the opening was not saved again")
    }
    let got = try autoreleasepool { () throws -> (kept: Int, result: Result) in
        let (model, backend, cache) = try server()
        let control = GenerationControl()
        backend.restoreProbe = { control.cancel() }
        do {
            _ = try backend.prepare(second, cancellation: control).run { _ in true }
            throw EngineError("synthetic \(name) restore cancel: the cancelled request ran to its end")
        } catch { if !control.isCancelled { throw error } }
        backend.restoreProbe = nil
        let kept = model.position
        return (kept, try finish(model, backend, cache))
    }
    try require(got.kept == restored && got.result.reused == restored, "kept \(got.kept), reused \(got.result.reused), not \(restored)")
    try require(got.result == reference, "not bitwise: \(got.result) vs \(reference)")
}

/// Memory pressure (the ledger refusing optional growth for a whole request) keeps prompt read-ahead and decode guesses
/// working inside the slots the cache already has: a second request with the cache already grown reads ahead about as
/// much as without pressure, waits on about as many demand misses, allocates no slab, and gives the same bits.
private func syntheticReadAheadUnderPressure(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let name = family == .glm ? "glm" : "deepseek"
    func require(_ condition: Bool, _ what: String) throws {
        if !condition { throw EngineError("synthetic \(name) read-ahead under pressure: " + what) }
    }
    let path = try syntheticFamilyGGUF(family, in: folder)
    func run(pressure: Bool) throws -> (bits: [String], counters: FamilyExperts.Counters) {
        try autoreleasepool {
            let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
            let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
            let model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 4_096) { _ in }
            let runner = model as! FamilyRunner
            let ledger = try MemoryLedger(ceiling: 1 << 36, transientMargin: 1, availableFloor: 1)
            runner.experts.ledger = ledger
            runner.guessesNext = true
            let prompt = syntheticPrompt(vocabulary: weights.geometry.vocab)
            // The first request grows the cache; the second runs as a request after the budget was cut does.
            _ = try readPrompt(prompt, into: model)
            try model.reset()
            ledger.setOptionalGrowthAllowed(!pressure)
            let allocated = runner.experts.residentBytes
            let before = runner.experts.counters
            var logits = try readPrompt(prompt, into: model)
            var bits = [sha256(logits)]
            runner.beginDecode()
            for _ in 0..<syntheticDecodeTokens { logits = try model.forward([greedy(logits)]) }
            bits.append(sha256(logits))
            runner.experts.waitForReads()
            try require(runner.experts.residentBytes == allocated, "pressure \(pressure): the cache allocated a slab")
            try requireFamilyMirror(runner.experts, "\(family)")
            var c = runner.experts.counters
            (c.readAheads, c.prefetches, c.misses) =
                (c.readAheads - before.readAheads, c.prefetches - before.prefetches, c.misses - before.misses)
            return (bits, c)
        }
    }
    let free = try run(pressure: false)
    let pressed = try run(pressure: true)
    try require(pressed.bits == free.bits, "pressure changed the output")
    let (f, p) = (free.counters, pressed.counters)
    try require(f.readAheads > 0 && f.prefetches > 0, "the fixture issued no read-ahead or guess: \(f)")
    try require(
        10 * p.readAheads >= 9 * f.readAheads && 10 * p.prefetches >= 9 * f.prefetches && 10 * p.misses <= 11 * f.misses + 10,
        "read-aheads \(p.readAheads) against \(f.readAheads), guesses \(p.prefetches) against \(f.prefetches), "
            + "demand misses \(p.misses) against \(f.misses)")
}

/// The saved prompts' canary (`FamilyServingBackend.canary`) on the synthetic model's real forward pass: the same bits run
/// to run and with another expert budget (two and three fifths of the routed experts, and all of them), and the model left reset; a
/// budget, a cache's contents or a drive must never change it.
private func syntheticCanary(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-synthetic-canary-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
    let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: weights.geometry.vocab - 261)
    var canaries: [String] = []
    for budget in [routed * 2 / 5, routed * 3 / 5, routed * 2] {
        try autoreleasepool {
            let model = try makeFamilyModel(weights, expertBudget: UInt64(budget), context: 4_096) { _ in }
            let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
            canaries += [try backend.canary(), try backend.canary()]
            guard model.position == 0 else { throw EngineError("synthetic canary: the model was not left reset") }
        }
    }
    guard Set(canaries).count == 1 else { throw EngineError("synthetic \(family) canary differs by run or budget: \(canaries)") }
}
