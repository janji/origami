import Foundation
import Metal

/// GLM-5.3-Flash (`glm5-next`) on Metal from a GGUF: mHC residual streams (four), recurrent KDA layers and nope-MLA
/// attention with the k-pool DSA indexer, DeepSeek-style MoE with a shared expert, the routed experts read on demand
/// (`FamilyExperts`). Follows llama.cpp's glm5-next graph (MIT). A prompt runs layer by layer over all its tokens,
/// so each layer's experts are read once per prompt, the last layer's only for the last row (the head's); decode is the
/// same code with one row.
package final class GLMModel: FamilyRunner, @unchecked Sendable {
    struct HC {
        let fn: FamilyWeights.Matrix
        let scale, base: MTLBuffer
    }
    struct KDA {
        let q, k, v: FamilyWeights.Matrix
        let convQ, convK, convV: MTLBuffer
        let fA, fB, beta, gA, gB, out: FamilyWeights.Matrix
        let dtBias, aLog, oNorm: MTLBuffer
        let aIsLog: Bool
    }
    struct Indexer {
        let qB, k, proj, gate: FamilyWeights.Matrix
        let kNormW, kNormB, ape: MTLBuffer
    }
    struct DSA {
        let qA, qB, kvA, kB, vB, out: FamilyWeights.Matrix
        let qANorm, kvANorm: MTLBuffer
        let indexer: Indexer?
    }
    struct Layer {
        let attnNorm, ffnNorm: MTLBuffer
        let hcAttn, hcFFN: HC
        let kda: KDA?
        let dsa: DSA?
        let dense: FFN?
        let router: FamilyWeights.Matrix?
        let routerBias: [Float]
        let shared: FFN?
    }

    let layers: [Layer]
    let embed, output: FamilyWeights.Matrix
    let outputNorm: MTLBuffer
    let eps: Float, hcEps: Float, sinkhorn: Int, swiglu: Float, routedScale: Float, normTopK: Bool
    let kdaHeads: Int, kdaDim: Int, lowerBound: Float
    let heads: Int, qkDim: Int, vDim: Int, qRank: Int, kvRank: Int
    let indexHeads: Int, indexDim: Int, indexTop: Int, pool: Int
    // Conversation state.
    private var kdaState: [Int: MTLBuffer] = [:], convState: [Int: MTLBuffer] = [:]
    /// The index key/gate buffers retain only one unfinished pool; latent and pooled rows remain append-only.
    private var latent: [Int: MTLBuffer] = [:], indexKey: [Int: MTLBuffer] = [:], indexGate: [Int: MTLBuffer] = [:],
        pooled: [Int: MTLBuffer] = [:]

    override package func scratchPlan(rows n: Int) -> [String: Int] {
        let c = min(n, chunk)
        let h = hidden
        let width = layers.flatMap { [$0.dense?.gate.outputs, $0.shared?.gate.outputs].compactMap { $0 } }.max() ?? 0
        var plan =
            layers.contains { $0.router != nil }
            ? moeScratchPlan(rows: n, sharedWidth: width)
            : ["ffn-gate": c * width, "ffn-up": c * width]
        let counts = [
            "streams-a": n * 4 * h, "streams-b": n * 4 * h, "embedded": n * h,
            "ffn-in": n * h, "ffn-out": n * h, "hc-pre": n * 4, "hc-post": n * 4, "hc-comb": n * 16,
            "hc-flat": c * 4 * h, "hc-mixes": c * 24, "mix-in": c * h, "mix-out": c * h,
            "mix-pre": c * 4, "mix-post": c * 4, "mix-comb": c * 16, "head-in": h, "logits": output.outputs,
        ]
        plan.merge(counts) { max($0, $1) }
        if layers.contains(where: { $0.kda != nil }) {
            for name in ["kda-raw", "kda-q", "kda-k", "kda-v", "kda-g", "kda-o"] { plan[name] = c * kdaHeads * kdaDim }
            plan["kda-low"] = c * kdaDim
            plan["kda-beta"] = c * kdaHeads
        }
        if layers.contains(where: { $0.dsa != nil }) {
            let selected = indexTop / pool * pool + pool - 1
            plan["dsa-qr"] = c * qRank
            for name in ["dsa-q", "dsa-qh"] { plan[name] = c * heads * qkDim }
            for name in ["dsa-qabs", "dsa-lat"] { plan[name] = c * heads * kvRank }
            for name in ["dsa-vh", "dsa-v"] { plan[name] = c * heads * vDim }
            plan["dsa-sel"] = c * selected
            plan["dsa-scores"] = c * heads * selected
        }
        if layers.contains(where: { $0.dsa?.indexer != nil }) {
            plan["idx-q"] = c * indexHeads * indexDim
            plan["idx-w"] = c * indexHeads
            plan["idx-scores"] = c * max(1, scratchPositionLimit(rows: n) / pool)
            if n > 1 {
                plan["pool-key"] = (c + pool - 1) * indexDim
                plan["pool-gate"] = (c + pool - 1) * indexDim
            }
        }
        if n == 1, guessesNext, layers.contains(where: { $0.router != nil }) {
            plan["guess-in"] = h
            plan["guess-pre"] = 4
            plan["guess-post"] = 4
            plan["guess-comb"] = 16
        }
        return plan
    }
    /// `expertBudget` nil: the expert slots sized from the memory limit (`FamilyRunner.makeExperts`), held to `ledger`'s
    /// budget when the memory governor runs.
    package init(
        weights: FamilyWeights, expertBudget: UInt64?, memoryLimit: UInt64? = nil, context: Int,
        contextFile: ((UInt64) -> URL?)? = nil, ledger: MemoryLedger? = nil, log: (String) -> Void = { _ in }
    ) throws {
        let geo = weights.geometry
        guard geo.family == .glm, geo.hyperConnections == 4 else {
            throw EngineError("GLMModel runs GLM-5.3-Flash with four hyper-connection streams", code: "model_incompatible")
        }
        try weights.beginStartup(ledger: ledger)
        defer { weights.endStartup() }
        func value(_ key: String, _ fallback: Double) -> Double { geo.values[key] ?? fallback }
        let maxCount = ModelGeometry.countLimit
        eps = Float(value("rms_norm_eps", 1e-5))
        hcEps = Float(value("hc_eps", 1e-6))
        sinkhorn = try geo.int("hc_sinkhorn_iters", 20, 0...1000)
        swiglu = Float(value("swiglu_limit", 0))
        routedScale = Float(value("routed_scaling_factor", 1))
        normTopK = value("norm_topk_prob", 1) != 0
        kdaHeads = try geo.int("linear_attn_config.num_heads", 64, 1...1024)
        kdaDim = try geo.int("linear_attn_config.head_dim", 128, 1...1024)
        // GLM-5.3-Flash is trained with a gate lower bound of -5 (its config.json); some GGUF converters leave the key out.
        lowerBound = Float(geo.values["linear_attn_config.gate_lower_bound"] ?? -5)
        heads = try geo.int("num_attention_heads", 64, 1...64)
        qkDim = try geo.int("qk_head_dim", 256, 1...4096)
        vDim = try geo.int("v_head_dim", 256, 1...4096)
        qRank = try geo.int("q_lora_rank", 1536, 1...maxCount)
        kvRank = try geo.int("kv_lora_rank", 512, 4...maxCount)
        indexHeads = try geo.int("index_n_heads", 32, 1...1024)
        indexDim = try geo.int("index_head_dim", 128, 4...4096)
        indexTop = try geo.int("index_topk", 2048, 1...maxCount)
        pool = try geo.int("index_kpool", 4, 1...8)
        guard kdaDim == 128, heads <= 64, kvRank % 4 == 0, indexDim % 4 == 0, pool <= 8,
            Int(value("linear_attn_config.short_conv_kernel_size", 4)) == 4,
            (geo.values["qk_rope_head_dim"] ?? 0) == 0
        else {
            throw EngineError(
                "This GLM-5.3-Flash geometry has no kernels: KDA head dim \(kdaDim), \(heads) heads, latent \(kvRank), pool \(pool), "
                    + "rotary dim \(geo.values["qk_rope_head_dim"] ?? 0)", code: "model_incompatible")
        }
        embed = try weights.matrix("token_embd.weight")
        output = try weights.matrix("output.weight")
        outputNorm = try weights.floatBuffer("output_norm.weight", count: geo.hidden)
        var layers: [Layer] = []
        for l in 0..<geo.layers {
            func hc(_ part: String) throws -> HC {
                HC(
                    fn: try weights.matrix("blk.{L}.hc_\(part)_fn.weight", layer: l),
                    scale: try weights.floatBuffer("blk.{L}.hc_\(part)_scale.weight", layer: l, count: 3),
                    base: try weights.floatBuffer("blk.{L}.hc_\(part)_base.weight", layer: l, count: 24))
            }
            func ffn(_ suffix: String) throws -> FFN {
                FFN(
                    gate: try weights.matrix("blk.{L}.ffn_gate\(suffix).weight", layer: l),
                    up: try weights.matrix("blk.{L}.ffn_up\(suffix).weight", layer: l),
                    down: try weights.matrix("blk.{L}.ffn_down\(suffix).weight", layer: l))
            }
            var kda: KDA?
            var dsa: DSA?
            if geo.mixers[l] == .recurrent {
                let aLog = weights.tensor(["blk.{L}.kda_a_log.weight"], layer: l)
                kda = KDA(
                    q: try weights.matrix("blk.{L}.kda_q.weight", "blk.{L}.attn_q.weight", layer: l),
                    k: try weights.matrix("blk.{L}.kda_k.weight", "blk.{L}.attn_k.weight", layer: l),
                    v: try weights.matrix("blk.{L}.kda_v.weight", "blk.{L}.attn_v.weight", layer: l),
                    convQ: try weights.floatBuffer(
                        "blk.{L}.kda_q_conv.weight", "blk.{L}.ssm_conv1d_q.weight", layer: l, count: kdaHeads * kdaDim * 4),
                    convK: try weights.floatBuffer(
                        "blk.{L}.kda_k_conv.weight", "blk.{L}.ssm_conv1d_k.weight", layer: l, count: kdaHeads * kdaDim * 4),
                    convV: try weights.floatBuffer(
                        "blk.{L}.kda_v_conv.weight", "blk.{L}.ssm_conv1d_v.weight", layer: l, count: kdaHeads * kdaDim * 4),
                    fA: try weights.matrix("blk.{L}.kda_f_a.weight", "blk.{L}.ssm_f_a.weight", layer: l),
                    fB: try weights.matrix("blk.{L}.kda_f_b.weight", "blk.{L}.ssm_f_b.weight", layer: l),
                    beta: try weights.matrix("blk.{L}.kda_beta.weight", "blk.{L}.ssm_beta.weight", layer: l),
                    gA: try weights.matrix("blk.{L}.kda_g_a.weight", "blk.{L}.ssm_g_a.weight", layer: l),
                    gB: try weights.matrix("blk.{L}.kda_g_b.weight", "blk.{L}.ssm_g_b.weight", layer: l),
                    out: try weights.matrix("blk.{L}.kda_output.weight", "blk.{L}.attn_output.weight", layer: l),
                    dtBias: try weights.floatBuffer(
                        "blk.{L}.kda_dt_bias.weight", "blk.{L}.ssm_dt.bias", layer: l, count: kdaHeads * kdaDim),
                    aLog: try weights.floatBuffer("blk.{L}.kda_a_log.weight", "blk.{L}.ssm_a", layer: l, count: kdaHeads),
                    oNorm: try weights.floatBuffer("blk.{L}.kda_o_norm.weight", "blk.{L}.ssm_norm.weight", layer: l, count: kdaDim),
                    aIsLog: aLog != nil)
            } else {
                var indexer: Indexer?
                if weights.tensor(["blk.{L}.indexer.attn_q_b.weight"], layer: l) != nil {
                    indexer = Indexer(
                        qB: try weights.matrix("blk.{L}.indexer.attn_q_b.weight", layer: l),
                        k: try weights.matrix("blk.{L}.indexer.attn_k.weight", layer: l),
                        proj: try weights.matrix("blk.{L}.indexer.proj.weight", layer: l),
                        gate: try weights.matrix("blk.{L}.indexer.pool_gate.weight", "blk.{L}.indexer_compressor_gate.weight", layer: l),
                        kNormW: try weights.floatBuffer("blk.{L}.indexer.k_norm.weight", layer: l, count: indexDim),
                        kNormB: try weights.floatBuffer("blk.{L}.indexer.k_norm.bias", layer: l, count: indexDim),
                        ape: try weights.floatBuffer(
                            "blk.{L}.indexer.pool_ape.weight", "blk.{L}.indexer_compressor_ape.weight", layer: l, count: indexDim * pool))
                }
                dsa = DSA(
                    qA: try weights.matrix("blk.{L}.attn_q_a.weight", layer: l),
                    qB: try weights.matrix("blk.{L}.attn_q_b.weight", layer: l),
                    kvA: try weights.matrix("blk.{L}.attn_kv_a_mqa.weight", layer: l),
                    kB: try weights.matrix("blk.{L}.attn_k_b.weight", layer: l),
                    vB: try weights.matrix("blk.{L}.attn_v_b.weight", layer: l),
                    out: try weights.matrix("blk.{L}.attn_output.weight", layer: l),
                    qANorm: try weights.floatBuffer("blk.{L}.attn_q_a_norm.weight", layer: l, count: qRank),
                    kvANorm: try weights.floatBuffer("blk.{L}.attn_kv_a_norm.weight", layer: l, count: kvRank), indexer: indexer)
            }
            let moe = geo.moe[l]
            layers.append(
                Layer(
                    attnNorm: try weights.floatBuffer("blk.{L}.attn_norm.weight", layer: l, count: geo.hidden),
                    ffnNorm: try weights.floatBuffer("blk.{L}.ffn_norm.weight", layer: l, count: geo.hidden),
                    hcAttn: try hc("attn"), hcFFN: try hc("ffn"), kda: kda, dsa: dsa, dense: moe ? nil : try ffn(""),
                    router: moe ? try weights.matrix("blk.{L}.ffn_gate_inp.weight", layer: l) : nil,
                    routerBias: moe ? (try weights.optionalFloats("blk.{L}.exp_probs_b.bias", layer: l) ?? []) : [],
                    shared: moe ? try ffn("_shexp") : nil))
        }
        guard layers.allSatisfy({ [0, geo.experts].contains($0.routerBias.count) }) else {
            throw EngineError("This GLM-5.3-Flash GGUF's router bias does not hold one value per expert", code: "model_incompatible")
        }
        self.layers = layers
        try super.init(weights: weights, context: context)
        // The conversation state first, so the expert slots are sized from what is free after it; its per-position part in
        // memory or on disk, as `contextFile` decides from its size.
        let positionBytes = UInt64(positionCounts().reduce(0, +) * 4)
        contextFolder = contextFile?(positionBytes)
        lendsState = expertBudget == nil && contextFolder == nil
        try reset()
        stateBytes = UInt64(stateParts(at: 0).reduce(0) { $0 + $1.bytes }) + positionStateBytes(at: contextLimit)
        // A prompt row's working floats: two copies of the four streams, its embedding, FFN input and output, router
        // logits and the mixes.
        slicePrompts(rowBytes: 4 * (11 * hidden + geo.experts + 24))
        let free = try makeExperts(budget: expertBudget, memoryLimit: memoryLimit, ledger: ledger)
        log(
            "GLM-5.3-Flash: \(geo.layers) layers, \(OrigamiPlan.gigabytes(UInt64(weights.residentBytes))) resident "
                + "(\(OrigamiPlan.gigabytes(UInt64(weights.wiredBytes))) wired), \(experts.slotCount) expert slots of \(experts.slotSizes)"
                + " (\(OrigamiPlan.gigabytes(UInt64(experts.poolBytes)))); \(OrigamiPlan.gigabytes(free)) was free")
    }

    /// The per-position state buffers' sizes in floats, as `reset` allocates them for the context.
    private func positionCounts() -> [Int] {
        layers.compactMap(\.dsa).flatMap { dsa in
            [contextLimit * kvRank] + (dsa.indexer != nil ? [(contextLimit / pool + 1) * indexDim] : [])
        }
    }

    /// Clears the conversation (the per-position buffers are allocated once, for `stateCapacity` positions).
    package func reset() throws {
        resetPosition()
        for l in 0..<g.layers where g.mixers[l] == .recurrent {
            if kdaState[l] == nil {
                kdaState[l] = try zeros(kdaHeads * kdaDim * kdaDim)
                convState[l] = try zeros(3 * 3 * kdaHeads * kdaDim)
            } else {
                memset(kdaState[l]!.contents(), 0, kdaState[l]!.length)
                memset(convState[l]!.contents(), 0, convState[l]!.length)
            }
        }
        if latent.isEmpty { try allocatePositionState(stateCapacity) }
    }
    /// The per-position buffers for `capacity` positions (`regrown`; allocated once by `reset`).
    override package func positionStateBytes(at capacity: Int) -> UInt64 {
        guard contextFolder == nil else { return 0 }
        var bytes = 0
        for l in 0..<g.layers where g.mixers[l] != .recurrent {
            bytes += max(16, capacity * kvRank * 4)
            if layers[l].dsa?.indexer != nil {
                bytes += max(16, (capacity / pool + 1) * indexDim * 4)
            }
        }
        return UInt64(bytes)
    }
    override package func allocatePositionState(_ capacity: Int, grow: (MTLBuffer?, Int) throws -> MTLBuffer) throws {
        var nextLatent = latent
        var nextPooled = pooled
        var nextKey = indexKey
        var nextGate = indexGate
        for l in 0..<g.layers where g.mixers[l] != .recurrent {
            nextLatent[l] = try grow(latent[l], capacity * kvRank)
            if layers[l].dsa?.indexer != nil {
                if indexKey[l] == nil {
                    nextKey[l] = try zeros(pool * indexDim)
                    nextGate[l] = try zeros(pool * indexDim)
                }
                nextPooled[l] = try grow(pooled[l], (capacity / pool + 1) * indexDim)
            }
        }
        latent = nextLatent
        pooled = nextPooled
        indexKey = nextKey
        indexGate = nextGate
    }

    override package func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] {
        stateLock.lock()
        defer { stateLock.unlock() }
        return (0..<g.layers).flatMap { l -> [(buffer: MTLBuffer, bytes: Int)] in
            if g.mixers[l] == .recurrent { return [(kdaState[l]!, kdaState[l]!.length), (convState[l]!, convState[l]!.length)] }
            var parts: [(buffer: MTLBuffer, bytes: Int)] = [(latent[l]!, position * kvRank * 4)]
            if layers[l].dsa?.indexer != nil {
                // Carry rows are recurrent even at position zero, so tool-call anchors capture them too.
                parts += [
                    (indexKey[l]!, indexKey[l]!.length), (indexGate[l]!, indexGate[l]!.length),
                    (pooled[l]!, position / pool * indexDim * 4),
                ]
            }
            return parts
        }
    }
    package func restored(_ tokens: [Int]) { restorePosition(tokens.count) }

    // MARK: forward

    /// Runs `tokens` after the conversation so far and returns the logits after the last one.
    package func forward(_ tokens: [Int]) throws -> [Float] {
        guard !tokens.isEmpty else { throw EngineError("Nothing to run") }
        guard position + tokens.count <= contextLimit else {
            throw EngineError(
                "The conversation needs \(position + tokens.count) tokens; the context holds \(contextLimit)", code: "context_exceeded")
        }
        return try forwardPass(rows: tokens.count) { try run(tokens) }
    }

    private func run(_ tokens: [Int]) throws -> [Float] {
        let n = tokens.count
        let h = hidden
        // Two full-size copies of the streams: a half-layer reads one and writes the other.
        var x = try work("streams-a", n * 4 * h)
        var y = try work("streams-b", n * 4 * h)
        let embedded = try work("embedded", n * h)
        for (i, token) in tokens.enumerated() {
            guard (0..<embed.outputs).contains(token) else { throw EngineError("Token \(token) is outside the vocabulary") }
            FamilyWeights.row(embed, token, into: embedded.contents().bindMemory(to: Float.self, capacity: n * h) + i * h)
        }
        do {
            let (grid, group) = grid2(h, n)
            try kernels.dispatch(encoder, "hc_broadcast", [Ref(embedded), Ref(x)], ints: [Int32(h), Int32(n)], grid: grid, group: group)
        }
        for (l, layer) in layers.enumerated() {
            // A one-row decode step (a couple of seconds at most) finishes: cancelling it would throw the conversation away;
            // the request honours cancellation between tokens. A prompt's rows stop between layers.
            guard n == 1 || shouldContinue() else {
                try flush(wait: true)
                throw EngineError("Cancelled", code: "cancelled")
            }
            // Attention half, in chunks of rows.
            step = "layer \(l) \(layer.kda != nil ? "KDA" : "DSA") rows \(position)+\(n)"
            var start = 0
            while start < n {
                let rows = min(chunk, n - start)
                try mixerHalf(encoder, layer: l, input: x, output: y, first: start, rows: rows)
                start += rows
                try commitIfLong()
            }
            swap(&x, &y)
            if n == 1, routing.delayedExperts != nil { try injectDelayedExperts(at: l, streams: x, combine: try work("mix-comb", 16)) }
            // Feed-forward half: the mixes per chunk, the experts over all rows (each expert read once).
            step = "layer \(l) feed-forward rows \(position)+\(n)"
            let ffnIn = try work("ffn-in", n * h)
            let mixed = try work("ffn-out", n * h)
            let pre = try work("hc-pre", n * 4)
            let post = try work("hc-post", n * 4)
            let comb = try work("hc-comb", n * 16)
            try chunked(n) { first, rows in
                try hcPre(
                    encoder, layer.hcFFN, streams: Ref(x, first * 4 * h * 4), rows: rows, out: Ref(ffnIn, first * h * 4),
                    pre: Ref(pre, first * 16), post: Ref(post, first * 16), comb: Ref(comb, first * 64))
                try rms(encoder, Ref(ffnIn, first * h * 4), layer.ffnNorm, Ref(ffnIn, first * h * 4), n: h, rows: rows, eps: eps)
            }
            if let dense = layer.dense {
                try chunked(n) { first, rows in
                    try swigluFFN(dense, Ref(ffnIn, first * h * 4), Ref(mixed, first * h * 4), rows: rows, limit: swiglu)
                }
            } else {
                let after = layers.indices.contains(l + 1) ? layers[l + 1] : nil
                var guessIn: MTLBuffer?
                if n == 1, guessesNext, let after, after.router != nil {
                    // Decode guesses l + 1's experts from its own mix and norm of these streams, not from l's: the input its router sees, one attention early.
                    let guess = try work("guess-in", h)
                    try hcPre(
                        encoder, after.hcFFN, streams: Ref(x), rows: 1, out: Ref(guess),
                        pre: Ref(try work("guess-pre", 4)), post: Ref(try work("guess-post", 4)), comb: Ref(try work("guess-comb", 16)))
                    try rms(encoder, Ref(guess), after.ffnNorm, Ref(guess), n: h, rows: 1, eps: eps)
                    guessIn = guess
                }
                // A prompt does not read ahead into the last layer, which reads only its last row's experts.
                let next = n > 1 && l + 2 == layers.count ? nil : after.flatMap { a in a.router.map { (l + 1, $0, a.routerBias, guessIn) } }
                // The last layer's FFN feeds only the head's last row: a prompt reads only that row's experts there.
                try moe(
                    layer: l, router: layer.router!, bias: layer.routerBias, shared: layer.shared!, input: ffnIn, output: mixed, rows: n,
                    probability: { 1 / (1 + exp(-$0)) }, normalize: normTopK, scale: routedScale, sum: { max($0, 6.103515625e-5) },
                    limit: swiglu,
                    next: next, keepRows: n > 1 && after == nil ? n - 1..<n : nil,
                    delayedPost: post, mayDefer: n == 1 && after != nil)
            }
            let (grid, group) = grid2(h, n)
            try kernels.dispatch(
                encoder, "hc_expand", [Ref(mixed), Ref(x), Ref(post), Ref(comb), Ref(y)], ints: [Int32(h), Int32(n)], grid: grid,
                group: group)
            swap(&x, &y)
            if n > 1 { try flush(wait: false) }
        }
        // Head: last row only.
        step = "head"
        let normed = try work("head-in", h)
        let (grid, group) = grid2(h, 1)
        try kernels.dispatch(
            encoder, "hc_mean", [Ref(x, (n - 1) * 4 * h * 4), Ref(normed)], ints: [Int32(h), 1], grid: grid, group: group)
        try rms(encoder, Ref(normed), outputNorm, Ref(normed), n: h, rows: 1, eps: eps)
        return try head(normed, output: output)
    }

    /// pre/post/comb from the streams of `rows` rows and the collapsed input (sum_h pre_h stream_h).
    func hcPre(_ e: MTLComputeCommandEncoder, _ hc: HC, streams: Ref, rows: Int, out: Ref, pre: Ref, post: Ref, comb: Ref) throws {
        let h = hidden
        let flat = try work("hc-flat", rows * 4 * h)
        let mixes = try work("hc-mixes", rows * 24)
        try rms(e, streams, nil, Ref(flat), n: 4 * h, rows: rows, eps: eps)
        try mm(e, hc.fn, Ref(flat), Ref(mixes), rows: rows)
        try kernels.dispatch(
            e, "hc_prepare", [Ref(mixes), Ref(hc.scale), Ref(hc.base), pre, post, comb],
            ints: [Int32(rows), Int32(sinkhorn)], floats: [hcEps],
            grid: MTLSize(width: (rows * 16 + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        let (grid, group) = grid2(h, rows)
        try kernels.dispatch(e, "hc_collapse", [streams, pre, out], ints: [Int32(h), Int32(rows)], grid: grid, group: group)
    }

    /// The attention half of layer `l` for rows `first ..< first + rows` (positions `position + first …`).
    func mixerHalf(_ e: MTLComputeCommandEncoder, layer l: Int, input: MTLBuffer, output: MTLBuffer, first: Int, rows: Int) throws {
        let layer = layers[l]
        let h = hidden
        let streams = Ref(input, first * 4 * h * 4)
        let cur = try work("mix-in", rows * h)
        let out = try work("mix-out", rows * h)
        let pre = try work("mix-pre", rows * 4)
        let post = try work("mix-post", rows * 4)
        let comb = try work("mix-comb", rows * 16)
        try hcPre(e, layer.hcAttn, streams: streams, rows: rows, out: Ref(cur), pre: Ref(pre), post: Ref(post), comb: Ref(comb))
        try rms(e, Ref(cur), layer.attnNorm, Ref(cur), n: h, rows: rows, eps: eps)
        if let kda = layer.kda {
            try kdaMixer(e, kda, layer: l, input: cur, output: out, rows: rows)
        } else if let dsa = layer.dsa {
            try dsaMixer(e, dsa, layer: l, input: cur, output: out, first: first, rows: rows)
        }
        let (grid, group) = grid2(h, rows)
        // `encoder`: the attention may have committed and opened a new one.
        try kernels.dispatch(
            encoder, "hc_expand", [Ref(out), streams, Ref(post), Ref(comb), Ref(output, first * 4 * h * 4)],
            ints: [Int32(h), Int32(rows)], grid: grid, group: group)
    }

    func kdaMixer(_ e: MTLComputeCommandEncoder, _ k: KDA, layer l: Int, input: MTLBuffer, output: MTLBuffer, rows: Int) throws {
        // On `encoder`, committed between projections when long (a chunk of a real model's is more than one buffer should hold).
        let c = kdaHeads * kdaDim
        let raw = try work("kda-raw", rows * c)
        let q = try work("kda-q", rows * c)
        let kk = try work("kda-k", rows * c)
        let v = try work("kda-v", rows * c)
        let low = try work("kda-low", rows * kdaDim)
        let gate = try work("kda-g", rows * c)
        let beta = try work("kda-beta", rows * kdaHeads)
        let o = try work("kda-o", rows * c)
        let state = convState[l]!
        for (i, (proj, conv, dst)) in [(k.q, k.convQ, q), (k.k, k.convK, kk), (k.v, k.convV, v)].enumerated() {
            try mm(encoder, proj, Ref(input), Ref(raw), rows: rows)
            try kernels.dispatch(
                encoder, "kda_conv", [Ref(raw), Ref(state, i * 3 * c * 4), Ref(conv), Ref(dst)], ints: [Int32(c), Int32(rows), Int32(c)],
                grid: MTLSize(width: (c + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            try commitIfLong()
        }
        try mm(encoder, k.fA, Ref(input), Ref(low), rows: rows)
        try mm(encoder, k.fB, Ref(low), Ref(gate), rows: rows)
        let (grid, group) = grid2(c, rows)
        try kernels.dispatch(
            encoder, "kda_gate", [Ref(gate), Ref(k.dtBias), Ref(k.aLog)],
            ints: [Int32(kdaHeads), Int32(kdaDim), Int32(rows), k.aIsLog ? 1 : 0],
            floats: [lowerBound], grid: grid, group: group)
        try mm(encoder, k.beta, Ref(input), Ref(beta), rows: rows)
        try kernels.dispatch(
            encoder, "sigmoid_inplace", [Ref(beta)], ints: [Int32(rows * kdaHeads)],
            grid: MTLSize(width: (rows * kdaHeads + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        for buffer in [q, kk] {
            try kernels.dispatch(
                encoder, "l2_norm_heads", [Ref(buffer)], ints: [Int32(kdaDim)], floats: [1e-6],
                grid: MTLSize(width: rows * kdaHeads, height: 1, depth: 1), group: MTLSize(width: kdaDim, height: 1, depth: 1))
        }
        try kernels.dispatch(
            encoder, "kda_scan", [Ref(q), Ref(kk), Ref(v), Ref(gate), Ref(beta), Ref(kdaState[l]!), Ref(o)],
            ints: [Int32(kdaHeads), Int32(rows), Int32(c), Int32(c), Int32(c)], floats: [1 / Float(kdaDim).squareRoot()],
            grid: MTLSize(width: kdaHeads * 4, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        try mm(encoder, k.gA, Ref(input), Ref(low), rows: rows)
        try mm(encoder, k.gB, Ref(low), Ref(raw), rows: rows)
        try kernels.dispatch(
            encoder, "kda_out", [Ref(o), Ref(k.oNorm), Ref(raw), Ref(o)], ints: [Int32(kdaDim)], floats: [eps],
            grid: MTLSize(width: rows * kdaHeads, height: 1, depth: 1), group: MTLSize(width: kdaDim, height: 1, depth: 1))
        try commitIfLong()
        try mm(encoder, k.out, Ref(o), Ref(output), rows: rows)
    }

    func dsaMixer(_ e: MTLComputeCommandEncoder, _ d: DSA, layer l: Int, input: MTLBuffer, output: MTLBuffer, first: Int, rows: Int) throws
    {
        let start = position + first
        let qr = try work("dsa-qr", rows * qRank)
        let q = try work("dsa-q", rows * heads * qkDim)
        let qh = try work("dsa-qh", rows * heads * qkDim)
        let qabs = try work("dsa-qabs", heads * rows * kvRank)
        try mm(e, d.qA, Ref(input), Ref(qr), rows: rows)
        try rms(e, Ref(qr), d.qANorm, Ref(qr), n: qRank, rows: rows, eps: eps)
        try mm(e, d.qB, Ref(qr), Ref(q), rows: rows)
        // The new latent rows go straight into the cache.
        let cache = latent[l]!
        try mm(e, d.kvA, Ref(input), Ref(cache, start * kvRank * 4), rows: rows)
        try rms(e, Ref(cache, start * kvRank * 4), d.kvANorm, Ref(cache, start * kvRank * 4), n: kvRank, rows: rows, eps: eps)
        // Absorb wk_b per head: [rows][heads][qk] -> [heads][rows][qk], then each head's matrix (kvRank x qk).
        try permute(e, Ref(q), Ref(qh), a: rows, b: heads, n: qkDim)
        try mmBatch(
            e, (0..<heads).map { rowsOf(d.kB, first: $0 * kvRank, count: kvRank) }, Ref(qh), inputStride: rows * qkDim * 4,
            Ref(qabs), outputStride: rows * kvRank * 4, rows: rows)
        // Selection: the indexer's top pools plus the tail, or every earlier row while they fit.
        let selected = (indexTop / pool) * pool + (pool - 1)
        let sel = try work("dsa-sel", rows * selected)
        // indexSelect commits between its slices: the attention below reads `encoder`, not `e`.
        if let indexer = d.indexer {
            try indexSelect(e, indexer, layer: l, input: input, qr: qr, sel: sel, first: start, rows: rows, selected: selected)
        } else {
            // No indexer: the most recent `selected` rows (a sliding window over the cache).
            try kernels.dispatch(
                e, "window_select", [Ref(sel)], ints: [Int32(selected), Int32(start)],
                grid: MTLSize(width: (selected + 63) / 64, height: rows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        }
        let scores = try work("dsa-scores", rows * heads * selected)
        let lat = try work("dsa-lat", heads * rows * kvRank)
        // The attention itself (f32, a prompt's slowest kernels) in batches of 64 token rows, each dispatched in slices of
        // rows that fit the open buffer (`sliced`); a row's arithmetic is the same in any slice, and a batch's slices take
        // its kernels. From here on the open encoder is `encoder`, not `e`. qabs is [heads][rows][kv]: q of token t, head h
        // at (h * rows + t) * kv.
        var batch = 0
        while batch < rows {
            let n = min(64, rows - batch)
            // The heads-parallel kernels, bitwise the same as the row kernels: eight heads a thread where they divide (they keep
            // the GPU busy on the few rows of a slice), else one head a thread for a few rows (decode).
            let eights = heads % 8 == 0
            let few = n <= 8
            let variant = eights ? "_heads8" : few ? "_heads" : ""
            let depth = eights ? heads / 8 : few ? heads : 1
            try sliced(n, perRow: Self.attendWork(heads: heads, selected: selected, dim: kvRank)) { done, m in
                let t = batch + done
                let s = Ref(scores, t * heads * selected * 4)
                let picked = Ref(sel, t * selected * 4)
                try kernels.dispatch(
                    encoder, "attend_scores" + variant, [Ref(qabs, t * kvRank * 4), Ref(cache), Ref(cache), picked, s],
                    ints: [Int32(heads), Int32(kvRank), Int32(selected), Int32(kvRank), Int32(rows * kvRank), 0, 0, Int32.max],
                    floats: [1 / Float(qkDim).squareRoot()],
                    grid: MTLSize(width: (selected + 63) / 64, height: m, depth: depth), group: MTLSize(width: 64, height: 1, depth: 1))
                try kernels.dispatch(
                    encoder, "softmax_rows", [s, s], ints: [Int32(selected), 0, 1],
                    grid: MTLSize(width: m * heads, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
                try kernels.dispatch(
                    encoder, "attend_values" + variant, [s, Ref(cache), Ref(cache), picked, Ref(lat, t * kvRank * 4)],
                    ints: [Int32(heads), Int32(kvRank), Int32(selected), 0, 0, Int32(kvRank), Int32(rows * kvRank), Int32.max],
                    floats: [0],
                    grid: MTLSize(width: ((eights ? kvRank / 4 : kvRank) + 63) / 64, height: m, depth: depth),
                    group: MTLSize(width: 64, height: 1, depth: 1))
            }
            batch += n
        }
        // wv_b per head: latent -> v (vDim), into [heads][rows][v], then back to [rows][heads][v] for the output projection.
        let vh = try work("dsa-vh", heads * rows * vDim)
        let v = try work("dsa-v", rows * heads * vDim)
        try mmBatch(
            encoder, (0..<heads).map { rowsOf(d.vB, first: $0 * vDim, count: vDim) }, Ref(lat), inputStride: rows * kvRank * 4,
            Ref(vh), outputStride: rows * vDim * 4, rows: rows)
        try permute(encoder, Ref(vh), Ref(v), a: heads, b: rows, n: vDim)
        try mm(encoder, d.out, Ref(v), Ref(output), rows: rows)
    }

    /// Writes the indexer's key, gate and the pools these rows complete, scores the visible pools and selects rows.
    func indexSelect(
        _ e: MTLComputeCommandEncoder, _ x: Indexer, layer l: Int, input: MTLBuffer, qr: MTLBuffer, sel: MTLBuffer,
        first: Int, rows: Int, selected: Int
    ) throws {
        let pools = pooled[l]!
        try poolRows(
            e, key: indexKey[l]!, gate: indexGate[l]!, ape: x.ape, output: pools,
            first: first, rows: rows, dim: indexDim, pool: pool
        ) { keys, gates in
            try mm(e, x.k, Ref(input), keys, rows: rows)
            try kernels.dispatch(
                e, "layer_norm", [keys, Ref(x.kNormW), Ref(x.kNormB), keys],
                ints: [Int32(indexDim)], floats: [1e-6], grid: MTLSize(width: rows, height: 1, depth: 1),
                group: MTLSize(width: indexDim, height: 1, depth: 1))
            try mm(e, x.gate, Ref(input), gates, rows: rows)
        }
        let lastPool = (first + rows) / pool
        let iq = try work("idx-q", rows * indexHeads * indexDim)
        let w = try work("idx-w", rows * indexHeads)
        try mm(e, x.qB, Ref(qr), Ref(iq), rows: rows)
        try mm(e, x.proj, Ref(input), Ref(w), rows: rows)
        let scale = 1 / Float(indexDim * indexHeads).squareRoot()
        try kernels.dispatch(
            e, "scale_inplace", [Ref(w)], ints: [Int32(rows * indexHeads)], floats: [scale],
            grid: MTLSize(width: (rows * indexHeads + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        let poolCount = max(1, lastPool)
        let scores = try work("idx-scores", rows * poolCount)
        // Scoring and selection grow with the context: in slices of rows, each its own share of a buffer (the caller
        // reads `encoder` afresh after this). Row t of a slice is at position at + t, the same arithmetic per row.
        try sliced(rows, perRow: Self.indexWork(entries: poolCount, heads: indexHeads, dim: indexDim)) { done, n in
            let at = first + done
            let s = Ref(scores, done * poolCount * 4)
            try kernels.dispatch(
                encoder, "index_scores", [Ref(iq, done * indexHeads * indexDim * 4), Ref(pools), Ref(w, done * indexHeads * 4), s],
                ints: [Int32(indexHeads), Int32(indexDim), Int32(poolCount), Int32(pool), Int32(at)],
                grid: MTLSize(width: (poolCount + 63) / 64, height: n, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
            try kernels.dispatch(
                encoder, "index_select", [s, Ref(sel, done * selected * 4)],
                ints: [
                    Int32(poolCount), Int32(pool), Int32(indexTop / pool), Int32(selected), Int32(at), 1, Int32(pool), 0, Int32(selected),
                ],
                grid: MTLSize(width: n, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
        }
    }
}
