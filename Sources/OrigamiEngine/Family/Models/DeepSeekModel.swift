import Foundation
import Metal

/// DeepSeek-V4.1-Flash (`deepseek41`) on Metal from a GGUF, following DeepSeek's inference/model.py and llama.cpp's
/// deepseek41 graph (MIT): four hyper-connection streams whose collapse lags one sublayer (attention collapses with the
/// previous FFN's pre-mix, the FFN with this attention's), latent attention with one shared 512-wide key/value row
/// over a 128-token sliding window plus, on compressed layers, the indexer's top compressed rows (pooled by
/// `compress_ratio` with a softmax gate, shared by the layers after each source), attention sinks, RoPE on the last 64
/// values taken off the output again, a grouped low-rank output, Engram n-gram rows added to the streams at its layers,
/// and sqrt-softplus routing over the routed experts plus a shared one. Prompts run layer by layer over all their
/// tokens; decode is the same code with one row. A long prompt's early rows run only up to the last key/value source
/// (`promptTail`, `feed`).
package final class DeepSeekModel: FamilyRunner, @unchecked Sendable {
    struct HC {
        let fn: FamilyWeights.Matrix
        let scale, base: MTLBuffer
    }
    struct Compressor {
        let kv: FamilyWeights.Matrix
        let gate: FamilyWeights.Matrix?
        let norm: MTLBuffer
        let indexK: FamilyWeights.Matrix
        let indexNorm: MTLBuffer
    }
    struct Indexer { let qB, proj: FamilyWeights.Matrix }
    struct Engram {
        let kv: FamilyWeights.Matrix
        let weight: MTLBuffer
        let index: Int
    }
    struct Layer {
        let attnNorm, ffnNorm, qANorm, kvNorm, sinks: MTLBuffer
        let qA, qB, kv, outA, outB: FamilyWeights.Matrix
        let hcAttn, hcFFN: HC
        let compressor: Compressor?
        let indexer: Indexer?
        let engram: Engram?
        let router: FamilyWeights.Matrix
        let routerBias: [Float]
        let shared: FFN
        let ratio: Int
        let freqs: MTLBuffer
    }

    let layers: [Layer]
    let embed, output: FamilyWeights.Matrix
    let outputNorm: MTLBuffer
    let eps: Float, hcEps: Float, sinkhorn: Int, swiglu: Float, routedScale: Float, normTopK: Bool
    let heads: Int, headDim: Int, rope: Int, qRank: Int, groups: Int, oRank: Int, window: Int
    let indexHeads: Int, indexDim: Int, indexTop: Int, candidateSource: Int, candidateBlocks: Int, candidateBlock: Int
    /// Per layer: the latest compressing (key/value source) and indexing layer at or before it.
    let kvSource: [Int], indexSource: [Int]
    /// The window ring holds the window plus one chunk, so a chunk never overwrites a row its own rows still read.
    let ring: Int
    /// Engram: hashing constants, the token map, and each engram layer's table and its file (one table for every
    /// layer, or one each).
    let engramLayers: [Int], ngram: Int, engramHeads: Int, engramDim: Int
    let multipliers: [UInt64], offsets: [Int64], primes: [UInt64], tokenMap: [Int32], pad: Int64
    let tables: [(tensor: GGUFSource.Tensor, file: ExactFile)]

    // Conversation state.
    private var history: [Int] = []
    /// The compressor's pooling has no per-position bias (the k-pool kernel's `ape`), made on first use.
    private var zeroBias: MTLBuffer?
    private func noPositionBias() throws -> MTLBuffer {
        if let zeroBias { return zeroBias }
        let buffer = try zeros(headDim * 8)
        zeroBias = buffer
        return buffer
    }
    private var ringKV: [Int: MTLBuffer] = [:], latent: [Int: MTLBuffer] = [:], indexKeys: [Int: MTLBuffer] = [:]
    /// Only the unfinished compression group; completed raw rows are never read again.
    private var compKV: [Int: MTLBuffer] = [:], compScore: [Int: MTLBuffer] = [:]
    /// The current forward's rows of the latest index source's top-k (offset by the ring size) and of the candidate
    /// source's kept blocks: written by the source over every chunk, read by the layers after it.
    private var topkRows: MTLBuffer!, candidateKeep: MTLBuffer!
    /// Bounded research caller only. Normal serving keeps the last-row head/FFN pruning.
    private var returnsAllLogits = false
    private var candidateStride: Int { (contextLimit + candidateBlock - 1) / candidateBlock }

    override func releasePromptReferences() {
        topkRows = nil
        candidateKeep = nil
    }

    override package func scratchPlan(rows n: Int) -> [String: Int] {
        let c = min(n, chunk)
        let h = hidden
        let selected = window + (layers.contains { $0.ratio > 0 } ? indexTop : 0)
        var plan = moeScratchPlan(rows: n, sharedWidth: layers.map { $0.shared.gate.outputs }.max() ?? 0)
        let counts = [
            "streams-a": n * 4 * h, "streams-b": n * 4 * h, "embedded": n * h,
            "ffn-in": n * h, "ffn-out": n * h, "pre-ffn": n * 4, "pre-attn": n * 4,
            "hc-post": n * 4, "hc-comb": n * 16, "hc-flat": c * 4 * h, "hc-mixes": c * 24,
            "attn-in": c * h, "attn-out": c * h,
            "head-in": (returnsAllLogits ? n : 1) * h,
            "logits": (returnsAllLogits ? n : 1) * output.outputs,
            "ds-topk": n * indexTop, "ds-candidates": candidateSource >= 0 ? n * candidateStride : 1,
            "ds-kv": c * headDim, "ds-qr": c * qRank, "ds-q": c * heads * headDim,
            "ds-sel": c * selected, "ds-scores": c * heads * selected,
            "ds-o": c * heads * headDim, "ds-og": c * heads * headDim,
            "ds-low": c * groups * oRank, "ds-low-rows": c * groups * oRank,
        ]
        plan.merge(counts) { max($0, $1) }
        if let ratio = layers.filter({ $0.indexer != nil }).map(\.ratio).min() {
            plan["ds-iq"] = c * indexHeads * indexDim
            plan["ds-iw"] = c * indexHeads
            plan["ds-iscores"] = c * max(1, scratchPositionLimit(rows: n) / ratio)
        }
        if n > 1, layers.contains(where: { $0.compressor != nil && $0.ratio > 1 }) {
            plan["pool-key"] = (c + 1) * headDim
            plan["pool-gate"] = (c + 1) * headDim
        }
        if !engramLayers.isEmpty {
            for e in engramLayers.indices { plan["engram-rows-\(e)"] = n * (ngram - 1) * engramHeads * engramDim }
            plan["engram-kv"] = c * 5 * h
        }
        if n == 1, guessesNext {
            plan["guess-in"] = h
            plan["guess-pre"] = 4
            plan["guess-post"] = 4
            plan["guess-comb"] = 16
        }
        return plan
    }

    package init(
        weights: FamilyWeights, expertBudget: UInt64?, memoryLimit: UInt64? = nil, context: Int,
        contextFile: ((UInt64) -> URL?)? = nil, ledger: MemoryLedger? = nil, log: (String) -> Void = { _ in }
    ) throws {
        try weights.beginStartup(ledger: ledger)
        defer { weights.endStartup() }
        let geo = weights.geometry
        guard geo.family == .deepseek, geo.hyperConnections == 4 else {
            throw EngineError("DeepSeekModel runs DeepSeek-V4.1-Flash with four hyper-connection streams", code: "model_incompatible")
        }
        func value(_ key: String, _ fallback: Double) -> Double { geo.values[key] ?? fallback }
        eps = Float(value("rms_norm_eps", 1e-20))
        hcEps = Float(value("hc_eps", 1e-6))
        sinkhorn = try geo.int("hc_sinkhorn_iters", 20, 0...1000)
        swiglu = Float(value("swiglu_limit", 0))
        routedScale = Float(value("routed_scaling_factor", 1))
        normTopK = value("norm_topk_prob", 1) != 0 && geo.expertsPerToken > 1
        let maxCount = ModelGeometry.countLimit
        heads = try geo.int("num_attention_heads", 64, 1...64)
        headDim = try geo.int("head_dim", 512, 4...4096)
        rope = try geo.int("qk_rope_head_dim", 64, 0...4096)
        qRank = try geo.int("q_lora_rank", 1280, 1...maxCount)
        groups = try geo.int("o_groups", 8, 1...64)
        oRank = try geo.int("o_lora_rank", 1024, 1...maxCount)
        window = try geo.int("sliding_window", 128, 1...maxCount)
        indexHeads = try geo.int("index_n_heads", 32, 1...1024)
        indexDim = try geo.int("index_head_dim", 128, 4...4096)
        indexTop = try geo.int("index_topk", 512, 1...maxCount)
        candidateSource = try geo.int("candidate_source_layer_id", -1, -1...(geo.layers - 1))
        candidateBlocks = try geo.int("candidate_topk_blocks", 2048, 1...maxCount)
        candidateBlock = try geo.int("candidate_block_size", 8, 1...maxCount)
        let ratios = geo.lists["compress_ratios"] ?? Array(repeating: 0, count: geo.layers)
        let kvSources = Set(geo.lists["kv_source_layer_ids"] ?? [])
        let indexSources = Set(geo.lists["index_source_layer_ids"] ?? [])
        guard heads % groups == 0, headDim % 4 == 0, rope % 2 == 0, rope <= headDim, rope <= indexDim, indexDim % 4 == 0,
            ratios.count >= geo.layers,
            ratios.prefix(geo.layers).allSatisfy({ (0...2).contains($0) }),
            kvSources.union(indexSources).allSatisfy({ (0..<geo.layers).contains($0) && ratios[$0] > 0 })
        else {
            throw EngineError(
                "This DeepSeek-V4.1-Flash geometry has no kernels: \(heads) heads of \(headDim), rotary \(rope), compress ratios \(ratios)",
                code: "model_incompatible")
        }
        var lastKV = -1
        var lastIndex = -1
        var kvOf: [Int] = []
        var indexOf: [Int] = []
        for l in 0..<geo.layers {
            if kvSources.contains(l) { lastKV = l }
            if indexSources.contains(l) { lastIndex = l }
            kvOf.append(lastKV)
            indexOf.append(lastIndex)
            if ratios[l] > 0, lastKV < 0 || lastIndex < 0 || ratios[lastKV] != ratios[l] {
                throw EngineError(
                    "DeepSeek-V4.1-Flash layer \(l) compresses (ratio \(ratios[l])) without a matching source layer",
                    code: "model_incompatible")
            }
        }
        kvSource = kvOf
        indexSource = indexOf
        ring = window + 256  // FamilyRunner.chunk: a chunk never overwrites a row it still reads

        // RoPE frequencies of the last `rope` values: plain base on window-only layers, YaRN on compressed ones.
        let ropeDims = rope
        func freqs(compressed: Bool) -> [Float] {
            let base = compressed ? value("compress_rope_theta", 160000) : value("rope_theta", 10000)
            let dim = Double(ropeDims)
            var f = (0..<(ropeDims / 2)).map { 1 / pow(base, Double(2 * $0) / dim) }
            let original = value("rope_scaling.original_max_position_embeddings", 0)
            if compressed && original > 0 {
                let factor = value("rope_scaling.factor", 1)
                let fast = value("rope_scaling.beta_fast", 32)
                let slow = value("rope_scaling.beta_slow", 1)
                func corrected(_ rotations: Double) -> Double {
                    dim * Foundation.log(original / (rotations * 2 * .pi)) / (2 * Foundation.log(base))
                }
                let low = max(floor(corrected(fast)), 0)
                let high = min(ceil(corrected(slow)), dim - 1)
                for i in f.indices {
                    let ramp = min(max((Double(i) - low) / max(high - low, 1e-3), 0), 1)
                    let smooth = 1 - ramp
                    f[i] = f[i] / factor * (1 - smooth) + f[i] * smooth
                }
            }
            return f.map { Float($0) }
        }
        let plainFreqs = try weights.buffer(freqs(compressed: false))
        let compressedFreqs = try weights.buffer(freqs(compressed: true))

        // Engram: its layers, hashing constants and the table (rows are read from the file as tokens need them).
        engramLayers = geo.lists["engram_layer_ids"] ?? []
        ngram = try geo.int("engram_max_ngram_size", 4, 2...16)
        engramHeads = try geo.int("engram_n_heads", 8, 1...1024)
        engramDim = try geo.int("engram_head_dim", 256, 1...maxCount)
        if !engramLayers.isEmpty {
            // Unsloth's keys (ple.*) or vcruz305's and DwarfStar's (engram.*).
            let meta = try GGUFSource(
                weights.gguf.folder.appendingPathComponent(weights.gguf.files[0]),
                arrays: [
                    "engram.token_map", "ple.layer_multipliers",
                    "engram.multipliers", "ple.head_offsets", "engram.offsets", "ple.head_vocab_sizes", "engram.primes",
                ])
            func array(_ keys: String...) -> [Int64] {
                (keys.lazy.compactMap { meta.arrays["deepseek41." + $0] }.first ?? []).compactMap(\.integer)
            }
            multipliers = array("ple.layer_multipliers", "engram.multipliers").map { UInt64(bitPattern: $0) }
            primes = array("ple.head_vocab_sizes", "engram.primes").map { UInt64(bitPattern: $0) }
            tokenMap = array("engram.token_map").map { Int32(truncatingIfNeeded: $0) }
            let cols = (ngram - 1) * engramHeads
            // One table for every engram layer (Unsloth's per_layer_token_embd, offsets running on across layers) or
            // one per layer (blk.N.engram_embd, offsets from 0 in each); DwarfStar writes no offsets, its buckets lie
            // back to back.
            let shared = weights.tensor(["per_layer_token_embd.weight"])
            var offsets = array("ple.head_offsets", "engram.offsets")
            var sum: Int64 = 0
            if offsets.isEmpty {
                for (k, prime) in primes.enumerated() {
                    if shared == nil && k % cols == 0 { sum = 0 }
                    offsets.append(sum)
                    sum &+= Int64(truncatingIfNeeded: prime)
                }
            }
            self.offsets = offsets
            guard multipliers.count == engramLayers.count * ngram, offsets.count == engramLayers.count * cols,
                primes.count == offsets.count,
                !tokenMap.isEmpty, primes.allSatisfy({ (2..<(1 << 31)).contains($0) }), offsets.allSatisfy({ (0..<(1 << 40)).contains($0) })
            else {
                throw EngineError(
                    "This DeepSeek-V4.1-Flash GGUF's Engram metadata is inconsistent (multipliers \(multipliers.count), offsets \(offsets.count), "
                        + "primes \(primes.count), token map \(tokenMap.count))", code: "model_incompatible")
            }
            let padId = try geo.int("engram_pad_token_id", 2, 0...maxCount)
            pad = Int64(tokenMap[min(padId, tokenMap.count - 1)])
            var files: [String: ExactFile] = [:]
            var tables: [(tensor: GGUFSource.Tensor, file: ExactFile)] = []
            for l in engramLayers {
                guard let t = shared ?? weights.tensor(["blk.{L}.engram_embd.weight"], layer: l),
                    Self.engramRowBytes(t, dim: engramDim) != nil
                else {
                    throw EngineError(
                        "This DeepSeek-V4.1-Flash GGUF has no readable Engram table for layer \(l) (per_layer_token_embd.weight or "
                            + "blk.\(l).engram_embd.weight)", code: "model_incompatible")
                }
                if files[t.file] == nil {
                    files[t.file] = try ExactFile(
                        url: weights.gguf.folder.appendingPathComponent(t.file), bypassCache: false, drive: weights.drive)
                }
                tables.append((t, files[t.file]!))
            }
            self.tables = tables
        } else {
            multipliers = []
            offsets = []
            primes = []
            tokenMap = []
            pad = 0
            tables = []
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
            var compressor: Compressor?
            var indexer: Indexer?
            var engram: Engram?
            if kvSources.contains(l) {
                compressor = Compressor(
                    kv: try weights.matrix("blk.{L}.attn_compressor_kv.weight", layer: l),
                    gate: ratios[l] > 1 ? try weights.matrix("blk.{L}.attn_compressor_gate.weight", layer: l) : nil,
                    norm: try weights.floatBuffer("blk.{L}.attn_compressor_norm.weight", layer: l, count: headDim),
                    indexK: try weights.matrix("blk.{L}.indexer.attn_k.weight", layer: l),
                    indexNorm: try weights.floatBuffer("blk.{L}.indexer.k_norm.weight", layer: l, count: indexDim))
            }
            if indexSources.contains(l) {
                indexer = Indexer(
                    qB: try weights.matrix("blk.{L}.indexer.attn_q_b.weight", layer: l),
                    proj: try weights.matrix("blk.{L}.indexer.proj.weight", layer: l))
            }
            if let e = engramLayers.firstIndex(of: l) {
                // The gate weight is q_weight * k_weight: one tensor (Unsloth) or its two factors (vcruz305's engram_q and
                // engram_k, DwarfStar's engram_q_norm and engram_k_norm).
                let weight: MTLBuffer
                if weights.tensor(["blk.{L}.engram_gate.weight"], layer: l) != nil {
                    weight = try weights.floatBuffer("blk.{L}.engram_gate.weight", layer: l, count: 4 * geo.hidden)
                } else {
                    let q = try weights.floats("blk.{L}.engram_q.weight", "blk.{L}.engram_q_norm.weight", layer: l)
                    let k = try weights.floats("blk.{L}.engram_k.weight", "blk.{L}.engram_k_norm.weight", layer: l)
                    guard q.count == k.count, q.count >= 4 * geo.hidden else {
                        throw EngineError(
                            "Layer \(l)'s Engram gate factors hold \(q.count) and \(k.count) values; this geometry reads \(4 * geo.hidden)",
                            code: "model_incompatible")
                    }
                    weight = try weights.buffer(zip(q, k).map(*))
                }
                engram = Engram(
                    kv: try weights.matrix("blk.{L}.engram_kv.weight", "blk.{L}.engram_wkv.weight", layer: l), weight: weight, index: e)
            }
            layers.append(
                Layer(
                    attnNorm: try weights.floatBuffer("blk.{L}.attn_norm.weight", layer: l, count: geo.hidden),
                    ffnNorm: try weights.floatBuffer("blk.{L}.ffn_norm.weight", layer: l, count: geo.hidden),
                    qANorm: try weights.floatBuffer("blk.{L}.attn_q_a_norm.weight", layer: l, count: qRank),
                    kvNorm: try weights.floatBuffer(
                        "blk.{L}.attn_kv_a_norm.weight", "blk.{L}.attn_kv_norm.weight", layer: l, count: headDim),
                    sinks: try weights.floatBuffer("blk.{L}.attn_sinks.weight", layer: l, count: heads),
                    qA: try weights.matrix("blk.{L}.attn_q_a.weight", layer: l),
                    qB: try weights.matrix("blk.{L}.attn_q_b.weight", layer: l),
                    kv: try weights.matrix("blk.{L}.attn_kv.weight", layer: l),
                    outA: try weights.matrix("blk.{L}.attn_output_a.weight", layer: l),
                    outB: try weights.matrix("blk.{L}.attn_output_b.weight", layer: l), hcAttn: try hc("attn"), hcFFN: try hc("ffn"),
                    compressor: compressor, indexer: indexer, engram: engram,
                    router: try weights.matrix("blk.{L}.ffn_gate_inp.weight", layer: l),
                    routerBias: try weights.optionalFloats("blk.{L}.exp_probs_b.bias", layer: l) ?? [],
                    shared: FFN(
                        gate: try weights.matrix("blk.{L}.ffn_gate_shexp.weight", layer: l),
                        up: try weights.matrix("blk.{L}.ffn_up_shexp.weight", layer: l),
                        down: try weights.matrix("blk.{L}.ffn_down_shexp.weight", layer: l)),
                    ratio: ratios[l], freqs: ratios[l] > 0 ? compressedFreqs : plainFreqs))
        }
        guard layers.allSatisfy({ [0, geo.experts].contains($0.routerBias.count) }) else {
            throw EngineError("This DeepSeek-V4.1-Flash GGUF's router bias does not hold one value per expert", code: "model_incompatible")
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
        // A prompt row's working floats: two copies of the four streams, its embedding, FFN input and output, Engram rows,
        // top-k rows and candidate blocks, router logits and the mixes.
        slicePrompts(
            rowBytes: 4
                * (11 * hidden + engramLayers.count * (ngram - 1) * engramHeads * engramDim + indexTop
                    + (candidateSource >= 0 ? candidateStride : 0) + geo.experts + 28))
        let free = try makeExperts(budget: expertBudget, memoryLimit: memoryLimit, ledger: ledger)
        log(
            "DeepSeek-V4.1-Flash: \(geo.layers) layers, \(OrigamiPlan.gigabytes(UInt64(weights.residentBytes))) resident "
                + "(\(OrigamiPlan.gigabytes(UInt64(weights.wiredBytes))) wired), \(experts.slotCount) expert slots of \(experts.slotSizes)"
                + " (\(OrigamiPlan.gigabytes(UInt64(experts.poolBytes)))); \(OrigamiPlan.gigabytes(free)) was free")
    }

    /// The per-position state buffers' sizes in floats, as `reset` allocates them for the context (the window ring
    /// stays in memory).
    private func positionCounts() -> [Int] {
        layers.flatMap { layer -> [Int] in
            guard layer.compressor != nil else { return [] }
            let entries = contextLimit / layer.ratio + 1
            return [entries * headDim, entries * indexDim]
        }
    }

    /// Clears the conversation.
    package func reset() throws {
        resetPosition()
        history = []
        for l in layers.indices where ringKV[l] == nil { ringKV[l] = try zeros(ring * headDim) }
        if latent.isEmpty { try allocatePositionState(stateCapacity) }
        // The running buffers saved whole start as a cold start's: an aborted read's bytes would stay in slots not written yet.
        for buffer in [ringKV, compKV, compScore].flatMap(\.values) { memset(buffer.contents(), 0, buffer.length) }
    }
    /// The per-position buffers for `capacity` positions (`regrown`; allocated once by `reset`).
    override package func positionStateBytes(at capacity: Int) -> UInt64 {
        guard contextFolder == nil else { return 0 }
        return UInt64(
            layers.reduce(0) { bytes, layer in
                guard layer.compressor != nil else { return bytes }
                let entries = capacity / layer.ratio + 1
                return bytes + max(16, entries * headDim * 4) + max(16, entries * indexDim * 4)
            })
    }
    override package func allocatePositionState(_ capacity: Int, grow: (MTLBuffer?, Int) throws -> MTLBuffer) throws {
        var nextLatent = latent
        var nextIndex = indexKeys
        var nextKV = compKV
        var nextScore = compScore
        for (l, layer) in layers.enumerated() {
            guard let c = layer.compressor else { continue }
            let entries = capacity / layer.ratio + 1
            nextLatent[l] = try grow(latent[l], entries * headDim)
            nextIndex[l] = try grow(indexKeys[l], entries * indexDim)
            if c.gate != nil, compKV[l] == nil {
                nextKV[l] = try zeros(layer.ratio * headDim)
                nextScore[l] = try zeros(layer.ratio * headDim)
            }
        }
        latent = nextLatent
        indexKeys = nextIndex
        compKV = nextKV
        compScore = nextScore
    }

    override package func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] {
        stateLock.lock()
        defer { stateLock.unlock() }
        return layers.enumerated().flatMap { l, layer -> [(buffer: MTLBuffer, bytes: Int)] in
            var parts: [(buffer: MTLBuffer, bytes: Int)] = [(ringKV[l]!, ringKV[l]!.length)]
            if let c = layer.compressor {
                // Completed groups are append-only; the unfinished raw group is recurrent state and must also be
                // captured at position zero by tool-call anchors before a later token overwrites it.
                parts += [(latent[l]!, position / layer.ratio * headDim * 4), (indexKeys[l]!, position / layer.ratio * indexDim * 4)]
                if c.gate != nil { parts += [(compKV[l]!, compKV[l]!.length), (compScore[l]!, compScore[l]!.length)] }
            }
            return parts
        }
    }
    /// The token history feeds the Engram look-back.
    package func restored(_ tokens: [Int]) {
        restorePosition(tokens.count)
        history = tokens
    }

    /// Rows before a kept point that need every layer. The layers after the last key/value source read earlier rows
    /// only through its compressed rows and their own windows: the last row needs the last layer's window of inputs,
    /// each such layer's inputs the window of the layer before (window - 1 rows further back) down to the source's
    /// output, while the decode rings need one row less. So rows further back need only the layers up to that source,
    /// which keeps its own window and compressed rows for them (`feed`).
    package static func promptTail(layers: Int, lastSource: Int, window: Int) -> Int { 1 + (layers - 1 - lastSource) * (window - 1) }
    package var promptTail: Int? { Self.promptTail(layers: layers.count, lastSource: kvSource.last ?? -1, window: window) }

    /// Runs `tokens` after the conversation so far and returns the logits after the last one.
    package func forward(_ tokens: [Int]) throws -> [Float] { try forward(tokens, sourcesOnly: false) }
    /// Scores every row of a small known-token block. This advances all state, just like `forward`; callers must
    /// restore a snapshot before rejecting tokens. It is not speculative generation or a claim of sequential
    /// bit parity: multirow projections and expert grouping can change floating-point reductions.
    /// `generation` preserves decode workspace/cache treatment for speculative verification. It is explicit
    /// because the same small-block primitive also serves prompt/capture experiments.
    package func forwardBlock(_ tokens: [Int], generation: Bool = false) throws -> [[Float]] {
        guard (1...8).contains(tokens.count) else { throw EngineError("Block scoring accepts 1...8 tokens", code: "invalid_configuration") }
        returnsAllLogits = true
        defer { returnsAllLogits = false }
        let flat = try forward(tokens, sourcesOnly: false, generation: generation)
        return (0..<tokens.count).map { Array(flat[($0 * output.outputs)..<(($0 + 1) * output.outputs)]) }
    }

    /// Runs `tokens` through the layers before the last key/value source and that source's window and compressed rows
    /// only; the rows must lie more than `promptTail` before the next kept point.
    package func feed(_ tokens: [Int]) throws { _ = try forward(tokens, sourcesOnly: true) }
    /// `feed` of the first `fed` rows and `forward` of the rest in one pass: the fed rows stop at the last key/value
    /// source as `feed` stops them, so each layer's experts are read once for both. Every row keeps its chunks and
    /// slices (`fed` is whole slices), so its bits are those of the two calls.
    package func forward(_ tokens: [Int], fed: Int) throws -> [Float] {
        guard fed % FamilyRunner.promptSlice == 0, fed < tokens.count else {
            throw EngineError("Fed rows must be whole slices before the rest")
        }
        return try forward(tokens, sourcesOnly: false, fed: fed)
    }
    private func forward(_ tokens: [Int], sourcesOnly: Bool, fed: Int = 0, generation: Bool = false) throws -> [Float] {
        guard !tokens.isEmpty else { throw EngineError("Nothing to run") }
        guard position + tokens.count <= contextLimit else {
            throw EngineError(
                "The conversation needs \(position + tokens.count) tokens; the context holds \(contextLimit)", code: "context_exceeded")
        }
        let logits = try forwardPass(rows: tokens.count, generation: generation) { try run(tokens, sourcesOnly: sourcesOnly, fed: fed) }
        history += tokens
        return logits
    }

    /// The Engram rows of `tokens` for engram layer `e`: (ngram - 1) * heads row ids each (DeepSeek's NgramHashState
    /// through the compressed token map; look-back stops at the start), read from the table and decoded, `cols * dim`
    /// floats a token.
    func engramRows(_ tokens: [Int], layer e: Int, into out: UnsafeMutablePointer<Float>) throws {
        guard e < tables.count else { return }
        let (table, tableFile) = tables[e]
        try Self.loadEngramRows(engramIDs(tokens, layer: e), table: table, file: tableFile, dim: engramDim, into: out)
    }

    private func engramIDs(_ tokens: [Int], layer: Int) -> [Int64] {
        DeepSeekEngramHash.rows(
            tokens: tokens, history: history, layer: layer, ngram: ngram, heads: engramHeads,
            multipliers: multipliers, primes: primes, offsets: offsets, tokenMap: tokenMap, pad: pad)
    }

    package static func loadEngramRows(
        _ rowIDs: [Int64], table: GGUFSource.Tensor, file: ExactFile, dim: Int,
        into out: UnsafeMutablePointer<Float>
    ) throws {
        let rowBytes = engramRowBytes(table, dim: dim)!
        let rows = table.shape.first ?? 0
        final class Failure: @unchecked Sendable {
            var error: Error?
            let lock = NSLock()
        }
        let failure = Failure()
        nonisolated(unsafe) let out = out  // each row writes its own part
        // Bounded workers reuse their row buffer. A long prompt no longer schedules one closure/allocation per
        // hash lookup, and cannot occupy every CPU worker while expert demand reads are runnable.
        // Small random reads: the drive serves more at a deeper queue (a prompt's start, before any expert read is queued).
        let workers = max(1, min(16, rowIDs.count))
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            var raw = [UInt8](repeating: 0, count: rowBytes)
            for k in stride(from: worker, to: rowIDs.count, by: workers) {
                let id = rowIDs[k]
                guard id >= 0, id < rows else {
                    failure.lock.lock()
                    failure.error = EngineError("Engram row \(id) is outside the table")
                    failure.lock.unlock()
                    return
                }
                do {
                    try raw.withUnsafeMutableBytes { try file.read(offset: table.offset + id * Int64(rowBytes), into: $0) }
                    raw.withUnsafeBytes { Self.decodeEngramRow($0.baseAddress!, type: table.type, dim: dim, into: out + k * dim) }
                } catch {
                    failure.lock.lock()
                    failure.error = error
                    failure.lock.unlock()
                    return
                }
            }
        }
        if let error = failure.error { throw error }
    }

    /// The bytes of one row of an Engram table of `dim` values: whole blocks of a ggml type, or DwarfStar's I8 rows
    /// (`e4m3_e8m0_32_row264`: `dim` E4M3 codes, then one E8M0 scale per 32); nil for a table no decoder reads.
    package static func engramRowBytes(_ table: GGUFSource.Tensor, dim: Int) -> Int? {
        if table.type == 24 { return dim % 32 == 0 && table.shape.last == dim + dim / 32 ? dim + dim / 32 : nil }
        guard table.shape.last == dim, GGUFQuant.readable.contains(table.type), let (block, bytes) = GGUFQuant.sizes[table.type],
            dim % block == 0
        else { return nil }
        return dim / block * bytes
    }

    /// One Engram row as floats. An I8 row's value i is E4M3 code i times 2^(scale - 127) of its 32, rounded to bf16
    /// (ties to even) as DwarfStar's reader does; a scale of 0xFF, like a NaN code, decodes as NaN.
    package static func decodeEngramRow(_ raw: UnsafeRawPointer, type: Int, dim: Int, into out: UnsafeMutablePointer<Float>) {
        guard type == 24 else {
            GGUFQuant.dequantize(raw, type: type, count: dim, into: out)
            return
        }
        for i in 0..<dim {
            let scale = raw.load(fromByteOffset: dim + i / 32, as: UInt8.self)
            let value =
                scale == 255
                ? Float.nan
                : GGUFQuant.e4m3(raw.load(fromByteOffset: i, as: UInt8.self))
                    * Float(sign: .plus, exponent: Int(scale) - 127, significand: 1)
            let bits = value.bitPattern
            out[i] = value.isNaN ? value : Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
        }
    }

    /// `sourcesOnly`: through the last key/value source's window and compressed rows, no logits (`feed`). `fed`: the
    /// first rows so, the rest whole; past that source only the rest run (rows `lo ..< n` of each step).
    private func run(_ tokens: [Int], sourcesOnly: Bool, fed: Int = 0) throws -> [Float] {
        let n = tokens.count
        let h = hidden
        let lastSource = kvSource.last ?? -1
        let through = sourcesOnly ? lastSource : layers.count
        var x = try work("streams-a", n * 4 * h)
        var y = try work("streams-b", n * 4 * h)
        let embedded = try work("embedded", n * h)
        for (i, token) in tokens.enumerated() {
            guard (0..<embed.outputs).contains(token) else { throw EngineError("Token \(token) is outside the vocabulary") }
            FamilyWeights.row(embed, token, into: embedded.contents().bindMemory(to: Float.self, capacity: n * h) + i * h)
        }
        let cols = (ngram - 1) * engramHeads
        var engramBuffers: [Int: MTLBuffer] = [:]
        for (e, l) in engramLayers.enumerated() where l <= through {
            let buffer = try work("engram-rows-\(e)", n * cols * engramDim)
            try engramRows(tokens, layer: e, into: buffer.contents().bindMemory(to: Float.self, capacity: n * cols * engramDim))
            engramBuffers[l] = buffer
        }
        // The first attention collapses the streams with the identity (stream 0); each later sublayer with the pre-mix
        // the sublayer before it computed.
        topkRows = try work("ds-topk", n * indexTop)
        candidateKeep = try work("ds-candidates", candidateSource >= 0 ? n * candidateStride : 1)
        let prevPre = try work("pre-ffn", n * 4)
        let attnPre = try work("pre-attn", n * 4)
        let identity = prevPre.contents().bindMemory(to: Float.self, capacity: n * 4)
        for r in 0..<n {
            identity[r * 4] = 1
            identity[r * 4 + 1] = 0
            identity[r * 4 + 2] = 0
            identity[r * 4 + 3] = 0
        }
        do {
            let (grid, group) = grid2(h, n)
            try kernels.dispatch(encoder, "hc_broadcast", [Ref(embedded), Ref(x)], ints: [Int32(h), Int32(n)], grid: grid, group: group)
        }
        let post = try work("hc-post", n * 4)
        let comb = try work("hc-comb", n * 16)
        for (l, layer) in layers.enumerated() where l <= through {
            let kvOnly = l == through
            // Rows from `lo` run this layer's attention, from `ffnLo` its feed-forward.
            let lo = l > lastSource ? fed : 0
            let ffnLo = l >= lastSource ? fed : 0
            // A one-row decode step (a couple of seconds at most) finishes: cancelling it would throw the conversation away;
            // the request honours cancellation between tokens. A prompt's rows stop between layers.
            guard n == 1 || shouldContinue() else {
                try flush(wait: true)
                throw EngineError("Cancelled", code: "cancelled")
            }
            if let engram = layer.engram, let rows = engramBuffers[l] {
                // A chunk at a time (the gate is per row), so its buffer does not grow with the prompt.
                try chunked(n - lo) { first, count in
                    let first = lo + first
                    let kv = try work("engram-kv", count * 5 * h)
                    try mm(encoder, engram.kv, Ref(rows, first * cols * engramDim * 4), Ref(kv), rows: count)
                    try kernels.dispatch(
                        encoder, "engram_gate", [Ref(x, first * 4 * h * 4), Ref(kv), Ref(engram.weight)], ints: [Int32(h), 4],
                        floats: [eps],
                        grid: MTLSize(width: count * 4, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
                }
            }
            // Attention half, in chunks: its mixes give this layer's FFN pre-mix (attnPre) and the post/comb that expand it.
            step = "layer \(l) attention rows \(position)+\(n)"
            try chunked(n - lo) { first, rows in
                let first = lo + first
                let kvOnly = kvOnly || l == lastSource && first < fed
                let streams = Ref(x, first * 4 * h * 4)
                if !kvOnly {
                    try mixes(
                        layer.hcAttn, streams: streams, rows: rows, pre: Ref(attnPre, first * 16), post: Ref(post, first * 16),
                        comb: Ref(comb, first * 64))
                }
                let cur = try work("attn-in", rows * h)
                let out = try work("attn-out", rows * h)
                try collapse(streams, pre: Ref(prevPre, first * 16), out: Ref(cur), rows: rows)
                try rms(encoder, Ref(cur), layer.attnNorm, Ref(cur), n: h, rows: rows, eps: eps)
                try attention(encoder, layer: l, input: cur, output: out, first: first, rows: rows, kvOnly: kvOnly)
                guard !kvOnly else { return }
                let (grid, group) = grid2(h, rows)
                try kernels.dispatch(
                    encoder, "hc_expand", [Ref(out), streams, Ref(post, first * 16), Ref(comb, first * 64), Ref(y, first * 4 * h * 4)],
                    ints: [Int32(h), Int32(rows)], grid: grid, group: group)
            }
            if kvOnly { break }
            swap(&x, &y)
            if n == 1, routing.delayedExperts != nil { try injectDelayedExperts(at: l, streams: x, combine: comb) }
            // FFN half: its mixes give the next attention's pre-mix (prevPre); it collapses with attnPre.
            step = "layer \(l) feed-forward rows \(position)+\(n)"
            // `ffnIn` and `mixed` hold rows from `ffnLo`, so `moe` sees the rows and slices a call of them alone has.
            let m = n - ffnLo
            let ffnIn = try work("ffn-in", n * h)
            let mixed = try work("ffn-out", n * h)
            try chunked(m) { at, rows in
                let first = ffnLo + at
                let streams = Ref(x, first * 4 * h * 4)
                try mixes(
                    layer.hcFFN, streams: streams, rows: rows, pre: Ref(prevPre, first * 16), post: Ref(post, first * 16),
                    comb: Ref(comb, first * 64))
                try collapse(streams, pre: Ref(attnPre, first * 16), out: Ref(ffnIn, at * h * 4), rows: rows)
                try rms(encoder, Ref(ffnIn, at * h * 4), layer.ffnNorm, Ref(ffnIn, at * h * 4), n: h, rows: rows, eps: eps)
            }
            // Not into a fed pass's last layer: its key/value source runs no feed-forward.
            var guessIn: MTLBuffer?
            if n == 1, guessesNext, l + 1 < through {
                // Decode guesses l + 1's experts from its own mixes and norm of these streams: its FFN collapses with the
                // pre-mix of the attention before it, so this mirrors that lag on the streams there are now.
                let after = layers[l + 1]
                let guessPre = try work("guess-pre", 4)
                let guess = try work("guess-in", h)
                try mixes(
                    after.hcAttn, streams: Ref(x), rows: 1, pre: Ref(guessPre), post: Ref(try work("guess-post", 4)),
                    comb: Ref(try work("guess-comb", 16)))
                try collapse(Ref(x), pre: Ref(guessPre), out: Ref(guess), rows: 1)
                try rms(encoder, Ref(guess), after.ffnNorm, Ref(guess), n: h, rows: 1, eps: eps)
                guessIn = guess
            }
            // Nor does a prompt read ahead into the last layer, which reads only its last row's experts.
            let next =
                l + 1 < through && (n == 1 || returnsAllLogits || l + 2 < layers.count)
                ? (l + 1, layers[l + 1].router, layers[l + 1].routerBias, guessIn) : nil
            // The last layer's FFN feeds only the head's last row (its attention wrote the state): a prompt reads only that row's experts there.
            try moe(
                layer: l, router: layer.router, bias: layer.routerBias, shared: layer.shared, input: ffnIn, output: mixed, rows: m,
                probability: { v in (v > 20 ? v : log1p(exp(v))).squareRoot() }, normalize: normTopK, scale: routedScale,
                sum: { $0 + 1e-20 }, limit: swiglu,
                next: next, keepRows: m > 1 && !returnsAllLogits && l == layers.count - 1 ? m - 1..<m : nil,
                delayedPost: post, mayDefer: n == 1 && !sourcesOnly && l + 1 < layers.count && layers[l + 1].engram == nil)
            let (grid, group) = grid2(h, m)
            try kernels.dispatch(
                encoder, "hc_expand",
                [Ref(mixed), Ref(x, ffnLo * 4 * h * 4), Ref(post, ffnLo * 16), Ref(comb, ffnLo * 64), Ref(y, ffnLo * 4 * h * 4)],
                ints: [Int32(h), Int32(m)], grid: grid, group: group)
            swap(&x, &y)
            if n > 1 { try flush(wait: false) }
        }
        // A fed pass reads no head: its last buffers are checked here, else a stopped one would leave its rows half written.
        if sourcesOnly {
            try flush(wait: true)
            return []
        }
        if returnsAllLogits {
            // Keep each head projection at one row, as decode does. The preceding multirow trunk remains an
            // explicit numerical experiment. A single final wait retires every output row before CPU copying.
            step = "block heads"
            let normed = try work("head-in", n * h)
            let logits = try work("logits", n * output.outputs)
            for row in 0..<n {
                let input = Ref(normed, row * h * 4)
                try collapse(Ref(x, row * 4 * h * 4), pre: Ref(prevPre, row * 16), out: input, rows: 1)
                try rms(encoder, input, outputNorm, input, n: h, rows: 1, eps: eps)
                try mm(encoder, output, input, Ref(logits, row * output.outputs * 4), rows: 1)
            }
            try flush(wait: true)
            let values = logits.contents().bindMemory(to: Float.self, capacity: n * output.outputs)
            return Array(UnsafeBufferPointer(start: values, count: n * output.outputs))
        }
        // Head: the last row's streams collapsed with the last FFN's pre-mix.
        step = "head"
        let normed = try work("head-in", h)
        try collapse(Ref(x, (n - 1) * 4 * h * 4), pre: Ref(prevPre, (n - 1) * 16), out: Ref(normed), rows: 1)
        try rms(encoder, Ref(normed), outputNorm, Ref(normed), n: h, rows: 1, eps: eps)
        return try head(normed, output: output)
    }

    /// pre/post/comb of `rows` rows from their streams (RMS-normalised flat streams through hc_fn, then Sinkhorn).
    func mixes(_ hc: HC, streams: Ref, rows: Int, pre: Ref, post: Ref, comb: Ref) throws {
        let flat = try work("hc-flat", rows * 4 * hidden)
        let mixes = try work("hc-mixes", rows * 24)
        try rms(encoder, streams, nil, Ref(flat), n: 4 * hidden, rows: rows, eps: eps)
        try mm(encoder, hc.fn, Ref(flat), Ref(mixes), rows: rows)
        try kernels.dispatch(
            encoder, "hc_prepare", [Ref(mixes), Ref(hc.scale), Ref(hc.base), pre, post, comb], ints: [Int32(rows), Int32(sinkhorn)],
            floats: [hcEps],
            grid: MTLSize(width: (rows * 16 + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    func collapse(_ streams: Ref, pre: Ref, out: Ref, rows: Int) throws {
        let (grid, group) = grid2(hidden, rows)
        try kernels.dispatch(encoder, "hc_collapse", [streams, pre, out], ints: [Int32(hidden), Int32(rows)], grid: grid, group: group)
    }
    func ropeTail(_ x: Ref, layer: Layer, dim: Int, vectors: Int, rows: Int, first: Int, step: Int = 1, inverse: Bool = false) throws {
        try kernels.dispatch(
            encoder, "rope_tail", [x, Ref(layer.freqs)],
            ints: [Int32(dim), Int32(rope), Int32(vectors), Int32(first), Int32(step), inverse ? 1 : 0],
            grid: MTLSize(width: 1, height: vectors, depth: rows), group: MTLSize(width: rope / 2, height: 1, depth: 1))
    }

    /// The attention of layer `l` for rows `first ..< first + rows` (positions `position + first …`); `kvOnly` writes
    /// only their window rows and compressed rows (`feed`).
    func attention(
        _ e: MTLComputeCommandEncoder, layer l: Int, input: MTLBuffer, output: MTLBuffer, first: Int, rows: Int, kvOnly: Bool = false
    ) throws {
        let layer = layers[l]
        let start = position + first
        let d = headDim
        // This token's key/value row goes into the window ring (slot = position % ring), possibly in two runs.
        let fresh = try work("ds-kv", rows * d)
        let ringBuffer = ringKV[l]!
        try mm(e, layer.kv, Ref(input), Ref(fresh), rows: rows)
        try rms(e, Ref(fresh), layer.kvNorm, Ref(fresh), n: d, rows: rows, eps: eps)
        try ropeTail(Ref(fresh), layer: layer, dim: d, vectors: 1, rows: rows, first: start)
        var done = 0
        while done < rows {
            let slot = (start + done) % ring
            let run = min(rows - done, ring - slot)
            let (grid, group) = grid2(d, run)
            try kernels.dispatch(
                e, "copy_rows", [Ref(fresh, done * d * 4), Ref(ringBuffer, slot * d * 4)], ints: [Int32(d), Int32(run)], grid: grid,
                group: group)
            done += run
        }
        if kvOnly {
            if layer.compressor != nil { try compress(e, layer: l, input: input, first: start, rows: rows) }
            return
        }
        let qr = try work("ds-qr", rows * qRank)
        let q = try work("ds-q", rows * heads * d)
        try mm(e, layer.qA, Ref(input), Ref(qr), rows: rows)
        try rms(e, Ref(qr), layer.qANorm, Ref(qr), n: qRank, rows: rows, eps: eps)
        try mm(e, layer.qB, Ref(qr), Ref(q), rows: rows)
        try ropeTail(Ref(q), layer: layer, dim: d, vectors: heads, rows: rows, first: start)
        // Selection: the window, then (compressed layers) the indexer's top compressed rows.
        let compressed = layer.ratio > 0
        let selected = window + (compressed ? indexTop : 0)
        let sel = try work("ds-sel", rows * selected)
        try kernels.dispatch(
            e, "ring_select", [Ref(sel)], ints: [Int32(window), Int32(ring), Int32(selected), Int32(start)],
            grid: MTLSize(width: (window + 63) / 64, height: rows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        var second = ringBuffer
        if compressed {
            let source = kvSource[l]
            if layer.compressor != nil { try compress(e, layer: l, input: input, first: start, rows: rows) }
            // index() commits between its slices: from here on the open encoder is `encoder`, not `e`.
            if layer.indexer != nil { try index(e, layer: l, input: input, qr: qr, first: first, rows: rows) }
            // Copy the source's top-k (already offset by the ring size) into the selection rows after the window.
            try kernels.dispatch(
                encoder, "copy_int_rows", [Ref(topkRows, first * indexTop * 4), Ref(sel, window * 4)],
                ints: [Int32(indexTop), Int32(rows), Int32(indexTop), Int32(selected)],
                grid: MTLSize(width: (indexTop + 63) / 64, height: rows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
            second = latent[source]!
        }
        let scores = try work("ds-scores", rows * heads * selected)
        let o = try work("ds-o", rows * heads * d)
        // The heads-parallel kernels, bitwise the same as the row kernels: eight heads a thread where they divide (they keep
        // the GPU busy on the few rows of a slice), else one head a thread for a few rows (decode).
        let eights = heads % 8 == 0
        let few = rows <= 8
        let variant = eights ? "_heads8" : few ? "_heads" : ""
        let depth = eights ? heads / 8 : few ? heads : 1
        // f32 scores and values in slices of rows that fit the open buffer (`sliced`): each row's arithmetic is its own.
        try sliced(rows, perRow: Self.attendWork(heads: heads, selected: selected, dim: d)) { t, m in
            let s = Ref(scores, t * heads * selected * 4)
            let picked = Ref(sel, t * selected * 4)
            try kernels.dispatch(
                encoder, "attend_scores" + variant, [Ref(q, t * heads * d * 4), Ref(ringBuffer), Ref(second), picked, s],
                ints: [Int32(heads), Int32(d), Int32(selected), Int32(heads * d), Int32(d), 0, 0, Int32(ring)],
                floats: [1 / Float(d).squareRoot()],
                grid: MTLSize(width: (selected + 63) / 64, height: m, depth: depth), group: MTLSize(width: 64, height: 1, depth: 1))
            try kernels.dispatch(
                encoder, "softmax_rows", [s, Ref(layer.sinks)], ints: [Int32(selected), 1, Int32(heads)],
                grid: MTLSize(width: m * heads, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            try kernels.dispatch(
                encoder, "attend_values" + variant, [s, Ref(ringBuffer), Ref(second), picked, Ref(o, t * heads * d * 4)],
                ints: [Int32(heads), Int32(d), Int32(selected), 0, 0, Int32(heads * d), Int32(d), Int32(ring)], floats: [0],
                grid: MTLSize(width: ((eights ? d / 4 : d) + 63) / 64, height: m, depth: depth),
                group: MTLSize(width: 64, height: 1, depth: 1))
        }
        try ropeTail(Ref(o), layer: layer, dim: d, vectors: heads, rows: rows, first: start, inverse: true)
        // Grouped low-rank output: each group of heads through its own rows of wo_a, then wo_b.
        let perGroup = heads / groups * d
        let og = try work("ds-og", groups * rows * perGroup)
        let low = try work("ds-low", groups * rows * oRank)
        let lowRows = try work("ds-low-rows", rows * groups * oRank)
        try permute(encoder, Ref(o), Ref(og), a: rows, b: groups, n: perGroup)
        try mmBatch(
            encoder, (0..<groups).map { rowsOf(layer.outA, first: $0 * oRank, count: oRank) }, Ref(og), inputStride: rows * perGroup * 4,
            Ref(low), outputStride: rows * oRank * 4, rows: rows)
        try permute(encoder, Ref(low), Ref(lowRows), a: groups, b: rows, n: oRank)
        try mm(encoder, layer.outB, Ref(lowRows), Ref(output), rows: rows)
    }

    /// A key/value source's compressor for these rows: ratio 1 projects each token, ratio 2 pools each completed pair of
    /// tokens through a softmax gate per channel; then the RMS norm, the index key (from the unrotated latent), and RoPE
    /// at the group's first position.
    func compress(_ e: MTLComputeCommandEncoder, layer l: Int, input: MTLBuffer, first: Int, rows: Int) throws {
        let layer = layers[l]
        let c = layer.compressor!
        let ratio = layer.ratio
        let d = headDim
        let latents = latent[l]!
        let keys = indexKeys[l]!
        let firstGroup: Int
        let groupsDone: Int
        if ratio == 1 {
            try mm(e, c.kv, Ref(input), Ref(latents, first * d * 4), rows: rows)
            firstGroup = first
            groupsDone = rows
        } else {
            let done = try poolRows(
                e, key: compKV[l]!, gate: compScore[l]!, ape: try noPositionBias(), output: latents,
                first: first, rows: rows, dim: d, pool: ratio
            ) { key, gate in
                try mm(e, c.kv, Ref(input), key, rows: rows)
                try mm(e, c.gate!, Ref(input), gate, rows: rows)
            }
            firstGroup = done.first
            groupsDone = done.count
            guard groupsDone > 0 else { return }
        }
        let rowsRef = Ref(latents, firstGroup * d * 4)
        try rms(e, rowsRef, c.norm, rowsRef, n: d, rows: groupsDone, eps: eps)
        let keyRef = Ref(keys, firstGroup * indexDim * 4)
        try mm(e, c.indexK, rowsRef, keyRef, rows: groupsDone)
        try rms(e, keyRef, c.indexNorm, keyRef, n: indexDim, rows: groupsDone, eps: eps)
        try ropeTail(keyRef, layer: layer, dim: indexDim, vectors: 1, rows: groupsDone, first: firstGroup * ratio, step: ratio)
        try ropeTail(rowsRef, layer: layer, dim: d, vectors: 1, rows: groupsDone, first: firstGroup * ratio, step: ratio)
    }

    /// An index source's top compressed rows for these rows (into the shared top-k rows, offset by the ring size),
    /// with DeepSeek's two-level candidate blocks from the candidate source on.
    /// `first` is the forward's row; positions are `position + first …`.
    func index(_ e: MTLComputeCommandEncoder, layer l: Int, input: MTLBuffer, qr: MTLBuffer, first relative: Int, rows: Int) throws {
        let layer = layers[l]
        let x = layer.indexer!
        let ratio = layer.ratio
        let source = kvSource[l]
        let first = position + relative
        let entries = (first + rows) / ratio
        let rowTopk = Ref(topkRows, relative * indexTop * 4)
        guard entries > 0 else {
            try kernels.dispatch(
                e, "fill_int", [rowTopk], ints: [Int32(rows * indexTop), -1],
                grid: MTLSize(width: (rows * indexTop + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        let iq = try work("ds-iq", rows * indexHeads * indexDim)
        let w = try work("ds-iw", rows * indexHeads)
        try mm(e, x.qB, Ref(qr), Ref(iq), rows: rows)
        try ropeTail(Ref(iq), layer: layer, dim: indexDim, vectors: indexHeads, rows: rows, first: first)
        try mm(e, x.proj, Ref(input), Ref(w), rows: rows)
        try kernels.dispatch(
            e, "scale_inplace", [Ref(w)], ints: [Int32(rows * indexHeads)], floats: [1 / Float(indexDim * indexHeads).squareRoot()],
            grid: MTLSize(width: (rows * indexHeads + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        let scores = try work("ds-iscores", rows * entries)
        let blocks = (entries + candidateBlock - 1) / candidateBlock
        let keep = candidateKeep!
        let stride = candidateStride
        let candidates = candidateSource >= 0 && blocks > 0 && l >= candidateSource
        if candidates {
            guard stride >= blocks else { throw EngineError("DeepSeek-V4.1-Flash: \(blocks) candidate blocks exceed the stride \(stride)") }
        }
        // Scoring and selection grow with the context: in slices of rows, each its own share of a buffer. From here on
        // the open encoder is `encoder`, not `e`. Every kernel is one dispatch over a slice's rows (row t at
        // position at + t), the same arithmetic per row as one row at a time.
        let perRow = Self.indexWork(
            entries: entries, heads: indexHeads, dim: indexDim, blocks: candidates && l == candidateSource ? blocks : 0)
        try sliced(rows, perRow: perRow) { done, n in
            let at = first + done
            let s = Ref(scores, done * entries * 4)
            let k = Ref(keep, (relative + done) * stride * 4)
            // Later index sources reuse the candidate source's block mask. Discarded scores were -inf after
            // apply_block_mask; writing that value directly avoids reading keys and computing discarded dots.
            // Surviving scores use the same accumulation order, and selection still sees the full score row.
            let masked = candidates && l > candidateSource
            var scoreRefs = [Ref(iq, done * indexHeads * indexDim * 4), Ref(indexKeys[source]!), Ref(w, done * indexHeads * 4), s]
            var scoreInts = [Int32(indexHeads), Int32(indexDim), Int32(entries), Int32(ratio), Int32(at)]
            if masked {
                scoreRefs.append(k)
                scoreInts += [Int32(candidateBlock), Int32(stride)]
            }
            try kernels.dispatch(
                encoder, masked ? "index_scores_masked" : "index_scores", scoreRefs, ints: scoreInts,
                grid: MTLSize(width: (entries + 63) / 64, height: n, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
            if candidates && l == candidateSource {
                // keep[row][b]; rows share the stride so later layers find them.
                try kernels.dispatch(
                    encoder, "candidate_blocks", [s, k],
                    ints: [
                        Int32(entries), Int32(candidateBlock), Int32(blocks), Int32(candidateBlocks), Int32(at), Int32(ratio),
                        Int32(stride),
                    ],
                    grid: MTLSize(width: n, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            }
            try kernels.dispatch(
                encoder, "index_select", [s, Ref(topkRows, (relative + done) * indexTop * 4)],
                ints: [Int32(entries), Int32(ratio), Int32(indexTop), Int32(indexTop), Int32(at), 0, 1, Int32(ring), Int32(indexTop)],
                grid: MTLSize(width: n, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
        }
    }
}
