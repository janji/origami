import Foundation

/// Host-joined execution of one layer's mixers for 1...256 chronological rows: the hyper-connection
/// entry (`hyperConnection`: 4 x 2560 residual streams -> one 2560 input plus 4 injection gates), the
/// gated delta net (`gdn`, layers with layer % 4 != 3) and sparse attention (`qsa`, layer % 4 == 3).
/// The whole-execution schedule is the normal serving path; this class serves short prefill scopes,
/// the readout mixer, replays after a flagged numerical exception, cache bypasses and the MTP pair step.
/// `linear`/`tensor` callbacks keep storage and residency outside the math. Every optional helper
/// (pinned kernels, fused HC/GDN, device block) is BITWISE against the host arithmetic it replaces, and
/// a helper that returns nil falls back to that arithmetic with the unchanged input state. Layer state
/// (`GDNState`, `QSAState`) is published only after every fallible step of the layer has succeeded.
public final class LayerMath {
    public typealias Linear = ([Float], String, Bool) throws -> [Float]
    public typealias Tensor = (String) throws -> [Float]
    public let kernels: ReferenceKernels
    private let linear: Linear
    private let tensor: Tensor
    private let reductions: PinnedReductions?
    private let selector: BatchedTopIndices?
    private let pinnedAttention: PinnedAttention?
    package var qsaExactKernelCounts: PinnedAttention.QSAExactKernelCounts? { pinnedAttention?.qsaExactKernelCounts }
    private var decodeRequestID: UUID?
    private var decodeCapacity = 0
    private var decodeStorage: [Int: QSADecodeStorage] = [:]
    private var pagedDecodeStorage: [Int: QSAPagedAttentionStorage] = [:]

    /// Image tokens need the pinned RoPE, the only path that applies M-RoPE positions.
    func setPositionMap(_ map: RoPEPositionMap?) throws {
        guard let pinnedRoPE else {
            if map == nil { return }
            throw EngineError("Image input needs the pinned RoPE configuration", code: "invalid_configuration")
        }
        pinnedRoPE.positionMap = map
    }
    func wholeQSAHelpers() throws -> (attention: PinnedAttention, rope: PinnedRoPE, indexer: PinnedIndexer) {
        guard let pinnedAttention, let pinnedRoPE, let pinnedIndexer else {
            throw EngineError("Whole QSA requires all original pinned helpers", code: "invalid_configuration")
        }
        return (pinnedAttention, pinnedRoPE, pinnedIndexer)
    }
    /// Shares the existing request-bound mirror owner. Invalidated speculative
    /// suffixes are removed only after every writer has joined/discarded.
    func wholeDecodeMirror(layer: Int, requestID: UUID) throws -> QSADecodeStorage {
        guard decodeRequestID == requestID, (0..<48).contains(layer), layer % 4 == 3, let pinnedAttention else {
            throw EngineError("Whole decode mirror request/layer identity differs")
        }
        if let value = decodeStorage[layer] {
            if !value.snapshot().invalidated { return value }
            decodeStorage.removeValue(forKey: layer)
        }
        let value = try pinnedAttention.makeDecodeStorage(capacityTokens: decodeCapacity)
        decodeStorage[layer] = value
        return value
    }
    func retireInvalidWholeMirror(layer: Int, requestID: UUID) throws {
        guard decodeRequestID == requestID else { throw EngineError("Whole mirror retirement request differs") }
        if let value = decodeStorage[layer], value.snapshot().invalidated { decodeStorage.removeValue(forKey: layer) }
        if let value = pagedDecodeStorage[layer], value.snapshot().invalidated { pagedDecodeStorage.removeValue(forKey: layer) }
    }
    /// Moves every request-owned QSA mirror/table of `requestID` back to `tokens` (MTP verify rollback).
    func rollbackWholeStorage(requestID: UUID, tokens: Int) throws {
        guard decodeRequestID == requestID else { throw EngineError("Whole storage rollback request differs") }
        for value in decodeStorage.values where !value.snapshot().invalidated { try value.rollback(to: tokens) }
        for value in pagedDecodeStorage.values where !value.snapshot().invalidated { try value.rollback(to: tokens) }
    }
    func wholePagedStorage(layer: Int, requestID: UUID) throws -> QSAPagedAttentionStorage {
        guard decodeRequestID == requestID, layer % 4 == 3, (0..<48).contains(layer), let pinnedAttention else {
            throw EngineError("Shared QSA request/layer identity differs")
        }
        if let value = pagedDecodeStorage[layer], !value.snapshot().invalidated { return value }
        let value = try pinnedAttention.makePagedStorage(capacityTokens: decodeCapacity)
        pagedDecodeStorage[layer] = value
        return value
    }

    public func beginDecodeStorage(requestID: UUID, capacity: Int) throws {
        guard pinnedAttention != nil, decodeRequestID == nil, decodeStorage.isEmpty, pagedDecodeStorage.isEmpty,
            (1...ModelContract.maximumContext).contains(capacity)
        else { throw EngineError("Invalid request-owned QSA decode admission") }
        decodeRequestID = requestID
        decodeCapacity = capacity
    }

    public func finishDecodeStorage(requestID: UUID) {
        guard decodeRequestID == requestID else { return }
        // Every pinned attention call joins before returning, including failure.
        decodeStorage.removeAll()
        pagedDecodeStorage.removeAll()
        decodeRequestID = nil
        decodeCapacity = 0
    }
    private let hyperConnectionEpilogue: FusedHyperConnection?
    private let gdnPreparation: FusedGDNPreparation?
    private let gdnBufferRegion: GDNBufferRegion?
    public var shouldContinue: () -> Bool = { true }
    private let batchedRoPE: Bool
    private let pinnedRoPE: PinnedRoPE?
    private let pinnedIndexer: PinnedIndexer?
    public typealias HCDecode = ([Float], String, Bool, Bool, () -> Bool) throws -> FusedHCDecode.Result?
    public var hyperConnectionDecode: HCDecode?
    public typealias GDNInputs = ([Float], String, () -> Bool) throws -> GDNInputProjections.Result?
    public var gdnInputProjections: GDNInputs?
    public typealias GDNBlock = ([Float], String, GDNState, Bool, () -> Bool) throws -> GDNDeviceBlock.Result?
    public var gdnDeviceBlock: GDNBlock?
    public var capture: ((String, [Float]) -> Void)?

    public init(
        kernels: ReferenceKernels, linear: @escaping Linear, tensor: @escaping Tensor,
        reductions: PinnedReductions? = nil, selector: BatchedTopIndices? = nil,
        pinnedAttention: PinnedAttention? = nil, batchedRoPE: Bool = false, pinnedRoPE: PinnedRoPE? = nil,
        pinnedIndexer: PinnedIndexer? = nil, hyperConnectionEpilogue: FusedHyperConnection? = nil,
        gdnPreparation: FusedGDNPreparation? = nil, gdnBufferRegion: GDNBufferRegion? = nil
    ) {
        self.hyperConnectionEpilogue = hyperConnectionEpilogue
        self.gdnPreparation = gdnPreparation
        self.gdnBufferRegion = gdnBufferRegion
        self.kernels = kernels
        self.linear = linear
        self.tensor = tensor
        self.pinnedAttention = pinnedAttention
        self.batchedRoPE = batchedRoPE
        self.pinnedRoPE = pinnedRoPE
        self.pinnedIndexer = pinnedIndexer
        self.reductions = reductions
        self.selector = selector
    }

    private func rows(_ x: [Float], width: Int) throws -> Int {
        guard !x.isEmpty, x.count % width == 0, x.count / width <= 256 else {
            throw EngineError("Reference layer requires 1...256 chronological rows")
        }
        return x.count / width
    }

    private func weights(_ name: String, count: Int) throws -> [Float] {
        let value = try tensor(name)
        guard value.count == count else { throw EngineError("Tensor shape mismatch: \(name)") }
        return value
    }

    private func project(_ x: [Float], _ name: String, count: Int) throws -> [Float] {
        let value = try linear(x, name, true)
        guard value.count == count else { throw EngineError("Projection shape mismatch: \(name)") }
        return value
    }

    private func rotate(_ x: [Float], width: Int, heads: Int, positions: [Int]) throws -> [Float] {
        if let pinnedRoPE { return try pinnedRoPE.rotation(x, width: width, heads: heads, positions: positions) }
        if batchedRoPE { return try kernels.ropeBatch(x: x, width: width, heads: heads, positions: positions) }
        var output: [Float] = []
        output.reserveCapacity(x.count)
        for (row, position) in positions.enumerated() {
            let start = row * heads * width
            output.append(contentsOf: try kernels.rope(x: Array(x[start..<(start + heads * width)]), width: width, position: position))
        }
        return output
    }

    /// `base` is an attn/mlp/final hyper-connection tensor prefix. The final
    /// mixer supplies useCombine=false and receives an empty injection array.
    public func hyperConnection(residual: [Float], base: String, useCombine: Bool = true)
        throws -> (mixed: [Float], injection: [Float])
    {
        try autoreleasepool {
            let tokens = try rows(residual, width: 10240)
            if tokens != 2, let result = try hyperConnectionDecode?(residual, base, useCombine, capture != nil, shouldContinue) {
                if let capture {
                    guard let norm = result.norm, let down = result.down, let up = result.sigmoidUp else {
                        throw EngineError("HC decode omitted requested activation capture")
                    }
                    capture(base + ".normed", norm)
                    capture(base + ".down", down)
                    capture(base + ".wup", up)
                }
                return (result.mixed, result.injection)
            }
            let normWeights = try weights(base + ".hc_norm.weight", count: 10240)
            let norm =
                try reductions?.groupedRMS(x: residual, weights: normWeights, groupSize: 2560)
                ?? kernels.rmsnorm(x: residual, weights: normWeights, groupSize: 2560)
            let down = try project(norm, base + ".input_mix_weight_down", count: tokens * 320)
            let activated = down.map { kernels.arithmetic.silu(LayerArithmetic.round($0 / 4)) }
            let rawUp = try project(activated, base + ".input_mix_weight_up", count: residual.count)
            let up: [Float]
            let mixed: [Float]
            if let hyperConnectionEpilogue, hyperConnectionEpilogue.usesGPU(rows: tokens) {
                let result = try hyperConnectionEpilogue.evaluate(
                    norm: norm, rawUp: rawUp,
                    captureUp: capture != nil, shouldContinue: shouldContinue)
                mixed = result.mixed
                up = result.sigmoidUp ?? []
            } else {
                // Retain the existing host path below the configured row floor.
                up = rawUp.map(kernels.arithmetic.sigmoid)
                var values = [Float](repeating: 0, count: tokens * 2560)
                for token in 0..<tokens {
                    for column in 0..<2560 {
                        var sum: Float = 0
                        for branch in 0..<4 {
                            let index = token * 10240 + branch * 2560 + column
                            // Pinned strided BF16 sum4 uses col_reduce_small:
                            // four sequential BF16 additions, then BF16 mean.
                            sum = LayerArithmetic.round(sum + LayerArithmetic.round(up[index] * norm[index]))
                        }
                        values[token * 2560 + column] = LayerArithmetic.round(sum / 4)
                    }
                }
                mixed = values
            }
            var injection: [Float] = []
            if useCombine {
                injection = try project(norm, base + ".block_inject_weight", count: tokens * 4).map {
                    LayerArithmetic.round(2 * kernels.arithmetic.sigmoid(LayerArithmetic.round($0 / 4)))
                }
            }
            capture?(base + ".normed", norm)
            capture?(base + ".down", down)
            capture?(base + ".wup", up)
            return (mixed, injection)
        }
    }

    public func gdn(x: [Float], layer: Int, state: inout GDNState) throws -> [Float] {
        try autoreleasepool {
            let tokens = try rows(x, width: 2560)
            guard (0..<48).contains(layer), layer % 4 != 3 else { throw EngineError("Layer is not GDN") }
            try state.validate()
            guard tokens <= ModelContract.maximumContext - state.offset else { throw EngineError("GDN context limit exceeded") }
            let base = "model.layers.\(layer).linear_attn"
            if let result = try gdnDeviceBlock?(x, base, state, capture != nil, shouldContinue) {
                state = result.state
                for name in [
                    "mixed", "z", "a", "b", "convolved", "query", "keys", "values", "beta", "decay", "recurrence", "normed", "gated",
                    "output",
                ] {
                    if let values = result.captures[name] { capture?(base + "." + name, values) }
                }
                return result.output
            }
            // Cache refusal or joined numerical exception replay falls through
            // to the unchanged original path with the original state.
            let mixed: [Float]
            let z: [Float]
            let b: [Float]
            let a: [Float]
            if let inputs = try gdnInputProjections?(x, base, shouldContinue) {
                guard inputs.qkv.count == tokens * 10240, inputs.z.count == tokens * 6144,
                    inputs.b.count == tokens * 48, inputs.a.count == tokens * 48
                else {
                    throw EngineError("GDN input composition returned invalid projection geometry")
                }
                mixed = inputs.qkv
                z = inputs.z
                b = inputs.b
                a = inputs.a
            } else {
                mixed = try project(x, base + ".in_proj_qkv", count: tokens * 10240)
                z = try project(x, base + ".in_proj_z", count: tokens * 6144)
                b = try project(x, base + ".in_proj_b", count: tokens * 48)
                a = try project(x, base + ".in_proj_a", count: tokens * 48)
            }
            let convolution = try weights(base + ".conv1d.weight", count: 10240 * 4)
            let dtBias = try weights(base + ".dt_bias", count: 48)
            let aLog = try weights(base + ".A_log", count: 48)
            let history = state.convolutionTail
            typealias GDNPrepared = (conv: [Float], q: [Float], k: [Float], v: [Float], beta: [Float], decay: [Float], nextTail: [UInt16])
            func originalPreparation() throws -> GDNPrepared {
                let raw = history.map(BF16.decode) + mixed
                func convolve() -> [Float] {
                    var conv = [Float](repeating: 0, count: mixed.count)
                    for token in 0..<tokens {
                        for channel in 0..<10240 {
                            var sum: Float = 0
                            for tap in 0..<4 { sum += raw[(token + tap) * 10240 + channel] * convolution[channel * 4 + tap] }
                            conv[token * 10240 + channel] = kernels.arithmetic.silu(LayerArithmetic.round(sum))
                        }
                    }
                    return conv
                }
                let conv = convolve()
                var q: [Float] = []
                var k: [Float] = []
                var v: [Float] = []
                q.reserveCapacity(tokens * 2048)
                k.reserveCapacity(tokens * 2048)
                v.reserveCapacity(tokens * 6144)
                for token in 0..<tokens {
                    let start = token * 10240
                    q.append(contentsOf: conv[start..<(start + 2048)])
                    k.append(contentsOf: conv[(start + 2048)..<(start + 4096)])
                    v.append(contentsOf: conv[(start + 4096)..<(start + 10240)])
                }
                let beta = b.map(kernels.arithmetic.sigmoid)
                let decay: [Float]
                if let reductions {
                    let softplus = a.enumerated().map { index, value in
                        kernels.arithmetic.softplus(LayerArithmetic.round(value + dtBias[index % 48]))
                    }
                    let prepared = try reductions.prepareGDN(query: q, keys: k, aLog: aLog, softplus: softplus)
                    q = prepared.query
                    k = prepared.keys
                    decay = prepared.decay
                } else {
                    q = try LayerArithmetic.l2norm(q, width: 128, scale: 1 / sqrtf(128))
                    k = try LayerArithmetic.l2norm(k, width: 128)
                    decay = a.enumerated().map { index, value in
                        expf(-expf(aLog[index % 48]) * kernels.arithmetic.softplus(LayerArithmetic.round(value + dtBias[index % 48])))
                    }
                }
                return (conv, q, k, v, beta, decay, raw.suffix(3 * 10240).map(BF16.encode))
            }
            let prepared: GDNPrepared
            let fusion = gdnPreparation.flatMap { $0.usesGPU(rows: tokens) ? $0 : nil }
            if let fusion {
                func fusedPreparation() throws -> FusedGDNPreparation.Prepared {
                    try fusion.prepare(
                        mixed: mixed, tail: history, convolution: convolution, a: a, b: b,
                        dtBias: dtBias, aLog: aLog, captureConvolved: capture != nil, shouldContinue: shouldContinue)
                }
                let result = try fusedPreparation()
                prepared = (result.convolved ?? [], result.query, result.keys, result.values, result.beta, result.decay, result.nextTail)
            } else {
                prepared = try originalPreparation()
            }
            let (conv, q, k, v, beta, decay, nextTail) = prepared
            var nextState = state
            let recurrenceOutput: [Float]
            let normed: [Float]
            let gated: [Float]
            if let gdnBufferRegion {
                let normWeights = try weights(base + ".norm.weight", count: 128)
                func region() throws -> GDNBufferRegion.Result {
                    try gdnBufferRegion.evaluate(
                        q: q, k: k, v: v, decay: decay, beta: beta, state: state,
                        normWeights: normWeights, z: z, capture: capture != nil, shouldContinue: shouldContinue)
                }
                let result = try region()
                nextState = result.state
                gated = result.gated
                recurrenceOutput = result.recurrence ?? []
                normed = result.normed ?? []
            } else {
                // Decode is the hot path. Keeping prefill's array control avoids
                // retaining a large tile's output in every immutable state buffer.
                let recurrence = try kernels.gdn(q: q, k: k, v: v, decay: decay, beta: beta, state: state.recurrent)
                recurrenceOutput = recurrence.output
                nextState.recurrent = recurrence.state
                normed = try kernels.rmsnorm(x: recurrenceOutput, weights: weights(base + ".norm.weight", count: 128))
                if let fusion {
                    func fusedGate() throws -> [Float] { try fusion.gate(normed: normed, z: z, shouldContinue: shouldContinue) }
                    gated = try fusedGate()
                } else {
                    gated = zip(normed, z).map { LayerArithmetic.round($0 * kernels.arithmetic.sigmoidFP32($1)) }
                }
            }
            let output = try project(gated, base + ".out_proj", count: tokens * 2560)
            // All fallible work precedes state publication.
            nextState.convolutionTail = nextTail
            nextState.offset += tokens
            state = nextState
            capture?(base + ".mixed", mixed)
            capture?(base + ".z", z)
            capture?(base + ".a", a)
            capture?(base + ".b", b)
            capture?(base + ".convolved", conv)
            capture?(base + ".query", q)
            capture?(base + ".keys", k)
            capture?(base + ".values", v)
            capture?(base + ".beta", beta)
            capture?(base + ".decay", decay)
            capture?(base + ".recurrence", recurrenceOutput)
            capture?(base + ".normed", normed)
            capture?(base + ".gated", gated)
            capture?(base + ".output", output)
            return output
        }
    }

    public func qsa(x: [Float], layer: Int, state: inout QSAState, requestID: UUID? = nil) throws -> [Float] {
        try autoreleasepool {
            let tokens = try rows(x, width: 2560)
            guard (0..<48).contains(layer), layer % 4 == 3 else { throw EngineError("Layer is not QSA") }
            try state.validate()
            guard tokens <= ModelContract.maximumContext - state.offset else { throw EngineError("QSA context limit exceeded") }
            let offset = state.offset
            let kvLength = offset + tokens
            let base = "model.layers.\(layer).self_attn"
            let indexBase = base + ".indexer"
            let qg = try project(x, base + ".q_proj", count: tokens * 24 * 512)
            var queries = [Float](repeating: 0, count: tokens * 24 * 256)
            var gates = queries
            for row in 0..<(tokens * 24) {
                queries.replaceSubrange((row * 256)..<((row + 1) * 256), with: qg[(row * 512)..<(row * 512 + 256)])
                gates.replaceSubrange((row * 256)..<((row + 1) * 256), with: qg[(row * 512 + 256)..<((row + 1) * 512)])
            }
            queries = try kernels.rmsnorm(x: queries, weights: weights(base + ".q_norm.weight", count: 256))
            var newKeys = try kernels.rmsnorm(
                x: project(x, base + ".k_proj", count: tokens * 512),
                weights: weights(base + ".k_norm.weight", count: 256))
            let newValues = try project(x, base + ".v_proj", count: tokens * 512).map(BF16.encode)
            queries = try rotate(queries, width: 256, heads: 24, positions: Array(offset..<kvLength))
            newKeys = try rotate(newKeys, width: 256, heads: 2, positions: Array(offset..<kvLength))
            let newKeyBits = newKeys.map(BF16.encode)
            let indexQK = try project(x, indexBase + ".index_qk_proj", count: tokens * 640)
            var indexQueries: [Float] = []
            var raw = state.rawTail.map(BF16.decode)
            indexQueries.reserveCapacity(tokens * 512)
            raw.reserveCapacity((offset % 4 + tokens) * 128)
            for token in 0..<tokens {
                indexQueries.append(contentsOf: indexQK[(token * 640)..<(token * 640 + 512)])
                raw.append(contentsOf: indexQK[(token * 640 + 512)..<((token + 1) * 640)])
            }
            let newBlocks = raw.count / (4 * 128)
            var pooled = [Float](repeating: 0, count: newBlocks * 128)
            for block in 0..<newBlocks {
                for channel in 0..<128 {
                    var total: Float = 0
                    for token in 0..<4 { total += raw[(block * 4 + token) * 128 + channel] }
                    pooled[block * 128 + channel] = LayerArithmetic.round(total / 4)
                }
            }
            if !pooled.isEmpty {
                pooled = try kernels.rmsnorm(x: pooled, weights: weights(indexBase + ".k_layernorm.weight", count: 128))
                pooled = try rotate(pooled, width: 128, heads: 1, positions: (0..<newBlocks).map { (offset / 4 + $0) * 4 })
            }
            let newPooled = pooled.map(BF16.encode)
            let newTail = raw.suffix((kvLength % 4) * 128).map(BF16.encode)
            // Pool the full incoming tile before any early-query visibility mask.
            // Pinned dot scratch is striped to 32 MiB. Complete score rows
            // remain inside the charged reference phase workspace.
            let allPooled = kvLength > 2048 ? state.pooledKeys.map(BF16.decode) + pooled : []
            if kvLength > 2048 {
                indexQueries = try kernels.rmsnorm(x: indexQueries, weights: weights(indexBase + ".q_layernorm.weight", count: 128))
                indexQueries = try rotate(indexQueries, width: 128, heads: 4, positions: Array(offset..<kvLength))
            }
            let pinnedScores = kvLength > 2048 ? try pinnedIndexer?.scores(query: indexQueries, pooledKeys: allPooled) : nil
            var batchedPositions: [[UInt32]]?
            if let selector, kvLength > 2048 {
                let scores: [Float]
                if let pinnedScores {
                    scores = pinnedScores
                } else {
                    var collected: [Float] = []
                    collected.reserveCapacity(tokens * (kvLength / 4))
                    for token in 0..<tokens {
                        let query = Array(indexQueries[(token * 512)..<((token + 1) * 512)])
                        collected.append(contentsOf: try kernels.indexerScores(query: query, pooledKeys: allPooled))
                    }
                    scores = collected
                }
                batchedPositions = try selector.selectedPositions(scores: scores, positions: Array(offset..<kvLength), kvLength: kvLength)
            }
            func selectedPositions(for token: Int) throws -> [UInt32] {
                if let batchedPositions { return batchedPositions[token] }
                let position = offset + token
                if kvLength > 2048 {
                    if let pinnedScores {
                        let blocks = kvLength / 4
                        return try kernels.selectedPositions(
                            scores: Array(pinnedScores[(token * blocks)..<((token + 1) * blocks)]),
                            position: position, kvLength: kvLength)
                    }
                    let range = (token * 512)..<((token + 1) * 512)
                    let iq = Array(indexQueries[range])
                    return try kernels.selectedPositions(
                        scores: kernels.indexerScores(query: iq, pooledKeys: allPooled),
                        position: position, kvLength: kvLength)
                }
                return (0...position).map(UInt32.init)
            }
            var attended: [Float] = []
            if let pinnedAttention {
                let positions = try (0..<tokens).map { try selectedPositions(for: $0) }
                if state.pages != nil {
                    let storage: QSAPagedAttentionStorage?
                    if let decodeRequestID {
                        guard requestID == decodeRequestID, tokens == 1 else {
                            throw EngineError("Shared QSA request identity/phase differs")
                        }
                        storage = try wholePagedStorage(layer: layer, requestID: decodeRequestID)
                    } else {
                        storage = nil
                    }
                    attended = try pinnedAttention.attentionPaged(
                        queries: queries, state: state, newKeys: newKeyBits,
                        newValues: newValues, selectedPositions: positions, storage: storage)
                } else {
                    var storage: QSADecodeStorage?
                    if let decodeRequestID {
                        guard requestID == decodeRequestID, tokens == 1 else {
                            throw EngineError("QSA decode storage request identity/phase differs")
                        }
                        // The same lookup as the whole schedule's: an invalidated mirror is replaced, never staged.
                        storage = try wholeDecodeMirror(layer: layer, requestID: decodeRequestID)
                    }
                    attended = try pinnedAttention.attention(
                        queries: queries, oldKeys: state.keys, newKeys: newKeyBits,
                        oldValues: state.values, newValues: newValues, selectedPositions: positions, decodeStorage: storage)
                }
            } else {
                attended.reserveCapacity(tokens * 6144)
                let oldKeys = try state.keys
                let oldValues = try state.values
                for token in 0..<tokens {
                    let positions = try selectedPositions(for: token)
                    let query = Array(queries[(token * 6144)..<((token + 1) * 6144)])
                    let output: [Float]
                    if positions.isEmpty {
                        // Preserve the independent unfused fallback's all-masked
                        // uniform row while never widening the entire K/V cache.
                        let probability = LayerArithmetic.round(1 / Float(kvLength))
                        var mean = [Float](repeating: 0, count: 512)
                        for key in 0..<kvLength {
                            for channel in 0..<512 {
                                let value = key < offset ? oldValues[key * 512 + channel] : newValues[(key - offset) * 512 + channel]
                                mean[channel] += probability * BF16.decode(value)
                            }
                        }
                        output = (0..<(24 * 256)).map { LayerArithmetic.round(mean[($0 / 256 / 12) * 256 + $0 % 256]) }
                    } else {
                        let selectedKeys = gather(oldKeys, newKeyBits, offset: offset, positions: positions)
                        let selectedValues = gather(oldValues, newValues, offset: offset, positions: positions)
                        // All gathered rows are visible; absolute positions have
                        // already affected RoPE and the full-domain selector.
                        output = try kernels.attention(
                            query: query, keys: selectedKeys, values: selectedValues,
                            kvLength: positions.count, position: positions.count - 1)
                    }
                    attended.append(contentsOf: output)
                }
            }
            let gated = zip(attended, gates).map { LayerArithmetic.round($0 * kernels.arithmetic.sigmoid($1)) }
            let output = try project(gated, base + ".o_proj", count: tokens * 2560)
            try state.append(keys: newKeyBits, values: newValues, pooledKeys: newPooled, rawTail: newTail, tokens: tokens)
            capture?(base + ".query", queries)
            capture?(base + ".attended", attended)
            capture?(base + ".output", output)
            return output
        }
    }

    private func gather(_ old: [UInt16], _ new: [UInt16], offset: Int, positions: [UInt32]) -> [Float] {
        var output = [Float](repeating: 0, count: 2 * positions.count * 256)
        for (row, absolute) in positions.enumerated() {
            let position = Int(absolute)
            for head in 0..<2 {
                for channel in 0..<256 {
                    let index = head * 256 + channel
                    let value = position < offset ? old[position * 512 + index] : new[(position - offset) * 512 + index]
                    output[(head * positions.count + row) * 256 + channel] = BF16.decode(value)
                }
            }
        }
        return output
    }
}
