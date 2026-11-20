import CExactIO
import Foundation
import Metal

/// Owner of the model's fixed (non-expert) weights for the executor. It reads tensors from the PackStore, keeps
/// them in the raw and dense resident caches, pins whole-layer and whole-token weight plans for the encoded
/// decode/prefill paths, and runs the synchronous projections: affine 4-bit on the GPU through ReferenceKernels,
/// BF16 dense through PinnedDenseLinear (pinned_dense, always on in served plans) or BLAS (reference only).
/// Uncached host copies are kept for one layer at a time (`enterLayer` retires them). Exactness follows
/// ReferenceKernels' pinned schedules; the BLAS branch widens BF16 to FP32 and rounds outputs to BF16 on request.
/// Not thread-safe, except `prefetchWholeLayer`, which touches only the locked caches.
public final class ReferenceWeights {
    public let store: PackStore
    public let kernels: ReferenceKernels
    private let ledger: MemoryLedger
    private var cached: [String: Data] = [:]
    private var reservations: [String: MemoryLedger.Reservation] = [:]
    private var currentLayer: Int?
    private var wholeConstants: [Int: WholeLayerConstants] = [:]
    private var wholeConstantBytes: UInt64 = 0
    private let wholeReaders = WholeLayerReaderBudget()
    private let tokenReaders = WholeLayerReaderBudget(maximum: 48)
    private var tokenWeights: [WholeLayerWeights] = []
    private var tokenPlanReservation: MemoryLedger.Reservation?
    static let tokenPlanOwnerBytes: UInt64 = 10 * 1024 * 1024
    var hasWholeTokenPlan: Bool { tokenWeights.count == 48 }

    /// The retained read-only leases remove per-token cache mutation, geometry
    /// construction and descriptor-stat batches. All payload bytes remain in
    /// the same admitted caches; no unchecked or partially pinned plan escapes.
    func prepareWholeTokenPlan(boundary: () throws -> Void) throws {
        guard tokenWeights.isEmpty, tokenPlanReservation == nil else { throw EngineError("Token weight plan is already active") }
        let prepared = try WholeTokenPlanConstruction.prepare(
            ledger: ledger, boundary: boundary,
            acquireLayer: { layer in
                try enterLayer(layer)
                guard let owner = try acquireWholeLayerWeights(layer: layer, forTokenPlan: true) else {
                    throw EngineError(
                        "Complete token weight plan does not fit fixed resident cache at layer \(layer)", code: "token_plan_admission")
                }
                return owner
            }, validate: { try store.validateBatchBoundary() })
        tokenWeights = prepared.owners
        tokenPlanReservation = prepared.reservation
    }
    /// Any layer's retained plan weights without changing the entered layer:
    /// decode pre-routing needs layer L+2's MoE HC and router while layer L+1
    /// is the entered lookahead layer. nil when no complete plan is held.
    func wholeTokenPlanWeights(layer: Int) -> WholeLayerWeights? {
        guard hasWholeTokenPlan, (0..<48).contains(layer) else { return nil }
        return tokenWeights[layer]
    }
    func validateWholeTokenPlan() throws {
        guard hasWholeTokenPlan else { throw EngineError("Complete token weight plan is unavailable") }
        try store.validateBatchBoundary()
    }
    /// Caller owns the serial scope and has joined every GPU reader first.
    func retireWholeTokenPlan() {
        tokenWeights.removeAll()
        if let reservation = tokenPlanReservation {
            try? ledger.release(reservation)
            tokenPlanReservation = nil
        }
    }
    public let simdMatrix: Bool
    public let residentBytes: UInt64
    public let residentSmallTensors: Bool
    public static let maximumSmallTensorBytes: Int64 = 1_048_576
    public let denseResidentBytes: UInt64
    private let residentCache: ResidentTensorCache?
    private let denseResidentCache: ResidentFloatTensorCache?
    private let fusedExpertExecution: FusedExpertExecution?
    private let denseKernels: PinnedDenseLinear?
    private let hcDecode: FusedHCDecode?
    private let gdnInputs: GDNInputProjections?
    private let hcDecodeEnabled, hcPrefillEnabled: Bool
    public var residentSnapshot: ResidentTensorCache.Snapshot? { residentCache?.snapshot() }
    public var denseResidentSnapshot: ResidentFloatTensorCache.Snapshot? { denseResidentCache?.snapshot() }

    public init(
        store: PackStore, kernels: ReferenceKernels, ledger: MemoryLedger,
        simdMatrix: Bool = false, residentBytes: UInt64 = 0, residentSmallTensors: Bool = false, denseResidentBytes: UInt64 = 0,
        fusedExperts: Bool = false,
        pinnedDense: Bool = false, groupedExpertTiled: Bool = false,
        groupedExpertBM32: Bool = false, groupedExpertBM32MinRows: Int = 128, expertScratchReuse: Bool = false, fusedHCDecode: Bool = false,
        fusedHCPrefill: Bool = false, gdnInputComposition: Bool = false,
        groupedExpertMMAMinRows: Int = 0, hcDecodeFusion: Bool = false, wireResident: Bool = false
    ) throws {
        guard !hcDecodeFusion || fusedHCDecode else {
            throw EngineError("hc_decode_fusion requires fused_hc_decode", code: "invalid_configuration")
        }
        guard !residentSmallTensors || residentBytes > 0 else {
            throw EngineError("Small fixed tensor residency requires a positive raw cache", code: "invalid_configuration")
        }
        self.residentSmallTensors = residentSmallTensors
        guard !gdnInputComposition || (pinnedDense && residentBytes > 0 && denseResidentBytes > 0) else {
            throw EngineError(
                "gdn_input_composition requires source pinned dense and positive resident caches", code: "invalid_configuration")
        }
        guard !(fusedHCDecode || fusedHCPrefill) || (pinnedDense && residentBytes > 0 && denseResidentBytes > 0) else {
            throw EngineError(
                "fused_hc_decode requires pinned dense source arithmetic and positive resident caches", code: "invalid_configuration")
        }
        guard !expertScratchReuse || fusedExperts else {
            throw EngineError("Expert scratch reuse requires fused experts", code: "invalid_configuration")
        }
        guard (1...256).contains(groupedExpertBM32MinRows), !groupedExpertBM32 || groupedExpertTiled else {
            throw EngineError("Grouped BM32 weights require grouped tiling and minimum rows in 1...256", code: "invalid_configuration")
        }
        guard !groupedExpertTiled || (fusedExperts && simdMatrix) else {
            throw EngineError("Grouped tiled expert weights require fused SIMD execution", code: "invalid_configuration")
        }
        hcDecodeEnabled = fusedHCDecode
        hcPrefillEnabled = fusedHCPrefill
        self.store = store
        self.kernels = kernels
        self.ledger = ledger
        self.simdMatrix = simdMatrix
        self.residentBytes = residentBytes
        self.denseResidentBytes = denseResidentBytes
        if residentBytes > 0 || denseResidentBytes > 0 {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Resident tensor Metal device unavailable") }
            residentCache =
                residentBytes > 0
                ? try ResidentTensorCache(store: store, device: device, ledger: ledger, byteLimit: residentBytes, wire: wireResident) : nil
            denseResidentCache =
                denseResidentBytes > 0
                ? try ResidentFloatTensorCache(
                    store: store, device: device, ledger: ledger,
                    byteLimit: denseResidentBytes, wire: wireResident) : nil
        } else {
            residentCache = nil
            denseResidentCache = nil
        }
        guard
            groupedExpertMMAMinRows == 0 || (kernels.quantizedMMABF16 && groupedExpertBM32 && (64...256).contains(groupedExpertMMAMinRows))
        else {
            throw EngineError(
                "grouped_expert_mma_min_rows requires quantized_mma_bf16, grouped_expert_bm32 and a floor in 64...256",
                code: "invalid_configuration")
        }
        fusedExpertExecution =
            fusedExperts
            ? try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: groupedExpertTiled, groupedBM32: groupedExpertBM32,
                groupedBM32MinRows: groupedExpertBM32MinRows, scratchReuse: expertScratchReuse,
                groupedPrefetch: groupedExpertBM32 && kernels.quantizedTiledPrefetch, groupedMMAMinRows: groupedExpertMMAMinRows) : nil
        denseKernels = pinnedDense ? try PinnedDenseLinear(ledger: ledger) : nil
        gdnInputs =
            gdnInputComposition
            ? try GDNInputProjections(kernels: kernels, dense: denseKernels!, ledger: ledger, simdMatrix: simdMatrix) : nil
        hcDecode =
            (fusedHCDecode || fusedHCPrefill)
            ? try FusedHCDecode(
                arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: fusedHCPrefill ? kernels : nil,
                prefillSIMDMatrix: simdMatrix, decodeFusion: hcDecodeFusion) : nil
    }
    deinit {
        if let tokenPlanReservation { try? ledger.release(tokenPlanReservation) }
        for reservation in reservations.values { try? ledger.release(reservation) }
    }

    public func enterLayer(_ layer: Int?) throws {
        if currentLayer == layer { return }
        cached.removeAll()
        for reservation in reservations.values { try ledger.release(reservation) }
        reservations.removeAll()
        currentLayer = layer
        // A token plan holds every immutable layer lease across this batch.
        // Token entry still validates all descriptors; actual reads and fallback
        // cache acquisitions also retain their own validation boundaries.
        if !hasWholeTokenPlan || layer == nil { try store.validateBatchBoundary() }
    }

    public func span(_ name: String) throws -> TensorSpan {
        let key = name.hasPrefix("language_model.") ? name : "language_model." + name
        guard let span = store.tensors[key] else { throw EngineError("Missing tensor: \(key)") }
        return span
    }

    public func data(_ name: String, row: Int? = nil, cache: Bool = true) throws -> Data {
        let key = name.hasPrefix("language_model.") ? name : "language_model." + name
        if row == nil, let value = cached[key] { return value }
        let geometry = try span(key)
        guard geometry.length > 0, geometry.length <= 400_000_000 else { throw EngineError("Reference tensor exceeds bounded loader") }
        let size: Int
        if let row {
            guard let rows = geometry.shape.first, rows > 0, row >= 0, row < rows,
                geometry.length % Int64(rows) == 0
            else { throw EngineError("Invalid tensor row") }
            size = Int(geometry.length / Int64(rows))
        } else {
            size = Int(geometry.length)
        }
        // Four copies conservatively cover Data, host typed view, BF16 decoding
        // and Metal's input copy. Call workspace reservations cover activations.
        // A preceding small resident access may already hold this exact
        // per-layer charge. Reuse it if a later acquisition bypasses; never
        // overwrite an unreleased token. Row reads keep separate lifetimes.
        let existing = row == nil ? reservations[key] : nil
        let reservation = try existing ?? ledger.reserve(bytes: UInt64(size) * 4, category: "reference_weight_and_transfer_bound")
        var retained = false
        defer { if !retained && existing == nil { try? ledger.release(reservation) } }
        var value = Data(count: size)
        try value.withUnsafeMutableBytes { try store.readTensor(key, row: row, into: $0) }
        if row == nil && cache {
            cached[key] = value
            reservations[key] = reservation
            retained = true
        }
        return value
    }

    public func tensor(_ name: String) throws -> [Float] {
        let geometry = try span(name)
        guard geometry.dtype == "BF16" || geometry.dtype == "F32" else {
            throw EngineError("Expected floating point tensor: \(name)")
        }
        if residentSmallTensors, geometry.length > 0, geometry.length <= Self.maximumSmallTensorBytes,
            let lease = try residentCache?.acquire(name)
        {
            // Same cache, exclusions, raw bytes and all-hit source validation as
            // projection residency. A lease survives the complete host copy.
            // Preserve the old per-layer weight/transfer charge; the permanent
            // raw buffer has its own independent physical allocation charge.
            return try withExtendedLifetime(lease) {
                let key = lease.name
                if reservations[key] == nil {
                    reservations[key] = try ledger.reserve(
                        bytes: UInt64(geometry.length) * 4,
                        category: "reference_weight_and_transfer_bound")
                }
                if geometry.dtype == "BF16" {
                    return UnsafeBufferPointer(
                        start: lease.buffer.contents().assumingMemoryBound(to: UInt16.self),
                        count: lease.length / 2
                    ).map(BF16.decode)
                }
                return Array(
                    UnsafeBufferPointer(
                        start: lease.buffer.contents().assumingMemoryBound(to: Float.self),
                        count: lease.length / 4))
            }
        }
        // Optional cache-capacity/eligibility bypass keeps the original path.
        // Identity, read and memory errors propagate; they are never misses.
        let value = try data(name)
        switch geometry.dtype {
        case "BF16": return Self.array(value, UInt16.self).map(BF16.decode)
        case "F32": return Self.array(value, Float.self)
        default: throw EngineError("Expected floating point tensor: \(name)")
        }
    }

    public func integers(_ name: String) throws -> [Int64] {
        guard try span(name).dtype == "I64" else { throw EngineError("Expected int64 tensor") }
        return Self.array(try data(name), Int64.self)
    }

    package func makeGDNDeviceBlock(preparation: FusedGDNPreparation) throws -> GDNDeviceBlock {
        guard let denseKernels, residentCache != nil, denseResidentCache != nil else {
            throw EngineError("GDN block requires resident projection caches")
        }
        return try GDNDeviceBlock(kernels: kernels, dense: denseKernels, preparation: preparation, ledger: ledger, simdMatrix: simdMatrix)
    }

    /// A matrix kept as floats has `shape` in the checkpoint: stored as BF16, or quantized and dequantized at load
    /// (group-32 builds quantize them; their router is 8-bit).
    private static func floatMatrix(_ lease: ResidentFloatTensorLease, _ shape: [Int]) -> Bool {
        lease.sourceSpan.dtype == "BF16"
            ? lease.sourceSpan.shape == shape
            : lease.sourceSpan.dtype == "U32" && lease.sourceSpan.shape.first == shape[0] && lease.elementCount == shape[0] * shape[1]
    }

    /// The tensors one whole-schedule layer acquires, in acquisition order: raw affine4/BF16
    /// tensors, dense float matrices and decoded constants.
    static func wholeLayerSpecs(layer: Int) -> (
        raw: [(String, [Int], String)], dense: [(String, [Int])],
        constants: [(name: String, count: Int)]
    ) {
        let base = "language_model.model.layers.\(layer)"
        let mlp = base + ".mlp"
        let gdnBase = base + ".linear_attn"
        let qsaBase = base + ".self_attn"
        var rawSpecs: [(String, [Int], String)] = []
        var denseSpecs: [(String, [Int])] = []
        var constantSpecs: [(name: String, count: Int)] = []
        func quantized(_ name: String, width: Int, outputs: Int) {
            rawSpecs += [
                (name + ".weight", [outputs, width / 8], "U32"),
                (name + ".scales", [outputs, width / ModelLayout.group], "BF16"),
                (name + ".biases", [outputs, width / ModelLayout.group], "BF16"),
            ]
        }
        func hc(_ name: String) {
            rawSpecs.append((name + ".hc_norm.weight", [10240], "BF16"))
            quantized(name + ".input_mix_weight_down", width: 10240, outputs: 320)
            quantized(name + ".input_mix_weight_up", width: 320, outputs: 10240)
            denseSpecs.append((name + ".block_inject_weight.weight", [4, 10240]))
        }
        hc(base + ".attn_hyper_connection")
        hc(base + ".mlp_hyper_connection")
        if layer % 4 == 3 {
            quantized(qsaBase + ".q_proj", width: 2560, outputs: 12288)
            quantized(qsaBase + ".k_proj", width: 2560, outputs: 512)
            quantized(qsaBase + ".v_proj", width: 2560, outputs: 512)
            denseSpecs.append((qsaBase + ".indexer.index_qk_proj.weight", [640, 2560]))
            quantized(qsaBase + ".o_proj", width: 6144, outputs: 2560)
            constantSpecs = [
                (qsaBase + ".q_norm.weight", 256), (qsaBase + ".k_norm.weight", 256),
                (qsaBase + ".indexer.q_layernorm.weight", 128), (qsaBase + ".indexer.k_layernorm.weight", 128),
            ]
        } else {
            quantized(gdnBase + ".in_proj_qkv", width: 2560, outputs: 10240)
            quantized(gdnBase + ".in_proj_z", width: 2560, outputs: 6144)
            quantized(gdnBase + ".out_proj", width: 6144, outputs: 2560)
            denseSpecs += [(gdnBase + ".in_proj_b.weight", [48, 2560]), (gdnBase + ".in_proj_a.weight", [48, 2560])]
            constantSpecs = [
                (gdnBase + ".conv1d.weight", 40960), (gdnBase + ".dt_bias", 48), (gdnBase + ".A_log", 48), (gdnBase + ".norm.weight", 128),
            ]
        }
        quantized(mlp + ".shared_expert.gate_proj", width: 2560, outputs: 640)
        quantized(mlp + ".shared_expert.up_proj", width: 2560, outputs: 640)
        quantized(mlp + ".shared_expert.down_proj", width: 640, outputs: 2560)
        denseSpecs += [(mlp + ".gate.weight", [512, 2560]), (mlp + ".shared_expert_gate.weight", [1, 2560])]
        return (rawSpecs, denseSpecs, constantSpecs)
    }

    /// Host-stage prefetch: warm the resident caches with layer `layer`'s whole-schedule tensors
    /// (same names and order as `acquireWholeLayerWeights`). Thread-safe: touches only the
    /// lock-protected caches. Returns tensor reads and whether every tensor is now resident.
    func prefetchWholeLayer(_ layer: Int) throws -> (reads: Int, complete: Bool) {
        guard (0..<48).contains(layer), residentSmallTensors, let residentCache, let denseResidentCache else { return (0, false) }
        let (rawSpecs, denseSpecs, constantSpecs) = Self.wholeLayerSpecs(layer: layer)
        var reads = 0
        var complete = true
        // `acquire` admits each group of up to 16 names in sorted order; mirror it.
        var groups = stride(from: 0, to: rawSpecs.count, by: 16).map { rawSpecs[$0..<min($0 + 16, rawSpecs.count)].map(\.0) }
        groups.append(constantSpecs.map(\.name))
        for name in groups.flatMap({ Array(Set($0)).sorted() }) {
            let result = try residentCache.prefetch(name)
            reads += result.read ? 1 : 0
            complete = complete && result.resident
        }
        for (name, _) in denseSpecs {
            let result = try denseResidentCache.prefetch(name)
            reads += result.read ? 1 : 0
            complete = complete && result.resident
        }
        return (reads, complete)
    }

    /// Byte layout of one embedding row (words, scales, biases), from the checkpoint spans.
    func embeddingRowLayout() throws -> EmbeddingRowPrefetch.Layout {
        func rowBytes(_ part: String, _ dtype: String) throws -> Int {
            let geometry = try span("model.embed_tokens." + part)
            guard geometry.dtype == dtype, geometry.shape.count == 2, geometry.shape[0] == 248_320,
                geometry.length % Int64(geometry.shape[0]) == 0
            else { throw EngineError("Embedding row geometry differs") }
            return Int(geometry.length / Int64(geometry.shape[0]))
        }
        let layout = EmbeddingRowPrefetch.Layout(
            wordBytes: try rowBytes("weight", "U32"), scaleBytes: try rowBytes("scales", "BF16"),
            biasBytes: try rowBytes("biases", "BF16"))
        guard layout.wordBytes / 4 * 8 == 2560, layout.scaleBytes == 2560 / ModelLayout.group * 2 else {
            throw EngineError("Embedding row geometry differs")
        }
        return layout
    }

    /// Explicit reader ownership for two adjacent whole-schedule transactions.
    /// This never borrows per-layer host Array charges that enterLayer retires.
    /// Existing raw/dense cache limits, source checks and bypass policy remain.
    func acquireWholeLayerWeights(layer: Int, forTokenPlan: Bool = false) throws -> WholeLayerWeights? {
        guard (0..<48).contains(layer), currentLayer == layer, residentSmallTensors,
            hcDecodeEnabled, hcPrefillEnabled
        else { throw EngineError("Whole-schedule layer/dependencies differ") }
        if hasWholeTokenPlan && !forTokenPlan { return tokenWeights[layer] }
        guard let residentCache, let denseResidentCache else { return nil }
        let reader = try (forTokenPlan ? tokenReaders : wholeReaders).acquire(ledger: ledger)
        let base = "language_model.model.layers.\(layer)"
        let mlp = base + ".mlp"
        let gdnBase = base + ".linear_attn"
        let qsaBase = base + ".self_attn"
        let (rawSpecs, denseSpecs, constantSpecs) = Self.wholeLayerSpecs(layer: layer)
        guard rawSpecs.count <= 48, denseSpecs.count <= 8 else { throw EngineError("Whole-schedule metadata exceeds fixed bound") }
        var raw: [ResidentTensorLease] = []
        var dense: [ResidentFloatTensorLease] = []
        for lo in stride(from: 0, to: rawSpecs.count, by: 16) {
            let specs = Array(rawSpecs[lo..<min(lo + 16, rawSpecs.count)])
            guard let leases = try residentCache.acquire(specs.map { $0.0 }) else { return nil }
            guard leases.count == specs.count,
                zip(leases, specs).allSatisfy({
                    $0.0.name == $0.1.0 && $0.0.span.shape == $0.1.1 && $0.0.span.dtype == $0.1.2
                })
            else { throw EngineError("Whole-schedule raw reader geometry differs") }
            raw.append(contentsOf: leases)
        }
        for (name, shape) in denseSpecs {
            guard let lease = try denseResidentCache.acquire(name) else { return nil }
            guard lease.name == name, Self.floatMatrix(lease, shape) else {
                throw EngineError("Whole-schedule dense reader geometry differs")
            }
            dense.append(lease)
        }
        // Reacquisition validates source identities on hits as it does for the
        // original small-tensor path; decoded constants are immutable copies.
        guard let constantLeases = try residentCache.acquire(constantSpecs.map(\.name)) else { return nil }
        let constants: WholeLayerConstants
        if let cached = wholeConstants[layer] {
            constants = cached
        } else {
            let bytes = try WholeLayerConstants.bytes(counts: constantSpecs.map(\.count))
            guard bytes <= 16 * 1024 * 1024 - wholeConstantBytes else { return nil }
            constants = try WholeLayerConstants(leases: constantLeases, expected: constantSpecs, device: kernels.device, ledger: ledger)
            wholeConstants[layer] = constants
            wholeConstantBytes += constants.chargedBytes
        }
        let r = Dictionary(uniqueKeysWithValues: raw.map { ($0.name, $0.buffer) })
        let d = Dictionary(uniqueKeysWithValues: dense.map { ($0.name, $0.buffer) })
        let c = constants.buffers
        func projection(_ name: String) -> GDNDeviceBlock.Projection {
            .init(words: r[name + ".weight"]!, scales: r[name + ".scales"]!, biases: r[name + ".biases"]!)
        }
        func hyper(_ name: String) -> DecodeFrontierWeights.HC {
            let down = projection(name + ".input_mix_weight_down")
            let up = projection(name + ".input_mix_weight_up")
            return .init(
                norm: r[name + ".hc_norm.weight"]!, injection: d[name + ".block_inject_weight.weight"]!,
                down: .init(words: down.words, scales: down.scales, biases: down.biases),
                up: .init(words: up.words, scales: up.scales, biases: up.biases))
        }
        let gdn: WholeLayerWeights.GDN?
        let qsa: WholeLayerWeights.QSA?
        if layer % 4 == 3 {
            gdn = nil
            qsa = .init(
                queryGate: projection(qsaBase + ".q_proj"), key: projection(qsaBase + ".k_proj"),
                value: projection(qsaBase + ".v_proj"), output: projection(qsaBase + ".o_proj"),
                index: d[qsaBase + ".indexer.index_qk_proj.weight"]!,
                queryNorm: c[qsaBase + ".q_norm.weight"]!, keyNorm: c[qsaBase + ".k_norm.weight"]!,
                indexQueryNorm: c[qsaBase + ".indexer.q_layernorm.weight"]!, indexKeyNorm: c[qsaBase + ".indexer.k_layernorm.weight"]!)
        } else {
            qsa = nil
            gdn = .init(
                qkv: projection(gdnBase + ".in_proj_qkv"), z: projection(gdnBase + ".in_proj_z"), output: projection(gdnBase + ".out_proj"),
                b: d[gdnBase + ".in_proj_b.weight"]!, a: d[gdnBase + ".in_proj_a.weight"]!, convolution: c[gdnBase + ".conv1d.weight"]!,
                bias: c[gdnBase + ".dt_bias"]!, logs: c[gdnBase + ".A_log"]!, norm: c[gdnBase + ".norm.weight"]!)
        }
        let result = WholeLayerWeights(
            layer: layer, attentionHC: hyper(base + ".attn_hyper_connection"), moeHC: hyper(base + ".mlp_hyper_connection"),
            gdn: gdn, qsa: qsa, router: d[mlp + ".gate.weight"]!, sharedOutputGate: d[mlp + ".shared_expert_gate.weight"]!,
            sharedGate: projection(mlp + ".shared_expert.gate_proj"), sharedUp: projection(mlp + ".shared_expert.up_proj"),
            sharedDown: projection(mlp + ".shared_expert.down_proj"), raw: raw + constantLeases, dense: dense,
            constants: constants, reader: reader)
        return result
    }

    package func decodeFrontierHelpers() throws -> (hc: FusedHCDecode, dense: PinnedDenseLinear, fused: FusedExpertExecution) {
        guard hcDecodeEnabled, let hcDecode, let denseKernels, let fusedExpertExecution,
            residentCache != nil, denseResidentCache != nil
        else {
            throw EngineError("Decode frontier requires the existing exact helpers and caches", code: "invalid_configuration")
        }
        return (hcDecode, denseKernels, fusedExpertExecution)
    }

    func gdnDeviceBlock(
        _ x: [Float], base: String, state: GDNState, block: GDNDeviceBlock,
        capture: Bool, shouldContinue: () -> Bool
    ) throws -> GDNDeviceBlock.Result? {
        guard let residentCache, let denseResidentCache else { return nil }
        let canonical = base.hasPrefix("language_model.") ? base : "language_model." + base
        let convolution = try tensor(base + ".conv1d.weight")
        let bias = try tensor(base + ".dt_bias")
        let logs = try tensor(base + ".A_log")
        let norm = try tensor(base + ".norm.weight")
        let names = ["in_proj_qkv", "in_proj_z", "out_proj"].flatMap { stem in
            ["weight", "scales", "biases"].map { canonical + "." + stem + "." + $0 }
        }
        guard let raw = try residentCache.acquire(names) else { return nil }
        return try withExtendedLifetime(raw) {
            guard raw.count == 9, zip(raw, names).allSatisfy({ $0.0.name == $0.1 }) else {
                throw EngineError("GDN block resident group differs")
            }
            for (group, width, outputs) in [(0, 2560, 10240), (1, 2560, 6144), (2, 6144, 2560)] {
                for (part, divisor, dtype) in [(0, 8, "U32"), (1, ModelLayout.group, "BF16"), (2, ModelLayout.group, "BF16")] {
                    let span = raw[group * 3 + part].span
                    guard span.shape == [outputs, width / divisor], span.dtype == dtype else {
                        throw EngineError("GDN block tensor violates checkpoint geometry")
                    }
                }
            }
            guard let b = try denseResidentCache.acquire(canonical + ".in_proj_b.weight") else { return nil }
            return try withExtendedLifetime(b) {
                guard let a = try denseResidentCache.acquire(canonical + ".in_proj_a.weight") else { return nil }
                return try withExtendedLifetime(a) {
                    guard [b, a].allSatisfy({ Self.floatMatrix($0, [48, 2560]) }) else {
                        throw EngineError("GDN block dense tensor violates checkpoint geometry")
                    }
                    func projection(_ i: Int) -> GDNDeviceBlock.Projection {
                        .init(words: raw[i].buffer, scales: raw[i + 1].buffer, biases: raw[i + 2].buffer)
                    }
                    return try block.evaluate(
                        x, qkv: projection(0), z: projection(3), b: b.buffer, a: a.buffer,
                        out: projection(6), convolution: convolution, dtBias: bias, aLog: logs, normWeights: norm,
                        state: state, capture: capture, shouldContinue: shouldContinue)
                }
            }
        }
    }

    public func gdnInputProjections(_ x: [Float], base: String, shouldContinue: () -> Bool) throws -> GDNInputProjections.Result? {
        guard let gdnInputs, let residentCache, let denseResidentCache else { return nil }
        func evaluate() throws -> GDNInputProjections.Result? {
            let canonical = base.hasPrefix("language_model.") ? base : "language_model." + base
            let names = [
                "in_proj_qkv.weight", "in_proj_qkv.scales", "in_proj_qkv.biases",
                "in_proj_z.weight", "in_proj_z.scales", "in_proj_z.biases",
            ].map { canonical + "." + $0 }
            guard let raw = try residentCache.acquire(names) else { return nil }
            return try withExtendedLifetime(raw) {
                guard raw.count == 6, zip(raw, names).allSatisfy({ $0.0.name == $0.1 }) else {
                    throw EngineError("GDN input resident group differs from requested tensors")
                }
                let groups = 2560 / ModelLayout.group
                for (index, shape, dtype) in [
                    (0, [10240, 320], "U32"), (1, [10240, groups], "BF16"),
                    (2, [10240, groups], "BF16"), (3, [6144, 320], "U32"),
                    (4, [6144, groups], "BF16"), (5, [6144, groups], "BF16"),
                ] {
                    guard raw[index].span.shape == shape, raw[index].span.dtype == dtype else {
                        throw EngineError("GDN input resident tensor violates checkpoint geometry")
                    }
                }
                guard let b = try denseResidentCache.acquire(canonical + ".in_proj_b.weight") else { return nil }
                return try withExtendedLifetime(b) {
                    guard let a = try denseResidentCache.acquire(canonical + ".in_proj_a.weight") else { return nil }
                    return try withExtendedLifetime(a) {
                        guard [b, a].allSatisfy({ Self.floatMatrix($0, [48, 2560]) }) else {
                            throw EngineError("GDN dense input group violates checkpoint geometry")
                        }
                        return try gdnInputs.evaluate(
                            x,
                            qkv: .init(words: raw[0].buffer, scales: raw[1].buffer, biases: raw[2].buffer),
                            z: .init(words: raw[3].buffer, scales: raw[4].buffer, biases: raw[5].buffer),
                            b: b.buffer, a: a.buffer, shouldContinue: shouldContinue)
                    }
                }
            }
        }
        return try evaluate()
    }

    public func linear(_ x: [Float], name: String, outputBF16: Bool = true) throws -> [Float] {
        try autoreleasepool {
            let canonical = name.hasPrefix("language_model.") ? name : "language_model." + name
            let weightName = canonical + ".weight"
            let geometry = try span(weightName)
            guard geometry.shape.count == 2 else { throw EngineError("Reference linear must be a matrix") }
            // A matrix kept as floats may come quantized (group-32 builds: an 8-bit router and gates, 4-bit block
            // injects): it is dequantized, as the whole schedule's resident copy is, never read as a 4-bit projection.
            let floatFormat =
                store.tensors[canonical + ".scales"] != nil && ModelContract.floatModules.contains(where: { canonical.hasSuffix($0) })
                ? store.model.format(ofModule: canonical) : nil
            if store.tensors[canonical + ".scales"] != nil, floatFormat == nil {
                guard outputBF16 else { throw EngineError("Quantized FP32-output path is not the pinned BF16 contract") }
                let width = geometry.shape[1] * 8
                if let residentCache,
                    let leases = try residentCache.acquire([weightName, canonical + ".scales", canonical + ".biases"])
                {
                    guard leases.count == 3, leases[0].span.dtype == "U32",
                        leases[1].span.dtype == "BF16", leases[2].span.dtype == "BF16"
                    else {
                        throw EngineError("Resident quantized tensor dtypes violate the checkpoint contract")
                    }
                    // Group admission keeps these tensors from evicting each
                    // other; explicit lifetime extends all pins through the
                    // kernel's final synchronous GPU completion on every path.
                    return try withExtendedLifetime(leases) {
                        try kernels.qlinearPinnedBF16(
                            x: x, width: width, outputs: geometry.shape[0],
                            words: leases[0].buffer, scales: leases[1].buffer, biases: leases[2].buffer,
                            simdMatrix: simdMatrix
                        ).output
                    }
                }
                // Nil is optional-cache bypass. Corrupt identities, short reads
                // and allocation errors propagate and never masquerade as misses.
                let words = Self.array(try data(weightName), UInt32.self)
                let scales = Self.array(try data(canonical + ".scales"), UInt16.self)
                let biases = Self.array(try data(canonical + ".biases"), UInt16.self)
                return try kernels.qlinearPinnedBF16(
                    x: x, width: width, words: words, scales: scales, biases: biases, simdMatrix: simdMatrix
                ).output
            }
            let width = floatFormat.map { geometry.shape[1] * 32 / $0.bits } ?? geometry.shape[1]
            let outputs = geometry.shape[0]
            guard width > 0, outputs > 0, !x.isEmpty, x.count % width == 0 else { throw EngineError("Dense linear geometry mismatch") }
            if let denseKernels {
                guard geometry.dtype == "BF16" || floatFormat != nil else {
                    throw EngineError("Pinned dense requires the checkpoint's BF16 weight storage")
                }
                if let lease = try denseResidentCache?.acquire(weightName) {
                    return try withExtendedLifetime(lease) {
                        try denseKernels.linear(x, weights: lease.buffer, width: width, outputs: outputs, outputBF16: outputBF16)
                    }
                }
                return try denseKernels.linear(
                    x, weights: floatWeights(canonical, format: floatFormat), width: width, outputs: outputs, outputBF16: outputBF16)
            }
            // Stored dense weights (including the router) widen to FP32 for
            // this unchanged BLAS call. outputBF16 controls the final boundary.
            var output = [Float](repeating: 0, count: x.count / width * outputs)
            if let lease = try denseResidentCache?.acquire(weightName) {
                withExtendedLifetime(lease) {
                    origami_sgemm_nt(
                        Int32(x.count / width), Int32(outputs), Int32(width), x,
                        lease.buffer.contents().assumingMemoryBound(to: Float.self), &output)
                }
            } else {
                let weight = try floatWeights(canonical, format: floatFormat)
                origami_sgemm_nt(Int32(x.count / width), Int32(outputs), Int32(width), x, weight, &output)
            }
            return outputBF16 ? output.map { BF16.decode(BF16.encode($0)) } : output
        }
    }

    /// The float values of matrix `module`: its stored floats, or for a quantized float matrix (`format`) its codes
    /// dequantized as `ResidentFloatTensorCache` does, refused unless the scales and biases match that layout.
    private func floatWeights(_ module: String, format: AffineFormat?) throws -> [Float] {
        guard let format else { return try tensor(module + ".weight") }
        let geometry = try span(module + ".weight")
        let rows = geometry.shape[0]
        let columns = geometry.shape[1] * 32 / format.bits
        let words = Self.array(try data(module + ".weight"), UInt32.self)
        let scales = Self.array(try data(module + ".scales"), UInt16.self)
        let biases = Self.array(try data(module + ".biases"), UInt16.self)
        guard geometry.dtype == "U32", !format.fp4, [2, 4, 8].contains(format.bits), format.group > 0, columns % format.group == 0,
            words.count == rows * geometry.shape[1], scales.count == rows * (columns / format.group), biases.count == scales.count
        else {
            throw EngineError(
                "Quantized float matrix \(module) differs from its \(format.bits)-bit group-\(format.group) layout",
                code: "model_incompatible")
        }
        var values = [Float](repeating: 0, count: rows * columns)
        values.withUnsafeMutableBufferPointer {
            ResidentFloatTensorCache.dequantize(
                words: words, scales: scales, biases: biases, rows: rows, columns: columns, format: format, into: $0.baseAddress!)
        }
        return values
    }

    /// Group admission pins all raw HC tensors together. Optional capacity
    /// bypass preserves the ordinary reference path; source/read errors throw.
    /// Both cache lease sets survive all GPU joins, exception repair and capture.
    public func hyperConnectionDecode(
        _ residual: [Float], base: String, useCombine: Bool,
        capture: Bool, shouldContinue: () -> Bool
    ) throws -> FusedHCDecode.Result? {
        let rows = residual.count / 10240
        guard residual.count % 10240 == 0,
            (rows == 1 && hcDecodeEnabled) || ((3...256).contains(rows) && hcPrefillEnabled),
            let hcDecode, let residentCache
        else { return nil }
        let canonical = base.hasPrefix("language_model.") ? base : "language_model." + base
        let names = [
            canonical + ".hc_norm.weight", canonical + ".input_mix_weight_down.weight",
            canonical + ".input_mix_weight_down.scales", canonical + ".input_mix_weight_down.biases",
            canonical + ".input_mix_weight_up.weight", canonical + ".input_mix_weight_up.scales",
            canonical + ".input_mix_weight_up.biases",
        ]
        guard let leases = try residentCache.acquire(names) else { return nil }
        return try withExtendedLifetime(leases) {
            guard leases.count == 7, zip(leases, names).allSatisfy({ pair in pair.0.name == pair.1 }),
                leases[0].span.dtype == "BF16", leases[0].span.length == 10240 * 2
            else {
                throw EngineError("HC resident norm/group differs from fixed checkpoint contract")
            }
            let g = ModelLayout.group
            for (index, shape, dtype) in [
                (1, [320, 1280], "U32"), (2, [320, 10240 / g], "BF16"),
                (3, [320, 10240 / g], "BF16"), (4, [10240, 40], "U32"),
                (5, [10240, 320 / g], "BF16"), (6, [10240, 320 / g], "BF16"),
            ] {
                guard leases[index].span.shape == shape, leases[index].span.dtype == dtype else {
                    throw EngineError("HC resident projection differs from fixed checkpoint contract")
                }
            }
            let injectionLease: ResidentFloatTensorLease?
            if useCombine {
                guard let lease = try denseResidentCache?.acquire(canonical + ".block_inject_weight.weight") else { return nil }
                guard Self.floatMatrix(lease, [4, 10240]) else {
                    throw EngineError("HC resident injection differs from fixed checkpoint contract")
                }
                injectionLease = lease
            } else {
                injectionLease = nil
            }
            return try withExtendedLifetime(injectionLease) {
                try hcDecode.evaluate(
                    residual: residual, normWeights: leases[0].buffer,
                    downWeights: .init(words: leases[1].buffer, scales: leases[2].buffer, biases: leases[3].buffer),
                    upWeights: .init(words: leases[4].buffer, scales: leases[5].buffer, biases: leases[6].buffer),
                    injectionWeights: injectionLease?.buffer, capture: capture, shouldContinue: shouldContinue)
            }
        }
    }

    public func embedding(_ ids: [Int]) throws -> [Float] {
        guard !ids.isEmpty, ids.allSatisfy({ (0..<248_320).contains($0) }) else { throw EngineError("Invalid embedding token IDs") }
        var rows: [Int: [Float]] = [:]
        for id in Set(ids) {
            let base = "model.embed_tokens"
            let words = Self.array(try data(base + ".weight", row: id, cache: false), UInt32.self)
            let scales = Self.array(try data(base + ".scales", row: id, cache: false), UInt16.self)
            let biases = Self.array(try data(base + ".biases", row: id, cache: false), UInt16.self)
            rows[id] = try Self.dequantizedRow(words: words, scales: scales, biases: biases, groupSize: ModelLayout.group)
        }
        return ids.flatMap { rows[$0]! }
    }

    public func expert(_ x: [Float], layer: Int, expert: Int, groupedExpert: Bool = false) throws -> [Float] {
        // The reference path reads the original record; the cached and scheduled
        // expert paths are checked against it.
        guard !x.isEmpty, x.count % 2560 == 0, x.count / 2560 <= 256 else { throw EngineError("Invalid expert activation tile") }
        guard ModelLayout.experts == AffineFormat(bits: 4, group: ModelLayout.group), ModelLayout.ggufExperts == nil else {
            throw EngineError("The reference expert path reads only the checkpoint's affine 4-bit records")
        }
        return try autoreleasepool {
            var record = Data(count: PackStore.expertBytes)
            try record.withUnsafeMutableBytes { try store.readExpert(layer: layer, expert: expert, into: $0, source: .original) }
            func projection(_ input: [Float], _ kind: ExpertProjection) throws -> [Float] {
                let width = kind.width
                let metadata = kind.biasOffset - kind.scaleOffset
                let words = Self.array(record.subdata(in: kind.weightOffset..<kind.scaleOffset), UInt32.self)
                let scales = Self.array(record.subdata(in: kind.scaleOffset..<kind.biasOffset), UInt16.self)
                let biases = Self.array(record.subdata(in: kind.biasOffset..<(kind.biasOffset + metadata)), UInt16.self)
                if groupedExpert {
                    // The caller chooses the grouped schedule for the whole batch, as
                    // the baseline does, even when an expert bucket or the final tile has only one row.
                    return try kernels.qlinearPinnedBF16(
                        x: input, width: width, words: words, scales: scales,
                        biases: biases, simdMatrix: simdMatrix, groupedExpert: true
                    ).output
                }
                // Pinned cached MoE is one gathered QMV per token/expert.
                // Grouping the records for I/O must not silently choose QMM.
                var projected: [Float] = []
                for start in stride(from: 0, to: input.count, by: width) {
                    projected.append(
                        contentsOf: try kernels.qlinearPinnedBF16(
                            x: Array(input[start..<(start + width)]),
                            width: width, words: words, scales: scales, biases: biases
                        ).output)
                }
                return projected
            }
            let gate = try projection(x, .gate)
            let up = try projection(x, .up)
            let product = zip(gate, up).map { LayerArithmetic.round(kernels.arithmetic.silu($0) * $1) }
            return try projection(product, .down)
        }
    }

    /// Same per-token QMV arithmetic as the original path, with an already
    /// verified record pinned by ExpertDemandWindow through GPU completion.
    public func expert(
        _ x: [Float], buffer: ExpertBuffer, groupedExpert: Bool = false,
        shouldContinue: () -> Bool = { true }
    ) throws -> [Float] {
        if let fusedExpertExecution {
            return try fusedExpertExecution.evaluate(
                x, buffer: buffer, groupedExpert: groupedExpert,
                simdMatrix: simdMatrix, shouldContinue: shouldContinue)
        }
        return try ExpertExecution.evaluate(x, buffer: buffer, kernels: kernels, groupedExpert: groupedExpert, simdMatrix: simdMatrix)
    }

    /// The enclosing demand batch owns all record pins until this synchronous
    /// call has joined its command, including every cancellation/error path.
    public func expertBatch(
        _ jobs: [FusedExpertExecution.Job], shouldContinue: () -> Bool = { true },
        consumeOutput: (Int, [Float]) throws -> Void
    ) throws {
        guard let fusedExpertExecution else {
            throw EngineError("Expert command batches require fused expert execution", code: "invalid_configuration")
        }
        try fusedExpertExecution.evaluateBatch(jobs, shouldContinue: shouldContinue, consumeOutput: consumeOutput)
    }
    public func expertDeviceBatch(
        _ jobs: [FusedExpertExecution.DeviceJob], workspace: PrefillMoEWorkspace,
        shouldContinue: () -> Bool = { true }
    ) throws {
        guard let fusedExpertExecution else { throw EngineError("Device MoE requires fused expert execution") }
        try fusedExpertExecution.evaluateDeviceBatch(jobs, workspace: workspace, shouldContinue: shouldContinue)
    }
    func expertDeviceLayer(
        experts: [Int], buffers: [ExpertBuffer], selectedRows: [Int: [(Int, Int)]],
        workspace: PrefillMoEWorkspace, ungroupedTailStart: Int? = nil,
        waves: Int = 1, groupedExpert: Bool = true, partial: Bool = false, shouldContinue: () -> Bool
    ) throws {
        guard let fusedExpertExecution else { throw EngineError("Layer command requires fused expert execution") }
        try fusedExpertExecution.evaluateDeviceLayer(
            experts: experts, buffers: buffers, selectedRows: selectedRows,
            workspace: workspace, simdMatrix: simdMatrix, ungroupedTailStart: ungroupedTailStart,
            waves: waves, groupedExpert: groupedExpert, partial: partial, shouldContinue: shouldContinue)
    }
    /// A prompt was read through: the layer command's buffers may grow back (`CommandBudget`).
    func recoverCommandBudget() { fusedExpertExecution?.commandBudget.recover() }
    public func trimExpertScratch() throws { try fusedExpertExecution?.trimScratch() }
    func makeGatheredExpertDecode(reductions: PinnedReductions) throws -> GatheredExpertDecode {
        guard let fusedExpertExecution else { throw EngineError("Gathered decode requires fused expert arithmetic") }
        return try GatheredExpertDecode(ledger: ledger, fused: fusedExpertExecution, reductions: reductions)
    }

    /// Resident PLE key/value projection leases for DevicePLE, with the same tensors and dtype
    /// checks as `linear`. Nil when the resident cache is absent or bypasses the group.
    func devicePLEProjections() throws -> DevicePLE.Projections? {
        guard let residentCache else { return nil }
        let base = "language_model.model.layers.1.ple."
        let names = ["key_proj", "value_proj"].flatMap { p in ["weight", "scales", "biases"].map { base + p + "." + $0 } }
        guard let leases = try residentCache.acquire(names), leases.count == 6,
            leases[0].span.dtype == "U32", leases[3].span.dtype == "U32",
            [1, 2, 4, 5].allSatisfy({ leases[$0].span.dtype == "BF16" }),
            leases[0].span.shape == [10240, 320], leases[3].span.shape == [2560, 320],
            leases.allSatisfy({ $0.buffer.storageMode == .shared })
        else { return nil }
        return .init(
            keyWords: leases[0].buffer, keyScales: leases[1].buffer, keyBiases: leases[2].buffer,
            valueWords: leases[3].buffer, valueScales: leases[4].buffer, valueBiases: leases[5].buffer,
            retained: leases)
    }

    public static func array<T>(_ data: Data, _ type: T.Type) -> [T] {
        precondition(data.count % MemoryLayout<T>.stride == 0)
        return data.withUnsafeBytes { raw in
            (0..<(raw.count / MemoryLayout<T>.stride)).map { raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.stride, as: T.self) }
        }
    }

    public static func dequantizedRow(words: [UInt32], scales: [UInt16], biases: [UInt16], groupSize: Int) throws -> [Float] {
        guard !words.isEmpty, words.count * 8 % groupSize == 0, scales.count == words.count * 8 / groupSize,
            scales.count == biases.count
        else { throw EngineError("Invalid affine row geometry") }
        return (0..<(words.count * 8)).map { column in
            let nibble = Float((words[column / 8] >> UInt32(column % 8 * 4)) & 15)
            let value = nibble * BF16.decode(scales[column / groupSize]) + BF16.decode(biases[column / groupSize])
            return BF16.decode(BF16.encode(value))
        }
    }
}
