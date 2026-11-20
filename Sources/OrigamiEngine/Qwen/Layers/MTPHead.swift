import Accelerate
import CExactIO
import Foundation
import Metal

/// The checkpoint's multi-token-prediction head as an MTP draft for speculative decoding (opt-in `speculation`).
/// A port of Ollama's `mtpDraft.Forward` (79% of depth-1 drafts were accepted on the ballpark replies): slot S
/// fuses the token at S+1 with the target's final multi-stream residual at S, runs one attention +
/// 512-expert MoE layer with its own key/value cache and predicts the token at S+2. Dense matrices are dequantized to
/// float once and run on the CPU; the ten routed experts run on the GPU from one buffer of the head's expert records.
/// Attention covers the last `window` slots (2,048, where the sparse selection would keep every block anyway).
/// NUMERIC: plain FP32 (Accelerate) arithmetic. A draft only decides which tokens are tried as a two-token step;
/// the emitted token always comes from the target's logits through `ReferenceSampler.verify`.
/// shortcut: one draft token (depth 1) and a sliding window instead of the indexer selection; deeper drafts and the
/// indexer when a measured Mac shows they pay.
package final class MTPHead {
    static let hc = 4, d = 2560, heads = 24, kvHeads = 2, headDim = 256, ropeDims = 64, theta: Float = 10_000_000
    static let eps: Float = 1e-6, window = 2048
    private struct Linear {
        let w: [Float]
        let rows, cols: Int
    }
    private let dense: [String: Linear]
    private let norms: [String: [Float]]
    let experts: MTLBuffer
    private let reservation: MemoryLedger.Reservation, ledger: MemoryLedger
    private var keys: [[Float]] = [], values: [[Float]] = [], positions: [Int] = []

    /// `file`: safetensors with `language_model.mtp.*` in MLX's layout (ARC4NUM's MLX-Serve build ships them). The file
    /// is untrusted: every tensor is checked against the size the head uses before it is read or copied.
    package init(file: URL, device: MTLDevice, ledger: MemoryLedger) throws {
        let spans = try ModelSource.header(of: file, file: file.lastPathComponent)
        let p = "language_model.mtp."
        let format = AffineFormat(bits: 4, group: 64)
        let recordBytes = format.expertRecordBytes
        // Each expert bank holds 512 experts' worth of its part (the three projections have equal part sizes), so the
        // record copy below fills the expert buffer exactly.
        let parts = [
            "weight": format.codeBytes(width: 2560, outputs: 640), "scales": format.scaleBytes(width: 2560, outputs: 640),
            "biases": format.biasBytes(width: 2560, outputs: 640),
        ]
        let banks = ["gate_proj", "up_proj", "down_proj"].map { p + "layers.0.mlp.switch_mlp.\($0)." }
        guard
            banks.allSatisfy({ bank in
                parts.allSatisfy { part, bytes in
                    spans[bank + part].map { $0.shape.first == 512 && $0.length == Int64(512 * bytes) } ?? false
                }
            })
        else {
            throw EngineError("\(file.path) has no MTP head in the 4-bit group-64 MLX layout", code: "model_incompatible")
        }
        self.ledger = ledger
        let reservation = try ledger.reserve(bytes: UInt64(512 * recordBytes) + 400_000_000, category: "mtp_head")
        var dense: [String: Linear] = [:]
        var norms: [String: [Float]] = [:]
        let buffer: MTLBuffer
        do {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            func bytes(_ name: String, row: Int? = nil) throws -> Data {
                guard let s = spans[name] else { throw EngineError("MTP head lacks \(name)", code: "model_incompatible") }
                var offset = s.offset
                var length = s.length
                if let row {
                    let per = s.length / Int64(s.shape[0])
                    offset += per * Int64(row)
                    length = per
                }
                try handle.seek(toOffset: UInt64(offset))
                guard let data = try handle.read(upToCount: Int(length)), data.count == Int(length) else {
                    throw EngineError("MTP head read short at \(name)")
                }
                return data
            }
            func bf16(_ data: Data) -> [Float] { data.withUnsafeBytes { $0.bindMemory(to: UInt16.self).map(BF16.decode) } }
            func linear(_ base: String) throws {
                guard let s = spans[p + base + ".weight"], s.shape.count == 2, s.shape[0] > 0, s.shape[1] > 0 else {
                    throw EngineError("MTP head lacks the matrix \(base)", code: "model_incompatible")
                }
                // The header tied each length to its shape and dtype, so a BF16 matrix holds rows × cols values.
                let w = try bytes(p + base + ".weight")
                if s.dtype == "BF16" {
                    dense[base] = Linear(w: bf16(w), rows: s.shape[0], cols: s.shape[1])
                    return
                }
                let scales = try bf16(bytes(p + base + ".scales"))
                let biases = try bf16(bytes(p + base + ".biases"))
                let rows = s.shape[0]
                let words = s.shape[1]
                let groups = scales.count / rows
                let cols = groups * 64
                // MLX packs whole 2-, 4- or 8-bit codes into each u32 word, one scale and bias per 64 columns.
                guard s.dtype == "U32", groups > 0, scales.count == rows * groups, biases.count == scales.count,
                    words * 32 % cols == 0, [2, 4, 8].contains(words * 32 / cols)
                else {
                    throw EngineError("MTP head \(base) is not a 2-, 4- or 8-bit group-64 matrix", code: "model_incompatible")
                }
                let bits = words * 32 / cols
                var out = [Float](repeating: 0, count: rows * cols)
                w.withUnsafeBytes { raw in
                    let packed = raw.bindMemory(to: UInt32.self)
                    let per = 32 / bits
                    let mask = UInt32((1 << bits) - 1)
                    for r in 0..<rows {
                        for c in 0..<cols {
                            let word = packed[r * words + c / per]
                            let q = Float((word >> UInt32((c % per) * bits)) & mask)
                            let g = r * groups + c / 64
                            out[r * cols + c] = q * scales[g] + biases[g]
                        }
                    }
                }
                dense[base] = Linear(w: out, rows: rows, cols: cols)
            }
            for base in [
                "fc_embedding", "fc_hidden", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj",
                "layers.0.self_attn.o_proj", "layers.0.mlp.gate", "layers.0.mlp.shared_expert.gate_proj",
                "layers.0.mlp.shared_expert.up_proj", "layers.0.mlp.shared_expert.down_proj", "layers.0.mlp.shared_expert_gate",
            ] {
                try linear(base)
            }
            for hcBase in ["layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection", "hyper_connection_mixer"] {
                try linear(hcBase + ".input_mix_weight_down")
                try linear(hcBase + ".input_mix_weight_up")
                norms[hcBase + ".hc_norm"] = try bf16(bytes(p + hcBase + ".hc_norm.weight"))
                if hcBase != "hyper_connection_mixer" { try linear(hcBase + ".block_inject_weight") }
            }
            for name in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "layers.0.self_attn.q_norm", "layers.0.self_attn.k_norm"] {
                norms[name] = try bf16(bytes(p + name + ".weight"))
            }
            // Every matrix and norm has the size `step` uses it at (free: the shared expert's width, each mixer's rank).
            let d = Self.d
            let streams = Self.hc * Self.d
            let a = "layers.0.self_attn"
            let m = "layers.0.mlp"
            let inner = dense[m + ".shared_expert.gate_proj"]?.rows ?? 0
            var shapes = [
                "fc_embedding": (d, d), "fc_hidden": (d, d), a + ".q_proj": (Self.heads * 2 * Self.headDim, d),
                a + ".k_proj": (Self.kvHeads * Self.headDim, d), a + ".v_proj": (Self.kvHeads * Self.headDim, d),
                a + ".o_proj": (d, Self.heads * Self.headDim), m + ".gate": (512, d), m + ".shared_expert.gate_proj": (inner, d),
                m + ".shared_expert.up_proj": (inner, d), m + ".shared_expert.down_proj": (d, inner), m + ".shared_expert_gate": (1, d),
            ]
            var lengths: [String: Set<Int>] = [
                "pre_fc_norm_embedding": [d], "pre_fc_norm_hidden": [streams],
                a + ".q_norm": [Self.headDim], a + ".k_norm": [Self.headDim],
            ]
            for hcBase in ["layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection", "hyper_connection_mixer"] {
                let rank = dense[hcBase + ".input_mix_weight_down"]?.rows ?? 0
                shapes[hcBase + ".input_mix_weight_down"] = (rank, streams)
                shapes[hcBase + ".input_mix_weight_up"] = (streams, rank)
                if hcBase != "hyper_connection_mixer" { shapes[hcBase + ".block_inject_weight"] = (Self.hc, streams) }
                lengths[hcBase + ".hc_norm"] = [d, streams]
            }
            guard shapes.allSatisfy({ name, shape in dense[name].map { $0.rows == shape.0 && $0.cols == shape.1 } ?? false }),
                lengths.allSatisfy({ name, allowed in norms[name].map { allowed.contains($0.count) } ?? false })
            else {
                throw EngineError(
                    "\(file.path): the MTP head's matrices or norms differ from Qwen3.8-Flash-Next's", code: "model_incompatible")
            }
            // The expert records, in the order PackStore reads a record: gate, up, down; weight, scales, biases each.
            guard let made = device.makeBuffer(length: 512 * recordBytes, options: .storageModeShared) else {
                throw EngineError("MTP expert buffer unavailable", code: "insufficient_memory")
            }
            var cursor = 0
            for e in 0..<512 {
                for bank in banks {
                    for part in ["weight", "scales", "biases"] {
                        let data = try bytes(bank + part, row: e)
                        data.withUnsafeBytes {
                            made.contents().advanced(by: cursor).copyMemory(from: $0.baseAddress!, byteCount: data.count)
                        }
                        cursor += data.count
                    }
                }
            }
            buffer = made
        } catch {
            // A failed load is not kept, so its charge goes now (deinit runs only for a finished head).
            try? ledger.release(reservation)
            throw error
        }
        experts = buffer
        self.dense = dense
        self.norms = norms
        self.reservation = reservation
    }
    deinit { try? ledger.release(reservation) }

    /// Forget the cache (a new request).
    package func reset() {
        keys.removeAll()
        values.removeAll()
        positions.removeAll()
    }
    package var cachedSlots: Int { positions.count }

    // MARK: arithmetic (float, row-major)
    private func mv(_ l: Linear, _ x: UnsafePointer<Float>) -> [Float] {
        var y = [Float](repeating: 0, count: l.rows)
        l.w.withUnsafeBufferPointer { w in
            origami_sgemv(Int32(l.rows), Int32(l.cols), w.baseAddress!, x, &y)
        }
        return y
    }
    private func mv(_ name: String, _ x: [Float]) -> [Float] { x.withUnsafeBufferPointer { mv(dense[name]!, $0.baseAddress!) } }
    private static func rms(_ x: inout [Float], _ w: [Float], group: Int) {
        for g in stride(from: 0, to: x.count, by: group) {
            var ss: Float = 0
            for i in g..<(g + group) { ss += x[i] * x[i] }
            let inv = 1 / (ss / Float(group) + eps).squareRoot()
            // A weight as long as x (per-stream norms) is indexed by position, a shorter one per group.
            for i in g..<(g + group) { x[i] = x[i] * inv * (w.count == x.count ? w[i] : w[i - g]) }
        }
    }
    private static func sigmoid(_ v: Float) -> Float { v >= 0 ? 1 / (1 + exp(-v)) : exp(v) / (1 + exp(v)) }
    private static func silu(_ v: Float) -> Float { v * sigmoid(v) }
    private static func rope(_ x: inout [Float], base: Int, position: Int) {
        let half = ropeDims / 2
        for i in 0..<half {
            let angle = Float(position) * pow(theta, -Float(2 * i) / Float(ropeDims))
            let c = cos(angle)
            let s = sin(angle)
            let a = x[base + i]
            let b = x[base + half + i]
            x[base + i] = a * c - b * s
            x[base + half + i] = b * c + a * s
        }
    }
    /// Hyper-connection prepare: per-stream norm, mixer, the weighted branch and the normed streams.
    private func prepare(_ x: [Float], _ base: String) -> (branch: [Float], normed: [Float]) {
        var normed = x
        Self.rms(&normed, norms[base + ".hc_norm"]!, group: Self.d)
        let down = mv(base + ".input_mix_weight_down", normed).map { Self.silu($0 / Float(Self.hc)) }
        let mix = mv(base + ".input_mix_weight_up", down)
        var branch = [Float](repeating: 0, count: Self.d)
        for s in 0..<Self.hc { for i in 0..<Self.d { branch[i] += normed[s * Self.d + i] * Self.sigmoid(mix[s * Self.d + i]) } }
        return (branch.map { $0 / Float(Self.hc) }, normed)
    }
    private func inject(_ x: inout [Float], _ normed: [Float], _ branch: [Float], _ base: String) {
        let w = mv(base + ".block_inject_weight", normed).map { 2 * Self.sigmoid($0 / Float(Self.hc)) }
        for s in 0..<Self.hc { for i in 0..<Self.d { x[s * Self.d + i] += w[s] * branch[i] } }
    }

    /// One slot: (token at S+1, embedding of it, target residual at S, position S). Appends the slot's key/value and,
    /// unless `cacheOnly`, returns the hidden the main lm_head turns into the draft for S+2 (`experts` runs the ten
    /// routed experts: input, record offsets, weights → weighted sum).
    package func step(
        embedding: [Float], hidden: [Float], position: Int, cacheOnly: Bool,
        experts run: ([Float], [Int], [Float]) throws -> [Float]
    ) throws -> [Float]? {
        precondition(embedding.count == Self.d && hidden.count == Self.hc * Self.d)
        var e = embedding
        Self.rms(&e, norms["pre_fc_norm_embedding"]!, group: Self.d)
        e = mv("fc_embedding", e)
        var h = hidden
        Self.rms(&h, norms["pre_fc_norm_hidden"]!, group: Self.hc * Self.d)
        var x = [Float](repeating: 0, count: Self.hc * Self.d)
        for s in 0..<Self.hc {
            let projected = h.withUnsafeBufferPointer { mv(dense["fc_hidden"]!, $0.baseAddress! + s * Self.d) }
            for i in 0..<Self.d { x[s * Self.d + i] = projected[i] + e[i] }
        }
        let attnHC = "layers.0.attn_hyper_connection"
        let a = "layers.0.self_attn"
        let (branch, normed) = prepare(x, attnHC)
        var k = mv(a + ".k_proj", branch)
        let v = mv(a + ".v_proj", branch)
        for head in 0..<Self.kvHeads {
            var slice = Array(k[(head * Self.headDim)..<((head + 1) * Self.headDim)])
            Self.rms(&slice, norms[a + ".k_norm"]!, group: Self.headDim)
            Self.rope(&slice, base: 0, position: position)
            k.replaceSubrange((head * Self.headDim)..<((head + 1) * Self.headDim), with: slice)
        }
        keys.append(k)
        values.append(v)
        positions.append(position)
        if keys.count > Self.window {
            keys.removeFirst()
            values.removeFirst()
            positions.removeFirst()
        }
        if cacheOnly { return nil }
        // Attention: 24 query heads (query + gate per head), 2 key/value heads, over the cached slots.
        let qg = mv(a + ".q_proj", branch)
        var out = [Float](repeating: 0, count: Self.heads * Self.headDim)
        let groupSize = Self.heads / Self.kvHeads
        let scale = 1 / Float(Self.headDim).squareRoot()
        for head in 0..<Self.heads {
            var q = Array(qg[(head * 2 * Self.headDim)..<(head * 2 * Self.headDim + Self.headDim)])
            Self.rms(&q, norms[a + ".q_norm"]!, group: Self.headDim)
            Self.rope(&q, base: 0, position: position)
            let kv = head / groupSize
            var scores = keys.map { key -> Float in
                var dot: Float = 0
                key.withUnsafeBufferPointer { kp in
                    vDSP_dotpr(q, 1, kp.baseAddress! + kv * Self.headDim, 1, &dot, vDSP_Length(Self.headDim))
                }
                return dot * scale
            }
            let top = scores.max()!
            var sum: Float = 0
            for i in scores.indices {
                scores[i] = exp(scores[i] - top)
                sum += scores[i]
            }
            for (i, value) in values.enumerated() {
                let wgt = scores[i] / sum
                for j in 0..<Self.headDim { out[head * Self.headDim + j] += wgt * value[kv * Self.headDim + j] }
            }
            for j in 0..<Self.headDim { out[head * Self.headDim + j] *= Self.sigmoid(qg[head * 2 * Self.headDim + Self.headDim + j]) }
        }
        inject(&x, normed, mv(a + ".o_proj", out), attnHC)
        // MoE: softmax over the top ten router logits, ten routed experts (GPU) plus the gated shared expert.
        let mlpHC = "layers.0.mlp_hyper_connection"
        let m = "layers.0.mlp"
        let (mb, mnormed) = prepare(x, mlpHC)
        let logits = mv(m + ".gate", mb)
        let ids = logits.indices.sorted { logits[$0] > logits[$1] || (logits[$0] == logits[$1] && $0 < $1) }.prefix(10)
        let top = logits[ids.first!]
        let ex = ids.map { exp(logits[$0] - top) }
        let total = ex.reduce(0, +)
        let routed = try run(mb, Array(ids), ex.map { $0 / total })
        let sg = mv(m + ".shared_expert.gate_proj", mb)
        let su = mv(m + ".shared_expert.up_proj", mb)
        let shared = mv(m + ".shared_expert.down_proj", zip(sg, su).map { Self.silu($0) * $1 })
        let gate = Self.sigmoid(mv(m + ".shared_expert_gate", mb)[0])
        inject(&x, mnormed, zip(routed, shared).map { $0 + $1 * gate }, mlpHC)
        return prepare(x, "hyper_connection_mixer").branch
    }
}
