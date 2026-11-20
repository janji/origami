import Foundation
import Metal
import OrigamiEngine

/// The family engines' kernels against plain CPU versions of llama.cpp's graph ops on random data: RMSNorm, mHC
/// (mixes to pre/post/comb with Sinkhorn, collapse, expand), SwiGLU with a clamp, the KDA conv, gate and recurrence,
/// latent attention over a selection, the k-pool and the indexer's top-k (a deterministic set in pool order), the
/// per-token helpers (norms at real widths, stream mean and broadcast, gather and scatter, copies, fills, the window,
/// the permute), DeepSeek's candidate blocks past 4,096 blocks, and the indexer's kernels over many rows against one row
/// at a time (bit for bit).
func familyKernelsMatchReference() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family kernels: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw FixtureUnavailable(description: "no Metal device")
    }
    let kernels = try FamilyKernels(device: device)
    var seed: UInt64 = 0x5EED
    func random(_ n: Int, scale: Float = 1) -> [Float] {
        (0..<n).map { _ in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(seed >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
    }
    func buffer(_ v: [Float]) -> MTLBuffer { device.makeBuffer(bytes: v, length: max(16, v.count * 4), options: .storageModeShared)! }
    func ints(_ v: [Int32]) -> MTLBuffer { device.makeBuffer(bytes: v, length: max(16, v.count * 4), options: .storageModeShared)! }
    func read(_ b: MTLBuffer, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }
    func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        try body(encoder)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
    }
    func close(_ a: [Float], _ b: [Float], _ what: String, tolerance: Float = 2e-4) throws {
        try require(a.count == b.count, "\(what): \(a.count) values for \(b.count)")
        let scale = max(1e-6, b.map(abs).max() ?? 1)
        let worst = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        try require(worst <= tolerance * scale, "\(what): worst difference \(worst) at scale \(scale)")
    }
    typealias Ref = FamilyKernels.Ref
    let one = MTLSize(width: 1, height: 1, depth: 1)

    // RMSNorm with and without a weight.
    let rows = 3
    let n = 1000
    let x = random(rows * n)
    let w = random(n)
    let xb = buffer(x)
    let wb = buffer(w)
    let yb = buffer([Float](repeating: 0, count: rows * n))
    try run {
        try kernels.dispatch(
            $0, "rms_norm", [Ref(xb), Ref(wb), Ref(yb)], ints: [Int32(n), 1], floats: [1e-5],
            grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 1000 / 32 * 32 + 32, height: 1, depth: 1))
    }
    var expected: [Float] = []
    for r in 0..<rows {
        let row = Array(x[(r * n)..<((r + 1) * n)])
        let s = 1 / (row.map { $0 * $0 }.reduce(0, +) / Float(n) + 1e-5).squareRoot()
        expected += zip(row, w).map { $0 * s * $1 }
    }
    try close(read(yb, rows * n), expected, "rms_norm")

    // mHC: mixes -> pre, post, comb (Sinkhorn), then collapse and expand.
    let mixes = random(rows * 24, scale: 3)
    let scale = [0.06, 0.04, 0.09].map(Float.init)
    let base = random(24, scale: 4)
    let pre = buffer([Float](repeating: 0, count: rows * 4))
    let post = buffer([Float](repeating: 0, count: rows * 4))
    let comb = buffer([Float](repeating: 0, count: rows * 16))
    try run {
        try kernels.dispatch(
            $0, "hc_prepare", [Ref(buffer(mixes)), Ref(buffer(scale)), Ref(buffer(base)), Ref(pre), Ref(post), Ref(comb)],
            ints: [Int32(rows), 20], floats: [1e-6], grid: MTLSize(width: (rows * 16 + 63) / 64, height: 1, depth: 1),
            group: MTLSize(width: 64, height: 1, depth: 1))
    }
    var ePre: [Float] = []
    var ePost: [Float] = []
    var eComb: [Float] = []
    for r in 0..<rows {
        let m = Array(mixes[(r * 24)..<((r + 1) * 24)])
        for h in 0..<4 {
            ePre.append(1 / (1 + exp(-(m[h] * scale[0] + base[h]))) + 1e-6)
            ePost.append(2 / (1 + exp(-(m[4 + h] * scale[1] + base[4 + h]))))
        }
        var c = (0..<16).map { m[8 + $0] * scale[2] + base[8 + $0] }
        for src in 0..<4 {
            let mx = (0..<4).map { c[$0 + 4 * src] }.max()!
            let e = (0..<4).map { exp(c[$0 + 4 * src] - mx) }
            let s = e.reduce(0, +)
            for d in 0..<4 { c[d + 4 * src] = e[d] / s + 1e-6 }
        }
        for it in 0..<20 {
            if it > 0 {
                for src in 0..<4 {
                    let s = (0..<4).map { c[$0 + 4 * src] }.reduce(0, +) + 1e-6
                    for d in 0..<4 { c[d + 4 * src] /= s }
                }
            }
            for d in 0..<4 {
                let s = (0..<4).map { c[d + 4 * $0] }.reduce(0, +) + 1e-6
                for src in 0..<4 { c[d + 4 * src] /= s }
            }
        }
        eComb += c
    }
    try close(read(pre, rows * 4), ePre, "hc pre")
    try close(read(post, rows * 4), ePost, "hc post")
    try close(read(comb, rows * 16), eComb, "hc comb", tolerance: 1e-3)
    let hn = 96
    let streams = random(rows * 4 * hn)
    let yIn = random(rows * hn)
    let collapsed = buffer([Float](repeating: 0, count: rows * hn))
    let next = buffer([Float](repeating: 0, count: rows * 4 * hn))
    try run { e in
        try kernels.dispatch(
            e, "hc_collapse", [Ref(buffer(streams)), Ref(pre), Ref(collapsed)], ints: [Int32(hn), Int32(rows)],
            grid: MTLSize(width: 1, height: rows, depth: 1), group: MTLSize(width: 128, height: 1, depth: 1))
        try kernels.dispatch(
            e, "hc_expand", [Ref(buffer(yIn)), Ref(buffer(streams)), Ref(post), Ref(comb), Ref(next)], ints: [Int32(hn), Int32(rows)],
            grid: MTLSize(width: 1, height: rows, depth: 1), group: MTLSize(width: 128, height: 1, depth: 1))
    }
    let gotPre = read(pre, rows * 4)
    let gotPost = read(post, rows * 4)
    let gotComb = read(comb, rows * 16)
    var eCollapsed: [Float] = []
    var eNext: [Float] = []
    for r in 0..<rows {
        for i in 0..<hn { eCollapsed.append((0..<4).map { streams[r * 4 * hn + $0 * hn + i] * gotPre[r * 4 + $0] }.reduce(0, +)) }
        for d in 0..<4 {
            for i in 0..<hn {
                eNext.append(
                    yIn[r * hn + i] * gotPost[r * 4 + d]
                        + (0..<4).map { gotComb[r * 16 + d + 4 * $0] * streams[r * 4 * hn + $0 * hn + i] }.reduce(0, +))
            }
        }
    }
    try close(read(collapsed, rows * hn), eCollapsed, "hc collapse")
    try close(read(next, rows * 4 * hn), eNext, "hc expand")

    // SwiGLU with a clamp at 10.
    let gate = random(rows * 64, scale: 20)
    let up = random(rows * 64, scale: 20)
    let act = buffer([Float](repeating: 0, count: rows * 64))
    try run {
        try kernels.dispatch(
            $0, "swiglu", [Ref(buffer(gate)), Ref(buffer(up)), Ref(act)], ints: [64, Int32(rows), 64, 64, 64], floats: [10],
            grid: MTLSize(width: 1, height: rows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    try close(
        read(act, rows * 64),
        zip(gate, up).map { g, u in
            let g = min(g, 10)
            return g / (1 + exp(-g)) * min(max(u, -10), 10)
        }, "swiglu")

    // KDA: conv (state advanced), gate with a lower bound, and the recurrence over five tokens of two heads.
    let heads = 2
    let dim = 128
    let channels = heads * dim
    let tokens = 5
    let raw = random(tokens * channels)
    let convW = random(channels * 4, scale: 0.5)
    let state0 = random(3 * channels)
    let convState = buffer(state0)
    let convOut = buffer([Float](repeating: 0, count: tokens * channels))
    try run {
        try kernels.dispatch(
            $0, "kda_conv", [Ref(buffer(raw)), Ref(convState), Ref(buffer(convW)), Ref(convOut)],
            ints: [Int32(channels), Int32(tokens), Int32(channels)],
            grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: channels, height: 1, depth: 1))
    }
    var eConv = [Float](repeating: 0, count: tokens * channels)
    var eState = state0
    for c in 0..<channels {
        var s = [state0[c], state0[channels + c], state0[2 * channels + c]]
        for t in 0..<tokens {
            let v = raw[t * channels + c]
            let y = convW[c * 4] * s[0] + convW[c * 4 + 1] * s[1] + convW[c * 4 + 2] * s[2] + convW[c * 4 + 3] * v
            eConv[t * channels + c] = y / (1 + exp(-y))
            s = [s[1], s[2], v]
        }
        eState[c] = s[0]
        eState[channels + c] = s[1]
        eState[2 * channels + c] = s[2]
    }
    try close(read(convOut, tokens * channels), eConv, "kda conv")
    try close(read(convState, 3 * channels), eState, "kda conv state")
    let f = random(tokens * channels)
    let dt = random(channels, scale: 0.3)
    let aLog = [1.2, 0.4].map(Float.init)
    let fb = buffer(f)
    try run {
        try kernels.dispatch(
            $0, "kda_gate", [Ref(fb), Ref(buffer(dt)), Ref(buffer(aLog))], ints: [Int32(heads), Int32(dim), Int32(tokens), 1], floats: [-5],
            grid: MTLSize(width: 1, height: tokens, depth: 1), group: MTLSize(width: channels, height: 1, depth: 1))
    }
    let g = read(fb, tokens * channels)
    try close(
        g, (0..<(tokens * channels)).map { i in -5 / (1 + exp(-(f[i] + dt[i % channels]) * exp(aLog[(i % channels) / dim]))) }, "kda gate")
    var q = random(tokens * channels)
    var k = random(tokens * channels)
    for t in 0..<tokens * heads {  // unit-length heads, as after l2_norm_heads
        for v in [0, 1] {
            var row = v == 0 ? Array(q[(t * dim)..<((t + 1) * dim)]) : Array(k[(t * dim)..<((t + 1) * dim)])
            let s = 1 / row.map { $0 * $0 }.reduce(0, +).squareRoot()
            row = row.map { $0 * s }
            if v == 0 {
                q.replaceSubrange((t * dim)..<((t + 1) * dim), with: row)
            } else {
                k.replaceSubrange((t * dim)..<((t + 1) * dim), with: row)
            }
        }
    }
    let v = random(tokens * channels)
    let beta = random(tokens * heads).map { 1 / (1 + exp(-$0)) }
    let s0 = random(heads * dim * dim, scale: 0.1)
    let sb = buffer(s0)
    let ob = buffer([Float](repeating: 0, count: tokens * channels))
    let scaleQ = 1 / Float(dim).squareRoot()
    try run {
        try kernels.dispatch(
            $0, "kda_scan", [Ref(buffer(q)), Ref(buffer(k)), Ref(buffer(v)), Ref(buffer(g)), Ref(buffer(beta)), Ref(sb), Ref(ob)],
            ints: [Int32(heads), Int32(tokens), Int32(channels), Int32(channels), Int32(channels)], floats: [scaleQ],
            grid: MTLSize(width: heads * 4, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }
    var eOut = [Float](repeating: 0, count: tokens * channels)
    var S = s0  // S[h][j * D + i] = S[i][j]
    for t in 0..<tokens {
        for h in 0..<heads {
            let base = t * channels + h * dim
            let st = h * dim * dim
            for j in 0..<dim { for i in 0..<dim { S[st + j * dim + i] *= exp(g[base + i]) } }
            var delta = [Float](repeating: 0, count: dim)
            for j in 0..<dim {
                var mem: Float = 0
                for i in 0..<dim { mem += S[st + j * dim + i] * k[base + i] }
                delta[j] = (v[base + j] - mem) * beta[t * heads + h]
            }
            for j in 0..<dim {
                var o: Float = 0
                for i in 0..<dim {
                    S[st + j * dim + i] += k[base + i] * delta[j]
                    o += S[st + j * dim + i] * q[base + i]
                }
                eOut[base + j] = o * scaleQ
            }
        }
    }
    try close(read(ob, tokens * channels), eOut, "kda scan output", tolerance: 1e-3)
    try close(read(sb, heads * dim * dim), S, "kda scan state", tolerance: 1e-3)

    // Latent attention: 3 token rows, 4 heads, latent 64, 6 selection slots (some -1), one sink row set.
    let aHeads = 4
    let latent = 64
    let cached = 9
    let slots = 6
    let aRows = 3
    let qa = random(aRows * aHeads * latent)
    let kv = random(cached * latent)
    let sel: [Int32] = [0, 1, 2, -1, -1, -1, 3, 0, 5, 7, -1, 2, 8, 6, 4, 2, 0, 1]
    let scores = buffer([Float](repeating: 0, count: aRows * aHeads * slots))
    let out = buffer([Float](repeating: 0, count: aRows * aHeads * latent))
    // Rows below `split` come from one buffer, the rest from a second (DeepSeek's window ring and compressed cache);
    // each head has a sink logit.
    let selB = ints(sel)
    let scaleA: Float = 0.125
    let split = 4
    let sinks = random(aHeads, scale: 2)
    let kvA = buffer(Array(kv[0..<(split * latent)]))
    let kvB = buffer(Array(kv[(split * latent)...]))
    try run { e in
        try kernels.dispatch(
            e, "attend_scores", [Ref(buffer(qa)), Ref(kvA), Ref(kvB), Ref(selB), Ref(scores)],
            ints: [Int32(aHeads), Int32(latent), Int32(slots), Int32(aHeads * latent), Int32(latent), 0, 0, Int32(split)], floats: [scaleA],
            grid: MTLSize(width: 1, height: aRows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        try kernels.dispatch(
            e, "softmax_rows", [Ref(scores), Ref(buffer(sinks))], ints: [Int32(slots), 1, Int32(aHeads)],
            grid: MTLSize(width: aRows * aHeads, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
        try kernels.dispatch(
            e, "attend_values", [Ref(scores), Ref(kvA), Ref(kvB), Ref(selB), Ref(out)],
            ints: [Int32(aHeads), Int32(latent), Int32(slots), 0, 0, Int32(aHeads * latent), Int32(latent), Int32(split)], floats: [0],
            grid: MTLSize(width: 1, height: aRows, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    var eAttn: [Float] = []
    for t in 0..<aRows {
        for h in 0..<aHeads {
            let qv = Array(qa[((t * aHeads + h) * latent)..<((t * aHeads + h + 1) * latent)])
            let chosen = (0..<slots).map { sel[t * slots + $0] }.filter { $0 >= 0 }.map(Int.init)
            let logits = chosen.map { r in (0..<latent).map { qv[$0] * kv[r * latent + $0] }.reduce(0, +) * scaleA }
            let mx = max(logits.max()!, sinks[h])
            let e = logits.map { exp($0 - mx) }
            let sum = e.reduce(0, +) + exp(sinks[h] - mx)
            eAttn += (0..<latent).map { c in zip(chosen, e).map { kv[$0 * latent + c] * $1 / sum }.reduce(0, +) }
        }
    }
    try close(read(out, aRows * aHeads * latent), eAttn, "latent attention")
    // The heads-parallel kernels (decode) give the same bits as the row kernels, here and at real sizes (64 heads,
    // latent 512, 100 slots with gaps, two KV buffers).
    for (heads, dim, slotCount, tokens, cachedRows, gap) in [(aHeads, latent, slots, aRows, cached, split), (64, 512, 100, 2, 300, 120)] {
        let q = buffer(random(tokens * heads * dim))
        let kvAll = random(cachedRows * dim)
        let kv1 = buffer(Array(kvAll[0..<(gap * dim)]))
        let kv2 = buffer(Array(kvAll[(gap * dim)...]))
        let picks = ints((0..<(tokens * slotCount)).map { i in i % 7 == 3 ? -1 : Int32((i * 37) % cachedRows) })
        let pBuffer = buffer(random(tokens * heads * slotCount))
        var results: [[Float]] = []
        for variant in ["", "_heads"] {
            let s = buffer([Float](repeating: 0, count: tokens * heads * slotCount))
            let o = buffer([Float](repeating: 0, count: tokens * heads * dim))
            let depth = variant.isEmpty ? 1 : heads
            try run { e in
                try kernels.dispatch(
                    e, "attend_scores" + variant, [Ref(q), Ref(kv1), Ref(kv2), Ref(picks), Ref(s)],
                    ints: [Int32(heads), Int32(dim), Int32(slotCount), Int32(heads * dim), Int32(dim), 0, 0, Int32(gap)], floats: [0.0625],
                    grid: MTLSize(width: (slotCount + 63) / 64, height: tokens, depth: depth),
                    group: MTLSize(width: 64, height: 1, depth: 1))
                try kernels.dispatch(
                    e, "attend_values" + variant, [Ref(pBuffer), Ref(kv1), Ref(kv2), Ref(picks), Ref(o)],
                    ints: [Int32(heads), Int32(dim), Int32(slotCount), 0, 0, Int32(heads * dim), Int32(dim), Int32(gap)], floats: [0],
                    grid: MTLSize(width: (dim + 63) / 64, height: tokens, depth: depth), group: MTLSize(width: 64, height: 1, depth: 1))
            }
            results.append(read(s, tokens * heads * slotCount) + read(o, tokens * heads * dim))
        }
        try require(
            zip(results[0], results[1]).allSatisfy { $0.bitPattern == $1.bitPattern || ($0.isNaN && $1.isNaN) },
            "heads-parallel attention differs from the row kernels (\(heads) heads, latent \(dim))")
    }

    // k-pool and the indexer's selection: 10 rows of pools of 4, width 8.
    let iDim = 8
    let pool = 4
    let keysRows = 10
    let poolsN = keysRows / pool
    let key = random(keysRows * iDim)
    let gateK = random(keysRows * iDim)
    let ape = random(iDim * pool)
    let pooled = buffer([Float](repeating: 0, count: (poolsN + 1) * iDim))
    try run {
        try kernels.dispatch(
            $0, "kpool_pool", [Ref(buffer(key)), Ref(buffer(gateK)), Ref(buffer(ape)), Ref(pooled)],
            ints: [Int32(iDim), 0, Int32(poolsN), Int32(pool)],
            grid: MTLSize(width: 1, height: poolsN, depth: 1), group: MTLSize(width: iDim, height: 1, depth: 1))
    }
    var ePooled: [Float] = []
    for p in 0..<poolsN {
        for c in 0..<iDim {
            let l = (0..<pool).map { gateK[(p * pool + $0) * iDim + c] + ape[c + iDim * $0] }
            let mx = l.max()!
            let e = l.map { exp($0 - mx) }
            let s = e.reduce(0, +)
            ePooled.append((0..<pool).map { e[$0] / s * key[(p * pool + $0) * iDim + c] }.reduce(0, +))
        }
    }
    try close(read(pooled, poolsN * iDim), ePooled, "kpool")
    // Selection over 37 pools with ties, top 8, for tokens at positions 3 (one pool, no tail), 12 (three pools and a tail), 150 (37, top 8).
    let pools = 37
    let top = 8
    let selected = top * pool + pool - 1
    var sc = random(3 * pools).map { ($0 * 4).rounded() }  // integer scores: ties at the threshold
    for t in 0..<3 { for p in 0..<pools where (p + 1) * pool - 1 > [3, 12, 150][t] { sc[t * pools + p] = -.infinity } }
    let selOut = device.makeBuffer(length: 3 * selected * 4, options: .storageModeShared)!
    for (t, position) in [3, 12, 150].enumerated() {
        try run {
            try kernels.dispatch(
                $0, "index_select", [Ref(buffer(Array(sc[(t * pools)..<((t + 1) * pools)]))), Ref(selOut, t * selected * 4)],
                ints: [Int32(pools), Int32(pool), Int32(top), Int32(selected), Int32(position), 1, Int32(pool), 0, Int32(selected)],
                grid: one, group: MTLSize(width: 64, height: 1, depth: 1))
        }
    }
    let got = Array(UnsafeBufferPointer(start: selOut.contents().bindMemory(to: Int32.self, capacity: 3 * selected), count: 3 * selected))
    for (t, position) in [3, 12, 150].enumerated() {
        let visible = min(pools, (position + 1) / pool)
        let row = Array(sc[(t * pools)..<((t + 1) * pools)])
        let ranked = (0..<visible).sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }.prefix(min(visible, top)).sorted()
        var want = ranked.flatMap { p in (0..<pool).map { Int32(p * pool + $0) } }
        if visible * pool <= position { want += (visible * pool...position).map(Int32.init).prefix(pool - 1) }
        want += [Int32](repeating: -1, count: selected - want.count)
        try require(
            Array(got[(t * selected)..<((t + 1) * selected)]) == want,
            "index_select at position \(position): \(got[(t * selected)..<((t + 1) * selected)]) for \(want)")
    }
    // DeepSeek's compressed rows: width 1, offset by the ring size, written at a column offset of a wider row.
    let dsRow = 12
    let ring = 5
    let dsTop = 3
    let dsOut = device.makeBuffer(length: 2 * dsRow * 4, options: .storageModeShared)!
    let dsScores: [Float] = [0.5, 2, -1, 3, 0.1, 9]  // six compressed entries of ratio 2; position 9 sees five
    try run {
        try kernels.dispatch(
            $0, "index_select", [Ref(buffer(dsScores)), Ref(dsOut, 4 * 4)],
            ints: [6, 2, Int32(dsTop), Int32(dsTop), 9, 0, 1, Int32(ring), Int32(dsRow)], grid: one,
            group: MTLSize(width: 32, height: 1, depth: 1))
    }
    try run {
        try kernels.dispatch(
            $0, "ring_select", [Ref(dsOut)], ints: [4, Int32(ring), Int32(dsRow), 9],
            grid: one, group: MTLSize(width: 4, height: 1, depth: 1))
    }
    let dsGot = Array(UnsafeBufferPointer(start: dsOut.contents().bindMemory(to: Int32.self, capacity: dsRow), count: 7))
    // Window of 4 up to position 9 in a ring of 5: positions 6..9 -> slots 1, 2, 3, 4; then entries 1, 3, 0 -> 5 + j in entry order.
    try require(dsGot == [1, 2, 3, 4, 5, 6, 8], "DeepSeek selection \(dsGot)")

    // Tail RoPE: the last 4 of 6 values as two pairs at position 3, forward then inverse.
    let ropeX = random(6)
    let freqs: [Float] = [0.5, 0.01]
    let ropeB = buffer(ropeX)
    try run {
        try kernels.dispatch(
            $0, "rope_tail", [Ref(ropeB), Ref(buffer(freqs))], ints: [6, 4, 1, 3, 1, 0], grid: one,
            group: MTLSize(width: 2, height: 1, depth: 1))
    }
    var ropeWant = ropeX
    for i in 0..<2 {
        let a = 3 * freqs[i]
        let x0 = ropeX[2 + 2 * i]
        let x1 = ropeX[3 + 2 * i]
        ropeWant[2 + 2 * i] = x0 * cos(a) - x1 * sin(a)
        ropeWant[3 + 2 * i] = x0 * sin(a) + x1 * cos(a)
    }
    try close(read(ropeB, 6), ropeWant, "rope_tail")
    try run {
        try kernels.dispatch(
            $0, "rope_tail", [Ref(ropeB), Ref(buffer(freqs))], ints: [6, 4, 1, 3, 1, 1], grid: one,
            group: MTLSize(width: 2, height: 1, depth: 1))
    }
    try close(read(ropeB, 6), ropeX, "rope_tail inverse")

    // Engram gate: one row, two streams of 64.
    let en = 64
    let eh = random(2 * en)
    let ekv = random(3 * en)
    let ew = random(2 * en, scale: 2)
    let ehB = buffer(eh)
    try run {
        try kernels.dispatch(
            $0, "engram_gate", [Ref(ehB), Ref(buffer(ekv)), Ref(buffer(ew))], ints: [Int32(en), 2], floats: [1e-20],
            grid: MTLSize(width: 2, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    var engramWant = eh
    for st in 0..<2 {
        let hrow = Array(eh[(st * en)..<((st + 1) * en)])
        let krow = Array(ekv[(st * en)..<((st + 1) * en)])
        let value = Array(ekv[(2 * en)...])
        let hh = hrow.map { $0 * $0 }.reduce(0, +) / Float(en)
        let kk = krow.map { $0 * $0 }.reduce(0, +) / Float(en)
        let d = (0..<en).map { hrow[$0] * ew[st * en + $0] * krow[$0] }.reduce(0, +)
        let dot = d / (hh + 1e-20).squareRoot() / (kk + 1e-20).squareRoot() / Float(en).squareRoot()
        let gate = 1 / (1 + exp(-(dot < 0 ? -max(abs(dot), 1e-6).squareRoot() : max(abs(dot), 1e-6).squareRoot())))
        for i in 0..<en { engramWant[st * en + i] = hrow[i] + gate * value[i] }
    }
    try close(read(ehB, 2 * en), engramWant, "engram gate")

    // Candidate blocks: 20 entries of ratio 1 in blocks of 4 at position 17 (18 visible): top 2 blocks plus the pinned
    // newest (block 4, partly visible); then the mask.
    let cs: [Float] = [1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 9, 0, 0, 0, 0, 0, 0]
    let keepB = buffer([Float](repeating: -1, count: 5))
    let csB = buffer(cs)
    try run { e in
        try kernels.dispatch(
            e, "candidate_blocks", [Ref(csB), Ref(keepB)], ints: [20, 4, 5, 2, 17, 1, 5], grid: one,
            group: MTLSize(width: 32, height: 1, depth: 1))
        try kernels.dispatch(
            e, "apply_block_mask", [Ref(csB), Ref(keepB)], ints: [20, 4, 5, 5], grid: MTLSize(width: 1, height: 1, depth: 1),
            group: MTLSize(width: 32, height: 1, depth: 1))
    }
    // Block maxima 1, 5, 2, 9, pinned: the pinned block 4 and the best other, block 3 (9).
    try require(read(keepB, 5) == [0, 0, 0, 1, 1], "candidate blocks \(read(keepB, 5))")
    try require(
        read(csB, 20).enumerated().allSatisfy { i, v in (i / 4 == 3 || i / 4 == 4) ? v == cs[i] : v == -.infinity },
        "block mask \(read(csB, 20))")
    // The per-token helpers, against their one-line CPU definitions, and the reductions at real widths (strided loops,
    // several SIMD groups): RMSNorm over 16,384, LayerNorm and the head norms over 128.
    do {
        let wide = 16384
        let wx = random(2 * wide)
        let wy = buffer([Float](repeating: 0, count: 2 * wide))
        let wxb = buffer(wx)
        try run {
            try kernels.dispatch(
                $0, "rms_norm", [Ref(wxb), Ref(wxb), Ref(wy)], ints: [Int32(wide), 0], floats: [1e-5],
                grid: MTLSize(width: 2, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
        }
        try close(
            read(wy, 2 * wide),
            (0..<2).flatMap { r -> [Float] in
                let row = Array(wx[(r * wide)..<((r + 1) * wide)])
                let s = 1 / (row.map { $0 * $0 }.reduce(0, +) / Float(wide) + 1e-5).squareRoot()
                return row.map { $0 * s }
            }, "rms_norm at 16,384")
        let d = 128
        let hr = 6
        let hx = random(hr * d, scale: 3)
        let hw = random(d)
        let hb = random(d)
        let hg = random(hr * d)
        let ln = buffer([Float](repeating: 0, count: hr * d))
        try run {
            try kernels.dispatch(
                $0, "layer_norm", [Ref(buffer(hx)), Ref(buffer(hw)), Ref(buffer(hb)), Ref(ln)], ints: [Int32(d)], floats: [1e-6],
                grid: MTLSize(width: hr, height: 1, depth: 1), group: MTLSize(width: d, height: 1, depth: 1))
        }
        var lnWant: [Float] = []
        var l2Want: [Float] = []
        var outWant: [Float] = []
        for r in 0..<hr {
            let row = Array(hx[(r * d)..<((r + 1) * d)])
            let mean = row.reduce(0, +) / Float(d)
            let v = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(d)
            let s = 1 / (v + 1e-6).squareRoot()
            lnWant += (0..<d).map { (row[$0] - mean) * s * hw[$0] + hb[$0] }
            let norm = max(row.map { $0 * $0 }.reduce(0, +).squareRoot(), 1e-6)
            l2Want += row.map { $0 / norm }
            let rs = 1 / (row.map { $0 * $0 }.reduce(0, +) / Float(d) + 1e-5).squareRoot()
            outWant += (0..<d).map { row[$0] * rs * hw[$0] / (1 + exp(-hg[r * d + $0])) }
        }
        try close(read(ln, hr * d), lnWant, "layer_norm")
        let l2 = buffer(hx)
        let ko = buffer([Float](repeating: 0, count: hr * d))
        try run { e in
            try kernels.dispatch(
                e, "kda_out", [Ref(l2), Ref(buffer(hw)), Ref(buffer(hg)), Ref(ko)], ints: [Int32(d)], floats: [1e-5],
                grid: MTLSize(width: hr, height: 1, depth: 1), group: MTLSize(width: d, height: 1, depth: 1))
            try kernels.dispatch(
                e, "l2_norm_heads", [Ref(l2)], ints: [Int32(d)], floats: [1e-6],
                grid: MTLSize(width: hr, height: 1, depth: 1), group: MTLSize(width: d, height: 1, depth: 1))
        }
        try close(read(ko, hr * d), outWant, "kda_out")
        try close(read(l2, hr * d), l2Want, "l2_norm_heads")

        // Streams: mean of four, broadcast to four; rows gathered, then scattered back weighted; copies; fills; scalars.
        let sn = 300
        let sr = 3
        let streams = random(sr * 4 * sn)
        let flat = random(sr * sn)
        let mean = buffer([Float](repeating: 0, count: sr * sn))
        let wideB = buffer([Float](repeating: 0, count: sr * 4 * sn))
        let (sg, sgroup) = (MTLSize(width: (sn + 255) / 256, height: sr, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
        try run { e in
            try kernels.dispatch(e, "hc_mean", [Ref(buffer(streams)), Ref(mean)], ints: [Int32(sn), Int32(sr)], grid: sg, group: sgroup)
            try kernels.dispatch(
                e, "hc_broadcast", [Ref(buffer(flat)), Ref(wideB)], ints: [Int32(sn), Int32(sr)], grid: sg, group: sgroup)
        }
        try close(
            read(mean, sr * sn),
            (0..<(sr * sn)).map { i in (0..<4).map { streams[(i / sn) * 4 * sn + $0 * sn + i % sn] }.reduce(0, +) / 4 }, "hc_mean")
        try close(read(wideB, sr * 4 * sn), (0..<(sr * 4 * sn)).map { i in flat[(i / (4 * sn)) * sn + i % sn] }, "hc_broadcast")
        let table = random(5 * sn)
        let picks: [Int32] = [4, 0, 2]
        let weightsOf: [Float] = [0.5, -2, 1.25]
        let gathered = buffer([Float](repeating: 0, count: 3 * sn))
        let target = random(5 * sn)
        let targetB = buffer(target)
        try run { e in
            try kernels.dispatch(
                e, "gather_rows", [Ref(buffer(table)), Ref(ints(picks)), Ref(gathered)], ints: [Int32(sn), 3], grid: sg, group: sgroup)
            try kernels.dispatch(
                e, "scatter_add_rows", [Ref(gathered), Ref(ints(picks)), Ref(buffer(weightsOf)), Ref(targetB)], ints: [Int32(sn), 3],
                grid: sg, group: sgroup)
        }
        let gatherWant = picks.flatMap { Array(table[(Int($0) * sn)..<((Int($0) + 1) * sn)]) }
        try close(read(gathered, 3 * sn), gatherWant, "gather_rows")
        var scatterWant = target
        for (r, row) in picks.enumerated() { for i in 0..<sn { scatterWant[Int(row) * sn + i] += weightsOf[r] * gatherWant[r * sn + i] } }
        try close(read(targetB, 5 * sn), scatterWant, "scatter_add_rows")
        let copied = buffer([Float](repeating: 0, count: sr * sn))
        let intsIn: [Int32] = (0..<24).map { Int32($0 * 3 - 7) }
        let intsOut = ints([Int32](repeating: 99, count: 3 * 6))
        let filled = ints([Int32](repeating: 0, count: 37))
        let squashed = buffer(flat)
        let scaled = buffer(flat)
        try run { e in
            try kernels.dispatch(e, "copy_rows", [Ref(buffer(flat)), Ref(copied)], ints: [Int32(sn), Int32(sr)], grid: sg, group: sgroup)
            try kernels.dispatch(
                e, "copy_int_rows", [Ref(ints(intsIn)), Ref(intsOut)], ints: [4, 3, 8, 6], grid: one,
                group: MTLSize(width: 32, height: 3, depth: 1))
            try kernels.dispatch(
                e, "fill_int", [Ref(filled)], ints: [37, -1], grid: MTLSize(width: 1, height: 1, depth: 1),
                group: MTLSize(width: 64, height: 1, depth: 1))
            try kernels.dispatch(
                e, "sigmoid_inplace", [Ref(squashed)], ints: [Int32(sr * sn)],
                grid: MTLSize(width: (sr * sn + 255) / 256, height: 1, depth: 1), group: sgroup)
            try kernels.dispatch(
                e, "scale_inplace", [Ref(scaled)], ints: [Int32(sr * sn)], floats: [-0.25],
                grid: MTLSize(width: (sr * sn + 255) / 256, height: 1, depth: 1), group: sgroup)
        }
        try close(read(copied, sr * sn), flat, "copy_rows")
        let intsGot = Array(UnsafeBufferPointer(start: intsOut.contents().bindMemory(to: Int32.self, capacity: 18), count: 18))
        try require(intsGot == (0..<18).map { $0 % 6 < 4 ? intsIn[($0 / 6) * 8 + $0 % 6] : 99 }, "copy_int_rows \(intsGot)")
        try require(
            Array(UnsafeBufferPointer(start: filled.contents().bindMemory(to: Int32.self, capacity: 37), count: 37)).allSatisfy {
                $0 == -1
            }, "fill_int")
        try close(read(squashed, sr * sn), flat.map { 1 / (1 + exp(-$0)) }, "sigmoid_inplace")
        try close(read(scaled, sr * sn), flat.map { $0 * -0.25 }, "scale_inplace")

        // The window of the last 5 rows for tokens at positions 2, 3 and 4 (-1 before the start); a 3D permute.
        let window = ints([Int32](repeating: 99, count: 15))
        try run {
            try kernels.dispatch(
                $0, "window_select", [Ref(window)], ints: [5, 2], grid: MTLSize(width: 1, height: 3, depth: 1),
                group: MTLSize(width: 64, height: 1, depth: 1))
        }
        let windowGot = Array(UnsafeBufferPointer(start: window.contents().bindMemory(to: Int32.self, capacity: 15), count: 15))
        try require(windowGot == [-1, -1, 0, 1, 2, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4], "window_select \(windowGot)")
        let (pa, pb, pn) = (3, 5, 70)
        let px = random(pa * pb * pn)
        let py = buffer([Float](repeating: 0, count: pa * pb * pn))
        try run {
            try kernels.dispatch(
                $0, "permute3", [Ref(buffer(px)), Ref(py)], ints: [Int32(pa), Int32(pb), Int32(pn)],
                grid: MTLSize(width: (pn + 63) / 64, height: pb, depth: pa), group: MTLSize(width: 64, height: 1, depth: 1))
        }
        try close(
            read(py, pa * pb * pn),
            (0..<(pa * pb * pn)).map { i in
                let (y, z, x) = (i / (pa * pn), (i / pn) % pa, i % pn)
                return px[(z * pb + y) * pn + x]
            }, "permute3")
    }

    // More blocks than a threadgroup array holds (5,000, past 32K tokens of ratio 1): the pinned newest and the top two.
    var many = [Float](repeating: 0, count: 5000)
    many[100] = 7
    many[3000] = 8
    let manyKeep = buffer([Float](repeating: -1, count: 5000))
    let manyB = buffer(many)
    try run { e in
        try kernels.dispatch(
            e, "candidate_blocks", [Ref(manyB), Ref(manyKeep)], ints: [5000, 1, 5000, 3, 4999, 1, 5000], grid: one,
            group: MTLSize(width: 256, height: 1, depth: 1))
    }
    let kept = read(manyKeep, 5000).enumerated().filter { $0.element == 1 }.map(\.offset)
    try require(
        kept == [100, 3000, 4999] && read(manyKeep, 5000).allSatisfy { $0 == 0 || $0 == 1 }, "5,000 candidate blocks kept \(kept.prefix(8))"
    )

    // The indexer's scoring and selection as one dispatch over a slice of rows (the engines' slices) give the bits of
    // one dispatch a row (rows 0 ..< 37 at positions 300 ...; 330 entries of ratio 1 in blocks of 8, keep rows 50
    // apart): index_scores, candidate_blocks (top 12 blocks), apply_block_mask, index_select (top 40).
    do {
        let (n, heads, dim, entries, at, block, blocks, stride, top) = (37, 4, 16, 330, 300, 8, 42, 50, 40)
        let q = buffer(random(n * heads * dim))
        let keys = buffer(random(entries * dim))
        let w = buffer(random(n * heads))
        let scoresAll = buffer([Float](repeating: 0, count: n * entries))
        let scoresOne = buffer([Float](repeating: 0, count: n * entries))
        let keepAll = buffer([Float](repeating: -1, count: n * stride))
        let keepOne = buffer([Float](repeating: -1, count: n * stride))
        let maskedAll = buffer([Float](repeating: 0, count: n * entries))
        let maskedOne = buffer([Float](repeating: 0, count: n * entries))
        let selAll = ints([Int32](repeating: 7, count: n * top))
        let selOne = ints([Int32](repeating: 7, count: n * top))
        func selection(_ scores: MTLBuffer, _ out: MTLBuffer, row: Int, rows: Int) throws -> (MTLComputeCommandEncoder) throws -> Void {
            { e in
                try kernels.dispatch(
                    e, "index_select", [Ref(scores, row * entries * 4), Ref(out, row * top * 4)],
                    ints: [Int32(entries), 1, Int32(top), Int32(top), Int32(at + row), 0, 1, 0, Int32(top)],
                    grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        try run { e in
            try kernels.dispatch(
                e, "index_scores", [Ref(q), Ref(keys), Ref(w), Ref(scoresAll)],
                ints: [Int32(heads), Int32(dim), Int32(entries), 1, Int32(at)],
                grid: MTLSize(width: (entries + 63) / 64, height: n, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
            try kernels.dispatch(
                e, "candidate_blocks", [Ref(scoresAll), Ref(keepAll)],
                ints: [Int32(entries), Int32(block), Int32(blocks), 12, Int32(at), 1, Int32(stride)],
                grid: MTLSize(width: n, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        }
        memcpy(maskedAll.contents(), scoresAll.contents(), n * entries * 4)
        try run { e in
            try kernels.dispatch(
                e, "apply_block_mask", [Ref(maskedAll), Ref(keepAll)], ints: [Int32(entries), Int32(block), Int32(blocks), Int32(stride)],
                grid: MTLSize(width: (entries + 255) / 256, height: n, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
            try selection(maskedAll, selAll, row: 0, rows: n)(e)
        }
        for t in 0..<n {
            try run { e in
                try kernels.dispatch(
                    e, "index_scores", [Ref(q, t * heads * dim * 4), Ref(keys), Ref(w, t * heads * 4), Ref(scoresOne, t * entries * 4)],
                    ints: [Int32(heads), Int32(dim), Int32(entries), 1, Int32(at + t)],
                    grid: MTLSize(width: (entries + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
                try kernels.dispatch(
                    e, "candidate_blocks", [Ref(scoresOne, t * entries * 4), Ref(keepOne, t * stride * 4)],
                    ints: [Int32(entries), Int32(block), Int32(blocks), 12, Int32(at + t), 1, Int32(stride)], grid: one,
                    group: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        memcpy(maskedOne.contents(), scoresOne.contents(), n * entries * 4)
        for t in 0..<n {
            try run { e in
                try kernels.dispatch(
                    e, "apply_block_mask", [Ref(maskedOne, t * entries * 4), Ref(keepOne, t * stride * 4)],
                    ints: [Int32(entries), Int32(block), Int32(blocks), Int32(stride)],
                    grid: MTLSize(width: (entries + 255) / 256, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
                try selection(maskedOne, selOne, row: t, rows: 1)(e)
            }
        }
        func bits(_ b: MTLBuffer, _ count: Int) -> [UInt32] {
            Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: count), count: count))
        }
        try require(bits(scoresAll, n * entries) == bits(scoresOne, n * entries), "index_scores over 37 rows differs from a row at a time")
        try require(bits(keepAll, n * stride) == bits(keepOne, n * stride), "candidate_blocks over 37 rows differs from a row at a time")
        let keptBlocks = (0..<n).map { t in read(keepAll, n * stride)[(t * stride)..<(t * stride + blocks)].filter { $0 == 1 }.count }
        try require(
            keptBlocks.allSatisfy { $0 == 12 }
                && read(keepAll, n * stride).enumerated().allSatisfy { $0.offset % stride < blocks || $0.element == -1 },
            "candidate_blocks kept \(keptBlocks) or wrote past a row's blocks")
        try require(
            bits(maskedAll, n * entries) == bits(maskedOne, n * entries), "apply_block_mask over 37 rows differs from a row at a time")
        try require(
            bits(selAll, n * top) == bits(selOne, n * top) && bits(selAll, n * top).allSatisfy { Int32(bitPattern: $0) >= 0 },
            "index_select over 37 rows differs from a row at a time")
    }
}

/// The family engines' routing: the trained top-k by probability plus bias with normalised, scaled weights, and each
/// dial knob (fewer experts, tail skip always or on misses only, substitution of resident experts for misses).
func familyRoutingDial() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family routing: " + what) } }
    let probs: [Float] = [0.9, 0.1, 0.5, 0.05, 0.7, 0.3, 0.02, 0.6]
    let bias: [Float] = [0, 0.5, 0, 0, 0, 0, 0, -0.5]
    let clamp: (Float) -> Float = { max($0, 6.103515625e-5) }
    func ids(_ r: [(expert: Int, weight: Float)]) -> [Int] { r.map(\.expert) }
    // Scores 0.9, 0.6, 0.5, 0.05, 0.7, 0.3, 0.02, 0.1: top 4 = 0, 4, 1, 2.
    let trained = FamilyRouting.select(probs: probs, bias: bias, k: 4, dial: .exact, resident: nil, normalize: true, scale: 2.5, sum: clamp)
    try require(ids(trained) == [0, 4, 1, 2], "trained top 4 \(ids(trained))")
    let mass: Float = 0.9 + 0.7 + 0.1 + 0.5
    try require(
        zip(trained.map(\.weight), [0.9, 0.7, 0.1, 0.5].map { $0 / mass * 2.5 }).allSatisfy { abs($0 - $1) < 1e-6 },
        "trained weights \(trained)")
    let capped = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(expertsPerToken: 2), resident: nil, normalize: true, scale: 1, sum: clamp)
    try require(ids(capped) == [0, 4] && abs(capped[0].weight - 0.9 / 1.6) < 1e-6, "two experts \(capped)")
    let skipped = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(tailSkip: 0.1), resident: nil, normalize: true, scale: 1, sum: clamp)
    try require(ids(skipped) == [0, 4, 2], "tail skip drops 1 (weight 0.045) \(ids(skipped))")
    let missesOnly = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true), resident: [1],
        normalize: true, scale: 1, sum: clamp)
    try require(ids(missesOnly) == [0, 4, 1, 2], "a resident light expert stays \(ids(missesOnly))")
    let substituted = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(lateMiss: .substitute), resident: [0, 1, 7, 5],
        normalize: true, scale: 1, sum: clamp)
    // Misses 4 and 2 take the best resident spares by score: 5 (0.3), then 7 (0.1).
    try require(ids(substituted) == [0, 5, 1, 7], "substitution \(ids(substituted))")
    let one = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(tailSkip: 0.99), resident: nil, normalize: true, scale: 1, sum: clamp)
    try require(ids(one) == [0] && abs(one[0].weight - 1) < 1e-6, "the heaviest always stays \(one)")
    // Late-miss skip: the misses 4 and 2 are dropped; the heaviest (0) stays even when it is a miss.
    let late = FamilyRouting.select(
        probs: probs, bias: bias, k: 4, dial: QualityDial(lateMiss: .skip), resident: [1, 2],
        normalize: true, scale: 1, sum: clamp)
    try require(ids(late) == [0, 1, 2], "late-miss skip \(ids(late))")
    let levels = QualityDial.levels.map(\.dial)
    try require(
        FamilyRouting.select(probs: probs, bias: bias, k: 4, dial: levels[0], resident: [], normalize: true, scale: 2.5, sum: clamp)
            .elementsEqual(trained) { $0.expert == $1.expert && $0.weight == $1.weight }, "exact is the trained rule")
}

/// `FamilyRouting.select`'s partial top-k against the full sort it replaced (copied below): the same experts in the
/// same order and the same weight bits, over many sizes, counts, ties, biases and every dial knob.
func familyRoutingPartialTopK() throws {
    func reference(
        probs: [Float], bias: [Float], k: Int, dial: QualityDial, resident: Set<Int>?, normalize: Bool, scale: Float,
        sum normaliser: (Float) -> Float
    ) -> [(expert: Int, weight: Float)] {
        let n = probs.count
        func score(_ i: Int) -> Float { probs[i] + (bias.isEmpty ? 0 : bias[i]) }
        let order = (0..<n).sorted { score($0) != score($1) ? score($0) > score($1) : $0 < $1 }
        var chosen = Array(order.prefix(max(1, min(k, dial.expertsPerToken ?? k))))
        if dial.lateMiss == .substitute, let resident {
            var taken = Set(chosen)
            let spare = order.filter { resident.contains($0) && !taken.contains($0) }
            var next = 0
            for (slot, expert) in chosen.enumerated() where !resident.contains(expert) && next < spare.count {
                taken.remove(expert)
                chosen[slot] = spare[next]
                taken.insert(spare[next])
                next += 1
            }
        }
        if dial.lateMiss == .skip, let resident, chosen.count > 1 {
            let heaviest = chosen.max { probs[$0] < probs[$1] }!
            chosen = chosen.filter { $0 == heaviest || resident.contains($0) }
        }
        if let tau = dial.tailSkip.map(Float.init), chosen.count > 1 {
            let mass = chosen.reduce(Float(0)) { $0 + probs[$1] }
            let heaviest = chosen.max { probs[$0] < probs[$1] }!
            chosen = chosen.filter { e in
                e == heaviest || probs[e] / mass >= tau || (dial.tailSkipMissesOnly && (resident?.contains(e) ?? true))
            }
        }
        let total = normaliser(chosen.reduce(Float(0)) { $0 + probs[$1] })
        return chosen.map { ($0, (normalize ? probs[$0] / total : probs[$0]) * scale) }
    }
    var random = SystemRandomNumberGenerator()
    let dials =
        [
            QualityDial.exact, QualityDial(expertsPerToken: 2), QualityDial(tailSkip: 0.1),
            QualityDial(tailSkip: 0.1, tailSkipMissesOnly: true),
            QualityDial(lateMiss: .skip), QualityDial(lateMiss: .substitute),
        ] + QualityDial.levels.map(\.dial)
    for trial in 0..<3000 {
        let n = [1, 2, 3, 7, 8, 64, 288, 384][trial % 8]
        let k = [1, 2, 6, 8, 10, 400][trial / 8 % 6]
        // Every third trial draws from eight values (many ties, -0 beside 0 among them); a bias may tie them further.
        let levels: [Float] = [-0.0, 0, 0.125, 0.25, 0.5, 0.5, 0.75, 1]
        let probs = (0..<n).map { _ in trial % 3 == 0 ? levels.randomElement(using: &random)! : Float.random(in: 0..<1, using: &random) }
        let bias =
            trial % 2 == 0
            ? []
            : (0..<n).map { _ in trial % 3 == 0 ? [0, 0.25].randomElement(using: &random)! : Float.random(in: -0.1..<0.1, using: &random) }
        let resident = trial % 5 == 0 ? nil : Set((0..<n).filter { _ in Bool.random(using: &random) })
        let dial = dials[trial % dials.count]
        let normalize = trial % 7 != 0
        let sum: (Float) -> Float = trial % 2 == 0 ? { max($0, 6.103515625e-5) } : { $0 + 1e-20 }
        let want = reference(probs: probs, bias: bias, k: k, dial: dial, resident: resident, normalize: normalize, scale: 2.5, sum: sum)
        let got = FamilyRouting.select(
            probs: probs, bias: bias, k: k, dial: dial, resident: resident, normalize: normalize, scale: 2.5, sum: sum)
        guard got.elementsEqual(want, by: { $0.expert == $1.expert && $0.weight.bitPattern == $1.weight.bitPattern }) else {
            throw EngineError("Family routing: trial \(trial) (\(n) experts, k \(k), \(dial)) chose \(got), the full sort \(want)")
        }
    }
}
