import Foundation
import Metal
import OrigamiEngine

/// Family kernels rewritten for speed give the bits their previous versions gave, on random data of the real shapes
/// (DeepSeek-V4.1-Flash and GLM-5.3-Flash decode and prompt rows), ties, -inf and NaN included. The previous kernels are
/// kept here verbatim (renamed `previous_*`), compiled with `FamilyKernels`' options.
private struct Previous {
    let device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, kernels: FamilyKernels
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device")
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        self.device = device
        self.queue = queue
        library = try device.makeLibrary(source: previousFamilyKernels, options: options)
        kernels = try FamilyKernels(device: device)
    }
    func buffer(_ v: [Float]) -> MTLBuffer { device.makeBuffer(bytes: v, length: max(16, v.count * 4), options: .storageModeShared)! }
    func filled(_ count: Int, _ value: UInt32 = 0x7FC0_0001) -> MTLBuffer {
        buffer([Float](repeating: Float(bitPattern: value), count: count))
    }
    func bits(_ b: MTLBuffer, _ count: Int, at: Int = 0) -> [UInt32] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: at + count) + at, count: count))
    }
    /// One dispatch of the previous kernel `name` (arguments packed as `FamilyKernels.dispatch` packs them).
    func previous(_ name: String, _ buffers: [MTLBuffer], ints: [Int32] = [], floats: [Float] = [], grid: MTLSize, group: MTLSize) throws {
        let state = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(state)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        var words = ints.map { UInt32(bitPattern: $0) } + floats.map(\.bitPattern)
        if words.isEmpty { words = [0] }
        encoder.setBytes(&words, length: words.count * 4, index: buffers.count)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
    }
    /// One dispatch of the engine's kernel `name`.
    func current(_ name: String, _ buffers: [MTLBuffer], ints: [Int32] = [], floats: [Float] = [], grid: MTLSize, group: MTLSize) throws {
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        try kernels.dispatch(encoder, name, buffers.map { FamilyKernels.Ref($0) }, ints: ints, floats: floats, grid: grid, group: group)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
    }
}

private struct Random {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
    /// Uniform in [-1, 1).
    mutating func float() -> Float { Float(next() >> 40) / Float(1 << 23) - 1 }
}

/// index_select (radix select, runs written by every thread) against the previous bisection with one writing thread:
/// DeepSeek's compressed rows (20,480 and 10,240 entries, top 512, offset by the ring) and GLM's pools (width 4 with
/// the tail), with fewer, as many and more entries visible than the top, several rows a dispatch, scores with many
/// ties, -inf runs (masked candidate blocks), all -inf, all equal, and NaNs; 1,024, 256 and 32 threads a row.
func familyIndexSelectMatchesPrevious() throws {
    let k = try Previous()
    var random = Random(state: 0x2545_F491_4F6C_DD1D)
    func scores(_ count: Int, pattern: Int) -> [Float] {
        (0..<count).map { i in
            switch pattern {
            case 0: return random.float()
            case 1: return Float(Int(random.float() * 6)) / 4  // many ties
            case 2: return i / 8 % 3 == 1 ? -Float.infinity : Float(Int(random.float() * 40)) / 8  // masked blocks
            case 3: return -.infinity
            case 4: return 0.5
            default: return i % 97 == 0 ? .nan : i % 89 == 0 ? -.nan : random.float()
            }
        }
    }
    // (entries, pool, top, selected, first position, tail, width, offset, rows)
    let cases: [(Int, Int, Int, Int, Int, Int, Int, Int, Int)] = [
        (20480, 1, 512, 512, 20479, 0, 1, 384, 1), (10240, 2, 512, 512, 20479, 0, 1, 384, 1), (20480, 1, 512, 512, 19000, 0, 1, 384, 3),
        (700, 1, 512, 512, 300, 0, 1, 384, 2), (512, 1, 512, 512, 511, 0, 1, 384, 1), (513, 1, 512, 512, 512, 0, 1, 384, 1),
        (5000, 4, 128, 515, 19999, 1, 4, 0, 1), (5000, 4, 128, 515, 19990, 1, 4, 0, 4), (40, 4, 128, 515, 150, 1, 4, 0, 3),
    ]
    var checked = 0
    for (entries, pool, top, selected, first, tail, width, offset, rows) in cases {
        for pattern in 0...5 {
            let input = k.buffer((0..<rows).flatMap { _ in scores(entries, pattern: pattern) })
            let ints = [
                Int32(entries), Int32(pool), Int32(top), Int32(selected), Int32(first), Int32(tail), Int32(width), Int32(offset),
                Int32(selected),
            ]
            let expected = k.filled(rows * selected)
            try k.previous(
                "previous_index_select", [input, expected], ints: ints, grid: MTLSize(width: rows, height: 1, depth: 1),
                group: MTLSize(width: 256, height: 1, depth: 1))
            for threads in [1024, 256, 32] {
                let got = k.filled(rows * selected)
                try k.current(
                    "index_select", [input, got], ints: ints, grid: MTLSize(width: rows, height: 1, depth: 1),
                    group: MTLSize(width: threads, height: 1, depth: 1))
                guard k.bits(got, rows * selected) == k.bits(expected, rows * selected) else {
                    throw EngineError(
                        "index_select differs from the previous kernel: \(entries) entries of pool \(pool), top \(top), rows \(rows) at \(first), pattern \(pattern), \(threads) threads"
                    )
                }
                checked += 1
            }
        }
    }
    guard checked == cases.count * 6 * 3 else { throw EngineError("index_select: \(checked) cases run") }
}

/// index_scores (eight heads as independent sums) against the previous one sum a head: DeepSeek's indexer (32 heads of
/// 128 over 20,480 entries of ratio 1, 10,240 of ratio 2) and smaller head counts that leave a remainder or only one,
/// one row and several rows a dispatch, entries not yet visible included.
func familyIndexScoresMatchesPrevious() throws {
    let k = try Previous()
    var random = Random(state: 0x9E37_79B9_7F4A_7C15)
    // (heads, dim, entries, pool, first position, rows)
    let cases: [(Int, Int, Int, Int, Int, Int)] = [
        (32, 128, 20480, 1, 20479, 1), (32, 128, 10240, 2, 20479, 1), (32, 128, 2000, 1, 1990, 5),
        (12, 128, 700, 4, 2700, 3), (4, 16, 330, 1, 300, 2), (64, 64, 900, 1, 950, 1),
    ]
    for (heads, dim, entries, pool, first, rows) in cases {
        let q = k.buffer((0..<rows * heads * dim).map { _ in random.float() })
        let keys = k.buffer((0..<entries * dim).map { _ in random.float() })
        let w = k.buffer((0..<rows * heads).map { _ in random.float() })
        let ints = [Int32(heads), Int32(dim), Int32(entries), Int32(pool), Int32(first)]
        let grid = MTLSize(width: (entries + 63) / 64, height: rows, depth: 1)
        let group = MTLSize(width: 64, height: 1, depth: 1)
        let expected = k.filled(rows * entries)
        let got = k.filled(rows * entries)
        try k.previous("previous_index_scores", [q, keys, w, expected], ints: ints, grid: grid, group: group)
        try k.current("index_scores", [q, keys, w, got], ints: ints, grid: grid, group: group)
        guard k.bits(got, rows * entries) == k.bits(expected, rows * entries) else {
            throw EngineError(
                "index_scores differs from the previous kernel: \(heads) heads of \(dim), \(entries) entries of pool \(pool), \(rows) rows")
        }
    }
}

/// The eight-heads-a-thread decode attention against the one-head kernels it replaces where heads divide by 8:
/// DeepSeek's (64 heads of 512 over the 128-row window ring and 512 compressed rows, two buffers) and GLM's (64 heads,
/// 2,051 selected rows of one cache, heads apart by rows) layouts, unselected (-1) slots included, one and several rows.
func familyAttentionHeads8MatchesHeads() throws {
    let k = try Previous()
    var random = Random(state: 0xD1B5_4A32_D192_ED03)
    // (heads, dim, selected, split, rows, GLM layout)
    let cases: [(Int, Int, Int, Int, Int, Bool)] = [
        (64, 512, 640, 384, 1, false), (64, 512, 128, 384, 1, false), (64, 512, 640, 384, 3, false),
        (64, 512, 2051, Int(Int32.max), 1, true), (16, 64, 300, Int(Int32.max), 4, true),
    ]
    for (heads, dim, selected, split, rows, glm) in cases {
        let ringRows = min(split, 384)
        let second = 3000
        let kv = k.buffer((0..<ringRows * dim).map { _ in random.float() })
        let kv2 = k.buffer((0..<second * dim).map { _ in random.float() })
        let sel = k.buffer(
            (0..<rows * selected).map { j in
                let r = Int(random.next() % 1000)
                let row: Int32 =
                    r < 30
                    ? -1
                    : split == Int(Int32.max) ? Int32(r % ringRows) : (j % selected < 128 ? Int32(r % ringRows) : Int32(split + r % second))
                return Float(bitPattern: UInt32(bitPattern: row))
            })
        let q = k.buffer((0..<rows * heads * dim).map { _ in random.float() })
        let p = k.buffer((0..<rows * heads * selected).map { _ in abs(random.float()) / Float(selected) })
        // q and the output: [rows][heads][dim] (DeepSeek) or [heads][rows][dim] (GLM).
        let (rowStride, headStride) = glm ? (dim, rows * dim) : (heads * dim, dim)
        let scoreInts = [Int32(heads), Int32(dim), Int32(selected), Int32(rowStride), Int32(headStride), 0, 0, Int32(clamping: split)]
        let valueInts = [Int32(heads), Int32(dim), Int32(selected), 0, 0, Int32(rowStride), Int32(headStride), Int32(clamping: split)]
        let group = MTLSize(width: 64, height: 1, depth: 1)
        for (name, ints, floats, input, width, count) in [
            ("attend_scores", scoreInts, [Float(0.044)], q, selected, rows * heads * selected),
            ("attend_values", valueInts, [Float(0)], p, dim, rows * heads * dim),
        ] {
            let one = k.filled(count)
            let eight = k.filled(count)
            try k.current(
                name + "_heads", [input, kv, kv2, sel, one], ints: ints, floats: floats,
                grid: MTLSize(width: (width + 63) / 64, height: rows, depth: heads), group: group)
            try k.current(
                name + "_heads8", [input, kv, kv2, sel, eight], ints: ints, floats: floats,
                grid: MTLSize(width: (width + 63) / 64, height: rows, depth: heads / 8), group: group)
            guard k.bits(eight, count) == k.bits(one, count) else {
                throw EngineError(
                    "\(name)_heads8 differs from \(name)_heads: \(heads) heads of \(dim), \(selected) selected, \(rows) rows, GLM layout \(glm)"
                )
            }
        }
    }
}

/// hc_prepare (sixteen threads a row) against the previous one thread a row: decode and prompt row counts (one row,
/// rows that end inside a SIMD group, 300 rows), DeepSeek's 20 Sinkhorn iterations and 0 and 1, large and small mixes.
func familyHCPrepareMatchesPrevious() throws {
    let k = try Previous()
    var random = Random(state: 0xA076_1D64_78BD_642F)
    for (rows, iterations, spread) in [(1, 20, Float(3)), (3, 20, 3), (5, 1, 3), (70, 20, 40), (300, 20, 3), (2, 0, 3)] {
        let mixes = k.buffer((0..<rows * 24).map { _ in random.float() * spread })
        let scale = k.buffer((0..<3).map { _ in random.float() * 0.2 })
        let base = k.buffer((0..<24).map { _ in random.float() * 4 })
        let outputs = [4, 4, 16].map { n in [k.filled(rows * n), k.filled(rows * n)] }  // [pre, post, comb] x [previous, current]
        try k.previous(
            "previous_hc_prepare", [mixes, scale, base, outputs[0][0], outputs[1][0], outputs[2][0]],
            ints: [Int32(rows), Int32(iterations)], floats: [1e-6],
            grid: MTLSize(width: (rows + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        try k.current(
            "hc_prepare", [mixes, scale, base, outputs[0][1], outputs[1][1], outputs[2][1]], ints: [Int32(rows), Int32(iterations)],
            floats: [1e-6],
            grid: MTLSize(width: (rows * 16 + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        for (part, n) in zip(outputs, [4, 4, 16]) where k.bits(part[1], rows * n) != k.bits(part[0], rows * n) {
            throw EngineError("hc_prepare differs from the previous kernel: \(rows) rows, \(iterations) iterations, mixes of \(spread)")
        }
    }
}

/// The previous family kernels, verbatim from FamilyKernels.source before they were rewritten, renamed `previous_*`.
private let previousFamilyKernels = #"""
    #include <metal_stdlib>
    using namespace metal;

    inline float group_sum(float v, threadgroup float *scratch, uint tid, uint threads) {
        v = simd_sum(v);
        uint lane = tid % 32, warp = tid / 32, warps = (threads + 31) / 32;
        if (lane == 0) scratch[warp] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint w = 0; w < warps; ++w) total += scratch[w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return total;
    }
    inline uint ordered_bits(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

    struct HCArgs { int rows; int iterations; float eps; };
    kernel void previous_hc_prepare(device const float *mixes [[buffer(0)]], device const float *scale [[buffer(1)]], device const float *base [[buffer(2)]],
                                    device float *pre [[buffer(3)]], device float *post [[buffer(4)]], device float *comb [[buffer(5)]],
                                    constant HCArgs &a [[buffer(6)]], uint row [[thread_position_in_grid]]) {
        if ((int)row >= a.rows) return;
        device const float *m = mixes + row * 24;
        for (int h = 0; h < 4; ++h) {
            pre[row * 4 + h] = 1.0f / (1.0f + exp(-(m[h] * scale[0] + base[h]))) + a.eps;
            post[row * 4 + h] = 2.0f / (1.0f + exp(-(m[4 + h] * scale[1] + base[4 + h])));
        }
        float c[16];
        for (int i = 0; i < 16; ++i) c[i] = m[8 + i] * scale[2] + base[8 + i];
        for (int src = 0; src < 4; ++src) {
            float mx = -INFINITY;
            for (int d = 0; d < 4; ++d) mx = max(mx, c[d + 4 * src]);
            float s = 0.0f;
            for (int d = 0; d < 4; ++d) { c[d + 4 * src] = exp(c[d + 4 * src] - mx); s += c[d + 4 * src]; }
            for (int d = 0; d < 4; ++d) c[d + 4 * src] = c[d + 4 * src] / s + a.eps;
        }
        for (int it = 0; it < a.iterations; ++it) {
            if (it > 0) {
                for (int src = 0; src < 4; ++src) {
                    float s = 0.0f;
                    for (int d = 0; d < 4; ++d) s += c[d + 4 * src];
                    for (int d = 0; d < 4; ++d) c[d + 4 * src] /= (s + a.eps);
                }
            }
            for (int d = 0; d < 4; ++d) {
                float s = 0.0f;
                for (int src = 0; src < 4; ++src) s += c[d + 4 * src];
                for (int src = 0; src < 4; ++src) c[d + 4 * src] /= (s + a.eps);
            }
        }
        for (int i = 0; i < 16; ++i) comb[row * 16 + i] = c[i];
    }

    struct IndexArgs { int heads; int dim; int pools; int pool; int firstPosition; };
    kernel void previous_index_scores(device const float *q [[buffer(0)]], device const float *pooled [[buffer(1)]], device const float *w [[buffer(2)]],
                                      device float *scores [[buffer(3)]], constant IndexArgs &a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
        if ((int)p.x >= a.pools) return;
        ulong t = p.y; int pos = a.firstPosition + (int)t;
        float s = -INFINITY;
        if ((int(p.x) + 1) * a.pool - 1 <= pos) {
            s = 0.0f;
            device const float4 *k = (device const float4 *)(pooled + (ulong)p.x * a.dim);
            for (int h = 0; h < a.heads; ++h) {
                device const float4 *qq = (device const float4 *)(q + (t * a.heads + h) * a.dim);
                float4 acc = 0.0f;
                for (int i = 0; i < a.dim / 4; ++i) acc += qq[i] * k[i];
                s += w[t * a.heads + h] * max(acc.x + acc.y + acc.z + acc.w, 0.0f);
            }
        }
        scores[t * a.pools + p.x] = s;
    }

    struct SelectArgs { int pools; int pool; int top; int selected; int firstPosition; int tail; int width; int offset; int stride; };
    kernel void previous_index_select(device const float *scores [[buffer(0)]], device int *sel [[buffer(1)]], constant SelectArgs &a [[buffer(2)]],
                                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
        threadgroup float scratch[32];
        device const float *sr = scores + (ulong)t * a.pools;
        device int *out = sel + (ulong)t * a.stride;
        int pos = a.firstPosition + (int)t;
        int visible = min(a.pools, (pos + 1) / a.pool);
        int take = min(visible, a.top);
        uint lo = 0, hi = 0xFFFFFFFFu;
        if (take < visible) {
            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2 + 1;
                float count = 0.0f;
                for (int i = tid; i < visible; i += threads) count += ordered_bits(sr[i]) >= mid ? 1.0f : 0.0f;
                count = group_sum(count, scratch, tid, threads);
                if (count >= float(take)) lo = mid; else hi = mid - 1;
            }
        }
        if (tid != 0) return;
        int above = 0;
        if (take < visible) for (int i = 0; i < visible; ++i) above += ordered_bits(sr[i]) > lo ? 1 : 0;
        int ties = take - above, slot = 0;
        for (int i = 0; i < visible && slot < take; ++i) {
            bool chosen = take == visible;
            if (!chosen) { uint u = ordered_bits(sr[i]); chosen = u > lo || (u == lo && ties-- > 0); }
            if (chosen && sr[i] == -INFINITY) chosen = false;
            if (chosen) { for (int m = 0; m < a.width; ++m) out[slot * a.width + m] = a.offset + i * a.width + m; ++slot; }
        }
        int filled = slot * a.width;
        if (a.tail) for (int r = visible * a.pool; r <= pos && filled < a.selected; ++r) out[filled++] = a.offset + r;
        for (int i = filled; i < a.selected; ++i) out[i] = -1;
    }
    """#
