// Fused compact prefill attention (`fused_selected_prefill_attention`, planned on the 32K tier): the QSA
// prefill's selected-key attention split into union, QK, softmax and PV kernels over each query's compact
// selection (at most 2051 keys) instead of full score rows. Compact selected scores retain the exact BF16
// cuts; QK reuses eight queries and twelve heads, and PV reuses the original query pair across the full
// 256-column value width. BITWISE: outputs are bit-identical to the original batched path
// (compact_prefill_reuse_* checks, up to 262,144 keys). With `qsa_exact_kernels` (the release setting) the
// QK and PV here are replaced by QSAExactPrefillKernels and remain as their references; the union and
// softmax kernels always run. All device storage remains inside the existing 32 MiB attention workspace.
package enum CompactPrefillReuse {
    /// Byte offsets inside the score scratch: the compact layout, then one union count per 8-query group,
    /// then each group's union of K8 tile IDs packed 15 bits apiece.
    package struct Layout {
        package let compact: CompactSelectedAttention.Layout
        package let unionCountsOffset, unionIDsOffset, unionWordsPerGroup, maximumUnionTiles, bytes: Int
    }
    package static func layout(rows: Int, keys: Int) throws -> Layout {
        guard (rows...ModelContract.maximumContext).contains(keys) else { throw EngineError("Invalid compact reuse key geometry") }
        let compact = try CompactSelectedAttention.layout(rows: rows)
        func aligned(_ n: Int) -> Int { (n + 255) / 256 * 256 }
        let groups = (rows + 7) / 8
        let tiles = min((keys + 7) / 8, min(rows, 8) * 2051)
        // Absolute K8 IDs need only 15 bits through the exact 262K ceiling.
        // Packing is GPU-only and keeps even the worst union inside 32 MiB.
        let words = (tiles * 15 + 31) / 32
        let counts = compact.bytes
        let ids = aligned(counts + groups * 4)
        let bytes = aligned(ids + groups * words * 4)
        guard bytes <= PinnedAttention.maximumScoreScratchBytes else { throw EngineError("Compact union exceeds existing owner") }
        return .init(
            compact: compact, unionCountsOffset: counts, unionIDsOffset: ids,
            unionWordsPerGroup: words, maximumUnionTiles: tiles, bytes: bytes)
    }
    static let source = #"""
        // Adds `amount` to a 64-bit path counter stored as two 32-bit words (low at `at`, carry at `at + 1`).
        inline void origami_compact_reuse_count(device atomic_uint *path, uint at, uint amount) {
            uint old = atomic_fetch_add_explicit(path + at, amount, memory_order_relaxed);
            if (old > 0xffffffffu - amount) atomic_fetch_add_explicit(path + at + 1, 1u, memory_order_relaxed);
        }
        // The `index`-th 15-bit K8 tile ID of a packed union (a field may straddle two words).
        inline uint origami_compact_union_id(device const uint *packed, uint index) {
            uint bit = index * 15, word = bit / 32, shift = bit % 32;
            uint value = packed[word] >> shift;
            if (shift > 17) value |= packed[word + 1] << (32 - shift);
            return value & 32767u;
        }
        // One 256-thread group per 8 query rows: marks every K8 tile (8 keys) any of the 8 queries selected in a
        // 32,768-bit threadgroup bitmap (262,144 keys), ranks the set bits with a prefix popcount scan and writes the
        // ascending union as packed 15-bit IDs plus its count. Integer only; the order is the sorted tile order.
        // Path counters: 10 groups, 12 union tiles, 14 selected tiles.
        kernel void origami_compact_union_prepare(device const uint *metadata [[buffer(0)]],
            device atomic_uint *packed [[buffer(1)]], device uint *counts [[buffer(2)]],
            device atomic_uint *path [[buffer(3)]], constant uint *shape [[buffer(4)]],
            uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
            threadgroup atomic_uint bitmap[1024];
            threadgroup uint prefix[1024];
            for (uint i = tid; i < 1024; i += 256) atomic_store_explicit(bitmap + i, 0u, memory_order_relaxed);
            for (uint i = tid; i < shape[3]; i += 256) atomic_store_explicit(packed + group * shape[3] + i, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = 0; q < min(8u, shape[0] - group * 8); ++q) {
                device const uint *meta = metadata + (group * 8 + q) * origami_compact_stride;
                for (uint i = tid; i < meta[origami_compact_stride - 1]; i += 256) {
                    uint tile = meta[1025 + origami_compact_max + i];
                    atomic_fetch_or_explicit(bitmap + tile / 32, 1u << (tile % 32), memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < 1024; i += 256) prefix[i] = popcount(atomic_load_explicit(bitmap + i, memory_order_relaxed));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint offset = 1; offset < 1024; offset *= 2) {
                uint previous[4];
                for (uint j = 0; j < 4; ++j) { uint i = tid + j * 256; previous[j] = i >= offset ? prefix[i - offset] : 0; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0; j < 4; ++j) prefix[tid + j * 256] += previous[j];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // This is the only writer group for its compact union. Retire every zero
            // before disjoint packed fields are OR-published, including shared words.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            for (uint i = tid; i < 1024; i += 256) {
                uint bits = atomic_load_explicit(bitmap + i, memory_order_relaxed);
                uint output = i == 0 ? 0 : prefix[i - 1];
                while (bits != 0) {
                    uint lane = ctz(bits), tile = i * 32 + lane;
                    bits &= bits - 1;
                    uint bit = output++ * 15, word = group * shape[3] + bit / 32, shift = bit % 32;
                    atomic_fetch_or_explicit(packed + word, tile << shift, memory_order_relaxed);
                    if (shift > 17) atomic_fetch_or_explicit(packed + word + 1, tile >> (32 - shift), memory_order_relaxed);
                }
            }
            if (tid == 0) {
                counts[group] = prefix[1023];
                origami_compact_reuse_count(path, 10, 1);
                origami_compact_reuse_count(path, 12, prefix[1023]);
                uint selected = 0;
                for (uint q = 0; q < min(8u, shape[0] - group * 8); ++q)
                    selected += metadata[(group * 8 + q) * origami_compact_stride + origami_compact_stride - 1];
                origami_compact_reuse_count(path, 14, selected);
            }
        }

            // QK scores for the union: grid (union tiles / 16, query groups of 8, 2 KV heads), 384 threads (12 SIMD
            // groups, one per 8 of the 96 query-head rows = 8 queries x 12 heads of the KV head). Queries are scaled by
            // 1/16 and rounded to bf16, keys loaded as bf16; FP32 simdgroup 8x8 MMA over the 256-wide head dimension in
            // 16-column steps, 16 tiles (128 keys) per group. Only scores a query actually selected are stored, as bf16,
            // at its compact (sorted) index. Same operand cuts and accumulation order as the original QK: BITWISE.
            kernel void origami_compact_qk_reuse(device const ushort *queries [[buffer(0)]],
                ORIGAMI_KV_ARGS keys [[buffer(1)]], device const uint *ids [[buffer(2)]],
                device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]],
                device ushort *scores [[buffer(5)]], device atomic_uint *path [[buffer(6)]],
                device const uint *packed [[buffer(7)]], device const uint *unionCounts [[buffer(8)]],
                constant uint *shape [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
                uint n = shape[1], queryBase = group.y * 8;
                uint kv = group.z, tileBase = group.x * 16;
                if (tileBase >= unionCounts[group.y]) {
                    if (tid == 0) origami_compact_reuse_count(path, 4, 1);
                    return;
                }
                uint tileCount = min(16u, unionCounts[group.y] - tileBase);
                packed += group.y * shape[3];
                threadgroup float as[96 * 16], bs[16 * 128];
                threadgroup uint mappings[8 * 16], selectedTiles[16];
                if (tid < 16) selectedTiles[tid] = tid < tileCount ? origami_compact_union_id(packed, tileBase + tid) : 32768u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // One search per query/K8 tile replaces eight per-key searches. The
                // compact maps retain each selected score's original sorted index.
                if (tid < 128) {
                    uint query = queryBase + tid / 16, tile = selectedTiles[tid % 16];
                    uint mapping = 0;
                    if (query < shape[0]) {
                        device const uint *meta = metadata + query * origami_compact_stride;
                        uint at = origami_compact_find(meta + 1025 + origami_compact_max,
                            meta[origami_compact_stride - 1], tile);
                        if (at < origami_compact_max) mapping = meta[1025 + 2 * origami_compact_max + at];
                    }
                    mappings[tid] = mapping;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 a, b, c[16];
                for (uint f = 0; f < 16; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint d = 0; d < 256; d += 16) {
                    for (uint i = tid; i < 96 * 16; i += 384) {
                        uint row = i / 16, col = i % 16;
                        uint query = queryBase + row / 12, head = kv * 12 + row % 12;
                        as[i] = query < shape[0] ? round_bf16(0.0625f * load_bf16(queries[(query * 24 + head) * 256 + d + col])) : 0.0f;
                    }
                    for (uint i = tid; i < 16 * 128; i += 384) {
                        uint row = i / 128, key = selectedTiles[(i % 128) / 8] * 8 + i % 8;
                        bs[i] = key < n ? load_bf16(origami_key_bits(keys, key * 512 + kv * 256 + d + row)) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint inner = 0; inner < 16; inner += 8) {
                        simdgroup_load(a, as + sg * 8 * 16 + inner, 16);
                        // Keep opaque matrix registers statically indexed while the
                        // uniform guard omits padded gathered columns.
                        #pragma unroll
                        for (uint f = 0; f < 16; ++f) {
                            if (f < tileCount) {
                                simdgroup_load(b, bs + inner * 128 + f * 8, 128);
                                simdgroup_multiply_accumulate(c[f], a, b, c[f]);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                // All operand reads joined above. Each SIMD group has its own store stripe.
                #pragma unroll
                for (uint f = 0; f < 16; ++f) {
                    if (f >= tileCount) continue;
                    simdgroup_store(c[f], bs + sg * 64, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint i = lane; i < 64; i += 32) {
                        uint row = sg * 8 + i / 8, column = f * 8 + i % 8;
                        uint query = queryBase + row / 12, head = kv * 12 + row % 12, key = selectedTiles[column / 8] * 8 + column % 8;
                        uint mapping = mappings[(row / 12) * 16 + column / 8], mask = mapping >> 12;
                        if (query < shape[0] && key < n && (mask & (1u << (column % 8)))) {
                            uint at = (mapping & 4095u) + popcount(mask & ((1u << (column % 8)) - 1u));
                            scores[(query * 24 + head) * origami_compact_max + at] = store_bf16(bs[sg * 64 + i]);
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) origami_compact_reuse_count(path, 2, 1);
            }

        // Softmax of one (query row, head) per 1024-thread group over all n causal keys, where unselected keys carry
        // the mask score (most negative finite bf16) and keys past n are -inf. n <= 4096: each thread owns 4 adjacent
        // keys; longer rows run an online max/sum over 4096-key rounds that hold a selected key (skipped rounds add
        // exp(mask - max) = 0). simd_max/simd_sum, then across the 32 SIMD groups, as the original batched softmax.
        // Writes bf16 probabilities in place and one bf16 default probability for unselected keys (uniform
        // 1/(n - excluded) when every key is masked, NaN propagated). BITWISE with the original softmax.
        kernel void origami_compact_softmax_reuse(device ushort *scores [[buffer(0)]],
            device const uint *ids [[buffer(1)]], device const uint *counts [[buffer(2)]],
            device const uint *metadata [[buffer(3)]], device ushort *defaults [[buffer(4)]],
            device atomic_uint *path [[buffer(5)]], constant uint *shape [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]]) {
            uint row = group.x, head = group.y, n = shape[1], count = min(counts[row], origami_compact_max);
            device const uint *meta = metadata + row * origami_compact_stride;
            device const uint *entries = meta + 1025, *selected = ids + row * shape[2];
            scores += (row * 24 + head) * origami_compact_max;
            threadgroup float maxima[32], sums[32];
            threadgroup uint uniform;
            if (tid < 32) { maxima[tid] = -INFINITY; sums[tid] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mask = load_bf16(ushort(0xff7f)), maximum = -FLT_MAX, sum = 0.0f;
            uint shortThreads = ((n + 127) / 128) * 32;
            if (n > 4096) {
                maximum = count < n ? mask : -FLT_MAX;
                ulong rounds = 0;
                for (uint e = meta[tid]; e < meta[tid + 1]; ++e) rounds |= 1ul << (selected[entries[e]] / 4096);
                while (rounds != 0) {
                    uint r = uint(ctz(rounds)); rounds &= rounds - 1;
                    uint offset = r * 4096 + tid * 4;
                    float x[4];
                    for (uint i = 0; i < 4; ++i) x[i] = offset + i < n ? mask : -INFINITY;
                    for (uint e = meta[tid]; e < meta[tid + 1]; ++e) {
                        uint at = entries[e], key = selected[at];
                        if (key / 4096 == r) x[key % 4] = load_bf16(scores[at]);
                    }
                    float previous = maximum;
                    for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
                    sum *= fast::exp(previous - maximum);
                    for (uint i = 0; i < 4; ++i) sum += fast::exp(x[i] - maximum);
                }
                float previous = maximum; maximum = simd_max(maximum);
                sum *= fast::exp(previous - maximum); sum = simd_sum(sum);
                if (lane == 0) maxima[sg] = maximum;
            } else if (tid < shortThreads) {
                float x[4];
                for (uint i = 0; i < 4; ++i) x[i] = tid * 4 + i < n ? mask : -INFINITY;
                for (uint e = meta[tid]; e < meta[tid + 1]; ++e) {
                    uint at = entries[e]; x[selected[at] % 4] = load_bf16(scores[at]);
                }
                for (uint i = 0; i < 4; ++i) maximum = maximum < x[i] ? x[i] : maximum;
                maximum = simd_max(maximum);
                if (lane == 0) maxima[sg] = maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float globalMaximum = simd_max(maxima[lane]);
            if (n > 4096) {
                sum *= fast::exp(maximum - globalMaximum);
                if (lane == 0) sums[sg] = sum;
            } else if (tid < shortThreads) {
                float x[4];
                for (uint i = 0; i < 4; ++i) x[i] = tid * 4 + i < n ? mask : -INFINITY;
                for (uint e = meta[tid]; e < meta[tid + 1]; ++e) {
                    uint at = entries[e]; x[selected[at] % 4] = load_bf16(scores[at]);
                }
                sum = 0.0f;
                for (uint i = 0; i < 4; ++i) { x[i] = fast::exp(x[i] - globalMaximum); sum += x[i]; }
                sum = simd_sum(sum);
                if (lane == 0) sums[sg] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inverse = 1.0f / simd_sum(sums[lane]);
            bool allMasked = count == 0 || globalMaximum == mask;
            if (tid == 0) {
                uniform = uint(store_bf16(fast::exp(mask - globalMaximum) * inverse));
                if (allMasked) {
                    uint excluded = 0; bool nonfiniteNaN = false;
                    for (uint at = 0; at < count; ++at) {
                        ushort bits = scores[at];
                        excluded += bits == ushort(0xff80);
                        nonfiniteNaN |= (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0;
                    }
                    uniform = uint(store_bf16(nonfiniteNaN ? NAN : 1.0f / float(n - excluded)));
                }
                defaults[row * 24 + head] = ushort(uniform);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint at = tid; at < count; at += 1024)
                scores[at] = allMasked
                    ? (scores[at] == ushort(0xff80) && (uniform & 0x7f80) != 0x7f80 ? ushort(0) : ushort(uniform))
                    : store_bf16(fast::exp(load_bf16(scores[at]) - globalMaximum) * inverse);
            if (tid == 0) origami_compact_reuse_count(path, 8, 1);
        }

        // PV for one query pair and head quad (8 MMA rows) per 256-thread group, grid ((rows + 1) / 2) x 6. Walks
        // 8-key tiles in the control's order (the partial BK=16 tail first, then the aligned prefix), merging both
        // queries' sorted tiles, or every tile when an unselected key has a nonzero default probability; skips
        // all-zero probability tiles; 8 SIMD groups x 32 value columns, four 8x8 FP32 accumulators each from zero;
        // stores round_bf16 values into the fp32 output. BITWISE with the original PV.
        kernel void origami_compact_pv_reuse(device const ushort *probabilities [[buffer(0)]],
            ORIGAMI_KV_ARGS values [[buffer(1)]], device const uint *ids [[buffer(2)]],
            device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]],
            device const ushort *defaults [[buffer(5)]], device float *out [[buffer(6)]],
            device atomic_uint *path [[buffer(7)]], constant uint *shape [[buffer(8)]],
            uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], unit = group.y, rowBase = (unit / 6) * 2;
            uint rows = min(2u, shape[0] - rowBase) * 4, head = (unit % 6) * 4;
            device const uint *meta0 = metadata + rowBase * origami_compact_stride;
            device const uint *meta1 = metadata + min(rowBase + 1, shape[0] - 1) * origami_compact_stride;
            device const uint *tiles0 = meta0 + 1025 + origami_compact_max;
            device const uint *tiles1 = meta1 + 1025 + origami_compact_max;
            uint count0 = meta0[origami_compact_stride - 1], count1 = rows == 8 ? meta1[origami_compact_stride - 1] : 0;
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16;
            uint tail0 = count0, tail1 = count1;
            while (tail0 > 0 && tiles0[tail0 - 1] * 8 >= alignedKeys) --tail0;
            while (tail1 > 0 && tiles1[tail1 - 1] * 8 >= alignedKeys) --tail1;
            bool fullDomain = false;
            for (uint row = 0; row < rows; ++row)
                fullDomain |= load_bf16(defaults[(rowBase + row / 4) * 24 + head + row % 4]) != 0.0f;
            threadgroup float as[64], bs[8 * 256], cs[8 * 64];
            threadgroup uint active;
            simdgroup_float8x8 a, b, c[4];
            for (uint f = 0; f < 4; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            // The partial BK=16 tail precedes the aligned prefix in the control.
            // Each phase merges sorted absolute tiles from both original queries.
            for (uint phase = 0; phase < 2; ++phase) {
                uint i0 = phase == 0 ? tail0 : 0, end0 = phase == 0 ? count0 : tail0;
                uint i1 = phase == 0 ? tail1 : 0, end1 = phase == 0 ? count1 : tail1;
                uint fullBegin = phase == 0 ? alignedKeys : 0;
                uint fullEnd = phase == 0 ? alignedKeys + tailSteps : alignedKeys;
                uint fullAt = fullBegin;
                while (fullDomain ? fullAt < fullEnd : (i0 < end0 || i1 < end1)) {
                    uint key, map0 = 0, map1 = 0;
                    if (fullDomain) { key = fullAt; fullAt += 8; }
                    else {
                        uint t0 = i0 < end0 ? tiles0[i0] : 0xffffffffu;
                        uint t1 = i1 < end1 ? tiles1[i1] : 0xffffffffu;
                        uint tile = min(t0, t1); key = tile * 8;
                        if (t0 == tile) { map0 = meta0[1025 + 2 * origami_compact_max + i0]; ++i0; }
                        if (t1 == tile) { map1 = meta1[1025 + 2 * origami_compact_max + i1]; ++i1; }
                    }
                    if (sg == 0) {
                        bool anyProbability = false;
                        for (uint index = lane; index < 64; index += 32) {
                            uint row = index / 8, col = index % 8;
                            float probability = 0.0f;
                            if (row < rows && key + col < n) {
                                uint q = rowBase + row / 4, h = head + row % 4;
                                uint mapping = row < 4 ? map0 : map1, mask = mapping >> 12;
                                uint at = fullDomain ? origami_compact_find(ids + q * shape[2], min(counts[q], origami_compact_max), key + col)
                                    : ((mask & (1u << col)) != 0 ? (mapping & 4095u) + popcount(mask & ((1u << col) - 1u)) : origami_compact_max);
                                probability = load_bf16(at < origami_compact_max
                                    ? probabilities[(q * 24 + h) * origami_compact_max + at] : defaults[q * 24 + h]);
                            }
                            as[index] = probability; anyProbability |= probability != 0.0f;
                        }
                        bool any = simd_any(anyProbability);
                        if (lane == 0) active = any;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    bool perform = active != 0;
                    // Retire every reader before the next tile rewrites the decision.
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (!perform) continue;
                    for (uint i = tid; i < 8 * 256; i += 256) {
                        uint row = i / 256, col = i % 256;
                        bs[i] = key + row < n ? load_bf16(origami_value_bits(values,
                            (key + row) * 512 + (head / 12) * 256 + col)) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(a, as, 8);
                    for (uint f = 0; f < 4; ++f) {
                        simdgroup_load(b, bs + sg * 32 + f * 8, 256);
                        simdgroup_multiply_accumulate(c[f], a, b, c[f]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint f = 0; f < 4; ++f) {
                simdgroup_store(c[f], cs + sg * 64, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) if (i / 8 < rows) {
                    uint row = i / 8, col = sg * 32 + f * 8 + i % 8;
                    out[((rowBase + row / 4) * 24 + head + row % 4) * 256 + col] = round_bf16(cs[sg * 64 + i]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) origami_compact_reuse_count(path, 6, 1);
        }

        // `compact_pv_grouped`: one threadgroup (24 SIMDgroups, 768 threads) per (query pair, KV head). SIMDgroups 8j..8j+7 own the original
        // 8-row unit j (head quad kv*3+j), 32 value columns each with the original four accumulators. The three
        // units share each 8x256 V tile load and its barriers; each keeps the original arithmetic: the same tiles
        // in the same order (BK=16 tail first, then the aligned prefix), its own all-zero skip, and FP32 MMA
        // accumulation from zero. If any unit needs the full domain, the group walks the full domain, which
        // visits the same nonzero tiles in the same order for the others.
        kernel void origami_compact_pv_reuse_kv(device const ushort *probabilities [[buffer(0)]],
            ORIGAMI_KV_ARGS values [[buffer(1)]], device const uint *ids [[buffer(2)]],
            device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]],
            device const ushort *defaults [[buffer(5)]], device float *out [[buffer(6)]],
            device atomic_uint *path [[buffer(7)]], constant uint *shape [[buffer(8)]],
            uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sgAll [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], kv = group.y % 2, rowBase = (group.y / 2) * 2;
            uint rows = min(2u, shape[0] - rowBase) * 4;
            uint unit = sgAll / 8, sg = sgAll % 8, head = (kv * 3 + unit) * 4;
            device const uint *meta0 = metadata + rowBase * origami_compact_stride;
            device const uint *meta1 = metadata + min(rowBase + 1, shape[0] - 1) * origami_compact_stride;
            device const uint *tiles0 = meta0 + 1025 + origami_compact_max;
            device const uint *tiles1 = meta1 + 1025 + origami_compact_max;
            uint count0 = meta0[origami_compact_stride - 1], count1 = rows == 8 ? meta1[origami_compact_stride - 1] : 0;
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16;
            uint tail0 = count0, tail1 = count1;
            while (tail0 > 0 && tiles0[tail0 - 1] * 8 >= alignedKeys) --tail0;
            while (tail1 > 0 && tiles1[tail1 - 1] * 8 >= alignedKeys) --tail1;
            bool fullDomain = false;
            for (uint u = 0; u < 3; ++u)
                for (uint row = 0; row < rows; ++row)
                    fullDomain |= load_bf16(defaults[(rowBase + row / 4) * 24 + (kv * 3 + u) * 4 + row % 4]) != 0.0f;
            threadgroup float as[3 * 64], bs[8 * 256], cs[24 * 64];
            threadgroup uint active[3];
            simdgroup_float8x8 a, b;
            simdgroup_float8x8 c[4];
            for (uint f = 0; f < 4; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint phase = 0; phase < 2; ++phase) {
                uint i0 = phase == 0 ? tail0 : 0, end0 = phase == 0 ? count0 : tail0;
                uint i1 = phase == 0 ? tail1 : 0, end1 = phase == 0 ? count1 : tail1;
                uint fullBegin = phase == 0 ? alignedKeys : 0;
                uint fullEnd = phase == 0 ? alignedKeys + tailSteps : alignedKeys;
                uint fullAt = fullBegin;
                while (fullDomain ? fullAt < fullEnd : (i0 < end0 || i1 < end1)) {
                    uint key, map0 = 0, map1 = 0;
                    if (fullDomain) { key = fullAt; fullAt += 8; }
                    else {
                        uint t0 = i0 < end0 ? tiles0[i0] : 0xffffffffu;
                        uint t1 = i1 < end1 ? tiles1[i1] : 0xffffffffu;
                        uint tile = min(t0, t1); key = tile * 8;
                        if (t0 == tile) { map0 = meta0[1025 + 2 * origami_compact_max + i0]; ++i0; }
                        if (t1 == tile) { map1 = meta1[1025 + 2 * origami_compact_max + i1]; ++i1; }
                    }
                    if (sg == 0) {
                        bool anyProbability = false;
                        for (uint index = lane; index < 64; index += 32) {
                            uint row = index / 8, col = index % 8;
                            float probability = 0.0f;
                            if (row < rows && key + col < n) {
                                uint q = rowBase + row / 4, h = head + row % 4;
                                uint mapping = row < 4 ? map0 : map1, mask = mapping >> 12;
                                uint at = fullDomain ? origami_compact_find(ids + q * shape[2], min(counts[q], origami_compact_max), key + col)
                                    : ((mask & (1u << col)) != 0 ? (mapping & 4095u) + popcount(mask & ((1u << col) - 1u)) : origami_compact_max);
                                probability = load_bf16(at < origami_compact_max
                                    ? probabilities[(q * 24 + h) * origami_compact_max + at] : defaults[q * 24 + h]);
                            }
                            as[unit * 64 + index] = probability; anyProbability |= probability != 0.0f;
                        }
                        bool any = simd_any(anyProbability);
                        if (lane == 0) active[unit] = any;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    bool perform = active[unit] != 0, anyPerform = active[0] != 0 || active[1] != 0 || active[2] != 0;
                    // Retire every reader before the next tile rewrites the decisions.
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (!anyPerform) continue;
                    for (uint i = tid; i < 8 * 256; i += 768) {
                        uint row = i / 256, col = i % 256;
                        bs[i] = key + row < n ? load_bf16(origami_value_bits(values, (key + row) * 512 + kv * 256 + col)) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (perform) {
                        simdgroup_load(a, as + unit * 64, 8);
                        for (uint f = 0; f < 4; ++f) {
                            simdgroup_load(b, bs + sg * 32 + f * 8, 256);
                            simdgroup_multiply_accumulate(c[f], a, b, c[f]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint f = 0; f < 4; ++f) {
                simdgroup_store(c[f], cs + sgAll * 64, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) if (i / 8 < rows) {
                    uint row = i / 8, col = sg * 32 + f * 8 + i % 8;
                    out[((rowBase + row / 4) * 24 + head + row % 4) * 256 + col] = round_bf16(cs[sgAll * 64 + i]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) origami_compact_reuse_count(path, 6, 3);
        }
        """#
}
