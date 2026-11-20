import Foundation

/// Context-independent score/metadata storage for the selected long-domain
/// attention path. One resource has aligned, disjoint subregions; the caller's
/// normal attention scratch owner retains and charges it until joined completion.
/// Batched prefill with at least `minimumKeys` keys (no fused prefill) stores only the selected
/// scores: [rows][24 heads][2051] BF16 instead of [rows][24][keys]. The kernels reproduce the
/// stripe path's QK, softmax and PV bit for bit (BITWISE, checked against the original batched path).
package enum CompactSelectedAttention {
    static let minimumKeys = 16_384
    /// Per row, 7179 uint32 of metadata (offsets of 1024 four-key buckets of a 4096-key round + 1,
    /// 2051 bucket entries, 2051 K8 tile ids, 2051 tile mappings, the tile count), then the BF16
    /// probabilities and the per-(row, head) BF16 default probability for unselected keys.
    package struct Layout {
        package let probabilitiesOffset, defaultsOffset, bytes: Int
    }
    package static func layout(rows: Int) throws -> Layout {
        // origami_compact_softmax keeps one bit per 4096-key round in a 64-bit mask.
        guard (3...PinnedAttention.maximumQueries).contains(rows), ModelContract.maximumContext <= 64 * 4096 else {
            throw EngineError("Invalid compact attention query geometry")
        }
        func aligned(_ value: Int) -> Int { (value + 255) / 256 * 256 }
        let probabilities = aligned(rows * 7179 * 4)
        let defaults = aligned(probabilities + rows * 24 * PinnedAttention.maximumSelectedKeys * 2)
        let bytes = aligned(defaults + rows * 24 * 2)
        guard bytes <= PinnedAttention.maximumScoreScratchBytes else {
            throw EngineError("Compact attention exceeds existing scratch ceiling")
        }
        return Layout(probabilitiesOffset: probabilities, defaultsOffset: defaults, bytes: bytes)
    }
    static let source = #"""
        constant uint origami_compact_max = 2051;
        constant uint origami_compact_stride = 1025 + 2051 + 2051 + 2051 + 1;

        // Index of `key` in a sorted selection, or origami_compact_max when absent.
        inline uint origami_compact_find(device const uint *ids, uint count, uint key) {
            uint lo = 0, hi = count;
            while (lo < hi) {
                uint middle = (lo + hi) / 2;
                if (ids[middle] < key) lo = middle + 1; else hi = middle;
            }
            return lo < count && ids[lo] == key ? lo : origami_compact_max;
        }

        // One 256-thread group per row builds the metadata: bucket counts (threadgroup atomics), prefix offsets,
        // the ascending K8 tile list with an 8-bit key mask and first selection index per tile (thread 0), then the
        // bucket entries. Entry order inside a bucket is not deterministic; consumers only use it as a set.
        kernel void origami_compact_prepare(device const uint *ids [[buffer(0)]],
            device const uint *counts [[buffer(1)]], device uint *metadata [[buffer(2)]],
            constant uint *shape [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]) {
            device const uint *selected = ids + row * shape[2];
            device uint *meta = metadata + row * origami_compact_stride;
            uint count = min(counts[row], origami_compact_max), n = shape[1];
            threadgroup atomic_uint buckets[1024];
            for (uint i = tid; i < 1024; i += 256) atomic_store_explicit(buckets + i, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < count; i += 256) {
                uint key = selected[i];
                if (key < n) atomic_fetch_add_explicit(buckets + ((key & 4095u) / 4), 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                uint offset = 0;
                for (uint i = 0; i < 1024; ++i) {
                    uint amount = atomic_load_explicit(buckets + i, memory_order_relaxed);
                    meta[i] = offset; atomic_store_explicit(buckets + i, offset, memory_order_relaxed); offset += amount;
                }
                meta[1024] = offset;
                uint tiles = 0, previous = 0xffffffffu;
                device uint *mapping = meta + 1025 + 2 * origami_compact_max;
                for (uint i = 0; i < count; ++i) if (selected[i] < n) {
                    uint tile = selected[i] / 8;
                    if (tile != previous) {
                        meta[1025 + origami_compact_max + tiles] = tile;
                        mapping[tiles] = i; ++tiles; previous = tile;
                    }
                    mapping[tiles - 1] |= 1u << (12 + selected[i] % 8);
                }
                meta[origami_compact_stride - 1] = tiles;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < count; i += 256) {
                uint key = selected[i];
                if (key < n) {
                    uint index = atomic_fetch_add_explicit(buckets + ((key & 4095u) / 4), 1u, memory_order_relaxed);
                    if (index < origami_compact_max) meta[1025 + index] = i;
                }
            }
        }

        // One SIMD group per (selected K8 tile, row, six heads): the stripe QK's 8x8 FP32 MMA in K-eight steps
        // (six head rows used), scores of the selected keys stored BF16 at their compact slots.
        kernel void origami_compact_qk(device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], device const uint *ids [[buffer(2)]],
            device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]],
            device ushort *scores [[buffer(5)]], constant uint *shape [[buffer(6)]],
            uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint row = group.y, head = group.z * 6, kv = head / 12, n = shape[1];
            device const uint *meta = metadata + row * origami_compact_stride;
            if (group.x >= meta[origami_compact_stride - 1]) return;
            uint keyBase = meta[1025 + origami_compact_max + group.x] * 8;
            uint mapping = meta[1025 + 2 * origami_compact_max + group.x], mask = mapping >> 12;
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 256; d += 8) {
                for (uint index = lane; index < 64; index += 32) {
                    uint h = index / 8, col = index % 8;
                    as[index] = h < 6 ? round_bf16(0.0625f * load_bf16(queries[(row * 24 + head + h) * 256 + d + col])) : 0.0f;
                    bs[index] = keyBase + col < n ? load_bf16(origami_key_bits(keys, (keyBase + col) * 512 + kv * 256 + d + h)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) {
                uint h = index / 8, col = index % 8;
                if (h < 6 && (mask & (1u << col)) != 0) {
                    uint at = (mapping & 4095u) + popcount(mask & ((1u << col) - 1u));
                    scores[(row * 24 + head + h) * origami_compact_max + at] = store_bf16(cs[index]);
                }
            }
        }

        // One 1024-thread group per (row, head) reproduces origami_softmax_short/long over the full domain: thread tid
        // owns keys tid*4.. of each 4096-key round, unselected keys count as the mask value, rounds without a
        // selected key of this thread are skipped (they add nothing). An all-masked row stores its uniform (or NaN)
        // probability in `defaults`, which PV then applies to every key.
        kernel void origami_compact_softmax(device ushort *scores [[buffer(0)]],
            device const uint *ids [[buffer(1)]], device const uint *counts [[buffer(2)]],
            device const uint *metadata [[buffer(3)]], device ushort *defaults [[buffer(4)]],
            constant uint *shape [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
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
                // One bit per 4096-key round: keys / 4096 < 64 holds up to ModelContract.maximumContext (262,144).
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
        }

        // One SIMD group per (8 value columns, unit of 2 queries x 4 heads): the stripe PV's MMA sequence over the
        // merged sorted tile lists of both queries (partial BK=16 tail first), or over the full domain when a default
        // probability is nonzero; all-zero probability tiles are skipped as in origami_pv_mma_batched.
        kernel void origami_compact_pv(device const ushort *probabilities [[buffer(0)]],
            ORIGAMI_KV_ARGS values [[buffer(1)]], device const uint *ids [[buffer(2)]],
            device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]],
            device const ushort *defaults [[buffer(5)]], device float *out [[buffer(6)]],
            constant uint *shape [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]]) {
            uint n = shape[1], unit = group.y, rowBase = (unit / 6) * 2;
            uint rows = min(2u, shape[0] - rowBase) * 4, head = (unit % 6) * 4, column = group.x * 8;
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
            threadgroup float as[64], bs[64], cs[64];
            simdgroup_float8x8 a, b, c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
                    if (!simd_any(anyProbability)) continue;
                    for (uint index = lane; index < 64; index += 32) {
                        uint row = index / 8, col = index % 8;
                        bs[index] = key + row < n ? load_bf16(origami_value_bits(values,
                            (key + row) * 512 + (head / 12) * 256 + column + col)) : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(a, as, 8); simdgroup_load(b, bs, 8); simdgroup_multiply_accumulate(c, a, b, c);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            simdgroup_store(c, cs, 8); threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = lane; index < 64; index += 32) if (index / 8 < rows)
                out[((rowBase + (index / 8) / 4) * 24 + head + (index / 8) % 4) * 256 + column + index % 8] = round_bf16(cs[index]);
        }

        """#
}
