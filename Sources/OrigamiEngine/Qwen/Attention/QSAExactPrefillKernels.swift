// Exact QSA prefill attention kernels behind `qsa_exact_kernels`. Per output element every kernel
// keeps the ordered FP32 8x8x8 MMA sequence and the operands of the kernel it replaces; only the data
// path, the threadgroup geometry and the operand staging change (BITWISE). The lane -> fragment element
// layout of simdgroup_float8x8 (the layout the BM32 QMM kernels also rely on) is assumed and proven
// bitwise again by QSAExactKernelSelfChecks, which compares these kernels with the ones they replace
// from the same host positions. Nothing here changes a dispatch the flag does not select.
enum QSAExactPrefillKernels {
    /// Always compiled behind the prefill-reuse kernels: the fragment helpers and the direct stripe PV
    /// (`origami_pv_direct_stripe_c64`, replacing `origami_pv_full_width` on the 8K prefill-reuse path).
    static let stripeSource = #"""
        // Lane -> (row, column pair) of an 8x8 simdgroup matrix fragment (thread_elements()[0], [1]).
        #define ORIGAMI_FRAG(lane, FM, FN) \
            const uint FM = ((lane) / 4 & 4) + (((lane) / 2) % 4); \
            const uint FN = ((lane) / 4 & 2) * 2 + ((lane) % 2) * 2;

        // Two consecutive BF16 values (4-byte aligned) widened exactly, as two load_bf16 calls would.
        inline float2 origami_load_bf16x2(device const ushort *p) {
            uint w = *(reinterpret_cast<device const uint *>(p));
            return float2(as_type<float>(w << 16), as_type<float>(w & 0xffff0000u));
        }

        // The 512-column BF16 value row of one token; the same bytes origami_value_bits reads, as a row pointer
        // so a SIMD group can load its paired columns directly (prefix pages or the uncommitted suffix).
        #if ORIGAMI_PAGED_KV
        inline device const ushort *origami_value_row(ORIGAMI_KV_ARGS table, uint token) {
            uint old = table.offset[0];
            return token < old ? table.values[token / 256] + (token % 256) * 512 : table.newValues + (token - old) * 512;
        }
        #else
        inline device const ushort *origami_value_row(ORIGAMI_KV_ARGS values, uint token) { return values + token * 512; }
        #endif

        // Direct PV: one SIMD group per (8-row unit, F x 8 value columns); no threadgroup memory, no barriers.
        // Each SIMD group walks its unit's K8 tile sequence exactly as the control, builds the 8x8 probability
        // fragment and F 8x8 V fragments in registers, decides the all-zero skip with simd_any over the same 64
        // probabilities, and prefetches the next tile's operands before this tile's MMAs.
        #define ORIGAMI_PV_LOAD_V(KEY, B) \
            for (uint f = 0; f < F; ++f) \
                B[f] = (KEY) + fm < n ? origami_load_bf16x2(origami_value_row(values, (KEY) + fm) + kv * 256 + column + f * 8 + fn) : float2(0.0f);

        #define ORIGAMI_PV_MMA(A0, A1, B) { \
            a.thread_elements()[0] = A0; a.thread_elements()[1] = A1; \
            _Pragma("unroll") \
            for (uint f = 0; f < F; ++f) { \
                b.thread_elements()[0] = B[f].x; b.thread_elements()[1] = B[f].y; \
                simdgroup_multiply_accumulate(c[f], a, b, c[f]); \
            } }

        #define ORIGAMI_PV_EPILOGUE \
            for (uint f = 0; f < F; ++f) if (fm < rows) for (uint j = 0; j < 2; ++j) \
                out[((rowBase + fm / 4) * 24 + head + fm % 4) * 256 + column + f * 8 + fn + j] = round_bf16(c[f].thread_elements()[j]);

        // 8K path: direct PV over the dense probability stripe (control: origami_pv_full_width; same unit rows, the
        // partial BK=16 tail first, then the aligned prefix, the same zero-tile skip per 8x8 probability tile).
        #define ORIGAMI_PV_STRIPE_GATHER(KEY, A0, A1) { \
            A0 = 0.0f; A1 = 0.0f; \
            if (fm < rows) { \
                if ((KEY) + fn < n) A0 = load_bf16(probabilities[fm * n + (KEY) + fn]); \
                if ((KEY) + fn + 1 < n) A1 = load_bf16(probabilities[fm * n + (KEY) + fn + 1]); \
            } }

        #define ORIGAMI_PV_DIRECT_STRIPE(NAME, FRAGS) \
        kernel void NAME(device const ushort *probabilities [[buffer(0)]], ORIGAMI_KV_ARGS values [[buffer(1)]], \
            device float *out [[buffer(2)]], constant uint *shape [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], \
            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
            uint sg [[simdgroup_index_in_threadgroup]]) { \
            constexpr uint F = FRAGS; \
            uint n = shape[1], unit = shape[3] + group.y; \
            uint rowBase = (unit / 6) * 2, rows = min(2u, shape[0] - rowBase) * 4; \
            uint head = (unit % 6) * 4, kv = head / 12, column = sg * (F * 8); \
            probabilities += group.y * 8 * n; \
            ORIGAMI_FRAG(lane, fm, fn) \
            simdgroup_float8x8 a, b, c[F]; \
            for (uint f = 0; f < F; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16, steps = alignedKeys + tailSteps; \
            uint key = tailSteps > 0 ? alignedKeys : 0; \
            float a0 = 0.0f, a1 = 0.0f; float2 bv[F]; \
            if (steps > 0) { ORIGAMI_PV_STRIPE_GATHER(key, a0, a1) ORIGAMI_PV_LOAD_V(key, bv) } \
            for (uint step = 0; step < steps; step += 8) { \
                uint nstep = step + 8, nkey = 0; \
                float na0 = 0.0f, na1 = 0.0f; float2 nbv[F]; \
                if (nstep < steps) { \
                    nkey = nstep < tailSteps ? alignedKeys + nstep : nstep - tailSteps; \
                    ORIGAMI_PV_STRIPE_GATHER(nkey, na0, na1) ORIGAMI_PV_LOAD_V(nkey, nbv) \
                } \
                if (simd_any(a0 != 0.0f || a1 != 0.0f)) { ORIGAMI_PV_MMA(a0, a1, bv) } \
                key = nkey; a0 = na0; a1 = na1; \
                for (uint f = 0; f < F; ++f) bv[f] = nbv[f]; \
            } \
            ORIGAMI_PV_EPILOGUE \
        }

        ORIGAMI_PV_DIRECT_STRIPE(origami_pv_direct_stripe_c64, 8)
        """#

    /// Compiled after the compact reuse kernels (fused-prefill libraries only): the BF16 double-buffered
    /// compact QK (`origami_compact_qk_reuse_dbuf`, replacing `origami_compact_qk_reuse`) and the direct compact PV
    /// (`origami_pv_direct_compact_c64`, replacing `origami_compact_pv_reuse_kv` / `origami_compact_pv_reuse`).
    static let compactSource = #"""
        // Compact QK with BF16 double-buffered operands, register prefetch of the next d16 step, fragments
        // built from thread_elements(), one barrier per step, accumulator-element epilogue. Same 96 x 128 x 256
        // threadgroup, same K8 order per output element (d = 0, 8, ..., 248 with A row round_bf16(0.0625 q) and
        // the K bits widened), same compact stores, same path counters. 14,912 bytes of threadgroup memory.
        #define ORIGAMI_QK_STAGE_LOAD(D) \
            for (uint k = 0; k < 4; ++k) { \
                uint i = tid + k * 384, row = i / 16, col = i % 16; \
                uint query = queryBase + row / 12, head = kv * 12 + row % 12; \
                qa[k] = query < shape[0] ? store_bf16(0.0625f * load_bf16(queries[(query * 24 + head) * 256 + (D) + col])) : ushort(0); \
            } \
            for (uint k = 0; k < 6; ++k) { \
                uint i = tid + k * 384; \
                if (i < 2048) { \
                    uint row = i / 128, key = selectedTiles[(i % 128) / 8] * 8 + i % 8; \
                    qb[k] = key < n ? origami_key_bits(keys, key * 512 + kv * 256 + (D) + row) : ushort(0); \
                } \
            }
        #define ORIGAMI_QK_STAGE_STORE(BUF) \
            for (uint k = 0; k < 4; ++k) as[BUF][tid + k * 384] = qa[k]; \
            for (uint k = 0; k < 6; ++k) { uint i = tid + k * 384; if (i < 2048) bs[BUF][i] = qb[k]; }

        #define ORIGAMI_QK_DBUF_BODY \
            threadgroup ushort as[2][96 * 16], bs[2][16 * 128]; \
            ORIGAMI_FRAG(lane, fm, fn) \
            simdgroup_float8x8 a, b, c[16]; \
            for (uint f = 0; f < 16; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
            ushort qa[4], qb[6]; \
            ORIGAMI_QK_STAGE_LOAD(0) \
            ORIGAMI_QK_STAGE_STORE(0) \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint d = 0; d < 256; d += 16) { \
                const uint cur = (d / 16) & 1; \
                if (d + 16 < 256) { ORIGAMI_QK_STAGE_LOAD(d + 16) } \
                for (uint inner = 0; inner < 16; inner += 8) { \
                    a.thread_elements()[0] = load_bf16(as[cur][(sg * 8 + fm) * 16 + inner + fn]); \
                    a.thread_elements()[1] = load_bf16(as[cur][(sg * 8 + fm) * 16 + inner + fn + 1]); \
                    _Pragma("unroll") \
                    for (uint f = 0; f < 16; ++f) { \
                        if (f < tileCount) { \
                            b.thread_elements()[0] = load_bf16(bs[cur][(inner + fm) * 128 + f * 8 + fn]); \
                            b.thread_elements()[1] = load_bf16(bs[cur][(inner + fm) * 128 + f * 8 + fn + 1]); \
                            simdgroup_multiply_accumulate(c[f], a, b, c[f]); \
                        } \
                    } \
                } \
                if (d + 16 < 256) { ORIGAMI_QK_STAGE_STORE(cur ^ 1) } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            }

        kernel void origami_compact_qk_reuse_dbuf(device const ushort *queries [[buffer(0)]],
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
            threadgroup uint mappings[8 * 16], selectedTiles[16];
            if (tid < 16) selectedTiles[tid] = tid < tileCount ? origami_compact_union_id(packed, tileBase + tid) : 32768u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 128) {
                uint query = queryBase + tid / 16, tile = selectedTiles[tid % 16];
                uint mapping = 0;
                if (query < shape[0]) {
                    device const uint *meta = metadata + query * origami_compact_stride;
                    uint at = origami_compact_find(meta + 1025 + origami_compact_max, meta[origami_compact_stride - 1], tile);
                    if (at < origami_compact_max) mapping = meta[1025 + 2 * origami_compact_max + at];
                }
                mappings[tid] = mapping;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ORIGAMI_QK_DBUF_BODY
            #pragma unroll
            for (uint f = 0; f < 16; ++f) {
                if (f >= tileCount) continue;
                uint row = sg * 8 + fm;
                uint query = queryBase + row / 12, head = kv * 12 + row % 12;
                uint mapping = mappings[(row / 12) * 16 + f], mask = mapping >> 12;
                for (uint j = 0; j < 2; ++j) {
                    uint column = f * 8 + fn + j, key = selectedTiles[f] * 8 + column % 8;
                    if (query < shape[0] && key < n && (mask & (1u << (column % 8)))) {
                        uint at = (mapping & 4095u) + popcount(mask & ((1u << (column % 8)) - 1u));
                        scores[(query * 24 + head) * origami_compact_max + at] = store_bf16(c[f].thread_elements()[j]);
                    }
                }
            }
            if (tid == 0) origami_compact_reuse_count(path, 2, 1);
        }

        // Direct compact PV: the control's tile walk per 8-row unit (query pair x head quad): the merged sorted
        // K8 tile lists of the two queries, the partial BK=16 tail first and the aligned prefix next, or the full
        // domain when any of the unit's heads carries a nonzero default; the same all-zero skip per 8x8
        // probability tile; FP32 accumulation from zero. One SIMD group per 64 value columns.
        struct origami_pv_cursor { uint phase, i0, end0, i1, end1, fullAt, fullEnd; };

        inline bool origami_pv_next(thread origami_pv_cursor &cur, bool fullDomain,
            device const uint *tiles0, device const uint *tiles1, device const uint *maps0, device const uint *maps1,
            uint tail0, uint tail1, uint alignedKeys, thread uint &key, thread uint &map0, thread uint &map1) {
            for (;;) {
                if (cur.phase >= 2) return false;
                if (fullDomain ? cur.fullAt < cur.fullEnd : (cur.i0 < cur.end0 || cur.i1 < cur.end1)) {
                    map0 = 0; map1 = 0;
                    if (fullDomain) { key = cur.fullAt; cur.fullAt += 8; }
                    else {
                        uint t0 = cur.i0 < cur.end0 ? tiles0[cur.i0] : 0xffffffffu;
                        uint t1 = cur.i1 < cur.end1 ? tiles1[cur.i1] : 0xffffffffu;
                        uint tile = min(t0, t1); key = tile * 8;
                        if (t0 == tile) { map0 = maps0[cur.i0]; ++cur.i0; }
                        if (t1 == tile) { map1 = maps1[cur.i1]; ++cur.i1; }
                    }
                    return true;
                }
                ++cur.phase;
                cur.i0 = 0; cur.end0 = tail0; cur.i1 = 0; cur.end1 = tail1; cur.fullAt = 0; cur.fullEnd = alignedKeys;
            }
        }

        // Probability pair of fragment row `fm` at keys key+fn, key+fn+1 for the compact layout.
        #define ORIGAMI_PV_COMPACT_GATHER(KEY, MAP0, MAP1, A0, A1) { \
            A0 = 0.0f; A1 = 0.0f; \
            if (fm < rows) { \
                uint q = rowBase + fm / 4, h = head + fm % 4; \
                uint mapping = fm < 4 ? MAP0 : MAP1, mask = mapping >> 12; \
                for (uint j = 0; j < 2; ++j) { \
                    uint col = fn + j; \
                    if ((KEY) + col < n) { \
                        uint at = fullDomain ? origami_compact_find(ids + q * shape[2], min(counts[q], origami_compact_max), (KEY) + col) \
                            : ((mask & (1u << col)) != 0 ? (mapping & 4095u) + popcount(mask & ((1u << col) - 1u)) : origami_compact_max); \
                        float p = load_bf16(at < origami_compact_max ? probabilities[(q * 24 + h) * origami_compact_max + at] : defaults[q * 24 + h]); \
                        if (j == 0) A0 = p; else A1 = p; \
                    } \
                } \
            } }

        #define ORIGAMI_PV_DIRECT_COMPACT(NAME, FRAGS) \
        kernel void NAME(device const ushort *probabilities [[buffer(0)]], \
            ORIGAMI_KV_ARGS values [[buffer(1)]], device const uint *ids [[buffer(2)]], \
            device const uint *counts [[buffer(3)]], device const uint *metadata [[buffer(4)]], \
            device const ushort *defaults [[buffer(5)]], device float *out [[buffer(6)]], \
            device atomic_uint *path [[buffer(7)]], constant uint *shape [[buffer(8)]], \
            uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
            constexpr uint F = FRAGS; \
            uint n = shape[1], unit = group.y, rowBase = (unit / 6) * 2; \
            uint rows = min(2u, shape[0] - rowBase) * 4, head = (unit % 6) * 4, kv = head / 12; \
            uint column = sg * (F * 8); \
            device const uint *meta0 = metadata + rowBase * origami_compact_stride; \
            device const uint *meta1 = metadata + min(rowBase + 1, shape[0] - 1) * origami_compact_stride; \
            device const uint *tiles0 = meta0 + 1025 + origami_compact_max, *tiles1 = meta1 + 1025 + origami_compact_max; \
            device const uint *maps0 = meta0 + 1025 + 2 * origami_compact_max, *maps1 = meta1 + 1025 + 2 * origami_compact_max; \
            uint count0 = meta0[origami_compact_stride - 1], count1 = rows == 8 ? meta1[origami_compact_stride - 1] : 0; \
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16; \
            uint tail0 = count0, tail1 = count1; \
            while (tail0 > 0 && tiles0[tail0 - 1] * 8 >= alignedKeys) --tail0; \
            while (tail1 > 0 && tiles1[tail1 - 1] * 8 >= alignedKeys) --tail1; \
            bool fullDomain = false; \
            for (uint row = 0; row < rows; ++row) \
                fullDomain |= load_bf16(defaults[(rowBase + row / 4) * 24 + head + row % 4]) != 0.0f; \
            ORIGAMI_FRAG(lane, fm, fn) \
            simdgroup_float8x8 a, b, c[F]; \
            for (uint f = 0; f < F; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
            origami_pv_cursor cur; cur.phase = 0; cur.i0 = tail0; cur.end0 = count0; cur.i1 = tail1; cur.end1 = count1; \
            cur.fullAt = alignedKeys; cur.fullEnd = alignedKeys + tailSteps; \
            uint key = 0, map0 = 0, map1 = 0; \
            bool has = origami_pv_next(cur, fullDomain, tiles0, tiles1, maps0, maps1, tail0, tail1, alignedKeys, key, map0, map1); \
            float a0 = 0.0f, a1 = 0.0f; float2 bv[F]; \
            if (has) { ORIGAMI_PV_COMPACT_GATHER(key, map0, map1, a0, a1) ORIGAMI_PV_LOAD_V(key, bv) } \
            while (has) { \
                uint nkey = 0, nmap0 = 0, nmap1 = 0; \
                bool nhas = origami_pv_next(cur, fullDomain, tiles0, tiles1, maps0, maps1, tail0, tail1, alignedKeys, nkey, nmap0, nmap1); \
                float na0 = 0.0f, na1 = 0.0f; float2 nbv[F]; \
                if (nhas) { ORIGAMI_PV_COMPACT_GATHER(nkey, nmap0, nmap1, na0, na1) ORIGAMI_PV_LOAD_V(nkey, nbv) } \
                if (simd_any(a0 != 0.0f || a1 != 0.0f)) { ORIGAMI_PV_MMA(a0, a1, bv) } \
                has = nhas; key = nkey; a0 = na0; a1 = na1; \
                for (uint f = 0; f < F; ++f) bv[f] = nbv[f]; \
            } \
            ORIGAMI_PV_EPILOGUE \
            if (tid == 0) origami_compact_reuse_count(path, 6, 1); \
        }

        ORIGAMI_PV_DIRECT_COMPACT(origami_pv_direct_compact_c64, 8)
        """#
}
