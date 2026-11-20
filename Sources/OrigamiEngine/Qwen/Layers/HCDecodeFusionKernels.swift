import Foundation

/// `hc_decode_fusion`: BITWISE fused kernels for the one-row (decode) hyper-connection.
///
/// Unfused, a row HC is ten encoders (about 190 us on an M2 Pro): narrow residual, grouped RMS, narrow
/// norm, QMV down (53 us), lookup narrow, QMV up (85 us), sigmoid/sum4 epilogue, dense injection GEMV (17 us),
/// lookup float, narrow mixed. Inter-encoder gaps are about 0.3 us, so the time is inside the two row-one QMVs and
/// the injection GEMV, which run far below bandwidth (1.84 MB each in 53/85 us): the down QMV has only 320
/// SIMD groups that each walk 20 dependent chunks; the up QMV has 10240 SIMD groups that each do one or two chunks
/// and a simd_sum, latency-bound. The fused path keeps every floating-point operation and its order:
///
/// - `hcd_down_act_v2` (QMV down + lookup narrow): one threadgroup per output row. Each (SIMD group k, lane L)
///   computes chunk k of lane L with `qmv_pinned_reference`'s verbatim chunk arithmetic into
///   threadgroup memory; SIMD group 0 then adds lane L's chunks from +0 in ascending k exactly as the reference
///   lane loop does, the same simd_sum over the same lane values, round_bf16, and the activation table lookup
///   of that stored value (`hc_decode_lookup_narrow`).
/// - `hcd_up_mix_v2` (QMV up + sigmoid/sum4 epilogue + narrow mixed): one SIMD group per mixed column c computes
///   the four up rows c, c+2560, c+5120, c+7680 with the reference lane/chunk map (the activation slice is loaded
///   once for the four rows), simd_sum and round_bf16 each, stores them as the reference does, and lane 0 runs
///   `hc_sigmoid_sum4`'s verbatim body on those values (same table, exception bitmap, capture), then writes the
///   BF16 narrowing of the stored mixed value (`hc_decode_narrow`) when the caller asks for it.
/// - `hcd_injection_lookup_v2` (dense injection + lookup float): `origami_dense_mv`'s verbatim body (same lane columns,
///   shuffle-down tree, ascending SIMD group partial sum, typed() BF16 boundary), with the K loop unrolled so
///   loads issue early, then `hc_decode_lookup_float` of the stored value. Compiled with PinnedDenseLinear's
///   source and options (safe math, fast functions, function constant 0 = BF16 output).
/// - The entry (narrow residual + grouped RMS + narrow norm) reuses `hc_grouped_rms_fused_v4` (hc_exact_fusion)
///   at one row: its geometry (640 threads per 2560-element group) does not depend on the row count.
///
/// `referenceSource` is compiled after ReferenceKernels' quantized source (preamble: `#pragma clang fp
/// contract(off)`, from_bf16, round_bf16) and FusedHCDecode.handoffSource (hc_decode_key), with MSL 3.1, safe
/// math and precise functions, exactly as FusedHCDecode compiles the current QMV and handoffs.
/// HCDecodeFusionSelfChecks compares every stored word of the fused row with the eight-dispatch row (also under group 32).
enum HCDecodeFusionKernels {
    static let downV2Name = "hcd_down_act_v2", upV2Name = "hcd_up_mix_v2", injectionV2Name = "hcd_injection_lookup_v2"
    /// Engine geometry of the v2 kernels (the fastest exact variant measured on an M2 Pro: 64.9 vs 155.9 us
    /// per HC): down: 20 SIMD groups (one per chunk) x 4 rows per threadgroup; up: 8 mixed columns per SIMD
    /// group, 4 SIMD groups per threadgroup; injection: the unrolled v2 GEMV.
    static let downSIMDGroups = 20, downRowsPerThreadgroup = 4
    static let upColumnsPerSIMDGroup = 8, upSIMDGroupsPerThreadgroup = 4

    static let referenceSource = #"""
        // ---- hc_decode_fusion ------------------------------------------------------------------------------
        // Verbatim FusedHyperConnection hc_bf16.
        inline float hcd_bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
                return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
            return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
        }

            // ---- v2: same arithmetic, fewer instructions per weight ---------------------------------------------
            // Measured on an M2 Pro, a first direct transcription left both projections at ~25-40 GB/s with no geometry sensitivity: they are
            // instruction-issue bound, not bandwidth bound. Per weight the reference re-derives the activation
            // (scalar ushort load + widen), the word address (runtime divides by width/group), and recomputes the
            // activation-only BF16 bias sums for every output row; the up QMV's second chunk runs on 8 of 32 lanes.
            // v2 keeps each lane's operations and order (products, quant/bias sums, fma, ascending chunk sum, the same
            // simd_sum over the same lane values) but: fixed geometry (literal shapes), vector loads of the activation
            // slice once per SIMD group/thread, the activation-only bias sums computed once and reused for every row
            // (same operands, same operations => same value), and (up) the 8-lane second chunk of four rows computed
            // by all 32 lanes at once and moved to its owning lane with simd_shuffle before the ordered add.
            inline void hcd_unpack8(uint4 a, thread float *v) {
                for (uint q = 0; q < 4; ++q) {
                    v[2 * q] = from_bf16(ushort(a[q] & 0xffffu));
                    v[2 * q + 1] = from_bf16(ushort(a[q] >> 16));
                }
            }
            // biasSum of qmv_pinned_reference over 4 * steps consecutive activations (from +0, sequential steps).
            inline float hcd_bias_sum(thread const float *v, uint steps) {
                float biasSum = 0.0f;
                for (uint h = 0; h < steps; ++h)
                    biasSum = biasSum + round_bf16(round_bf16(round_bf16(v[4*h]+v[4*h+1])+v[4*h+2])+v[4*h+3]);
                return biasSum;
            }
            // quantSum + fma of qmv_pinned_reference for 8 activations against one packed word (value t = nibble t).
            inline float hcd_dot8(uint word, thread const float *v, float biasSum, ushort scale, ushort bias) {
                float quantSum = 0.0f;
                for (uint h = 0; h < 2; ++h) {
                    float products[4];
                    for (uint j = 0; j < 4; ++j) products[j] = v[4*h+j] * float((word >> ((4*h+j)*4)) & 15u);
                    quantSum = quantSum + (((products[0]+products[1])+products[2])+products[3]);
                }
                return fma(from_bf16(scale), quantSum, biasSum*from_bf16(bias));
            }
            // As hcd_dot8 for 16 activations against two packed words (values 0-7 from w.x, 8-15 from w.y).
            inline float hcd_dot16(uint2 w, thread const float *v, float biasSum, ushort scale, ushort bias) {
                float quantSum = 0.0f;
                for (uint h = 0; h < 4; ++h) {
                    uint word = h < 2 ? w.x : w.y;
                    float products[4];
                    for (uint j = 0; j < 4; ++j) products[j] = v[4*h+j] * float((word >> (((4*h+j)%8)*4)) & 15u);
                    quantSum = quantSum + (((products[0]+products[1])+products[2])+products[3]);
                }
                return fma(from_bf16(scale), quantSum, biasSum*from_bf16(bias));
            }

            // QMV down (K10240 -> 320, 16 values/lane, group QGROUP) + lookup narrow. Threadgroup = 20 SIMD groups (SIMD group k
            // owns chunk k: lane L reads x[k*512 + L*16, +16)); shape[0] = rows per threadgroup R (320 % R == 0, R <= 8).
            // Each thread loads its 16 activations and bias sum once, then for each of R rows writes its chunk dot; SIMD
            // group s then adds rows s, s+20, ... in ascending chunk order from +0 and simd_sums, as the reference.
            kernel void hcd_down_act_v2(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
                device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
                constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
                device ushort *activation [[buffer(6)]], device const ushort *table [[buffer(7)]],
                uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
                const uint R = shape[0];
                threadgroup float dots[8 * 640];
                const uint lo = sg * 512 + lane * 16;
                float v[16];
                device const uint4 *xv = (device const uint4 *)(x + lo);
                hcd_unpack8(xv[0], v); hcd_unpack8(xv[1], v + 8);
                const float biasSum = hcd_bias_sum(v, 4);
                for (uint r = 0; r < R; ++r) {
                    const uint row = tg * R + r;
                    const uint2 w = *((device const uint2 *)(words + row * 1280 + lo / 8));
                    const uint index = row * (10240 / QGROUP) + lo / QGROUP;
                    dots[r * 640 + sg * 32 + lane] = hcd_dot16(w, v, biasSum, scales[index], biases[index]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint r = sg; r < R; r += groups) {
                    float result = 0.0f;
                    for (uint k = 0; k < 20; ++k) result = result + dots[r * 640 + k * 32 + lane];
                    result = simd_sum(result);
                    if (lane == 0) {
                        const uint row = tg * R + r;
                        const float value = round_bf16(result);
                        out[row] = value;
                        activation[row] = table[hc_decode_key(value)];
                    }
                }
            }

            // QMV up (K320 -> 10240 = 4 x 2560, 8 values/lane, group QGROUP) + hc_sigmoid_sum4 + narrow mixed. Each SIMD group owns
            // shape[2] consecutive mixed columns (cps <= 32, 2560 % cps == 0); shape[0] capture, shape[1] write mixed BF16.
            // Lane L: chunk 0 of every row = x[L*8, +8) (word L); chunk 1 (lanes 0-7 in the reference) of the four rows of
            // a column is computed by all lanes (lane L: row L/8, sub-lane L%8) and shuffled to lane L%8 of row L/8.
            // Lane k < cps keeps the four rounded ups of column first + k and runs the verbatim epilogue for it.
            kernel void hcd_up_mix_v2(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
                device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
                constant uint *shape [[buffer(4)]], device float *rawUp [[buffer(5)]],
                device const float *norm [[buffer(6)]], device const ushort *sigmoid [[buffer(7)]],
                device float *mixed [[buffer(8)]], device float *captured [[buffer(9)]],
                device atomic_uint *exceptions [[buffer(10)]], device ushort *mixedBF16 [[buffer(11)]],
                uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
                const uint cps = shape[2];
                const uint first = (tg * groups + sg) * cps;
                if (first >= 2560) return;
                const uint s = lane & 7u, rb = lane >> 3;
                float v0[8], v1[8];
                device const uint4 *xv = (device const uint4 *)x;
                hcd_unpack8(xv[lane], v0); hcd_unpack8(xv[32 + s], v1);
                const float bias0 = hcd_bias_sum(v0, 2), bias1 = hcd_bias_sum(v1, 2);
                float mine[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint k = 0; k < cps; ++k) {
                    const uint column = first + k;
                    const uint row1 = rb * 2560 + column;
                    // Word w holds values 8w...8w+7, all in group 8w / QGROUP.
                    const uint g1 = row1 * (320 / QGROUP) + (32 + s) * 8 / QGROUP;
                    const float dot1 = hcd_dot8(words[row1 * 40 + 32 + s], v1, bias1, scales[g1], biases[g1]);
                    #pragma clang loop unroll(full)
                    for (uint branch = 0; branch < 4; ++branch) {
                        const uint row = branch * 2560 + column;
                        float result = 0.0f;
                        const uint g0 = row * (320 / QGROUP) + lane * 8 / QGROUP;
                        result = result + hcd_dot8(words[row * 40 + lane], v0, bias0, scales[g0], biases[g0]);
                        const float second = simd_shuffle(dot1, branch * 8 + s);
                        if (lane < 8) result = result + second;
                        const float up = round_bf16(simd_sum(result));
                        if (lane == k) mine[branch] = up;
                    }
                }
                if (lane >= cps) return;
                const uint column = first + lane;
                #pragma clang loop unroll(full)
                for (uint branch = 0; branch < 4; ++branch) rawUp[branch * 2560 + column] = mine[branch];
                // Verbatim hc_sigmoid_sum4 body (index == column at one row; rawUp[source] == mine[branch]).
                const uint index = column;
                const uint token = index / 2560;
                float sum = 0.0f;
                bool repair = false;
                #pragma clang loop unroll(full)
                for (uint branch = 0; branch < 4; ++branch) {
                    const uint source = token * 10240 + branch * 2560 + column;
                    const uint key = as_type<uint>(hcd_bf16(mine[branch])) >> 16;
                    const float weight = as_type<float>(uint(sigmoid[key]) << 16);
                    if (shape[0]) captured[source] = weight;
                    const uint a = as_type<uint>(weight) & 0x7fffffffu, b = as_type<uint>(norm[source]) & 0x7fffffffu;
                    const uint ea = a >> 23, eb = b >> 23;
                    repair |= ea == 255 || eb == 255 ||
                        (a != 0 && b != 0 && (ea == 0 || eb == 0 || ea + eb <= 128));
                    const float product = hcd_bf16(weight * norm[source]);
                    const uint c = as_type<uint>(sum) & 0x7fffffffu, d = as_type<uint>(product) & 0x7fffffffu;
                    const uint ec = c >> 23, ed = d >> 23;
                    repair |= c != 0 && d != 0 && (ec == 0 || ed == 0 || max(ec, ed) <= 7);
                    sum = hcd_bf16(sum + product);
                    repair |= ((as_type<uint>(sum) >> 23) & 255u) == 255;
                }
                const uint absolute = as_type<uint>(sum) & 0x7fffffffu;
                repair |= absolute != 0 && (absolute >> 23) <= 2;
                const float value = hcd_bf16(sum / 4.0f);
                mixed[index] = value;
                if (repair) atomic_fetch_or_explicit(&exceptions[index >> 5], 1u << (index & 31), memory_order_relaxed);
                if (shape[1]) mixedBF16[index] = hc_decode_key(value);
            }
        """# + "\n"

    static let denseSource = #"""
        // ---- hc_decode_fusion (PinnedDenseLinear library and options) ------------------------------------------
        inline ushort hcd_key(float x) {  // verbatim body of FusedHCDecode.handoffSource hc_decode_key
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
            // v2: origami_dense_mv at the literal row-one injection shape (K10240, N4, 8 SIMD groups), K loop fully unrolled so
            // the 40 activation/weight loads per lane issue together; per-lane statement order unchanged.
            kernel void hcd_injection_lookup_v2(device const float *matrix [[buffer(0)]], device const float *vectors [[buffer(1)]],
                device float *output [[buffer(2)]], constant uint *shape [[buffer(3)]],
                device float *injection [[buffer(4)]], device const ushort *table [[buffer(5)]],
                uint3 block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
                const uint k = 10240, n = 4, groups = 8;
                uint index = sg * 128 + lane * 4;
                float sum = 0;
                #pragma clang loop unroll(full)
                for (uint base = 0; base < k; base += groups * 128) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; ++i) {
                        uint column = base + index + i;
                        float coefficient = column < k ? typed(vectors[block.y * k + column]) : 0;
                        float value = column < k ? typed(matrix[block.x * k + column]) : 0;
                        sum += value * coefficient;
                    }
                }
                #pragma clang loop unroll(full)
                for (ushort offset = 16; offset >= 1; offset >>= 1) sum += simd_shuffle_down(sum, offset);
                threadgroup float partials[8];
                if (lane == 0) partials[sg] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0 && lane == 0) {
                    for (uint part = 1; part < groups; ++part) sum += partials[part];
                    float result = typed(sum);
                    output[block.y * n + block.x] = result;
                    injection[block.y * n + block.x] = as_type<float>(uint(table[hcd_key(result)]) << 16);
                }
            }
        """# + "\n"
}
