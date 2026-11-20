import Foundation

/// `hc_exact_fusion`: BITWISE replacements for two groups of small prefill dispatches (split-K reduction and
/// the HC tile entry). Each is compared word for word against the kernels it replaces by
/// HCExactFusionSelfChecks (clean and NaN/Inf/-0/subnormal/3e38 seeded inputs, production and ragged shapes).
///
/// - `qmm_split_reduce_serial` / `qmm_split_reduce_serial4` replace `qmm_pinned_split_reduce` for 2-31
///   partitions: the current kernel gives every output a 32-lane SIMD group in which lane j < laneCount
///   accumulates parts j, j + laneCount, ... with round_bf16 after each add and lane 0 folds the lane totals in
///   lane order; for fewer than 32 partitions that order is fully specified, so one thread per output (or per
///   four consecutive outputs) replays it with the same operations. 32+ partitions use simd_sum and are kept.
///   Compiled into the ReferenceKernels library (same preamble round_bf16, safe math, precise functions).
/// - `hc_grouped_rms_fused_v4` replaces hc_decode_narrow(residual) + pinned_grouped_rms + hc_decode_narrow(norm)
///   for tile HCs: same per-element expressions (hc_decode_key's verbatim body as `hcf_key`), the same thread
///   to element map and in-thread order for the squared sum, the same simd_sum/partials reduction, precise rsqrt.
///   Compiled with PinnedReductions' source and options (safe math, fast functions; no fast function is used).
enum HCExactFusionKernels {
    static let splitReduceName = "qmm_split_reduce_serial"
    static let splitReduceVectorName = "qmm_split_reduce_serial4"
    static let entryName = "hc_grouped_rms_fused_v4"

    static let splitReduceSource = #"""
        // One thread per output: lane-strided BF16 partial sums, then the lane totals in lane order (2-31 splits).
        kernel void qmm_split_reduce_serial(device const float *partials [[buffer(0)]],
            constant uint *shape [[buffer(1)]], device float *out [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            uint count = shape[1] * shape[2], splits = shape[4], lanes = shape[7];
            if (index >= count) return;
            float sum = 0.0f;
            for (uint j = 0; j < lanes; ++j) {
                float total = 0.0f;
                for (uint part = j; part < splits; part += lanes)
                    total = round_bf16(total + partials[part * count + index]);
                sum = round_bf16(sum + total);
            }
            out[index] = round_bf16(sum);
        }

        // One thread per four consecutive outputs (count % 4 == 0), float4 loads/stores; per-element order as above.
        kernel void qmm_split_reduce_serial4(device const float4 *partials [[buffer(0)]],
            constant uint *shape [[buffer(1)]], device float4 *out [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            uint count = shape[1] * shape[2], splits = shape[4], lanes = shape[7], quads = count / 4;
            if (index >= quads) return;
            float4 sum = 0.0f;
            for (uint j = 0; j < lanes; ++j) {
                float4 total = 0.0f;
                for (uint part = j; part < splits; part += lanes) {
                    float4 p = partials[part * quads + index];
                    total = float4(round_bf16(total.x + p.x), round_bf16(total.y + p.y),
                                   round_bf16(total.z + p.z), round_bf16(total.w + p.w));
                }
                sum = float4(round_bf16(sum.x + total.x), round_bf16(sum.y + total.y),
                             round_bf16(sum.z + total.z), round_bf16(sum.w + total.w));
            }
            out[index] = float4(round_bf16(sum.x), round_bf16(sum.y), round_bf16(sum.z), round_bf16(sum.w));
        }
        """# + "\n"

    static let entrySource = #"""
        inline ushort hcf_key(float x) {  // verbatim body of FusedHCDecode.handoffSource hc_decode_key
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }

        // One threadgroup of 640 threads per 2560-element group (rows * 4 groups): each thread narrows its four
        // residual elements to BF16 (writing inputBF16), squares them in order, simd_sum + threadgroup partials,
        // precise rsqrt(mean + eps), then BF16(BF16(x * inv) * weight) and its BF16 bits, with vector stores.
        // Requires group == 4 * threads (2560 = 4 x 640).
        kernel void hc_grouped_rms_fused_v4(device const float4 *x [[buffer(0)]], device const ushort4 *weights [[buffer(1)]],
            device float4 *output [[buffer(2)]], constant uint *shape [[buffer(3)]], constant float *constants [[buffer(4)]],
            device ushort4 *inputBF16 [[buffer(5)]], device ushort4 *outputBF16 [[buffer(6)]],
            uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
            uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simdGroup [[simdgroup_index_in_threadgroup]]) {
            uint width = shape[0], group = shape[1];
            uint q = (row * group) / 4 + tid;                  // this thread's float4 index
            threadgroup float partials[32];
            threadgroup float inverse;
            if (simdGroup == 0) partials[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 v = x[q];
            ushort4 key = ushort4(hcf_key(v.x), hcf_key(v.y), hcf_key(v.z), hcf_key(v.w));
            inputBF16[q] = key;
            float4 value = float4(from_bf16(key.x), from_bf16(key.y), from_bf16(key.z), from_bf16(key.w));
            float total = 0.0f;
            { float s0 = value.x * value.x; total = s0 + total; }
            { float s1 = value.y * value.y; total = s1 + total; }
            { float s2 = value.z * value.z; total = s2 + total; }
            { float s3 = value.w * value.w; total = s3 + total; }
            total = simd_sum(total);
            if (lane == 0) partials[simdGroup] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                float sum = threads == 32 ? total : simd_sum(partials[lane]);
                if (lane == 0) {
                    float mean = sum * constants[0];
                    inverse = metal::precise::rsqrt(mean + constants[1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = inverse;
            ushort4 w = weights[(q * 4 % width) / 4];
            float4 result;
            result.x = round_bf16(round_bf16(value.x * inv) * from_bf16(w.x));
            result.y = round_bf16(round_bf16(value.y * inv) * from_bf16(w.y));
            result.z = round_bf16(round_bf16(value.z * inv) * from_bf16(w.z));
            result.w = round_bf16(round_bf16(value.w * inv) * from_bf16(w.w));
            output[q] = result;
            outputBF16[q] = ushort4(hcf_key(result.x), hcf_key(result.y), hcf_key(result.z), hcf_key(result.w));
        }
        """# + "\n"
}
