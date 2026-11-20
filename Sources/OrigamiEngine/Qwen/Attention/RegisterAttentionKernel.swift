// `register_qsa_decode`: one-dispatch decode attention. The original 1025...32768-key path keeps its
// BF16 partials in registers. The long variants retain the already charged partial buffers and the
// same partition/reduction schedule; their numerical equivalence still needs runtime validation.
enum RegisterAttentionKernel {
    static let source =
        shortSource + longSource(partitions: 512, maximumKeys: 65536)
        + longSource(partitions: 1024, maximumKeys: 262144)
    private static let shortSource = #"""
        // Sparse partition masks replace per-key searches; paged K/V use explicit per-key page pointers.
        // One threadgroup/head, one complete attention dispatch. A bounded selection
        // bitset replaces binary searches/global bucket construction. Each SIMDgroup
        // computes the same modulo-128/256 partitions in ascending key order, keeps
        // their original BF16 partials packed in registers, and performs the unchanged
        // final max/sum/output reduction. No device partial buffers or spin barriers.
        constant uint PARTITIONS [[function_constant(0)]];
        kernel void origami_sdpa_register_fused(
            device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], ORIGAMI_KV_ARGS values [[buffer(2)]],
            device const uint *ids [[buffer(3)]], device const uint *counts [[buffer(4)]],
            device float *out [[buffer(5)]], constant uint *shape [[buffer(6)]],
            uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            threadgroup atomic_uint membership[1024];
            threadgroup float maxima[256], sums[256], transpose[1024];
            atomic_store_explicit(membership + tid, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < min(counts[0], shape[2]); i += 1024) {
                uint key = ids[i];
                if (key < shape[1] && key < 32768)
                    atomic_fetch_or_explicit(membership + (key % PARTITIONS) * 4 + (key / PARTITIONS) / 32, 1u << ((key / PARTITIONS) % 32), memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float q[8];
            uint4 partial[8];
            uint kv = head / 12;
            for (uint j = 0; j < 8; ++j)
                q[j] = 0.0625f * load_bf16(queries[head * 256 + lane * 8 + j]);
            #pragma unroll
            for (uint b = 0; b < PARTITIONS / 32; ++b) {
                uint block = sg + 32 * b;
                float maximum = -FLT_MAX, sum = 0, o[8] = {0};
                for (uint word = 0; word < 4; ++word) {
                    uint mask = lane == 0 ? atomic_load_explicit(membership + block * 4 + word, memory_order_relaxed) : 0;
                    mask = simd_broadcast_first(mask);
                    while (mask != 0) {
                        uint key = block + (word * 32 + ctz(mask)) * PARTITIONS;
                        mask &= mask - 1;
                        // Resolve an indirect page once per selected key, rather than
                        // repeating page/header/address arithmetic for all 8 elements.
                        #if ORIGAMI_PAGED_KV
                        uint oldK = keys.offset[0], oldV = values.offset[0];
                        device const ushort *kp = key < oldK ? keys.keys[key / 256] + (key % 256) * 512
                            : keys.newKeys + (key - oldK) * 512;
                        device const ushort *vp = key < oldV ? values.values[key / 256] + (key % 256) * 512
                            : values.newValues + (key - oldV) * 512;
                        #else
                        device const ushort *kp = keys + key * 512;
                        device const ushort *vp = values + key * 512;
                        #endif
                        kp += kv * 256 + lane * 8; vp += kv * 256 + lane * 8;
                        float score = 0;
                        for (uint j = 0; j < 8; ++j) score += q[j] * load_bf16(kp[j]);
                        score = simd_sum(score);
                        float next = max(maximum, score), factor = fast::exp(maximum - next), e = fast::exp(score - next);
                        maximum = next; sum = sum * factor + e;
                        for (uint j = 0; j < 8; ++j) o[j] = o[j] * factor + e * load_bf16(vp[j]);
                    }
                }
                if (lane == 0) { maxima[block] = maximum; sums[block] = sum; }
                uint4 packed = uint4(0);
                for (uint j = 0; j < 8; ++j)
                    packed[j / 2] |= uint(store_bf16(o[j])) << (16 * (j % 2));
                partial[b] = packed;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maximum = -FLT_MAX, sum = 0;
            for (uint b = 0; b < PARTITIONS / 32; ++b)
                maximum = max(maximum, maxima[lane + 32 * b]);
            maximum = simd_max(maximum);
            for (uint b = 0; b < PARTITIONS / 32; ++b)
                sum += fast::exp(maxima[lane + 32 * b] - maximum) * sums[lane + 32 * b];
            sum = simd_sum(sum);
            float o[8] = {0};
            #pragma unroll
            for (uint b = 0; b < PARTITIONS / 32; ++b) {
                float factor = fast::exp(maxima[sg + 32 * b] - maximum);
                for (uint j = 0; j < 8; ++j) {
                    ushort bits = ushort(partial[b][j / 2] >> (16 * (j % 2)));
                    o[j] += factor * load_bf16(bits);
                }
            }
            for (uint j = 0; j < 8; ++j) {
                transpose[lane * 32 + sg] = o[j];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[j] = simd_sum(transpose[sg * 32 + lane]);
                o[j] = sum == 0 ? o[j] : o[j] / sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) for (uint j = 0; j < 8; ++j)
                out[head * 256 + sg * 8 + j] = round_bf16(o[j]);
        }

        """#

    /// Long contexts cannot keep 512/1024 BF16 partials in each thread's registers economically.
    /// Keep the caller's existing charged partial buffers, but compute and consume them within one head-owned
    /// threadgroup. No cross-threadgroup dependency or spin barrier. The selection bitmap (at most 32 KiB)
    /// becomes the finish transpose after every partition has completed its reads and device writes.
    /// Partition traversal, BF16 partial stores and the finish loop/reduction order match the two-pass kernels.
    private static func longSource(partitions: Int, maximumKeys: Int) -> String {
        let wordsPerPartition = maximumKeys / partitions / 32
        return #"""
            kernel void origami_sdpa_long_fused_\#(partitions)(
                device const ushort *queries [[buffer(0)]],
                ORIGAMI_KV_ARGS keys [[buffer(1)]], ORIGAMI_KV_ARGS values [[buffer(2)]],
                device const uint *ids [[buffer(3)]], device const uint *counts [[buffer(4)]],
                device ushort *partials [[buffer(5)]], device float *sums [[buffer(6)]], device float *maxima [[buffer(7)]],
                device float *out [[buffer(8)]], constant uint *shape [[buffer(9)]],
                uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
                const uint partitions = \#(partitions), words = \#(wordsPerPartition);
                threadgroup uint shared[\#(maximumKeys / 32)];
                threadgroup atomic_uint *membership = (threadgroup atomic_uint *)shared;
                for (uint i = tid; i < \#(maximumKeys / 32); i += 1024)
                    atomic_store_explicit(membership + i, 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < min(counts[0], shape[2]); i += 1024) {
                    uint key = ids[i];
                    if (key < shape[1] && key < \#(maximumKeys))
                        atomic_fetch_or_explicit(membership + (key % partitions) * words + (key / partitions) / 32,
                            1u << ((key / partitions) % 32), memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float q[8];
                uint kv = head / 12, base = head * partitions;
                for (uint j = 0; j < 8; ++j)
                    q[j] = 0.0625f * load_bf16(queries[head * 256 + lane * 8 + j]);
                for (uint b = 0; b < partitions / 32; ++b) {
                    uint block = sg + 32 * b;
                    float maximum = -FLT_MAX, sum = 0, o[8] = {0};
                    for (uint word = 0; word < words; ++word) {
                        uint mask = lane == 0 ? atomic_load_explicit(membership + block * words + word, memory_order_relaxed) : 0;
                        mask = simd_broadcast_first(mask);
                        while (mask != 0) {
                            uint key = block + (word * 32 + ctz(mask)) * partitions;
                            mask &= mask - 1;
                            #if ORIGAMI_PAGED_KV
                            uint oldK = keys.offset[0], oldV = values.offset[0];
                            device const ushort *kp = key < oldK ? keys.keys[key / 256] + (key % 256) * 512
                                : keys.newKeys + (key - oldK) * 512;
                            device const ushort *vp = key < oldV ? values.values[key / 256] + (key % 256) * 512
                                : values.newValues + (key - oldV) * 512;
                            #else
                            device const ushort *kp = keys + key * 512;
                            device const ushort *vp = values + key * 512;
                            #endif
                            kp += kv * 256 + lane * 8; vp += kv * 256 + lane * 8;
                            float score = 0;
                            for (uint j = 0; j < 8; ++j) score += q[j] * load_bf16(kp[j]);
                            score = simd_sum(score);
                            float next = max(maximum, score), factor = fast::exp(maximum - next), e = fast::exp(score - next);
                            maximum = next; sum = sum * factor + e;
                            for (uint j = 0; j < 8; ++j) o[j] = o[j] * factor + e * load_bf16(vp[j]);
                        }
                    }
                    if (lane == 0) { maxima[base + block] = maximum; sums[base + block] = sum; }
                    for (uint j = 0; j < 8; ++j)
                        partials[(base + block) * 256 + lane * 8 + j] = store_bf16(o[j]);
                }
                // Every reader of the bitmap has finished; all device partials are visible to this same group.
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                threadgroup float *transpose = (threadgroup float *)shared;
                float maximum = -FLT_MAX, sum = 0, o[8] = {0};
                for (uint b = 0; b < partitions / 32; ++b)
                    maximum = max(maximum, maxima[base + lane + 32 * b]);
                maximum = simd_max(maximum);
                for (uint b = 0; b < partitions / 32; ++b)
                    sum += fast::exp(maxima[base + lane + 32 * b] - maximum) * sums[base + lane + 32 * b];
                sum = simd_sum(sum);
                for (uint b = 0; b < partitions / 32; ++b) {
                    float factor = fast::exp(maxima[base + sg + 32 * b] - maximum);
                    for (uint j = 0; j < 8; ++j)
                        o[j] += factor * load_bf16(partials[(base + sg + 32 * b) * 256 + lane * 8 + j]);
                }
                for (uint j = 0; j < 8; ++j) {
                    transpose[lane * 32 + sg] = o[j];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    o[j] = simd_sum(transpose[sg * 32 + lane]);
                    o[j] = sum == 0 ? o[j] : o[j] / sum;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0) for (uint j = 0; j < 8; ++j)
                    out[head * 256 + sg * 8 + j] = round_bf16(o[j]);
            }

            """#
    }
}
