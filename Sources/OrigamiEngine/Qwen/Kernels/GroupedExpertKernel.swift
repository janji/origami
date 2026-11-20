// Copyright © 2023-2024 Apple Inc.
// Compact specialization of MLX quantized.h / quantized_utils.h / steel/gemm
// at 1f8e74e3f12f31365464a6867c6579f0e9b29d85. MIT license: engine/THIRD_PARTY_NOTICES.md.

/// QMM tile BM16/BN32/BK32, WM1/WN2: 64 threads, 3,840 bytes of threadgroup memory (BF16 x 16x40 and dequantized
/// w 32x40; stride 40 pads the rows). Two specializations share loaders and MMA: `qmm_grouped_tiled_16_32_32`
/// (grouped expert prefill, one accumulator across K, expert formats via QBITS/QFP4) and `qmm_pinned_tiled_partitions`
/// (dense split-K QMM, affine 4-bit, into the caller's partials). Exactness: weights are dequantized and BF16-rounded
/// before the FP32 8x8 MMAs, which advance K in ascending 8-steps in MLX's serpentine fragment order; outputs are
/// BF16-rounded per partition, BITWISE equal to `qmm_pinned_simd_partitions`. Callers keep N and K multiples of 32
/// (N rows are not bounds-checked); only M has a guarded tail. M1 runs it unchanged (FP32 fragments).
enum GroupedExpertKernel {
    static let source = #"""
        // Adapted from Apple's MIT-licensed pinned gather-QMM loaders and BlockMMA.
        // QuantizedBlockLoader reads eight packed bytes and one metadata pair per
        // thread. BlockMMA retains four fragments and uses the same K8 progression.
        template <bool SplitK>
        [[kernel]] void qmm_tiled_16_32_32(
            device const ushort *x [[buffer(0)]], device const uchar *packed [[buffer(1)]],
            device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
            constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
            uint3 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint width = shape[0], outputs = shape[1], rows = shape[2];
            const uint part = SplitK ? tile.z : 0, partition = SplitK ? shape[5] : width;
            const uint kBegin = part * partition, kEnd = kBegin + partition;
            const uint outputBase = part * rows * outputs;
            const uint m = tile.y * 16, n = tile.x * 32, linearThread = simd * 32 + lane;
            threadgroup ushort xs[16 * 40];
            threadgroup ushort ws[32 * 40];
            // BlockLoader<T,16,32,40,1,64>: eight contiguous input BF16s/thread.
            const uint xm = linearThread / 4, xk = (linearThread % 4) * 8;
            // QuantizedBlockLoader<T,32,32,40,1,64,64,4>: sixteen BF16s/thread.
            const uint wn = linearThread / 2, wk = (linearThread % 2) * 16;
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint sn = simd * 8 + fn;
            simdgroup_float8x8 a0, a1, b0, b1, c00, c01, c10, c11;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; ++j) {
                c00.thread_elements()[j] = 0.0f; c01.thread_elements()[j] = 0.0f;
                c10.thread_elements()[j] = 0.0f; c11.thread_elements()[j] = 0.0f;
            }
            for (uint start = kBegin; start < kEnd; start += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j)
                    xs[xm * 40 + xk + j] = m + xm < rows ? x[(m + xm) * width + start + xk + j] : 0;
                const uint metadata = (n + wn) * (width / QGROUP) + (start + wk) / QGROUP;
                const float scale = QSCALE(scales, biases, metadata), bias = QBIAS(biases, metadata);
                #if QFP4
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    const uchar value = packed[(n + wn) * (width / 2) + (start + wk) / 2 + j];
                    ws[wn * 40 + wk + 2 * j] = ushort(as_type<uint>(round_bf16(scale * QVAL(value & 15))) >> 16);
                    ws[wn * 40 + wk + 2 * j + 1] = ushort(as_type<uint>(round_bf16(scale * QVAL(value >> 4))) >> 16);
                }
                #else
                const float highScale = scale / 16.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    const uchar value = packed[(n + wn) * (width / 2) + (start + wk) / 2 + j];
                    // Match pinned dequantize<T,2,4>; each tile value is rounded
                    // before widening to the FP32 SIMD-matrix accumulator type.
                    ws[wn * 40 + wk + 2 * j] = ushort(as_type<uint>(round_bf16(scale * float(value & 15) + bias)) >> 16);
                    ws[wn * 40 + wk + 2 * j + 1] = ushort(as_type<uint>(round_bf16(highScale * float(value & 240) + bias)) >> 16);
                }
                #endif
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (uint kk = 0; kk < 32; kk += 8) {
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 2; ++j) {
                        a0.thread_elements()[j] = from_bf16(xs[fm * 40 + kk + fn + j]);
                        a1.thread_elements()[j] = from_bf16(xs[(fm + 8) * 40 + kk + fn + j]);
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 2; ++j) {
                        b0.thread_elements()[j] = from_bf16(ws[(sn + j) * 40 + kk + fm]);
                        b1.thread_elements()[j] = from_bf16(ws[(sn + 16 + j) * 40 + kk + fm]);
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    // Pinned tile_matmad uses a serpentine fragment order.
                    simdgroup_multiply_accumulate(c00, a0, b0, c00);
                    simdgroup_multiply_accumulate(c01, a0, b1, c01);
                    simdgroup_multiply_accumulate(c11, a1, b1, c11);
                    simdgroup_multiply_accumulate(c10, a1, b0, c10);
                }
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; ++j) {
                if (m + fm < rows) {
                    out[outputBase + (m + fm) * outputs + n + sn + j] = round_bf16(c00.thread_elements()[j]);
                    out[outputBase + (m + fm) * outputs + n + sn + 16 + j] = round_bf16(c01.thread_elements()[j]);
                }
                if (m + fm + 8 < rows) {
                    out[outputBase + (m + fm + 8) * outputs + n + sn + j] = round_bf16(c10.thread_elements()[j]);
                    out[outputBase + (m + fm + 8) * outputs + n + sn + 16 + j] = round_bf16(c11.thread_elements()[j]);
                }
            }
        }
        template [[host_name("qmm_grouped_tiled_16_32_32")]] [[kernel]]
            decltype(qmm_tiled_16_32_32<false>) qmm_tiled_16_32_32<false>;
        template [[host_name("qmm_pinned_tiled_partitions")]] [[kernel]]
            decltype(qmm_tiled_16_32_32<true>) qmm_tiled_16_32_32<true>;
        """#
}
