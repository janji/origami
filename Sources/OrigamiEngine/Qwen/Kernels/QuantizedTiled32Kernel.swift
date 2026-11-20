// Copyright © 2023-2024 Apple Inc.
// Compact BM32 specialization of pinned MLX affine QMM (1f8e74e3).
// MIT license: engine/THIRD_PARTY_NOTICES.md.

/// QMM tile BM32/BN32/BK32, WM2/WN2: 128 threads, 5,120 bytes of threadgroup memory. Dense split-K (affine 4-bit)
/// and grouped expert prefill (expert formats via QBITS/QFP4). Same exactness contract as `GroupedExpertKernel`:
/// BF16-rounded dequantized weights, the ascending K8 FP32 MMA chain per partition and the BF16 partial boundary, so
/// it is BITWISE equal to the 16 x 32 tile. N and K multiples of 32; M has a guarded tail.
enum QuantizedTiled32Kernel {
    static let source = #"""
        kernel void qmm_pinned_tiled32_partitions(
            device const ushort *x [[buffer(0)]], device const uchar *packed [[buffer(1)]],
            device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
            constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
            uint3 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint width = shape[0], outputs = shape[1], rows = shape[2];
            const uint part = tile.z, partition = shape[5];
            const uint kBegin = part * partition, kEnd = kBegin + partition;
            const uint outputBase = part * rows * outputs;
            const uint m = tile.y * 32, n = tile.x * 32, linearThread = simd * 32 + lane;
            threadgroup ushort xs[32 * 40];
            threadgroup ushort ws[32 * 40];
            // BlockLoader<T,32,32,40,1,128>: eight contiguous input BF16s/thread.
            const uint xm = linearThread / 4, xk = (linearThread % 4) * 8;
            // QuantizedBlockLoader<T,32,32,40,1,128,64,4>: eight BF16s/thread.
            const uint wn = linearThread / 4, wk = (linearThread % 4) * 8;
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint sn = (simd % 2) * 8 + fn, sm = (simd / 2) * 8 + fm;
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
                for (uint j = 0; j < 4; ++j) {
                    const uchar value = packed[(n + wn) * (width / 2) + (start + wk) / 2 + j];
                    ws[wn * 40 + wk + 2 * j] = ushort(as_type<uint>(round_bf16(scale * QVAL(value & 15))) >> 16);
                    ws[wn * 40 + wk + 2 * j + 1] = ushort(as_type<uint>(round_bf16(scale * QVAL(value >> 4))) >> 16);
                }
                #else
                const float highScale = scale / 16.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
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
                        a0.thread_elements()[j] = from_bf16(xs[sm * 40 + kk + fn + j]);
                        a1.thread_elements()[j] = from_bf16(xs[(sm + 16) * 40 + kk + fn + j]);
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
                if (m + sm < rows) {
                    out[outputBase + (m + sm) * outputs + n + sn + j] = round_bf16(c00.thread_elements()[j]);
                    out[outputBase + (m + sm) * outputs + n + sn + 16 + j] = round_bf16(c01.thread_elements()[j]);
                }
                if (m + sm + 16 < rows) {
                    out[outputBase + (m + sm + 16) * outputs + n + sn + j] = round_bf16(c10.thread_elements()[j]);
                    out[outputBase + (m + sm + 16) * outputs + n + sn + 16 + j] = round_bf16(c11.thread_elements()[j]);
                }
            }
        }
        """#
}
