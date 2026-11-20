// Copyright © 2023-2024 Apple Inc.
// BM64 x BN64 bfloat-fragment variant of the compact BM32 specialization of pinned MLX affine QMM (1f8e74e3).
// MIT license: engine/THIRD_PARTY_NOTICES.md.

/// Prefill QMM for the dense matrices and the grouped experts (rows >= 64), BITWISE. Same arithmetic contract as
/// `qmm_pinned_tiled32_prefetch_partitions`: BF16 activations, affine 4-bit weights at group `QGROUP` (64 or 32), BF16-rounded
/// dequantized weights, one 8x8x8 MMA per ascending K8 block per output element from a zero FP32
/// accumulator, `round_bf16` of the accumulator, and the BF16 partial boundary per split-K partition.
/// What differs is the tile and the operand path; `engine-checks` (`grouped_mma_bf16_boundaries_counters`, also at
/// group 32) compares the outputs bit for bit with the prefetch kernel on the GPU it runs on. Needs bfloat SIMD-group
/// matrices (Apple8, M2 and later; off on M1); on M5 `QuantizedTensorOpsKernel` replaces it under the same name:
/// - BM64 x BN64 x BK32, 128 threads = 4 SIMD groups in a 2 x 2 arrangement, each owning a 32 x 32
///   sub-tile as 4 x 4 `simdgroup_float8x8` accumulators; threadgroup tiles `xs[64 x 40]`, `ws[64 x 40]`
///   (10,240 bytes); grid `N/64 x ceil(M/64) x splitK`.
/// - Register prefetch of the next K32 step (sixteen activations as four ushort4, eight packed bytes as
///   one uint2, one scale/bias pair per thread) and vector threadgroup stores.
/// - Operand fragments are loaded with `simdgroup_load` of `bfloat` 8x8 blocks (the B block transposed)
///   and multiplied directly into the float accumulators: bfloat x bfloat -> float reproduces the
///   float x float chain bit for bit on this GPU family (proved by the check, not assumed).
/// - Exact fast dequantization: when a group's scale and bias are both finite, `scale * q + bias` is never
///   NaN, so the NaN test of `round_bf16` is skipped for the group's eight values (one uniform branch per
///   group per thread); otherwise the full `round_bf16` runs. The low word of each rounded value is zero,
///   so the pack is `(low >> 16) | high`.
/// Requires N % 64 == 0; callers keep rows >= 64 for the tile to pay off and guarantee 8-byte aligned
/// activation and weight bindings (uint2 / ushort4 loads).
enum QuantizedMMABF16Kernel {
    static let name = "qmm_pinned_mma_bf16_64x64_partitions"
    static let threadgroupBytes = 2 * 64 * 40 * 2
    static let source = #"""
        inline uint qmm_mma_round_fast_bits(float v) {
            const uint b = as_type<uint>(v);
            return (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
        }
        kernel void qmm_pinned_mma_bf16_64x64_partitions(
            device const ushort *x [[buffer(0)]], device const uchar *packed [[buffer(1)]],
            device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
            constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
            uint3 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint width = shape[0], outputs = shape[1], rows = shape[2];
            const uint part = tile.z, partition = shape[5];
            const uint kBegin = part * partition, kEnd = kBegin + partition;
            const uint outputBase = part * rows * outputs;
            const uint m = tile.y * 64, n = tile.x * 64, linearThread = simd * 32 + lane;
            threadgroup ushort xs[64 * 40];
            threadgroup ushort ws[64 * 40];
            // Sixteen contiguous activation BF16s per thread (two threads per tile row).
            const uint xm = linearThread / 2, xk = (linearThread % 2) * 16;
            // Sixteen dequantized weights per thread (two threads per output row): eight packed bytes.
            const uint wn = linearThread / 2, wk = (linearThread % 2) * 16;
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint sRow = (simd / 2) * 8, sCol = (simd % 2) * 8;
            const bool rowValid = m + xm < rows;
            simdgroup_float8x8 c[4][4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) { c[i][j].thread_elements()[0] = 0.0f; c[i][j].thread_elements()[1] = 0.0f; }
            }
            // Registers for the next step: sixteen activations (four ushort4), eight packed bytes (one uint2).
            ushort4 xr0 = ushort4(0), xr1 = ushort4(0), xr2 = ushort4(0), xr3 = ushort4(0);
            uint pr0 = 0, pr1 = 0; ushort sr = 0, br = 0;
            {
                device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + kBegin + xk);
                xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                xr2 = rowValid ? xv[2] : ushort4(0); xr3 = rowValid ? xv[3] : ushort4(0);
                const uint metadata = (n + wn) * (width / QGROUP) + kBegin / QGROUP;
                sr = scales[metadata]; br = biases[metadata];
                const uint2 pv = *((device const uint2 *)(packed + (n + wn) * (width / 2) + (kBegin + wk) / 2));
                pr0 = pv.x; pr1 = pv.y;
            }
            for (uint start = kBegin; start < kEnd; start += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                    threadgroup ushort4 *xs4 = (threadgroup ushort4 *)(xs + xm * 40 + xk);
                    xs4[0] = xr0; xs4[1] = xr1; xs4[2] = xr2; xs4[3] = xr3;
                    const float scale = from_bf16(sr), bias = from_bf16(br);
                    const float highScale = scale / 16.0f;
                    threadgroup uint *ws2 = (threadgroup uint *)(ws + wn * 40 + wk);
                    const bool exceptional = ((sr & 0x7f80u) == 0x7f80u) || ((br & 0x7f80u) == 0x7f80u);
                    if (!exceptional) {
                        // Finite scale and bias: scale * q + bias is finite or +-Inf, never NaN, so the
                        // NaN test of round_bf16 is redundant and the rounding is the same bits.
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 8; ++j) {
                            const uchar value = uchar(((j < 4 ? pr0 : pr1) >> (8 * (j % 4))) & 255);
                            ws2[j] = (qmm_mma_round_fast_bits(scale * float(value & 15) + bias) >> 16)
                                | qmm_mma_round_fast_bits(highScale * float(value & 240) + bias);
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 8; ++j) {
                            const uchar value = uchar(((j < 4 ? pr0 : pr1) >> (8 * (j % 4))) & 255);
                            // Match pinned dequantize<T,2,4>; each tile value is rounded
                            // before widening to the FP32 SIMD-matrix accumulator type.
                            const uint low = as_type<uint>(round_bf16(scale * float(value & 15) + bias)) >> 16;
                            const uint high = as_type<uint>(round_bf16(highScale * float(value & 240) + bias)) >> 16;
                            ws2[j] = low | (high << 16);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (start + 32 < kEnd) {
                    device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + start + 32 + xk);
                    xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                    xr2 = rowValid ? xv[2] : ushort4(0); xr3 = rowValid ? xv[3] : ushort4(0);
                    const uint metadata = (n + wn) * (width / QGROUP) + (start + 32) / QGROUP;
                    sr = scales[metadata]; br = biases[metadata];
                    const uint2 pv = *((device const uint2 *)(packed + (n + wn) * (width / 2) + (start + 32 + wk) / 2));
                    pr0 = pv.x; pr1 = pv.y;
                }
                #pragma clang loop unroll(full)
                for (uint kk = 0; kk < 32; kk += 8) {
                    simdgroup_bfloat8x8 a[4], b[4];
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; ++i)
                        simdgroup_load(a[i], (threadgroup const bfloat *)(xs + (sRow + i * 16) * 40 + kk), 40);
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j)
                        simdgroup_load(b[j], (threadgroup const bfloat *)(ws + (sCol + j * 16) * 40 + kk), 40, ulong2(0, 0), true);
                    simdgroup_barrier(mem_flags::mem_none);
                    // Serpentine fragment order, as the pinned tile_matmad; each accumulator still
                    // receives exactly one MMA per K8 block.
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; ++i) {
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 4; ++j) {
                            const uint jj = (i % 2 == 0) ? j : 3 - j;
                            simdgroup_multiply_accumulate(c[i][jj], a[i], b[jj], c[i][jj]);
                        }
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                const uint row = m + sRow + i * 16 + fm;
                if (row < rows) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        const uint col = n + sCol + j * 16 + fn;
                        out[outputBase + row * outputs + col] = round_bf16(c[i][j].thread_elements()[0]);
                        out[outputBase + row * outputs + col + 1] = round_bf16(c[i][j].thread_elements()[1]);
                    }
                }
            }
        }
        """#
}
