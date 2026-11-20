// Copyright © 2023-2024 Apple Inc.
// Load-scheduling variant of the compact BM32 specialization of pinned MLX affine QMM (1f8e74e3).
// MIT license: engine/THIRD_PARTY_NOTICES.md.

/// Same contract as `qmm_pinned_tiled32_partitions`: BF16 affine4/group64, BM32/BN32, WM2/WN2,
/// 128 threads, the same simdgroup fragments, BF16-rounded dequantized weights, the ascending K8
/// MMA chain per output element and the BF16 partial boundary per split-K partition. Only load
/// scheduling differs: the next K32 step's activations (two ushort4) and packed weights (one uint)
/// are issued into registers before this step's MMAs, and threadgroup stores are vectorized (four
/// activations, two dequantized weights per store). Measured 1.10-1.18x faster on 256-row prefill shapes on an
/// M2 Pro, bit-identical. Affine 4-bit only (no QFP4 or qcode branch; callers fall back to the BM32 kernel).
/// Callers guarantee 8-byte aligned activation and weight bindings.
enum QuantizedTiled32PrefetchKernel {
    static let source = #"""
        kernel void qmm_pinned_tiled32_prefetch_partitions(
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
            const bool rowValid = m + xm < rows;
            simdgroup_float8x8 a0, a1, b0, b1, c00, c01, c10, c11;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; ++j) {
                c00.thread_elements()[j] = 0.0f; c01.thread_elements()[j] = 0.0f;
                c10.thread_elements()[j] = 0.0f; c11.thread_elements()[j] = 0.0f;
            }
            // Registers for the next step: eight activations (two ushort4) and four packed bytes (one uint).
            ushort4 xr0, xr1; uint pr; ushort sr, br;
            {
                device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + kBegin + xk);
                xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                const uint metadata = (n + wn) * (width / QGROUP) + kBegin / QGROUP;
                sr = scales[metadata]; br = biases[metadata];
                pr = *((device const uint *)(packed + (n + wn) * (width / 2) + (kBegin + wk) / 2));
            }
            threadgroup ushort4 *xs4 = (threadgroup ushort4 *)xs;
            threadgroup uint *ws2 = (threadgroup uint *)ws;
            for (uint start = kBegin; start < kEnd; start += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                xs4[(xm * 40 + xk) / 4] = xr0; xs4[(xm * 40 + xk) / 4 + 1] = xr1;
                const float scale = from_bf16(sr), bias = from_bf16(br);
                const float highScale = scale / 16.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    const uchar value = uchar((pr >> (8 * j)) & 255);
                    // Match pinned dequantize<T,2,4>; each tile value is rounded
                    // before widening to the FP32 SIMD-matrix accumulator type.
                    const uint low = as_type<uint>(round_bf16(scale * float(value & 15) + bias)) >> 16;
                    const uint high = as_type<uint>(round_bf16(highScale * float(value & 240) + bias)) >> 16;
                    ws2[(wn * 40 + wk) / 2 + j] = low | (high << 16);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (start + 32 < kEnd) {
                    device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + start + 32 + xk);
                    xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                    const uint metadata = (n + wn) * (width / QGROUP) + (start + 32) / QGROUP;
                    sr = scales[metadata]; br = biases[metadata];
                    pr = *((device const uint *)(packed + (n + wn) * (width / 2) + (start + 32 + wk) / 2));
                }
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
        // Gate and up of one grouped expert tile in one threadgroup, then the SiLU-table product (BITWISE): each output
        // element is computed by exactly the steps of `qmm_pinned_tiled32_prefetch_partitions` (one partition, the same
        // loads, dequantization, fragment loads and serpentine MMA chain from a zero accumulator, `round_bf16`), once with
        // the gate's weights and once with the up's, sharing only the activation tile; the epilogue is
        // `expert_table_silu_product`'s arithmetic on those two values. Bindings: 1-3 gate, 6-8 up, 9 the table, 5 the
        // BF16 product [rows][outputs]. Grid N/32 x ceil(M/32); 128 threads.
        kernel void qmm_gate_up_tiled32_prefetch_product(
            device const ushort *x [[buffer(0)]], device const uchar *packedG [[buffer(1)]],
            device const ushort *scalesG [[buffer(2)]], device const ushort *biasesG [[buffer(3)]],
            constant uint *shape [[buffer(4)]], device ushort *product [[buffer(5)]],
            device const uchar *packedU [[buffer(6)]], device const ushort *scalesU [[buffer(7)]],
            device const ushort *biasesU [[buffer(8)]], device const ushort *silu [[buffer(9)]],
            uint3 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            const uint width = shape[0], outputs = shape[1], rows = shape[2];
            const uint kBegin = 0, kEnd = width;
            const uint m = tile.y * 32, n = tile.x * 32, linearThread = simd * 32 + lane;
            threadgroup ushort xs[32 * 40];
            threadgroup ushort wg[32 * 40];
            threadgroup ushort wu[32 * 40];
            const uint xm = linearThread / 4, xk = (linearThread % 4) * 8;
            const uint wn = linearThread / 4, wk = (linearThread % 4) * 8;
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint sn = (simd % 2) * 8 + fn, sm = (simd / 2) * 8 + fm;
            const bool rowValid = m + xm < rows;
            simdgroup_float8x8 a0, a1, b0, b1, g00, g01, g10, g11, u00, u01, u10, u11;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; ++j) {
                g00.thread_elements()[j] = 0.0f; g01.thread_elements()[j] = 0.0f;
                g10.thread_elements()[j] = 0.0f; g11.thread_elements()[j] = 0.0f;
                u00.thread_elements()[j] = 0.0f; u01.thread_elements()[j] = 0.0f;
                u10.thread_elements()[j] = 0.0f; u11.thread_elements()[j] = 0.0f;
            }
            ushort4 xr0, xr1; uint prG, prU; ushort srG, brG, srU, brU;
            {
                device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + kBegin + xk);
                xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                const uint metadata = (n + wn) * (width / QGROUP) + kBegin / QGROUP;
                srG = scalesG[metadata]; brG = biasesG[metadata];
                srU = scalesU[metadata]; brU = biasesU[metadata];
                prG = *((device const uint *)(packedG + (n + wn) * (width / 2) + (kBegin + wk) / 2));
                prU = *((device const uint *)(packedU + (n + wn) * (width / 2) + (kBegin + wk) / 2));
            }
            threadgroup ushort4 *xs4 = (threadgroup ushort4 *)xs;
            threadgroup uint *wg2 = (threadgroup uint *)wg;
            threadgroup uint *wu2 = (threadgroup uint *)wu;
            for (uint start = kBegin; start < kEnd; start += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                xs4[(xm * 40 + xk) / 4] = xr0; xs4[(xm * 40 + xk) / 4 + 1] = xr1;
                {
                    const float scale = from_bf16(srG), bias = from_bf16(brG);
                    const float highScale = scale / 16.0f;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        const uchar value = uchar((prG >> (8 * j)) & 255);
                        const uint low = as_type<uint>(round_bf16(scale * float(value & 15) + bias)) >> 16;
                        const uint high = as_type<uint>(round_bf16(highScale * float(value & 240) + bias)) >> 16;
                        wg2[(wn * 40 + wk) / 2 + j] = low | (high << 16);
                    }
                }
                {
                    const float scale = from_bf16(srU), bias = from_bf16(brU);
                    const float highScale = scale / 16.0f;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        const uchar value = uchar((prU >> (8 * j)) & 255);
                        const uint low = as_type<uint>(round_bf16(scale * float(value & 15) + bias)) >> 16;
                        const uint high = as_type<uint>(round_bf16(highScale * float(value & 240) + bias)) >> 16;
                        wu2[(wn * 40 + wk) / 2 + j] = low | (high << 16);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (start + 32 < kEnd) {
                    device const ushort4 *xv = (device const ushort4 *)(x + (m + xm) * width + start + 32 + xk);
                    xr0 = rowValid ? xv[0] : ushort4(0); xr1 = rowValid ? xv[1] : ushort4(0);
                    const uint metadata = (n + wn) * (width / QGROUP) + (start + 32) / QGROUP;
                    srG = scalesG[metadata]; brG = biasesG[metadata];
                    srU = scalesU[metadata]; brU = biasesU[metadata];
                    prG = *((device const uint *)(packedG + (n + wn) * (width / 2) + (start + 32 + wk) / 2));
                    prU = *((device const uint *)(packedU + (n + wn) * (width / 2) + (start + 32 + wk) / 2));
                }
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
                        b0.thread_elements()[j] = from_bf16(wg[(sn + j) * 40 + kk + fm]);
                        b1.thread_elements()[j] = from_bf16(wg[(sn + 16 + j) * 40 + kk + fm]);
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(g00, a0, b0, g00);
                    simdgroup_multiply_accumulate(g01, a0, b1, g01);
                    simdgroup_multiply_accumulate(g11, a1, b1, g11);
                    simdgroup_multiply_accumulate(g10, a1, b0, g10);
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 2; ++j) {
                        b0.thread_elements()[j] = from_bf16(wu[(sn + j) * 40 + kk + fm]);
                        b1.thread_elements()[j] = from_bf16(wu[(sn + 16 + j) * 40 + kk + fm]);
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(u00, a0, b0, u00);
                    simdgroup_multiply_accumulate(u01, a0, b1, u01);
                    simdgroup_multiply_accumulate(u11, a1, b1, u11);
                    simdgroup_multiply_accumulate(u10, a1, b0, u10);
                }
            }
            // expert_table_silu_product on the rounded gate and up values.
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; ++j) {
                const float g[4] = {round_bf16(g00.thread_elements()[j]), round_bf16(g01.thread_elements()[j]),
                    round_bf16(g10.thread_elements()[j]), round_bf16(g11.thread_elements()[j])};
                const float u[4] = {round_bf16(u00.thread_elements()[j]), round_bf16(u01.thread_elements()[j]),
                    round_bf16(u10.thread_elements()[j]), round_bf16(u11.thread_elements()[j])};
                ushort p[4];
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) {
                    const float activated = from_bf16(silu[as_type<uint>(round_bf16(g[q])) >> 16]);
                    p[q] = ushort(as_type<uint>(round_bf16(activated * u[q])) >> 16);
                }
                if (m + sm < rows) {
                    product[(m + sm) * outputs + n + sn + j] = p[0];
                    product[(m + sm) * outputs + n + sn + 16 + j] = p[1];
                }
                if (m + sm + 16 < rows) {
                    product[(m + sm + 16) * outputs + n + sn + j] = p[2];
                    product[(m + sm + 16) * outputs + n + sn + 16 + j] = p[3];
                }
            }
        }
        """#
}
