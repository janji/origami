// M5 (Apple10) variant of `QuantizedMMABF16Kernel`: the same name, bindings, grid, dequantization and outputs, with the
// 64 x 64 x 32 tile multiplied by Metal 4 TensorOps (the GPU's matrix units) instead of simdgroup 8 x 8 MMAs.
// Numerically close, not bit-identical (accumulation order); chosen when a library is built, on M5 only (checks force it
// on earlier GPUs, where TensorOps run without matrix units). Copyright © 2023-2024 Apple Inc. (MIT, derived
// from MLX's QMM; engine/THIRD_PARTY_NOTICES.md).
// Affine 4-bit only (no QFP4 or qcode branch); callers select it only for affine-4 dense and expert matrices, with
// N % 64 == 0 and 8-byte aligned bindings. It is compiled alone as Metal language 4.0 (`pipeline`), so every other
// kernel keeps its owner's language version.
import Metal

package enum QuantizedTensorOpsKernel {
    /// Set by checks to compile the TensorOps twin on a GPU without matrix units.
    nonisolated(unsafe) package static var forced = false
    /// M5 (Metal family Apple10) and later; a Mac has one GPU.
    static let matrixUnits: Bool = {
        if #available(macOS 26.0, *) { return MTLCreateSystemDefaultDevice()?.supportsFamily(.apple10) ?? false }
        return false
    }()
    static var enabled: Bool {
        if #available(macOS 26.0, *) { return forced || matrixUnits }
        return false
    }
    /// The 64 x 64 prefill MMA pipeline for this Mac (`QuantizedMMABF16Kernel.name`), compiled in a library of its own
    /// with the owner's `defines` and `options`; only the TensorOps twin raises the language version, to 4.0.
    static func pipeline(device: MTLDevice, defines: String, options: MTLCompileOptions) throws -> MTLComputePipelineState {
        let own = options.copy() as! MTLCompileOptions
        var kernel = QuantizedMMABF16Kernel.source
        if #available(macOS 26.0, *), enabled {
            own.languageVersion = .version4_0
            kernel = source
        }
        let library = try device.makeLibrary(source: defines + ReferenceKernels.quantizedBF16Preamble + kernel, options: own)
        guard let function = library.makeFunction(name: QuantizedMMABF16Kernel.name) else {
            throw EngineError("Missing 64 x 64 MMA kernel")
        }
        return try device.makeComputePipelineState(function: function)
    }
    static let source =
        "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n" + #"""
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
                const bool rowValid = m + xm < rows;
                // C[64 rows x 64 outputs] += A[64 x 32] * W[64 x 32]^T per K32 step, from the threadgroup tiles (row stride 40).
                constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(64, 64, 32, false, true, false,
                    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
                mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<4>> op;
                auto ta = metal::tensor<threadgroup bfloat, metal::dextents<int32_t, 2>, metal::tensor_inline>(
                    (threadgroup bfloat *)xs, metal::dextents<int32_t, 2>(32, 64), metal::array<int32_t, 2>{1, 40});
                auto tw = metal::tensor<threadgroup bfloat, metal::dextents<int32_t, 2>, metal::tensor_inline>(
                    (threadgroup bfloat *)ws, metal::dextents<int32_t, 2>(32, 64), metal::array<int32_t, 2>{1, 40});
                auto c = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tw), float>();
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;
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
                    op.run(ta, tw, c);
                }
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                    if (!c.is_valid_element(i)) continue;
                    const auto index = c.get_multidimensional_index(i);
                    const uint row = m + uint(index[1]), col = n + uint(index[0]);
                    if (row < rows) out[outputBase + row * outputs + col] = round_bf16(c[i]);
                }
            }
            """#
}
