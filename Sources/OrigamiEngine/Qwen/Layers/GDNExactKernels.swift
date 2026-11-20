/// Faster BITWISE replacements for the GDN recurrence (`gdn_reference`) and the preparation convolution
/// (`gdn_conv_normalize`), selected by `gdn_exact_kernels` (on in the release configuration): the
/// recurrence for every `ReferenceKernels.encodeGDNRecurrence` whose q/k bindings are 16-byte aligned
/// (GDNDeviceBlock, GDNBufferRegion, whole-layer decode and prefill; `ReferenceKernels.gdn`
/// keeps `gdn_reference`), the convolution in every FusedGDNPreparation dispatch. Both keep every
/// arithmetic step, rounding, check and lane assignment of the kernels they replace and change only how
/// work is grouped and how operands are loaded. They were compared word for word (outputs, carried state,
/// capture and exception word) in a standalone harness on the development GPU; engine-checks has no
/// model-free parity check of them against the reference kernels.
enum GDNExactKernels {
    static let recurrenceName = "gdn_exact_rows8"
    static let convolutionName = "gdn_conv_exact_vec4"

    /// Recurrence: eight value rows of one value head per SIMD group (the reference carries one). For every
    /// value row the thread <-> Dk mapping (lane*4..+3), the decay multiply, the Kahan order over the four
    /// lanes, the two `simd_sum` trees over the same 32 lane values, the fma/non-fma selection by shape[1],
    /// `round_bf16` by lane 0, the zero write for masked tokens and the unchanged FP32 state are the
    /// reference's; q, k, decay and beta are loaded once per token for the eight rows (float4 q/k loads,
    /// so the caller guarantees 16-byte aligned q/k bindings). Grid: 48 * 128 * 128 / 32 threads.
    static let recurrenceSource = #"""
        kernel void gdn_exact_rows8(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
            device const float *v [[buffer(2)]], device const float *decay [[buffer(3)]],
            device const float *beta [[buffer(4)]], device const float *initial [[buffer(5)]],
            device const uint *mask [[buffer(6)]], constant uint *shape [[buffer(7)]],
            device float *out [[buffer(8)]], uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            const uint group = tid / 32, row0 = group * 8, hv = row0 / 128, dv0 = row0 % 128;
            const uint tokens = shape[0], kvHead = hv / 3;
            const bool contract = shape[1] != 0;
            float state[8][4];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                const uint stateBase = (row0 + r) * 128 + lane * 4;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) state[r][j] = initial[stateBase + j];
            }
            for (uint t = 0; t < tokens; ++t) {
                const uint qBase = t * 16 * 128 + kvHead * 128 + lane * 4;
                const float4 kk = *reinterpret_cast<device const float4 *>(k + qBase);
                const float4 qq = *reinterpret_cast<device const float4 *>(q + qBase);
                const float d = decay[t * 48 + hv], b = beta[t * 48 + hv];
                const uint m = mask[t];
                const uint valueBase = t * 48 * 128 + hv * 128 + dv0;
                if (m) {
                    float memory[8];
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < 8; ++r) {
                        memory[r] = 0.0f;
                        {
                            #pragma clang fp reassociate(off)
                            #pragma clang fp contract(off)
                            float compensation = 0.0f;
                            #pragma clang loop unroll(full)
                            for (uint j = 0; j < 4; ++j) {
                                state[r][j] = state[r][j] * d;
                                float product = state[r][j] * kk[j];
                                float corrected = product - compensation;
                                float next = memory[r] + corrected;
                                compensation = (next - memory[r]) - corrected;
                                memory[r] = next;
                            }
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < 8; ++r) memory[r] = simd_sum(memory[r]);
                    float value[8];
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < 8; ++r) {
                        const float delta = (v[valueBase + r] - memory[r]) * b;
                        value[r] = 0.0f;
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 4; ++j) {
                            state[r][j] = contract ? fma(kk[j], delta, state[r][j]) : state[r][j] + kk[j] * delta;
                            value[r] = contract ? fma(state[r][j], qq[j], value[r]) : value[r] + state[r][j] * qq[j];
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < 8; ++r) value[r] = simd_sum(value[r]);
                    if (lane == 0) {
                        #pragma clang loop unroll(full)
                        for (uint r = 0; r < 8; ++r) out[valueBase + r] = round_bf16(value[r]);
                    }
                } else if (lane == 0) {
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < 8; ++r) out[valueBase + r] = 0.0f;
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                const uint stateBase = (row0 + r) * 128 + lane * 4;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) out[tokens * 48 * 128 + stateBase + j] = state[r][j];
            }
        }
        """#

    /// Preparation convolution: the four ascending taps, `sum = sum + product` from 0.0f, every gdn_check /
    /// gdn_check_product call on the same values, the SiLU table lookup, the capture write, the reads4 L2
    /// total, simd_sum, precise rsqrt and both BF16 roundings of `gdn_conv_normalize`; only the loads and
    /// stores are vectorized (ushort4 / float4 for a lane's four contiguous columns, float4 for the four taps
    /// of a column). Same grid: rows * 80 threadgroups of 32 lanes (one 128-column head each); all bindings
    /// are at offset 0 and 16-byte aligned. Compiled into FusedGDNPreparation's library (same helpers).
    static let convolutionSource = #"""
        kernel void gdn_conv_exact_vec4(device const float *mixed [[buffer(0)]], device const ushort *tail [[buffer(1)]],
            device const float *weights [[buffer(2)]], device const ushort *silu [[buffer(3)]], device float *qkv [[buffer(4)]],
            device float *capture [[buffer(5)]], constant uint *shape [[buffer(6)]], constant float *constants [[buffer(7)]],
            device atomic_uint *exceptions [[buffer(8)]],
            uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            const uint rows = shape[0];
            const uint token = group / 80, head = group % 80, column = head * 128 + lane * 4;
            float4 w[4];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) w[j] = *reinterpret_cast<device const float4 *>(weights + (column + j) * 4);
            float4 window[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                const uint row = token + i;
                if (row < 3) {
                    const ushort4 t = *reinterpret_cast<device const ushort4 *>(tail + row * 10240 + column);
                    window[i] = float4(gdn_from_bf16(t.x), gdn_from_bf16(t.y), gdn_from_bf16(t.z), gdn_from_bf16(t.w));
                } else if (row - 3 < rows) {
                    window[i] = *reinterpret_cast<device const float4 *>(mixed + (row - 3) * 10240 + column);
                } else {
                    window[i] = float4(0.0f);
                }
            }
            float values[4];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) {
                float sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint tap = 0; tap < 4; ++tap) {
                    float value = window[tap][j];
                    float weight = w[j][tap]; gdn_check(value, exceptions); gdn_check(weight, exceptions);
                    gdn_check_product(value, weight, exceptions);
                    float product = value * weight;
                    gdn_check(product, exceptions);
                    sum = sum + product;
                    gdn_check(sum, exceptions);
                }
                values[j] = gdn_from_bf16(silu[gdn_key(sum)]);
                gdn_check(sum, exceptions); gdn_check(values[j], exceptions);
            }
            if (shape[1]) *reinterpret_cast<device float4 *>(capture + token * 10240 + column) = float4(values[0], values[1], values[2], values[3]);
            if (head < 32) {
                float total = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) { float squared = values[j] * values[j]; total = squared + total; }
                total = simd_sum(total);
                float inverse = metal::precise::rsqrt(total + constants[0]);
                uint base = (head < 16 ? 0 : shape[0] * 2048) + token * 2048 + (head % 16) * 128 + lane * 4;
                float4 result;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    float normalized = gdn_bf16(values[j] * inverse);
                    result[j] = head < 16 ? gdn_bf16(normalized * constants[1]) : normalized;
                }
                *reinterpret_cast<device float4 *>(qkv + base) = result;
            } else {
                uint base = shape[0] * 4096 + token * 6144 + (head - 32) * 128 + lane * 4;
                *reinterpret_cast<device float4 *>(qkv + base) = float4(values[0], values[1], values[2], values[3]);
            }
        }
        """#
}
