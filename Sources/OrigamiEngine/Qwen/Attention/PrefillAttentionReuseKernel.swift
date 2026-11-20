// The 8K prefill fast path (256 queries, 256-7680 keys, no fused prefill): a QK that shares each K tile
// across the 12 heads of a KV head and 8 queries, and a PV that covers all 256 value columns per unit.
// Both write and read the batched stripe layout and keep every output's MMA order (BITWISE, checked
// against the original batched kernels by PrefillAttentionReuseSelfChecks).
enum PrefillAttentionReuseKernel {
    static let source = #"""
        // Eight queries and all twelve heads sharing a KV head reuse Q/K operands.
        // Score addresses and every output's ascending K8 MMA order remain unchanged.
        kernel void origami_qk_gqa_reuse_k16(device const ushort *queries [[buffer(0)]],
            ORIGAMI_KV_ARGS keys [[buffer(1)]], device const uint *ids [[buffer(2)]],
            device const uint *counts [[buffer(3)]], device ushort *scores [[buffer(4)]],
            constant uint *shape [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], queryBase = shape[3] / 3 + group.y * 8;
            uint keyBase = group.x * 128, kv = group.z;
            threadgroup float as[96 * 16], bs[16 * 128];
            threadgroup uint masks[128], active;
            if (tid < 128) {
                uint key = keyBase + tid, bits = 0;
                for (uint q = 0; q < 8; ++q) {
                    uint query = queryBase + q;
                    if (query < shape[0] && key < n && selected(ids + query * shape[2], counts[query], key)) bits |= 1u << q;
                }
                masks[tid] = bits;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                bool any = false;
                for (uint i = lane; i < 128; i += 32) any |= masks[i] != 0;
                bool selectedTile = simd_any(any);
                if (lane == 0) active = selectedTile;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!active) {
                for (uint i = tid; i < 96 * 128; i += 384) {
                    uint row = i / 128, key = keyBase + i % 128;
                    uint query = queryBase + row / 12, head = kv * 12 + row % 12;
                    uint unit = (query / 2) * 6 + head / 4, slot = (query % 2) * 4 + head % 4;
                    if (query < shape[0] && key < n && unit - shape[3] < shape[4]) scores[((unit - shape[3]) * 8 + slot) * n + key] = ushort(0xff7f);
                }
                return;
            }
            simdgroup_float8x8 a, b, c[16];
            for (uint f = 0; f < 16; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 256; d += 16) {
                for (uint i = tid; i < 96 * 16; i += 384) {
                    uint row = i / 16, col = i % 16;
                    uint query = queryBase + row / 12, head = kv * 12 + row % 12;
                    as[i] = query < shape[0] ? round_bf16(0.0625f * load_bf16(queries[(query * 24 + head) * 256 + d + col])) : 0.0f;
                }
                for (uint i = tid; i < 16 * 128; i += 384) {
                    uint row = i / 128, key = keyBase + i % 128;
                    bs[i] = key < n ? load_bf16(origami_key_bits(keys, key * 512 + kv * 256 + d + row)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint inner = 0; inner < 16; inner += 8) {
                    simdgroup_load(a, as + sg * 8 * 16 + inner, 16);
                    for (uint f = 0; f < 16; ++f) {
                        simdgroup_load(b, bs + inner * 128 + f * 8, 128);
                        simdgroup_multiply_accumulate(c[f], a, b, c[f]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // All operand reads joined above. Each SIMD group has its own store stripe.
            for (uint f = 0; f < 16; ++f) {
                simdgroup_store(c[f], bs + sg * 64, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) {
                    uint row = sg * 8 + i / 8, column = f * 8 + i % 8;
                    uint query = queryBase + row / 12, head = kv * 12 + row % 12, key = keyBase + column;
                    uint unit = (query / 2) * 6 + head / 4, slot = (query % 2) * 4 + head % 4;
                    if (query < shape[0] && key < n && unit - shape[3] < shape[4]) {
                        scores[((unit - shape[3]) * 8 + slot) * n + key] = (masks[column] & (1u << (row / 12))) ? store_bf16(bs[sg * 64 + i]) : ushort(0xff7f);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Same logical eight-row unit, absolute K8 order and BF16 boundary as control.
        // All 256 value columns share one probability load and zero-tile decision.
        kernel void origami_pv_full_width(device const ushort *probabilities [[buffer(0)]],
            ORIGAMI_KV_ARGS values [[buffer(1)]], device float *out [[buffer(2)]],
            constant uint *shape [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]]) {
            uint n = shape[1], unit = shape[3] + group.y;
            uint rowBase = (unit / 6) * 2, rows = min(2u, shape[0] - rowBase) * 4;
            uint head = (unit % 6) * 4;
            probabilities += group.y * 8 * n;
            threadgroup float as[64], bs[8 * 256], cs[8 * 64];
            threadgroup uint active;
            simdgroup_float8x8 a, b;
            simdgroup_float8x8 c[4];
            for (uint f = 0; f < 4; ++f) c[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            uint alignedKeys = (n / 16) * 16, tailSteps = n % 16 == 0 ? 0 : 16;
            for (uint step = 0; step < alignedKeys + tailSteps; step += 8) {
                uint key = step < tailSteps ? alignedKeys + step : step - tailSteps;
                if (sg == 0) {
                    bool anyProbability = false;
                    for (uint i = lane; i < 64; i += 32) {
                        uint row = i / 8, col = i % 8;
                        float p = row < rows && key + col < n ? load_bf16(probabilities[row * n + key + col]) : 0.0f;
                        as[i] = p; anyProbability |= p != 0.0f;
                    }
                    bool any = simd_any(anyProbability);
                    if (lane == 0) active = any;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                bool perform = active != 0;
                // Every group member consumes the decision before the next iteration
                // can overwrite it, including a wholly skipped tile.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!perform) continue;
                for (uint i = tid; i < 8 * 256; i += 256) {
                    uint row = i / 256, col = i % 256;
                    bs[i] = key + row < n ? load_bf16(origami_value_bits(values, (key + row) * 512 + (head / 12) * 256 + col)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a, as, 8);
                for (uint f = 0; f < 4; ++f) {
                    simdgroup_load(b, bs + sg * 32 + f * 8, 256);
                    simdgroup_multiply_accumulate(c[f], a, b, c[f]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint f = 0; f < 4; ++f) {
                simdgroup_store(c[f], cs + sg * 64, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) if (i / 8 < rows) {
                    uint row = i / 8, col = sg * 32 + f * 8 + i % 8;
                    out[((rowBase + row / 4) * 24 + head + row % 4) * 256 + col] = round_bf16(cs[sg * 64 + i]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        """#
}
