import Foundation
import Metal

/// The element-wise and mixing kernels the family engines (GLM-5.3-Flash, DeepSeek-V4.1-Flash) share, in f32:
/// norms, hyper-connection mixing (mHC with Sinkhorn), SwiGLU with a clamp, the KDA recurrence (a gated delta rule
/// with one decay per key channel), the latent (MLA) attention over a selection of cached tokens, and the DSA
/// indexer's pooled keys, scores and top-k. Matrices are `GGMLMatrixKernels`. Written from llama.cpp's
/// glm5-next/deepseek41 graphs (MIT), whose numerics they follow; every kernel takes a row count so decode
/// (one row) and prefill (many) are the same code. Exactness: f32 with safe math, deterministic for a given Mac and
/// macOS, not bit-identical to llama.cpp (the checks compare them with CPU references to about 2e-4).
package final class FamilyKernels {
    let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    /// The prefill benchmark's profile (`FamilyGPUProfile`): given each dispatch's encoder and kernel name, the encoder
    /// to encode it into.
    var willDispatch: ((MTLComputeCommandEncoder, String) throws -> MTLComputeCommandEncoder)?

    package init(device: MTLDevice) throws {
        self.device = device
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        library = try device.makeLibrary(source: Self.source, options: options)
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] { return cached }
        guard let function = library.makeFunction(name: name) else { throw EngineError("Missing family kernel \(name)") }
        let state = try device.makeComputePipelineState(function: function)
        pipelines[name] = state
        return state
    }

    /// A buffer and a byte offset in it.
    package struct Ref {
        package let buffer: MTLBuffer
        package let offset: Int
        package init(_ buffer: MTLBuffer, _ offset: Int = 0) {
            self.buffer = buffer
            self.offset = offset
        }
    }

    /// Encodes `name` over `grid` threadgroups of `group` threads; `buffers` bind in order, then `constants` as
    /// one packed struct of 32-bit values (ints as bit patterns of UInt32, floats as Float).
    package func dispatch(
        _ encoder: MTLComputeCommandEncoder, _ name: String, _ buffers: [Ref], ints: [Int32] = [], floats: [Float] = [],
        grid: MTLSize, group: MTLSize
    ) throws {
        let encoder = try willDispatch?(encoder, name) ?? encoder
        let state = try pipeline(name)
        guard group.width * group.height * group.depth <= state.maxTotalThreadsPerThreadgroup else {
            throw EngineError(
                "Family kernel \(name) takes at most \(state.maxTotalThreadsPerThreadgroup) threads a group, not \(group.width * group.height * group.depth)"
            )
        }
        encoder.setComputePipelineState(state)
        for (i, ref) in buffers.enumerated() { encoder.setBuffer(ref.buffer, offset: ref.offset, index: i) }
        var words = ints.map { UInt32(bitPattern: $0) } + floats.map(\.bitPattern)
        if words.isEmpty { words = [0] }
        encoder.setBytes(&words, length: words.count * 4, index: buffers.count)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
    }

    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;

        // Sum over a threadgroup of up to 1024 threads (32 SIMD groups); every thread gets the total.
        inline float group_sum(float v, threadgroup float *scratch, uint tid, uint threads) {
            v = simd_sum(v);
            uint lane = tid % 32, warp = tid / 32, warps = (threads + 31) / 32;
            if (lane == 0) scratch[warp] = v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0.0f;
            for (uint w = 0; w < warps; ++w) total += scratch[w];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return total;
        }
        inline float group_max(float v, threadgroup float *scratch, uint tid, uint threads) {
            v = simd_max(v);
            uint lane = tid % 32, warp = tid / 32, warps = (threads + 31) / 32;
            if (lane == 0) scratch[warp] = v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m = -INFINITY;
            for (uint w = 0; w < warps; ++w) m = max(m, scratch[w]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return m;
        }

        struct NormArgs { int n; int hasWeight; float eps; };
        // RMSNorm of each row (one threadgroup per row): x * rsqrt(mean(x^2) + eps) [* w]. In place is allowed.
        kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
                             constant NormArgs &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            device const float *xr = x + (ulong)row * a.n; device float *yr = y + (ulong)row * a.n;
            float s = 0.0f;
            for (int i = tid; i < a.n; i += threads) s += xr[i] * xr[i];
            float scale = rsqrt(group_sum(s, scratch, tid, threads) / float(a.n) + a.eps);
            for (int i = tid; i < a.n; i += threads) yr[i] = xr[i] * scale * (a.hasWeight ? w[i] : 1.0f);
        }

        struct LayerNormArgs { int n; float eps; };
        // LayerNorm with weight and bias (the DSA indexer's key norm).
        kernel void layer_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device const float *b [[buffer(2)]],
                               device float *y [[buffer(3)]], constant LayerNormArgs &a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            device const float *xr = x + (ulong)row * a.n; device float *yr = y + (ulong)row * a.n;
            float s = 0.0f;
            for (int i = tid; i < a.n; i += threads) s += xr[i];
            float mean = group_sum(s, scratch, tid, threads) / float(a.n);
            float v = 0.0f;
            for (int i = tid; i < a.n; i += threads) { float d = xr[i] - mean; v += d * d; }
            float scale = rsqrt(group_sum(v, scratch, tid, threads) / float(a.n) + a.eps);
            for (int i = tid; i < a.n; i += threads) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
        }

        struct HCArgs { int rows; int iterations; float eps; };
        // mHC (four streams): from the 24 mixes of a row, pre = sigmoid(m*s0+b) + eps, post = 2 sigmoid(m*s1+b), and the
        // 4x4 combination (element dst + 4 src) through a row softmax over dst, then alternating column and row
        // normalisations (Sinkhorn), as llama.cpp's build_hc_sinkhorn. Sixteen threads a row, one per element of the 4x4
        // (and of pre and post): each does the one-thread-a-row arithmetic for its element in the same order, reading the
        // others of its row or column by SIMD shuffle (one thread a row ran the whole chain alone: 43 us a decode row).
        // Grid: rows * 16 threads in groups of a multiple of 32.
        kernel void hc_prepare(device const float *mixes [[buffer(0)]], device const float *scale [[buffer(1)]], device const float *base [[buffer(2)]],
                               device float *pre [[buffer(3)]], device float *post [[buffer(4)]], device float *comb [[buffer(5)]],
                               constant HCArgs &a [[buffer(6)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint row = gid / 16, i = gid % 16;
            if ((int)row >= a.rows) return;   // a row's sixteen threads share a SIMD group and return together
            device const float *m = mixes + row * 24;
            if (i < 4) pre[row * 4 + i] = 1.0f / (1.0f + exp(-(m[i] * scale[0] + base[i]))) + a.eps;
            else if (i < 8) post[row * 4 + i - 4] = 2.0f / (1.0f + exp(-(m[i] * scale[1] + base[i])));
            // Element i is c[d + 4 src]; its row's elements sit at lanes first + 4 src + k, its column's at first + d + 4 k.
            ushort first = lane - i, src = i / 4, d = i % 4;
            float c = m[8 + i] * scale[2] + base[8 + i];
            // softmax over dst (the contiguous index) for each src
            float mx = -INFINITY;
            for (ushort k = 0; k < 4; ++k) mx = max(mx, simd_shuffle(c, first + 4 * src + k));
            c = exp(c - mx);
            float s = 0.0f;
            for (ushort k = 0; k < 4; ++k) s += simd_shuffle(c, first + 4 * src + k);
            c = c / s + a.eps;
            for (int it = 0; it < a.iterations; ++it) {
                if (it > 0) {   // rows: sum over dst for each src
                    s = 0.0f;
                    for (ushort k = 0; k < 4; ++k) s += simd_shuffle(c, first + 4 * src + k);
                    c /= (s + a.eps);
                }
                s = 0.0f;   // columns: sum over src for each dst
                for (ushort k = 0; k < 4; ++k) s += simd_shuffle(c, first + d + 4 * k);
                c /= (s + a.eps);
            }
            comb[row * 16 + i] = c;
        }

        struct HCMixArgs { int n; int rows; };
        // out[r][i] = sum_h pre[r][h] x[r][h][i]   (x rows hold four streams of n).
        kernel void hc_collapse(device const float *x [[buffer(0)]], device const float *pre [[buffer(1)]], device float *out [[buffer(2)]],
                                constant HCMixArgs &a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            device const float *xr = x + (ulong)p.y * 4 * a.n; device const float *w = pre + p.y * 4;
            out[(ulong)p.y * a.n + p.x] = xr[p.x] * w[0] + xr[a.n + p.x] * w[1] + xr[2 * a.n + p.x] * w[2] + xr[3 * a.n + p.x] * w[3];
        }
        // next[r][d][i] = y[r][i] post[r][d] + sum_s comb[r][d + 4 s] residual[r][s][i]. `next` may not alias `residual`.
        kernel void hc_expand(device const float *y [[buffer(0)]], device const float *residual [[buffer(1)]], device const float *post [[buffer(2)]],
                              device const float *comb [[buffer(3)]], device float *next [[buffer(4)]], constant HCMixArgs &a [[buffer(5)]],
                              uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            ulong r = p.y; device const float *res = residual + r * 4 * a.n; device const float *c = comb + r * 16;
            float v = y[r * a.n + p.x];
            float s0 = res[p.x], s1 = res[a.n + p.x], s2 = res[2 * a.n + p.x], s3 = res[3 * a.n + p.x];
            for (int d = 0; d < 4; ++d) {
                next[r * 4 * a.n + d * a.n + p.x] = v * post[r * 4 + d] + c[d] * s0 + c[d + 4] * s1 + c[d + 8] * s2 + c[d + 12] * s3;
            }
        }
        // Approximate delayed FFN delta at the next attention boundary. Source post is transported
        // through the frozen next-attention residual combine; the attention branch is not replayed.
        kernel void hc_delayed_add(device const float *delta [[buffer(0)]], device const float *post [[buffer(1)]],
                                   device const float *comb [[buffer(2)]], device float *streams [[buffer(3)]],
                                   constant int &n [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= n) return;
            for (int d = 0; d < 4; ++d) {
                float scale = comb[d] * post[0] + comb[d + 4] * post[1] + comb[d + 8] * post[2] + comb[d + 12] * post[3];
                streams[d * n + p.x] += delta[p.x] * scale;
            }
        }
        // The mean of the four streams (the head reads it).
        kernel void hc_mean(device const float *x [[buffer(0)]], device float *out [[buffer(1)]], constant HCMixArgs &a [[buffer(2)]],
                            uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            device const float *xr = x + (ulong)p.y * 4 * a.n;
            out[(ulong)p.y * a.n + p.x] = (xr[p.x] + xr[a.n + p.x] + xr[2 * a.n + p.x] + xr[3 * a.n + p.x]) * 0.25f;
        }
        // Copies one row of n into all four streams (the first layer's input).
        kernel void hc_broadcast(device const float *x [[buffer(0)]], device float *out [[buffer(1)]], constant HCMixArgs &a [[buffer(2)]],
                                 uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            float v = x[(ulong)p.y * a.n + p.x];
            for (int h = 0; h < 4; ++h) out[(ulong)p.y * 4 * a.n + h * a.n + p.x] = v;
        }

        struct SwigluArgs { int n; int rows; int gateStride; int upStride; int outStride; float limit; };
        // silu(min(gate, limit)) * clamp(up, -limit, limit); limit 0 means no clamp.
        kernel void swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *out [[buffer(2)]],
                           constant SwigluArgs &a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            float g = gate[(ulong)p.y * a.gateStride + p.x], u = up[(ulong)p.y * a.upStride + p.x];
            if (a.limit > 1e-6f) { g = min(g, a.limit); u = clamp(u, -a.limit, a.limit); }
            out[(ulong)p.y * a.outStride + p.x] = g / (1.0f + exp(-g)) * u;
        }

        struct CopyArgs { int n; int rows; };
        // y[r] = x[r] for `rows` rows of n (exact copies).
        kernel void copy_rows(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant CopyArgs &a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x < a.n && (int)p.y < a.rows) y[(ulong)p.y * a.n + p.x] = x[(ulong)p.y * a.n + p.x];
        }
        struct IntRowsArgs { int n; int rows; int srcStride; int dstStride; };
        // Copies `rows` rows of n ints between strides (selections and ring slots).
        kernel void copy_int_rows(device const int *x [[buffer(0)]], device int *y [[buffer(1)]], constant IntRowsArgs &a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x < a.n && (int)p.y < a.rows) y[(ulong)p.y * a.dstStride + p.x] = x[(ulong)p.y * a.srcStride + p.x];
        }
        struct FillArgs { int n; int value; };
        // x[i] = value for i < n.
        kernel void fill_int(device int *x [[buffer(0)]], constant FillArgs &a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if ((int)i < a.n) x[i] = a.value;
        }

        struct GatherArgs { int n; int rows; };
        // out[r] = x[index[r]]; scatter: y[index[r]] += w[r] out[r] (rows of one expert, so indices are distinct).
        kernel void gather_rows(device const float *x [[buffer(0)]], device const int *index [[buffer(1)]], device float *out [[buffer(2)]],
                                constant GatherArgs &a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            out[(ulong)p.y * a.n + p.x] = x[(ulong)index[p.y] * a.n + p.x];
        }
        kernel void scatter_add_rows(device const float *x [[buffer(0)]], device const int *index [[buffer(1)]], device const float *w [[buffer(2)]],
                                     device float *y [[buffer(3)]], constant GatherArgs &a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            y[(ulong)index[p.y] * a.n + p.x] += w[p.y] * x[(ulong)p.y * a.n + p.x];
        }

        struct OrderedAddArgs { int n; int rows; };
        // y[target[u]] += w[q] * x[q] for q = order[start[u]] ..< order[start[u + 1]], in that order: the adds the
        // scatter_add_rows of a prompt's expert runs make to a row, in run order, for all of a group's runs at once.
        kernel void scatter_add_ordered(device const float *x [[buffer(0)]], device const int *order [[buffer(1)]], device const float *w [[buffer(2)]],
                                        device float *y [[buffer(3)]], device const int *target [[buffer(4)]], device const int *start [[buffer(5)]],
                                        constant OrderedAddArgs &a [[buffer(6)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.n || (int)p.y >= a.rows) return;
            device float *row = y + (ulong)target[p.y] * a.n + p.x;
            float acc = *row;
            for (int j = start[p.y]; j < start[p.y + 1]; ++j) {
                int q = order[j];
                acc += w[q] * x[(ulong)q * a.n + p.x];
            }
            *row = acc;
        }

        struct WeightedArgs { int n; int count; float w[64]; };
        // y += w[z] * x[z] for z in order (one row; decode's routed experts): the adds `count` scatter_add_rows of one
        // row each would make, in the same order.
        kernel void add_weighted_rows(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant WeightedArgs &a [[buffer(2)]],
                                      uint p [[thread_position_in_grid]]) {
            if ((int)p >= a.n) return;
            float acc = y[p];
            for (int z = 0; z < a.count; ++z) acc += a.w[z] * x[(ulong)z * a.n + p];
            y[p] = acc;
        }
        struct WeightedAtArgs { int n; int count; int row[64]; float w[64]; };
        // add_weighted_rows with x's row of z at row[z]: the same adds in the same order, whatever order x's rows are in.
        kernel void add_weighted_rows_at(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant WeightedAtArgs &a [[buffer(2)]],
                                         uint p [[thread_position_in_grid]]) {
            if ((int)p >= a.n) return;
            float acc = y[p];
            for (int z = 0; z < a.count; ++z) acc += a.w[z] * x[(ulong)a.row[z] * a.n + p];
            y[p] = acc;
        }

        struct ConvArgs { int channels; int tokens; int stride; };
        // Causal depthwise conv of width 4 over each channel, then SiLU (KDA's q/k/v). `state` holds the previous three
        // inputs (oldest first) and is advanced; `x` rows are `stride` apart, outputs `channels` apart.
        kernel void kda_conv(device const float *x [[buffer(0)]], device float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                             device float *out [[buffer(3)]], constant ConvArgs &a [[buffer(4)]], uint c [[thread_position_in_grid]]) {
            if ((int)c >= a.channels) return;
            float s0 = state[c], s1 = state[a.channels + c], s2 = state[2 * a.channels + c];
            float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
            for (int t = 0; t < a.tokens; ++t) {
                float v = x[(ulong)t * a.stride + c];
                float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
                out[(ulong)t * a.channels + c] = y / (1.0f + exp(-y));
                s0 = s1; s1 = s2; s2 = v;
            }
            state[c] = s0; state[a.channels + c] = s1; state[2 * a.channels + c] = s2;
        }

        struct GateArgs { int heads; int dim; int rows; int aIsLog; float lowerBound; };
        // KDA decay: g = lower * sigmoid(exp(A_log) (f + dt_bias)) with a lower bound, else -exp(A_log) softplus(f + dt_bias).
        // `a` is A_log (DwarfStar) or -exp(A_log) (llama.cpp's ssm_a); g is written over f.
        kernel void kda_gate(device float *f [[buffer(0)]], device const float *dtBias [[buffer(1)]], device const float *alog [[buffer(2)]],
                             constant GateArgs &a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
            int n = a.heads * a.dim;
            if ((int)p.x >= n || (int)p.y >= a.rows) return;
            float A = a.aIsLog ? -exp(alog[p.x / a.dim]) : alog[p.x / a.dim];   // -exp(A_log)
            float v = f[(ulong)p.y * n + p.x] + dtBias[p.x];
            float g = a.lowerBound / (1.0f + exp(v * A));
            f[(ulong)p.y * n + p.x] = g;
        }
        struct SigmoidArgs { int n; };
        // x[i] = 1 / (1 + exp(-x[i])) in place.
        kernel void sigmoid_inplace(device float *x [[buffer(0)]], constant SigmoidArgs &a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if ((int)i < a.n) x[i] = 1.0f / (1.0f + exp(-x[i]));
        }

        struct ScaleArgs { int n; float scale; };
        // x[i] *= scale in place.
        kernel void scale_inplace(device float *x [[buffer(0)]], constant ScaleArgs &a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if ((int)i < a.n) x[i] *= a.scale;
        }
        struct WindowArgs { int selected; int firstPosition; };
        // The last `selected` rows up to each token (oldest first; -1 before the start).
        kernel void window_select(device int *sel [[buffer(0)]], constant WindowArgs &a [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.selected) return;
            int pos = a.firstPosition + (int)p.y, row = pos - (a.selected - 1) + (int)p.x;
            sel[(ulong)p.y * a.selected + p.x] = row >= 0 ? row : -1;
        }

        struct HeadNormArgs { int dim; float eps; };
        // L2 normalisation of each head vector (one threadgroup of `dim` threads per head row).
        kernel void l2_norm_heads(device float *x [[buffer(0)]], constant HeadNormArgs &a [[buffer(1)]], uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            device float *xr = x + (ulong)row * a.dim;
            float s = 0.0f;
            for (int i = tid; i < a.dim; i += threads) s += xr[i] * xr[i];
            float scale = rsqrt(max(group_sum(s, scratch, tid, threads), a.eps * a.eps));
            for (int i = tid; i < a.dim; i += threads) xr[i] *= scale;
        }
        // KDA output: RMSNorm of each head with the shared weight, times sigmoid of the output gate.
        kernel void kda_out(device const float *o [[buffer(0)]], device const float *w [[buffer(1)]], device const float *g [[buffer(2)]],
                            device float *y [[buffer(3)]], constant HeadNormArgs &a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            ulong base = (ulong)row * a.dim;
            float s = 0.0f;
            for (int i = tid; i < a.dim; i += threads) s += o[base + i] * o[base + i];
            float scale = rsqrt(group_sum(s, scratch, tid, threads) / float(a.dim) + a.eps);
            for (int i = tid; i < a.dim; i += threads) y[base + i] = o[base + i] * scale * w[i] / (1.0f + exp(-g[base + i]));
        }

        struct ScanArgs { int heads; int tokens; int qStride; int kStride; int vStride; float scale; };
        // The KDA recurrence over `tokens` steps (dim 128), one threadgroup of 256 threads per (head, block of 32 value
        // columns): value columns are independent in the delta rule, so a head splits across four groups that share only
        // the inputs. Thread (j, part) holds S[16 part .. 16 part + 15][j], the state stored as llama.cpp's
        // gated_delta_net does (S[i][j]: key channel i, value channel j; column j contiguous). Per token:
        // S[i][:] *= exp(g[i]); delta_j = (v_j - sum_i S[i][j] k_i) * beta; S[i][j] += k_i delta_j; o_j = scale * sum_i S[i][j] q_i.
        kernel void kda_scan(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                             device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *state [[buffer(5)]],
                             device float *out [[buffer(6)]], constant ScanArgs &a [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
            const int D = 128;
            threadgroup float kq[2 * D], gg[D];
            uint head = group / 4, j = (group % 4) * 32 + tid / 8, part = tid % 8;
            device float *S = state + (ulong)head * D * D;
            float s[16];
            for (int r = 0; r < 16; ++r) s[r] = S[j * D + part * 16 + r];
            for (int t = 0; t < a.tokens; ++t) {
                ulong hq = (ulong)t * a.qStride + head * D, hk = (ulong)t * a.kStride + head * D, hv = (ulong)t * a.vStride + head * D;
                if (tid < D) { kq[tid] = k[hk + tid]; kq[D + tid] = q[hq + tid]; gg[tid] = exp(g[(ulong)t * a.heads * D + head * D + tid]); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float b = beta[(ulong)t * a.heads + head];
                float mem = 0.0f;
                for (int r = 0; r < 16; ++r) { int i = part * 16 + r; s[r] *= gg[i]; mem += s[r] * kq[i]; }
                mem += simd_shuffle_xor(mem, 1); mem += simd_shuffle_xor(mem, 2); mem += simd_shuffle_xor(mem, 4);
                float delta = (v[hv + j] - mem) * b;
                float o = 0.0f;
                for (int r = 0; r < 16; ++r) { int i = part * 16 + r; s[r] += kq[i] * delta; o += s[r] * kq[D + i]; }
                o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 4);
                if (part == 0) out[(ulong)t * a.heads * D + head * D + j] = o * a.scale;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int r = 0; r < 16; ++r) S[j * D + part * 16 + r] = s[r];
        }

        struct PermuteArgs { int a; int b; int n; };
        // [a][b][n] -> [b][a][n]
        kernel void permute3(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant PermuteArgs &p [[buffer(2)]],
                             uint3 i [[thread_position_in_grid]]) {
            if ((int)i.x >= p.n || (int)i.y >= p.b || (int)i.z >= p.a) return;
            y[((ulong)i.y * p.a + i.z) * p.n + i.x] = x[((ulong)i.z * p.b + i.y) * p.n + i.x];
        }

        struct AttendArgs { int heads; int dim; int selected; int qStride; int qHeadStride; int outStride; int outHeadStride; int split; float scale; };
        // Latent attention of token rows over their selected cached rows (`sel` holds `selected` row indices per token,
        // -1 for none). A row index below `split` is a row of `kv`, the others rows of `kv2` from `split` on (DeepSeek's
        // window ring and compressed cache in one selection). All heads share the rows as keys and values (MLA with
        // absorbed projections), so one thread reads a cached row once for every head. Heads are at most 64.
        inline device const float *attend_row(device const float *kv, device const float *kv2, int row, int split, int dim) {
            return row < split ? kv + (ulong)row * dim : kv2 + (ulong)(row - split) * dim;
        }
        // Step 1: scores[t][h][j] = scale * q[t][h] . row(sel[t][j]); one thread per (slot j, token t).
        kernel void attend_scores(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                  device const int *sel [[buffer(3)]], device float *scores [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                  uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.selected) return;
            ulong t = p.y;
            int row = sel[t * a.selected + p.x];
            if (row < 0) { for (int h = 0; h < a.heads; ++h) scores[(t * a.heads + h) * a.selected + p.x] = -INFINITY; return; }
            float acc[64];
            for (int h = 0; h < a.heads; ++h) acc[h] = 0.0f;
            device const float4 *kk = (device const float4 *)attend_row(kv, kv2, row, a.split, a.dim);
            device const float *qt = q + t * a.qStride;
            for (int i = 0; i < a.dim / 4; ++i) {
                float4 k4 = kk[i];
                for (int h = 0; h < a.heads; ++h) acc[h] += dot(((device const float4 *)(qt + (ulong)h * a.qHeadStride))[i], k4);
            }
            for (int h = 0; h < a.heads; ++h) scores[(t * a.heads + h) * a.selected + p.x] = acc[h] * a.scale;
        }
        // Step 1 for few token rows (decode): the same scores, one thread per (slot j, token t, head h), so a single row
        // keeps the GPU busy; each score sums its terms in the same order as `attend_scores` (bitwise the same).
        kernel void attend_scores_heads(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                        device const int *sel [[buffer(3)]], device float *scores [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                        uint3 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.selected || (int)p.z >= a.heads) return;
            ulong t = p.y, h = p.z;
            int row = sel[t * a.selected + p.x];
            if (row < 0) { scores[(t * a.heads + h) * a.selected + p.x] = -INFINITY; return; }
            float acc = 0.0f;
            device const float4 *kk = (device const float4 *)attend_row(kv, kv2, row, a.split, a.dim);
            device const float4 *qh = (device const float4 *)(q + t * a.qStride + h * a.qHeadStride);
            for (int i = 0; i < a.dim / 4; ++i) acc += dot(qh[i], kk[i]);
            scores[(t * a.heads + h) * a.selected + p.x] = acc * a.scale;
        }
        struct SoftmaxArgs { int n; int hasSink; int sinkPeriod; };
        // Softmax of each row in place (one threadgroup per row); with a sink, the row's sink logit (sink[row % period]:
        // one per head) joins the denominator.
        kernel void softmax_rows(device float *x [[buffer(0)]], device const float *sink [[buffer(1)]], constant SoftmaxArgs &a [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            device float *xr = x + (ulong)row * a.n;
            float s0 = a.hasSink ? sink[row % (uint)a.sinkPeriod] : -INFINITY;
            float m = s0;
            for (int i = tid; i < a.n; i += threads) m = max(m, xr[i]);
            m = group_max(m, scratch, tid, threads);
            float s = 0.0f;
            for (int i = tid; i < a.n; i += threads) { float e = xr[i] == -INFINITY ? 0.0f : exp(xr[i] - m); xr[i] = e; s += e; }
            s = group_sum(s, scratch, tid, threads) + (a.hasSink ? exp(s0 - m) : 0.0f);
            float inv = s > 0.0f ? 1.0f / s : 0.0f;
            for (int i = tid; i < a.n; i += threads) xr[i] *= inv;
        }
        // Step 2: out[t][h][c] = sum_j p[t][h][j] row(sel[t][j])[c]; one thread per (channel c, token t) for all heads.
        kernel void attend_values(device const float *p [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                  device const int *sel [[buffer(3)]], device float *out [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                  uint2 i [[thread_position_in_grid]]) {
            if ((int)i.x >= a.dim) return;
            ulong t = i.y;
            device const int *sr = sel + t * a.selected;
            device const float *pt = p + t * a.heads * a.selected;
            float acc[64];
            for (int h = 0; h < a.heads; ++h) acc[h] = 0.0f;
            for (int j = 0; j < a.selected; ++j) {
                int row = sr[j];
                if (row < 0) continue;
                float v = attend_row(kv, kv2, row, a.split, a.dim)[i.x];
                for (int h = 0; h < a.heads; ++h) acc[h] += pt[(ulong)h * a.selected + j] * v;
            }
            for (int h = 0; h < a.heads; ++h) out[t * a.outStride + (ulong)h * a.outHeadStride + i.x] = acc[h];
        }
        // Step 2 for few token rows (decode): one thread per (channel c, token t, head h), the sum over j in the same order
        // as `attend_values` (bitwise the same).
        kernel void attend_values_heads(device const float *p [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                        device const int *sel [[buffer(3)]], device float *out [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                        uint3 i [[thread_position_in_grid]]) {
            if ((int)i.x >= a.dim || (int)i.z >= a.heads) return;
            ulong t = i.y, h = i.z;
            device const int *sr = sel + t * a.selected;
            device const float *ph = p + (t * a.heads + h) * a.selected;
            float acc = 0.0f;
            for (int j = 0; j < a.selected; ++j) {
                int row = sr[j];
                if (row < 0) continue;
                acc += ph[j] * attend_row(kv, kv2, row, a.split, a.dim)[i.x];
            }
            out[t * a.outStride + h * a.outHeadStride + i.x] = acc;
        }
        // The heads-parallel kernels with eight heads a thread (z: heads 8z ..< 8z + 8; heads a multiple of 8): one read of a
        // cached row's values serves eight sums, each over its terms in the same order (bitwise the same). One head a thread
        // read every row once per head, 64 times for DeepSeek's heads.
        kernel void attend_scores_heads8(device const float *q [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                         device const int *sel [[buffer(3)]], device float *scores [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                         uint3 p [[thread_position_in_grid]]) {
            int h0 = (int)p.z * 8;
            if ((int)p.x >= a.selected || h0 + 8 > a.heads) return;
            ulong t = p.y;
            int row = sel[t * a.selected + p.x];
            if (row < 0) { for (int g = 0; g < 8; ++g) scores[(t * a.heads + h0 + g) * a.selected + p.x] = -INFINITY; return; }
            float acc[8];
            for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
            device const float4 *kk = (device const float4 *)attend_row(kv, kv2, row, a.split, a.dim);
            for (int i = 0; i < a.dim / 4; ++i) {
                float4 k4 = kk[i];
                for (int g = 0; g < 8; ++g) acc[g] += dot(((device const float4 *)(q + t * a.qStride + (ulong)(h0 + g) * a.qHeadStride))[i], k4);
            }
            for (int g = 0; g < 8; ++g) scores[(t * a.heads + h0 + g) * a.selected + p.x] = acc[g] * a.scale;
        }
        kernel void attend_values_heads8(device const float *p [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *kv2 [[buffer(2)]],
                                         device const int *sel [[buffer(3)]], device float *out [[buffer(4)]], constant AttendArgs &a [[buffer(5)]],
                                         uint3 i [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
                                         uint3 size [[threads_per_threadgroup]]) {
            // Four value columns a thread (x: columns 4x ..< 4x + 4; dim a multiple of 4), and the threadgroup (one token row,
            // eight heads) stages 64 selected rows' indices and probabilities at a time, which every thread reads: each sum
            // is over the same terms in the same order (bitwise the same as a column a thread).
            threadgroup int rows[64];
            threadgroup float probs[8][64];
            int h0 = (int)i.z * 8;
            if (h0 + 8 > a.heads) return;  // the whole threadgroup
            bool live = (int)i.x * 4 < a.dim;
            ulong t = i.y;
            device const int *sr = sel + t * a.selected;
            device const float *pt = p + (t * a.heads + h0) * a.selected;
            float4 acc[8];
            for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
            for (int j0 = 0; j0 < a.selected; j0 += 64) {
                int n = min(64, a.selected - j0);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int l = (int)lane; l < n; l += (int)size.x) {
                    rows[l] = sr[j0 + l];
                    for (int g = 0; g < 8; ++g) probs[g][l] = pt[(ulong)g * a.selected + j0 + l];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!live) continue;
                for (int j = 0; j < n; ++j) {
                    int row = rows[j];
                    if (row < 0) continue;
                    float4 v = ((device const float4 *)attend_row(kv, kv2, row, a.split, a.dim))[i.x];
                    for (int g = 0; g < 8; ++g) acc[g] += probs[g][j] * v;
                }
            }
            if (!live) return;
            for (int g = 0; g < 8; ++g) {
                device float *o = out + t * a.outStride + (ulong)(h0 + g) * a.outHeadStride + i.x * 4;
                o[0] = acc[g].x;
                o[1] = acc[g].y;
                o[2] = acc[g].z;
                o[3] = acc[g].w;
            }
        }

        struct PoolArgs { int dim; int first; int count; int pool; };
        // DSA k-pool: for pools first ..< first + count (each `pool` consecutive cached rows), the pooled key per channel is
        // sum_m softmax_m(gate[m][c] + ape[c + dim m]) key[m][c].
        kernel void kpool_pool(device const float *key [[buffer(0)]], device const float *gate [[buffer(1)]], device const float *ape [[buffer(2)]],
                               device float *pooled [[buffer(3)]], constant PoolArgs &a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.dim || (int)p.y >= a.count) return;
            int pool = a.first + (int)p.y, c = (int)p.x;
            float logits[8], mx = -INFINITY;
            for (int m = 0; m < a.pool; ++m) { logits[m] = gate[(ulong)(pool * a.pool + m) * a.dim + c] + ape[c + a.dim * m]; mx = max(mx, logits[m]); }
            float s = 0.0f, acc = 0.0f;
            for (int m = 0; m < a.pool; ++m) { float e = exp(logits[m] - mx); s += e; acc += e * key[(ulong)(pool * a.pool + m) * a.dim + c]; }
            pooled[(ulong)pool * a.dim + c] = acc / s;
        }

        struct IndexArgs { int heads; int dim; int pools; int pool; int firstPosition; };
        // Indexer score of every pool for each token row: sum_h w[t][h] relu(q[t][h] . pooled[p]); a pool is visible to the
        // token at position pos when it ends at or before it ((p + 1) * pool - 1 <= pos), else -inf.
        kernel void index_scores(device const float *q [[buffer(0)]], device const float *pooled [[buffer(1)]], device const float *w [[buffer(2)]],
                                 device float *scores [[buffer(3)]], constant IndexArgs &a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.pools) return;
            ulong t = p.y; int pos = a.firstPosition + (int)t;
            float s = -INFINITY;
            if ((int(p.x) + 1) * a.pool - 1 <= pos) {
                s = 0.0f;
                device const float4 *k = (device const float4 *)(pooled + (ulong)p.x * a.dim), *qt = (device const float4 *)(q + t * a.heads * a.dim);
                int n4 = a.dim / 4, h = 0;
                // Eight heads at a time as independent sums, each over its terms in order (one chain a head left the ALUs
                // waiting: 1.2 ms where this takes 0.2 at 20K entries), then the heads added in order.
                for (; h + 8 <= a.heads; h += 8) {
                    float4 acc[8];
                    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
                    for (int i = 0; i < n4; ++i) {
                        float4 k4 = k[i];
                        for (int g = 0; g < 8; ++g) acc[g] += qt[(h + g) * n4 + i] * k4;
                    }
                    for (int g = 0; g < 8; ++g) s += w[t * a.heads + h + g] * max(acc[g].x + acc[g].y + acc[g].z + acc[g].w, 0.0f);
                }
                for (; h < a.heads; ++h) {
                    float4 acc = 0.0f;
                    for (int i = 0; i < n4; ++i) acc += qt[h * n4 + i] * k[i];
                    s += w[t * a.heads + h] * max(acc.x + acc.y + acc.z + acc.w, 0.0f);
                }
            }
            scores[t * a.pools + p.x] = s;
        }

        // The same ordered dot products as index_scores for surviving candidates. Keep the score layout and
        // index_select unchanged so causal visibility, tie ordering and selected IDs retain their baseline bits.
        struct MaskedIndexArgs { int heads; int dim; int pools; int pool; int firstPosition; int blockSize; int keepStride; };
        kernel void index_scores_masked(device const float *q [[buffer(0)]], device const float *pooled [[buffer(1)]], device const float *w [[buffer(2)]],
                                 device float *scores [[buffer(3)]], device const float *keep [[buffer(4)]], constant MaskedIndexArgs &a [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.pools) return;
            ulong t = p.y; int pos = a.firstPosition + (int)t;
            // A discarded block would be overwritten with -inf by apply_block_mask. Skip its dot products.
            if (keep[t * a.keepStride + p.x / a.blockSize] == 0.0f) {
                scores[t * a.pools + p.x] = -INFINITY;
                return;
            }
            float s = -INFINITY;
            if ((int(p.x) + 1) * a.pool - 1 <= pos) {
                s = 0.0f;
                device const float4 *k = (device const float4 *)(pooled + (ulong)p.x * a.dim), *qt = (device const float4 *)(q + t * a.heads * a.dim);
                int n4 = a.dim / 4, h = 0;
                // Eight heads at a time as independent sums, each over its terms in order (one chain a head left the ALUs
                // waiting: 1.2 ms where this takes 0.2 at 20K entries), then the heads added in order.
                for (; h + 8 <= a.heads; h += 8) {
                    float4 acc[8];
                    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
                    for (int i = 0; i < n4; ++i) {
                        float4 k4 = k[i];
                        for (int g = 0; g < 8; ++g) acc[g] += qt[(h + g) * n4 + i] * k4;
                    }
                    for (int g = 0; g < 8; ++g) s += w[t * a.heads + h + g] * max(acc[g].x + acc[g].y + acc[g].z + acc[g].w, 0.0f);
                }
                for (; h < a.heads; ++h) {
                    float4 acc = 0.0f;
                    for (int i = 0; i < n4; ++i) acc += qt[h * n4 + i] * k[i];
                    s += w[t * a.heads + h] * max(acc.x + acc.y + acc.z + acc.w, 0.0f);
                }
            }
            scores[t * a.pools + p.x] = s;
        }

        struct SelectArgs { int pools; int pool; int top; int selected; int firstPosition; int tail; int width; int offset; int stride; };
        // The rows a token attends: the `top` visible entries of highest score (all of them when fewer are visible; an entry
        // j is visible to the token at position pos when (j + 1) * pool - 1 <= pos), each written as `width` rows
        // offset + j * width + m, then (with `tail`) the rows after the last complete entry up to the token itself.
        // GLM's pools: pool = width = 4; DeepSeek's compressed rows: pool = ratio, width 1, offset = the window ring size.
        // One threadgroup per token (up to 1,024 threads) finds the top-th largest score by a radix select over
        // order-preserving float bits (any count); the chosen entries are written in entry order, ties at the threshold
        // lowest first, each thread's run of entries at the slots the runs before it filled, so the result is deterministic
        // (llama.cpp orders by score; the set is what attention reads). `selected` slots of a row (`stride` apart) are
        // written; unused ones are -1.
        inline uint ordered_bits(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
        // Exclusive prefix sum of v over the threadgroup in thread order (exact integers); `total` gets the whole sum.
        inline uint group_prefix(uint v, threadgroup uint *scratch, uint tid, uint threads, thread uint &total) {
            uint lane = tid % 32, warp = tid / 32, warps = (threads + 31) / 32;
            uint before = simd_prefix_exclusive_sum(v);
            if (lane == 31 || tid == threads - 1) scratch[warp] = before + v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint offset = 0; total = 0;
            for (uint w = 0; w < warps; ++w) { uint c = scratch[w]; if (w < warp) offset += c; total += c; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return offset + before;
        }
        // The top-th largest order-preserving bits among sr[0 ..< count] (top <= count): a byte at a time from the top, each
        // from a 256-bin histogram of the values that match the bytes found so far.
        inline uint radix_threshold(device const float *sr, int count, int top, threadgroup atomic_uint *hist, threadgroup uint *pick, uint tid, uint threads) {
            uint prefix = 0, mask = 0, remaining = (uint)top;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (uint b = tid; b < 256; b += threads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = tid; i < count; i += threads) {
                    uint u = ordered_bits(sr[i]);
                    if ((u & mask) == prefix) atomic_fetch_add_explicit(&hist[(u >> shift) & 255u], 1u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    uint above = 0, b = 255;
                    for (;; --b) { uint c = atomic_load_explicit(&hist[b], memory_order_relaxed); if (above + c >= remaining || b == 0) break; above += c; }
                    pick[0] = b; pick[1] = remaining - above;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                prefix |= pick[0] << shift; mask |= 255u << shift; remaining = pick[1];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            return prefix;
        }
        kernel void index_select(device const float *scores [[buffer(0)]], device int *sel [[buffer(1)]], constant SelectArgs &a [[buffer(2)]],
                                 uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup uint scratch[32], pick[2];
            threadgroup atomic_uint hist[256];
            device const float *sr = scores + (ulong)t * a.pools;
            device int *out = sel + (ulong)t * a.stride;
            int pos = a.firstPosition + (int)t;
            int visible = min(a.pools, (pos + 1) / a.pool);
            int take = min(visible, a.top);
            bool partial = take < visible;
            uint lo = partial ? radix_threshold(sr, visible, take, hist, pick, tid, threads) : 0;
            // Each thread takes a run of entries in thread order: the ties at lo before its run rank its own ties, and the
            // entries chosen before it give its first slot.
            int per = (visible + (int)threads - 1) / (int)threads, b = min(visible, (int)tid * per), e = min(visible, b + per);
            uint myAbove = 0, myTies = 0;
            if (partial) for (int i = b; i < e; ++i) { uint u = ordered_bits(sr[i]); myAbove += u > lo ? 1 : 0; myTies += u == lo ? 1 : 0; }
            uint above, tiesTotal, chosenTotal, myChosen = 0;
            group_prefix(myAbove, scratch, tid, threads, above);
            uint tieRank = group_prefix(myTies, scratch, tid, threads, tiesTotal);
            int ties = take - (int)above;
            for (int i = b, r = (int)tieRank; i < e; ++i) {
                bool chosen = !partial;
                if (!chosen) { uint u = ordered_bits(sr[i]); chosen = u > lo || (u == lo && r++ < ties); }
                // An entry the scores rule out entirely (-inf, as outside DeepSeek's candidate blocks) is not attended.
                myChosen += chosen && sr[i] != -INFINITY ? 1 : 0;
            }
            uint slot = group_prefix(myChosen, scratch, tid, threads, chosenTotal);
            for (int i = b, r = (int)tieRank; i < e; ++i) {
                bool chosen = !partial;
                if (!chosen) { uint u = ordered_bits(sr[i]); chosen = u > lo || (u == lo && r++ < ties); }
                if (chosen && sr[i] != -INFINITY) { for (int m = 0; m < a.width; ++m) out[slot * a.width + m] = a.offset + i * a.width + m; ++slot; }
            }
            int filled = (int)chosenTotal * a.width;
            int tail = a.tail ? max(0, min(pos - visible * a.pool + 1, a.selected - filled)) : 0;
            for (int i = tid; i < tail; i += threads) out[filled + i] = a.offset + visible * a.pool + i;
            for (int i = filled + tail + tid; i < a.selected; i += threads) out[i] = -1;
        }

        struct RingArgs { int window; int ring; int stride; int firstPosition; };
        // DeepSeek's sliding window: the ring slots of positions pos - window + 1 ... pos (position p lives in slot p % ring),
        // -1 before the start.
        kernel void ring_select(device int *sel [[buffer(0)]], constant RingArgs &a [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.window) return;
            int pos = a.firstPosition + (int)p.y, at = pos - (a.window - 1) + (int)p.x;
            sel[(ulong)p.y * a.stride + p.x] = at >= 0 ? at % a.ring : -1;
        }

        struct RopeArgs { int dim; int rope; int vectors; int firstPosition; int positionStep; int inverse; };
        // Rotates the last `rope` values of each `dim`-long vector (`vectors` per row) as adjacent pairs (x[2i], x[2i+1])
        // by position * freq[i]; row r is at position firstPosition + r * positionStep. `inverse` rotates back (DeepSeek
        // takes the query's rotation off the attention output).
        kernel void rope_tail(device float *x [[buffer(0)]], device const float *freq [[buffer(1)]], constant RopeArgs &a [[buffer(2)]],
                              uint3 p [[thread_position_in_grid]]) {
            int pairs = a.rope / 2;
            if ((int)p.x >= pairs || (int)p.y >= a.vectors) return;
            float angle = float(a.firstPosition + (int)p.z * a.positionStep) * freq[p.x];
            float c = cos(angle), s = a.inverse ? -sin(angle) : sin(angle);
            device float *v = x + ((ulong)p.z * a.vectors + p.y) * a.dim + (a.dim - a.rope) + 2 * p.x;
            float x0 = v[0], x1 = v[1];
            v[0] = x0 * c - x1 * s; v[1] = x0 * s + x1 * c;
        }

        struct EngramArgs { int n; int streams; float eps; };
        // DeepSeek's Engram gate for one (row, stream): dot = sum(h * w * key) * rsqrt(mean h^2 + eps) * rsqrt(mean key^2 +
        // eps) / sqrt(n); gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))); h += gate * value. `kv` rows hold `streams`
        // keys then the value; `w` is the per-stream weight (q_weight * k_weight).
        kernel void engram_gate(device float *x [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *w [[buffer(2)]],
                                constant EngramArgs &a [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            uint row = group / a.streams, stream = group % a.streams;
            device float *h = x + ((ulong)row * a.streams + stream) * a.n;
            device const float *key = kv + (ulong)row * (a.streams + 1) * a.n + (ulong)stream * a.n;
            device const float *value = kv + (ulong)row * (a.streams + 1) * a.n + (ulong)a.streams * a.n;
            device const float *ws = w + (ulong)stream * a.n;
            float hh = 0.0f, kk = 0.0f, d = 0.0f;
            for (int i = tid; i < a.n; i += threads) { float hv = h[i], kv0 = key[i]; hh += hv * hv; kk += kv0 * kv0; d += hv * ws[i] * kv0; }
            hh = group_sum(hh, scratch, tid, threads); kk = group_sum(kk, scratch, tid, threads); d = group_sum(d, scratch, tid, threads);
            float dot = d * rsqrt(hh / float(a.n) + a.eps) * rsqrt(kk / float(a.n) + a.eps) / sqrt(float(a.n));
            float mag = sqrt(max(fabs(dot), 1e-6f));
            float gate = 1.0f / (1.0f + exp(-(dot < 0.0f ? -mag : mag)));
            for (int i = tid; i < a.n; i += threads) h[i] += gate * value[i];
        }

        struct CandidateArgs { int entries; int blockSize; int blocks; int top; int firstPosition; int ratio; int keepStride; };
        // DeepSeek's candidate blocks (level one of its two-level top-k): per token, each block of `blockSize` compressed
        // entries scores its best visible entry; the block holding the newest visible entry is pinned in; the `top` best
        // blocks with any visible entry are kept (keep[b] = 1, else 0). Bisection as index_select; ties lowest block first.
        // Each block's best score is staged in its own keep slot (device memory, so any context length fits). One
        // threadgroup per token row: scores rows are `entries` apart, keep rows `keepStride`.
        kernel void candidate_blocks(device const float *scores [[buffer(0)]], device float *keep [[buffer(1)]], constant CandidateArgs &a [[buffer(2)]],
                                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
            threadgroup float scratch[32];
            device const float *sr = scores + (ulong)t * a.entries;
            device float *kr = keep + (ulong)t * a.keepStride, *best = kr;
            int pos = a.firstPosition + (int)t, visible = min(a.entries, (pos + 1) / a.ratio);
            int last = visible > 0 ? (visible - 1) / a.blockSize : -1;
            for (int b = tid; b < a.blocks; b += threads) {
                float m = -INFINITY;
                for (int j = b * a.blockSize; j < min(visible, (b + 1) * a.blockSize); ++j) m = max(m, sr[j]);
                best[b] = b == last ? 1e30f : m;
            }
            threadgroup_barrier(mem_flags::mem_device);
            int reachable = last + 1, take = min(reachable, a.top);
            uint lo = 0, hi = 0xFFFFFFFFu;
            if (take < reachable) {
                while (lo < hi) {
                    uint mid = lo + (hi - lo) / 2 + 1;
                    float count = 0.0f;
                    for (int b = tid; b < reachable; b += threads) count += ordered_bits(best[b]) >= mid ? 1.0f : 0.0f;
                    count = group_sum(count, scratch, tid, threads);
                    if (count >= float(take)) lo = mid; else hi = mid - 1;
                }
            }
            if (tid != 0) return;
            int above = 0;
            if (take < reachable) for (int b = 0; b < reachable; ++b) above += ordered_bits(best[b]) > lo ? 1 : 0;
            int ties = take - above, kept = 0;
            for (int b = 0; b < a.blocks; ++b) {
                bool chosen = b < reachable && best[b] > -INFINITY && kept < take;
                if (chosen && take < reachable) { uint u = ordered_bits(best[b]); chosen = u > lo || (u == lo && ties-- > 0); }
                kr[b] = chosen ? 1.0f : 0.0f; kept += chosen ? 1 : 0;
            }
        }
        struct BlockMaskArgs { int entries; int blockSize; int blocks; int keepStride; };
        // Level two: scores outside the kept blocks become -inf (row p.y; keep rows `keepStride` apart).
        kernel void apply_block_mask(device float *scores [[buffer(0)]], device const float *keep [[buffer(1)]], constant BlockMaskArgs &a [[buffer(2)]],
                                     uint2 p [[thread_position_in_grid]]) {
            if ((int)p.x >= a.entries) return;
            if (keep[(ulong)p.y * a.keepStride + p.x / a.blockSize] == 0.0f) scores[(ulong)p.y * a.entries + p.x] = -INFINITY;
        }
        """#
}
