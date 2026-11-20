import Foundation
import Metal

/// Counters of the device PLE route for diagnostics: encoded tiles/tokens, host fallbacks and row reads.
public struct DevicePLECounts: Codable, Sendable {
    public var prefillTiles: UInt64 = 0, prefillRows: UInt64 = 0, decodeTokens: UInt64 = 0
    public var hostFallbackTiles: UInt64 = 0
    public var rowsRequested: UInt64 = 0, rowsRead: UInt64 = 0
    public var rowWaitSeconds = 0.0
    /// The same wait split by phase (rowWaitSeconds is their sum).
    public var prefillRowWaitSeconds = 0.0, decodeRowWaitSeconds = 0.0
    /// Prefill scopes whose rows were read ahead during the previous scope.
    public var lookaheadScopes: UInt64 = 0
}

/// BITWISE device port of `ReferencePLE.call` plus its caller's residual addition, for whole-execution
/// prefill tiles and decode tokens, encoded into the caller's layer-1 command
/// (`device_ple_exact_against_reference_arithmetic`).
///
/// Arithmetic contract, step by step against ReferencePLE:
/// - row bytes come from the same PackStore rows; dequantization reproduces
///   `ReferenceWeights.dequantizedRow` (nibble*scale is exact in FP32, then +bias, BF16 round);
/// - key/value projections use `ReferenceKernels.encodePinnedProjection` with the plan the
///   synchronous `qlinearPinnedBF16` selects for the same rows;
/// - the three grouped RMS norms and the PLE dot use the existing pinned pipelines with the
///   synchronous entry's thread counts (norm weights are tiled to 10240, giving identical values);
/// - the gate is a 65,536-entry table evaluated on the host with the reference scalar code for
///   every BF16 dot value; the SiLU table is the pinned host table;
/// - convolution taps use FMA in reference order; every BF16 rounding quiets NaNs like BF16.encode,
///   and NaN operands propagate in the host's (AArch64) operand order.
/// The host convolution state stays authoritative: callers pass it in, and adopt `nextState`
/// only after a joined command whose PLE result they use.
package final class DevicePLE {

    /// Projection tensors stay in the resident cache; callers keep leases until the command joins.
    package struct Projections {
        let keyWords, keyScales, keyBiases, valueWords, valueScales, valueBiases: MTLBuffer
        let retained: [AnyObject]
        package init(
            keyWords: MTLBuffer, keyScales: MTLBuffer, keyBiases: MTLBuffer, valueWords: MTLBuffer, valueScales: MTLBuffer,
            valueBiases: MTLBuffer, retained: [AnyObject]
        ) {
            self.keyWords = keyWords
            self.keyScales = keyScales
            self.keyBiases = keyBiases
            self.valueWords = valueWords
            self.valueScales = valueScales
            self.valueBiases = valueBiases
            self.retained = retained
        }
    }

    /// Per-row-count scratch for one encoded PLE (owned by the caller, reusable once its command joined):
    /// packed rows, BF16 embedding, key/value projections, BF16 narrowings, dots, gated values and the
    /// [9 x 10240] BF16 convolution state in and out. All 16 KiB page-charged to the ledger.
    package final class Workspace {
        let rows: Int
        let packed, embedding, keyRaw, key16, keyNorm, value, hidden16, query, dots, gated, gated16, normalized: MTLBuffer
        let state, nextState: MTLBuffer
        let keyPartials, valuePartials: MTLBuffer?
        private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
        fileprivate init(device: MTLDevice, rows: Int, keyPartialBytes: Int, valuePartialBytes: Int, ledger: MemoryLedger) throws {
            guard (1...256).contains(rows) else { throw EngineError("Device PLE workspace rows must be 1...256") }
            let wide = rows * 10240
            let sizes = [
                rows * 1600, rows * 2560 * 2, wide * 4, wide * 2, wide * 4, rows * 2560 * 4,
                wide * 2, wide * 4, rows * 16, wide * 4, wide * 2, wide * 4, 92160 * 2, 92160 * 2, keyPartialBytes, valuePartialBytes,
            ]
            let total = sizes.reduce(0) { $0 + (($1 + 16383) / 16384) * 16384 }
            reservation = try ledger.reserve(bytes: UInt64(total), category: "device_ple_workspace")
            self.ledger = ledger
            self.rows = rows
            func make(_ bytes: Int) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                    throw EngineError("Device PLE workspace allocation failed")
                }
                return buffer
            }
            do {
                packed = try make(sizes[0])
                embedding = try make(sizes[1])
                keyRaw = try make(sizes[2])
                key16 = try make(sizes[3])
                keyNorm = try make(sizes[4])
                value = try make(sizes[5])
                hidden16 = try make(sizes[6])
                query = try make(sizes[7])
                dots = try make(sizes[8])
                gated = try make(sizes[9])
                gated16 = try make(sizes[10])
                normalized = try make(sizes[11])
                state = try make(sizes[12])
                nextState = try make(sizes[13])
                keyPartials = keyPartialBytes > 0 ? try make(keyPartialBytes) : nil
                valuePartials = valuePartialBytes > 0 ? try make(valuePartialBytes) : nil
            } catch {
                try? ledger.release(reservation)
                throw error
            }
        }
        deinit { try? ledger.release(reservation) }
    }

    private let device: MTLDevice
    private let kernels: ReferenceKernels, reductions: PinnedReductions
    private let simdMatrix: Bool
    private let dequant, narrow, gate, convolution, stateTail: MTLComputePipelineState
    private let gateTable, siluTable, normKey, normQuery, normConv, convolutionWeights: MTLBuffer
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    package private(set) var counts = DevicePLECounts()

    /// - Parameters: norm weights are 2560 or 10240 values, convolution weights 10240*4 values,
    ///   exactly as ReferencePLE reads them; `arithmetic` is the unary policy ReferencePLE uses.
    package init(
        kernels: ReferenceKernels, reductions: PinnedReductions, simdMatrix: Bool, arithmetic: LayerUnaryArithmetic,
        normKey: [Float], normQuery: [Float], normConv: [Float], convolution: [Float], ledger: MemoryLedger
    ) throws {
        let device = kernels.device
        guard convolution.count == 40960 else { throw EngineError("Device PLE convolution weights differ") }
        func tiled(_ values: [Float]) throws -> [UInt16] {
            if values.count == 10240 { return values.map(BF16.encode) }
            guard values.count == 2560 else { throw EngineError("Device PLE norm weights must be 2560 or 10240 values") }
            let row = values.map(BF16.encode)
            return row + row + row + row
        }
        reservation = try ledger.reserve(bytes: 524_288, category: "device_ple_tables_and_weights")
        self.ledger = ledger
        do {
            func buffer<T>(_ values: [T]) throws -> MTLBuffer {
                guard
                    let value = values.withUnsafeBytes({
                        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                    })
                else {
                    throw EngineError("Device PLE table allocation failed")
                }
                return value
            }
            gateTable = try buffer(Self.gateTable(arithmetic: arithmetic))
            siluTable = try buffer((0..<65_536).map { BF16.encode(arithmetic.silu(BF16.decode(UInt16($0)))) })
            self.normKey = try buffer(try tiled(normKey))
            self.normQuery = try buffer(try tiled(normQuery))
            self.normConv = try buffer(try tiled(normConv))
            convolutionWeights = try buffer(convolution)
            let options = MTLCompileOptions()
            options.languageVersion = .version3_1
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            let library = try device.makeLibrary(source: Self.source, options: options)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else { throw EngineError("Missing device PLE kernel \(name)") }
                return try device.makeComputePipelineState(function: function)
            }
            dequant = try pipeline("ple_device_dequant")
            narrow = try pipeline("ple_device_narrow")
            gate = try pipeline("ple_device_gate")
            self.convolution = try pipeline("ple_device_conv")
            stateTail = try pipeline("ple_device_state")
        } catch {
            try? ledger.release(reservation)
            throw error
        }
        self.device = device
        self.kernels = kernels
        self.reductions = reductions
        self.simdMatrix = simdMatrix
    }
    deinit { try? ledger.release(reservation) }

    /// ReferencePLE's scalar gate for every BF16 value of the pinned PLE dot.
    static func gateTable(arithmetic: LayerUnaryArithmetic) -> [UInt16] {
        let root = LayerArithmetic.round(sqrtf(2560))
        let floor = LayerArithmetic.round(1e-6)
        return (0..<65_536).map { bits in
            let reduced = BF16.decode(UInt16(bits))
            let scaled = LayerArithmetic.round(reduced / root)
            let magnitude = max(abs(scaled), floor)
            let gate = arithmetic.sigmoid(LayerArithmetic.round(sqrtf(magnitude)) * (scaled < 0 ? -1 : (scaled > 0 ? 1 : 0)))
            return BF16.encode(gate)
        }
    }

    package func makeWorkspace(rows: Int) throws -> Workspace {
        let key = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        let value = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 2560, simdMatrix: simdMatrix)
        return try Workspace(
            device: device, rows: rows, keyPartialBytes: key.partialBytes, valuePartialBytes: value.partialBytes, ledger: ledger)
    }

    /// Encode PLE for `rows` tokens, updating `residual` (rows*10240 FP32, BF16 values) in place.
    /// `packedRows` holds 16 rows of 100 bytes per token in hash order; `state` is the host
    /// convolution state (9*10240 BF16). The caller keeps `projections` and `workspace` alive until join.
    package func encode(
        _ command: MTLCommandBuffer, residual: MTLBuffer, rows: Int, packedRows: UnsafeRawBufferPointer,
        state: [UInt16], projections: Projections, workspace: Workspace, decode: Bool
    ) throws {
        guard rows == workspace.rows, packedRows.count == rows * 1600, state.count == 92160,
            residual.length >= rows * 40960, residual.storageMode == .shared,
            command.status == .notEnqueued, command.commandQueue.device.registryID == device.registryID
        else {
            throw EngineError("Device PLE encode geometry or command differs")
        }
        workspace.packed.contents().copyMemory(from: packedRows.baseAddress!, byteCount: packedRows.count)
        state.withUnsafeBytes { workspace.state.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let wide = rows * 10240
        try dispatch(command, dequant, [workspace.packed, workspace.embedding], count: rows * 2560)
        let keyPlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 10240, simdMatrix: simdMatrix)
        try kernels.encodePinnedProjection(
            command, input: workspace.embedding, words: projections.keyWords,
            scales: projections.keyScales, biases: projections.keyBiases, output: workspace.keyRaw,
            partials: workspace.keyPartials, plan: keyPlan, simdMatrix: simdMatrix)
        let valuePlan = try kernels.pinnedProjectionPlan(rows: rows, width: 2560, outputs: 2560, simdMatrix: simdMatrix)
        try kernels.encodePinnedProjection(
            command, input: workspace.embedding, words: projections.valueWords,
            scales: projections.valueScales, biases: projections.valueBiases, output: workspace.value,
            partials: workspace.valuePartials, plan: valuePlan, simdMatrix: simdMatrix)
        try dispatch(command, narrow, [workspace.keyRaw, workspace.key16], count: wide)
        try reductions.encodeGroupedRMS(
            command, input: workspace.key16, weights: normKey, output: workspace.keyNorm,
            rows: rows, width: 10240, groupSize: 2560)
        try dispatch(command, narrow, [residual, workspace.hidden16], count: wide)
        try reductions.encodeGroupedRMS(
            command, input: workspace.hidden16, weights: normQuery, output: workspace.query,
            rows: rows, width: 10240, groupSize: 2560)
        try dispatch(command, narrow, [workspace.keyNorm, workspace.key16], count: wide)
        try dispatch(command, narrow, [workspace.query, workspace.hidden16], count: wide)
        try reductions.encodePLEDot(command, key: workspace.key16, query: workspace.hidden16, output: workspace.dots, rows: rows * 4)
        try dispatch(command, gate, [workspace.dots, gateTable, workspace.value, workspace.gated, workspace.gated16], count: wide)
        try reductions.encodeGroupedRMS(
            command, input: workspace.gated16, weights: normConv, output: workspace.normalized,
            rows: rows, width: 10240, groupSize: 2560)
        try dispatch(
            command, convolution, [workspace.state, workspace.normalized, convolutionWeights, workspace.gated, siluTable, residual],
            count: wide)
        try dispatch(command, stateTail, [workspace.state, workspace.normalized, workspace.nextState], count: 92160, extra: UInt32(rows))
        if decode {
            counts.decodeTokens += 1
        } else {
            counts.prefillTiles += 1
            counts.prefillRows += UInt64(rows)
        }
    }

    /// The convolution state after the joined command; adopt only when the PLE result is used.
    package func nextState(_ workspace: Workspace) -> [UInt16] {
        Array(UnsafeBufferPointer(start: workspace.nextState.contents().assumingMemoryBound(to: UInt16.self), count: 92160))
    }

    func recordFallback() { counts.hostFallbackTiles += 1 }
    func recordLookahead() { counts.lookaheadScopes += 1 }
    func recordRows(requested: Int, read: Int, waitSeconds: Double, decode: Bool) {
        counts.rowsRequested += UInt64(requested)
        counts.rowsRead += UInt64(read)
        counts.rowWaitSeconds += waitSeconds
        if decode { counts.decodeRowWaitSeconds += waitSeconds } else { counts.prefillRowWaitSeconds += waitSeconds }
    }

    private func dispatch(
        _ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer],
        count: Int, extra: UInt32? = nil
    ) throws {
        guard count > 0, let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Device PLE encoder unavailable")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: buffers.count)
        if var extra { encoder.setBytes(&extra, length: 4, index: buffers.count + 1) }
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        // BF16.encode: round to nearest even; every NaN keeps its upper payload and becomes quiet.
        inline ushort ple_key(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
            return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
        inline float ple_float(ushort bits) { return as_type<float>(uint(bits) << 16); }
        inline float ple_round(float x) { return ple_float(ple_key(x)); }
        // Host arithmetic is AArch64 FMUL/FADD/FMADD with FPCR.DN = 0: a NaN operand propagates
        // quieted, signaling NaNs first, in operand order (FMADD checks the addend first), and an
        // invalid operation yields the default NaN 0x7fc00000. The GPU's canonical NaN differs, so
        // these helpers reproduce the host rule exactly; finite arithmetic is unchanged.
        inline bool ple_snan(float x) { return isnan(x) && (as_type<uint>(x) & 0x00400000u) == 0u; }
        inline float ple_quiet(float x) { return as_type<float>(as_type<uint>(x) | 0x00400000u); }
        inline float ple_default(float r) { return isnan(r) ? as_type<float>(0x7fc00000u) : r; }
        inline float ple_mul(float a, float b) {
            if (ple_snan(a)) return ple_quiet(a);
            if (ple_snan(b)) return ple_quiet(b);
            if (isnan(a)) return a;
            if (isnan(b)) return b;
            return ple_default(a * b);
        }
        inline float ple_add(float a, float b) {
            if (ple_snan(a)) return ple_quiet(a);
            if (ple_snan(b)) return ple_quiet(b);
            if (isnan(a)) return a;
            if (isnan(b)) return b;
            return ple_default(a + b);
        }
        // c + a*b as FMADD(addend c, op1 a, op2 b).
        inline float ple_fma(float a, float b, float c) {
            bool invalid = (isinf(a) && b == 0.0f) || (a == 0.0f && isinf(b));
            if (ple_snan(c)) return ple_quiet(c);
            if (ple_snan(a)) return ple_quiet(a);
            if (ple_snan(b)) return ple_quiet(b);
            if (isnan(c)) return invalid ? as_type<float>(0x7fc00000u) : c;
            if (isnan(a)) return a;
            if (isnan(b)) return b;
            return ple_default(fma(a, b, c));
        }

        // ReferenceWeights.dequantizedRow over 16 rows of 100 bytes per token: 80 bytes of nibbles,
        // then five BF16 scales and five BF16 biases (group 32). nibble*scale is exact in FP32.
        kernel void ple_device_dequant(device const uchar *packed [[buffer(0)]], device ushort *embedding [[buffer(1)]],
            constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint token = i / 2560u, j = i % 2560u, row = j / 160u, k = j % 160u;
            uint base = (token * 16u + row) * 100u, w = base + (k / 8u) * 4u;
            uint word = uint(packed[w]) | (uint(packed[w + 1u]) << 8) | (uint(packed[w + 2u]) << 16) | (uint(packed[w + 3u]) << 24);
            float nibble = float((word >> ((k % 8u) * 4u)) & 15u);
            uint g = base + 80u + (k / 32u) * 2u, h = base + 90u + (k / 32u) * 2u;
            float scale = ple_float(ushort(uint(packed[g]) | (uint(packed[g + 1u]) << 8)));
            float bias = ple_float(ushort(uint(packed[h]) | (uint(packed[h + 1u]) << 8)));
            embedding[i] = ple_key(ple_add(ple_mul(nibble, scale), bias));
        }
        // FP32 -> BF16 bits, one thread per value (BF16.encode).
        kernel void ple_device_narrow(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]],
            constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < count) y[i] = ple_key(x[i]);
        }
        // gate = table[BF16(dot)], then round(gate * value) per branch column.
        kernel void ple_device_gate(device const float *dots [[buffer(0)]], device const ushort *table [[buffer(1)]],
            device const float *value [[buffer(2)]], device float *gated [[buffer(3)]], device ushort *gated16 [[buffer(4)]],
            constant uint &count [[buffer(5)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint token = i / 10240u, branch = (i % 10240u) / 2560u, column = i % 2560u;
            float g = ple_float(table[ple_key(dots[token * 4u + branch])]);
            ushort key = ple_key(ple_mul(g, value[token * 2560u + column]));
            gated[i] = ple_float(key); gated16[i] = key;
        }
        // Four FMA taps at stride 3 over [9 state rows | normalized rows], SiLU table, gated sum,
        // then the residual addition, each rounded to BF16 exactly as ReferencePLE and its caller.
        kernel void ple_device_conv(device const ushort *state [[buffer(0)]], device const float *normalized [[buffer(1)]],
            device const float *weights [[buffer(2)]], device const float *gated [[buffer(3)]], device const ushort *silu [[buffer(4)]],
            device float *residual [[buffer(5)]], constant uint &count [[buffer(6)]], uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint token = i / 10240u, column = i % 10240u;
            float sum = 0.0f;
            for (uint tap = 0u; tap < 4u; ++tap) {
                uint k = token + tap * 3u;
                float x = k < 9u ? ple_float(state[k * 10240u + column]) : normalized[(k - 9u) * 10240u + column];
                sum = ple_fma(x, weights[column * 4u + tap], sum);
            }
            float activated = ple_float(silu[ple_key(sum)]);
            float output = ple_round(ple_add(gated[i], activated));
            residual[i] = ple_round(ple_add(residual[i], output));
        }
        // state = BF16 of the last nine rows of [state | normalized].
        kernel void ple_device_state(device const ushort *state [[buffer(0)]], device const float *normalized [[buffer(1)]],
            device ushort *next [[buffer(2)]], constant uint &count [[buffer(3)]], constant uint &rows [[buffer(4)]],
            uint i [[thread_position_in_grid]]) {
            if (i >= count) return;
            uint row = i / 10240u, column = i % 10240u, source = rows + row;
            float x = source < 9u ? ple_float(state[source * 10240u + column]) : normalized[(source - 9u) * 10240u + column];
            next[i] = ple_key(x);
        }
        """#
}
