import Foundation
import Metal

/// GGUF-native quantized matrices on Metal, ported from ggml-metal (llama.cpp commit `b72050d03`, MIT). Weights are a
/// row-major ggml matrix of `outputs` rows by `width` inputs in `type`'s blocks (GGUF dims [width, outputs]);
/// activations and results are f32. The kernels are ggml's: `mul_mv` per type for a few input rows (one threadgroup row
/// per input row, ggml's ne11 path), and the simdgroup `kernel_mul_mm` (64x32 tiles, weights and activations staged as
/// half, or bfloat for BF16) for more. ggml's threadgroup shapes, rows per simdgroup and IQ split mode are kept; two
/// changes make any output count safe on an exactly sized weight buffer: rows past the matrix are not read (ggml reads
/// past its end for the last tile) and mul_mm's partial tiles are stored a float at a time. The IQ codebooks come from
/// `GGUFQuant`, the CPU reference that `gguf_dequantize` checks against gguf-py.
package final class GGMLMatrixKernels {
    /// ggml type ids this class runs: F32 0, F16 1, BF16 30, Q8_0 8, Q4_0 2, Q4_1 3, Q5_0 6, Q5_1 7, Q2_0 42, Q2_K 10,
    /// Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14, IQ1_S 19, IQ1_M 29, IQ2_XXS 16, IQ2_XS 17, IQ2_S 22, IQ3_XXS 18, IQ3_S 21,
    /// IQ4_NL 20, IQ4_XS 23, MXFP4 39. Q2_0 comes from llama.cpp commit `889f2ec9` (MIT), its kernels ported as the others.
    package static let types: Set<Int> = Set(kernels.keys)
    /// ggml's kernel name and its mul_mv simdgroups, rows per simdgroup and threadgroup memory (ggml-metal-device.cpp
    /// `get_pipeline_mul_mv`; the float types choose theirs by width). IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S must keep 2
    /// simdgroups: their kernels load the codebook into threadgroup memory as `(32 * sgitg + tiisg) * nval`.
    private static let kernels: [Int: (name: String, simdgroups: Int, rows: Int, memory: Int)] = [
        0: ("f32", 0, 0, 0), 1: ("f16", 0, 0, 0), 30: ("bf16", 0, 0, 0),
        2: ("q4_0", 2, 4, 0), 3: ("q4_1", 2, 4, 0), 42: ("q2_0", 2, 8, 0), 6: ("q5_0", 2, 4, 0), 7: ("q5_1", 2, 4, 0),
        8: ("q8_0", 4, 2, 256),
        10: ("q2_K", 2, 4, 0), 11: ("q3_K", 2, 2, 0), 12: ("q4_K", 2, 2, 0), 13: ("q5_K", 2, 1, 0), 14: ("q6_K", 2, 2, 0),
        16: ("iq2_xxs", 2, 4, 256 * 8 + 128), 17: ("iq2_xs", 2, 4, 512 * 8 + 128), 22: ("iq2_s", 2, 4, 0),
        18: ("iq3_xxs", 2, 4, 256 * 4 + 128), 21: ("iq3_s", 2, 4, 512 * 4), 19: ("iq1_s", 2, 4, 0), 29: ("iq1_m", 2, 4, 0),
        20: ("iq4_nl", 2, 2, 32 * 4), 23: ("iq4_xs", 2, 2, 32 * 4), 39: ("mxfp4", 2, 2, 32 * 4),
    ]
    /// The 256-block IQ types whose mul_mv splits short rows (fewer than 32 chunks of 32) across lanes, eight rows each.
    private static let splits: Set<Int> = [16, 17, 18, 19, 21, 22, 29]
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    /// The family prefill benchmark's profile: given each dispatch's encoder and kernel ("mm_12", "mv_12", "mvbatch_12"
    /// by ggml type), the encoder to encode it into.
    package var willDispatch: ((MTLComputeCommandEncoder, String) throws -> MTLComputeCommandEncoder)?

    package init(device: MTLDevice) throws {
        self.device = device
        // Safe math, unlike ggml-metal's default fast math: on GLM-5.3 it is as fast (the same GPU time for a 316-token
        // prompt and 16 decoded tokens) and closer to the reference implementation. Qwen's kernels do not use this library.
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1  // bfloat
        options.mathMode = .safe
        library = try device.makeLibrary(source: Self.header + Self.tables() + Self.source, options: options)
    }

    /// output[r][o] = sum_i W[o][i] * input[r][i] for r in 0..<rows (input rows `width` f32 apart, output rows
    /// `outputs` f32 apart). Small `rows` (decode, up to 8) use ggml's mul_mv kernels, larger ones mul_mm.
    /// Validates sizes and alignment and throws EngineError on misuse. Encodes into the caller's encoder.
    /// Offsets are multiples of 16 bytes; rows x outputs stays below 2^31 (ggml's int32 output indexing, both paths).
    package func encode(
        _ encoder: MTLComputeCommandEncoder, type: Int, weights: MTLBuffer, weightOffset: Int,
        width: Int, outputs: Int, input: MTLBuffer, inputOffset: Int, rows: Int,
        output: MTLBuffer, outputOffset: Int
    ) throws {
        let shape = try self.shape(type: type, width: width, outputs: outputs, rows: rows)
        try check(
            shape, weights: weights, weightOffset: weightOffset, input: input, inputOffset: inputOffset, output: output,
            outputOffset: outputOffset)
        let pipeline: MTLComputePipelineState
        let args: [UInt8]
        let memory: Int
        let groups: MTLSize
        let simdgroups: Int
        if rows > 8 && width >= 64 {
            let edge = outputs % 64 != 0 || rows % 32 != 0
            pipeline = try self.pipeline(
                "kernel_mul_mm_\(shape.kernel.name)_f32", [(700, width % 32 != 0 ? 1 : 0), (701, edge ? 1 : 0)],
                shorts: [702, 703, 704, 705], simdgroups: 4)
            args = Self.kargs([
                (width, 4), (1, 4), (shape.rowBytes, 8), (shape.weightBytes, 8), (shape.weightBytes, 8), (1, 4), (4, 8), (width * 4, 8),
                (shape.inputBytes, 8), (shape.inputBytes, 8), (outputs, 4), (rows, 4), (1, 2), (1, 2),
            ])
            memory = edge ? 8192 : 4096 + 2048
            groups = MTLSize(width: (rows + 31) / 32, height: (outputs + 63) / 64, depth: 1)
            simdgroups = 4
        } else {
            (pipeline, args, memory, groups, simdgroups) = try mulMV(
                shape, type: type, width: width, outputs: outputs, rows: rows, variant: "")
        }
        let encoder = try willDispatch?(encoder, "\(rows > 8 && width >= 64 ? "mm" : "mv")_\(type)") ?? encoder
        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(weights, offset: weightOffset, index: 1)
        encoder.setBuffer(input, offset: inputOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        // Always set: the length is encoder state, and a kernel that uses none would inherit the previous dispatch's.
        encoder.setThreadgroupMemoryLength(max(memory, 16), index: 0)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: simdgroups, depth: 1))
    }

    /// Most matrices one `encodeBatch` dispatch takes (`ORIGAMI_MV_TABLE`).
    package static let batchCapacity = 128
    /// Whether `encodeBatch` runs here: the weights are reached by GPU address (argument buffers tier 2), and only
    /// quantized types have a batch kernel.
    package func batches(type: Int) -> Bool {
        device.argumentBuffersSupport == .tier2 && (GGUFQuant.sizes[type]?.block ?? 1) > 1 && Self.kernels[type] != nil
    }

    /// `encode` of up to `batchCapacity` matrices of one type and shape in one dispatch, the same bits as one `encode`
    /// each: output z = W_z · input z, where W_z is at `matrices[z]` (any buffers), input z at `inputOffset + z *
    /// inputStride` (0 shares one input) and output z at `outputOffset + z * outputStride`; up to 8 rows (mul_mv).
    /// For a layer's routed experts and a stacked tensor's heads, which are otherwise a dispatch each.
    package func encodeBatch(
        _ encoder: MTLComputeCommandEncoder, type: Int, matrices: [(buffer: MTLBuffer, offset: Int)], width: Int, outputs: Int,
        input: MTLBuffer, inputOffset: Int, inputStride: Int, rows: Int,
        output: MTLBuffer, outputOffset: Int, outputStride: Int
    ) throws {
        let shape = try self.shape(type: type, width: width, outputs: outputs, rows: rows)
        guard batches(type: type), rows <= 8, (1...Self.batchCapacity).contains(matrices.count), inputStride >= 0, inputStride % 16 == 0,
            outputStride >= shape.outputBytes || matrices.count == 1
        else {
            throw EngineError(
                "A batch of \(matrices.count) \(GGUFQuant.name(type)) matrices of \(outputs) x \(width) cannot take \(rows) rows "
                    + "with input stride \(inputStride) and output stride \(outputStride)")
        }
        for (z, matrix) in matrices.enumerated() {
            try check(
                shape, weights: matrix.buffer, weightOffset: matrix.offset, input: input, inputOffset: inputOffset + z * inputStride,
                output: output, outputOffset: outputOffset + z * outputStride)
        }
        let (pipeline, args, memory, groups, simdgroups) = try mulMV(
            shape, type: type, width: width, outputs: outputs, rows: rows, variant: "batch_")
        // origami_mv_table: the strides, then a GPU address per matrix.
        var table = [UInt64](repeating: 0, count: 2 + Self.batchCapacity)
        table[0] = UInt64(inputStride)
        table[1] = UInt64(outputStride)
        var used: [ObjectIdentifier: MTLBuffer] = [:]
        for (z, matrix) in matrices.enumerated() {
            table[2 + z] = matrix.buffer.gpuAddress + UInt64(matrix.offset)
            used[ObjectIdentifier(matrix.buffer)] = matrix.buffer
        }
        let encoder = try willDispatch?(encoder, "mvbatch_\(type)") ?? encoder
        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(input, offset: inputOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        table.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        for buffer in used.values { encoder.useResource(buffer, usage: .read) }
        encoder.setThreadgroupMemoryLength(max(memory, 16), index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups.width, height: groups.height, depth: matrices.count),
            threadsPerThreadgroup: MTLSize(width: 32, height: simdgroups, depth: 1))
    }

    /// Most matrices one `encodeRowsBatch` dispatch takes (`ORIGAMI_MM_TABLE`).
    package static let rowsBatchCapacity = 128

    /// `encode` of up to `rowsBatchCapacity` matrices of one type and shape, each over its own rows, in one dispatch, the
    /// same bits as one `encode` each: output rows z = W_z · input rows z, where W_z is at `matrices[z]`, its
    /// `rows[z]` input rows at `inputOffsets[z]` of `input` and its output rows at `outputOffsets[z]` of `output`. Only for
    /// what `encode` runs on mul_mm (more than 8 rows each, width 64 or more): a prompt's routed experts.
    package func encodeRowsBatch(
        _ encoder: MTLComputeCommandEncoder, type: Int, matrices: [(buffer: MTLBuffer, offset: Int)], width: Int, outputs: Int,
        input: MTLBuffer, inputOffsets: [Int], rows: [Int], output: MTLBuffer, outputOffsets: [Int]
    ) throws {
        let count = matrices.count
        guard (1...Self.rowsBatchCapacity).contains(count), inputOffsets.count == count, rows.count == count, outputOffsets.count == count,
            width >= 64, rows.allSatisfy({ $0 > 8 })
        else {
            throw EngineError("A batch of \(count) \(GGUFQuant.name(type)) matrices of \(outputs) x \(width) cannot take rows \(rows)")
        }
        var shape = try self.shape(type: type, width: width, outputs: outputs, rows: rows[0])
        for z in 0..<count {
            shape = try self.shape(type: type, width: width, outputs: outputs, rows: rows[z])
            try check(
                shape, weights: matrices[z].buffer, weightOffset: matrices[z].offset, input: input, inputOffset: inputOffsets[z],
                output: output,
                outputOffset: outputOffsets[z])
        }
        // Every partial tile is stored as `encode`'s edge kernel stores it; a whole tile as both store it.
        let pipeline = try self.pipeline(
            "kernel_mul_mm_batch_\(shape.kernel.name)_f32", [(700, width % 32 != 0 ? 1 : 0), (701, 1)], shorts: [702, 703, 704, 705],
            simdgroups: 4)
        let most = rows.max()!
        // ne1 is each matrix's own (the kernel sets it); the input's batch strides are not read.
        let args = Self.kargs([
            (width, 4), (1, 4), (shape.rowBytes, 8), (shape.weightBytes, 8), (shape.weightBytes, 8), (1, 4), (4, 8), (width * 4, 8),
            (most * width * 4, 8), (most * width * 4, 8), (outputs, 4), (most, 4), (1, 2), (1, 2),
        ])
        // origami_mm_table: the weights' GPU addresses, then input and output offsets (u64) and row counts (i32).
        let n = Self.rowsBatchCapacity
        var table = [UInt64](repeating: 0, count: 3 * n + n / 2)
        table.withUnsafeMutableBytes { raw in
            let words = raw.bindMemory(to: UInt64.self)
            let ints = (raw.baseAddress! + 3 * n * 8).bindMemory(to: Int32.self, capacity: n)
            for z in 0..<count {
                words[z] = matrices[z].buffer.gpuAddress + UInt64(matrices[z].offset)
                words[n + z] = UInt64(inputOffsets[z])
                words[2 * n + z] = UInt64(outputOffsets[z])
                ints[z] = Int32(rows[z])
            }
        }
        var used: [ObjectIdentifier: MTLBuffer] = [:]
        for matrix in matrices { used[ObjectIdentifier(matrix.buffer)] = matrix.buffer }
        let encoder = try willDispatch?(encoder, "mmbatch_\(type)") ?? encoder
        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        table.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        for buffer in used.values { encoder.useResource(buffer, usage: .read) }
        encoder.setThreadgroupMemoryLength(8192, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (most + 31) / 32, height: (outputs + 63) / 64, depth: count),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    /// `encode` of up to `rowsBatchCapacity` matrices of one type and shape, each over its own `rows` (8 or fewer, the
    /// same for all) in one dispatch, the same bits as one `encode` each (mul_mv): input rows z at `inputOffsets[z]` of
    /// `input`, output rows z at `outputOffsets[z]` of `output`. A prompt's routed experts with few rows in a slice.
    package func encodeFewRowsBatch(
        _ encoder: MTLComputeCommandEncoder, type: Int, matrices: [(buffer: MTLBuffer, offset: Int)], width: Int, outputs: Int,
        input: MTLBuffer, inputOffsets: [Int], rows: Int, output: MTLBuffer, outputOffsets: [Int]
    ) throws {
        let count = matrices.count
        let shape = try self.shape(type: type, width: width, outputs: outputs, rows: rows)
        guard batches(type: type), rows <= 8, (1...Self.rowsBatchCapacity).contains(count), inputOffsets.count == count,
            outputOffsets.count == count
        else {
            throw EngineError("A batch of \(count) \(GGUFQuant.name(type)) matrices of \(outputs) x \(width) cannot take \(rows) rows each")
        }
        for z in 0..<count {
            try check(
                shape, weights: matrices[z].buffer, weightOffset: matrices[z].offset, input: input, inputOffset: inputOffsets[z],
                output: output,
                outputOffset: outputOffsets[z])
        }
        let (pipeline, args, memory, groups, simdgroups) = try mulMV(
            shape, type: type, width: width, outputs: outputs, rows: rows, variant: "rows_")
        // origami_mv_rows_table: the weights' GPU addresses, then input and output offsets.
        let n = Self.rowsBatchCapacity
        var table = [UInt64](repeating: 0, count: 3 * n)
        var used: [ObjectIdentifier: MTLBuffer] = [:]
        for z in 0..<count {
            table[z] = matrices[z].buffer.gpuAddress + UInt64(matrices[z].offset)
            table[n + z] = UInt64(inputOffsets[z])
            table[2 * n + z] = UInt64(outputOffsets[z])
            used[ObjectIdentifier(matrices[z].buffer)] = matrices[z].buffer
        }
        let encoder = try willDispatch?(encoder, "mvrows_\(type)") ?? encoder
        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        table.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        for buffer in used.values { encoder.useResource(buffer, usage: .read) }
        encoder.setThreadgroupMemoryLength(max(memory, 16), index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups.width, height: groups.height, depth: count),
            threadsPerThreadgroup: MTLSize(width: 32, height: simdgroups, depth: 1))
    }

    /// `encodeBatch` of one input row for up to ten matrices that are not known when it is encoded (queued decode): matrix z
    /// is at the GPU address `records[z]` (ten u64s, written before the dispatch runs) plus `part`, its input at `inputOffset
    /// + ranks[z] * inputStride` of `input` and its output at `outputOffset + ranks[z] * outputStride` of `output` (`ranks`:
    /// ten UInt32s). The grid is read when it runs: three UInt32s at `grid`, any width, the number of matrices (at most ten)
    /// and 1. The same bits as `encode` per matrix. The caller makes the matrices' buffers resident and keeps the outputs of
    /// different ranks apart.
    package func encodeGathered(
        _ encoder: MTLComputeCommandEncoder, type: Int, part: Int, width: Int, outputs: Int, records: MTLBuffer, ranks: MTLBuffer,
        grid: (buffer: MTLBuffer, offset: Int), input: MTLBuffer, inputOffset: Int, inputStride: Int,
        output: MTLBuffer, outputOffset: Int, outputStride: Int
    ) throws {
        let shape = try self.shape(type: type, width: width, outputs: outputs, rows: 1)
        guard batches(type: type), part >= 0, part % 16 == 0, inputStride >= 0, inputStride % 16 == 0, outputStride >= shape.outputBytes,
            records.length >= 80, ranks.length >= 40, grid.offset >= 0, grid.offset % 4 == 0, grid.offset + 12 <= grid.buffer.length,
            inputOffset >= 0, inputOffset % 16 == 0, inputOffset + 9 * inputStride + shape.inputBytes <= input.length,
            outputOffset >= 0, outputOffset % 16 == 0, outputOffset + 9 * outputStride + shape.outputBytes <= output.length,
            [records, ranks, grid.buffer, input, output].allSatisfy({ $0.device.registryID == device.registryID }), input !== output
        else {
            throw EngineError("Invalid gathered \(GGUFQuant.name(type)) matrices of \(outputs) x \(width)")
        }
        let (pipeline, args, memory, groups, simdgroups) = try mulMV(
            shape, type: type, width: width, outputs: outputs, rows: 1, variant: "gather_")
        var table = [UInt64(part), UInt64(inputStride), UInt64(outputStride), UInt64(groups.width)]
        let encoder = try willDispatch?(encoder, "mvgather_\(type)") ?? encoder
        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(input, offset: inputOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBytes(&table, length: 32, index: 4)
        encoder.setBuffer(records, offset: 0, index: 5)
        encoder.setBuffer(ranks, offset: 0, index: 6)
        encoder.setThreadgroupMemoryLength(max(memory, 16), index: 0)
        encoder.dispatchThreadgroups(
            indirectBuffer: grid.buffer, indirectBufferOffset: grid.offset,
            threadsPerThreadgroup: MTLSize(width: 32, height: simdgroups, depth: 1))
    }

    private struct Shape {
        let kernel: (name: String, simdgroups: Int, rows: Int, memory: Int), type, block, bytes, rowBytes, weightBytes, inputBytes,
            outputBytes: Int
    }
    private func shape(type: Int, width: Int, outputs: Int, rows: Int) throws -> Shape {
        guard let kernel = Self.kernels[type], let (block, bytes) = GGUFQuant.sizes[type] else {
            throw EngineError("ggml type \(GGUFQuant.name(type)) has no Metal matrix kernel")
        }
        let limit = 1 << 24
        guard width > 0, outputs > 0, rows > 0, width <= limit, outputs <= limit, rows <= limit, width % block == 0,
            rows * outputs <= Int(Int32.max)
        else {
            throw EngineError("A \(GGUFQuant.name(type)) matrix of \(outputs) x \(width) cannot take \(rows) rows")
        }
        let rowBytes = width / block * bytes
        return Shape(
            kernel: kernel, type: type, block: block, bytes: bytes, rowBytes: rowBytes, weightBytes: outputs * rowBytes,
            inputBytes: rows * width * 4, outputBytes: rows * outputs * 4)
    }
    private func check(
        _ shape: Shape, weights: MTLBuffer, weightOffset: Int, input: MTLBuffer, inputOffset: Int, output: MTLBuffer, outputOffset: Int
    ) throws {
        func fits(_ buffer: MTLBuffer, _ offset: Int, _ length: Int) -> Bool {
            buffer.device.registryID == device.registryID && offset >= 0 && offset % 16 == 0 && offset <= buffer.length - length
        }
        func apart(_ other: MTLBuffer, _ offset: Int, _ length: Int) -> Bool {
            other !== output || offset + length <= outputOffset || outputOffset + shape.outputBytes <= offset
        }
        guard fits(weights, weightOffset, shape.weightBytes), fits(input, inputOffset, shape.inputBytes),
            fits(output, outputOffset, shape.outputBytes),
            apart(weights, weightOffset, shape.weightBytes), apart(input, inputOffset, shape.inputBytes)
        else {
            throw EngineError(
                "Invalid \(GGUFQuant.name(shape.type)) matrix buffers: \(shape.weightBytes) weight, \(shape.inputBytes) input and \(shape.outputBytes) "
                    + "output bytes at offsets \(weightOffset), \(inputOffset), \(outputOffset) (16-byte aligned, on this device, output apart)"
            )
        }
    }
    /// ggml's mul_mv choice for a matrix of `type` (`variant` "batch_", "rows_" or "gather_": its `kernel_mul_mv_batch_`,
    /// `kernel_mul_mv_rows_` or `kernel_mul_mv_gather_` twin).
    private func mulMV(_ shape: Shape, type: Int, width: Int, outputs: Int, rows: Int, variant: String) throws
        -> (MTLComputePipelineState, [UInt8], Int, MTLSize, Int)
    {
        var name = "kernel_mul_mv_\(variant)\(shape.kernel.name)_f32"
        var sg = shape.kernel.simdgroups
        var nr0 = shape.kernel.rows
        var smem = shape.kernel.memory
        var split = false
        if shape.block == 1 {
            if width < 32 {
                sg = 1
                nr0 = 32
                name += "_short"
            } else {
                sg = min(4, (width + 127) / 128)
                nr0 = 2
                smem = 32 * 4 * nr0
                if width % 4 == 0 { name += "_4" }
            }
        } else if Self.splits.contains(type), width / 32 < 32, 32 % (width / 32) == 0 {
            nr0 = 8
            split = true
        }
        let pipeline = try self.pipeline(name, [(600, sg), (605, split ? 1 : 0)], shorts: [602, 603, 604], simdgroups: sg)
        let args = Self.kargs([
            (width, 4), (outputs, 4), (1, 4), (shape.bytes, 8), (shape.rowBytes, 8), (shape.weightBytes, 8), (shape.weightBytes, 8),
            (width, 4), (rows, 4), (1, 4), (4, 8), (width * 4, 8), (shape.inputBytes, 8), (shape.inputBytes, 8),
            (outputs, 4), (rows, 4), (nr0, 4), (1, 2), (1, 2),
        ])
        // The float kernels and Q8_0 split a threadgroup's rows across its simdgroups; the others give each simdgroup its own.
        let perGroup = shape.block == 1 || type == 8 ? nr0 : nr0 * sg
        return (pipeline, args, smem, MTLSize(width: (outputs + perGroup - 1) / perGroup, height: rows, depth: 1), sg)
    }

    /// The pipeline of `name` with ggml's function constants: `values` (index, value; bools as 0 or 1 at the bool
    /// indices 605, 700, 701), and 1 for the batch constants `shorts` (ne12, ne13, r2, r3: Origami has no batch dims).
    private func pipeline(_ name: String, _ values: [(Int, Int)], shorts: [Int], simdgroups: Int) throws -> MTLComputePipelineState {
        let key = name + values.map { ":\($0.1)" }.joined()
        if let cached = pipelines[key] { return cached }
        let constants = MTLFunctionConstantValues()
        for (index, value) in values + shorts.map({ ($0, 1) }) {
            if [605, 700, 701].contains(index) {
                var flag = value != 0
                constants.setConstantValue(&flag, type: .bool, index: index)
            } else {
                var short = Int16(value)
                constants.setConstantValue(&short, type: .short, index: index)
            }
        }
        let state = try device.makeComputePipelineState(function: library.makeFunction(name: name, constantValues: constants))
        guard state.threadExecutionWidth == 32, state.maxTotalThreadsPerThreadgroup >= 32 * simdgroups else {
            throw EngineError(
                "\(name) needs \(simdgroups) simdgroups of 32 threads; this GPU offers \(state.maxTotalThreadsPerThreadgroup) threads of width \(state.threadExecutionWidth)"
            )
        }
        pipelines[key] = state
        return state
    }

    /// A ggml_metal_kargs struct as Metal lays it out: (value, byte size) fields at their natural alignment, padded to 8.
    private static func kargs(_ fields: [(Int, Int)]) -> [UInt8] {
        var bytes: [UInt8] = []
        for (value, size) in fields {
            bytes += repeatElement(0, count: (size - bytes.count % size) % size)
            bytes += withUnsafeBytes(of: Int64(value).littleEndian) { Array($0.prefix(size)) }
        }
        return bytes + repeatElement(0, count: (8 - bytes.count % 8) % 8)
    }

    /// ggml-common.h's tables, built from `GGUFQuant`'s codebooks: the IQ2/IQ3 grids pack one value per byte; IQ1's GPU
    /// grid holds value + 1 in nibbles (byte i: value i low, value i + 4 high).
    private static func tables() -> String {
        func table(_ type: String, _ name: String, _ values: [UInt64]) -> String {
            "constexpr constant static \(type) \(name)[\(values.count)] = {"
                + values.map { "0x" + String($0, radix: 16) }.joined(separator: ",") + "};\n"
        }
        func packed(_ grid: [UInt8], _ width: Int) -> [UInt64] {
            stride(from: 0, to: grid.count, by: width).map { (e: Int) -> UInt64 in
                (0..<width).reduce(UInt64(0)) { (word: UInt64, j: Int) -> UInt64 in word | UInt64(grid[e + j]) << UInt64(8 * j) }
            }
        }
        let iq1 = GGUFQuant.iq1sGrid
        let nibbles: [UInt8] = (0..<(iq1.count / 2)).map { (i: Int) -> UInt8 in
            let at = i / 4 * 8 + i % 4
            return iq1[at] | iq1[at + 4] << 4
        }
        let iq1gpu = packed(nibbles, 4)
        let tables: [String] = [
            table("uint8_t", "kmask_iq2xs", (0..<8).map { UInt64(1) << $0 }),
            table("uint8_t", "ksigns_iq2xs", GGUFQuant.ksigns.map(UInt64.init)),
            table("uint64_t", "iq2xxs_grid", packed(GGUFQuant.iq2xxsGrid, 8)),
            table("uint64_t", "iq2xs_grid", packed(GGUFQuant.iq2xsGrid, 8)),
            table("uint64_t", "iq2s_grid", packed(GGUFQuant.iq2sGrid, 8)),
            table("uint32_t", "iq3xxs_grid", packed(GGUFQuant.iq3xxsGrid, 4)),
            table("uint32_t", "iq3s_grid", packed(GGUFQuant.iq3sGrid, 4)), table("uint32_t", "iq1s_grid_gpu", iq1gpu),
        ]
        return tables.joined()
    }

    /// ggml-common.h's blocks (plain fields where ggml has unions), ggml-metal-impl.h's arguments and rows per
    /// simdgroup, and the parts of common.h the kernels use.
    private static let header = #"""
        // Ported from ggml-metal (llama.cpp commit b72050d03, MIT; see THIRD_PARTY_NOTICES.md). Upstream changes: `Origami:`.
        #include <metal_stdlib>
        using namespace metal;

        #define MIN(x, y) ((x) < (y) ? (x) : (y))
        #define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
        #define N_SIMDWIDTH 32
        typedef matrix<bfloat, 4, 4> bfloat4x4;
        typedef matrix<bfloat, 2, 4> bfloat2x4;

        #define QK_K 256
        #define K_SCALE_SIZE 12
        #define QK4_0 32
        #define QK2_0 64
        #define QK4_1 32
        #define QK5_0 32
        #define QK5_1 32
        #define QK8_0 32
        #define QK_MXFP4 32
        #define QK4_NL 32
        #define QK_NL 16
        #define IQ3S_N_SCALE QK_K/64
        #define IQ1S_DELTA 0.125f
        #define IQ1M_DELTA 0.125f

        typedef struct { half d; uint8_t qs[QK4_0 / 2]; } block_q4_0;
        typedef struct { half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
        typedef struct { half d; half m; uint8_t qs[QK4_1 / 2]; } block_q4_1;
        typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
        typedef struct { half d; uint8_t qh[4]; uint8_t qs[QK5_0 / 2]; } block_q5_0;
        typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[QK5_1 / 2]; } block_q5_1;
        typedef struct { half d; int8_t qs[QK8_0]; } block_q8_0;
        typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; half d; half dmin; } block_q2_K;
        typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; half d; } block_q3_K;
        typedef struct { half d; half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
        typedef struct { half d; half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;
        typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; half d; } block_q6_K;
        typedef struct { half d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
        typedef struct { half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
        typedef struct { half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
        typedef struct { half d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
        typedef struct { half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[IQ3S_N_SCALE]; } block_iq3_s;
        typedef struct { half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
        typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m;
        typedef union { half f16; uint16_t u16; } iq1m_scale_t;
        typedef struct { half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
        typedef struct { half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;

        #define N_R0_Q4_0 4
        #define N_R0_Q2_0 8
        #define N_R0_Q4_1 4
        #define N_R0_Q5_0 4
        #define N_R0_Q5_1 4
        #define N_R0_Q8_0 2
        #define N_R0_MXFP4 2
        #define N_R0_Q2_K 4
        #define N_R0_Q3_K 2
        #define N_R0_Q4_K 2
        #define N_R0_Q5_K 1
        #define N_R0_Q6_K 2
        #define N_R0_IQ1_S 4
        #define N_R0_IQ1_S_SPLIT 8
        #define N_R0_IQ1_M 4
        #define N_R0_IQ1_M_SPLIT 8
        #define N_R0_IQ2_XXS 4
        #define N_R0_IQ2_XXS_SPLIT 8
        #define N_R0_IQ2_XS 4
        #define N_R0_IQ2_XS_SPLIT 8
        #define N_R0_IQ2_S 4
        #define N_R0_IQ2_S_SPLIT 8
        #define N_R0_IQ3_XXS 4
        #define N_R0_IQ3_XXS_SPLIT 8
        #define N_R0_IQ3_S 4
        #define N_R0_IQ3_S_SPLIT 8
        #define N_R0_IQ4_NL 2
        #define N_R0_IQ4_XS 2

        #define FC_MUL_MV 600
        #define FC_MUL_MM 700

        typedef struct {
            int32_t  ne00;
            int32_t  ne01;
            int32_t  ne02;
            uint64_t nb00;
            uint64_t nb01;
            uint64_t nb02;
            uint64_t nb03;
            int32_t  ne10;
            int32_t  ne11;
            int32_t  ne12;
            uint64_t nb10;
            uint64_t nb11;
            uint64_t nb12;
            uint64_t nb13;
            int32_t  ne0;
            int32_t  ne1;
            int32_t  nr0;
            int16_t  r2;
            int16_t  r3;
        } ggml_metal_kargs_mul_mv;

        typedef struct {
            int32_t  ne00;
            int32_t  ne02;
            uint64_t nb01;
            uint64_t nb02;
            uint64_t nb03;
            int32_t  ne12;
            uint64_t nb10;
            uint64_t nb11;
            uint64_t nb12;
            uint64_t nb13;
            int32_t  ne0;
            int32_t  ne1;
            int16_t  r2;
            int16_t  r3;
        } ggml_metal_kargs_mul_mm;

        constexpr constant static float kvalues_iq4nl_f[16] = {
            -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
        };

        constexpr constant static float kvalues_mxfp4_f[16] = {
            0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
        };

        static inline float e8m0_to_fp32(uint8_t x) {
            uint32_t bits;

            if (x == 0) {
                bits = 0x00400000;
            } else {
                bits = (uint32_t) x << 23;
            }

            return as_type<float>(bits);
        }

        """#

    /// dequantize.h, mul_mv.metal and mul_mm.metal for the types above (ported from ggml-metal, MIT), kept as ggml has
    /// them (its comments, unused defines and dispatch switches included) so the port diffs cleanly against upstream,
    /// except where marked `Origami:`, where a row pointer is clamped with `MIN(..., args.ne01 - 1)`, a row loop also
    /// stops at `args.ne01`, and mul_mm's partial tiles are stored a float at a time.
    private static let source = #"""
        // ---- dequantize.h

        // Not a dequantization: f32 weights only fit the template.

        template <typename type4x4>
        void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
            reg = (type4x4)(*src);
        }

        template <typename type4x4>
        void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
            reg = (type4x4)(*src);
        }

        template <typename type4x4>
        void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
            reg = (type4x4)(*src);
        }

        template <typename type4x4>
        void dequantize_q2_0(device const block_q2_0 * xb, short il, thread type4x4 & reg) {
            device const uint8_t * qs = xb->qs;
            const float d = xb->d;

            const int byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
            float4x4 reg_f;

            for (int i = 0; i < 4; i++) {
                const uint8_t b = qs[byte_offset + i];
                reg_f[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
                reg_f[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
                reg_f[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
                reg_f[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
            device const uint16_t * qs = ((device const uint16_t *)xb + 1);
            const float d1 = il ? (xb->d / 16.h) : xb->d;
            const float d2 = d1 / 256.f;
            const float md = -8.h * xb->d;
            const ushort mask0 = il ? 0x00F0 : 0x000F;
            const ushort mask1 = mask0 << 8;

            float4x4 reg_f;

            for (int i = 0; i < 8; i++) {
                reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
                reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
            device const uint16_t * qs = ((device const uint16_t *)xb + 2);
            const float d1 = il ? (xb->d / 16.h) : xb->d;
            const float d2 = d1 / 256.f;
            const float  m = xb->m;
            const ushort mask0 = il ? 0x00F0 : 0x000F;
            const ushort mask1 = mask0 << 8;

            float4x4 reg_f;

            for (int i = 0; i < 8; i++) {
                reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
                reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
            device const uint16_t * qs = ((device const uint16_t *)xb + 3);
            const float d = xb->d;
            const float md = -16.h * xb->d;
            const ushort mask = il ? 0x00F0 : 0x000F;

            const uint32_t qh = *((device const uint32_t *)xb->qh);

            const int x_mv = il ? 4 : 0;

            const int gh_mv = il ? 12 : 0;
            const int gh_bk = il ?  0 : 4;

            float4x4 reg_f;

            for (int i = 0; i < 8; i++) {
                // extract the 5-th bits for x0 and x1
                const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
                const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

                // combine the 4-bits from qs with the 5th bit
                const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
                const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

                reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
                reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
            device const uint16_t * qs = ((device const uint16_t *)xb + 4);
            const float d = xb->d;
            const float m = xb->m;
            const ushort mask = il ? 0x00F0 : 0x000F;

            const uint32_t qh = *((device const uint32_t *)xb->qh);

            const int x_mv = il ? 4 : 0;

            const int gh_mv = il ? 12 : 0;
            const int gh_bk = il ?  0 : 4;

            float4x4 reg_f;

            for (int i = 0; i < 8; i++) {
                // extract the 5-th bits for x0 and x1
                const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
                const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

                // combine the 4-bits from qs with the 5th bit
                const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
                const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

                reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
                reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
            device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
            const float d = xb->d;

            float4x4 reg_f;

            for (int i = 0; i < 4; ++i) {
                reg_f[i] = float4(qs[4*il + i]) * d;
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
            device const uint8_t * q2 = (device const uint8_t *)xb->qs;

            const float d = e8m0_to_fp32(xb->e);
            const uint8_t shr = il >= 1 ? 4 : 0;

            float4x4 reg_f;

            for (int i = 0; i < 4; ++i) {
                reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
                reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
                reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
                reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
            }

            reg = (type4x4) reg_f;
        }

        template <typename type4x4>
        void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
            const float d = xb->d;
            const float min = xb->dmin;
            device const uint8_t * q = (device const uint8_t *)xb->qs;
            float dl, ml;
            uint8_t sc = xb->scales[il];

            q = q + 32*(il/8) + 16*(il&1);
            il = (il/2)%4;

            half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
            uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
            dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
            for (int i = 0; i < 16; ++i) {
                reg[i/4][i%4] = dl * (q[i] & mask) - ml;
            }
        }

        template <typename type4x4>
        void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
            const half d_all = xb->d;
            device const uint8_t * q = (device const uint8_t *)xb->qs;
            device const uint8_t * h = (device const uint8_t *)xb->hmask;
            device const int8_t * scales = (device const int8_t *)xb->scales;

            q = q + 32 * (il/8) + 16 * (il&1);
            h = h + 16 * (il&1);
            uint8_t m = 1 << (il/2);
            uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                         ((il/4)>0 ? 12  : 3);
            uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
            uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
            int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                                       : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
            float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
            const float ml = 4.f * dl;

            il = (il/2) & 3;
            const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
            const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
            dl *= coef;

            for (int i = 0; i < 16; ++i) {
                reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
            }
        }

        static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
            return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                         : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
        }

        template <typename type4x4>
        void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
            device const uchar * q = xb->qs;

            short is = (il/4) * 2;
            q = q + (il/4) * 32 + 16 * (il&1);
            il = il & 3;
            const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
            const float d   = il < 2 ? xb->d : xb->d / 16.h;
            const float min = xb->dmin;
            const float dl = d * sc[0];
            const float ml = min * sc[1];

            const ushort mask = il < 2 ? 0x0F : 0xF0;
            for (int i = 0; i < 16; ++i) {
                reg[i/4][i%4] = dl * (q[i] & mask) - ml;
            }
        }

        template <typename type4x4>
        void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
            device const uint8_t * q  = xb->qs;
            device const uint8_t * qh = xb->qh;

            short is = (il/4) * 2;
            q  = q + 32 * (il/4) + 16 * (il&1);
            qh = qh + 16 * (il&1);
            uint8_t ul = 1 << (il/2);
            il = il & 3;
            const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
            const float d = il < 2 ? xb->d : xb->d / 16.f;
            const float min = xb->dmin;
            const float dl = d * sc[0];
            const float ml = min * sc[1];

            const ushort mask  = il<2 ? 0x0F : 0xF0;
            const float qh_val = il<2 ? 16.f : 256.f;
            for (int i = 0; i < 16; ++i) {
                reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
            }
        }

        template <typename type4x4>
        void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
            const half d_all = xb->d;
            device const uint16_t * ql = (device const uint16_t *)xb->ql;
            device const uint16_t * qh = (device const uint16_t *)xb->qh;
            device const int8_t * scales = (device const int8_t *)xb->scales;

            ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
            qh = qh + 16*(il/8) + 8*(il&1);
            float sc = scales[(il%2) + 2 * ((il/2))];
            il = (il/2) & 3;

            const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
            const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
            const float ml = d_all * sc * 32.f;
            const float dl0 = d_all * sc;
            const float dl1 = dl0 / 256.f;
            const float dl2 = dl0 / (256.f * 256.f);
            const float dl3 = dl0 / (256.f * 256.f * 256.f);
            const uint8_t shr_h = il>2 ? 2 : 0;
            const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
            const uint8_t shr_l = il>1 ? 4 : 0;
            for (int i = 0; i < 4; ++i) {
                const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
                const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
                const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
                reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
                reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
                reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
            }
        }

        template <typename type4x4>
        void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const float d = xb->d;
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
            device const uint16_t * q2 = xb->qs + 4*ib32;
            const uint32_t aux32_g = q2[0] | (q2[1] << 16);
            const uint32_t aux32_s = q2[2] | (q2[3] << 16);
            thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
            const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
            constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
            uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
            for (int i = 0; i < 8; ++i) {
                reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
            }
            grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
            signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
            for (int i = 0; i < 8; ++i) {
                reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
            }
        }

        template <typename type4x4>
        void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const float d = xb->d;
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            device const uint16_t * q2 = xb->qs + 4*ib32;
            const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
            constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
            uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
            for (int i = 0; i < 8; ++i) {
                reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
            }
            grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
            signs = ksigns_iq2xs[q2[2*il+1] >> 9];
            for (int i = 0; i < 8; ++i) {
                reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
            }
        }

        template <typename type4x4>
        void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const float d = xb->d;
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            device const uint8_t * q3 = xb->qs + 8*ib32;
            device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
            constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
            constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
            uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
            for (int i = 0; i < 4; ++i) {
                reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
                reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
            }
            grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
            grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
            signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
            for (int i = 0; i < 4; ++i) {
                reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
                reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
            }
        }

        template <typename type4x4>
        void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const float d = xb->d;
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            device const uint8_t * qs = xb->qs + 8*ib32;
            device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
            const uint8_t qh = xb->qh[ib32] >> 4*il;
            const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
            constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
            for (int i = 0; i < 4; ++i) {
                reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
                reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
            }
            grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
            grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
            for (int i = 0; i < 4; ++i) {
                reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
                reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
            }
        }

        template <typename type4x4>
        void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const float d = xb->d;
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
            device const uint8_t * signs = qs + QK_K/8;
            const uint8_t qh = xb->qh[ib32] >> 4*il;
            const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
            constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
            for (int i = 0; i < 8; ++i) {
                reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
                reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
            }
        }

        template <typename type4x4>
        void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const int ib32 = il/2;
            il = il%2;
            const float d = xb->d;
            device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
            device const uint16_t * qh = xb->qh;
            const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
            const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
            const uint16_t h = qh[ib32] >> 6*il;
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
            for (int i = 0; i < 4; ++i) {
                reg[0][i] = dl * (grid1[i] & 0xf) + ml;
                reg[1][i] = dl * (grid1[i] >>  4) + ml;
                reg[2][i] = dl * (grid2[i] & 0xf) + ml;
                reg[3][i] = dl * (grid2[i] >>  4) + ml;
            }
        }

        template <typename type4x4>
        void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const int ib32 = il/2;
            il = il%2;
            device const uint16_t * sc = (device const uint16_t *)xb->scales;

            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const float d = scale.f16;

            device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
            device const uint8_t * qh = xb->qh + 2*ib32 + il;

            const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
            const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            for (int i = 0; i < 4; ++i) {
                reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
                reg[1][i] = dl * (grid1[i] >>  4) + ml1;
                reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
                reg[3][i] = dl * (grid2[i] >>  4) + ml2;
            }
        }

        template <typename type4x4>
        void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
            device const uint16_t * q4 = (device const uint16_t *)xb->qs;
            const float d = xb->d;
            uint32_t aux32;
            thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
            for (int i = 0; i < 4; ++i) {
                aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
                reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
                reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
                reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
                reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
            }
        }

        template <typename type4x4>
        void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
            // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
            const int ib32 = il/2;
            il = il%2;
            // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
            device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
            const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
            const float d = (float)xb->d * (ls - 32);
            uint32_t aux32;
            thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
            for (int i = 0; i < 4; ++i) {
                aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
                reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
                reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
                reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
                reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
            }
        }

        // ---- mul_mv.metal

        // Q2_0 dot: d * (sum_lo(y) + 2*sum_hi(y) - sumy) via per-bit conditional adds
        inline float block_q_n_dot_y(device const block_q2_0 * qb_curr, float sumy, thread float * yl, int il) {
            device const uint8_t * qs = qb_curr->qs + (il / 4);
            const uint8_t b0 = qs[0];
            const uint8_t b1 = qs[1];
            const uint8_t b2 = qs[2];
            const uint8_t b3 = qs[3];

            // Accumulate where low bit is set (bits 0,2,4,6 of each byte)
            float acc_lo = 0.0f;
            acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01));
            acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
            acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10));
            acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
            acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01));
            acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
            acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10));
            acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
            acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01));
            acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
            acc_lo += select(0.0f, yl[10], bool(b2 & 0x10));
            acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
            acc_lo += select(0.0f, yl[12], bool(b3 & 0x01));
            acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
            acc_lo += select(0.0f, yl[14], bool(b3 & 0x10));
            acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));

            // Accumulate where high bit is set (bits 1,3,5,7 of each byte)
            float acc_hi = 0.0f;
            acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02));
            acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
            acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20));
            acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
            acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02));
            acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
            acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20));
            acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
            acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02));
            acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
            acc_hi += select(0.0f, yl[10], bool(b2 & 0x20));
            acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
            acc_hi += select(0.0f, yl[12], bool(b3 & 0x02));
            acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
            acc_hi += select(0.0f, yl[14], bool(b3 & 0x20));
            acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));

            return qb_curr->d * (acc_lo + 2.0f * acc_hi - sumy);
        }

        // function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
        // il indicates where the q4 quants begin (0 or QK4_0/4)
        // we assume that the yl's have been multiplied with the appropriate scale factor
        // that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
        inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
            float d = qb_curr->d;

            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

            for (int i = 0; i < 8; i += 2) {
                acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        // function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
        // il indicates where the q4 quants begin (0 or QK4_0/4)
        // we assume that the yl's have been multiplied with the appropriate scale factor
        // that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
        inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
            float d = qb_curr->d;
            float m = qb_curr->m;

            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

            for (int i = 0; i < 8; i+=2) {
                acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
        }

        // function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
        // il indicates where the q5 quants begin (0 or QK5_0/4)
        // we assume that the yl's have been multiplied with the appropriate scale factor
        // that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
        inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
            float d = qb_curr->d;

            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
                   const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

            for (int i = 0; i < 8; i+=2) {
                acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
                acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
                acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
                acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
            }

            return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        // function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
        // il indicates where the q5 quants begin (0 or QK5_1/4)
        // we assume that the yl's have been multiplied with the appropriate scale factor
        // that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
        inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
            float d = qb_curr->d;
            float m = qb_curr->m;

            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
                   const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

            for (int i = 0; i < 8; i+=2) {
                acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
                acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
                acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
                acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
            }

            return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
        }

        template<short NR0>
        static inline void helper_mv_reduce_and_write(
                device float * dst_f32,
                float sumf[NR0],
                const int r0,
                const int ne01,
                ushort tiisg,
                ushort sgitg,
                threadgroup char * shmem) {
            constexpr short NW = N_SIMDWIDTH;

            threadgroup float * shmem_f32[NR0];

            for (short row = 0; row < NR0; ++row) {
                shmem_f32[row] = (threadgroup float *) shmem + NW*row;

                if (sgitg == 0) {
                    shmem_f32[row][tiisg] = 0.0f;
                }

                sumf[row] = simd_sum(sumf[row]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short row = 0; row < NR0; ++row) {
                if (tiisg == 0) {
                    shmem_f32[row][sgitg] = sumf[row];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
                float tot = simd_sum(shmem_f32[row][tiisg]);

                if (tiisg == 0 && sgitg == 0) {
                    dst_f32[r0 + row] = tot;
                }
            }
        }

        constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
        constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
        constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
        constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];
        constant bool  FC_mul_mv_split [[function_constant(FC_MUL_MV + 5)]];

        template<typename block_q_type, short NR0, typename args_t>
        void mul_vec_q_n_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr short NW = N_SIMDWIDTH;
            constexpr short NQ = 16;

            const int nb = args.ne00/QK4_0;

            const int r0 = (tgpig.x*NSG + sgitg)*NR0;
            const int r1 =  tgpig.y;
            const int im =  tgpig.z;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const float        * y = (device const float        *) (src1 + offset1);

            // pointers to src0 rows
            device const block_q_type * ax[NR0];
            FOR_UNROLL (int row = 0; row < NR0; ++row) {
                const uint64_t offset0 = MIN(r0 + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;  // Origami: clamped to the last row

                ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
            }

            float sumf[NR0] = {0.f};

            const short ix = (tiisg/(NW/NQ));
            const short il = (tiisg%(NW/NQ))*8;

            const int ib0 = ix;

            float yl[16]; // src1 vector cache

            device const float * yb = y + ib0*QK4_0 + il;

            // each thread in a SIMD group deals with half a block.
            for (int ib = ib0; ib < nb; ib += NQ) {
                float sumy[2] = { 0.f, 0.f };

                FOR_UNROLL (short i = 0; i < 8; i += 2) {
                    sumy[0]  += yb[i +  0] + yb[i +  1];
                    yl[i + 0] = yb[i +  0];
                    yl[i + 1] = yb[i +  1]/256.f;

                    sumy[1]  += yb[i + 16] + yb[i + 17];
                    yl[i + 8] = yb[i + 16]/16.f;
                    yl[i + 9] = yb[i + 17]/4096.f;
                }

                FOR_UNROLL (short row = 0; row < NR0; row++) {
                    sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
                }

                yb += QK4_0 * 16;
            }

            device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;


            for (int row = 0; row < NR0; ++row) {
                const float tot = simd_sum(sumf[row]);

                if (tiisg == 0 && r0 + row < args.ne01) {
                    dst_f32[r0 + row] = tot;
                }
            }
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_q2_0_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK2_0;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

            device const float * y = (device const float *) (src1 + offset1);

            device const block_q2_0 * ax[nr0];
            for (int row = 0; row < nr0; ++row) {
                const uint64_t offset0 = MIN(first_row + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;  // Origami: clamped to the last row
                ax[row] = (device const block_q2_0 *) ((device char *) src0 + offset0);
            }

            float yl[16];
            float sumf[nr0] = {0.f};

            // group 64: 4 sub-blocks of 16 weights per Q2_0 block
            const short ix = (tiisg/4);
            const short il = (tiisg%4)*16;

            device const float * yb = y + ix*QK2_0 + il;

            for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
                float sumy = 0.f;

                FOR_UNROLL (short i = 0; i < 16; i++) {
                    yl[i] = yb[i];
                    sumy += yb[i];
                }

                FOR_UNROLL (short row = 0; row < nr0; row++) {
                    sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
                }

                yb += QK2_0 * (N_SIMDWIDTH/4);
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0; ++row) {
                const float tot = simd_sum(sumf[row]);

                if (tiisg == 0 && first_row + row < args.ne01) {
                    dst_f32[first_row + row] = tot;
                }
            }
        }

        [[host_name("kernel_mul_mv_q2_0_f32")]]
        kernel void kernel_mul_mv_q2_0_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        kernel void kernel_mul_mv_q4_0_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        kernel void kernel_mul_mv_q4_1_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
             mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        kernel void kernel_mul_mv_q5_0_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        kernel void kernel_mul_mv_q5_1_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<short NR0, typename args_t>
        void kernel_mul_mv_q8_0_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr short NW = N_SIMDWIDTH;
            constexpr short NQ = 8;

            const int nb = args.ne00/QK8_0;

            const int r0 = tgpig.x*NR0;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const float      * y = (device const float      *) (src1 + offset1);

            // pointers to src0 rows
            device const block_q8_0 * ax[NR0];
            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                const uint64_t offset0 = MIN(r0 + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;  // Origami: clamped to the last row

                ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
            }

            float sumf[NR0] = { 0.f };

            const short ix = tiisg/(NW/NQ);
            const short il = tiisg%(NW/NQ);

            const int ib0 = sgitg*NQ + ix;

            float yl[NQ];

            device const float * yb = y + ib0*QK8_0 + il*NQ;

            // each thread in a SIMD group deals with NQ quants at a time
            for (int ib = ib0; ib < nb; ib += NSG*NQ) {
                for (short i = 0; i < NQ; ++i) {
                    yl[i] = yb[i];
                }

                for (short row = 0; row < NR0; row++) {
                    device const int8_t * qs = ax[row][ib].qs + il*NQ;

                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        sumq += qs[i] * yl[i];
                    }

                    sumf[row] += sumq*ax[row][ib].d;
                }

                yb += NSG*NQ*QK8_0;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
        }

        [[host_name("kernel_mul_mv_q8_0_f32")]]
        kernel void kernel_mul_mv_q8_0_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<typename T0, typename T1, short NR0, typename args_t>
        void kernel_mul_mv_t_t_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr short NW = N_SIMDWIDTH;
            constexpr short NB = 32;
            constexpr short NF = 8;

            const int nb = args.ne00/NB;

            const int r0 = tgpig.x*NR0;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const T1 * y = (device const T1 *) (src1 + offset1);

            // pointers to src0 rows
            device const T0 * ax [NR0];
            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                const uint64_t offset0 = MIN(r0 + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;  // Origami: clamped to the last row

                ax[row] = (device const T0 *) ((device char *) src0 + offset0);
            }

            float sumf[NR0] = { 0.f };

            const short ix = tiisg/(NW/NF);
            const short il = tiisg%(NW/NF);

            const int ib0 = sgitg*NF + ix;

            T1 yl[NF];

            device const T1 * yb = y + (ib0*NB + il*NF);

            for (int ib = ib0; ib < nb; ib += NSG*NF) {
                for (short i = 0; i < NF; ++i) {
                    yl[i] = yb[i];
                }

                for (short row = 0; row < NR0; row++) {
                    device const T0 * xb = ax[row] + (ib*NB + il*NF);

                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF; ++i) {
                        sumq += xb[i] * yl[i];
                    }

                    sumf[row] += sumq;
                }

                yb += NSG*NF*NW;
            }

            for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
                for (short row = 0; row < NR0; row++) {
                    sumf[row] += ax[row][i] * y[i];
                }
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
        }

        template<typename T0, typename T1, typename args_t>
        void kernel_mul_mv_t_t_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<typename T0, typename T1>
        kernel void kernel_mul_mv_t_t(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_t_t_disp<T0, T1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

        template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
        template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;
        template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, float>;

        template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
        void kernel_mul_mv_t_t_4_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr short NW = N_SIMDWIDTH;
            constexpr short NB  = 32;
            constexpr short NF  = 16;
            constexpr short NF4 = NF/4;

            const int nb = args.ne00/NB;

            const int r0 = tgpig.x*NR0;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const T1  * y  = (device const T1  *) (src1 + offset1);
            device const T14 * y4 = (device const T14 *) (src1 + offset1);

            // pointers to src0 rows
            device const T0  * ax [NR0];
            device const T04 * ax4[NR0];
            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                const uint64_t offset0 = MIN(r0 + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;  // Origami: clamped to the last row

                ax [row] = (device const T0  *) ((device char *) src0 + offset0);
                ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
            }

            float sumf[NR0] = { 0.f };

            const short ix = tiisg/(NW/NF);
            const short il = tiisg%(NW/NF);

            const int ib0 = sgitg*NF + ix;

            T14 yl4[NF4];

            device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

            for (int ib = ib0; ib < nb; ib += NSG*NF) {
                for (short i = 0; i < NF4; ++i) {
                    yl4[i] = yb4[i];
                }

                for (short row = 0; row < NR0; row++) {
                    device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i) {
                        sumq += dot(float4(xb4[i]), float4(yl4[i]));
                    }

                    sumf[row] += sumq;
                }

                yb4 += NSG*NF*NW/4;
            }

            for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
                for (short row = 0; row < NR0; row++) {
                    sumf[row] += ax[row][i] * y[i];
                }
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
        }

        template<typename T0, typename T04, typename T1, typename T14, typename args_t>
        void kernel_mul_mv_t_t_4_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<typename T0, typename T04, typename T1, typename T14>
        kernel void kernel_mul_mv_t_t_4(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

        template [[host_name("kernel_mul_mv_f32_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
        template [[host_name("kernel_mul_mv_f16_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;
        template [[host_name("kernel_mul_mv_bf16_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, float,  float4>;

        template<typename T0, typename T1, typename args_t>
        void kernel_mul_mv_t_t_short_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig,
                ushort tiisg) {
            const int r0 = tgpig.x*32 + tiisg;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            if (r0 >= args.ne01) {
                return;
            }

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

            device const T0 * x = (device const T0 *) (src0 + offset0);

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

            const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

            device const T1 * y = (device const T1 *) (src1 + offset1);

            float res = 0.0f;

            for (int i = 0; i < args.ne00; ++i) {
                res += (float) x[i] * (float) y[i];
            }

            dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
        }

        template<typename T0, typename T1>
        kernel void kernel_mul_mv_t_t_short(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]]) {
            kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>(
                args,
                src0,
                src1,
                dst,
                tgpig,
                tiisg);
        }

        typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

        template [[host_name("kernel_mul_mv_f32_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
        template [[host_name("kernel_mul_mv_f16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;
        template [[host_name("kernel_mul_mv_bf16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, float>;

        template<int nr0, typename args_t>
        void kernel_mul_mv_q2_K_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
            device const float      * y = (device const float      *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            const short ix = tiisg/8;  // 0...3
            const short it = tiisg%8;  // 0...7
            const short iq = it/4;     // 0 or 1
            const short ir = it%4;     // 0...3
            const short is = (8*ir)/16;// 0 or 1

            device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

            for (int ib = ix; ib < nb; ib += 4) {
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 8; ++i) {
                    yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                    yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                    yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                    yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
                }

                device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
                device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
                device const half     * dh = &x[ib].d;

                for (short row = 0; row < nr0 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                    qs += args.nb01/2;
                    sc += args.nb01;
                    dh += args.nb01/2;
                }

                y4 += 4 * QK_K;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_q2_K_f32")]]
        kernel void kernel_mul_mv_q2_K_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_q3_K_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
            device const float     * yy = (device const float      *) (src1 + offset1);

            float yl[32];


            const short tid = tiisg/4;
            const short ix  = tiisg%4;
            const short ip  = tid/4;          // 0 or 1
            const short il  = 2*((tid%4)/2);  // 0 or 2
            const short ir  = tid%2;
            const short l0  = 8*ir;

            // One would think that the Metal compiler would figure out that ip and il can only have
            // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
            // with these two tales.
            //
            // Possible masks for the high bit
            const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                                   {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                                   {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                                   {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

            // Possible masks for the low 2 bits
            const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

            const ushort4 hm = mm[2*ip + il/2];

            const short shift = 2*il;

            const float v1 = il == 0 ? 4.f : 64.f;
            const float v2 = 4.f * v1;

            const uint16_t s_shift1 = 4*ip;
            const uint16_t s_shift2 = s_shift1 + il;

            const short q_offset = 32*ip + l0;
            const short y_offset = 128*ip + 32*il + l0;

            device const float * y1 = yy + ix*QK_K + y_offset;

            uint32_t scales32, aux32;
            thread uint16_t * scales16 = (thread uint16_t *)&scales32;
            thread const int8_t * scales = (thread const int8_t *)&scales32;

            float sumf1[nr0] = {0.f};
            float sumf2[nr0] = {0.f};

            for (int i = ix; i < nb; i += 4) {
                for (short l = 0; l < 8; ++l) {
                    yl[l+ 0] = y1[l+ 0];
                    yl[l+ 8] = y1[l+16];
                    yl[l+16] = y1[l+32];
                    yl[l+24] = y1[l+48];
                }

                device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
                device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
                device const uint16_t * a = (device const uint16_t *)(x[i].scales);
                device const half * dh = &x[i].d;

                for (short row = 0; row < nr0 && first_row + row < args.ne01; ++row) {  // Origami: stops at the last row
                    const float d_all = (float)dh[0];

                    scales16[0] = a[4];
                    scales16[1] = a[5];
                    aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
                    scales16[0] = a[il+0];
                    scales16[1] = a[il+1];
                    scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

                    float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
                    for (short l = 0; l < 8; l += 2) {
                        const int32_t qs = q[l/2];
                        s1 += yl[l+0] * (qs & qm[il/2][0]);
                        s2 += yl[l+1] * (qs & qm[il/2][1]);
                        s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                        s4 += yl[l+16] * (qs & qm[il/2][2]);
                        s5 += yl[l+17] * (qs & qm[il/2][3]);
                        s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
                    }
                    float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
                    float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
                    sumf1[row] += d1 * (scales[0] - 32);
                    sumf2[row] += d2 * (scales[2] - 32);

                    s1 = s2 = s3 = s4 = s5 = s6 = 0;
                    for (short l = 0; l < 8; l += 2) {
                        const int32_t qs = q[l/2+8];
                        s1 += yl[l+8] * (qs & qm[il/2][0]);
                        s2 += yl[l+9] * (qs & qm[il/2][1]);
                        s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                        s4 += yl[l+24] * (qs & qm[il/2][2]);
                        s5 += yl[l+25] * (qs & qm[il/2][3]);
                        s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
                    }
                    d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
                    d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
                    sumf1[row] += d1 * (scales[1] - 32);
                    sumf2[row] += d2 * (scales[3] - 32);

                    q  += args.nb01/2;
                    h  += args.nb01/2;
                    a  += args.nb01/2;
                    dh += args.nb01/2;
                }

                y1 += 4 * QK_K;
            }

            for (int row = 0; row < nr0; ++row) {
                const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
                sumf1[row] = simd_sum(sumf);
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            if (tiisg == 0) {
                for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                    dst_f32[first_row + row] = sumf1[row];
                }
            }
        }

        [[host_name("kernel_mul_mv_q3_K_f32")]]
        kernel void kernel_mul_mv_q3_K_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_q4_K_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr uint16_t kmask1 = 0x3f3f;
            constexpr uint16_t kmask2 = 0x0f0f;
            constexpr uint16_t kmask3 = 0xc0c0;

            const short ix = tiisg/8;  // 0...3
            const short it = tiisg%8;  // 0...7
            const short iq = it/4;     // 0 or 1
            const short ir = it%4;     // 0...3

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
            device const float      * y = (device const float      *) (src1 + offset1);

            float yl[16];
            float yh[16];

            float sumf[nr0]={0.f};

            device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

            uint16_t sc16[4];
            thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

            for (int ib = ix; ib < nb; ib += 4) {
                float4 sumy = {0.f, 0.f, 0.f, 0.f};

                for (short i = 0; i < 8; ++i) {
                    yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                    yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                    yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                    yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
                }

                device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
                device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
                device const half     * dh = &x[ib].d;

                for (short row = 0; row < nr0 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t * q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                          (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                    q1 += args.nb01/2;
                    sc += args.nb01/2;
                    dh += args.nb01/2;
                }

                y4 += 4 * QK_K;
            }

            device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_q4_K_f32")]]
        kernel void kernel_mul_mv_q4_K_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_q5_K_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
            device const float     * yy = (device const float      *) (src1 + offset1);

            float sumf[nr0]={0.f};

            float yl[16], yh[16];

            constexpr uint16_t kmask1 = 0x3f3f;
            constexpr uint16_t kmask2 = 0x0f0f;
            constexpr uint16_t kmask3 = 0xc0c0;

            const short tid = tiisg/4;
            const short ix  = tiisg%4;
            const short iq  = tid/4;
            const short ir  = tid%4;

            const short l0 = 8*ir;
            const short q_offset = 32*iq + l0;
            const short y_offset = 64*iq + l0;

            const uint8_t hm1 = 1u << (2*iq);
            const uint8_t hm2 = hm1 << 1;
            const uint8_t hm3 = hm1 << 4;
            const uint8_t hm4 = hm2 << 4;

            uint16_t sc16[4];
            thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

            device const float * y1 = yy + ix*QK_K + y_offset;

            for (int i = ix; i < nb; i += 4) {
                device const uint8_t * q1 = x[i].qs + q_offset;
                device const uint8_t * qh = x[i].qh + l0;
                device const half * dh = &x[i].d;
                device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

                device const float * y2 = y1 + 128;
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                for (short l = 0; l < 8; ++l) {
                    yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
                    yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                    yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
                    yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
                }

                for (short row = 0; row < nr0 && first_row + row < args.ne01; ++row) {  // Origami: stops at the last row
                    device const uint8_t * q2 = q1 + 64;

                    sc16[0] = a[0] & kmask1;
                    sc16[1] = a[2] & kmask1;
                    sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                    sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                    float4 acc1 = {0.f};
                    float4 acc2 = {0.f};
                    FOR_UNROLL (short l = 0; l < 8; ++l) {
                        uint8_t h = qh[l];
                        acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                        acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                        acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                        acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                        acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                        acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                        acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                        acc2[3] += h & hm4 ? yh[l+8] : 0.f;
                    }

                    sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                          sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                          sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                          sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                    q1 += args.nb01;
                    qh += args.nb01;
                    dh += args.nb01/2;
                    a  += args.nb01/2;
                }

                y1 += 4 * QK_K;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                const float tot = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = tot;
                }
            }
        }

        [[host_name("kernel_mul_mv_q5_K_f32")]]
        kernel void kernel_mul_mv_q5_K_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_q6_K_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            constexpr uint8_t kmask1 = 0x03;
            constexpr uint8_t kmask2 = 0x0C;
            constexpr uint8_t kmask3 = 0x30;
            constexpr uint8_t kmask4 = 0xC0;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
            device const float     * yy = (device const float      *) (src1 + offset1);

            float sumf[nr0] = { 0.f };

            float yl[16];

            const short tid = tiisg/2;
            const short ix  = tiisg%2;
            const short ip  = tid/8;         // 0 or 1
            const short il  = tid%8;
            const short l0  = 4*il;
            const short is  = 8*ip + l0/16;

            const short y_offset   = 128*ip + l0;
            const short q_offset_l =  64*ip + l0;
            const short q_offset_h =  32*ip + l0;

            for (int i = ix; i < nb; i += 2) {
                device const uint8_t * q1 = x[i].ql + q_offset_l;
                device const uint8_t * q2 = q1 + 32;
                device const uint8_t * qh = x[i].qh + q_offset_h;
                device const int8_t  * sc = x[i].scales + is;
                device const half    * dh = &x[i].d;

                device const float * y = yy + i * QK_K + y_offset;

                for (short l = 0; l < 4; ++l) {
                    yl[4*l + 0] = y[l +  0];
                    yl[4*l + 1] = y[l + 32];
                    yl[4*l + 2] = y[l + 64];
                    yl[4*l + 3] = y[l + 96];
                }

                for (short row = 0; row < nr0 && first_row + row < args.ne01; ++row) {  // Origami: stops at the last row
                    float4 sums = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short l = 0; l < 4; ++l) {
                        sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                        sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                        sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                        sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                    }

                    sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

                    q1 += args.nb01;
                    q2 += args.nb01;
                    qh += args.nb01;
                    sc += args.nb01;
                    dh += args.nb01/2;
                }
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_q6_K_f32")]]
        kernel void kernel_mul_mv_q6_K_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        // ======================= "True" 2-bit

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq2_xxs_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
            device const float         * y = (device const float         *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
            threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
            {
                int nval = 4;
                int pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
                nval = 2;
                pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq2_xxs * xr = x + ibl;
                device const uint16_t * q2 = xr->qs + 4 * ib;
                device const half * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    const float db = dh[0];
                    device const uint8_t * aux8 = (device const uint8_t *)q2;
                    const uint32_t aux32 = q2[2] | (q2[3] << 16);
                    const float d = db * (0.5f + (aux32 >> 28));

                    float sum = 0;
                    for (short l = 0; l < 4; ++l) {
                        const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                        const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                        for (short j = 0; j < 8; ++j) {
                            sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                        }
                    }
                    sumf[row] += d * sum;

                    dh += args.nb01/2;
                    q2 += args.nb01/2;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all * 0.25f;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq2_xxs_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq2_xxs_f32")]]
        kernel void kernel_mul_mv_iq2_xxs_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_iq2_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq2_xs_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
            device const float        * y = (device const float        *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
            threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
            {
                int nval = 8;
                int pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
                nval = 2;
                pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq2_xs * xr = x + ibl;
                device const uint16_t * q2 = xr->qs + 4 * ib;
                device const uint8_t  * sc = xr->scales + ib;
                device const half * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    const float db = dh[0];
                    const uint8_t ls1 = sc[0] & 0xf;
                    const uint8_t ls2 = sc[0] >>  4;
                    const float d1 = db * (0.5f + ls1);
                    const float d2 = db * (0.5f + ls2);

                    float sum1 = 0, sum2 = 0;
                    for (short l = 0; l < 2; ++l) {
                        const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                        const uint8_t signs = ssigns[(q2[l] >> 9)];
                        for (short j = 0; j < 8; ++j) {
                            sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                        }
                    }
                    for (short l = 2; l < 4; ++l) {
                        const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                        const uint8_t signs = ssigns[(q2[l] >> 9)];
                        for (short j = 0; j < 8; ++j) {
                            sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                        }
                    }
                    sumf[row] += d1 * sum1 + d2 * sum2;

                    dh += args.nb01/2;
                    q2 += args.nb01/2;
                    sc += args.nb01;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all * 0.25f;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq2_xs_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq2_xs_f32")]]
        kernel void kernel_mul_mv_iq2_xs_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq2_xs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        // FC_mul_mv_split: for nb32 < 32 (nb32 divides 32), 32/nb32 threads share each chunk and each takes a slice of the rows
        template<int nr0, typename args_t>
        void kernel_mul_mv_iq3_xxs_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
            device const float         * y = (device const float         *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
            threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
            {
                int nval = 4;
                int pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
                nval = 2;
                pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq3_xxs * xr = x + ibl;
                device const uint8_t  * q3 = xr->qs + 8 * ib;
                device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
                device const half * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    const float db = dh[0];
                    const uint32_t aux32 = gas[0] | (gas[1] << 16);
                    const float d = db * (0.5f + (aux32 >> 28));

                    float2 sum = {0};
                    for (short l = 0; l < 4; ++l) {
                        const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                        const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                        const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                        for (short j = 0; j < 4; ++j) {
                            sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                            sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                        }
                    }
                    sumf[row] += d * (sum[0] + sum[1]);

                    dh  += args.nb01/2;
                    q3  += args.nb01;
                    gas += args.nb01/2;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all * 0.5f;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq3_xxs_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq3_xxs_f32")]]
        kernel void kernel_mul_mv_iq3_xxs_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq3_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq3_s_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
            device const float       * y = (device const float       *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
            {
                int nval = 8;
                int pos  = (32*sgitg + tiisg)*nval;
                for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq3_s * xr = x + ibl;
                device const uint8_t * qs = xr->qs + 8 * ib;
                device const uint8_t * qh = xr->qh + ib;
                device const uint8_t * sc = xr->scales + (ib/2);
                device const uint8_t * signs = xr->signs + 4 * ib;
                device const half * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    const float db = dh[0];
                    const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

                    float2 sum = {0};
                    for (short l = 0; l < 4; ++l) {
                        const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                        const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                        const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                        const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                        for (short j = 0; j < 4; ++j) {
                            sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                            sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                        }
                    }
                    sumf[row] += d * (sum[0] + sum[1]);

                    dh    += args.nb01/2;
                    qs    += args.nb01;
                    qh    += args.nb01;
                    sc    += args.nb01;
                    signs += args.nb01;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq3_s_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq3_s_f32")]]
        kernel void kernel_mul_mv_iq3_s_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq3_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq2_s_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
            device const float       * y = (device const float       *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};


            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq2_s * xr = x + ibl;
                device const uint8_t * qs = xr->qs + 4 * ib;
                device const uint8_t * qh = xr->qh + ib;
                device const uint8_t * sc = xr->scales + ib;
                device const uint8_t * signs = qs + QK_K/8;
                device const half * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    const float db = dh[0];
                    const float d1 = db * (0.5f + (sc[0] & 0xf));
                    const float d2 = db * (0.5f + (sc[0] >>  4));

                    float2 sum = {0};
                    for (short l = 0; l < 2; ++l) {
                        constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                        constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                        for (short j = 0; j < 8; ++j) {
                            sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                            sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                        }
                    }
                    sumf[row] += d1 * sum[0] + d2 * sum[1];

                    dh    += args.nb01/2;
                    qs    += args.nb01;
                    qh    += args.nb01;
                    sc    += args.nb01;
                    signs += args.nb01;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all * 0.25f;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq2_s_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq2_s_f32")]]
        kernel void kernel_mul_mv_iq2_s_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq2_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq1_s_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
            device const float       * y = (device const float       *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            device const float * y4 = y + 32 * ix;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                float sumy = 0;
                for (short i = 0; i < 32; ++i) {
                    yl[i] = y4[i];
                    sumy += yl[i];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq1_s * xr = x + ibl;
                device const uint8_t  * qs = xr->qs + 4 * ib;
                device const uint16_t * qh = xr->qh + ib;
                device const half     * dh = &xr->d;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
                    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
                    constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
                    constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

                    float sum = 0;
                    for (short j = 0; j < 4; ++j) {
                        sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                             + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                             + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                             + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
                    }
                    sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

                    dh += args.nb01/2;
                    qs += args.nb01;
                    qh += args.nb01/2;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq1_s_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq1_s_f32")]]
        kernel void kernel_mul_mv_iq1_s_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq1_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int nr0, typename args_t>
        void kernel_mul_mv_iq1_m_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            const int nb = args.ne00/QK_K;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * nr0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const int nb32 = nb * (QK_K / 32);

            const short ntx  = FC_mul_mv_split ? nb32 : 32;
            const short nrep = 32 / ntx;

            const short ix   = tiisg % ntx;
            const short irep = tiisg / ntx;

            const short row0 = (nr0 * irep      ) / nrep;
            const short row1 = (nr0 * (irep + 1)) / nrep;

            const uint64_t offset0 = (first_row + row0)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =                r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
            device const float       * y = (device const float       *) (src1 + offset1);

            float yl[32];
            float sumf[nr0]={0.f};

            device const float * y4 = y + 32 * ix;

            iq1m_scale_t scale;

            for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
                float4 sumy = {0.f};
                for (short i = 0; i < 8; ++i) {
                    yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                    yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
                    yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
                    yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
                }

                const int ibl = ib32 / (QK_K / 32);
                const int ib  = ib32 % (QK_K / 32);

                device const block_iq1_m * xr = x + ibl;
                device const uint8_t  * qs = xr->qs + 4 * ib;
                device const uint8_t  * qh = xr->qh + 2 * ib;
                device const uint16_t * sc = (device const uint16_t *)xr->scales;

                for (short row = row0; row < row1 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

                    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
                    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
                    constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
                    constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

                    float2 sum = {0.f};
                    for (short j = 0; j < 4; ++j) {
                        sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                                + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                        sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                                + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
                    }
                    const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
                    const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

                    sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                                     (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

                    sc += args.nb01/2;
                    qs += args.nb01;
                    qh += args.nb01;
                }

                y4 += 32 * ntx;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        template<typename args_t>
        void kernel_mul_mv_iq1_m_f32_disp(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            if (FC_mul_mv_split) {
                kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            } else {
                kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
            }
        }

        [[host_name("kernel_mul_mv_iq1_m_f32")]]
        kernel void kernel_mul_mv_iq1_m_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq1_m_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
        }

        template<int NR0, typename args_t>
        void kernel_mul_mv_iq4_nl_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            threadgroup float * shmem_f32 = (threadgroup float *) shmem;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * NR0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
            device const float        * y = (device const float        *) (src1 + offset1);

            const int nb   = args.ne00/QK4_NL;
            const int ns01 = args.nb01/args.nb00;

            const short ix = tiisg/2;  // 0...15
            const short it = tiisg%2;  // 0 or 1

            shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 yl[4];
            float sumf[NR0]={0.f};

            device const float * yb = y + ix*QK4_NL + it*8;

            uint32_t aux32[2];
            thread const uint8_t * q8 = (thread const uint8_t *)aux32;

            float4 qf1, qf2;

            for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
                device const float4 * y4 = (device const float4 *)yb;
                yl[0] = y4[0];
                yl[1] = y4[4];
                yl[2] = y4[1];
                yl[3] = y4[5];

                for (short row = 0; row < NR0 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    device const block_iq4_nl & xb = x[row*ns01 + ib];
                    device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

                    float4 acc1 = {0.f}, acc2 = {0.f};

                    aux32[0] = q4[0] | (q4[1] << 16);
                    aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                    aux32[0] &= 0x0f0f0f0f;
                    qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                    qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                    acc1 += yl[0] * qf1;
                    acc2 += yl[1] * qf2;

                    aux32[0] = q4[2] | (q4[3] << 16);
                    aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                    aux32[0] &= 0x0f0f0f0f;
                    qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                    qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                    acc1 += yl[2] * qf1;
                    acc2 += yl[3] * qf2;

                    acc1 += acc2;

                    sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
                }

                yb += 16 * QK4_NL;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_iq4_nl_f32")]]
        kernel void kernel_mul_mv_iq4_nl_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int NR0, typename args_t>
        void kernel_mul_mv_iq4_xs_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            threadgroup float * shmem_f32 = (threadgroup float *) shmem;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;
            const int first_row = (r0 * NSG + sgitg) * NR0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
            device const float        * y = (device const float        *) (src1 + offset1);

            const int nb   = args.ne00/QK_K;
            const int ns01 = args.nb01/args.nb00;

            const short ix = tiisg/16;  // 0 or 1
            const short it = tiisg%16;  // 0...15
            const short ib = it/2;
            const short il = it%2;

            shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 yl[4];
            float sumf[NR0]={0.f};

            device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

            uint32_t aux32[2];
            thread const uint8_t * q8 = (thread const uint8_t *)aux32;

            float4 qf1, qf2;

            for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
                device const float4 * y4 = (device const float4 *)yb;
                yl[0] = y4[0];
                yl[1] = y4[4];
                yl[2] = y4[1];
                yl[3] = y4[5];

                for (short row = 0; row < NR0 && first_row + row < args.ne01; ++row) {  // Origami: stops at the last row
                    device const block_iq4_xs & xb = x[row*ns01 + ibl];
                    device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

                    float4 acc1 = {0.f}, acc2 = {0.f};

                    aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
                    aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
                    qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                    qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                    acc1 += yl[0] * qf1;
                    acc2 += yl[1] * qf2;

                    aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
                    aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
                    qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                    qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                    acc1 += yl[2] * qf1;
                    acc2 += yl[3] * qf2;

                    acc1 += acc2;

                    const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
                    sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
                }

                yb += 2 * QK_K;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_iq4_xs_f32")]]
        kernel void kernel_mul_mv_iq4_xs_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        template<int NR0, typename args_t>
        void kernel_mul_mv_mxfp4_f32_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiisg,
                ushort sgitg) {
            const short NSG = FC_mul_mv_nsg;

            threadgroup float * shmem_f32 = (threadgroup float *) shmem;

            const int r0 = tgpig.x;
            const int r1 = tgpig.y;
            const int im = tgpig.z;

            const int first_row = (r0 * NSG + sgitg) * NR0;

            const uint i12 = im%FC_mul_mv_ne12;
            const uint i13 = im/FC_mul_mv_ne12;

            const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
            const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

            device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
            device const float       * y = (device const float       *) (src1 + offset1);

            const int nb   = args.ne00/QK_MXFP4;
            const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

            const short ix = tiisg/2;  // 0...15
            const short it = tiisg%2;  // 0 or 1

            shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 yl[4];
            float sumf[NR0]={0.f};

            device const float * yb = y + ix*QK_MXFP4 + it*8;

            // `ib < ns01` repeats `ib < nb`; upstream keeps it because the kernel runs faster with it.
            for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
                device const float4 * y4 = (device const float4 *) yb;

                yl[0] = y4[0];
                yl[1] = y4[4];
                yl[2] = y4[1];
                yl[3] = y4[5];

                FOR_UNROLL (short row = 0; row < NR0 && first_row + row < args.ne01; row++) {  // Origami: stops at the last row
                    device const block_mxfp4 & xb = x[row*ns01 + ib];
                    device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

                    float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
                    float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
                    float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
                    float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

                    acc1 = (acc1 + acc3) + (acc2 + acc4);

                    sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
                }

                yb += 16 * QK_MXFP4;
            }

            device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

            for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
                float sum_all = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    dst_f32[first_row + row] = sum_all;
                }
            }
        }

        [[host_name("kernel_mul_mv_mxfp4_f32")]]
        kernel void kernel_mul_mv_mxfp4_f32(
                constant ggml_metal_kargs_mul_mv & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiisg[[thread_index_in_simdgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {

            kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
        }

        // ---- mul_mm.metal

        constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
        constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
        constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
        constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
        constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
        constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];

        // Origami: the body of kernel_mul_mm, shared with kernel_mul_mm_batch (args_t: where the arguments live).
        template<
            typename S0, typename S0_4x4, typename S0_8x8,
            typename S1, typename S1_2x4, typename S1_8x8,
            typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
            typename T0, typename T0_4x4, typename T1, typename T1_2x4, typename args_t>
        void kernel_mul_mm_impl(
                args_t args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem,
                uint3  tgpig,
                ushort tiitg,
                ushort sgitg) {

            threadgroup S0 * sa = (threadgroup S0 *)(shmem);
            threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

            constexpr int NR0 = 64;
            constexpr int NR1 = 32;

            constexpr int NK  = 32;
            constexpr int NL0 = NK/16;
            constexpr int NL1 = NK/8;

            const int im = tgpig.z;
            const int r0 = tgpig.y*NR0;
            const int r1 = tgpig.x*NR1;

            // if this block is of 64x32 shape or smaller
            const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
            const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

            // a thread shouldn't load data outside of the matrix
            const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
            const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

            const short il0 = (tiitg % NL0);

            short il = il0;

            const int i12 = im % FC_mul_mm_ne12;
            const int i13 = im / FC_mul_mm_ne12;

            const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
            const short    offset1 = il0/nl;

            device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

            const short iy = 8*(tiitg % NL1);

            device const T1 * y = (device const T1 *)(src1
                + args.nb13*i13
                + args.nb12*i12
                + args.nb11*(r1 + lr1)
                + args.nb10*iy);

            S0_8x8 ma[4];
            S1_8x8 mb[2];

            simdgroup_float8x8 mc[8];

            for (short i = 0; i < 8; i++){
                mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
            }

            for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
                // load data and store to threadgroup memory
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // no need for dequantization
                    for (short i = 0; i < 16; i++) {
                        const short sx = 2*il0 + i/8;
                        const short sy = (tiitg/NL0)/8;

                        const short lx = (tiitg/NL0)%8;
                        const short ly = i%8;

                        const short ib = 8*sx + sy;

                        *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
                    }
                } else {
                    S0_4x4 temp_a;
                    dequantize_func(x, il, temp_a);

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        const short sx = 2*il0 + i/8;
                        const short sy = (tiitg/NL0)/8;

                        const short lx = (tiitg/NL0)%8;
                        const short ly = i%8;

                        const short ib = 8*sx + sy;


                        *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
                    }
                }

                if (FC_mul_mm_bc_inp) {
                    for (short i = 0; i < 8; ++i) {
                        const short sx = (tiitg%NL1);
                        const short sy = (tiitg/NL1)/8;

                        const short lx = i;
                        const short ly = (tiitg/NL1)%8;

                        const short ib = 4*sx + sy;

                        *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                    }
                } else {
                    const short sx = (tiitg%NL1);
                    const short sy = (tiitg/NL1)/8;


                    const short ly = (tiitg/NL1)%8;

                    const short ib = 4*sx + sy;

                    *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
                }

                il = (il + 2 < nl) ? il + 2 : il % 2;
                x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

                y += NK;

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // load matrices from threadgroup memory and conduct outer products
                threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
                threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

                FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                    simdgroup_barrier(mem_flags::mem_none);

                    FOR_UNROLL (short i = 0; i < 4; i++) {
                        simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                    }

                    simdgroup_barrier(mem_flags::mem_none);

                    FOR_UNROLL (short i = 0; i < 2; i++) {
                        simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                    }

                    simdgroup_barrier(mem_flags::mem_none);

                    FOR_UNROLL (short i = 0; i < 8; i++){
                        simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                    }

                    lsma += 8*64;
                    lsmb += 4*64;
                }
            }

            if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
                // if no bounds checks on the output are needed, we can directly write to device memory
                device float * C = (device float *) dst +
                    (r0 + 32*(sgitg &  1)) + \
                    (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

                for (short i = 0; i < 8; i++) {
                    simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
                }
            } else {
                // block is smaller than 64x32, we should avoid writing data outside of the matrix
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

                for (short i = 0; i < 8; i++) {
                    simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (sgitg == 0) {
                    for (int j = tiitg; j < nr1; j += NR1) {
                        device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;

                        threadgroup float  * C  = temp_str + (j*NR0);

                        // Origami: scalar stores, so rows of any output count stay aligned (ggml stores float4s)
                        for (int i = 0; i < nr0; i++) {
                            *(D + i) = *(C + i);
                        }
                    }
                }
            }
        }

        template<
            typename S0, typename S0_4x4, typename S0_8x8,
            typename S1, typename S1_2x4, typename S1_8x8,
            typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
            typename T0, typename T0_4x4, typename T1, typename T1_2x4>
        kernel void kernel_mul_mm(
                constant ggml_metal_kargs_mul_mm & args,
                device const char * src0,
                device const char * src1,
                device       char * dst,
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiitg[[thread_index_in_threadgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mm_impl<S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, block_q, nl, dequantize_func, T0, T0_4x4, T1, T1_2x4, constant ggml_metal_kargs_mul_mm &>(args, src0, src1, dst, shmem, tgpig, tiitg, sgitg);
        }

        // Origami: one dispatch for several matrices of one type and shape, each over its own rows (a prompt's routed
        // experts). Threadgroup layer z runs exactly what one kernel_mul_mm dispatch of table.rows[z] rows runs for matrix
        // z: its weights at the GPU address table.weights[z], its input rows at src1 + table.input[z] and its output rows at
        // dst + table.output[z]; the grid is as wide as the most rows, and a threadgroup past z's rows does nothing.
        #define ORIGAMI_MM_TABLE 128
        struct origami_mm_table {
            device const char * weights[ORIGAMI_MM_TABLE];
            uint64_t input[ORIGAMI_MM_TABLE];
            uint64_t output[ORIGAMI_MM_TABLE];
            int rows[ORIGAMI_MM_TABLE];
        };
        template<
            typename S0, typename S0_4x4, typename S0_8x8,
            typename S1, typename S1_2x4, typename S1_8x8,
            typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
            typename T0, typename T0_4x4, typename T1, typename T1_2x4>
        kernel void kernel_mul_mm_batch(
                constant ggml_metal_kargs_mul_mm & args [[buffer(0)]],
                device const char * src1 [[buffer(2)]],
                device       char * dst [[buffer(3)]],
                constant origami_mm_table & table [[buffer(4)]],
                threadgroup  char * shmem [[threadgroup(0)]],
                uint3  tgpig[[threadgroup_position_in_grid]],
                ushort tiitg[[thread_index_in_threadgroup]],
                ushort sgitg[[simdgroup_index_in_threadgroup]]) {
            const int rows = table.rows[tgpig.z];
            if ((int)tgpig.x * 32 >= rows) return;
            ggml_metal_kargs_mul_mm own = args;
            own.ne1 = rows;
            kernel_mul_mm_impl<S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, block_q, nl, dequantize_func, T0, T0_4x4, T1, T1_2x4, thread const ggml_metal_kargs_mul_mm &>(
                own, table.weights[tgpig.z], src1 + table.input[tgpig.z], dst + table.output[tgpig.z], shmem, uint3(tgpig.x, tgpig.y, 0),
                tiitg, sgitg);
        }

        typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;
        typedef decltype(kernel_mul_mm_batch<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_batch_t;

        template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
        template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
        template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_f32_f32")]]     kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_f16_f32")]]     kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_bf16_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q2_0_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q4_0_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q4_1_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q5_0_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q5_1_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q8_0_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_mxfp4_f32")]]   kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q2_K_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q3_K_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q4_K_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q5_K_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_q6_K_f32")]]    kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq2_xxs_f32")]] kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq2_xs_f32")]]  kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq3_xxs_f32")]] kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq3_s_f32")]]   kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq2_s_f32")]]   kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq1_s_f32")]]   kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq1_m_f32")]]   kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq4_nl_f32")]]  kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
        template [[host_name("kernel_mul_mm_batch_iq4_xs_f32")]]  kernel mul_mm_batch_t kernel_mul_mm_batch<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;

        // Origami: one dispatch for several matrices of one type and shape (a layer's routed experts, a stacked tensor's
        // heads). Threadgroup layer z runs exactly what one kernel_mul_mv_<type>_f32 dispatch runs for matrix z: its
        // weights at the GPU address table.weights[z] (the matrices may sit in different buffers), its input at
        // src1 + z * inputStride and its output at dst + z * outputStride.
        #define ORIGAMI_MV_TABLE 128
        struct origami_mv_table {
            uint64_t inputStride;
            uint64_t outputStride;
            device const char * weights[ORIGAMI_MV_TABLE];
        };
        #define ORIGAMI_MV_BATCH(NAME, ...) \
        [[host_name("kernel_mul_mv_batch_" #NAME "_f32")]] \
        kernel void kernel_mul_mv_batch_##NAME##_f32( \
                constant ggml_metal_kargs_mul_mv & args [[buffer(0)]], \
                device const char * src1_base [[buffer(2)]], \
                device       char * dst_base [[buffer(3)]], \
                constant origami_mv_table & table [[buffer(4)]], \
                threadgroup  char * shmem [[threadgroup(0)]], \
                uint3  position[[threadgroup_position_in_grid]], \
                ushort tiisg[[thread_index_in_simdgroup]], \
                ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
            const uint3 tgpig = uint3(position.x, position.y, 0); \
            device const char * src0 = table.weights[position.z]; \
            device const char * src1 = src1_base + position.z * table.inputStride; \
            device       char * dst  = dst_base  + position.z * table.outputStride; \
            __VA_ARGS__; \
        }
        ORIGAMI_MV_BATCH(q2_0, kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q4_0, mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q4_1, mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q5_0, mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q5_1, mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q8_0, kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q2_K, kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q3_K, kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q4_K, kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q5_K, kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(q6_K, kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq2_xxs, kernel_mul_mv_iq2_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq2_xs, kernel_mul_mv_iq2_xs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq3_xxs, kernel_mul_mv_iq3_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq3_s, kernel_mul_mv_iq3_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq2_s, kernel_mul_mv_iq2_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq1_s, kernel_mul_mv_iq1_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq1_m, kernel_mul_mv_iq1_m_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq4_nl, kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(iq4_xs, kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_BATCH(mxfp4, kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        // Origami: as ORIGAMI_MV_BATCH, with each matrix's input and output at its own offset (a prompt's expert runs).
        struct origami_mv_rows_table {
            device const char * weights[ORIGAMI_MV_TABLE];
            uint64_t input[ORIGAMI_MV_TABLE];
            uint64_t output[ORIGAMI_MV_TABLE];
        };
        #define ORIGAMI_MV_ROWS(NAME, ...) \
        [[host_name("kernel_mul_mv_rows_" #NAME "_f32")]] \
        kernel void kernel_mul_mv_rows_##NAME##_f32( \
                constant ggml_metal_kargs_mul_mv & args [[buffer(0)]], \
                device const char * src1_base [[buffer(2)]], \
                device       char * dst_base [[buffer(3)]], \
                constant origami_mv_rows_table & table [[buffer(4)]], \
                threadgroup  char * shmem [[threadgroup(0)]], \
                uint3  position[[threadgroup_position_in_grid]], \
                ushort tiisg[[thread_index_in_simdgroup]], \
                ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
            const uint3 tgpig = uint3(position.x, position.y, 0); \
            device const char * src0 = table.weights[position.z]; \
            device const char * src1 = src1_base + table.input[position.z]; \
            device       char * dst  = dst_base  + table.output[position.z]; \
            __VA_ARGS__; \
        }
        ORIGAMI_MV_ROWS(q2_0, kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q4_0, mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q4_1, mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q5_0, mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q5_1, mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q8_0, kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q2_K, kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q3_K, kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q4_K, kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q5_K, kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(q6_K, kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq2_xxs, kernel_mul_mv_iq2_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq2_xs, kernel_mul_mv_iq2_xs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq3_xxs, kernel_mul_mv_iq3_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq3_s, kernel_mul_mv_iq3_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq2_s, kernel_mul_mv_iq2_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq1_s, kernel_mul_mv_iq1_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq1_m, kernel_mul_mv_iq1_m_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq4_nl, kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(iq4_xs, kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_ROWS(mxfp4, kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        // Origami: queued decode's GGUF experts. As ORIGAMI_MV_BATCH for up to ten matrices that are not known when the
        // dispatch is encoded, on an indirect grid of (any width, record count): threadgroup row y runs the matrix at
        // records[y] + part (GPU addresses written before it runs), its input at src1 + ranks[y] * inputStride and its output
        // at dst + ranks[y] * outputStride, for every ggml threadgroup x of the matrix (`groups`) in steps of the grid width.
        struct origami_mv_gather {
            uint64_t part;
            uint64_t inputStride;
            uint64_t outputStride;
            uint64_t groups;
        };
        struct origami_records10 { device const char * values[10]; };
        #define ORIGAMI_MV_GATHER(NAME, ...) \
        [[host_name("kernel_mul_mv_gather_" #NAME "_f32")]] \
        kernel void kernel_mul_mv_gather_##NAME##_f32( \
                constant ggml_metal_kargs_mul_mv & args [[buffer(0)]], \
                device const char * src1_base [[buffer(2)]], \
                device       char * dst_base [[buffer(3)]], \
                constant origami_mv_gather & table [[buffer(4)]], \
                constant origami_records10 & records [[buffer(5)]], \
                constant uint * ranks [[buffer(6)]], \
                threadgroup  char * shmem [[threadgroup(0)]], \
                uint3  position[[threadgroup_position_in_grid]], \
                uint3  grid[[threadgroups_per_grid]], \
                ushort tiisg[[thread_index_in_simdgroup]], \
                ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
            device const char * src0 = records.values[position.y] + table.part; \
            device const char * src1 = src1_base + ranks[position.y] * table.inputStride; \
            device       char * dst  = dst_base  + ranks[position.y] * table.outputStride; \
            for (uint x = position.x; x < table.groups; x += grid.x) { \
                const uint3 tgpig = uint3(x, 0, 0); \
                __VA_ARGS__; \
            } \
        }
        ORIGAMI_MV_GATHER(q2_0, kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q4_0, mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q4_1, mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q5_0, mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q5_1, mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q8_0, kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q2_K, kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q3_K, kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q4_K, kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q5_K, kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(q6_K, kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq2_xxs, kernel_mul_mv_iq2_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq2_xs, kernel_mul_mv_iq2_xs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq3_xxs, kernel_mul_mv_iq3_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq3_s, kernel_mul_mv_iq3_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq2_s, kernel_mul_mv_iq2_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq1_s, kernel_mul_mv_iq1_s_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq1_m, kernel_mul_mv_iq1_m_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq4_nl, kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(iq4_xs, kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        ORIGAMI_MV_GATHER(mxfp4, kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg))
        """#
}
