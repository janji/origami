import Foundation
import Metal

/// Decode QMV for the dense affine-4 projections (`projection_tile8`): one SIMD group computes eight output rows,
/// sharing the activation expansion and the staged BF16 bias sums. Layout: packed nibbles `[outputs][width/8]` UInt32,
/// BF16 scales and biases `[outputs][width/group]`; FP32 output holding a BF16-rounded value. Each lane takes 16
/// consecutive inputs per 512-wide step, the pinned `qmv_fast` lane schedule (four-product tree, BF16 bias-sum
/// boundaries, one SIMD sum), so one row is BITWISE equal to `qmv_pinned_reference`. Two matrices sharing one input
/// can run in one dispatch. Pipelines are specialized (width, first rows, input rows) and cached per instance.
/// `inputRows` 2 (the MTP verify step): each packed weight load serves both activation rows with the one-row
/// arithmetic, so at widths 2560/6144/10240 a row equals the one-row kernel bit for bit. Width 320 (HC up) takes this
/// kernel only with two rows; its one-row path is the 8-per-lane `qmv` schedule, so the pair step is NUMERIC there.
package final class DecodeProjectionKernel {
    /// One matrix: packed words, scales and biases (bound at offset 0, shared storage) and its output-row count.
    package struct Weights {
        package let words, scales, biases: MTLBuffer
        package let rows: Int
        package init(words: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, rows: Int) {
            self.words = words
            self.scales = scales
            self.biases = biases
            self.rows = rows
        }
    }
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    static func supports(width: Int, outputs: Int) -> Bool {
        [2560, 6144, 10240].contains(width) && outputs > 0 && outputs <= 248320 && outputs % 8 == 0
    }
    /// Two rows also take the HC up projection (width 320, twenty active lanes); one row keeps the pinned schedule.
    static func supports(width: Int, outputs: Int, inputRows: Int) -> Bool {
        supports(width: width, outputs: outputs) || (inputRows == 2 && width == 320 && outputs % 8 == 0 && outputs <= 248320)
    }
    init(device: MTLDevice) throws {
        self.device = device
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        library = try device.makeLibrary(
            source: ModelLayout.metalDefines + ReferenceKernels.quantizedBF16Preamble + Self.source, options: options)
    }

    func encode(
        _ command: MTLCommandBuffer, input: MTLBuffer, width: Int,
        first: Weights, firstOutput: MTLBuffer,
        second: Weights? = nil, secondOutput: MTLBuffer? = nil, inputRows: Int = 1
    ) throws {
        let weights = [first] + (second.map { [$0] } ?? [])
        let outputs = [firstOutput] + (secondOutput.map { [$0] } ?? [])
        let reads = [input] + weights.flatMap { [$0.words, $0.scales, $0.biases] }
        guard Self.supports(width: width, outputs: first.rows, inputRows: inputRows),
            (second == nil) == (secondOutput == nil), weights.count == outputs.count,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            [1, 2].contains(inputRows), inputRows == 1 || second == nil, input.length >= width * 2 * inputRows,
            (reads + outputs).allSatisfy({ $0.device.registryID == device.registryID && $0.storageMode == .shared }),
            Set(outputs.map { ObjectIdentifier($0) }).count == outputs.count,
            !outputs.contains(where: { output in reads.contains(where: { $0 === output }) }),
            zip(weights, outputs).allSatisfy({ w, output in
                Self.supports(width: width, outputs: w.rows, inputRows: inputRows) && w.words.length >= width * w.rows / 2
                    && w.scales.length >= width * w.rows / ModelLayout.group * 2
                    && w.biases.length >= width * w.rows / ModelLayout.group * 2 && output.length >= w.rows * 4 * inputRows
            })
        else { throw EngineError("Invalid tile8 decode projection resources") }
        let key = "\(width):\(first.rows):\(inputRows)"
        let pipeline: MTLComputePipelineState
        if let cached = pipelines[key] {
            pipeline = cached
        } else {
            let constants = MTLFunctionConstantValues()
            var w = UInt32(width)
            var n = UInt32(first.rows)
            var x = UInt32(inputRows)
            constants.setConstantValue(&w, type: .uint, index: 0)
            constants.setConstantValue(&n, type: .uint, index: 1)
            constants.setConstantValue(&x, type: .uint, index: 2)
            pipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "projection_tile8", constantValues: constants))
            guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 256,
                pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength
            else {
                throw EngineError("Tile8 decode projection resource limit differs")
            }
            pipelines[key] = pipeline
        }
        guard let e = command.makeComputeCommandEncoder() else { throw EngineError("Tile8 projection encoder unavailable") }
        let other = second ?? first
        e.setComputePipelineState(pipeline)
        for (i, buffer) in [
            input, first.words, first.scales, first.biases, other.words, other.scales, other.biases,
            firstOutput, secondOutput ?? firstOutput,
        ].enumerated() {
            e.setBuffer(buffer, offset: 0, index: i)
        }
        e.dispatchThreads(
            MTLSize(width: (first.rows + (second?.rows ?? 0)) / 8 * 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding()
    }

    static let source = #"""
        // Eight output rows share the original activation expansion and staged BF16
        // bias sum. The per-row lane schedule, four-product tree and final SIMD sum
        // are unchanged. Safe math forbids reassociation/implicit contraction.
        constant uint WIDTH [[function_constant(0)]];
        constant uint FIRST_ROWS [[function_constant(1)]];
        constant uint XROWS [[function_constant(2)]];
        kernel void projection_tile8(
            device const ushort *x [[buffer(0)]],
            device const uint *w0 [[buffer(1)]], device const ushort *s0 [[buffer(2)]], device const ushort *b0 [[buffer(3)]],
            device const uint *w1 [[buffer(4)]], device const ushort *s1 [[buffer(5)]], device const ushort *b1 [[buffer(6)]],
            device float *out0 [[buffer(7)]], device float *out1 [[buffer(8)]],
            uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint outputRow=(tid/32)*8;
            bool first=outputRow<FIRST_ROWS;
            uint row=first ? outputRow:outputRow-FIRST_ROWS;
            device float *out=first?out0:out1;
            device const uint *words=first?w0:w1;
            device const ushort *scales=first?s0:s1;
            device const ushort *biases=first?b0:b1;
            float result[2][8]={{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0}};
            for(uint lo=lane*16;lo<WIDTH;lo+=512) {
                float values[2][16];float biasSum[2]={0,0};
                #pragma unroll
                for(uint xr=0;xr<XROWS;xr++) {
                    #pragma unroll
                    for(uint i=0;i<16;i+=4) {
                        #pragma unroll
                        for(uint j=0;j<4;j++)values[xr][i+j]=from_bf16(x[xr*WIDTH+lo+i+j]);
                        biasSum[xr]=biasSum[xr]+round_bf16(round_bf16(round_bf16(values[xr][i]+values[xr][i+1])+values[xr][i+2])+values[xr][i+3]);
                    }
                }
                #pragma unroll
                for(uint r=0;r<8;r++) {
                    uint2 packed=*reinterpret_cast<device const uint2*>(words+(row+r)*(WIDTH/8)+lo/8);
                    uint metadata=(row+r)*(WIDTH/QGROUP)+lo/QGROUP;
                    float scale=from_bf16(scales[metadata]),bias=from_bf16(biases[metadata]);
                    #pragma unroll
                    for(uint xr=0;xr<XROWS;xr++) {
                        float quantSum=0;
                        #pragma unroll
                        for(uint i=0;i<16;i+=4) {
                            uint word=packed[i/8];
                            float p0=values[xr][i]*float((word>>((i%8)*4))&15);
                            float p1=values[xr][i+1]*float((word>>(((i+1)%8)*4))&15);
                            float p2=values[xr][i+2]*float((word>>(((i+2)%8)*4))&15);
                            float p3=values[xr][i+3]*float((word>>(((i+3)%8)*4))&15);
                            quantSum=quantSum+(((p0+p1)+p2)+p3);
                        }
                        float dot=fma(scale,quantSum,biasSum[xr]*bias);
                        result[xr][r]=result[xr][r]+dot;
                    }
                }
            }
            uint stride=first?FIRST_ROWS:0;
            #pragma unroll
            for(uint xr=0;xr<XROWS;xr++) {
                #pragma unroll
                for(uint r=0;r<8;r++) {float sum=simd_sum(result[xr][r]);if(lane==0)out[xr*stride+row+r]=round_bf16(sum);}
            }
        }

        """#
}

extension DecodeProjectionKernel {
    /// Check: two input rows give each row exactly what the one-row kernel gives (widths 2560, 6144, 10240), and the
    /// two-row-only width 320 agrees with a float reference within bf16 rounding.
    package static func twoRowSelfCheck() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Missing Metal device")
        }
        let kernel = try DecodeProjectionKernel(device: device)
        var state: UInt64 = 3
        func next() -> UInt32 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt32(truncatingIfNeeded: state >> 33)
        }
        func buffer<T>(_ values: [T]) -> MTLBuffer {
            values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        for width in [2560, 6144, 10240, 320] {
            let outputs = 64
            let group = ModelLayout.group
            let words = (0..<(outputs * width / 8)).map { _ in next() }
            let scales = (0..<(outputs * width / group)).map { _ in BF16.encode(Float(next() % 19 + 1) / 512) }
            let biases = (0..<(outputs * width / group)).map { _ in BF16.encode(Float(Int(next() % 23) - 11) / 256) }
            let x = (0..<(2 * width)).map { BF16.encode(Float(($0 * 103) % 991 - 495) / 193) }
            let weights = Weights(words: buffer(words), scales: buffer(scales), biases: buffer(biases), rows: outputs)
            func run(_ input: [UInt16], rows: Int) throws -> [Float] {
                let out = device.makeBuffer(length: rows * outputs * 4, options: .storageModeShared)!
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Command unavailable") }
                try kernel.encode(command, input: buffer(input), width: width, first: weights, firstOutput: out, inputRows: rows)
                command.commit()
                try command.waitBounded()
                return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: rows * outputs))
            }
            let pair = try run(x, rows: 2)
            if width != 320 {
                let one = try run(Array(x[0..<width]), rows: 1) + run(Array(x[width...]), rows: 1)
                guard pair.map(\.bitPattern) == one.map(\.bitPattern) else {
                    throw EngineError("Two-row projection differs from one row at width \(width)")
                }
            } else {
                for row in 0..<2 {
                    for o in 0..<outputs {
                        var sum: Double = 0
                        for i in 0..<width {
                            let word = words[o * width / 8 + i / 8]
                            let q = Double((word >> UInt32((i % 8) * 4)) & 15)
                            let g = o * width / group + i / group
                            sum +=
                                Double(BF16.decode(x[row * width + i]))
                                * (q * Double(BF16.decode(scales[g])) + Double(BF16.decode(biases[g])))
                        }
                        guard abs(Double(pair[row * outputs + o]) - sum) <= 0.02 * max(1, abs(sum)) else {
                            throw EngineError("Two-row width-320 projection row \(row) output \(o): \(pair[row * outputs + o]) vs \(sum)")
                        }
                    }
                }
            }
        }
    }
}
