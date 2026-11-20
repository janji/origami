import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// Separate one-time offline preparation; never rewrites a GGUF or an existing destination.
func buildResidentPrecisionPack(_ arguments: [String]) throws {
    guard arguments.count == 2 else { throw EngineError("Usage: engine-checks --build-resident-precision <shard1.gguf> <new-folder>") }
    let gguf = try GGUFSource(URL(fileURLWithPath: arguments[0]))
    let geometry = try ModelGeometry(gguf: gguf)
    let destination = URL(fileURLWithPath: arguments[1], isDirectory: true)
    let started = ProcessInfo.processInfo.systemUptime
    try FamilyResidentPack.build(gguf, geometry: geometry, into: destination) { done, total in
        FileHandle.standardError.write(Data("Resident conversion: \(done)/\(total) matrices\n".utf8))
    }
    let pack = try FamilyResidentPack.open(destination, gguf: gguf, geometry: geometry)
    var result = pack.status
    result["folder"] = destination.path
    result["wall_seconds"] = ProcessInfo.processInfo.systemUptime - started
    result["scope"] = "Offline conversion only; no model quality or speed validation"
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]))
    FileHandle.standardOutput.write(Data("\n".utf8))
}

/// A bounded, read-only O40 screen: real resident matrices, synthetic input rows, existing Q8/Q4 kernels.
/// This measures conversion error and isolated GPU cost, never language-model quality or cache hit rate.
func residentPrecisionBenchmark(_ arguments: [String]) throws {
    guard arguments.count == 1 else { throw EngineError("Usage: engine-checks --bench-resident-precision <shard1.gguf>") }
    let source = try GGUFSource(URL(fileURLWithPath: arguments[0]))
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("Resident precision screen needs Metal")
    }
    let kernels = try GGMLMatrixKernels(device: device)
    var records: [[String: Any]] = []
    for name in ["blk.0.attn_q_b.weight", "blk.0.attn_output_b.weight", "blk.0.ffn_gate_shexp.weight"] {
        try autoreleasepool {
            guard let t = source.tensors[name], t.type == 8, t.shape.count == 2,
                let width = t.shape.last, width > 0, width % 256 == 0,
                t.shape[0] > 0, t.length > 0, t.length <= 128 * 1024 * 1024
            else {
                throw EngineError("Expected a bounded rank-two Q8_0 matrix: \(name)")
            }
            let outputs = t.shape[0]
            let sourceRow = GGUFQuantize.bytes(8, count: width)
            let targetRow = GGUFQuantize.bytes(12, count: width)
            guard Int(t.length) == outputs * sourceRow else { throw EngineError("Invalid matrix span: \(name)") }
            func buffer(_ size: Int) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: size, options: .storageModeShared) else {
                    throw EngineError("Resident precision screen allocation failed")
                }
                return b
            }
            let q8 = try buffer(Int(t.length))
            let q4 = try buffer(outputs * targetRow)
            let file = try ExactFile(url: source.folder.appendingPathComponent(t.file))
            try file.read(offset: t.offset, into: UnsafeMutableRawBufferPointer(start: q8.contents(), count: Int(t.length)))
            try file.validateIdentity()
            let digest = SHA256.hash(data: Data(bytesNoCopy: q8.contents(), count: Int(t.length), deallocator: .none))
                .map { String(format: "%02x", $0) }.joined()
            var original = [Float](repeating: 0, count: width)
            var decoded = original
            var signal: Double = 0
            var error: Double = 0
            var maxError: Double = 0
            var encodeSeconds: Double = 0
            let start = ProcessInfo.processInfo.systemUptime
            for row in 0..<outputs {
                try original.withUnsafeMutableBufferPointer { a in
                    GGUFQuant.dequantize(q8.contents() + row * sourceRow, type: 8, count: width, into: a.baseAddress!)
                    guard a.allSatisfy(\.isFinite) else { throw EngineError("Non-finite source values: \(name)") }
                    let encodeStart = ProcessInfo.processInfo.systemUptime
                    GGUFQuantize.encode(a.baseAddress!, count: width, type: 12, into: q4.contents() + row * targetRow)
                    encodeSeconds += ProcessInfo.processInfo.systemUptime - encodeStart
                    try decoded.withUnsafeMutableBufferPointer { b in
                        GGUFQuant.dequantize(q4.contents() + row * targetRow, type: 12, count: width, into: b.baseAddress!)
                        guard b.allSatisfy(\.isFinite) else { throw EngineError("Non-finite converted values: \(name)") }
                        for i in 0..<width {
                            let x = Double(a[i])
                            let delta = Double(b[i]) - x
                            signal += x * x
                            error += delta * delta
                            maxError = max(maxError, abs(delta))
                        }
                    }
                }
            }
            guard signal > 0 else { throw EngineError("All-zero source matrix: \(name)") }
            let conversionAndValidation = ProcessInfo.processInfo.systemUptime - start
            let input = try buffer(width * 32 * 4)
            let out8 = try buffer(outputs * 32 * 4)
            let out4 = try buffer(outputs * 32 * 4)
            var rng: UInt64 = 0x42b3_e52a_b187
            let x = input.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(width * 32) {
                rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                x[i] = Float(Int((rng >> 32) % 20001) - 10000) / 10000
            }
            func run(type: Int, rows: Int) throws -> (gpu: Double, wall: Double) {
                let begin = ProcessInfo.processInfo.systemUptime
                guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                    throw EngineError("Resident precision screen could not create command")
                }
                try kernels.encode(
                    e, type: type, weights: type == 8 ? q8 : q4, weightOffset: 0,
                    width: width, outputs: outputs, input: input, inputOffset: 0, rows: rows,
                    output: type == 8 ? out8 : out4, outputOffset: 0)
                e.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                guard cb.status == .completed, cb.gpuEndTime > cb.gpuStartTime else {
                    throw EngineError("Resident precision screen GPU failure: \(String(describing: cb.error))")
                }
                return ((cb.gpuEndTime - cb.gpuStartTime) * 1000, (ProcessInfo.processInfo.systemUptime - begin) * 1000)
            }
            var trials: [[String: Any]] = []
            for rows in [1, 8, 32] {
                for _ in 0..<2 {
                    _ = try run(type: 8, rows: rows)
                    _ = try run(type: 12, rows: rows)
                }
                var timing: [Int: [(gpu: Double, wall: Double)]] = [:]
                for round in 0..<8 {
                    for type in round % 2 == 0 ? [8, 12] : [12, 8] {
                        timing[type, default: []].append(try run(type: type, rows: rows))
                    }
                }
                let a = out8.contents().assumingMemoryBound(to: Float.self)
                let b = out4.contents().assumingMemoryBound(to: Float.self)
                var aa: Double = 0
                var bb: Double = 0
                var ab: Double = 0
                var delta2: Double = 0
                for i in 0..<(outputs * rows) {
                    guard a[i].isFinite, b[i].isFinite else { throw EngineError("Non-finite GPU output: \(name)") }
                    let av = Double(a[i])
                    let bv = Double(b[i])
                    aa += av * av
                    bb += bv * bv
                    ab += av * bv
                    delta2 += (av - bv) * (av - bv)
                }
                guard aa > 0, bb > 0 else { throw EngineError("All-zero GPU output: \(name)") }
                func median(_ values: [Double]) -> Double {
                    let s = values.sorted()
                    return (s[3] + s[4]) / 2
                }
                trials.append([
                    "input_rows": rows, "q8_gpu_ms": timing[8]!.map(\.gpu), "q4_gpu_ms": timing[12]!.map(\.gpu),
                    "q8_gpu_ms_median": median(timing[8]!.map(\.gpu)), "q4_gpu_ms_median": median(timing[12]!.map(\.gpu)),
                    "q8_wall_ms_median": median(timing[8]!.map(\.wall)), "q4_wall_ms_median": median(timing[12]!.map(\.wall)),
                    "output_relative_l2": sqrt(delta2 / aa), "output_cosine": ab / sqrt(aa * bb), "all_outputs_finite": true,
                ])
            }
            try file.validateIdentity()
            records.append([
                "name": name, "shape": t.shape, "source_sha256": digest, "source_file": t.file,
                "source_offset": t.offset, "q8_bytes": Int(t.length), "q4_bytes": q4.length,
                "weight_relative_l2": sqrt(error / signal), "weight_max_absolute_error": maxError,
                "encode_seconds": encodeSeconds, "conversion_and_validation_seconds": conversionAndValidation, "trials": trials,
            ])
            FileHandle.standardError.write(Data("Completed resident precision screen: \(name)\n".utf8))
        }
    }
    let result: [String: Any] = [
        "scope":
            "Read-only real resident matrix payloads; Q8_0 to Q4_K; synthetic input rows; warm isolated GPU kernels. No full-model generation, cache-policy, language quality or end-to-end speed measurement.",
        "utc": ISO8601DateFormatter().string(from: Date()), "device": device.name, "source": arguments[0],
        "correctness": "non-exact; numerical error is a proxy, not task quality", "records": records,
    ]
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]))
    FileHandle.standardOutput.write(Data("\n".utf8))
}
