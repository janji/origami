import Foundation
import OrigamiEngine

/// `engine-checks --measure-conversion-error <file.gguf> [<mlx.safetensors>]`: what `model add` of a Qwen GGUF does
/// to each 2-D tensor of the file (ROADMAP D4's offline measure, no model run): the GGUF's own values (Origami's ggml
/// dequantizer), quantized as the conversion does (MLX's 4-bit affine rule, group 64) and decoded, and the relative
/// RMS error between the two. With a safetensors file holding, per GGUF tensor name, MLX 4-bit codes of the same
/// weights quantized from BF16 (`<name>.weight`, `.scales`, `.biases`; a 4-bit MLX build's tensors, renamed), also
/// the error between those and the GGUF's values. Prints one JSON object per tensor.
func measureConversionError(_ arguments: [String]) throws {
    guard let path = arguments.first else {
        throw EngineError("Usage: engine-checks --measure-conversion-error <file.gguf> [<mlx.safetensors>]")
    }
    let gguf = try GGUFSource(URL(fileURLWithPath: path))
    let mlx = try arguments.dropFirst().first.map { file -> (URL, [String: TensorSpan]) in
        let url = URL(fileURLWithPath: file)
        return (url, try ModelSource.header(of: url, file: url.lastPathComponent))
    }
    func bytes(_ url: URL, _ offset: Int64, _ length: Int64) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(offset))
        return try handle.read(upToCount: Int(length)) ?? Data()
    }
    func relative(_ a: [Float], _ b: [Float]) -> Double {
        var error = 0.0
        var norm = 0.0
        for i in a.indices {
            error += Double(a[i] - b[i]) * Double(a[i] - b[i])
            norm += Double(b[i]) * Double(b[i])
        }
        return norm > 0 ? (error / norm).squareRoot() : 0
    }
    for tensor in gguf.tensors.values.sorted(by: { $0.name < $1.name }) where tensor.shape.count == 2 {
        let rows = tensor.shape[0]
        let columns = tensor.shape[1]
        guard columns % 64 == 0, GGUFQuant.readable.contains(tensor.type) else { continue }
        let raw = try bytes(gguf.url(tensor.file), tensor.offset, tensor.length)
        var values = [Float](repeating: 0, count: rows * columns)
        raw.withUnsafeBytes { source in
            values.withUnsafeMutableBufferPointer {
                GGUFQuant.dequantize(source.baseAddress!, type: tensor.type, count: rows * columns, into: $0.baseAddress!)
            }
        }
        var codes = [UInt32](repeating: 0, count: values.count / 8)
        var scales = [UInt16](repeating: 0, count: values.count / 64)
        var biases = scales
        values.withUnsafeBytes { v in
            codes.withUnsafeMutableBufferPointer { c in
                scales.withUnsafeMutableBufferPointer { s in
                    biases.withUnsafeMutableBufferPointer { b in
                        CheckpointConverter.quantize(
                            v.baseAddress!, dtype: "F32", rows: rows, columns: columns, group: 64, codes: c.baseAddress!,
                            scales: s.baseAddress!, biases: b.baseAddress!)
                    }
                }
            }
        }
        var converted = [Float](repeating: 0, count: values.count)
        codes.withUnsafeBytes {
            CheckpointConverter.decodeAffine(
                $0.baseAddress!, rows: 0..<rows, columns: columns, bits: 4, group: 64, scales: scales.map(BF16.decode),
                biases: biases.map(BF16.decode), into: &converted)
        }
        var line: [String: Any] = [
            "name": tensor.name, "type": GGUFQuant.name(tensor.type), "shape": tensor.shape,
            "converted_vs_gguf": relative(converted, values),
        ]
        if let (url, header) = mlx, let w = header[tensor.name + ".weight"], let s = header[tensor.name + ".scales"],
            let b = header[tensor.name + ".biases"], s.shape.count == 2, s.shape[0] == rows, columns % s.shape[1] == 0
        {
            func floats(_ span: TensorSpan) throws -> [Float] {
                try bytes(url, span.offset, span.length).withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }.map(BF16.decode)
            }
            var reference = [Float](repeating: 0, count: values.count)
            try bytes(url, w.offset, w.length).withUnsafeBytes {
                CheckpointConverter.decodeAffine(
                    $0.baseAddress!, rows: 0..<rows, columns: columns, bits: 4, group: columns / s.shape[1], scales: try floats(s),
                    biases: try floats(b), into: &reference)
            }
            line["mlx_group"] = columns / s.shape[1]
            line["mlx4_vs_gguf"] = relative(reference, values)
            line["converted_vs_mlx4"] = relative(converted, reference)
        }
        print(String(decoding: try JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]), as: UTF8.self))
    }
}
