import Foundation
import OrigamiEngine

private func bm32Require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func bm32Reject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid BM32 configuration accepted")
}
func quantizedBM32Configuration() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("origami-bm32-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: file) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: file)
        return try RuntimeConfiguration.load(file)
    }
    let defaults = try load([:])
    try bm32Require(!defaults.quantizedTiledBM32 && defaults.quantizedBM32MinRows == 32, "BM32 default changed")
    let flags: [String: Any] = ["simd_matrix": true, "quantized_tiled": true, "quantized_tiled_bm32": true]
    try bm32Require(try load(flags).quantizedTiledBM32, "BM32 explicit selection ignored")
    try bm32Reject { _ = try load(["quantized_tiled_bm32": true, "simd_matrix": true]) }
    for floor in [0, 257, 1.5, true, "32"] as [Any] {
        var input = flags
        input["quantized_bm32_min_rows"] = floor
        try bm32Reject { _ = try load(input) }
    }
    for value in [1, "true"] as [Any] {
        var input = flags
        input["quantized_tiled_bm32"] = value
        try bm32Reject { _ = try load(input) }
    }
    try bm32Reject { _ = try ReferenceKernels(quantizedTiledBM32: true) }
    try bm32Reject { _ = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 0) }
}
func quantizedBM32BoundaryParity() throws {
    let old = try ReferenceKernels(quantizedTiled: true)
    let next = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true)
    let always = try ReferenceKernels(quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 1)
    let shapes = [(16, 2560, 640), (17, 10240, 320), (32, 2560, 512), (33, 6144, 2560), (255, 640, 2560), (256, 320, 10240)]
    for (rows, width, outputs) in shapes {
        try autoreleasepool {
            let input = (0..<(rows * width)).map { BF16.decode(BF16.encode(Float(($0 * 29 + 5) % 503 - 251) / 43)) }
            let words = (0..<(width * outputs / 8)).map { UInt32(truncatingIfNeeded: UInt64($0 + 3) &* 0x9e37_79b9_7f4a_7c15) }
            let scales = (0..<(width * outputs / ModelLayout.group)).map { BF16.encode(Float($0 % 19 + 1) / 257) }
            let biases = scales.indices.map { BF16.encode(Float($0 % 23 - 11) / 131) }
            let a = try old.qlinearPinnedBF16(x: input, width: width, words: words, scales: scales, biases: biases, simdMatrix: true)
            let b = try next.qlinearPinnedBF16(x: input, width: width, words: words, scales: scales, biases: biases, simdMatrix: true)
            let c = try always.qlinearPinnedBF16(x: input, width: width, words: words, scales: scales, biases: biases, simdMatrix: true)
            try bm32Require(
                a.schedule == b.schedule && a.schedule == c.schedule && a.output.map(\.bitPattern) == b.output.map(\.bitPattern)
                    && a.output.map(\.bitPattern) == c.output.map(\.bitPattern),
                "BM32 changed M\(rows)/K\(width)/N\(outputs) arithmetic")
            let expected = rows >= 32 ? "simdgroup_tiled32x32_k32_bf16_partitions" : a.implementation
            try bm32Require(
                b.implementation == expected && c.implementation == "simdgroup_tiled32x32_k32_bf16_partitions",
                "BM32 row floor or explicit opt-in bypassed")
        }
    }
    // QMV, grouped experts, another group size and unaligned N retain their previous paths.
    let group = ModelLayout.group
    let other = group == 64 ? 32 : 64
    for (rows, width, outputs, group, grouped) in [
        (1, 2560, 640, group, false), (33, 128, 48, group, false), (33, 128, 64, other, false), (33, 128, 64, group, true),
    ] {
        let input = [Float](repeating: 0.125, count: rows * width)
        let words = [UInt32](repeating: 0x7654_3210, count: width * outputs / 8)
        let scales = [UInt16](repeating: 0x3e80, count: width * outputs / group)
        let biases = [UInt16](repeating: 0xbf00, count: scales.count)
        let a = try old.qlinearPinnedBF16(
            x: input, width: width, words: words, scales: scales, biases: biases, groupSize: group, simdMatrix: true, groupedExpert: grouped
        )
        let b = try always.qlinearPinnedBF16(
            x: input, width: width, words: words, scales: scales, biases: biases, groupSize: group, simdMatrix: true, groupedExpert: grouped
        )
        try bm32Require(
            a.implementation == b.implementation && a.output.map(\.bitPattern) == b.output.map(\.bitPattern),
            "BM32 changed an ineligible dispatch")
    }
}
