import Darwin
import Foundation
import Metal
import OrigamiEngine

private func quantizedTiledRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func quantizedTiledReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid tiled quantized input was accepted")
}

func quantizedTiledMatchesSplitKAndLeavesOtherSchedulesAlone() throws {
    let old = try ReferenceKernels()
    let tiled = try ReferenceKernels(quantizedTiled: true)
    // Includes source model shapes and both canonical split reducer regimes.
    // Uneven M, group32, scalar, grouped-expert, QMV and unaligned N are explicit.
    let g = ModelLayout.group
    let other = g == 64 ? 32 : 64
    let shapes: [(Int, Int, Int, Int, Bool, Bool)] = [
        (6, 10240, 320, g, true, false), (17, 2560, 640, g, true, false),
        (33, 6144, 2560, g, true, false), (65, 2560, 512, g, true, false),
        (256, 320, 10240, g, true, false), (15, 128, 32, g, true, false),
        (9, 2560, 640, g, true, false), (17, 128, 48, g, true, false),
        (17, 128, 64, other, true, false), (17, 128, 64, g, false, false),
        (17, 128, 64, g, true, true),
    ]
    for (rows, width, outputs, group, simd, grouped) in shapes {
        try autoreleasepool {
            let input = (0..<(rows * width)).map { BF16.decode(BF16.encode(Float(($0 * 113 + 29) % 1009 - 504) / 129)) }
            let words = (0..<(outputs * width / 8)).map { UInt32(truncatingIfNeeded: UInt64($0 + 1) &* 0x9e37_79b9_7f4a_7c15) }
            let scales = (0..<(outputs * width / group)).map { BF16.encode(Float($0 % 19 + 1) / 257) }
            let biases = scales.indices.map { BF16.encode(Float($0 % 23 - 11) / 131) }
            let a = try old.qlinearPinnedBF16(
                x: input, width: width, words: words, scales: scales, biases: biases,
                groupSize: group, simdMatrix: simd, groupedExpert: grouped)
            let b = try tiled.qlinearPinnedBF16(
                x: input, width: width, words: words, scales: scales, biases: biases,
                groupSize: group, simdMatrix: simd, groupedExpert: grouped)
            let eligible =
                simd && !grouped && group == ModelLayout.group && outputs % 32 == 0 && a.schedule.kind.hasPrefix("qmm")
                && a.schedule.partitionSize % 64 == 0
            try quantizedTiledRequire(
                a.schedule == b.schedule && a.output.map(\.bitPattern) == b.output.map(\.bitPattern),
                "Tiled projection changed split arithmetic at M\(rows), K\(width), N\(outputs), G\(group)")
            try quantizedTiledRequire(
                eligible ? b.implementation == "simdgroup_tiled16x32_k32_bf16_partitions" : a.implementation == b.implementation,
                "Tiled projection selected an ineligible schedule or skipped an eligible shape")
        }
    }
}

func quantizedTiledBufferOffsetsAndInvalidSpansStayExact() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing quantized fixture Metal device") }
    let old = try ReferenceKernels()
    let tiled = try ReferenceKernels(quantizedTiled: true)
    let rows = 17
    let width = 2560
    let outputs = 512
    let input = (0..<(rows * width)).map { Float($0 % 31 - 15) / 32 }
    let wordOffset = 256
    let scaleOffset = wordOffset + outputs * width / 2 + 128
    let biasOffset = scaleOffset + outputs * width / 64 * 2 + 192
    guard let buffer = device.makeBuffer(length: biasOffset + outputs * width / 64 * 2 + 128, options: .storageModeShared) else {
        throw EngineError("Quantized offset fixture allocation failed")
    }
    memset(buffer.contents(), 0xa5, buffer.length)
    let words = buffer.contents().advanced(by: wordOffset).assumingMemoryBound(to: UInt32.self)
    for i in 0..<(outputs * width / 8) { words[i] = UInt32(truncatingIfNeeded: UInt64(i + 7) &* 0x9e37_79b9_7f4a_7c15) }
    let scales = buffer.contents().advanced(by: scaleOffset).assumingMemoryBound(to: UInt16.self)
    let biases = buffer.contents().advanced(by: biasOffset).assumingMemoryBound(to: UInt16.self)
    for i in 0..<(outputs * width / 64) {
        scales[i] = BF16.encode(Float(i % 11 + 1) / 128)
        biases[i] = BF16.encode(Float(i % 7 - 3) / 64)
    }
    func project(_ kernels: ReferenceKernels, offset: Int = 256) throws -> [Float] {
        try kernels.qlinearPinnedBF16(
            x: input, width: width, outputs: outputs, words: buffer, scales: buffer, biases: buffer,
            wordOffset: offset, scaleOffset: scaleOffset, biasOffset: biasOffset, simdMatrix: true
        ).output
    }
    try quantizedTiledRequire(
        try project(old).map(\.bitPattern) == project(tiled).map(\.bitPattern), "Tiled split-K lost a nonzero record binding offset")
    for offset in [-4, 1, buffer.length - 4, Int.max] { try quantizedTiledReject { _ = try project(tiled, offset: offset) } }
    try quantizedTiledReject {
        _ = try tiled.qlinearPinnedBF16(
            x: [1], width: width, outputs: outputs, words: buffer, scales: buffer, biases: buffer, simdMatrix: true)
    }
}

func quantizedTiledConfigurationIsDefaultOffAndRequiresSIMD() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-quantized-tiled-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    try quantizedTiledRequire(
        try !load([:]).quantizedTiled && load(["quantized_tiled": true, "simd_matrix": true]).quantizedTiled,
        "Default or explicit tiled quantized configuration changed")
    try quantizedTiledReject { _ = try load(["quantized_tiled": true]) }
    for value in [1, "true"] as [Any] { try quantizedTiledReject { _ = try load(["quantized_tiled": value, "simd_matrix": true]) } }
}
