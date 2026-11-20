import Foundation
import Metal
import OrigamiEngine

/// Every returned position must be a real head, including earlier last-layer experts. Exercise odd pooling
/// carries and a wrapped window; a full snapshot is the simple research rollback baseline.
func deepSeekBlockLogits() throws {
    func require(_ ok: Bool, _ message: String) throws { if !ok { throw EngineError("DeepSeek block: " + message) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-block-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = try poolModelFixture(.deepseek, in: folder)
    let model = try DeepSeekModel(weights: FamilyWeights(gguf: GGUFSource(url), device: device), expertBudget: 1 << 20, context: 512)
    for invalid in [[], Array(repeating: 1, count: 9)] {
        do {
            _ = try model.forwardBlock(invalid)
            throw EngineError("DeepSeek block: accepted invalid width")
        } catch let error as EngineError where error.code == "invalid_configuration" {}
    }
    for count in [0, 3, 263, 264] {
        try model.reset()
        let prefix = (0..<count).map { 1 + $0 % 7 }
        if !prefix.isEmpty { _ = try model.forward(prefix) }
        let parts = model.stateParts(at: count)
        let snapshot = parts.map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
        func restore() {
            for (part, data) in zip(parts, snapshot) where !data.isEmpty {
                data.withUnsafeBytes { part.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
            model.restored(prefix)
        }
        for width in [1, 2, 4, 8] {
            restore()
            let block = (0..<width).map { 8 + $0 }
            let rows = try model.forwardBlock(block)
            try require(rows.count == width && rows.allSatisfy { $0.count == 32 && $0.allSatisfy(\.isFinite) }, "row shape/finiteness")
            let blockState = model.stateParts(at: count + width).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            restore()
            let last = try model.forward(block)
            let normalState = model.stateParts(at: count + width).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            try require(
                rows.last!.map(\.bitPattern) == last.map(\.bitPattern) && blockState == normalState,
                "last-row pruning changed final state/logits at \(count)+\(width)")
            // Independent causal prefixes compute only their final FFN/head. All earlier rows must agree with
            // these unpruned heads; allow the small trunk rounding drift inherent in differing matrix shapes.
            for row in block.indices {
                restore()
                let expected = try model.forward(Array(block.prefix(row + 1)))
                let worst = zip(expected, rows[row]).map { abs($0 - $1) }.max() ?? 0
                try require(worst < 0.00001, "head \(row) is not the causal prefix head: error \(worst)")
            }
            // Restore after a fully rejected block, then take one ordinary token. It must match the same
            // snapshot's ordinary path bit for bit, including the partial compression group and window ring.
            restore()
            let expected = try model.forward([19])
            let expectedState = model.stateParts(at: count + 1).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            restore()
            _ = try model.forwardBlock(block)
            restore()
            let got = try model.forward([19])
            let gotState = model.stateParts(at: count + 1).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            try require(
                got.map(\.bitPattern) == expected.map(\.bitPattern) && gotState == expectedState,
                "full-snapshot rollback did not restore ordinary continuation")
        }
    }
}

/// Multirow generation must not switch to prompt scan admission. Compare the same block arithmetic,
/// then exercise an error and a normal prefill to prove the scoped generation purpose does not leak.
func deepSeekGenerationBlockCachePhase() throws {
    func require(_ ok: Bool, _ message: String) throws { if !ok { throw EngineError("Generation block: " + message) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-generation-block-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = try poolModelFixture(.deepseek, in: folder)
    func make() throws -> DeepSeekModel {
        let model = try DeepSeekModel(weights: FamilyWeights(gguf: GGUFSource(url), device: device), expertBudget: 1 << 20, context: 512)
        model.beginDecode()
        return model
    }
    let reference = try make()
    let candidate = try make()
    let tokens = [1, 2, 3, 4]
    let before = try reference.forwardBlock(tokens)
    try require(
        !reference.isDecodePhase && reference.experts.uses.isEmpty,
        "default prompt path unexpectedly used generation policy")
    let after = try candidate.forwardBlock(tokens, generation: true)
    try require(
        candidate.isDecodePhase && !candidate.experts.uses.isEmpty,
        "generation block switched to prompt phase or bypassed usage recording")
    try require(
        before.map { $0.map(\.bitPattern) } == after.map { $0.map(\.bitPattern) }, "cache policy changed logits")
    let expected = reference.stateParts(at: tokens.count).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
    let actual = candidate.stateParts(at: tokens.count).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
    try require(expected == actual, "cache policy changed block state")
    candidate.shouldContinue = { false }
    do {
        _ = try candidate.forwardBlock([5, 6], generation: true)
        throw EngineError("Cancellation was ignored")
    } catch let error as EngineError where error.code == "cancelled" {}
    candidate.shouldContinue = { true }
    try candidate.reset()
    let uses = candidate.experts.uses
    _ = try candidate.forwardBlock([7, 8])
    try require(
        !candidate.isDecodePhase && candidate.experts.uses == uses,
        "generation purpose leaked through cancellation into a normal prefill")
}
