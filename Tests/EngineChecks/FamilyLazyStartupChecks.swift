import Foundation
import Metal
import OrigamiEngine

/// Real family construction below the old fixed workspace reservation, followed by prompt and decode.
/// Tiny nonzero weights; the automatic ledger tests ownership independently of this test process's footprint.
func familyLazyStartupWorkspace() throws {
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Lazy startup workspace: " + message) }
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-lazy-startup-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let cap: UInt64 = 128 << 20
    for family in [ModelFamily.deepseek, .glm] {
        try autoreleasepool {
            let url = try poolModelFixture(family, in: folder)
            let reference = try makeFamilyModel(
                FamilyWeights(gguf: GGUFSource(url), device: device),
                expertBudget: 1 << 20, context: 8192
            ) { _ in }
            let input = (0..<269).map { 1 + $0 % 7 }
            let expectedPrompt = try reference.forward(input)
            (reference as! FamilyRunner).beginDecode()
            let expectedNext = try reference.forward([9])
            let expectedState = reference.stateParts(at: 270).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            let ledger = try MemoryLedger(ceiling: cap, transientMargin: 1, availableFloor: 1)
            let model = try makeFamilyModel(
                FamilyWeights(gguf: GGUFSource(url), device: device), expertBudget: nil,
                memoryLimit: cap, context: 8192, ledger: ledger
            ) { _ in }
            let runner = model as! FamilyRunner
            try require(
                runner.workingBytes > 0 && runner.workingBytes < cap
                    && runner.workingAllocatedBytes == 0, "startup did not reserve only a small unallocated forward plan")
            try require(ledger.reservedBytes < cap, "startup exceeded cap")
            // A first prompt read with no room for its workspace is refused as before, and leaves the model usable.
            runner.experts.setLimit(bytes: 0)
            let occupied = try ledger.reserve(bytes: ledger.maximumUsableBytes - ledger.reservedBytes, category: "lazy-check-pressure")
            let reserved = ledger.reservedBytes
            do {
                _ = try model.forward(input)
                throw EngineError("Lazy startup workspace: a prompt read without room for its workspace ran")
            } catch let error as EngineError where error.code == "insufficient_memory" {}
            // The unallocated one-row allowance may be returned; nothing is allocated or charged on top.
            try require(
                runner.workingAllocatedBytes == 0 && runner.position == 0 && ledger.reservedBytes <= reserved,
                "a refused prompt read left scratch, state or a charge")
            try ledger.release(occupied)
            runner.fitExperts()
            let actualPrompt = try model.forward(input)
            try require(actualPrompt.map(\.bitPattern) == expectedPrompt.map(\.bitPattern), "prompt logits differ")
            try require(
                runner.workingBytes >= runner.workingAllocatedBytes && ledger.reservedBytes < cap,
                "prompt scratch was not admitted within cap")
            runner.beginDecode()
            let actualNext = try model.forward([9])
            try require(actualNext.map(\.bitPattern) == expectedNext.map(\.bitPattern), "decode logits differ")
            let actualState = model.stateParts(at: 270).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
            try require(actualState == expectedState, "continued state differs")
            try require(
                runner.workingBytes >= runner.workingAllocatedBytes && ledger.reservedBytes < cap,
                "decode scratch was not admitted within cap")
        }
    }
}
