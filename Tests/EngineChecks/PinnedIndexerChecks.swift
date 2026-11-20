import Foundation
import OrigamiEngine

private func indexerRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
func pinnedIndexerConfigurationAndBounds() throws {
    try indexerRequire(!RuntimeConfiguration().pinnedIndexer, "Pinned indexer must default off")
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-indexer-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func config(_ value: Any) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: [
            "schema": "origami-config-v1", "profile": "exact", "execution": ["pinned_indexer": value],
        ]).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    try indexerRequire(try config(true).pinnedIndexer, "Pinned indexer flag was not loaded")
    for value in [1, "true"] as [Any] {
        var rejected = false
        do { _ = try config(value) } catch is EngineError { rejected = true }
        try indexerRequire(rejected, "Nonboolean pinned indexer flag accepted")
    }
    for (rows, blocks, tile) in [(256, 512, 256), (256, 8192, 256), (256, 32768, 64), (256, 65536, 32), (33, 65536, 32), (1, 65536, 1)] {
        try indexerRequire(try PinnedIndexer.tileRows(rows: rows, blocks: blocks) == tile, "Pinned indexer dot scratch bound changed")
    }
    for (rows, blocks) in [(0, 512), (257, 512), (1, 511), (1, 65537), (Int.max, Int.max)] {
        var rejected = false
        do { _ = try PinnedIndexer.tileRows(rows: rows, blocks: blocks) } catch is EngineError { rejected = true }
        try indexerRequire(rejected, "Invalid indexer plan accepted")
    }
}
func pinnedIndexerKnownAnswerAndRetirement() throws {
    let ledger = try MemoryLedger()
    try autoreleasepool {
        let indexer = try PinnedIndexer(ledger: ledger)
        let selector = try BatchedTopIndices(ledger: ledger)
        let blocks = 513
        var query = [Float](repeating: 0, count: 2 * 512)
        var pooled = [Float](repeating: 0, count: blocks * 128)
        query[0] = 1
        query[512 + 2 * 128 + 1] = 1
        query[512 + 3 * 128 + 1] = 1
        for block in 0..<blocks {
            pooled[block * 128] = block % 4 == 0 ? 1 : (block % 4 == 1 ? -1 : 0)
            pooled[block * 128 + 1] = block % 4 == 2 ? 1 : (block % 4 == 3 ? -1 : 0)
        }
        let scores = try indexer.scores(query: query, pooledKeys: pooled)
        for i in scores.indices {
            let row = i / blocks
            let block = i % blocks
            let expected: UInt32 = row == 0 && block % 4 == 0 ? 0x3db5_04f3 : row == 1 && block % 4 == 2 ? 0x3e35_04f3 : 0
            try indexerRequire(scores[i].bitPattern == expected, "Pinned indexer positive-head score changed")
        }
        let selected = try selector.selectedPositions(scores: scores, positions: [2048, 2051], kvLength: 2052)
        try indexerRequire(
            selected == [(0...2048).map(UInt32.init), (0..<2048).map(UInt32.init)], "Pinned indexer causal mask/tail/tie boundary changed")
    }
    try indexerRequire(ledger.snapshot().reserved == 0, "Pinned indexer scratch did not retire")
    let limited = try MemoryLedger(ceiling: 65_536, transientMargin: 0)
    try autoreleasepool {
        let indexer = try PinnedIndexer(ledger: limited)
        var rejected = false
        do { _ = try indexer.scores(query: .init(repeating: 0, count: 512), pooledKeys: .init(repeating: 0, count: 513 * 128)) } catch let
            error as EngineError
        { rejected = error.code == "insufficient_memory" }
        try indexerRequire(rejected, "Pinned indexer allocation bypassed ledger admission")
    }
    try indexerRequire(limited.snapshot().reserved == 0, "Pinned indexer allocation failure retained scratch")
}
