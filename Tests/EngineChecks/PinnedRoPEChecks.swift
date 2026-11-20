import Darwin
import Foundation
import OrigamiEngine

private func ropeRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
func pinnedRoPEConfigurationAndGeometry() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-rope-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func config(_ value: Any) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": ["pinned_rope": value]])
            .write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    try ropeRequire(!RuntimeConfiguration().pinnedRoPE, "Pinned RoPE must default off")
    try ropeRequire(try config(true).pinnedRoPE, "Pinned RoPE flag was not loaded")
    for value in [1, "true"] as [Any] {
        var rejected = false
        do { _ = try config(value) } catch is EngineError { rejected = true }
        try ropeRequire(rejected, "Pinned RoPE accepted nonboolean config")
    }
    let ledger = try MemoryLedger()
    try autoreleasepool {
        let rope = try PinnedRoPE(ledger: ledger)
        for positions in [[], [-1], [262144], Array(0...256)] {
            var rejected = false
            do { _ = try rope.tables(positions: positions) } catch is EngineError { rejected = true }
            try ropeRequire(rejected, "Pinned RoPE accepted invalid positions")
        }
        var rejected = false
        do { _ = try rope.rotation([0], width: 256, heads: 1, positions: [0]) } catch is EngineError { rejected = true }
        try ropeRequire(rejected, "Pinned RoPE accepted invalid input geometry")
    }
    try ropeRequire(ledger.snapshot().reserved == 0, "Pinned RoPE validation leaked a reservation")
}
func pinnedRoPERoundingCacheAndRetirement() throws {
    let ledger = try MemoryLedger()
    try autoreleasepool {
        let rope = try PinnedRoPE(ledger: ledger)
        let inverseBytes = ledger.snapshot().reserved
        let table = try rope.tables(positions: [64, 128])
        try ropeRequire(
            table.cosines[3].bitPattern == 0x3c67_0000 && table.sines[35].bitPattern == 0x3ce7_0000,
            "RoPE reverted to CPU libm or non-pinned trig rounding")
        _ = try rope.tables(positions: [64, 128])
        try ropeRequire(rope.tableBuilds == 1 && rope.tableHits == 1, "Pinned RoPE cache missed identical absolute geometry")
        _ = try rope.tables(positions: [0])
        _ = try rope.tables(positions: [1])
        _ = try rope.tables(positions: [64, 128])
        try ropeRequire(rope.tableBuilds == 4, "Pinned RoPE cache did not retain exactly two recent entries")
        var x = [Float](repeating: 0, count: 2 * 256)
        x[3] = 1
        x[256 + 35] = 1
        let y = try rope.rotation(x, width: 256, heads: 1, positions: [64, 128])
        try ropeRequire(
            y[3].bitPattern == 0x3c67_0000 && y[256 + 3].bitPattern == 0xbce7_0000,
            "Pinned rotation changed BF16 boundaries or half-rotation sign")
        rope.clearTables()
        try ropeRequire(ledger.snapshot().reserved == inverseBytes, "Pinned RoPE retained tables after clear")
    }
    try ropeRequire(ledger.snapshot().reserved == 0, "Pinned RoPE retained buffers after lifetime")
    let limited = try MemoryLedger(ceiling: 65_536, transientMargin: 0)
    try autoreleasepool {
        let rope = try PinnedRoPE(ledger: limited)
        var rejected = false
        do { _ = try rope.rotation(.init(repeating: 0, count: 256 * 24 * 256), width: 256, heads: 24, positions: Array(0..<256)) } catch let
            error as EngineError
        { rejected = error.code == "insufficient_memory" }
        try ropeRequire(rejected, "Pinned RoPE bypassed memory admission")
        rope.clearTables()
    }
    try ropeRequire(limited.snapshot().reserved == 0, "Pinned RoPE failed admission leaked memory")
}
