import Darwin
import Foundation
import OrigamiEngine

private func batchRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}

func pinnedAttentionBatchPlanBounds() throws {
    for (rows, keys, unitsPerBatch, batches, bytes) in [
        (256, 1024, 768, 1, 12_582_912),
        (256, 8192, 256, 3, 33_554_432),
        (256, 32768, 64, 12, 33_554_432),
        (256, 262144, 8, 96, 33_554_432),
        (3, 24, 12, 1, 4_608),
        (9, 8193, 30, 1, 3_932_640),
    ] {
        let plan = try PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys)
        try batchRequire(
            plan.units == ((rows + 1) / 2) * 6 && plan.unitsPerBatch == unitsPerBatch
                && plan.batches == batches && plan.scoreScratchBytes == bytes && plan.encoders == batches * 3,
            "Batched attention dispatch/scratch boundary changed")
    }
    for (rows, keys) in [(0, 1), (1, 1), (2, 2), (257, 1024), (24, 23), (3, 262145), (Int.max, Int.max)] {
        var rejected = false
        do { _ = try PinnedAttention.unfusedBatchPlan(queries: rows, keys: keys) } catch is EngineError { rejected = true }
        try batchRequire(rejected, "Invalid batched attention plan accepted")
    }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-batch-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func config(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    try batchRequire(!RuntimeConfiguration().batchedPinnedAttention, "Attention batching must default off")
    try batchRequire(
        try config(["pinned_attention": true, "batched_pinned_attention": true]).batchedPinnedAttention,
        "Attention batching config was not wired")
    let invalid: [[String: Any]] = [["batched_pinned_attention": true], ["batched_pinned_attention": 1]]
    for execution in invalid {
        var rejected = false
        do { _ = try config(execution) } catch is EngineError { rejected = true }
        try batchRequire(rejected, "Invalid attention batching config accepted")
    }
}

func pinnedAttentionBatchParityAndRetirement() throws {
    let ledger = try MemoryLedger()
    let old = try PinnedAttention(ledger: ledger, architecture: .medium)
    let batched = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
    for (rows, keys) in [(3, 129), (9, 4097), (24, 32768)] {
        try autoreleasepool {
            let q = (0..<(rows * 6144)).map { BF16.decode(BF16.encode(Float(($0 * 31 + 13) % 509 - 254) / 513)) }
            let k = (0..<(keys * 512)).map { BF16.encode(Float(($0 * 67 + 11) % 509 - 254) / 513) }
            let v = (0..<(keys * 512)).map { BF16.encode(Float(($0 * 113 + 19) % 1021 - 510) / 1025) }
            let split = (keys - rows) * 512
            let oldK = Array(k[..<split])
            let newK = Array(k[split...])
            let oldV = Array(v[..<split])
            let newV = Array(v[split...])
            for positions in [
                (0..<rows).map { $0 % 3 == 0 ? [] : [UInt32(0), UInt32(keys - rows + $0)] },
                [[UInt32]](repeating: [], count: rows),
            ] {
                let expected = try old.attention(
                    queries: q, oldKeys: oldK, newKeys: newK, oldValues: oldV, newValues: newV, selectedPositions: positions)
                let actual = try batched.attention(
                    queries: q, oldKeys: oldK, newKeys: newK, oldValues: oldV, newValues: newV, selectedPositions: positions)
                try batchRequire(
                    zip(expected, actual).allSatisfy { $0.isFinite && $1.isFinite && $0.bitPattern == $1.bitPattern },
                    "Batched attention changed a stripe's exact arithmetic or row/head address")
            }
        }
        try batchRequire(ledger.snapshot().reserved == 0, "Batched attention retained scratch after synchronous work")
    }
}

func pinnedAttentionBatchScratchDenialRetiresBuffers() throws {
    let q = [Float](repeating: 0, count: 3 * 6144)
    let kv = [UInt16](repeating: 0, count: 1024 * 512)
    let positions = [[UInt32]](repeating: [0], count: 3)
    let controlLedger = try MemoryLedger()
    let old = try PinnedAttention(ledger: controlLedger, architecture: .medium)
    _ = try old.attention(queries: q, oldKeys: kv, newKeys: [], oldValues: kv, newValues: [], selectedPositions: positions)
    let limited = try MemoryLedger(ceiling: controlLedger.snapshot().reservationHighWater, transientMargin: 0)
    let batched = try PinnedAttention(ledger: limited, architecture: .medium, batched: true)
    var rejected = false
    do {
        _ = try batched.attention(queries: q, oldKeys: kv, newKeys: [], oldValues: kv, newValues: [], selectedPositions: positions)
    } catch let error as EngineError { rejected = error.code == "insufficient_memory" }
    try batchRequire(
        rejected && limited.snapshot().reserved == 0,
        "Batched score scratch bypassed admission or retained earlier buffers on denial")
}
