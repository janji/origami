import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of `ragged_prefill_scope` (a read scope whose last compute tile is shorter than 256
/// rows is merged into the scope instead of running as a separate scope): the merged schedule keeps the
/// token extent, the per-tile row counts and the 256-token frontiers, gets its own checkpoint signature,
/// and its whole-prefill layout stays within the full 8192-token scope's memory; the switch requires
/// whole execution and the prefill layer command.
package enum MixedTailSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError(message) }
    }
    private static func reject(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Invalid mixed-tail input was admitted")
    }
    package static func schedulesAndIdentity() throws {
        for offset in [0, 257] {
            for tail in 1..<256 {
                let end = offset + 7424 + tail
                let original = try PrefillReadSchedule(promptTokens: end, computeTile: 256, readScope: 8192, startOffset: offset)
                let mixed = try PrefillReadSchedule(
                    promptTokens: end, computeTile: 256, readScope: 8192,
                    startOffset: offset, mergeTerminalTile: true)
                try require(
                    original.parts.count == 2 && mixed.parts.count == 1 && mixed.parts[0].range == offset..<end,
                    "Mixed scope omitted a valid tail or changed its token extent")
                let a = try original.checkpointSegments(arithmeticEpoch: "fixture")
                let b = try mixed.checkpointSegments(arithmeticEpoch: "fixture")
                try require(
                    a.map(\.count) == b.map(\.count) && b.last!.count == tail && b.last!.keyExtent == end,
                    "Mixed scope padded or changed attention projection rows")
                try require(
                    zip(a, b).allSatisfy { $0.dispatchSHA256 != $1.dispatchSHA256 },
                    "Mixed expert grouping reused an original checkpoint signature")
                let offsets = PersistentConversation.scopeOffsets(mixed.parts[0].range, spacing: 256)
                try require(
                    offsets.count == 29 && offsets.last == offset + 7424,
                    "Mixed scope lost the full-tile frontier before its tail")
            }
        }
        for count in [1, 2, 255, 256, 8192, 8193, 32256] {
            let a = try PrefillReadSchedule(promptTokens: count, computeTile: 256, readScope: 8192)
            let b = try PrefillReadSchedule(promptTokens: count, computeTile: 256, readScope: 8192, mergeTerminalTile: true)
            try require(a.parts == b.parts, "Mixed scope changed an unmergeable or aligned schedule")
        }
    }
    package static func completeMemoryAndConfiguration() throws {
        let maximum = try WholePrefillScope.layout(tokens: 8192)!
        for tokens in 1...8192 {
            if let layout = try WholePrefillScope.layout(tokens: tokens, allowTail: true) {
                try require(
                    layout.deviceBytes <= layout.originalExtra && layout.remainingHostBytes + layout.deviceBytes == layout.originalExtra
                        && layout.originalExtra <= maximum.originalExtra, "Mixed scope exceeded the original complete escrow")
                try require(zip(layout.sizes, maximum.sizes).allSatisfy { $0 <= $1 }, "Mixed scope buffer exceeds the admitted maximum")
            }
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
        let ledger = try MemoryLedger(ceiling: 3_000_000_000, transientMargin: 100_000_000)
        for tokens in [7678, 8191] {
            try autoreleasepool {
                let layout = try WholePrefillScope.layout(tokens: tokens, allowTail: true)!
                let scope = try WholePrefillScope(device: device, ledger: ledger, layout: layout)
                try require(
                    scope[.residual].length == tokens * 10240 * 4 && ledger.snapshot().reserved == layout.originalExtra,
                    "Actual mixed scope allocation lost its complete host/device ownership")
            }
            try require(ledger.snapshot().reserved == 0, "Mixed scope retained an allocation")
        }
        let base = CheckFixtures.file("config/prefill-layer-command-8192.json")
        var config = try JSONSerialization.jsonObject(with: Data(contentsOf: base)) as! [String: Any]
        var execution = config["execution"] as! [String: Any]
        execution["ragged_prefill_scope"] = true
        config["execution"] = execution
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: path) }
        try JSONSerialization.data(withJSONObject: config).write(to: path)
        try require(try RuntimeConfiguration.load(path).raggedPrefillScope, "Runtime ignored mixed scope selection")
        for (key, value) in [
            ("whole_execution_schedule", false as Any), ("expert_prefill_layer_command", false), ("ragged_prefill_scope", 1),
        ] {
            var bad = config
            var flags = execution
            flags[key] = value
            bad["execution"] = flags
            try JSONSerialization.data(withJSONObject: bad).write(to: path)
            try reject { _ = try RuntimeConfiguration.load(path) }
        }
    }
}
