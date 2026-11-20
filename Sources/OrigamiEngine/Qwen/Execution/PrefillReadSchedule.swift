import CryptoKit
import Foundation

/// Splits a prompt, or the suffix after `startOffset`, into read scopes of at most `readScope` tokens, each computed
/// in `computeTile`-row tiles counted from the scope start; a remainder shorter than one tile becomes its own part
/// with that smaller tile (the cached-QMV tail). With `mergeTerminalTile` (ragged prefill scopes, C256 only) the
/// remainder joins the last scope when the merged scope still has a whole-prefill layout: attention keeps 256-row
/// tiles but expert grouping spans the whole scope, a different arithmetic, so the producing read scope is bound
/// into every checkpoint signature.
package struct PrefillReadSchedule: Sendable {
    /// One read scope: its absolute token range and the arithmetic tile rows inside it.
    package struct Part: Sendable, Equatable {
        package let range: Range<Int>
        package let computeTile: Int
    }
    package let parts: [Part]
    package init(promptTokens: Int, computeTile: Int, readScope: Int, startOffset: Int = 0, mergeTerminalTile: Bool = false) throws {
        guard promptTokens > 0, promptTokens <= ModelContract.maximumContext, startOffset >= 0, startOffset < promptTokens,
            (1...256).contains(computeTile), readScope >= computeTile,
            readScope <= 8192, readScope % computeTile == 0
        else {
            throw EngineError(
                "Invalid prefill read/compute schedule: prompt \(promptTokens), tile \(computeTile), scope \(readScope), offset \(startOffset)"
            )
        }
        var result: [Part] = []
        var start = startOffset
        let fullEnd = startOffset + (promptTokens - startOffset) / computeTile * computeTile
        while start < fullEnd {
            let end = min(fullEnd, start + readScope)
            result.append(Part(range: start..<end, computeTile: computeTile))
            start = end
        }
        if start < promptTokens {
            if mergeTerminalTile, computeTile == 256, let last = result.last,
                promptTokens - last.range.lowerBound <= readScope,
                try WholePrefillScope.layout(tokens: promptTokens - last.range.lowerBound, allowTail: true) != nil
            {
                result[result.count - 1] = Part(range: last.range.lowerBound..<promptTokens, computeTile: computeTile)
            } else {
                result.append(Part(range: start..<promptTokens, computeTile: promptTokens - start))
            }
        }
        parts = result
    }

    /// Binds both compute segmentation and its enclosing committed read scope.
    /// Even when C is unchanged, a different R may change grouped MoE projection
    /// dispatches; that prefix must not be claimed byte-identical to a fresh run.
    package func checkpointSegments(arithmeticEpoch: String) throws -> [CheckpointSegment] {
        guard !arithmeticEpoch.isEmpty, arithmeticEpoch.utf8.count <= 256 else {
            throw EngineError("Checkpoint schedule requires a bounded arithmetic epoch")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var result: [CheckpointSegment] = []
        for part in parts {
            for start in stride(from: part.range.lowerBound, to: part.range.upperBound, by: part.computeTile) {
                let rows = min(part.computeTile, part.range.upperBound - start)
                let signature = CheckpointDispatchSignature(
                    version: 1, arithmeticEpoch: arithmeticEpoch,
                    phase: "prefill", readScopeStart: part.range.lowerBound, readScopeCount: part.range.count,
                    computeTile: rows, queryStart: start, queryCount: rows,
                    absoluteKeyExtent: start + rows, projectionRows: rows,
                    mask: "absolute_causal_v1")
                let digest = SHA256.hash(data: try encoder.encode(signature)).map { String(format: "%02x", $0) }.joined()
                result.append(
                    try CheckpointSegment(
                        start: start, count: rows,
                        projectionRows: rows, queryTile: rows, keyExtent: start + rows,
                        phase: .prefill, dispatchSHA256: digest))
            }
        }
        return result
    }
    package static func decodeSegment(start: Int, arithmeticEpoch: String) throws -> CheckpointSegment {
        let signature = CheckpointDispatchSignature(
            version: 1, arithmeticEpoch: arithmeticEpoch,
            phase: "decode", readScopeStart: start, readScopeCount: 1, computeTile: 1,
            queryStart: start, queryCount: 1, absoluteKeyExtent: start + 1, projectionRows: 1, mask: "absolute_causal_v1")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let digest = SHA256.hash(data: try encoder.encode(signature)).map { String(format: "%02x", $0) }.joined()
        return try CheckpointSegment(
            start: start, count: 1, projectionRows: 1, queryTile: 1,
            keyExtent: start + 1, phase: .decode, dispatchSHA256: digest)
    }
}

/// Hashed as sorted-key JSON into each segment's dispatchSHA256: a saved frontier is reused only by a segment that
/// was produced by the same dispatch shape and arithmetic epoch.
private struct CheckpointDispatchSignature: Encodable {
    let version: Int
    let arithmeticEpoch, phase: String
    let readScopeStart, readScopeCount, computeTile, queryStart, queryCount, absoluteKeyExtent, projectionRows: Int
    let mask: String
}
