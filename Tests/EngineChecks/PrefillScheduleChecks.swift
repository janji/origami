import Foundation
import OrigamiEngine

func prefillReadScopesPreserveArithmeticTilesAndTails() throws {
    func require(_ condition: Bool) throws {
        if !condition { throw EngineError("Read scope changed an arithmetic tile boundary") }
    }
    for count in [1, 24, 255, 256, 257, 511, 8192, 8193, 32_256] {
        for tile in [1, 24, 128, 256] {
            let baseline = try PrefillReadSchedule(promptTokens: count, computeTile: tile, readScope: tile)
            for multiple in [1, 2, 4, 16, 32] where tile * multiple <= 8192 {
                let grouped = try PrefillReadSchedule(promptTokens: count, computeTile: tile, readScope: tile * multiple)
                let rebuilt = grouped.parts.flatMap { part in
                    stride(from: part.range.lowerBound, to: part.range.upperBound, by: part.computeTile).map {
                        $0..<($0 + part.computeTile)
                    }
                }
                try require(rebuilt == baseline.parts.map(\.range))
                try require(grouped.parts.allSatisfy { $0.range.count <= tile * multiple })
            }
        }
    }
    for (count, tile, scope) in [(0, 256, 256), (257, 256, 257), (12, 0, 256), (12, 257, 257), (12, 256, 16384)] {
        do { _ = try PrefillReadSchedule(promptTokens: count, computeTile: tile, readScope: scope) } catch is EngineError { continue }
        throw EngineError("Invalid read schedule was admitted")
    }
    // A 7,678-token prompt is shorter than R8192. Its full arithmetic tiles
    // form one partial final read scope, followed by a separate C254 tail.
    let partialControl = try PrefillReadSchedule(promptTokens: 7678, computeTile: 256, readScope: 256)
    let partialCandidate = try PrefillReadSchedule(promptTokens: 7678, computeTile: 256, readScope: 8192)
    guard partialControl.parts.count == 30, partialControl.parts != partialCandidate.parts,
        partialCandidate.parts.map(\.range) == [0..<7424, 7424..<7678],
        partialCandidate.parts.map(\.computeTile) == [256, 254],
        partialCandidate.parts.flatMap({ part in
            stride(from: part.range.lowerBound, to: part.range.upperBound, by: part.computeTile).map {
                $0..<($0 + part.computeTile)
            }
        }) == partialControl.parts.map(\.range)
    else {
        throw EngineError("Partial final R did not preserve distinct scopes and identical arithmetic tiles")
    }
    // Prompts with at most one full tile give identical plans for R256 and R8192,
    // so a comparison of the two schedules is vacuous for them.
    for count in [1, 24, 256, 257, 511] {
        let control = try PrefillReadSchedule(promptTokens: count, computeTile: 256, readScope: 256)
        let candidate = try PrefillReadSchedule(promptTokens: count, computeTile: 256, readScope: 8192)
        guard control.parts == candidate.parts else {
            throw EngineError("Short scope comparison unexpectedly became nonvacuous")
        }
    }
    try prefillCheckpointMetadataBindsScopeComputeAndEpoch()
}

private func prefillCheckpointMetadataBindsScopeComputeAndEpoch() throws {
    func segments(_ count: Int, _ tile: Int, _ scope: Int, epoch: String = "fixture-v1") throws -> [CheckpointSegment] {
        try PrefillReadSchedule(promptTokens: count, computeTile: tile, readScope: scope).checkpointSegments(arithmeticEpoch: epoch)
    }
    let ordinary = try segments(512, 256, 256)
    let grouped = try segments(512, 256, 512)
    guard ordinary.map(\.count) == [256, 256], grouped.map(\.count) == [256, 256],
        grouped.map(\.keyExtent) == [256, 512],
        ordinary[0].dispatchSHA256 != grouped[0].dispatchSHA256,
        grouped != (try segments(512, 128, 512)),
        grouped != (try segments(512, 256, 512, epoch: "fixture-v2"))
    else {
        throw EngineError("Checkpoint signature omitted read scope, compute rows, keys or arithmetic epoch")
    }
    let extensionPlan = try segments(768, 256, 512)
    guard Array(extensionPlan.prefix(2)) == grouped else { throw EngineError("Identical committed read scope lost prefix compatibility") }
    let shortenedScope = try segments(256, 256, 512)
    guard shortenedScope[0].dispatchSHA256 != grouped[0].dispatchSHA256 else {
        throw EngineError("Interior compute boundary was treated as a complete enclosing read scope")
    }
    let shortTail = try segments(300, 256, 256)
    let longer = try segments(512, 256, 256)
    guard shortTail.map(\.count) == [256, 44], shortTail[0] == longer[0], shortTail[1] != longer[1] else {
        throw EngineError("Checkpoint short terminal tile lost its producing shape")
    }
}
