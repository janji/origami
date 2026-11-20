import Foundation

/// The QMV/QMM choice, split-K count and partition size of pinned MLX core 1f8e74e3 (backend/metal/quantized.cpp:85,
/// 1120, 1735, as MLX dispatches on an M2-class GPU) for a BF16 affine-4 projection. It fixes the rounding order, not
/// speed, so every Mac uses it unchanged; `groupedExpert` selects the no-split grouped-expert QMM.
public struct QuantizedSchedule: Codable, Equatable, Sendable {
    public let kind: String
    public let vectorLimit, splitK, partitionSize, valuesPerLane, laneCount: Int

    public init(rows: Int, width: Int, outputs: Int, groupSize: Int = ModelLayout.group, groupedExpert: Bool = false) throws {
        guard rows > 0, rows <= 4096, width > 0, width <= 10_240, outputs > 0, outputs <= 248_320,
            [32, 64].contains(groupSize), width % groupSize == 0
        else {
            throw EngineError("Unsupported pinned BF16 affine4 geometry")
        }
        vectorLimit = width <= 2048 && outputs <= 2048 ? 14 : (width <= 4096 && outputs <= 4096 ? 10 : 6)
        if groupedExpert {
            // Pinned GatherQMM sorted-rhs dispatch (quantized.cpp:1905) uses
            // BM16/BN32/BK32 and one accumulator across K. Dense QLinear's
            // split-K policy and QMV threshold must never enter this path.
            kind = "qmm_grouped_expert_no_split"
            splitK = 1
            partitionSize = width
            valuesPerLane = 0
            laneCount = 1
            return
        }
        if rows < vectorLimit {
            let quad = width == 64 || width == 128
            let fast = !quad && outputs % 8 == 0 && width % 512 == 0
            kind = quad ? "qmv_quad" : (fast ? "qmv_fast" : "qmv")
            splitK = 1
            partitionSize = width
            valuesPerLane = quad ? width / 4 : (fast ? 16 : 8)
            laneCount = quad ? 4 : 32
        } else {
            let tiles = ((rows + 31) / 32) * ((outputs + 31) / 32)
            var count = min(max(1, 512 / tiles), width / max(32, groupSize))
            while count > 1 && width % (count * max(32, groupSize)) != 0 { count -= 1 }
            guard count <= 256 else { throw EngineError("Pinned two-pass split-K reducer is not implemented") }
            kind = count == 1 ? "qmm" : "qmm_splitk"
            splitK = count
            partitionSize = width / count
            valuesPerLane = 0
            laneCount = count < 32 ? min(8, count) : 32
        }
    }
}
