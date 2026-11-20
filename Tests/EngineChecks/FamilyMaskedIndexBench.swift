import Foundation
import Metal
import OrigamiEngine

/// Isolated O05 timings at populated context shapes. Synthetic keys and queries, real engine kernels;
/// no model weights, expert I/O, or end-to-end speed claim. Alternating order reduces clock/order bias.
func familyMaskedIndexBenchmark() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("No Metal device")
    }
    let kernels = try FamilyKernels(device: device)
    typealias Ref = FamilyKernels.Ref
    func buffer(_ count: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: count * 4, options: .storageModeShared) else { throw EngineError("Allocation failed") }
        return b
    }
    func fill(_ b: MTLBuffer, salt: UInt32) {
        let p = b.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<b.length / 4 {
            let bits = (UInt32(truncatingIfNeeded: i) &* 1_664_525 &+ salt) ^ (UInt32(truncatingIfNeeded: i) >> 3)
            p[i] = Float(Int32(bitPattern: bits) % 8191) / 8192
        }
    }
    func bytes(_ b: MTLBuffer) -> Data { Data(bytes: b.contents(), count: b.length) }
    func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
    var results: [[String: Any]] = []
    for entries in [16384, 66108, 131072] {
        try autoreleasepool {
            let heads = 32
            let dim = 128
            let blocks = (entries + 7) / 8
            let selectedBlocks = min(2048, blocks)
            let q = try buffer(heads * dim)
            let keys = try buffer(entries * dim)
            let w = try buffer(heads)
            let keep = try buffer(blocks)
            let full = try buffer(entries)
            let masked = try buffer(entries)
            let fullIDs = try buffer(512)
            let maskedIDs = try buffer(512)
            fill(q, salt: 1_013_904_223)
            fill(keys, salt: 1_664_525)
            fill(w, salt: 12345)
            for pattern in ["contiguous", "spread"] {
                let mask = keep.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<blocks { mask[i] = 0 }
                for i in 0..<selectedBlocks {
                    // Uniformly spread whole eight-entry blocks; no duplicate IDs.
                    mask[pattern == "contiguous" ? i : i * blocks / selectedBlocks] = 1
                }
                for selection in [false, true] {
                    func run(_ optimized: Bool) throws -> (gpu: Double, wall: Double) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        guard let command = queue.makeCommandBuffer(), let e = command.makeComputeCommandEncoder() else {
                            throw EngineError("No Metal command")
                        }
                        do {
                            let scores = optimized ? masked : full
                            var refs = [Ref(q), Ref(keys), Ref(w), Ref(scores)]
                            var ints = [Int32(heads), Int32(dim), Int32(entries), 1, Int32(entries - 1)]
                            if optimized {
                                refs.append(Ref(keep))
                                ints += [8, Int32(blocks)]
                            }
                            try kernels.dispatch(
                                e, optimized ? "index_scores_masked" : "index_scores", refs, ints: ints,
                                grid: MTLSize(width: (entries + 63) / 64, height: 1, depth: 1),
                                group: MTLSize(width: 64, height: 1, depth: 1))
                            if !optimized {
                                try kernels.dispatch(
                                    e, "apply_block_mask", [Ref(scores), Ref(keep)],
                                    ints: [Int32(entries), 8, Int32(blocks), Int32(blocks)],
                                    grid: MTLSize(width: (entries + 255) / 256, height: 1, depth: 1),
                                    group: MTLSize(width: 256, height: 1, depth: 1))
                            }
                            if selection {
                                try kernels.dispatch(
                                    e, "index_select", [Ref(scores), Ref(optimized ? maskedIDs : fullIDs)],
                                    ints: [Int32(entries), 1, 512, 512, Int32(entries - 1), 0, 1, 128, 512],
                                    grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
                            }
                        } catch {
                            e.endEncoding()
                            throw error
                        }
                        e.endEncoding()
                        command.commit()
                        command.waitUntilCompleted()
                        guard command.status == .completed else {
                            throw EngineError("Metal command failed: \(String(describing: command.error))")
                        }
                        return (command.gpuEndTime - command.gpuStartTime, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
                    }
                    for _ in 0..<2 {
                        _ = try run(false)
                        _ = try run(true)
                    }
                    var fullGPU: [Double] = []
                    var maskedGPU: [Double] = []
                    var fullWall: [Double] = []
                    var maskedWall: [Double] = []
                    for repeatIndex in 0..<9 {
                        for optimized in repeatIndex % 2 == 0 ? [false, true] : [true, false] {
                            let r = try run(optimized)
                            if optimized {
                                maskedGPU.append(r.gpu)
                                maskedWall.append(r.wall)
                            } else {
                                fullGPU.append(r.gpu)
                                fullWall.append(r.wall)
                            }
                        }
                        guard bytes(full) == bytes(masked), !selection || bytes(fullIDs) == bytes(maskedIDs) else {
                            throw EngineError("Score/selection mismatch at \(entries), \(pattern)")
                        }
                    }
                    results.append([
                        "entries": entries, "mask": pattern, "kept_blocks": selectedBlocks,
                        "includes_selection": selection, "scores_bitwise_equal": true, "selected_ids_equal": selection,
                        "full_gpu_seconds": fullGPU, "masked_gpu_seconds": maskedGPU,
                        "full_wall_seconds": fullWall, "masked_wall_seconds": maskedWall,
                        "median_full_gpu_seconds": median(fullGPU), "median_masked_gpu_seconds": median(maskedGPU),
                        "median_full_wall_seconds": median(fullWall), "median_masked_wall_seconds": median(maskedWall),
                    ])
                }
            }
        }
    }
    let result: [String: Any] = [
        "scope": "Synthetic populated buffers, real engine kernels, one later index source; no model inference or disk traffic.",
        "device": device.name, "repeats": 9, "results": results,
    ]
    print(String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self))
}
