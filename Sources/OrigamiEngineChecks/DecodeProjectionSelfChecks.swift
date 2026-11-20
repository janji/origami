import Foundation
import Metal
import OrigamiEngine

/// Check: the tile8 decode projection, reached through the real encoders (planned, paired and synchronous routes), is
/// bit-identical to `qmv_pinned_reference` at the production decode shapes; aliased outputs are refused, and shapes
/// outside tile8 keep their original schedule. Registered for group 64 and group 32.
package enum DecodeProjectionSelfChecks {
    package static func actualEncoderParity() throws {
        let control = try ReferenceKernels(quantizedTiled: false)
        let candidate = try ReferenceKernels(quantizedTiled: true)
        let device = candidate.device
        guard let queue = device.makeCommandQueue() else { throw EngineError("Projection check queue unavailable") }
        func need(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
        func buffer(_ bytes: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw EngineError("Projection check allocation failed")
            }
            memset(b.contents(), 0, bytes)
            return b
        }
        var seed: UInt64 = 7_619_823
        func random() -> UInt32 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt32(truncatingIfNeeded: seed >> 32)
        }
        func weights(_ width: Int, _ rows: Int) throws -> DecodeProjectionKernel.Weights {
            let groups = width * rows / ModelLayout.group
            let w = try buffer(width * rows / 2)
            let s = try buffer(groups * 2)
            let b = try buffer(groups * 2)
            let wp = w.contents().assumingMemoryBound(to: UInt32.self)
            let sp = s.contents().assumingMemoryBound(to: UInt16.self)
            let bp = b.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<width * rows / 8 { wp[i] = random() }
            for i in 0..<groups {
                sp[i] = BF16.encode(Float(Int(random() % 201) - 100) / 4096)
                bp[i] = BF16.encode(Float(Int(random() % 201) - 100) / 2048)
            }
            return .init(words: w, scales: s, biases: b, rows: rows)
        }
        func command(_ body: (MTLCommandBuffer) throws -> Void) throws {
            guard let c = queue.makeCommandBuffer() else { throw EngineError("Projection check command unavailable") }
            try body(c)
            c.commit()
            c.waitUntilCompleted()
            try need(c.status == .completed, "Projection GPU failed")
        }
        func encode(
            _ kernels: ReferenceKernels, _ c: MTLCommandBuffer, _ x: MTLBuffer, _ width: Int,
            _ w: DecodeProjectionKernel.Weights, _ output: MTLBuffer
        ) throws {
            let plan = try kernels.pinnedProjectionPlan(rows: 1, width: width, outputs: w.rows, simdMatrix: true)
            try need(
                plan.kernel == (kernels.quantizedTiled ? "projection_tile8" : "qmv_pinned_reference"), "Projection route was not exercised")
            try kernels.encodePinnedProjection(
                c, input: x, words: w.words, scales: w.scales, biases: w.biases,
                output: output, partials: nil, plan: plan, simdMatrix: true)
        }
        func equal(_ a: MTLBuffer, _ b: MTLBuffer, _ count: Int) throws {
            let ap = a.contents().assumingMemoryBound(to: Float.self)
            let bp = b.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                try need(ap[i].isFinite && bp[i].isFinite && ap[i].bitPattern == bp[i].bitPattern, "Projection bits differ at \(i)")
            }
        }
        // All GDN projections, QSA query-gate/K/V and HC down; irregular tile
        // count also checks dispatchThreads' final partially occupied group.
        for (width, n0, n1) in [
            (2560, 248320, 8), (2560, 10240, 6144), (6144, 2560, 8), (2560, 12288, 512), (10240, 320, 16), (2560, 8, 16),
        ] {
            try autoreleasepool {
                let x = try buffer(width * 2)
                let w0 = try weights(width, n0)
                let w1 = try weights(width, n1)
                let xp = x.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<width { xp[i] = BF16.encode(Float(Int(random() % 2001) - 1000) / 256) }
                let a0 = try buffer(n0 * 4)
                let a1 = try buffer(n1 * 4)
                let b0 = try buffer(n0 * 4)
                let b1 = try buffer(n1 * 4)
                try command { c in
                    try encode(control, c, x, width, w0, a0)
                    try encode(control, c, x, width, w1, a1)
                }
                try command { c in
                    try encode(candidate, c, x, width, w0, b0)
                    try encode(candidate, c, x, width, w1, b1)
                }
                try equal(a0, b0, n0)
                try equal(a1, b1, n1)
                let synchronous = try candidate.qlinearPinnedBF16(
                    x: (0..<width).map { BF16.decode(xp[$0]) },
                    width: width, outputs: n0, words: w0.words, scales: w0.scales, biases: w0.biases, simdMatrix: true)
                try need(synchronous.implementation == "pinned_qmv_tile8_shared_activation", "Synchronous tile8 route was bypassed")
                let expected = a0.contents().assumingMemoryBound(to: Float.self)
                try need(
                    synchronous.output.count == n0
                        && (0..<n0).allSatisfy {
                            synchronous.output[$0].isFinite && synchronous.output[$0].bitPattern == expected[$0].bitPattern
                        }, "Synchronous full-size tile8 output differs")
                try command { c in
                    try need(
                        try candidate.encodePairedDecodeProjection(
                            c, input: x, width: width, first: w0,
                            firstOutput: b0, second: w1, secondOutput: b1), "Paired candidate did not dispatch")
                }
                try equal(a0, b0, n0)
                try equal(a1, b1, n1)
                guard let unsubmitted = queue.makeCommandBuffer() else { throw EngineError("Projection guard command unavailable") }
                var refused = false
                do {
                    _ = try candidate.encodePairedDecodeProjection(
                        unsubmitted, input: x, width: width, first: w0,
                        firstOutput: b0, second: w1, secondOutput: b0)
                } catch { refused = true }
                try need(refused && unsubmitted.status == .notEnqueued, "Paired alias was accepted")
                try need(
                    try !control.encodePairedDecodeProjection(
                        unsubmitted, input: x, width: width, first: w0,
                        firstOutput: b0, second: w1, secondOutput: b1), "Reference unexpectedly enabled tile8")
            }
        }
        // These shapes keep their schedule. Two rows do take tile8 (the MTP verify step; see two_row_projection_matches_one_row).
        for (rows, width, outputs) in [(3, 2560, 640), (1, 640, 2560), (1, 320, 10240), (1, 128, 8)] {
            try need(
                try candidate.pinnedProjectionPlan(rows: rows, width: width, outputs: outputs, simdMatrix: true).kernel
                    != "projection_tile8",
                "Unsupported geometry did not retain original schedule")
        }
    }
}
