import Darwin
import Foundation
import Metal
import ObjectiveC
import OrigamiEngine

/// `engine-checks --bench-qwen <prefill|decode> [tokens]`: model-free timings of Qwen's GPU work at the real shapes, with
/// random weights, through the engine's own owners and kernel choices (the release plan's switches). Prefill: one layer
/// of a scope of `tokens` (default 2526): per 256-row tile the two HCs, the GDN block, the router and the shared expert;
/// per layer the expert layer command (`evaluateDeviceLayer`, uniform random top-10 routing) and the ten-rank reduction.
/// Decode: one layer's row work (three HC rows, GDN, router twice, shared expert, ten experts), a 48-layer token of it
/// with the LM head, and the host readout. Each component reports GPU seconds (command-buffer GPU start to end), host
/// encode seconds, compute dispatches, compute encoders and command buffers per instance. Not covered: sparse
/// attention (QSA, 12 of 48 layers), small elementwise kernels (narrowing, injection, mix), PLE, reads, event waits.
package enum QwenBench {
    // Dispatch and encoder counters: the encoder classes' methods are swapped for counting ones once.
    nonisolated(unsafe) static var dispatches = 0
    nonisolated(unsafe) static var encoders = 0
    nonisolated(unsafe) static var installed = false
    static func install(_ device: MTLDevice) {
        guard !installed, let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer() else { return }
        installed = true
        var classes: [AnyClass] = []
        if let e = command.makeComputeCommandEncoder() {
            classes.append(object_getClass(e)!)
            e.endEncoding()
        }
        if let e = command.makeComputeCommandEncoder(dispatchType: .concurrent) {
            let c: AnyClass = object_getClass(e)!
            if !classes.contains(where: { $0 == c }) { classes.append(c) }
            e.endEncoding()
        }
        for cls in classes {
            typealias Sizes = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
            for name in ["dispatchThreadgroups:threadsPerThreadgroup:", "dispatchThreads:threadsPerThreadgroup:"] {
                let selector = NSSelectorFromString(name)
                guard let method = class_getInstanceMethod(cls, selector) else { continue }
                let original = unsafeBitCast(method_getImplementation(method), to: Sizes.self)
                let block: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = { object, a, b in
                    dispatches += 1
                    original(object, selector, a, b)
                }
                method_setImplementation(method, imp_implementationWithBlock(block))
            }
            typealias Indirect = @convention(c) (AnyObject, Selector, AnyObject, Int, MTLSize) -> Void
            let indirect = NSSelectorFromString("dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:")
            if let method = class_getInstanceMethod(cls, indirect) {
                let original = unsafeBitCast(method_getImplementation(method), to: Indirect.self)
                let block: @convention(block) (AnyObject, AnyObject, Int, MTLSize) -> Void = { object, buffer, offset, size in
                    dispatches += 1
                    original(object, indirect, buffer, offset, size)
                }
                method_setImplementation(method, imp_implementationWithBlock(block))
            }
            typealias End = @convention(c) (AnyObject, Selector) -> Void
            let end = NSSelectorFromString("endEncoding")
            if let method = class_getInstanceMethod(cls, end) {
                let original = unsafeBitCast(method_getImplementation(method), to: End.self)
                let block: @convention(block) (AnyObject) -> Void = { object in
                    encoders += 1
                    original(object, end)
                }
                method_setImplementation(method, imp_implementationWithBlock(block))
            }
        }
    }

    struct Row {
        let name: String
        let perLayer: Double
        var gpu = 0.0, host = 0.0, dispatches = 0.0, encoders = 0.0, commands = 0.0
    }
    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
    static func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

    /// Time `body` encoded `copies` times into one command buffer, `runs` times; per-instance medians.
    static func measure(
        _ name: String, perLayer: Double, queue: MTLCommandQueue, copies: Int = 8, runs: Int = 5,
        _ body: (MTLCommandBuffer, Int) throws -> Void
    ) throws -> Row {
        var gpu: [Double] = []
        var host: [Double] = []
        var row = Row(name: name, perLayer: perLayer)
        for run in 0...runs {
            guard let command = queue.makeCommandBuffer() else { throw EngineError("Bench command unavailable") }
            let (d0, e0) = (dispatches, encoders)
            let start = now()
            for copy in 0..<copies { try body(command, copy) }
            let encoded = now() - start
            row.dispatches = Double(dispatches - d0) / Double(copies)
            row.encoders = Double(encoders - e0) / Double(copies)
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw EngineError.gpu("Bench command failed", command.error) }
            if run > 0 {  // the first run warms pipelines and pages
                gpu.append((command.gpuEndTime - command.gpuStartTime) / Double(copies))
                host.append(encoded / Double(copies))
            }
        }
        row.gpu = median(gpu)
        row.host = median(host)
        row.commands = 1 / Double(copies)
        return row
    }

    static func print(_ rows: [Row], layers: (Row) -> Double, title: String) {
        Swift.print(title)
        Swift.print("component                 per-inst GPU ms  host ms  disp  enc   count/tok-or-prompt  GPU s total  host s total")
        var gpuTotal = 0.0
        var hostTotal = 0.0
        for r in rows {
            let n = layers(r)
            gpuTotal += r.gpu * n
            hostTotal += r.host * n
            Swift.print(
                String(
                    format: "%@ %12.4f %8.4f %5.0f %4.0f %14.0f %16.3f %12.3f",
                    r.name.padding(toLength: 25, withPad: " ", startingAt: 0) as NSString, r.gpu * 1e3, r.host * 1e3,
                    r.dispatches, r.encoders, n, r.gpu * n, r.host * n))
        }
        Swift.print(String(format: "sum of covered components: GPU %.3f s, host encode %.3f s", gpuTotal, hostTotal))
    }

    // Random-ish affine 4-bit projections and buffers.
    static func filled(_ device: MTLDevice, _ bytes: Int, seed: UInt64) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw EngineError("Bench buffer unavailable", code: "insufficient_memory")
        }
        let p = b.contents().assumingMemoryBound(to: UInt64.self)
        var x = seed &* 0x9e37_79b9_7f4a_7c15 | 1
        for i in 0..<(b.length / 8) {
            x ^= x << 13
            x ^= x >> 7
            x ^= x << 17
            p[i] = x
        }
        return b
    }
    static func constant(_ device: MTLDevice, count: Int, _ value: (Int) -> UInt16) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(count * 2, 16), options: .storageModeShared) else {
            throw EngineError("Bench buffer unavailable", code: "insufficient_memory")
        }
        let p = b.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { p[i] = value(i) }
        return b
    }
    static func floats(_ device: MTLDevice, count: Int, _ value: (Int) -> Float) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(count * 4, 16), options: .storageModeShared) else {
            throw EngineError("Bench buffer unavailable", code: "insufficient_memory")
        }
        let p = b.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { p[i] = value(i) }
        return b
    }
    static func zeroed(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw EngineError("Bench buffer unavailable", code: "insufficient_memory")
        }
        memset(b.contents(), 0, b.length)
        return b
    }
    static func projection(_ device: MTLDevice, width: Int, outputs: Int, seed: UInt64) throws -> GDNDeviceBlock.Projection {
        let groups = outputs * width / ModelLayout.group
        return try .init(
            words: filled(device, outputs * width / 2, seed: seed),
            scales: constant(device, count: groups) { BF16.encode(Float($0 % 5 + 1) / 2048) },
            biases: constant(device, count: groups) { BF16.encode(-Float($0 % 5 + 1) * 7.5 / 2048) })
    }

    struct Fixture {
        let device: MTLDevice, queue: MTLCommandQueue, ledger: MemoryLedger
        let kernels: ReferenceKernels, dense: PinnedDenseLinear, reductions: PinnedReductions
        let selector: BatchedTopIndices, gdn: GDNDeviceBlock, hc: FusedHCDecode, scalars: WholeExecutionScalars
        let fused: FusedExpertExecution
    }
    static func fixture(mmaMinRows: Int = 192) throws -> Fixture {
        let ledger = try MemoryLedger(ceiling: 24_000_000_000, transientMargin: 1_000_000)
        // The release plan's kernel switches.
        let kernels = try ReferenceKernels(
            pinnedUnary: true, ledger: ledger, pinnedRMS: true, quantizedTiled: true, quantizedTiledBM32: true,
            quantizedBM32MinRows: 256, quantizedTiledPrefetch: true, quantizedMMABF16: true, gdnExactKernels: true,
            hcExactFusion: true)
        let device = kernels.device
        install(device)
        guard let queue = device.makeCommandQueue() else { throw EngineError("Bench queue unavailable") }
        let dense = try PinnedDenseLinear(ledger: ledger)
        let reductions = try PinnedReductions(ledger: ledger)
        let preparation = try FusedGDNPreparation(
            arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger, minimumRows: 8, exactKernels: true)
        let gdn = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: preparation, ledger: ledger, simdMatrix: true)
        let hc = try FusedHCDecode(
            arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true, decodeFusion: true)
        let scalars = try WholeExecutionScalars(
            device: device, sigmoidTable: zeroed(device, 65536 * 2), siluTable: zeroed(device, 65536 * 2))
        let fused = try FusedExpertExecution(
            arithmetic: kernels.arithmetic, ledger: ledger, groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 32,
            scratchReuse: true, groupedPrefetch: true, groupedMMAMinRows: mmaMinRows)
        return Fixture(
            device: device, queue: queue, ledger: ledger, kernels: kernels, dense: dense, reductions: reductions,
            selector: try BatchedTopIndices(ledger: ledger), gdn: gdn, hc: hc, scalars: scalars, fused: fused)
    }

    /// Per-layer weights of the dense row/tile work (GDN, HCs, router, shared expert).
    struct LayerWeights {
        let qkv, z, out, sharedGate, sharedUp, sharedDown: GDNDeviceBlock.Projection
        let b, a, router: MTLBuffer
        let hcDown, hcUp: FusedHCDecode.Projection
        let hcNorm, hcInjection: MTLBuffer
    }
    static func layerWeights(_ f: Fixture, seed: UInt64) throws -> LayerWeights {
        let d = f.device
        let hcDown = try projection(d, width: 10240, outputs: 320, seed: seed + 7)
        let hcUp = try projection(d, width: 320, outputs: 10240, seed: seed + 8)
        return try LayerWeights(
            qkv: projection(d, width: 2560, outputs: 10240, seed: seed + 1), z: projection(d, width: 2560, outputs: 6144, seed: seed + 2),
            out: projection(d, width: 6144, outputs: 2560, seed: seed + 3),
            sharedGate: projection(d, width: 2560, outputs: 640, seed: seed + 4),
            sharedUp: projection(d, width: 2560, outputs: 640, seed: seed + 5),
            sharedDown: projection(d, width: 640, outputs: 2560, seed: seed + 6),
            b: floats(d, count: 48 * 2560) { Float(($0 * 31) % 127 - 63) / 4096 },
            a: floats(d, count: 48 * 2560) { Float(($0 * 73) % 109 - 54) / 4096 },
            router: floats(d, count: 2560 * 512) { Float(($0 * 37 + Int(seed)) % 211 - 105) / 8192 },
            hcDown: .init(words: hcDown.words, scales: hcDown.scales, biases: hcDown.biases),
            hcUp: .init(words: hcUp.words, scales: hcUp.scales, biases: hcUp.biases),
            hcNorm: constant(d, count: 10240) { BF16.encode(Float($0 % 7 + 1) / 8) },
            hcInjection: floats(d, count: 4 * 10240) { Float($0 % 9 - 4) / 64 })
    }

    /// One row/tile's scratch for the dense work.
    struct Scratch {
        let rows: Int
        let residual, residualBF16, input, inputBF16: MTLBuffer
        let gdn: GDNDeviceBlock.RowBuffers
        let gdnPartials: GDNDeviceBlock.TilePartials?
        let recurrence, initial: MTLBuffer
        let hc: FusedHCDecode.RowBuffers
        let hcPartials: (down: MTLBuffer?, up: MTLBuffer?, injection: MTLBuffer?)
        let logits, routerPartial, pairs, selected, routerWeights, routerFlags: MTLBuffer
        let sharedGate, sharedUp, sharedProduct, sharedOut, sharedFlags: MTLBuffer
        let sharedPartials: [MTLBuffer?]
        let mixedBF16: MTLBuffer
    }
    static func scratch(_ f: Fixture, rows: Int) throws -> Scratch {
        let d = f.device
        func z(_ bytes: Int) throws -> MTLBuffer { try zeroed(d, bytes) }
        func opt(_ bytes: Int) throws -> MTLBuffer? { bytes == 0 ? nil : try zeroed(d, bytes) }
        let input = try floats(d, count: rows * 10240) { Float(($0 * 73) % 1009 - 504) / 512 }
        let x = try floats(d, count: rows * 2560) { Float(($0 * 71) % 1013 - 506) / 512 }
        let p = rows > 1 ? try f.gdn.tilePartialBytes(rows: rows) : (qkv: 0, z: 0, dense: 0, output: 0)
        let h = rows > 1 ? try f.hc.tilePartialBytes(rows: rows) : (down: 0, up: 0, injection: 0)
        let routerPlan = try f.dense.executionPlan(rows: rows, width: 2560, outputs: 512)
        let routerPartial = routerPlan.kind == .splitK ? rows * 512 * routerPlan.splitKPartitions * 4 : 0
        let shared = try [(2560, 640), (2560, 640), (640, 2560)].map {
            try f.kernels.pinnedProjectionPlan(rows: rows, width: $0.0, outputs: $0.1, simdMatrix: true).partialBytes
        }
        return try Scratch(
            rows: rows, residual: input, residualBF16: constant(d, count: rows * 10240) { BF16.encode(Float($0 % 977) / 977) },
            input: x, inputBF16: constant(d, count: rows * 2560) { BF16.encode(Float($0 % 991 - 495) / 512) },
            gdn: .init(
                mixed: z(rows * 10240 * 4), z: z(rows * 6144 * 4), b: z(rows * 48 * 4), a: z(rows * 48 * 4), history: z(3 * 10240 * 2),
                convolution: floats(d, count: 40960) { Float(($0 * 17 + 7) % 251 - 125) / 128 },
                bias: floats(d, count: 48) { Float($0 % 5 - 2) / 16 }, logs: floats(d, count: 48) { Float($0 % 7 - 3) / 16 },
                prepared: z(rows * 10240 * 4), beta: z(rows * 48 * 4), decay: z(rows * 48 * 4), prepareFlags: z(4),
                mask: constant(d, count: rows * 2) { $0 % 2 == 0 ? 1 : 0 }, normWeights: floats(d, count: 128) { Float($0 % 7 + 1) / 8 },
                normed: z(rows * 6144 * 4), gated: z(rows * 6144 * 4), gateFlags: z(4), narrowed: z(rows * 6144 * 2),
                output: z(rows * 2560 * 4), convolved: nil),
            gdnPartials: rows > 1
                ? .init(qkv: opt(p.qkv), z: opt(p.z), b: opt(p.dense), a: opt(p.dense), output: opt(p.output)) : nil,
            recurrence: z((rows * 6144 + 48 * 128 * 128) * 4), initial: z(48 * 128 * 128 * 4),
            hc: .init(
                norm: z(rows * 10240 * 4), normBF16: z(rows * 10240 * 2), down: z(rows * 320 * 4), activation: z(rows * 320 * 2),
                rawUp: z(rows * 10240 * 4), mixed: z(rows * 2560 * 4), exceptions: z(rows * 80 * 4), rawInjection: z(rows * 16),
                injection: z(rows * 16), capturedUp: nil),
            hcPartials: (opt(h.down), opt(h.up), opt(h.injection)),
            logits: z(rows * 512 * 4), routerPartial: z(routerPartial), pairs: z(rows * 10 * 8 + 4096), selected: z(rows * 40),
            routerWeights: z(rows * 40), routerFlags: z(4),
            sharedGate: z(rows * 640 * 4), sharedUp: z(rows * 640 * 4), sharedProduct: z(rows * 640 * 2), sharedOut: z(rows * 2560 * 4),
            sharedFlags: z(4), sharedPartials: shared.map { try opt($0) }, mixedBF16: z(rows * 2560 * 2))
    }

    static func encodeHC(_ f: Fixture, _ c: MTLCommandBuffer, _ w: LayerWeights, _ s: Scratch) throws {
        if s.rows == 1 {
            try f.hc.encodeRow(
                c, residualBF16: s.residualBF16, normWeights: w.hcNorm, downWeights: w.hcDown, upWeights: w.hcUp,
                injectionWeights: w.hcInjection, buffers: s.hc, residualFP32: s.residual, mixedBF16: s.mixedBF16)
        } else {
            try f.hc.encodeTile(
                c, rows: s.rows, residualBF16: s.residualBF16, normWeights: w.hcNorm, downWeights: w.hcDown, upWeights: w.hcUp,
                injectionWeights: w.hcInjection, buffers: s.hc, downPartials: s.hcPartials.down, upPartials: s.hcPartials.up,
                injectionPartials: s.hcPartials.injection, residualFP32: s.residual)
        }
    }
    static func encodeGDN(_ f: Fixture, _ c: MTLCommandBuffer, _ w: LayerWeights, _ s: Scratch, next: GDNRecurrentStorage?) throws {
        if s.rows == 1 {
            try f.gdn.encodeRow(
                c, inputFP32: s.input, inputBF16: s.inputBF16, qkv: w.qkv, z: w.z, b: w.b, a: w.a, out: w.out,
                initial: s.recurrence, initialOffset: 0, next: next!, buffers: s.gdn)
        } else {
            try f.gdn.encodeTile(
                c, rows: s.rows, inputFP32: s.input, inputBF16: s.inputBF16, qkv: w.qkv, z: w.z, b: w.b, a: w.a, out: w.out,
                initial: s.initial, initialOffset: 0, recurrence: s.recurrence, buffers: s.gdn,
                partials: s.gdnPartials!)
        }
    }
    static func encodeRouter(_ f: Fixture, _ c: MTLCommandBuffer, _ w: LayerWeights, _ s: Scratch) throws {
        try WholeLayerExecution.encodeRouterSelection(
            f.dense, f.selector, c, mixed: s.input, router: w.router, logits: s.logits,
            partial: s.routerPartial.length > 16 ? s.routerPartial : nil, pairs: s.pairs, rows: s.rows)
        try f.scalars.encodeRouterGather(
            c, logits: s.logits, pairs: s.pairs, selected: s.selected, exceptions: s.routerFlags, rows: s.rows)
        try f.reductions.encodeRouterRows(c, input: s.selected, output: s.routerWeights, rows: s.rows)
    }
    static func encodeShared(_ f: Fixture, _ c: MTLCommandBuffer, _ w: LayerWeights, _ s: Scratch) throws {
        func project(_ p: GDNDeviceBlock.Projection, _ input: MTLBuffer, _ output: MTLBuffer, _ partial: MTLBuffer?, _ width: Int, _ n: Int)
            throws
        {
            let plan = try f.kernels.pinnedProjectionPlan(rows: s.rows, width: width, outputs: n, simdMatrix: true)
            try f.kernels.encodePinnedProjection(
                c, input: input, words: p.words, scales: p.scales, biases: p.biases, output: output, partials: partial, plan: plan,
                simdMatrix: true)
        }
        try project(w.sharedGate, s.inputBF16, s.sharedGate, s.sharedPartials[0], 2560, 640)
        try project(w.sharedUp, s.inputBF16, s.sharedUp, s.sharedPartials[1], 2560, 640)
        try f.scalars.encodeProduct(
            c, gate: s.sharedGate, up: s.sharedUp, output: s.sharedProduct, exceptions: s.sharedFlags, rows: s.rows)
        try project(w.sharedDown, s.sharedProduct, s.sharedOut, s.sharedPartials[2], 640, 2560)
    }

    /// `ExpertProjection`-shaped records with sane scales and biases, `count` of them in buffers of 64.
    static func records(_ device: MTLDevice, count: Int, seed: UInt64) throws -> [ExpertBuffer] {
        var result: [ExpertBuffer] = []
        var left = count
        var block = 0
        while left > 0 {
            let n = min(64, left)
            let buffer = try filled(device, n * PackStore.expertBytes, seed: seed + UInt64(block))
            for i in 0..<n {
                let base = buffer.contents().advanced(by: i * PackStore.expertBytes)
                for kind in ExpertProjection.allCases {
                    let groups = kind.width * kind.outputs / ModelLayout.experts.group
                    let scales = base.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
                    let biases = base.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
                    for g in 0..<groups {
                        scales[g] = BF16.encode(Float((g + i) % 7 + 1) / 1024)
                        biases[g] = BF16.encode(Float((g + i) % 9 - 4) / 256)
                    }
                }
                result.append(ExpertBuffer(replicaBuffer: buffer, offset: i * PackStore.expertBytes))
            }
            left -= n
            block += 1
        }
        return result
    }

    package static func run(_ arguments: [String]) throws {
        guard let mode = arguments.first, ["prefill", "decode"].contains(mode) else {
            throw EngineError("Usage: engine-checks --bench-qwen <prefill [tokens] [waves] [MMA floor] [even]|decode>")
        }
        let f = try fixture(mmaMinRows: arguments.count > 3 ? Int(arguments[3]) ?? 192 : 192)
        if mode == "prefill" {
            let tokens = arguments.count > 1 ? Int(arguments[1]) ?? 2526 : 2526
            guard (256...8192).contains(tokens) else { throw EngineError("Prefill bench tokens must be in 256...8192") }
            try prefill(
                f, tokens: tokens, waves: arguments.count > 2 ? Int(arguments[2]) ?? 8 : 8,
                even: arguments.count > 4 && arguments[4] == "even")
        } else {
            try decode(f)
        }
    }

    static func prefill(_ f: Fixture, tokens: Int, waves: Int, even: Bool) throws {
        let tiles = Double((tokens + 255) / 256)
        let gdnLayers = 36.0
        let layers = 48.0
        let w = try layerWeights(f, seed: 11)
        let s = try scratch(f, rows: 256)
        var rows: [Row] = []
        rows.append(try measure("hc tile (x2)", perLayer: 2 * tiles, queue: f.queue) { c, _ in try encodeHC(f, c, w, s) })
        rows.append(try measure("gdn tile", perLayer: tiles, queue: f.queue) { c, _ in try encodeGDN(f, c, w, s, next: nil) })
        // The GDN tile's three projections alone (part of "gdn tile", not added to the sum).
        var parts: [Row] = []
        for (name, p, width, outputs) in [
            ("  gdn qkv projection", w.qkv, 2560, 10240), ("  gdn z projection", w.z, 2560, 6144),
            ("  gdn out projection", w.out, 6144, 2560),
        ] {
            let out = try zeroed(f.device, 256 * outputs * 4)
            let x = try constant(f.device, count: 256 * width) { BF16.encode(Float($0 % 991 - 495) / 512) }
            let plan = try f.kernels.pinnedProjectionPlan(rows: 256, width: width, outputs: outputs, simdMatrix: true)
            let partial = plan.partialBytes == 0 ? nil : try zeroed(f.device, plan.partialBytes)
            parts.append(
                try measure(name, perLayer: tiles, queue: f.queue) { c, _ in
                    try f.kernels.encodePinnedProjection(
                        c, input: x, words: p.words, scales: p.scales, biases: p.biases, output: out, partials: partial, plan: plan,
                        simdMatrix: true)
                })
        }
        rows.append(try measure("router tile", perLayer: tiles, queue: f.queue) { c, _ in try encodeRouter(f, c, w, s) })
        rows.append(try measure("shared expert tile", perLayer: tiles, queue: f.queue) { c, _ in try encodeShared(f, c, w, s) })
        // Ten-rank reduction for the scope.
        let contributions = try zeroed(f.device, tokens * 10 * 2560 * 4)
        let rankWeights = try floats(f.device, count: tokens * 10) { _ in 0.1 }
        let reduced = try zeroed(f.device, tokens * 2560 * 4)
        rows.append(
            try measure("reduce10 (scope)", perLayer: 1, queue: f.queue, copies: 2) { c, _ in
                try f.reductions.encodeWeightedExpertReduce10(
                    c, input: contributions, weights: rankWeights, output: reduced, tokens: tokens)
            })
        // The expert layer command at the release plan's eight waves.
        let records = try records(f.device, count: 512, seed: 97)
        var rng = SplitMix(seed: 5)
        var selected: [Int: [(Int, Int)]] = [:]
        for token in 0..<tokens {
            // "even": every expert gets the same number of rows (no partial tiles beyond the count's own).
            var chosen: [Int] = even ? (0..<10).map { (token * 10 + $0) % 512 } : []
            while chosen.count < 10 {
                let e = Int(rng.next() % 512)
                if !chosen.contains(e) { chosen.append(e) }
            }
            for (rank, e) in chosen.enumerated() { selected[e, default: []].append((token, rank)) }
        }
        let experts = selected.keys.sorted()
        let buffers = experts.map { records[$0] }
        let transfer = try PrefillMoETransfer()
        let input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
        var row = Row(name: "expert layer command", perLayer: 1)
        var gpu: [Double] = []
        var host: [Double] = []
        var wall: [Double] = []
        for run in 0...3 {
            try autoreleasepool {
                let budget = try PrefillMoEBudget(ledger: f.ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                let workspace = try PrefillMoEWorkspace(input: input, selectedRows: selected, budget: budget, transfer: transfer)
                let before = f.fused.snapshot()
                let (d0, e0, b0) = (dispatches, encoders, f.fused.layerBuffers)
                let start = now()
                try f.fused.evaluateDeviceLayer(
                    experts: experts, buffers: buffers, selectedRows: selected, workspace: workspace, simdMatrix: true, waves: waves)
                let elapsed = now() - start
                let after = f.fused.snapshot()
                row.dispatches = Double(dispatches - d0)
                row.encoders = Double(encoders - e0)
                row.commands = Double(f.fused.layerBuffers - b0)
                if run > 0 {
                    gpu.append(after.gpuSeconds - before.gpuSeconds)
                    host.append((after.hostPrepareSeconds ?? 0) - (before.hostPrepareSeconds ?? 0))
                    wall.append(elapsed)
                }
            }
        }
        row.gpu = median(gpu)
        row.host = median(host)
        rows.append(row)
        let mean = Double(tokens * 10) / 512
        Swift.print(
            String(
                format: "prefill, one layer of a %d-token scope (%.0f tiles; %d experts, mean %.1f rows each); expert layer wall %.3f s",
                tokens, tiles, experts.count, mean, median(wall)))
        let tileRows = ["hc tile (x2)", "router tile", "shared expert tile", "reduce10 (scope)", "expert layer command"]
        print(
            rows, layers: { r in r.perLayer * (tileRows.contains(r.name) ? layers : gdnLayers) },
            title: "scaled to the prompt (GDN on 36 layers, the rest on 48):")
        print(parts, layers: { $0.perLayer * gdnLayers }, title: "of which, in the gdn tile:")
        let covered = rows.reduce(0.0) { $0 + $1.gpu * $1.perLayer * (tileRows.contains($1.name) ? layers : gdnLayers) }
        Swift.print(
            String(
                format: "covered GPU per prompt token: %.2f ms (real model: about 11 ms at 2,526 tokens, 28 s GPU busy)",
                covered / Double(tokens) * 1e3))
    }

    static func decode(_ f: Fixture) throws {
        // Four rotating layer sets defeat the system cache, as 48 distinct layers do.
        let sets = try (0..<4).map { try layerWeights(f, seed: UInt64(100 * $0)) }
        let s = try scratch(f, rows: 1)
        let records = try records(f.device, count: 40, seed: 31)
        let gathered = try GatheredExpertDecode(ledger: f.ledger, fused: f.fused, reductions: f.reductions)
        let region = try GatheredExpertDecode.RegionBuffers(
            gate: zeroed(f.device, 10 * 640 * 4), up: zeroed(f.device, 10 * 640 * 4), product: zeroed(f.device, 10 * 640 * 2),
            down: zeroed(f.device, 10 * 2560 * 4), descriptors: zeroed(f.device, gathered.regionArgumentBufferBytes),
            ranks: zeroed(f.device, 40))
        let rankWeights = try floats(f.device, count: 10) { _ in 0.1 }
        let expertOut = try zeroed(f.device, 2560 * 4)
        let next = try (0..<8).map { _ in try GDNRecurrentStorage(device: f.device, outputCount: 6144, ledger: f.ledger) }
        func experts(_ c: MTLCommandBuffer, _ i: Int) throws {
            let group = (0..<10).map { records[(i % 4) * 10 + $0] }
            try gathered.encodeRegion(
                c, inputBF16: s.inputBF16, weights: rankWeights,
                records: group.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) },
                canonicalRanks: Array(0..<10), output: expertOut, buffers: region)
        }
        var rows: [Row] = []
        rows.append(try measure("hc row (x3)", perLayer: 3, queue: f.queue) { c, i in try encodeHC(f, c, sets[i % 4], s) })
        rows.append(try measure("gdn row", perLayer: 1, queue: f.queue) { c, i in try encodeGDN(f, c, sets[i % 4], s, next: next[i]) })
        rows.append(try measure("router row (x2)", perLayer: 2, queue: f.queue) { c, i in try encodeRouter(f, c, sets[i % 4], s) })
        rows.append(try measure("shared expert row", perLayer: 1, queue: f.queue) { c, i in try encodeShared(f, c, sets[i % 4], s) })
        rows.append(try measure("ten experts + reduce", perLayer: 1, queue: f.queue) { c, i in try experts(c, i) })
        // The LM head: 2560 -> 248,320 on one row, then the host's greedy readout of the logits.
        let head = try projection(f.device, width: 2560, outputs: 248_320, seed: 3)
        let logits = try zeroed(f.device, 248_320 * 4)
        let plan = try f.kernels.pinnedProjectionPlan(rows: 1, width: 2560, outputs: 248_320, simdMatrix: true)
        rows.append(
            try measure("lm head", perLayer: 1.0 / 48, queue: f.queue, copies: 1) { c, _ in
                try f.kernels.encodePinnedProjection(
                    c, input: s.inputBF16, words: head.words, scales: head.scales, biases: head.biases, output: logits, partials: nil,
                    plan: plan, simdMatrix: true)
            })
        print(rows, layers: { r in r.perLayer * (r.name == "gdn row" ? 36 : 48) }, title: "decode, one token (GDN on 36 layers):")
        // The host side of the readout: copy out the logits and take the argmax (temperature 0).
        var readout: [Double] = []
        for _ in 0..<5 {
            let start = now()
            let p = logits.contents().assumingMemoryBound(to: Float.self)
            let copy = Array(UnsafeBufferPointer(start: p, count: 248_320))
            var best = 0
            for i in copy.indices where copy[i] > copy[best] { best = i }
            readout.append(now() - start + Double(best & 0) * 0)
        }
        Swift.print(String(format: "host readout (copy 248,320 logits + argmax): %.3f ms", median(readout) * 1e3))
        // A whole token as production commits it: two commands a layer (attention + router + shared, then experts),
        // 48 layers back to back, then the LM head; GPU busy (sum of command intervals) and the span.
        var spans: [Double] = []
        var busy: [Double] = []
        var encode: [Double] = []
        var counts: [Double] = []
        var commits: [Double] = []
        var singleEncode: [Double] = []
        // Runs 5 to 8 encode the same token into one command buffer: what the host's encode costs without new buffers.
        for run in 0...8 {
            let single = run >= 5 ? f.queue.makeCommandBuffer() : nil
            var commands: [MTLCommandBuffer] = []
            let start = now()
            var committing = 0.0
            let (d0, e0) = (dispatches, encoders)
            for layer in 0..<48 {
                let w = sets[layer % 4]
                guard let a = single ?? f.queue.makeCommandBuffer(), let x = single ?? f.queue.makeCommandBuffer() else {
                    throw EngineError("Bench command unavailable")
                }
                try encodeHC(f, a, w, s)
                try encodeGDN(f, a, w, s, next: next[layer % 8])
                try encodeHC(f, a, w, s)
                try encodeRouter(f, a, w, s)
                try encodeHC(f, a, sets[(layer + 1) % 4], s)
                try encodeRouter(f, a, sets[(layer + 1) % 4], s)
                try encodeShared(f, a, w, s)
                var t = now()
                if single == nil { a.commit() }
                committing += now() - t
                try experts(x, layer)
                t = now()
                if single == nil { x.commit() }
                committing += now() - t
                if single == nil { commands += [a, x] }
            }
            if let single {
                single.commit()
                commands.append(single)
            }
            guard let h = f.queue.makeCommandBuffer() else { throw EngineError("Bench command unavailable") }
            try f.kernels.encodePinnedProjection(
                h, input: s.inputBF16, words: head.words, scales: head.scales, biases: head.biases, output: logits, partials: nil,
                plan: plan, simdMatrix: true)
            h.commit()
            commands.append(h)
            let encoded = now() - start
            h.waitUntilCompleted()
            guard commands.allSatisfy({ $0.status == .completed }) else { throw EngineError("Bench token failed") }
            if run >= 5 {
                singleEncode.append(encoded)
            } else if run > 0 {
                spans.append(h.gpuEndTime - commands[0].gpuStartTime)
                busy.append(commands.reduce(0) { $0 + $1.gpuEndTime - $1.gpuStartTime })
                encode.append(encoded)
                commits.append(committing)
                counts.append(Double(dispatches - d0) + Double(encoders - e0) / 1e6)
            }
        }
        Swift.print(
            String(
                format:
                    "token (48 GDN-shaped layers, 97 commands, %.0f dispatches): GPU busy %.2f ms, GPU span %.2f ms, host encode and commit %.2f ms (commits %.2f ms; includes waits at the queue's in-flight limit)",
                counts[0].rounded(.down), median(busy) * 1e3, median(spans) * 1e3, median(encode) * 1e3, median(commits) * 1e3))
        Swift.print(String(format: "the same token encoded into one command buffer: host encode %.2f ms", median(singleEncode) * 1e3))
        Swift.print("real model: about 100 ms of GPU per token (73% of 137 ms at 7.3 tok/s)")
        try encoderCost(f)
    }

    /// The GPU cost of a compute encoder: 1,000 one-threadgroup dispatches, each in its own encoder or all in one.
    static func encoderCost(_ f: Fixture) throws {
        let library = try f.device.makeLibrary(
            source:
                "#include <metal_stdlib>\nkernel void bench_touch(device uint *p [[buffer(0)]], uint i [[thread_position_in_grid]]) { p[i] += 1u; }",
            options: nil)
        let pipeline = try f.device.makeComputePipelineState(function: library.makeFunction(name: "bench_touch")!)
        let target = try zeroed(f.device, 4096)
        for shared in [false, true] {
            var gpu: [Double] = []
            for _ in 0..<5 {
                guard let c = f.queue.makeCommandBuffer() else { throw EngineError("Bench command unavailable") }
                var e = c.makeComputeCommandEncoder()!
                for i in 0..<1000 {
                    if !shared && i > 0 {
                        e.endEncoding()
                        e = c.makeComputeCommandEncoder()!
                    }
                    e.setComputePipelineState(pipeline)
                    e.setBuffer(target, offset: 0, index: 0)
                    e.dispatchThreads(
                        MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                }
                e.endEncoding()
                c.commit()
                c.waitUntilCompleted()
                gpu.append(c.gpuEndTime - c.gpuStartTime)
            }
            Swift.print(
                String(
                    format: "1,000 dependent tiny dispatches, %@: %.1f µs each", shared ? "one encoder" : "one encoder each",
                    median(gpu) * 1e3))
        }
    }

    struct SplitMix {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z >> 31)
        }
    }
}
