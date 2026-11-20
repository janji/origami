import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of the whole-execution schedule's parts: command ownership and joins, workspace, scope and
/// constant admission, and the device buffer interfaces. They check primitives, not whole-model replies or speed.
package enum WholeExecutionSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private static func expectThrow(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Whole execution accepted an invalid boundary")
    }

    package static func sharedGateActualRows() throws {
        let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
        let host = try ledger.reserve(bytes: 32_000_000, category: "whole_shared_gate_fixture")
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let dense = try PinnedDenseLinear(ledger: ledger)
            let weights = (0..<2560).map { (i: Int) in BF16.decode(BF16.encode(Float((i * 113 + 29) % 1009 - 504) / 129)) }
            let w = try Check.buffer(weights, device)
            for rows in [1, 3, 116, 160, 161, 256] {
                let input = (0..<(rows * 2560)).map { (index: Int) -> Float in
                    BF16.decode(BF16.encode(Float((index * 71 + 17) % 991 - 495) / 193))
                }
                let expected = try dense.linear(input, weights: w, width: 2560, outputs: 1, outputBF16: true)
                let x = try Check.buffer(input, device)
                let out = try Check.buffer([Float](repeating: -123, count: rows + 1), device)
                let command = queue.makeCommandBuffer()!
                try dense.encodeSharedGateRows(command, input: x, weights: w, output: out, rows: rows)
                try Check.joined(command)
                try Check.equal(Check.values(out, rows), expected, "shared gate actual-row bytes")
                try Check.require(out.contents().assumingMemoryBound(to: Float.self)[rows] == -123, "Shared gate exceeded actual rows")
                let unsubmitted = queue.makeCommandBuffer()!
                try expectThrow { try dense.encodeSharedGateRows(unsubmitted, input: x, weights: w, output: x, rows: rows) }
            }
        }
        try ledger.release(host)
        try Check.require(ledger.snapshot().reserved == 0, "Shared gate actual-row fixture leaked")
    }

    package static func denseIndexActualRows() throws {
        let ledger = try MemoryLedger(ceiling: 96_000_000, transientMargin: 1_000_000)
        let host = try ledger.reserve(bytes: 48_000_000, category: "whole_dense_index_fixture")
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let dense = try PinnedDenseLinear(ledger: ledger)
            let weights = (0..<(640 * 2560)).map { (index: Int) -> Float in
                BF16.decode(BF16.encode(Float((index * 113 + 29) % 1009 - 504) / 129))
            }
            let w = try Check.buffer(weights, device)
            for rows in [1, 116, 256] {
                try autoreleasepool {
                    let input = (0..<(rows * 2560)).map { (index: Int) -> Float in
                        BF16.decode(BF16.encode(Float((index * 71 + 17) % 991 - 495) / 193))
                    }
                    let expected = try dense.linear(input, weights: w, width: 2560, outputs: 640, outputBF16: true)
                    let x = try Check.buffer(input, device)
                    let out = try Check.buffer([Float](repeating: -123, count: rows * 640 + 1), device)
                    let plan = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 640)
                    let partial =
                        plan.kind == .splitK
                        ? try Check.buffer([Float](repeating: 0, count: rows * 640 * plan.splitKPartitions), device) : nil
                    let command = queue.makeCommandBuffer()!
                    try dense.encodeRows(
                        command, input: x, weights: w, output: out, partials: partial, rows: rows, width: 2560, outputs: 640,
                        outputBF16: true)
                    try Check.joined(command)
                    try Check.equal(Check.values(out, rows * 640), expected, "QSA dense index actual-row bytes")
                    try Check.require(
                        out.contents().assumingMemoryBound(to: Float.self)[rows * 640] == -123, "Dense index exceeded actual rows")
                }
            }
        }
        try ledger.release(host)
        try Check.require(ledger.snapshot().reserved == 0, "Dense index actual-row fixture leaked")
    }

    /// No model tensors are read. Exercise the actual queue owner, both fixed
    /// generations, cancellation before/after commit and joined pin retirement.
    package static func scheduleOwnership(store: PackStore) throws {
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let weights = try ReferenceWeights(
                store: store, kernels: kernels, ledger: ledger, simdMatrix: true,
                residentBytes: 1_000_000, residentSmallTensors: true, denseResidentBytes: 1_000_000,
                fusedExperts: true, pinnedDense: true, fusedHCDecode: true, fusedHCPrefill: true)
            let helpers = try weights.decodeFrontierHelpers()
            let reductions = try PinnedReductions(ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            let rope = try PinnedRoPE(ledger: ledger)
            let indexer = try PinnedIndexer(ledger: ledger)
            let math = LayerMath(
                kernels: kernels, linear: { _, _, _ in throw EngineError("Unexpected fixture tensor read") },
                tensor: { _ in throw EngineError("Unexpected fixture tensor read") }, reductions: reductions, selector: selector,
                pinnedAttention: attention, pinnedRoPE: rope, pinnedIndexer: indexer)
            let preparation = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let gdn = try weights.makeGDNDeviceBlock(preparation: preparation)
            let gathered = try GatheredExpertDecode(ledger: ledger, fused: helpers.fused, reductions: reductions)
            let injection = try FusedResidualInjection(ledger: ledger)
            let schedule = try WholeExecutionSchedule(
                weights: weights, math: math, ledger: ledger, context: 8192,
                gdn: gdn, gathered: gathered, selector: selector, reductions: reductions, injection: injection)
            try expectThrow { _ = try schedule.command("outside-scope") }
            try schedule.beginScope(decode: true)
            defer {
                try? schedule.joinAll()
                schedule.endScope()
            }
            try expectThrow { try schedule.beginScope(decode: false) }
            try expectThrow { try schedule.trim() }
            try expectThrow { try schedule.prepareDecode() }
            try expectThrow { _ = try schedule.decodeTile(2) }
            let a = try schedule.decodeTile(0)
            let b = try schedule.decodeTile(1)
            try Check.require(
                a !== b && a[.result] !== b[.result]
                    && schedule.snapshot().decodeGenerationBytes == a.layout.admittedBytes + b.layout.admittedBytes,
                "Two decode generations alias or are not simultaneously charged")
            guard let foreignQueue = kernels.device.makeCommandQueue(), let foreign = foreignQueue.makeCommandBuffer() else {
                throw EngineError("Fixture foreign command unavailable")
            }
            try expectThrow { try schedule.submit(foreign, shouldContinue: { true }) }
            let cancelled = try schedule.command("cancel-before-commit")
            var refused = false
            do { try schedule.submit(cancelled, shouldContinue: { false }) } catch is IOOperationCancelled { refused = true }
            try Check.require(
                refused && cancelled.status == .notEnqueued && schedule.snapshot().decode.submitted == 0,
                "Precommit cancellation submitted work")
            try expectThrow { try schedule.join(cancelled) }
            func fill(_ command: MTLCommandBuffer, _ buffer: MTLBuffer, _ byte: UInt8) throws {
                guard let encoder = command.makeBlitCommandEncoder() else { throw EngineError("Fixture blit unavailable") }
                encoder.fill(buffer: buffer, range: 0..<buffer.length, value: byte)
                encoder.endEncoding()
            }
            let shared = try schedule.command("first-reader")
            let next = try schedule.command("dependent-reader")
            try fill(shared, a[.result], 0x3c)
            try fill(next, b[.result], 0x4d)
            try schedule.submit(shared, shouldContinue: { true })
            try schedule.submit(next, shouldContinue: { true })
            try expectThrow { try schedule.submit(shared, shouldContinue: { true }) }
            // Pre-encoded decode: a layer's ready group, its experts and the next layer's attention may be in flight together.
            let third = try schedule.command("third-reader")
            try fill(third, a[.residual], 0x2b)
            try schedule.submit(third, shouldContinue: { true })
            let fourth = try schedule.command("fourth-reader-refused")
            try expectThrow { try schedule.submit(fourth, shouldContinue: { true }) }
            try schedule.join(third)
            try schedule.join(next)
            try schedule.join(shared)
            try expectThrow { try schedule.join(next) }
            try Check.require(
                a[.result].contents().load(as: UInt32.self) == 0x3c3c_3c3c && b[.result].contents().load(as: UInt32.self) == 0x4d4d_4d4d,
                "Joined generation writes differ")
            var calls = 0
            let late = try schedule.command("cancel-after-commit")
            try fill(late, b[.result], 0x5e)
            refused = false
            do {
                try schedule.joined(
                    late,
                    shouldContinue: {
                        calls += 1
                        return calls == 1
                    })
            } catch is IOOperationCancelled { refused = true }
            try Check.require(
                refused && late.status == .completed && b[.result].contents().load(as: UInt32.self) == 0x5e5e_5e5e,
                "Postcommit cancellation did not join its writer")

            // The real ten-record demand owner must retain every pin through
            // both joined readers, including a thrown cancellation afterward.
            let config = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 12, slotsPerSlab: 4,
                demandWindow: 10, readConcurrency: 2)
            let executor = try ExpertExecution(device: kernels.device, ledger: ledger, configuration: config, packedAvailable: { false }) {
                _, _, destination in
                destination.initializeMemory(as: UInt8.self, repeating: 0)
            }
            try executor.beginSequence()
            defer { try? executor.finishSequence() }
            let window = try executor.makeWindow(layer: 0, experts: Array(0..<10))
            defer { window.close() }
            refused = false
            do {
                try window.consumeTen(experts: Array(0..<10)) { records in
                    try Check.require(records.count == 10 && executor.snapshot().cache.pinnedSlots == 10, "Missing live expert pins")
                    let command = try schedule.command("ten-pinned-readers")
                    guard let encoder = command.makeBlitCommandEncoder() else { throw EngineError("Fixture pin blit unavailable") }
                    for (index, record) in records.enumerated() {
                        encoder.copy(
                            from: record.buffer, sourceOffset: record.offset, to: a[.result], destinationOffset: index * 4, size: 4)
                    }
                    encoder.endEncoding()
                    try schedule.submit(command, shouldContinue: { true })
                    try schedule.joinAll()
                    try Check.require(executor.snapshot().cache.pinnedSlots == 10, "Pins retired before their consumer returned")
                    throw IOOperationCancelled()
                }
            } catch is IOOperationCancelled { refused = true }
            let io = executor.snapshot()
            let count = schedule.snapshot().decode
            try Check.require(
                refused && io.cache.pinnedSlots == 0 && io.cache.inFlightReads == 0 && io.cache.reservedSlots == 0
                    && io.io.drives.allSatisfy({ $0.outstanding == 0 }), "Cancelled whole demand retained readers/producers")
            try Check.require(
                count.submitted == 5 && count.joined == 5 && count.failed == 0 && count.timed + count.missingTimestamps == count.joined,
                "Whole command accounting does not reconcile")
            try WholeTokenPlanSelfChecks.commandMissWindows(schedule: schedule, workspace: a, ledger: ledger, device: kernels.device)
            try QwenPairScheduleSelfChecks.readOverlap(schedule: schedule, workspace: a, ledger: ledger, device: kernels.device)
        }
        try Check.require(ledger.snapshot().reserved == 0, "Whole schedule command/pin fixture leaked owners")
    }

    package static func constantOwnership(store: PackStore, names: [String], expected: [[UInt32]]) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1)
        var escaped: MTLBuffer?
        var charge: UInt64 = 0
        try autoreleasepool {
            let cache = try ResidentTensorCache(store: store, device: device, ledger: ledger, byteLimit: 1_000_000)
            let canonical = names.map { "language_model." + $0 }
            guard let leases = try cache.acquire(canonical) else { throw EngineError("Constant fixture bypassed") }
            let specs = zip(canonical, expected).map { (name: $0.0, count: $0.1.count) }
            let constants = try WholeLayerConstants(leases: leases, expected: specs, device: device, ledger: ledger)
            charge = constants.chargedBytes
            for (name, bits) in zip(canonical, expected) {
                let pointer = constants.buffers[name]!.contents().assumingMemoryBound(to: UInt32.self)
                try Check.require(
                    Array(UnsafeBufferPointer(start: pointer, count: bits.count)) == bits, "Retained constants changed source bits")
            }
            let denied = try MemoryLedger(ceiling: charge - 1, transientMargin: 0)
            try expectThrow { _ = try WholeLayerConstants(leases: leases, expected: specs, device: device, ledger: denied) }
            try Check.require(denied.snapshot().reserved == 0, "Constant admission denial leaked")
            let before = ledger.snapshot().reserved
            try expectThrow { _ = try WholeLayerConstants(leases: leases.reversed(), expected: specs, device: device, ledger: ledger) }
            try Check.require(ledger.snapshot().reserved == before, "Constant identity refusal leaked")
            escaped = constants.buffers[canonical[0]]
        }
        try Check.require(
            escaped != nil && ledger.snapshot().reserved == charge, "Escaped decoded constant lost independent charge or retained raw cache"
        )
        autoreleasepool { escaped = nil }
        try Check.require(ledger.snapshot().reserved == 0, "Decoded constants failed retirement")
    }

    package static func workspaceAndReaders() throws {
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1_000_000)
        let readers = WholeLayerReaderBudget()
        var a: WholeLayerReader? = try readers.acquire(ledger: ledger)
        var b: WholeLayerReader? = try readers.acquire(ledger: ledger)
        try expectThrow { _ = try readers.acquire(ledger: ledger) }
        try Check.require(
            a != nil && b != nil && readers.activeReaders == 2 && ledger.snapshot().reserved == 262144,
            "Two-layer metadata admission differs")
        a = nil
        try Check.require(readers.activeReaders == 1 && ledger.snapshot().reserved == 131072, "Reader charge retired too early/late")
        a = try readers.acquire(ledger: ledger)
        a = nil
        b = nil
        let denied = try MemoryLedger(ceiling: 131071, transientMargin: 0)
        try expectThrow { _ = try readers.acquire(ledger: denied) }
        try Check.require(
            readers.activeReaders == 0 && denied.snapshot().reserved == 0 && ledger.snapshot().reserved == 0,
            "Reader denial did not restore bound")
        try autoreleasepool {
            let kernels = try ReferenceKernels(
                pinnedUnary: true, ledger: ledger, pinnedRMS: true,
                quantizedTiled: true, quantizedTiledBM32: true, quantizedBM32MinRows: 256)
            let dense = try PinnedDenseLinear(ledger: ledger)
            let reductions = try PinnedReductions(ledger: ledger)
            let preparation = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let gdn = try GDNDeviceBlock(kernels: kernels, dense: dense, preparation: preparation, ledger: ledger, simdMatrix: true)
            let hc = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger, prefillKernels: kernels, prefillSIMDMatrix: true)
            let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
            let before = ledger.snapshot().reserved
            for rows in [1, 256] {
                for capacity in [262144, 8192, 32768] {
                    let layout = try WholeExecutionWorkspace.layout(
                        rows: rows, capacity: capacity, capture: false, argumentBytes: 4096,
                        kernels: kernels, simdMatrix: true, separateExpertBindings: rows == 1 && capacity != 262144, hc: hc, gdn: gdn,
                        attention: attention)
                    if rows == 1 {
                        try Check.require(
                            layout.admittedBytes <= ConversationGeometry.maximumDecodeTileBytes,
                            "Smaller decode workspace exceeds reused maximum-context allowance")
                    }
                    if capacity == ConversationGeometry.maximumFileContext {
                        let bound = rows == 1 ? ConversationGeometry.maximumDecodeTileBytes : ConversationGeometry.maximumPrefillTileBytes
                        try Check.require(
                            layout.admittedBytes + layout.savedBytes == bound,
                            "Runtime phase bound differs from the maximum workspace and its exact storage saving")
                    }
                    let refused = try MemoryLedger(ceiling: layout.admittedBytes - 1, transientMargin: 0)
                    try expectThrow { _ = try WholeExecutionWorkspace(device: kernels.device, ledger: refused, layout: layout) }
                    try Check.require(refused.snapshot().reserved == 0, "Whole tile admission denial leaked")
                    var escaped: MTLBuffer?
                    try autoreleasepool {
                        let s = try WholeExecutionWorkspace(device: kernels.device, ledger: ledger, layout: layout)
                        s.clearFlags()
                        escaped = s[.result]
                        let all = WholeExecutionWorkspace.Slot.allCases.compactMap { s.optional($0) }
                        var physical: [Int: ObjectIdentifier] = [:]
                        for slot in WholeExecutionWorkspace.Slot.allCases {
                            guard let buffer = s.optional(slot) else { continue }
                            let storage = layout.storageIndices[slot.rawValue]
                            let identity = ObjectIdentifier(buffer)
                            if let expected = physical[storage] {
                                try Check.require(identity == expected, "Declared whole tile alias has different physical storage")
                            } else {
                                physical[storage] = identity
                            }
                        }
                        let expectedAllocations = layout.allocationSizes.filter { $0 > 0 }.count
                        try Check.require(
                            physical.count == expectedAllocations && Set(physical.values).count == expectedAllocations
                                && Set(all.map { ObjectIdentifier($0) }).count == expectedAllocations,
                            "Whole tile aliases differ from the declared physical allocation map")
                        try Check.require(
                            ledger.snapshot().reserved == before + layout.admittedBytes, "Whole tile charge differs from layout")
                        if capacity == 262144 {
                            guard let queue = kernels.device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
                                let encoder = command.makeBlitCommandEncoder()
                            else { throw EngineError("Maximum tile physical check unavailable") }
                            for buffer in all { encoder.fill(buffer: buffer, range: 0..<buffer.length, value: 0x6b) }
                            encoder.endEncoding()
                            try Check.joined(command)
                            for buffer in all {
                                try Check.require(
                                    buffer.contents().load(as: UInt8.self) == 0x6b
                                        && buffer.contents().load(fromByteOffset: buffer.length - 1, as: UInt8.self) == 0x6b,
                                    "Maximum tile allocation did not materialize")
                            }
                        }
                        if rows == 256, capacity == 8192 {
                            try joinedScopePublication(device: kernels.device, ledger: ledger, workspace: s)
                            try qsaTailPublication(workspace: s)
                        }
                        // Every actual key frontier must fit the capacity plan,
                        // including rounding changes in stripe/merge geometry.
                        for keys in stride(from: max(rows, 2049), through: capacity, by: 127) {
                            let p = try attention.bufferScratchPlan(rows: rows, keys: keys)
                            let t = try BatchedTopIndices.bufferPlan(rows: rows, columns: keys / 4, count: 512)
                            try Check.require(
                                p.intermediateBytes <= layout.sizes[WholeExecutionWorkspace.Slot.qsaAttentionPartial.rawValue]
                                    && p.scoreBytes <= layout.sizes[WholeExecutionWorkspace.Slot.qsaAttentionScores.rawValue]
                                    && t.firstBytes <= s[.qsaSortFirst].length
                                    && t.secondBytes <= (s.optional(.qsaSortSecond)?.length ?? 0), "Whole QSA frontier exceeds fixed layout"
                            )
                        }
                    }
                    try Check.require(
                        escaped != nil && ledger.snapshot().reserved == before + layout.admittedBytes,
                        "Escaped whole frontier outlived admission")
                    autoreleasepool { escaped = nil }
                    try Check.require(ledger.snapshot().reserved == before, "Whole frontier did not retire")
                }
            }
        }
        try Check.require(ledger.snapshot().reserved == 0, "Whole workspace helpers leaked")
    }

    /// Regression for a joined copy falsely returning its source as a Swift
    /// error in the optimized throwing ABI. Use the real scope/workspace and
    /// a throwing caller, including rejected and first/last valid frontiers.
    private static func joinedScopePublication(device: MTLDevice, ledger: MemoryLedger, workspace s: WholeExecutionWorkspace) throws {
        let before = ledger.snapshot().reserved
        try autoreleasepool {
            let scope = try WholePrefillScope(device: device, ledger: ledger, layout: WholePrefillScope.layout(tokens: 1536)!)
            guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer() else {
                throw EngineError("Publication fixture command unavailable")
            }
            let count = 256 * 10240
            let sentinel: UInt32 = 0xa55a_a55a
            let value: UInt32 = 0x3ba8_0000
            scope[.residual].contents().assumingMemoryBound(to: UInt32.self).initialize(repeating: sentinel, count: 1536 * 10240)
            s[.result].contents().assumingMemoryBound(to: UInt32.self).initialize(repeating: value, count: count)
            func publish(_ start: Int) throws {
                guard scope.publishMix(start: start, workspace: s, command: command) else {
                    throw EngineError("Expected publication boundary refusal")
                }
            }
            try expectThrow { try publish(0) }
            try Check.joined(command)
            for invalid in [-256, 1, 1536] { try expectThrow { try publish(invalid) } }
            let p = scope[.residual].contents().assumingMemoryBound(to: UInt32.self)
            try Check.require((0..<(1536 * 10240)).allSatisfy { p[$0] == sentinel }, "Rejected publication changed scope bytes")
            try publish(0)
            try publish(1280)
            for i in 0..<(1536 * 10240) {
                let expected = i < count || i >= 1280 * 10240 ? value : sentinel
                try Check.require(p[i] == expected, "Joined publication changed extent/bits")
            }
        }
        try Check.require(ledger.snapshot().reserved == before, "Publication fixture retained scope owners")
    }

    private static func qsaTailPublication(workspace s: WholeExecutionWorkspace) throws {
        let buffer = s[.qsaOldTail]
        let p = buffer.contents().assumingMemoryBound(to: UInt16.self)
        let count = buffer.length / 2
        let sentinel: UInt16 = 0xa55a
        func stage(_ values: [UInt16]) throws {
            guard QSAExecutionBlock.stageOldTail(values, into: buffer) else { throw EngineError("Expected QSA tail refusal") }
        }
        for rows in 0...3 {
            p.initialize(repeating: sentinel, count: count)
            let values = (0..<(rows * 128)).map { UInt16(truncatingIfNeeded: $0 &* 127 &+ 0x3f0e) }
            try stage(values)
            try Check.require(
                (0..<count).allSatisfy { p[$0] == ($0 < values.count ? values[$0] : sentinel) },
                "QSA tail copy changed bits or sentinel extent")
        }
        p.initialize(repeating: sentinel, count: count)
        for invalid in [1, 127, 129, 512] { try expectThrow { try stage([UInt16](repeating: 1, count: invalid)) } }
        try Check.require((0..<count).allSatisfy { p[$0] == sentinel }, "Rejected QSA tail copy mutated storage")
    }

    package static func scopeEscrowAdoption() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        try Check.require(
            WholePrefillScope.layout(tokens: 256) == nil && WholePrefillScope.layout(tokens: 512) == nil,
            "Small scope silently borrowed the phase reserve")
        for tokens in [1536, 7680, 8192] {
            guard let layout = try WholePrefillScope.layout(tokens: tokens) else {
                throw EngineError("Declared scope did not fit its existing allowance")
            }
            try Check.require(
                layout.originalExtra == UInt64(tokens - 256) * 240000
                    && layout.deviceBytes + layout.remainingHostBytes == layout.originalExtra, "Scope escrow changed the original bound")
        }
        let layout = try WholePrefillScope.layout(tokens: 1536)!
        let denied = try MemoryLedger(ceiling: layout.originalExtra - 1, transientMargin: 0)
        try expectThrow { _ = try WholePrefillScope(device: device, ledger: denied, layout: layout) }
        try Check.require(denied.snapshot().reserved == 0, "Scope partial admission denial leaked")
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1_000_000)
        var escaped: MTLBuffer?
        try autoreleasepool {
            let scope = try WholePrefillScope(device: device, ledger: ledger, layout: layout)
            try Check.require(ledger.snapshot().reserved == layout.originalExtra, "Whole scope admission differs from original allowance")
            let embeddings = (0..<(1536 * 2560)).map { (index: Int) -> Float in
                let value = (index * 17 + index / 2560) % 1024
                return Float(value) / 16
            }
            try scope.stageEmbeddings(embeddings)
            let p = scope[.residual].contents().assumingMemoryBound(to: Float.self)
            for row in 0..<1536 {
                for branch in 0..<4 {
                    for column in [0, 2559] {
                        try Check.require(
                            p[row * 10240 + branch * 2560 + column].bitPattern == embeddings[row * 2560 + column].bitPattern,
                            "Scope embedding branch address changed")
                    }
                }
            }
            let selected = Dictionary(uniqueKeysWithValues: (0..<10).map { rank in (rank, (0..<1536).map { ($0, rank) }) })
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let adopted = try PrefillMoEWorkspace(
                tokens: 1536, input: scope[.input], selectedRows: selected, budget: scope.budget, transfer: transfer)
            try Check.require(
                adopted.input === scope[.input] && ledger.snapshot().reserved == layout.originalExtra,
                "Adopted MoE input was copied or double-charged")
            let before = scope.budget.liveBytes
            try expectThrow {
                _ = try PrefillMoEWorkspace(tokens: 1536, input: scope[.input], selectedRows: [:], budget: scope.budget, transfer: transfer)
            }
            try Check.require(scope.budget.liveBytes == before, "Rejected MoE coverage leaked its permit")
            try expectThrow { _ = try adopted.reduce(weights: [Float](repeating: 0.1, count: 15360), using: reductions) }
            let other = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: 1536))
            try expectThrow {
                _ = try PrefillMoEWorkspace(tokens: 1536, input: scope[.input], selectedRows: selected, budget: other, transfer: transfer)
            }
            try Check.require(other.liveBytes == 0, "Cross-escrow adoption consumed another budget")
            escaped = adopted.input
        }
        try Check.require(
            escaped != nil && ledger.snapshot().reserved == layout.originalExtra, "Escaped adopted input lost its complete scope charge")
        autoreleasepool { escaped = nil }
        try Check.require(ledger.snapshot().reserved == 0, "Whole scope failed retirement")
    }

    package static func bufferInterfaces() throws {
        let ledger = try MemoryLedger(ceiling: 384 * 1024 * 1024, transientMargin: 16 * 1024 * 1024)
        let host = try ledger.reserve(bytes: 192 * 1024 * 1024, category: "whole_qsa_interface_fixture_bound")
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedRMS: true)
            let rope = try PinnedRoPE(ledger: ledger)
            let indexer = try PinnedIndexer(ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let device = kernels.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("Metal unavailable") }
            func values(_ count: Int, _ m: Int) -> [Float] {
                (0..<count).map { LayerArithmetic.round(Float(($0 * m + 7) % 127 - 63) / 64) }
            }
            func output(_ count: Int) throws -> MTLBuffer { try Check.buffer([Float](repeating: -123, count: max(1, count)), device) }
            for (rows, blocks) in [(1, 512), (3, 513), (116, 541), (256, 8192)] {
                try autoreleasepool {
                    let x = values(rows * 512, 17)
                    let weights = [Float](repeating: 1, count: 128)
                    let positions = Array((blocks * 4 - rows)..<(blocks * 4))
                    let k = values(blocks * 128, 19)
                    let normalized = try kernels.rmsnorm(x: x, weights: weights)
                    let rotated = try rope.rotation(normalized, width: 128, heads: 4, positions: positions)
                    let scores = try indexer.scores(query: rotated, pooledKeys: k)
                    let selected = try selector.topIndices(scores: scores, columns: blocks, count: min(512, blocks))
                    rope.clearTables()
                    let input = try Check.buffer(x, device)
                    let gain = try Check.buffer(weights, device)
                    let key = try Check.buffer(k, device)
                    let norm = try output(x.count)
                    let rotation = try output(x.count)
                    let dots = try output(rows * 4 * blocks)
                    let score = try output(rows * blocks)
                    let plan = try BatchedTopIndices.bufferPlan(rows: rows, columns: blocks, count: min(512, blocks))
                    let first = try output(plan.firstBytes / 4)
                    let second = plan.secondBytes == 0 ? nil : try output(plan.secondBytes / 4)
                    let command = queue.makeCommandBuffer()!
                    try kernels.encodeQSARMS(command, input: input, weights: gain, output: norm, groups: rows * 4, width: 128)
                    let tables = try rope.encodeTables(command, positions: positions)
                    try expectThrow { try rope.publishTables(tables) }
                    try rope.encodeRotation(command, input: norm, output: rotation, width: 128, heads: 4, tables: tables)
                    try indexer.encodeScores(
                        command, query: rotation, pooledKeys: key, dots: dots, output: score, rows: rows, blocks: blocks)
                    let pairs = try selector.encodeTopIndices(
                        command, scores: score, first: first, second: second, rows: rows, columns: blocks, count: min(512, blocks))
                    try Check.joined(command)
                    try rope.publishTables(tables)
                    try Check.equal(Check.values(norm, x.count), normalized, "buffer RMS128")
                    try Check.equal(Check.values(rotation, x.count), rotated, "buffer RoPE")
                    try Check.equal(Check.values(score, scores.count), scores, "buffer indexer")
                    let p = pairs.contents().assumingMemoryBound(to: UInt32.self)
                    let count = min(512, blocks)
                    try Check.require(
                        (0..<rows).allSatisfy { r in (0..<count).allSatisfy { p[(r * count + $0) * 2 + 1] == selected[r][$0] } },
                        "buffer selection differs")
                    try expectThrow {
                        try rope.encodeRotation(
                            queue.makeCommandBuffer()!, input: norm, output: rotation, width: 128, heads: 4, tables: tables)
                    }
                    try expectThrow {
                        try kernels.encodeQSARMS(
                            queue.makeCommandBuffer()!, input: input, weights: gain, output: input, groups: rows * 4, width: 128)
                    }
                }
            }
            for (rows, keys, batched) in [
                (1, 2048, true), (1, 32768, true), (2, 8193, true), (3, 2049, false), (116, 1908, true), (256, 2304, true),
            ] {
                try autoreleasepool {
                    let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: batched)
                    let q = values(rows * 6144, 17)
                    let k = values(keys * 512, 29).map(BF16.encode)
                    let v = values(keys * 512, 13).map(BF16.encode)
                    let selected = (0..<rows).map { (row: Int) -> [UInt32] in
                        (0...(keys - rows + row)).filter { $0 % max(2, keys / 1024) == 0 }.map(UInt32.init)
                    }
                    let expected = try attention.attention(
                        queries: q, oldKeys: [], newKeys: k, oldValues: [], newValues: v, selectedPositions: selected)
                    let query = try Check.buffer(q.map(BF16.encode), device)
                    let key = try Check.buffer(k, device)
                    let value = try Check.buffer(v, device)
                    let out = try output(q.count)
                    let ids = try output(rows * 2051)
                    let counts = try output(rows)
                    let plan = try attention.bufferScratchPlan(rows: rows, keys: keys)
                    func scratch(_ bytes: Int) throws -> MTLBuffer? { bytes == 0 ? nil : try output(bytes / 4) }
                    let buffers = try PinnedAttention.BufferScratch(
                        intermediate: scratch(plan.intermediateBytes), sums: scratch(plan.sumsBytes),
                        maxima: scratch(plan.maximaBytes), scores: scratch(plan.scoreBytes))
                    let command = queue.makeCommandBuffer()!
                    let ticket = try attention.encodeHostPositions(command, selected: selected, keys: keys, ids: ids, counts: counts)
                    try attention.encodeAttention(
                        command, query: query, key: key, value: value, positions: ticket, output: out, scratch: buffers)
                    try Check.joined(command)
                    try Check.equal(Check.values(out, q.count), expected, "buffer attention \(rows)/\(keys)")
                    if rows <= 2, plan.scoreBytes > 0 {
                        // A flagged device producer can contain duplicates. Its
                        // dependent attention must stay within the admitted span
                        // before the transaction inspects the flag and repairs.
                        let guarded = try output((plan.scoreBytes + 32) / 4)
                        let sentinel: UInt32 = 0xa55a_a55a
                        guarded.contents().advanced(by: plan.scoreBytes).bindMemory(to: UInt32.self, capacity: 8)
                            .initialize(repeating: sentinel, count: 8)
                        let malformedCommand = queue.makeCommandBuffer()!
                        let malformedPositions = try attention.encodeHostPositions(
                            malformedCommand, selected: selected,
                            keys: keys, ids: ids, counts: counts)
                        ids.contents().initializeMemory(as: UInt32.self, repeating: 0, count: ids.length / 4)
                        counts.contents().initializeMemory(as: UInt32.self, repeating: UInt32.max, count: rows)
                        let guardedScratch = PinnedAttention.BufferScratch(
                            intermediate: buffers.intermediate, sums: buffers.sums,
                            maxima: buffers.maxima, scores: guarded)
                        try attention.encodeAttention(
                            malformedCommand, query: query, key: key, value: value,
                            positions: malformedPositions, output: out, scratch: guardedScratch)
                        try Check.joined(malformedCommand)
                        let singleton = try attention.attention(
                            queries: q, oldKeys: [], newKeys: k, oldValues: [], newValues: v,
                            selectedPositions: Array(repeating: [UInt32(0)], count: rows))
                        try Check.equal(Check.values(out, q.count), singleton, "bounded duplicate partition keys")
                        let tail = guarded.contents().advanced(by: plan.scoreBytes).assumingMemoryBound(to: UInt32.self)
                        try Check.require((0..<8).allSatisfy { tail[$0] == sentinel }, "Partition writer escaped its admitted span")
                        for invalid in [nil, key, try scratch(plan.scoreBytes - 4)] {
                            let refused = queue.makeCommandBuffer()!
                            let positions = try attention.encodeHostPositions(
                                refused, selected: selected, keys: keys, ids: ids, counts: counts)
                            let malformed = PinnedAttention.BufferScratch(
                                intermediate: buffers.intermediate, sums: buffers.sums,
                                maxima: buffers.maxima, scores: invalid)
                            try expectThrow {
                                try attention.encodeAttention(
                                    refused, query: query, key: key, value: value,
                                    positions: positions, output: out, scratch: malformed)
                            }
                            try Check.require(refused.status == .notEnqueued, "Invalid partition workspace reached the queue")
                        }
                    }
                    try expectThrow {
                        _ = try attention.encodeHostPositions(
                            queue.makeCommandBuffer()!, selected: [[1, 1]], keys: keys, ids: ids, counts: counts)
                    }
                    try expectThrow {
                        try attention.encodeAttention(
                            queue.makeCommandBuffer()!, query: query, key: key, value: value, positions: ticket, output: out,
                            scratch: buffers)
                    }
                }
            }
            // The Q/K norm uses RMS256; verify its separate reduction width.
            let x = values(3 * 24 * 256, 17)
            let w = [Float](repeating: 1, count: 256)
            let input = try Check.buffer(x, device)
            let gain = try Check.buffer(w, device)
            let out = try output(x.count)
            let command = queue.makeCommandBuffer()!
            try kernels.encodeQSARMS(command, input: input, weights: gain, output: out, groups: 3 * 24, width: 256)
            try Check.joined(command)
            try Check.equal(Check.values(out, x.count), kernels.rmsnorm(x: x, weights: w), "buffer RMS256")
        }
        try ledger.release(host)
        try Check.require(ledger.snapshot().reserved == 0, "QSA interface owner leaked")
    }

    /// Reuses the existing stronger HC fixture and expected captures; allocates
    /// only the new interface backing under an additional bounded reservation.
    package static func hcTile(
        helper: FusedHCDecode, ledger: MemoryLedger, residual: [Float], normWeights: MTLBuffer,
        down: FusedHCDecode.Projection, up: FusedHCDecode.Projection,
        injection: MTLBuffer, expected: FusedHCDecode.Result
    ) throws {
        let reserve = try ledger.reserve(bytes: 96 * 1024 * 1024, category: "whole_hc_interface_fixture")
        defer { try? ledger.release(reserve) }
        try autoreleasepool {
            let rows = residual.count / 10240
            let device = normWeights.device
            guard let queue = device.makeCommandQueue() else { throw EngineError("Metal unavailable") }
            func output(_ bytes: Int) throws -> MTLBuffer { try Check.buffer([UInt32](repeating: 0, count: max(1, bytes / 4)), device) }
            let partial = try helper.tilePartialBytes(rows: rows)
            let s = try FusedHCDecode.RowBuffers(
                norm: output(rows * 10240 * 4), normBF16: output(rows * 10240 * 2),
                down: output(rows * 320 * 4), activation: output(rows * 320 * 2), rawUp: output(rows * 10240 * 4),
                mixed: output(rows * 2560 * 4), exceptions: output(rows * 80 * 4), rawInjection: output(rows * 16),
                injection: output(rows * 16), capturedUp: output(rows * 10240 * 4))
            let input = try Check.buffer(residual.map(BF16.encode), device)
            let command = queue.makeCommandBuffer()!
            try helper.encodeTile(
                command, rows: rows, residualBF16: input, normWeights: normWeights,
                downWeights: down, upWeights: up, injectionWeights: injection, buffers: s,
                downPartials: partial.down > 0 ? output(partial.down) : nil,
                upPartials: partial.up > 0 ? output(partial.up) : nil,
                injectionPartials: partial.injection > 0 ? output(partial.injection) : nil)
            try Check.joined(command)
            try Check.require(
                (0..<(rows * 80)).allSatisfy({ s.exceptions.contents().assumingMemoryBound(to: UInt32.self)[$0] == 0 }),
                "HC tile interface required repair")
            for (b, values, label) in [
                (s.norm, expected.norm!, "norm"), (s.down, expected.down!, "down"),
                (s.rawUp, expected.rawUp!, "rawUp"), (s.mixed, expected.mixed, "mixed"), (s.capturedUp!, expected.sigmoidUp!, "sigmoid"),
                (s.rawInjection, expected.rawInjection!, "rawInjection"), (s.injection, expected.injection, "injection"),
            ] {
                try Check.equal(Check.values(b, values.count), values, "HC tile interface " + label)
            }
            let activation = Array(
                UnsafeBufferPointer(start: s.activation.contents().assumingMemoryBound(to: UInt16.self), count: rows * 320)
            ).map(BF16.decode)
            try Check.equal(activation, expected.activated!, "HC tile activation")
        }
    }
}
