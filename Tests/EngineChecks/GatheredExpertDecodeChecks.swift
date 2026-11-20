import CryptoKit
import Darwin
import Dispatch
import Foundation
import Metal
import OrigamiEngine

private let gatheredRanks = [3, 8, 0, 9, 1, 6, 4, 2, 7, 5]
private func gatheredRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try value() else { throw EngineError(message) }
}
private func gatheredReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Gathered invalid operation unexpectedly succeeded")
}
private struct GatheredGenerator {
    var seed: UInt64
    mutating func next() -> UInt32 {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt32(truncatingIfNeeded: seed >> 32)
    }
    mutating func bf(_ exponent: UInt32) -> UInt16 {
        let b = next()
        return UInt16(truncatingIfNeeded: ((b & 0x807f_0000) | ((exponent + ((b >> 21) % 9)) << 23)) >> 16)
    }
}
private func gatheredInput(seed: UInt64, count: Int) -> [Float] {
    var g = GatheredGenerator(seed: seed)
    return (0..<count).map { _ in BF16.decode(g.bf(120)) }
}
private func gatheredEngine(_ ledger: MemoryLedger, failureExpert: Int? = nil) throws -> ExpertExecution {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    return try ExpertExecution(
        device: device, ledger: ledger,
        configuration: .init(
            mode: .slabPackedPreferred,
            cacheSlots: 12, slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2), packedAvailable: { false }
    ) { key, _, bytes in
        if key.expert == failureExpert { throw EngineError("Injected gathered producer failure") }
        let rank = gatheredRanks[key.expert]
        for (index, kind) in ExpertProjection.allCases.enumerated() {
            var g = GatheredGenerator(seed: UInt64(43000 + index * 1000 + rank * 193))
            let words = bytes.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
            let scales = bytes.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
            let biases = bytes.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(kind.width * kind.outputs / 8) { words[i] = g.next() }
            for i in 0..<(kind.width * kind.outputs / ModelLayout.experts.group) { scales[i] = g.bf(116) }
            for i in 0..<(kind.width * kind.outputs / ModelLayout.experts.group) { biases[i] = g.bf(115) }
        }
    }
}
private func gatheredRetired(_ engine: ExpertExecution, _ ledger: MemoryLedger) throws {
    let s = engine.snapshot()
    try gatheredRequire(
        s.cache.pinnedSlots == 0 && s.cache.reservedSlots == 0 && s.cache.inFlightReads == 0
            && s.io.drives.allSatisfy { $0.outstanding == 0 }, "Gathered producer/consumer survived joined scope")
    try gatheredRequire(ledger.snapshot().categories["expert-consumer-batch-metadata"] == nil, "Gathered batch metadata leaked")
}
private func gatheredHash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

func gatheredExpertConfigurationAndStartup() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("config.json")
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    try gatheredRequire(try !load([:]).gatheredExpertDecode, "Gathered decode default enabled")
    let valid: [String: Any] = [
        "gathered_expert_decode": true, "fused_experts": true, "pinned_reductions": true,
        "expert_mode": "slabPackedPreferred", "expert_cache_slots": 16, "expert_slots_per_slab": 2, "expert_window": 10,
        "io_queue_depth": 2,
    ]
    let configuration = try load(valid)
    try gatheredRequire(!configuration.moeSharedReadOverlap, "Shared read overlap enabled by default")
    var overlap = valid
    overlap["moe_shared_read_overlap"] = true
    try gatheredRequire(try load(overlap).moeSharedReadOverlap, "Shared read overlap ignored")
    overlap["gathered_expert_decode"] = false
    try gatheredReject { _ = try load(overlap) }
    overlap = valid
    overlap["moe_shared_read_overlap"] = 1
    try gatheredReject { _ = try load(overlap) }
    for key in ["fused_experts", "pinned_reductions"] {
        var changed = valid
        changed[key] = false
        try gatheredReject { _ = try load(changed) }
    }
    var short = valid
    short["expert_window"] = 8
    try gatheredReject { _ = try load(short) }
    var malformed = valid
    malformed["gathered_expert_decode"] = 1
    try gatheredReject { _ = try load(malformed) }
}

/// The gathered ten-expert decode equals the fused per-expert path bit for bit (QMV, tile8 and the
/// weighted reduction), without frozen references; registered for the group-64 and group-32 layouts.
func gatheredExpertMatchesFusedExperts() throws {
    let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reducer = try PinnedReductions(ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let helper = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reducer)
        let engine = try gatheredEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
        // consumeBatch keeps its eight-record limit; ten records need consumeTen.
        try gatheredReject { try window.consumeBatch(experts: Array(0..<10)) { _ in } }
        try gatheredRequire(!window.isClosed && window.remainingCount == 10, "Invalid batch consumed a pending handle")
        try window.consumeTen(experts: Array(0..<10)) { buffers in
            try gatheredRequire(
                buffers.contains { $0.offset > 0 } && Set(buffers.map { ObjectIdentifier($0.buffer) }).count > 1,
                "Gather fixture missed nonzero offsets/noncontiguous slabs")
            let records = buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
            let before = buffers.map { gatheredHash(Data($0.bytes)) }
            let x = gatheredInput(seed: 42001, count: 2560)
            var original: [String: [[Float]]] = [:]
            for label in ["gate", "up", "product", "down"] { original[label] = .init(repeating: [], count: 10) }
            for start in [0, 8] {
                let jobs = buffers[start..<min(start + 8, 10)].map { FusedExpertExecution.Job(input: x, buffer: $0) }
                try fused.evaluateBatch(
                    jobs,
                    capture: { index, label, values in
                        original[label]![gatheredRanks[start + index]] = values
                    }
                ) { index, values in original["down"]![gatheredRanks[start + index]] = values }
            }
            let group = ModelLayout.group
            for (label, kind) in zip(["gate", "up", "down"], ExpertProjection.allCases) {
                let input = kind == .down ? original["product"]!.flatMap { $0 } : x
                let actual = try helper.diagnosticProjection(input, records: records, canonicalRanks: gatheredRanks, kind: kind)
                try gatheredRequire(
                    actual.map(\.bitPattern) == original[label]!.flatMap { $0 }.map(\.bitPattern),
                    "Gathered QMV \(label) differs from the fused experts at group \(group)")
            }
            let scores: [Float] = (0..<10).map { Float($0 + 1) / 55 }
            let expected = try reducer.weightedExpertReduce10(experts: original["down"]!.flatMap { $0 }, weights: scores)
            let actual = try helper.evaluate(
                x, records: records, canonicalRanks: gatheredRanks, weights: scores,
                capture: { label, values in
                    try gatheredRequire(
                        values.map(\.bitPattern) == original[label]!.flatMap { $0 }.map(\.bitPattern),
                        "Gathered composed \(label) differs from the fused experts at group \(group)")
                })
            try gatheredRequire(actual.map(\.bitPattern) == expected.map(\.bitPattern), "Gathered reduction differs at group \(group)")
            try gatheredRequire(buffers.map { gatheredHash(Data($0.bytes)) } == before, "Gathered reader mutated an expert record")
            // The production tile8 kernels (ready, resident-first and all-ten groups): empty, single and sparse
            // non-contiguous first groups give the canonical bits despite descriptor reuse after a join.
            for first in [[], [7], [0, 2, 4, 6, 8], Array(1..<10), Array(0..<10)] {
                let split = try helper.diagnosticReadySplit(
                    x, records: records, canonicalRanks: gatheredRanks,
                    weights: scores, firstIndices: first, independentBindings: true)
                try gatheredRequire(
                    split.map(\.bitPattern) == expected.map(\.bitPattern),
                    "Tile8 split \(first) differs from the fused experts at group \(group)")
            }
            // Cold records can arrive in several out-of-order groups. Descriptors are joined before reuse;
            // every group writes its canonical span and the final command performs exactly one reduction.
            for groups in [[[4], [7], [0, 3, 8], [1, 2, 5, 6, 9]], (0..<10).reversed().map { [$0] }] {
                let arrival = try helper.diagnosticReadySplit(
                    x, records: records, canonicalRanks: gatheredRanks,
                    weights: scores, firstIndices: [], groups: groups)
                try gatheredRequire(
                    arrival.map(\.bitPattern) == expected.map(\.bitPattern),
                    "Canonical arrival groups differ from fused experts at group \(group)")
            }
            let reads = helper.snapshot().floatReadbackBytes
            let repeated = try helper.evaluate(x, records: records, canonicalRanks: gatheredRanks, weights: scores)
            try gatheredRequire(
                repeated.map(\.bitPattern) == expected.map(\.bitPattern) && helper.snapshot().floatReadbackBytes - reads == 10240,
                "Uncaptured gathered decode read expert outputs or changed scratch reuse")
            try gatheredReject {
                _ = try helper.evaluate(x, records: records, canonicalRanks: .init(repeating: 0, count: 10), weights: scores)
            }
            for stopAt in [1, 2, 3] {
                let count = helper.snapshot().submittedCommands
                var calls = 0
                do {
                    _ = try helper.evaluate(
                        x, records: records, canonicalRanks: gatheredRanks, weights: scores,
                        shouldContinue: {
                            calls += 1
                            return calls < stopAt
                        })
                    throw EngineError("Gathered cancellation succeeded")
                } catch is IOOperationCancelled {}
                let delta: UInt64 = stopAt == 3 ? 1 : 0
                try gatheredRequire(
                    helper.snapshot().submittedCommands == count + delta && helper.snapshot().completedCommands == count + delta,
                    "Gathered cancellation returned before all readers joined")
            }
            try helper.trim()
            try gatheredRequire(
                ledger.snapshot().categories["gathered-expert-decode"] == nil, "Gathered scratch/descriptor retained after trim")
        }
        try gatheredRetired(engine, ledger)
        try engine.finishSequence()
    }
    try gatheredRequire(ledger.snapshot().reserved == 0, "Gathered fixture leaked owners")
}

/// Queued-experts decode: a layer's experts queued before their records are known compute the ready group's bits.
func gatheredDeferredGroupParity() throws {
    let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
        let reducer = try PinnedReductions(ledger: ledger)
        let fused = try FusedExpertExecution(arithmetic: kernels.arithmetic, ledger: ledger, scratchReuse: true)
        let helper = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reducer)
        let engine = try gatheredEngine(ledger)
        try engine.beginSequence()
        let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
        try window.consumeTen(experts: Array(0..<10)) { buffers in
            let records = buffers.map { GatheredExpertDecode.Record(buffer: $0.buffer, offset: $0.offset) }
            try helper.deferredGroupParity(
                records: Array(records.prefix(7)), canonicalRanks: Array(gatheredRanks.prefix(7)),
                input: gatheredInput(seed: 42002, count: 2560))
            try helper.trim()
        }
        try gatheredRetired(engine, ledger)
        try engine.finishSequence()
    }
    try gatheredRequire(ledger.snapshot().reserved == 0, "Deferred group fixture leaked owners")
}

func gatheredExpertFailureAndAdmissionPinOwnership() throws {
    let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
    let failed = try gatheredEngine(ledger, failureExpert: 5)
    try failed.beginSequence()
    let partial = try failed.makeWindow(layer: 0, experts: Array(0..<10))
    var entered = false
    try gatheredReject { try partial.consumeTen(experts: Array(0..<10)) { _ in entered = true } }
    try gatheredRequire(!entered && partial.isClosed, "Failed producer entered the gathered body")
    try gatheredRetired(failed, ledger)
    try failed.finishSequence()
    let engine = try gatheredEngine(ledger)
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
    let small = try MemoryLedger(ceiling: 180_000, transientMargin: 1)
    try autoreleasepool {
        let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: small)
        let reducer = try PinnedReductions(ledger: small)
        let helper = try GatheredExpertDecode(ledger: small, fused: fused, reductions: reducer)
        let before = small.snapshot().reserved
        try gatheredReject {
            try window.consumeTen(experts: Array(0..<10)) { buffers in
                _ = try helper.evaluate(
                    .init(repeating: 1, count: 2560), records: buffers.map { .init(buffer: $0.buffer, offset: $0.offset) },
                    canonicalRanks: gatheredRanks, weights: .init(repeating: 0.1, count: 10))
            }
        }
        try gatheredRequire(
            window.isClosed && helper.snapshot().submittedCommands == 0 && small.snapshot().reserved == before,
            "Denied gathered allocation published or leaked partial scratch")
    }
    try gatheredRetired(engine, ledger)
    try engine.finishSequence()
    try engine.beginSequence()
    let callbackWindow = try engine.makeWindow(layer: 0, experts: Array(0..<10))
    let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger)
    let reducer = try PinnedReductions(ledger: ledger)
    let helper = try GatheredExpertDecode(ledger: ledger, fused: fused, reductions: reducer)
    try gatheredReject {
        try callbackWindow.consumeTen(experts: Array(0..<10)) { buffers in
            _ = try helper.evaluate(
                .init(repeating: 0.125, count: 2560), records: buffers.map { .init(buffer: $0.buffer, offset: $0.offset) },
                canonicalRanks: gatheredRanks, weights: .init(repeating: 0.1, count: 10),
                capture: { _, _ in
                    try gatheredReject { try helper.trim() }
                    throw EngineError("Injected gathered capture failure")
                })
        }
    }
    try gatheredRequire(
        callbackWindow.isClosed && helper.snapshot().submittedCommands == 1 && helper.snapshot().completedCommands == 1
            && helper.snapshot().retainedBytes == 0, "Capture failure failed to join/retire cold gathered scratch")
    try gatheredRetired(engine, ledger)
    try engine.finishSequence()
}

/// Shared projection failure occurs before consumeTen obtains consumer pins.
/// Pending producers must still be joined before their handoff storage retires.
func gatheredSharedFailureBeforeConsumeJoinsProducers() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 16,
        slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2)
    let engine = try ExpertExecution(device: device, ledger: ledger, configuration: config, packedAvailable: { false }) {
        _, _, destination in
        started.signal()
        guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Shared-failure producer timeout") }
        destination.initializeMemory(as: UInt8.self, repeating: 0x5a)
    }
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 0, experts: Array(0..<10))
    guard started.wait(timeout: .now() + 5) == .success else {
        for _ in 0..<10 { release.signal() }
        window.close()
        throw EngineError("No producer started before shared failure")
    }
    let group = DispatchGroup()
    group.enter()
    DispatchQueue.global().async {
        Thread.sleep(forTimeInterval: 0.05)
        for _ in 0..<10 { release.signal() }
        group.leave()
    }
    do {
        defer { window.close() }
        throw EngineError("Injected shared projection failure before consume")
    } catch let error as EngineError {
        try gatheredRequire(error.description == "Injected shared projection failure before consume", "Wrong shared failure")
    }
    group.wait()
    try gatheredRequire(window.isClosed && engine.snapshot().consumedExperts == 0, "Shared failure consumed a routed expert")
    try gatheredRetired(engine, ledger)
    try engine.finishSequence()
    // A later sequence must safely reuse the retired slots.
    try engine.beginSequence()
    let reused = try engine.makeWindow(layer: 0, experts: [0])
    try reused.consume(expert: 0) { buffer in
        try gatheredRequire(buffer.bytes[0] == 0x5a, "Shared-failure cleanup exposed stale storage")
    }
    try gatheredRetired(engine, ledger)
    try engine.finishSequence()
}

func residentFirstDemandJoinsAndRetires() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    for scenario in ["complete", "early-failure", "late-failure", "cancel", "read-failure"] {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let release = DispatchSemaphore(value: 0)
            let engine = try ExpertExecution(
                device: device, ledger: ledger,
                configuration: .init(
                    mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2,
                    demandWindow: 10, readConcurrency: 2), packedAvailable: { false }
            ) { key, _, bytes in
                if key.expert % 2 == 1 {
                    guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Resident fixture timed out") }
                    if scenario == "read-failure", key.expert == 3 { throw EngineError("Injected missing expert failure") }
                }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try engine.beginSequence()
            let even = stride(from: 0, to: 10, by: 2).map { $0 }
            let prime = try engine.makeWindow(layer: 0, experts: even)
            for expert in even { try prime.consume(expert: expert) { _ in } }
            let gpuOwner = try ledger.reserve(bytes: 65536, category: "resident_cleanup_gpu_fixture")
            defer { try? ledger.release(gpuOwner) }
            guard let queue = device.makeCommandQueue(), let gate = device.makeSharedEvent(),
                let output = device.makeBuffer(length: 16384)
            else { throw EngineError("Cleanup fixture Metal unavailable") }
            var earlyCommand: MTLCommandBuffer?
            var cancelled = false
            var enteredReady = false
            var enteredFinish = false
            var cleanupCalls = 0
            var cleanupHadPins = false
            var gpuRetired = false
            let window = try engine.makeWindow(layer: 0, experts: Array(0..<10), shouldContinue: { !cancelled })
            var failed = false
            do {
                try window.consumeTenReadyFirst(
                    experts: Array(0..<10),
                    joinReaders: {
                        cleanupCalls += 1
                        cleanupHadPins = engine.snapshot().cache.pinnedSlots >= 5
                        gate.signaledValue = 1
                        earlyCommand?.waitUntilCompleted()
                        gpuRetired = earlyCommand?.status == .completed && output.contents().load(as: UInt32.self) == 0
                    },
                    ready: { ids, buffers in
                        enteredReady = true
                        try gatheredRequire(ids == even && buffers.count == 5, "Resident group changed acquisition classification")
                        for (id, b) in zip(ids, buffers) { try gatheredRequire(b.bytes[0] == UInt8(id), "Resident bytes differ") }
                        try gatheredRequire(engine.snapshot().cache.pinnedSlots >= 5, "Resident callback lacks pins")
                        let command = queue.makeCommandBuffer()!
                        command.encodeWaitForEvent(gate, value: 1)
                        let blit = command.makeBlitCommandEncoder()!
                        blit.copy(from: buffers[0].buffer, sourceOffset: buffers[0].offset, to: output, destinationOffset: 0, size: 4)
                        blit.endEncoding()
                        earlyCommand = command
                        command.commit()
                        for _ in 0..<5 { release.signal() }
                        if scenario == "early-failure" { throw EngineError("Injected early compute failure") }
                        if scenario == "cancel" { cancelled = true }
                    },
                    finish: { ids, buffers in
                        enteredFinish = true
                        try gatheredRequire(ids == [1, 3, 5, 7, 9] && buffers.count == 5, "Remaining expert group differs")
                        for (id, b) in zip(ids, buffers) { try gatheredRequire(b.bytes[0] == UInt8(id), "Remaining bytes differ") }
                        try gatheredRequire(engine.snapshot().cache.pinnedSlots == 10, "Early pins retired before final reader")
                        if scenario == "late-failure" { throw EngineError("Injected final compute failure") }
                    })
            } catch {
                // An assertion inside an expected-failure callback must still
                // fail the check; accept only the deliberately injected error.
                let expected: String?
                switch scenario {
                case "early-failure": expected = "Injected early compute failure"
                case "late-failure": expected = "Injected final compute failure"
                case "read-failure": expected = "Original expert read failed: Injected missing expert failure"
                default: expected = nil
                }
                if scenario == "cancel" {
                    guard error is IOOperationCancelled else { throw error }
                } else {
                    guard let expected, String(describing: error) == expected else { throw error }
                }
                failed = true
            }
            try gatheredRequire(cleanupCalls == 1 && cleanupHadPins && gpuRetired, "GPU cleanup did not precede pin retirement")
            try gatheredRequire(enteredReady && failed == (scenario != "complete"), "Resident-first failure outcome differs")
            try gatheredRequire(
                enteredFinish == (scenario == "complete" || scenario == "late-failure"), "Failed read/cancel entered final compute")
            try gatheredRetired(engine, ledger)
            try gatheredRequire(
                window.isClosed && engine.snapshot().consumedExperts == (scenario == "complete" ? 15 : 5),
                "Resident-first partial computation became committed consumption")
            try engine.finishSequence()
            if scenario == "complete" {
                try engine.beginSequence()
                let warm = try engine.makeWindow(layer: 0, experts: Array(0..<10))
                try warm.consumeTenReadyFirst(
                    experts: Array(0..<10),
                    ready: { ids, buffers in
                        try gatheredRequire(ids.isEmpty && buffers.isEmpty, "All-resident window was split unnecessarily")
                    },
                    finish: { ids, buffers in
                        try gatheredRequire(ids == Array(0..<10) && buffers.count == 10, "All-resident full group differs")
                    })
                try gatheredRequire(
                    engine.snapshot().residentFirstWindows == 1 && engine.snapshot().residentFirstExperts == 5,
                    "Resident-first group accounting differs")
                try gatheredRetired(engine, ledger)
                try engine.finishSequence()
            }
        }
        try gatheredRequire(ledger.snapshot().reserved == 0, "Resident-first owner leaked after \(scenario)")
    }
}
