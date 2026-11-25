import Darwin
import Dispatch
import Foundation
import Metal
import OrigamiEngine

func delayedExpertChecks() -> [(String, () throws -> Void)] {
    [
        ("delayed_expert_policy_and_modes", delayedExpertPolicyAndModes),
        ("delayed_expert_hc_reference", delayedExpertHCReference),
        ("delayed_expert_family_kernel", delayedExpertFamilyKernel),
        ("delayed_expert_storage_ownership", delayedExpertStorageOwnership),
        ("delayed_expert_read_lifecycle", delayedExpertReadLifecycle),
    ]
}

private func delayedRequire(_ condition: Bool, _ message: String) throws {
    guard condition else { throw EngineError("Delayed expert: " + message) }
}
private func delayedRefuses(_ body: () throws -> Void) -> Bool {
    do {
        try body()
        return false
    } catch { return true }
}

private func delayedExpertPolicyAndModes() throws {
    let policy = try DelayedExpertPolicy().validated()
    let gates: [Float] = [0.9, 0.04, 0.04, 0.02]
    try delayedRequire(
        policy.rank(weights: gates, late: [0, 1, 2, 3], capacity: 1) == 3,
        "smallest admissible original mass was not selected")
    try delayedRequire(
        policy.rank(weights: gates, late: [0, 1, 2], capacity: 1) == 1,
        "equal-mass tie is not deterministic")
    try delayedRequire(
        policy.rank(weights: gates, late: [0, 1, 2, 3], resident: [3], capacity: 1) == 1,
        "resident expert was delayed")
    try delayedRequire(
        policy.rank(weights: gates, late: [0], capacity: 1) == nil
            && policy.rank(weights: gates, late: [3], capacity: 0) == nil
            && policy.rank(weights: [.nan, 0.01], late: [1], capacity: 1) == nil
            && policy.rank(weights: [0, 0], late: [1], capacity: 1) == nil,
        "largest/invalid/no-capacity case was admitted")
    try delayedRequire(
        policy.rank(weights: gates.map { $0 * 8 }, late: [3], capacity: 1) == 3,
        "selection depends on the trained gate scale")
    for invalid in [
        DelayedExpertPolicy(version: 2), DelayedExpertPolicy(maximumMass: 0.11),
        DelayedExpertPolicy(maximumExpertMass: .nan), DelayedExpertPolicy(maximumExperts: 2),
    ] {
        try delayedRequire(delayedRefuses { _ = try invalid.validated() }, "invalid bounded policy accepted")
    }
    try delayedRequire(delayedRefuses { _ = try QualityDial(delayedExperts: policy).validated() }, "exact mode accepted deferral")
    var faster = QualityDial.level("faster")!
    var dynamic = QualityDial.level("dynamic")!
    faster.delayedExperts = policy
    dynamic.delayedExperts = policy
    try delayedRequire(
        try faster.validated() == faster && faster.name == "fast" && !faster.dropsExperts
            && faster.keptRanks(gates) == Array(gates.indices), "base mode changed or original gates were dropped")
    try delayedRequire(
        dynamic.step(after: [10, 0]).delayedExperts == policy
            && dynamic.step(after: [0, 0]) == .exact, "dynamic confidence gate leaked the policy into exact steps")
    try delayedRequire(
        faster.signature != QualityDial.level("faster")!.signature
            && FastLateMissSkip.timelineEpoch("base", quality: faster)
                != FastLateMissSkip.timelineEpoch("base", quality: QualityDial.level("faster")!),
        "approximate history identity did not change")
    let saved = #"{"tailSkipMissesOnly":true,"tailSkip":0.1,"lateMiss":"wait"}"#
    try delayedRequire(
        try JSONDecoder().decode(QualityDial.self, from: Data(saved.utf8)).delayedExperts == nil
            && JSONDecoder().decode(QualityDial.self, from: JSONEncoder().encode(faster)) == faster, "policy JSON compatibility differs")
    let exact = FamilyRouting.select(probs: gates, bias: [], k: 4, dial: .exact, resident: nil, normalize: true, scale: 2, sum: { $0 })
    let delayed = FamilyRouting.select(probs: gates, bias: [], k: 4, dial: faster, resident: [0], normalize: true, scale: 2, sum: { $0 })
    try delayedRequire(
        exact.map(\.expert) == delayed.map(\.expert) && exact.map(\.weight) == delayed.map(\.weight),
        "family deferral changed the original router gates or renormalized the immediate set")
}

/// Freeze the next attention branch and its HC coefficients. Transport the source delta through its
/// residual map only; this deliberately does not claim equivalence with a replayed full network.
private func delayedExpertHCReference() throws {
    let expert: [Float] = [0.25, -2, 3]
    let post: [Float] = [0.5, 1, -0.25, 2]
    let gate: Float = 0.03125
    let streams = (0..<12).map { Float($0 - 3) / 8 }
    let combine: [Float] = [
        0.5, 0.25, 0.125, 0.125, 0.1, 0.6, 0.2, 0.1,
        0.2, 0.1, 0.6, 0.1, 0.125, 0.125, 0.25, 0.5,
    ]
    for family in [DelayedExpertReference.Family.qwen, .glm, .deepseek] {
        let zero = try DelayedExpertReference.inject(
            streams, expert: expert, gate: 0, sourcePost: post, family: family, nextCombine: combine)
        try delayedRequire(zero == streams, "no-delay reference changed streams for \(family)")
        let actual = try DelayedExpertReference.inject(
            streams, expert: expert, gate: gate, sourcePost: post, family: family, nextCombine: combine)
        var expected = streams
        for d in 0..<4 {
            for j in expert.indices {
                if family == .qwen {
                    expected[d * 3 + j] = LayerArithmetic.round(
                        streams[d * 3 + j]
                            + LayerArithmetic.round(LayerArithmetic.round(expert[j] * gate) * post[d]))
                } else {
                    // Explicit source-stream transport checks the column-major combine convention.
                    var contribution: Float = 0
                    for s in 0..<4 { contribution += combine[d + 4 * s] * post[s] * (expert[j] * gate) }
                    expected[d * 3 + j] += contribution
                }
            }
        }
        try delayedRequire(zip(actual, expected).allSatisfy { abs($0 - $1) < 1e-6 }, "HC transport differs for \(family)")
        let twice = try DelayedExpertReference.inject(
            actual, expert: expert, gate: gate, sourcePost: post, family: family, nextCombine: combine)
        try delayedRequire(twice != actual, "fixture cannot detect a double application")
    }
    try delayedRequire(
        delayedRefuses { _ = try DelayedExpertReference.inject([1], expert: [1], gate: 1, sourcePost: post, family: .qwen) },
        "flattened/wrong stream geometry accepted")
}

private func delayedExpertFamilyKernel() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
        let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else {
        throw FixtureUnavailable(description: "Metal unavailable")
    }
    let kernels = try FamilyKernels(device: device)
    let expert: [Float] = [0.25, -2, 3]
    let gate: Float = 0.03125
    let post: [Float] = [0.5, 1, -0.25, 2]
    let streams = (0..<12).map { Float($0 - 3) / 8 }
    let combine = (0..<16).map { Float(($0 * 7 + 3) % 11) / 16 }
    func buffer(_ values: [Float]) throws -> MTLBuffer {
        guard
            let value = values.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        else {
            throw EngineError("No fixture buffer")
        }
        return value
    }
    let delta = try buffer(expert.map { $0 * gate })
    let sourcePost = try buffer(post)
    let c = try buffer(combine)
    let target = try buffer(streams)
    try kernels.dispatch(
        encoder, "hc_delayed_add", [.init(delta), .init(sourcePost), .init(c), .init(target)],
        ints: [3], grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    let actual = Array(UnsafeBufferPointer(start: target.contents().assumingMemoryBound(to: Float.self), count: 12))
    for family in [DelayedExpertReference.Family.glm, .deepseek] {
        let reference = try DelayedExpertReference.inject(
            streams, expert: expert, gate: gate, sourcePost: post, family: family, nextCombine: combine)
        try delayedRequire(
            zip(actual, reference).allSatisfy { abs($0 - $1) < 1e-6 }, "shared family kernel differs from \(family) reference")
    }
}

private func delayedExpertStorageOwnership() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    // A synthetic zero footprint keeps admission off the host's real process footprint; ownership remains measured exactly.
    let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    var escaped: MTLBuffer?
    var escapedBytes: UInt64 = 0
    try autoreleasepool {
        let slot = try DelayedExpertStorage(device: device, ledger: ledger, hidden: 8, footprint: 0)
        try delayedRequire(
            Set([slot.input, slot.output, slot.post].map { ObjectIdentifier($0) }).count == 3,
            "pending activations alias a live contribution")
        let expectedBytes = UInt64(3 * getpagesize())
        try delayedRequire(
            slot.allocatedBytes == expectedBytes,
            "optional physical extent \(slot.allocatedBytes) differs from three VM pages \(expectedBytes)")
        try delayedRequire(
            ledger.reservedBytes == slot.allocatedBytes,
            "optional reservation \(ledger.reservedBytes) differs from owned physical extent \(slot.allocatedBytes)")
        escaped = slot.input
        escapedBytes = UInt64(slot.input.allocatedSize)
    }
    try withExtendedLifetime(escaped) {
        try delayedRequire(
            escaped != nil && ledger.reservedBytes == escapedBytes,
            "escaped final Metal owner charge \(ledger.reservedBytes) differs from its physical extent \(escapedBytes)")
    }
    autoreleasepool { escaped = nil }
    try delayedRequire(ledger.reservedBytes == 0, "last pending owner leaked its reservation")
    try autoreleasepool {
        let slot = try DelayedExpertStorage(device: device, ledger: ledger, hidden: Int(getpagesize()) / 4 + 1, footprint: 0)
        let expectedBytes = UInt64(5 * getpagesize())  // Two pages each for input/output, one for post gates.
        try delayedRequire(
            slot.allocatedBytes == expectedBytes && ledger.reservedBytes == expectedBytes,
            "cross-page extent \(slot.allocatedBytes), reservation \(ledger.reservedBytes), expected \(expectedBytes)")
    }
    try delayedRequire(ledger.reservedBytes == 0, "cross-page slot leaked its reservation")
    let tiny = try MemoryLedger(ceiling: 2, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    try delayedRequire(
        delayedRefuses { _ = try DelayedExpertStorage(device: device, ledger: tiny, hidden: 8, footprint: 0) }
            && tiny.reservedBytes == 0, "capacity refusal leaked or bypassed the cap")
}

/// The actual demand-window transfer keeps one read alive after its source window closes. Its record is
/// pinned only while the consumer owns it; double application, read errors, cancellation and throw cleanup
/// exercise the same production owner, with a semaphore instead of real disk latency.
private func delayedExpertReadLifecycle() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    final class Reads: @unchecked Sendable {
        let release = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var failedReadEntered = false
        func run(_ key: ExpertKey, _ bytes: UnsafeMutableRawBufferPointer) throws {
            if key.expert == 9 {
                if key.layer == 3 {
                    lock.lock()
                    let retry = failedReadEntered
                    failedReadEntered = true
                    lock.unlock()
                    if retry { throw EngineError("Injected delayed retry failure", code: "fixture_read") }
                }
                guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Delayed fixture read timed out") }
                if key.layer == 3 { throw EngineError("Injected delayed read failure", code: "fixture_read") }
            }
            bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
        }
    }
    let ledger = try MemoryLedger(ceiling: 500_000_000, transientMargin: 1_000_000)
    try autoreleasepool {
        let reads = Reads()
        let config = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 32, slotsPerSlab: 8, demandWindow: 16, readConcurrency: 2)
        let execution = try ExpertExecution(
            device: device, ledger: ledger, configuration: config, fastSkipReserve: 1, packedAvailable: { false }
        ) { key, _, bytes in
            try reads.run(key, bytes)
        }
        let retained = execution.makeDelayedRead()
        for layer in 0..<4 {
            try execution.beginSequence()
            let warm = try execution.makeWindow(layer: layer, experts: Array(0..<9))
            for expert in 0..<9 { try warm.consume(expert: expert) { _ in } }
            try execution.beginDecodePhase()
            let window = try execution.makeWindow(layer: layer, experts: Array(0..<10))
            try window.consumeTenReadyFirst(
                experts: Array(0..<10),
                skipLate: { late, capacity in
                    try delayedRequire(late == [9] && capacity == 1, "fixture did not retain exactly one late read")
                    return [9]
                }, delayed: retained, ready: { _, _ in }, finish: { (_: [Int], _: [ExpertBuffer]) -> Void in })
            try delayedRequire(window.isClosed && retained.isPending, "source window discarded its delayed read")
            reads.release.signal()
            if layer == 0 {
                var applications = 0
                try retained.consume(shouldContinue: { true }) { record in
                    applications += 1
                    try delayedRequire(record.bytes[0] == 9 && execution.snapshot().cache.pinnedSlots == 1, "wrong/unpinned source record")
                }
                try delayedRequire(
                    applications == 1 && !retained.isPending
                        && delayedRefuses { try retained.consume(shouldContinue: { true }) { _ in applications += 1 } }
                        && applications == 1, "contribution could be consumed twice")
            } else if layer == 1 {
                try delayedRequire(
                    delayedRefuses { try retained.consume(shouldContinue: { true }) { _ in throw EngineError("Injected consumer error") } },
                    "consumer error was hidden")
            } else if layer == 2 {
                var called = false
                try delayedRequire(
                    delayedRefuses { try retained.consume(shouldContinue: { false }) { _ in called = true } } && !called,
                    "cancellation was hidden or the consumer ran")
            } else {
                var called = false
                try delayedRequire(
                    delayedRefuses { try retained.consume(shouldContinue: { true }) { _ in called = true } } && !called,
                    "read failure was hidden or the consumer ran")
            }
            retained.discard()
            retained.discard()
            try execution.finishSequence()
            let snapshot = execution.snapshot()
            try delayedRequire(
                !retained.isPending && snapshot.cache.pinnedSlots == 0 && snapshot.cache.inFlightReads == 0,
                "joined request left a read or pin after lifecycle case \(layer)")
        }
    }
    for _ in 0..<2_000 where ledger.reservedBytes > 0 { Thread.sleep(forTimeInterval: 0.001) }
    try delayedRequire(ledger.reservedBytes == 0, "delayed read lifecycle leaked its owner")
}
