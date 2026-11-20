import Foundation
import OrigamiEngine

private func checkpointRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func checkpointReject(_ body: () throws -> Void) throws {
    var rejected = false
    do { try body() } catch { rejected = true }
    try checkpointRequire(rejected, "Checkpoint invalid operation was accepted")
}
private func checkpointIdentity(_ epoch: String = "unit-fixture-v1") throws -> CheckpointIdentity {
    let hash = String(repeating: "a", count: 64)
    return try CheckpointIdentity(
        modelSHA256: hash, tokenizerSHA256: hash, templateSHA256: hash,
        quantizationSHA256: hash, arithmeticEpoch: epoch)
}

func checkpointFrontierAndSegmentationChecks() throws {
    for count in [0, 1, 3, 4, 5, 255, 256, 257, 2047, 2048, 2049, 32_768] {
        let ready = try CheckpointFrontier(consumedTokenCount: count, producedTokenCount: 0, emittedButUnconsumed: nil)
        try ready.validate(intent: .appendInput, hasLogits: false)
        try checkpointReject { try ready.validate(intent: .consumePending, hasLogits: true) }
        try checkpointReject { try ready.validate(intent: .sampleFrontier, hasLogits: false) }
        if count > 0 { try ready.validate(intent: .sampleFrontier, hasLogits: true) }
        let pending = try CheckpointFrontier(consumedTokenCount: count, producedTokenCount: 1, emittedButUnconsumed: 248_044)
        try pending.validate(intent: .consumePending, hasLogits: false)
        try checkpointReject { try pending.validate(intent: .sampleFrontier, hasLogits: true) }
        try checkpointReject { try pending.validate(intent: .appendInput, hasLogits: true) }
    }
    try checkpointReject { _ = try CheckpointFrontier(consumedTokenCount: 1, producedTokenCount: 0, emittedButUnconsumed: 48) }
    try checkpointReject { _ = try CheckpointFrontier(consumedTokenCount: 1, producedTokenCount: 2, emittedButUnconsumed: nil) }
    try checkpointReject { _ = try CheckpointFrontier(consumedTokenCount: 1, producedTokenCount: 1, emittedButUnconsumed: 248_320) }
    let hash = String(repeating: "b", count: 64)
    let short = try CheckpointSegment(
        start: 0, count: 24, projectionRows: 24, queryTile: 24, keyExtent: 24, phase: .prefill, dispatchSHA256: hash)
    let padded = try CheckpointSegment(
        start: 0, count: 24, projectionRows: 256, queryTile: 24, keyExtent: 24, phase: .prefill, dispatchSHA256: hash)
    try checkpointRequire(short != padded, "Projection schedule did not enter checkpoint identity")
    try checkpointRequire(
        checkpointIdentity("epoch-a") != checkpointIdentity("epoch-b"), "Arithmetic epoch did not enter checkpoint identity")
    try checkpointReject {
        _ = try CheckpointSegment(
            start: 0, count: 24, projectionRows: 1, queryTile: 24, keyExtent: 24, phase: .prefill, dispatchSHA256: hash)
    }
    try checkpointReject {
        _ = try CheckpointSegment(
            start: 0, count: 24, projectionRows: 24, queryTile: 24, keyExtent: 24, phase: .decode, dispatchSHA256: hash)
    }
}

/// One real-shaped ~116 MB checkpoint in memory: capture keeps raw logits bits, admission refuses a capture that does
/// not fit, and a restore with another identity or other token IDs leaves the target untouched. Mechanics only:
/// nonzero model restore+continue requires a model fixture and is intentionally not claimed by these CPU tests.
func checkpointCaptureAndRestoreChecks() throws {
    try checkpointArrayAdmissionChecks()
    let ledger = try MemoryLedger(ceiling: 1_200_000_000, transientMargin: 100_000_000)
    let identity = try checkpointIdentity()
    try withExtendedLifetime(ledger) {
        let source = try SequenceState(inputTokens: 1, outputTokens: 1, ledger: ledger, contextLimit: 2)
        var logits = [Float](repeating: 0, count: 248_320)
        logits[0] = Float(bitPattern: 0x8000_0000)  // preserve signed zero
        logits[1] = Float(bitPattern: 0x7fc0_0042)  // preserve NaN payload
        logits[2] = Float(bitPattern: 0xff80_0000)
        let checkpoint = try ModelCheckpoint.capture(
            state: source, consumedIDs: [], identity: identity,
            segments: [], nextLogits: logits, ledger: ledger)
        try checkpointRequire(
            checkpoint.nextLogits?.prefix(3).map(\.bitPattern) == logits.prefix(3).map(\.bitPattern),
            "Checkpoint did not preserve Float32 raw bits")
        let tinyLedger = try MemoryLedger(ceiling: 1_000_000, transientMargin: 1)
        try checkpointReject {
            _ = try ModelCheckpoint.capture(
                state: source, consumedIDs: [], identity: identity,
                segments: [], ledger: tinyLedger)
        }
        try checkpointRequire(tinyLedger.snapshot().reserved == 0, "Failed capture leaked reservation")

        let target = try SequenceState(inputTokens: 1, outputTokens: 1, ledger: ledger, contextLimit: 2)
        let targetID = target.id
        try checkpointReject {
            _ = try checkpoint.restore(
                into: target, expectedIdentity: checkpointIdentity("changed"),
                expectedSegments: [], expectedConsumedIDs: [], intent: .appendInput)
        }
        try checkpointReject {
            _ = try checkpoint.restore(
                into: target, expectedIdentity: identity,
                expectedSegments: [], expectedConsumedIDs: [1], intent: .appendInput)
        }
        try checkpointRequire(
            target.tokenCount == 0 && target.valid && target.id == targetID,
            "Rejected restore mutated target")
        _ = try checkpoint.restore(
            into: target, expectedIdentity: identity, expectedSegments: [], expectedConsumedIDs: [], intent: .appendInput)
        try target.validate()
    }
    try checkpointRequire(ledger.snapshot().reserved == 0, "Checkpoint and sequence lifetimes leaked reservations")
}

/// A synthetic cached block gives deterministic capacity inflation: reducing
/// count keeps native Array capacity, independent of Darwin cache/reclaim timing.
private func checkpointArrayAdmissionChecks() throws {
    let quota = CheckpointMemoryQuota(maximumBytes: 1000)
    let quotaLedger = try MemoryLedger(ceiling: 1000, transientMargin: 0)
    var charged: CheckpointMemoryLease? = try CheckpointMemoryLease(ledger: quotaLedger, bytes: 100, quota: quota, extraQuotaBytes: 700)
    try checkpointRequire(
        quota.liveBytes == 800 && quotaLedger.reservedBytes == 100,
        "Retained GPU state was omitted from quota or charged twice to the ledger")
    try checkpointReject { _ = try CheckpointMemoryLease(ledger: quotaLedger, bytes: 100, quota: quota, extraQuotaBytes: 101) }
    try charged!.shrink(to: 50)
    try checkpointRequire(quota.liveBytes == 750 && quotaLedger.reservedBytes == 50, "Checkpoint shrink lost retained GPU quota")
    charged = nil
    try checkpointRequire(quota.liveBytes == 0 && quotaLedger.reservedBytes == 0, "Tool anchor quota outlived its reader")
    let deniedAnchor = try MemoryLedger(ceiling: 10, transientMargin: 0)
    try checkpointReject { _ = try CheckpointMemoryLease(ledger: deniedAnchor, bytes: 20, quota: quota, extraQuotaBytes: 700) }
    try checkpointRequire(quota.liveBytes == 0, "Ledger denial leaked retained GPU quota")
    let count = 1024
    let metadata: UInt64 = 128
    let bound = try CheckpointMemoryLease.arrayAdmissionBytes(count: count, element: UInt16.self)
    let ledger = try MemoryLedger(ceiling: metadata + 8 * bound + 1, transientMargin: 1)
    try withExtendedLifetime(ledger) {
        let owner = try CheckpointMemoryLease(ledger: ledger, bytes: metadata)
        let inflated: [UInt16] = try owner.allocateArray(count: count) { requested in
            try checkpointRequire(
                ledger.snapshot().reserved == metadata + bound,
                "Checkpoint array allocated before worst-case admission")
            var block = [UInt16](repeating: 0, count: requested + requested / 2)
            block.removeLast(requested / 2)
            return block
        }
        let actual = UInt64(inflated.capacity * 2)
        try checkpointRequire(
            inflated.count == count && actual > UInt64(count * 2),
            "Allocator regression did not create an oversized retained block")
        try checkpointRequire(
            owner.bytes == metadata + actual && ledger.snapshot().reserved == owner.bytes,
            "Checkpoint array slack was not atomically returned")
        let second: [UInt16] = try owner.allocateArray(count: count) { requested in
            try checkpointRequire(
                ledger.snapshot().reserved == metadata + actual + bound,
                "Next array lost the earlier array's reservation")
            return [UInt16](repeating: 0, count: requested)
        }
        try checkpointRequire(
            owner.bytes == metadata + actual + UInt64(second.capacity * 2),
            "Checkpoint lease did not retain every measured capacity")
        let beforeFailure = ledger.snapshot().reserved
        try checkpointReject {
            let _: [UInt16] = try owner.allocateArray(count: count) { _ in
                throw EngineError("Injected checkpoint allocation failure")
            }
        }
        try checkpointRequire(
            ledger.snapshot().reserved == beforeFailure,
            "Partial checkpoint decode leaked its next-array reservation")
        // A count mismatch fails closed without changing existing ownership.
        try checkpointReject { let _: [UInt16] = try owner.allocateArray(count: count) { _ in [] } }
        try checkpointRequire(
            ledger.snapshot().reserved == beforeFailure,
            "Invalid checkpoint array geometry leaked admission")
        // The fixture owns a separate reservation for this deliberately
        // over-bound allocation; the decoder must reject it, never enlarge its
        // own reservation after allocation or publish the result.
        do {
            let fixtureToken = try ledger.reserve(bytes: 4 * bound, category: "checkpoint_bound_fixture")
            defer { try? ledger.release(fixtureToken) }
            var oversized = [UInt16](repeating: 0, count: Int(bound / 2) + 1)
            oversized.removeLast(oversized.count - count)
            try checkpointReject {
                let _: [UInt16] = try owner.allocateArray(count: count) { _ in oversized }
            }
            try checkpointRequire(
                ledger.snapshot().reserved == beforeFailure + 4 * bound,
                "Over-bound array changed checkpoint ownership")
            withExtendedLifetime(oversized) {}
        }
        withExtendedLifetime((owner, inflated, second)) {}
    }
    try checkpointRequire(ledger.snapshot().reserved == 0, "Per-array checkpoint leases outlived their owner")

    let denied = try MemoryLedger(ceiling: metadata + bound, transientMargin: 1)
    var allocationCalled = false
    do {
        let owner = try CheckpointMemoryLease(ledger: denied, bytes: metadata)
        try checkpointReject {
            let _: [UInt16] = try owner.allocateArray(count: count) { requested in
                allocationCalled = true
                return [UInt16](repeating: 0, count: requested)
            }
        }
        try checkpointRequire(
            !allocationCalled && denied.snapshot().reserved == metadata,
            "Denied checkpoint admission invoked its allocator")
    }
    try checkpointRequire(denied.snapshot().reserved == 0, "Denied checkpoint owner leaked metadata")

    let token = try ledger.reserve(bytes: 8, category: "checkpoint_shrink_test")
    try ledger.shrink(token, to: 3)
    try checkpointRequire(ledger.snapshot().reserved == 3, "Ledger shrink lost its live reservation")
    try checkpointReject { try ledger.shrink(token, to: 4) }
    try checkpointRequire(ledger.snapshot().reserved == 3, "Rejected shrink changed reservation")
    try ledger.release(token)
    try checkpointReject { try ledger.shrink(token, to: 0) }
    try checkpointRequire(ledger.snapshot().reserved == 0, "Ledger shrink leaked a token")
}
