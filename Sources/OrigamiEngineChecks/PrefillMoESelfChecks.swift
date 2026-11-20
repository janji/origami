import Darwin
import Foundation
import Metal
import OrigamiEngine

/// Checks of the device MoE prefill (`moe_device_prefill`): configuration dependencies, BF16 gather and FP32 scatter
/// bits over the whole BF16 domain, the canonical reduction (host and encoded), expert parity across tile and rank
/// boundaries, and escrow/permit lifetime through refusals, cancellation and escaped buffers.
package enum PrefillMoESelfChecks {
    private static func demand(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError(message) }
    }
    private static func rejects(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Invalid device MoE operation succeeded")
    }
    private static func rows(_ tokens: Int) -> [Int: [(Int, Int)]] {
        Dictionary(
            uniqueKeysWithValues: (0..<10).map { expert in
                (expert, (0..<tokens).map { ($0, (expert + $0) % 10) })
            })
    }
    package static func configuration() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("origami-device-moe-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: path) }
        func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
            try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
                to: path)
            return try RuntimeConfiguration.load(path)
        }
        try demand(try !load([:]).moeDevicePrefill, "Device MoE must default off")
        let valid: [String: Any] = [
            "expert_mode": "slabPackedPreferred", "fused_experts": true,
            "grouped_expert_prefill": true, "pinned_reductions": true, "expert_command_batch": 8, "moe_device_prefill": true,
        ]
        try demand(try load(valid).moeDevicePrefill, "Device MoE option was ignored")
        for key in ["fused_experts", "grouped_expert_prefill", "pinned_reductions"] {
            var bad = valid
            bad[key] = false
            try rejects { _ = try load(bad) }
        }
        for value in [1, "true"] as [Any] {
            var bad = valid
            bad["moe_device_prefill"] = value
            try rejects { _ = try load(bad) }
        }
        var bad = valid
        bad["expert_command_batch"] = 1
        try rejects { _ = try load(bad) }
    }
    private static func engine(_ ledger: MemoryLedger) throws -> ExpertExecution {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Device MoE fixture has no Metal") }
        let config = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 4, slotsPerSlab: 2,
            demandWindow: 4, readConcurrency: 2)
        return try ExpertExecution(device: device, ledger: ledger, configuration: config, packedAvailable: { false }) {
            key, _, destination in
            for kind in ExpertProjection.allCases {
                let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
                let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
                let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
                for i in 0..<(819200 / 4) { words[i] = UInt32(truncatingIfNeeded: UInt64(i + key.expert + 1) &* 0x9e37_79b9_7f4a_7c15) }
                for i in 0..<(51200 / 2) {
                    scales[i] = BF16.encode(Float((i + key.expert) % 7 + 1) / 1024)
                    biases[i] = BF16.encode(Float((i + key.expert) % 9 - 4) / 256)
                }
            }
        }
    }
    private static func retired(_ engine: ExpertExecution) throws {
        let snapshot = engine.snapshot()
        try demand(
            snapshot.cache.pinnedSlots == 0 && snapshot.cache.inFlightReads == 0 && snapshot.cache.reservedSlots == 0
                && snapshot.io.drives.allSatisfy({ $0.outstanding == 0 }),
            "Device MoE retained source reads or recyclable record pins")
    }

    package static func transferBitsAndCanonicalReduction() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let transfer = try PrefillMoETransfer()
            let tokens = 32
            guard let queue = transfer.device.makeCommandQueue() else { throw EngineError("No fixture command queue") }
            // Cover every BF16 bit pattern, then FP32 ties, subnormals and NaN
            // payloads that are not already representable at the BF16 boundary.
            var input = (0..<(tokens * 2560)).map { Float(bitPattern: UInt32($0 % 65536) << 16) }
            for (i, bits) in [UInt32]([
                1, 0x8000_0001, 0x3f80_8000, 0x3f81_8000, 0x007f_ffff,
                0x7f80_0001, 0xff81_2345, 0x7fc0_0001,
            ]).enumerated() { input[65536 + i] = Float(bitPattern: bits) }
            let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
            let workspace = try PrefillMoEWorkspace(input: input, selectedRows: rows(tokens), budget: budget, transfer: transfer)
            let scratchCharge = try ledger.reserve(bytes: UInt64(tokens * 2560 * 8), category: "device_moe_fixture_scratch")
            defer { try? ledger.release(scratchCharge) }
            guard let gathered = transfer.device.makeBuffer(length: input.count * 2, options: .storageModeShared),
                let scattered = transfer.device.makeBuffer(length: input.count * 4, options: .storageModeShared)
            else {
                throw EngineError("Fixture allocation failed")
            }
            // Finite arbitrary output bit patterns keep the canonical reducer
            // well-defined while input gather covers the complete BF16 domain.
            let outputs = (0..<input.count).map { (i: Int) in Float((i * 103 % 991) - 495) / 193 }
            outputs.withUnsafeBytes { scattered.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            var expected = [Float](repeating: 0, count: tokens * 10 * 2560)
            for expert in 0..<10 {
                try autoreleasepool {
                    let slots = (0..<tokens).map { UInt32($0 * 10 + (expert + $0) % 10) }
                    try workspace.beginBatch(experts: [expert], slots: [slots])
                    var success = false
                    defer { workspace.completeBatch(success: success) }
                    let map = try workspace.mapBuffer(slots)
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("Fixture command failed") }
                    try workspace.encodeGather(command, map: map, output: gathered, rows: tokens)
                    try workspace.encodeScatter(command, map: map, input: scattered, rows: tokens)
                    command.commit()
                    command.waitUntilCompleted()
                    try demand(command.status == .completed, "Fixture transfer command failed")
                    let bits = gathered.contents().assumingMemoryBound(to: UInt16.self)
                    for i in input.indices { try demand(bits[i] == BF16.encode(input[i]), "Device gather changed a BF16 handoff bit") }
                    for token in 0..<tokens {
                        let target = Int(slots[token]) * 2560
                        expected.replaceSubrange(target..<(target + 2560), with: outputs[(token * 2560)..<((token + 1) * 2560)])
                    }
                    success = true
                }
            }
            let raw = workspace.contributions.contents().assumingMemoryBound(to: Float.self)
            for i in expected.indices { try demand(raw[i].bitPattern == expected[i].bitPattern, "Canonical scatter changed a Float32 bit") }
            let reductions = try PinnedReductions(ledger: ledger)
            let weights = (0..<(tokens * 10)).map { Float($0 % 11 - 5) / 16 }
            let reference = try reductions.weightedExpertReduce10(experts: expected, weights: weights)
            let actual = try workspace.reduce(weights: weights, using: reductions)
            try demand(actual.map(\.bitPattern) == reference.map(\.bitPattern), "Device canonical reduction changed its tree/boundary")
            // Reuse this populated workspace and the already charged output
            // scratch to check the new enclosing-command completion ticket.
            guard let command = queue.makeCommandBuffer(), let wrong = queue.makeCommandBuffer(),
                let weightBuffer = transfer.device.makeBuffer(length: weights.count * 4, options: .storageModeShared)
            else {
                throw EngineError("Encoded reduction fixture unavailable")
            }
            weights.withUnsafeBytes { weightBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            try workspace.encodeReduction(command, weights: weightBuffer, output: scattered, using: reductions)
            try rejects { try workspace.completeEncodedReduction(command) }
            try rejects { try workspace.completeEncodedReduction(wrong) }
            try rejects { try workspace.encodeReduction(wrong, weights: weightBuffer, output: scattered, using: reductions) }
            command.commit()
            command.waitUntilCompleted()
            try workspace.completeEncodedReduction(command)
            try rejects { try workspace.completeEncodedReduction(command) }
            let encoded = scattered.contents().assumingMemoryBound(to: Float.self)
            for i in reference.indices {
                try demand(encoded[i].bitPattern == reference[i].bitPattern, "Encoded canonical reduction changed a bit")
            }
            try demand(transfer.snapshot().completedReductions == 2, "Encoded reduction completion was not counted exactly once")
        }
        try demand(ledger.snapshot().reserved == 0, "Transfer fixture retained ledger owners")
    }

    /// The layers of one whole-prefill scope pass one contribution buffer on: a layer whose reduction finished hands it
    /// to the next (the same buffer, nothing new taken from the escrow), whose reduction reads only its own writes and is
    /// bit-identical to the reduction of exactly those values; a layer that never reduced passes nothing on (the next
    /// allocates), and dropping the scope's spare returns the whole escrow.
    package static func contributionsPassToNextLayer() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let tokens = 32
            guard let queue = transfer.device.makeCommandQueue() else { throw EngineError("No fixture command queue") }
            let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
            let input = try budget.buffer(bytes: tokens * 2560 * 4, device: transfer.device)
            let scratchCharge = try ledger.reserve(bytes: UInt64(tokens * 2560 * 4), category: "device_moe_fixture_scratch")
            defer { try? ledger.release(scratchCharge) }
            guard let scattered = transfer.device.makeBuffer(length: tokens * 2560 * 4, options: .storageModeShared) else {
                throw EngineError("Fixture allocation failed")
            }
            let weights = (0..<(tokens * 10)).map { Float($0 % 11 - 5) / 16 }
            // One layer: every slot written from `seed`'s values, then reduced (or not); its buffer and reduction.
            // (Command buffers are autoreleased with what they hold: each layer and batch drains its own pool.)
            func layer(_ seed: Int, reduce: Bool) throws -> (buffer: ObjectIdentifier, matches: Bool) {
                try autoreleasepool {
                    let workspace = try PrefillMoEWorkspace(
                        tokens: tokens, input: input, selectedRows: rows(tokens), budget: budget, transfer: transfer)
                    var expected = [Float](repeating: 0, count: tokens * 10 * 2560)
                    for expert in 0..<10 {
                        try autoreleasepool {
                            let outputs = (0..<(tokens * 2560)).map { Float(($0 * 103 + seed * 37 + expert * 11) % 991 - 495) / 193 }
                            outputs.withUnsafeBytes { scattered.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                            let slots = (0..<tokens).map { UInt32($0 * 10 + (expert + $0) % 10) }
                            try workspace.beginBatch(experts: [expert], slots: [slots])
                            var success = false
                            defer { workspace.completeBatch(success: success) }
                            let map = try workspace.mapBuffer(slots)
                            guard let command = queue.makeCommandBuffer() else { throw EngineError("Fixture command failed") }
                            try workspace.encodeScatter(command, map: map, input: scattered, rows: tokens)
                            command.commit()
                            command.waitUntilCompleted()
                            try demand(command.status == .completed, "Fixture transfer command failed")
                            for token in 0..<tokens {
                                let target = Int(slots[token]) * 2560
                                expected.replaceSubrange(target..<(target + 2560), with: outputs[(token * 2560)..<((token + 1) * 2560)])
                            }
                            success = true
                        }
                    }
                    let identity = ObjectIdentifier(workspace.contributions)
                    guard reduce else { return (identity, true) }
                    let reference = try reductions.weightedExpertReduce10(experts: expected, weights: weights)
                    return (
                        identity, try workspace.reduce(weights: weights, using: reductions).map(\.bitPattern) == reference.map(\.bitPattern)
                    )
                }
            }
            let contributionBytes = PrefillMoEBudget.pageRounded(tokens * 10 * 2560 * 4)
            let first = try layer(1, reduce: true)
            let kept = budget.liveBytes
            let second = try layer(2, reduce: true)
            try demand(
                first.matches && second.matches && second.buffer == first.buffer && budget.liveBytes == kept,
                "A finished layer's contributions were not passed on, or the next layer's reduction read old values")
            _ = try layer(3, reduce: false)
            try demand(budget.liveBytes == kept - contributionBytes, "An unreduced layer passed its contributions on")
            let fourth = try layer(4, reduce: true)
            try demand(fourth.matches && budget.liveBytes == kept, "A fresh contribution buffer after an unreduced layer differs")
            budget.dropSpare()
            try demand(budget.liveBytes == kept - contributionBytes, "Dropping the scope's spare kept its escrow")
            withExtendedLifetime(input) {}
        }
        try demand(ledger.snapshot().reserved == 0, "Contribution reuse fixture retained ledger owners")
    }

    package static func expertParityAcrossTiles() throws {
        let ledger = try MemoryLedger(ceiling: 700_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let host = try ledger.reserve(bytes: 220_000_000, category: "device_moe_fixture_host_bound")
            defer { try? ledger.release(host) }
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 128)
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let executor = try engine(ledger)
            for tokens in [17, 127, 128, 255, 256, 257, 512] {
                try autoreleasepool {
                    let input = (0..<(tokens * 2560)).map { Float(($0 * 103 % 991) - 495) / 193 }
                    let selected = rows(tokens)
                    let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                    let workspace = try PrefillMoEWorkspace(input: input, selectedRows: selected, budget: budget, transfer: transfer)
                    var reference = [Float](repeating: 0, count: tokens * 10 * 2560)
                    try executor.beginSequence()
                    defer { try? executor.finishSequence() }
                    let window = try executor.makeWindow(layer: 0, experts: Array(0..<10))
                    defer { window.close() }
                    for first in stride(from: 0, to: 10, by: 2) {
                        let experts = Array(first..<min(first + 2, 10))
                        try window.consumeBatch(experts: experts) { buffers in
                            for lower in stride(from: 0, to: tokens, by: 256) {
                                let upper = min(tokens, lower + 256)
                                let jobs = experts.enumerated().map { index, expert in
                                    FusedExpertExecution.DeviceJob(
                                        expert: expert,
                                        slots: selected[expert]![lower..<upper].map { UInt32($0.0 * 10 + $0.1) },
                                        buffer: buffers[index], groupedExpert: true, simdMatrix: true)
                                }
                                for (index, expert) in experts.enumerated() {
                                    let tile = Array(input[(lower * 2560)..<(upper * 2560)])
                                    let result = try fused.evaluate(tile, buffer: buffers[index], groupedExpert: true, simdMatrix: true)
                                    for token in lower..<upper {
                                        let target = (token * 10 + (expert + token) % 10) * 2560
                                        reference.replaceSubrange(
                                            target..<(target + 2560),
                                            with:
                                                result[((token - lower) * 2560)..<((token - lower + 1) * 2560)])
                                    }
                                }
                                let before = fused.snapshot()
                                try fused.evaluateDeviceBatch(jobs, workspace: workspace)
                                let after = fused.snapshot()
                                try demand(
                                    after.submittedCommands == before.submittedCommands + 1
                                        && after.completedCommands == after.submittedCommands, "Device jobs changed joined batch boundaries"
                                )
                            }
                        }
                    }
                    let actualBits = workspace.contributions.contents().assumingMemoryBound(to: Float.self)
                    for i in reference.indices {
                        try demand(
                            actualBits[i].bitPattern == reference[i].bitPattern, "Device expert output changed at tile/rank boundary")
                    }
                    let weights = (0..<(tokens * 10)).map { Float($0 % 17 + 1) / 128 }
                    let a = try workspace.reduce(weights: weights, using: reductions)
                    let b = try reductions.weightedExpertReduce10(experts: reference, weights: weights)
                    try demand(a.map(\.bitPattern) == b.map(\.bitPattern), "Final routed output changed")
                    try retired(executor)
                }
            }
        }
        try demand(ledger.snapshot().reserved == 0, "Expert parity fixture retained resources")
    }

    package static func admissionCancellationAndEscape() throws {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let transfer = try PrefillMoETransfer()
            for tokens in stride(from: 512, through: 8192, by: 256) {
                try demand(
                    try PrefillMoEBudget.bytesForScope(tokens: tokens) <= UInt64(tokens - 256) * 240000,
                    "Owned GPU escrow exceeds the original conservative scope frontier")
            }
            let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: 17))
            var escaped: MTLBuffer?
            try autoreleasepool {
                let workspace = try PrefillMoEWorkspace(
                    input: .init(repeating: 1, count: 17 * 2560),
                    selectedRows: rows(17), budget: budget, transfer: transfer)
                escaped = workspace.contributions
            }
            try demand(
                escaped != nil && budget.liveBytes == PrefillMoEBudget.pageRounded(17 * 10 * 2560 * 4),
                "Escaped contribution lost its escrow owner")
            try rejects {
                _ = try PrefillMoEWorkspace(
                    input: .init(repeating: 1, count: 17 * 2560), selectedRows: rows(17), budget: budget, transfer: transfer)
            }
            escaped = nil
            try demand(budget.liveBytes == 0, "Escaped contribution never retired its permit")
            var incomplete = rows(17)
            incomplete[0]!.removeLast()
            try rejects {
                _ = try PrefillMoEWorkspace(
                    input: .init(repeating: 1, count: 17 * 2560), selectedRows: incomplete, budget: budget, transfer: transfer)
            }
            try demand(budget.liveBytes == 0, "Refused router map leaked its host permit")
            try autoreleasepool {
                let smallBytes =
                    try PrefillMoEBudget.hostBytesForScope(tokens: 17)
                    + PrefillMoEBudget.pageRounded(17 * 2560 * 4) + PrefillMoEBudget.batchMetadataBytes
                let smallBudget = try PrefillMoEBudget(ledger: ledger, bytes: smallBytes)
                try rejects {
                    _ = try PrefillMoEWorkspace(
                        input: .init(repeating: 1, count: 17 * 2560),
                        selectedRows: rows(17), budget: smallBudget, transfer: transfer)
                }
                try demand(smallBudget.liveBytes == 0, "Second device allocation denial leaked first allocation/host permit")
            }
            try autoreleasepool {
                let workspace = try PrefillMoEWorkspace(
                    input: .init(repeating: 1, count: 17 * 2560),
                    selectedRows: rows(17), budget: budget, transfer: transfer)
                // The first slot is provisionally staged; the second belongs
                // to another expert. Refusal must roll back the first slot.
                try rejects { try workspace.beginBatch(experts: [0], slots: [[0, 1]]) }
                try workspace.beginBatch(experts: [0], slots: [[0, 11]])
                workspace.completeBatch(success: false)
            }
            try demand(budget.liveBytes == 0, "Refused batch metadata retained its mapped pages")
            let executor = try engine(ledger)
            let fused = try FusedExpertExecution(arithmetic: .hostReference, ledger: ledger, groupedTiled: true)
            for allowed in [0, 1, 2] {
                try autoreleasepool {
                    let workspace = try PrefillMoEWorkspace(
                        input: .init(repeating: 1, count: 17 * 2560), selectedRows: rows(17), budget: budget, transfer: transfer)
                    try executor.beginSequence()
                    defer { try? executor.finishSequence() }
                    let window = try executor.makeWindow(layer: 0, experts: [0, 1])
                    defer { window.close() }
                    var checks = 0
                    do {
                        try window.consumeBatch(experts: [0, 1]) { buffers in
                            let jobs = (0..<2).map { expert in
                                FusedExpertExecution.DeviceJob(
                                    expert: expert,
                                    slots: (0..<17).map { UInt32($0 * 10 + (expert + $0) % 10) },
                                    buffer: buffers[expert], groupedExpert: true, simdMatrix: true)
                            }
                            try fused.evaluateDeviceBatch(
                                jobs, workspace: workspace,
                                shouldContinue: {
                                    checks += 1
                                    return checks <= allowed
                                })
                        }
                        throw EngineError("Cancellation unexpectedly completed")
                    } catch is IOOperationCancelled {}
                    try retired(executor)
                    try demand(
                        fused.snapshot().submittedCommands == fused.snapshot().completedCommands,
                        "Cancellation returned before GPU completion")
                    try rejects { try workspace.beginBatch(experts: [0], slots: [[0]]) }
                }
                // A command's buffers (and their permits) go when Metal drops the command, which can trail the join
                // by a moment (`PrefillMoEBudget.retireWait`, which a next charge waits for too).
                let until = Date().addingTimeInterval(PrefillMoEBudget.retireWait)
                while budget.liveBytes > 0 && Date() < until { usleep(1_000) }
                try demand(budget.liveBytes == 0, "Failed device batch leaked workspace permits")
            }
        }
        try demand(ledger.snapshot().reserved == 0, "Admission/cancellation fixture retained resources")
        var escaped: MTLBuffer?
        let capacity = try PrefillMoEBudget.bytesForScope(tokens: 1)
        try autoreleasepool {
            let transfer = try PrefillMoETransfer()
            let budget = try PrefillMoEBudget(ledger: ledger, bytes: capacity)
            let workspace = try PrefillMoEWorkspace(
                input: .init(repeating: 1, count: 2560),
                selectedRows: rows(1), budget: budget, transfer: transfer)
            escaped = workspace.contributions
        }
        try demand(
            escaped != nil && ledger.snapshot().reserved == capacity,
            "Raw escaped buffer lost its parent scope reservation")
        escaped = nil
        try demand(ledger.snapshot().reserved == 0, "Raw buffer retirement did not retire its last escrow owner")
        let small = try MemoryLedger(ceiling: 20000, transientMargin: 1)
        try rejects { _ = try PrefillMoEBudget(ledger: small, bytes: PrefillMoEBudget.bytesForScope(tokens: 1)) }
        try demand(small.snapshot().reserved == 0, "Escrow denial leaked a ledger reservation")
    }
}

extension PrefillMoESelfChecks {
    /// A charge waits for a finished command's buffers to retire (Metal may let go of a joined command a moment after
    /// `waitUntilCompleted`, 2.3.4's "Insufficient memory" on a ~1,460-token Qwen prompt), and refuses at once what can
    /// never fit. Returns how many of 500 joined commands still held their buffer right after the caller let go.
    @discardableResult
    package static func chargeWaitsForRetirement() throws -> Int {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device")
        }
        let page = PrefillMoEBudget.pageRounded(1)
        // Room for exactly one page: each buffer must retire before the next is charged, as a batch's maps must.
        let budget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.batchMetadataBytes + page)
        var late = 0
        func joined(_ count: Int) throws {
            for _ in 0..<count {
                try autoreleasepool {
                    let buffer = try budget.buffer(bytes: 16, device: device)
                    guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
                        throw EngineError("No Metal command")
                    }
                    blit.fill(buffer: buffer, range: 0..<16, value: 1)
                    blit.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                }
                if budget.liveBytes > 0 { late += 1 }
                _ = try budget.buffer(bytes: 16, device: device)  // the next batch's charge
            }
        }
        try joined(250)
        // The other half with the GPU keep-alive spinning beside them, as a Qwen reply runs by default.
        let keepAlive = try RequestGPUKeepAlive(ledger: ledger)  // must outlive the session: its deinit closes it
        let session = try keepAlive.begin()
        defer { withExtendedLifetime(keepAlive) { _ = session.close() } }
        try joined(250)
        // A buffer another thread lets go of 0.1 s later: the next charge waits for it.
        try autoreleasepool {
            final class Held: @unchecked Sendable {
                let buffer: MTLBuffer
                init(_ buffer: MTLBuffer) { self.buffer = buffer }
            }
            let held = Held(try budget.buffer(bytes: 16, device: device))
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { withExtendedLifetime(held) {} }
        }
        var start = Date()
        _ = try budget.buffer(bytes: 16, device: device)
        let waited = Date().timeIntervalSince(start)
        try demand(waited >= 0.05 && waited < PrefillMoEBudget.retireWait, "A charge did not wait for a buffer being released")
        // More than the escrow holds: refused without waiting.
        start = Date()
        try rejects { _ = try budget.buffer(bytes: Int(page) + 1, device: device) }
        try demand(Date().timeIntervalSince(start) < 0.1 && budget.liveBytes == 0, "An impossible charge waited or leaked")
        return late
    }
}
