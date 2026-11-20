import Foundation
import OrigamiEngine

/// Model-free checks of GDN state ownership: GDNBufferRegion against the host recurrence (bits of output and FP32
/// state), immutability of published state, transactional publication in LayerMath, admission, cancellation, and
/// checkpoint sharing of resident state. Primitive/storage checks only. These do not claim model correctness or speed.
package enum GDNBufferRegionSelfChecks {
    private static func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try condition() else { throw EngineError(message) }
    }
    private static func rejects(_ body: () throws -> Void, code: String? = nil) throws {
        do { try body() } catch let error as EngineError {
            if let code { try require(error.code == code, "GDN region rejection used the wrong error") }
            return
        }
        throw EngineError("GDN region accepted an invalid operation")
    }
    private struct Inputs {
        let q, k, v, decay, beta: [Float]
        init(rows: Int, seed: Int = 0) {
            q = (0..<(rows * 2048)).map { Float(($0 * 7 + seed) % 17 - 8) / 128 }
            k = (0..<(rows * 2048)).map { Float(($0 * 11 + seed) % 19 - 9) / 256 }
            v = (0..<(rows * 6144)).map { Float(($0 * 5 + seed) % 31 - 15) / 64 }
            decay = (0..<(rows * 48)).map { Float(($0 + seed) % 3 + 1) / 4 }
            beta = (0..<(rows * 48)).map { Float(($0 + seed) % 5 + 1) / 8 }
        }
    }

    package static func bufferRegionNumericsAdmissionAndCancellation() throws {
        let ledger = try MemoryLedger(ceiling: 320_000_000, transientMargin: 1)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedRMS: true)
            let reductions = try PinnedReductions(ledger: ledger)
            let gate = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let fixture = try ledger.reserve(bytes: 128_000_000, category: "gdn_region_test_fixture")
            defer { try? ledger.release(fixture) }
            let baseline = ledger.snapshot().reserved
            for contract in [true, false] {
                try autoreleasepool {
                    let region = try GDNBufferRegion(kernels: kernels, ledger: ledger, gate: gate)
                    var state = GDNState()
                    let normWeights = (0..<128).map { Float($0 % 7 + 1) / 8 }
                    for (step, rows) in [1, 1, 7, 8, 116, 256, 1].enumerated() {
                        try autoreleasepool {
                            let input = Inputs(rows: rows, seed: step)
                            let mask = (0..<rows).map { UInt32(($0 + step) % 5 == 0 ? 0 : 1) }
                            var z = (0..<(rows * 6144)).map { LayerArithmetic.round(Float($0 % 29 - 14) / 8) }
                            // Exercise unchanged host repair after the embedded
                            // wide gate, including nonfinite/subnormal behavior.
                            if rows == 116 {
                                z[0] = .nan
                                z[1] = .infinity
                                z[2] = Float(bitPattern: 1)
                            }
                            let previous = state
                            let beforeBits = state.recurrent.map(\.bitPattern)
                            let expected = try kernels.gdn(
                                q: input.q, k: input.k, v: input.v, decay: input.decay,
                                beta: input.beta, state: state.recurrent, mask: mask, contractUpdates: contract)
                            let norm = try kernels.rmsnorm(x: expected.output, weights: normWeights)
                            let expectedGate =
                                gate.usesGPU(rows: rows)
                                ? try gate.gate(normed: norm, z: z)
                                : zip(norm, z).map { LayerArithmetic.round($0 * kernels.arithmetic.sigmoidFP32($1)) }
                            let fallbackBefore = gate.gatingFallbacks
                            let actual = try region.evaluate(
                                q: input.q, k: input.k, v: input.v, decay: input.decay,
                                beta: input.beta, state: state, normWeights: normWeights, z: z, mask: mask,
                                contractUpdates: contract, capture: true)
                            try require(
                                actual.recurrence!.map(\.bitPattern) == expected.output.map(\.bitPattern)
                                    && actual.normed!.map(\.bitPattern) == norm.map(\.bitPattern)
                                    && actual.gated.map(\.bitPattern) == expectedGate.map(\.bitPattern)
                                    && actual.state.recurrent.map(\.bitPattern) == expected.state.map(\.bitPattern),
                                "Composed GDN changed recurrence/state/RMS/gate bytes at rows\(rows), contract\(contract)")
                            try require(
                                previous.recurrent.map(\.bitPattern) == beforeBits,
                                "Composed GDN mutated its old state")
                            try require(
                                (actual.state.recurrentStorage != nil) == (rows == 1),
                                "Composed prefill retained its large output in persistent state")
                            if rows == 116 {
                                try require(gate.gatingFallbacks == fallbackBefore + 1, "Embedded gate failed to repair exceptions")
                            }
                            if rows == 1, let old = state.recurrentStorage {
                                try require(actual.state.recurrentStorage !== old, "Composed GDN reused a published owner")
                            }
                            state = actual.state
                        }
                        try require(
                            ledger.snapshot().categories["gdn_buffer_region", default: 0] == 0,
                            "Composed GDN retained transient scratch after returning")
                    }
                    let stats = region.snapshot()
                    let resident = region.decodeSnapshot()
                    try require(
                        stats.submittedCommands == 7 && stats.completedCommands == 7 && stats.wideGateCommands == 3
                            && resident.submittedCommands == 3 && resident.inputStateUploads == 2,
                        "Composed GDN row floor or warm-state upload count changed")
                    let beforeBits = state.recurrent.map(\.bitPattern)
                    let retained = ledger.snapshot().reserved
                    let input = Inputs(rows: 1)
                    for stopAt in [1, 2, 3, 4] {
                        let prior = region.snapshot()
                        try autoreleasepool {
                            var calls = 0
                            do {
                                _ = try region.evaluate(
                                    q: input.q, k: input.k, v: input.v, decay: input.decay,
                                    beta: input.beta, state: state, normWeights: normWeights, z: .init(repeating: 0, count: 6144),
                                    shouldContinue: {
                                        calls += 1
                                        return calls < stopAt
                                    })
                                throw EngineError("GDN region ignored cancellation")
                            } catch is IOOperationCancelled {}
                        }
                        let after = region.snapshot()
                        let joined: UInt64 = stopAt >= 3 ? 1 : 0
                        try require(
                            after.submittedCommands == prior.submittedCommands + joined
                                && after.completedCommands == prior.completedCommands + joined && ledger.snapshot().reserved == retained
                                && state.recurrent.map(\.bitPattern) == beforeBits,
                            "GDN region cancellation failed to join/retire or changed state")
                    }
                    try rejects {
                        _ = try region.evaluate(
                            q: [1], k: [1], v: [1], decay: [1], beta: [1],
                            state: state, normWeights: normWeights, z: [1])
                    }
                    let tiny = try MemoryLedger(ceiling: 2_000_000, transientMargin: 1)
                    let denied = try GDNBufferRegion(kernels: kernels, ledger: tiny)
                    try rejects(
                        {
                            _ = try denied.evaluate(
                                q: input.q, k: input.k, v: input.v, decay: input.decay,
                                beta: input.beta, state: state, normWeights: normWeights, z: .init(repeating: 0, count: 6144))
                        }, code: "insufficient_memory")
                    try require(
                        tiny.snapshot().reserved == 0 && denied.snapshot().submittedCommands == 0,
                        "Denied GDN region left partial allocations or commands")
                }
                try require(ledger.snapshot().reserved == baseline, "GDN region owners did not retire")
            }
        }
        try require(ledger.snapshot().reserved == 0, "GDN region fixture did not retire")
    }

    package static func bufferRegionLayerParityAndPublication() throws {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedRMS: true)
            let reductions = try PinnedReductions(ledger: ledger)
            let gate = try FusedGDNPreparation(arithmetic: kernels.arithmetic, reductions: reductions, ledger: ledger)
            let region = try GDNBufferRegion(kernels: kernels, ledger: ledger, gate: gate)
            var rejectOutput = false
            let linear: LayerMath.Linear = { input, name, _ in
                if name.hasSuffix("out_proj") {
                    if rejectOutput { throw EngineError("Injected output projection failure") }
                    return (0..<(input.count / 6144 * 2560)).map { input[$0 / 2560 * 6144 + $0 % 2560] }
                }
                let rows = input.count / 2560
                let width = name.hasSuffix("in_proj_qkv") ? 10240 : (name.hasSuffix("in_proj_z") ? 6144 : 48)
                return (0..<(rows * width)).map { LayerArithmetic.round(Float(($0 * 7 + 3) % 31 - 15) / 16) }
            }
            let tensor: LayerMath.Tensor = { name in
                if name.hasSuffix("conv1d.weight") { return (0..<40960).map { Float($0 % 4 + 1) / 16 } }
                if name.hasSuffix("norm.weight") { return .init(repeating: 1, count: 128) }
                return .init(repeating: 0, count: 48)
            }
            let array = LayerMath(kernels: kernels, linear: linear, tensor: tensor, reductions: reductions, gdnPreparation: gate)
            let resident = LayerMath(
                kernels: kernels, linear: linear, tensor: tensor, reductions: reductions, gdnPreparation: gate, gdnBufferRegion: region)
            var expected = GDNState()
            var actual = GDNState()
            var arrayCaptures: [String: [UInt32]] = [:]
            var residentCaptures: [String: [UInt32]] = [:]
            array.capture = { arrayCaptures[$0] = $1.map(\.bitPattern) }
            resident.capture = { residentCaptures[$0] = $1.map(\.bitPattern) }
            for rows in [3, 1, 1, 8, 116, 1] {
                arrayCaptures.removeAll()
                residentCaptures.removeAll()
                let input = [Float](repeating: 1, count: rows * 2560)
                let a = try array.gdn(x: input, layer: 0, state: &expected)
                let b = try resident.gdn(x: input, layer: 0, state: &actual)
                try require(
                    a.map(\.bitPattern) == b.map(\.bitPattern) && arrayCaptures == residentCaptures
                        && actual.recurrent.map(\.bitPattern) == expected.recurrent.map(\.bitPattern)
                        && actual.convolutionTail == expected.convolutionTail && actual.offset == expected.offset,
                    "The GDN region changed full layer values, captures or state")
                try require((actual.recurrentStorage != nil) == (rows == 1), "Prefill did not retain the array control")
            }
            let recurrenceStats = region.decodeSnapshot()
            try require(
                recurrenceStats.submittedCommands == 3 && recurrenceStats.inputStateUploads == 2,
                "GDN region decode-only boundary or restored-array upload differs")
            let before = actual
            let reserved = ledger.snapshot().reserved
            rejectOutput = true
            try rejects { _ = try resident.gdn(x: .init(repeating: 1, count: 2560), layer: 0, state: &actual) }
            try require(
                before.offset == actual.offset && before.convolutionTail == actual.convolutionTail
                    && before.recurrentStorage === actual.recurrentStorage && ledger.snapshot().reserved == reserved,
                "Failed output projection published or leaked a partially updated GDN state")
        }
        try require(ledger.snapshot().reserved == 0, "GDN region layer retirement leaked allocations")
    }

    package static func bufferRegionCheckpointOwnership() throws {
        let ledger = try MemoryLedger(ceiling: 900_000_000, transientMargin: 1)
        try autoreleasepool { try checkpointExercise(ledger) }
        try require(ledger.snapshot().reserved == 0, "Resident checkpoint retirement leaked a reservation")
    }
    private static func checkpointExercise(_ ledger: MemoryLedger) throws {
        let kernels = try ReferenceKernels(pinnedRMS: true)
        let region = try GDNBufferRegion(kernels: kernels, ledger: ledger)
        let input = Inputs(rows: 1)
        let hash = String(repeating: "a", count: 64)
        func transition(_ state: GDNState) throws -> GDNState {
            try region.evaluate(
                q: input.q, k: input.k, v: input.v, decay: input.decay, beta: input.beta,
                state: state, normWeights: .init(repeating: 1, count: 128), z: .init(repeating: 0, count: 6144)
            ).state
        }
        func uploads() -> UInt64 { region.decodeSnapshot().inputStateUploads }
        let identity = try CheckpointIdentity(
            modelSHA256: hash, tokenizerSHA256: hash, templateSHA256: hash,
            quantizationSHA256: hash, arithmeticEpoch: "resident-gdn-primitive-fixture")
        var source: SequenceState? = try SequenceState(inputTokens: 1, outputTokens: 1, ledger: ledger, contextLimit: 2)
        source!.gdn[0] = try transition(source!.gdn[0]!)
        let tensorBits = source!.gdn[0]!.recurrent.map(\.bitPattern)
        let residentBytes = ledger.snapshot().categories["gdn_resident_state", default: 0]
        var image: ModelCheckpoint? = try ModelCheckpoint.capture(
            state: source!, consumedIDs: [], identity: identity,
            segments: [], ledger: ledger)
        let before = (try image!.tensorSHA256())
        try require(
            residentBytes > 0 && ledger.snapshot().categories["gdn_resident_state"] == residentBytes,
            "Checkpoint charged a second lease for the same CPU/Metal allocation")
        let hostBytes = ledger.snapshot().categories["model_checkpoint", default: 0]
        try require(image!.reservedBytes == hostBytes + residentBytes, "Prefix quota omitted retained Metal bytes")
        var sawRawTensor = false
        try image!.withTensors { name, _, bytes in
            if name == "gdn.0.recurrent" {
                sawRawTensor = true
                try require(
                    Array(bytes.bindMemory(to: UInt32.self)) == tensorBits,
                    "Checkpoint codec changed raw FP32 recurrence bytes")
            }
        }
        try require(sawRawTensor, "Checkpoint codec omitted resident recurrence")
        source = nil
        try require(
            ledger.snapshot().categories["gdn_resident_state"] == residentBytes && (try image!.tensorSHA256()) == before,
            "Retiring the source invalidated the checkpoint's resident tensor")
        var target: SequenceState? = try SequenceState(inputTokens: 1, outputTokens: 1, ledger: ledger, contextLimit: 2)
        try image!.restore(into: target!, expectedIdentity: identity, expectedSegments: [], expectedConsumedIDs: [], intent: .appendInput)
        // A restored resident owner is fed directly, with no state upload.
        let beforeUploads = uploads()
        target!.gdn[0] = try transition(target!.gdn[0]!)
        try require(
            uploads() == beforeUploads && (try image!.tensorSHA256()) == before,
            "Restored continuation uploaded state again or mutated its checkpoint")
        try require(
            ledger.snapshot().categories["gdn_resident_state", default: 0] == 2 * residentBytes,
            "Diverged checkpoint and sequence did not hold distinct state leases")
        image = nil
        try require(
            ledger.snapshot().categories["gdn_resident_state"] == residentBytes,
            "Last checkpoint reader did not release its original state")
        target = nil
        try require(
            ledger.snapshot().categories["gdn_resident_state", default: 0] == 0,
            "Last continuation reader did not release its state")
    }
}
