import Dispatch
import Foundation
import Metal
import OrigamiEngine

/// Standalone checks for decode next-layer pre-routing (no XCTest).
package enum DecodePreroutingSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private final class Recorded: @unchecked Sendable {
        let lock = NSLock()
        var calls: [[Int]] = []
        func record(_ ids: [Int]) {
            lock.lock()
            calls.append(ids)
            lock.unlock()
        }
        var values: [[Int]] {
            lock.lock()
            defer { lock.unlock() }
            return calls
        }
    }
    private final class ReadLog: @unchecked Sendable {
        let lock = NSLock()
        var keys: [ExpertKey] = []
        func add(_ key: ExpertKey) {
            lock.lock()
            keys.append(key)
            lock.unlock()
        }
        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return keys.count
        }
    }

    /// Small synthetic fixed-shape MoE HC and router weights; no model is read.
    private struct Fixture {
        let device: MTLDevice
        let norm: [Float]
        let hc: DecodeFrontierWeights.HC
        let router: MTLBuffer
        let normBuffer: MTLBuffer
        init(device: MTLDevice, nonfiniteNorm: Bool = false) throws {
            self.device = device
            func buffer<T>(_ values: [T]) throws -> MTLBuffer {
                try values.withUnsafeBytes { bytes in
                    guard let result = device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared) else {
                        throw EngineError("Pre-routing fixture allocation failed")
                    }
                    return result
                }
            }
            func projection(_ width: Int, _ outputs: Int, _ seed: UInt32) throws -> FusedHCDecode.Projection {
                let words = (0..<(width * outputs / 8)).map { UInt32(truncatingIfNeeded: $0) &* 2_654_435_761 &+ seed }
                let scales = (0..<(width * outputs / 64)).map { BF16.encode(Float($0 % 3 + 1) / 512) }
                let biases = (0..<(width * outputs / 64)).map { BF16.encode(-Float($0 % 3 + 1) * 7.5 / 512) }
                return try .init(words: buffer(words), scales: buffer(scales), biases: buffer(biases))
            }
            var norm = (0..<10240).map { BF16.decode(BF16.encode(Float($0 % 17 + 1) / 16)) }
            if nonfiniteNorm { norm[5] = Float.nan }
            self.norm = norm
            let injection = (0..<(4 * 10240)).map { BF16.decode(BF16.encode(Float(($0 * 37) % 31 - 15) / 512)) }
            normBuffer = try buffer(norm.map(BF16.encode))
            hc = .init(
                norm: normBuffer, injection: try buffer(injection), down: try projection(10240, 320, 17), up: try projection(320, 10240, 71)
            )
            // Deterministic pseudo-random router rows; FP32 as the dense router helper reads them.
            var state: UInt32 = 0x9e37_79b9
            let weights: [Float] = (0..<(2560 * 512)).map { _ in
                state = state &* 1_664_525 &+ 1_013_904_223
                return Float(Int32(bitPattern: state) >> 8) / Float(1 << 23) / 32
            }
            router = try buffer(weights)
        }
    }

    private static func listener() -> MTLSharedEventListener {
        MTLSharedEventListener(dispatchQueue: DispatchQueue(label: "origami.decode.prerouting.check", qos: .userInitiated))
    }

    /// The prediction kernels are the layer's own MoE HC and router over the
    /// supplied row: ids and margins equal the engine's reference evaluation;
    /// gating passes exactly the entries at or above the margin; HC exceptions
    /// fail the batch (counted) without issuing anything.
    package static func predictorExactness() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        let fixtureBound = try ledger.reserve(bytes: 64 * 1024 * 1024, category: "prerouting_check_fixture")
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let hc = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger)
            let dense = try PinnedDenseLinear(ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let fixture = try Fixture(device: device)
            let residual = (0..<10240).map { (i: Int) in BF16.decode(BF16.encode(Float((i * 73 + 5) % 1009 - 504) / 29)) }
            let residualBF16 = residual.map(BF16.encode)
            guard
                let input = residualBF16.withUnsafeBytes({
                    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                })
            else {
                throw EngineError("Pre-routing fixture input allocation failed")
            }
            // Reference: the engine's one-row HC evaluation and dense router over the same weights.
            let mixed = try hc.evaluate(
                residual: residual, normWeights: fixture.normBuffer, downWeights: fixture.hc.down,
                upWeights: fixture.hc.up, injectionWeights: nil
            ).mixed
            let logits = try dense.linear(mixed, weights: fixture.router, width: 2560, outputs: 512, outputBF16: false)
            let order = (0..<512).sorted { (a: Int, b: Int) in logits[a] == logits[b] ? a < b : logits[a] > logits[b] }
            let expectedIDs = Array(order.prefix(10))
            let eleventh = logits[order[10]]
            let expectedMargins = expectedIDs.map { logits[$0] - eleventh }
            try Check.require(expectedMargins[9] > 1e-3, "Fixture router has no clear tenth/eleventh gap")
            let before = ledger.snapshot().reserved
            for (threshold, label) in [(0.0, "ungated"), (Double(expectedMargins[4]), "median gate"), (64.0, "all gated")] {
                let owner = try DecodePrerouting(
                    device: device, ledger: ledger, listener: listener(), hc: hc, dense: dense, selector: selector, margin: threshold)
                try Check.require(
                    (ledger.snapshot().categories["decode_prerouting_metadata"] ?? 0) >= DecodePrerouting.metadataBytes
                        && (ledger.snapshot().categories["decode_prerouting_workspace"] ?? 0) > 0,
                    "Pre-routing owners were not charged in their categories")
                let recorded = Recorded()
                let command = queue.makeCommandBuffer()!
                let batch = try owner.stage(
                    command: command, afterAttentionBF16: input, moeHC: fixture.hc, router: fixture.router, target: 7,
                    operation: { ids, layer, _ in
                        guard layer == 7 else { throw EngineError("Pre-routing target differs") }
                        recorded.record(ids)
                        return DecodePrerouting.Issue()
                    }, discard: { _ in })
                command.commit()
                command.waitUntilCompleted()
                let observation = try batch.adopt()
                let outcome = try observation.outcome.get()
                try Check.require(
                    Set(outcome.predicted) == Set(expectedIDs),
                    "\(label): predicted top ten differs from the reference router: \(outcome.predicted) != \(expectedIDs)")
                let referenceMargin = Dictionary(uniqueKeysWithValues: zip(expectedIDs, expectedMargins))
                try Check.require(
                    zip(outcome.predicted, outcome.margins).allSatisfy({
                        abs($0.1 - referenceMargin[$0.0]!) <= 1e-4 * max(1, abs(referenceMargin[$0.0]!))
                    }),
                    "\(label): predicted margins differ from the reference: \(outcome.margins) != \(expectedMargins)")
                let expectedGated = Set(zip(expectedIDs, expectedMargins).filter { $0.1 >= Float(threshold) }.map(\.0))
                try Check.require(
                    Set(outcome.gated) == expectedGated && outcome.gated.count == expectedGated.count,
                    "\(label): gate passed \(outcome.gated), expected \(expectedGated)")
                try Check.require(
                    recorded.values.map { Set($0) } == (expectedGated.isEmpty ? [] : [expectedGated]),
                    "\(label): factory received \(recorded.values)")
                // A second stage on the same parity must wait for retirement; the other parity is free.
                let other = queue.makeCommandBuffer()!
                let second = try owner.stage(
                    command: other, afterAttentionBF16: input, moeHC: fixture.hc, router: fixture.router, target: 8,
                    operation: { _, _, _ in DecodePrerouting.Issue() }, discard: { _ in })
                do {
                    _ = try owner.stage(
                        command: other, afterAttentionBF16: input, moeHC: fixture.hc, router: fixture.router, target: 10,
                        operation: { _, _, _ in DecodePrerouting.Issue() }, discard: { _ in })
                    throw EngineError("Pre-routing admitted a third live batch on a busy workspace")
                } catch let error as EngineError { guard String(describing: error).contains("two-batch") else { throw error } }
                other.commit()
                other.waitUntilCompleted()
                owner.adopt(second, actual: expectedIDs)
                try Check.require(
                    owner.counts.batchesStaged == 2 && owner.counts.batchesAdopted == 1
                        && owner.counts.predictedExperts == 10 && owner.counts.coveredDemand == UInt64(expectedGated.count)
                        && owner.counts.predictionsGated == UInt64(10 - expectedGated.count) && owner.counts.predictionsFailed == 0
                        && owner.counts.marginThreshold == threshold, "\(label): counters differ: \(owner.counts)")
            }
            // Non-finite HC weights raise exception words: the batch fails, nothing is issued, and it is counted.
            let faulty = try Fixture(device: device, nonfiniteNorm: true)
            let owner = try DecodePrerouting(
                device: device, ledger: ledger, listener: listener(), hc: hc, dense: dense, selector: selector, margin: 0)
            let recorded = Recorded()
            let command = queue.makeCommandBuffer()!
            let batch = try owner.stage(
                command: command, afterAttentionBF16: input, moeHC: faulty.hc, router: faulty.router, target: 3,
                operation: { ids, _, _ in
                    recorded.record(ids)
                    return DecodePrerouting.Issue()
                }, discard: { _ in })
            command.commit()
            command.waitUntilCompleted()
            owner.adopt(batch, actual: expectedIDs)
            try Check.require(
                recorded.values.isEmpty && owner.counts.predictionsFailed == 1 && owner.counts.predictionsIssued == 0
                    && owner.counts.batchesAdopted == 1, "Exceptional pre-routing issued reads or went uncounted: \(owner.counts)")
            // A cancelled (never submitted) batch is closed and counted without a factory call.
            let unsubmitted = queue.makeCommandBuffer()!
            let closed = try owner.stage(
                command: unsubmitted, afterAttentionBF16: input, moeHC: fixture.hc, router: fixture.router, target: 4,
                operation: { ids, _, _ in
                    recorded.record(ids)
                    return DecodePrerouting.Issue()
                }, discard: { _ in })
            owner.close(closed)
            owner.close(closed)
            try Check.require(
                recorded.values.isEmpty && owner.counts.batchesClosedUnadopted == 1, "Closed pre-routing batch ran or was miscounted")
            // Earlier owners retire with their Metal resources; the one live owner stays within its declared bound.
            for _ in 0..<2_000 where ledger.snapshot().reserved - before > DecodePrerouting.ownerBytes {
                Thread.sleep(forTimeInterval: 0.001)
            }
            try Check.require(
                ledger.snapshot().reserved - before <= DecodePrerouting.ownerBytes, "Pre-routing owners exceeded their declared bound")
        }
        try ledger.release(fixtureBound)
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Pre-routing fixture or owner retained reservations")
    }

    /// A predicted read in flight is adopted by the authoritative demand: one
    /// producer, one slot, the same bytes; resident records issue nothing;
    /// dropped prediction handles leave evictable verified records and no pins.
    package static func adoptionWithoutDuplication() throws {
        let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let log = ReadLog()
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                log.add(key)
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Pre-routing fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.layer * 16 + key.expert))
            }
            do {
                _ = try execution.preroutingFactory()
                throw EngineError("Pre-routing factory admitted outside decode")
            } catch let error as EngineError { guard String(describing: error).contains("decode generation") else { throw error } }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let factory = try execution.preroutingFactory()
            var issue = try factory([3, 4], 1, { true })
            try Check.require(
                issue.issued == 2 && issue.requests.count == 2 && issue.resident == 0 && issue.noSlot == 0 && issue.refused == 0,
                "Pre-routing did not issue both predicted reads: \(issue.issued)/\(issue.resident)/\(issue.noSlot)/\(issue.refused)")
            _ = started.wait(timeout: .now() + 10)
            _ = started.wait(timeout: .now() + 10)
            try Check.require(
                log.count == 2 && execution.snapshot().cache.inFlightReads == 2, "Predicted reads were not started on the prefetch lanes")
            let window = try execution.makeWindow(layer: 1, experts: [4, 3])
            try Check.require(
                log.count == 2 && execution.snapshot().cache.misses == 2 && execution.snapshot().cache.inFlightReads == 2,
                "Authoritative demand duplicated a predicted read in flight")
            issue.requests.removeAll()  // The window holds its own handles; dropping the prediction's changes nothing.
            gate.signal()
            gate.signal()
            for expert in [3, 4] {
                try window.consume(expert: expert) { buffer in
                    try Check.require(
                        buffer.bytes[0] == UInt8(16 + expert) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(16 + expert),
                        "Adopted predicted record carried wrong bytes")
                }
            }
            try Check.require(log.count == 2 && window.isClosed, "Consumption of adopted predictions re-read a record")
            // Resident records issue nothing; a fresh prediction reads and, with its handle dropped, leaves no pin.
            issue = try factory([3, 4, 5], 1, { true })
            try Check.require(
                issue.resident == 2 && issue.issued == 1 && issue.requests.count == 1, "Resident predicted records were re-read")
            _ = started.wait(timeout: .now() + 10)
            issue.requests.removeAll()
            gate.signal()
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            let snapshot = execution.snapshot()
            try Check.require(
                snapshot.cache.inFlightReads == 0 && snapshot.cache.pinnedSlots == 0 && snapshot.cache.misses == 3 && log.count == 3,
                "Dropped prediction left a pin or an unfinished producer: \(snapshot.cache)")
            // The dropped read is now resident: the authoritative demand finds it without a read.
            let later = try execution.makeWindow(layer: 1, experts: [5])
            try later.consume(expert: 5) { buffer in
                try Check.require(buffer.bytes[0] == UInt8(16 + 5), "Resident predicted record differs")
            }
            try Check.require(
                log.count == 3 && snapshot.cache.hits + 1 == execution.snapshot().cache.hits, "Completed prediction was not a demand hit")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Pre-routing adoption fixture leaked")
    }

    /// Miss counting for the mirror ranking: a prompt miss weighs 1, a decode miss 4, a prediction still reading that
    /// demand adopts counts once, a resident hit and a prediction demand never asks for count nothing.
    package static func missAccounting() throws {
        let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                started.signal()
                if key.layer == 1 && key.expert == 3 {
                    guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Fixture read timed out") }
                }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            let routing = RoutingProfile()
            execution.routing = routing
            func consumeAll(_ window: ExpertDemandWindow) throws {
                for key in window.keys { try window.consume(expert: key.expert) { _ in } }
            }
            try execution.beginSequence()
            try consumeAll(try execution.makeWindow(layer: 0, experts: [2]))
            try execution.beginDecodePhase()
            let prediction = try execution.preroutingFactory()([3], 1, { true })
            guard prediction.issued == 1, started.wait(timeout: .now() + 10) == .success, started.wait(timeout: .now() + 10) == .success
            else {
                throw EngineError("Fixture prediction did not start")
            }
            let window = try execution.makeWindow(layer: 1, experts: [3, 4])
            gate.signal()
            try consumeAll(window)
            try consumeAll(try execution.makeWindow(layer: 1, experts: [3, 4]))  // both resident now
            var unused = try execution.preroutingFactory()([6], 1, { true })
            try Check.require(unused.issued == 1, "Fixture's unused prediction was not issued")
            unused.requests.removeAll()
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            let missed = routing.missed
            try Check.require(
                missed[2] == 1 && missed[512 + 3] == 4 && missed[512 + 4] == 4 && missed[512 + 6] == 0
                    && missed.reduce(0, +) == 9,
                "Miss counts differ: prompt \(missed[2]), adopted \(missed[512 + 3]), read \(missed[512 + 4]), unused \(missed[512 + 6])")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Miss-accounting fixture leaked")
    }

    /// An adopted prediction whose read still runs when decode ends (it started before adoption, so read hygiene
    /// cannot cancel it) is kept and joined by the next phase change, which would otherwise refuse with
    /// `expert_phase_busy` and fail a finished reply.
    package static func adoptedReadJoinedAtPhaseChange() throws {
        let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Pre-routing fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            execution.retainPredictions(try execution.preroutingFactory()([3], 1, { true }).requests)
            guard started.wait(timeout: .now() + 10) == .success else { throw EngineError("Predicted read did not start") }
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { gate.signal() }
            try execution.beginDecodePhase()  // as the frontier-loan reclaim after the last token
            let cache = execution.snapshot().cache
            try Check.require(
                cache.inFlightReads == 0 && cache.pinnedSlots == 0, "Phase change left the prediction running or pinned: \(cache)")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Adopted-prediction fixture leaked")
    }

    /// Without a spare slot (every slot reading or pinned; or, under reuse
    /// admission, every cached record protected by the current row) the
    /// prediction is dropped and counted; the reservation state is untouched.
    package static func dropWithoutSpareSlot() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let log = ReadLog()
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 2, slotsPerSlab: 2, demandWindow: 2, readConcurrency: 1)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                log.add(key)
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Pre-routing fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let factory = try execution.preroutingFactory()
            let window = try execution.makeWindow(layer: 0, experts: [0, 1])
            // Both demand reads run (demand may use every lane); both slots are reading.
            _ = started.wait(timeout: .now() + 10)
            _ = started.wait(timeout: .now() + 10)
            let before = execution.snapshot().cache
            try Check.require(before.reservedSlots + before.inFlightReads == 2 && log.count == 2, "Fixture did not occupy both slots")
            let issue = try factory([7], 1, { true })
            let after = execution.snapshot().cache
            try Check.require(
                issue.noSlot == 1 && issue.issued == 0 && issue.requests.isEmpty && issue.resident == 0 && issue.refused == 0,
                "Pre-routing forced a slot: \(issue.issued)/\(issue.resident)/\(issue.noSlot)/\(issue.refused)")
            try Check.require(
                after.misses == before.misses && after.evictions == before.evictions && after.reservedSlots == before.reservedSlots
                    && after.activeSlots == before.activeSlots && after.inFlightReads == before.inFlightReads && log.count == 2,
                "Dropped prediction mutated the cache: \(after) vs \(before), reads \(log.count)")
            gate.signal()
            gate.signal()
            for expert in [0, 1] { try window.consume(expert: expert) { _ in } }
            // Evictable verified records are ordinary CLOCK victims for a prediction.
            let admitted = try factory([7], 1, { true })
            try Check.require(
                admitted.issued == 1 && execution.snapshot().cache.evictions == before.evictions + 1,
                "Prediction did not take an evictable slot")
            _ = started.wait(timeout: .now() + 10)
            gate.signal()
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            try execution.finishSequence()
        }
        try autoreleasepool {
            // Reuse admission: the current row's ten records are protected; a prediction never evicts them
            // and never borrows the transient bypass arena.
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 32, slotsPerSlab: 8, demandWindow: 10,
                readConcurrency: 4, reuseAdmission: true)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let factory = try execution.preroutingFactory()
            let window = try execution.makeWindow(layer: 0, experts: Array(0..<10))
            try window.consumeTen(experts: Array(0..<10)) { buffers in try Check.require(buffers.count == 10, "Row fixture differs") }
            let before = execution.snapshot().cache
            let issue = try factory([20, 21], 0, { true })
            let after = execution.snapshot().cache
            try Check.require(
                issue.noSlot == 2 && issue.issued == 0 && after.misses == before.misses && after.evictions == before.evictions
                    && after.reuseBypasses == before.reuseBypasses && after.reuseBorrowedBypasses == before.reuseBorrowedBypasses,
                "Prediction evicted a protected record or borrowed the bypass arena: \(issue.noSlot)/\(issue.issued) \(after)")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Pre-routing drop fixture leaked")
    }

    /// The flag defaults off, and the configuration refuses unsupported combinations.
    package static func countersAbsentWhenOff() throws {
        try Check.require(
            !RuntimeConfiguration().decodeNextLayerPrerouting && RuntimeConfiguration().decodePreroutingMargin == 0.25,
            "Pre-routing defaults differ")

        let base = CheckFixtures.file("config/trace-prerouting-8192.json")
        let enabled = try RuntimeConfiguration.load(base)
        try Check.require(
            enabled.decodeNextLayerPrerouting && enabled.decodePreroutingMargin == 0.25 && enabled.decodeTokenPlan,
            "The pre-routing trace fixture does not enable pre-routing")
        let previous = try RuntimeConfiguration.load(CheckFixtures.file("config/trace-before-prerouting-8192.json"))
        try Check.require(!previous.decodeNextLayerPrerouting, "An older configuration enabled pre-routing")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-prerouting-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        var object2 = try JSONSerialization.jsonObject(with: Data(contentsOf: base)) as! [String: Any]
        func refused(_ mutate: (inout [String: Any]) -> Void, _ label: String) throws {
            var execution = object2["execution"] as! [String: Any]
            mutate(&execution)
            object2["execution"] = execution
            try JSONSerialization.data(withJSONObject: object2).write(to: url)
            do { _ = try RuntimeConfiguration.load(url) } catch is EngineError { return }
            throw EngineError("Configuration accepted " + label)
        }
        let original = object2
        try refused({ $0["decode_token_plan"] = false }, "pre-routing without the token plan")
        object2 = original
        try refused({ $0["decode_next_layer_prerouting"] = 1 }, "a non-boolean flag")
        object2 = original
        try refused({ $0["decode_prerouting_margin"] = "0.25" }, "a string margin")
        object2 = original
        try refused({ $0["decode_prerouting_margin"] = true }, "a boolean margin")
        object2 = original
        try refused({ $0["decode_prerouting_margin"] = -0.5 }, "a negative margin")
        object2 = original
        try refused({ $0["decode_prerouting_margin"] = 65 }, "an out-of-range margin")
        object2 = original
        var execution = object2["execution"] as! [String: Any]
        execution["decode_prerouting_margin"] = 0
        object2["execution"] = execution
        try JSONSerialization.data(withJSONObject: object2).write(to: url)
        try Check.require(try RuntimeConfiguration.load(url).decodePreroutingMargin == 0, "An ungated margin was refused")
        execution["decode_next_layer_prerouting"] = false
        object2["execution"] = execution
        try JSONSerialization.data(withJSONObject: object2).write(to: url)
        try Check.require(!(try RuntimeConfiguration.load(url).decodeNextLayerPrerouting), "Explicit false was not honoured")
    }
}

extension DecodePreroutingSelfChecks {
    /// `decode_prerouting_read_hygiene` through DecodePrerouting.adopt: after the
    /// target window holds its own handles, the issued predictions outside the
    /// route that are still queued are cancelled (never read) and counted; a
    /// started false prediction finishes; routed predictions are adopted with
    /// one read each and their own bytes; existing counters keep their meaning.
    package static func readHygieneAdoption() throws {
        let ledger = try MemoryLedger(ceiling: 600_000_000, transientMargin: 1_000_000)
        let fixtureBound = try ledger.reserve(bytes: 64 * 1024 * 1024, category: "prerouting_check_fixture")
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let hc = try FusedHCDecode(arithmetic: kernels.arithmetic, ledger: ledger)
            let dense = try PinnedDenseLinear(ledger: ledger)
            let selector = try BatchedTopIndices(ledger: ledger)
            let fixture = try Fixture(device: device)
            let residualBF16 = (0..<10240).map { (i: Int) in BF16.encode(Float((i * 73 + 5) % 1009 - 504) / 29) }
            guard
                let input = residualBF16.withUnsafeBytes({
                    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                })
            else {
                throw EngineError("Pre-routing fixture input allocation failed")
            }
            let log = ReadLog()
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            func waitStarted(_ label: String) throws {
                guard started.wait(timeout: .now() + 10) == .success else {
                    throw EngineError("Read-hygiene adoption fixture timed out: " + label)
                }
            }
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, readHygiene: true,
                packedAvailable: { false }
            ) { key, _, bytes in
                log.add(key)
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Pre-routing fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.layer * 16 + key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let factory = try execution.preroutingFactory()
            let owner = try DecodePrerouting(
                device: device, ledger: ledger, listener: listener(), hc: hc, dense: dense, selector: selector,
                margin: 0, readHygiene: true)
            try Check.require(
                owner.counts.readHygieneCancelled == 0 && owner.counts.readHygieneCancelRaces == 0, "Read-hygiene counters absent when on")
            let command = queue.makeCommandBuffer()!
            // Whatever the fixture router predicts, the factory issues 3, 4, 5 and 6 of target layer 1.
            let batch = try owner.stage(
                command: command, afterAttentionBF16: input, moeHC: fixture.hc, router: fixture.router, target: 1,
                operation: { _, layer, keepGoing in try factory([3, 4, 5, 6], layer, keepGoing) }, discard: execution.earlyDemandDiscard())
            command.commit()
            command.waitUntilCompleted()
            try waitStarted("first prediction")
            try waitStarted("second prediction")  // Two prefetch lanes: 3 and 4 run, 5 and 6 queue.
            let window = try execution.makeWindow(layer: 1, experts: [3, 5])
            try waitStarted("promoted demand")
            let unrouted = owner.adopt(batch, actual: [3, 5], cancelUnrouted: execution.preroutingCancelUnrouted()).map(\.key.expert)
                .sorted()
            let counts = owner.counts
            try Check.require(unrouted == [4, 6], "Adoption did not hand back the unrouted predictions: \(unrouted)")
            try Check.require(
                counts.batchesAdopted == 1 && counts.predictionsIssued == 4 && counts.predictedHits == 2 && counts.predictedFalse == 2
                    && counts.readHygieneCancelled == 1 && counts.readHygieneCancelRaces == 0 && counts.predictionsFailed == 0
                    && counts.bytesPredictedRead == UInt64(3 * PackStore.expertBytes)
                    && counts.bytesPredictedFalse == UInt64(PackStore.expertBytes),
                "Read-hygiene adoption counters differ: \(counts)")
            try Check.require(
                log.count == 3 && execution.readHygieneCounts?.cancelledQueued == 1
                    && execution.snapshot().io.drives.allSatisfy({ $0.queuedPrefetch == 0 }), "Cancelled prediction was read or left queued"
            )
            gate.signal()
            gate.signal()
            gate.signal()
            for expert in [3, 5] {
                try window.consume(expert: expert) { buffer in
                    try Check.require(
                        buffer.bytes[0] == UInt8(16 + expert) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(16 + expert),
                        "Adopted prediction carried wrong bytes under read hygiene")
                }
            }
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            let snapshot = execution.snapshot().cache
            try Check.require(
                log.count == 3 && snapshot.inFlightReads == 0 && snapshot.pinnedSlots == 0,
                "Read-hygiene adoption re-read a record or left a pin: \(snapshot)")
            try execution.finishSequence()
        }
        try ledger.release(fixtureBound)
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Read-hygiene adoption fixture leaked")
    }
}
