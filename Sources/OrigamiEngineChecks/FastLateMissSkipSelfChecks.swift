import Dispatch
import Foundation
import Metal
import OrigamiEngine

/// Standalone checks for the opt-in fast late-miss skip (no XCTest): the gate decision and renormalization,
/// the window's late definition (only reads without an outcome at the join point; resident and finished records
/// are never skipped), uncancelled completion of skipped reads into the cache, placeholder binding, the read
/// boundary join, refusals, the configuration block and the report key.
package enum FastLateMissSkipSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks

    package static func decision() throws {
        let ids = [40, 3, 17, 250, 9, 11, 500, 64, 128, 7]
        let weights: [Float] = [0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
        let mass = weights.reduce(Float(0), +)
        func near(_ a: Float, _ b: Float) -> Bool { abs(a - b) <= 1e-6 }
        let none = try FastLateMissSkip.decide(ids: ids, weights: weights, late: [], tau: 1, capacity: 10)
        try Check.require(
            none.skipped.isEmpty && !none.keptHeaviestLate && none.weights.map(\.bitPattern) == weights.map(\.bitPattern),
            "Fast skip changed gates without a late record")
        // tau = 1 skips every late record (rank order); kept gates keep their ratios and the routed mass.
        let all = try FastLateMissSkip.decide(ids: ids, weights: weights, late: [7, 500, 17], tau: 1, capacity: 10)
        try Check.require(
            all.skipped == [17, 500, 7] && [2, 6, 9].allSatisfy { all.weights[$0] == 0 }
                && near(all.weights.reduce(0, +), mass) && near(all.skippedWeights.reduce(0, +), 0.19 / mass)
                && near(all.weights[0] / all.weights[1], weights[0] / weights[1]) && !all.keptHeaviestLate,
            "Fast skip at tau 1 did not zero the late gates and renormalize to the routed mass: \(all.weights)")
        // tau = 0.04: only late records with normalized gate below tau.
        let low = try FastLateMissSkip.decide(ids: ids, weights: weights, late: [17, 500, 7], tau: 0.04, capacity: 10)
        try Check.require(
            low.skipped == [7] && low.weights[9] == 0 && low.weights[6] > 0 && low.weights[2] > 0,
            "Fast skip ignored tau: \(low.skipped)")
        // Every routed record late: the heaviest is kept and carries the whole mass.
        let every = try FastLateMissSkip.decide(ids: ids, weights: weights, late: ids, tau: 1, capacity: 10)
        try Check.require(
            every.keptHeaviestLate && every.skipped == Array(ids.dropFirst()) && near(every.weights[0], mass)
                && every.weights.dropFirst().allSatisfy { $0 == 0 }, "All-late layer did not keep its heaviest record")
        // Skipping both late records would leave only zero gates: the heaviest late one is kept instead.
        let zeroTail: [Float] = [0.30, 0.10] + [Float](repeating: 0, count: 8)
        let onlyZeroKept = try FastLateMissSkip.decide(ids: ids, weights: zeroTail, late: [40, 3], tau: 1, capacity: 10)
        try Check.require(
            onlyZeroKept.skipped == [3] && onlyZeroKept.keptHeaviestLate && near(onlyZeroKept.weights[0], 0.40),
            "Skip left only zero-weight gates")
        // The skipped-read reserve: only the lightest late records within capacity are skipped, the rest are declined.
        let partial = try FastLateMissSkip.decide(ids: ids, weights: weights, late: [7, 500, 17], tau: 1, capacity: 2)
        try Check.require(
            partial.skipped == [500, 7] && partial.declinedForCapacity == 1 && partial.weights[2] > 0
                && near(partial.weights.reduce(0, +), mass), "Capacity did not keep the heaviest late record: \(partial.skipped)")
        let full = try FastLateMissSkip.decide(ids: ids, weights: weights, late: [7, 500, 17], tau: 1, capacity: 0)
        try Check.require(
            full.skipped.isEmpty && full.declinedForCapacity == 3 && full.weights.map(\.bitPattern) == weights.map(\.bitPattern),
            "A full reserve still skipped or changed gates")
        let fewer = try FastLateMissSkip.decide(ids: ids, weights: weights, late: ids, tau: 1, capacity: 3)
        try Check.require(
            fewer.skipped == [64, 128, 7] && fewer.declinedForCapacity == 7 && !fewer.keptHeaviestLate,
            "All-late layer ignored the reserve: \(fewer.skipped)")
        try Check.require(
            none.declinedForCapacity == 0 && all.declinedForCapacity == 0 && every.declinedForCapacity == 0,
            "Declines counted without a full reserve")
        let nanGate: [Float] = [Float.nan] + weights.dropFirst()
        let negativeGate: [Float] = [Float(-0.3)] + weights.dropFirst()
        let zeroMass = [Float](repeating: 0, count: 10)
        let nineIDs = Array(ids.dropLast())
        let nineGates = Array(weights.dropLast())
        let cases: [(String, [Int], [Float], [Int], Double)] = [
            ("an unrouted late expert", ids, weights, [41], 1.0), ("a duplicate late expert", ids, weights, [7, 7], 1.0),
            ("tau 0", ids, weights, [7], 0.0), ("tau above 1", ids, weights, [7], 1.5), ("tau NaN", ids, weights, [7], Double.nan),
            ("nine routed experts", nineIDs, nineGates, [7], 1.0), ("a NaN gate", ids, nanGate, [7], 1.0),
            ("a negative gate", ids, negativeGate, [7], 1.0), ("zero routed mass", ids, zeroMass, [7], 1.0),
        ]
        for (label, i, w, late, tau) in cases {
            var refused = false
            do { _ = try FastLateMissSkip.decide(ids: i, weights: w, late: late, tau: tau, capacity: 10) } catch is EngineError {
                refused = true
            }
            try Check.require(refused, "Fast skip accepted " + label)
        }
    }

    private final class Gates: @unchecked Sendable {
        let lock = NSLock()
        var semaphores: [Int: DispatchSemaphore] = [:], reads: [Int: Int] = [:]
        func semaphore(_ expert: Int) -> DispatchSemaphore {
            lock.lock()
            defer { lock.unlock() }
            if let value = semaphores[expert] { return value }
            let value = DispatchSemaphore(value: 0)
            semaphores[expert] = value
            return value
        }
        func read(_ expert: Int) {
            lock.lock()
            reads[expert, default: 0] += 1
            lock.unlock()
        }
        func count(_ expert: Int) -> Int {
            lock.lock()
            defer { lock.unlock() }
            return reads[expert] ?? 0
        }
    }

    /// Experts 0-5 are resident at request, 6's read finishes before the join point, 7-9 are still running there.
    /// The skip sees exactly [7, 8, 9]; skipping 7 and 8 binds them to the first resident record, never cancels
    /// their reads, and a read boundary joins them into the cache (a later window finds them resident, unread).
    package static func windowSkipsOnlyLateReads() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let gates = Gates()
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 32, slotsPerSlab: 8, demandWindow: 16, readConcurrency: 4)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration,
                fastSkipReserve: FastLateMissSkip.readReserve, packedAvailable: { false }
            ) { key, _, bytes in
                gates.read(key.expert)
                if key.expert >= 6, gates.semaphore(key.expert).wait(timeout: .now() + 10) != .success {
                    throw EngineError("Fast skip fixture read timed out")
                }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            let warm = try execution.makeWindow(layer: 0, experts: Array(0..<6))
            for expert in 0..<6 { try warm.consume(expert: expert) { _ in } }
            try execution.beginDecodePhase()
            let window = try execution.makeWindow(layer: 0, experts: Array(0..<10))
            try Check.require(window.residentAtRequestCount == 6, "Fast skip fixture warm records were not resident")
            gates.semaphore(6).signal()
            for _ in 0..<5_000 where execution.snapshot().cache.inFlightReads > 3 { Thread.sleep(forTimeInterval: 0.001) }
            try Check.require(execution.snapshot().cache.inFlightReads == 3, "Fast skip fixture did not finish record 6 first")
            var seenLate: [Int] = []
            var seenCapacity = -1
            var readyIDs: [Int] = []
            var readyBuffers: [ExpertBuffer] = []
            var finishIDs: [Int] = []
            var finishBuffers: [ExpertBuffer] = []
            var finishBytes: [UInt8] = []
            try window.consumeTenReadyFirst(
                experts: Array(0..<10),
                skipLate: { late, capacity in
                    seenLate = late
                    seenCapacity = capacity
                    gates.semaphore(9).signal()  // the kept late record may now finish
                    return [7, 8]
                },
                ready: { ids, buffers in
                    readyIDs = ids
                    readyBuffers = buffers
                },
                finish: { (ids: [Int], buffers: [ExpertBuffer]) -> Void in
                    finishIDs = ids
                    finishBuffers = buffers
                    finishBytes = buffers.map { $0.bytes[0] }
                })
            try Check.require(
                readyIDs == Array(0..<6) && seenLate == [7, 8, 9] && finishIDs == [6, 7, 8, 9]
                    && seenCapacity == FastLateMissSkip.readReserve,
                "Fast skip late set differs: ready \(readyIDs), late \(seenLate), finish \(finishIDs)")
            try Check.require(
                finishBuffers.count == 4 && readyBuffers.count == 6 && finishBuffers[1] === readyBuffers[0]
                    && finishBuffers[2] === readyBuffers[0] && finishBytes == [6, 0, 0, 9],
                "Skipped ranks were not bound to the first pinned record: \(finishBytes)")
            let held = execution.snapshot()
            try Check.require(
                held.io.drives.allSatisfy { $0.cancelled == 0 } && held.cache.inFlightReads == 2
                    && gates.count(7) == 1 && gates.count(8) == 1, "Skipped reads were cancelled or repeated")
            gates.semaphore(7).signal()
            gates.semaphore(8).signal()
            try execution.joinReadProducers()  // joins the detached reads and retires their handoff pins
            let joined = execution.snapshot()
            try Check.require(
                joined.cache.inFlightReads == 0 && joined.cache.pinnedSlots == 0
                    && joined.io.drives.allSatisfy { $0.cancelled == 0 }, "Read boundary left skipped reads or pins: \(joined.cache)")
            let later = try execution.makeWindow(layer: 0, experts: [7, 8])
            try Check.require(later.residentAtRequestCount == 2, "Skipped reads did not complete into the cache")
            for expert in [7, 8] {
                try later.consume(expert: expert) { buffer in
                    try Check.require(
                        buffer.bytes[0] == UInt8(expert) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(expert),
                        "Skipped record carried wrong bytes")
                }
            }
            try Check.require(gates.count(7) == 1 && gates.count(8) == 1, "A skipped record was read twice")
            // All ten resident: nothing is late; naming a record that is not late is refused and leaks no pin.
            let resident = try execution.makeWindow(layer: 0, experts: Array(0..<10))
            var refused = false
            var residentLate: [Int]?
            do {
                try resident.consumeTenReadyFirst(
                    experts: Array(0..<10),
                    skipLate: { late, _ in
                        residentLate = late
                        return [3]
                    },
                    ready: { _, _ in }, finish: { (_: [Int], _: [ExpertBuffer]) -> Void in })
            } catch is EngineError { refused = true }
            try Check.require(
                refused && residentLate == [] && execution.snapshot().cache.pinnedSlots == 0,
                "Fast skip accepted a resident record or reported one late")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Fast skip fixture leaked its reservations")
    }

    /// Reserve starvation: skipped reads stay outstanding demand jobs, and the exact limits (demand reserve 10 here,
    /// all on one drive) refuse a demand submit instead of waiting. Unbounded, two layers of four skipped misses plus a
    /// third layer's four demand reads need 12 records and the third window is refused. With a reserve of 6: skips stop
    /// at 6 outstanding (the heaviest late records are joined instead and counted), every window is admitted, a full
    /// ten-miss window still fits beside a full reserve (16 records, beyond the exact 10), and no skipped read is cancelled.
    package static func readReserveBoundsSkippedReads() throws {
        let ledger = try MemoryLedger(ceiling: 600_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let gates = Gates()
            let reserve = 6
            // Ten lanes: every blocked read runs, so released records never queue behind held ones.
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 128, slotsPerSlab: 8, demandWindow: 10, readConcurrency: 10)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, fastSkipReserve: reserve, packedAvailable: { false }
            ) {
                key, _, bytes in
                let id = key.layer * 16 + key.expert
                gates.read(id)
                if key.expert >= 6 || key.layer == 4, gates.semaphore(id).wait(timeout: .now() + 20) != .success {
                    throw EngineError("Fast skip reserve fixture read timed out")
                }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            for layer in 0..<4 {
                let warm = try execution.makeWindow(layer: layer, experts: Array(0..<6))
                for expert in 0..<6 { try warm.consume(expert: expert) { _ in } }
            }
            try execution.beginDecodePhase()
            func outstanding() -> Int { execution.snapshot().io.drives.map(\.outstanding).max() ?? 0 }
            let weights: [Float] = [0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
            var capacities: [Int] = []
            var skipped: [[Int]] = []
            var declined = 0
            for layer in 0..<4 {
                let window = try execution.makeWindow(layer: layer, experts: Array(0..<10))
                try Check.require(outstanding() <= 10 + reserve, "Skipped reads exceeded the reserve: \(outstanding())")
                try window.consumeTenReadyFirst(
                    experts: Array(0..<10),
                    skipLate: { late, capacity in
                        let decision = try FastLateMissSkip.decide(
                            ids: Array(0..<10), weights: weights, late: late, tau: 1, capacity: capacity)
                        capacities.append(capacity)
                        skipped.append(decision.skipped)
                        declined += decision.declinedForCapacity
                        for expert in late where !decision.skipped.contains(expert) { gates.semaphore(layer * 16 + expert).signal() }
                        return decision.skipped
                    }, ready: { _, _ in }, finish: { (_: [Int], _: [ExpertBuffer]) -> Void in })
            }
            try Check.require(
                capacities == [6, 2, 0, 0] && skipped == [[6, 7, 8, 9], [8, 9], [], []] && declined == 10,
                "Reserve did not bound the skips: capacity \(capacities), skipped \(skipped), declined \(declined)")
            try Check.require(
                execution.snapshot().io.drives.allSatisfy { $0.cancelled == 0 }
                    && [6, 7, 8, 9, 24, 25].allSatisfy { gates.count($0) == 1 }, "Skipped reads were cancelled or repeated")
            let full = try execution.makeWindow(layer: 4, experts: Array(0..<10))
            try Check.require(outstanding() == 10 + reserve, "A full window beside a full reserve was not admitted: \(outstanding())")
            for id in [6, 7, 8, 9, 24, 25] + (64..<74) { gates.semaphore(id).signal() }
            full.close()
            try execution.joinReadProducers()
            let joined = execution.snapshot()
            try Check.require(
                joined.cache.inFlightReads == 0 && joined.cache.pinnedSlots == 0, "Reserve fixture left reads or pins: \(joined.cache)")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Fast skip reserve fixture leaked its reservations")
    }

    /// The configuration only allows fast (`fast.allow_opt_in` with τ in (0, 1]): off by default and in the trace
    /// fixture (whose `fast` block says enabled false); `fast.enabled: true` is refused (fast never runs process-wide); the report key
    /// is absent for exact requests and present for fast ones.
    package static func configurationAndReport() throws {
        try Check.require(RuntimeConfiguration().fastOptInTau == nil, "Fast allowance defaults on")
        let base = CheckFixtures.file("config/trace-fast-late-miss-skip-8192.json")
        try Check.require(try RuntimeConfiguration.load(base).fastOptInTau == nil, "The fast late-miss skip trace fixture allows fast")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-fast-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: base)) as! [String: Any]
        func load(_ fast: Any?) throws -> RuntimeConfiguration {
            var object = original
            object["fast"] = fast
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return try RuntimeConfiguration.load(url)
        }
        let one: [String: Any] = ["enabled": false, "allow_opt_in": true, "late_miss_skip_tau": 1.0]
        let quarter: [String: Any] = ["enabled": false, "allow_opt_in": true, "late_miss_skip_tau": 0.25]
        let refusedAllowance: [String: Any] = ["enabled": false, "allow_opt_in": false, "late_miss_skip_tau": 0.5]
        let noAllowance: [String: Any] = ["enabled": false, "late_miss_skip_tau": 0.5]
        try Check.require(try load(one).fastOptInTau == 1, "tau 1 was not honoured")
        try Check.require(try load(quarter).fastOptInTau == 0.25, "tau 0.25 was not honoured")
        try Check.require(try load(refusedAllowance).fastOptInTau == nil, "allow_opt_in false allowed fast")
        try Check.require(try load(noAllowance).fastOptInTau == nil, "A fast block without allow_opt_in allowed fast")
        try Check.require(try load(nil).fastOptInTau == nil, "Absent fast was allowed")
        // A process-wide `enabled` never turns fast on; it is ignored with a warning.
        try Check.require(
            try load(["enabled": true, "late_miss_skip_tau": 1.0] as [String: Any]).fastOptInTau == nil,
            "Process-wide fast.enabled allowed fast")
        let invalid: [(String, [String: Any])] = [
            ("no tau", ["enabled": false, "allow_opt_in": true]),
            ("tau 0", ["enabled": false, "allow_opt_in": true, "late_miss_skip_tau": 0]),
            ("tau above 1", ["enabled": false, "allow_opt_in": true, "late_miss_skip_tau": 1.5]),
            ("a boolean tau", ["enabled": false, "allow_opt_in": true, "late_miss_skip_tau": true]),
            ("a string allowance", ["enabled": false, "allow_opt_in": "true", "late_miss_skip_tau": 1.0]),
            ("a numeric allowance", ["enabled": false, "allow_opt_in": 1, "late_miss_skip_tau": 1.0]),
        ]
        for (label, fast) in invalid {
            var refused = false
            do { _ = try load(fast) } catch is EngineError { refused = true }
            try Check.require(refused, "Configuration accepted fast with " + label)
        }
        var refused = false
        do { _ = try load("on") } catch is EngineError { refused = true }
        try Check.require(refused, "Configuration accepted a non-object fast block")
    }

    /// Per-request opt-in. Exact requests take the unchanged path (no τ, so the decode layer
    /// builds no skip closure) with or without the allowance; a non-exact request runs with the allowed τ and is refused
    /// without it (`notAllowed`, never run exact instead). Requests parse `origami_mode` (absent/null/"exact"/"fast";
    /// anything else 400) and the `<alias>-fast` model id (fast; with `origami_mode: "exact"` 400);
    /// usage keeps its exact-only keys for exact responses and names the level for the others.
    package static func requestModeSelection() throws {
        let faster = QualityDial.level("fast")!
        try Check.require(
            try FastLateMissSkip.requestTau(.exact, allowedTau: nil) == nil
                && FastLateMissSkip.requestTau(.exact, allowedTau: 1) == nil
                && FastLateMissSkip.requestTau(faster, allowedTau: 1) == 1
                && FastLateMissSkip.requestTau(.lateMissSkip, allowedTau: 0.25) == 0.25, "Request mode selection differs")
        var refusal: EngineError?
        do { _ = try FastLateMissSkip.requestTau(faster, allowedTau: nil) } catch let error as EngineError { refusal = error }
        try Check.require(
            refusal?.code == "fast_not_allowed" && refusal?.description == FastLateMissSkip.notAllowed,
            "A fast request without the allowance was not refused")

        func parse(_ extra: [String: Any]) throws -> ServingRequest {
            var body: [String: Any] = ["model": "qwen3.8-flash-next:4bit", "messages": [["role": "user", "content": "hi"]]]
            for (key, value) in extra { body[key] = value }
            return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8192)
        }
        try Check.require(
            try parse([:]).quality == .exact && parse(["origami_mode": "exact"]).quality == .exact
                && parse(["origami_mode": "fast"]).quality == faster && parse(["origami_mode": NSNull()]).quality == .exact,
            "origami_mode parsing differs")
        for bad in ["FAST", "", 1, true, ["fast"]] as [Any] {
            var refused = false
            do { _ = try parse(["origami_mode": bad]) } catch let failure as ServingFailure { refused = failure.status == 400 }
            try Check.require(refused, "origami_mode accepted \(bad)")
        }
        // The `<alias>-fast` model id is the same per-request opt-in; the plain alias stays exact and is echoed as sent.
        let fastID = "qwen3.8-flash-next:4bit-fast"
        let named = try parse(["model": fastID])
        let redundant = try parse(["model": fastID, "origami_mode": "fast"])
        let plain = try parse([:])
        try Check.require(
            named.quality == faster && named.model == fastID && redundant.quality == faster
                && plain.quality == .exact && plain.model == "qwen3.8-flash-next:4bit", "The fast model id did not select fast")
        for (extra, status) in [(["model": fastID, "origami_mode": "exact"], 400)] as [([String: Any], Int)] {
            var refused = false
            do { _ = try parse(extra) } catch let failure as ServingFailure { refused = failure.status == status }
            try Check.require(refused, "Fast model id case was not refused with \(status): \(extra)")
        }

        let exact = try ServingWire.usage(GenerationStatistics(promptTokens: 5, completionTokens: 2), family: .qwen)
        let fast = try ServingWire.usage(GenerationStatistics(promptTokens: 5, completionTokens: 2, quality: faster), family: .qwen)
        try Check.require(
            Set(exact.keys) == ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"]
                && Set(fast.keys) == Set(exact.keys).union(["origami_mode"]) && fast["origami_mode"] as? String == "fast",
            "Usage does not report the level alone")
    }

    /// The cache/state rule. Exact decode rows keep the conversation epoch, so an exact-only timeline is unchanged;
    /// fast rows carry the fast epoch, and a timeline holding one anywhere is not exact (an exact request on a
    /// configuration that allows fast never continues from it: live, frontier image or durable image). Prefill rows
    /// never count, and another conversation epoch is never taken for this one's exact rows.
    package static func conversationTimelineRule() throws {
        let epoch = "origami-native-" + String(repeating: "a", count: 64)
        try Check.require(
            FastLateMissSkip.timelineEpoch(epoch, quality: .exact) == epoch
                && FastLateMissSkip.timelineEpoch(epoch, quality: .lateMissSkip) != epoch, "Timeline epochs differ from the rule")
        let prefill = try PrefillReadSchedule(promptTokens: 300, computeTile: 256, readScope: 256).checkpointSegments(
            arithmeticEpoch: epoch)
        func decode(_ start: Int, _ mode: QualityDial) throws -> CheckpointSegment {
            try PrefillReadSchedule.decodeSegment(start: start, arithmeticEpoch: FastLateMissSkip.timelineEpoch(epoch, quality: mode))
        }
        try Check.require(
            try decode(300, .exact) == PrefillReadSchedule.decodeSegment(start: 300, arithmeticEpoch: epoch),
            "Exact decode rows changed their signature")
        let exact = try prefill + [decode(300, .exact), decode(301, .exact)]
        let fastTail = try exact + [decode(302, .lateMissSkip)]
        let fastMiddle = try prefill + [decode(300, .exact), decode(301, .lateMissSkip), decode(302, .exact)]
        try Check.require(
            try FastLateMissSkip.isExactTimeline(prefill, arithmeticEpoch: epoch)
                && FastLateMissSkip.isExactTimeline(exact, arithmeticEpoch: epoch)
                && !FastLateMissSkip.isExactTimeline(fastTail, arithmeticEpoch: epoch)
                && !FastLateMissSkip.isExactTimeline(fastMiddle, arithmeticEpoch: epoch), "Fast decode rows were not recognized")
        try Check.require(
            try !FastLateMissSkip.isExactTimeline(exact, arithmeticEpoch: epoch + "-other"),
            "Another epoch's decode rows were taken as exact")
    }
}
