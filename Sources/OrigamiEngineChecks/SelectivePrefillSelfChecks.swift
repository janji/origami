import Darwin
import Foundation
import Metal
import OrigamiEngine

package enum SelectivePrefillSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try value() else { throw EngineError(message) }
    }
    private static func rejects(_ body: () throws -> Void) throws {
        do { try body() } catch { return }
        throw EngineError("Selective prefill accepted invalid publication")
    }
    package static func policy() throws {
        var policy = PrefillReadPolicy()
        try require(!policy.begin(rows: 8192), "Unobserved coverage enabled broad prefill")
        for _ in 0..<47 { policy.observe(experts: 500, domain: 512) }
        policy.end(completed: true, domain: 512)
        try require(policy.begin(rows: 8192), "Dense observed coverage did not admit broad prefill")
        for _ in 0..<47 { policy.observe(experts: 100, domain: 512) }
        policy.end(completed: true, domain: 512)
        try require(!policy.begin(rows: 8192), "Sparse observed coverage retained broad prefill")
        for _ in 0..<47 { policy.observe(experts: 512, domain: 512) }
        policy.end(completed: false, domain: 512)
        try require(!policy.begin(rows: 8192), "Abandoned scope trained broad prefill")
        for _ in 0..<47 { policy.observe(experts: 512, domain: 512) }
        policy.end(completed: true, domain: 512)
        // The threshold follows the routing geometry: 10 of 512 experts per row give 351 rows, not a fixed 1,024.
        try require(PrefillReadPolicy.minimumBroadRows == 351, "Broad threshold is not derived from top-k and experts")
        // The same rule for GLM-5.3 (8 of 288) and DeepSeek-V4.1 (6 of 384), whose read-ahead starts there.
        try require(
            PromptReadRows.minimum(expertsPerToken: 8, experts: 288) == 246
                && PromptReadRows.minimum(expertsPerToken: 6, experts: 384) == 439
                && PromptReadRows.minimum(expertsPerToken: 4, experts: 4) == 1,
            "Family read-ahead thresholds are not derived from top-k and experts")
        for rows in [1, 41, 236, 255, 256, 350] {
            try require(!policy.begin(rows: rows), "Short scope enabled broad prefill")
            policy.end(completed: false, domain: 512)
        }
        for rows in [351, 600, 1024] {
            try require(policy.begin(rows: rows), "Large dense scope lost its learned coverage")
            policy.end(completed: false, domain: 512)
        }
    }

    private final class Reads: @unchecked Sendable {
        let lock = NSLock(), blocked = DispatchSemaphore(value: 0)
        var keys: [ExpertKey] = []
        var blockExpert: Int?
        func read(_ key: ExpertKey, into destination: UnsafeMutableRawBufferPointer) throws {
            lock.lock()
            keys.append(key)
            let shouldBlock = blockExpert == key.expert
            lock.unlock()
            if shouldBlock, blocked.wait(timeout: .now() + .seconds(5)) != .success {
                throw EngineError("Ready expert did not bypass the blocked earlier read")
            }
            destination.baseAddress!.storeBytes(of: UInt32(key.expert), as: UInt32.self)
        }
        var snapshot: [ExpertKey] {
            lock.lock()
            defer { lock.unlock() }
            return keys
        }
    }
    private static func configuration() -> ExpertExecutionConfiguration {
        ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 528, slotsPerSlab: 8,
            demandWindow: 16, readConcurrency: 4, prefillReadAhead: true, prefillLayerCommand: true)
    }
    private static func retired(_ engine: ExpertExecution) throws {
        let s = engine.snapshot()
        try require(
            s.cache.pinnedSlots == 0 && s.cache.inFlightReads == 0 && s.cache.reservedSlots == 0
                && s.io.drives.allSatisfy { $0.outstanding == 0 }, "Selective prefill retained a consumer or writer")
    }

    private final class PriorityFixture: @unchecked Sendable {
        let lock = NSLock(), started = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let done = DispatchSemaphore(value: 0)
        var seen: [Int] = [], failure: Error?
        func record(_ expert: Int) {
            lock.lock()
            seen.append(expert)
            lock.unlock()
        }
        func fail(_ error: Error) {
            lock.lock()
            failure = error
            lock.unlock()
        }
        var result: ([Int], Error?) {
            lock.lock()
            defer { lock.unlock() }
            return (seen, failure)
        }
    }
    /// A broad producer blocked on expert zero must serve routed expert 511
    /// next, without reading IDs 1...510 first or increasing its one-read bound.
    package static func demandBypass() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
        let ledger = try MemoryLedger(ceiling: 1_800_000_000, transientMargin: 16_000_000)
        try autoreleasepool {
            let metadata = try ledger.reserve(bytes: PrefillExpertReadAhead.metadataBytes, category: "selective-priority-fixture")
            defer { try? ledger.release(metadata) }
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 528,
                slotsPerSlab: 8, demandWindow: 2, readConcurrency: 1, prefillReadAhead: true)
            let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: configuration.cacheBytes)
            let limit = IOScheduler.Limits(
                maximumConcurrent: 1, reservedDemandConcurrent: 1,
                maximumOutstanding: 2, reservedDemandOutstanding: 1,
                reservedDemandBytes: 10 * UInt64(PackStore.expertBytes))
            let scheduler = try IOScheduler(ledger: ledger, internalLimits: limit, externalLimits: limit)
            let fixture = PriorityFixture()
            let loader = SourceExpertLoader(cache: cache, scheduler: scheduler, packedAvailable: { false }) { key, _, bytes in
                fixture.record(key.expert)
                if key.expert == 0 {
                    fixture.started.signal()
                    guard fixture.release.wait(timeout: .now() + .seconds(5)) == .success else {
                        throw EngineError("Priority fixture gate timed out")
                    }
                }
                bytes.baseAddress!.storeBytes(of: UInt32(key.expert), as: UInt32.self)
            }
            let ahead = try PrefillExpertReadAhead(
                loader: loader, generation: scheduler.currentGeneration,
                configuration: configuration, layers: 1)
            let consumer = DispatchGroup()
            defer {
                fixture.release.signal()
                ahead.close()
                consumer.wait()
                scheduler.shutdown()
            }
            try require(fixture.started.wait(timeout: .now() + .seconds(5)) == .success, "Broad fixture did not start")
            consumer.enter()
            DispatchQueue.global().async {
                defer {
                    consumer.leave()
                    fixture.done.signal()
                }
                do {
                    let key = try ExpertKey(layer: 0, expert: 511)
                    let request = try ahead.request(key, shouldContinue: { true })
                    let lease = try loader.joinAndPin(request)
                    defer { try? lease.retireAfterGPUCompletion() }
                    try require(lease.buffer.bytes.load(as: UInt32.self) == 511, "Priority bypass supplied another expert")
                    ahead.release([key][...])
                } catch { fixture.fail(error) }
            }
            let deadline = Date(timeIntervalSinceNow: 5)
            while ahead.snapshot().pendingDemands != 1 && Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
            try require(ahead.snapshot().pendingDemands == 1, "Demand did not reach the bounded producer")
            fixture.release.signal()
            try require(fixture.done.wait(timeout: .now() + .seconds(5)) == .success, "Priority demand did not finish")
            consumer.wait()
            ahead.close()
            let result = fixture.result
            if let failure = result.1 { throw failure }
            try require(
                Array(result.0.prefix(2)) == [0, 511] && result.0.filter { $0 == 511 }.count == 1,
                "Actual demand did not bypass the scan exactly once")
            try require(
                ahead.snapshot().maximumPending <= 1 && (ahead.snapshot().demandBypasses ?? 0) > 0,
                "Demand bypass exceeded its queue bound or lost attribution")
        }
        try require(ledger.snapshot().reserved == 0, "Priority bypass retained an owner")
    }

    /// Real bounded I/O: a resident expert must compute before an earlier cold
    /// read is released; all selected records are read exactly once and no
    /// unselected record is requested. Callback failures drain every reader.
    package static func routingAndCleanup() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
        let ledger = try MemoryLedger(ceiling: 1_000_000_000, transientMargin: 16_000_000)
        try autoreleasepool {
            let reads = Reads()
            let engine = try ExpertExecution(device: device, ledger: ledger, configuration: configuration(), packedAvailable: { false }) {
                key, _, out in
                try reads.read(key, into: out)
            }
            try engine.beginSequence()
            defer { try? engine.finishSequence() }
            let warm = try engine.makeWindow(layer: 0, experts: [0])
            try warm.consume(expert: 0) { _ in }
            reads.blockExpert = 1
            defer { reads.blocked.signal() }
            try engine.beginPrefillReadAhead(layers: 1, rows: 41)
            try require(engine.usesSelectivePrefill, "41-token scope selected broad scan")
            let window = try engine.makeWindow(layer: 0, experts: Array(0..<24))
            var seen: Set<Int> = []
            var residentWasFirst = false
            try window.consumePrefillReady { experts, buffers in
                if seen.isEmpty {
                    residentWasFirst = experts.contains(0) && !experts.contains(1)
                    reads.blocked.signal()
                }
                for (expert, buffer) in zip(experts, buffers) {
                    try require(
                        seen.insert(expert).inserted && buffer.bytes.load(as: UInt32.self) == UInt32(expert),
                        "Ready group repeated or changed an expert")
                }
            }
            try engine.finishPrefillReadAheadLayer(0)
            engine.finishPrefillReadAhead()
            try require(residentWasFirst && seen == Set(0..<24), "Resident prefill failed exact completion coverage")
            try require(
                reads.snapshot.count == 24 && Set(reads.snapshot.map(\.expert)) == Set(0..<24),
                "Selective prefill read unselected or duplicate records")
            try retired(engine)
            try engine.beginPrefillReadAhead(layers: 1, rows: 236)
            let failed = try engine.makeWindow(layer: 0, experts: Array(24..<48))
            try rejects { try failed.consumePrefillReady { _, _ in throw EngineError("Injected consumer failure") } }
            engine.finishPrefillReadAhead()
            try retired(engine)
            try require(
                engine.prefillSchedulingSnapshot.selectiveScopes == 2
                    && engine.prefillSchedulingSnapshot.broadScopes == 0
                    && engine.prefillSchedulingSnapshot.readyExperts == 24,
                "Coarse selective prefill counters differ")
        }
        try require(ledger.snapshot().reserved == 0, "Selective prefill I/O owners leaked")
    }

    private static func expert(_ key: ExpertKey, _ destination: UnsafeMutableRawBufferPointer) {
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

    /// Compare the original eight-job QMV batches with new 16-expert partial
    /// wave commands in reverse group order, including exceptional input bits.
    /// Canonical contribution bytes and final weighted reduction must match.
    package static func exactGroups() throws {
        let ledger = try MemoryLedger(ceiling: 1_400_000_000, transientMargin: 16_000_000)
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger)
            let fused = try FusedExpertExecution(
                arithmetic: kernels.arithmetic, ledger: ledger,
                groupedTiled: true, groupedBM32: true, groupedBM32MinRows: 32)
            let transfer = try PrefillMoETransfer()
            let reductions = try PinnedReductions(ledger: ledger)
            let engine = try ExpertExecution(
                device: kernels.device, ledger: ledger, configuration: configuration(), packedAvailable: { false }
            ) { key, _, out in
                expert(key, out)
            }
            for tokens in [41, 236] {
                try autoreleasepool {
                    var input = (0..<tokens * 2560).map { Float(($0 * 103 % 991) - 495) / 193 }
                    input[17] = Float(bitPattern: 0x7fc3_0000)
                    input[2560 + 19] = -0.0
                    var rows: [Int: [(Int, Int)]] = [:]
                    for token in 0..<tokens { for rank in 0..<10 { rows[(token + rank) % 24, default: []].append((token, rank)) } }
                    let aBudget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                    let bBudget = try PrefillMoEBudget(ledger: ledger, bytes: PrefillMoEBudget.bytesForScope(tokens: tokens))
                    let a = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: aBudget, transfer: transfer)
                    let b = try PrefillMoEWorkspace(input: input, selectedRows: rows, budget: bBudget, transfer: transfer)
                    try engine.beginSequence()
                    defer { try? engine.finishSequence() }
                    let control = try engine.makeWindow(layer: 0, experts: Array(0..<24))
                    for first in stride(from: 0, to: 24, by: 8) {
                        let experts = Array(first..<first + 8)
                        try control.consumeBatch(experts: experts) { buffers in
                            let jobs = zip(experts, buffers).map { expert, buffer in
                                FusedExpertExecution.DeviceJob(
                                    expert: expert, slots: rows[expert]!.map { UInt32($0.0 * 10 + $0.1) },
                                    buffer: buffer, groupedExpert: false, simdMatrix: true)
                            }
                            try fused.evaluateDeviceBatch(jobs, workspace: a)
                        }
                    }
                    try engine.beginPrefillReadAhead(layers: 1, rows: tokens)
                    let candidate = try engine.makeWindow(layer: 0, experts: Array(0..<24))
                    let before = fused.snapshot().completedCommands
                    try candidate.consumeLayer { buffers in
                        for range in [16..<24, 0..<16] {
                            let experts = Array(range)
                            let chosen = experts.map { buffers[$0] }
                            let selected = Dictionary(uniqueKeysWithValues: experts.map { ($0, rows[$0]!) })
                            try fused.evaluateDeviceLayer(
                                experts: experts, buffers: chosen, selectedRows: selected,
                                workspace: b, simdMatrix: true, waves: 8, groupedExpert: false, partial: true)
                            if range.lowerBound == 16 {
                                try rejects { _ = try b.reduce(weights: Array(repeating: 0.1, count: tokens * 10), using: reductions) }
                                try rejects {
                                    try fused.evaluateDeviceLayer(
                                        experts: experts, buffers: chosen, selectedRows: selected,
                                        workspace: b, simdMatrix: true, waves: 8, groupedExpert: false, partial: true)
                                }
                            }
                        }
                    }
                    try require(fused.snapshot().completedCommands - before == 2, "Selective groups retained eight-expert command joins")
                    try require(
                        b.completedSlots == tokens * 10
                            && memcmp(a.contributions.contents(), b.contributions.contents(), tokens * 10 * 2560 * 4) == 0,
                        "Partial QMV groups changed canonical contributions")
                    let weights = (0..<tokens * 10).map { Float($0 % 10 + 1) / 55 }
                    let aa = try a.reduce(weights: weights, using: reductions)
                    let bb = try b.reduce(weights: weights, using: reductions)
                    try require(aa.map(\.bitPattern) == bb.map(\.bitPattern), "Partial QMV group reduction changed bits")
                    try engine.finishPrefillReadAheadLayer(0)
                    engine.finishPrefillReadAhead()
                    try retired(engine)
                }
            }
        }
        try require(ledger.snapshot().reserved == 0, "Selective prefill GPU owners leaked")
    }
}
