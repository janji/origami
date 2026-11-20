import Dispatch
import Foundation
import Metal
import OrigamiEngine

/// Standalone checks for `decode_prerouting_read_hygiene` (no XCTest): queued-only
/// cancellation of unrouted predictions, reader-pool priority, the decode PLE
/// prefetch hold, and the off state.
package enum DecodePreroutingReadHygieneSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private final class Log: @unchecked Sendable {
        let lock = NSLock()
        var items: [String] = []
        func add(_ item: String) {
            lock.lock()
            items.append(item)
            lock.unlock()
        }
        var values: [String] {
            lock.lock()
            defer { lock.unlock() }
            return items
        }
        func count(_ item: String) -> Int { values.filter { $0 == item }.count }
    }
    private static func wait(_ semaphore: DispatchSemaphore, _ label: String) throws {
        guard semaphore.wait(timeout: .now() + 10) == .success else { throw EngineError("Read-hygiene fixture timed out: " + label) }
    }
    private static func drive(_ snapshot: ExpertExecution.Snapshot, _ source: IOSource = .internalSSD) -> IOScheduler.DriveSnapshot {
        snapshot.io.drives.first { $0.source == source }!
    }

    /// Cancelling an unrouted prediction that is still queued releases exactly
    /// its own admission (one outstanding job and one record of bytes on its
    /// drive, its reserved cache slot; no MemoryLedger change) and reads nothing;
    /// a started prediction and one shared with demand are left alone; the
    /// cancelled key, demanded later, is read once with its own bytes. The same
    /// holds for a speculative replica race (the decode path with tail replicas).
    package static func cancelReleasesExactly() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let log = Log()
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            // Two demand lanes plus pre-routing's two prefetch-only lanes per drive.
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, readHygiene: true,
                packedAvailable: { false }
            ) { key, _, bytes in
                log.add("\(key.layer).\(key.expert)")
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Read-hygiene fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.layer * 16 + key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let factory = try execution.preroutingFactory()
            guard let cancelUnrouted = execution.preroutingCancelUnrouted() else {
                throw EngineError("Read hygiene produced no cancellation path")
            }
            var issue = try factory([3, 4, 5, 6], 1, { true })
            try Check.require(issue.issued == 4 && issue.requests.count == 4, "Fixture predictions were not all issued: \(issue.issued)")
            try wait(started, "first prefetch")
            try wait(started, "second prefetch")
            let issuedState = drive(execution.snapshot())
            try Check.require(
                issuedState.runningPrefetch == 2 && issuedState.queuedPrefetch == 2 && log.values.count == 2,
                "Fixture did not hold two predicted reads running and two queued: \(issuedState)")
            // Authoritative route of layer 1 = {3, 5}: 3 is shared while running, 5 is promoted from the queue.
            let window = try execution.makeWindow(layer: 1, experts: [3, 5])
            try wait(started, "promoted demand")
            let held = execution.snapshot()
            let heldDrive = drive(held)
            let heldLedger = ledger.snapshot().reserved
            try Check.require(
                heldDrive.queuedPrefetch == 1 && log.count("1.5") == 1, "Promoted prediction did not start as demand: \(heldDrive)")
            // A request demand shares is never cancelled; it is counted as a race (expected 0 in the engine).
            let shared = cancelUnrouted(issue.requests.filter { $0.key.expert == 5 })
            try Check.require(shared.cancelled == 0 && shared.races == 1, "A demand-shared prediction was cancelled: \(shared)")
            let result = cancelUnrouted(issue.requests.filter { ![3, 5].contains($0.key.expert) })
            let after = execution.snapshot()
            let afterDrive = drive(after)
            try Check.require(
                result.cancelled == 1 && result.races == 0, "Unrouted cancellation differs (expected 4 started, 6 cancelled): \(result)")
            try Check.require(
                afterDrive.outstanding == heldDrive.outstanding - 1
                    && afterDrive.outstandingBytes == heldDrive.outstandingBytes - UInt64(PackStore.expertBytes)
                    && afterDrive.cancelled == heldDrive.cancelled + 1 && afterDrive.queuedPrefetch == 0
                    && afterDrive.runningPrefetch == heldDrive.runningPrefetch && afterDrive.runningDemand == heldDrive.runningDemand,
                "Cancellation released other than exactly one queued record of admission: \(afterDrive) vs \(heldDrive)")
            try Check.require(
                after.cache.activeSlots == held.cache.activeSlots - 1 && after.cache.inFlightReads == held.cache.inFlightReads - 1
                    && after.cache.pinnedSlots == held.cache.pinnedSlots && after.cache.evictions == held.cache.evictions
                    && ledger.snapshot().reserved == heldLedger,
                "Cancellation released other than the record's own slot: \(after.cache) vs \(held.cache)")
            try Check.require(
                log.count("1.6") == 0 && execution.readHygieneCounts?.cancelledQueued == 1,
                "A cancelled prediction was read or went uncounted")
            issue.requests.removeAll()
            gate.signal()
            gate.signal()
            gate.signal()
            for expert in [3, 5] {
                try window.consume(expert: expert) { buffer in
                    try Check.require(
                        buffer.bytes[0] == UInt8(16 + expert) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(16 + expert),
                        "Routed record carried wrong bytes after cancellation")
                }
            }
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            // The cancelled key, later demanded, is an ordinary miss: one read, its own bytes.
            let later = try execution.makeWindow(layer: 1, experts: [6])
            try wait(started, "later demand")
            gate.signal()
            try later.consume(expert: 6) { buffer in
                try Check.require(
                    buffer.bytes[0] == UInt8(16 + 6) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(16 + 6),
                    "A cancelled key demanded later carried wrong bytes")
            }
            try Check.require(
                log.count("1.6") == 1 && log.count("1.4") == 1 && log.values.count == 4, "Reads differ after cancellation: \(log.values)")
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try autoreleasepool {
            // Speculative replica race (tail replicas): the same queued-only cancellation.
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(8 * PackStore.expertBytes), slotsPerSlab: 8)
            let limits = IOScheduler.Limits(
                maximumConcurrent: 2, reservedDemandConcurrent: 1, maximumOutstanding: 8, reservedDemandOutstanding: 4,
                maximumOutstandingBytes: UInt64(8 * PackStore.expertBytes), reservedDemandBytes: UInt64(4 * PackStore.expertBytes))
            let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits, readHygiene: true)
            defer { scheduler.shutdown() }
            let arena = try ExpertReplicaArena(ledger: ledger, device: device)
            let log = Log()
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let loader = SourceExpertLoader(cache: cache, scheduler: scheduler, replicaArena: arena, packedAvailable: { true }) {
                key, _, bytes in
                log.add("\(key.expert)")
                started.signal()
                guard gate.wait(timeout: .now() + 10) == .success else { throw EngineError("Replica fixture read timed out") }
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try loader.setReplicasEnabled(true)
            let generation = scheduler.currentGeneration
            func speculative(_ expert: Int) throws -> ExpertLoadRequest {
                guard
                    case .issued(let request) = try loader.requestSpeculative(
                        ExpertKey(layer: 2, expert: expert), generation: generation,
                        source: .externalSSD)
                else {
                    throw EngineError("Replica fixture prediction was not issued")
                }
                return request
            }
            var running: ExpertLoadRequest? = try speculative(1)
            try wait(started, "replica primary")
            var queued: ExpertLoadRequest? = try speculative(2)  // One prefetch lane: queued behind the running primary.
            try Check.require(loader.cancelQueuedSpeculative(running!) == .notQueued, "A started speculative race was cancelled")
            try Check.require(loader.cancelQueuedSpeculative(queued!) == .cancelled, "A queued speculative race was not cancelled")
            try Check.require(loader.cancelQueuedSpeculative(queued!) == .notQueued, "A cancelled race was cancelled twice")
            loader.joinDiscarding(queued!)
            try Check.require(
                !loader.isInFlight(queued!) && cache.snapshot().inFlightReads == 1 && log.values == ["1"]
                    && scheduler.hygieneCounts()?.cancelledQueued == 1 && arena.snapshot().leasedRecords == 0,
                "Cancelled race did not retire its entry and slot without a read: \(cache.snapshot())")
            queued = nil
            running = nil
            gate.signal()
            for _ in 0..<2_000 where cache.snapshot().inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            let demand = try loader.request(ExpertKey(layer: 2, expert: 2), generation: generation, source: .externalSSD)
            try wait(started, "replica later demand")
            gate.signal()
            let loaded = try loader.joinAndPin(demand)
            try Check.require(
                loaded.buffer.bytes[0] == 2 && loaded.buffer.bytes[PackStore.expertBytes - 1] == 2,
                "Cancelled race key demanded later carried wrong bytes")
            try loaded.retireAfterGPUCompletion()
            gate.signal()
            gate.signal()  // Any hedge replica the demand may have raced.
            try loader.cancelAndJoin(generation)
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Read-hygiene cancellation fixture leaked")
    }

    /// One reader thread and a synthetic queue: demand spans and PLE rows are
    /// taken before queued predicted spans (FIFO within each class), every
    /// bypass is counted; a promoted prediction is ordinary FIFO again; with no
    /// tokens (hygiene off) the order is the plain FIFO of today.
    package static func poolPriorityOrder() throws {
        let (scoped, _) = IOReadPriority.makeForChecks()
        try Check.require(IOReadPriority.current == nil, "A thread outside any read had a read priority")
        let inside = IOReadPriority.run(scoped) { IOReadPriority.current === scoped }
        try Check.require(inside && IOReadPriority.current == nil, "Read-priority scope did not set or restore its thread token")

        func scenario(prioritized: Bool, promote: Bool) throws -> (order: [String], deferrals: UInt64) {
            let pool = try SpanReadPool(threads: 1)
            defer { pool.stop() }
            let (token, stats) = IOReadPriority.makeForChecks()
            let order = Log()
            let failures = Log()
            let group = DispatchGroup()
            let blocker = DispatchSemaphore(value: 0)
            let blocking = DispatchSemaphore(value: 0)
            group.enter()
            pool.enqueue {
                blocking.signal()
                _ = blocker.wait(timeout: .now() + 10)
                group.leave()
            }
            try wait(blocking, "pool blocker")
            func queued(_ count: Int) throws {
                for _ in 0..<5_000 where pool.queuedTasks < count { Thread.sleep(forTimeInterval: 0.001) }
                try Check.require(pool.queuedTasks == count, "Synthetic pool queue did not reach \(count) items")
            }
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                do {
                    try IOReadPriority.run(prioritized ? token : nil) { try pool.run((0..<3).map { index in { order.add("P\(index)") } }) }
                } catch { failures.add(String(describing: error)) }
            }
            try queued(3)
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                do { try pool.run((0..<3).map { index in { order.add("D\(index)") } }) } catch { failures.add(String(describing: error)) }
            }
            try queued(6)
            group.enter()
            pool.enqueue {
                order.add("R")
                group.leave()
            }
            try queued(7)
            if promote { token.promoteForChecks() }
            blocker.signal()
            guard group.wait(timeout: .now() + 10) == .success else { throw EngineError("Synthetic pool queue did not drain") }
            try Check.require(failures.values.isEmpty, "Synthetic pool runs failed: \(failures.values)")
            return (order.values, stats.poolDeferrals)
        }
        let fifo = ["P0", "P1", "P2", "D0", "D1", "D2", "R"]
        let prioritized = try scenario(prioritized: true, promote: false)
        try Check.require(
            prioritized.order == ["D0", "D1", "D2", "R", "P0", "P1", "P2"] && prioritized.deferrals == 4,
            "Reader pool did not serve demand and PLE before predicted spans: \(prioritized)")
        let promoted = try scenario(prioritized: true, promote: true)
        try Check.require(promoted.order == fifo && promoted.deferrals == 0, "A promoted prediction was still deferred: \(promoted)")
        let off = try scenario(prioritized: false, promote: false)
        try Check.require(off.order == fifo && off.deferrals == 0, "Reader pool without tokens is not plain FIFO: \(off)")
    }

    /// Scheduler: a predicted read runs with a deferrable token that demand
    /// adoption clears mid-read; a queued prediction is cancelled only while
    /// queued, releasing exactly its admission; a hold defers prefetch starts
    /// (counted once per job) but never demand; off, none of this exists.
    package static func schedulerHoldAndCancellation() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let bytes = 4096
        // One prefetch lane, one reserved demand lane.
        let limits = IOScheduler.Limits(
            maximumConcurrent: 2, reservedDemandConcurrent: 1, maximumOutstanding: 8, reservedDemandOutstanding: 4,
            maximumOutstandingBytes: UInt64(8 * bytes), reservedDemandBytes: UInt64(4 * bytes))
        func internalDrive(_ scheduler: IOScheduler) -> IOScheduler.DriveSnapshot {
            scheduler.snapshot().drives.first { $0.source == .internalSSD }!
        }
        do {
            // Off: no token, no hold, no queued-only cancellation; the queued prediction still runs.
            let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
            defer { scheduler.shutdown() }
            let log = Log()
            let gate = DispatchSemaphore(value: 0)
            let started = DispatchSemaphore(value: 0)
            let generation = scheduler.currentGeneration
            let first = try scheduler.submit(source: .internalSSD, priority: .prefetch, generation: generation, byteCount: bytes) { _ in
                log.add(IOReadPriority.current == nil ? "untokened" : "tokened")
                started.signal()
                _ = gate.wait(timeout: .now() + 10)
            }
            try wait(started, "off prefetch")
            let second = try scheduler.submit(source: .internalSSD, priority: .prefetch, generation: generation, byteCount: bytes) { _ in
                log.add("second")
            }
            try Check.require(
                scheduler.holdPrefetch() == nil && scheduler.hygieneCounts() == nil && !scheduler.cancelQueued(second),
                "Read hygiene acted with the flag off")
            gate.signal()
            guard case .succeeded = first.join(), case .succeeded = second.join() else {
                throw EngineError("Off-state predicted reads did not complete")
            }
            try Check.require(log.values == ["untokened", "second"], "Off-state predicted reads differ: \(log.values)")
        }
        let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits, readHygiene: true)
        defer { scheduler.shutdown() }
        let log = Log()
        let gate = DispatchSemaphore(value: 0)
        let started = DispatchSemaphore(value: 0)
        let generation = scheduler.currentGeneration
        let running = try scheduler.submit(source: .internalSSD, priority: .prefetch, generation: generation, byteCount: bytes) { _ in
            log.add("deferrable \(IOReadPriority.current?.isDeferrable == true)")
            started.signal()
            _ = gate.wait(timeout: .now() + 10)
            log.add("deferrable \(IOReadPriority.current?.isDeferrable == true)")
        }
        try wait(started, "tokened prefetch")
        let queued = try scheduler.submit(source: .internalSSD, priority: .prefetch, generation: generation, byteCount: bytes) { _ in
            log.add("queued ran")
        }
        let before = internalDrive(scheduler)
        try Check.require(
            before.queuedPrefetch == 1 && before.outstanding == 2 && before.outstandingBytes == UInt64(2 * bytes),
            "Scheduler fixture did not queue the second prediction: \(before)")
        try Check.require(!scheduler.cancelQueued(running), "A running prediction was cancelled")
        try Check.require(scheduler.cancelQueued(queued) && !scheduler.cancelQueued(queued), "Queued-only cancellation differs")
        guard case .cancelled = queued.join() else { throw EngineError("Cancelled queued prediction did not complete as cancelled") }
        let after = internalDrive(scheduler)
        try Check.require(
            after.queuedPrefetch == 0 && after.outstanding == 1 && after.outstandingBytes == UInt64(bytes)
                && after.cancelled == before.cancelled + 1 && after.runningPrefetch == 1 && scheduler.hygieneCounts()?.cancelledQueued == 1,
            "Queued cancellation released other than its own admission: \(after)")
        // Demand adopting the running read clears its token (no queue promotion for a running job).
        try Check.require(!scheduler.promote(running), "A running job was queue-promoted")
        gate.signal()
        guard case .succeeded = running.join() else { throw EngineError("Tokened prediction failed") }
        try Check.require(
            log.values == ["deferrable true", "deferrable false"], "Prediction token differs across promotion: \(log.values)")

        // Hold: the prediction waits, demand does not; one deferral per job; release starts it.
        guard let hold = scheduler.holdPrefetch() else { throw EngineError("Read hygiene returned no hold") }
        let held = try scheduler.submit(source: .internalSSD, priority: .prefetch, generation: generation, byteCount: bytes) { _ in
            log.add("held ran")
        }
        let demand = try scheduler.submit(source: .internalSSD, priority: .demand, generation: generation, byteCount: bytes) { _ in
            log.add("demand ran")
        }
        guard case .succeeded = demand.join() else { throw EngineError("Demand waited for or failed under a prefetch hold") }
        try Check.require(
            held.join(timeout: 0.05) == nil && internalDrive(scheduler).queuedPrefetch == 1 && !log.values.contains("held ran")
                && scheduler.hygieneCounts()?.holdDeferrals == 1 && scheduler.hygieneCounts()?.liveHolds == 1,
            "Prefetch hold did not defer exactly one prediction start: \(String(describing: scheduler.hygieneCounts()))")
        hold.release()
        hold.release()
        guard case .succeeded = held.join() else { throw EngineError("Held prediction did not run after release") }
        var second = scheduler.holdPrefetch()
        try Check.require(second != nil && scheduler.hygieneCounts()?.liveHolds == 1, "Second hold differs")
        second = nil
        try Check.require(
            scheduler.hygieneCounts()?.liveHolds == 0 && scheduler.hygieneCounts()?.holdDeferrals == 1,
            "A dropped hold was not released")
    }

    /// The flag defaults off; off, the execution exposes no hygiene path; a trace configuration
    /// that sets it loads, an older one does not, and it is refused without pre-routing.
    package static func offStateAndConfiguration() throws {
        let ledger = try MemoryLedger(ceiling: 200_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let reader: SourceExpertLoader.Reader = { key, _, bytes in bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            do {
                _ = try ExpertExecution(
                    device: device, ledger: ledger, configuration: configuration, readHygiene: true, packedAvailable: { false },
                    reader: reader)
                throw EngineError("Read hygiene admitted without pre-routing")
            } catch let error as EngineError {
                guard String(describing: error).contains("requires decode pre-routing") else { throw error }
            }
            let off = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }, reader: reader)
            try Check.require(
                off.preroutingCancelUnrouted() == nil && off.holdPredictedReads() == nil && off.readHygieneCounts == nil,
                "Pre-routing without read hygiene exposed a hygiene path")
            let on = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, readHygiene: true,
                packedAvailable: { false }, reader: reader)
            try Check.require(
                on.preroutingCancelUnrouted() != nil && on.readHygieneCounts?.cancelledQueued == 0, "Read hygiene path missing when on")
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Read-hygiene off-state fixture leaked")

        try Check.require(!RuntimeConfiguration().decodePreroutingReadHygiene, "Read hygiene defaults on")

        let base = CheckFixtures.file("config/trace-read-hygiene-8192.json")
        let enabled = try RuntimeConfiguration.load(base)
        try Check.require(
            enabled.decodePreroutingReadHygiene && enabled.decodeNextLayerPrerouting && enabled.decodePreroutingSlotOverride,
            "The read hygiene trace fixture does not enable read hygiene")
        let previous = try RuntimeConfiguration.load(CheckFixtures.file("config/trace-before-read-hygiene-8192.json"))
        try Check.require(!previous.decodePreroutingReadHygiene, "The fixture before read hygiene enabled it")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-read-hygiene-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: base)) as! [String: Any]
        func load(_ mutate: (inout [String: Any]) -> Void) throws -> RuntimeConfiguration {
            var object = original
            var execution = original["execution"] as! [String: Any]
            mutate(&execution)
            object["execution"] = execution
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return try RuntimeConfiguration.load(url)
        }
        func refused(_ mutate: (inout [String: Any]) -> Void, _ label: String) throws {
            do { _ = try load(mutate) } catch is EngineError { return }
            throw EngineError("Configuration accepted " + label)
        }
        try refused(
            {
                $0["decode_next_layer_prerouting"] = false
                $0["decode_prerouting_slot_override"] = false
            }, "read hygiene without pre-routing")
        try refused({ $0["decode_prerouting_read_hygiene"] = 1 }, "a non-boolean read-hygiene flag")
        try refused({ $0["decode_prerouting_read_hygiene"] = "true" }, "a string read-hygiene flag")
        try Check.require(
            !(try load({ $0["decode_prerouting_read_hygiene"] = false }).decodePreroutingReadHygiene), "Explicit false was not honoured")
        try Check.require(
            !(try load({ $0.removeValue(forKey: "decode_prerouting_read_hygiene") }).decodePreroutingReadHygiene), "Absent flag was not off"
        )
    }
}
