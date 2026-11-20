import Dispatch
import Foundation
import Metal
import OrigamiEngine

/// Standalone checks for the decode pre-routing slot override (no XCTest).
package enum DecodePreroutingSlotOverrideSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private final class ReadLog: @unchecked Sendable {
        let lock = NSLock()
        var keys: [ExpertKey] = []
        func add(_ key: ExpertKey) {
            lock.lock()
            keys.append(key)
            lock.unlock()
        }
        func count(_ key: ExpertKey) -> Int {
            lock.lock()
            defer { lock.unlock() }
            return keys.filter { $0 == key }.count
        }
    }

    /// A key the reuse-score rule refuses is admitted only by evicting a cold,
    /// unprotected, unpinned verified record outside its target layer and the
    /// layer before it; never a transient bypass slot, a pinned record, or a
    /// record of those two layers. Without the override nothing changes.
    package static func evictsOnlyColdRecords() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let cache = try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(40 * PackStore.expertBytes), reuseAdmission: true)
            try cache.setReuseDecode(true)
            func row(_ layer: Int) throws {
                var tickets: [ExpertTicket] = []
                for id in try cache.beginDecodeRow(layer: layer, experts: Array(0..<10)) {
                    let a = try cache.acquire(ExpertKey(layer: layer, expert: id))
                    if a.newlyReserved {
                        let bytes = try cache.beginRead(a.ticket)
                        bytes.bytes.baseAddress!.storeBytes(of: UInt32(layer * 512 + id), as: UInt32.self)
                        try Check.require(cache.completeRead(a.ticket, verified: true), "Override fixture producer failed")
                    }
                    _ = try cache.pin(a.ticket)
                    tickets.append(a.ticket)
                }
                for ticket in tickets { try cache.unpinAfterGPUCompletion(ticket) }
            }
            // Sixteen retained slots: layer 0's row, then layer 1's (current, protected) row.
            try row(0)
            try row(1)
            let before = cache.snapshot()
            let layer0 = try cache.draftResidentExperts(layer: 0)
            let layer1 = try cache.draftResidentExperts(layer: 1)
            try Check.require(!layer0.isEmpty && before.allocatedBytes == before.byteLimit, "Override fixture did not fill the cache")
            guard case .noSlot = try cache.acquireSpeculative(ExpertKey(layer: 2, expert: 20)) else {
                throw EngineError("Fixture key was not refused by the reuse-score rule")
            }
            // Target layer 1: every eligible record is of layer 1 or 0, so the override finds nothing.
            guard case .noSlot = try cache.acquireSpeculative(ExpertKey(layer: 1, expert: 30), overrideScore: true) else {
                throw EngineError("Override evicted a record of the target layer or the layer before it")
            }
            try Check.require(
                cache.snapshot().evictions == before.evictions && cache.snapshot().misses == before.misses,
                "Refused predictions mutated the cache")
            guard case .overridden(let admitted) = try cache.acquireSpeculative(ExpertKey(layer: 2, expert: 20), overrideScore: true) else {
                throw EngineError("Override did not admit a refused prediction over a cold record")
            }
            let after = cache.snapshot()
            let remaining0 = try cache.draftResidentExperts(layer: 0)
            let remaining1 = try cache.draftResidentExperts(layer: 1)
            try Check.require(
                after.evictions == before.evictions + 1 && after.reuseAdmissions == before.reuseAdmissions
                    && after.reuseBypasses == before.reuseBypasses && after.reuseBorrowedBypasses == before.reuseBorrowedBypasses
                    && remaining0.count == layer0.count - 1 && remaining1 == layer1,
                "Override took a slot other than one cold layer-0 record: \(after)")
            let bytes = try cache.beginRead(admitted.ticket)
            bytes.bytes.baseAddress!.storeBytes(of: UInt32(2 * 512 + 20), as: UInt32.self)
            try Check.require(cache.completeRead(admitted.ticket, verified: true), "Overridden producer failed")
            guard case .resident = try cache.acquireSpeculative(ExpertKey(layer: 2, expert: 20), overrideScore: true) else {
                throw EngineError("Overridden record is not resident")
            }
            // GPU-pinned cold records are never victims.
            var pinned: [ExpertTicket] = []
            for expert in try cache.draftResidentExperts(layer: 0) {
                let a = try cache.acquire(ExpertKey(layer: 0, expert: expert))
                _ = try cache.pin(a.ticket)
                pinned.append(a.ticket)
            }
            let held = cache.snapshot()
            guard case .noSlot = try cache.acquireSpeculative(ExpertKey(layer: 2, expert: 21), overrideScore: true) else {
                throw EngineError("Override evicted a GPU-pinned record")
            }
            try Check.require(cache.snapshot().evictions == held.evictions, "Refused override mutated the cache")
            for ticket in pinned { try cache.unpinAfterGPUCompletion(ticket) }
            try cache.setReuseDecode(false)
            _ = try cache.resizeQuiescent(to: 0)
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Override cache fixture leaked")
    }

    /// Through the loader and the pre-routing factory: the width bounds the
    /// overrides per target layer in rank order, every outcome is counted, and
    /// the target layer's demand adopts the overridden record with the same
    /// bytes and no second read.
    package static func widthAndAdoption() throws {
        let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let log = ReadLog()
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 40, slotsPerSlab: 8, demandWindow: 10,
                readConcurrency: 4, reuseAdmission: true)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                log.add(key)
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            do {
                _ = try execution.preroutingFactory(overrideWidth: 11)
                throw EngineError("Override width 11 admitted")
            } catch let error as EngineError { guard String(describing: error).contains("width in 1...10") else { throw error } }
            let plain = try execution.preroutingFactory()
            let factory = try execution.preroutingFactory(overrideWidth: 1)
            for layer in 0..<2 {
                let window = try execution.makeWindow(layer: layer, experts: Array(0..<10))
                try window.consumeTen(experts: Array(0..<10)) { buffers in try Check.require(buffers.count == 10, "Row fixture differs") }
            }
            let before = execution.snapshot().cache
            let off = try plain([20, 21], 2, { true })
            try Check.require(
                off.noSlot == 2 && off.issued == 0 && off.overrides.isEmpty && off.overrideNoVictim == 0
                    && off.overrideWidthExhausted == 0 && execution.snapshot().cache.evictions == before.evictions,
                "Pre-routing without the override changed: \(off.issued)/\(off.noSlot)")
            var issue = try factory([20, 21], 2, { true })
            try Check.require(
                issue.issued == 1 && issue.requests.count == 1 && issue.overrides == [20] && issue.noSlot == 1
                    && issue.overrideWidthExhausted == 1 && issue.overrideNoVictim == 0 && issue.resident == 0 && issue.refused == 0,
                "Override width or counters differ: \(issue.issued)/\(issue.overrides)/\(issue.noSlot)/\(issue.overrideWidthExhausted)/\(issue.overrideNoVictim)"
            )
            let key = try ExpertKey(layer: 2, expert: 20)
            issue.requests.removeAll()
            for _ in 0..<2_000 where execution.snapshot().cache.inFlightReads > 0 { Thread.sleep(forTimeInterval: 0.001) }
            let after = execution.snapshot().cache
            try Check.require(
                after.inFlightReads == 0 && after.pinnedSlots == 0 && after.evictions == before.evictions + 1
                    && after.reuseBypasses == before.reuseBypasses && log.count(key) == 1,
                "Overridden read did not complete as one evictable record: \(after)")
            let experts = Array(0..<9) + [20]
            let window = try execution.makeWindow(layer: 2, experts: experts)
            try window.consumeTen(experts: experts) { buffers in
                try Check.require(
                    buffers[9].bytes[0] == 20 && buffers[9].bytes[PackStore.expertBytes - 1] == 20, "Overridden record carried wrong bytes")
            }
            try Check.require(log.count(key) == 1, "Demand re-read an overridden record")
            try execution.finishSequence()
        }
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
            let configuration = ExpertExecutionConfiguration(
                mode: .slabPackedPreferred, cacheSlots: 16, slotsPerSlab: 2, demandWindow: 4, readConcurrency: 2)
            let execution = try ExpertExecution(
                device: device, ledger: ledger, configuration: configuration, prerouting: true, packedAvailable: { false }
            ) {
                key, _, bytes in
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            do {
                _ = try execution.preroutingFactory(overrideWidth: 1)
                throw EngineError("Override admitted without reuse admission")
            } catch let error as EngineError { guard String(describing: error).contains("reuse admission") else { throw error } }
            try execution.finishSequence()
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Override execution fixture leaked")
    }

    /// The flag defaults off; the configuration parses the width and refuses unsupported combinations.
    package static func countersAndConfiguration() throws {
        try Check.require(
            !RuntimeConfiguration().decodePreroutingSlotOverride && RuntimeConfiguration().decodePreroutingOverrideWidth == 10,
            "Slot-override defaults differ")

        let base = CheckFixtures.file("config/trace-slot-override-8192.json")
        let enabled = try RuntimeConfiguration.load(base)
        try Check.require(
            enabled.decodePreroutingSlotOverride && enabled.decodePreroutingOverrideWidth == 10 && enabled.decodeNextLayerPrerouting
                && enabled.experts.reuseAdmission, "The slot override trace fixture does not enable the slot override")
        let previous = try RuntimeConfiguration.load(CheckFixtures.file("config/trace-before-slot-override-8192.json"))
        try Check.require(!previous.decodePreroutingSlotOverride, "The fixture before the slot override enabled it")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-slot-override-" + UUID().uuidString + ".json")
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
        try refused({ $0["decode_next_layer_prerouting"] = false }, "the override without pre-routing")
        try refused({ $0["expert_reuse_admission"] = false }, "the override without reuse admission")
        try refused({ $0["decode_prerouting_slot_override"] = 1 }, "a non-boolean override flag")
        try refused({ $0["decode_prerouting_override_width"] = 0 }, "a zero width")
        try refused({ $0["decode_prerouting_override_width"] = 11 }, "a width above ten")
        try refused({ $0["decode_prerouting_override_width"] = "10" }, "a string width")
        try refused({ $0["decode_prerouting_override_width"] = 2.5 }, "a fractional width")
        try Check.require(
            try load({ $0["decode_prerouting_override_width"] = 4 }).decodePreroutingOverrideWidth == 4, "A width of four was refused")
        try Check.require(
            !(try load({ $0["decode_prerouting_slot_override"] = false }).decodePreroutingSlotOverride), "Explicit false was not honoured")
    }
}
