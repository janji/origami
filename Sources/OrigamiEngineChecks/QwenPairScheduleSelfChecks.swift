import Foundation
import Metal
import OrigamiEngine

/// The fixture's reads cannot finish before independent GPU work completes. This makes joining the reads before
/// submitting shared work a bounded failure, rather than relying on timing measurements or an arbitrary sleep.
private final class PairReadGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var released = false
    func open() {
        condition.lock()
        released = true
        condition.broadcast()
        condition.unlock()
    }
    func wait() throws {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: 5)
        while !released {
            guard condition.wait(until: deadline) || released else {
                throw EngineError("Pair read was joined before independent GPU work could complete")
            }
        }
    }
}

enum QwenPairScheduleSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks

    static func readOverlap(
        schedule: WholeExecutionSchedule, workspace: WholeExecutionWorkspace,
        ledger: MemoryLedger, device: MTLDevice
    ) throws {
        // Two disjoint top-ten rows exercise the maximum pair union, beyond release's sixteen-record window.
        // Early demand reserves twenty outstanding reads; both rows must stay pinned through their consumer.
        let ids = Array(0..<20)
        let historyRow = Array(0..<10)
        for resident in [false, true] {
            for cancel in [false, true] {
                try autoreleasepool {
                    let gate = PairReadGate()
                    // Reuse admission keeps twenty retained records beyond its twenty-four transient bypass slots.
                    let config = ExpertExecutionConfiguration(
                        mode: .slabPackedPreferred, cacheSlots: 48, slotsPerSlab: 8,
                        demandWindow: 16, readConcurrency: 2, reuseAdmission: true)
                    let executor = try ExpertExecution(
                        device: device, ledger: ledger, configuration: config,
                        earlyDemand: true, packedAvailable: { false }
                    ) { key, _, destination in
                        try gate.wait()
                        destination.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert + 1))
                    }
                    // Exceptional fixture cleanup also releases readers if shared command admission failed.
                    defer {
                        gate.open()
                        try? schedule.joinAll()
                        try? executor.finishSequence()
                    }
                    try executor.beginSequence()
                    try executor.beginDecodePhase()
                    if resident {
                        gate.open()
                        let warm = try executor.makeWindow(layer: 0, experts: ids, historyRow: historyRow)
                        try warm.consumeUnion { _ in () }
                    }
                    let window = try executor.makeWindow(layer: 0, experts: ids, historyRow: historyRow)
                    defer {
                        gate.open()
                        window.close()
                    }
                    try Check.require(
                        window.pendingCount == ids.count && window.residentAtRequestCount == (resident ? ids.count : 0),
                        "Pair fixture did not admit the complete two-row union")
                    let before = schedule.snapshot().decode
                    var sharedEncodes = 0
                    var refused = false
                    func encodeShared(_ command: MTLCommandBuffer) throws {
                        sharedEncodes += 1
                        guard let blit = command.makeBlitCommandEncoder() else { throw EngineError("Pair fixture shared blit unavailable") }
                        blit.fill(buffer: workspace[.result], range: 0..<4, value: 0x6d)
                        blit.endEncoding()
                        command.addCompletedHandler { _ in gate.open() }
                    }
                    let shared = try schedule.pairSharedDuringReads(
                        missing: window.residentAtRequestCount < ids.count,
                        shouldContinue: { true }, encode: encodeShared)
                    try Check.require(
                        (shared == nil) == resident && sharedEncodes == (resident ? 0 : 1),
                        "Pair shared work did not coalesce for a resident union")
                    do {
                        try window.consumeUnion { records in
                            try Check.require(
                                records.count == ids.count && executor.snapshot().cache.pinnedSlots == ids.count,
                                "Pair union lost a row's pins before GPU work")
                            let command = try schedule.command("pair-fixture-union")
                            if shared == nil { try encodeShared(command) }
                            guard let blit = command.makeBlitCommandEncoder() else {
                                throw EngineError("Pair fixture union blit unavailable")
                            }
                            blit.copy(from: workspace[.result], sourceOffset: 0, to: workspace[.residual], destinationOffset: 0, size: 4)
                            for (index, record) in records.enumerated() {
                                blit.copy(
                                    from: record.buffer, sourceOffset: record.offset, to: workspace[.result],
                                    destinationOffset: (index + 1) * 4, size: 4)
                            }
                            blit.endEncoding()
                            try schedule.joined(command, shouldContinue: { true })
                            try Check.require(
                                executor.snapshot().cache.pinnedSlots == ids.count,
                                "Pair pins retired before the complete consumer returned")
                            if cancel { throw IOOperationCancelled() }
                        }
                    } catch is IOOperationCancelled { refused = true }
                    if let shared { try schedule.join(shared) }
                    let after = schedule.snapshot().decode
                    let cache = executor.snapshot().cache
                    let commands: UInt64 = resident ? 1 : 2
                    try Check.require(
                        refused == cancel && sharedEncodes == 1 && after.submitted - before.submitted == commands
                            && after.joined - before.joined == commands,
                        "Pair shared/union command accounting differs")
                    try Check.require(
                        cache.pinnedSlots == 0 && cache.inFlightReads == 0 && cache.reservedSlots == 0,
                        "Pair union retained a producer or consumer after completion")
                    let result = workspace[.result].contents().assumingMemoryBound(to: UInt32.self)
                    try Check.require(
                        result[0] == 0x6d6d_6d6d && workspace[.residual].contents().load(as: UInt32.self) == result[0],
                        "Pair dependent command lost shared output")
                    for id in ids {
                        try Check.require(
                            result[id + 1] == UInt32(id + 1) * 0x0101_0101,
                            "Pair union changed canonical expert bytes")
                    }
                }
            }
        }
    }
}
