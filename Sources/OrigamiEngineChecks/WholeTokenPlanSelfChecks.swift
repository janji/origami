import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of the decode token plan: reader admission, cleanup of the plan's construction at every
/// boundary and failure, ready/miss command windows, and the device convolution tail's bytes. They check
/// ownership and bit identity of primitives, not whole-model replies or speed.
package enum WholeTokenPlanSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks

    package static func readerAdmission() throws {
        let ledger = try MemoryLedger(ceiling: 12_000_000, transientMargin: 1_000_000)
        let budget = WholeLayerReaderBudget(maximum: 48)
        var readers: [WholeLayerReader] = []
        for _ in 0..<48 { readers.append(try budget.acquire(ledger: ledger)) }
        try Check.require(
            budget.activeReaders == 48 && ledger.snapshot().reserved == 48 * 131072,
            "Full token weight readers were not charged")
        var refused = false
        do { _ = try budget.acquire(ledger: ledger) } catch is EngineError { refused = true }
        try Check.require(refused && budget.activeReaders == 48, "49th token layer reader admitted")
        var retained: WholeLayerReader? = readers.removeLast()
        readers.removeAll()
        try Check.require(
            budget.activeReaders == 1 && ledger.snapshot().reserved == 131072,
            "Escaped pinned reader lost its owner")
        withExtendedLifetime(retained) {}
        retained = nil
        try Check.require(budget.activeReaders == 0 && ledger.snapshot().reserved == 0, "Token readers did not retire")
        let denied = try MemoryLedger(ceiling: 131072, transientMargin: 1)
        refused = false
        do { _ = try budget.acquire(ledger: denied) } catch is EngineError { refused = true }
        try Check.require(
            refused && budget.activeReaders == 0 && denied.snapshot().reserved == 0,
            "Allocation denial consumed a token reader")
    }

    /// Exercise the production construction transaction without model buffers:
    /// cancellation at every layer and after final source validation must drop
    /// all acquired readers and the plan's own metadata before propagating.
    package static func constructionBoundaries() throws {
        let ledger = try MemoryLedger(ceiling: 12_000_000, transientMargin: 1_000_000)
        let budget = WholeLayerReaderBudget(maximum: 48)
        for stop in 0...48 {
            var boundaries = 0
            var acquired = 0
            var validations = 0
            var cancelled = false
            do {
                _ = try WholeTokenPlanConstruction.prepare(
                    ledger: ledger,
                    boundary: {
                        try Check.require(
                            budget.activeReaders == acquired
                                && ledger.snapshot().reserved == WholeTokenPlanConstruction.metadataBytes + UInt64(acquired) * 131072,
                            "Construction boundary lost an acquired owner")
                        let position = boundaries
                        boundaries += 1
                        if position == stop { throw IOOperationCancelled() }
                    },
                    acquireLayer: { layer in
                        try Check.require(layer == acquired, "Construction skipped or repeated a layer")
                        acquired += 1
                        return try budget.acquire(ledger: ledger)
                    }, validate: { validations += 1 })
            } catch is IOOperationCancelled { cancelled = true }
            try Check.require(
                cancelled && boundaries == stop + 1 && acquired == stop && validations == (stop == 48 ? 1 : 0),
                "Construction cancellation crossed its boundary")
            try Check.require(
                budget.activeReaders == 0 && ledger.snapshot().reserved == 0,
                "Cancelled construction retained readers or plan metadata")
        }
        // Live headroom failures use the same throwing boundary. Allocation and
        // source failures must preserve their errors and use the same cleanup.
        enum Failure: Error, Equatable { case headroom, acquire, source }
        for failure in [Failure.headroom, .acquire, .source] {
            var acquired = 0
            var observed = false
            do {
                _ = try WholeTokenPlanConstruction.prepare(
                    ledger: ledger,
                    boundary: {
                        if case .headroom = failure, acquired == 24 { throw Failure.headroom }
                    },
                    acquireLayer: { layer in
                        if case .acquire = failure, layer == 24 { throw Failure.acquire }
                        acquired += 1
                        return try budget.acquire(ledger: ledger)
                    }, validate: { throw Failure.source })
            } catch let error as Failure {
                switch (failure, error) {
                case (.headroom, .headroom), (.acquire, .acquire), (.source, .source): observed = true
                default: break
                }
            }
            try Check.require(
                observed && acquired == (failure == .source ? 48 : 24) && budget.activeReaders == 0 && ledger.snapshot().reserved == 0,
                "Construction failure changed its error or retained owners")
        }
        let denied = try MemoryLedger(
            ceiling: WholeTokenPlanConstruction.metadataBytes + 4 * 131072 + 1,
            transientMargin: 1)
        var allocationDenied = false
        do {
            _ = try WholeTokenPlanConstruction.prepare(
                ledger: denied, boundary: {},
                acquireLayer: { _ in try budget.acquire(ledger: denied) }, validate: {})
        } catch let error as EngineError { allocationDenied = error.code == "insufficient_memory" }
        try Check.require(
            allocationDenied && budget.activeReaders == 0 && denied.snapshot().reserved == 0,
            "Partial allocation denial retained construction readers or metadata")
        var boundaries = 0
        var validations = 0
        var prepared: (owners: [WholeLayerReader], reservation: MemoryLedger.Reservation)? =
            try WholeTokenPlanConstruction.prepare(
                ledger: ledger, boundary: { boundaries += 1 },
                acquireLayer: { _ in try budget.acquire(ledger: ledger) }, validate: { validations += 1 })
        try Check.require(
            boundaries == 49 && validations == 1 && budget.activeReaders == 48
                && ledger.snapshot().reserved == WholeTokenPlanConstruction.metadataBytes + 48 * 131072,
            "Complete construction published before its final boundary")
        let reservation = prepared!.reservation
        var escaped = prepared!.owners.last
        prepared = nil
        try ledger.release(reservation)
        try Check.require(
            budget.activeReaders == 1 && ledger.snapshot().reserved == 131072,
            "Retiring a plan released an escaped reader's ownership")
        withExtendedLifetime(escaped) {}
        escaped = nil
        try Check.require(
            budget.activeReaders == 0 && ledger.snapshot().reserved == 0,
            "Complete construction readers did not retire")
    }

    /// Uses the actual demand owner, real asynchronous producers and Metal
    /// consumers. Verify coalescing at first/middle/last layer, mixed pins and
    /// cancellation after submission without releasing an unjoined consumer.
    static func commandMissWindows(
        schedule: WholeExecutionSchedule, workspace: WholeExecutionWorkspace,
        ledger: MemoryLedger, device: MTLDevice
    ) throws {
        for layer in [0, 24, 47] {
            for resident in [0, 3, 10] {
                for cancel in [false, true] {
                    try autoreleasepool {
                        let config = ExpertExecutionConfiguration(
                            mode: .slabPackedPreferred, cacheSlots: 12,
                            slotsPerSlab: 4, demandWindow: 10, readConcurrency: 2)
                        let executor = try ExpertExecution(
                            device: device, ledger: ledger, configuration: config, packedAvailable: { false }
                        ) { key, _, destination in
                            if key.expert == 9 { Thread.sleep(forTimeInterval: 0.002) }
                            destination.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert + 1))
                        }
                        try executor.beginSequence()
                        defer { try? executor.finishSequence() }
                        if resident > 0 {
                            let warm = try executor.makeWindow(layer: layer, experts: Array(0..<resident))
                            for expert in 0..<resident { try warm.consume(expert: expert) { _ in () } }
                            warm.close()
                        }
                        let window = try executor.makeWindow(layer: layer, experts: Array(0..<10))
                        defer { window.close() }
                        try Check.require(window.residentAtRequestCount == resident, "Fixture did not take declared miss window")
                        let before = schedule.snapshot().decode
                        func encode(_ command: MTLCommandBuffer, _ ids: [Int], _ records: [ExpertBuffer]) throws {
                            guard let encoder = command.makeBlitCommandEncoder() else { throw EngineError("Missing token fixture blit") }
                            for (id, record) in zip(ids, records) {
                                encoder.copy(
                                    from: record.buffer, sourceOffset: record.offset, to: workspace[.result],
                                    destinationOffset: id * 4, size: 4)
                            }
                            encoder.endEncoding()
                        }
                        var cancelled = false
                        do {
                            try window.consumeTenReadyFirst(
                                experts: Array(0..<10), joinReaders: { try? schedule.joinAll() },
                                ready: { ids, records in
                                    let command = try schedule.independentCommand(
                                        sharedPreencoded: true, readyCount: records.count,
                                        coalesce: true, missing: resident < 10)
                                    if let command {
                                        try encode(command, ids, records)
                                        try schedule.submit(command, shouldContinue: { true })
                                    }
                                },
                                finish: { ids, records in
                                    try Check.require(
                                        executor.snapshot().cache.pinnedSlots == 10, "Token split dropped a resident consumer pin")
                                    let command = try schedule.command("token-fixture-demand-continuation")
                                    try encode(command, ids, records)
                                    try schedule.submit(command, shouldContinue: { true })
                                    if cancel { throw IOOperationCancelled() }
                                    try schedule.joinAll()
                                })
                        } catch is IOOperationCancelled { cancelled = true }
                        let after = schedule.snapshot().decode
                        let commands: UInt64 = resident == 3 ? 2 : 1
                        try Check.require(
                            cancelled == cancel && after.submitted - before.submitted == commands
                                && after.joined - before.joined == commands, "Miss command/cancellation accounting differs")
                        try Check.require(
                            executor.snapshot().cache.pinnedSlots == 0 && executor.snapshot().cache.inFlightReads == 0,
                            "Miss consumer released before GPU/producer retirement")
                        let result = workspace[.result].contents().assumingMemoryBound(to: UInt32.self)
                        for expert in 0..<10 {
                            try Check.require(
                                result[expert] == UInt32(expert + 1) * 0x0101_0101,
                                "Ready/miss canonical consumer bytes differ")
                        }
                    }
                }
            }
        }
    }

    package static func convolutionTailBytes() throws {
        let ledger = try MemoryLedger(ceiling: 128_000_000, transientMargin: 1_000_000)
        let host = try ledger.reserve(bytes: 16_000_000, category: "token_tail_fixture_host")
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Token fixture needs Metal")
            }
            let arithmetic = try LayerUnaryArithmetic.pinned(ledger: ledger)
            let fused = try FusedExpertExecution(arithmetic: arithmetic, ledger: ledger)
            let tables = try DecodeFrontierScalars(device: device, ledger: ledger, arithmetic: arithmetic, fused: fused)
            let scalar = try WholeExecutionScalars(
                device: device, sigmoidTable: tables.bf16SigmoidTable,
                siluTable: fused.decodeRegionSiLUTable)
            let exceptional: [UInt32] = [
                0, 0x8000_0000, 1, 0x8000_0001, 0x3f80_8000, 0x3f81_8000,
                0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0x7fa1_0000, 0xffa1_8000, 0x7fc1_8000,
            ]
            var saved: GDNState?
            var savedTail: [UInt16] = []
            for step in 0..<4 {
                try autoreleasepool {
                    // Four windows cover every BF16 input bit pattern in the
                    // retained portion, including NaN payload/sign quieting.
                    let old = (0..<30720).map { UInt16(($0 + step * 20480) & 65535) }
                    let mixed: [Float] = (0..<10240).map { i in
                        i < exceptional.count
                            ? Float(bitPattern: exceptional[i])
                            : Float(bitPattern: UInt32(truncatingIfNeeded: i &* 104729 &+ step &* 65537))
                    }
                    let expected = old.suffix(20480).map { BF16.encode(BF16.decode($0)) } + mixed.map(BF16.encode)
                    let history = try Check.buffer(old, device)
                    let input = try Check.buffer(mixed, device)
                    let next = try GDNRecurrentStorage(device: device, outputCount: 6144, ledger: ledger, retainConvolutionTail: true)
                    let regular = try GDNRecurrentStorage(device: device, outputCount: 6144, ledger: ledger)
                    try Check.require(
                        next.allocationBytes - regular.allocationBytes <= 65536,
                        "Tail exceeds prospective per-state charge")
                    next.buffer.contents().initializeMemory(as: UInt8.self, repeating: 0x93, count: next.buffer.length)
                    let command = queue.makeCommandBuffer()!
                    try scalar.encodeConvolutionTail(command, history: history, mixed: input, next: next)
                    try Check.joined(command)
                    let actual = next.withConvolutionBytes { Array($0.bindMemory(to: UInt16.self)) }
                    try Check.require(actual == expected, "Device convolution tail differs from independent BF16 host contract")
                    try Check.require(
                        next.buffer.contents().load(as: UInt32.self) == 0x9393_9393
                            && next.buffer.contents().advanced(by: next.byteOffset).load(as: UInt32.self) == 0x9393_9393,
                        "Tail writer overwrote output/recurrent state")
                    try Check.require(
                        Array(UnsafeBufferPointer(start: history.contents().assumingMemoryBound(to: UInt16.self), count: 30720)) == old,
                        "Tail writer mutated prior state")
                    var state = GDNState()
                    state.publishRecurrent(next)
                    state.publishConvolution(next)
                    state.offset = step + 1
                    try state.validate()
                    try Check.require(
                        state.convolutionTail == expected && state.residentCapacityBytes == next.allocationBytes
                            && state.hostCapacityBytes == 0, "Tail materialization/unique owner accounting differs")
                    if step == 0 {
                        saved = state
                        savedTail = expected
                    } else {
                        try Check.require(saved!.convolutionTail == savedTail, "Later token changed retained frontier")
                    }
                    // Legacy/reference publication deliberately materializes a
                    // separate host tail while the recurrent owner stays valid.
                    state.convolutionTail = old
                    try Check.require(
                        state.convolutionStorage == nil && state.convolutionTail == old && state.recurrentStorage === next,
                        "Legacy tail fallback damaged recurrent ownership")
                }
            }
            saved = nil
        }
        try ledger.release(host)
        try Check.require(ledger.snapshot().reserved == 0, "Tail fixture leaked immutable backing")
    }
}
