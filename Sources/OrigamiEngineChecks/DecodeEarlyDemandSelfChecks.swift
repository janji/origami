import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of DecodeEarlyDemand: the router event releases dependent GPU work only after
/// its callback, cancelled or failed batches issue nothing and leak no charge, and staged requests are
/// adopted with their own bytes or drained on a key mismatch.
package enum DecodeEarlyDemandSelfChecks {
    private struct SignalledBuffer: @unchecked Sendable { let buffer: MTLBuffer }
    private typealias Check = DecodeFrontierSelfChecks
    private final class Probe: @unchecked Sendable {
        let lock = NSLock()
        private var calls = 0
        func call() {
            lock.lock()
            calls += 1
            lock.unlock()
        }
        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return calls
        }
    }
    package static func eventOwnership() throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
                let gate = device.makeSharedEvent(), let bytes = device.makeBuffer(length: 16384)
            else { throw EngineError("Metal unavailable") }
            let owner = try DecodeEarlyDemand(device: device, ledger: ledger)
            try Check.require(
                ledger.snapshot().categories["decode_early_demand_metadata"] == DecodeEarlyDemand.metadataBytes,
                "Event owner was not prospectively charged")
            let signalled = SignalledBuffer(buffer: bytes)
            let probe = Probe()
            let command = queue.makeCommandBuffer()!
            let blit = command.makeBlitCommandEncoder()!
            blit.fill(buffer: bytes, range: 0..<bytes.length, value: 0x37)
            blit.endEncoding()
            let batch = try owner.stage(
                command: command,
                operation: { keepGoing in
                    defer { gate.signaledValue = 1 }
                    guard keepGoing(), signalled.buffer.contents().load(as: UInt32.self) == 0x3737_3737 else {
                        throw EngineError("Router signal preceded GPU writes or cancellation")
                    }
                    probe.call()
                    return []
                }, discard: { _ in })
            // GPU completion depends on the CPU listener: proves this is an
            // in-command event, not a command-completion callback in disguise.
            command.encodeWaitForEvent(gate, value: 1)
            let refused = queue.makeCommandBuffer()!
            do {
                _ = try owner.stage(command: refused, operation: { _ in [] }, discard: { _ in })
                throw EngineError("Second stage admitted")
            } catch let error as EngineError { guard String(describing: error).contains("one-batch") else { throw error } }
            command.commit()
            command.waitUntilCompleted()
            try Check.require(command.status == .completed && probe.count == 1, "Router event did not release dependent GPU work")
            _ = try batch.adopt(expected: [])
            batch.close()
            try Check.require(
                batch.observation.beganBeforeCompletion && batch.observation.callbackNanoseconds > 0,
                "Event observation missed proven pre-completion work")
            for cancelled in [true, false] {
                let c = queue.makeCommandBuffer()!
                let p = Probe()
                let b = try owner.stage(
                    command: c,
                    operation: { _ in
                        p.call()
                        return []
                    }, discard: { _ in })
                if !cancelled {
                    c.commit()
                    c.waitUntilCompleted()
                }
                b.close()
                b.close()
                try Check.require(!cancelled || p.count == 0, "Forced cancellation notification issued reads")
            }
            let failed = queue.makeCommandBuffer()!
            let failure = try owner.stage(
                command: failed, operation: { _ in throw EngineError("Injected listener failure") }, discard: { _ in })
            failed.commit()
            failed.waitUntilCompleted()
            do {
                _ = try failure.adopt(expected: [])
                throw EngineError("Listener error was ignored")
            } catch let error as EngineError { guard String(describing: error) == "Injected listener failure" else { throw error } }
            failure.close()
        }
        try Check.require(ledger.snapshot().reserved == 0, "Router event or callback retained its metadata")
    }
    package static func publicationOwnership() throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
                let gate = device.makeSharedEvent(), let bytes = device.makeBuffer(length: 16384)
            else { throw EngineError("Metal unavailable") }
            let owner = try DecodeEarlyDemand(device: device, ledger: ledger)
            let command = queue.makeCommandBuffer()!
            let before = command.makeBlitCommandEncoder()!
            before.fill(buffer: bytes, range: 0..<bytes.length, value: 0x63)
            before.endEncoding()
            let frontier = try owner.checkpoint(command: command)
            command.encodeWaitForEvent(gate, value: 1)
            command.commit()
            // Always release the fixture GPU even if host assertions throw.
            defer {
                gate.signaledValue = 1
                command.waitUntilCompleted()
                frontier.close()
            }
            let early = try frontier.waitUntilReady(shouldContinue: { true })
            try Check.require(
                early && bytes.contents().load(as: UInt32.self) == 0x6363_6363,
                "Host publication did not follow completed writes and precede next GPU work")
            gate.signaledValue = 1
            command.waitUntilCompleted()
            frontier.close()
            for submitted in [false, true] {
                let c = queue.makeCommandBuffer()!
                let f = try owner.unsignalledCheckpointForSelfCheck(command: c)
                if submitted {
                    c.commit()
                    c.waitUntilCompleted()
                    let beforeCompletion = try f.waitUntilReady(shouldContinue: { true })
                    try Check.require(!beforeCompletion, "Unsignalled completion claimed early publication")
                } else {
                    do {
                        _ = try f.waitUntilReady(shouldContinue: { true })
                        throw EngineError("Unsubmitted publication accepted")
                    } catch let error as EngineError { guard String(describing: error).contains("not submitted") else { throw error } }
                }
                f.close()
                f.close()
            }
            let cancelled = queue.makeCommandBuffer()!
            let f = try owner.checkpoint(command: cancelled)
            cancelled.commit()
            do {
                _ = try f.waitUntilReady(shouldContinue: { false })
                throw EngineError("Cancelled publication accepted")
            } catch is IOOperationCancelled {}
            cancelled.waitUntilCompleted()
            f.close()
        }
        try Check.require(ledger.snapshot().reserved == 0, "Publication checkpoint retained notification metadata")
    }
    package static func stagedRequests() throws {
        let ledger = try MemoryLedger(ceiling: 160_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let execution = try ExpertExecution(
                device: device, ledger: ledger,
                configuration: .init(mode: .slabPackedPreferred, cacheSlots: 24, slotsPerSlab: 2, demandWindow: 10, readConcurrency: 2),
                earlyDemand: true, packedAvailable: { false }
            ) { key, _, bytes in
                bytes.initializeMemory(as: UInt8.self, repeating: UInt8(key.layer * 16 + key.expert))
            }
            try execution.beginSequence()
            try execution.beginDecodePhase()
            let owner = try DecodeEarlyDemand(device: device, ledger: ledger)
            let factory = try execution.earlyDemandFactory()
            let discard = execution.earlyDemandDiscard()
            let current = try execution.makeWindow(layer: 0, experts: Array(0..<10))
            let c = queue.makeCommandBuffer()!
            let batch = try owner.stage(command: c, operation: { keepGoing in try factory(Array(0..<10), 1, keepGoing) }, discard: discard)
            c.commit()
            c.waitUntilCompleted()
            var requests = try batch.adopt(expected: (0..<10).map { try ExpertKey(layer: 1, expert: $0) })
            try current.consumeTen(experts: Array(0..<10)) { buffers in
                try Check.require(buffers.count == 10, "Current producer displaced by future requests")
            }
            let next = try execution.makeWindow(layer: 1, experts: Array(0..<10), staged: requests)
            requests.removeAll()
            try next.consumeTen(experts: Array(0..<10)) { buffers in
                for (id, buffer) in buffers.enumerated() {
                    try Check.require(buffer.bytes[0] == UInt8(16 + id), "Staged record identity changed")
                }
            }
            // A joined mismatch must cancel/drain every unadopted producer.
            let mismatchCommand = queue.makeCommandBuffer()!
            let mismatch = try owner.stage(
                command: mismatchCommand, operation: { keepGoing in try factory(Array(0..<10), 2, keepGoing) }, discard: discard)
            mismatchCommand.commit()
            mismatchCommand.waitUntilCompleted()
            do {
                _ = try mismatch.adopt(expected: [])
                throw EngineError("Mismatched keys were adopted")
            } catch let error as EngineError {
                guard String(describing: error).contains("authoritative joined routing") else { throw error }
            }
            mismatch.close()
            try execution.finishSequence()
            let snapshot = execution.snapshot()
            try Check.require(
                snapshot.demandedExperts == 20 && snapshot.consumedExperts == 20 && snapshot.cache.pinnedSlots == 0
                    && snapshot.cache.inFlightReads == 0 && snapshot.io.drives.allSatisfy({ $0.outstanding == 0 }),
                "Staged producers/pins/counters did not retire")
        }
        try Check.require(ledger.snapshot().reserved == 0, "Staged request owner leaked")
    }
}
