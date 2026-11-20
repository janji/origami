import Darwin
import Foundation
import Metal
import ObjectiveC

/// The ledger charge of decoded constants, associated with each output buffer so it retires with the last one.
private final class WholeConstantAllocation {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "whole_schedule_constant_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum WholeConstantAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Build the complete immutable reader set before publishing any of it. The
/// same transaction is exercised with lightweight readers in standalone checks.
package enum WholeTokenPlanConstruction {
    static let layerCount = 48
    package static let metadataBytes: UInt64 = 65_536

    package static func prepare<Owner>(
        ledger: MemoryLedger, boundary: () throws -> Void,
        acquireLayer: (Int) throws -> Owner, validate: () throws -> Void
    )
        throws -> (owners: [Owner], reservation: MemoryLedger.Reservation)
    {
        let reservation = try ledger.reserve(bytes: metadataBytes, category: "whole_token_plan_metadata")
        var owners: [Owner] = []
        owners.reserveCapacity(layerCount)
        do {
            for layer in 0..<layerCount {
                try boundary()
                owners.append(try acquireLayer(layer))
            }
            try validate()
            // The last source check may itself take time. Cancellation and live
            // headroom must still pass immediately before the plan escapes.
            try boundary()
            return (owners, reservation)
        } catch {
            owners.removeAll()
            try? ledger.release(reservation)
            throw error
        }
    }
}

/// Admission precedes creation of either layer's dynamic reader metadata.
/// Deallocation may occur after command retirement, so the counter is locked.
package final class WholeLayerReaderBudget {
    private let lock = NSLock()
    private var active = 0
    private let maximum: Int
    package init(maximum: Int = 2) {
        precondition((2...48).contains(maximum))
        self.maximum = maximum
    }
    package func acquire(ledger: MemoryLedger) throws -> WholeLayerReader {
        lock.lock()
        guard active < maximum else {
            lock.unlock()
            throw EngineError("Whole-schedule layer reader limit exceeded")
        }
        active += 1
        lock.unlock()
        do { return try WholeLayerReader(budget: self, ledger: ledger) } catch {
            retire()
            throw error
        }
    }
    fileprivate func retire() {
        lock.lock()
        defer { lock.unlock() }
        precondition(active > 0)
        active -= 1
    }
    package var activeReaders: Int {
        lock.lock()
        defer { lock.unlock() }
        return active
    }
}
/// One admitted layer reader: 128 KiB of metadata charge plus a slot in its budget, both returned on deinit.
package final class WholeLayerReader {
    private let budget: WholeLayerReaderBudget, ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    fileprivate init(budget: WholeLayerReaderBudget, ledger: MemoryLedger) throws {
        self.budget = budget
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: 131072, category: "whole_schedule_weight_reader_metadata")
    }
    deinit {
        try? ledger.release(reservation)
        budget.retire()
    }
}

/// Read-only decoded small constants. Each buffer retains the complete charge
/// independently of layer changes or the owning dictionary. This does not use
/// ReferenceWeights' per-layer host Array/transfer reservation.
package final class WholeLayerConstants {
    package let buffers: [String: MTLBuffer]
    package let chargedBytes: UInt64

    static func bytes(counts: [Int]) throws -> UInt64 {
        guard (1...8).contains(counts.count), counts.allSatisfy({ (1...40960).contains($0) }) else {
            throw EngineError("Invalid whole-layer constant domain")
        }
        let page = UInt64(getpagesize())
        // One page covers the fixed dictionary, names and allocation owner.
        return counts.reduce(page) { $0 + (UInt64($1 * 4) + page - 1) / page * page }
    }

    package init(leases: [ResidentTensorLease], expected: [(name: String, count: Int)], device: MTLDevice, ledger: MemoryLedger) throws {
        guard leases.count == expected.count, Set(expected.map(\.name)).count == expected.count else {
            throw EngineError("Whole-layer constant identities differ")
        }
        chargedBytes = try Self.bytes(counts: expected.map(\.count))
        let owner = try WholeConstantAllocation(ledger: ledger, bytes: chargedBytes)
        let page = Int(getpagesize())
        var result: [String: MTLBuffer] = [:]
        for (lease, spec) in zip(leases, expected) {
            let sourceBytes = spec.count * (lease.span.dtype == "BF16" ? 2 : 4)
            let bytes = spec.count * 4
            guard lease.name == spec.name, ["BF16", "F32"].contains(lease.span.dtype), lease.length == sourceBytes,
                lease.buffer.device.registryID == device.registryID
            else {
                throw EngineError("Whole-layer constant identity/geometry differs")
            }
            guard let output = device.makeBuffer(length: bytes, options: .storageModeShared),
                output.allocatedSize <= (bytes + page - 1) / page * page
            else {
                throw EngineError("Whole-layer constant allocation exceeds admission", code: "insufficient_memory")
            }
            objc_setAssociatedObject(output, &WholeConstantAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            if lease.span.dtype == "BF16" {
                let input = lease.buffer.contents().assumingMemoryBound(to: UInt16.self)
                let destination = output.contents().assumingMemoryBound(to: UInt32.self)
                for i in 0..<spec.count { destination[i] = UInt32(input[i]) << 16 }
            } else {
                output.contents().copyMemory(from: lease.buffer.contents(), byteCount: bytes)
            }
            result[spec.name] = output
        }
        buffers = result
    }
}

/// A layer's explicit immutable reader set. At most two transactions retain
/// these owners in the execution schedule. Raw-cache pins and decoded matrices
/// survive through the last joined reader, including failure/cancellation.
final class WholeLayerWeights {
    /// A QSA (sparse attention) layer's projections, indexer and norm weights.
    struct QSA {
        let queryGate, key, value, output: GDNDeviceBlock.Projection
        let index: MTLBuffer
        let queryNorm, keyNorm, indexQueryNorm, indexKeyNorm: MTLBuffer
    }
    /// A GDN (gated delta net) layer's projections, convolution, decay and norm weights.
    struct GDN {
        let qkv, z, output: GDNDeviceBlock.Projection
        let b, a, convolution, bias, logs, norm: MTLBuffer
    }
    let layer: Int
    let attentionHC, moeHC: DecodeFrontierWeights.HC
    let gdn: GDN?, qsa: QSA?
    let router, sharedOutputGate: MTLBuffer
    let sharedGate, sharedUp, sharedDown: GDNDeviceBlock.Projection
    private let raw: [ResidentTensorLease], dense: [ResidentFloatTensorLease]
    private let constants: WholeLayerConstants
    private let reader: WholeLayerReader

    init(
        layer: Int, attentionHC: DecodeFrontierWeights.HC, moeHC: DecodeFrontierWeights.HC, gdn: GDN?, qsa: QSA?,
        router: MTLBuffer, sharedOutputGate: MTLBuffer, sharedGate: GDNDeviceBlock.Projection,
        sharedUp: GDNDeviceBlock.Projection, sharedDown: GDNDeviceBlock.Projection,
        raw: [ResidentTensorLease], dense: [ResidentFloatTensorLease], constants: WholeLayerConstants,
        reader: WholeLayerReader
    ) {
        self.layer = layer
        self.attentionHC = attentionHC
        self.moeHC = moeHC
        self.gdn = gdn
        self.qsa = qsa
        self.router = router
        self.sharedOutputGate = sharedOutputGate
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
        self.raw = raw
        self.dense = dense
        self.constants = constants
        self.reader = reader
    }
}
