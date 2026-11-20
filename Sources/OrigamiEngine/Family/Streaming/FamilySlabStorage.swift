import CExactIO
import Foundation
import Metal
import ObjectiveC

/// Wiring and ledger ownership have different lifetimes: a cache may unwire a retired slab, but a Metal command
/// can still retain its allocation. Its reservation follows that final Metal owner, never the cache array alone.
package final class FamilySlabWiring: @unchecked Sendable {
    package init() {}
    /// Logical live allocations, not the host's resident-page/physical-footprint measurement.
    package struct Snapshot: Sendable {
        package var ownedBytes = 0, wiredBytes = 0, attempts = 0, failures = 0, failedBytes = 0
        package var unwiredOwnedBytes: Int { ownedBytes - wiredBytes }
        package var unwireAttempts = 0, unwireFailures = 0
        /// Coarse cumulative page preparation/wire time, including failed attempts. Sampled only on slab growth.
        package var wireSeconds = 0.0
    }
    private let lock = NSLock()
    private var value = Snapshot()
    package var snapshot: Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
    func allocated(bytes: Int, wired: Bool, wireSeconds: Double) {
        lock.lock()
        defer { lock.unlock() }
        value.attempts += 1
        value.ownedBytes += bytes
        value.wireSeconds += wireSeconds
        if wired {
            value.wiredBytes += bytes
        } else {
            value.failures += 1
            value.failedBytes += bytes
        }
    }
    func unwired(bytes: Int, success: Bool) {
        lock.lock()
        defer { lock.unlock() }
        value.unwireAttempts += 1
        if success { value.wiredBytes -= bytes } else { value.unwireFailures += 1 }
    }
    func destroyed(bytes: Int, stillWired: Int) {
        lock.lock()
        defer { lock.unlock() }
        value.wiredBytes -= stillWired
        value.ownedBytes -= bytes
    }
}

private enum FamilySlabAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }
private final class FamilySlabOwner {
    let ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?, wiring: FamilySlabWiring
    let bytes: Int
    var wiredBytes: Int
    init(ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?, wiring: FamilySlabWiring, bytes: Int, wiredBytes: Int) {
        self.ledger = ledger
        self.reservation = reservation
        self.wiring = wiring
        self.bytes = bytes
        self.wiredBytes = wiredBytes
    }
    deinit {
        wiring.destroyed(bytes: bytes, stillWired: wiredBytes)
        if let reservation { try? ledger?.release(reservation) }
    }
}

package final class FamilySlabStorage {
    package let buffer: MTLBuffer
    /// Cache accounting must not borrow a Metal view and extend its autorelease lifetime.
    package let bytes: Int
    private let owner: FamilySlabOwner
    // Injectable outcomes allow checks to exercise failed wire/unwire calls without depending on machine limits.
    nonisolated(unsafe) package static var wire: (UnsafeMutableRawPointer, Int) -> Bool = { pointer, length in
        var pages: UInt64 = 0
        return origami_wire_pages(pointer, length, Int(getpagesize()), &pages) == 0
    }
    nonisolated(unsafe) package static var unwire: (UnsafeMutableRawPointer, Int) -> Bool = {
        origami_unwire_pages($0, $1) == 0
    }
    package init(buffer: MTLBuffer, ledger: MemoryLedger?, reservation: MemoryLedger.Reservation?, wiring: FamilySlabWiring) {
        self.buffer = buffer
        bytes = buffer.length
        let started = Date()
        let wired = Self.wire(buffer.contents(), buffer.length)
        wiring.allocated(bytes: buffer.length, wired: wired, wireSeconds: max(0, Date().timeIntervalSince(started)))
        owner = FamilySlabOwner(
            ledger: ledger, reservation: reservation, wiring: wiring, bytes: buffer.length, wiredBytes: wired ? buffer.length : 0)
        objc_setAssociatedObject(buffer, &FamilySlabAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
    deinit {
        // Never call munlock for a failed wire: munlock can succeed on unwired memory and hide another slab's bytes.
        if owner.wiredBytes > 0 {
            let success = Self.unwire(buffer.contents(), owner.wiredBytes)
            owner.wiring.unwired(bytes: owner.wiredBytes, success: success)
            if success { owner.wiredBytes = 0 }
        }
    }
}
