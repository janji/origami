import Dispatch
import Foundation
import OrigamiEngine

private func physicalRequire(_ condition: Bool, _ message: String) throws {
    guard condition else { throw EngineError(message) }
}

private func physicalDenied(_ operation: () throws -> Void) throws {
    do { try operation() } catch is EngineError { return }
    throw EngineError("Physical admission accepted an allocation beyond its envelope")
}

/// Synthetic footprints make the lazy-residency failure reproducible without
/// filling RAM, relying on VM timing, or allocating real model buffers.
func physicalAdmissionCountsLazyOwnersAndAllowances() throws {
    let ledger = try MemoryLedger(
        ceiling: 1_000, transientMargin: 1,
        availableFloor: 0, budgetMode: .explicit)
    let dense = try ledger.reserve(bytes: 200, category: "dense")
    let allowance = try ledger.reserveAllowance(bytes: 300, category: "working")
    ledger.reconcilePhysicalOverhead(footprint: 250)
    try physicalRequire(
        ledger.physicalOverheadBytes == 50,
        "Unallocated workspace hid measured process overhead")
    let cache = try ledger.reservePhysical(bytes: 449, category: "family-experts", footprint: 250)
    try physicalDenied { _ = try ledger.reservePhysical(bytes: 2, category: "family-experts", footprint: 250) }
    try physicalRequire(
        ledger.reservedBytes == 949,
        "Denied cache admission changed live owner charges")
    let declined = try ledger.reservePhysicalIfAvailable(bytes: 2, category: "expert-slabs", footprint: 250)
    try physicalRequire(
        declined == nil && ledger.reservedBytes == 949,
        "Dynamic physical denial did not preserve the cache-reuse fallback")
    try physicalDenied {
        _ = try ledger.reservePhysicalIfAvailable(bytes: 51, category: "expert-slabs", footprint: 250)
    }
    try physicalRequire(
        ledger.reservedBytes == 949,
        "Logical admission failure mutated ownership or silently became physical fallback")
    try ledger.shrink(allowance, to: 100)
    ledger.reconcilePhysicalOverhead(footprint: 250)
    try physicalRequire(
        ledger.physicalOverheadBytes == 50,
        "Shrinking an allowance changed materialized ownership")
    try physicalDenied { _ = try ledger.reservePhysical(bytes: .max, category: "overflow", footprint: 250) }
    try ledger.release(cache)
    try ledger.release(allowance)
    try ledger.release(dense)
    try physicalRequire(ledger.reservedBytes == 0, "Owner retirement leaked a reservation")
}

private final class PhysicalAdmissionResults: @unchecked Sendable {
    private let lock = NSLock()
    private var reservations: [MemoryLedger.Reservation] = []
    func append(_ value: MemoryLedger.Reservation) {
        lock.lock()
        defer { lock.unlock() }
        reservations.append(value)
    }
    var values: [MemoryLedger.Reservation] {
        lock.lock()
        defer { lock.unlock() }
        return reservations
    }
}

/// Concurrent admissions see an unchanged, low physical footprint while the
/// accepted allocations are still lazy. Their full charges must serialize.
func physicalAdmissionConcurrentReservationsAndRecovery() throws {
    let ledger = try MemoryLedger(
        ceiling: 1_001, transientMargin: 1,
        availableFloor: 0, budgetMode: .explicit)
    ledger.reconcilePhysicalOverhead(footprint: 101)
    let results = PhysicalAdmissionResults()
    DispatchQueue.concurrentPerform(iterations: 12) { _ in
        if let reservation = try? ledger.reservePhysical(bytes: 300, category: "expert-slabs", footprint: 101) {
            results.append(reservation)
        }
    }
    try physicalRequire(
        results.values.count == 3 && ledger.reservedBytes == 900,
        "Concurrent lazy allocations oversubscribed the physical envelope")
    ledger.reconcilePhysicalOverhead(footprint: 910)
    try physicalRequire(
        ledger.physicalOverheadBytes == 101,
        "A partially resident observation erased outstanding overhead")
    ledger.reconcilePhysicalOverhead(footprint: 910, residentOwnedBytes: 800)
    try physicalRequire(
        ledger.physicalOverheadBytes >= 101,
        "An incomplete resident-owner proof erased possible overhead")
    ledger.reconcilePhysicalOverhead(footprint: 910, residentOwnedBytes: 900)
    try physicalRequire(
        ledger.physicalOverheadBytes == 10,
        "Verified resident owners could not reclaim temporary overhead")
    ledger.reconcilePhysicalOverhead(footprint: 910, fullyResident: true)
    try physicalRequire(
        ledger.physicalOverheadBytes == 10,
        "A joined, fully resident boundary could not reclaim transient overhead")
    let recovered = try ledger.reservePhysical(bytes: 90, category: "expert-slabs", footprint: 910)
    try physicalRequire(
        ledger.maximumUsableBytes == 1_000 && ledger.usableBytes == 1_000,
        "Physical admission permanently changed the user's target")
    try ledger.release(recovered)
    for reservation in results.values { try ledger.release(reservation) }
    ledger.reconcilePhysicalOverhead(footprint: 0, fullyResident: true)
    try physicalRequire(
        ledger.physicalOverheadBytes == 0 && ledger.reservedBytes == 0,
        "Retired transient overhead remained permanently reserved")
}
