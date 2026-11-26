import Foundation
import Metal
import ObjectiveC

/// The GPU residency table, so a queued decode command starts a layer's cached experts without the
/// host. `gpuKeys[key]` packs the record's slot + 1 (low 32 bits, 0 when absent) and the epoch it was verified in
/// (high 32 bits); `gpuSlots[slot]` is the slot's GPU address. A command trusts only records verified before it was
/// committed (the commit makes their bytes visible), and the records of a layer whose command is queued are not
/// evicted until it finishes. Bypass slots are never published. With several record size classes, a class's slots take
/// their own range of the table (`base[c] + slot`). An observer of `ExpertCache`'s slot core: every method runs under the
/// core lock.
package final class ExpertGPUResidency: ExpertSlotObserver {
    private unowned(unsafe) let eviction: ExpertCacheEviction
    private(set) var gpuKeys: MTLBuffer?, gpuSlots: MTLBuffer?
    private var gpuEpoch: UInt32 = 1, gpuHeldLayers: UInt64 = 0
    /// Per class, its slots' table entries: the first one's index and how many there are.
    private var base: [Int] = [0], capacity: [Int] = [0]
    /// The core's classes (`configure`).
    private var classes: [ExpertSlotClass] = []
    /// Per class and slot: its record is in `gpuKeys`.
    private var gpuPublished: [[Bool]] = [[]]

    init(eviction: ExpertCacheEviction) { self.eviction = eviction }

    /// Before any slab: the core's classes, each given room for the slots its largest limit allows, two slabs of
    /// retirement holes and the transient slots.
    func configure(classes: [ExpertSlotClass], slotsPerSlab: Int) {
        self.classes = classes
        capacity = classes.map {
            Int($0.maximumByteLimit / UInt64($0.recordBytes)) + 2 * slotsPerSlab + ExpertReusePolicy.transientSlots
        }
        base = capacity.indices.map { c in capacity[..<c].reduce(0, +) }
        gpuPublished = Array(repeating: [], count: classes.count)
    }

    // MARK: ExpertSlotObserver

    package func isHeld(_ slot: Int, key: Int?, sizeClass c: Int) -> Bool {
        guard gpuHeldLayers != 0, gpuPublished[c][slot], let key else { return false }
        return gpuHeldLayers & (1 << UInt64(key / 512)) != 0
    }
    package func didVerify(_ slot: Int, sizeClass c: Int, in slots: ExpertSlotClass) { publishGPU(slot, c, slots) }
    package func willForget(_ key: Int, slot: Int?, sizeClass c: Int) {
        if let gpuKeys { gpuKeys.contents().storeBytes(of: UInt64(0), toByteOffset: key * 8, as: UInt64.self) }
        if let slot { gpuPublished[c][slot] = false }
    }
    package func didInsertSlab(_ id: Int, sizeClass c: Int, in slots: ExpertSlotClass) {
        for index in slots.slabs[id]!.slots {
            while gpuPublished[c].count <= index { gpuPublished[c].append(false) }
            gpuPublished[c][index] = false
        }
        publishSlab(id, c, slots)
    }

    // MARK: The table

    private func publishGPU(_ index: Int, _ c: Int, _ slots: ExpertSlotClass) {
        guard let gpuKeys, index < capacity[c], let slot = slots.slots[index], let key = slot.key, !eviction.isBypass(index, sizeClass: c)
        else {
            return
        }
        gpuKeys.contents().storeBytes(of: UInt64(base[c] + index + 1) | UInt64(gpuEpoch) << 32, toByteOffset: key * 8, as: UInt64.self)
        gpuPublished[c][index] = true
    }
    private func publishSlab(_ id: Int, _ c: Int, _ slots: ExpertSlotClass) {
        guard let gpuSlots, let slab = slots.slabs[id] else { return }
        for index in slab.slots where index < capacity[c] {
            gpuSlots.contents().storeBytes(
                of: slab.buffer.gpuAddress + UInt64(slots.slots[index]!.offset), toByteOffset: (base[c] + index) * 8, as: UInt64.self)
        }
    }
    /// The residency table (allocated on first use): per key, slot + 1 and epoch; per slot, its GPU address.
    func table(device: MTLDevice, ledger: MemoryLedger) throws -> (keys: MTLBuffer, slots: MTLBuffer) {
        if let gpuKeys, let gpuSlots { return (gpuKeys, gpuSlots) }
        let capacity = capacity.reduce(0, +)
        let bytes = UInt64(48 * 512 * 8 + capacity * 8)
        let reservation = try ledger.reserve(bytes: bytes, category: "decode-residency-table")
        guard let keys = device.makeBuffer(length: 48 * 512 * 8, options: .storageModeShared),
            let addresses = device.makeBuffer(length: capacity * 8, options: .storageModeShared)
        else {
            try ledger.release(reservation)
            throw EngineError("Decode residency table unavailable")
        }
        objc_setAssociatedObject(
            keys, &AllocationAssociation.key,
            ExpertSlabReservation(ledger: ledger, reservation: reservation), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        memset(keys.contents(), 0, keys.length)
        memset(addresses.contents(), 0, addresses.length)
        gpuKeys = keys
        gpuSlots = addresses
        for (c, slots) in classes.enumerated() {
            for id in slots.slabs.keys { publishSlab(id, c, slots) }
            for index in slots.slots.indices where [.verified, .evictable, .pinned].contains(slots.slots[index]?.phase) {
                publishGPU(index, c, slots)
            }
        }
        return (keys, addresses)
    }
    /// Holds `layer`'s published records for a command about to be committed and returns its epoch: the command trusts
    /// only records published before this call.
    func hold(layer: Int) -> UInt32 {
        if gpuEpoch > UInt32.max - 1_000_000, gpuHeldLayers == 0, let gpuKeys {
            // Restart the epochs (after billions of commands, with none queued): every published record becomes epoch 0.
            let entries = gpuKeys.contents().assumingMemoryBound(to: UInt64.self)
            for i in 0..<(48 * 512) where entries[i] != 0 { entries[i] &= 0xFFFF_FFFF }
            gpuEpoch = 1
        }
        gpuHeldLayers |= 1 << UInt64(layer)
        gpuEpoch += 1
        return gpuEpoch
    }
    func release(layer: Int?) {
        if let layer {
            gpuHeldLayers &= ~(1 << UInt64(layer))
        } else {
            gpuHeldLayers = 0
        }
    }
}
