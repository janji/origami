import Foundation
import Metal
import ObjectiveC

/// The GPU residency table, so a queued decode command starts a layer's cached experts without the
/// host. `gpuKeys[key]` packs the record's slot + 1 (low 32 bits, 0 when absent) and the epoch it was verified in
/// (high 32 bits); `gpuSlots[slot]` is the slot's GPU address. A command trusts only records verified before it was
/// committed (the commit makes their bytes visible), and the records of a layer whose command is queued are not
/// evicted until it finishes. Bypass slots are never published. An observer of `ExpertCache`'s slot core: every
/// method runs under the core lock.
package final class ExpertGPUResidency: ExpertSlotObserver {
    private unowned(unsafe) let eviction: ExpertCacheEviction
    private(set) var gpuKeys: MTLBuffer?, gpuSlots: MTLBuffer?, gpuSlotCapacity = 0
    private var gpuEpoch: UInt32 = 1, gpuHeldLayers: UInt64 = 0
    /// Per slot: its record is in `gpuKeys`.
    private var gpuPublished: [Bool] = []

    init(eviction: ExpertCacheEviction) { self.eviction = eviction }

    // MARK: ExpertSlotObserver

    package func isHeld(_ slot: Int, key: Int?, sizeClass: Int) -> Bool {
        guard gpuHeldLayers != 0, gpuPublished[slot], let key else { return false }
        return gpuHeldLayers & (1 << UInt64(key / 512)) != 0
    }
    package func didVerify(_ slot: Int, sizeClass: Int, in slots: ExpertSlotClass) { publishGPU(slot, slots) }
    package func willForget(_ key: Int, slot: Int?, sizeClass: Int) {
        if let gpuKeys { gpuKeys.contents().storeBytes(of: UInt64(0), toByteOffset: key * 8, as: UInt64.self) }
        if let slot { gpuPublished[slot] = false }
    }
    package func didInsertSlab(_ id: Int, sizeClass: Int, in slots: ExpertSlotClass) {
        for index in slots.slabs[id]!.slots {
            while gpuPublished.count <= index { gpuPublished.append(false) }
            gpuPublished[index] = false
        }
        publishSlab(id, slots)
    }

    // MARK: The table

    private func publishGPU(_ index: Int, _ slots: ExpertSlotClass) {
        guard let gpuKeys, index < gpuSlotCapacity, let slot = slots.slots[index], let key = slot.key, !eviction.bypass[index] else {
            return
        }
        gpuKeys.contents().storeBytes(of: UInt64(index + 1) | UInt64(gpuEpoch) << 32, toByteOffset: key * 8, as: UInt64.self)
        gpuPublished[index] = true
    }
    private func publishSlab(_ id: Int, _ slots: ExpertSlotClass) {
        guard let gpuSlots, let slab = slots.slabs[id] else { return }
        for index in slab.slots where index < gpuSlotCapacity {
            gpuSlots.contents().storeBytes(
                of: slab.buffer.gpuAddress + UInt64(slots.slots[index]!.offset), toByteOffset: index * 8, as: UInt64.self)
        }
    }
    /// The residency table (allocated on first use): per key, slot + 1 and epoch; per slot, its GPU address.
    func table(device: MTLDevice, ledger: MemoryLedger, slots: ExpertSlotClass) throws -> (keys: MTLBuffer, slots: MTLBuffer) {
        if let gpuKeys, let gpuSlots { return (gpuKeys, gpuSlots) }
        let capacity =
            Int(slots.maximumByteLimit / UInt64(PackStore.expertBytes)) + 2 * slots.slotsPerSlab + ExpertReusePolicy.transientSlots
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
        gpuSlotCapacity = capacity
        for id in slots.slabs.keys { publishSlab(id, slots) }
        for index in slots.slots.indices where [.verified, .evictable, .pinned].contains(slots.slots[index]?.phase) {
            publishGPU(index, slots)
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
