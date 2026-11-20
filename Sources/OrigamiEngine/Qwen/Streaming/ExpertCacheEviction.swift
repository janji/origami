import Foundation

/// Qwen's victim choice over the slot core (`ExpertCache`'s policy): CLOCK, in prefill after the nearest passed scan
/// record, or the reuse-score policy (`ExpertReusePolicy`) in decode with reuse admission, with its 24 transient bypass
/// slots, and speculative admission with the slot override. Keys are flat (`layer * 512 +
/// expert`). Every method runs under the core lock (the core calls the `ExpertSlotPolicy` requirements; `ExpertCache`
/// takes the lock before the others).
package final class ExpertCacheEviction: ExpertSlotPolicy {
    package typealias Observer = ExpertGPUResidency
    package typealias Core = ExpertSlotCore<ExpertCacheEviction>
    /// Per slot: the CLOCK reference bit, the bypass mark (reuse decode's transient and borrowed slots), and
    /// filled during prefill and not used by decode since (`prefillScan`).
    private(set) var referenced: [Bool] = [], bypass: [Bool] = [], scan: [Bool] = []
    private var clockHand = 0
    /// Prefill phase: records it fills are `scan`, and a miss evicts the nearest passed scan record first.
    private var prefillScan = false
    var reuse: ExpertReusePolicy?
    private(set) var reuseDecode = false
    private var reuseVictims: [(index: Int, score: Double)] = []
    private var reuseVictimsPrepared = false
    private(set) var reuseBypasses: UInt64 = 0, reuseAdmissions: UInt64 = 0
    private(set) var reuseBorrowedBypasses: UInt64 = 0
    /// Per layer, the lowest (score, slot) speculative victim (`reuseMinima`), recomputed only once dirty: every layer
    /// when the scores change (each row), one layer when a record of it changes eligibility.
    private var layerMin = [(index: Int, score: Double)?](repeating: nil, count: 48)
    private var layerDirty = [Bool](repeating: true, count: 48)

    init() {}

    private func gpuHeld(_ index: Int, _ slot: ExpertSlotClass.Slot, _ core: Core) -> Bool {
        core.observer.isHeld(index, key: slot.key, sizeClass: 0)
    }
    func markDirty(_ layer: Int) { layerDirty[layer] = true }
    func markAllDirty() { layerDirty = Array(repeating: true, count: 48) }

    // MARK: ExpertSlotPolicy

    package func didHit(_ slot: Int, sizeClass: Int) {
        referenced[slot] = true
        if !prefillScan { scan[slot] = false }
    }
    package func didReserve(_ slot: Int, sizeClass: Int) {
        referenced[slot] = true
        if !reuseDecode { bypass[slot] = false }
        scan[slot] = prefillScan
    }
    /// A speculative reservation never keeps a bypass mark.
    func didReserveSpeculative(_ slot: Int) {
        referenced[slot] = true
        bypass[slot] = false
        scan[slot] = prefillScan
    }
    package func didChange(_ slot: Int, key: Int, sizeClass: Int, to phase: ExpertSlotPhase) {
        if phase == .pinned { referenced[slot] = true }
        layerDirty[key / 512] = true
    }
    /// Reuse decode: a transient bypass slot goes straight back to free when its last pin retires.
    package func releasesOnLastUnpin(_ slot: Int, sizeClass: Int) -> Bool { reuseDecode && bypass[slot] }
    package func didForget(_ key: Int) { layerDirty[key / 512] = true }
    package func didInsertSlab(_ slots: [Int], sizeClass: Int) {
        for index in slots {
            while referenced.count <= index {
                referenced.append(false)
                bypass.append(false)
                scan.append(false)
            }
            referenced[index] = false
            bypass[index] = false
            scan[index] = false
        }
    }

    package func victim(for key: Int, sizeClass: Int, in core: Core) throws -> Int? {
        if reuseDecode, let reuse { return try reuseVictim(key, policy: reuse, core) }
        let slots = core.classes[0]
        if let index = slots.firstFree(from: 0) { return index }
        if let index = try core.allocateSlabLocked(sizeClass: 0) { return index }
        guard !slots.slots.isEmpty else { return nil }
        if prefillScan, let index = passedScanVictim(key, core) { return index }
        let all = slots.slots
        for _ in 0..<(all.count * 2) {
            let index = clockHand % all.count
            clockHand = (index + 1) % all.count
            guard let slot = all[index], [.verified, .evictable].contains(slot.phase), slot.pins == 0, !gpuHeld(index, slot, core)
            else { continue }
            if referenced[index] {
                referenced[index] = false
                continue
            }
            core.evictLocked(index, sizeClass: 0)
            return index
        }
        return nil
    }

    /// Prefill requests keys in ascending index order (layer by layer, each layer's sorted union), so a scan record
    /// below the requested key is not needed again before the next scope: the nearest one is the sweep's Belady victim.
    /// CLOCK would also take records still ahead in this sweep and the records decode kept (the scan flag is clear).
    private func passedScanVictim(_ key: Int, _ core: Core) -> Int? {
        // One read of each array: nothing in the loop changes them.
        let lookup = core.lookup
        let all = core.classes[0].slots
        for passed in stride(from: key - 1, through: 0, by: -1) {
            guard let index = lookup[passed], let slot = all[index], scan[index], slot.pins == 0, !gpuHeld(index, slot, core),
                [.verified, .evictable].contains(slot.phase)
            else { continue }
            core.evictLocked(index, sizeClass: 0)
            return index
        }
        return nil
    }
    func setPrefillScan(_ enabled: Bool) { prefillScan = enabled }

    /// Phase switches occur after the same quiescent ownership check as resize.
    func setReuseDecode(_ enabled: Bool, _ core: Core) throws {
        let slots = core.classes[0]
        guard let reuse else { return }
        guard slots.quiescent else {
            throw EngineError("Reuse phase requires joined readers and consumers")
        }
        if enabled && !reuseDecode {
            reuse.reset()
            // These existing physical slots become the charged bypass arena.
            for i in 0..<min(ExpertReusePolicy.transientSlots, slots.slots.count) {
                guard slots.slots[i] != nil else { continue }
                core.freeLocked(i, sizeClass: 0)
                bypass[i] = true
            }
        }
        for i in slots.slots.indices where slots.slots[i] != nil && bypass[i] {
            core.freeLocked(i, sizeClass: 0)
            bypass[i] = enabled && i < ExpertReusePolicy.transientSlots
        }
        layerDirty = Array(repeating: true, count: 48)
        reuseDecode = enabled
        reuseVictims.removeAll(keepingCapacity: true)
        reuseVictimsPrepared = false
    }

    func beginDecodeRow(layer: Int, experts: [Int]) throws -> [Int] {
        guard reuseDecode, let reuse else { return experts.sorted() }
        reuseVictims.removeAll(keepingCapacity: true)
        reuseVictimsPrepared = false
        layerDirty = Array(repeating: true, count: 48)
        return try reuse.beginRow(layer: layer, experts: experts)
    }

    func setStorageCosts(_ costs: [Double]) {
        guard reuse?.setStorageCosts(costs) == true else { return }
        layerDirty = Array(repeating: true, count: 48)
        reuseVictims.removeAll(keepingCapacity: true)
        reuseVictimsPrepared = false
    }

    private func reuseVictim(_ key: Int, policy: ExpertReusePolicy, _ core: Core) throws -> Int? {
        let began = DispatchTime.now().uptimeNanoseconds
        defer { policy.recordDecisionTime(DispatchTime.now().uptimeNanoseconds - began) }
        let slots = core.classes[0]
        let transient = ExpertReusePolicy.transientSlots
        if let i = slots.firstFree(from: transient) {
            bypass[i] = false
            reuseAdmissions += 1
            return i
        }
        while let i = try core.allocateSlabLocked(sizeClass: 0) {
            if i >= transient {
                bypass[i] = false
                reuseAdmissions += 1
                return i
            }
        }
        if !reuseVictimsPrepared { prepareReuseVictims(policy, core) }
        while let candidate = reuseVictims.first {
            guard let slot = slots.slots[candidate.index], let old = slot.key, slot.pins == 0, !gpuHeld(candidate.index, slot, core),
                [.verified, .evictable].contains(slot.phase), !policy.isProtected(old)
            else {
                reuseVictims.removeFirst()
                continue
            }
            if policy.score(key) <= candidate.score { break }
            reuseVictims.removeFirst()
            core.evictLocked(candidate.index, sizeClass: 0)
            reuseAdmissions += 1
            bypass[candidate.index] = false
            return candidate.index
        }
        // A bypass is still a complete exact demand read. Its slab remains
        // owned until producer and all handoff/GPU pins have retired.
        for i in 0..<min(transient, slots.slots.count) {
            guard let slot = slots.slots[i], slot.pins == 0,
                [.free, .verified, .evictable].contains(slot.phase)
            else { continue }
            if slot.key != nil { core.evictLocked(i, sizeClass: 0) }
            bypass[i] = true
            reuseBypasses += 1
            return i
        }
        // Slow losing primary reads can outlive several replica consumers.
        // Borrow a retained slot inside the SAME byte limit; never recycle a
        // live destination or fail exact demand just because 24 staging slots
        // are busy. The borrowed slot is freed after its last actual owner.
        for rescan in [false, true] {
            // Prediction reads may have taken every prepared victim since the
            // row's scan; rescan once before refusing exact demand.
            if rescan { prepareReuseVictims(policy, core) }
            while let candidate = reuseVictims.first {
                reuseVictims.removeFirst()
                guard let slot = slots.slots[candidate.index], let old = slot.key, slot.pins == 0, !gpuHeld(candidate.index, slot, core),
                    [.verified, .evictable].contains(slot.phase), !policy.isProtected(old)
                else { continue }
                core.evictLocked(candidate.index, sizeClass: 0)
                bypass[candidate.index] = true
                reuseBypasses += 1
                reuseBorrowedBypasses += 1
                return candidate.index
            }
        }
        return nil
    }

    /// Keep only ten minimum eligible scores in one linear scan. No
    /// whole-cache sort or repeated probability/exponential evaluation.
    private func prepareReuseVictims(_ policy: ExpertReusePolicy, _ core: Core) {
        let all = core.classes[0].slots
        for i in all.indices where i >= ExpertReusePolicy.transientSlots {
            // Held records are refused by both uses: ten of them (the current layer's) would hide every real victim.
            guard let slot = all[i], let old = slot.key, slot.pins == 0, !gpuHeld(i, slot, core),
                [.verified, .evictable].contains(slot.phase), !policy.isProtected(old)
            else { continue }
            let score = policy.score(old)
            if reuseVictims.count == 10, score >= reuseVictims.last!.score { continue }
            let insertion = reuseVictims.firstIndex { score < $0.score } ?? reuseVictims.count
            reuseVictims.insert((i, score), at: insertion)
            if reuseVictims.count > 10 { reuseVictims.removeLast() }
        }
        reuseVictimsPrepared = true
    }

    /// The speculative victim (`ExpertCache.acquireSpeculative`): only a free non-transient slot, a new slab, a CLOCK
    /// victim, or (reuse admission) the lowest-score record the key's own score beats, or with `overrideScore` the
    /// slot override's cold record (`overridden`).
    func speculativeVictim(_ key: Int, overrideScore: Bool, _ core: Core) throws -> (index: Int, overridden: Bool)? {
        let slots = core.classes[0]
        let transient = reuseDecode ? ExpertReusePolicy.transientSlots : 0
        if let i = slots.firstFree(from: transient) {
            return (i, false)
        }
        while let i = try core.allocateSlabLocked(sizeClass: 0) {
            if i >= transient {
                return (i, false)
            }
        }
        if reuseDecode, let reuse {
            let began = DispatchTime.now().uptimeNanoseconds
            defer { reuse.recordDecisionTime(DispatchTime.now().uptimeNanoseconds - began) }
            let (victim, lowest) = reuseMinima(key, reuse: reuse, core)
            let cold = overrideScore ? lowest : nil
            if let victim, slots.slots[victim.index]?.key != nil, reuse.score(key) > victim.score {
                core.evictLocked(victim.index, sizeClass: 0)
                reuseAdmissions += 1
                return (victim.index, false)
            }
            // Bounded by the caller's per-layer width; not a reuse admission.
            guard let cold, slots.slots[cold.index]?.key != nil else { return nil }
            core.evictLocked(cold.index, sizeClass: 0)
            return (cold.index, true)
        }
        guard !slots.slots.isEmpty else { return nil }
        let all = slots.slots
        for _ in 0..<(all.count * 2) {
            let index = clockHand % all.count
            clockHand = (index + 1) % all.count
            guard let slot = all[index], [.verified, .evictable].contains(slot.phase), slot.pins == 0, !gpuHeld(index, slot, core)
            else { continue }
            if referenced[index] {
                referenced[index] = false
                continue
            }
            core.evictLocked(index, sizeClass: 0)
            return (index, false)
        }
        return nil
    }

    /// The lowest-score eligible speculative victim and the lowest outside `key`'s layer and the layer before it (the slot
    /// override's cold record): what one scan of every slot gives, first lowest index on a tie, from per-layer minima
    /// over `lookup` (a record is found through its key) refreshed only for dirty layers. Same `score` products.
    private func reuseMinima(_ key: Int, reuse: ExpertReusePolicy, _ core: Core) -> (
        victim: (index: Int, score: Double)?, cold: (index: Int, score: Double)?
    ) {
        let lookup = core.lookup
        let all = core.classes[0].slots
        func lower(_ a: (index: Int, score: Double), _ b: (index: Int, score: Double)?) -> Bool {
            guard let b else { return true }
            return a.score < b.score || a.score == b.score && a.index < b.index
        }
        for layer in 0..<48 where layerDirty[layer] {
            var best: (index: Int, score: Double)?
            for k in (layer * 512)..<((layer + 1) * 512) {
                guard let i = lookup[k], i >= ExpertReusePolicy.transientSlots, let slot = all[i], let old = slot.key,
                    slot.pins == 0, !bypass[i], !gpuHeld(i, slot, core), [.verified, .evictable].contains(slot.phase),
                    !reuse.isProtected(old)
                else { continue }
                assert(old == k, "Lookup points at another key's slot")
                let candidate = (index: i, score: reuse.score(old))
                if lower(candidate, best) { best = candidate }
            }
            layerMin[layer] = best
            layerDirty[layer] = false
        }
        let previous = (key / 512 + 47) % 48
        var victim: (index: Int, score: Double)?
        var cold: (index: Int, score: Double)?
        for layer in 0..<48 {
            guard let entry = layerMin[layer] else { continue }
            if lower(entry, victim) { victim = entry }
            if layer != key / 512, layer != previous, lower(entry, cold) { cold = entry }
        }
        return (victim, cold)
    }

    /// `cache_victim_index`: the counter and per-layer answers beside a copy of the full scans they replaced (the
    /// speculative one as if `overrideScore`). Changes nothing but the minima's refresh; nil when all agree.
    func victimIndexMismatch(_ key: Int, _ core: Core) -> String? {
        let slots = core.classes[0]
        let transient = ExpertReusePolicy.transientSlots
        let all = slots.slots.firstIndex(where: { $0?.phase == .free })
        let regular = slots.slots.indices.first(where: { $0 >= transient && slots.slots[$0]?.phase == .free })
        guard slots.firstFree(from: 0) == all, slots.firstFree(from: transient) == regular else {
            return
                "free slot \(String(describing: slots.firstFree(from: 0)))/\(String(describing: slots.firstFree(from: transient))) vs \(String(describing: all))/\(String(describing: regular))"
        }
        guard reuseDecode, let reuse else { return nil }
        let previous = (key / 512 + 47) % 48
        var victim: (index: Int, score: Double)?
        var cold: (index: Int, score: Double)?
        for i in slots.slots.indices where i >= transient {
            guard let slot = slots.slots[i], let old = slot.key, slot.pins == 0, !bypass[i], !gpuHeld(i, slot, core),
                [.verified, .evictable].contains(slot.phase), !reuse.isProtected(old)
            else { continue }
            let score = reuse.score(old)
            if victim == nil || score < victim!.score { victim = (i, score) }
            if old / 512 != key / 512, old / 512 != previous, cold == nil || score < cold!.score { cold = (i, score) }
        }
        let indexed = reuseMinima(key, reuse: reuse, core)
        guard indexed.victim?.index == victim?.index, indexed.victim?.score == victim?.score,
            indexed.cold?.index == cold?.index, indexed.cold?.score == cold?.score
        else {
            return
                "victim \(String(describing: indexed.victim))/\(String(describing: indexed.cold)) vs \(String(describing: victim))/\(String(describing: cold))"
        }
        return nil
    }

    // MARK: Shrinking

    package func orderRetirement(_ candidates: inout [Int], sizeClass: Int, in core: Core) {
        let slots = core.classes[0]
        // Compute expensive reuse scores once per eligible slab, not on every sort
        // comparison. Eligibility and scores cannot change while this lock is held.
        var ranked: [(id: Int, transient: Bool, score: Double, referenced: Int)] = []
        for id in candidates {
            let slab = slots.slabs[id]!
            let transient = reuseDecode && slab.slots.contains { $0 < ExpertReusePolicy.transientSlots }
            var score = 0.0
            var referenced = 0
            for index in slab.slots {
                if slots.slots[index] != nil && self.referenced[index] { referenced += 1 }
                if reuseDecode, !transient, let reuse, let key = slots.slots[index]?.key { score += reuse.score(key) }
            }
            ranked.append((id, transient, score, referenced))
        }
        ranked.sort {
            if $0.transient != $1.transient { return !$0.transient }
            if $0.score != $1.score { return $0.score < $1.score }
            return $0.referenced != $1.referenced ? $0.referenced < $1.referenced : $0.id < $1.id
        }
        candidates = ranked.map(\.id)
    }
}
