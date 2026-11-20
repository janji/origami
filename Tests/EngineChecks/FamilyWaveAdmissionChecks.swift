import Foundation
import Metal
import OrigamiEngine

private func waveAdmissionRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError("Family physical wave admission: " + message) }
}

/// A large logical pool must not promise a large wave when pressure permits only its two-wave floor.
/// Verify the real record bytes while the following fetch protects the previous wave, including a pressure
/// transition after preparation. The synthetic latch and one-slot slabs avoid inducing host pressure.
private func familyPhysicalWaveAdmission() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-wave-admission-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let saved = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    let budget = UInt64(8 * (212_992 + 114_688))
    let recordPart = GGUFQuantize.bytes(8, count: 256 * 256)
    func verify(_ slots: [FamilyExperts.Slot], _ experts: [Int]) throws {
        try waveAdmissionRequire(slots.count == experts.count, "fetch returned an incomplete wave")
        for (slot, expert) in zip(slots, experts) {
            for part in 0..<3 {
                let bytes = UnsafeBufferPointer(
                    start: (slot.buffer.contents() + slot.offsets[part])
                        .assumingMemoryBound(to: UInt8.self), count: recordPart)
                try waveAdmissionRequire(
                    bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) },
                    "a following wave overwrote expert \(expert), part \(part)")
            }
        }
    }
    for startsUnderPressure in [true, false] {
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        try autoreleasepool {
            let pool = try FamilyExperts(weights: weights, budget: budget)
            pool.ledger = ledger
            try waveAdmissionRequire(
                pool.slots(layer: 1) >= 8 && pool.allocatedSlots(layer: 1) == 0,
                "fixture did not start with a large, entirely lazy pool")
            ledger.setOptionalGrowthAllowed(!startsUnderPressure)
            let width = try pool.prepareWave(layer: 1, maximumExperts: 4)
            let allocated = pool.allocatedSlots(layer: 1)
            try waveAdmissionRequire(
                width == (startsUnderPressure ? 1 : 4)
                    && allocated == (startsUnderPressure ? 2 : 4), "wave width used logical rather than allocated slots")
            try waveAdmissionRequire(
                pool.counters.misses == 0 && pool.resident(layer: 1).isEmpty,
                "preparation read or published expert records")
            let firstIDs = startsUnderPressure ? [0] : [0, 1]
            var previous = try pool.fetch(layer: 1, experts: firstIDs)
            var previousIDs = firstIDs
            ledger.setOptionalGrowthAllowed(false)  // Prepared slots remain usable after the pressure transition.
            let following = startsUnderPressure ? [[1], [2], [3]] : [[2, 3]]
            for ids in following {
                let next = try pool.fetch(layer: 1, experts: ids, keepingLast: true)
                try verify(previous, previousIDs)
                try verify(next, ids)
                previous = next
                previousIDs = ids
            }
            try waveAdmissionRequire(
                pool.allocatedSlots(layer: 1) == allocated && pool.counters.misses == 4,
                "pressure admitted optional slots or re-read a completed expert")
            try waveAdmissionRequire(
                ledger.reservedBytes == UInt64(pool.residentBytes),
                "allocated floor lost its final-owner reservation")
            pool.waitForReads()
            try waveAdmissionRequire(pool.pendingReads == 0, "a completed wave left queued readers")
        }
        try waveAdmissionRequire(ledger.reservedBytes == 0, "joined wave buffers leaked their owner charges")
    }
}

/// Exercise the production MoE loop, not just the admission helper. A four-expert prompt computed under a cold
/// pressure latch or a latch closed after preparation must retain every output bit of the unconstrained wave.
private func familyPhysicalWaveOutputParity() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-wave-output-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder)
    let hidden = 256
    let rows = 8
    let saved = (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire)
    FamilyExperts.slabBytes = 1
    FamilySlabStorage.wire = { _, _ in false }
    FamilySlabStorage.unwire = { _, _ in true }
    defer { (FamilyExperts.slabBytes, FamilySlabStorage.wire, FamilySlabStorage.unwire) = saved }
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { output in
            (0..<(hidden / 32)).flatMap { block in
                [0x00, code(output, block) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(output, block), count: 32)
            }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    let router = FamilyWeights.Matrix(
        buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8,
        width: hidden, outputs: 4, name: "router")
    let zero = FamilyWeights.Matrix(
        buffer: q8(hidden) { _, _ in 0 }, offset: 0, type: 8,
        width: hidden, outputs: hidden, name: "shared")
    let input = (0..<rows).flatMap { row in (0..<hidden).map { $0 / 32 == row % 4 ? Float(1) : 0 } }
    enum Pressure { case none, cold, afterPreparation }
    func run(_ pressure: Pressure) throws -> [UInt32] {
        let ledger = try MemoryLedger(ceiling: 32 << 20, transientMargin: 1, availableFloor: 1)
        let result: [UInt32] = try autoreleasepool {
            let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
            let runner = try FamilyRunner(weights: weights, context: 64)
            runner.experts = try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688)))
            runner.experts.ledger = ledger
            runner.poolMaximum = runner.experts.budget
            try runner.prepareScratch(runner.moeScratchPlan(rows: rows, sharedWidth: hidden))
            switch pressure {
            case .none: break
            case .cold: ledger.setOptionalGrowthAllowed(false)
            case .afterPreparation:
                let width = try runner.experts.prepareWave(layer: 1, maximumExperts: 4)
                try waveAdmissionRequire(width == 4, "preparation did not admit the reference wave")
                ledger.setOptionalGrowthAllowed(false)
            }
            let x = device.makeBuffer(bytes: input, length: input.count * 4, options: .storageModeShared)!
            let y = device.makeBuffer(length: input.count * 4, options: .storageModeShared)!
            runner.begin()
            do {
                try runner.moe(
                    layer: 1, router: router, bias: [], shared: FamilyRunner.FFN(gate: zero, up: zero, down: zero),
                    input: x, output: y, rows: rows, probability: { $0 }, normalize: false, scale: 1, sum: { $0 }, limit: 0)
            } catch {
                runner.end()
                throw error
            }
            runner.end()
            let values = Array(UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float.self), count: input.count))
            try waveAdmissionRequire(values.allSatisfy(\.isFinite) && values.contains(where: { $0 != 0 }), "non-finite or empty output")
            try waveAdmissionRequire(runner.experts.counters.misses == 4, "production loop lost or repeated an expert")
            if case .cold = pressure {
                try waveAdmissionRequire(runner.experts.allocatedSlots(layer: 1) == 2, "cold pressure escaped the two-wave floor")
            }
            return values.map(\.bitPattern)
        }
        try waveAdmissionRequire(ledger.reservedBytes == 0, "production wave retained owner charges after its joined scope")
        return result
    }
    let reference = try run(.none)
    try waveAdmissionRequire(try run(.cold) == reference, "cold-pressure waves changed output bits")
    try waveAdmissionRequire(try run(.afterPreparation) == reference, "pressure after preparation changed output bits")
}

func familyWaveAdmission() throws {
    try familyPhysicalWaveAdmission()
    try familyPhysicalWaveOutputParity()
}
