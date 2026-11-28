import CExactIO
import Foundation
import Metal
import OrigamiEngine

/// The bounded compressor gives exactly the full-history kernel's bits for decode, uneven prompt chunks, a
/// prompt/decode transition and an unfinished group restored after it has been overwritten. Both families use
/// this path; include every supported GLM pool width, the real dimensions and the maximum 256-row chunk.
func familyBoundedPooling() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family bounded pooling: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-pool-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let runner = try FamilyRunner(
        weights: FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device), context: 1024)
    let kernels = try FamilyKernels(device: device)
    func buffer(_ values: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: values, length: max(16, values.count * 4), options: .storageModeShared)!
    }
    let rows = 600
    for pool in 1...8 {
        let dim = pool == 2 ? 512 : 128
        let keys = buffer((0..<rows * dim).map { Float(($0 * 37 + 17) % 1009 - 504) / 127 })
        let gates = buffer((0..<rows * dim).map { Float(($0 * 13 + 5) % 211 - 105) / 17 })
        let ape = buffer((0..<pool * dim).map { Float(($0 * 19) % 113 - 56) / 31 })
        let expected = buffer([Float](repeating: .nan, count: (rows / pool + 1) * dim))
        runner.begin()
        try kernels.dispatch(
            runner.encoder, "kpool_pool", [keys, gates, ape, expected].map { FamilyKernels.Ref($0) },
            ints: [Int32(dim), 0, Int32(rows / pool), Int32(pool)],
            grid: MTLSize(width: (dim + 63) / 64, height: rows / pool, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        try runner.flush(wait: true)
        runner.end()
        for pattern in [[1, 256, 1, 3, 17, 1, 256, 65], [3, 2, 1, 7, 255, 1, 1, 256, 74]] {
            try require(pattern.reduce(0, +) == rows, "invalid test pattern")
            let key = buffer([Float](repeating: .nan, count: pool * dim))
            let gate = buffer([Float](repeating: .nan, count: pool * dim))
            let actual = buffer([Float](repeating: .nan, count: (rows / pool + 1) * dim))
            var first = 0
            let capacity = (pool - 1 + (pattern.max() ?? 1)) * dim
            try runner.prepareScratch(["pool-key": capacity, "pool-gate": capacity])
            runner.begin()
            for count in pattern {
                let result = try runner.poolRows(
                    runner.encoder, key: key, gate: gate, ape: ape, output: actual,
                    first: first, rows: count, dim: dim, pool: pool
                ) { k, g in
                    for (source, target) in [(keys, k), (gates, g)] {
                        try kernels.dispatch(
                            runner.encoder, "copy_rows", [FamilyKernels.Ref(source, first * dim * 4), target],
                            ints: [Int32(dim), Int32(count)], grid: MTLSize(width: (dim + 255) / 256, height: count, depth: 1),
                            group: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }
                try require(result.first == first / pool && result.count == (first + count) / pool - first / pool, "completed range")
                first += count
            }
            try runner.flush(wait: true)
            runner.end()
            try require(memcmp(expected.contents(), actual.contents(), rows / pool * dim * 4) == 0, "pool \(pool), chunks \(pattern)")
            let tail = rows % pool
            if tail > 0 {
                try require(
                    memcmp(key.contents(), keys.contents() + (rows - tail) * dim * 4, tail * dim * 4) == 0
                        && memcmp(gate.contents(), gates.contents() + (rows - tail) * dim * 4, tail * dim * 4) == 0, "pool \(pool) carry")
            }
        }
    }
}

/// Real model state descriptors on tiny nonzero fixtures: raw carry is fixed-size recurrent state at position zero,
/// and restoring it through either the tool-anchor convention or a saved state reproduces the next decode bits.
func familyPoolCheckpoints() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family pool checkpoint: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-pool-state-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for family in [ModelFamily.deepseek, .glm] {
        let gguf = try poolModelFixture(family, in: folder)
        let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
        let model = try makeFamilyModel(weights, expertBudget: 1 << 20, context: 64) { _ in }
        let initial = model.stateParts(at: 0)
        let at32 = model.stateParts(at: 32)
        let recurrentBytes = initial.reduce(0) { $0 + $1.bytes }
        let expectedRecurrent = family == .deepseek ? (8 + 256) * 8 * 4 + 2 * 2 * 8 * 4 : 2 * 4 * 8 * 4
        try require(recurrentBytes == expectedRecurrent, "\(family) recurrent state \(recurrentBytes), wanted \(expectedRecurrent)")
        try require(zip(initial, at32).filter { $0.0.bytes > 0 }.allSatisfy { $0.0.bytes == $0.1.bytes }, "carry grows with context")
        let prefix = [1, 2, 3]
        let logits = try model.forward(prefix)
        let anchor = model.stateParts(at: 0).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
        let saved = folder.appendingPathComponent(family.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: saved, withIntermediateDirectories: true)
        try FamilyStateFile.write(model.stateParts(at: prefix.count), logits: logits, tokens: prefix, epoch: "pool", to: saved)
        let expected = try model.forward([9])
        let expectedState = model.stateParts(at: prefix.count + 1).map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
        func stateMatches() -> Bool {
            zip(model.stateParts(at: prefix.count + 1), expectedState).allSatisfy { part, data in
                part.bytes == data.count
                    && (data.isEmpty || data.withUnsafeBytes { memcmp(part.buffer.contents(), $0.baseAddress!, $0.count) == 0 })
            }
        }
        // Releasing prompt scratch must leave the incomplete group and other recurrent state untouched.
        (model as? FamilyRunner)?.beginDecode()
        // An unfinished group overwrites the carry; ending on a completed group would leave old carry untouched.
        _ = try model.forward([10, 11, 12, 13, 14])
        for (part, data) in zip(model.stateParts(at: 0), anchor) where !data.isEmpty {
            data.withUnsafeBytes { part.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        model.restored(prefix)
        let anchored = try model.forward([9])
        try require(expected.map(\.bitPattern) == anchored.map(\.bitPattern) && stateMatches(), "\(family) tool-anchor carry restore")
        try model.reset()
        let header = try FamilyStateFile.header(saved, epoch: "pool", prompt: prefix + [9])
        _ = try FamilyStateFile.read(saved, header: header, into: model.stateParts(at: prefix.count))
        model.restored(prefix)
        let restored = try model.forward([9])
        try require(
            expected.map(\.bitPattern) == restored.map(\.bitPattern) && stateMatches() && expected.allSatisfy(\.isFinite),
            "\(family) saved carry restore")
    }
}

/// Actual page-rounded scratch replaces the startup allowance and is lent after its final buffers are released.
/// Each forward reclaims only its real plan from experts. Explicit expert_memory remains a cap; ledger charges,
/// state loans and retry after a denied allocation follow actual ownership through both phase transitions.
func familyWorkingSetPhases() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family working phases: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-working-phases-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let oldSlabs = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 64 << 10
    defer { FamilyExperts.slabBytes = oldSlabs }
    for family in [ModelFamily.deepseek, .glm] {
        for mode in 0..<3 {  // automatic with ledger, explicit cap with ledger, automatic without ledger
            let limit = FamilyRunner.baseWorkingBytes + (64 << 20)
            let ledger = mode < 2 ? try MemoryLedger(ceiling: limit + 1, transientMargin: 1, availableFloor: 1) : nil
            let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(family, in: folder)), device: device)
            let model = try makeFamilyModel(
                weights, expertBudget: mode == 1 ? 1 << 20 : nil, memoryLimit: limit,
                context: 8192, ledger: ledger
            ) { _ in }
            let runner = model as! FamilyRunner
            ledger?.setUsable(limit)
            runner.fitExperts()
            let full = runner.workingBytes
            let initialMaximum = runner.poolMaximum
            try require(full > 0 && runner.promptSlices > 1, "fixture has no extra slices")
            func expectedMaximum(working: UInt64, addedState: UInt64 = 0) -> UInt64 {
                (mode == 1 ? initialMaximum : initialMaximum + full - working) - addedState
            }
            func checkWorkingOwner(_ stage: String) throws {
                try require(
                    runner.workingBytes == runner.workingRetainedBytes
                        && runner.workingRetainedBytes == runner.workingAllocatedBytes,
                    "\(family), mode \(mode), \(stage): allowance differs from completed scratch owners")
                if let ledger {
                    try require(
                        ledger.snapshot().categories["family-working", default: 0] == runner.workingRetainedBytes,
                        "\(family), mode \(mode), \(stage): scratch ledger charge differs from its final owners")
                }
            }
            _ = try model.forward([1, 2, 3])
            try checkWorkingOwner("prefill")
            try require(
                runner.workingAllocatedBytes > 0 && runner.poolMaximum == expectedMaximum(working: runner.workingBytes),
                "prefill changed the explicit expert cap")
            let initialSlots = runner.experts.slotCount
            runner.beginDecode()
            try checkWorkingOwner("decode handoff")
            try require(
                runner.workingBytes == 0 && runner.workingAllocatedBytes == 0,
                "\(family), mode \(mode): prompt scratch or allowance retained")
            try require(runner.poolMaximum == expectedMaximum(working: 0), "explicit cap changed or actual scratch loan missing")
            try require(
                mode == 1 ? runner.experts.slotCount <= initialSlots : runner.experts.slotCount >= initialSlots,
                "returning scratch reduced the automatic pool or exceeded the explicit cap")
            _ = try model.forward([4])
            try checkWorkingOwner("decode")
            let decodeAllocated = runner.workingAllocatedBytes
            let decodeMaximum = runner.poolMaximum
            try require(decodeMaximum == expectedMaximum(working: decodeAllocated), "decode did not reclaim its actual scratch")
            runner.beginDecode()
            try require(
                runner.poolMaximum == decodeMaximum && runner.workingAllocatedBytes == decodeAllocated && decodeAllocated > 0,
                "repeated decode handoff changed its loan or discarded decode scratch")
            if mode == 0, let ledger {
                // No eager maximum reservation is required merely to mark prefill.
                // Exhaust real admission after optional expert slabs have retired,
                // then require the larger actual plan to fail without a phantom loan.
                runner.experts.setLimit(bytes: 0)
                let left = ledger.maximumUsableBytes - ledger.reservedBytes
                try require(left > 0, "no room for denied-admission fixture")
                let occupied = try ledger.reserve(bytes: left, category: "phase-check-pressure")
                let held = runner.heldState
                let position = runner.position
                let deniedReserved = ledger.reservedBytes
                try runner.beginPrefill()
                try require(
                    runner.workingBytes == decodeAllocated && runner.poolMaximum == decodeMaximum,
                    "marking prefill eagerly allocated its maximum")
                do {
                    try runner.prepareScratch(runner.scratchPlan(rows: 256), footprint: { 0 })
                    throw EngineError("Family working phases: denied actual plan succeeded")
                } catch let error as EngineError where error.code == "insufficient_memory" {}
                try checkWorkingOwner("denied prefill")
                try require(
                    runner.workingBytes == decodeAllocated && runner.poolMaximum == decodeMaximum
                        && runner.heldState == held && runner.position == position && ledger.reservedBytes == deniedReserved,
                    "denied actual plan retained a prospective allowance, changed state or leaked a charge")
                try ledger.release(occupied)
                try runner.prepareScratch(runner.scratchPlan(rows: 256), footprint: { 0 })
                try checkWorkingOwner("prefill retry")
                try require(
                    runner.workingAllocatedBytes > decodeAllocated
                        && runner.poolMaximum == expectedMaximum(working: runner.workingBytes),
                    "released admission did not permit the actual prefill plan")
            }
            let workingBeforePrefill = runner.workingBytes
            let maximumBeforePrefill = runner.poolMaximum
            try runner.beginPrefill()
            try require(
                runner.workingBytes == workingBeforePrefill && runner.poolMaximum == maximumBeforePrefill,
                "prefill marker eagerly reclaimed unused scratch")
            // Direct model callers admit their actual multi-row plan too.
            runner.beginDecode()
            _ = try model.forward([5, 6])
            try checkWorkingOwner("direct prefill")
            try require(
                runner.workingBytes > 0
                    && runner.poolMaximum == expectedMaximum(working: runner.workingBytes), "direct prefill bypassed actual admission")
            // State growth and working-set loans compose without either being counted twice.
            runner.beginDecode()
            let beforeState = runner.heldState
            let beforeMaximum = runner.poolMaximum
            try model.reserveState(8192)
            let added = runner.heldState - beforeState
            try require(runner.poolMaximum == beforeMaximum - added, "state growth lost the prompt loan")
            try runner.beginPrefill()
            try require(runner.poolMaximum == beforeMaximum - added, "prefill marker changed the state loan")
            try runner.prepareScratch(runner.scratchPlan(rows: 2), footprint: { 0 })
            try checkWorkingOwner("prefill after state growth")
            try require(
                runner.poolMaximum == expectedMaximum(working: runner.workingBytes, addedState: added),
                "actual prefill scratch and state growth were counted twice")
        }
    }
}

/// At position 0 nothing is live, so `reserveState` lets the old per-position buffers go before it allocates the new
/// ones: a shrink needs no headroom and a growth only the difference, never old and new together. A growth whose
/// footprint lags (released pages not reclaimed yet) waits and succeeds; one that cannot fit is refused and leaves an
/// empty, consistently charged state that the next, smaller request allocates. With live rows a denial or a failure
/// after the first replacement allocation leaves every old state pointer and byte intact, and the wait ignores
/// cancellation. Footprints are injected (only charges count); no kernels run.
func familyStateResizeAdmission() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family state admission: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-admission-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    // Keep the tiny fixture's physical cache minimum below its 64 MiB allowance. Production slabs are 64 MiB;
    // one such slab plus dense/state and extra prompt slices cannot fit this deliberately small owner budget.
    let previousSlabBytes = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 64 << 10
    defer { FamilyExperts.slabBytes = previousSlabBytes }
    for family in [ModelFamily.deepseek, .glm] {
        let path = try poolModelFixture(family, in: folder)
        let deniedWeights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let deniedLedger = try MemoryLedger(
            ceiling: 1 << 16, transientMargin: 1, availableFloor: 1,
            budgetMode: .explicit)
        do {
            _ = try makeFamilyModel(
                deniedWeights, expertBudget: nil, memoryLimit: deniedLedger.ceiling,
                context: 8192, ledger: deniedLedger
            ) { _ in }
            throw EngineError("Family state admission: impossible startup succeeded")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try require(
            deniedWeights.residentBytes == 0 && deniedWeights.startupAdmission == nil && deniedLedger.reservedBytes == 0,
            "\(family): impossible startup allocated dense weights or leaked charges")
        let limit = FamilyRunner.baseWorkingBytes + (64 << 20)
        // A large explicit ceiling lets real startup admissions pass; an occupying charge sets the headroom.
        let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: limit, context: 8192, ledger: ledger) { _ in }
        let runner = model as! FamilyRunner
        try require(
            weights.startupAdmission == nil && ledger.snapshot().categories["family-startup"] == nil,
            "\(family): startup charges were not transferred to final owners")
        try require(
            runner.experts.leastAllocatedBytes > runner.experts.leastBytes,
            "\(family): fixture no longer exercises slab-rounded minimum admission")
        runner.fitExperts()
        let small = runner.positionStateBytes(at: 4096)
        let large = runner.positionStateBytes(at: 8192)
        let empty = runner.positionStateBytes(at: 0)
        let fixed = runner.heldState - runner.positionStateBytes(at: runner.stateCapacity)
        try require(runner.stateCapacity == 4096 && large > small && small > empty, "\(family): unexpected fixture state sizes")
        func charge() -> UInt64 { ledger.snapshot().categories["family-state"] ?? 0 }
        func consistent(_ capacity: Int) -> Bool {
            runner.stateCapacity == capacity && charge() == runner.heldState
                && runner.heldState == fixed + runner.positionStateBytes(at: capacity)
        }
        // Physical headroom of `room` bytes: footprint samples of 0 leave only charges and the recorded overhead.
        func occupy(_ room: UInt64) throws -> MemoryLedger.Reservation {
            try ledger.reserve(
                bytes: ledger.ceiling - ledger.reservedBytes - ledger.physicalOverheadBytes - room, category: "state-check-occupied")
        }
        // A resize, returning the state's highest charge while a replacement buffer was allocated.
        func resize(_ positions: Int, footprint: @escaping () -> UInt64 = { 0 }) throws -> UInt64 {
            var peak: UInt64 = 0
            try autoreleasepool {
                try runner.reserveState(
                    positions,
                    grow: { old, count in
                        peak = max(peak, charge())
                        return try runner.regrown(old, count)
                    }, footprint: footprint)
            }
            return peak
        }
        // Growth at position 0 while released pages are still counted: the first samples are over the ceiling.
        var probes = 0
        let lagging: () -> UInt64 = {
            probes += 1
            return probes <= 3 ? ledger.ceiling : 0
        }
        var peak = try resize(8192, footprint: lagging)
        try require(probes >= 5 && consistent(8192), "\(family): a growth did not wait for reclamation (\(probes) samples)")
        try require(peak <= fixed + large + empty, "\(family): growth at position 0 held the old buffers (\(peak))")
        // A shrink at position 0 with less headroom than the new size: the old buffers go first.
        var occupied = try occupy(small - 1)
        peak = try resize(100)
        try require(consistent(4096) && peak <= fixed + small + empty, "\(family): a shrink at position 0 needed extra memory")
        try ledger.release(occupied)
        // A growth that cannot fit even once the old buffers went: refused without a wait, empty and consistent; the
        // next, smaller request is served (no stuck server).
        occupied = try occupy(large - small + empty - 1)
        let refused = Date()
        do {
            _ = try resize(8192)
            throw EngineError("Family state admission: an impossible growth succeeded")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try require(Date().timeIntervalSince(refused) < 0.5, "\(family): a refusal no reclamation can fix waited")
        try require(consistent(0) && runner.position == 0, "\(family): a refused growth left inconsistent state or charges")
        _ = try resize(100)
        try require(consistent(4096), "\(family): the request after a refusal was not served")
        try ledger.release(occupied)
        // A client leaving during the wait at position 0 costs no conversation: cancelled, empty and consistent.
        runner.shouldContinue = { false }
        do {
            _ = try resize(8192, footprint: { ledger.ceiling })
            throw EngineError("Family state admission: a cancelled wait succeeded")
        } catch let error as EngineError where error.code == "cancelled" {}
        try require(consistent(0), "\(family): a cancelled wait left inconsistent state or charges")
        runner.shouldContinue = { true }
        _ = try resize(100)

        // Live rows: the replacement coexists with the old buffers and copies them.
        model.restored(Array(repeating: 1, count: 1000))
        let capacity = runner.stateCapacity
        let maximum = runner.poolMaximum
        let held = runner.heldState
        let original = runner.stateParts(at: capacity)
        for (i, part) in original.enumerated() { memset(part.buffer.contents(), Int32(17 + i), part.bytes) }
        let bytes = original.map { Data(bytes: $0.buffer.contents(), count: $0.bytes) }
        func unchanged() -> Bool {
            let current = runner.stateParts(at: capacity)
            return runner.stateCapacity == capacity && runner.poolMaximum == maximum && current.count == original.count
                && zip(current, original).allSatisfy { $0.0.buffer === $0.1.buffer }
                && zip(current, bytes).allSatisfy { part, data in
                    part.bytes == data.count
                        && (data.isEmpty || data.withUnsafeBytes { memcmp(part.buffer.contents(), $0.baseAddress!, $0.count) == 0 })
                }
        }
        occupied = try occupy(large / 2)
        let deniedReserved = ledger.reservedBytes
        do {
            _ = try resize(8192)
            throw EngineError("Family state admission: denied replacement succeeded")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try require(unchanged() && ledger.reservedBytes == deniedReserved, "\(family): denial changed pointers, bytes or charges")
        try ledger.release(occupied)
        var allocations = 0
        do {
            try autoreleasepool {
                try runner.reserveState(8192) { old, count in
                    allocations += 1
                    if allocations == 2 { throw EngineError("injected replacement failure", code: "state_fixture_failure") }
                    return try runner.regrown(old, count)
                }
            }
            throw EngineError("Family state admission: injected replacement succeeded")
        } catch let error as EngineError where error.code == "state_fixture_failure" {}
        try require(
            allocations == 2 && unchanged() && charge() == held,
            "\(family): partial allocation published or leaked its state charge")
        // A lagging footprint with the client gone: the wait finishes, as a cancel must not cost live rows.
        runner.shouldContinue = { false }
        probes = 0
        _ = try resize(8192, footprint: lagging)
        runner.shouldContinue = { true }
        let current = runner.stateParts(at: capacity)
        let added = runner.heldState - held
        try require(
            probes >= 5 && consistent(8192) && runner.poolMaximum == maximum - added,
            "\(family): successful replacement lost exact allocation accounting")
        try require(
            zip(current, bytes).allSatisfy { part, data in
                data.isEmpty || data.withUnsafeBytes { memcmp(part.buffer.contents(), $0.baseAddress!, $0.count) == 0 }
            }, "\(family): successful replacement changed prefix bytes")
    }
}

/// The serving backend, the synthetic model and the state lent to the expert slots: a long conversation, then a
/// different prompt in the same session with less memory left than its new state takes (the old state goes first at
/// position 0), then a third one. Both are served, their teacher-forced log-probabilities bitwise a fresh session's.
/// Before the fix the second was refused (`family-state` denied) and so was every request after it.
func familySecondConversationAfterLongOne() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-second-conversation-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}")
    let teacher = [97, 98, 99, 100]
    // Byte tokens only: lowercase letters, never the "Hi" merge.
    func text(_ count: Int, seed: UInt64) -> String {
        var state = seed
        return String(
            (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Character(UnicodeScalar(UInt8(97 + (state >> 33) % 26)))
            })
    }
    let prompts = [text(4_500, seed: 1), text(1_200, seed: 2), text(1_300, seed: 3)]
    for family in [ModelFamily.glm, .deepseek] {
        let path = try syntheticFamilyGGUF(family, in: folder)
        func session() throws -> (FamilyServingBackend, FamilyRunner, MemoryLedger) {
            let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 1)
            let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
            let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: 8 << 30, context: 8192, ledger: ledger) { _ in }
            let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
            return (backend, model as! FamilyRunner, ledger)
        }
        func serve(_ backend: FamilyServingBackend, _ content: String) throws -> [UInt64] {
            let body: [String: Any] = [
                "model": "family", "messages": [["role": "user", "content": content]], "max_tokens": teacher.count,
                "temperature": 0, "origami_teacher_tokens": teacher, "prompt_cache_policy": "no_store",
            ]
            let request = try ServingRequest.parse(
                JSONSerialization.data(withJSONObject: body), contextTier: 8192, defaults: ServingDefaults())
            let stats = try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
            let scores = stats.teacher?.tokenLogProbabilities ?? []
            guard scores.count == teacher.count else { throw EngineError("\(family): the reply was not teacher-forced") }
            return scores.map { ($0 ?? .nan).bitPattern }
        }
        let fresh = try [prompts[1], prompts[2]].map { prompt in try autoreleasepool { try serve(session().0, prompt) } }
        try autoreleasepool {
            let (backend, runner, ledger) = try session()
            _ = try serve(backend, prompts[0])
            let small = runner.positionStateBytes(at: 4096)
            guard runner.stateCapacity == 8192, runner.positionStateBytes(at: 8192) > small else {
                throw EngineError("\(family): the first conversation did not grow the state (\(runner.stateCapacity))")
            }
            // The synthetic state is smaller than its prompt scratch, unlike a real model's at a long context: the
            // second prompt's scratch is allocated first, so only the state needs room. Less room than the next state
            // takes: holding both sizes at once would be refused.
            try runner.prepareScratch(runner.scratchPlan(rows: prompts[1].count))
            let occupied = try ledger.reserve(
                bytes: ledger.maximumUsableBytes - ledger.reservedBytes - (small - 1), category: "second-conversation-occupied")
            let second = try serve(backend, prompts[1])
            try ledger.release(occupied)
            guard runner.stateCapacity == 4096, second == fresh[0] else {
                throw EngineError("\(family): the second conversation differs from a fresh session's (\(runner.stateCapacity))")
            }
            guard try serve(backend, prompts[2]) == fresh[1] else {
                throw EngineError("\(family): the third conversation differs from a fresh session's")
            }
        }
    }
}

/// Startup owner and page-rounded physical checks are independent, deterministic and overflow safe. Temporary CPU
/// decode staging releases its charge, while a partially admitted failed startup drains every previous reservation.
func familyStartupAdmissionBoundaries() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family startup bounds: " + what) } }
    let page = UInt64(getpagesize())
    let limit = 16 * page
    let ledger = try MemoryLedger(ceiling: limit, transientMargin: 1, availableFloor: 0, budgetMode: .explicit)
    do {
        let admission = try FamilyStartupAdmission(ledger: ledger, fixedBytes: 4 * page, footprint: { 2 * page })
        let staging = try admission.reserve(bytes: 3 * page)
        try require(ledger.reservedBytes == 7 * page, "CPU staging is uncharged")
        admission.release(staging)
        try require(ledger.reservedBytes == 4 * page, "CPU staging charge survived its scope")
        _ = try admission.reserve(bytes: 8 * page)
        do {
            _ = try admission.reserve(bytes: 4 * page)
            throw EngineError("Family startup bounds: owner overflow accepted")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        do {
            _ = try admission.reserve(bytes: .max)
            throw EngineError("Family startup bounds: arithmetic overflow accepted")
        } catch let error as EngineError where error.code == "insufficient_memory" {}
        try require(ledger.reservedBytes == 12 * page, "denied growth changed previous charges")
    }
    try require(ledger.reservedBytes == 0, "failed startup did not drain earlier charges")
    let tight = try FamilyStartupAdmission(ledger: ledger, fixedBytes: 0, footprint: { limit - page / 2 })
    do {
        _ = try tight.reserve(bytes: 1)
        throw EngineError("Family startup bounds: rounded physical overflow accepted")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    tight.close()
    try require(ledger.reservedBytes == 0, "physical denial leaked a charge")
}

/// Hard reservation failure cannot create an uncharged physical slab or a false resident hit; retry rereads its bytes.
func familyExpertSlabAdmission() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family slab admission: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-slab-admission-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let priorSlab = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 32 << 10
    defer { FamilyExperts.slabBytes = priorSlab }
    let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(.glm, in: folder, experts: 3)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: 1 << 20)
    let denied = try MemoryLedger(ceiling: 2, transientMargin: 1, availableFloor: 1)
    experts.ledger = denied
    do {
        _ = try experts.fetch(layer: 0, experts: [0])
        throw EngineError("Family slab admission: denied slab allocated")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    try require(experts.residentBytes == 0 && denied.reservedBytes == 0, "denial left an allocation or charge")
    let admitted = try MemoryLedger(ceiling: (32 << 10) + 1, transientMargin: 1, availableFloor: 1)
    experts.ledger = admitted
    let result = try experts.fetch(layer: 0, experts: [0, 1])
    try require(
        experts.counters.hits == 0 && experts.counters.readBytes > 0
            && admitted.snapshot().categories["family-experts"] == UInt64(experts.residentBytes),
        "retry used a false resident hit or omitted its physical charge")
    var expected = Data(count: experts.bank(0)!.bytes.reduce(0, +))
    try expected.withUnsafeMutableBytes { try experts.readRecord(layer: 0, expert: 0, into: $0.baseAddress!) }
    try require(
        expected.withUnsafeBytes { memcmp(result[0].buffer.contents() + result[0].offsets[0], $0.baseAddress!, $0.count) == 0 },
        "retry did not read exact record bytes")
    let reused = try experts.fetch(layer: 0, experts: [2])
    try expected.withUnsafeMutableBytes { try experts.readRecord(layer: 0, expert: 2, into: $0.baseAddress!) }
    try require(
        experts.residentBytes == 32 << 10 && admitted.reservedBytes == 32 << 10
            && expected.withUnsafeBytes { memcmp(reused[0].buffer.contents() + reused[0].offsets[0], $0.baseAddress!, $0.count) == 0 },
        "denied growth did not reuse an existing eligible slot")
}

/// Backpressure for a demand slab (ROADMAP R5): denied while pages just released are still in the footprint, it waits
/// for them (bounded) and is admitted, and the denial does not count those pages as lasting overhead.
func familyExpertSlabWaitsForReclaim() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family slab backpressure: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-slab-wait-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let priorSlab = FamilyExperts.slabBytes
    FamilyExperts.slabBytes = 32 << 10
    defer { FamilyExperts.slabBytes = priorSlab }
    let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(.glm, in: folder, experts: 3)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: 1 << 20)
    // Memory another owner just gave back: 64 MiB still in the footprint, against 32 MiB of room, freed 0.1 s later.
    let held = 64 << 20
    let ledger = try MemoryLedger(
        ceiling: origami_physical_footprint() + (32 << 20), transientMargin: 1, availableFloor: 1, budgetMode: .explicit)
    experts.ledger = ledger
    let memory = mmap(nil, held, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
    guard let memory, memory != MAP_FAILED else { throw EngineError("Family slab backpressure: mmap failed") }
    memset(memory, 1, held)
    let address = Int(bitPattern: memory)
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { munmap(UnsafeMutableRawPointer(bitPattern: address), held) }
    let result = try experts.fetch(layer: 0, experts: [0])
    var expected = Data(count: experts.bank(0)!.bytes.reduce(0, +))
    try expected.withUnsafeMutableBytes { try experts.readRecord(layer: 0, expert: 0, into: $0.baseAddress!) }
    try require(
        experts.residentBytes == 32 << 10 && ledger.snapshot().categories["family-experts"] == 32 << 10
            && expected.withUnsafeBytes { memcmp(result[0].buffer.contents() + result[0].offsets[0], $0.baseAddress!, $0.count) == 0 },
        "the slab was not admitted after the memory came back")
    try require(
        ledger.physicalOverheadBytes + (16 << 20) < ledger.ceiling,
        "the denial counted released pages as overhead: \(ledger.physicalOverheadBytes) of \(ledger.ceiling)")
}

func poolModelFixture(_ family: ModelFamily, in folder: URL, experts: Int = 2) throws -> URL {
    let architecture = family == .deepseek ? "deepseek41" : "glm5-next"
    var config: [String: Any] = [
        "model_type": family == .deepseek ? "deepseek_v41" : "glm5_next", "num_hidden_layers": 1,
        "hidden_size": 32, "vocab_size": 32, "n_routed_experts": experts, "num_experts_per_tok": 1, "moe_intermediate_size": 32,
        "hc_mult": 4, "hc_sinkhorn_iters": 3, "num_attention_heads": 1, "q_lora_rank": 8, "index_n_heads": 1,
        "index_head_dim": 8, "index_topk": 8, "max_position_embeddings": 8192,
    ]
    var tensors: [(String, [Int], Int)] = []
    var data: [String: Data] = [:]
    func add(_ name: String, _ shape: [Int], ones: Bool = false) {
        tensors.append((name, shape, 0))
        let values = (0..<shape.reduce(1, *)).map { ones ? Float(1) : Float(($0 * 13 + name.utf8.count) % 23 - 11) / 128 }
        data[name] = values.withUnsafeBytes { Data($0) }
    }
    for name in ["token_embd.weight", "output.weight"] { add(name, [32, 32]) }
    add("output_norm.weight", [32], ones: true)
    for name in ["attn_norm", "ffn_norm"] { add("blk.0.\(name).weight", [32], ones: true) }
    for part in ["attn", "ffn"] {
        add("blk.0.hc_\(part)_fn.weight", [24, 128])
        add("blk.0.hc_\(part)_scale.weight", [3])
        add("blk.0.hc_\(part)_base.weight", [24])
    }
    add("blk.0.attn_q_a.weight", [8, 32])
    add("blk.0.attn_q_a_norm.weight", [8], ones: true)
    add("blk.0.attn_q_b.weight", [8, 8])
    add("blk.0.indexer.attn_q_b.weight", [8, 8])
    add("blk.0.indexer.attn_k.weight", [8, family == .deepseek ? 8 : 32])
    add("blk.0.indexer.k_norm.weight", [8], ones: true)
    add("blk.0.indexer.proj.weight", [1, 32])
    add("blk.0.ffn_gate_inp.weight", [experts, 32])
    for part in ["gate", "up", "down"] {
        add("blk.0.ffn_\(part)_exps.weight", [experts, 32, 32])
        add("blk.0.ffn_\(part)_shexp.weight", [32, 32])
    }
    if family == .deepseek {
        config["head_dim"] = 8
        config["qk_rope_head_dim"] = 4
        config["o_groups"] = 1
        config["o_lora_rank"] = 8
        config["sliding_window"] = 8
        config["compress_ratios"] = [2]
        config["kv_source_layer_ids"] = [0]
        config["index_source_layer_ids"] = [0]
        add("blk.0.attn_kv.weight", [8, 32])
        add("blk.0.attn_kv_norm.weight", [8], ones: true)
        add("blk.0.attn_sinks.weight", [1])
        add("blk.0.attn_output_a.weight", [8, 8])
        add("blk.0.attn_output_b.weight", [32, 8])
        for part in ["kv", "gate"] { add("blk.0.attn_compressor_\(part).weight", [8, 32]) }
        add("blk.0.attn_compressor_norm.weight", [8], ones: true)
    } else {
        config["layer_types"] = ["full_attention"]
        config["qk_head_dim"] = 8
        config["v_head_dim"] = 8
        config["kv_lora_rank"] = 8
        config["index_kpool"] = 4
        add("blk.0.attn_kv_a_mqa.weight", [8, 32])
        add("blk.0.attn_kv_a_norm.weight", [8], ones: true)
        add("blk.0.attn_k_b.weight", [8, 8])
        add("blk.0.attn_v_b.weight", [8, 8])
        add("blk.0.attn_output.weight", [32, 8])
        add("blk.0.indexer.k_norm.bias", [8])
        add("blk.0.indexer.pool_gate.weight", [8, 32])
        add("blk.0.indexer.pool_ape.weight", [4, 8])
    }
    let path = folder.appendingPathComponent("\(architecture).gguf")
    let json = String(decoding: try JSONSerialization.data(withJSONObject: config), as: UTF8.self)
    try familyGGUF([("general.architecture", architecture), (architecture + ".config", json)], tensors: tensors, data: data, to: path)
    return path
}

/// Cancellation stops a prompt between layers but lets a one-row decode step finish (a client leaving mid-token costs
/// the reply, not the conversation): the state advances by the token, and the backend stops before the next one.
func familyDecodeStepFinishesOnCancel() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family decode cancel: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-decode-cancel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for family in [ModelFamily.deepseek, .glm] {
        let weights = try FamilyWeights(gguf: GGUFSource(try poolModelFixture(family, in: folder)), device: device)
        let model = try makeFamilyModel(weights, expertBudget: 1 << 20, context: 64) { _ in }
        _ = try model.forward([1, 2, 3])
        model.shouldContinue = { false }
        let logits = try model.forward([4])
        try require(model.position == 4 && logits.allSatisfy(\.isFinite), "\(family): a cancelled decode step did not finish")
        var stopped = false
        do { _ = try model.forward([5, 6]) } catch let error as EngineError where error.code == "cancelled" { stopped = true }
        try require(stopped, "\(family): a cancelled prompt ran to its end")
    }
}

/// The least memory limit a family model reports (`FamilyRunner.minimumLimit`, `/v1/status`'s `minimum_bytes`) is one it
/// starts at, and 1 MB less is refused as insufficient memory. Pools with unequal layer counts start only once the pool
/// with the fewest layers gets two waves from its share, which is more than two waves of every pool's records.
func familyStartMinimum() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family start minimum: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-start-minimum-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    func start(_ path: URL, limit: UInt64) throws -> FamilyRunner {
        // Automatic mode: only the limit decides, not this process's footprint after the checks run before this one.
        let ledger = try MemoryLedger(ceiling: limit, transientMargin: 1, availableFloor: 1, budgetMode: .automatic)
        let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
        return try makeFamilyModel(weights, expertBudget: nil, memoryLimit: limit, context: 8192, ledger: ledger) { _ in } as! FamilyRunner
    }
    for family in [ModelFamily.deepseek, .glm] {
        let path = try poolModelFixture(family, in: folder)
        let minimum = try start(path, limit: 8 << 30).minimumLimit
        try require(try start(path, limit: minimum).minimumLimit == minimum, "\(family): the minimum moved when started at it")
        do {
            _ = try start(path, limit: minimum - 1_000_000)
            throw EngineError("Family start minimum: \(family) started 1 MB under its minimum")
        } catch let error as EngineError where error.code == "insufficient_memory" {
            try require(error.description.hasPrefix("the model does not fit in a "), "\(family): \(error)")
        }
    }
    // More than this Mac allows at its most: said so, with both figures, not "raise the limit".
    let beyond = FamilyRunner.beyondThisMac(least: 21_000_000_000, maximum: 18_000_000_000)?.description ?? ""
    try require(
        beyond.hasPrefix("this model does not fit this Mac at this context") && beyond.contains(OrigamiPlan.gigabytes(21_000_000_000))
            && beyond.contains(OrigamiPlan.gigabytes(18_000_000_000))
            && FamilyRunner.beyondThisMac(least: 17_000_000_000, maximum: 18_000_000_000) == nil, "beyond this Mac: \(beyond)")
    // Layer 1 alone in its pool, layers 2 and 3 in another: the share of the one-layer pool sets the start.
    let weights = try FamilyWeights(
        gguf: GGUFSource(try expertFixtureGGUF(in: folder.appendingPathComponent("pools"), types: [1: 8, 2: 12, 3: 12])),
        device: device)
    let size = [1: 212_992, 2: 114_688]
    let least = 2 * weights.geometry.expertsPerToken
    let startBytes = try FamilyExperts.startBytes(weights)
    try require(
        startBytes == UInt64(least * (size[1]! + 2 * size[2]!)) && startBytes > UInt64(least * (size[1]! + size[2]!)),
        "start \(startBytes)")
    try require(try FamilyExperts(weights: weights, budget: startBytes).leastBytes < startBytes, "pools at the start")
    do {
        _ = try FamilyExperts(weights: weights, budget: startBytes - 1)
        throw EngineError("Family start minimum: pools started under their start")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
}
