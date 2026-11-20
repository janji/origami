import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func sourceRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError(message) }
}
private func sourceWait(_ gate: DispatchSemaphore) throws {
    guard gate.wait(timeout: .now() + 5) == .success else { throw EngineError("Source-selection gate timed out") }
}
private func sourceReject(_ operation: () throws -> Void) throws {
    do { try operation() } catch is EngineError { return }
    throw EngineError("Invalid source-selection operation was accepted")
}
private let sourceEqualRates = ExpertSourceCalibration(
    queueDepths: [1], internalBytesPerSecond: [1e9], externalBytesPerSecond: [1e9], label: "equal-fixture")
private let sourceInternalFaster = ExpertSourceCalibration(
    queueDepths: [1], internalBytesPerSecond: [4e9], externalBytesPerSecond: [1e9], label: "internal-fixture")
private let sourceExternalFaster = ExpertSourceCalibration(
    queueDepths: [1], internalBytesPerSecond: [1e9], externalBytesPerSecond: [4e9], label: "external-fixture")

private final class SourceObservations: @unchecked Sendable {
    private let lock = NSLock()
    private var sources: [IOSource] = []
    private var tickets: [IOTicket] = []
    private var failures: [String] = []
    @discardableResult func read(_ source: IOSource) -> Int {
        lock.lock()
        defer { lock.unlock() }
        sources.append(source)
        return sources.count
    }
    func add(_ ticket: IOTicket) {
        lock.lock()
        tickets.append(ticket)
        lock.unlock()
    }
    func fail(_ error: Error) {
        lock.lock()
        failures.append(String(describing: error))
        lock.unlock()
    }
    var reads: [IOSource] {
        lock.lock()
        defer { lock.unlock() }
        return sources
    }
    var admitted: [IOTicket] {
        lock.lock()
        defer { lock.unlock() }
        return tickets
    }
    var errors: [String] {
        lock.lock()
        defer { lock.unlock() }
        return failures
    }
}
private func sourceScheduler(outstanding: Int = 4, concurrent: Int = 1) throws -> IOScheduler {
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let limits = IOScheduler.Limits(
        maximumConcurrent: concurrent, reservedDemandConcurrent: concurrent,
        maximumOutstanding: outstanding, reservedDemandOutstanding: outstanding,
        maximumOutstandingBytes: 32_000_000, reservedDemandBytes: 32_000_000)
    return try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
}
private func sourceCache(_ slots: Int = 2) throws -> ExpertCache {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    return try ExpertCache(device: device, ledger: ledger, byteLimit: UInt64(slots * PackStore.expertBytes), slotsPerSlab: slots)
}

/// A fixed curve for these checks (one Mac's measured expert-record rates); production starts from `.neutral`.
let sourceFixedCurve = ExpertSourceCalibration(
    queueDepths: [1, 2, 4, 8, 16, 32, 64],
    internalBytesPerSecond: [
        1118673754.743, 1785915819.884, 2568691445.413, 3152003990.172, 3228403835.240, 3201845682.216, 3196423273.294,
    ],
    externalBytesPerSecond: [
        2850114159.420, 3264761193.143, 3290701691.273, 3280216892.311, 3280254898.103, 3265024748.565, 3273458344.518,
    ],
    internalLatencySeconds: 0.000008501953069262134, externalLatencySeconds: 0, label: "fixed-check-curve")

func expertSourceCalibrationIsBoundedAndDefaultsAreOff() throws {
    let c = sourceFixedCurve
    try c.validate()
    let bytes: UInt64 = 12_000_000
    let estimate = c.estimatedSeconds(source: .internalSSD, bytesIncludingNew: bytes, jobsIncludingNew: 12, maximumConcurrent: 4)
    try sourceRequire(
        abs(estimate - (c.internalLatencySeconds + Double(bytes) / 2568691445.413)) < 1e-12,
        "Four-worker queue used a saturation curve from another depth")
    let defaults = ExpertExecutionConfiguration()
    try sourceRequire(
        defaults.prefillSourcePolicy == .packedPreferred && defaults.decodeSourcePolicy == .packedPreferred,
        "Adaptive placement was enabled by default")
    for invalid in [
        ExpertSourceCalibration(queueDepths: [], internalBytesPerSecond: [], externalBytesPerSecond: [], label: "empty"),
        ExpertSourceCalibration(
            queueDepths: [1, 1], internalBytesPerSecond: [1e9, 1e9], externalBytesPerSecond: [1e9, 1e9], label: "duplicate"),
        ExpertSourceCalibration(queueDepths: [1], internalBytesPerSecond: [.nan], externalBytesPerSecond: [1e9], label: "nan"),
        ExpertSourceCalibration(queueDepths: [1], internalBytesPerSecond: [1e9], externalBytesPerSecond: [.infinity], label: "infinity"),
        ExpertSourceCalibration(
            queueDepths: [1], internalBytesPerSecond: [1e9], externalBytesPerSecond: [1e9], internalLatencySeconds: -1, label: "negative"),
        ExpertSourceCalibration(queueDepths: [1], internalBytesPerSecond: [1e9], externalBytesPerSecond: [1e9], label: ""),
    ] { try sourceReject { try invalid.validate() } }
    try sourceReject {
        _ = try ExpertExecutionConfiguration(
            mode: .legacyOriginal, cacheBytes: UInt64(PackStore.expertBytes * 8),
            prefillSourcePolicy: .earliestCompletion)
    }
}

func expertSourceRuntimeConfigurationAndLegacyDecoding() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-source-config-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("runtime.json")
    func load(_ execution: [String: Any]) throws -> ExpertExecutionConfiguration {
        try JSONSerialization.data(withJSONObject: [
            "schema": "origami-config-v1", "profile": "exact",
            "packs_path": "/unused/verified-packs", "execution": execution,
        ]).write(to: file)
        return try RuntimeConfiguration.load(file).experts
    }
    let defaults = try load([:])
    try sourceRequire(
        defaults.prefillSourcePolicy == .packedPreferred && defaults.decodeSourcePolicy == .packedPreferred,
        "Omitted runtime policy enabled placement")
    let good: [String: Any] = [
        "expert_mode": "slabPackedPreferred", "expert_prefill_source_policy": "earliestCompletion",
        "expert_decode_source_policy": "packedPreferred",
    ]
    let parsed = try load(good)
    try sourceRequire(
        parsed.prefillSourcePolicy == .earliestCompletion && parsed.decodeSourcePolicy == .packedPreferred,
        "Runtime phase policies were lost")
    for invalid in [true, 1, "hedge", NSNull()] as [Any] {
        var changed = good
        changed["expert_prefill_source_policy"] = invalid
        try sourceReject { _ = try load(changed) }
    }
    var wrongMode = good
    wrongMode["expert_mode"] = "legacyOriginal"
    try sourceReject { _ = try load(wrongMode) }
    var custom = good
    custom["expert_source_calibration"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(sourceInternalFaster))
    try sourceRequire(try load(custom).sourceCalibration.internalBytesPerSecond == [4e9], "Custom calibration ignored")
    var old = try JSONSerialization.jsonObject(with: JSONEncoder().encode(parsed)) as! [String: Any]
    for key in ["prefillSourcePolicy", "decodeSourcePolicy", "sourceCalibration", "prefillCacheSlots"] { old.removeValue(forKey: key) }
    let restored = try JSONDecoder().decode(ExpertExecutionConfiguration.self, from: JSONSerialization.data(withJSONObject: old))
    try sourceRequire(
        restored.prefillSourcePolicy == .packedPreferred && restored.decodeSourcePolicy == .packedPreferred
            && restored.sourceCalibration.label == ExpertSourceCalibration.neutral.label,
        "Old execution snapshot no longer decodes with placement disabled")
}

func expertSourceQueuedBytesAndStableTiesChooseOneSource() throws {
    let scheduler = try sourceScheduler()
    let gate = DispatchSemaphore(value: 0)
    let started = DispatchSemaphore(value: 0)
    defer {
        gate.signal()
        scheduler.shutdown()
    }
    let generation = scheduler.currentGeneration
    let first = try scheduler.submitEarliest(
        availableSources: [.internalSSD, .externalSSD], estimate: sourceEqualRates.estimatedSeconds,
        priority: .demand, generation: generation, byteCount: 100
    ) { source, _ in
        try sourceRequire(source == .externalSSD, "Stable estimator tie did not choose packed")
        started.signal()
        try sourceWait(gate)
    }
    try sourceWait(started)
    let second = try scheduler.submitEarliest(
        availableSources: [.externalSSD, .internalSSD], estimate: sourceEqualRates.estimatedSeconds,
        priority: .demand, generation: generation, byteCount: 1
    ) { source, _ in
        try sourceRequire(source == .internalSSD, "Queued bytes did not shift placement")
    }
    try sourceRequire(second.source == .internalSSD && second.join() == .succeeded(bytes: 1), "Second source failed")
    let drives = scheduler.snapshot().drives
    try sourceRequire(
        drives.first { $0.source == .externalSSD }!.outstandingBytes == 100,
        "Losing source acquired a duplicate charge")
    gate.signal()
    try sourceRequire(first.join() == .succeeded(bytes: 100), "First source failed")
    try sourceRequire(
        scheduler.snapshot().drives.reduce(UInt64(0)) { $0 + $1.successfulBytes } == 101,
        "Two placement requests duplicated physical read bytes")
}

func expertSourceConcurrentSelectionAndAdmissionAreAtomic() throws {
    let scheduler = try sourceScheduler(outstanding: 1)
    let gate = DispatchSemaphore(value: 0)
    let observations = SourceObservations()
    let group = DispatchGroup()
    let generation = scheduler.currentGeneration
    defer {
        gate.signal()
        gate.signal()
        scheduler.shutdown()
    }
    for _ in 0..<32 {
        group.enter()
        DispatchQueue.global().async {
            defer { group.leave() }
            do {
                let ticket = try scheduler.submitEarliest(
                    availableSources: [.internalSSD, .externalSSD], estimate: sourceEqualRates.estimatedSeconds,
                    priority: .demand, generation: generation, byteCount: 1,
                    operation: { _, _ in try sourceWait(gate) },
                    completion: { _, _ in _ = scheduler.snapshot() })
                observations.add(ticket)
            } catch { observations.fail(error) }
        }
    }
    try sourceRequire(group.wait(timeout: .now() + 5) == .success, "Concurrent admission did not finish")
    try sourceRequire(
        observations.admitted.count == 2 && Set(observations.admitted.map(\.source)).count == 2,
        "Concurrent selection oversubscribed one source or failed to use free capacity")
    try sourceRequire(
        observations.errors.count == 30 && observations.errors.allSatisfy { $0.contains("admission denied") },
        "Full queues returned an unrelated failure")
    try sourceRequire(
        scheduler.snapshot().drives.allSatisfy { $0.outstanding == 1 && $0.outstandingBytes == 1 },
        "Atomic admission failed to retain both actual charges")
    gate.signal()
    gate.signal()
    for ticket in observations.admitted { try sourceRequire(ticket.join(timeout: 5) == .succeeded(bytes: 1), "Callback join deadlocked") }
    try sourceRequire(scheduler.snapshot().drives.allSatisfy { $0.outstanding == 0 && $0.outstandingBytes == 0 }, "Admission charge leaked")
    try sourceReject {
        _ = try scheduler.submitEarliest(
            availableSources: [], estimate: sourceEqualRates.estimatedSeconds,
            priority: .demand, generation: generation, byteCount: 1
        ) { _, _ in }
    }
}

func expertSourceBothPlacementsPublishIdenticalCompleteBytes() throws {
    let expected = (0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: ($0 / 257) ^ $0 ^ 0x5b) }
    for (calibration, chosen) in [(sourceInternalFaster, IOSource.internalSSD), (sourceExternalFaster, IOSource.externalSSD)] {
        let scheduler = try sourceScheduler()
        let cache = try sourceCache()
        let observations = SourceObservations()
        let gate = DispatchSemaphore(value: 0)
        let started = DispatchSemaphore(value: 0)
        let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { _, source, destination in
            observations.read(source)
            started.signal()
            try sourceWait(gate)
            expected.withUnsafeBytes { destination.copyMemory(from: $0) }
        }
        defer {
            gate.signal()
            scheduler.shutdown()
        }
        let key = try ExpertKey(layer: 3, expert: 7)
        let generation = scheduler.currentGeneration
        let request = try loader.request(key, generation: generation, policy: .earliestCompletion, calibration: calibration)
        try sourceWait(started)
        let duplicate = try loader.request(key, generation: generation, source: chosen == .internalSSD ? .externalSSD : .internalSSD)
        try sourceRequire(
            request.source == chosen && duplicate.source == chosen && observations.reads == [chosen],
            "Coalesced request changed placement or created a second writer")
        gate.signal()
        let a = try loader.joinAndPin(request)
        let b = try loader.joinAndPin(duplicate)
        try sourceRequire(
            Array(a.buffer.bytes) == expected && Array(b.buffer.bytes) == expected && a.ticket == b.ticket,
            "Exact source failed complete-byte parity or shared identity")
        let hit = try loader.request(key, generation: generation, policy: .earliestCompletion, calibration: calibration)
        let h = try loader.joinAndPin(hit)
        try sourceRequire(observations.reads == [chosen], "Resident hit issued placement I/O")
        try a.retireAfterGPUCompletion()
        try b.retireAfterGPUCompletion()
        try h.retireAfterGPUCompletion()
        try sourceRequire(
            scheduler.snapshot().drives.reduce(UInt64(0)) { $0 + $1.successfulBytes } == UInt64(PackStore.expertBytes),
            "Coalesced exact-source bytes were misattributed")
    }
}

func hotExpertPlacementOverridesCalibrationAndKeepsExplicitSource() throws {
    // Expert 7 of every layer is in the store's hot copy (`PackStore.hasHotExpert`), the others are not.
    final class Sevens: HotCopySource {
        func hotCopyRecordBytes(_ key: Int) -> Int? { 4096 }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
            into.initializeMemory(as: UInt8.self, repeating: 7, count: 4096)
        }
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-placement-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try PackStore(model: ModelSource(fixture: root, spans: [:]))
    store.attachHotCopy(
        try HotCopy.write(source: Sevens(), keys: (0..<48).map { $0 * 512 + 7 }, folder: root.appendingPathComponent("hot"), identity: "p"))
    let scheduler = try sourceScheduler()
    let cache = try sourceCache(2)
    let observations = SourceObservations()
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler, hotInternal: { store.hasHotExpert($0) }) { _, source, destination in
        observations.read(source)
        destination.initializeMemory(as: UInt8.self, repeating: 37)
    }
    defer { scheduler.shutdown() }
    for (expert, explicit, expected) in [
        (7, Optional<IOSource>.none, IOSource.internalSSD),
        (8, nil, .externalSSD), (7, .externalSSD, .externalSSD),
    ] {
        // Distinct layers prevent a resident hit from hiding the requested path.
        let key = try ExpertKey(layer: observations.reads.count, expert: expert)
        let request = try loader.request(
            key, generation: scheduler.currentGeneration, source: explicit,
            policy: .earliestCompletion, calibration: sourceExternalFaster)
        let loaded = try loader.joinAndPin(request)
        try sourceRequire(
            observations.reads.last == expected && loaded.buffer.bytes[0] == 37,
            "Hot placement failed to select exact internal bytes or honor explicit source")
        try loaded.retireAfterGPUCompletion()
    }
}

func expertSourceAdaptiveCancellationJoinsActualWriter() throws {
    let scheduler = try sourceScheduler()
    let cache = try sourceCache(1)
    let observations = SourceObservations()
    let gate = DispatchSemaphore(value: 0)
    let started = DispatchSemaphore(value: 0)
    let loader = SourceExpertLoader(cache: cache, scheduler: scheduler) { _, source, destination in
        let call = observations.read(source)
        if call == 1 {
            started.signal()
            try sourceWait(gate)
        }
        destination.initializeMemory(as: UInt8.self, repeating: call == 1 ? 0xb0 : 0x31)
    }
    defer {
        gate.signal()
        scheduler.shutdown()
    }
    let key = try ExpertKey(layer: 0, expert: 0)
    let old = try loader.request(
        key, generation: scheduler.currentGeneration, policy: .earliestCompletion, calibration: sourceInternalFaster)
    try sourceWait(started)
    try sourceRequire(old.source == .internalSSD && loader.cancel(old), "Actual selected writer was not cancelled")
    let next = try scheduler.beginGeneration()
    try sourceReject { _ = try loader.request(key, generation: next, policy: .earliestCompletion, calibration: sourceInternalFaster) }
    try sourceRequire(cache.snapshot().inFlightReads == 1, "Cancellation freed a running writer")
    gate.signal()
    do {
        _ = try loader.joinAndPin(old)
        throw EngineError("Cancelled record reached consumer")
    } catch is IOOperationCancelled {}
    let replacement = try loader.request(key, generation: next, policy: .earliestCompletion, calibration: sourceInternalFaster)
    let value = try loader.joinAndPin(replacement)
    try sourceRequire(
        value.buffer.bytes.allSatisfy { $0 == 0x31 } && observations.reads == [.internalSSD, .internalSSD],
        "Cancelled source was retried externally or overwrote new generation")
    try value.retireAfterGPUCompletion()
    try sourceRequire(
        scheduler.snapshot().drives.first { $0.source == .internalSSD }!.cancelled == 1, "Cancellation attributed to wrong source")
}

func expertSourceAdaptiveFallbackAndDriveAbsenceStayExact() throws {
    for packedAvailable in [true, false] {
        let scheduler = try sourceScheduler()
        let cache = try sourceCache()
        let observations = SourceObservations()
        let loader = SourceExpertLoader(cache: cache, scheduler: scheduler, packedAvailable: { packedAvailable }) {
            key, source, destination in
            observations.read(source)
            if source == .externalSSD {
                destination[0] = 0xff
                throw EngineError("external drive disappeared after partial read")
            }
            if key.expert == 1 { throw EngineError("original drive missing") }
            destination.initializeMemory(as: UInt8.self, repeating: 0x5c)
        }
        defer { scheduler.shutdown() }
        let request = try loader.request(
            ExpertKey(layer: 0, expert: 0), generation: scheduler.currentGeneration,
            policy: .earliestCompletion, calibration: sourceExternalFaster)
        let value = try loader.joinAndPin(request)
        try sourceRequire(value.buffer.bytes.allSatisfy { $0 == 0x5c }, "Partial packed data leaked through fallback")
        try sourceRequire(
            observations.reads == (packedAvailable ? [.externalSSD, .internalSSD] : [.internalSSD]),
            "Fallback/absent drive used incorrect queue or duplicate reads")
        try value.retireAfterGPUCompletion()
        let broken = try loader.request(
            ExpertKey(layer: 0, expert: 1), generation: scheduler.currentGeneration,
            policy: .earliestCompletion, calibration: sourceExternalFaster)
        do {
            _ = try loader.joinAndPin(broken)
            throw EngineError("Original read failure was hidden")
        } catch let error as EngineError {
            try sourceRequire(error.description.contains("Original expert read failed"), "Original failure was substituted")
        }
        let stats = scheduler.snapshot().drives
        try sourceRequire(
            stats.first { $0.source == .externalSSD }!.failed == (packedAvailable ? 2 : 0)
                && stats.first { $0.source == .internalSSD }!.succeeded == 1
                && stats.first { $0.source == .internalSSD }!.failed == 1, "Fallback failed actual per-drive attribution")
    }
}

func expertSourcePoliciesAreAppliedAtPhaseBoundaries() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    for (prefill, decode, expected) in [
        (ExpertSourcePolicy.packedPreferred, ExpertSourcePolicy.earliestCompletion, [IOSource.externalSSD, .internalSSD]),
        (.earliestCompletion, .packedPreferred, [.internalSSD, .externalSSD]),
    ] {
        let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
        let observations = SourceObservations()
        let configuration = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 2, slotsPerSlab: 1,
            demandWindow: 1, readConcurrency: 1, prefillSourcePolicy: prefill, decodeSourcePolicy: decode,
            sourceCalibration: sourceInternalFaster)
        let engine = try ExpertExecution(device: device, ledger: ledger, configuration: configuration) { _, source, destination in
            observations.read(source)
            destination.initializeMemory(as: UInt8.self, repeating: 0x2c)
        }
        try engine.beginSequence()
        let first = try engine.makeWindow(layer: 0, experts: [0])
        try first.consume(expert: 0) { _ in }
        try engine.beginDecodePhase()
        let second = try engine.makeWindow(layer: 0, experts: [1])
        try second.consume(expert: 1) { _ in }
        try engine.finishSequence()
        try sourceRequire(observations.reads == expected, "Per-phase source policy was not propagated into demand window")
        try sourceRequire(engine.snapshot().io.drives.allSatisfy { $0.outstanding == 0 }, "Phase policy left I/O unjoined")
    }
}
