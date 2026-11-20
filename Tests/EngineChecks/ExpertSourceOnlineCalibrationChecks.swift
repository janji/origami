import Dispatch
import Foundation
import Metal
import OrigamiEngine

private func onlineRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError(message) }
}
private func onlineReject(_ operation: () throws -> Void) throws {
    do { try operation() } catch is EngineError { return }
    throw EngineError("Invalid online-calibration operation was accepted")
}
private func onlineWait(_ gate: DispatchSemaphore) throws {
    guard gate.wait(timeout: .now() + 5) == .success else { throw EngineError("Online-calibration gate timed out") }
}
private func near(_ a: Double, _ b: Double, _ tolerance: Double = 1e-12) -> Bool { abs(a - b) <= tolerance }

/// Hand-checkable curve: one record (1,000,000 bytes) takes 1 ms internal and 0.5 ms external at depth one,
/// and the aggregate rate doubles with depth so one read among `d` running reads is predicted at the same time.
private let onlineCurve = ExpertSourceCalibration(
    queueDepths: [1, 2, 4],
    internalBytesPerSecond: [1e9, 2e9, 4e9], externalBytesPerSecond: [2e9, 4e9, 8e9], label: "online-fixture")
private let onlineBytes = 1_000_000
private final class OnlineReads: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [IOSource] = []
    func record(_ source: IOSource) {
        lock.lock()
        values.append(source)
        lock.unlock()
    }
    var sources: [IOSource] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

func expertSourceOnlineEstimatorConvergesWithinBounds() throws {
    let settings = ExpertSourceOnlineCalibration(floorMultiple: 0.25, ceilingMultiple: 4, smoothing: 0.5, warmupObservations: 4)
    let estimator = try ExpertSourceOnlineEstimator(calibration: onlineCurve, settings: settings, maximumConcurrent: 4)
    try onlineRequire(
        near(estimator.predictedSeconds(source: .internalSSD, bytes: onlineBytes, concurrency: 1), 0.001)
            && near(estimator.predictedSeconds(source: .internalSSD, bytes: onlineBytes, concurrency: 2), 0.001)
            && near(estimator.predictedSeconds(source: .externalSSD, bytes: onlineBytes, concurrency: 4), 0.0005)
            && near(estimator.predictedSeconds(source: .externalSSD, bytes: onlineBytes, concurrency: 9), 0.0005),
        "Static per-read prediction differs from the curve at the observed depth")
    // Warm-up: three observations of a 3x slow internal drive leave the decision multiple at the static value.
    for _ in 0..<3 { estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 0.003, concurrency: 2) }
    var s = estimator.snapshot()
    try onlineRequire(
        estimator.multiple(for: .internalSSD) == 1 && !s.internalSSD.warm && s.internalSSD.observations == 3
            && near(s.internalSSD.smoothedMultiple, 3) && s.externalSSD.observations == 0 && s.observations == 3,
        "Warm-up did not hold the static multiple while training")
    estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 0.003, concurrency: 2)
    s = estimator.snapshot()
    try onlineRequire(
        s.internalSSD.warm && near(estimator.multiple(for: .internalSSD), 3)
            && near(s.internalSSD.singleRecordSeconds, 3 * Double(PackStore.expertBytes) / 1e9)
            && near(s.internalSSD.staticSingleRecordSeconds, Double(PackStore.expertBytes) / 1e9)
            && near(
                estimator.predictedSeconds(source: .internalSSD, bytes: onlineBytes, concurrency: 2)
                    * estimator.multiple(for: .internalSSD), 0.003),
        "Warm estimate is not the static curve scaled by the observed multiple")
    // Convergence toward a new constant ratio follows the closed form s_k = r + (s_0 - r)(1 - a)^k.
    for k in 1...10 {
        estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 0.004, concurrency: 2)
        let expected = 4 + (3 - 4) * pow(0.5, Double(k))
        try onlineRequire(
            near(estimator.multiple(for: .internalSSD), expected, 1e-9), "Exponential update differs from its closed form at step \(k)")
    }
    // A 1000x stall is clamped to the ceiling before it enters the estimate; the estimate never exceeds it.
    for _ in 0..<40 { estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 3, concurrency: 1) }
    s = estimator.snapshot()
    try onlineRequire(
        near(estimator.multiple(for: .internalSSD), 4, 1e-9) && s.internalSSD.maximumObservedMultiple == 4
            && s.internalSSD.smoothedMultiple <= 4, "Ceiling bound was exceeded")
    // Implausibly fast reads converge to the floor, never below it.
    for _ in 0..<60 { estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 1e-7, concurrency: 1) }
    s = estimator.snapshot()
    try onlineRequire(
        near(estimator.multiple(for: .internalSSD), 0.25, 1e-9) && s.internalSSD.minimumObservedMultiple == 0.25
            && s.internalSSD.smoothedMultiple >= 0.25, "Floor bound was crossed")
    // Rejected observations never train and are counted.
    for seconds in [0.0, -1.0, Double.nan, Double.infinity] {
        estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: seconds, concurrency: 1)
    }
    estimator.observe(source: .externalSSD, bytes: 0, seconds: 0.001, concurrency: 1)
    s = estimator.snapshot()
    try onlineRequire(
        s.externalSSD.observations == 0 && s.externalSSD.rejected == 5 && estimator.multiple(for: .externalSSD) == 1,
        "Invalid observations trained the external estimate")
    // The external source warms independently: four reads at twice the predicted time give multiple 2.
    for _ in 0..<4 { estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.001, concurrency: 4) }
    s = estimator.snapshot()
    try onlineRequire(
        near(estimator.multiple(for: .externalSSD), 2) && s.externalSSD.warm && s.externalSSD.observations == 4
            && near(s.externalSSD.observedSeconds, 0.004) && s.externalSSD.observedBytes == 4_000_000 && s.observations == 118,
        "External source did not warm independently")
    for invalid in [
        ExpertSourceOnlineCalibration(floorMultiple: 0), .init(floorMultiple: 1.5), .init(ceilingMultiple: 0.5),
        .init(ceilingMultiple: 65), .init(smoothing: 0), .init(smoothing: 1.5), .init(warmupObservations: 0),
        .init(floorMultiple: .nan), .init(ceilingMultiple: .infinity),
        // The hedge may never fire later than the static delay, and its floor cannot exceed its ceiling.
        .init(delayCeilingMultiple: 1.0000001), .init(delayCeilingMultiple: 4), .init(delayCeilingMultiple: .nan),
        .init(floorMultiple: 0.5, delayCeilingMultiple: 0.25),
    ] {
        try onlineReject { try invalid.validate() }
        try onlineReject { _ = try ExpertSourceOnlineEstimator(calibration: onlineCurve, settings: invalid, maximumConcurrent: 4) }
    }
    try onlineReject { _ = try ExpertSourceOnlineEstimator(calibration: onlineCurve, settings: settings, maximumConcurrent: 0) }
    try ExpertSourceOnlineCalibration().validate()
    try ExpertSourceOnlineCalibration(delayCeilingMultiple: 0.5).validate()
    try onlineRequire(ExpertSourceOnlineCalibration().delayCeilingMultiple == 1, "Default delay ceiling is not the forced maximum")
}

func expertSourceOnlineHedgeAndSelectionMatchHandValues() throws {
    let settings = ExpertSourceOnlineCalibration(floorMultiple: 0.25, ceilingMultiple: 4, smoothing: 0.5, warmupObservations: 1)
    let estimator = try ExpertSourceOnlineEstimator(calibration: onlineCurve, settings: settings, maximumConcurrent: 4)
    let base = 0.003880266
    // Cold: the hedge delay is the static base and does not count as differing.
    try onlineRequire(near(estimator.hedgeDelaySeconds(base: base), base), "Cold hedge delay differs from the static base")
    // External twice as slow as calibrated: hedge twice as soon. Internal twice as slow too: back to the base.
    estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.001, concurrency: 1)
    try onlineRequire(near(estimator.hedgeDelaySeconds(base: base), base / 2), "Slow external primary did not hedge sooner")
    estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 0.002, concurrency: 1)
    try onlineRequire(near(estimator.hedgeDelaySeconds(base: base), base), "Equal drift changed the hedge delay")
    // External at 3 (clamped from 100x, smoothed from 2) against internal at 2: scale 2/3.
    estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.05, concurrency: 1)
    try onlineRequire(
        near(estimator.multiple(for: .externalSSD), 3) && near(estimator.hedgeDelaySeconds(base: base), base * 2 / 3),
        "Clamped observation entered the hedge scale unclamped")
    // Internal at the floor (0.25) against external 3.75: scale 1/15, held at the delay floor 0.25 x base.
    estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.05, concurrency: 1)
    estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.05, concurrency: 1)
    for _ in 0..<40 { estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 1e-6, concurrency: 1) }
    try onlineRequire(
        near(estimator.multiple(for: .externalSSD), 3.75) && near(estimator.multiple(for: .internalSSD), 0.25, 1e-9)
            && near(estimator.hedgeDelaySeconds(base: base), base * 0.25, 1e-15), "Hedge delay crossed its floor")
    // Internal slow (4) against external fast (0.25): scale 16, but the delay never exceeds the static base.
    for _ in 0..<40 { estimator.observe(source: .internalSSD, bytes: onlineBytes, seconds: 1, concurrency: 1) }
    for _ in 0..<40 { estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 1e-6, concurrency: 1) }
    try onlineRequire(
        near(estimator.multiple(for: .internalSSD), 4, 1e-9) && near(estimator.multiple(for: .externalSSD), 0.25, 1e-9)
            && estimator.hedgeDelaySeconds(base: base) == base, "Hedge delay fired later than the static delay")
    // A slower internal replica alone (external at 1) still never delays the hedge.
    for _ in 0..<40 { estimator.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.0005, concurrency: 1) }
    try onlineRequire(
        near(estimator.multiple(for: .externalSSD), 1, 1e-9) && estimator.hedgeDelaySeconds(base: base) == base,
        "Slow internal replica delayed the hedge")
    var s = estimator.snapshot()
    try onlineRequire(
        s.hedgeDelays == 7 && s.hedgeDelaysDiffering == 3 && s.decisionsDiffering == 3 && s.lastHedgeDelaySeconds == base
            && near(s.minimumHedgeDelaySeconds, base * 0.25, 1e-15) && s.maximumHedgeDelaySeconds == base,
        "Hedge counters differ from the seven decisions made")
    // A lower configured delay ceiling binds the same way: internal 1, external 2 would give base/2; ceiling 0.4 x base holds at 0.4.
    let low = try ExpertSourceOnlineEstimator(
        calibration: onlineCurve,
        settings: ExpertSourceOnlineCalibration(
            floorMultiple: 0.25, ceilingMultiple: 4, smoothing: 0.5, warmupObservations: 1, delayCeilingMultiple: 0.4),
        maximumConcurrent: 4)
    try onlineRequire(low.hedgeDelaySeconds(base: base) == base * 0.4, "Configured delay ceiling was not applied cold")
    // External three times slower than calibrated: scale 1/3 is below the 0.4 ceiling and above the 0.25 floor.
    low.observe(source: .externalSSD, bytes: onlineBytes, seconds: 0.0015, concurrency: 1)
    try onlineRequire(near(low.hedgeDelaySeconds(base: base), base / 3), "Configured delay ceiling blocked an earlier hedge")

    // The static rule remains unchanged without an estimator. A loader with a warm external
    // slowdown consumes a frozen online curve and chooses the now-faster internal source.
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 1, reservedDemandConcurrent: 1, maximumOutstanding: 4,
        reservedDemandOutstanding: 4, maximumOutstandingBytes: 32_000_000, reservedDemandBytes: 32_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    defer { scheduler.shutdown() }
    let generation = scheduler.currentGeneration
    let plain = try scheduler.submitEarliest(
        availableSources: [.internalSSD, .externalSSD], estimate: onlineCurve.estimatedSeconds,
        priority: .demand, generation: generation, byteCount: onlineBytes
    ) { _, _ in }
    try onlineRequire(plain.source == .externalSSD && plain.join() == .succeeded(bytes: onlineBytes), "Static placement changed")
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let cacheLedger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let cache = try ExpertCache(device: device, ledger: cacheLedger, byteLimit: UInt64(2 * PackStore.expertBytes), slotsPerSlab: 2)
    let trained = try ExpertSourceOnlineEstimator(calibration: onlineCurve, settings: settings, maximumConcurrent: 1)
    for _ in 0..<40 { trained.observe(source: .externalSSD, bytes: PackStore.expertBytes, seconds: 1, concurrency: 1) }
    trained.observe(source: .internalSSD, bytes: PackStore.expertBytes, seconds: Double(PackStore.expertBytes) / 1e9, concurrency: 1)
    try onlineRequire(near(trained.multiple(for: .externalSSD), 4, 1e-9), "Fixture estimator did not reach the external ceiling")
    // Consume the first idle-probe hint so this request proves curve-based placement, rather than probing.
    try onlineRequire(trained.idleProbeSource(hotInternal: false) == .internalSSD, "Initial idle-path probe differs")
    let reads = OnlineReads()
    let loader = SourceExpertLoader(
        cache: cache, scheduler: scheduler, replicaArena: nil, estimator: trained,
        packedAvailable: { true }
    ) { _, source, destination in
        reads.record(source)
        destination.initializeMemory(as: UInt8.self, repeating: 0x5a)
    }
    let request = try loader.request(
        ExpertKey(layer: 0, expert: 0), generation: generation, policy: .earliestCompletion, calibration: onlineCurve)
    let loaded = try loader.joinAndPin(request)
    try onlineRequire(
        request.source == .internalSSD && reads.sources == [.internalSSD] && loaded.buffer.bytes[0] == 0x5a,
        "Online placement did not use the faster observed exact source")
    try loaded.retireAfterGPUCompletion()
    s = trained.snapshot()
    try onlineRequire(
        s.externalSSD.observations == 40 && s.internalSSD.observations == 2 && s.hedgeDelays == 0
            && s.placementDecisions == 1 && s.placementDecisionsDiffering == 1,
        "Placement observation/curve counters differ, or a hedge was counted without a replica")
    try onlineRequire(
        scheduler.snapshot().drives.reduce(UInt64(0)) { $0 + $1.successfulBytes } == UInt64(onlineBytes + PackStore.expertBytes),
        "Placement duplicated physical read bytes")
}

func expertSourceOnlineRuntimeConfigurationAndLoaderWiring() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-online-config-" + UUID().uuidString)
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
    let packed: [String: Any] = ["expert_mode": "slabPackedPreferred"]
    let legacyDefaults = try load([:])
    let packedDefaults = try load(packed)
    try onlineRequire(
        legacyDefaults.onlineCalibration == nil && packedDefaults.onlineCalibration == nil
            && ExpertExecutionConfiguration().onlineCalibration == nil, "Online calibration was on by default")
    var off = packed
    off["expert_source_online_calibration"] = false
    try onlineRequire(try load(off).onlineCalibration == nil, "A false flag enabled the online path")
    var on = packed
    on["expert_source_online_calibration"] = true
    try onlineRequire(try load(on).onlineCalibration == ExpertSourceOnlineCalibration(), "The flag did not select the default bounds")
    var bounded = on
    bounded["expert_source_online_bounds"] = ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16]
    try onlineRequire(
        try load(bounded).onlineCalibration
            == ExpertSourceOnlineCalibration(
                floorMultiple: 0.5, ceilingMultiple: 3,
                smoothing: 0.1, warmupObservations: 16), "Explicit bounds were lost (an absent delay ceiling is the forced 1)")
    var lowered = on
    lowered["expert_source_online_bounds"] = [
        "floor_multiple": 0.25, "ceiling_multiple": 4, "smoothing": 0.02, "warmup_observations": 32, "delay_ceiling_multiple": 1.0,
    ]
    try onlineRequire(
        try load(lowered).onlineCalibration == ExpertSourceOnlineCalibration(), "An explicit delay ceiling of 1 differs from the default")
    lowered["expert_source_online_bounds"] = [
        "floor_multiple": 0.25, "ceiling_multiple": 4, "smoothing": 0.02, "warmup_observations": 32, "delay_ceiling_multiple": 0.5,
    ]
    try onlineRequire(try load(lowered).onlineCalibration?.delayCeilingMultiple == 0.5, "A lower delay ceiling was lost")
    for invalid in [1, "true", NSNull()] as [Any] {
        var changed = packed
        changed["expert_source_online_calibration"] = invalid
        try onlineReject { _ = try load(changed) }
    }
    var withoutFlag = packed
    withoutFlag["expert_source_online_bounds"] = bounded["expert_source_online_bounds"]
    try onlineReject { _ = try load(withoutFlag) }
    for bad in [
        ["floor_multiple": 2, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16],
        ["floor_multiple": 0.5, "ceiling_multiple": 0.5, "smoothing": 0.1, "warmup_observations": 16],
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0, "warmup_observations": 16],
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 0],
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1],
        // The hedge may not fire later than the static delay: a delay ceiling above 1 is refused.
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16, "delay_ceiling_multiple": 1.5],
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16, "delay_ceiling_multiple": 0.25],
        ["floor_multiple": 0.5, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16, "delay_ceiling_multiple": "1"],
        ["floor_multiple": "0.5", "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16],
        ["floor_multiple": true, "ceiling_multiple": 3, "smoothing": 0.1, "warmup_observations": 16],
    ] as [[String: Any]] {
        var changed = on
        changed["expert_source_online_bounds"] = bad
        try onlineReject { _ = try load(changed) }
    }
    var legacyMode = on
    legacyMode["expert_mode"] = "legacyOriginal"
    try onlineReject { _ = try load(legacyMode) }
    let loaded = try load(bounded)
    var old = try JSONSerialization.jsonObject(with: JSONEncoder().encode(loaded)) as! [String: Any]
    try onlineRequire(
        (old["onlineCalibration"] as? [String: Any])?["warmup_observations"] as? Int == 16, "Snapshot omitted the online bounds")
    old.removeValue(forKey: "onlineCalibration")
    let restored = try JSONDecoder().decode(ExpertExecutionConfiguration.self, from: JSONSerialization.data(withJSONObject: old))
    try onlineRequire(restored.onlineCalibration == nil, "Old execution snapshot no longer decodes with the online path off")

    // Loader wiring: external, original-span and hot-mirror reads have separate curves;
    // every placement publishes the same complete bytes. Static/off retains the original mirror preference.
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal") }
    let expected = (0..<PackStore.expertBytes).map { UInt8(truncatingIfNeeded: ($0 / 251) ^ $0 ^ 0x6e) }
    for online in [false, true] {
        let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
        let reads = OnlineReads()
        let configuration = ExpertExecutionConfiguration(
            mode: .slabPackedPreferred, cacheSlots: 4, slotsPerSlab: 1,
            demandWindow: 1, readConcurrency: 1, prefillSourcePolicy: .earliestCompletion, decodeSourcePolicy: .earliestCompletion,
            sourceCalibration: onlineCurve,
            onlineCalibration: online
                ? ExpertSourceOnlineCalibration(floorMultiple: 0.25, ceilingMultiple: 4, smoothing: 0.5, warmupObservations: 1) : nil)
        let engine = try ExpertExecution(
            device: device, ledger: ledger, configuration: configuration,
            hotInternal: { $0.expert == 7 }
        ) { _, source, destination in
            reads.record(source)
            expected.withUnsafeBytes { destination.copyMemory(from: $0) }
        }
        try onlineRequire(
            (engine.snapshot().expertSourceOnline != nil) == online, "Online snapshot presence differs from the configuration")
        try engine.beginSequence()
        // One record per window keeps the two drives' reads ordered for the source list below.
        let hot = try engine.makeWindow(layer: 0, experts: [7])
        try hot.consume(expert: 7) { buffer in try onlineRequire(Array(buffer.bytes) == expected, "Hot record bytes differ") }
        let placed = try engine.makeWindow(layer: 0, experts: [8])
        try placed.consume(expert: 8) { buffer in try onlineRequire(Array(buffer.bytes) == expected, "Placed record bytes differ") }
        try engine.finishSequence()
        try onlineRequire(
            reads.sources == [.internalSSD, .externalSSD],
            "Hot probe/static preference or cold original curve choice changed: \(reads.sources)")
        let s = engine.snapshot()
        if online {
            guard let counters = s.expertSourceOnline else { throw EngineError("Online counters are missing") }
            try onlineRequire(
                counters.hotInternalReadsSkipped == 0 && counters.hotMirror.observations == 1 && counters.internalSSD.observations == 0
                    && counters.externalSSD.observations == 1 && counters.observations == 2
                    && counters.placementDecisions == 2
                    && counters.hedgeDelays == 0 && counters.decisionsDiffering == 0,
                "Loader observations differ from separate mirror/original/packed curves")
        } else {
            try onlineRequire(
                s.expertSourceOnline == nil && engine.onlineCalibrationSnapshot == nil, "Static configuration reported online counters")
        }
    }
}
