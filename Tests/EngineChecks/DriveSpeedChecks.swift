import Foundation
import OrigamiEngine

// Per-Mac drive measurements (`DriveSpeed`, `DriveSpeeds`, `DriveSpeedReview`) on fixture files in a temp folder, and
// the neutral curve the engine uses until a drive is measured.

private func driveRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError(message) }
}

/// Two 24 MB fixture files of distinct bytes in a fresh temp folder.
private func driveFixture() throws -> (folder: URL, files: [URL]) {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-drive-speed-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let files = try (0..<2).map { index -> URL in
        let url = folder.appendingPathComponent("record-\(index).bin")
        try Data(repeating: UInt8(index + 1), count: 24_000_000).write(to: url)
        return url
    }
    return (folder, files)
}

func driveSpeedMeasurement() throws {
    let (folder, files) = try driveFixture()
    defer { try? FileManager.default.removeItem(at: folder) }
    let started = Date()
    guard
        let speed = try DriveSpeed.measure(files: files, recordBytes: 1 << 20, name: "fixture", secondsPerPoint: 0.1, busy: { _ in false })
    else { throw EngineError("An idle fixture drive was not measured") }
    let elapsed = Date().timeIntervalSince(started)
    try driveRequire(
        speed.valid && speed.queueDepths == DriveSpeed.queueDepths && speed.recordBytes == 1 << 20
            && speed.bytesPerSecond.allSatisfy { $0 > 1e6 && $0.isFinite } && abs(speed.measuredAt - Date().timeIntervalSince1970) < 60,
        "Drive measurement is not a sane curve: \(speed)")
    // Three points of 0.1 s each, plus one read's tail and thread start-up.
    try driveRequire(elapsed < 3 * 0.1 + 1.5, "Drive measurement overran its bound: \(elapsed) s")
    try driveRequire(speed.rate(depth: 2) == speed.bytesPerSecond[0] && speed.rate(depth: 64) == speed.bytesPerSecond[2], "Depth lookup")
    // The default busy test reads the drive's byte counters: where macOS has them, the fixture's reads moved them.
    try driveRequire(DriveSpeed.driveBytes(folder).map { $0 > 0 } ?? true, "The drive's byte counters read as zero")
    // Busy before the first point, or between points (a request started): nothing is measured or kept.
    try driveRequire(
        try DriveSpeed.measure(files: files, recordBytes: 1 << 20, name: "fixture", secondsPerPoint: 0.1, busy: { _ in true }) == nil,
        "A busy drive was measured")
    var asked = 0
    let interrupted = try DriveSpeed.measure(
        files: files, recordBytes: 1 << 20, name: "fixture", secondsPerPoint: 0.1,
        busy: { _ in
            asked += 1
            return asked > 1
        })
    try driveRequire(interrupted == nil && asked == 2, "A measurement went on after a request began")
    // No file large enough holds a record: nothing to measure.
    try driveRequire(
        try DriveSpeed.measure(files: files, recordBytes: 32_000_000, name: "fixture", busy: { _ in false }) == nil,
        "Records larger than the files were measured")
}

func driveSpeedSavedReusedStale() throws {
    let (folder, files) = try driveFixture()
    defer { try? FileManager.default.removeItem(at: folder) }
    let saved = folder.appendingPathComponent("data/drive-speeds.json")
    guard let key = DriveSpeeds.key(folder) else { return }  // a volume without a UUID is never saved
    var lines: [String] = []
    func ensure(force: Bool = false, refusal: String? = nil) -> [String: DriveSpeed] {
        DriveSpeeds.ensure(
            saved, drives: [(folder, files, "fixture"), (files[0], [], "fixture")], recordBytes: 1 << 20, force: force,
            secondsPerPoint: 0.05, refusal: { refusal }, busy: { _ in false }, log: { lines.append($0) })
    }
    let first = ensure()
    try driveRequire(first.count == 1 && first[key]?.valid == true && DriveSpeeds.read(saved) == first, "First measurement not saved")
    try driveRequire(lines.count == 1, "Two entries on one drive were measured twice: \(lines)")
    // Reused while current: no new measurement.
    try driveRequire(ensure() == first && lines.count == 1, "A current measurement was taken again")
    // Forced (`origami setup`, `origami tune`): measured again.
    let forced = ensure(force: true)
    try driveRequire(lines.count == 2 && forced[key] != nil, "A forced measurement was skipped")
    // Stale (older than four weeks): measured again at the next start.
    let old = forced[key]!
    let aged = DriveSpeed(
        queueDepths: old.queueDepths, bytesPerSecond: old.bytesPerSecond, recordBytes: old.recordBytes,
        measuredAt: Date().timeIntervalSince1970 - DriveSpeed.maximumAge - 60, name: old.name)
    try JSONEncoder().encode([key: aged]).write(to: saved)
    try driveRequire(aged.stale() && !old.stale(), "Staleness rule")
    let renewed = ensure()
    try driveRequire(lines.count == 3 && renewed[key].map { !$0.stale() } == true, "A stale measurement was not renewed")
    // On battery or low power: skipped politely, the old one kept.
    try JSONEncoder().encode([key: aged]).write(to: saved)
    let refused = ensure(refusal: "the Mac is on battery")
    try driveRequire(
        refused[key] == aged && lines.last?.contains("on battery") == true && DriveSpeeds.read(saved)[key] == aged,
        "A refused measurement replaced the saved one")
    // Contradicted (serving's reads disagree for ten minutes): forgotten, so the next start measures again.
    DriveSpeeds.forget([key], in: saved)
    try driveRequire(DriveSpeeds.read(saved)[key] == nil, "A contradicted measurement was kept")
    let count = lines.count
    try driveRequire(ensure()[key] != nil && lines.count == count + 1, "A forgotten drive was not measured again")
    // Moved to another port or enclosure (J90): measured again; on the same connection, reused.
    var port = "USB External | bus-a"
    func plugged() -> [String: DriveSpeed] {
        DriveSpeeds.ensure(
            saved, drives: [(folder, files, "fixture")], recordBytes: 1 << 20, secondsPerPoint: 0.05, refusal: { nil },
            busy: { _ in false },
            connection: { _ in port }, log: { lines.append($0) })
    }
    let before = lines.count
    try driveRequire(plugged()[key]?.connection == port && lines.count == before + 1, "A measurement without its connection was reused")
    try driveRequire(plugged()[key]?.connection == port && lines.count == before + 1, "The same connection was measured again")
    port = "USB External | bus-b"
    try driveRequire(
        plugged()[key]?.connection == port && lines.count == before + 2 && DriveSpeeds.read(saved)[key]?.connection == port,
        "A drive on another port kept its old measurement")
    // Serving's per-read rate against the measured range: one reader's rate down to the deepest depth's share.
    let range = DriveSpeed(queueDepths: [1, 4, 8], bytesPerSecond: [1e9, 3e9, 4e9], recordBytes: 1 << 20, measuredAt: 0, name: "x")
    try driveRequire(
        range.perReadRatio(0.6e9) == 1 && abs(range.perReadRatio(0.1e9) - 0.2) < 1e-9 && abs(range.perReadRatio(3e9) - 3) < 1e-9,
        "Per-read ratio")
    // A damaged file reads as unmeasured.
    try Data("{\"x\": 1".utf8).write(to: saved)
    try driveRequire(DriveSpeeds.read(saved).isEmpty, "A damaged file was read")

    let review = DriveSpeedReview()
    try driveRequire(
        !review.observe(drive: "a", ratio: 5, now: 0) && !review.observe(drive: "a", ratio: 4, now: 300), "Early contradiction")
    try driveRequire(!review.observe(drive: "a", ratio: 1, now: 400), "Agreement did not reset")
    try driveRequire(
        !review.observe(drive: "a", ratio: 0.2, now: 500) && !review.observe(drive: "a", ratio: 0.2, now: 1000), "Reset ignored")
    try driveRequire(review.observe(drive: "a", ratio: 0.2, now: 1100), "Ten minutes of disagreement not reported")
    try driveRequire(!review.observe(drive: "a", ratio: 0.2, now: 2000), "A contradiction was reported twice")
    try driveRequire(!review.observe(drive: "b", ratio: .nan, now: 0) && !review.observe(drive: "b", ratio: .nan, now: 700), "NaN counted")
}

/// With nothing measured the engine plans with the neutral curve (equal drives: reads split by outstanding bytes, ties to
/// the pack); with a measurement it follows the measured drives. Asserted through the scheduler's own drive choice.
func driveSpeedNeutralDefaultAndMeasuredChoice() throws {
    let folder = FileManager.default.temporaryDirectory
    try driveRequire(ExpertExecutionConfiguration().sourceCalibration.label == "neutral", "The default curve is not neutral")
    func planned(_ storage: OrigamiStorage) throws -> ExpertSourceCalibration {
        let execution =
            try OrigamiPlan.baseObject(
                context: 8192, storage: storage, promptCache: folder, allowFast: true, bfloatMatrices: true, overrides: [:])["execution"]
            as! [String: Any]
        return try JSONDecoder().decode(
            ExpertSourceCalibration.self,
            from: JSONSerialization.data(withJSONObject: execution["expert_source_calibration"] as! [String: Any]))
    }
    let twoDrives = OrigamiStorage(packFolder: folder, packSharesModelDrive: false)
    let neutral = try planned(twoDrives)
    try driveRequire(
        neutral.label == "neutral" && neutral.internalBytesPerSecond == neutral.externalBytesPerSecond, "Two unmeasured drives not neutral")
    let now = Date().timeIntervalSince1970
    let fast = DriveSpeed(queueDepths: [1, 4, 8], bytesPerSecond: [2e9, 5e9, 6e9], recordBytes: 1 << 20, measuredAt: now, name: "fast")
    let slow = DriveSpeed(queueDepths: [1, 4, 8], bytesPerSecond: [0.5e9, 1e9, 1e9], recordBytes: 1 << 20, measuredAt: now, name: "slow")
    var measuredStorage = twoDrives
    measuredStorage.calibration = ExpertSourceCalibration(model: fast, pack: slow, recordBytes: 1 << 20)
    let measured = try planned(measuredStorage)
    try measured.validate()
    try driveRequire(
        measured.internalBytesPerSecond == [2e9, 5e9, 6e9] && measured.externalBytesPerSecond == [0.5e9, 1e9, 1e9]
            && abs(measured.replicaDelaySeconds! - 1.33 * Double(1 << 20) / 0.5e9) < 1e-12,
        "Measured curves or hedge delay lost in the plan")
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1_000_000)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 4, reservedDemandConcurrent: 4, maximumOutstanding: 64, reservedDemandOutstanding: 64,
        maximumOutstandingBytes: 512_000_000, reservedDemandBytes: 512_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits)
    // `model`, `pack`: reads already queued on the model's drive and on the pack's.
    func choose(_ curve: ExpertSourceCalibration, model: Int, pack: Int) throws -> IOSource? {
        try scheduler.earliestSource(
            availableSources: [.internalSSD, .externalSSD],
            estimate: { curve.estimatedSeconds(source: $0, bytesIncludingNew: $1, jobsIncludingNew: $2, maximumConcurrent: $3) },
            ties: .externalSSD, priority: .demand, byteCount: 1 << 20,
            pendingBytes: [.internalSSD: UInt64(model << 20), .externalSSD: UInt64(pack << 20)],
            pendingJobs: [.internalSSD: model, .externalSSD: pack])
    }
    // Neutral: a tie goes to the pack, and the less loaded drive wins either way.
    try driveRequire(
        choose(neutral, model: 0, pack: 0) == .externalSSD && choose(neutral, model: 0, pack: 2) == .internalSSD
            && choose(neutral, model: 2, pack: 0) == .externalSSD, "Neutral curve favoured a drive")
    // Measured: the fast model drive takes reads the neutral curve would give the pack.
    try driveRequire(
        choose(measured, model: 0, pack: 0) == .internalSSD && choose(measured, model: 3, pack: 0) == .internalSSD, "Measured curve ignored"
    )
}
