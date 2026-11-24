import Foundation
import OrigamiEngine

/// The slow-drive signal the status and `origami doctor` report (`SlowDriveReads`, ROADMAP R15).
func slowDriveReadsChecks() -> [(String, () throws -> Void)] {
    [
        ("slow_drive_reads_share", slowDriveReadsShare), ("drive_read_history", driveReadHistory),
        ("unreadable_state_files", unreadableStateFiles),
        ("slow_drive_reads_floor_keeps_slow", slowDriveReadsFloorKeepsSlow),
        ("slow_drive_reads_floor_follows_measurement", slowDriveReadsFloorFollowsMeasurement),
    ]
}

/// Five minutes of reads at a tenth of the drive's saved measurement, in a process that starts with them: every minute
/// of it counts as slow (without the floor the healthy rate decays to the slow one within minutes, and starts at zero).
private func slowDriveReadsFloorKeepsSlow() throws {
    var now = 0.0
    let reads = SlowDriveReads { now }
    reads.floorHealthy(1e9)
    for _ in 0..<6000 {
        now += 0.05
        reads.add(bytes: 1e8 * 0.05, seconds: 0.05)
    }
    let snapshot = reads.snapshot()
    try slowRequire(
        snapshot.slowSeconds > 0.95 * snapshot.busySeconds && snapshot.high, "a sustained slowdown stopped counting: \(snapshot)")
}

/// The floor follows the drive's saved measurement through `DriveReadLog.record`: set from it, and cleared once it is
/// dropped (`DriveSpeeds.forget`, as `DriveSpeedReview` does), so the healthy rate then decays to the live one.
private func slowDriveReadsFloorFollowsMeasurement() throws {
    let data = FileManager.default.temporaryDirectory.appendingPathComponent("origami-floor-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: data) }
    try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
    guard let key = DriveSpeeds.key(data) else { throw FixtureUnavailable(description: "no volume UUID for the temporary folder") }
    let speeds = data.appendingPathComponent("drive-speeds.json")
    let speed = DriveSpeed(
        queueDepths: [1], bytesPerSecond: [1e9], recordBytes: 1 << 20, measuredAt: Date().timeIntervalSince1970, name: "check")
    try JSONEncoder().encode([key: speed]).write(to: speeds)
    var now = 0.0
    let reads = SlowDriveReads { now }
    func read(_ steps: Int) {
        for _ in 0..<steps {
            now += 0.05
            reads.add(bytes: 0.5e9 * 0.05, seconds: 0.05)
        }
    }
    let log = DriveReadLog(data: data)
    read(20)
    log.record(data, reads: reads)
    try slowRequire(reads.rates.healthy == 1e9, "not floored at the measurement: \(reads.rates)")
    DriveSpeeds.forget([key], in: speeds)
    read(20)
    log.record(data, reads: reads)
    read(2000)
    try slowRequire(reads.rates.healthy < 0.7e9, "a dropped measurement still floors the healthy rate: \(reads.rates)")
}

/// A state file Origami writes by itself that cannot be read (`drive-reads.json`, `memory-history.json`) is set aside
/// with its bytes, never written over; a readable one is kept.
private func unreadableStateFiles() throws {
    let data = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: data) }
    try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
    let garbage = Data("{not json".utf8)
    func aside(_ name: String) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: data, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix(name + ".unreadable-") }
    }
    let reads = DriveReadLog.file(data)
    try garbage.write(to: reads)
    DriveReadLog.rewritten(data, data: data)
    try slowRequire(
        aside("drive-reads.json").count == 1 && (try? Data(contentsOf: aside("drive-reads.json")[0])) == garbage,
        "an unreadable drive-reads.json was not kept aside")
    try slowRequire(DriveReadLog.read(reads)[DriveReadLog.key(data)]?.rewritten == true, "no new drive-reads.json after it")
    DriveReadLog.rewritten(data, data: data)
    try slowRequire(aside("drive-reads.json").count == 1, "a readable drive-reads.json was set aside")
    let history = data.appendingPathComponent("memory-history.json")
    try garbage.write(to: history)
    let memory = MemoryHistory(file: history, identity: "check")
    memory.trimmed(to: 1_000_000_000, limit: 2_000_000_000)
    try slowRequire(
        aside("memory-history.json").count == 1 && (try? Data(contentsOf: aside("memory-history.json")[0])) == garbage,
        "an unreadable memory-history.json was not kept aside")
    try slowRequire(FileManager.default.fileExists(atPath: history.path), "no new memory-history.json after it")
}

private func slowRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw EngineError("slow drive reads: " + message) }
}

/// Healthy reads keep the share at zero (rates within a factor of two); stalled reads (a tenth of the rate) raise it, in
/// both the busy-span and the overlapping-read forms, and over the threshold the advice names `origami refresh`.
private func slowDriveReadsShare() throws {
    var now = 0.0
    let healthy = SlowDriveReads { now }
    for step in 0..<2000 {
        now += 0.05
        healthy.add(bytes: (step % 2 == 0 ? 1.0 : 0.6) * 2e9 * 0.05, seconds: 0.05)
    }
    let calm = healthy.snapshot()
    try slowRequire(calm.slowSeconds == 0 && calm.busySeconds > 99 && !calm.high, "healthy reads counted slow: \(calm)")
    try slowRequire(SlowDriveReads.advice(calm.report, modelBytes: 1 << 30) == nil, "advice for a healthy drive")
    for step in 0..<2000 {
        now += 0.05
        let slow = step % 400 >= 200  // 10 s slow, 10 s healthy
        healthy.add(bytes: (slow ? 0.1 : 1.0) * 2e9 * 0.05, seconds: 0.05)
    }
    let sick = healthy.snapshot()
    try slowRequire(sick.slowSeconds > 40 && sick.hourShare > SlowDriveReads.threshold && sick.high, "stalled reads not counted: \(sick)")
    // Too little read to blame the drive: no advice yet. With a terabyte behind it, the advice names `origami refresh`.
    try slowRequire(SlowDriveReads.advice(sick.report, modelBytes: 1 << 30) == nil, "advice before enough was read: \(sick)")
    for step in 0..<16000 {
        now += 0.05
        healthy.add(bytes: (step % 400 >= 200 ? 0.1 : 1.0) * 2e9 * 0.05, seconds: 0.05)
    }
    let worn = healthy.snapshot()
    try slowRequire(
        worn.bytes >= SlowDriveReads.settledBytes && worn.high
            && SlowDriveReads.advice(worn.report, modelBytes: 1 << 30)?.contains("origami refresh") == true, "no advice: \(worn)")
    // Overlapping single reads: two at a time, healthy then a tenth as fast.
    let reads = SlowDriveReads { now }
    for step in 0..<4000 {
        let slow = step >= 3000
        reads.begin()
        reads.begin()
        now += slow ? 0.1 : 0.01
        reads.end(bytes: 16e6)
        reads.end(bytes: 16e6)
    }
    let overlapped = reads.snapshot()
    try slowRequire(overlapped.slowSeconds > 0.9 * 100 && overlapped.busySeconds < 131, "overlapping reads mismeasured: \(overlapped)")
}

/// The drive history (`DriveReadLog`): what each folder's counters read adds up per folder and survives a restart (a new
/// log and new counters on the same file); a rewrite starts the count again and keeps the bytes the slowdown began
/// after, and the advice comes once 80% of those are read again; a hot copy's new generation counts as a rewrite.
private func driveReadHistory() throws {
    let data = FileManager.default.temporaryDirectory.appendingPathComponent("origami-drive-reads-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: data) }
    let model = data.appendingPathComponent("model")
    let hot = data.appendingPathComponent("hot")
    var now = 0.0
    // `slow`: a tenth of the healthy rate every other 10 s, as the worn drive read.
    func read(_ reads: SlowDriveReads, seconds: Double, slow: Bool) {
        for step in 0..<Int(seconds / 0.05) {
            now += 0.05
            reads.add(bytes: (slow && step % 400 >= 200 ? 0.1 : 1.0) * 1e9 * 0.05, seconds: 0.05)
        }
    }
    func history(_ folder: URL) throws -> DriveReadHistory {
        guard let found = DriveReadLog.history(folder, data: data) else { throw EngineError("drive reads: no history") }
        return found
    }
    let day = Date(timeIntervalSince1970: 1_800_000_000)
    var log = DriveReadLog(data: data)
    var counter = SlowDriveReads { now }
    read(counter, seconds: 100, slow: false)
    log.record(model, reads: counter, now: day)
    read(counter, seconds: 100, slow: false)
    log.record(model, reads: counter, now: day)
    let copy = SlowDriveReads { now }
    read(copy, seconds: 10, slow: false)
    log.record(hot, reads: copy, generation: "a@1", now: day)
    var entry = try history(model)
    try slowRequire(abs(entry.bytes - 200e9) < 1e9 && entry.slowSeconds == 0 && !entry.high, "counted wrong: \(entry)")
    try slowRequire(
        entry.text() == "200 GB read since Origami started counting on 2027-01-15; slow 0% of the reading time.", "text: \(entry.text())")
    try slowRequire(abs(try history(hot).bytes - 10e9) < 1e8, "hot copy not kept apart")
    // A restart: a new log and counter add to the saved count; the drive slows down and the slow share goes high.
    log = DriveReadLog(data: data)
    counter = SlowDriveReads { now }
    read(counter, seconds: 400, slow: true)
    log.record(model, reads: counter, now: day)
    entry = try history(model)
    try slowRequire(
        entry.bytes > 300e9 && entry.high && entry.slowAfter == entry.bytes && entry.text().contains("origami refresh"), "\(entry)")
    // Under a terabyte and no earlier slowdown: the text says so instead of calling the drive slow.
    try slowRequire(!entry.settled && entry.text().contains("too little read yet"), "a first hour called the drive slow: \(entry.text())")
    // `origami refresh`: from zero, the last slowdown's bytes kept; quiet below 80% of them, advice from there on.
    let slowAfter = entry.slowAfter!
    DriveReadLog.rewritten(model, data: data, now: day)
    log = DriveReadLog(data: data)
    counter = SlowDriveReads { now }
    entry = try history(model)
    try slowRequire(entry.bytes == 0 && entry.rewritten && entry.earlier == [slowAfter] && !entry.due, "refresh did not reset: \(entry)")
    read(counter, seconds: 0.7 * slowAfter / 1e9, slow: false)
    log.record(model, reads: counter, now: day)
    entry = try history(model)
    try slowRequire(!entry.due && entry.text().contains("Last time the slowdown began after about"), "early advice: \(entry.text())")
    read(counter, seconds: 0.15 * slowAfter / 1e9, slow: false)
    log.record(model, reads: counter, now: day)
    entry = try history(model)
    try slowRequire(
        entry.due && entry.text().contains("likely soon") && entry.text().contains("origami refresh"), "no advice: \(entry.text())")
    // A hot copy rebuilt (a new payload): counted from zero, its advice `refresh --copy`.
    let rebuilt = SlowDriveReads { now }
    read(rebuilt, seconds: 1, slow: false)
    log.record(hot, reads: rebuilt, generation: "b@2", now: day)
    entry = try history(hot)
    try slowRequire(
        abs(entry.bytes - 1e9) < 1e8 && entry.rewritten && entry.text(hot: true).contains("since the copy was written"), "\(entry)")
}
