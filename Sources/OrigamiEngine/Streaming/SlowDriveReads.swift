import Foundation

/// How much of the model drive's reading time went far below its own healthy rate (ROADMAP R15): the sign of a drive
/// that holds heavily read blocks at a fraction of its speed, which rewriting the model files in place (`origami
/// refresh`) cures. Both engines feed it the model files' reads: busy spans with their bytes (`add`), or single reads
/// that may overlap (`begin`/`end`, turned into busy wall time). Machine-independent: every rate is the drive's own.
///
/// The current rate is bytes over busy seconds, both decaying with a one-second time constant (as Family's hedge rule
/// measures a drive). The healthy rate is the current rate's running maximum, decaying toward it with a one-minute time
/// constant, so a slow period of seconds does not lower it. A span is slow while the current rate is below a third of
/// the healthy rate: a healthy drive's short-window rate moves within about a factor of two with queue depth and
/// record size, so a third is outside that, and the hedge rule calls the same thing a stall.
package final class SlowDriveReads: @unchecked Sendable {
    /// More than this share of busy time slow, over at least `minimumBusySeconds`, is reported as a slow drive: a healthy
    /// drive stays near zero, the development Mac's worn SSD showed 40–70%, and 20% of the time at a third of the speed
    /// or less already makes reading at least 40% longer.
    package static let threshold = 0.2
    package static let minimumBusySeconds = 60.0
    /// Slow reading calls for a rewrite only with this much read behind it (or an earlier slowdown on record for the
    /// files): before a terabyte, something else on the drive is more likely (the first hot copy being written there,
    /// another app, an enclosure that throttles), and a first hour of use should not end in rewriting the model.
    package static let settledBytes = 1e12
    /// The current rate must hold this much busy time before it counts: a few fast reads alone set no healthy rate.
    private static let settledSeconds = 0.1

    private let lock = NSLock()
    private let clock: () -> Double
    private var window = (bytes: 0.0, seconds: 0.0, at: 0.0)
    private var healthy = (rate: 0.0, at: 0.0)
    /// The drive's saved one-reader measurement (`floorHealthy`): the healthy rate never decays below it, so a sustained
    /// slowdown keeps counting as slow, and a new process does not start from zero.
    private var healthyFloor = 0.0
    private var busy = 0.0, slow = 0.0, total = 0.0
    /// The last hour per minute.
    private var minutes: [(minute: Int, busy: Double, slow: Double)] = []
    private var running = 0, mark = 0.0, pendingSeconds = 0.0
    /// Set from the files' read history (`DriveReadLog.record`): `since_rewrite` says high or due.
    private var historyWear = false

    /// Whether these files are being worn into their slow state: the history says high or due, or the live share is
    /// high. Split reads put no part on such a drive.
    package var wearing: Bool { lock.withLock { historyWear } || snapshot().high }
    package func markWear(_ on: Bool) { lock.withLock { historyWear = on } }
    /// Sets the healthy rate's floor: the drive's saved measurement (`DriveSpeed.rate(depth: 1)`); 0 clears it (the
    /// measurement was dropped or is stale).
    package func floorHealthy(_ rate: Double) {
        let rate = rate.isFinite ? max(0, rate) : 0
        lock.withLock {
            healthyFloor = rate
            healthy.rate = max(healthy.rate, rate)
        }
    }
    /// The current rate (bytes over busy seconds, one-second decay) and the healthy rate: the drive's own "slow" test,
    /// current below a third of healthy. Zero until measured.
    package var rates: (current: Double, healthy: Double) {
        lock.withLock { (window.seconds > 0 ? window.bytes / window.seconds : 0, healthy.rate) }
    }

    /// `clock`: seconds, monotonic (checks pass their own).
    package init(clock: @escaping () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }) { self.clock = clock }

    /// `bytes` read over `seconds` of busy time ending now.
    package func add(bytes: Double, seconds: Double) {
        guard seconds > 0, seconds.isFinite, bytes >= 0 else { return }
        lock.withLock { addLocked(bytes: bytes, seconds: seconds, now: clock()) }
    }
    /// A single read starts (reads may overlap: busy time is wall time with at least one running).
    package func begin() {
        lock.withLock {
            let now = clock()
            if running > 0 { pendingSeconds += now - mark }
            mark = now
            running += 1
        }
    }
    /// A read that `begin` started ended with `bytes`.
    package func end(bytes: Double) {
        lock.withLock {
            let now = clock()
            guard running > 0 else { return }
            pendingSeconds += now - mark
            mark = now
            running -= 1
            let seconds = pendingSeconds
            pendingSeconds = 0
            if seconds > 0 {
                addLocked(bytes: bytes, seconds: seconds, now: now)
            } else {
                window.bytes += bytes
                total += bytes
            }
        }
    }

    private func addLocked(bytes: Double, seconds: Double, now: Double) {
        let keep = exp(-max(0, now - window.at))
        window = (window.bytes * keep + bytes, window.seconds * keep + seconds, now)
        let current = window.bytes / window.seconds
        let settled = window.seconds >= Self.settledSeconds
        let stalled = settled && healthy.rate > 0 && current < healthy.rate / 3
        if settled {
            let decay = exp(-max(0, now - healthy.at) / 60)
            healthy = (max(healthyFloor, current, current + (healthy.rate - current) * decay), now)
        }
        busy += seconds
        total += bytes
        if stalled { slow += seconds }
        let minute = Int(now / 60)
        if minutes.last?.minute == minute {
            minutes[minutes.count - 1].busy += seconds
            if stalled { minutes[minutes.count - 1].slow += seconds }
        } else {
            minutes.append((minute, seconds, stalled ? seconds : 0))
        }
        minutes.removeAll { $0.minute <= minute - 60 }
    }

    package struct Snapshot: Equatable, Sendable {
        package let busySeconds, slowSeconds, hourBusySeconds, hourSlowSeconds, healthyBytesPerSecond: Double
        /// Bytes read since start.
        package var bytes = 0.0
        package var share: Double { busySeconds > 0 ? slowSeconds / busySeconds : 0 }
        package var hourShare: Double { hourBusySeconds > 0 ? hourSlowSeconds / hourBusySeconds : 0 }
        /// Over the threshold, since start or in the last hour, with enough reading behind it.
        package var high: Bool {
            (busySeconds >= SlowDriveReads.minimumBusySeconds && share > SlowDriveReads.threshold)
                || (hourBusySeconds >= SlowDriveReads.minimumBusySeconds && hourShare > SlowDriveReads.threshold)
        }
        /// `slow_reads` in `/v1/status`.
        package var report: [String: Any] {
            [
                "busy_seconds": busySeconds, "slow_seconds": slowSeconds, "share": share, "hour_busy_seconds": hourBusySeconds,
                "hour_slow_seconds": hourSlowSeconds, "hour_share": hourShare, "healthy_bytes_per_second": healthyBytesPerSecond,
                "threshold": SlowDriveReads.threshold, "high": high, "bytes": bytes,
            ]
        }
    }
    package func snapshot() -> Snapshot {
        lock.withLock {
            let minute = Int(clock() / 60)
            let hour = minutes.filter { $0.minute > minute - 60 }
            return Snapshot(
                busySeconds: busy, slowSeconds: slow, hourBusySeconds: hour.reduce(0) { $0 + $1.busy },
                hourSlowSeconds: hour.reduce(0) { $0 + $1.slow }, healthyBytesPerSecond: healthy.rate, bytes: total)
        }
    }

    /// What the status and `origami doctor` say for a `slow_reads` report that is high (nil otherwise): the share, and
    /// what `origami refresh` does and about how long it takes for `modelBytes` (it reads each file twice and writes it
    /// once; the drive's healthy read rate stands in for its write rate, so the time is a lower bound).
    /// `experts`: a low-bit variant's experts file and its drive, which `origami refresh` of the variant rewrites too.
    package static func advice(_ report: [String: Any], modelBytes: UInt64, experts: String? = nil) -> String? {
        guard report["high"] as? Bool == true, (report["bytes"] as? Double ?? 0) >= settledBytes else { return nil }
        let share = max(report["share"] as? Double ?? 0, report["hour_share"] as? Double ?? 0)
        let rate = report["healthy_bytes_per_second"] as? Double ?? 0
        let minutes = rate > 0 ? Int((3 * Double(modelBytes) / rate / 60).rounded(.up)) : nil
        return (experts.map { "The drives of the model and of its low-bit experts file (\($0))" } ?? "The model's drive")
            + " read its files below a third of its own speed for \(Int((100 * share).rounded()))% of its reading time. "
            + "Rewriting them in place usually cures this: stop the server and run `origami refresh`"
            + (minutes.map { " (at least \($0) min for \(OrigamiPlan.gigabytes(modelBytes)))" } ?? "") + "."
    }
}

/// What Origami read from one set of files (a model's weight files, or a hot copy) since they were last written, by the
/// files' folder, in `<data_dir>/drive-reads.json` (ROADMAP R15): bytes, and busy and slow seconds as `SlowDriveReads`
/// counts them, kept across restarts. The bytes read when the slow share first went high are kept across rewrites, so
/// the status can say from this drive's own history when the next rewrite is due, before reading slows down.
package struct DriveReadHistory: Codable, Equatable, Sendable {
    /// The volume UUID of the files' drive.
    package var drive: String?
    /// The files' identity where a rebuild writes new ones (a hot copy's payload name and birth time): a change counts
    /// as a rewrite.
    package var generation: String?
    /// When counting started, or when the files were last rewritten (`rewritten`).
    package var since: Date
    package var rewritten = false
    package var bytes = 0.0, busySeconds = 0.0, slowSeconds = 0.0
    /// `bytes` when the slow share first went high since `since`.
    package var slowAfter: Double?
    /// `slowAfter` of earlier rewrites, oldest first.
    package var earlier: [Double] = []
    /// Whether the slow share was high at the last count.
    package var high = false

    package init(drive: String?, generation: String? = nil, since: Date, rewritten: Bool = false, earlier: [Double] = []) {
        self.drive = drive
        self.generation = generation
        self.since = since
        self.rewritten = rewritten
        self.earlier = earlier
    }

    /// Advice comes once this share of the bytes the last slowdown began after is read again.
    package static let ahead = 0.8
    package var share: Double { busySeconds > 0 ? slowSeconds / busySeconds : 0 }
    /// Enough is read, or a slowdown is on record, for a high share to mean the files need rewriting.
    package var settled: Bool { bytes >= SlowDriveReads.settledBytes || !earlier.isEmpty }
    /// Within `ahead` of the bytes the last slowdown began after, and not slow yet.
    package var due: Bool { !high && earlier.last.map { bytes >= Self.ahead * $0 } == true }

    /// The files were rewritten `at`: counting starts again; the bytes this round's slowdown began after are kept.
    package mutating func rewrite(at: Date, generation: String? = nil) {
        let kept = Array((earlier + [slowAfter].compactMap { $0 }).suffix(8))
        self = Self(drive: drive, generation: generation, since: at, rewritten: true, earlier: kept)
    }

    package static func amount(_ bytes: Double) -> String {
        bytes >= 1e12 ? String(format: "%.1f TB", bytes / 1e12) : String(format: "%.0f GB", bytes / 1e9)
    }

    /// One plain sentence for the status and `origami doctor`: what was read and how much of it slowly, then, from this
    /// drive's history, whether a rewrite is due. `hot`: a hot copy, cured by rebuilding it rather than a refresh.
    package func text(hot: Bool = false) -> String {
        let day = since.formatted(.iso8601.year().month().day())
        let start =
            !rewritten
            ? "since Origami started counting on \(day)"
            : hot ? "since the copy was written on \(day)" : "since its files were last rewritten on \(day)"
        var text = "\(Self.amount(bytes)) read \(start); slow \(Int((100 * share).rounded()))% of the reading time."
        if let last = earlier.last { text += " Last time the slowdown began after about \(Self.amount(last))." }
        let cure =
            hot
            ? "rebuild the copy (delete it in the dashboard's Models view, then `origami config unset hot_copy` and reload; the server writes it again on fresh blocks)"
            : "stop the server and run `origami refresh`"
        if high && settled {
            text += " The drive reads these files slowly now: \(cure)."
        } else if high {
            text +=
                " It reads them slowly at times, with too little read yet to blame the drive; if that lasts with nothing else using it, \(cure)."
        } else if due {
            text += " A slowdown is likely soon: \(cure)."
        }
        return text
    }

    /// The `since_rewrite` object of a status row.
    package func report(hot: Bool = false) -> [String: Any] {
        [
            "bytes": bytes, "busy_seconds": busySeconds, "slow_seconds": slowSeconds, "share": share,
            "since": since.formatted(.iso8601), "rewritten": rewritten, "last_slow_after_bytes": earlier.last.map { $0 as Any } ?? NSNull(),
            "high": high, "due": due, "text": text(hot: hot),
        ]
    }
}

/// `<data_dir>/drive-reads.json`: a `DriveReadHistory` per folder. `record` adds what a `SlowDriveReads` counted since
/// its last record (a counter not seen before, after a restart or for a new copy, from zero).
package final class DriveReadLog: @unchecked Sendable {
    package let file: URL
    private let lock = NSLock()
    private var last: [String: (counter: ObjectIdentifier, snapshot: SlowDriveReads.Snapshot)] = [:]
    package init(data: URL) { file = Self.file(data) }

    package static func file(_ data: URL) -> URL { data.appendingPathComponent("drive-reads.json") }
    /// `folder`'s key: its path with links resolved, so every caller names it alike.
    package static func key(_ folder: URL) -> String { folder.resolvingSymlinksInPath().standardizedFileURL.path }
    package static func read(_ file: URL) -> [String: DriveReadHistory] {
        (try? Data(contentsOf: file)).flatMap { try? decode($0) } ?? [:]
    }
    static func decode(_ data: Data) throws -> [String: DriveReadHistory] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([String: DriveReadHistory].self, from: data)
    }
    /// What a writer adds to: an unreadable file is set aside first (`StateFile`); nil when it cannot be.
    static func readForWrite(_ file: URL) -> [String: DriveReadHistory]? { StateFile.load(file, empty: [:], decode) }
    static func write(_ all: [String: DriveReadHistory], to file: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? encoder.encode(all).write(to: file, options: .atomic)
    }
    package static func history(_ folder: URL, data: URL) -> DriveReadHistory? { read(file(data))[key(folder)] }

    /// Adds what `reads` counted since its last record to `folder`'s history; `generation` as in `DriveReadHistory`.
    package func record(_ folder: URL, reads: SlowDriveReads, generation: String? = nil, now: Date = Date()) {
        let key = Self.key(folder)
        let snapshot = reads.snapshot()
        lock.withLock {
            let before = last[key].flatMap { $0.counter == ObjectIdentifier(reads) ? $0.snapshot : nil }
            last[key] = (ObjectIdentifier(reads), snapshot)
            let bytes = snapshot.bytes - (before?.bytes ?? 0)
            guard bytes > 0 else { return }
            guard var all = Self.readForWrite(file) else { return }
            var entry = all[key] ?? DriveReadHistory(drive: DriveSpeeds.key(folder), generation: generation, since: now)
            if entry.generation != generation { entry.rewrite(at: now, generation: generation) }
            entry.bytes += bytes
            entry.busySeconds += snapshot.busySeconds - (before?.busySeconds ?? 0)
            entry.slowSeconds += snapshot.slowSeconds - (before?.slowSeconds ?? 0)
            entry.high = snapshot.high
            if snapshot.high, entry.slowAfter == nil { entry.slowAfter = entry.bytes }
            reads.markWear(entry.high || entry.due)
            // The drive's saved measurement floors its healthy rate (`drive-speeds.json`, beside this file); none (dropped
            // as wrong, or stale): no floor.
            let speeds = DriveSpeeds.read(file.deletingLastPathComponent().appendingPathComponent("drive-speeds.json"))
            let speed = entry.drive.flatMap { speeds[$0] }.flatMap { $0.valid && !$0.stale() ? $0 : nil }
            reads.floorHealthy(speed?.rate(depth: 1) ?? 0)
            all[key] = entry
            Self.write(all, to: file)
        }
    }

    /// `origami refresh` rewrote the files in `folder`: their count starts again.
    package static func rewritten(_ folder: URL, data: URL, now: Date = Date()) {
        guard var all = readForWrite(file(data)) else { return }
        let key = key(folder)
        var entry = all[key] ?? DriveReadHistory(drive: DriveSpeeds.key(folder), since: now)
        entry.rewrite(at: now, generation: entry.generation)
        all[key] = entry
        write(all, to: file(data))
    }
}
