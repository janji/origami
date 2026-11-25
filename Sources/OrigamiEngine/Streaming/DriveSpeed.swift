import Foundation
import IOKit
import IOKit.ps

/// One drive's uncached read rate at the queue depths the readers use, measured on this Mac with record-sized reads at
/// random aligned offsets of the model's (or a copy's) own files. It only steers which drive reads a record first and
/// when a hedge fires: every drive holds the same bytes, so no measurement can change output.
public struct DriveSpeed: Codable, Sendable, Equatable {
    /// One reader, the Qwen scheduler's four and the family engines' up to eight (a token's experts at once).
    public static let queueDepths = [1, 4, 8]
    /// A measurement older than this is taken again at the next start.
    public static let maximumAge: TimeInterval = 28 * 86_400

    public let queueDepths: [Int]
    public let bytesPerSecond: [Double]
    public let recordBytes: Int
    /// Seconds since 1970.
    public let measuredAt: Double
    /// The drive as status shows it.
    public let name: String
    /// How the drive was connected when measured (`connection(_:)`; nil: measured before 3.1 recorded it).
    public var connection: String?

    public init(queueDepths: [Int], bytesPerSecond: [Double], recordBytes: Int, measuredAt: Double, name: String, connection: String? = nil)
    {
        self.queueDepths = queueDepths
        self.bytesPerSecond = bytesPerSecond
        self.recordBytes = recordBytes
        self.measuredAt = measuredAt
        self.name = name
        self.connection = connection
    }
    enum CodingKeys: String, CodingKey {
        case queueDepths = "queue_depths"
        case bytesPerSecond = "bytes_per_second"
        case recordBytes = "record_bytes"
        case measuredAt = "measured_at"
        case name
        case connection
    }

    /// The rate at the deepest measured depth not above `depth`.
    public func rate(depth: Int) -> Double { bytesPerSecond[queueDepths.lastIndex { $0 <= max(1, depth) } ?? 0] }
    public var valid: Bool {
        !queueDepths.isEmpty && queueDepths.first == 1 && queueDepths == queueDepths.sorted() && bytesPerSecond.count == queueDepths.count
            && bytesPerSecond.allSatisfy { $0.isFinite && (1_000_000...64_000_000_000).contains($0) } && recordBytes > 0
            && measuredAt.isFinite
    }
    /// Serving's per-read rate of this drive (a record's bytes over one read's time) against the measurement: 1 inside
    /// the measured range, else how far outside it. A read's time grows with the reads beside it, so the range runs from
    /// the deepest measured depth's share of its rate (rate / depth) to one reader's rate.
    public func perReadRatio(_ rate: Double) -> Double {
        let shares = zip(queueDepths, bytesPerSecond).map { $1 / Double($0) }
        guard let low = shares.min(), let high = shares.max(), rate > 0 else { return 1 }
        return rate / min(max(rate, low), high)
    }
    /// Too old, or dated in the future (a clock that was wrong).
    public func stale(now: Date = Date()) -> Bool {
        let age = now.timeIntervalSince1970 - measuredAt
        return age > Self.maximumAge || age < -86_400
    }
    public var report: [String: Any] {
        [
            "drive": name, "measured_at": ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: measuredAt)),
            "record_bytes": recordBytes,
            "read_gbps": Dictionary(
                uniqueKeysWithValues: zip(queueDepths.map { "qd\($0)" }, bytesPerSecond.map { ($0 / 1e7).rounded() / 100 })),
        ]
    }

    /// Why no drive should be measured now (low power, on battery), or nil.
    public static func refusal() -> String? {
        if ProcessInfo.processInfo.isLowPowerModeEnabled { return "low power mode is on" }
        if let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
            let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String?, type == kIOPMBatteryPowerKey
        {
            return "the Mac is on battery"
        }
        return nil
    }

    /// The physical disk holding `url` ("disk3": APFS volumes of one container share it, as `Stores.volume` names it), or
    /// nil where statfs gives none; kept per folder for the life of the process, since a split read asks per read.
    /// shortcut: a drive plugged in again under another disk number keeps its old name, which only decides which of two
    /// sources is dropped (`SplitReads.onePerDisk`); look again if a wrong name ever costs more than that.
    package static func disk(_ url: URL) -> String? {
        let path = url.standardizedFileURL.path
        if let known = diskNames.lock.withLock({ diskNames.names[path] }) { return known }
        var info = statfs()
        var name: String?
        if statfs(path, &info) == 0 {
            let device = withUnsafeBytes(of: info.f_mntfromname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
            let volume = device.split(separator: "/").last.map(String.init) ?? device
            name = volume.hasPrefix("disk") ? "disk" + volume.dropFirst(4).prefix { $0.isNumber } : volume
        }
        diskNames.lock.withLock { diskNames.names[path] = name }
        return name
    }
    private final class DiskNames: @unchecked Sendable {
        let lock = NSLock()
        var names: [String: String?] = [:]
    }
    private static let diskNames = DiskNames()

    /// Bytes the drive holding `url` has read and written since it appeared (IOKit's block-storage statistics), or nil
    /// where macOS gives none (a network volume).
    package static func driveBytes(_ url: URL) -> UInt64? {
        var info = statfs()
        guard statfs(url.path, &info) == 0 else { return nil }
        let device = withUnsafeBytes(of: info.f_mntfromname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        guard let name = device.split(separator: "/").last, let matching = IOBSDNameMatching(kIOMainPortDefault, 0, String(name))
        else { return nil }
        // Up from the volume to the first entry counting bytes (an APFS volume's own statistics hold none): the
        // container scheme or the disk's block-storage driver.
        var entry = IOServiceGetMatchingService(kIOMainPortDefault, matching)
        while entry != 0 {
            let current = entry
            defer { IOObjectRelease(current) }
            if let statistics = IORegistryEntryCreateCFProperty(current, "Statistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any], let read = statistics["Bytes (Read)"] as? NSNumber
            {
                return read.uint64Value &+ ((statistics["Bytes (Write)"] as? NSNumber)?.uint64Value ?? 0)
            }
            var parent: io_registry_entry_t = 0
            guard IORegistryEntryGetParentEntry(current, kIOServicePlane, &parent) == KERN_SUCCESS else { return nil }
            entry = parent
        }
        return nil
    }
    /// How the drive holding `url` is connected, as macOS reports it: the interconnect and its location (internal or
    /// external), the registry path from the bus to the drive's controller (it names the port and the enclosure's
    /// bridges, not the volume) and the PCIe link status where there is one; nil where macOS gives none. A drive moved
    /// to another port or enclosure has another, so its measurement is taken again (`DriveSpeeds.ensure`).
    public static func connection(_ url: URL) -> String? {
        var info = statfs()
        guard statfs(url.path, &info) == 0 else { return nil }
        let device = withUnsafeBytes(of: info.f_mntfromname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        guard let name = device.split(separator: "/").last, let matching = IOBSDNameMatching(kIOMainPortDefault, 0, String(name))
        else { return nil }
        let entry = IOServiceGetMatchingService(kIOMainPortDefault, matching)
        guard entry != 0 else { return nil }
        defer { IOObjectRelease(entry) }
        let up = IOOptionBits(kIORegistryIterateRecursively | kIORegistryIterateParents)
        func find(_ key: String) -> Any? {
            IORegistryEntrySearchCFProperty(entry, kIOServicePlane, key as CFString, kCFAllocatorDefault, up)
        }
        // Copied, not into a 512-byte `io_string_t`: behind hubs, docks and chained enclosures the path is longer.
        guard let path = IORegistryEntryCopyPath(entry, kIOServicePlane)?.takeRetainedValue() as String? else { return nil }
        let bus = path.components(separatedBy: "/IOBlockStorageDriver").first ?? ""
        let link = (find("Protocol Characteristics") as? [String: Any]).map {
            [$0["Physical Interconnect"], $0["Physical Interconnect Location"]].map { $0.map { "\($0)" } ?? "" }.joined(separator: " ")
        }
        return [link ?? "", bus, (find("IOPCIExpressLinkStatus") as? NSNumber).map { "link \($0)" } ?? ""].joined(separator: " | ")
    }
    /// Whether something else moves more than 32 MB/s on `url`'s drive over `seconds` (nothing of ours runs meanwhile).
    public static func othersBusy(_ url: URL, seconds: Double = 0.2) -> Bool {
        guard let before = driveBytes(url) else { return false }
        Thread.sleep(forTimeInterval: seconds)
        guard let after = driveBytes(url) else { return false }
        return Double(after &- before) / seconds > 32e6
    }

    /// Measures one drive: for each depth, that many readers read `recordBytes` at random 16 KiB-aligned offsets of
    /// `files` (uncached) for `secondsPerPoint`. Nil, with nothing measured kept, when `busy` says the drive is in use
    /// before a point (by default: something moves data on it, another process or a request of ours). Bounded: about
    /// (`secondsPerPoint` + 0.2 s) per depth, plus one read's tail.
    public static func measure(
        files: [URL], recordBytes: Int, name: String, secondsPerPoint: Double = 0.75, depths: [Int] = queueDepths,
        busy: (URL) -> Bool = { DriveSpeed.othersBusy($0) }
    ) throws -> DriveSpeed? {
        let alignment = 16_384
        let opened = try files.map { try ExactFile(url: $0) }.filter { $0.identity.size >= UInt64(recordBytes) }
        guard let first = files.first, !opened.isEmpty, recordBytes > 0 else { return nil }
        var rates: [Double] = []
        for depth in depths {
            if busy(first) { return nil }
            let tally = Tally()
            let started = DispatchTime.now().uptimeNanoseconds
            let deadline = started + UInt64(secondsPerPoint * 1e9)
            DispatchQueue.concurrentPerform(iterations: depth) { _ in
                let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: recordBytes, alignment: alignment)
                defer { buffer.deallocate() }
                var random = SystemRandomNumberGenerator()
                while DispatchTime.now().uptimeNanoseconds < deadline {
                    let file = opened[Int.random(in: 0..<opened.count, using: &random)]
                    let room = Int64(file.identity.size) - Int64(recordBytes)
                    let offset = Int64.random(in: 0...room, using: &random) / Int64(alignment) * Int64(alignment)
                    do { try file.read(offset: offset, into: buffer) } catch {
                        tally.fail("\(error)")
                        return
                    }
                    tally.add(recordBytes)
                }
            }
            let (read, problem) = tally.result
            if let problem { throw EngineError("Drive measurement failed: \(problem)") }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
            rates.append(min(64e9, max(1e6, Double(read) / seconds)))
        }
        return DriveSpeed(
            queueDepths: depths, bytesPerSecond: rates, recordBytes: recordBytes, measuredAt: Date().timeIntervalSince1970, name: name)
    }
}

private final class Tally: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes = 0
    private var problem: String?
    func add(_ count: Int) { lock.withLock { bytes += count } }
    func fail(_ message: String) { lock.withLock { problem = problem ?? message } }
    var result: (Int, String?) { lock.withLock { (bytes, problem) } }
}

/// The measured drives, saved in one file of the data folder, keyed by volume UUID (so a renamed mount or moved folder
/// keeps its measurement). A damaged file or entry reads as unmeasured.
public enum DriveSpeeds {
    /// The volume UUID of the drive holding `url` (or its nearest existing parent), nil where macOS has none.
    public static func key(_ url: URL) -> String? {
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path) && probe.path != "/" { probe = probe.deletingLastPathComponent() }
        return (try? probe.resourceValues(forKeys: [.volumeUUIDStringKey]))?.volumeUUIDString
    }
    public static func read(_ file: URL) -> [String: DriveSpeed] {
        ((try? Data(contentsOf: file)).flatMap { try? JSONDecoder().decode([String: DriveSpeed].self, from: $0) } ?? [:]).filter(
            \.value.valid)
    }
    static func write(_ all: [String: DriveSpeed], to file: URL) throws {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(all).write(to: file, options: .atomic)
    }
    /// Drops the saved measurement of `keys`: the next start measures them again.
    public static func forget(_ keys: [String], in file: URL) {
        var all = read(file)
        guard keys.contains(where: { all[$0] != nil }) else { return }
        for key in keys { all[key] = nil }
        try? write(all, to: file)
    }

    /// The current measurement of each drive among `drives` (a folder or file on it, the model's or copy's files to read
    /// there, and its name), measuring the unmeasured and stale ones (all with `force`) unless `refusal` or `busy` stop it.
    /// Keyed by volume UUID; a drive without one is skipped. `log` says what was measured or skipped.
    @discardableResult
    public static func ensure(
        _ file: URL, drives: [(at: URL, files: [URL], name: String)], recordBytes: Int, force: Bool = false,
        secondsPerPoint: Double = 0.75, refusal: () -> String? = DriveSpeed.refusal, busy: (URL) -> Bool = { DriveSpeed.othersBusy($0) },
        connection: (URL) -> String? = DriveSpeed.connection, log: (String) -> Void = { _ in }
    ) -> [String: DriveSpeed] {
        var all = read(file)
        var grouped: [String: (files: [URL], name: String)] = [:]
        var order: [String] = []
        var connected: [String: String] = [:]
        for drive in drives {
            guard let key = key(drive.at) else { continue }
            if grouped[key] == nil {
                order.append(key)
                connected[key] = connection(drive.at)
            }
            grouped[key, default: ([], drive.name)].files += drive.files
        }
        var changed = false
        // Unmeasured, stale, or measured on another port or enclosure (one measured before connections were kept counts
        // as another).
        for key in order
        where force || all[key].map({ $0.stale() || (connected[key] != nil && $0.connection != connected[key]) }) ?? true {
            let (files, name) = grouped[key]!
            if let reason = refusal() {
                log("drive speed of \(name) not measured: \(reason)")
                continue
            }
            let started = Date()
            do {
                guard
                    let speed = try DriveSpeed.measure(
                        files: files, recordBytes: recordBytes, name: name, secondsPerPoint: secondsPerPoint, busy: busy)
                else {
                    log("drive speed of \(name) not measured: the drive is busy")
                    continue
                }
                all[key] = speed
                all[key]?.connection = connected[key]
                changed = true
                log(
                    String(
                        format: "drive speed of %@: %@ (%.1f s)", name,
                        zip(speed.queueDepths, speed.bytesPerSecond).map { String(format: "%.2f GB/s at %d", $1 / 1e9, $0) }
                            .joined(separator: ", "), Date().timeIntervalSince(started)))
            } catch { log("drive speed of \(name) not measured: \(error)") }
        }
        if changed { try? write(all, to: file) }
        return all.filter { grouped[$0.key] != nil }
    }
}

/// A drive's live estimate (a rate, or a time multiple of the measurement) settled back toward its stored measurement
/// once the drive has had no reads for `graceSeconds`, with a `settleSeconds` time constant: a drive that slowed and was
/// then avoided is tried again at its measured speed instead of being judged forever on its last slow reads.
public enum LiveDriveEstimate {
    public static let graceSeconds = 0.25
    public static let settleSeconds = 1.0
    /// A read observed `gapSeconds` after the drive's previous one weighs at least this much in a moving average: rare
    /// reads (a drive mostly avoided) stand for more time, so its estimate follows the drive within seconds, not reads.
    public static func weight(_ smoothing: Double, gapSeconds: Double) -> Double {
        max(smoothing, 1 - exp(-max(0, gapSeconds) / 0.5))
    }
    public static func settled(_ live: Double, stored: Double, idleSeconds: Double) -> Double {
        stored + (live - stored) * exp(-max(0, idleSeconds - graceSeconds) / settleSeconds)
    }
}

/// Serving's own view of a drive against its saved measurement: once their ratio stays outside [1/3, 3] for ten
/// minutes of observations in a row, the measurement is taken as wrong (a different drive behind the same volume, a
/// worn or throttling one) and `observe` says so once, so the caller forgets it and the next start measures again.
public final class DriveSpeedReview: @unchecked Sendable {
    public static let factor = 3.0
    public static let sustainedSeconds = 600.0
    private let lock = NSLock()
    private var since: [String: Double] = [:]
    private var reported: Set<String> = []
    public init() {}
    /// `ratio`: observed over measured rate of `drive` (ignored unless positive and finite); `now`: seconds.
    public func observe(drive: String, ratio: Double, now: Double = Date().timeIntervalSince1970) -> Bool {
        guard ratio.isFinite, ratio > 0 else { return false }
        return lock.withLock {
            guard !reported.contains(drive) else { return false }
            guard ratio > Self.factor || ratio < 1 / Self.factor else {
                since[drive] = nil
                return false
            }
            let start = since[drive] ?? now
            since[drive] = start
            guard now - start >= Self.sustainedSeconds else { return false }
            reported.insert(drive)
            return true
        }
    }
}
