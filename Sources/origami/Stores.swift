import Darwin
import Foundation
import OrigamiEngine

/// The drives and folders Origami may write expert copies to, and what is in them.
enum Stores {
    /// The volume a folder is on (statfs of its nearest existing ancestor): name, physical disk, space, and whether
    /// a /Volumes path is really mounted.
    struct Volume {
        let name: String
        /// Physical disk ("disk3"): two folders on one disk share its bandwidth.
        let disk: String
        let free: UInt64
        let total: UInt64
        let mounted: Bool
        /// The volume's UUID (nil for file systems without one): a drive's identity, whatever its mount path.
        let uuid: String?
    }
    /// One store and this model's payloads in its model folder (pack, mirror), each header checked
    /// against the model; a payload that fails is left out and its reason kept in `problems`.
    struct Found {
        let store: URL
        let folder: URL
        let volume: Volume
        var pack: StoreHeader?
        var mirror: StoreHeader?
        /// Records of this model's further copy in the hot-copy format (`copy/` in the model folder, as `origami setup`
        /// now writes one; an older `mirror` is still read in place), 0 when none.
        var further = 0
        /// Why a payload there was not usable (for `origami status`).
        var problems: [String] = []
    }

    static func volume(_ url: URL) -> Volume {
        // Walk up to an existing ancestor: a store folder may not exist yet.
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path) && probe.path != "/" { probe = probe.deletingLastPathComponent() }
        var info = statfs()
        guard statfs(probe.path, &info) == 0 else { return Volume(name: "?", disk: "?", free: 0, total: 0, mounted: false, uuid: nil) }
        let device = withUnsafeBytes(of: info.f_mntfromname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let mount = withUnsafeBytes(of: info.f_mntonname) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        // "/dev/disk3s3s1" -> "disk3": APFS volumes of one container share a disk number.
        let name = device.split(separator: "/").last.map(String.init) ?? device
        let wholeDisk = name.hasPrefix("disk") ? "disk" + name.dropFirst(4).prefix { $0.isNumber } : name
        // A store on an unplugged drive points into /Volumes, which then resolves to the boot volume.
        let mounted = !url.path.hasPrefix("/Volumes/") || mount.hasPrefix("/Volumes/")
        let free = max(PromptCache.freeDiskBytes(at: probe), UInt64(info.f_bavail) * UInt64(info.f_bsize))
        let isInternal = mount == "/" || mount == "/System/Volumes/Data"
        return Volume(
            name: isInternal ? "internal drive" : (mount as NSString).lastPathComponent, disk: wholeDisk,
            free: free, total: UInt64(info.f_blocks) * UInt64(info.f_bsize), mounted: mounted,
            uuid: (try? probe.resourceValues(forKeys: [.volumeUUIDStringKey]))?.volumeUUIDString)
    }

    /// The problem a store reports when its drive did not answer (macOS waiting on removable-volume permission).
    static let noAnswer =
        "drive did not answer within 10 s (macOS may be waiting for permission to read removable volumes; see Troubleshooting in the README)"

    /// Whether `url` (a file or folder) can be opened within `seconds`. Off /Volumes it always can; on a removable volume
    /// without macOS's permission, opening waits on a prompt nobody may see (a server started by launchd), so the
    /// attempt runs on its own thread and is abandoned after the timeout. Opening it is also what makes macOS ask.
    static func answers(_ url: URL, within seconds: Double = 10) -> Bool {
        guard url.path.hasPrefix("/Volumes/") else { return true }
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            let fd = open(url.path, O_RDONLY)
            if fd >= 0 { close(fd) }
            done.signal()
        }
        return done.wait(timeout: .now() + seconds) == .success
    }

    /// Where a model's file or folder stands, as the dashboard's model menu shows it: `checking` and `slow` are a check
    /// still waiting (under 3 s, longer), never a verdict; `missing` is said only by a drive that answered.
    enum Availability: String { case available, checking, slow, notConnected = "not_connected", noPermission = "no_permission", missing }

    /// Whether `path` opens now, waiting as long as its drive takes (a sleeping one wakes): its /Volumes drive is not
    /// mounted, macOS or the file's permissions refuse it, nothing is there, or it opens (`available`).
    static func probe(_ path: String) -> Availability {
        if path.hasPrefix("/Volumes/") {
            // An unplugged drive's path resolves to the boot volume (or nowhere): its mount point is not its own.
            var info = statfs()
            let mount = "/Volumes/" + driveName(path)
            guard statfs(mount, &info) == 0,
                withUnsafeBytes(of: info.f_mntonname, { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }).lowercased()
                    == mount.lowercased()
            else { return .notConnected }
        }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            let error = errno
            return error == EPERM || error == EACCES ? .noPermission : .missing
        }
        close(fd)
        return .available
    }

    /// `probe` on its own thread, abandoned after `seconds` as `slow` (a drive waking, or macOS waiting on a permission
    /// prompt nobody may see), so a caller is never held longer.
    static func check(_ path: String, within seconds: Double) -> Availability {
        final class Result: @unchecked Sendable { var state = Availability.slow }
        let result = Result()
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            result.state = probe(path)
            done.signal()
        }
        return done.wait(timeout: .now() + seconds) == .success ? result.state : .slow
    }

    /// The drive a path is on, by name: "internal drive", or the volume's name under /Volumes.
    static func driveName(_ path: String) -> String {
        path.hasPrefix("/Volumes/") ? URL(fileURLWithPath: path).pathComponents.dropFirst(2).first ?? "?" : "internal drive"
    }

    /// Every configured store, the hot copy's folder and the data folder's own store, with this model's payloads.
    static func scan(model: ModelSource, settings: Settings) -> [Found] {
        settings.storeFolders.map { store in
            guard store.path.hasPrefix("/Volumes/") else { return scan(store, model: model, settings: settings) }
            // Without permission to read removable volumes, file access waits on a prompt nobody may see (a server
            // started by launchd): give up after 10 s, as the engine does, so `serve` still starts listening.
            final class Result: @unchecked Sendable { var found: Found? }
            let result = Result()
            let done = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                result.found = scan(store, model: model, settings: settings)
                done.signal()
            }
            if done.wait(timeout: .now() + 10) == .success, let found = result.found { return found }
            return Found(
                store: store, folder: store,
                volume: Volume(name: driveName(store.path), disk: "?", free: 0, total: 0, mounted: false, uuid: nil),
                problems: [noAnswer])
        }
    }

    /// One store and this model's payloads there.
    private static func scan(_ store: URL, model: ModelSource, settings: Settings) -> Found {
        let folder = StoreHeader.modelFolder(store: store, model: model)
        var found = Found(store: store, folder: folder, volume: volume(store))
        guard found.volume.mounted else {
            found.problems.append("drive not connected")
            return found
        }
        for kind in [StoreHeader.Kind.pack, .mirror] {
            let label = kind.rawValue
            do {
                guard let header = try StoreHeader.load(kind, in: folder) else { continue }
                if let reason = header.mismatch(for: model, folder: folder) {
                    found.problems.append("\(label): \(reason)")
                    continue
                }
                if kind == .pack { found.pack = header } else { found.mirror = header }
            } catch { found.problems.append("\(label): \(error)") }
        }
        if let copy = OrigamiEngine.HotCopy.open(
            folder: folder.appendingPathComponent(PackStore.furtherCopyFolder, isDirectory: true), identity: hotCopyIdentity(model))
        {
            found.further = copy.keys.count
        }
        return found
    }

    /// Chooses the pack and its copies to serve from, and the hot copy's folder. A pack on another drive than the model
    /// wins (two drives in parallel); the hot copy belongs in `hot_copy_dir`, else on the model's drive.
    static func plan(model: ModelSource, settings: Settings, found: [Found]) -> OrigamiStorage {
        // GGUF experts: packs and further copies hold MLX records; the hot copy holds the GGUF's own, and pays only off
        // the GGUF's drive (three reads there against one here, at the same speed).
        if model.ggufExperts != nil {
            let hot = settings.hotCopy == 0 ? nil : hotCopyFolder(model: model, settings: settings, found: found)
            return OrigamiStorage(hotCopyFolder: hot.flatMap { volume($0).disk == volume(model.expertsLocation).disk ? nil : $0 })
        }
        let modelDisk = volume(model.directory).disk
        // With copies on several drives, read the one off the model's drive that measured fastest.
        let packs = found.filter { $0.pack != nil }
        let speed = { (item: Found) -> Double in Calibrations.speed(settings, item.folder)?.bytesPerSecond.max() ?? 0 }
        let pack = packs.filter { $0.volume.disk != modelDisk }.max { speed($0) < speed($1) } ?? packs.first
        let packSharesModelDrive = pack.map { $0.volume.disk == modelDisk } ?? false
        var storage = OrigamiStorage(
            packFolder: pack?.folder, packSharesModelDrive: packSharesModelDrive,
            hotCopyFolder: settings.hotCopy == 0 || packSharesModelDrive
                ? nil : hotCopyFolder(model: model, settings: settings, found: found))
        if let pack, !storage.packSharesModelDrive {
            storage.calibration = Calibrations.load(settings: settings, model: model, pack: pack.folder)
        }
        // Every other copy off the model's drive is read beside the pack: packs fastest first, then most-used experts.
        // An earlier version's offline hot mirror where the hot copy now goes is not one of them.
        let legacy = settings.hotCopy == 0 ? nil : legacyMirror(model: model, settings: settings, found: found)?.folder
        if let pack {
            let copies =
                packs.filter { $0.volume.disk != modelDisk }.sorted { speed($0) > speed($1) }
                + found.filter { $0.pack == nil && ($0.mirror != nil || $0.further > 0) && $0.volume.disk != modelDisk }
            storage.copyFolders = copies.map(\.folder).filter { $0 != pack.folder && $0 != legacy }
            storage.packDrives = Set(([pack] + copies.filter { storage.copyFolders.contains($0.folder) }).map(\.volume.disk)).count
        }
        return storage
    }

    /// Smaller experts an earlier Origami made (`expert_bits`, since removed), which nothing reads: Qwen's 3- and 2-bit
    /// packs (`pack-<n>bit.json` and its payload) in the stores, and GLM and DeepSeek's converted experts
    /// (`family-<key>/pack-<type>`) with the hot copies made of them (`hot-<type>`, also in the data folder). Each with
    /// its bytes and the command that deletes it; nothing is deleted here.
    static func retiredExperts(_ settings: Settings) -> [(path: String, bytes: UInt64, delete: String)] {
        let manager = FileManager.default
        func names(_ folder: URL) -> [String] { ((try? manager.contentsOfDirectory(atPath: folder.path)) ?? []).sorted() }
        func size(_ url: URL) -> UInt64 {
            let urls =
                manager.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]).map {
                    $0.compactMap { $0 as? URL }
                } ?? [url]
            return urls.reduce(0) {
                $0 + UInt64((try? $1.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize ?? 0)
            }
        }
        func quoted(_ path: String) -> String { "'" + path.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        var out: [(path: String, bytes: UInt64, delete: String)] = []
        var seen = Set<String>()
        for store in settings.storeFolders + FamilySetup.stores(settings) where seen.insert(store.standardizedFileURL.path).inserted {
            for folder in Commands.origamiFolders(in: store) {
                for header in ["pack-3bit.json", "pack-2bit.json"] {
                    let url = folder.appendingPathComponent(header)
                    guard let payload = (try? Settings.readConfig(url))?["expert_file"] as? String else { continue }
                    let files = [url] + (payload.contains("/") ? [] : [folder.appendingPathComponent(payload)])
                    out.append(
                        (url.path, files.reduce(0) { $0 + size($1) }, "rm " + files.map { quoted($0.path) }.joined(separator: " ")))
                }
            }
            for base in names(store) where base.range(of: "^family-[0-9a-f]{16}$", options: .regularExpression) != nil {
                let folder = store.appendingPathComponent(base, isDirectory: true)
                for name in names(folder)
                where name.hasPrefix("hot-")
                    || name.hasPrefix("pack-") && manager.fileExists(atPath: folder.appendingPathComponent(name + "/pack.json").path)
                {
                    let url = folder.appendingPathComponent(name, isDirectory: true)
                    out.append((url.path, size(url), "rm -r " + quoted(url.path)))
                }
            }
        }
        let family = settings.dataDir.appendingPathComponent("family", isDirectory: true)
        for key in names(family) {
            for name in names(family.appendingPathComponent(key)) where name.hasPrefix("hot-") {
                let url = family.appendingPathComponent(key + "/" + name, isDirectory: true)
                out.append((url.path, size(url), "rm -r " + quoted(url.path)))
            }
        }
        return out
    }
    /// Where Qwen's hot copy goes (and `origami setup` wrote its offline mirror before 3.1): in `hot_copy_dir`, else a
    /// store on the model's drive, else the last store (the data folder's).
    static func mirrorHome(_ item: Found, settings: Settings, modelDisk: String) -> Bool {
        settings.hotCopyDir.map { item.store == $0 } ?? (item.volume.disk == modelDisk)
    }
    static func mirrorTarget(settings: Settings, found: [Found], modelDisk: String) -> Found? {
        found.first { mirrorHome($0, settings: settings, modelDisk: modelDisk) && $0.volume.mounted } ?? found.last
    }
    /// Qwen's hot copy folder: `hot` in this model's folder of `hot_copy_dir` (connected or not), else of the store
    /// `mirrorTarget` picks (GGUF experts: the data folder's store, as the family engines' copy goes to the data
    /// folder). Its own folder: a pack's `experts.bin` beside it is never a hot copy's payload.
    static func hotCopyFolder(model: ModelSource, settings: Settings, found: [Found]) -> URL? {
        guard
            let store = settings.hotCopyDir
                ?? (model.ggufExperts != nil
                    ? settings.homeStore : mirrorTarget(settings: settings, found: found, modelDisk: volume(model.directory).disk)?.store)
        else { return nil }
        return StoreHeader.modelFolder(store: store, model: model).appendingPathComponent("hot", isDirectory: true)
    }
    /// This model's hot copy in `item`'s `hot` subfolder, read from the drive now, in the words of `origami status`'s
    /// hot copy line ("hot copy 2889 experts (8.0 GB)"); nil when it holds none.
    static func hotCopyHeld(_ item: Found, model: ModelSource) -> String? {
        guard item.volume.mounted,
            let copy = OrigamiEngine.HotCopy.open(
                folder: item.folder.appendingPathComponent("hot", isDirectory: true), identity: hotCopyIdentity(model))
        else { return nil }
        return "hot copy \(copy.keys.count) experts (\(OrigamiPlan.gigabytes(UInt64(copy.bytes))))"
    }
    /// The model a Qwen hot copy holds records of: its fingerprint and expert layout.
    static func hotCopyIdentity(_ model: ModelSource) -> String { PackStore.hotCopyIdentity(model) }
    /// The offline mirror (`mirror.json`, this model's) an earlier version's `origami setup` wrote where the hot copy now
    /// goes: the one `plan` then read as the hot mirror. A mirror elsewhere is a further copy on another drive and stays.
    static func legacyMirror(model: ModelSource, settings: Settings, found: [Found]) -> Found? {
        let modelDisk = volume(model.directory).disk
        return found.first { $0.mirror != nil && mirrorHome($0, settings: settings, modelDisk: modelDisk) }
    }
    /// Removes the offline mirror `legacyMirror` finds when the hot copy in `folder` (its `hot` subfolder) starts its
    /// first build: header first, so no reader pairs it with a missing payload. The bytes freed, or nil when there is none.
    static func retireLegacyMirror(model: ModelSource, settings: Settings, hotCopy folder: URL) -> UInt64? {
        guard let legacy = legacyMirror(model: model, settings: settings, found: scan(model: model, settings: settings)),
            let header = legacy.mirror,
            legacy.folder.standardizedFileURL == folder.deletingLastPathComponent().standardizedFileURL
        else { return nil }
        let payload = legacy.folder.appendingPathComponent(header.expertFile)
        let bytes = ((try? FileManager.default.attributesOfItem(atPath: payload.path))?[.size] as? NSNumber)?.uint64Value ?? 0
        guard (try? FileManager.default.removeItem(at: legacy.folder.appendingPathComponent(StoreHeader.fileName(.mirror)))) != nil
        else { return nil }
        try? FileManager.default.removeItem(at: payload)
        return bytes
    }

    /// What `origami setup` may change for a Qwen model: each header in the stores and the hot copy's folder, as its path
    /// and when it was written, with its payload's bytes. A hot-copy job reloads the model only when this changes.
    static func signature(_ settings: Settings) -> [String: UInt64] {
        var out: [String: UInt64] = [:]
        for folder in settings.storeFolders.flatMap(Commands.origamiFolders(in:)) {
            for name in ["pack.json", "mirror.json"] {
                let header = folder.appendingPathComponent(name)
                guard let written = (try? header.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate else {
                    continue
                }
                let payload = (try? Settings.readConfig(header))?["expert_file"] as? String
                out["\(header.path) \(written.timeIntervalSince1970)"] =
                    payload.flatMap {
                        ((try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent($0).path))?[.size] as? NSNumber)?
                            .uint64Value
                    } ?? 0
            }
        }
        return out
    }

    /// How expert reads are spread, in words.
    static func reads(_ storage: OrigamiStorage) -> String {
        guard storage.packFolder != nil else { return "the model files" + (storage.hotCopyFolder == nil ? "" : " and the hot copy") }
        let drives = storage.packDrives + (storage.packSharesModelDrive ? 0 : 1)
        return drives == 1 ? "a copy of the experts on the model drive" : "\(drives) drives at once"
    }
}

extension ExpertSourceCalibration {
    /// The model and pack drives' rates at four readers, as status shows them.
    var atFourReaders: (model: Double, pack: Double) {
        let index = queueDepths.lastIndex { $0 <= 4 }!
        return (internalBytesPerSecond[index], externalBytesPerSecond[index])
    }
}

/// Serving's own reads against each measured drive, once per finished request: the "3x for 10 minutes" rule
/// (`DriveSpeedReview`, `Calibrations.review`) for the model's drive, a pack's and every copy place.
final class DriveReview: @unchecked Sendable {
    let review = DriveSpeedReview()
    private let lock = NSLock()
    private var before: [(count: UInt64, seconds: Double)] = []

    /// Qwen: with a pack and two measured drives (`online`), the online estimator's multiples, as before; else the model's
    /// and the pack's drives from the reads since the last request (a record's bytes over each read's time, at least 32
    /// reads, `DriveSpeed.perReadRatio`); and every copy place from its lane.
    func qwen(_ settings: Settings, backend: NativeServingBackend, model: URL, pack: URL?, online: Bool, log: (String) -> Void) {
        var drives: [(URL, Double)] = []
        if online, let pack, let estimate = backend.sourceOnline {
            drives = [(model, estimate.internalSSD), (pack, estimate.externalSSD)].filter { $0.1.warm }.map {
                ($0.0, 1 / $0.1.smoothedMultiple)
            }
        } else if let now = backend.expertReadTimes {
            let previous = lock.withLock {
                defer { before = now.map { ($0.count, $0.seconds) } }
                return before
            }
            for (index, url) in [(0, pack), (1, model as URL?)] {
                guard let url, index < now.count, let speed = Calibrations.speed(settings, url) else { continue }
                let count = now[index].count &- (index < previous.count ? previous[index].count : 0)
                let seconds = now[index].seconds - (index < previous.count ? previous[index].seconds : 0)
                guard count >= 32, count < 1 << 40, seconds > 0 else { continue }
                drives.append((url, speed.perReadRatio(Double(PackStore.expertBytes) * Double(count) / seconds)))
            }
        }
        Calibrations.review(review, settings: settings, drives: drives + Self.places(backend.hotCopySet, settings), log: log)
    }
    /// Each copy place against its drive's measurement, from its lane's smoothed rate per read.
    static func places(_ set: HotCopySet?, _ settings: Settings) -> [(URL, Double)] {
        (set?.laneRates ?? []).compactMap { place in
            Calibrations.speed(settings, place.folder).map { (place.folder, $0.perReadRatio(place.bytesPerSecond)) }
        }
    }
}

/// This Mac's drive measurements (`DriveSpeed`), in `<data_dir>/drive-speeds.json` keyed by volume UUID: taken at the
/// first start on a drive (before the server listens, so never during a request), by `origami setup` and `origami tune`,
/// again when four weeks old or when serving keeps disagreeing with them (`DriveSpeedReview`).
enum Calibrations {
    /// The model's files and every copy of it the stores hold, by drive.
    static func drives(model: ModelSource, found: [Stores.Found]) -> [(at: URL, files: [URL], name: String)] {
        let own = ((try? FileManager.default.contentsOfDirectory(at: model.directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
        return [(model.directory, own, Stores.volume(model.directory).name)]
            + found.filter { ($0.pack != nil || $0.mirror != nil || $0.further > 0) && $0.volume.mounted }.map {
                let copy = $0.folder.appendingPathComponent(PackStore.furtherCopyFolder, isDirectory: true)
                return (
                    $0.folder,
                    ((try? FileManager.default.contentsOfDirectory(at: $0.folder, includingPropertiesForKeys: nil)) ?? []).filter {
                        $0.lastPathComponent != PackStore.furtherCopyFolder
                    } + ((try? FileManager.default.contentsOfDirectory(at: copy, includingPropertiesForKeys: nil)) ?? []),
                    $0.volume.name
                )
            }
    }
    /// Measures the drives of a Qwen model that are unmeasured or stale (every one with `force`).
    static func measure(settings: Settings, model: ModelSource, found: [Stores.Found], force: Bool = false, log: (String) -> Void) {
        guard model.ggufExperts == nil else { return }
        DriveSpeeds.ensure(
            settings.driveSpeedsFile, drives: drives(model: model, found: found), recordBytes: PackStore.expertBytes, force: force,
            log: log)
    }
    /// Measures the configured model's drives, Qwen's or a family GGUF's (`origami tune`; serving applies them at its next
    /// start, as it does a tuning result).
    static func measureConfigured(_ settings: Settings, force: Bool, log: (String) -> Void) {
        if let url = FamilySetup.gguf(settings) {
            guard let gguf = try? ExpertVariant.source(url, settings: settings), let geometry = try? ModelGeometry(gguf: gguf) else {
                return
            }
            let hot = HotCopy.placeSettings(settings).map { FamilySetup.hotFolder($0, key: FamilySupport.key(gguf)) }
            FamilySetup.measureDrives(
                gguf, geometry: geometry, hot: hot[0], more: Array(hot.dropFirst()), settings: settings, force: force, log: log)
        } else if let model = try? Commands.model(settings) {
            measure(settings: settings, model: model, found: Stores.scan(model: model, settings: settings), force: force, log: log)
        }
    }
    static func speed(_ settings: Settings, _ url: URL) -> DriveSpeed? {
        DriveSpeeds.key(url).flatMap { DriveSpeeds.read(settings.driveSpeedsFile)[$0] }
    }
    /// Two-drive curves from the measured model and pack drives, nil until both are measured.
    static func load(settings: Settings, model: ModelSource, pack: URL) -> ExpertSourceCalibration? {
        guard let modelDrive = speed(settings, model.directory), let packDrive = speed(settings, pack) else { return nil }
        return ExpertSourceCalibration(model: modelDrive, pack: packDrive, recordBytes: PackStore.expertBytes)
    }
    /// Serving's view against the measurements, once per finished request (`ratio`: observed over measured rate): a drive
    /// outside [1/3, 3] for ten minutes in a row (`DriveSpeedReview`) is forgotten, so the next start measures it again.
    static func review(_ review: DriveSpeedReview, settings: Settings, drives: [(URL, Double)], log: (String) -> Void) {
        for (url, ratio) in drives where review.observe(drive: url.path, ratio: ratio) {
            guard let key = DriveSpeeds.key(url) else { continue }
            DriveSpeeds.forget([key], in: settings.driveSpeedsFile)
            log("drive speed of \(Stores.volume(url).name) disagrees with serving's own reads; it is measured again at the next start")
        }
    }
    /// A copy place's drive, measured when the place is added rather than once it holds a copy (until then it had no
    /// files to read and came last in the places' order): a 256 MiB probe of random bytes written uncached in its folder,
    /// read as records are (Qwen's, or the family model's average record, `FamilySetup.measureDrives`), then deleted.
    static func measurePlace(_ folder: URL, settings: Settings, qwen: Bool, log: (String) -> Void) {
        let probe = folder.appendingPathComponent(".origami-drive-probe")
        defer { try? FileManager.default.removeItem(at: probe) }
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            guard FileManager.default.createFile(atPath: probe.path, contents: nil) else { return }
            let handle = try FileHandle(forWritingTo: probe)
            defer { try? handle.close() }
            _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
            var chunk = Data(count: 8 << 20)
            chunk.withUnsafeMutableBytes { arc4random_buf($0.baseAddress, $0.count) }
            for _ in 0..<32 { try handle.write(contentsOf: chunk) }
            try handle.synchronize()
        } catch {
            log("drive speed of \(Stores.volume(folder).name) not measured: \(error)")
            return
        }
        if qwen {
            DriveSpeeds.ensure(
                settings.driveSpeedsFile, drives: [(folder, [probe], Stores.volume(folder).name)], recordBytes: PackStore.expertBytes,
                log: log)
        } else if let url = FamilySetup.gguf(settings), let gguf = try? ExpertVariant.source(url, settings: settings),
            let geometry = try? ModelGeometry(gguf: gguf)
        {
            _ = FamilySetup.measureDrives(gguf, geometry: geometry, hot: folder, settings: settings, log: log)
        }
    }
    /// What `/v1/status`, `origami status` and `report` show: each measured drive, its rates and when.
    static func report(_ settings: Settings) -> [[String: Any]] {
        DriveSpeeds.read(settings.driveSpeedsFile).sorted { $0.key < $1.key }.map {
            $0.value.report.merging(["volume_uuid": $0.key, "stale": $0.value.stale()]) { $1 }
        }
    }
    /// The same in words, one line per drive (`origami status`, `origami doctor`).
    static func lines(_ settings: Settings) -> [String] {
        let all = DriveSpeeds.read(settings.driveSpeedsFile).values.sorted { $0.name < $1.name }
        guard !all.isEmpty else { return ["read speeds not measured yet (measured at the first start, or by `origami setup`)"] }
        let day = DateFormatter()
        day.dateFormat = "yyyy-MM-dd"
        return all.map { speed in
            "\(speed.name): " + speed.bytesPerSecond.map { String(format: "%.2f", $0 / 1e9) }.joined(separator: ", ")
                + " GB/s at \(speed.queueDepths.map(String.init).joined(separator: ", ")) readers, measured "
                + day.string(from: Date(timeIntervalSince1970: speed.measuredAt))
                + (speed.stale() ? " (stale: measured again at the next start)" : "")
        }
    }
}
