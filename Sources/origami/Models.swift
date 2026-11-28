import CryptoKit
import Foundation
import OrigamiEngine

/// Finding the model on disk, or fetching it from Hugging Face.
enum Models {
    /// Public builds known to run (4-bit group 64, the original architecture).
    static let defaultRepo = "carloslfu/Qwen3.8-Flash-Next-MLX-4bit"
    static let knownRepos = [defaultRepo]

    /// The folder for a model setting: a path, or a repo id found in a known download location.
    static func locate(_ spec: String?, settings: Settings) -> URL? {
        if let path = ModelRegistry.entry(spec, settings: settings)?["path"] as? String {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        guard let spec else { return knownRepos.lazy.compactMap { find(repo: $0, settings: settings) }.first }
        if spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix(".") {
            let url = URL(fileURLWithPath: Settings.expand(spec), isDirectory: true)
            return FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) ? url : nil
        }
        return find(repo: spec, settings: settings)
    }

    /// Looks where Origami, Hugging Face tools, SlotStream and LM Studio keep downloads.
    static func find(repo: String, settings: Settings) -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let hub =
            ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
            ?? ProcessInfo.processInfo.environment["HF_HOME"].map { $0 + "/hub" } ?? home + "/.cache/huggingface/hub"
        var candidates = [
            settings.modelsDir.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--")).path,
            home + "/.lmstudio/models/" + repo, home + "/.cache/lm-studio/models/" + repo,
        ]
        if repo == defaultRepo { candidates.append(home + "/.slotstream/models/qwen38-flash-next-mlx-4bit") }
        let snapshots = hub + "/models--" + repo.replacingOccurrences(of: "/", with: "--") + "/snapshots"
        candidates += ((try? FileManager.default.contentsOfDirectory(atPath: snapshots)) ?? []).map { snapshots + "/" + $0 }
        return candidates.lazy.map { URL(fileURLWithPath: $0, isDirectory: true) }
            .first { (try? ModelSource(directory: $0)) != nil }
    }

    /// Downloads `repo` into the data folder. Resumes partial files and checks each file's SHA-256.
    /// `include`: the repo's top-level files to fetch (by default weights, configs, tokenizer, template and licence);
    /// `folder`: where they go (by default `<models_dir>/<owner>--<name>`).
    static func download(repo: String, settings: Settings, include: ((String) -> Bool)? = nil, into folder: URL? = nil) async throws -> URL
    {
        guard repo.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil else {
            throw CLIError("\(repo) is not a Hugging Face repo id; give it as owner/name, such as \(defaultRepo)")
        }
        let destination =
            folder ?? settings.modelsDir.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let token = HubRepo.token
        func request(_ url: URL) -> URLRequest {
            var request = URLRequest(url: url)
            if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            return request
        }
        let info = URL(string: "https://huggingface.co/api/models/\(repo)/revision/main?blobs=true")!
        let data: Data
        let response: URLResponse
        do { (data, response) = try await URLSession.shared.data(for: request(info)) } catch {
            throw CLIError(
                "Cannot reach huggingface.co (\(error.localizedDescription)); check the connection and run the same command again")
        }
        guard (response as? HTTPURLResponse)?.statusCode == 200,
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let revision = object["sha"] as? String, revision.range(of: "^[0-9A-Za-z]+$", options: .regularExpression) != nil,
            let siblings = object["siblings"] as? [[String: Any]]
        else {
            throw CLIError(
                "Hugging Face does not list \(repo) (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)); check the name, "
                    + "and for a gated or private repo set HF_TOKEN and accept the model's licence on huggingface.co")
        }
        let wanted = siblings.compactMap { entry -> (name: String, size: Int64, sha256: String?)? in
            // Plain top-level names only: no folders, no hidden files, nothing a URL or a path could misread.
            guard let name = entry["rfilename"] as? String,
                name.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil,
                include?(name)
                    ?? (name.hasSuffix(".safetensors") || name.hasSuffix(".json") || name.hasSuffix(".jinja")
                        || ["merges.txt", "vocab.json", "LICENSE"].contains(name))
            else { return nil }
            let lfs = entry["lfs"] as? [String: Any]
            return (name, (entry["size"] as? NSNumber)?.int64Value ?? 0, lfs?["sha256"] as? String)
        }
        let total = wanted.reduce(Int64(0)) { $0 + $1.size }
        let free = Int64(PromptCache.freeDiskBytes(at: destination))
        let present = wanted.reduce(Int64(0)) { sum, file in
            sum
                + (fileSize(destination.appendingPathComponent(file.name)) ?? fileSize(
                    destination.appendingPathComponent(file.name + ".part")) ?? 0)
        }
        guard free + present >= total + 5_000_000_000 else {
            throw CLIError(
                "Downloading \(repo) needs \(OrigamiPlan.gigabytes(UInt64(total))) and \(destination.path) has \(OrigamiPlan.gigabytes(UInt64(max(0, free)))) free. "
                    + "Download to a bigger drive with `origami model add \(repo) --to /Volumes/…`, or pass --model with a folder you already have."
            )
        }
        print("Downloading \(repo) (\(OrigamiPlan.gigabytes(UInt64(total)))) to \(destination.path)")
        // A drive on a USB 2 cable or hub writes 30-40 MB/s: said before hours of download, not found out after.
        if total >= 1_000_000_000, let rate = writeRate(destination), rate < 60e6 {
            print(
                String(
                    format:
                        "  This drive writes about %.0f MB/s, as over a USB 2 cable or hub: the download would take hours and the "
                        + "model would run very slowly. A USB 3 or Thunderbolt cable and port fix that.", rate / 1e6))
        }
        let progress = DownloadProgress(total: total)
        let recordURL = destination.appendingPathComponent(".origami-download.json")
        let record = try? Data(contentsOf: recordURL)
        try await withThrowingTaskGroup(of: Void.self) { group in
            var pending = wanted.makeIterator()
            var running = 0
            func next() {
                guard let file = pending.next() else { return }
                running += 1
                let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(file.name)")!
                let target = destination.appendingPathComponent(file.name)
                let present = DownloadResume.decide(
                    name: file.name, existing: fileSize(target), size: file.size, sha256: file.sha256, revision: revision, record: record)
                group.addTask {
                    try await fetch(request(url), to: target, size: file.size, sha256: file.sha256, present: present, progress: progress)
                }
            }
            for _ in 0..<4 { next() }
            while running > 0 {
                try await group.next()
                running -= 1
                next()
            }
        }
        progress.finish()
        try DownloadResume.record(repo: repo, revision: revision, files: wanted).write(to: recordURL, options: .atomic)
        return destination
    }

    /// What `model add` and `setup` know of `repo` before downloading it (as the dashboard checks it): refused, without a
    /// byte of weights fetched, when it is GGUF files or a model Origami cannot use; else its one report.
    static func checkRepo(_ repo: String) throws -> CompatibilityReport {
        guard repo.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil else {
            throw CLIError("\(repo) is not a Hugging Face repo id; give it as owner/name, such as \(defaultRepo)")
        }
        let reports: [CompatibilityReport]
        do { reports = try ModelCompatibility.check(hub: HubRepo(repo)) } catch {
            throw CLIError(
                "Cannot read \(repo) on Hugging Face (\(error)); check the name and the connection, and for a gated or private repo set HF_TOKEN"
            )
        }
        if let refusal = ModelCompatibility.downloadRefusal(repo, reports: reports) { throw CLIError(refusal) }
        return reports[0]
    }

    /// The official config and tokenizer files a GGUF conversion needs (a few MB), kept for later conversions.
    static func ggufSupport(settings: Settings) async throws -> URL {
        let folder = ggufSupportFolder(settings)
        if ["config.json", "tokenizer.json", "tokenizer_config.json"].allSatisfy({
            FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
        }) {
            return folder
        }
        print("Fetching the official config and tokenizer the conversion needs (a few MB).")
        return try await download(
            repo: "Qwen/Qwen3.8-Flash-Next", settings: settings,
            include: { name in
                name != "model.safetensors.index.json"
                    && (name.hasSuffix(".json") || name.hasSuffix(".jinja") || ["merges.txt", "vocab.json", "LICENSE"].contains(name))
            }, into: folder)
    }

    /// Where `ggufSupport` keeps them.
    static func ggufSupportFolder(_ settings: Settings) -> URL {
        settings.modelsDir.appendingPathComponent(".gguf-support", isDirectory: true)
    }

    /// The converter `model add` runs on a GLM-5.3-Flash safetensors checkpoint: experts in Q2_K (`bits` "2", the
    /// default, as the dashboard adds it), Q4_K or Q8_0.
    static func familyConverter(_ url: URL, bits: String = "2") throws -> FamilyConverter {
        guard let type = ["2": 10, "4": 12, "8": 8][bits] else {
            throw CLIError("--expert-bits \"\(bits)\" is not 2, 4 or 8 for a GLM conversion")
        }
        return try FamilyConverter(source: url, expertType: type)
    }

    /// The bytes `model add` writes converting the model at `url` once (one the compatibility check says converts),
    /// from its headers and with the converter `model add` runs: a GLM checkpoint's GGUF, else Qwen's 4-bit layout of a
    /// checkpoint (unquantized, FP8, NVFP4 or MLX at another width) or a GGUF. The dashboard's check reads it, so the page
    /// and `model add` judge the room by one number. Throws what `model add` would refuse with.
    static func conversionBytes(_ url: URL, family: ModelFamily, settings: Settings) throws -> UInt64 {
        if family == .glm { return UInt64(try familyConverter(url).outputBytes) }
        guard isGGUF(url.path) else { return try CheckpointConverter(source: url).outputBytes }
        let source = try GGUFSource(url)
        let support = ggufSupportFolder(settings)
        if let problem = CheckpointConverter.ggufProblem(source) { throw CLIError("Origami cannot convert this GGUF: \(problem)") }
        // shortcut: until the first GGUF conversion fetches the support files, the GGUF's own size stands in, about an upper
        // bound (4- to 8-bit weights become 4.5-bit, native experts stay in it); fetch them here if it ever refuses a fit.
        guard FileManager.default.fileExists(atPath: support.appendingPathComponent("config.json").path) else { return source.bytes }
        return try CheckpointConverter(gguf: source, support: support).outputBytes
    }

    /// `conversionBytes` of a Hugging Face checkpoint before it is downloaded (the folder `base` of `repo`, "" for its top):
    /// its config, index and template, and each shard's header in a sparse file of the shard's size, read by the same
    /// converters in a temporary folder. No weight is fetched or stored; the headers are read once more by HTTP range.
    static func conversionBytes(hub repo: HubRepo, base: String, family: ModelFamily, settings: Settings) throws -> UInt64 {
        let local = FileManager.default.temporaryDirectory.appendingPathComponent("origami-check-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: local) }
        let prefix = base.isEmpty ? "" : base + "/"
        let kept: Set = ["config.json", "model.safetensors.index.json", "chat_template.jinja", "tokenizer_config.json"]
        for file in try repo.files() where file.path.hasPrefix(prefix) {
            // A repo's names are untrusted: only plain names directly in `base` are laid out.
            let name = String(file.path.dropFirst(prefix.count))
            guard !name.contains("/"), !name.hasPrefix("."), name.hasSuffix(".safetensors") || kept.contains(name) else { continue }
            let url = local.appendingPathComponent(name)
            guard name.hasSuffix(".safetensors") else {
                try repo.data(file.path).write(to: url)
                continue
            }
            let bytes = repo.bytes(file)
            let head = try bytes.read(at: 0, count: 8)
            let length = head.count == 8 ? head.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) } : 0
            guard length > 0, length <= 64 * 1024 * 1024, length + 8 <= file.size else {
                throw CLIError("\(file.path) is not a safetensors file")
            }
            guard FileManager.default.createFile(atPath: url.path, contents: head + (try bytes.read(at: 8, count: Int(length)))) else {
                throw CLIError("Cannot write \(url.path)")
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: file.size)
        }
        return try conversionBytes(local, family: family, settings: settings)
    }

    /// The free space converting `bytes` once needs where it is written: the output and 10 GB kept free.
    static func conversionNeeds(_ bytes: UInt64) -> UInt64 { bytes + 10_000_000_000 }

    /// Models on this Mac that Origami could add, judged from their headers (`ModelCompatibility`): folders with a
    /// config.json and safetensors, and GGUF files (a split one once, by its first part), under the usual download places
    /// (Origami's, Hugging Face's cache, LM Studio's, ~/Models), the `models_dirs` folders and the Models folder of each
    /// drive Origami already reads.
    /// Each carries what its packs and copies are named by: `fingerprint` (Qwen) or `key` (a GGUF).
    /// `known`: paths already configured or registered, left out. Each drive is given 20 s to wake (the search runs off the
    /// request path); one not read is in `skipped` with why, and in `unreadable` when macOS may be waiting on permission.
    /// `partial`: why the search missed more (its folder or model limit reached, a search folder not readable).
    static func found(settings: Settings, known: Set<String>) -> (
        models: [[String: Any]], unreadable: [String], skipped: [String: Stores.Availability], partial: String?
    ) {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let hub =
            ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
            ?? ProcessInfo.processInfo.environment["HF_HOME"].map { $0 + "/hub" } ?? home + "/.cache/huggingface/hub"
        var roots =
            [
                settings.modelsDir.path, hub, home + "/.lmstudio/models", home + "/.cache/lm-studio/models", home + "/Models",
                home + "/models",
            ]
            + settings.modelsDirs.map(\.path)
        // Drives are not searched whole: only the top-level Models folder of a drive a store or a model is on.
        let drives = Set(
            (settings.stores.map(\.path) + Array(known)).filter { $0.hasPrefix("/Volumes/") }.map { "/Volumes/" + Stores.driveName($0) })
        for drive in drives.sorted() { roots += [drive + "/Models", drive + "/models"] }
        // Folders compare by file identity, not spelling: on a drive that ignores case ~/models is ~/Models, and a
        // symlink or a trailing slash names the same folder again.
        func identity(_ path: String) -> String? {
            var info = stat()
            return stat(path, &info) == 0 ? "\(info.st_dev):\(info.st_ino)" : nil
        }
        let knownIDs = Set(known.compactMap(identity))
        var unreadable: [String] = []
        var candidates: [URL] = []
        var visited = 0
        var walked = Set<String>()
        var partial: String?
        func walk(_ url: URL, depth: Int) {
            guard visited < 5_000, candidates.count < 200 else {
                partial = "the search stopped at its limit of 5,000 folders or 200 models"
                return
            }
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: url.path) else {
                if depth == 0 { partial = url.path + " could not be read" }
                return
            }
            visited += 1
            if isCheckpoint(names) {
                candidates.append(url)
                return
            }
            for name in names.sorted() where !name.hasPrefix(".") {
                let child = url.appendingPathComponent(name)
                if name.lowercased().hasSuffix(".gguf") {
                    if isLaterPart(name) { continue }
                    candidates.append(child)
                } else if depth < 5, (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                    walk(child, depth: depth + 1)
                }
            }
        }
        var skipped: [String: Stores.Availability] = [:]
        for root in roots where skipped[Stores.driveName(root)] == nil {
            let url = URL(fileURLWithPath: root, isDirectory: true)
            if root.hasPrefix("/Volumes/"), case let state = Stores.check(root, within: 20),
                [.slow, .noPermission, .notConnected].contains(state)
            {
                skipped[Stores.driveName(root)] = state
                if state != .notConnected { unreadable.append(Stores.driveName(root)) }
                continue
            }
            guard let id = identity(root), walked.insert(id).inserted else { continue }
            walk(url, depth: 0)
        }
        var seen = Set<String>()
        var models: [[String: Any]] = []
        for url in candidates {
            guard !known.contains(url.path), let id = identity(url.path), !knownIDs.contains(id), seen.insert(id).inserted else { continue }
            guard let report = try? ModelCompatibility.check(local: url), let family = report.family, report.verdict != .refused else {
                continue
            }
            var model: [String: Any] = [
                "path": url.path, "name": displayName(url), "family": family.displayName, "bytes": report.totalBytes,
                "verdict": report.verdict.rawValue, "reason": report.reason, "drive": Stores.driveName(url.path),
            ]
            // What its packs and copies are named by (`Storage`): a Qwen folder's fingerprint, a GGUF's key.
            if isGGUF(url.path) {
                if let parts = try? GGUFSource.parts(url) { model["key"] = FamilySupport.key(folder: parts.folder, files: parts.files) }
            } else if family == .qwen, let fingerprint = try? ModelSource.fingerprint(directory: url) {
                model["fingerprint"] = fingerprint
            }
            models.append(model)
        }
        return (models, unreadable, skipped, partial)
    }

    /// A folder's names make a checkpoint: a config.json and safetensors.
    static func isCheckpoint(_ names: [String]) -> Bool { names.contains("config.json") && names.contains { $0.hasSuffix(".safetensors") } }
    /// A split GGUF's part after the first, which is found (and added) by its first part alone.
    static func isLaterPart(_ name: String) -> Bool { name.firstMatch(of: /-(\d{5})-of-\d{5}\.gguf$/).map { $0.1 != "00001" } ?? false }

    /// What to call a found model: `owner/name` in Hugging Face's cache, else its folder's or file's name.
    static func displayName(_ url: URL) -> String {
        let parts = url.pathComponents
        if let index = parts.firstIndex(where: { $0.hasPrefix("models--") }) {
            return parts[index].dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
        }
        let name = url.lastPathComponent
        return name.lowercased().hasSuffix(".gguf")
            ? name.replacingOccurrences(of: "(-\\d{5}-of-\\d{5})?\\.gguf$", with: "", options: [.regularExpression, .caseInsensitive])
            : name
    }

    /// A .gguf file, or a folder of them without a config.json.
    static func isGGUF(_ path: String) -> Bool {
        if path.lowercased().hasSuffix(".gguf") { return FileManager.default.fileExists(atPath: path) }
        guard !FileManager.default.fileExists(atPath: path + "/config.json") else { return false }
        return ((try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []).contains { $0.lowercased().hasSuffix(".gguf") }
    }

    /// The sampling a model's files recommend, read as its server reads them: a GGUF's metadata, else a checkpoint
    /// folder's generation_config.json (or the GGUF its experts come from).
    static func recommendedSampling(_ path: String) -> SamplingDefaults {
        if isGGUF(path), let gguf = try? GGUFSource(URL(fileURLWithPath: path)) {
            return SamplingDefaults.recommended(gguf.metadata, splitModes: ModelFamily.detect(gguf: gguf.metadata) == .qwen)
        }
        let folder = URL(fileURLWithPath: path, isDirectory: true)
        let experts = (try? Data(contentsOf: folder.appendingPathComponent(GGUFExperts.file))).flatMap { try? GGUFExperts(manifest: $0) }
        return SamplingDefaults.recommended(folder: folder, gguf: experts?.files.first?.path)
    }

    static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }

    /// Bytes per second writing 32 MB into `folder` past the file cache; nil when that cannot be tried.
    static func writeRate(_ folder: URL) -> Double? {
        let url = folder.appendingPathComponent(".origami-write-probe")
        guard FileManager.default.createFile(atPath: url.path, contents: nil), let handle = FileHandle(forWritingAtPath: url.path)
        else { return nil }
        defer {
            try? handle.close()
            try? FileManager.default.removeItem(at: url)
        }
        _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
        let block = Data(count: 4 << 20)
        let started = DispatchTime.now().uptimeNanoseconds
        do {
            for _ in 0..<8 { try handle.write(contentsOf: block) }
            try handle.synchronize()
        } catch { return nil }
        return Double(32 << 20) / max(1e-6, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9)
    }

    /// One file: kept when `present` trusts the copy already there (`DownloadResume`), else resume `<name>.part` with a
    /// Range request, then check size and digest and rename.
    static func fetch(
        _ request: URLRequest, to url: URL, size: Int64, sha256: String?, present: DownloadResume, progress: DownloadProgress
    ) async throws {
        if present == .keep || (present == .hash && (try? digest(url)) == sha256) {
            progress.add(size, onDisk: true)
            return
        }
        let part = url.appendingPathExtension("part")
        if !FileManager.default.fileExists(atPath: part.path) { FileManager.default.createFile(atPath: part.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: part)
        defer { try? handle.close() }
        var offset = Int64(try handle.seekToEnd())
        if size > 0 && offset > size {
            try handle.truncate(atOffset: 0)
            offset = 0
        }
        progress.add(offset, onDisk: true)
        var attempts = 0
        while size == 0 || offset < size {
            // A failed attempt may have written part of its range; continue from the actual end.
            let actual = Int64(try handle.seekToEnd())
            offset = actual  // progress already counted those bytes as they arrived
            var ranged = request
            if offset > 0 { ranged.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
            do {
                let result = try await ChunkedFetch(handle: handle, progress: progress).run(ranged)
                // Refusals do not change with retries.
                if [401, 403].contains(result.status) {
                    throw CLIError(
                        "Hugging Face refused \(url.lastPathComponent) (HTTP \(result.status)): set HF_TOKEN to a token that can read this repo, "
                            + "and accept the model's licence on its huggingface.co page")
                }
                if result.status == 404 {
                    throw CLIError("Hugging Face has no \(url.lastPathComponent) in this repo (HTTP 404); check the repo name")
                }
                guard result.status == 200 || result.status == 206 else {
                    throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(result.status)"])
                }
                if result.status == 200 && offset > 0 {
                    // The server ignored the range: what was written restarts the file.
                    progress.add(-offset, onDisk: true)
                    offset = 0
                }
                offset += result.bytes
                if size == 0 { break }
            } catch let refusal as CLIError { throw refusal } catch {
                attempts += 1
                guard attempts <= 8 else {
                    throw CLIError(
                        "Download of \(url.lastPathComponent) keeps failing (\(error.localizedDescription)); check the connection and run the same command again to resume"
                    )
                }
                try await Task.sleep(for: .seconds(min(30, 1 << attempts)))
            }
        }
        try handle.synchronize()
        if let sha256 {
            guard try digest(part) == sha256 else {
                try? FileManager.default.removeItem(at: part)
                throw CLIError("\(url.lastPathComponent) arrived damaged (checksum differs); run the same command again")
            }
        }
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: part, to: url)
    }
    static func digest(_ url: URL) throws -> String {
        let reader = try FileHandle(forReadingFrom: url)
        defer { try? reader.close() }
        var hash = SHA256()
        while let chunk = try reader.read(upToCount: 8 << 20), !chunk.isEmpty { hash.update(data: chunk) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// One progress line, rewritten in place on a terminal.
final class DownloadProgress: @unchecked Sendable {
    private let lock = NSLock(), total: Int64, started = Date()
    private var done: Int64 = 0, onDisk: Int64 = 0, shown = Date.distantPast
    init(total: Int64) { self.total = total }
    /// `onDisk`: bytes an earlier run left (a kept file, the part a resume continues): counted as done, not in the rate.
    func add(_ bytes: Int64, onDisk kept: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        done += bytes
        if kept { onDisk += bytes }
        guard Date().timeIntervalSince(shown) >= 1 else { return }
        shown = Date()
        let rate = Double(done - onDisk) / max(1, Date().timeIntervalSince(started))
        Terminal.progress(String(format: "  %.1f / %.1f GB  (%.0f MB/s)", Double(done) / 1e9, Double(total) / 1e9, rate / 1e6))
    }
    func finish() {
        Terminal.progress(nil)
        print("  done")
    }
}

/// Streams one HTTP response into a file in the chunks URLSession delivers.
final class ChunkedFetch: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let handle: FileHandle, progress: DownloadProgress
    private var status = 0, written: Int64 = 0, failure: Error?, startOffset: UInt64 = 0
    private var continuation: CheckedContinuation<(status: Int, bytes: Int64), Error>?
    init(handle: FileHandle, progress: DownloadProgress) {
        self.handle = handle
        self.progress = progress
    }

    func run(_ request: URLRequest) async throws -> (status: Int, bytes: Int64) {
        startOffset = try handle.offset()
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: queue)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            session.dataTask(with: request).resume()
        }
    }
    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        status = (response as? HTTPURLResponse)?.statusCode ?? 0
        // A full response to a range request replaces the partial file.
        if status == 200 && startOffset > 0 { try? handle.truncate(atOffset: 0) }
        completionHandler(status == 200 || status == 206 ? .allow : .cancel)
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil else { return }
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
            progress.add(Int64(data.count))
        } catch {
            failure = error
            dataTask.cancel()
        }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let failure {
            continuation?.resume(throwing: failure)
        }
        // Status 0: no response at all (a transport error), which the caller retries and names.
        else if let error, [0, 200, 206].contains(status) {
            continuation?.resume(throwing: error)
        } else {
            continuation?.resume(returning: (status, written))
        }
        continuation = nil
    }
}

/// Named models (`origami model`), kept in `<data_dir>/models.json`. An entry is a folder with, optionally, its own
/// expert width (a 3-bit preparation of a checkpoint is a model of its own) and a manifest: what the folder held when
/// it was added (`origami inspect --json`), its fingerprint and its chat template's sha256.
enum ModelRegistry {
    static func file(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("models.json") }
    /// Throws for a file that exists but cannot be read, so `model add` never overwrites it.
    static func entries(_ settings: Settings) throws -> [[String: Any]] {
        try Settings.readConfig(file(settings))["models"] as? [[String: Any]] ?? []
    }
    static func entry(_ name: String?, settings: Settings) -> [String: Any]? {
        name.flatMap { name in (try? entries(settings))?.first { $0["name"] as? String == name } }
    }
    static func save(_ entries: [[String: Any]], settings: Settings) throws {
        try FileManager.default.createDirectory(at: settings.dataDir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["models": entries], options: [.prettyPrinted, .sortedKeys]).write(
            to: file(settings), options: .atomic)
    }
    /// A read-modify-write of the file under its lock (`StateFile.locked`): concurrent writers never lose an entry.
    @discardableResult
    static func update<T>(_ settings: Settings, _ body: (inout [[String: Any]]) throws -> T) throws -> T {
        try StateFile.locked(file(settings)) {
            var all = try entries(settings)
            let result = try body(&all)
            try save(all, settings: settings)
            return result
        }
    }
    /// The family names a request may use instead of a model's own (`qwen`, `glm`, `deepseek`), each to the registered
    /// model of that family it means now: the default model (`default`) if it is one, else the most recently added. A model
    /// registered under the alias's own name keeps it, so an exact name always wins.
    static func aliases(_ entries: [[String: Any]], default: String?) -> [String: String] {
        let families: [String: ModelFamily] = ["qwen": .qwen, "glm": .glm, "deepseek": .deepseek]
        let names = entries.compactMap { $0["name"] as? String }
        var aliases: [String: String] = [:]
        for (alias, family) in families where !names.contains(alias) {
            // Ties in `added` go to the later entry: `put` appends, so the file's order is the order of adding.
            let mine = entries.enumerated().filter {
                $0.element["name"] is String
                    && ((($0.element["family"] as? String).flatMap(ModelFamily.init(rawValue:)) ?? .qwen) == family)
            }
            let pick =
                mine.first { $0.element["name"] as? String == `default` }
                ?? mine.max { ($0.element["added"] as? String ?? "", $0.offset) < ($1.element["added"] as? String ?? "", $1.offset) }
            aliases[alias] = pick?.element["name"] as? String
        }
        return aliases
    }
    /// Adds `entry`, replacing one of the same name; true when it replaced one.
    static func put(_ entry: [String: Any], settings: Settings) throws -> Bool {
        let name = entry["name"] as? String
        let old = try update(settings) { entries in
            let old = entries.first { $0["name"] as? String == name }
            entries.removeAll { $0["name"] as? String == name }
            entries.append(entry)
            return old
        }
        if let old { retire(old, replacedBy: name, settings: settings) }
        return old != nil
    }

    /// `<data_dir>/former-models.json`: by identity (a Qwen fingerprint12, a GGUF key), the name a model had when it was
    /// removed or replaced or its identity changed, so the dashboard names the files left keyed by it.
    static func formerFile(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("former-models.json") }
    static func former(_ settings: Settings) -> [String: [String: String]] {
        ((try? Settings.readConfig(formerFile(settings))) ?? [:]).compactMapValues { $0 as? [String: String] }
    }
    /// Keeps `entry`'s name for its identities (`ids`, else what it has now: its manifest's or its folder's fingerprint12,
    /// its GGUF key), with today's date and the model now in its place, if any. Called as a model leaves the list or its
    /// identity changes: `model remove`, the dashboard's Unlist and Delete, an add under its name, `origami refresh`, and
    /// a replace. A record whose identity is registered again is not shown, so calling it early is harmless.
    static func retire(_ entry: [String: Any], replacedBy: String? = nil, ids: [String]? = nil, settings: Settings) {
        guard let name = entry["name"] as? String else { return }
        let ids = ids ?? identities(entry)
        guard !ids.isEmpty else { return }
        var record = ["name": name, "at": String(ISO8601DateFormatter().string(from: Date()).prefix(10))]
        record["replaced_by"] = replacedBy
        let file = formerFile(settings)
        try? StateFile.locked(file) {
            var all = try Settings.readConfig(file)
            for id in ids { all[id] = record }
            try JSONSerialization.data(withJSONObject: all, options: [.prettyPrinted, .sortedKeys]).write(to: file, options: .atomic)
        }
    }
    /// The identities the files Origami keeps for `entry` are keyed by: its GGUF key, or its fingerprint12.
    private static func identities(_ entry: [String: Any]) -> [String] {
        guard let path = entry["path"] as? String else { return [] }
        if entry["family"] != nil {
            guard let parts = try? GGUFSource.parts(URL(fileURLWithPath: path)) else { return [] }
            return [FamilySupport.key(folder: parts.folder, files: parts.files, experts: ExpertVariant.expertFiles(entry))]
        }
        let fingerprint =
            (entry["manifest"] as? [String: Any])?["fingerprint"] as? String
            ?? (try? ModelSource.fingerprint(directory: URL(fileURLWithPath: path)))
        return fingerprint.map { [String($0.prefix(12))] } ?? []
    }
    /// How the page names a former model: "<name> (removed 2026-11-26)", "(replaced by <name> …)" or "(refreshed …)".
    static func formerName(_ record: [String: String]) -> String {
        let name = record["name"] ?? "?"
        let by = record["replaced_by"]
        return name + " (" + (by == nil ? "removed" : by == name ? "refreshed" : "replaced by " + by!) + " " + (record["at"] ?? "") + ")"
    }
}
